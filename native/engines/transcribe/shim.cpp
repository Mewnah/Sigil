// SPDX-License-Identifier: AGPL-3.0-only
//
// shim.cpp - sigil_transcribe.dll: transcribe.cpp (pinned in UPSTREAM.json) as a sigil_engine ABI v1
// engine. The 16 exports come from Sigil's shared implementation (native/common/
// sigil_engine_impl.cpp, compiled in by sigil_add_engine); this file implements sigil::Engine over
// the transcribe.cpp C API (include/transcribe.h). Owned by P1-G2; decision record
// docs/adr/0003b-transcribe-cpp.md. transcribe.cpp and its vendored ggml are MIT.
//
// Build shape (CMakeLists.txt): one DLL per variant with transcribe.cpp and ggml linked statically
// (GGML_BACKEND_DL=OFF, GGML_NATIVE=OFF, /MD), so the two in-process ggml copies (whisper-rs's inside
// Sigil.exe and this one) never share a DLL name or a symbol.
//
// JSON contract (all strings UTF-8; invalid UTF-8 in model output is replaced by U+FFFD):
//   create  {} -> an instance without a model that answers {"op":"info"} / {"op":"devices"} (the
//            build, the backends and devices this PC exposes) and refuses to transcribe;
//           {"model": "<abs path .gguf|.bin>", "backend": "auto|cpu|cpu_accel|cuda|vulkan",
//            "device": <index among that backend's devices> | "<pci id or name>",
//            "threads": 0..64, "kv_type": "auto|f16|f32", "n_ctx": >= 0}
//     backend defaults to the build's own backend (cuda13 -> cuda, vulkan -> vulkan, cpu* -> cpu);
//     an explicit GPU backend that is unavailable fails with SIGIL_E_BACKEND_UNAVAILABLE (the host
//     retries on another variant or on "cpu" and reports why), never a silent CPU fallback.
//   process (16 kHz audio; channels are averaged to mono) with options
//     {"language": "ja"|"auto"|null, "task": "transcribe|translate", "target_language",
//      "timestamps": "none|auto|segment|word|token" (clamped to the model's maximum),
//      "words": bool, "pnc"/"itn"/"diarize": "default|on|off", "spec_k_drafts": -1..16,
//      "keep_special_tags": bool, "prompt": "<whisper initial prompt>",
//      "temperature_fallback": bool, "no_speech_threshold": 0..1}
//     -> {"text","language","language_source","timestamps","segments":[{"start_ms","end_ms","text",
//         "speaker"?}],"words"?,"truncated","audio_ms","wall_ms","rtf","timings":{...},"backend"}
//   query {"op": "info"|"devices"|"memory"|"languages"|"limits"|"warmup"}
//   stream (models whose capabilities say streaming) with the process options plus
//     {"commit": "auto|on_finalize|stable_prefix", "agreement": 0..16}; push returns
//     {"partial","committed","tentative","text","revision","input_ms",...} when the hypothesis
//     changed (nothing otherwise); finish returns {"text","committed","truncated","language",...}
//     and the stream accepts the next utterance.
// One compute per model (transcribe.h 0.x: "at most one transcribe_run / transcribe_run_batch /
// active stream may be in flight across ALL sessions of a given model"): a stream is ACTIVE from its
// first push until finish (or a failed / cancelled push, or close). While one stream of the model is
// ACTIVE, process(), the warm-up and the first push of any other stream fail with
// SIGIL_E_INVALID_ARGUMENT ("another stream is active on this model"); load a second model instead.
// Cancellation: sigil_engine_cancel is polled by transcribe.cpp between chunks / decode steps
// (Qwen3-ASR 0.2.4 only checks before its single-shot decode starts).
// Fatal errors: ggml ends the process on errors it cannot return (GGML_ABORT / GGML_ASSERT, and
// ggml-cuda's CUDA_CHECK on a failed pool growth or a lost device). The shim installs ggml's abort
// callback, so the message reaches the host log (synchronously) before abort(); the host keeps VRAM
// headroom so these paths are not reached in normal use (docs/adr/0003b-transcribe-cpp.md).

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ggml.h"  // ggml_set_abort_callback only (this DLL's own static ggml)
#include "sigil_engine_impl.hpp"
#include "transcribe.h"
#include "transcribe/whisper.h"

#ifndef SIGIL_ENGINE_VERSION
#error "SIGIL_ENGINE_VERSION is set by sigil_add_engine()"
#endif
#ifndef SIGIL_TRANSCRIBE_DEFAULT_BACKEND
#define SIGIL_TRANSCRIBE_DEFAULT_BACKEND "cpu"
#endif
#ifndef SIGIL_TRANSCRIBE_GPU
#define SIGIL_TRANSCRIBE_GPU 0
#endif
#ifndef SIGIL_TRANSCRIBE_UPSTREAM_INFO
#define SIGIL_TRANSCRIBE_UPSTREAM_INFO "transcribe.cpp " TRANSCRIBE_VERSION
#endif

using sigil::fail;
using sigil::json::Value;
using sigil::json::Writer;

namespace {

constexpr int kSampleRate = 16000;
constexpr int64_t kMaxThreads = 64;

/* ------------------------------------------------------------------------------------------ */
/* Logging: transcribe.cpp and ggml -> the host's log callback                                */
/* ------------------------------------------------------------------------------------------ */

std::once_flag g_log_install_once;

// Most recent WARN / ERROR line (any thread), appended to error messages.
std::mutex g_engine_log_mutex;
std::string g_engine_log_last;
uint64_t g_engine_log_seq = 0;

uint64_t engine_log_seq() {
    std::lock_guard<std::mutex> lock(g_engine_log_mutex);
    return g_engine_log_seq;
}

std::string engine_log_since(uint64_t seq) {
    std::lock_guard<std::mutex> lock(g_engine_log_mutex);
    return g_engine_log_seq != seq ? g_engine_log_last : std::string();
}

void log_trampoline(transcribe_log_level level, const char* msg, void* /*userdata*/) {
    if (msg == nullptr) return;
    int32_t sigil_level = SIGIL_LOG_TRACE;
    switch (level) {
    case TRANSCRIBE_LOG_LEVEL_ERROR: sigil_level = SIGIL_LOG_ERROR; break;
    case TRANSCRIBE_LOG_LEVEL_WARN: sigil_level = SIGIL_LOG_WARN; break;
    case TRANSCRIBE_LOG_LEVEL_INFO: sigil_level = SIGIL_LOG_INFO; break;
    case TRANSCRIBE_LOG_LEVEL_DEBUG: sigil_level = SIGIL_LOG_DEBUG; break;
    default: sigil_level = SIGIL_LOG_TRACE; break;  // NONE / CONT (progress fragments)
    }
    size_t len = std::strlen(msg);
    while (len > 0 && (msg[len - 1] == '\n' || msg[len - 1] == '\r')) --len;
    if (len == 0) return;
    try {
        std::string line(msg, len);
        if (sigil_level >= SIGIL_LOG_WARN) {
            std::lock_guard<std::mutex> lock(g_engine_log_mutex);
            g_engine_log_last = line;
            ++g_engine_log_seq;
        }
        sigil::log(sigil_level, line);
    } catch (...) {
        // Never let an allocation failure escape into ggml / transcribe.cpp.
    }
}

/** ggml's abort callback: GGML_ABORT / GGML_ASSERT (including ggml-cuda's CUDA_CHECK / CU_CHECK /
 *  CUBLAS_CHECK on a failed memory-pool growth or a lost device) call it right before abort(), which
 *  ends the host process. It cannot be recovered from (throwing through ggml's C frames is
 *  undefined), so it only records why: synchronously to the host log (Sigil's file log) and to
 *  stderr (headless probes). ggml-cuda already logged the CUDA error text through the trampoline. */
void ggml_abort_trampoline(const char* message) {
    const char* text = message != nullptr ? message : "(no message)";
    try {
        sigil::log(SIGIL_LOG_ERROR,
                   std::string("fatal ggml error, the process ends now (in-process engine; see docs/adr/0003b-transcribe-cpp.md): ") + text);
    } catch (...) {
        // Out of memory while reporting: stderr below still gets the message.
    }
    std::fprintf(stderr, "sigil_transcribe: fatal ggml error: %s\n", text);
    std::fflush(stderr);
}

/** Route every transcribe.cpp and ggml message through the trampoline, once per process, before
 *  any model exists (transcribe.h's supported usage), so the engine never writes to stdout/stderr;
 *  and install ggml's abort callback (process-wide for this DLL's static ggml only: whisper-rs's
 *  ggml inside Sigil.exe keeps its own). */
void ensure_log_installed() {
    std::call_once(g_log_install_once, [] {
        transcribe_log_set(&log_trampoline, nullptr);
        ggml_set_abort_callback(&ggml_abort_trampoline);
    });
}

/* ------------------------------------------------------------------------------------------ */
/* Errors                                                                                     */
/* ------------------------------------------------------------------------------------------ */

sigil_status map_status(transcribe_status s) {
    switch (s) {
    case TRANSCRIBE_OK: return SIGIL_OK;
    case TRANSCRIBE_ERR_INVALID_ARG:
    case TRANSCRIBE_ERR_BAD_STRUCT_SIZE:
    case TRANSCRIBE_ERR_SAMPLE_RATE:
    case TRANSCRIBE_ERR_UNSUPPORTED_LANGUAGE:
    case TRANSCRIBE_ERR_UNSUPPORTED_TASK:
    case TRANSCRIBE_ERR_UNSUPPORTED_TIMESTAMPS:
    case TRANSCRIBE_ERR_UNSUPPORTED_PNC:
    case TRANSCRIBE_ERR_UNSUPPORTED_ITN:
    case TRANSCRIBE_ERR_INPUT_TOO_LONG: return SIGIL_E_INVALID_ARGUMENT;
    case TRANSCRIBE_ERR_NOT_IMPLEMENTED: return SIGIL_E_UNSUPPORTED;
    case TRANSCRIBE_ERR_FILE_NOT_FOUND:
    case TRANSCRIBE_ERR_GGUF:
    case TRANSCRIBE_ERR_UNSUPPORTED_ARCH:
    case TRANSCRIBE_ERR_UNSUPPORTED_VARIANT: return SIGIL_E_LOAD_FAILED;
    case TRANSCRIBE_ERR_OOM: return SIGIL_E_OUT_OF_MEMORY;
    case TRANSCRIBE_ERR_BACKEND: return SIGIL_E_BACKEND_UNAVAILABLE;
    case TRANSCRIBE_ERR_ABORTED: return SIGIL_E_CANCELLED;
    default: return SIGIL_E_RUNTIME;
    }
}

/** Throw "<what>: <status text> (transcribe status N)[; engine log: <last warn/error>]". */
[[noreturn]] void fail_transcribe(const std::string& what, transcribe_status s, uint64_t log_seq) {
    std::string msg = what;
    msg += ": ";
    const char* text = transcribe_status_string(static_cast<int>(s));
    msg += text != nullptr ? text : "unknown status";
    msg += " (transcribe status ";
    msg += std::to_string(static_cast<int>(s));
    msg += ")";
    const std::string log = engine_log_since(log_seq);
    if (!log.empty()) {
        msg += "; engine log: ";
        msg += log;
    }
    fail(map_status(s), msg);
}

/* ------------------------------------------------------------------------------------------ */
/* Text                                                                                       */
/* ------------------------------------------------------------------------------------------ */

/** Valid UTF-8 copy of `s`: malformed sequences (byte-level BPE pieces can split a code point)
 *  become U+FFFD, so every host can parse the JSON strictly. */
std::string utf8_clean(std::string_view s) {
    std::string out;
    out.reserve(s.size());
    size_t i = 0;
    const size_t n = s.size();
    while (i < n) {
        const unsigned char c = static_cast<unsigned char>(s[i]);
        if (c < 0x80) {
            out.push_back(static_cast<char>(c));
            ++i;
            continue;
        }
        size_t len = 0;
        uint32_t cp = 0;
        uint32_t min_cp = 0;
        if (c >= 0xC2 && c <= 0xDF) { len = 2; cp = c & 0x1F; min_cp = 0x80; }
        else if (c >= 0xE0 && c <= 0xEF) { len = 3; cp = c & 0x0F; min_cp = 0x800; }
        else if (c >= 0xF0 && c <= 0xF4) { len = 4; cp = c & 0x07; min_cp = 0x10000; }
        bool ok = len != 0 && i + len <= n;
        for (size_t k = 1; ok && k < len; ++k) {
            const unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc & 0xC0) != 0x80) ok = false;
            else cp = (cp << 6) | (cc & 0x3F);
        }
        if (ok && (cp < min_cp || cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))) ok = false;
        if (ok) {
            out.append(s.data() + i, len);
            i += len;
        } else {
            out += "\xEF\xBF\xBD";
            ++i;
        }
    }
    return out;
}

std::string clean(const char* s) { return s != nullptr ? utf8_clean(s) : std::string(); }

void field_or_null(Writer& w, std::string_view key, const char* s) {
    w.key(key);
    if (s == nullptr) w.null();
    else w.value(utf8_clean(s));
}

/* ------------------------------------------------------------------------------------------ */
/* Options                                                                                    */
/* ------------------------------------------------------------------------------------------ */

int64_t int_or(const Value& o, std::string_view key, int64_t lo, int64_t hi, int64_t fallback) {
    const Value* v = o.find(key);
    if (v == nullptr || v->is_null()) return fallback;
    const double d = v->is_number() ? v->as_number() : std::nan("");
    if (!(d == std::floor(d)) || d < static_cast<double>(lo) || d > static_cast<double>(hi)) {
        fail(SIGIL_E_INVALID_ARGUMENT, "\"" + std::string(key) + "\" must be an integer in [" + std::to_string(lo) + ", " +
                                           std::to_string(hi) + "]");
    }
    return static_cast<int64_t>(d);
}

/** `"default" | "on" | "off"` (0 / 2 / 1 in transcribe.h's pnc / itn / diarize enums). */
int toggle_or(const Value& o, std::string_view key) {
    const std::string s = o.string_or(key, "default");
    if (s == "default") return 0;
    if (s == "off") return 1;
    if (s == "on") return 2;
    fail(SIGIL_E_INVALID_ARGUMENT, "\"" + std::string(key) + "\" must be \"default\", \"on\" or \"off\"");
}

const char* timestamp_kind_name(transcribe_timestamp_kind k) {
    switch (k) {
    case TRANSCRIBE_TIMESTAMPS_NONE: return "none";
    case TRANSCRIBE_TIMESTAMPS_AUTO: return "auto";
    case TRANSCRIBE_TIMESTAMPS_SEGMENT: return "segment";
    case TRANSCRIBE_TIMESTAMPS_WORD: return "word";
    case TRANSCRIBE_TIMESTAMPS_TOKEN: return "token";
    default: return "unknown";
    }
}

/** Per-run options shared by process() and streams (strings owned here). */
struct RunOptions {
    std::string language;  // empty = detect
    std::string target_language;
    std::string prompt;
    bool translate = false;
    transcribe_timestamp_kind timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
    bool words = false;
    int pnc = 0;
    int itn = 0;
    int diarize = 0;
    int64_t spec_k_drafts = -1;
    bool keep_special_tags = false;
    bool temperature_fallback = true;
    bool has_no_speech = false;
    double no_speech_threshold = 0.6;
    transcribe_stream_commit_policy commit = TRANSCRIBE_STREAM_COMMIT_AUTO;
    int64_t agreement = 0;

    static RunOptions parse(const Value& o) {
        RunOptions r;
        const std::string lang = o.string_or("language", "");
        if (lang != "auto") r.language = lang;
        const std::string task = o.string_or("task", "transcribe");
        if (task == "translate") r.translate = true;
        else if (task != "transcribe") fail(SIGIL_E_INVALID_ARGUMENT, "\"task\" must be \"transcribe\" or \"translate\"");
        r.target_language = o.string_or("target_language", "");
        const std::string ts = o.string_or("timestamps", "segment");
        if (ts == "none") r.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
        else if (ts == "auto") r.timestamps = TRANSCRIBE_TIMESTAMPS_AUTO;
        else if (ts == "segment") r.timestamps = TRANSCRIBE_TIMESTAMPS_SEGMENT;
        else if (ts == "word") r.timestamps = TRANSCRIBE_TIMESTAMPS_WORD;
        else if (ts == "token") r.timestamps = TRANSCRIBE_TIMESTAMPS_TOKEN;
        else fail(SIGIL_E_INVALID_ARGUMENT, "\"timestamps\" must be none, auto, segment, word or token");
        r.words = o.bool_or("words", false);
        r.pnc = toggle_or(o, "pnc");
        r.itn = toggle_or(o, "itn");
        r.diarize = toggle_or(o, "diarize");
        r.spec_k_drafts = int_or(o, "spec_k_drafts", -1, 16, -1);
        r.keep_special_tags = o.bool_or("keep_special_tags", false);
        r.prompt = o.string_or("prompt", "");
        r.temperature_fallback = o.bool_or("temperature_fallback", true);
        const Value* ns = o.find("no_speech_threshold");
        if (ns != nullptr && !ns->is_null()) {
            if (!ns->is_number() || !(ns->as_number() >= 0.0 && ns->as_number() <= 1.0)) {
                fail(SIGIL_E_INVALID_ARGUMENT, "\"no_speech_threshold\" must be a number in [0, 1]");
            }
            r.has_no_speech = true;
            r.no_speech_threshold = ns->as_number();
        }
        const std::string commit = o.string_or("commit", "auto");
        if (commit == "auto") r.commit = TRANSCRIBE_STREAM_COMMIT_AUTO;
        else if (commit == "on_finalize") r.commit = TRANSCRIBE_STREAM_COMMIT_ON_FINALIZE;
        else if (commit == "stable_prefix") r.commit = TRANSCRIBE_STREAM_COMMIT_STABLE_PREFIX;
        else fail(SIGIL_E_INVALID_ARGUMENT, "\"commit\" must be auto, on_finalize or stable_prefix");
        r.agreement = int_or(o, "agreement", 0, 16, 0);
        return r;
    }
};

/* ------------------------------------------------------------------------------------------ */
/* Shared model state                                                                         */
/* ------------------------------------------------------------------------------------------ */

/** What the abort callback reads: the owning instance's cancel token, cleared when the instance
 *  is destroyed (a stream that outlives its instance, a host bug, then aborts instead of reading
 *  freed memory). */
struct CancelLink {
    std::atomic<const sigil::CancelToken*> token{nullptr};
};

bool abort_callback(void* user_data) {
    const sigil::CancelToken* token = static_cast<CancelLink*>(user_data)->token.load(std::memory_order_acquire);
    return token == nullptr || token->requested();
}

/** The loaded model, shared by the instance's session and every stream session. */
struct SharedModel {
    transcribe_model* model = nullptr;
    // transcribe.cpp 0.x allows one compute in flight per model across all of its sessions, and an
    // ACTIVE stream counts as in flight from begin to finalize (not only during a feed call). The
    // host serializes the individual calls; `compute` makes each call exclusive and
    // `active_streams` (guarded by `compute`) refuses an overlapping run or stream begin.
    std::mutex compute;
    int active_streams = 0;
    std::shared_ptr<CancelLink> cancel = std::make_shared<CancelLink>();
    transcribe_session_params session_params{};
    transcribe_capabilities caps{};
    std::string requested_backend;

    ~SharedModel() {
        if (model != nullptr) transcribe_model_free(model);
    }
};

/** 16 kHz input -> mono float, non-finite samples zeroed. */
void prepare_audio(const sigil::Audio& audio, std::vector<float>& mono) {
    if (audio.frames == 0) fail(SIGIL_E_INVALID_ARGUMENT, "empty audio");
    if (audio.sample_rate != static_cast<uint32_t>(kSampleRate)) {
        fail(SIGIL_E_INVALID_ARGUMENT,
             "expected 16000 Hz audio, got " + std::to_string(audio.sample_rate) + " Hz (resample on the host)");
    }
    if (audio.frames > static_cast<uint64_t>(INT_MAX)) fail(SIGIL_E_INVALID_ARGUMENT, "audio too long");
    const size_t frames = static_cast<size_t>(audio.frames);
    const size_t ch = audio.channels;
    mono.resize(frames);
    const float* in = audio.samples;
    const float scale = 1.0f / static_cast<float>(ch);
    for (size_t i = 0; i < frames; ++i) {
        float acc = 0.0f;
        for (size_t c = 0; c < ch; ++c) acc += in[i * ch + c];
        const float v = ch == 1 ? acc : acc * scale;
        mono[i] = std::isfinite(v) ? v : 0.0f;
    }
}

void write_device(Writer& w, transcribe_device_t dev) {
    transcribe_device_info info;
    transcribe_device_info_init(&info);
    if (dev == nullptr || transcribe_device_get_info(dev, &info) != TRANSCRIBE_OK) {
        w.null();
        return;
    }
    const char* type = "unknown";
    switch (info.device_type) {
    case TRANSCRIBE_DEVICE_TYPE_CPU: type = "cpu"; break;
    case TRANSCRIBE_DEVICE_TYPE_GPU: type = "gpu"; break;
    case TRANSCRIBE_DEVICE_TYPE_IGPU: type = "igpu"; break;
    case TRANSCRIBE_DEVICE_TYPE_ACCEL: type = "accel"; break;
    }
    w.begin_object();
    field_or_null(w, "name", info.name);
    field_or_null(w, "description", info.description);
    field_or_null(w, "kind", info.kind);
    field_or_null(w, "device_id", info.device_id);
    w.field("type", type).field("memory_total", info.memory_total).field("memory_free", info.memory_free);
    w.end_object();
}

void write_languages(Writer& w, const transcribe_capabilities& caps) {
    w.begin_array();
    for (int i = 0; i < caps.n_languages && caps.languages != nullptr; ++i) {
        if (caps.languages[i] != nullptr) w.value(clean(caps.languages[i]));
    }
    w.end_array();
}

void write_capabilities(Writer& w, const transcribe_model* model, const transcribe_capabilities& caps) {
    w.begin_object();
    w.field("sample_rate", static_cast<int64_t>(caps.native_sample_rate));
    w.key("languages");
    write_languages(w, caps);
    w.field("language_detect", caps.supports_language_detect)
        .field("translate", caps.supports_translate)
        .field("streaming", caps.supports_streaming)
        .field("spec_decode", caps.supports_spec_decode)
        .field("max_timestamps", timestamp_kind_name(caps.max_timestamp_kind))
        .field("max_audio_ms", static_cast<int64_t>(caps.max_audio_ms));
    w.key("features").begin_object();
    w.field("initial_prompt", transcribe_model_supports(model, TRANSCRIBE_FEATURE_INITIAL_PROMPT))
        .field("temperature_fallback", transcribe_model_supports(model, TRANSCRIBE_FEATURE_TEMPERATURE_FALLBACK))
        .field("long_form", transcribe_model_supports(model, TRANSCRIBE_FEATURE_LONG_FORM))
        .field("cancellation", transcribe_model_supports(model, TRANSCRIBE_FEATURE_CANCELLATION))
        .field("pnc", transcribe_model_supports(model, TRANSCRIBE_FEATURE_PNC))
        .field("itn", transcribe_model_supports(model, TRANSCRIBE_FEATURE_ITN))
        .field("diarization", transcribe_model_supports(model, TRANSCRIBE_FEATURE_DIARIZATION));
    w.end_object();
    w.end_object();
}

void write_timings(Writer& w, const transcribe_session* session, double wall_ms) {
    transcribe_timings t;
    transcribe_timings_init(&t);
    transcribe_get_timings(session, &t);
    w.begin_object()
        .field("load_ms", static_cast<double>(t.load_ms))
        .field("mel_ms", static_cast<double>(t.mel_ms))
        .field("encode_ms", static_cast<double>(t.encode_ms))
        .field("decode_ms", static_cast<double>(t.decode_ms))
        .field("wall_ms", wall_ms)
        .end_object();
}

void write_limits(Writer& w, const transcribe_session_limits& lim) {
    w.begin_object()
        .field("n_ctx", static_cast<int64_t>(lim.effective_n_ctx))
        .field("max_audio_ms", static_cast<int64_t>(lim.effective_max_audio_ms))
        .field("max_kv_bytes", static_cast<int64_t>(lim.max_kv_bytes))
        .end_object();
}

/** Result of the most recent offline run on `session`. */
std::string result_json(const transcribe_session* session, const transcribe_model* model, const RunOptions& opts,
                        size_t samples, double wall_ms, bool truncated) {
    Writer w;
    w.begin_object();
    w.field("text", clean(transcribe_full_text(session)));
    if (!opts.language.empty()) {
        w.field("language", opts.language).field("language_source", "hint");
    } else {
        const char* det = transcribe_detected_language(session);
        const bool detected = det != nullptr && *det != '\0';
        w.field("language", detected ? clean(det) : std::string()).field("language_source", detected ? "detected" : "none");
    }
    w.field("timestamps", timestamp_kind_name(transcribe_returned_timestamp_kind(session)));
    w.key("segments").begin_array();
    const int n_seg = transcribe_n_segments(session);
    for (int i = 0; i < n_seg; ++i) {
        transcribe_segment seg;
        transcribe_segment_init(&seg);
        if (transcribe_get_segment(session, i, &seg) != TRANSCRIBE_OK || seg.text == nullptr) continue;
        w.begin_object()
            .field("start_ms", static_cast<int64_t>(seg.t0_ms))
            .field("end_ms", static_cast<int64_t>(seg.t1_ms))
            .field("text", clean(seg.text));
        if (seg.speaker_id > 0) w.field("speaker", static_cast<int64_t>(seg.speaker_id));
        w.end_object();
    }
    w.end_array();
    if (opts.words) {
        w.key("words").begin_array();
        const int n_words = transcribe_n_words(session);
        for (int i = 0; i < n_words; ++i) {
            transcribe_word word;
            transcribe_word_init(&word);
            if (transcribe_get_word(session, i, &word) != TRANSCRIBE_OK || word.text == nullptr) continue;
            w.begin_object()
                .field("start_ms", static_cast<int64_t>(word.t0_ms))
                .field("end_ms", static_cast<int64_t>(word.t1_ms))
                .field("segment", static_cast<int64_t>(word.seg_index))
                .field("text", clean(word.text))
                .end_object();
        }
        w.end_array();
    }
    const double audio_ms = static_cast<double>(samples) * 1000.0 / kSampleRate;
    w.field("truncated", truncated).field("audio_ms", audio_ms).field("wall_ms", wall_ms);
    w.field("rtf", audio_ms > 0.0 ? wall_ms / audio_ms : 0.0);
    w.key("timings");
    write_timings(w, session, wall_ms);
    w.field("backend", clean(transcribe_model_backend(model)));
    w.end_object();
    return w.take();
}

/** Fill transcribe run params from `opts` (pointers into `opts` / `whisper` stay valid for the call). */
void fill_run_params(const transcribe_model* model, const transcribe_capabilities& caps, const RunOptions& opts,
                     transcribe_run_params& p, transcribe_whisper_run_ext& whisper, bool for_stream) {
    transcribe_run_params_init(&p);
    p.task = opts.translate ? TRANSCRIBE_TASK_TRANSLATE : TRANSCRIBE_TASK_TRANSCRIBE;
    // A request finer than the model can produce is clamped (transcribe.cpp would reject it).
    transcribe_timestamp_kind ts = opts.timestamps;
    if (ts != TRANSCRIBE_TIMESTAMPS_AUTO && ts > caps.max_timestamp_kind) ts = caps.max_timestamp_kind;
    p.timestamps = ts;
    p.pnc = static_cast<transcribe_pnc_mode>(opts.pnc);
    p.itn = static_cast<transcribe_itn_mode>(opts.itn);
    p.diarize = static_cast<transcribe_diarize_mode>(opts.diarize);
    p.language = opts.language.empty() ? nullptr : opts.language.c_str();
    p.target_language = opts.target_language.empty() ? nullptr : opts.target_language.c_str();
    p.keep_special_tags = opts.keep_special_tags;
    p.spec_k_drafts = static_cast<int32_t>(opts.spec_k_drafts);
    p.family = nullptr;
    const bool wants_whisper_ext = !opts.prompt.empty() || !opts.temperature_fallback || opts.has_no_speech;
    if (!for_stream && wants_whisper_ext &&
        transcribe_model_accepts_ext_kind(model, TRANSCRIBE_EXT_SLOT_RUN, TRANSCRIBE_EXT_KIND_WHISPER_RUN)) {
        transcribe_whisper_run_ext_init(&whisper);
        if (!opts.prompt.empty()) whisper.initial_prompt = opts.prompt.c_str();
        if (!opts.temperature_fallback) whisper.temperature_inc = 0.0f;  // no re-decodes
        if (opts.has_no_speech) whisper.no_speech_thold = static_cast<float>(opts.no_speech_threshold);
        p.family = &whisper.ext;
    }
}

constexpr const char* kStreamActiveMessage =
    "another stream is active on this model (between its first push and finish); transcribe.cpp runs one "
    "compute per model at a time: finish that stream first, or load a second model";

/** Offline run on `session` (compute serialized per model; refused while a stream is ACTIVE). */
std::string run_offline(SharedModel& m, transcribe_session* session, const float* pcm, size_t n, const RunOptions& opts) {
    transcribe_run_params p;
    transcribe_whisper_run_ext whisper;
    fill_run_params(m.model, m.caps, opts, p, whisper, false);
    const uint64_t log_seq = engine_log_seq();
    std::lock_guard<std::mutex> lock(m.compute);
    if (m.active_streams > 0) fail(SIGIL_E_INVALID_ARGUMENT, kStreamActiveMessage);
    const auto t0 = std::chrono::steady_clock::now();
    transcribe_status st = transcribe_run(session, pcm, static_cast<int>(n), &p);
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    bool truncated = false;
    if (st == TRANSCRIBE_ERR_OUTPUT_TRUNCATED) {
        truncated = true;  // the partial transcript is kept and flagged, never passed off as complete
        st = TRANSCRIBE_OK;
    }
    if (st != TRANSCRIBE_OK) fail_transcribe("process", st, log_seq);
    return result_json(session, m.model, opts, n, wall_ms, truncated);
}

bool backend_from_name(const std::string& name, transcribe_backend_request& out, const char*& kind) {
    if (name == "auto") { out = TRANSCRIBE_BACKEND_AUTO; kind = nullptr; }
    else if (name == "cpu") { out = TRANSCRIBE_BACKEND_CPU; kind = "cpu"; }
    else if (name == "cpu_accel") { out = TRANSCRIBE_BACKEND_CPU_ACCEL; kind = "cpu"; }
    else if (name == "cuda") { out = TRANSCRIBE_BACKEND_CUDA; kind = "cuda"; }
    else if (name == "vulkan") { out = TRANSCRIBE_BACKEND_VULKAN; kind = "vulkan"; }
    else return false;
    return true;
}

std::string registered_devices_summary() {
    std::string s;
    const int n = transcribe_device_count();
    for (int i = 0; i < n; ++i) {
        transcribe_device_info info;
        transcribe_device_info_init(&info);
        if (transcribe_device_get_info(transcribe_device_get(i), &info) != TRANSCRIBE_OK) continue;
        if (!s.empty()) s += ", ";
        s += info.kind != nullptr ? info.kind : "?";
        s += ":";
        s += info.description != nullptr ? info.description : (info.name != nullptr ? info.name : "?");
    }
    return s.empty() ? std::string("none") : s;
}

/** ASCII case-insensitive equality (PCI bus ids: CUDA and Vulkan may differ in hex case). */
bool iequals(std::string_view a, const char* b) {
    if (b == nullptr) return false;
    const std::string_view bv(b);
    if (a.size() != bv.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        const auto lower = [](char c) { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; };
        if (lower(a[i]) != lower(bv[i])) return false;
    }
    return true;
}

/** Resolve the `device` config member to a registered device of `kind` (nullptr when absent): an
 *  index among that kind's devices, or a stable id (the PCI bus id, case-insensitive; or the name /
 *  description), which the host uses to target the same GPU on every backend. */
transcribe_device_t resolve_device(const Value* dev, const char* kind) {
    if (dev == nullptr || dev->is_null()) return nullptr;
    if (kind == nullptr) fail(SIGIL_E_INVALID_ARGUMENT, "\"device\" needs an explicit \"backend\" (cpu, cuda or vulkan)");
    int64_t want_index = -1;
    std::string want_id;
    if (dev->is_number() && dev->as_number() >= 0 && dev->as_number() == std::floor(dev->as_number()) && dev->as_number() <= 1024) {
        want_index = static_cast<int64_t>(dev->as_number());
    } else if (dev->is_string()) {
        want_id = dev->as_string();
    } else {
        fail(SIGIL_E_INVALID_ARGUMENT, "\"device\" must be a non-negative integer or a device id string");
    }
    int64_t seen = 0;
    const int n = transcribe_device_count();
    for (int i = 0; i < n; ++i) {
        transcribe_device_t d = transcribe_device_get(i);
        transcribe_device_info info;
        transcribe_device_info_init(&info);
        if (d == nullptr || transcribe_device_get_info(d, &info) != TRANSCRIBE_OK || info.kind == nullptr) continue;
        if (std::strcmp(info.kind, kind) != 0) continue;
        if (want_index >= 0) {
            if (seen++ == want_index) return d;
        } else if (iequals(want_id, info.device_id) || (info.name != nullptr && want_id == info.name) ||
                   (info.description != nullptr && want_id == info.description)) {
            return d;
        }
    }
    fail(SIGIL_E_BACKEND_UNAVAILABLE,
         std::string("no ") + kind + " device matches \"device\" (registered: " + registered_devices_summary() + ")");
}

/* ------------------------------------------------------------------------------------------ */
/* Streams                                                                                    */
/* ------------------------------------------------------------------------------------------ */

std::string stream_json(const transcribe_session* session, const transcribe_stream_update& u, bool final_result) {
    transcribe_stream_text t;
    transcribe_stream_text_init(&t);
    transcribe_stream_get_text(session, &t);
    auto view = [](const char* p, uint64_t n) {
        return p != nullptr ? utf8_clean(std::string_view(p, static_cast<size_t>(n))) : std::string();
    };
    const std::string committed = view(t.committed_text, t.committed_text_bytes);
    const std::string tentative = view(t.tentative_text, t.tentative_text_bytes);
    const std::string full = view(t.full_text, t.full_text_bytes);
    Writer w;
    w.begin_object();
    if (final_result) {
        // full_text is the authoritative final transcript (committed_text is append-only and may
        // disagree after a late revision).
        w.field("text", full).field("committed", committed).field("truncated", transcribe_was_truncated(session));
        w.field("language", clean(transcribe_detected_language(session)));
    } else {
        w.field("partial", committed + tentative).field("committed", committed).field("tentative", tentative).field("text", full);
    }
    w.field("revision", static_cast<int64_t>(u.revision))
        .field("input_ms", static_cast<int64_t>(u.input_received_ms))
        .field("committed_audio_ms", static_cast<int64_t>(u.audio_committed_ms))
        .field("buffered_ms", static_cast<int64_t>(u.buffered_ms))
        .field("committed_changed", u.committed_changed);
    w.end_object();
    return w.take();
}

class TranscribeStream final : public sigil::Stream {
public:
    TranscribeStream(std::shared_ptr<SharedModel> shared, transcribe_session* session, RunOptions options)
        : shared_(std::move(shared)), session_(session), options_(std::move(options)) {}

    ~TranscribeStream() override {
        if (active_) {
            // Closed mid-utterance: the model is free again.
            std::lock_guard<std::mutex> lock(shared_->compute);
            release();
        }
        if (session_ != nullptr) transcribe_session_free(session_);
    }

    std::optional<std::string> push(const sigil::Audio& audio) override {
        prepare_audio(audio, mono_);
        SharedModel& m = *shared_;
        const uint64_t log_seq = engine_log_seq();
        std::lock_guard<std::mutex> lock(m.compute);
        sync_active();
        if (!active_) begin(log_seq);  // first push / next utterance
        transcribe_stream_update u;
        transcribe_stream_update_init(&u);
        const transcribe_status st = transcribe_stream_feed(session_, mono_.data(), static_cast<int>(mono_.size()), &u);
        sync_active();  // a terminal failure (or a cancel) ends the stream's claim on the model
        if (st != TRANSCRIBE_OK) fail_transcribe("stream push", st, log_seq);
        if (!u.result_changed) return std::nullopt;
        return stream_json(session_, u, false);
    }

    std::string finish() override {
        SharedModel& m = *shared_;
        const uint64_t log_seq = engine_log_seq();
        std::lock_guard<std::mutex> lock(m.compute);
        sync_active();
        transcribe_stream_update u;
        transcribe_stream_update_init(&u);
        if (!active_) {
            // Nothing pushed since the last finish: an empty final.
            Writer w;
            w.begin_object().field("text", "").field("committed", "").field("truncated", false).field("language", "");
            w.field("revision", static_cast<int64_t>(0)).field("input_ms", static_cast<int64_t>(0)).end_object();
            return w.take();
        }
        const transcribe_status st = transcribe_stream_finalize(session_, &u);
        sync_active();
        if (st != TRANSCRIBE_OK) fail_transcribe("stream finish", st, log_seq);
        return stream_json(session_, u, true);
    }

private:
    /** Drop this stream's claim on the model (caller holds `compute`). */
    void release() {
        if (!active_) return;
        active_ = false;
        if (shared_->active_streams > 0) --shared_->active_streams;
    }

    /** Match the claim to the session's state (caller holds `compute`): transcribe.cpp leaves
     *  ACTIVE on finalize and on terminal feed / finalize failures (FAILED, e.g. after a cancel). */
    void sync_active() {
        if (active_ && transcribe_stream_get_state(session_) != TRANSCRIBE_STREAM_ACTIVE) release();
    }

    /** Start the next utterance (caller holds `compute`); refused while another stream of the model
     *  is ACTIVE (one compute per model). */
    void begin(uint64_t log_seq) {
        SharedModel& m = *shared_;
        if (m.active_streams > 0) fail(SIGIL_E_INVALID_ARGUMENT, kStreamActiveMessage);
        transcribe_run_params p;
        transcribe_whisper_run_ext whisper;
        fill_run_params(m.model, m.caps, options_, p, whisper, true);
        transcribe_stream_params sp;
        transcribe_stream_params_init(&sp);
        sp.commit_policy = options_.commit;
        sp.stable_prefix_agreement_n = static_cast<uint32_t>(options_.agreement);
        const transcribe_status st = transcribe_stream_begin(session_, &p, &sp);
        if (st != TRANSCRIBE_OK) fail_transcribe("stream begin", st, log_seq);
        active_ = true;
        ++m.active_streams;
    }

    std::shared_ptr<SharedModel> shared_;
    transcribe_session* session_ = nullptr;
    RunOptions options_;
    std::vector<float> mono_;
    bool active_ = false;  // this stream holds the model's one compute slot (guarded by `compute`)
};

/* ------------------------------------------------------------------------------------------ */
/* Engine                                                                                     */
/* ------------------------------------------------------------------------------------------ */

class TranscribeEngine final : public sigil::Engine {
public:
    TranscribeEngine(std::shared_ptr<SharedModel> shared, transcribe_session* session)
        : shared_(std::move(shared)), session_(session) {
        shared_->cancel->token.store(&cancel_token(), std::memory_order_release);
        transcribe_set_abort_callback(session_, &abort_callback, shared_->cancel.get());
    }

    ~TranscribeEngine() override {
        shared_->cancel->token.store(nullptr, std::memory_order_release);
        if (session_ != nullptr) transcribe_session_free(session_);
    }

    std::string backend() const override { return clean(transcribe_model_backend(shared_->model)); }

    std::string query(const Value& request) override {
        const std::string op = request.string_or("op", "info");
        SharedModel& m = *shared_;
        Writer w;
        if (op == "info") {
            const sigil::EngineDescriptor& d = sigil::engine_descriptor();
            w.begin_object();
            w.field("name", d.name).field("version", d.version).field("variant", d.variant).field("kind", d.kind);
            w.field("backend", backend()).field("requested_backend", m.requested_backend);
            w.field("upstream_version", transcribe_version()).field("upstream_commit", transcribe_version_commit());
            field_or_null(w, "arch", transcribe_model_arch_string(m.model));
            field_or_null(w, "model_variant", transcribe_model_variant_string(m.model));
            field_or_null(w, "model_name", transcribe_model_meta_val_str(m.model, "general.name"));
            w.key("device");
            write_device(w, transcribe_model_device(m.model));
            w.key("capabilities");
            write_capabilities(w, m.model, m.caps);
            transcribe_session_limits lim;
            transcribe_session_limits_init(&lim);
            if (transcribe_session_get_limits(session_, &lim) == TRANSCRIBE_OK) {
                w.key("limits");
                write_limits(w, lim);
            }
            w.field("threads", static_cast<int64_t>(m.session_params.n_threads));
            transcribe_timings t;
            transcribe_timings_init(&t);
            transcribe_get_timings(session_, &t);
            w.field("load_ms", static_cast<double>(t.load_ms));
            w.end_object();
        } else if (op == "devices") {
            w.begin_object().key("devices").begin_array();
            const int n = transcribe_device_count();
            for (int i = 0; i < n; ++i) write_device(w, transcribe_device_get(i));
            w.end_array().end_object();
        } else if (op == "memory") {
            w.begin_object().key("device");
            write_device(w, transcribe_model_device(m.model));
            w.end_object();
        } else if (op == "languages") {
            w.begin_object().key("languages");
            write_languages(w, m.caps);
            w.field("detect", m.caps.supports_language_detect).end_object();
        } else if (op == "limits") {
            transcribe_session_limits lim;
            transcribe_session_limits_init(&lim);
            const uint64_t log_seq = engine_log_seq();
            const transcribe_status st = transcribe_session_get_limits(session_, &lim);
            if (st != TRANSCRIBE_OK) fail_transcribe("query limits", st, log_seq);
            write_limits(w, lim);
        } else if (op == "warmup") {
            // Pay the first-run costs (graph allocation, GPU kernel / pipeline setup, weight upload)
            // before the first real utterance. The text is discarded.
            const double seconds = request.number_or("seconds", 1.0);
            if (!(seconds >= 0.1 && seconds <= 30.0)) fail(SIGIL_E_INVALID_ARGUMENT, "\"seconds\" must be in [0.1, 30]");
            std::vector<float> noise(static_cast<size_t>(seconds * kSampleRate));
            uint32_t state = 0x9E3779B9u;
            for (float& s : noise) {
                state = state * 1664525u + 1013904223u;
                s = (static_cast<float>(state >> 8) / 16777216.0f - 0.5f) * 2e-4f;
            }
            RunOptions opts;
            opts.timestamps = TRANSCRIBE_TIMESTAMPS_NONE;
            const auto t0 = std::chrono::steady_clock::now();
            run_offline(m, session_, noise.data(), noise.size(), opts);
            const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            w.begin_object().field("seconds", seconds).field("wall_ms", wall_ms).end_object();
        } else {
            fail(SIGIL_E_UNSUPPORTED, "unknown query op \"" + op + "\" (info, devices, memory, languages, limits, warmup)");
        }
        return w.take();
    }

    std::string process(const sigil::Audio& audio, const Value& options) override {
        const RunOptions opts = RunOptions::parse(options);
        prepare_audio(audio, mono_);
        return run_offline(*shared_, session_, mono_.data(), mono_.size(), opts);
    }

    std::unique_ptr<sigil::Stream> open_stream(const Value& options) override {
        SharedModel& m = *shared_;
        if (!m.caps.supports_streaming) {
            const char* arch = transcribe_model_arch_string(m.model);
            fail(SIGIL_E_UNSUPPORTED, std::string("the ") + (arch != nullptr ? arch : "loaded") +
                                          " model does not stream; transcribe each utterance instead");
        }
        RunOptions opts = RunOptions::parse(options);
        if (opts.translate) fail(SIGIL_E_INVALID_ARGUMENT, "streaming is transcribe-only");
        const uint64_t log_seq = engine_log_seq();
        transcribe_session* session = nullptr;
        const transcribe_status st = transcribe_session_init(m.model, &m.session_params, &session);
        if (st != TRANSCRIBE_OK) fail_transcribe("stream open", st, log_seq);
        transcribe_set_abort_callback(session, &abort_callback, m.cancel.get());
        return std::make_unique<TranscribeStream>(shared_, session, std::move(opts));
    }

private:
    std::shared_ptr<SharedModel> shared_;
    transcribe_session* session_ = nullptr;
    std::vector<float> mono_;  // reused input buffer
};

/** An instance without a model (`create` config without "model"): reports the build and the compute
 *  devices this PC exposes to it (a device picker, the pack checker), and refuses to transcribe. */
class DeviceProbeEngine final : public sigil::Engine {
public:
    std::string backend() const override { return "none"; }

    std::string query(const Value& request) override {
        const std::string op = request.string_or("op", "info");
        Writer w;
        if (op == "info") {
            const sigil::EngineDescriptor& d = sigil::engine_descriptor();
            w.begin_object();
            w.field("name", d.name).field("version", d.version).field("variant", d.variant).field("kind", d.kind);
            w.field("backend", "none").key("model").null();
            w.field("upstream_version", transcribe_version());
            w.key("backends").begin_object();
            w.field("cpu", transcribe_backend_available(TRANSCRIBE_BACKEND_CPU))
                .field("cuda", transcribe_backend_available(TRANSCRIBE_BACKEND_CUDA))
                .field("vulkan", transcribe_backend_available(TRANSCRIBE_BACKEND_VULKAN));
            w.end_object();
            w.key("devices");
            write_devices(w);
            w.end_object();
        } else if (op == "devices") {
            w.begin_object().key("devices");
            write_devices(w);
            w.end_object();
        } else {
            fail(SIGIL_E_INVALID_ARGUMENT, "no model is loaded (create the instance with {\"model\": \"<path>\"}); only info and devices work");
        }
        return w.take();
    }

    std::string process(const sigil::Audio&, const Value&) override {
        fail(SIGIL_E_INVALID_ARGUMENT, "no model is loaded (create the instance with {\"model\": \"<path>\"})");
    }

    std::unique_ptr<sigil::Stream> open_stream(const Value&) override {
        fail(SIGIL_E_INVALID_ARGUMENT, "no model is loaded (create the instance with {\"model\": \"<path>\"})");
    }

private:
    static void write_devices(Writer& w) {
        w.begin_array();
        const int n = transcribe_device_count();
        for (int i = 0; i < n; ++i) write_device(w, transcribe_device_get(i));
        w.end_array();
    }
};

}  // namespace

namespace sigil {

const EngineDescriptor& engine_descriptor() {
    static const EngineDescriptor d{
        "transcribe",
        SIGIL_ENGINE_VERSION,
        SIGIL_ENGINE_VARIANT,
        "asr",
        SIGIL_ENGINE_BUILD_INFO ", " SIGIL_TRANSCRIBE_UPSTREAM_INFO,
        SIGIL_CAP_PROCESS | SIGIL_CAP_STREAM | SIGIL_CAP_CANCEL | (SIGIL_TRANSCRIBE_GPU ? SIGIL_CAP_GPU : 0ull),
    };
    return d;
}

std::unique_ptr<Engine> create_engine(const json::Value& config) {
    ensure_log_installed();
    const std::string model_path = config.string_or("model", "");
    if (model_path.empty()) return std::make_unique<DeviceProbeEngine>();
    const std::string backend_name = config.string_or("backend", SIGIL_TRANSCRIBE_DEFAULT_BACKEND);
    transcribe_backend_request backend = TRANSCRIBE_BACKEND_AUTO;
    const char* kind = nullptr;
    if (!backend_from_name(backend_name, backend, kind)) {
        fail(SIGIL_E_INVALID_ARGUMENT, "\"backend\" must be auto, cpu, cpu_accel, cuda or vulkan");
    }
    const int64_t threads = int_or(config, "threads", 0, kMaxThreads, 0);
    const int64_t n_ctx = int_or(config, "n_ctx", 0, INT32_MAX, 0);
    const std::string kv = config.string_or("kv_type", "auto");
    transcribe_kv_type kv_type = TRANSCRIBE_KV_TYPE_AUTO;
    if (kv == "f16") kv_type = TRANSCRIBE_KV_TYPE_F16;
    else if (kv == "f32") kv_type = TRANSCRIBE_KV_TYPE_F32;
    else if (kv != "auto") fail(SIGIL_E_INVALID_ARGUMENT, "\"kv_type\" must be auto, f16 or f32");

    if (backend != TRANSCRIBE_BACKEND_AUTO && !transcribe_backend_available(backend)) {
        fail(SIGIL_E_BACKEND_UNAVAILABLE, "the " + backend_name + " backend is not available on this PC (variant " +
                                              SIGIL_ENGINE_VARIANT + "; registered devices: " + registered_devices_summary() + ")");
    }
    const transcribe_device_t device = resolve_device(config.find("device"), kind);

    auto shared = std::make_shared<SharedModel>();
    shared->requested_backend = backend_name;
    transcribe_model_load_params lp;
    transcribe_model_load_params_init(&lp);
    lp.backend = backend;
    lp.device = device;
    const uint64_t log_seq = engine_log_seq();
    transcribe_status st = transcribe_model_load_file(model_path.c_str(), &lp, &shared->model);
    if (st != TRANSCRIBE_OK) fail_transcribe("loading " + model_path, st, log_seq);

    transcribe_capabilities_init(&shared->caps);
    st = transcribe_model_get_capabilities(shared->model, &shared->caps);
    if (st != TRANSCRIBE_OK) fail_transcribe("reading the model capabilities", st, log_seq);
    if (shared->caps.native_sample_rate != 0 && shared->caps.native_sample_rate != kSampleRate) {
        fail(SIGIL_E_LOAD_FAILED, "the model expects " + std::to_string(shared->caps.native_sample_rate) +
                                      " Hz input; only 16000 Hz models are supported");
    }
    transcribe_session_params_init(&shared->session_params);
    shared->session_params.n_threads = static_cast<int>(threads);
    shared->session_params.kv_type = kv_type;
    shared->session_params.n_ctx = static_cast<int32_t>(n_ctx);

    transcribe_session* session = nullptr;
    st = transcribe_session_init(shared->model, &shared->session_params, &session);
    if (st != TRANSCRIBE_OK) fail_transcribe("creating a session", st, log_seq);
    return std::make_unique<TranscribeEngine>(std::move(shared), session);
}

}  // namespace sigil
