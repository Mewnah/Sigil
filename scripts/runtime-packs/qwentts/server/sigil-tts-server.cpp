// sigil-tts-server.cpp: Sigil's build of the qwentts.cpp OpenAI-compatible TTS server (runtime packs
// runtime/qwentts-server-*, P6-E). Started by Sigil's process host as the `tts-qwen3` helper: a
// kill-on-close Job Object, 127.0.0.1 only, a random port and key.
//
// Derived from qwentts.cpp tools/tts-server.cpp (MIT License, Copyright (c) 2023-2026 The
// omnivoice.cpp authors; https://github.com/ServeurpersoCom/qwentts.cpp). It reuses the shared HTTP
// core of src/tts-server.h and the public qt_* ABI of src/qwen.h unchanged; build.ps1 copies this
// file into the pinned source tree as tools/sigil-tts-server.cpp and appends the target from
// server/CMakeLists.append.cmake. Differences from upstream's tts-server:
//
//   - Bearer key. When the environment variable QWENTTS_API_KEY is set (Sigil's process host sets a
//     random 256-bit key; it is never accepted on the command line, which every local process can
//     read), every route except GET /health answers 401 without `Authorization: Bearer <key>`.
//   - No wildcard CORS and no OPTIONS route: Sigil's client is native code; web pages in a browser
//     on this PC are not meant to drive the server.
//   - Loopback only: --host other than 127.0.0.1 / localhost / ::1 is refused unless --allow-remote.
//   - POST /v1/audio/voices with a WAV answers the latents it extracted (spk_b64 + rvq_b64, the
//     .spk / .rvq payloads of the latents form), so the caller can cache them per voice and
//     register them without the extraction next time.
//   - GET /sigil/info: model type, speakers, languages, sample rate, codebooks and versions.
//   - A request language the model does not know is spoken with "auto" instead of failing, and an
//     instruction sent to a base model (which rejects one) is dropped; both are logged once per value.
//
// Endpoints otherwise as upstream: POST /v1/audio/speech (pcm streams s16le 24 kHz mono as it is
// generated, wav answers one file), GET /v1/models, GET /v1/audio/voices, DELETE
// /v1/audio/voices/{name}, GET /health.

#include "tts-server.h"

#include "qwen.h"
#include "rvq-file.h"
#include "version.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#ifndef SIGIL_QWENTTS_COMMIT
#    define SIGIL_QWENTTS_COMMIT "unknown"
#endif
#ifndef SIGIL_PACK_VERSION
#    define SIGIL_PACK_VERSION "dev"
#endif
#ifndef SIGIL_PACK_VARIANT
#    define SIGIL_PACK_VARIANT "unknown"
#endif

// Packed .rvq code width, fixed by the Qwen3-TTS 12 Hz codec (2048 entries per codebook).
static const int RVQ_CODE_BITS = 11;

// Environment variable holding the API key (never a command-line argument).
static const char * API_KEY_ENV = "QWENTTS_API_KEY";

// One registered cloned voice (as upstream): the extraction latents in the ABI ownership contract
// (malloc-owned, released by qt_voice_ref_free) plus the reference transcript (ICL clone mode).
struct voice_entry {
    struct qt_voice_ref ref;
    std::string         ref_text;
};

static std::mutex                                   g_voices_mutex;
static std::unordered_map<std::string, voice_entry> g_voices;

// Values already logged by the lenient request handling (log once per value, bounded).
static std::mutex            g_logged_mutex;
static std::set<std::string> g_logged;

static void log_once(const std::string & key, const char * fmt, const char * value) {
    {
        std::lock_guard<std::mutex> lock(g_logged_mutex);
        if (g_logged.size() > 256 || !g_logged.insert(key).second) {
            return;
        }
    }
    fprintf(stderr, fmt, value);
}

static void print_usage(const char * prog) {
    fprintf(stderr, "sigil-tts-server %s (qwentts.cpp %s, %s)\n\n", SIGIL_PACK_VERSION, SIGIL_QWENTTS_COMMIT, SIGIL_PACK_VARIANT);
    fprintf(stderr,
            "Usage: %s --model <gguf> --codec <gguf> [options]\n\n"
            "Required:\n"
            "  --model <gguf>          Talker LM GGUF (qwen-talker-*.gguf)\n"
            "  --codec <gguf>          Codec GGUF (qwen-tokenizer-*.gguf)\n\n"
            "Optional:\n"
            "  --alias <name>          Report this model id instead of the GGUF file name\n"
            "  --host <ip>             Listen address (default: 127.0.0.1; loopback only without --allow-remote)\n"
            "  --port <n>              Listen port (default: 8080)\n"
            "  --lang <name>           Language label when a request omits one (default: auto)\n"
            "  --max-batch <n>         Concurrent requests batched on the GPU (default: 1)\n"
            "  --no-fa                 Disable flash attention\n"
            "  --clamp-fp16            Clamp hidden states to FP16 range\n"
            "  --codec-chunk-dur <f>   Codec decode chunk duration in seconds, wav responses (default: 24.0)\n"
            "  --allow-remote          Allow a non-loopback --host\n\n"
            "Environment:\n"
            "  QWENTTS_API_KEY         Require 'Authorization: Bearer <key>' on every route but GET /health\n",
            prog);
}

// Trim a path down to its file name for the reported model id.
static std::string basename_of(const char * path) {
    std::string s = path;
    size_t      p = s.find_last_of("/\\");
    return p == std::string::npos ? s : s.substr(p + 1);
}

static std::string lower(std::string s) {
    for (char & c : s) {
        c = (char) std::tolower((unsigned char) c);
    }
    return s;
}

static bool is_loopback_host(const std::string & host) {
    const std::string h = lower(host);
    return h == "127.0.0.1" || h == "localhost" || h == "::1" || h == "[::1]";
}

// Compare without an early exit on the first differing byte.
static bool constant_time_equal(const std::string & a, const std::string & b) {
    unsigned char diff = (unsigned char) (a.size() != b.size());
    const size_t  n    = a.size() < b.size() ? a.size() : b.size();
    for (size_t i = 0; i < n; i++) {
        diff |= (unsigned char) (a[i] ^ b[i]);
    }
    return diff == 0;
}

static bool authorized(const std::string & key, const httplib::Request & req) {
    if (key.empty()) {
        return true;
    }
    const std::string header = req.get_header_value("Authorization");
    const std::string prefix = "Bearer ";
    if (header.size() <= prefix.size() || lower(header.substr(0, prefix.size())) != lower(prefix)) {
        return false;
    }
    return constant_time_equal(header.substr(prefix.size()), key);
}

static std::string b64_encode(const uint8_t * data, size_t size) {
    static const char * alpha = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string         out;
    out.reserve((size + 2) / 3 * 4);
    size_t i = 0;
    for (; i + 2 < size; i += 3) {
        const uint32_t v = ((uint32_t) data[i] << 16) | ((uint32_t) data[i + 1] << 8) | (uint32_t) data[i + 2];
        out.push_back(alpha[(v >> 18) & 63]);
        out.push_back(alpha[(v >> 12) & 63]);
        out.push_back(alpha[(v >> 6) & 63]);
        out.push_back(alpha[v & 63]);
    }
    if (i < size) {
        uint32_t v = (uint32_t) data[i] << 16;
        if (i + 1 < size) {
            v |= (uint32_t) data[i + 1] << 8;
        }
        out.push_back(alpha[(v >> 18) & 63]);
        out.push_back(alpha[(v >> 12) & 63]);
        out.push_back(i + 1 < size ? alpha[(v >> 6) & 63] : '=');
        out.push_back('=');
    }
    return out;
}

// Latents of a registration, answered to the client (the .spk and .rvq payload forms).
struct voice_latents {
    bool        extracted = false;  // computed from a WAV (else they came from the client)
    std::string spk;                // raw f32 values
    std::string rvq;                // packed codes
    int         ref_T     = 0;
    int         spk_dim   = 0;
};

// Register (or replace) a cloned voice from a WAV (server side extraction) or from latents.
static bool register_voice(struct qt_context * q, const tts_voice_upload & up, std::string & err, voice_latents & out) {
    const std::string mt = qt_model_type(q);
    if (mt != "base") {
        err = "voice registration is only valid for base models (loaded: " + mt + ")";
        return false;
    }

    voice_entry entry;
    entry.ref      = {};
    entry.ref_text = up.ref_text;

    if (!up.wav.empty()) {
        int     T   = 0;
        float * pcm = audio_read_mono_buf((const uint8_t *) up.wav.data(), up.wav.size(), 24000, &T);
        if (!pcm) {
            err = "cannot decode the WAV payload";
            return false;
        }
        enum qt_status rc = qt_extract_voice_ref(q, pcm, T, &entry.ref);
        free(pcm);
        if (rc != QT_STATUS_OK) {
            err = qt_last_error();
            return false;
        }
        out.extracted = true;
        out.spk.assign((const char *) entry.ref.ref_spk_emb, (size_t) entry.ref.ref_spk_dim * sizeof(float));
        if (entry.ref.ref_codes && entry.ref.ref_T > 0 && entry.ref.num_codebooks > 0) {
            std::vector<int32_t> codes(entry.ref.ref_codes,
                                       entry.ref.ref_codes + (size_t) entry.ref.num_codebooks * (size_t) entry.ref.ref_T);
            std::vector<uint8_t> packed = rvq_pack_codes(codes, RVQ_CODE_BITS);
            out.rvq.assign((const char *) packed.data(), packed.size());
        }
    } else {
        if (up.spk.size() % sizeof(float) != 0 || up.spk.empty()) {
            err = "'spk_b64' must decode to a positive multiple of 4 bytes";
            return false;
        }
        std::vector<int32_t> codes;
        int                  ref_T = 0;
        const int            K     = qt_num_codebooks(q);
        if (!rvq_read_buf((const uint8_t *) up.rvq.data(), up.rvq.size(), K, RVQ_CODE_BITS, codes, &ref_T)) {
            err = "'rvq_b64' does not decode to a valid packed code stream";
            return false;
        }
        entry.ref.ref_spk_dim = (int) (up.spk.size() / sizeof(float));
        entry.ref.ref_spk_emb = (float *) malloc(up.spk.size());
        if (!entry.ref.ref_spk_emb) {
            err = "out of memory";
            return false;
        }
        std::memcpy(entry.ref.ref_spk_emb, up.spk.data(), up.spk.size());
        entry.ref.ref_T         = ref_T;
        entry.ref.num_codebooks = K;
        entry.ref.ref_codes     = (int32_t *) malloc(codes.size() * sizeof(int32_t));
        if (!entry.ref.ref_codes) {
            qt_voice_ref_free(&entry.ref);
            err = "out of memory";
            return false;
        }
        std::memcpy(entry.ref.ref_codes, codes.data(), codes.size() * sizeof(int32_t));
    }
    out.ref_T   = entry.ref.ref_T;
    out.spk_dim = entry.ref.ref_spk_dim;

    std::lock_guard<std::mutex> lock(g_voices_mutex);
    auto                        it = g_voices.find(up.name);
    if (it != g_voices.end()) {
        qt_voice_ref_free(&it->second.ref);
        g_voices.erase(it);
    }
    fprintf(stderr, "[Server] voice '%s' registered (T=%d, spk_dim=%d, ref_text=%s, %s)\n", up.name.c_str(), entry.ref.ref_T,
            entry.ref.ref_spk_dim, entry.ref_text.empty() ? "no" : "yes", out.extracted ? "extracted" : "latents");
    g_voices.emplace(up.name, std::move(entry));
    return true;
}

static void handle_voice_register(struct qt_context * q, const httplib::Request & http_req, httplib::Response & res) {
    tts_voice_upload up;
    std::string      err;
    if (!tts_parse_voice_upload(http_req.body, up, err)) {
        tts_json_error(res, 400, "invalid_request_error", err.c_str());
        return;
    }
    voice_latents latents;
    if (!register_voice(q, up, err, latents)) {
        tts_json_error(res, 400, "invalid_request_error", err.empty() ? "voice registration failed" : err.c_str());
        return;
    }
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_strcpy(doc, root, "name", up.name.c_str());
    yyjson_mut_obj_add_str(doc, root, "status", "registered");
    yyjson_mut_obj_add_bool(doc, root, "extracted", latents.extracted);
    yyjson_mut_obj_add_int(doc, root, "ref_T", latents.ref_T);
    yyjson_mut_obj_add_int(doc, root, "spk_dim", latents.spk_dim);
    yyjson_mut_obj_add_int(doc, root, "num_codebooks", qt_num_codebooks(q));
    if (latents.extracted && !latents.spk.empty() && !latents.rvq.empty()) {
        const std::string spk = b64_encode((const uint8_t *) latents.spk.data(), latents.spk.size());
        const std::string rvq = b64_encode((const uint8_t *) latents.rvq.data(), latents.rvq.size());
        yyjson_mut_obj_add_strcpy(doc, root, "spk_b64", spk.c_str());
        yyjson_mut_obj_add_strcpy(doc, root, "rvq_b64", rvq.c_str());
    }
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

static void handle_info(struct qt_context *              q,
                        const std::string &              model_id,
                        const std::vector<std::string> & languages,
                        int                              max_batch,
                        httplib::Response &              res) {
    yyjson_mut_doc * doc  = yyjson_mut_doc_new(NULL);
    yyjson_mut_val * root = yyjson_mut_obj(doc);
    yyjson_mut_doc_set_root(doc, root);
    yyjson_mut_obj_add_str(doc, root, "server", "sigil-tts-server");
    yyjson_mut_obj_add_str(doc, root, "pack_version", SIGIL_PACK_VERSION);
    yyjson_mut_obj_add_str(doc, root, "variant", SIGIL_PACK_VARIANT);
    yyjson_mut_obj_add_str(doc, root, "upstream_commit", SIGIL_QWENTTS_COMMIT);
    yyjson_mut_obj_add_strcpy(doc, root, "library_version", qt_version());
    yyjson_mut_obj_add_strcpy(doc, root, "model", model_id.c_str());
    yyjson_mut_obj_add_strcpy(doc, root, "model_type", qt_model_type(q));
    yyjson_mut_obj_add_int(doc, root, "sample_rate", 24000);
    yyjson_mut_obj_add_int(doc, root, "num_codebooks", qt_num_codebooks(q));
    yyjson_mut_obj_add_int(doc, root, "max_batch", max_batch);
    yyjson_mut_val * speakers = yyjson_mut_arr(doc);
    for (int i = 0; i < qt_n_speakers(q); i++) {
        const char * name = qt_speaker_name(q, i);
        if (name) {
            yyjson_mut_arr_add_strcpy(doc, speakers, name);
        }
    }
    yyjson_mut_obj_add_val(doc, root, "speakers", speakers);
    yyjson_mut_val * langs = yyjson_mut_arr(doc);
    for (const std::string & l : languages) {
        yyjson_mut_arr_add_strcpy(doc, langs, l.c_str());
    }
    yyjson_mut_obj_add_val(doc, root, "languages", langs);
    char * json = yyjson_mut_write(doc, 0, NULL);
    res.set_content(json ? json : "{}", "application/json");
    if (json) {
        free(json);
    }
    yyjson_mut_doc_free(doc);
}

int main(int argc, char ** argv) {
    const char *  talker_path = NULL;
    const char *  codec_path  = NULL;
    std::string   model_alias;
    std::string   lang = "auto";
    server_config cfg;
    bool          use_fa          = true;
    bool          clamp_fp16      = false;
    bool          allow_remote    = false;
    int           max_batch       = 1;
    float         codec_chunk_dur = 0.0f;

    for (int i = 1; i < argc; i++) {
        const char * arg = argv[i];
        if (!std::strcmp(arg, "--model") && i + 1 < argc) {
            talker_path = argv[++i];
        } else if (!std::strcmp(arg, "--codec") && i + 1 < argc) {
            codec_path = argv[++i];
        } else if (!std::strcmp(arg, "--alias") && i + 1 < argc) {
            model_alias = argv[++i];
        } else if (!std::strcmp(arg, "--host") && i + 1 < argc) {
            cfg.host = argv[++i];
        } else if (!std::strcmp(arg, "--port") && i + 1 < argc) {
            cfg.port = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--lang") && i + 1 < argc) {
            lang = argv[++i];
        } else if (!std::strcmp(arg, "--no-fa")) {
            use_fa = false;
        } else if (!std::strcmp(arg, "--clamp-fp16")) {
            clamp_fp16 = true;
        } else if (!std::strcmp(arg, "--max-batch") && i + 1 < argc) {
            max_batch = std::atoi(argv[++i]);
        } else if (!std::strcmp(arg, "--codec-chunk-dur") && i + 1 < argc) {
            codec_chunk_dur = (float) std::atof(argv[++i]);
        } else if (!std::strcmp(arg, "--allow-remote")) {
            allow_remote = true;
        } else if (!std::strcmp(arg, "--version")) {
            printf("sigil-tts-server %s (qwentts.cpp %s, library %s, %s)\n", SIGIL_PACK_VERSION, SIGIL_QWENTTS_COMMIT, qt_version(),
                   SIGIL_PACK_VARIANT);
            return 0;
        } else {
            print_usage(argv[0]);
            return 1;
        }
    }

    if (!talker_path || !codec_path) {
        print_usage(argv[0]);
        return 1;
    }
    if (cfg.port <= 0 || cfg.port > 65535) {
        fprintf(stderr, "[Server] FATAL: invalid --port\n");
        return 1;
    }
    if (!allow_remote && !is_loopback_host(cfg.host)) {
        fprintf(stderr, "[Server] FATAL: --host %s is not a loopback address (pass --allow-remote to listen on the network)\n",
                cfg.host.c_str());
        return 2;
    }
    if (max_batch < 1) {
        max_batch = 1;
    }

    const char *      key_env = std::getenv(API_KEY_ENV);
    const std::string api_key = key_env ? key_env : "";

    fprintf(stderr, "[Server] sigil-tts-server %s (qwentts.cpp %s, %s)\n", SIGIL_PACK_VERSION, SIGIL_QWENTTS_COMMIT, SIGIL_PACK_VARIANT);

    struct qt_init_params iparams;
    qt_init_default_params(&iparams);
    iparams.talker_path     = talker_path;
    iparams.codec_path      = codec_path;
    iparams.use_fa          = use_fa;
    iparams.clamp_fp16      = clamp_fp16;
    iparams.max_batch       = max_batch;
    iparams.codec_chunk_sec = codec_chunk_dur;

    struct qt_context * q = qt_init(&iparams);
    if (!q) {
        fprintf(stderr, "[Server] FATAL: %s\n", qt_last_error());
        return 1;
    }

    const std::string model_type = qt_model_type(q);
    std::vector<std::string> languages;
    for (int i = 0; i < qt_n_languages(q); i++) {
        const char * name = qt_language_name(q, i);
        if (name) {
            languages.push_back(lower(name));
        }
    }

    tts_backend be;
    be.model_id = model_alias.empty() ? basename_of(talker_path) : model_alias;
    for (int i = 0; i < qt_n_speakers(q); i++) {
        be.voices.push_back(qt_speaker_name(q, i));
    }

    be.remove_voice = [](const std::string & name) -> bool {
        std::lock_guard<std::mutex> lock(g_voices_mutex);
        auto                        it = g_voices.find(name);
        if (it == g_voices.end()) {
            return false;
        }
        qt_voice_ref_free(&it->second.ref);
        g_voices.erase(it);
        return true;
    };

    be.registered_voices = []() -> std::vector<std::string> {
        std::lock_guard<std::mutex> lock(g_voices_mutex);
        std::vector<std::string>    names;
        names.reserve(g_voices.size());
        for (const auto & kv : g_voices) {
            names.push_back(kv.first);
        }
        return names;
    };

    // As upstream's adapter, plus the lenient language and instruction handling.
    be.synthesize = [q, &lang, &languages, &model_type](const tts_request & req, const tts_sink & sink, std::string & err) -> int {
        struct qt_tts_params p;
        qt_tts_default_params(&p);
        p.text = req.input.c_str();

        std::string req_lang = lower(req.lang.empty() ? lang : req.lang);
        if (req_lang != "auto") {
            bool known = false;
            for (const std::string & l : languages) {
                known = known || l == req_lang;
            }
            if (!known) {
                log_once("lang:" + req_lang, "[Server] language '%s' is not one of the model's: speaking with auto\n", req_lang.c_str());
                req_lang = "auto";
            }
        }
        p.lang = req_lang.c_str();

        std::vector<float>   voice_spk;
        std::vector<int32_t> voice_codes;
        std::string          voice_ref_text;
        int                  voice_ref_T = 0;
        bool                 have_voice  = false;
        if (!req.voice.empty()) {
            std::lock_guard<std::mutex> lock(g_voices_mutex);
            auto                        vit = g_voices.find(req.voice);
            if (vit != g_voices.end()) {
                const voice_entry & v = vit->second;
                have_voice            = true;
                voice_spk.assign(v.ref.ref_spk_emb, v.ref.ref_spk_emb + v.ref.ref_spk_dim);
                if (!v.ref_text.empty() && v.ref.ref_codes) {
                    voice_codes.assign(v.ref.ref_codes,
                                       v.ref.ref_codes + (size_t) v.ref.num_codebooks * (size_t) v.ref.ref_T);
                    voice_ref_T    = v.ref.ref_T;
                    voice_ref_text = v.ref_text;
                }
            }
        }
        if (have_voice) {
            p.ref_spk_emb = voice_spk.data();
            p.ref_spk_dim = (int) voice_spk.size();
            if (!voice_codes.empty()) {
                p.ref_codes = voice_codes.data();
                p.ref_T     = voice_ref_T;
                p.ref_text  = voice_ref_text.c_str();
            }
        } else if (!req.voice.empty() && qt_n_speakers(q) > 0) {
            p.speaker = req.voice.c_str();
        } else if (!req.voice.empty()) {
            err = "unknown voice '" + req.voice + "'";
            return (int) QT_STATUS_INVALID_PARAMS;
        }
        if (!req.instructions.empty()) {
            if (model_type == "base") {
                log_once("instruct:base", "[Server] %s models take no instruction: ignored\n", "base");
            } else {
                p.instruct = req.instructions.c_str();
            }
        }

        p.seed = req.seed;
        if (req.max_new_tokens != -1) {
            p.max_new_tokens = req.max_new_tokens;
        }
        if (req.top_k != -1) {
            p.top_k = req.top_k;
        }
        if (!std::isnan(req.temperature)) {
            p.temperature = req.temperature;
        }
        if (!std::isnan(req.top_p)) {
            p.top_p = req.top_p;
        }
        if (req.subtalker_top_k != -1) {
            p.subtalker_top_k = req.subtalker_top_k;
        }
        if (!std::isnan(req.subtalker_temperature)) {
            p.subtalker_temperature = req.subtalker_temperature;
        }
        if (!std::isnan(req.subtalker_top_p)) {
            p.subtalker_top_p = req.subtalker_top_p;
        }
        if (!std::isnan(req.repetition_penalty)) {
            p.repetition_penalty = req.repetition_penalty;
        }

        const tts_sink * sink_ptr = &sink;
        if (req.format == "pcm") {
            p.on_chunk = [](const float * s, int ns, void * u) -> bool {
                return (*static_cast<const tts_sink *>(u))(s, ns);
            };
            p.on_chunk_user_data = (void *) sink_ptr;
        }

        struct qt_audio out = {};
        enum qt_status  rc  = qt_synthesize(q, &p, &out);
        if (rc == QT_STATUS_OK && p.on_chunk == NULL && out.n_samples > 0) {
            sink(out.samples, out.n_samples);
        }
        qt_audio_free(&out);
        if (rc != QT_STATUS_OK) {
            err = qt_last_error();
            return (int) rc;
        }
        return 0;
    };

    httplib::Server svr;
    g_svr = &svr;
    svr.set_read_timeout(60);
    svr.set_write_timeout(120);
    svr.set_payload_max_length(32 * 1024 * 1024);
    svr.set_tcp_nodelay(true);
    svr.set_socket_options([](socket_t sock) {
        int one = 1;
#ifdef _WIN32
        // Exclusive: no other socket may bind the same port while this one listens.
        setsockopt(sock, SOL_SOCKET, SO_EXCLUSIVEADDRUSE, (const char *) &one, sizeof(one));
#else
        setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    });

    svr.set_pre_routing_handler([&api_key](const httplib::Request & req, httplib::Response & res) {
        if (req.method == "GET" && req.path == "/health") {
            return httplib::Server::HandlerResponse::Unhandled;
        }
        if (!authorized(api_key, req)) {
            tts_json_error(res, 401, "authentication_error", "missing or wrong API key");
            return httplib::Server::HandlerResponse::Handled;
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    svr.Post("/v1/audio/speech", [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_speech(be, req, res); });
    svr.Get("/v1/models", [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_models(be, req, res); });
    svr.Get("/v1/audio/voices", [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voices(be, req, res); });
    svr.Post("/v1/audio/voices", [q](const httplib::Request & req, httplib::Response & res) { handle_voice_register(q, req, res); });
    svr.Delete(R"(/v1/audio/voices/(.+))",
               [&be](const httplib::Request & req, httplib::Response & res) { tts_handle_voice_delete(be, req, res); });
    svr.Get("/sigil/info", [q, &be, &languages, max_batch](const httplib::Request &, httplib::Response & res) {
        handle_info(q, be.model_id, languages, max_batch, res);
    });
    svr.Get("/health", tts_handle_health);

    signal(SIGINT, tts_on_signal);
    signal(SIGTERM, tts_on_signal);

    qt_log(QT_LOG_INFO, "[Server] model %s (%s), %d speakers, %d languages, key %s", be.model_id.c_str(), model_type.c_str(),
           (int) be.voices.size(), (int) languages.size(), api_key.empty() ? "off" : "required");
    qt_log(QT_LOG_INFO, "[Server] listening on %s:%d", cfg.host.c_str(), cfg.port);
    int rc = 0;
    if (!svr.listen(cfg.host.c_str(), cfg.port)) {
        qt_log(QT_LOG_ERROR, "[Server] FATAL: cannot bind %s:%d", cfg.host.c_str(), cfg.port);
        rc = 1;
    }
    g_svr = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_voices_mutex);
        for (auto & kv : g_voices) {
            qt_voice_ref_free(&kv.second.ref);
        }
        g_voices.clear();
    }
    qt_free(q);
    return rc;
}
