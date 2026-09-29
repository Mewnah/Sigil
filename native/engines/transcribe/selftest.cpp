/*
 * selftest.cpp - sigil_transcribe_selftest.exe: loads sigil_transcribe.dll the way Sigil does
 * (LoadLibraryExW by full path with LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | DEFAULT_DIRS, nothing linked
 * at load time), checks the sigil_engine ABI v1 contract, and optionally transcribes a 16 kHz WAV.
 * Used by build.ps1 -SelfTest to verify every pack right after it is built. Not shipped.
 *
 * Part of Sigil (AGPL-3.0-only). Owned by P1-G2.
 *
 *   sigil_transcribe_selftest <pack dir> [--dll-dir <dir>]... [--info-json <out file>]
 *                             [--model <file> --wav <16k wav>] [--config <json>] [--options <json>]
 *                             [--repeat <n>]
 * Exit code 0 = every check passed.
 */
#ifndef _WIN32
#error "sigil_transcribe_selftest is Windows-only"
#endif

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include "sigil_engine.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Api {
    decltype(&sigil_engine_abi_version) abi_version = nullptr;
    decltype(&sigil_engine_get_info) get_info = nullptr;
    decltype(&sigil_engine_last_error) last_error = nullptr;
    decltype(&sigil_engine_set_log_callback) set_log_callback = nullptr;
    decltype(&sigil_engine_create) create = nullptr;
    decltype(&sigil_engine_destroy) destroy = nullptr;
    decltype(&sigil_engine_cancel) cancel = nullptr;
    decltype(&sigil_engine_query) query = nullptr;
    decltype(&sigil_engine_process) process = nullptr;
    decltype(&sigil_engine_render) render = nullptr;
    decltype(&sigil_stream_open) stream_open = nullptr;
    decltype(&sigil_stream_push) stream_push = nullptr;
    decltype(&sigil_stream_finish) stream_finish = nullptr;
    decltype(&sigil_stream_close) stream_close = nullptr;
    decltype(&sigil_free_string) free_string = nullptr;
    decltype(&sigil_free_audio) free_audio = nullptr;
};

int g_failures = 0;

void check(bool ok, const char* what) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", what);
    if (!ok) ++g_failures;
}

std::wstring widen(const std::string& s) {
    if (s.empty()) return std::wstring();
    int n = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), static_cast<int>(s.size()), w.data(), n);
    return w;
}

template <typename T>
bool resolve(HMODULE m, const char* name, T& out) {
    out = reinterpret_cast<T>(reinterpret_cast<void*>(GetProcAddress(m, name)));
    if (out == nullptr) std::printf("missing export %s\n", name);
    return out != nullptr;
}

void SIGIL_ENGINE_CALL on_log(void*, int32_t level, const char* message) {
    if (level >= SIGIL_LOG_WARN) std::fprintf(stderr, "  [engine %s] %s\n", level >= SIGIL_LOG_ERROR ? "error" : "warn", message);
}

/** Minimal RIFF reader: PCM16 or float32, any channel count; returns interleaved floats. */
bool read_wav(const std::string& path, std::vector<float>& samples, uint32_t& rate, uint32_t& channels) {
    std::ifstream f(widen(path), std::ios::binary);
    if (!f) return false;
    std::vector<char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    if (data.size() < 12 || std::memcmp(data.data(), "RIFF", 4) != 0 || std::memcmp(data.data() + 8, "WAVE", 4) != 0) return false;
    size_t pos = 12;
    uint16_t format = 0, bits = 0;
    bool have_fmt = false;
    while (pos + 8 <= data.size()) {
        uint32_t size = 0;
        std::memcpy(&size, data.data() + pos + 4, 4);
        const char* id = data.data() + pos;
        const size_t body = pos + 8;
        if (body + size > data.size()) size = static_cast<uint32_t>(data.size() - body);
        if (std::memcmp(id, "fmt ", 4) == 0 && size >= 16) {
            uint16_t ch = 0;
            std::memcpy(&format, data.data() + body, 2);
            std::memcpy(&ch, data.data() + body + 2, 2);
            std::memcpy(&rate, data.data() + body + 4, 4);
            std::memcpy(&bits, data.data() + body + 14, 2);
            if (format == 0xFFFE && size >= 26) std::memcpy(&format, data.data() + body + 24, 2);  // extensible
            channels = ch;
            have_fmt = true;
        } else if (std::memcmp(id, "data", 4) == 0 && have_fmt && channels > 0) {
            if (format == 1 && bits == 16) {
                samples.resize(size / 2);
                for (size_t i = 0; i < samples.size(); ++i) {
                    int16_t v = 0;
                    std::memcpy(&v, data.data() + body + i * 2, 2);
                    samples[i] = static_cast<float>(v) / 32768.0f;
                }
                return true;
            }
            if (format == 3 && bits == 32) {
                samples.resize(size / 4);
                std::memcpy(samples.data(), data.data() + body, samples.size() * 4);
                return true;
            }
            return false;
        }
        pos = body + size + (size & 1u);
    }
    return false;
}

std::string take(const Api& api, char* s) {
    std::string out = s != nullptr ? s : "";
    api.free_string(s);
    return out;
}

std::string json_str(const char* s) {
    std::string out = "\"";
    for (const char* p = s != nullptr ? s : ""; *p != '\0'; ++p) {
        const unsigned char c = static_cast<unsigned char>(*p);
        if (c == '"' || c == '\\') {
            out += '\\';
            out += static_cast<char>(c);
        } else if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "\\u%04x", c);
            out += buf;
        } else {
            out += static_cast<char>(c);
        }
    }
    return out + "\"";
}

/** The DLL's own sigil_engine_get_info as JSON (build.ps1 copies it into sigil-pack.json). */
bool write_info_json(const std::string& path, const sigil_engine_info* info) {
    std::ofstream f(widen(path), std::ios::binary | std::ios::trunc);
    if (!f) return false;
    f << "{\"abi_version\":" << info->abi_version << ",\"name\":" << json_str(info->name)
      << ",\"version\":" << json_str(info->version) << ",\"variant\":" << json_str(info->variant)
      << ",\"kind\":" << json_str(info->kind) << ",\"build_info\":" << json_str(info->build_info)
      << ",\"capabilities\":" << info->capabilities << "}\n";
    return static_cast<bool>(f);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: sigil_transcribe_selftest <pack dir> [--dll-dir <dir>]... [--model <file> --wav <16k wav>]\n"
                     "                                 [--config <json>] [--options <json>] [--repeat <n>]\n");
        return 2;
    }
    std::string pack_dir = argv[1];
    std::vector<std::string> dll_dirs;
    std::string model, wav, config_extra, info_json, options = "{}";
    int repeat = 3;
    for (int i = 2; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return i + 1 < argc ? std::string(argv[++i]) : std::string(); };
        if (a == "--dll-dir") dll_dirs.push_back(next());
        else if (a == "--info-json") info_json = next();
        else if (a == "--model") model = next();
        else if (a == "--wav") wav = next();
        else if (a == "--config") config_extra = next();
        else if (a == "--options") options = next();
        else if (a == "--repeat") repeat = std::max(1, std::atoi(next().c_str()));
        else {
            std::fprintf(stderr, "unknown argument %s\n", a.c_str());
            return 2;
        }
    }

    // Same search rules as Sigil: the DLL's own folder, System32, and directories added explicitly.
    SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    for (const auto& d : dll_dirs) {
        if (AddDllDirectory(widen(d).c_str()) == nullptr) std::printf("warning: AddDllDirectory(%s) failed\n", d.c_str());
    }
    const std::string dll_path = pack_dir + "\\sigil_transcribe.dll";
    HMODULE lib = LoadLibraryExW(widen(dll_path).c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (lib == nullptr) {
        std::printf("[FAIL] LoadLibraryExW(%s): Win32 error %lu\n", dll_path.c_str(), GetLastError());
        return 1;
    }
    check(GetModuleHandleW(L"ggml.dll") == nullptr && GetModuleHandleW(L"ggml-base.dll") == nullptr,
          "no ggml*.dll module was loaded (ggml is static inside the engine)");

    Api api;
    bool all = resolve(lib, "sigil_engine_abi_version", api.abi_version) & resolve(lib, "sigil_engine_get_info", api.get_info) &
               resolve(lib, "sigil_engine_last_error", api.last_error) &
               resolve(lib, "sigil_engine_set_log_callback", api.set_log_callback) &
               resolve(lib, "sigil_engine_create", api.create) & resolve(lib, "sigil_engine_destroy", api.destroy) &
               resolve(lib, "sigil_engine_cancel", api.cancel) & resolve(lib, "sigil_engine_query", api.query) &
               resolve(lib, "sigil_engine_process", api.process) & resolve(lib, "sigil_engine_render", api.render) &
               resolve(lib, "sigil_stream_open", api.stream_open) & resolve(lib, "sigil_stream_push", api.stream_push) &
               resolve(lib, "sigil_stream_finish", api.stream_finish) & resolve(lib, "sigil_stream_close", api.stream_close) &
               resolve(lib, "sigil_free_string", api.free_string) & resolve(lib, "sigil_free_audio", api.free_audio);
    check(all, "all 16 ABI v1 exports resolve");
    if (!all) return 1;
    check(api.abi_version() == SIGIL_ENGINE_ABI_VERSION, "sigil_engine_abi_version() == 1");
    const sigil_engine_info* info = api.get_info();
    check(info != nullptr && info->struct_size >= sizeof(sigil_engine_info) && info->abi_version == SIGIL_ENGINE_ABI_VERSION,
          "sigil_engine_get_info() is v1");
    if (info == nullptr) return 1;
    std::printf("engine: %s %s variant=%s kind=%s caps=0x%llx\n  %s\n", info->name, info->version, info->variant, info->kind,
                static_cast<unsigned long long>(info->capabilities), info->build_info);
    check(std::strcmp(info->name, "transcribe") == 0 && std::strcmp(info->kind, "asr") == 0, "name/kind = transcribe/asr");
    if (!info_json.empty()) check(write_info_json(info_json, info), "wrote --info-json");
    check((info->capabilities & (SIGIL_CAP_PROCESS | SIGIL_CAP_STREAM | SIGIL_CAP_CANCEL)) ==
              (SIGIL_CAP_PROCESS | SIGIL_CAP_STREAM | SIGIL_CAP_CANCEL),
          "capabilities include PROCESS | STREAM | CANCEL");
    api.set_log_callback(&on_log, nullptr);

    // Contract checks that need no model.
    sigil_engine* e = nullptr;
    char* out = nullptr;
    check(api.create("{}", nullptr) == SIGIL_E_INVALID_ARGUMENT, "create(out = NULL) -> INVALID_ARGUMENT");
    // Without a model: an instance that only reports the build and its devices.
    check(api.create("{}", &e) == SIGIL_OK && e != nullptr, "create({}) -> device-probe instance");
    if (e != nullptr) {
        check(api.query(e, "{\"op\":\"info\"}", &out) == SIGIL_OK && out != nullptr, "device-probe info");
        std::printf("       %s\n", take(api, out).c_str());
        float silence[160] = {};
        sigil_audio_view one{silence, 160, 16000, 1};
        check(api.process(e, &one, "{}", &out) == SIGIL_E_INVALID_ARGUMENT && out == nullptr && std::strlen(api.last_error()) > 0,
              "process without a model -> INVALID_ARGUMENT + message");
        sigil_audio_buffer rb{};
        check(api.render(e, nullptr, nullptr, &rb) == SIGIL_E_UNSUPPORTED && rb.samples == nullptr, "render -> UNSUPPORTED");
        api.destroy(e);
        e = nullptr;
    }
    check(api.create("{\"model\": ", &e) == SIGIL_E_INVALID_ARGUMENT && e == nullptr, "malformed JSON -> INVALID_ARGUMENT");
    check(api.create("{\"model\":\"C:\\\\definitely\\\\missing.gguf\",\"backend\":\"cpu\"}", &e) == SIGIL_E_LOAD_FAILED &&
              e == nullptr,
          "missing model file -> LOAD_FAILED");
    std::printf("       (message: %s)\n", api.last_error());
    check(api.create("{\"model\":\"x.gguf\",\"backend\":\"quantum\"}", &e) == SIGIL_E_INVALID_ARGUMENT, "unknown backend -> INVALID_ARGUMENT");
    out = reinterpret_cast<char*>(1);
    check(api.query(nullptr, "{}", &out) == SIGIL_E_INVALID_ARGUMENT && out == nullptr, "query(NULL engine) -> INVALID_ARGUMENT, out = NULL");
    sigil_audio_buffer buf{};
    check(api.render(nullptr, nullptr, nullptr, &buf) == SIGIL_E_INVALID_ARGUMENT && buf.samples == nullptr,
          "render(NULL engine) -> INVALID_ARGUMENT");
    api.free_string(nullptr);
    api.free_audio(nullptr);
    api.free_audio(&buf);
    api.stream_close(nullptr);
    api.destroy(nullptr);
    api.cancel(nullptr);
    check(true, "NULL-tolerant free/close/destroy/cancel");

    if (model.empty()) {
        std::printf("%s (no --model: ABI checks only)\n", g_failures == 0 ? "PASS" : "FAILED");
        return g_failures == 0 ? 0 : 1;
    }

    // Model run.
    std::string cfg = "{\"model\":\"";
    for (char c : model) cfg += (c == '\\') ? std::string("\\\\") : std::string(1, c);
    cfg += "\"";
    if (!config_extra.empty()) {
        std::string inner = config_extra;
        if (!inner.empty() && inner.front() == '{') inner = inner.substr(1, inner.size() - 2);
        if (!inner.empty()) cfg += "," + inner;
    }
    cfg += "}";
    auto t0 = std::chrono::steady_clock::now();
    sigil_status st = api.create(cfg.c_str(), &e);
    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    check(st == SIGIL_OK && e != nullptr, "create(model)");
    if (st != SIGIL_OK) {
        std::printf("       status %d: %s\n", st, api.last_error());
        return 1;
    }
    std::printf("load: %.1f ms\n", load_ms);
    check(api.query(e, "{\"op\":\"info\"}", &out) == SIGIL_OK && out != nullptr, "query info");
    std::printf("info: %s\n", take(api, out).c_str());
    check(api.query(e, "{\"op\":\"nope\"}", &out) == SIGIL_E_UNSUPPORTED && out == nullptr, "unknown query op -> UNSUPPORTED");
    t0 = std::chrono::steady_clock::now();
    check(api.query(e, "{\"op\":\"warmup\"}", &out) == SIGIL_OK, "warmup");
    std::printf("warmup: %s (%.1f ms)\n", take(api, out).c_str(),
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());

    if (!wav.empty()) {
        std::vector<float> pcm;
        uint32_t rate = 0, channels = 0;
        if (!read_wav(wav, pcm, rate, channels)) {
            check(false, "read the WAV (PCM16 or float32)");
            return 1;
        }
        sigil_audio_view view{pcm.data(), pcm.size() / channels, rate, channels};
        sigil_audio_view bad_rate = view;
        bad_rate.sample_rate = 44100;
        check(api.process(e, &bad_rate, "{}", &out) == SIGIL_E_INVALID_ARGUMENT && out == nullptr, "non-16 kHz input -> INVALID_ARGUMENT");
        check(api.process(e, &view, "{\"timestamps\":\"bogus\"}", &out) == SIGIL_E_INVALID_ARGUMENT, "bad option -> INVALID_ARGUMENT");
        for (int i = 0; i < repeat; ++i) {
            t0 = std::chrono::steady_clock::now();
            st = api.process(e, &view, options.c_str(), &out);
            double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            check(st == SIGIL_OK && out != nullptr, "process(wav)");
            if (st != SIGIL_OK) {
                std::printf("       status %d: %s\n", st, api.last_error());
                break;
            }
            std::string result = take(api, out);
            std::printf("run %d: %.1f ms (audio %.1f ms, RTF %.4f)\n", i + 1, ms, view.frames * 1000.0 / rate,
                        ms / (view.frames * 1000.0 / rate));
            if (i == repeat - 1) std::printf("result: %s\n", result.c_str());
        }
        check(api.query(e, "{\"op\":\"memory\"}", &out) == SIGIL_OK, "query memory");
        std::printf("memory: %s\n", take(api, out).c_str());
        sigil_stream* s = nullptr;
        st = api.stream_open(e, "{}", &s);
        if (st == SIGIL_E_UNSUPPORTED) {
            check(s == nullptr, "non-streaming model: stream_open -> UNSUPPORTED");
        } else {
            check(st == SIGIL_OK && s != nullptr, "stream_open");
            const uint64_t chunk = rate / 5;  // 200 ms
            {
                // One compute per model: while `s` is mid-utterance, another stream's first push and
                // an offline process() are refused; closing `s` mid-utterance frees the model.
                sigil_audio_view first{pcm.data(), std::min<uint64_t>(chunk, view.frames), rate, channels};
                out = nullptr;
                check(api.stream_push(s, &first, &out) == SIGIL_OK, "stream_push (first chunk)");
                if (out != nullptr) api.free_string(out);
                sigil_stream* s2 = nullptr;
                check(api.stream_open(e, "{}", &s2) == SIGIL_OK && s2 != nullptr, "second stream_open");
                out = nullptr;
                check(api.stream_push(s2, &first, &out) == SIGIL_E_INVALID_ARGUMENT && out == nullptr,
                      "second stream's push while another is active -> INVALID_ARGUMENT");
                check(api.process(e, &view, "{}", &out) == SIGIL_E_INVALID_ARGUMENT && out == nullptr,
                      "process while a stream is active -> INVALID_ARGUMENT");
                api.stream_close(s);
                s = nullptr;
                out = nullptr;
                check(api.stream_push(s2, &first, &out) == SIGIL_OK, "push once the active stream closed mid-utterance");
                if (out != nullptr) api.free_string(out);
                check(api.stream_finish(s2, &out) == SIGIL_OK, "stream_finish (second stream)");
                take(api, out);
                api.stream_close(s2);
                check(api.stream_open(e, "{}", &s) == SIGIL_OK && s != nullptr, "stream_open (again)");
            }
            int partials = 0;
            for (uint64_t f = 0; st == SIGIL_OK && f < view.frames; f += chunk) {
                sigil_audio_view part{pcm.data() + f * channels, std::min<uint64_t>(chunk, view.frames - f), rate, channels};
                out = nullptr;
                st = api.stream_push(s, &part, &out);
                if (out != nullptr) {
                    ++partials;
                    api.free_string(out);
                }
            }
            check(st == SIGIL_OK, "stream_push (200 ms chunks)");
            check(api.stream_finish(s, &out) == SIGIL_OK, "stream_finish");
            std::printf("stream: %d partial updates, final %s\n", partials, take(api, out).c_str());
            api.stream_close(s);
        }
    }
    api.destroy(e);
    check(true, "destroy");
    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
