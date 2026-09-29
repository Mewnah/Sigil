// SPDX-License-Identifier: AGPL-3.0-only
//
// sigil_engine_impl.cpp - the 16 exports of sigil_engine.h (ABI v1), shared by every C++ engine.
// See sigil_engine_impl.hpp. Compiled into each engine DLL with SIGIL_ENGINE_BUILD defined.
#include "sigil_engine_impl.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <thread>

#if !defined(SIGIL_ENGINE_BUILD)
#error "sigil_engine_impl.cpp must be compiled with SIGIL_ENGINE_BUILD (use sigil_add_engine)"
#endif

// Opaque handles of the header.
struct sigil_engine {
    std::unique_ptr<sigil::Engine> impl;
    std::atomic<int> open_streams{0};
};

struct sigil_stream {
    sigil_engine* owner = nullptr;
    std::unique_ptr<sigil::Stream> impl;
};

namespace sigil {

void fail(sigil_status status, const std::string& message) { throw Error(status, message); }

namespace {

// Largest input the shared layer accepts (samples, all channels): 2^30 f32 = 4 GiB.
constexpr uint64_t kMaxSamples = uint64_t{1} << 30;
constexpr uint32_t kMaxChannels = 64;
constexpr uint32_t kMaxSampleRate = 1536000;

thread_local std::string t_last_error;

std::mutex g_log_mutex;
sigil_log_fn g_log_fn = nullptr;
void* g_log_user = nullptr;

void set_last_error(const char* message) noexcept {
    try {
        t_last_error = message ? message : "";
    } catch (...) {
        // Out of memory while reporting: keep whatever was there.
    }
}

/** Run `body`, turning every exception into a status code and the thread's last error. */
template <class F>
sigil_status guarded(F&& body) noexcept {
    try {
        t_last_error.clear();
        return body();
    } catch (const Error& e) {
        set_last_error(e.what());
        return e.status() < 0 ? e.status() : SIGIL_E_RUNTIME;
    } catch (const std::bad_alloc&) {
        set_last_error("out of memory");
        return SIGIL_E_OUT_OF_MEMORY;
    } catch (const std::invalid_argument& e) {
        set_last_error(e.what());
        return SIGIL_E_INVALID_ARGUMENT;
    } catch (const std::exception& e) {
        set_last_error(e.what());
        return SIGIL_E_RUNTIME;
    } catch (...) {
        set_last_error("unknown C++ exception");
        return SIGIL_E_RUNTIME;
    }
}

/** A NULL-terminated copy owned by the caller (released by sigil_free_string). */
char* dup_string(const std::string& s) {
    char* p = static_cast<char*>(std::malloc(s.size() + 1));
    if (!p) throw std::bad_alloc();
    std::memcpy(p, s.data(), s.size());
    p[s.size()] = '\0';
    return p;
}

/** Parse a JSON argument that must be an object (NULL or blank: {}). */
json::Value parse_object(const char* text, const char* what) {
    if (!text) return json::Value::object({});
    std::string_view view(text);
    if (view.find_first_not_of(" \t\r\n") == std::string_view::npos) return json::Value::object({});
    json::Value v;
    try {
        v = json::Value::parse(view);
    } catch (const std::invalid_argument& e) {
        fail(SIGIL_E_INVALID_ARGUMENT, std::string(what) + ": " + e.what());
    }
    if (!v.is_object()) fail(SIGIL_E_INVALID_ARGUMENT, std::string(what) + " must be a JSON object");
    return v;
}

/** Validate a host audio view. */
Audio check_audio(const sigil_audio_view* view) {
    if (!view) fail(SIGIL_E_INVALID_ARGUMENT, "audio is NULL");
    if (view->channels == 0 || view->channels > kMaxChannels) {
        fail(SIGIL_E_INVALID_ARGUMENT, "unsupported channel count " + std::to_string(view->channels));
    }
    if (view->sample_rate == 0 || view->sample_rate > kMaxSampleRate) {
        fail(SIGIL_E_INVALID_ARGUMENT, "unsupported sample rate " + std::to_string(view->sample_rate));
    }
    if (view->frames > kMaxSamples / view->channels) fail(SIGIL_E_INVALID_ARGUMENT, "audio is too long");
    if (view->frames > 0 && !view->samples) fail(SIGIL_E_INVALID_ARGUMENT, "audio samples are NULL");
    Audio a;
    a.samples = view->frames ? view->samples : nullptr;
    a.frames = view->frames;
    a.sample_rate = view->sample_rate;
    a.channels = view->channels;
    return a;
}

sigil::Engine& engine_of(sigil_engine* engine) {
    if (!engine || !engine->impl) fail(SIGIL_E_INVALID_ARGUMENT, "engine is NULL");
    return *engine->impl;
}

const sigil_engine_info& info_block() {
    static const sigil_engine_info info = [] {
        const EngineDescriptor& d = engine_descriptor();
        sigil_engine_info i{};
        i.struct_size = static_cast<uint32_t>(sizeof(sigil_engine_info));
        i.abi_version = SIGIL_ENGINE_ABI_VERSION;
        i.name = d.name ? d.name : "";
        i.version = d.version ? d.version : "";
        i.variant = d.variant ? d.variant : "";
        i.kind = d.kind ? d.kind : "";
        i.build_info = d.build_info ? d.build_info : "";
        i.capabilities = d.capabilities;
        return i;
    }();
    return info;
}

bool has_cap(uint64_t cap) { return (info_block().capabilities & cap) == cap; }

}  // namespace

void log(int32_t level, const std::string& message) noexcept {
    sigil_log_fn fn = nullptr;
    void* user = nullptr;
    try {
        std::lock_guard<std::mutex> lock(g_log_mutex);
        fn = g_log_fn;
        user = g_log_user;
    } catch (...) {
        return;
    }
    if (fn) fn(user, level, message.c_str());
}

void sleep_cancellable(const CancelToken& token, double ms) {
    using clock = std::chrono::steady_clock;
    if (!(ms > 0.0)) return;
    const auto until = clock::now() + std::chrono::duration<double, std::milli>(std::min(ms, 3.6e6));
    while (clock::now() < until) {
        token.check();
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    token.check();
}

std::string Engine::query(const json::Value& request) {
    const std::string op = request.string_or("op", "info");
    if (op != "info") fail(SIGIL_E_INVALID_ARGUMENT, "unknown query op \"" + op + "\"");
    const sigil_engine_info& i = info_block();
    json::Writer w;
    w.begin_object()
        .field("name", i.name)
        .field("version", i.version)
        .field("variant", i.variant)
        .field("kind", i.kind)
        .field("backend", backend())
        .end_object();
    return w.take();
}

std::string Engine::process(const Audio&, const json::Value&) {
    fail(SIGIL_E_UNSUPPORTED, "this engine does not process audio clips");
}

AudioOut Engine::render(const Audio*, const json::Value&) {
    fail(SIGIL_E_UNSUPPORTED, "this engine does not render audio");
}

std::unique_ptr<Stream> Engine::open_stream(const json::Value&) {
    fail(SIGIL_E_UNSUPPORTED, "this engine does not stream");
}

}  // namespace sigil

using sigil::fail;

extern "C" {

SIGIL_ENGINE_API uint32_t SIGIL_ENGINE_CALL sigil_engine_abi_version(void) { return SIGIL_ENGINE_ABI_VERSION; }

SIGIL_ENGINE_API const sigil_engine_info* SIGIL_ENGINE_CALL sigil_engine_get_info(void) {
    try {
        return &sigil::info_block();
    } catch (...) {
        return nullptr;
    }
}

SIGIL_ENGINE_API const char* SIGIL_ENGINE_CALL sigil_engine_last_error(void) { return sigil::t_last_error.c_str(); }

SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_engine_set_log_callback(sigil_log_fn fn, void* user_data) {
    try {
        std::lock_guard<std::mutex> lock(sigil::g_log_mutex);
        sigil::g_log_fn = fn;
        sigil::g_log_user = user_data;
    } catch (...) {
    }
}

SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_engine_create(const char* config_json, sigil_engine** out_engine) {
    return sigil::guarded([&]() -> sigil_status {
        if (!out_engine) fail(SIGIL_E_INVALID_ARGUMENT, "out_engine is NULL");
        *out_engine = nullptr;
        const sigil::json::Value config = sigil::parse_object(config_json, "config");
        std::unique_ptr<sigil::Engine> impl = sigil::create_engine(config);
        if (!impl) fail(SIGIL_E_RUNTIME, "the engine returned no instance");
        auto* engine = new sigil_engine();
        engine->impl = std::move(impl);
        *out_engine = engine;
        return SIGIL_OK;
    });
}

SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_engine_destroy(sigil_engine* engine) {
    if (!engine) return;
    try {
        if (engine->open_streams.load() > 0) {
            sigil::log(SIGIL_LOG_ERROR, "sigil_engine_destroy called with open streams (host bug)");
        }
        delete engine;
    } catch (...) {
    }
}

SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_engine_cancel(sigil_engine* engine) {
    try {
        if (engine && engine->impl && sigil::has_cap(SIGIL_CAP_CANCEL)) engine->impl->cancel_token().request();
    } catch (...) {
    }
}

SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_engine_query(sigil_engine* engine, const char* request_json, char** out_json) {
    return sigil::guarded([&]() -> sigil_status {
        if (!out_json) fail(SIGIL_E_INVALID_ARGUMENT, "out_json is NULL");
        *out_json = nullptr;
        sigil::Engine& e = sigil::engine_of(engine);
        e.cancel_token().reset();
        const sigil::json::Value request = sigil::parse_object(request_json, "request");
        *out_json = sigil::dup_string(e.query(request));
        return SIGIL_OK;
    });
}

SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_engine_process(sigil_engine* engine,
                                                                     const sigil_audio_view* audio,
                                                                     const char* options_json,
                                                                     char** out_json) {
    return sigil::guarded([&]() -> sigil_status {
        if (!out_json) fail(SIGIL_E_INVALID_ARGUMENT, "out_json is NULL");
        *out_json = nullptr;
        sigil::Engine& e = sigil::engine_of(engine);
        if (!sigil::has_cap(SIGIL_CAP_PROCESS)) fail(SIGIL_E_UNSUPPORTED, "this engine does not process audio clips");
        const sigil::Audio a = sigil::check_audio(audio);
        const sigil::json::Value options = sigil::parse_object(options_json, "options");
        e.cancel_token().reset();
        *out_json = sigil::dup_string(e.process(a, options));
        return SIGIL_OK;
    });
}

SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_engine_render(sigil_engine* engine,
                                                                    const sigil_audio_view* input,
                                                                    const char* options_json,
                                                                    sigil_audio_buffer* out) {
    return sigil::guarded([&]() -> sigil_status {
        if (!out) fail(SIGIL_E_INVALID_ARGUMENT, "out is NULL");
        *out = sigil_audio_buffer{};
        sigil::Engine& e = sigil::engine_of(engine);
        if (!sigil::has_cap(SIGIL_CAP_RENDER)) fail(SIGIL_E_UNSUPPORTED, "this engine does not render audio");
        std::optional<sigil::Audio> a;
        if (input) a = sigil::check_audio(input);
        const sigil::json::Value options = sigil::parse_object(options_json, "options");
        e.cancel_token().reset();
        sigil::AudioOut r = e.render(a ? &*a : nullptr, options);
        if (r.samples.empty()) {
            out->sample_rate = r.sample_rate;
            out->channels = r.channels;
            return SIGIL_OK;
        }
        if (r.channels == 0 || r.channels > sigil::kMaxChannels || r.samples.size() % r.channels != 0) {
            fail(SIGIL_E_RUNTIME, "render produced a malformed buffer");
        }
        if (r.sample_rate == 0 || r.sample_rate > sigil::kMaxSampleRate) fail(SIGIL_E_RUNTIME, "render produced no sample rate");
        auto* samples = static_cast<float*>(std::malloc(r.samples.size() * sizeof(float)));
        if (!samples) throw std::bad_alloc();
        std::memcpy(samples, r.samples.data(), r.samples.size() * sizeof(float));
        out->samples = samples;
        out->frames = r.samples.size() / r.channels;
        out->sample_rate = r.sample_rate;
        out->channels = r.channels;
        return SIGIL_OK;
    });
}

SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_stream_open(sigil_engine* engine, const char* options_json, sigil_stream** out_stream) {
    return sigil::guarded([&]() -> sigil_status {
        if (!out_stream) fail(SIGIL_E_INVALID_ARGUMENT, "out_stream is NULL");
        *out_stream = nullptr;
        sigil::Engine& e = sigil::engine_of(engine);
        if (!sigil::has_cap(SIGIL_CAP_STREAM)) fail(SIGIL_E_UNSUPPORTED, "this engine does not stream");
        const sigil::json::Value options = sigil::parse_object(options_json, "options");
        e.cancel_token().reset();
        std::unique_ptr<sigil::Stream> impl = e.open_stream(options);
        if (!impl) fail(SIGIL_E_RUNTIME, "the engine returned no stream");
        auto* stream = new sigil_stream();
        stream->owner = engine;
        stream->impl = std::move(impl);
        engine->open_streams.fetch_add(1);
        *out_stream = stream;
        return SIGIL_OK;
    });
}

SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_stream_push(sigil_stream* stream, const sigil_audio_view* audio, char** out_json) {
    return sigil::guarded([&]() -> sigil_status {
        if (!out_json) fail(SIGIL_E_INVALID_ARGUMENT, "out_json is NULL");
        *out_json = nullptr;
        if (!stream || !stream->impl || !stream->owner) fail(SIGIL_E_INVALID_ARGUMENT, "stream is NULL");
        const sigil::Audio a = sigil::check_audio(audio);
        sigil::engine_of(stream->owner).cancel_token().reset();
        std::optional<std::string> partial = stream->impl->push(a);
        if (partial) *out_json = sigil::dup_string(*partial);
        return SIGIL_OK;
    });
}

SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_stream_finish(sigil_stream* stream, char** out_json) {
    return sigil::guarded([&]() -> sigil_status {
        if (!out_json) fail(SIGIL_E_INVALID_ARGUMENT, "out_json is NULL");
        *out_json = nullptr;
        if (!stream || !stream->impl || !stream->owner) fail(SIGIL_E_INVALID_ARGUMENT, "stream is NULL");
        sigil::engine_of(stream->owner).cancel_token().reset();
        *out_json = sigil::dup_string(stream->impl->finish());
        return SIGIL_OK;
    });
}

SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_stream_close(sigil_stream* stream) {
    if (!stream) return;
    try {
        if (stream->owner) stream->owner->open_streams.fetch_sub(1);
        delete stream;
    } catch (...) {
    }
}

SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_free_string(char* s) { std::free(s); }

SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_free_audio(sigil_audio_buffer* buffer) {
    if (!buffer) return;
    std::free(buffer->samples);
    *buffer = sigil_audio_buffer{};
}

}  // extern "C"
