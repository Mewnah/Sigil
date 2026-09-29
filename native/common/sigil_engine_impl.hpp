// SPDX-License-Identifier: AGPL-3.0-only
//
// sigil_engine_impl.hpp - the shared implementation of the sigil_engine ABI v1 for C++ engines.
//
// sigil_engine_impl.cpp defines all 16 exports of sigil_engine.h once, with every rule of the
// header enforced in one place: exceptions never cross the boundary (they become status codes plus
// a thread-local last error), JSON arguments are parsed and validated, audio views are checked,
// returned strings and audio are allocated here and released by sigil_free_string /
// sigil_free_audio of the same DLL, cancellation is an atomic flag per instance, and log lines go
// to the host callback. An engine DLL (native/engines/<name>/) only:
//   1. defines sigil::engine_descriptor() (static info) and sigil::create_engine(config), and
//   2. derives from sigil::Engine, overriding the calls its capabilities announce.
// sigil_add_engine() (native/cmake/SigilEngine.cmake) compiles this implementation into the DLL
// with SIGIL_ENGINE_BUILD defined, so the exports come from the engine DLL itself.
#ifndef SIGIL_ENGINE_IMPL_HPP
#define SIGIL_ENGINE_IMPL_HPP

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "sigil_engine.h"
#include "sigil_json.hpp"

namespace sigil {

/** An error with a sigil_status; thrown anywhere inside an engine call. */
class Error : public std::runtime_error {
public:
    Error(sigil_status status, const std::string& message) : std::runtime_error(message), status_(status) {}
    sigil_status status() const noexcept { return status_; }

private:
    sigil_status status_;
};

/** Throw sigil::Error(status, message). */
[[noreturn]] void fail(sigil_status status, const std::string& message);

/** Static description of the engine build (every string static and non-NULL). */
struct EngineDescriptor {
    const char* name;        // engine id, e.g. "echo"
    const char* version;     // e.g. SIGIL_ENGINE_VERSION
    const char* variant;     // e.g. SIGIL_ENGINE_VARIANT
    const char* kind;        // "asr", "tts", "vc", "lid", "test"
    const char* build_info;  // e.g. SIGIL_ENGINE_BUILD_INFO
    uint64_t capabilities;   // SIGIL_CAP_* bits
};

/** Borrowed, validated input audio (interleaved f32). */
struct Audio {
    const float* samples = nullptr;  // NULL only when frames == 0
    uint64_t frames = 0;
    uint32_t sample_rate = 0;
    uint32_t channels = 0;

    size_t sample_count() const noexcept { return static_cast<size_t>(frames) * channels; }
    double seconds() const noexcept { return sample_rate ? static_cast<double>(frames) / sample_rate : 0.0; }
};

/** Audio produced by render(). */
struct AudioOut {
    std::vector<float> samples;  // interleaved; size must be a multiple of channels
    uint32_t sample_rate = 0;
    uint32_t channels = 0;
};

/** Per-instance cancellation flag. The host's sigil_engine_cancel sets it from any thread; the
 *  shared implementation clears it when a call starts, so a cancel arriving while no call runs is
 *  ignored (header: "cancellation of the running call"). Long work polls requested(). */
class CancelToken {
public:
    void request() noexcept { flag_.store(true, std::memory_order_release); }
    void reset() noexcept { flag_.store(false, std::memory_order_release); }
    bool requested() const noexcept { return flag_.load(std::memory_order_acquire); }
    /** Throw SIGIL_E_CANCELLED when a cancel was requested. */
    void check() const {
        if (requested()) fail(SIGIL_E_CANCELLED, "cancelled");
    }

private:
    std::atomic<bool> flag_{false};
};

/** A streaming session (SIGIL_CAP_STREAM). */
class Stream {
public:
    virtual ~Stream() = default;
    /** Push audio; return a partial-result JSON document, or nothing. */
    virtual std::optional<std::string> push(const Audio& audio) = 0;
    /** Flush and return the final-result JSON; the stream must accept new audio afterwards. */
    virtual std::string finish() = 0;
};

/** One engine instance. Calls on an instance are serialized by the host. */
class Engine {
public:
    virtual ~Engine() = default;

    /** sigil_engine_query. The default answers {"op":"info"} (the header's minimum) and rejects
     *  every other op with SIGIL_E_INVALID_ARGUMENT. */
    virtual std::string query(const json::Value& request);
    /** sigil_engine_process (SIGIL_CAP_PROCESS). Default: SIGIL_E_UNSUPPORTED. */
    virtual std::string process(const Audio& audio, const json::Value& options);
    /** sigil_engine_render (SIGIL_CAP_RENDER); `input` is nullptr for generators. Default:
     *  SIGIL_E_UNSUPPORTED. */
    virtual AudioOut render(const Audio* input, const json::Value& options);
    /** sigil_stream_open (SIGIL_CAP_STREAM). Default: SIGIL_E_UNSUPPORTED. */
    virtual std::unique_ptr<Stream> open_stream(const json::Value& options);
    /** Backend in use, reported by the default {"op":"info"} ("cpu", "vulkan", "cuda"). */
    virtual std::string backend() const { return "cpu"; }

    CancelToken& cancel_token() noexcept { return cancel_; }
    bool cancelled() const noexcept { return cancel_.requested(); }

private:
    CancelToken cancel_;
};

/** Implemented once by every engine DLL. */
const EngineDescriptor& engine_descriptor();
/** Implemented once by every engine DLL: create an instance from the create config (a JSON
 *  object; `{}` when the host passed NULL). Throw sigil::Error on invalid configuration
 *  (SIGIL_E_INVALID_ARGUMENT), a missing model (SIGIL_E_LOAD_FAILED) or an unavailable backend
 *  (SIGIL_E_BACKEND_UNAVAILABLE: the host then tries the next build variant). */
std::unique_ptr<Engine> create_engine(const json::Value& config);

/** Send a log line to the host (SIGIL_LOG_*). Never throws; dropped when no callback is set. */
void log(int32_t level, const std::string& message) noexcept;

/** Sleep for up to `ms` milliseconds in short slices, throwing SIGIL_E_CANCELLED as soon as the
 *  token is cancelled (for engines simulating or waiting on long work). */
void sleep_cancellable(const CancelToken& token, double ms);

}  // namespace sigil

#endif  // SIGIL_ENGINE_IMPL_HPP
