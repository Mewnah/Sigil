/*
 * sigil_engine.h - Sigil native engine ABI, version 1.
 *
 * Every in-process native engine Sigil loads lazily (sigil_<engine>.dll: echo, transcribe, ...)
 * exports exactly this C ABI. Sigil.exe never links an engine at load time: native_runtime (Rust)
 * loads the DLL by full path with LoadLibraryExW(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
 * LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) from its runtime-pack folder, verifies its hash, calls
 * sigil_engine_abi_version() FIRST and refuses the DLL unless it returns SIGIL_ENGINE_ABI_VERSION,
 * then resolves every function below. A missing export refuses the DLL (return
 * SIGIL_E_UNSUPPORTED from functions an engine does not implement instead of omitting them).
 *
 * Frozen by P1-S0b. ABI v1 never changes; incompatible changes bump SIGIL_ENGINE_ABI_VERSION and
 * the Rust mirror in src-tauri/src/services/native_runtime (module `abi`) together.
 *
 * Build rules for engine DLLs (P1-D1 CMake presets): /MD, ggml linked statically inside the DLL
 * under a unique DLL name, GGML_NATIVE=OFF, GGML_BACKEND_DL=OFF; separate cpu-avx2 / cpu-avx512 /
 * vulkan / cuda13 variants. Dependencies (cudart64_13, ...) are found through the DLL's own folder
 * or directories added with AddDllDirectory by the host.
 *
 * Contract for every function:
 *   - Strings are UTF-8 and NUL-terminated. JSON arguments may be NULL, meaning "{}".
 *   - Memory returned by the engine (char* JSON, sigil_audio_buffer samples) is owned by the
 *     caller and released ONLY with sigil_free_string / sigil_free_audio of the same DLL.
 *   - Memory passed in by the host is borrowed for the duration of the call only.
 *   - Calls on one sigil_engine (and its streams) must be serialized by the host. Different
 *     engine instances may be used from different threads concurrently. sigil_engine_cancel may
 *     be called from any thread at any time while the instance is alive.
 *   - No C++ exception, longjmp, abort() or exit() may cross the boundary: engines catch
 *     everything and return an error status instead.
 *   - On failure a function returns a negative sigil_status and sets a message readable with
 *     sigil_engine_last_error() on the same thread, valid until the next call on that thread.
 *   - Engines must not print to stdout/stderr; they log through the callback set with
 *     sigil_engine_set_log_callback (which may be called from any engine thread).
 *   - Threads an engine starts are joined before sigil_engine_destroy returns.
 *   - Audio is 32-bit float, interleaved, nominal range [-1, 1].
 */
#ifndef SIGIL_ENGINE_H
#define SIGIL_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------------------------ */
/* Export / calling convention                                                                */
/* ------------------------------------------------------------------------------------------ */

#if defined(_WIN32)
#  if defined(SIGIL_ENGINE_BUILD)
#    define SIGIL_ENGINE_API __declspec(dllexport)
#  else
#    define SIGIL_ENGINE_API __declspec(dllimport)
#  endif
#  define SIGIL_ENGINE_CALL __cdecl
#else
#  if defined(SIGIL_ENGINE_BUILD)
#    define SIGIL_ENGINE_API __attribute__((visibility("default")))
#  else
#    define SIGIL_ENGINE_API
#  endif
#  define SIGIL_ENGINE_CALL
#endif

/* ------------------------------------------------------------------------------------------ */
/* Versions, status codes, capabilities                                                       */
/* ------------------------------------------------------------------------------------------ */

/** The ABI described by this header. */
#define SIGIL_ENGINE_ABI_VERSION 1u

/** Status code: 0 = success, negative = failure (details in sigil_engine_last_error()). */
typedef int32_t sigil_status;

#define SIGIL_OK 0
/** A NULL pointer, malformed JSON, unsupported sample rate / channel count, ... */
#define SIGIL_E_INVALID_ARGUMENT (-1)
/** The engine does not implement this function (see capabilities). */
#define SIGIL_E_UNSUPPORTED (-2)
/** The model or backend failed to load (missing / corrupt file, incompatible model). */
#define SIGIL_E_LOAD_FAILED (-3)
/** Any other failure while running. */
#define SIGIL_E_RUNTIME (-4)
/** Allocation failed (host RAM or device memory). */
#define SIGIL_E_OUT_OF_MEMORY (-5)
/** sigil_engine_cancel interrupted the call. */
#define SIGIL_E_CANCELLED (-6)
/** The requested backend is unavailable on this PC (no CUDA device, driver too old, ...). */
#define SIGIL_E_BACKEND_UNAVAILABLE (-7)

/** sigil_engine_process: one audio clip -> result JSON (e.g. offline ASR transcript). */
#define SIGIL_CAP_PROCESS (1ull << 0)
/** sigil_stream_*: incremental audio -> partial results JSON (e.g. streaming ASR). */
#define SIGIL_CAP_STREAM (1ull << 1)
/** sigil_engine_render: audio (or text in options) -> audio (echo, voice conversion, TTS). */
#define SIGIL_CAP_RENDER (1ull << 2)
/** sigil_engine_cancel interrupts running calls. */
#define SIGIL_CAP_CANCEL (1ull << 3)
/** This build can run on a GPU backend (vulkan / cuda). */
#define SIGIL_CAP_GPU (1ull << 4)

/** Log levels passed to the log callback. */
#define SIGIL_LOG_TRACE 0
#define SIGIL_LOG_DEBUG 1
#define SIGIL_LOG_INFO 2
#define SIGIL_LOG_WARN 3
#define SIGIL_LOG_ERROR 4

/* ------------------------------------------------------------------------------------------ */
/* Types                                                                                      */
/* ------------------------------------------------------------------------------------------ */

/** Static description of the engine build. Every string is static, UTF-8, never NULL. */
typedef struct sigil_engine_info {
    /** sizeof(sigil_engine_info) as compiled into the engine (later versions only append). */
    uint32_t struct_size;
    /** == SIGIL_ENGINE_ABI_VERSION. */
    uint32_t abi_version;
    /** Engine id, e.g. "echo", "transcribe". */
    const char* name;
    /** Engine version, e.g. "0.3.1+g1a2b3c4". */
    const char* version;
    /** Runtime variant: "cpu", "cpu-avx2", "cpu-avx512", "vulkan", "cuda13". */
    const char* variant;
    /** Engine kind: "asr", "tts", "vc", "lid", "test". */
    const char* kind;
    /** Free-form build details (compiler, upstream commit, ggml flags). */
    const char* build_info;
    /** SIGIL_CAP_* bit set. */
    uint64_t capabilities;
} sigil_engine_info;

/** Borrowed input audio. */
typedef struct sigil_audio_view {
    const float* samples; /* interleaved, frames * channels values */
    uint64_t frames;
    uint32_t sample_rate;
    uint32_t channels;
} sigil_audio_view;

/** Engine-allocated output audio; release with sigil_free_audio. */
typedef struct sigil_audio_buffer {
    float* samples; /* interleaved, frames * channels values; NULL when frames == 0 */
    uint64_t frames;
    uint32_t sample_rate;
    uint32_t channels;
} sigil_audio_buffer;

/** Opaque engine instance (one loaded model / backend). */
typedef struct sigil_engine sigil_engine;
/** Opaque streaming session of one engine instance. */
typedef struct sigil_stream sigil_stream;

/** Log callback: `message` is valid only during the call. May be called from any engine thread. */
typedef void(SIGIL_ENGINE_CALL* sigil_log_fn)(void* user_data, int32_t level, const char* message);

/* ------------------------------------------------------------------------------------------ */
/* Functions (all exported by every ABI v1 engine)                                             */
/* ------------------------------------------------------------------------------------------ */

/** Returns SIGIL_ENGINE_ABI_VERSION. The host calls this first and nothing else on a mismatch. */
SIGIL_ENGINE_API uint32_t SIGIL_ENGINE_CALL sigil_engine_abi_version(void);

/** Static build description (never NULL, valid while the DLL is loaded). */
SIGIL_ENGINE_API const sigil_engine_info* SIGIL_ENGINE_CALL sigil_engine_get_info(void);

/** Message of the last failure on the calling thread ("" when none). Valid until the next call. */
SIGIL_ENGINE_API const char* SIGIL_ENGINE_CALL sigil_engine_last_error(void);

/** Route engine logs to `fn` (NULL: drop logs). Process-wide; call before sigil_engine_create. */
SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_engine_set_log_callback(sigil_log_fn fn, void* user_data);

/**
 * Create an instance. `config_json` carries at least {"model": "<absolute path>"} for engines with
 * models, plus engine options such as {"backend": "cuda", "device": 0, "threads": 2}.
 */
SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_engine_create(const char* config_json, sigil_engine** out_engine);

/** Destroy an instance (NULL is ignored). Its streams must be closed first. */
SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_engine_destroy(sigil_engine* engine);

/** Request cancellation of the running call on `engine` (thread-safe; no-op without SIGIL_CAP_CANCEL). */
SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_engine_cancel(sigil_engine* engine);

/**
 * Engine-specific request/response ("info", "languages", "warmup", ...). Always implemented;
 * {"op":"info"} returns at least {"name","version","variant","backend"}.
 */
SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_engine_query(sigil_engine* engine, const char* request_json, char** out_json);

/** Process one clip (SIGIL_CAP_PROCESS). ASR engines return {"text", "language", "segments":[...]}. */
SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_engine_process(sigil_engine* engine,
                                                                     const sigil_audio_view* audio,
                                                                     const char* options_json,
                                                                     char** out_json);

/**
 * Produce audio (SIGIL_CAP_RENDER). `input` may be NULL for generators (TTS: text in options_json).
 * On success `*out` owns engine memory; release it with sigil_free_audio.
 */
SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_engine_render(sigil_engine* engine,
                                                                    const sigil_audio_view* input,
                                                                    const char* options_json,
                                                                    sigil_audio_buffer* out);

/** Open a streaming session (SIGIL_CAP_STREAM). */
SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_stream_open(sigil_engine* engine, const char* options_json, sigil_stream** out_stream);

/** Push audio; `*out_json` receives the current partial result ({"partial": "...", ...}) or NULL. */
SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_stream_push(sigil_stream* stream, const sigil_audio_view* audio, char** out_json);

/** Flush and return the final result of the session; the stream may then be pushed again (new utterance). */
SIGIL_ENGINE_API sigil_status SIGIL_ENGINE_CALL sigil_stream_finish(sigil_stream* stream, char** out_json);

/** Close a streaming session (NULL is ignored). */
SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_stream_close(sigil_stream* stream);

/** Release a string returned by this engine (NULL is ignored). */
SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_free_string(char* s);

/** Release the samples of a buffer filled by this engine and zero the struct (NULL is ignored). */
SIGIL_ENGINE_API void SIGIL_ENGINE_CALL sigil_free_audio(sigil_audio_buffer* buffer);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* SIGIL_ENGINE_H */
