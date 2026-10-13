# Qwen3-TTS server packs (`runtime/qwentts-server-*`)

`sigil-tts-server.exe` runs Qwen3-TTS 12 Hz (0.6B / 1.7B, GGUF) for Sigil's `qwen3` text-to-speech
backend. It is [qwentts.cpp](https://github.com/ServeurpersoCom/qwentts.cpp) (MIT, the GGML port of
Qwen3-TTS) with Sigil's server program on top, run by Sigil as the `tts-qwen3` helper process
(127.0.0.1 only, a random port and API key, a kill-on-close Job Object). Engine code:
`src-tauri/src/services/tts_helpers/qwen.rs` and `qwen/`; catalog: `src-tauri/model-catalog/runtime-qwentts.json`
(packs) and `tts-qwen3.json` (models).

| Path | What |
| --- | --- |
| `UPSTREAM.json` | Pinned sources: qwentts.cpp `6fae929` and its ggml fork submodule `40e16e4` (archive sha256 and size), the CUDA architectures, the shim revision and catalog version |
| `server/sigil-tts-server.cpp` | Sigil's server: upstream's `tools/tts-server.cpp` (attribution kept) plus a Bearer key from `QWENTTS_API_KEY` on every route but `/health`, no wildcard CORS, a loopback-only bind (`--allow-remote` to override), exclusive port use, the extracted voice latents (`spk_b64`, `rvq_b64`) in the `POST /v1/audio/voices` answer, `GET /sigil/info`, unknown languages spoken as `auto`, and instructions dropped for Base models |
| `server/CMakeLists.append.cmake` | The `sigil-tts-server` target, appended to upstream's `CMakeLists.txt` in the prepared source tree |
| `build.ps1` | Fetch the pinned archives (once), prepare the tree, build with MSVC + Ninja, check imports, stage and zip the pack, and measure it |
| `NOTICE.txt` | Attribution shipped in every pack (with the generated `THIRD-PARTY-LICENSES.txt`) |

## Build

Requires Visual Studio 2022 Build Tools with the C++ workload ("C++ CMake tools" provides Ninja),
PowerShell 7, the CUDA 13 toolkit for `cuda13` and the Vulkan SDK (`VULKAN_SDK`) for `vulkan`.

```powershell
# Once: download the two pinned archives (about 5 MB, one request each).
pwsh scripts/runtime-packs/qwentts/build.ps1 -FetchOnly

# The packs: %LOCALAPPDATA%\sigil-native\packs\qwentts-server-<variant>\ and
#            ...\sigil-qwentts-server-6fae929-s1-<variant>.zip (+ .zip.sha256)
pwsh scripts/runtime-packs/qwentts/build.ps1 -Variant cuda13,cpu
pwsh scripts/runtime-packs/qwentts/build.ps1 -Variant vulkan        # on a PC with the Vulkan SDK

# A faster cuda13 build for one GPU (its zip is named ...-cuda13-dev.zip; not for friends' PCs).
pwsh scripts/runtime-packs/qwentts/build.ps1 -Variant cuda13 -CudaArchitectures 89-real -Dev
```

ggml is linked statically (no backend DLLs), with `GGML_NATIVE=OFF` and the AVX2 baseline. The exe
imports only Windows and MSVC runtime DLLs plus `cublas64_13.dll` (cuda13: Sigil puts
`runtime/nvidia-cuda13` on the helper's PATH) or `vulkan-1.dll` (vulkan: from the GPU driver); the build
refuses anything else. Measured on the dev PC (16 cores, 6 jobs): cuda13 for one architecture about
6 minutes (32.7 MB exe), cpu under a minute (2 MB); the pinned five-architecture cuda13 build 12 minutes
(54 build steps, 101.6 MB exe, a 91.6 MB zip; an earlier estimate of 1.5 to 2 hours was wrong), still best
started when the PC is otherwise idle. A build that is stopped resumes where it stopped.

The source tree is prepared from the pinned archives (no git), so `tools/version.cmake` is replaced
with one that stamps the pinned commit: `--version` reports `library 6fae929 (2026-09-28)`, and the
build refuses an exe that reports another library version. A change to the preparation bumps
`$PrepRevision` in `build.ps1`, which prepares the tree again (tar keeps the archive's file times,
so only what really changed recompiles).

Install a zip in Sigil: open **Models & Runtimes**, find **Qwen3-TTS server (CUDA 13 / Vulkan /
processor)** and choose **Import**, or put the zip in a folder named by `SIGIL_LOCAL_ASSETS`. Sigil
checks the pack's `sigil-pack.json` (engine, variant, the exe's size and sha256) before every start.

## Models

The talkers and the tokenizer are published Q8_0 GGUFs (`tts-qwen3.json`, pinned to
`Serveurperso/Qwen3-TTS-GGUF@b7ee2e8`), downloaded from the Qwen3-TTS panel after consent:

| Mode | 1.7B | 0.6B |
| --- | --- | --- |
| Built-in voices (CustomVoice; style instructions on 1.7B only) | `tts/qwen3-tts-1.7b` (2.0 GB) | `tts/qwen3-tts-0.6b` (0.97 GB) |
| Voice cloning (Base) | `tts/qwen3-tts-1.7b-base` (2.1 GB) | `tts/qwen3-tts-0.6b-base` (0.99 GB) |
| Voice design (VoiceDesign) | `tts/qwen3-tts-1.7b-voicedesign` (2.0 GB) | none |

Every talker requires `tts/qwen3-tts-tokenizer` (0.29 GB).

## Measure (P6-E verification: time to first audio, RTF, cloning)

With models installed (or any folder holding the GGUFs):

```powershell
# Forward slashes work in PowerShell paths too (the model folders are <id rest>/<version>/).
$m = "$env:APPDATA/com.sigil.app/models/tts"
pwsh scripts/runtime-packs/qwentts/build.ps1 -Variant cuda13 -CudaArchitectures 89-real -Dev -SelfTest `
    -Talker "$m/qwen3-tts-1.7b-base/b7ee2e8/qwen-talker-1.7b-base-Q8_0.gguf" `
    -Codec  "$m/qwen3-tts-tokenizer/b7ee2e8/qwen-tokenizer-12hz-Q8_0.gguf" `
    -RefWav "$env:APPDATA\com.sigil.app\voices\<voice id>\ref.wav" -RefText "<its transcript>" `
    -GpuLock "<scratchpad>\gpu_timing.lock"
```

The self-test starts the server with a random key on a free loopback port, checks that a request
without the key is refused, reads `/sigil/info`, and with a Base talker registers the reference clip
(upstream's `examples/freeman.wav` without `-RefWav`), then registers the returned latents under a
second name (Sigil's cached path) and speaks with them. Every `-Text` line is timed: first audio (ms),
audio seconds, total time and RTF, in `...\packs\qwentts-selftest\report.json`, with the WAVs and the
server log next to it. `-GpuLock` waits for, takes and releases the shared GPU timing lock so another
benchmark is never disturbed. Preset and design talkers take `-Voice` / `-Instruct`. `-SkipBuild`
tests the packs already staged in `...\packs\qwentts-server-<variant>\` without building (the cuda13
dir is shared by the pinned and the `-Dev` builds, so switching between them recompiles CUDA).

Sigil's own path (process host, `/sigil/info` check, warm-up, clone registry and its speaker-data
cache, VRAM claim) runs against the real server in an ignored Rust test:

```powershell
$env:SIGIL_QWENTTS_EXE = "$env:LOCALAPPDATA\sigil-native\packs\qwentts-server-cuda13\sigil-tts-server.exe"
$env:SIGIL_TEST_MODELS_DIR = "<folder with qwen-talker-0.6b-base-Q8_0.gguf (or 1.7b) and qwen-tokenizer-12hz-Q8_0.gguf>"
cargo test --manifest-path src-tauri/Cargo.toml real_model_clone_end_to_end -- --ignored --nocapture
```

In the app: choose **Qwen3-TTS** in Text to speech, mode **Clone a voice from your library**, pick a
recorded voice (with its transcript), and **Test**. Sigil's log (`sigil::tts::qwen3`) shows the plan,
`ready in … ms`, the warm-up's first audio and RTF, the voice registration (from the clip, then from
the cached speaker data after a restart) and the measured video memory. Every sentence adds the
server's own timing to the helper's log (`tts-qwen3` in `sigil.log`): `[Perf] TTFA … ms` (time to
the first audio frame) and `[Perf] Total … ms (… audio … s, RTF …)`. The panel shows each sentence's
time to first sound as heard (including the request and the sink).
