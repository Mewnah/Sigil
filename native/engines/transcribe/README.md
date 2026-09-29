# sigil_transcribe.dll: transcribe.cpp runtime packs

[transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT) runs GGUF speech-to-text models
(Qwen3-ASR, Cohere Transcribe, Voxtral, Parakeet, Nemotron, SenseVoice, Whisper and more) on ggml.
Sigil builds it into `sigil_transcribe.dll`: transcribe.cpp and its vendored ggml linked statically,
behind the sigil_engine ABI v1 (`native/common/sigil_engine.h`). The 16 exports come from Sigil's
shared implementation in `native/common/sigil_engine_impl.cpp`, which `sigil_add_engine` compiles in;
`shim.cpp` implements `sigil::Engine` on top of `transcribe.h`. The decision and the measurements are
in [ADR 0003b](../../../docs/adr/0003b-transcribe-cpp.md). The Rust side is
`src-tauri/src/services/speech_rt/transcribe/`, and the catalog entries are in
`src-tauri/model-catalog/runtime-transcribe.json`.

| File | What it is |
| --- | --- |
| `UPSTREAM.json` | Pinned upstream: tag, commit, archive URL / size / sha256, ABI hash, CUDA architectures, shim revision. |
| `engine.json` | Engine name, version (`<upstream>-s<shim revision>`), variants, extra pack files; read by both build scripts. |
| `CMakeLists.txt` | One variant per build tree through `native/cmake` (`SigilVariant`, `sigil_add_engine`): static transcribe.cpp + ggml, `/MD`, `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=OFF`. Works inside `native/` and standalone. |
| `shim.cpp` | `sigil::Engine` over `transcribe.h` (JSON contract documented at its top; invalid UTF-8 in model output becomes U+FFFD). `create({})` gives a model-less instance that reports the build and its devices. |
| `selftest.cpp` | `sigil_transcribe_selftest.exe`: loads the DLL like Sigil does, checks the ABI contract, dumps the DLL's info for `sigil-pack.json`, optionally transcribes a WAV. Not shipped. |
| `build.ps1` | Fetch + verify the pinned source, build, check exports/imports, stage packs, zip, self-test. |
| `NOTICE.txt` | Shipped in every pack next to transcribe.cpp's `LICENSE` and `THIRD-PARTY-LICENSES.md`. |

## Variants

| Variant | CPU floor | GPU | Imports beyond the system / VC++ runtime |
| --- | --- | --- | --- |
| `cpu` | SSE4.2 | - | none |
| `cpu-avx2` | AVX2 + FMA + F16C + BMI2 | - | none |
| `cpu-avx512` | AVX-512 F/CD/BW/DQ/VL | - | none |
| `cuda13` | AVX2 | CUDA 13 (sm_75 real, sm_80 PTX, sm_86/89/120a real: `UPSTREAM.json`), driver R580+ | `cublas64_13.dll` (→ `cublasLt64_13.dll`) from `runtime/nvidia-cuda13`; cudart is static |
| `vulkan` | AVX2 | Vulkan 1.2+ | `vulkan-1.dll` (every GPU driver ships it) |

`native_runtime` picks the best installed variant this PC supports (`cuda13`, `vulkan`, `cpu-avx512`,
`cpu-avx2`, `cpu`), verifies every file against `sigil-pack.json`, and falls through to the next variant
when a DLL or its GPU backend fails.

## Building

Needs Visual Studio 2022 Build Tools with the C++ workload (MSVC v143; CUDA 13's host compiler),
CMake 3.25+, PowerShell 7, and for `cuda13` the CUDA 13 toolkit. `vulkan` also needs the Vulkan SDK
(`VULKAN_SDK`). The script fetches the pinned source once (about 44 MB, one request) into
`%LOCALAPPDATA%\sigil-dev-cache\transcribe.cpp\` and refuses it unless the size and sha256 match.

```powershell
pwsh native/engines/transcribe/build.ps1 -Variant cpu,cpu-avx2,cpu-avx512,cuda13 -SelfTest
# quicker CUDA build for one GPU generation while developing (a -dev pack, never the shippable one):
pwsh native/engines/transcribe/build.ps1 -Variant cuda13 -CudaArchitectures 89-real -Dev
# a real transcription as part of the self-test:
pwsh native/engines/transcribe/build.ps1 -Variant cpu-avx2 -SelfTest -Model <model.gguf> -Wav <16 kHz wav>
# only fetch + verify the pinned source (for the generic script below):
pwsh native/engines/transcribe/build.ps1 -FetchOnly
```

The generic `pwsh scripts/native/build-runtime-packs.ps1 -Engine transcribe -Variant <v>` (P1-D1) delegates
to this `build.ps1` (P1-I0), writing the packs to its `native/out/local-assets/`. Configuring
`native/CMakeLists.txt` directly with `-DSIGIL_ENGINES=transcribe` still builds the same DLL once the
pinned source is in the dev cache (or with `-DSIGIL_TRANSCRIBE_FETCH=ON`), but only `build.ps1` ships
the licenses in the pack, checks the imports and exports, supplies the CUDA DLL folders to the checks
and runs the self-test.

Build trees go to `%LOCALAPPDATA%\sigil-native\transcribe\<variant>` (or `SIGIL_NATIVE_BUILD_ROOT`),
outside the repository and short enough for ggml's Vulkan shader generator. Packs go to
`%LOCALAPPDATA%\sigil-native\packs\` (or `-OutDir`):

- `transcribe-<variant>\`: `sigil_transcribe.dll`, `sigil-pack.json`, the licenses and the notice;
- `sigil-transcribe-<version>-<variant>.zip` (+ `.sha256`): what the catalog entry
  `runtime/sigil-transcribe-<variant>` installs.

Clean builds are byte-reproducible (`/Brepro`, fixed zip timestamps and order). Measured on a Ryzen 7
7800X3D: the CPU variants build in about 16 s each, `cuda13` with all five architectures in about 14 min.

The CUDA architectures come from `UPSTREAM.json` on every configure (`build.ps1` always passes the
effective list, so a build tree reused after a `-CudaArchitectures` build returns to the pin). A
list that differs from the pin needs `-Dev`: that pack's zip is `sigil-transcribe-<version>-cuda13-dev.zip`
and its `sigil-pack.json` says `"dev_build": true`, so it is never installed as the catalog pack.
`build.ps1` also reads the architectures back from the DLL's build info and refuses to stage a
pack whose kernels differ from what was asked.

## Installing a local pack

Packs are not published (catalog status `unpublished`). Install one with the Models & Runtimes tab's
local import (pick the zip), or let the model store find it:

```powershell
$env:SIGIL_LOCAL_ASSETS = "$env:LOCALAPPDATA\sigil-native\packs"
Sigil.exe --headless fetch runtime/sigil-transcribe-cpu-avx2
```

`runtime/sigil-transcribe-cuda13` requires `runtime/nvidia-cuda13` (installed with it).

## Checking a model

```powershell
Sigil.exe --headless engine-probe transcribe <model.gguf> <audio.wav> [--variant cuda13] [--runtime-dir <pack dir>] `
    -- --repeat 5 [--language ja] [--whisper-rs <ggml .bin>] [--cancel-after-ms 50] [--ort-canary dml] `
       [--device-id <pci id>] [--warmup-seconds 20] [--min-free-vram-mb 256]
```

The report has the runtime chosen (and why others were skipped), the backend actually used, load and
warm-up time, per-run RTF, this process's VRAM (DXGI) at start, right before the engine (after the ORT
canary and whisper-rs, so `load_delta` / `total_delta` are the engine's own) and after, and the
transcript. With a runtime folder and no NVIDIA pack installed, `SIGIL_TRANSCRIBE_CUDA_BIN`
(`;`-separated) may point at a CUDA 13 toolkit `bin` for development.

## Fatal errors, VRAM headroom and one compute per model

- ggml ends the process on errors it cannot return: `GGML_ABORT` / `GGML_ASSERT`, including
  ggml-cuda's `CUDA_CHECK` when its memory pool cannot grow on a full GPU or the device is lost to a
  driver reset. The shim installs ggml's abort callback, so the message reaches Sigil's log before
  `abort()`. The Rust wrapper keeps runs away from those paths (VRAM checks before loading and before
  every GPU run, GPU warm-up at the longest utterance); see ADR 0003b.
- transcribe.cpp 0.x runs one compute per model: a stream counts from its first push until `finish`.
  While one is mid-utterance, `process`, the warm-up and other streams' first push of that model fail
  with `SIGIL_E_INVALID_ARGUMENT`. Load a second model for a second caption source.

Rust tests: `cargo test --manifest-path src-tauri/Cargo.toml speech_rt::transcribe`, plus the ignored
real-model tests (`-- --ignored --test-threads=1`) described at the top of
`speech_rt/transcribe/tests.rs`. A missing model or pack fails those tests unless
`SIGIL_TEST_ALLOW_SKIP=1` is set.

## Upgrading transcribe.cpp

1. Update `UPSTREAM.json` (tag, commit, version, archive size and sha256, `abihash`,
   `ggml_commit`). The configure step refuses a tree whose `transcribe.abihash` differs from the pin.
2. Read the `transcribe.h` changes. transcribe.cpp is pre-1.0, so the ABI may break on minor releases.
   Adapt `shim.cpp`, and bump `shim_revision` whenever the shim changes.
3. Rebuild every variant with `-SelfTest`, then run the ignored Rust tests and the engine probe on
   whisper-tiny and Qwen3-ASR-1.7B.
4. Bump `version` and the zip names of every `runtime/sigil-transcribe-*` entry in
   `runtime-transcribe.json` together.
