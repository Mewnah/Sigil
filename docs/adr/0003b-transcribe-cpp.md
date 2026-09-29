# ADR 0003b: transcribe.cpp as an in-process sigil_engine DLL (GPU tier of STT)

- **Status:** accepted. Go for the `cuda13` and CPU variants, which were built and measured. The
  `vulkan` variant has a ready build path but has not been built or measured yet. The go for
  `cuda13` in-process holds only with the VRAM guards in "Known in-process abort paths" below;
  those paths are the recorded trigger for fallback 1.
- **Date:** 2026-09-29 (revised the same day after review: shim `0.2.4-s2`, abort logging, VRAM
  guards, one-compute-per-model enforcement, CUDA architecture pinning)
- **Workstream:** P1-G2
- **Plan decision:** "STT runtimes: sherpa-onnx vs transcribe.cpp vs own ORT", item (2), and
  "ggml isolation and whisper-rs" (docs/SPEECH_STACK_IMPLEMENTATION_PLAN.md)

## Context

The GPU tier of the STT plan re-decodes each finished utterance with a stronger model: Qwen3-ASR-1.7B
for ja/ko/zh/ru, Cohere Transcribe for English and European languages, and Voxtral Realtime and
Confucius4-R2T2 for streaming. Those GGUF models run only on transcribe.cpp (research §1.1, §1.3).
The research flagged three catches:

- The Handy app ships transcribe.cpp Vulkan-only on Windows, and the Windows CUDA build was
  undocumented. Upstream now publishes a CUDA 12 Windows bundle (option B), but no CUDA 13 one.
- The project has a single maintainer.
- The ABI broke at 0.2.0.

Sigil also has constraints of its own:

- Sigil.exe already links one static ggml (whisper-rs). A second static ggml fails at the MSVC link,
  and two DLL copies of `ggml*.dll` bind by base name.
- Nothing new may be load-time linked, and the app must start with no pack installed.
- The CUDA runtime must be CUDA 13, to match the ORT rc.13 NVIDIA pack (ADR 0001).
- Downloads are one file at a time, and packs are local-only for now (the user's decision of
  2026-09-28).

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A. `sigil_transcribe.dll`: transcribe.cpp and ggml linked **statically** into one uniquely named DLL, behind the sigil_engine ABI v1 shim, loaded lazily by `native_runtime` | One file per variant. No ggml DLL name can clash. CUDA 13 cuBLAS is shared with ORT's pack. The JSON and f32 ABI isolates Sigil from upstream's 0.x ABI churn. Hash-verified packs with variant fallback. | Sigil builds and maintains the packs. The `cuda13` DLL is 103 MB (5 GPU architectures). | Measured below. |
| B. Upstream's prebuilt `transcribe-native-*-windows` bundles (`GGML_BACKEND_DL` layout: `transcribe.dll` plus `ggml-*.dll` modules) | Nothing to build. | CUDA 12 only (`cudart64_12` / `cublas64_12`), so a second CUDA runtime family next to ORT's CUDA 13. The `ggml*.dll` base names can collide with other in-process engines. Uses upstream's C ABI directly. | v0.2.4 release assets (cu12 wheel and bundle, 200 MB). |
| C. The `transcribe-cpp` Rust crate, linked statically | Idiomatic Rust. | A second static ggml clashes with whisper-rs at link time. Its build.rs runs CMake inside Sigil's cargo build. | Plan decision "ggml isolation". |
| D. A helper process (`sigil-asr-host.exe`) hosting the same shim over pipes | Crash isolation. | An extra process, IPC per utterance, and a second CUDA context (about 400 MB). | Kept as the fallback. |
| E. Defer the GPU tier, and use Qwen3-ASR-0.6B int8 on the CPU through sherpa (P1-G1) | No new runtime. | Loses the GPU finals the research ranks best for ja/ko/zh/ru. | Kept as the last fallback. |

## Decision

**A.** transcribe.cpp **v0.2.4** is pinned: commit `4807edaf210d0d7e8a6f7fb2a44b65966a2797f0`, ggml
`353b63b`, ABI hash `7df72bf9e667b8c2`, and a sha256-pinned source tarball in
`native/engines/transcribe/UPSTREAM.json`. It is built into `sigil_transcribe.dll` packs by
`native/engines/transcribe/build.ps1`, or by P1-D1's generic `scripts/native/build-runtime-packs.ps1
-Engine transcribe`. Both go through `native/cmake` (`SigilVariant`, `sigil_add_engine`):

- **Variants:** `cpu` (SSE4.2), `cpu-avx2`, `cpu-avx512`, `cuda13` (AVX2 host code) and `vulkan`
  (AVX2 host code).
- **Build flags:** static ggml, `GGML_BACKEND_DL=OFF`, `GGML_NATIVE=OFF`, `/MD`, `/Brepro`,
  `/guard:cf`.
- **Exports:** exactly the 16 `sigil_*` functions, from Sigil's shared ABI implementation
  (`native/common/sigil_engine_impl.cpp`). `shim.cpp` implements `sigil::Engine` over `transcribe.h`.
- **Version:** the DLL reports `0.2.4-s2` (the upstream version plus the shim revision; `s1` was the
  first spike build, `s2` adds the abort callback, the active-stream guard and case-insensitive PCI
  ids). The upstream commit and, for `cuda13`, the CUDA architectures are in `build_info`.
- **Model-less instance:** `create({})` returns an instance without a model. It reports the build and
  the backends and devices this PC exposes to it, which the pack checker and a device picker use.
- **Catalog:** the packs are the entries `runtime/sigil-transcribe-<variant>`
  (`src-tauri/model-catalog/runtime-transcribe.json`). They stay `unpublished`: they are installed by
  local import or `SIGIL_LOCAL_ASSETS`, and the NVIDIA runtime pack is required for `cuda13`.

Rust uses them through `speech_rt::transcribe::TranscribeModel`:

- It picks the best installed variant.
- On a GPU failure that the engine reports as a status code it falls through to the next variant or
  to the same DLL's CPU backend, and reports why (`fallback_reason`), including when a CPU runtime
  takes over from a GPU runtime that failed. A failure that belongs to the model or the options (a
  corrupt or unsupported GGUF, a bad `n_ctx`) ends the open at once instead of reloading the file on
  every variant; the final error kind comes from the attempts (`OutOfMemory` when a GPU ran out).
- A GPU is named by index or, preferably, by its stable id (PCI bus id); an index resolves to that id
  on the first GPU backend, so a later backend with another device order never picks a different
  GPU. The CPU backend never receives a device.
- `Sigil.exe --headless engine-probe transcribe` is the measurement tool.

**Fallback ladder:**

1. **In-process instability** (crashes or hangs inside the DLL that the shim cannot contain,
   including the ggml abort paths below if they are seen in practice despite the VRAM guards, for
   example GPU driver resets during games): host the same shim in a helper process under
   `process_host` (option D). The ABI carries only JSON and f32 buffers, so it maps directly onto
   pipes, and a helper that aborts is restarted without taking Sigil down.
2. **transcribe.cpp itself fails** (a regression we cannot pin around): defer the GPU tier. Finals
   then use Qwen3-ASR-0.6B int8 on the CPU through sherpa-onnx (option E).
3. **Vulkan:** if it cannot be built or measured before P2-C, non-NVIDIA GPUs run the CPU variants,
   and the GPU tier is offered only on NVIDIA.

## Evidence

**Rig H2.** Windows 11 Pro 26200. RTX 4090 24 GB, driver 616.56. Ryzen 7 7800X3D (8 cores, 16
threads), 32 GB of RAM. The shim DLLs are release builds, so the debug `Sigil.exe` host does not
affect engine time.

Commands:
- Packs: `pwsh native/engines/transcribe/build.ps1 -Variant cpu,cpu-avx2,cpu-avx512,cuda13`.
- Measurements:
  `Sigil.exe --headless engine-probe transcribe <model> <wav> --runtime-dir <pack> -- --repeat 3..5`.
  Each measurement is the median after a 1 s warm-up. RTF is wall time divided by audio time.
- Tests: `cargo test speech_rt::transcribe -- --ignored --test-threads=1`. Since the review a missing
  model or pack fails these tests (unless `SIGIL_TEST_ALLOW_SKIP=1`), so a green run always
  exercised the real runtime.
- Audio: transcribe.cpp's samples, 16 kHz mono: `jfk` 11.0 s, `ja-short` / `ko-short` / `ru-short`
  11.0 s, `ja-long` 28.2 s.

### Speed and memory (Sigil's own measurements)

| Model | Variant / backend | Clip | Wall (median) | RTF | Process VRAM (DXGI) |
| --- | --- | --- | ---: | ---: | --- |
| Qwen3-ASR-1.7B Q8_0 (2.19 GB) | cuda13 / CUDA0 | ja-short | 205 ms | 0.0187 | +3352 MB loaded, +3444 MB after runs |
| | cuda13 / CUDA0 | ja-long 28.2 s | 497 ms | 0.0176 | +3496 MB after runs |
| | cuda13 / CUDA0 | jfk (en) | 190 ms | 0.0173 | +3444 MB |
| | cuda13 / CUDA0 | ko-short | 250 ms | 0.0228 | +3444 MB |
| | cuda13 / CUDA0 | ru-short | 280 ms | 0.0255 | +3444 MB |
| | cpu-avx512, 6 threads | ja-short | 3110 ms | 0.279 | - |
| | cpu-avx2, 8 threads | ja-short | 2735 ms | 0.249 | - |
| whisper-tiny Q8_0 (46 MB) | cuda13 / CUDA0 | jfk | 56 ms | 0.0051 | +378 MB loaded, +462 MB after runs |
| | cuda13 / CUDA0 | ja-short | 72 ms | 0.0066 | +462 MB |
| | cpu-avx2, 6 threads | jfk | 221 to 226 ms | 0.020 | - |
| | cpu-avx512, 6 threads | jfk | 212 ms | 0.019 | - |
| | cpu (SSE4.2), 6 threads | jfk | 1817 ms | 0.165 | - |
| whisper-rs tiny.en f16 (Sigil's current STT), same process | CPU, 6 threads | jfk | 335 to 349 ms | 0.031 | - |

Loading and warm-up:
- Qwen3-ASR-1.7B: 1.8 s to load the model from a warm disk cache, then a 145 ms warm-up on 1 s of
  audio (about 190 ms on the 20 s GPU warm-up of shim `s2`; see "Known in-process abort paths").
- whisper-tiny: 250 ms to load.

The final DLLs, built with `sigil_add_engine` and the shared ABI implementation, were measured again.
The numbers match the table within noise:

| Model and backend | Clip | Wall | RTF |
| --- | --- | ---: | ---: |
| Qwen3-ASR-1.7B, CUDA | ja-short | 200 ms | 0.0182 |
| Qwen3-ASR-1.7B, CUDA | ja-long | 444 ms | 0.0157 |
| Qwen3-ASR-1.7B, AVX-512 | ja-short | 2924 ms | 0.263 |
| whisper-tiny, CUDA | jfk | 57 ms | - |
| whisper-tiny, AVX2 | jfk | 195 ms | - |

VRAM was the same.

VRAM:
- Capping `n_ctx` at 4096 did not change VRAM (the KV cache grows per run).
- Unloading the Qwen3 model returns process VRAM to 378 MB, and reloading brings it back to 3444 MB
  with no growth (ignored test `real_qwen3_asr_on_cuda`; 380 / 3472 MB with shim `s2`'s 20 s GPU
  warm-up). The 378 MB left over is the CUDA context
  and cuBLAS state of the pinned `cuda13` DLL.
- Budget for Qwen3-ASR-1.7B Q8_0: about **3.5 GB for the model plus 0.4 GB of CUDA context while the
  engine is loaded** (the catalog's `min_vram_mb` for it is 3900). On paper that fits an 8 GB card
  next to the voice-changer target of 650 MB or less (T3), but every number here comes from a 24 GB
  RTX 4090 with no VRAM pressure. On a full 8 or 12 GB card (VRChat, OBS, a CUDA voice changer) the
  guards below decide: the model moves to the CPU instead of running at the edge.

Quality spot checks (the WER campaign is P7):
- Qwen3-ASR-1.7B transcribed the ja, ko, ru and en clips correctly, including punctuation. On the
  Japanese clip it wrote 通信経路 and 西洋 correctly, where whisper-tiny produced 通信系路 and 整容.
- whisper-tiny's English matches whisper-rs.

### Known in-process abort paths

Only failures transcribe.cpp returns as status codes are recoverable in-process. In the pinned
ggml, ggml-cuda handles CUDA errors it cannot return with `CUDA_CHECK` / `CU_CHECK` /
`CUBLAS_CHECK`, which call `ggml_cuda_error`, then `GGML_ABORT`, then `abort()`. That ends
Sigil.exe (captions, TTS, the OBS / Twitch integrations). The cases that matter:

- **The compute memory pool growing on a full GPU.** `ggml_cuda_pool_vmm::alloc` runs
  `CU_CHECK(cuMemCreate(...))` (ggml-cuda.cu:589); the legacy pool runs `CUDA_CHECK(err)` after one
  retry. The pool grows with the input length, so a model warmed up on 1 s of audio grows it on the
  first long utterance.
- **A lost device** after a driver reset (TDR), for example one caused by a game.
- Any other `GGML_ASSERT` inside ggml.

Weight and graph buffers go through ggml's allocators, which return failures, so a model that does
not fit is expected to fail its load with `SIGIL_E_OUT_OF_MEMORY` rather than abort (read from the
source; real VRAM exhaustion was not reproduced on the 24 GB rig).

Mitigations (shim `0.2.4-s2` and the Rust wrapper):

1. **The reason is always logged.** The shim installs `ggml_set_abort_callback` (this DLL's own
   static ggml only): the message goes synchronously through the host log callback into Sigil's
   file log (a synchronous writer) and to stderr, before `abort()`.
2. **Loads need room.** Before a GPU load, a model-less instance reads the device's free memory; the
   model needs its estimated VRAM (1.5 × the file: Qwen3-ASR-1.7B Q8_0 is about 3.1 GB) plus
   `min_free_vram_mb` (default 256 MB). Otherwise the GPU variant is refused as out of memory and
   the next variant or the CPU backend takes over, with the reason reported.
3. **The pool is sized at load time.** GPU warm-ups decode 20 s of audio (`GPU_WARMUP_SECONDS`, the
   top of P2's 15 to 20 s VAD cap, clamped to the model's input limit) instead of 1 s. On a CPU the
   warm-up stays at 1 s. Measured on H2 with the engine probe's new `after_warmup` / `run_growth`
   fields (process VRAM, MiB):

   | Model, warm-up | After load | After warm-up | After 3 runs | Growth in runs | Warm-up |
   | --- | ---: | ---: | ---: | ---: | ---: |
   | Qwen3-ASR-1.7B, 1 s, ja-short 11 s | 3352 | 3416 | 3444 | +28 | 166 ms |
   | Qwen3-ASR-1.7B, 20 s, ja-short 11 s | 3352 | 3470 | 3470 | 0 | 190 ms |
   | Qwen3-ASR-1.7B, 20 s, ja-long 28.2 s (over the cap) | 3352 | 3470 | 3496 | +26 | 185 ms |
   | whisper-tiny, 20 s, jfk 11 s | 378 | 462 | 462 | 0 | 100 ms |

   Utterances within the warm-up length no longer grow the pool; the extra warm-up costs about 25 ms.
4. **Every GPU run checks the headroom first.** Before each `transcribe` (and after the warm-up and
   before a stream opens) the wrapper reads the device's free memory; below `min_free_vram_mb` the
   model moves to the CPU (`cpu_fallback`, the default) or the call fails with `OutOfMemory`,
   instead of starting a run that could abort.

Residual risk, accepted for P2: a driver reset still aborts the process, and decoder temporaries of
very long transcripts can still grow the pool slightly past the warm-up. If either shows up in
practice (a `sigil::engine` log line containing "fatal ggml error"), fallback 1 applies. P2 must
keep utterances at or below the VAD cap and hold the VRAM claim in the `resources` ledger.

### Build and packaging

- **Toolchain:** VS 2022 Build Tools 17.14 (MSVC 19.44.35224), CMake 4.2.3, the VS Ninja, CUDA
  13.2.51. The CUDA architectures are `75-real;80-virtual;86-real;89-real;120a-real`: Turing,
  Ampere, Ada and Blackwell as native code, plus PTX for anything else from sm_80 up. They come from
  `UPSTREAM.json` on every configure. A development list (`-CudaArchitectures`) needs `-Dev` and
  produces a `-dev` zip, and `build.ps1` reads the list back from the DLL's `build_info` before it
  stages a pack. The spike's first build kept a cached list across builds, so a quick sm_89 build
  could have been staged later as the full pack and aborted on other GPUs.
- **Build time on H2 with 16 jobs:** about 16 s for each clean CPU variant, 832 s for `cuda13`. The
  upstream source needed no patches.
- **Sizes:**

  | Variant | DLL | Zip |
  | --- | ---: | ---: |
  | `cpu` | 3.1 MB | 1.38 MB |
  | `cpu-avx2` | 3.2 MB | 1.43 MB |
  | `cpu-avx512` | 3.3 MB | 1.45 MB |
  | `cuda13` | 102.8 MB | 92.1 MB |

- **Imports:**
  - The CPU variants import only the system, UCRT and VC++ runtime DLLs. Sigil.exe already imports
    `MSVCP140`, `VCRUNTIME140` and `VCRUNTIME140_1`, so the packs add no new dependency.
  - `cuda13` additionally imports **only `cublas64_13.dll`**, which loads `cublasLt64_13.dll`. nvcc
    links cudart statically, and cudart loads the driver at run time. **cuDNN is not needed.**
- **Exports:** exactly the 16 ABI functions. `build.ps1` fails on any other export and on any
  `ggml*.dll` import. At run time, `GetModuleHandle` finds no `ggml*.dll` module in the process.
- **Reproducibility:** clean rebuilds produce byte-identical DLLs and zips. This was checked twice:
  - Before moving onto `sigil_add_engine`: `cpu` DLL sha256 `0d57519b…`, zip `b7dd79d3…`; `cpu-avx2`
    DLL `39e78004…`, zip `3864746f…`.
  - After, with the shared ABI implementation: `cpu` DLL `c9b98c54…`, zip `5df27fe9…`.
  - The `s2` packs were rebuilt incrementally from the same trees (shim, ABI implementation and
    resource objects only; the `cuda13` kernels kept the pinned architectures, which `build.ps1`
    now reads back from `build_info`); all four self-tests pass.

  The source tarball is sha256-pinned, and configure refuses a tree whose ABI hash differs.
- **Generic script:** `scripts/native/build-runtime-packs.ps1 -Engine transcribe -Variant cpu` builds
  the same engine with the default Visual Studio 2026 generator (MSVC 19.50). The checker passes
  (`create({})`), and the catalog entry matches.
- **Model store round trip:**
  - `Sigil.exe --headless fetch runtime/sigil-transcribe-cpu-avx2 runtime/sigil-transcribe-cpu-avx512`
    with `SIGIL_LOCAL_ASSETS` pointing at the packs folder installed both packs into
    `runtimes/sigil-transcribe-<variant>/0.2.4-s1/<variant>/`.
  - `engine-probe transcribe` without `--runtime-dir` then chose `cpu-avx512` through
    `native_runtime::open_engine`.
  - When one byte of the `cpu-avx512` pack was flipped (same size), its `sigil-pack.json` hash check
    rejected it, and the probe fell through to `cpu-avx2` with the reason recorded. A size change
    makes the store treat the pack as not installed at all.

### Coexistence in one process

- **With whisper-rs:** Sigil.exe links whisper-rs's static ggml, and the engine DLL carries its own
  static ggml. Loading the engine and interleaving whisper-rs and transcribe runs left whisper-rs's
  output byte-identical. This was checked both in the Sigil.exe probe (`--whisper-rs`) and in the
  test binary (`coexists_with_whisper_rs_in_one_process`), and whisper-rs still worked after the
  engine model was dropped.
- **With Sigil's ONNX Runtime:** the DirectML canary passed before the CUDA engine loaded and again
  after its runs (`--ort-canary dml`: 790 ms and 301 ms).
- **ORT's CUDA EP next to ggml-cuda was not verified.** This PC has no cuDNN, and the 1.9 GB NVIDIA
  pack was not downloaded. Both share `cublas64_13` and carry their own static cudart. Re-run
  `--ort-canary cuda` once the pack is installed.
- **A partial cuDNN kills the process.** A partial cuDNN on the DLL search path (DaVinci Resolve's
  9.8 without `cudnn_graph64_9.dll`, added through the dev variable) made cuDNN terminate the process
  with exit code 127 inside ORT's CUDA provider. This is an ORT/cuDNN hazard for ADR 0003 and 0007,
  not a transcribe one.

### Cancellation

- **Whisper models:** `sigil_engine_cancel` interrupts between decode steps; an encoder pass that
  is already running finishes first.
  - On CUDA, a cancel 20 ms into a whisper-tiny run returned `SIGIL_E_CANCELLED` 1.9 ms later.
  - On the CPU, with a 44 s clip, a cancel fired at 300 ms returned 1 to 164 ms later, depending on
    whether an encoder pass was running.
  - The next call works normally (ignored test `cancel_interrupts_a_running_decode`, which now runs
    one thread on the 44 s clip, measures the uncancelled decode first and fails unless the
    cancelled run returns `Cancelled` well before it).
- **Qwen3-ASR 0.2.4:** `run()` checks the abort flag **only before it starts** (upstream comment:
  "single-shot"), so a cancel 50 ms into a 206 ms run did not interrupt it. The batched decode path
  polls on every step.

## Consequences

- **What was added:**
  - `native/engines/transcribe/`: CMake, shim, self-test, build script and pin.
  - `speech_rt::transcribe`: the wrapper, the engine probe and tests.
  - The `runtime/sigil-transcribe-*` and `bench/transcribe-*` catalog entries.

  Nothing is load-time linked. With no pack installed, `TranscribeModel::open` returns
  `NotInstalled` and nothing else changes.
- **P2 (speech plugin, GPU tier):**
  - Use `TranscribeModel` from a worker thread, one model per concurrent decode. transcribe.cpp 0.x
    runs one compute per model at a time, and an open stream counts from its first push until
    `finish`: while one is mid-utterance, `transcribe()` and other streams of that model fail with
    `InvalidInput` (enforced in the shim). Two live caption sources (mic and loopback) need two
    loaded models.
  - Keep utterances bounded; the VAD cap is 15 to 20 s (the GPU warm-up is sized to 20 s). Rely on
    epoch discipline, not on cancel, to drop stale Qwen3 results.
  - Claim about 3.5 GB of VRAM for Qwen3-ASR-1.7B Q8_0 in the `resources` ledger, plus 0.4 GB while
    any `cuda13` engine is loaded (catalog `min_vram_mb` 3900).
  - Show `fallback_reason()` when it is set: it now also covers a GPU without VRAM headroom and a
    CPU runtime taking over from a failed GPU runtime. A CPU fallback closes the model's open
    streams (their next call fails with `Backend`; reopen them).
  - Store the chosen GPU as its stable id (`ModelOptions::device_id`, the PCI bus id from
    `{"op":"devices"}`), not an index.
  - Add the user-facing STT model entries (`stt/...`). The `bench/` entries here are measurement
    assets only.
- **Follow-ups:**
  - P2 or upstream: make Qwen3-ASR decode abortable, either by calling `transcribe_run_batch` with
    n=1 (it polls per decode step) or by upstreaming an abort poll in `run()`.
  - P1-D1: `cuda13` transcribe needs only cuBLAS (about 505 MB of DLLs), not the cuDNN that makes up
    most of the 1.9 GB NVIDIA pack. A cuBLAS-only subset would make GPU STT much cheaper to install
    for users who don't run the voice changer on CUDA.
  - Vulkan: build and measure on a machine with the Vulkan SDK (`build.ps1 -Variant vulkan`). There
    is no CI workflow, since packs are local-only by the user's decision. Upstream builds Vulkan on
    Windows in its own CI, so the path is known to compile with MSVC.
  - `scripts/native/build-runtime-packs.ps1 -Engine transcribe` should delegate to
    `native/engines/transcribe/build.ps1`.
  - P7: decide whether whisper-rs retires. On the same clip, transcribe's whisper-tiny Q8_0 on AVX2
    was about 1.5× faster than whisper-rs tiny.en f16 (221 ms vs 335 ms), and about 6× faster on
    CUDA.
- **What is harder now:**
  - Sigil owns the builds: each upgrade re-pins, rebuilds five variants, and re-runs the probe.
  - The `cuda13` build takes about 14 minutes.
  - transcribe.cpp is pre-1.0, so the CMake ABI-hash gate and a `shim_revision` bump are required
    whenever `transcribe.h` changes.
