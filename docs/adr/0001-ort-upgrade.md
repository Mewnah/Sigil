# ADR 0001: ONNX Runtime 1.28 (ort 2.0.0-rc.13, CUDA 13) with KittenTTS vendored

- **Status:** accepted (P1 gate passed on the dev rig; CUDA pack install and a packaged-build smoke remain manual, see Consequences)
- **Date:** 2026-09-28
- **Workstream:** P1-S0a
- **Plan decision:** "ONNX Runtime version and the kittentts pin", "CUDA 13 hardware floor",
  "Delivering GPU execution providers" (P1 part: `register_ep_library`)

## Context

Sigil linked ONNX Runtime 1.23.2 statically through `ort` 2.0.0-rc.11 (pyke's `cu12` dist). The
`kittentts` 0.3.0 crate (neural-onnx-tts backend) pinned `ort = "=2.0.0-rc.11"` and pulled `hf-hub`
(blocking `ureq`, a cache under `~/.cache/huggingface`), so no ORT upgrade was possible (research
§1.3: it also blocked parakeet-rs and ORT alignment with sherpa-onnx 1.13.8, which bundles ORT 1.28).
The voice-engine plan needs rc.13 APIs: `with_cuda_graph`, `with_compute_stream`, IOBinding,
`with_external_initializer_file_in_memory`, `with_dimension_override`, `ep::directml::api()`
(OrtDmlApi, for the DirectML high-priority queue, research §4.3#6) and `register_ep_library`
(plugin-EP registration, so the CUDA provider can live in an app-data pack instead of the installer).
The user approved the one-time pyke dist download and the CUDA 13 floor on 2026-09-28.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A. Vendor kittentts (Apache-2.0) into `local_tts/kitten/`, then ort `=2.0.0-rc.13` (ORT 1.28.0) with `cuda, directml, nvrtx, tensorrt, half, preload-dylibs, copy-dylibs, ndarray` | Zero behaviour change for Kitten; every rc.13 API above; ORT matches sherpa-onnx 1.13.8; CUDA provider shrinks 144 MB → 62 MB | CUDA 13 floor (driver R580+, Turing or newer); the Windows dist exists only as `cuda13` | Chosen; measurements below |
| B. rc.12 with `ORT_CUDA_VERSION=12` | Keeps CUDA 12 (older drivers) | No plugin-EP work beyond rc.12, second migration later | Fallback rung 1 |
| C. Stay on rc.11, kittentts vendored | No API churn | No OrtDmlApi (DirectML HP queue), no TensorRT-RTX, sherpa/parakeet misaligned | Fallback rung 2 |
| D. Replace Kitten with an in-house runner instead of vendoring | — | Behaviour change inside a gate that must be zero-change | Rejected |

## Decision

1. **KittenTTS is vendored** from kittentts 0.3.0 into `src-tauri/src/services/local_tts/kitten/`
   (model, npz, phonemize, preprocess, tokenize; SPDX/attribution headers, `LICENSE`, `NOTICE`).
   Sigil depends on `espeak-ng 0.1` (`bundled-data`) directly; `hf-hub` and `kittentts` are gone from
   `Cargo.lock`. Sessions come from `neural_audio::ort_session::build_session` (explicit CPU EP,
   spinning off). `hub.rs` replaces upstream's `download.rs`: an existing hf-hub cache snapshot is
   reused in place, otherwise `config.json`, the model and the voices file are fetched one at a time
   with `http_client::http_download_get` into `<app_data>/models/kitten/<org>--<name>/` (`.partial` +
   rename, HTML/truncation/NPZ checks, file names from `config.json` validated as plain names). The
   model store (P1-A) takes over these downloads and can adopt that folder.
2. **ort `=2.0.0-rc.13`** (ONNX Runtime **1.28.0**), features `cuda, directml, nvrtx, tensorrt, half,
   preload-dylibs, copy-dylibs, ndarray` (default API level `api-27`; nothing in rc.13 is gated on
   `api-28`). ort-sys links pyke's `x86_64-pc-windows-msvc+cuda13,tensorrt,nvrtx,directml` dist
   (sha256 `efb686cb…78293`). Call-site migration was small: `ort::tensor::TensorElementType` →
   `ort::value::TensorElementType`, `SessionBuilder::commit_from_*` now take `&mut self`, and EP
   types are feature-gated (`ort::ep::CUDA` needs the `cuda` feature).
3. **The CUDA provider is CUDA 13.** It load-time imports only `cublas64_13.dll` and
   `cublasLt64_13.dll` (cudart is static) and loads `cudnn64_9.dll`, `cufft64_12.dll` and nvrtc at run
   time. The interim NVIDIA runtime pack (`diagnostics/nvidia_pack.rs`) is re-pinned to CUDA 13.2.0
   (cudart 13.2.51, cuBLAS 13.3.0.5, cuFFT 12.2.0.37) + cuDNN 9.26.0.51 `cuda13`, family
   `nvidia-cu13`, label `cuda-13.2.0_cudnn-9.26.0` (1,885,987,422 bytes). All user-facing CUDA
   messages say CUDA 13 and name the R580 driver floor.
4. **Stale provider guard.** ort-sys's `copy-dylibs` never overwrites an existing non-symlink file,
   and ORT's provider bridge has no ABI check, so a 1.23 provider left next to a 1.28 exe could crash
   the process. `ort_session::provider_version_mismatch` compares the provider's PE file version
   (read through `version.dll`, loaded lazily from System32) with `ort_session::runtime_version()`;
   the CUDA rung is skipped with a "reinstall Sigil" reason, and the EP probe lists the note.
5. **`register_ep_library` works for the CUDA provider from an app-data folder** (evidence below).
   Sessions then reach it through its EP device: `Environment::current()?.register_ep_library("CUDA",
   <folder>/onnxruntime_providers_cuda.dll)`, then `SessionBuilder::with_devices(devices where ep() ==
   "CUDAExecutionProvider", options)`. The legacy `with_execution_providers([CUDA])` path keeps loading
   the provider from `Sigil.exe`'s folder, so P1-D1 must switch the CUDA rung to the device path when
   it moves the provider out of the installer. `onnxruntime_providers_shared.dll` (10 KB) stays next
   to the exe.
6. **Fallback ladder** (if a later gate regresses): rc.12 with `ORT_CUDA_VERSION=12` (CUDA 12/13), then
   rc.11 with kittentts still vendored (keeping CUDA graphs, IOBinding and the CUDA priority stream,
   deferring the DirectML high-priority queue and TensorRT-RTX). Rollback source:
   `scratchpad/checkpoints/P1/P1-S0a-{tracked.diff,untracked.zip}` plus `Cargo.{toml,lock}.pre`.

## Evidence

Dev rig H2: RTX 4090 (driver 616.56, CUDA driver API 13.4) + AMD Radeon iGPU, Windows 11 26200,
CUDA 13.2 toolkit (`bin\x64` on PATH), no cuDNN except DaVinci Resolve's cuDNN 9.8 (a `cuda12` build,
used only for the manual smoke).

| Check | Result |
| --- | --- |
| pyke dist download (one file) | 62,083,515 bytes `.tar.lzma2`; `onnxruntime.lib` 363.9 MB, `onnxruntime_providers_cuda.dll` 62.3 MB (was 143.8 MB), `onnxruntime_providers_tensorrt.dll` 0.7 MB, new `onnxruntime_providers_nv_tensorrt_rtx.dll` 0.7 MB, `DirectML.dll` 1.15.4 unchanged |
| `ort_session::runtime_version()` | `1.28.0` (test `linked_runtime_is_onnx_runtime_1_28`) |
| OrtDmlApi (research §4.3#6 prerequisite) | `ort::ep::directml::api()` returns the API, including `SessionOptionsAppendExecutionProvider_DML1(device, queue)` for a Sigil-owned high-priority D3D12 queue (test `directml_api_for_a_custom_command_queue_is_exposed`) |
| `cargo test` (full crate) | 367 passed, 0 failed, 5 ignored (3 new GPU-only manual tests + 2 pre-existing) |
| `cargo clippy --all-targets` | 0 warnings (vendored Kitten code has a documented lint allow-list) |
| `pnpm typecheck` / `pnpm test:unit` / `pnpm i18n:check` | green / 39 files, 502 tests / OK |
| `Sigil.exe` load-time imports | debug 41, release (LTO, `cargo build --release` 5 m 56 s) 43, all matching `scripts/release/expected-imports.txt`; nothing new. Release `target/release` providers are the fresh 1.28.0 copies |
| Silero VAD (CPU) | all `speech_core` tests pass on 1.28, including the embedded-model I/O contract |
| Kitten (CPU) | offline fake speech package through the full pipeline (espeak-ng, tokenizer, NPZ, ORT session, chunking, tail trim, speed priors, aliases) and through the `neural-onnx-tts` plugin state to WAV: pass |
| VC path, CPU | fake RVC slot (Conv content encoder + f0 generator, dio pitch) through `RvcModel::load/convert`: 4,000 samples @ 40 kHz, F0-following output |
| VC path, DirectML | same slot on `dml`: EP outcome `granted dml`; deep probe canary DirectML OK on the 4090 (366 ms) and the iGPU (446 ms) |
| VC path, CUDA 13 (manual) | `SIGIL_TEST_CUDA_BIN=<CUDA 13.2>\bin\x64 SIGIL_TEST_CUDNN_BIN=<cuDNN 9 folder> cargo test -- --ignored cuda13_runtime_smoke fake_rvc_slot_converts_on_cuda13`: canary on CUDA in 378–978 ms (first session in the process, includes CUDA context creation; the slower run shared the GPU with a concurrent test), RVC slot `granted cuda` for `vc` and `vc.content` |
| CUDA without cuDNN | reason `cudnn64_9.dll (CUDA 13 / cuDNN 9 runtime) was not found; install the NVIDIA GPU runtime pack` (ORT 1.28 wording `LoadLibrary failed for cudnn64_9.dll` is recognised), falls back to DirectML |
| `register_ep_library` from an app-data stand-in folder (manual) | `cargo test -- --ignored cuda_provider_registers_from_an_app_data_folder`: registered in 62 ms; `GetModuleFileName` shows the provider loaded from the registered folder, not the exe folder; `env.devices()` lists `CPUExecutionProvider`, `DmlExecutionProvider` (4090, iGPU) and `CUDAExecutionProvider` (4090); a `with_devices` session runs the Conv canary correctly; `unregister` succeeds |
| `Sigil.exe --headless ep-probe` | deep report: CUDA canary fails only on cuDNN (expected without the pack), DirectML OK on both GPUs, recommends `dml`, pack label `cuda-13.2.0_cudnn-9.26.0` |

## Consequences

- **Hardware floor:** the CUDA EP needs driver R580+ and a Turing or newer GPU. Older NVIDIA GPUs and
  drivers fall back to DirectML automatically with a visible reason (unchanged ladder).
- **Installer:** the bundled CUDA provider shrinks by about 81 MB. The WiX MSI now also sweeps up the
  unused `onnxruntime_providers_nv_tensorrt_rtx.dll` next to the exe (`verify-bundle.ps1` should list
  it as a known extra); NSIS ships exactly the staged set.
- **NVIDIA pack:** 1.9 GB instead of 1.4 GB, mostly the cuDNN 9.26 `cuda13` archive (1.28 GB). Its
  archive layout (`bin\x64`) is handled by the recursive DLL search, but a real install has not been
  run in this gate (large download); P1-D1 moves the pack to the model store and may pick a smaller
  cuDNN `cuda13` release. Packs installed before this gate (`runtimes/nvidia-cu12`) are no longer used
  and are not deleted automatically.
- **Dev builds:** after the upgrade, delete stale `DirectML.dll` / `onnxruntime_providers_*.dll` copies
  from every `target/<profile>` (and `deps`, `examples`) and run `cargo clean -p ort-sys`, because
  ort-sys will not overwrite them. Without symlink rights (no Developer Mode) ort-sys copies only into
  `target/<profile>`, so tests in `target/<profile>/deps` see no provider next to them unless copied
  by hand (the CUDA rung then reports the missing provider; nothing fails).
- **Auto EP selection:** rc.13's `SessionBuilder::new()` sets ORT's `MaxEfficiency` device policy,
  which applies only when no EP is appended. Every Sigil session appends one explicitly through
  `ort_session::builder_for_ep`, so behaviour is unchanged; never build sessions elsewhere.
- **Follow-ups:** P1-D1 switches the CUDA rung to registered-device sessions and moves the provider to
  a pack (decision 5); P1-A adopts `<app_data>/models/kitten/` and the legacy hf-hub cache; P3 uses
  `half`, IOBinding, CUDA graphs and `ep::directml::api()`; release tooling should verify staged
  provider versions against the linked ORT (INTEGRATE).
