# ADR 0003: Delivering GPU execution providers and reporting the one in use

- **Status:** accepted (P0 interim; P1 revisits CUDA provider placement after the ORT rc.13 gate).
  Update (P1-I0): the P1 follow-ups below are resolved by [ADR 0007](0007-native-runtime-packs.md):
  the NVIDIA runtime is the model-store pack `runtime/nvidia-cuda13` (the interim installer keeps
  working and is preferred second), and the `register_ep_library` provider path is implemented but
  the installer keeps bundling the CUDA provider until `runtime/ort-cuda13-provider` is published.
- **Date:** 2026-09-28
- **Workstream:** P0-H
- **Plan decision:** "Delivering GPU execution providers (today CUDA silently runs on CPU in releases)",
  "CUDA 13 hardware floor", "Testing and benchmark placement"

## Context

Sigil links ONNX Runtime 1.23.2 (ort 2.0.0-rc.11, pyke "cu12" build) statically. Research §4.3#10
and the build map found three silent failures:

1. **The provider DLLs shipped only by accident.** `onnxruntime_providers_cuda.dll` (144 MB) and its
   bridge `onnxruntime_providers_shared.dll` must sit next to the module containing ORT (the exe,
   because ORT is linked statically), and Sigil.exe load-time imports `DirectML.dll`. The NSIS
   installer contained only Sigil.exe (so DirectML resolved to the older System32 copy and CUDA could
   never register); the MSI happened to contain them because the WiX bundler adds every `*.dll`
   next to the built exe (checked on the 2026-09-08 build). Either way no CUDA runtime was present.
2. **ort falls back to the CPU silently** unless `error_on_failure` is set, and some failures appear
   only at session creation (driver init) or at the first run (a missing cuDNN sub-library).
3. **No log sink**: release builds use the GUI subsystem, so the ORT warning went nowhere.

The CUDA provider also needs the CUDA 12 runtime, cuBLAS/cuBLASLt, cuFFT and cuDNN 9, which NVIDIA's
driver does not install (~1.4 GB of NVIDIA redistributables). Constraints: quality > local > free;
the app must start on every PC with nothing extra installed; nothing new may be load-time linked;
downloads are explicit, sequential and hash-verified (the user's router blocks connection bursts).

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A. Provider DLLs in `bundle.resources` of `tauri.conf.json` (plan text) | One config | tauri-build copies resources at **compile time** and fails when they are missing, so every `cargo check/clippy/test` on a clean checkout (CI rust job) breaks, and ort-sys produces the DLLs only during that same build | tauri-build 2.5.6 `copy_resources`, tauri-utils 2.8.3 `ResourcePathNotFound` |
| B. Release-only overlay config (`tauri build --config scripts/release/tauri.release.conf.json`) mapping a staging **directory** to the install root; `--prepare` empties it before the build, `--stage` fills it before bundling | Dev/CI builds untouched; an empty directory is a valid resource at compile time; stale DLLs from another ORT version can never be copied over ort-sys's fresh ones | Release builds must use the overlay (release.yml does) | tauri-utils walks directories and yields nothing for an empty one |
| C. NSIS hooks / WiX fragments copying files | No compile-time check | Two installer technologies to maintain | — |
| D. ort `load-dynamic` + full ORT runtime pack in app data | Smallest installer | Changing EP family needs a restart; bigger change on the rc.11 → rc.13 path | plan fallback |
| CUDA runtime bundled in the installer | Zero setup | +1.4 GB installer, WiX 2 GB cab limit, most users are not on CUDA | — |
| CUDA runtime as a runtime pack in app data, preloaded by full path | App starts without it; opt-in 1.4 GB | One more download step for NVIDIA users | ort's own `preload_dylibs` uses the same mechanism |

## Decision

1. **Packaging (option B).** `scripts/release/tauri.release.conf.json` maps
   `src-tauri/target/native-bundle/` to the install root and sets NSIS `installMode: currentUser`
   (recommended installer; MSI still built). `scripts/release/stage-native-dlls.mjs --prepare`
   runs in `beforeBuildCommand`, `--stage` in `beforeBundleCommand`; it copies the three DLLs from
   `target/<profile>/` as real files, checks the PE header and sha256 and writes
   `src-tauri/target/native-bundle.json`. There is deliberately no DirectML-only (`--without-cuda`)
   variant: the WiX bundler adds every `*.dll` next to the built `Sigil.exe` to the MSI anyway, so
   leaving the CUDA provider out of the staging directory would only shrink the NSIS installer and
   make the two installers differ. `scripts/release/verify-bundle.ps1` checks both installers, the import table and the installed
   layout's headless EP probe; release.yml runs it and also ships a portable zip with the DLLs.
2. **NVIDIA runtime pack** (`diagnostics::nvidia_pack`): CUDA 12.9.1 cudart / cuBLAS / cuFFT and
   cuDNN 9.10.2 (cuda12), archive URLs, sizes and sha256 pinned from NVIDIA's
   `redistrib_12.9.1.json` / `redistrib_9.10.2.json`. Installed on explicit user action (Settings →
   GPU & diagnostics) into `app_data/runtimes/nvidia-cu12/<label>/bin/`: one archive at a time, HTTP
   Range resume, streaming sha256, bounded retries, cancel; only `*.dll` members extracted with
   Windows' own `System32\tar.exe` into a staging directory renamed into place. Right before the
   first CUDA session the DLLs are loaded by full path in dependency order (plus `AddDllDirectory`),
   so the provider's imports bind to them regardless of `PATH`. `SetDefaultDllDirectories` is **not**
   called: it would drop `PATH` for every later `LoadLibrary` and break users whose CUDA/cuDNN comes
   from a system install. Removal of a loaded pack is deferred to the next start.
3. **Truthful sessions** (`neural_audio::ort_session`): every session walks CUDA → DirectML → CPU
   (DirectML → CPU for `dml`). Each GPU rung runs a preflight (CUDA provider DLL and its
   `onnxruntime_providers_shared.dll` bridge present, NVIDIA adapter, pack preloaded), then a cached
   **canary** — a hand-encoded 1-D Conv ONNX graph whose output is checked — then registration with
   `error_on_failure` and the commit. The reason for every skipped rung becomes `EpOutcome.reason`,
   recorded per component (`vc`, `vc.content`, `vc.pitch`, `vc.onnx`, …), emitted as
   `diagnostics:ep_outcome`, shown in the system log and the diagnostics panel, and written to the
   log. Falling back from CUDA to DirectML targets the same NVIDIA card (DXGI order = DirectML
   device id); that DirectML session is not bound by the CUDA-only `vram_cap_mb`, and its reason
   says so. Canary passes and permanent failures (missing DLL, no such device, driver too old) stay
   cached until the pack changes or the user retries ("Test GPU acceleration"; a voice-changer
   start once the integrator wires `canary::forget_failures`); transient failures (out of memory,
   device removed / TDR, busy, unrecognised) are reused for 30 s only, so a game that held the VRAM
   does not disable the GPU until Sigil restarts.
4. **Probe**: a light probe at every start (no GPU runtime loaded: a CUDA context costs hundreds of
   MB of VRAM) and a deep probe (`Sigil.exe --headless ep-probe`, run as a child process from the
   UI's "Test GPU acceleration") with driver/runtime/cuDNN versions and canaries per EP and adapter.
   Installing or removing the pack recomputes and re-emits the light probe; a saved deep report
   from another Sigil version or another pack state is not shown.
5. **Log**: `app_log_dir/sigil.log`, 10 MiB × 5, synchronous writes, panics included; WARN/ERROR
   from Sigil and ERROR from anything else (ORT included; ORT's routine node-placement warnings stay
   in the file) forwarded to the system log (rate limited, backfilled on attach); system-log
   lines written back into the file. EcoQoS and "ignore timer resolution" throttling are switched
   off for the process at startup.

## Evidence

Dev rig H2 (RTX 4090, driver 616.56 supporting CUDA 13.4, AMD iGPU, Windows 10.0.26200; CUDA 12.8
toolkit on `PATH`, no cuDNN), `cargo test -- --ignored --nocapture headless_probe`:

- CUDA canary: **failed in 426 ms** — "cudnn64_9.dll (CUDA 12 / cuDNN 9 runtime) was not found;
  install the NVIDIA GPU runtime pack in Settings" (before: the CUDA EP silently ran on the CPU).
- DirectML canary on the RTX 4090: ok in 283 ms; on the AMD iGPU: ok in 175 ms.
- Recommendation: DirectML; the light probe reaches the same answer from file checks alone.

Packaged build (`pnpm tauri build --config scripts/release/tauri.release.conf.json`, 6 min release
build on H2, then `pwsh scripts/release/verify-bundle.ps1`): PASSED — both the MSI and the NSIS
installer put the three DLLs next to Sigil.exe byte-identical to the staged copies; Sigil.exe still
has exactly the 37 baseline load-time imports (no CUDA/cuDNN/ORT DLL); the installed layout's
headless probe loads the bundled `DirectML.dll`, reports the CUDA failure above and DirectML working
on both GPUs. With `DirectML.dll` deleted from the installed layout, Sigil still starts, loads
`C:\WINDOWS\SYSTEM32\directml.dll` and says so in the probe notes (DirectML still worked on this
Windows 11 build). The MSI additionally carries ort-sys's unused `onnxruntime_providers_tensorrt.dll`
(WiX auto-inclusion); verify-bundle lists it as a known extra.

Unit/integration tests: 40+ Rust tests (canary protobuf computes `1..=14` on the CPU EP, EP ladder
and outcome records, error explanations, pack pins and verification, loopback download with Range
resume / corrupt-partial restart / hash mismatch / cancel, extraction with the real `tar.exe`
including a refused `../` entry, log rotation), vitest for the atomic settings file, formatting and
the staging script. Packaged-build results are in the P0-H report; the CUDA-with-pack run is the
manual test below (the 1.4 GB pack is not downloaded by agents).

## Consequences

- Installers grow by ~163 MB (CUDA provider 144 MB, DirectML 18.5 MB). NVIDIA users download the
  1.4 GB pack once; everyone else never does. DirectML is the automatic GPU path until then.
- The CUDA floor stays CUDA 12 minor-version compatibility (driver R525+) until the rc.13 gate
  (P1-S0a) moves it to CUDA 13 / R580+ and a new pack label; `nvidia_pack` pins must change with it.
- Release builds must pass `--config scripts/release/tauri.release.conf.json`; plain `pnpm tauri
  build` still works but produces a CPU/System32-DirectML-only installer (verify-bundle fails it).
- **Manual acceptance test (NVIDIA PC):** install the NSIS build; Settings → GPU & diagnostics shows
  "Best available: DirectML" and a note that the runtime pack is missing; start the voice changer on
  CUDA → the panel and system log show "CUDA → DirectML" with the cuDNN reason; install the pack
  (progress, cancel/resume works); stop/start the voice changer → "CUDA"; "Test GPU acceleration"
  lists CUDA ok. Then rename `DirectML.dll` in the install dir and start on DirectML → falls back to
  the System32 copy (note shown) or to the CPU with the reason. `verify-bundle.ps1 -RequireCuda`
  automates the check on the build machine once the pack is installed.
- Follow-ups: P1 moves the CUDA provider into the pack via `register_ep_library` (rc.13) and the
  interim downloader/extractor into the model store; call sites pass explicit components
  (`VcSessionBuilder::component`) instead of file-name inference.
