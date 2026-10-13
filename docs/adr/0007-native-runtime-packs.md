# ADR 0007: Native runtime packs: lazily loaded sigil_engine DLLs, and the CUDA 13 runtime and provider as model-store packs

- **Status:** accepted. The loader, ABI implementation, build tooling, NVIDIA runtime pack and
  registered-provider path are implemented and tested on the dev rig. The CUDA provider stays in the
  installer until its pack is published (see Decision 6).
- **Date:** 2026-09-29
- **Workstream:** P1-D1
- **Plan decision:** "ggml isolation and whisper-rs", "Delivering GPU execution providers" (the P1
  part), "Model and runtime store", "Hosting Sigil-produced assets"; global rule "Nothing new is
  load-time linked" (research §0 "each ggml runtime loads as a separate DLL", §1.3; ADR 0001 decision 5;
  ADR 0003)

## Context

Sigil needs in-process native engines (transcribe.cpp for the GPU speech tier, later other ggml
engines) and it needs the CUDA execution provider's runtime from somewhere other than the
installer. The constraints are:

- The app must start, and every existing feature must work, with no pack installed. Nothing new may
  be load-time linked.
- Two in-process engines that both depend on a same-named `ggml.dll` bind to whichever loaded first,
  because the Windows loader matches by base name. A second static ggml cannot be linked into
  Sigil.exe (whisper-rs already has one).
- Friends' PCs range from pre-AVX2 CPUs to RTX 40 cards. A build tuned for the build machine
  (`GGML_NATIVE=ON`) crashes elsewhere.
- Downloads are explicit, sequential and hash-verified.
- Asset hosting is **local only** for now (user decision, 2026-09-28). Sigil-built packs are built
  by local scripts and installed through local import or `SIGIL_LOCAL_ASSETS`, and their catalog
  entries stay `unpublished`. No publishing or runtime-pack release workflows are created.
  *Update 2026-10-13:* the user asked for the download buttons. The assets are uploaded to the
  `assets-v1` release by hand with `scripts/release/publish-assets.ps1` (it uploads only with
  `-Execute`) and `scripts/models/publish-entries.mjs` flips the entries after hashing the uploaded
  bytes; there is still no CI workflow for it.
- P0 bundled the CUDA provider (144 MB on rc.11, 62 MB on rc.13) next to Sigil.exe and installed
  NVIDIA's CUDA / cuDNN DLLs with an interim downloader (ADR 0003). ADR 0001 verified that
  `register_ep_library` can load the provider from an app-data folder.

## Options considered

| Topic | Option | Pros | Cons | Verdict |
| --- | --- | --- | --- | --- |
| Loading | Import libraries / delay-load | Simple calls | Load-time dependency, and a missing pack stops Sigil from starting | Rejected |
| Loading | `libloading` + `LoadLibraryExW(DLL_LOAD_DIR \| DEFAULT_DIRS)` by full path, plus `AddDllDirectory` for pack folders | Nothing is load-time linked; `PATH` and the current directory are never searched | Every call goes through a function table | **Chosen** |
| Loading | `SetDefaultDllDirectories` | Strict search order | Drops `PATH` for every other `LoadLibrary` in the process (system CUDA installs break; ADR 0003) | Rejected |
| Hash check at load | The model store's `installed.json` record | Already has per-file hashes | Private to P1-A, not a frozen contract | Rejected |
| Hash check at load | `sigil-pack.json` inside the pack (sizes and sha256 of its files, written by the build script) and the catalog pin for plain-file packs (the CUDA provider) | Works for installed packs and `--runtime-dir` folders; the archive is pinned for published packs | The manifest itself is only as trustworthy as the install | **Chosen** |
| Variant choice | Load every variant and ask it | Exact | Loads GPU drivers into Sigil just to list packs | Rejected |
| Variant choice | CPUID, sysinfo's CUDA eligibility and a Vulkan heuristic, then fall through to the next variant when a DLL fails to load or reports `SIGIL_E_BACKEND_UNAVAILABLE` at create | Cheap, and a wrong guess costs one failed attempt | Heuristic Vulkan detection | **Chosen** |
| CUDA provider | Keep it next to Sigil.exe (installer) | Proven (verify-bundle), no download | +62 MB installer | **Kept for now** (Decision 6) |
| CUDA provider | App-data pack registered with `register_ep_library`; sessions use its EP devices | Smaller installer; the provider installs with the NVIDIA runtime | The pack must be hosted somewhere | **Implemented**, used when the installer's copy is missing |
| CUDA provider | Copy the pack's provider next to the exe on per-user installs | No plugin-EP path | Writes into the install folder, and the updater may fight it | Not needed (registration works) |
| CUDA provider | ort `load-dynamic` + a full ORT runtime pack | Smallest installer | Changing EP family needs a restart; large change | Rejected |
| Pack builds | `.github/workflows/runtime-packs.yml` | Reproducible CI builds | Contradicts the local-only decision | **Not created**; local script instead |

## Decision

1. **Loader** (`src-tauri/src/services/native_runtime/`). `EngineLibrary::load(dir, file)`:
   - reuses an already loaded copy of the same file;
   - checks the pack manifest's sizes and sha256 before any code from the folder runs;
   - loads the DLL by full path;
   - calls `sigil_engine_abi_version()` first and refuses (and unloads) any version other than 1;
   - resolves all 16 exports, naming any missing ones;
   - validates `sigil_engine_get_info()` (size ≥ 56, ABI field, UTF-8, engine name) and compares it
     with the manifest;
   - routes engine logs to `tracing` (`sigil::engine`).

   Calls on one instance and its streams are serialized by a per-instance lock. `sigil_engine_cancel`
   runs from any thread without it (`EngineCanceller`). Engine strings and audio are copied and freed
   by the engine's own `sigil_free_*`. The thread-local last error is read before any other call. A
   stream keeps its instance alive, and `destroy` runs only after the last stream is closed.
   GPU-capable libraries (`SIGIL_CAP_GPU`) stay loaded until exit; others unload when unused.
   A missing, damaged or foreign DLL is always an `Err`, never a panic.
2. **Packs and variants.** A Sigil-built engine pack is the catalog entry
   `runtime/sigil-<engine>-<variant>` (kind `runtime`, `variant` set) holding `sigil_<engine>.dll`
   and `sigil-pack.json`. Its folder comes only from `model_store::installed_path`.
   - `open_engine(engine, &OpenRequest)` tries installed variants in the order cuda13, vulkan,
     cpu-avx512, cpu-avx2, cpu, among those this PC supports. The GPU variants carry cpu-avx2 CPU
     kernels, so they need AVX2.
   - A variant that fails to load, or whose create reports `SIGIL_E_BACKEND_UNAVAILABLE` (or an
     out-of-memory, load or runtime failure on a GPU variant), falls through to the next one.
   - `cuda13` variants preload the NVIDIA runtime first. Every skipped or failed variant is
     reported.
3. **ABI implementation and builds** (`native/`).
   - `common/sigil_engine_impl.{hpp,cpp}` implements all 16 exports once. Exceptions become status
     codes plus a thread-local last error; it also handles JSON validation, audio checks,
     engine-owned allocations, per-call cancel reset and the log bridge.
   - Engines derive from `sigil::Engine` and define `engine_descriptor()` and `create_engine()`.
   - `cmake/SigilVariant.cmake` applies to every target: `/MD`, `BUILD_SHARED_LIBS=OFF` (static
     ggml inside each engine DLL), `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=OFF`,
     `GGML_CPU_ALL_VARIANTS=OFF`, no OpenMP, and the variant's instruction sets (FORCEd):
     - cpu: SSE4.2 baseline;
     - cpu-avx2: `/arch:AVX2` + FMA, F16C, BMI2;
     - cpu-avx512: `/arch:AVX512`;
     - vulkan: GGML_VULKAN;
     - cuda13: GGML_CUDA with sm_75 through sm_120.
   - `sigil_add_engine()` builds a uniquely named DLL. It exports only the ABI, carries a
     VERSIONINFO resource, and is built with `/W4`, `/guard:cf` and deterministic `/Brepro` output.
   - `CMakePresets.json` has one preset per variant.
   - `sigil_engine_check.exe` validates a built DLL the way Sigil loads it.
   - `engines/echo/` is the test engine.
   - `scripts/native/build-runtime-packs.ps1` builds, checks, stages (`native/out/dev/…`), writes
     reproducible zips (`native/out/packs/…`) and a `SIGIL_LOCAL_ASSETS` tree, and checks the
     catalog entry. `versions.json` holds the toolchain pins.
4. **Catalog.**
   - `runtime-nvidia.json`: `runtime/nvidia-cuda13`, published. It pins NVIDIA's cudart, cuBLAS,
     cuFFT and cuDNN 9.26 `cuda13` archives (the same pins as the interim installer; a test keeps
     them equal) and extracts only `bin/**` DLLs.
   - `runtimes.json`: `runtime/ort-cuda13-provider` (unpublished, sha256-pinned, requires the NVIDIA
     runtime), and `runtime/sigil-echo-cpu` / `-cpu-avx2` (unpublished, unpinned local builds).
5. **NVIDIA runtime** (`native_runtime::nvidia`).
   - Before the first CUDA session or `cuda13` engine, the model-store pack is preloaded by full
     path in dependency order, with its folders added via `AddDllDirectory`. Without it, the
     interim P0 pack is used if installed, so nobody downloads it twice.
   - Once loaded, it stays loaded. A pack removed while Sigil runs is reported as removed: the CUDA
     rung falls back to DirectML until the next start.
   - Installing or removing a pack changes a fingerprint, which drops the cached EP canary results.
   - Leftovers are listed as `legacy/*` rows and deleted with `remove_legacy_runtime`: the P0-era
     `runtimes/nvidia-cu12` (never loaded by ORT 1.28), and the interim pack once the store pack is
     installed.
6. **CUDA provider** (`ort_session::cuda_provider_source`, chosen once per process). The provider
   next to `Sigil.exe` is used when present and version-matched. Otherwise the
   `runtime/ort-cuda13-provider` pack (pinned sha256, ORT version checked) is registered with
   `register_ep_library`. Every CUDA session (and the canary) then appends that EP device through
   `with_devices`, with `device_id`, `gpu_mem_limit` and `arena_extend_strategy` as provider
   options. `SIGIL_CUDA_PROVIDER=exe|pack` forces a source (development).
   `onnxruntime_providers_shared.dll` (10 KB) stays next to the exe.

   **The installer still bundles the 62 MB provider.** The pack is a Sigil-built asset, asset
   hosting is local-only, and the entry is therefore `unpublished`: no user could download it.
   Dropping it from the installer now would silently remove CUDA for every NVIDIA user. When the
   maintainer publishes the pack:
   1. upload `onnxruntime_providers_cuda.dll` (staged by
      `build-runtime-packs.ps1 -Engine ort-cuda-provider`) to the `assets-v1` release (as
      `onnxruntime_providers_cuda-1.28.0-cuda13.dll`, `publish-assets.ps1`);
   2. flip the entry to `published` (`publish-entries.mjs --write`);
   3. add `runtime/ort-cuda13-provider` to `runtime/nvidia-cuda13`'s `requires`;
   4. remove the provider from `scripts/release/stage-native-dlls.mjs`, `verify-bundle.ps1` and the
      expected installer contents.

   The runtime side needs no change.

## Evidence

Dev rig H2: RTX 4090 (driver 616.56), AMD iGPU, Windows 11 26200, MSVC 19.50 (VS 2026 Build Tools),
CMake 4.2.3.

| Check | Result |
| --- | --- |
| `pwsh scripts/native/build-runtime-packs.ps1 -Engine echo -Variant cpu` | Builds, passes `sigil_engine_check` (ABI v1, 16 exports, info, create + query), stages and packs. `sigil_echo.dll` is 214 KB and exports exactly the 16 ABI symbols. Its only imports are MSVCP140, VCRUNTIME140(_1), UCRT and KERNEL32, all in Sigil.exe's baseline. No `/W4` warnings. |
| Reproducibility | After `-Clean`, a rebuild gives the same zip (`sigil-echo-0.1.0-cpu.zip` sha256 `12e69dd0…488c46`) with the same toolchain. |
| `cargo test native_runtime` | 30 passed, 1 ignored. These tests run the loader against an in-process fake engine: ABI v2 and bad info refused, every string and buffer freed once, streams outliving their instance, cross-thread cancel. They also cover manifest and hash tampering, the catalog convention and planning, variant fallback order, DLL discovery and preload failures, the provider pin, leftover removal, IPC rows, probe helpers, and the embedded NVIDIA entry matching the interim pins and preload order. A garbage file and a renamed system DLL are refused without panics. |
| `SIGIL_TEST_ECHO_DIR=… cargo test -- --ignored real_echo_engine_end_to_end` (cpu and cpu-avx2) | Pass. Real DLL: hash-verified load, calls, stream, cancel in about 60 ms, create failures with their status, a tampered and an ABI-v2 manifest refused, and the probe report. |
| `Sigil.exe --headless engine-probe echo --runtime-dir native\out\dev\echo\cpu -- gain=0.5` | `ok: true`, open 2.2 ms, `echo_check` max abs diff 0.0, cancel passes. |
| `SIGIL_LOCAL_ASSETS=native\out\local-assets Sigil.exe --headless fetch runtime/sigil-echo-cpu runtime/sigil-echo-cpu-avx2`, then `engine-probe echo` (scratch `SIGIL_APP_DATA_DIR`) | Installed through the model store into `runtimes/sigil-echo-<v>/0.1.0/<v>/`. The probe selects `cpu-avx2` on this AVX2 PC. `--variant cpu` works; `--variant cuda13` gives "not installed". `-- fail=backend` falls through cpu-avx2, then cpu, then a clean error. A byte flipped in the installed cpu-avx2 DLL is caught by the manifest hash, and the probe falls back to cpu. |
| `cargo test -- --ignored registered_provider_pack_builds_cuda_sessions` (CUDA 13.2 cuBLAS + a cuDNN 9 stand-in) | The provider, copied to a `runtimes/ort-cuda13-provider/1.28.0/cuda13` folder, registers. `build_session("cuda")` with a 512 MB VRAM cap is granted CUDA in 212 ms, the Conv canary output is correct, the provider module is loaded from the pack folder, and device 7 gives "CUDA device 7 does not exist". |
| NVIDIA pack, real install | The four pinned archives (1,885,987,422 bytes) were downloaded one at a time into `%LOCALAPPDATA%\sigil-dev-cache\nvidia-redist`, and all four sha256 match the pins. `SIGIL_LOCAL_ASSETS=<that folder> Sigil.exe --headless fetch runtime/nvidia-cuda13` then installed 1.4 GB of DLLs in 6 s into `runtimes/nvidia-cuda13/cuda-13.2.0_cudnn-9.26.0/cuda13/bin/x64/` (the `bin/x64` layout is confirmed). |
| cuDNN 9.26 sub-libraries | The first probe failed: "Failed to initialize CUDNN Frontend". cuDNN 9.26 ships two sub-libraries the fixed list did not know, `cudnn_engines_tensor_ir64_9.dll` and `cudnn_ext64_9.dll`, which cuDNN loads by name at run time. The preload now also loads every other `cudnn*64_9.dll` in the pack (optional), and the catalog extracts `cudnn*64_9.dll`. The interim P0 installer extracts only its eight named cuDNN DLLs, so it has the same failure with this pin (follow-up below). |
| `Sigil.exe --headless ep-probe` with the store pack (scratch app data, nothing on `PATH`) | CUDA canary OK in 238–256 ms on the RTX 4090, and `cuda` is recommended. After hiding the pack's install record, a new process reports "cudnn64_9.dll … was not found" and recommends `dml`. After restoring it, CUDA works again. |
| `cargo test -- --ignored nvidia_pack_install_and_removal_switch_cuda_and_directml` (one process) | With the pack, `granted cuda`. After removal: `dml`, with the reason "the NVIDIA GPU runtime pack was removed while Sigil was running; restart Sigil to finish removing it". |
| Provider pack, real install | `Sigil.exe --headless fetch runtime/ort-cuda13-provider` from the staged local assets installed it, pin verified. With `SIGIL_CUDA_PROVIDER=pack`, the probe runs the CUDA canary through the registered pack provider (280 ms). With the pack hidden, it reports "…pack is not installed". With a byte flipped in it, it reports "…does not match the pinned CUDA provider", and falls back to DirectML. |
| Sigil.exe load-time imports (debug) | 41, all in `scripts/release/expected-imports.txt`; nothing new. |
| `cargo clippy --all-targets` | 0 warnings. |

## Consequences

- **Nothing loads at startup.** The runtime plugin only sweeps interrupted leftover removals on a
  background thread. Engines load on first use. Without packs, nothing changes for users.
- The Models & Runtimes tab lists `runtime/nvidia-cuda13`. Users can install the CUDA 13 runtime
  there, with store resume and mirror handling. Settings → GPU & diagnostics still uses the interim
  installer until INTEGRATE points it at the store.
- Engine workstreams (P1-G2 transcribe, P2) add `native/engines/<name>/` with an `engine.json`,
  their own catalog fragment with `runtime/sigil-<name>-<variant>` entries (`cuda13` ones require
  `runtime/nvidia-cuda13`), and call `native_runtime::open_engine`.
- **Manual acceptance (NVIDIA PC):**
  1. Install `runtime/nvidia-cuda13` from the Models & Runtimes tab, or with
     `Sigil.exe --headless fetch runtime/nvidia-cuda13`.
  2. Start the voice changer on CUDA. The EP outcome is `granted cuda`.
  3. Remove the pack in the tab and start the voice changer again. It reports "the NVIDIA GPU
     runtime pack was removed…" and falls back to DirectML.
  4. After a restart the reason is the missing cuDNN, and it is still DirectML.
  5. With `SIGIL_CUDA_PROVIDER=pack` and the provider pack installed, the log shows "CUDA provider
     registered from …\runtimes\ort-cuda13-provider\…".
- **Follow-ups:**
  - INTEGRATE, **urgent:** the interim installer (`diagnostics::nvidia_pack`, Settings → GPU &
    diagnostics) produces a pack in which cuDNN 9.26 cannot initialize (the two sub-libraries
    above are missing). The fix is to point its install/remove commands at the store pack
    (`model_store::ensure(["runtime/nvidia-cuda13"])`) and retire its downloader. The smallest
    alternative is to add both DLLs to its cuDNN `dlls` list and to its `PRELOAD_ORDER` as
    optional entries.
  - INTEGRATE: the EP probe's pack status moves to the store pack (`native_runtime::nvidia`,
    `ort_session::cuda_provider_in_use`).
  - INTEGRATE: `src/types.ts` `RustRuntimePackStatus` gains `title` and `legacy`, and the ADR index
    links this record.
  - Maintainer, when publishing: see Decision 6.
