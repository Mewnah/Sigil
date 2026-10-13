# Native runtime packs

Sigil loads native engines (transcribe.cpp, the echo test engine, future ggml engines) as
**runtime packs**: one uniquely named `sigil_<engine>.dll` per build variant, installed by the model
store into `app_data/runtimes/…` and loaded lazily by `src-tauri/src/services/native_runtime`. Nothing
is load-time linked, so Sigil starts and every existing feature works with no pack installed. The
decision record is [`docs/adr/0007-native-runtime-packs.md`](../../docs/adr/0007-native-runtime-packs.md).

This folder builds and stages packs on your PC; nothing is downloaded (except by CMake projects you add
yourself) and nothing is published by these scripts. Catalog entries of Sigil-built packs start
`"status": "unpublished"` and are installed by local import or `SIGIL_LOCAL_ASSETS` until the zips are
on the `assets-v1` release: `scripts/release/publish-assets.ps1` uploads them and
`scripts/models/publish-entries.mjs` flips the entries (`scripts/release/README.md`, "The assets-v1
release"). A rebuilt zip has other bytes: pin it again (`scripts/models/pin-staged.mjs --force`).

## Requirements

- Windows 10/11 x64, PowerShell 7.2+ (`pwsh`).
- CMake 3.25+ and the MSVC C++ build tools (Visual Studio 2022 or 2026, "Desktop development with
  C++"). CMake's default generator (the newest Visual Studio) is used with `-A x64`.
- `vulkan` variant: the Vulkan SDK (`VULKAN_SDK`).
- `cuda13` variant: the CUDA 13 toolkit (13.0+, tested with 13.2). With a Visual Studio generator the
  CUDA Visual Studio integration must be installed; otherwise run from a Developer PowerShell with
  `-Generator Ninja`.

Versions and pins the script checks are in [`versions.json`](versions.json).

## Build an engine

```powershell
pwsh scripts/native/build-runtime-packs.ps1 -Engine echo -Variant cpu   # one variant
pwsh scripts/native/build-runtime-packs.ps1 -Engine echo                # engine.json default_variants
pwsh scripts/native/build-runtime-packs.ps1 -List                       # engines and variants
```

For each engine and variant the script configures `native/build/<engine>/<variant>`, builds
`sigil_<engine>.dll` and `sigil_engine_check.exe`, validates the DLL the way Sigil loads it (ABI
version first, all 16 exports, `sigil_engine_get_info`, imports, create + `{"op":"info"}`), and writes
to `native/out/` (gitignored).

- Both files are taken from the exact path CMake recorded for `-Config`
  (`<tree>/sigil-outputs/<config>/<target>.path`, written by `sigil_record_output` in
  `native/cmake/SigilEngine.cmake`). A Visual Studio tree keeps `Debug/` and `Release/` side by side,
  so the tree is never searched for the newest DLL.
- A DLL importing a debug CRT (`ucrtbased.dll`, `vcruntime140d.dll`, …) or an OpenMP runtime is refused.
  `-Config Debug` builds link the debug CRT, which exists only on PCs with Visual Studio. They are
  staged as `dev/<engine>/<variant>-debug/` for `engine-probe` on this PC, and no pack is written.
- `cuda13` checks resolve `cudart64_13.dll` / `cublas64_13.dll` from the CUDA toolkit
  (`sigil_engine_check --dll-dir <toolkit>\bin\x64`). Sigil itself takes them from `runtime/nvidia-cuda13`.
- Sometimes the build PC cannot run the variant, and create is skipped so that only the structural
  checks run. The script prints a warning, and `create_skipped` in `build-summary.json` records why.
  Before relying on such a pack, probe it on a PC that can run it. Create is skipped when:
  - the CPU lacks the variant's instruction sets. `sigil_engine_check --variant` detects this itself;
    an AVX-512 build would otherwise die with an illegal instruction on the build PC.
  - there is no NVIDIA driver (`nvcuda.dll`) or Vulkan loader (`vulkan-1.dll`).
  - the engine reports its GPU backend unavailable.
  - you pass `-NoCreate`.
- A checker that dies without printing a result is reported with its exit status (for example
  `0xC000001D`, illegal instruction).

`pwsh scripts/native/test-build-runtime-packs.ps1` self-tests these helpers (no build, no network).

The outputs:

| Output | Use |
|---|---|
| `dev/<engine>/<variant>/` | `sigil_<engine>.dll` + `sigil-pack.json` (sizes and sha256 Sigil checks before loading) |
| `packs/sigil-<engine>-<version>-<variant>.zip` | the pack archive (reproducible: fixed timestamps, sorted entries) |
| `local-assets/runtime/sigil-<engine>-<variant>/…zip` | the layout `SIGIL_LOCAL_ASSETS` expects (`<dir>/<entry id>/<file>`) |
| `symbols/<engine>/<variant>/*.pdb` | debug symbols (never packed) |
| `build-summary.json` | what was built, sizes, sha256, catalog check |

An engine folder with its own `build.ps1` (today `native/engines/transcribe`) is delegated to that
script instead. That script owns the pinned-source fetch, the import allow-list, the CUDA DLL
folders its self-test needs, and the license files its packs must carry. Its packs and zips land in `local-assets/` (`SIGIL_LOCAL_ASSETS` finds a
catalog file by its plain name there), and `build-summary.json` lists them with `delegated` set.

Then either probe the engine without installing it, or install it through the model store:

```powershell
# Probe (loads the DLL like the app, runs query / process / render / stream / cancel):
Sigil.exe --headless engine-probe echo --runtime-dir native\out\dev\echo\cpu
Sigil.exe --headless engine-probe echo --runtime-dir native\out\dev\echo\cpu -- gain=0.5

# Install from the staged local assets, then probe the installed pack (variant auto-selected):
$env:SIGIL_LOCAL_ASSETS = "$PWD\native\out\local-assets"
Sigil.exe --headless fetch runtime/sigil-echo-cpu runtime/sigil-echo-cpu-avx2
Sigil.exe --headless engine-probe echo
```

(In development use `src-tauri\target\debug\Sigil.exe`; set `SIGIL_APP_DATA_DIR` to a scratch folder
to keep test installs out of your real app data.)

## The ONNX Runtime CUDA provider pack

`runtime/ort-cuda13-provider` is `onnxruntime_providers_cuda.dll` of the ONNX Runtime build Sigil
links (1.28.0, from ort-sys; pinned in `src-tauri/model-catalog/runtimes.json`). Sigil registers it
with `register_ep_library` when the provider is not installed next to `Sigil.exe`:

```powershell
pwsh scripts/native/build-runtime-packs.ps1 -Engine ort-cuda-provider
$env:SIGIL_LOCAL_ASSETS = "$PWD\native\out\local-assets"
Sigil.exe --headless fetch runtime/ort-cuda13-provider   # also installs runtime/nvidia-cuda13 (1.9 GB from NVIDIA)
```

`SIGIL_CUDA_PROVIDER=pack` forces sessions to use the pack even when the installer's copy is present
(`exe` forces the installer's copy).

## Adding an engine

1. Create `native/engines/<name>/` with `engine.json` (`name`, `version`, `kind`, `variants`,
   `default_variants`) and a `CMakeLists.txt` that calls
   `sigil_add_engine(<name> VERSION <version> SOURCES … [LIBRARIES …])`.
2. Implement `sigil::engine_descriptor()` and `sigil::create_engine()` and derive from
   `sigil::Engine` (`native/common/sigil_engine_impl.hpp`); the shared implementation provides all
   16 exports, error handling, JSON, cancellation and logging. See `native/engines/echo/echo.cpp`.
3. Third-party ggml code is added with `add_subdirectory` inside the engine's CMakeLists and linked
   statically (`LIBRARIES ggml`): `cmake/SigilVariant.cmake` already forces `BUILD_SHARED_LIBS=OFF`,
   `GGML_NATIVE=OFF`, `GGML_BACKEND_DL=OFF`, `/MD` and the variant's instruction sets.
4. Add catalog entries `runtime/sigil-<name>-<variant>` (kind `runtime`, `variant` set, status
   `unpublished`, the zip as an archive file) to your workstream's catalog fragment. A `cuda13`
   entry lists `"requires": ["runtime/nvidia-cuda13"]`.
5. Load it from Rust with `native_runtime::open_engine("<name>", &OpenRequest { … })`.

## Rules (ABI v1)

- Exactly the 16 exports of `native/common/sigil_engine.h`; `sigil_engine_abi_version()` must return 1.
- `/MD`, static third-party code inside the DLL, unique DLL name, `GGML_NATIVE=OFF`,
  `GGML_BACKEND_DL=OFF`, no OpenMP runtime; CPU variants never use instructions above their level.
- No exceptions, `abort` or `exit` across the boundary; no stdout/stderr output (use the log callback).
- Engine memory is freed only by the engine's `sigil_free_*`; host memory is borrowed per call.
- CUDA runtime DLLs (cudart64_13, cublas64_13) come from the NVIDIA runtime pack, never from an engine pack.
