# Bergamot runtime pack (`runtime/bergamot`)

`sigil-bergamot.exe` runs Mozilla's Bergamot translation engine (Firefox Translations) as a helper
process for Sigil's `bergamot` translation backend. It speaks JSON lines over stdio. Decision record,
protocol and measurements: [docs/adr/0008-bergamot.md](../../../docs/adr/0008-bergamot.md).

| Path | What |
| --- | --- |
| `build.ps1` | Fetch the pinned sources (once), build with MSVC, stage and zip the pack, self-test it |
| `gen-catalog.mjs` | Regenerate `src-tauri/model-catalog/mt-bergamot.json` from Mozilla's model registry |
| `helper/UPSTREAM.json` | Pinned upstream: `mozilla/translations` `inference/` plus its git submodules |
| `helper/CMakeLists.txt` | The helper target on top of upstream's `bergamot-translator` library |
| `helper/src/main.cpp` | Protocol shell: argument parsing, the stdin reader, the scheduler (loads, then finals, then interims; cancel) and stdout isolation |
| `helper/src/json.cpp` | Strict, depth-limited JSON parser and writer |
| `helper/src/engine_bergamot.cpp` | The upstream adapter; the only file that uses Bergamot's API |
| `helper/src/engine_fake.cpp` | Fake engine for protocol tests (`-Fake`) |
| `helper/src/json_test.cpp` | Unit tests of the JSON code (built and run by `-Fake`) |

## Build

Requires Visual Studio 2022 Build Tools with the C++ workload (the "C++ CMake tools" component
provides Ninja) and PowerShell 7.

```powershell
# Once: download the pinned archives (about 7, one at a time) and record their sha256 in helper/UPSTREAM.json.
pwsh scripts/runtime-packs/bergamot/build.ps1 -FetchOnly -PinArchives

# Build the default (AVX2) pack:
#   %LOCALAPPDATA%\sigil-native\packs\bergamot-avx2\   and   ...\sigil-bergamot-<version>-cpu.zip
pwsh scripts/runtime-packs/bergamot/build.ps1

# Compatibility build for CPUs without AVX2 (the zip gets a -sse2 suffix; import it for runtime/bergamot):
pwsh scripts/runtime-packs/bergamot/build.ps1 -Arch sse2

# Protocol shell only, no network and no upstream sources (runs the JSON unit tests and the protocol self-test):
pwsh scripts/runtime-packs/bergamot/build.ps1 -Fake
```

Install the zip in Sigil: open **Models & Runtimes**, find **Firefox Translations engine (Bergamot)**
and choose **Import**. Alternatively, put the zip in a folder named by `SIGIL_LOCAL_ASSETS`. Then
download the directions you need (*Firefox Translations models*), or let the Translation panel offer
them.

## Measure (P4-E acceptance: en↔ja under 150 ms per line)

```powershell
pwsh scripts/runtime-packs/bergamot/build.ps1 -SelfTest -Models "$env:APPDATA\com.sigil.app\models\mt-bergamot"
```

`-Models` expects one subfolder per direction key (`en_ja`, `ja_en`) holding the Mozilla files, which is
the model store's layout (a version folder in between is fine). The self-test loads each direction,
translates ten caption-style lines after a warm-up, prints p50, p95 and max, and fails when p95 exceeds
`-MaxP95Ms` (default 150).

The same run through Sigil's Rust client and process host:

```powershell
$env:SIGIL_BERGAMOT_EXE = "$env:LOCALAPPDATA\sigil-native\packs\bergamot-avx2\sigil-bergamot.exe"
$env:SIGIL_TEST_MODELS_DIR = "$env:APPDATA\com.sigil.app\models"
cargo test --manifest-path src-tauri/Cargo.toml external_helper -- --ignored --nocapture
```

With the fake build's exe and no models, the same test checks protocol conformance instead.

## Models

```powershell
node scripts/runtime-packs/bergamot/gen-catalog.mjs --fetch --write   # one request to Mozilla's registry
node scripts/runtime-packs/bergamot/gen-catalog.mjs --check           # is the fragment current?
```

The generator follows Firefox on Windows (release channel). For each direction it takes the highest
non-prerelease version whose filter expression applies and whose file set is complete. The registry's
sha256 and sizes are copied into the catalog.
