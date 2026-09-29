# sherpa-onnx runtime pack (`runtime/sherpa-onnx`)

Sigil's CPU speech tier (Nemotron / SenseVoice / Qwen3-ASR / Parakeet / Moonshine recognition,
Kokoro / Kitten / Piper voices, spoken language ID, speaker embeddings) runs on the
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) C API, loaded in-process from this pack by
`src-tauri/src/services/speech_rt/sherpa/`. Decision record: `docs/adr/0002-sherpa-onnx.md`.

## What the pack is

| | |
|---|---|
| Catalog entry | `src-tauri/model-catalog/runtime-sherpa.json`, id `runtime/sherpa-onnx`, status `published` |
| Upstream asset | `sherpa-onnx-v1.13.8-win-x64-shared-MT-Release-lib.tar.bz2` (8,032,957 bytes, sha256 `b8eedf41…a0b9e7`), GitHub release `v1.13.8` of k2-fsa/sherpa-onnx |
| Installed files | `onnxruntime.dll` (ONNX Runtime 1.28.2, CPU) and `sherpa-onnx-c-api.dll`, about 22 MB |
| Install folder | `<app_data>/runtimes/sherpa-onnx/1.13.8/cpu/` (model store layout) |
| Build | static CRT (`/MT`): no VC++ redistributable needed; imports only `onnxruntime.dll`, KERNEL32, ADVAPI32 |

It is the official upstream binary, so nothing is built locally. No execution-provider DLLs are
installed: sherpa's ONNX Runtime stays CPU-only, and Sigil's own statically linked ONNX Runtime owns
every GPU session.

## Install

- In the app: Models & Runtimes (P2 wires engines to it), or
- `Sigil.exe --headless fetch runtime/sherpa-onnx` (one 8 MB download, hash-verified), or
- offline: `SIGIL_LOCAL_ASSETS=<folder with the .tar.bz2> Sigil.exe --headless fetch runtime/sherpa-onnx`.

Development override without installing: `SIGIL_SHERPA_RUNTIME_DIR=<folder with the two DLLs>`, or
`--runtime-dir <folder>` on the probe.

## Probe

```powershell
# Runtime only (versions, which onnxruntime.dll the process bound)
Sigil.exe --headless engine-probe sherpa

# A model package folder (as extracted from the k2-fsa asr-models / tts-models releases) + a WAV
Sigil.exe --headless engine-probe sherpa <sense-voice folder> <file.wav> -- threads=2 repeat=5
Sigil.exe --headless engine-probe sherpa <kokoro folder> -- text="Hello there." sid=0 out=hello.wav

# Same decode while Sigil's own ONNX Runtime keeps a CUDA session busy (70% duty)
Sigil.exe --headless engine-probe sherpa <folder> <file.wav> -- cuda_load=0.7

# The whole ADR matrix into bench-results\sherpa\<timestamp>\
pwsh scripts/runtime-packs/sherpa/probe-matrix.ps1
```

Extras after `--` are documented at the top of `speech_rt/sherpa/probe.rs` (`kind=`, `mode=`,
`threads=`, `lang=`, `model_type=`, `hotwords=`, `repeat=`, `chunk_ms=`, `text=`, `sid=`,
`speed=`, `silence=`, `out=`, `config=<file.json>`, `load_only=1`, `validate_child=1`,
`cuda_load=`, `load_ep=`, `cuda_preload=`). Unsupported option combinations (for example
`hotwords=` on SenseVoice) end the probe with exit code 1 and a message, never sherpa's `exit(-1)`.

## Tests

```powershell
cargo test --manifest-path src-tauri/Cargo.toml speech_rt::sherpa            # no DLL / model needed
$env:SIGIL_TEST_MODELS_DIR = "$env:LOCALAPPDATA\sigil-dev-cache"               # sherpa-onnx\pack-1.13.8 + models\...
$env:SIGIL_TEST_CUDA_BIN = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64"  # optional
$env:SIGIL_TEST_CUDNN_BIN = "<folder with cudnn64_9.dll>"                                        # optional
cargo test --manifest-path src-tauri/Cargo.toml speech_rt::sherpa -- --include-ignored --test-threads=1
```

## Upgrading sherpa-onnx

1. `pwsh scripts/runtime-packs/sherpa/pin-pack.ps1 -Version <x.y.z> -Verify` and paste the printed
   file object into `runtime-sherpa.json`; bump `version`.
2. `pwsh scripts/runtime-packs/sherpa/gen-bindings.ps1 -Version <x.y.z> -HeaderSha256 <sha of c-api.h>`
   (needs LLVM's `libclang.dll`; bindgen runs offline from the cargo registry).
3. Bump `EXPECTED_VERSION` (and `ORT_API_LEVEL` if sherpa moved to a newer ONNX Runtime) in
   `speech_rt/sherpa/lib.rs`; update the size / offset table in `speech_rt/sherpa/tests.rs` from
   the regenerated layout asserts.
4. Re-run the tests with models and `probe-matrix.ps1`, and record the numbers in ADR 0002.

The runtime refuses any other sherpa-onnx version than `EXPECTED_VERSION`: the C API's config
structs grow between releases, so a mismatched pack would read past the structs Sigil passes.
