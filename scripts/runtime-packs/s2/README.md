# Fish S2 Pro through s2.cpp (`runtime/s2-server-*`, P6-F)

Sigil's `s2` voice engine (Text to speech → **Experimental GPU voices** → Fish S2 Pro) runs
[s2.cpp](https://github.com/rodrigomatta/s2.cpp) (the C++/ggml port of Fish Audio S2 Pro; code and
weights under the Fish Audio Research License) as the `tts-s2` helper process. Engine code:
`src-tauri/src/services/tts_helpers/s2.rs`, the shared runner and adapter in `tts_helpers/ggml/`.
Fish S2 Pro also works through the **Fish Speech** backend with Fish Audio's official server
(Linux / WSL2 is much faster than native Windows there).

**Outcome of the P6-F spike: experimental preset, Sigil-built packs.** Upstream publishes no Windows
binaries.

| Path | What |
| --- | --- |
| `UPSTREAM.json` | The pinned s2.cpp commit (`2c33261`, 2026-05-27) and, after the first fetch, its archive pin and its ggml submodule pin |
| `build.ps1` | Fetch the pinned archive(s) once, build with MSVC + Ninja (cuda13, vulkan, cpu), check imports, read the GPU switches from `s2.exe --help`, stage and zip the pack |
| `NOTICE.txt` | Attribution shipped in every pack (with upstream's LICENSE) |

| Catalog id | What |
| --- | --- |
| `runtime/s2-server-cuda13` / `-vulkan` / `-cpu` | the packs this script builds (**unpublished**: Import or `SIGIL_LOCAL_ASSETS`) |
| `tts/s2-pro-q4_k_m` / `-q6_k` / `-q8_0` | GGUFs (3.6 / 4.5 / 5.6 GB; about 7.2 / 8.4 / 10 GB of video memory), `rodrigomt/s2-pro-gguf@a732069`, pinned, published |
| `tts/s2-pro-tokenizer` | `tokenizer.json` (Qwen3 BPE, 12 MB), same repo |

## Build

Requires Visual Studio 2022 Build Tools with the C++ workload (and "C++ CMake tools" for Ninja),
PowerShell 7, the CUDA 13 toolkit for `cuda13` and the Vulkan SDK (`VULKAN_SDK`) for `vulkan`.

```powershell
# Once: fetch the pinned archive (one request; a ggml submodule adds one API call and one archive),
# review the source, and record the archive pins in UPSTREAM.json.
pwsh scripts/runtime-packs/s2/build.ps1 -FetchOnly -PinFirstFetch

# The packs: %LOCALAPPDATA%\sigil-native\packs\s2-server-<variant>\ and
#            ...\sigil-s2-server-2c33261-s1-<variant>.zip (+ .zip.sha256)
pwsh scripts/runtime-packs/s2/build.ps1 -Variant cuda13,cpu
pwsh scripts/runtime-packs/s2/build.ps1 -Variant vulkan        # on a PC with the Vulkan SDK
```

ggml is linked statically (`BUILD_SHARED_LIBS=OFF`, `GGML_BACKEND_DL=OFF`) with `GGML_NATIVE=OFF`
and the AVX2 baseline. The build reads `s2.exe --help` and writes the switches it found into
`sigil-pack.json`: `server_args` (how this variant selects the GPU, e.g. `--cuda 0`) and
`low_vram_args` (`--gpu-layers 18 --codec-cpu` when offered, for the panel's low video memory switch).
Sigil passes only switches made of plain words and numbers, never a host, port, model or path.

Install a zip: Models & Runtimes → **Fish S2 Pro server (…)** → **Import** (Sigil checks the program's
size and sha256 against `sigil-pack.json` before every start).

## How Sigil runs it

`s2 --model <gguf> --tokenizer <tokenizer.json> --server --host 127.0.0.1 --port <auto>` plus the
pack's switches. The server has no health route (Sigil waits until it answers any HTTP request: it
listens once the model is loaded), no key and one synthesis at a time (503 for a second), and its
API is a multipart `POST /generate`. Sigil talks to it through its in-process adapter
(`tts_helpers/ggml/adapter.rs`): a key-protected loopback endpoint that turns each sentence into
`text` + `params` (`output_format: pcm_s16le`, `chunked: false`: the live chunks re-decode the
confirmed prefix, slower overall) and, for a cloned voice-library voice, `reference` (the clip) +
`reference_text`. It never forwards a voice path or a voice folder (s2.cpp would read any `.s2voice`
path it is given). Residual risk as for Breeze: other programs on this PC can reach the server's own
loopback port while it runs.

Without a reference S2 Pro picks a new voice for every sentence; choose a library voice for a steady
one. Quality degrades past about 37 s of audio, far above a caption sentence.

## Manual speak test (P6-F verification)

```powershell
$env:SIGIL_GGML_ENGINE = "s2"
$env:SIGIL_GGML_PACK_DIR = "$env:LOCALAPPDATA\sigil-native\packs\s2-server-cuda13"
$env:SIGIL_TEST_MODELS_DIR = "<folder with s2-pro-q4_k_m.gguf and tokenizer.json>"
$env:SIGIL_GGML_VARIANT = "cuda13"                 # needs runtime/nvidia-cuda13 installed in Sigil
$env:SIGIL_GGML_VOICE = "<a voice-library id>"     # recommended: a steady voice
$env:SIGIL_GPU_LOCK = "<scratchpad>\gpu_timing.lock"
cargo test --manifest-path src-tauri/Cargo.toml real_ggml_engine_speaks -- --ignored --nocapture
```
