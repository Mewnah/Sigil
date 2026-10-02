# Magpie TTS through NeMo-Speech.cpp (`runtime/nemo-speech-*`, P6-F)

Sigil's `magpie` voice engine (Text to speech → **Experimental GPU voices** → NVIDIA Magpie TTS) runs
NVIDIA's [NeMo-Speech.cpp](https://github.com/NVIDIA/NeMo-Speech.cpp) `nemo-speech serve` as the
`tts-magpie` helper process: 127.0.0.1 only, a random port, a random key in
`NEMO_SPEECH_HTTP_API_KEY` (the user's other `NEMO_SPEECH_*` variables are removed), a kill-on-close
Job Object. Engine code: `src-tauri/src/services/tts_helpers/magpie.rs` and the shared runner in
`tts_helpers/ggml/`; catalog: `src-tauri/model-catalog/tts-ggml-experimental.json`.

**Outcome of the P6-F spike: experimental preset.** Nothing is built by Sigil.

| Catalog id | What | Source |
| --- | --- | --- |
| `runtime/nemo-speech-cuda` / `-vulkan` / `-cpu` | the server, NVIDIA's own v0.1.0 Windows release zips (4.7 / 22 / 106 MB) | GitHub release, pinned by digest, **published** |
| `tts/magpie-tts-357m-v2602` | Magpie TTS Multilingual 357M, v2602 f16 GGUF (449 MB) | `nvidia/magpie_tts_multilingual_357m@452ef56`, pinned, **published** |
| `tts/nemo-nano-codec-22khz` | NanoCodec 22 kHz decoder GGUF (79 MB) | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps@fc00890`, pinned, **published** |
| `tts/magpie-tts-357m-v2602-tokenizer` | the 7 tokenizer files inside NVIDIA's `.nemo` | `fetch-magpie-tokenizer.ps1`, each file pinned, **unpublished** (Import) |
| `tts/magpie-tts-357m-v2607` + `-tokenizer` | v2607: adds Korean, Arabic, Brazilian Portuguese | `convert-magpie-v2607.ps1` (local conversion), **unpublished** (Import) |

## Why some parts are local

- NVIDIA ships Magpie's pronunciation dictionaries only inside the `.nemo` archive. NeMo-Speech.cpp's
  own model index downloads just the first 32 MiB of it (the tokenizer members sit before the
  weights); Sigil's model store cannot do ranged partial-archive installs, so
  `fetch-magpie-tokenizer.ps1` does that once and Sigil imports the checked files. (A model-store
  feature request for `tar` archives with a byte range would make this a normal download.)
- v2607 uses factor-2 frame stacking and NVIDIA publishes it only as a `.nemo`; NeMo-Speech.cpp's
  docs say to convert it locally.

```powershell
# The v2602 tokenizer: one 32 MiB request, every member checked against the catalog hashes.
pwsh scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1
# -> Models & Runtimes -> "Magpie TTS v2602 text tokenizer" -> Import -> the printed folder.

# v2607 (needs Python 3.10-3.12; installs torch CPU into its own venv; downloads the .nemo once).
pwsh scripts/runtime-packs/nemo-speech/convert-magpie-v2607.ps1
# -> Import the GGUF and the tokenizer zip it prints.
```

The packs, the v2602 GGUF and the codec download from the Magpie panel's requirement card (or
`Sigil.exe --headless fetch runtime/nemo-speech-vulkan`, `... tts/magpie-tts-357m-v2602`, ...).

## Server and requests

`nemo-speech serve --tts-model <gguf> --codec-model <codec> --tokenizer-dir <tokenizer folder>
--host 127.0.0.1 --port <auto> --no-ui --device cpu|vulkan|cuda:0 --tts.language-code <default>
--tts.voice-name <default>`. `GET /health` answers 200 once the model is loaded (process host's
readiness). Sigil then sends a short warm-up sentence and logs its time to first audio and RTF.

Each sentence: OpenAI `POST /v1/audio/speech` `{input, voice, language, response_format: "pcm"}`,
answered with one buffered 22.05 kHz clip (the server does not stream; `speed` must be 1). Voices:
John, Sofia, Aria, Jason, Leo. Languages of the official packs: en, es, de, fr, it, vi, hi (v2602);
plus ko, ar (MSA), pt-BR (v2607). Mandarin and Japanese need a source build with
`NEMO_SPEECH_TTS_WITH_ZH` / `_JA` (`scripts/windows/build.ps1 -Profile full` upstream); Sigil
refuses those sentences with a message instead of reading them with the wrong frontend.

Not shared with STT: the workstream asked to share NeMo-Speech.cpp with Nemotron if feasible, but
Sigil's Nemotron runs on sherpa-onnx (P2), so this runtime serves only Magpie.

## Manual speak test (P6-F verification)

Through Sigil's own path (plan, process host, readiness, warm-up, key):

```powershell
$env:SIGIL_GGML_ENGINE = "magpie"
$env:SIGIL_GGML_PACK_DIR = "<extracted nemo-speech-0.1.0-windows-x86_64-vulkan.zip>"
$env:SIGIL_TEST_MODELS_DIR = "<folder with the GGUF, the codec GGUF and the 7 tokenizer files>"
$env:SIGIL_GGML_VARIANT = "vulkan"            # or cuda / cpu
$env:SIGIL_GPU_LOCK = "<scratchpad>\gpu_timing.lock"   # when another benchmark may be timing the GPU
cargo test --manifest-path src-tauri/Cargo.toml real_ggml_engine_speaks -- --ignored --nocapture
```

It prints time to first audio and RTF per sentence and writes the WAVs. In the app: pick **NVIDIA
Magpie TTS**, install what the card lists, **Test**; `sigil.log` (`sigil::tts::ggml`) has the plan,
`ready in … ms` and the warm-up timing.

Not verified in P6-F (no downloads in this phase): whether the official CUDA zip carries its CUDA
runtime (Sigil puts `runtime/nvidia-cuda13` on its PATH as a fallback; if it needs CUDA 12 DLLs it
fails to start and the Vulkan pack is the answer), and the measured VRAM (the catalog says 1.5 GB,
provisional).
