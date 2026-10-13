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
| `runtime/nemo-speech-cuda` / `-vulkan` / `-cpu` | the server, NVIDIA's own **v0.2.0** Windows release zips (CUDA 102 MB, Vulkan 19 MB, processor 5.5 MB) | GitHub release, pinned by digest, **published** |
| `tts/magpie-tts-357m-v2602` | Magpie TTS Multilingual 357M, v2602 f16 GGUF (449 MB) | `nvidia/magpie_tts_multilingual_357m@452ef56`, pinned, **published** |
| `tts/nemo-nano-codec-22khz` | NanoCodec 22 kHz decoder GGUF (79 MB) | `nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps@fc00890`, pinned, **published** |
| `tts/magpie-tts-357m-v2602-tokenizer` | the 7 tokenizer files inside NVIDIA's `.nemo` | `fetch-magpie-tokenizer.ps1` rebuilds them; each file pinned; flat `assets-v1` release assets, **unpublished** until uploaded |
| `tts/magpie-tts-357m-v2607-tokenizer` | the v2607 members (one zip) | `fetch-magpie-tokenizer.ps1 -Version v2607`; pinned; `assets-v1` asset, **unpublished** until uploaded |
| `tts/magpie-tts-357m-v2607` | v2607 f16 GGUF (569 MB): adds Korean, Arabic, Brazilian Portuguese | NVIDIA's own file, used as it is: `nvidia/magpie_tts_multilingual_357m@1980687`, pinned; **unpublished** until `real_ggml_engine_speaks` has passed on the v0.2.0 packs (see below) |

## Why the tokenizers are Sigil-hosted

- NVIDIA ships Magpie's pronunciation dictionaries only inside the `.nemo` archive. NeMo-Speech.cpp's
  own model index downloads just the tokenizer members of it (the first 32 MiB for v2602; for v2607
  two byte ranges, about 31 MB, because the members v2607 changed are appended after the weights);
  Sigil's model store cannot do ranged partial-archive installs, so `fetch-magpie-tokenizer.ps1`
  does that once, checks every member and writes the files that are uploaded to the `assets-v1`
  release (`scripts/release/README.md`). Until they are uploaded and the entries are flipped,
  Import the checked files. (A model-store feature request for `tar` archives with byte ranges
  would make this a normal download.)

## NeMo-Speech.cpp v0.2.0 and the two checkpoints

- **v2607** uses factor-2 frame stacking. The v0.1.0 server refuses it (`unsupported frame_stacking_factor=2;
  this example currently supports 1`, checked 2026-10-13 with `real_ggml_engine_speaks`,
  `SIGIL_GGML_MODEL=v2607`, the CPU pack). **v0.2.0** (released 2026-10-02) reads it, and its model
  index installs NVIDIA's own GGUF (`magpie_tts_multilingual_357m.v2607.f16.gguf`, the Hugging Face
  commit `19806879`), so Sigil converts nothing: `convert-magpie-v2607.ps1` is retired (deleted; git
  history has it). The three `runtime/nemo-speech-*` entries are v0.2.0 now.
- **v2602 still runs** on v0.2.0. Its docs and tokenizer loader name both checkpoints: the server takes
  the checkpoint from the GGUF (a `magpietts.tokenizer_profile` key, or the text vocabulary and frame
  stacking dimensions of an older file) and from the tokenizer folder's `model_config.yaml`
  (`nemo_version` 2.6.0rc0 / 2.8.0rc0 and the exact tokenizer list), and it refuses a mismatched
  pair at startup. Each catalog model therefore requires the tokenizer of its own checkpoint (a
  test checks it). Read against the loader's code, the pinned v2602 tokenizer (from the earlier
  `34d7e40` revision of the `.nemo`) satisfies every check of the v2602 profile; the run proves it.
- A user who installed the v0.1.0 packs keeps working installs (the model store counts an older
  install as installed until it is updated). v2602 runs on them; for v2607 the plan refuses with
  "needs the NeMo-Speech.cpp server 0.2.0 or newer ... update it in Models & Runtimes"
  (`magpie::MagpieModel::min_pack`, read from the pack's folder in the model store layout).
- The default model stays v2602, so a saved profile without a model name keeps speaking the same
  voice with the same files. v2607 is an explicit choice (Model: Magpie v2607).

### Publishing v2607

Not done until the run below passes for **both** checkpoints on the v0.2.0 packs and the v2607
tokenizer zip is on the `assets-v1` release (`scripts/release/README.md`):

```powershell
node scripts/models/publish-entries.mjs --ids tts/magpie-tts-357m-v2607-tokenizer,tts/magpie-tts-357m-v2607 --write
# then remove "tts/magpie-tts-357m-v2607" and the v2607 tokenizer from GGML_LOCAL_ONLY_IDS
# (src/core/services/tts/services/ggmlTts.ts): a test keeps that set `unpublished`.
```

```powershell
# The v2602 tokenizer: one 32 MiB request, every member checked against the catalog hashes.
pwsh scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1
# -> Models & Runtimes -> "Magpie TTS v2602 text tokenizer" -> Import -> the printed folder.

# The v2607 tokenizer zip (the release asset of tts/magpie-tts-357m-v2607-tokenizer): two ranged
# requests (about 31 MB, not the whole .nemo), NVIDIA's index (NeMo-Speech.cpp v0.2.0 models/index.json)
# checked, every member hashed. The .nemo is the v2602 tar with the members v2607 changed appended
# after the weights, so the later copy of model_config.yaml and two dictionaries wins, as in tar
# extraction.
pwsh scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1 -Version v2607
# -> the zip and its .sha256 in %LOCALAPPDATA%\sigil-dev-cache\nemo-speech\magpie-tts-357m-v2607-tokenizer\
```

Until the entries are published, in the app: Models & Runtimes -> "Magpie TTS Multilingual 357M
(v2607)" -> Import the GGUF downloaded from Hugging Face
(`https://huggingface.co/nvidia/magpie_tts_multilingual_357m/resolve/19806879b16d3f2ccf28fb112b1bcd16a3c7923e/magpie_tts_multilingual_357m.v2607.f16.gguf`,
568,663,328 bytes, sha256 `30d27551fc4a050e8095139f0185796024493373e83d49443c8ddba33c514e68`) and the
tokenizer zip.

The packs, the v2602 GGUF and the codec download from the Magpie panel's requirement card (or
`Sigil.exe --headless fetch runtime/nemo-speech-vulkan`, `... tts/magpie-tts-357m-v2602`, ...).

## Server and requests

`nemo-speech serve --tts-model <gguf> --codec-model <codec> --tokenizer-dir <tokenizer folder>
--host 127.0.0.1 --port <auto> --no-ui --device cpu|vulkan|cuda:0`. `GET /health` answers 200 once
the model is loaded (process host's readiness). Sigil then sends a short warm-up sentence and logs
its time to first audio and RTF. These flags, the key variable, `/health` and the speech route are
the same in v0.1.0 and v0.2.0 (compared in the two tags' `docs/server.md`, `docs/cli.md`,
`docs/api.md`, `app/serve.cpp` and the HTTP server). What v0.2.0 adds does not matter to Sigil:
`--tts.preempt` (cancels older synthesis for the newest request; off by default, and Sigil sends one
sentence at a time), `tts.chunk-frames` 3 -> 4 (only for streamed audio; Sigil asks for a buffered
clip), the VoiceChat routes (CUDA source builds only), and ITN / TN grammars (Linux and macOS
archives).

Each sentence: OpenAI `POST /v1/audio/speech` `{input, voice, language, response_format: "pcm"}`,
answered with one buffered 22.05 kHz clip (the server does not stream; `speed` must be 1). Voices:
John, Sofia, Aria, Jason, Leo. Languages of the official packs: en, es, de, fr, it, vi, hi (v2602);
plus ko, ar (MSA), pt-BR (v2607). The server normalizes `ko-KR` to `ko`, `pt-BR` and `ar-MSA` stay.
Mandarin and Japanese need a source build with `NEMO_SPEECH_TTS_WITH_ZH` / `_JA`
(`scripts/windows/build.ps1 -Profile full` upstream; the release archives are the `server` profile);
Sigil refuses those sentences with a message instead of reading them with the wrong frontend.

NVIDIA's docs say the fused CUDA decode path needs a Q8_0 GGUF and that the F16 GGUF the model
index installs runs the unfused path. Sigil uses the F16 files NVIDIA hosts; a Q8_0 conversion
(`convert_model.py` in NeMo-Speech.cpp, torch) would be a new maintainer step and is not part of
the catalog.

Not shared with STT: the workstream asked to share NeMo-Speech.cpp with Nemotron if feasible, but
Sigil's Nemotron runs on sherpa-onnx (P2), so this runtime serves only Magpie.

## Manual speak test (P6-F verification)

Through Sigil's own path (plan, process host, readiness, warm-up, key). The test finds each model by
file name in `SIGIL_TEST_MODELS_DIR`, so use **one folder per checkpoint** (the two tokenizers share
the name `model_config.yaml`) and set `SIGIL_APP_DATA_DIR` to a scratch folder so nothing reads or
writes `%APPDATA%\com.sigil.app`:

```powershell
$env:SIGIL_GGML_ENGINE = "magpie"
$env:SIGIL_GGML_PACK_DIR = "<extracted nemo-speech-0.2.0-windows-x86_64-vulkan.zip>"   # the folder that holds bin\nemo-speech.exe
$env:SIGIL_TEST_MODELS_DIR = "<v2602 folder: the GGUF, the codec GGUF and the 7 tokenizer files>"
#                          or  "<v2607 folder: the GGUF, the codec GGUF and magpie-tts-357m-v2607-tokenizer.zip>"
$env:SIGIL_GGML_MODEL = "v2602"               # or v2607
$env:SIGIL_GGML_VARIANT = "vulkan"            # or cuda / cpu
$env:SIGIL_GGML_LANGUAGE = "ko"               # optional; v2607: ko, ar or pt-BR. Put a sentence of that language in
$env:SIGIL_GGML_TEXT = "안녕하세요. 시길에서 말하고 있습니다."   # SIGIL_GGML_TEXT (one sentence per line)
$env:SIGIL_APP_DATA_DIR = "<scratch folder>"
$env:SIGIL_GPU_LOCK = "<scratchpad>\gpu_timing.lock"   # when another benchmark may be timing the GPU
cargo test --manifest-path src-tauri/Cargo.toml real_ggml_engine_speaks -- --ignored --nocapture
```

The test unpacks the v2607 tokenizer zip next to itself (what the model store does on install). It
prints time to first audio and RTF per sentence and writes the WAVs. Point `SIGIL_GGML_PACK_DIR` at
a plain folder, not at an installed `runtimes\nemo-speech-*\<version>\...` one: Sigil reads the
version from that layout, and a 0.1.0 folder refuses v2607. In the app: pick **NVIDIA Magpie TTS**,
install what the card lists, **Test**; `sigil.log` (`sigil::tts::ggml`) has the plan, `ready in … ms`
and the warm-up timing.

Not verified yet: whether the v0.2.0 CUDA zip starts on this driver (its packaging script requires a
self-contained package: a `cublas64_*.dll` shim ships in `bin\`, built with the CUDA 13.0 toolkit for
SM 75, 80, 86, 89 and 120a; Sigil still puts `runtime/nvidia-cuda13` on its PATH as a fallback and
only offers the CUDA variant on a CUDA 13 capable card), the zips' exact folder layout (the
packaging script keeps a top-level folder named after the archive; the program lookup finds
`nemo-speech.exe` below the pack folder either way), and the measured VRAM (the catalog says 1.5 GB,
provisional).
