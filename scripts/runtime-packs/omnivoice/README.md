# OmniVoice through CrispASR (`runtime/crispasr-*`, P6-F)

Sigil's `omnivoice` voice engine (Text to speech → **Experimental GPU voices** → OmniVoice) runs
[CrispASR](https://github.com/CrispStrobe/CrispASR)'s server (MIT) with its `omnivoice` backend as the
`tts-omnivoice` helper process. Engine code: `src-tauri/src/services/tts_helpers/omnivoice.rs` and the
shared runner in `tts_helpers/ggml/`.

**Outcome of the P6-F spike: experimental preset, on CrispASR instead of omnivoice.cpp.**

Why not omnivoice.cpp: it publishes no Windows binaries (Sigil would build and hand out its own
packs), its `tts-server` takes no key, and CrispASR's own measurements put its OmniVoice about 2x
faster on the same GGUF (a fused step graph and a faster codec). CrispASR ships official, hash-pinned
Windows builds, takes a key (`CRISPASR_API_KEYS`), has a health route that waits for the model, and
speaks the OpenAI request shape Sigil already uses.

| Catalog id | What | Source |
| --- | --- | --- |
| `runtime/crispasr-cuda13` | CUDA 13 build **without** NVIDIA's DLLs (110 MB); uses `runtime/nvidia-cuda13` (installed with it, shared with the voice changer) | v0.8.40 release, pinned by digest, **published** |
| `runtime/crispasr-vulkan` / `-cpu` | Vulkan (38 MB) and AVX2 processor (9 MB) builds | same |
| `tts/omnivoice-q8_0` / `-f16` | OmniVoice LM (818 MB / 1.2 GB) | `cstr/omnivoice-GGUF@9226673`, pinned, **published** |
| `tts/omnivoice-tokenizer-f16` | HiggsAudio v2 codec (403 MB) | same |

## How Sigil runs it

`crispasr --server --backend omnivoice -m <lm> --codec-model <codec> --host 127.0.0.1 --port <auto>
--voice-dir <app data>\tts-helpers\omnivoice\voices --tts-steps 32|16`, the key in
`CRISPASR_API_KEYS` (the user's `CRISPASR_*` / `OMNIVOICE_*` variables are removed first). Per
sentence: OpenAI `POST /v1/audio/speech` `{input, language, seed, response_format: "pcm", stream:
true}` → 24 kHz PCM (one chunk per sentence: OmniVoice is not autoregressive). The language is always
sent (OmniVoice's own advice); a fixed seed keeps the automatic voice steady.

Cloning: at every start Sigil copies the chosen voice-library voice into the server's voice folder as
`<id>.wav` + `<id>.txt` (its transcript) and names it in each request. CrispASR implements the EU AI
Act's marking duties: every clip carries an inaudible spread-spectrum watermark (1.5-4.8 kHz), a
cloned voice needs a consent attestation, and cloned speech starts with a spoken AI notice unless the
request attests that the user discloses AI speech themselves. The panel has a switch for each (both
off by default); Sigil sends the attestation texts only when they are on.

## Manual speak test (P6-F verification)

```powershell
$env:SIGIL_GGML_ENGINE = "omnivoice"
$env:SIGIL_GGML_PACK_DIR = "<extracted crispasr-windows-x86_64-vulkan.zip>"
$env:SIGIL_TEST_MODELS_DIR = "<folder with omnivoice-q8_0.gguf and omnivoice-tokenizer-f16.gguf>"
$env:SIGIL_GGML_VARIANT = "vulkan"                 # cuda13 needs runtime/nvidia-cuda13 installed in Sigil
$env:SIGIL_GGML_VOICE = "<a voice-library id>"     # optional: clone it
$env:SIGIL_GPU_LOCK = "<scratchpad>\gpu_timing.lock"
cargo test --manifest-path src-tauri/Cargo.toml real_ggml_engine_speaks -- --ignored --nocapture
```

Not verified in P6-F (no downloads in this phase): that a bare voice name resolves through
`--voice-dir` for the omnivoice backend (CrispASR documents it for the server generally and for
qwen3-tts explicitly), and the measured VRAM (the catalog says 2.2 GB, provisional).
