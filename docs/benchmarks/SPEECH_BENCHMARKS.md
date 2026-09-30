# Speech benchmarks

Sigil's own speech measurements. Defaults and routing come from these numbers (speech-stack plan,
Principles and §8), not from vendor figures. This page starts with the P2 dev-rig numbers; P7-A's
campaign (word error rates per language, CPU speed on H1 / H3 rigs, GPU under load) replaces and
extends them.

**Dev rig:** Ryzen 7 7800X3D (8 cores), RTX 4090 24 GB (driver 616.56), Windows 11. Unless noted,
timings come from the debug `sigil.exe` with the release runtime DLLs (sherpa-onnx 1.13.8,
transcribe.cpp 0.2.4), median of 3 runs, 2026-09-29. RTF = processing time / audio time (lower is
faster; below 1 is faster than real time). Raw reports live in the git-ignored `bench-results/`.

## Speech-to-text on the CPU (sherpa-onnx)

`Sigil.exe --headless engine-probe sherpa <catalog id> <clip> -- repeat=3 lang=<code> threads=<n>`
(P2-B; 55 runs, all deterministic; `bench-results/sherpa/p2b-20260929-161048/summary.md`).

| Model | Threads | RTF | Notes |
|---|---|---|---|
| Nemotron 3.5 streaming 560 ms (default live model) | 1 | 0.30–0.35 (en, ja, zh, ru, de) | first partial 1.1–1.7 s into speech; 2 threads: 0.22 at 2.5 cores |
| Nemotron 3.5 streaming 160 ms | 1 | 0.80–0.87 | first partial 0.6–1.4 s; too heavy for 6-core PCs |
| Parakeet Ultra int8 / Parakeet TDT v3 int8 | 2 | 0.063–0.075 | en / de / es / fr / ru; identical transcripts except one Spanish word each way |
| Qwen3-ASR 0.6B int8 | 2 | 0.16–0.27 | en / ja / ko / ru / zh |
| SenseVoice small int8 | 2 | 0.029–0.041 | en / ja / ko / yue / zh, with lang / emotion / event tags |
| Moonshine base-en int8 | 2 / 1 | 0.017 / 0.026 | English |
| Whisper-base language ID | 2 | 0.026–0.043 | named en / ja / zh / ru / de / es correctly |

Finals through the speech pipeline (real Silero VAD), time from the end of speech to the final:
Moonshine 132 ms, Nemotron 161 ms, SenseVoice 190 ms, Parakeet 295 ms, Qwen3-ASR 1.2 s.

Utterance engines on 2 threads spin 20–75 ms of CPU per second for about 1 s after each decode
(ONNX Runtime's spin-wait cannot be turned off through sherpa's C API); streaming stays on 1 thread
and does not spin.

## Speech-to-text on the GPU (transcribe.cpp, CUDA 13)

`cargo test speech::engines::transcribe -- --ignored --test-threads=1` with the P2-D model store
(P2-D). VRAM is the process VRAM including the ~0.38 GB CUDA context.

| Model | VRAM | Result |
|---|---|---|
| Qwen3-ASR 1.7B Q8_0 | 3470–3494 MiB | ja-short in 245 ms; a 28 s clip split and decoded in 600 ms; RTF 0.023 |
| Qwen3-ASR 1.7B BF16 | 5914 MiB | jfk in 226 ms |
| Cohere Transcribe Q4_K_M | 1950 MiB | jfk in 293–306 ms; refuses ja; needs a language |
| Granite Speech 5.0 TurboCTC Q8_0 | 890 MiB | jfk in 56–68 ms; lower-case, no punctuation (never a caption final) |
| Voxtral Mini 4B Realtime Q4_K_M | ~4100 MiB | 9–10 streaming partials per clip; append-only partials never retract |

**Two-pass (P2 exit):** Japanese, Qwen3-ASR 1.7B Q8_0 final through the real two-pass queue: 198
and 204 ms after the utterance closed on an 8 s clip (target ≤ 800 ms). 380 MiB of CUDA context stay
after the last model is dropped. ORT's CUDA execution provider and ggml-cuda share one CUDA context
(canary before 509 ms, after 4 ms).

## Smart Turn v3.2

About 31 ms per prediction on 2 threads (model), window plus log-mel features 1.6 ms in a release
build; probabilities match onnxruntime + transformers within ±0.02.

## Text-to-speech on the CPU

`cargo test local_tts -- --include-ignored --test-threads=1` (P2-H; re-run at P2 integration on an
otherwise idle PC). Kokoro v1.0 fp32, 4 threads, sherpa-onnx and Sigil's ORT interleaved sentence by
sentence (3.9–5.2 s sentences):

| Kokoro runtime | Median RTF | CPU s per audio s | Notes |
|---|---|---|---|
| Sigil's ORT with the CPU memory arena (default since P2-I0) | 0.294 | 0.90–1.00 | new and repeated sentence lengths alike |
| inside sherpa-onnx | 0.315–0.319 | 1.17–1.26 | |
| Sigil's ORT without the arena (P2-H) | 0.29–0.40 on new lengths, 0.23–0.24 repeated | 0.93–1.09 | ort rc.13 default |

- First audio of a long first sentence with the first-clause split: 656–679 ms (its opening clause
  is about 2 s of audio); load plus warm-up 1.4 s (sherpa) / 2.8 s (Sigil's ORT, once per model).
- Kokoro int8 through sherpa: RTF about 1.0–1.3 (not catalogued).
- Piper ru (irina medium): RTF 0.04 idle, 0.15–0.16 under load. Cancel takes effect in about 0.6 s.
- Intelligibility (SenseVoice transcripts): English and Mandarin exact; Japanese kana correct,
  kanji read with Mandarin readings (sherpa-onnx 1.13.8 frontend).

## OpenAI-compatible TTS streaming

Against `scripts/neural-sidecar/tts_sidecar.py --chunk-delay 0.2` (P2-I): first audio after about
205 ms, full response after about 2.84 s, for both PCM and WAV (`openai_speech_live_server_streams`).

## Reproducing

- sherpa engines: `SIGIL_APP_DATA_DIR=<store with the models> SIGIL_TEST_MODELS_DIR=<dev cache>
  cargo test --manifest-path src-tauri/Cargo.toml sherpa -- --include-ignored --test-threads=1`;
  the probe matrix script lives in `scripts/runtime-packs/sherpa/`.
- GPU tier: `SIGIL_APP_DATA_DIR=<store> SIGIL_TEST_MODELS_DIR=<dev cache> SIGIL_STT_WORKER_EXE=<debug
  sigil.exe copy with the ORT provider DLLs> cargo test speech::engines::transcribe -- --ignored
  --test-threads=1`.
- Whisper through the pipeline: `SIGIL_TEST_MODELS_DIR=<folder with ggml-tiny.en.bin> cargo test
  real_model -- --ignored`.
- Local TTS: `SIGIL_APP_DATA_DIR=<store> SIGIL_TEST_MODELS_DIR=<dev cache> cargo test local_tts --
  --include-ignored --test-threads=1 --nocapture`.
