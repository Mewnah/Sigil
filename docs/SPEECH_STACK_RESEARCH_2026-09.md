# Sigil speech stack research, September 2026 (revised)

Working research doc, not synced to `public/docs/`. It compares Sigil's speech-to-text (STT), text-to-speech (TTS), translation and voice-changer backends with what is available now, for the people who actually use Sigil: VRChat streamers on mid-range PCs (RTX 3060/4060-class with 8–12 GB, or CPU-only), running VRChat, OBS and sometimes an AI voice changer at the same time.

**Priorities:** quality first, then local, then free. Paid cloud APIs are covered only in Appendix A. Licenses are listed as plain facts, because they decide one integration question: can the code link into AGPL-3.0 Sigil in-process, or does it need to run as a sidecar process? Nothing is excluded on license grounds.

**How this was researched:**
- **First pass:** three multi-agent research runs with three-vote claim verification.
- **Re-audit:** a second pass without the permissive-license filter. 15 searches found 292 candidates (250 unique). A triage step picked a balanced shortlist, and 106 were profiled from primary sources, each checked by an independent skeptical verifier. Completeness critics then added 16 missed candidates.
- **Code checks:** every Sigil code claim below was checked in this repo.

**Caveats:**
- Almost every quality and speed number is the vendor's own.
- None were measured on an RTX 3060/4060 that was also running VRChat, OBS NVENC and a voice changer.
- Section 8 lists what Sigil has to benchmark itself before choosing defaults.

---

## 0. Short version

1. **Run the models inside Sigil.** Today Kokoro, Chatterbox, Melo, Fish, Moonshine and LibreTranslate are HTTP clients to servers the user has to run, which is the biggest usability problem.
   - Two engines cover most of the recommended models: **sherpa-onnx** (ONNX) and **transcribe.cpp** (ggml, MIT; Rust crate `transcribe-cpp`, already shipping in the competing VRChat app PuriPuly-heart).
   - Build rules: each ggml runtime loads as a **separate DLL**. Each extra ONNX Runtime copy is either built against Sigil's `ort` or runs in a child process.
2. **STT: choose the model by language, and use two passes on GPU.**
   - Nemotron 3.5 Streaming produces the live partial captions.
   - A stronger model re-decodes each finished utterance: **Qwen3-ASR** for Japanese, Korean, Chinese and Russian (the biggest miss last round), **Cohere Transcribe** for English and European languages, **Parakeet Ultra** as a drop-in upgrade for Parakeet v3.
   - Replace the RMS silence gate with **Silero VAD**, and harden Whisper against hallucinated phrases.
3. **TTS:**
   - Kokoro moves in-process, with espeak-ng for phoneme conversion.
   - **Qwen3-TTS** becomes the GPU tier: streaming, voice cloning, 10 languages.
   - **Breeze TTS 2** and **Fish S2 Pro** set the quality ceiling on high-end cards.
   - A Japanese track: AivisSpeech / Style-Bert-VITS2, plus VOICEVOX.
   - GPT-SoVITS for character voices users have already trained.
   - Keyless Edge voices for languages nothing else covers.
4. **Translation:**
   - Add an **LLM translation backend**, run by a bundled `llama-server` helper or in-process llama.cpp.
   - Models: **Hy-MT2** as the default family, **MiLMMT-46-4B** as the 8 GB challenger.
   - **Mozilla Bergamot** as the zero-VRAM CPU fallback, and **NLLB-200** for rare languages.
   - Translate partial captions with a **LocalAgreement** commit policy so the text doesn't flicker.
5. **Voice changer:** the Rust RVC engine stays the core. New work:
   - Import the `.zip`/`.pth` files users actually have.
   - SwiftF0 pitch extraction on CPU.
   - Support the SPIN-v2 and CJK HuBERT embedders and the RefineGAN vocoder.
   - A latency budget that drops stale audio.
   - Real-time audio-thread priority.

   **Seed-VC, X-VC and StreamVoiceAnon** become optional zero-shot sidecars, and **DDSP-SVC** an optional engine for singers.
6. **Features competitors have and Sigil lacks:**
   - Transcribing and translating *other players* through loopback capture.
   - A SteamVR caption overlay.
   - Routing every TTS backend through the voice changer.
   - Syncing with VRChat's mic mute.
   - Authentication on the LAN web server.
   - Following audio-device changes.
7. **Verified bugs in the current code:** see §6.

---

## 1. Speech-to-text

### 1.1 Recommended lineup

| Tier | Pick | Role | Confidence |
|---|---|---|---|
| **CPU-only** | **NVIDIA Nemotron 3.5 ASR Streaming 0.6B** int8 (~650 MB) | Live partials; the final text too for its 19 "transcription-ready" locales. Always pass the language: auto-detect is worse on every language. Mandarin is weak (CER ~20), so route zh elsewhere. | high |
| | **Moondream Parakeet Ultra** int8 (post-trained Parakeet TDT 0.6B v3) | Final pass for 25 European languages, per utterance cut by the VAD. A pure weights swap on the sherpa-onnx transducer path. Two third parties confirm it beats v3 (FLEURS 24-language mean 14.81→11.67; Russian 7.18→5.97). | medium |
| | **Qwen3-ASR-0.6B** int8 (official sherpa-onnx package, 0.94 GB) | Final pass for ja/ko/zh/ru plus 30 languages and 22 Chinese dialects, with automatic language ID. CPU speed is contested (RTF 0.1–0.17 vs a 13 s-per-11 s report), so benchmark it. | medium |
| | **SenseVoiceSmall** int8 (228 MB) | Fast Mandarin and Cantonese (RTF 0.042 on a laptop i5, independent). Also emits emotion and event tags such as [laughs] and [music]. | medium |
| | **Moonshine v2** Small/Medium | English CPU streaming alternative. Choose between it and Nemotron with a local benchmark. | medium |
| **Mid GPU** | **Two-pass captioning** | Nemotron partials on CPU, then a GPU re-decode when VAD or end-of-turn closes the segment. Fits Sigil's existing `onInterim`/`onFinal` contract. Big gain for zh/ja/ru; small for English (~0.6–1.0 WER). | medium |
| | **Qwen3-ASR-1.7B** GGUF Q8_0 (2.19 GB) | CJK/Russian/multilingual final. Strongest independent evidence of any open model: Open ASR 5.76 (confirmed by the maintainers), Microsoft CoreAI's best batch model, best Japanese conversational CER (0.140 vs Whisper-turbo 0.184). Keep Q8_0 (Q4 costs WER). Does not stream. | medium |
| | **Cohere Transcribe 03-2026** Q4_K (1.51 GB) | English/European final. The only independent consumer-GPU measurement: RTX 4060 Ti, RTFx ~32, 7.67% non-CJK WER. **Never route ja/zh/ko to it:** it silently drops Japanese sentences. No Russian, no language ID. | medium |
| | **Confucius4-R2T2** Q8_0 (NetEase streaming fine-tune of Qwen3-ASR-1.7B) | Opt-in true streaming with append-only (flicker-free) en/zh captions. Vendor numbers only: en WER 6.13 vs Nemotron 9.49 at 160 ms. Three weeks old. Benchmark on a 3060/4060 first. | low |
| | **Granite Speech 5.0 470M TurboCTC** | Small English final pass (Open ASR 5.00 self-reported; CTC so it can't loop). Punctuation and casing unconfirmed. | low |
| **High GPU** | **Voxtral Mini 4B Realtime** (Q4_K_M 2.83 GB keeps WER) | Multilingual native streaming, 13 languages including ja/ko/zh. FLEURS 8.72 at 480 ms, 7.70 at 960 ms. | medium |
| | **Qwen3-ASR-1.7B** BF16 alongside R2T2 or Voxtral partials | Both passes on the GPU for 16 GB+ cards. | medium |
| **Zero setup** | Silero VAD v6.2 (bundled, always on) + Nemotron auto-download + **automatic model pick by language** | en→Nemotron or Moonshine; European→Parakeet Ultra final; zh/yue→SenseVoice; ja/ko/ru→Qwen3-ASR-0.6B; offer the two-pass upgrade when a GPU is present. | high |
| **Bring your own** | Anime-Whisper (litagin) ggml | Opt-in Japanese preset for expressive, anime-style voices on the existing `whisper-rs`. Validate on real VRChat mic clips. | low |

**Rejected, with reasons:**
- **Kotoba-Whisper:** last of 10 in an independent Japanese test, CER 0.495.
- **ReazonSpeech-k2 as the Japanese default:** CER 0.445 on conversation.
- **Orukeet:** evidence of benchmark overfitting.
- **Parakeet Redux:** its speed only exists in an unlicensed runtime.
- **Fun-ASR-Nano:** redundant with SenseVoice.
- **Kyutai STT:** en/fr only and not real-time on CPU.
- **Audio8-ASR:** needs 9–10 GB and Linux vLLM.
- **TEN-VAD:** high false-alarm rate.
- **Qwen3-ASR native or chunked streaming:** vLLM-only; chunking raises WER 1.6–1.8×.

### 1.2 Pipeline fixes that matter as much as the models

- **Shared, stateful Silero VAD v6.2 gate** (MIT, ~2 MB, under 1 ms of CPU per 32 ms frame), running on ort.
  - Why the current gating fails:
    - `whisper.rs` gates on RMS (-40 dBFS, 1500 ms silence), so pure-silence buffers still get flushed to Whisper. In an ICASSP 2025 study, 1 s non-speech clips made Whisper hallucinate 52% of the time.
    - The buffer has no length cap under steady noise.
    - The Moonshine/HTTP path (`rustCapturePolling.ts`) has no gate at all.
    - whisper-rs `FullParams::enable_vad` is silently ignored on `WhisperState::full()`, the call path Sigil uses.
  - In the same study, Silero VAD cut WER on silence-padded speech from 104.8% to 8.0%.
  - Settings: hysteresis 0.5/0.35, 300–500 ms pre-roll, 300–700 ms end silence, a forced flush at 15–20 s.
  - Resample with `rubato` (already a dependency) instead of nearest-neighbour decimation.
- **Harden the Whisper path:**
  - `set_suppress_nst(true)`.
  - Drop segments on the no-speech probability alone.
  - A curated per-language list of "thanks for watching"-type phrases, matched against whole segments only. Using the raw Bag-of-Hallucinations list would delete real speech like "let's go".
  - Remove repeats only at 3 or more, using character n-grams for CJK.
  - Cap `max_tokens`.
  - Reuse one `WhisperState`.
- **Smart Turn v3.2** (BSD-2, 8 MB int8 ONNX, 10–20 ms of CPU per pause, 23 languages): an end-of-turn gate that decides when to commit translation, TTS and chatbox output (~600–800 ms instead of a fixed 1.5–2.5 s). Captions keep streaming from partials. An independent test found 35% false cut-offs at a 300 ms budget, so don't use it for instant commits.
- **Optional:**
  - FireRedVAD's music/singing detector to drop music-only segments before Whisper. Untested hypothesis; A/B it on VRChat world music.
  - SenseVoice emotion and event tags as caption tags or VRChat OSC avatar parameters.
- **Keep transcribing the raw mic, before the voice changer.** Evidence from the first pass: voice conversion raises ASR error rates 1.5–4×, and neural denoisers placed in front of Whisper also *raise* WER.

### 1.3 Runtime decision (needs a spike)

- **transcribe.cpp** (MIT) would replace `whisper-rs`, keeping a single ggml copy. It covers Nemotron, Qwen3-ASR, Cohere, Voxtral Realtime, SenseVoice, Granite, Whisper and Moonshine. The catches:
  - Handy ships it with Vulkan only on Windows; the CUDA build is undocumented.
  - It has a single maintainer, and an ABI break at 0.2.0.
- **sherpa-onnx** (Apache-2.0, official Rust crate 1.13.8+) bundles ONNX Runtime 1.28.2 statically. Its `shared` feature can build against Sigil's ORT instead. The prebuilt Windows archives are `/MT` with no DirectML.
- **parakeet-rs** is blocked: `kittentts` pins `ort =2.0.0-rc.11` in every version through 0.4.1.

### 1.4 Free cloud (optional)

- **Gemini 3.5 Transcribe Live**, with the user's own key.
  - Quality: Artificial Analysis measured 2.6% WER non-streaming, the best free streaming option. 85+ languages, custom vocabulary.
  - Limits: sessions stop at 10 minutes, so rotate them with overlap; free-tier quotas are unpublished.
- **Groq whisper-large-v3-turbo** on the free plan: AA-WER 4.6%. It needs VAD segmentation and a rate limiter first (20 requests/min, a 10 s billing minimum per request). The current `rustCapturePolling.ts` posts every 500 ms and would trip the limits within seconds.

---

## 2. Text-to-speech

### 2.1 Recommended lineup

| Tier | Pick | Role | Confidence |
|---|---|---|---|
| **CPU-only** | **Kokoro 82M v1.0**, in-process | Default preset voices (en, ja, zh, es, fr, hi, it, pt-BR). No CPU model ranks higher on independent listener votes (Artificial Analysis open-weights arena Elo 1065, tied with Magpie). Cap threads at 2–4 because VRChat is often CPU-bound. | high |
| | **espeak-ng via `espeak-rs`** | In-process phoneme conversion for Kokoro's non-English voices and for Piper. Serialize calls, since espeak-ng keeps global state. | high |
| | **Kyutai Pocket TTS** | CPU voice cloning only; its preset voices rank below Kokoro. ~30 ms to first audio in PocketTTS.cpp (self-reported). | medium |
| | **Piper voices (piper1-gpl)** on Sigil's ort | Offline fallback for ru, ar and other languages Kokoro lacks. | medium |
| | **AivisSpeech Engine** now, **Style-Bert-VITS2 JP-Extra / AIVMX in-process** later | Expressive Japanese character voices. Independent paper MOS 4.37 vs 4.38 for human recordings. Step 1 is an AivisSpeech preset (port 10101) in the existing VOICEVOX backend. Step 2 vendors `sbv2_core`, which has pure-Rust OpenJTalk and an AIVMX loader. | medium |
| | **VOICEVOX** (keep) | Users pick it for its well-known characters. | low |
| | **Kokoro v1.1-zh** | A second Kokoro model with 100 Mandarin voices. Test mixed zh/en lines. | low |
| **Mid GPU** | **Qwen3-TTS 0.6B/1.7B via `qwentts.cpp`** | The main GPU tier: 3-second cloning, 9 preset voices that take emotion instructions, voice design from a text description (1.7B only), and en/ja/ko/zh/es/pt/ru/de/fr/it. Best independent intelligibility among open models (WER 1.46, SIM 0.715). Measured on an idle RTX 4060: 1.7B at 1.83× real time with 460 ms to first audio; 0.6B at 2.26×. Plain PyTorch manages only 0.23×. Default to 0.6B when the voice changer is running. Needs a clean reference clip: cloning collapses on in-the-wild clips. Its arena rank is unresolved. | medium |
| | **NVIDIA Magpie TTS Multilingual 357M** via NeMo-Speech.cpp | Fixed voices for ko/de/ar/vi at Kokoro-level quality (Elo 1064). The same runtime also serves Nemotron ASR. | medium |
| | **Breeze TTS 2** Q8_0 via Breeze-TTS-2.cpp | Stretch option for 12 GB cards that aren't also running a GPU voice changer. The #1 open model on the arena (1204). ~1.2× real time on an idle 3060. English and Chinese only. | low |
| | **Chatterbox Multilingual V3** (0.5B, ort) | Fallback for in-process cloning if the Qwen3-TTS DLL route fails. Weak on ja/ko. | low |
| | **OmniVoice** via `omnivoice.cpp` | Long-tail languages (646). | low |
| **High GPU** | **Breeze TTS 2** | The quality ceiling for English/Chinese on 16–24 GB cards. | medium |
| | **Fish Audio S2 Pro** (5B) through the existing `fishSpeech` backend | Multilingual quality pick for 24 GB cards or a second PC (#2 on the arena, 1118). ~22 GB peak on the official stack; `s2.cpp` q4_k_m is ~7.2 GB but alpha. | medium |
| | **Qwen3-TTS 1.7B** BF16/Q8_0 incl. VoiceDesign | 4.2× real time and 174 ms to first audio on a 4090. | medium |
| | VoxCPM2 (2B) | 30-language cloning at 48 kHz. Every number is from its authors. | low |
| **Zero setup** | Kokoro bundled in-process + **Edge Read Aloud voices** through a native Rust client | Edge covers any language with no key, but is unreliable (503s, stalls, some locales failing), so always pair it with a local fallback. | high / medium |
| **Bring your own** | **GPT-SoVITS** v2ProPlus/v4 through its `api_v2` server | Voices users have already trained; the de facto VTuber standard. | medium |
| | User AIVMX / Style-Bert-VITS2 voices | File-drop import once the SBV2 path is in-process. | medium |
| | IndexTTS-2.5; Voxtral-4B-TTS through a user-run vLLM server | Niche: emotion/duration control; a documented server target. | low |

**Rejected, with reasons:**
- **Maya1:** reads its own voice description aloud ~40% of the time.
- **ZONOS2:** stalled, and fails to stop in Japanese.
- **Step-Audio-EditX:** 13 minutes for a 25 s clip on a 5090.
- **CosyVoice 3:** its fast accelerators are Linux-only.
- **NeuTTS:** weak cloning.
- **OpenAudio S1-mini:** ranks at or below Kokoro.
- **Higgs TTS 3:** below Kokoro on the arena.

### 2.2 TTS infrastructure work

- **Fix `fishSpeech`:** see §6.
- **Sentence chunking and pipelining in the TTS service.** There is no split or chunk logic today, and `voicevox.ts` sends the whole text in one query. Most backends return whole utterances, so time to first audio depends on splitting at sentence boundaries (including Japanese punctuation) and synthesizing ahead.
- **One generic OpenAI-compatible `/v1/audio/speech` client** with streamed pcm/wav playback, replacing per-model clients:
  - `kokoro_tts.rs` and `chatterbox_tts.rs` hard-code mp3 output.
  - `neural_sidecar_tts.rs` buffers the whole WAV before playing.
  - Any sidecar Sigil starts must bind to `127.0.0.1`.
- **VOICEVOX/AivisSpeech inspector fixes:** see §6.
- **A shared "record a clean 3–10 s reference voice" flow,** with a cached embedding per backend (Qwen3-TTS, Breeze, Pocket, Chatterbox, GPT-SoVITS).

### 2.3 Free cloud (optional)

- **Gemini 3.8 Flash TTS / Flash-Lite TTS** on the user's own free key: arena Elo 1265 / 1239, above every open model, streaming raw PCM. Fall back to Kokoro on a 429 or network error.
- **Fish S2.1 Pro free API** through the existing `fishSpeech` backend: free until 2026-11-30.
- **Uberduck** is still alive. Sigil calls its undocumented legacy endpoints; migrate to `POST /v1/text-to-speech`.

---

## 3. Translation

### 3.1 Recommended lineup

| Tier | Pick | Role | Confidence |
|---|---|---|---|
| **CPU-only** | **Hy-MT2 1.8B** GGUF Q4 (or 1.25-bit, 440 MB) | Quality default on CPU. WMT24++ XCOMET 82.61, ahead of TranslateGemma-4B (75.97). Expect roughly 0.5–1.5 s per caption line on 6–8 cores (unmeasured estimate). | medium |
| | **Mozilla Firefox Translations (Bergamot)** int8 students | Fast offline fallback. Also the automatic fallback when the GPU is busy or the LLM/cloud fails, and a cheap engine for re-translating partials. 17–60 MB per direction, no VRAM. Beats NLLB-600M, OPUS-MT and Argos on en↔ja/ko/zh in Mozilla's evals (en-ja COMET22 0.900). 111 directions, actively retrained. Build: bergamot C++ via FFI with `USE_WASM_COMPATIBLE_SOURCE=ON`. | high |
| | **NLLB-200 distilled 600M/1.3B** (ONNX on the existing ort) | Long tail: ~200 languages, direct any-to-any. ~0.9 GB. VRCT's offline default. Route only languages nothing else covers. | medium |
| **Mid GPU** | **Hy-MT2 7B** Q4 | 12 GB default. On 8 GB cards, only when no RVC voice changer is running. | medium |
| | **MiLMMT-46-4B v1.0** (Xiaomi, Q4_K_M 2.49 GB) | 8 GB challenger. WMT24++ 85.86, roughly level with Hy-MT2-7B at half the size. Self-reported, and scored with its own RL reward metrics. A/B it against Hy-MT2. | medium |
| | **Hy-MT2 1.8B** | Default when the card also runs STT and RVC. | medium |
| | TranslateGemma 4B/12B | Alternative preset. Strong on conversational JA↔EN in an LLM-judge test; behind MiLMMT-4B on WMT24++. | medium |
| | NiuTrans LMT-60-4B | A/B only; its headline scores are optimized for the metric it reports. | low |
| **High GPU** | **MiLMMT-46-12B v1.0** | Highest self-reported open WMT24++ score (87.88). Needs a self-made GGUF. | medium |
| | **Hy-MT2 30B-A3B**; **Seed-X-PPO-7B** | Seed-X has the best reference-based FLORES+ score; it needs a raw prompt ending in a language tag. | medium |
| **Zero setup** | Bergamot downloaded on demand; Hy-MT2 1.8B as the default GGUF once a runtime exists | | high / medium |
| **Bring your own** | JA↔EN specialists: CAT-Translate 3.3B/7B, Shisa V2.1 8B/14B, VNTL. ja→zh-Hans: SakuraLLM/GalTransl. Indic: IndicTrans2. Rare languages: MADLAD-400-3B. Stop-gap: LTEngine through the existing LibreTranslate backend | Opt-in presets. | low |

**Rejected, with reasons:**
- **Tower+:** mid-to-lower pack on EN-JA/KO.
- **Tiny Aya and Aya Expanse:** superseded.
- **Hunyuan-MT-7B:** superseded by Hy-MT2-7B.
- **GemmaX2:** superseded by MiLMMT.
- **OPUS-MT:** weak on CJK.
- **MADLAD 7B/10B:** too large for the gain.
- **SeamlessStreaming, NLLB as a quality default, the full SimulStreaming cascade:** Python/PyTorch, too heavy.

### 3.2 Prerequisites and cheap wins

- **A GGUF runtime is required for every LLM translator.** In-process `llama-cpp-2` has real Windows blockers:
  - its static ggml clashes with `whisper-rs`, which fails at the MSVC link step;
  - GGUF files over 4 GB fail to load on MSVC (#951), which hits 7B Q4 files;
  - builds have crashed on PCs other than the build machine.

  A **`llama-server` helper that Sigil bundles and starts itself** is an equally good first option.
- **A generic OpenAI-compatible translation backend.** Reuse `transform/services/llm.ts`, but raise `max_tokens` above 100 and add streaming, `reasoning_effort`, per-model prompt templates and a raw completion path. This serves the bundled llama-server, Gemini Flash-Lite, OpenRouter and user-run servers.
- **A per-model prompt builder with rolling context and a glossary.** Hy-MT2 documents templates for background context and for terminology; send one block per request. Filter the glossary to terms that appear in the current line.
- **LocalAgreement-2 commit policy for translating partial captions.** Commit the longest common prefix of the last two translations and flush on the STT final. It is model-agnostic and a few hundred lines of code. Pair it with Bergamot or Hy-MT2 1.8B.
- **A language-pair router with a fallback chain:** default LLM → specialists → NLLB/MADLAD → Bergamot, with cloud errors falling back to local.

### 3.3 Free cloud (optional)

- **Gemini Flash-Lite** on the free tier. Reachable today through the transform service's custom provider, but the 100-token cap and missing `reasoning_effort` get in the way.
- **DeepL API Free.** It must be called from Rust because of CORS, and must use the `DeepL-Auth-Key` header.
- **Gemini 3.5 Live Translate** (preview): speech in, translated captions and voice out.

---

## 4. Real-time voice changer

### 4.1 Where the field stands

- **w-okada VCClient** is at v2.2.2-beta and supports Beatrice v2. Its latency is chunk + conversion time, and it stutters whenever conversion takes longer than the chunk.
- **The Deiteris fork** runs the extra-context step on the GPU. Chunk latency on an RTX 3070: 30–38 ms, vs 170–213 ms for prebuilt w-okada. Last build December 2024.
- **Applio** v3.5+ has a real-time tab (MIT).
- **The official RVC WebUI 2.3.260718** now has a real-time CUDA-graph engine and a VST plugin (MIT).
- **Rust prior art:**
  - `vc-rs`: real-time RVC on ONNX Runtime and native TensorRT (MIT).
  - `pthrs`: a pure-Rust reader for `.pth` checkpoints and FAISS indexes (Apache-2.0).
  - `Ramo-Inc/rvc-realtime`: 15.7 ms per 60 ms block on an RTX 3060 Ti. It has no license, so it is only a benchmark target.
- **User pain points:** ~1–1.2 GB of VRAM per RVC model, and VRChat freezing when VRAM runs out. Under GPU contention the official RVC VST builds up ~9 s of stale audio (RVC WebUI issue #2859).

### 4.2 Recommended lineup

| Tier | Pick | Role | Confidence |
|---|---|---|---|
| **Mid GPU** | **Sigil's in-process RVC v2 ONNX engine** (fp16, CUDA/DirectML) | Main engine. Every community v2 `.pth` works. | high |
| **High GPU** | The same engine with 48 kHz models, RMVPE on GPU and smaller blocks; optional TensorRT | vc-rs trtexec: 13.9 ms p95 per 100 ms chunk. Costs: 86–93 s engine rebuilds, and TensorRT/CUDA runtimes to bundle. | high |
| | **X-VC** (sidecar) | Zero-shot from one clip, with the best measured similarity (SIM 0.62 vs Seed-VC 0.40). EN/ZH, 16 kHz, 5 GB, PyTorch. For 12 GB+ cards. | low |
| | **Seed-VC** (GPL-3.0, sidecar) | Zero-shot conversion, singing, and V2 accent conversion/anonymization. ~430–690 ms live. Archived upstream; build on `jiaheguo521/seed-vc-realtime`. | low |
| | **StreamVoiceAnon** (sidecar) | Privacy mode: 237 ms with a 92 ms chunk on an RTX 3060 Laptop. | low |
| | **DDSP-SVC 6.3** | Optional second engine for singers (44.1 kHz, E2–D#7). Upstream already exports `encoder.onnx` + `velocity.onnx`; only the Euler loop needs writing in Rust. | low |
| **CPU-only** | **Beatrice 2**: host the user-installed official VST3 | The best-fitting CPU option: ~50 ms, one CPU thread, no VRAM. **Downgraded** from last round: its runtime `beatrice.lib` is closed (per-project grants), the project has been stuck at release-candidate since August 2025, and nobody has an open ONNX port. | medium |
| | RVC on ort's CPU execution provider with SwiftF0 | Fallback for CPU users who already own RVC voices. Ship it behind a built-in benchmark. | low |
| | MeanVC2 | Experimental zero-shot on CPU. Mandarin-trained, 16 kHz. | low |
| **Zero setup** | RVC engine + ContentVec + RMVPE/FCPE/SwiftF0 ONNX + an MIT default voice (PhoenixStormJr/RVC-V2-default-voice) + **drag-and-drop `.zip`/`.pth` import** | | high |
| **Bring your own** | SPIN-v2 and zh/ja/ko HuBERT embedders; the wok000 `weights_gpl` ONNX pack; so-vits-svc 4.1 generators exported by the user; external voice changers on a virtual cable | | medium / low |

### 4.3 Engineering work, in priority order

1. **Import the checkpoints users actually have**, instead of relying on "bring your own ONNX." Community voices ship as a `.zip` holding a `.pth` and an `added_*.index` (30,000+ on voice-models.com); almost none are ONNX. The importer:
   - reads the pickle with an allow-list (`pthrs`);
   - folds `weight_g`/`weight_v`;
   - injects the tensors into per-config fp16 ONNX templates and caches the result by hash;
   - reads the checkpoint's `embedder_model` / `vocoder` / `version` / `sr` / `f0` metadata to pick templates, and rejects unsupported models clearly;
   - parses untrusted files in a child process or behind size and shape checks, and guards against zip bombs and path traversal.
2. **A latency budget that drops stale audio:** per-block timestamps, 300 ms default, 100–2000 ms range. This prevents the 9-second backlog seen in RVC WebUI issue #2859, and applies to every engine and sidecar.
3. **Audio-thread priority.** `cpal 0.15.2` sets none on WASAPI threads.
   - Register the callback threads with MMCSS, or move to `cpal 0.17.3`.
   - Replace the 2 ms sleep-poll (`voice_changer.rs:1898`) with a condvar.
   - Revisit `THREAD_PRIORITY_BELOW_NORMAL` (`:1311`).
   - Never give the inference worker MMCSS priority.
4. **SwiftF0** (MIT, 135 KB ONNX, ~180× real time on one core): the CPU default for pitch, and an option that frees GPU time. Degraded-audio F1 0.781 vs RMVPE 0.768, self-reported. Run a cents-level A/B against RMVPE first.
5. **Embedder and vocoder coverage:** SPIN-v2 and zh/ja/ko HuBERT (Sigil exports and hosts the ONNX files), and a RefineGAN generator template. Applio-trained RefineGAN voices don't run in w-okada or mainline RVC.
6. **GPU scheduling:**
   - CUDA graphs + IOBinding at fixed block lengths. The current `convert_block` uses a variable `new_frames`, so pad to fixed shapes.
   - A priority CUDA stream (possible on ort rc.11).
   - A high-priority DirectML queue (needs rc.13).
   - Never request admin rights. Note that on RTX 30/40 cards, CUDA kernels running alongside graphics are limited to 48 KiB of shared memory, so idle-GPU benchmarks overstate performance.
7. **One owned mic capture, fanned out:** the raw signal goes to STT, and a conditioned copy goes to the voice changer.
8. **An optional input denoiser on the voice-changer path only.** DPDFNet or DeepFilterNet3 for quality; GTCRN for the lowest CPU.
9. **Check FAISS `.index` retrieval:** compare Sigil's `faiss.rs` against `pthrs`'s implementation of RVC's blend, and parallelize the search.
10. **Surface execution-provider failures:** `ort` silently falls back to CPU unless `.error_on_failure()` is set. Also check CUDA/cuDNN versions at startup.

---

## 5. Features competitors have that Sigil lacks

The profiled competitors were VRCT, PuriPuly-heart / PuriPulyHeart+ (AGPL, so their code can be ported with attribution), Whispering Tiger, Applio / rvc-tts-webui, LiveCaptions-Translator and VRCTextboxSTT. A grep of Sigil found none of the following:

| Feature | Who has it | Notes |
|---|---|---|
| **Transcribe and translate other players** (loopback capture) | VRCT, PuriPuly-heart | Phase 1: `cpal 0.15.2` already opens loopback on output devices. Phase 2: per-process loopback via `wasapi-rs` (Windows 11). Separate pubsub topic, off by default, translated in the reverse direction. |
| **SteamVR caption overlay** | VRCT, PuriPuly-heart (native Rust overlay) | `ovr_overlay` or `openvr_sys`. A quick first step is an always-on-top caption window users pin in XSOverlay or OVR Toolkit. |
| **LLM translation backend** | VRCT, PuriPuly-heart, LiveCaptions-Translator | §3.2. |
| **Automatic STT model by language** | PuriPuly-heart | §1.1. |
| **Route every TTS backend through the voice changer** | Applio, Whispering Tiger | Today only `neuralOnnx`/`neuralSidecar` have `routeThroughVoiceChanger`. Needs one TTS audio sink, the aux-queue fix (§6), and muting the mic while TTS plays. |
| **Sync with VRChat's mic mute** (OSC `MuteSelf`) | PuriPuly-heart (claimed) | Also keeps long silence away from streaming models. |
| **Authenticated LAN web/WebSocket server** | VRCT 3.5.1 | Sigil binds `0.0.0.0` with no auth. A per-session token in the `/client` and `mic.html` URLs. High-risk change; roll it out carefully. |
| **Follow audio-device changes and recover stalled streams** | VRCT | Headset and virtual-cable switches mid-stream. |
| Hotword/keyword biasing | PuriPuly-heart, CrispASR, Granite 4.1 | Whisper initial prompt; Qwen3-ASR system-prompt context. |
| Japanese romaji/hiragana and Chinese pinyin readings | VRCT, PuriPulyHeart+ | |
| Model installer with mirror fallback | PuriPuly-heart | For friends where Hugging Face is slow or blocked. |

---

## 6. Bugs found in the current code (verified in this repo)

| Where | Bug | Fix |
|---|---|---|
| `src/core/services/translation/services/onDevice.ts:23,44` | Checks `typeof Translator/LanguageDetector === "object"`, but those globals are functions, so on-device translation can never start. | Accept `"function"`. Reuse one detector, and `destroy()` replaced translators. |
| `src-tauri/src/services/fish_speech.rs:57,126` | Health check and voice listing call `GET /v1/models`, which fish-speech doesn't expose, so the backend likely never starts against the official server. | Use `/v1/health` and `/v1/references/list`, add streamed WAV, and add an optional Bearer key / `model` header. |
| `src-tauri/src/services/voice_changer.rs:1165–1168` | The aux (TTS injection) queue is capped at 192,000 samples (2 s at 48 kHz stereo) and drains from the front, so longer TTS clips lose their beginning. | Use a proper queue with back-pressure or a larger cap. |
| `src-tauri/src/services/voice_changer.rs:1898, 1311` | A 2 ms sleep-poll, and the worker at `THREAD_PRIORITY_BELOW_NORMAL`. | Condvar wake; revisit the priority (§4.3). |
| `src-tauri/src/services/whisper.rs:356,834` | RMS gate (-40 dBFS / 1500 ms) flushes silence to Whisper; the buffer has no length cap. | Silero VAD gate (§1.2). |
| `src/core/ui/inspector/inspector_tts.tsx:285` | Pitch slider min -0.15, max 0.15, step 0.1, so 0 can't be reached once moved. AivisSpeech degrades with any non-zero pitch. | step 0.01, or add a reset. |
| `src/core/services/tts/services/voicevox.ts:93` | `parseFloat(intonationScale) \|\| 1.0` turns a valid 0 into 1.0. | Use `??` / `Number.isFinite`. |
| `src/core/services/stt/services/deepgram.ts:104` | Sends `tier=` but no `model=`, so it may not be using Nova-3. | Add `model=nova-3`. |
| `src/core/services/stt/services/azure.ts:48–55` | `Speech_SegmentationSilenceTimeoutMs` is never set, which slows phrase finalization. | Expose it with a caption-friendly default. |
| `src-tauri/src/services/uberduck_tts/mod.rs:43,58` | Calls the undocumented legacy endpoints. | Migrate to `/v1/text-to-speech`. |
| `src/core/services/transform/schema.ts:26` | The default `openRouterModel` is the stale `google/gemini-1.5-flash`. | Update the default. |

---

## 7. Prioritized roadmap

**P0: quick fixes (days)**
- Every bug in §6.
- A Silero VAD gate in place of the RMS gate.
- Whisper hallucination hardening.
- An AivisSpeech preset in the VOICEVOX backend.

**P1: in-process speech engine**
- A runtime spike: transcribe.cpp vs sherpa-onnx `shared` next to ort rc.11. Then set the ggml-as-separate-DLL / ORT alignment policy.
- STT: Nemotron 3.5 streaming + Parakeet Ultra + Qwen3-ASR-0.6B/1.7B + SenseVoice, with automatic selection by language and two-pass captions on GPU.
- TTS: Kokoro in-process with espeak-ng; Piper as a fallback; the generic OpenAI-compatible TTS client; sentence chunking.

**P2: translation**
- A bundled `llama-server` helper and an OpenAI-compatible LLM translation backend (Hy-MT2 1.8B/7B; MiLMMT-4B A/B).
- Bergamot as the CPU fallback; NLLB for rare languages.
- The prompt builder, LocalAgreement for partials, and the router with its fallback chain.

**P3: voice changer** (following `VOICE_CHANGER_AI_PLAN.md`, updated with §4.3)
- The `.pth`/`.zip` importer, the latency budget, audio-thread priority, SwiftF0, embedder and RefineGAN coverage, and CUDA graphs.
- Then the optional zero-shot sidecars and DDSP-SVC.

**P4: features**
- Loopback transcription of other players, the SteamVR overlay, TTS→RVC routing for all backends, VRChat mute sync, LAN auth, device-follow.
- Qwen3-TTS via `qwentts.cpp`; Breeze/Fish sidecars; GPT-SoVITS; Edge voices.
- Optional free-cloud backends (Gemini STT/TTS/translate, Groq, DeepL Free).

## 8. What Sigil has to benchmark itself

- **Hardware under load:** every GPU pick on an RTX 3060/4060 **with VRChat + OBS NVENC + the voice changer running**. That covers Qwen3-ASR-1.7B, Cohere, R2T2, Voxtral Realtime Q4, `qwentts.cpp`, Breeze Q8, Hy-MT2-7B, MiLMMT-4B, and the RVC engine's p99 latency. No published number was measured this way.
- **CPU speed:** Qwen3-ASR-0.6B, Hy-MT2 1.8B and Bergamot CJK models on a 6–8 core CPU while VRChat is running.
- **STT accuracy on real audio:** WER on real pre-conversion VRChat mic clips (plus game/music bleed), per backend.
- **Translation quality:** a blind A/B test on real caption logs (casual JA/KO/ZH, ASR errors) using non-reward metrics (chrF/MetricX) plus a quick human check. Most headline translation scores come from the same metrics the models were trained on.
- **Listening tests:** Qwen3-TTS vs Kokoro vs Magpie; X-VC vs Seed-VC vs MeanVC2; SwiftF0 vs RMVPE pitch accuracy in cents.

---

## Appendix A. Paid cloud (low priority)

From the first pass. Artificial Analysis streaming STT leaderboard, English only, read 2026-09-27:

| Service | WER | Time to final | $/hour |
|---|---|---|---|
| ElevenLabs Scribe v2 Realtime | 3.59% | 0.14 s | $0.39 |
| AssemblyAI U3.5 (min-latency) | 4.02% | 0.19 s | $0.45 |
| Soniox v5 | 4.50% | 0.054 s | $0.12 |
| Azure (already integrated) | 5.25% | 0.62 s | ~$1.00 |
| Deepgram Nova-3 (already integrated) | 6.59% | 0.066 s | $0.29 |

Translation: Azure Translator NMT has a permanent 2 M characters/month free tier. Google NMT runs ~100 ms. DeepL's `latency_optimized` mode supports glossaries and custom instructions.

## Appendix B. Primary sources (from the verified profiles)

**STT:**
- Qwen3-ASR-1.7B / Qwen3-ASR-0.6B (plus Qwen3-ForcedAligner-0.6B): https://huggingface.co/Qwen/Qwen3-ASR-1.7B · https://huggingface.co/Qwen/Qwen3-ASR-1.7B/raw/main/README.md
- SenseVoiceSmall (FunAudioLLM / now QwenAudio): https://huggingface.co/FunAudioLLM/SenseVoiceSmall · https://huggingface.co/api/models/FunAudioLLM/SenseVoiceSmall
- Cohere Transcribe (CohereLabs/cohere-transcribe-03-2026), plus the related cohere-transcri: https://huggingface.co/CohereLabs/cohere-transcribe-03-2026 · https://huggingface.co/api/models/CohereLabs/cohere-transcribe-03-2026?expand[]=cardData&expand[]=gated
- Confucius4-R2T2 (NetEase Youdao), a true-streaming fine-tune of Qwen3-ASR-1.7B: https://huggingface.co/netease-youdao/Confucius4-R2T2 · https://huggingface.co/api/models/netease-youdao/Confucius4-R2T2
- Moondream Parakeet Ultra / Parakeet Redux (post-trained NVIDIA parakeet-tdt-0.6b-v3): https://huggingface.co/moondream/parakeet-ultra · https://huggingface.co/moondream/parakeet-ultra/raw/main/config.json
- Fun-ASR-Nano-2512 / Fun-ASR-MLT-Nano-2512 (Alibaba Tongyi / FunAudioLLM): https://huggingface.co/FunAudioLLM/Fun-ASR-Nano-2512 · https://huggingface.co/FunAudioLLM/Fun-ASR-MLT-Nano-2512
- Kyutai STT (stt-1b-en_fr / stt-2.6b-en, Delayed Streams Modeling): https://github.com/kyutai-labs/delayed-streams-modeling · https://huggingface.co/kyutai/stt-1b-en_fr
- ReazonSpeech-k2-v2 (Japanese Zipformer RNN-T, sherpa-onnx package sherpa-onnx-zipformer-ja: https://huggingface.co/reazon-research/reazonspeech-k2-v2 · https://hilab.jp/blog/reazonspeech-v21-setting-a-new-standard-in-japanese-asr
- Kotoba-Whisper v2.x (v2.0 / v2.2 / bilingual-v1.0) + Anime-Whisper (litagin), Japanese Whi: https://huggingface.co/kotoba-tech/kotoba-whisper-v2.2 · https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0
- transcribe.cpp (handy-computer) + official Rust crate `transcribe-cpp`: https://github.com/handy-computer/transcribe.cpp · https://raw.githubusercontent.com/handy-computer/transcribe.cpp/main/LICENSE
- parakeet-rs (altunenes/parakeet-rs): https://github.com/altunenes/parakeet-rs · https://raw.githubusercontent.com/altunenes/parakeet-rs/master/README.md
- Two-pass live captioning: streaming partials, then the VAD/EOU-closed segment is re-decode: https://huggingface.co/nvidia/parakeet-unified-en-0.6b · https://huggingface.co/nvidia/nemotron-speech-streaming-en-0.6b
- VAD-segmented pseudo-streaming with spoken-language-ID routing (sherpa-onnx VAD + offline : https://github.com/k2-fsa/sherpa-onnx · https://github.com/k2-fsa/sherpa-onnx/releases
- FireRedVAD (Stream-VAD, non-streaming VAD, and multi-label AED for speech/singing/music): https://github.com/FireRedTeam/FireRedVAD · https://raw.githubusercontent.com/FireRedTeam/FireRedVAD/main/README.md
- Silero VAD v6.2 (snakers4/silero-vad), in-process via ort ONNX or the whisper.cpp GGML con: https://github.com/snakers4/silero-vad · https://github.com/snakers4/silero-vad/blob/master/LICENSE
- whisper.cpp decode-time hallucination guards (no_speech_thold, suppress_nst, entropy/logpr: https://github.com/ggml-org/whisper.cpp/blob/master/include/whisper.h · https://github.com/ggml-org/whisper.cpp/blob/master/src/whisper.cpp
- Bag of Hallucinations (BoH) + delooping post-filter (DSP-AGH, ICASSP 2025): https://github.com/DSP-AGH/ICASSP2025_Whisper_Hallucination · https://raw.githubusercontent.com/DSP-AGH/ICASSP2025_Whisper_Hallucination/main/BoH.csv
- Smart Turn v3.2 (pipecat-ai), audio-only semantic end-of-turn detector: https://huggingface.co/pipecat-ai/smart-turn-v3 · https://github.com/pipecat-ai/smart-turn
- Gemini 3.5 Transcribe Live (gemini-3.5-transcribe-live): https://ai.google.dev/gemini-api/docs/models/gemini-3.5-transcribe · https://ai.google.dev/gemini-api/docs/pricing
- Groq Whisper large-v3 / large-v3-turbo (GroqCloud free plan): https://console.groq.com/docs/speech-to-text · https://console.groq.com/docs/rate-limits
- sherpa-onnx (official Rust crate `sherpa-onnx` 1.13.8 + 2026 model zoo): https://github.com/k2-fsa/sherpa-onnx · https://github.com/k2-fsa/sherpa-onnx/releases
- Orukeet v0.1.0 (oruk/orukeet), a post-train of NVIDIA Parakeet TDT 0.6B v3 with frozen Gab: https://huggingface.co/oruk/orukeet · https://huggingface.co/oruk/orukeet/tree/main
- IBM Granite Speech 5.0 470M TurboCTC (ibm-granite/granite-speech-5.0-470m-turboctc): https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc · https://huggingface.co/ibm-granite/granite-speech-5.0-470m-turboctc/raw/main/README.md
- SimulStreaming (UFAL, AlignAtt simultaneous policy for Whisper): https://github.com/ufal/SimulStreaming · https://github.com/ufal/SimulStreaming/releases
- Audio8-ASR-Infinite (Edge0/Audio8-ASR-Infinite): https://huggingface.co/Edge0/Audio8-ASR-Infinite · https://huggingface.co/api/models/Edge0/Audio8-ASR-Infinite
- Mistral AI Studio (La Plateforme) Free mode for Voxtral transcription: Voxtral Mini Transc: https://mistral.ai/pricing · https://mistral.ai/pricing/api/

**TTS:**
- Qwen3-TTS (12Hz 0.6B / 1.7B: Base, CustomVoice, VoiceDesign): https://github.com/QwenLM/Qwen3-TTS · https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/LICENSE
- NVIDIA Magpie TTS Multilingual 357M (v2607): https://huggingface.co/nvidia/magpie_tts_multilingual_357m · https://huggingface.co/nvidia/magpie_tts_multilingual_357m/raw/main/README.md
- Chatterbox Multilingual V3 (Resemble AI, 0.5B Llama-backbone T3 + S3Gen decoder), plus its: https://huggingface.co/ResembleAI/chatterbox · https://huggingface.co/ResembleAI/chatterbox/raw/main/README.md
- AivisSpeech Engine: https://github.com/Aivis-Project/AivisSpeech-Engine · https://raw.githubusercontent.com/Aivis-Project/AivisSpeech-Engine/master/README.md
- GPT-SoVITS (v2ProPlus / v4): https://github.com/RVC-Boss/GPT-SoVITS · https://huggingface.co/lj1995/GPT-SoVITS
- VOICEVOX CORE (voicevox_core, in-process Rust/C library): https://github.com/VOICEVOX/voicevox_core · https://github.com/VOICEVOX/voicevox_core/releases
- Fish Audio S2 Pro (5B) / OpenAudio S1-mini (0.5B), plus the hosted S2.1 Pro Free API: https://huggingface.co/fishaudio/s2-pro · https://huggingface.co/fishaudio/openaudio-s1-mini
- Breeze TTS 2 (BreezeBlue, 3B): https://huggingface.co/BreezeBlue/Breeze-TTS-2 · https://huggingface.co/BreezeBlue/Breeze-TTS-2/blob/main/LICENSE
- Voxtral-4B-TTS-2603 (Mistral Voxtral TTS, open weights): https://huggingface.co/mistralai/Voxtral-4B-TTS-2603 · https://arxiv.org/abs/2603.25551
- Step-Audio-EditX (StepFun, 3B): https://huggingface.co/stepfun-ai/Step-Audio-EditX · https://github.com/stepfun-ai/Step-Audio-EditX
- ZONOS2 (Zyphra) 8B-total / 0.9B-active MoE TTS, with the zonos2.cpp GGUF runtime: https://huggingface.co/Zyphra/ZONOS2 · https://www.zyphra.com/our-work/zonos2
- Kyutai Pocket TTS (english_2026-04 refresh + multilingual 6L/24L variants + community mode: https://github.com/kyutai-labs/pocket-tts · https://huggingface.co/kyutai/pocket-tts
- Kokoro-82M-v1.1-zh (sherpa-onnx package: kokoro-multi-lang-v1_1): https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh · https://k2-fsa.github.io/sherpa/onnx/tts/pretrained_models/kokoro.html
- espeak-ng in-process G2P + Piper (OHF-Voice/piper1-gpl, libpiper): https://github.com/OHF-Voice/piper1-gpl · https://github.com/OHF-Voice/piper1-gpl/tree/main/libpiper
- NeuTTS Air / NeuTTS Nano / NeuTTS-2E (Neuphonic): https://github.com/neuphonic/neutts · https://huggingface.co/neuphonic/neutts-air
- Microsoft Edge Read Aloud neural voices (edge-tts protocol; msedge-tts Rust crate): https://github.com/rany2/edge-tts · https://pypi.org/project/edge-tts/
- IndexTTS-2.5 / IndexTTS2 (bilibili IndexTeam): https://huggingface.co/IndexTeam/IndexTTS-2.5 · https://arxiv.org/abs/2601.03888
- VoxCPM2 (2B) / VoxCPM1.5 (0.6B) - OpenBMB: https://github.com/OpenBMB/VoxCPM · https://huggingface.co/openbmb/VoxCPM2
- Fun-CosyVoice3-0.5B-2512 (Alibaba FunAudioLLM CosyVoice 3): https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512 · https://github.com/FunAudioLLM/CosyVoice
- OmniVoice (k2-fsa / Xiaomi): https://huggingface.co/k2-fsa/OmniVoice · https://github.com/k2-fsa/OmniVoice
- Maya1 (Maya Research, 3B expressive TTS with voice design): https://huggingface.co/maya-research/maya1 · https://huggingface.co/maya-research/maya1/commits/main
- Gemini 3.8 Flash TTS / Gemini 3.8 Flash-Lite TTS (Gemini API free tier): https://ai.google.dev/gemini-api/docs/pricing · https://ai.google.dev/gemini-api/docs/speech-generation
- Higgs TTS 3 (Higgs Audio v3 TTS 4B), Boson AI - HF repo bosonai/higgs-audio-v3-tts-4b: https://huggingface.co/bosonai/higgs-audio-v3-tts-4b · https://huggingface.co/bosonai/higgs-audio-v3-tts-4b/blob/main/LICENSE
- Style-Bert-VITS2 (litagin02), including JP-Extra, plus SBV2-ONNX / AIVMX (Aivis Project) p: https://github.com/litagin02/Style-Bert-VITS2 · https://raw.githubusercontent.com/litagin02/Style-Bert-VITS2/master/docs/CHANGELOG.md

**Translation:**
- Seed-X-PPO-7B (ByteDance Seed): https://huggingface.co/ByteDance-Seed/Seed-X-PPO-7B · https://huggingface.co/api/models/ByteDance-Seed/Seed-X-PPO-7B
- CAT-Translate (CyberAgent) 0.8B / 1.4B / 3.3B / 7B, JA<->EN translation LLMs: https://huggingface.co/cyberagent/CAT-Translate-1.4b · https://huggingface.co/api/models/cyberagent/CAT-Translate-1.4b
- Firefox Translations models (Mozilla Bergamot / Marian int8 student models) + bergamot-tra: https://github.com/mozilla/translations · https://raw.githubusercontent.com/mozilla/translations/main/README.md
- NiuTrans LMT-60 (0.6B / 1.7B / 4B / 8B): https://huggingface.co/NiuTrans/LMT-60-4B · https://arxiv.org/html/2511.07003
- MiLMMT-46-v1.0 (Xiaomi; 1B / 4B / 12B; successor to GemmaX2-28): https://huggingface.co/xiaomi-research/MiLMMT-46-4B-v1.0 · https://huggingface.co/collections/xiaomi-research/milmmt-46
- MADLAD-400-3B-MT (also MADLAD-400-7B-MT and 10B-MT, which the paper calls 7.2B and 10.7B): https://huggingface.co/google/madlad400-3b-mt · https://ar5iv.labs.arxiv.org/html/2309.04662
- NLLB-200 distilled 600M / distilled 1.3B (Meta), via CTranslate2 int8 or ONNX: https://huggingface.co/facebook/nllb-200-distilled-600M · https://github.com/facebookresearch/fairseq/tree/nllb
- Tower+ (Tower-Plus-9B / Tower-Plus-2B / Tower-Plus-72B), Unbabel; supersedes TowerInstruct: https://huggingface.co/Unbabel/Tower-Plus-9B · https://huggingface.co/Unbabel/Tower-Plus-2B
- Cohere Labs Tiny Aya (Global / Water, 3.35B), plus Aya Expanse 8B/32B as the larger siblin: https://huggingface.co/CohereLabs/tiny-aya-global · https://huggingface.co/CohereLabs/tiny-aya-global-GGUF
- Shisa V2.1 (lfm2-1.2b, llama3.2-3b, qwen3-8b, unphi4-14b, llama3.3-70b) + Shisa V2 mistral: https://huggingface.co/shisa-ai/shisa-v2.1-qwen3-8b · https://blog.shisa.ai/posts/shisa-v2.1/
- SakuraLLM family: GalTransl-v4-4B-2601, Sakura-GalTransl-7B-v3.7, Sakura-GalTransl-14B-v3.: https://huggingface.co/SakuraLLM/Sakura-GalTransl-7B-v3.7 · https://github.com/SakuraLLM/SakuraLLM
- VNTL models (lmg-anon vntl-llama3-8b-v2, vntl-gemma2-2b; also vntl-gemma2-27b): https://huggingface.co/lmg-anon/vntl-llama3-8b-v2-hf · https://huggingface.co/lmg-anon/vntl-llama3-8b-v2-gguf
- Rolling-context + glossary prompting for chat-line MT (shared prompt builder for Hy-MT2 / : https://github.com/Tencent-Hunyuan/Hy-MT2 · https://raw.githubusercontent.com/Tencent-Hunyuan/Hy-MT2/main/README.md
- OPUS-MT tc-big (Tatoeba-Challenge) bilingual Marian models via CTranslate2 int8, plus tc-b: https://huggingface.co/Helsinki-NLP/opus-mt-tc-bible-big-mul-mul · https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-en-pt
- IndicTrans2 (AI4Bharat): 1B En-Indic / Indic-En / Indic-Indic, plus distilled 200M En-Indi: https://github.com/AI4Bharat/IndicTrans2 · https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M
- LTEngine (LibreTranslate/LTEngine): local LLM machine-translation server, LibreTranslate-c: https://github.com/LibreTranslate/LTEngine · https://raw.githubusercontent.com/LibreTranslate/LTEngine/main/README.md
- Gemini 3.5 Live Translate (gemini-3.5-live-translate-preview): https://ai.google.dev/gemini-api/docs/live-api/live-translate · https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-live-3-5-translate/
- Gemini Flash-Lite free tier (gemini-3.5-flash-lite / 3.1-flash-lite / 2.5-flash-lite) as a: https://ai.google.dev/gemini-api/docs/pricing · https://ai.google.dev/gemini-api/docs/rate-limits
- llama-cpp-2 / llama-cpp-sys-2 (utilityai/llama-cpp-rs, Rust bindings to llama.cpp): https://github.com/utilityai/llama-cpp-rs · https://crates.io/api/v1/crates/llama-cpp-sys-2
- Simultaneous-translation commit policies: LocalAgreement / AlignAtt (UFAL SimulStreaming r: https://github.com/ufal/SimulStreaming · https://raw.githubusercontent.com/ufal/SimulStreaming/main/README.md
- ct2rs (ctranslate2-rs): Rust bindings for OpenNMT CTranslate2: https://github.com/jkawamoto/ctranslate2-rs · https://crates.io/api/v1/crates/ct2rs
- DeepL API Free: https://developers.deepl.com/docs/getting-started/intro · https://developers.deepl.com/llms.txt

**Voice changer:**
- MeanVC2 (MeanVC 2, ASLP-lab / Northwestern Polytechnical University): https://github.com/ASLP-lab/MeanVC2 · https://raw.githubusercontent.com/ASLP-lab/MeanVC2/main/README.md
- Beatrice 2 (Project Beatrice): trainer 2.0.0-rc.0, official VST 2.0.0-rc.3: https://huggingface.co/fierce-cats/beatrice-trainer · https://huggingface.co/fierce-cats/beatrice-trainer/blob/main/README.md
- RVC WebUI 2.3.260718 (RVC-Project mainline: realtime GUI with CUDA Graphs, FCPE, and the o: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/releases/tag/2.3.260718 · https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI/blob/main/docs/en/Changelog_EN.md
- vc-rs (shirohata) - Rust real-time RVC voice changer (GUI, CLI, VST3; TensorRT / Windows M: https://github.com/shirohata/vc-rs · https://github.com/shirohata/vc-rs/releases
- pthrs (TamKungZ/pthrs), a pure-Rust reader for PyTorch .pth checkpoints and FAISS IndexIVF: https://github.com/TamKungZ/pthrs · https://raw.githubusercontent.com/TamKungZ/pthrs/main/README.md
- RVC community model import path (.zip of .pth + .index -> local fp16 ONNX): https://voice-models.com/ · https://docs.aihub.gg/essentials/voice-models/
- SPIN / SPIN-v2 embedders plus Chinese/Japanese/Korean HuBERT content encoders for RVC: https://github.com/vectominist/spin · https://arxiv.org/abs/2305.11072
- Steinberg ASIO SDK 2.3.4 (dual-licensed, GPL-3.0 option), used through cpal's `asio` featu: https://github.com/obsproject/obs-studio/pull/12733 · https://github.com/RustAudio/cpal/issues/1299
- Windows 11 QoS opt-out (HighQoS via SetProcessInformation/SetThreadInformation ProcessPowe: https://learn.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-setprocessinformation · https://learn.microsoft.com/en-us/windows/win32/procthread/quality-of-service
- Seed-VC (Plachtaa) - real-time 'seed-uvit-tat-xlsr-tiny', offline/singing DiT models, and : https://github.com/Plachtaa/seed-vc · https://raw.githubusercontent.com/Plachtaa/seed-vc/main/README.md
- X-VC (Zero-shot Streaming Voice Conversion in Codec Space): https://github.com/Jerrister/X-VC · https://huggingface.co/chenxie95/X-VC
- GTCRN (Gated Temporal Convolutional Recurrent Network), input denoiser for the voice-chang: https://github.com/Xiaobin-Rong/gtcrn · https://github.com/Xiaobin-Rong/gtcrn/tree/main/stream
- DDSP-SVC 6.x (yxlllc; DDSP + rectified-flow cascade singing/speech voice conversion): https://github.com/yxlllc/DDSP-SVC · https://raw.githubusercontent.com/yxlllc/DDSP-SVC/master/README.md
- Applio 3.6.x Realtime (IAHispano): https://github.com/IAHispano/Applio · https://github.com/IAHispano/Applio/releases
- RefineGAN / MRF-HiFi-GAN / RingFormer RVC generator (vocoder) variants: https://github.com/IAHispano/Applio/tree/main/rvc/lib/algorithm/generators · https://raw.githubusercontent.com/IAHispano/Applio/main/rvc/lib/algorithm/generators/refinegan.py
- SwiftF0: https://github.com/lars76/swift-f0 · https://arxiv.org/abs/2508.18440
- w-okada VCClient v2 + wok000/weights_gpl ONNX pack (with deiteris / tg-develop RVC forks a: https://github.com/w-okada/voice-changer · https://github.com/w-okada/voice-changer/releases
- Ramo-Inc rvc-realtime: https://github.com/Ramo-Inc/rvc-realtime · https://raw.githubusercontent.com/Ramo-Inc/rvc-realtime/main/README.md
- PESTO (Pitch Estimation with Self-supervised Transposition-equivariant Objective), streami: https://github.com/SonyCSLParis/pesto · https://raw.githubusercontent.com/SonyCSLParis/pesto/master/README.md
- GPU queue/stream priority for ONNX Runtime (CUDA user_compute_stream + DirectML DML1 custo: https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html · https://onnxruntime.ai/docs/execution-providers/DirectML-ExecutionProvider.html
- so-vits-svc 4.1 (svc-develop-team) / so-vits-svc-fork (voicepaw): https://github.com/svc-develop-team/so-vits-svc · https://github.com/svc-develop-team/so-vits-svc/releases
- StreamVoiceAnon / StreamVoiceAnon+ (Kuzmin, Liu, Lee, Chng; NTU / A*STAR I2R / PolyU; repo: https://github.com/Plachtaa/StreamVoiceAnon · https://raw.githubusercontent.com/Plachtaa/StreamVoiceAnon/main/README.md

**Competitors / cross-area:**
- ct2rs (Rust bindings for OpenNMT CTranslate2): https://github.com/jkawamoto/ctranslate2-rs · https://raw.githubusercontent.com/jkawamoto/ctranslate2-rs/main/README.md
- CrispASR: https://github.com/CrispStrobe/CrispASR · https://raw.githubusercontent.com/CrispStrobe/CrispASR/main/README.md
- NeMo-Speech.cpp (NVIDIA native C++/ggml runtime for Nemotron/Parakeet/Magpie/Riva Translat: https://github.com/NVIDIA/NeMo-Speech.cpp · https://raw.githubusercontent.com/NVIDIA/NeMo-Speech.cpp/main/README.md
- Windows ML (Windows-managed ONNX Runtime with on-demand IHV execution providers); DirectML: https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/overview · https://learn.microsoft.com/en-us/windows/ai/new-windows-ml/supported-execution-providers
- TTS-to-RVC voice chaining (route every Sigil TTS backend through the voice changer / RVC e: https://github.com/Sharrnah/whispering-plugins/blob/main/Documentation/Speech-Conversion-Plugins/RVC-Plugin.md · https://github.com/litagin02/rvc-tts-webui
- Chrome/Edge built-in AI through Sigil's browser mic page (Translator API, LanguageDetector: https://developer.chrome.com/docs/ai/translator-api · https://blogs.windows.com/msedgedev/2026/06/02/expanding-on-device-ai-in-microsoft-edge-new-models-and-apis-for-the-web/
- Speaker/loopback transcription + SteamVR caption overlay (VRCT feature set): https://github.com/misyaguziya/VRCT · https://github.com/misyaguziya/VRCT/releases
- IBM Granite Speech 4.1 2B (+ -plus, -nar variants) and granite-4.0-1b-speech: https://huggingface.co/ibm-granite/granite-speech-4.1-2b · https://huggingface.co/ibm-granite/granite-speech-4.1-2b-plus
- PuriPulyHeart+ (Sappha1 fork of kapitalismho/PuriPuly-heart): https://github.com/Sappha1/PuriPuly-heart-plus · https://github.com/Sappha1/PuriPuly-heart-plus/releases
- CTranslate2 in-process via Rust crate ct2rs (ctranslate2-rs), the faster-whisper engine: https://github.com/jkawamoto/ctranslate2-rs · https://github.com/jkawamoto/ctranslate2-rs/releases
- audio.cpp (0xShug0/audio.cpp) + audio-cpp/audio.cpp-gguf: https://github.com/0xShug0/audio.cpp · https://raw.githubusercontent.com/0xShug0/audio.cpp/main/README.md
- TTS -> RVC character-voice chaining (TTS output routed through Sigil's own RVC voice-chang: https://github.com/IAHispano/Applio · https://raw.githubusercontent.com/IAHispano/Applio/main/tabs/tts/tts.py

**First-pass sources** (still relevant): arxiv.org/abs/2510.06961 (Open ASR Leaderboard) · artificialanalysis.ai/speech-to-text/streaming · artificialanalysis.ai/text-to-speech/leaderboard/provider-voice/open-weights · github.com/Plachtaa/seed-vc/blob/main/EVAL.md (ASR on converted speech) · arxiv.org/html/2403.07937 (Speech Robust Bench) · arxiv.org/abs/2607.24958 and arxiv.org/abs/2603.04710 (denoisers raise Whisper WER) · docs.aihub.gg/realtime-voice-changer/local/deiteris-w-okada-fork · huggingface.co/fierce-cats/beatrice-trainer · github.com/Ramo-Inc/rvc-realtime · docs.nvidia.com/deeplearning/tensorrt-rtx/latest/inference-library/compute-graphics.html · ort.pyke.io/perf/execution-providers
