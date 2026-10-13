# Operator reference

Detailed reference for **choosing engines for your PC**, **STT/TTS backends**, **reliability**, **voice changer** behavior, the **streamer features**, **interface size and the window** and **local ONNX speech packages**. User-facing walkthroughs live in the [user manual](manual/README.md).

## Table of contents

- [Choosing engines for your PC](#choosing-engines-for-your-pc)
- [STT / TTS backend minimum configuration](#stt--tts-backend-minimum-configuration)
- [Remote client](#remote-client)
- [Speech on this PC (automatic STT and local voices)](#speech-on-this-pc-automatic-stt-and-local-voices)
- [Models & Runtimes](#models--runtimes)
- [STT reliability and troubleshooting](#stt-reliability-and-troubleshooting)
- [Voice changer](#voice-changer)
- [GPU acceleration, diagnostics and logs](#gpu-acceleration-diagnostics-and-logs)
- [Text-to-speech behaviour](#text-to-speech-behaviour)
- [Translation and Rewrite](#translation-and-rewrite)
- [Streamer features and extended voices](#streamer-features-and-extended-voices)
- [Local web server and LAN pages](#local-web-server-and-lan-pages)
- [Interface size and the window](#interface-size-and-the-window)
- [Local speech package (ONNX)](#local-speech-package-onnx)
- [Sidecar scripts (repo)](#sidecar-scripts-repo)

---

## Choosing engines for your PC

Sigil is built for VRChat streamers who run captions into OBS while the game is running. Its defaults are **local, CPU-first and light**: no GPU is needed, the voice detector keeps silence away from every engine that runs in Sigil's own speech pipeline (built-in, Whisper, local HTTP, Groq; Azure, Deepgram, Gemini Live, Vosk and Web Speech do not use it), and models download only after you agree. The heavier engines (the GPU speech models, the larger translation models, voices that run on the GPU) are **opt-in**: switch them on when you have the hardware and have checked that the game still runs well.

This section says what to pick for your PC and what each choice costs; the detail of every engine follows in the sections after it. Numbers are Sigil's own measurements on the dev rig (Ryzen 7 7800X3D, RTX 4090 24 GB), taken outside VR with the benchmark's game-load simulator standing in for a running game ([benchmarks/SPEECH_BENCHMARKS.md](benchmarks/SPEECH_BENCHMARKS.md), [benchmarks/VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md)). A slower processor needs proportionally more time, so check your own PC: **Benchmark this PC** in the voice changer, and Task Manager with the game running. RTF is processing time divided by audio time (below 1 is faster than real time).

**How to read the numbers.** Speech-to-text error rates are for **your own microphone** (clean speech plus a reverberant room and a poor headset link): the case where headphones keep the game out of the microphone. The "other players' captions" feature hears VRChat's own audio, and its figures (music, other talkers, the game mix) are in the [bleed tables](benchmarks/SPEECH_BENCHMARKS.md#bleed-the-vrchat-mix-music-and-babble-other-players-captions-speaker-setups). The corpus is read speech with simulated rooms and bleed, 30 utterances per cell, so differences of a point or two are noise, and nothing ran inside VR.

### Pick your tier

| Tier | Typical PC | What Sigil does by default | What you can switch on |
|---|---|---|---|
| **CPU only** | No NVIDIA GPU, an old GPU, or a gaming PC whose GPU must stay free for the game | Speech-to-text and local voices run on the processor (**Hardware → Processor only** pins speech-to-text there). With fewer than 6 physical cores the automatic speech-to-text uses one pass (the route preview says so). **Translation is the exception:** the default language model (**Runs on → Automatic**) loads on the graphics card when one has room (1.5–1.8 GB of video memory); to keep it off the card choose **Runs on → Processor** (1.5 cores for final lines), Firefox Translations or NLLB, see [Translation by tier](#translation-by-tier). While the language model runs on the processor, English with French, Spanish, Portuguese and Dutch goes to NLLB (beam 4) first when it is installed. The voice changer's default preset is **Streamer** (CPU) | Nothing needed: this is the default route |
| **Mid-range GPU** (8–12 GB, for example RTX 3060 / 4060) | A gaming PC that renders VRChat on the same card | Speech-to-text stays on the processor unless the GPU speech runtime pack is installed and a GPU model fits next to everything else; while the voice changer runs on the GPU, 8–12 GB cards are kept for it | The voice changer's **Low latency** preset (CUDA with the NVIDIA GPU runtime pack, else DirectML); a GPU final model (**Hardware → Graphics card**); a bigger translation model when the plan box says it fits next to the voice changer |
| **High-end GPU** (16 GB or more) | A strong PC, or a second PC that does the streaming | The GPU tier of the automatic route may share the card with the voice changer | GPU speech and translation models, the voice changer's **Quality** / **Studio-TRT** presets, the GPU voices |

The tier names are this guide's shorthand: the speech-to-text inspector's **Hardware** setting offers **Best for my PC**, **Processor only** and **Graphics card**. The shipped **Best for my PC** route is the processor route unless you installed the GPU speech runtime pack and a GPU model that fits next to everything else; it downloads nothing without asking. The heavy GPU models (Cohere Transcribe, Qwen3-ASR 1.7B) are **opt-in and were not benchmarked in P7**: whether one costs the game more than it gives on a mid-range card is unmeasured. If frame rate comes first, set **Hardware → Processor only**.

### Speech-to-text by tier

Pick **Built-in (on your PC)** unless you have a reason not to: it chooses the models for your language and PC (see [Speech on this PC](#speech-on-this-pc-automatic-stt-and-local-voices)). The table is what each model costs; the footprint columns are measured under the game-load simulator.

| Model | Role | Download | Speed (RTF, dev rig CPU) | Error rate, own microphone (WER; CER for Japanese, Korean, Chinese) | CPU and memory next to a game |
|---|---|---|---|---|---|
| Nemotron 3.5 streaming 560 ms | Live partials; its own finals for Arabic, Hindi, Turkish, Norwegian | 475 MB | 0.30–0.35 on 1 thread; first partial 1.1–1.7 s into speech | English 10.3 (9.4 on clean speech); other languages 8.7–15.8 on clean speech (12 utterances each) | 0.31 cores idle, 0.44 next to a game, while speaking; about 0.8 GB |
| Parakeet Ultra (Parakeet TDT v3 as fallback) | Accurate finals, 25 European languages | 630 MB (487 MB) | 0.063–0.075 on 2 threads | English 6.2, Spanish 7.2, Russian 7.3, French 8.5, Portuguese 9.1, German 10.1 | 0.18 cores while a sentence decodes (182 CPU ms per second of audio); 0.9 GB |
| Qwen3-ASR 0.6B | Finals for Korean and the 30 languages nothing else covers; detects the language | 879 MB | 0.16–0.27 on 2 threads | Korean 5.1, Chinese 3.8, English 6.9; Japanese 10.0, German 12.4, Russian 22.1 | 0.42 cores (415 CPU ms per second of audio); 2.4 GB |
| SenseVoice small | Japanese, Chinese and Cantonese finals, emotion and sound tags | 163 MB | 0.03–0.04 | Japanese 6.8, Chinese 4.5, Korean 7.9, English 8.7 | 0.08 cores (76 CPU ms per second of audio); 0.4 GB |
| Moonshine base | English finals, the lightest recognizer; not used by the automatic route ([long utterances](#moonshine-inside-sigil-sherpa-onnx)) | 111 MB | 0.017; a final 132 ms after the end of speech | English 13.4 on clean speech (10-utterance screening) | 0.05 cores (51 CPU ms per second of audio); 0.3 GB |
| Whisper (Rust, ggml) | Any language, one final per sentence; `base` / `small` Q5_1 on the processor | per model (the panel shows it) | the fp16 `base` model: RTF 1.09 on 4 threads, 1.18 next to a game (the quantized models should be cheaper; not measured) | the fp16 `base` model: English 25.8 on clean speech (10-utterance screening) | the fp16 `base` model: 3.8 CPU seconds per second of audio, 0.4 GB; the quantized models should cost less |

Opt-in GPU models that the route can plan (the transcribe.cpp runtime pack; **Hardware → Graphics card**, or **Best for my PC** when one fits). They are not part of the P7 campaign; the figures are the P2 measurements.

| Model | Download | Video memory | Notes |
|---|---|---|---|
| Qwen3-ASR 1.7B Q8_0 | 2.2 GB | about 3.5 GB | Finals for Chinese, Japanese, Korean, Russian and the multilingual languages; a final about 0.2 s after an 8 s sentence on the RTX 4090 |
| Qwen3-ASR 1.7B BF16 | 4.1 GB | about 6 GB | Cards with 16 GB or more |
| Cohere Transcribe Q4_K | 1.6 GB | about 2 GB | English and European languages only; never Chinese, Japanese, Korean or Russian |

Voxtral Mini 4B Realtime Q4_K_M (2.8 GB download, about 4.1 GB of video memory; streaming, append-only partials) can be installed from **Models & Runtimes**, but **no route plans it**: no row of the routing table lists it, so **Hardware** never selects it and no inspector field does. Only the speech benchmark and the developer probes run it.

Cloud engines (Groq, Deepgram, Azure, Gemini Live) cost no processor time here but send your speech to the service; see their rows in [STT / TTS backend minimum configuration](#speech-to-text).

**Defaults:** new desktop installs start on **Built-in (on your PC)** with **Hardware → Best for my PC**; the route per language is the table under [Speech on this PC](#automatic-speech-to-text-built-in-on-your-pc). The P7 campaign set the processor route from the own-microphone results (clean speech plus the two microphone-side conditions; the game-audio conditions play no part): English is Nemotron partials with a Parakeet Ultra final; the other Parakeet languages and **Russian** are Ultra finals (Russian moved off Qwen3-ASR 0.6B: 7.3 against 22.1); **Japanese** is a SenseVoice final (6.8 against 10.0, moved off Qwen3-ASR 0.6B); Korean stays on Qwen3-ASR 0.6B (5.1 against 7.9); Chinese and Cantonese stay on SenseVoice. Moonshine does not join English: it is the lightest recognizer but about 5 points behind Ultra on clean speech. A saved profile keeps its language and hardware choices; only the table that maps a language to models changed, so the next start of a Japanese or Russian profile plans the new final and asks before downloading it. **Setting the spoken language** matters more than any engine choice: with **Detect automatically** (the default of a new install) every language gets Qwen3-ASR 0.6B finals, which are 1.2 to 4 times worse than Ultra on the European languages and hold 1 GB more memory.

### Translation by tier

| Engine | Runs on | Download | Quality (chrF2, three sets pooled) | Speed per caption line | Footprint next to a game |
|---|---|---|---|---|---|
| Firefox Translations (Bergamot) | Processor, no video memory | 17–60 MB per direction plus the runtime pack | not measured: the runtime pack is not built or published yet ([build steps](../scripts/runtime-packs/bergamot/README.md)) | — | — |
| NLLB-200 600M int8 | Processor, 2 threads | 0.9 GB | 49.8 greedy, 52.7 beam 4 | 0.9 s greedy (p95 1.3 s), 1.6 s with beam 4 (p95 2.4 s) for a caption line | 1.7–1.9 cores while a line translates; 2.6 GB |
| Hy-MT2 1.8B Q4 (the default language model) | `llama-server`: Vulkan on the GPU, or the processor pack | 1.1 GB; about 1.5 GB video memory on the GPU | 53.5 (processor), 53.1 (Vulkan) | Vulkan: 51 ms idle, 116 ms under the VRChat + OBS GPU stand-in; processor: 0.38 s | Vulkan: about 0.5 core, 1.6 GB memory, 1.5 GB video memory; processor: 4.7–5.4 cores in bursts, 2.4 GB |
| DeepL API Free, Gemini Flash-Lite (free tier) | Cloud | — | — | — | no local cost; text goes to the service |

Opt-in larger models (not part of the P7 campaign): Hy-MT2 7B (4.6 GB, about 5.3 GB video memory), MiLMMT-46 4B and TranslateGemma 4B (2.5 GB, about 3 GB each). They run on `llama-server`, and the plan box refuses one that does not fit next to a running voice changer. chrF2 pools 240 casual caption lines, WMT24++ and FLORES-200 (24 lines per direction each): Hy-MT2 beats NLLB by 5–16 points into Japanese, Korean and Chinese, and NLLB with beam 4 is ahead on the European pairs with English (the next paragraph says which and what it costs). chrF scores Chinese and Japanese targets low for every engine, and one reference per line cannot rank close engines.

**Defaults:** **Automatic (best available)** with the chain default language model → Firefox Translations → NLLB; the campaign confirmed it (the audience's targets ja, ko and zh stay on the better engine, and a PC without the Firefox Translations pack falls through to NLLB). **One built-in exception, for a processor-only PC** (MT-Europe, measured after the campaign): while the language model runs on the processor, English with **French, Portuguese, Dutch and Spanish** (both directions; the source is your **Source language** setting or, left on automatic, the language the speech engine reported for the line) is translated by NLLB with beam 4 first, with the language model behind it. NLLB scores 2.8–6.0 chrF2 points higher on those pairs (95 % interval above zero) and takes 1.8 cores for 1.6 s a caption line, where the language model takes 3.7–5.0 cores for 0.4 s; German and Italian tied and stay on the language model, and nothing changes on a graphics card, where the language model answers in about 0.1 s for half a core. The route needs the NLLB model (0.9 GB, **Models & Runtimes**); without it everything runs as before, and a rule of your own under **Language pair rules**, or a chain you edited, turns the route off ([measurements](benchmarks/SPEECH_BENCHMARKS.md#european-pairs-on-a-processor-only-pc-mt-europe-2026-10-11)). What NLLB does not read: it uses neither the glossary nor the background text, so a line that has a glossary term, or any line while the background text is filled in, keeps going to the language model first; and it does not read the rolling context (the earlier lines) either, it translates each line alone. The measured gain is for a line translated alone (the language model was scored without its context too, because the benchmark has no European set of consecutive lines; how much its rolling context would add against NLLB was not measured), so if you rely on the context, add a pair rule that puts the language model first for the pair. **Interim (partial) translations are off by default while the default language model runs on the processor**: they took 4.0 cores next to the game (6.7 at the peak) against 1.5 with final lines only, and the finals are always translated. On a GPU variant (the model server's default `auto` picks one when the card has room) they stay on at 0.2–0.4 cores. Tick **Interim results** in the Automatic or Language model panel to turn them on; a choice made earlier is kept.

### Text-to-speech by tier

| Voice | Runs on | Download | Speed | Footprint next to a game |
|---|---|---|---|---|
| Kokoro v1.0 (**Kokoro / Piper (local)**, the default) | Processor, 2–4 threads | 350 MB (+ 8 MB runtime) | RTF 0.36 on 4 threads, 0.55 on 2; first audio 0.74 s (1.1 s on 2 threads) | 3.0 cores on 4 threads and 1.8 on 2, for the length of a sentence; 0.7 GB while loaded |
| Piper voices | Processor | 67–129 MB per voice | RTF 0.04 idle, 0.15–0.16 under load | not re-measured in P7 |
| Kitten nano / micro / mini (English) | Processor | 44–68 MB | not re-measured in P7 | not re-measured in P7 |
| Edge voices | Cloud, keyless | none | not timed (the benchmark has no cloud backend) | falls back to Kokoro when the service fails |
| Windows voices (SAPI) | System | none | a sentence renders in 10–15 ms (p50) | 2–3 ms of processor time per second of audio; about 130 MB |

Opt-in GPU and cloning voices (Qwen3-TTS, Magpie, Breeze, OmniVoice, Fish S2, Chatterbox, Pocket) run as helpers or in process and are not part of the P7 campaign; their downloads and video memory are listed in [Local cloning and GPU voices](#local-cloning-and-gpu-voices).

**Defaults:** new desktop installs start on **Kokoro / Piper (local)**, and the campaign kept it. Windows voices (SAPI) are the free option that needs no download and costs next to nothing, with only the voices Windows ships; Edge is the optional cloud voice and falls back to Kokoro. Kokoro is the one speech engine that takes real processor time (a burst of 2–3 cores for the length of each sentence); on a busy PC set its **Threads** to 2.

### Voice changer by tier

The voice changer is optional and used less than captions. The preset follows your tier; the figures are the P7-B final scorecard of the shipped defaults and P3B's thread sweep (dev rig; [benchmarks/VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md#p7-b-final-scorecard-of-the-shipped-defaults)).

| Tier | Preset | Provider | What it costs |
|---|---|---|---|
| CPU only | **Streamer** | CPU | Converts fast enough on 4 to 6 threads at below-normal priority and, with the block it takes by itself (120 ms from 7 physical cores, 160 ms with 6, 200 ms with 5 or fewer) and the output buffer that block gets automatically, plays out what it converts in the benchmark, but is not real time under a game (P7: the preset takes all physical cores but two, but never fewer than 4 and never more than 6 threads, so a PC with 6 or more physical cores keeps two cores for the game, a 5-core PC keeps one and a 4-core PC none; the one-thread preset it replaces ran at RTF 2.7 and fell behind). On the 8-core dev rig at 120 ms blocks, with the PC idle: RTF p99 0.83 offline and 0.74 paced on 6 threads (2.8 cores paced), 0.89 and 0.96 on 4 threads (2.6 cores). At the old 35 / 70 ms output queue the benchmark played only 48-63 % of the converted audio ([why](benchmarks/VC_BENCHMARKS.md#diagnosis-the-recovery-never-ends)); with the 110 / 136 ms buffer that 120 ms blocks now get it played **99.1-99.8 %** in the paced and game-load runs on 6 threads (worst single run 98.8 %), with 0-0.4 s of dry output per minute and 265-278 ms from microphone to speaker (median; [re-run](benchmarks/VC_BENCHMARKS.md#re-run-with-the-floor)). **Chunk size** on Auto gives the Streamer on the processor a chunk by your physical cores: 120 ms from 7 cores (what the 8-core measurements in this row ran), 160 ms with exactly 6, 200 ms with 5 or fewer, because a smaller PC cannot keep up at 120 ms (below), and the 20 ms the default Audio device buffer would derive cannot keep up on any PC, because a model call has a large fixed part: a 20 ms block measured RTF p99 3.8 on 6 threads (call p50 57 ms, p99 76 ms; 26 s of 40 s of input dropped). A **Chunk size** you set, **Autotune**'s pick, or an Audio device buffer that already gives 60 ms or more wins; the environment variable `SIGIL_VC_STREAMER_BLOCK=0` restores the old rule. **With the button's own FCPE on the final build** (8-core rig, 6 threads, three passes) it plays 99.1 % of what it converts paced and 99.3 % under the game stand-in, drops no input, and reads 278 ms (p99) paced and 348 ms under the game (SwiftF0: 273 and 297 ms); FCPE costs about 0.04-0.12 of RTF p99 and keeps the better-sounding audio, so it stays the Streamer's F0. **With 6 or fewer physical cores the Streamer falls behind at 120 ms while a game runs** (the same build with the processor cut to whole cores, 4 threads): under the game stand-in it drops **9.5 %** of the speech on 6 cores (7.3 % with SwiftF0), **14 %** on 5 cores and **20 %** on 4 cores, and the voice arrives 0.4-0.5 s late; with no game 5 and 6 cores deliver everything (99.6 % played, RTF p99 1.04-1.06, p99 301 ms), 4 cores do not keep up even then (RTF p99 1.16, 15 % of the blocks over budget, 445 ms). That is why Auto takes a longer block on those PCs, **160 ms on 6 cores and 200 ms on 5 or fewer**: a longer block stopped the dropping at a price in delay. On 6 cores 160 ms dropped no input under the game stand-in in the two runs on file and 0.5 % in one quick check made with the PC busy (44 % total CPU; 99.04 % of the converted audio played, 345 ms median from microphone to speaker); 200 ms dropped none in three runs but played 97.86 % of the converted audio in its quick check (411 ms), just under the 98 % the benchmark asks for. The user judged the 160 ms run's drop to be the busy PC and chose 160 ms for its lower delay. On 4 cores 200 ms dropped no input in three runs (400-493 ms end to end; the paced run played 96 % of the converted audio with 2 s of dry output a minute, below the 98 % the benchmark asks for), and 5 cores were not run with a longer block and take the 4-core figure. On a small PC the least speech skipped matters more than the delay, and 6 cores are the one size where that was weighed against 60 ms of delay and 160 ms chosen. The inspector shows a note directly under **Chunk size** only for the Streamer on the processor, when a size you set is shorter than the chunk Auto takes on your PC (200 ms on 5 or fewer cores, 160 ms on 6; on 7 or more cores, or while the core count is not known yet, anything under 120 ms, without naming a core count; none on Auto) and names both sizes; **Autotune** was not part of the test, and a graphics-card preset takes the load off the processor ([numbers](benchmarks/VC_BENCHMARKS.md#the-cpu-streamer-on-4-5-and-6-physical-cores)). Open: (1) under the game stand-in the 8-core rig goes over its time budget in about 2.6 % of the blocks (RTF p99 1.08) and reads 348 ms end to end (p99), above the 280 ms target; (2) a listening check on a real audio device. Leave **Voice changer priority** on **Automatic** (below normal on the processor); **Protect game FPS** keeps the Streamer on one thread, which cannot keep up |
| Mid-range GPU | **Low latency** | CUDA, else DirectML | Per-block compute p95 8.0 ms on CUDA, and 8.9 ms (full GPU clock) to 13.8 ms (idle clock) on DirectML; under a 70 % load from another process 11.1 ms (CUDA) and 12.4–25.7 ms (DirectML, depending on the clock). Engine-added delay p99 115–143 ms (115 ms with the app's own 20 ms blocks; target 85, missed). Video memory about 1.3 GB on CUDA (including the CUDA context) and 0.85 GB on DirectML; a second hot voice adds 131–142 MB; 0.04–0.05 cores while converting and 0.009 while silent; a voice loads in 1.0–1.2 s; a 30-minute soak under a game-like load had 0 underruns |
| High-end GPU | **Quality**, **Studio-TRT** | CUDA, TensorRT-RTX | Adds a retrieval index and 20 ms of lookahead; see [Presets](#presets) |

See [Defaults and opt-ins at a glance](#defaults-and-opt-ins-at-a-glance) for what the engine does on its own and what stays a switch.

### What Sigil itself costs

Measured on the release app (frontend embedded) next to the benchmark's game stand-in (four busy processes at 60 % duty), with the OBS captions page open in a headless browser and caption text arriving at speaking pace; averages over two minutes, memory is the private memory of the whole process tree (WebView2 and the translation server included). The captions listened to a real microphone in an ordinary room, so with nobody speaking they idle; the cost of speech itself is in [the pipeline table](benchmarks/SPEECH_BENCHMARKS.md#the-whole-caption-pipeline-silero-vad-router-two-passes-at-real-time) (0.2–0.5 cores while talking continuously, a few hundredths over a normal stream). Details: [benchmarks/SPEECH_BENCHMARKS.md → App footprint](benchmarks/SPEECH_BENCHMARKS.md#app-footprint).

| Scenario | Processor (cores) | Memory | GPU / video memory |
|---|---|---|---|
| Idle, app open, nothing started | 0.00 cores | 336 MB | 64 MB (the window's own) |
| Captions into OBS (English, automatic speech-to-text, OBS page open, caption traffic) | 0.14 cores; the OBS page's browser 0.02 cores | 1.8 GB; the browser 0.36 GB more | 46 MB |
| Captions only, nobody speaking (English) | 0.02–0.03 cores | 1.75 GB | 48 MB |
| Captions with **Detect automatically** (the new-install default) | 0.09 cores (p95 0.36) | 2.8 GB | 48 MB |
| Captions with **Enable interim results** off | 0.05 cores | 1.0 GB | 38 MB |
| Captions and translation (en to ja), default language model on the GPU (server variant `auto`) | 0.24 cores | 5.8 GB | 1.8 GB video memory |
| The same on Vulkan | 0.43 cores | 5.2 GB | 1.5 GB video memory |
| The same on the processor, final lines only (the default since P7) | 1.5 cores (p95 3.2) | 4.5 GB | 51 MB |
| The same on the processor with interim translations (opt-in) | 4.0 cores (p95 6.3, peak 6.7) | 4.6 GB | 52 MB |

### Keeping Sigil light next to VRChat

- **Speech-to-text:** leave **Hardware** on **Best for my PC** (or **Processor only**); pick the spoken language instead of **Detect automatically**; turn **Two-pass accuracy** off if the processor is busy (fewer than 6 physical cores already means one pass). The voice detector means silence and keyboard noise cost almost nothing (about 0.3 % of one core). **Smart Turn** is off by default; turn it on only for TTS, translation or the chatbox that should wait for the end of your turn.
- **Whisper:** use a smaller or quantized model, and set **CPU threads** (default about half the logical processors, at most 4) and **Waiting sentences** so a slow model drops old sentences instead of showing them late.
- **Translation:** Firefox Translations and NLLB need no video memory. The local language model's default variant (`auto`) uses the GPU when the card has room (0.2–0.4 cores, 1.5–1.8 GB of video memory); on the processor it takes 1.5 cores for final lines, so interim translations are off by default there (4.0 cores with them). The local model server ends itself 10 s after translation stops using it and is always ended with Sigil. A 7B model on an 8 GB card next to the voice changer is refused unless you start it anyway.
- **Voice changer:** **Voice changer priority → Protect game FPS** (below-normal priority, at most 2 ONNX Runtime threads) is for the GPU presets: on the Streamer preset on the processor it holds the voice changer to one thread, which is not real time; the **Streamer** preset on the processor (all physical cores but two, never fewer than 4 threads nor more than 6, at below-normal priority on **Automatic**; 2.6–2.8 cores on the 8-core rig, so a PC with 6 or more physical cores keeps two cores for the game, a 5-core PC keeps one and a 4-core PC none: on those run **Autotune** for a longer block and watch the frame rate; measured at 120 ms, a PC with 6 or fewer physical cores drops 9.5-20 % of the Streamer's speech while a game runs (the shipped FCPE preset: 9.5 % on 6 cores, 14 % on 5, 20 % on 4), which is why Auto takes a longer block there, 160 ms on 6 cores and 200 ms on 5 or fewer, and why the inspector warns about a shorter Chunk size set by hand), or **Low latency** on the GPU; set the **Quality governor** lever to **On** (Auto leaves it off) to let the engine step quality down under load instead of stuttering; the **Latency budget** (default 300 ms) drops stale audio instead of piling it up. Sigil never asks for administrator rights.
- **OBS:** lower the browser source's frame rate; **Shutdown source when not visible** is optional (it saves processor time while the scene is hidden; the page reconnects by itself when the scene is shown, and the first seconds may lack captions); use the `http://localhost:<port>/…` URL.

### Paid cloud services

Azure and Deepgram (speech-to-text) and Azure Translator are the paid services Sigil already integrates. ElevenLabs Scribe, AssemblyAI and Soniox are not integrated; the research's published numbers for them are in [SPEECH_STACK_RESEARCH_2026-09.md → Appendix A](SPEECH_STACK_RESEARCH_2026-09.md#appendix-a-paid-cloud-low-priority).

---

## STT / TTS backend minimum configuration

Use this when testing or supporting users. Engines are selected in the Sigil host app (inspectors).

For **timeouts, wrong microphone, and “connected but no text”** behavior across STT engines, see [STT reliability and troubleshooting](#stt-reliability-and-troubleshooting).

### Text-to-speech

| Backend | Minimum required |
|--------|-------------------|
| **Kokoro / Piper (local)** | Nothing to run: pick a voice model and voice, then agree to its one-time download in the panel (Kokoro v1.0 with the runtime: about 360 MB). The default for new desktop installs. See [Speech on this PC](#local-voices-kokoro--piper-local). |
| **OpenAI-compatible (HTTP)** | A server exposing `POST /v1/audio/speech`: pick a **preset** (Kokoro-FastAPI, Chatterbox, Voxtral-4B-TTS / IndexTTS-2.5 / VoxCPM2 on vLLM-Omni, qwentts `tts-server`) or a custom **base URL**, a **model** and a **voice**; the key is optional and goes only in the `Authorization` header. |
| **Native (browser)** | Select a **voice** in Text to Speech settings. |
| **Windows** | **Voice** (native Windows SAPI); the output device is optional (empty uses the app's output device). Plays through Sigil's TTS output since P6. |
| **Azure** | Speech **key**, **region/location**, **voice**. The output device is optional (empty uses the app's output device). |
| **Uberduck** | **API key** and **voice** (v1 API; the old secret is no longer used). |
| **VOICEVOX / AivisSpeech** | **Engine preset**: VOICEVOX (`http://localhost:50021`), AivisSpeech (`http://localhost:10101`) or a custom URL, plus a speaker. Plain `http` works only for an engine on this PC. |
| **Fish Speech** | **Endpoint**. Self-hosted fish-speech needs an **API key** only when started with `--api-key`. For Fish Audio's hosted API use `https://api.fish.audio`, your key and a **model** such as `s2.1-pro-free`. |
| **Kokoro / Melo / Chatterbox (server)** | **Endpoint** (and voice/speaker where applicable), often localhost for self-hosted stacks. Kokoro-FastAPI and Chatterbox settings run through the OpenAI-compatible client (Kokoro: `{endpoint}/v1`, model `kokoro`); for Kokoro without a server use **Kokoro / Piper (local)**. |
| **ONNX (local)** | A Hugging Face repo (KittenML 0.8 repos install as verified catalog packages) or a voice folder on this PC; runs on the same in-process engine as Kokoro / Piper (local). |
| **Edge voices** | Nothing: pick a voice (keyless, Microsoft's unofficial Read Aloud service). Falls back to Kokoro when the service fails. See [Cloud voices](#cloud-voices-with-a-local-fallback-edge-gemini). |
| **Gemini TTS** | A free Google AI Studio **key**, a model and a voice. Falls back to Kokoro. |
| **Pocket TTS** / **Chatterbox Multilingual** | The model download (98 MB / 1.8 GB); Chatterbox also needs a **voice-library voice**. See [Local cloning and GPU voices](#local-cloning-and-gpu-voices). |
| **Qwen3-TTS** | The model downloads and an imported runtime pack (Vulkan, CUDA 13 or processor); a library voice for cloning. |
| **Experimental GPU voices** | An engine (Magpie, Breeze, OmniVoice, Fish S2 Pro), its runtime pack and model. |
| **Style-Bert-VITS2 / AivisSpeech** | The Japanese BERT and dictionary downloads, then an imported `.aivmx` voice. |
| **GPT-SoVITS** | Your `api_v2` server URL (default `http://127.0.0.1:9880`), weights and a reference clip. |

### Speech-to-text

| Backend | Minimum required |
|--------|-------------------|
| **Built-in (on your PC)** | Nothing to set up: pick the spoken **language** (or **Detect automatically**) and press **Start**; Sigil shows which models it will use and asks once before downloading them. The default for new desktop installs. See [Speech on this PC](#speech-on-this-pc-automatic-stt-and-local-voices). |
| **Whisper (Rust)** | First run downloads the chosen GGML **`.bin`** into **`app_data/whisper/`** through Sigil's model store: pinned to a fixed revision and checksum, resumable (a cancelled download resumes next time), with mirror fallback, and listed in [Models & Runtimes](#models--runtimes) (also `Sigil.exe --headless fetch whisper/<id>`). Files already in `whisper/` are adopted without downloading again; a local `.bin` you pick is hashed while it is copied in. You can also drop a matching `.bin` into that folder manually. Model ids: `tiny`, `base`, `small`, `medium` (each also `.en`), their `-q5_1` / `-q8_0` builds (`medium` uses `-q5_0`), `large-v3-turbo` (plus `-q5_0` / `-q8_0`) and `large-v3` (plus `-q5_0`). The old `medium-q5_1` ids never existed upstream and load `-q5_0`; an unknown id is an error instead of silently loading `base.en`. **Anime-Whisper** is bring-your-own: convert `litagin/anime-whisper` with whisper.cpp's `models/convert-h5-to-ggml.py` and select the `.bin` as a local model (Japanese is forced and no prompt is used). |
| **Vosk** | The catalog is served from the **Rust** plugin (`get_vosk_models`). The app downloads **`{app_data}/vosk-models/{model_id}.zip`** through Sigil's model store (official Alpha Cephei URLs unless you set **Custom model URL**): one download at a time, resumed after a break (Stop keeps the partial file), a 30 s no-data limit, HTML pages rejected, and a full zip check before an existing model is replaced. The default `small-en-us-0.15` is pinned to a checksum; importing a local zip is checked the same way. **vosk-browser** loads the zip from AppData via Tauri **`convertFileSrc`** (asset protocol, scoped to **`$APPDATA/vosk-models/**`** only); repeat use does not re-fetch the CDN. Custom URL must point to a full **`.zip`**; it overrides the catalog URL but the on-disk name still follows the selected **model id**. |
| **Moonshine server** (retired) | Nothing: it is no longer in the engine list. A desktop profile still on it opens on **Built-in (on your PC)** with its microphone and language carried over and a notice, and the first **Start** asks before downloading the models; nothing auto-starts that launch. Moonshine base itself still runs inside the built-in engine ([Moonshine inside Sigil](#moonshine-inside-sigil-sherpa-onnx)). |
| **Local HTTP API (OpenAI-style)** | **Base URL** including `/v1` (e.g. `http://127.0.0.1:8000/v1`). Sigil sends one `POST …/audio/transcriptions` per spoken sentence (multipart `file` as 16 kHz mono WAV, `model`, optional `language`, `prompt`, `temperature=0`, `response_format=json`), matching the [OpenAI transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription) shape. You must set **model id** in the inspector to whatever your **local** server expects (field defaults empty). **API key** is optional for localhost; it is sent only in the `Authorization` header. The request is sent by Sigil's Rust side, so servers elsewhere on the LAN work too. |
| **Groq Whisper (free cloud)** | A free API key from [console.groq.com/keys](https://console.groq.com/keys). Model `whisper-large-v3-turbo` (default) or `whisper-large-v3`. Speech is sent to Groq. See [Groq preset](#groq-whisper-preset). |
| **Deepgram** | **API key** and **language**. Model: Nova-3 (default) or Nova-2. |
| **Azure** | **Key**, **location**, **language**. Optional **phrase end silence** (100–5000 ms, default 500). |
| **Gemini Live** | Your Gemini **key** (or the Translation settings' Gemini key). Transcribe or Live Translate (preview). See [Gemini Live](#gemini-live-speech-to-text). |
| **Azure / Deepgram / native** | Keys or browser permissions as prompted in the STT inspector. Azure and Deepgram capture in the webview, so their **device** pickers list **browser (Web Audio)** microphones, not the native cpal devices used by the Rust-captured engines. |
| **Chrome / Edge** | The browser's own speech recognition (Web Speech) in a **microphone page** (`mic.html`) that Sigil opens in Chrome or Edge; keep that window open while you stream. It sends your voice to Google (Chrome) or Microsoft (Edge). Listed under **Chrome / Edge microphone page**. See below for why it is no longer the default. |

**The engine list** of the Speech to Text inspector is grouped: **Recommended** (Built-in (on your PC)), **Online (your own account or key)** (Groq, Deepgram, Azure, Gemini Live), **Chrome / Edge microphone page**, and **Other engines** (Whisper, Vosk, the OpenAI-style HTTP server; Web Speech only in the web build).

**Why the built-in engine replaced the browser engines as the default.** Chrome and Edge were the default before the built-in engine existed. Measured once on Sigil's English test corpus (30 clean utterances), Chrome's speech recognition made about 21 mistakes per 100 words and Edge's about 10, against about 6 for the built-in engine, and both send your voice to Google or Microsoft ([measurements](benchmarks/SPEECH_BENCHMARKS.md#browser-speech-recognition-chrome-and-edge-web-speech-the-old-default-english)). New installs start on the built-in engine; a profile still on Chrome or Edge is asked once whether to switch, and **Keep Chrome** / **Keep Edge** keeps it for good (see the one-time offer under [Desktop default](#stt-reliability-and-troubleshooting)).

## Remote client

The `/client` page does not run Tauri plugins; it syncs with the **host** over PeerJS. Configure STT/TTS on the **desktop host**, not in the browser client.

An OBS canvas source needs no attention while you stream. Its link is plain `http://localhost:<port>/client`: no host, id or peer parameters, and no token for a page on this PC (**Copy LAN link** adds `?token=` for another PC, see [LAN access](#lan-access-stage-2)). The manual's streamer version is [If the connection drops](manual/00-overview.md#if-the-connection-drops).

- **Start order does not matter.** A source that loads before Sigil is listening keeps showing "Connecting to Sigil…" and connects as soon as Sigil is up. After three failed tries the loading view names the address and the last error.
- **It reconnects by itself, for as long as it lives.** When it loses Sigil (restart, crash, sleep, a network drop) the yellow bar "Lost connection to Sigil — reconnecting." covers the top of the source and it tries again 1, 2, 4, 8 and 16 s apart, then every 30 s, with no limit (the old stop after about 8 minutes is gone). An attempt that stalls is dropped after 20 s, and an open link whose connection stays down for 10 s is dropped and rebuilt. Once back, it shows the host's current scene and language again. Deleting the active scene moves the sources to **main**. A lost signaling connection alone (a Wi-Fi roam, the web server rebinding) does not tear down a healthy link: the channel stays for up to 15 s while the page registers again.
- **A refusal is the one message that needs you.** If Sigil refuses the link (the LAN token changed, or **This PC only**), a running source shows "Disconnected from Sigil…" and asks again every minute; copying the LAN link again fixes it. A refusal when the page loads (or a page that can never work: no WebRTC, a wrong host or port) shows the dark "Application failed to start" panel instead, in OBS too, and that panel does not retry.
- **A page that crashes heals itself.** A render crash on `/client` shows nothing on stream and the page reloads itself after 4 s once Sigil answers, twice as long for each further reload within 10 minutes (at most 2 minutes). Any other start-up failure stays transparent in OBS (detected through `window.obsstudio`) and reloads on the same schedule; in an ordinary browser tab the panel stays and says when the next try is.
- **Refresh cache of current page** (the source's properties) is needed only after the dark panel once its cause is fixed, and once after a Sigil update. **Shutdown source when not visible** is optional (processor time) and no longer a workaround for a stuck source.
- **`/obs-captions.html`** reconnects by itself too and ignores start order. Each connection picks a random id, so two sources with the same URL are fine.

## Speech on this PC (automatic STT and local voices)

Sigil runs its recommended speech models **inside the app**: Parakeet, Nemotron, Qwen3-ASR, SenseVoice, Moonshine and Whisper for speech-to-text, Kokoro, Piper and Kitten for text-to-speech. No Python, Docker or sidecar server is needed, and your voice never leaves the PC. Servers (speaches, vLLM, Kokoro-FastAPI) still work through the **Local HTTP API** and **OpenAI-compatible (HTTP)** backends, but they are optional.

### Automatic speech-to-text ("Built-in (on your PC)")

Pick the spoken **language** and press **Start**. Sigil plans which models to use for that language and this PC (the panel's route preview shows them, with CPU / GPU and notes), asks once before downloading anything, and then runs:

- **live partials** from a streaming model while you speak, and
- **a more accurate final** for each sentence: the second pass re-decodes the whole sentence and replaces the partial on the same caption line.

| Language | CPU (every PC) | NVIDIA GPU tier |
|---|---|---|
| English | Nemotron 3.5 partials + Parakeet Ultra final | Cohere Transcribe final |
| German, Spanish, French, Italian, Portuguese, Dutch | Parakeet Ultra finals | Cohere or Qwen3-ASR 1.7B final |
| Polish, Greek and other European languages | Parakeet Ultra finals | Cohere final (pl, el) |
| Chinese, Cantonese | SenseVoice finals (no live partials) | Qwen3-ASR 1.7B final |
| Japanese | Nemotron partials + SenseVoice final | Qwen3-ASR 1.7B final |
| Korean | Nemotron partials + Qwen3-ASR 0.6B final | Qwen3-ASR 1.7B final |
| Russian | Nemotron partials + Parakeet Ultra final | Qwen3-ASR 1.7B final |
| Arabic, Hindi, Turkish | Nemotron (its own finals) | — |
| Detect automatically | Qwen3-ASR detects the language, Nemotron follows it | Qwen3-ASR 1.7B |
| Other languages | Whisper | — |

| Model | Download | Languages | Speed on a Ryzen 7 7800X3D (RTF, lower is faster) |
|---|---|---|---|
| Nemotron 3.5 streaming 560 ms (default live model) | 475 MB | 19 locales | 0.30–0.35 on 1 thread; first partial 1.1–1.7 s into speech |
| Nemotron 3.5 streaming 160 ms (low latency) | 475 MB | same | 0.80–0.87 on 1 thread (too heavy for 6-core PCs) |
| Parakeet Ultra (Parakeet TDT v3 as fallback) | 630 MB (487 MB) | 25 European languages | 0.063–0.075 on 2 threads |
| Qwen3-ASR 0.6B | 879 MB | 30 languages + dialects | 0.16–0.27 on 2 threads |
| SenseVoice | 163 MB | zh, yue, en, ja, ko (+ emotion and sound-event tags) | 0.03–0.04 |
| Moonshine base | 111 MB | English (utterances over 9 s are [cut at quiet points](#moonshine-inside-sigil-sherpa-onnx)) | 0.017 |
| Spoken-language ID (Whisper base) | 160 MB | 99 languages | 0.03 |
| CPU speech runtime (sherpa-onnx) | 8 MB | — | installed automatically with the first model |

On the dev PC, finals arrived 130–300 ms after the end of speech (Qwen3-ASR about 1.2 s). The first download for English with two passes is about 1.1 GB (Nemotron plus Parakeet Ultra); the consent dialog shows the exact size and offers a **smaller setup** (one pass) when it saves at least a fifth.

- **Language:** an explicit language gives the best live captions. Nemotron always needs one: with **Detect automatically** it runs only next to a model that detects the language.
- **Set your language.** It is the biggest accuracy step for European languages and Russian, and it saves memory. On **Detect automatically** each sentence goes through the language-ID model and then one general final model; the P7 screening scored that 1.4 to 3.6 times worse than a set language on the European languages it measured (word error rate on automatic against a set language: Portuguese 5.5 against 3.9, German 11.5 against 6.7, French 17.3 against 8.8, Spanish 13.1 against 6.1, Russian 22.0 against 6.1), and the footprint run held about 1 GB more memory on automatic than with English set (2756 against 1765 MB); that screening found no accuracy loss on automatic for English, Japanese, Korean and Chinese, where the gain is memory; see [auto mode](benchmarks/SPEECH_BENCHMARKS.md#auto-mode-no-language-set-the-default-of-a-new-install). Sigil never sets a language for you: it asks, and **Detect automatically** stays a choice everywhere.
  - **Setup wizard, STT step** (first run only: the wizard opens when there is no settings file, and never for a profile that already exists): shows the **Built-in (on your PC)** card and asks *What language do you speak?*. It starts on the first language of the Windows list that Sigil routes and that says something about the speaker: the display language, then any other preferred language the webview lists (it reads the webview's language list, not the keyboard layouts). The wizard preselects **English** too when Windows is in English (most of Sigil's streamers speak English; they confirm it with **Continue**). The one-click hint in the Speech to Text panel still skips English, because English is the display language of many PCs whose owner speaks something else; there the session suggestion finds English by listening. Nothing is stored until **Continue**. Choosing the list entry **Detect automatically** is recorded as an answer (the suggestion below then waits a week).
  - **Setup wizard, auto-start:** under the language is the checkbox **Start captions when Sigil opens** (on by default; "Captions start by themselves each time you open Sigil. Press Start once first to download the speech models."). **Continue** saves it with the language as `stt.autoStart`, the switch the inspector shows as **Start automatically**; **Skip setup** or quitting before **Continue** saves nothing, so it stays off. Both answers survive **Back**. The launch that shows the wizard starts nothing, and neither does a launch that shows it again after you quit in the middle (the app under the wizard is inert): press **Start** once, which asks before the speech models download. From the next launch captions start by themselves. An automatic start never asks and never downloads, so when the Built-in models are not on disk it stops with the information notice "The speech models (size) are not downloaded yet. Open Speech-to-text and press Start to download them." (a toast that closes after 12 s and a line in the system log, not a red error that stays). With the box on and no working microphone, captions keep retrying (1, 2, 5, 10 and 20 s, then about every 45 s) until you press Stop.
  - **STT panel:** a profile on the automatic backend that never answered sees "Your Windows language is French. Lock captions to French?" above the Start button. It is a question, never applied for you, and it is not asked for English (see above): an English-speaking profile reaches the same Lock through the suggestion below, which rests on what you actually said.
  - **Suggestion during a session:** on **Detect automatically**, after three finals in a row that name the same language, a card asks "Detected French. Lock captions to French?" with **Lock to French**, **Keep automatic** and **Don't ask again**. It sits above the Start button in the STT panel, or is a toast that takes no focus while the panel is closed (closing it with its X counts as **Keep automatic**). The pipeline reports no language confidence, so agreement across sentences stands in for it: a final counts when it holds enough speech (about three words, or six Chinese, Japanese or Korean characters; shorter finals neither count nor break the run), music-tagged finals are ignored, one final per spoken segment is counted, and a counting final in another language restarts the count, so a session that mixes languages is never asked. At most one suggestion per session, and a new Start counts from zero. A card nobody answered is taken down, without an answer, when a counting final in another language arrives or when the next session starts, so it never names a language nobody has spoken since. It applies only to the automatic backend on a language still set to automatic, and the language must be one the panel can lock to.
  - **First download:** when the language is still **Detect automatically**, the dialog that asks before the first download adds a tip to cancel and choose the language first (the automatic route downloads a language-ID model and a general model on top of the language's own; a set language usually downloads less).
  - **How Lock takes effect:** the language is saved at once. Running captions restart at once, as an automatic start that never asks anything, only when every model of the new route is already installed (a couple of seconds without captions); a restart is never turned into a download. Otherwise it applies the next time you press **Start**, which asks before downloading the missing models like any Start, and the card says which of the two happened. The language is saved in both cases, so an automatic start in the meantime (**Start on launch**, or a restart that must not ask) cannot download or ask: when the language's models are not installed it runs the installed automatic route instead of stopping, and says so in the system log, so a streamer in VR is not left without captions; the Start you press yourself asks and downloads as always.
  - **Your answer is remembered** (`stt.local.languageDecision`): **Lock** and **Don't ask again** are final; **Keep automatic** silences the session suggestion for a week; the panel's Windows-language hint is shown only until the first answer. Choosing a language in the Language list yourself needs none of this.
- **Hardware → Best for my PC** (default) uses the GPU tier only when a GPU model fits next to everything else, and keeps 8–12 GB cards for the voice changer while it runs on the GPU; **Graphics card** shares the card anyway; **Processor only** never uses the GPU. Fewer than 6 physical cores, or the voice changer running on the CPU, means one pass. The GPU tier needs the transcribe.cpp runtime pack (imported locally until it is published) and, on NVIDIA, the NVIDIA GPU runtime pack; without them, or when the GPU is busy, Sigil uses the CPU route and says why.
- **GPU tier models:** Qwen3-ASR 1.7B Q8_0 (2.2 GB download, about 3.5 GB of video memory), Qwen3-ASR 1.7B BF16 (4.1 GB, about 6 GB, cards with 16 GB or more), Cohere Transcribe Q4_K (1.6 GB, about 2 GB; English and European languages only, never Chinese, Japanese, Korean or Russian); these three are the only GPU models the router plans. Voxtral Mini 4B Realtime (2.8 GB, about 4.1 GB, streaming) is installable but not routed (no row of the routing table lists it). Granite Speech TurboCTC is available as a model but writes lower-case text without punctuation, so it is never picked for captions.
- **Two-pass accuracy** (default on): the live partial is replaced by the final under the same caption line (on the dev GPU about 0.2 s after the end of an 8 s sentence). With it off Sigil runs one model.
- **Enable interim results** (default on; applies the next time speech-to-text starts): shows words while you are still speaking. Off, each sentence appears once, when it is finished, and Sigil does not run the live model whenever a more accurate one also runs, which saves its processor time (0.3–0.4 cores while speaking) and memory (about 0.8 GB; English on 8 cores becomes Parakeet Ultra alone, 0.24–0.28 cores). A language whose only engine is a live model keeps it and still shows its finished sentences. With interim results off there is no second pass to run, because there is no live caption to replace.
- **Smart Turn** (off by default, 9 MB): detects when you finished your turn. It only lets translation, TTS and the VRChat chatbox commit sooner; it never changes captions.
- **Hotwords:** names and terms to recognize reliably. Qwen3-ASR and Parakeet use them (modestly); Nemotron cannot.
- **Hallucination filter** and **Voice detection** work as for Whisper.
- After a microphone loss or an engine crash the session restarts by itself: 1, 2, 5, 10 and 20 s apart, then about every 45 s until you press Stop, with the status line "Speech recognition is not running (…). Trying again in N s…" and no error toast. A missing model, an invalid or unsupported request, and an out-of-memory error a restart cannot fix stop with their message. A first start that fails because the microphone is busy (another app, exclusive mode) or an engine fails to load is retried the same way, **Start on launch** included. Details: [Automatic restarts and reconnects](#automatic-restarts-and-reconnects).
- Existing desktop users on the browser default are asked once whether to switch to it (answer no and Sigil uses Whisper, because Web Speech cannot work in the desktop app); users of any other backend are never switched.

### Local voices ("Kokoro / Piper (local)")

| Voice model | Download | Languages |
|---|---|---|
| Kokoro v1.0 (default) | 350 MB (+ 8 MB runtime) | English, Japanese, Chinese, Spanish, French, Hindi, Italian, Brazilian Portuguese (54 voices) |
| Kokoro v1.1-zh | 365 MB | Mandarin (103 voices, with English) |
| Piper | 67–129 MB per voice | 36 voices, including Russian and Arabic and many more languages |
| Kitten 0.8 mini / micro / nano | 44–68 MB | English |

- Everything runs on the CPU (the UI never waits for it). **Threads**: automatic (half the physical cores, 2–4) or 2–4. Kokoro on 4 threads speaks about 3.4× faster than real time (RTF 0.29 on a Ryzen 7 7800X3D); the first clause of a long first sentence is synthesized on its own so speech starts sooner (0.65 s for a two-second opening clause; shorter clauses start faster). Piper runs at RTF 0.04–0.16.
- **Preview** plays the chosen voice; **Voice folder on this PC** accepts an official sherpa-onnx Kokoro / Kitten / Piper package or a `speech_package_v1` folder (its voices appear after the first start).
- **Japanese:** Kokoro reads kana correctly but reads kanji with Chinese readings; for Japanese use VOICEVOX or AivisSpeech (the panel says so).
- New desktop installs start on this backend; existing users on Web Speech or Windows voices are asked once whether to switch (no keeps their voice).
- `SIGIL_LOCAL_TTS_KOKORO=sherpa` runs Kokoro entirely inside sherpa-onnx (diagnostics only; the default runs its model on Sigil's own ONNX Runtime, which is faster and uses less CPU).

Re-run a quick latency and accuracy check on your own machine (VRChat + OBS load) after changing engines; published benchmarks do not replace on-device measurement.

## Models & Runtimes

The **Models & Runtimes** tab (Sigil navigation, above Settings) manages everything Sigil downloads: speech models, runtime packs, voice templates and benchmark assets. It is backed by one model store shared with `Sigil.exe --headless fetch`.

- **Entries** are grouped by kind (Models, Runtime packs, Voice templates, Benchmark assets) and then by family, with a search box and an **Installed only** filter (entries that are downloading, failed or paused always show). Each row shows the size, version, license (as plain information), variant and VRAM notes, and the state.
- **Actions:** **Download** / **Update** / **Retry** / **Resume**, **Cancel** (keeps the partial download so the next attempt resumes), **Check files** (re-hashes the files; a damaged install is removed), **Delete** (with a confirmation), and **Import…** (a file for single-file entries, a folder for multi-file ones). Progress shows percent, bytes, speed, time left, the server in use, the attempt number and the file.
- **"Import only"** entries are Sigil-built assets (runtime packs, templates) that are not published anywhere yet: install them with **Import…**, a custom mirror, or `SIGIL_LOCAL_ASSETS`.
- **"Found on this PC"** means an existing file in `whisper\` or `vosk-models\` was recognised by its checksum and adopted without downloading again.
- **Download source:** **Automatic** (the catalog's order; every Hugging Face file also has an hf-mirror.com fallback), a preferred mirror (Hugging Face, hf-mirror.com, ModelScope, GitHub releases), or a **custom mirror** laid out as `<URL>/<entry id>/<file path>` (https anywhere, http only on the local network). Files from any source are checked against the catalog's pinned checksums.
- **Parallel downloads:** 1 by default, at most 2, and never two from the same server. A download that receives nothing for 30 s is retried with resume; there is no overall time limit.
- **Storage:** total and per-family disk use (installed files plus resumable partial downloads), free space, and the models folder (`%APPDATA%\com.sigil.app\models\`; runtime packs in `runtimes\`).
- **GPU and runtimes** card: your GPUs, driver and CUDA versions, video memory (Sigil's share and all apps), which acceleration options work on this PC and why not (for example "driver too old" or "GPU too old"), and suggested model sizes for local translation, Qwen3-TTS and the GPU speech-to-text final pass. Suggestions leave room for the voice changer while it runs (a 7B translation model only when it fits next to it; Qwen3-TTS 0.6B while the voice changer runs on a card under 16 GB). **Check again** re-probes.

**Command line** (for troubleshooting and scripted installs; development builds are in `src-tauri\target\debug\`):

```
Sigil.exe --headless fetch <id>... [--mirror auto|huggingface|hf-mirror|modelscope|github-release|custom] [--custom-url URL]
Sigil.exe --headless fetch --list          # catalog, install states and sizes
Sigil.exe --headless sysinfo [--out f]     # GPUs, drivers, VRAM budget, suggested model sizes (JSON)
Sigil.exe --headless engine-probe <sherpa|transcribe|echo> [model] [wav]   # load a native runtime and run a smoke test
Sigil.exe --headless mt-bench <spec.json> [--out <dir>]   # translation benchmark (chrF, latency) on a local corpus
Sigil.exe --headless speech-bench <spec.json> [--out <dir>]   # speech benchmark: STT accuracy, CPU / GPU speed under load, MT quality, TTS renderings (developer tool)
```

`SIGIL_APP_DATA_DIR=<folder>` points any of these at a scratch profile instead of `%APPDATA%\com.sigil.app`. The full list of headless commands is in the [developer guide](DEVELOPER_GUIDE.md#native-ipc).

### CPU speech runtime pack (sherpa-onnx)

`runtime/sherpa-onnx` (sherpa-onnx 1.13.8, CPU only; 8 MB download, 22 MB installed) is the CPU runtime of the automatic speech-to-text models (Nemotron, Parakeet, Qwen3-ASR 0.6B, SenseVoice, Moonshine, language ID) and the local voices (Kokoro, Piper, Kitten). It installs automatically with the first model that needs it; Sigil starts and every other feature works without it. Install it from Models & Runtimes or with `Sigil.exe --headless fetch runtime/sherpa-onnx`; check it with `Sigil.exe --headless engine-probe sherpa [catalog id or model folder] [wav] [-- lang=en]`. When it is installed it loads at startup (`SIGIL_SHERPA_PRELOAD=0` turns that off). The error **"another ONNX Runtime is already loaded"** means some other `onnxruntime.dll` (for example Windows ML's copy in System32) was loaded into Sigil before the pack; restart Sigil so the pack loads first. Models that are not from the catalog are test-loaded in a separate process first, so a damaged model folder cannot crash Sigil.

### GPU speech runtime packs (transcribe.cpp)

`runtime/sigil-transcribe-<variant>` (`cuda13`, `vulkan`, `cpu-avx512`, `cpu-avx2`, `cpu`) run GGUF speech models such as Qwen3-ASR 1.7B. They are built locally and installed by **Import…** or `SIGIL_LOCAL_ASSETS` until they are published. The best variant for the PC is picked automatically; `cuda13` needs driver R580 or newer, a GeForce 16/20 series or newer GPU and the NVIDIA GPU runtime pack (only its cuBLAS is used); GPU variants need an AVX2 CPU. Budget about 3.5 GB of video memory for Qwen3-ASR 1.7B Q8_0. The GPU tier of **Built-in (on your PC)** uses them, and only when the resource governor allows it (enough free video memory; not next to a GPU voice changer on a card under 16 GB in the **Best for my PC** tier); otherwise the model runs on the CPU pack with the reason shown. `SIGIL_STT_ISOLATION=worker` runs the engine in a separate `Sigil.exe --headless stt-worker` process (a crash then restarts it and, after repeated GPU crashes, moves it to the CPU).

### Local LLM runtime packs (llama-server)

`runtime/llama-server-vulkan` (default GPU build), `runtime/llama-server-cuda13` and `runtime/llama-server-cpu` are the official llama.cpp server builds behind the translation **Local model server** (see [Translation backends](#translation-backends)). Sigil runs them as helper processes that listen only on `127.0.0.1` with a random key and are killed together with Sigil, even when Sigil is ended in Task Manager. The translation models (family **Translation models (language model)**, ids `mt/…`) are GGUF files pinned to their Hugging Face commits; Hy-MT2 1.8B Q4 (about 1.1 GB) is the default.

### Offline translation models and packs

- **NLLB-200** (`mt/nllb-200-distilled-600m-int8`, about 0.9 GB): runs on the processor inside Sigil. The 1.3B model is built locally (`scripts/mt-bench/export_nllb.py`) and imported.
- **Firefox Translations (Bergamot)** (`mt-bergamot/<from>_<to>`, one small model per direction, about 17–60 MB each): needs the `runtime/bergamot` pack, which is built locally (`scripts/runtime-packs/bergamot/build.ps1`, [ADR 0008](adr/0008-bergamot.md)) and installed with **Import…**. Until it is imported, downloading a direction shows the "import first" message.

`Sigil.exe --headless mt-bench <spec.json> [--out <dir>]` scores translation engines on a local corpus (chrF and latency; [`scripts/mt-bench/README.md`](../scripts/mt-bench/README.md)).

---


## STT reliability and troubleshooting

Operator and developer reference for how Sigil’s STT behaves in **shipping builds**, where failures appear, and which code paths matter. Use this when debugging “connected but no text”, wrong mic, or stuck “connecting”.

### Orchestrator ([`src/core/services/stt/index.ts`](../src/core/services/stt/index.ts))

All backends implement the same contract toward `Service_STT`:

| Callback | Meaning |
|----------|---------|
| `onStart` | Engine is ready; UI shows **connected**. |
| `onStop(error?, { retry, slow, unproven }?)` | Session ended; optional error toast (sticky if `error` set). With `retry` the failure may pass by itself and `Service_STT` reconnects instead of showing an error ([Automatic restarts and reconnects](#automatic-restarts-and-reconnects)). |
| `onInterim` | Non-final recognized text only. |
| `onFinal` | Final segment for pubsub/captions (downstream may treat as a discrete line). |
| `onStatus(text, {progress, level})` | Engine status (download and load progress, retries, rate-limit waits). Shown **under the Start button** only; it is never published, so it never reaches captions, the chatbox or TTS. |
| `isMuted()` | Whether STT is muted; utterance engines skip sending muted speech. |

**Connecting timeouts** (orchestrator clears “connecting” and tears down the engine). These are **inactivity** windows: every status or progress report from the active engine while connecting (`onStatus`, such as download or model-load progress) restarts the timer, so a long model download does not time out while it is still advancing. Whisper reports download, local-model import and model-load progress (`whisper:download_progress`, `whisper:load_status`); Vosk reports its model download (`vosk:download_progress`, also when the server sends no file size). A download that stops advancing still times out after the window below.

| Backend group | Timeout without progress |
|---------------|---------|
| Default (e.g. Azure, Deepgram, native, local HTTP / Groq) | 45s |
| Built-in (on your PC), Whisper, Vosk | 300s |
| Chrome / Edge external mic | 6s (mic page must signal start) |

**Session epoch:** `#sessionEpoch` ignores stale callbacks after stop or backend switch.

**Auto-start:** `native` is excluded from `autoStart` in `init()` (known Web Speech quirks in the shell).

### Automatic restarts and reconnects

A session the user still wants running is not left stopped by a failure that can pass. The status line says Sigil is trying again, there is no error toast, and **Stop** ends it. The pure rules live in [`stt/restartPolicy.ts`](../src/core/services/stt/restartPolicy.ts) (`RESTART_DELAYS_MS`, `SLOW_RESTART_MS`, `restartCanFix`, `startFailureCanRetry`, `networkFailureCanRetry`).

- **Ladder, then slow pace.** 1, 2, 5, 10 and 20 s apart, then every `SLOW_RESTART_MS` = 45 s for as long as STT is meant to run. A session that ran 60 s (`STABLE_RUN_MS`) starts a new streak. The system log keeps the five fast lines and the first slow one. The timers run in the webview, so a minimized window can stretch 45 s to about 60 s.
- **What still stops.** A missing model (`not_installed`), `invalid_input`, `unsupported` and an out-of-memory error a restart cannot fix stop with their message, also when a slow retry meets one.
- **Built-in (on your PC):** `LocalDriver` restarts after a lost microphone or an engine crash (status `stt.local_status_retrying`). Whisper and Vosk have no automatic restart; they stop with their message. A **first start** that fails with something a restart can fix (the microphone is busy, an engine fails to load) is retried the same way (`stt.local_status_start_retry`, or `_capture` for a microphone that could not open), whether you pressed Start or it was **Start on launch**. While it waits, a device change (a microphone plugged back in) retries after 0.8 s, at most every 5 s. Errors from consent, download or preparation are not retried.
- **A thread that panics.** A panic in the speech pipeline's capture thread, utterance or stream worker, or two-pass worker ends the session as a `capture_error` or `engine_error` fatal, which is on the retry list above (`TwoPassOutcome::WorkerDied` carries the two-pass case). The voice-detection worker of the Whisper and Vosk capture reports its panic as a session error (`SpeechCaptureEvent::Error`). Before, a panic left a session that looked connected and heard nothing.
- **OpenAI-compatible, Groq (Rust HTTP engines), Deepgram, Azure:** a lost connection, a network failure or a close by the service is reported with `onStop(error, { retry: true })`. `Service_STT` releases the dead engine, shows "Reconnecting in N s…" (status stays *connecting*) and restarts it on the same ladder, then every 45 s. A rejected key, bad settings (`invalid_input`) and `unsupported` stop with the message. A rate limit (`slow`) waits 45 s before the first reconnect and goes on at that pace; a used-up daily quota looks the same and costs one quiet attempt per 45 s. A connection that never opened (`unproven`: a refused handshake also looks like a wrong key or region) is retried on the fast ladder only while no session of the streak has connected, then stops with the message; once a session connected, the same failure is an outage and retries go on at 45 s. A Start you press yourself on these engines that fails shows its error right away; a start on launch is retried.
- **Unreadable replies.** The Rust HTTP engine retries a reply it cannot read (an HTML login page, cut-off JSON, an empty body), 408, 425 and 5xx four times and then reports a network error that the rules above retry; an HTML page with 401, 403, 404, 407 or 410, a JSON `error` body and other 4xx stay `invalid_input`. A custom server that answers every path with an HTML page therefore loops visibly at 45 s instead of stopping.
- **Stop with Twitch stream** (Speech to Text, Text to Speech, Rewrite): `stream.on_ended` only arms a 60 s timer (`StreamStop`, [`stt/streamStop.ts`](../src/core/services/stt/streamStop.ts)). At expiry the service stops only if the option is still on, it is still running (connecting and reconnecting count) and Twitch's `liveStatus` is not connected (unknown counts as offline). The stream coming back cancels a pending stop and restarts only what this option stopped; a Start by hand clears that memory.

**Desktop default (one-time):** Web Speech (`native`) cannot work in the desktop webview. New desktop installs start on **Built-in (on your PC)** (set by this one-time migration; the setup wizard then shows it as its speech step, where **Start captions when Sigil opens** turns on `stt.autoStart` for the launches after that one: the launch with the wizard starts nothing). A profile counts as a new install while `onboardingComplete` is false; the first launch of a build with the wizard marks every profile that already exists as finished (`firstRunDecided`, see [the setup wizard](#automatic-speech-to-text-built-in-on-your-pc)), so a profile that was still on Web Speech gets the question below instead of a silent switch. Existing profiles still on `native` (or the legacy `browser` value) are **asked once** ("Switch to Sigil's built-in speech recognition?", buttons **Switch to built-in** / **Not now**) whether to switch to it; if they answer no (or were already asked) they are switched to **Whisper**: `small` with 6 or more estimated physical cores, otherwise `base`, as the Q5_1 build (`.en` when the language is English), with a toast. Existing Whisper choices are kept, nothing starts automatically (the first **Start** asks before downloading), and a later deliberate choice of `native`, or the engine picked during onboarding, is never overridden (`stt.desktopDefaultMigrated`, `stt.localOfferShown`). A profile on **Chrome** or **Edge** gets the same one-time question, worded for its browser ("You're using Chrome's speech recognition, which sends your voice to Google ... about 6 mistakes per 100 words, versus about 21 for Chrome", buttons **Switch to built-in** / **Keep Chrome**, or Edge with Microsoft, about 10 and **Keep Edge**): those engines work, so **Keep** changes nothing, is recorded as `stt.localOfferShown` and never asks again, and auto-start waits for the answer (a kept Chrome or Edge then starts as usual; switching to the built-in engine never auto-starts, its first **Start** asks before downloading). A profile that is on the built-in engine, or has been (a new install, the setup wizard, an accepted offer), counts as having answered, so picking Chrome or Edge by hand later is never questioned. The language and microphone carry over where the old engine kept them in Sigil's settings (language: Web Speech, Whisper, the OpenAI-style server; microphone: Whisper, Vosk, the OpenAI-style server); the Chrome / Edge page keeps its own in the browser, so a profile coming from there starts on the system default microphone and the automatic language, and the language card then offers to lock it. The numbers in the question are one constant, `LOCAL_OFFER_ENGLISH_ERRORS` in [`localOffer.ts`](../src/core/services/stt/localOffer.ts), which a unit test checks against the benchmark table. Profiles on any other backend are never switched. A profile on the retired **Moonshine server** moves to the built-in engine at launch without a question (its microphone and language carried over, a notice, no auto-start), because the engine it used is gone.

**Mute and shortcuts:** Muting stops audio at the source for every Rust-captured engine (the open sentence is dropped, never decoded or sent) and Deepgram pauses its stream (nothing sent or billed); Azure and Web Speech keep listening but publish nothing. **STT → Shortcuts** sets global **start/stop** and **mute** shortcuts that work while Sigil is in the background (for example in VRChat). Recording finishes on the first full combination, **Esc** cancels, and a bare letter needs a modifier. The sound-effects mute shortcut is under **Settings → Shortcuts**. Stored shortcuts that can never register are cleared once with a toast; a shortcut held by another app is kept with a warning.

**Start chain:** If `#runStartAsync` throws outside the inner `try`, the promise rejection is caught on `#startChain` and surfaces a toast (`toasts.stt_start_failed`).

### Silent microphone and a reloaded page

- **A silent microphone is named.** A microphone that delivers only digital silence looks healthy to the capture hub (callbacks keep arriving), so the session just hears nothing. The mic pipeline's capture thread (never the audio callback) watches the 16 kHz feed, and after 5 s of it the **Speech to Text** panel shows, under **Start**, "Sigil hears nothing from" and the microphone's name, with the Windows setting to check (**Settings → Privacy & security → Microphone → Let desktop apps access your microphone**) and a headset's mute switch; the footer's STT card shows a warning **No signal** instead of ONLINE, with the sentence as its tooltip. It is a status line only: no toast, no dialog, and the capture is never restarted. It clears by itself when sound returns, and on Stop, an engine stop, a reconnect and a new Start. The system log gets one line per run, however often it comes and goes.
  - **What counts as silence.** The 5 s are counted in captured audio, not clock time. A chunk is dead only when every sample lies within one 16-bit step of zero *and* the samples barely differ (exact zeros, or a stuck ±1 step), so a quiet room, a microphone with its Windows input volume turned far down and the noise of a clean low-gain interface all read as alive. 100 ms of real signal clears the report.
  - **Noise gates and headset mutes.** Once a session has heard real signal, the wait is 60 s instead of 5: NVIDIA Broadcast, Krisp or a headset's gate send exact zeros between phrases, and a pause is not a broken microphone. A device that never produced signal (privacy off, a dead converter, "System default" following Windows to a silent endpoint) is still named after 5 s, and a headset muted in mid-session after a minute. A restarted stream or a move to another device forgets what the old one delivered and reports again with the new device's name; an overrun gap does not. A mute of STT ends the report (a muted source is silent by design). `SIGIL_LOG=sigil=debug` shows the `no signal from` lines (the first outage of a session is logged at info level, later ones at debug).
  - **Where.** Only the built-in engine and the engines that use the Rust capture (the HTTP engines) report it, and only for the microphone, never for other players' captions. Whisper (old), Vosk, Gemini Live and Chrome / Edge do not.
  - **Not yet confirmed on a real PC:** the headline case (Windows microphone privacy off) is caught only if Windows hands the capture zeros; an open error would show through the usual capture-error path instead.
- **A crashed or hung page is reloaded.** Every caption passes through the Sigil window's page, which publishes it to OBS, the chatbox and the rest. When the page's WebView2 renderer process exits, Sigil reloads the page after 1 s (then 5 s and 15 s for the next ones), at most 3 times in 120 s; when that budget is spent the failure is logged once at error level and one more reload is scheduled for the moment the oldest one leaves the window, so a burst (a graphics driver reset, memory pressure) does not leave a blank window for good. A renderer that is only busy or "not responding" (a PC loaded by VRChat and OBS, a wake from sleep) is not reloaded: the first event is logged, and only one that stays unresponsive for 30 s is reloaded, once (at most 2 per 10 minutes). Before the reload Rust stops the microphone speech session the old page owned, so the microphone is not held under a dead window. A failure of the WebView2 browser process itself is only logged (Sigil must be restarted), and so are GPU and helper process exits (WebView2 restarts those). The `sigil.log` lines start with `webview recovery:`. **F5** and **Ctrl+R** still reload the page.
- **What the new page does first.** `Service_STT.init` stops what a page that reloaded left running before anything can start: the microphone pipeline (also one that is still loading its models), the old Whisper recording and the raw audio capture of the other backends, each bounded (8 s in all) so a hung Rust call never holds Start back, and a failure there never blocks the start. It writes a system-log line only when it found a microphone session. With **Start automatically** on, captions then start again by themselves; with it off they stay off until you press **Start**. OBS `/client` sources reconnect on their own.

### Input device resolution ([`src/utils/rustAudioInput.ts`](../src/utils/rustAudioInput.ts))

Backends that use **Rust capture** (Built-in (on your PC), Whisper, Vosk, local HTTP STT, Groq) call `resolveRustInputDeviceForInvoke(perServiceDevice)`, which uses the first of these that is set:

1. The STT inspector’s per-engine **input device** (a cpal device name).
2. **Settings → Audio → Default input device** — on the desktop app this is the **native (cpal)** device, persisted as `audioInputDeviceNative`.
3. **`null`** → Rust uses the system default input.

A per-engine value of `default` / `communications` means “system default” and stops there (it does **not** fall through to the Settings device); the same sentinels in the Settings field also mean system default. The Web Audio `deviceId` in `audioInputDevice` is **never** passed to cpal; it is only used by Deepgram (webview capture) when Deepgram’s own device is empty. Azure uses only its own device field. Older builds stored the Settings cpal name in `audioInputDevice`; on first run it is migrated to `audioInputDeviceNative` when it exactly matches a device returned by `plugin:audio|list_input_devices`.

**User impact:** Pick the mic once under **Settings → Audio** and every Rust-captured engine uses it; override per engine in the STT inspector only when one engine needs a different mic.

**Shared microphone and device changes (capture hub):** every Rust-captured engine now shares **one stream per microphone**, so two engines on the same mic open it once. With **System default** selected, STT follows the Windows default microphone when it changes. Unplugging a microphone no longer ends STT: a toast says the device stopped and that Sigil reconnects it, and when it is plugged back in (or a stream that delivered no audio for 2.5 s is restarted) a second toast confirms it is working again, in the same session. A saved device that is not connected stays selected and is shown as **(not connected)**. A saved device is found by its exact name; when Windows renumbered its endpoint (a USB port change or a driver update turns "Microphone (USB Audio)" into "Microphone (2- USB Audio)") the one device with the same family name (`device_family` in `audio/device_resolve.rs`, shared with the voice changer) is used, under the same session. With none or several candidates it still fails with "Input device 'X' not found" and never falls back to the system default. While the hub reopens a microphone it lost, only a family member that was not listed when the loss began counts (`match_runtime_device`), so a second identical headset never stands in for a busy one. The Settings pickers compare exact names, so they can still show "(not connected)" for a renumbered device that Sigil uses. Muting STT stops audio reaching the engines at the source. Browser-captured engines (Azure, Deepgram, Web Speech) restart by themselves after the default microphone changes. `SIGIL_CAPTURE_SHARING=0` is an internal switch that restores one stream per engine (for A/B soak tests only); stream opens, failures and recoveries are logged under `sigil::audio::hub`.

### Voice detection (Silero VAD)

Every Rust speech engine (Built-in (on your PC), Whisper, local HTTP, Groq) runs behind the bundled **Silero VAD v6.2** model, on the CPU inside Sigil (about 0.3% of one core, including resampling to 16 kHz with rubato). **Silence never reaches the recognizer:** a request or a decode happens only for audio the detector classified as speech (streaming models see only speech segments too).

| Setting (STT inspector → Voice detection) | Default | Meaning |
|---------|---------|---------|
| Speech sensitivity | 0.5 (ends below 0.35) | Probability at which speech starts; lower catches quiet speech, higher ignores more noise. |
| Pause that ends speech | 500 ms | Silence that closes a sentence (a 200 ms tail is kept). |
| Audio kept before speech | 400 ms | Pre-roll, so the first syllable is not clipped (never overlaps the previous sentence). |
| Shortest speech kept | 250 ms | Shorter blips are discarded; a sentence starts only once this much speech is confirmed. |
| Longest sentence | 18000 ms | Longer speech is split at its quietest moment. |

The old volume-gate fields (silence threshold dB, silence duration, minimum chunk) are ignored. For Whisper, turning **VAD off** now means 5-second forced splits, still Silero-gated. A device outage closes the open sentence first; the capture then continues once the device is back.

### Per-backend notes

#### Whisper (Rust + TS [`whisper.ts`](../src/core/services/stt/services/whisper.ts))

- Tauri event **`whisper:download_progress`** (`{ file, progress }`, percent, at most ~5 per second): interim text while a model downloads or a local model file is copied in.
- Tauri event **`whisper:load_status`**: interim text while a local model file is copied in (“Copying local Whisper model...”) and while the GGML model loads into memory (offloaded to a blocking thread).
- Tauri event **`whisper:segment_transcript`** (formerly `whisper:partial_result`): one payload per **completed VAD utterance**, not a streaming partial inside one utterance.
- The TS layer maps each payload to **`onFinal`**. Expect **multiple finals** per session (one per sentence), not one long partial stream.
- **Ordered, bounded backlog:** utterances are transcribed in order. At most **3** wait (setting "Waiting sentences", 1–10); when the model is slower than real time the waiting sentences are **merged into one decode** (one caption line for several sentences), the oldest is dropped past the cap, and a sentence that waited more than **20 s** is dropped instead of shown late. Drops are logged and reported as `whisper:utterance_dropped` `{seq, audio_ms, reason}` (`backlog` or `latency_budget`).
- Whisper now runs on the same speech pipeline as the automatic backend; its IPC and events are unchanged. Model downloads go through the model store: pinned, resumable (a cancelled download resumes), with mirrors, and shown in Models & Runtimes.
- **No-speech drop:** each utterance's no-speech probability is computed the way OpenAI defines it (P(`<|nospeech|>`) at the start of the transcript). Above **0.6** (setting "No-speech cutoff"; 1 disables) the utterance is dropped. This replaces whisper.cpp's own value, which is stale for multilingual models and prompts (they used to print "you" / "Thank you." on silence).
- **Hallucination filter** (default on): drops curated stock phrases ("Thanks for watching", per-language lists) only when they are the whole sentence, collapses repeat loops (3 or more; CJK uses character n-grams) and strips markers such as `[MUSIC]` or `(laughs)`.
- **Hotwords** and the **context prompt** are passed to Whisper as its initial prompt. **CPU threads** default to about half the logical CPUs (at most 4). Stopping cancels a running decode promptly.
- Other events: `whisper:speech_start` `{seq, start_ms}` and `whisper:capture_error` `{message}` (the recording ended: voice detection failed, or the engine failed 5 times in a row or fatally). A lost microphone is not an error any more: it is reported as `audio:stream_error` / `audio:stream_recovered` and the recording continues.
- **Silence check:** every decode logs `transcribe utterance seq=… audio_ms=…` at info level under the log target `sigil::stt::whisper` in the [log file](#gpu-acceleration-diagnostics-and-logs). Ten minutes of silence and keyboard noise should add no such line.

#### Vosk ([`vosk.ts`](../src/core/services/stt/services/vosk.ts))

- Models load from **AppData** via Tauri **`convertFileSrc`**. The asset protocol scope in `tauri.conf.json` is limited to **`$APPDATA/vosk-models/**`** — the rest of app data (settings with API keys, other models) and the WebView2 profile under `$APPLOCALDATA` are **not** readable through it. vosk-browser fetches the zip as data from a `blob:` worker, so release builds need CSP `worker-src blob:` and `connect-src` for `asset:` (macOS/Linux) and `https://asset.localhost` (Windows, because the window uses `useHttpsScheme`).
- Tauri event **`vosk:download_progress`** (`{ file, progress }`, same shape and rate as Whisper’s; `-1` when the server sends no size) while the model zip downloads; shown as interim status (“Downloading …”).
- Model WASM load has a **120s** timeout.
- Capture via `plugin:audio|start_audio_capture` from the shared capture hub. With `sampleRate: 16000` the `audio:chunk` payloads are 16 kHz mono (about a sixth of the old 48 kHz stereo traffic); every payload states its own sample rate and channel count, and `vosk.ts` downmixes when needed and passes that rate to the recognizer.

#### Built-in (on your PC) ([`local.ts`](../src/core/services/stt/services/local.ts))

- **Start** plans the models (`plugin:speech|preview_route`), asks before downloading any that are missing (three buttons: the full download, a smaller single-pass setup when it saves at least a fifth, or not now), downloads them through the model store with overall progress under the Start button (Stop cancels the downloads this start queued; partial files are kept for resume), then starts the pipeline. An automatic start at launch never asks: it stops with a "press Start to download" message.
- Status codes under the Start button: loading the model or runtime, "Speech recognition: …" warnings, a microphone gap, a backup model taking over (for example a GPU model replaced by its CPU fallback), and "the live model is guessing the spoken language" (choose a language).
- Captions: live partials, then one final per sentence; with two passes the first-pass final waits (as a partial) for its replacement for at most the two-pass budget plus 1 s.
- **Route preview** in the panel: each role's model and CPU / GPU, the video memory, and notes (why a model was chosen or skipped); **Models in use** and the restart count while running.

#### Utterance engines: local HTTP, Groq ([`rustCapturePolling.ts`](../src/core/services/stt/rustCapturePolling.ts))

These engines run inside Sigil's Rust speech pipeline (voice detection, request pacing, merging and retries in Rust), so they send **one request per spoken sentence** and nothing for silence. The request is sent by Rust, not the webview, so servers on the LAN work.

- Requests go out **one at a time, in order**. Sentences that pile up while a request is in flight (or rate-limited) are **merged** into the next request, up to 28 s with 0.2 s of silence between them.
- **429** waits for `Retry-After` (or 5 s when the header is unreadable) and does not count as a failure. **Other 4xx** stop the session at once with a readable message (bad key, bad model). **Network errors, timeouts and 5xx** are retried 4 times (1, 2, 4 s apart); 5 failed sentences in a row stop the session.
- A sentence that waited more than **20 s** (for example while Groq's limit is exhausted) is dropped with a warning on the status line instead of being sent late.
- In-flight requests are cancelled when STT stops. Settings are read at start, so edits apply on the next start.

#### Moonshine inside Sigil (sherpa-onnx)

`stt/moonshine-base-en-int8` (111 MB, English) is the lightest recognizer in Sigil: RTF 0.017 on the dev rig, and a final 132 ms after the end of speech through the speech pipeline. It runs on the sherpa-onnx CPU runtime pack like the other CPU models. The automatic route does not plan it (on clean speech it is about 5 points behind Parakeet Ultra: WER 13.4 against 8.1 in the 10-utterance screening), so nothing in the inspector selects it by itself; it stays in the catalog and in the benchmark.

- **Utterances over 9 s are cut at quiet points.** The Moonshine v2 model inside sherpa-onnx 1.13.8 returns an empty result, without any error, for one decode of 148,351 samples (9.272 s at 16 kHz) or more, and the speech pipeline closes utterances at up to 18 s, so long unbroken speech used to lose its whole caption (found by the P7 benchmark's smoke run, fixed in commit f8eedb8). An utterance longer than 9.0 s is now cut at its quietest moment (judged over about 140 ms in the back half of the 9 s window, never closer than a third of the limit to either end), every piece is decoded on its own and the texts are joined: no space between pieces in Chinese, Japanese, Thai, Lao, Khmer or Burmese, and none before closing punctuation. A clearly voiced piece that still decodes to nothing is decoded again as two halves. Shorter utterances are decoded whole, exactly as before.
- **Option `moonshineMaxSegmentMs`.** An engine option (camelCase, in the engine's options of a speech-bench spec or an explicit pipeline engine; there is no inspector field). It can only **lower** the limit: values from 2000 to 9000 are used as given, a smaller non-zero value becomes 2000, a larger one becomes 9000, and **`0` never splits** (utterances over 9 s then come back empty, and the log warns about it). A bring-your-own Moonshine v1 folder (preprocessor with two decoders) is split only when this option is set.
- **What the cut costs.** On the P7 English corpus, utterances over 9.3 s now return captions with a word error rate of 19.9 % (they returned nothing before: 100 %). Utterances of 8–9 s are decoded whole at 11 %, which is why the limit sits at 9 s and not 8 s (cut at 8 s they scored 19 %). Words at a seam can come out doubled, split or wrong. The measurement, its sample and how to reproduce it: [SPEECH_BENCHMARKS.md → Moonshine and long utterances](benchmarks/SPEECH_BENCHMARKS.md#moonshine-and-long-utterances-the-9-s-cut).
- **A warning in the log** (target `sigil::sherpa`, at most one a minute, with the number it skipped): audio of 3 s or more with at least half of its frames voiced decoded to no text, so the model may have failed inside sherpa-onnx (sherpa swallows that error and answers with an empty result). Only Moonshine raises this warning: the other models answer non-speech such as music or applause with no text on purpose.
- **A sherpa-onnx upgrade must re-measure the limit.** The ignored real-model test `real_moonshine_sherpa_length_limit_is_still_where_the_split_assumes` pins the 148,350 / 148,351 sample threshold.

#### Moonshine server (retired)

The external Moonshine HTTP server backend (`GET /health`, `POST /transcribe`) was retired: the built-in engine runs Moonshine and more accurate models inside Sigil, with no server to run. `Service_STT` moves a desktop profile still on it (`backend` = `moonshine`) to **Built-in (on your PC)** at launch and again at **Start**, carrying its microphone and language over (`localBackendCarryOver`), logs and toasts the notice `stt.moonshine_retired`, and skips auto-start in that launch so no download starts without a click. The web build keeps the saved value. The service code, the `stt.moonshine` settings slice, the Rust `moonshine-stt` plugin and the HTTP preset stay in the code (a saved web profile still reads them).

#### Groq Whisper preset

Stored as the `openai_audio` backend with preset `groq` (shown as **Groq Whisper (free cloud)**). The key goes only in the `Authorization` header. One shared limiter in Rust paces every session of the app run (a quick stop/start cannot reset it) inside Groq's free plan: **20 requests per minute**, **2,000 per day**, and **7,200 audio seconds per hour** (counting Groq's 10 s minimum billing per request; short sentences are held up to 3.5 s, or 8 s while you keep speaking, and merged, so fewer requests are billed). A custom URL pointing at `api.groq.com` gets the same limits. Rate-limit waits appear as status under the Start button.

#### Local HTTP OpenAI-style API ([`openaiAudio.ts`](../src/core/services/stt/services/openaiAudio.ts))

- Follows the utterance-engine rules above; `[MUSIC]`-style markers are stripped from results.

#### Deepgram ([`deepgram.ts`](../src/core/services/stt/services/deepgram.ts))

- The inspector’s **device** picker lists **browser (Web Audio)** microphones. An empty Deepgram device falls back to the Web Audio default input in `audioInputDevice` (picked under **Settings → Audio** in the web build; the desktop app’s Settings picker sets the native device instead, so there it is normally empty). If the resulting id is empty or `default` → `getUserMedia({ audio: true })`; a stored id is used as an exact `deviceId` only if it still exists, otherwise Deepgram falls back to the default mic and logs a warning (covers cpal device names saved by older builds).
- **`getUserMedia`** rejection (permission, no device) → immediate **`onStop`** with a clear message (avoids waiting for the 45s orchestrator timeout). A connection refused before it opened and a lost connection (with its close code) have their own messages.
- While STT is muted the recorder pauses and Sigil sends Deepgram KeepAlive messages every 4 s, so muted speech is neither sent nor billed.
- **Stop** before `getUserMedia` resolves: the late stream’s tracks are stopped and the WebSocket is never opened (no hot mic, no billed audio).
- WebSocket **error** → **`onStop`**; **`onclose`** is guarded so **`onStop`** is not duplicated.
- The listen URL sends `model` (Nova-3 by default, or Nova-2), `language`, `punctuate`, `interim_results` and `profanity_filter`. Only a key and a language are required to start.

#### Azure ([`azure.ts`](../src/core/services/stt/services/azure.ts))

- Validates **key, location, language** (device is optional). The Azure Speech SDK loads only when an Azure session starts; cancellations (bad key, connection, quota, bad request) show translated messages, and results carry the detected language.
- **Empty device** → `AudioConfig.fromDefaultMicrophoneInput()`; non-empty → `fromMicrophoneInput(deviceId)`.
- **Phrase end silence** sets `Speech_SegmentationSilenceTimeoutMs` (100–5000 ms, default 500), so captions finalize quickly after a pause.

#### Native Web Speech ([`native.ts`](../src/core/services/stt/services/native.ts))

- In Tauri, **network** error on Web Speech → message to use Whisper/Vosk/Chrome.
- **`onend` → `start()`** loop keeps recognition alive (Web Speech quirk).
- **800ms** fallback promotes to **connected** if `onstart` never fires.

#### Chrome / Edge external mic ([`externalMic.ts`](../src/core/services/stt/services/externalMic.ts))

- **`onStart`** depends on mic page reporting lifecycle via pubsub (`syncExternalMicLifecycle`).

### Pubsub trust (external STT)

Chrome/Edge **mic.html** sends `text.stt` over the host’s **pubsub WebSocket**. The handshake must pass the [Origin allow-list](#local-web-server-and-lan-pages), so a website open in a browser cannot connect, and the host applies an inbound topic allow-list to everything it receives. Since LAN security stage 2 a tool on another PC also needs the LAN token; `mic.html` on this PC needs none (opened from a LAN link it forwards `?token=` and explains a refusal). The token is not encryption: the LAN traffic is still plain HTTP / WebSocket. **mic.html** always connects to the Sigil that served it (`?host=` is ignored) and reconnects by itself after Sigil restarts. See [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md#network-security-and-threat-model).

### Implementation hardening (recent)

- **Local HTTP / Groq:** one request per VAD utterance through the Rust speech pipeline (P2; replaces the P0 TypeScript utterance pipeline and the 500 ms polling loop before it).
- **Segment metadata:** engines with segments (the automatic backend, the HTTP engines) publish `segmentId`, `revision` and `lang` with their text, so consumers can replace a caption line instead of appending.
- **Audio format:** Rust resamples the device audio to **mono 16 kHz** with rubato (band-limited, exact length) before the VAD, so the WAV sent to servers is true 16 kHz mono.
- **Status vs text:** engine status (downloads, retries, waits) goes through `onStatus` and is never published as caption text.
- **Native Web Speech:** one **`onFinal` / `onInterim` per `onresult` event**; **`#shouldRestartAfterEnd`** blocks `onend` → `start()` after explicit `stop()`.
- **UwU filter:** failures in `plugin:uwu|translate` are caught; transcript still flows; **System Logs** records a warning.

### Symptoms → likely causes

| Symptom | Likely cause |
|---------|----------------|
| STT “on” but no text (Whisper) | Voice detection too strict (lower **Speech sensitivity**), no-speech cutoff too low, or the wrong physical mic (check **Settings → Audio → Default input device** and the inspector’s input device). |
| Many short “final” lines | One final per sentence by design; raise **Pause that ends speech** to join sentences. |
| Sentences skipped under load (Whisper) | The model is slower than real time and the backlog dropped the oldest waiting sentence, or one waited more than 20 s (`whisper:utterance_dropped` in the log). Pick a smaller or quantized model, or more CPU threads. |
| A long sentence (over about 9 s) gets no caption, and the log says audio "decoded to no text" (Moonshine) | The model inside sherpa-onnx 1.13.8 returns nothing for one decode of 9.27 s or more. Current builds cut such utterances at quiet points ([Moonshine inside Sigil](#moonshine-inside-sigil-sherpa-onnx)); an engine option `moonshineMaxSegmentMs` of `0` turns the cutting off |
| Automatic backend: "The live model is guessing the spoken language" | Nemotron needs the spoken language: choose it instead of **Detect automatically**. |
| Automatic backend uses the CPU although there is a GPU | The route preview notes say why: the voice changer runs on a card under 16 GB (choose **Graphics card** to share it), the card is full, no runtime pack, or no GPU model for the language. |
| Groq session stopped | A rejected key (4xx) or bad settings: the status line and toast say which. A rate limit or a used-up daily cap does not stop it: it waits and retries quietly every 45 s ("The service is limiting requests…"). |
| Stuck “connecting” ~45s | Backend never called `onStart` (e.g. Deepgram before `getUserMedia` catch; reduced after fix). |
| Wrong microphone | Rust engines: per-engine **Input device**, else the native **Settings → Audio** default (see [Input device resolution](#input-device-resolution-srcutilsrustaudioinputts)). Azure / Deepgram: their own browser-device picker. |
| "Audio device … stopped" toast | The microphone was unplugged, disabled or stopped delivering audio. Sigil reconnects it by itself; STT stays running, also when Windows gives the device a new numbered name on another USB port. If it never recovers, check the device in Windows sound settings; a saved device shown as "(not connected)" needs replugging or another choice. |
| "Sigil hears nothing from …" under Start, **No signal** on the footer card | The microphone has sent only silence for 5 s (60 s if it had been heard earlier in the session): Windows microphone privacy for desktop apps is off, a headset mute switch, "System default" following Windows to a muted or virtual device, or a noise-gated virtual microphone sitting at exact zeros while you are quiet. It clears by itself when sound returns; nothing is restarted ([Silent microphone and a reloaded page](#silent-microphone-and-a-reloaded-page)). |
| The Sigil window went blank or reloaded and captions stopped | The page's WebView2 renderer crashed or hung. Sigil reloads it (after 1, 5 and 15 s, at most 3 times in 2 minutes) and, with **Start automatically** on, captions return; check `sigil.log` for `webview recovery:` lines. A browser-process failure needs Sigil restarted. |
| "Speech recognition is not running (…). Trying again in N s…" | The engine stopped on a failure that can pass (microphone busy, engine crash, lost connection). Sigil retries by itself every 45 s after the quick tries until you press Stop; free the microphone or fix the connection and it resumes without Start. |
| Sigil is still in Task Manager after I closed it | It ends itself within 20 s of the first close request: an exit guard (`src-tauri/src/exit_guard.rs`) writes one `exit guard:` warning to `%LOCALAPPDATA%\com.sigil.app\logs\sigil.log` and ends the process. A script that closes every window of the process (`taskkill` without `/F`, some installers) can leave it waiting; close the window titled "Sigil" instead. Starting Sigil again within those 20 s works: the new copy waits for the port by itself (up to about 25 s) and then starts |

### Related files

- Schema / persisted options: [`src/core/services/stt/schema/`](../src/core/services/stt/schema/index.ts) (one slice per backend)
- Inspector UI: [`src/core/ui/inspector/stt/`](../src/core/ui/inspector/stt/index.tsx)
- Speech pipeline and engines: [DEVELOPER_GUIDE.md → Speech pipeline](DEVELOPER_GUIDE.md#speech-pipeline-speech)
- Feature matrix: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit)
- Performance / timeouts: [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md#performance-and-connection-timeout-audit)

---

## Voice changer

Sigil’s voice changer runs in the **`voice-changer`** Tauri plugin (`src-tauri/src/services/voice_changer/`): the microphone comes from Sigil's shared capture (speech-to-text keeps its own untouched copy of the same microphone, so the voice changer's gain and noise removal never reach STT) → optional **noise removal** → **latency budget** → process (**Built-in** phase vocoder, **ONNX** file, **RVC voices** on the Sigil Voice Engine, or **WebSocket** sidecar) → optional post-DSP → playback queue. It is **not** compatible with [w-okada/voice-changer](https://github.com/w-okada/voice-changer)’s Python **Socket.IO / REST** server as a drop-in; Sigil’s sidecar speaks **raw WebSocket binary PCM** as documented in [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md).

### RVC voices (Sigil Voice Engine)

**RVC voices** mode runs RVC voices on Sigil's own real-time engine, the **Sigil Voice Engine**. You import a voice once (`.pth` or `.zip`); Sigil converts it for its engine and keeps it in its voice library. Voices you exported to ONNX yourself (bring-your-own ONNX) keep working in the same mode. The panel is laid out as **Voices**, **Preset**, **Voice settings**, **RVC tuning**, **Engine**, **Base model paths** and **Benchmark and tuning**. Measured performance and quality: [docs/benchmarks/VC_BENCHMARKS.md](benchmarks/VC_BENCHMARKS.md).

What the engine does that other real-time RVC clients do not (each part can be switched off under **Engine**, see [Levers](#engine-settings-and-levers)): it decodes only the audio it plays (so more context costs almost nothing), uses the same model noise and a continuous excitation for the same moment on every block (clean joins), tracks pitch only for new audio, keeps its work on the GPU between blocks (CUDA graphs on NVIDIA), drops stale audio instead of queueing it, and can step quality down under load instead of stuttering.

#### Voices

- **Import:** drop `.pth` or `.zip` files on the import box, or use **Import…**. Accepted: exported voices from RVC WebUI, Applio, Mangio and w-okada merges (`.pth`), voice `.zip` archives, training checkpoints (`G_*.pth`) and bare state dicts. From an archive Sigil picks the exported voice: training by-products (`G_`/`D_`/`f0G40k` files) give way to it, else the latest `G_<step>` is used; system metadata files are ignored; an archive that holds only ONNX files is refused with a pointer to bring-your-own ONNX voices. Imports run one at a time with progress and **Cancel**. Each import runs in a separate helper process with a memory and time limit, so a broken or malicious file cannot harm Sigil (nothing inside it is executed). A 55 MB voice imports in about a second.
- **Retrieval index:** an `added_*.index` next to the `.pth` or inside the zip is adopted automatically when there is only one candidate (a name match wins); dropping one `.index` together with one `.pth` pairs them. A `trained_*.index` is never used, and an index that cannot be read is only a warning.
- **The same file again** returns the voice already imported; importing a voice again keeps its settings and its index.
- **Library actions:** **Rename**, **Duplicate**, **Delete** (not the voice that is running) and **Merge** (a weighted blend of voices with the same model type and content encoder; the index is not carried over). Each voice row shows read-only facts: engine, RVC version, sample rate, pitch guidance, content encoder, vocoder, index, speakers, and any problem that stops it from running.
- **Bring-your-own ONNX voices:** a folder with `gen.onnx` / `model.onnx` (optional `.index`) under `%AppData%/com.sigil.app/voice-changer/models/`, or anywhere with **Other folder…**. `manifest.json` is optional and never written: an existing one wins; without one, Sigil infers the settings in memory from the generator ONNX (w-okada export metadata, the `pitch` / `pitchf` inputs, the `phone` input's width), so the folder can be read-only. If the sample rate cannot be determined Sigil assumes 40 kHz and warns. Shared base models for these voices resolve from the **Base model paths** or `%AppData%/com.sigil.app/voice-changer/base/` (`hubert_base.onnx` / `contentvec.onnx`, `fcpe.onnx`, `rmvpe.onnx`), then from the model store. fp16 exports and exports with `skip_head` / `return_length` inputs run too (the latter decode only the returned audio).
- **First run:** the **Get started** card installs the content encoder, the pitch model your settings use and the MIT default voice in one click, then imports and selects the voice. Models Sigil builds itself (the ContentVec content encoder and the FCPE pitch model) are not downloadable yet: the card shows **Import** for them (install them from a folder with Models & Runtimes → Import).
- **Permission notice:** the first RVC start asks you to confirm you only use voices whose owners agreed. The acknowledgement is stored in the Rust settings.

**Voice kinds:**

| Voice | Imports and runs |
|---|---|
| RVC v2 at 24, 32, 40 or 48 kHz with pitch guidance, HiFi-GAN vocoder | Yes |
| RVC v1 at 32, 40 or 48 kHz with pitch guidance | Yes |
| Voices without pitch guidance (v1 or v2) | Not yet: refused as "not supported yet" (the templates ship; the engine support is a follow-up) |
| Applio RefineGAN and MRF HiFi-GAN voices (v2, 24–48 kHz) | Not yet, as above |
| RVC v1 at 24 kHz, and RefineGAN / MRF on v1 or without pitch | No trainer produces these |
| DDSP-SVC 6.3, so-vits-svc 4.x ONNX exports, zero-shot voices, sidecar engines | As voice families: **New voice…** ([below](#zero-shot-singing-and-sidecar-voices-voice-families)) |

Content encoders: **ContentVec** (768 channels for v2, 256 for v1), **SPIN**, **SPIN v2**, and the **Japanese**, **Korean** and **Chinese HuBERT**; Sigil picks the one the voice was trained with (Applio's `embedder_model`). A voice trained with a custom encoder is refused, because it would play back as the wrong words.

**Import errors** (the panel shows a translated message for each):

| Code | Meaning | What to do |
|---|---|---|
| `not_a_checkpoint` | Not an RVC voice checkpoint. | Import the voice's `.pth` (from the trainer's weights folder), not an optimizer or D file. |
| `legacy_format` | An old-style (pre-zip) PyTorch file. | Save it again with a current RVC or Applio. |
| `torchscript` | A TorchScript export. | Use the original `.pth`. |
| `archive_rejected` | The zip is too large, has too many files or unsafe paths (or the import hit its memory limit). | Extract the voice's `.pth` and import that. |
| `ambiguous_archive` | The zip holds more than one voice. | Import each `.pth` on its own. |
| `pickle_rejected` | The file contains code, so it was refused. | Use a file from a trusted source; Sigil never runs code from voice files. |
| `tensor_invalid`, `non_finite` | Damaged weights or invalid numbers. | Re-download or re-export the voice. |
| `missing_tensor` | Weights are missing (often a training checkpoint). | Export the finished voice from your trainer. |
| `shape_mismatch` | The weights match no known model type. | Check the voice was trained with RVC / Applio. |
| `unsupported_model` | A kind of voice Sigil does not run yet (for example no pitch guidance, RefineGAN, MRF, a custom encoder). | Use a supported voice kind (table above). |
| `no_template` | No engine template for this model type. | As above. |
| `layout_invalid` | A template of this installation is damaged. | Reinstall Sigil. |
| `session` | The converted voice failed its test run. | Report it with the voice's details. |
| `time_limit` | The import took too long. | Try again with nothing heavy running. |
| `precision_refused` | The voice does not convert accurately enough at this precision. | Set **Precision** to Auto or fp32. |
| `io`, `job` | A file could not be read or written, or the request could not be done (unknown voice, cancelled, voice in use). | Check the file and folder. |

#### Presets

A preset fills the settings below; values you change afterwards are used as set. Every preset also selects the engine's benchmark-decided lever defaults (**Auto**).

| Preset | Execution provider | F0 | Index rate | Extra context | Lookahead | VRAM cap |
|---|---|---|---|---|---|---|
| **Streamer** (default) | CPU | FCPE | 0 | 64 ms | off | — |
| **Low latency** | CUDA | RMVPE | 0 | 96 ms | off | 768 MB |
| **Quality** | CUDA | RMVPE | 0.5 | 192 ms | 20 ms | 1536 MB |
| **Studio-TRT** | TensorRT-RTX (NVIDIA RTX) | RMVPE | 0.5 | 192 ms | 20 ms | 1536 MB |

Which preset suits which PC: [Voice changer by tier](#voice-changer-by-tier). On imported voices, extra context is cheap (only new audio is analysed and only the played audio is decoded); bring-your-own ONNX voices re-encode all of it every block, so keep it small on the CPU. **Autotune** (below) picks the fastest safe settings for your PC and can replace the preset's.

#### Voice settings

- **Pitch** (±24 semitones) is applied to the model's F0. **F0 method:** **FCPE**, **RMVPE**, **DIO**, **SwiftF0** (fast, CPU), **CREPE tiny**, **PESTO** (import its model) and **Fusion** (voicing from SwiftF0, pitch from RMVPE on a GPU or FCPE on the CPU: the fewest octave errors). A method this build or provider cannot run shows "(not available)" with what runs instead (RMVPE runs as FCPE on the CPU provider). **Pitch range:** Speech (50–1100 Hz) or Singing (up to 2.5 kHz).
- **Index rate**, **Protect** and the **Speaker** of multi-speaker voices work as in RVC.
- **Volume envelope** (rms_mix): 1 keeps the model's own dynamics, lower values follow how loud you speak.
- **Formant** (±12 semitones): a true formant shift on the RVC output (the pitch stays within 5 cents). While it is not 0, the older **Brightness** tilt does not apply to RVC voices, so the two never stack.
- **Variation** (noise scale, imported voices only): lower is steadier and flatter.
- Pitch, index rate, protect, F0 method, speaker, pitch range, volume envelope and formant are saved with the voice (`slot.json`) and applied when it is selected.

#### Engine settings and levers

- **Execution provider:** CPU, CUDA, DirectML, plus two accelerators: **TensorRT-RTX** (NVIDIA RTX, compute capability 8.6, 8.9 or 12.0+; needs the TensorRT-RTX runtime pack) and **DirectML high priority** (DirectML on a high-priority GPU queue; no admin rights). The first TensorRT-RTX start builds engines for the voice (up to about a minute, with progress per component); later starts load them from the cache (`%AppData%/com.sigil.app/voice-changer/trt-cache`, the 16 most recently used or 8 GB). If TensorRT-RTX cannot run, the voice runs on CUDA and the panel says why. The `runtime/tensorrt-rtx` pack is not downloadable yet: it is a local pack (a TensorRT-RTX plugin execution provider plus the runtime DLLs its imports name).
- **Precision:** Auto (half precision on the GPU where Sigil's parity tests show it is accurate, full precision elsewhere), fp32 or fp16.
- **Chunk size** (`block_ms`; in the RVC settings directly under **Preset**; it was called **Block size** and sat in the closed **RVC tuning** section): how much new audio the voice engine converts in one model call. A smaller chunk lowers the delay and costs more processing per second of audio; a larger one is steadier on a slower PC and adds delay. Changing it restarts the voice changer.
  - **The choices:** **Auto (N ms on this PC)** first, then 20 to 200 ms in steps of 20 (the engine plans whole multiples of 20 ms, at most 200 ms; the 30 and 50 ms of older versions ran as 40 and 60 ms and are gone, a stored value like that stays visible and is marked "runs as 40 ms"). While the voice changer runs, Auto shows the chunk the engine itself reports; stopped, it shows "Auto (about N ms on this PC)", an estimate from the voice, preset, execution provider and your physical cores that assumes a 48 kHz device and cannot see `SIGIL_VC_STREAMER_BLOCK=0` or a graphics-card provider that fell back to the processor. With the core count not known yet it reads plain "Auto". A "Running N ms chunks." line and the **Chunk** row of the status panel say what actually runs.
  - **What Auto takes:** the chunk follows the **Audio device buffer** (20 ms at the default 512 frames at 48 kHz), except for the **Streamer** preset on the processor, which takes the chunk your processor keeps up with because a shorter one cannot: 120 ms with 7 or more physical cores, 160 ms with 6 and 200 ms with 5 or fewer, at a larger delay on the smaller PCs (a device buffer that already gives 60 ms or more is kept).
  - **Which voices read it:** imported voices (and the default setup) run exactly the chunk, rounded up to 20 ms. **so-vits-svc** voices take it too (their Auto is the device-buffer chunk, without the Streamer rule). **DDSP-SVC** voices take it as their block, offered from 60 ms (the engine runs 50 ms or more, in steps of 10); their Auto is the voice preset's block (100 ms, or 500 ms for Upstream). It is greyed out with a one-line reason for **bring-your-own ONNX** voices (they convert one Audio device buffer per call: set that under Advanced), **Sigil-ZS** (the preset sets the chunk), **MeanVC2** (the tick length in the voice options) and **sidecar** voices (the helper picks its own frame size; the stored chunk is not proposed to it).
  - **What changes it:** only you. It is one global setting, saved in `settings.json`; RVC presets, the **Playback feel** presets, switching voices or the execution provider and restarts never touch it. The one exception is **Apply these settings** after **Autotune**, which writes the tuner's pick.
  - **What follows it:** the automatic output buffer on the processor (below) is sized for the chunk that runs; the **Latency budget** is raised while the voice runs when it is under one chunk plus two device buffers ([Latency budget](#latency-budget)); the **crossfade** cannot be longer than one chunk, so its slider is capped at Auto's chunk (or the running engine's) instead of the 10 ms device buffer, and a crossfade change that the cap makes no different is sent to the running engine without a restart.
  - **Short chunks on a small PC:** for the Streamer on the processor, a size you set that is shorter than the chunk Auto takes on this PC gets a note under the control (see [Voice changer by tier](#voice-changer-by-tier)). Auto never does.
- **Audio device buffer** (`chunk_frames`, Advanced; it was called **Processing chunk size**): how much audio the sound card hands over at a time (512 samples, 10.7 ms at 48 kHz, by default). It is separate from the Chunk size. For imported voices only the Auto chunk is derived from it; for bring-your-own ONNX voices and WebSocket converters it is the conversion chunk itself (a WebSocket converter must expect exactly this many samples). Changing it restarts the voice changer.
- **Lookahead** (imported voices): future audio decoded into every block before it is output. It gives cleaner joins and steadier pitch and voicing and adds exactly that much delay. On the default voice (CUDA), 20 ms raised the speaker similarity to the offline conversion from 0.940 to 0.975 and cut voicing errors from 7.4 % to 3.5 % (DNSMOS 3.12 → 3.29). It also runs on DirectML since the content encoder became a fixed-shape plan there (about 1.5 ms per call instead of 35–49 ms); if a voice's encoder file cannot (an explicit fp16-I/O encoder), the panel says so and the voice runs without lookahead. Low latency uses none: a 10 ms lookahead would raise speaker similarity by about 0.03 and costs exactly 10 ms of the delay that preset exists to save.
- **Advanced pacing** (imported voices; each change restarts a running engine):
  - **Output cushion:** *Fixed* holds the output prefill from the buffer settings and is the safe choice. *Adaptive* (experimental) starts there, watches how late your blocks really are for about half a minute, then gives back what the stream never needs, during pauses in your speech first; it never holds more than the prefill, and if the output runs dry once it grows the cushion again and remembers how much it needed. It lowered the delay by about 18 ms on a calm PC but still ran dry now and then while other programs kept the PC very busy, so it is off by default. The panel shows the cushion held now.
  - **Output buffer on the processor (automatic):** an imported voice that runs on the processor and converts a long block in one call (an engine block of about 50 ms or more: the 120 to 200 ms the Streamer takes by itself, Autotune's longer blocks, or a block you set by hand) plays with at least an output buffer that covers that call: **110 / 136 ms** (prefill / maximum queue) for 120 ms blocks, 70 / 96 ms at 80 ms blocks, 145 / 171 ms at 160 ms blocks, 180 / 206 ms at 200 ms blocks. A smaller buffer runs dry during every call, and the output then throws away the audio that piles up while it recovers (the benchmark played 48-63 % of what the Streamer shape converted at the default 35 / 70 ms). Nothing you set is changed. The two named buffers under **Playback feel** (**Low latency** 35 / 70 ms and **Stable** 80 / 150 ms) are templates for short blocks and are raised to the floor (Stable's 80 / 150 ms becomes 110 / 150 ms); a larger number always wins. A buffer you typed yourself (Playback feel shows **Custom**) is your choice and is kept, however far under the floor it is, unless the run proves it cannot work: after about two seconds of conversion, if the middle one of the last 33 model calls is longer than your prefill, the output runs dry in more than half of the blocks, and from then on the voice plays with the floor (the log says so). **Holding N ms now** and the note under Playback feel show the buffer in use. The price is delay: the queue stands 75 ms higher than the default 35 / 70 ms one (110 against 35), and a continuous stream leaves the engine 265-278 ms after it entered (median) on the benchmark rig. Voices on a graphics card, blocks of 40 ms or less (Auto derives the chunk from the Audio device buffer and comes to 20 ms, which the Streamer on the processor replaces by 120, 160 or 200 ms by your cores), bring-your-own ONNX voices and the voice families are not affected. The figure assumes the converter finishes a block in about nine tenths of the block's time (the dev rig measured 0.74-0.89 on 6 threads, where it was re-run), so a faster processor holds more than it needs at long blocks, and a slower one (the 4-thread policy at 0.96, FCPE) leaves a dry hold in its slowest calls; a buffer above the first model call (118 ms or more at 120 ms blocks) plays everything but stands 20-40 ms higher. A playback device with longer callbacks raises the standing queue by itself. Set the environment variable `SIGIL_VC_BLOCK_BUFFER=0` (any case) to switch the automatic floor off. Only the benchmark and the replays of its recordings have seen this on a real device so far: a live listening check on your own audio device is still open.
  - **Device buffer mode** (it was called **Device block**): *As set (the Audio device buffer)* uses the Audio device buffer. *Divides the engine chunk (10 ms)* hands the voice 10 ms buffers (a size that divides the Chunk size, so each model call finishes exactly when its audio is due and the output buffer holds back less): about 17 ms less delay at 40 ms chunks in our paced test on CUDA, with no extra underruns. It is not the default until it has been measured on DirectML, the CPU and a real audio device. It applies to imported voices only; the panel says so for the others and shows "Running N ms device buffers.".
  - **Join search window:** how far the engine searches around each block join for the best-matching position (automatic = half the crossfade). Narrower windows lowered fidelity in our tests; leave it on automatic unless you are tuning.
  - **Content context:** how much earlier speech the content encoder hears before the block (automatic = the preset's window). More gave the encoder real context at word starts and cost a little quality elsewhere (1 s: MCD +0.08–0.10 dB); it adds no delay.
- **Silence threshold** (−90 to −10 dBFS, default −42): below it the engine skips conversion (no GPU work while you are quiet), with a short hangover so word endings are kept.
- **Voices kept loaded** (1–8) and **VRAM budget**: loaded voices switch instantly (hotkeys); the content encoder and the pitch model (RMVPE) are shared between them, so a second voice adds about 130 MB instead of a full copy. 0 MB = automatic.
- **Load time and VRAM:** a voice loads in about 1 s on CUDA and 1.1–1.3 s on DirectML (the load builds its parts side by side). The CUDA path uses a blocking wait that costs a tenth of the CPU the old spinning wait did (0.04 cores instead of 0.63 on the dev rig). RMVPE with a 64-frame plan holds about 510 MB on DirectML; a 32-frame "VRAM saver" plan exists (environment switch `SIGIL_RMVPE_PLAN_FRAMES=32`, about 200 MB less, less accurate pitch on noisy or creaky speech) but has no setting yet.
- **Levers** (Auto / On / Off each; **Set all to Auto** resets them). Off always means the proven older path. The panel shows for each lever whether this build and provider can run it and, while running, whether it is in use (with the reason when not).

  | Lever | What it does | Auto |
  |---|---|---|
  | Partial decode | Decodes only the audio that is played, not the whole context. | On |
  | Anchored noise | The same model noise for the same moment on every block, so overlapping blocks match. | On |
  | Continuous excitation | Keeps the voice's excitation phase continuous from block to block. | On |
  | Pitch cache | Tracks pitch only for new audio and reuses the rest. | On |
  | Device buffers (IOBinding) | Keeps inputs and outputs on the GPU between blocks, with no memory allocation per block. | On for CUDA and DirectML |
  | CUDA graphs | Records each block's GPU work once and replays it (NVIDIA). | On for CUDA |
  | GPU priority | A high-priority CUDA stream / DirectML queue. | Off (no measured gain yet) |
  | GPU index search | Searches the voice's index exactly on the GPU. | Off |
  | Pitch smoothing across blocks (experimental) | Decides each new pitch frame together with the frames before it, which can correct octave jumps. RMVPE only, never for singing. It did not improve pitch accuracy on real speech and slightly lowered voice similarity. | Off |
  | Steady content normalisation (opt-in) | Normalises the speech features with running statistics of your recent speech instead of each short window's own. About +0.01 voice similarity for a little spectral accuracy; it needs the content encoder's streaming variant, which is not part of the downloads yet, so the engine switches it off and says so. | Off |
  | Quality governor | Under load, lowers quality step by step (index off, then lighter pitch tracking, then a lighter denoiser) instead of stuttering, and raises it again after about 5 s of calm. The panel shows the current quality level. | Off |

  Only the governor and anchored noise apply to bring-your-own ONNX voices (partial decode too when the export supports it); anchored noise only when set to **On** (Auto keeps such a voice on its export's zero noise, which is what it was made with).
- **Noise removal** (every mode): Off, **GTCRN** (lightest, about 16–32 ms added; inside the engine for RVC voices), **DPDFNet** (best, 48 kHz, about 50–60 ms added). A stored **DeepFilterNet3** runs DPDFNet (its published model cannot stream). The models install from Models & Runtimes (`vc/denoise-*`); speech-to-text never hears the denoised copy.

#### Defaults and opt-ins at a glance

What the engine does on its own, as shipped by the voice-engine optimization ([ADR 0010](adr/0010-sve-optimization-p3b.md); numbers in [VC_BENCHMARKS.md → P3B](benchmarks/VC_BENCHMARKS.md#p3b-voice-engine-optimization-p3b-i0)). The first table is lossless or measured better and is on for everyone. The second table lists paths that can lower fidelity, or are not yet proven on every kind of hardware: they stay off, or are switches you turn on yourself. Every path also has an environment kill switch for troubleshooting (listed in the [developer guide](DEVELOPER_GUIDE.md#voice-engine-optimization-p3b)). The final scorecard of these defaults (T1–T10 on CUDA, DirectML and the processor: 37 of 67 target rows met, 30 missed, none unmeasured) is in [VC_BENCHMARKS.md → P7-B](benchmarks/VC_BENCHMARKS.md#p7-b-final-scorecard-of-the-shipped-defaults). The one default P7 changed is the processor Streamer's thread count (first table).

| On by default | What it does |
|---|---|
| Fixed-shape content-encoder plan, with pitch tracking overlapped | The content step costs 1.8 ms on CUDA and 1.5 ms on DirectML instead of 4.9 and 4.3 ms; lookahead runs on DirectML too |
| Final shapes planned at load, parallel cold start | A voice loads in about 1 s on CUDA (was 1.8 s) and 1.1–1.3 s on DirectML (was 2.5 s) |
| One shared RMVPE plan, built once | No second copy while loading; a second hot voice adds about 130 MB |
| Device-resident generator chain | Identical audio, one wait fewer per call |
| Hybrid CUDA wait | The worker's CPU on CUDA falls from 0.63 to 0.04 cores |
| Arena shrink before graph capture; CPU session arena | 40 MB less on CUDA; the Streamer shape 5.9 % faster |
| Streamer preset on the processor: all physical cores but two, never fewer than 4 threads nor more than 6, at below-normal priority | The one-thread Streamer ran at RTF 2.7 and fell behind; this one converts fast enough on an idle PC (RTF p99 0.74–0.89 on the dev rig, 2.6–2.8 cores), and at 120 ms blocks, with the 110 / 136 ms output buffer they get automatically, the benchmark plays 99.1-99.8 % of the converted audio (48-63 % at the old 35 / 70 ms; [details](benchmarks/VC_BENCHMARKS.md#re-run-with-the-floor)); under the game stand-in the 6-thread policy still reads RTF p99 1.04 and 305 ms end to end (p99 with the benchmark's SwiftF0; 288 ms on the default pool, 295 ms on an explicit 6 threads; the button's own FCPE reads 348 ms on the 8-core rig), so it is not called real time under a game. With **Chunk size** on Auto the Streamer on the processor takes a chunk by itself (120 ms from 7 physical cores, 160 ms with 6, 200 ms with 5 or fewer) with the output buffer that block gets; the 20 ms chunk the default Audio device buffer derives is not real time there. The thread count never changes the audio. `SIGIL_VC_STREAMER_THREADS=1` restores one thread, `SIGIL_VC_STREAMER_BLOCK=0` the old block rule, `SIGIL_VC_PHYSICAL_CORES=<n>` (1-64) makes the voice changer's processor rules (the thread count and this block) read n physical cores instead of the probe's, for a benchmark on a slice of the cores |
| Parallel SwiftF0 / RMVPE fusion lane | Fusion's pitch step 4.7 → 3.1 ms on CUDA |

| Off by default (switch or opt-in) | Why |
|---|---|
| 32-frame RMVPE "VRAM saver" (environment switch only) | About 200 MB less on DirectML, but less accurate pitch on noisy or creaky speech |
| fp16 RMVPE | Failed its accuracy check on DirectML |
| Pitch-cache priming, onset continuity, filled pitch calls | Structural gains only; the evidence is mixed, so no default until per-frame proof |
| Cross-tick pitch decoding, steady content normalisation, 1 s content context | Did not help, or cost a little fidelity |
| Adaptive output cushion | 18 ms less delay on a calm PC, but ran dry under heavy load |
| 10 ms device buffers (**Device buffer mode** *Divides the engine chunk*; recommended next, not the default yet) | 9–16 % less delay on DirectML and CUDA in P7-B's interleaved sessions, with compute, processor use and underruns unchanged; it needs a check on a live audio device and a real microphone first |
| Shared DirectML queue | Its content stage got slower under load |
| 10 ms lookahead for Low latency; 10 ms crossfade; narrower join search | Spends the delay the preset exists to save, or lowers fidelity |
| GPU prewarm | Holds a 338 MB CUDA context for the whole session |
| A thread-count field for the Streamer | The preset picks 4–6 threads from your physical cores; more than 6 (or the default pool of 8) is no faster and costs 0.5–0.6 more cores. `SIGIL_VC_STREAMER_THREADS=<n>` overrides it for troubleshooting |

#### Status, readouts and live changes

- The status panel shows the provider actually in use (with the reason for any fallback), mic-to-speaker latency (estimated from the per-block timeline, including noise removal and lookahead), processing p50 / p95 / p99, skipped audio, playback gaps, VRAM, the quality level and the levers and precision in use. TensorRT-RTX build progress appears per component. Two rows give the sizes: **Chunk** (what the engine converts per call) and **Device buffer** (what the sound card hands over). For a voice whose chunk is at least six device buffers long (the 120 to 200 ms chunks of the Streamer on the processor) a **Conversion per chunk (p50 / p99)** row times the model calls themselves, because the per-tick percentiles are mostly the idle ticks between calls and hide a slow call; the verdict then reads "slower than real time" when the typical call takes as long as the chunk and "spikes" when the slowest one does. A healthy long-chunk voice therefore no longer reads as slower than real time, as it did when its 120 to 200 ms calls were judged against the 10 ms device tick.
- **Switching voices while running needs no restart:** a loaded voice switches within one block with a crossfade; another voice loads in the background while the current one keeps playing. The F0 method and pitch range also switch without a restart. Changes to the engine's shape (Chunk size, lookahead, levers, precision, context, provider, noise removal, devices) restart the engine once.
- **Voice hotkeys** (global, work while a game has focus): start / stop the voice changer, next / previous voice, and voice hotkeys 1–4 (each set to a voice in the panel; a voice hotkey also switches to RVC voices).
- **Devices are followed:** a microphone the capture keeps reconnecting is reported after 1 s, and a USB microphone Windows renamed on another port is found again; outputs and the monitor retry every 250 ms and follow the system default when set to it. Messages: `device_lost`, `device_restored`, `denoiser_warning`; `rvc_load` while running means a voice switch failed and the current voice keeps playing.

#### Benchmark and tuning

- **Autotune** (Quick or Full): tries execution providers, chunk sizes, F0 methods, precision, context, the GPU levers (under a simulated game load) and, on the CPU, the number of threads, on a test clip; it picks the lowest-latency configuration that stays real time with headroom (p99 real-time factor ≤ 0.5) and full quality, and stores the result for this PC and driver. **Apply these settings** changes only the fields the tuner may set (it can also set the RVC preset and worker priority for the CPU thread policy). The Chunk size is among them: it is the only way anything but you changes that setting.
- **Benchmark this PC:** a paced run of the selected voice with your current settings on a simulated microphone and speaker (48 kHz), showing speed, steadiness, latency, CPU and VRAM.
- **Live recording:** records how the running voice changer performs during a real session (up to 10 minutes; no audio, no file paths) into `app_data/bench/results/live-<time>/`, to share as evidence from other PCs.

### Zero-shot, singing and sidecar voices (voice families)

Besides RVC voices, **RVC voices** mode runs other kinds of voices on the same engine, with the same latency budget, metrics, hotkeys, TTS mixing and hot-voice switching. In the voice library, **New voice…** (shown when this build runs at least one family) opens a form per family; each family is labelled with what it is (**Zero-shot**, **Singing**, **Experimental**) and the execution providers it runs on. A voice that cannot run on the selected provider says so and offers **Use CUDA / DirectML / CPU**. Family options are saved with the voice and apply the next time it loads (the panel offers **Restart now** while the engine runs). Measured delays and quality: [docs/benchmarks/VC_BENCHMARKS.md → P5 voice families](benchmarks/VC_BENCHMARKS.md#p5-voice-families).

| Family | What it is | Made from | Runs on | Delay (measured on an RTX 4090) |
|---|---|---|---|---|
| **Sigil-ZS** | Sounds like any voice from a short clip, no training (Seed-VC real-time inside Sigil's engine) | A reference voice from the voice library | CUDA, DirectML (about 1.1–1.4 GB VRAM) | Low delay about 250 ms; Upstream about 350 ms |
| **DDSP-SVC 6.3** | A trained singing voice; follows your pitch from E2 to D#7 | Its `model_*.pt` with `config.yaml` | CUDA, DirectML | Singing about 185–215 ms; Upstream about 615 ms |
| **MeanVC2** (experimental) | Zero-shot on the CPU, leaving the GPU free (16 kHz, trained on Mandarin) | A reference voice plus a speaker embedding | CPU (on any provider setting) | 80 ms tick about 290 ms; 160 ms tick about 350 ms |
| **so-vits-svc 4.0 / 4.1** (experimental) | A so-vits-svc voice exported to ONNX | Its `.onnx` export | CUDA, DirectML, CPU | Your Chunk size (no partial decode: more context costs more) |
| **Sidecar** (experimental) | X-VC, StreamVoiceAnon, seed-vc-realtime or your own server, as a separate program on this PC | A recipe or a `ws://` address on this PC | Your Python environment's GPU | Seed-VC Low delay about 300 ms; its default about 420 ms |

The delays are engine-added (microphone block in to converted audio out, including the output buffer these voices need), on an otherwise idle fast GPU; a slower GPU adds its compute time. Families that compute a whole block at once play with at least the output buffer that block needs, whatever the output buffer setting says, because a smaller buffer runs dry while the block computes: DDSP-SVC 70 ms (80 ms on DirectML, 100–150 ms for Upstream), Sigil-ZS 80–95 ms (Low delay: 95 ms on CUDA, 85 ms on DirectML; Lean: 80 ms, the **Stable** buffer; Upstream 120–125 ms), MeanVC2 80 ms with the 160 ms tick. The latency readout includes it.

#### Reference voices (voice library)

Zero-shot voices (Sigil-ZS, MeanVC2, sidecar recipes that take a reference) convert toward a **reference voice** from the shared voice library, which TTS voice cloning will use too. Record one from the picker (**Add a voice…**) or import a WAV, FLAC, Ogg or MP3 file:

- Speak 5–10 s in a quiet room, one speaker, no music; Sigil-ZS uses the 4 s with the most speech (quiet ends are ignored). Recording stops by itself at 12 s.
- Silence at the ends is trimmed, long pauses are shortened and the level is evened out (−20 LUFS). The recorder shows the level, clipping and how much speech it heard; a take with no speech or too little speech cannot be saved, other issues (clipping, noise) are warnings.
- A transcript is made in the background with the speech-to-text models already installed (nothing is downloaded); you can edit it, and your edit wins.
- **Re-record** replaces the clip in place and keeps the voice's id, so voices using it pick up the new clip. Deleting a reference voice leaves the voices that used it asking for another one.
- Clips live in `%AppData%/com.sigil.app/voices/<id>/` (`ref.wav`, `meta.json`, and per-engine caches of what was computed from the clip).
- A voice's reference can be switched while the voice changer runs (the voice reloads in the background and crossfades).

#### Sigil-ZS (zero-shot)

- **Install** `vc/zs-seed-vc` under Models & Runtimes (**Import** from a local export folder, `scripts/vc-families/seed_vc/export.py`; it is not downloadable yet), then **New voice… → Sigil-ZS** and pick or record the reference voice.
- **Presets:** **Low delay** (default: 100 ms blocks; the preset sets the chunk, so **Chunk size** is greyed out for this voice; 6 diffusion steps with guidance, 4 s of the reference; matched or beat upstream Seed-VC's offline speaker similarity on our test clips), **Lean** (4 steps, no guidance, 3 s: lighter, for slower GPUs, less similar), **Upstream** (180 ms blocks, 10 steps; more delay, chosen by hand only). Steps, guidance and reference length can be set per voice.
- Runs on CUDA or DirectML, not the CPU. Half-precision switches exist but change the sound (the fp16 diffusion model does not match the full-precision audio); keep them off unless the GPU cannot keep up.
- **Tip:** when no game is running, an idle NVIDIA GPU lowers its clocks for this bursty work and blocks can arrive late (playback gaps); setting NVIDIA's **Power management mode** to **Prefer maximum performance** for Sigil.exe avoids it. Sigil never changes driver settings itself.

#### DDSP-SVC singing voices

- **Install** `vc/ddsp-svc-6.3` (Sigil's template graphs), `vc/vocoder-pc-nsf-hifigan-44k` and the ContentVec content encoder (Models & Runtimes, **Import** while they are not downloadable).
- **Import** a DDSP-SVC 6.3 voice with **New voice… → DDSP-SVC**, or drop its `model_*.pt` on the import box (it opens the DDSP-SVC form): the `.pt` with its `config.yaml` in the same folder, the folder itself, or a `.zip` holding both (the highest training step wins). The checkpoint is read by the same isolated, allow-list reader as RVC voices; nothing in it is executed.
- A new DDSP-SVC voice starts in the **Singing** pitch range with **CREPE tiny**. Keep a method that reaches the top notes: CREPE, SwiftF0 or Fusion. RMVPE reads notes above about D6 (1.15 kHz) an octave low and FCPE reads them up to two semitones flat; the panel warns about both.
- **Chunk size** sets the voice's block: sizes from 60 to 200 ms are offered, and Auto is the preset's block (100 ms for **Singing**, 500 ms for **Upstream**).
- **Presets:** **Singing** (100 ms blocks, 10 flow steps; about 185–215 ms), **Singing, fast** (5 steps; barely faster on CUDA), **Upstream** (upstream's 500 ms window, 20 steps; about 615 ms).
- Runs on CUDA and DirectML, not the CPU (it cannot keep up there).

#### MeanVC2 voices (experimental)

- Zero-shot conversion on the CPU (about 0.2–0.3 of one core), so the GPU stays free for the game. Output is 16 kHz, and the model was trained on Mandarin, so other languages may sound less natural. It follows the reference speaker's intonation rather than yours.
- **Install** `vc/meanvc2` (a local export from `scripts/vc-families/meanvc/export.py`). The speaker embedding needs one of two manual steps, because the speaker encoder's weights are a Google Drive download Sigil cannot fetch: import a speaker-encoder `model.onnx` as `vc/meanvc2-speaker`, or download `wavlm_large_finetune.pth`, run upstream's `preprocess/extract_spk_emb.py` on the reference clip and copy the result as `speaker.npy` into the voice's folder (`%AppData%/com.sigil.app/voice-changer/models/<id>/`).
- **Options:** steps 1 or 2 (default 2); tick 80 ms (default, about 290 ms; the tick length is this voice's chunk, so **Chunk size** is greyed out) or 160 ms (a little less CPU, more delay; it always plays with at least the **Stable** output buffer, 80 / 150 ms; to save CPU without the extra delay use 1 step). Controls: formant, volume envelope (rms_mix; otherwise MeanVC2 speaks at its own level) and the silence threshold.

#### so-vits-svc voices (experimental)

- **New voice… → so-vits-svc** with the `<name>_SoVits.onnx` export (or its folder, or a `.zip`). The training `config.json` and the MoeVS `<name>.json` are picked up from the same folder or one level up; a FAISS `.index` enables **Cluster / index mix**.
- Runs so-vits-svc 4.0 / 4.1 exports from `onnx_export.py` / MoeVS (including multi-speaker and speaker-mix exports) and w-okada's so-vits 4.0 exports, with `vec768l12` / `vec256l9` content (ContentVec), in fp32. Refused, with the reason: training checkpoints (`G_*.pth`), so-vits 3.0, other speech encoders (hubert-soft, whisper-ppg, …), fp16 exports; `kmeans_*.pt` and `feature_and_index.pkl` are skipped (Python pickles are never read).
- Controls: pitch, formant, volume envelope, speaker, variation (the default equals so-vits' noise scale 0.4) and the silence threshold. These exports cannot decode partially, so extra context costs compute every block. **Chunk size** applies as for imported voices (planned to 20 ms); Auto follows the Audio device buffer.

#### Sidecar voices (research engines)

- **New voice… → Sidecar** with a recipe: **X-VC** and **StreamVoiceAnon** (privacy mode: makes your voice unrecognizable) need a GPU with 12 GB or more and their checkpoints installed in your own Python environment (several GB; see each script's docstring), **seed-vc-realtime** runs Sigil's Seed-VC export in Python, **Custom** connects to your own protocol v2 server at a `ws://` address on this PC (nothing else is accepted). Recipes take the Python interpreter, the model folder, extra arguments and a per-block timeout.
- Sigil starts the recipe's script itself (bound to 127.0.0.1 with a random key) and stops it when the last voice using it unloads. Release builds ship the scripts in `resources/neural-sidecar`; a development checkout uses `scripts/neural-sidecar`, and copies in `%AppData%/com.sigil.app/voice-changer/sidecars` win.
- A sidecar voice never waits: each frame is waited for up to the timeout (by default the latency budget), then muted and cancelled, and its late reply is dropped; after a stall it is back to its normal delay within a pause or two. The older **WebSocket** mode ("Another program over WebSocket") is unchanged and still waits per chunk.
- The helper answers with a frame size of its own, so **Chunk size** is greyed out for sidecar voices and the stored size is not proposed to the helper.
- Measured with Seed-VC on an RTX 4090: about 300 ms heard with `--preset sigil_low` (100 ms frames), 400–440 ms with the recipe's default upstream preset, on CUDA and DirectML. Wire format: [`scripts/neural-sidecar/PROTOCOL_V2.md`](../scripts/neural-sidecar/PROTOCOL_V2.md).

#### w-okada / wok000 files

- Bring-your-own ONNX RVC exports from w-okada read their metadata's `embedder`: Japanese HuBERT voices use `hubert-jp-{h,f}.onnx` next to the voice or in `voice-changer/base` when `vc/embedder-hubert-ja` is not installed.
- wok000's `contentvec-f/-h.onnx` and `hubert-jp-f/-h.onnx` serve both v1 (256) and v2 (768) voices as the content encoder; `rmvpe_20231006.onnx` and `rmvpe_s.onnx` work as the RMVPE model.
- Files of another kind are refused by what they are: VCClient's fused `rmvpe2_silence.onnx`, MMVC and LLVC models, so-vits, DDSP-SVC and Seed-VC / MeanVC parts, vocoders, pitch models set as the content encoder (and the reverse), PyTorch, JSON, NumPy and GGUF files.
- **FCPE** also accepts mel-input exports (Sigil computes torchfcpe's 128-band log-mel; latent or F0 output) and the community `[1, N, 1]` waveform export; RMVPE exports set as FCPE are refused.

### Using another voice changer over a virtual cable

Sigil's own voice changer is optional. If you prefer another one (w-okada VCClient, RVC WebUI, Applio or a VST host), run it beside Sigil and connect them with a virtual audio cable such as VB-Cable. Sigil does not need to know about it:

- **Your voice in VRChat:** the other voice changer reads your real microphone and plays into the cable's playback end; VRChat's microphone is the cable's recording end.
- **Captions:** keep Sigil's speech-to-text on the **real microphone** (the STT inspector's input device, or **Settings → Audio → Default input device**), so it hears your natural voice and not the converted one. Sigil opens the microphone in WASAPI shared mode, so both programs can use it; if the other program insists on exclusive access, switch it to shared mode.
- **Text-to-speech in VRChat:** set the TTS panel's **Output device** to the cable's playback end. To have the other voice changer convert the speech too, send it to the playback device that program listens to instead.
- **Measuring the chain:** the benchmark's end-to-end loopback and compare modes can time any external chain over a cable ([benchmarks/README.md](benchmarks/README.md), [`scripts/vc-bench/README.md`](../scripts/vc-bench/README.md)); it plays audio to the devices you chose, so never point it at a cable VRChat is using.

### Feature parity (conceptual vs VCClient)

| VCClient-style area | VCClient (typical) | Sigil |
|--------------------|--------------------|--------|
| **Device chunk / batch size** | Client `inputChunkNum` × fixed block size | **Audio device buffer**, **`chunk_frames`** (Advanced; it was called **Processing chunk size**): allowlisted sizes from **160** up to **48000** (~**1 s** at 48 kHz) frames per channel, with **coarser steps** above 4096 for very slow hosts (see `ALLOWED_CHUNK_FRAMES` / `VOICE_CHANGER_CHUNK_FRAMES`). Changing value **restarts** the engine. **WebSocket / ONNX** should match server/model shape when possible; Sigil adapts when it can. |
| **Conversion chunk** | Client chunk size | **Chunk size** (`block_ms`, RVC settings directly under **Preset**): **Auto** or 20 to 200 ms in steps of 20, how much audio an imported voice converts per model call, independent of the device buffer above (only Auto's number derives from it). One global setting that presets and voice switches never change. See [Engine settings](#engine-settings-and-levers). |
| **Pitch** | Server + client | **±24** semitones. Phase vocoder on the mic in Built-in, or on converter/model **output** after WebSocket / ONNX returns audio (same post-DSP chain as brightness / vocoder refinement). In **RVC** mode the shift is applied to the model’s F0 instead. |
| **Formant / brightness** | Model-dependent | **Brightness** (formant tilt) [−1, 1] on processed audio in every mode; RVC voices also have a true **Formant** shift (±12 semitones, pitch kept), and while it is not 0 the tilt does not apply to them. |
| **Input / output gain** | Client gain nodes | **`input_gain` / `output_gain`** (0–4) in Rust on the capture/playback path. |
| **Output buffer** | Prefill + max queue (worklet) | **`output_prefill_ms`**, **`output_max_queue_ms`** (converted to samples using **output device rate**). Independent of chunk size: they only govern the **playback deque** after processing. **Prefill** = minimum queued audio before cpal starts pulling to speakers (reduces underruns). **Max queue** = cap on queued audio; excess drops oldest samples. The cap is raised automatically to at least the ring target (the larger of the prefill and one output device callback, plus one processed chunk, one capture callback and ~1.3 ms), since a smaller cap would drop audio on every tick. Presets **low_latency** / **stable** fill templates; **custom** when edited. |
| **Silence truncation** | `volTrancate*` thresholds | Optional **RMS + consecutive silent chunks** trims playback deque toward safe minimum. |
| **Metrics UI** | buf / res | IPC **`get_voice_changer_metrics`**: **buffer_ms** (output queue), **process_ms** (EMA) and **process_p99_ms** (p99 over the last 512 blocks), **chunk_ms** (the device buffer, ms), **engine_block_ms** (the chunk the running engine converts per call), **backlog_ms** (captured audio waiting), **dropped_ms** (audio skipped by the latency budget), **underruns** (playback gaps after start) and **ep** (the execution provider actually in use), plus **process_p50_ms** / **process_p95_ms**, **call_p50_ms** / **call_p99_ms** (the model calls alone, for a voice whose chunk is at least six device buffers long; empty until 8 calls), **e2e_ms** (estimated mic-to-speaker latency), **vram_mb**, **governor_level** and **levers** (what the running engine uses). The status panel warns when processing is slower than real time, p99 spikes (judged per model call against the chunk for long-chunk voices, per device tick against the device buffer otherwise), the backlog nears the budget, or playback runs dry; while devices and the model load it shows “Starting…”. |
| **Monitor output** | Separate monitor device + gain | **Yes** — second **cpal** stream: **`monitor_output_device`** (empty = off, or device name / `"default"`), **`monitor_gain`** (0–4) on **pre-main-gain** PCM; channel layout and sample rate are **adapted** (channel mix plus a stateful band-limited resampler, so there are no periodic clicks) when the monitor device differs from the main output. |
| **Noise / AEC** | Browser constraints, VoiceFocus, etc. | **Noise removal** built in (GTCRN or DPDFNet, on the voice changer's input only; speech-to-text keeps the raw microphone). No echo cancellation; use OBS / OS / mic vendor filters for that. |
| **Wire protocol** | Socket.IO + REST | **WebSocket binary** sidecar only; see sidecar script for framing expectations. |

### Chunk size vs playback buffer (ms)

- **`chunk_frames`** (the **Audio device buffer**) controls the **device tick**: how much audio is consumed per processing callback. For a bring-your-own ONNX voice or a WebSocket converter that is also the **DSP / inference step**: larger buffers tend to **lower CPU/GPU utilization per real-time second** (fewer invocations) but add **algorithmic/step latency**. An imported voice converts its own **Chunk size** (`block_ms`) per model call whatever the device tick; the same trade holds for it (a longer chunk costs less per second of audio and arrives later), and only Auto's number is derived from the device buffer. Neither directly sets the values of `output_prefill_ms` / `output_max_queue_ms`; the output buffer of a voice on the processor is raised to what its chunk needs ([Engine settings](#engine-settings-and-levers)).
- **`output_prefill_ms`** and **`output_max_queue_ms`** control the **output ring buffer** in `OutputPlayback`: prefill is the startup cushion before the first non-silent output block; max queue bounds worst-case listening delay when production outpaces consumption. Both are translated to sample counts with the **output sample rate** (`voice_changer/playback.rs`). The lock-free playback ring is kept filled to at least the larger of the prefill and the output device’s **callback period**, plus the last processed chunk and one **capture** callback (chunks are only produced when the input device delivers, so the ring must bridge one capture period), so devices with long periods (e.g. 10 ms WASAPI callbacks) do not underrun every period. Playback starts once the queue holds the prefill or one output callback, whichever is larger; after an underrun the callback ramps back from the held sample to live audio instead of jumping.

### Latency budget

**Latency budget** (default **300 ms**, range 100–2000) is the most captured audio that may wait for conversion. When the engine falls behind (a GPU spike, a busy game), audio older than the budget is **skipped instead of queued**: the oldest whole blocks are dropped down to half the budget with a 5 ms crossfade splice, so the voice never drifts further and further behind. It applies right away, in every mode (built-in, ONNX, WebSocket sidecar, RVC), and it is also how long Sigil waits for a WebSocket converter's reply: a late reply is discarded and that block is muted (a failed conversion in any mode is muted, never the dry microphone). A voice that converts a long chunk in one go needs the budget to hold one chunk plus two Audio device buffers (the model call takes up to a chunk, and the next buffer arrives meanwhile): while it runs, a stored budget under that is **raised** to it (a 100 ms budget with 120 ms chunks used to drop most of the speech after every call), and the field says so. The default 300 ms covers every chunk up to 200 ms; a larger value is kept as it is. Watch **backlog_ms**, **dropped_ms**, **p99** and **underruns** in the status panel.

### Worker priority and MMCSS

**Voice changer priority** (applies right away; Sigil never asks for administrator rights):

| Setting | Worker thread |
|---------|---------------|
| **Automatic** (default) | NORMAL priority when the model runs on a GPU EP, BELOW_NORMAL on the CPU, so a CPU voice does not cost game frames. |
| **Protect game FPS** | Always BELOW_NORMAL, and ONNX Runtime uses at most 2 threads (the Streamer preset on the processor then runs on one thread, which is not real time; **Automatic** is also below normal on the processor and gives the Streamer its 4–6 threads). |
| **Protect voice** | NORMAL plus the Windows power-throttling (EcoQoS) opt-out. |

ONNX Runtime's own pool threads start with the same policy as the worker, so its CPU work never runs at a higher priority than the setting allows. For ONNX Runtime threads, a change applies from the next voice-changer start.

The audio callback threads (capture, output, monitor) register with MMCSS **Pro Audio**; the inference worker never does. The worker sleeps until the capture callback wakes it (no polling).

### TTS through the voice changer

Text-to-speech reaches the voice changer through the TTS panel's **Play speech on** setting (**Output device**, **Voice changer** or **Both**), which applies to every backend that plays through Sigil's TTS output (all except Web Speech). While the voice changer is stopped, speech routed to it plays on the output device instead. Speech is queued sentence by sentence **and never truncated**: the queue accepts audio while less than 60 s is waiting, and a single clip can be up to 10 minutes; **Skip** and **Stop speaking** also cut speech already queued in the voice changer (with a 5 ms fade). **Microphone while TTS plays**: **Mix with TTS** (default), **Lower the microphone** (about −12 dB, the voice changer's own mixer) or **Mute the microphone**, with a 250 ms hold between sentences. With **Mute**, Sigil also closes the voice changer's microphone at the capture hub while speech plays into it, and keeps it closed between the sentences of one reply (up to 2 s while the next sentence is still being synthesized, about 330 ms after the last one).

### Settings persistence

Rust is authoritative. Every voice-changer setting, including the **input, output and monitor devices** and the engine settings (Chunk size, lookahead, precision, levers, noise removal, voices kept loaded, VRAM budget), is saved to `%AppData%/com.sigil.app/voice-changer/settings.json` (atomic writes, a `.bak` of the last good file; a damaged file is kept as `.corrupt-<time>` and the backup is used). Each voice's **pitch, brightness, index rate, protect, F0 method, speaker id, pitch range, volume envelope and formant** are saved in `<voice folder>/slot.json` and applied when that voice is selected in RVC mode. Imported voices live in `%AppData%/com.sigil.app/voice-changer/models/<voice>/` (`manifest.json`, `weights.f32.bin`, `speakers.f32.bin`, `added.index`, `sigil-import.json`); voice hotkeys are app settings. The voice changer never starts by itself.

### ONNX execution provider (GPU acceleration)

**ONNX** mode and **RVC** mode each have their own provider setting (`onnx_execution_provider` and `rvc_execution_provider`); for RVC, **GPU device index** picks the CUDA / DirectML adapter and **VRAM cap** limits CUDA memory only (per ONNX session: the content encoder, generator and pitch extractor each get the cap, so total VRAM can reach several times the cap plus CUDA's own context). RVC also offers the **TensorRT-RTX** and **DirectML high priority** accelerators ([Engine settings](#engine-settings-and-levers)).

| Provider | When to use | Requirements |
|----------|-------------|--------------|
| **CPU** | Default; maximum compatibility | Always available |
| **CUDA** | NVIDIA GPU acceleration | GeForce 16/20 series (Turing) or newer, driver **R580+** (CUDA 13), and the **NVIDIA GPU runtime pack** (Models & Runtimes, or Settings → GPU & diagnostics). Without the pack Sigil falls back automatically. |
| **DirectML** | Windows GPU without CUDA | AMD/Intel/NVIDIA via DirectML; the installer bundles `DirectML.dll` |

A GPU request never falls back silently: sessions try **CUDA → DirectML → CPU** (a DirectML request tries DirectML → CPU), each GPU step is checked with a tiny test model before the real one loads, and the provider actually in use plus the reason for any fallback appear in the voice-changer status panel, a toast, the System log and **Settings → GPU & diagnostics**. See [GPU acceleration, diagnostics and logs](#gpu-acceleration-diagnostics-and-logs).

Local text-to-speech (Kokoro / Piper (local) and ONNX (local)) always runs on the **CPU**, on a background thread, so the UI stays responsive; the old execution-provider choice of the ONNX panel is gone. Kokoro's model runs on Sigil's ONNX Runtime, recorded under **Local TTS** in the diagnostics panel. KittenML Hub repos install as verified catalog packages through the model store; a copy downloaded by an earlier Sigil into `%APPDATA%\com.sigil.app\models\kitten\` keeps working without a new download.

### ONNX chunk shape

Models should use **float32** input/output. Sigil reads the graph’s fixed element count when present and **adapts** mic chunks when sizes differ (mono/stereo and length). Prefer a model whose `N` matches **`chunk_frames × channels`** (the Audio device buffer; a bring-your-own ONNX voice converts one buffer per call, so the Chunk size does not apply to it) for minimal extra buffering; otherwise adaptation still applies when supported.

### Related

- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit) — where the voice changer sits in the host matrix.
- [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md) — WebSocket echo server and identity ONNX helper.

---

## GPU acceleration, diagnostics and logs

**Settings → GPU & diagnostics** shows the execution providers in use per component (voice changer model, content encoder, pitch model, ONNX voice model, local TTS, speech detection, end-of-turn detection, music filter, the Chatterbox language model / decoder / vocoder, the Style-Bert-VITS2 voice and BERT), the graphics adapters, a **Test GPU acceleration** button, the NVIDIA runtime pack and **Open log folder**. The **Models & Runtimes** tab's **GPU and runtimes** card shows which acceleration options this PC supports and why not.

- **What the installer contains:** the installers (NSIS per-user, recommended, and MSI) put `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll` (ONNX Runtime 1.28) and `DirectML.dll` next to `Sigil.exe`. **They do not contain NVIDIA's CUDA runtime**, so no installer runs CUDA out of the box: CUDA needs the NVIDIA GPU runtime pack below, and without it NVIDIA GPUs use DirectML. Nothing GPU-related is load-time linked: Sigil starts on every PC.
- **CUDA requirements:** an NVIDIA GeForce 16/20 series (Turing) or newer GPU, driver **R580 or newer** (CUDA 13), and the NVIDIA GPU runtime pack. Older GPUs and drivers use DirectML automatically, with the reason shown.
- **NVIDIA GPU runtime pack** (about 1.9 GB download: CUDA 13.2 runtime, cuBLAS 13.3, cuFFT 12.2 and cuDNN 9.26 for CUDA 13; about 1.4 GB installed). Install it from **Models & Runtimes** as **NVIDIA GPU runtime (CUDA 13, cuDNN 9)** (`runtime/nvidia-cuda13`, installed into `%APPDATA%\com.sigil.app\runtimes\nvidia-cuda13\<version>\cuda13\`; also `Sigil.exe --headless fetch runtime/nvidia-cuda13`), or from **Settings → GPU & diagnostics** (the older installer, `runtimes\nvidia-cu13`). Sigil uses the Models & Runtimes pack when both are installed, so nobody needs to download it twice. Downloads come only from pinned NVIDIA URLs, one file at a time, resumable, with pinned sha256; only the DLLs are extracted. Without the pack, a CUDA request runs on DirectML, then CPU, with the reason shown. A pack removed while Sigil runs keeps working as DirectML until the next start (loaded DLLs cannot be unloaded).
- **Old runtime folders:** packs from earlier Sigil versions (for example `runtimes\nvidia-cu12`, CUDA 12) are never loaded by this version and can be deleted to free 1.4–1.9 GB (`plugin:runtime|remove_legacy_runtime`; a Remove button in the Models tab is planned).
- **Test GPU acceleration** runs `Sigil.exe --headless ep-probe` in a separate process (the app never pays for a CUDA context it does not use); the report is saved in `%APPDATA%\com.sigil.app\diagnostics\ep-probe-deep.json` and is discarded when Sigil is updated or a runtime pack is installed or removed. It reports which NVIDIA runtime install and which CUDA provider were used. A light probe runs at every start without loading any GPU runtime.
- **Log file:** `%LOCALAPPDATA%\com.sigil.app\logs\sigil.log` (10 MB × 5 rotating files `sigil.1.log` … `sigil.4.log`, panics included; **Open log folder** opens it). Warnings and errors are also forwarded to the in-app System log. `SIGIL_LOG` sets verbosity with tracing `Targets` syntax, for example `SIGIL_LOG=sigil=debug,warn` (default `sigil=info,ort=info,warn`). Useful targets: `sigil::ort` (execution providers), `sigil::runtime` (runtime packs), `sigil::models` (downloads), `sigil::audio::hub` (microphone streams opening, failing and recovering), `sigil::helper` (helper-process output), `sigil::sherpa`, `sigil::transcribe`, `sigil::stt`. `Sigil.exe --headless …` commands print to the terminal instead of the log file.
- **Settings recovery:** the app settings file is written atomically with a `.bak` of the last good version. A damaged file is kept as `settings.corrupt-<time>.bak` and the backup is restored, with a message in the System log.
- **Power throttling:** Sigil opts out of Windows 11 EcoQoS and timer-resolution throttling at startup.

---

## Text-to-speech behaviour

- **Sentence by sentence:** every backend except Web Speech splits text into sentences (Latin, Japanese, Chinese, Hindi, Arabic and other punctuation, line breaks; abbreviations and decimals never split) and plays them through Sigil's TTS output, so a long text starts speaking after its first sentence and later sentences are synthesized while earlier ones play. Server backends (Kokoro, Melo, Chatterbox, Fish, HTTP server, OpenAI-compatible) stream: speech starts before the server finishes.
- **Output:** the backend's **output device** (Windows device names; old saved names from the SAPI list are migrated automatically; empty uses **Settings → Audio → Default output device**, then the system default), its **volume**, and **Play speech on** (output device / voice changer / both, see [TTS through the voice changer](#tts-through-the-voice-changer)). Unplugging the output device fails the playing sentence once; the next one reopens the device.
- **Skip** (the current sentence) and **Stop speaking** (everything queued) are on the status bar's TTS card while sentences are queued; the service keeps running. **Stop** ends the service: queued and in-flight speech is dropped, a start that is still connecting is cancelled, and late callbacks from a replaced backend are ignored. Web Speech finishes what it is already saying.
- A single failed sentence goes to the **System log** and a rate-limited toast and is skipped; the service keeps running.
- **OpenAI-compatible (HTTP):** presets Kokoro-FastAPI (port 8880), Chatterbox (5555), Voxtral-4B-TTS / IndexTTS-2.5 / VoxCPM2 through `vllm serve <model> --omni` (8000), qwentts `tts-server` (8080). PCM is the lowest-latency format and its sample rate must match the server (24 kHz for Kokoro, Voxtral and qwentts; 48 kHz for VoxCPM2). A URL without a path gets `/v1`. **Extra fields** (JSON) are added to every request; **Timeout** 0 = automatic (local servers get longer). **Test voice** plays a sample; the voice list comes from the server when it offers one. Voice and speed changes apply from the next sentence. Kokoro-FastAPI and Chatterbox settings of the older backends run through this client too. Server voice lists show an error toast instead of an empty list when the server is down.
- **Azure** streams raw 48 kHz PCM (no MP3 artifacts; about 96 KB/s).
- **VOICEVOX / AivisSpeech:** choose the **Engine preset**. Choosing AivisSpeech resets **pitch to 0** (its voices degrade otherwise) and a **Reset pitch** button is available; the pitch slider steps by 0.01, and 0 intonation/volume values are kept. Speaker ids differ per engine; the first style is picked when the engine changes. Lines are synthesized one at a time, in order. A VOICEVOX-compatible engine on **another PC** must use `https` (the app only allows plain `http` to this PC).
- **Azure:** text with `&` or `<` is spoken (the SSML is escaped), **style** and **role** take effect (`mstts:express-as`), `xml:lang` follows the voice, and prosody left at default is omitted. A bad key stops the service; other failures only skip that line.
- **Uberduck:** v1 API with a Bearer key; lines are spoken one after another; text is limited to 10,000 characters.
- **Windows SAPI:** rates below 1.0 work (0.1 is the slowest, SAPI −10); rates from 1 to 5 map as before.
- **Timeouts** (Rust HTTP backends): health checks 10 s, voice and language lists 20 s; streamed speech waits up to 60 s plus 1 s per character (at most 10 min) for the first audio from a local or LAN server (60–120 s for a remote one) and 60 s (remote 30 s) between chunks, while waiting for playback never counts; plus a 120 s no-data limit on every connection, model downloads included. Error messages include the server's own message (JSON error field or HTML page title), with keys and credentials removed from URLs.

---

## Translation and Rewrite

- **Finals are always delivered, in order**; interims are live and latest-wins (at most one interim request in flight per sentence, and no closer together than **Minimum gap between partials**, 300 ms by default). Speech and typed text are sequenced separately. Each request has a deadline (15 s for interims, 30 s plus 10 ms per character for finals, longer for slow chains); a failed final is retried once, then reported. An untranslated line is never published as a translation (TTS and the chatbox would read it in the wrong language).
- Failed lines show a throttled notice and are written to the **System log**. The `[...]` cut-off marker STT publishes on stop is never translated.
- **Steady partial translations** (LocalAgreement, on by default): partial captions show only the words two partial translations in a row agree on, so they grow without flickering; **Also show the unconfirmed end** adds the newest words in the interim colour. The final translation of the whole sentence may still reword the line once.
- **Interim (partial) translations** follow **Interim results** of the Automatic and Language model panels. The default is automatic: on, except while the local language model runs on the processor (its variant is `cpu`, or `auto` resolved to the processor), where they cost 4.0 cores next to a game against 1.5 for final lines only; the panel shows the switch as off with a note until you choose. Ticking or unticking it is your choice from then on and is kept. Finals are always translated.
- **Wait until you finish speaking** (Smart Turn, on by default; needs Smart Turn in speech-to-text): a sentence Smart Turn marks as unfinished is joined with the next one and translated as one line, at most **Longest wait for the end of a turn** (2.5 s) after the last words, or 280 characters.
- **Context:** the last 3 lines (**Earlier lines to send**, 0–10; only lines with the same target language, forgotten after 5 minutes) and a free-text **Background** go to language models (and DeepL's context field) so names and style stay consistent.
- **Glossary:** only the terms that occur in a line are sent (whole words in spaced scripts, anywhere in Chinese / Japanese, at most 24 per line), optionally limited to a source or target language and case-sensitive. **Paste a list** accepts one term per line: term and translation separated by a tab (as copied from a spreadsheet), `=`, `->` or a comma, with optional language columns and `#` comments. Language models use it; DeepL, Firefox Translations and NLLB ignore it. Glossary, context and partial-caption settings apply to the next line while translation runs; engine settings apply on the next start.
- **Azure Translator:** the region field is optional (needed only for regional or multi-service resources); a blank source language auto-detects; the request timeout is 20 s. 401/403 are reported as key problems and not retried.
- **LibreTranslate:** the endpoint and key are passed with every call (the key in the request body, never in a URL); translation requests time out after 20 s plus 10 ms per character (at most 60 s). The **LTEngine** server preset sets port 5050, no key and source **auto** (LTEngine detects the language itself; start it with `ltengine`, larger models with `-m gemma3-12b`). Language codes are mapped to the server's own list (for example `zt` for Traditional Chinese).

### Translation backends

The language pair of the new backends is the panel's **Source language** (automatic = the language speech-to-text reports, else a guess from the script) and **Target language**; Azure, LibreTranslate and On-device keep their own pairs. Each backend's panel shows what is missing (a key, a download) under **Start**.

- **Automatic (best available)**: an ordered engine chain (default language model → Firefox Translations → NLLB), with rules per language pair. Every engine gets **Time per engine** (6 s) before the next one is tried; an engine that keeps failing is skipped for 15 s, then up to 5 minutes; after a cloud engine fails, the offline engines are tried even when they are not in the chain (**Fall back to offline engines**). Partial captions use the cheapest ready offline engine (Firefox Translations first) unless you pick one. The status line names the engine that translated the last line and warns while a fallback is in use.
- **Language model**, **Model runs on**:
  - **This PC (Sigil's model server)**: the **Local model server** block. **Model**: **Automatic (best for this PC)** (Hy-MT2 1.8B on most PCs, larger models when the graphics card has room), a catalog model (Hy-MT2 1.8B / 7B / 30B-A3B, MiLMMT-46 4B / 12B, TranslateGemma 4B / 12B, Seed-X 7B, LMT-60 4B), an opt-in specialist (CAT-Translate, Shisa, VNTL for ja → en, GalTransl for ja → zh, MADLAD-400, marked experimental), or **A GGUF file on this PC…**. **Runs on**: Automatic (CUDA 13 only when its pack is installed and the NVIDIA card and driver qualify, else Vulkan, else the processor), Vulkan, CUDA 13 or Processor; context size, layers on the graphics card and flash attention. The plan box says where the model will run and how much video memory it needs; a plan that does not fit (for example a 7B model on an 8 GB card while the voice changer runs on it) is refused with the reason, and **Start it anyway** overrides that for this session only. Missing files show a download card. **Test load**, **Stop server** and **Show server log** work while translation runs. The server stops 10 s after translation stops using it and is always ended with Sigil. The Hy-MT2 1.25-bit build is listed but needs a newer llama.cpp.
  - **Google Gemini (free tier available)**: a free Google AI Studio key; the default model is `gemini-3.5-flash-lite` (new keys cannot use the 2.5 models). **OpenRouter**, **OpenAI**: their keys and model names.
  - **Your own server (Ollama, LM Studio, …)**: presets for Ollama, LM Studio, llama.cpp and vLLM fill the URL; **Check connection** lists the server's models and picking one fills the name.
  - **Prompt format**: automatic (from the catalog or the model name) or a family (Hy-MT2, TranslateGemma, MiLMMT, Seed-X, LMT, Sakura, VNTL, MADLAD, generic); **Custom** takes a template with `{source_lang}`, `{target_lang}`, `{text}`, `{context}`, `{glossary}` and `{background}` (typos are flagged, and a preview shows the prompt). The spoken text is always sent as quoted data, never as instructions.
  - **Reasoning effort**: translation models (Hy-MT2, MiLMMT, TranslateGemma) do not reason, so keep **Model default**; for thinking models **None** or **Minimal** keeps captions fast. Also max output tokens, temperature, **Timeout per line**, **Stream the reply** (the first words appear while the model writes), partial captions and **Route** (see Rewrite below).
- **DeepL**: paste the key (keys ending in `:fx` are API Free, 500,000 characters a month; **Plan** automatic / Free / Pro). The panel shows this month's usage, the code the target is sent as (for example `EN-US`), formality (DeepL's choice / more / less, sent so languages without formality still work) and the model type. The last 3 lines go along as context, which DeepL does not bill. Partials are off by default because each one uses quota. 403 means a wrong key or plan, 456 the monthly quota is used up; the status line warns at 90 %. The key is sent only in a request header.
- **Firefox Translations (offline)** (Mozilla Bergamot): runs on the processor with no video memory and no internet; needs an AVX2 processor (an SSE2 build of the pack exists). Install the `runtime/bergamot` pack by **Import…**, then the direction models the panel lists (about 17–60 MB each). About 57 languages translate to and from English; other pairs go through English (two models). The spoken language must be known: set the source language or use a speech-to-text backend that reports it.
- **NLLB-200 (offline, 200 languages)**: `mt/nllb-200-distilled-600m-int8` (about 0.9 GB, processor only). About 1 s per caption line with 2 threads (up to 4); **Beam width** 1–4 applies to finals (beam 4 is a little better and about twice as slow); partials are off by default. The source may stay automatic (guessed from the script). Best for languages no other offline engine covers; the 1.3B model is built locally and imported.
- **Azure**, **LibreTranslate**, **On-device (WebView)**: as before (above).

### Rewrite

- **Where it is:** the **Rewrite** tab of the Speech to Text inspector (left menu → Speech → Speech to Text). The bottom-bar **Rewrite** card opens that tab, `Ctrl+2` toggles the service, and a dot on the tab shows it running (green) or starting (amber). It runs next to speech to text, so start both; choose **Rewritten text** as the text source of a caption, Text to Speech or the VRChat chatbox to use its output. The service, its settings (`services.transform`) and its topics keep their old ids (`Services.transform`), so saved profiles and layouts work unchanged; `changeTab({ tab: transform })` resolves to the STT inspector on that tab (`resolveInspectorTab`).
- **Providers:** OpenAI, OpenRouter, **Google Gemini (free tier available)** (a free Google AI Studio key; Flash-Lite answers fastest and has the most generous free limits; default `gemini-3.5-flash-lite`) and a custom OpenAI-compatible server (Ollama, LM Studio, a llama.cpp server, vLLM). **Check connection** lists the server's models and says whether the configured model is among them.
- **Max output tokens** (default 1024) and **Reasoning effort** (default "Model default", which sends nothing). Thinking models need a larger token limit; use reasoning **None** or **Minimal** to keep captions fast (Gemini 3 cannot turn thinking off, so None is sent as Minimal). Settings apply to the next sentence without a restart.
- **Stream the reply** (on by default): captions bound to the synced source ("Rewrite: original text (synced)") fill in word by word as the reply arrives; TTS, the chatbox and the "Rewritten text" source still get only the finished line, in sentence order.
- **Timeout per line** (15 s, 3–120 s): on failure or timeout the original text is sent on with a notice, so captions and TTS keep flowing; stop and timeout cancel the request itself. **Stop with stream** works, and "Show the original text in step with Rewrite" captions follow STT when Rewrite is off.
- **Connection route:** **Automatic** sends cloud requests directly and routes servers on your network through Sigil, which also reaches local servers that block the app window (Ollama and LM Studio by default); **Direct from the app window** or **Through Sigil** force one route (change it only if a server cannot be reached). Keys are sent only in request headers and are hidden in error messages.

## Streamer features and extended voices

Everything in this section is off until you turn it on or pick the backend.

### VRChat streamer quick start

The light setup for captions in OBS while you play. Every step uses a feature that is on by default or one click away, and links to its details.

1. **Speech-to-text:** pick [Built-in (on your PC)](#automatic-speech-to-text-built-in-on-your-pc), choose your spoken language (an explicit language gives the best live captions) and leave **Hardware** on **Best for my PC**. Set the start/stop and mute shortcuts under **STT → Shortcuts**; they work while VRChat has focus.
2. **Captions in OBS:** in the OBS Studio inspector copy the browser source link and paste it into an OBS Browser source (`http://localhost:<port>/…`; a saved HTML file is refused). Lower the source's frame rate. Start order does not matter and the source reconnects by itself after a Sigil restart ([Remote client](#remote-client)); OBS's **Shutdown source when not visible** is optional.
3. **VRChat:** enable OSC in VRChat. In Sigil's VRChat tab turn on **Mute captions when I mute in VRChat** (OSCQuery needs no port setup) and, for the chatbox, **Split long text into pages**; see [VRChat chatbox, mute sync and tags](#vrchat-chatbox-mute-sync-and-tags).
4. **Other players (optional):** the **Other players** tab captions only VRChat's sound; keep its hardware tier on **Processor** so the game stays smooth ([Other players' captions](#other-players-captions)).
5. **Captions in VR (optional):** the [caption window](#caption-window-and-steamvr-overlay-vrchat-tab) pinned in XSOverlay or OVR Toolkit, or the SteamVR overlay.
6. **Translation and voice (optional):** [Firefox Translations or NLLB](#translation-by-tier) need no video memory; [Kokoro / Piper (local)](#text-to-speech-by-tier) is the default voice and Edge voices fall back to it when the service fails.
7. **Check with the game running:** watch Task Manager, and see [Keeping Sigil light next to VRChat](#keeping-sigil-light-next-to-vrchat) if frames drop.

### Cloud voices with a local fallback (Edge, Gemini)

- **Edge voices (keyless):** hundreds of Microsoft Read Aloud voices in more than 70 languages, with rate and pitch. It is Microsoft's unofficial browser service, so it can change without notice.
- **Gemini TTS:** your own free Google AI Studio key; models `gemini-3.8-flash-tts` and `gemini-3.8-flash-lite-tts` (editable), 30 voices or any voice id ("Another voice…", including designed `voice_…` ids), a style instruction with presets, and an optional proxy URL (which receives the key). **Check** tests the key and model without playing audio.
- **Local fallback** (on by default; needs the Kokoro voice, which the panel offers to download): when the service answers 429, 5xx, fails to connect or stays silent for **Wait for the service** (3 s), that sentence is spoken by Kokoro at once, and the following sentences stay on Kokoro until the service is worth trying again (15 s, doubling up to 5 minutes; Google's retry delay after a 429; 10 minutes for a refused key, voice or model). Changing the settings tries the service again at once. Words already spoken are never repeated. While the fallback is on, Kokoro stays loaded (about 0.5–0.8 GB of memory) so it can take over instantly. A Kokoro voice downloaded while Edge or Gemini runs is used from the next sentence on (no restart needed). A sentence the service cannot say (for example Gemini treating a short question as a chat prompt) is spoken by Kokoro, and the service is asked again for the next one.

### Local cloning and GPU voices

All of these take their reference voice from the shared **voice library** (record or import a clean 3–10 s clip with its transcript; see [Reference voices](#reference-voices-voice-library)).

- **Pocket TTS (processor):** English, cloning on the CPU, about 98 MB plus the sherpa-onnx runtime. Without a library voice it uses its built-in voice. Options: temperature, steps, seed. While TTS runs the model stays loaded however long the pauses are; Stop frees it.
- **Chatterbox Multilingual (in Sigil):** 23 languages, needs a library voice; about 1.8 GB plus an optional 144 MB GPU decoder. A graphics card is recommended (on the processor each sentence takes a while, and sentences play once fully generated). **Runs on** (automatic: CUDA on an NVIDIA card, else DirectML on your largest dedicated graphics card with 2 GB or more (integrated graphics are skipped: they are slower than the processor for this model), else the processor; the panel shows where it actually runs and why it fell back), **Quality** (6 or 10 decoder steps), expressiveness, voice guidance (set it to 0 when the clip's language differs from the text) and variation. While TTS runs the model stays loaded however long the pauses are; Stop frees it.
- **Qwen3-TTS:** 9 built-in voices with style and emotion instructions (1.7B only), cloning a library voice (best with its transcript), or a voice designed from a description (1.7B only; a fixed seed keeps it steady). **Size** automatic picks 0.6B while the voice changer runs on a graphics card with less than 16 GB, else 1.7B. Runtime packs: Vulkan, CUDA 13 (needs **NVIDIA GPU runtime**) or processor (slower than real time); they are imported from zips built with `scripts/runtime-packs/qwentts/build.ps1` until they are published. Languages: zh, en, ja, ko, de, fr, ru, pt, es, it (others are spoken as automatic). The log target `sigil::tts::qwen3` shows the plan, the warm-up time to first audio and real-time factor, voice registrations and measured video memory; the `tts-qwen3` helper log has per-sentence `[Perf]` lines.
- **Experimental GPU voices** (backend "Experimental GPU voices"): NVIDIA **Magpie TTS** (NeMo-Speech.cpp 0.2.0: en, es, de, fr, it, vi and hi; Korean, Arabic and Brazilian Portuguese need the v2607 model, NVIDIA's own file, which only the 0.2.0 server packs can read; no Mandarin or Japanese in the official packs), **Breeze TTS 2** (voice description, direction and cloning), **OmniVoice** (CrispASR; cloning needs the consent switch, and a short AI notice is spoken before cloned speech unless you switch on that you disclose AI speech yourself; every clip carries an inaudible watermark) and **Fish S2 Pro** (s2.cpp; a low-video-memory switch; the **Fish Speech** backend is an alternative for S2 Pro). Each runs as a helper on this PC only. Some parts are Sigil-built assets that download once they are on the `assets-v1` release and are imported until then (the Magpie tokenizers and the v2607 model, the s2 packs); their scripts are in `scripts/runtime-packs/`.
- **Style-Bert-VITS2 / AivisSpeech voices (in Sigil):** Japanese text. Downloads: the Japanese BERT (653 MB) and the dictionary (29 MB). Import an `.aivmx` by dropping it on the panel or **Add voice…** (or an SBV2 ONNX export folder with `style_vectors.json`); `.aivm` and PyTorch files are refused with how to convert them. Speaker and style, style strength, tempo variation and speed; processor, DirectML or CUDA. Voices live in `%APPDATA%\com.sigil.app\tts\sbv2\<id>\`. The AivisSpeech VOICEVOX preset remains the server-based alternative.
- **GPT-SoVITS (your server):** start `python api_v2.py -a 127.0.0.1 -p 9880`; GPT and SoVITS weight paths (on the server) are applied at start or with **Apply now**; the reference is a library voice (server on this PC) or a clip path on the server (**Browse…** for a server on this PC) with its transcript and language; text language **auto** uses the caption's language; streaming modes fragments / medium / fast (medium and fast need the v2ProPlus `api_v2`); extra JSON fields (`top_k`, `temperature`, `text_split_method`, `sample_steps`).
- **Fish Audio hosted:** the Fish Speech panel's **Use Fish Audio hosted (S2.1 Pro free)** button fills `https://api.fish.audio` and `s2.1-pro-free`; add your key.
- **Windows voices** now play through Sigil's TTS output: sentence by sentence, Skip / Stop speaking, **Play speech on** device / voice changer / both, volume up to 200 %. An output saved from the old SAPI list keeps working and is converted to the device name; empty uses the default output device.
- **Kokoro, Japanese voices:** with the Japanese dictionary (`text/jpreprocess-naist-jdic`, shared with readings and Style-Bert-VITS2) installed, kanji and numbers are read with Japanese readings instead of Chinese ones.
- **Wait until I finish speaking** (TTS panel, off by default; needs Smart Turn in **Built-in** speech-to-text): a sentence you leave unfinished waits and is spoken together with the rest, at most **Longest wait after a pause** (2.5 s) after your last words. Typed text is never held; translated text follows the translation's own turn setting.

### Gemini Live speech-to-text

- Your own Gemini key (or the Translation settings' Gemini key). Audio goes to Google; the free tier may be used for training.
- **Transcribe** (default) or **Live Translate (preview)**: Live Translate publishes your captions and their translation in one session. While it runs, the Translation service skips those captions (they already have their translation).
- Custom vocabulary (up to 1,000 terms), verbatim or smart style, editable model ids and API version (v1beta; the other version is tried automatically when the token endpoint moved).
- Sessions are renewed every 9.5 minutes (editable) with an overlap, so no words are lost or repeated; a dropped connection reconnects and replays the last 30 s. Needs the desktop app (audio comes from Sigil's capture).

### Other players' captions

- **Other players** tab: by default Sigil captures only VRChat's sound (Windows 11, or Windows 10 build 20348 and later), so your own voice, TTS and music are not included. If VRChat is not running, or on older Windows, the default output is used instead while **Use the default output while the program isn't running** is on (that also includes your own TTS and music), or Sigil waits. You can also pick another program or an output device.
- Language and hardware tier (**Processor** keeps the game smooth); models install from the panel. **Translate into my language** uses the Translation tab's engine in the other direction.
- Show them with a text element, the caption window or the SteamVR overlay using the source **Other players: speech** or **Other players: translation**. They never reach the chatbox, TTS or the transcript history.

### Caption window and SteamVR overlay (VRChat tab)

- **Caption window:** an always-on-top, click-through window with your chosen captions (sources, interim text, font, colours, lines, clear time). Turn click-through off to move and resize it; **Reset position** brings a lost window back. It never takes focus from a game, closes with Sigil, and can be pinned in XSOverlay or OVR Toolkit by its title "Sigil captions".
- **SteamVR overlay:** captions attached to your head, a hand or a place in the world (with **Recenter**), with size, distance, height, opacity and colours. It needs the OpenVR runtime (install it from Models & Runtimes or the inline prompt). Sigil never starts SteamVR: it waits until SteamVR runs and lets go when SteamVR quits. Japanese, Chinese, Korean and right-to-left text are supported.

### VRChat chatbox, mute sync and tags

- Every chatbox message is queued at least **Time between chatbox messages** apart (1.3–10 s, default 1.5 s) because VRChat drops faster ones; short messages still waiting are merged. **Split long text into pages** sends text over 144 characters as numbered " (n/N)" pages (sentence ends first, then clause punctuation, then spaces; only page 1 plays the notification sound). Without pages, text is cut at a word boundary with "…".
- **Wait until I finish speaking** (chatbox): as for TTS above; translated text follows the translation's own turn setting.
- **Mute captions when I mute in VRChat:** follows VRChat's microphone mute; the microphone feed to speech engines is cut the moment the mute arrives. **Off by default** (turn it on in the VRChat tab): someone who stays muted in the game and talks only to the stream, or who uses speech as the chatbox's voice, would otherwise lose every caption. What the mute does to a sentence: the sentence you are still speaking is dropped, closed with a bare `[...]` and never published afterwards, whatever the engine still sends for it, to the chatbox, text-to-speech, OBS or the history; a sentence you had already finished, still waiting for its two-pass replacement, is published once with its first-pass text (you spoke it unmuted); nothing is published for audio that arrives while you are muted, and a sentence that was open during the mute does not come out after the unmute. Twitch, Kick and Discord posts never carry the `[...]`. A VRChat unmute lifts only a mute VRChat caused. By default Sigil uses OSCQuery (a free port VRChat finds by itself, so no clash on 9001); turning OSCQuery off listens on a fixed port (default 9001; a port in use is reported and retried). For VRChat on another PC, launch it with `--osc=9000:<this PC's IP>:<port>`. The panel shows the port, whether VRChat was found and whether messages arrive. The VRChat tab says next to each setting what Windows may ask (whether it asks is Windows' decision; nothing was measured on a profile with no rule for `Sigil.exe`): **OSCQuery** listens on UDP port 5353 on every network adapter to find VRChat, although Sigil only uses a VRChat on this PC (limiting the mDNS library to loopback was evaluated and rejected: it binds the wildcard address in its constructor); a **fixed listen port** is first checked on every address to be sure no other app holds it, before Sigil takes it on 127.0.0.1; and a **VRChat on another PC** is reached over your network (sending opens a UDP socket on all adapters when the first message goes out, and with mute sync Sigil also listens for the mute on every adapter, which has to be allowed or the mute never arrives). The notes show only while the VRChat integration is on. Choose **Allow access** (Private networks at home) and never **Cancel**: Cancel makes Windows write a Block rule for `Sigil.exe` that wins over any later allow rule, silences the question for good and breaks LAN access and mute sync later (delete that rule in Windows Defender Firewall with Advanced Security if it happened). If VRChat is not found, turn OSCQuery off and use a fixed listen port.
- **Emotion and sound tags** (SenseVoice): two Int avatar parameters (default `SigilEmotion` and `SigilEvent`; empty = not sent), reset to 0 after **Back to 0 after** (default 5 s; 0 keeps them). Emotion: neutral 0, happy 1, sad 2, angry 3, fearful 4, disgusted 5, surprised 6. Sound: laughter 1, applause 2, cry 3, cough 4, sneeze 5, breath 6, bgm 7. These numbers never change. An option shows the tags as emoji at the start of the chatbox text.

### Readings and the music filter

- **Readings** (bottom of the speech-to-text panel): hiragana and romaji for Japanese captions (needs the 29 MB Japanese dictionary) and pinyin for Chinese (marks, numbers or none; no download). Sources, interim text, and a preview of the last line. Lines without a language tag are read only when they contain kana. Readings are published on their own topic (`readings.stt`); caption displays do not show them yet.
- **Music filter (experimental):** **Built-in** speech-to-text → **Ignore music and singing (experimental)**. Segments that are only music or singing are dropped (FireRedVAD, about 2.4 MB, from the card under the toggle); speech over background music is kept, speech over loud vocal songs may be dropped.

---

## Local web server and LAN pages

The embedded web server (default port **3030**) serves `/client`, the OBS pages, `mic.html`, `inputfield.html` and the pubsub / PeerJS WebSockets. A **new install** listens on **127.0.0.1** only (**This PC only**), so the first launch raises no Windows Firewall question; in **Allowed with the LAN token** mode (a profile that already exists keeps the mode it has, and one that never saved a choice is in this mode) it listens on all IPv4 interfaces (see [LAN access](#lan-access-stage-2)).

- **Origin allow-list (LAN security stage 1):** `/pubsub` and `/peer` accept a WebSocket only from Sigil's own webviews (`tauri.localhost`), from pages this server served on this machine (`http://localhost:<port>`, `http://127.0.0.1:<port>`, this PC's LAN IPs, `COMPUTERNAME` / `COMPUTERNAME.local`), and origin-less clients on this machine. Everything else (other websites, `file://` pages, browser extensions, DNS-rebinding names) gets **403**, a log line and a "Blocked a connection" toast.
- **Kill switch:** **Settings → Local web server → Only allow Sigil pages to connect** (on by default). The environment variable `SIGIL_LAN_ORIGIN_CHECK=0/1` overrides the setting.
- **OBS** must load `http://localhost:<port>/client` or `/obs-captions.html`, not a local HTML file (a `file://` page is refused).
- Binary WebSocket frames are refused (the old binary-audio-to-Whisper path is gone), messages are capped at 256 KiB, and the server no longer relays pubsub between peers: peer text goes to the host, which applies its inbound topic allow-list and re-broadcasts.
- **The server keeps itself listening.** A bind that fails (port taken, or a Windows-reserved range) is reported once, with a toast and in **Settings → Local web server** ("Not running: …. Sigil keeps trying every few seconds."), and retried quietly after 1, 2, 5 and 10 s and then every 15 s for as long as Sigil runs, on the requested address and on the previous mode's. The first success clears the message. A listener that ends, or whose task crashes, is reported as stopped and bound again with the same delays (they start over once it ran 60 s); a change of the LAN setting starts a full round at once. OBS and browser sources reconnect when it is back, so switching LAN access or restarting Sigil is not needed. The fallback to the previous mode after a failed switch is not retried in the background.
- **A busy port at launch** (`src-tauri/src/port_precheck.rs`): the start polls the port every 250 ms and names the holder from the Windows socket table. A running Sigil with a responding window is brought to the front and the second start ends without a message. A Sigil without a window (still closing) or an unidentified holder is waited for silently up to 25 s (the 20 s exit guard plus 5 s); another program, or a port Windows refuses with "access denied" (a reserved range), gets 3 s; a Sigil whose window does not answer gets 10 s and then a box that says so. Several launches in a row take turns, so only one starts a copy. If the wait runs out, one box names the holder (program and process), the Windows error and a remedy (the `netsh int ipv4 show excludedportrange protocol=tcp` hint for a reserved range, and `--port` with the next number; the OBS links change with the port). Runs with `SIGIL_START_MINIMIZED=1` (harnesses) never raise another window and never show the box: they write one line to stderr and to `sigil.log` (`Sigil did not start: …`) and exit with code 1.
- **`/pubsub` ids belong to one socket.** A client sends a unique random id for every connection (Sigil's own pages do). A second connection with the same id replaces the first, which is closed with code **4409** ("another connection took this id"); an id that was just taken over is not taken again within 10 s (the newcomer gets 4409 "this id is in use…" and the holder stays), so two clients that share a fixed id settle instead of evicting each other. On 4409, pick another id.

### LAN access (stage 2)

- **Settings → LAN access → Other devices:** **Allowed with the LAN token** (Sigil listens on all IPv4 interfaces) or **This PC only** (Sigil listens on 127.0.0.1 only; the switch applies live, and a failed switch keeps the previous mode and shows the error).
- **Which one you start with.** A **new install** (no settings file) starts on **This PC only**, written into its new settings file at the first launch, so it never raises the Windows Firewall question and OBS on this PC, `mic.html` and the caption pages work as always. A profile that **already exists** keeps its setting, and one saved before this switch existed (no saved choice) stays on **Allowed with the LAN token**. Turn it on for a phone, a second PC or Link apps: pick **Allowed with the LAN token**, then copy the LAN links; Windows may ask once to allow Sigil on the network. While it is off, the section shows only the switch, a line about the Windows question and no token or copy buttons; the OBS tab hides **Copy LAN link (OBS on another PC)** and shows a line that says where to switch (Link apps' **Copy my address** gets the same line). A settings file that exists but cannot be read counts as an existing profile (the backup is tried, then **Allowed with the LAN token**); a missing file counts as new even if `settings.bak` is still there. Rust takes this decision itself before the window opens (the port pre-check binds the same address), so there is no moment on 0.0.0.0 first.
- Devices on **other PCs** (a phone, OBS on a second PC, Link apps) need the LAN token. **Copy LAN canvas link**, **Copy LAN captions link**, the OBS tab's **Copy LAN link (OBS on another PC)** and Link apps' **Copy my address** include it; paste a copied address into the other Sigil's address field, which also fills **Remote token**. Pages on this PC (OBS here, `mic.html`) never need it, and their links carry no token.
- **New token** disconnects everything that used the old one; copy the links again. The token is shown masked (**Show**, **Copy token**).
- **Allow devices without the token (legacy)** is a temporary escape hatch, off by default, shown with a caution.
- Refused pages say why (missing or old token, or This PC only); the host shows a "Refused a device on your network" toast. The token travels unencrypted on your network: use it on a trusted network, make a new token if a link leaked, and never tunnel the port to the internet.

---

## Interface size and the window

**Settings → Interface size** sets how large every menu, label and panel of the Sigil window is. The choices are **Auto** (the default), **S**, **M**, **L** and **X**. A line under the row says what **Auto** picked ("Auto picks 150% for the monitor Sigil is on.") or, when Windows would not describe the monitor, "Sigil could not read this monitor, so Auto stays at 100%." Only the Sigil window follows it: the OBS pages (`/client`, `obs-captions.html`), the caption window and the other browser pages keep their own sizes.

| Choice | Size | Notes |
|--------|------|-------|
| **Auto** | 100% to 200%, from the monitor Sigil is on | The default. Never below 100%, so it only ever enlarges |
| **S** | 80% | Fixed |
| **M** | 100% | Fixed; the size before Auto existed |
| **L** | 120% | Fixed |
| **X** (extra large) | 140% | Fixed |

What **Auto** picks for some monitors (the rule is the first bullet below):

| Monitor and Windows scale | Auto |
|---------------------------|------|
| 1080p, 1440p, 3440 x 1440 and 5120 x 1440 ultrawides, a 1366 x 768 laptop, all at 100% | 100% |
| 4K (3840 x 2160) at 100% | 150% |
| 4K at 125% | 120% |
| 4K at 150% or 200% | 100% |
| 3200 x 1800 at 100% | 125% |
| 3840 x 1600 at 100% | 110% |
| 5K at 100% | 200% |
| 8K at 100% | 200% (the cap) |

- **The Auto rule.** Windows' display scale already scales the interface (a 4K monitor at 150% looks like 1440p), so Auto works on the logical desktop: logical size = monitor pixels / Windows scale. The size is the smaller of (short side / 1440) and (long side / 2560), kept between 100% and 200% and rounded to the nearest 5%, so a logical 2560 x 1440 desktop is 100%. A monitor that cannot be read counts as 100%.
- **Auto follows the window.** Sigil reads the monitor before anything is drawn (so the setup wizard already has the right size), and again about a quarter of a second after the window moves to another monitor, is resized, or the Windows scale or desktop resolution changes. The browser build has no monitor and stays at 100%.
- **Picking a size.** **S** to **X** turn Auto off and keep that size until you pick **Auto** again (the fixed size is remembered meanwhile). Fixed sizes are held between 80% and 150%.
- **Existing profiles.** The first launch of this version decides once: a profile that had picked **S**, **L** or **X** keeps it with Auto off; a profile on **M**, and every new install, uses **Auto**. Where Auto is 100% (1080p, 1440p, a 4K monitor that Windows scales to 150% or more) nothing looks different.
- **What scales.** Everything sized in `rem` (all text, buttons and most panels, the small 10 and 11 px labels included). A few side panels and dialogs have a fixed pixel width and do not grow with it; they are listed in the [developer guide](DEVELOPER_GUIDE.md#interface-size-auto).

### The window

- **First launch.** A launch with no saved window for the main window (a new install, or after you remove it as described below) opens at **960 x 720** logical pixels times the Auto size, at most 90% of the monitor's work area (the screen without the taskbar), at least 800 x 600, and centred. That is 960 x 720 on 1080p or 1440p and 1440 x 1080 on a 4K monitor at 100%; it used to be a fixed 800 x 600.
- **Later launches** restore the size and place you left (`%APPDATA%\com.sigil.app\.window-state.json`), then check them once. A window larger than the work area of the monitor it comes back on (an undocked laptop, another Windows scale) is cut to fit. With **Auto** above 100%, a window smaller than 800 x 600 interface units (800 x 600 times the Auto size: 1200 x 900 on a 4K monitor at 100%) is made that large, because the menus would not fit in it. The same floor applies while Sigil runs, when you move the window to another monitor or switch to **Auto**. A window you made small stays small at an Auto size of 100% or with a fixed size, and a maximized, minimized or full-screen window is never resized.
- **See the first-launch window again.** Close Sigil and remove only the `"main"` entry from `.window-state.json`. Deleting the whole file also forgets where the caption window was put (**Reset position** in its panel brings that window back instead).

---

## Local speech package (ONNX)

When you use **ONNX (local)** or **Kokoro / Piper (local)** text-to-speech and choose a folder on this PC, Sigil loads a voice package: a directory that contains everything the in-process engine needs, not a single arbitrary `.onnx` file. Both backends run on the same engine (sherpa-onnx with espeak-ng; Kitten also on Sigil's own copy). The folder is test-loaded in a separate process first, so a damaged package cannot crash Sigil.

### Format: `speech_package_v1` (current)

The supported layout matches what the app downloads from a Hugging Face **model repository** for this backend: a small JSON manifest, one ONNX graph, and a voice data archive.

| File | Role |
|------|------|
| `config.json` | Declares model type (`ONNX1` / `ONNX2`), filenames for the ONNX model and voice data, optional speed priors and voice aliases. |
| ONNX file | Name given by `model_file` in `config.json`. |
| Voices archive | Name given by `voices` in `config.json` (NPZ with per-voice style data). |

Hub download and a local folder use the **same** layout, so you can copy an extracted repo or point at a cache directory that contains these three pieces.

### Hub download

With **Model from → Hugging Face repo**, a KittenML 0.8 repo installs as its verified catalog package (Models & Runtimes); a copy downloaded by an earlier Sigil keeps working. Any other repo is downloaded on **Start** (pinned to what the Hub reports, one file at a time).

### Official sherpa-onnx packages

An extracted official sherpa-onnx **Kokoro**, **Kitten** or **Piper** (VITS) package folder also works: the model family is detected from the ONNX metadata, and its voices appear after the first start. Other layouts are rejected with a clear error.

---

## Sidecar scripts (repo)

Example HTTP/WebSocket servers for local development and testing (the voice changer's sidecar v2 scripts also ship with the app):

- **[`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md)** — the sidecar protocol v2 server and recipes the voice changer's sidecar voices start ([`PROTOCOL_V2.md`](../scripts/neural-sidecar/PROTOCOL_V2.md); shipped with the app since P5), WebSocket voice-changer echo, identity ONNX helper, and the reference TTS HTTP server `tts_sidecar.py` (standard library only; `POST /speech`, `GET /health`, a streaming mode that sends a chunked WAV word by word, and OpenAI-compatible `/v1/models`, `/v1/audio/voices`, `/v1/audio/speech`; binds `127.0.0.1` by default and warns otherwise; `--port`, `--rate`, `--chunk-delay`, `--host`).
- **[`scripts/translation-sidecar/README.md`](../scripts/translation-sidecar/README.md)** — Docker-oriented LibreTranslate-compatible sidecar for the `translate` plugin.

Full request/response shapes and ports live in those READMEs. The [connectivity audit](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit) describes how Sigil wires Tauri plugins to these patterns. Neural TTS HTTP, voice-changer WebSocket binary PCM, and LibreTranslate-style translation HTTP contracts are maintained in the script folders above; use them for copy-paste Docker commands and API shapes—this document focuses on inspector fields and backend behavior in the app.
