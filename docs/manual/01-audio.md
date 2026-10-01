# Audio and AI

This chapter covers **speech-to-text**, **text-to-speech**, **translation**, **AI transform**, and the **voice changer**.

## Table of contents

- [Speech to text (STT)](#speech-to-text-stt)
- [Text to speech (TTS)](#text-to-speech-tts)
- [Translation](#translation)
- [AI transform](#ai-transform)
- [Voice changer](#voice-changer)

## Speech to text (STT)

![STT inspector](assets/stt-inspector.png)

### Purpose

Turn microphone (or other) audio into **text** for captions, chat pipelines, and downstream **TTS** or **translation**. STT runs on the **host** (desktop app), not on the `/client` mirror.

### Open the inspector

Sidebar → **Speech to Text**.

### Start and stop

- Use **Start** / **Stop** in the inspector or the bottom **STT** control on the dashboard strip.
- **Keyboard:** `Ctrl+1` / `Cmd+1` toggles STT start/stop (when the shortcut is not captured by the browser or OS).
- **Global shortcuts:** under **Shortcuts** in the STT inspector, set a start/stop and a mute shortcut that also work while Sigil is in the background (for example in VRChat). Press **Set**, then the key combination; **Esc** cancels, and a plain letter needs Ctrl, Alt or Shift.
- Model downloads, loading and retries show as a status line under the **Start** button. They never appear in your captions.

### Engines (summary)

Pick a **backend** in the inspector. Minimum requirements per engine are in [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#speech-to-text).

| Engine class | Notes |
|--------------|--------|
| **On this PC (automatic)** | Recommended and the default for new desktop installs. Sigil picks the best speech models for your language and your PC (live captions while you speak, then a more accurate final line) and runs them on this PC. |
| **Whisper (Rust)** | Local; model download on first use. |
| **Vosk** | Local models via Tauri + browser loader |
| **OpenAI-style HTTP** | Your own server `POST …/audio/transcriptions` (on this PC or the LAN), one request per sentence |
| **Groq Whisper (free cloud)** | Whisper large-v3 Turbo on Groq's free plan; needs a free API key and sends your speech to Groq. Sigil paces requests to stay inside the free limits. |
| **Azure / Deepgram** | Cloud; keys in settings (Deepgram uses Nova-3 by default) |
| **Native / Web Speech** | Browser speech APIs where available (does not work inside the desktop app; desktop users on it are asked once to switch to the automatic backend, else they move to Whisper) |
| **Chrome / Edge** | Optional path to open a **mic helper** page (`mic.html`) for browser-based capture |
| **Moonshine server (legacy)** | An external Moonshine HTTP server; the automatic backend now runs Moonshine inside Sigil |

### First start with "On this PC (automatic)"

1. In the STT inspector, pick **On this PC (automatic)** and your **language** (an explicit language gives the best live captions; **Detect automatically** also works).
2. The panel shows which models Sigil will use on this PC and why. Leave **Hardware** on **Best for my PC** unless you want to force the CPU or share the GPU with the voice changer.
3. Press **Start**. Sigil asks once before downloading (for English about 1.1 GB; **Smaller setup** downloads less and uses one model). Progress shows under the Start button and in **Models & Runtimes**; you can stop and resume later.
4. Speak: live words appear, then each sentence is replaced by its final, more accurate version on the same line.

Models download once and stay on this PC; your voice never leaves it. If the microphone drops out or a model crashes, the session restarts by itself.

### Audio input

- **On this PC (automatic), Whisper, Vosk, Moonshine, and local HTTP (OpenAI-style)** capture audio natively on the desktop app. They use the inspector’s **input device** when one is chosen; otherwise they use **Settings → Audio → Default input device** (a native device on the desktop app), and finally the system default.
- **Azure and Deepgram** capture in the webview, so their device pickers list **browser** microphones. Leave the picker on the default to use the system microphone.
- If a microphone is **unplugged** while STT runs, Sigil shows a toast and reconnects it by itself when it comes back; STT keeps running. With the system default selected, STT follows a change of the Windows default microphone. A saved microphone that is not plugged in is shown as **(not connected)**.

Details: [Input device resolution](../OPERATOR_REFERENCE.md#input-device-resolution-srcutilsrustaudioinputts).

### Voice detection

The automatic backend, Whisper, Moonshine, local HTTP and Groq only hear you when the built-in voice detector (Silero VAD) decides you are speaking, so silence, keyboard noise and game audio are never transcribed. Adjust it under **Voice detection** in the STT inspector: **Speech sensitivity** (lower catches quiet speech), **Pause that ends speech** (shorter feels faster but can split sentences), audio kept before speech, shortest speech kept, and the longest sentence. **Reset to defaults** restores them.

Whisper also has **Accuracy and performance** settings: the hallucination filter (drops phantom phrases such as “Thanks for watching”), the no-speech cutoff, **hotwords** (names and terms Whisper should spell correctly), a context prompt, CPU threads and how many sentences may wait. The automatic backend has **Two passes** (the accurate final line), **Smart Turn** (lets translation, TTS and the chatbox react sooner when you finish talking; captions are unchanged), **hotwords** and the hallucination filter.

### Tips

- Prefer **local** engines for latency-sensitive VRChat or game audio; **On this PC (automatic)** needs no server.
- For **OpenAI-compatible** servers, confirm they accept **non-streaming multipart** uploads (one upload per sentence).
- Use **mute** (or its shortcut) to pause captions without stopping the service. The Rust engines and Deepgram do not hear or send muted speech at all.

For timeouts, wrong microphone, and “connected but no text” behavior, see [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#stt-reliability-and-troubleshooting).

### Related

- [Translation](#translation) — can run on STT output  
- [AI transform](#ai-transform) — optional rewrite of transcribed text  
- [OBS Studio](03-integrations.md#obs-studio) — showing captions in a browser source  

## Text to speech (TTS)

![TTS inspector](assets/tts-inspector.png)

### Purpose

Speak text aloud with the built-in **Kokoro / Piper** voices, **VOICEVOX / AivisSpeech**, **Azure**, **Uberduck**, **Windows SAPI**, **browser** voices, any **OpenAI-compatible** speech server, or other configured backends. Often used with chat, macros, or transformed captions.

### Open the inspector

Sidebar → **Text to Speech**.

### Start and stop

- Inspector **Start** / **Stop** or bottom bar **TTS**.
- **Keyboard:** `Ctrl+3` / `Cmd+3` toggles TTS start/stop.

### Configuration (summary)

Each backend needs different fields (voice, endpoint, API keys). See [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#text-to-speech).

- **Kokoro / Piper (local)** is the default for new desktop installs: natural voices that run on this PC (Kokoro speaks English, Japanese, Chinese, Spanish, French, Hindi, Italian and Portuguese; Piper voices cover many more languages). Pick a voice model and a voice, agree to the one-time download in the panel (about 360 MB for Kokoro), and use **Preview** to listen. Existing users on browser or Windows voices are asked once whether to switch.
- **Play speech on** chooses where speech goes: the **output device**, the **voice changer** (your converted voice speaks the text; while the voice changer is stopped it plays on the device), or **both**. The microphone can be lowered or muted while TTS plays (voice changer → **Microphone while TTS plays**). Web Speech and Windows voices always play by themselves on their device.
- The **output device** lists your Windows audio devices; names saved by older versions are updated automatically.
- **OpenAI-compatible (HTTP)** works with Kokoro-FastAPI, Chatterbox, vLLM (Voxtral-4B-TTS, IndexTTS-2.5, VoxCPM2) and Qwen3-TTS servers: pick a preset, check the voice with **Test voice**.

Neural ONNX and local **speech package** layout are documented in [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#local-speech-package-onnx).

### In-app input

A **TTS input** bar can appear above the bottom of the canvas (host) so you can type or send phrases to TTS without leaving the editor.

### Tips

- Long text starts speaking after its first sentence; **Skip** (the current sentence) and **Stop speaking** (everything queued) are on the status bar's TTS card.
- **VOICEVOX / Kokoro-FastAPI / similar servers** need their engine running; start it before starting Sigil’s TTS service. **Kokoro / Piper (local)** needs no server. For Japanese, VOICEVOX or AivisSpeech read kanji better than Kokoro.
- **VOICEVOX / AivisSpeech:** pick the **Engine preset** (VOICEVOX on port 50021, AivisSpeech on port 10101, or a custom URL). AivisSpeech sets pitch to 0, because its voices sound worse with any other pitch; **Reset pitch** puts it back to 0.
- **Fish Speech** also works with Fish Audio's hosted API: endpoint `https://api.fish.audio`, your API key and a model such as `s2.1-pro-free`.
- **Stop** drops everything that is queued and cuts the sentence that is playing (browser and Windows voices may finish their current line).
- A line that fails to speak shows a short notice and is written to the **System log**; the service keeps running.
- **Windows** backend uses its own (SAPI) output device selection in the inspector.
- On the **client mirror**, heavy TTS may be limited; prefer driving TTS from the **host**.

### Related

- [AI transform](#ai-transform)  
- [Translation](#translation)  
- [Discord](03-integrations.md#discord) — outbound hooks may carry text  

## Translation

![Translation inspector](assets/translation-inspector.png)

### Purpose

Translate text (often from **STT** or chat) into another language for captions or downstream TTS.

### Open the inspector

Sidebar → **Translation**.

### Typical flow

1. Pick a **provider** at the top of the panel. **Automatic (best available)** tries several engines in order and falls back when one does not answer.
2. Set the **source** and **target** language. Leave the source on automatic to use the language speech-to-text recognized.
3. Follow the panel's hints: it shows what is still missing (an API key, a download) under **Start**.
4. **Start** translation. Speech-to-text (or typed text) feeds it automatically.

### Providers

- **Automatic (best available)**: an engine order (by default AI model, then Firefox Translations, then NLLB), plus optional rules for some language pairs (for example a Japanese specialist for Japanese to English). Each engine gets a few seconds (**Time per engine**) before the next one is tried; when an online engine fails, the offline ones are tried too. The status line shows which engine translated the last line and warns while a fallback is in use.
- **AI model (LLM)**: the best quality. **Model runs on**:
  - **This PC (Sigil's model server)**: Sigil downloads a translation model (Hy-MT2 1.8B, about 1.1 GB, by default) and runs it on your graphics card or processor. The **Local model server** box shows where it will run and how much video memory it needs. If it does not fit next to your game or the voice changer, it is refused with the reason; **Start it anyway** overrides that until Sigil closes. **Test load** checks it; the server stops a few seconds after translation stops and always ends with Sigil. You can also pick a larger model, a specialist, or your own GGUF file.
  - **Google Gemini (free tier available)**: create a free key in Google AI Studio.
  - **OpenRouter** or **OpenAI**: paste the key and a model name.
  - **Your own server (Ollama, LM Studio, …)**: pick the preset, press **Check connection** and choose a model from the list.
  - **Prompt format** is automatic for the catalog models; **Custom** lets you write your own template (a preview shows what is sent). Keep **Reasoning effort** on **Model default** for translation models.
- **DeepL**: paste your DeepL API key (Free keys end in `:fx`, 500,000 characters a month). The panel shows this month's usage. Partial captions are off by default because they use quota.
- **Firefox Translations (offline)**: fast, offline, no video memory. Import the Firefox Translations engine pack in **Models & Runtimes**, then download the language models the panel lists (small downloads; pairs without English go through English). It needs to know the spoken language.
- **NLLB-200 (offline, 200 languages)**: about 0.9 GB, runs on the processor, about a second per line. Best for languages the other offline engines do not cover.
- **Azure**, **LibreTranslate** (including the **LTEngine** preset) and **On-device (WebView)** work as before.

### Live settings

These apply to the next line while translation runs; no restart is needed.

- **Glossary**: names and terms and how to translate them. Only the terms in a line are sent. **Paste a list** accepts one term per line (term and translation separated by a tab, `=`, `->` or a comma), and the tester shows which terms a sample line would use. AI models use the glossary; DeepL, Firefox Translations and NLLB ignore it.
- **Context**: earlier lines (3 by default) and a short **Background** (topic, names, speaking style) help AI models and DeepL keep names and tone consistent.
- **Partial captions**: **Steady partial translations** show only the words two partial translations agree on, so captions grow without flickering; **Also show the unconfirmed end** shows the newest words too (they may still change). **Wait until you finish speaking** joins a sentence Smart Turn marks as unfinished with the next one.
- **Use keyboard input**: whether typed text is translated too.

### What to expect

- **Final** translations are always delivered, in the order you spoke. **Interim** (live) translations update as you talk and are replaced by the final one, which may reword the line once because it sees the whole sentence.
- A line that fails to translate shows a short notice and is written to the **System log**; untranslated text is never passed off as a translation.
- **Azure:** the **Region** field is optional (only regional or multi-service resources need it). Leave the source language empty to auto-detect.

### Tips

- For the best quality with no cloud account, use **AI model** on **This PC**. For older PCs or while the voice changer needs the graphics card, **Firefox Translations** or **NLLB** run on the processor.
- If a local server (Ollama, LM Studio) cannot be reached, set **Route** to **Through Sigil**.
- If output looks wrong, check the source language, and whether your text source sends interim or only final text.

### Related

- [Speech to text (STT)](#speech-to-text-stt)  
- [Text to speech (TTS)](#text-to-speech-tts)  
- [AI transform](#ai-transform)  

## AI transform

![AI transform inspector](assets/transform-inspector.png)

### Purpose

Rewrite, summarize, or stylize text with an **LLM** (OpenAI, OpenRouter, Google Gemini, or a **local** OpenAI-compatible server such as Ollama or LM Studio). Often applied to **STT** output before display or TTS.

### Open the inspector

Sidebar → **AI Transform**.

### Start and stop

- Inspector controls or bottom bar **AI**.
- **Keyboard:** `Ctrl+2` / `Cmd+2` toggles the transform service start/stop.

### Configuration

Set the **provider** (OpenAI, OpenRouter, **Google Gemini**, or your own server), **model**, **API key** (if required), and **base URL** for local or proxy setups. **Check connection** lists the models the server offers and says whether yours is among them. Prompts and options are in the inspector; keep them aligned with your stream’s tone and latency budget.

- **Google Gemini (free tier available):** create a free key in Google AI Studio (aistudio.google.com). Flash-Lite models answer fastest and have the most generous free limits.
- **Max output tokens** (default 1024) limits the reply length. Thinking models spend tokens on reasoning, so give them a higher limit.
- **Reasoning effort** (default **Model default**) controls how long reasoning models think. Choose **None** or **Minimal** to keep captions fast. If a model rejects a level, keep **Model default**.
- **Stream the reply** (on by default): captions bound to the synced AI source ("AI: Original Text (Synced)") fill in word by word as the reply arrives. TTS, the chatbox and the AI Transform source still get only the finished line.
- **Timeout per line** (15 seconds by default). On a failure or timeout the **original text** is sent on (with a short notice), so captions and TTS keep flowing.
- **Connection route:** **Automatic** sends cloud requests directly and routes servers on your network through Sigil, which also reaches local servers that block browser requests (Ollama, LM Studio). Change it to **Direct from the app window** or **Through Sigil** only if a server cannot be reached.
- Changes apply to the next sentence; no restart is needed.
- **Stop with stream** works, and “Sync timing” captions follow STT when AI is off.

### Tips

- **Local endpoints** reduce cloud cost and privacy risk; latency still depends on GPU/CPU.  
- Short, focused prompts reduce delay versus long system prompts.  
- If the service errors, check the **logs** panel and provider rate limits.

### Related

- [Speech to text (STT)](#speech-to-text-stt)  
- [Text to speech (TTS)](#text-to-speech-tts)  
- [Settings](04-system.md#settings) for related keys  

## Voice changer

![Voice changer inspector](assets/voice-changer-inspector.png)

### Purpose

Apply **real-time voice modification** through the Tauri **voice changer** plugin when enabled in your build. Availability depends on **native features** and platform support.

For DSP modes, sidecar protocol, and ONNX details, see [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#voice-changer).

### Open the inspector

Sidebar → **Voice Changer**.

### Usage

1. Open the inspector and review **device** / **model** options exposed by the plugin.  
2. **Start** processing when you are ready; **Stop** when finished.  
3. If options are missing, confirm you are on a **desktop host** build (not the `/client` page).

### RVC voices

**RVC voices (Sigil Voice Engine)** runs your RVC voices on Sigil's own real-time engine.

- **Get started:** on first use, **Add the default voice** installs a ready-to-use voice and the models the voice changer needs, then selects it.
- **Import a voice:** drop a `.pth` or `.zip` voice (RVC WebUI, Applio, Mangio, w-okada) on the import box, or use **Import…**. Sigil converts it once for its engine, usually in a second or two, and picks up the voice's `.index` file when it sits next to the `.pth` or inside the zip. Voices without pitch guidance and Applio RefineGAN / MRF voices are not supported yet; the import says so.
- **Your voices:** pick a voice from the list; **Rename**, **Duplicate**, **Delete** and **Merge** (blend two similar voices) are in each voice's menu. Voices you exported to ONNX yourself still work: put the folder under `%AppData%/com.sigil.app/voice-changer/models/` or choose **Other folder…**.
- **Permission:** the first time you start RVC, Sigil asks you to confirm you only use voices whose owners agreed.
- **Preset:** **Streamer** (CPU), **Low latency** (GPU), **Quality** (GPU, retrieval index, 20 ms lookahead for cleaner, steadier output on NVIDIA GPUs) or **Studio-TRT** (Quality on NVIDIA TensorRT-RTX). A preset fills the settings below; you can change any of them afterwards.
- **Voice settings:** **Pitch**, **F0 method** (FCPE, RMVPE, DIO, SwiftF0, CREPE tiny, PESTO, Fusion), **Pitch range** (Speech or Singing), **Index rate**, **Protect**, **Speaker**, **Volume envelope** and **Formant** (moves the voice's resonances without changing its pitch) are saved with each voice; **Variation** applies to every imported voice.
- **Engine** (expert): execution provider, precision, block size, lookahead, noise removal, silence threshold, voices kept loaded, and the engine's levers (**Auto** uses the settings Sigil's benchmarks chose for your GPU).
- **Switching voices while it runs** has no gap: the new voice fades in within one block. Voice hotkeys (the **Hotkeys** section under Voices) switch voices or start and stop the voice changer while a game has focus.
- **Benchmark and tuning:** **Autotune** finds the fastest settings that stay smooth on your PC; **Benchmark this PC** shows how fast and steady your current settings are; **Live recording** measures a real session (no audio is recorded).

Field-by-field reference: [RVC voices (Sigil Voice Engine)](../OPERATOR_REFERENCE.md#rvc-voices-sigil-voice-engine).

### Status, latency and priority

- The status area shows the device the model **actually** runs on (**CUDA**, **DirectML** or **CPU**). If a GPU could not be used, a warning box explains why (for example the NVIDIA GPU runtime is not installed; install it under **Models & Runtimes** or **Settings → GPU & diagnostics**).
- Live numbers show the estimated mic-to-speaker latency, processing time (p50, p95, p99) against the block length, the audio waiting against the latency budget, the output buffer, skipped audio, playback gaps, VRAM and the quality level, each with a warning when it matters.
- **Latency budget** (default 300 ms): when the voice changer falls behind, audio older than this is skipped instead of piling up, so your voice never drifts further behind. Applies right away.
- **Voice changer priority:** **Automatic**, **Protect game FPS** or **Protect voice**. Applies right away; Sigil never asks for administrator rights.
- **Microphone while TTS plays:** mix, lower (about −12 dB) or mute your microphone while text-to-speech is sent through the voice changer. TTS clips of any length play intact.
- **Noise removal** (every mode) cleans the voice changer's microphone copy only; speech-to-text still hears the raw microphone.
- If a microphone or speaker is unplugged, Sigil reports it and picks it up again when it returns (also a USB microphone moved to another port).
- All settings, including the input, output and monitor devices, are remembered across restarts. The voice changer never starts by itself.

### Tips

- Voice changer is **host-only**; do not expect it in the browser mirror.
- CPU/GPU load affects game and stream performance—test with VRChat or your title running.

### Related

- [Speech to text (STT)](#speech-to-text-stt) — separate from VC; both use audio paths  
- [Settings](04-system.md#settings)  
