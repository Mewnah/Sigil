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
| **Whisper (Rust)** | Local; model download on first use. The default on the desktop app. |
| **Vosk** | Local models via Tauri + browser loader |
| **Moonshine** | HTTP server (e.g. container) |
| **OpenAI-style HTTP** | Local sidecar `POST …/audio/transcriptions`, one request per sentence |
| **Groq Whisper (free cloud)** | Whisper large-v3 Turbo on Groq's free plan; needs a free API key and sends your speech to Groq. Sigil paces requests to stay inside the free limits. |
| **Azure / Deepgram** | Cloud; keys in settings (Deepgram uses Nova-3 by default) |
| **Native / Web Speech** | Browser speech APIs where available (does not work inside the desktop app; desktop users on it are switched to Whisper once) |
| **Chrome / Edge** | Optional path to open a **mic helper** page (`mic.html`) for browser-based capture |

### Audio input

- **Whisper, Vosk, Moonshine, and local HTTP (OpenAI-style)** capture audio natively on the desktop app. They use the inspector’s **input device** when one is chosen; otherwise they use **Settings → Audio → Default input device** (a native device on the desktop app), and finally the system default.
- **Azure and Deepgram** capture in the webview, so their device pickers list **browser** microphones. Leave the picker on the default to use the system microphone.

Details: [Input device resolution](../OPERATOR_REFERENCE.md#input-device-resolution-srcutilsrustaudioinputts).

### Voice detection

Whisper, Moonshine, local HTTP and Groq only hear you when the built-in voice detector (Silero VAD) decides you are speaking, so silence, keyboard noise and game audio are never transcribed. Adjust it under **Voice detection** in the STT inspector: **Speech sensitivity** (lower catches quiet speech), **Pause that ends speech** (shorter feels faster but can split sentences), audio kept before speech, shortest speech kept, and the longest sentence. **Reset to defaults** restores them.

Whisper also has **Accuracy and performance** settings: the hallucination filter (drops phantom phrases such as “Thanks for watching”), the no-speech cutoff, **hotwords** (names and terms Whisper should spell correctly), a context prompt, CPU threads and how many sentences may wait.

### Tips

- Prefer **local** engines for latency-sensitive VRChat or game audio.
- For **OpenAI-compatible** local servers, confirm they accept **non-streaming multipart** uploads; see [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md) “Upstream verification.”
- Use **mute** (or its shortcut) to pause captions without stopping the service. The HTTP engines do not send muted speech at all.

For timeouts, wrong microphone, and “connected but no text” behavior, see [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#stt-reliability-and-troubleshooting).

### Related

- [Translation](#translation) — can run on STT output  
- [AI transform](#ai-transform) — optional rewrite of transcribed text  
- [OBS Studio](03-integrations.md#obs-studio) — showing captions in a browser source  

## Text to speech (TTS)

![TTS inspector](assets/tts-inspector.png)

### Purpose

Speak text aloud with **Windows SAPI**, **Azure**, **VOICEVOX**, **Uberduck**, **browser** voices, or other configured backends. Often used with chat, macros, or transformed captions.

### Open the inspector

Sidebar → **Text to Speech**.

### Start and stop

- Inspector **Start** / **Stop** or bottom bar **TTS**.
- **Keyboard:** `Ctrl+3` / `Cmd+3` toggles TTS start/stop.

### Configuration (summary)

Each backend needs different fields (voice, endpoint, API keys). See [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#text-to-speech).

Neural ONNX and local **speech package** layout are documented in [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#local-speech-package-onnx).

### In-app input

A **TTS input** bar can appear above the bottom of the canvas (host) so you can type or send phrases to TTS without leaving the editor.

### Tips

- **VOICEVOX / Kokoro / similar** usually expect a **local HTTP** engine; start the engine before starting Sigil’s TTS service.
- **VOICEVOX / AivisSpeech:** pick the **Engine preset** (VOICEVOX on port 50021, AivisSpeech on port 10101, or a custom URL). AivisSpeech sets pitch to 0, because its voices sound worse with any other pitch; **Reset pitch** puts it back to 0.
- **Fish Speech** also works with Fish Audio's hosted API: endpoint `https://api.fish.audio`, your API key and a model such as `s2.1-pro-free`.
- **Stop** drops everything that is queued. A sentence that is already playing through the app's audio output may finish.
- A line that fails to speak shows a short notice and is written to the **System log**; the service keeps running.
- **Windows** backend uses native output device selection in the inspector.
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

1. **Start** the translation service when your pipeline is ready.  
2. Set **source** and **target** languages (options depend on configured provider).  
3. Ensure **STT** (or another text source) is producing text that feeds the same pipeline.

### Configuration

Providers and API keys vary by integration; use the fields in the inspector and any cloud account requirements. For engine-specific minimums, see [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md) where applicable.

### What to expect

- **Final** translations are always delivered, in the order you spoke. **Interim** (live) translations update as you talk and are replaced by the final one.
- A line that fails to translate shows a short notice and is written to the **System log**.
- **Azure:** the **Region** field is optional (only regional or multi-service resources need it). Leave the source language empty to auto-detect.

### Tips

- Match **language codes** to what your STT engine outputs.
- If output looks wrong, verify whether **interim** vs **final** text is being translated in your setup.

### Related

- [Speech to text (STT)](#speech-to-text-stt)  
- [Text to speech (TTS)](#text-to-speech-tts)  
- [AI transform](#ai-transform)  

## AI transform

![AI transform inspector](assets/transform-inspector.png)

### Purpose

Rewrite, summarize, or stylize text with an **LLM** (OpenAI, OpenRouter, or a **local** HTTP endpoint). Often applied to **STT** output before display or TTS.

### Open the inspector

Sidebar → **AI Transform**.

### Start and stop

- Inspector controls or bottom bar **AI**.
- **Keyboard:** `Ctrl+2` / `Cmd+2` toggles the transform service start/stop.

### Configuration

Set the **provider**, **model**, **API key** (if required), and **base URL** for local or proxy setups. Prompts and options are in the inspector; keep them aligned with your stream’s tone and latency budget.

- **Max output tokens** (default 1024) limits the reply length. Thinking models spend tokens on reasoning, so give them a higher limit.
- **Reasoning effort** (default **Model default**) controls how long reasoning models think. For Gemini 2.5 Flash through OpenRouter, choose **None**. If a model rejects a level, keep **Model default**.
- Changes apply to the next sentence; no restart is needed.
- Each sentence has 15 seconds. On a failure or timeout the **original text** is sent on (with a short notice), so captions and TTS keep flowing.
- **Stop with stream** now works, and “Sync timing” captions follow STT when AI is off.

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

### RVC mode

**RVC (BYO ONNX)** runs a voice model you export yourself.

- **Voice:** pick a voice from the list (folders under `%AppData%/com.sigil.app/voice-changer/models/`), **Rescan voices** after adding one, or choose **Other folder…**. A voice folder holds the generator (`gen.onnx` or `model.onnx`) and, optionally, the model’s `.index` file. A `manifest.json` is optional: Sigil reads the settings from the model. Sigil saves your per-voice settings (pitch, brightness, index rate, protect, F0 method, speaker) in a small `slot.json` in that folder and restores them when you pick the voice again. If Sigil cannot tell the model’s sample rate, it assumes 40 kHz and shows a warning suggesting you add a `manifest.json`.
- **Permission:** the first time you start RVC, Sigil asks you to confirm you only use voices whose owners agreed.
- **Quality profile:** **Streamer** (default, CPU, 64 ms extra context), **Low latency** (GPU, 96 ms), or **Quality** (GPU, 192 ms, index rate 0.5). A profile sets the RVC execution provider, pitch extractor, index rate, and extra context together; **Low latency** and **Quality** also set the VRAM cap (768 MB / 1536 MB), while **Streamer** runs on the CPU and leaves it unchanged.
- **Index rate** blends in the model’s FAISS `.index`; raise it above 0 when the slot has an index.
- **RVC execution provider** is separate from the ONNX mode’s provider. **GPU device index** picks which GPU CUDA or DirectML uses; **VRAM cap** only applies to CUDA, and it applies to each ONNX session (HuBERT, generator, pitch extractor) separately, so total VRAM use can reach about three times the cap.
- **Pitch** shifts the model’s F0; **Brightness** tilts the converted output.
- **F0 method:** RMVPE needs a GPU provider; on the CPU, FCPE is used instead and the inspector says so.
- Changing **extra context** or **crossfade** restarts the engine if it is running. **Crossfade** cannot be longer than one processing chunk, so its maximum follows the chunk size (about 10 ms at 512 samples).

Field-by-field reference: [RVC (bring-your-own ONNX)](../OPERATOR_REFERENCE.md#rvc-bring-your-own-onnx).

### Status, latency and priority

- The status area shows the device the model **actually** runs on (**CUDA**, **DirectML** or **CPU**). If a GPU could not be used, a warning box explains why (for example the NVIDIA GPU runtime is not installed; install it under **Settings → GPU & diagnostics**).
- Live numbers show processing time (average and p99) against the block length, the audio waiting against the latency budget, the output buffer, skipped audio and playback gaps, each with a warning when it matters.
- **Latency budget** (default 300 ms): when the voice changer falls behind, audio older than this is skipped instead of piling up, so your voice never drifts further behind. Applies right away.
- **Voice changer priority:** **Automatic**, **Protect game FPS** or **Protect voice**. Applies right away; Sigil never asks for administrator rights.
- **Microphone while TTS plays:** mix, lower (about −12 dB) or mute your microphone while text-to-speech is sent through the voice changer. TTS clips of any length play intact.
- All settings, including the input, output and monitor devices, are remembered across restarts. The voice changer never starts by itself.

### Tips

- Voice changer is **host-only**; do not expect it in the browser mirror.
- CPU/GPU load affects game and stream performance—test with VRChat or your title running.

### Related

- [Speech to text (STT)](#speech-to-text-stt) — separate from VC; both use audio paths  
- [Settings](04-system.md#settings)  
