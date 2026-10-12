# Speech and voice

This chapter covers **speech to text** (with **Rewrite**, which runs next to it), **translation**, **text to speech**, and the **voice changer**. In the app they sit together under **Speech** in the left menu.

## Table of contents

- [Speech to text (STT)](#speech-to-text-stt), including [Rewrite](#rewrite)
- [Translation](#translation)
- [Text to speech (TTS)](#text-to-speech-tts)
- [Voice changer](#voice-changer)

## Speech to text (STT)

![STT inspector](assets/stt-inspector.png)

### Purpose

Turn microphone (or other) audio into **text** for captions, chat pipelines, and downstream **TTS** or **translation**. STT runs on the **host** (desktop app), not on the `/client` mirror.

### Open the inspector

Left menu → **Speech** → **Speech to Text**. The inspector has two tabs: **Speech** (engine, language and start) and **Rewrite** (see [Rewrite](#rewrite) below).

### Start and stop

- Use **Start** / **Stop** in the inspector or the bottom **STT** control on the dashboard strip.
- **Keyboard:** `Ctrl+1` / `Cmd+1` toggles STT start/stop (when the shortcut is not captured by the browser or OS).
- **Global shortcuts:** under **Shortcuts** in the STT inspector, set a start/stop and a mute shortcut that also work while Sigil is in the background (for example in VRChat). Press **Set**, then the key combination; **Esc** cancels, and a plain letter needs Ctrl, Alt or Shift.
- Model downloads, loading and retries show as a status line under the **Start** button. They never appear in your captions.

### Engines (summary)

Pick the engine from the inspector's list, which is grouped: **Recommended** (the built-in engine), **Online (your own account or key)**, **Chrome / Edge microphone page** and **Other engines**. Minimum requirements per engine are in [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#speech-to-text).

| Engine class | Notes |
|--------------|--------|
| **Built-in (on your PC)** | Recommended and the default for new desktop installs. Sigil picks the best speech models for your language and your PC (live captions while you speak, then a more accurate final line) and runs them on this PC. Works without internet; your voice never leaves the PC. |
| **Groq Whisper (free cloud)** | Whisper large-v3 Turbo on Groq's free plan; needs a free API key and sends your speech to Groq. Sigil paces requests to stay inside the free limits. |
| **Azure / Deepgram** | Cloud; keys in settings (Deepgram uses Nova-3 by default) |
| **Chrome / Edge** | The browser's own speech recognition, run from a **microphone page** (`mic.html`) that Sigil opens in Chrome or Edge. It sends your voice to Google (Chrome) or Microsoft (Edge). In Sigil's English test it made about 21 (Chrome) and 10 (Edge) mistakes per 100 words, against about 6 for the built-in engine ([measurements](../OPERATOR_REFERENCE.md#speech-to-text)). A profile that still uses it is asked once whether to switch to the built-in engine; **Keep Chrome** / **Keep Edge** keeps it and is never asked again. |
| **Whisper (Rust)** | Local; model download on first use. |
| **Vosk** | Local models via Tauri + browser loader |
| **OpenAI-style HTTP** | Your own server `POST …/audio/transcriptions` (on this PC or the LAN), one request per sentence |
| **Native / Web Speech** | Browser speech APIs, listed only in the web build. It does not work inside the desktop app; a desktop profile still on it is asked once to switch to the built-in engine, else it moves to Whisper. |

**Which one for my PC?** Start with **Built-in (on your PC)**: it runs on the processor by default, so the game keeps the graphics card, and with fewer than 6 physical cores it uses one lighter pass. Models for the graphics card (a more accurate final line, more languages) are opt-in. What each engine downloads and costs, by kind of PC: [Choosing engines for your PC](../OPERATOR_REFERENCE.md#choosing-engines-for-your-pc).

### First start with "Built-in (on your PC)"

1. In the STT inspector, pick **Built-in (on your PC)** and your **language**. Setting the language is the biggest step for accurate captions and also saves memory; **Detect automatically** works, but is noticeably less accurate for languages such as French, German, Spanish and Russian (see [Choosing your language](#choosing-your-language)).
2. The panel shows which models Sigil will use on this PC and why. Leave **Hardware** on **Best for my PC** unless you want to force **Processor only** or let speech recognition share the graphics card with the voice changer (**Graphics card**).
3. Press **Start**. Sigil asks once before downloading (for English about 1.1 GB; **Smaller setup** downloads less and uses one model). Progress shows under the Start button and in **Models & Runtimes**; you can stop and resume later.
4. Speak: live words appear, then each sentence is replaced by its final, more accurate version on the same line.

Models download once and stay on this PC; your voice never leaves it. If the microphone drops out or a model crashes, the session restarts by itself: five quick tries (1, 2, 5, 10 and 20 s apart), then one about every 45 s until you press Stop. The status line under the Start button says it is trying again; nothing needs your attention. A microphone that Windows renamed after a USB port change ("Microphone (USB Audio)" becomes "Microphone (2- USB Audio)") is found again when only one device has that name.

### Choosing your language

Sigil never picks a language for you. It asks, and **Detect automatically** stays a choice everywhere.

- **Setup wizard (first run):** on a new install the speech step asks *What language do you speak?*, starting on your Windows language when Sigil supports it and it is not English (otherwise on **Detect automatically**); nothing is saved until you press **Continue**. The touchpoints below are what you meet afterwards, and what a profile that already existed meets.
- **STT inspector:** while the language is still on **Detect automatically** and you have not answered, a card above **Start** says "Your Windows language is French. Lock captions to French?" (not when Windows is in English, which says nothing about what you speak).
- **Before the first download:** if the language is still on **Detect automatically**, the question "Download speech models?" adds a tip: cancel, choose your language in the inspector, and press **Start** again (a set language usually downloads less).
- **While you talk:** if Sigil hears the same language in three sentences in a row, it asks "Detected French. Lock captions to French?" in the STT inspector, or as a small toast (no window pops up and nothing takes focus) when the inspector is closed. A session that mixes languages is never asked, and the card goes away again if the next sentences are in another language.

The card has three buttons. **Lock to French** saves the language: running captions restart at once when French's models are already installed (a couple of seconds without captions), otherwise it applies the next time you press **Start**, which asks before downloading what is missing; the card tells you which. Until then, captions that start by themselves (**Start on launch**) keep detecting the language, so you are never left without captions. **Keep automatic** keeps detection and stays quiet for a week (closing the toast does the same). **Don't ask again** keeps detection for good. You can always change the **Language** list in the inspector yourself.

### Audio input

- **Built-in (on your PC), Whisper, Vosk, and local HTTP (OpenAI-style)** capture audio natively on the desktop app. They use the inspector’s **input device** when one is chosen; otherwise they use **Settings → Audio → Default input device** (a native device on the desktop app), and finally the system default.
- **Azure and Deepgram** capture in the webview, so their device pickers list **browser** microphones. Leave the picker on the default to use the system microphone.
- If a microphone is **unplugged** while STT runs, Sigil shows a toast and reconnects it by itself when it comes back; STT keeps running. With the system default selected, STT follows a change of the Windows default microphone. A saved microphone that is not plugged in is shown as **(not connected)**.

Details: [Input device resolution](../OPERATOR_REFERENCE.md#input-device-resolution-srcutilsrustaudioinputts).

### Voice detection

The built-in engine, Whisper, local HTTP and Groq only hear you when the built-in voice detector (Silero VAD) decides you are speaking, so silence, keyboard noise and game audio are never transcribed. Adjust it under **Voice detection** in the STT inspector: **Speech sensitivity** (lower catches quiet speech), **Pause that ends speech** (shorter feels faster but can split sentences), audio kept before speech, shortest speech kept, and the longest sentence. **Reset to defaults** restores them.

Whisper also has **Accuracy and performance** settings: the hallucination filter (drops phantom phrases such as “Thanks for watching”), the no-speech cutoff, **hotwords** (names and terms Whisper should spell correctly), a context prompt, CPU threads and how many sentences may wait. The built-in engine has **Two-pass accuracy** (the accurate final line), **Enable interim results** (words while you are still speaking; turn it off to get each sentence once it is finished, which also saves the live model's processor time and memory, about 0.3-0.4 cores and 0.8 GB, and applies at the next start), **Smart Turn** (lets translation, TTS and the chatbox react sooner when you finish talking; captions are unchanged), **hotwords** and the hallucination filter.

### Rewrite

![Rewrite tab](assets/transform-inspector.png)

Rewrite sends each finished sentence through a **language model** (OpenAI, OpenRouter, **Google Gemini**, or a **local** OpenAI-compatible server such as Ollama or LM Studio) with your instructions: fix grammar, change the tone, shorten, stylize. It runs next to Speech to Text, so **start both**. To use the result, choose **Rewritten text** as the text source of a caption, of Text to Speech or of the VRChat chatbox.

Open it with the **Rewrite** tab of the Speech to Text inspector, or with the **Rewrite** card in the bottom bar (it opens the same tab). A small dot on the tab shows that Rewrite is running (green) or starting (amber).

- **Start and stop:** **Start rewriting** / **Stop rewriting** in the tab or on the bottom-bar card. **Keyboard:** `Ctrl+2` / `Cmd+2` toggles it.
- **Provider, model and key:** pick the **Provider** (OpenAI, OpenRouter, **Google Gemini**, or your own server) and fill in the **model name**, the **API key** (if required) and the **API URL** for local or proxy setups. **Check connection** lists the models the server offers and says whether yours is among them. **Instructions** holds your prompt; short, focused instructions answer faster than long ones.
- **Google Gemini (free tier available):** create a free key in Google AI Studio (aistudio.google.com). Flash-Lite models answer fastest and have the most generous free limits.
- **Enable context history** sends your previous sentences to the model for better context, at the cost of a few more tokens.
- **More options** (a fold at the bottom of the tab):
  - **Treat speech as text to rewrite** (on by default, recommended) keeps the model from answering spoken questions.
  - **Max output tokens** (default 1024) limits the reply length. Thinking models spend tokens on reasoning, so give them a higher limit.
  - **Reasoning effort** (default **Model default**) controls how long reasoning models think. Choose **None** or **Minimal** to keep captions fast. If a model rejects a level, keep **Model default**.
  - **Stream the reply** (on by default): captions bound to "Rewrite: original text (synced)" fill in word by word as the reply arrives. Text to Speech, the chatbox and the **Rewritten text** source still get only the finished line.
  - **Timeout per line** (15 seconds by default). On a failure or timeout the **original text** is sent on (with a short notice), so captions and Text to Speech keep flowing.
  - **Connection route:** **Automatic** sends cloud requests directly and routes servers on your network through Sigil, which also reaches local servers that block browser requests (Ollama, LM Studio). Change it to **Direct from the app window** or **Through Sigil** only if a server cannot be reached.
- Changes apply to the next sentence; no restart is needed.
- **Stop with Twitch stream** works: it stops Rewrite once the stream has stayed offline for 60 s (a dropped connection does not end it) and starts it again if the stream comes back during the same session. Captions set to **Show the original text in step with Rewrite** follow speech to text when Rewrite is off.
- **Local endpoints** reduce cloud cost and privacy risk; latency still depends on your graphics card and processor. If Rewrite errors, check the **System log** and the provider's rate limits.

### Tips

- Prefer **local** engines for latency-sensitive VRChat or game audio; **Built-in (on your PC)** needs no server.
- If frames drop in the game while captions run, see [Keeping Sigil light next to VRChat](../OPERATOR_REFERENCE.md#keeping-sigil-light-next-to-vrchat) (a smaller model, one pass instead of two; for the voice changer, **Protect game FPS** together with a graphics-card preset, because on the processor Streamer preset it makes the voice changer fall behind).
- For **OpenAI-compatible** servers, confirm they accept **non-streaming multipart** uploads (one upload per sentence).
- Use **mute** (or its shortcut) to pause captions without stopping the service. The Rust engines and Deepgram do not hear or send muted speech at all.

For timeouts, wrong microphone, and “connected but no text” behavior, see [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#stt-reliability-and-troubleshooting).

### Related

- [Translation](#translation) — can run on STT output  
- [Rewrite](#rewrite) — optional rewrite of transcribed text  
- [OBS Studio](03-integrations.md#obs-studio) — showing captions in a browser source  

## Translation

![Translation inspector](assets/translation-inspector.png)

### Purpose

Translate text (often from **STT** or chat) into another language for captions or downstream TTS.

### Open the inspector

Left menu → **Speech** → **Translation**.

### Typical flow

1. Pick a **provider** at the top of the panel. **Automatic (best available)** tries several engines in order and falls back when one does not answer.
2. Set the **source** and **target** language. Leave the source on automatic to use the language speech-to-text recognized.
3. Follow the panel's hints: it shows what is still missing (an API key, a download) under **Start**.
4. **Start** translation. Speech-to-text (or typed text) feeds it automatically.

### Providers

- **Automatic (best available)**: an engine order (by default the language model, then Firefox Translations, then NLLB), plus optional rules for some language pairs (for example a Japanese specialist for Japanese to English). Each engine gets a few seconds (**Time per engine**) before the next one is tried; when an online engine fails, the offline ones are tried too. The status line shows which engine translated the last line and warns while a fallback is in use. While the language model runs on the processor, English with French, Spanish, Portuguese and Dutch (either direction) is tried with NLLB first when it is installed: it translates those pairs better, in about 1.6 seconds a line instead of 0.4. A rule of your own for the pair, or an engine order you changed, turns that off.
- **Language model**: the best quality. **Model runs on**:
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

- **Glossary**: names and terms and how to translate them. Only the terms in a line are sent. **Paste a list** accepts one term per line (term and translation separated by a tab, `=`, `->` or a comma), and the tester shows which terms a sample line would use. Language models use the glossary; DeepL, Firefox Translations and NLLB ignore it.
- **Context**: earlier lines (3 by default) and a short **Background** (topic, names, speaking style) help language models and DeepL keep names and tone consistent.
- **Partial captions**: **Steady partial translations** show only the words two partial translations agree on, so captions grow without flickering; **Also show the unconfirmed end** shows the newest words too (they may still change). **Wait until you finish speaking** joins a sentence Smart Turn marks as unfinished with the next one.
- **Use keyboard input**: whether typed text is translated too.

### What to expect

- **Final** translations are always delivered, in the order you spoke. **Interim** (live) translations update as you talk and are replaced by the final one, which may reword the line once because it sees the whole sentence. While the local language model runs on your processor, interim translations are off by default (they would take about 4 cores next to the game, final lines only 1.5): tick **Interim results** in the Automatic or Language model panel to turn them on. On a graphics card they stay on.
- A line that fails to translate shows a short notice and is written to the **System log**; untranslated text is never passed off as a translation.
- **Azure:** the **Region** field is optional (only regional or multi-service resources need it). Leave the source language empty to auto-detect.

### Tips

- For the best quality with no cloud account, use **Language model** on **This PC**. For older PCs or while the voice changer needs the graphics card, **Firefox Translations** or **NLLB** run on the processor.
- If a local server (Ollama, LM Studio) cannot be reached, set **Route** to **Through Sigil**.
- If output looks wrong, check the source language, and whether your text source sends interim or only final text.

### Related

- [Speech to text (STT)](#speech-to-text-stt)  
- [Text to speech (TTS)](#text-to-speech-tts)  
- [Rewrite](#rewrite)  

## Text to speech (TTS)

![TTS inspector](assets/tts-inspector.png)

### Purpose

Speak text aloud with the built-in **Kokoro / Piper** voices, **VOICEVOX / AivisSpeech**, **Azure**, **Uberduck**, **Windows SAPI**, **browser** voices, any **OpenAI-compatible** speech server, or other configured backends. Often used with chat, macros, or rewritten captions.

### Open the inspector

Left menu → **Speech** → **Text to Speech**.

### Start and stop

- Inspector **Start** / **Stop** or bottom bar **TTS**.
- **Keyboard:** `Ctrl+3` / `Cmd+3` toggles TTS start/stop.

### Configuration (summary)

Each backend needs different fields (voice, endpoint, API keys). See [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#text-to-speech).

- **Kokoro / Piper (local)** is the default for new desktop installs: natural voices that run on this PC (Kokoro speaks English, Japanese, Chinese, Spanish, French, Hindi, Italian and Portuguese; Piper voices cover many more languages). Pick a voice model and a voice, agree to the one-time download in the panel (about 360 MB for Kokoro), and use **Preview** to listen. Existing users on browser or Windows voices are asked once whether to switch.
- **Play speech on** chooses where speech goes: the **output device**, the **voice changer** (your converted voice speaks the text; while the voice changer is stopped it plays on the device), or **both**. The microphone can be lowered or muted while TTS plays (voice changer → **Microphone while TTS plays**). Web Speech voices always play by themselves on the default device; every other backend, Windows voices included, can use **Play speech on**.
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

- [Rewrite](#rewrite)  
- [Translation](#translation)  
- [Discord](03-integrations.md#discord) — outbound hooks may carry text  

## Voice changer

![Voice changer inspector](assets/voice-changer-inspector.png)

### Purpose

Apply **real-time voice modification** through the Tauri **voice changer** plugin when enabled in your build. Availability depends on **native features** and platform support.

For DSP modes, sidecar protocol, and ONNX details, see [OPERATOR_REFERENCE.md](../OPERATOR_REFERENCE.md#voice-changer).

### Open the inspector

Left menu → **Speech** → **Voice Changer**.

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

### Zero-shot, singing and other voices

**New voice…** in the voice list makes voices that are not RVC voices; they run on the same engine, with the same hotkeys, switching and TTS mixing.

- **Sigil-ZS (zero-shot):** sounds like anyone from a 5–10 s clip of them speaking, with no training. Record or import the clip once in the **reference voice** picker (Sigil trims the silence and evens out the level); you can switch the reference while the voice changer runs. Needs an NVIDIA (CUDA) or DirectML graphics card; about 250 ms of delay on a fast card.
- **DDSP-SVC (singing):** import a trained DDSP-SVC 6.3 voice (`model_*.pt` with its `config.yaml`, or a `.zip`). It follows your pitch up to D#7; new singing voices start in the **Singing** pitch range with CREPE.
- **MeanVC2** (experimental): zero-shot on the processor, leaving the graphics card to your game.
- **so-vits-svc** (experimental): a so-vits-svc 4.x voice exported to ONNX.
- **Sidecar** (experimental): research engines (X-VC, StreamVoiceAnon, seed-vc-realtime) or your own server, run as a separate program on this PC; a late block is dropped, so a stalled engine never stalls your voice.

Some of these need models you import under **Models & Runtimes** first; the voice shows what is missing. Details and measured delays: [Zero-shot, singing and sidecar voices](../OPERATOR_REFERENCE.md#zero-shot-singing-and-sidecar-voices-voice-families).

### Which preset for my PC

- **Processor only:** the default **Streamer** preset (it keeps the graphics card free for the game). It runs on all your physical processor cores but two, never fewer than 4 threads and never more than 6, at below-normal priority. On an 8-core processor Sigil measured it converting each block faster than the block lasts (6 threads, and 4 threads) while nothing else was running. With **Block size** on Auto the Streamer takes blocks of 120 ms on the processor of a PC with 7 or more physical cores, 160 ms with 6 and 200 ms with 5 or fewer (the 20 ms the default chunk would give cannot keep up on any PC, and a PC with fewer cores needs a longer block, at a larger delay), and Sigil raises the output buffer by itself to match (110 / 136 ms at 120 ms, 145 / 171 ms at 160 ms, 180 / 206 ms at 200 ms; prefill / maximum queue) so that the voice plays everything it converts: on the 8-core test PC the benchmark played 99.1-99.8 % of the converted audio, about 270 ms after you spoke. A **Block size** you set, or the one **Autotune** finds for your PC, is used instead. With a game running, the 6-thread preset still falls behind on a few blocks in the test (about 2 in 100) and the delay reaches 300 ms at the worst. If the voice stutters, run **Autotune** or choose a graphics-card preset. The two named output buffers (**Low latency**, **Stable**) are raised for these voices; a buffer you typed yourself is kept unless the voice proves it cannot hold its model calls, and `SIGIL_VC_BLOCK_BUFFER=0` switches the automatic buffer off. Leave **Voice changer priority** on **Automatic**: **Protect game FPS** holds the Streamer to one thread, which falls behind.
  - With 6 or more physical cores you keep two cores for the game (a 6-core processor runs 4 threads, an 8-core one 6). **With 6 or fewer physical cores a 120 ms block falls behind while a game runs**: in Sigil's test with a stand-in for the game (about 2.4 cores busy) it skipped about a tenth of the speech on 6 cores and about a seventh on 5 cores, with a delay of 0.4 to 0.5 seconds; an 8-core processor skipped none. That is why Auto takes a longer block there, **160 ms on 6 cores and 200 ms on 5 cores or fewer**: a longer block stopped the skipping in the tests (160 ms and 200 ms on 6-core slices, 200 ms on a 4-core one; 5 cores were not run with a longer block). The price is delay (about 350 ms from microphone to speaker at 160 ms, 400 to 500 ms at 200 ms) and an output that played 96 to 99.7 % of what the voice converted. One quick check of 160 ms on 6 cores, made while the PC was busy with other work, still skipped half a percent of the speech; that was judged to be the busy PC, and 160 ms was kept for its lower delay. On a smaller PC the least speech skipped matters more than the delay. If you set a **Block size** shorter than your PC's, Sigil shows a note under **Preset**. **Autotune** was not part of that test. A graphics-card preset avoids the problem.
  - With 4 or 5 physical cores there is no core to spare: the Streamer still takes 4 threads, which is every core of a 4-core processor and all but one of a 5-core one, because fewer threads cannot keep up. On a 4-core processor a 120 ms block does not keep up even when no game is running (15 % of the blocks took longer than the block lasts), and next to the game stand-in it skipped about a fifth of the speech; Auto therefore takes 200 ms there (it stopped the skipping in one run, but with no game running that run played 96 % of the converted audio with about 2 seconds of silent gaps a minute, below the 98 % the benchmark asks for). A graphics-card preset avoids the problem; watch the game's frame rate.
  - On a slower processor of any size, run **Autotune** to find a longer block that keeps up.
- **Mid-range graphics card (8–12 GB):** **Low latency** on CUDA (NVIDIA, with the NVIDIA GPU runtime from **Models & Runtimes**) or DirectML.
- **16 GB or more:** **Quality**, or **Studio-TRT** on NVIDIA RTX.
- **Prefer another voice changer?** Run it beside Sigil over a virtual audio cable and keep Sigil's speech-to-text on your real microphone: [Using another voice changer over a virtual cable](../OPERATOR_REFERENCE.md#using-another-voice-changer-over-a-virtual-cable). What each preset costs and which engine paths are on or off by default: [Choosing engines for your PC](../OPERATOR_REFERENCE.md#voice-changer-by-tier) and [Defaults and opt-ins at a glance](../OPERATOR_REFERENCE.md#defaults-and-opt-ins-at-a-glance).

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
