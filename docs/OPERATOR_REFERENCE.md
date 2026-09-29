# Operator reference

Detailed reference for **STT/TTS backends**, **reliability**, **voice changer** behavior, and **local ONNX speech packages**. User-facing walkthroughs live in the [user manual](manual/README.md).

## Table of contents

- [STT / TTS backend minimum configuration](#stt--tts-backend-minimum-configuration)
- [Remote client](#remote-client)
- [Upstream verification (Parakeet, speaches, sherpa-onnx)](#upstream-verification-parakeet-speaches-sherpa-onnx)
- [Models & Runtimes](#models--runtimes)
- [STT reliability and troubleshooting](#stt-reliability-and-troubleshooting)
- [Voice changer](#voice-changer)
- [GPU acceleration, diagnostics and logs](#gpu-acceleration-diagnostics-and-logs)
- [Text-to-speech behaviour](#text-to-speech-behaviour)
- [Translation and AI transform](#translation-and-ai-transform)
- [Local web server and LAN pages](#local-web-server-and-lan-pages)
- [Local speech package (ONNX)](#local-speech-package-onnx)
- [Sidecar scripts (repo)](#sidecar-scripts-repo)

---

## STT / TTS backend minimum configuration

Use this when testing or supporting users. Engines are selected in the Sigil host app (inspectors).

For **timeouts, wrong microphone, and “connected but no text”** behavior across STT engines, see [STT reliability and troubleshooting](#stt-reliability-and-troubleshooting).

### Text-to-speech

| Backend | Minimum required |
|--------|-------------------|
| **Native (browser)** | Select a **voice** in Text to Speech settings. |
| **Windows** | **Output device** and **voice** (native Windows SAPI). |
| **Azure** | Speech **key**, **region/location**, **voice**. The output device is optional (empty uses the app's output device). |
| **Uberduck** | **API key** and **voice** (v1 API; the old secret is no longer used). |
| **VOICEVOX / AivisSpeech** | **Engine preset**: VOICEVOX (`http://localhost:50021`), AivisSpeech (`http://localhost:10101`) or a custom URL, plus a speaker. Plain `http` works only for an engine on this PC. |
| **Fish Speech** | **Endpoint**. Self-hosted fish-speech needs an **API key** only when started with `--api-key`. For Fish Audio's hosted API use `https://api.fish.audio`, your key and a **model** such as `s2.1-pro-free`. |
| **Kokoro / Melo / Chatterbox** | **Endpoint** (and voice/speaker where applicable), often localhost for self-hosted stacks. |

### Speech-to-text

| Backend | Minimum required |
|--------|-------------------|
| **Whisper (Rust)** | First run downloads the chosen GGML **`.bin`** from Hugging Face (`ggerganov/whisper.cpp`) into **`app_data/whisper/`** (the same models can also be installed from [Models & Runtimes](#models--runtimes) or with `Sigil.exe --headless fetch whisper/<id>`, pinned to a fixed revision and checksum; files already in `whisper/` are adopted without downloading again). Each download run writes its own **`.partial`** file and **renames** it on success; failed/incomplete/cancelled downloads remove only that run’s partial file, never a completed model. Very small or HTML-looking payloads are rejected. You can also drop a matching `.bin` into that folder manually. Model ids: `tiny`, `base`, `small`, `medium` (each also `.en`), their `-q5_1` / `-q8_0` builds (`medium` uses `-q5_0`), `large-v3-turbo` (plus `-q5_0` / `-q8_0`) and `large-v3` (plus `-q5_0`). The old `medium-q5_1` ids never existed upstream and load `-q5_0`; an unknown id is an error instead of silently loading `base.en`. **Anime-Whisper** is bring-your-own: convert `litagin/anime-whisper` with whisper.cpp's `models/convert-h5-to-ggml.py` and select the `.bin` as a local model (Japanese is forced and no prompt is used). |
| **Vosk** | The catalog is served from the **Rust** plugin (`get_vosk_models`). The app downloads **`{app_data}/vosk-models/{model_id}.zip`** through Sigil's model store (official Alpha Cephei URLs unless you set **Custom model URL**): one download at a time, resumed after a break (Stop keeps the partial file), a 30 s no-data limit, HTML pages rejected, and a full zip check before an existing model is replaced. The default `small-en-us-0.15` is pinned to a checksum; importing a local zip is checked the same way. **vosk-browser** loads the zip from AppData via Tauri **`convertFileSrc`** (asset protocol, scoped to **`$APPDATA/vosk-models/**`** only); repeat use does not re-fetch the CDN. Custom URL must point to a full **`.zip`**; it overrides the catalog URL but the on-disk name still follows the selected **model id**. |
| **Moonshine** | **Moonshine server** reachable at the configured URL (default `http://localhost:8090`). Must expose `GET /health` and `POST /transcribe` (JSON body with base64 WAV), as in Useful Sensors’ published container (`moonshine-onnx-server`). Example: `docker run -p 8090:8090 useful-sensors/moonshine-onnx-server` (or `:tiny` for a smaller model). Set **language** to what the server expects (e.g. `en`). |
| **Local HTTP API (OpenAI-style)** | **Base URL** including `/v1` (e.g. `http://127.0.0.1:8000/v1`). Sigil sends one `POST …/audio/transcriptions` per spoken sentence (multipart `file` as 16 kHz mono WAV, `model`, optional `language`, `prompt`, `temperature=0`, `response_format=json`), matching the [OpenAI transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription) shape. You must set **model id** in the inspector to whatever your **local** server expects (field defaults empty). **API key** is optional for localhost; it is sent only in the `Authorization` header. |
| **Groq Whisper (free cloud)** | A free API key from [console.groq.com/keys](https://console.groq.com/keys). Model `whisper-large-v3-turbo` (default) or `whisper-large-v3`. Speech is sent to Groq. See [Groq preset](#groq-whisper-preset). |
| **Deepgram** | **API key** and **language**. Model: Nova-3 (default) or Nova-2. |
| **Azure** | **Key**, **location**, **language**. Optional **phrase end silence** (100–5000 ms, default 500). |
| **Azure / Deepgram / native** | Keys or browser permissions as prompted in the STT inspector. Azure and Deepgram capture in the webview, so their **device** pickers list **browser (Web Audio)** microphones, not the native cpal devices used by the Rust-captured engines. |

## Remote client

The `/client` page does not run Tauri plugins; it syncs with the **host** over PeerJS. Configure STT/TTS on the **desktop host**, not in the browser client.

## Upstream verification (Parakeet, speaches, sherpa-onnx)

Before relying on a local stack for **NVIDIA Parakeet** or **sherpa-onnx**, confirm on the **exact** project version you deploy:

- **[speaches](https://github.com/speaches-ai/speaches)** (evolution of faster-whisper–style OpenAI-compatible servers): check the current README and Docker tags for which **engines** and **model IDs** are supported, whether **Windows** is first-class or Docker/WSL-only, and whether streaming uses **SSE/WebSocket** vs single **multipart** file upload. Sigil’s **OpenAI-compatible** backend sends **one non-streaming multipart upload per spoken sentence** (cut by the Silero voice detector); servers that only expose streaming may need a different client or proxy.
- **sherpa-onnx**: model artifacts (ONNX, tokens, config) are engine-specific; Parakeet TDT support depends on upstream recipes and export pipelines—not assumed here.
- **Moonshine**: the official **Docker** flow is the supported integration path; custom builds should keep the same HTTP contract as [moonshine-onnx-server](https://github.com/usefulsensors/moonshine).

Re-run a quick **latency and WER** check on your own machine (VRChat + OBS load) after any engine change; published benchmarks do not replace on-device measurement.

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
```

`SIGIL_APP_DATA_DIR=<folder>` points any of these at a scratch profile instead of `%APPDATA%\com.sigil.app`.

### CPU speech runtime pack (sherpa-onnx)

`runtime/sherpa-onnx` (sherpa-onnx 1.13.8, CPU only; 8 MB download, 22 MB installed) is the CPU runtime for the upcoming local speech engines (SenseVoice, Nemotron streaming, Qwen3-ASR 0.6B, Kokoro, Piper). It is optional: Sigil starts and every existing feature works without it. Install it from Models & Runtimes or with `Sigil.exe --headless fetch runtime/sherpa-onnx`; check it with `Sigil.exe --headless engine-probe sherpa [model folder] [wav]`. The error **"another ONNX Runtime is already loaded"** means some other `onnxruntime.dll` (for example Windows ML's copy in System32) was loaded into Sigil before the pack; restart Sigil so the pack loads first. Models that are not from the catalog are test-loaded in a separate process first, so a damaged model folder cannot crash Sigil.

### GPU speech runtime packs (transcribe.cpp)

`runtime/sigil-transcribe-<variant>` (`cuda13`, `vulkan`, `cpu-avx512`, `cpu-avx2`, `cpu`) run GGUF speech models such as Qwen3-ASR 1.7B. They are built locally and installed by **Import…** or `SIGIL_LOCAL_ASSETS` until they are published. The best variant for the PC is picked automatically; `cuda13` needs driver R580 or newer, a GeForce 16/20 series or newer GPU and the NVIDIA GPU runtime pack (only its cuBLAS is used); GPU variants need an AVX2 CPU. Budget about 3.5 GB of video memory for Qwen3-ASR 1.7B Q8_0. The speech-to-text backend that uses them arrives with the P2 speech engine.

### Local LLM runtime packs (llama-server)

`runtime/llama-server-vulkan` (default GPU build), `runtime/llama-server-cuda13` and `runtime/llama-server-cpu` are the official llama.cpp server builds used by the upcoming local translation. Sigil runs them as helper processes that listen only on `127.0.0.1` with a random key and are killed together with Sigil, even when Sigil is ended in Task Manager.

---


## STT reliability and troubleshooting

Operator and developer reference for how Sigil’s STT behaves in **shipping builds**, where failures appear, and which code paths matter. Use this when debugging “connected but no text”, wrong mic, or stuck “connecting”.

### Orchestrator ([`src/core/services/stt/index.ts`](../src/core/services/stt/index.ts))

All backends implement the same contract toward `Service_STT`:

| Callback | Meaning |
|----------|---------|
| `onStart` | Engine is ready; UI shows **connected**. |
| `onStop(error?)` | Session ended; optional error toast (sticky if `error` set). |
| `onInterim` | Non-final recognized text only. |
| `onFinal` | Final segment for pubsub/captions (downstream may treat as a discrete line). |
| `onStatus(text, {progress, level})` | Engine status (download and load progress, retries, rate-limit waits). Shown **under the Start button** only; it is never published, so it never reaches captions, the chatbox or TTS. |
| `isMuted()` | Whether STT is muted; utterance engines skip sending muted speech. |

**Connecting timeouts** (orchestrator clears “connecting” and tears down the engine). These are **inactivity** windows: every status or progress report from the active engine while connecting (`onStatus`, such as download or model-load progress) restarts the timer, so a long model download does not time out while it is still advancing. Whisper reports download, local-model import and model-load progress (`whisper:download_progress`, `whisper:load_status`); Vosk reports its model download (`vosk:download_progress`, also when the server sends no file size). A download that stops advancing still times out after the window below.

| Backend group | Timeout without progress |
|---------------|---------|
| Default (e.g. Azure, Deepgram, native, OpenAI-audio promise path) | 45s |
| Whisper, Vosk, Moonshine | 300s |
| Chrome / Edge external mic | 6s (mic page must signal start) |

**Session epoch:** `#sessionEpoch` ignores stale callbacks after stop or backend switch.

**Auto-start:** `native` is excluded from `autoStart` in `init()` (known Web Speech quirks in the shell).

**Desktop default (one-time switch):** Web Speech (`native`) cannot work in the desktop webview. On the first desktop launch after this change, users still on `native` (or the legacy `browser` value) are switched once to **Whisper**: `small` with 6 or more estimated physical cores, otherwise `base`, as the Q5_1 build (`.en` when the language is English). Existing Whisper choices are kept, nothing starts automatically (one click on **Start** downloads the model), and a toast explains the switch. A later deliberate choice of `native`, or the engine picked during onboarding, is never overridden (`stt.desktopDefaultMigrated`).

**Mute and shortcuts:** Muting stops publishing text (Whisper keeps transcribing in Rust while muted; the HTTP engines do not send muted speech at all). **STT → Shortcuts** sets global **start/stop** and **mute** shortcuts that work while Sigil is in the background (for example in VRChat). Recording finishes on the first full combination, **Esc** cancels, and a bare letter needs a modifier. The sound-effects mute shortcut is under **Settings → Shortcuts**. Stored shortcuts that can never register are cleared once with a toast; a shortcut held by another app is kept with a warning.

**Start chain:** If `#runStartAsync` throws outside the inner `try`, the promise rejection is caught on `#startChain` and surfaces a toast (`toasts.stt_start_failed`).

### Input device resolution ([`src/utils/rustAudioInput.ts`](../src/utils/rustAudioInput.ts))

Backends that use **Rust capture** (Whisper, Vosk, Moonshine, local HTTP STT, Groq) call `resolveRustInputDeviceForInvoke(perServiceDevice)`, which uses the first of these that is set:

1. The STT inspector’s per-engine **input device** (a cpal device name).
2. **Settings → Audio → Default input device** — on the desktop app this is the **native (cpal)** device, persisted as `audioInputDeviceNative`.
3. **`null`** → Rust uses the system default input.

A per-engine value of `default` / `communications` means “system default” and stops there (it does **not** fall through to the Settings device); the same sentinels in the Settings field also mean system default. The Web Audio `deviceId` in `audioInputDevice` is **never** passed to cpal; it is only used by Deepgram (webview capture) when Deepgram’s own device is empty. Azure uses only its own device field. Older builds stored the Settings cpal name in `audioInputDevice`; on first run it is migrated to `audioInputDeviceNative` when it exactly matches a device returned by `plugin:audio|list_input_devices`.

**User impact:** Pick the mic once under **Settings → Audio** and every Rust-captured engine uses it; override per engine in the STT inspector only when one engine needs a different mic.

**Shared microphone and device changes (capture hub):** every Rust-captured engine now shares **one stream per microphone**, so two engines on the same mic open it once. With **System default** selected, STT follows the Windows default microphone when it changes. Unplugging a microphone no longer ends STT: a toast says the device stopped and that Sigil reconnects it, and when it is plugged back in (or a stream that delivered no audio for 2.5 s is restarted) a second toast confirms it is working again, in the same session. A saved device that is not connected stays selected and is shown as **(not connected)**. Muting STT stops audio reaching the engines at the source. Browser-captured engines (Azure, Deepgram, Web Speech) restart by themselves after the default microphone changes. `SIGIL_CAPTURE_SHARING=0` is an internal switch that restores one stream per engine (for A/B soak tests only); stream opens, failures and recoveries are logged under `sigil::audio::hub`.

### Voice detection (Silero VAD)

Whisper and the utterance engines (Moonshine, local HTTP, Groq) are gated by the bundled **Silero VAD v6.2** model, running on the CPU inside Sigil (about 0.3% of one core, including resampling to 16 kHz with rubato). **Silence never reaches the recognizer:** a request or a Whisper decode happens only for audio the detector classified as speech.

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
- **Ordered, bounded backlog:** utterances are transcribed in order by one worker per recording. At most **3** wait (setting "Waiting sentences", 1–10); when the model is slower than real time the oldest waiting one is dropped, logged, and reported as `whisper:utterance_dropped` `{seq, audio_ms, reason}`.
- **No-speech drop:** each utterance's no-speech probability is computed the way OpenAI defines it (P(`<|nospeech|>`) at the start of the transcript). Above **0.6** (setting "No-speech cutoff"; 1 disables) the utterance is dropped. This replaces whisper.cpp's own value, which is stale for multilingual models and prompts (they used to print "you" / "Thank you." on silence).
- **Hallucination filter** (default on): drops curated stock phrases ("Thanks for watching", per-language lists) only when they are the whole sentence, collapses repeat loops (3 or more; CJK uses character n-grams) and strips markers such as `[MUSIC]` or `(laughs)`.
- **Hotwords** and the **context prompt** are passed to Whisper as its initial prompt. **CPU threads** default to about half the logical CPUs (at most 4). Stopping cancels a running decode promptly.
- Other events: `whisper:speech_start` `{seq, start_ms}` and `whisper:capture_error` `{message}` (voice detection failed and the recording ended). A lost microphone is not an error any more: it is reported as `audio:stream_error` / `audio:stream_recovered` and the recording continues.
- **Silence check:** every decode logs `transcribe utterance seq=… audio_ms=…` at info level under the log target `sigil::stt::whisper` in the [log file](#gpu-acceleration-diagnostics-and-logs). Ten minutes of silence and keyboard noise should add no such line.

#### Vosk ([`vosk.ts`](../src/core/services/stt/services/vosk.ts))

- Models load from **AppData** via Tauri **`convertFileSrc`**. The asset protocol scope in `tauri.conf.json` is limited to **`$APPDATA/vosk-models/**`** — the rest of app data (settings with API keys, other models) and the WebView2 profile under `$APPLOCALDATA` are **not** readable through it. vosk-browser fetches the zip as data from a `blob:` worker, so release builds need CSP `worker-src blob:` and `connect-src` for `asset:` (macOS/Linux) and `https://asset.localhost` (Windows, because the window uses `useHttpsScheme`).
- Tauri event **`vosk:download_progress`** (`{ file, progress }`, same shape and rate as Whisper’s; `-1` when the server sends no size) while the model zip downloads; shown as interim status (“Downloading …”).
- Model WASM load has a **120s** timeout.
- Capture via `plugin:audio|start_audio_capture` from the shared capture hub. With `sampleRate: 16000` the `audio:chunk` payloads are 16 kHz mono (about a sixth of the old 48 kHz stereo traffic); every payload states its own sample rate and channel count, and `vosk.ts` downmixes when needed and passes that rate to the recognizer.

#### Utterance engines: Moonshine, local HTTP, Groq ([`rustCapturePolling.ts`](../src/core/services/stt/rustCapturePolling.ts), [`utterancePipeline.ts`](../src/core/services/stt/utterancePipeline.ts))

These engines use Rust **utterance capture** (`plugin:audio|start_utterance_capture` → `audio:utterance` events → `take_utterance` binary 16 kHz mono PCM), so they send **one request per spoken sentence** and nothing for silence.

- Requests go out **one at a time, in order**. Sentences that pile up while a request is in flight (or rate-limited) are **merged** into the next request, up to 28 s with 0.2 s of silence between them.
- **429** waits for `Retry-After` (or 5 s when the header is unreadable) and does not count as a failure. **Other 4xx** stop the session at once (bad key, bad model). **Network errors, timeouts and 5xx** are retried with backoff 4 times, then the session stops with a clear message.
- The session also stops when 300 s of speech is waiting, or when the server asks to wait more than 10 minutes (for example Groq's daily cap).
- Timeouts: local HTTP **120 s**, cloud (Groq) **60 s** per request; Moonshine's Rust request allows 30 s plus the clip length (at most 120 s). An in-flight request is aborted when STT stops. Settings are read at start, so edits apply on the next start.

#### Moonshine ([`moonshine.ts`](../src/core/services/stt/services/moonshine.ts))

- Server down: **`onStop`** with setup instructions (good failure mode).
- Transcribe failures follow the utterance-engine retry rules above.

#### Groq Whisper preset

Stored as the `openai_audio` backend with preset `groq` (shown as **Groq Whisper (free cloud)**). The key goes only in the `Authorization` header. One shared limiter paces every session (a quick stop/start cannot reset it) inside Groq's free plan: **20 requests per minute** (a burst of 5, then an even pace), **2,000 per day**, and **7,200 audio seconds per hour** (kept under 97%, counting Groq's 10 s minimum billing per request). A custom URL pointing at `api.groq.com` gets the same limits. Rate-limit waits appear as status under the Start button.

#### Local HTTP OpenAI-style API ([`openaiAudio.ts`](../src/core/services/stt/services/openaiAudio.ts))

- Follows the utterance-engine rules above; `[MUSIC]`-style markers are stripped from results.

#### Deepgram ([`deepgram.ts`](../src/core/services/stt/services/deepgram.ts))

- The inspector’s **device** picker lists **browser (Web Audio)** microphones. An empty Deepgram device falls back to the Web Audio default input in `audioInputDevice` (picked under **Settings → Audio** in the web build; the desktop app’s Settings picker sets the native device instead, so there it is normally empty). If the resulting id is empty or `default` → `getUserMedia({ audio: true })`; a stored id is used as an exact `deviceId` only if it still exists, otherwise Deepgram falls back to the default mic and logs a warning (covers cpal device names saved by older builds).
- **`getUserMedia`** rejection (permission, no device) → immediate **`onStop`** with a clear message (avoids waiting for the 45s orchestrator timeout).
- **Stop** before `getUserMedia` resolves: the late stream’s tracks are stopped and the WebSocket is never opened (no hot mic, no billed audio).
- WebSocket **error** → **`onStop`**; **`onclose`** is guarded so **`onStop`** is not duplicated.
- The listen URL sends `model` (Nova-3 by default, or Nova-2), `language`, `punctuate`, `interim_results` and `profanity_filter`. Only a key and a language are required to start.

#### Azure ([`azure.ts`](../src/core/services/stt/services/azure.ts))

- Validates **key, location, language** (device is optional).
- **Empty device** → `AudioConfig.fromDefaultMicrophoneInput()`; non-empty → `fromMicrophoneInput(deviceId)`.
- **Phrase end silence** sets `Speech_SegmentationSilenceTimeoutMs` (100–5000 ms, default 500), so captions finalize quickly after a pause.

#### Native Web Speech ([`native.ts`](../src/core/services/stt/services/native.ts))

- In Tauri, **network** error on Web Speech → message to use Whisper/Vosk/Chrome.
- **`onend` → `start()`** loop keeps recognition alive (Web Speech quirk).
- **800ms** fallback promotes to **connected** if `onstart` never fires.

#### Chrome / Edge external mic ([`externalMic.ts`](../src/core/services/stt/services/externalMic.ts))

- **`onStart`** depends on mic page reporting lifecycle via pubsub (`syncExternalMicLifecycle`).

### Pubsub trust (external STT)

Chrome/Edge **mic.html** sends `text.stt` over the host’s **pubsub WebSocket**. The handshake must pass the [Origin allow-list](#local-web-server-and-lan-pages), so a website open in a browser cannot connect, and the host applies an inbound topic allow-list to everything it receives. There is still **no cryptographic auth**: a non-browser tool on the LAN can forge an Origin header and inject STT-shaped traffic (token auth is planned as LAN security stage 2). **mic.html** always connects to the Sigil that served it (`?host=` is ignored) and reconnects by itself after Sigil restarts. See [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md#network-security-and-threat-model).

### Implementation hardening (recent)

- **Moonshine / local HTTP / Groq:** one request per VAD utterance through an ordered pipeline (replaces the 500 ms polling loop and its `transcribeInFlight` lock).
- **Audio format:** Rust resamples the device audio to **mono 16 kHz** with rubato (band-limited, exact length) before the VAD, so the WAV sent to servers is true 16 kHz mono.
- **Status vs text:** engine status (downloads, retries, waits) goes through `onStatus` and is never published as caption text.
- **Native Web Speech:** one **`onFinal` / `onInterim` per `onresult` event**; **`#shouldRestartAfterEnd`** blocks `onend` → `start()` after explicit `stop()`.
- **UwU filter:** failures in `plugin:uwu|translate` are caught; transcript still flows; **System Logs** records a warning.

### Symptoms → likely causes

| Symptom | Likely cause |
|---------|----------------|
| STT “on” but no text (Whisper) | Voice detection too strict (lower **Speech sensitivity**), no-speech cutoff too low, or the wrong physical mic (check **Settings → Audio → Default input device** and the inspector’s input device). |
| Many short “final” lines | One final per sentence by design; raise **Pause that ends speech** to join sentences. |
| Sentences skipped under load (Whisper) | The model is slower than real time and the backlog dropped the oldest waiting sentence (`whisper:utterance_dropped` in the log). Pick a smaller or quantized model, or more CPU threads. |
| Groq session stopped | Daily cap reached, or a rejected key (4xx). The status line and toast say which. |
| Stuck “connecting” ~45s | Backend never called `onStart` (e.g. Deepgram before `getUserMedia` catch; reduced after fix). |
| Wrong microphone | Rust engines: per-engine **Input device**, else the native **Settings → Audio** default (see [Input device resolution](#input-device-resolution-srcutilsrustaudioinputts)). Azure / Deepgram: their own browser-device picker. |
| "Audio device … stopped" toast | The microphone was unplugged, disabled or stopped delivering audio. Sigil reconnects it by itself; STT stays running. If it never recovers, check the device in Windows sound settings; a saved device shown as "(not connected)" needs replugging or another choice. |

### Related files

- Schema / persisted options: [`src/core/services/stt/schema.ts`](../src/core/services/stt/schema.ts)
- Inspector UI: [`src/core/ui/inspector/inspector_stt.tsx`](../src/core/ui/inspector/inspector_stt.tsx)
- Feature matrix: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit)
- Performance / timeouts: [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md#performance-and-connection-timeout-audit)

---

## Voice changer

Sigil’s voice changer runs in the **`voice-changer`** Tauri plugin (`src-tauri/src/services/voice_changer/`): **cpal** capture (any device sample format) → latency budget → process (built-in phase vocoder, **ONNX** file, **RVC** BYO ONNX slot, or **WebSocket** sidecar) → optional post-DSP (formant tilt, phase vocoder refinement) → playback queue. It is **not** compatible with [w-okada/voice-changer](https://github.com/w-okada/voice-changer)’s Python **Socket.IO / REST** server as a drop-in; Sigil’s sidecar speaks **raw WebSocket binary PCM** as documented in [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md).

### RVC (bring-your-own ONNX)

**RVC** mode loads a user-exported slot (`gen.onnx` / `model.onnx`, optional `.index` such as `added_IVF256_Flat_nprobe_1.index`) plus shared HuBERT/FCPE/RMVPE ONNX — no bundled voice weights. Shared bases resolve from inspector paths or `%AppData%/com.sigil.app/voice-changer/base/` (`hubert_base.onnx` / `contentvec.onnx`, `fcpe.onnx`, `rmvpe.onnx`). While streaming, prefer Streamer or encode with **NVENC**.

- **Voice list:** the **Voice** picker lists the folders under `%AppData%/com.sigil.app/voice-changer/models/`, with **Rescan voices** and **Other folder…** for a folder anywhere else. Picking a voice applies its saved per-voice settings.
- **Permission notice:** the first RVC start asks you to confirm you only use voices whose owners agreed. The acknowledgement is stored in the Rust settings (the old browser flag is migrated once).
- **`manifest.json` is optional and never written.** (Sigil does write a small `slot.json` with your per-voice settings; see [Settings persistence](#settings-persistence).) An existing `manifest.json` in the slot always wins. Without one, Sigil infers the settings **in memory** from the generator ONNX: w-okada-style export metadata (sample rate, F0, embedding channels/layer, final projection, version), the graph inputs (`pitch` / `pitchf` ⇒ F0 model), and the `phone` input’s last dimension (embedding channels). The slot folder can therefore be read-only (only the per-voice `slot.json` is then not saved). If the sample rate still cannot be determined, Sigil assumes **40 kHz** and shows a warning toast asking you to add a `manifest.json`; other non-fatal load problems (for example an unreadable index) are reported the same way.
- **FAISS index:** standard RVC `.index` files (FAISS `IndexIVFFlat`, plus flat L2/IP indexes) load, so **Index rate** takes effect. Retrieval follows RVC (8 nearest neighbours, inverse-squared-distance weights, blended by index rate). An index that exists but cannot be parsed is skipped with a warning.
- **Profiles:**

  | Profile | RVC execution provider | F0 | Index rate | Extra context | VRAM cap |
  |---------|------------------------|----|-----------|---------------|----------|
  | **Streamer** (default) | CPU | FCPE | 0 | 64 ms | — |
  | **Low latency** | CUDA | RMVPE | 0 | 96 ms | 768 MB |
  | **Quality** | CUDA | RMVPE | 0.5 | 192 ms | 1536 MB |

  Extra context is re-encoded by HuBERT every tick, so keep it small on CPU. Use **dio** F0 if FCPE is missing. The profile only fills these fields; values you change afterwards (for example Index rate 0 under Quality) are used as set. **RMVPE is GPU-only:** on the CPU provider the inspector marks it "(GPU only)" and FCPE runs instead, with a note. The old **crepe** option always ran RMVPE; it is no longer offered and saved `crepe` settings load as `rmvpe`.
- **RVC execution provider** (`rvc_execution_provider`: CPU / CUDA / DirectML) is separate from the ONNX-mode provider (`onnx_execution_provider`), so switching RVC to CUDA does not change the ONNX chunk mode and vice versa.
- **GPU device index** selects the adapter for the RVC **CUDA** or **DirectML** provider (and the pitch extractor sessions). It has no effect on CPU.
- **VRAM cap** applies to the **CUDA** provider only (ONNX Runtime CUDA memory limit); DirectML and CPU ignore it, and the inspector disables the field unless the RVC provider is CUDA. The limit is set **per ONNX session**: HuBERT, the generator and the pitch extractor (FCPE / RMVPE; dio has none) each get the full cap, so total VRAM can reach about three times the cap, plus CUDA’s own context.
- **Pitch** is applied to the model’s F0 (no phase vocoder on RVC output). **Brightness (formant tilt)** is applied to the RVC output like in the other modes.
- **Crossfade** and **extra context** are read when the conversion pipeline is built, so changing them restarts the engine while running. The crossfade never spans more than one processing chunk (the pipeline clamps it), so the inspector caps it at the chunk length (at 48 kHz, e.g. 10 ms for 512 samples); a stored longer value takes effect again with a larger chunk.

### Feature parity (conceptual vs VCClient)

| VCClient-style area | VCClient (typical) | Sigil |
|--------------------|--------------------|--------|
| **Chunk / batch size** | Client `inputChunkNum` × fixed block size | **`chunk_frames`**: allowlisted sizes from **160** up to **48000** (~**1 s** at 48 kHz) frames per channel, with **coarser steps** above 4096 for very slow hosts (see `ALLOWED_CHUNK_FRAMES` / `VOICE_CHANGER_CHUNK_FRAMES`). Changing value **restarts** the engine. **WebSocket / ONNX** should match server/model shape when possible; Sigil adapts when it can. |
| **Pitch** | Server + client | **±24** semitones. Phase vocoder on the mic in Built-in, or on converter/model **output** after WebSocket / ONNX returns audio (same post-DSP chain as brightness / vocoder refinement). In **RVC** mode the shift is applied to the model’s F0 instead. |
| **Formant / brightness** | Model-dependent | **Formant tilt** [−1, 1] on processed audio in every mode, including the **RVC** output. |
| **Input / output gain** | Client gain nodes | **`input_gain` / `output_gain`** (0–4) in Rust on the capture/playback path. |
| **Output buffer** | Prefill + max queue (worklet) | **`output_prefill_ms`**, **`output_max_queue_ms`** (converted to samples using **output device rate**). Independent of chunk size: they only govern the **playback deque** after processing. **Prefill** = minimum queued audio before cpal starts pulling to speakers (reduces underruns). **Max queue** = cap on queued audio; excess drops oldest samples. The cap is raised automatically to at least the ring target (the larger of the prefill and one output device callback, plus one processed chunk, one capture callback and ~1.3 ms), since a smaller cap would drop audio on every tick. Presets **low_latency** / **stable** fill templates; **custom** when edited. |
| **Silence truncation** | `volTrancate*` thresholds | Optional **RMS + consecutive silent chunks** trims playback deque toward safe minimum. |
| **Metrics UI** | buf / res | IPC **`get_voice_changer_metrics`**: **buffer_ms** (output queue), **process_ms** (EMA) and **process_p99_ms** (p99 over the last 512 blocks), **chunk_ms** (block length), **backlog_ms** (captured audio waiting), **dropped_ms** (audio skipped by the latency budget), **underruns** (playback gaps after start) and **ep** (the execution provider actually in use). The status panel warns when processing is slower than real time, p99 spikes, the backlog nears the budget, or playback runs dry; while devices and the model load it shows “Starting…”. |
| **Monitor output** | Separate monitor device + gain | **Yes** — second **cpal** stream: **`monitor_output_device`** (empty = off, or device name / `"default"`), **`monitor_gain`** (0–4) on **pre-main-gain** PCM; channel layout and sample rate are **adapted** (channel mix plus a stateful band-limited resampler, so there are no periodic clicks) when the monitor device differs from the main output. |
| **Noise / AEC** | Browser constraints, VoiceFocus, etc. | **Not built in**; use **OBS / OS / mic vendor** filters or a virtual cable stack. RNNoise-class DSP remains a possible future plugin. |
| **Wire protocol** | Socket.IO + REST | **WebSocket binary** sidecar only; see sidecar script for framing expectations. |

### Chunk size vs playback buffer (ms)

- **`chunk_frames`** controls the **DSP / inference step** (how much audio is consumed per processing callback). Larger chunks tend to **lower CPU/GPU utilization per real-time second** (fewer invocations) but add **algorithmic/step latency**. They do **not** directly set the values of `output_prefill_ms` / `output_max_queue_ms`.
- **`output_prefill_ms`** and **`output_max_queue_ms`** control the **output ring buffer** in `OutputPlayback`: prefill is the startup cushion before the first non-silent output block; max queue bounds worst-case listening delay when production outpaces consumption. Both are translated to sample counts with the **output sample rate** (`voice_changer/playback.rs`). The lock-free playback ring is kept filled to at least the larger of the prefill and the output device’s **callback period**, plus the last processed chunk and one **capture** callback (chunks are only produced when the input device delivers, so the ring must bridge one capture period), so devices with long periods (e.g. 10 ms WASAPI callbacks) do not underrun every period. Playback starts once the queue holds the prefill or one output callback, whichever is larger; after an underrun the callback ramps back from the held sample to live audio instead of jumping.

### Latency budget

**Latency budget** (default **300 ms**, range 100–2000) is the most captured audio that may wait for conversion. When the engine falls behind (a GPU spike, a busy game), audio older than the budget is **skipped instead of queued**: the oldest whole blocks are dropped down to half the budget with a 5 ms crossfade splice, so the voice never drifts further and further behind. It applies right away, in every mode (built-in, ONNX, WebSocket sidecar, RVC), and it is also how long Sigil waits for a WebSocket converter's reply: a late reply is discarded and that block is muted (a failed conversion in any mode is muted, never the dry microphone). Watch **backlog_ms**, **dropped_ms**, **p99** and **underruns** in the status panel.

### Worker priority and MMCSS

**Voice changer priority** (applies right away; Sigil never asks for administrator rights):

| Setting | Worker thread |
|---------|---------------|
| **Automatic** (default) | NORMAL priority when the model runs on a GPU EP, BELOW_NORMAL on the CPU, so a CPU voice does not cost game frames. |
| **Protect game FPS** | Always BELOW_NORMAL, and ONNX Runtime uses at most 2 threads. |
| **Protect voice** | NORMAL plus the Windows power-throttling (EcoQoS) opt-out. |

ONNX Runtime's own pool threads start with the same policy as the worker, so its CPU work never runs at a higher priority than the setting allows. For ONNX Runtime threads, a change applies from the next voice-changer start.

The audio callback threads (capture, output, monitor) register with MMCSS **Pro Audio**; the inference worker never does. The worker sleeps until the capture callback wakes it (no polling).

### TTS through the voice changer

TTS clips sent through the voice changer are queued **whole and never truncated**: the queue accepts clips while less than 60 s is waiting, and a single clip can be up to 10 minutes. **Microphone while TTS plays**: **Mix with TTS** (default), **Lower the microphone** (about −12 dB) or **Mute the microphone**, with a 250 ms hold between sentences.

### Settings persistence

Rust is authoritative. Every voice-changer setting, including the **input, output and monitor devices**, is saved to `%AppData%/com.sigil.app/voice-changer/settings.json` (atomic writes, a `.bak` of the last good file; a damaged file is kept as `.corrupt-<time>` and the backup is used). Each voice's **pitch, brightness, index rate, protect, F0 method and speaker id** are saved in `<voice folder>/slot.json` and applied when that voice is selected in RVC mode. The voice changer never starts by itself.

### ONNX execution provider (GPU acceleration)

**ONNX** mode and **RVC** mode each have their own provider setting (`onnx_execution_provider` and `rvc_execution_provider`); for RVC, **GPU device index** picks the CUDA / DirectML adapter and **VRAM cap** limits CUDA memory only (per ONNX session: HuBERT, generator and pitch extractor each get the cap).

| Provider | When to use | Requirements |
|----------|-------------|--------------|
| **CPU** | Default; maximum compatibility | Always available |
| **CUDA** | NVIDIA GPU acceleration | GeForce 16/20 series (Turing) or newer, driver **R580+** (CUDA 13), and the **NVIDIA GPU runtime pack** (Models & Runtimes, or Settings → GPU & diagnostics). Without the pack Sigil falls back automatically. |
| **DirectML** | Windows GPU without CUDA | AMD/Intel/NVIDIA via DirectML; the installer bundles `DirectML.dll` |

A GPU request never falls back silently: sessions try **CUDA → DirectML → CPU** (a DirectML request tries DirectML → CPU), each GPU step is checked with a tiny test model before the real one loads, and the provider actually in use plus the reason for any fallback appear in the voice-changer status panel, a toast, the System log and **Settings → GPU & diagnostics**. See [GPU acceleration, diagnostics and logs](#gpu-acceleration-diagnostics-and-logs).

Neural ONNX TTS (Kitten, local speech package) always runs on the **CPU**: the KittenTTS code (now part of Sigil) builds its session on the CPU provider, so the TTS inspector’s execution provider choice is only stored. The real outcome is recorded under **Local TTS** in the diagnostics panel. Loading and synthesis run on a background thread, so the UI stays responsive. Hub models are downloaded one file at a time into `%APPDATA%\com.sigil.app\models\kitten\` (an existing Hugging Face cache copy is reused).

### ONNX chunk shape

Models should use **float32** input/output. Sigil reads the graph’s fixed element count when present and **adapts** mic chunks when sizes differ (mono/stereo and length). Prefer a model whose `N` matches **`chunk_frames × channels`** for minimal extra buffering; otherwise adaptation still applies when supported.

### Related

- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit) — where the voice changer sits in the host matrix.
- [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md) — WebSocket echo server and identity ONNX helper.

---

## GPU acceleration, diagnostics and logs

**Settings → GPU & diagnostics** shows the execution providers in use per component (voice changer model, content encoder, pitch model, ONNX voice model, local TTS, speech detection), the graphics adapters, a **Test GPU acceleration** button, the NVIDIA runtime pack and **Open log folder**. The **Models & Runtimes** tab's **GPU and runtimes** card shows which acceleration options this PC supports and why not.

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

- **Stop** always ends the service: queued and in-flight speech is dropped, a start that is still connecting is cancelled (status bar card or the inspector's **Connecting** button), and late callbacks from a replaced backend are ignored. A clip already handed to Rust (Azure, VOICEVOX, Uberduck) or speech SAPI is already saying finishes.
- A single failed line goes to the **System log** and a rate-limited toast; the service keeps running.
- **Kokoro, Melo, Chatterbox, Fish, ONNX (local) and HTTP server** play clips in order without overlapping. Their voice lists show an error toast instead of an empty list when the server is down. These Web Audio backends still ignore the output device field.
- **VOICEVOX / AivisSpeech:** choose the **Engine preset**. Choosing AivisSpeech resets **pitch to 0** (its voices degrade otherwise) and a **Reset pitch** button is available; the pitch slider steps by 0.01, and 0 intonation/volume values are kept. Speaker ids differ per engine; the first style is picked when the engine changes. Lines are synthesized one at a time, in order. A VOICEVOX-compatible engine on **another PC** must use `https` (the app only allows plain `http` to this PC).
- **Azure:** text with `&` or `<` is spoken (the SSML is escaped), **style** and **role** take effect (`mstts:express-as`), `xml:lang` follows the voice, and prosody left at default is omitted. A bad key stops the service; other failures only skip that line.
- **Uberduck:** v1 API with a Bearer key; lines are spoken one after another; text is limited to 10,000 characters.
- **Windows SAPI:** rates below 1.0 work (0.1 is the slowest, SAPI −10); rates from 1 to 5 map as before.
- **Timeouts** (Rust HTTP backends): health checks 10 s, voice and language lists 20 s, TTS 30 s plus 100 ms per character (at most 120 s), plus a 120 s no-data limit on every connection, model downloads included. Error messages include the server's own message (JSON error field or HTML page title), with keys and credentials removed from URLs.

---

## Translation and AI transform

- **Finals are always delivered, in order**; interims are live and latest-wins (at most one interim request in flight per sentence). Speech and typed text are sequenced separately. Each request has a deadline (15 s for interims, 30 s for finals); a fast, retryable failure of a final is retried once.
- Failed lines show a throttled notice and are written to the **System log**. The `[...]` cut-off marker STT publishes on stop is never translated.
- **Azure Translator:** the region field is optional (needed only for regional or multi-service resources); a blank source language auto-detects; the request timeout is 20 s. 401/403 are reported as key problems and not retried.
- **LibreTranslate:** the endpoint and key are passed with every call (the key in the request body, never in a URL); translation requests time out after 20 s plus 10 ms per character (at most 60 s).
- **AI transform:** **Max output tokens** (default 1024) and **Reasoning effort** (default "Model default", which sends nothing). Thinking models need a larger token limit; for Gemini 2.5 Flash through OpenRouter use reasoning **None**. Settings apply to the next sentence without a restart. Each sentence has a **15 s** deadline; on failure or timeout the original text is sent on with a notice, so captions and TTS keep flowing. **Stop with stream** works, and "Sync timing" captions follow STT when AI is off.

---

## Local web server and LAN pages

The embedded web server (default port **3030**) serves `/client`, the OBS pages, `mic.html`, `inputfield.html` and the pubsub / PeerJS WebSockets on all IPv4 interfaces.

- **Origin allow-list (LAN security stage 1):** `/pubsub` and `/peer` accept a WebSocket only from Sigil's own webviews (`tauri.localhost`), from pages this server served on this machine (`http://localhost:<port>`, `http://127.0.0.1:<port>`, this PC's LAN IPs, `COMPUTERNAME` / `COMPUTERNAME.local`), and origin-less clients on this machine. Everything else (other websites, `file://` pages, browser extensions, DNS-rebinding names) gets **403**, a log line and a "Blocked a connection" toast.
- **Kill switch:** **Settings → Local web server → Only allow Sigil pages to connect** (on by default). The environment variable `SIGIL_LAN_ORIGIN_CHECK=0/1` overrides the setting.
- **OBS** must load `http://localhost:<port>/client` or `/obs-captions.html`, not a local HTML file (a `file://` page is refused).
- Binary WebSocket frames are refused (the old binary-audio-to-Whisper path is gone), messages are capped at 256 KiB, and the server no longer relays pubsub between peers: peer text goes to the host, which applies its inbound topic allow-list and re-broadcasts.
- A failed bind (port taken) is reported with a toast and in **Settings → Local web server** instead of silently disabling the server.

---

## Local speech package (ONNX)

When you use **ONNX (local)** text-to-speech and choose **Folder on this PC**, Sigil loads a **speech package**: a directory that contains everything the in-process engine needs, not a single arbitrary `.onnx` file.

### Format: `speech_package_v1` (current)

The supported layout matches what the app downloads from a Hugging Face **model repository** for this backend: a small JSON manifest, one ONNX graph, and a voice data archive.

| File | Role |
|------|------|
| `config.json` | Declares model type (`ONNX1` / `ONNX2`), filenames for the ONNX model and voice data, optional speed priors and voice aliases. |
| ONNX file | Name given by `model_file` in `config.json`. |
| Voices archive | Name given by `voices` in `config.json` (NPZ with per-voice style data). |

Hub download and a local folder use the **same** layout, so you can copy an extracted repo or point at a cache directory that contains these three pieces.

### Hub download

With **Model from → Hugging Face repo**, the first run may download and cache that same package automatically.

### Other formats

Support for additional on-disk layouts (for example lighter single-file or two-file voice kits) may be added later. Unsupported formats will be rejected with a clear error.

---

## Sidecar scripts (repo)

Example HTTP/WebSocket servers for local development and testing (not shipped with the app):

- **[`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md)** — WebSocket voice-changer echo, identity ONNX helper, optional TTS HTTP sidecar defaults (`POST /speech`, `GET /health`).
- **[`scripts/translation-sidecar/README.md`](../scripts/translation-sidecar/README.md)** — Docker-oriented LibreTranslate-compatible sidecar for the `translate` plugin.

Full request/response shapes and ports live in those READMEs. The [connectivity audit](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit) describes how Sigil wires Tauri plugins to these patterns. Neural TTS HTTP, voice-changer WebSocket binary PCM, and LibreTranslate-style translation HTTP contracts are maintained in the script folders above; use them for copy-paste Docker commands and API shapes—this document focuses on inspector fields and backend behavior in the app.
