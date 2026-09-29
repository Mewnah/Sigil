# Operator reference

Detailed reference for **STT/TTS backends**, **reliability**, **voice changer** behavior, and **local ONNX speech packages**. User-facing walkthroughs live in the [user manual](manual/README.md).

## Table of contents

- [STT / TTS backend minimum configuration](#stt--tts-backend-minimum-configuration)
- [Remote client](#remote-client)
- [Upstream verification (Parakeet, speaches, sherpa-onnx)](#upstream-verification-parakeet-speaches-sherpa-onnx)
- [In-process sherpa-onnx (optional / future)](#in-process-sherpa-onnx-optional--future)
- [STT reliability and troubleshooting](#stt-reliability-and-troubleshooting)
- [Voice changer](#voice-changer)
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
| **Azure** | Speech **key**, **region/location**, **voice**, optional device. |
| **Uberduck** | **API key**, **secret**, **voice**. |
| **VOICEVOX / Kokoro / Melo / Chatterbox / Fish Speech** | **Endpoint** (and voice/speaker where applicable), often localhost for self-hosted stacks. |

### Speech-to-text

| Backend | Minimum required |
|--------|-------------------|
| **Whisper (Rust)** | First run downloads the chosen GGML **`.bin`** from Hugging Face (`ggerganov/whisper.cpp`) into **`app_data/whisper/`**. Each download run writes its own **`.partial`** file and **renames** it on success; failed/incomplete/cancelled downloads remove only that run’s partial file, never a completed model. Very small or HTML-looking payloads are rejected. You can also drop a matching `.bin` into that folder manually. |
| **Vosk** | The catalog is served from the **Rust** plugin (`get_vosk_models`). The app downloads **`{app_data}/vosk-models/{model_id}.zip`** via the Rust HTTP client (official Alpha Cephei URLs unless you set **Custom model URL**). **vosk-browser** loads the zip from AppData via Tauri **`convertFileSrc`** (asset protocol, scoped to **`$APPDATA/vosk-models/**`** only); repeat use does not re-fetch the CDN. Custom URL must point to a full **`.zip`**; it overrides the catalog URL but the on-disk name still follows the selected **model id**. |
| **Moonshine** | **Moonshine server** reachable at the configured URL (default `http://localhost:8090`). Must expose `GET /health` and `POST /transcribe` (JSON body with base64 WAV), as in Useful Sensors’ published container (`moonshine-onnx-server`). Example: `docker run -p 8090:8090 useful-sensors/moonshine-onnx-server` (or `:tiny` for a smaller model). Set **language** to what the server expects (e.g. `en`). |
| **Local HTTP API (OpenAI-style)** | **Base URL** including `/v1` (e.g. `http://127.0.0.1:8000/v1`). Sigil sends `POST …/audio/transcriptions` with multipart `file` (WAV), `model`, and optional `language`, matching the [OpenAI transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription) shape. You must set **model id** in the inspector to whatever your **local** server expects (field defaults empty). **API key** is optional for localhost. Pointing the URL at a cloud host is supported but optional; the feature is intended for local sidecars. |
| **Azure / Deepgram / native** | Keys or browser permissions as prompted in the STT inspector. Azure and Deepgram capture in the webview, so their **device** pickers list **browser (Web Audio)** microphones, not the native cpal devices used by the Rust-captured engines. |

## Remote client

The `/client` page does not run Tauri plugins; it syncs with the **host** over PeerJS. Configure STT/TTS on the **desktop host**, not in the browser client.

## Upstream verification (Parakeet, speaches, sherpa-onnx)

Before relying on a local stack for **NVIDIA Parakeet** or **sherpa-onnx**, confirm on the **exact** project version you deploy:

- **[speaches](https://github.com/speaches-ai/speaches)** (evolution of faster-whisper–style OpenAI-compatible servers): check the current README and Docker tags for which **engines** and **model IDs** are supported, whether **Windows** is first-class or Docker/WSL-only, and whether streaming uses **SSE/WebSocket** vs single **multipart** file upload. Sigil’s **OpenAI-compatible** backend uses **non-streaming** multipart uploads in ~500 ms chunks; servers that only expose streaming may need a different client or proxy.
- **sherpa-onnx**: model artifacts (ONNX, tokens, config) are engine-specific; Parakeet TDT support depends on upstream recipes and export pipelines—not assumed here.
- **Moonshine**: the official **Docker** flow is the supported integration path; custom builds should keep the same HTTP contract as [moonshine-onnx-server](https://github.com/usefulsensors/moonshine).

Re-run a quick **latency and WER** check on your own machine (VRChat + OBS load) after any engine change; published benchmarks do not replace on-device measurement.

## In-process sherpa-onnx (optional / future)

Bundling **sherpa-onnx** inside the Sigil Tauri binary via Rust **FFI** could reduce HTTP overhead and ease single-artifact distribution, at the cost of **larger binaries**, **native build complexity** (per OS/CPU), **crash blast radius** vs an isolated sidecar, and **model update** shipping. Prefer the **sidecar** pattern (Moonshine Docker or OpenAI-compatible server) until a concrete product requirement (e.g. offline boxed builds) justifies FFI maintenance.

---

## STT reliability and troubleshooting

Operator and developer reference for how Sigil’s STT behaves in **shipping builds**, where failures appear, and which code paths matter. Use this when debugging “connected but no text”, wrong mic, or stuck “connecting”.

### Orchestrator ([`src/core/services/stt/index.ts`](../src/core/services/stt/index.ts))

All backends implement the same contract toward `Service_STT`:

| Callback | Meaning |
|----------|---------|
| `onStart` | Engine is ready; UI shows **connected**. |
| `onStop(error?)` | Session ended; optional error toast (sticky if `error` set). |
| `onInterim` | Non-final text (or transient status strings like `...`). |
| `onFinal` | Final segment for pubsub/captions (downstream may treat as a discrete line). |

**Connecting timeouts** (orchestrator clears “connecting” and tears down the engine). These are **inactivity** windows: every progress report from the active engine while connecting (an `onInterim` status such as download or model-load progress) restarts the timer, so a long model download does not time out while it is still advancing. Whisper reports download, local-model import and model-load progress (`whisper:download_progress`, `whisper:load_status`); Vosk reports its model download (`vosk:download_progress`, also when the server sends no file size). A download that stops advancing still times out after the window below.

| Backend group | Timeout without progress |
|---------------|---------|
| Default (e.g. Azure, Deepgram, native, OpenAI-audio promise path) | 45s |
| Whisper, Vosk, Moonshine | 300s |
| Chrome / Edge external mic | 6s (mic page must signal start) |

**Session epoch:** `#sessionEpoch` ignores stale callbacks after stop or backend switch.

**Auto-start:** `native` is excluded from `autoStart` in `init()` (known Web Speech quirks in the shell).

**Start chain:** If `#runStartAsync` throws outside the inner `try`, the promise rejection is caught on `#startChain` and surfaces a toast (`toasts.stt_start_failed`).

### Input device resolution ([`src/utils/rustAudioInput.ts`](../src/utils/rustAudioInput.ts))

Backends that use **Rust cpal** (Whisper, Vosk, Moonshine, local HTTP STT) call `resolveRustInputDeviceForInvoke(perServiceDevice)`, which uses the first of these that is set:

1. The STT inspector’s per-engine **input device** (a cpal device name).
2. **Settings → Audio → Default input device** — on the desktop app this is the **native (cpal)** device, persisted as `audioInputDeviceNative`.
3. **`null`** → Rust uses the system default input.

A per-engine value of `default` / `communications` means “system default” and stops there (it does **not** fall through to the Settings device); the same sentinels in the Settings field also mean system default. The Web Audio `deviceId` in `audioInputDevice` is **never** passed to cpal; it is only used by Deepgram (webview capture) when Deepgram’s own device is empty. Azure uses only its own device field. Older builds stored the Settings cpal name in `audioInputDevice`; on first run it is migrated to `audioInputDeviceNative` when it exactly matches a device returned by `plugin:audio|list_input_devices`.

**User impact:** Pick the mic once under **Settings → Audio** and every Rust-captured engine uses it; override per engine in the STT inspector only when one engine needs a different mic.

### Per-backend notes

#### Whisper (Rust + TS [`whisper.ts`](../src/core/services/stt/services/whisper.ts))

- Tauri event **`whisper:download_progress`** (`{ file, progress }`, percent, at most ~5 per second): interim text while a model downloads or a local model file is copied in.
- Tauri event **`whisper:load_status`**: interim text while a local model file is copied in (“Copying local Whisper model...”) and while the GGML model loads into memory (offloaded to a blocking thread).
- Tauri event **`whisper:segment_transcript`** (formerly `whisper:partial_result`): one payload per **completed VAD/time segment**, not a streaming partial inside one utterance.
- The TS layer maps each payload to **`onFinal`**. Expect **multiple finals** per session (one per silence window), not one long partial stream.
- Strict VAD / `minChunkDurationMs` / `silenceDurationMs` can yield **connected** with **no text** for long periods (different from the “connecting” timeout).

#### Vosk ([`vosk.ts`](../src/core/services/stt/services/vosk.ts))

- Models load from **AppData** via Tauri **`convertFileSrc`**. The asset protocol scope in `tauri.conf.json` is limited to **`$APPDATA/vosk-models/**`** — the rest of app data (settings with API keys, other models) and the WebView2 profile under `$APPLOCALDATA` are **not** readable through it. vosk-browser fetches the zip as data from a `blob:` worker, so release builds need CSP `worker-src blob:` and `connect-src` for `asset:` (macOS/Linux) and `https://asset.localhost` (Windows, because the window uses `useHttpsScheme`).
- Tauri event **`vosk:download_progress`** (`{ file, progress }`, same shape and rate as Whisper’s; `-1` when the server sends no size) while the model zip downloads; shown as interim status (“Downloading …”).
- Model WASM load has a **120s** timeout.
- Capture via `plugin:audio|start_audio_capture`. `audio:chunk` payloads carry the **device’s** sample rate and channel count (the requested 16 kHz is not applied in Rust); `vosk.ts` downmixes to mono and passes the device rate to the recognizer.

#### Moonshine ([`moonshine.ts`](../src/core/services/stt/services/moonshine.ts))

- Server down: **`onStop`** with setup instructions (good failure mode).
- Transcribe failures: first cycles show **`onInterim`**; after **3** consecutive errors the pipeline **stops** with **`onStop`** and a summary message.

#### Local HTTP OpenAI-style API ([`openaiAudio.ts`](../src/core/services/stt/services/openaiAudio.ts))

- Requests go through `fetchWithTimeout` with a **120s** timeout; an in-flight request is **aborted** when STT stops or the engine is torn down.
- HTTP / JSON errors count toward **3** consecutive failures, then **`onStop`** (same pattern as Moonshine).

#### Deepgram ([`deepgram.ts`](../src/core/services/stt/services/deepgram.ts))

- The inspector’s **device** picker lists **browser (Web Audio)** microphones. An empty Deepgram device falls back to the Web Audio default input in `audioInputDevice` (picked under **Settings → Audio** in the web build; the desktop app’s Settings picker sets the native device instead, so there it is normally empty). If the resulting id is empty or `default` → `getUserMedia({ audio: true })`; a stored id is used as an exact `deviceId` only if it still exists, otherwise Deepgram falls back to the default mic and logs a warning (covers cpal device names saved by older builds).
- **`getUserMedia`** rejection (permission, no device) → immediate **`onStop`** with a clear message (avoids waiting for the 45s orchestrator timeout).
- **Stop** before `getUserMedia` resolves: the late stream’s tracks are stopped and the WebSocket is never opened (no hot mic, no billed audio).
- WebSocket **error** → **`onStop`**; **`onclose`** is guarded so **`onStop`** is not duplicated.

#### Azure ([`azure.ts`](../src/core/services/stt/services/azure.ts))

- Validates **key, location, language** (device is optional).
- **Empty device** → `AudioConfig.fromDefaultMicrophoneInput()`; non-empty → `fromMicrophoneInput(deviceId)`.

#### Native Web Speech ([`native.ts`](../src/core/services/stt/services/native.ts))

- In Tauri, **network** error on Web Speech → message to use Whisper/Vosk/Chrome.
- **`onend` → `start()`** loop keeps recognition alive (Web Speech quirk).
- **800ms** fallback promotes to **connected** if `onstart` never fires.

#### Chrome / Edge external mic ([`externalMic.ts`](../src/core/services/stt/services/externalMic.ts))

- **`onStart`** depends on mic page reporting lifecycle via pubsub (`syncExternalMicLifecycle`).

### Pubsub trust (external STT)

Chrome/Edge **mic.html** sends `text.stt` over the host’s **pubsub WebSocket**. Messages are schema-checked, but there is **no cryptographic auth**: any peer that can reach the host port on the LAN could inject STT-shaped traffic. See [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md#network-security-and-threat-model) (`text.stt`, linked `ws://` limitation, PeerJS `secure` flag).

### Implementation hardening (recent)

- **Moonshine / local HTTP STT:** `processAudioBuffer` uses a **`transcribeInFlight`** lock so a slow transcribe cannot overlap the next poll tick.
- **Moonshine / local HTTP STT audio format:** Rust capture arrives at the device’s rate and channel count; [`audioConvert.ts`](../src/core/services/stt/audioConvert.ts) downmixes and resamples it (stateful windowed-sinc low-pass) to **mono 16 kHz** before the WAV is built. Earlier builds sent device-rate audio labelled as 16 kHz mono.
- **Native Web Speech:** one **`onFinal` / `onInterim` per `onresult` event**; **`#shouldRestartAfterEnd`** blocks `onend` → `start()` after explicit `stop()`.
- **UwU filter:** failures in `plugin:uwu|translate` are caught; transcript still flows; **System Logs** records a warning.

### Symptoms → likely causes

| Symptom | Likely cause |
|---------|----------------|
| STT “on” but no text (Whisper) | VAD thresholds; wrong physical mic (check **Settings → Audio → Default input device** and the inspector’s input device). |
| Many short “final” lines | Whisper segment design (`onFinal` per segment). |
| Connected, errors only in the transcript strip | Before fix: Moonshine/OpenAI-audio only used interim for errors; now both **stop after 3** failures. |
| Stuck “connecting” ~45s | Backend never called `onStart` (e.g. Deepgram before `getUserMedia` catch; reduced after fix). |
| Wrong microphone | Rust engines: per-engine **Input device**, else the native **Settings → Audio** default (see [Input device resolution](#input-device-resolution-srcutilsrustaudioinputts)). Azure / Deepgram: their own browser-device picker. |

### Related files

- Schema / persisted options: [`src/core/services/stt/schema.ts`](../src/core/services/stt/schema.ts)
- Inspector UI: [`src/core/ui/inspector/inspector_stt.tsx`](../src/core/ui/inspector/inspector_stt.tsx)
- Feature matrix: [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit)
- Performance / timeouts: [RELEASE_AND_ENGINEERING.md](RELEASE_AND_ENGINEERING.md#performance-and-connection-timeout-audit)

---

## Voice changer

Sigil’s voice changer runs in the **`voice-changer`** Tauri plugin: **cpal** capture → process (built-in phase vocoder, **ONNX** file, **RVC** BYO ONNX slot, or **WebSocket** sidecar) → optional post-DSP (formant tilt, phase vocoder refinement) → playback queue. It is **not** compatible with [w-okada/voice-changer](https://github.com/w-okada/voice-changer)’s Python **Socket.IO / REST** server as a drop-in; Sigil’s sidecar speaks **raw WebSocket binary PCM** as documented in [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md).

### RVC (bring-your-own ONNX)

**RVC** mode loads a user-exported slot (`gen.onnx` / `model.onnx`, optional `.index` such as `added_IVF256_Flat_nprobe_1.index`) plus shared HuBERT/FCPE/RMVPE ONNX — no bundled voice weights. Shared bases resolve from inspector paths or `%AppData%/com.sigil.app/voice-changer/base/` (`hubert_base.onnx` / `contentvec.onnx`, `fcpe.onnx`, `rmvpe.onnx`). While streaming, prefer Streamer or encode with **NVENC**.

- **`manifest.json` is optional and never written.** An existing `manifest.json` in the slot always wins. Without one, Sigil infers the settings **in memory** from the generator ONNX: w-okada-style export metadata (sample rate, F0, embedding channels/layer, final projection, version), the graph inputs (`pitch` / `pitchf` ⇒ F0 model), and the `phone` input’s last dimension (embedding channels). The slot folder can therefore be read-only. If the sample rate still cannot be determined, Sigil assumes **40 kHz** and shows a warning toast asking you to add a `manifest.json`; other non-fatal load problems (for example an unreadable index) are reported the same way.
- **FAISS index:** standard RVC `.index` files (FAISS `IndexIVFFlat`, plus flat L2/IP indexes) load, so **Index rate** takes effect. Retrieval follows RVC (8 nearest neighbours, inverse-squared-distance weights, blended by index rate). An index that exists but cannot be parsed is skipped with a warning.
- **Profiles:**

  | Profile | RVC execution provider | F0 | Index rate | Extra context | VRAM cap |
  |---------|------------------------|----|-----------|---------------|----------|
  | **Streamer** (default) | CPU | FCPE | 0 | 64 ms | — |
  | **Low latency** | CUDA | RMVPE | 0 | 96 ms | 768 MB |
  | **Quality** | CUDA | RMVPE | 0.5 | 192 ms | 1536 MB |

  Extra context is re-encoded by HuBERT every tick, so keep it small on CPU. Use **dio** F0 if FCPE is missing. The profile only fills these fields; values you change afterwards (for example Index rate 0 under Quality) are used as set. **RMVPE / CREPE are GPU-only:** the inspector disables them on the CPU provider, and a stale choice falls back to FCPE with a warning toast.
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
| **Metrics UI** | buf / res | IPC **`get_voice_changer_metrics`**: **buffer_ms** (queued audio), **process_ms** (EMA of per-chunk processing time), **chunk_ms** (current chunk duration). Inspector highlights when **process_ms ≥ chunk_ms** (not realtime). |
| **Monitor output** | Separate monitor device + gain | **Yes** — second **cpal** stream: **`monitor_output_device`** (empty = off, or device name / `"default"`), **`monitor_gain`** (0–4) on **pre-main-gain** PCM; channel layout and sample rate are **adapted** (mix / linear resample) when the monitor device differs from the main output. |
| **Noise / AEC** | Browser constraints, VoiceFocus, etc. | **Not built in**; use **OBS / OS / mic vendor** filters or a virtual cable stack. RNNoise-class DSP remains a possible future plugin. |
| **Wire protocol** | Socket.IO + REST | **WebSocket binary** sidecar only; see sidecar script for framing expectations. |

### Chunk size vs playback buffer (ms)

- **`chunk_frames`** controls the **DSP / inference step** (how much audio is consumed per processing callback). Larger chunks tend to **lower CPU/GPU utilization per real-time second** (fewer invocations) but add **algorithmic/step latency**. They do **not** directly set the values of `output_prefill_ms` / `output_max_queue_ms`.
- **`output_prefill_ms`** and **`output_max_queue_ms`** control the **output ring buffer** in `OutputPlayback`: prefill is the startup cushion before the first non-silent output block; max queue bounds worst-case listening delay when production outpaces consumption. Both are translated to sample counts with the **output sample rate** (`voice_changer.rs`). The lock-free playback ring is kept filled to at least the larger of the prefill and the output device’s **callback period**, plus the last processed chunk and one **capture** callback (chunks are only produced when the input device delivers, so the ring must bridge one capture period), so devices with long periods (e.g. 10 ms WASAPI callbacks) do not underrun every period. Playback starts once the queue holds the prefill or one output callback, whichever is larger; after an underrun the callback ramps back from the held sample to live audio instead of jumping.

### ONNX execution provider (unified release)

Sigil ships **one** Windows installer with **CPU**, **CUDA**, and **DirectML** ONNX Runtime execution providers compiled in (`ort` crate). Pick the provider in the voice changer inspector at **runtime** — no separate CPU/GPU download. **ONNX** mode and **RVC** mode each have their own provider setting (`onnx_execution_provider` and `rvc_execution_provider`); for RVC, **GPU device index** picks the CUDA / DirectML adapter and **VRAM cap** limits CUDA memory only (per ONNX session: HuBERT, generator and pitch extractor each get the cap).

| Provider | When to use | Requirements |
|----------|-------------|--------------|
| **CPU** | Default; maximum compatibility | Always available |
| **CUDA** | NVIDIA GPU acceleration | NVIDIA GPU + current drivers; session creation errors usually mean missing CUDA stack — switch to CPU or DirectML |
| **DirectML** | Windows GPU without CUDA | Windows only; AMD/Intel/NVIDIA via DirectML |

Neural ONNX TTS (local speech package) currently always runs on the **CPU**: the `kittentts` crate builds its own ONNX Runtime session with the default provider, so the TTS inspector’s execution provider choice is only stored (sent with `neural_onnx_tts_load_model`) and does not select a GPU yet.

### ONNX chunk shape

Models should use **float32** input/output. Sigil reads the graph’s fixed element count when present and **adapts** mic chunks when sizes differ (mono/stereo and length). Prefer a model whose `N` matches **`chunk_frames × channels`** for minimal extra buffering; otherwise adaptation still applies when supported.

### Related

- [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md#service-and-feature-connectivity-audit) — where the voice changer sits in the host matrix.
- [`scripts/neural-sidecar/README.md`](../scripts/neural-sidecar/README.md) — WebSocket echo server and identity ONNX helper.

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
