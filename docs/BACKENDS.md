# STT / TTS backend minimum configuration

Use this when testing or supporting users. Engines are selected in the Sigil host app (inspectors).

## Text-to-speech

| Backend | Minimum required |
|--------|-------------------|
| **Native (browser)** | Select a **voice** in Text to Speech settings. |
| **Windows** | **Output device** and **voice** (native Windows SAPI). |
| **Azure** | Speech **key**, **region/location**, **voice**, optional device. |
| **Uberduck** | **API key**, **secret**, **voice**. |
| **VOICEVOX / Kokoro / Melo / Chatterbox / Fish Speech** | **Endpoint** (and voice/speaker where applicable), often localhost for self-hosted stacks. |

## Speech-to-text

| Backend | Minimum required |
|--------|-------------------|
| **Whisper (Rust)** | First run downloads the chosen GGML **`.bin`** from Hugging Face (`ggerganov/whisper.cpp`) into **`app_data/whisper/`**. Downloads use a **`.partial`** file and **rename** on success; failed/incomplete downloads are removed. Very small or HTML-looking payloads are rejected. You can also drop a matching `.bin` into that folder manually. |
| **Vosk** | The catalog is served from the **Rust** plugin (`get_vosk_models`). The app downloads **`{app_data}/vosk-models/{model_id}.zip`** via the Rust HTTP client (official Alpha Cephei URLs unless you set **Custom model URL**). **vosk-browser** loads the zip from disk through the Tauri FS API (`readFile` + Blob URL), so repeat use does not re-fetch the CDN. Custom URL must point to a full **`.zip`**; it overrides the catalog URL but the on-disk name still follows the selected **model id**. |
| **Moonshine** | **Moonshine server** reachable at the configured URL (default `http://localhost:8090`). Must expose `GET /health` and `POST /transcribe` (JSON body with base64 WAV), as in Useful Sensors’ published container (`moonshine-onnx-server`). Example: `docker run -p 8090:8090 useful-sensors/moonshine-onnx-server` (or `:tiny` for a smaller model). Set **language** to what the server expects (e.g. `en`). |
| **Local HTTP API (OpenAI-style)** | **Base URL** including `/v1` (e.g. `http://127.0.0.1:8000/v1`). Sigil sends `POST …/audio/transcriptions` with multipart `file` (WAV), `model`, and optional `language`, matching the [OpenAI transcription API](https://platform.openai.com/docs/api-reference/audio/createTranscription) shape. You must set **model id** in the inspector to whatever your **local** server expects (field defaults empty). **API key** is optional for localhost. Pointing the URL at a cloud host is supported but optional; the feature is intended for local sidecars. |
| **Azure / Deepgram / native** | Keys or browser permissions as prompted in the STT inspector. |

### Upstream verification (Parakeet, speaches, sherpa-onnx)

Before relying on a local stack for **NVIDIA Parakeet** or **sherpa-onnx**, confirm on the **exact** project version you deploy:

- **[speaches](https://github.com/speaches-ai/speaches)** (evolution of faster-whisper–style OpenAI-compatible servers): check the current README and Docker tags for which **engines** and **model IDs** are supported, whether **Windows** is first-class or Docker/WSL-only, and whether streaming uses **SSE/WebSocket** vs single **multipart** file upload. Sigil’s **OpenAI-compatible** backend uses **non-streaming** multipart uploads in ~500 ms chunks; servers that only expose streaming may need a different client or proxy.
- **sherpa-onnx**: model artifacts (ONNX, tokens, config) are engine-specific; Parakeet TDT support depends on upstream recipes and export pipelines—not assumed here.
- **Moonshine**: the official **Docker** flow is the supported integration path; custom builds should keep the same HTTP contract as [moonshine-onnx-server](https://github.com/usefulsensors/moonshine).

Re-run a quick **latency and WER** check on your own machine (VRChat + OBS load) after any engine change; published benchmarks do not replace on-device measurement.

### In-process sherpa-onnx (optional / future)

Bundling **sherpa-onnx** inside the Sigil Tauri binary via Rust **FFI** could reduce HTTP overhead and ease single-artifact distribution, at the cost of **larger binaries**, **native build complexity** (per OS/CPU), **crash blast radius** vs an isolated sidecar, and **model update** shipping. Prefer the **sidecar** pattern (Moonshine Docker or OpenAI-compatible server) until a concrete product requirement (e.g. offline boxed builds) justifies FFI maintenance.

## Remote client

The `/client` page does not run Tauri plugins; it syncs with the **host** over PeerJS. Configure STT/TTS on the **desktop host**, not in the browser client.
