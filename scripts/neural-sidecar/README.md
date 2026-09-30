# Neural sidecar examples (Sigil)

Reference servers and helpers for Sigil’s **voice changer (WebSocket)** and **speech over HTTP**.

## Voice changer — WebSocket (`vc_sidecar.py`)

- **Default URL:** `ws://127.0.0.1:8765`
- **Dependency:** `pip install websockets` (required for this script only)
- **Run:** `python vc_sidecar.py`
- The server **echoes** PCM unchanged. Replace `process_chunk` with your ONNX / GPU VC pipeline.

In Sigil, set voice changer to **Another program over WebSocket** and use `ws://127.0.0.1:8765`.

## Voice changer — ONNX file on this PC

- Sigil loads an ONNX graph with one **float32** input and one **float32** output, typically shaped `[1, N]` or `[N]`. The app reads `N` from the model and **adapts** each mic chunk when `N` does not match the live capture size (mono/stereo and length), so you do not need to hand-match `512 × microphone_channels` unless you want zero extra processing.
- Generate a **pass-through test model:** `pip install onnx` then `python create_identity_vc_onnx.py -o identity_vc.onnx` (defaults: stereo, N=1024). You can still use this script to build a model that matches your device exactly; matching is optional when adaptation applies.
- In Sigil, choose **ONNX model file on this PC** and set the full path to that `.onnx` file.

## TTS — HTTP (`tts_sidecar.py`)

A stand-in speech server for testing Sigil's TTS pipeline without any ML dependencies: it answers
with a tone per word, generated word by word so streaming, stop / skip and routing can be heard.

- **Run:** `python tts_sidecar.py` — **Python stdlib only** (no `pip install` needed).
  Options: `--port 8766`, `--rate 24000` (sample rate), `--chunk-delay 0.05` (simulated generation
  time per word; try `0.2` to hear streaming), `--host 127.0.0.1`.
- **Binds `127.0.0.1` by default** (this PC only). It has no authentication, so binding another
  address (for example `--host 0.0.0.0`) prints a warning; only do that on a network you trust.
- **Sigil "HTTP server" backend** (default URL `http://127.0.0.1:8766`):
  `GET /health`, and `POST /speech` with JSON `{"text":"Hello","voice":"default","stream":true}`.
  Sigil always asks for `"stream": true`: the answer is a chunked WAV whose header declares
  placeholder sizes (`0xFFFFFFFF`), followed by the audio as it is generated, so playback starts
  after the first word. Without `stream` the server sends one complete WAV.
- **Sigil "OpenAI-compatible (HTTP)" backend** (Custom preset, base URL `http://127.0.0.1:8766/v1`):
  `GET /v1/models`, `GET /v1/audio/voices` (`default`, `low`, `high`), and
  `POST /v1/audio/speech` with `{"input","voice","response_format":"pcm"|"wav","speed","stream"}`.
  PCM is raw 16-bit little-endian mono at `--rate` (`Content-Type: audio/pcm;rate=<rate>`), so set
  the backend's PCM sample rate to the same value.
- Replace `synthesize_words` with your model (yield 16-bit PCM pieces as they are ready).
- Rust check against a running server:
  `SIGIL_TEST_OPENAI_SPEECH_URL=http://127.0.0.1:8766/v1 cargo test --manifest-path src-tauri/Cargo.toml openai_speech_live -- --ignored --nocapture`
  (reports the time to the first audio and to the end).

## Notes

- **WebSocket control frames:** The `websockets` library (v11+) responds to **Ping** with **Pong** on sync connections; your handler only sees Text/Binary payloads. If you reimplement the server on raw sockets, you must handle Ping yourself.
- Chunk length is tied to the app **processing chunk** setting (**256 / 512 / 1024** frames per block in the voice-changer plugin). Sample rate follows the live capture device; keep your server’s expectations aligned with what Sigil sends.
- For production VC, run the heavy model in this process (or another host) so Sigil stays lightweight and you can pin a **separate GPU** if needed.
