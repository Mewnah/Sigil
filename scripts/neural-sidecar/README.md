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

- **Default URL:** `http://127.0.0.1:8766` (Sigil default endpoint matches this)
- **Run:** `python tts_sidecar.py` — **Python stdlib only** (no `pip install` needed)
- **POST** `/speech` with JSON `{"text":"Hello","voice":"default"}` returns a short WAV tone for pipeline testing.

In Sigil, choose TTS **Speech from your HTTP server** and set the endpoint if needed.

## Notes

- **WebSocket control frames:** The `websockets` library (v11+) responds to **Ping** with **Pong** on sync connections; your handler only sees Text/Binary payloads. If you reimplement the server on raw sockets, you must handle Ping yourself.
- Chunk length is tied to the app **processing chunk** setting (**256 / 512 / 1024** frames per block in the voice-changer plugin). Sample rate follows the live capture device; keep your server’s expectations aligned with what Sigil sends.
- For production VC, run the heavy model in this process (or another host) so Sigil stays lightweight and you can pin a **separate GPU** if needed.
