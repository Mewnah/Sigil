# Neural sidecar examples (Sigil)

Reference servers and helpers for Sigil’s **voice changer (WebSocket)** and **speech over HTTP**.

## Voice changer — sidecar voices (protocol v2)

Research voice-conversion models that only run in Python become **sidecar voices**: Sigil starts
the script as a helper process (bound to `127.0.0.1`, with a random key, stopped with Sigil) or
connects to a server you run, and plays it inside the voice changer under the latency budget. A
slow or stalled sidecar is muted, never allowed to block the voice changer or build up a backlog.
The wire format is [`PROTOCOL_V2.md`](PROTOCOL_V2.md).

| Script | Recipe | Model | Frame | Notes |
|---|---|---|---|---|
| `xvc_sidecar.py` | `xvc` | [X-VC](https://github.com/Jerrister/X-VC) zero-shot (best speaker similarity) | 120 ms | PyTorch, 16 kHz, about 5 GB; for 12 GB+ GPUs |
| `streamvoiceanon_sidecar.py` | `streamvoiceanon` | [StreamVoiceAnon](https://github.com/Plachtaa/StreamVoiceAnon) (privacy mode) | 92 ms | PyTorch; converts for one voice at a time (a voice switched to takes it over; changing the target speaker needs no restart); `--alpha` below 1 anonymizes more |
| `seedvc_sidecar.py` | `seedvc` | Seed-VC real-time on Sigil's ONNX export (`scripts/vc-families/seed_vc/export.py`) | 180 ms (`--preset sigil_low`/`sigil_lean`: 100 ms) | ONNX Runtime on CUDA, DirectML or CPU; the fallback while Sigil-ZS is unavailable |
| `vc_sidecar.py` | (custom) | echo (protocol v1 and v2) | any | a test server |

Measured on the dev PC (RTX 4090, idle GPU, a 600 ms budget; `sidecar_seedvc_model_live`, the
helper under Sigil's process host and the voice in the real stream harness): the Seed-VC sidecar's
**heard delay** (input speech to converted speech, by envelope cross-correlation) is about
**300 ms** with `--preset sigil_low` and 280–340 ms with `sigil_lean` (CUDA and DirectML alike),
and **400–440 ms** with the default upstream preset, with no muted block. Compute per tick is
36–64 ms per 100 ms frame (`sigil_low`) and 78–127 ms per 180 ms frame (upstream). Upstream's own
real-time GUI reports 430–690 ms. With another program loading the GPU heavily, late frames are
muted at the budget and the delay grows up to the budget, then returns within a pause or two;
the voice changer's loop never waits. X-VC and StreamVoiceAnon need their multi-GB checkpoints and
are not measured yet (the authors report 237 ms end to end for StreamVoiceAnon on an RTX 3060
Laptop).

Every script needs `pip install numpy websockets` (`scipy` and `soundfile` are used when
installed) on top of its model's own requirements, and imports `sigil_vc_v2.py`, which must stay
in the same folder. Several voices may use one helper at a time (a voice's replacement loads while
it plays, idle voices stay loaded): a voice that connects has its reference prepared beside the
playing voice's frames, never in front of them.

**Set up a recipe** (the models are large, so this is a manual step):

1. Install the model as its script's docstring says (repository, requirements, checkpoints) into a
   Python environment of its own.
2. Check it by hand: `python xvc_sidecar.py --model-dir <repository> --self-test reference.wav`
   prints the time per frame and the model's delay as JSON (`--device cuda|dml|cpu`).
3. In Sigil's voice changer, add a sidecar voice: the recipe, the environment's `python.exe`, the
   model folder, and a reference voice from the voice library. Extra arguments (for example
   `--steps 6`) go to the script; `--host`, `--port` and the key are Sigil's.

Sigil finds the scripts in `voice-changer/sidecars` in its data folder, next to the installed app
(`neural-sidecar`), or in the folder `SIGIL_SIDECAR_SCRIPTS` names (developers: the source tree
is used in debug builds). Each helper's output is in Sigil's log
(`%LOCALAPPDATA%\com.sigil.app\logs\sigil.log`).

**Your own model:** subclass `sigil_vc_v2.Converter` / `Stream` and call `sigil_vc_v2.main` (see
[Writing a sidecar](PROTOCOL_V2.md#writing-a-sidecar)), run it
(`python my_sidecar.py --port 8765`), and add a custom sidecar voice with
`ws://127.0.0.1:8765/vc`. `--echo` replaces any script's model with a pass-through for testing the
wiring (`--echo-delay-ms 700` answers each frame later; `--echo-compute-ms 150` simulates an
overloaded model; `--echo-single-stream` behaves like StreamVoiceAnon's single stream).

**Tests.** `python test_sigil_vc_v2.py` checks the shared server (v2 and v1, the key, the
reference cache and its preparation beside other sessions, stale drops, cancel, single-stream
sessions) against a real socket, and the scripts' streams (a `DISCONTINUITY` forgets what came
before it). Rust checks of a recipe helper end to end (the helper under the process host, the
voice in the stream harness):

- no model (`seedvc_sidecar.py --echo`): `SIGIL_TEST_SIDECAR_PYTHON=python cargo test --manifest-path src-tauri/Cargo.toml sidecar_recipe_live -- --ignored --nocapture`;
- two voices on one single-stream helper (`streamvoiceanon_sidecar.py --echo --echo-single-stream`):
  `sidecar_single_stream_live`, same variable;
- the real Seed-VC model: `sidecar_seedvc_model_live` (its doc comment lists the variables: the
  Python with onnxruntime-gpu or -directml, the export folder, a speech WAV, the provider, the
  CUDA runtime folder and the preset); it prints the heard delay, muted blocks and the playout
  summary.

## Voice changer — WebSocket, protocol v1 (`vc_sidecar.py`)

- **Default URL:** `ws://127.0.0.1:8765`
- **Dependency:** `pip install websockets` (plus `numpy` for protocol v2)
- **Run:** `python vc_sidecar.py`
- The server **echoes** PCM unchanged: every v1 packet comes back byte for byte, at any capture
  rate and with every channel, without added delay (it serves protocol v2 too). To convert, write
  a converter as above.

In Sigil, set voice changer to **Another program over WebSocket** and use `ws://127.0.0.1:8765`.
In this mode the voice changer waits for each chunk (at most the latency budget, then that chunk
is muted); sidecar voices (above) never wait.

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
- **Protocol v1** chunk length is tied to the app **processing chunk** setting (frames per block in the voice-changer plugin) and its sample rate follows the live capture device. **Protocol v2** servers choose their own rates and frame in the welcome.
- Sidecars run on this PC only: Sigil refuses sidecar URLs that are not `127.0.0.1`, `localhost` or `[::1]`, and the scripts refuse to listen anywhere else.
