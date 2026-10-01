# Sigil voice-changer sidecar protocol v2

Protocol v2 lets Sigil's voice changer use a voice-conversion model that runs in another process
on this PC (a Python research model such as X-VC, StreamVoiceAnon or Seed-VC) as if it were one
of its own voices: the sidecar runs inside Sigil's stream harness, under the voice changer's
latency budget, with the same resampling, SOLA joins, metrics and TTS mixing as every other voice.

Implementations:

- **Client:** `src-tauri/src/services/neural_audio/sidecar_bridge.rs` (`v2`), used by the
  `sidecar_v2` voice family (`src-tauri/src/services/neural_audio/vc/families/sidecar_v2/`).
- **Server:** `sigil_vc_v2.py` (this folder): the shared server every sidecar script uses. Write
  a converter, not a server (see [Writing a sidecar](#writing-a-sidecar)).

Protocol v1 (the voice changer's **Another program over WebSocket** mode) is unchanged, and both
sides stay compatible with it ([Compatibility with v1](#compatibility-with-v1)).

## What v2 adds over v1

| | v1 | v2 |
|---|---|---|
| Exchange | one chunk, wait for its reply | frames are pipelined: Sigil never waits |
| Timeout | the voice changer's worker blocks up to the budget per chunk | per-frame deadline equal to the latency budget; late frames are dropped, never played |
| Rates | the capture rate, both ways | the server picks its input rate from Sigil's list and its own output rate |
| Frame | the device chunk | the server picks its frame (its model's own block) |
| Reference voice | none | uploaded from the voice library (cached by the server per clip) |
| Parameters | none | per-frame pitch shift; `params` messages (speaker, formant, ...) |
| Recovery | reconnect by restarting the voice changer | reconnects in the background, re-sends the reference, resumes |

## Transport

- **WebSocket, `ws://` only, on this PC only.** Sigil refuses any sidecar URL whose host is not
  `127.0.0.1`, `localhost` or `[::1]`, and `sigil_vc_v2.py` refuses to bind any other address.
  Sigil's own helpers listen on `ws://127.0.0.1:<port>/vc` (the path of a server you run is
  whatever its URL says).
- **Key.** When Sigil starts a sidecar it passes a random key in the `SIGIL_SIDECAR_KEY`
  environment variable (never on the command line) and sends `Authorization: Bearer <key>` with
  the WebSocket upgrade. A server with a key answers any other upgrade with HTTP 401 (Sigil reports
  "the sidecar refused Sigil's key"). A server you start yourself without the variable has no key.
- **Health.** `GET /health` answers `200` once the model is loaded (no key needed). Sigil's
  process host polls it before connecting; a custom server does not need it.
- **Message size.** Sigil accepts messages up to 64 MiB. It sends the reference voice in chunks
  of 65,536 samples (256 KiB), under the 1 MiB default limit of Python's `websockets`.
- All binary numbers are **little-endian**. Audio is mono `f32` in −1..1.

## Session

```
client                                          server
  |-- WebSocket upgrade (+ Authorization) ------->|
  |-- hello (text) ------------------------------>|
  |<---------------------------- welcome (text) --|   or {"ok": false, "error": ...}
  |-- reference (text) + reference frames ------->|   only when the welcome asks for it
  |<------------------------ reference_ready -----|
  |== audio-in frames (binary) ==================>|   pipelined, never waiting
  |<================= audio-out frames (binary) ==|   matched by sequence number
  |-- params / cancel (text) -------------------->|
  |<----------------------- dropped / error ------|
```

### 1. Hello (client → server, text)

```json
{
  "type": "sigil_vc_handshake",
  "protocol_version": 2,
  "sample_rate_in": 16000,
  "channels_in": 1,
  "chunk_frames": 1920,
  "client": "sigil/0.9.0",
  "sample_rates": [16000, 22050, 24000, 32000, 44100, 48000],
  "frame_ms": 120.0,
  "frame_timeout_ms": 300,
  "max_in_flight": 5,
  "params": {"formant_semitones": 0.0, "speaker_id": 0},
  "reference": {"voice_id": "v-1a2b", "name": "Aria", "sha256": "…", "duration_ms": 8200,
                "transcript": "…", "language": "en"},
  "features": ["pitch", "params", "cancel", "reference", "discontinuity"]
}
```

| Field | Meaning |
|---|---|
| `sample_rate_in`, `channels_in`, `chunk_frames` | the v1 handshake fields (the first offered rate, mono, `frame_ms` at that rate), so a v1 server can answer this hello |
| `sample_rates` | input rates Sigil can feed, preferred first |
| `frame_ms` | the frame Sigil proposes (a recipe's own frame, else the voice changer's block, 20–500 ms) |
| `frame_timeout_ms` | the per-frame deadline: the voice changer's latency budget (or the voice's `timeout_ms`), 20–5000 ms. A frame not answered in time is played as silence and its reply discarded |
| `max_in_flight` | frames Sigil expects to keep outstanding (`ceil(budget / frame) + 2` for the proposed frame; informational: Sigil recomputes it for the frame the server picks) |
| `params` | initial parameter values ([Parameters](#parameters)) |
| `reference` | the voice-library clip the voice converts toward (`null` when it has none). The audio itself follows only if the server asks for it |
| `features` | what this client supports (informational) |

Unknown fields must be ignored by servers, so later clients can add fields.

### 2. Welcome (server → client, text)

```json
{
  "type": "sigil_vc_welcome",
  "protocol_version": 2,
  "sample_rate_in": 16000,
  "sample_rate_out": 22050,
  "frame_samples": 2880,
  "latency_ms": 96.5,
  "delay_samples": 5733,
  "reference_required": true,
  "reference_needed": true,
  "reference_cached": false,
  "reference_sample_rate": 22050,
  "max_reference_seconds": 25,
  "params": ["steps", "cfg_rate"],
  "pitch": false,
  "model": "seed-vc",
  "device": "cuda",
  "vram_mb": 1480.0,
  "max_in_flight": 8
}
```

| Field | Default | Meaning |
|---|---|---|
| `sample_rate_in` | required | one of the offered `sample_rates`; the input frames come at this rate |
| `sample_rate_out` | `sample_rate_in` | the rate of the converted audio (8–192 kHz) |
| `frame_samples` | `frame_ms` rounded | input samples per frame (10–2000 ms); `frame_samples × out / in` must be a whole number |
| `latency_ms` | none | the server's compute time per frame (measured at warm-up): Sigil sizes its jitter buffer from it and warns when it exceeds the frame (slower than real time) or the budget |
| `delay_samples` | 0 | the model's algorithmic delay in output samples: the reply to frame *n* is the audio that ends this long before frame *n*'s end (lookahead, overlap). At most 10 s |
| `reference_required` | `false` | the model cannot run without a reference voice: a voice without one is refused before streaming |
| `reference_needed` | `reference_required` | upload the reference now (step 3) |
| `reference_cached` | `false` | the server already prepared the clip with this `sha256` (no upload) |
| `reference_sample_rate` | `sample_rate_in` | the rate of the uploaded clip |
| `max_reference_seconds` | 30 | longer clips are cut (at most 120 s) |
| `params` | `[]` | parameter names the server understands |
| `pitch` | `false` | the server applies the per-frame pitch shift |
| `model`, `device`, `vram_mb` | — | shown and logged; `device` (`cuda`, `dml`, `cpu`, …) becomes the voice's execution-provider outcome, `vram_mb` its GPU memory claim |
| `max_in_flight` | no limit | frames the server queues at most (1–1024). Sigil keeps `ceil(budget / frame) + 2` frames of the negotiated frame in flight, or this many when it is lower |

To refuse the session, send `{"ok": false, "error": "why"}` (`message` or `reason` work too) and
close; Sigil shows the reason. A welcome Sigil cannot use (a rate it did not offer, a frame that is
not a whole number of output samples, …) fails the voice's load with a message naming the field.
Any other first message (`{"ok": true}`, `ok`, a binary message) is a **v1 acknowledgement**.

### 3. Reference upload (only when `reference_needed`)

1. Text: `{"type": "reference", "voice_id", "name", "sha256", "transcript", "language",
   "sample_rate", "samples"}`.
2. Binary frames of kind 3 (`reference`): `seq` is the chunk index, at most 65,536 samples each,
   the last one flagged `LAST` (0x02). An empty clip is one empty frame flagged `LAST`.
3. The server prepares the voice (speaker embedding, prompt features, …) and answers
   `{"type": "reference_ready"}`, or `{"type": "error", "message": …}` /
   `{"ok": false, …}` to refuse it (the voice's load fails with that message).

Sigil waits up to 120 s for `reference_ready`. Servers should cache prepared voices by `sha256`
(the reference server keeps the last four) and answer `reference_cached: true` the next time,
so switching back to a voice or reconnecting costs no upload. Prepare a reference **beside** the
frames of other sessions, not on their inference thread: Sigil loads a new voice while the current
one keeps playing, and frames held up behind a reference preparation go stale (the reference
server prepares references on a thread of their own).

### 4. Streaming

**Binary frame** (20-byte header, then the samples):

| Offset | Type | Field |
|---|---|---|
| 0 | 4 bytes | magic `SGV2` |
| 4 | u8 | kind: 1 = audio in, 2 = audio out, 3 = reference |
| 5 | u8 | flags: 0x01 = `DISCONTINUITY` (audio in), 0x02 = `LAST` (reference) |
| 6 | u16 | reserved, 0 |
| 8 | u32 | sequence number |
| 12 | u32 | sample count *n* (at most 2²⁰) |
| 16 | f32 | audio in: the pitch shift in semitones; otherwise 0 |
| 20 | f32 × *n* | samples |

The message length must be exactly `20 + 4n`.

- **Audio in** (client → server): exactly `frame_samples` samples at `sample_rate_in`. Sequence
  numbers start at 0 and count frames (wrapping at 2³²); a number may be skipped (a frame Sigil
  could not send). `DISCONTINUITY` marks a frame that does not continue the previous one (the
  stream's first frame, a skipped frame, a reconnect, a pause, a voice switch): reset the streaming
  state (rolling context, lookahead, overlap, crossfade tail, delay line). Audio from before it
  must never come out after it: the reply to a frame ends `delay_samples` before the frame's end,
  so a window kept across a pause would play the end of the previous phrase in front of the new
  one. The pitch field is the voice's pitch shift; apply it only if the welcome said
  `"pitch": true`.
- **Audio out** (server → client): kind 2 with the **same sequence number**, `frame_samples × out
  / in` samples at `sample_rate_out` (a length off by at most 2 % or 16 samples is padded or cut;
  anything else, or NaN/inf, mutes that frame). Replies normally come in order but are matched by
  number. Every frame gets a reply, a `dropped` message or an `error` message.
- **Pipelining.** Sigil sends each frame the moment it is captured and never waits for a reply;
  up to `max_in_flight` frames are outstanding. Run the model on one inference thread in frame
  order and keep reading the socket meanwhile.
- **Deadline.** A frame that has waited in the server's queue longer than `frame_timeout_ms`
  should be dropped unconverted with `{"type": "dropped", "seq": n, "reason": "stale"}`; Sigil
  has played silence for it already. Converting it anyway only wastes time: its reply is discarded.

**Text, client → server**

| Message | Meaning |
|---|---|
| `{"type": "params", "params": {…}}` | new parameter values, applying from the next frame |
| `{"type": "cancel", "before_seq": n}` | frames numbered before *n* are no longer wanted: drop them unconverted (Sigil gave them up at the budget, or the stream restarted) |

**Text, server → client**

| Message | Meaning |
|---|---|
| `{"type": "dropped", "seq": n, "reason": "…"}` | frame *n* will not be answered (Sigil mutes it and reports it) |
| `{"type": "dropped", "seq": n, "reason": "superseded"}` | frame *n* was not converted because another session holds the model ([Single-stream models](#single-stream-models)); Sigil plays silence for it without an error |
| `{"type": "error", "seq": n, "message": "…"}` | frame *n* failed (Sigil mutes it) |
| `{"type": "error", "message": "…"}` | a non-fatal problem (logged) |
| `{"type": "error", "message": "…", "fatal": true}` | the session is broken: Sigil drops the connection and reconnects |
| `{"type": "stats", …}` | optional telemetry (ignored for now) |

Unknown message types are ignored on both sides.

### Parameters

Sigil sends a parameter only when the welcome lists it:

| Name | Type | From |
|---|---|---|
| `speaker_id` | integer | the voice's speaker |
| `formant_semitones` | float | the voice's formant control |
| `noise_scale` | float | the voice's noise control |

A server may list its own (the Seed-VC sidecar takes `steps` and `cfg_rate`, StreamVoiceAnon
`alpha`); today their values come from the recipe's command-line arguments.

### Several sessions

Sigil opens one session per loaded voice, and several voices of the same helper can be connected
at once: a voice's replacement loads (and connects) while the voice still plays, and idle voices
stay loaded (`hot_slots`). A server must therefore accept a second session while the first is
connected, never refuse it.

### Single-stream models

A model that keeps one streaming state (and one reference voice) for the whole process, such as
StreamVoiceAnon's real-time wrapper, still serves every session, but converts for one at a time:

- The session whose frame carries `DISCONTINUITY` takes the model over (Sigil flags the first
  frame of a voice that starts, is switched to, or restarts after a pause): the server installs
  that session's reference voice and parameters and resets the streaming state, then converts it.
- A session also takes the model over when the holder has sent nothing for 0.5 s (or four of its
  frames, whichever is longer): a switch that was abandoned gives the model back.
- Every other session's frames are answered `dropped` with reason `superseded`: the voice being
  replaced plays silence for the moment before the switch, without an error.
- A session that closes frees the model at once.

### Reconnecting

When the connection drops (or the server sends a fatal error), Sigil reports it, mutes the frames
in flight, and reconnects in the background (backoff 0.25 → 5 s): hello, welcome, the reference
again (or `reference_cached`). The first frame after a reconnect carries `DISCONTINUITY`. A server
that comes back with another protocol version, input or output rate or frame stops the voice
("load the voice again").

## How Sigil plays a sidecar voice

The harness runs the sidecar voice in fixed blocks of about 20 ms (a *slot*: a divisor of the
frame at both rates; X-VC's 120 ms frame is 6 slots, Seed-VC's 180 ms frame 9). Every slot, the
voice takes the replies that arrived, sends the frame that slot completes, and plays one slot from
a jitter buffer. Nothing in the voice changer's loop ever waits for the sidecar.

- **Start.** The first reply sets the delay: its round trip plus 30 ms of slack, rounded up to
  slots (never less than the welcome's `latency_ms` estimate allows for).
- **Late frames** are waited for one slot at a time (a short gap; the delay grows by that slot)
  until the frame's first slot is a whole latency budget past its send. Then it is **given up**:
  that block is muted (an error the voice changer reports, never the dry microphone), Sigil sends
  `cancel`, and the late reply is discarded when it comes. A stalled sidecar therefore costs muted
  audio, never a blocked loop or a growing backlog.
- **Lowering the delay.** When every reply in the last 4 s came back faster than the current delay
  (with slack), the delay target drops to what the slowest of them needed plus the slack, and quiet
  slots (below −45 dBFS) are skipped until the delay is there; speech is never cut. After a stall
  the voice is back at its steady delay within a pause or two.
- **End-to-end delay** of a sidecar voice ≈ frame + round trip (compute, transfer) + slack + the
  server's `delay_samples` + the harness's SOLA and resampler delay (the same as for every voice).
- After a pause of more than 2 s plus the budget (a hot voice that was not playing), everything
  outstanding is stale: the voice starts a new stream (`cancel`, then `DISCONTINUITY`) from its
  starting delay (a delay a stall grew to before the pause is forgotten).
- **Loading.** A voice whose load is no longer wanted (another voice was selected, or the voice
  changer stopped) stops waiting for its helper's start and its handshake at once; a helper start
  nobody needs any more is called off.

## Compatibility with v1

- **v2 client, v1 server.** The hello carries the v1 fields, so a v1 server acknowledges it as
  usual. Sigil then sends v1 packets (`u32` sample count, then `f32` samples) of `chunk_frames`
  samples at the first offered rate, mono, and matches the same-length replies in order. No pitch,
  parameters, cancellation or reference; the voice loads with a warning saying so, and a recipe
  that needs a reference refuses the server.
- **v1 client, v2 server.** `sigil_vc_v2.py` serves v1 hellos (`protocol_version` 1): same-length
  replies at the client's rate and channel count, resampled to and from the model, with a FIFO
  holding the model's frame. The echo (`vc_sidecar.py`, `--echo`) returns every v1 packet as it
  came: any rate, every channel, no added delay (and it runs without numpy). A model that needs a
  reference refuses v1 clients.

## Writing a sidecar

Subclass `sigil_vc_v2.Converter` (the loaded model: rates, frame, delay, reference policy,
parameters) and `sigil_vc_v2.Stream` (one session's streaming state), and call
`sigil_vc_v2.main(make_converter, description)`:

```python
import sigil_vc_v2 as v2

class MyStream(v2.Stream):
    def __init__(self, conv, frame_out):
        self.conv, self.frame_out = conv, frame_out

    def set_reference(self, prepared):          # what Converter.prepare_reference returned
        self.style = prepared

    def process(self, pcm, pitch, discontinuity):
        # pcm: frame_samples float32 at the input rate; return frame_out samples at the output rate
        if discontinuity:
            ...                                 # forget context, overlap and crossfade tail
        return v2.fit_length(self.conv.model(pcm), self.frame_out)

class MyConverter(v2.Converter):
    name = "my-vc"
    input_rates = (16000,)
    output_rate = 16000
    frame_ms = 100.0          # None = take Sigil's proposal
    delay_samples = 0         # lookahead / overlap in output samples
    reference = "optional"    # "none", "optional" or "required"
    pitch = False

    def __init__(self, args):
        self.device = args.device

    def load(self):
        self.model = ...      # load from args.model_dir on args.device

    def warmup(self):         # measure latency_ms (reported in the welcome)
        s = self.open_stream(16000, 1600)
        self.time_warmup(lambda: s.process(v2.synthetic_voice(0.1, 16000), 0.0, False))

    def prepare_reference(self, pcm, rate, meta):
        return ...            # cached per sha256 and shared between sessions; runs on its own
                              # thread beside other sessions' frames: read the model, change no
                              # stream (set parallel_reference = False if it cannot)

    def open_stream(self, rate_in, frame_in):
        return MyStream(self, frame_in * self.output_rate // rate_in)

if __name__ == "__main__":
    v2.main(MyConverter, "My voice-conversion sidecar for Sigil")
```

A model with one streaming state for the whole process sets `single_stream = True` and implements
`Stream.activate()` (install this session's reference and parameters in the model, reset its
state): the server then arbitrates between sessions as in [Single-stream models](#single-stream-models)
(see `streamvoiceanon_sidecar.py`).

The shared server handles the WebSocket, the key, `/health`, negotiation, the reference upload and
its cache (prepared beside the frames), the inference queue, stale-frame drops, `cancel`,
`params`, single-stream arbitration and v1 clients. Every
script also takes `--echo` (no model: a pass-through with `--echo-delay-ms` /
`--echo-compute-ms` / `--echo-out-rate` / `--echo-single-stream`) and `--self-test
[reference.wav]` (convert a synthetic phrase and print timings as JSON).
