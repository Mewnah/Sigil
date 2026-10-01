#!/usr/bin/env python3
"""
Reference Sigil neural voice-changer sidecar (WebSocket): an echo server for protocol v1 and v2.

Protocol v1 (Sigil's "Another program over WebSocket" mode; `neural_audio/sidecar_bridge.rs`):
  1. Client sends Text JSON: {"type":"sigil_vc_handshake","protocol_version":1,
     "sample_rate_in":48000,"channels_in":1,"chunk_frames":512}
  2. Client sends Binary: u32 LE sample count + f32 LE interleaved PCM.
  3. Server responds Binary: same layout and length.

Protocol v2 (Sigil's sidecar voices; see PROTOCOL_V2.md): sequenced, pipelined frames with
pitch, rate negotiation and a reference-voice upload. Both are served by `sigil_vc_v2.py`.

This server passes audio through unchanged: a v1 packet comes back byte for byte (any rate, every
channel, no added delay), a v2 frame at the rate Sigil prefers. To build your own converter,
subclass `sigil_vc_v2.Converter` / `Stream` (see `seedvc_sidecar.py` for a complete one) and call
`sigil_vc_v2.main`.

Run: `python vc_sidecar.py` (ws://127.0.0.1:8765; needs `pip install websockets`, plus `numpy` for
protocol v2). Options: `--port`, `--echo-delay-ms 300` (answer each frame later: pipelined
latency), `--echo-compute-ms 150` (spend time per frame: an overloaded model), `--echo-out-rate
22050` (v2), `--echo-single-stream` (one streaming state for every session, as StreamVoiceAnon).
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sigil_vc_v2 as v2  # noqa: E402


def main(argv=None) -> None:
    v2.main(lambda args: v2.EchoConverter(args), "Sigil VC sidecar (echo, protocol v1 and v2)", argv=list(argv or sys.argv[1:]) + ["--echo"])


if __name__ == "__main__":
    main()
