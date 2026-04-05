#!/usr/bin/env python3
"""
Reference Sigil neural voice-changer sidecar (WebSocket).

Protocol v1 (see Sigil `neural_audio/sidecar_bridge.rs`):
  1. Client sends Text JSON: {"type":"sigil_vc_handshake","protocol_version":1,
     "sample_rate_in":48000,"channels_in":1,"chunk_frames":512}
  2. Client sends Binary: u32 LE sample count + f32 LE interleaved PCM.
  3. Server responds Binary: same layout.

This echo server passes audio through unchanged (identity) for testing. Replace
`process_chunk` with your ONNX RVC / VC pipeline.

With `websockets` 11+, the sync server answers Ping with Pong automatically;
`recv()` only yields Text/Binary application data, not raw control frames.
"""

from __future__ import annotations

import json
import struct

from websockets.sync.server import serve


def decode_chunk(data: bytes) -> list[float]:
    if len(data) < 4:
        return []
    (n,) = struct.unpack_from("<I", data, 0)
    need = 4 + n * 4
    if len(data) < need:
        return []
    return list(struct.unpack_from("<" + "f" * n, data, 4))


def encode_chunk(samples: list[float]) -> bytes:
    n = len(samples)
    out = struct.pack("<I", n)
    out += struct.pack("<" + "f" * n, *samples)
    return out


def process_chunk(samples: list[float]) -> list[float]:
    # Echo (identity). Swap for real VC.
    return samples


def handler(ws):
    first = ws.recv()
    if isinstance(first, str):
        try:
            msg = json.loads(first)
        except json.JSONDecodeError:
            return
        if msg.get("type") != "sigil_vc_handshake":
            return
        ws.send(json.dumps({"ok": True, "mode": "echo"}))
    else:
        return

    while True:
        try:
            raw = ws.recv()
        except Exception:
            break
        if isinstance(raw, str):
            continue
        if not isinstance(raw, (bytes, bytearray)) or len(raw) < 4:
            continue
        samples = decode_chunk(raw)
        if not samples:
            continue
        out = process_chunk(samples)
        ws.send(encode_chunk(out))


def main():
    host = "127.0.0.1"
    port = 8765
    print(f"Sigil VC sidecar (echo) on ws://{host}:{port}")
    with serve(handler, host, port) as server:
        server.serve_forever()


if __name__ == "__main__":
    main()
