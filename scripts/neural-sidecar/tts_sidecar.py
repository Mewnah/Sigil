#!/usr/bin/env python3
"""
Minimal HTTP TTS sidecar for Sigil `neural-sidecar-tts` plugin.

Endpoints:
  GET  /health  -> 200 OK
  POST /speech  JSON {"text":"...","voice":"default"} -> raw WAV bytes (16-bit PCM mono 22050 Hz placeholder).

Replace `synthesize` with your server (RVC, XTTS, etc.).
"""

from __future__ import annotations

import io
import json
import math
import struct
import wave
from http.server import BaseHTTPRequestHandler, HTTPServer


def synthesize(text: str, voice: str) -> bytes:
    """Tiny beep WAV so the pipeline can be tested without ML deps."""
    _ = voice
    sr = 22050
    duration = min(1.5, 0.15 + len(text) * 0.02)
    n = int(sr * duration)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(n):
            t = i / sr
            sample = int(8000 * math.sin(2 * math.pi * 440 * t))
            frames.extend(struct.pack("<h", max(-32768, min(32767, sample))))
        w.writeframes(bytes(frames))
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")
            return
        self.send_error(404)

    def do_POST(self):
        if self.path.rstrip("/") != "/speech":
            self.send_error(404)
            return
        ln = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(ln) if ln else b"{}"
        try:
            data = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self.send_error(400)
            return
        text = data.get("text") or ""
        voice = data.get("voice") or "default"
        wav = synthesize(text, voice)
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)


def main():
    host = "127.0.0.1"
    port = 8766
    print(f"Sigil neural TTS sidecar on http://{host}:{port}")
    HTTPServer((host, port), Handler).serve_forever()


if __name__ == "__main__":
    main()
