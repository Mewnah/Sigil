#!/usr/bin/env python3
"""
Reference HTTP TTS sidecar for Sigil (Python stdlib only, no ML dependencies).

It speaks two APIs, so it can stand in for a real speech server while testing Sigil's TTS pipeline:

Sigil "HTTP speech server" backend (`neural-sidecar-tts` plugin):
  GET  /health                          -> 200 "ok"
  POST /speech {"text", "voice", "stream"?}
       -> WAV (16-bit mono). With "stream": true the answer is chunked: a WAV header whose sizes are
          placeholders (0xFFFFFFFF), then the audio word by word as it is "generated".

OpenAI-compatible API (Sigil's "OpenAI-compatible (HTTP)" backend, base URL http://127.0.0.1:8766/v1):
  GET  /v1/models                       -> {"object": "list", "data": [{"id": "sigil-test-tone", ...}]}
  GET  /v1/audio/voices                 -> {"voices": ["default", "low", "high"]}
  POST /v1/audio/speech {"model", "input", "voice", "response_format": "pcm"|"wav", "speed", "stream"}
       -> raw s16le mono PCM at --rate (Content-Type: audio/pcm;rate=<rate>) or WAV; streamed
          (chunked, word by word) when "stream" is true.

The "speech" is a tone per word (pitch by voice), so you can hear sentence splitting, streaming,
stop / skip and routing work. Replace `synthesize_words` with your model (RVC, XTTS, ...).

Security: the server binds 127.0.0.1 by default and has no authentication. Binding another address
(--host 0.0.0.0) exposes it to your network; the script warns when you do.

Usage:
  python tts_sidecar.py                     # http://127.0.0.1:8766
  python tts_sidecar.py --port 8767 --rate 22050 --chunk-delay 0.15
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import math
import struct
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Iterator

VOICES = {"default": 440.0, "low": 220.0, "high": 660.0}
MODEL_ID = "sigil-test-tone"
MAX_BODY = 1 << 20  # 1 MiB of JSON is plenty for one sentence


class Config:
    rate = 24000
    chunk_delay = 0.05  # simulated generation time per word (seconds)


def wav_header(rate: int, data_size: int = 0xFFFFFFFF) -> bytes:
    """16-bit mono WAV header. The default sizes are the streaming placeholders (length unknown)."""
    riff_size = 0xFFFFFFFF if data_size == 0xFFFFFFFF else 36 + data_size
    return (
        b"RIFF"
        + struct.pack("<I", riff_size)
        + b"WAVEfmt "
        + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
        + b"data"
        + struct.pack("<I", data_size)
    )


def tone(freq: float, seconds: float, rate: int) -> bytes:
    """A short tone with 10 ms fades (no clicks), followed by a little silence."""
    n = max(1, int(rate * seconds))
    fade = max(1, int(rate * 0.01))
    out = bytearray()
    for i in range(n):
        env = min(1.0, i / fade, (n - 1 - i) / fade)
        sample = int(9000 * env * math.sin(2 * math.pi * freq * i / rate))
        out += struct.pack("<h", sample)
    out += b"\x00\x00" * int(rate * 0.04)
    return bytes(out)


def synthesize_words(text: str, voice: str, speed: float, rate: int, delay: float) -> Iterator[bytes]:
    """Yield 16-bit PCM word by word, as a streaming model would (replace with your model)."""
    base = VOICES.get(voice, VOICES["default"])
    speed = speed if 0.25 <= speed <= 4.0 else 1.0
    words = text.split() or [""]
    for k, word in enumerate(words):
        if delay > 0:
            time.sleep(delay)
        seconds = (0.08 + 0.03 * min(len(word), 12)) / speed
        yield tone(base * (1.0 + 0.06 * (k % 4)), seconds, rate)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "SigilTtsSidecar/2"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # -- helpers ---------------------------------------------------------------------------

    def send_json(self, status: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length > MAX_BODY:
            self.send_json(413, {"detail": "request too large"})
            return None
        raw = self.rfile.read(length) if length else b"{}"
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self.send_json(400, {"detail": "the body is not JSON"})
            return None
        if not isinstance(data, dict):
            self.send_json(400, {"detail": "the body must be a JSON object"})
            return None
        return data

    def send_audio(self, content_type: str, head: bytes, parts: Iterator[bytes], stream: bool, wav: bool) -> None:
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            try:
                if head:
                    self.write_chunk(head)
                for part in parts:
                    self.write_chunk(part)
                self.wfile.write(b"0\r\n\r\n")
            except (BrokenPipeError, ConnectionResetError):
                # The client cancelled (stop / skip in Sigil): stop generating.
                self.close_connection = True
            return
        audio = b"".join(parts)
        body = (wav_header(Config.rate, len(audio)) if wav else b"") + audio
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def write_chunk(self, data: bytes) -> None:
        if not data:
            return
        self.wfile.write(b"%x\r\n" % len(data) + data + b"\r\n")
        self.wfile.flush()

    # -- routes ----------------------------------------------------------------------------

    def do_GET(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif path == "/v1/models":
            self.send_json(200, {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "sigil"}]})
        elif path == "/v1/audio/voices":
            self.send_json(200, {"voices": list(VOICES)})
        else:
            self.send_json(404, {"detail": "Not Found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0].rstrip("/")
        if path not in ("/speech", "/v1/audio/speech"):
            self.send_json(404, {"detail": "Not Found"})
            return
        data = self.read_json()
        if data is None:
            return
        if path == "/speech":
            text = str(data.get("text") or "")
            voice = str(data.get("voice") or "default")
            parts = synthesize_words(text, voice, 1.0, Config.rate, Config.chunk_delay)
            stream = bool(data.get("stream"))
            head = wav_header(Config.rate) if stream else b""
            self.send_audio("audio/wav", head, parts, stream, wav=True)
            return
        text = str(data.get("input") or "")
        voice = str(data.get("voice") or "default")
        fmt = str(data.get("response_format") or "mp3")
        if fmt not in ("pcm", "wav"):
            self.send_json(422, {"detail": [{"loc": ["body", "response_format"], "msg": f"unsupported format {fmt!r}: use pcm or wav"}]})
            return
        if voice not in VOICES:
            self.send_json(422, {"detail": [{"loc": ["body", "voice"], "msg": f"Voice {voice!r} not found"}]})
            return
        try:
            speed = float(data.get("speed", 1.0))
        except (TypeError, ValueError):
            speed = 1.0
        stream = bool(data.get("stream"))
        parts = synthesize_words(text, voice, speed, Config.rate, Config.chunk_delay)
        if fmt == "pcm":
            self.send_audio(f"audio/pcm;rate={Config.rate}", b"", parts, stream, wav=False)
        else:
            head = wav_header(Config.rate) if stream else b""
            self.send_audio("audio/wav", head, parts, stream, wav=True)


def is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Reference HTTP TTS sidecar for Sigil (test tones).")
    parser.add_argument("--host", default="127.0.0.1", help="address to bind (default 127.0.0.1: this PC only)")
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--rate", type=int, default=24000, help="sample rate of the audio (Hz)")
    parser.add_argument("--chunk-delay", type=float, default=0.05, help="simulated generation time per word (s)")
    args = parser.parse_args()
    if not 8000 <= args.rate <= 192000:
        parser.error("--rate must be 8000-192000")
    Config.rate = args.rate
    Config.chunk_delay = max(0.0, args.chunk_delay)
    if not is_loopback(args.host):
        print(
            f"WARNING: binding {args.host} exposes this server (no authentication) to your network. "
            "Use the default 127.0.0.1 unless you really need it.",
            file=sys.stderr,
        )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    print(f"Sigil TTS sidecar on http://{args.host}:{args.port} (OpenAI-compatible base URL: http://{args.host}:{args.port}/v1)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
