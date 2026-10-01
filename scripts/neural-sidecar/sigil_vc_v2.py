# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sigil voice-changer sidecar protocol v2: the server side shared by every sidecar script.

The wire format is specified in ``PROTOCOL_V2.md`` (next to this file); the Rust client is
``src-tauri/src/services/neural_audio/sidecar_bridge.rs`` (``v2``). In short:

* WebSocket on ``ws://127.0.0.1:<port>/vc`` (this module refuses any other bind address), plus
  ``GET /health`` (200 once the model is loaded). When ``SIGIL_SIDECAR_KEY`` is set (Sigil sets it
  for the helpers it starts), every WebSocket upgrade needs ``Authorization: Bearer <key>``.
* Client hello (text) -> server welcome (text): the server picks the input rate, its output rate
  and the frame; optional reference upload (text ``reference`` + binary chunks) answered by
  ``reference_ready``.
* Binary audio frames (20-byte ``SGV2`` header with a sequence number and the pitch shift) are
  queued and converted in order on one inference thread; each reply carries the frame's sequence
  number. Frames that waited longer than the client's frame timeout, or that the client cancelled,
  are dropped without being converted (the client already played silence for them). Reference
  voices are prepared on a thread of their own, so a voice that connects never holds up the frames
  of one that is playing.
* A model with one streaming state for the whole process (``Converter.single_stream``) still takes
  any number of sessions: one converts at a time, and a session takes the model over when its
  stream (re)starts (a frame flagged ``DISCONTINUITY``) or when the holder has gone quiet. The
  other sessions' frames are answered ``dropped`` / ``superseded``.
* Protocol v1 clients (Sigil's "another program over WebSocket" mode) are served too: same-length
  replies in order, at the client's rate and channel count (the echo converter returns them
  untouched).

A sidecar script provides a :class:`Converter` (the model) whose :class:`Stream` converts one
frame at a time, and calls :func:`main`. ``--echo`` replaces the model with a pass-through (with
optional simulated latency), which tests the protocol and Sigil's helper wiring without a model.

Dependencies: ``websockets`` (10 or newer; both the legacy and the new asyncio API are supported)
and ``numpy`` (every model, and protocol v2; the v1 echo runs without it). ``scipy`` is used for
resampling when it is installed.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import concurrent.futures
import hmac
import ipaddress
import json
import logging
import math
import os
import struct
import sys
import time
import wave
from typing import Callable, Optional

try:
    import numpy as np
except ImportError:  # only the v1 echo works without it (`main` says so)
    np = None

PROTOCOL_VERSION = 2
HELLO_TYPE = "sigil_vc_handshake"
WELCOME_TYPE = "sigil_vc_welcome"
MAGIC = b"SGV2"
HEADER = struct.Struct("<4sBBHIIf")  # magic, kind, flags, reserved, seq, samples, pitch
KIND_AUDIO_IN, KIND_AUDIO_OUT, KIND_REFERENCE = 1, 2, 3
FLAG_DISCONTINUITY, FLAG_LAST = 0x01, 0x02
MAX_FRAME_SAMPLES = 1 << 20
MAX_MESSAGE_BYTES = 64 << 20
KEY_ENV = "SIGIL_SIDECAR_KEY"
HEALTH_PATH = "/health"
WS_PATHS = ("/vc", "/")
REFERENCE_CACHE = 4
#: Single-stream models: the stream holding the model is quiet (no frame) for this long, or four of
#: its frames, whichever is longer, before another session may take the model over without
#: restarting its stream.
OWNER_IDLE_S = 0.5
#: ``dropped`` reasons.
DROP_STALE, DROP_SUPERSEDED = "stale", "superseded"

LOG = logging.getLogger("sigil-sidecar")


# ---------------------------------------------------------------------------------------------
# Wire format
# ---------------------------------------------------------------------------------------------

def encode_frame(kind: int, flags: int, seq: int, pcm, pitch: float = 0.0) -> bytes:
    pcm = np.ascontiguousarray(pcm, dtype="<f4").reshape(-1)
    return HEADER.pack(MAGIC, kind, flags, 0, seq & 0xFFFFFFFF, pcm.size, float(pitch)) + pcm.tobytes()


def decode_frame(data: bytes):
    """``(kind, flags, seq, pitch, pcm)``; raises ValueError on a malformed frame."""
    if len(data) < HEADER.size:
        raise ValueError(f"frame too short ({len(data)} bytes)")
    magic, kind, flags, _reserved, seq, n, pitch = HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not a protocol v2 frame")
    if n > MAX_FRAME_SAMPLES or len(data) != HEADER.size + 4 * n:
        raise ValueError(f"frame declares {n} samples but carries {len(data) - HEADER.size} bytes")
    pcm = np.frombuffer(data, dtype="<f4", count=n, offset=HEADER.size).astype(np.float32)
    return kind, flags, seq, pitch, pcm


def encode_v1(pcm) -> bytes:
    pcm = np.ascontiguousarray(pcm, dtype="<f4").reshape(-1)
    return struct.pack("<I", pcm.size) + pcm.tobytes()


def v1_packet(data: bytes) -> bytes:
    """The ``u32`` count and the samples of a v1 packet (trailing bytes cut); raises ValueError
    when it is short. Needs no numpy (the v1 echo)."""
    if len(data) < 4:
        raise ValueError("v1 packet too short")
    (n,) = struct.unpack_from("<I", data, 0)
    if len(data) < 4 + 4 * n:
        raise ValueError(f"v1 packet needs {4 + 4 * n} bytes, has {len(data)}")
    return bytes(data[: 4 + 4 * n])


def decode_v1(data: bytes) -> np.ndarray:
    packet = v1_packet(data)
    return np.frombuffer(packet, dtype="<f4", count=(len(packet) - 4) // 4, offset=4).astype(np.float32)


def seq_before(a: int, b: int) -> bool:
    """``a`` comes before ``b`` (32-bit sequence numbers, wrapping)."""
    return a != b and ((b - a) & 0xFFFFFFFF) < 0x80000000


# ---------------------------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------------------------

def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Band-limited when scipy is installed, else linear (good enough for the echo / v1 paths)."""
    x = np.asarray(x, np.float32).reshape(-1)
    if sr_from == sr_to or x.size == 0:
        return x
    try:
        from scipy.signal import resample_poly

        g = math.gcd(int(sr_from), int(sr_to))
        return resample_poly(x, sr_to // g, sr_from // g).astype(np.float32)
    except ImportError:
        n_out = int(round(x.size * sr_to / sr_from))
        t = np.arange(n_out, dtype=np.float64) * (sr_from / sr_to)
        return np.interp(t, np.arange(x.size), x).astype(np.float32)


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, np.float32).reshape(-1)
    if x.size == n:
        return x
    if x.size > n:
        return x[:n]
    return np.concatenate([x, np.zeros(n - x.size, np.float32)])


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    return 10.0 * math.log10(float(np.mean(np.square(x, dtype=np.float64))) + 1e-12)


class OutputFifo:
    """Turns a converter's variable-length output into exactly ``n`` samples per frame: starts with
    ``prefill`` samples of silence (the algorithmic delay it reports) and pads with silence when
    the converter falls short."""

    def __init__(self, prefill: int = 0):
        self.buf = np.zeros(int(prefill), np.float32)

    def push(self, x) -> None:
        self.buf = np.concatenate([self.buf, np.asarray(x, np.float32).reshape(-1)])

    def pop(self, n: int) -> np.ndarray:
        out = fit_length(self.buf[:n], n)
        self.buf = self.buf[n:]
        return out


def cosine_fade(n: int):
    """``(fade_in, fade_out)`` windows of ``n`` samples (equal-power cosine)."""
    if n <= 0:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    t = (np.arange(n, dtype=np.float32) + 0.5) / n
    fade_in = np.sin(0.5 * np.pi * t) ** 2
    return fade_in.astype(np.float32), (1.0 - fade_in).astype(np.float32)


def read_wav_mono(path: str):
    """``(samples, rate)`` of a WAV file (PCM 16/24/32-bit or float), channels averaged."""
    try:
        import soundfile as sf

        data, rate = sf.read(path, dtype="float32", always_2d=True)
        return data.mean(axis=1).astype(np.float32), int(rate)
    except ImportError:
        pass
    with wave.open(path, "rb") as w:
        ch, width, rate, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 2:
        x = np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0
    elif width == 4:
        x = np.frombuffer(raw, "<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"{path}: {8 * width}-bit WAV needs the soundfile package")
    return x.reshape(-1, ch).mean(axis=1), rate


def write_wav16(path: str, x: np.ndarray, rate: int) -> None:
    pcm = (np.clip(np.asarray(x, np.float32), -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(rate))
        w.writeframes(pcm.tobytes())


# ---------------------------------------------------------------------------------------------
# Converter interface
# ---------------------------------------------------------------------------------------------

class Stream:
    """One client session's streaming state. Every method runs on the inference thread."""

    def set_reference(self, prepared) -> None:
        """Install the reference voice (what :meth:`Converter.prepare_reference` returned)."""
        pass

    def set_params(self, params: dict) -> None:
        pass

    def activate(self) -> None:
        """Single-stream models (:attr:`Converter.single_stream`): this session's stream takes the
        model over (install its reference and parameters, start a fresh streaming state). The
        server calls it before the stream's first frame and whenever it takes the model back."""
        pass

    def process(self, pcm: np.ndarray, pitch: float, discontinuity: bool) -> np.ndarray:
        """Convert one frame (``frame_in`` samples at the stream's input rate) and return the
        frame's output (``frame_out`` samples at the output rate; the reply to frame n is the audio
        that ends ``delay_samples`` before frame n's end). ``discontinuity``: the frame does not
        continue the previous one (the stream's first frame, a pause, frames dropped in between):
        forget the streaming state (context, overlap, crossfade tail), so audio from before it
        never plays after it."""
        raise NotImplementedError

    def close(self) -> None:
        pass


class Converter:
    """A loaded model. Attributes are what the welcome reports."""

    name = "converter"
    #: Input rates the model takes, preferred first (the client offers 16 kHz first); ``None`` =
    #: any rate (the client's preferred one).
    input_rates: Optional[tuple] = (16000,)
    output_rate = 16000
    #: The model's own frame (ms); ``None`` = take the client's proposal.
    frame_ms: Optional[float] = None
    #: Algorithmic delay in output samples (see :meth:`Stream.process`).
    delay_samples = 0
    #: Measured compute per frame (set by :meth:`warmup`).
    latency_ms: Optional[float] = None
    #: Parameter names the streams understand (``params`` messages).
    params = ()
    #: The streams apply the per-frame pitch shift.
    pitch = False
    #: ``none``, ``optional`` or ``required``.
    reference = "none"
    reference_rate = 16000
    max_reference_seconds = 30.0
    device = "cpu"
    #: The model keeps one streaming state (and one reference voice) for the whole process: the
    #: server lets one session convert at a time (see the module docs) and calls
    #: :meth:`Stream.activate` when a session takes the model over.
    single_stream = False
    #: :meth:`prepare_reference` may run on its own thread while frames convert (it reads the model
    #: and returns the prepared voice without changing any stream). ``False`` runs it on the
    #: inference thread, between frames.
    parallel_reference = True
    #: Protocol v1 replies are the client's packets, untouched (the echo).
    v1_passthrough = False

    def rate_for(self, offered) -> Optional[int]:
        """The input rate to use of the client's ``offered`` rates (preferred first), if any."""
        if self.input_rates is None:
            return int(offered[0]) if offered else None
        return next((r for r in self.input_rates if r in offered), None)

    def output_rate_for(self, rate_in: int) -> int:
        return self.output_rate

    def frame_samples(self, rate_in: int, proposed: int) -> int:
        """Frame at ``rate_in`` (a whole number of output samples)."""
        n = proposed if self.frame_ms is None else int(round(self.frame_ms * rate_in / 1000.0))
        out_rate = self.output_rate_for(rate_in)
        unit = rate_in // math.gcd(rate_in, out_rate)
        return max(unit, int(round(n / unit)) * unit)

    def vram_mb(self) -> Optional[float]:
        """GPU memory the model holds (MB), or ``None`` when unknown (Sigil then claims the recipe's
        estimate). PyTorch models report what torch reserved; other runtimes override this."""
        if self.device in ("cpu", "dml") or "torch" not in sys.modules:
            return None
        try:
            import torch

            if torch.cuda.is_available():
                reserved = float(torch.cuda.memory_reserved()) / (1 << 20)
                # torch imported for something else (a front end) holds nothing: unknown, not 0.
                return reserved if reserved > 0 else None
        except Exception:  # noqa: BLE001
            pass
        return None

    def load(self) -> None:
        pass

    def warmup(self) -> None:
        pass

    def prepare_reference(self, pcm: np.ndarray, rate: int, meta: dict):
        """Prepare a reference voice (cached by sha256 and shared between streams). Runs on a
        thread of its own while other sessions convert (unless :attr:`parallel_reference` is
        false): return the prepared voice, change no stream (:meth:`Stream.set_reference` installs
        it)."""
        return pcm

    def open_stream(self, rate_in: int, frame_in: int) -> Stream:
        raise NotImplementedError

    def time_warmup(self, run: Callable[[], None], runs: int = 5, untimed: int = 2) -> None:
        """Set :attr:`latency_ms` to the median of ``runs`` timed calls after ``untimed`` calls
        (first-call setup, and time for a GPU to leave its idle clocks)."""
        for _ in range(untimed):
            run()
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            run()
            times.append((time.perf_counter() - t0) * 1000.0)
        self.latency_ms = float(np.median(times))


def nvidia_process_vram_mb(pid: Optional[int] = None) -> Optional[float]:
    """GPU memory a process uses on NVIDIA GPUs (``nvidia-smi``; this process by default), or
    ``None`` when unknown (no NVIDIA driver, or the process holds none)."""
    import subprocess

    pid = os.getpid() if pid is None else pid
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    used = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0] == str(pid):
            try:
                used.append(float(parts[1]))
            except ValueError:
                pass
    return sum(used) if used else None


class EchoStream(Stream):
    def __init__(self, conv: "EchoConverter", rate_in: int, frame_out: int):
        self.conv, self.rate_in, self.frame_out = conv, rate_in, frame_out

    def activate(self):
        self.conv.activations += 1

    def process(self, pcm, pitch, discontinuity):
        if self.conv.compute_s > 0:
            time.sleep(self.conv.compute_s)
        return fit_length(resample(pcm, self.rate_in, self.conv.output_rate_for(self.rate_in)), self.frame_out)


class EchoConverter(Converter):
    """Pass-through: tests the protocol and the helper wiring. It takes any input rate (the
    client's), and protocol v1 packets come back untouched. ``--echo-delay-ms`` answers each frame
    that much later without holding up the next one (pipelined latency); ``--echo-compute-ms``
    spends that long on the inference thread (an overloaded model); ``--echo-single-stream``
    behaves like a model with one streaming state (StreamVoiceAnon)."""

    name = "echo"
    input_rates = None
    reference = "optional"
    v1_passthrough = True
    #: Common rates planned at warm-up.
    WARM_RATES = (16000, 22050, 24000, 32000, 44100, 48000)

    def __init__(self, args, reference: str = "optional", single_stream: Optional[bool] = None):
        self.reply_delay_s = max(0.0, float(getattr(args, "echo_delay_ms", 0) or 0) / 1000.0)
        self.compute_s = max(0.0, float(getattr(args, "echo_compute_ms", 0) or 0) / 1000.0)
        self.out_rate = int(getattr(args, "echo_out_rate", 0) or 0)
        self.frame_ms = getattr(args, "frame_ms", None)
        self.reference = reference
        self.single_stream = bool(getattr(args, "echo_single_stream", False) if single_stream is None else single_stream)
        self.latency_ms = self.reply_delay_s * 1000.0 + self.compute_s * 1000.0
        self.pitch = False
        #: Times a session took the model over (single-stream; tests).
        self.activations = 0

    def output_rate_for(self, rate_in):
        return self.out_rate or rate_in

    def warmup(self):
        # Import and plan the resampler now, as a model's warm-up would, so the first frame is not
        # a one-off slow call.
        if np is None:
            return
        for rate in self.WARM_RATES:
            resample(np.zeros(rate // 50, np.float32), rate, self.output_rate_for(rate))

    def open_stream(self, rate_in, frame_in):
        out = frame_in * self.output_rate_for(rate_in) // rate_in
        return EchoStream(self, rate_in, out)


# ---------------------------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------------------------

def is_loopback(host: str) -> bool:
    h = host.strip().strip("[]")
    if h.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(h).is_loopback
    except ValueError:
        return False


class _Session:
    """One v2 connection, as its reader, its processor and the server's arbitration see it."""

    def __init__(self, frame_s: float):
        self.frame_s = frame_s
        self.cancel_before = None
        #: ``time.monotonic()`` of the newest audio frame received.
        self.last_audio: Optional[float] = None
        #: A frame flagged DISCONTINUITY was dropped before it could take the model over (stale or
        #: cancelled): the next frame restarts the stream in its place.
        self.restart_pending = False


class Server:
    def __init__(self, conv: Converter):
        self.conv = conv
        self.key = os.environ.get(KEY_ENV) or None
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sidecar-infer")
        # Reference voices are prepared beside the frames (seconds of audio through the model's
        # front end), so a voice that connects never holds up the one that plays.
        self.ref_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="sidecar-reference")
        self.references = collections.OrderedDict()
        self.streams = 0
        #: Single-stream models: the session whose stream holds the model.
        self.owner: Optional[_Session] = None

    async def run_blocking(self, fn, *args):
        return await asyncio.get_running_loop().run_in_executor(self.executor, fn, *args)

    async def run_reference(self, fn, *args):
        executor = self.ref_executor if self.conv.parallel_reference else self.executor
        return await asyncio.get_running_loop().run_in_executor(executor, fn, *args)

    def claim(self, sess: _Session, restart: bool) -> Optional[bool]:
        """Single-stream models: may ``sess`` convert its next frame? ``False``: yes, it holds the
        model; ``True``: yes, it takes the model over now (its stream (re)starts, ``restart``, or
        the holder has gone quiet); ``None``: no, another session's stream holds it."""
        owner = self.owner
        if owner is sess:
            return False
        quiet = owner is None or owner.last_audio is None or (
            time.monotonic() - owner.last_audio > max(OWNER_IDLE_S, 4.0 * owner.frame_s))
        if restart or quiet:
            self.owner = sess
            return True
        return None

    def release(self, sess: _Session) -> None:
        if self.owner is sess:
            self.owner = None

    # -- HTTP: health and auth --------------------------------------------------------------

    def http_gate(self, path: str, headers):
        """``None`` lets the WebSocket handshake proceed, else ``(status, body)``."""
        path = (path or "/").split("?", 1)[0]
        if path == HEALTH_PATH:
            return 200, "ok\n"
        if path not in WS_PATHS:
            return 404, "not found\n"
        if self.key:
            got = headers.get("Authorization") or headers.get("authorization") or ""
            if not hmac.compare_digest(got.encode(), f"Bearer {self.key}".encode()):
                return 401, "unauthorized\n"
        return None

    def serve(self, host: str, port: int):
        """The server object (an async context manager) of whichever websockets API is installed."""
        try:
            from websockets.asyncio.server import serve as serve_new  # websockets >= 13
        except ImportError:
            serve_new = None
        common = dict(max_size=MAX_MESSAGE_BYTES, ping_interval=20, ping_timeout=60)
        if serve_new is not None:
            def process_request(connection, request):
                r = self.http_gate(request.path, request.headers)
                return None if r is None else connection.respond(r[0], r[1])

            return serve_new(self.handle, host, port, process_request=process_request, **common)
        import websockets.server as legacy

        async def process_request(path, headers):
            r = self.http_gate(path, headers)
            if r is None:
                return None
            return r[0], [("Content-Type", "text/plain")], r[1].encode()

        async def handler(ws, path=None):
            await self.handle(ws)

        return legacy.serve(handler, host, port, process_request=process_request, **common)

    async def run(self, host: str, port: int) -> None:
        asyncio.get_running_loop().set_exception_handler(_quiet_known_noise)
        async with self.serve(host, port):
            LOG.info("listening on ws://%s:%d/vc (model %s, device %s)", host, port, self.conv.name, self.conv.device)
            print(f"Sigil VC sidecar ({self.conv.name}) ready on ws://{host}:{port}/vc", flush=True)
            await asyncio.Future()

    # -- Sessions ---------------------------------------------------------------------------

    async def handle(self, ws) -> None:
        try:
            raw = await asyncio.wait_for(ws.recv(), 15)
            hello = json.loads(raw) if isinstance(raw, str) else None
        except Exception:  # noqa: BLE001
            return
        if not isinstance(hello, dict) or hello.get("type") != HELLO_TYPE:
            await self.refuse(ws, "expected a sigil_vc_handshake")
            return
        # Every session is served, also by a single-stream model (Sigil loads a voice's
        # replacement while the voice still plays): `claim` decides which one converts.
        self.streams += 1
        try:
            if int(hello.get("protocol_version") or 1) >= PROTOCOL_VERSION:
                await self.serve_v2(ws, hello)
            else:
                await self.serve_v1(ws, hello)
        except Exception as e:  # noqa: BLE001
            if not _is_closed(e):
                LOG.exception("session failed: %s", e)
        finally:
            self.streams -= 1

    @staticmethod
    async def refuse(ws, why: str) -> None:
        try:
            await ws.send(json.dumps({"ok": False, "error": why}))
        except Exception:  # noqa: BLE001
            pass

    async def serve_v2(self, ws, hello: dict) -> None:
        conv = self.conv
        if np is None:
            await self.refuse(ws, "this sidecar needs numpy for protocol v2 (pip install numpy)")
            return
        offered = [int(r) for r in (hello.get("sample_rates") or [hello.get("sample_rate_in") or 16000])]
        rate = conv.rate_for(offered)
        if rate is None:
            await self.refuse(ws, f"this sidecar takes {', '.join(map(str, conv.input_rates))} Hz input")
            return
        out_rate = conv.output_rate_for(rate)
        proposed = int(round(float(hello.get("frame_ms") or 100.0) * rate / 1000.0))
        frame_in = conv.frame_samples(rate, proposed)
        frame_out = frame_in * out_rate // rate
        ref_meta = hello.get("reference") if isinstance(hello.get("reference"), dict) else None
        if conv.reference == "required" and not ref_meta:
            await self.refuse(ws, "this sidecar converts toward a reference voice; choose one in Sigil")
            return
        timeout_s = float(hello.get("frame_timeout_ms") or 0) / 1000.0
        stream = await self.run_blocking(conv.open_stream, rate, frame_in)
        sess = _Session(frame_in / float(rate))
        try:
            cache_key = None
            cached = None
            if ref_meta and conv.reference != "none":
                sha = str(ref_meta.get("sha256") or "")
                cache_key = (sha, conv.reference_rate) if sha else None
                cached = self.references.get(cache_key) if cache_key else None
            params = hello.get("params") if isinstance(hello.get("params"), dict) else {}
            if params:
                await self.run_blocking(stream.set_params, params)
            welcome = {
                "type": WELCOME_TYPE,
                "protocol_version": PROTOCOL_VERSION,
                "ok": True,
                "sample_rate_in": rate,
                "sample_rate_out": out_rate,
                "frame_samples": frame_in,
                "latency_ms": conv.latency_ms,
                "delay_samples": int(conv.delay_samples),
                "reference_required": conv.reference == "required",
                "reference_needed": bool(ref_meta) and conv.reference != "none" and cached is None,
                "reference_cached": cached is not None,
                "reference_sample_rate": conv.reference_rate,
                "max_reference_seconds": conv.max_reference_seconds,
                "params": list(conv.params),
                "pitch": bool(conv.pitch),
                "model": conv.name,
                "device": conv.device,
                "vram_mb": conv.vram_mb(),
            }
            await ws.send(json.dumps(welcome))
            if welcome["reference_needed"]:
                prepared = await self.receive_reference(ws, conv)
                if prepared is None:
                    return
                if cache_key:
                    self.references[cache_key] = prepared
                    while len(self.references) > REFERENCE_CACHE:
                        self.references.popitem(last=False)
                await self.run_blocking(stream.set_reference, prepared)
                await ws.send(json.dumps({"type": "reference_ready"}))
            elif cached is not None:
                self.references.move_to_end(cache_key)
                await self.run_blocking(stream.set_reference, cached)
            LOG.info("session: %d -> %d Hz, %d-sample frames, timeout %.0f ms", rate, out_rate, frame_in, timeout_s * 1000)
            await self.stream_v2(ws, sess, stream, frame_in, frame_out, timeout_s)
        finally:
            self.release(sess)
            await self.run_blocking(stream.close)

    async def receive_reference(self, ws, conv: Converter):
        meta = None
        chunks = []
        total = 0
        limit = int(conv.max_reference_seconds * conv.reference_rate) + 1
        while True:
            msg = await asyncio.wait_for(ws.recv(), 60)
            if isinstance(msg, str):
                m = json.loads(msg)
                if m.get("type") == "reference":
                    meta = m
                continue
            kind, flags, _seq, _pitch, pcm = decode_frame(msg)
            if kind != KIND_REFERENCE:
                continue
            total += pcm.size
            if total > limit:
                await ws.send(json.dumps({"type": "error", "message": "the reference voice is too long", "fatal": False}))
                return None
            chunks.append(pcm)
            if flags & FLAG_LAST:
                break
        pcm = np.concatenate(chunks) if chunks else np.zeros(0, np.float32)
        rate = int((meta or {}).get("sample_rate") or conv.reference_rate)

        def prepare():
            # Off the event loop (a long clip's resampling too): other sessions keep streaming.
            x = resample(pcm, rate, conv.reference_rate) if rate != conv.reference_rate else pcm
            return conv.prepare_reference(x, conv.reference_rate, meta or {})

        t0 = time.perf_counter()
        try:
            prepared = await self.run_reference(prepare)
        except Exception as e:  # noqa: BLE001
            LOG.exception("reference preparation failed")
            await ws.send(json.dumps({"type": "error", "message": f"cannot use the reference voice: {e}"}))
            return None
        LOG.info("reference voice ready (%.1f s, %.0f ms)", pcm.size / max(1, rate), (time.perf_counter() - t0) * 1000)
        return prepared

    async def stream_v2(self, ws, sess: _Session, stream: Stream, frame_in: int, frame_out: int, timeout_s: float) -> None:
        queue: asyncio.Queue = asyncio.Queue()
        state = sess
        single = bool(self.conv.single_stream)
        delay_s = float(getattr(self.conv, "reply_delay_s", 0.0) or 0.0)
        pending = set()

        async def send(data):
            try:
                await ws.send(data)
            except Exception:  # noqa: BLE001
                pass

        async def send_later(at: float, data: bytes):
            await asyncio.sleep(max(0.0, at - time.monotonic()))
            await send(data)

        async def reader():
            try:
                await read_messages()
            except Exception as e:  # noqa: BLE001  (a dropped connection ends the session)
                if not _is_closed(e):
                    LOG.warning("session read: %s", e)
            finally:
                queue.put_nowait(None)

        async def read_messages():
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    try:
                        kind, flags, seq, pitch, pcm = decode_frame(bytes(msg))
                    except ValueError as e:
                        await send(json.dumps({"type": "error", "message": str(e), "fatal": True}))
                        return
                    if kind != KIND_AUDIO_IN:
                        continue
                    if pcm.size != frame_in:
                        await send(json.dumps({"type": "error", "seq": seq,
                                               "message": f"frame of {pcm.size} samples, expected {frame_in}"}))
                        continue
                    state.last_audio = time.monotonic()
                    queue.put_nowait(("audio", seq, flags, pitch, pcm, state.last_audio))
                else:
                    try:
                        m = json.loads(msg)
                    except ValueError:
                        continue
                    if m.get("type") == "params" and isinstance(m.get("params"), dict):
                        queue.put_nowait(("params", m["params"]))
                    elif m.get("type") == "cancel":
                        before = int(m.get("before_seq") or 0) & 0xFFFFFFFF
                        if state.cancel_before is None or seq_before(state.cancel_before, before):
                            state.cancel_before = before

        async def processor():
            discontinuity = False
            # Compute per frame: a frame that cannot be back before its deadline is dropped
            # unconverted, so an overloaded model spends its time on frames that still count. The
            # estimate is the median of the recent frames (one slow frame, such as a first call that
            # imports or autotunes, does not drop the ones behind it), starting from the warm-up's.
            # (`--echo-delay-ms` is a reply delay, not compute.)
            recent = collections.deque(maxlen=9)
            warm_s = max(0.0, float(self.conv.latency_ms or 0.0) / 1000.0 - delay_s)
            dropped = converted = superseded = 0
            while True:
                item = await queue.get()
                if item is None:
                    if dropped or superseded:
                        LOG.info("session: %d frames converted, %d dropped as stale, %d while another voice held the model",
                                 converted, dropped, superseded)
                    return
                if item[0] == "params":
                    await self.run_blocking(stream.set_params, item[1])
                    continue
                _, seq, flags, pitch, pcm, received = item
                restart = bool(flags & FLAG_DISCONTINUITY)
                if state.cancel_before is not None and seq_before(seq, state.cancel_before):
                    discontinuity = True
                    state.restart_pending |= restart
                    continue
                compute_s = float(np.median(recent)) if recent else warm_s
                if timeout_s > 0 and time.monotonic() - received + compute_s > timeout_s:
                    discontinuity = True
                    state.restart_pending |= restart
                    dropped += 1
                    await send(json.dumps({"type": "dropped", "seq": seq, "reason": DROP_STALE}))
                    continue
                if single:
                    # One stream holds the model: a session takes it over when its stream (re)starts
                    # (Sigil flags the first frame of a voice that starts or is switched to) or when
                    # the holder has gone quiet; the others' frames are dropped meanwhile (the voice
                    # being replaced, for the moment before the switch).
                    claim = self.claim(state, restart or state.restart_pending)
                    if claim is None:
                        superseded += 1
                        await send(json.dumps({"type": "dropped", "seq": seq, "reason": DROP_SUPERSEDED}))
                        continue
                    if claim:
                        try:
                            await self.run_blocking(stream.activate)
                        except Exception as e:  # noqa: BLE001
                            LOG.exception("taking the model over failed")
                            self.release(state)
                            await send(json.dumps({"type": "error", "seq": seq, "message": str(e)[:200]}))
                            continue
                        discontinuity = True
                t0 = time.monotonic()
                try:
                    out = await self.run_blocking(stream.process, pcm, float(pitch), restart or discontinuity)
                except Exception as e:  # noqa: BLE001
                    LOG.exception("frame %d failed", seq)
                    await send(json.dumps({"type": "error", "seq": seq, "message": str(e)[:200]}))
                    discontinuity = True
                    continue
                recent.append(time.monotonic() - t0)
                converted += 1
                discontinuity = False
                state.restart_pending = False
                data = encode_frame(KIND_AUDIO_OUT, 0, seq, fit_length(out, frame_out))
                if delay_s > 0:
                    task = asyncio.ensure_future(send_later(received + delay_s, data))
                    pending.add(task)
                    task.add_done_callback(pending.discard)
                else:
                    await send(data)

        reader_task = asyncio.ensure_future(reader())
        try:
            await processor()
        finally:
            reader_task.cancel()
            for t in list(pending):
                t.cancel()

    async def serve_v1(self, ws, hello: dict) -> None:
        conv = self.conv
        if conv.reference == "required":
            await self.refuse(ws, f"{conv.name} converts toward a reference voice: use Sigil's sidecar voice (protocol v2)")
            return
        rate = int(hello.get("sample_rate_in") or 48000)
        ch = max(1, int(hello.get("channels_in") or 1))
        chunk = int(hello.get("chunk_frames") or 512)
        if conv.v1_passthrough:
            await self.serve_v1_passthrough(ws, rate, ch, chunk)
            return
        if np is None:
            await self.refuse(ws, "this sidecar needs numpy (pip install numpy)")
            return
        model_rate = conv.rate_for([rate]) or conv.input_rates[0]
        out_rate = conv.output_rate_for(model_rate)
        frame_in = conv.frame_samples(model_rate, max(1, int(round(chunk * model_rate / rate))))
        frame_out = frame_in * out_rate // model_rate
        stream = await self.run_blocking(conv.open_stream, model_rate, frame_in)
        # Same-length replies: whatever the frame, the output lags by one frame through the FIFO.
        same = model_rate == rate and out_rate == rate and frame_in == chunk
        fifo_in = np.zeros(0, np.float32)
        fifo_out = OutputFifo(0 if same else frame_in * rate // model_rate)
        await ws.send(json.dumps({"ok": True, "mode": conv.name}))
        LOG.info("v1 session: %d Hz x %d channels, %d-frame chunks", rate, ch, chunk)
        try:
            async for msg in ws:
                if not isinstance(msg, (bytes, bytearray)):
                    continue
                try:
                    x = decode_v1(bytes(msg))
                except ValueError:
                    continue
                frames = x.size // ch
                mono = x[: frames * ch].reshape(frames, ch).mean(axis=1) if ch > 1 else x
                fifo_in = np.concatenate([fifo_in, resample(mono, rate, model_rate)])
                while fifo_in.size >= frame_in:
                    block, fifo_in = fifo_in[:frame_in], fifo_in[frame_in:]
                    out = await self.run_blocking(stream.process, block, 0.0, False)
                    fifo_out.push(resample(fit_length(out, frame_out), out_rate, rate))
                reply = fifo_out.pop(frames)
                if ch > 1:
                    reply = np.repeat(reply, ch)
                await ws.send(encode_v1(reply))
        finally:
            await self.run_blocking(stream.close)

    async def serve_v1_passthrough(self, ws, rate: int, ch: int, chunk: int) -> None:
        """v1 echo: every packet comes back as it came (any rate, interleaved channels kept, no
        added delay), after ``--echo-compute-ms`` / ``--echo-delay-ms`` when set."""
        conv = self.conv
        wait_s = float(getattr(conv, "compute_s", 0.0) or 0.0) + float(getattr(conv, "reply_delay_s", 0.0) or 0.0)
        await ws.send(json.dumps({"ok": True, "mode": conv.name}))
        LOG.info("v1 session: %d Hz x %d channels, %d-frame chunks (pass-through)", rate, ch, chunk)
        async for msg in ws:
            if not isinstance(msg, (bytes, bytearray)):
                continue
            try:
                packet = v1_packet(bytes(msg))
            except ValueError:
                continue
            if wait_s > 0:
                await asyncio.sleep(wait_s)
            await ws.send(packet)


def _is_closed(e: BaseException) -> bool:
    return type(e).__name__ in ("ConnectionClosed", "ConnectionClosedOK", "ConnectionClosedError", "CancelledError")


def _quiet_known_noise(loop, context) -> None:
    """websockets' legacy server (12 and older) leaves a failed close task behind when it refuses an
    upgrade over HTTP (a wrong key); that traceback is noise. Everything else is reported as usual."""
    e = context.get("exception")
    if isinstance(e, AttributeError) and "transfer_data_task" in str(e):
        LOG.debug("ignored websockets close noise after a refused upgrade")
        return
    loop.default_exception_handler(context)


# ---------------------------------------------------------------------------------------------
# Self test and entry point
# ---------------------------------------------------------------------------------------------

def synthetic_voice(seconds: float, rate: int, f0: float = 140.0) -> np.ndarray:
    """A voiced test signal: a band-limited pulse train with vibrato and a syllable envelope."""
    t = np.arange(int(seconds * rate)) / rate
    f = f0 * (1.0 + 0.03 * np.sin(2 * np.pi * 5.0 * t))
    phase = 2 * np.pi * np.cumsum(f) / rate
    x = sum(np.sin(k * phase) / k for k in range(1, 12) if k * f0 < rate / 2.2)
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3.0 * t) ** 2
    return (0.15 * x * env).astype(np.float32)


def self_test(conv: Converter, reference: Optional[str], seconds: float = 4.0) -> int:
    """Convert a synthetic phrase frame by frame and print timings (JSON)."""
    rate = (conv.input_rates or (16000,))[0]
    out_rate = conv.output_rate_for(rate)
    frame_in = conv.frame_samples(rate, int(0.1 * rate))
    stream = conv.open_stream(rate, frame_in)
    if conv.reference != "none":
        if reference:
            ref, ref_rate = read_wav_mono(reference)
            ref = resample(ref, ref_rate, conv.reference_rate)
        else:
            ref = synthetic_voice(6.0, conv.reference_rate, 210.0)
        t0 = time.perf_counter()
        stream.set_reference(conv.prepare_reference(ref, conv.reference_rate, {"sha256": "self-test"}))
        ref_ms = (time.perf_counter() - t0) * 1000.0
    else:
        ref_ms = 0.0
    stream.activate()
    src = synthetic_voice(seconds, rate)
    times, outs = [], []
    for i in range(0, src.size - frame_in + 1, frame_in):
        t0 = time.perf_counter()
        outs.append(fit_length(stream.process(src[i:i + frame_in], 0.0, i == 0), frame_in * out_rate // rate))
        times.append((time.perf_counter() - t0) * 1000.0)
    stream.close()
    out = np.concatenate(outs) if outs else np.zeros(0, np.float32)
    frame_ms = frame_in * 1000.0 / rate
    report = {
        "model": conv.name, "device": conv.device, "rate_in": rate, "rate_out": out_rate, "frame_ms": round(frame_ms, 2),
        "frames": len(times), "reference_ms": round(ref_ms, 1),
        "ms_p50": round(float(np.percentile(times, 50)), 2) if times else None,
        "ms_p95": round(float(np.percentile(times, 95)), 2) if times else None,
        "rtf_p95": round(float(np.percentile(times, 95)) / frame_ms, 3) if times else None,
        "delay_ms": round(conv.delay_samples * 1000.0 / out_rate, 1),
        "output_rms_db": round(rms_db(out), 1), "finite": bool(np.isfinite(out).all()), "vram_mb": conv.vram_mb(),
    }
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["finite"] and report["frames"] > 0 else 1


def build_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--host", default="127.0.0.1", help="bind address (this PC only: 127.0.0.1, localhost or ::1)")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--device", default="cuda", help="cuda, dml or cpu (what Sigil's voice changer runs on)")
    p.add_argument("--model-dir", default="", help="the model / checkpoint folder")
    p.add_argument("--frame-ms", type=float, default=None, help="override the model's frame")
    p.add_argument("--echo", action="store_true", help="no model: pass audio through (protocol and wiring tests)")
    p.add_argument("--echo-delay-ms", type=float, default=0.0, help="--echo: answer each frame this much later")
    p.add_argument("--echo-compute-ms", type=float, default=0.0, help="--echo: spend this long per frame")
    p.add_argument("--echo-out-rate", type=int, default=0, help="--echo: answer at this rate")
    p.add_argument("--echo-single-stream", action="store_true",
                   help="--echo: one streaming state for every session, as StreamVoiceAnon (tests voice switches)")
    p.add_argument("--self-test", nargs="?", const="", default=None, metavar="REFERENCE_WAV",
                   help="convert a synthetic phrase (toward REFERENCE_WAV) and print timings, then exit")
    p.add_argument("--log-level", default="INFO")
    return p


def main(make_converter: Callable[[argparse.Namespace], Converter], description: str,
         add_arguments: Optional[Callable[[argparse.ArgumentParser], None]] = None, argv=None,
         echo_reference: str = "optional") -> None:
    parser = build_parser(description)
    if add_arguments:
        add_arguments(parser)
    args = parser.parse_args(argv)
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", stream=sys.stdout)
    if not is_loopback(args.host):
        parser.error("Sigil sidecars listen on this PC only (127.0.0.1, localhost or ::1)")
    if np is None and (not args.echo or args.self_test is not None):
        parser.error("this sidecar needs numpy (pip install numpy); only the protocol v1 echo runs without it")
    if np is None:
        LOG.warning("numpy is not installed: protocol v2 sessions are refused (the v1 echo works)")
    conv = EchoConverter(args, echo_reference) if args.echo else make_converter(args)
    t0 = time.perf_counter()
    conv.load()
    conv.warmup()
    LOG.info("%s loaded in %.1f s on %s (%.1f ms per frame)", conv.name, time.perf_counter() - t0, conv.device,
             conv.latency_ms or 0.0)
    if args.self_test is not None:
        sys.exit(self_test(conv, args.self_test or None))
    try:
        asyncio.run(Server(conv).run(args.host, args.port))
    except KeyboardInterrupt:
        pass
