#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests of the shared sidecar server (``sigil_vc_v2.py``), protocol v2 and v1, against a real
socket with the echo converter. Needs ``numpy`` and ``websockets``; no model, no network beyond
127.0.0.1.

Run: ``python scripts/neural-sidecar/test_sigil_vc_v2.py`` (or ``python -m unittest`` from this folder).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import threading
import time
import unittest
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sigil_vc_v2 as v2  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def echo(cls=None, **kw) -> v2.EchoConverter:
    args = argparse.Namespace(echo_delay_ms=kw.get("delay_ms", 0), echo_compute_ms=kw.get("compute_ms", 0),
                              echo_out_rate=kw.get("out_rate", 0), frame_ms=kw.get("frame_ms"))
    conv = (cls or v2.EchoConverter)(args, kw.get("reference", "optional"), single_stream=kw.get("single_stream", False))
    conv.warmup()
    return conv


class Running:
    """A server on a free port in a thread of its own (stopped with the context)."""

    def __init__(self, conv: v2.Converter, key: str | None = None):
        self.conv, self.key, self.port = conv, key, free_port()
        self.loop = asyncio.new_event_loop()
        self.stop = None

    def __enter__(self):
        if self.key:
            os.environ[v2.KEY_ENV] = self.key
        else:
            os.environ.pop(v2.KEY_ENV, None)
        server = v2.Server(self.conv)
        os.environ.pop(v2.KEY_ENV, None)
        ready = threading.Event()

        async def main():
            self.stop = asyncio.Event()
            asyncio.get_running_loop().set_exception_handler(v2._quiet_known_noise)  # as Server.run does
            async with server.serve("127.0.0.1", self.port):
                ready.set()
                await self.stop.wait()

        self.thread = threading.Thread(target=lambda: self.loop.run_until_complete(main()), daemon=True)
        self.thread.start()
        assert ready.wait(10), "server did not start"
        return self

    def __exit__(self, *exc):
        self.loop.call_soon_threadsafe(self.stop.set)
        self.thread.join(10)
        if not self.thread.is_alive():
            self.loop.close()

    @property
    def url(self) -> str:
        return f"ws://127.0.0.1:{self.port}/vc"


def connect(url: str, key: str | None = None):
    headers = {"Authorization": f"Bearer {key}"} if key else None
    try:
        from websockets.asyncio.client import connect as connect_new  # websockets >= 13

        return connect_new(url, additional_headers=headers, max_size=v2.MAX_MESSAGE_BYTES)
    except ImportError:
        import websockets.client as legacy

        return legacy.connect(url, extra_headers=headers, max_size=v2.MAX_MESSAGE_BYTES)


def hello(**extra) -> str:
    h = {"type": v2.HELLO_TYPE, "protocol_version": 2, "sample_rate_in": 16000, "channels_in": 1, "chunk_frames": 1600,
         "sample_rates": [16000, 22050, 48000], "frame_ms": 100.0, "frame_timeout_ms": 1000, "max_in_flight": 8,
         "params": {}, "reference": None}
    h.update(extra)
    return json.dumps(h)


def frame(seq: int, n: int, value: float = 0.25, flags: int = 0, pitch: float = 0.0) -> bytes:
    return v2.encode_frame(v2.KIND_AUDIO_IN, flags, seq, np.full(n, value, np.float32), pitch)


async def recv_until(ws, done, timeout: float = 5.0):
    """Messages until ``done(messages)`` (or the timeout)."""
    got = []
    end = time.monotonic() + timeout
    while not done(got):
        left = end - time.monotonic()
        if left <= 0:
            break
        try:
            got.append(await asyncio.wait_for(ws.recv(), left))
        except asyncio.TimeoutError:
            break
    return got


def replies(messages):
    out = {}
    for m in messages:
        if isinstance(m, (bytes, bytearray)):
            kind, _flags, seq, _pitch, pcm = v2.decode_frame(bytes(m))
            if kind == v2.KIND_AUDIO_OUT:
                out[seq] = pcm
    return out


def controls(messages, kind):
    return [json.loads(m) for m in messages if isinstance(m, str) and json.loads(m).get("type") == kind]


def run(coro):
    return asyncio.run(coro)


class WireFormat(unittest.TestCase):
    def test_frames_round_trip_and_bad_frames_are_refused(self):
        data = v2.encode_frame(v2.KIND_AUDIO_IN, v2.FLAG_DISCONTINUITY, 0xDEADBEEF, np.array([0.25, -1.0], np.float32), -3.5)
        self.assertEqual(len(data), 20 + 8)
        self.assertEqual(data[:4], b"SGV2")
        kind, flags, seq, pitch, pcm = v2.decode_frame(data)
        self.assertEqual((kind, flags, seq, pitch), (1, 1, 0xDEADBEEF, -3.5))
        self.assertEqual(pcm.tolist(), [0.25, -1.0])
        with self.assertRaises(ValueError):
            v2.decode_frame(data[:-1])
        with self.assertRaises(ValueError):
            v2.decode_frame(b"XGV2" + data[4:])

    def test_sequence_numbers_wrap(self):
        self.assertTrue(v2.seq_before(1, 2) and not v2.seq_before(2, 1) and not v2.seq_before(3, 3))
        self.assertTrue(v2.seq_before(0xFFFFFFFF, 2))


class ProtocolV2(unittest.TestCase):
    def test_welcome_rates_frame_and_replies_by_sequence(self):
        with Running(echo(out_rate=22050)) as srv:
            async def go():
                async with connect(srv.url) as ws:
                    await ws.send(hello(params={"speaker_id": 2}))
                    w = json.loads(await ws.recv())
                    self.assertEqual((w["type"], w["protocol_version"]), (v2.WELCOME_TYPE, 2))
                    self.assertEqual((w["sample_rate_in"], w["sample_rate_out"], w["frame_samples"]), (16000, 22050, 1600))
                    for seq in range(3):
                        await ws.send(frame(seq, 1600, flags=v2.FLAG_DISCONTINUITY if seq == 0 else 0, pitch=2.0))
                    await ws.send(json.dumps({"type": "params", "params": {"speaker_id": 3}}))
                    got = replies(await recv_until(ws, lambda m: len(replies(m)) == 3))
                    self.assertEqual(sorted(got), [0, 1, 2])
                    self.assertTrue(all(p.size == 2205 for p in got.values()))
                    # A frame of the wrong length is refused for that frame only.
                    await ws.send(frame(3, 1000))
                    errs = controls(await recv_until(ws, lambda m: bool(controls(m, "error"))), "error")
                    self.assertEqual(errs[0]["seq"], 3)
            run(go())

    def test_the_key_is_required_when_set(self):
        with Running(echo(), key="k3y") as srv:
            async def go():
                with self.assertRaises(Exception):
                    async with connect(srv.url) as ws:
                        await ws.send(hello())
                        await ws.recv()
                async with connect(srv.url, key="k3y") as ws:
                    await ws.send(hello())
                    self.assertEqual(json.loads(await ws.recv())["type"], v2.WELCOME_TYPE)
            run(go())
            # Health needs no key.
            with urllib.request.urlopen(f"http://127.0.0.1:{srv.port}/health", timeout=5) as r:
                self.assertEqual(r.status, 200)

    def test_reference_is_uploaded_once_then_cached(self):
        with Running(echo(reference="required")) as srv:
            meta = {"voice_id": "v1", "name": "V", "sha256": "abc", "duration_ms": 2000}

            async def session(expect_upload: bool):
                async with connect(srv.url) as ws:
                    await ws.send(hello(reference=meta))
                    w = json.loads(await ws.recv())
                    self.assertTrue(w["reference_required"])
                    self.assertEqual((w["reference_needed"], w["reference_cached"]), (expect_upload, not expect_upload))
                    if expect_upload:
                        rate = w["reference_sample_rate"]
                        pcm = np.zeros(rate * 2, np.float32)
                        await ws.send(json.dumps({"type": "reference", **meta, "sample_rate": rate, "samples": pcm.size}))
                        chunks = [pcm[i:i + 20000] for i in range(0, pcm.size, 20000)]
                        for i, c in enumerate(chunks):
                            last = v2.FLAG_LAST if i == len(chunks) - 1 else 0
                            await ws.send(v2.encode_frame(v2.KIND_REFERENCE, last, i, c))
                        self.assertEqual(json.loads(await ws.recv())["type"], "reference_ready")
                    await ws.send(frame(0, w["frame_samples"]))
                    self.assertEqual(list(replies(await recv_until(ws, lambda m: bool(replies(m))))), [0])

            run(session(True))
            run(session(False))

            async def without():
                async with connect(srv.url) as ws:
                    await ws.send(hello())
                    m = json.loads(await ws.recv())
                    self.assertIs(m["ok"], False)
                    self.assertIn("reference", m["error"])
            run(without())

    def test_stale_frames_are_dropped_unconverted(self):
        # 200 ms of compute per 100 ms frame, a 300 ms deadline: the queue cannot keep up.
        with Running(echo(compute_ms=200)) as srv:
            async def go():
                async with connect(srv.url) as ws:
                    await ws.send(hello(frame_timeout_ms=300))
                    await ws.recv()
                    for seq in range(6):
                        await ws.send(frame(seq, 1600))
                    got = await recv_until(ws, lambda m: len(replies(m)) + len(controls(m, "dropped")) == 6, 5)
                    answered, dropped = set(replies(got)), {d["seq"] for d in controls(got, "dropped")}
                    self.assertEqual(answered | dropped, set(range(6)), "every frame is answered or dropped")
                    self.assertTrue(dropped and 0 in answered, (answered, dropped))
            run(go())

    def test_cancelled_frames_are_not_converted(self):
        with Running(echo(compute_ms=100)) as srv:
            async def go():
                async with connect(srv.url) as ws:
                    await ws.send(hello(frame_timeout_ms=5000))
                    await ws.recv()
                    for seq in range(8):
                        await ws.send(frame(seq, 1600))
                    await ws.send(json.dumps({"type": "cancel", "before_seq": 6}))
                    got = replies(await recv_until(ws, lambda m: 7 in replies(m), 5))
                    self.assertIn(6, got)
                    self.assertLess(len(got), 8, "some cancelled frames were skipped")
            run(go())

    def test_a_reference_being_prepared_does_not_hold_up_another_sessions_frames(self):
        """Voice B loads (its reference takes 1 s to prepare) while voice A plays with a 400 ms
        deadline: A's frames keep converting meanwhile (none goes stale)."""

        class SlowReference(v2.EchoConverter):
            def prepare_reference(self, pcm, rate, meta):
                time.sleep(1.0)
                return pcm

        with Running(echo(SlowReference)) as srv:
            async def go():
                async with connect(srv.url) as a, connect(srv.url) as b:
                    await a.send(hello(frame_timeout_ms=400))
                    await a.recv()

                    async def play():
                        for seq in range(15):
                            await a.send(frame(seq, 1600))
                            await asyncio.sleep(0.1)
                        return await recv_until(a, lambda m: len(replies(m)) + len(controls(m, "dropped")) == 15, 5)

                    async def load():
                        await asyncio.sleep(0.25)
                        meta = {"voice_id": "b", "name": "B", "sha256": "slow", "duration_ms": 1000}
                        await b.send(hello(reference=meta))
                        w = json.loads(await b.recv())
                        self.assertTrue(w["reference_needed"])
                        await b.send(json.dumps({"type": "reference", **meta, "sample_rate": 16000, "samples": 16000}))
                        await b.send(v2.encode_frame(v2.KIND_REFERENCE, v2.FLAG_LAST, 0, np.zeros(16000, np.float32)))
                        t0 = time.monotonic()
                        self.assertEqual(json.loads(await b.recv())["type"], "reference_ready")
                        return time.monotonic() - t0

                    got, prep_s = await asyncio.gather(play(), load())
                    self.assertGreater(prep_s, 0.9, "the reference took its second")
                    self.assertEqual(controls(got, "dropped"), [], "no frame of the playing voice went stale")
                    self.assertEqual(sorted(replies(got)), list(range(15)))
            run(go())


class SingleStream(unittest.TestCase):
    """A model with one streaming state (StreamVoiceAnon): every session is served, one converts."""

    def test_a_second_session_is_served_and_takes_over_when_its_stream_starts(self):
        conv = echo(single_stream=True)
        old_idle = v2.OWNER_IDLE_S
        v2.OWNER_IDLE_S = 0.5
        try:
            with Running(conv) as srv:
                async def answer(ws, seq, flags=0):
                    await ws.send(frame(seq, 320, flags=flags))
                    got = await recv_until(ws, lambda m: seq in replies(m) or any(d["seq"] == seq for d in controls(m, "dropped")))
                    dropped = [d for d in controls(got, "dropped") if d["seq"] == seq]
                    return dropped[0]["reason"] if dropped else "converted"

                async def go():
                    async with connect(srv.url) as a:
                        await a.send(hello(frame_ms=20.0))
                        self.assertEqual(json.loads(await a.recv())["type"], v2.WELCOME_TYPE)
                        self.assertEqual(await answer(a, 0, v2.FLAG_DISCONTINUITY), "converted")
                        self.assertEqual(conv.activations, 1)
                        # The replacement voice connects while A plays: served, not refused.
                        async with connect(srv.url) as b:
                            await b.send(hello(frame_ms=20.0))
                            self.assertEqual(json.loads(await b.recv())["type"], v2.WELCOME_TYPE)
                            self.assertEqual(await answer(a, 1), "converted", "connecting takes nothing over")
                            # B's stream starts (the switch): B takes the model, A is superseded.
                            self.assertEqual(await answer(b, 0, v2.FLAG_DISCONTINUITY), "converted")
                            self.assertEqual(conv.activations, 2)
                            self.assertEqual(await answer(a, 2), v2.DROP_SUPERSEDED)
                            self.assertEqual(await answer(b, 1), "converted")
                            # B goes quiet (an abandoned switch): A gets the model back by itself.
                            await asyncio.sleep(0.7)
                            self.assertEqual(await answer(a, 3), "converted")
                            self.assertEqual(conv.activations, 3)
                            self.assertEqual(await answer(b, 2), v2.DROP_SUPERSEDED)
                        # A closed holder frees the model at once.
                        async with connect(srv.url) as c:
                            await c.send(hello(frame_ms=20.0))
                            await c.recv()
                            self.assertEqual(await answer(c, 0), v2.DROP_SUPERSEDED, "A still plays")
                    await asyncio.sleep(0.1)  # the server notices A's close
                    async with connect(srv.url) as c:
                        await c.send(hello(frame_ms=20.0))
                        await c.recv()
                        self.assertEqual(await answer(c, 0), "converted", "nobody holds the model")
                run(go())
        finally:
            v2.OWNER_IDLE_S = old_idle


class ModelStreams(unittest.TestCase):
    """The scripts' streams forget what came before a DISCONTINUITY (no stale audio after a pause),
    and StreamVoiceAnon's streams share its one wrapper correctly."""

    @staticmethod
    def loud(n, k):
        t = (np.arange(n) + k * n) / 16000.0
        return (0.5 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    def test_seedvc_resets_its_window_and_tail_on_a_discontinuity(self):
        import seedvc_sidecar as sv

        class Conv:
            block_units, sola_buffer_units, sola_search_units = 5, 2, 1
            shapes = {"n16": 40 * sv.UNIT_16K, "n_mel": 4}
            gate_db = None  # no silence gate: the window decides
            preset = None

            def __init__(self):
                units = self.block_units + self.sola_buffer_units + self.sola_search_units

                class Zs:
                    # The window's newest `units` at 22.05 kHz (an echo of what the tick sees).
                    @staticmethod
                    def tick(ref, window, preset, z, harm, noise, use_step_graph, hift_mode):
                        src = window[-units * sv.UNIT_16K:]
                        t = np.linspace(0, src.size - 1, units * sv.UNIT_22K)
                        return np.interp(t, np.arange(src.size), src).astype(np.float32), None

                self.zs = Zs()

        def last_reply(discontinuity):
            s = sv.SeedVcStream(Conv(), 1)
            s.set_reference({"prompt_mel": np.zeros((1, 80, 3), np.float32)})
            for k in range(6):
                s.process(self.loud(1600, k), 0.0, k == 0)
            return s.process(np.zeros(1600, np.float32), 0.0, discontinuity)

        self.assertGreater(float(np.abs(last_reply(False)).max()), 0.1, "the test sees the window's old audio")
        self.assertLess(float(np.abs(last_reply(True)).max()), 1e-6, "nothing from before the gap plays")

    def test_xvc_resets_its_window_and_overlap_on_a_discontinuity(self):
        import xvc_sidecar as xs

        class Conv:
            current, smooth, future = 1920, 320, 1600
            window = 6000
            cur_start = window - current - smooth - future
            cur_end = cur_start + current

            @staticmethod
            def convert_window(w, cond):
                return w.copy()

        def last_reply(discontinuity):
            s = xs.XvcStream(Conv())
            s.set_reference(("speaker", "frames"))
            for k in range(6):
                s.process(self.loud(1920, k), 0.0, k == 0)
            return s.process(np.zeros(1920, np.float32), 0.0, discontinuity)

        self.assertGreater(float(np.abs(last_reply(False)).max()), 0.1, "the test sees the lookahead's old audio")
        self.assertLess(float(np.abs(last_reply(True)).max()), 1e-6, "nothing from before the gap plays")

    def test_streamvoiceanon_streams_share_the_wrapper(self):
        import streamvoiceanon_sidecar as sva

        class Conv(sva.SvaConverter):
            def __init__(self):  # no model: the wrapper calls are recorded
                self.alpha, self.reset_on_gap, self.active = 1.0, True, None
                self.delay_samples, self.frame_out, self.log = 4, 8, []

            def apply_reference(self, pcm, alpha):
                self.log.append(("reference", pcm, alpha))

            def reset_upstream(self):
                self.log.append(("reset",))

            def convert_chunk(self, pcm, pitch):
                return pcm

        c = Conv()
        a, b = c.open_stream(22050, 8), c.open_stream(22050, 8)
        a.set_reference("A")
        b.set_reference("B")
        self.assertEqual(c.log, [], "connecting voices leave the wrapper alone")
        out = a.process(np.ones(8, np.float32), 0.0, True)
        self.assertEqual(c.log, [("reference", "A", 1.0), ("reset",)], "the first frame takes the wrapper")
        self.assertEqual(out.tolist(), [0.0] * 4 + [1.0] * 4, "the model's delay")
        b.activate()
        self.assertIs(c.active, b)
        self.assertEqual(c.log[-2:], [("reference", "B", 1.0), ("reset",)])
        n = len(c.log)
        a.close()
        self.assertEqual(len(c.log), n, "closing a voice that does not hold the wrapper leaves it alone")
        b.set_params({"alpha": 0.5})
        self.assertEqual(c.log[-1], ("reference", "B", 0.5), "alpha applies to the holder at once")
        b.process(np.ones(8, np.float32), 0.0, False)
        out = b.process(np.full(8, 2.0, np.float32), 0.0, True)
        self.assertEqual(c.log[-1], ("reset",))
        self.assertEqual(out.tolist(), [0.0] * 4 + [2.0] * 4, "the delay line restarts: no audio from before the gap")
        b.close()
        self.assertIsNone(c.active)


class ProtocolV1(unittest.TestCase):
    def test_v1_clients_get_same_length_replies(self):
        with Running(echo(frame_ms=20.0)) as srv:
            async def go():
                async with connect(srv.url) as ws:
                    await ws.send(json.dumps({"type": v2.HELLO_TYPE, "protocol_version": 1, "sample_rate_in": 48000,
                                              "channels_in": 2, "chunk_frames": 960}))
                    self.assertIs(json.loads(await ws.recv())["ok"], True)
                    for _ in range(4):
                        await ws.send(v2.encode_v1(np.full(1920, 0.5, np.float32)))
                        out = v2.decode_v1(await asyncio.wait_for(ws.recv(), 5))
                        self.assertEqual(out.size, 1920)
            run(go())

    def test_the_v1_echo_returns_every_packet_untouched(self):
        # A 96 kHz interface with two different channels: the echo is transparent and immediate.
        with Running(echo()) as srv:
            async def go():
                async with connect(srv.url) as ws:
                    await ws.send(json.dumps({"type": v2.HELLO_TYPE, "protocol_version": 1, "sample_rate_in": 96000,
                                              "channels_in": 2, "chunk_frames": 512}))
                    self.assertIs(json.loads(await ws.recv())["ok"], True)
                    rng = np.random.default_rng(3)
                    for _ in range(3):
                        pcm = rng.uniform(-1, 1, 1024).astype(np.float32)  # L and R differ
                        packet = v2.encode_v1(pcm)
                        await ws.send(packet)
                        self.assertEqual(await asyncio.wait_for(ws.recv(), 5), packet)
            run(go())

    def test_a_model_serves_v1_at_its_own_rate(self):
        # A model (not the echo) still converts v1 packets: resampled to its rate and back.
        class Model(v2.EchoConverter):
            input_rates = (16000,)
            v1_passthrough = False

        with Running(echo(Model, frame_ms=20.0)) as srv:
            async def go():
                async with connect(srv.url) as ws:
                    await ws.send(json.dumps({"type": v2.HELLO_TYPE, "protocol_version": 1, "sample_rate_in": 48000,
                                              "channels_in": 1, "chunk_frames": 960}))
                    self.assertIs(json.loads(await ws.recv())["ok"], True)
                    for _ in range(3):
                        await ws.send(v2.encode_v1(np.full(960, 0.5, np.float32)))
                        self.assertEqual(v2.decode_v1(await asyncio.wait_for(ws.recv(), 5)).size, 960)
            run(go())

    def test_a_reference_model_refuses_v1(self):
        with Running(echo(reference="required")) as srv:
            async def go():
                async with connect(srv.url) as ws:
                    await ws.send(json.dumps({"type": v2.HELLO_TYPE, "protocol_version": 1, "sample_rate_in": 48000,
                                              "channels_in": 1, "chunk_frames": 480}))
                    self.assertIs(json.loads(await ws.recv())["ok"], False)
            run(go())


class SeedVcRouting(unittest.TestCase):
    """``seedvc_sidecar.Routed``: per-tick shapes go to the pinned session, other shapes to an
    unpinned one made on first use (on the device asked for); a refused pin falls back."""

    class FakeInput:
        def __init__(self, name, shape):
            self.name, self.shape = name, shape

    class FakeSession:
        def __init__(self, tag, inputs):
            self.tag, self.inputs, self.calls = tag, inputs, 0

        def get_inputs(self):
            return self.inputs

        def get_providers(self):
            return ["DmlExecutionProvider"]

        def run(self, outputs, feeds):
            self.calls += 1
            return [self.tag]

    def test_shapes_route_between_pinned_and_unpinned_sessions(self):
        import seedvc_sidecar as sv

        made = []

        def make(path, dims=None, device=None):
            # ORT may still declare a pinned dimension by its symbolic name.
            s = self.FakeSession(("pinned" if dims else "other", device), [self.FakeInput("wav16k", [1, "N"])])
            made.append(s)
            return s

        r = sv.Routed(make, "xlsr.onnx", {"N": 640}, "cpu")
        self.assertEqual(len(made), 1, "the unpinned session waits for its first use")
        self.assertEqual(r.run(None, {"wav16k": np.zeros((1, 640), np.float32)}), [("pinned", None)])
        self.assertEqual(r.run(None, {"wav16k": np.zeros((1, 48000), np.float32)}), [("other", "cpu")])
        self.assertEqual(r.run(None, {"wav16k": np.zeros((1, 640), np.float32)}), [("pinned", None)])
        self.assertEqual(len(made), 2)

        def refuse(path, dims=None, device=None):
            if dims:
                raise RuntimeError("The parameter is incorrect.")
            return self.FakeSession(("other", device), [self.FakeInput("mel", [1, 80, "M"])])

        r = sv.Routed(refuse, "hift.onnx", {"M": 40})
        self.assertEqual(r.run(None, {"mel": np.zeros((1, 80, 40), np.float32)}), [("other", None)])


class Binding(unittest.TestCase):
    def test_only_loopback_hosts(self):
        for ok in ("127.0.0.1", "localhost", "::1", "[::1]"):
            self.assertTrue(v2.is_loopback(ok), ok)
        for bad in ("0.0.0.0", "192.168.1.2", "example.com"):
            self.assertFalse(v2.is_loopback(bad), bad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
