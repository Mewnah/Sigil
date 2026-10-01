#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference output for the Rust MeanVC2 port (P5-C, ``vc::families::meanvc``).

Runs ``sampler.py``'s ``SigilMeanVC`` on the exported graphs the way the Rust engine streams a whole
tick per call: 80 ms ticks (``--tick 1280``) or 160 ms, BN upsampled to 4k + 1 frames for k new BN
frames (the ADR 0006 generalisation), the 720-sample phase lead, and Sigil's counter-based anchored
noise (``engine::noise::AnchoredNoise``, stream ``Flow``, index = channel * 2^40 + mel frame) as the
z draws. The speaker is a seeded random vector (``--spk-seed``, scaled to ``--spk-norm``): parity of
the loop does not depend on the voice.

Writes ``<out>/input.f32`` (16 kHz mono), ``spks.npy``, ``reference.f32`` (the engine's history
stream: 960 zeros, then the loop's output) and ``meta.json``. The Rust test
``real_model_matches_the_python_reference_loop`` (``#[ignore]``) reads them from
``SIGIL_MEANVC_REFERENCE_DIR`` and gates on SNR >= 35 dB (P5-C measured 85.3 dB on ``jfk.wav``).

Usage (repo root):
``$LOCALAPPDATA/sigil-dev-cache/venv-ort128-cuda/Scripts/python scripts/vc-families/meanvc/rust_reference.py --out <dir>``
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import common as cm  # noqa: E402
import sampler as smp  # noqa: E402

M64 = (1 << 64) - 1
GOLD = 0x9E3779B97F4A7C15
FLOW_KEY = 0x243F6A8885A308D3
FRAME_STRIDE = 1 << 40
LEAD = 720


def mix64(z):
    z &= M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


class Anchored:
    def __init__(self, seed):
        self.seed = mix64(seed ^ GOLD)
        self.k = mix64(self.seed ^ FLOW_KEY)

    def unit(self, index):
        h = mix64(self.k ^ ((index * GOLD) & M64))
        u1 = ((h >> 32) + 1.0) * (1.0 / 4294967296.0)
        u2 = (h & 0xFFFFFFFF) * (1.0 / 4294967296.0)
        return np.float32(math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2))

    def frames(self, first, n, width=80):
        out = np.empty((1, n, width), np.float32)
        for f in range(n):
            for c in range(width):
                out[0, f, c] = self.unit(((c * FRAME_STRIDE) + first + f) & M64)
        return out


class Generalised(smp.SigilMeanVC):
    """BN of k new frames -> 4k + 1 frames (first dropped)."""

    def encode_chunk(self, samples):
        padded = np.concatenate((self.samples_cache, samples)) if self.samples_cache.size else samples
        if len(padded) < 400:
            self.samples_cache = padded
            return None
        fb = self._run("fbank", {"wav16k": padded[None].astype(np.float32)})[0][0]
        self.samples_cache = padded[160 * fb.shape[0]:]
        if self.frame_cache is not None:
            fb = np.concatenate([self.frame_cache, fb], axis=0)
        if self.asr_offset >= 4000:
            self.asr_offset = max(self.att_cache.shape[2], self.asr_offset - 4000)
        bns = []
        i = 0
        while i + smp.BN_WINDOW <= fb.shape[0]:
            out, self.att_cache, self.cnn_cache = self._run("asr", {
                "fbank": np.ascontiguousarray(fb[i:i + smp.BN_WINDOW][None]), "offset": np.array(self.asr_offset, np.int64),
                "rcs": np.array(smp.RCS, np.int64), "att_cache": self.att_cache, "cnn_cache": self.cnn_cache})
            bns.append(out[0])
            self.asr_offset += smp.ASR_OFFSET_STEP
            i += smp.BN_STRIDE
        self.frame_cache = fb[i:]
        if not bns:
            return None
        new = np.concatenate(bns, axis=0)[None]
        k = new.shape[1]
        bn = new if self.encoder_output_cache is None else np.concatenate([self.encoder_output_cache, new], axis=1)
        self.encoder_output_cache = bn[:, -1:]
        if bn.shape[1] < 2:
            return None
        return smp.upsample_linear_align_corners(bn, 4 * k + 1)[:, 1:]


def run(args):
    d = os.path.join(cm.DEV_CACHE, "vc-families", "meanvc")
    sess = {n: cm.session(os.path.join(d, n + ".onnx"), "cpu", threads=1) for n in ["fbank", "asr", "gtm", "dit_step", "vocos"]}
    wav = cm.load_wav16k(args.wav, seconds=args.seconds)
    rng = np.random.default_rng(args.spk_seed)
    spks = rng.standard_normal((1, 256)).astype(np.float32)
    spks /= np.linalg.norm(spks)
    spks *= np.float32(args.spk_norm)
    noise = Anchored(args.noise_seed)
    m = Generalised(sess, spks, steps=args.steps, audio_chunk=args.tick)
    m.reset()
    frame = [0]

    def draws():
        while True:
            z = noise.frames(frame[0], 8)
            frame[0] += 4
            yield z

    it = draws()
    ticks = len(wav) // args.tick
    hist = [np.zeros(960, np.float32)]
    produced = []
    for t in range(ticks):
        block = wav[t * args.tick:(t + 1) * args.tick]
        if t == 0:
            block = np.concatenate([np.zeros(LEAD, np.float32), block])
        o = m.process_chunk(np.ascontiguousarray(block, np.float32), it)
        n = 0 if o is None else len(o)
        produced.append(n)
        if o is not None:
            hist.append(o.astype(np.float32))
    ref = np.concatenate(hist)
    os.makedirs(args.out, exist_ok=True)
    wav[: ticks * args.tick].astype("<f4").tofile(os.path.join(args.out, "input.f32"))
    ref.astype("<f4").tofile(os.path.join(args.out, "reference.f32"))
    np.save(os.path.join(args.out, "spks.npy"), spks[0])
    # content delay: envelope cross-correlation (10 ms RMS frames)
    def env(x):
        n = len(x) // 160
        return np.sqrt(np.mean(x[: n * 160].reshape(n, 160) ** 2, axis=1) + 1e-12)
    a = np.log(env(wav[: ticks * args.tick]) + 1e-4)
    b = np.log(env(ref) + 1e-4)
    n = min(len(a), len(b))
    a = a[:n] - a[:n].mean()
    b = b[:n] - b[:n].mean()
    best = max(range(0, 30), key=lambda lag: float(np.dot(a[: n - lag], b[lag:n])))
    meta = {"ticks": ticks, "tick": args.tick, "steps": args.steps, "noise_seed": args.noise_seed, "lead": LEAD,
            "produced_per_tick": produced[:6], "ref_len": int(len(ref)), "envelope_delay_frames": best,
            "spk_norm": args.spk_norm, "rms_out": float(np.sqrt(np.mean(ref ** 2))), "rms_in": float(np.sqrt(np.mean(wav ** 2)))}
    json.dump(meta, open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--wav", default="jfk.wav")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--tick", type=int, default=1280)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--noise-seed", type=int, default=7)
    ap.add_argument("--spk-seed", type=int, default=0)
    ap.add_argument("--spk-norm", type=float, default=1.0)
    run(ap.parse_args())
