#!/usr/bin/env python3
"""Write Sigil's synthetic benchmark clips as WAV files (Python standard library only).

The benchmark harness generates these signals in-process (`input.kind = "synthetic"`), so specs
need no files. This script writes the same kinds of clips to disk for:

- the local corpus folder (`app_data/bench/corpus`, see `corpus.example.json`), next to the
  consented voice recordings;
- feeding other apps (competitor voice changers, DAWs) the exact input Sigil was measured with;
- the loopback meter's manual checks (play `chirp_train.wav` through any path and record it).

Signals (same definitions as src-tauri/src/services/bench/signals.rs; values are not bit-identical
to the Rust generator, which uses its own PRNG):

- voice:         speech-like voiced source: gliding F0 (95-230 Hz) with 5.5 Hz vibrato, harmonics
                 to 4 kHz shaped by vowel formants that move per syllable, fricative bursts, pauses.
- tones:         311 + 523 + 797 Hz.
- chirp_train:   40 ms Hann-windowed log chirps (200 Hz - 4 kHz), irregularly spaced (~0.35 s).
- impulse_train: single-sample clicks every 0.25-0.45 s.
- noise:         white noise.
- music_bleed:   a quiet synthetic chord progression (for "world music at -20 dB" mixes).

Usage:
    python scripts/bench/make-test-clips.py --out bench-clips
    python scripts/bench/make-test-clips.py --out clips --rate 44100 --seconds 20 --signals voice chirp_train
    python scripts/bench/make-test-clips.py --out clips --mix voice music_bleed:-20

Files are 16-bit PCM mono (`--float` writes 32-bit float) peak-normalized to --level-dbfs.

Interim local corpus (P3B-C; plan T6 dry run), from real speech clips already on this PC:

    python scripts/bench/make-test-clips.py --local-corpus --out <dir> [--inputs a.wav b.wav ...]

Each input is resampled to 16 kHz mono; clips of at least --min-seconds (10 s; DNSMOS needs 9.01 s)
stay whole, shorter ones are joined in a seeded order with 300 ms gaps until they reach it. Every
clip is loudness-normalised to --lufs (-26 LUFS, ITU-R BS.1770-4: a voice changer's silence gate
then sees a mic-like level) and listed in `<dir>/corpus.json` (`sigil-compare-corpus/1`, with each
clip's sha256 and sources). The default inputs are the dev cache's speech clips (the sherpa-onnx
test clips and the transcribe samples). This mode needs numpy, scipy and soundfile; the corpus is
local only and never committed.
"""

from __future__ import annotations

import argparse
import math
import os
import random
import struct
import sys
import wave

SIGNALS = ("voice", "tones", "chirp_train", "impulse_train", "noise", "music_bleed", "silence")

VOWELS = [
    (730.0, 1090.0, 2440.0),  # a
    (270.0, 2290.0, 3010.0),  # i
    (300.0, 870.0, 2240.0),  # u
    (530.0, 1840.0, 2480.0),  # e
    (570.0, 840.0, 2410.0),  # o
]
FORMANT_BW = (90.0, 110.0, 170.0)
FORMANT_GAIN = (1.0, 0.6, 0.3)


def voice(rate: int, n: int, rng: random.Random) -> list[float]:
    total = n / rate
    plan: list[tuple[float, float, int]] = []
    t = 0.15
    phrase_left = rng.uniform(1.2, 2.4)
    while t < total:
        dur = rng.uniform(0.16, 0.30)
        plan.append((t, dur, rng.randrange(len(VOWELS))))
        t += dur + rng.uniform(0.01, 0.05)
        phrase_left -= dur
        if phrase_left <= 0:
            t += rng.uniform(0.25, 0.5)
            phrase_left = rng.uniform(1.2, 2.4)
    base_f0 = rng.uniform(105.0, 145.0)
    p1, p2 = rng.uniform(0, 6), rng.uniform(0, 6)
    out = [0.0] * n
    phase = 0.0
    syl = 0
    noise_lp = 0.0
    two_pi = 2.0 * math.pi
    for i in range(n):
        t = i / rate
        while syl + 1 < len(plan) and plan[syl][0] + plan[syl][1] < t:
            syl += 1
        f0 = base_f0 * (1 + 0.28 * math.sin(two_pi * 0.21 * t + p1) + 0.12 * math.sin(two_pi * 0.63 * t + p2))
        f0 *= 1 + 0.012 * math.sin(two_pi * 5.5 * t)
        f0 = min(max(f0, 95.0), 230.0)
        phase = (phase + two_pi * f0 / rate) % (two_pi * 1024)
        if not plan:
            continue
        start, dur, vowel = plan[syl]
        local = (t - start) / dur
        if not 0.0 <= local <= 1.0:
            continue
        prev = plan[syl - 1][2] if syl > 0 else vowel
        g = min(local / 0.3, 1.0)
        formants = [VOWELS[prev][k] + g * (VOWELS[vowel][k] - VOWELS[prev][k]) for k in range(3)]
        env = math.sin(math.pi * local) ** 0.6
        s = 0.0
        for h in range(1, max(int(4000.0 / f0), 1) + 1):
            fh = h * f0
            shape = sum(FORMANT_GAIN[k] / (1 + ((fh - formants[k]) / FORMANT_BW[k]) ** 2) for k in range(3))
            s += (0.05 + shape) / h * math.sin(h * phase)
        white = rng.uniform(-1, 1)
        hp = white - noise_lp
        noise_lp = 0.7 * noise_lp + 0.3 * white
        fric = 0.25 * hp * (1 - local / 0.18) if (syl % 3 == 0 and local < 0.18) else 0.0
        out[i] = env * s + fric
    return out


def burst(tau: float, burst_s: float, f_lo: float = 200.0, f_hi: float = 4000.0) -> float:
    if not 0.0 <= tau < burst_s:
        return 0.0
    k = math.log(f_hi / f_lo) / burst_s
    ph = 2 * math.pi * f_lo * (math.exp(k * tau) - 1) / k
    return (0.5 - 0.5 * math.cos(2 * math.pi * tau / burst_s)) * math.sin(ph)


def chirp_train(rate: int, n: int, rng: random.Random, burst_s: float = 0.04, gap: float = 0.35) -> list[float]:
    out = [0.0] * n
    t = 0.05
    seconds = n / rate
    while t + burst_s < seconds:
        first = math.ceil(t * rate)
        last = min(math.ceil((t + burst_s) * rate), n)
        for i in range(first, last):
            out[i] += burst(i / rate - t, burst_s)
        t += burst_s + gap * rng.uniform(0.8, 1.2)
    return out


def music_bleed(rate: int, n: int, rng: random.Random) -> list[float]:
    """Slow chords (sine triads with soft attacks), a stand-in for world music in the mic."""
    roots = [220.0, 196.0, 174.61, 164.81]
    out = [0.0] * n
    bar = 2.0
    for i in range(n):
        t = i / rate
        chord = roots[int(t / bar) % len(roots)]
        local = (t % bar) / bar
        env = min(local * 20, 1.0) * (1 - 0.5 * local)
        out[i] = env * sum(math.sin(2 * math.pi * chord * r * t) for r in (1.0, 1.26, 1.5))
    return out


def synth(kind: str, rate: int, seconds: float, rng: random.Random) -> list[float]:
    n = int(round(seconds * rate))
    if kind == "silence":
        return [0.0] * n
    if kind == "voice":
        return voice(rate, n, rng)
    if kind == "tones":
        return [sum(math.sin(2 * math.pi * f * i / rate) for f in (311.0, 523.0, 797.0)) for i in range(n)]
    if kind == "chirp_train":
        return chirp_train(rate, n, rng)
    if kind == "impulse_train":
        out = [0.0] * n
        t = 0.1
        while t < seconds:
            i = int(t * rate)
            if i < n:
                out[i] = 1.0
            t += rng.uniform(0.25, 0.45)
        return out
    if kind == "noise":
        return [rng.uniform(-1, 1) for _ in range(n)]
    if kind == "music_bleed":
        return music_bleed(rate, n, rng)
    raise ValueError(f"unknown signal {kind!r} (expected one of {', '.join(SIGNALS)})")


def normalize(x: list[float], peak: float) -> list[float]:
    m = max((abs(v) for v in x), default=0.0)
    return x if m == 0 else [v * peak / m for v in x]


def write_wav(path: str, x: list[float], rate: int, as_float: bool) -> None:
    if as_float:
        # The wave module only writes PCM; write a WAVE_FORMAT_IEEE_FLOAT file by hand.
        data = struct.pack(f"<{len(x)}f", *x)
        fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)
        with open(path, "wb") as f:
            f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE")
            f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
            f.write(b"data" + struct.pack("<I", len(data)) + data)
        return
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", max(-32767, min(32767, int(round(v * 32767))))) for v in x))


def parse_mix(items: list[str]) -> list[tuple[str, float]]:
    """`voice music_bleed:-20` -> [(voice, 0 dB), (music_bleed, -20 dB)] (levels relative to peak)."""
    out = []
    for item in items:
        name, _, db = item.partition(":")
        if name not in SIGNALS:
            raise SystemExit(f"--mix: unknown signal {name!r}")
        out.append((name, float(db) if db else 0.0))
    return out


def default_local_inputs() -> list[str]:
    cache = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
    out = []
    for sub in (os.path.join("sherpa-onnx", "clips"), os.path.join("models", "transcribe", "samples")):
        d = os.path.join(cache, sub)
        if os.path.isdir(d):
            out += sorted(os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".wav"))
    return out


def local_corpus(out: str, inputs: list[str], seed: int, min_s: float, gap_ms: float, lufs: float) -> str:
    """Build the interim local corpus (see the module docs); returns the manifest's path."""
    import hashlib
    import json

    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "vc-bench"))
    import vcbench_common as vb

    rate = 16000
    clips = []
    for p in inputs:
        x, sr = sf.read(p, dtype="float32", always_2d=True)
        x = x.mean(axis=1)
        if sr != rate:
            g = math.gcd(sr, rate)
            x = resample_poly(x, rate // g, sr // g).astype("float32")
        clips.append((os.path.splitext(os.path.basename(p))[0], p, x))
    long = [c for c in clips if len(c[2]) / rate >= min_s]
    short = [c for c in clips if len(c[2]) / rate < min_s]
    random.Random(seed).shuffle(short)
    groups = [[c] for c in long]
    cur, total = [], 0.0
    for c in short:
        cur.append(c)
        total += len(c[2]) / rate + (gap_ms / 1000.0 if len(cur) > 1 else 0.0)
        if total >= min_s:
            groups.append(cur)
            cur, total = [], 0.0
    if cur:
        if groups and len(groups[-1]) > 1:
            groups[-1] += cur  # the remainder joins the last joined clip
        else:
            print(f"left over (under {min_s} s): {[c[0] for c in cur]}")
    os.makedirs(os.path.join(out, "clips"), exist_ok=True)
    entries = []
    for g in groups:
        parts = []
        for i, c in enumerate(g):
            if i:
                parts.append(np.zeros(int(gap_ms / 1000.0 * rate), dtype="float32"))
            parts.append(c[2])
        x = np.concatenate(parts)
        loud = vb.loudness_lufs(x, rate)
        gain = (lufs - loud) if loud is not None else 0.0
        peak = float(np.max(np.abs(x))) * 10 ** (gain / 20)
        if peak > 10 ** (-1 / 20):
            gain -= 20 * math.log10(peak / 10 ** (-1 / 20))
        x = (x * 10 ** (gain / 20)).astype("float32")
        cid = g[0][0] if len(g) == 1 else "join-" + "-".join(c[0] for c in g)
        cid = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in cid)[:90]
        rel = f"clips/{cid}.wav"
        path = os.path.join(out, rel)
        sf.write(path, x, rate, subtype="PCM_16")
        with open(path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        entries.append({"id": cid, "input": rel, "sha256": sha, "seconds": round(len(x) / rate, 3),
                        "source": " + ".join(os.path.basename(c[1]) for c in g),
                        "notes": f"16 kHz mono, {lufs} LUFS (gain {gain:+.1f} dB); local dev-cache speech, never committed"})
    manifest = {"schema": "sigil-compare-corpus/1", "name": os.path.basename(os.path.normpath(out)),
                "description": f"Interim local corpus (make-test-clips.py --local-corpus, seed {seed}): the dev cache's speech clips, {min_s} s or longer, {lufs} LUFS.",
                "clips": entries, "points": [], "min_clips": 20}
    mpath = os.path.join(out, "corpus.json")
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    print(f"{len(entries)} clips -> {mpath}")
    return mpath


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument("--local-corpus", action="store_true", help="build the interim local corpus (see the docs)")
    ap.add_argument("--inputs", nargs="+", default=None, help="--local-corpus inputs (default: the dev cache's speech clips)")
    ap.add_argument("--min-seconds", type=float, default=10.0)
    ap.add_argument("--gap-ms", type=float, default=300.0)
    ap.add_argument("--lufs", type=float, default=-26.0)
    ap.add_argument("--rate", type=int, default=48000)
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--level-dbfs", type=float, default=-12.0, help="peak level of every clip")
    ap.add_argument("--seed", type=int, default=20260928)
    ap.add_argument("--signals", nargs="+", default=["voice", "tones", "chirp_train", "impulse_train", "noise"],
                    choices=SIGNALS)
    ap.add_argument("--mix", nargs="+", metavar="SIGNAL[:DB]",
                    help="also write one clip mixing these signals (e.g. voice music_bleed:-20)")
    ap.add_argument("--float", action="store_true", help="write 32-bit float WAV instead of 16-bit PCM")
    args = ap.parse_args(argv)
    if args.local_corpus:
        inputs = args.inputs or default_local_inputs()
        if not inputs:
            ap.error("no inputs (and no dev-cache clips found)")
        local_corpus(args.out, inputs, args.seed, args.min_seconds, args.gap_ms, args.lufs)
        return 0
    if not 8000 <= args.rate <= 384000 or not 0 < args.seconds <= 3600:
        ap.error("rate must be 8000-384000 and seconds 0-3600")
    os.makedirs(args.out, exist_ok=True)
    peak = 10 ** (args.level_dbfs / 20)
    written = []
    for i, kind in enumerate(args.signals):
        x = normalize(synth(kind, args.rate, args.seconds, random.Random(args.seed + i)), peak)
        path = os.path.join(args.out, f"{kind}_{args.rate}.wav")
        write_wav(path, x, args.rate, args.float)
        written.append(path)
    if args.mix:
        parts = parse_mix(args.mix)
        n = int(round(args.seconds * args.rate))
        mix = [0.0] * n
        for j, (kind, db) in enumerate(parts):
            x = normalize(synth(kind, args.rate, args.seconds, random.Random(args.seed + 100 + j)), 10 ** (db / 20))
            mix = [a + b for a, b in zip(mix, x)]
        name = "mix_" + "_".join(f"{k}{int(d)}" if d else k for k, d in parts)
        path = os.path.join(args.out, f"{name}_{args.rate}.wav")
        write_wav(path, normalize(mix, peak), args.rate, args.float)
        written.append(path)
    for p in written:
        print(p)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
