#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tracker-independent F0 truth from an electroglottograph (EGG) channel, for Sigil's compare
mode (plan T6: gross pitch error against truth no pitch tracker shares).

CMU ARCTIC's bdl, jmk and slt recordings carry a simultaneous EGG channel (the ``orig/`` WAVs:
speech in channel 0, EGG in channel 1). The EGG measures vocal-fold contact; its derivative (dEGG)
peaks sharply at every glottal closure instant (GCI). F0 is then one over the time between
consecutive GCIs: no acoustic pitch tracker is involved, so neither RMVPE nor any other analyzer
an engine might share is the ruler.

Method:
1. EGG high-passed (a 20 ms moving-average subtraction), dEGG = its first difference lightly
   smoothed; the closure polarity is detected (the sign whose peaks are sharper) or given.
2. GCIs: dEGG peaks above 25 % of the local maximum (a 30 ms window), at least 1/f0_max apart.
3. Voicing: the EGG's 10 ms RMS within ``--voicing-db`` (30 dB) of its 99th percentile and 10 dB
   above its noise floor (the 5th percentile).
4. Each GCI interval within [1/f0_max, 1/f0_min] whose ends are voiced gives F0 = 1/interval at its
   midpoint; frame k (at k x 10 ms) takes the median of the intervals within 10 ms of it. A voiced
   frame with no interval is unknown (``nan``, never scored); an unvoiced one is 0.
5. Cross-check (reported, never used as truth): an autocorrelation tracker on the EGG itself, and
   WORLD's DIO on the EGG when ``pyworld`` is installed; the share of frames within 50 cents.

Output: ``time_s,f0_hz`` CSV on the 10 ms grid (``compare.rs`` reads it as a clip's ``f0_truth``;
it is before the transpose: the compare run multiplies it).

    python scripts/vc-bench/egg_f0.py arctic_a0001.wav --out arctic_a0001.f0.csv
    # utterances of one speaker concatenated to clips of at least 10 s with 300 ms gaps
    python scripts/vc-bench/egg_f0.py --concat out/bdl --min-seconds 10 --gap-ms 300 cmu_us_bdl_arctic/orig/arctic_a00*.wav

``--self-test`` checks the extractor on a synthetic EGG with a known F0 contour (no files).
Needs numpy and soundfile.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

HOP_S = 0.01


def _np():
    import numpy as np

    return np


def moving_average(x, n: int):
    np = _np()
    if n <= 1:
        return np.asarray(x, dtype="float64")
    k = np.ones(n) / n
    return np.convolve(x, k, mode="same")


def degg(egg, rate: int, polarity: str = "auto"):
    """Smoothed derivative of the high-passed EGG, signed so closures are positive peaks."""
    np = _np()
    x = np.asarray(egg, dtype="float64")
    x = x - moving_average(x, max(1, int(0.02 * rate)))
    d = np.diff(x, prepend=x[:1])
    d = moving_average(d, max(1, int(0.0001 * rate) * 2 + 1))
    if polarity == "auto":
        hi = np.percentile(d, 99.9)
        lo = -np.percentile(d, 0.1)
        polarity = "pos" if hi >= lo else "neg"
    return (d if polarity == "pos" else -d), polarity


def running_max(x, half: int):
    """Maximum of ``x`` over ±``half`` samples (scipy when installed; else block maxima over ±1-2
    blocks of ``half``, a slightly wider window)."""
    np = _np()
    try:
        from scipy.ndimage import maximum_filter1d

        return maximum_filter1d(x, size=2 * half + 1, mode="nearest")
    except ImportError:
        n = len(x)
        b = max(1, half)
        nb = (n + b - 1) // b
        pad = np.full(nb * b, -np.inf)
        pad[:n] = x
        bm = pad.reshape(nb, b).max(axis=1)
        bm3 = np.maximum(np.maximum(bm, np.r_[bm[1:], -np.inf]), np.r_[-np.inf, bm[:-1]])
        return np.repeat(bm3, b)[:n]


def gcis(dx, rate: int, f0_max: float = 500.0, rel: float = 0.25):
    """Glottal closure instants (sample indices): local maxima of ``dx`` above ``rel`` of the
    maximum within ±15 ms, at least 1/f0_max apart (the larger one wins)."""
    np = _np()
    local_max = running_max(dx, max(1, int(0.015 * rate)))
    is_peak = (dx[1:-1] > dx[:-2]) & (dx[1:-1] >= dx[2:]) & (dx[1:-1] > 0)
    idx = np.nonzero(is_peak)[0] + 1
    idx = idx[dx[idx] >= rel * np.maximum(local_max[idx], 1e-12)]
    min_gap = int(rate / f0_max)
    out = []
    for i in idx:
        if out and i - out[-1] < min_gap:
            if dx[i] > dx[out[-1]]:
                out[-1] = int(i)
            continue
        out.append(int(i))
    return np.asarray(out, dtype="int64")


def voiced_frames(egg, rate: int, voicing_db: float = 30.0):
    np = _np()
    x = np.asarray(egg, dtype="float64")
    x = x - moving_average(x, max(1, int(0.02 * rate)))
    hop = int(HOP_S * rate)
    n = len(x) // hop
    rms = np.array([math.sqrt(float(np.mean(x[i * hop : (i + 1) * hop] ** 2)) + 1e-20) for i in range(n)])
    if n == 0:
        return rms > 0
    ref = np.percentile(rms, 99)
    floor = np.percentile(rms, 5)
    # Within voicing_db of the loud frames and at least 10 dB over the noise floor (the quietest
    # frames: the gaps between utterances).
    thr = max(ref * 10 ** (-voicing_db / 20.0), floor * 10 ** (10.0 / 20.0))
    return rms > thr


def f0_from_gcis(g, rate: int, voiced, n_frames: int, f0_min: float = 50.0, f0_max: float = 500.0):
    """Frame F0 (Hz; 0 unvoiced; nan voiced without a valid interval) on the 10 ms grid."""
    np = _np()
    hop = int(HOP_S * rate)
    mids, f0s = [], []
    for a, b in zip(g[:-1], g[1:]):
        t = (b - a) / rate
        if not (1.0 / f0_max <= t <= 1.0 / f0_min):
            continue
        fa, fb = a // hop, b // hop
        if fa >= len(voiced) or fb >= len(voiced) or not (voiced[fa] and voiced[fb]):
            continue
        mids.append((a + b) / 2.0 / rate)
        f0s.append(1.0 / t)
    mids = np.asarray(mids)
    f0s = np.asarray(f0s)
    out = np.zeros(n_frames)
    for k in range(n_frames):
        if k >= len(voiced) or not voiced[k]:
            continue
        t = k * HOP_S
        sel = np.abs(mids - t) <= HOP_S if len(mids) else np.zeros(0, dtype=bool)
        out[k] = float(np.median(f0s[sel])) if sel.any() else float("nan")
    return out


def egg_truth(egg, rate: int, polarity: str = "auto", voicing_db: float = 30.0, f0_min: float = 50.0, f0_max: float = 500.0):
    """``(f0 per 10 ms frame, polarity used, GCI count)``."""
    dx, pol = degg(egg, rate, polarity)
    g = gcis(dx, rate, f0_max)
    voiced = voiced_frames(egg, rate, voicing_db)
    n = len(egg) // int(HOP_S * rate)
    return f0_from_gcis(g, rate, voiced, n, f0_min, f0_max), pol, len(g)


def acf_f0(egg, rate: int, f0_min: float = 50.0, f0_max: float = 500.0, threshold: float = 0.5):
    """Cross-check: an autocorrelation tracker on the EGG (40 ms windows on the 10 ms grid)."""
    np = _np()
    x = np.asarray(egg, dtype="float64")
    x = x - moving_average(x, max(1, int(0.02 * rate)))
    hop = int(HOP_S * rate)
    win = int(0.04 * rate)
    lo, hi = int(rate / f0_max), int(rate / f0_min)
    n = len(x) // hop
    out = np.zeros(n)
    for k in range(n):
        c = k * hop
        seg = x[max(0, c - win // 2) : c + win // 2]
        if len(seg) < hi + 2:
            continue
        seg = seg - seg.mean()
        e = float(np.dot(seg, seg))
        if e <= 1e-12:
            continue
        r = np.correlate(seg, seg, mode="full")[len(seg) - 1 :] / e
        if hi >= len(r) - 1:
            continue
        lag = lo + int(np.argmax(r[lo:hi]))
        if r[lag] < threshold:
            continue
        a, b0, c1 = r[lag - 1], r[lag], r[lag + 1]
        den = a - 2 * b0 + c1
        frac = 0.5 * (a - c1) / den if abs(den) > 1e-12 else 0.0
        out[k] = rate / (lag + max(-0.5, min(0.5, frac)))
    return out


def agreement(a, b, cents: float = 50.0):
    """Share of frames voiced (finite, > 0) in both that agree within ``cents``."""
    np = _np()
    n = min(len(a), len(b))
    a, b = np.asarray(a[:n]), np.asarray(b[:n])
    m = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    if not m.any():
        return None
    return float(np.mean(np.abs(1200 * np.log2(a[m] / b[m])) <= cents))


def write_csv(path: str, f0) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("time_s,f0_hz\n")
        for k, v in enumerate(f0):
            f.write(f"{k * HOP_S:.3f},{'nan' if not math.isfinite(v) else f'{v:.3f}'}\n")


def read_channels(path: str, speech_channel: int, egg_channel: int):
    import soundfile as sf

    x, rate = sf.read(path, dtype="float32", always_2d=True)
    if x.shape[1] <= max(speech_channel, egg_channel):
        raise SystemExit(f"{path}: {x.shape[1]} channel(s); the EGG is channel {egg_channel} (ARCTIC's orig/ WAVs have speech + EGG)")
    return x[:, speech_channel], x[:, egg_channel], rate


def synthetic_egg(rate: int = 16000, seconds: float = 3.0, seed: int = 1):
    """A synthetic EGG with a known F0 contour: voiced stretches with gliding F0 (100-220 Hz),
    unvoiced gaps; each period a fast contact rise at closure and a slower decay, plus noise.
    ``(egg, true f0 per 10 ms frame)``."""
    np = _np()
    rng = np.random.default_rng(seed)
    n = int(seconds * rate)
    t = np.arange(n) / rate
    # F0 glides of speech-like speed (up to about 200 Hz/s).
    f0 = 140 + 50 * np.sin(2 * np.pi * 0.5 * t) + 4 * np.sin(2 * np.pi * 5.0 * t)
    voiced = ((t % 1.0) > 0.15) & ((t % 1.0) < 0.9)
    phase = np.cumsum(f0 / rate)
    frac = phase - np.floor(phase)
    # A fast, smooth contact rise at closure (3 % of the period: its derivative peaks once, mid
    # rise), then a slower decay as the folds open.
    rise = 0.03
    pulse = np.where(frac < rise, 0.5 - 0.5 * np.cos(np.pi * frac / rise), np.exp(-(frac - rise) / 0.25))
    egg = np.where(voiced, pulse, 0.0) + 0.01 * rng.standard_normal(n)
    hop = int(HOP_S * rate)
    frames = n // hop
    truth = np.array([f0[k * hop] if voiced[k * hop] else 0.0 for k in range(frames)])
    return egg.astype("float32"), truth


def self_test() -> bool:
    np = _np()
    ok = True
    for rate in (16000, 32000):
        egg, truth = synthetic_egg(rate)
        f0, pol, n_gci = egg_truth(egg, rate)
        both = (truth > 0) & np.isfinite(f0) & (f0 > 0)
        err = np.abs(1200 * np.log2(f0[both] / truth[both]))
        vuv = np.mean((truth > 0) != (np.nan_to_num(f0, nan=1.0) > 0))
        cross = agreement(f0, acf_f0(egg, rate))
        print(f"self-test {rate} Hz: {n_gci} GCIs, polarity {pol}, median error {np.median(err):.1f} cents, "
              f"p95 {np.percentile(err, 95):.1f}, V/UV error {vuv:.3f}, ACF cross-check agreement {cross:.3f}")
        ok &= bool(np.median(err) < 5 and np.percentile(err, 95) < 25 and vuv < 0.05 and (cross or 0) > 0.9)
        # Inverted EGG polarity is detected.
        f0n, poln, _ = egg_truth(-egg, rate)
        ok &= poln != pol and np.nanmedian(np.abs(f0n[both] - f0[both])) < 2.0
    print("self-test", "passed" if ok else "FAILED")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="*", help="WAV files with speech and EGG channels")
    ap.add_argument("--out", help="CSV of one input")
    ap.add_argument("--concat", help="output prefix: concatenate the inputs into clips of --min-seconds")
    ap.add_argument("--min-seconds", type=float, default=10.0)
    ap.add_argument("--gap-ms", type=float, default=300.0)
    ap.add_argument("--speech-channel", type=int, default=0)
    ap.add_argument("--egg-channel", type=int, default=1)
    ap.add_argument("--polarity", default="auto", choices=("auto", "pos", "neg"))
    ap.add_argument("--voicing-db", type=float, default=30.0)
    ap.add_argument("--f0-min", type=float, default=50.0)
    ap.add_argument("--f0-max", type=float, default=500.0)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.inputs:
        ap.error("no input (or --self-test)")
    np = _np()
    import soundfile as sf

    def truth_of(path):
        speech, egg, rate = read_channels(path, args.speech_channel, args.egg_channel)
        f0, pol, n = egg_truth(egg, rate, args.polarity, args.voicing_db, args.f0_min, args.f0_max)
        cross = agreement(f0, acf_f0(egg, rate, args.f0_min, args.f0_max))
        dio = None
        try:
            import pyworld

            d, _t = pyworld.dio(egg.astype("float64"), rate, f0_floor=args.f0_min, f0_ceil=args.f0_max, frame_period=10.0)
            dio = agreement(f0, d)
        except Exception:  # noqa: BLE001 (pyworld is optional)
            pass
        print(f"{os.path.basename(path)}: {n} GCIs, polarity {pol}, {np.mean(np.nan_to_num(f0) > 0):.0%} voiced, "
              f"cross-check ACF {cross if cross is None else f'{cross:.3f}'}, DIO {dio if dio is None else f'{dio:.3f}'}")
        return speech, f0, rate

    if args.concat:
        clip, f0s, n_out, rate0 = [], [], 0, None
        for path in args.inputs + [None]:
            if path is not None:
                speech, f0, rate = truth_of(path)
                if rate0 is not None and rate != rate0:
                    raise SystemExit(f"{path}: {rate} Hz, the others {rate0} Hz")
                rate0 = rate
                hop = int(HOP_S * rate)
                speech = speech[: len(f0) * hop]
                if clip:
                    gap = int(round(args.gap_ms / 1000.0 / HOP_S))
                    clip.append(np.zeros(gap * hop, dtype="float32"))
                    f0s.append(np.zeros(gap))
                clip.append(speech)
                f0s.append(f0)
            seconds = sum(len(c) for c in clip) / rate0 if clip else 0.0
            if clip and (seconds >= args.min_seconds or path is None):
                if seconds < args.min_seconds:
                    print(f"left over: {seconds:.1f} s (under {args.min_seconds} s), not written")
                    break
                n_out += 1
                stem = f"{args.concat}_{n_out:02d}"
                os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
                sf.write(stem + ".wav", np.concatenate(clip), rate0, subtype="PCM_16")
                write_csv(stem + ".f0.csv", np.concatenate(f0s))
                print(f"-> {stem}.wav ({seconds:.1f} s) and {stem}.f0.csv")
                clip, f0s = [], []
        return
    if len(args.inputs) != 1 or not args.out:
        ap.error("one input with --out, or --concat")
    _speech, f0, _rate = truth_of(args.inputs[0])
    write_csv(args.out, f0)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
