"""Gross pitch error of a converted singing sweep against the intended notes (T10: <= 2 % over
E2 -> D#7), with librosa's pYIN (P5-A).

    python scripts/bench/score_singing.py <converted.wav> <semitones>

``<converted.wav>`` is a benchmark's ``offline.output.wav`` (or any recording) of the sweep from
``singing_sweep.py`` converted at ``<semitones>`` (+12 puts the top note at D#7). The lag between
input and output is searched (0-2 s). Prints the voiced share, the gross pitch error (more than
20 % off), the RMSE in cents of the rest, and the error over the frames at or above 2 kHz. Needs
numpy, soundfile and librosa.
"""

import sys

import librosa
import numpy as np
import soundfile as sf

from singing_sweep import HI_HZ, LEAD_S, LO_HZ, SECONDS


def intended(frames: int, semitones: float) -> np.ndarray:
    """The sweep's pitch every 10 ms (0 where nothing is sung), shifted by ``semitones``."""
    t = np.arange(frames) * 0.01 - LEAD_S
    sung = 0.5 + SECONDS + 0.5
    base = np.where(t < 0.5, LO_HZ, LO_HZ * (HI_HZ / LO_HZ) ** np.clip((t - 0.5) / SECONDS, 0, 1))
    f = base * 2 ** (0.25 * np.sin(2 * np.pi * 5.5 * t) / 12) * 2 ** (semitones / 12)
    return np.where((t < 0.1) | (t >= sung), 0, f)


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    path, semitones = sys.argv[1], float(sys.argv[2])
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    hop = sr // 100
    f0, _, _ = librosa.pyin(y, fmin=70, fmax=2800, sr=sr, frame_length=4096 if sr > 40_000 else 2048, hop_length=hop)
    want = intended(1000, semitones)
    best = None
    for lag in range(0, 200):
        fp = f0[lag : lag + len(want)]
        if len(fp) < len(want):
            fp = np.concatenate([fp, np.full(len(want) - len(fp), np.nan)])
        m = (want > 0) & np.isfinite(fp)
        if m.sum() < 200:
            continue
        med = np.median(np.abs(1200 * np.log2(fp[m] / want[m])))
        if best is None or med < best[0]:
            best = (med, lag, fp, m)
    if best is None:
        print(f"{path}: fewer than 200 voiced frames line up with the sweep")
        return 1
    _, lag, fp, m = best
    r = fp[m] / want[m]
    gross = np.abs(r - 1) > 0.2
    top = m & (want >= 2000)
    gross_top = np.abs(fp[top] / want[top] - 1) > 0.2
    cents = 1200 * np.log2(r[~gross])
    voiced = (want > 0).sum()
    print(
        f"{path}: lag {lag * 10} ms, voiced {m.sum()}/{voiced} ({m.sum() / voiced:.0%}), GPE {gross.mean():.2%}, "
        f"cents RMSE {np.sqrt(np.mean(cents**2)):.1f}, >= 2 kHz voiced {top.sum()} GPE {gross_top.mean() if top.any() else 0:.2%}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
