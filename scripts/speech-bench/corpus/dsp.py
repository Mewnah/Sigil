# SPDX-License-Identifier: AGPL-3.0-or-later
"""Signal processing of the P7 corpus builder (numpy + scipy; everything float32 / float64 mono at
16 kHz, deterministic given a seeded ``numpy.random.Generator``).

Levels. ``active_power`` is the mean power of the 20 ms frames whose power lies within
``ACTIVE_RANGE_DB`` (30 dB) of the 95th-percentile frame power, frames of exact digital silence
(power below ``FLOOR_DBFS`` = -300 dBFS) never counting: a simple, scale-invariant active-level
measure in the spirit of ITU-T P.56 that ignores pauses in speech and gaps between effects. Every SNR of the
corpus is ``10 log10(active_power(speech) / active_power(interferer))``.

Speech level. FLEURS recordings range from about -20 to -66 dBFS RMS; the builder brings every
utterance to an active level of ``speech_level.active_dbfs`` (-26 dBFS, P.56's nominal level) plus
a seeded per-utterance offset within +-``jitter_db`` (gain-staging variety), the same gain under
every condition of that utterance, before any mixing. Real drop-in clips keep their own level.

Rooms. ``synthetic_rir``: a direct path (gain 1) plus an exponentially decaying Gaussian tail
(60 dB decay over ``rt60``), starting 2 ms after the direct path, slightly low-passed (air /
absorption), scaled so that direct-to-reverberant energy is ``drr_db``. Convolution keeps the input
length (causal, the tail past the end is cut).

Mic. Butterworth high / low-pass (causal ``sosfilt``), ``mu_law`` round trip at ``bits``
(G.711-like quantization noise), white hiss at an RMS level.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.signal import butter, fftconvolve, lfilter, resample_poly, sosfilt

SR = 16000
FRAME_S = 0.020
ACTIVE_RANGE_DB = 30.0
FLOOR_DBFS = -300.0


def db_to_power(db: float) -> float:
    return 10.0 ** (db / 10.0)


def power_to_db(p: float) -> float:
    return 10.0 * math.log10(max(p, 1e-20))


def frame_powers(x: np.ndarray, sr: int = SR, frame_s: float = FRAME_S) -> np.ndarray:
    n = max(1, int(round(frame_s * sr)))
    frames = len(x) // n
    if frames == 0:
        return np.array([float(np.mean(np.square(x, dtype=np.float64)))]) if len(x) else np.zeros(1)
    f = np.asarray(x[: frames * n], dtype=np.float64).reshape(frames, n)
    return np.mean(f * f, axis=1)


def active_power(x: np.ndarray, sr: int = SR) -> float:
    """Mean power of the active frames (see the module doc); 0 for silence."""
    p = frame_powers(x, sr)
    floor = db_to_power(FLOOR_DBFS)
    p = p[p >= floor]
    if p.size == 0:
        return 0.0
    ref = float(np.percentile(p, 95))
    active = p[p >= ref * db_to_power(-ACTIVE_RANGE_DB)]
    return float(np.mean(active)) if active.size else 0.0


def active_dbfs(x: np.ndarray, sr: int = SR) -> float:
    return power_to_db(active_power(x, sr))


def silent_fraction(x: np.ndarray, threshold_dbfs: float = -50.0, sr: int = SR) -> float:
    p = frame_powers(x, sr)
    return float(np.mean(p < db_to_power(threshold_dbfs))) if p.size else 1.0


def gain_for_snr(speech_power: float, noise_power: float, snr_db: float) -> float:
    """Amplitude gain that puts ``noise`` ``snr_db`` below ``speech`` (active levels)."""
    if noise_power <= 0.0 or speech_power <= 0.0:
        return 0.0
    return math.sqrt(speech_power / (noise_power * db_to_power(snr_db)))


def fit_length(x: np.ndarray, n: int) -> np.ndarray:
    """``x`` cut or tiled (repeated) to exactly ``n`` samples."""
    if len(x) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(x) >= n:
        return x[:n]
    reps = int(math.ceil(n / len(x)))
    return np.tile(x, reps)[:n]


def synthetic_rir(rng: np.random.Generator, rt60_s: float, drr_db: float, sr: int = SR, gap_ms: float = 2.0) -> np.ndarray:
    n = max(int(rt60_s * 1.2 * sr), 2)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.907755 * t / max(rt60_s, 1e-3))
    tail[: int(gap_ms / 1000.0 * sr) + 1] = 0.0
    tail = lfilter([0.6], [1.0, -0.4], tail)  # gentle air / wall absorption
    energy = float(np.sum(tail * tail))
    if energy > 0:
        tail *= math.sqrt(db_to_power(-drr_db) / energy)
    h = tail
    h[0] = 1.0
    return h.astype(np.float64)


def convolve_same(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal convolution, cut to ``len(x)``."""
    return fftconvolve(np.asarray(x, dtype=np.float64), h)[: len(x)]


def butter_filter(x: np.ndarray, kind: str, cutoff_hz: float, order: int = 4, sr: int = SR) -> np.ndarray:
    if kind == "lowpass" and cutoff_hz >= sr / 2 * 0.98:
        return np.asarray(x, dtype=np.float64)
    sos = butter(order, cutoff_hz, btype=kind, fs=sr, output="sos")
    return sosfilt(sos, np.asarray(x, dtype=np.float64))


def mu_law(x: np.ndarray, bits: int = 8) -> np.ndarray:
    """mu-law compress, quantize to ``bits``, expand (``mu = 2**bits - 1``)."""
    mu = float(2 ** bits - 1)
    y = np.clip(np.asarray(x, dtype=np.float64), -1.0, 1.0)
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / math.log1p(mu)
    levels = float(2 ** (bits - 1) - 1)
    q = np.round(c * levels) / levels
    return np.sign(q) * ((1.0 + mu) ** np.abs(q) - 1.0) / mu


def hiss(rng: np.random.Generator, n: int, rms_dbfs: float) -> np.ndarray:
    return rng.standard_normal(n) * math.sqrt(db_to_power(rms_dbfs))


def trim_active(x: np.ndarray, sr: int = SR, range_db: float = 35.0, pad_s: float = 0.05) -> np.ndarray:
    """``x`` without leading / trailing frames more than ``range_db`` below its loudest frame."""
    p = frame_powers(x, sr)
    if p.size == 0 or p.max() <= 0:
        return x[:0]
    keep = np.nonzero(p >= p.max() * db_to_power(-range_db))[0]
    n = int(round(FRAME_S * sr))
    pad = int(pad_s * sr)
    start = max(0, keep[0] * n - pad)
    end = min(len(x), (keep[-1] + 1) * n + pad)
    return x[start:end]


def resample(x: np.ndarray, src_rate: int, dst_rate: int = SR) -> np.ndarray:
    if src_rate == dst_rate:
        return np.asarray(x, dtype=np.float64)
    g = math.gcd(int(src_rate), int(dst_rate))
    return resample_poly(np.asarray(x, dtype=np.float64), dst_rate // g, src_rate // g)


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    return x if x.ndim == 1 else x.mean(axis=1)


def headroom(y: np.ndarray, peak: float = 0.98) -> tuple[np.ndarray, float]:
    """``y`` scaled down when its peak exceeds ``peak``; returns (signal, gain in dB applied)."""
    m = float(np.max(np.abs(y))) if len(y) else 0.0
    if m <= peak or m == 0.0:
        return y, 0.0
    g = peak / m
    return y * g, 20.0 * math.log10(g)


def to_pcm16(y: np.ndarray) -> np.ndarray:
    return np.clip(np.round(np.asarray(y, dtype=np.float64) * 32767.0), -32768, 32767).astype(np.int16)
