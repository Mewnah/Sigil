# SPDX-License-Identifier: AGPL-3.0-or-later
"""CPU NumPy replicas of the benchmark's fidelity scorer (P3B-B, ADR 0009), for prototyping scorer
v2 on saved stream / ceiling WAV pairs before (and as a check of) the Rust implementation.

What is replicated, and how closely:
- ``mcd_db`` / ``mel_cepstra``: ``bench/dsp.rs`` + ``bench/quality.rs`` exactly (f64, Kaldi mel
  filters, the real-cepstrum scale, the 60 dB per-frame floor, frames within 40 dB of the reference's
  loudest frame);
- ``envelope_mcd_db``: the v2 envelope MCD (``quality::envelope_mcd_db``): a CheapTrick-style
  F0-adaptive envelope (Hann window of 3 periods, a 2F0/3 rectangular smoothing of the power
  spectrum, a sinc smoothing lifter and the q1 = -0.15 compensation lifter) put through the same
  mel-cepstrum and MCD as ``mcd_db``. ``tests`` of the Rust port compare against values from here;
- ``rmvpe_f0``: ``vc/rmvpe.rs``'s whole-clip analysis (RVC's mel, end-aligned in a multiple of 32
  frames with the log-mel floor in front, RVC's local-average decoding, the 3-tap median) on the
  ONNX model of ``vc/pitch-rmvpe``;
- the CAM++ front end through ``torchaudio.compliance.kaldi.fbank`` (the export contract);
- resampling with ``scipy.signal.resample_poly`` (the Rust bench uses rubato's sinc resampler: the
  replicas agree with the reports to about 1e-3 on cosines and 1e-2 dB on MCD, which is far below
  every effect measured here).

Nothing here downloads, writes outside the paths it is given, or touches the GPU.
"""

from __future__ import annotations

import json
import math
import os
from functools import lru_cache

import numpy as np

DEV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
SCRATCH = os.environ.get("SIGIL_SCORE_V2_SCRATCH", "")


def lower_priority():
    """Run below every timed benchmark on the PC (Windows IDLE priority class; nice 19 elsewhere)."""
    try:
        if os.name == "nt":
            import ctypes

            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)
        else:
            os.nice(19)
    except Exception:  # best effort
        pass


lower_priority()


# ---------------------------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------------------------

def read_mono(path, channel=0):
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    return x[:, channel].copy(), sr


def to16k(x, sr):
    if sr == 16000:
        return np.asarray(x, np.float32)
    from scipy.signal import resample_poly

    g = math.gcd(16000, sr)
    return resample_poly(np.asarray(x, np.float64), 16000 // g, sr // g).astype(np.float32)


def load_run(run_dir, run="fidelity"):
    """A saved quality run: its report row, the stream (channel 0) and the ceiling at 16 kHz, and the
    scored regions as the report placed them (``lag`` rounded like ``score_pair``)."""
    rep = json.load(open(os.path.join(run_dir, "report.json"), encoding="utf-8"))
    row = next(r for r in rep["runs"] if r["name"] == run)
    s, ssr = read_mono(os.path.join(run_dir, f"{run}.output.wav"))
    c, csr = read_mono(os.path.join(run_dir, f"{run}.ceiling.wav"))
    s16, c16 = to16k(s, ssr), to16k(c, csr)
    d = row["details"]
    lag = int(round(d["alignment"]["lag"]))
    a, b = d["region_16k"]
    return {
        "report": rep, "row": row, "stream": s, "stream_rate": ssr, "ceiling": c, "ceiling_rate": csr,
        "s16": s16, "c16": c16, "lag": lag, "region": (a, b),
        "cs": c16[a:b], "ss": s16[a + lag: b + lag],
    }


# ---------------------------------------------------------------------------------------------
# dsp.rs replica: frames, power spectra, Kaldi mel filters, cepstrum, MCD
# ---------------------------------------------------------------------------------------------

def hann_symmetric(n):
    if n <= 1:
        return np.ones(n)
    return 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n) / (n - 1))


def frame_starts(length, win, hop):
    count = 0 if win == 0 or length < win else 1 + (length - win) // max(hop, 1)
    return np.arange(count) * max(hop, 1)


def kaldi_mel(hz):
    return 1127.0 * np.log(1.0 + np.asarray(hz, np.float64) / 700.0)


def mel_filters(n_mels, n_fft, sr, low, high):
    bins = n_fft // 2 + 1
    bw = sr / n_fft
    lo, hi = kaldi_mel(low), kaldi_mel(high)
    delta = (hi - lo) / (n_mels + 1)
    k = np.arange(bins)
    m = kaldi_mel(bw * k)
    out = np.zeros((n_mels, bins))
    for b in range(n_mels):
        left = lo + b * delta
        center, right = left + delta, left + 2 * delta
        up = (m - left) / (center - left)
        down = (right - m) / (right - center)
        w = np.maximum(np.minimum(up, down), 0.0)
        w[k >= n_fft // 2] = 0.0
        out[b] = w
    return out


def cepstrum(log_bands, n_out):
    k = len(log_bands)
    n = np.arange(n_out)[:, None]
    i = np.arange(k)[None, :]
    return (np.asarray(log_bands)[None, :] * np.cos(np.pi * n * (i + 0.5) / k)).sum(1) / k


MCD_K = 10.0 / math.log(10.0)
MCD_CFG = {"frame_ms": 25.0, "hop_ms": 10.0, "n_mels": 40, "order": 24, "low_hz": 20.0, "high_hz": 8000.0, "range_db": 60.0}


def _mel_ceps_from_power(p, filters, cfg):
    mel = filters @ p
    loudest = max(mel.max(), 0.0)
    floor = max(loudest * 10 ** (-cfg["range_db"] / 10.0), 1e-30)
    log_mel = 0.5 * np.log(np.maximum(mel, floor))
    return cepstrum(log_mel, cfg["order"] + 1)


def mel_cepstra(x, sr=16000, cfg=MCD_CFG):
    x = np.asarray(x, np.float64)
    win = max(int(round(cfg["frame_ms"] / 1000 * sr)), 16)
    hop = max(int(round(cfg["hop_ms"] / 1000 * sr)), 1)
    n_fft = 1 << (win - 1).bit_length()
    window = hann_symmetric(win)
    filters = mel_filters(cfg["n_mels"], n_fft, sr, cfg["low_hz"], min(cfg["high_hz"], sr / 2))
    ceps, energy = [], []
    for s in frame_starts(len(x), win, hop):
        p = np.abs(np.fft.rfft(x[s:s + win] * window, n_fft)) ** 2
        energy.append(10 * np.log10(p.sum() / (n_fft * win) + 1e-12))
        ceps.append(_mel_ceps_from_power(p, filters, cfg))
    return np.array(ceps), np.array(energy)


def mcd_from_ceps(cr, er, ct, speech_range_db=40.0, order=24):
    n = min(len(cr), len(ct))
    if n == 0:
        raise ValueError("no frames")
    peak = er[:n].max()
    keep = er[:n] >= peak - speech_range_db
    d = cr[:n, 1:order + 1] - ct[:n, 1:order + 1]
    per = MCD_K * np.sqrt(2.0 * (d * d).sum(1))
    return float(per[keep].mean()), per, keep


def mcd_db(reference, test, sr=16000):
    n = min(len(reference), len(test))
    cr, er = mel_cepstra(reference[:n], sr)
    ct, _ = mel_cepstra(test[:n], sr)
    return mcd_from_ceps(cr, er, ct)[0]


# ---------------------------------------------------------------------------------------------
# v2: CheapTrick-style envelope MCD
# ---------------------------------------------------------------------------------------------

ENV_F0_FLOOR = 71.0       # WORLD's default f0 floor: the longest window is 3 / 71 s
ENV_F0_UNVOICED = 120.0   # unvoiced frames: a 3-period window of 120 Hz = the STFT MCD's 25 ms frame
                          # (WORLD's 500 Hz default gives 6 ms windows: high-variance noise estimates)
ENV_F0_CEIL = 800.0
ENV_Q1 = -0.15            # CheapTrick's compensation lifter
ENV_FFT = 1024            # 2 ** (1 + floor(log2(3 * 16000 / 71)))


def _linear_smoothing(p, width_hz, sr, n_fft):
    """WORLD ``LinearSmoothing``: the power spectrum averaged over a rectangle of ``width_hz``
    (mirrored at DC and Nyquist, linear interpolation of the cumulative sum at fractional edges)."""
    bins = n_fft // 2 + 1
    df = sr / n_fft
    half_w = width_hz / 2.0
    pad = int(math.ceil(half_w / df)) + 2
    # Mirror: index -j -> j, bins-1+j -> bins-1-j.
    ext = np.concatenate([p[1:pad + 1][::-1], p, p[-pad - 1:-1][::-1]])
    freqs = (np.arange(len(ext)) - pad) * df
    # cumulative integral at bin edges (each bin spans [f - df/2, f + df/2])
    cum = np.concatenate([[0.0], np.cumsum(ext * df)])
    edges = np.concatenate([freqs - df / 2, [freqs[-1] + df / 2]])
    f = np.arange(bins) * df
    lo = np.interp(f - half_w, edges, cum)
    hi = np.interp(f + half_w, edges, cum)
    return (hi - lo) / width_hz


def cheaptrick_power(x, centre, f0, sr=16000, n_fft=ENV_FFT):
    """The CheapTrick power envelope of ``x`` around sample ``centre`` at ``f0`` (``n_fft // 2 + 1``
    bins); samples outside ``x`` are zero."""
    f0 = ENV_F0_UNVOICED if not (f0 > 0) else min(max(f0, ENV_F0_FLOOR), ENV_F0_CEIL)
    half = int(math.floor(1.5 * sr / f0 + 0.5))  # round half away from zero, as the Rust port
    idx = np.arange(-half, half + 1)
    w = 0.5 * np.cos(np.pi * idx / (1.5 * sr / f0)) + 0.5
    w = w / math.sqrt(float((w * w).sum()))
    pos = centre + idx
    seg = np.zeros(len(idx))
    ok = (pos >= 0) & (pos < len(x))
    seg[ok] = x[pos[ok]]
    wf = seg * w
    wf = wf - w * (wf.sum() / w.sum())  # WORLD's DC removal under the window (no dither)
    frame = np.zeros(n_fft)
    m = min(len(wf), n_fft)
    frame[:m] = wf[:m]
    p = np.abs(np.fft.rfft(frame)) ** 2
    p = _linear_smoothing(p, 2.0 * f0 / 3.0, sr, n_fft)
    p = np.maximum(p, 1e-20)
    # Liftering in the quefrency domain (smoothing sinc lifter + compensation lifter).
    log_p = np.log(p)
    full = np.concatenate([log_p, log_p[-2:0:-1]])
    cep = np.fft.ifft(full).real
    q = np.arange(n_fft) / sr
    q = np.minimum(q, (n_fft - np.arange(n_fft)) / sr)
    with np.errstate(divide="ignore", invalid="ignore"):
        smooth = np.where(q == 0, 1.0, np.sin(np.pi * f0 * q) / (np.pi * f0 * q))
    comp = (1.0 - 2.0 * ENV_Q1) + 2.0 * ENV_Q1 * np.cos(2.0 * np.pi * f0 * q)
    env = np.fft.fft(cep * smooth * comp).real[: n_fft // 2 + 1]
    return np.exp(env)


def envelope_ceps(x, f0, sr=16000, cfg=MCD_CFG):
    """Mel-cepstra of the CheapTrick envelope on the STFT MCD's frame grid (frame ``i`` centred at
    ``160 i + 200``; its F0 is ``f0[(160 i + 280) // 160]``, the 10 ms analysis frame nearest the
    centre; 0 = unvoiced) and the frame energies of the plain 25 ms frames (the speech mask)."""
    x = np.asarray(x, np.float64)
    win, hop = 400, 160
    filters = mel_filters(cfg["n_mels"], ENV_FFT, sr, cfg["low_hz"], min(cfg["high_hz"], sr / 2))
    window = hann_symmetric(win)
    ceps, energy = [], []
    for s in frame_starts(len(x), win, hop):
        i = s // hop
        k = (s + 280) // hop
        fv = float(f0[k]) if k < len(f0) else 0.0
        p = cheaptrick_power(x, s + 200, fv, sr)
        ceps.append(_mel_ceps_from_power(p, filters, cfg))
        pe = np.abs(np.fft.rfft(x[s:s + win] * window, 512)) ** 2
        energy.append(10 * np.log10(pe.sum() / (512 * win) + 1e-12))
    return np.array(ceps), np.array(energy)


def envelope_mcd_db(reference, test, f0_ref, sr=16000):
    """Envelope MCD: both renditions analysed with the reference's F0 per frame (one analysis grid,
    so the distance is the signals', not the analyzer's voicing decisions on the test)."""
    n = min(len(reference), len(test))
    cr, er = envelope_ceps(reference[:n], f0_ref, sr)
    ct, _ = envelope_ceps(test[:n], f0_ref, sr)
    return mcd_from_ceps(cr, er, ct)[0]


# ---------------------------------------------------------------------------------------------
# rmvpe.rs replica (whole-clip analysis)
# ---------------------------------------------------------------------------------------------

def _htk_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz) / 700.0)


def _htk_hz(m):
    return 700.0 * (10 ** (np.asarray(m) / 2595.0) - 1.0)


@lru_cache(maxsize=1)
def _rmvpe_mel_basis():
    n_fft, n_mels, sr = 1024, 128, 16000
    freqs = np.arange(n_fft // 2 + 1) * sr / n_fft
    mel_f = _htk_hz(np.linspace(_htk_mel(30.0), _htk_mel(8000.0), n_mels + 2))
    out = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = mel_f[i], mel_f[i + 1], mel_f[i + 2]
        w = np.maximum(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0.0)
        out[i] = w * (2.0 / (hi - lo))
    return out.astype(np.float32)


@lru_cache(maxsize=2)
def _rmvpe_session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def rmvpe_model_path():
    for p in (os.path.join(SCRATCH, "p3f", "appdata", "models", "vc", "pitch-rmvpe", "1", "model.onnx") if SCRATCH else "",
              os.path.join(DEV_CACHE, "vc-assets", "vc", "pitch-rmvpe", "model.onnx")):
        if p and os.path.isfile(p):
            return p
    raise SystemExit("RMVPE model not found: set SIGIL_SCORE_V2_SCRATCH to the scratchpad holding p3f/appdata")


def rmvpe_f0(x16, model=None):
    """Whole-clip RMVPE F0 (Hz per 10 ms frame, 0 = unvoiced), as ``RmvpePitchExtractor::f0`` on the
    plain CPU session."""
    x = np.asarray(x16, np.float32)
    n = len(x)
    if n == 0:
        return np.zeros(0, np.float32)
    frames = 1 + n // 160
    padded = 32 * ((n // 160) // 32 + 1)
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(1024) / 1024)).astype(np.float32)
    pad = 512
    period = 2 * (n - 1) if n > 1 else 1
    mel = np.full((128, padded), math.log(1e-5), np.float32)
    basis = _rmvpe_mel_basis()
    lead = padded - frames
    for t in range(frames):
        idx = t * 160 - pad + np.arange(1024)
        j = np.mod(idx, period)
        j = np.where(j >= n, period - j, j)
        mag = np.abs(np.fft.rfft(x[j] * window)).astype(np.float32)
        mel[:, lead + t] = np.log(np.maximum(basis @ mag, 1e-5))
    sess = _rmvpe_session(model or rmvpe_model_path())
    sal = sess.run(None, {"input": mel[None]})[0][0]  # [padded, 360]
    sal = sal[-frames:]
    f0 = np.zeros(frames, np.float32)
    offset = np.float32(1997.3794)
    for t, row in enumerate(sal):
        c = int(np.argmax(row))
        peak = row[c]
        s, e = max(c - 4, 0), min(c + 5, 360)
        w = row[s:e]
        ws = w.sum()
        if peak <= 0.03 or not ws > 0:
            continue
        cents = float((w * (20.0 * np.arange(s, e) + offset)).sum() / ws)
        f0[t] = 10.0 * 2 ** (cents / 1200.0)
    # smooth_f0 / smooth_f0_stream: 3-tap median, zero-padded at the edges.
    p = np.concatenate([[0.0], f0, [0.0]])
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0).astype(np.float32)


# ---------------------------------------------------------------------------------------------
# F0 errors v1 / v2
# ---------------------------------------------------------------------------------------------

def speech_frames(x16, range_db):
    n = len(x16)
    e = np.array([10 * np.log10(np.mean(np.asarray(x16[i:i + 160], np.float64) ** 2) + 1e-12) for i in range(0, n, 160)])
    return e >= e.max() - range_db


def f0_errors_v2(ref, test, mask=None, gross_ratio=0.2):
    n = min(len(ref), len(test))
    r, t = np.asarray(ref[:n], np.float64), np.asarray(test[:n], np.float64)
    if mask is not None:
        m = np.asarray(mask[:n], bool)
        idx = np.nonzero(m)[0]
        r, t = r[m], t[m]
    else:
        idx = np.arange(n)
    rv, tv = r > 0, t > 0
    both = rv & tv
    cents = np.zeros(len(r))
    cents[both] = 1200 * np.log2(t[both] / r[both])
    gross = both & (np.abs(t / np.where(r > 0, r, 1.0) - 1.0) > gross_ratio)
    fine = both & ~gross
    nb = int(both.sum())
    return {
        "frames": int(len(r)), "voiced_both": nb,
        "rmse_cents": float(np.sqrt(np.mean(cents[both] ** 2))) if nb else float("nan"),
        "fine_rmse_cents": float(np.sqrt(np.mean(cents[fine] ** 2))) if fine.any() else float("nan"),
        "gross_frames": int(gross.sum()),
        "gross_positions": [int(i) for i in idx[gross]],
        "gross_cents": [float(c) for c in cents[gross]],
        "gross_pitch_error": float(gross.sum() / nb) if nb else float("nan"),
        "vuv_error": float((rv != tv).mean()) if len(r) else float("nan"),
    }


# ---------------------------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------------------------

def binom_quantile(q, n, p):
    """Smallest k with P(X <= k) >= q for X ~ Binomial(n, p)."""
    from scipy.stats import binom

    return int(binom.ppf(q, n, p))


def binomial_clicks(at_join, total, coverage, q=0.975):
    return max(0, at_join - binom_quantile(q, total, coverage))


def bootstrap_ci(values, reps=10000, alpha=0.05, seed=20261001):
    v = np.asarray(values, np.float64)
    if len(v) < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(reps, len(v)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))
