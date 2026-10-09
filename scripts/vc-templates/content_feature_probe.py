#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P3B-D CPU feature experiment: how the content encoder's per-window normalisation and window length
change the features the Sigil Voice Engine feeds its generator (plan item T5-P3, and T5-P7).

ContentVec / HuBERT normalise the first convolution's output with a GroupNorm of 512 groups over 512
channels: every channel is normalised by its mean and variance **over the whole input window**. The
engine encodes a 0.44 s window every 40 ms tick, so the same audio gets different statistics (and
features) in consecutive ticks, in the 0.7 s window of the offline ceiling, and in the whole
utterance; a window that is mostly silence normalises speech by silence statistics. This script
measures that on real speech in PyTorch fp32 (CPU), against the whole-utterance features:

| condition | window | GroupNorm statistics |
|---|---|---|
| `w440` | 0.44 s (the stream today) | the window's own |
| `w700` | 0.70 s (the ceiling's) | the window's own |
| `w440-frozen` | 0.44 s | the whole utterance's (an oracle) |
| `w440-run3` | 0.44 s | running: the last 3 s of speech-gated frames |
| `w440-slice` | 0.44 s | training slice: the last 3.7 s with silences > 0.5 s removed (short pauses kept) |
| `w1000`, `w2000` | 1 s / 2 s of left context | the window's own |
| `w1000-run3`, `w2000-run3` | 1 s / 2 s | running (as `w440-run3`) |
| `w1000-slice`, `w2000-slice` | 1 s / 2 s | training slice |

Each clip (active speech normalised to -20 dBFS) is real speech with a 1.2 s pause inserted (and 0.5 s before it), filled with digital
zeros (`zero`) or -60 dBFS white noise (`noise`, a microphone's noise floor), so the frames right
after >= 1 s of silence (the onset-muting question) exist in every clip.

Reported per condition and floor, over speech frames (20 ms RMS above -50 dBFS):
- `ret`: mean cosine to the whole-utterance features of the newest 4 content frames of each window
  (the frames the tick returns, 40 ms + the SOLA tail);
- `margin`: the frames of the flowdec window before them (positions 4-21 from the end);
- `enc`: every frame of the 0.44 s `enc` window (the generator's input);
- `agree`: cosine of the same frame computed by consecutive ticks (2 frames apart in the window);
- `onset_cos` / `onset_norm`: cosine and norm ratio (window / whole utterance) of the returned
  frames in the first 200 ms after the inserted pause.

T5-P7 (frame grid): `p7_newest` compares the newest returned frame's cosine on the current grid
(the window's last frame ends 15 ms before the newest sample) with an 80-sample-shifted grid
(the last frame ends at the newest sample), and `p7_shift_cos` is the cosine between whole-utterance
features 5 ms apart (the cost of misregistering content against F0 by 5 ms on every frame).

Usage (repo root; CPU only, no GPU, nothing downloaded; ~10-20 min):

    python scripts/vc-templates/content_feature_probe.py [--clips <dir> ...] [--out <json>]

Inputs: the dev cache's ContentVec checkpoint (``%LOCALAPPDATA%/sigil-dev-cache/applio-hf/embedders/
contentvec``) and 16 kHz (or resampled) speech clips (the dev cache's transcribe samples and
sherpa-onnx clips by default). Output: a Markdown table on stdout and the JSON report.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from export_embedders import SRC, load_hubert  # noqa: E402

DEV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
DEFAULT_CLIPS = [
    os.path.join(DEV_CACHE, "models", "transcribe", "samples"),
    os.path.join(DEV_CACHE, "sherpa-onnx", "clips"),
]
SR = 16000
HOP = 320
BLOCK = 640  # 40 ms ticks
ENC_FRAMES50 = 22  # the 0.44 s enc window in 50 Hz frames
RET = 4  # newest 50 Hz frames the tick returns (40 ms + the SOLA tail)
GATE_DB = -50.0
PAUSE_S, LEAD_S = 1.2, 0.5
EPS = 1e-5  # GroupNorm eps


def load_clip(path):
    import soundfile as sf
    from scipy.signal import resample_poly

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    # Loudness: active speech (frames within 30 dB of the loudest) at -20 dBFS RMS, a typical
    # microphone level (some dev clips peak near -47 dBFS).
    db = frame_rms_db(x)
    active = db > db.max() - 30
    gain = 10 ** ((-20 - 10 * np.log10(np.mean(10 ** (db[active] / 10)))) / 20)
    return np.clip(x * gain, -1, 1).astype(np.float32)


def frame_rms_db(x, hop=HOP, win=HOP):
    n = max(0, (len(x) - win) // hop + 1)
    out = np.full(n, -120.0)
    for k in range(n):
        seg = x[k * hop:k * hop + win]
        out[k] = 10 * np.log10(np.mean(seg.astype(np.float64) ** 2) + 1e-12)
    return out


def with_pause(x, floor, rng):
    """`x` with LEAD_S of floor before it and a PAUSE_S pause inserted at its quietest 100 ms point
    in the middle third; returns (audio, pause_end_sample)."""
    def silence(n):
        if floor == "zero":
            return np.zeros(n, np.float32)
        return (rng.standard_normal(n) * 10 ** (-60 / 20)).astype(np.float32)

    third = len(x) // 3
    best, at = None, third
    for s in range(third, 2 * third, 800):
        e = np.mean(x[s:s + 1600] ** 2)
        if best is None or e < best:
            best, at = e, s
    at -= at % HOP
    a = np.concatenate([silence(int(LEAD_S * SR)), x[:at], silence(int(PAUSE_S * SR)), x[at:]])
    pad = (-len(a)) % HOP
    a = np.concatenate([a, silence(pad)]).astype(np.float32)
    return a, int(LEAD_S * SR) + at + int(PAUSE_S * SR)


class StatNorm(torch.nn.Module):
    """GroupNorm(512, 512) of the first conv with optional external statistics (mean, var) per
    channel and batch item ([B, 512, 1]); none = the window's own (the stock module)."""

    def __init__(self, gn):
        super().__init__()
        self.gn = gn
        self.stats = None

    def forward(self, x):
        if self.stats is None:
            return self.gn(x)
        mean, var = self.stats
        return (x - mean) / torch.sqrt(var + EPS) * self.gn.weight[None, :, None] + self.gn.bias[None, :, None]


def features(model, audio_batch):
    with torch.no_grad():
        return model(torch.from_numpy(np.stack(audio_batch))).last_hidden_state.numpy()


def cos_rows(a, b):
    num = np.sum(a * b, -1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12
    return num / den


def conv0_frames(conv0, audio):
    """The first conv's output over `audio` (time-major [T0, 512], before the norm)."""
    with torch.no_grad():
        return conv0(torch.from_numpy(audio)[None, None]).numpy()[0].T


_ANALYSIS = {}


def analyse(model, conv0, audio):
    """Whole-utterance features, the speech gate and the first conv's frames of `audio` (cached)."""
    key = id(audio)
    if key in _ANALYSIS:
        return _ANALYSIS[key]
    ref = features(model, [audio])[0]  # whole utterance, [T, C]
    rms = frame_rms_db(audio)
    c0 = conv0_frames(conv0, audio)
    c0_t = np.minimum((np.arange(c0.shape[0]) * 5 + 5) // HOP, len(rms) - 1)  # 50 Hz frame of each conv0 frame
    silent = ~(rms > GATE_DB)
    long_silent = np.zeros_like(silent)
    k = 0
    while k < len(silent):
        if silent[k]:
            j = k
            while j < len(silent) and silent[j]:
                j += 1
            if (j - k) * HOP / SR > 0.5:
                long_silent[k:j] = True
            k = j
        else:
            k += 1
    out = {"ref": ref, "rms": rms, "c0": c0, "c0_speech": rms[c0_t] > GATE_DB, "c0_keep": ~long_silent[c0_t],
           "whole": (c0.mean(0), c0.var(0))}
    _ANALYSIS[key] = out
    return out


def run_clip(name, model, norm, conv0, audio, pause_end, cond):
    """Per-frame metrics of one condition on one clip (every condition evaluates the same ticks:
    from 2 s on, so the longest window has its history)."""
    window_s, stats_mode = cond
    wlen = int(round(window_s * SR / HOP)) * HOP
    an = analyse(model, conv0, audio)
    ref, rms, c0 = an["ref"], an["rms"], an["c0"]
    n50 = ref.shape[0]
    speech = rms[:n50] > GATE_DB
    c0_hop = 5
    c0_speech, c0_keep, whole_stats = an["c0_speech"], an["c0_keep"], an["whole"]

    def stats_at(end_sample):
        if stats_mode == "window":
            return None
        if stats_mode == "frozen":
            return whole_stats
        e0 = min(c0.shape[0], end_sample // c0_hop)
        if stats_mode == "run3":
            s0 = max(0, e0 - 3 * SR // c0_hop)
            sel = c0[s0:e0][c0_speech[s0:e0]]
        else:  # slice: the last 3.7 s of kept audio (long silences removed)
            idx = np.nonzero(c0_keep[:e0])[0]
            idx = idx[-int(3.7 * SR / c0_hop):]
            sel = c0[idx]
        if sel.shape[0] < 0.25 * SR / c0_hop:  # minimum: 250 ms of frames, else the window's own
            return None
        return (sel.mean(0), sel.var(0))

    ends = list(range(2 * SR, len(audio) + 1, BLOCK))
    rows = {"ret": [], "margin": [], "enc": [], "agree": [], "onset_cos": [], "onset_norm": [], "p7_newest": []}
    prev = None
    batch, metas = [], []

    def flush():
        nonlocal prev
        if not batch:
            return
        st = [m[1] for m in metas]
        if all(s is None for s in st):
            norm.stats = None
        else:
            # Items without external stats use their own window's statistics.
            means, vars_ = [], []
            for (s, a) in zip(st, batch):
                if s is None:
                    w0 = conv0_frames(conv0, a)
                    s = (w0.mean(0), w0.var(0))
                means.append(s[0])
                vars_.append(s[1])
            norm.stats = (torch.from_numpy(np.stack(means)[:, :, None].astype(np.float32)),
                          torch.from_numpy(np.stack(vars_)[:, :, None].astype(np.float32)))
        feats = features(model, batch)
        norm.stats = None
        for f, (end, _) in zip(feats, metas):
            first = (end - len(batch[0])) // HOP  # absolute 50 Hz frame of window frame 0
            nf = f.shape[0]
            enc_from = nf - ENC_FRAMES50
            for p in range(enc_from, nf):
                a = first + p
                if a >= n50 or not speech[a]:
                    continue
                c = float(cos_rows(f[p], ref[a]))
                rows["enc"].append(c)
                if p >= nf - RET:
                    rows["ret"].append(c)
                elif p >= nf - 22:
                    rows["margin"].append(c)
                if p >= nf - RET and pause_end <= a * HOP < pause_end + int(0.2 * SR):
                    rows["onset_cos"].append(c)
                    rows["onset_norm"].append(float(np.linalg.norm(f[p]) / (np.linalg.norm(ref[a]) + 1e-12)))
            if prev is not None:
                pf, pfirst = prev
                for p in range(enc_from, nf):
                    a = first + p
                    q = a - pfirst
                    if 0 <= q < pf.shape[0] and q >= pf.shape[0] - ENC_FRAMES50 and a < n50 and speech[a]:
                        rows["agree"].append(float(cos_rows(f[p], pf[q])))
            prev = (f, first)
        batch.clear()
        metas.clear()

    for end in ends:
        batch.append(audio[end - wlen:end])
        metas.append((end, stats_at(end)))
        if len(batch) == 16:
            flush()
    flush()
    return {k: (float(np.mean(v)) if v else None, len(v)) for k, v in rows.items()}


def p7(model, audio):
    """T5-P7: newest-frame agreement on the current grid vs the end-aligned grid, and the cosine of
    whole-utterance features 5 ms apart."""
    ref0 = features(model, [audio])[0]  # ref0[k] covers audio [320 k, 320 k + 400)
    # 80 zeros first: ref80[m] covers audio [320 m - 80, 320 m + 320), the end-aligned grid.
    ref80 = features(model, [np.concatenate([np.zeros(80, np.float32), audio])])[0]
    n = min(ref0.shape[0], ref80.shape[0])
    speech = frame_rms_db(audio)[:n] > GATE_DB
    shift = cos_rows(ref0[1:n], ref80[1:n])
    shift_cos = float(np.mean(shift[speech[1:n]]))
    cur, end_al = [], []
    for end in range(2 * SR, len(audio) - 400, BLOCK * 2):
        f = features(model, [audio[end - 7040:end]])[0]
        a = (end - 7040) // HOP + f.shape[0] - 1
        if a < n and speech[a]:
            cur.append(float(cos_rows(f[-1], ref0[a])))
        f2 = features(model, [audio[end - 7120:end]])[0]  # last frame covers [end - 400, end)
        m = (end - 320) // HOP
        if m < n and speech[min(m, n - 1)]:
            end_al.append(float(cos_rows(f2[-1], ref80[m])))
    return {"p7_newest_current": float(np.mean(cur)) if cur else None, "p7_newest_end_aligned": float(np.mean(end_al)) if end_al else None,
            "p7_shift_cos": shift_cos}


CONDITIONS = {
    "w440": (0.44, "window"),
    "w700": (0.70, "window"),
    "w440-frozen": (0.44, "frozen"),
    "w440-run3": (0.44, "run3"),
    "w440-slice": (0.44, "slice"),
    "w1000": (1.0, "window"),
    "w1000-run3": (1.0, "run3"),
    "w1000-slice": (1.0, "slice"),
    "w2000": (2.0, "window"),
    "w2000-run3": (2.0, "run3"),
    "w2000-slice": (2.0, "slice"),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", action="append", help="folders of speech WAVs (default: the dev cache's)")
    ap.add_argument("--max-clips", type=int, default=8)
    ap.add_argument("--min-seconds", type=float, default=4.5)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--only", action="append", choices=sorted(CONDITIONS))
    ap.add_argument("--floors", default="zero,noise")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(HERE, "reports", "content_feature_probe.json"))
    args = ap.parse_args()
    if os.name == "nt":
        # A long CPU job on a shared dev rig: idle priority, so timing runs of other work are not
        # disturbed (this measures features, not time).
        import ctypes

        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x40)  # IDLE_PRIORITY_CLASS
    torch.set_num_threads(args.threads)
    hubert, _, _ = load_hubert(os.path.join(args.src, "contentvec"))
    conv_layer = hubert.feature_extractor.conv_layers[0]
    norm = StatNorm(conv_layer.layer_norm)
    conv_layer.layer_norm = norm
    conv0 = conv_layer.conv
    files = []
    for d in args.clips or DEFAULT_CLIPS:
        files += sorted(glob.glob(os.path.join(d, "*.wav")))
    clips = []
    for f in files:
        x = load_clip(f)
        if len(x) >= args.min_seconds * SR and len(clips) < args.max_clips:
            clips.append((os.path.basename(f), x[: int(11 * SR)]))
    print(f"{len(clips)} clips: {', '.join(c[0] for c in clips)}", flush=True)
    rng = np.random.default_rng(7)
    report = {"clips": [c[0] for c in clips], "conditions": {}, "p7": {}}
    conds = args.only or list(CONDITIONS)
    t0 = time.time()
    for floor in args.floors.split(","):
        prepared = [(n,) + with_pause(x, floor, rng) for n, x in clips]
        for cname in conds:
            per_clip = {}
            for n, audio, pause_end in prepared:
                per_clip[n] = run_clip(n, hubert, norm, conv0, audio, pause_end, CONDITIONS[cname])
            agg = {}
            for metric in ("ret", "margin", "enc", "agree", "onset_cos", "onset_norm"):
                vals = [(v[metric][0], v[metric][1]) for v in per_clip.values() if v[metric][0] is not None]
                tot = sum(c for _, c in vals)
                agg[metric] = sum(m * c for m, c in vals) / tot if tot else None
            report["conditions"].setdefault(floor, {})[cname] = {"mean": agg, "per_clip": per_clip}
            print(f"[{time.time() - t0:6.0f} s] {floor:5s} {cname:12s} " + " ".join(
                f"{k} {v:.4f}" if v is not None else f"{k} -" for k, v in agg.items()), flush=True)
        _ANALYSIS.clear()
    for n, x in clips:
        a, _ = with_pause(x, "zero", rng)
        report["p7"][n] = p7(hubert, a)
    p7m = {k: float(np.mean([v[k] for v in report["p7"].values() if v[k] is not None])) for k in ("p7_newest_current", "p7_newest_end_aligned", "p7_shift_cos")}
    report["p7_mean"] = p7m
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)
    # Markdown summary.
    print("\n| floor | condition | ret | margin | enc | agree | onset cos | onset norm |")
    print("|---|---|---|---|---|---|---|---|")
    for floor, conds_ in report["conditions"].items():
        for cname, r in conds_.items():
            m = r["mean"]
            print(f"| {floor} | {cname} | " + " | ".join(f"{m[k]:.4f}" if m[k] is not None else "-" for k in ("ret", "margin", "enc", "agree", "onset_cos", "onset_norm")) + " |")
    print(f"\nT5-P7: newest frame cosine, current grid {p7m['p7_newest_current']:.4f}, end-aligned grid {p7m['p7_newest_end_aligned']:.4f}; "
          f"whole-utterance features 5 ms apart: cosine {p7m['p7_shift_cos']:.4f}")
    print(f"report: {args.out}")


if __name__ == "__main__":
    main()
