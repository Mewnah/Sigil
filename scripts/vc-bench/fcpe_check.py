#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Check Sigil's FCPE adapter against upstream torchfcpe on the same clips (report only).

Before any FCPE-against-FCPE comparison in T6 (rvc-realtime and the Deiteris fork's FCPE option),
Sigil's own FCPE path (``vc/fcpe.rs``: its ONNX export plus Sigil's pre- and post-processing) must be
shown to track like upstream ``torchfcpe``; P3 measured Sigil's FCPE at 103 cents and 8.2 % V/UV on
real speech, which could be the adapter or the model.

1. Run a compare v2 corpus with ``"dump_f0": true`` and ``"analyzers": ["fcpe", "rmvpe", ...]``:
   the report then holds every analyzer's F0 of each input clip (``details.v2.clips[].f0``).
2. Run this script on that report in an environment with ``torchfcpe`` (``pip install torchfcpe``;
   a manual step, the model comes with the package):

       python scripts/vc-bench/fcpe_check.py <out>/report.json --run h2h [--threshold 0.006]

It runs torchfcpe on the same input clips (16 kHz, 10 ms hop, the same frame grid) and reports,
per clip and pooled: frames compared, the voicing agreement, the median and p95 pitch difference in
cents over frames voiced in both, the share of gross differences (over 20 %), and, where the clip has
EGG truth (``f0_truth``), each tracker's gross pitch error and fine error against it. Any fix to
``fcpe.rs`` goes to the integrator as a request; this script changes nothing.

``--self-test`` checks the comparison arithmetic on synthetic contours (no torchfcpe needed).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys


def compare_contours(a, b, gross: float = 0.2) -> dict:
    """``a`` against ``b`` (Hz per frame, 0 = unvoiced; the common length): voicing agreement,
    pitch difference in cents over frames voiced in both (median, p95), gross share."""
    n = min(len(a), len(b))
    voiced_a = [x > 0 and math.isfinite(x) for x in a[:n]]
    voiced_b = [x > 0 and math.isfinite(x) for x in b[:n]]
    both = [i for i in range(n) if voiced_a[i] and voiced_b[i]]
    cents = sorted(abs(1200 * math.log2(a[i] / b[i])) for i in both)
    pct = lambda q: cents[min(len(cents) - 1, int(round((len(cents) - 1) * q)))] if cents else None  # noqa: E731
    return {
        "frames": n,
        "voicing_agreement": (sum(1 for i in range(n) if voiced_a[i] == voiced_b[i]) / n) if n else None,
        "voiced_both": len(both),
        "median_cents": pct(0.5),
        "p95_cents": pct(0.95),
        "gross_share": (sum(1 for i in both if abs(a[i] / b[i] - 1) > gross) / len(both)) if both else None,
    }


def against_truth(f0, truth) -> dict:
    """Gross pitch error, fine error (RMSE over non-gross frames, cents) and V/UV error against EGG
    truth (NaN truth frames skipped)."""
    n = min(len(f0), len(truth))
    keep = [i for i in range(n) if math.isfinite(truth[i])]
    both = [i for i in keep if truth[i] > 0 and f0[i] > 0]
    gross = [i for i in both if abs(f0[i] / truth[i] - 1) > 0.2]
    fine = [1200 * math.log2(f0[i] / truth[i]) for i in both if i not in set(gross)]
    return {
        "frames": len(keep),
        "gpe": len(gross) / len(both) if both else None,
        "fpe_cents": math.sqrt(sum(c * c for c in fine) / len(fine)) if fine else None,
        "vuv": sum(1 for i in keep if (truth[i] > 0) != (f0[i] > 0)) / len(keep) if keep else None,
    }


def read_truth(path: str) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                t, v = float(parts[0]), float(parts[1])
            except ValueError:
                continue
            k = int(round(t / 0.01))
            while len(out) <= k:
                out.append(float("nan"))
            out[k] = v
    return out


def torchfcpe_contour(path: str, frames: int, threshold: float) -> list:
    import numpy as np
    import soundfile as sf
    import torch
    from torchfcpe import spawn_bundled_infer_model

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        import librosa

        x = librosa.resample(x, orig_sr=sr, target_sr=16000)
    model = torchfcpe_contour.model if hasattr(torchfcpe_contour, "model") else spawn_bundled_infer_model(device="cpu")
    torchfcpe_contour.model = model
    wav = torch.from_numpy(np.ascontiguousarray(x)).float().unsqueeze(0).unsqueeze(-1)
    f0 = model.infer(wav, sr=16000, decoder_mode="local_argmax", threshold=threshold, f0_min=50, f0_max=1100, interp_uv=False, output_interp_target_length=frames)
    return [float(v) for v in f0.squeeze().cpu().numpy().tolist()]


def self_test() -> bool:
    a = [0, 100, 200, 300, 0, 150]
    b = [0, 101, 400, 300, 120, 150]
    r = compare_contours(a, b)
    ok = r["frames"] == 6 and abs(r["voicing_agreement"] - 5 / 6) < 1e-9 and r["voiced_both"] == 4
    ok &= abs(r["gross_share"] - 0.25) < 1e-9 and r["median_cents"] is not None
    t = against_truth([100, 0, 210, 400], [100, 0, 200, float("nan")])
    ok &= t["frames"] == 3 and t["gpe"] == 0 and abs(t["vuv"]) < 1e-9 and t["fpe_cents"] > 0
    print("self-test", "passed" if ok else "FAILED", r, t)
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?", help="report.json of a compare run with dump_f0")
    ap.add_argument("--run", default="", help="the compare run's name (default: the first compare run)")
    ap.add_argument("--threshold", type=float, default=0.006, help="torchfcpe's voicing threshold")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.report:
        ap.error("a report.json (or --self-test)")
    with open(args.report, encoding="utf-8") as f:
        rep = json.load(f)
    runs = [r for r in rep.get("runs", []) if r.get("mode") == "compare" and (not args.run or r.get("name") == args.run)]
    if not runs:
        raise SystemExit("no compare run in the report")
    v2 = runs[0]["details"].get("v2") or {}
    clips = v2.get("clips", [])
    manifest = (v2.get("corpus") or {}).get("manifest")
    truths = {}
    if manifest and os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as f:
            m = json.load(f)
        base = os.path.dirname(os.path.abspath(manifest))
        truths = {c["id"]: os.path.join(base, c["f0_truth"]) for c in m.get("clips", []) if c.get("f0_truth")}
    pooled_a, pooled_b, rows = [], [], []
    for c in clips:
        f0 = (c.get("f0") or {}).get("fcpe")
        if not f0 or not c.get("path"):
            print(f"{c.get('id')}: no Sigil FCPE contour (run the corpus with dump_f0 and fcpe among the analyzers)")
            continue
        up = torchfcpe_contour(c["path"], len(f0), args.threshold)
        r = {"clip": c["id"], "sigil_vs_torchfcpe": compare_contours(f0, up)}
        if c["id"] in truths:
            t = read_truth(truths[c["id"]])
            r["sigil_vs_egg"] = against_truth(f0, t)
            r["torchfcpe_vs_egg"] = against_truth(up, t)
        rows.append(r)
        pooled_a += f0
        pooled_b += up
        print(json.dumps(r))
    if rows:
        print(json.dumps({"pooled": compare_contours(pooled_a, pooled_b), "clips": len(rows)}))


if __name__ == "__main__":
    main()
