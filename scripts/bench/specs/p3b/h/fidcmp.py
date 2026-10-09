#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P3B-H: paired comparison of scorer-v2 fidelity corpora (run_fid.ps1 output folders).

    python fidcmp.py <reference dir> <other dir> [<other dir> ...] [--metrics a,b,c] [--md out.md]

Per metric, the corpus mean of each folder and, for every other folder, the mean of the per-clip
differences (other - reference, clips matched by file name) with a percentile-bootstrap 95 % CI
(P3B-B's `common.bootstrap_ci`). Run the reference twice (an A/A pair) to read the run-to-run
noise: a lever is "within A/A" when its CI sits inside the A/A pair's.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "bench", "score_v2"))

import aggregate as ag  # noqa: E402
import common as cm  # noqa: E402

DEFAULT = (
    "speaker_set_sim_delta", "speaker_cosine_v2", "env_mcd_over_floor_db", "env_mcd_db", "mcd_db", "fed_input_fpe_cents",
    "fed_input_gpe", "fed_input_vuv_error", "f0_fine_rmse_cents", "clicks", "clicks_binomial", "clicks_narrow", "artifact_events",
    "dnsmos_ovrl", "onset_rel_db", "fed_input_onset_lag_frames",
)


def per_clip(folder):
    out = {}
    for r in ag.load_runs([folder]):
        out[r["clip"]] = r["metrics"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("others", nargs="+")
    ap.add_argument("--metrics", default=",".join(DEFAULT))
    ap.add_argument("--md", default="")
    a = ap.parse_args()
    metrics = [m for m in a.metrics.split(",") if m]
    ref = per_clip(a.ref)
    others = {os.path.basename(o.rstrip("/\\")): per_clip(o) for o in a.others}
    lines = [f"Reference `{os.path.basename(a.ref.rstrip('/\\'))}` ({len(ref)} clips); other columns show the mean per-clip difference (other - reference) [bootstrap 95 % CI].", ""]
    head = "| metric | reference |" + "".join(f" {n} |" for n in others)
    lines += [head, "|---|---|" + "---|" * len(others)]
    for m in metrics:
        base = [v[m] for v in ref.values() if m in v]
        if not base:
            continue
        row = f"| {m} | {sum(base) / len(base):.4g} |"
        for name, data in others.items():
            diffs = [data[c][m] - ref[c][m] for c in ref if c in data and m in data[c] and m in ref[c]]
            if not diffs:
                row += " - |"
                continue
            lo, hi = cm.bootstrap_ci(diffs)
            mean = sum(diffs) / len(diffs)
            row += f" {mean:+.4g} [{lo:+.3g}, {hi:+.3g}] (n {len(diffs)}) |"
        lines.append(row)
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
