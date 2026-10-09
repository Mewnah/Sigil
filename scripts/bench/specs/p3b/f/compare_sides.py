#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paired comparison of two side folders of `run_fidelity.ps1` runs (P3B-F).

    python compare_sides.py <run1>/B <run2>/B          # a side of one run against a side of another
    python compare_sides.py <run>/A <run>/B            # the two sides of one run, with the extra rows

A side folder holds one `<clip>/report.json` per clip. The engine is deterministic, so the same
configuration gives bit-identical numbers in separate runs (the default side of every run of a
campaign matched to the last digit), and sides of different runs pair exactly clip by clip: the way
to answer a question no single A/B asked (for example the 32-frame plan against the 64-frame plan
with a filled lead). For every metric the mean difference B - A over the shared clips, its paired
Student t 95 % interval, the number of clips where B is not worse and better, and a verdict.

The rows `run_fidelity.ps1` leaves out are here too: the same metrics against the whole-utterance
ceiling (`*_vs_whole`). The onset lag is signed in the report (negative: the engine voiced before the
reference), so it is scored by its size.
"""
import json
import math
import os
import sys

# Two-sided 95 % Student t quantiles by degrees of freedom.
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086}
ABS_KEYS = {"fed_input_onset_lag_frames"}
# (metric, lower is better)
KEYS = [
    ("fed_input_fpe_cents", True), ("fed_input_gpe", True), ("fed_input_vuv_error", True), ("fed_input_onset_lag_frames", True),
    ("speaker_cosine_v2", False), ("speaker_cosine_v2_vs_whole", False), ("env_mcd_db", True), ("env_mcd_db_vs_whole", True),
    ("mcd_db", True), ("mcd_db_vs_whole", True), ("dnsmos_ovrl", False), ("whole_fed_input_vuv_error", True),
]


def metrics(side, clip):
    with open(os.path.join(side, clip, "report.json"), encoding="utf-8") as f:
        return json.load(f)["runs"][0]["metrics"]


def compare(a_side, b_side):
    clips = sorted(c for c in os.listdir(a_side) if os.path.exists(os.path.join(b_side, c, "report.json"))
                   and os.path.exists(os.path.join(a_side, c, "report.json")))
    print(f"{a_side}  vs  {b_side}: {len(clips)} clips")
    for key, lower in KEYS:
        a_vals, b_vals, diffs = [], [], []
        for clip in clips:
            ma, mb = metrics(a_side, clip), metrics(b_side, clip)
            if ma.get(key) is None or mb.get(key) is None:
                continue
            xa, xb = (abs(ma[key]), abs(mb[key])) if key in ABS_KEYS else (ma[key], mb[key])
            a_vals.append(xa)
            b_vals.append(xb)
            diffs.append(xb - xa)
        n = len(diffs)
        if n == 0:
            continue
        mean = sum(diffs) / n
        sd = math.sqrt(sum((d - mean) ** 2 for d in diffs) / (n - 1)) if n > 1 else 0.0
        half = T975.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else float("nan")
        lo, hi = mean - half, mean + half
        not_worse = sum(1 for d in diffs if (d <= 1e-9 if lower else d >= -1e-9))
        better = sum(1 for d in diffs if (d < -1e-9 if lower else d > 1e-9))
        if n < 2:
            verdict = "n/a"
        elif (lower and hi < 0) or (not lower and lo > 0):
            verdict = "B better"
        elif (lower and lo > 0) or (not lower and hi < 0):
            verdict = "B worse"
        else:
            verdict = "within noise"
        print(f"{key:30s} A {sum(a_vals) / n:9.4f} B {sum(b_vals) / n:9.4f} d {mean:+.5f} CI [{lo:+.5f}, {hi:+.5f}] "
              f"not worse {not_worse}/{n} better {better}/{n} {verdict}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    compare(sys.argv[1], sys.argv[2])
