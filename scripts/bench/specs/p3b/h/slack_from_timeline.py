#!/usr/bin/env python
"""P3B-H measurement (3): the slack the output queue keeps before the render deadline, from a paced or
contention run's timeline CSV (`<run>.timeline.csv`, written with `output.write_timeline`).

Each tick records the queue level after its push (`queued_out_ms`) at time `t_ms`. The device drains
the queue in real time, so the level just before the next push is the previous level minus the time
between the two pushes. Its smallest values are how close playback came to running dry: the slack
the fixed 35 ms prefill never used is `min(level_before)` over a steady stretch of the run.

    python slack_from_timeline.py <report dir> [--skip-s 25] [--json out.json]

Prints, per timeline, the percentiles (min, p1, p5, p50, p99, max) of the level before the push in
ms, the push gap (p50, max) and how many ticks were starved (level before <= 0).
"""
import argparse
import csv
import json
import os
import statistics
import sys


def pct(v, q):
    if not v:
        return float("nan")
    v = sorted(v)
    k = min(len(v) - 1, max(0, int(round((len(v) - 1) * q / 100.0))))
    return v[k]


def analyze(path, skip_s):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                rows.append((float(r["t_ms"]), float(r["queued_out_ms"]), float(r.get("dropped_ms") or 0.0)))
            except (KeyError, ValueError):
                continue
    before, gaps = [], []
    for (t0, q0, _), (t1, _q1, d1) in zip(rows, rows[1:]):
        if t1 / 1000.0 < skip_s:
            continue
        dt = t1 - t0
        gaps.append(dt)
        before.append(q0 - dt)
    starved = sum(1 for b in before if b <= 0.0)
    return {
        "ticks": len(before),
        "level_before_ms": {k: round(pct(before, q), 2) for k, q in (("min", 0), ("p1", 1), ("p5", 5), ("p50", 50), ("p99", 99), ("max", 100))},
        "push_gap_ms": {"p50": round(pct(gaps, 50), 2), "p99": round(pct(gaps, 99), 2), "max": round(max(gaps), 2) if gaps else None},
        "starved_ticks": starved,
        "mean_level_before_ms": round(statistics.fmean(before), 2) if before else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--skip-s", type=float, default=25.0)
    ap.add_argument("--json")
    a = ap.parse_args()
    out = {}
    for root, _dirs, files in os.walk(a.dir):
        for name in sorted(files):
            if name.endswith(".timeline.csv") and not name.startswith("offline"):
                rel = os.path.relpath(os.path.join(root, name), a.dir)
                out[rel] = analyze(os.path.join(root, name), a.skip_s)
    for k, v in out.items():
        lv = v["level_before_ms"]
        print(f"{k}: {v['ticks']} ticks, level before the push min {lv['min']} p1 {lv['p1']} p5 {lv['p5']} p50 {lv['p50']} max {lv['max']} ms; "
              f"gap p50 {v['push_gap_ms']['p50']} max {v['push_gap_ms']['max']} ms; starved {v['starved_ticks']}")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
