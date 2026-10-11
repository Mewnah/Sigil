#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tables of the MT-Europe campaign (``specs/mt_europe.json``): per direction and per language pair, NLLB-200 600M beam 4
against Hy-MT2 1.8B on the CPU.

    python campaign/mt_europe_table.py --report <out>/report.json --set <corpus>/mt/sigil-casual.jsonl [--set ...]
        [--a nllb-beam4] [--b hy-mt2-1.8b-cpu] [--bootstrap 1000] [--json <file>] [--md <file>]

The score of a direction is the one ``mt_table.py`` and the tables of ``SPEECH_BENCHMARKS.md`` use: the chrF2 of each set that
has the direction (sacrebleu defaults, one corpus score per set), averaged weighted by the number of lines of the set. The
paired bootstrap resamples the lines inside every set (the same draw for both engines) and recomputes that weighted mean; the
interval is the 2.5 / 97.5 percentile. A language pair (``en-X`` and ``X-en`` together) is the weighted mean over all of its
direction-and-set cells, bootstrapped the same way. Latency is p50 / p95 of the translate calls (warm-up lines are not in the
report), over all lines of the direction and over the casual caption lines alone.

The decision rule used in the docs: a language pair goes to A when the 95 % interval of A - B over both directions stays
above zero; a pair that is only probably ahead, or tied, keeps B.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from score.bootstrap import DEFAULT_B, DEFAULT_SEED  # noqa: E402
from score.common import iter_results, load_json  # noqa: E402
from score.score_mt import chrf_from_stats, chrf_stats, load_set  # noqa: E402

CASUAL = "sigil-casual"


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def collect(report: dict, sets: dict[str, dict]) -> dict:
    """engine -> direction -> set -> [{id, hyp, refs, latency, failed}] over every mt run of the report."""
    out: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for run, result, _ in iter_results(report, task="mt"):
        for it in result.get("items", []):
            line = next((s[it["item"]] for s in sets.values() if it["item"] in s), None)
            if line is None:
                continue
            failed = it.get("error") is not None or it.get("text") is None
            out[result["engine"]][it.get("direction") or f"{line['source']}-{line['target']}"][line.get("set") or run["name"]].append({
                "id": it["item"], "hyp": "" if failed else it["text"], "refs": line["refs"], "latency": it.get("latency_ms"), "failed": failed,
            })
    return out


class SetStats:
    """The per-line chrF statistics of one engine on one set of one direction (rows aligned by line id)."""

    def __init__(self, rows: list[dict]):
        self.metric, stats = chrf_stats([r["hyp"] for r in rows], [r["refs"] for r in rows])
        self.stats = np.asarray(stats, dtype=np.float64)

    @property
    def lines(self) -> int:
        return int(self.stats.shape[0])

    def score(self, idx=None) -> float:
        s = self.stats if idx is None else self.stats[idx]
        return chrf_from_stats(self.metric, s.sum(axis=0))


def aligned(rows_a: list[dict], rows_b: list[dict]) -> tuple[list[dict], list[dict]]:
    by_b = {r["id"]: r for r in rows_b}
    by_a = {r["id"]: r for r in rows_a}
    ids = sorted(i for i in by_a if i in by_b)
    return [by_a[i] for i in ids], [by_b[i] for i in ids]


def weighted(values: list[float], weights: list[int]) -> float:
    return sum(v * w for v, w in zip(values, weights)) / sum(weights)


def compare(cells: list[tuple[SetStats, SetStats]], boot: int, rng) -> dict:
    """A, B and A - B (line-weighted mean over the cells) with the stratified paired bootstrap interval."""
    weights = [c[0].lines for c in cells]
    a = weighted([c[0].score() for c in cells], weights)
    b = weighted([c[1].score() for c in cells], weights)
    out = {"lines": sum(weights), "a": a, "b": b, "delta": a - b, "ci95": [None, None], "prob_a_better": None}
    if boot > 0:
        deltas = np.empty(boot)
        for k in range(boot):
            va, vb = [], []
            for ca, cb in cells:
                idx = rng.integers(0, ca.lines, size=ca.lines)
                va.append(ca.score(idx))
                vb.append(cb.score(idx))
            deltas[k] = weighted(va, weights) - weighted(vb, weights)
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out.update(ci95=[float(lo), float(hi)], prob_a_better=float(np.mean(deltas > 0)))
    return out


def latency(data: dict, engine: str, direction: str, only_set: str | None = None) -> dict:
    lat = [r["latency"] for s, rows in data.get(engine, {}).get(direction, {}).items() if only_set in (None, s) for r in rows if r["latency"] is not None]
    return {"lines": len(lat), "p50_ms": percentile(lat, 0.5), "p95_ms": percentile(lat, 0.95)}


def pair_of(direction: str) -> str | None:
    """The language pair with English of ``en-X`` / ``X-en`` (the X); None for a direction without English."""
    s, _, t = direction.partition("-")
    if s == "en" and t:
        return t
    if t == "en" and s:
        return s
    return None


def analyse(data: dict, a: str, b: str, boot: int, seed: int) -> dict:
    directions = sorted({d for e in data.values() for d in e})
    rng = np.random.default_rng(seed)
    cells: dict[tuple[str, str], tuple[SetStats, SetStats]] = {}
    for d in directions:
        for s in sorted(set(data.get(a, {}).get(d, {})) & set(data.get(b, {}).get(d, {}))):
            ra, rb = aligned(data[a][d][s], data[b][d][s])
            if ra:
                cells[(d, s)] = (SetStats(ra), SetStats(rb))
    rows = []
    for d in directions:
        mine = {s: c for (dd, s), c in cells.items() if dd == d}
        if not mine:
            continue
        row = {"direction": d, "sets": {s: {"lines": c[0].lines, a: c[0].score(), b: c[1].score()} for s, c in mine.items()},
               **compare(list(mine.values()), boot, rng)}
        row["latency"] = {e: {"all": latency(data, e, d), "casual": latency(data, e, d, CASUAL)} for e in (a, b)}
        row["failed"] = {e: sum(1 for rs in data.get(e, {}).get(d, {}).values() for r in rs if r["failed"]) for e in (a, b)}
        rows.append(row)
    pairs = []
    for p in sorted({pair_of(d) for d in directions} - {None}):
        mine = [c for (d, _), c in cells.items() if pair_of(d) == p]
        dirs = sorted({d for (d, _) in cells if pair_of(d) == p})
        if mine:
            pairs.append({"pair": f"en<->{p}", "directions": dirs, **compare(mine, boot, rng)})
    return {"a": a, "b": b, "bootstrap": {"resamples": boot, "seed": seed}, "directions": rows, "pairs": pairs}


def fmt(x, nd=1):
    return "-" if x is None else f"{x:.{nd}f}"


def interval(r: dict) -> str:
    return f"{r['delta']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]" if r["ci95"][0] is not None else f"{r['delta']:+.1f}"


def to_markdown(res: dict) -> str:
    a, b = res["a"], res["b"]
    out = [f"chrF2 of a direction = the line-weighted mean of its sets' chrF2; A = {a}, B = {b}; 95 % paired bootstrap inside every set "
           f"({res['bootstrap']['resamples']} resamples).\n"]
    out.append("Per language pair (both directions with English):\n")
    out.append(f"| pair | lines | {a} | {b} | A - B [95 % CI] | P(A better) |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for r in res["pairs"]:
        out.append(f"| {r['pair']} | {r['lines']} | {r['a']:.1f} | {r['b']:.1f} | {interval(r)} | {fmt(r['prob_a_better'], 2)} |")
    out.append("\nPer direction:\n")
    out.append(f"| direction | lines | {a} | {b} | A - B [95 % CI] | P(A better) | p50 / p95 ms A (casual lines) | p50 / p95 ms B (casual lines) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    def lat(r, e):
        x = r["latency"][e]
        cas = f" ({fmt(x['casual']['p50_ms'], 0)} / {fmt(x['casual']['p95_ms'], 0)})" if x["casual"]["lines"] else ""
        return f"{fmt(x['all']['p50_ms'], 0)} / {fmt(x['all']['p95_ms'], 0)}{cas}"

    for r in res["directions"]:
        out.append(f"| {r['direction']} | {r['lines']} | {r['a']:.1f} | {r['b']:.1f} | {interval(r)} | {fmt(r['prob_a_better'], 2)} | {lat(r, a)} | {lat(r, b)} |")
    sets = sorted({s for r in res["directions"] for s in r["sets"]})
    out.append("\nPer set (chrF2, A / B):\n")
    out.append("| direction | " + " | ".join(sets) + " |")
    out.append("|---|" + "---:|" * len(sets))
    for r in res["directions"]:
        cells = [f"{r['sets'][s][a]:.1f} / {r['sets'][s][b]:.1f}" if s in r["sets"] else "-" for s in sets]
        out.append(f"| {r['direction']} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", required=True, help="the mt_quality report.json")
    ap.add_argument("--set", action="append", default=[], required=True, help="MT set JSON lines (references); repeatable")
    ap.add_argument("--a", default="nllb-beam4")
    ap.add_argument("--b", default="hy-mt2-1.8b-cpu")
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args(argv)
    sets = {os.path.abspath(p): load_set(p) for p in args.set}
    data = collect(load_json(args.report), sets)
    if args.a not in data or args.b not in data:
        print(f"engines {args.a} / {args.b} not both in the report (found: {', '.join(sorted(data))})", file=sys.stderr)
        return 1
    res = analyse(data, args.a, args.b, args.bootstrap, args.seed)
    md = to_markdown(res)
    print(md)
    if args.json:
        with open(args.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
    if args.md:
        with open(args.md, "w", encoding="utf-8", newline="\n") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
