#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score a blind A/B listening test built by ``make_test.py`` (plan T6: Sigil preferred in at least
60 % of the decided trials, with 5 or more listeners and 20 or more pairs each).

    python tools/listening-test/score.py listening-test-1/key.json answers/*.json [--system sigil] [--json out.json]

Per system pair: how often each side was preferred (ties excluded from the preference share,
reported separately), a Wilson 95 % interval of the share, the number of listeners and trials, and
whether the T6 criterion holds for ``--system``. Answers from another test (a different test id)
are refused.
"""

import argparse
import json
import math
import sys
from collections import defaultdict


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key")
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--system", default="sigil", help="the system whose preference T6 checks")
    ap.add_argument("--min-listeners", type=int, default=5)
    ap.add_argument("--min-pairs", type=int, default=20)
    ap.add_argument("--target", type=float, default=0.60)
    ap.add_argument("--json", help="also write the scores as JSON")
    args = ap.parse_args()
    with open(args.key, encoding="utf-8") as f:
        key = json.load(f)
    trials = {t["id"]: t for t in key["trials"]}
    stats = defaultdict(lambda: {"wins": defaultdict(int), "ties": 0, "listeners": set(), "trials": 0})
    per_listener = defaultdict(lambda: {"answered": 0})
    for path in args.answers:
        with open(path, encoding="utf-8") as f:
            a = json.load(f)
        if a.get("test_id") != key["test_id"]:
            sys.exit(f"{path}: answers of another test ({a.get('test_id')} vs {key['test_id']})")
        who = a.get("listener") or path
        for tid, ans in (a.get("answers") or {}).items():
            t = trials.get(tid)
            choice = (ans or {}).get("choice")
            if t is None or choice not in ("A", "B", "none"):
                continue
            pair = tuple(sorted((t["A"], t["B"])))
            s = stats[pair]
            s["trials"] += 1
            s["listeners"].add(who)
            per_listener[who]["answered"] += 1
            if choice == "none":
                s["ties"] += 1
            else:
                s["wins"][t[choice]] += 1
    out = {"test_id": key["test_id"], "pairs": [], "listeners": {k: v["answered"] for k, v in per_listener.items()}}
    print(f"Test {key['test_id']}: {len(per_listener)} listener(s)")
    for pair, s in sorted(stats.items()):
        decided = sum(s["wins"].values())
        row = {"systems": list(pair), "trials": s["trials"], "ties": s["ties"], "listeners": len(s["listeners"]), "shares": {}}
        print(f"\n{pair[0]} vs {pair[1]}: {s['trials']} answers from {len(s['listeners'])} listener(s), {s['ties']} no preference")
        for system in pair:
            k = s["wins"][system]
            share = k / decided if decided else 0.0
            lo, hi = wilson(k, decided)
            row["shares"][system] = {"preferred": k, "share": share, "ci95": [lo, hi]}
            print(f"  {system:<16} preferred {k:>4} / {decided:<4} = {share * 100:5.1f} %  (95 % CI {lo * 100:.1f}-{hi * 100:.1f} %)")
        if args.system in pair:
            share = row["shares"][args.system]["share"]
            enough = len(s["listeners"]) >= args.min_listeners and s["trials"] >= args.min_pairs * args.min_listeners
            ok = enough and share >= args.target
            row["t6"] = {"system": args.system, "met": ok, "enough_data": enough}
            verdict = "MET" if ok else ("not met" if enough else f"not enough data (needs {args.min_listeners}+ listeners x {args.min_pairs} pairs)")
            print(f"  T6 ({args.system} preferred >= {args.target * 100:.0f} %): {verdict}")
        out["pairs"].append(row)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
