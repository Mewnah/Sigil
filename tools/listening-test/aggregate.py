#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Aggregate the answers friends send back from an MT A/B or TTS listening test (P7).

    python tools/listening-test/aggregate.py <out>/key.json answers/*.json [--baseline <system>]
        [--min-complete 0.5] [--bootstrap 2000] [--seed 20261002] [--json results.json] [--md results.md]

(The voice-changer A/B of ``make_test.py`` keeps its own scorer, ``score.py``.)

Rules:
- answers of another test (other ``test_id``) are refused; a listener who exported twice counts
  once (the latest export); listeners who completed fewer than ``--min-complete`` of the trials are
  listed and left out;
- pairwise (MT, TTS ``pairwise``): per system pair (overall and per direction / language) the wins
  of each side, ties ("about the same" / "no preference") and "both bad", the preference share
  among decided answers with a Wilson 95 % interval, and with three or more systems a
  Bradley-Terry strength per system (ties count half a win to each side; half a pseudo-win each
  way per compared pair keeps every strength finite);
- MUSHRA-lite (TTS ``mushra``): per system the mean rating and a 95 % interval from a bootstrap
  over trials (every listener's ratings of a trial move together), per language too; a z-scored
  mean (each listener's ratings standardized) as a check on listener bias; and, against
  ``--baseline``, the mean paired difference (same listener, same trial) with its bootstrap
  interval.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import os
import random
import sys
from collections import defaultdict


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load_answers(key: dict, paths: list[str], min_complete: float):
    """{listener: answers} (latest export per listener) and the excluded listeners."""
    by_listener: dict[str, dict] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            a = json.load(f)
        if a.get("schema") != "sigil-listening-results/1":
            raise SystemExit(f"{p}: not a listening result (schema {a.get('schema')!r})")
        if a.get("test_id") != key["test_id"]:
            raise SystemExit(f"{p}: answers of another test ({a.get('test_id')} vs {key['test_id']})")
        who = (a.get("rater") or "").strip() or os.path.basename(p)
        prev = by_listener.get(who)
        if prev is None or (a.get("exported_unix_ms") or 0) >= (prev.get("exported_unix_ms") or 0):
            by_listener[who] = a
    trials = {t["id"]: t for t in key["trials"]}
    kept, excluded = {}, {}
    for who, a in by_listener.items():
        ans = a.get("answers") or {}
        done = sum(1 for tid, x in ans.items() if tid in trials and complete(key, trials[tid], x))
        frac = done / len(trials) if trials else 0.0
        if frac < min_complete:
            excluded[who] = round(frac, 3)
        else:
            kept[who] = ans
    return kept, excluded


def complete(key: dict, trial: dict, ans: dict) -> bool:
    if not ans:
        return False
    if key.get("mode") == "mushra":
        r = ans.get("ratings") or {}
        return all(isinstance(r.get(s), (int, float)) for s in trial.get("stimuli", {}))
    return ans.get("choice") in ("A", "B", "tie", "both_bad")


def bradley_terry(wins: dict[tuple[str, str], float], systems: list[str], iters: int = 500, prior: float = 0.5) -> dict[str, float]:
    """MM estimate of Bradley-Terry strengths (normalized to sum to the number of systems), with
    ``prior`` pseudo-wins each way for every pair that was compared, so a system that never won
    keeps a finite, small strength."""
    w = {}
    for i in systems:
        for j in systems:
            if i != j and (wins.get((i, j), 0.0) + wins.get((j, i), 0.0)) > 0:
                w[(i, j)] = wins.get((i, j), 0.0) + prior
    p = {s: 1.0 for s in systems}
    for _ in range(iters):
        new = {}
        for i in systems:
            num = sum(w.get((i, j), 0.0) for j in systems if j != i)
            den = sum((w.get((i, j), 0.0) + w.get((j, i), 0.0)) / (p[i] + p[j]) for j in systems if j != i and ((i, j) in w or (j, i) in w))
            new[i] = num / den if den > 0 else p[i]
        total = sum(new.values()) or 1.0
        p = {s: v * len(systems) / total for s, v in new.items()}
    return p


def pairwise(key: dict, listeners: dict[str, dict]) -> dict:
    trials = {t["id"]: t for t in key["trials"]}
    groups = defaultdict(lambda: {"wins": defaultdict(int), "ties": 0, "both_bad": 0, "answers": 0, "listeners": set()})
    bt = defaultdict(float)
    for who, ans in listeners.items():
        for tid, x in ans.items():
            t = trials.get(tid)
            c = (x or {}).get("choice")
            if t is None or c not in ("A", "B", "tie", "both_bad"):
                continue
            pair = tuple(sorted((t["A"], t["B"])))
            scope = t.get("direction") or t.get("lang") or "all"
            for g in (("all", pair), (scope, pair)):
                s = groups[g]
                s["answers"] += 1
                s["listeners"].add(who)
                if c == "tie":
                    s["ties"] += 1
                elif c == "both_bad":
                    s["both_bad"] += 1
                else:
                    s["wins"][t[c]] += 1
            if c in ("A", "B"):
                other = "B" if c == "A" else "A"
                bt[(t[c], t[other])] += 1.0
            elif c == "tie":
                bt[(t["A"], t["B"])] += 0.5
                bt[(t["B"], t["A"])] += 0.5
    rows = []
    for (scope, pair), s in sorted(groups.items(), key=lambda kv: (kv[0][0] != "all", kv[0])):
        decided = sum(s["wins"].values())
        shares = {}
        for sysname in pair:
            k = s["wins"][sysname]
            lo, hi = wilson(k, decided)
            shares[sysname] = {"preferred": k, "share": (k / decided) if decided else None, "ci95": [lo, hi]}
        rows.append({"scope": scope, "systems": list(pair), "answers": s["answers"], "listeners": len(s["listeners"]),
                     "ties": s["ties"], "both_bad": s["both_bad"], "decided": decided, "shares": shares})
    systems = sorted(key["systems"])
    strengths = bradley_terry(bt, systems) if len(systems) >= 3 else None
    return {"pairs": rows, "bradley_terry": strengths}


def mushra(key: dict, listeners: dict[str, dict], baseline: str | None, b: int, seed: int) -> dict:
    trials = {t["id"]: t for t in key["trials"]}
    # responses[trial][listener] = {system: rating}
    responses: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for who, ans in listeners.items():
        for tid, x in ans.items():
            t = trials.get(tid)
            r = (x or {}).get("ratings") or {}
            if t is None:
                continue
            got = {t["stimuli"][sid]: float(v) for sid, v in r.items() if sid in t["stimuli"] and isinstance(v, (int, float))}
            if got:
                responses[tid][who] = got
    # z-scores per listener
    per_listener = defaultdict(list)
    for tid, by in responses.items():
        for who, got in by.items():
            per_listener[who] += list(got.values())
    stats = {w: (sum(v) / len(v), (sum((x - sum(v) / len(v)) ** 2 for x in v) / max(1, len(v) - 1)) ** 0.5 or 1.0) for w, v in per_listener.items()}
    rng = random.Random(seed)

    def summarize(tids: list[str]):
        out = {}
        systems = sorted({s for tid in tids for got in responses[tid].values() for s in got})
        for sysname in systems:
            vals = [got[sysname] for tid in tids for got in responses[tid].values() if sysname in got]
            zs = [(got[sysname] - stats[w][0]) / stats[w][1] for tid in tids for w, got in responses[tid].items() if sysname in got]
            mean = sum(vals) / len(vals) if vals else None
            boots = []
            if tids and b > 0:
                for _ in range(b):
                    draw = [tids[rng.randrange(len(tids))] for _ in tids]
                    v = [got[sysname] for tid in draw for got in responses[tid].values() if sysname in got]
                    if v:
                        boots.append(sum(v) / len(v))
            boots.sort()
            ci = [boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))]] if boots else [None, None]
            out[sysname] = {"mean": mean, "ci95": ci, "ratings": len(vals), "z_mean": (sum(zs) / len(zs)) if zs else None}
        diffs = {}
        if baseline:
            for sysname in systems:
                if sysname == baseline:
                    continue
                pairs = [(tid, got[sysname] - got[baseline]) for tid in tids for got in responses[tid].values() if sysname in got and baseline in got]
                if not pairs:
                    continue
                d = [x for _, x in pairs]
                by_trial = defaultdict(list)
                for tid, x in pairs:
                    by_trial[tid].append(x)
                keys = list(by_trial)
                boots = []
                for _ in range(b):
                    draw = [keys[rng.randrange(len(keys))] for _ in keys]
                    v = [x for k in draw for x in by_trial[k]]
                    boots.append(sum(v) / len(v))
                boots.sort()
                diffs[sysname] = {"vs": baseline, "mean_diff": sum(d) / len(d), "pairs": len(d),
                                  "ci95": [boots[int(0.025 * (len(boots) - 1))], boots[int(0.975 * (len(boots) - 1))]] if boots else [None, None]}
        return out, diffs

    all_tids = sorted(responses)
    overall, diffs = summarize(all_tids)
    by_lang = {}
    for lang in sorted({trials[t]["lang"] for t in all_tids}):
        s, d = summarize([t for t in all_tids if trials[t]["lang"] == lang])
        by_lang[lang] = {"systems": s, "vs_baseline": d}
    return {"systems": overall, "vs_baseline": diffs, "by_language": by_lang}


def to_markdown(res: dict) -> str:
    out = io.StringIO()
    out.write(f"# Listening test {res['test_id']} ({res['kind']}, {res['mode']})\n\n")
    out.write(f"{len(res['listeners'])} listener(s) kept")
    if res["excluded"]:
        out.write(f"; left out (completed share below {res['min_complete']}): " + ", ".join(f"{k} ({v:.0%})" for k, v in res["excluded"].items()))
    out.write(".\n\n")
    if res["mode"] == "pairwise":
        out.write("| scope | A | B | answers | listeners | A preferred | B preferred | same | both bad |\n|---|---|---|---|---|---|---|---|---|\n")
        for r in res["pairwise"]["pairs"]:
            a, b = r["systems"]
            fa, fb = r["shares"][a], r["shares"][b]
            fmt = lambda f: "-" if f["share"] is None else f"{f['share'] * 100:.0f} % [{f['ci95'][0] * 100:.0f}-{f['ci95'][1] * 100:.0f}]"  # noqa: E731
            out.write(f"| {r['scope']} | {a} | {b} | {r['answers']} | {r['listeners']} | {fmt(fa)} | {fmt(fb)} | {r['ties']} | {r['both_bad']} |\n")
        if res["pairwise"]["bradley_terry"]:
            out.write("\nBradley-Terry strength: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(res["pairwise"]["bradley_terry"].items(), key=lambda kv: -kv[1])) + "\n")
    else:
        m = res["mushra"]
        out.write("| system | mean (0-100) [95 % CI] | ratings | z-scored mean |\n|---|---|---|---|\n")
        for k, v in sorted(m["systems"].items(), key=lambda kv: -(kv[1]["mean"] or 0)):
            ci = v["ci95"]
            out.write(f"| {k} | {v['mean']:.1f} [{ci[0]:.1f}-{ci[1]:.1f}] | {v['ratings']} | {v['z_mean']:+.2f} |\n")
        if m["vs_baseline"]:
            out.write("\n| system | vs | mean difference [95 % CI] | pairs |\n|---|---|---|---|\n")
            for k, v in m["vs_baseline"].items():
                out.write(f"| {k} | {v['vs']} | {v['mean_diff']:+.1f} [{v['ci95'][0]:+.1f}, {v['ci95'][1]:+.1f}] | {v['pairs']} |\n")
        for lang, block in m["by_language"].items():
            out.write(f"\n**{lang}**: " + ", ".join(f"{k} {v['mean']:.1f} [{v['ci95'][0]:.1f}-{v['ci95'][1]:.1f}]" for k, v in sorted(block["systems"].items(), key=lambda kv: -(kv[1]["mean"] or 0))) + "\n")
    return out.getvalue()


def aggregate(key: dict, paths: list[str], baseline: str | None = None, min_complete: float = 0.5, b: int = 2000, seed: int = 20261002) -> dict:
    if key.get("schema") != "sigil-listening-key/1":
        raise SystemExit("not a sigil-listening-key/1 file (the voice-changer A/B uses score.py)")
    listeners, excluded = load_answers(key, paths, min_complete)
    res = {"schema": "sigil-listening-aggregate/1", "test_id": key["test_id"], "kind": key["kind"], "mode": key.get("mode", "pairwise"),
           "min_complete": min_complete, "listeners": sorted(listeners), "excluded": excluded}
    if res["mode"] == "mushra":
        res["mushra"] = mushra(key, listeners, baseline, b, seed)
    else:
        res["pairwise"] = pairwise(key, listeners)
    return res


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("key")
    ap.add_argument("answers", nargs="+")
    ap.add_argument("--baseline")
    ap.add_argument("--min-complete", type=float, default=0.5)
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--json")
    ap.add_argument("--md")
    args = ap.parse_args(argv)
    with open(args.key, encoding="utf-8") as f:
        key = json.load(f)
    res = aggregate(key, args.answers, args.baseline, args.min_complete, args.bootstrap, args.seed)
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
