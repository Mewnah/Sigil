#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a blind MT A/B test (P7 translation check) for friends who read both languages.

Input: the ``pairs.<direction>.jsonl`` files ``scripts/speech-bench/score/score_mt.py
--export-pairs`` writes (one line per source line: {id, source, target, src, ref, systems: {label:
translation}}).

    python tools/listening-test/make_mt_test.py --pairs pairs.ja-en.jsonl [--pairs ...]
        --systems hy-mt2-1.8b,bergamot [--per-direction 30] [--seed 7] [--show-reference]
        [--context corpus/mt/bsd.jsonl] [--title "..."] --out mt-ab-1

With more than two ``--systems`` every pair of them gets ``--per-direction`` trials. Lines where a
system failed, or where both translations are identical (after collapsing whitespace), carry no
preference and are skipped; ``key.json`` counts them (``selection``). The A / B side and the trial
order are drawn from ``--seed`` (else a random seed, kept only in key.json); the page shuffles the
order again per listener. ``--context`` shows the previous source line of the same document
(sets whose ``doc`` is ``<document>#<position>``: WMT24++, BSD).

Give ``<out>/test/`` to the listeners (open ``index.html``; no network, no install); keep
``<out>/key.json``. Score the returned ``answers-*.json`` with ``aggregate.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from listening_common import new_test_id, ranked, rng_for, write_test  # noqa: E402

DEFAULT_INSTRUCTIONS = ("You will see a line in one language and two translations of it. Pick the translation that "
                        "keeps the meaning best and reads most naturally for a casual chat or a live caption.")


def load_pairs(paths: list[str]) -> list[dict]:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows += [json.loads(l) for l in f if l.strip()]
    return rows


def context_index(path: str | None) -> dict[str, str]:
    """line id -> previous source line of the same document."""
    if not path:
        return {}
    docs: dict[str, list[tuple[float, str, str]]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            doc = r.get("doc") or ""
            m = re.match(r"^(.*)#(\d+)$", doc)
            if not m:
                continue
            key = f"{r['source']}-{r['target']}|{m.group(1)}"
            docs.setdefault(key, []).append((float(m.group(2)), r["id"], r["src"]))
    out = {}
    for rows in docs.values():
        rows.sort()
        for prev, cur in zip(rows, rows[1:]):
            out[cur[1]] = prev[2]
    return out


def same(a: str, b: str) -> bool:
    return " ".join(a.split()) == " ".join(b.split())


def build(rows: list[dict], systems: list[str], per_direction: int, seed: int | None, show_reference: bool, contexts: dict[str, str], title: str, instructions: str):
    rng, seed = rng_for(seed)
    trials, key_trials, selection = [], [], {}
    directions = sorted({f"{r['source']}-{r['target']}" for r in rows})
    for a_sys, b_sys in itertools.combinations(systems, 2):
        for d in directions:
            pool, identical, failed = [], 0, 0
            for r in rows:
                if f"{r['source']}-{r['target']}" != d:
                    continue
                ha, hb = (r.get("systems") or {}).get(a_sys), (r.get("systems") or {}).get(b_sys)
                if not ha or not hb:
                    failed += 1
                    continue
                if same(ha, hb):
                    identical += 1
                    continue
                pool.append(r)
            chosen = ranked(pool, seed, "mt-ab", a_sys, b_sys, d)[:per_direction]
            selection[f"{a_sys} vs {b_sys} | {d}"] = {"eligible": len(pool), "chosen": len(chosen), "identical_skipped": identical, "failed_skipped": failed}
            for r in chosen:
                flip = rng.random() < 0.5
                sa, sb = (b_sys, a_sys) if flip else (a_sys, b_sys)
                trials.append({"direction": d, "source": r["src"], "context": contexts.get(r["id"]), "reference": r.get("ref") if show_reference else None,
                               "a": r["systems"][sa], "b": r["systems"][sb], "_key": {"item": r["id"], "direction": d, "A": sa, "B": sb}})
    rng.shuffle(trials)
    page_trials = []
    for n, t in enumerate(trials, 1):
        tid = "t%03d" % n
        k = t.pop("_key")
        page_trials.append({"id": tid, **{x: t[x] for x in ("direction", "source", "context", "reference", "a", "b")}})
        key_trials.append({"id": tid, **k})
    test_id = new_test_id()
    manifest = {"schema": "sigil-listening-mt/1", "test_id": test_id, "title": title, "instructions": instructions,
                "question": "Which translation is better?", "show_reference": show_reference, "allow_both_bad": True, "trials": page_trials}
    key = {"schema": "sigil-listening-key/1", "test_id": test_id, "kind": "mt", "mode": "pairwise", "title": title, "seed": seed,
           "systems": systems, "selection": selection, "trials": key_trials}
    return manifest, key


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", action="append", required=True)
    ap.add_argument("--systems", required=True, help="comma-separated labels (two or more)")
    ap.add_argument("--per-direction", type=int, default=30)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--show-reference", action="store_true")
    ap.add_argument("--context", help="MT set JSON lines with doc ids (shows the previous line)")
    ap.add_argument("--title", default="Translation check")
    ap.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    if len(systems) < 2:
        ap.error("--systems needs two or more labels")
    rows = load_pairs(args.pairs)
    known = set().union(*[set((r.get("systems") or {}).keys()) for r in rows]) if rows else set()
    missing = [s for s in systems if s not in known]
    if missing:
        ap.error(f"unknown systems {missing}; the pairs files have {sorted(known)}")
    manifest, key = build(rows, systems, args.per_direction, args.seed, args.show_reference, context_index(args.context), args.title, args.instructions)
    if not manifest["trials"]:
        print("no trial: every line was identical or failed for these systems", file=sys.stderr)
        return 1
    key["sources"] = [os.path.basename(p) for p in args.pairs]
    test_dir = write_test(args.out, "mt-ab.html", manifest, key)
    print(f"{len(manifest['trials'])} trials -> {test_dir} (give this folder to readers); key -> {os.path.join(args.out, 'key.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
