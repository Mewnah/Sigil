#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score STT hypotheses of speech-bench reports against the corpus manifest: WER / CER per engine,
language and condition with bootstrap 95 % intervals, the degradation of every condition against
clean (paired by utterance), and paired engine-vs-engine comparisons.

    python score/score_stt.py --manifest <corpus>/corpus.json --report <out>/report.json [--report ...]
        --out <scores dir> [--bootstrap 1000] [--seed 20261002] [--baseline <label>]
        [--failed-as empty|exclude] [--items]

Writes ``scores.json`` (``sigil-speech-scores/1``), ``scores.md`` and ``scores.csv``; ``--items``
also writes ``items.jsonl`` (normalized reference / hypothesis and edit counts per item, for error
analysis). Every STT result with item text is scored, whatever the run mode (accuracy under load
must equal accuracy idle: a difference is a bug worth seeing).

Rules (documented in README.md, "Scoring"):
- normalization and metric per language: ``normalize.py`` (``sigil-norm/1``); WER languages also
  get CER as a secondary column;
- an item the engine failed counts as an empty hypothesis (all deletions) with ``--failed-as
  empty`` (default; failing is an error) or is dropped with ``exclude``; both are counted;
- comparisons pair items by id (same audio) on the items both labels scored.
"""

from __future__ import annotations

import argparse
import csv
import io
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score.bootstrap import DEFAULT_B, DEFAULT_SEED, bootstrap_ci, paired_bootstrap  # noqa: E402
from score.common import fmt_pct, iter_results, load_json, md_table, now_ms, sha256_file, write_json, write_text  # noqa: E402
from score.normalize import describe, metric_for, normalize  # noqa: E402
from score.wer import EditCounts, score_pair, unit_for  # noqa: E402

SCORES_SCHEMA = "sigil-speech-scores/1"
CLEAN = "clean"


def manifest_index(manifest: dict) -> dict[str, dict]:
    return {it["id"]: it for it in manifest.get("items", [])}


def collect(reports: list[tuple[str, dict]], items_by_id: dict[str, dict], failed_as: str):
    """Per label: {item id: (counts, secondary char counts, failed, item)}; plus label metadata."""
    labelled: dict[str, dict] = {}
    meta: dict[str, dict] = {}
    multi_report = len(reports) > 1
    unknown = 0
    for rlabel, report in reports:
        scored_runs = {run["name"] for run, _, _ in iter_results(report, task="stt")}
        for run, result, info in iter_results(report, task="stt"):
            parts = []
            if multi_report:
                parts.append(rlabel)
            if len(scored_runs) > 1:
                parts.append(run["name"])
            parts.append(result["engine"])
            label = "/".join(parts)
            bucket = labelled.setdefault(label, {})
            meta[label] = {
                "report": rlabel,
                "run": run["name"],
                "mode": run.get("mode"),
                "engine": result["engine"],
                "backend": info.get("backend"),
                "model": info.get("model"),
                "execution_provider_used": result.get("execution_provider_used") or info.get("execution_provider_used"),
            }
            for it in result.get("items", []):
                item = items_by_id.get(it.get("item"))
                if item is None:
                    unknown += 1
                    continue
                failed = it.get("error") is not None or it.get("text") is None
                if failed and failed_as == "exclude":
                    bucket[item["id"]] = (None, None, True, item)
                    continue
                hyp = "" if failed else it.get("text", "")
                lang = item["language"]
                counts = score_pair(item["reference"], hyp, lang)
                secondary = score_pair(item["reference"], hyp, lang, "char") if unit_for(lang) == "word" else None
                bucket[item["id"]] = (counts, secondary, failed, item)
    return labelled, meta, unknown


def summarize(entries, b: int, seed: int) -> dict:
    """Corpus counts, rate and CI of a list of (counts, secondary, failed, item)."""
    total = EditCounts()
    sec = EditCounts()
    errs, units, failed, excluded = [], [], 0, 0
    has_secondary = False
    for counts, secondary, was_failed, _ in entries:
        if was_failed:
            failed += 1
        if counts is None:
            excluded += 1
            continue
        total.add(counts)
        errs.append(counts.errors)
        units.append(counts.ref_len)
        if secondary is not None:
            has_secondary = True
            sec.add(secondary)
    rate, lo, hi = bootstrap_ci(errs, units, b, seed)
    row = total.as_dict()
    row.update(rate=rate, ci95=[lo, hi], failed=failed, excluded=excluded)
    if has_secondary:
        row["cer"] = sec.rate()
    return row


def score(reports: list[tuple[str, dict]], manifest: dict, b: int = DEFAULT_B, seed: int = DEFAULT_SEED, failed_as: str = "empty", baseline: str | None = None, max_pairs: int = 8) -> tuple[dict, list[dict]]:
    items_by_id = manifest_index(manifest)
    labelled, meta, unknown = collect(reports, items_by_id, failed_as)
    rows, lang_rows, deltas, comparisons, per_item = [], [], [], [], []
    for label in sorted(labelled):
        by_lc = defaultdict(list)
        by_l = defaultdict(list)
        for item_id, entry in labelled[label].items():
            item = entry[3]
            by_lc[(item["language"], item["condition"])].append(entry)
            by_l[item["language"]].append(entry)
            counts = entry[0]
            per_item.append({
                "label": label,
                "item": item_id,
                "language": item["language"],
                "condition": item["condition"],
                "failed": entry[2],
                "reference": normalize(item["reference"], item["language"]),
                "counts": counts.as_dict() if counts is not None else None,
            })
        for (lang, cond), entries in sorted(by_lc.items()):
            row = {"label": label, **meta[label], "language": lang, "condition": cond, "metric": metric_for(lang)}
            row.update(summarize(entries, b, seed))
            rows.append(row)
        for lang, entries in sorted(by_l.items()):
            row = {"label": label, **meta[label], "language": lang, "condition": "*", "metric": metric_for(lang)}
            row.update(summarize(entries, b, seed))
            lang_rows.append(row)
        # Condition vs clean, paired by utterance.
        clean = {}
        for item_id, entry in labelled[label].items():
            if entry[3]["condition"] == CLEAN and entry[0] is not None:
                clean[entry[3]["utterance"]] = entry
        for (lang, cond), entries in sorted(by_lc.items()):
            if cond == CLEAN:
                continue
            pairs = [(e, clean[e[3]["utterance"]]) for e in entries if e[0] is not None and e[3]["utterance"] in clean]
            if not pairs:
                continue
            res = paired_bootstrap(
                [p[0][0].errors for p in pairs], [p[0][0].ref_len for p in pairs],
                [p[1][0].errors for p in pairs], [p[1][0].ref_len for p in pairs], b, seed,
            )
            deltas.append({"label": label, "language": lang, "condition": cond, "metric": metric_for(lang), "items": res["items"],
                           "rate": res["a"], "clean": res["b"], "delta": res["delta"], "ci95": res["ci95"], "p_two_sided": res["p_two_sided"]})
    # Engine vs engine on the same items.
    labels = sorted(labelled)
    if baseline is not None and baseline not in labelled:
        raise SystemExit(f"--baseline {baseline!r} is not a scored label ({', '.join(labels)})")
    if baseline is not None:
        pairs_of_labels = [(baseline, x) for x in labels if x != baseline]
    elif len(labels) <= max_pairs:
        pairs_of_labels = [(a, x) for i, a in enumerate(labels) for x in labels[i + 1:]]
    else:
        pairs_of_labels = []
    for a, x in pairs_of_labels:
        groups = defaultdict(list)
        for item_id, ea in labelled[a].items():
            ex = labelled[x].get(item_id)
            if ex is None or ea[0] is None or ex[0] is None:
                continue
            groups[(ea[3]["language"], ea[3]["condition"])].append((ea[0], ex[0]))
        for (lang, cond), pairs in sorted(groups.items()):
            res = paired_bootstrap([p[0].errors for p in pairs], [p[0].ref_len for p in pairs],
                                   [p[1].errors for p in pairs], [p[1].ref_len for p in pairs], b, seed)
            comparisons.append({
                "a_label": a, "b_label": x, "language": lang, "condition": cond, "metric": metric_for(lang),
                "items": res["items"], "a_rate": res["a"], "b_rate": res["b"], "delta": res["delta"], "ci95": res["ci95"],
                "prob_a_better": res["prob_a_better"], "p_two_sided": res["p_two_sided"],
            })
    scores = {
        "schema": SCORES_SCHEMA,
        "created_unix_ms": now_ms(),
        "normalizer": describe(),
        "bootstrap": {"resamples": b, "seed": seed, "alpha": 0.05, "method": "percentile; items resampled; paired comparisons draw the same items"},
        "failed_items": failed_as,
        "manifest": {"name": manifest.get("name"), "recipe_sha256": manifest.get("recipe_sha256")},
        "labels": {k: meta[k] for k in labels},
        "unknown_items": unknown,
        "rows": rows,
        "language_rows": lang_rows,
        "deltas_vs_clean": deltas,
        "comparisons": comparisons,
    }
    return scores, per_item


def to_markdown(scores: dict) -> str:
    out = io.StringIO()
    out.write("# STT scores\n\n")
    nz = scores["normalizer"]
    out.write(f"Normalizer `{nz['version']}` (English spelling table: {nz['english_spelling']['source']}; zh: {nz['zh_traditional_to_simplified']}). ")
    out.write(f"Rates in %, 95 % bootstrap interval in brackets ({scores['bootstrap']['resamples']} resamples, seed {scores['bootstrap']['seed']}); failed items count as `{scores['failed_items']}`.\n\n")
    by_lang = defaultdict(list)
    for r in scores["rows"]:
        by_lang[r["language"]].append(r)
    for lang in sorted(by_lang):
        rows = by_lang[lang]
        metric = rows[0]["metric"].upper()
        conds = sorted({r["condition"] for r in rows}, key=lambda c: (c != CLEAN, c))
        labels = sorted({r["label"] for r in rows})
        cell = {(r["label"], r["condition"]): r for r in rows}
        out.write(f"## {lang} ({metric})\n\n")
        table = []
        for label in labels:
            line = [label]
            for c in conds:
                r = cell.get((label, c))
                line.append(fmt_pct(r["rate"], *r["ci95"]) if r else "-")
            table.append(line)
        out.write(md_table(["engine"] + conds, table))
        out.write("\n")
        dl = [d for d in scores["deltas_vs_clean"] if d["language"] == lang]
        if dl:
            out.write(f"Degradation vs clean ({metric} points, paired by utterance):\n\n")
            dcell = {(d["label"], d["condition"]): d for d in dl}
            dconds = [c for c in conds if c != CLEAN]
            table = []
            for label in labels:
                line = [label]
                for c in dconds:
                    d = dcell.get((label, c))
                    line.append("-" if not d or d["delta"] is None else f"{d['delta'] * 100:+.1f} [{d['ci95'][0] * 100:+.1f}, {d['ci95'][1] * 100:+.1f}]")
                table.append(line)
            out.write(md_table(["engine"] + dconds, table))
            out.write("\n")
    if scores["comparisons"]:
        out.write("## Paired comparisons (A - B, lower is better for A)\n\n")
        table = []
        for c in scores["comparisons"]:
            if c["delta"] is None:
                continue
            table.append([c["language"], c["condition"], c["a_label"], c["b_label"], str(c["items"]),
                          fmt_pct(c["a_rate"]), fmt_pct(c["b_rate"]),
                          f"{c['delta'] * 100:+.2f} [{c['ci95'][0] * 100:+.2f}, {c['ci95'][1] * 100:+.2f}]",
                          f"{c['p_two_sided']:.3f}" if c["p_two_sided"] is not None else "-"])
        out.write(md_table(["lang", "condition", "A", "B", "items", "A %", "B %", "delta (pts)", "p"], table))
    return out.getvalue()


def to_csv(scores: dict) -> str:
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["label", "run", "mode", "engine", "language", "condition", "metric", "items", "failed", "excluded", "ref_units", "sub", "del", "ins", "rate", "ci_lo", "ci_hi", "cer"])
    for r in scores["rows"] + scores["language_rows"]:
        w.writerow([r["label"], r["run"], r["mode"], r["engine"], r["language"], r["condition"], r["metric"], r["items"], r["failed"], r["excluded"],
                    r["ref_units"], r["sub"], r["del"], r["ins"], r["rate"], r["ci95"][0], r["ci95"][1], r.get("cer")])
    return out.getvalue()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--report", action="append", required=True, help="speech-bench report.json (repeatable; label=path names it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--baseline", help="compare every label against this one only")
    ap.add_argument("--failed-as", choices=("empty", "exclude"), default="empty")
    ap.add_argument("--items", action="store_true", help="also write items.jsonl")
    args = ap.parse_args(argv)
    manifest = load_json(args.manifest)
    reports = []
    for spec in args.report:
        label, path = (spec.split("=", 1) if "=" in spec and not os.path.exists(spec) else (None, spec))
        report = load_json(path)
        reports.append((label or report.get("name") or os.path.basename(os.path.dirname(os.path.abspath(path))), report))
    scores, per_item = score(reports, manifest, args.bootstrap, args.seed, args.failed_as, args.baseline)
    scores["reports"] = [{"label": l, "name": r.get("name"), "spec_sha256": r.get("spec_sha256"), "rig": (r.get("environment") or {}).get("sigil_rig")} for l, r in reports]
    scores["manifest"]["file_sha256"] = sha256_file(args.manifest)
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "scores.json"), scores)
    write_text(os.path.join(args.out, "scores.md"), to_markdown(scores))
    write_text(os.path.join(args.out, "scores.csv"), to_csv(scores))
    if args.items:
        import json  # noqa: PLC0415

        write_text(os.path.join(args.out, "items.jsonl"), "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in per_item))
    print(f"{len(scores['rows'])} rows, {len(scores['comparisons'])} comparisons -> {args.out}")
    if scores["unknown_items"]:
        print(f"warning: {scores['unknown_items']} report items are not in the manifest (another corpus?)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
