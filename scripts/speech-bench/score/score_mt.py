#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Score MT hypotheses: chrF2 (primary, sacrebleu ``CHRF()`` defaults, the same number mt-bench
reports) with a bootstrap 95 % interval, BLEU as a secondary column (tokenizer ``zh`` for Chinese
targets, ``char`` for Japanese / Korean, ``13a`` otherwise), paired bootstrap between systems per
direction, and MetricX-24 when ``metricx.py`` scores are given (lower is better, 0-25).

Systems come from speech-bench reports (``mt_quality`` and MT speed runs; references from the MT
set named by the run) and / or mt-bench output folders (``hyps/<engine>.<src>-<tgt>.jsonl``).

    python score/score_mt.py --report <out>/report.json [--mt-bench <mt-bench out dir>] ...
        [--set corpus/mt/wmt24pp.jsonl] [--metricx metricx.jsonl] --out <dir>
        [--bootstrap 1000] [--seed 20261002] [--baseline <label>]

Writes ``mt_scores.json`` (``sigil-mt-scores/1``) and ``mt_scores.md``. ``--export-pairs <dir>``
also writes, per direction, the JSON lines ``make_mt_test.py`` (tools/listening-test) and
``metricx.py`` read: ``{id, source, target, src, ref, systems: {label: hyp}}``.
"""

from __future__ import annotations

import argparse
import glob
import io
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score.bootstrap import DEFAULT_B, DEFAULT_SEED, bootstrap_stat_ci, paired_bootstrap_stat  # noqa: E402
from score.common import iter_results, load_json, load_jsonl, md_table, now_ms, write_json, write_text  # noqa: E402

MT_SCORES_SCHEMA = "sigil-mt-scores/1"


def bleu_tokenizer(target: str) -> str:
    base = re.split(r"[-_]", target.lower())[0]
    if base in ("zh", "yue"):
        return "zh"
    if base in ("ja", "ko", "th"):
        return "char"
    return "13a"


def _chrf():
    from sacrebleu.metrics import CHRF  # noqa: PLC0415

    return CHRF()


def chrf_stats(hyps: list[str], refs: list[list[str]]):
    """Per-line sufficient statistics [lines, k] of sacrebleu's chrF2 (best reference per line)."""
    metric = _chrf()
    width = max(len(r) for r in refs) if refs else 1
    streams = [[r[i] if i < len(r) else r[0] for r in refs] for i in range(width)]
    return metric, metric._extract_corpus_statistics(hyps, streams)  # noqa: SLF001 - sacrebleu 2.x public-in-practice API


def chrf_from_stats(metric, summed) -> float:
    return float(metric._compute_score_from_stats([int(round(x)) for x in summed]).score)  # noqa: SLF001


def bleu(hyps: list[str], refs: list[list[str]], target: str) -> float | None:
    from sacrebleu.metrics import BLEU  # noqa: PLC0415

    if not hyps:
        return None
    width = max(len(r) for r in refs)
    streams = [[r[i] if i < len(r) else r[0] for r in refs] for i in range(width)]
    return float(BLEU(tokenize=bleu_tokenizer(target)).corpus_score(hyps, streams).score)


def load_set(path: str) -> dict[str, dict]:
    out = {}
    for row in load_jsonl(path):
        refs = list(row.get("refs") or [])
        if row.get("ref"):
            refs.insert(0, row["ref"])
        out[row["id"]] = {**row, "refs": refs}
    return out


def systems_from_reports(reports: list[tuple[str, dict]], sets: dict[str, dict], base_dirs: dict[str, str]) -> dict[str, dict[str, dict]]:
    """label -> {line id: {direction, hyp, failed, src, refs}}."""
    out: dict[str, dict[str, dict]] = {}
    multi = len(reports) > 1
    for rlabel, report in reports:
        runs = {run["name"] for run, _, _ in iter_results(report, task="mt")}
        spec_runs = {r["name"]: r for r in (report.get("spec") or {}).get("runs", [])}
        for run, result, _ in iter_results(report, task="mt"):
            set_path = ((spec_runs.get(run["name"]) or {}).get("mt") or {}).get("set")
            lines = None
            if set_path:
                p = set_path if os.path.isabs(set_path) else os.path.join(base_dirs.get(rlabel, "."), set_path)
                if os.path.isfile(p):
                    lines = sets.setdefault(os.path.abspath(p), load_set(p))
            label = "/".join(([rlabel] if multi else []) + ([run["name"]] if len(runs) > 1 else []) + [result["engine"]])
            bucket = out.setdefault(label, {})
            for it in result.get("items", []):
                line = None
                for s in ([lines] if lines else []) + list(sets.values()):
                    if s and it["item"] in s:
                        line = s[it["item"]]
                        break
                if line is None:
                    continue
                failed = it.get("error") is not None or it.get("text") is None
                bucket[it["item"]] = {"direction": it.get("direction") or f"{line['source']}-{line['target']}", "hyp": "" if failed else it["text"],
                                      "failed": failed, "src": line["src"], "refs": line["refs"], "target": line["target"], "source": line["source"]}
    return out


def systems_from_mt_bench(folders: list[str]) -> dict[str, dict[str, dict]]:
    out: dict[str, dict[str, dict]] = {}
    for folder in folders:
        prefix = os.path.basename(os.path.normpath(folder)) if len(folders) > 1 else ""
        for path in sorted(glob.glob(os.path.join(folder, "hyps", "*.jsonl"))):
            stem = os.path.basename(path)[: -len(".jsonl")]
            engine, _, direction = stem.rpartition(".")
            source, _, target = direction.partition("-")
            label = f"{prefix}/{engine}" if prefix else engine
            bucket = out.setdefault(label, {})
            for row in load_jsonl(path):
                hyp = row.get("hyp")
                bucket[row["id"]] = {"direction": direction, "hyp": hyp or "", "failed": hyp is None, "src": row.get("src", ""),
                                     "refs": [row["ref"]] if isinstance(row.get("ref"), str) else list(row.get("ref") or []),
                                     "target": target, "source": source}
    return out


def score(systems: dict[str, dict[str, dict]], metricx: dict[tuple[str, str], float] | None = None, b: int = DEFAULT_B, seed: int = DEFAULT_SEED, baseline: str | None = None, max_pairs: int = 10) -> dict:
    rows, comparisons = [], []
    by_dir: dict[str, dict[str, dict[str, dict]]] = defaultdict(dict)
    for label, lines in systems.items():
        for line_id, row in lines.items():
            by_dir[row["direction"]].setdefault(label, {})[line_id] = row
    for direction in sorted(by_dir):
        labels = sorted(by_dir[direction])
        stats_by_label = {}
        for label in labels:
            lines = by_dir[direction][label]
            ids = sorted(lines)
            hyps = [lines[i]["hyp"] for i in ids]
            refs = [lines[i]["refs"] for i in ids]
            target = lines[ids[0]]["target"]
            metric, st = chrf_stats(hyps, refs)
            stats_by_label[label] = (ids, st, metric)
            value, lo, hi = bootstrap_stat_ci(lambda s, m=metric: chrf_from_stats(m, s), st, b, seed)
            row = {"label": label, "direction": direction, "lines": len(ids), "failed": sum(1 for i in ids if lines[i]["failed"]),
                   "chrf": value, "chrf_ci95": [lo, hi], "bleu": bleu(hyps, refs, target), "bleu_tokenize": bleu_tokenizer(target)}
            if metricx:
                vals = [metricx[(label, i)] for i in ids if (label, i) in metricx]
                if vals:
                    mx = [[v, 1.0] for v in vals]
                    m, mlo, mhi = bootstrap_stat_ci(lambda s: s[0] / s[1], mx, b, seed)
                    row.update(metricx=m, metricx_ci95=[mlo, mhi], metricx_lines=len(vals))
            rows.append(row)
        if baseline is not None:
            pairs = [(baseline, x) for x in labels if x != baseline and baseline in labels]
        elif len(labels) <= max_pairs:
            pairs = [(a, x) for i, a in enumerate(labels) for x in labels[i + 1:]]
        else:
            pairs = []
        for a, x in pairs:
            ida, sta, metric = stats_by_label[a]
            idx_b = {i: k for k, i in enumerate(stats_by_label[x][0])}
            common = [(k, idx_b[i]) for k, i in enumerate(ida) if i in idx_b]
            if not common:
                continue
            sa = [sta[k] for k, _ in common]
            sb = [stats_by_label[x][1][j] for _, j in common]
            res = paired_bootstrap_stat(lambda s, m=metric: chrf_from_stats(m, s), sa, sb, b, seed, higher_is_better=True)
            comp = {"direction": direction, "a_label": a, "b_label": x, "metric": "chrf", "lines": res["items"], "a": res["a"], "b": res["b"],
                    "delta": res["delta"], "ci95": res["ci95"], "prob_a_better": res["prob_a_better"], "p_two_sided": res["p_two_sided"]}
            comparisons.append(comp)
            if metricx:
                ids_common = [ida[k] for k, _ in common]
                pa = [(metricx.get((a, i)), metricx.get((x, i))) for i in ids_common]
                pa = [(u, v) for u, v in pa if u is not None and v is not None]
                if pa:
                    res = paired_bootstrap_stat(lambda s: s[0] / s[1], [[u, 1.0] for u, _ in pa], [[v, 1.0] for _, v in pa], b, seed, higher_is_better=False)
                    comparisons.append({"direction": direction, "a_label": a, "b_label": x, "metric": "metricx", "lines": res["items"], "a": res["a"], "b": res["b"],
                                        "delta": res["delta"], "ci95": res["ci95"], "prob_a_better": res["prob_a_better"], "p_two_sided": res["p_two_sided"]})
    import sacrebleu  # noqa: PLC0415

    return {
        "schema": MT_SCORES_SCHEMA,
        "created_unix_ms": now_ms(),
        "chrf": f"sacrebleu {sacrebleu.__version__} CHRF() (chrF2: char order 6, word order 0, beta 2), best reference per line",
        "bootstrap": {"resamples": b, "seed": seed, "alpha": 0.05, "method": "percentile; lines resampled; paired comparisons draw the same lines"},
        "metricx": "MetricX-24 hybrid large (bf16), reference-based, lower is better (0-25)" if metricx else None,
        "rows": rows,
        "comparisons": comparisons,
    }


def to_markdown(s: dict) -> str:
    out = io.StringIO()
    out.write("# MT scores\n\n")
    out.write(f"{s['chrf']}; 95 % bootstrap intervals ({s['bootstrap']['resamples']} resamples). Failed lines count as empty translations.\n\n")
    has_mx = any("metricx" in r for r in s["rows"])
    header = ["direction", "system", "lines", "failed", "chrF2 [95 % CI]", "BLEU"] + (["MetricX-24 (lower better)"] if has_mx else [])
    table = []
    for r in s["rows"]:
        line = [r["direction"], r["label"], str(r["lines"]), str(r["failed"]),
                f"{r['chrf']:.2f} [{r['chrf_ci95'][0]:.2f}-{r['chrf_ci95'][1]:.2f}]" if r["chrf"] is not None else "-",
                f"{r['bleu']:.2f} ({r['bleu_tokenize']})" if r["bleu"] is not None else "-"]
        if has_mx:
            line.append(f"{r['metricx']:.2f} [{r['metricx_ci95'][0]:.2f}-{r['metricx_ci95'][1]:.2f}]" if "metricx" in r else "-")
        table.append(line)
    out.write(md_table(header, table))
    if s["comparisons"]:
        out.write("\n## Paired comparisons (A - B)\n\n")
        table = [[c["direction"], c["metric"], c["a_label"], c["b_label"], str(c["lines"]), f"{c['delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]",
                  f"{c['prob_a_better']:.2f}" if c["prob_a_better"] is not None else "-", f"{c['p_two_sided']:.3f}"] for c in s["comparisons"]]
        out.write(md_table(["direction", "metric", "A", "B", "lines", "delta", "P(A better)", "p"], table))
    return out.getvalue()


def export_pairs(systems: dict[str, dict[str, dict]], out_dir: str) -> list[str]:
    """Per direction: {id, source, target, src, ref, systems: {label: hyp}} lines (blind A/B input)."""
    by_dir = defaultdict(lambda: defaultdict(dict))
    for label, lines in systems.items():
        for line_id, row in lines.items():
            d = by_dir[row["direction"]][line_id]
            d.update(id=line_id, source=row["source"], target=row["target"], src=row["src"], ref=row["refs"][0] if row["refs"] else "")
            d.setdefault("systems", {})[label] = None if row["failed"] else row["hyp"]
    written = []
    import json  # noqa: PLC0415

    for direction, lines in sorted(by_dir.items()):
        path = os.path.join(out_dir, f"pairs.{direction}.jsonl")
        write_text(path, "".join(json.dumps(lines[i], ensure_ascii=False) + "\n" for i in sorted(lines)))
        written.append(path)
    return written


def load_metricx(path: str) -> dict[tuple[str, str], float]:
    """metricx.py output: JSON lines {system, id, score}."""
    return {(r["system"], r["id"]): float(r["score"]) for r in load_jsonl(path) if r.get("score") is not None}


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="append", default=[], help="speech-bench report.json (repeatable)")
    ap.add_argument("--mt-bench", action="append", default=[], help="mt-bench output folder with hyps/ (repeatable)")
    ap.add_argument("--set", action="append", default=[], help="MT set JSON lines (references for report items)")
    ap.add_argument("--metricx", help="metricx.py scores (JSON lines {system, id, score})")
    ap.add_argument("--out", required=True)
    ap.add_argument("--export-pairs", help="also write pairs.<direction>.jsonl here")
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--baseline")
    args = ap.parse_args(argv)
    if not args.report and not args.mt_bench:
        ap.error("give --report and / or --mt-bench")
    sets = {os.path.abspath(p): load_set(p) for p in args.set}
    reports, base_dirs = [], {}
    for path in args.report:
        r = load_json(path)
        label = r.get("name") or os.path.basename(os.path.dirname(os.path.abspath(path)))
        reports.append((label, r))
        base_dirs[label] = os.path.dirname(os.path.abspath(path))
    systems = systems_from_reports(reports, sets, base_dirs)
    systems.update(systems_from_mt_bench(args.mt_bench))
    if not systems:
        print("no MT hypotheses found", file=sys.stderr)
        return 1
    mx = load_metricx(args.metricx) if args.metricx else None
    s = score(systems, mx, args.bootstrap, args.seed, args.baseline)
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "mt_scores.json"), s)
    write_text(os.path.join(args.out, "mt_scores.md"), to_markdown(s))
    if args.export_pairs:
        for p in export_pairs(systems, args.export_pairs):
            print("pairs:", p)
    print(f"{len(s['rows'])} rows, {len(s['comparisons'])} comparisons -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
