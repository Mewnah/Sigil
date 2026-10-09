#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Summarize TTS runs of speech-bench reports per engine and language: time to first audio, RTF,
failures, and the automatic pre-screens the run measured (intelligibility = WER / CER of the
back-transcription against the prompt with ``sigil-norm/1``, DNSMOS P.835, CAM++ speaker cosine
for cloning). The listening test (tools/listening-test) is the verdict; these numbers screen out
broken engines before friends spend time on them.

    python score/score_tts.py --report <out>/report.json [--report ...] --prompts corpus/tts/prompts.jsonl --out <dir>
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score.bootstrap import DEFAULT_B, DEFAULT_SEED, bootstrap_ci  # noqa: E402
from score.common import fmt_num, fmt_pct, iter_results, load_json, load_jsonl, md_table, now_ms, percentile, write_json, write_text  # noqa: E402
from score.normalize import describe, metric_for  # noqa: E402
from score.wer import score_pair  # noqa: E402

TTS_SCORES_SCHEMA = "sigil-tts-scores/1"


def _mean(values):
    v = [x for x in values if x is not None]
    return sum(v) / len(v) if v else None


def score(reports: list[tuple[str, dict]], prompts: dict[str, dict], b: int = DEFAULT_B, seed: int = DEFAULT_SEED) -> dict:
    groups = defaultdict(list)
    multi = len(reports) > 1
    for rlabel, report in reports:
        runs = {run["name"] for run, _, _ in iter_results(report, task="tts")}
        for run, result, info in iter_results(report, task="tts"):
            label = "/".join(([rlabel] if multi else []) + ([run["name"]] if len(runs) > 1 else []) + [result["engine"]])
            for it in result.get("items", []):
                groups[(label, it.get("language") or (prompts.get(it["item"]) or {}).get("lang", "?"))].append((it, info, run))
    rows = []
    for (label, lang), entries in sorted(groups.items()):
        items = [e[0] for e in entries]
        ok = [i for i in items if i.get("error") is None]
        errs, units = [], []
        for i in ok:
            p = prompts.get(i["item"])
            if p is None or i.get("asr_text") is None:
                continue
            c = score_pair(p["text"], i["asr_text"], lang)
            errs.append(c.errors)
            units.append(c.ref_len)
        rate, lo, hi = bootstrap_ci(errs, units, b, seed)
        rows.append({
            "label": label,
            "language": lang,
            "execution_provider_used": entries[0][1].get("execution_provider_used"),
            "items": len(items),
            "errors": len(items) - len(ok),
            "ttfa_ms_p50": percentile([i.get("ttfa_ms") for i in ok], 50),
            "ttfa_ms_p95": percentile([i.get("ttfa_ms") for i in ok], 95),
            "rtf_p50": percentile([i.get("rtf") for i in ok], 50),
            "rtf_p95": percentile([i.get("rtf") for i in ok], 95),
            "audio_s": sum(i.get("audio_s") or 0 for i in ok),
            "intelligibility_metric": metric_for(lang),
            "intelligibility": rate,
            "intelligibility_ci95": [lo, hi],
            "intelligibility_items": len(errs),
            "dnsmos_ovrl": _mean([i.get("dnsmos_ovrl") for i in ok]),
            "dnsmos_sig": _mean([i.get("dnsmos_sig") for i in ok]),
            "dnsmos_bak": _mean([i.get("dnsmos_bak") for i in ok]),
            "speaker_cosine": _mean([i.get("speaker_cosine") for i in ok]),
        })
    return {"schema": TTS_SCORES_SCHEMA, "created_unix_ms": now_ms(), "normalizer": describe(),
            "bootstrap": {"resamples": b, "seed": seed}, "rows": rows}


def to_markdown(s: dict) -> str:
    out = io.StringIO()
    out.write("# TTS pre-screen\n\nTime to first audio and RTF are percentiles of successful renderings; intelligibility is the back-transcription's WER / CER against the prompt (95 % bootstrap interval).\n\n")
    table = []
    for r in s["rows"]:
        table.append([r["language"], r["label"], r["execution_provider_used"] or "-", f"{r['items'] - r['errors']}/{r['items']}",
                      fmt_num(r["ttfa_ms_p50"], 0), fmt_num(r["ttfa_ms_p95"], 0), fmt_num(r["rtf_p50"], 3),
                      f"{r['intelligibility_metric'].upper()} " + fmt_pct(r["intelligibility"], *r["intelligibility_ci95"]) if r["intelligibility"] is not None else "-",
                      fmt_num(r["dnsmos_ovrl"]), fmt_num(r["speaker_cosine"], 3)])
    out.write(md_table(["lang", "engine", "EP", "ok", "TTFA p50 ms", "TTFA p95 ms", "RTF p50", "intelligibility %", "DNSMOS OVRL", "speaker cos"], table))
    return out.getvalue()


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="append", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=DEFAULT_B)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args(argv)
    prompts = {p["id"]: p for p in load_jsonl(args.prompts)}
    reports = []
    for path in args.report:
        r = load_json(path)
        reports.append((r.get("name") or os.path.basename(os.path.dirname(os.path.abspath(path))), r))
    s = score(reports, prompts, args.bootstrap, args.seed)
    os.makedirs(args.out, exist_ok=True)
    write_json(os.path.join(args.out, "tts_scores.json"), s)
    write_text(os.path.join(args.out, "tts_scores.md"), to_markdown(s))
    print(f"{len(s['rows'])} rows -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
