#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Merge speech-bench exports (friends' ``export.json``) and reports into one speed table per mode:
which machine (rig label, CPU, GPU, driver), which load, which engine on which device, and the
timing / resource aggregates. Every export is checked against the privacy rules first.

    python score/merge_exports.py friends/*.json bench-results/speech/*/report.json --out merged [--md]

Writes ``merged.json`` (``sigil-speech-bench-merged/1``) and ``merged.md``. The table columns per
mode: stt -> RTF p50 / p95, compute p95, final latency p95, first partial p95, CPU cores, VRAM peak;
mt -> latency p50 / p95, chars/s, VRAM peak; tts -> TTFA p50 / p95, RTF p50, VRAM peak.
"""

from __future__ import annotations

import argparse
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from score.common import fmt_num, load_json, md_table, now_ms, write_json, write_text  # noqa: E402

COLUMNS = {
    "stt": ["rtf_p50", "rtf_p95", "compute_ms_p95", "final_latency_ms_p95", "first_partial_ms_p95", "cpu_cores_mean", "vram_peak_mb"],
    "mt": ["latency_ms_p50", "latency_ms_p95", "chars_per_s", "cpu_cores_mean", "vram_peak_mb"],
    "tts": ["ttfa_ms_p50", "ttfa_ms_p95", "rtf_p50", "cpu_cores_mean", "vram_peak_mb"],
}


def machine(env: dict) -> str:
    gpus = ", ".join(f"{g.get('name')} ({g.get('driver')})" for g in env.get("gpus") or []) or "no GPU"
    rig = env.get("sigil_rig")
    return (f"{rig}: " if rig else "") + f"{env.get('cpu') or '?'} / {gpus}"


def rows_of(doc: dict, source: str) -> list[dict]:
    engines = {e["id"]: e for e in doc.get("engines", [])}
    runs_spec = {}
    if "spec_summary" in doc:
        runs_spec = {r["name"]: r for r in doc["spec_summary"].get("runs", [])}
    elif "spec" in doc:
        runs_spec = {r["name"]: r for r in doc["spec"].get("runs", [])}
    out = []
    for run in doc.get("runs", []):
        if run.get("mode") not in ("cpu_speed", "gpu_under_load", "accuracy", "tts_listening", "mt_quality"):
            continue
        rs = runs_spec.get(run["name"], {})
        load = (run.get("load") or {}).get("profile") or rs.get("load_profile") or "none"
        external = (run.get("load") or {}).get("external") or rs.get("external_load") or ""
        for res in run.get("results", []):
            e = engines.get(res["engine"], {})
            out.append({
                "source": source, "machine": machine(doc.get("environment") or {}), "sigil_version": (doc.get("environment") or {}).get("sigil_version"),
                "spec_sha256": doc.get("spec_sha256"), "run": run["name"], "mode": run["mode"], "load": load + (f" + {external}" if external else ""),
                "engine": res["engine"], "task": e.get("task"), "model": e.get("model"), "ep": res.get("execution_provider_used") or e.get("execution_provider_used"),
                "status": res.get("status"), "warnings": len(run.get("warnings") or []) + len(((run.get("state") or {}).get("warnings")) or []),
                "aggregate": res.get("aggregate") or {},
            })
    return out


def to_markdown(rows: list[dict]) -> str:
    out = io.StringIO()
    out.write("# Speech benchmark exports\n\n")
    for task, cols in COLUMNS.items():
        sel = [r for r in rows if r["task"] == task and r["mode"] in ("cpu_speed", "gpu_under_load")]
        if not sel:
            continue
        out.write(f"## {task.upper()} speed\n\n")
        table = [[r["machine"], r["run"], r["load"], r["engine"], r["ep"] or "-", r["status"] or "-", str(r["aggregate"].get("items", "-"))]
                 + [fmt_num(r["aggregate"].get(c), 3 if c.startswith("rtf") or c == "cpu_cores_mean" else 0) for c in cols] for r in sel]
        out.write(md_table(["machine", "run", "load", "engine", "EP", "status", "items"] + cols, table))
        out.write("\n")
    return out.getvalue()


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="export.json and / or report.json files")
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args(argv)
    import validate  # noqa: PLC0415

    rows, refused = [], []
    for path in args.files:
        doc = load_json(path)
        if doc.get("schema") == "sigil-speech-bench-export/1":
            errs = validate.privacy_errors(doc)
            if errs:
                refused.append({"file": os.path.basename(path), "errors": errs[:10]})
                continue
        elif doc.get("schema") != "sigil-speech-bench-report/1":
            refused.append({"file": os.path.basename(path), "errors": [f"not a report or export ({doc.get('schema')})"]})
            continue
        rows += rows_of(doc, os.path.basename(path))
    merged = {"schema": "sigil-speech-bench-merged/1", "created_unix_ms": now_ms(), "rows": rows, "refused": refused}
    write_json(args.out + ".json", merged)
    write_text(args.out + ".md", to_markdown(rows))
    print(f"{len(rows)} rows from {len(args.files) - len(refused)} file(s) -> {args.out}.md")
    for r in refused:
        print(f"refused {r['file']}: {r['errors'][0]}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
