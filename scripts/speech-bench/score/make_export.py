#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
r"""Derive the friend-shareable export (``sigil-speech-bench-export/1``) from a speech-bench report.

This is the reference implementation of the export rules the Rust runner follows when it writes
``export.json`` (P7-A); use it to re-export an old report or to check the runner's output:

    python score/make_export.py <out>/report.json [--include-text] [--no-items] [--note "..."] [--out export.json]

Rules: no audio paths and no ``files``; every absolute path inside a string becomes
``<path>/<file name>`` (folders may hold spaces, as in ``C:\Users\Jane Doe\...``; a bare profile folder
leaves no name); a bring-your-own model is ``byo``; host / user name keys never leave the
environment block; no names of other programs (a run state's other GPU users become a count, the run
warning that lists them keeps only the share) and a power plan that is not one of Windows' built-in
ones reads ``custom``; item text (hypotheses, translations, back-transcriptions, segment text) is kept
only with ``--include-text`` and never for real clips (ids ``real/...``); ``--no-items`` keeps
aggregates and groups only.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time

CATALOG_ID = re.compile(r"^[a-z0-9-]+/[A-Za-z0-9._-]+$")
# A user's profile folder (C:\Users\<name>, /Users/<name>, /home/<name>; the name may hold spaces). Group 1
# is what follows the name when the path goes on (a separator and one more character); a profile folder that
# is the path's end is replaced whole, so the user name does not survive as a "file name".
PROFILE_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/](?i:Users)[\\/]|/Users/|/home/)[^\\/:*?\"'<>|\r\n;,]+(?:([\\/][^\s\"'|;,\\/])|[\\/])?")
# Absolute paths: a drive (starting a word, so http://x is not one), a UNC prefix or a Unix home / mount
# folder, then the folders up to the last separator (they may hold spaces: Jane Doe, Program Files (x86)),
# then the file name up to the next space or delimiter.
ABS_PATH = re.compile(r"(?:\b[A-Za-z]:[\\/]|\\\\|/(?:Users|home|mnt|tmp)/)(?:[^\\/:*?\"'<>|\r\n;,]*[\\/])*[^\s\"'|;,]*")
# The run warning that names the other GPU users (bench::gpu_state::state_warnings).
OTHER_GPU_WARNING = re.compile(r"^another process used the GPU \(up to ([0-9.]+) % SM")
PRIVATE_ENV_KEYS = {"hostname", "host", "computer_name", "user", "username", "user_name"}
# Windows' built-in power plans (as an English Windows names them); any other name is not exported.
BUILTIN_POWER_PLANS = {"balanced", "high performance", "power saver", "ultimate performance"}


def scrub_str(text: str) -> str:
    """Replace absolute paths in a string with ``<path>/<basename>``."""
    text = PROFILE_PATH.sub(lambda m: m.group(0) if m.group(1) else "<path>/", text)
    return ABS_PATH.sub(lambda m: "<path>/" + re.split(r"[\\/]", m.group(0).rstrip("\\/"))[-1], text)


def scrub(value):
    """Replace absolute paths inside strings (recursively) with ``<path>/<basename>``."""
    if isinstance(value, str):
        return scrub_str(value)
    if isinstance(value, list):
        return [scrub(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items()}
    return value


def power_plan_label(name: str) -> str:
    """A built-in Windows power plan as it is, anything else ``custom``."""
    return name if name.strip().lower() in BUILTIN_POWER_PLANS else "custom"


def redact_warning(warning: str) -> str:
    """A run warning without the names of the programs that used the GPU."""
    m = OTHER_GPU_WARNING.match(warning)
    return f"another process used the GPU (up to {m.group(1)} % SM)" if m else warning


def _redact_sample(sample) -> None:
    if not isinstance(sample, dict):
        return
    sample.pop("other_top_process", None)
    if isinstance(sample.get("power_plan"), str):
        sample["power_plan"] = power_plan_label(sample["power_plan"])


def _redact_warnings(warnings) -> None:
    if isinstance(warnings, list):
        warnings[:] = [redact_warning(w) if isinstance(w, str) else w for w in warnings]


def redact_state(state) -> None:
    """A run's ``state`` for the export: the names of the programs that used the GPU become a count,
    power plans follow ``power_plan_label``."""
    if not isinstance(state, dict):
        return
    for key in ("start", "end"):
        _redact_sample(state.get(key))
    for sample in state.get("series") or []:
        _redact_sample(sample)
    summary = state.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("other_top_processes"), list):
        summary["other_top_processes_count"] = len(summary.pop("other_top_processes"))
    _redact_warnings(state.get("warnings"))


def model_label(model: str) -> str:
    return model if (not model or CATALOG_ID.match(model)) else "byo"


def spec_summary(spec: dict) -> dict:
    return {
        "engines": [{"id": e["id"], "task": e["task"], "backend": e["backend"], "model": model_label(e.get("model", "")),
                     "execution_provider": e.get("execution_provider", "auto"), "variant": e.get("variant"), "threads": e.get("threads")}
                    for e in spec.get("engines", [])],
        "runs": [{k: v for k, v in {"name": r["name"], "mode": r["mode"], "load_profile": r.get("load_profile", "none"),
                                    "external_load": r.get("external_load") or None, "feed": (r.get("feed") or {}).get("kind")}.items() if v is not None}
                 for r in spec.get("runs", [])],
    }


def make_export(report: dict, include_text: bool = False, include_items: bool = True, note: str | None = None, now_ms: int | None = None) -> dict:
    env = {k: v for k, v in (report.get("environment") or {}).items() if k.lower() not in PRIVATE_ENV_KEYS}
    if isinstance(env.get("power_plan"), str):
        env["power_plan"] = power_plan_label(env["power_plan"])
    engines = []
    for e in report.get("engines", []):
        e = copy.deepcopy(e)
        e["model"] = model_label(e.get("model", ""))
        engines.append(e)
    runs = []
    for run in report.get("runs", []):
        run = copy.deepcopy(run)
        run.pop("files", None)
        redact_state(run.get("state"))
        _redact_warnings(run.get("warnings"))
        for res in run.get("results", []):
            if not include_items:
                res["items"] = []
                continue
            for it in res.get("items", []):
                real = str(it.get("item", "")).startswith("real/")
                keep = include_text and not real
                if "audio" in it:
                    it["audio"] = None
                for key in ("text", "asr_text"):
                    if key in it and not keep:
                        it[key] = None
                if not keep and it.get("segments"):
                    for seg in it["segments"]:
                        seg["text"] = ""
        runs.append(run)
    out = {
        "schema": "sigil-speech-bench-export/1",
        "report_schema": report.get("schema", "sigil-speech-bench-report/1"),
        "name": report["name"],
        "description": report.get("description", ""),
        "spec_sha256": report["spec_sha256"],
        "spec_summary": spec_summary(report.get("spec") or {}),
        "started_unix_ms": report["started_unix_ms"],
        "duration_ms": report["duration_ms"],
        "passed": report["passed"],
        "failures": report.get("failures", []),
        "environment": env,
        "engines": engines,
        "runs": runs,
        "export": {"created_unix_ms": int(now_ms if now_ms is not None else time.time() * 1000), "include_text": bool(include_text), "include_items": bool(include_items)},
    }
    if report.get("corpus"):
        out["corpus"] = report["corpus"]
    if note:
        out["export"]["note"] = note
    return scrub(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report")
    ap.add_argument("--include-text", action="store_true")
    ap.add_argument("--no-items", action="store_true")
    ap.add_argument("--note")
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    with open(args.report, encoding="utf-8") as f:
        report = json.load(f)
    export = make_export(report, args.include_text, not args.no_items, args.note)
    out = args.out or os.path.join(os.path.dirname(os.path.abspath(args.report)), "export.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(export, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"export -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
