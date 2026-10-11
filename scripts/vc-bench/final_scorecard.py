# SPDX-License-Identifier: AGPL-3.0-or-later
"""P7-B: build the final voice-changer scorecard from a campaign folder.

`final_scorecard.ps1` writes one folder per campaign:

    <root>/steps.jsonl          one JSON line per step (name, kind, start / end, quiet check, exit code)
    <root>/load/<step>.csv      the system load sampled every 2 s while the step ran
    <root>/gate/<pass>/**/gate-summary.json
                                run-perf-gate.ps1 -Profile final sessions (pass1, pass2, pass3 in
                                rotated order, plus `once` for the rows that need one pass)
    <root>/corpus/<ep>/summary.json
                                scripts/bench/score_v2/aggregate.py over run_corpus.ps1 (T5, scorer v2)
    <root>/decisions/<name>/gate-summary.json
                                interleaved (ABBA) sessions that decide a default

This script reads them and writes `final-scorecard.json` and `final-scorecard.md`:

- every profile target as a row: the median over the passes (a target is judged on the median, as the
  gate does in its interleaved mode, never on one pass: a single pass carries the rig's 11-19 %
  per-pair spread), the min..max over the passes, and its status against the revised limit;
- the T5 v2 rows per execution provider from the corpus summaries, against ADR 0009's limits;
- the decision sessions as paired rows (A median, B median, the interval, the verdict);
- a load report per step, flagging the steps that ran beside a compiler or a busy desktop (the
  scorecard says so instead of hiding it).

Standard library only. `python scripts/vc-bench/final_scorecard.py <root> [--md out.md] [--json out.json]`.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import sys
from datetime import datetime, timezone
from typing import Any, Iterable

SCHEMA = "sigil-vc-final-scorecard/1"

# A step ran "disturbed" when a compiler ran beside it or the desktop used more than this share of the
# CPU (the orchestrator's quiet window: no rustc, CPU under about 25 %).
QUIET_FOREIGN_PCT = 25.0
# The 90th percentile of the foreign load may not exceed this either (a short burst is a disturbance).
QUIET_FOREIGN_P90_PCT = 45.0

# Targets whose status words are reported as they are (the gate's words are `met`, `MISSED`,
# `not measured` and `skipped`); the scorecard adds `provisional` for the rows the plan marks so.
PROVISIONAL_MARKERS = ("provisional H1", "provisional H3")


# ---------------------------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------------------------
def read_json(path: str) -> Any:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def median(values: Iterable[float]) -> float | None:
    v = sorted(x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return None
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def percentile(values: list[float], q: float) -> float | None:
    v = sorted(values)
    if not v:
        return None
    if len(v) == 1:
        return v[0]
    pos = q * (len(v) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def fmt(value: Any, digits: int = 4) -> str:
    """A number as the tables print it: at most `digits` significant digits, no trailing zeros."""
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1000:
            return f"{value:.0f}"
        s = f"{value:.{digits}g}"
        return s
    return str(value)


class LimitError(ValueError):
    """A limit string the scorecard cannot read (the gate changed its format). A row with such a limit is
    never reported `met`: it gets the status `limit not parsed`."""


# PowerShell prints a small limit such as 0.00001 as `1E-05`, so the number accepts an exponent.
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?"
_LIMIT_RE = re.compile(rf"^\s*(?:(?P<lo>{_NUM})\s*\.\.\s*(?P<hi>{_NUM})|(?P<op><=|>=)\s*(?P<v>{_NUM}))\s*$")


def parse_limit(text: str) -> tuple[float | None, float | None]:
    """The gate's limit string as (min, max): `<= 85`, `>= 0.6` or `-1 .. 1`. An empty string is "no limit"
    (None, None); any other text that is not one of these three forms (`< 85`, `<= n/a`) raises LimitError
    instead of reading as "no limit", which `judge` would call `met` whatever the value."""
    if not (text or "").strip():
        return None, None
    m = _LIMIT_RE.match(text)
    if not m:
        raise LimitError(f"cannot read the limit {text!r} (expected '<= x', '>= x' or 'a .. b')")
    if m.group("lo") is not None:
        return float(m.group("lo")), float(m.group("hi"))
    v = float(m.group("v"))
    return (None, v) if m.group("op") == "<=" else (v, None)


def judge(value: float | None, lo: float | None, hi: float | None) -> str:
    """The gate's words for a value against a limit."""
    if value is None:
        return "not measured"
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        return "MISSED"
    return "met"


# Statuses for a measured value whose limit could not be used: never `met`.
LIMIT_NOT_PARSED = "limit not parsed"
NO_LIMIT = "no limit"


def judge_limit(value: float | None, limit: str | None) -> tuple[str, str | None]:
    """(status, limit error) of a value against the gate's limit string. `met` only when a limit was read:
    an unreadable limit gives `limit not parsed` and an empty one `no limit`, whatever the value is."""
    try:
        lo, hi = parse_limit(limit or "")
    except LimitError as e:
        return (LIMIT_NOT_PARSED if value is not None else "not measured"), str(e)
    if value is not None and lo is None and hi is None:
        return NO_LIMIT, None
    return judge(value, lo, hi), None


def limit_bounds(limit: str | None) -> tuple[float | None, float | None]:
    """parse_limit for a caller that only wants the bounds: (None, None) when the limit is unusable."""
    try:
        return parse_limit(limit or "")
    except LimitError:
        return None, None


# ---------------------------------------------------------------------------------------------
# the gate sessions (T1-T4, T7, T9)
# ---------------------------------------------------------------------------------------------
def gate_summaries(root: str, sub: str) -> list[tuple[str, str]]:
    """(pass name, path) of every gate-summary.json under `<root>/<sub>/<pass>/`."""
    out = []
    base = os.path.join(root, sub)
    for path in sorted(glob.glob(os.path.join(base, "**", "gate-summary.json"), recursive=True)):
        rel = os.path.relpath(path, base)
        out.append((rel.split(os.sep)[0], path))
    return out


def target_rows(root: str) -> list[dict]:
    """One row per profile target: the median over the passes that measured it."""
    rows: dict[tuple, dict] = {}
    order: list[tuple] = []
    for pass_name, path in gate_summaries(root, "gate"):
        try:
            doc = read_json(path)
        except (OSError, ValueError):
            continue
        for t in doc.get("targets", []) or []:
            key = (t.get("id"), t.get("source"), t.get("limit"), t.get("label"))
            if key not in rows:
                rows[key] = {
                    "id": t.get("id"), "label": t.get("label"), "source": t.get("source"),
                    "limit": t.get("limit"), "gate": bool(t.get("gate")),
                    "values": [], "passes": [], "statuses": [],
                }
                order.append(key)
            r = rows[key]
            r["statuses"].append(t.get("status"))
            v = t.get("value")
            if isinstance(v, (int, float)):
                r["values"].append(float(v))
                r["passes"].append(pass_name)
    out = []
    for key in order:
        r = rows[key]
        lo, hi = limit_bounds(r["limit"])
        med = median(r["values"])
        limit_error = None
        if med is None:
            # Nothing measured it: skipped (the template was not selected) or not measured.
            measured_status = "skipped" if all(s == "skipped" for s in r["statuses"]) else "not measured"
            status = measured_status
        else:
            status, limit_error = judge_limit(med, r["limit"])
        provisional = any(m in (r["label"] or "") for m in PROVISIONAL_MARKERS)
        out.append({
            "id": r["id"], "label": r["label"], "source": r["source"], "limit": r["limit"], "gate": r["gate"],
            "value": med, "min": min(r["values"]) if r["values"] else None, "max": max(r["values"]) if r["values"] else None,
            "n": len(r["values"]), "passes": r["passes"], "by_pass": dict(zip(r["passes"], r["values"])),
            "status": status, "provisional": provisional, "limit_error": limit_error,
            "all_passes_agree": len({judge(v, lo, hi) for v in r["values"]}) <= 1 if r["values"] else True,
        })
    return out


# ---------------------------------------------------------------------------------------------
# headline metrics straight from the reports (the stage tables)
# ---------------------------------------------------------------------------------------------
def report_paths(root: str, spec: str) -> list[str]:
    paths = []
    for pass_name, path in gate_summaries(root, "gate"):
        cand = os.path.join(os.path.dirname(path), spec, "report.json")
        if os.path.isfile(cand):
            paths.append(cand)
    return paths


def run_metric(report: dict, run: str, metric: str) -> float | None:
    for r in report.get("runs", []) or []:
        if r.get("name") == run:
            v = (r.get("metrics") or {}).get(metric)
            return float(v) if isinstance(v, (int, float)) else None
    return None


def headline(root: str, spec: str, wanted: dict[str, tuple[str, str]]) -> dict[str, float | None]:
    """`wanted` maps a column name to (run, metric); the value is the median over the passes' reports."""
    reports = []
    for p in report_paths(root, spec):
        try:
            reports.append(read_json(p))
        except (OSError, ValueError):
            pass
    return {name: median(run_metric(rep, run, metric) for rep in reports) for name, (run, metric) in wanted.items()} | {"_reports": len(reports)}  # type: ignore[dict-item]


STAGE_COLUMNS: dict[str, tuple[str, str]] = {
    "call_p50_ms": ("offline", "call_ms_p50"),
    "call_p95_ms": ("offline", "call_ms_p95"),
    "call_p99_ms": ("offline", "call_ms_p99"),
    "content_p50_ms": ("offline", "content_ms_p50"),
    "f0_p50_ms": ("offline", "f0_ms_p50"),
    "enc_p50_ms": ("offline", "enc_ms_p50"),
    "flowdec_p50_ms": ("offline", "flowdec_ms_p50"),
    "syncs_per_call_p50": ("offline", "syncs_per_call_p50"),
    "rtf_p99": ("offline", "rtf_p99"),
    "offline_cores": ("offline", "cpu_cores"),
    "vram_mb": ("offline", "vram_engine_mb"),
    "engine_load_ms": ("offline", "engine_load_ms"),
    "paced_latency_p50_ms": ("paced", "latency_ms_p50"),
    "paced_latency_p99_ms": ("paced", "latency_ms_p99"),
    "paced_call_p95_ms": ("paced", "call_ms_p95"),
    "paced_cores": ("paced", "cpu_cores"),
    "paced_underruns": ("paced", "underruns"),
    "paced_rtf_p99": ("paced", "rtf_p99"),
}


def stage_table(root: str, specs: Iterable[str]) -> dict[str, dict]:
    return {spec: headline(root, spec, STAGE_COLUMNS) for spec in specs}


# ---------------------------------------------------------------------------------------------
# T5 (scorer v2) from the corpus summaries
# ---------------------------------------------------------------------------------------------
T5_LABELS = {
    "speaker_set_sim_delta": "identity delta (speaker similarity to the target set, stream minus whole ceiling)",
    "fed_input_fpe_cents": "fed fine pitch error (cents)",
    "fed_input_gpe_pooled": "fed gross pitch error (pooled)",
    "fed_input_vuv_error": "fed V/UV error",
    "env_mcd_over_floor_db": "envelope MCD over its floor (dB)",
    "clicks_binomial": "clicks (binomial, model-delay windows)",
    "clicks_narrow": "clicks (narrow crossfade windows)",
}


def t5_rows(root: str) -> dict[str, dict]:
    """T5 per execution provider: ADR 0009's rows with the Low-latency limits, plus the headline means."""
    out: dict[str, dict] = {}
    for path in sorted(glob.glob(os.path.join(root, "corpus", "*", "summary.json"))):
        ep = os.path.basename(os.path.dirname(path))
        try:
            doc = read_json(path)
        except (OSError, ValueError):
            continue
        rows = []
        for key, row in (doc.get("t5_v2") or {}).items():
            rows.append({
                "metric": key, "label": T5_LABELS.get(key, key), "mean": row.get("mean"),
                "limit_low_latency": row.get("limit_low_latency"), "limit_quality": row.get("limit_quality"),
                "met": bool(row.get("met_low_latency")), "n": row.get("n"),
                "ci95": (doc.get("metrics", {}).get(key) or {}).get("ci95"),
            })
        heads = {}
        for key in ("speaker_cosine_v2", "mcd_db", "env_mcd_db", "dnsmos_ovrl", "ceiling_dnsmos_ovrl", "f0_rmse_cents", "vuv_error", "gross_pitch_error", "clicks"):
            m = (doc.get("metrics") or {}).get(key)
            if m:
                heads[key] = {"mean": m.get("mean"), "ci95": m.get("ci95"), "n": m.get("n")}
        out[ep] = {"clips": doc.get("clips"), "rows": rows, "metrics": heads, "summary": path}
    return out


# ---------------------------------------------------------------------------------------------
# the CPU F0 comparison on the same clips (SwiftF0 corpus against the FCPE corpus)
# ---------------------------------------------------------------------------------------------
# (metric, label, scale to the printed unit): the rows of the comparison, in the order they print.
PAIRED_F0_ROWS: tuple[tuple[str, str, float], ...] = (
    ("speaker_cosine_v2", "speaker cosine v2", 1.0),
    ("dnsmos_ovrl", "DNSMOS OVRL", 1.0),
    ("env_mcd_db", "envelope MCD (dB)", 1.0),
    ("vuv_error", "output V/UV error (points)", 100.0),
    ("fed_input_vuv_error", "fed V/UV error (points)", 100.0),
    ("fed_input_fpe_cents", "fed fine pitch error (cents)", 1.0),
    ("fed_input_gpe", "fed gross pitch error, mean per clip (points)", 100.0),
    ("f0_rmse_cents", "output F0 RMSE (cents)", 1.0),
)


def bootstrap_mean_ci(values: list[float], reps: int = 10000, alpha: float = 0.05, seed: int = 20261001) -> tuple[float, float] | None:
    """Percentile bootstrap of the mean (the corpus summaries' rule, standard library only)."""
    import random
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return percentile(means, alpha / 2), percentile(means, 1 - alpha / 2)  # type: ignore[return-value]


def corpus_clip_metrics(corpus_dir: str) -> dict[str, dict[str, float]]:
    """clip file name -> the metrics of its scorer-v2 quality run, from the per-clip reports of a corpus
    folder (`<corpus_dir>/<clip folder>/report.json`; the values are the medians over the clip's offsets)."""
    out: dict[str, dict[str, float]] = {}
    for path in sorted(glob.glob(os.path.join(corpus_dir, "*", "report.json"))):
        try:
            rep = read_json(path)
        except (OSError, ValueError):
            continue
        clip = os.path.basename((((rep.get("spec") or {}).get("input")) or {}).get("path", "") or "")
        for run in rep.get("runs", []) or []:
            if run.get("mode") == "quality" and ((run.get("details") or {}).get("scorer") == "v2") and clip:
                out[clip] = {k: float(v) for k, v in (run.get("metrics") or {}).items() if isinstance(v, (int, float))}
    return out


def paired_f0(root: str, a: str = "cpu", b: str = "cpu-fcpe") -> dict | None:
    """Corpus `b` minus corpus `a` on the clips both ran (paired per clip, bootstrap interval of the mean
    difference). None when either corpus is missing or fewer than three clips are shared."""
    ma = corpus_clip_metrics(os.path.join(root, "corpus", a))
    mb = corpus_clip_metrics(os.path.join(root, "corpus", b))
    shared = sorted(set(ma) & set(mb))
    if len(shared) < 3:
        return None
    rows = []
    for key, label, scale in PAIRED_F0_ROWS:
        pairs = [(ma[c][key], mb[c][key]) for c in shared if key in ma[c] and key in mb[c]]
        if len(pairs) < 3:
            continue
        diffs = [(y - x) * scale for x, y in pairs]
        ci = bootstrap_mean_ci(diffs)
        rows.append({
            "metric": key, "label": label, "n": len(pairs),
            "a_mean": sum(x for x, _ in pairs) / len(pairs) * scale, "b_mean": sum(y for _, y in pairs) / len(pairs) * scale,
            "diff": sum(diffs) / len(diffs), "lo": ci[0] if ci else None, "hi": ci[1] if ci else None,
            "b_higher": sum(1 for d in diffs if d > 0),
        })
    return {"a": a, "b": b, "clips": shared, "rows": rows}


def signed(value: float | None, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:+.{digits}g}"


# ---------------------------------------------------------------------------------------------
# decision sessions (interleaved A/B)
# ---------------------------------------------------------------------------------------------
def decision_rows(root: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(root, "decisions", "*", "gate-summary.json"))):
        name = os.path.basename(os.path.dirname(path))
        try:
            doc = read_json(path)
        except (OSError, ValueError):
            continue
        paired = doc.get("paired") or []
        if not isinstance(paired, list):
            paired = []
        rows = []
        for p in paired:
            row = {k: p.get(k) for k in ("spec", "run", "metric", "n", "a_median", "b_median", "median_pct", "lo_pct", "hi_pct", "verdict", "note", "regime_mismatched")}
            a, b = row.get("a_median"), row.get("b_median")
            # The difference of the medians in the metric's own unit: the figure to read where the gate's percentage is
            # relative to a floor (a count near zero, a ratio in 0..1).
            row["b_minus_a"] = (b - a) if isinstance(a, (int, float)) and isinstance(b, (int, float)) else None
            row["linear"] = str(row.get("note") or "").startswith("linear")
            rows.append(row)
        out.append({
            "name": name,
            "compare_vars": doc.get("compare_vars") or "",
            "compare_env": doc.get("compare_env") or "",
            "compare_exe": doc.get("compare_exe"),
            "rounds": doc.get("rounds") or {},
            "warnings": doc.get("warnings") or [],
            "state_flags": len(doc.get("state_flags") or []),
            "rows": rows,
        })
    return out


# ---------------------------------------------------------------------------------------------
# the load log
# ---------------------------------------------------------------------------------------------
def read_steps(root: str) -> list[dict]:
    path = os.path.join(root, "steps.jsonl")
    steps = []
    if os.path.isfile(path):
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        steps.append(json.loads(line))
                    except ValueError:
                        continue
    return steps


def read_load_csv(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "t": float(r.get("t_unix") or 0),
                    "cpu_pct": float(r.get("cpu_total_pct") or 0),
                    "sigil_cores": float(r.get("sigil_cores") or 0),
                    "rustc": int(float(r.get("rustc") or 0)),
                    "cargo": int(float(r.get("cargo") or 0)),
                    "heavy": [x for x in (r.get("heavy_other") or "").split(";") if x],
                    "lock_owner": (r.get("lock_owner") or "").strip(),
                })
            except ValueError:
                continue
    return rows


def parse_time(text: Any) -> float | None:
    """An ISO timestamp as written by PowerShell (`o` format, 7 fractional digits) as unix seconds."""
    if not text:
        return None
    t = re.sub(r"(\.\d{6})\d+", r"\1", str(text))
    try:
        d = datetime.fromisoformat(t)
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return d.timestamp()


def select_rows(step: dict, rows: list[dict]) -> tuple[list[dict], int]:
    """The samples that belong to a step: inside its time window and, when the lock's owner was logged,
    taken while the lock was this campaign's (a step whose harness takes the lock per run shares the
    machine with other agents between its runs). Returns the rows and how many samples the owner
    filter set aside."""
    t0, t1 = parse_time(step.get("start")), parse_time(step.get("end"))
    if step.get("kind") == "sub" and t0 is not None and t1 is not None:
        rows = [r for r in rows if t0 - 1.0 <= r["t"] <= t1 + 1.0]
    tag = step.get("lock_tag") or ""
    if tag and any(r.get("lock_owner") for r in rows):
        mine = [r for r in rows if (r.get("lock_owner") or "").startswith(tag)]
        return mine, len(rows) - len(mine)
    return rows, 0


def step_load(step: dict, rows: list[dict], threads: int) -> dict:
    """The load summary of one step: foreign CPU share (total minus Sigil's own), compilers, other busy programs."""
    rows, other_owner = select_rows(step, rows)
    foreign = []
    for r in rows:
        own_pct = 100.0 * r["sigil_cores"] / max(threads, 1)
        foreign.append(max(0.0, r["cpu_pct"] - own_pct))
    heavy = sorted({h for r in rows for h in r["heavy"]})
    rustc_max = max((r["rustc"] for r in rows), default=0)
    cargo_max = max((r["cargo"] for r in rows), default=0)
    mean_foreign = sum(foreign) / len(foreign) if foreign else None
    p90_foreign = percentile(foreign, 0.9)
    reasons = []
    if rustc_max > 0 or cargo_max > 0:
        reasons.append(f"a compiler ran beside it (rustc {rustc_max}, cargo {cargo_max})")
    if mean_foreign is not None and mean_foreign > QUIET_FOREIGN_PCT:
        reasons.append(f"foreign CPU {mean_foreign:.0f} % on average")
    elif p90_foreign is not None and p90_foreign > QUIET_FOREIGN_P90_PCT:
        reasons.append(f"foreign CPU burst to {p90_foreign:.0f} % (p90)")
    pre = step.get("pre") or {}
    if pre and pre.get("quiet") is False:
        reasons.append("the quiet window was not reached before it started" + (f" (CPU {pre.get('cpu_pct')} %, rustc {pre.get('rustc')})" if pre.get("cpu_pct") is not None else ""))
    return {
        "step": step.get("name"), "kind": step.get("kind"), "pass": step.get("pass"), "specs": step.get("specs") or [],
        "start": step.get("start"), "end": step.get("end"),
        "exit": step.get("exit"), "samples": len(rows), "samples_other_lock_owner": other_owner,
        "foreign_cpu_mean_pct": mean_foreign, "foreign_cpu_p90_pct": p90_foreign,
        "cpu_total_mean_pct": (sum(r["cpu_pct"] for r in rows) / len(rows)) if rows else None,
        "sigil_cores_mean": (sum(r["sigil_cores"] for r in rows) / len(rows)) if rows else None,
        "rustc_max": rustc_max, "cargo_max": cargo_max, "heavy": heavy,
        "pre_cpu_pct": pre.get("cpu_pct"), "pre_rustc": pre.get("rustc"),
        "disturbed": bool(reasons), "reasons": reasons,
    }


def load_report(root: str) -> list[dict]:
    steps = read_steps(root)
    env_path = os.path.join(root, "env.json")
    threads = 16
    if os.path.isfile(env_path):
        try:
            threads = int(read_json(env_path).get("logical_processors") or 16)
        except (OSError, ValueError, TypeError):
            pass
    out = []
    for s in steps:
        csv_name = s.get("csv") if s.get("kind") == "sub" else s.get("name")
        csv_path = os.path.join(root, "load", f"{csv_name}.csv")
        rows = read_load_csv(csv_path) if os.path.isfile(csv_path) else []
        out.append(step_load(s, rows, threads))
    return out


# A GPU that has only this work to do lowers its clocks between bursts (the driver's `gpu_idle` reason):
# P3B measured 1.1-1.8 GHz in a real-time cadence against 2.8 GHz back to back, and the p95 of a call moves
# by that much. A game keeps the clocks up, so the high-clock runs are what a streamer sees; both are shown.
HIGH_CLOCK_SHARE = 0.85
# Another process used at least this share of the GPU's SMs (median over the run): the run shared the GPU.
GPU_SHARED_PCT = 10.0


def parse_source(source: str) -> tuple[str, str, str] | None:
    """`spec/run metric` as the gate prints it; None for a ratio row."""
    if not source or " / " in source:
        return None
    m = re.match(r"^(?P<spec>[^/\s]+)/(?P<run>\S+) (?P<metric>\S+)$", source)
    return (m.group("spec"), m.group("run"), m.group("metric")) if m else None


def report_index(root: str) -> dict[tuple[str, str], str]:
    """(pass, spec) -> report.json for every plain gate session under <root>/gate."""
    idx: dict[tuple[str, str], str] = {}
    for pass_name, path in gate_summaries(root, "gate"):
        base = os.path.dirname(path)
        try:
            names = os.listdir(base)
        except OSError:
            continue
        for n in names:
            rp = os.path.join(base, n, "report.json")
            if os.path.isfile(rp):
                idx[(pass_name, n)] = rp
    return idx


def attach_regime(root: str, rows: list[dict]) -> float | None:
    """Add the GPU's median SM clock of every pass's run to the rows and the median over the high-clock
    runs (`value_high`, `n_high`). Returns the clock threshold in MHz (None without any clock record)."""
    idx = report_index(root)
    cache: dict[str, dict] = {}

    def sm_clock(pass_name: str, spec: str, run: str) -> float | None:
        rp = idx.get((pass_name, spec))
        if not rp:
            return None
        if rp not in cache:
            try:
                cache[rp] = read_json(rp)
            except (OSError, ValueError):
                cache[rp] = {}
        return run_metric(cache[rp], run, "state_sm_clock_mhz")

    def foreign_gpu(pass_name: str, spec: str, run: str) -> float | None:
        """The median over the run's 1 Hz samples of the SM share other processes used (a browser, an
        RGB-lighting service, the desktop compositor): the gate's own load process is not counted by
        the probe when it is the engine's, but is listed as `sigil.exe` for the out-of-process load."""
        rp = idx.get((pass_name, spec))
        if not rp or rp not in cache or run.endswith("oop") or run.endswith("-dml"):
            return None  # the out-of-process load is a Sigil process of ours: the probe would count it as foreign
        for rr in cache[rp].get("runs", []) or []:
            if rr.get("name") == run:
                ser = (((rr.get("details") or {}).get("state") or {}).get("series")) or []
                vals = [float(x.get("other_gpu_util_pct")) for x in ser if isinstance(x.get("other_gpu_util_pct"), (int, float))]
                return median(vals) if vals else None
        return None

    for r in rows:
        parsed = parse_source(r.get("source") or "")
        r["sm_by_pass"] = {}
        r["foreign_gpu_by_pass"] = {}
        if parsed and "cpu" not in parsed[0]:  # a CPU engine's rows say nothing about the GPU's clock or sharing
            spec, run, _ = parsed
            for p in r.get("passes", []):
                c = sm_clock(p, spec, run)
                if c is not None:
                    r["sm_by_pass"][p] = c
                f = foreign_gpu(p, spec, run)
                if f is not None:
                    r["foreign_gpu_by_pass"][p] = f
        r["gpu_shared_in"] = [p for p, f in r["foreign_gpu_by_pass"].items() if f >= GPU_SHARED_PCT]
    clocks = [c for r in rows for c in r["sm_by_pass"].values()]
    if not clocks:
        for r in rows:
            r["value_high"], r["n_high"] = None, 0
        return None
    threshold = HIGH_CLOCK_SHARE * max(clocks)
    for r in rows:
        high = [v for p, v in (r.get("by_pass") or {}).items() if r["sm_by_pass"].get(p, 0.0) >= threshold]
        r["n_high"] = len(high)
        r["value_high"] = median(high) if high else None
    return threshold


CTX_LABEL_PREFIX = "extra context 64 -> 512 ms"


def apply_context_session(rows: list[dict], decisions: list[dict]) -> None:
    """The context-scaling target compares two specs that ran in different passes (different GPU clock
    regimes), so the gate cannot judge it: the ABBA session `ctx512` (one executable, 64 against 512 ms on
    side B, both sides in one hold) does. The row takes B / A of the DirectML call p95 from that session."""
    ses = next((d for d in decisions if d["name"] == "ctx512"), None)
    if ses is None:
        return
    pair = next((p for p in ses["rows"] if p["spec"] == "ctx-dml" and p["run"] == "offline" and p["metric"] == "call_ms_p95"), None)
    if pair is None or not pair.get("a_median") or pair.get("b_median") is None:
        return
    for r in rows:
        if (r.get("label") or "").startswith(CTX_LABEL_PREFIX):
            ratio = pair["b_median"] / pair["a_median"]
            status, limit_error = judge_limit(ratio, r["limit"])
            r.update({
                "source": "ctx512 session (ABBA, ctx-dml/offline call_ms_p95, B / A)", "value": ratio, "min": ratio, "max": ratio,
                "n": pair.get("n") or 0, "status": status, "limit_error": limit_error, "passes": ["ctx512"], "by_pass": {"ctx512": ratio},
                "all_passes_agree": True,
            })


def annotate_disturbance(rows: list[dict], load: list[dict]) -> None:
    """Mark the rows whose pass ran beside a compiler or a busy desktop, and add the median over the
    undisturbed passes (`value_clean`, `n_clean`; None when every pass was disturbed). The status stays
    the one of all the passes: a disturbed pass is shown, not dropped."""
    disturbed: dict[tuple[str, str], bool] = {}
    for s in load:
        if s.get("kind") != "sub" or not s.get("pass"):
            continue
        for spec in s.get("specs") or []:
            key = (s["pass"], spec)
            disturbed[key] = disturbed.get(key, False) or bool(s.get("disturbed"))
    for r in rows:
        spec = (r.get("source") or "").split("/")[0]
        bad = [p for p in r.get("passes", []) if disturbed.get((p, spec))]
        r["disturbed_in"] = bad
        clean = [v for p, v in (r.get("by_pass") or {}).items() if p not in bad]
        r["n_clean"] = len(clean)
        r["value_clean"] = median(clean) if clean else None


# ---------------------------------------------------------------------------------------------
# what the output queue played, and rows judged from the reports
# ---------------------------------------------------------------------------------------------
# A paced or contention run converts the clip block by block and plays it through the production output queue
# (prefill, cap, recovery and drift trims). An engine can be fast enough on call time and still lose a large
# share of what it converted at that queue (CPU Streamer rows: 120 ms blocks against the 35 ms prefill, whose
# recovery trims every catch-up burst; the 70 ms cap is not the cause), and the latency of such a run is the
# latency of the audio that survived. So every such run reports the share played, and a target that reads
# "met" on call time or latency says so when the run lost audio.
#
# 0.98: the 3 s stall of the H1 simulation costs 0.3-0.5 % of a 60 s run and the GPU rows play 99.7-99.9 %;
# the CPU rows played 48-63 % at the default 35 / 70 ms queue and play 99.1-99.8 % with the output queue
# floor of voice_changer/block_buffer.rs.
DELIVERY_MIN_PLAYED = 0.98
DELIVERY_RUNS = ("paced", "h3-game", "h1-sim", "h2-load70", "h2-load70-oop")
DELIVERY_SPECS = ("sve-dml", "sve-cuda", "app-dml", "app-cuda", "sve-cpu", "sve-cpu-pool", "sve-cpu-t6")
DELIVERY_METRICS = ("played_ratio", "output_trimmed_ms", "underrun_ms", "underruns", "over_budget_ratio", "rtf_p99", "latency_ms_p99")

# The benchmark's CPU rows run SwiftF0 (the engine's `auto` F0 for the CPU) while the Streamer button applies FCPE
# (presets.ts); the generated rows say so, because the profile's labels call them "the shipped Streamer preset".
CPU_F0_NOTE = "F0 is SwiftF0 (the benchmark's CPU F0); the Streamer button applies FCPE"
SPEC_NOTES = {s: CPU_F0_NOTE for s in ("sve-cpu", "sve-cpu-pool", "sve-cpu-t6", "silent-cpu", "load-cpu")}


def played_ratio(run: dict) -> float | None:
    """Played frames over emitted frames of a paced or contention run: the share of the converted audio that
    reached the output (the rest was trimmed by the output queue)."""
    det = run.get("details") or {}
    e, p = det.get("emitted_frames"), det.get("played_frames")
    if isinstance(e, (int, float)) and isinstance(p, (int, float)) and e > 0:
        return float(p) / float(e)
    return None


def derived_metric(run: dict, metric: str) -> float | None:
    """A metric of one run of a report; `played_ratio` is computed from the run's details."""
    if metric == "played_ratio":
        return played_ratio(run)
    v = (run.get("metrics") or {}).get(metric)
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None


class ReportStore:
    """The gate reports of a campaign folder, read once: (pass, spec) -> report.json."""

    def __init__(self, root: str):
        self.index = report_index(root)
        self._cache: dict[str, dict] = {}

    def report(self, path: str) -> dict:
        if path not in self._cache:
            try:
                self._cache[path] = read_json(path)
            except (OSError, ValueError):
                self._cache[path] = {}
        return self._cache[path]

    def values(self, spec: str, run: str, metric: str) -> dict[str, float]:
        """pass -> the metric of the spec's run in that pass (passes that did not measure it are absent)."""
        out: dict[str, float] = {}
        for (pass_name, s), path in sorted(self.index.items()):
            if s != spec:
                continue
            for r in self.report(path).get("runs", []) or []:
                if r.get("name") == run:
                    v = derived_metric(r, metric)
                    if v is not None:
                        out[pass_name] = v
        return out


def delivery_table(root: str, store: ReportStore | None = None) -> list[dict]:
    """One row per (spec, run) of the paced and contention runs: the median over the passes of the share played,
    the audio trimmed and the dry output, the underruns, the blocks over budget and the p99 of the RTF and the
    latency. `flag` is set when the run played less than DELIVERY_MIN_PLAYED of what it converted."""
    store = store or ReportStore(root)
    rows = []
    for spec in DELIVERY_SPECS:
        for run in DELIVERY_RUNS:
            per = {m: store.values(spec, run, m) for m in DELIVERY_METRICS}
            if not per["played_ratio"]:
                continue
            played = list(per["played_ratio"].values())
            row: dict[str, Any] = {"spec": spec, "run": run, "n": len(played), "passes": sorted(per["played_ratio"])}
            for m in DELIVERY_METRICS:
                row[m] = median(per[m].values())
            row["played_min"], row["played_max"] = min(played), max(played)
            row["flag"] = row["played_ratio"] < DELIVERY_MIN_PLAYED
            rows.append(row)
    return rows


def _delivery_targets() -> list[dict]:
    """The rows the profile has no target for: what the CPU and GPU runs delivered, and the over-budget half of
    the T2b H1-simulation row (the profile only judges the p99 of the call there)."""
    out: list[dict] = []
    cpu = (
        ("sve-cpu", "the Streamer preset as the 38f426d build ships it: one ONNX Runtime thread"),
        ("sve-cpu-pool", "the CPU engine on ONNX Runtime's default pool"),
        ("sve-cpu-t6", "6 threads at below-normal priority"),
    )
    for spec, what in cpu:
        out.append({"id": "T4", "spec": spec, "run": "paced", "metric": "played_ratio", "min": DELIVERY_MIN_PLAYED,
                    "label": f"Streamer (CPU, 120 ms blocks, SwiftF0) share of the converted audio that reached the output, paced: >= 98 % ({what})"})
        out.append({"id": "T1", "spec": spec, "run": "h3-game", "metric": "played_ratio", "min": DELIVERY_MIN_PLAYED,
                    "label": f"Streamer (CPU, 120 ms blocks, SwiftF0) share of the converted audio that reached the output under game load (provisional H3): >= 98 % ({what})"})
    gpu = (
        ("sve-dml", "DirectML, 40 ms blocks"), ("sve-cuda", "CUDA, 40 ms blocks"),
        ("app-dml", "DirectML, the app's shape"), ("app-cuda", "CUDA, the app's shape"),
    )
    for spec, what in gpu:
        out.append({"id": "T1", "spec": spec, "run": "paced", "metric": "played_ratio", "min": DELIVERY_MIN_PLAYED,
                    "label": f"share of the converted audio that reached the output, paced: >= 98 % ({what})"})
    for spec, what in gpu:
        out.append({"id": "T2b", "spec": spec, "run": "h1-sim", "metric": "over_budget_ratio", "max": 0.001,
                    "label": f"H1 simulation: <= 0.1 % of blocks over budget (provisional H1; {what}); the profile has no target for it"})
    return out


DERIVED_TARGETS = _delivery_targets()


def derived_limit(t: dict) -> str:
    """The limit text of a derived target, in the gate's own forms."""
    lo, hi = t.get("min"), t.get("max")
    if lo is not None and hi is not None:
        return f"{lo:g} .. {hi:g}"
    return f">= {lo:g}" if lo is not None else f"<= {hi:g}"


def derived_rows(root: str, store: ReportStore | None = None) -> list[dict]:
    """DERIVED_TARGETS judged on the median over the passes' reports, in the shape of `target_rows`."""
    store = store or ReportStore(root)
    out = []
    for t in DERIVED_TARGETS:
        by = store.values(t["spec"], t["run"], t["metric"])
        vals = list(by.values())
        limit = derived_limit(t)
        med = median(vals)
        status, limit_error = judge_limit(med, limit) if med is not None else ("not measured", None)
        lo, hi = limit_bounds(limit)
        out.append({
            "id": t["id"], "label": t["label"], "source": f"{t['spec']}/{t['run']} {t['metric']}", "limit": limit, "gate": False,
            "value": med, "min": min(vals) if vals else None, "max": max(vals) if vals else None, "n": len(vals),
            "passes": list(by), "by_pass": dict(by), "status": status, "limit_error": limit_error,
            "provisional": any(m in t["label"] for m in PROVISIONAL_MARKERS), "derived": True,
            "all_passes_agree": len({judge(v, lo, hi) for v in vals}) <= 1 if vals else True,
        })
    return out


def annotate_delivery(rows: list[dict], delivery: list[dict]) -> None:
    """Mark the profile rows whose run played less than DELIVERY_MIN_PLAYED of its converted audio
    (`delivery_flag`, `delivery_note`): their status stays what the number says, and says so when it is `met`."""
    flagged = {(d["spec"], d["run"]): d for d in delivery if d["flag"]}
    for r in rows:
        r["delivery_flag"] = False
        parsed = parse_source(r.get("source") or "")
        if r.get("derived") or not parsed or (parsed[0], parsed[1]) not in flagged:
            continue
        d = flagged[(parsed[0], parsed[1])]
        r["delivery_flag"] = True
        r["delivery_note"] = (f"the run played only {100 * d['played_ratio']:.0f} % of the audio it converted "
                              f"({fmt(d['output_trimmed_ms'] / 1000.0, 3)} s trimmed by the output queue, {fmt(d['underruns'], 4)} underruns): "
                              f"the number is that of the audio that survived")


# ---------------------------------------------------------------------------------------------
# the markdown
# ---------------------------------------------------------------------------------------------
def md_escape(text: Any) -> str:
    return str(text if text is not None else "").replace("|", "\\|").replace("\n", " ")


def status_word(row: dict) -> str:
    s = row["status"]
    word = {"MISSED": "missed", "met": "met"}.get(s, s)
    if row.get("provisional") and s in ("met", "MISSED"):
        word += " (provisional)"
    if s == "met" and row.get("delivery_flag"):
        word += ", but audio lost"
    return word


def row_note(row: dict) -> str:
    """The generated row's note: the F0 the CPU rows ran, a run that lost audio, an unreadable limit."""
    notes = []
    spec = (parse_source(row.get("source") or "") or ("",))[0]
    if spec in SPEC_NOTES:
        notes.append(SPEC_NOTES[spec])
    if row.get("delivery_note"):
        notes.append(row["delivery_note"])
    if row.get("limit_error"):
        notes.append(row["limit_error"])
    return "; ".join(notes)


def high_cell(row: dict) -> str:
    if not row.get("sm_by_pass"):
        return ""
    if row.get("value_high") is None:
        return "none"
    return f"{fmt(row['value_high'])} ({row['n_high']})"


def spread(row: dict) -> str:
    if row.get("value") is None or row.get("n", 0) < 2:
        return ""
    return f"{fmt(row['min'])} .. {fmt(row['max'])}"


def target_table(rows: list[dict]) -> list[str]:
    """The markdown table of target rows (profile targets and derived rows alike)."""
    L = ["| ID | Row | Source | Limit | Median | Spread | High-clock median (n) | Passes | Status | Disturbed | Note |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        bad = r.get("disturbed_in") or []
        dist = (f"{', '.join(bad)} (clean median {fmt(r.get('value_clean'))})" if bad else "")
        shared = r.get("gpu_shared_in") or []
        if shared:
            dist += ("; " if dist else "") + "GPU shared in " + ", ".join(f"{p} ({r['foreign_gpu_by_pass'][p]:.0f} %)" for p in shared)
        L.append(f"| {md_escape(r['id'])} | {md_escape(r['label'])} | `{md_escape(r['source'])}` | {md_escape(r['limit'])} | {fmt(r['value'])} | {spread(r)} | {high_cell(r)} | {r['n']} | {status_word(r)} | {md_escape(dist)} | {md_escape(row_note(r))} |")
    return L


# The committed copy of the campaign of record (`--publish`): the generated tables behind VC_BENCHMARKS.md.
PUBLISH_BANNER = ("<!-- Generated by scripts/vc-bench/final_scorecard.py from the P7-B campaign folder (2026-10-09 to 2026-10-10, "
                  "plus the 6-thread FCPE session of the review fixes). Do not edit by hand: rerun the tool with --publish. -->")
PUBLISH_INTRO = ("The complete tables behind [VC_BENCHMARKS.md → P7-B final scorecard of the shipped defaults]"
                 "(../../../docs/benchmarks/VC_BENCHMARKS.md#p7-b-final-scorecard-of-the-shipped-defaults): every profile target with its "
                 "per-pass median and spread, the rows derived from the reports (what the output queue delivered, the H1 simulation's "
                 "over-budget half), the delivery of the converted audio by run, the T5 (scorer v2) rows per execution provider, the CPU F0 "
                 "comparison, the per-call stage table, every decision session and the load log of every step. `disturbed` marks a step that "
                 "ran beside a compiler or a busy desktop{disturbed}.")


def render_markdown(doc: dict, publish: bool = False) -> str:
    L: list[str] = []
    env = doc.get("environment") or {}
    if publish:
        L.append(PUBLISH_BANNER)
        L.append("")
    L.append("# P7-B final scorecard (generated)")
    L.append("")
    if publish:
        steps = doc.get("disturbed_steps") or []
        L.append(PUBLISH_INTRO.format(disturbed="; none did" if not steps else "; " + ", ".join(steps) + " did"))
        L.append("")
    if env:
        L.append(f"Executable `{env.get('exe_sha256', '')[:16]}`, {env.get('gpu', '')}, {env.get('cpu', '')}; passes: {', '.join(doc.get('passes', []))}.")
        L.append("")
    L.append("## Targets (median over the passes; the spread is min .. max)")
    L.append("")
    if doc.get("high_clock_threshold_mhz"):
        L.append(f"GPU runs at or above {doc['high_clock_threshold_mhz']:.0f} MHz (85 % of the highest SM clock seen) count as high-clock; a GPU with only this work to do lowers its clocks between bursts, a game keeps them up.")
        L.append("")
    L.extend(target_table(doc["rows"]))
    if doc.get("derived_rows"):
        L.append("")
        L.append("## Derived rows (judged from the reports; the profile has no target for them)")
        L.append("")
        L.append("The share of the converted audio that reached the output (`played_ratio`: played over emitted frames of the run) "
                 f"must be at least {100 * DELIVERY_MIN_PLAYED:.0f} %, and the H1 simulation's over-budget half of T2b is judged here because the profile only gates the p99 of the call there. "
                 "Same median-over-the-passes rule as above.")
        L.append("")
        L.extend(target_table(doc["derived_rows"]))
    if doc.get("delivery"):
        L.append("")
        L.append("## Delivery of the converted audio by run (median over the passes)")
        L.append("")
        L.append(f"`played` is the share of the converted audio that reached the output; a run below {100 * DELIVERY_MIN_PLAYED:.0f} % is flagged. "
                 "The H1 simulation's 3 s stall costs 297 underruns by design. `trimmed` and `dry` are the audio the output queue cut and the output that ran empty.")
        L.append("")
        L.append("| Spec | Run | Passes | Played (min .. max) | Trimmed ms | Dry ms | Underruns | Over budget | RTF p99 | Latency p99 ms | Flag |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for d in doc["delivery"]:
            played = f"{100 * d['played_ratio']:.1f} % ({100 * d['played_min']:.1f} .. {100 * d['played_max']:.1f})"
            flag = "audio lost" if d["flag"] else ""
            L.append(f"| {d['spec']} | {d['run']} | {d['n']} | {played} | {fmt(d['output_trimmed_ms'])} | {fmt(d['underrun_ms'])} | {fmt(d['underruns'])} | "
                     f"{fmt(d['over_budget_ratio'])} | {fmt(d['rtf_p99'])} | {fmt(d['latency_ms_p99'])} | {flag} |")
    for ep, t5 in (doc.get("t5") or {}).items():
        L.append("")
        L.append(f"## T5 (scorer v2), {ep}: {t5.get('clips')} clips")
        L.append("")
        L.append("| Row | Corpus mean | 95 % CI | Limit (Low latency / Quality) | Status |")
        L.append("|---|---|---|---|---|")
        for r in t5["rows"]:
            ci = r.get("ci95")
            ci_s = f"{fmt(ci[0])} .. {fmt(ci[1])}" if ci else ""
            L.append(f"| {md_escape(r['label'])} | {fmt(r['mean'])} | {ci_s} | {fmt(r['limit_low_latency'])} / {fmt(r['limit_quality'])} | {'met' if r['met'] else 'missed'} |")
        if t5.get("metrics"):
            L.append("")
            L.append("| Metric | Mean | 95 % CI |")
            L.append("|---|---|---|")
            for k, m in t5["metrics"].items():
                ci = m.get("ci95")
                L.append(f"| {k} | {fmt(m.get('mean'))} | {fmt(ci[0]) + ' .. ' + fmt(ci[1]) if ci else ''} |")
    pf = doc.get("paired_f0")
    if pf:
        L.append("")
        L.append(f"## CPU F0 on the same {len(pf['clips'])} clips: {pf['b']} against {pf['a']} (paired per clip; difference = {pf['b']} minus {pf['a']}, 95 % bootstrap interval of the mean)")
        L.append("")
        L.append(f"| Metric | {pf['a']} | {pf['b']} | Difference | Clips where {pf['b']} is higher |")
        L.append("|---|---|---|---|---|")
        for r in pf["rows"]:
            iv = f" [{signed(r['lo'])}, {signed(r['hi'])}]" if r.get("lo") is not None else ""
            L.append(f"| {md_escape(r['label'])} | {fmt(r['a_mean'])} | {fmt(r['b_mean'])} | {signed(r['diff'])}{iv} | {r['b_higher']} of {r['n']} |")
    if doc.get("stages"):
        L.append("")
        L.append("## Per call, by stage (median over the passes)")
        L.append("")
        cols = list(STAGE_COLUMNS)
        L.append("| Spec | " + " | ".join(cols) + " | reports |")
        L.append("|---|" + "---|" * (len(cols) + 1))
        for spec, row in doc["stages"].items():
            L.append(f"| {spec} | " + " | ".join(fmt(row.get(c)) for c in cols) + f" | {row.get('_reports')} |")
    if doc.get("decisions"):
        L.append("")
        L.append("## Decision sessions (ABBA; B is the candidate)")
        for d in doc["decisions"]:
            L.append("")
            L.append(f"### {d['name']}")
            L.append("")
            sw = []
            if d.get("compare_vars"):
                sw.append(f"vars `{d['compare_vars']}`")
            if d.get("compare_env"):
                sw.append(f"env `{d['compare_env']}`")
            L.append("B = " + (", ".join(sw) or "the same configuration") + f"; rounds {json.dumps(d.get('rounds'))}; GPU or host state flags: {d.get('state_flags')}.")
            L.append("")
            L.append("| Spec | Run | Metric | n | A median | B median | B - A | Change % | Interval % | Verdict |")
            L.append("|---|---|---|---|---|---|---|---|---|---|")
            for p in d["rows"]:
                iv = f"{fmt(p.get('lo_pct'))} .. {fmt(p.get('hi_pct'))}" if p.get("lo_pct") is not None else ""
                L.append(f"| {p['spec']} | {p['run']} | {p['metric']} | {p.get('n')} | {fmt(p.get('a_median'))} | {fmt(p.get('b_median'))} | {signed(p.get('b_minus_a'), 3)} | {fmt(p.get('median_pct'))} | {iv} | {p.get('verdict')} {md_escape(p.get('note') or '')}".rstrip() + " |")
            if any(p.get("linear") for p in d["rows"]):
                L.append("")
                L.append("Rows noted `linear` are judged on a linear scale with a floor (the change is relative to max(|A median|, 0.25), as the gate prints it): "
                         "for counts near zero and ratios in 0..1 such as `underruns` and `over_budget_ratio` that hides a real change, so read the A and B medians and B - A there, not the percentage or the verdict.")
    if doc.get("load"):
        L.append("")
        L.append("## Load log (every step; `disturbed` = a compiler or a busy desktop beside it)")
        L.append("")
        L.append("| Step | Pre CPU % | Foreign CPU mean / p90 % | Sigil cores | rustc max | Disturbed |")
        L.append("|---|---|---|---|---|---|")
        for s in doc["load"]:
            why = "; ".join(s["reasons"]) if s["disturbed"] else "no"
            if s.get("heavy"):
                why += " (busy programs seen: " + ", ".join(s["heavy"]) + ")"
            L.append(f"| {md_escape(s['step'])} | {fmt(s.get('pre_cpu_pct'), 3)} | {fmt(s.get('foreign_cpu_mean_pct'), 3)} / {fmt(s.get('foreign_cpu_p90_pct'), 3)} | {fmt(s.get('sigil_cores_mean'), 3)} | {s['rustc_max']} | {md_escape(why)} |")
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------
def build(root: str) -> dict:
    rows = target_rows(root)
    passes = sorted({p for r in rows for p in r["passes"]})
    env = {}
    env_path = os.path.join(root, "env.json")
    if os.path.isfile(env_path):
        try:
            env = read_json(env_path)
        except (OSError, ValueError):
            env = {}
    load = load_report(root)
    decisions = decision_rows(root)
    apply_context_session(rows, decisions)
    store = ReportStore(root)
    derived = derived_rows(root, store)
    delivery = delivery_table(root, store)
    annotate_delivery(rows, delivery)
    # The derived rows read the same reports, so they get the same disturbance and GPU-clock annotations; one
    # call each keeps the high-clock threshold the same for every row.
    annotate_disturbance(rows + derived, load)
    clock_threshold = attach_regime(root, rows + derived)
    disturbed_steps = {s["step"] for s in load if s["disturbed"]}

    def tally(rs: list[dict]) -> dict:
        return {
            "rows": len(rs),
            "met": sum(1 for r in rs if r["status"] == "met"),
            "missed": sum(1 for r in rs if r["status"] == "MISSED"),
            "not_measured": sum(1 for r in rs if r["status"] == "not measured"),
            "skipped": sum(1 for r in rs if r["status"] == "skipped"),
            # a limit that could not be used: never counted as met or missed
            "unparsed": sum(1 for r in rs if r["status"] in (LIMIT_NOT_PARSED, NO_LIMIT)),
        }

    doc = {
        "schema": SCHEMA,
        "environment": env,
        "passes": passes,
        "high_clock_threshold_mhz": clock_threshold,
        "rows": rows,
        "derived_rows": derived,
        "delivery": delivery,
        "t5": t5_rows(root),
        "paired_f0": paired_f0(root),
        "stages": stage_table(root, ("sve-dml", "sve-cuda", "app-dml", "app-cuda", "sve-cpu", "sve-cpu-pool", "sve-cpu-t6")),
        "decisions": decisions,
        "load": load,
        "disturbed_steps": sorted(s for s in disturbed_steps if s),
        "counts": tally(rows),
        "derived_counts": tally(derived),
    }
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="the campaign folder")
    ap.add_argument("--json", default="", help="write the scorecard JSON here (default <root>/final-scorecard.json)")
    ap.add_argument("--md", default="", help="write the scorecard markdown here (default <root>/final-scorecard.md)")
    ap.add_argument("--publish", action="store_true", help="write the markdown with the banner and introduction of the committed copy (scripts/vc-bench/final/)")
    args = ap.parse_args(argv)
    if not os.path.isdir(args.root):
        print(f"no such folder: {args.root}", file=sys.stderr)
        return 2
    doc = build(args.root)
    json_path = args.json or os.path.join(args.root, "final-scorecard.json")
    md_path = args.md or os.path.join(args.root, "final-scorecard.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(render_markdown(doc, publish=args.publish))
    c = doc["counts"]
    d = doc["derived_counts"]
    print(f"{c['rows']} rows: {c['met']} met, {c['missed']} missed, {c['not_measured']} not measured, {c['skipped']} skipped, "
          f"{c['unparsed']} with a limit that could not be read; {len(doc['disturbed_steps'])} disturbed step(s)")
    print(f"{d['rows']} derived rows: {d['met']} met, {d['missed']} missed, {d['not_measured']} not measured")
    flagged = [f"{x['spec']}/{x['run']} ({100 * x['played_ratio']:.0f} % played)" for x in doc["delivery"] if x["flag"]]
    if flagged:
        print("runs that lost audio: " + ", ".join(flagged))
    if c["unparsed"] or d["unparsed"]:
        print("WARNING: a row has a limit the scorecard could not read (status 'limit not parsed' / 'no limit'); fix the gate's limit format or parse_limit", file=sys.stderr)
    print(json_path)
    print(md_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
