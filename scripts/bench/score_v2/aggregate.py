# SPDX-License-Identifier: AGPL-3.0-or-later
"""Corpus-level summary of scorer-v2 quality runs (P3B-B, ADR 0009).

Reads every ``report.json`` under the given roots, keeps the ``quality`` runs scored with
``scoring.analyzer: v2`` (one clip each; the headline metrics are medians over the clip's offsets),
merges in the target-speaker similarities of v2 ``rescore`` runs over their saved pairs (the corpus
runner's leave-one-out ``rescore-whole``), and prints per metric the corpus mean with a percentile-bootstrap 95 % CI
(10 000 resamples, fixed seed) and the clip count; a metric measured on one clip is flagged. It also
averages the oracle-ablation tables (what each ablation recovered, per ceiling) and the floors, and
checks the T5 v2 rows against the limits of ADR 0009.

Usage: ``python scripts/bench/score_v2/aggregate.py <root> [<root> ...] [--json out.json] [--md out.md]``
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as cm  # noqa: E402

# ADR 0009 (Low latency; Quality in parentheses). Pre-registered, fixed numbers: the fine pitch
# limit is 15 cents because the analyzer floor measured with the RMVPE tracker (the analyzer's own
# model: ``whole_fed_input_fpe_cents``) was below 10 cents; it is never moved by a corpus's floor.
LIMITS = {
    "speaker_set_sim_delta": (">=", -0.02, -0.01),
    "fed_input_fpe_cents": ("<=", 15.0, 15.0),
    "fed_input_gpe_pooled": ("<=", 0.003, 0.003),
    "fed_input_vuv_error": ("<=", 0.03, 0.03),
    "env_mcd_over_floor_db": ("<=", 0.6, 0.3),
    "clicks_binomial": ("<=", 0.0, 0.0),
    "clicks_narrow": ("<=", 0.0, 0.0),
}

HEADLINE = [
    "speaker_cosine", "speaker_cosine_v2", "speaker_set_sim", "ceiling_speaker_set_sim", "whole_speaker_set_sim", "speaker_set_sim_delta",
    "mcd_db", "env_mcd_db", "mcd_offset_corrected_db", "env_mcd_offset_corrected_db",
    "mcd_floor_db", "env_mcd_floor_db", "mcd_floor_seed_db", "env_mcd_floor_seed_db", "mcd_offline_floor_db", "env_mcd_offline_floor_db",
    "env_mcd_db_vs_long_shape", "env_mcd_over_floor_db", "mcd_db_vs_b", "env_mcd_db_vs_b", "mcd_db_vs_whole", "env_mcd_db_vs_whole", "mcd_db_vs_whole_fp32",
    "f0_rmse_cents", "f0_fine_rmse_cents", "f0_gross_frames", "gross_pitch_error", "vuv_error",
    "fed_fpe_cents", "fed_gpe", "fed_vuv_error", "fed_onset_lag_frames",
    "fed_input_fpe_cents", "fed_input_gpe", "fed_input_vuv_error", "fed_input_onset_lag_frames", "fed_input_onsets_missed",
    "ceiling_fed_input_fpe_cents", "ceiling_fed_input_gpe", "ceiling_fed_input_vuv_error",
    "whole_fed_input_fpe_cents", "whole_fed_input_gpe", "whole_fed_input_vuv_error",
    "clicks", "clicks_binomial", "clicks_narrow", "artifact_events",
    "dnsmos_ovrl", "ceiling_dnsmos_ovrl", "ceiling_b_dnsmos_ovrl", "whole_dnsmos_ovrl", "whole_fp32_dnsmos_ovrl",
    "onset_rel_db", "onset_rel_db_long_shape", "onset_rel_db_b", "onset_rel_db_whole",
    "speaker_cosine_v2_spread", "env_mcd_db_spread", "mcd_db_spread", "fed_input_fpe_cents_spread",
]


SET_METRICS = ("speaker_set_sim", "ceiling_speaker_set_sim", "speaker_set_sim_delta")


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def load_runs(roots):
    """The v2 quality runs, with the target-speaker similarities of v2 rescore runs merged in.

    A rescore pair is matched to the quality run whose folder holds its stream WAV; the median of
    the set metrics over that folder's pairs (one per offset; leave one clip out over the rescore's
    clips) fills the quality run's row unless that run measured them itself
    (``fidelity.reference_clips``).
    """
    runs, rescores = [], []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "report.json"), recursive=True)):
            try:
                rep = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            spec = rep.get("spec", {}) or {}
            clip = (spec.get("input", {}) or {}).get("path", "")
            for run in rep.get("runs", []):
                d = run.get("details") or {}
                if d.get("scorer") != "v2":
                    continue
                if run.get("mode") == "quality":
                    runs.append({"report": path, "clip": os.path.basename(clip), "metrics": dict(run.get("metrics", {})), "details": d,
                                 "status": run.get("status"), "engine": rep.get("engine", {}) or {}})
                elif run.get("mode") == "rescore":
                    folders = {}
                    for sr in spec.get("runs", []):
                        if sr.get("name") == run.get("name"):
                            folders = {p["name"]: _norm(os.path.dirname(p["stream"])) for p in sr.get("pairs", [])}
                    rescores.append((folders, d.get("pairs", [])))
    # Reports of builds before the whole_fed_input_* metrics: the median over the offsets of the
    # details' whole-vs-input errors (the analyzer floor of the fine pitch error).
    for r in runs:
        if "whole_fed_input_fpe_cents" in r["metrics"]:
            continue
        rows = [((o.get("fed_f0") or {}).get("whole_vs_input") or {}) for o in r["details"].get("offsets", [])]
        for key, name in (("fpe", "whole_fed_input_fpe_cents"), ("gpe", "whole_fed_input_gpe"), ("vuv", "whole_fed_input_vuv_error")):
            v = sorted(x[key] for x in rows if x.get(key) is not None)
            if v:
                n = len(v)
                r["metrics"][name] = v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])
    by_folder = {_norm(os.path.dirname(r["report"])): r for r in runs}
    for folders, pairs in rescores:
        found = {}
        for p in pairs:
            folder = folders.get(p.get("name"), "")
            if folder not in by_folder:
                continue
            for k in SET_METRICS:
                if k in p.get("metrics", {}):
                    found.setdefault((folder, k), []).append(p["metrics"][k])
        for (folder, k), v in found.items():
            r = by_folder[folder]
            if k not in r["metrics"]:
                r["metrics"][k] = _median(v)
                if len(v) > 1:
                    r["metrics"].setdefault(k + "_offsets", len(v))
    return runs


def _median(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def summarize(runs):
    out = {"clips": len(runs), "metrics": {}, "ablations": {}, "t5_v2": {}}
    for k in HEADLINE:
        v = [r["metrics"][k] for r in runs if k in r["metrics"]]
        if not v:
            continue
        lo, hi = cm.bootstrap_ci(v)
        out["metrics"][k] = {"mean": sum(v) / len(v), "ci95": [lo, hi], "n": len(v), "single_clip": len(v) == 1}
    # Ablations: mean recovered per ablation and ceiling.
    acc = {}
    for r in runs:
        table = r["details"].get("ablations") or {}
        for row in table.get("rows", []):
            for ceiling, rec in (row.get("recovered") or {}).items():
                for metric, val in rec.items():
                    if metric == "seconds" or val is None:
                        continue
                    acc.setdefault((row["ablation"], ceiling, metric), []).append(val)
        for ceiling, base in (table.get("baseline") or {}).items():
            for metric, val in base.items():
                if metric != "seconds" and val is not None:
                    acc.setdefault(("baseline", ceiling, metric), []).append(val)
    for (abl, ceiling, metric), v in sorted(acc.items()):
        lo, hi = cm.bootstrap_ci(v)
        out["ablations"].setdefault(abl, {}).setdefault(ceiling, {})[metric] = {"mean": sum(v) / len(v), "ci95": [lo, hi], "n": len(v)}
    # Corpus-level gross pitch error, pooled: gross frames over voiced-in-both frames of every
    # offset of every clip (the mean of per-clip shares weighs a short clip like a long one).
    pooled = {}
    for r in runs:
        for off in r["details"].get("offsets", []):
            for key, name in (("vs_input", "fed_input_gpe_pooled"), ("vs_ceiling", "fed_gpe_pooled")):
                e = (off.get("fed_f0") or {}).get(key) or {}
                if "voiced_both" in e:
                    g, n = pooled.get(name, (0, 0))
                    pooled[name] = (g + len(e.get("gross") or []), n + e["voiced_both"])
    for name, (g, n) in pooled.items():
        if n:
            out["metrics"][name] = {"mean": g / n, "ci95": [g / n, g / n], "n": n, "single_clip": False, "pooled_frames": n, "gross_frames": g}
    # The analyzer floor of the fine pitch error per tracker (the run's tracker over the whole clip
    # against the analyzer): with RMVPE (the analyzer's model) it is the floor ADR 0009 applied the
    # plan's rule to; with another tracker it is that tracker's disagreement with the analyzer, a
    # diagnostic, not a floor. The limits stay the pre-registered numbers either way.
    by_tracker = {}
    for r in runs:
        v = r["metrics"].get("whole_fed_input_fpe_cents")
        if v is not None:
            by_tracker.setdefault(r["engine"].get("pitch_extractor") or "?", []).append(v)
    for tracker, v in sorted(by_tracker.items()):
        lo, hi = cm.bootstrap_ci(v)
        out.setdefault("fpe_analyzer_floor_cents", {})[tracker] = {"mean": sum(v) / len(v), "ci95": [lo, hi], "n": len(v)}
    for k, (op, low, quality) in LIMITS.items():
        m = out["metrics"].get(k)
        if not m:
            continue
        ok = (m["mean"] >= low) if op == ">=" else (m["mean"] <= low)
        out["t5_v2"][k] = {"mean": m["mean"], "limit_low_latency": low, "limit_quality": quality, "met_low_latency": ok, "n": m["n"]}
    return out


def markdown(s):
    lines = [f"Clips: {s['clips']}", "", "| metric | mean | 95 % CI | n |", "|---|---|---|---|"]
    for k, m in s["metrics"].items():
        flag = " (single clip)" if m["single_clip"] else ""
        lines.append(f"| {k} | {m['mean']:.4g} | {m['ci95'][0]:.4g} .. {m['ci95'][1]:.4g} | {m['n']}{flag} |")
    if s["ablations"]:
        lines += ["", "Ablations (mean distance for `baseline`, mean recovered for the others; positive = closer):", "",
                  "| ablation | ceiling | MCD dB | env MCD dB | cosine v2 | V/UV | fine F0 cents | n |", "|---|---|---|---|---|---|---|---|"]
        for abl, per in s["ablations"].items():
            for ceiling, m in per.items():
                g = lambda k: f"{m[k]['mean']:+.3f}" if k in m else ""  # noqa: E731
                n = max((v["n"] for v in m.values()), default=0)
                lines.append(f"| {abl} | {ceiling} | {g('mcd_db')} | {g('env_mcd_db')} | {g('speaker_cosine_v2')} | {g('vuv_error')} | {g('f0_fine_rmse_cents')} | {n} |")
    for tracker, f in (s.get("fpe_analyzer_floor_cents") or {}).items():
        lines += ["", f"Fine pitch analyzer floor ({tracker} tracker, whole clip vs the analyzer): {f['mean']:.2f} cents "
                  f"[{f['ci95'][0]:.2f}, {f['ci95'][1]:.2f}] (n {f['n']})"]
    if s["t5_v2"]:
        lines += ["", "| T5 v2 row | corpus mean | limit (Low latency / Quality) | met (Low latency) |", "|---|---|---|---|"]
        for k, t in s["t5_v2"].items():
            lines.append(f"| {k} | {t['mean']:.4g} | {t['limit_low_latency']:.4g} / {t['limit_quality']:.4g} | {'yes' if t['met_low_latency'] else 'no'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--json", default="")
    ap.add_argument("--md", default="")
    args = ap.parse_args()
    runs = load_runs(args.roots)
    if not runs:
        raise SystemExit("no scorer-v2 quality runs found")
    s = summarize(runs)
    s["runs"] = [{"report": r["report"], "clip": r["clip"], "status": r["status"]} for r in runs]
    md = markdown(s)
    print(md)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(s, f, indent=1)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)


if __name__ == "__main__":
    main()
