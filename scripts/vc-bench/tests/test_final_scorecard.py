# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-checks of the P7-B final-scorecard tooling (no GPU, no executable, no network):

    python -m pytest scripts/vc-bench/tests/test_final_scorecard.py -q

Covers the limit parser and the gate's status words, the median-over-the-passes rule for the target
rows, the T5 v2 rows from a corpus summary, the decision (interleaved) rows, the load log's
disturbed-step rule, the markdown, the command line, and the `final` perf-gate profile: every
template exists, every `${VAR}` of every template materializes to valid JSON with the variables the
profile gives it, and every target names a spec and a run that the materialized specs contain.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
VC_BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(VC_BENCH))
BENCH = os.path.join(REPO, "scripts", "bench")
sys.path.insert(0, VC_BENCH)

import final_scorecard as fs  # noqa: E402


# ---- helpers ---------------------------------------------------------------------------------
def write_json(path, doc):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def gate_doc(targets):
    return {"passed": True, "profile": "final", "mode": "single", "targets": targets, "paired": [], "specs": []}


def target(id_, source, limit, value, label="row", gate=False, status=None):
    if status is None:
        status = "not measured" if value is None else "met"
    return {"id": id_, "label": label, "source": source, "limit": limit, "value": value, "status": status, "gate": gate}


# ---- limits and words ------------------------------------------------------------------------
def test_limits_parse_the_gate_strings():
    assert fs.parse_limit("<= 85") == (None, 85.0)
    assert fs.parse_limit(">= 0.6") == (0.6, None)
    assert fs.parse_limit("-1 .. 1") == (-1.0, 1.0)
    # PowerShell prints a small limit with an exponent
    assert fs.parse_limit("<= 1E-05") == (None, 1e-05)
    assert fs.parse_limit("-1.5e+2 .. 2") == (-150.0, 2.0)
    # an empty limit is "no limit"; any other text the parser does not know is an error, not "no limit"
    assert fs.parse_limit("") == (None, None) and fs.parse_limit(None) == (None, None)
    for bad in ("nonsense", "< 85", "> 0.6", "<= n/a", "85", "<= 85 ms", "1 .. "):
        try:
            fs.parse_limit(bad)
        except fs.LimitError:
            continue
        raise AssertionError(f"parse_limit({bad!r}) read an unknown limit format as a limit")
    assert fs.limit_bounds("< 85") == (None, None)


def test_a_value_is_met_only_against_a_limit_that_was_read():
    assert fs.judge_limit(10.0, "<= 85") == ("met", None)
    assert fs.judge_limit(100.0, "<= 85") == ("MISSED", None)
    status, err = fs.judge_limit(100.0, "< 85")
    assert status == "limit not parsed" and "< 85" in err  # not "met", whatever the value is
    assert fs.judge_limit(10.0, "<= 1e-05")[0] == "MISSED"
    assert fs.judge_limit(10.0, "") == ("no limit", None)
    assert fs.judge_limit(None, "< 85")[0] == "not measured"


def test_a_row_with_an_unreadable_limit_is_counted_apart_and_never_met():
    with tempfile.TemporaryDirectory() as root:
        for name in ("pass1", "pass2"):
            write_json(os.path.join(root, "gate", name, "g", "gate-summary.json"), gate_doc([
                target("T1", "sve-cuda/paced latency_ms_p99", "< 85", 10.0, label="strict limit", status="met"),
                target("T2a", "sve-cuda/offline call_ms_p95", "<= 6", 5.0, label="plain", status="met"),
            ]))
        rows = {r["label"]: r for r in fs.target_rows(root)}
        assert rows["strict limit"]["status"] == "limit not parsed" and "< 85" in rows["strict limit"]["limit_error"]
        assert rows["plain"]["status"] == "met" and rows["plain"]["limit_error"] is None
        assert fs.status_word(rows["strict limit"]) == "limit not parsed"
        assert fs.main([root]) == 0
        doc = json.load(open(os.path.join(root, "final-scorecard.json"), encoding="utf-8"))
        assert doc["counts"]["met"] == 1 and doc["counts"]["unparsed"] == 1 and doc["counts"]["missed"] == 0
        md = open(os.path.join(root, "final-scorecard.md"), encoding="utf-8").read()
        assert "limit not parsed" in md and "cannot read the limit" in md


def test_status_words_follow_the_gate():
    assert fs.judge(80.0, None, 85.0) == "met"
    assert fs.judge(85.0, None, 85.0) == "met"  # on the limit is met, as in the gate
    assert fs.judge(85.1, None, 85.0) == "MISSED"
    assert fs.judge(0.5, 0.6, None) == "MISSED"
    assert fs.judge(-2.0, -1.0, 1.0) == "MISSED"
    assert fs.judge(None, None, 1.0) == "not measured"


def test_median_and_format():
    assert fs.median([3.0, 1.0, 2.0]) == 2.0
    assert fs.median([1.0, 2.0, 3.0, 10.0]) == 2.5
    assert fs.median([None, 4.0]) == 4.0
    assert fs.median([]) is None
    assert fs.fmt(None) == "n/a"
    assert fs.fmt(132.5913) == "132.6"
    assert fs.fmt(0.0) == "0"
    assert fs.fmt(2500.4) == "2500"
    assert fs.fmt(3) == "3"


# ---- target rows: the median over the passes -------------------------------------------------
def test_a_target_is_judged_on_the_median_of_its_passes_not_on_one_pass():
    with tempfile.TemporaryDirectory() as root:
        # T2a offline p95 <= 6 ms: passes read 5.9, 7.4 and 7.6: the median 7.4 misses although one pass met it.
        for name, v in (("pass1", 5.9), ("pass2", 7.4), ("pass3", 7.6)):
            write_json(os.path.join(root, "gate", name, "g", "gate-summary.json"), gate_doc([
                target("T2a", "sve-cuda/offline call_ms_p95", "<= 6", v, label="T2a CUDA offline", gate=True,
                       status="met" if v <= 6 else "MISSED"),
                target("T9", "load-cuda/offline engine_load_ms", "<= 2500", 1000.0 + v, label="T9 CUDA"),
            ]))
        rows = fs.target_rows(root)
        t2a = next(r for r in rows if r["id"] == "T2a")
        assert t2a["value"] == 7.4 and t2a["min"] == 5.9 and t2a["max"] == 7.6 and t2a["n"] == 3
        assert t2a["status"] == "MISSED" and t2a["all_passes_agree"] is False
        t9 = next(r for r in rows if r["id"] == "T9")
        assert t9["status"] == "met" and t9["all_passes_agree"] is True
        assert fs.status_word(t9) == "met" and fs.status_word(t2a) == "missed"


def test_not_measured_and_skipped_rows_stay_distinct_and_provisional_rows_say_so():
    with tempfile.TemporaryDirectory() as root:
        write_json(os.path.join(root, "gate", "once", "g", "gate-summary.json"), gate_doc([
            target("T4", "silent-cpu/silent cpu_cores", "<= 0.02", None, label="silent", status="not measured"),
            target("T7", "soak-cuda/soak underruns", "<= 0", None, label="soak", status="skipped"),
            target("T1", "sve-dml/h1-sim latency_ms_p99", "<= 110", 148.2, label="H1 (provisional H1)", status="MISSED"),
        ]))
        rows = {r["id"]: r for r in fs.target_rows(root)}
        assert rows["T4"]["status"] == "not measured"
        assert rows["T7"]["status"] == "skipped"
        assert rows["T1"]["status"] == "MISSED" and rows["T1"]["provisional"] is True
        assert fs.status_word(rows["T1"]) == "missed (provisional)"


def test_decision_sessions_are_not_scorecard_rows():
    with tempfile.TemporaryDirectory() as root:
        write_json(os.path.join(root, "decisions", "devblock", "gate-summary.json"),
                   {"mode": "interleaved", "compare_vars": "BLOCK_FRAMES=480", "compare_env": "", "rounds": {"devblock-cuda": 4},
                    "targets": [target("T1", "devblock-cuda/paced latency_ms_p99", "<= 85", 115.2)],
                    "paired": [{"spec": "devblock-cuda", "run": "paced", "metric": "latency_ms_p99", "n": 4, "a_median": 132.6,
                                "b_median": 115.2, "median_pct": -13.1, "lo_pct": -14.8, "hi_pct": -12.0, "verdict": "IMPROVED",
                                "note": "", "regime_mismatched": 0}]}
                   )
        assert fs.target_rows(root) == []
        d = fs.decision_rows(root)
        assert d[0]["name"] == "devblock" and d[0]["compare_vars"] == "BLOCK_FRAMES=480"
        assert d[0]["rows"][0]["verdict"] == "IMPROVED" and d[0]["rows"][0]["median_pct"] == -13.1


# ---- T5 ---------------------------------------------------------------------------------------
def test_t5_rows_carry_the_adr_0009_limits_per_execution_provider():
    with tempfile.TemporaryDirectory() as root:
        write_json(os.path.join(root, "corpus", "cuda", "summary.json"), {
            "clips": 18,
            "metrics": {"speaker_cosine_v2": {"mean": 0.93, "ci95": [0.92, 0.94], "n": 18}},
            "t5_v2": {
                "speaker_set_sim_delta": {"mean": -0.054, "limit_low_latency": -0.02, "limit_quality": -0.01, "met_low_latency": False, "n": 18},
                "clicks_binomial": {"mean": 0.0, "limit_low_latency": 0.0, "limit_quality": 0.0, "met_low_latency": True, "n": 18},
            },
        })
        t5 = fs.t5_rows(root)
        assert list(t5) == ["cuda"] and t5["cuda"]["clips"] == 18
        by = {r["metric"]: r for r in t5["cuda"]["rows"]}
        assert by["speaker_set_sim_delta"]["met"] is False and by["clicks_binomial"]["met"] is True
        assert "identity delta" in by["speaker_set_sim_delta"]["label"]
        assert t5["cuda"]["metrics"]["speaker_cosine_v2"]["mean"] == 0.93


def clip_report(root, corpus, clip, metrics, scorer="v2", mode="quality"):
    """A corpus folder's per-clip report: <root>/corpus/<corpus>/fid2-<corpus>-<clip>/report.json."""
    path = os.path.join(root, "corpus", corpus, f"fid2-{corpus}-{clip}", "report.json")
    write_json(path, {
        "spec": {"input": {"path": f"C:/clips/{clip}.wav"}},
        "runs": [{"name": "fidelity", "mode": mode, "details": {"scorer": scorer}, "metrics": metrics}],
    })


def test_the_cpu_f0_comparison_pairs_the_clips_both_corpora_ran():
    with tempfile.TemporaryDirectory() as root:
        # SwiftF0 ran four clips, FCPE three of them: the fourth is not compared.
        for clip, cos, fpe in (("a", 0.80, 10.0), ("b", 0.82, 12.0), ("c", 0.84, 14.0), ("d", 0.9, 50.0)):
            clip_report(root, "cpu", clip, {"speaker_cosine_v2": cos, "fed_input_fpe_cents": fpe, "fed_input_vuv_error": 0.4, "dnsmos_ovrl": 2.5})
        for clip, cos, fpe in (("a", 0.88, 30.0), ("b", 0.86, 34.0), ("c", 0.90, 38.0)):
            clip_report(root, "cpu-fcpe", clip, {"speaker_cosine_v2": cos, "fed_input_fpe_cents": fpe, "fed_input_vuv_error": 0.08, "dnsmos_ovrl": 2.8})
        pf = fs.paired_f0(root)
        assert pf is not None and pf["clips"] == ["a.wav", "b.wav", "c.wav"] and pf["a"] == "cpu" and pf["b"] == "cpu-fcpe"
        by = {r["metric"]: r for r in pf["rows"]}
        cos = by["speaker_cosine_v2"]
        assert cos["n"] == 3 and abs(cos["a_mean"] - 0.82) < 1e-9 and abs(cos["b_mean"] - 0.88) < 1e-9
        assert abs(cos["diff"] - 0.06) < 1e-9 and cos["b_higher"] == 3
        assert cos["lo"] <= cos["diff"] <= cos["hi"]
        # a percentage-point metric is scaled to points; a metric no run measured is left out
        assert abs(by["fed_input_vuv_error"]["diff"] - (-32.0)) < 1e-9 and by["fed_input_vuv_error"]["b_higher"] == 0
        assert abs(by["fed_input_fpe_cents"]["diff"] - 22.0) < 1e-9
        assert "output F0 RMSE (cents)" not in [r["label"] for r in pf["rows"]]


def test_the_cpu_f0_comparison_needs_both_corpora_and_scorer_v2_quality_runs():
    with tempfile.TemporaryDirectory() as root:
        assert fs.paired_f0(root) is None  # no corpus at all
        for clip in ("a", "b", "c"):
            clip_report(root, "cpu", clip, {"dnsmos_ovrl": 2.5})
        assert fs.paired_f0(root) is None  # no FCPE corpus
        for clip in ("a", "b", "c"):
            clip_report(root, "cpu-fcpe", clip, {"dnsmos_ovrl": 2.8}, scorer="v1")  # not a scorer-v2 run: ignored
        assert fs.paired_f0(root) is None
        for clip in ("a", "b"):
            clip_report(root, "cpu-fcpe", clip, {"dnsmos_ovrl": 2.8})
        assert fs.paired_f0(root) is None  # two shared clips are too few for an interval
        clip_report(root, "cpu-fcpe", "c", {"dnsmos_ovrl": 2.8})
        assert fs.paired_f0(root) is not None


def test_the_bootstrap_interval_is_seeded_and_brackets_the_mean():
    values = [0.1, 0.4, 0.2, 0.5, 0.3, 0.35, 0.15, 0.45, 0.25]
    lo, hi = fs.bootstrap_mean_ci(values)
    assert lo < sum(values) / len(values) < hi
    assert fs.bootstrap_mean_ci(values) == (lo, hi)  # the same seed, the same interval
    assert fs.bootstrap_mean_ci([1.0]) is None


def test_the_cpu_f0_comparison_reaches_the_json_and_the_markdown():
    with tempfile.TemporaryDirectory() as root:
        for clip in ("a", "b", "c"):
            clip_report(root, "cpu", clip, {"dnsmos_ovrl": 2.5})
            clip_report(root, "cpu-fcpe", clip, {"dnsmos_ovrl": 2.8})
        assert fs.main([root]) == 0
        doc = json.load(open(os.path.join(root, "final-scorecard.json"), encoding="utf-8"))
        assert doc["paired_f0"]["rows"][0]["metric"] == "dnsmos_ovrl"
        md = open(os.path.join(root, "final-scorecard.md"), encoding="utf-8").read()
        assert "CPU F0 on the same 3 clips: cpu-fcpe against cpu" in md and "| DNSMOS OVRL |" in md


# ---- the load log -----------------------------------------------------------------------------
def load_rows(foreign_pct, sigil_cores=2.0, threads=16, n=30, rustc=0):
    cpu_pct = foreign_pct + 100.0 * sigil_cores / threads
    return [{"t": float(i), "cpu_pct": cpu_pct, "sigil_cores": sigil_cores, "rustc": rustc, "cargo": 0, "heavy": []} for i in range(n)]


def test_a_quiet_step_is_not_flagged_and_sigils_own_cores_do_not_count_as_foreign_load():
    s = fs.step_load({"name": "quiet", "pre": {"cpu_pct": 5.0, "rustc": 0, "quiet": True}}, load_rows(8.0, sigil_cores=6.0), 16)
    assert s["disturbed"] is False and abs(s["foreign_cpu_mean_pct"] - 8.0) < 1e-9
    assert s["sigil_cores_mean"] == 6.0


def test_a_compiler_a_busy_desktop_or_a_missed_quiet_window_flags_the_step():
    assert fs.step_load({"name": "a"}, load_rows(5.0, rustc=2), 16)["disturbed"] is True
    busy = fs.step_load({"name": "b"}, load_rows(40.0), 16)
    assert busy["disturbed"] is True and "foreign CPU" in busy["reasons"][0]
    burst = load_rows(5.0)
    for r in burst[:6]:
        r["cpu_pct"] += 70.0
    assert fs.step_load({"name": "c"}, burst, 16)["disturbed"] is True  # a burst: p90 over the limit
    late = fs.step_load({"name": "d", "pre": {"cpu_pct": 60.0, "rustc": 0, "quiet": False}}, load_rows(5.0), 16)
    assert late["disturbed"] is True and "quiet window" in late["reasons"][0]
    heavy = load_rows(5.0)
    heavy[3]["heavy"] = ["ffmpeg"]
    seen = fs.step_load({"name": "e"}, heavy, 16)
    assert seen["heavy"] == ["ffmpeg"] and seen["disturbed"] is False  # a busy program is listed; the CPU share decides


def test_load_csv_and_steps_round_trip_through_the_report():
    with tempfile.TemporaryDirectory() as root:
        with open(os.path.join(root, "steps.jsonl"), "w", encoding="utf-8") as f:
            f.write(json.dumps({"name": "pass1-sve-cuda", "kind": "gate", "exit": 0, "pre": {"cpu_pct": 4.0, "rustc": 0, "quiet": True}}) + "\n")
            f.write("\n")
            f.write(json.dumps({"name": "pass1-sve-dml", "kind": "gate", "exit": 0}) + "\n")
        write_json(os.path.join(root, "env.json"), {"logical_processors": 16})
        os.makedirs(os.path.join(root, "load"))
        with open(os.path.join(root, "load", "pass1-sve-cuda.csv"), "w", encoding="utf-8", newline="") as f:
            f.write("t_unix,cpu_total_pct,sigil_cores,rustc,cargo,heavy_other\n")
            for i in range(10):
                f.write(f"{1790000000 + 2 * i},{12.5 + (80 if i == 4 else 0)},1.5,{1 if i == 4 else 0},0,{'ffmpeg;obs64' if i == 7 else ''}\n")
        report = fs.load_report(root)
        assert [s["step"] for s in report] == ["pass1-sve-cuda", "pass1-sve-dml"]
        first, second = report
        assert first["samples"] == 10 and first["rustc_max"] == 1 and first["disturbed"] is True
        assert first["heavy"] == ["ffmpeg", "obs64"]
        assert second["samples"] == 0 and second["disturbed"] is False  # no samples is not evidence of a disturbance


def test_samples_outside_a_substeps_window_or_under_another_agents_lock_do_not_count():
    base = 1790000000.0
    iso = lambda t: __import__("datetime").datetime.fromtimestamp(t, __import__("datetime").timezone.utc).isoformat()  # noqa: E731
    rows = []
    for i in range(30):
        owner = "P7-B-pass1" if 10 <= i < 20 else "P7-A-stt-screen"
        busy = 5.0 if 10 <= i < 20 else 90.0
        rows.append({"t": base + 2 * i, "cpu_pct": busy, "sigil_cores": 0.0, "rustc": 0, "cargo": 0, "heavy": [], "lock_owner": owner})
    # A sampled step: the harness took the lock per run, the other agent's 90 % between runs is not ours.
    step = {"name": "corpus-cuda", "kind": "sampled", "lock_tag": "P7-B"}
    s = fs.step_load(step, rows, 16)
    assert s["samples"] == 10 and s["samples_other_lock_owner"] == 20 and s["disturbed"] is False
    # A sub-step: only its own time window of the parent's file (and, with a lock tag, only our samples).
    mine = {"name": "pass1/sve-cuda", "kind": "sub", "csv": "pass1", "lock_tag": "P7-B", "start": iso(base + 20), "end": iso(base + 38)}
    s_mine = fs.step_load(mine, rows, 16)
    assert 9 <= s_mine["samples"] <= 11 and s_mine["disturbed"] is False
    theirs = {"name": "pass1/x", "kind": "sub", "csv": "pass1", "lock_tag": "P7-B", "start": iso(base + 0), "end": iso(base + 18)}
    assert fs.step_load(theirs, rows, 16)["samples"] == 0  # the window holds only the other agent's samples
    untagged = dict(theirs)
    untagged.pop("lock_tag")
    s_all = fs.step_load(untagged, rows, 16)
    assert 9 <= s_all["samples"] <= 11 and s_all["disturbed"] is True  # without the owner filter that load counts
    # PowerShell writes seven fractional digits.
    assert abs(fs.parse_time("2026-10-09T18:30:12.1234567-04:00") - fs.parse_time("2026-10-09T22:30:12.123456+00:00")) < 1e-6
    assert fs.parse_time("garbage") is None and fs.parse_time("") is None


def test_rows_say_which_passes_were_disturbed_and_give_the_clean_median():
    rows = [
        {"id": "T2a", "source": "sve-cuda/offline call_ms_p95", "passes": ["pass1", "pass2", "pass3"], "by_pass": {"pass1": 7.4, "pass2": 12.0, "pass3": 7.6}},
        {"id": "T9", "source": "load-cuda/offline engine_load_ms", "passes": ["once"], "by_pass": {"once": 1000.0}},
    ]
    load = [
        {"kind": "sub", "pass": "pass1", "specs": ["sve-cuda"], "disturbed": False},
        {"kind": "sub", "pass": "pass2", "specs": ["sve-cuda"], "disturbed": True},
        {"kind": "sub", "pass": "pass2", "specs": ["sve-dml"], "disturbed": False},
        {"kind": "sub", "pass": "pass3", "specs": ["sve-cuda"], "disturbed": False},
        {"kind": "sub", "pass": "once", "specs": ["load-dml", "load-cuda"], "disturbed": True},
        {"kind": "locked", "name": "pass1", "disturbed": True},  # a parent step does not mark a row by itself
    ]
    fs.annotate_disturbance(rows, load)
    assert rows[0]["disturbed_in"] == ["pass2"] and rows[0]["n_clean"] == 2 and rows[0]["value_clean"] == 7.5
    assert rows[1]["disturbed_in"] == ["once"] and rows[1]["value_clean"] is None and rows[1]["n_clean"] == 0


def test_rows_carry_the_gpu_clock_of_each_pass_and_the_high_clock_median():
    with tempfile.TemporaryDirectory() as root:
        # CUDA offline call p95 in three passes: the GPU sat at 1515 MHz in the first, at 2790 MHz in the others.
        for name, v, sm in (("pass1", 12.0, 1515.0), ("pass2", 7.5, 2790.0), ("pass3", 7.3, 2775.0)):
            d = os.path.join(root, "gate", name, "sve-cuda")
            write_json(os.path.join(d, "gate-summary.json"), gate_doc([
                target("T2a", "sve-cuda/offline call_ms_p95", "<= 6", v, label="T2a CUDA offline", gate=True, status="MISSED"),
                target("T2", "sve-dml-ctx512/offline call_ms_p95 / sve-dml/offline call_ms_p95", "<= 1.3", 1.0, label="ratio"),
            ]))
            write_json(os.path.join(d, "sve-cuda", "report.json"),
                       {"runs": [{"name": "offline", "metrics": {"call_ms_p95": v, "state_sm_clock_mhz": sm}}]})
        rows = fs.target_rows(root)
        thr = fs.attach_regime(root, rows)
        assert abs(thr - 0.85 * 2790.0) < 1e-6
        t2a = next(r for r in rows if r["id"] == "T2a")
        assert t2a["sm_by_pass"] == {"pass1": 1515.0, "pass2": 2790.0, "pass3": 2775.0}
        assert t2a["value"] == 7.5 and t2a["value_high"] == 7.4 and t2a["n_high"] == 2  # median of 7.5 and 7.3
        ratio = next(r for r in rows if r["label"] == "ratio")
        assert ratio["sm_by_pass"] == {} and fs.high_cell(ratio) == ""  # a ratio row has no single run to read the clock of
        assert fs.high_cell(t2a) == "7.4 (2)"
        assert fs.parse_source("sve-cuda/offline call_ms_p95") == ("sve-cuda", "offline", "call_ms_p95")
        assert fs.parse_source("a/b m / c/d m") is None


def test_rows_say_when_another_process_shared_the_gpu():
    def series(vals):
        return [{"other_gpu_util_pct": v} for v in vals]
    with tempfile.TemporaryDirectory() as root:
        for name, shared in (("pass1", [30.0, 40.0, 50.0]), ("pass2", [0.0, 2.0, 1.0])):
            d = os.path.join(root, "gate", name, "sve-dml")
            write_json(os.path.join(d, "gate-summary.json"), gate_doc([
                target("T2a", "sve-dml/paced call_ms_p95", "<= 6", 20.0, label="paced", status="MISSED"),
                target("T2b", "sve-dml/h2-load70-oop call_ms_p95", "<= 10", 12.0, label="oop", status="MISSED"),
            ]))
            write_json(os.path.join(d, "sve-dml", "report.json"), {"runs": [
                {"name": "paced", "metrics": {"state_sm_clock_mhz": 2790.0}, "details": {"state": {"series": series(shared)}}},
                {"name": "h2-load70-oop", "metrics": {"state_sm_clock_mhz": 2790.0}, "details": {"state": {"series": series([60.0, 70.0])}}},
            ]})
        rows = fs.target_rows(root)
        fs.attach_regime(root, rows)
        paced = next(r for r in rows if r["label"] == "paced")
        assert paced["foreign_gpu_by_pass"] == {"pass1": 40.0, "pass2": 1.0} and paced["gpu_shared_in"] == ["pass1"]
        oop = next(r for r in rows if r["label"] == "oop")
        assert oop["foreign_gpu_by_pass"] == {} and oop["gpu_shared_in"] == []  # our own load process is not foreign
        doc = {"rows": rows, "high_clock_threshold_mhz": 2000.0, "passes": ["pass1", "pass2"]}
        assert "GPU shared in pass1 (40 %)" in fs.render_markdown(doc)


def test_the_context_scaling_row_comes_from_the_paired_session():
    with tempfile.TemporaryDirectory() as root:
        label = "extra context 64 -> 512 ms raises the per-block compute p95 by <= 30 % (DirectML)"
        write_json(os.path.join(root, "gate", "once", "ctx512", "gate-summary.json"), gate_doc([
            target("T2", "sve-dml-ctx512/offline call_ms_p95 / sve-dml/offline", "<= 1.3", None, label=label, status="not measured")]))
        write_json(os.path.join(root, "decisions", "ctx512", "gate-summary.json"), {
            "mode": "interleaved", "compare_vars": "EXTRA_MS=512", "rounds": {"ctx-dml": 6}, "targets": [],
            "paired": [{"spec": "ctx-dml", "run": "offline", "metric": "call_ms_p95", "n": 6, "a_median": 8.0, "b_median": 8.4,
                        "median_pct": 5.0, "lo_pct": 1.0, "hi_pct": 9.0, "verdict": "NEUTRAL"}]})
        rows = fs.target_rows(root)
        assert rows[0]["status"] == "not measured"  # the gate alone cannot judge a ratio across passes
        fs.apply_context_session(rows, fs.decision_rows(root))
        assert abs(rows[0]["value"] - 1.05) < 1e-9 and rows[0]["status"] == "met" and rows[0]["n"] == 6
        # A session where 512 ms costs 40 % more misses the 30 % limit.
        slow = [{"name": "ctx512", "rows": [{"spec": "ctx-dml", "run": "offline", "metric": "call_ms_p95", "n": 6, "a_median": 8.0, "b_median": 11.2}]}]
        again = fs.target_rows(root)
        fs.apply_context_session(again, slow)
        assert again[0]["status"] == "MISSED"
        # Without the session the row is left as the gate had it.
        untouched = fs.target_rows(root)
        fs.apply_context_session(untouched, [])
        assert untouched[0]["status"] == "not measured"


def test_without_any_clock_record_the_rows_have_no_high_clock_value():
    with tempfile.TemporaryDirectory() as root:
        write_json(os.path.join(root, "gate", "pass1", "load-cpu", "gate-summary.json"), gate_doc([
            target("T9", "load-cpu/offline engine_load_ms", "<= 2500", 700.0, label="T9 CPU")]))
        rows = fs.target_rows(root)
        assert fs.attach_regime(root, rows) is None
        assert rows[0]["value_high"] is None and rows[0]["n_high"] == 0


# ---- the stage table, the markdown and the command line --------------------------------------
def report_doc(call_p95, latency_p99):
    return {"runs": [
        {"name": "offline", "metrics": {"call_ms_p50": call_p95 - 1, "call_ms_p95": call_p95, "call_ms_p99": call_p95 + 1, "cpu_cores": 0.2, "vram_engine_mb": 1300.0}},
        {"name": "paced", "metrics": {"latency_ms_p50": latency_p99, "latency_ms_p99": latency_p99, "underruns": 0, "cpu_cores": 0.04}},
    ]}


def test_the_stage_table_takes_the_median_over_the_passes_reports():
    with tempfile.TemporaryDirectory() as root:
        for name, call, lat in (("pass1", 7.0, 132.6), ("pass2", 8.0, 132.6), ("pass3", 9.0, 140.0)):
            write_json(os.path.join(root, "gate", name, "g", "gate-summary.json"), gate_doc([]))
            write_json(os.path.join(root, "gate", name, "g", "sve-cuda", "report.json"), report_doc(call, lat))
        row = fs.stage_table(root, ["sve-cuda", "sve-dml"])
        assert row["sve-cuda"]["call_p95_ms"] == 8.0 and row["sve-cuda"]["paced_latency_p99_ms"] == 132.6
        assert row["sve-cuda"]["_reports"] == 3 and row["sve-dml"]["_reports"] == 0 and row["sve-dml"]["call_p95_ms"] is None


def test_the_command_line_writes_the_json_and_the_markdown():
    with tempfile.TemporaryDirectory() as root:
        write_json(os.path.join(root, "env.json"), {"exe_sha256": "ab" * 32, "gpu": "RTX 4090", "cpu": "Ryzen 7 7800X3D", "logical_processors": 16})
        for name, v in (("pass1", 132.6), ("pass2", 132.6)):
            write_json(os.path.join(root, "gate", name, "g", "gate-summary.json"), gate_doc([
                target("T1", "sve-cuda/paced latency_ms_p99", "<= 85", v, label="T1 CUDA | paced", status="MISSED"),
            ]))
        assert fs.main([root]) == 0
        doc = json.load(open(os.path.join(root, "final-scorecard.json"), encoding="utf-8"))
        assert doc["schema"] == fs.SCHEMA and doc["counts"]["missed"] == 1 and doc["passes"] == ["pass1", "pass2"]
        md = open(os.path.join(root, "final-scorecard.md"), encoding="utf-8").read()
        assert "| T1 | T1 CUDA \\| paced |" in md and "missed" in md and "132.6" in md
        assert fs.main([os.path.join(root, "missing")]) == 2


# ---- what the output queue delivered, and the rows judged from the reports --------------------------
def paced_run(name, emitted, played, trimmed_ms=0.0, underrun_ms=0.0, underruns=0, **metrics):
    m = {"output_trimmed_ms": trimmed_ms, "underrun_ms": underrun_ms, "underruns": underruns, "rtf_p99": 0.9,
         "latency_ms_p99": 277.0, "over_budget_ratio": 0.0, "cpu_cores": 3.5}
    m.update(metrics)
    return {"name": name, "metrics": m, "details": {"emitted_frames": emitted, "played_frames": played}}


def write_spec_report(root, pass_name, spec, runs, targets=()):
    d = os.path.join(root, "gate", pass_name, spec)
    write_json(os.path.join(d, "gate-summary.json"), gate_doc(list(targets)))
    write_json(os.path.join(d, spec, "report.json"), {"runs": runs})


EMITTED = 2_879_488  # 60 s at 48 kHz, as the paced runs of the campaign emitted


def test_the_played_share_is_played_over_emitted_frames():
    assert abs(fs.played_ratio(paced_run("paced", EMITTED, 1_742_560)) - 0.6052) < 1e-3
    assert fs.played_ratio({"details": {"emitted_frames": 0, "played_frames": 0}}) is None
    assert fs.played_ratio({"details": {}}) is None and fs.played_ratio({}) is None
    assert fs.derived_metric(paced_run("paced", 100, 50, underruns=7), "underruns") == 7.0
    assert fs.derived_metric(paced_run("paced", 100, 50), "no_such_metric") is None


def test_a_cpu_row_that_is_met_on_latency_but_loses_audio_says_so():
    """P7-B review: the CPU pool and 6-thread rows read `met` on latency and cores while the production output
    queue threw away 40-50 % of the converted audio (the latency is that of the audio that survived)."""
    with tempfile.TemporaryDirectory() as root:
        for name, played in (("pass1", 1_742_560), ("pass2", 1_730_000), ("pass3", 1_790_000)):
            write_spec_report(root, name, "sve-cpu-pool", [
                paced_run("paced", EMITTED, played, trimmed_ms=23_347.0, underrun_ms=23_355.0, underruns=2685),
                paced_run("h3-game", EMITTED, played, trimmed_ms=26_419.0, underrun_ms=26_426.0, underruns=2992),
            ], targets=[
                target("T1", "sve-cpu-pool/h3-game latency_ms_p99", "<= 280", 277.9, label="pool under game load (provisional H3)"),
                target("T4", "sve-cpu-pool/paced cpu_cores", "<= 3.5", 3.1, label="pool cores"),
            ])
            # a GPU run that delivered everything
            write_spec_report(root, name, "sve-dml", [paced_run("paced", EMITTED, 2_877_000)],
                              targets=[target("T1", "sve-dml/paced latency_ms_p99", "<= 85", 132.6, label="dml", status="MISSED")])
        doc = fs.build(root)
        by = {(d["spec"], d["run"]): d for d in doc["delivery"]}
        assert by[("sve-cpu-pool", "h3-game")]["flag"] is True and by[("sve-dml", "paced")]["flag"] is False
        assert 0.60 < by[("sve-cpu-pool", "paced")]["played_ratio"] < 0.62 and by[("sve-cpu-pool", "paced")]["underruns"] == 2685
        rows = {r["label"]: r for r in doc["rows"]}
        lat = rows["pool under game load (provisional H3)"]
        assert lat["status"] == "met" and lat["delivery_flag"] is True
        assert fs.status_word(lat) == "met (provisional), but audio lost"
        assert "played only 61 %" in lat["delivery_note"] and "trimmed" in lat["delivery_note"]
        assert fs.status_word(rows["pool cores"]) == "met, but audio lost"  # the paced run lost audio too
        assert rows["dml"]["delivery_flag"] is False and fs.status_word(rows["dml"]) == "missed"
        # the derived rows judge the share itself: the pool misses it, the GPU row meets it
        der = {(r["source"]): r for r in doc["derived_rows"]}
        assert der["sve-cpu-pool/h3-game played_ratio"]["status"] == "MISSED"
        assert der["sve-cpu-pool/paced played_ratio"]["status"] == "MISSED"
        assert der["sve-dml/paced played_ratio"]["status"] == "met"
        assert der["sve-cpu/paced played_ratio"]["status"] == "not measured"  # no such report in this campaign
        assert doc["derived_counts"]["missed"] >= 2 and doc["derived_counts"]["met"] >= 1
        md = fs.render_markdown(doc)
        assert "audio lost" in md and "Derived rows" in md and "Delivery of the converted audio by run" in md
        assert "F0 is SwiftF0" in md  # the CPU rows say which F0 they ran


def test_a_cpu_row_that_delivers_its_audio_is_not_flagged():
    with tempfile.TemporaryDirectory() as root:
        for name in ("pass1", "pass2"):
            write_spec_report(root, name, "sve-cpu-t6", [paced_run("paced", EMITTED, 2_870_000), paced_run("h3-game", EMITTED, 2_860_000)],
                              targets=[target("T1", "sve-cpu-t6/h3-game latency_ms_p99", "<= 280", 250.0, label="t6 (provisional H3)")])
        doc = fs.build(root)
        row = doc["rows"][0]
        assert row["delivery_flag"] is False and fs.status_word(row) == "met (provisional)"
        assert not any(d["flag"] for d in doc["delivery"])
        derived = {r["source"]: r for r in doc["derived_rows"]}
        assert derived["sve-cpu-t6/h3-game played_ratio"]["status"] == "met"


def test_the_over_budget_half_of_the_h1_simulation_row_is_judged_from_the_reports():
    """The profile only gates the p99 of the call under the H1 simulation; the over-budget half of the row was
    claimed `met` without a target. The median of the passes decides, the worst pass is shown."""
    def campaign(root, ratios):
        for (name, ratio) in zip(("pass1", "pass2", "pass3"), ratios):
            write_spec_report(root, name, "sve-dml", [paced_run("h1-sim", EMITTED, 2_870_000, over_budget_ratio=ratio, underruns=297)])

    with tempfile.TemporaryDirectory() as root:
        campaign(root, [0.0, 0.0, 0.01191])  # only the low-clock pass 3 is over budget
        doc = fs.build(root)
        row = next(r for r in doc["derived_rows"] if r["source"] == "sve-dml/h1-sim over_budget_ratio")
        assert row["limit"] == "<= 0.001" and row["value"] == 0.0 and row["status"] == "met"
        assert row["max"] == 0.01191 and row["all_passes_agree"] is False  # the spread shows the pass that was over
        assert row["provisional"] is True
        assert "0.01191" in fs.render_markdown(doc)
    with tempfile.TemporaryDirectory() as root:
        campaign(root, [0.0, 0.002, 0.01191])  # two of three passes over: the median is over the limit
        row = next(r for r in fs.build(root)["derived_rows"] if r["source"] == "sve-dml/h1-sim over_budget_ratio")
        assert row["value"] == 0.002 and row["status"] == "MISSED"


def test_decision_rows_carry_the_difference_and_the_linear_floor():
    """P7-B review: `over_budget_ratio` 0.0123 -> 0.019 and underruns 1.5 -> 0.5 read NEUTRAL (a linear scale with a
    0.25 floor); the table shows the medians, B - A and a footnote instead of leaving the verdict alone."""
    with tempfile.TemporaryDirectory() as root:
        write_json(os.path.join(root, "decisions", "devblock-cuda", "gate-summary.json"), {
            "mode": "interleaved", "compare_vars": "BLOCK_FRAMES=480", "rounds": {"devblock-cuda": 4}, "targets": [],
            "paired": [
                {"spec": "devblock-cuda", "run": "paced", "metric": "over_budget_ratio", "n": 4, "a_median": 0.0123, "b_median": 0.019,
                 "median_pct": 2.7, "lo_pct": -2.24, "hi_pct": 7.97, "verdict": "NEUTRAL",
                 "note": "linear (values <= 0; relative to max(|A median|, 0.25))"},
                {"spec": "devblock-cuda", "run": "h2-load70-oop", "metric": "underruns", "n": 4, "a_median": 26.5, "b_median": 6.5,
                 "median_pct": -75.3, "lo_pct": -87.2, "hi_pct": -55.3, "verdict": "IMPROVED", "note": ""},
            ]})
        d = fs.decision_rows(root)[0]
        ob, un = d["rows"]
        assert abs(ob["b_minus_a"] - 0.0067) < 1e-9 and ob["linear"] is True
        assert un["b_minus_a"] == -20.0 and un["linear"] is False
        md = fs.render_markdown({"rows": [], "decisions": [d], "passes": []})
        assert "| B - A |" in md and "+0.0067" in md and "-20" in md
        assert "Rows noted `linear`" in md


def test_the_published_copy_carries_its_banner_and_introduction():
    with tempfile.TemporaryDirectory() as root:
        write_json(os.path.join(root, "gate", "pass1", "g", "gate-summary.json"), gate_doc([target("T1", "sve-cuda/paced latency_ms_p99", "<= 85", 80.0)]))
        out = os.path.join(root, "published.md")
        assert fs.main([root, "--md", out, "--publish"]) == 0
        text = open(out, encoding="utf-8").read()
        assert text.startswith("<!-- Generated by scripts/vc-bench/final_scorecard.py")
        assert "The complete tables behind [VC_BENCHMARKS.md" in text and "none did" in text
        plain = os.path.join(root, "plain.md")
        assert fs.main([root, "--md", plain]) == 0
        assert not open(plain, encoding="utf-8").read().startswith("<!--")


# ---- the final perf-gate profile --------------------------------------------------------------
PROFILE = os.path.join(BENCH, "profiles", "final.json")


def materialize(template, vars_):
    """What run-perf-gate.ps1 does to a template file: `*_JSON` raw, the rest as JSON string contents."""
    def sub(m):
        k = m.group(1)
        v = vars_.get(k, "")
        if k.endswith("_JSON"):
            return v if v else "null"
        return json.dumps(str(v).replace("\\", "/"))[1:-1]
    return re.sub(r"\$\{([A-Z0-9_]+)\}", sub, template)


def profile_specs():
    prof = json.load(open(PROFILE, encoding="utf-8"))
    out = {}
    fake = {"SLOT": "C:/v/slot", "CLIP": "C:/c/clip.wav", "KNN_INDEX": "C:/k/index.npy", "SOAK": "1", "DECIDE": "1"}
    for t in prof["templates"]:
        tvars = {k.upper(): str(v) for k, v in (t.get("vars") or {}).items()}
        tvars["NAME"] = t["name"]
        merged = {**{k: str(v) for k, v in prof["vars"].items()}, **tvars, **fake}
        path = os.path.normpath(os.path.join(BENCH, t["file"]))
        text = open(path, encoding="utf-8").read()
        out[t["name"]] = (t, json.loads(materialize(text, merged)), text)
    return prof, out


def test_every_template_of_the_final_profile_exists_and_materializes_to_a_valid_spec():
    prof, specs = profile_specs()
    names = [t["name"] for t in prof["templates"]]
    assert len(names) == len(set(names)), "duplicate template names"
    for name, (t, spec, text) in specs.items():
        assert spec["schema"] == "sigil-bench-spec/1" and spec["name"] == name
        assert spec["engine"]["kind"] == "rvc" and spec["runs"], name
        # Every placeholder the template uses has a value in the profile or the template's own vars
        # (an unset `${X}` would materialize to an empty string and fail late, inside the bench).
        used = set(re.findall(r"\$\{([A-Z0-9_]+)\}", text))
        have = {k.upper() for k in prof["vars"]} | {k.upper() for k in (t.get("vars") or {})} | {"NAME", "SLOT", "CLIP", "KNN_INDEX"}
        assert used <= have, f"{name}: no value for {sorted(used - have)}"
        for dep in t["requires"]:
            assert dep in ("SLOT", "CLIP", "KNN_INDEX", "SOAK", "DECIDE"), dep


def test_gated_work_only_runs_when_asked_for():
    prof, _ = profile_specs()
    by = {t["name"]: t for t in prof["templates"]}
    assert "SOAK" in by["soak-cuda"]["requires"]  # a plain scorecard run never starts the 30 minute soak
    for n in ("devblock-dml", "devblock-cuda", "devblock-cpu", "dml-shared", "cpu-streamer-lean", "cpu-t6-lean"):
        assert "DECIDE" in by[n]["requires"], n
    for n in ("sve-dml", "sve-cuda", "sve-cpu", "sve-cpu-pool", "sve-cpu-t6", "app-dml", "app-cuda", "load-dml", "load-cuda", "load-cpu", "vram-dml", "vram-cuda"):
        assert not ({"SOAK", "DECIDE"} & set(by[n]["requires"])), n


def test_every_target_names_a_spec_and_run_the_profile_produces():
    prof, specs = profile_specs()
    runs = {name: {r["name"] for r in spec["runs"]} for name, (_, spec, _) in specs.items()}
    assert prof["targets"], "no targets"
    ids = {t["id"] for t in prof["targets"]}
    assert {"T1", "T2a", "T2b", "T3", "T4", "T7", "T9"} <= ids
    for t in prof["targets"]:
        refs = [t] if "ratio" not in t else [t["ratio"]["num"], t["ratio"]["den"]]
        for ref in refs:
            assert ref["spec"] in runs, f"{t['label']}: unknown spec {ref['spec']}"
            assert ref["run"] in runs[ref["spec"]], f"{t['label']}: spec {ref['spec']} has no run {ref['run']}"
        assert ("max" in t) or ("min" in t), t["label"]
        lo, hi = t.get("min"), t.get("max")
        assert lo is None or hi is None or lo <= hi


def test_the_plain_gate_specs_exist_and_the_new_templates_keep_the_app_default_shape():
    prof, specs = profile_specs()
    for s in prof["specs"]:
        assert os.path.isfile(os.path.join(BENCH, s)), s
    # The app's own shape: block_ms 0 (the engine block follows the device block), the Low-latency preset.
    for n in ("app-dml", "app-cuda"):
        eng = specs[n][1]["engine"]
        assert eng["block_ms"] == 0 and eng["preset"] == "low_latency" and eng["pitch_extractor"] == "rmvpe"
        assert specs[n][1]["stream"]["extra_context_ms"] == 96  # the Low-latency preset's context (presets.ts)
    # The reference rows keep the plan's 40 ms blocks.
    assert specs["sve-cuda"][1]["engine"]["block_ms"] == 40 and specs["sve-dml"][1]["stream"]["extra_context_ms"] == 64
    soak = specs["soak-cuda"][1]["runs"][0]
    assert soak["seconds"] == 1800 and soak["stalls"] == [] and soak["load"]["gpu"]["out_of_process"] is True


def test_every_derived_target_names_a_spec_and_run_the_profile_produces():
    """The rows judged from the reports must read runs the profile's specs really contain (a rename would
    otherwise turn them into silent `not measured` rows)."""
    _, specs = profile_specs()
    runs = {name: {r["name"] for r in spec["runs"]} for name, (_, spec, _) in specs.items()}
    assert fs.DERIVED_TARGETS
    seen = set()
    for t in fs.DERIVED_TARGETS:
        assert t["spec"] in runs, f"{t['label']}: unknown spec {t['spec']}"
        assert t["run"] in runs[t["spec"]], f"{t['label']}: spec {t['spec']} has no run {t['run']}"
        assert ("min" in t) != ("max" in t), t["label"]
        seen.add((t["spec"], t["run"], t["metric"]))
        fs.parse_limit(fs.derived_limit(t))  # the limit text is in the gate's own form
    assert ("sve-cpu-t6", "h3-game", "played_ratio") in seen and ("sve-dml", "h1-sim", "over_budget_ratio") in seen
    for spec in fs.DELIVERY_SPECS:
        assert spec in runs, spec
        assert {r for r in fs.DELIVERY_RUNS} & runs[spec], f"{spec} has none of the delivery runs"


# ---- the driver's helper functions (PowerShell; skipped where pwsh is not installed) ---------------
PWSH = shutil.which("pwsh")
DRIVER = os.path.join(VC_BENCH, "final_scorecard.ps1")


def run_driver_functions(names, body):
    """Run PowerShell `body` after loading the named functions of final_scorecard.ps1 (their definitions are
    parsed out of the file; the script itself, which would start a campaign, is not run). Returns stdout lines."""
    quoted = ",".join(f"'{n}'" for n in names)
    prelude = (
        "$ErrorActionPreference = 'Stop'; Set-StrictMode -Version 3.0; "
        f"$tok = $null; $err = $null; $ast = [System.Management.Automation.Language.Parser]::ParseFile('{DRIVER}', [ref]$tok, [ref]$err); "
        f"foreach ($fn in @({quoted})) {{ "
        "$def = $ast.FindAll({ param($a) $a -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $a.Name -eq $fn }.GetNewClosure(), $true) | Select-Object -First 1; "
        "if ($null -eq $def) { throw \"no function $fn in the driver\" }; Invoke-Expression $def.Extent.Text }; "
    )
    r = subprocess.run([PWSH, "-NoProfile", "-NonInteractive", "-Command", prelude + body], capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, f"pwsh failed:\n{r.stdout}\n{r.stderr}"
    return [line.strip() for line in r.stdout.splitlines() if line.strip()]


needs_pwsh = pytest.mark.skipif(PWSH is None, reason="pwsh is not installed")


@needs_pwsh
def test_a_decision_summary_counts_only_when_this_run_wrote_it_with_every_round(tmp_path):
    out = str(tmp_path).replace("\\", "/")
    body = f"""
$d = '{out}'
function Show($name, $value) {{ "$name=$value" }}
$summary = Join-Path $d 'gate-summary.json'
Show 'no_file' (Test-DecisionComplete -OutDir $d)
Set-Content -LiteralPath $summary -Value '{{"paired":[{{"spec":"a"}}],"rounds":{{"a":4,"b":4}}}}'
Show 'any_summary' (Test-DecisionComplete -OutDir $d)
Show 'all_rounds' (Test-DecisionComplete -OutDir $d -Specs @('a','b') -Rounds 4)
Show 'fewer_rounds_ran' (Test-DecisionComplete -OutDir $d -Specs @('a','b') -Rounds 5)
Show 'spec_missing' (Test-DecisionComplete -OutDir $d -Specs @('a','c') -Rounds 4)
Show 'stale_for_this_run' (Test-DecisionComplete -OutDir $d -Since (Get-Date).AddMinutes(5) -Specs @('a') -Rounds 4)
Show 'fresh_for_this_run' (Test-DecisionComplete -OutDir $d -Since (Get-Date).AddMinutes(-5) -Specs @('a') -Rounds 4)
Set-Content -LiteralPath $summary -Value '{{"paired":[],"rounds":{{"a":4}}}}'
Show 'no_paired_rows' (Test-DecisionComplete -OutDir $d -Specs @('a') -Rounds 4)
Set-Content -LiteralPath $summary -Value '{{"paired":[{{"spec":"a"}}]}}'
Show 'no_rounds_field' (Test-DecisionComplete -OutDir $d -Specs @('a') -Rounds 1)
Set-Content -LiteralPath $summary -Value 'not json at all'
Show 'garbage' (Test-DecisionComplete -OutDir $d)
"""
    got = dict(line.split("=", 1) for line in run_driver_functions(["Test-DecisionComplete"], body))
    assert got == {
        "no_file": "False", "any_summary": "True", "all_rounds": "True", "fewer_rounds_ran": "False", "spec_missing": "False",
        "stale_for_this_run": "False", "fresh_for_this_run": "True", "no_paired_rows": "False", "no_rounds_field": "False", "garbage": "False",
    }


@needs_pwsh
def test_a_rerun_deletes_the_summary_of_an_earlier_run_before_the_gate_starts(tmp_path):
    out = str(tmp_path).replace("\\", "/")
    for name in ("gate-summary.json", "gate-summary.md", "gate.log", "keep.txt"):
        (tmp_path / name).write_text("x", encoding="utf-8")
    body = f"Remove-DecisionSummary -OutDir '{out}'; Remove-DecisionSummary -OutDir '{out}/does-not-exist'; 'done'"
    assert run_driver_functions(["Remove-DecisionSummary"], body) == ["done"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["gate.log", "keep.txt"]
    # The driver calls it before every decision gate and judges a nonzero exit with the fresh summary only.
    text = open(DRIVER, encoding="utf-8").read()
    assert text.index("Remove-DecisionSummary -OutDir $out") < text.index("Invoke-Gate -OutDir $out -Only $only -Extra $extra")
    assert "Test-DecisionComplete -OutDir $out -Since $began -Specs $only -Rounds $wantRounds" in text


@needs_pwsh
def test_a_private_corpus_lock_left_by_a_stopped_run_is_cleared_unless_its_owner_lives(tmp_path):
    d = str(tmp_path).replace("\\", "/")
    body = f"""
$d = '{d}'
function Show($name, $value) {{ "$name=$value" }}
$lock = Join-Path $d 'corpus.lock'
Show 'no_lock' (Clear-StalePrivateLock -Path $lock)
Set-Content -LiteralPath $lock -Value '{{"tag":"P7-B-corpus","pid":2000000000,"start_unix_ms":1}}'
Show 'dead_owner_removed' (Clear-StalePrivateLock -Path $lock)
Show 'dead_owner_file_gone' (-not (Test-Path -LiteralPath $lock))
Set-Content -LiteralPath $lock -Value ('{{"tag":"P7-B-corpus","pid":' + $PID + '}}')
Show 'live_owner_kept' (Clear-StalePrivateLock -Path $lock)
Show 'live_owner_file_stays' (Test-Path -LiteralPath $lock)
Set-Content -LiteralPath $lock -Value 'P6-B 18:02'
Show 'hand_written_removed' (Clear-StalePrivateLock -Path $lock)
Set-Content -LiteralPath $lock -Value '{{"tag":"x"}}'
Show 'no_pid_removed' (Clear-StalePrivateLock -Path $lock)
"""
    got = dict(line.split("=", 1) for line in run_driver_functions(["Clear-StalePrivateLock"], body))
    assert got == {
        "no_lock": "False", "dead_owner_removed": "True", "dead_owner_file_gone": "True", "live_owner_kept": "False",
        "live_owner_file_stays": "True", "hand_written_removed": "True", "no_pid_removed": "True",
    }
    text = open(DRIVER, encoding="utf-8").read()
    assert text.index("Clear-StalePrivateLock -Path $priv") < text.index("$env:SIGIL_GPU_LOCK = $priv")  # cleared before the runner takes it


@needs_pwsh
def test_the_driver_script_parses_and_defines_the_t6_fcpe_session():
    lines = run_driver_functions(["Get-DecisionDefs"], "$script:DecideRounds = 4; $DecideRounds = 4; $d = Get-DecisionDefs; ($d.Keys -join ','); $d['cpu-t6-f0'].only; $d['cpu-t6-f0'].vars; $d['cpu-t6-f0'].runs")
    assert "cpu-t6-f0" in lines[0].split(",") and "cpu-f0" in lines[0].split(",")
    assert lines[1] == "cpu-t6-lean" and lines[2] == "PITCH=fcpe"
    runs = json.loads(lines[3])
    assert [r["name"] for r in runs] == ["offline", "paced", "h3-game"]


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
