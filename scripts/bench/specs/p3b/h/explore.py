#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Interleaved exploration runner for single bench specs (P3B-H: output-cushion tuning).

Round-robin over arms (the order reverses on odd rounds, ABBA style), one bench process per run, the GPU
timing lock held per run. An arm sets dotted spec fields ("stream.output_cushion") and environment
variables; `SIGIL_VC_CUSHION_TRACE` is pointed at a file beside each run's report (one line per shed or
underrun and a status line every 5 s of device time). Prints a table of medians per arm and writes
<out>/results.json.

    python explore.py --exe EXE --spec SPEC.json --out DIR --rounds 3 --arms ARMS.json --appdata APPDATA [--run paced]

ARMS.json: [{"name": "fixed", "set": {"stream.output_cushion": "fixed"}, "env": {}},
            {"name": "adaptive", "set": {"stream.output_cushion": "adaptive"}, "env": {"SIGIL_VC_CUSHION_MARGIN_MS": "3"}}]
SPEC.json is a materialized paced spec (the `specs/` folder of a run-perf-gate.ps1 run).
"""
import argparse
import copy
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.abspath(os.path.join(HERE, "..", "..", "..", "gpu-lock.ps1"))
KEYS = ("latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "latency_ms_max", "underruns", "output_trimmed_ms", "call_ms_p50",
        "call_ms_p95", "call_ms_p99", "t1_at_p50_output_ms", "t1_output_ms_p50", "join_step_ms_p99", "over_budget_ratio")


def setpath(d, path, value):
    cur = d
    parts = path.split(".")
    for p in parts[:-1]:
        cur = cur[p]
    cur[parts[-1]] = value


def lock(args, verb, tag):
    cmd = ["pwsh", "-NoProfile", "-File", LOCK]
    if verb == "acquire":
        cmd += ["-Acquire", tag, "-Wait", "600", "-PollSeconds", "2", "-ExpectedMinutes", str(args.minutes), "-OwnerPid", str(os.getpid())]
    else:
        cmd += ["-Release", tag]
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--run", default=None, help="name of the run whose metrics are read (default: the first)")
    ap.add_argument("--minutes", type=int, default=4, help="expected minutes of one run (the lock file's estimate)")
    ap.add_argument("--tag", default="P3B-H")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--appdata", default=os.environ.get("SIGIL_APP_DATA_DIR", ""), help="the app-data folder with the voice and the model store")
    args = ap.parse_args()
    if not args.appdata:
        ap.error("--appdata (or SIGIL_APP_DATA_DIR) is required")
    base = json.load(open(args.spec, encoding="utf-8"))
    arms = json.load(open(args.arms, encoding="utf-8"))
    os.makedirs(args.out, exist_ok=True)
    results = {a["name"]: [] for a in arms}
    for r in range(args.rounds):
        order = arms if r % 2 == 0 else list(reversed(arms))
        for a in order:
            rundir = os.path.join(args.out, f"{a['name']}-r{r + 1}")
            rep = os.path.join(rundir, "report.json")
            if not os.path.exists(rep) or not args.resume:
                os.makedirs(rundir, exist_ok=True)
                spec = copy.deepcopy(base)
                spec["name"] = f"{base.get('name', 'explore')}-{a['name']}"
                for k, v in a.get("set", {}).items():
                    setpath(spec, k, v)
                specpath = os.path.join(rundir, "spec.json")
                json.dump(spec, open(specpath, "w", encoding="utf-8"), indent=1)
                env = dict(os.environ)
                env.update({k: str(v) for k, v in a.get("env", {}).items()})
                env["SIGIL_APP_DATA_DIR"] = os.path.abspath(args.appdata)
                env["SIGIL_VC_CUSHION_TRACE"] = os.path.join(rundir, "cushion.trace")
                tag = f"{args.tag} explore {a['name']} r{r + 1}"
                if lock(args, "acquire", tag) != 0:
                    print("could not take the GPU lock", file=sys.stderr)
                    return 1
                try:
                    t = time.time()
                    with open(os.path.join(rundir, "bench.log"), "w") as log:
                        subprocess.run([args.exe, "--headless", "bench", specpath, "--out", rundir], env=env, stdout=log, stderr=subprocess.STDOUT)
                    print(f"[{time.strftime('%H:%M:%S')}] {a['name']} r{r + 1}: {time.time() - t:.0f} s", flush=True)
                finally:
                    lock(args, "release", tag)
            try:
                report = json.load(open(rep, encoding="utf-8"))
                runs = report["runs"]
                run = next((x for x in runs if args.run in (None, x["name"])), runs[0])
                m = run["metrics"]
                results[a["name"]].append({k: m.get(k) for k in KEYS})
            except Exception as e:  # noqa: BLE001
                print(f"  {a['name']} r{r + 1}: no report ({e})", flush=True)
    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    print()
    print(f"{'arm':<22}{'n':>3}" + "".join(f"{k.replace('latency_ms_', 'lat_').replace('output_trimmed_ms', 'trim'):>12}" for k in KEYS[:7]))
    for name, rows in results.items():
        if not rows:
            continue
        cols = []
        for k in KEYS[:7]:
            v = [x[k] for x in rows if x.get(k) is not None]
            cols.append(f"{statistics.median(v):12.1f}" if v else f"{'-':>12}")
        under = [x["underruns"] for x in rows if x.get("underruns") is not None]
        print(f"{name:<22}{len(rows):>3}" + "".join(cols) + f"   underruns/run {under}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
