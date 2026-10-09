#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
"""SOLA join statistics over the corpus clips for crossfade x search-window variants (P3B-H): offline
bench runs of the default voice, one GPU-lock hold for the whole batch.

Per variant it pools the joins of every clip and prints the share of joins moved >= 1 ms from the
nominal point, the mean normalized cross-correlation at the join (and the share below 0.5), and the
mean / share >= 4 ms of the step of the alignment between consecutive joins.

    python joins_batch.py --exe EXE --spec SPEC.json --out DIR --appdata APPDATA
                          [--crossfades 20,10] [--searches auto,10,4,2,0] [--slack aligned]

SPEC.json is a materialized offline spec (e.g. `specs/h-load-cuda.json` of a gate run). The search window
is set with SIGIL_VC_SOLA_SEARCH_MS, the tail rounding with SIGIL_VC_SOLA_SLACK.
"""
import argparse
import copy
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
LOCK = os.path.abspath(os.path.join(HERE, "..", "..", "..", "gpu-lock.ps1"))


def lock(verb, tag, minutes=15):
    cmd = ["pwsh", "-NoProfile", "-File", LOCK]
    cmd += (["-Acquire", tag, "-Wait", "600", "-PollSeconds", "2", "-ExpectedMinutes", str(minutes), "-OwnerPid", str(os.getpid())] if verb == "acquire" else ["-Release", tag])
    return subprocess.run(cmd, capture_output=True, text=True).returncode


def rows_of(h):
    return [(r[0], r[1]) for r in (h or {}).get("rows", [])], (h or {}).get("bin", 1.0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--spec", required=True)
    ap.add_argument("--appdata", default=os.environ.get("SIGIL_APP_DATA_DIR", ""))
    ap.add_argument("--crossfades", default="20,10")
    ap.add_argument("--searches", default="auto,10,4,2,0")
    ap.add_argument("--clips", default="", help="';'-separated wav paths (default: the first 8 of the dev cache's corpus)")
    ap.add_argument("--slack", default="")
    a = ap.parse_args()
    if not a.appdata:
        ap.error("--appdata (or SIGIL_APP_DATA_DIR) is required")
    base = json.load(open(a.spec, encoding="utf-8"))
    cache = os.path.expandvars(r"%LOCALAPPDATA%\sigil-dev-cache")
    clips = [c for c in a.clips.split(";") if c] or (sorted(glob.glob(os.path.join(cache, "sherpa-onnx", "clips", "*.wav"))) + sorted(glob.glob(os.path.join(cache, "models", "transcribe", "samples", "*.wav"))))[:8]
    os.makedirs(a.out, exist_ok=True)
    appdata = os.path.abspath(a.appdata)
    res = {}
    tag = "P3B-H joins"
    if lock("acquire", tag) != 0:
        print("no lock", file=sys.stderr)
        return 1
    try:
        for xf in a.crossfades.split(","):
            for sr in a.searches.split(","):
                cfg = f"xf{xf}-s{sr}" + (f"-{a.slack}" if a.slack else "")
                tot = {"joins": 0, "off_ge1": 0, "off_sum": 0.0, "ncc_sum": 0.0, "ncc_low": 0, "step_sum": 0.0, "step_ge4": 0, "step_n": 0}
                per_clip = []
                for clip in clips:
                    stem = os.path.splitext(os.path.basename(clip))[0]
                    rundir = os.path.join(a.out, cfg, stem)
                    os.makedirs(rundir, exist_ok=True)
                    spec = copy.deepcopy(base)
                    spec["name"] = f"joins-{cfg}-{stem}"
                    spec["input"]["path"] = clip.replace("\\", "/")
                    spec["stream"]["crossfade_ms"] = float(xf)
                    specpath = os.path.join(rundir, "spec.json")
                    json.dump(spec, open(specpath, "w", encoding="utf-8"), indent=1)
                    env = dict(os.environ)
                    env["SIGIL_APP_DATA_DIR"] = appdata
                    if sr != "auto":
                        env["SIGIL_VC_SOLA_SEARCH_MS"] = sr
                    else:
                        env.pop("SIGIL_VC_SOLA_SEARCH_MS", None)
                    if a.slack:
                        env["SIGIL_VC_SOLA_SLACK"] = a.slack
                    rep = os.path.join(rundir, "report.json")
                    if not os.path.exists(rep):
                        with open(os.path.join(rundir, "bench.log"), "w") as log:
                            subprocess.run([a.exe, "--headless", "bench", specpath, "--out", rundir], env=env, stdout=log, stderr=subprocess.STDOUT)
                    try:
                        r = json.load(open(rep, encoding="utf-8"))
                        run = r["runs"][0]
                        j = run["details"]["joins"]
                        m = run["metrics"]
                    except Exception as e:  # noqa: BLE001
                        print(f"{cfg} {stem}: no result ({e})")
                        continue
                    off, ob = rows_of(j.get("offset_ms"))
                    ncc, nb = rows_of(j.get("ncc"))
                    step, sb = rows_of(j.get("step_ms"))
                    n = sum(c for _, c in off)
                    tot["joins"] += n
                    tot["off_ge1"] += sum(c for lo, c in off if lo >= 1.0 - 1e-9)
                    tot["off_sum"] += sum((lo + ob / 2) * c for lo, c in off)
                    tot["ncc_sum"] += sum((lo + nb / 2) * c for lo, c in ncc)
                    tot["ncc_low"] += sum(c for lo, c in ncc if lo < 0.5 - 1e-9)
                    ns = sum(c for _, c in step)
                    tot["step_n"] += ns
                    tot["step_sum"] += sum((lo + sb / 2) * c for lo, c in step)
                    tot["step_ge4"] += sum(c for lo, c in step if lo >= 4.0 - 1e-9)
                    per_clip.append({"clip": stem, "joins": n, "ncc_p50": m.get("join_ncc_p50"), "step_p99": m.get("join_step_ms_p99"), "offset_p50": m.get("join_offset_ms_p50")})
                n = max(tot["joins"], 1)
                res[cfg] = {
                    "joins": tot["joins"],
                    "share_delta_ge_1ms": tot["off_ge1"] / n,
                    "mean_delta_ms": tot["off_sum"] / n,
                    "mean_ncc": tot["ncc_sum"] / n,
                    "share_ncc_below_0.5": tot["ncc_low"] / n,
                    "mean_step_ms": tot["step_sum"] / max(tot["step_n"], 1),
                    "share_step_ge_4ms": tot["step_ge4"] / max(tot["step_n"], 1),
                    "clips": per_clip,
                }
                print(f"[{time.strftime('%H:%M:%S')}] {cfg}: {tot['joins']} joins, delta>=1ms {res[cfg]['share_delta_ge_1ms']:.1%}, mean delta {res[cfg]['mean_delta_ms']:.2f} ms, mean NCC {res[cfg]['mean_ncc']:.3f}, NCC<0.5 {res[cfg]['share_ncc_below_0.5']:.1%}, mean step {res[cfg]['mean_step_ms']:.2f} ms, step>=4ms {res[cfg]['share_step_ge_4ms']:.1%}", flush=True)
    finally:
        lock("release", tag)
    json.dump(res, open(os.path.join(a.out, "joins.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
