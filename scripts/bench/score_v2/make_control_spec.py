# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the identity control of ADR 0009 as a ``rescore`` spec (P3B-B).

T5 v2's identity row is SIM(stream, target set) - SIM(whole ceiling, target set), where the set is
the whole-utterance renditions of the *other* clips. That delta carries "whatever separates any
streamed rendition from an offline one", so the row needs a control: the same delta for an OFFLINE
rendition standing in for the stream. This script builds, from the saved outputs of a v2 corpus run
(``scripts/bench/specs/p3b/b/run_corpus.ps1``), a folder in which every clip's ``--rendition``
(``primary`` = ceiling A, the long-shape ceiling; ``b`` = ceiling B) takes the stream's place next to
the whole-utterance ceiling, with a ``report.json`` whose alignment is that rendition's own place on
the absolute timeline, and then writes the rescore spec over it (``make_rescore_spec.py --ceiling
whole``). Running the spec reads ``speaker_set_sim`` = SIM(rendition, whole set),
``ceiling_speaker_set_sim`` = SIM(whole, whole set) and their difference, with the same leave-one-clip-out
set, speech frames and exact CAM++ as the real row.

Usage: ``python scripts/bench/score_v2/make_control_spec.py <corpus dir> --tmp <folder to build>
--out spec.json [--rendition primary|b] [--run fidelity] [--name control]``; then
``Sigil.exe --headless bench spec.json --out <dir>`` (CPU scoring of saved WAVs; no GPU).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("--tmp", required=True, help="folder to build (created; existing content is replaced)")
    ap.add_argument("--out", required=True, help="the rescore spec to write")
    ap.add_argument("--rendition", default="primary", choices=("primary", "b"))
    ap.add_argument("--run", default="fidelity")
    ap.add_argument("--name", default="control")
    args = ap.parse_args()
    suffix = "" if args.rendition == "primary" else "_b"
    mode_name = "long_shape" if args.rendition == "primary" else "b"
    if os.path.isdir(args.tmp):
        shutil.rmtree(args.tmp)
    made = 0
    for stream in sorted(glob.glob(os.path.join(args.corpus, "**", f"{args.run}.output.wav"), recursive=True)):
        folder = os.path.dirname(stream)
        report_path = os.path.join(folder, "report.json")
        if not os.path.isfile(report_path):
            continue
        report = json.load(open(report_path, encoding="utf-8"))
        runs = [r for r in report.get("runs", []) if r.get("name") == args.run]
        if not runs:
            continue
        run = runs[0]
        offsets = (run.get("details") or {}).get("offsets") or []
        dest = os.path.join(args.tmp, os.path.basename(folder))
        os.makedirs(dest, exist_ok=True)
        kept = 0
        for i, off in enumerate(offsets):
            tag = "" if i == 0 else f"_o{i}"
            src = os.path.join(folder, f"{args.run}.ceiling{suffix}{tag}.wav")
            whole = os.path.join(folder, f"{args.run}.ceiling_whole{tag}.wav")
            if not (os.path.isfile(src) and os.path.isfile(whole)):
                break
            starts = {c["mode"]: c["start_16k"] for c in off.get("ceilings", [])}
            if mode_name not in starts or "whole" not in starts:
                break
            primary = off["ceilings"][0]["start_16k"]
            shutil.copyfile(src, os.path.join(dest, f"{args.run}.output{tag}.wav"))
            shutil.copyfile(whole, os.path.join(dest, f"{args.run}.ceiling_whole{tag}.wav"))
            # The rendition stands in for the stream: its alignment is its own start against the
            # primary ceiling's (the rescore moves it onto the whole ceiling's indices).
            off["alignment"] = {"lag": float(primary - starts[mode_name]), "method": "timeline"}
            off["calls"] = []
            off["metrics"] = {}
            kept += 1
        if not kept:
            shutil.rmtree(dest, ignore_errors=True)
            continue
        run["details"]["offsets"] = offsets[:kept]
        # A copy of the primary ceiling too, so the folder looks like a corpus run's.
        shutil.copyfile(os.path.join(folder, f"{args.run}.ceiling.wav"), os.path.join(dest, f"{args.run}.ceiling.wav"))
        json.dump(report, open(os.path.join(dest, "report.json"), "w", encoding="utf-8"))
        made += 1
    if not made:
        raise SystemExit("no saved v2 runs with both the rendition and the whole ceiling found")
    here = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(
        [sys.executable, os.path.join(here, "make_rescore_spec.py"), args.tmp, "--out", args.out, "--ceiling", "whole", "--name", args.name, "--run", args.run],
        check=True,
    )
    print(f"{made} clips built under {args.tmp}")


if __name__ == "__main__":
    main()
