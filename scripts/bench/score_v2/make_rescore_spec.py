# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write a ``rescore`` spec (P3B-B) over saved quality-run outputs.

Every ``<run dir>/<run>.output.wav`` next to a ``<run>.ceiling[_<mode>].wav`` and a
``report.json`` becomes a pair, and so does every other offset of a v2 run
(``<run>.output_o<i>.wav`` with ``<run>.ceiling[_<mode>]_o<i>.wav``; ``--first-offset-only`` skips
them); ``--ceiling`` picks which saved ceiling the stream is scored against (``primary`` =
``<run>.ceiling.wav``; ``whole`` / ``b`` / ``long_shape`` = that v2 ceiling). The rescore treats the
offsets of one run as one clip (the median over them, as the quality run does). With
``scoring.analyzer: v2`` and two or more clips, it also measures the target-speaker similarity,
leave one clip out: each pair against the other clips' ceilings (use ``--ceiling whole`` for T5
v2's identity row: the set is the whole-utterance renditions of the other clips).

Usage: ``python scripts/bench/score_v2/make_rescore_spec.py <dir> [<dir> ...] --out spec.json
[--ceiling primary|whole|b|long_shape] [--analyzer v2] [--run fidelity] [--name rescore]
[--first-offset-only]``
(directories are searched recursively). Paths in the spec are absolute.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re


def pair_name(raw, taken):
    """A spec-valid pair name (1-64 of letters, digits, '-', '_', '.'; not starting with '.'), unique
    among ``taken``: other characters become '_', a long name keeps its end (the folder and clip)."""
    name = re.sub(r"[^A-Za-z0-9_.-]", "_", raw).lstrip(".") or "pair"
    if len(name) > 64:
        name = name[-64:].lstrip(".") or "pair"
    base, i = name, 2
    while name in taken:
        suffix = f"-{i}"
        name = base[: 64 - len(suffix)] + suffix
        i += 1
    return name


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ceiling", default="primary", choices=("primary", "whole", "b", "long_shape"))
    ap.add_argument("--analyzer", default="v2", choices=("v1", "v2"))
    ap.add_argument("--run", default="fidelity", help="the quality run's name in the reports")
    ap.add_argument("--name", default="rescore")
    ap.add_argument("--speaker-model", default="", help="scoring.speaker_model (empty: the store's / dev cache's)")
    ap.add_argument("--first-offset-only", action="store_true", help="only <run>.output.wav (not a v2 run's other offsets)")
    args = ap.parse_args()
    suffix = "" if args.ceiling == "primary" else f"_{args.ceiling}"
    pairs = []
    for d in args.dirs:
        for stream in sorted(glob.glob(os.path.join(d, "**", f"{args.run}.output.wav"), recursive=True)):
            folder = os.path.dirname(stream)
            ceiling = os.path.join(folder, f"{args.run}.ceiling{suffix}.wav")
            if not (os.path.isfile(ceiling) and os.path.isfile(os.path.join(folder, "report.json"))):
                continue
            rep = json.load(open(os.path.join(folder, "report.json"), encoding="utf-8"))
            clip = os.path.basename((rep.get("spec", {}).get("input", {}) or {}).get("path", "")) or os.path.basename(folder)
            rel = os.path.relpath(folder, os.path.dirname(os.path.dirname(os.path.abspath(d)))).replace("\\", "/")
            found = [("", stream, ceiling)]
            if not args.first_offset_only:
                i = 1
                while True:
                    s_i = os.path.join(folder, f"{args.run}.output_o{i}.wav")
                    c_i = os.path.join(folder, f"{args.run}.ceiling{suffix}_o{i}.wav")
                    if not (os.path.isfile(s_i) and os.path.isfile(c_i)):
                        break
                    found.append((f"_o{i}", s_i, c_i))
                    i += 1
            for tag, s_path, c_path in found:
                pairs.append({"name": pair_name(f"{rel}_{clip}{tag}", {p["name"] for p in pairs}), "stream": os.path.abspath(s_path).replace("\\", "/"),
                              "ceiling": os.path.abspath(c_path).replace("\\", "/")})
    if not pairs:
        raise SystemExit("no saved pairs found")
    spec = {
        "schema": "sigil-bench-spec/1",
        "name": args.name,
        "description": f"Rescore of {len(pairs)} saved stream / {args.ceiling} ceiling pairs with scorer {args.analyzer} (P3B-B).",
        "engine": {"kind": "stub"},
        "runs": [{
            "mode": "rescore",
            "name": args.name,
            "pairs": pairs,
            "scoring": {"f0_method": "auto", "analysis_ep": "cpu", "analyzer": args.analyzer, "speaker_model": args.speaker_model},
        }],
        "output": {"write_timeline": False, "write_audio": False},
    }
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    print(f"wrote {args.out}: {len(pairs)} pairs")


if __name__ == "__main__":
    main()
