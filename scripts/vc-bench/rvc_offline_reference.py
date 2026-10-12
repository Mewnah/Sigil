#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Neutral references for Sigil's compare mode (plan T6): upstream RVC WebUI's own offline
conversion of each corpus clip, ``VC.vc_single`` (``infer/modules/vc/modules.py``): PyTorch fp32,
RMVPE, the whole utterance in one pass, with the corpus's transpose and the same index rate and
protect every system uses. Every engine, Sigil's included, is then scored by its envelope MCD to
these renders (``compare.rs``: ``mcd_ref_db``), which no engine's own tracker or noise favours.

RVC's generator draws random numbers (the flow prior's noise, the NSF source's initial phase and
noise), so each clip is rendered with at least two seeds (``torch.manual_seed``): the distance
between two seeds is the stochastic floor that compare reports next to every MCD
(``reference_floor_db``).

It runs in YOUR RVC WebUI environment (its Python, torch, ContentVec/HuBERT and
``assets/rmvpe/rmvpe.pt``); Sigil installs nothing. This is a manual step: the dev PC has no RVC
WebUI install.

    python scripts/vc-bench/rvc_offline_reference.py --rvc-root C:/RVC --pth C:/RVC/assets/weights/voice.pth \\
        --corpus <corpus>/corpus.json --out-dir <corpus>/refs --seeds 1,2 --index-rate 0 --protect 0.33

It writes ``<out-dir>/<clip>.seed<k>.wav`` with a sidecar each, and
``<corpus>.with-refs.json`` (a copy of the manifest with every clip's ``references`` filled in;
the original is never changed). Check that the renders are as long as their inputs.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcbench_common as vb  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rvc-root", required=True, help="RVC WebUI checkout (assets/hubert and assets/rmvpe)")
    ap.add_argument("--pth", required=True, help="the voice (the same file every system loaded)")
    ap.add_argument("--index", default="")
    ap.add_argument("--corpus", required=True, help="a sigil-compare-corpus/1 manifest")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seeds", default="1,2", help="noise seeds (at least two: the stochastic floor)")
    ap.add_argument("--transpose", type=int, default=None, help="default: the corpus's transpose (0)")
    ap.add_argument("--index-rate", type=float, default=0.0)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--filter-radius", type=int, default=3)
    ap.add_argument("--rms-mix-rate", type=float, default=1.0, help="1 = the model's own envelope (off)")
    ap.add_argument("--sid", type=int, default=0, help="speaker id of a multi-speaker voice")
    args = ap.parse_args(argv)
    seeds = parse_seeds(args.seeds)
    if len(seeds) < 2:
        ap.error("--seeds needs at least two seeds (the seed-to-seed distance is the stochastic floor)")
    args.seed_list = seeds
    return args


def parse_seeds(text: str) -> list:
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if part:
            v = int(part)
            if v not in out:
                out.append(v)
    return out


def reference_name(clip_id: str, seed: int) -> str:
    return f"{clip_id}.seed{seed}.wav"


def single_kwargs(vc_single, args, path: str, transpose: int, index: str) -> dict:
    """``vc_single``'s arguments by name: the signature changed across RVC WebUI versions (the
    2026-07 build has no ``f0_file``, ``file_index2`` or ``filter_radius``). A parameter this
    driver has no value for stops the run instead of being guessed."""
    import inspect

    values = {
        "sid": args.sid,
        "input_audio_path": path,
        "f0_up_key": transpose,
        "f0_file": None,
        "f0_method": "rmvpe",
        "file_index": index,
        "file_index2": index,
        "index_rate": args.index_rate,
        "filter_radius": args.filter_radius,
        "resample_sr": 0,
        "rms_mix_rate": args.rms_mix_rate,
        "protect": args.protect,
    }
    names = [n for n, p in inspect.signature(vc_single).parameters.items() if n != "self"]
    unknown = [n for n in names if n not in values]
    if unknown:
        raise SystemExit(f"this RVC WebUI version's vc_single takes {unknown}, which this driver has no value for; update single_kwargs")
    return {n: values[n] for n in names}


def with_references(manifest: dict, refs: dict, manifest_dir: str, out_dir: str) -> dict:
    """A copy of ``manifest`` whose clips list their references (paths relative to the manifest)."""
    m = json.loads(json.dumps(manifest))
    for c in m.get("clips", []):
        names = refs.get(c["id"], [])
        if names:
            c["references"] = [os.path.relpath(os.path.join(out_dir, n), manifest_dir).replace("\\", "/") for n in names]
    return m


def main():
    args = parse_args()
    root = os.path.abspath(args.rvc_root)
    corpus = os.path.abspath(args.corpus)
    out_dir = os.path.abspath(args.out_dir)
    pth = os.path.abspath(args.pth)
    index = os.path.abspath(args.index) if args.index else ""
    with open(corpus, encoding="utf-8") as f:
        manifest = json.load(f)
    transpose = args.transpose if args.transpose is not None else int(round(manifest.get("transpose") or 0))
    clips = vb.corpus_clips(corpus)
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(root)
    sys.path.insert(0, root)
    os.environ["weight_root"] = os.path.dirname(pth)
    os.environ.setdefault("index_root", os.path.dirname(index) if index else os.path.dirname(pth))
    os.environ.setdefault("rmvpe_root", os.path.join(root, "assets", "rmvpe"))
    import numpy as np
    import soundfile as sf
    import torch

    argv, sys.argv = sys.argv, sys.argv[:1]  # RVC's Config parses the command line
    try:
        from configs.config import Config

        config = Config()
    finally:
        sys.argv = argv
    config.is_half = False  # fp32: the neutral reference
    try:
        from infer.modules.vc.modules import VC
    except ImportError:
        from infer.vc.modules import VC  # the 2026-07 layout (RVC20260718Nvidia)

    vc = VC(config)
    vc.get_vc(os.path.basename(pth), args.protect, args.protect)
    commit = vb.git_commit(root)
    refs = {}
    for clip_id, path in clips:
        for seed in args.seed_list:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            kw = single_kwargs(vc.vc_single, args, path, transpose, index)
            info, result = vc.vc_single(**kw)
            if not isinstance(result, tuple) or result[1] is None:
                raise SystemExit(f"{clip_id} seed {seed}: vc_single failed: {info}")
            sr, audio = result
            audio = np.asarray(audio)
            if audio.dtype.kind in "iu":
                audio = audio.astype("float32") / 32768.0
            name = reference_name(clip_id, seed)
            dest = os.path.join(out_dir, name)
            sf.write(dest, audio, int(sr), subtype="FLOAT")
            in_len = sf.info(path).duration
            out_len = len(audio) / float(sr)
            cfg = {
                "f0_method": "rmvpe",
                "index_rate": args.index_rate,
                "protect": args.protect,
                "rms_mix": args.rms_mix_rate,
                "filter_radius": kw.get("filter_radius"),  # None: this version has no median filter argument
                "precision": "fp32",
                "whole_utterance": True,
                "seed": seed,
                "transpose": transpose,
                "sid": args.sid,
            }
            side = vb.make_sidecar(
                system="rvc_webui_vc_single",
                app_version=commit[:8],
                commit=commit,
                input_path=path,
                voice_path=pth,
                config=cfg,
                gpu=vb.gpu_name(),
                output_rate=int(sr),
                notes=["neutral reference: upstream vc_single, fp32, RMVPE, whole utterance (scripts/vc-bench/rvc_offline_reference.py)"],
                extra={"input_seconds": in_len, "output_seconds": out_len},
            )
            vb.write_sidecar(dest, side)
            refs.setdefault(clip_id, []).append(name)
            warn = "" if abs(out_len - in_len) < 0.05 else f"  (length {out_len:.2f} s vs input {in_len:.2f} s: check)"
            print(f"{clip_id} seed {seed}: {out_len:.2f} s at {sr} Hz -> {dest}{warn}")
    updated = with_references(manifest, refs, os.path.dirname(corpus), out_dir)
    stem, ext = os.path.splitext(corpus)
    out_manifest = f"{stem}.with-refs{ext or '.json'}"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=1)
    print(f"manifest with references -> {out_manifest}")


if __name__ == "__main__":
    main()
