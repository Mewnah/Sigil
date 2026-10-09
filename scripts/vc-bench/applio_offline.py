#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline driver for Applio's realtime voice changer (IAHispano/Applio, ``rvc/realtime/core.py``).

Feeds a WAV block by block to ``rvc.realtime.core.VoiceChanger.on_request`` (the object Applio's
realtime tab drives from its audio callback) and writes the blocks it returns, i.e. what the
output device would have played (block ``b`` at ``b x block`` in the file). Sigil's ``compare``
mode scores the result; it measures each rendition's latency itself (its offset against the
input, plus one block).

It runs in YOUR Applio environment (its Python, torch and models); Sigil installs nothing.

Every WAV gets a JSON sidecar (``<wav>.json``, ``sigil-vc-bench-sidecar/1``): the operating point
(block, extra, crossfade, F0 method, embedder, index rate, protect, volume envelope, EP, GPU,
transpose), the input's and the voice's sha256, the per-block compute p50 / p95 / p99 (the first two
blocks left out), the checkout's commit, the real-time feasibility (p99 below the block period) and
the latency the realtime tab declares (block + crossfade + 10 ms SOLA search; ``latency_kind``
``declared``, a cross-check of Sigil's measurement).

Usage:

    python scripts/vc-bench/applio_offline.py --applio-root C:/Applio --pth C:/Applio/logs/voice/voice.pth \\
        --index C:/Applio/logs/voice/voice.index --input speech.wav --out applio.wav \\
        --f0-method rmvpe --block-ms 250 --crossfade-s 0.05 --extra-s 2.5

    python scripts/vc-bench/applio_offline.py --applio-root C:/Applio --pth voice.pth \\
        --corpus <corpus>/corpus.json --out-dir <corpus>/out/applio --sweep 40,60,100,250

The constructor and ``on_request`` arguments follow Applio at commit c7665ac9 (``VoiceChanger(
block_frame, cross_fade_overlap_size, extra_convert_size, model_path, index_path, f0_method,
embedder_model, ...)``, ``on_request(audio_input, f0_up_key, index_rate, protect,
volume_envelope, ...) -> (audio, vol, perf)``); arguments another version does not have are
dropped with a note (see ``call_with``). ``block_frame`` is in samples at ``--sample-rate``, the
overlap and extra sizes in seconds, as the realtime tab passes them; check your version's
``tabs/realtime/realtime.py`` if the output length does not match the input.
"""

import argparse
import inspect
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcbench_common as vb  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--applio-root", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--index", default="")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="one input clip")
    src.add_argument("--corpus", help="a sigil-compare-corpus/1 manifest: every clip")
    ap.add_argument("--out", help="output WAV (one clip; with --sweep, '{block}' in it or .b<block> is added)")
    ap.add_argument("--out-dir", help="output folder (--corpus): <out-dir>[/b<block>]/<clip>.wav")
    ap.add_argument("--point-name", default="applio")
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--block-ms", type=float, default=250.0)
    ap.add_argument("--crossfade-s", type=float, default=0.05)
    ap.add_argument("--extra-s", type=float, default=2.5)
    ap.add_argument("--sweep", default="", help="block sizes in ms, e.g. 40,60,100,250 (one operating point each)")
    ap.add_argument("--f0-method", default="rmvpe")
    ap.add_argument("--embedder", default="contentvec")
    ap.add_argument("--pitch", type=int, default=0)
    # Sigil's shipping settings (index 0, protect 0.33, volume envelope 1 = off), not Applio's UI
    # defaults (index 0.75, protect 0.5): a comparison needs the same retrieval and protect settings
    # for every system, and Sigil flags a point whose settings differ from its rows'.
    ap.add_argument("--index-rate", type=float, default=0.0)
    ap.add_argument("--protect", type=float, default=0.33)
    ap.add_argument("--volume-envelope", type=float, default=1.0)
    ap.add_argument("--sid", type=int, default=0)
    ap.add_argument("--app-version", default="")
    args = ap.parse_args(argv)
    if args.corpus and not args.out_dir:
        ap.error("--corpus needs --out-dir")
    if args.input and not args.out:
        ap.error("--input needs --out")
    return args


def call_with(fn, notes, **values):
    """Call `fn` with the keyword arguments it takes (a version without one drops it, noted)."""
    params = inspect.signature(fn).parameters
    takes_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {k: v for k, v in values.items() if k in params or takes_kwargs}
    for k in values:
        if k not in kwargs:
            notes.append(f"{getattr(fn, '__qualname__', fn)} has no '{k}'")
    return fn(**kwargs)


def convert_clip(make_vc, audio, block_frame, flush, args, notes):
    """One clip through a FRESH ``VoiceChanger`` (``make_vc()``), then ``flush`` blocks of silence.

    Applio's object keeps its input context and crossfade buffers between calls: reused over clips
    it would open clip N+1 with the tail of clip N (a rendition that depends on the order the
    clips ran in), while Sigil's engine starts cold per clip. The trailing silence lets the loop
    emit the last of the speech (its delay is the block plus the crossfade); those blocks are not
    timed. Returns ``(output, per-block ms of the clip's own blocks, n_blocks, n_total_blocks)``."""
    import numpy as np

    vc = make_vc()
    try:
        padded, n_blocks = vb.pad_blocks(audio, block_frame, flush)
        out_blocks, times = [], []
        for b in range(n_blocks + flush):
            block = padded[b * block_frame : (b + 1) * block_frame]
            t0 = time.perf_counter()
            result = call_with(
                vc.on_request,
                notes,
                audio_input=block,
                f0_up_key=args.pitch,
                index_rate=args.index_rate,
                protect=args.protect,
                volume_envelope=args.volume_envelope,
            )
            if b < n_blocks:
                times.append((time.perf_counter() - t0) * 1000.0)
            out = result[0] if isinstance(result, tuple) else result
            if hasattr(out, "detach"):
                out = out.detach().float().cpu().numpy()
            out_blocks.append(np.asarray(out, dtype="float32").reshape(-1))
        return np.concatenate(out_blocks), times, n_blocks, n_blocks + flush
    finally:
        del vc


def release_gpu():
    """Give the previous clip's VoiceChanger back to the allocator (best effort)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def out_path(args, clip_id: str, block_ms: float, sweeping: bool) -> str:
    tag = f"b{block_ms:g}"
    if args.corpus:
        d = os.path.join(args.out_dir, tag) if sweeping else args.out_dir
        return os.path.join(d, f"{clip_id}.wav")
    if not sweeping:
        return args.out
    if "{block}" in args.out:
        return args.out.replace("{block}", f"{block_ms:g}")
    stem, ext = os.path.splitext(args.out)
    return f"{stem}.{tag}{ext or '.wav'}"


def main():
    args = parse_args()
    root = os.path.abspath(args.applio_root)
    for attr in ("out", "out_dir", "input", "corpus", "pth", "index"):
        v = getattr(args, attr)
        if v:
            setattr(args, attr, os.path.abspath(v))
    clips = vb.corpus_clips(args.corpus) if args.corpus else [(os.path.splitext(os.path.basename(args.input))[0], args.input)]
    os.chdir(root)  # Applio loads its assets by relative path
    sys.path.insert(0, root)
    import soundfile as sf
    import librosa
    from rvc.realtime.core import VoiceChanger

    commit = vb.git_commit(root)
    gpu = vb.gpu_name()
    blocks = vb.parse_sweep(args.sweep) if args.sweep else [args.block_ms]
    shapes = vb.sweep_shapes(blocks, args.crossfade_s * 1000.0, args.extra_s * 1000.0)
    sweeping = bool(args.sweep)
    snippets = []
    for shp in shapes:
        block_frame = int(round(shp["block_ms"] * args.sample_rate / 1000.0))
        notes = []
        # Applio's realtime tab passes the overlap and extra sizes in seconds. Its output lags the
        # input by the block + the crossfade + the SOLA search: the flush covers that, plus a block.
        flush = vb.flush_blocks(shp["block_ms"] + shp["crossfade_ms"] + 10.0, shp["block_ms"])

        def make_vc(notes=notes, block_frame=block_frame, shp=shp):
            return call_with(
                VoiceChanger,
                notes,
                block_frame=block_frame,
                cross_fade_overlap_size=shp["crossfade_ms"] / 1000.0,
                extra_convert_size=shp["extra_ms"] / 1000.0,
                model_path=args.pth,
                index_path=args.index,
                f0_method=args.f0_method,
                embedder_model=args.embedder,
                sid=args.sid,
            )

        all_times = []
        for clip_id, path in clips:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            audio = librosa.to_mono(audio.T)
            if sr != args.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sample_rate)
            out, times, n_blocks, n_total = convert_clip(make_vc, audio, block_frame, flush, args, notes)
            release_gpu()
            # The realtime tab plays the blocks at the device rate; a block of another length means
            # the model's rate was returned.
            rate = args.sample_rate
            per_block = len(out) / n_total
            if abs(per_block - block_frame) > 1:
                rate = int(round(args.sample_rate * per_block / block_frame))
                notes.append(f"blocks came back with {per_block:.0f} samples for {block_frame}: written at {rate} Hz")
            dest = out_path(args, clip_id, shp["block_ms"], sweeping)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            sf.write(dest, out, rate, subtype="FLOAT")
            steady = times[2:] or times
            call_ms = vb.percentiles(steady)
            cfg = {
                "block_ms": block_frame * 1000.0 / args.sample_rate,
                "extra_ms": shp["extra_ms"],
                "crossfade_ms": shp["crossfade_ms"],
                "f0_method": args.f0_method,
                "embedder": args.embedder,
                "index_rate": args.index_rate,
                "protect": args.protect,
                "rms_mix": args.volume_envelope,
                "ep": "cuda" if gpu else "cpu",
                "gpu": gpu,
                "transpose": args.pitch,
                "samplerate": args.sample_rate,
            }
            side = vb.make_sidecar(
                system="applio",
                app_version=args.app_version or commit[:8],
                commit=commit,
                input_path=path,
                voice_path=args.pth,
                config=cfg,
                call_ms=call_ms,
                latency_ms=cfg["block_ms"] + shp["crossfade_ms"] + 10.0,
                latency_kind="declared",
                latency_note="block + crossfade + 10 ms SOLA search, as the realtime tab declares it (no device latency); Sigil measures it from the rendition",
                gpu=gpu,
                output_rate=rate,
                notes=["offline render of rvc.realtime.core.VoiceChanger (scripts/vc-bench/applio_offline.py)"] + sorted(set(notes)),
                extra={"blocks": n_blocks, "flush_blocks": flush},
            )
            vb.write_sidecar(dest, side)
            all_times += steady
            print(
                f"Applio realtime {clip_id}: {n_blocks} blocks of {block_frame} samples ({cfg['block_ms']:.0f} ms), "
                f"compute p50 {call_ms['p50']:.2f} / p95 {call_ms['p95']:.2f} / p99 {call_ms['p99']:.2f} ms -> {dest}"
            )
        total = vb.percentiles(all_times)
        ok = vb.feasible(total, shp["block_ms"])
        print(f"  block {shp['block_ms']:g} ms: p99 {total['p99']:.2f} ms over every clip -> {'real time' if ok else 'NOT real time (excluded from T6 at this block)'}")
        for n in dict.fromkeys(notes):
            print(f"note: {n}")
        if args.corpus:
            pattern = os.path.relpath(out_path(args, "{clip}", shp["block_ms"], sweeping), os.path.dirname(args.corpus)).replace("\\", "/")
            name = f"{args.point_name}-{shp['block_ms']:g}" if sweeping else args.point_name
            snippets.append(vb.point_snippet(name, "applio", pattern, {"block_ms": shp["block_ms"], "f0_method": args.f0_method}, args.app_version or commit[:8]))
    if snippets:
        print("\npoints for the corpus manifest:")
        print(json.dumps(snippets, indent=1))


if __name__ == "__main__":
    main()
