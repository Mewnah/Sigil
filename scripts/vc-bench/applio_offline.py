#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline driver for Applio's realtime voice changer (IAHispano/Applio, ``rvc/realtime/core.py``).

Feeds a WAV block by block to ``rvc.realtime.core.VoiceChanger.on_request`` (the object Applio's
realtime tab drives from its audio callback) and writes the blocks it returns, i.e. what the
output device would have played. Sigil's ``compare`` mode scores the result against Sigil's own
output of the same input. The live app's latency is measured separately with Sigil's e2e-loopback
meter.

It runs in YOUR Applio environment (its Python, torch and models); Sigil installs nothing.

Usage:

    python scripts/vc-bench/applio_offline.py --applio-root C:/Applio --pth C:/Applio/logs/voice/voice.pth \
        --index C:/Applio/logs/voice/voice.index --input speech.wav --out applio.wav \
        --f0-method rmvpe --block-ms 250 --crossfade-s 0.05 --extra-s 2.5

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
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--applio-root", required=True)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--index", default="")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sample-rate", type=int, default=48000)
    ap.add_argument("--block-ms", type=float, default=250.0)
    ap.add_argument("--crossfade-s", type=float, default=0.05)
    ap.add_argument("--extra-s", type=float, default=2.5)
    ap.add_argument("--f0-method", default="rmvpe")
    ap.add_argument("--embedder", default="contentvec")
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--index-rate", type=float, default=0.75)
    ap.add_argument("--protect", type=float, default=0.5)
    ap.add_argument("--volume-envelope", type=float, default=1.0)
    ap.add_argument("--sid", type=int, default=0)
    return ap.parse_args()


def call_with(fn, notes, **values):
    """Call `fn` with the keyword arguments it takes (a version without one drops it, noted)."""
    params = inspect.signature(fn).parameters
    takes_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    kwargs = {k: v for k, v in values.items() if k in params or takes_kwargs}
    for k in values:
        if k not in kwargs:
            notes.append(f"{getattr(fn, '__qualname__', fn)} has no '{k}'")
    return fn(**kwargs)


def main():
    args = parse_args()
    root = os.path.abspath(args.applio_root)
    out_path = os.path.abspath(args.out)
    in_path = os.path.abspath(args.input)
    os.chdir(root)  # Applio loads its assets by relative path
    sys.path.insert(0, root)
    import numpy as np
    import soundfile as sf
    import librosa
    from rvc.realtime.core import VoiceChanger

    audio, sr = sf.read(in_path, dtype="float32", always_2d=True)
    audio = librosa.to_mono(audio.T)
    if sr != args.sample_rate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.sample_rate)
    block_frame = int(round(args.block_ms * args.sample_rate / 1000.0))
    notes = []
    vc = call_with(
        VoiceChanger,
        notes,
        block_frame=block_frame,
        cross_fade_overlap_size=args.crossfade_s,
        extra_convert_size=args.extra_s,
        model_path=args.pth,
        index_path=args.index,
        f0_method=args.f0_method,
        embedder_model=args.embedder,
        sid=args.sid,
    )
    n_blocks = (len(audio) + block_frame - 1) // block_frame
    padded = np.zeros(n_blocks * block_frame, dtype="float32")
    padded[: len(audio)] = audio
    out_blocks, times = [], []
    for b in range(n_blocks):
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
        times.append((time.perf_counter() - t0) * 1000.0)
        out = result[0] if isinstance(result, tuple) else result
        if hasattr(out, "detach"):
            out = out.detach().float().cpu().numpy()
        out_blocks.append(np.asarray(out, dtype="float32").reshape(-1))
    out = np.concatenate(out_blocks)
    # The realtime tab plays the blocks at the device rate; a block of another length means the
    # model's rate was returned.
    rate = args.sample_rate
    per_block = len(out) / n_blocks
    if abs(per_block - block_frame) > 1:
        rate = int(round(args.sample_rate * per_block / block_frame))
        notes.append(f"blocks came back with {per_block:.0f} samples for {block_frame}: written at {rate} Hz")
    sf.write(out_path, out, rate, subtype="FLOAT")
    t = np.array(times[2:] or times)
    print(
        f"Applio realtime: {n_blocks} blocks of {block_frame} samples ({args.block_ms:.0f} ms), "
        f"compute p50 {np.percentile(t, 50):.2f} / p95 {np.percentile(t, 95):.2f} / p99 {np.percentile(t, 99):.2f} ms -> {out_path}"
    )
    for n in dict.fromkeys(notes):
        print(f"note: {n}")


if __name__ == "__main__":
    main()
