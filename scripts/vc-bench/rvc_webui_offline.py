#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline driver for RVC WebUI's realtime path (RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

Runs a WAV through the exact block loop of RVC WebUI's ``realtime_gui.py`` (``audio_callback``):
input ring buffers, the input resampler to 16 kHz, ``rtrvc.RVC.infer`` with ``skip_head`` /
``return_length``, the output resampler, the optional volume-envelope mix, and the SOLA crossfade,
then writes what the output device would have played. Sigil's ``compare`` mode scores the result
against Sigil's own output of the same input (``scripts/bench/specs`` / ``docs/benchmarks``).

It runs in YOUR RVC WebUI environment (its Python, torch and models); Sigil installs nothing. The
latency of the live app is measured separately with Sigil's e2e-loopback meter.

Usage (from anywhere):

    python scripts/vc-bench/rvc_webui_offline.py --rvc-root C:/RVC --pth C:/RVC/assets/weights/voice.pth \
        --index C:/RVC/logs/voice/added_IVF.index --input speech.wav --out rvc-webui.wav \
        --pitch 0 --f0method rmvpe --block-time 0.25 --crossfade-time 0.05 --extra-time 2.5

The defaults are the GUI's defaults. The block loop follows realtime_gui.py at commit 81eed5e8
(2026-08); older checkouts keep ``tools/rvc_for_realtime.py`` with other constructor arguments,
which this driver maps by name (see ``make_rvc``). Noise reduction and the phase vocoder are off
(the GUI's defaults). Per-block compute (``infer`` + resampling + SOLA) is printed as p50 / p95 /
p99 so it can be put next to Sigil's ``block_ms``.
"""

import argparse
import inspect
import os
import sys
import time


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rvc-root", required=True, help="RVC WebUI checkout (its assets/ must hold hubert and rmvpe)")
    ap.add_argument("--pth", required=True)
    ap.add_argument("--index", default="")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--formant", type=float, default=0.0)
    ap.add_argument("--index-rate", type=float, default=0.0)
    ap.add_argument("--f0method", default="rmvpe", choices=("pm", "harvest", "crepe", "rmvpe", "fcpe"))
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--block-time", type=float, default=0.25)
    ap.add_argument("--crossfade-time", type=float, default=0.05)
    ap.add_argument("--extra-time", type=float, default=2.5)
    ap.add_argument("--rms-mix-rate", type=float, default=0.0, help="the GUI's 'Loudness factor' (1 = off)")
    ap.add_argument("--threshold", type=float, default=-60.0, help="the GUI's response threshold (dB); -60 = off")
    return ap.parse_args()


def make_rvc(rtrvc, args, config):
    """The RVC object, mapping arguments by name (the constructor changed across versions)."""
    params = inspect.signature(rtrvc.RVC.__init__).parameters
    values = {
        "key": args.pitch,
        "formant": args.formant,
        "pth_path": args.pth,
        "index_path": args.index,
        "index_rate": args.index_rate,
        "config": config,
        "last_rvc": None,
        "n_cpu": os.cpu_count() or 4,
        "inp_q": None,
        "opt_q": None,
    }
    kwargs = {k: values[k] for k in params if k != "self" and k in values}
    missing = [k for k, p in params.items() if k != "self" and k not in kwargs and p.default is inspect.Parameter.empty]
    if missing:
        raise SystemExit(f"this RVC WebUI version's RVC() needs {missing}; update the driver's make_rvc")
    return rtrvc.RVC(**kwargs)


def main():
    args = parse_args()
    root = os.path.abspath(args.rvc_root)
    out_path = os.path.abspath(args.out)
    in_path = os.path.abspath(args.input)
    os.chdir(root)  # RVC loads its assets by relative path
    sys.path.insert(0, root)
    import numpy as np
    import soundfile as sf
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    import librosa

    argv, sys.argv = sys.argv, sys.argv[:1]  # RVC's Config parses the command line
    try:
        from configs.config import Config

        config = Config()
    finally:
        sys.argv = argv
    try:
        from infer import rtrvc
    except ImportError:
        from tools import rvc_for_realtime as rtrvc  # older checkouts
    rvc = make_rvc(rtrvc, args, config)
    device = config.device

    audio, sr = sf.read(in_path, dtype="float32", always_2d=True)
    audio = librosa.to_mono(audio.T)
    if sr != args.samplerate:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=args.samplerate)
    samplerate = args.samplerate

    # start_vc (realtime_gui.py)
    zc = samplerate // 100
    block_frame = int(np.round(args.block_time * samplerate / zc)) * zc
    block_frame_16k = 160 * block_frame // zc
    crossfade_frame = int(np.round(args.crossfade_time * samplerate / zc)) * zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    sola_search_frame = zc
    extra_frame = int(np.round(args.extra_time * samplerate / zc)) * zc
    input_wav = torch.zeros(extra_frame + crossfade_frame + sola_search_frame + block_frame, device=device, dtype=torch.float32)
    input_wav_res = torch.zeros(160 * input_wav.shape[0] // zc, device=device, dtype=torch.float32)
    rms_buffer = np.zeros(4 * zc, dtype="float32")
    sola_buffer = torch.zeros(sola_buffer_frame, device=device, dtype=torch.float32)
    skip_head = extra_frame // zc
    return_length = (block_frame + sola_buffer_frame + sola_search_frame) // zc
    fade_in_window = torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=sola_buffer_frame, device=device, dtype=torch.float32)) ** 2
    fade_out_window = 1 - fade_in_window
    resampler = tat.Resample(orig_freq=samplerate, new_freq=16000, dtype=torch.float32).to(device)
    resampler2 = tat.Resample(orig_freq=rvc.tgt_sr, new_freq=samplerate, dtype=torch.float32).to(device) if rvc.tgt_sr != samplerate else None
    sola_den_kernel = torch.ones(1, 1, sola_buffer_frame, device=device)

    out_blocks = []
    times = []
    n_blocks = (len(audio) + block_frame - 1) // block_frame
    padded = np.zeros(n_blocks * block_frame, dtype="float32")
    padded[: len(audio)] = audio
    for b in range(n_blocks):
        indata = padded[b * block_frame : (b + 1) * block_frame].copy()
        t0 = time.perf_counter()
        # audio_callback (realtime_gui.py), without noise reduction and phase vocoder
        if args.threshold > -60:
            indata = np.append(rms_buffer, indata)
            rms = librosa.feature.rms(y=indata, frame_length=4 * zc, hop_length=zc)[:, 2:]
            rms_buffer[:] = indata[-4 * zc :]
            indata = indata[2 * zc - zc // 2 :]
            db = librosa.amplitude_to_db(rms, ref=1.0)[0] < args.threshold
            for i in range(db.shape[0]):
                if db[i]:
                    indata[i * zc : (i + 1) * zc] = 0
            indata = indata[zc // 2 :]
        input_wav[:-block_frame] = input_wav[block_frame:].clone()
        input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(device)
        input_wav_res[:-block_frame_16k] = input_wav_res[block_frame_16k:].clone()
        input_wav_res[-160 * (indata.shape[0] // zc + 1) :] = resampler(input_wav[-indata.shape[0] - 2 * zc :])[160:]
        infer_wav = rvc.infer(input_wav_res, block_frame_16k, skip_head, return_length, args.f0method)
        if resampler2 is not None:
            infer_wav = resampler2(infer_wav)
        if args.rms_mix_rate < 1:
            iw = input_wav[extra_frame:]
            rms1 = librosa.feature.rms(y=iw[: infer_wav.shape[0]].cpu().numpy(), frame_length=4 * zc, hop_length=zc)
            rms1 = torch.from_numpy(rms1).to(device)
            rms1 = F.interpolate(rms1.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = librosa.feature.rms(y=infer_wav[:].cpu().numpy(), frame_length=4 * zc, hop_length=zc)
            rms2 = torch.from_numpy(rms2).to(device)
            rms2 = F.interpolate(rms2.unsqueeze(0), size=infer_wav.shape[0] + 1, mode="linear", align_corners=True)[0, 0, :-1]
            rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)
            infer_wav *= torch.pow(rms1 / rms2, 1.0 - args.rms_mix_rate)
        conv_input = infer_wav[None, None, : sola_buffer_frame + sola_search_frame]
        cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
        cor_den = torch.sqrt(F.conv1d(conv_input**2, sola_den_kernel) + 1e-8)
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        infer_wav = infer_wav[sola_offset:]
        infer_wav[:sola_buffer_frame] *= fade_in_window
        infer_wav[:sola_buffer_frame] += sola_buffer * fade_out_window
        sola_buffer[:] = infer_wav[block_frame : block_frame + sola_buffer_frame]
        out = infer_wav[:block_frame].cpu().numpy()
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
        out_blocks.append(out)
    sf.write(out_path, np.concatenate(out_blocks), samplerate, subtype="FLOAT")
    t = np.array(times[2:] or times)
    print(
        f"RVC WebUI realtime: {n_blocks} blocks of {block_frame} frames ({args.block_time * 1000:.0f} ms), "
        f"compute p50 {np.percentile(t, 50):.2f} / p95 {np.percentile(t, 95):.2f} / p99 {np.percentile(t, 99):.2f} ms -> {out_path}"
    )


if __name__ == "__main__":
    main()
