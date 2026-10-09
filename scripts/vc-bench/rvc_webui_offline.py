#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Offline driver for RVC WebUI's realtime path (RVC-Project/Retrieval-based-Voice-Conversion-WebUI).

Runs a WAV through the exact block loop of RVC WebUI's ``realtime_gui.py`` (``audio_callback``):
input ring buffers, the input resampler to 16 kHz, ``rtrvc.RVC.infer`` with ``skip_head`` /
``return_length``, the output resampler, the optional volume-envelope mix, and the SOLA crossfade,
then writes what the output device would have played (output block ``b`` at ``b x block`` in the
file). Sigil's ``compare`` mode scores the result (``src-tauri/src/services/bench/compare.rs``).

It runs in YOUR RVC WebUI environment (its Python, torch and models); Sigil installs nothing.

Every WAV gets a JSON sidecar (``<wav>.json``, ``sigil-vc-bench-sidecar/1``): the operating point
(block, extra, crossfade and the SOLA buffer the GUI really uses, F0 method, index rate, rms mix,
precision, EP, GPU, transpose), the input's and the voice's sha256, the per-block compute p50 / p95
/ p99 (``infer`` + resampling + SOLA, the first two blocks left out), the RVC checkout's commit, the
real-time feasibility (p99 below the block period) and the algorithmic latency of the loop:
block + SOLA buffer + SOLA search - the mean SOLA offset (what the GUI's "delay" shows, minus the
device's own latency). Sigil measures the latency again from the renditions themselves and flags a
disagreement.

Usage (from anywhere):

    # one clip, the GUI's defaults (250 ms block, 50 ms crossfade, 2.5 s extra)
    python scripts/vc-bench/rvc_webui_offline.py --rvc-root C:/RVC --pth C:/RVC/assets/weights/voice.pth \\
        --input speech.wav --out rvc-webui.wav --f0method rmvpe

    # every clip of a compare corpus, at four block sizes (one operating point each)
    python scripts/vc-bench/rvc_webui_offline.py --rvc-root C:/RVC --pth voice.pth \\
        --corpus <corpus>/corpus.json --out-dir <corpus>/out/rvc_webui --sweep 40,60,100,250

The sweep keeps the extra context (``--extra-time``) and caps the crossfade at half the block; it
prints the ``points`` entries to paste into the corpus manifest (one per block, ``wav`` with
``{clip}``). Points whose p99 compute reaches the block period are marked infeasible in their
sidecars and do not count in T6 (they cannot run live at that block on this GPU).

The block loop follows realtime_gui.py at commit 81eed5e8 (2026-08); older checkouts keep
``tools/rvc_for_realtime.py`` with other constructor arguments, which this driver maps by name (see
``make_rvc``). Noise reduction and the phase vocoder are off (the GUI's defaults).
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
    ap.add_argument("--rvc-root", required=True, help="RVC WebUI checkout (its assets/ must hold hubert and rmvpe)")
    ap.add_argument("--pth", required=True)
    ap.add_argument("--index", default="")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", help="one input clip")
    src.add_argument("--corpus", help="a sigil-compare-corpus/1 manifest: every clip")
    ap.add_argument("--out", help="output WAV (one clip; with --sweep, '{block}' in it or .b<block> is added)")
    ap.add_argument("--out-dir", help="output folder (--corpus): <out-dir>[/b<block>]/<clip>.wav")
    ap.add_argument("--point-name", default="rvc_webui", help="name of the point in the printed manifest entries")
    ap.add_argument("--pitch", type=int, default=0)
    ap.add_argument("--formant", type=float, default=0.0)
    ap.add_argument("--index-rate", type=float, default=0.0, help="retrieval strength (0 = off, Sigil's shipping default); use Sigil's value for the comparison")
    ap.add_argument("--f0method", default="rmvpe", choices=("pm", "harvest", "crepe", "rmvpe", "fcpe"))
    ap.add_argument("--samplerate", type=int, default=48000)
    ap.add_argument("--block-time", type=float, default=0.25)
    ap.add_argument("--crossfade-time", type=float, default=0.05)
    ap.add_argument("--extra-time", type=float, default=2.5)
    ap.add_argument("--sweep", default="", help="block sizes in ms, e.g. 40,60,100,250 (one operating point each)")
    ap.add_argument("--rms-mix-rate", type=float, default=1.0, help="the GUI's 'Loudness factor' (1 = off, Sigil's default; the GUI itself starts at 0)")
    ap.add_argument("--threshold", type=float, default=-60.0, help="the GUI's response threshold (dB); -60 = off")
    ap.add_argument("--app-version", default="", help="RVC WebUI version label (default: the checkout's commit)")
    args = ap.parse_args(argv)
    if args.corpus and not args.out_dir:
        ap.error("--corpus needs --out-dir")
    if args.input and not args.out:
        ap.error("--input needs --out")
    return args


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


def make_rvc(rtrvc, args, config, last=None):
    """The RVC object, mapping arguments by name (the constructor changed across versions).

    The driver builds a NEW object for every clip, passing the previous one as ``last`` (RVC's
    ``last_rvc``: the loaded models are reused, the pitch / pitchf cache and the retrieval state
    start empty). One object reused over clips would open clip N+1 with the tail of clip N in its
    F0 cache, so a rendition would depend on the order the clips ran in; Sigil's engine starts
    cold per clip. A version without ``last_rvc`` reloads the models for each clip (slower, the
    same result)."""
    params = inspect.signature(rtrvc.RVC.__init__).parameters
    values = {
        "key": args.pitch,
        "formant": args.formant,
        "pth_path": args.pth,
        "index_path": args.index,
        "index_rate": args.index_rate,
        "config": config,
        "last_rvc": last,
        "n_cpu": os.cpu_count() or 4,
        "inp_q": None,
        "opt_q": None,
    }
    kwargs = {k: values[k] for k in params if k != "self" and k in values}
    missing = [k for k, p in params.items() if k != "self" and k not in kwargs and p.default is inspect.Parameter.empty]
    if missing:
        raise SystemExit(f"this RVC WebUI version's RVC() needs {missing}; update the driver's make_rvc")
    return rtrvc.RVC(**kwargs)


def render(rvc, device, audio, args, samplerate, block_time, crossfade_time, extra_time):
    """The GUI's block loop over ``audio`` (mono at ``samplerate``): ``(output, per-block ms, SOLA
    offsets in samples, shape)``."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torchaudio.transforms as tat
    import librosa

    # start_vc (realtime_gui.py)
    zc = samplerate // 100
    block_frame = int(np.round(block_time * samplerate / zc)) * zc
    block_frame_16k = 160 * block_frame // zc
    crossfade_frame = int(np.round(crossfade_time * samplerate / zc)) * zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    sola_search_frame = zc
    extra_frame = int(np.round(extra_time * samplerate / zc)) * zc
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

    out_blocks, times, offsets = [], [], []
    # After the clip the loop keeps running on silence until its delay (block + SOLA buffer + SOLA
    # search, the most it can be) has emitted the last of the speech; those blocks are neither
    # timed nor counted in the SOLA offsets.
    flush = vb.flush_blocks((block_frame + sola_buffer_frame + sola_search_frame) * 1000.0 / samplerate, block_time * 1000.0)
    padded, n_blocks = vb.pad_blocks(audio, block_frame, flush)
    for b in range(n_blocks + flush):
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
        if b < n_blocks:
            times.append((time.perf_counter() - t0) * 1000.0)
            offsets.append(sola_offset)
        out_blocks.append(out)
    shape = {
        "flush_blocks": flush,
        "block_frame": block_frame,
        "crossfade_frame": crossfade_frame,
        "sola_buffer_frame": sola_buffer_frame,
        "sola_search_frame": sola_search_frame,
        "extra_frame": extra_frame,
    }
    return np.concatenate(out_blocks), times, offsets, shape


def algorithmic_latency_ms(shape: dict, offsets, samplerate: int) -> float:
    """Block + SOLA buffer + SOLA search - the mean SOLA offset (ms): an output sample of block b
    converts the input ``sola_buffer + search + block - offset`` samples before it plays."""
    mean_off = sum(offsets) / len(offsets) if offsets else 0.0
    frames = shape["block_frame"] + shape["sola_buffer_frame"] + shape["sola_search_frame"] - mean_off
    return frames * 1000.0 / samplerate


def main():
    args = parse_args()
    root = os.path.abspath(args.rvc_root)
    for attr in ("out", "out_dir", "input", "corpus", "pth", "index"):
        v = getattr(args, attr)
        if v:
            setattr(args, attr, os.path.abspath(v))
    clips = vb.corpus_clips(args.corpus) if args.corpus else [(os.path.splitext(os.path.basename(args.input))[0], args.input)]
    os.chdir(root)  # RVC loads its assets by relative path
    sys.path.insert(0, root)
    import numpy as np  # noqa: F401
    import soundfile as sf
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
    rvc = None  # built per clip (make_rvc)
    device = config.device
    commit = vb.git_commit(root)
    gpu = vb.gpu_name() if str(device).startswith("cuda") else ""
    ep = "cuda" if str(device).startswith("cuda") else ("dml" if "privateuseone" in str(device) or "dml" in str(device) else "cpu")
    blocks = vb.parse_sweep(args.sweep) if args.sweep else [args.block_time * 1000.0]
    shapes = vb.sweep_shapes(blocks, args.crossfade_time * 1000.0, args.extra_time * 1000.0)
    sweeping = bool(args.sweep)
    snippets = []
    for shp in shapes:
        all_times, lat = [], []
        for clip_id, path in clips:
            audio, sr = sf.read(path, dtype="float32", always_2d=True)
            audio = librosa.to_mono(audio.T)
            if sr != args.samplerate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=args.samplerate)
            rvc = make_rvc(rtrvc, args, config, last=rvc)  # fresh state per clip, models reused
            out, times, offsets, shape = render(
                rvc, device, audio, args, args.samplerate, shp["block_ms"] / 1000.0, shp["crossfade_ms"] / 1000.0, shp["extra_ms"] / 1000.0
            )
            dest = out_path(args, clip_id, shp["block_ms"], sweeping)
            os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
            sf.write(dest, out, args.samplerate, subtype="FLOAT")
            steady = times[2:] or times
            call_ms = vb.percentiles(steady)
            latency = algorithmic_latency_ms(shape, offsets, args.samplerate)
            cfg = {
                "block_ms": shape["block_frame"] * 1000.0 / args.samplerate,
                "extra_ms": shape["extra_frame"] * 1000.0 / args.samplerate,
                "crossfade_ms": shape["crossfade_frame"] * 1000.0 / args.samplerate,
                "sola_buffer_ms": shape["sola_buffer_frame"] * 1000.0 / args.samplerate,
                "sola_search_ms": shape["sola_search_frame"] * 1000.0 / args.samplerate,
                "f0_method": args.f0method,
                "index_rate": args.index_rate,
                "protect": None,
                "rms_mix": args.rms_mix_rate,
                "precision": "fp16" if getattr(config, "is_half", False) else "fp32",
                "ep": ep,
                "gpu": gpu,
                "transpose": args.pitch,
                "formant": args.formant,
                "threshold_db": args.threshold,
                "samplerate": args.samplerate,
            }
            side = vb.make_sidecar(
                system="rvc_webui",
                app_version=args.app_version or commit[:8],
                commit=commit,
                input_path=path,
                voice_path=args.pth,
                config=cfg,
                call_ms=call_ms,
                latency_ms=latency,
                latency_kind="algorithmic",
                latency_note="block + SOLA buffer + SOLA search - mean SOLA offset, from the driver's block loop (no device latency)",
                gpu=gpu,
                output_rate=args.samplerate,
                notes=["offline render of realtime_gui.py's block loop (scripts/vc-bench/rvc_webui_offline.py)"],
                extra={"blocks": len(times), "flush_blocks": shape["flush_blocks"], "mean_sola_offset_frames": sum(offsets) / max(len(offsets), 1)},
            )
            vb.write_sidecar(dest, side)
            all_times += steady
            lat.append(latency)
            print(
                f"RVC WebUI realtime {clip_id}: {len(times)} blocks of {shape['block_frame']} frames ({cfg['block_ms']:.0f} ms), "
                f"compute p50 {call_ms['p50']:.2f} / p95 {call_ms['p95']:.2f} / p99 {call_ms['p99']:.2f} ms, latency {latency:.1f} ms -> {dest}"
            )
        total = vb.percentiles(all_times)
        ok = vb.feasible(total, shp["block_ms"])
        print(f"  block {shp['block_ms']:g} ms: p99 {total['p99']:.2f} ms over every clip -> {'real time' if ok else 'NOT real time (excluded from T6 at this block)'}")
        if args.corpus:
            pattern = os.path.relpath(out_path(args, "{clip}", shp["block_ms"], sweeping), os.path.dirname(args.corpus)).replace("\\", "/")
            name = f"{args.point_name}-{shp['block_ms']:g}" if sweeping else args.point_name
            snippets.append(vb.point_snippet(name, "rvc_webui", pattern, {"block_ms": shp["block_ms"], "f0_method": args.f0method}, args.app_version or commit[:8]))
    if snippets:
        print("\npoints for the corpus manifest:")
        print(json.dumps(snippets, indent=1))


if __name__ == "__main__":
    main()
