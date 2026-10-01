#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DDSP-SVC 6.3: offline parity of the exported graphs against upstream PyTorch, and their per-call
cost (P3-H, ADR 0006).

Reference: upstream ``reflow/vocoder.py`` ``Unit2Wav.forward(..., vocoder=, infer=True,
return_wav=True, method="euler", t_start=, silence_front=)`` exactly as ``gui_reflow.py``
``SvcDDSP.infer`` calls it for one real-time tick, in PyTorch fp32 on the CPU, with every random
draw recorded (``common.recorded_random``: the DDSP noise and the flow's start noise) and the DDSP
mel and final mel observed (instance wrappers of ``Vocoder.extract`` / ``Vocoder.infer``). The DDSP
and flow networks carry the seeded weights of ``export.py`` (no public 6.3 checkpoint); the vocoder
is the real openvpi PC-NSF-HiFiGAN. Candidate: ``sampler.DdspSvc`` on the ONNX graphs, fed the same
inputs and draws. Gate: SNR >= 35 dB on the output audio and on every stage.

Inputs of the tick: ``jfk.wav`` resampled to 44.1 kHz (the window), its ContentVec units in
upstream's "tta2x" form, upstream's volume envelope, and a synthetic singing F0 over the decoded
part: a log sweep E2 (82.4 Hz) -> D#7 (2489 Hz) with 5.5 Hz vibrato, which is the T10 singing range.
A second row drives the vocoder alone over 4 s of real mel with the same sweep. The units row checks
Sigil's shared embedder (``vc/embedder-contentvec`` ONNX, run twice and interleaved) against
upstream's ``Audio2ContentVec768L12TTA2X`` on the same 16 kHz audio.

The upstream-exported pair (``encoder.onnx`` / ``velocity.onnx`` from upstream's own exporter) is
measured too, for the record: its time input is int64 ``1000 t`` and its sinc divides 0 by 0.

The chain rows compare the model call (``Unit2Wav.forward``, ``threshold_db=None``);
``chain_audio_volume_gate`` adds ``SvcDDSP.infer``'s volume gate on both sides (upstream's own
``Volume_Extractor`` and ``ddsp.core.upsample`` against ``sampler.volume_mask``), at a threshold that
mutes part of the decoded frames.

Cost: per call at the shapes of each ``sampler.PRESETS`` entry, and per tick = 2 x ContentVec (tta2x)
+ ddsp + steps x reflow_step + vocoder, on CPU (2 threads) and CUDA (host / IOBinding / CUDA graph).
``--profile`` adds ORT's own profile of every graph of the tick at each preset (wall and kernel time
per run, the split by provider and op, nodes per provider, CPU-placed ops, and whether the session's
dimensions stayed pinned); the DirectML report carries it. Every section records the sha256 of the
graphs it measured (``graphs``).

Usage (repo root):
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-cuda/Scripts/python scripts/vc-families/ddsp_svc/parity.py   # CPU EP of ORT 1.28
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-cuda/Scripts/python scripts/vc-families/ddsp_svc/parity.py --ep cuda
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-dml/Scripts/python scripts/vc-families/ddsp_svc/parity.py --ep dml --profile
Writes ``reports/ddsp_svc_<ep>.json``.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import common as cm  # noqa: E402
import export as ex  # noqa: E402
import sampler as sp  # noqa: E402

SOURCE_WAV = "jfk.wav"
WINDOW_START_S = 0.5
E2, DS7 = 82.4069, 2489.0159
CONTENTVEC_HF = os.path.join(cm.DEV_CACHE, "applio-hf", "embedders", "contentvec", "pytorch_model.bin")
CONTENTVEC_ONNX = os.path.join(cm.DEV_CACHE, "vc-assets", "vc", "embedder-contentvec", "model.fp32.onnx")
GRAPHS = ("ddsp", "reflow_step", "velocity_f", "vocoder", "encoder", "velocity")


def singing_f0(n_frames, start, hop=sp.HOP, sr=sp.SR):
    """E2 before ``start``; from ``start`` a log sweep E2 -> D#7 with +-25 cent vibrato at 5.5 Hz,
    clipped to the range."""
    f0 = np.full(n_frames, E2, np.float64)
    k = np.arange(n_frames - start)
    frac = k / max(1, n_frames - start - 1)
    sweep = E2 * (DS7 / E2) ** frac * 2.0 ** (0.25 * np.sin(2 * np.pi * 5.5 * k * hop / sr) / 12.0)
    f0[start:] = np.clip(sweep, E2, DS7)
    return f0.astype(np.float32)[None]


def units_upstream(wav16k):
    """Upstream ``Audio2ContentVec768L12TTA2X`` (ddsp/vocoder.py) on the HF-format ContentVec
    checkpoint (Applio's copy, the class DDSP-SVC loads; strict)."""
    from ddsp.vocoder import Audio2ContentVec768L12TTA2X
    enc = Audio2ContentVec768L12TTA2X(cm.need(CONTENTVEC_HF, "Applio contentvec pytorch_model.bin (P3-A fetched it)"), device="cpu")
    return enc(torch.from_numpy(wav16k)[None]).numpy()


def units_sigil(sess, wav16k):
    """The shared embedder twice (the audio, and the audio delayed by 160 samples), interleaved."""
    a = cm.run(sess, {"audio": wav16k[None]})[0]
    b = cm.run(sess, {"audio": np.concatenate([np.zeros(160, np.float32), wav16k])[None]})[0]
    return sp.tta2x(a, b)


class Tick:
    """One upstream real-time tick's inputs for a preset."""

    def __init__(self, preset, wav44):
        s = preset.shapes()
        self.preset, self.shapes = preset, s
        start = int(WINDOW_START_S * sp.SR)
        self.audio = np.ascontiguousarray(wav44[start: start + s["n44"]])
        import torchaudio
        # upstream Units_Encoder.encode: torchaudio Resample(44100 -> 16000, lowpass_filter_width 128)
        self.wav16k = torchaudio.transforms.Resample(sp.SR, 16000, lowpass_filter_width=128)(torch.from_numpy(self.audio)[None])[0].numpy()
        self.n_frames = s["frames"]
        self.start_frame = s["start_frame"]
        self.f0 = singing_f0(self.n_frames, self.start_frame)
        self.volume = sp.volume_envelope(self.audio)[None]
        safe_prefix = preset.extra - preset.crossfade - 0.01 - 0.02
        self.silence_front = safe_prefix - 0.03 if safe_prefix > 0.03 else 0.0


def upstream_volume_gate(audio44, out, threshold_db):
    """``SvcDDSP.infer``'s gate on ``Unit2Wav.forward``'s output ``out``, with upstream's own code:
    ``Volume_Extractor(hop, volume_smooth_size)`` on the window, ``volume > 10^(threshold / 20)``,
    ``ddsp.core.upsample`` by the block size, ``output *= mask[:, -len:]``."""
    from ddsp.core import upsample
    from ddsp.vocoder import Volume_Extractor
    a = ex.load_config()
    volume = Volume_Extractor(a.data.block_size, a.data.volume_smooth_size).extract(audio44)
    mask = (volume > 10 ** (float(threshold_db) / 20)).astype("float")
    mask = upsample(torch.from_numpy(mask).float().unsqueeze(-1).unsqueeze(0), a.data.block_size).squeeze(-1)
    o = torch.from_numpy(np.asarray(out, np.float32))[None]
    o = o * mask[:, -o.shape[-1]:]
    return o[0].numpy(), float(1.0 - mask[:, -o.shape[-1]:].mean())


def run_upstream(up, tick, units, formant_shift, seed):
    """``Unit2Wav.forward`` as ``SvcDDSP.infer`` calls it (the volume gate after it is
    ``upstream_volume_gate``)."""
    model, vocoder = up["model"], up["vocoder"]
    m2p = sp.mel2ph(tick.n_frames, units.shape[1])
    aligned = torch.from_numpy(units)[:, m2p[0]]
    seen = {}
    ext, inf = vocoder.extract, vocoder.infer
    vocoder.extract = lambda audio, *a, **k: seen.setdefault("ddsp_mel", ext(audio, *a, **k))
    vocoder.infer = lambda mel, f0: (seen.setdefault("mel", mel), inf(mel, f0))[1]
    try:
        with torch.no_grad(), cm.recorded_random(seed) as rec:
            out = model(aligned, torch.from_numpy(tick.f0)[..., None], torch.from_numpy(tick.volume)[..., None],
                        aug_shift=torch.tensor([[float(formant_shift)]]), vocoder=vocoder, infer=True, return_wav=True,
                        infer_step=tick.preset.steps, method="euler", t_start=tick.preset.t_start,
                        silence_front=tick.silence_front, use_tqdm=False)
    finally:
        del vocoder.extract, vocoder.infer
    draws = {"noise": next(t for k, t in rec if k == "randn_like").numpy(), "z": next(t for k, t in rec if k == "randn").numpy()}
    return out.numpy().reshape(-1), {"ddsp_mel": seen["ddsp_mel"].numpy(), "mel": seen["mel"].numpy()}, draws, m2p


def parity(args, sessions, up, report, files=None):
    preset = sp.PRESETS[args.preset]
    wav16 = cm.load_wav16k(SOURCE_WAV)
    wav44 = cm.resample(wav16, 16000, sp.SR)
    tick = Tick(preset, wav44)
    rows = {}

    # units: Sigil's shared ContentVec embedder (tta2x) vs upstream's encoder
    u_up = units_upstream(tick.wav16k)
    emb = cm.session(CONTENTVEC_ONNX, args.ep, args.threads)
    u_sg = units_sigil(emb, tick.wav16k)
    rows["units_tta2x_shared_embedder"] = cm.gate(cm.compare(u_up, u_sg))
    # (the session stays alive: deleting a DirectML session and then running another one crashed
    # onnxruntime-directml 1.24.4 in this process, an access violation inside run())

    # one tick of the chain, upstream vs Sigil, same units and draws
    out_u, seen, d, m2p = run_upstream(up, tick, u_up, args.formant, args.seed)
    zs = sp.DdspSvc(sessions)
    common_in = dict(units=u_up, m2p=m2p, f0=tick.f0, volume=tick.volume, noise=d["noise"], z=d["z"],
                     start_frame=tick.start_frame, steps=preset.steps, t_start=preset.t_start, formant_shift=args.formant,
                     threshold_db=None)
    out_s, st = zs.tick(**common_in)
    ddsp_mel = seen["ddsp_mel"].transpose(0, 2, 1)                       # [1,128,R]
    rows["ddsp_cond"] = cm.gate(cm.compare(ddsp_mel, st["cond"]))
    rows["ddsp_x"] = cm.gate(cm.compare((ddsp_mel + 12.0) / 14.0 * 2 - 1, st["x_ddsp"][:, 0]))
    rows["chain_mel"] = cm.gate(cm.compare(seen["mel"].transpose(0, 2, 1), st["mel"]))
    rows["chain_audio"] = cm.gate(cm.compare_audio(out_u, out_s.ravel(), sp.SR))
    out_f, _ = zs.tick(**common_in, loop="float")
    rows["chain_audio_host_update"] = cm.gate(cm.compare_audio(out_u, out_f.ravel(), sp.SR))
    # formant shift 0 (the other code path of the aug_shift embedding)
    out_u0, _, d0, _ = run_upstream(up, tick, u_up, 0.0, args.seed + 1)
    out_s0, _ = zs.tick(**{**common_in, "noise": d0["noise"], "z": d0["z"], "formant_shift": 0.0})
    rows["chain_audio_formant0"] = cm.gate(cm.compare_audio(out_u0, out_s0.ravel(), sp.SR))
    # SvcDDSP.infer's volume gate on both sides, at a threshold that mutes about half of the decoded
    # frames (the median frame volume, rounded to the GUI slider's whole dB) so both branches run
    r = out_u.size // sp.HOP
    thr = float(np.round(20 * np.log10(max(float(np.median(tick.volume[0, -r:])), 1e-9))))
    gated_u, muted = upstream_volume_gate(tick.audio, out_u, thr)
    out_g, _ = zs.tick(**{**common_in, "threshold_db": thr})
    row = cm.gate(cm.compare_audio(gated_u, out_g.ravel(), sp.SR))
    row.update({"threshold_db": thr, "muted_fraction": round(muted, 3)})
    rows["chain_audio_volume_gate"] = row

    # the vocoder alone over the singing range: 4 s of real mel, the E2 -> D#7 sweep
    seg = wav44[: 4 * sp.SR]
    with torch.no_grad():
        mel = up["vocoder"].extract(torch.from_numpy(seg)[None])            # [1,T,128]
        n = mel.shape[1]
        f0 = torch.from_numpy(singing_f0(n, 0))
        a_ref = up["vocoder"].infer(mel, f0[..., None]).numpy().ravel()
    a_got = cm.run(sessions["vocoder"], {"mel": mel.transpose(1, 2).numpy(), "f0": f0.numpy(), "phase0": np.zeros(1, np.float32)})[0].ravel()
    rows["vocoder_sweep_E2_Ds7_4s"] = cm.gate(cm.compare_audio(a_ref, a_got, sp.SR))

    # upstream's own exporter (for the record; not gated)
    ref = {}
    x_e, cond_e = cm.run(sessions["encoder"], {"hubert": u_up, "mel2ph": m2p, "f0": tick.f0, "volume": tick.volume, "randn": d["noise"]})
    ref["encoder_cond_tail_vs_upstream"] = cm.compare(ddsp_mel, cond_e[..., -ddsp_mel.shape[-1]:])
    for loop in ("int_round", "int_trunc"):
        o, _ = zs.tick(**common_in, loop=loop)
        ref[f"velocity_int64_t_{loop}_chain_audio"] = cm.compare_audio(out_u, o.ravel(), sp.SR)
    ref["encoder_nan_at_f0_0"] = bool(np.isnan(cm.run(sessions["encoder"], {
        "hubert": u_up, "mel2ph": m2p, "f0": np.zeros_like(tick.f0), "volume": tick.volume, "randn": d["noise"]})[1]).any())
    report["upstream_exporter"] = ref
    report["parity"] = rows
    report["parity_setup"] = {"preset": dataclasses.asdict(preset), "shapes": preset.shapes(), "source": SOURCE_WAV,
                              "window_start_s": WINDOW_START_S, "seed": args.seed, "formant_shift": args.formant,
                              "f0": "log sweep E2 (82.4 Hz) -> D#7 (2489 Hz) over the decoded frames, +-25 cents vibrato at 5.5 Hz",
                              "weights": f"DDSP + LYNXNet2: seeded random ({ex.SEED}); vocoder: openvpi pc_nsf_hifigan_44.1k_hop512_128bin_2025.02",
                              "reference_impl": "upstream Unit2Wav.forward (6.3), PyTorch fp32 CPU; SvcDDSP.infer's volume gate in chain_audio_volume_gate"}
    if files:
        report["parity_setup"]["graphs"] = cm.graph_stamp({**files, "contentvec": CONTENTVEC_ONNX})
    return report


# ---------------------------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------------------------

def tick_calls(files, preset, f):
    """(graph path, pinned dims, feeds) of every call of one tick at a preset's shapes."""
    s = preset.shapes()
    t, r, u = s["frames"], s["reflow_frames"], s["units_tta2x"]
    n16 = s["n16"]
    cv_t = (n16 - 400) // 320 + 1
    return {
        "contentvec": (CONTENTVEC_ONNX, {"N": n16, "T": cv_t}, {"audio": f(1, n16) * 0.1}),
        "ddsp": (files["ddsp"], {"U": u, "T": t, "S": 512 * t, "R": r},
                 {"hubert": f(1, u, 768), "mel2ph": sp.mel2ph(t, u), "f0": np.full((1, t), 220.0, np.float32),
                  "volume": np.abs(f(1, t)) * 0.05, "randn": f(1, 512 * t), "start_frame": np.array([s["start_frame"]], np.int64),
                  "formant_shift": np.zeros((1, 1), np.float32)}),
        "reflow_step": (files["reflow_step"], {"T": r}, {"x": f(1, 1, 128, r), "t": np.array([0.8], np.float32),
                                                         "dt": np.array([0.015], np.float32), "cond": f(1, 128, r) - 5.0}),
        "vocoder": (files["vocoder"], {"T": r, "S": 512 * r}, {"mel": f(1, 128, r) - 5.0, "f0": np.full((1, r), 220.0, np.float32),
                                                              "phase0": np.zeros(1, np.float32)}),
    }


def cost(args, files, report):
    report["timing_conditions"] = cm.timing_conditions(args.ep)
    rng = np.random.default_rng(0)
    f = lambda *shape: rng.standard_normal(shape).astype(np.float32)  # noqa: E731
    out = {}
    for name, preset in sp.PRESETS.items():
        s = preset.shapes()
        calls = tick_calls(files, preset, f)
        rows = {}
        for g, (path, dims, feeds) in calls.items():
            sess = cm.session(path, args.ep, args.threads, dims=dims)
            rows[g] = cm.timed_session_call(sess, feeds, args.ep, args.warm, args.runs, path=path, dims=dims, threads=args.threads)
            del sess
        totals = {}
        for mode in ("host", "bound", "graph"):
            if not any(mode in rr and "median_ms" in rr[mode] for rr in rows.values()):
                continue
            pick = lambda rr, mode=mode: rr[mode]["median_ms"] if mode in rr and "median_ms" in rr[mode] else cm.best_ms(rr)[0]  # noqa: E731
            tick_ms = 2 * pick(rows["contentvec"]) + pick(rows["ddsp"]) + preset.steps * pick(rows["reflow_step"]) + pick(rows["vocoder"])
            totals[mode] = {"tick_ms": round(tick_ms, 2), "per_step_ms": pick(rows["reflow_step"]), "rtf": round(tick_ms / (preset.block * 1000), 3)}
        rows["tick"] = totals
        rows["block_ms"] = round(preset.block * 1000, 1)
        rows["preset"] = dataclasses.asdict(preset)
        rows["shapes"] = s
        out[name] = rows
        print(f"{name}: block {rows['block_ms']} ms; " + "; ".join(
            f"{k} tick {v['tick_ms']} ms (step {v['per_step_ms']} x {preset.steps}, RTF {v['rtf']})" for k, v in totals.items()))
    out["note"] = "tick = 2 x contentvec (tta2x) + ddsp + steps x reflow_step + vocoder; F0 (RMVPE/FCPE) and resampling not included"
    out["graphs"] = cm.graph_stamp({**files, "contentvec": CONTENTVEC_ONNX})
    report["cost"] = out


def profile(args, files, report):
    """ORT's profile of every graph of the tick at each preset (``common.profile_session_call``)."""
    rng = np.random.default_rng(0)
    f = lambda *shape: rng.standard_normal(shape).astype(np.float32)  # noqa: E731
    out = {}
    for name, preset in sp.PRESETS.items():
        out[name] = {}
        for g, (path, dims, feeds) in tick_calls(files, preset, f).items():
            out[name][g] = cm.profile_session_call(path, feeds, args.ep, args.threads, dims=dims, warm=2, runs=args.profile_runs)
            p = out[name][g]
            print(f"profile {name} {g}: wall {p['wall_ms_per_run']} ms, kernels {p['kernel_ms_per_run']} ms, nodes {p['nodes_by_provider']}, "
                  f"pinned {p['session']['dims_pinned']}")
    out["graphs"] = cm.graph_stamp({**files, "contentvec": CONTENTVEC_ONNX})
    report["profile"] = out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", default="cpu", choices=sorted(cm.EPS))
    ap.add_argument("--preset", default="upstream_gui", choices=sorted(sp.PRESETS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--formant", type=float, default=2.0, help="formant shift (semitones, upstream aug_shift) of the main chain row")
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--profile", action="store_true", help="add ORT's profile of every tick graph at each preset (merges)")
    ap.add_argument("--profile-runs", type=int, default=5)
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()

    cm.import_ort(args.ep)
    files = {n: cm.need(os.path.join(ex.OUT, f"{n}.onnx"), "run export.py first") for n in GRAPHS}
    report = {"family": "ddsp_svc", "model": "DDSP-SVC 6.3 architecture (reflow.yaml) + openvpi PC-NSF-HiFiGAN 44.1k",
              "env": cm.env_info(args.ep), "threads": args.threads, "files": {n: os.path.basename(p) for n, p in files.items()}}
    if not args.no_parity:
        up = ex.load_upstream("cpu")
        sessions = {n: cm.session(p, args.ep, args.threads) for n, p in files.items()}
        parity(args, sessions, up, report, files)
        bad = [k for k, r in report["parity"].items() if not r["pass"]]
        print(f"[{args.ep}] parity: {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}; tick audio {report['parity']['chain_audio']['snr_db']} dB")
        for k, r in report["parity"].items():
            print(cm.row_line(k, r))
        for k, r in report["upstream_exporter"].items():
            print(f"   upstream exporter {k}: {r if isinstance(r, bool) else r['snr_db']}")
        del sessions, up
    if not args.no_time:
        cost(args, files, report)
    if args.profile:
        profile(args, files, report)
    # a --no-parity / --no-time rerun keeps the other half of an existing report
    path = cm.write_report(f"ddsp_svc_{args.ep}.json", report, merge=args.no_parity or args.no_time)
    print("wrote", path)


if __name__ == "__main__":
    main()
