#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P3B-G margin trim probe: how much of the decoder's left margin a *trained* voice really needs.

``dec_margin`` (11 frames for v2-40k HiFi-GAN) is the decoder's structural receptive field, measured
from float64 gradients on random weights (``export_rvc.measure_margins``): any non-zero dependence
counts. The engine decodes those frames every tick and throws their audio away, so each frame of
margin a voice does not need is decoder work saved (T4: about 1/23 of the CPU decoder at 80 ms
blocks per frame).

This probe runs the template's flowdec arithmetic (``rvc_arch.split``: flow reverse + NSF source +
HiFi-GAN) in **float64** with a trained checkpoint, on the same window twice: once with the full
margin, once starting ``k`` frames later with ``dec_margin - k`` (the phase carried exactly like the
engine's, ``parity_test.nsf_phase``), and reports the largest change of the returned audio relative
to its peak. A trim of ``k`` frames is admissible only when that change is at most ``--gate``
(1e-5, ten times under the fp32 parity gate) for every window, input and speaker tested; P3B's rule
adopts per-voice margins only if at least one frame survives on every tested voice.

    python scripts/vc-templates/margin_probe.py --pth %LOCALAPPDATA%/sigil-dev-cache/rvc-default-voice/default.pth \\
        --out scripts/bench/specs/p3b/g/reports/margin_probe.json

CPU only (torch, float64); a few minutes. No network.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_rvc  # noqa: E402
import parity_test  # noqa: E402
from rvc_arch import configs, split  # noqa: E402


def excitation64(m_source, sr, upp, f0, phase0, src_noise):
    """``split.excitation`` in float64 (the template computes the source in fp32; the probe keeps
    every step exact so only the margin differs between the two windows)."""
    gen = m_source.l_sin_gen
    f0 = f0.double().unsqueeze(-1)
    rad = f0 / sr * torch.arange(1, upp + 1, dtype=torch.float64)
    rad2 = torch.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
    rad_acc = rad2.cumsum(dim=1).fmod(1.0)
    rad_acc = torch.cat([torch.zeros_like(rad_acc[:, :1, :]), rad_acc[:, :-1, :]], dim=1)
    rad = (rad + rad_acc).reshape(1, -1, 1) + phase0.double().reshape(1, 1, 1)
    sine = torch.sin(2 * math.pi * rad) * gen.sine_amp
    uv = (f0 > gen.voiced_threshold).double()
    uv = uv.expand(-1, -1, upp).reshape(1, -1, 1)
    noise_amp = uv * gen.noise_std + (1 - uv) * gen.sine_amp / 3
    noise = noise_amp * src_noise.double().reshape(1, -1, 1)
    sine = sine * uv + noise
    lin = m_source.l_linear
    har = torch.tanh(torch.nn.functional.linear(sine, lin.weight.double(), lin.bias.double()))
    return har.transpose(1, 2)


def flowdec64(net, sr, upp, dec_margin, m_p, logs_p, z_noise, g, f0, phase0, src_noise):
    """``split.SigilFlowDec.forward`` in float64 with ``dec_margin``: the last
    ``F - FLOW_MARGIN - dec_margin`` frames of audio."""
    z_p = m_p + torch.exp(logs_p) * z_noise
    z = split._flow_reverse(net.flow, z_p, g)[:, :, split.FLOW_MARGIN :]
    har = excitation64(net.dec.m_source, sr, upp, f0, phase0, src_noise)[:, :, split.FLOW_MARGIN * upp :]
    audio = split._decoder(net.dec, z, har, g)
    return audio[:, 0, dec_margin * upp :]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pth", action="append", required=True, help="trained RVC checkpoint (repeatable)")
    ap.add_argument("--template", default="v2-40k-f0-hifigan")
    ap.add_argument("--seeds", default="1,2,3,4,5", help="input seeds (smooth random-walk content, the parity inputs)")
    ap.add_argument("--returns", default="8,12", help="returned frames per window (40 ms and 80 ms blocks + tail)")
    ap.add_argument("--max-k", type=int, default=5)
    ap.add_argument("--gate", type=float, default=1e-5)
    ap.add_argument("--frames", type=int, default=220)
    ap.add_argument("--out", default=None)
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    spec = configs.TEMPLATES[args.template]
    layout_path = os.path.join(export_rvc.DEFAULT_OUT, args.template, "layout.json")
    with open(layout_path, encoding="utf-8") as f:
        streaming = json.load(f)["streaming"]
    dec_margin = int(streaming["dec_margin"])
    upp = configs.upp(spec["config"])
    sr = configs.sample_rate(spec["config"])
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    returns = [int(r) for r in args.returns.split(",") if r.strip()]
    report = {"template": args.template, "dec_margin": dec_margin, "flow_margin": split.FLOW_MARGIN, "gate": args.gate, "voices": []}
    started = time.time()
    for pth in args.pth:
        ckpt = parity_test.load_checkpoint(pth)
        state = parity_test.model_state(ckpt)
        net = parity_test.reference_net(state, spec)
        export_rvc.remove_weight_norm_(net)
        net = net.double().eval()
        speakers = int(state["emb_g.weight"].shape[0])
        worst = {k: 0.0 for k in range(1, args.max_k + 1)}
        rows = []
        with torch.no_grad():
            for seed in seeds:
                inp = parity_test.make_inputs(spec, args.frames, seed)
                t = {k: torch.from_numpy(v) for k, v in inp.items()}
                lengths = torch.tensor([args.frames])
                m_p, logs_p, _ = net.enc_p(t["phone"].double(), t["pitch"], lengths)
                z_noise = t["z_noise_unit"].double() * float(streaming.get("noise_scale", 0.66666))
                phase = parity_test.nsf_phase(inp["f0"][0], sr, upp)
                for sid in sorted({0, speakers - 1}):
                    g = net.emb_g(torch.tensor([sid])).unsqueeze(-1).double()
                    for r in returns:
                        width = r + split.FLOW_MARGIN + dec_margin
                        # Windows ending in voiced speech, at an unvoiced stretch and near the end.
                        for e in sorted({width + 5, int(args.frames * 0.36) + r, int(args.frames * 0.6), args.frames - 2}):
                            s = e - width
                            if s < 0:
                                continue

                            def window(start, margin):
                                return flowdec64(
                                    net,
                                    sr,
                                    upp,
                                    margin,
                                    m_p[:, :, start:e],
                                    logs_p[:, :, start:e],
                                    z_noise[:, :, start:e],
                                    g,
                                    t["f0"][:, start:e].double(),
                                    torch.tensor([[phase[start]]], dtype=torch.float64),
                                    t["src_noise"][:, start * upp : e * upp].double(),
                                )

                            full = window(s, dec_margin)
                            peak = float(full.abs().max())
                            row = {"seed": seed, "speaker": sid, "return_frames": r, "end": e, "peak": peak}
                            for k in range(1, args.max_k + 1):
                                trimmed = window(s + k, dec_margin - k)
                                rel = float((trimmed - full).abs().max()) / max(peak, 1e-12)
                                row[f"k{k}"] = rel
                                worst[k] = max(worst[k], rel)
                            rows.append(row)
        survive = 0
        for k in range(1, args.max_k + 1):
            if worst[k] <= args.gate:
                survive = k
            else:
                break
        voice = {
            "pth": os.path.basename(pth),
            "speakers": speakers,
            "windows": len(rows),
            "worst_relative_change": {f"k{k}": worst[k] for k in worst},
            "frames_that_survive_the_gate": survive,
            "rows": rows,
        }
        report["voices"].append(voice)
        print(f"{os.path.basename(pth)}: dec_margin {dec_margin}, {len(rows)} windows")
        for k in worst:
            print(f"  trim {k} frame(s): worst relative change {worst[k]:.3e} ({'passes' if worst[k] <= args.gate else 'fails'} the {args.gate:g} gate)")
        print(f"  frames that survive: {survive}")
    report["frames_that_survive_every_voice"] = min(v["frames_that_survive_the_gate"] for v in report["voices"])
    report["seconds"] = round(time.time() - started, 1)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
