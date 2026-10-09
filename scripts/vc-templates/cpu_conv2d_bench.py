#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P3B-G CPU experiment: the flowdec decoder with 2-D (1 x k) convolutions, so ONNX Runtime's NCHWc
direct-convolution kernels can run (its NCHWc transformer converts only Conv nodes with 4-D weights;
every template conv is 1-D today).

The decoder of the template's flowdec (``rvc_arch.split.SigilFlowDec``) is copied with every
``Conv1d`` / ``ConvTranspose1d`` replaced by the equivalent ``Conv2d`` / ``ConvTranspose2d`` over
``[N, C, 1, T]`` (the same weights viewed ``[O, I, 1, K]``: identical bytes, so ``weights.f32.bin``,
``weights_id`` and the importer would not change); the flow stays 1-D. Both graphs are exported
with a trained checkpoint's weights at one fixed window, checked for parity (fp32, against each
other), dumped as ONNX Runtime optimizes them on the CPU (NchwcConv and reorder nodes counted) and
timed interleaved (ABAB) at 1, 2 and 4 intra-op threads with Sigil's session options (sequential,
Level-3 optimization, no spinning). P3B's gate: adopt a CPU-only 2-D variant only when it is at
least 15 % faster at every thread count measured (the CPU engine runs at several).

    python scripts/vc-templates/cpu_conv2d_bench.py --pth %LOCALAPPDATA%/sigil-dev-cache/rvc-default-voice/default.pth \\
        --out scripts/bench/specs/p3b/g/reports/cpu_conv2d_bench.json

CPU only. Run it under the GPU timing lock (a timing run, P3B rule R2) on a quiet machine.
"""

import argparse
import copy
import json
import os
import statistics
import sys
import tempfile
import time

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_rvc  # noqa: E402
import parity_test  # noqa: E402
from rvc_arch import configs, split  # noqa: E402


class Conv2dAs1d(nn.Module):
    """A ``Conv1d`` as a 1 x k ``Conv2d`` over ``[N, C, 1, T]`` (same weights)."""

    def __init__(self, c: nn.Conv1d):
        super().__init__()
        self.conv = nn.Conv2d(
            c.in_channels,
            c.out_channels,
            (1, c.kernel_size[0]),
            stride=(1, c.stride[0]),
            padding=(0, c.padding[0]),
            dilation=(1, c.dilation[0]),
            groups=c.groups,
            bias=c.bias is not None,
        )
        with torch.no_grad():
            self.conv.weight.copy_(c.weight.detach().unsqueeze(2))
            if c.bias is not None:
                self.conv.bias.copy_(c.bias.detach())

    def forward(self, x):
        return self.conv(x)


class ConvT2dAs1d(nn.Module):
    """A ``ConvTranspose1d`` as a 1 x k ``ConvTranspose2d`` over ``[N, C, 1, T]``."""

    def __init__(self, c: nn.ConvTranspose1d):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            c.in_channels,
            c.out_channels,
            (1, c.kernel_size[0]),
            stride=(1, c.stride[0]),
            padding=(0, c.padding[0]),
            output_padding=(0, c.output_padding[0]),
            dilation=(1, c.dilation[0]),
            groups=c.groups,
            bias=c.bias is not None,
        )
        with torch.no_grad():
            self.conv.weight.copy_(c.weight.detach().unsqueeze(2))
            if c.bias is not None:
                self.conv.bias.copy_(c.bias.detach())

    def forward(self, x):
        return self.conv(x)


def to_2d_(module):
    """Replace every Conv1d / ConvTranspose1d under ``module`` (in place)."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.ConvTranspose1d):
            setattr(module, name, ConvT2dAs1d(child))
        elif isinstance(child, nn.Conv1d):
            setattr(module, name, Conv2dAs1d(child))
        else:
            to_2d_(child)
    return module


class FlowDec2d(nn.Module):
    """``split.SigilFlowDec`` with the decoder on 4-D tensors (the flow stays 1-D)."""

    def __init__(self, flow, dec, sr, dec_margin):
        super().__init__()
        self.flow = flow
        self.dec = to_2d_(copy.deepcopy(dec))
        self.m_source = dec.m_source
        self.sr = int(sr)
        self.upp = int(dec.upp)
        self.dec_margin = int(dec_margin)

    def forward(self, m_p, logs_p, z_noise, g, f0, phase0, src_noise):
        z_p = m_p + torch.exp(logs_p) * z_noise
        z = split._flow_reverse(self.flow, z_p, g)[:, :, split.FLOW_MARGIN :]
        har = split.excitation(self.m_source, self.sr, self.upp, f0, phase0, src_noise)[:, :, split.FLOW_MARGIN * self.upp :]
        audio = split._decoder(self.dec, z.unsqueeze(2), har.unsqueeze(2), g.unsqueeze(2))  # [1, 1, 1, S]
        return audio.reshape(1, -1)[:, self.dec_margin * self.upp :].float()


def export(module, args, path):
    names = ["m_p", "logs_p", "z_noise", "g", "f0", "phase0", "src_noise"]
    with torch.no_grad():
        torch.onnx.export(module, args, path, input_names=names, output_names=["audio"], opset_version=export_rvc.OPSET, do_constant_folding=True)


def session(path, threads, optimized=None):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    if optimized:
        so.optimized_model_filepath = optimized
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def count_ops(path):
    import onnx

    m = onnx.load(path, load_external_data=False)
    counts = {}
    for n in m.graph.node:
        key = n.op_type if not n.domain or n.domain == "ai.onnx" else f"{n.domain}.{n.op_type}"
        counts[key] = counts.get(key, 0) + 1
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pth", required=True)
    ap.add_argument("--template", default="v2-40k-f0-hifigan")
    ap.add_argument("--return-frames", type=int, default=15, help="returned frames (a 120 ms block + the SOLA tail)")
    ap.add_argument("--threads", default="1,2,3,4")
    ap.add_argument("--runs", type=int, default=24, help="timed runs per side per thread count")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(2)
    spec = configs.TEMPLATES[args.template]
    with open(os.path.join(export_rvc.DEFAULT_OUT, args.template, "layout.json"), encoding="utf-8") as f:
        streaming = json.load(f)["streaming"]
    dec_margin = int(streaming["dec_margin"])
    upp = configs.upp(spec["config"])
    sr = configs.sample_rate(spec["config"])
    state = parity_test.model_state(parity_test.load_checkpoint(args.pth))
    net = parity_test.reference_net(state, spec)
    export_rvc.remove_weight_norm_(net)
    net = net.float().eval()
    frames = args.return_frames + split.FLOW_MARGIN + dec_margin
    inp = parity_test.make_inputs(spec, frames + 40, 3)
    t = {k: torch.from_numpy(v) for k, v in inp.items()}
    with torch.no_grad():
        m_p, logs_p, _ = net.enc_p(t["phone"], t["pitch"], torch.tensor([frames + 40]))
        g = net.emb_g(torch.tensor([0])).unsqueeze(-1)
    s = 40
    phase = parity_test.nsf_phase(inp["f0"][0], sr, upp)
    feeds = {
        "m_p": m_p[:, :, s:].numpy(),
        "logs_p": logs_p[:, :, s:].numpy(),
        "z_noise": (t["z_noise_unit"][:, :, s:] * 0.66666).numpy(),
        "g": g.numpy(),
        "f0": inp["f0"][:, s:],
        "phase0": np.array([[phase[s]]], dtype=np.float32),
        "src_noise": inp["src_noise"][:, s * upp :],
    }
    targs = tuple(torch.from_numpy(feeds[k]) for k in ["m_p", "logs_p", "z_noise", "g", "f0", "phase0", "src_noise"])
    one = split.SigilFlowDec(net.flow, net.dec, sr, dec_margin).eval()
    two = FlowDec2d(net.flow, net.dec, sr, dec_margin).eval()
    with torch.no_grad():
        a, b = one(*targs), two(*targs)
    torch_rel = float((a - b).abs().max() / a.abs().max())
    report = {"template": args.template, "frames": frames, "return_frames": args.return_frames, "torch_parity_rel": torch_rel}
    print(f"torch: 2-D vs 1-D max relative difference {torch_rel:.2e}")
    with tempfile.TemporaryDirectory() as tmp:
        p1, p2 = os.path.join(tmp, "flowdec1d.onnx"), os.path.join(tmp, "flowdec2d.onnx")
        export(one, targs, p1)
        export(two, targs, p2)
        o1, o2 = os.path.join(tmp, "flowdec1d.opt.onnx"), os.path.join(tmp, "flowdec2d.opt.onnx")
        s1, s2 = session(p1, 1, o1), session(p2, 1, o2)
        y1, y2 = s1.run(None, feeds)[0], s2.run(None, feeds)[0]
        ort_rel = float(np.abs(y1 - y2).max() / np.abs(y1).max())
        report["ort_parity_rel"] = ort_rel
        report["ops_1d"] = count_ops(o1)
        report["ops_2d"] = count_ops(o2)
        nchwc = {k: v for k, v in report["ops_2d"].items() if "Nchwc" in k or "Reorder" in k}
        print(f"ORT CPU: 2-D vs 1-D max relative difference {ort_rel:.2e}; 2-D optimized graph NCHWc / reorder nodes {nchwc}")
        print(f"  1-D optimized graph: {dict(sorted(report['ops_1d'].items(), key=lambda kv: -kv[1])[:8])}")
        report["timing"] = []
        for threads in [int(x) for x in args.threads.split(",") if x.strip()]:
            s1, s2 = session(p1, threads), session(p2, threads)
            for _ in range(4):
                s1.run(None, feeds)
                s2.run(None, feeds)
            t1, t2 = [], []
            for i in range(args.runs):
                for sess, acc in ((s1, t1), (s2, t2)) if i % 2 == 0 else ((s2, t2), (s1, t1)):
                    st = time.perf_counter()
                    sess.run(None, feeds)
                    acc.append((time.perf_counter() - st) * 1000.0)
            m1, m2 = statistics.median(t1), statistics.median(t2)
            ratios = sorted(b / a for a, b in zip(t1, t2))
            row = {"threads": threads, "ms_1d": m1, "ms_2d": m2, "speedup": m1 / m2, "paired_ratio_median": statistics.median(ratios)}
            report["timing"].append(row)
            print(f"  {threads} thread(s): 1-D {m1:.2f} ms, 2-D {m2:.2f} ms, speedup {m1 / m2:.3f} (paired ratio median {statistics.median(ratios):.3f})")
    # The CPU engine runs at several thread counts (the Streamer preset's and the autotuner's), so
    # the variant must be at least 15 % faster at every count measured, at parity.
    worst = min(r["speedup"] for r in report["timing"])
    report["min_speedup"] = worst
    report["adopt"] = bool(worst >= 1.15 and report["ort_parity_rel"] <= 1e-5)
    print(f"adopt (>= 15 % faster at every thread count, at parity): {report['adopt']} (slowest speedup {worst:.3f})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
