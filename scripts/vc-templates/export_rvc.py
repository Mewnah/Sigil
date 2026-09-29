#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export Sigil's RVC split-graph templates: graph-only ONNX files plus ``layout.json``.

For one template configuration (``rvc_arch/configs.py``) this builds the upstream RVC synthesizer
with random weights, removes weight_norm, and exports three graphs (see ``rvc_arch/split.py``):

    enc.{fp32,fp16}.onnx       phone [1,T,E] f32, pitch [1,T] i64      -> m_p, logs_p [1,C,T] f32
    flowdec.{fp32,fp16}.onnx   m_p, logs_p, z_noise [1,C,F], g [1,G,1], f0 [1,F], phase0 [1,1],
                               src_noise [1,S=F*upp]                   -> audio [1,R*upp] f32
    knn.{fp32,fp16}.onnx       feats [T,E] f32, index [N,E] (f16 in fp16), index_sq [N], index_rate [1]
                                                                       -> feats [T,E] f32

Every weight is an *external initializer* named after its RVC parameter and pointing into
``weights.f32.bin`` (fp32 graphs) or ``weights.f16.bin`` (fp16 graphs, the same layout with every
offset halved, i.e. the f32 file cast word by word). The graphs carry no weights, so a voice is
"imported" by writing those files from its checkpoint (Rust: ``neural_audio::vc::import``) and handing
them to ONNX Runtime in memory. ``layout.json`` maps every initializer to its checkpoint source
(``key`` or ``weight_norm`` g/v/dim) with its offset, and records the streaming margins.

Usage (from the repo root; needs torch and onnx, see requirements.txt):

    python scripts/vc-templates/export_rvc.py                       # v2-40k-f0-hifigan -> src-tauri/resources/vc-templates
    python scripts/vc-templates/export_rvc.py --template v2-16k-f0-hifigan --out src-tauri/test-fixtures/vc/tiny
        (the tiny test config; parity_test.py --write-fixture regenerates the whole fixture)

The export is deterministic: the same torch / onnx versions produce byte-identical graphs.
"""

import argparse
import copy
import io
import math
import os
import sys
import warnings

import numpy as np
import onnx
import torch
from onnx import TensorProto, numpy_helper

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import sigil_layout as sl  # noqa: E402
from rvc_arch import configs, models, split  # noqa: E402

OPSET = 17
RVC_COMMIT = "81eed5e8f68b6bed1789f682fe78cdd324495afc"
# Non-weight constants larger than this (elements) would mean a weight was folded into the graph.
MAX_INLINE_CONST = 4096
DEFAULT_OUT = os.path.join(REPO, "src-tauri", "resources", "vc-templates")
GRAPHS = ("enc", "flowdec", "knn")
# Variant to run per device class, from the GPU parity measurements in docs/adr/0004-rvc-templates.md:
# fp16 activations cost the text encoder accuracy on every GPU EP (and it is ~2% of the compute),
# while the fp16 flowdec matches upstream RVC's own half-precision inference. The autotuner (P3-F)
# may override per PC.
# fp16 flowdec mixed precision: the flow and / or decoder stages computed in fp32 (weights stay
# fp16 in storage). Decoder stage 0 (the widest resblocks: 256 channels, kernels up to 11, ~18% of
# the MACs) in fp32 is what brings DirectML's fp16 numerics under the 5e-3 gate (9.0e-3 -> 2.9e-3
# on the default voice, still 3x faster than fp32); an fp32 flow does not help. ADR 0004 has the
# table. Experiments: --fp32-flow / --fp32-stages.
FLOWDEC_FP16 = {"fp32_flow": False, "fp32_stages": (0,)}
RECOMMENDED = {
    "enc": {"cpu": "fp32", "gpu": "fp32"},
    "flowdec": {"cpu": "fp32", "gpu": "fp16"},
    "knn": {"cpu": "fp32", "gpu": "fp16"},
}
# Parity status per graph variant and device class, written to layout.json ("parity"): the ADR 0004
# gate (fp32 <= 1e-4, fp16 <= 5e-3) measured by parity_test.py (CPU) and default_voice_parity
# (DirectML and CUDA; "gpu" is the worst GPU EP, CUDA fp32 with TF32 off). Sigil refuses a variant
# marked "fail" on that device class (session::template_session), so "recommended" can never
# point at one. enc fp16 misses the gate on every GPU EP (DirectML 2.1e-2, CUDA 6.0e-3).
PARITY = {
    "enc": {"fp32": {"cpu": "pass", "gpu": "pass"}, "fp16": {"cpu": "pass", "gpu": "fail"}},
    "flowdec": {"fp32": {"cpu": "pass", "gpu": "pass"}, "fp16": {"cpu": "pass", "gpu": "pass"}},
    "knn": {"fp32": {"cpu": "pass", "gpu": "pass"}, "fp16": {"cpu": "pass", "gpu": "pass"}},
}
for _graph, _classes in RECOMMENDED.items():
    for _cls, _variant in _classes.items():
        assert PARITY[_graph][_variant][_cls] != "fail", f"{_graph}: recommended {_cls} variant fails parity"
VARIANTS = {"fp32": torch.float32, "fp16": torch.float16}

warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------


def synthesizer_class(version, f0):
    return {
        ("v1", True): models.SynthesizerTrnMs256NSFsid,
        ("v1", False): models.SynthesizerTrnMs256NSFsid_nono,
        ("v2", True): models.SynthesizerTrnMs768NSFsid,
        ("v2", False): models.SynthesizerTrnMs768NSFsid_nono,
    }[(version, bool(f0))]


def build_net(spec):
    net = synthesizer_class(spec["version"], spec["f0"])(*spec["config"], is_half=False)
    if hasattr(net, "enc_q"):
        del net.enc_q  # training-only posterior encoder (RVC drops it from inference checkpoints)
    return net.eval()


def weight_norm_map(net):
    """{folded parameter: (g key, v key, dim)} for every weight_norm'd module of ``net``."""
    from torch.nn.utils.weight_norm import WeightNorm

    out = {}
    for mname, mod in net.named_modules():
        for hook in mod._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                base = f"{mname}.{hook.name}"
                out[base] = (f"{base}_g", f"{base}_v", int(hook.dim))
    return out


def perturb_(net, seed, scale):
    """Add seeded Gaussian noise to every parameter: unique values everywhere (RVC zero-initialises
    the coupling ``post`` layers, which would make the flow an identity)."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in net.named_parameters():
            p.add_(torch.randn(p.shape, generator=gen) * scale)


def remove_weight_norm_(net):
    net.remove_weight_norm()
    from torch.nn.utils.weight_norm import WeightNorm

    for mname, mod in net.named_modules():
        for hook in mod._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                raise RuntimeError(f"weight_norm left on {mname}")


# ---------------------------------------------------------------------------------------------
# Streaming margins (measured, not assumed)
# ---------------------------------------------------------------------------------------------


def _dependency_span(grad):
    """(first, last) index along the last axis where ``grad`` [1, C, L] is non-zero in any channel."""
    nz = torch.nonzero(grad.abs().sum(1)[0] > 0)
    return int(nz.min()), int(nz.max())


def measure_margins(net, spec, frames=96):
    """Left/right receptive fields of the flow (frames) and the decoder's left one (frames, taking
    the NSF source into account), from float64 input gradients around the middle frame."""
    cfg = configs.config_dict(spec["config"])
    upp = configs.upp(spec["config"])
    flow = copy.deepcopy(net.flow).double()
    dec = copy.deepcopy(net.dec).double()
    gen = torch.Generator().manual_seed(7)
    c = frames // 2
    g = torch.randn(1, cfg["gin_channels"], 1, generator=gen, dtype=torch.float64)

    z = torch.randn(1, cfg["inter_channels"], frames, generator=gen, dtype=torch.float64, requires_grad=True)
    out = split._flow_reverse(flow, z, g)
    out[:, :, c].sum().backward()
    lo, hi = _dependency_span(z.grad)
    flow_left, flow_right = c - lo, hi - c

    zd = torch.randn(1, cfg["inter_channels"], frames, generator=gen, dtype=torch.float64, requires_grad=True)
    har = torch.randn(1, 1, frames * upp, generator=gen, dtype=torch.float64, requires_grad=True)
    audio = split._decoder(dec, zd, har, g)
    audio[0, 0, c * upp : (c + 1) * upp].sum().backward()
    z_lo, _ = _dependency_span(zd.grad)
    h_lo, _ = _dependency_span(har.grad)
    dec_left_frames = c - z_lo
    dec_left_samples = c * upp - h_lo
    dec_margin = max(dec_left_frames, math.ceil(dec_left_samples / upp))
    if flow_left > split.FLOW_MARGIN or flow_right > split.FLOW_MARGIN:
        raise RuntimeError(f"flow receptive field {flow_left}/{flow_right} exceeds FLOW_MARGIN")
    if dec_margin >= c:
        raise RuntimeError("decoder receptive field reaches the probe window edge; raise `frames`")
    return {
        "flow_margin": split.FLOW_MARGIN,
        "flow_receptive_left": flow_left,
        "flow_receptive_right": flow_right,
        "dec_margin": dec_margin,
        "dec_receptive_left_frames": dec_left_frames,
        "dec_receptive_left_samples": dec_left_samples,
    }


# ---------------------------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------------------------


def graph_specs(net, spec, dec_margin, dtype, device, mixed=None):
    """{graph: (module, example args, input names, output names, dynamic axes)}."""
    cfg = configs.config_dict(spec["config"])
    sr = configs.sample_rate(spec["config"])
    upp = configs.upp(spec["config"])
    emb = spec["emb_channels"]
    inter, gin = cfg["inter_channels"], cfg["gin_channels"]
    gen = torch.Generator().manual_seed(3)

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device)

    t = 64
    enc = split.SigilEnc(net.enc_p, f0=spec["f0"], dtype=dtype)
    enc_args = [randn(1, t, emb)]
    enc_in = ["phone"]
    enc_axes = {"phone": {1: "T"}, "m_p": {2: "T"}, "logs_p": {2: "T"}}
    if spec["f0"]:
        enc_args.append(torch.randint(1, 255, (1, t), generator=gen).to(device))
        enc_in.append("pitch")
        enc_axes["pitch"] = {1: "T"}

    if not spec["f0"]:
        raise NotImplementedError("flowdec for f0-less (nono) generators is P3-A")
    f = split.FLOW_MARGIN + dec_margin + 16
    f0 = (torch.rand(1, f, generator=gen) * 300 + 80).to(device)
    flowdec = split.SigilFlowDec(net.flow, net.dec, sr, dec_margin, dtype=dtype, **(mixed or {}))
    flowdec_args = [
        randn(1, inter, f),
        randn(1, inter, f) * 0.1,
        randn(1, inter, f) * 0.66666,
        randn(1, gin, 1),
        f0,
        torch.zeros(1, 1, device=device),
        randn(1, f * upp),
    ]
    flowdec_in = ["m_p", "logs_p", "z_noise", "g", "f0", "phase0", "src_noise"]
    flowdec_axes = {
        "m_p": {2: "F"},
        "logs_p": {2: "F"},
        "z_noise": {2: "F"},
        "f0": {1: "F"},
        "src_noise": {1: "S"},
        "audio": {1: "A"},
    }

    n = 128
    index = randn(n, emb).to(dtype)
    knn_args = [randn(16, emb), index, (index.float() ** 2).sum(-1), torch.full((1,), 0.5, device=device)]
    knn_in = ["feats", "index", "index_sq", "index_rate"]
    knn_axes = {"feats": {0: "T"}, "index": {0: "N"}, "index_sq": {0: "N"}, "feats_out": {0: "T"}}

    return {
        "enc": (enc, enc_args, enc_in, ["m_p", "logs_p"], enc_axes),
        "flowdec": (flowdec, flowdec_args, flowdec_in, ["audio"], flowdec_axes),
        "knn": (split.SigilKnn(), knn_args, knn_in, ["feats_out"], knn_axes),
    }


def export_onnx(module, args, input_names, output_names, dynamic_axes):
    buf = io.BytesIO()
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            module,
            tuple(args),
            buf,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=OPSET,
            do_constant_folding=False,  # keep every weight as a named, untransformed parameter
            export_params=True,
            keep_initializers_as_inputs=False,
        )
    return onnx.load_from_string(buf.getvalue())


def module_params(module):
    """{initializer name: numpy value} of a wrapper's parameters (names as the exporter uses)."""
    return {name: p.detach().cpu().numpy() for name, p in module.named_parameters()}


def externalize(model, params, offsets, weights_file, scale):
    """Point every parameter initializer at ``weights_file``; fail on anything that looks like a
    weight baked into the graph. ``scale`` = bytes per element / 4 (1 for f32, 0.5 for f16)."""
    seen = set()
    for init in model.graph.initializer:
        if init.name in params:
            arr = numpy_helper.to_array(init)
            want = params[init.name]
            if arr.shape != want.shape or arr.dtype != want.dtype or not np.array_equal(
                arr.view(np.uint8), np.ascontiguousarray(want).view(np.uint8)
            ):
                raise RuntimeError(f"initializer {init.name} does not hold its parameter's value")
            off, nbytes = offsets[init.name]
            init.ClearField("raw_data")
            for field in ("float_data", "int32_data", "int64_data", "double_data"):
                init.ClearField(field)
            init.data_location = TensorProto.EXTERNAL
            del init.external_data[:]
            for key, value in (("location", weights_file), ("offset", int(off * scale)), ("length", int(nbytes * scale))):
                e = init.external_data.add()
                e.key, e.value = key, str(value)
            seen.add(init.name)
        else:
            n = int(np.prod(init.dims)) if len(init.dims) else 1
            if n > MAX_INLINE_CONST:
                raise RuntimeError(f"unexpected large constant {init.name} {list(init.dims)} in the graph")
    missing = sorted(set(params) - seen)
    if missing:
        raise RuntimeError(f"parameters missing from the exported graph: {missing[:8]}")
    return seen


def io_spec(model):
    def one(v):
        tt = v.type.tensor_type
        dims = [d.dim_param if d.dim_param else int(d.dim_value) for d in tt.shape.dim]
        return {"name": v.name, "dtype": onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type).name, "shape": dims}

    return [one(v) for v in model.graph.input], [one(v) for v in model.graph.output]


def check_dynamic(model, module, args, graph):
    """Run the exported fp32 graph (weights still inline) in ONNX Runtime at the export length and at
    another length, against PyTorch."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]
    worst = 0.0
    for trial in range(2):
        a = [x.clone() for x in args]
        if trial == 1:
            a = _resize_args(graph, a)
        with torch.no_grad():
            ref = module(*a)
        ref = ref if isinstance(ref, tuple) else (ref,)
        feeds = {n: x.detach().cpu().numpy() for n, x in zip(names, a)}
        got = sess.run(None, feeds)
        for r, g in zip(ref, got):
            worst = max(worst, float(np.abs(r.detach().cpu().numpy() - g).max()))
    return worst


def _resize_args(graph, args):
    gen = torch.Generator().manual_seed(11)
    if graph == "enc":
        t = args[0].shape[1] + 37
        out = [torch.randn(1, t, args[0].shape[2], generator=gen)]
        if len(args) > 1:
            out.append(torch.randint(1, 255, (1, t), generator=gen))
        return out
    if graph == "flowdec":
        f = args[0].shape[2] + 29
        upp = args[6].shape[1] // args[0].shape[2]
        c, gin = args[0].shape[1], args[3].shape[1]
        return [
            torch.randn(1, c, f, generator=gen),
            torch.randn(1, c, f, generator=gen) * 0.1,
            torch.randn(1, c, f, generator=gen) * 0.66666,
            torch.randn(1, gin, 1, generator=gen),
            torch.rand(1, f, generator=gen) * 300 + 80,
            torch.full((1, 1), 0.25),
            torch.randn(1, f * upp, generator=gen),
        ]
    t, n, e = 23, 300, args[0].shape[1]
    index = torch.randn(n, e, generator=gen)
    return [torch.randn(t, e, generator=gen), index, (index**2).sum(-1), torch.full((1,), 0.75)]


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------


def export_template(template_id, out_root, seed=1234, verbose=True):
    spec = configs.TEMPLATES[template_id]
    out_dir = os.path.join(out_root, template_id)
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    net = build_net(spec)
    wn = weight_norm_map(net)
    perturb_(net, seed, 0.02)
    remove_weight_norm_(net)
    margins = measure_margins(net, spec)
    dec_margin = margins["dec_margin"]
    if verbose:
        print(f"[{template_id}] margins {margins}")

    fp16_device = "cuda" if torch.cuda.is_available() else "cpu"
    net16 = split.cast_weights_to_input_(copy.deepcopy(net).half().to(fp16_device))
    specs = {
        "fp32": graph_specs(net, spec, dec_margin, torch.float32, "cpu"),
        "fp16": graph_specs(net16, spec, dec_margin, torch.float16, fp16_device, FLOWDEC_FP16),
    }

    # One weights file for every graph: enc tensors first, then flowdec, in state-dict order.
    tensors, offsets, cursor = [], {}, 0
    order = []
    for graph in ("enc", "flowdec"):
        for name, arr in module_params(specs["fp32"][graph][0]).items():
            if name in offsets:
                raise RuntimeError(f"{name} is used by two graphs")
            nbytes = arr.size * 4
            offsets[name] = (cursor, nbytes)
            if name in wn:
                gk, vk, dim = wn[name]
                g_shape = [arr.shape[i] if i == dim else 1 for i in range(arr.ndim)]
                source = {"weight_norm": {"g": gk, "v": vk, "dim": dim, "folded": name, "g_shape": g_shape}}
            else:
                source = {"key": name}
            tensors.append(
                {
                    "name": name,
                    "graph": graph,
                    "shape": list(arr.shape),
                    "offset": cursor,
                    "bytes": nbytes,
                    "source": source,
                }
            )
            order.append(name)
            cursor = sl.align_up(cursor + nbytes)
    weights_id = sl.tensor_table_id(tensors)

    graphs = {}
    for graph in GRAPHS:
        graphs[graph] = {"recommended": RECOMMENDED[graph], "variants": {}}
        for variant, dtype in VARIANTS.items():
            module, args, in_names, out_names, axes = specs[variant][graph]
            model = export_onnx(module, args, in_names, out_names, axes)
            if variant == "fp32":
                worst = check_dynamic(model, module, args, graph)
                if verbose:
                    print(f"[{template_id}] {graph}.fp32 ORT vs torch (2 lengths): max abs {worst:.3g}")
                if worst > 1e-4:
                    raise RuntimeError(f"{graph}: exported graph disagrees with PyTorch ({worst})")
            onnx.checker.check_model(model)  # structure, with the weights still inline
            params = module_params(module)
            if graph != "knn":
                wfile = sl.WEIGHTS_F32 if variant == "fp32" else sl.WEIGHTS_F16
                externalize(model, params, offsets, wfile, 1.0 if variant == "fp32" else 0.5)
            elif params:
                raise RuntimeError("knn has no weights")
            model.producer_name = "sigil-vc-templates"
            model.doc_string = f"Sigil RVC template {template_id}: {graph} ({variant}); weights are external"
            meta = {
                "sigil.format": "sigil-rvc-template",
                "sigil.template": template_id,
                "sigil.graph": graph,
                "sigil.variant": variant,
                "sigil.weights_id": weights_id if graph != "knn" else "",
            }
            del model.metadata_props[:]
            for k, v in meta.items():
                p = model.metadata_props.add()
                p.key, p.value = k, v
            fname = f"{graph}.{variant}.onnx"
            path = os.path.join(out_dir, fname)
            onnx.save_model(model, path)
            inputs, outputs = io_spec(model)
            graphs[graph]["variants"][variant] = {
                "file": fname,
                "sha256": sl.sha256_file(path),
                "bytes": os.path.getsize(path),
                "weights_file": (sl.WEIGHTS_F32 if variant == "fp32" else sl.WEIGHTS_F16) if graph != "knn" else None,
                "parity": PARITY[graph][variant],
                "inputs": inputs,
                "outputs": outputs,
            }
            if verbose:
                print(f"[{template_id}] wrote {fname} ({os.path.getsize(path)} bytes)")

    cfg = configs.config_dict(spec["config"])
    sr = configs.sample_rate(spec["config"])
    upp = configs.upp(spec["config"])
    layout = {
        "format": sl.LAYOUT_FORMAT,
        "format_version": sl.LAYOUT_VERSION,
        "template": template_id,
        "arch": {
            "version": spec["version"],
            "f0": spec["f0"],
            "vocoder": spec["vocoder"],
            "sample_rate": sr,
            "upp": upp,
            "frame_rate": sr // upp,
            "emb_channels": spec["emb_channels"],
            "inter_channels": cfg["inter_channels"],
            "gin_channels": cfg["gin_channels"],
            "config": spec["config"],
            "arch_fields": list(configs.ARCH_FIELDS),
        },
        "streaming": {
            **margins,
            "noise_scale": 0.66666,
            "knn_k": split.KNN_K,
            "knn_coarse_k": split.KNN_COARSE_K,
            "knn_eps": split.KNN_EPS,
            "min_index_rows": split.KNN_COARSE_K,
            "dims": {
                "enc": "T = context frames (10 ms each)",
                "flowdec": "S = F * upp; A = (F - flow_margin - dec_margin) * upp",
                "knn": "T = 20 ms content frames; N >= min_index_rows",
            },
        },
        "weights": {
            "file_f32": sl.WEIGHTS_F32,
            "file_f16": sl.WEIGHTS_F16,
            "alignment": sl.ALIGN,
            "bytes_f32": cursor,
            "weights_id": weights_id,
            "tensors": tensors,
        },
        "speakers": {"key": "emb_g.weight", "dim": cfg["gin_channels"], "file": sl.SPEAKERS_F32},
        "ignored_keys": ["enc_q."],
        "graphs": graphs,
        "exporter": {
            "torch": torch.__version__.split("+")[0],
            "onnx": onnx.__version__,
            "opset": OPSET,
            "rvc_commit": RVC_COMMIT,
        },
    }
    sl.write_json(os.path.join(out_dir, "layout.json"), layout)
    if verbose:
        print(f"[{template_id}] {len(tensors)} tensors, weights.f32.bin = {cursor} bytes, id {weights_id[:12]}")
    return layout, net, wn


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--template", action="append", choices=sorted(configs.TEMPLATES), help="template id (repeatable)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output root (a <template> folder is created inside)")
    ap.add_argument("--fp32-flow", action="store_true", help="experiment: fp16 flowdec computes the flow in fp32")
    ap.add_argument("--fp32-stages", default=None, help="experiment: decoder stages computed in fp32, e.g. 0,1")
    args = ap.parse_args()
    if args.fp32_flow:
        FLOWDEC_FP16["fp32_flow"] = True
    if args.fp32_stages is not None:
        FLOWDEC_FP16["fp32_stages"] = tuple(int(x) for x in args.fp32_stages.split(",") if x.strip())
    for tid in args.template or ["v2-40k-f0-hifigan"]:
        if configs.TEMPLATES[tid].get("test_only") and os.path.abspath(args.out) == os.path.abspath(DEFAULT_OUT):
            raise SystemExit(f"{tid} is a test-only configuration; export it with --out <fixture dir>")
        export_template(tid, args.out)


if __name__ == "__main__":
    main()
