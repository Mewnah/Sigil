#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export Sigil's RVC split-graph templates: graph-only ONNX files plus ``layout.json``.

For one template configuration (``rvc_arch/configs.py``) this builds the RVC synthesizer with random
weights (the decoder is upstream RVC's HiFi-GAN, or Sigil's port of Applio's RefineGAN / MRF HiFi-GAN,
``rvc_arch/vocoders.py``), removes weight_norm, and exports three graphs (see ``rvc_arch/split.py``):

    enc.{fp32,fp16}.onnx       phone [1,T,E] f32 (+ pitch [1,T] i64 for f0 voices) -> m_p, logs_p [1,C,T] f32
    flowdec.{fp32,fp16}.onnx   m_p, logs_p, z_noise [1,C,F], g [1,G,1]
                               (+ f0 [1,F], phase0 [1,1], src_noise [1,S] or [1,S,H] for f0 voices,
                               + noise_offsets [1,F] i64 for RefineGAN, whose m_p / logs_p / z_noise
                               cover only its latent window [1,C,Fz])  -> audio [1,R*upp] f32
    knn.{fp32,fp16}.onnx       feats [T,E] f32, index [N,E] (f16 in fp16), index_sq [N], index_rate [1]
                                                                          -> feats [T,E] f32

Every weight is an *external initializer* named after its RVC parameter and pointing into
``weights.f32.bin`` (fp32 graphs) or ``weights.f16.bin`` (fp16 graphs, the same layout with every
offset halved, i.e. the f32 file cast word by word). RefineGAN's flowdec also points its AdaIN noise
pool at ``noise.f32.bin`` (iid N(0, 1) values the engine generates). The graphs carry no weights, so a
voice is "imported" by writing those files from its checkpoint (Rust: ``neural_audio::vc::import``)
and handing them to ONNX Runtime in memory. ``layout.json`` maps every initializer to its checkpoint
source (``key`` or ``weight_norm`` g/v/dim) with its offset, and records the streaming contract
(margins, excitation, noise).

Usage (from the repo root; needs torch and onnx, see requirements.txt):

    python scripts/vc-templates/export_rvc.py                       # v2-40k-f0-hifigan -> src-tauri/resources/vc-templates
    python scripts/vc-templates/export_matrix.py --template v2-32k-f0-refinegan --dml-python ...
        (any other bundled template: export_matrix measures its parity and writes its recommendation;
        export_rvc.py refuses to write one into src-tauri/resources/vc-templates)
    python scripts/vc-templates/export_rvc.py --template v2-32k-f0-refinegan --out <scratch dir>
        (a scratch export: its layout says parity "untested" and recommends fp32 everywhere)
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
from rvc_arch import configs, models, split, vocoders  # noqa: E402

OPSET = 17
RVC_COMMIT = "81eed5e8f68b6bed1789f682fe78cdd324495afc"
APPLIO_COMMIT = "c7665ac9a305b3683570ed914f1577d53d4b75c4"
# Non-weight constants larger than this (elements) would mean a weight was folded into the graph.
# Architecture constants that are legitimately larger (RefineGAN's sinc resampler kernels) are
# declared by name by their wrapper (``architecture_constants``).
MAX_INLINE_CONST = 4096
DEFAULT_OUT = os.path.join(REPO, "src-tauri", "resources", "vc-templates")
GRAPHS = ("enc", "flowdec", "knn")
LAYOUT_V1_VOCODERS = ("hifigan",)
# RefineGAN's AdaIN noise pool: 2^21 iid N(0, 1) f32 values (8 MiB), at least one frame's worth of
# noise (<= 294,912 values at 48 kHz); each window frame reads its own random slice.
ADAIN_POOL = 1 << 21
ADAIN_NOISE_FILE = "noise.f32.bin"
# Variant to run per device class, from the GPU parity measurements in docs/adr/0004-rvc-templates.md:
# fp16 activations cost the text encoder accuracy on every GPU EP (and it is ~2% of the compute),
# while the fp16 flowdec matches upstream RVC's own half-precision inference. The autotuner (P3-F)
# may override per PC.
# fp16 flowdec mixed precision: the flow and / or decoder stages computed in fp32 (weights stay
# fp16 in storage). Decoder stage 0 (the widest resblocks: 256 channels, kernels up to 11, ~18% of
# the MACs) in fp32 is what brings DirectML's fp16 numerics under the 5e-3 gate (9.0e-3 -> 2.9e-3
# on the default voice, still 3x faster than fp32); an fp32 flow does not help. ADR 0004 has the
# table. Experiments: --fp32-flow / --fp32-stages. The same rule is applied to every vocoder and
# re-measured per vocoder by parity_test.py (--gpu).
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
# These are the P1 measurements of v2-40k-f0-hifigan (P1_TEMPLATE), so export_template writes them
# into that template only (and into the test-only fixtures, which stand in for a shipped template in
# the Rust precision-policy tests); every other template gets UNMEASURED until export_matrix.py
# rewrites its parity (and recommendation) from its own gate run.
P1_TEMPLATE = "v2-40k-f0-hifigan"
PARITY = {
    "enc": {"fp32": {"cpu": "pass", "gpu": "pass"}, "fp16": {"cpu": "pass", "gpu": "fail"}},
    "flowdec": {"fp32": {"cpu": "pass", "gpu": "pass"}, "fp16": {"cpu": "pass", "gpu": "pass"}},
    "knn": {"fp32": {"cpu": "pass", "gpu": "pass"}, "fp16": {"cpu": "pass", "gpu": "pass"}},
}
for _graph, _classes in RECOMMENDED.items():
    for _cls, _variant in _classes.items():
        assert PARITY[_graph][_variant][_cls] != "fail", f"{_graph}: recommended {_cls} variant fails parity"
# A template nobody measured yet: nothing claimed, fp32 everywhere (Sigil never refuses "untested",
# and the recommended fp32 graphs are the ones every vocoder's fp32 export check covers).
UNMEASURED_PARITY = {g: {v: {"cpu": "untested", "gpu": "untested"} for v in ("fp32", "fp16")} for g in PARITY}
UNMEASURED_RECOMMENDED = {g: {"cpu": "fp32", "gpu": "fp32"} for g in RECOMMENDED}


def measured_parity(template_id):
    """(parity table, recommendation) export_template writes: the P1 measurements for P1_TEMPLATE
    and the test-only fixtures, UNMEASURED for every other template."""
    if template_id == P1_TEMPLATE or configs.TEMPLATES[template_id].get("test_only"):
        return PARITY, RECOMMENDED
    return UNMEASURED_PARITY, UNMEASURED_RECOMMENDED
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


def applio_decoder(spec):
    """Sigil's port of the Applio generator a RefineGAN / MRF HiFi-GAN voice was trained with, built
    the way Applio's ``Synthesizer`` builds it from the checkpoint config."""
    cfg = configs.config_dict(spec["config"])
    sr = configs.sample_rate(spec["config"])
    if spec["vocoder"] == "refinegan":
        return vocoders.RefineGANDecoder(
            sr, cfg["upsample_rates"], cfg["inter_channels"], gin_channels=cfg["gin_channels"],
            upsample_initial_channel=cfg["upsample_initial_channel"],
        )
    if spec["vocoder"] == "mrf-hifigan":
        return vocoders.MRFDecoder(
            cfg["inter_channels"], cfg["upsample_initial_channel"], cfg["upsample_rates"], cfg["upsample_kernel_sizes"],
            cfg["resblock_kernel_sizes"], cfg["resblock_dilation_sizes"], cfg["gin_channels"], sr,
        )
    raise ValueError(spec["vocoder"])


def build_net(spec):
    """The synthesizer of a template: RVC's text encoder, flow and speaker table (Applio keeps RVC's
    modules and key names for those) with the template's decoder."""
    net = synthesizer_class(spec["version"], spec["f0"])(*spec["config"], is_half=False)
    if hasattr(net, "enc_q"):
        del net.enc_q  # training-only posterior encoder (RVC drops it from inference checkpoints)
    if spec["vocoder"] != "hifigan":
        if not spec["f0"]:
            raise ValueError(f"no f0-less {spec['vocoder']} (see configs.UNSUPPORTED)")
        net.dec = applio_decoder(spec)  # keeps `dec`'s place in the module order
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
    """Fold every weight_norm hook (RVC's and the Applio ports', all ``torch.nn.utils.weight_norm``)."""
    from torch.nn.utils.weight_norm import WeightNorm

    for mod in net.modules():
        for hook in list(mod._forward_pre_hooks.values()):
            if isinstance(hook, WeightNorm):
                torch.nn.utils.remove_weight_norm(mod, hook.name)
    for mname, mod in net.named_modules():
        for hook in mod._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm):
                raise RuntimeError(f"weight_norm left on {mname}")


# ---------------------------------------------------------------------------------------------
# Streaming margins (measured, not assumed)
# ---------------------------------------------------------------------------------------------


def _span_last_axis(grad):
    """(first, last) index along the last axis where ``grad`` [1, ..., L] is non-zero anywhere."""
    g = grad.abs().reshape(grad.shape[0], -1, grad.shape[-1]).sum(1)[0]
    nz = torch.nonzero(g > 0)
    return int(nz.min()), int(nz.max())


def probe_frames(spec):
    return 256 if spec["vocoder"] == "refinegan" else 96


def measure_margins(net, spec, frames=None):
    """Receptive fields from float64 input gradients around the middle frame ``c``: the flow's left /
    right (frames), the decoder's left (frames, taking the source into account) and right, i.e. how
    many frames *after* a frame its decoded audio depends on (the latents', and the source's: f0, the
    source noise). RefineGAN also gets ``main_margin``: the left receptive field of its main path
    alone (see SigilFlowDecRefineGAN).

    These are structural bounds (any non-zero dependence on random weights). A streaming tick with no
    lookahead returns its last frames without that right context, so they differ from an offline
    decode; how much, for trained weights, is what ``parity_test.py``'s ``edge`` rows measure."""
    frames = frames or probe_frames(spec)
    cfg = configs.config_dict(spec["config"])
    upp = configs.upp(spec["config"])
    flow = copy.deepcopy(net.flow).double()
    dec = copy.deepcopy(net.dec).double()
    gen = torch.Generator().manual_seed(7)
    c = frames // 2
    last = (c + 1) * upp - 1  # the centre frame's last output sample
    g = torch.randn(1, cfg["gin_channels"], 1, generator=gen, dtype=torch.float64)

    z = torch.randn(1, cfg["inter_channels"], frames, generator=gen, dtype=torch.float64, requires_grad=True)
    out = split._flow_reverse(flow, z, g)
    out[:, :, c].sum().backward()
    lo, hi = _span_last_axis(z.grad)
    flow_left, flow_right = c - lo, hi - c

    zd = torch.randn(1, cfg["inter_channels"], frames, generator=gen, dtype=torch.float64, requires_grad=True)
    voc, f0 = spec["vocoder"], spec["f0"]
    extra = {}
    if voc == "refinegan":
        f0v = torch.full((1, frames), 150.0, dtype=torch.float64, requires_grad=True)
        noise = torch.randn(1, frames * upp, generator=gen, dtype=torch.float64, requires_grad=True)
        audio = dec.reference(zd, f0v, g, noise, lambda i, layer, y: torch.zeros_like(y))
        audio[0, 0, c * upp : (c + 1) * upp].sum().backward()
        z_lo, z_hi = _span_last_axis(zd.grad)
        h_lo, h_hi = _span_last_axis(noise.grad.reshape(1, 1, -1))
        _, f_hi = _span_last_axis(f0v.grad.reshape(1, 1, -1))  # linear f0: one more frame than the source
        main = c - z_lo
        dec_left_samples = c * upp - h_lo
        dec_left_frames = main
        dec_margin = max(main, math.ceil(dec_left_samples / upp))
        extra["main_margin"] = main
        dec_right_samples = h_hi - last
        dec_right_frames = max(z_hi - c, math.ceil(dec_right_samples / upp), f_hi - c)
    else:
        if f0:
            har = torch.randn(1, 1, frames * upp, generator=gen, dtype=torch.float64, requires_grad=True)
            if voc == "hifigan":
                audio = split._decoder(dec, zd, har, g)
            else:
                audio = split._decoder_mrf(dec, zd, har, g)
        else:
            har = None
            audio = split._decoder_nof0(dec, zd, g)
        audio[0, 0, c * upp : (c + 1) * upp].sum().backward()
        z_lo, z_hi = _span_last_axis(zd.grad)
        dec_left_frames = c - z_lo
        if har is not None:
            # f0 reaches the decoder only through the source (repeated per frame), so the source's
            # right span is f0's too
            h_lo, h_hi = _span_last_axis(har.grad)
            dec_left_samples = c * upp - h_lo
            dec_right_samples = h_hi - last
        else:
            dec_left_samples = dec_left_frames * upp
            dec_right_samples = (z_hi - c) * upp
        dec_margin = max(dec_left_frames, math.ceil(dec_left_samples / upp))
        dec_right_frames = max(z_hi - c, math.ceil(dec_right_samples / upp))
    if flow_left > split.FLOW_MARGIN or flow_right > split.FLOW_MARGIN:
        raise RuntimeError(f"flow receptive field {flow_left}/{flow_right} exceeds FLOW_MARGIN")
    if dec_margin >= c or dec_right_frames >= frames - c - 1:
        raise RuntimeError("decoder receptive field reaches the probe window edge; raise `frames`")
    out = {
        "flow_margin": split.FLOW_MARGIN,
        "flow_receptive_left": flow_left,
        "flow_receptive_right": flow_right,
        "dec_margin": dec_margin,
        "dec_receptive_left_frames": dec_left_frames,
        "dec_receptive_left_samples": dec_left_samples,
        "dec_receptive_right_frames": dec_right_frames,
        "dec_receptive_right_samples": dec_right_samples,
    }
    out.update(extra)
    return out


# ---------------------------------------------------------------------------------------------
# Graphs
# ---------------------------------------------------------------------------------------------


def flowdec_example(spec, net, f, gen, device, pool=None, per_frame=0, latent_offset=0):
    """Example flowdec inputs for a window of ``f`` frames: (names, tensors, dynamic axes). RefineGAN's
    latents (m_p, logs_p, z_noise) cover its latent window, the last ``f - latent_offset`` frames
    (dynamic dim ``Fz``)."""
    cfg = configs.config_dict(spec["config"])
    upp = configs.upp(spec["config"])
    inter, gin = cfg["inter_channels"], cfg["gin_channels"]

    def randn(*shape):
        return torch.randn(*shape, generator=gen).to(device)

    names = ["m_p", "logs_p", "z_noise", "g"]
    fz, zdim = (f - latent_offset, "Fz") if spec["vocoder"] == "refinegan" else (f, "F")
    args = [randn(1, inter, fz), randn(1, inter, fz) * 0.1, randn(1, inter, fz) * 0.66666, randn(1, gin, 1)]
    axes = {"m_p": {2: zdim}, "logs_p": {2: zdim}, "z_noise": {2: zdim}, "audio": {1: "A"}}
    if spec["f0"]:
        f0 = (torch.rand(1, f, generator=gen) * 300 + 80).to(device)
        if spec["vocoder"] == "hifigan":
            names += ["f0", "phase0", "src_noise"]
            args += [f0, torch.zeros(1, 1, device=device), randn(1, f * upp)]
        else:
            # Applio vocoders: the frame-level phase of every window frame (split.FRAME_PHASE)
            names += ["f0", "phase", "src_noise"]
            axes["phase"] = {1: "F"}
            noise = randn(1, f * upp, net.dec.dim) if spec["vocoder"] == "mrf-hifigan" else randn(1, f * upp)
            args += [f0, torch.rand(1, f, generator=gen).to(device), noise]
        axes.update({"f0": {1: "F"}, "src_noise": {1: "S"}})
    if spec["vocoder"] == "refinegan":
        hi = pool - per_frame + 1
        names.append("noise_offsets")
        args.append(torch.randint(0, hi, (1, f), generator=gen).to(device))
        axes["noise_offsets"] = {1: "F"}
    return names, args, axes


def flowdec_module(spec, net, margins, dtype, device, mixed):
    voc = spec["vocoder"]
    mixed = dict(mixed or {})
    if voc != "refinegan":
        mixed.pop("fp32_ends", None)  # RefineGAN's main-path ends (mel conv, conv_post); see split.py
    if not spec["f0"]:
        return split.SigilFlowDecNoF0(net.flow, net.dec, configs.upp(spec["config"]), margins["dec_margin"], dtype=dtype, **mixed)
    if voc == "hifigan":
        sr = configs.sample_rate(spec["config"])
        return split.SigilFlowDec(net.flow, net.dec, sr, margins["dec_margin"], dtype=dtype, **mixed)
    if voc == "mrf-hifigan":
        return split.SigilFlowDecMRF(net.flow, net.dec, margins["dec_margin"], vocoders.MRF_PHASE_INI, dtype=dtype, **mixed).to(device)
    pool = spec.get("adain_pool", ADAIN_POOL)
    m = split.SigilFlowDecRefineGAN(net.flow, net.dec, margins["dec_margin"], margins["main_margin"], pool, dtype=dtype, **mixed)
    gen = torch.Generator().manual_seed(17)
    m.adain_noise_pool.copy_(torch.randn(pool, generator=gen))
    return m.to(device)


def graph_specs(net, spec, margins, dtype, device, mixed=None):
    """{graph: (module, example args, input names, output names, dynamic axes)}."""
    cfg = configs.config_dict(spec["config"])
    upp = configs.upp(spec["config"])
    emb = spec["emb_channels"]
    inter, gin = cfg["inter_channels"], cfg["gin_channels"]
    dec_margin = margins["dec_margin"]
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

    f = split.FLOW_MARGIN + dec_margin + 16
    if spec["f0"] and spec["vocoder"] == "hifigan":
        # The P1 contract (kept byte for byte: same draws as the P1 exporter).
        f0 = (torch.rand(1, f, generator=gen) * 300 + 80).to(device)
        hifi_mixed = {k: v for k, v in (mixed or {}).items() if k != "fp32_ends"}
        flowdec = split.SigilFlowDec(net.flow, net.dec, configs.sample_rate(spec["config"]), dec_margin, dtype=dtype, **hifi_mixed)
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
    else:
        flowdec = flowdec_module(spec, net, margins, dtype, device, mixed)
        per_frame = getattr(flowdec, "per_frame", 0)
        flowdec_in, flowdec_args, flowdec_axes = flowdec_example(
            spec, net, f, gen, device, spec.get("adain_pool", ADAIN_POOL), per_frame, getattr(flowdec, "latent_offset", 0),
        )

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


def architecture_constants(module):
    """Names of the wrapper's large non-weight constants (RefineGAN's resampler kernels)."""
    return {name for name, _ in module.named_buffers() if name.startswith("resample_kernel_")}


def externalize(model, params, offsets, weights_file, scale, extra=None, constants=()):
    """Point every parameter initializer at ``weights_file``; fail on anything that looks like a
    weight baked into the graph. ``scale`` = bytes per element / 4 (1 for f32, 0.5 for f16).
    ``extra`` = {initializer: (file, offset, length)} for other external data (the AdaIN pool);
    ``constants`` = names of architecture constants allowed to stay inline whatever their size."""
    extra = extra or {}
    seen = set()

    def point(init, location, offset, length):
        init.ClearField("raw_data")
        for field in ("float_data", "int32_data", "int64_data", "double_data"):
            init.ClearField(field)
        init.data_location = TensorProto.EXTERNAL
        del init.external_data[:]
        for key, value in (("location", location), ("offset", int(offset)), ("length", int(length))):
            e = init.external_data.add()
            e.key, e.value = key, str(value)

    for init in model.graph.initializer:
        if init.name in params:
            arr = numpy_helper.to_array(init)
            want = params[init.name]
            if arr.shape != want.shape or arr.dtype != want.dtype or not np.array_equal(
                arr.view(np.uint8), np.ascontiguousarray(want).view(np.uint8)
            ):
                raise RuntimeError(f"initializer {init.name} does not hold its parameter's value")
            off, nbytes = offsets[init.name]
            point(init, weights_file, off * scale, nbytes * scale)
            seen.add(init.name)
        elif init.name in extra:
            point(init, *extra[init.name])
            seen.add(init.name)
        elif init.name not in constants:
            n = int(np.prod(init.dims)) if len(init.dims) else 1
            if n > MAX_INLINE_CONST:
                raise RuntimeError(f"unexpected large constant {init.name} {list(init.dims)} in the graph")
    missing = sorted((set(params) | set(extra)) - seen)
    if missing:
        raise RuntimeError(f"parameters missing from the exported graph: {missing[:8]}")
    return seen


def io_spec(model):
    def one(v):
        tt = v.type.tensor_type
        dims = [d.dim_param if d.dim_param else int(d.dim_value) for d in tt.shape.dim]
        return {"name": v.name, "dtype": onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type).name, "shape": dims}

    return [one(v) for v in model.graph.input], [one(v) for v in model.graph.output]


def check_dynamic(model, module, args, graph, spec=None, net=None):
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
            a = _resize_args(graph, a, spec, net, module)
        with torch.no_grad():
            ref = module(*a)
        ref = ref if isinstance(ref, tuple) else (ref,)
        feeds = {n: x.detach().cpu().numpy() for n, x in zip(names, a)}
        got = sess.run(None, feeds)
        for r, g in zip(ref, got):
            worst = max(worst, float(np.abs(r.detach().cpu().numpy() - g).max()))
    return worst


def _resize_args(graph, args, spec=None, net=None, module=None):
    gen = torch.Generator().manual_seed(11)
    if graph == "enc":
        t = args[0].shape[1] + 37
        out = [torch.randn(1, t, args[0].shape[2], generator=gen)]
        if len(args) > 1:
            out.append(torch.randint(1, 255, (1, t), generator=gen))
        return out
    if graph == "flowdec":
        f = args[0].shape[2] + getattr(module, "latent_offset", 0) + 29  # the source window F (m_p covers Fz)
        if spec is None or (spec["f0"] and spec["vocoder"] == "hifigan"):
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
        _, a, _ = flowdec_example(
            spec, net, f, gen, "cpu", spec.get("adain_pool", ADAIN_POOL), getattr(module, "per_frame", 0), getattr(module, "latent_offset", 0),
        )
        return a
    t, n, e = 23, 300, args[0].shape[1]
    index = torch.randn(n, e, generator=gen)
    return [torch.randn(t, e, generator=gen), index, (index**2).sum(-1), torch.full((1,), 0.75)]


# ---------------------------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------------------------


def excitation_layout(spec, net):
    """The source contract of a template's flowdec (layout.json streaming.excitation)."""
    if not spec["f0"]:
        return {"kind": "none"}
    wrap = "wrap(x) = fmod(x + 0.5, 1) - 0.5"
    if spec["vocoder"] == "hifigan":
        phase = {
            "input": "phase0",
            "shape": [1, 1],
            "carry": f"phase0' = frac(phase0 + sum(wrap(upp * f0 / sr))) over the frames that slide out; {wrap}",
        }
        return {"kind": "nsf-sine", "harmonics": 1, "f0_to_samples": "repeat", "f0_lookahead_frames": 0, "phase": phase,
                "noise": {"input": "src_noise", "shape": [1, "S"], "per": "sample"}}
    phase = {
        "input": "phase",
        "shape": [1, "F"],
        "value": f"phase[j] = {split.FRAME_PHASE}; {wrap} (the same carry as the HiFi-GAN templates' phase0, at every window frame)",
    }
    if spec["vocoder"] == "mrf-hifigan":
        return {
            "kind": "nsf-sine", "harmonics": net.dec.dim, "f0_to_samples": "repeat", "f0_lookahead_frames": 0, "phase": phase,
            "phase_ini": list(vocoders.MRF_PHASE_INI),
            "noise": {"input": "src_noise", "shape": [1, "S", net.dec.dim], "per": "sample and harmonic"},
        }
    return {
        "kind": "nsf-sine", "harmonics": 1, "f0_to_samples": "linear", "f0_lookahead_frames": 1, "phase": phase,
        "linear_note": (
            "Applio's half-pixel linear f0 interpolation is computed in the graph in closed form; window frame "
            "flow_margin - 1 is its left context. Its second half-frame reads the next frame's f0, so the window's "
            "last frame is edge-clamped: with no lookahead, the phase at the end of a tick's last frame differs from "
            "the next tick's phase at that boundary by upp / (8 sr) * (f0[j+1] - f0[j]) cycles (up to ~0.2 cycle at a "
            "voicing offset). Decoding f0_lookahead_frames more frames than are returned (drop them, or delay the "
            "output by that much) keeps the excitation phase-continuous across ticks"
        ),
        "noise": {"input": "src_noise", "shape": [1, "S"], "per": "sample"},
    }


def adain_layout(flowdec):
    layers = [{"stage": s, "layer": l, "channels": c, "rate": r, "base": b} for s, l, c, r, b in flowdec.layers]
    return {
        "input": "noise_offsets",
        "offsets": "noise_offsets[0, j] = any fixed function of window frame j's absolute frame index, in [0, pool.count - per_frame]; equal inputs for equal absolute frames make overlapping ticks identical",
        "pool": {"initializer": "adain_noise_pool", "file": ADAIN_NOISE_FILE, "count": flowdec.pool_size, "dtype": "float32",
                 "fill": "iid standard normal values, generated once by the engine (any generator)"},
        "per_frame": flowdec.per_frame,
        "element": "layer (stage, layer), channel c, frame j, sample s -> pool[noise_offsets[j] + base + c * rate + s]",
        "layers": layers,
    }


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
    if verbose:
        print(f"[{template_id}] margins {margins}")

    fp16_device = "cuda" if torch.cuda.is_available() else "cpu"
    net16 = split.cast_weights_to_input_(copy.deepcopy(net).half().to(fp16_device))
    specs = {
        "fp32": graph_specs(net, spec, margins, torch.float32, "cpu"),
        "fp16": graph_specs(net16, spec, margins, torch.float16, fp16_device, FLOWDEC_FP16),
    }

    # One weights file for every graph: enc tensors first, then flowdec, in state-dict order.
    tensors, offsets, cursor = [], {}, 0
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
            cursor = sl.align_up(cursor + nbytes)
    weights_id = sl.tensor_table_id(tensors)

    flowdec32 = specs["fp32"]["flowdec"][0]
    is_refinegan = spec["vocoder"] == "refinegan"
    extra = {"adain_noise_pool": (ADAIN_NOISE_FILE, 0, flowdec32.pool_size * 4)} if is_refinegan else {}
    parity_table, recommended = measured_parity(template_id)
    if verbose and parity_table is not PARITY:
        print(f"[{template_id}] parity not measured here: the layout says untested and recommends fp32 "
              f"(export_matrix.py --template {template_id} measures and rewrites it)")
    graphs = {}
    for graph in GRAPHS:
        graphs[graph] = {"recommended": recommended[graph], "variants": {}}
        for variant, dtype in VARIANTS.items():
            module, args, in_names, out_names, axes = specs[variant][graph]
            model = export_onnx(module, args, in_names, out_names, axes)
            if variant == "fp32":
                worst = check_dynamic(model, module, args, graph, spec, net)
                if verbose:
                    print(f"[{template_id}] {graph}.fp32 ORT vs torch (2 lengths): max abs {worst:.3g}")
                if worst > 1e-4:
                    raise RuntimeError(f"{graph}: exported graph disagrees with PyTorch ({worst})")
            onnx.checker.check_model(model)  # structure, with the weights still inline
            params = module_params(module)
            if graph != "knn":
                wfile = sl.WEIGHTS_F32 if variant == "fp32" else sl.WEIGHTS_F16
                externalize(
                    model, params, offsets, wfile, 1.0 if variant == "fp32" else 0.5,
                    extra=extra if graph == "flowdec" else None, constants=architecture_constants(module),
                )
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
            entry = {
                "file": fname,
                "sha256": sl.sha256_file(path),
                "bytes": os.path.getsize(path),
                "weights_file": (sl.WEIGHTS_F32 if variant == "fp32" else sl.WEIGHTS_F16) if graph != "knn" else None,
                "parity": parity_table[graph][variant],
                "inputs": inputs,
                "outputs": outputs,
            }
            if graph == "flowdec" and extra:
                entry["extra_external_files"] = [ADAIN_NOISE_FILE]
            graphs[graph]["variants"][variant] = entry
            if verbose:
                print(f"[{template_id}] wrote {fname} ({os.path.getsize(path)} bytes)")

    cfg = configs.config_dict(spec["config"])
    sr = configs.sample_rate(spec["config"])
    upp = configs.upp(spec["config"])
    new_contract = spec["vocoder"] not in LAYOUT_V1_VOCODERS or not spec["f0"]
    streaming = {
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
    }
    streaming.setdefault("main_margin", margins["dec_margin"])
    # Frames the latent inputs (m_p, logs_p, z_noise) start after the source window's first frame:
    # they cover Fz = F - latent_offset frames (RefineGAN: its main path's shorter margin; 0 elsewhere).
    streaming["latent_offset"] = streaming["dec_margin"] - streaming["main_margin"] if is_refinegan else 0
    if is_refinegan:
        streaming["dims"]["flowdec"] = (
            "F = source window (f0, phase, noise_offsets; S = F * upp); Fz = F - latent_offset = flow_margin + "
            "main_margin + R = latent window (m_p, logs_p, z_noise: the text encoder's output is needed over these "
            "frames only); A = (F - flow_margin - dec_margin) * upp; f0 frames before flow_margin - 1, phase / "
            "src_noise frames before flow_margin and noise_offsets before flow_margin + latent_offset are not read"
        )
    streaming["excitation"] = excitation_layout(spec, net)
    if is_refinegan:
        streaming["adain"] = adain_layout(flowdec32)
    layout = {
        "format": sl.LAYOUT_FORMAT,
        "format_version": 2 if new_contract else sl.LAYOUT_VERSION,
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
        "streaming": streaming,
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
    if new_contract:
        layout["features"] = sorted(
            {"flowdec-inputs-by-layout"}
            | ({"nof0"} if not spec["f0"] else set())
            | ({"nsf-harmonics"} if spec["vocoder"] == "mrf-hifigan" else set())
            | ({"adain-noise-pool", "linear-f0"} if is_refinegan else set())
        )
    if spec["vocoder"] != "hifigan":
        layout["exporter"]["applio_commit"] = APPLIO_COMMIT
        layout["exporter"]["decoder"] = "Sigil port of Applio's generator (scripts/vc-templates/rvc_arch/vocoders.py)"
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
    ap.add_argument("--fp32-ends", action="store_true", help="experiment: RefineGAN's mel conv / cond and conv_post in fp32")
    args = ap.parse_args()
    if args.fp32_flow:
        FLOWDEC_FP16["fp32_flow"] = True
    if args.fp32_stages is not None:
        FLOWDEC_FP16["fp32_stages"] = tuple(int(x) for x in args.fp32_stages.split(",") if x.strip())
    if args.fp32_ends:
        FLOWDEC_FP16["fp32_ends"] = True
    bundled = os.path.normcase(os.path.abspath(args.out)) == os.path.normcase(os.path.abspath(DEFAULT_OUT))
    for tid in args.template or [P1_TEMPLATE]:
        if configs.TEMPLATES[tid].get("test_only") and bundled:
            raise SystemExit(f"{tid} is a test-only configuration; export it with --out <fixture dir>")
        if bundled and tid != P1_TEMPLATE:
            # Its bundled layout carries measured parity and the recommendation derived from it; a bare
            # export would replace them with "untested" (export_matrix measures and finalises it).
            raise SystemExit(
                f"{tid}: bundled templates other than {P1_TEMPLATE} are exported by "
                f"`export_matrix.py --template {tid} --dml-python ...` (it measures parity); use --out <dir> for a scratch export"
            )
        export_template(tid, args.out)


if __name__ == "__main__":
    main()
