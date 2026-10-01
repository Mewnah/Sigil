#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DDSP-SVC 6.3 template graphs for Sigil's importer and engine (P5-A, ADR 0006 "P5-A").

DDSP-SVC voices are trained per user, so Sigil cannot ship their graphs. Like the RVC templates
(ADR 0004), it ships the 6.3 architecture's graphs *without weights* plus a ``layout.json`` saying
where each weight lives in a voice's ``weights.f32.bin`` and which checkpoint tensor it comes from.
The importer (Rust, ``vc::families::ddsp_svc``) reads a user's ``model_*.pt`` + ``config.yaml``
with the allow-list unpickler in the isolated worker and writes ``weights.f32.bin``; the engine
hands each graph its section of that image as in-memory external data.

Graphs (opset 17, every weight external, every other constant inside):

  ddsp         hubert f32 [1,U,C], mel2ph i64 [1,T], f0 f32 [1,T] Hz, volume f32 [1,T],
               randn f32 [1,hop*T], spk f32 [1,D], formant_shift f32 [1,1], start_frame i64 [1]
               -> x f32 [1,1,M,R] (normalised DDSP mel), cond f32 [1,M,R] (DDSP log-mel), R = T - start_frame
               ``export.SigilDdsp`` (ADR 0006: upstream's inference path, the vocoder's own mel, a
               NaN-free sinc) plus two changes:
               * the inverse STFT is a MatMul with the iSTFT kernel and an overlap-add of shifted
                 frames (Pad + Add) instead of ``ConvTranspose1d``: DirectML runs the transposed
                 convolution (2050 -> 1 channels, kernel 2048, stride 512) in about 1.8 s per call
                 (the whole "ddsp takes 2-4.6 s on DirectML" of ADR 0006), the MatMul form in
                 0.5 ms, equal to 4.5e-7;
               * ``spk``: the speaker vector added after the volume embedding (upstream adds
                 ``spk_embed(spk_id - 1)`` there), so single- and multi-speaker voices share the
                 graph (zeros for a single speaker; Rust looks the row up, and can blend rows);
               ``formant_shift`` is upstream's ``aug_shift`` (semitones), always wired: a voice
               trained without pitch augmentation gets a zero embedding weight.
  reflow_step  x, t, dt f32 [1], cond -> x + dt * v(x, 1000 t, cond)     (one Euler step, LYNXNet2)

Weights: the graphs reference their weights as external data in ``ddsp.weights`` and
``reflow_step.weights``; the two are consecutive sections of ``weights.f32.bin`` (64-byte aligned
tensors). ``layout.json`` maps every external initializer to its source in the training checkpoint
(``ddsp_model.*`` / ``reflow_model.*``): a key, or a weight_norm pair folded as torch does, with a
transform (identity, transpose for Linear weights that constant folding transposed into MatMul,
reshape for PReLU slopes). The mapping is found by value, not by name: the graphs are exported twice
with different random weights, the initializers that differ are the weights, and each must equal
exactly one checkpoint tensor under one transform.

Outputs:
  default      <dev-cache>/vc-assets/vc/ddsp-svc-6.3/{ddsp.onnx, reflow_step.onnx, layout.json}
               (the ``vc/ddsp-svc-6.3`` model-store entry, installed from SIGIL_LOCAL_ASSETS) and
               the engine's embedded copy src-tauri/src/services/neural_audio/vc/families/ddsp_svc/layout.json
  --vocoder    also stage <dev-cache>/vc-assets/vc/vocoder-pc-nsf-hifigan-44k/{model.onnx, config.json}
               from export.py's ``vocoder.onnx`` and the openvpi config
  --voice DIR  a synthetic DDSP-SVC 6.3 voice of the default architecture (seeded random weights;
               no public 6.3 checkpoint exists): DIR/model_0.pt + DIR/config.yaml
  --tiny       the CI fixture src-tauri/test-fixtures/vc/ddsp_svc/: a tiny architecture's template,
               a training checkpoint of it (model_100.pt + config.yaml) and one reference tick
               (reference.json + reference.bin) through the template graphs on ONNX Runtime's CPU EP

Usage (repo root): ``python scripts/vc-families/ddsp_svc/export_templates.py [--vocoder] [--voice DIR] [--tiny]``
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import common as cm  # noqa: E402
import export as ex  # noqa: E402  (installs the upstream import stubs and path)

REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
TEMPLATE_ID = "ddsp-svc-6.3"
LAYOUT_FORMAT = "sigil-ddsp-layout"
LAYOUT_VERSION = 1
ALIGN = 64
ASSETS = os.path.join(cm.DEV_CACHE, "vc-assets")
TEMPLATE_OUT = os.path.join(ASSETS, "vc", TEMPLATE_ID)
VOCODER_OUT = os.path.join(ASSETS, "vc", "vocoder-pc-nsf-hifigan-44k")
ENGINE_LAYOUT = os.path.join(REPO, "src-tauri", "src", "services", "neural_audio", "vc", "families", "ddsp_svc", "layout.json")
TINY_OUT = os.path.join(REPO, "src-tauri", "test-fixtures", "vc", "ddsp_svc")
SEED_A, SEED_B = 20251, 20252
DDSP_INPUTS = ["hubert", "mel2ph", "f0", "volume", "randn", "spk", "formant_shift", "start_frame"]
DDSP_OUTPUTS = ["x", "cond"]
FLOW_INPUTS = ["x", "t", "dt", "cond"]
FLOW_OUTPUTS = ["x_next"]
SPEAKERS_KEY = "ddsp_model.unit2ctrl.spk_embed.weight"
AUG_KEY = "ddsp_model.unit2ctrl.aug_shift_embed.weight"
# Checkpoint entries the graphs do not read: the DDSP's constant buffers and two embeddings
# upstream defines but never calls (Unit2Control.f0_embed / phase_embed).
IGNORED_KEYS = ["ddsp_model.sampling_rate", "ddsp_model.block_size", "ddsp_model.win_length", "ddsp_model.window",
                "ddsp_model.unit2ctrl.f0_embed.weight", "ddsp_model.unit2ctrl.f0_embed.bias",
                "ddsp_model.unit2ctrl.phase_embed.weight", "ddsp_model.unit2ctrl.phase_embed.bias"]


class DotDict(dict):
    def __getattr__(self, k):
        v = self.get(k)
        return DotDict(v) if isinstance(v, dict) else v


# ---------------------------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------------------------

def default_arch():
    """configs/reflow.yaml (6.3) and the openvpi pc_nsf_hifigan_44.1k_hop512_128bin_2025.02 config."""
    from nsf_hifigan.models import load_config
    args = DotDict(ex.load_config())
    h = load_config(cm.need(ex.VOC_CKPT, "openvpi pc_nsf_hifigan_44.1k_hop512_128bin_2025.02 (export.py)"))
    return args, h


def tiny_arch():
    """A tiny architecture of the same code paths (the CI fixture): 8-channel units, hop 32,
    window 64, 8 mel bins, one conformer and one LYNXNet2 layer of 8 channels."""
    args, _ = default_arch()
    args = copy.deepcopy(dict(args))
    args["data"].update({"block_size": 32, "encoder_out_channels": 8})
    args["model"].update({"win_length": 64, "n_aux_layers": 1, "n_aux_chans": 8, "n_layers": 1, "n_chans": 8, "use_pitch_aug": True})
    from nsf_hifigan.env import AttrDict
    h = AttrDict({"sampling_rate": 44100, "num_mels": 8, "hop_size": 32, "n_fft": 64, "win_size": 64, "fmin": 40, "fmax": 16000})
    return DotDict(args), h


def arch_record(args, h):
    a = args
    return {"sampling_rate": int(a.data.sampling_rate), "block_size": int(a.data.block_size), "win_length": int(a.model.win_length),
            "encoder_out_channels": int(a.data.encoder_out_channels), "n_aux_layers": int(a.model.n_aux_layers),
            "n_aux_chans": int(a.model.n_aux_chans), "n_layers": int(a.model.n_layers), "n_chans": int(a.model.n_chans),
            "use_attention": bool(a.model.use_attention), "out_dims": int(h.num_mels)}


def vocoder_record(h):
    return {k: (float(h[k]) if k in ("fmin", "fmax") else int(h[k])) for k in ("sampling_rate", "hop_size", "num_mels", "n_fft", "win_size", "fmin", "fmax")}


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------

def training_model(args, out_dims, seed):
    """Upstream's training ``Unit2Wav`` (weight_norm modules as constructed, the state dict a
    training checkpoint holds) with every parameter seeded and perturbed, so two seeds differ in
    every weight (LayerNorm and PReLU start at constants) and LYNXNet2's zero-initialised output
    projection is not zero."""
    from reflow.vocoder import Unit2Wav
    torch.manual_seed(seed)
    a = args
    m = Unit2Wav(a.data.sampling_rate, a.data.block_size, a.model.win_length, a.data.encoder_out_channels, a.model.n_spk,
                 a.model.use_norm, a.model.use_attention, a.model.use_pitch_aug, out_dims, a.model.n_aux_layers,
                 a.model.n_aux_chans, a.model.n_layers, a.model.n_chans)
    g = torch.Generator().manual_seed(seed + 7)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("output_projection.weight") or name.endswith("output_projection.bias"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
            else:
                p.add_(torch.randn(p.shape, generator=g) * 0.01)
    return m.eval()


def export_copy(train, args, out_dims):
    """Upstream's exporter ``Unit2Wav`` carrying the training model's weights (strict)."""
    import export_onnx as upx
    a = args
    m = upx.Unit2Wav(a.data.sampling_rate, a.data.block_size, a.model.win_length, a.data.encoder_out_channels, a.model.n_spk,
                     a.model.use_norm, a.model.use_attention, a.model.use_pitch_aug, out_dims, a.model.n_aux_layers,
                     a.model.n_aux_chans, a.model.n_layers, a.model.n_chans, a.data.f0_min)
    m.load_state_dict(train.state_dict(), strict=True)
    return cm.freeze(m.eval())


class OlaISTFT(nn.Module):
    """Upstream ``export_onnx.iSTFT`` ('continue' framing, fft_len = win_len) as a MatMul with its
    iSTFT kernel and an overlap-add of shifted frames, which DirectML runs in a fraction of a
    millisecond (see the module docs). Same values as the ConvTranspose1d form."""

    def __init__(self, istft):
        super().__init__()
        n, hop = int(istft.fft_len), int(istft.win_hop)
        if n % hop:
            raise SystemExit(f"OlaISTFT needs win_length ({n}) to be a multiple of block_size ({hop})")
        self.register_buffer("kt", istft.ifft_k[:, 0, :].clone())          # [2 nb, n]
        self.register_buffer("w2", istft.padded_window.clone().float())    # [n] window^2
        self.n, self.hop, self.r = n, hop, n // hop

    def ola(self, frames):
        """[B, F, n] frames -> [B, (F + r - 1) hop] overlap-added samples."""
        b, f = frames.shape[0], frames.shape[1]
        parts = frames.reshape(b, f, self.r, self.hop)
        acc = None
        for j in range(self.r):
            p = F.pad(parts[:, :, j, :], (0, 0, j, self.r - 1 - j))
            acc = p if acc is None else acc + p
        return acc.reshape(b, -1)

    def forward(self, spec):
        inputs = torch.cat([spec[..., 0], spec[..., 1]], dim=1)            # [B, 2 nb, F]
        frames = torch.matmul(inputs.transpose(1, 2), self.kt)            # [B, F, n]
        out = self.ola(frames)
        coff = self.ola(self.w2[None, None, :].expand(inputs.shape[0], inputs.shape[2], self.n))
        pad = self.n // 2
        out = out[..., pad:-pad]
        coff = coff[..., pad:-pad]
        coff = torch.where(coff > 1e-8, coff, torch.ones_like(coff))
        return out / coff


class TemplateDdsp(ex.SigilDdsp):
    """``export.SigilDdsp`` with the overlap-add iSTFT and the ``spk`` input (module docs)."""

    def __init__(self, ddsp, h):
        super().__init__(ddsp, h)
        if ddsp.unit2ctrl.aug_shift_embed is None:
            raise SystemExit("templates are exported with use_pitch_aug = true (a voice without it gets a zero weight)")
        self.istft = OlaISTFT(ddsp.istft_method)

    def forward(self, hubert, mel2ph, f0, volume, randn, spk, formant_shift, start_frame):
        m = self.m
        units = torch.gather(hubert, 1, mel2ph.unsqueeze(2).repeat([1, 1, hubert.shape[-1]]))
        volume = volume.unsqueeze(-1)
        f0_frames = f0.unsqueeze(-1)
        n = torch.arange(self.bs, device=f0.device)
        s0 = f0_frames / m.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / m.block_size
        s0 = s0 + ds0 * n / m.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad - torch.round(rad)
        combtooth = ex._safe_sinc(rad / (s0 + 1e-5)).reshape(f0.shape[0], -1)
        combtooth_frames = combtooth.view(1, -1, self.bs)
        noise_frames = randn.view(1, -1, self.bs)
        u = m.unit2ctrl
        exciter = torch.cat((combtooth_frames, noise_frames), dim=-1).transpose(1, 2)
        x = u.stack(units.transpose(1, 2)) + u.stack2(exciter)
        x = x.transpose(1, 2) + u.volume_embed(volume)
        x = x + spk.unsqueeze(1)
        x = x + u.aug_shift_embed(formant_shift / 5)
        e = u.dense_out(u.norm(u.decoder(x)))
        hm, hp, nm, npase = torch.split(e, u.o_sp, dim=-1)
        src = m.complex_exp(hm.unsqueeze(-1), torch.ones_like(hm).unsqueeze(-1) * np.pi * hp.unsqueeze(-1))
        src = torch.cat((src, src[:, -1:, :]), 1)
        nf = m.complex_exp(nm.unsqueeze(-1), torch.ones_like(nm).unsqueeze(-1) * np.pi * npase.unsqueeze(-1)) / 128
        nf = torch.cat((nf, nf[:, -1:, :]), 1)
        sig = m.complex_mul(self._stft(combtooth), src.permute(0, 2, 1, 3)) + m.complex_mul(self._stft(randn), nf.permute(0, 2, 1, 3))
        signal = self.istft(sig)
        signal = signal[:, start_frame[0] * self.bs:]
        mel = self.mel(signal, self.basis)
        return m.norm_spec(mel).unsqueeze(1), mel


def template_modules(train, args, h):
    out_dims = int(h.num_mels)
    ev = export_copy(train, args, out_dims)
    ddsp = TemplateDdsp(cm.strip_weight_norm(copy.deepcopy(ev.ddsp_model)), h).eval()
    flow = ex.ReflowStep(copy.deepcopy(train.reflow_model.velocity_fn)).eval()
    return ddsp, flow


def example_inputs(args, h, frames=24, start=10, units=30):
    torch.manual_seed(0)
    c, d, bs = int(args.data.encoder_out_channels), int(args.model.n_aux_chans), int(args.data.block_size)
    m = int(h.num_mels)
    r = frames - start
    ddsp = (torch.randn(1, units, c), torch.clamp(torch.round(torch.arange(frames) * (units / frames)), max=units - 1).long()[None],
            torch.full((1, frames), 220.0), torch.rand(1, frames) * 0.1, torch.randn(1, frames * bs), torch.zeros(1, d),
            torch.tensor([[0.5]]), torch.tensor([start]))
    flow = (torch.randn(1, 1, m, r), torch.tensor([0.7]), torch.tensor([0.015]), torch.randn(1, m, r) - 5)
    return ddsp, flow


def export_pair(train, args, h, folder):
    """Export ddsp.onnx and reflow_step.onnx (weights inside) of ``train`` into ``folder``."""
    ddsp, flow = template_modules(train, args, h)
    d_in, f_in = example_inputs(args, h)
    paths = {"ddsp": os.path.join(folder, "ddsp.onnx"), "reflow_step": os.path.join(folder, "reflow_step.onnx")}
    with torch.no_grad():
        torch.onnx.export(ddsp, d_in, paths["ddsp"], input_names=DDSP_INPUTS, output_names=DDSP_OUTPUTS, opset_version=17,
                          do_constant_folding=True,
                          dynamic_axes={"hubert": {1: "U"}, "mel2ph": {1: "T"}, "f0": {1: "T"}, "volume": {1: "T"}, "randn": {1: "S"},
                                        "x": {3: "R"}, "cond": {2: "R"}})
        torch.onnx.export(flow, f_in, paths["reflow_step"], input_names=FLOW_INPUTS, output_names=FLOW_OUTPUTS, opset_version=17,
                          do_constant_folding=True, dynamic_axes={"x": {3: "R"}, "cond": {2: "R"}, "x_next": {3: "R"}})
    return paths


# ---------------------------------------------------------------------------------------------
# Layout: which initializer is which checkpoint tensor
# ---------------------------------------------------------------------------------------------

def checkpoint_candidates(train):
    """(source, array) for every tensor a graph can hold: each state-dict entry, and each
    weight_norm pair folded as torch folds it (``torch._weight_norm(v, g, 0)``)."""
    sd = {k: v.detach().cpu() for k, v in train.state_dict().items()}
    out = []
    for k, v in sd.items():
        if k.endswith(".weight_g") or k.endswith(".weight_v"):
            continue
        if v.dtype.is_floating_point:
            out.append(({"key": k}, v.numpy().astype(np.float32)))
    for k in sd:
        if k.endswith(".weight_g"):
            base = k[: -len(".weight_g")]
            g, v = sd[k], sd[base + ".weight_v"]
            w = torch._weight_norm(v, g, 0)
            out.append(({"weight_norm": {"g": k, "v": base + ".weight_v", "dim": 0}}, w.numpy().astype(np.float32)))
    return out


def initializers(path):
    import onnx
    from onnx import numpy_helper
    m = onnx.load(path)
    inits = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    consts = {}
    for n in m.graph.node:
        if n.op_type == "Constant":
            for a in n.attribute:
                if a.name == "value":
                    consts[n.output[0]] = numpy_helper.to_array(a.t)
    return m, inits, consts


def source_key(src):
    return src.get("key") or src["weight_norm"]["g"][: -len("_g")]


def match(arr, cands):
    """The single (source, transform, source_shape) whose tensor equals ``arr`` exactly."""
    hits = []
    for src, t in cands:
        if t.shape == arr.shape and np.array_equal(t, arr):
            hits.append((src, "identity", list(t.shape)))
        elif t.ndim == 2 and t.T.shape == arr.shape and np.array_equal(t.T, arr):
            hits.append((src, "transpose", list(t.shape)))
        elif t.size == arr.size and t.shape != arr.shape and np.array_equal(t.reshape(-1), arr.reshape(-1)):
            hits.append((src, "reshape", list(t.shape)))
    return hits


def build_layout(args, h, train_a, files_a, files_b, template_id):
    """Map every weight initializer of the seed-A export to its checkpoint source (module docs)."""
    cands = checkpoint_candidates(train_a)
    tensors, sections, offset = [], {}, 0
    used = set()
    for graph in ("ddsp", "reflow_step"):
        _, ia, ca = initializers(files_a[graph])
        _, ib, cb = initializers(files_b[graph])
        if set(ia) != set(ib) or set(ca) != set(cb):
            raise SystemExit(f"{graph}: the two exports have different initializers / constants")
        for name in ca:
            if ca[name].shape != cb[name].shape or not np.array_equal(ca[name], cb[name]):
                raise SystemExit(f"{graph}: Constant node {name} depends on the weights; it would not be external")
        start = offset
        for name in ia:  # graph order
            a, b = ia[name], ib[name]
            if a.shape == b.shape and np.array_equal(a, b):
                continue  # a constant (STFT bases, mel basis, shape tensors): stays in the graph
            if a.dtype != np.float32:
                raise SystemExit(f"{graph}: weight {name} is {a.dtype}, not float32")
            hits = match(a, cands)
            if len(hits) != 1:
                raise SystemExit(f"{graph}: weight {name} {list(a.shape)} matches {len(hits)} checkpoint tensors")
            src, transform, src_shape = hits[0]
            key = source_key(src)
            if key in used:
                raise SystemExit(f"{graph}: {key} feeds two initializers")
            used.add(key)
            offset = (offset + ALIGN - 1) // ALIGN * ALIGN
            nbytes = int(a.size) * 4
            tensors.append({"name": name, "graph": graph, "shape": [int(s) for s in a.shape], "offset": offset - start,
                            "bytes": nbytes, "source": src, "transform": transform, "source_shape": src_shape})
            offset += nbytes
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN
        sections[graph] = {"offset": start, "bytes": offset - start, "location": f"{graph}.weights"}
    read = set()
    for t in tensors:
        wn = t["source"].get("weight_norm")
        read.update([wn["g"], wn["v"]] if wn else [t["source"]["key"]])
    unused = sorted(set(train_a.state_dict().keys()) - read)
    unexpected = [k for k in unused if k not in IGNORED_KEYS and k != SPEAKERS_KEY]
    if unexpected:
        raise SystemExit(f"checkpoint tensors no graph reads: {unexpected}")
    if not any(source_key(t["source"]) == AUG_KEY for t in tensors):
        raise SystemExit(f"{AUG_KEY} is not in the ddsp graph")
    table = {"tensors": tensors, "sections": sections, "arch": arch_record(args, h), "vocoder": vocoder_record(h)}
    weights_id = hashlib.sha256(json.dumps(table, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {
        "format": LAYOUT_FORMAT,
        "format_version": LAYOUT_VERSION,
        "template": template_id,
        "source": "yxlllc/DDSP-SVC@d6dd52fc (6.3) architecture; Sigil export (scripts/vc-families/ddsp_svc/export_templates.py)",
        "arch": arch_record(args, h),
        "vocoder": vocoder_record(h),
        "spec_min": -12.0,
        "spec_max": 2.0,
        "graphs": {
            "ddsp": {"file": "ddsp.onnx", "inputs": DDSP_INPUTS, "outputs": DDSP_OUTPUTS},
            "reflow_step": {"file": "reflow_step.onnx", "inputs": FLOW_INPUTS, "outputs": FLOW_OUTPUTS},
        },
        "weights": {"file": "weights.f32.bin", "alignment": ALIGN, "bytes": offset, "weights_id": weights_id,
                    "sections": sections, "tensors": tensors},
        "optional_zero": [AUG_KEY],
        "speakers": {"key": SPEAKERS_KEY, "dim": int(args.model.n_aux_chans)},
        "ignored_keys": IGNORED_KEYS,
    }


def weights_image(layout, train):
    """``weights.f32.bin`` of ``train`` by the layout (what the Rust importer writes)."""
    cands = {json.dumps(src, sort_keys=True): arr for src, arr in checkpoint_candidates(train)}
    img = bytearray(layout["weights"]["bytes"])
    for t in layout["tensors"] if "tensors" in layout else layout["weights"]["tensors"]:
        arr = cands[json.dumps(t["source"], sort_keys=True)]
        if t["transform"] == "transpose":
            arr = arr.T
        arr = np.ascontiguousarray(arr.reshape(t["shape"]), np.float32)
        at = layout["weights"]["sections"][t["graph"]]["offset"] + t["offset"]
        img[at: at + t["bytes"]] = arr.tobytes()
    return bytes(img)


def strip_weights(layout, files_a, out_dir):
    """Write the graph-only templates: every layout tensor becomes external data in its section."""
    import onnx
    from onnx.external_data_helper import set_external_data
    by_graph = {}
    for t in layout["weights"]["tensors"]:
        by_graph.setdefault(t["graph"], {})[t["name"]] = t
    written = {}
    for graph, path in files_a.items():
        m = onnx.load(path)
        loc = layout["weights"]["sections"][graph]["location"]
        seen = 0
        for init in m.graph.initializer:
            t = by_graph.get(graph, {}).get(init.name)
            if t is None:
                continue
            set_external_data(init, location=loc, offset=t["offset"], length=t["bytes"])
            init.data_location = onnx.TensorProto.EXTERNAL
            init.ClearField("raw_data")
            init.ClearField("float_data")
            seen += 1
        if seen != len(by_graph.get(graph, {})):
            raise SystemExit(f"{graph}: {seen} of {len(by_graph[graph])} weights found")
        del m.metadata_props[:]
        m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.source", value=layout["source"]))
        m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.graph", value=f"ddsp_svc/{graph} (template, external weights)"))
        m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.weights_id", value=layout["weights"]["weights_id"]))
        dst = os.path.join(out_dir, os.path.basename(path))
        onnx.save(m, dst)
        written[graph] = dst
    return written


def section_files(layout, image, folder):
    for g, s in layout["weights"]["sections"].items():
        with open(os.path.join(folder, s["location"]), "wb") as f:
            f.write(image[s["offset"]: s["offset"] + s["bytes"]])


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------

def ort_session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.inter_op_num_threads = 1
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def numpy_feeds(names, tensors):
    return {n: t.numpy() for n, t in zip(names, tensors)}


def verify_templates(layout, templates, train, embedded, args, h):
    """The graph-only templates fed the image of ``train`` reproduce the embedded export, and the
    torch modules (the ONNX ConvTranspose-free iSTFT against upstream's) agree."""
    image = weights_image(layout, train)
    d_in, f_in = example_inputs(args, h, frames=40, start=18, units=50)
    with tempfile.TemporaryDirectory() as tmp:
        for g, p in templates.items():
            shutil.copy(p, tmp)
        section_files(layout, image, tmp)
        for graph, names, feeds in (("ddsp", DDSP_INPUTS, d_in), ("reflow_step", FLOW_INPUTS, f_in)):
            got = ort_session(os.path.join(tmp, os.path.basename(templates[graph]))).run(None, numpy_feeds(names, feeds))
            ref = ort_session(embedded[graph]).run(None, numpy_feeds(names, feeds))
            for o, r in zip(got, ref):
                if not np.array_equal(o, r):
                    raise SystemExit(f"{graph}: template + weights image != embedded export (max |d| {np.abs(o - r).max():.3g})")
    # The overlap-add iSTFT against upstream's ConvTranspose1d iSTFT, through the whole DDSP.
    ev = export_copy(train, args, int(h.num_mels))
    plain = ex.SigilDdsp(cm.strip_weight_norm(copy.deepcopy(ev.ddsp_model)), h).eval()
    tmpl, _ = template_modules(train, args, h)
    with torch.no_grad():
        a = plain(*d_in[:5], d_in[7], d_in[6])          # SigilDdsp: (..., randn, start_frame, formant_shift), spk = 0
        b = tmpl(*d_in)
    for o, r in zip(b, a):
        row = cm.compare(r.numpy(), o.numpy())
        if not (row["snr_db"] == "inf" or row["snr_db"] >= 100):
            raise SystemExit(f"overlap-add iSTFT differs from upstream's: {row}")
    print(f"   templates verified: external weights bit-exact, overlap-add iSTFT vs ConvTranspose {row['snr_db']} dB")


# ---------------------------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------------------------

def make_templates(args, h, out_dir, template_id):
    os.makedirs(out_dir, exist_ok=True)
    out_dims = int(h.num_mels)
    train_a = training_model(args, out_dims, SEED_A)
    train_b = training_model(args, out_dims, SEED_B)
    with tempfile.TemporaryDirectory() as ta, tempfile.TemporaryDirectory() as tb:
        files_a = export_pair(train_a, args, h, ta)
        files_b = export_pair(train_b, args, h, tb)
        layout = build_layout(args, h, train_a, files_a, files_b, template_id)
        templates = strip_weights(layout, files_a, out_dir)
        verify_templates(layout, templates, train_a, files_a, args, h)
    for g, p in templates.items():
        layout["graphs"][g]["bytes"] = os.path.getsize(p)
        layout["graphs"][g]["sha256"] = cm.graph_sha256(p)
    path = os.path.join(out_dir, "layout.json")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(layout, f, indent=1)
        f.write("\n")
    n = len(layout["weights"]["tensors"])
    print(f"wrote {out_dir}: {n} weights, {layout['weights']['bytes'] / 1e6:.1f} MB image, "
          + ", ".join(f"{g} {layout['graphs'][g]['bytes'] / 1e6:.1f} MB" for g in templates))
    return layout, templates


def write_config_yaml(args, path):
    import yaml
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(json.loads(json.dumps(args)), f, sort_keys=False)


def save_checkpoint(train, path, step):
    """What upstream's trainer saves (``logger/saver.py``, ``save_opt: false``)."""
    torch.save({"global_step": step, "model": train.state_dict()}, path)


def make_voice(folder):
    args, h = default_arch()
    os.makedirs(folder, exist_ok=True)
    train = training_model(args, int(h.num_mels), 4242)
    save_checkpoint(train, os.path.join(folder, "model_0.pt"), 0)
    shutil.copy(ex.CONFIG, os.path.join(folder, "config.yaml"))
    print(f"wrote a synthetic DDSP-SVC 6.3 voice to {folder} (seeded random weights)")


def make_tiny(out_dir):
    """The CI fixture (module docs)."""
    args, h = tiny_arch()
    layout, templates = make_templates(args, h, out_dir, "ddsp-svc-6.3-tiny")
    train = training_model(args, int(h.num_mels), 100)
    save_checkpoint(train, os.path.join(out_dir, "model_100.pt"), 100)
    write_config_yaml(args, os.path.join(out_dir, "config.yaml"))
    # One reference tick through the template graphs with the checkpoint's weights.
    rng = np.random.default_rng(5)
    c, d, bs, m = int(args.data.encoder_out_channels), int(args.model.n_aux_chans), int(args.data.block_size), int(h.num_mels)
    frames, start, units, steps, t_start = 48, 20, 70, 3, 0.7
    r = frames - start
    f0 = np.linspace(110.0, 880.0, frames, dtype=np.float32)[None]
    f0[0, :5] = 0.0  # unvoiced frames must not NaN
    feeds = {
        "hubert": rng.standard_normal((1, units, c)).astype(np.float32),
        "mel2ph": np.minimum(np.round(np.arange(frames) * (units / frames)).astype(np.int64), units - 1)[None],
        "f0": f0,
        "volume": (rng.random((1, frames)) * 0.1).astype(np.float32),
        "randn": rng.standard_normal((1, frames * bs)).astype(np.float32),
        "spk": (rng.standard_normal((1, d)) * 0.1).astype(np.float32),
        "formant_shift": np.array([[1.5]], np.float32),
        "start_frame": np.array([start], np.int64),
    }
    z = rng.standard_normal((1, 1, m, r)).astype(np.float32)
    image = weights_image(layout, train)
    with tempfile.TemporaryDirectory() as tmp:
        for p in templates.values():
            shutil.copy(p, tmp)
        section_files(layout, image, tmp)
        x, cond = ort_session(os.path.join(tmp, "ddsp.onnx")).run(None, feeds)
        flow = ort_session(os.path.join(tmp, "reflow_step.onnx"))
        xs = (np.float32(t_start) * x + np.float32(1.0 - t_start) * z).astype(np.float32)
        t, dt = np.float32(t_start), np.float32((1.0 - t_start) / steps)
        for _ in range(steps):
            xs = flow.run(None, {"x": xs, "t": np.array([t], np.float32), "dt": np.array([dt], np.float32), "cond": cond})[0]
            t = np.float32(t + dt)
    # torch cross-check of the same tick
    ddsp_t, flow_t = template_modules(train, args, h)
    with torch.no_grad():
        xt, ct = ddsp_t(*[torch.from_numpy(feeds[k]) for k in DDSP_INPUTS])
    print("   tiny tick: torch vs ORT cond", cm.compare(ct.numpy(), cond)["snr_db"], "dB")
    arrays = [("hubert", feeds["hubert"]), ("mel2ph", feeds["mel2ph"].astype(np.float32)), ("f0", feeds["f0"]), ("volume", feeds["volume"]),
              ("randn", feeds["randn"]), ("spk", feeds["spk"]), ("z", z), ("x", x), ("cond", cond), ("x_final", xs)]
    blob, index, off = bytearray(), {}, 0
    for name, a in arrays:
        b = np.ascontiguousarray(a, np.float32).tobytes()
        index[name] = {"offset": off, "shape": list(a.shape)}
        blob += b
        off += len(b)
    with open(os.path.join(out_dir, "reference.bin"), "wb") as f:
        f.write(blob)
    ref = {"frames": frames, "start_frame": start, "units": units, "steps": steps, "t_start": t_start, "formant_shift": 1.5,
           "arrays": index, "weights_sha256": hashlib.sha256(image).hexdigest(),
           "note": "export_templates.py --tiny: inputs, the ddsp outputs and the flow after `steps` Euler steps from t_start, ONNX Runtime CPU"}
    with open(os.path.join(out_dir, "reference.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(ref, f, indent=1)
        f.write("\n")
    print(f"wrote the tiny fixture to {out_dir}")


def stage_vocoder():
    src = os.path.join(ex.OUT, "vocoder.onnx")
    cm.need(src, "run export.py --only vocoder first")
    os.makedirs(VOCODER_OUT, exist_ok=True)
    shutil.copy(src, os.path.join(VOCODER_OUT, "model.onnx"))
    shutil.copy(os.path.join(ex.VOC_DIR, "config.json"), os.path.join(VOCODER_OUT, "config.json"))
    print(f"staged the vocoder in {VOCODER_OUT} (sha256 {cm.graph_sha256(os.path.join(VOCODER_OUT, 'model.onnx'))})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-template", action="store_true", help="skip the default-architecture template")
    ap.add_argument("--vocoder", action="store_true", help="stage the vocoder entry")
    ap.add_argument("--voice", help="write a synthetic default-architecture voice into this folder")
    ap.add_argument("--tiny", action="store_true", help="write the CI fixture")
    a = ap.parse_args()
    if not a.no_template:
        args, h = default_arch()
        layout, _ = make_templates(args, h, TEMPLATE_OUT, TEMPLATE_ID)
        os.makedirs(os.path.dirname(ENGINE_LAYOUT), exist_ok=True)
        shutil.copy(os.path.join(TEMPLATE_OUT, "layout.json"), ENGINE_LAYOUT)
        print(f"copied layout.json to {ENGINE_LAYOUT}")
    if a.vocoder:
        stage_vocoder()
    if a.voice:
        make_voice(a.voice)
    if a.tiny:
        make_tiny(TINY_OUT)


if __name__ == "__main__":
    main()
