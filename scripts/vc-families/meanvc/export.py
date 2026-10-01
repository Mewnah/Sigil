#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MeanVC2 (40 ms chunk + 40 ms future) -> ONNX graphs for the experimental CPU zero-shot family
(P3-H, ADR 0006).

Upstream is vendored unchanged under ``upstream/`` (ASLP-lab/MeanVC2@13acf84c, Apache-2.0 per its
README); ``x_transformers_shim.py`` carries the two x-transformers 2.2.11 functions the DiT imports
(MIT). The streaming pipeline is upstream ``runtime/run_rt.py`` ``VCRunner``:

  16 kHz mic -> Kaldi fbank (25/10 ms) -> Fast-U2++ streaming ASR encoder (TorchScript, att/cnn
  caches) -> BN features (40 ms) -> 4x linear upsampling -> chunked DiT with KV cache (mean-flow,
  2 steps, t: 1 -> 0.5 -> 0) conditioned on a global speaker embedding through the GTM/TVT timbre
  memory -> 80-bin mel (10 ms) -> Vocos (ISTFT head) -> 16 kHz audio with a 10 ms overlap-add.

Graphs (``sigil-dev-cache/vc-families/meanvc/<name>.onnx``, fp32, opset 17, weights inside):

  fbank     wav16k f32 [1,N] (in [-1,1]; x32768 inside)          -> fbank f32 [1,F,80], F = (N-400)//160+1
            torchaudio ``kaldi.fbank`` as run_rt calls it (povey window, preemphasis 0.97, DC removal,
            512-point power spectrum, 80 Kaldi mel bins 20 Hz..8 kHz, log(max(e, eps)), energy_floor 0)
  asr       fbank f32 [1,11,80], offset i64 [], rcs i64 [] (always 4), att_cache f32 [6,4,4,128],
            cnn_cache f32 [6,1,256,8]                              -> bn f32 [1,2,256], att_cache, cnn_cache
            the upstream TorchScript encoder exported as is; fixed shapes
  gtm       spks f32 [1,256]                                       -> k_mem f32 [1,32,256], v_mem f32 [1,32,256]
            once per voice (the Global Timbre Memory of the speaker embedding)
  dit       x f32 [1,8,80], t f32 [1], r f32 [1], cond f32 [1,8,256], spks f32 [1,256],
            k_mem, v_mem, offset i64 [1], n_valid i64 [1], cache f32 [4,2,2,12,64]
                                                                   -> u f32 [1,8,80], cache_out f32 [4,2,2,12,64]
  dit_step  the same plus dt f32 [1]                               -> x_next = x - dt u, cache_out
  vocos     mel f32 [1,80,M] (already (mel+1)/2, as run_rt feeds it) -> wav f32 [1,160(M-1)]

What Sigil changes to make the graphs portable and fixed-shape (checked equal by ``parity.py``):

- ``dit``: upstream grows each layer's KV cache from nothing to (t_p+1) chunks and rebuilds the
  chunk mask per call. The graph keeps one fixed 12-frame buffer per layer (front-padded while it
  fills) plus ``n_valid``; padded keys are masked with -inf, so they get exactly zero weight, and the
  frames a smaller upstream cache would have dropped fall outside the layer's chunk mask anyway.
  Keys are cached normalised and unrotated, as upstream does, and RoPE is applied to positions
  ``offset + 8 - 20 ... offset + 7``. With every shape fixed, the CUDA EP can capture the step.
- ``vocos``: upstream is TorchScript; ``Vocos`` below loads the same state dict (strict) into the same
  layers, and ``torch.istft`` (no ONNX export) becomes a transposed convolution with an inverse-DFT
  basis divided by the squared-window envelope, as in the Seed-VC HiFT graph.
- ``asr``: the TorchScript archive was saved by torch >= 2.5, whose ``pad_sequence`` gained a
  ``padding_side`` argument; ``load_asr`` rewrites that one call (only the attention decoder, which
  inference never runs, uses it) so torch 2.3 can load and export the encoder.

Inputs (``sigil-dev-cache/meanvc2/ASLP-lab_MeanVC2``, HF ASLP-lab/MeanVC2 @39cdd195, fetched once):
  meanvc2_40ms_40ms.safetensors (70,839,448 bytes), fastu2pp_80ms.pt (246,292,087), vocos.pt (33,223,674).
The speaker encoder (WavLM-Large + a fine-tuned ECAPA-TDNN, ``wavlm_large_finetune.pth``) is a
Google Drive download that cannot be scripted; it runs once per reference voice, off the real-time
path, and is not exported here (see ADR 0006).

Usage (repo root): ``python scripts/vc-families/meanvc/export.py [--only fbank,asr,gtm,dit,dit_step,vocos]``
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import common as cm  # noqa: E402
import x_transformers_shim  # noqa: E402

x_transformers_shim.install()
sys.path.insert(0, os.path.join(UPSTREAM, "runtime"))

MVC = os.path.join(cm.DEV_CACHE, "meanvc2", "ASLP-lab_MeanVC2")
DIT_CKPT = os.path.join(MVC, "meanvc2_40ms_40ms.safetensors")
ASR_CKPT = os.path.join(MVC, "fastu2pp_80ms.pt")
VOCOS_CKPT = os.path.join(MVC, "vocos.pt")
CONFIG = os.path.join(UPSTREAM, "src", "config", "config_40ms_40ms.json")
OUT = cm.family_out("meanvc")

# run_rt.py MODEL_PATHS["40ms"] and VCRunner constants
CHUNK = 4             # DiT chunk (frames of 10 ms)
BLOCK = 4             # future block
BN_WINDOW = 11        # fbank frames per ASR call
BN_STRIDE = 8
RCS = 4               # required_cache_size
ASR_OFFSET_INIT = 4
ASR_OFFSET_STEP = 2
ATT_CACHE = (6, 4, RCS, 128)
CNN_CACHE = (6, 1, 256, 8)
CAP = 12              # max over layers of (t_p + 1) * CHUNK
HOP = 160
N_FFT = 640


# ---------------------------------------------------------------------------------------------
# Upstream models
# ---------------------------------------------------------------------------------------------

def load_dit():
    """``run_rt._load_vc_model`` (strict=False as upstream; the only missing keys are the unused
    ``cache_embed``)."""
    from safetensors.torch import load_file
    from src.dit import DiT
    cm.need(DIT_CKPT, "HF ASLP-lab/MeanVC2 meanvc2_40ms_40ms.safetensors")
    cfg = json.load(open(CONFIG, encoding="utf-8"))
    model = DiT(**cfg["model"])
    res = model.load_state_dict(load_file(DIT_CKPT), strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("cache_embed.")] + list(res.unexpected_keys)
    if bad:
        raise SystemExit(f"MeanVC2 DiT checkpoint mismatch: {bad[:8]}")
    model.float().eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_asr():
    """The Fast-U2++ TorchScript encoder, loadable by torch 2.3 (see the module docstring). The
    patched archive is cached next to the graphs."""
    cm.need(ASR_CKPT, "HF ASLP-lab/MeanVC2 fastu2pp_80ms.pt")
    dst = os.path.join(OUT, "_torch23", "fastu2pp_80ms.pt")
    if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(ASR_CKPT):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        old = b"torch.pad_sequence(sequences0, batch_first, padding_value, padding_side)"
        new = b"torch.pad_sequence(sequences0, batch_first, padding_value)"
        patched = 0
        with zipfile.ZipFile(ASR_CKPT) as zin, zipfile.ZipFile(dst + ".tmp", "w", zipfile.ZIP_STORED) as zout:
            for info in zin.infolist():
                data = zin.read(info.filename)
                if info.filename.endswith("code/__torch__/torch/nn/utils/rnn.py"):
                    patched += data.count(old)
                    data = data.replace(old, new)
                elif info.filename.endswith("code/__torch__/torch/nn/utils/rnn.py.debug_pkl"):
                    continue  # source ranges of the edited file; TorchScript loads without them
                zi = zipfile.ZipInfo(info.filename, date_time=info.date_time)
                zi.compress_type = zipfile.ZIP_STORED
                zout.writestr(zi, data)
        if torch.__version__ < "2.5" and patched != 1:
            raise SystemExit(f"unexpected Fast-U2++ archive layout ({patched} pad_sequence calls patched)")
        os.replace(dst + ".tmp", dst)
    asr = torch.jit.load(dst, map_location="cpu")
    asr.eval()
    return asr


def load_vocos_jit():
    cm.need(VOCOS_CKPT, "HF ASLP-lab/MeanVC2 vocos.pt")
    v = torch.jit.load(VOCOS_CKPT, map_location="cpu")
    v.eval()
    return v


def load_upstream():
    return {"dit": load_dit(), "asr": load_asr(), "vocos_jit": load_vocos_jit()}


# ---------------------------------------------------------------------------------------------
# Graph modules
# ---------------------------------------------------------------------------------------------

class KaldiFbank(nn.Module):
    """``torchaudio.compliance.kaldi.fbank(wav * 32768, frame_length=25, frame_shift=10,
    snip_edges=True, num_mel_bins=80, energy_floor=0.0, dither=0.0, sample_frequency=16000)``.

    Per frame, DC removal, preemphasis (0.97, first sample against itself), the povey window and the
    512-point real DFT are all linear, so they fold into one [514, 400] matrix applied as a strided
    convolution (frame shift 160); then power, Kaldi mel bins and log(max(e, float eps))."""

    def __init__(self):
        super().__init__()
        import torchaudio.compliance.kaldi as kaldi
        n, npad = 400, 512
        mel, _ = kaldi.get_mel_banks(80, npad, 16000.0, 20.0, 0.0, 100.0, -500.0, 1.0)
        self.register_buffer("mel", F.pad(mel, (0, 1)).T.contiguous().float())            # [257,80]
        eye = torch.eye(n, dtype=torch.float64)
        dc = eye - 1.0 / n
        shift = torch.zeros(n, n, dtype=torch.float64)
        shift[1:, :-1] = torch.eye(n - 1, dtype=torch.float64)
        shift[0, 0] = 1.0
        pre = eye - 0.97 * shift
        win = torch.hann_window(n, periodic=False, dtype=torch.float64).pow(0.85)
        frame_op = win[:, None] * (pre @ dc)                                                # raw frame -> windowed frame
        k = torch.arange(npad // 2 + 1, dtype=torch.float64)[:, None]
        t = torch.arange(n, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * t / npad                                                    # zero padding: only 400 taps
        basis = torch.cat([torch.cos(ang), -torch.sin(ang)], 0) @ frame_op                  # [514,400]
        self.register_buffer("basis", basis[:, None, :].float())
        self.nb = npad // 2 + 1
        self.eps = float(torch.finfo(torch.float32).eps)

    def forward(self, wav16k):
        spec = F.conv1d(wav16k[:, None, :] * 32768.0, self.basis, stride=160)              # [1,514,F]
        re, im = spec[:, : self.nb], spec[:, self.nb:]
        power = (re * re + im * im).transpose(1, 2)                                         # [1,F,257]
        return torch.log(torch.clamp(power @ self.mel, min=self.eps))


class Gtm(nn.Module):
    def __init__(self, dit):
        super().__init__()
        self.gtm = dit.gtm

    def forward(self, spks):
        return self.gtm(spks)


def _rope_freqs(inv_freq, pos):
    """``RotaryEmbedding.forward(pos)[0]`` (no xpos, interpolation factor 1) without the
    ``torch.amp.autocast`` decorator, which the ONNX tracer cannot pass: [1, S, 2 * len(inv_freq)]."""
    freqs = pos.to(inv_freq.dtype)[:, None] * inv_freq[None, :]
    return torch.stack((freqs, freqs), dim=-1).reshape(1, pos.shape[0], -1)


def _rotate_half(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).reshape(*x.shape[:-2], -1)


def _apply_rope(t, freqs):
    """``apply_rotary_pos_emb(t, freqs, 1.)`` for a rotary width equal to the head dim."""
    f = freqs[:, -t.shape[-2]:, :][:, None]
    return t * f.cos() + _rotate_half(t) * f.sin()


def _chunk_mask(n_keys, t_p, t_f):
    """``BlockAttnProcessor.try_cached_mask`` over ``n_keys`` keys (bool [n_keys, n_keys])."""
    idx = torch.arange(n_keys)
    ci = torch.div(idx, CHUNK, rounding_mode="floor")
    qi, kj = ci[:, None], ci[None, :]
    same = qi == kj
    prev = (kj >= (qi - t_p)) & (kj < qi)
    nxt = (kj == (qi + 1)) & ((idx % CHUNK)[None, :] < BLOCK * t_f)
    return same | prev | nxt


class DitStream(nn.Module):
    """One upstream ``DiT.forward(x, t, r, cond=, spks=, offset=, is_inference=True, kv_cache=,
    gtm_kv=)`` call on fixed-size caches (see the module docstring). ``cache`` is
    [layer, key/value, head, frame, dim]; ``n_valid`` counts the valid (most recent) frames."""

    def __init__(self, dit, step=False):
        super().__init__()
        self.dit = dit
        self.step = step
        blocks = dit.transformer_blocks
        self.heads = blocks[0].attn.heads
        self.dim_head = blocks[0].attn.inner_dim // self.heads
        self.depth = len(blocks)
        n_keys = CAP + CHUNK + BLOCK
        masks = [_chunk_mask(n_keys, b.attn.processor.t_p, b.attn.processor.t_f)[-(CHUNK + BLOCK):] for b in blocks]
        self.register_buffer("masks", torch.stack(masks))                                  # [depth,8,20] bool
        self.register_buffer("key_idx", torch.arange(n_keys, dtype=torch.int64))

    def _attend(self, attn, h, k_cache, v_cache, freqs, bias):
        b = h.shape[0]
        q = attn.to_q(h).view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        k = attn.to_k(h).view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        v = attn.to_v(h).view(b, -1, self.heads, self.dim_head).transpose(1, 2)
        if attn.q_norm is not None:
            q = attn.q_norm(q)
        if attn.k_norm is not None:
            k = attn.k_norm(k)
        k = torch.cat([k_cache[None], k], dim=2)                                           # [1,h,20,d]
        v = torch.cat([v_cache[None], v], dim=2)
        new_k = k[:, :, :-BLOCK][:, :, -CAP:]
        new_v = v[:, :, :-BLOCK][:, :, -CAP:]
        q = _apply_rope(q, freqs)
        k = _apply_rope(k, freqs)
        w = torch.softmax(q @ k.transpose(-2, -1) * (1.0 / math.sqrt(self.dim_head)) + bias, dim=-1)
        o = (w @ v).transpose(1, 2).reshape(b, -1, self.heads * self.dim_head)
        return attn.to_out[0](o), new_k[0], new_v[0]

    def forward(self, x, t, r, cond, spks, k_mem, v_mem, offset, n_valid, cache, dt=None):
        d = self.dit
        t_emb = d.t_time_embed(t) + d.r_time_embed(r)
        timbre = d.temporal_timbre(cond, k_mem, v_mem)
        spk = spks.unsqueeze(1).expand(-1, cond.shape[1], -1)
        h = d.input_embed(x, timbre, spk)
        n_keys = self.key_idx.shape[0]
        # RoPE positions of the keys: the last one is offset + 7 (upstream arange(offset + 8)[-S:])
        pos = (offset - n_keys + CHUNK + BLOCK) + self.key_idx
        freqs = _rope_freqs(d.rotary_embed.inv_freq, pos)
        valid = self.key_idx[None, :] >= (CAP - n_valid)                                  # [1,20]
        outs = []
        for i, blk in enumerate(d.transformer_blocks):
            bias = torch.where(self.masks[i] & valid, 0.0, float("-inf"))[None, None]      # [1,1,8,20]
            norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = blk.attn_norm(h, emb=t_emb)
            a, nk, nv = self._attend(blk.attn, norm, cache[i, 0], cache[i, 1], freqs, bias)
            h = h + gate_msa.unsqueeze(1) * a
            norm = blk.ff_norm(h) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
            h = h + gate_mlp.unsqueeze(1) * blk.ff(norm)
            outs.append(torch.stack([nk, nv]))
        h = d.norm_out(h, t_emb)
        u = d.proj_out(h)
        cache_out = torch.stack(outs)
        if self.step:
            return x - dt * u, cache_out
        return u, cache_out


class DitStep(DitStream):
    def __init__(self, dit):
        super().__init__(dit, step=True)

    def forward(self, x, t, r, dt, cond, spks, k_mem, v_mem, offset, n_valid, cache):
        return super().forward(x, t, r, cond, spks, k_mem, v_mem, offset, n_valid, cache, dt=dt)


def istft_kernels(n_fft, window):
    """Transposed-convolution kernels for ``torch.istft(spec, n_fft, hop, n_fft, window, center=True)``
    of a onesided spectrum given as [real | imag] channels: the inverse real DFT (DC and Nyquist
    weighted 1, the other bins 2, over n_fft) times the synthesis window, and the squared window for
    the overlap-add envelope."""
    win = window.double()
    k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
    t = torch.arange(n_fft, dtype=torch.float64)[None, :]
    ang = 2 * math.pi * k * t / n_fft
    a = torch.full((n_fft // 2 + 1, 1), 2.0, dtype=torch.float64)
    a[0] = 1.0
    a[-1] = 1.0
    basis = torch.cat([a * torch.cos(ang), -a * torch.sin(ang)], 0) / n_fft * win         # [n_fft+2, n_fft]
    return basis[:, None, :].float(), (win ** 2)[None, None, :].float()


def istft_conv(spec, basis, win_sq, hop):
    """``torch.istft(..., center=True)`` of ``spec`` [1, n_fft+2, F] ([real | imag]): overlap-add of
    the windowed frames divided by the window envelope, n_fft / 2 trimmed at both ends
    (length hop (F - 1))."""
    n_fft = basis.shape[-1]
    ola = F.conv_transpose1d(spec, basis, stride=hop)                                      # [1,1,n_fft+hop(F-1)]
    env = F.conv_transpose1d(torch.ones_like(spec[:, :1]), win_sq, stride=hop)
    pad = n_fft // 2
    return (ola / env)[:, 0, pad:-pad]


class Vocos(nn.Module):
    """The MeanVC2 Vocos (ConvNeXt backbone dim 320 x 8, ISTFT head n_fft 640, hop 160, center)
    rebuilt from the TorchScript state dict; ``decode(mel)`` == upstream ``vocoder.decode``."""

    class Block(nn.Module):
        def __init__(self, dim=320, inner=1536):
            super().__init__()
            self.dwconv = nn.Conv1d(dim, dim, 7, padding=3, groups=dim)
            self.norm = nn.LayerNorm(dim, eps=1e-6)
            self.pwconv1 = nn.Linear(dim, inner)
            self.act = nn.GELU()
            self.pwconv2 = nn.Linear(inner, dim)
            self.gamma = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            y = self.dwconv(x).transpose(1, 2)
            y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
            return x + (self.gamma * y).transpose(1, 2)

    def __init__(self, jit):
        super().__init__()
        sd = jit.state_dict()
        n_blocks = len({k.split(".")[2] for k in sd if k.startswith("backbone.convnext.")})
        dim, n_mels = sd["backbone.embed.weight"].shape[:2]
        inner = sd["backbone.convnext.0.pwconv1.weight"].shape[0]
        self.backbone = nn.Module()
        self.backbone.embed = nn.Conv1d(n_mels, dim, 7, padding=3)
        self.backbone.norm = nn.LayerNorm(dim, eps=1e-6)
        self.backbone.convnext = nn.ModuleList([Vocos.Block(dim, inner) for _ in range(n_blocks)])
        self.backbone.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.head = nn.Module()
        self.head.out = nn.Linear(dim, N_FFT + 2)
        self.head.istft = nn.Module()
        self.head.istft.register_buffer("window", torch.hann_window(N_FFT))
        self.load_state_dict(sd, strict=True)
        basis, win_sq = istft_kernels(N_FFT, self.head.istft.window)
        self.register_buffer("istft_basis", basis)
        self.register_buffer("win_sq", win_sq)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, mel):
        bb = self.backbone
        x = bb.norm(bb.embed(mel).transpose(1, 2)).transpose(1, 2)
        for blk in bb.convnext:
            x = blk(x)
        x = bb.final_layer_norm(x.transpose(1, 2))
        x = self.head.out(x).transpose(1, 2)                                              # [1,642,M]
        mag, p = x[:, : N_FFT // 2 + 1], x[:, N_FFT // 2 + 1:]
        mag = torch.clamp(torch.exp(mag), max=100.0)
        spec = torch.cat([mag * torch.cos(p), mag * torch.sin(p)], 1)
        return istft_conv(spec, self.istft_basis, self.win_sq, HOP)


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------

def graph_specs(up):
    torch.manual_seed(0)
    dit = up["dit"]
    heads = dit.transformer_blocks[0].attn.heads
    dh = dit.transformer_blocks[0].attn.inner_dim // heads
    cache = torch.randn(len(dit.transformer_blocks), 2, heads, CAP, dh)
    n = CHUNK + BLOCK
    dit_args = (torch.randn(1, n, 80), torch.tensor([1.0]), torch.tensor([0.5]), torch.randn(1, n, 256), torch.randn(1, 256),
                torch.randn(1, 32, 256), torch.randn(1, 32, 256), torch.tensor([40]), torch.tensor([8]), cache)
    dit_in = ["x", "t", "r", "cond", "spks", "k_mem", "v_mem", "offset", "n_valid", "cache"]
    return {
        "fbank": (KaldiFbank(), (torch.randn(1, 2560) * 0.1,), ["wav16k"], ["fbank"], {"wav16k": {1: "N"}, "fbank": {1: "F"}}),
        # the TorchScript module itself (a traced wrapper cannot call into it); rcs stays an input
        "asr": (up["asr"], (torch.randn(1, BN_WINDOW, 80), torch.tensor(ASR_OFFSET_INIT, dtype=torch.int64),
                            torch.tensor(RCS, dtype=torch.int64), torch.zeros(ATT_CACHE), torch.zeros(CNN_CACHE)),
                ["fbank", "offset", "rcs", "att_cache", "cnn_cache"], ["bn", "att_cache_out", "cnn_cache_out"], None),
        "gtm": (Gtm(dit), (torch.randn(1, 256),), ["spks"], ["k_mem", "v_mem"], None),
        "dit": (DitStream(dit), dit_args, dit_in, ["u", "cache_out"], None),
        "dit_step": (DitStep(dit), dit_args[:3] + (torch.tensor([0.5]),) + dit_args[3:],
                     dit_in[:3] + ["dt"] + dit_in[3:], ["x_next", "cache_out"], None),
        "vocos": (Vocos(up["vocos_jit"]), (torch.rand(1, 80, 6),), ["mel"], ["wav"], {"mel": {2: "M"}, "wav": {1: "S"}}),
    }


def _stamp(path, name):
    import onnx
    m = onnx.load(path)
    onnx.checker.check_model(m)
    del m.metadata_props[:]
    m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.source", value="ASLP-lab/MeanVC2@13acf84c meanvc2_40ms_40ms"))
    m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.graph", value=f"meanvc/{name}"))
    onnx.save(m, path)


def export(names):
    up = load_upstream()
    specs = graph_specs(up)
    written = []
    for name in names:
        mod, args, ins, outs, dyn = specs[name]
        mod.eval()
        path = os.path.join(OUT, f"{name}.onnx")
        with torch.no_grad():
            torch.onnx.export(mod, args, path, input_names=ins, output_names=outs, dynamic_axes=dyn, opset_version=17)
        _stamp(path, name)
        written.append((path, os.path.getsize(path)))
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="fbank,asr,gtm,dit,dit_step,vocos")
    args = ap.parse_args()
    export([n.strip() for n in args.only.split(",") if n.strip()])


if __name__ == "__main__":
    main()
