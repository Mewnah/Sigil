#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seed-VC realtime (``seed-uvit-tat-xlsr-tiny``) -> ONNX graphs for Sigil-ZS (P3-H, ADR 0006).

Upstream model code is vendored unchanged under ``upstream/`` (Plachtaa/seed-vc@51383efd, GPL-3.0;
the CAM++ files are 3D-Speaker's, Apache-2.0; the HiFT files are CosyVoice's, Apache-2.0). The
model is loaded exactly as upstream ``real-time-gui.py`` ``load_models`` does, and every graph
wraps the upstream modules; Sigil code only replaces what ONNX cannot express portably:

- ``lr``: upstream ``F.interpolate(mode="nearest")`` becomes a gather over indices computed the way
  PyTorch does (``floor(i * (float)T_in / T_out)`` in float32), because ONNX ``Resize`` computes
  ``floor(i / (T_out / T_in))`` and lands one frame early at exact multiples on some EPs;
- ``hift``: randomness becomes inputs (``harm_phase``, ``sine_noise``); the NSF phase is summed
  per frame and wrapped (float32 per-sample cumsum over a whole window drifts on GPUs);
  ``torch.stft`` / ``torch.istft`` become strided convolutions with a DFT basis (``istft`` has no
  ONNX export; ``STFT`` has no kernel on every EP). ``hift_f0`` + ``hift_nsf`` are the same
  arithmetic split at upstream's own ``forward(x, f0)`` seam, for the phase-continuous NSF lever
  and partial decode (``sampler.py``);
- ``mel`` (reference mel): the STFT is a strided convolution too;
- ``dit_step``: one Euler step with CFG fused around the upstream estimator (the sampler state
  stays on the device between steps).

Inputs (``sigil-dev-cache/seed-vc``, fetched once, one file at a time):
  Plachta_Seed-VC/DiT_uvit_tat_xlsr_ema.pth     HF Plachta/Seed-VC @257283f9 (142,112,203 bytes)
  wav2vec2-xls-r-300m/{config,preprocessor_config}.json, pytorch_model.bin
                                                HF facebook/wav2vec2-xls-r-300m @1a640f32 (1,269,737,156 bytes)
  CosyVoice-300M/hift.pt                        HF FunAudioLLM/CosyVoice-300M @24c40509 (81,896,716 bytes)
  campplus_cn_common.bin                        in the seed-vc repository archive (= HF funasr/campplus)

Graphs (``sigil-dev-cache/vc-families/seed_vc/<name>.onnx``, weights inside; ``<name>.fp16.onnx`` =
fp16 inside, fp32 I/O), opset 17, every length dynamic:

  xlsr      wav16k f32 [1,N] (raw, the graph normalises)       -> feats f32 [1,T,1024], T = (N-400)//320+1
  lr        feats f32 [1,T,1024], n_frames i64 [1]              -> cond f32 [1,n_frames,384]
  dit       x f32 [B,80,L], prompt_x f32 [B,80,L], x_lens i64 [B], t f32 [B], style f32 [B,192],
            cond f32 [B,L,384]                                  -> v f32 [B,80,L]
  dit_step  x f32 [1,80,L], prompt_x f32 [1,80,L], cond f32 [1,L,384], style f32 [1,192], t f32 [1],
            dt f32 [1], cfg_rate f32 [1], prompt_len i64 [1]   -> x_next f32 [1,80,L]   (CFG, batch 2 inside)
  dit_step1 the same without cfg_rate                          -> x_next f32 [1,80,L]   (cfg 0, batch 1)
  hift      mel f32 [1,80,M], harm_phase f32 [1,9,1], sine_noise f32 [1,9,256M]
                                                                -> audio f32 [1,256M], f0 f32 [1,M]
  hift_f0   mel f32 [1,80,M]                                    -> f0 f32 [1,M]   (fp32 only)
  hift_nsf  mel f32 [1,80,W], f0 f32 [1,W], harm_phase f32 [1,9,1] (phase at the slice's first
            sample), sine_noise f32 [1,9,256W]                  -> audio f32 [1,256W]
  campplus  fbank f32 [1,F,80] (Kaldi fbank, mean-subtracted)  -> style f32 [1,192]
  mel       wav22k f32 [1,S]                                    -> mel f32 [1,80,M], M = (S+768-1024)//256+1

fp16 precision islands (as under upstream's fp16 autocast, which keeps parameters, buffers and the
sampler's t in fp32): the DiT timestep embedders get the exact fp32 ``freqs`` and the exact fp32
``t``; HiFT's Snake ``alpha`` keeps its exact fp32 values (each copied before ``.half()`` and put
back after).

Usage (repo root): ``python scripts/vc-families/seed_vc/export.py [--only xlsr,dit] [--no-fp16]``
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
UPSTREAM = os.path.join(HERE, "upstream")
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, UPSTREAM)

import common as cm  # noqa: E402

# Upstream imports that inference never calls: munch (config attribute access), dac (the vector
# quantizer of discrete-content configs; this config is continuous).
cm.install_stubs({"munch": {"Munch": cm.Munch}, "dac.nn.quantize": {"VectorQuantize": None}})

SEED = os.path.join(cm.DEV_CACHE, "seed-vc")
DIT_CKPT = os.path.join(SEED, "Plachta_Seed-VC", "DiT_uvit_tat_xlsr_ema.pth")
XLSR_DIR = os.path.join(SEED, "wav2vec2-xls-r-300m")
HIFT_CKPT = os.path.join(SEED, "CosyVoice-300M", "hift.pt")
CAMPPLUS_CKPT = os.path.join(cm.DEV_CACHE, "seed-vc-src", "seed-vc-main", "campplus_cn_common.bin")
CONFIG = os.path.join(UPSTREAM, "configs", "presets", "config_dit_mel_seed_uvit_xlsr_tiny.yml")
HIFT_CONFIG = os.path.join(UPSTREAM, "configs", "hifigan.yml")
OUT = cm.family_out("seed_vc")

SR = 22050
HOP = 256
N_HARM = 9  # HiFT nb_harmonics 8 + the fundamental
XLSR_LAYERS = 12


# ---------------------------------------------------------------------------------------------
# Upstream model, loaded as real-time-gui.py load_models() does (fp32, eval)
# ---------------------------------------------------------------------------------------------

def load_upstream(device="cpu"):
    import yaml
    from modules.commons import build_model, load_checkpoint, recursive_munch
    from modules.campplus.DTDNN import CAMPPlus
    from modules.hifigan.generator import HiFTGenerator
    from modules.hifigan.f0_predictor import ConvRNNF0Predictor
    from transformers import Wav2Vec2Model

    for p, hint in ((DIT_CKPT, "HF Plachta/Seed-VC DiT_uvit_tat_xlsr_ema.pth"), (HIFT_CKPT, "HF FunAudioLLM/CosyVoice-300M hift.pt"),
                    (os.path.join(XLSR_DIR, "pytorch_model.bin"), "HF facebook/wav2vec2-xls-r-300m"),
                    (CAMPPLUS_CKPT, "the seed-vc repository archive (campplus_cn_common.bin)")):
        cm.need(p, hint)

    config = yaml.safe_load(open(CONFIG, "r", encoding="utf-8"))
    model_params = recursive_munch(config["model_params"])
    model_params.dit_type = "DiT"
    model = build_model(model_params, stage="DiT")
    model, _, _, _ = load_checkpoint(model, None, DIT_CKPT, load_only_params=True, ignore_modules=[], is_distributed=False)
    for key in model:
        model[key].eval()
        model[key].to(device)
    model.cfm.estimator.setup_caches(max_batch_size=1, max_seq_length=8192)

    campplus = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus.load_state_dict(torch.load(CAMPPLUS_CKPT, map_location="cpu"))
    campplus.eval().to(device)

    hift_config = yaml.safe_load(open(HIFT_CONFIG, "r", encoding="utf-8"))
    hift = HiFTGenerator(**hift_config["hift"], f0_predictor=ConvRNNF0Predictor(**hift_config["f0_predictor"]))
    hift.load_state_dict(torch.load(HIFT_CKPT, map_location="cpu"))
    hift.eval().to(device)

    xlsr = Wav2Vec2Model.from_pretrained(XLSR_DIR)
    # The checkpoint stores the positional conv's weight_norm as weight_g / weight_v. transformers
    # 4.39 under torch >= 2.1 builds it as a parametrization and leaves it randomly initialised
    # (upstream pins transformers 4.46.3, which renames the keys on load); copy them explicitly.
    sd = torch.load(os.path.join(XLSR_DIR, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    conv = xlsr.encoder.pos_conv_embed.conv
    with torch.no_grad():
        if hasattr(conv, "parametrizations"):
            conv.parametrizations.weight.original0.copy_(sd["wav2vec2.encoder.pos_conv_embed.conv.weight_g"])
            conv.parametrizations.weight.original1.copy_(sd["wav2vec2.encoder.pos_conv_embed.conv.weight_v"])
        else:
            conv.weight_g.copy_(sd["wav2vec2.encoder.pos_conv_embed.conv.weight_g"])
            conv.weight_v.copy_(sd["wav2vec2.encoder.pos_conv_embed.conv.weight_v"])
    del sd
    # Fold the parametrization once (weight = g v / |v|, the value upstream recomputes each call);
    # deep copies of parametrized modules are unreliable, and the graphs want plain weights.
    cm.strip_weight_norm(xlsr)
    xlsr.encoder.layers = xlsr.encoder.layers[:XLSR_LAYERS]
    xlsr.eval().to(device)

    for m in (model, campplus, hift, xlsr):
        cm.freeze(m)
    sp = config["preprocess_params"]["spect_params"]
    mel_args ={"n_fft": sp["n_fft"], "win_size": sp["win_length"], "hop_size": sp["hop_length"], "num_mels": sp["n_mels"],
                "sampling_rate": config["preprocess_params"]["sr"], "fmin": sp.get("fmin", 0),
                "fmax": None if sp.get("fmax", "None") == "None" else 8000, "center": False}
    return {"model": model, "campplus": campplus, "hift": hift, "xlsr": xlsr, "mel_args": mel_args}


def xlsr_normalize(wav):
    """Wav2Vec2FeatureExtractor(do_normalize=True) on one unpadded utterance: zero mean, unit
    variance (population variance, + 1e-7)."""
    mean = wav.mean(dim=-1, keepdim=True)
    var = ((wav - mean) ** 2).mean(dim=-1, keepdim=True)
    return (wav - mean) / torch.sqrt(var + 1e-7)


# ---------------------------------------------------------------------------------------------
# Graph wrappers
# ---------------------------------------------------------------------------------------------

class Xlsr(nn.Module):
    """wav16k [1,N] -> XLS-R-300M layer-12 hidden states (after the encoder's final LayerNorm),
    as upstream ``semantic_fn`` (the feature extractor's normalisation included; no attention mask,
    upstream passes none)."""

    def __init__(self, xlsr, half=False):
        super().__init__()
        self.m = xlsr.half() if half else xlsr
        self.half = half

    def forward(self, wav16k):
        x = xlsr_normalize(wav16k)
        if self.half:
            x = x.half()
        return self.m(x).last_hidden_state.float()


class LengthRegulator(nn.Module):
    """Upstream InterpolateRegulator (continuous content, no f0) with PyTorch's nearest-neighbour
    source index computed explicitly: ``min(floor(i * (float)T_in / n), T_in - 1)`` in float32."""

    def __init__(self, lr):
        super().__init__()
        self.lr = lr

    def forward(self, feats, n_frames):
        x = self.lr.content_in_proj(feats)                               # [1,T,384]
        t_in = torch.ones_like(feats[0, :, 0]).sum()                     # float T_in
        n = n_frames[0]
        pos = torch.arange(n, device=feats.device).to(torch.float32)
        scale = t_in / n.to(torch.float32)
        idx = torch.minimum(torch.floor(pos * scale), t_in - 1).to(torch.int64)
        x = torch.index_select(x, 1, idx)                                # [1,n,384]
        return self.lr.model(x.transpose(1, 2)).transpose(1, 2)          # mask is all ones (one item)


class Estimator(nn.Module):
    """The upstream DiT estimator unchanged (one velocity evaluation)."""

    def __init__(self, dit, half=False):
        super().__init__()
        self.half = half
        if half:
            # As under upstream's fp16 autocast, where parameters, buffers and the sampler's t all stay
            # fp32: the timestep sinusoid (1000 t x frequency reaches 1000 rad, where an fp16 ulp is
            # 0.5 rad) is computed from the exact fp32 ``freqs`` and the exact fp32 ``t``, and only
            # the MLP after it runs in fp16. ``freqs`` is copied before ``.half()`` rounds it (and put
            # back after); ``t`` never passes through fp16 (``forward``).
            import types
            embedders = [m for m in dit.modules() if type(m).__name__ == "TimestepEmbedder"]   # t_embedder (+ t_embedder2)
            exact = [m.freqs.detach().clone().float() for m in embedders]
            dit = dit.half()
            for te, freqs in zip(embedders, exact):
                te.freqs = freqs                                                # the registered buffer, fp32 again
                te.forward = types.MethodType(lambda m, t: m.mlp(m.timestep_embedding(t.float()).to(m.mlp[0].weight.dtype)), te)
        # .half() converts parameters and registered buffers; the RoPE table (freqs_cis, a plain
        # attribute set by setup_caches) stays fp32, and upstream applies RoPE in fp32 anyway.
        self.dit = dit
        # upstream computes ``x_lens + self.style_as_token + self.time_as_token`` with Python bools,
        # which the TorchScript tracer cannot record; the same flags as ints behave identically.
        self.dit.style_as_token = int(self.dit.style_as_token)
        self.dit.time_as_token = int(self.dit.time_as_token)

    def forward(self, x, prompt_x, x_lens, t, style, cond):
        if self.half:
            # t stays fp32: the DiT uses it only in the timestep embedders' fp32 sinusoid
            x, prompt_x, style, cond = x.half(), prompt_x.half(), style.half(), cond.half()
        return self.dit(x, prompt_x, x_lens, t, style, cond).float()


class DitStep(nn.Module):
    """One step of upstream ``BASECFM.solve_euler``. With CFG (``dit_step``): the conditional and
    unconditional (prompt, style and content zeroed) passes run as one batch of 2, then
    ``x + dt * ((1 + cfg) * v_c - cfg * v_u)``; without (``dit_step1``, upstream's
    ``inference_cfg_rate == 0`` branch): one pass, ``x + dt * v``. Then the prompt frames are
    zeroed again. ``x_lens`` is derived from the input's length, so pinning ``L`` folds all shape
    arithmetic and the CUDA EP can capture the step as a CUDA graph."""

    def __init__(self, dit, half=False, cfg=True):
        super().__init__()
        self.est = Estimator(dit, half)
        self.cfg = cfg

    def _step(self, x, prompt_x, cond, style, t, dt, cfg_rate, prompt_len):
        pos = torch.cumsum(torch.ones_like(x[0, 0, :]), 0) - 1.0         # 0..L-1 (float)
        big_l = (pos[-1:] + 1.0).to(torch.int64)                          # [1]
        if self.cfg:
            xs = torch.cat([x, x], 0)
            ps = torch.cat([prompt_x, torch.zeros_like(prompt_x)], 0)
            ss = torch.cat([style, torch.zeros_like(style)], 0)
            cs = torch.cat([cond, torch.zeros_like(cond)], 0)
            v = self.est(xs, ps, torch.cat([big_l, big_l], 0), torch.cat([t, t], 0), ss, cs)
            v = (1.0 + cfg_rate) * v[0:1] - cfg_rate * v[1:2]
        else:
            v = self.est(x, prompt_x, big_l, t, style, cond)
        x = x + dt * v
        keep = (pos >= prompt_len.to(torch.float32)).to(x.dtype)
        return x * keep

    def forward(self, x, prompt_x, cond, style, t, dt, cfg_rate, prompt_len):
        return self._step(x, prompt_x, cond, style, t, dt, cfg_rate, prompt_len)


class DitStep1(DitStep):
    def __init__(self, dit, half=False):
        super().__init__(dit, half, cfg=False)

    def forward(self, x, prompt_x, cond, style, t, dt, prompt_len):
        return self._step(x, prompt_x, cond, style, t, dt, None, prompt_len)


def _frac(x):
    return x - torch.floor(x)


def _snake_fp32(self, x):
    """Upstream Snake.forward (alpha_logscale False) computed in fp32 and returned in x's dtype."""
    alpha = self.alpha.unsqueeze(0).unsqueeze(-1)
    xf = x.float()
    return (xf + (1.0 / (alpha + self.no_div_by_zero)) * torch.pow(torch.sin(xf * alpha), 2)).to(x.dtype)


class HiftDecoder(nn.Module):
    """The upstream HiFT decoder body (conv_pre ... conv_post) exactly as in HiFTGenerator.forward."""

    def __init__(self, h):
        super().__init__()
        self.conv_pre, self.ups, self.source_downs = h.conv_pre, h.ups, h.source_downs
        self.source_resblocks, self.resblocks = h.source_resblocks, h.resblocks
        self.reflection_pad, self.conv_post = h.reflection_pad, h.conv_post
        self.lrelu_slope, self.num_upsamples, self.num_kernels = h.lrelu_slope, h.num_upsamples, h.num_kernels

    def forward(self, x, s_stft):
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, self.lrelu_slope)
            x = self.ups[i](x)
            if i == self.num_upsamples - 1:
                x = self.reflection_pad(x)
            si = self.source_downs[i](s_stft)
            si = self.source_resblocks[i](si)
            x = x + si
            xs = None
            for j in range(self.num_kernels):
                r = self.resblocks[i * self.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        return self.conv_post(x)


class Hift(nn.Module):
    """Upstream HiFTGenerator.forward with explicit randomness, a frame-wrapped NSF phase and a
    convolutional STFT / iSTFT (see the module docstring). ``half`` runs the decoder (conv_pre,
    upsamplers, resblocks, source fusion, conv_post) in fp16; the F0 predictor, the source and both
    transforms stay fp32.

    ``forward(mel, harm_phase, sine_noise)`` is the whole of upstream ``forward(x)`` (the ``hift``
    graph). The same arithmetic is split for the engine's phase-continuous NSF lever: ``HiftF0``
    (mel -> f0) and ``HiftNsf`` (mel, f0, harm_phase, sine_noise -> audio, upstream
    ``forward(x, f0)``). With the F0 known before synthesis, the engine can (a) choose
    ``harm_phase`` so that the excitation's phase at the output region continues the previous
    tick's (``sampler.harm_phase_for_anchor``), and (b) synthesise only the frames it returns plus
    the decoder's receptive field (partial decode, ``sampler.Preset.hift_margin``)."""

    def __init__(self, hift, half=False):
        super().__init__()
        h = cm.strip_weight_norm(copy.deepcopy(hift))
        self.f0_predictor = h.f0_predictor
        self.l_linear = h.m_source.l_linear
        self.sine_amp = h.m_source.l_sin_gen.sine_amp
        self.noise_std = h.m_source.l_sin_gen.noise_std
        self.voiced_threshold = h.m_source.l_sin_gen.voiced_threshold
        self.sr = h.sampling_rate
        self.upp = int(np.prod([8, 8]) * h.istft_params["hop_len"])     # 256
        self.n_fft = h.istft_params["n_fft"]                            # 16
        self.hop = h.istft_params["hop_len"]                            # 4
        self.audio_limit = h.audio_limit
        self.dec = HiftDecoder(h)
        self.half = half
        if half:
            # Snake (x + sin^2(a x) / a) stays fp32, as under upstream's fp16 autocast where the
            # alpha parameters stay fp32: trained alphas go down to 1.1e-5, so in fp16 1/a overflows
            # and sin^2(a x) underflows, and the product is NaN. The exact fp32 alphas are copied
            # before ``.half()`` rounds them, and put back after.
            import types
            snakes = [m for m in self.dec.modules() if type(m).__name__ == "Snake"]
            exact = [m.alpha.detach().clone().float() for m in snakes]
            self.dec.half()
            for m, alpha in zip(snakes, exact):
                m.alpha.data = alpha
                m.forward = types.MethodType(_snake_fp32, m)
        n = self.n_fft
        win = h.stft_window.double()                                    # scipy hann, periodic
        k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]
        t = torch.arange(n, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * t / n
        # torch.stft (onesided, e^{-jwt}): real = sum w x cos, imag = -sum w x sin
        self.register_buffer("stft_basis", torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0)[:, None, :].float())
        # torch.istft (irfft of a onesided spectrum): x[t] = (1/n) sum_k a_k (Re cos - Im sin), a_0 = a_{n/2} = 1, else 2
        a = torch.full((n // 2 + 1, 1), 2.0, dtype=torch.float64)
        a[0] = 1.0
        a[-1] = 1.0
        ib = torch.cat([a * torch.cos(ang), -a * torch.sin(ang)], 0) / n * win   # [18,16], synthesis window applied
        self.register_buffer("istft_basis", ib[:, None, :].float())
        self.register_buffer("win_sq", (win ** 2)[None, None, :].float())
        self.register_buffer("harmonics", torch.arange(1, N_HARM + 1, dtype=torch.float32)[None, :, None])
        self.register_buffer("steps", torch.arange(1, self.upp + 1, dtype=torch.float32))

    def nsf_phase(self, f0):
        """Each harmonic's NSF phase in cycles (wrapped) at every sample, relative to the window's
        first sample: [1,9,256M]. ``sampler.nsf_frame_cycles`` is the same arithmetic in numpy."""
        # SineGen: theta = 2 pi frac(cumsum(h f0 / sr)) + phase, with f0 held per 256-sample frame
        inc = f0[:, None, :] * self.harmonics / self.sr                 # [1,9,M] cycles / sample
        adv = _frac(inc * self.upp)                                     # cycles / frame, wrapped
        start = _frac(torch.cumsum(adv, -1) - adv)                      # phase before each frame
        ph = _frac(start[..., None] + inc[..., None] * self.steps)      # [1,9,M,256]
        return ph.reshape(1, N_HARM, -1)

    def synth(self, mel, f0, harm_phase, sine_noise):
        """Upstream ``forward(x, f0)``: NSF source, STFT, decoder, iSTFT. ``harm_phase`` [1,9,1] is
        each harmonic's phase (radians) at the first sample of the window."""
        ph = self.nsf_phase(f0)
        sine = self.sine_amp * torch.sin(2 * math.pi * ph + harm_phase)
        f0_up = f0[:, :, None].expand(-1, -1, self.upp).reshape(1, 1, -1)
        uv = (f0_up > self.voiced_threshold).to(torch.float32)
        noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
        sine_waves = sine * uv + noise_amp * sine_noise                 # [1,9,S]
        s = torch.tanh(self.l_linear(sine_waves.transpose(1, 2))).transpose(1, 2)   # [1,1,S]
        # STFT (n_fft 16, hop 4, periodic hann, center + reflect) as a strided convolution
        s_stft = F.conv1d(F.pad(s, (self.n_fft // 2, self.n_fft // 2), mode="reflect"), self.stft_basis, stride=self.hop)
        if self.half:
            x = self.dec(mel.half(), s_stft.half()).float()
        else:
            x = self.dec(mel, s_stft)
        nb = self.n_fft // 2 + 1
        magnitude = torch.clamp(torch.exp(x[:, :nb, :]), max=1e2)
        phase = torch.sin(x[:, nb:, :])
        spec = torch.cat([magnitude * torch.cos(phase), magnitude * torch.sin(phase)], 1)   # [1,18,F]
        ola = F.conv_transpose1d(spec, self.istft_basis, stride=self.hop)                  # [1,1,16+4(F-1)]
        env = F.conv_transpose1d(torch.ones_like(spec[:, :1, :]), self.win_sq, stride=self.hop)
        pad = self.n_fft // 2
        y = (ola / env)[:, 0, pad:-pad]
        return torch.clamp(y, -self.audio_limit, self.audio_limit)

    def forward(self, mel, harm_phase, sine_noise):
        f0 = self.f0_predictor(mel)                                     # [1,M]
        return self.synth(mel, f0, harm_phase, sine_noise), f0


class HiftF0(nn.Module):
    """HiFT's F0 predictor alone (``hift_f0``: mel [1,80,M] -> f0 [1,M] Hz), always fp32. It runs
    over the whole tick mel, so the F0 of every frame matches upstream's full-window prediction."""

    def __init__(self, hift):
        super().__init__()
        self.f0_predictor = cm.strip_weight_norm(copy.deepcopy(hift.f0_predictor))

    def forward(self, mel):
        return self.f0_predictor(mel)


class HiftNsf(Hift):
    """HiFT synthesis from a given F0 (``hift_nsf``: mel [1,80,W], f0 [1,W], harm_phase [1,9,1],
    sine_noise [1,9,256W] -> audio [1,256W]): upstream ``forward(x, f0)``. Fed a slice of the tick
    (the returned frames plus ``hift_margin`` frames of left context) with ``harm_phase`` advanced
    to the slice's first sample, its output over the returned frames equals the full-window
    decode; ``half`` as in ``Hift``."""

    def forward(self, mel, f0, harm_phase, sine_noise):
        return self.synth(mel, f0, harm_phase, sine_noise)


class Mel(nn.Module):
    """Upstream ``modules.audio.mel_spectrogram`` (22.05 kHz, n_fft 1024, hop 256, 80 bins, fmin 0,
    fmax 8000, center False after a (n_fft - hop)/2 reflect pad), STFT as a strided convolution."""

    def __init__(self, mel_args):
        super().__init__()
        from librosa.filters import mel as librosa_mel_fn
        n = mel_args["n_fft"]
        self.pad = (n - mel_args["hop_size"]) // 2
        self.hop = mel_args["hop_size"]
        win = torch.hann_window(mel_args["win_size"]).double()
        k = torch.arange(n // 2 + 1, dtype=torch.float64)[:, None]
        t = torch.arange(n, dtype=torch.float64)[None, :]
        ang = 2 * math.pi * k * t / n
        self.register_buffer("basis", torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0)[:, None, :].float())
        mel = librosa_mel_fn(sr=mel_args["sampling_rate"], n_fft=n, n_mels=mel_args["num_mels"], fmin=mel_args["fmin"], fmax=mel_args["fmax"])
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.nb = n // 2 + 1

    def forward(self, wav22k):
        y = F.pad(wav22k[:, None, :], (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(y, self.basis, stride=self.hop)                  # [1,1026,M]
        re, im = spec[:, : self.nb], spec[:, self.nb:]
        mag = torch.sqrt(re * re + im * im + 1e-9)
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, mag), min=1e-5))


def _exact_seg_pooling(self, x, seg_len=100, stype="avg"):
    """CAMLayer.seg_pooling (avg over 100-frame segments, ceil_mode) without ``AveragePool``: ONNX
    Runtime divides the last, partial segment by ``seg_len`` where PyTorch divides by its valid
    frames (measured: CAM++ exported with AveragePool gives 4.5 dB SNR on a 3 s reference). Segment
    means as a matmul with a one-hot segment matrix, then broadcast back to every frame."""
    pos = torch.cumsum(torch.ones_like(x[0, 0, :]), 0) - 1.0
    seg_id = torch.floor(pos / seg_len)
    n_seg = (seg_id[-1] + 1.0).to(torch.int64)
    segs = torch.arange(n_seg, device=x.device).to(x.dtype)
    a = (seg_id[:, None] == segs[None, :]).to(x.dtype)                  # [T, n_seg]
    means = torch.matmul(x, a) / a.sum(0)                                # [B, C, n_seg]
    return torch.matmul(means, a.transpose(0, 1))                        # [B, C, T]


class CampPlus(nn.Module):
    def __init__(self, campplus):
        super().__init__()
        import types
        for m in campplus.modules():
            if type(m).__name__ == "CAMLayer":
                m.seg_pooling = types.MethodType(_exact_seg_pooling, m)
        self.m = campplus

    def forward(self, fbank):
        return self.m(fbank)


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------

def graph_specs(up, half):
    """(name, module, example args, input names, output names, dynamic axes) per graph."""
    torch.manual_seed(0)
    model = up["model"]
    L, P, M = 120, 80, 40
    dit = copy.deepcopy(model.cfm.estimator)
    specs = {
        "xlsr": (Xlsr(copy.deepcopy(up["xlsr"]), half), (torch.randn(1, 16000 * 2) * 0.1,),
                 ["wav16k"], ["feats"], {"wav16k": {1: "N"}, "feats": {1: "T"}}),
        "lr": (LengthRegulator(copy.deepcopy(model.length_regulator)), (torch.randn(1, 99, 1024), torch.tensor([171])),
               ["feats", "n_frames"], ["cond"], {"feats": {1: "T"}, "cond": {1: "n"}}),
        "dit": (Estimator(dit, half), (torch.randn(2, 80, L), torch.randn(2, 80, L), torch.tensor([L, L]), torch.tensor([0.3, 0.3]),
                                        torch.randn(2, 192), torch.randn(2, L, 384)),
                ["x", "prompt_x", "x_lens", "t", "style", "cond"], ["v"],
                {"x": {0: "B", 2: "L"}, "prompt_x": {0: "B", 2: "L"}, "x_lens": {0: "B"}, "t": {0: "B"}, "style": {0: "B"},
                 "cond": {0: "B", 1: "L"}, "v": {0: "B", 2: "L"}}),
        "dit_step": (DitStep(copy.deepcopy(model.cfm.estimator), half),
                     (torch.randn(1, 80, L), torch.randn(1, 80, L), torch.randn(1, L, 384), torch.randn(1, 192),
                      torch.tensor([0.1]), torch.tensor([0.1]), torch.tensor([0.7]), torch.tensor([P])),
                     ["x", "prompt_x", "cond", "style", "t", "dt", "cfg_rate", "prompt_len"], ["x_next"],
                     {"x": {2: "L"}, "prompt_x": {2: "L"}, "cond": {1: "L"}, "x_next": {2: "L"}}),
        "dit_step1": (DitStep1(copy.deepcopy(model.cfm.estimator), half),
                      (torch.randn(1, 80, L), torch.randn(1, 80, L), torch.randn(1, L, 384), torch.randn(1, 192),
                       torch.tensor([0.1]), torch.tensor([0.1]), torch.tensor([P])),
                      ["x", "prompt_x", "cond", "style", "t", "dt", "prompt_len"], ["x_next"],
                      {"x": {2: "L"}, "prompt_x": {2: "L"}, "cond": {1: "L"}, "x_next": {2: "L"}}),
        "hift": (Hift(up["hift"], half), (torch.randn(1, 80, M) - 5, torch.zeros(1, N_HARM, 1), torch.randn(1, N_HARM, M * 256)),
                 ["mel", "harm_phase", "sine_noise"], ["audio", "f0"],
                 {"mel": {2: "M"}, "sine_noise": {2: "S"}, "audio": {1: "S"}, "f0": {1: "M"}}),
        "hift_f0": (HiftF0(up["hift"]), (torch.randn(1, 80, M) - 5,), ["mel"], ["f0"], {"mel": {2: "M"}, "f0": {1: "M"}}),
        "hift_nsf": (HiftNsf(up["hift"], half), (torch.randn(1, 80, M) - 5, torch.full((1, M), 180.0), torch.zeros(1, N_HARM, 1),
                                                 torch.randn(1, N_HARM, M * 256)),
                     ["mel", "f0", "harm_phase", "sine_noise"], ["audio"],
                     {"mel": {2: "W"}, "f0": {1: "W"}, "sine_noise": {2: "S"}, "audio": {1: "S"}}),
        "campplus": (CampPlus(copy.deepcopy(up["campplus"])), (torch.randn(1, 300, 80),), ["fbank"], ["style"], {"fbank": {1: "F"}}),
        "mel": (Mel(up["mel_args"]), (torch.randn(1, 22050) * 0.1,), ["wav22k"], ["mel"], {"wav22k": {1: "S"}, "mel": {2: "M"}}),
    }
    return specs


HALF_GRAPHS = ("xlsr", "dit", "dit_step", "dit_step1", "hift", "hift_nsf")


def _stamp(path, name, half):
    import onnx
    m = onnx.load(path)
    onnx.checker.check_model(m)
    del m.metadata_props[:]
    m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.source", value="Plachtaa/seed-vc@51383efd seed-uvit-tat-xlsr-tiny"))
    m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.graph", value=f"seed_vc/{name}{' fp16' if half else ''}"))
    onnx.save(m, path)


def _to_device(mod, args, dev):
    mod = mod.to(dev)
    for m in mod.modules():  # the DiT's RoPE table and causal mask are plain attributes, not buffers
        for attr in ("freqs_cis", "causal_mask"):
            if torch.is_tensor(getattr(m, attr, None)):
                setattr(m, attr, getattr(m, attr).to(dev))
    return mod, tuple(a.to(dev) for a in args)


def export(names, fp16=True, out=None):
    """fp32 graphs traced on the CPU; fp16 graphs (fp16 inside, fp32 I/O) traced on CUDA from
    ``.half()`` copies, since several fp16 kernels exist only there. Precision islands kept in fp32:
    XLS-R's input normalisation, the DiT timestep sinusoid (exact fp32 ``freqs`` and ``t``), the
    sampler update in ``dit_step``, and HiFT's F0 predictor, NSF source, Snake activations (exact
    fp32 ``alpha``) and both transforms (only its decoder convolutions are fp16)."""
    up = load_upstream("cpu")
    written = []
    for half in ([False, True] if fp16 else [False]):
        if half and not torch.cuda.is_available():
            print("fp16 graphs need CUDA for tracing; skipped")
            break
        specs = graph_specs(up, half)
        for name in names:
            if half and name not in HALF_GRAPHS:
                continue
            mod, args, ins, outs, dyn = specs[name]
            mod.eval()
            if half:
                mod, args = _to_device(mod, args, "cuda")
            path = os.path.join(out or OUT, f"{name}{'.fp16' if half else ''}.onnx")
            with torch.no_grad():
                # torch 2.3's constant folding turns folded fp16 weights (the transposed Linear
                # weights) into fp32 initializers; ONNX Runtime folds those transposes at load anyway.
                torch.onnx.export(mod, args, path, input_names=ins, output_names=outs, dynamic_axes=dyn,
                                  opset_version=17, do_constant_folding=not half)
            _stamp(path, name, half)
            written.append((path, os.path.getsize(path)))
            print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
            if half:
                del mod
                torch.cuda.empty_cache()
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="xlsr,lr,dit,dit_step,dit_step1,hift,hift_f0,hift_nsf,campplus,mel")
    ap.add_argument("--no-fp16", action="store_true")
    ap.add_argument("--out", default=None, help="write the graphs here instead of sigil-dev-cache/vc-families/seed_vc")
    args = ap.parse_args()
    if args.out:
        os.makedirs(args.out, exist_ok=True)
    export([n.strip() for n in args.only.split(",") if n.strip()], fp16=not args.no_fp16, out=args.out)


if __name__ == "__main__":
    main()
