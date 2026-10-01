# SPDX-License-Identifier: AGPL-3.0-or-later
"""Sigil split-graph wrappers over the vendored RVC modules (the graphs ``export_rvc.py`` exports).

RVC's ``SynthesizerTrnMs768NSFsid.infer`` is split into three graphs so the real-time engine can
run each part over exactly the frames it needs, at fixed shapes:

- ``enc``: content features + coarse pitch -> prior ``m_p`` / ``logs_p`` over the whole context T
  (the text encoder is a non-causal transformer, so it sees the whole context).
- ``flowdec``: flow (reverse) + NSF HiFi-GAN decoder over the *return window* only (partial decode).
  Its inputs cover one window of ``F`` frames; the first ``FLOW_MARGIN`` frames feed only the flow's
  receptive field and the next ``dec_margin`` frames only the decoder's, so the returned audio (the
  last ``F - FLOW_MARGIN - dec_margin`` frames) equals the same frames of a full decode. The random
  parts of RVC are inputs, so the engine can anchor them in time (identical noise wherever two ticks
  overlap): ``z_noise`` (the prior's Gaussian noise, already multiplied by the noise scale, 0.66666
  in RVC), ``src_noise`` (the NSF source's unit Gaussian noise, one value per output sample) and
  ``phase0`` (the NSF sine phase, in cycles, at the first sample of the window), which makes the
  excitation phase-continuous across ticks. The excitation (sine, voicing, noise, source linear) is
  always computed in fp32.
- ``knn``: exact retrieval blend against the voice's feature index (RVC ``index_rate`` blending with
  k = 8 and 1/d^2 weights), a coarse top-``KNN_COARSE_K`` pass on ``|x|^2 - 2 q.x`` followed by an
  exact fp32 re-rank on ``|q - x|^2``, so an fp16 index gives the same neighbours as fp32.

All three are mask-free (the real-time context is always full) and use only data-driven index math,
so every length dimension stays dynamic in the ONNX graph; the engine pins them per preset with ONNX
Runtime free-dimension overrides. The fp16 variants keep fp32 inputs and outputs and cast inside.

The other vocoders' flowdecs (P3-A; layout version 2) keep that contract and differ only where their
decoder does: ``SigilFlowDecNoF0`` (f0-less voices: no source inputs), ``SigilFlowDecMRF`` (Applio's
MRF HiFi-GAN: 9 harmonics, ``src_noise`` [1, S, 9]) and ``SigilFlowDecRefineGAN`` (Applio's
RefineGAN: linear f0 interpolation, AdaIN noise from a pool indexed per absolute frame, and its main
path decoded over its own shorter margin). Both Applio vocoders take the frame-level phase of every
window frame (``phase`` [1, F], ``FRAME_PHASE``) instead of ``phase0``, so no phase accumulates in fp32
inside the graph.

Nothing here changes RVC's arithmetic; ``parity_test.py`` checks every graph against the upstream
``infer`` path within tolerance.
"""

import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from . import modules

# Frames of left context the flow needs: 4 coupling layers x WN(kernel 5, 3 layers, dilation 1),
# each +-6 frames (RVC's own realtime path uses the same 24).
FLOW_MARGIN = 24
KNN_K = 8
KNN_COARSE_K = 64
KNN_EPS = 1e-8


# ---------------------------------------------------------------------------------------------
# enc
# ---------------------------------------------------------------------------------------------


def _positions(x):
    """0..T-1 as int64, derived from the data so T stays a dynamic dimension (x: [1, C, T])."""
    ones = torch.ones_like(x[0, 0, :], dtype=torch.int64)
    return torch.cumsum(ones, 0) - 1


def _rel_attention(attn, x):
    """``attentions.MultiHeadAttention`` self-attention with windowed relative positions, no mask.

    Same arithmetic as upstream's pad/reshape relative-position trick, written with gathers so the
    sequence length is not baked into the graph:
    scores[i, j] += (q_i / sqrt(dk)) . E_k[j - i + W] and out_i += sum_j p[i, j] E_v[j - i + W] for
    |j - i| <= W.
    """
    heads, dk, w = attn.n_heads, attn.k_channels, attn.window_size
    q = attn.conv_q(x)
    k = attn.conv_k(x)
    v = attn.conv_v(x)
    q = q.view(1, heads, dk, -1).transpose(2, 3)
    k = k.view(1, heads, dk, -1).transpose(2, 3)
    v = v.view(1, heads, dk, -1).transpose(2, 3)

    qs = q / math.sqrt(dk)
    scores = torch.matmul(qs, k.transpose(-2, -1))  # [1, H, T, T]

    pos = _positions(x)  # [T]
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # [i, j] = j - i
    near = diff.abs() <= w
    idx = torch.where(near, diff + w, torch.zeros_like(diff))
    rel_k = torch.matmul(qs, attn.emb_rel_k.unsqueeze(0).transpose(-2, -1))  # [1, H, T, 2W+1]
    local = torch.gather(rel_k, 3, idx.unsqueeze(0).unsqueeze(0).expand(1, heads, -1, -1))
    scores = scores + torch.where(near, local, torch.zeros_like(local))

    p_attn = F.softmax(scores, dim=-1)
    out = torch.matmul(p_attn, v)  # [1, H, T, dk]

    offsets = torch.arange(2 * w + 1, dtype=torch.int64, device=x.device) - w  # [2W+1]
    cols = pos.unsqueeze(1) + offsets.unsqueeze(0)  # [T, 2W+1]: column j = i + r - W
    inside = (cols >= 0) & (cols <= pos[-1])
    cols = torch.where(inside, cols, torch.zeros_like(cols))
    rel_w = torch.gather(p_attn, 3, cols.unsqueeze(0).unsqueeze(0).expand(1, heads, -1, -1))
    rel_w = torch.where(inside, rel_w, torch.zeros_like(rel_w))  # [1, H, T, 2W+1]
    out = out + torch.matmul(rel_w, attn.emb_rel_v.unsqueeze(0))

    out = out.transpose(2, 3).reshape(1, heads * dk, -1)
    return attn.conv_o(out)


def _ffn(ffn, x):
    """``attentions.FFN`` (non-causal, relu) without the mask."""
    assert not ffn.causal and not ffn.is_activation
    pad = [(ffn.kernel_size - 1) // 2, ffn.kernel_size // 2]
    y = x if ffn.kernel_size == 1 else F.pad(x, pad)
    y = torch.relu(ffn.conv_1(y))
    y = y if ffn.kernel_size == 1 else F.pad(y, pad)
    return ffn.conv_2(y)


def _encoder(enc, x):
    """``attentions.Encoder.forward`` with an all-ones mask (a full real-time context)."""
    layers = zip(enc.attn_layers, enc.norm_layers_1, enc.ffn_layers, enc.norm_layers_2)
    for attn, norm_1, ffn, norm_2 in layers:
        x = norm_1(x + _rel_attention(attn, x))
        x = norm_2(x + _ffn(ffn, x))
    return x


class SigilEnc(nn.Module):
    """``enc_p`` of RVC: phone [1, T, E] f32 (+ pitch [1, T] int64) -> m_p, logs_p [1, C, T] f32."""

    def __init__(self, enc_p, f0=True, dtype=torch.float32):
        super().__init__()
        self.enc_p = enc_p
        self.f0 = f0
        self.dtype = dtype

    def forward(self, phone, pitch=None):
        e = self.enc_p
        x = e.emb_phone(phone.to(self.dtype))
        if self.f0:
            x = x + e.emb_pitch(pitch)
        x = x * math.sqrt(e.hidden_channels)
        x = F.leaky_relu(x, 0.1)
        x = torch.transpose(x, 1, -1)  # [1, H, T]
        x = _encoder(e.encoder, x)
        stats = e.proj(x)
        m, logs = torch.split(stats, e.out_channels, dim=1)
        return m.float(), logs.float()


# ---------------------------------------------------------------------------------------------
# flowdec
# ---------------------------------------------------------------------------------------------


def _wn(wn, x, g):
    """``modules.WN.forward`` without the mask."""
    hidden = wn.hidden_channels
    g = wn.cond_layer(g)
    output = None
    for i in range(wn.n_layers):
        x_in = wn.in_layers[i](x)
        in_act = x_in + g[:, i * 2 * hidden : (i + 1) * 2 * hidden, :]
        acts = torch.tanh(in_act[:, :hidden, :]) * torch.sigmoid(in_act[:, hidden:, :])
        res_skip = wn.res_skip_layers[i](acts)
        if i < wn.n_layers - 1:
            x = x + res_skip[:, :hidden, :]
            skip = res_skip[:, hidden:, :]
        else:
            skip = res_skip
        output = skip if output is None else output + skip
    return output


def _coupling_reverse(layer, x, g):
    """``modules.ResidualCouplingLayer`` (mean_only) in reverse, without the mask."""
    assert layer.mean_only
    half = layer.half_channels
    x0 = x[:, :half, :]
    x1 = x[:, half:, :]
    h = layer.pre(x0)
    h = _wn(layer.enc, h, g)
    m = layer.post(h)
    return torch.cat([x0, x1 - m], 1)


def _flow_reverse(flow, x, g):
    for f in reversed(flow.flows):
        if isinstance(f, modules.Flip):
            x = torch.flip(x, [1])
        else:
            x = _coupling_reverse(f, x, g)
    return x


def excitation(m_source, sr, upp, f0, phase0, src_noise):
    """NSF source (``SineGen`` + ``SourceModuleHnNSF``), fp32, with the sine phase offset and the
    Gaussian noise as inputs: f0 [1, F] Hz, phase0 [1, 1] cycles, src_noise [1, F*upp] -> [1, 1, F*upp].

    Same arithmetic as upstream ``SineGen._f02sine`` (harmonic_num = 0, so upstream's random
    initial phase is always 0 and is replaced by ``phase0``).
    """
    gen = m_source.l_sin_gen
    assert gen.dim == 1, "templates cover harmonic_num = 0 (every RVC NSF generator)"
    f0 = f0.float().unsqueeze(-1)  # [1, F, 1]
    rad = f0 / sr * torch.arange(1, upp + 1, dtype=torch.int64, device=f0.device)  # [1, F, upp]
    rad2 = torch.fmod(rad[..., -1:] + 0.5, 1.0) - 0.5
    rad_acc = rad2.cumsum(dim=1).fmod(1.0)
    rad_acc = torch.cat([torch.zeros_like(rad_acc[:, :1, :]), rad_acc[:, :-1, :]], dim=1)
    rad = (rad + rad_acc).reshape(1, -1, 1) + phase0.float().reshape(1, 1, 1)
    sine = torch.sin(2 * np.pi * rad) * gen.sine_amp
    uv = (f0 > gen.voiced_threshold).float()  # [1, F, 1]
    uv = uv.expand(-1, -1, upp).reshape(1, -1, 1)
    noise_amp = uv * gen.noise_std + (1 - uv) * gen.sine_amp / 3
    noise = noise_amp * src_noise.float().reshape(1, -1, 1)
    sine = sine * uv + noise
    lin = m_source.l_linear
    har = torch.tanh(F.linear(sine, lin.weight.float(), lin.bias.float()))  # [1, S, 1]
    return har.transpose(1, 2)


def cast_weights_to_input_(module):
    """Make every Conv1d / ConvTranspose1d / Linear under ``module`` compute in its *input's* dtype
    (weights cast in the graph). With fp16 weights this lets a region run in fp32 while the weights
    stay fp16 in storage (one weights.f16.bin); a no-op where the dtypes already match."""

    def conv(self, x):
        return self._conv_forward(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    def conv_t(self, x, output_size=None):
        return F.conv_transpose1d(
            x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype),
            self.stride, self.padding, self.output_padding, self.groups, self.dilation,
        )

    def linear(self, x):
        return F.linear(x, self.weight.to(x.dtype), None if self.bias is None else self.bias.to(x.dtype))

    import types

    for m in module.modules():
        if isinstance(m, nn.ConvTranspose1d):
            m.forward = types.MethodType(conv_t, m)
        elif isinstance(m, nn.Conv1d):
            m.forward = types.MethodType(conv, m)
        elif isinstance(m, nn.Linear):
            m.forward = types.MethodType(linear, m)
    return module


def _decoder(dec, z, har, g, fp32_stages=()):
    """``models.GeneratorNSF.forward`` with a precomputed source ``har`` [1, 1, samples] (fp32).
    Stages listed in ``fp32_stages`` (upsample + its resblocks) compute in fp32 whatever the
    weights' dtype (see ``cast_weights_to_input_``)."""
    dt = z.dtype
    source = {dt: har.to(dt), torch.float32: har}
    x = dec.conv_pre(z)
    x = x + dec.cond(g)
    nk = dec.num_kernels
    for i in range(dec.num_upsamples):
        sd = torch.float32 if i in fp32_stages else dt
        x = F.leaky_relu(x, dec.lrelu_slope).to(sd)
        x = dec.ups[i](x)
        x = x + dec.noise_convs[i](source[sd])
        xs = None
        for j in range(nk):
            r = dec.resblocks[i * nk + j](x)
            xs = r if xs is None else xs + r
        x = xs / nk
    x = F.leaky_relu(x.to(dt))
    x = dec.conv_post(x)
    return torch.tanh(x)


class SigilFlowDec(nn.Module):
    """flow (reverse) + NSF decoder over one window of F frames; returns the last
    ``F - FLOW_MARGIN - dec_margin`` frames of audio [1, R * upp] f32 (see the module docs)."""

    def __init__(self, flow, dec, sr, dec_margin, dtype=torch.float32, fp32_flow=False, fp32_stages=()):
        super().__init__()
        self.flow = flow
        self.dec = dec
        self.sr = int(sr)
        self.upp = int(dec.upp)
        self.dec_margin = int(dec_margin)
        self.dtype = dtype
        # Mixed precision for fp16 weights (see cast_weights_to_input_): the flow and / or decoder
        # stages computed in fp32.
        self.fp32_flow = fp32_flow
        self.fp32_stages = tuple(fp32_stages)

    def forward(self, m_p, logs_p, z_noise, g, f0, phase0, src_noise):
        dt = self.dtype
        z_p = m_p + torch.exp(logs_p) * z_noise  # fp32
        fdt = torch.float32 if self.fp32_flow else dt
        z = _flow_reverse(self.flow, z_p.to(fdt), g.to(fdt))
        z = z[:, :, FLOW_MARGIN:].to(dt)
        gg = g.to(dt)
        har = excitation(self.dec.m_source, self.sr, self.upp, f0, phase0, src_noise)
        har = har[:, :, FLOW_MARGIN * self.upp :]
        audio = _decoder(self.dec, z, har, gg, self.fp32_stages)  # [1, 1, (F - FLOW_MARGIN) * upp]
        audio = audio[:, 0, self.dec_margin * self.upp :]
        return audio.float()


# ---------------------------------------------------------------------------------------------
# flowdec for f0-less voices (RVC `Generator`, no NSF source)
# ---------------------------------------------------------------------------------------------


def _decoder_nof0(dec, z, g, fp32_stages=()):
    """``models.Generator.forward`` (the f0-less HiFi-GAN of ``SynthesizerTrnMs*NSFsid_nono``)."""
    dt = z.dtype
    x = dec.conv_pre(z)
    x = x + dec.cond(g)
    nk = dec.num_kernels
    for i in range(dec.num_upsamples):
        sd = torch.float32 if i in fp32_stages else dt
        x = F.leaky_relu(x, modules.LRELU_SLOPE).to(sd)
        x = dec.ups[i](x)
        xs = None
        for j in range(nk):
            r = dec.resblocks[i * nk + j](x)
            xs = r if xs is None else xs + r
        x = xs / nk
    x = F.leaky_relu(x.to(dt))
    x = dec.conv_post(x)
    return torch.tanh(x)


class SigilFlowDecNoF0(nn.Module):
    """flow (reverse) + f0-less HiFi-GAN over one window of F frames:
    m_p, logs_p, z_noise [1, C, F], g [1, G, 1] -> audio [1, (F - FLOW_MARGIN - dec_margin) * upp]."""

    def __init__(self, flow, dec, upp, dec_margin, dtype=torch.float32, fp32_flow=False, fp32_stages=()):
        super().__init__()
        self.flow = flow
        self.dec = dec
        self.upp = int(upp)
        self.dec_margin = int(dec_margin)
        self.dtype = dtype
        self.fp32_flow = fp32_flow
        self.fp32_stages = tuple(fp32_stages)

    def forward(self, m_p, logs_p, z_noise, g):
        dt = self.dtype
        z_p = m_p + torch.exp(logs_p) * z_noise  # fp32
        fdt = torch.float32 if self.fp32_flow else dt
        z = _flow_reverse(self.flow, z_p.to(fdt), g.to(fdt))
        z = z[:, :, FLOW_MARGIN:].to(dt)
        audio = _decoder_nof0(self.dec, z, g.to(dt), self.fp32_stages)
        return audio[:, 0, self.dec_margin * self.upp :].float()


# The frame-level NSF phase the Applio-vocoder templates take as ``phase`` [1, F]: for each window
# frame j, frac(phase0 + sum over the frames before j of wrap(upp * f0 / sr)), wrap(x) =
# fmod(x + 0.5, 1) - 0.5, i.e. the HiFi-GAN templates' ``phase0`` carried to every frame. The engine
# computes it in float64 from its f0 history (parity_test.nsf_phase is the reference), so no fp32
# phase accumulates inside the graphs (it drifts by ~1e-5 cycles over a few seconds, which a trained
# RefineGAN amplifies to the size of the fp32 gate).
FRAME_PHASE = "frac(phase0 + sum(wrap(upp * f0 / sr)) over the earlier frames), per window frame, computed in f64"


# ---------------------------------------------------------------------------------------------
# flowdec for Applio's MRF HiFi-GAN (9-harmonic NSF source)
# ---------------------------------------------------------------------------------------------


def excitation_harmonics(l_linear, sr, upp, f0, phase, src_noise, phase_ini):
    """Applio's MRF source over a window, fp32: f0 [1, F] Hz (repeated per sample), phase [1, F]
    (the fundamental's NSF phase in cycles at each window frame's first sample, see ``FRAME_PHASE``),
    src_noise [1, F*upp, H] unit normals, phase_ini [H] cycles (constant) -> [1, 1, F*upp].

    Applio accumulates ``(k * f0 / sr) % 1`` per sample for harmonic k and adds its initial phase at
    the first sample, so harmonic k's phase at sample i of frame j is
    ``phase_ini[k] + k * (phase[j] + (i + 1) * f0[j] / sr)`` (mod 1). Nothing accumulates across
    frames inside the graph, so fp32 keeps its precision however long the window is.
    """
    h = phase_ini.shape[0]
    f0 = f0.float().unsqueeze(-1)  # [1, F, 1]
    ramp = torch.arange(1, upp + 1, dtype=torch.int64, device=f0.device)
    phi = torch.fmod(phase.float().unsqueeze(-1) + f0 / sr * ramp, 1.0).reshape(1, -1, 1)  # [1, S, 1]
    k = torch.arange(1, h + 1, dtype=torch.int64, device=f0.device).reshape(1, 1, -1)
    ph = torch.fmod(phi * k + phase_ini.float().reshape(1, 1, -1), 1.0)  # [1, S, H]
    sine = torch.sin(2 * np.pi * ph) * 0.1
    uv = (f0 > 0).float().expand(-1, -1, upp).reshape(1, -1, 1)
    noise_amp = uv * 0.003 + (1 - uv) * (0.1 / 3)
    sine = sine * uv + noise_amp * src_noise.float()
    har = torch.tanh(F.linear(sine, l_linear.weight.float(), l_linear.bias.float()))  # [1, S, 1]
    return har.transpose(1, 2)


def _decoder_mrf(dec, z, har, g, fp32_stages=()):
    """``HiFiGANMRFGenerator.forward`` (Applio) with a precomputed fp32 source ``har``."""
    from .vocoders import MRF_SLOPE, mrf_block

    dt = z.dtype
    source = {dt: har.to(dt), torch.float32: har}
    x = dec.conv_pre(z)
    x = x + dec.cond(g)
    nk = dec.num_kernels
    for i in range(len(dec.upsamples)):
        sd = torch.float32 if i in fp32_stages else dt
        x = F.leaky_relu(x, MRF_SLOPE).to(sd)
        x = dec.upsamples[i](x)
        x = x + dec.noise_convs[i](source[sd])
        xs = None
        for block in dec.mrfs[i]:
            r = mrf_block(block, x)
            xs = r if xs is None else xs + r
        x = xs / nk
    x = F.leaky_relu(x.to(dt))
    return torch.tanh(dec.conv_post(x))


class SigilFlowDecMRF(nn.Module):
    """flow (reverse) + MRF HiFi-GAN over one window of F frames. Inputs as ``SigilFlowDec`` except
    ``phase`` [1, F] (``FRAME_PHASE``, instead of ``phase0``) and ``src_noise`` [1, F*upp, H] (one unit
    normal per sample and harmonic)."""

    def __init__(self, flow, dec, dec_margin, phase_ini, dtype=torch.float32, fp32_flow=False, fp32_stages=()):
        super().__init__()
        self.flow = flow
        self.dec = dec
        self.sr = int(dec.sample_rate)
        self.upp = int(dec.upp)
        self.dec_margin = int(dec_margin)
        self.register_buffer("phase_ini", torch.tensor(phase_ini, dtype=torch.float32), persistent=False)
        self.dtype = dtype
        self.fp32_flow = fp32_flow
        self.fp32_stages = tuple(fp32_stages)

    def forward(self, m_p, logs_p, z_noise, g, f0, phase, src_noise):
        dt = self.dtype
        z_p = m_p + torch.exp(logs_p) * z_noise
        fdt = torch.float32 if self.fp32_flow else dt
        z = _flow_reverse(self.flow, z_p.to(fdt), g.to(fdt))
        z = z[:, :, FLOW_MARGIN:].to(dt)
        har = excitation_harmonics(self.dec.m_source.l_linear, self.sr, self.upp, f0, phase, src_noise, self.phase_ini)
        har = har[:, :, FLOW_MARGIN * self.upp :]
        audio = _decoder_mrf(self.dec, z, har, g.to(dt), self.fp32_stages)
        return audio[:, 0, self.dec_margin * self.upp :].float()


# ---------------------------------------------------------------------------------------------
# flowdec for Applio's RefineGAN
# ---------------------------------------------------------------------------------------------


def refinegan_excitation(merge_weight, sr, upp, f0, phase, src_noise, start):
    """RefineGAN's source over window frames [start, F), fp32: -> [1, 1, (F - start) * upp].

    Applio interpolates f0 linearly to the sample rate (half-pixel, edge-clamped) and accumulates
    ``f0 / sr`` per sample. The window's frame ``start - 1`` is the interpolation's left context, so
    every sample from frame ``start`` on equals the whole-utterance interpolation. Summing that
    interpolation over whole frames telescopes: the phase before frame j's first sample is
    ``phase[j] + upp / (8 sr) * (f0[j] - f0[j - 1])``, with ``phase`` the frame-level NSF phase every
    template's engine carries (``FRAME_PHASE``). Nothing accumulates across frames inside the graph.
    """
    f0 = f0.float()
    # The interpolation in closed form (no Resize op, whose coordinate arithmetic differs between
    # execution providers): sample i of frame j sits at d_i = (i + 0.5) / upp - 0.5 frames from the
    # frame centre, so f0(j, i) = f0[j] + min(d_i, 0) (f0[j] - f0[j-1]) + max(d_i, 0) (f0[j+1] - f0[j])
    # (edge-clamped at the window end, like the whole-utterance interpolation at the utterance end),
    # and the within-frame running sum of f0 has the same form with the running sums of d_i.
    fd = f0[:, start:].unsqueeze(-1)  # [1, Fd, 1]
    d_prev = fd - f0[:, start - 1 : -1].unsqueeze(-1)
    d_next = torch.cat([f0[:, start + 1 :], f0[:, -1:]], dim=1).unsqueeze(-1) - fd
    d = (np.arange(upp, dtype=np.float64) + 0.5) / upp - 0.5
    neg, pos = np.minimum(d, 0.0), np.maximum(d, 0.0)
    w_neg = torch.from_numpy(neg.astype(np.float32)).to(f0.device)
    w_pos = torch.from_numpy(pos.astype(np.float32)).to(f0.device)
    c_neg = torch.from_numpy(np.cumsum(neg).astype(np.float32)).to(f0.device)
    c_pos = torch.from_numpy(np.cumsum(pos).astype(np.float32)).to(f0.device)
    ramp = torch.arange(1, upp + 1, dtype=torch.int64, device=f0.device)
    f0s = fd + d_prev * w_neg + d_next * w_pos  # [1, Fd, upp] Hz
    inner = (fd * ramp + d_prev * c_neg + d_next * c_pos) / sr  # inclusive running phase in the frame (cycles)
    q = phase.float()[:, start:].unsqueeze(-1) + (upp / (8.0 * sr)) * d_prev  # phase before each frame's first sample
    ph = torch.fmod(q + inner, 1.0).reshape(1, -1, 1)  # [1, S, 1]
    sine = torch.sin(2 * np.pi * ph) * 0.1
    uv = (f0s > 0).float().reshape(1, -1, 1)
    noise_amp = uv * 0.003 + (1 - uv) * (0.1 / 3)
    sine = sine * uv + noise_amp * src_noise[:, start * upp :].float().reshape(1, -1, 1)
    har = torch.tanh(F.linear(sine, merge_weight.float()))  # [1, S, 1]
    return har.transpose(1, 2)


def linear_upsample(x, rate):
    """``F.interpolate(x, scale_factor=rate, mode="linear")`` (half-pixel, edge-clamped) for an
    integer ``rate``, in closed form: output sample i of input sample j is
    x[j] + min(d_i, 0) (x[j] - x[j-1]) + max(d_i, 0) (x[j+1] - x[j]), d_i = (i + 0.5) / rate - 0.5.
    Exact and the same on every execution provider (ONNX Resize's coordinate arithmetic is not)."""
    d = (np.arange(rate, dtype=np.float64) + 0.5) / rate - 0.5
    w_neg = torch.from_numpy(np.minimum(d, 0.0).astype(np.float32)).to(x.device, x.dtype)
    w_pos = torch.from_numpy(np.maximum(d, 0.0).astype(np.float32)).to(x.device, x.dtype)
    prev = torch.cat([x[:, :, :1], x[:, :, :-1]], dim=2)
    nxt = torch.cat([x[:, :, 1:], x[:, :, -1:]], dim=2)
    xu = x.unsqueeze(-1)
    y = xu + (xu - prev.unsqueeze(-1)) * w_neg + (nxt.unsqueeze(-1) - xu) * w_pos  # [1, C, T, rate]
    return y.reshape(x.shape[0], x.shape[1], -1)


class SigilFlowDecRefineGAN(nn.Module):
    """flow (reverse) + RefineGAN over one window of F frames, with the main path decoded only where
    the returned audio needs it.

    Two windows that end at the same frame:

    - the *source window* of ``F = FLOW_MARGIN + dec_margin + R`` frames: ``f0`` [1, F], ``phase``
      [1, F] (``FRAME_PHASE``, instead of ``phase0``), ``src_noise`` [1, F * upp] and
      ``noise_offsets`` [1, F] int64 (the AdaIN noise pool offset of each window frame, a pure
      function of its absolute frame index chosen by the engine, each in [0, pool - per_frame]);
    - the *latent window* of ``Fz = FLOW_MARGIN + main_margin + R = F - latent_offset`` frames
      (``latent_offset = dec_margin - main_margin``): ``m_p``, ``logs_p``, ``z_noise`` [1, C, Fz].

    The AdaIN noise pool is the ``adain_noise_pool`` initializer (external, ``noise.f32.bin``: ``pool``
    iid N(0, 1) f32 values the engine generates once).

    RefineGAN's source path (sine, pre_conv, then torchaudio's 64-zero-crossing sinc resampler down to
    the frame rate) sees about 0.7 s of left context, while its main path (mel conv, linear upsampling,
    parallel residual blocks) sees about 0.1 s. The source window covers ``dec_margin`` = the larger
    of the two; the main path, and the flow and text encoder feeding it, only the latent window, so
    the engine runs the content encoder over ``Fz`` frames (not ``F``: 40 instead of 117 at 32 kHz
    with R = 4) and the decoder's main path costs about 5x less per tick than a full-window decode.
    Of the source inputs, the graph reads ``f0`` from window frame ``FLOW_MARGIN - 1`` (the
    interpolation's left context), ``phase`` and ``src_noise`` from frame ``FLOW_MARGIN`` and
    ``noise_offsets`` from frame ``FLOW_MARGIN + latent_offset``; earlier values are ignored. A
    mismatched pair of window lengths (``F - Fz != latent_offset``) fails in ONNX Runtime (the main
    path's concat), never silently.
    """

    def __init__(
        self, flow, dec, dec_margin, main_margin, pool_size, dtype=torch.float32, fp32_flow=False, fp32_stages=(), fp32_ends=False,
    ):
        super().__init__()
        from .vocoders import adain_layers, sinc_resample_kernel

        self.flow = flow
        self.dec = dec
        self.sr = int(dec.sample_rate)
        self.upp = int(dec.upp)
        self.dec_margin = int(dec_margin)
        self.main_margin = int(main_margin)
        assert self.main_margin <= self.dec_margin
        # latent window = source window minus its first latent_offset frames (layout streaming.latent_offset)
        self.latent_offset = self.dec_margin - self.main_margin
        self.dtype = dtype
        self.fp32_flow = fp32_flow
        self.fp32_stages = tuple(fp32_stages)
        # fp16 mix: the main path's ends (mel conv + speaker cond, and conv_post) in fp32
        self.fp32_ends = bool(fp32_ends)
        self.resample = []
        for i, (old, new) in enumerate(dec.df0):
            assert old % new == 0, "RefineGAN reductions are integer rates"
            kernel, width = sinc_resample_kernel(old // new)
            self.register_buffer(f"resample_kernel_{i}", kernel, persistent=False)
            self.resample.append((width, old // new))
        self.layers, self.per_frame = adain_layers(dec)
        self.pool_size = int(pool_size)
        assert self.per_frame <= self.pool_size
        # Persistent, so the exporter writes it as the named initializer the exporter externalizes.
        self.register_buffer("adain_noise_pool", torch.zeros(self.pool_size, dtype=torch.float32))

    def forward(self, m_p, logs_p, z_noise, g, f0, phase, src_noise, noise_offsets):
        from .vocoders import REFINEGAN_SLOPE, adain_noise_from_pool, sinc_resample

        dt = self.dtype
        dec = self.dec
        o = self.latent_offset  # the latent window starts at source window frame o
        # flow over the latent window (source frames [o, F)): the main region [FLOW_MARGIN + o, F) plus
        # the flow's margin
        z_p = m_p + torch.exp(logs_p) * z_noise
        fdt = torch.float32 if self.fp32_flow else dt
        z = _flow_reverse(self.flow, z_p.to(fdt), g.to(fdt))
        z = z[:, :, FLOW_MARGIN:]  # [1, C, main_margin + R], in the flow's dtype (cast by the mel conv)

        # source path over the whole decoder region [FLOW_MARGIN, F), fp32 (cheap next to the main path)
        har = refinegan_excitation(dec.m_source.merge[0].weight, self.sr, self.upp, f0, phase, src_noise, FLOW_MARGIN)
        x = F.conv1d(har, dec.pre_conv.weight.float(), dec.pre_conv.bias.float(), padding=3)
        downs = []
        for i, (block, (width, rate)) in enumerate(zip(dec.downsample_blocks, self.resample)):
            x = F.leaky_relu(x, REFINEGAN_SLOPE)
            downs.append(x)
            x = sinc_resample(x, getattr(self, f"resample_kernel_{i}"), width, rate)
            x = F.conv1d(x, block.weight.float(), block.bias.float(), padding=3)

        # main path over the main region
        edt = torch.float32 if self.fp32_ends else dt
        m = dec.mel_conv(z.to(edt)) + dec.cond(g.to(edt))
        x = torch.cat([m, x[:, :, o:].to(edt)], dim=1)
        offsets = noise_offsets[0, FLOW_MARGIN + o :]
        pool = self.adain_noise_pool
        rates = dec.stage_rates()
        bases = {(s, l): base for s, l, _c, _r, base in self.layers}
        for i, (rate, res) in enumerate(zip(dec.upsample_rates, dec.upsample_conv_blocks)):
            sd = torch.float32 if i in self.fp32_stages else dt
            x = F.leaky_relu(x, REFINEGAN_SLOPE).to(sd)
            x = linear_upsample(x, rate)
            down = downs[len(downs) - 1 - i][:, :, o * rates[i] :]
            x = torch.cat([x, down.to(sd)], dim=1)
            x = self._parallel(res, x, lambda layer, y, i=i: adain_noise_from_pool(pool, offsets, bases[(i, layer)], y).to(y.dtype))
        x = F.leaky_relu(x.to(edt), REFINEGAN_SLOPE)
        x = torch.tanh(dec.conv_post(x))
        return x[:, 0, self.main_margin * self.upp :].float()

    @staticmethod
    def _parallel(prb, x, noise):
        from .vocoders import parallel_resblock

        return parallel_resblock(prb, x, noise)


# ---------------------------------------------------------------------------------------------
# knn
# ---------------------------------------------------------------------------------------------


class SigilKnn(nn.Module):
    """Exact RVC retrieval blend: feats [T, C] f32, index [N, C] (f32, or f16 in the fp16 variant),
    index_sq [N] f32 (squared row norms of ``index`` as stored), index_rate [1] f32 -> [T, C] f32.

    N must be at least ``KNN_COARSE_K``. Weights are RVC's ``(1 / d)^2`` normalised per frame, with d
    the squared L2 distance (what FAISS returns), floored at ``KNN_EPS``.
    """

    def forward(self, feats, index, index_sq, index_rate):
        q = feats.to(index.dtype)
        dot = torch.matmul(q, index.transpose(0, 1)).float()  # [T, N]
        coarse = index_sq.unsqueeze(0) - 2.0 * dot
        _, cand = torch.topk(coarse, KNN_COARSE_K, dim=1, largest=False, sorted=True)  # [T, Kc]
        cv = index[cand].float()  # [T, Kc, C]
        diff = cv - feats.unsqueeze(1)
        dist = (diff * diff).sum(-1)  # [T, Kc] exact squared L2
        d8, pos = torch.topk(dist, KNN_K, dim=1, largest=False, sorted=True)
        nb = torch.gather(cv, 1, pos.unsqueeze(-1).expand(-1, -1, cv.shape[-1]))  # [T, K, C]
        w = 1.0 / torch.clamp(d8, min=KNN_EPS)
        w = w * w
        w = w / w.sum(-1, keepdim=True)
        blended = (nb * w.unsqueeze(-1)).sum(1)
        return blended * index_rate + (1.0 - index_rate) * feats


def knn_reference(feats, index, index_rate, k=KNN_K):
    """float64 numpy reference of ``SigilKnn`` (brute force over every row)."""
    q = feats.astype(np.float64)
    x = index.astype(np.float64)
    # In float64 the expansion's cancellation error (~1e-13 relative) cannot reorder real neighbours.
    d = (q * q).sum(1, keepdims=True) - 2.0 * q @ x.T + (x * x).sum(1)[None, :]
    best = np.argsort(d, axis=1, kind="stable")[:, :k]
    dist = ((q[:, None, :] - x[best]) ** 2).sum(-1)
    w = (1.0 / np.maximum(dist, KNN_EPS)) ** 2
    w /= w.sum(1, keepdims=True)
    blended = (x[best] * w[..., None]).sum(1)
    return blended * index_rate + (1.0 - index_rate) * q, best
