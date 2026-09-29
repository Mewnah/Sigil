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
