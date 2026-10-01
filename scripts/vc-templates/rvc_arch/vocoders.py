# SPDX-License-Identifier: AGPL-3.0-or-later
# Architecture ported from IAHispano/Applio (MIT, see applio_arch/LICENSE):
#   rvc/lib/algorithm/generators/refinegan.py and hifigan_mrf.py at commit
#   c7665ac9a305b3683570ed914f1577d53d4b75c4 (vendored unchanged in ../applio_arch for the
#   maintainer cross-check `parity_test.py --upstream-applio`).
"""Sigil ports of Applio's RefineGAN and MRF HiFi-GAN generators (the vocoders of Applio-trained RVC
voices that mainline RVC and w-okada cannot run).

The module trees and parameter names are Applio's, so an Applio checkpoint's ``dec.*`` tensors load
into them unchanged (weight_norm as ``weight_g`` / ``weight_v``; the importer also accepts Applio's
newer ``parametrizations.weight.original0/1`` keys). Each class has a ``reference`` forward: the full,
non-streaming decode with Applio's arithmetic, its random draws replaced by explicit inputs so the
split graphs (``split.py``) can be checked against it:

- ``src_noise``: the source's unit Gaussian noise, one value per output sample and harmonic
  (Applio: ``torch.randn_like(sine_waves)``).
- ``phase_ini``: the harmonics' initial phases in cycles (Applio: ``torch.rand``; the fundamental's is
  always 0). Sigil fixes them per template (``MRF_PHASE_INI``) so the excitation is deterministic.
- ``adain``: RefineGAN's per-layer noise (``AdaIN``: ``x + randn_like(x) * weight``), a callable
  ``(stage, layer, x) -> unit normals shaped like x``. Sigil draws it from a fixed pool by absolute
  frame (``adain_noise_from_pool``), so overlapping ticks see identical noise.

Everything else (sine source, voicing, resampling, upsampling, residual blocks) follows Applio line by
line; ``parity_test.py --upstream-applio`` compares these references with Applio's own
``Synthesizer.infer`` on the same weights and noise.
"""

import math

import numpy as np
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import weight_norm

SINE_AMP = 0.1
NOISE_STD = 0.003
REFINEGAN_SLOPE = 0.2
MRF_SLOPE = 0.1
MRF_HARMONICS = 8  # Applio's Synthesizer builds the MRF generator with harmonic_num=8 (9 sines)
# RefineGAN's anti-aliasing resampler (torchaudio sinc_interp_kaiser), exactly Applio's settings.
RESAMPLE = {
    "lowpass_filter_width": 64,
    "rolloff": 0.9475937167399596,
    "resampling_method": "sinc_interp_kaiser",
    "beta": 14.769656459379492,
}
# Fixed initial phases (cycles) of the MRF harmonics 2..9 (Applio draws them with torch.rand per
# call; any values are as valid, fixed ones make the excitation reproducible). The fundamental's is
# 0, as in Applio. Written to layout.json (streaming.excitation.phase_ini).
MRF_PHASE_INI = tuple(
    [0.0] + [float(x) for x in np.random.default_rng(20260929).random(MRF_HARMONICS).round(6)]
)


def _pad(kernel_size, dilation=1):
    return (kernel_size * dilation - dilation) // 2


# ---------------------------------------------------------------------------------------------
# Excitation (Applio SineGenerator, per sample)
# ---------------------------------------------------------------------------------------------


def sine_source(f0_samples, sample_rate, dim, phase_ini, noise):
    """Applio ``SineGenerator`` (refinegan.py / hifigan_mrf.py) with explicit randomness.

    f0_samples [B, S] Hz at the output rate, phase_ini [dim] cycles, noise [B, S, dim] unit normals
    -> [B, S, dim] (sines on voiced samples plus noise). Computed in the dtype of ``f0_samples``
    (fp32 like Applio, or float64 for the parity reference: Applio's per-sample fp32 cumulative sum
    drifts by ~1e-5 cycles over a few seconds, which the gate should not blame on the graphs).
    """
    dt = f0_samples.dtype if f0_samples.is_floating_point() else torch.float32
    f0 = f0_samples.to(dt).unsqueeze(-1)  # [B, S, 1]
    f0_buf = f0 * torch.arange(1, dim + 1, dtype=dt, device=f0.device)  # harmonic k+1
    rad = (f0_buf / sample_rate) % 1
    rad = torch.cat([rad[:, :1, :] + phase_ini.reshape(1, 1, -1).to(dt), rad[:, 1:, :]], dim=1)
    tmp = torch.cumsum(rad, 1) % 1
    wrapped = (tmp[:, 1:, :] - tmp[:, :-1, :]) < 0
    shift = torch.cat([torch.zeros_like(rad[:, :1, :]), wrapped.to(dt) * -1.0], dim=1)
    sines = torch.sin(torch.cumsum(rad + shift, dim=1) * 2 * np.pi) * SINE_AMP
    uv = (f0 > 0).to(dt)
    noise_amp = uv * NOISE_STD + (1 - uv) * SINE_AMP / 3
    return sines * uv + noise_amp * noise.to(dt)


# ---------------------------------------------------------------------------------------------
# RefineGAN
# ---------------------------------------------------------------------------------------------


class AdaIN(nn.Module):
    """Applio ``AdaIN``: ``leaky_relu(x + N(0, 1) * weight[c])`` (the noise comes from outside)."""

    def __init__(self, channels):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels) * 1e-4)


class RefineResBlock(nn.Module):
    """Applio refinegan ``ResBlock`` (weight-normed dilated convs, no mask)."""

    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=d, padding=_pad(kernel_size, d))) for d in dilation]
        )
        self.convs2 = nn.ModuleList(
            [weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, dilation=1, padding=_pad(kernel_size))) for _ in dilation]
        )


class ParallelResBlock(nn.Module):
    """Applio ``ParallelResBlock``: input conv, then three (AdaIN, ResBlock, AdaIN) branches averaged."""

    def __init__(self, in_channels, out_channels, kernel_sizes=(3, 7, 11), dilation=(1, 3, 5)):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels, out_channels, 7, 1, padding=3)
        self.blocks = nn.ModuleList(
            [nn.Sequential(AdaIN(out_channels), RefineResBlock(out_channels, k, dilation), AdaIN(out_channels)) for k in kernel_sizes]
        )


def refine_resblock(block, x, slope=REFINEGAN_SLOPE):
    for c1, c2 in zip(block.convs1, block.convs2):
        xt = F.leaky_relu(x, slope)
        xt = c1(xt)
        xt = F.leaky_relu(xt, slope)
        xt = c2(xt)
        x = xt + x
    return x


def parallel_resblock(prb, x, noise, slope=REFINEGAN_SLOPE):
    """``noise(layer, y)`` -> unit normals shaped like ``y`` for AdaIN layer 0..5 (2 per branch)."""
    x = prb.input_conv(x)
    outs = []
    for j, (ada1, res, ada2) in enumerate(prb.blocks):
        y = F.leaky_relu(x + noise(2 * j, x) * ada1.weight.to(x.dtype)[None, :, None], slope)
        y = refine_resblock(res, y, slope)
        y = F.leaky_relu(y + noise(2 * j + 1, y) * ada2.weight.to(y.dtype)[None, :, None], slope)
        outs.append(y)
    return torch.stack(outs, dim=0).mean(dim=0)


class RefineSource(nn.Module):
    """Applio refinegan ``SineGenerator`` parameters: ``merge`` = Linear(1, 1, bias=False) + Tanh."""

    def __init__(self):
        super().__init__()
        self.merge = nn.Sequential(nn.Linear(1, 1, bias=False), nn.Tanh())


class RefineGANDecoder(nn.Module):
    """Applio ``RefineGANGenerator`` (as Applio's Synthesizer builds it: start_channels 16,
    upsample_initial_channel 512, gin 256, num_mels = inter_channels)."""

    def __init__(self, sample_rate, upsample_rates, num_mels, gin_channels=256, upsample_initial_channel=512, start_channels=16):
        super().__init__()
        self.sample_rate = int(sample_rate)
        self.upsample_rates = [int(u) for u in upsample_rates]
        self.upp = int(np.prod(self.upsample_rates))
        self.m_source = RefineSource()
        self.pre_conv = weight_norm(nn.Conv1d(1, 16, 7, 1, padding=3))
        channels, size = start_channels, self.upp
        self.df0 = []
        self.downsample_blocks = nn.ModuleList()
        for i in range(len(self.upsample_rates)):
            new_size = int(size / self.upsample_rates[-i - 1])
            self.df0.append((size, new_size))
            size = new_size
            self.downsample_blocks.append(weight_norm(nn.Conv1d(channels, channels * 2, 7, 1, padding=3)))
            channels *= 2
        channels = upsample_initial_channel
        self.mel_conv = weight_norm(nn.Conv1d(num_mels, channels // 2, 7, 1, padding=3))
        if gin_channels:
            self.cond = nn.Conv1d(256, channels // 2, 1)
        self.upsample_conv_blocks = nn.ModuleList()
        self.stage_channels = []
        for _ in self.upsample_rates:
            new_channels = channels // 2
            self.upsample_conv_blocks.append(ParallelResBlock(channels + channels // 4, new_channels))
            self.stage_channels.append(new_channels)
            channels = new_channels
        self.conv_post = weight_norm(nn.Conv1d(channels, 1, 7, 1, padding=3, bias=False))

    def stage_rates(self):
        """Samples per 10 ms frame after each upsampling stage."""
        out, r = [], 1
        for u in self.upsample_rates:
            r *= u
            out.append(r)
        return out

    def reference(self, mel, f0, g, src_noise, adain):
        """Full decode, Applio arithmetic: mel [1, C, T], f0 [1, T] Hz, g [1, 256, 1],
        src_noise [1, T*upp] unit normals, adain(stage, layer, x) -> unit normals like x."""
        t = mel.shape[-1]
        dt = torch.float64 if mel.dtype == torch.float64 else torch.float32
        f0s = F.interpolate(f0.to(dt).unsqueeze(1), size=t * self.upp, mode="linear")  # [1, 1, S]
        sine = sine_source(f0s[:, 0, :], self.sample_rate, 1, torch.zeros(1, device=f0.device), src_noise.reshape(1, -1, 1))
        har = torch.tanh(F.linear(sine, self.m_source.merge[0].weight.to(dt))).transpose(1, 2).to(mel.dtype)
        x = self.pre_conv(har)
        downs = []
        for block, (old, new) in zip(self.downsample_blocks, self.df0):
            x = F.leaky_relu(x, REFINEGAN_SLOPE)
            downs.append(x)
            x = torchaudio.functional.resample(x.contiguous(), orig_freq=int(t * old), new_freq=int(t * new), **RESAMPLE)
            x = block(x)
        m = self.mel_conv(mel)
        if g is not None:
            m = m + self.cond(g)
        x = torch.cat([m, x], dim=1)
        for i, (rate, res, down) in enumerate(zip(self.upsample_rates, self.upsample_conv_blocks, reversed(downs))):
            x = F.leaky_relu(x, REFINEGAN_SLOPE)
            x = F.interpolate(x, scale_factor=rate, mode="linear")
            x = torch.cat([x, down], dim=1)
            x = parallel_resblock(res, x, lambda layer, y, i=i: adain(i, layer, y))
        x = F.leaky_relu(x, REFINEGAN_SLOPE)
        x = self.conv_post(x)
        return torch.tanh(x)


def sinc_resample_kernel(orig, dtype=torch.float32):
    """torchaudio's kernel for an ``orig`` -> 1 reduction with Applio's settings: (kernel [1, 1, K], width)."""
    from torchaudio.functional.functional import _get_sinc_resample_kernel

    kernel, width = _get_sinc_resample_kernel(
        orig, 1, 1, RESAMPLE["lowpass_filter_width"], RESAMPLE["rolloff"], RESAMPLE["resampling_method"],
        RESAMPLE["beta"], torch.device("cpu"), dtype,
    )
    return kernel, int(width)


def sinc_resample(x, kernel, width, orig):
    """``torchaudio.functional.resample(x, orig * n, n)`` for x [1, C, T] with T a multiple of ``orig``,
    written with data-driven shapes (exportable with a dynamic T): [1, C, T / orig]."""
    y = x.reshape(-1, 1, x.shape[-1])
    y = F.pad(y, (width, width + orig))
    y = F.conv1d(y, kernel.to(y.dtype), stride=orig)  # T / orig + 1 outputs
    y = y[:, :, :-1]
    return y.reshape(1, -1, y.shape[-1])


def adain_layers(decoder):
    """AdaIN noise layout per 10 ms frame: [(stage, layer, channels, rate, base)], per_frame total.
    Stage by stage, branch by branch (pre, post), channel-major within a layer."""
    out, base = [], 0
    for i, (c, r) in enumerate(zip(decoder.stage_channels, decoder.stage_rates())):
        for layer in range(6):
            out.append((i, layer, c, r, base))
            base += c * r
    return out, base


def adain_noise_from_pool(pool, offsets, base, like):
    """Unit normals for one AdaIN layer over n frames, shaped like ``like`` [1, C, n * rate]:
    pool [P] f32, offsets [n] int64 (one per absolute frame); element (c, frame j, sample s) is
    pool[offsets[j] + base + c * rate + s]. The index grid is built from the tensors' shapes (Range in
    ONNX), so no per-layer index constant lands in the graph."""
    n = offsets.shape[0]
    channels = like.shape[1]
    rate = like.shape[2] // n
    c_idx = torch.arange(channels, dtype=torch.int64, device=offsets.device).reshape(1, -1, 1) * rate
    s_idx = torch.arange(rate, dtype=torch.int64, device=offsets.device).reshape(1, 1, -1)
    idx = offsets.reshape(-1, 1, 1) + (c_idx + s_idx + base)  # [n, C, r]
    z = pool[idx]
    return z.permute(1, 0, 2).reshape(1, channels, -1)


# ---------------------------------------------------------------------------------------------
# MRF HiFi-GAN
# ---------------------------------------------------------------------------------------------


class MRFLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=_pad(kernel_size, dilation), dilation=dilation))
        self.conv2 = weight_norm(nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, dilation=1))


class MRFBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList([MRFLayer(channels, kernel_size, d) for d in dilations])


def mrf_block(block, x):
    for layer in block.layers:
        y = F.leaky_relu(x, MRF_SLOPE)
        y = layer.conv1(y)
        y = F.leaky_relu(y, MRF_SLOPE)
        y = layer.conv2(y)
        x = x + y
    return x


class MRFSource(nn.Module):
    """Applio hifigan_mrf ``SourceModuleHnNSF`` parameters: ``l_linear`` = Linear(harmonics + 1, 1)."""

    def __init__(self, harmonic_num):
        super().__init__()
        self.l_linear = nn.Linear(harmonic_num + 1, 1)


class MRFDecoder(nn.Module):
    """Applio ``HiFiGANMRFGenerator`` (harmonic_num 8, as Applio's Synthesizer builds it)."""

    def __init__(
        self, in_channel, upsample_initial_channel, upsample_rates, upsample_kernel_sizes, resblock_kernel_sizes,
        resblock_dilations, gin_channels, sample_rate, harmonic_num=MRF_HARMONICS,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.sample_rate = int(sample_rate)
        self.upp = int(math.prod(upsample_rates))
        self.dim = harmonic_num + 1
        self.m_source = MRFSource(harmonic_num)
        self.conv_pre = weight_norm(nn.Conv1d(in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3))
        self.upsamples = nn.ModuleList()
        self.noise_convs = nn.ModuleList()
        stride_f0s = [math.prod(upsample_rates[i + 1 :]) if i + 1 < len(upsample_rates) else 1 for i in range(len(upsample_rates))]
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            padding = (k - u) // 2 if u % 2 == 0 else u // 2 + u % 2
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2**i), upsample_initial_channel // (2 ** (i + 1)), kernel_size=k,
                        stride=u, padding=padding, output_padding=u % 2,
                    )
                )
            )
            stride = stride_f0s[i]
            kernel = 1 if stride == 1 else stride * 2 - stride % 2
            pad = 0 if stride == 1 else (kernel - stride) // 2
            self.noise_convs.append(nn.Conv1d(1, upsample_initial_channel // (2 ** (i + 1)), kernel_size=kernel, stride=stride, padding=pad))
        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(nn.ModuleList([MRFBlock(channel, k, d) for k, d in zip(resblock_kernel_sizes, resblock_dilations)]))
        self.conv_post = weight_norm(nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3))
        if gin_channels:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def source(self, f0_samples, src_noise, phase_ini):
        sine = sine_source(f0_samples, self.sample_rate, self.dim, phase_ini, src_noise)
        lin = self.m_source.l_linear
        return torch.tanh(F.linear(sine, lin.weight.to(sine.dtype), lin.bias.to(sine.dtype))).transpose(1, 2)  # [1, 1, S]

    def reference(self, x, f0, g, src_noise, phase_ini):
        """Full decode, Applio arithmetic: x [1, C, T], f0 [1, T] Hz, g [1, gin, 1],
        src_noise [1, T*upp, 9] unit normals, phase_ini [9] cycles."""
        dt = torch.float64 if x.dtype == torch.float64 else torch.float32
        f0s = f0.to(dt).repeat_interleave(self.upp, dim=1)  # nn.Upsample(nearest)
        har = self.source(f0s, src_noise, phase_ini).to(x.dtype)
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)
        for ups, mrf, noise_conv in zip(self.upsamples, self.mrfs, self.noise_convs):
            x = F.leaky_relu(x, MRF_SLOPE)
            x = ups(x)
            x = x + noise_conv(har)
            x = sum(mrf_block(b, x) for b in mrf) / self.num_kernels
        x = F.leaky_relu(x)
        return torch.tanh(self.conv_post(x))
