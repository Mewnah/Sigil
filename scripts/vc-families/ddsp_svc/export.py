#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""DDSP-SVC 6.3 (DDSP + rectified flow, 44.1 kHz singing) -> ONNX graphs (P3-H, ADR 0006).

Upstream code is vendored unchanged under ``upstream/`` (yxlllc/DDSP-SVC branch 6.3 @d6dd52fc,
MIT), including upstream's own exporter ``export_onnx.py``. DDSP-SVC voices are trained per user
and no public 6.3 checkpoint exists (the 5.0 release's ``model_0.pt`` is incompatible), so the
DDSP and velocity graphs are exported from the 6.3 architecture (``configs/reflow.yaml``) with
seeded random weights; the parity test checks the export against upstream PyTorch on those
weights. The vocoder is the real one: openvpi pc_nsf_hifigan_44.1k_hop512_128bin_2025.02 (its
``model.ckpt`` + ``config.json``, fetched once into ``sigil-dev-cache/ddsp-svc/openvpi``).

Graphs (``sigil-dev-cache/vc-families/ddsp_svc``), every length dynamic:

upstream (``export_onnx.py``, reused; its single-speaker call references an undefined ``test_sid``,
so the exporter's own classes are called with ``g=None``):
  encoder     hubert f32 [1,U,768], mel2ph i64 [1,T], f0 f32 [1,T] Hz, volume f32 [1,T], randn f32 [1,512T]
              -> x f32 [1,1,128,T] (normalised DDSP mel), cond f32 [1,128,T] (DDSP log-mel)   opset 18
  velocity    x f32 [1,1,128,T], t i64 [1] (= 1000 t), cond f32 [1,128,T] -> o f32 [1,1,128,T]  opset 18

Sigil:
  ddsp        the ``encoder`` inputs plus start_frame i64 [1] and formant_shift f32 [1,1] (semitones;
              upstream's aug_shift, which the exporter drops) -> x f32 [1,1,128,T-start], cond f32 [1,128,T-start]:
              upstream's inference path, the mel of the DDSP audio after ``silence_front``; the STFT as a
              strided convolution (no ONNX ``STFT`` op),
              a NaN-free sinc for the comb excitation (upstream's export divides 0 by 0 wherever
              f0 is 0), and the mel exactly as upstream PyTorch computes it: ``vocoder.extract``, i.e.
              the vocoder's own STFT settings (n_fft, win_size, hop_size, num_mels, fmin, fmax from its
              config.json; ``VocoderMel``), not the exporter's hard-coded ones
  velocity_f  x f32 [1,1,128,T], t f32 [1] (0..1; the graph scales by 1000 as upstream PyTorch does),
              cond f32 [1,128,T] -> v f32 [1,1,128,T]
  reflow_step x, t, dt f32 [1], cond -> x + dt * v(x, 1000 t, cond)                (one Euler step)
  vocoder     mel f32 [1,128,T] (natural-log mel), f0 f32 [1,T] Hz, phase0 f32 [1] (cycles added to the
              sine source; 0 = upstream) -> audio f32 [1,512T] at 44.1 kHz
              (PC-NSF-HiFiGAN, mini-NSF source; weights folded; CC BY-NC-SA 4.0 weights)

Usage (repo root): ``python scripts/vc-families/ddsp_svc/export.py [--only ddsp,vocoder]``
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

# Only upstream's F0_Extractor / HubertSoft use these; the graphs take f0 and units as inputs.
cm.install_stubs({"pyworld": {}, "parselmouth": {}, "torchcrepe": {}, "resampy": {},
                  "encoder.hubert.model": {"HubertSoft": None}})

VOC_DIR = os.path.join(cm.DEV_CACHE, "ddsp-svc", "openvpi", "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02")
VOC_CKPT = os.path.join(VOC_DIR, "model.ckpt")
CONFIG = os.path.join(UPSTREAM, "configs", "reflow.yaml")
OUT = cm.family_out("ddsp_svc")
SEED = 1234


def load_config():
    import yaml
    from reflow.vocoder import DotDict
    return DotDict(yaml.safe_load(open(CONFIG, "r", encoding="utf-8")))


def _randomise(model, seed=SEED):
    """Seeded weights for the architecture (no public 6.3 checkpoint): PyTorch's default init at
    construction (seeded by the caller), except LYNXNet2's output projection, which upstream
    zero-initialises (the velocity would be a constant); it gets N(0, 0.02)."""
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        op = model.reflow_model.velocity_fn.output_projection
        op.weight.normal_(0.0, 0.02)
        op.bias.normal_(0.0, 0.02)
    return model


def check_vocoder_config(args, h):
    """``Unit2Wav.forward`` hands the DDSP audio straight to ``vocoder.extract`` (no resampling) and
    pairs the vocoder's mel frames with the DDSP frames and the F0, so the vocoder must run at the
    model's sampling rate and hop. Everything else about the mel (n_fft, window, bins, band) comes
    from the vocoder's config (``VocoderMel``)."""
    bad = []
    if int(h.sampling_rate) != int(args.data.sampling_rate):
        bad.append(f"vocoder sampling_rate {h.sampling_rate} != model data.sampling_rate {args.data.sampling_rate}")
    if int(h.hop_size) != int(args.data.block_size):
        bad.append(f"vocoder hop_size {h.hop_size} != model data.block_size {args.data.block_size}")
    if int(h.win_size) > int(h.n_fft):
        bad.append(f"vocoder win_size {h.win_size} > n_fft {h.n_fft}")
    if bad:
        raise SystemExit("DDSP-SVC model and vocoder do not fit together: " + "; ".join(bad))


def load_upstream(device="cpu"):
    """Upstream 6.3 PyTorch (reflow/vocoder.py Unit2Wav + Vocoder), as load_model_vocoder builds it."""
    from reflow.vocoder import Unit2Wav, Vocoder
    cm.need(VOC_CKPT, "openvpi pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.zip (GitHub release), unzipped")
    args = load_config()
    vocoder = Vocoder(args.vocoder.type, VOC_CKPT, device=device)
    check_vocoder_config(args, vocoder.vocoder.h)
    torch.manual_seed(SEED)
    model = Unit2Wav(args.data.sampling_rate, args.data.block_size, args.model.win_length, args.data.encoder_out_channels,
                     args.model.n_spk, args.model.use_norm, args.model.use_attention, args.model.use_pitch_aug,
                     vocoder.dimension, args.model.n_aux_layers, args.model.n_aux_chans, args.model.n_layers, args.model.n_chans)
    _randomise(model)
    model.eval().to(device)
    cm.freeze(model)
    vocoder.vocoder.forward(torch.zeros(1, 4, vocoder.dimension, device=device), torch.full((1, 4), 200.0, device=device))  # lazy load
    for p in vocoder.vocoder.model.parameters():
        p.requires_grad_(False)
    return {"model": model, "vocoder": vocoder, "args": args}


def export_version(up):
    """Upstream's exporter model (export_onnx.Unit2Wav) carrying the same weights."""
    import export_onnx as upx
    a = up["args"]
    m = upx.Unit2Wav(a.data.sampling_rate, a.data.block_size, a.model.win_length, a.data.encoder_out_channels, a.model.n_spk,
                     a.model.use_norm, a.model.use_attention, a.model.use_pitch_aug, up["vocoder"].dimension, a.model.n_aux_layers,
                     a.model.n_aux_chans, a.model.n_layers, a.model.n_chans, a.data.f0_min)
    m.load_state_dict(up["model"].state_dict(), strict=True)
    return cm.freeze(m.eval())


# ---------------------------------------------------------------------------------------------
# Wrappers
# ---------------------------------------------------------------------------------------------

class UpstreamEncoder(nn.Module):
    """export_onnx.CombSubSuperFast.forward, single speaker (``g=None``)."""

    def __init__(self, ddsp):
        super().__init__()
        self.m = ddsp

    def forward(self, hubert, mel2ph, f0, volume, randn):
        return self.m(hubert, mel2ph, f0, volume, None, randn)


def _stft_basis(n_fft, window):
    win = window.double()
    k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]
    t = torch.arange(n_fft, dtype=torch.float64)[None, :]
    ang = 2 * math.pi * k * t / n_fft
    return torch.cat([torch.cos(ang) * win, -torch.sin(ang) * win], 0)[:, None, :].float()


class VocoderMel(nn.Module):
    """``nsf_hifigan.nvSTFT.STFT.get_mel`` (keyshift 0, center False) with the vocoder's own config
    ``h`` (what upstream's ``vocoder.extract`` computes): reflect pad (win - hop) // 2 left and
    (win - hop + 1) // 2 right, |STFT| (n_fft, hop, periodic hann(win) centred in n_fft) as a strided
    convolution, the librosa mel basis (num_mels, fmin, fmax), log of the clamp at 1e-5.
    ``share`` = (n_fft, window) of a DFT basis the caller already holds: when the vocoder's STFT is
    that one, ``forward`` takes the caller's basis instead of storing a second copy."""

    def __init__(self, h, share=None):
        super().__init__()
        from librosa.filters import mel as librosa_mel_fn
        n_fft, win, hop = int(h.n_fft), int(h.win_size), int(h.hop_size)
        window = torch.zeros(n_fft)
        left = (n_fft - win) // 2
        window[left: left + win] = torch.hann_window(win)                # torch.stft's centred window
        self.shared = share is not None and share[0] == n_fft and torch.equal(share[1].float(), window)
        if not self.shared:
            self.register_buffer("basis", _stft_basis(n_fft, window))
        mel = librosa_mel_fn(sr=int(h.sampling_rate), n_fft=n_fft, n_mels=int(h.num_mels), fmin=h.fmin, fmax=h.fmax)
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.pad = ((win - hop) // 2, (win - hop + 1) // 2)
        self.hop, self.nb = hop, n_fft // 2 + 1

    def forward(self, signal, basis=None):
        y = F.pad(signal[:, None, :], self.pad, mode="reflect")
        spec = F.conv1d(y, basis if self.shared else self.basis, stride=self.hop)
        mag = torch.sqrt(spec[:, : self.nb] ** 2 + spec[:, self.nb:] ** 2)
        return torch.log(torch.clamp(torch.matmul(self.mel_basis, mag), min=1e-5))


def _safe_sinc(x):
    """torch.sinc (1 at 0), which upstream PyTorch uses; the exporter's sin(pi x)/(pi x) is NaN at 0."""
    px = math.pi * x
    tiny = px.abs() < 1e-20
    return torch.where(tiny, torch.ones_like(px), torch.sin(px) / torch.where(tiny, torch.ones_like(px), px))


class SigilDdsp(nn.Module):
    """The exporter's CombSubSuperFast.forward with ONNX-portable parts: both STFTs (center, reflect
    pad by n_fft/2, hann) as strided convolutions, the NaN-free sinc, and the mel of the DDSP audio
    exactly as upstream PyTorch computes it: ``vocoder.extract`` with the vocoder's own STFT config
    ``h`` (``VocoderMel``; the exporter instead hard-codes n_fft = win = DDSP win_length, 128 bins,
    40-16000 Hz, and adds 1e-9 under the magnitude's square root)."""

    def __init__(self, ddsp, h):
        super().__init__()
        self.m = ddsp
        self.bs = int(ddsp.bs)
        self.wl = int(ddsp.wl)
        self.register_buffer("basis", _stft_basis(self.wl, ddsp.window))
        self.nb = self.wl // 2 + 1
        self.mel = VocoderMel(h, share=(self.wl, ddsp.window))

    def _stft(self, x):
        pad = self.wl // 2
        mode = "reflect" if x.shape[-1] > pad else "constant"
        y = F.conv1d(F.pad(x[:, None, :], (pad, pad), mode=mode), self.basis, stride=self.bs)   # [B, 2nb, F]
        return torch.stack([y[:, : self.nb], y[:, self.nb:]], -1)                                  # [B, nb, F, 2]

    def forward(self, hubert, mel2ph, f0, volume, randn, start_frame, formant_shift):
        m = self.m
        units = torch.gather(hubert, 1, mel2ph.unsqueeze(2).repeat([1, 1, hubert.shape[-1]]))
        volume = volume.unsqueeze(-1)
        f0_frames = f0.unsqueeze(-1)
        # fast_source_gen with torch.sinc semantics
        n = torch.arange(self.bs, device=f0.device)
        s0 = f0_frames / m.sampling_rate
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * (n + 1) + 0.5 * ds0 * n * (n + 1) / m.block_size
        s0 = s0 + ds0 * n / m.block_size
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0_frames)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad - torch.round(rad)
        combtooth = _safe_sinc(rad / (s0 + 1e-5)).reshape(f0.shape[0], -1)
        combtooth_frames = combtooth.view(1, -1, self.bs)
        noise_frames = randn.view(1, -1, self.bs)
        # Unit2Control.forward (ddsp/unit2control.py, single speaker) with upstream's formant shift
        # (``aug_shift``, the GUI's "formant_shift_key" in semitones), which the exporter drops
        u = m.unit2ctrl
        exciter = torch.cat((combtooth_frames, noise_frames), dim=-1).transpose(1, 2)
        h = u.stack(units.transpose(1, 2)) + u.stack2(exciter)
        h = h.transpose(1, 2) + u.volume_embed(volume)
        if u.aug_shift_embed is not None:
            h = h + u.aug_shift_embed(formant_shift / 5)
        e = u.dense_out(u.norm(u.decoder(h)))
        hm, hp, nm, npase = torch.split(e, u.o_sp, dim=-1)
        src = m.complex_exp(hm.unsqueeze(-1), torch.ones_like(hm).unsqueeze(-1) * np.pi * hp.unsqueeze(-1))
        src = torch.cat((src, src[:, -1:, :]), 1)
        nf = m.complex_exp(nm.unsqueeze(-1), torch.ones_like(nm).unsqueeze(-1) * np.pi * npase.unsqueeze(-1)) / 128
        nf = torch.cat((nf, nf[:, -1:, :]), 1)
        sig = m.complex_mul(self._stft(combtooth), src.permute(0, 2, 1, 3)) + m.complex_mul(self._stft(randn), nf.permute(0, 2, 1, 3))
        signal = m.istft_method(sig)
        # upstream (reflow/vocoder.py Unit2Wav.forward) takes the mel of the DDSP audio after
        # ``silence_front``: ddsp_wav[:, start_frame * block_size:]
        signal = signal[:, start_frame[0] * self.bs:]
        # mel: vocoder.extract = nvSTFT.get_mel with the vocoder's config (VocoderMel)
        mel = self.mel(signal, self.basis)
        return m.norm_spec(mel).unsqueeze(1), mel


class VelocityF(nn.Module):
    """LYNXNet2 as upstream RectifiedFlow calls it: ``velocity_fn(x, 1000 * t, cond)`` with float t."""

    def __init__(self, vel):
        super().__init__()
        self.v = vel

    def forward(self, x, t, cond):
        return self.v(x, 1000.0 * t, cond)


class ReflowStep(nn.Module):
    """RectifiedFlow.sample_euler: ``x + dt * velocity_fn(x, 1000 t, cond)``."""

    def __init__(self, vel):
        super().__init__()
        self.v = vel

    def forward(self, x, t, dt, cond):
        return x + self.v(x, 1000.0 * t, cond) * dt


class Vocoder(nn.Module):
    """openvpi PC-NSF-HiFiGAN ``Generator.forward`` (mini-NSF, weight norm removed by upstream's
    load_model) with one Sigil input: ``phase0`` (cycles) is added to the sine source's phase, so
    the engine can continue the excitation across overlapping windows (time-anchored, the
    phase-continuous NSF lever). ``phase0 = 0`` is upstream exactly."""

    def __init__(self, gen):
        super().__init__()
        self.g = gen

    def forward(self, mel, f0, phase0):
        g = self.g
        n = torch.arange(1, g.upp + 1, device=f0.device)
        s0 = f0.unsqueeze(-1) / g.source_sr
        ds0 = F.pad(s0[:, 1:, :] - s0[:, :-1, :], (0, 0, 0, 1))
        rad = s0 * n + 0.5 * ds0 * n * (n - 1) / g.upp
        rad2 = torch.fmod(rad[..., -1:].float() + 0.5, 1.0) - 0.5
        rad_acc = rad2.cumsum(dim=1).fmod(1.0).to(f0)
        rad = rad + F.pad(rad_acc[:, :-1, :], (0, 0, 1, 0))
        rad = rad.reshape(f0.shape[0], 1, -1) + phase0
        har_source = torch.sin(2 * np.pi * rad)
        x = g.conv_pre(mel)
        for i in range(g.num_upsamples):
            x = F.leaky_relu(x, 0.1)
            x = g.ups[i](x)
            if i == 1:
                x = x + g.source_conv(har_source)
            xs = None
            for j in range(g.num_kernels):
                r = g.resblocks[i * g.num_kernels + j](x)
                xs = r if xs is None else xs + r
            x = xs / g.num_kernels
        x = F.leaky_relu(x)
        return torch.tanh(g.conv_post(x))[:, 0, :]


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------

def graph_specs(up):
    torch.manual_seed(0)
    T, U = 40, 70
    nm = up["vocoder"].dimension
    h = up["vocoder"].vocoder.h
    ev = export_version(up)
    enc_args = (torch.randn(1, U, 768), torch.clamp(torch.round(torch.arange(T) * 1.1611), max=U - 1).long()[None],
                torch.full((1, T), 220.0), torch.rand(1, T) * 0.1, torch.randn(1, T * 512))
    enc_dyn = {"hubert": {1: "U"}, "mel2ph": {1: "T"}, "f0": {1: "T"}, "volume": {1: "T"}, "randn": {1: "S"}}
    vel = copy.deepcopy(up["model"].reflow_model.velocity_fn)
    x, cond = torch.randn(1, 1, nm, T), torch.randn(1, nm, T) - 5
    gen = up["vocoder"].vocoder.model  # not copied: its AttrDict config does not deep-copy; export never mutates it
    return {
        # weight_norm folded: torch 2.3 exports it at opset 18 as a ReduceL2 with the removed ``axes``
        # attribute (upstream exports with torch 2.9); folding computes the same weights once.
        "encoder": (UpstreamEncoder(cm.strip_weight_norm(copy.deepcopy(ev.ddsp_model))), enc_args, ["hubert", "mel2ph", "f0", "volume", "randn"], ["x", "cond"],
                    enc_dyn, 18, False),
        "velocity": (copy.deepcopy(ev.reflow_model.velocity_fn), (x, torch.tensor([0], dtype=torch.int64), cond), ["x", "t", "cond"], ["o"],
                     {"x": {3: "frame"}, "cond": {2: "frame"}}, 18, True),
        "ddsp": (SigilDdsp(cm.strip_weight_norm(copy.deepcopy(ev.ddsp_model)), h), enc_args + (torch.tensor([10]), torch.tensor([[0.5]])),
                 ["hubert", "mel2ph", "f0", "volume", "randn", "start_frame", "formant_shift"], ["x", "cond"],
                 {**enc_dyn, "x": {3: "R"}, "cond": {2: "R"}}, 17, True),
        "velocity_f": (VelocityF(copy.deepcopy(vel)), (x, torch.tensor([0.7]), cond), ["x", "t", "cond"], ["v"],
                       {"x": {3: "T"}, "cond": {2: "T"}, "v": {3: "T"}}, 17, True),
        "reflow_step": (ReflowStep(copy.deepcopy(vel)), (x, torch.tensor([0.7]), torch.tensor([0.015]), cond), ["x", "t", "dt", "cond"], ["x_next"],
                        {"x": {3: "T"}, "cond": {2: "T"}, "x_next": {3: "T"}}, 17, True),
        "vocoder": (Vocoder(gen), (torch.randn(1, nm, T) - 5, torch.full((1, T), 220.0), torch.zeros(1)), ["mel", "f0", "phase0"], ["audio"],
                    {"mel": {2: "T"}, "f0": {1: "T"}, "audio": {1: "S"}}, 17, True),
    }


def export(names):
    import onnx
    up = load_upstream("cpu")
    specs = graph_specs(up)
    written = []
    for name in names:
        mod, args, ins, outs, dyn, opset, fold = specs[name]
        path = os.path.join(OUT, f"{name}.onnx")
        with torch.no_grad():
            torch.onnx.export(mod.eval(), args, path, input_names=ins, output_names=outs, dynamic_axes=dyn,
                              opset_version=opset, do_constant_folding=fold)
        m = onnx.load(path)
        onnx.checker.check_model(m)
        del m.metadata_props[:]
        src = "openvpi pc_nsf_hifigan_44.1k_hop512_128bin_2025.02" if name == "vocoder" else \
            f"yxlllc/DDSP-SVC@d6dd52fc (6.3) architecture, seeded random weights ({SEED})"
        m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.source", value=src))
        m.metadata_props.append(onnx.StringStringEntryProto(key="sigil.graph", value=f"ddsp_svc/{name}"))
        onnx.save(m, path)
        written.append(path)
        print(f"wrote {path} ({os.path.getsize(path) / 1e6:.1f} MB)")
    return written


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="encoder,velocity,ddsp,velocity_f,reflow_step,vocoder")
    args = ap.parse_args()
    export([n.strip() for n in args.only.split(",") if n.strip()])


if __name__ == "__main__":
    main()
