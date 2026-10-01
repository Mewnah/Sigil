# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference sampler for Sigil-ZS (Seed-VC realtime on the exported graphs): the numerical
reference a Rust port of the engine loop is checked against (ADR 0006).

It reproduces upstream ``real-time-gui.py`` ``custom_infer`` (Plachtaa/seed-vc@51383efd) on the
ONNX graphs of ``export.py``, with numpy only between graph calls:

reference (once per voice, ``prepare_reference``)
  ref22k = reference audio at 22.05 kHz, cut to ``max_prompt`` seconds
  ref16k = resample(ref22k, 22050 -> 16000)            (upstream: torchaudio sinc resample)
  S_ori  = xlsr(ref16k)                                 [1,T,1024]
  style  = campplus(fbank(ref16k) - mean over frames)   [1,192]  (Kaldi fbank, 80 bins, dither 0)
  prompt_mel = mel(ref22k)                              [1,80,P]
  prompt_cond = lr(S_ori, P)                            [1,P,384]

tick (every block, ``tick``); the window is ``extra + crossfade + sola_search + block + right``
units of 20 ms at 16 kHz (see ``Preset``)
  S_alt = xlsr(window16k)[:, ce:]                       ce = (extra_time_ce - extra_time) * 50 frames
  cond  = lr(S_alt, n_mel)                              n_mel = int((skip_head + return + skip_tail - ce) / 50 * 22050 // 256)
  mu    = concat(prompt_cond, cond)                     [1,L,384], L = P + n_mel
  x     = z (N(0,1), [1,80,L]); x[..., :P] = 0; prompt_x = [prompt_mel | 0]
  t_span = linspace(0, 1, steps + 1) (float32, PyTorch's formula: ``linspace_f32``); t = 0
  for k in 1..steps:  dt = t_span[k] - t_span[k-1]
                      x = dit_step(x, prompt_x, mu, style, t, dt, cfg, P)  (CFG + Euler + prompt re-zero)
                      t = t + dt                         (float32 accumulation, as upstream)
  mel_out = x[..., P:]                                  [1,80,n_mel]
  vocoder, by ``hift_mode``:
    "full"     wav = hift(mel_out, harm_phase, sine_noise)     [1, 256 n_mel] at 22.05 kHz (upstream, one graph)
    "partial"  f0 = hift_f0(mel_out)                           [1,n_mel]: the whole tick, so the same F0
               d  = first returned frame - hift_margin         (the decoder's receptive field: 12 frames
                                                                reproduce the full decode, > 90 dB)
               wav[256 d:] = hift_nsf(mel_out[..., d:], f0[:, d:], phase_d, sine_noise[..., 256 d:])
               phase_d = harm_phase + 2 pi nsf_start_cycles(f0, d)   (parity: upstream's draw carried to
                                                                frame d), or set by ``phase_anchor``
    "split"    the same two graphs over the whole window (d = 0)
  out = wav[-(return + skip_tail) * 441 : -skip_tail * 441]   (the block plus SOLA margins)

``cfg_rate == 0`` uses ``dit_step1`` (batch 1, as upstream; 1.5x cheaper). ``use_step_graph=False``
runs the bare estimator ``dit`` instead and does the CFG combination and the update on the host
(the same arithmetic, checked equal by the parity test). Random inputs (``z``, ``harm_phase`` = the
9 harmonic start phases in radians, fundamental 0, ``sine_noise`` N(0,1) [1,9,S]) come from the
caller, so the Rust engine can use time-anchored noise and the parity test can feed upstream's own
draws.

Coherent ticks (the engine's time-anchored levers; ``parity.py`` ``stream_join`` measures them).
Upstream draws a fresh z, fresh harmonic phases and fresh sine noise every tick, so the two ticks
in a SOLA crossfade are two different renditions of the overlapping audio: their mels agree to about
14 dB there, and their waveforms line up at no fixed offset (measured, ``upstream_readme``; ADR 0006).
  - Time base: a tick turns ``lr_in`` content frames (20 ms each) into n_mel mel frames (11.6 ms
    each), e.g. 0.74 s of content into 0.755 s of audio for ``upstream_readme``, so one block of
    input moves the content by ``Preset.frames_per_block()`` mel frames (15.81 there), not by the
    block's 15.5 frames of output.
  - DiT noise: anchor z on the mel-frame grid of that time base, z[..., P + j] = Z(k F + j) for
    tick k with F = round(frames_per_block) (a counter-based draw per absolute frame). The
    overlapping mels then agree better and the waveforms line up at a fixed offset: the previous
    tick's matching sample is F * 256 samples on, not one block's output samples (-127 samples
    from upstream's join for ``upstream_readme``, outside SOLA's 0..441 search). Choosing the block
    so that F * 256 equals the block's output samples puts that offset at 0.
  - NSF phase: HiFT predicts its own F0 from each tick's mel, so even a phase carried from the
    window start drifts by 2 pi h df t before the returned audio (about 17 rad for the 9th harmonic
    after 0.3 s at df = 1 Hz). ``partial`` mode knows the tick's F0 before synthesis, so it anchors
    the phase where the ticks meet:
      1. after tick k keep (f0_k[d:], phase_k), the decoded slice's F0 and harm_phase;
      2. for tick k+1, (n_this, n_prev) = ``Preset.join_anchor(F * 256)`` (the middle of the SOLA
         crossfade at the start of tick k+1's returned audio, counted from tick k+1's slice start
         and from tick k's, whose matching sample is F * 256 later);
         target = nsf_cycles_at(f0_k[d:], phase_k, n_prev)
         tick(..., hift_mode="partial", phase_anchor=(n_this, target)).
    The phases then agree exactly at the anchor; only the F0 difference over half a crossfade
    (20 ms) remains.
"""

from __future__ import annotations

import dataclasses

import numpy as np

SR = 22050
HOP = 256
ZC_16K = 320          # 20 ms at 16 kHz
ZC_22K = SR // 50     # 441 samples = 20 ms at 22.05 kHz
N_HARM = 9            # HiFT: the fundamental + 8 harmonics
HIFT_MARGIN = 12      # frames of left context hift_nsf needs to reproduce the full decode


def linspace_f32(n_steps):
    """``torch.linspace(0, 1, n_steps + 1)`` in float32, bit for bit: step = 1 / n in float32;
    the first half counts up from 0, the second half down from 1."""
    steps = n_steps + 1
    step = np.float32(1.0) / np.float32(n_steps)
    half = steps // 2
    out = np.empty(steps, np.float32)
    for i in range(steps):
        out[i] = np.float32(i) * step if i < half else np.float32(1.0) - step * np.float32(steps - i - 1)
    return out


def _frac32(x):
    x = np.asarray(x, np.float32)
    return (x - np.floor(x)).astype(np.float32)


def nsf_frame_cycles(f0):
    """The ``hift`` / ``hift_nsf`` graphs' NSF phase arithmetic in numpy float32: per harmonic h,
    ``inc`` = h f0 / sr (cycles per sample, f0 held per 256-sample frame) and ``start`` = the phase
    (cycles, wrapped) before each frame, relative to the window's first sample. f0 [1,M] or [M]
    -> (inc, start), each [9, M]."""
    f0 = np.asarray(f0, np.float32).reshape(-1)
    h = np.arange(1, N_HARM + 1, dtype=np.float32)[:, None]
    inc = ((f0[None, :] * h) / np.float32(SR)).astype(np.float32)
    adv = _frac32(inc * np.float32(HOP))
    start = _frac32(np.cumsum(adv, axis=-1, dtype=np.float32) - adv)
    return inc, start


def nsf_start_cycles(f0, frame):
    """Phase (cycles, [9]) at the first sample of ``frame`` relative to the window start: what
    ``harm_phase`` must advance by (times 2 pi) for a slice starting at ``frame`` to continue the
    whole window's excitation."""
    return nsf_frame_cycles(f0)[1][:, frame]


def nsf_cycles_at(f0, harm_phase, n):
    """Phase (cycles in [0, 1), [9]) of excitation sample ``n`` of a window that starts at
    ``harm_phase`` (radians, [1,9,1]); the graph's sine argument is
    ``2 pi frac(start_j + inc_j (n % 256 + 1)) + harm_phase`` with j = n // 256."""
    inc, start = nsf_frame_cycles(f0)
    j, k = int(n) // HOP, int(n) % HOP + 1
    ph = _frac32(start[:, j] + inc[:, j] * np.float32(k)).astype(np.float64)
    c = ph + np.asarray(harm_phase, np.float64).reshape(-1) / (2 * np.pi)
    return c - np.floor(c)


def harm_phase_for_anchor(f0, n, target_cycles):
    """``harm_phase`` (radians, [1,9,1], each in [-pi, pi)) for a window with this F0 whose
    excitation sample ``n`` must have phase ``target_cycles`` ([9], cycles): the time-anchored,
    phase-continuous NSF lever (see the module docstring)."""
    own = nsf_cycles_at(f0, np.zeros((1, N_HARM, 1)), n)
    d = np.asarray(target_cycles, np.float64).reshape(-1) - own
    d = d - np.floor(d + 0.5)                                            # [-0.5, 0.5) cycles
    return (2 * np.pi * d).astype(np.float32).reshape(1, N_HARM, 1)


@dataclasses.dataclass
class Preset:
    """Upstream real-time GUI parameters (seconds). ``upstream_readme`` is the README's RTX 3060
    Laptop recommendation: steps 10, cfg 0.7, prompt 3 s, block 0.18 s, crossfade 0.04 s, left
    context 2.5 s (DiT sees 0.5 s of it), right context 0.02 s."""
    name: str = "upstream_readme"
    steps: int = 10
    cfg_rate: float = 0.7
    max_prompt: float = 3.0
    block: float = 0.18
    crossfade: float = 0.04
    extra_ce: float = 2.5     # left context of the content encoder
    extra_dit: float = 0.5    # left context the DiT keeps (upstream "extra_time")
    extra_right: float = 0.02
    hift_margin: int = HIFT_MARGIN   # partial decode: frames of decoder context before the returned audio

    def units(self):
        u = lambda s: int(np.round(s * 50))  # noqa: E731  (upstream rounds each part to 20 ms)
        extra, cf, block, right = u(self.extra_ce), u(self.crossfade), u(self.block), u(self.extra_right)
        sola_buffer, sola_search = min(cf, 4), 1
        return {
            "window_units": extra + cf + sola_search + block + right,
            "skip_head": extra, "skip_tail": right, "return_length": block + sola_buffer + sola_search,
            "ce_frames": int((self.extra_ce - self.extra_dit) * 50),
        }

    def shapes(self, prompt_seconds=None):
        u = self.units()
        n16 = ZC_16K * u["window_units"]
        t_x = (n16 - 400) // 320 + 1
        n_mel = int((u["skip_head"] + u["return_length"] + u["skip_tail"] - u["ce_frames"]) / 50 * SR // HOP)
        ps = self.max_prompt if prompt_seconds is None else min(prompt_seconds, self.max_prompt)
        p = (int(SR * ps) + 768 - 1024) // HOP + 1
        out_samples, tail = u["return_length"] * ZC_22K, u["skip_tail"] * ZC_22K
        out_start = n_mel * HOP - out_samples - tail                      # first returned sample of the window
        dec_start = max(0, out_start // HOP - self.hift_margin)          # partial decode's first frame
        return {**u, "n16": n16, "xlsr_frames": t_x, "lr_in": t_x - u["ce_frames"], "n_mel": n_mel, "prompt_frames": p,
                "L": p + n_mel, "hift_samples": n_mel * HOP, "out_samples": out_samples, "out_start": out_start,
                "dec_start": dec_start, "dec_frames": n_mel - dec_start, "dec_samples": (n_mel - dec_start) * HOP}

    def frames_per_block(self):
        """How far one block of input moves the content on a tick's mel-frame grid: lr maps
        ``lr_in`` content frames (20 ms) onto n_mel mel frames, so a block of b seconds is
        b / (lr_in * 0.02) * n_mel frames (15.81 for ``upstream_readme``, against 15.5 frames of
        output audio per block)."""
        s = self.shapes()
        return self.block / (s["lr_in"] * 0.02) * s["n_mel"]

    def join_anchor(self, shift=None):
        """Phase-continuous NSF across ticks: the anchor (the middle of the SOLA crossfade that
        starts this tick's returned audio) as a sample index of this tick's partial-decode slice,
        and the sample of the previous tick's slice that holds the same audio: ``(n_this, n_prev)``,
        n_prev = n_this + ``shift`` (default: the block's output samples; with DiT noise anchored on
        the mel-frame grid, F * 256, see the module docstring)."""
        s = self.shapes()
        cf = int(np.round(self.crossfade * 50))
        sola_buffer = min(cf, 4)                                          # as ``units``
        n_this = s["out_start"] - s["dec_start"] * HOP + sola_buffer * ZC_22K // 2
        return n_this, n_this + (int(np.round(self.block * 50)) * ZC_22K if shift is None else int(shift))


PRESETS = {
    "upstream_readme": Preset(),
    # Sigil candidates: shorter left context for the encoder, fewer steps; same model and prompt.
    "sigil_low": Preset("sigil_low", steps=6, cfg_rate=0.7, max_prompt=3.0, block=0.10, extra_ce=1.0, extra_dit=0.3),
    "sigil_lean": Preset("sigil_lean", steps=4, cfg_rate=0.0, max_prompt=1.5, block=0.10, extra_ce=1.0, extra_dit=0.3),
}


class SigilZS:
    """The exported graphs driven like the engine will drive them (sessions from ``common.session``)."""

    def __init__(self, sessions):
        self.s = sessions  # dict: xlsr, lr, dit, dit_step, dit_step1, hift, hift_f0, hift_nsf, campplus, mel

    def _run(self, name, feeds):
        sess = self.s[name]
        names = {i.name for i in sess.get_inputs()}
        return sess.run(None, {k: v for k, v in feeds.items() if k in names})

    def prepare_reference(self, ref22k, ref16k, fbank_centered):
        """``ref22k`` / ``ref16k``: the reference cut to max_prompt seconds (float32 1-D);
        ``fbank_centered``: Kaldi fbank of ref16k minus its per-bin mean, [F,80]."""
        s_ori = self._run("xlsr", {"wav16k": ref16k[None, :].astype(np.float32)})[0]
        style = self._run("campplus", {"fbank": fbank_centered[None].astype(np.float32)})[0]
        prompt_mel = self._run("mel", {"wav22k": ref22k[None, :].astype(np.float32)})[0]
        p = prompt_mel.shape[-1]
        prompt_cond = self._run("lr", {"feats": s_ori, "n_frames": np.array([p], np.int64)})[0]
        return {"style": style, "prompt_mel": prompt_mel, "prompt_cond": prompt_cond}

    def cfm(self, z, prompt_mel, mu, style, steps, cfg_rate, use_step_graph=True):
        p = prompt_mel.shape[-1]
        x = z.astype(np.float32).copy()
        x[..., :p] = 0.0
        prompt_x = np.zeros_like(x)
        prompt_x[..., :p] = prompt_mel[..., :p]
        t_span = linspace_f32(steps)
        t = t_span[0]
        big_l = x.shape[-1]
        for k in range(1, steps + 1):
            dt = np.float32(t_span[k] - t_span[k - 1])
            if cfg_rate > 0 and use_step_graph:
                x = self._run("dit_step", {"x": x, "prompt_x": prompt_x, "cond": mu, "style": style,
                                           "t": np.array([t], np.float32), "dt": np.array([dt], np.float32),
                                           "cfg_rate": np.array([cfg_rate], np.float32), "prompt_len": np.array([p], np.int64)})[0]
            elif cfg_rate > 0:
                v = self._run("dit", {"x": np.concatenate([x, x]), "prompt_x": np.concatenate([prompt_x, np.zeros_like(prompt_x)]),
                                      "x_lens": np.array([big_l, big_l], np.int64), "t": np.array([t, t], np.float32),
                                      "style": np.concatenate([style, np.zeros_like(style)]),
                                      "cond": np.concatenate([mu, np.zeros_like(mu)])})[0]
                v = np.float32(1.0 + cfg_rate) * v[0:1] - np.float32(cfg_rate) * v[1:2]
                x = x + dt * v
                x[..., :p] = 0.0
            elif use_step_graph:
                x = self._run("dit_step1", {"x": x, "prompt_x": prompt_x, "cond": mu, "style": style,
                                            "t": np.array([t], np.float32), "dt": np.array([dt], np.float32),
                                            "prompt_len": np.array([p], np.int64)})[0]
            else:
                v = self._run("dit", {"x": x, "prompt_x": prompt_x, "x_lens": np.array([big_l], np.int64),
                                      "t": np.array([t], np.float32), "style": style, "cond": mu})[0]
                x = x + dt * v
                x[..., :p] = 0.0
            t = np.float32(t + dt)
        return x

    def vocode(self, mel_out, harm_phase, sine_noise, preset: Preset, hift_mode="full", phase_anchor=None):
        """``mel_out`` -> the tick's 22.05 kHz audio (``hift_mode``, see the module docstring).
        Returns (wav, info); in ``partial`` mode ``wav`` holds only the decoded slice, which starts
        at window sample ``info["dec_start"] * 256``. ``phase_anchor`` = (n, target_cycles) sets the
        slice's harm_phase so that its excitation sample n has that phase (the engine's carried
        phase); otherwise ``harm_phase`` is the window-start phase (upstream's draw), carried to the
        slice's first sample."""
        harm_phase = np.asarray(harm_phase, np.float32).reshape(1, N_HARM, 1)
        sine_noise = np.asarray(sine_noise, np.float32)
        if hift_mode == "full":
            if phase_anchor is not None:
                raise ValueError("phase_anchor needs hift_mode partial or split (the F0 is known only after hift)")
            wav, f0 = self._run("hift", {"mel": mel_out, "harm_phase": harm_phase, "sine_noise": sine_noise})
            return wav[0], {"dec_start": 0, "f0": f0, "harm_phase": harm_phase}
        f0 = self._run("hift_f0", {"mel": mel_out})[0]
        if hift_mode == "split":
            d = 0
        elif hift_mode == "partial":
            u = preset.units()
            out_start = mel_out.shape[-1] * HOP - (u["return_length"] + u["skip_tail"]) * ZC_22K
            d = max(0, out_start // HOP - preset.hift_margin)
        else:
            raise ValueError(f"hift_mode {hift_mode!r}")
        f0_d = np.ascontiguousarray(f0[:, d:])
        if phase_anchor is not None:
            phase = harm_phase_for_anchor(f0_d, phase_anchor[0], phase_anchor[1])
        else:
            phase = (harm_phase.astype(np.float64) + 2 * np.pi * nsf_start_cycles(f0, d).reshape(1, N_HARM, 1)).astype(np.float32)
        wav = self._run("hift_nsf", {"mel": np.ascontiguousarray(mel_out[..., d:]), "f0": f0_d, "harm_phase": phase,
                                     "sine_noise": np.ascontiguousarray(sine_noise[..., d * HOP:])})[0]
        return wav[0], {"dec_start": d, "f0": f0, "f0_dec": f0_d, "harm_phase": phase}

    def tick(self, ref, window16k, preset: Preset, z, harm_phase, sine_noise, use_step_graph=True, hift_mode="full",
             phase_anchor=None):
        u = preset.units()
        s_alt = self._run("xlsr", {"wav16k": window16k[None, :].astype(np.float32)})[0][:, u["ce_frames"]:]
        n_mel = int((u["skip_head"] + u["return_length"] + u["skip_tail"] - u["ce_frames"]) / 50 * SR // HOP)
        cond = self._run("lr", {"feats": s_alt, "n_frames": np.array([n_mel], np.int64)})[0]
        mu = np.concatenate([ref["prompt_cond"], cond], axis=1)
        x = self.cfm(z, ref["prompt_mel"], mu, ref["style"], preset.steps, preset.cfg_rate, use_step_graph)
        mel_out = np.ascontiguousarray(x[..., ref["prompt_mel"].shape[-1]:])
        wav, voc = self.vocode(mel_out, harm_phase, sine_noise, preset, hift_mode, phase_anchor)
        out_len, tail = u["return_length"] * ZC_22K, u["skip_tail"] * ZC_22K
        end = n_mel * HOP - tail - voc["dec_start"] * HOP                 # in the decoded slice's samples
        # upstream slices wav[-out_len - tail : -tail], which is empty when tail == 0
        return wav[end - out_len: end], {"mel_out": mel_out, "cond": cond, **voc}
