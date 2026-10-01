# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference sampler for DDSP-SVC 6.3 on the exported graphs: the numerical reference for the Rust
Euler loop (ADR 0006). It reproduces upstream ``gui_reflow.py`` ``SvcDDSP.infer`` (the model call
``reflow/vocoder.py`` ``Unit2Wav.forward`` and then its volume gate; yxlllc/DDSP-SVC 6.3 @d6dd52fc)
with numpy between graph calls.

Per tick (window = max(block + crossfade + sola_search + 2 x last_delay, block + extra) at 44.1 kHz,
each part truncated to whole samples as upstream ``set_values`` does):
  units   ContentVec 768 L12 (Sigil's ``vc/embedder-contentvec``) twice, on the 16 kHz window and on
          it delayed by 160 samples, interleaved ("tta2x", hop 160 samples)            [1,U,768]
  mel2ph  round(r * arange(T)) clamped to U - 1, r = (512 / 44100) / (160 / 16000), T = N // 512 + 1
  f0      Hz per 512-sample frame (RMVPE, unvoiced frames interpolated), x 2^(key / 12)  [1,T]
  volume  RMS-like envelope (upstream Volume_Extractor: hop 512, window 1024)          [1,T]
  x, cond = ddsp(units, mel2ph, f0, volume, noise N(0,1) [1,512T], start_frame, formant_shift)
          start_frame = int(silence_front * 44100 / 512); silence_front = extra - crossfade - 0.06 s
  t = t_start (float32); dt = float32((1 - t_start) / steps)
  x = t_start * x + (1 - t_start) * z,   z ~ N(0,1) [1,1,128,R], R = T - start_frame
  repeat steps: x = reflow_step(x, t, dt, cond); t = float32(t + dt)
  mel = (x[:, 0] + 1) / 2 * 14 - 12                                                  [1,128,R]
  audio = vocoder(mel, f0[:, -R:], phase0)   (phase0 = 0: upstream; the engine carries it)    [1,512R]
  audio *= volume_mask(volume, threshold)[-512R:]   (SvcDDSP.infer's gate: frames whose volume is
          at or below threshold dBFS (the GUI slider, default -45) are muted; the per-frame 0/1
          mask is upsampled linearly between frames (ddsp.core.upsample), so a gate opens and
          closes over one 512-sample ramp. Without it, silence still comes out as the DDSP noise
          excitation filtered by the flow. ``threshold_db=None`` returns Unit2Wav.forward's audio)

Upstream-exported pairs (``encoder.onnx`` / ``velocity.onnx`` from export_onnx.py) also work with
this loop: pass ``t`` to ``velocity`` as int64 ``round(1000 t)`` (the exporter's example truncates
with ``astype(int64)``, one step early whenever float32 t lands just below a multiple of 1/1000),
and take the mel of the whole window (the upstream graph has no ``start_frame``; slice its
outputs, which differ from upstream PyTorch in the first two frames of the slice).
"""

from __future__ import annotations

import dataclasses

import numpy as np

SR = 44100
HOP = 512
SPEC_MIN, SPEC_MAX = -12.0, 2.0


@dataclasses.dataclass
class Preset:
    name: str = "upstream_gui"
    block: float = 0.5
    crossfade: float = 0.04
    extra: float = 2.0
    steps: int = 20
    t_start: float = 0.7

    def shapes(self):
        sola_search, last_delay = 0.01, 0.02
        block, cf, sola, delay, extra = (int(s * SR) for s in (self.block, self.crossfade, sola_search, last_delay, self.extra))
        n44 = max(block + cf + sola + 2 * delay, block + extra)       # gui_reflow.py input_frame
        t = n44 // HOP + 1
        safe_prefix = self.extra - self.crossfade - 0.01 - 0.02
        silence_front = safe_prefix - 0.03 if safe_prefix > 0.03 else 0.0
        start = int(silence_front * SR / HOP)
        n16 = int(np.ceil(n44 * 16000 / SR))
        units = (max(n16, 400) - 400) // 320 + 1
        return {"n44": n44, "frames": t, "start_frame": start, "reflow_frames": t - start, "n16": n16,
                "units_tta2x": 2 * units, "audio_out": (t - start) * HOP}


PRESETS = {
    # gui_reflow.py defaults (block 0.5 s, crossfade 0.04 s, extra 2 s, 20 steps from t_start 0.7)
    "upstream_gui": Preset(),
    # Sigil candidates for live singing: 100 ms blocks, 1 s of context
    "sigil_singing": Preset("sigil_singing", block=0.1, extra=1.0, steps=10, t_start=0.7),
    "sigil_singing_fast": Preset("sigil_singing_fast", block=0.1, extra=1.0, steps=5, t_start=0.7),
}


def tta2x(feats, feats_shift):
    """Upstream Audio2ContentVec768L12TTA2X: interleave the encoder on the audio and on the audio
    delayed by 160 samples (``feats_shift``), dropping the first frame."""
    n = feats_shift.shape[1] - feats.shape[1]
    if n > 0:
        feats = np.pad(feats, ((0, 0), (0, 1), (0, 0)))
    out = np.concatenate([feats_shift, feats], axis=2).reshape(feats.shape[0], -1, feats.shape[-1])[:, 1:, :]
    if n > 0:
        out = out[:, :-1, :]
    return np.ascontiguousarray(out, np.float32)


def mel2ph(n_frames, n_units, hop=HOP, sr=SR, enc_hop=160, enc_sr=16000):
    """Upstream Units_Encoder.encode alignment (torch.round rounds half to even, as np.round does)."""
    ratio = (hop / sr) / (enc_hop / enc_sr)
    idx = np.round(ratio * np.arange(n_frames)).astype(np.int64)
    return np.minimum(idx, n_units - 1)[None, :]


def volume_mask(volume, threshold_db, n_samples=None, hop=HOP):
    """Upstream ``SvcDDSP.infer``'s output gate: ``mask = volume > 10^(threshold_db / 20)`` per frame
    (``volume`` = ``volume_envelope``, [1,T] or [T]), upsampled to samples by ``ddsp.core.upsample``
    (linear between frame values with align_corners, the last frame held), the last ``n_samples``
    of the T * hop samples. Bit-exact with upstream (the weights are multiples of 1 / hop)."""
    m = (np.asarray(volume, np.float64).reshape(-1) > 10 ** (float(threshold_db) / 20)).astype(np.float64)
    ext = np.concatenate([m, m[-1:]])
    pos = np.arange(m.size * hop, dtype=np.float64) / hop
    i0 = np.floor(pos).astype(np.int64)
    lam = pos - i0
    up = (ext[i0] * (1.0 - lam) + ext[i0 + 1] * lam).astype(np.float32)
    return up if n_samples is None else up[up.size - n_samples:]


def volume_envelope(audio, hop=HOP, win=1024):
    """Upstream Volume_Extractor.extract."""
    n_frames = int(len(audio) // hop) + 1
    a = np.pad(audio, (int(win // 2), int((win + 1) // 2)), mode="reflect")
    a2 = a ** 2
    mean = np.array([np.mean(a[int(n * hop): int(n * hop + win)]) for n in range(n_frames)])
    mean_sq = np.array([np.mean(a2[int(n * hop): int(n * hop + win)]) for n in range(n_frames)])
    return np.sqrt(np.clip(mean_sq - mean ** 2, 0, None)).astype(np.float32)


class DdspSvc:
    def __init__(self, sessions):
        self.s = sessions  # ddsp, reflow_step, vocoder (+ velocity / velocity_f for the host loop)

    def _run(self, name, feeds):
        sess = self.s[name]
        names = {i.name for i in sess.get_inputs()}
        return sess.run(None, {k: v for k, v in feeds.items() if k in names})

    def tick(self, units, m2p, f0, volume, noise, z, start_frame, steps, t_start, formant_shift=0.0, loop="step", phase0=0.0,
             threshold_db=-45.0):
        x, cond = self._run("ddsp", {"hubert": units, "mel2ph": m2p, "f0": f0, "volume": volume, "randn": noise,
                                     "start_frame": np.array([start_frame], np.int64),
                                     "formant_shift": np.array([[formant_shift]], np.float32)})
        x_ddsp = x
        x = (np.float32(t_start) * x + np.float32(1.0 - t_start) * z).astype(np.float32)
        t = np.float32(t_start)
        dt = np.float32((1.0 - t_start) / steps)
        for _ in range(steps):
            if loop == "step":
                x = self._run("reflow_step", {"x": x, "t": np.array([t], np.float32), "dt": np.array([dt], np.float32), "cond": cond})[0]
            elif loop == "float":
                x = x + self._run("velocity_f", {"x": x, "t": np.array([t], np.float32), "cond": cond})[0] * dt
            else:  # upstream velocity.onnx, int64 time
                ti = np.floor(np.float32(1000.0) * t) if loop == "int_trunc" else np.round(np.float32(1000.0) * t)
                x = x + self._run("velocity", {"x": x, "t": np.array([ti], np.int64), "cond": cond})[0] * dt
            t = np.float32(t + dt)
        mel = ((x[:, 0] + 1.0) / 2.0 * (SPEC_MAX - SPEC_MIN) + SPEC_MIN).astype(np.float32)
        r = mel.shape[-1]
        audio = self._run("vocoder", {"mel": mel, "f0": np.ascontiguousarray(f0[:, -r:]), "phase0": np.array([phase0], np.float32)})[0][0]
        if threshold_db is not None:
            audio = audio * volume_mask(volume, threshold_db, audio.size)
        return audio, {"x_ddsp": x_ddsp, "cond": cond, "mel": mel}
