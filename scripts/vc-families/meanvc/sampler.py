# SPDX-License-Identifier: AGPL-3.0-or-later
"""Reference streaming loop for the MeanVC2 family on the exported graphs: the numerical reference a
Rust port is checked against (P3-H, ADR 0006).

It reproduces upstream ``runtime/run_rt.py`` ``VCRunner`` (ASLP-lab/MeanVC2@13acf84c, 40 ms model)
with numpy between graph calls, state for state:

per audio chunk (upstream: 2560 samples = 160 ms)
  padded = samples_cache + chunk; F = (len - 400) // 160 + 1 fbank frames; samples_cache = padded[160 F:]
  fb = frame_cache + fbank(padded)
  ASR every 8 frames over an 11-frame window: bn [2,256] per call; offset += 2 (reset below 4000)
  bn = [last bn of the previous chunk | new bn]; bn_up = linear (align_corners) to 17 frames, first dropped
  bn_buffer += bn_up
  while bn_buffer holds 8 frames (4 current + 4 future):
    x = z [1,8,80] with x[:, :4] = the previous call's last 4 noise frames (upstream's anchored noise)
    for i in 0..N-1 (N = 2): t = 1 - i/N, r = max(0, t - 1/N); x = dit_step(x, t, r, 1/N, bn[:8], ...),
        every step from the same KV cache; the last step's cache is kept
    mel = x[:, :4] (10 ms frames); bn_buffer drops 4 frames; offset += 4 (reset at 4000)
    vocos over [2 previous mel frames | 4 new] -> 800 samples; a 160-sample linear crossfade with the
    previous tail; 640 new samples (40 ms) out

``flush`` is upstream ``process_file``'s tail handling (pads the last chunks with the last BN frame).
The random ``z`` draws come from the caller (upstream draws ``torch.randn(1, 8, 80)``), so the Rust
engine can feed counter-based time-anchored noise and the parity test can feed upstream's own draws.
"""

from __future__ import annotations

import numpy as np

CHUNK = 4
BLOCK = 4
BN_WINDOW = 11
BN_STRIDE = 8
RCS = 4
ASR_OFFSET_INIT = 4
ASR_OFFSET_STEP = 2
CAP = 12
VC_CHUNK = 16            # upstream interpolates every audio chunk's BN to 16 + 1 frames
VOC_OVERLAP = 2          # mel frames
WAV_OVERLAP = 160        # samples
AUDIO_CHUNK = 2560       # upstream CHUNK


def linspace_f32(start, end, steps):
    """``torch.linspace(start, end, steps)`` in float32, bit for bit (the first half counts from
    ``start``, the second half back from ``end``)."""
    step = (np.float32(end) - np.float32(start)) / np.float32(steps - 1)
    half = steps // 2
    out = np.empty(steps, np.float32)
    for i in range(steps):
        out[i] = np.float32(start) + step * np.float32(i) if i < half else np.float32(end) - step * np.float32(steps - i - 1)
    return out


def upsample_linear_align_corners(x, out_len):
    """``F.interpolate(x.transpose(1, 2), size=out_len, mode='linear', align_corners=True)`` on
    [1,T,C] float32 (PyTorch's CPU kernel: scale = (T-1)/(out-1) in float32, i0 = floor(src))."""
    t_in = x.shape[1]
    scale = np.float32(t_in - 1) / np.float32(out_len - 1) if out_len > 1 else np.float32(0)
    src = scale * np.arange(out_len, dtype=np.float32)
    i0 = np.floor(src).astype(np.int64)
    i1 = np.minimum(i0 + 1, t_in - 1)
    l1 = (src - i0.astype(np.float32))[:, None]
    l0 = np.float32(1.0) - l1
    return (l0 * x[0, i0] + l1 * x[0, i1])[None].astype(np.float32)


class SigilMeanVC:
    """The exported graphs driven like the engine will drive them. ``sessions``: fbank, asr, gtm,
    dit_step, vocos (``common.session``); ``spks``: the [1,256] L2-normalised speaker embedding."""

    def __init__(self, sessions, spks, steps=2, audio_chunk=AUDIO_CHUNK):
        self.s = sessions
        self.spks = np.ascontiguousarray(spks, np.float32)
        self.steps = steps
        self.audio_chunk = audio_chunk
        self.k_mem, self.v_mem = self._run("gtm", {"spks": self.spks})
        dit_cache = [i for i in self.s["dit_step"].get_inputs() if i.name == "cache"][0].shape
        self.cache_shape = tuple(int(d) for d in dit_cache)
        self.down = linspace_f32(1.0, 0.0, WAV_OVERLAP)
        self.up = linspace_f32(0.0, 1.0, WAV_OVERLAP)
        self.reset()

    def _run(self, name, feeds):
        sess = self.s[name]
        names = {i.name for i in sess.get_inputs()}
        return sess.run(None, {k: v for k, v in feeds.items() if k in names})

    def reset(self):
        self.samples_cache = np.zeros(0, np.float32)
        self.frame_cache = None
        self.encoder_output_cache = None
        self.asr_offset = ASR_OFFSET_INIT
        self.att_cache = np.zeros((6, 4, RCS, 128), np.float32)
        self.cnn_cache = np.zeros((6, 1, 256, 8), np.float32)
        self.kv = np.zeros(self.cache_shape, np.float32)
        self.n_valid = 0
        self.vc_offset = 0
        self.noise_cache = None
        self.vocoder_cache = None
        self.last_wav = None
        self.bn_buffer = None
        self.trace = {"bn_up": [], "mel": [], "wav": []}

    # --- ASR ---------------------------------------------------------------------------------
    def encode_chunk(self, samples):
        padded = np.concatenate((self.samples_cache, samples)) if self.samples_cache.size else samples
        if len(padded) < 400:
            self.samples_cache = padded
            return None
        fb = self._run("fbank", {"wav16k": padded[None].astype(np.float32)})[0][0]
        self.samples_cache = padded[160 * fb.shape[0]:]
        if self.frame_cache is not None:
            fb = np.concatenate([self.frame_cache, fb], axis=0)
        if self.asr_offset >= 4000:
            self.asr_offset = max(self.att_cache.shape[2], self.asr_offset - 4000)
        bns = []
        i = 0
        while i + BN_WINDOW <= fb.shape[0]:
            out, self.att_cache, self.cnn_cache = self._run("asr", {
                "fbank": np.ascontiguousarray(fb[i:i + BN_WINDOW][None]), "offset": np.array(self.asr_offset, np.int64),
                "rcs": np.array(RCS, np.int64), "att_cache": self.att_cache, "cnn_cache": self.cnn_cache})
            bns.append(out[0])
            self.asr_offset += ASR_OFFSET_STEP
            i += BN_STRIDE
        self.frame_cache = fb[i:]
        if not bns:
            return None
        bn = np.concatenate(bns, axis=0)[None]
        if self.encoder_output_cache is not None:
            bn = np.concatenate([self.encoder_output_cache, bn], axis=1)
        self.encoder_output_cache = bn[:, -1:]
        if bn.shape[1] < 2:
            return None
        return upsample_linear_align_corners(bn, VC_CHUNK + 1)[:, 1:]

    # --- DiT (mean flow) ---------------------------------------------------------------------
    def vc_step(self, cond, z):
        x = np.array(z, np.float32, copy=True)
        if self.noise_cache is not None:
            x[:, :BLOCK] = self.noise_cache
        self.noise_cache = x[:, -BLOCK:].copy()
        if self.vc_offset >= 4000:
            self.vc_offset = 0
            self.kv[...] = 0.0
            self.n_valid = 0
        dt = np.float32(1.0 / self.steps)
        pre = self.kv
        for i in range(self.steps):
            t = 1.0 - i * (1.0 / self.steps)
            r = max(0.0, t - 1.0 / self.steps)
            x, cache = self._run("dit_step", {
                "x": x, "t": np.array([t], np.float32), "r": np.array([r], np.float32), "dt": np.array([dt], np.float32),
                "cond": np.ascontiguousarray(cond), "spks": self.spks, "k_mem": self.k_mem, "v_mem": self.v_mem,
                "offset": np.array([self.vc_offset], np.int64), "n_valid": np.array([self.n_valid], np.int64), "cache": pre})
            if i == self.steps - 1:
                self.kv = cache
        self.n_valid = min(self.n_valid + CHUNK, CAP)
        self.vc_offset += CHUNK
        return np.ascontiguousarray(x[:, :CHUNK].transpose(0, 2, 1))                   # [1,80,4]

    # --- vocoder -----------------------------------------------------------------------------
    def decode_mel(self, mel):
        if self.vocoder_cache is not None:
            mel = np.concatenate([self.vocoder_cache, mel], axis=-1)
        self.vocoder_cache = mel[:, :, -VOC_OVERLAP:]
        wav = self._run("vocos", {"mel": ((mel + np.float32(1.0)) / np.float32(2.0)).astype(np.float32)})[0][0]
        if self.last_wav is not None:
            front = wav[:WAV_OVERLAP]
            smooth = self.last_wav * self.down + front * self.up
            new = np.concatenate([smooth, wav[WAV_OVERLAP:-WAV_OVERLAP]])
        else:
            new = wav[:-WAV_OVERLAP]
        self.last_wav = wav[-WAV_OVERLAP:]
        return new

    def _emit(self, cond, draws):
        mel = self.vc_step(cond, next(draws))
        wav = self.decode_mel(mel)
        self.trace["mel"].append(mel)
        self.trace["wav"].append(wav)
        return wav

    def process_chunk(self, samples, draws):
        bn_new = self.encode_chunk(samples)
        if bn_new is None:
            return None
        self.trace["bn_up"].append(bn_new)
        self.bn_buffer = bn_new if self.bn_buffer is None else np.concatenate([self.bn_buffer, bn_new], axis=1)
        parts = []
        for _ in range(4):
            if self.bn_buffer.shape[1] < CHUNK + BLOCK:
                break
            cond = self.bn_buffer[:, :CHUNK + BLOCK]
            self.bn_buffer = self.bn_buffer[:, CHUNK:]
            parts.append(self._emit(cond, draws))
        return np.concatenate(parts) if parts else None

    def flush(self, draws):
        parts = []
        while self.bn_buffer is not None and self.bn_buffer.shape[1] >= CHUNK:
            cur = self.bn_buffer[:, :CHUNK]
            if self.bn_buffer.shape[1] > CHUNK:
                fut = self.bn_buffer[:, CHUNK:CHUNK + BLOCK]
            else:
                fut = np.repeat(self.bn_buffer[:, -1:], BLOCK, axis=1)
            cond = np.concatenate([cur, fut], axis=1)
            self.bn_buffer = self.bn_buffer[:, CHUNK:]
            if self.bn_buffer.shape[1] == 0:
                self.bn_buffer = None
            parts.append(self._emit(cond, draws))
        if self.bn_buffer is not None and self.bn_buffer.shape[1] > 0:
            n_rem = self.bn_buffer.shape[1]
            cur = np.repeat(self.bn_buffer[:, -1:], CHUNK, axis=1)
            cur[:, :n_rem] = self.bn_buffer
            fut = np.repeat(self.bn_buffer[:, -1:], BLOCK, axis=1)
            self.bn_buffer = None
            parts.append(self._emit(np.concatenate([cur, fut], axis=1), draws))
        if self.last_wav is not None:
            parts.append(self.last_wav)
        return parts

    def convert(self, wav16k, draws):
        """Upstream ``process_file`` on an in-memory utterance (chunks of ``audio_chunk`` samples, no
        padding of the last one, then ``flush``)."""
        self.reset()
        draws = iter(draws)
        out = []
        pos = 0
        while pos < len(wav16k):
            end = min(pos + self.audio_chunk, len(wav16k))
            o = self.process_chunk(np.ascontiguousarray(wav16k[pos:end], np.float32), draws)
            pos = end
            if o is not None:
                out.append(o)
        out.extend(self.flush(draws))
        return np.concatenate(out)
