#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""StreamVoiceAnon sidecar for Sigil (protocol v2; the `streamvoiceanon` recipe): the privacy mode.

StreamVoiceAnon (Plachtaa/StreamVoiceAnon: streaming speaker anonymization with an autoregressive
codec-token model; the authors report 237 ms end to end with a 92 ms chunk on an RTX 3060 Laptop)
re-voices speech toward a reference speaker while removing the source speaker's identity;
``--alpha`` below 1 mixes noise into the speaker vectors for stronger anonymization. It has no ONNX
path, so Sigil runs it here and drives it over protocol v2 under its latency budget.

The adapter drives the repository's own real-time wrapper (``evaluations/infer_arvc.py``,
``InferenceWrapper.process_one_chunk``), which keeps one streaming state and one reference voice
for the whole process (``single_stream``): every Sigil voice may connect (Sigil loads a voice's
replacement while it still plays, and keeps idle voices loaded), and one converts at a time. A
voice takes the wrapper over when its stream (re)starts (the first frame after a voice switch),
which installs its own reference voice and alpha and resets the wrapper; the voice being replaced
is answered ``superseded`` for the moment before the switch. Its chunk is ``decode_chunk_frames``
codec frames (``SAMPLES_PER_FRAME`` samples each at the wrapper's rate); the converted audio goes
through a FIFO that holds the model's delay (``--delay`` codec frames) so every reply is exactly
one frame long.

Setup (manual: the checkpoints come from Hugging Face):

1. ``git clone https://github.com/Plachtaa/StreamVoiceAnon`` and ``pip install -r requirements.txt``
   (on Windows also ``pip install triton-windows==3.2.0.post13``), plus ``websockets``;
2. ``hf download Plachta/StreamVoiceAnon --local-dir pretrained_checkpoints/`` in the repository;
3. in Sigil, make a sidecar voice with the recipe ``streamvoiceanon``: the environment's
   ``python.exe`` as the interpreter, the repository folder as the model folder, and the voice
   library clip of the target (pseudo) speaker as its reference;
4. check it by hand first: ``python streamvoiceanon_sidecar.py --model-dir <repo> --self-test ref.wav``.

Written against the repository as published in 2026-09 (``InferenceWrapper(config_path,
checkpoint_path, compile_ar, compile_decoder, compile_encoder, fp16)``, ``process_one_chunk(chunk,
pitch_shift)``); the method that loads the reference voice is found by name (``--reference-method``
overrides it).
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigil_vc_v2 as v2  # noqa: E402

LOG = logging.getLogger("sigil-sidecar.streamvoiceanon")
REFERENCE_METHODS = ("set_reference", "load_reference", "prepare_reference", "update_reference", "set_ref", "load_ref")


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, np.float32).reshape(-1)


class SvaStream(v2.Stream):
    """One Sigil voice: its reference voice and alpha, installed in the shared wrapper while it
    holds it (:meth:`activate`)."""

    def __init__(self, conv: "SvaConverter"):
        self.c = conv
        self.ref = None
        self.alpha = conv.alpha
        self.fifo = v2.OutputFifo(conv.delay_samples)

    def set_reference(self, prepared):
        self.ref = prepared
        if self.c.active is self:
            self.c.apply_reference(self.ref, self.alpha)

    def set_params(self, params):
        alpha = params.get("alpha")
        if isinstance(alpha, (int, float)) and 0.0 <= float(alpha) <= 1.0 and float(alpha) != self.alpha:
            self.alpha = float(alpha)
            # The wrapper mixes alpha into the speaker vectors when it loads the reference.
            if self.c.active is self and self.ref is not None:
                self.c.apply_reference(self.ref, self.alpha)

    def activate(self):
        c = self.c
        c.active = self
        if self.ref is not None:
            c.apply_reference(self.ref, self.alpha)
        c.reset_upstream()
        self.fifo = v2.OutputFifo(c.delay_samples)

    def process(self, pcm, pitch, discontinuity):
        c = self.c
        if c.active is not self:
            # Outside the server's arbitration (warm-up, self-test): take the wrapper first.
            self.activate()
        elif discontinuity and c.reset_on_gap:
            # Nothing from before the gap plays after it: a fresh wrapper state and delay line.
            c.reset_upstream()
            self.fifo = v2.OutputFifo(c.delay_samples)
        out = c.convert_chunk(pcm, pitch)
        self.fifo.push(out)
        return self.fifo.pop(c.frame_out)

    def close(self):
        # Another voice may hold the wrapper: leave it alone (the next holder resets it).
        if self.c.active is self:
            self.c.active = None


class SvaConverter(v2.Converter):
    name = "streamvoiceanon"
    reference = "required"
    pitch = True
    params = ("alpha",)
    # One streaming state and one reference voice in the wrapper: the server lets one voice
    # convert at a time (the others stay connected).
    single_stream = True
    # The reference is only resampled here (the wrapper loads it in `apply_reference`).
    parallel_reference = True

    def __init__(self, args):
        self.args = args
        self.repo = os.path.abspath(args.model_dir) if args.model_dir else ""
        self.device = "cuda" if args.device == "cuda" else "cpu"
        self.alpha = float(args.alpha)
        self.reset_on_gap = not args.keep_state_on_gap
        #: The stream whose reference the wrapper holds.
        self.active = None
        if args.device == "dml":
            LOG.warning("StreamVoiceAnon runs on PyTorch: DirectML is not supported, using the CPU")

    def reset_upstream(self):
        reset = getattr(self.wrapper, "reset", None) or getattr(self.wrapper, "reset_state", None)
        if callable(reset):
            reset()

    def load(self):
        if not self.repo or not os.path.isdir(os.path.join(self.repo, "evaluations")):
            raise SystemExit("streamvoiceanon: pass --model-dir, the StreamVoiceAnon repository folder")
        sys.path.insert(0, self.repo)
        sys.path.insert(0, os.path.join(self.repo, "evaluations"))
        os.chdir(self.repo)  # its configs name checkpoints relative to the repository
        try:
            mod = importlib.import_module("infer_arvc")
        except ImportError as e:
            raise SystemExit(f"streamvoiceanon: cannot import evaluations/infer_arvc.py ({e}); install the repository's requirements")
        wrapper_cls = getattr(mod, "InferenceWrapper", None)
        if wrapper_cls is None:
            raise SystemExit("streamvoiceanon: evaluations/infer_arvc.py has no InferenceWrapper (a newer version?)")
        kwargs = dict(compile_ar=self.args.compile, compile_decoder=self.args.compile, compile_encoder=self.args.compile, fp16=self.args.fp16)
        accepted = inspect.signature(wrapper_cls).parameters
        kwargs = {k: v for k, v in kwargs.items() if k in accepted}
        self.wrapper = wrapper_cls(self.args.config, self.args.checkpoint, **kwargs)
        if not hasattr(self.wrapper, "process_one_chunk"):
            raise SystemExit("streamvoiceanon: InferenceWrapper has no process_one_chunk (a newer version?)")
        for name, value in (("set_delay", self.args.delay),):
            fn = getattr(self.wrapper, name, None)
            if callable(fn) and value is not None:
                fn(value)
        self.sr = int(getattr(self.wrapper, "sr", 0) or self.args.rate)
        spf = int(getattr(mod, "SAMPLES_PER_FRAME", 2048))
        self.input_rates = (self.sr,)
        self.output_rate = int(getattr(self.wrapper, "output_sr", 0) or self.sr)
        self.reference_rate = self.sr
        self.frame_in = spf * max(1, int(self.args.decode_chunk_frames))
        self.frame_out = self.frame_in * self.output_rate // self.sr
        self.frame_ms = self.frame_in * 1000.0 / self.sr
        self.delay_samples = int((self.args.delay or 0) * spf * self.output_rate // self.sr)
        self.reference_method = self.args.reference_method or next(
            (m for m in REFERENCE_METHODS if callable(getattr(self.wrapper, m, None))), "")
        if not self.reference_method:
            raise SystemExit("streamvoiceanon: no reference loader found on InferenceWrapper (looked for "
                             f"{', '.join(REFERENCE_METHODS)}); pass --reference-method NAME")

    def prepare_reference(self, pcm, rate, meta):
        # The wrapper loads the reference from a file (librosa at its own rate): keep the clip.
        return v2.resample(pcm, rate, self.sr)

    def apply_reference(self, pcm, alpha: float):
        """Load a voice's reference (and its alpha) into the wrapper (inference thread)."""
        fn = getattr(self.wrapper, self.reference_method)
        with tempfile.TemporaryDirectory(prefix="sigil-sva-") as d:
            path = os.path.join(d, "reference.wav")
            v2.write_wav16(path, pcm, self.sr)
            attempts = ({"ref_paths": [path], "alpha": alpha}, {"ref_path": path, "alpha": alpha},
                        {"ref_paths": [path]}, {"ref_path": path})
            params = inspect.signature(fn).parameters
            for kw in attempts:
                if all(k in params for k in kw):
                    fn(**kw)
                    return
            fn(path)

    def convert_chunk(self, pcm, pitch):
        import torch

        w = self.wrapper
        with torch.inference_mode():
            try:
                out = w.process_one_chunk(torch.from_numpy(np.ascontiguousarray(pcm)), pitch_shift=float(pitch))
            except (TypeError, AttributeError):
                out = w.process_one_chunk(np.ascontiguousarray(pcm), pitch_shift=float(pitch))
        if isinstance(out, (tuple, list)):
            out = out[0]
        return _to_numpy(out) if out is not None else np.zeros(0, np.float32)

    def warmup(self):
        stream = self.open_stream(self.sr, self.frame_in)
        stream.set_reference(self.prepare_reference(v2.synthetic_voice(4.0, self.sr, 210.0), self.sr, {}))
        stream.activate()
        voice = v2.synthetic_voice(1.0, self.sr)[: self.frame_in]
        self.time_warmup(lambda: stream.process(voice, 0.0, False))
        stream.close()

    def frame_samples(self, rate_in, proposed):
        return self.frame_in

    def open_stream(self, rate_in, frame_in):
        return SvaStream(self)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/config_firefly_arvcasr_8192_delay0_8.yaml", help="relative to the repository")
    p.add_argument("--checkpoint", default="pretrained_checkpoints/dual_ar_delay_0_8.pth", help="relative to the repository")
    p.add_argument("--delay", type=int, default=None, help="look-ahead in codec frames (the model's default when unset)")
    p.add_argument("--decode-chunk-frames", type=int, default=1, help="codec frames per chunk (the frame)")
    p.add_argument("--alpha", type=float, default=1.0, help="speaker-vector noise mixing: 1 = the reference, lower = more anonymous")
    p.add_argument("--rate", type=int, default=22050, help="the wrapper's rate when it does not say")
    p.add_argument("--compile", action="store_true", help="torch.compile the model (slow first start)")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--reference-method", default="", help="InferenceWrapper method that loads the reference voice")
    p.add_argument("--keep-state-on-gap", action="store_true", help="do not reset the stream after a gap")


if __name__ == "__main__":
    v2.main(SvaConverter, "StreamVoiceAnon (privacy mode) sidecar for Sigil", add_arguments, echo_reference="required")
