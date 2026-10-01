#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""X-VC zero-shot sidecar for Sigil (protocol v2; the `xvc` recipe).

X-VC (Jerrister/X-VC: zero-shot streaming voice conversion in a speech codec's latent space; the best
speaker similarity of the research lineup, 16 kHz, about 5 GB in PyTorch, for 12 GB+ GPUs) has no
ONNX path, so Sigil runs it here and drives it over protocol v2 under its latency budget.

The live loop is X-VC's chunkwise inference (paper §3.5): every frame of ``current`` ms, a window of
``chunk`` ms of input (history, current, overlap ``smooth``, ``future`` lookahead) is converted by the
repository's ``run_stream_chunk_forward``; the current region is emitted, its first ``smooth`` ms
cross-faded (cosine) with the overlap kept from the previous step. Defaults are the paper's
streaming setting: chunk 2400, current 120, smooth 20, future 100 ms, so the reply to each frame is
the audio that ends ``smooth + future`` = 120 ms before it (``delay_samples``). The reference voice's
speaker and frame conditions are computed once per clip (``precompute_conditions``).

Setup (manual: the checkpoints are several GB):

1. ``git clone https://github.com/Jerrister/X-VC`` and install its ``requirements.txt`` in a Python
   3.10 environment, plus ``websockets``;
2. download the pretrained models as its README says (GLM-4-Voice tokenizer, ERes2Net speaker
   encoder; paths in ``configs/xvc.yaml``) and the X-VC checkpoint to ``ckpts/xvc.pt``;
3. in Sigil, make a sidecar voice with the recipe ``xvc``: the environment's ``python.exe`` as the
   interpreter and the repository folder as the model folder;
4. check it by hand first: ``python xvc_sidecar.py --model-dir <repo> --self-test reference.wav``.

Written against X-VC's ``bins/infer_utils.py`` (``load_xvc``, ``load_pair_as_tensors``,
``precompute_conditions``, ``run_stream_chunk_forward``) as published in 2026-09; if a newer
version renames them, the error names the function to look for.
"""

from __future__ import annotations

import argparse
import inspect
import logging
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigil_vc_v2 as v2  # noqa: E402

LOG = logging.getLogger("sigil-sidecar.xvc")
UPSTREAM_FUNCTIONS = ("load_xvc", "load_pair_as_tensors", "precompute_conditions", "run_stream_chunk_forward")


def _call(fn, **kwargs):
    """Call ``fn`` with the keyword arguments it accepts (upstream signatures vary by version)."""
    params = inspect.signature(fn).parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        return fn(**kwargs)
    return fn(**{k: v for k, v in kwargs.items() if k in params})


def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    return np.asarray(x, np.float32).reshape(-1)


class XvcStream(v2.Stream):
    def __init__(self, conv: "XvcConverter"):
        self.c = conv
        self.buf = np.zeros(conv.window, np.float32)
        self.overlap = np.zeros(conv.smooth, np.float32)
        self.fade_in, self.fade_out = v2.cosine_fade(conv.smooth)
        self.cond = None

    def set_reference(self, prepared):
        self.cond = prepared

    def process(self, pcm, pitch, discontinuity):
        c = self.c
        if discontinuity:
            # The window's history and lookahead, and the overlap, are from before the gap: the
            # first reply after it would otherwise play their last `smooth + future` ms out of time.
            self.buf[:] = 0.0
            self.overlap[:] = 0.0
        n = pcm.size
        self.buf[:-n] = self.buf[n:]
        self.buf[-n:] = pcm
        if self.cond is None:
            return np.zeros(c.current, np.float32)
        y = v2.fit_length(c.convert_window(self.buf, self.cond), c.window)
        region = y[c.cur_start: c.cur_end + c.smooth]
        head = region[: c.smooth] * self.fade_in + self.overlap * self.fade_out
        self.overlap = region[c.current: c.current + c.smooth].copy()
        return np.concatenate([head, region[c.smooth: c.current]])


class XvcConverter(v2.Converter):
    name = "x-vc"
    reference = "required"
    pitch = False

    def __init__(self, args):
        self.args = args
        self.repo = os.path.abspath(args.model_dir) if args.model_dir else ""
        self.device = "cuda" if args.device == "cuda" else "cpu"
        # A new voice's reference is prepared beside the playing voice's frames, unless the
        # upstream version at hand cannot share the model between two threads.
        self.parallel_reference = not getattr(args, "serial_reference", False)
        if args.device == "dml":
            LOG.warning("X-VC runs on PyTorch: DirectML is not supported, using the CPU")

    def load(self):
        if not self.repo or not os.path.isdir(os.path.join(self.repo, "bins")):
            raise SystemExit("xvc: pass --model-dir, the X-VC repository folder (with bins/ and configs/)")
        sys.path.insert(0, self.repo)
        os.chdir(self.repo)  # configs/xvc.yaml names the pretrained models relative to the repository
        try:
            import bins.infer_utils as iu
        except ImportError as e:
            raise SystemExit(f"xvc: cannot import X-VC's bins.infer_utils ({e}); install the repository's requirements")
        missing = [f for f in UPSTREAM_FUNCTIONS if not hasattr(iu, f)]
        if missing:
            raise SystemExit(f"xvc: X-VC's bins/infer_utils.py has no {', '.join(missing)} (a newer version?)")
        self.iu = iu
        self.run_stream_chunk_forward = iu.run_stream_chunk_forward
        config = os.path.join(self.repo, self.args.config)
        ckpt = os.path.join(self.repo, self.args.ckpt)
        for f in (config, ckpt):
            if not os.path.isfile(f):
                raise SystemExit(f"xvc: {f} is missing (see the X-VC README)")
        device_id = 0 if self.device == "cuda" else "cpu"
        self.cfg, self.model, self.torch_device = iu.load_xvc(config, ckpt, device_id, self.args.ema)
        if hasattr(self.model, "eval"):
            self.model.eval()
        sr = int(self.cfg["sample_rate"]) if "sample_rate" in self.cfg else 16000
        self.input_rates = (sr,)
        self.output_rate = sr
        self.reference_rate = sr
        ms = lambda v: int(round(v * sr / 1000.0))  # noqa: E731
        self.current, self.smooth, self.future = ms(self.args.current_ms), ms(self.args.smooth_ms), ms(self.args.future_ms)
        self.window = ms(self.args.chunk_ms)
        history = self.window - self.current - self.smooth - self.future
        if history < 0:
            raise SystemExit("xvc: --chunk-ms must cover current + smooth + future")
        self.cur_start, self.cur_end = history, history + self.current
        self.frame_ms = self.args.current_ms
        self.delay_samples = self.smooth + self.future

    def convert_window(self, window: np.ndarray, cond) -> np.ndarray:
        """One chunkwise step: the converted window (samples at the model rate)."""
        import torch

        with torch.inference_mode():
            chunk = torch.from_numpy(np.ascontiguousarray(window, np.float32).copy()).to(self.torch_device).view(1, 1, -1)
            out = self.run_stream_chunk_forward(self.model, chunk, cond[0], cond[1])
        if isinstance(out, (tuple, list)):
            out = out[0]
        return _to_numpy(out)

    def prepare_reference(self, pcm, rate, meta):
        # Runs on the server's reference thread while other voices convert (the model is only
        # read: speaker and frame conditions of the clip).
        with tempfile.TemporaryDirectory(prefix="sigil-xvc-") as d:
            path = os.path.join(d, "reference.wav")
            v2.write_wav16(path, v2.resample(pcm, rate, self.reference_rate), self.reference_rate)
            hop = getattr(self.args, "latent_hop", None) or int(self.cfg.get("latent_hop_length", 256) if hasattr(self.cfg, "get") else 256)
            loaded = _call(self.iu.load_pair_as_tensors, source_wav_path=path, target_wav_path=path, cfg=self.cfg,
                           device=self.torch_device, latent_hop_length=hop, mask_target_condition=self.args.mask_target_condition)
        _source, target_wav, target_cond = loaded
        import torch

        with torch.inference_mode():
            return tuple(self.iu.precompute_conditions(self.model, target_wav, target_cond))

    def warmup(self):
        stream = self.open_stream(self.input_rates[0], self.current)
        stream.set_reference(self.prepare_reference(v2.synthetic_voice(4.0, self.reference_rate, 210.0), self.reference_rate, {}))
        voice = v2.synthetic_voice(0.2, self.input_rates[0])[: self.current]
        self.time_warmup(lambda: stream.process(voice, 0.0, False))

    def frame_samples(self, rate_in, proposed):
        return self.current

    def open_stream(self, rate_in, frame_in):
        return XvcStream(self)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default="configs/xvc.yaml", help="X-VC config (relative to the repository)")
    p.add_argument("--ckpt", default="ckpts/xvc.pt", help="X-VC checkpoint (relative to the repository)")
    p.add_argument("--ema", action="store_true", help="load the EMA weights")
    p.add_argument("--chunk-ms", type=int, default=2400, help="processing window (the training segment)")
    p.add_argument("--current-ms", type=int, default=120, help="audio emitted per step (the frame)")
    p.add_argument("--smooth-ms", type=int, default=20, help="cross-faded overlap")
    p.add_argument("--future-ms", type=int, default=100, help="lookahead")
    p.add_argument("--latent-hop", type=int, default=None, help="codec hop (samples); default from the config or 256")
    p.add_argument("--mask-target-condition", action="store_true", help="as upstream's flag of the same name")
    p.add_argument("--serial-reference", action="store_true",
                   help="prepare reference voices between frames instead of beside them (if parallel preparation misbehaves)")


if __name__ == "__main__":
    v2.main(XvcConverter, "X-VC zero-shot sidecar for Sigil", add_arguments, echo_reference="required")
