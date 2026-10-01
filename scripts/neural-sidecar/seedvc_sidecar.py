#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seed-VC real-time zero-shot sidecar for Sigil (protocol v2; the `seedvc` recipe).

The zero-shot fallback while Sigil-ZS (the same model inside Sigil's engine) is not available on a
PC: Seed-VC's real-time model (seed-uvit-tat-xlsr-tiny, 22.05 kHz) on ONNX Runtime, from Sigil's
own export of it (``scripts/vc-families/seed_vc/export.py``: the graphs the in-engine port runs,
parity-tested against upstream). Each tick follows upstream ``real-time-gui.py``: a rolling 16 kHz
window (left context, crossfade, search, block, right context), the reference sampler of
``scripts/vc-families/seed_vc/sampler.py`` (XLS-R content, length regulator, CFM/Euler with
classifier-free guidance, HiFT vocoder), then SOLA at 22.05 kHz. The reply to each block is the
block of audio that ends ``crossfade + search/2 + right context`` before it (``delay_samples``).

The reference voice arrives from Sigil (the voice library clip) and is prepared once per clip
(CAM++ style, prompt mel and prompt condition; cached by sha256 while the sidecar runs), on the
server's reference thread: a voice that loads never holds up the frames of the voice that plays.

Run by Sigil (Voice changer -> sidecar voice, recipe ``seedvc``), or by hand::

    python seedvc_sidecar.py --model-dir <folder with xlsr.onnx, lr.onnx, dit_step.onnx,
        dit_step1.onnx, hift.onnx, campplus.onnx, mel.onnx> [--device cuda|dml|cpu]
        [--preset upstream_readme|sigil_low|sigil_lean] [--steps N] [--cfg-rate X]
    python seedvc_sidecar.py --model-dir ... --self-test reference.wav   # timings, then exit

Needs ``numpy``, ``websockets`` and ``onnxruntime`` (``onnxruntime-gpu`` 1.28 for CUDA 13, or
``onnxruntime-directml``); ``torchaudio`` or ``scipy`` improve the reference front end when
installed. The GPU rung follows ONNX Runtime's providers (CUDA needs the CUDA 13 / cuDNN 9 runtime:
Sigil passes its NVIDIA runtime pack with ``--dll-dir``).

On a GPU the graphs a tick runs are pinned to the preset's shapes (``Routed``): DirectML compiles
fixed shapes, which makes it real time (RTX 4090, ``sigil_low``: 36 ms per 100 ms tick pinned
against 168-243 ms unpinned), and CUDA gains about a quarter. The reference voice's one-off
XLS-R pass runs on the CPU instead of a second copy of XLS-R in GPU memory (0.4-0.6 s per
voice); ``--no-pin`` turns pinning off.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import logging
import os
import sys
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sigil_vc_v2 as v2  # noqa: E402

LOG = logging.getLogger("sigil-sidecar.seedvc")
GRAPH_NAMES = ("xlsr", "lr", "dit_step", "dit_step1", "hift", "campplus", "mel")
SR_IN, SR_OUT = 16000, 22050
UNIT_16K, UNIT_22K = 320, 441  # 20 ms
PROVIDERS = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}


def load_sampler(path: str = ""):
    """Sigil's reference sampler (``scripts/vc-families/seed_vc/sampler.py``)."""
    candidates = [path] if path else [
        os.path.join(HERE, "seed_vc_sampler.py"),
        os.path.join(HERE, "..", "vc-families", "seed_vc", "sampler.py"),
    ]
    for c in candidates:
        if c and os.path.isfile(c):
            spec = importlib.util.spec_from_file_location("sigil_seedvc_sampler", c)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod  # dataclasses look their module up while it loads
            spec.loader.exec_module(mod)
            return mod
    raise SystemExit("seedvc: the Seed-VC sampler was not found (scripts/vc-families/seed_vc/sampler.py); pass --sampler")


def default_graphs_dir() -> str:
    env = os.environ.get("SIGIL_SEEDVC_GRAPHS", "")
    if env:
        return env
    dev = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache", "vc-families", "seed_vc")
    return dev if os.path.isdir(dev) else ""


def import_ort(device: str, dll_dirs):
    """onnxruntime with the CUDA runtime folders on the DLL search path first."""
    if device == "cuda" and hasattr(os, "add_dll_directory"):
        dirs = list(dll_dirs or [])
        cuda = os.environ.get("CUDA_PATH", "")
        if cuda:
            dirs += [os.path.join(cuda, "bin", "x64"), os.path.join(cuda, "bin")]
        for d in dirs:
            if os.path.isdir(d):
                os.add_dll_directory(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
    import onnxruntime as ort

    # Errors only from the environment logger: ORT 1.28 warns once per CUDA session that no plugin
    # EP device is registered before it uses the built-in provider (harmless, and wide-character
    # text that garbles Sigil's helper log).
    try:
        ort.set_default_logger_severity(3)
    except AttributeError:
        pass
    return ort


def kaldi_fbank(wav16k: np.ndarray) -> np.ndarray:
    """Kaldi fbank (80 bins, 25/10 ms, povey window, dither 0) minus its per-bin mean: CAM++'s input,
    as upstream computes it with torchaudio (used when installed)."""
    try:
        import torch
        import torchaudio.compliance.kaldi as kaldi

        fb = kaldi.fbank(torch.from_numpy(np.ascontiguousarray(wav16k, np.float32))[None], num_mel_bins=80, dither=0,
                         sample_frequency=16000)
        return (fb - fb.mean(dim=0, keepdim=True)).numpy()
    except ImportError:
        pass
    x = np.asarray(wav16k, np.float64)
    win, hop, nfft = 400, 160, 512
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n = 1 + (x.size - win) // hop
    frames = np.lib.stride_tricks.as_strided(x, (n, win), (x.strides[0] * hop, x.strides[0])).copy()
    frames -= frames.mean(axis=1, keepdims=True)
    frames[:, 1:] -= 0.97 * frames[:, :-1].copy()
    frames[:, 0] -= 0.97 * frames[:, 0]
    frames *= (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(win) / (win - 1))) ** 0.85
    power = np.abs(np.fft.rfft(frames, nfft)) ** 2
    mel = lambda f: 1127.0 * np.log(1.0 + f / 700.0)  # noqa: E731
    lo, hi = mel(20.0), mel(8000.0)
    delta = (hi - lo) / 81
    fft_mel = mel(np.arange(nfft // 2) * 16000.0 / nfft)
    banks = np.zeros((80, nfft // 2 + 1))
    for b in range(80):
        left, center, right = lo + b * delta, lo + (b + 1) * delta, lo + (b + 2) * delta
        up = (fft_mel - left) / (center - left)
        down = (right - fft_mel) / (right - center)
        banks[b, : nfft // 2] = np.maximum(0.0, np.minimum(up, down))
    fb = np.log(np.maximum(power @ banks.T, np.finfo(np.float32).eps))
    return (fb - fb.mean(axis=0, keepdims=True)).astype(np.float32)


class Routed:
    """One graph as the sampler sees it (``get_inputs`` / ``run``): a session pinned to the per-tick
    shapes (``dims``), and an unpinned one, made on first use (on ``other_device`` when given), for
    calls of any other shape. Without dims, or when the provider refuses the pinned graph (DirectML
    does for some), only the unpinned session exists.

    Thread safe: reference voices are prepared on the server's reference thread while frames
    convert on the inference thread. ONNX Runtime runs one session from several threads, except
    DirectML (one ``Run`` per session at a time), so DirectML sessions take a lock per session."""

    def __init__(self, make, path, dims=None, other_device=None, serialize=False):
        self.make, self.path, self.other_device = make, path, other_device
        self.pinned = None
        self.other = None
        self._make_lock = threading.Lock()
        self._serialize = serialize
        self._run_locks = {}
        if dims:
            try:
                self.pinned = make(path, dims)
            except Exception as e:  # noqa: BLE001
                LOG.warning("%s: pinned shapes refused (%s); unpinned", os.path.basename(path), str(e).splitlines()[0][-120:])
        if self.pinned is None:
            self.other = make(path)
            self.other_device = None
        self.fixed = self._fixed_shapes(self.pinned, dims) if self.pinned is not None else {}

    @staticmethod
    def _fixed_shapes(sess, dims):
        """Input name -> its dimensions under the pins (``None`` where free). The declared shape
        may still name a pinned dimension symbolically, so the pins are applied here."""
        def fixed(d):
            if isinstance(d, int):
                return d
            return int(dims[d]) if isinstance(d, str) and d in dims else None

        return {i.name: [fixed(d) for d in i.shape] for i in sess.get_inputs()}

    def ensure_other(self) -> None:
        """Make the unpinned session now (before a stream needs it)."""
        if self.other is None:
            with self._make_lock:
                if self.other is None:
                    self.other = self.make(self.path, None, self.other_device)

    def provider(self) -> str:
        return (self.pinned or self.other).get_providers()[0]

    def get_inputs(self):
        return (self.pinned or self.other).get_inputs()

    def fits(self, feeds) -> bool:
        for name, dims in self.fixed.items():
            x = feeds.get(name)
            if x is None:
                continue
            shape = np.shape(x)
            if len(shape) != len(dims) or any(d is not None and d != s for d, s in zip(dims, shape)):
                return False
        return True

    def run(self, outputs, feeds):
        if self.pinned is not None and self.fits(feeds):
            sess = self.pinned
        else:
            self.ensure_other()
            sess = self.other
        if not self._serialize:
            return sess.run(outputs, feeds)
        lock = self._run_locks.setdefault(id(sess), threading.Lock())
        with lock:
            return sess.run(outputs, feeds)


class SeedVcStream(v2.Stream):
    def __init__(self, conv: "SeedVcConverter", seed: int):
        self.c = conv
        self.preset = conv.preset
        sh = conv.shapes
        self.window = np.zeros(sh["n16"], np.float32)
        self.block16 = conv.block_units * UNIT_16K
        self.block22 = conv.block_units * UNIT_22K
        self.sola_buffer = conv.sola_buffer_units * UNIT_22K
        self.sola_search = conv.sola_search_units * UNIT_22K
        self.fade_in, self.fade_out = v2.cosine_fade(self.sola_buffer)
        self.tail = np.zeros(self.sola_buffer, np.float32)
        self.ref = None
        self.rng = np.random.default_rng(seed)

    def set_reference(self, prepared):
        self.ref = prepared

    def set_params(self, params):
        steps = params.get("steps")
        cfg = params.get("cfg_rate")
        changes = {}
        if isinstance(steps, (int, float)) and 1 <= int(steps) <= 50:
            changes["steps"] = int(steps)
        if isinstance(cfg, (int, float)) and 0.0 <= float(cfg) <= 1.5:
            changes["cfg_rate"] = float(cfg)
        if changes:
            self.preset = dataclasses.replace(self.preset, **changes)

    def process(self, pcm, pitch, discontinuity):
        b = self.block16
        if discontinuity:
            # The rolling window and the SOLA tail are from before the gap: the first reply after it
            # would otherwise play their last `delay_samples` (about 70 ms) out of time.
            self.window[:] = 0.0
            self.tail[:] = 0.0
        self.window[:-b] = self.window[b:]
        self.window[-b:] = pcm
        if self.ref is None or (self.c.gate_db is not None and v2.rms_db(pcm) < self.c.gate_db):
            # Silence (or no voice yet): no tick; the next voiced block fades in from silence.
            self.tail[:] = 0.0
            return np.zeros(self.block22, np.float32)
        sh = self.c.shapes
        p = self.ref["prompt_mel"].shape[-1]
        z = self.rng.standard_normal((1, 80, p + sh["n_mel"])).astype(np.float32)
        harm = self.rng.uniform(-np.pi, np.pi, (1, 9, 1)).astype(np.float32)
        harm[:, 0] = 0.0
        noise = self.rng.standard_normal((1, 9, sh["n_mel"] * 256)).astype(np.float32)
        wav, _ = self.c.zs.tick(self.ref, self.window, self.preset, z, harm, noise, use_step_graph=True, hift_mode="full")
        wav = np.asarray(wav, np.float32).reshape(-1)
        sb, ss, blk = self.sola_buffer, self.sola_search, self.block22
        off = 0
        if sb > 0 and np.any(self.tail):
            head = wav[: sb + ss]
            nom = np.correlate(head, self.tail, "valid")
            den = np.sqrt(np.convolve(head * head, np.ones(sb, np.float32), "valid") + 1e-8)
            off = int(np.argmax(nom / den))
        out = wav[off: off + blk + sb].copy()
        if out.size < blk + sb:
            out = v2.fit_length(out, blk + sb)
        out[:sb] = out[:sb] * self.fade_in + self.tail * self.fade_out
        self.tail = out[blk: blk + sb].copy()
        return out[:blk]


class SeedVcConverter(v2.Converter):
    name = "seed-vc"
    input_rates = (SR_IN,)
    output_rate = SR_OUT
    reference = "required"
    reference_rate = SR_OUT
    params = ("steps", "cfg_rate")
    pitch = False

    def __init__(self, args):
        self.args = args
        self.sp = load_sampler(args.sampler)
        preset = self.sp.PRESETS.get(args.preset)
        if preset is None:
            raise SystemExit(f"seedvc: unknown preset {args.preset!r} (one of {', '.join(self.sp.PRESETS)})")
        changes = {k: v for k, v in (("steps", args.steps), ("cfg_rate", args.cfg_rate)) if v is not None}
        if args.frame_ms:
            changes["block"] = round(args.frame_ms / 20.0) * 0.02
        self.preset = dataclasses.replace(preset, **changes) if changes else preset
        u = self.preset.units()
        self.shapes = self.preset.shapes()
        self.block_units = int(round(self.preset.block * 50))
        self.sola_buffer_units = u["return_length"] - self.block_units - 1
        self.sola_search_units = 1
        self.frame_ms = self.block_units * 20.0
        self.max_reference_seconds = float(self.preset.max_prompt)
        right = u["skip_tail"]
        self.delay_samples = (self.sola_buffer_units * UNIT_22K + UNIT_22K // 2 + right * UNIT_22K)
        self.gate_db = None if args.silence_db is None or args.silence_db <= -200 else float(args.silence_db)
        self.device = args.device if args.device in PROVIDERS else "cpu"
        self.graphs = args.model_dir or default_graphs_dir()
        self.seed = 0

    def load(self):
        if not self.graphs or not os.path.isdir(self.graphs):
            raise SystemExit("seedvc: pass --model-dir, the folder of Sigil's Seed-VC export (export.py)")
        ort = import_ort(self.device, self.args.dll_dir)
        want = PROVIDERS[self.device]
        if want not in ort.get_available_providers():
            LOG.warning("onnxruntime %s has no %s; running on the CPU", ort.__version__, want)
            self.device, want = "cpu", PROVIDERS["cpu"]
        threads = max(1, int(self.args.threads))

        def make(path, dims=None, device=None):
            device = device or self.device
            so = ort.SessionOptions()
            for k, v in (dims or {}).items():
                so.add_free_dimension_override_by_name(k, int(v))
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.intra_op_num_threads = threads
            so.inter_op_num_threads = 1
            so.add_session_config_entry("session.intra_op.allow_spinning", "0")
            so.add_session_config_entry("session.inter_op.allow_spinning", "0")
            so.log_severity_level = 3
            if device == "dml":
                so.enable_mem_pattern = False
            options = {"cuda": {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}, "dml": {"device_id": 0}}.get(device)
            provider = PROVIDERS[device]
            providers = ([(provider, options)] if options else [provider]) + (["CPUExecutionProvider"] if device != "cpu" else [])
            return ort.InferenceSession(path, sess_options=so, providers=providers)

        # Every tick runs one shape per graph (the preset's window, and the prompt of a reference
        # at least `max_prompt` long, which voice-library clips are): those sessions are pinned to
        # it, which DirectML needs to run fast (measured on an RTX 4090, sigil_low: XLS-R 63 -> 4 ms,
        # a DiT step 14 -> 3 ms) and which folds the shape arithmetic on CUDA. Other shapes (the
        # reference voice, a shorter reference) use an unpinned session made on first use; the
        # reference's XLS-R pass runs on the CPU rather than a second copy of XLS-R in GPU memory.
        sh = self.shapes
        pins = {} if self.args.no_pin or self.device == "cpu" else {
            "xlsr": {"N": sh["n16"], "T": sh["xlsr_frames"]},
            "lr": {"T": sh["lr_in"], "n": sh["n_mel"]},
            "dit_step": {"L": sh["L"]},
            "dit_step1": {"L": sh["L"]},
            "hift": {"M": sh["n_mel"], "S": sh["n_mel"] * 256},
        }
        sessions = {}
        for name in GRAPH_NAMES:
            path = os.path.join(self.graphs, f"{name}.onnx")
            if not os.path.isfile(path):
                raise SystemExit(f"seedvc: {path} is missing (run scripts/vc-families/seed_vc/export.py)")
            routed = Routed(make, path, pins.get(name), "cpu" if name == "xlsr" else None, serialize=self.device == "dml")
            sessions[name] = routed
            used = routed.provider()
            if used != want:
                LOG.warning("%s runs on %s instead of %s", name, used, want)
                if name in ("dit_step", "dit_step1"):
                    # The estimator is most of the work: report the device it really runs on
                    # (Sigil shows it as the voice's execution provider).
                    self.device = next((k for k, v in PROVIDERS.items() if v == used), "cpu")
        self.zs = self.sp.SigilZS(sessions)

    def prepare_reference(self, pcm, rate, meta):
        ref22 = v2.resample(pcm, rate, SR_OUT)[: int(SR_OUT * self.preset.max_prompt)]
        if ref22.size < SR_OUT // 2:
            raise ValueError("the reference voice is shorter than 0.5 s")
        ref16 = v2.resample(ref22, SR_OUT, SR_IN)
        prepared = self.zs.prepare_reference(ref22, ref16, kaldi_fbank(ref16))
        if prepared["prompt_mel"].shape[-1] != self.shapes["prompt_frames"]:
            # A reference shorter than the preset's prompt: its ticks have another shape than the
            # pinned estimator's. Make the unpinned estimator now, not on the first spoken frame.
            LOG.info("reference shorter than %.1f s: the estimator runs unpinned (slower on DirectML)", self.preset.max_prompt)
            for name in ("dit_step", "dit_step1"):
                self.zs.s[name].ensure_other()
        return prepared

    def warmup(self):
        ref = self.prepare_reference(v2.synthetic_voice(3.0, SR_OUT, 210.0), SR_OUT, {})
        stream = self.open_stream(SR_IN, self.block_units * UNIT_16K)
        stream.set_reference(ref)
        voice = v2.synthetic_voice(self.preset.block + 0.1, SR_IN)[: stream.block16]
        self.time_warmup(lambda: stream.process(voice, 0.0, False))
        # ONNX Runtime's GPU memory, measured once the sessions have run (torch, if a front end
        # imported it, holds none of it).
        self._vram = v2.nvidia_process_vram_mb() if self.device == "cuda" else None
        LOG.info("seed-vc preset %s: steps %d, cfg %.2f, block %.0f ms; %.1f ms per tick on %s (%s MB of GPU memory)",
                 self.preset.name, self.preset.steps, self.preset.cfg_rate, self.frame_ms, self.latency_ms or 0.0,
                 self.device, "?" if self._vram is None else f"{self._vram:.0f}")

    def vram_mb(self):
        return getattr(self, "_vram", None)

    def frame_samples(self, rate_in, proposed):
        return self.block_units * UNIT_16K

    def open_stream(self, rate_in, frame_in):
        self.seed += 1
        return SeedVcStream(self, self.seed)


def add_arguments(p: argparse.ArgumentParser) -> None:
    p.add_argument("--preset", default="upstream_readme", help="upstream_readme (quality), sigil_low or sigil_lean")
    p.add_argument("--steps", type=int, default=None, help="diffusion steps (overrides the preset)")
    p.add_argument("--cfg-rate", type=float, default=None, help="classifier-free guidance (0 = off, 1.5x faster)")
    p.add_argument("--silence-db", type=float, default=-55.0, help="skip ticks whose new audio is below this (dBFS)")
    p.add_argument("--threads", type=int, default=2, help="CPU threads per graph")
    p.add_argument("--sampler", default="", help="path of scripts/vc-families/seed_vc/sampler.py")
    p.add_argument("--dll-dir", action="append", default=[], help="extra DLL folder (CUDA runtime); repeatable")
    p.add_argument("--no-pin", action="store_true", help="do not pin the graphs to the preset's shapes (slower, mostly on DirectML)")


if __name__ == "__main__":
    v2.main(SeedVcConverter, "Seed-VC real-time zero-shot sidecar for Sigil", add_arguments, echo_reference="required")
