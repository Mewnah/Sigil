#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity gate for Sigil's RVC split-graph templates (fp32 <= 1e-4, fp16 <= 5e-3, partial decode ==
the full-decode tail).

The reference is the whole-utterance synthesizer in PyTorch fp32 with weight_norm hooks active, its
random draws replaced by fixed noise:
- HiFi-GAN templates (f0 and f0-less): *upstream* RVC (vendored in ``rvc_arch``),
  ``SynthesizerTrnMs*NSFsid[_nono].infer``;
- RefineGAN / MRF HiFi-GAN templates: RVC's text encoder and flow (Applio keeps them) with the
  ``reference`` decode of Sigil's port of Applio's generator (``rvc_arch/vocoders.py``, Applio's
  arithmetic line by line; ``--upstream-applio`` checks the port against Applio's own code).
The candidate is the template graphs in ONNX Runtime with the weights written from the checkpoint by
the layout (Python reference writer, or the Rust importer's output with ``--rust-weights``) and
injected in memory, exactly as the Rust engine does.

Checks (per variant):
    enc        m_p / logs_p vs the reference text encoder
    flowdec    whole-window decode vs the reference decode (its first flow+dec margin frames excluded)
    chain      ORT enc -> ORT flowdec vs the reference
    chain-mix  (fp16) enc fp32 -> flowdec fp16, the recommended GPU mix
    partial    flowdec over the last R + margins frames with phase0 (and the noise of those frames)
               vs the reference tail and vs the graph's own full decode
    knn        vs a float64 brute-force RVC retrieval blend
    fallback   graph + weights file on disk (ONNX external-data convention) == in-memory result

Weights: a real voice where one is at hand (``--pth``; the MIT default voice for v2-40k-f0-hifigan,
Applio's RefineGAN 32 kHz pretrain for the RefineGAN templates), otherwise a *donor* checkpoint:
every tensor of the template's architecture taken from the nearest real voice where the shape
matches (MRF blocks from HiFi-GAN resblocks, which have the same structure), the rest LeCun-normal
random. Trained magnitudes keep the fp16 rows meaningful; random-init networks would decay to
near-silence and pass anything. Each row says where its weights came from.

Usage (repo root):

    python scripts/vc-templates/parity_test.py
        MIT default voice (PhoenixStormJr/RVC-V2-default-voice default.pth, from
        %LOCALAPPDATA%/sigil-dev-cache/rvc-default-voice or --pth) against the bundled
        v2-40k-f0-hifigan template; writes the reference bundle the Rust test
        `vc::import::tests::default_voice_parity` reads (--bundle-out).
    python scripts/vc-templates/parity_test.py --all [--report <json>]
        Every template of the matrix (configs.MATRIX) under --templates, with the weights above.
    python scripts/vc-templates/parity_test.py --template v2-32k-f0-refinegan [--pth <voice>]
    python scripts/vc-templates/parity_test.py --write-fixture src-tauri/test-fixtures/vc/tiny
        Regenerates the tiny fixture (random tiny checkpoint, its templates and reference outputs)
        used by `cargo test vc::import`.
    python scripts/vc-templates/parity_test.py --write-fixture-v2 src-tauri/test-fixtures/vc/tiny/v2
        Writes tiny fixtures of the layout-version-2 kinds (f0-less, MRF, RefineGAN; see
        write_tiny_fixture_v2) for the importer / engine tests.
    --rust-weights <slot dir>   also require the Rust importer's weights.f32.bin to be
                                byte-identical to the Python writer's.
    --upstream-applio           (maintainer check, needs the vendored applio_arch package) compare
                                the RefineGAN / MRF reference decodes with Applio's Synthesizer.infer,
                                both in float64, every checkpoint key accounted for.
"""

import argparse
import collections
import contextlib
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import time
import warnings

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import export_rvc  # noqa: E402
import sigil_layout as sl  # noqa: E402
from rvc_arch import configs, split, vocoders  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")

TOL = {"fp32": 1e-4, "fp16": 5e-3}
DEV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
DEFAULT_PTH = os.path.join(DEV_CACHE, "rvc-default-voice", "default.pth")
# Applio's RefineGAN 32 kHz pretrained generator (IAHispano/Applio Resources/refinegan/f0G32k.pth,
# 140,604,497 bytes, sha256 daad0a22...0203484): a trained RefineGAN (its decoder's tensors do not
# depend on the sample rate, so it serves every RefineGAN template).
REFINEGAN_PTH = os.path.join(DEV_CACHE, "applio-hf", "refinegan", "f0G32k.pth")
DEFAULT_TEMPLATE = "v2-40k-f0-hifigan"
TINY_TEMPLATE = "v2-16k-f0-hifigan"  # test-only tiny config (rvc_arch/configs.py)
DEFAULT_TEMPLATES_DIR = os.path.join(REPO, "src-tauri", "resources", "vc-templates")
# Execution provider of the template sessions (--ep): the CPU by default; "dml" needs an
# onnxruntime-directml build in the Python environment (the GPU device class of layout.json parity).
EP_PROVIDERS = ["CPUExecutionProvider"]
EPS = {
    "cpu": ["CPUExecutionProvider"],
    "dml": [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
    "cuda": [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0}), "CPUExecutionProvider"],
}


# ---------------------------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------------------------


def _bare_state_dict(ckpt):
    """A bare state dict (every value a tensor, holding ``enc_p.emb_phone.weight``), optionally saved
    from a DataParallel model (a ``module.`` prefix on every key): the keys without the prefix, else
    None."""
    if not isinstance(ckpt, dict) or not ckpt or not all(torch.is_tensor(v) for v in ckpt.values()):
        return None
    keys = list(ckpt.keys())
    if all(isinstance(k, str) and k.startswith("module.") for k in keys):
        ckpt = {k[len("module."):]: v for k, v in ckpt.items()}
    return ckpt if "enc_p.emb_phone.weight" in ckpt else None


def load_checkpoint(path):
    """An RVC inference checkpoint ({weight, config, ...}), a training checkpoint ({model, ...},
    Applio's pretrains) or a bare state dict (optionally DataParallel's ``module.`` keys). Tensors
    only (weights_only), never code."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "weight" in ckpt:
        return ckpt
    # A training checkpoint or a bare state dict holds fp32 weights; the voices users import hold
    # them in fp16 (RVC's and Applio's model extraction save `.half()`), so round them the same way.
    def rounded(state):
        return {k: v.half() if v.is_floating_point() else v for k, v in state.items()}

    if "model" in ckpt and isinstance(ckpt["model"], dict):
        return {"weight": rounded(ckpt["model"]), "config": None, "info": f"training checkpoint {os.path.basename(path)}, rounded to fp16"}
    bare = _bare_state_dict(ckpt)
    if bare is not None:
        return {"weight": rounded(bare), "config": None, "info": f"bare state dict {os.path.basename(path)}, rounded to fp16"}
    raise SystemExit(f"{path}: not an RVC checkpoint (no 'weight', 'model' or bare state dict with enc_p.emb_phone.weight)")


def model_state(ckpt):
    """The inference tensors of a checkpoint, float32, weight_norm keys normalised, enc_q dropped."""
    state = {k: v.float() for k, v in ckpt["weight"].items() if not k.startswith("enc_q.")}
    return {sl.normalize_key(k): v for k, v in state.items()}


def _lecun(shape, gen, transpose_stride=None):
    shape = list(shape)
    if len(shape) == 3:
        fan = shape[1] * shape[2] if transpose_stride is None else shape[0] * shape[2] / transpose_stride
    elif len(shape) == 2:
        fan = shape[1]
    else:
        fan = max(shape[0], 1)
    return torch.randn(shape, generator=gen) / math.sqrt(max(fan, 1))


def _mrf_donor_key(key):
    """MRF blocks have ResBlock1's structure: dec.mrfs.i.j.layers.l.conv{1,2} <- dec.resblocks.(3i+j).convs{1,2}.l."""
    parts = key.split(".")
    if len(parts) > 6 and parts[0] == "dec" and parts[1] == "mrfs" and parts[4] == "layers":
        i, j, l, conv = int(parts[2]), int(parts[3]), parts[5], parts[6]
        rest = ".".join(parts[7:])
        return f"dec.resblocks.{3 * i + j}.convs{conv[-1]}.{l}.{rest}"
    if key.startswith("dec.upsamples."):
        return key.replace("dec.upsamples.", "dec.ups.", 1)
    return None


def donor_checkpoint(spec, donor, seed=97):
    """A checkpoint of the template's architecture (RVC key names, weight_norm as g / v): every
    tensor from ``donor`` (a real voice's normalised state) where the key (or MRF's HiFi-GAN
    equivalent) exists with the same shape, the rest LeCun-normal random. Returns (state, provenance)."""
    net = export_rvc.build_net(spec)  # weight_norm hooks active: the checkpoint's key names
    gen = torch.Generator().manual_seed(seed)
    cfg = configs.config_dict(spec["config"])
    rates = cfg["upsample_rates"]
    state, from_donor, random = {}, 0, 0
    for key, value in net.state_dict().items():
        src = None
        if donor is not None:
            for cand in (key, _mrf_donor_key(key)):
                if cand is not None and cand in donor and tuple(donor[cand].shape) == tuple(value.shape):
                    src = donor[cand]
                    break
        if src is not None:
            state[key] = src.clone().float()
            from_donor += 1
            continue
        random += 1
        if key.endswith("weight_g"):
            continue  # set with its weight_v below
        stride = None
        if ".ups." in key or ".upsamples." in key:
            stride = rates[int(key.split(".")[2])]
        if key.endswith("weight_v"):
            v = _lecun(value.shape, gen, stride)
            state[key] = v
            gkey = key[:-1] + "g"
            g = torch.linalg.vector_norm(v.reshape(v.shape[0], -1), dim=1).reshape(net.state_dict()[gkey].shape)
            state[gkey] = g * (0.8 + 0.4 * torch.rand(g.shape, generator=gen))
        elif key.endswith(".bias") or key.endswith(".beta"):
            state[key] = torch.randn(value.shape, generator=gen) * 0.02
        elif key.endswith(".gamma"):
            state[key] = 1.0 + torch.randn(value.shape, generator=gen) * 0.05
        elif key.endswith("emb_rel_k") or key.endswith("emb_rel_v"):
            state[key] = torch.randn(value.shape, generator=gen) * (value.shape[-1] ** -0.5)
        elif key == "emb_g.weight" or key.endswith("emb_pitch.weight"):
            state[key] = torch.randn(value.shape, generator=gen) * 0.5
        elif value.dim() == 1 and "blocks" in key:  # RefineGAN AdaIN scales (trained ~0.01-0.06)
            state[key] = torch.randn(value.shape, generator=gen).abs() * 0.03
        else:
            state[key] = _lecun(value.shape, gen, stride)
    # fp16 storage, as trainers save inference checkpoints
    state = {k: v.half().float() for k, v in state.items()}
    return state, {"from_donor": from_donor, "random": random}


def reference_net(state, spec):
    """The whole-utterance reference model with ``state`` loaded (weight_norm hooks active). The
    speaker count comes from the checkpoint (templates take any: ``g`` is an input)."""
    spec = dict(spec, config=list(spec["config"]))
    spec["config"][15] = int(state["emb_g.weight"].shape[0])
    if spec["vocoder"] == "hifigan":
        net = export_rvc.synthesizer_class(spec["version"], spec["f0"])(*spec["config"], is_half=False)
        if hasattr(net, "enc_q"):
            del net.enc_q
    else:
        net = export_rvc.build_net(spec)
    want = {k: v for k, v in state.items()}
    missing, unexpected = net.load_state_dict(want, strict=False)
    bad = [k for k in missing if not k.startswith("enc_q.")]
    if bad or unexpected:
        raise SystemExit(f"checkpoint does not fit the model: missing {bad[:5]}, unexpected {unexpected[:5]}")
    return net.float().eval()


class FixedNoise:
    """Replace ``torch.randn_like`` (RVC's random draws in ``infer``) by fixed tensors, by shape."""

    def __init__(self, by_shape):
        self.by_shape = {tuple(k): v for k, v in by_shape.items()}

    def __enter__(self):
        self._orig = torch.randn_like

        def fake(x, *args, **kwargs):
            try:
                return self.by_shape[tuple(x.shape)].clone().to(dtype=x.dtype)
            except KeyError:
                raise RuntimeError(f"unexpected torch.randn_like{tuple(x.shape)}") from None

        torch.randn_like = fake
        return self

    def __exit__(self, *exc):
        torch.randn_like = self._orig


def coarse_pitch(f0):
    """RVC pipeline's f0 -> 1..255 mel bins."""
    f0_min, f0_max = 50.0, 1100.0
    mel_min = 1127 * np.log(1 + f0_min / 700)
    mel_max = 1127 * np.log(1 + f0_max / 700)
    mel = 1127 * np.log(1 + f0 / 700)
    voiced = mel > 0
    mel[voiced] = (mel[voiced] - mel_min) * 254 / (mel_max - mel_min) + 1
    mel[mel <= 1] = 1
    mel[mel > 255] = 255
    return np.rint(mel).astype(np.int64)


def per_frame_noise(spec, net):
    if spec["vocoder"] != "refinegan":
        return 0
    return vocoders.adain_layers(net.dec)[1]


def make_inputs(spec, frames, seed, net=None):
    cfg = configs.config_dict(spec["config"])
    upp = configs.upp(spec["config"])
    rng = np.random.default_rng(seed)
    t = np.arange(frames)
    f0 = 155 + 45 * np.sin(t / 11.0) + 12 * np.sin(t / 3.1)
    f0[(t > frames * 0.30) & (t < frames * 0.40)] = 0.0  # unvoiced stretches (noise excitation)
    f0[(t > frames * 0.75) & (t < frames * 0.79)] = 0.0
    # content features: a smooth random walk (HuBERT-like temporal correlation), unit-ish scale
    walk = np.cumsum(rng.standard_normal((frames, spec["emb_channels"])) * 0.25, axis=0)
    phone = (walk - walk.mean(0)) / (walk.std(0) + 1e-6) * 0.6
    inp = {
        "phone": phone[None].astype(np.float32),
        "pitch": coarse_pitch(f0.copy())[None],
        "f0": f0[None].astype(np.float32),
        "z_noise_unit": rng.standard_normal((1, cfg["inter_channels"], frames)).astype(np.float32),
        "sid": np.array([0], dtype=np.int64),
    }
    if spec["vocoder"] == "mrf-hifigan":
        inp["src_noise"] = rng.standard_normal((1, frames * upp, vocoders.MRF_HARMONICS + 1)).astype(np.float32)
    else:
        inp["src_noise"] = rng.standard_normal((1, frames * upp)).astype(np.float32)
    if spec["vocoder"] == "refinegan":
        per_frame = per_frame_noise(spec, net)
        pool = spec.get("adain_pool", export_rvc.ADAIN_POOL)
        inp["pool"] = rng.standard_normal(pool).astype(np.float32)
        inp["noise_offsets"] = rng.integers(0, pool - per_frame + 1, (1, frames)).astype(np.int64)
    return inp


def reference_outputs(net, spec, inp):
    frames = inp["phone"].shape[1]
    upp = configs.upp(spec["config"])
    t = {k: torch.from_numpy(v) for k, v in inp.items()}
    lengths = torch.tensor([frames])
    with torch.no_grad():
        g = net.emb_g(t["sid"]).unsqueeze(-1)
        pitch = t["pitch"] if spec["f0"] else None
        m_p, logs_p, _ = net.enc_p(t["phone"], pitch, lengths)
        if spec["vocoder"] == "hifigan":
            noise = {t["z_noise_unit"].shape: t["z_noise_unit"]}
            if spec["f0"]:
                noise[(1, frames * upp, 1)] = t["src_noise"].reshape(1, -1, 1)
            with FixedNoise(noise):
                if spec["f0"]:
                    audio, _, _ = net.infer(t["phone"], lengths, t["pitch"], t["f0"], t["sid"])
                else:
                    audio, _, _ = net.infer(t["phone"], lengths, t["sid"])
        else:
            # Sigil's ports of Applio's generators: evaluated in float64 (the exact model), because
            # Applio's per-sample fp32 phase sum drifts by ~1e-5 cycles over a few seconds, which
            # a trained generator amplifies to the size of the fp32 gate.
            g, m_p, logs_p, audio = applio_port_decode(net, spec, inp, torch.float64)  # net: in place
    # m_p / logs_p / g also feed the graphs (fp32); the audio stays in the reference's precision.
    return {
        "m_p": m_p.float().numpy(),
        "logs_p": logs_p.float().numpy(),
        "audio": audio.reshape(1, -1).numpy(),
        "g": g.float().numpy(),
    }


def applio_port_decode(net, spec, inp, dtype=torch.float64):
    """The RefineGAN / MRF reference in ``dtype``: RVC's text encoder and flow, then the ``reference``
    decode of Sigil's port of Applio's generator, with ``inp``'s fixed noise. ``net`` is converted to
    ``dtype`` in place. Returns (g, m_p, logs_p, audio) in ``dtype``."""
    frames = inp["phone"].shape[1]
    lengths = torch.tensor([frames])
    net = net.to(dtype)
    d = {k: (torch.from_numpy(v).to(dtype) if np.issubdtype(v.dtype, np.floating) else torch.from_numpy(v)) for k, v in inp.items()}
    with torch.no_grad():
        g = net.emb_g(d["sid"]).unsqueeze(-1)
        m_p, logs_p, _ = net.enc_p(d["phone"], d["pitch"], lengths)
        mask = torch.ones(1, 1, frames, dtype=dtype)
        z_p = (m_p + torch.exp(logs_p) * d["z_noise_unit"] * 0.66666) * mask
        z = net.flow(z_p, mask, g=g, reverse=True)
        if spec["vocoder"] == "mrf-hifigan":
            ini = torch.tensor(vocoders.MRF_PHASE_INI, dtype=dtype)
            audio = net.dec.reference(z * mask, d["f0"], g, d["src_noise"], ini)
        else:
            pool, offs = d["pool"], d["noise_offsets"][0]
            layers = {(s, l): base for s, l, _c, _r, base in vocoders.adain_layers(net.dec)[0]}
            audio = net.dec.reference(
                z * mask, d["f0"], g, d["src_noise"],
                lambda i, layer, y: vocoders.adain_noise_from_pool(pool, offs, layers[(i, layer)], y),
            )
    return g, m_p, logs_p, audio


def nsf_phase(f0_frames, sr, upp):
    """NSF sine phase (cycles, [0, 1)) at the first sample of each frame, float64 (what the Rust
    engine carries across ticks as ``phase0``, for every vocoder)."""
    step = np.fmod(upp * np.asarray(f0_frames, dtype=np.float64) / sr + 0.5, 1.0) - 0.5
    return np.mod(np.concatenate([[0.0], np.cumsum(step)]), 1.0)


# ---------------------------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------------------------


def ort_session(template_dir, layout, graph, variant, weights_f32, weights_f16, pool=None, providers=None):
    """Session for one template graph with every weight (and RefineGAN's noise pool) injected from
    memory by name."""
    import onnx
    import onnxruntime as ort

    info = layout["graphs"][graph]["variants"][variant]
    path = os.path.join(template_dir, info["file"])
    if sl.sha256_file(path) != info["sha256"]:
        raise SystemExit(f"{info['file']} does not match layout.json (re-export the template)")
    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = providers or EP_PROVIDERS
    if any((p[0] if isinstance(p, tuple) else p) == "DmlExecutionProvider" for p in providers):
        so.enable_mem_pattern = False  # DirectML requirements
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    keep = []
    if info["weights_file"]:
        model = onnx.load(path, load_external_data=False)
        meta = {p.key: p.value for p in model.metadata_props}
        if meta.get("sigil.weights_id") != layout["weights"]["weights_id"]:
            raise SystemExit(f"{info['file']}: weights id does not match layout.json")
        buf = weights_f32 if variant == "fp32" else weights_f16
        dtype, scale = (np.float32, 1) if variant == "fp32" else (np.float16, 2)
        by_name = {t["name"]: t for t in layout["weights"]["tensors"]}
        names, values = [], []
        for init in model.graph.initializer:
            if init.name in by_name:
                t = by_name[init.name]
                arr = np.frombuffer(buf, dtype=dtype, count=t["bytes"] // 4, offset=t["offset"] // scale).reshape(t["shape"])
                keep.append(arr)
                names.append(init.name)
                values.append(ort.OrtValue.ortvalue_from_numpy(arr))
            elif init.name == "adain_noise_pool":
                arr = np.ascontiguousarray(pool, dtype=np.float32)
                keep.append(arr)
                names.append(init.name)
                values.append(ort.OrtValue.ortvalue_from_numpy(arr))
        so.add_external_initializers(names, values)
        keep.append(values)
    sess = ort.InferenceSession(path, so, providers=providers)
    if sess.get_providers()[0] != (providers[0][0] if isinstance(providers[0], tuple) else providers[0]):
        raise SystemExit(f"{info['file']}: {providers[0]} did not load (got {sess.get_providers()})")
    return sess, keep


def run(sess, feeds):
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, {n: feeds[n] for n in names})


def fallback_session(template_dir, layout, graph, variant, weights, pool=None):
    """ONNX external-data convention: the graph next to its weights file, loaded from disk."""
    import onnxruntime as ort

    info = layout["graphs"][graph]["variants"][variant]
    tmp = tempfile.mkdtemp(prefix="sigil-vc-")
    shutil.copy(os.path.join(template_dir, info["file"]), tmp)
    with open(os.path.join(tmp, info["weights_file"]), "wb") as f:
        f.write(weights)
    for extra in info.get("extra_external_files", []):
        with open(os.path.join(tmp, extra), "wb") as f:
            f.write(np.ascontiguousarray(pool, dtype="<f4").tobytes())
    so = ort.SessionOptions()
    so.log_severity_level = 3
    if any((p[0] if isinstance(p, tuple) else p) == "DmlExecutionProvider" for p in EP_PROVIDERS):
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(os.path.join(tmp, info["file"]), so, providers=EP_PROVIDERS), tmp


# ---------------------------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------------------------


class Report:
    """Max-abs gate per check. ``fp16_scaled`` (tiny random fixture only) multiplies the fp16
    tolerance by max(1, peak |reference|): fp16's absolute resolution grows with magnitude, and the
    random tiny model has larger activations than trained voices (whose gate stays absolute)."""

    def __init__(self, fp16_scaled=False, quiet=False):
        self.rows = []
        self.failed = False
        self.fp16_scaled = fp16_scaled
        self.quiet = quiet

    def add(self, check, variant, got, want, tol, note="", gate=True):
        """``gate=False``: recorded and printed, but it does not fail the run (a row that decides a
        variant's parity class without being a shipped path by itself)."""
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)
        if got.shape != want.shape:
            raise SystemExit(f"{check}/{variant}: shape {got.shape} != {want.shape}")
        if self.fp16_scaled and variant == "fp16":
            tol = tol * max(1.0, float(np.abs(want).max()))
        err = float(np.abs(got - want).max())
        rel = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12))
        ok = err <= tol
        if gate:
            self.failed |= not ok
        self.rows.append({"check": check, "variant": variant, "max_abs": err, "rel_l2": rel, "tol": tol, "ok": ok, "gate": gate, "note": note, "peak": float(np.abs(want).max())})
        if not self.quiet:
            verdict = ("PASS" if ok else "FAIL") if gate else ("pass" if ok else "miss")
            print(f"  {verdict}  {check:<10} {variant:<5} max|d| {err:.3e}  rel_l2 {rel:.3e}  (tol {tol:.3g}) {note}")


def flowdec_feeds(spec, ref, inp, z_noise, s=0, m_p=None, logs_p=None, e=None, latent_offset=0):
    """flowdec inputs for the window of utterance frames [s, e) (``e``: the utterance end by
    default). The latents (m_p, logs_p, z_noise) cover [s + latent_offset, e): RefineGAN's latent
    window (layout ``streaming.latent_offset``; 0 for the other vocoders). The phase comes from the
    whole-utterance f0 (``nsf_phase``, float64), exactly as the engine carries it."""
    upp = configs.upp(spec["config"])
    sr = configs.sample_rate(spec["config"])
    e = inp["f0"].shape[1] if e is None else e
    zs = s + latent_offset
    feeds = {
        "m_p": (ref["m_p"] if m_p is None else m_p)[:, :, zs:e],
        "logs_p": (ref["logs_p"] if logs_p is None else logs_p)[:, :, zs:e],
        "z_noise": z_noise[:, :, zs:e],
        "g": ref["g"],
    }
    if spec["f0"]:
        phase = nsf_phase(inp["f0"][0], sr, upp)
        feeds["f0"] = inp["f0"][:, s:e]
        if spec["vocoder"] == "hifigan":
            feeds["phase0"] = np.array([[phase[s]]], dtype=np.float32)
        else:
            feeds["phase"] = phase[s:e][None].astype(np.float32)
        feeds["src_noise"] = inp["src_noise"][:, s * upp : e * upp]
    if spec["vocoder"] == "refinegan":
        feeds["noise_offsets"] = inp["noise_offsets"][:, s:e]
    return feeds


EDGE_RETURN_FRAMES = 48  # frames an edge window returns (longer than any template's measured seam)


def edge_windows(f0, margin, returned=EDGE_RETURN_FRAMES):
    """Mid-utterance window ends for the ``edge`` rows: [(end, kind, returned frames)] with a voiced
    end (62% into the utterance) and the latest voicing offset (the first unvoiced frame after a
    voiced one: the window's last frame is voiced, the next one is not), each returning
    min(``returned``, end - ``margin``) >= 4 frames (``margin`` = flow_margin + dec_margin)."""
    n = len(f0)
    ends = [(int(0.62 * n), "voiced")]
    offsets = [j for j in range(1, n) if f0[j - 1] > 0 and f0[j] <= 0 and j - margin >= 4]
    if offsets:
        ends.append((offsets[-1], "voicing offset"))
    out = []
    for end, kind in ends:
        r = min(returned, end - margin)
        if r >= 4 and f0[end - 1] > 0:
            out.append((end, kind, r))
    return out


def edge_profile(got, want, upp, tol):
    """Per-frame max |got - want| over a window's returned frames (oldest first) and how many of
    its last frames reach back to the earliest frame over ``tol`` (0 when none is): the length of
    the no-lookahead seam against the offline decode, saturating at the frames returned."""
    d = np.abs(np.asarray(got, np.float64) - np.asarray(want, np.float64)).reshape(-1, upp).max(axis=1)
    bad = np.nonzero(d > tol)[0]
    return d, (len(d) - int(bad[0])) if bad.size else 0


def parity(
    template_dir, layout, state, spec, frames, block_frames, seed, rust_weights=None, bundle_out=None,
    knn_rows=2048, knn_queries=96, fp16_scaled=False, weights_note="", quiet=False, bundle_stem="reference",
    report_name="parity_report.json",
):
    sr, upp = configs.sample_rate(spec["config"]), configs.upp(spec["config"])
    st = layout["streaming"]
    fm, dm = st["flow_margin"], st["dec_margin"]
    lo = st.get("latent_offset", 0)
    if configs.arch_key(spec["config"]) != configs.arch_key(layout["arch"]["config"]):
        raise SystemExit("checkpoint architecture does not match the template")

    def say(msg):
        if not quiet:
            print(msg)

    t0 = time.time()
    weights = sl.write_weights(layout, state)
    weights16 = sl.f16_weights(weights)
    speakers = sl.speaker_table(layout, state)
    say(f"weights.f32.bin: {len(weights)} bytes ({time.time() - t0:.2f} s, Python reference writer)")
    if rust_weights:
        rw = open(os.path.join(rust_weights, sl.WEIGHTS_F32), "rb").read()
        same = rw == weights
        print(f"  {'PASS' if same else 'FAIL'}  rust weights.f32.bin byte-identical to the Python writer ({len(rw)} bytes)")
        rs = np.fromfile(os.path.join(rust_weights, sl.SPEAKERS_F32), dtype="<f4").reshape(speakers.shape)
        same_spk = np.array_equal(rs, speakers)
        print(f"  {'PASS' if same_spk else 'FAIL'}  rust speakers.f32.bin identical")
        if not (same and same_spk):
            raise SystemExit(1)

    net = reference_net({k: torch.from_numpy(np.asarray(v)) if isinstance(v, np.ndarray) else v for k, v in state.items()}, spec)
    inp = make_inputs(spec, frames, seed, net)
    t0 = time.time()
    ref = reference_outputs(net, spec, inp)
    say(f"reference: {time.time() - t0:.2f} s for {frames} frames")
    if not np.allclose(ref["g"][0, :, 0], speakers[0], atol=0):
        raise SystemExit("speaker table row 0 != emb_g(0)")

    z_noise = (inp["z_noise_unit"] * np.float32(st["noise_scale"])).astype(np.float32)
    phase = nsf_phase(inp["f0"][0], sr, upp)
    first_full = (fm + dm) * upp
    ref_tail = ref["audio"][:, first_full:]
    win = block_frames + fm + dm
    s = frames - win
    if s <= 0:
        raise SystemExit(f"--frames {frames} is too short for a {win}-frame partial window")
    rep = Report(fp16_scaled, quiet)
    outputs = {}
    edges = {}
    pool = inp.get("pool")

    for variant in ("fp32", "fp16"):
        tol = TOL[variant]
        enc, _k1 = ort_session(template_dir, layout, "enc", variant, weights, weights16)
        fd, _k2 = ort_session(template_dir, layout, "flowdec", variant, weights, weights16, pool)
        enc_feeds = {"phone": inp["phone"]}
        if spec["f0"]:
            enc_feeds["pitch"] = inp["pitch"]
        m, logs = run(enc, enc_feeds)
        rep.add("enc.m_p", variant, m, ref["m_p"], tol, weights_note)
        rep.add("enc.logs_p", variant, logs, ref["logs_p"], tol)
        feeds = flowdec_feeds(spec, ref, inp, z_noise, latent_offset=lo)
        (full,) = run(fd, feeds)
        rep.add("flowdec", variant, full, ref_tail, tol, f"{frames} frames")
        (chain,) = run(fd, flowdec_feeds(spec, ref, inp, z_noise, m_p=m, logs_p=logs, latent_offset=lo))
        # enc fp16 -> flowdec fp16: never a recommended pairing (enc runs fp32 on every class), so it
        # only decides whether enc fp16 may be offered at all (export_matrix.class_parity).
        rep.add("chain", variant, chain, ref_tail, tol, "" if variant == "fp32" else "enc fp16 -> flowdec fp16 (decides enc fp16)", gate=variant == "fp32")
        if variant == "fp16":
            # The recommended GPU mix (layout.json "recommended"): enc fp32, flowdec fp16.
            (mixed,) = run(fd, flowdec_feeds(
                spec, ref, inp, z_noise, m_p=outputs["fp32"]["m_p"], logs_p=outputs["fp32"]["logs_p"], latent_offset=lo,
            ))
            rep.add("chain-mix", variant, mixed, ref_tail, tol, "enc fp32 -> flowdec fp16")
        (part,) = run(fd, flowdec_feeds(spec, ref, inp, z_noise, s=s, latent_offset=lo))
        rep.add("partial", variant, part, ref["audio"][:, -block_frames * upp :], tol, f"R={block_frames} of F={win} frames, vs reference")
        rep.add("partial", variant, part, full[:, -block_frames * upp :], tol, "vs this graph's full decode")
        outputs[variant] = {"m_p": m, "logs_p": logs, "audio_full": full, "audio_partial": part}

        # T5 evidence (not gated): a window ending mid-utterance with no lookahead, against the offline
        # decode of the same frames. Its last frames lack the flow's and the decoder's right context
        # (layout streaming.flow_receptive_right / dec_receptive_right_frames, and RefineGAN's
        # one-frame f0 lookahead), so they differ from offline; the seam is how far back that reaches.
        for end, kind, r in edge_windows(inp["f0"][0], fm + dm):
            (edge,) = run(fd, flowdec_feeds(spec, ref, inp, z_noise, s=end - fm - dm - r, e=end, latent_offset=lo))
            want = ref["audio"][:, (end - r) * upp : end * upp]
            per_frame, seam = edge_profile(edge, want, upp, tol)
            rep.add("edge", variant, edge, want, tol,
                    f"window ends at frame {end} of {frames} ({kind}), no lookahead: seam {seam} of {r} frames over tol", gate=False)
            last, last_want = edge[0, -upp:].astype(np.float64), want[0, -upp:].astype(np.float64)
            edges.setdefault(variant, []).append({
                "window_end": end, "kind": kind, "returned_frames": r, "tol": tol, "seam_frames": seam,
                "saturated": seam >= r, "max_abs": float(per_frame.max()), "last_frame_max_abs": float(per_frame[-1]),
                "last_frame_rel_rms": float(np.sqrt(np.mean((last - last_want) ** 2)) / (np.sqrt(np.mean(last_want**2)) + 1e-12)),
                "per_frame_max_abs": [float(x) for x in per_frame],
            })

        if variant == "fp32":
            sess, tmp = fallback_session(template_dir, layout, "flowdec", "fp32", weights, pool)
            (fb,) = run(sess, feeds)
            rep.add("fallback", variant, fb, full, 0.0, "graph + weights file on disk == in-memory")
            del sess
            shutil.rmtree(tmp, ignore_errors=True)

    # knn: synthetic clustered index (retrieval has no weights)
    rng = np.random.default_rng(seed + 1)
    e = spec["emb_channels"]
    n_idx, n_q = knn_rows, knn_queries
    centers = rng.standard_normal((64, e)).astype(np.float32)
    index = (centers[rng.integers(0, 64, n_idx)] + 0.35 * rng.standard_normal((n_idx, e))).astype(np.float32)
    feats = (centers[rng.integers(0, 64, n_q)] + 0.35 * rng.standard_normal((n_q, e))).astype(np.float32)
    rate = np.array([0.75], np.float32)
    for variant in ("fp32", "fp16"):
        idx = index if variant == "fp32" else index.astype(np.float16)
        want, best = split.knn_reference(feats, idx.astype(np.float32), float(rate[0]))
        knn, _ = ort_session(template_dir, layout, "knn", variant, None, None)
        index_sq = (idx.astype(np.float64) ** 2).sum(1).astype(np.float32)
        (got,) = run(knn, {"feats": feats, "index": idx, "index_sq": index_sq, "index_rate": rate})
        rep.add("knn", variant, got, want, TOL[variant], f"N={n_idx}, T={n_q}")
        outputs[f"knn_{variant}"] = got

    report = {
        "template": layout["template"],
        "weights": weights_note,
        "frames": frames,
        "block_frames": block_frames,
        "window_frames": win,
        "flow_margin": fm,
        "dec_margin": dm,
        "latent_offset": lo,
        "rows": rep.rows,
        # mid-utterance windows with no lookahead vs the offline decode, per variant (the edge rows)
        "right_edge": edges,
        "ok": not rep.failed,
    }
    if bundle_out:
        os.makedirs(bundle_out, exist_ok=True)
        arrays = collections.OrderedDict(
            [
                ("phone", inp["phone"]),
                ("pitch", inp["pitch"]),
                ("f0", inp["f0"]),
                ("z_noise", z_noise),
                ("src_noise", inp["src_noise"]),
                ("sid", inp["sid"]),
                ("phase0_partial", np.array([[phase[s]]], dtype=np.float32)),
                ("ref_m_p", ref["m_p"]),
                ("ref_logs_p", ref["logs_p"]),
                ("ref_audio", ref["audio"].astype(np.float32)),
                ("ref_g", ref["g"]),
                ("knn_feats", feats),
                ("knn_index", index),
                ("knn_rate", rate),
                ("knn_ref_fp32", split.knn_reference(feats, index, float(rate[0]))[0].astype(np.float32)),
                ("knn_ref_fp16", split.knn_reference(feats, index.astype(np.float16).astype(np.float32), float(rate[0]))[0].astype(np.float32)),
            ]
        )
        if spec["f0"] and spec["vocoder"] != "hifigan":
            arrays["phase_frames"] = phase[:frames][None].astype(np.float32)  # the `phase` input over the utterance
        if spec["vocoder"] == "refinegan":
            arrays["noise_offsets"] = inp["noise_offsets"]
            arrays["adain_pool"] = inp["pool"]
        meta = {
            "template": layout["template"],
            "frames": frames,
            "block_frames": block_frames,
            "partial_start": s,
            "weights_id": layout["weights"]["weights_id"],
            "weights_f32_sha256": hashlib.sha256(weights).hexdigest(),
            "speakers_f32_sha256": hashlib.sha256(speakers.astype("<f4").tobytes()).hexdigest(),
            "speakers": int(speakers.shape[0]),
            "tolerance": TOL,
            "fp16_tolerance_scaled_by_peak": fp16_scaled,
            "source": ("upstream RVC SynthesizerTrnMs*NSFsid[_nono].infer, PyTorch fp32, fixed noise" if spec["vocoder"] == "hifigan"
                       else "RVC enc_p / flow + Sigil's port of Applio's generator (rvc_arch/vocoders.py), float64, fixed noise"),
            "layout_version": layout["format_version"],
        }
        sl.write_bundle(os.path.join(bundle_out, bundle_stem), arrays, meta)
        sl.write_json(os.path.join(bundle_out, report_name), report)
        say(f"reference bundle -> {bundle_out}")
    return report


# ---------------------------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------------------------


def _state_of(path):
    return model_state(load_checkpoint(path)) if path and os.path.isfile(path) else None


def matrix_weights(template_id, spec, donors):
    """(state, note) for a template: a real voice of exactly that architecture, else a donor build."""
    real = donors.get(template_id)
    if real is not None:
        name, st = real
        return st, f"real voice ({name})"
    if spec["vocoder"] == "refinegan":
        name, donor = donors.get("refinegan", (None, None))
    else:
        name, donor = donors.get("hifigan", (None, None))
    state, prov = donor_checkpoint(spec, donor)
    note = f"donor build: {prov['from_donor']} tensors from {name or 'nothing'}, {prov['random']} random"
    return {k: v.numpy() for k, v in state.items()}, note


def parity_matrix(templates_dir, only=None, seed=5, quiet=True):
    donors = {}
    default_state = _state_of(DEFAULT_PTH)
    if default_state is not None:
        donors["hifigan"] = ("MIT default voice", default_state)
        donors[DEFAULT_TEMPLATE] = ("MIT default voice", {k: v.numpy() for k, v in default_state.items()})
    refine_state = _state_of(REFINEGAN_PTH)
    if refine_state is not None:
        donors["refinegan"] = ("Applio RefineGAN 32k pretrain", refine_state)
        donors["v2-32k-f0-refinegan"] = ("Applio RefineGAN 32k pretrain", {k: v.numpy() for k, v in refine_state.items()})
    results = {}
    for tid in configs.MATRIX:
        if only and tid not in only:
            continue
        spec = configs.TEMPLATES[tid]
        tdir = os.path.join(templates_dir, tid)
        layout = json.load(open(os.path.join(tdir, "layout.json"), encoding="utf-8"))
        state, note = matrix_weights(tid, spec, donors)
        frames = 300 if spec["vocoder"] == "refinegan" else 200
        t0 = time.time()
        rep = parity(tdir, layout, state, spec, frames, 20, seed, knn_rows=2048, knn_queries=96, weights_note=note, quiet=quiet)
        rep["seconds"] = round(time.time() - t0, 1)
        results[tid] = rep
        worst = {}
        for row in rep["rows"]:
            key = (row["check"], row["variant"])
            worst[key] = max(worst.get(key, 0.0), row["max_abs"])
        fails = [f"{r['check']}/{r['variant']} {r['max_abs']:.2e}" for r in rep["rows"] if not r["ok"]]
        print(f"{'PASS' if rep['ok'] else 'FAIL'}  {tid:<24} fp32 flowdec {worst[('flowdec', 'fp32')]:.1e} partial {worst[('partial', 'fp32')]:.1e} | "
              f"fp16 chain-mix {worst[('chain-mix', 'fp16')]:.1e} partial {worst[('partial', 'fp16')]:.1e} | {note} ({rep['seconds']} s)"
              + (f"  FAILED: {', '.join(fails)}" if fails else ""))
    return results


def cpu_parity_by_variant(rep):
    """{graph: {variant: pass|fail}} from one parity report (the CPU device class)."""
    out = {"enc": {}, "flowdec": {}, "knn": {}}
    graph_of = {"enc.m_p": "enc", "enc.logs_p": "enc", "flowdec": "flowdec", "partial": "flowdec", "fallback": "flowdec", "knn": "knn"}
    for row in rep["rows"]:
        graph = graph_of.get(row["check"])
        if graph is None:
            continue  # chain rows mix graphs; enc and flowdec rows decide
        ok = out[graph].get(row["variant"], "pass") == "pass" and row["ok"]
        out[graph][row["variant"]] = "pass" if ok else "fail"
    return out


# ---------------------------------------------------------------------------------------------
# Tiny fixture
# ---------------------------------------------------------------------------------------------


def write_tiny_fixture(out_dir, seed=20260928):
    """Random tiny RVC checkpoint in RVC's inference-checkpoint format (fp16 weights with weight_g /
    weight_v, 'config', 'info', 'sr', 'f0', 'version'), its templates and reference outputs."""
    spec = configs.TEMPLATES[TINY_TEMPLATE]
    os.makedirs(out_dir, exist_ok=True)
    export_rvc.export_template(TINY_TEMPLATE, out_dir, verbose=True)
    torch.manual_seed(seed)
    net = export_rvc.build_net(spec)
    export_rvc.perturb_(net, seed, 0.05)
    weight = {k: v.detach().half() for k, v in net.state_dict().items() if not k.startswith("enc_q.")}
    ckpt = collections.OrderedDict()
    ckpt["weight"] = weight  # a plain dict, as RVC's savee / extract_small_model write it
    ckpt["config"] = spec["config"]
    ckpt["info"] = "sigil tiny fixture (random weights)"
    ckpt["sr"] = "16k"
    ckpt["f0"] = 1
    ckpt["version"] = "v2"
    path = os.path.join(out_dir, "tiny.pth")
    torch.save(ckpt, path)
    print(f"wrote {path} ({os.path.getsize(path)} bytes)")
    return path


def _applio_keys(state):
    """RVC ``weight_g`` / ``weight_v`` keys -> the ``parametrizations.weight.original0/1`` names newer
    Applio checkpoints carry (the importer accepts both)."""
    out = collections.OrderedDict()
    for k, v in state.items():
        if k.endswith(".weight_g"):
            k = k[: -len(".weight_g")] + ".parametrizations.weight.original0"
        elif k.endswith(".weight_v"):
            k = k[: -len(".weight_v")] + ".parametrizations.weight.original1"
        out[k] = v
    return out


def write_tiny_fixture_v2(out_dir, seed=20260930):
    """Tiny fixtures of the layout-version-2 kinds (``configs.TINY_V2``) in ``out_dir``: each template
    folder, a checkpoint as its trainer writes it and the reference outputs:

    - ``tiny-nof0.pth``: RVC WebUI style (``weight_g`` / ``weight_v``, f0 0, sr "16k");
    - ``tiny-mrf.pth`` / ``tiny-refinegan.pth``: Applio style (``parametrizations.weight.original0/1``
      keys, ``vocoder`` "MRF HiFi-GAN" / "RefineGAN", ``embedder_model`` "contentvec", sr 16000);
    - ``reference-<kind>.{json,bin}`` and ``parity_report-<kind>.json`` (RefineGAN's bundle carries
      its AdaIN pool: the tiny template's pool is 16384 values).
    """
    os.makedirs(out_dir, exist_ok=True)
    kinds = {"nof0": "v2-16k-nof0-hifigan", "mrf": "v2-16k-f0-mrf-hifigan", "refinegan": "v2-16k-f0-refinegan"}
    ok = True
    for kind, tid in kinds.items():
        spec = configs.TEMPLATES[tid]
        export_rvc.export_template(tid, out_dir, verbose=True)
        torch.manual_seed(seed)
        net = export_rvc.build_net(spec)
        export_rvc.perturb_(net, seed, 0.05)
        weight = collections.OrderedDict((k, v.detach().half()) for k, v in net.state_dict().items() if not k.startswith("enc_q."))
        ckpt = collections.OrderedDict()
        if spec["vocoder"] == "hifigan":
            ckpt["weight"] = dict(weight)
            ckpt["config"] = spec["config"]
            ckpt["info"] = f"sigil tiny {kind} fixture (random weights)"
            ckpt["sr"] = "16k"
            ckpt["f0"] = 0
            ckpt["version"] = "v2"
        else:
            ckpt["weight"] = dict(_applio_keys(weight))
            ckpt["config"] = spec["config"]
            ckpt["info"] = f"sigil tiny {kind} fixture (random weights)"
            ckpt["sr"] = 16000
            ckpt["f0"] = 1
            ckpt["version"] = "v2"
            ckpt["vocoder"] = {"mrf": "MRF HiFi-GAN", "refinegan": "RefineGAN"}[kind]
            ckpt["embedder_model"] = "contentvec"
        path = os.path.join(out_dir, f"tiny-{kind}.pth")
        torch.save(ckpt, path)
        print(f"wrote {path} ({os.path.getsize(path)} bytes)")
        layout = json.load(open(os.path.join(out_dir, tid, "layout.json"), encoding="utf-8"))
        frames = 200 if kind == "refinegan" else 64
        rep = parity(
            os.path.join(out_dir, tid), layout, model_state(load_checkpoint(path)), spec, frames, 12, 5, None, out_dir,
            knn_rows=64, knn_queries=8, fp16_scaled=True, bundle_stem=f"reference-{kind}", report_name=f"parity_report-{kind}.json",
        )
        ok &= rep["ok"]
    return ok


# ---------------------------------------------------------------------------------------------
# Maintainer cross-check against Applio's own code (not run by default)
# ---------------------------------------------------------------------------------------------


@contextlib.contextmanager
def default_dtype(dtype):
    """``torch.set_default_dtype`` for a block: modules built and tensors created inside (Applio's
    ``torch.zeros`` / ``torch.rand`` buffers included) take ``dtype``."""
    old = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old)


def applio_state(state, target_keys):
    """``state`` (RVC key names, weight_norm as ``<w>_g`` / ``<w>_v``) with each weight_norm pair
    renamed to ``parametrizations.<w>.original0`` / ``original1`` wherever the target model (its
    ``state_dict`` keys) uses torch's parametrized weight_norm, as Applio's generators do."""
    out = collections.OrderedDict()
    for key, value in state.items():
        for suffix, index in (("_g", 0), ("_v", 1)):
            if key.endswith(suffix) and key not in target_keys:
                base, name = key[: -len(suffix)].rsplit(".", 1)
                alt = f"{base}.parametrizations.{name}.original{index}"
                if alt in target_keys:
                    key = alt
                break
        out[key] = value
    return out


def load_exact(module, state, ignore=("enc_q.",)):
    """``load_state_dict`` that fails on any key the module does not take or does not get (prefixes in
    ``ignore`` may be missing), naming them all: a silently skipped key would keep its random init and
    surface only as an unexplained error."""
    missing, unexpected = module.load_state_dict(state, strict=False)
    missing = [k for k in missing if not k.startswith(tuple(ignore))]
    if missing or unexpected:
        raise SystemExit(f"{type(module).__name__}: the checkpoint does not fit: missing {missing}, unexpected {list(unexpected)}")


def port_precision_floor(state, spec, inp):
    """max |port fp32 - port float64| on these inputs: how far a correct fp32 implementation (Applio's
    own ``infer`` runs fp32) already is from the float64 reference."""
    want = applio_port_decode(reference_net(state, spec), spec, inp, torch.float64)[3]
    got = applio_port_decode(reference_net(state, spec), spec, inp, torch.float32)[3]
    return float((got.double() - want).abs().max())


APPLIO_TOL = 1e-4  # float64 against float64: an exact port lands near 1e-15; this only absorbs op order


def upstream_applio_check(template_id, seed=5, frames=160):
    """Sigil's RefineGAN / MRF reference decode vs Applio's ``Synthesizer.infer`` (vendored in
    applio_arch) on the same weights and noise, both in float64 (the port's fp32 decode alone is
    ~2e-4 from its float64 decode at 48 kHz, more than the gate, so mixed precisions would blame the
    port for fp32 rounding). Applio's model is built under a float64 default dtype, so its internal
    buffers are float64 too, and loaded with every key accounted for. Returns (error, fp32 floor).
    An exact port lands at ~1e-15. A maintainer step: it runs the vendored Applio code."""
    from applio_arch import synthesizers  # noqa: F401  (maintainer step only)

    spec = configs.TEMPLATES[template_id]
    if spec["vocoder"] == "hifigan":
        raise SystemExit("HiFi-GAN templates are checked against upstream RVC by the default run")
    cfg = configs.config_dict(spec["config"])
    donor = _state_of(REFINEGAN_PTH if spec["vocoder"] == "refinegan" else DEFAULT_PTH)
    state, _ = donor_checkpoint(spec, donor)
    ours = reference_net(state, spec)
    inp = make_inputs(spec, frames, seed, ours)
    floor = port_precision_floor(state, spec, inp)
    # The reference under the same float64 default dtype as Applio's model below: constants built from
    # Python floats at run time (torchaudio's Kaiser-window beta, for one) then match too (a float32
    # default rounds them, ~5e-8 here: harmless, but it would blur a sharp comparison).
    with default_dtype(torch.float64):
        want = reference_outputs(reference_net(state, spec), spec, inp)["audio"]  # float64
    vocoder = {"refinegan": "RefineGAN", "mrf-hifigan": "MRF HiFi-GAN"}[spec["vocoder"]]
    config = list(spec["config"])
    config[15] = int(state["emb_g.weight"].shape[0])  # the checkpoint's speaker count, as reference_net
    upp = configs.upp(spec["config"])
    z_unit = torch.from_numpy(inp["z_noise_unit"])
    src = torch.from_numpy(inp["src_noise"])
    pool = torch.from_numpy(inp["pool"]) if "pool" in inp else None
    adain_layers = {(s, l): (c, r, b) for s, l, c, r, b in vocoders.adain_layers(ours.dec)[0]} if spec["vocoder"] == "refinegan" else {}
    calls = {"adain": 0}
    orig_randn_like, orig_rand = torch.randn_like, torch.rand

    def fake_randn_like(x, *a, **k):
        if tuple(x.shape) == tuple(z_unit.shape):
            return z_unit.clone().to(x.dtype)
        if x.dim() == 3 and x.shape[1] == frames * upp:  # sine source noise [1, S, H]
            return src.reshape(1, frames * upp, -1).clone().to(x.dtype)
        i = calls["adain"]  # AdaIN: 6 per stage, in call order
        calls["adain"] += 1
        c, r, b = adain_layers[(i // 6, i % 6)]
        return vocoders.adain_noise_from_pool(pool, torch.from_numpy(inp["noise_offsets"][0]), b, x).to(x.dtype)

    def fake_rand(*shape, **k):
        out = orig_rand(*shape, **k)
        if len(shape) == 2 and shape[1] == len(vocoders.MRF_PHASE_INI):
            out = torch.tensor([vocoders.MRF_PHASE_INI], dtype=out.dtype)
        return out

    with default_dtype(torch.float64):
        theirs = synthesizers.Synthesizer(*config, use_f0=True, text_enc_hidden_dim=spec["emb_channels"], vocoder=vocoder)
        if hasattr(theirs, "enc_q"):
            del theirs.enc_q
        load_exact(theirs, applio_state(state, set(theirs.state_dict())))
        theirs = theirs.double().eval()
        t = {k: torch.from_numpy(inp[k]) for k in ("phone", "pitch", "f0", "sid")}
        t["phone"], t["f0"] = t["phone"].double(), t["f0"].double()
        torch.randn_like, torch.rand = fake_randn_like, fake_rand
        try:
            with torch.no_grad():
                got, _, _ = theirs.infer(t["phone"], torch.tensor([frames]), t["pitch"], t["f0"], t["sid"])
        finally:
            torch.randn_like, torch.rand = orig_randn_like, orig_rand
    if got.dtype != torch.float64:
        raise SystemExit(f"{template_id}: Applio's infer returned {got.dtype}, not float64 (an fp32 path would hide port errors under rounding)")
    err = float(np.abs(got.reshape(1, -1).numpy() - want).max())
    print(f"{template_id}: Sigil reference vs Applio Synthesizer.infer (both float64) max |d| {err:.3e} (gate {APPLIO_TOL:g}; "
          f"the port's own fp32 vs float64 floor here {floor:.1e}) ({cfg['upsample_rates']})")
    return err, floor


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pth", default=None, help="RVC .pth checkpoint (default: the MIT default voice for v2-40k-f0-hifigan)")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(configs.TEMPLATES))
    ap.add_argument("--templates", default=DEFAULT_TEMPLATES_DIR)
    ap.add_argument("--frames", type=int, default=None, help="context frames (10 ms each)")
    ap.add_argument("--block-frames", type=int, default=20, help="frames returned by the partial decode")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--bundle-out", default=None, help="write the reference bundle for the Rust test here")
    ap.add_argument("--rust-weights", default=None, help="slot dir written by the Rust importer")
    ap.add_argument("--write-fixture", default=None, help="regenerate the tiny fixture in this folder")
    ap.add_argument("--write-fixture-v2", default=None, help="write the tiny layout-version-2 fixtures (f0-less, MRF, RefineGAN) in this folder")
    ap.add_argument("--all", action="store_true", help="every template of the matrix")
    ap.add_argument("--report", default=None, help="--all: write the per-template rows here (JSON)")
    ap.add_argument("--upstream-applio", action="store_true", help="maintainer check of the RefineGAN / MRF ports against Applio's code")
    ap.add_argument("--ep", default="cpu", choices=sorted(EPS), help="execution provider of the template sessions (gpu class: dml / cuda)")
    args = ap.parse_args()
    global EP_PROVIDERS
    EP_PROVIDERS = EPS[args.ep]

    if args.write_fixture:
        pth = write_tiny_fixture(args.write_fixture)
        template_dir = os.path.join(args.write_fixture, TINY_TEMPLATE)
        layout = json.load(open(os.path.join(template_dir, "layout.json"), encoding="utf-8"))
        rep = parity(
            template_dir, layout, model_state(load_checkpoint(pth)), configs.TEMPLATES[TINY_TEMPLATE], 64, 12, args.seed, None,
            args.write_fixture, knn_rows=64, knn_queries=8, fp16_scaled=True,
        )
        raise SystemExit(0 if rep["ok"] else 1)

    if args.write_fixture_v2:
        raise SystemExit(0 if write_tiny_fixture_v2(args.write_fixture_v2) else 1)

    if args.upstream_applio:
        errs = [upstream_applio_check(t)[0] for t in configs.MATRIX if configs.TEMPLATES[t]["vocoder"] != "hifigan"]
        raise SystemExit(0 if max(errs) <= APPLIO_TOL else 1)

    if args.all:
        results = parity_matrix(args.templates)
        if args.report:
            sl.write_json(args.report, results)
        bad = [t for t, r in results.items() if not r["ok"]]
        print(f"{len(results) - len(bad)}/{len(results)} templates pass" + (f"; failing: {', '.join(bad)}" if bad else ""))
        raise SystemExit(1 if bad else 0)

    spec = configs.TEMPLATES[args.template]
    template_dir = os.path.join(args.templates, args.template)
    layout = json.load(open(os.path.join(template_dir, "layout.json"), encoding="utf-8"))
    pth = args.pth or (DEFAULT_PTH if args.template == DEFAULT_TEMPLATE else REFINEGAN_PTH if args.template == "v2-32k-f0-refinegan" else None)
    if pth and os.path.isfile(pth):
        state = {k: v.numpy() for k, v in model_state(load_checkpoint(pth)).items()}
        note = f"real voice ({os.path.basename(pth)})"
    elif args.template == DEFAULT_TEMPLATE:
        raise SystemExit(f"checkpoint not found: {pth} (fetch PhoenixStormJr/RVC-V2-default-voice default.pth)")
    else:
        donor_path = REFINEGAN_PTH if spec["vocoder"] == "refinegan" else DEFAULT_PTH
        state, prov = donor_checkpoint(spec, _state_of(donor_path))
        state = {k: v.numpy() for k, v in state.items()}
        note = f"donor build: {prov['from_donor']} tensors from {os.path.basename(donor_path)}, {prov['random']} random"
    frames = args.frames or (300 if spec["vocoder"] == "refinegan" else 200)
    default_bundle = args.template == DEFAULT_TEMPLATE and args.ep == "cpu" and not args.report
    bundle = args.bundle_out or (os.path.join(DEV_CACHE, "vc-parity", args.template) if default_bundle else None)
    print(f"template {args.template}  weights: {note}  ep: {args.ep}")
    rep = parity(template_dir, layout, state, spec, frames, args.block_frames, args.seed, args.rust_weights, bundle, weights_note=note)
    rep["ep"] = args.ep
    if args.report:
        sl.write_json(args.report, rep)
    print("PARITY OK" if rep["ok"] else "PARITY FAILED")
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
