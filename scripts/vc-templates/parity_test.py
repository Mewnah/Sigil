#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Parity gate for Sigil's RVC split-graph templates (fp32 <= 1e-4, fp16 <= 5e-3, partial decode ==
the full-decode tail).

The reference is the *upstream* RVC synthesizer (vendored in ``rvc_arch``) running
``SynthesizerTrnMs768NSFsid.infer`` in PyTorch fp32 with weight_norm hooks active, its random draws
replaced by fixed noise. The candidate is the template graphs in ONNX Runtime with the weights written
from the checkpoint by the layout (Python reference writer, or the Rust importer's output with
``--rust-weights``) and injected in memory, exactly as the Rust engine does.

Checks (per variant):
    enc        m_p / logs_p vs upstream enc_p
    flowdec    whole-window decode vs upstream infer (its first flow+dec margin frames excluded)
    chain      ORT enc -> ORT flowdec vs upstream infer
    partial    flowdec over the last R + margins frames with phase0 vs the full-decode tail
    knn        vs a float64 brute-force RVC retrieval blend
    fallback   graph + weights file on disk (ONNX external-data convention) == in-memory result

Usage (repo root):

    python scripts/vc-templates/parity_test.py
        MIT default voice (PhoenixStormJr/RVC-V2-default-voice default.pth, from
        %LOCALAPPDATA%/sigil-dev-cache/rvc-default-voice or --pth) against the bundled
        v2-40k-f0-hifigan template; writes the reference bundle the Rust test
        `vc::import::tests::default_voice_parity` reads (--bundle-out).
    python scripts/vc-templates/parity_test.py --write-fixture src-tauri/test-fixtures/vc/tiny
        Regenerates the tiny fixture (random tiny checkpoint, its templates and reference outputs)
        used by `cargo test vc::import`.
    --rust-weights <slot dir>   also require the Rust importer's weights.f32.bin to be
                                byte-identical to the Python writer's.
"""

import argparse
import collections
import hashlib
import json
import os
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
from rvc_arch import configs, split  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")

TOL = {"fp32": 1e-4, "fp16": 5e-3}
DEV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
DEFAULT_PTH = os.path.join(DEV_CACHE, "rvc-default-voice", "default.pth")
DEFAULT_TEMPLATE = "v2-40k-f0-hifigan"
TINY_TEMPLATE = "v2-16k-f0-hifigan"  # test-only tiny config (rvc_arch/configs.py)


# ---------------------------------------------------------------------------------------------
# Checkpoint and reference
# ---------------------------------------------------------------------------------------------


def load_checkpoint(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if "weight" not in ckpt:
        raise SystemExit(f"{path}: not an RVC inference checkpoint (no 'weight')")
    return ckpt


def reference_net(ckpt, spec):
    net = export_rvc.synthesizer_class(spec["version"], spec["f0"])(*ckpt["config"], is_half=False)
    missing, unexpected = net.load_state_dict(ckpt["weight"], strict=False)
    bad = [k for k in missing if not k.startswith("enc_q.")]
    if bad or unexpected:
        raise SystemExit(f"checkpoint does not fit the upstream model: missing {bad[:5]}, unexpected {unexpected[:5]}")
    if hasattr(net, "enc_q"):
        del net.enc_q
    return net.float().eval()


class FixedNoise:
    """Replace ``torch.randn_like`` (RVC's two random draws in ``infer``) by fixed tensors, by shape."""

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


def make_inputs(spec, frames, seed):
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
    return {
        "phone": phone[None].astype(np.float32),
        "pitch": coarse_pitch(f0.copy())[None],
        "f0": f0[None].astype(np.float32),
        "z_noise_unit": rng.standard_normal((1, cfg["inter_channels"], frames)).astype(np.float32),
        "src_noise": rng.standard_normal((1, frames * upp)).astype(np.float32),
        "sid": np.array([0], dtype=np.int64),
    }


def reference_outputs(net, spec, inp):
    frames = inp["phone"].shape[1]
    upp = configs.upp(spec["config"])
    t = {k: torch.from_numpy(v) for k, v in inp.items()}
    lengths = torch.tensor([frames])
    noise = {
        t["z_noise_unit"].shape: t["z_noise_unit"],
        (1, frames * upp, 1): t["src_noise"].reshape(1, -1, 1),
    }
    with torch.no_grad(), FixedNoise(noise):
        m_p, logs_p, _ = net.enc_p(t["phone"], t["pitch"], lengths)
        audio, _, _ = net.infer(t["phone"], lengths, t["pitch"], t["f0"], t["sid"])
        g = net.emb_g(t["sid"]).unsqueeze(-1)
    return {
        "m_p": m_p.numpy(),
        "logs_p": logs_p.numpy(),
        "audio": audio.reshape(1, -1).numpy(),
        "g": g.numpy(),
    }


def nsf_phase(f0_frames, sr, upp):
    """NSF sine phase (cycles, [0, 1)) at the first sample of each frame, float64 (what the Rust
    engine carries across ticks as ``phase0``)."""
    step = np.fmod(upp * np.asarray(f0_frames, dtype=np.float64) / sr + 0.5, 1.0) - 0.5
    return np.mod(np.concatenate([[0.0], np.cumsum(step)]), 1.0)


# ---------------------------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------------------------


def ort_session(template_dir, layout, graph, variant, weights_f32, weights_f16):
    """Session for one template graph with every weight injected from memory by name."""
    import onnx
    import onnxruntime as ort

    info = layout["graphs"][graph]["variants"][variant]
    path = os.path.join(template_dir, info["file"])
    if sl.sha256_file(path) != info["sha256"]:
        raise SystemExit(f"{info['file']} does not match layout.json (re-export the template)")
    so = ort.SessionOptions()
    so.log_severity_level = 3
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
        so.add_external_initializers(names, values)
        keep.append(values)
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]), keep


def run(sess, feeds):
    names = [i.name for i in sess.get_inputs()]
    return sess.run(None, {n: feeds[n] for n in names})


def fallback_session(template_dir, layout, graph, variant, weights):
    """ONNX external-data convention: the graph next to its weights file, loaded from disk."""
    import onnxruntime as ort
    import shutil

    info = layout["graphs"][graph]["variants"][variant]
    tmp = tempfile.mkdtemp(prefix="sigil-vc-")
    shutil.copy(os.path.join(template_dir, info["file"]), tmp)
    with open(os.path.join(tmp, info["weights_file"]), "wb") as f:
        f.write(weights)
    so = ort.SessionOptions()
    so.log_severity_level = 3
    return ort.InferenceSession(os.path.join(tmp, info["file"]), so, providers=["CPUExecutionProvider"]), tmp


# ---------------------------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------------------------


class Report:
    """Max-abs gate per check. ``fp16_scaled`` (tiny random fixture only) multiplies the fp16
    tolerance by max(1, peak |reference|): fp16's absolute resolution grows with magnitude, and the
    random tiny model has larger activations than trained voices (whose gate stays absolute)."""

    def __init__(self, fp16_scaled=False):
        self.rows = []
        self.failed = False
        self.fp16_scaled = fp16_scaled

    def add(self, check, variant, got, want, tol, note=""):
        got = np.asarray(got, dtype=np.float64)
        want = np.asarray(want, dtype=np.float64)
        if got.shape != want.shape:
            raise SystemExit(f"{check}/{variant}: shape {got.shape} != {want.shape}")
        if self.fp16_scaled and variant == "fp16":
            tol = tol * max(1.0, float(np.abs(want).max()))
        err = float(np.abs(got - want).max())
        rel = float(np.linalg.norm(got - want) / max(np.linalg.norm(want), 1e-12))
        ok = err <= tol
        self.failed |= not ok
        self.rows.append({"check": check, "variant": variant, "max_abs": err, "rel_l2": rel, "tol": tol, "ok": ok, "note": note})
        print(f"  {'PASS' if ok else 'FAIL'}  {check:<10} {variant:<5} max|d| {err:.3e}  rel_l2 {rel:.3e}  (tol {tol:.3g}) {note}")


def parity(
    template_dir, layout, ckpt, spec, frames, block_frames, seed, rust_weights=None, bundle_out=None,
    knn_rows=2048, knn_queries=96, fp16_scaled=False,
):
    sr, upp = configs.sample_rate(spec["config"]), configs.upp(spec["config"])
    st = layout["streaming"]
    fm, dm = st["flow_margin"], st["dec_margin"]
    if configs.arch_key(ckpt["config"]) != configs.arch_key(layout["arch"]["config"]):
        raise SystemExit("checkpoint architecture does not match the template")

    t0 = time.time()
    state = sl.normalized_state({k: v.float().numpy() for k, v in ckpt["weight"].items()})
    weights = sl.write_weights(layout, state)
    weights16 = sl.f16_weights(weights)
    speakers = sl.speaker_table(layout, state)
    print(f"weights.f32.bin: {len(weights)} bytes ({time.time() - t0:.2f} s, Python reference writer)")
    if rust_weights:
        rw = open(os.path.join(rust_weights, sl.WEIGHTS_F32), "rb").read()
        same = rw == weights
        print(f"  {'PASS' if same else 'FAIL'}  rust weights.f32.bin byte-identical to the Python writer ({len(rw)} bytes)")
        rs = np.fromfile(os.path.join(rust_weights, sl.SPEAKERS_F32), dtype="<f4").reshape(speakers.shape)
        same_spk = np.array_equal(rs, speakers)
        print(f"  {'PASS' if same_spk else 'FAIL'}  rust speakers.f32.bin identical")
        if not (same and same_spk):
            raise SystemExit(1)

    net = reference_net(ckpt, spec)
    inp = make_inputs(spec, frames, seed)
    t0 = time.time()
    ref = reference_outputs(net, spec, inp)
    print(f"upstream PyTorch reference: {time.time() - t0:.2f} s for {frames} frames")
    if not np.allclose(ref["g"][0, :, 0], speakers[0], atol=0):
        raise SystemExit("speaker table row 0 != upstream emb_g(0)")

    z_noise = (inp["z_noise_unit"] * np.float32(st["noise_scale"])).astype(np.float32)
    phase = nsf_phase(inp["f0"][0], sr, upp)
    first_full = (fm + dm) * upp
    ref_tail = ref["audio"][:, first_full:]
    win = block_frames + fm + dm
    s = frames - win
    rep = Report(fp16_scaled)
    outputs = {}

    for variant in ("fp32", "fp16"):
        tol = TOL[variant]
        enc, _k1 = ort_session(template_dir, layout, "enc", variant, weights, weights16)
        fd, _k2 = ort_session(template_dir, layout, "flowdec", variant, weights, weights16)
        m, logs = run(enc, {"phone": inp["phone"], "pitch": inp["pitch"]})
        rep.add("enc.m_p", variant, m, ref["m_p"], tol)
        rep.add("enc.logs_p", variant, logs, ref["logs_p"], tol)
        feeds = {
            "m_p": ref["m_p"],
            "logs_p": ref["logs_p"],
            "z_noise": z_noise,
            "g": ref["g"],
            "f0": inp["f0"],
            "phase0": np.zeros((1, 1), np.float32),
            "src_noise": inp["src_noise"],
        }
        (full,) = run(fd, feeds)
        rep.add("flowdec", variant, full, ref_tail, tol, f"{frames} frames")
        (chain,) = run(fd, dict(feeds, m_p=m, logs_p=logs))
        rep.add("chain", variant, chain, ref_tail, tol)
        if variant == "fp16":
            # The recommended GPU mix (layout.json "recommended"): enc fp32, flowdec fp16.
            (mixed,) = run(fd, dict(feeds, m_p=outputs["fp32"]["m_p"], logs_p=outputs["fp32"]["logs_p"]))
            rep.add("chain-mix", variant, mixed, ref_tail, tol, "enc fp32 -> flowdec fp16")
        part_feeds = {
            "m_p": ref["m_p"][:, :, s:],
            "logs_p": ref["logs_p"][:, :, s:],
            "z_noise": z_noise[:, :, s:],
            "g": ref["g"],
            "f0": inp["f0"][:, s:],
            "phase0": np.array([[phase[s]]], dtype=np.float32),
            "src_noise": inp["src_noise"][:, s * upp :],
        }
        (part,) = run(fd, part_feeds)
        rep.add("partial", variant, part, ref["audio"][:, -block_frames * upp :], tol, f"R={block_frames} of F={win} frames, vs upstream")
        rep.add("partial", variant, part, full[:, -block_frames * upp :], tol, "vs this graph's full decode")
        outputs[variant] = {"m_p": m, "logs_p": logs, "audio_full": full, "audio_partial": part}

        if variant == "fp32":
            sess, tmp = fallback_session(template_dir, layout, "flowdec", "fp32", weights)
            (fb,) = run(sess, feeds)
            rep.add("fallback", variant, fb, full, 0.0, "graph + weights file on disk == in-memory")
            del sess

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
        "frames": frames,
        "block_frames": block_frames,
        "window_frames": win,
        "flow_margin": fm,
        "dec_margin": dm,
        "rows": rep.rows,
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
                ("ref_audio", ref["audio"]),
                ("ref_g", ref["g"]),
                ("knn_feats", feats),
                ("knn_index", index),
                ("knn_rate", rate),
                ("knn_ref_fp32", split.knn_reference(feats, index, float(rate[0]))[0].astype(np.float32)),
                ("knn_ref_fp16", split.knn_reference(feats, index.astype(np.float16).astype(np.float32), float(rate[0]))[0].astype(np.float32)),
            ]
        )
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
            "source": "upstream RVC SynthesizerTrnMs768NSFsid.infer, PyTorch fp32, fixed noise",
        }
        sl.write_bundle(os.path.join(bundle_out, "reference"), arrays, meta)
        sl.write_json(os.path.join(bundle_out, "parity_report.json"), report)
        print(f"reference bundle -> {bundle_out}")
    return report


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pth", default=DEFAULT_PTH, help="RVC .pth checkpoint (default: the MIT default voice)")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE, choices=sorted(configs.TEMPLATES))
    ap.add_argument("--templates", default=os.path.join(REPO, "src-tauri", "resources", "vc-templates"))
    ap.add_argument("--frames", type=int, default=200, help="context frames (10 ms each)")
    ap.add_argument("--block-frames", type=int, default=20, help="frames returned by the partial decode")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--bundle-out", default=None, help="write the reference bundle for the Rust test here")
    ap.add_argument("--rust-weights", default=None, help="slot dir written by the Rust importer")
    ap.add_argument("--write-fixture", default=None, help="regenerate the tiny fixture in this folder")
    args = ap.parse_args()

    if args.write_fixture:
        pth = write_tiny_fixture(args.write_fixture)
        template_dir = os.path.join(args.write_fixture, TINY_TEMPLATE)
        layout = json.load(open(os.path.join(template_dir, "layout.json"), encoding="utf-8"))
        rep = parity(
            template_dir, layout, load_checkpoint(pth), configs.TEMPLATES[TINY_TEMPLATE], 64, 12, args.seed, None,
            args.write_fixture, knn_rows=64, knn_queries=8, fp16_scaled=True,
        )
        raise SystemExit(0 if rep["ok"] else 1)

    template_dir = os.path.join(args.templates, args.template)
    layout = json.load(open(os.path.join(template_dir, "layout.json"), encoding="utf-8"))
    if not os.path.isfile(args.pth):
        raise SystemExit(f"checkpoint not found: {args.pth} (fetch PhoenixStormJr/RVC-V2-default-voice default.pth)")
    bundle = args.bundle_out or os.path.join(DEV_CACHE, "vc-parity", args.template)
    print(f"template {args.template}  checkpoint {args.pth}")
    rep = parity(template_dir, layout, load_checkpoint(args.pth), configs.TEMPLATES[args.template], args.frames, args.block_frames, args.seed, args.rust_weights, bundle)
    print("PARITY OK" if rep["ok"] else "PARITY FAILED")
    raise SystemExit(0 if rep["ok"] else 1)


if __name__ == "__main__":
    main()
