#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export the voice changer's content encoders (``vc/embedder-*``) to ONNX, with their parity gate.

Sources are the Hugging Face-format checkpoints Applio distributes (IAHispano/Applio,
``Resources/embedders/<name>/{config.json, pytorch_model.bin}``: HuBERT-base architectures with a
``final_proj``). They are read with ``torch.load(weights_only=True)`` into the installed
``transformers`` ``HubertModel`` (no downloaded code runs).

Every embedder takes a raw 16 kHz waveform (no normalisation, as RVC and Applio feed it) and
returns time-major features: ``audio`` f32 [1, N] -> ``feats`` f32 [1, T, C], with
T = floor((N - 400) / 320) + 1 (a 400-sample receptive field, 320-sample hop: 20 ms frames; checked
for every checkpoint). That is the contract ``rvc.rs`` already feeds its BYO encoders, so the legacy
engine can use these files too.

| id | checkpoint | output |
| --- | --- | --- |
| vc/embedder-contentvec | contentvec | layer 12 (last hidden state), 768: RVC v2 / Applio |
| vc/embedder-contentvec-256 | contentvec | layer 9 through final_proj, 256: RVC v1 |
| vc/embedder-spin-v2 | spin-v2 | last hidden state, 768 (Applio "spin-v2") |
| vc/embedder-spin | spin | last hidden state, 768 (Applio "spin") |
| vc/embedder-hubert-ja / -ko / -zh | japanese / korean / chinese_hubert_base | last hidden state, 768 |

Variants per id (``vc/assets.rs``: ``model.<variant>.onnx``):
- ``fp32``;
- ``fp16``: fp16 inside, fp32 input and output (the GPU variant); the convolutional front end
  (with its whole-signal GroupNorm) stays fp32, the transformer runs fp16;
- ``int8``: ONNX Runtime dynamic quantisation of the attention (and feature) projections
  (per-channel int8 weights; the feed-forward MatMuls stay fp32, see ``quantize_int8``), for CPUs
  with VNNI (``int8_gate``).

Parity (``--check``, default on): each variant in ONNX Runtime against the PyTorch fp32 model on real
speech (the dev-cache transcribe samples), gated on relative L2: fp32 <= 1e-4, fp16 <= 1e-2 with mean
frame cosine >= 0.9995 (the accuracy class of upstream RVC's own half-precision HuBERT), int8 <= 7e-2
with cosine >= 0.997. ``--ep dml`` measures the GPU class. Results go to reports/embedders*.json, and
each entry folder gets an ``embedder.json`` (contract, parity per variant and device class, and the
recommended variant per class) that the engine reads.

Usage (repo root):

    python scripts/vc-templates/export_embedders.py                       # every embedder
    python scripts/vc-templates/export_embedders.py --only contentvec --only contentvec-256
    python scripts/vc-templates/export_embedders.py --ep dml   (parity of existing exports on DirectML,
                                                                in an onnxruntime-directml environment)

Outputs land in ``--out`` (default ``%LOCALAPPDATA%/sigil-dev-cache/vc-assets``) as
``<id>/model.<variant>.onnx``: the SIGIL_LOCAL_ASSETS layout the model store installs from.
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch
from torch import nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigil_layout as sl  # noqa: E402

DEV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
SRC = os.path.join(DEV_CACHE, "applio-hf", "embedders")
OUT = os.path.join(DEV_CACHE, "vc-assets")
SAMPLES = os.path.join(DEV_CACHE, "models", "transcribe", "samples")
REPORTS = os.path.join(HERE, "reports")
OPSET = 17
RECEPTIVE_FIELD, HOP = 400, 320

# name -> (checkpoint folder, hidden layers kept (None = all), final_proj)
EMBEDDERS = {
    "contentvec": ("contentvec", None, False),
    "contentvec-256": ("contentvec", 9, True),
    "spin-v2": ("spin-v2", None, False),
    "spin": ("spin", None, False),
    "hubert-ja": ("japanese_hubert_base", None, False),
    "hubert-ko": ("korean_hubert_base", None, False),
    "hubert-zh": ("chinese_hubert_base", None, False),
}
# Applio's HF repo pin (the checkpoints' source; see the catalog fragment vc-embedders.json).
APPLIO_HF_REVISION = "70ed563897504c756ec94067c12c902c4fd42025"
# int8 is an opt-in CPU speed lever (int8_gate); its real acceptance is the downstream fidelity
# (T5) the benchmark measures, so its gate here only catches a broken export.
# fp16: upstream RVC runs HuBERT in half precision on GPUs (`is_half`); in PyTorch on CUDA that is
# 2e-3 - 9e-3 relative L2 on these clips, so the fp16 gate asks for the same class of accuracy.
GATES = {"fp32": {"rel_l2": 1e-4}, "fp16": {"rel_l2": 1e-2, "cosine": 0.9995}, "int8": {"rel_l2": 7e-2, "cosine": 0.997}}

warnings.filterwarnings("ignore")


def catalog_id(name):
    return f"vc/embedder-{name}"


# ---------------------------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------------------------


def load_hubert(folder):
    """(HubertModel with weights, final_proj Linear or None, config dict). weight_norm folded."""
    from transformers import HubertConfig, HubertModel

    with open(os.path.join(folder, "config.json"), encoding="utf-8") as f:
        cfg_json = json.load(f)
    cfg = HubertConfig(**{k: v for k, v in cfg_json.items() if k not in ("architectures", "transformers_version")})
    cfg._attn_implementation = "eager"
    kernels, strides = cfg.conv_kernel, cfg.conv_stride
    rf, hop = 1, 1
    for k, s in zip(kernels, strides):
        rf += (k - 1) * hop
        hop *= s
    if (rf, hop) != (RECEPTIVE_FIELD, HOP):
        raise SystemExit(f"{folder}: front end {rf}/{hop}, the engine assumes {RECEPTIVE_FIELD}/{HOP}")
    model = HubertModel(cfg)
    state = torch.load(os.path.join(folder, "pytorch_model.bin"), map_location="cpu", weights_only=True)
    proj = None
    if "final_proj.weight" in state:
        proj = nn.Linear(state["final_proj.weight"].shape[1], state["final_proj.weight"].shape[0])
        proj.weight.data.copy_(state.pop("final_proj.weight"))
        proj.bias.data.copy_(state.pop("final_proj.bias"))
    want = set(model.state_dict())
    mapped = {}
    for k, v in state.items():
        k2 = k[len("hubert."):] if k.startswith("hubert.") else k
        if k2 not in want:
            for a, b in ((".weight_g", ".parametrizations.weight.original0"), (".weight_v", ".parametrizations.weight.original1")):
                if k2.endswith(a) and k2[: -len(a)] + b in want:
                    k2 = k2[: -len(a)] + b
        mapped[k2] = v
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    missing = [k for k in missing if k != "masked_spec_embed"]
    if missing or unexpected:
        raise SystemExit(f"{folder}: missing {missing[:5]}, unexpected {unexpected[:5]}")
    conv = model.encoder.pos_conv_embed.conv
    if hasattr(conv, "parametrizations"):
        torch.nn.utils.parametrize.remove_parametrizations(conv, "weight")
    else:
        torch.nn.utils.remove_weight_norm(conv)
    return model.eval(), proj, cfg_json


class Embedder(nn.Module):
    """audio [1, N] f32 -> feats [1, T, C] f32. ``half``: transformer (feature projection on) in fp16,
    the convolutional front end in fp32."""

    def __init__(self, hubert, layers=None, proj=None, half=False):
        super().__init__()
        self.hubert = hubert
        if layers is not None:
            self.hubert.encoder.layers = self.hubert.encoder.layers[:layers]
        self.proj = proj
        self.half_ = half
        if half:
            self.hubert.feature_projection.half()
            self.hubert.encoder.half()
            if proj is not None:
                proj.half()

    def forward(self, audio):
        h = self.hubert
        x = h.feature_extractor(audio.float())  # [1, 512, T] fp32
        x = x.transpose(1, 2)
        if self.half_:
            x = x.half()
        x = h.feature_projection(x)
        x = h.encoder(x).last_hidden_state
        if self.proj is not None:
            x = self.proj(x)
        return x.float()


def reference(hubert, layers, proj, audio):
    """PyTorch fp32 reference through transformers' own forward."""
    with torch.no_grad():
        out = hubert(audio, output_hidden_states=layers is not None)
        x = out.hidden_states[layers] if layers is not None else out.last_hidden_state
        if proj is not None:
            x = proj(x)
    return x.numpy()


# ---------------------------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------------------------


def export(module, path, example):
    import io

    import onnx

    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            module, (example,), buf, input_names=["audio"], output_names=["feats"],
            dynamic_axes={"audio": {1: "N"}, "feats": {1: "T"}}, opset_version=OPSET, do_constant_folding=True,
        )
    model = onnx.load_from_string(buf.getvalue())
    model.producer_name = "sigil-vc-assets"
    del model.metadata_props[:]
    for k, v in (("sigil.format", "sigil-vc-embedder"), ("sigil.io", "audio [1,N] f32 16 kHz -> feats [1,T,C] f32, T = (N - 400) // 320 + 1")):
        p = model.metadata_props.add()
        p.key, p.value = k, v
    onnx.save_model(model, path)


def quantize_int8(src, dst):
    """Dynamic int8 of the attention projections (and the feature projection) only. HuBERT's
    feed-forward activations carry outlier channels that per-tensor dynamic quantisation cannot
    represent (quantising them costs 10-40 % relative L2 on real speech), so the feed-forward MatMuls
    stay fp32; weight-by-activation MatMuls only (``MatMulConstBOnly``), per-channel weights."""
    import onnx
    from onnxruntime.quantization import QuantType, quantize_dynamic

    graph = onnx.load(src, load_external_data=False).graph
    exclude = [n.name for n in graph.node if n.op_type == "MatMul" and "feed_forward" in n.name]
    quantize_dynamic(
        src, dst, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul"], per_channel=True, reduce_range=False,
        nodes_to_exclude=exclude, extra_options={"MatMulConstBOnly": True},
    )


def export_embedder(name, out_root, src_root):
    folder_name, layers, final_proj = EMBEDDERS[name]
    folder = os.path.join(src_root, folder_name)
    out = os.path.join(out_root, catalog_id(name))
    os.makedirs(out, exist_ok=True)
    example = torch.randn(1, 16000, generator=torch.Generator().manual_seed(1))
    t0 = time.time()
    hubert, proj, _ = load_hubert(folder)
    export(Embedder(hubert, layers, proj if final_proj else None), os.path.join(out, "model.fp32.onnx"), example)
    quantize_int8(os.path.join(out, "model.fp32.onnx"), os.path.join(out, "model.int8.onnx"))
    hubert, proj, _ = load_hubert(folder)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    m16 = Embedder(hubert, layers, proj if final_proj else None, half=True).to(device)
    export(m16, os.path.join(out, "model.fp16.onnx"), example.to(device))
    print(f"[{name}] exported in {time.time() - t0:.0f} s -> {out}")


# ---------------------------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------------------------


def speech_inputs():
    import soundfile as sf

    out = []
    for f in ("jfk.wav", "ja-short.wav"):
        p = os.path.join(SAMPLES, f)
        if os.path.isfile(p):
            audio, sr = sf.read(p, dtype="float32")
            if sr != 16000:
                raise SystemExit(f"{p}: {sr} Hz, expected 16 kHz")
            out.append((f, audio[: 16000 * 6]))
    if not out:  # synthetic fallback: harmonic tone with vibrato and noise
        t = np.arange(16000 * 4) / 16000
        f0 = 150 + 20 * np.sin(2 * np.pi * 3 * t)
        ph = 2 * np.pi * np.cumsum(f0) / 16000
        x = sum(np.sin(k * ph) / k for k in range(1, 12)) * 0.2 + 0.01 * np.random.default_rng(0).standard_normal(t.size)
        out.append(("synthetic", x.astype(np.float32)))
    return out


def session(path, ep):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if ep == "dml":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ort.InferenceSession(path, so, providers=providers)


def check_embedder(name, out_root, src_root, ep="cpu"):
    folder_name, layers, final_proj = EMBEDDERS[name]
    hubert, proj, _ = load_hubert(os.path.join(src_root, folder_name))
    proj = proj if final_proj else None
    out = os.path.join(out_root, catalog_id(name))
    rows = []
    status = {}
    for variant in ("fp32", "fp16", "int8"):
        path = os.path.join(out, f"model.{variant}.onnx")
        if not os.path.isfile(path):
            status[variant] = "absent"
            continue
        if variant == "int8" and ep != "cpu":
            status[variant] = "cpu only"
            continue
        status[variant] = "pass"
        sess = session(path, ep)
        for label, audio in speech_inputs():
            for n in (len(audio), 16000 + 123):  # a long clip and an odd real-time context
                x = torch.from_numpy(audio[:n][None].copy())
                want = reference(hubert, layers, proj, x)
                (got,) = sess.run(None, {"audio": x.numpy()})
                frames = (n - RECEPTIVE_FIELD) // HOP + 1
                if got.shape != want.shape or got.shape[1] != frames:
                    raise SystemExit(f"{name}/{variant}: shape {got.shape}, reference {want.shape}, expected T={frames}")
                d = got.astype(np.float64) - want
                rel = float(np.linalg.norm(d) / np.linalg.norm(want))
                cos = float(np.mean(np.sum(got * want, -1) / (np.linalg.norm(got, axis=-1) * np.linalg.norm(want, axis=-1) + 1e-12)))
                gate = GATES[variant]
                good = rel <= gate["rel_l2"] and cos >= gate.get("cosine", -1)
                if not good:
                    status[variant] = "fail"
                rows.append({"variant": variant, "input": f"{label}[:{n}]", "frames": frames, "rel_l2": rel, "max_abs": float(np.abs(d).max()),
                             "cosine": cos, "peak": float(np.abs(want).max()), "ok": good})
                print(f"  {'PASS' if good else 'FAIL'}  {name:<15} {variant} {ep:<4} {label}[:{n}] T={frames} rel_l2 {rel:.2e} max|d| {np.abs(d).max():.2e} cos {cos:.5f}")
    # fp32 and fp16 must pass; int8 is optional: an export that misses its gate is not shipped (the
    # engine then runs fp32 on the CPU), which is decided on the CPU run.
    if ep == "cpu" and status.get("int8") == "fail":
        os.remove(os.path.join(out, "model.int8.onnx"))
        status["int8"] = "refused (missed the int8 gate; not shipped)"
    ok = status.get("fp32") == "pass" and status.get("fp16") == "pass"
    files = {}
    for v in ("fp32", "fp16", "int8"):
        path = os.path.join(out, f"model.{v}.onnx")
        if os.path.isfile(path):
            files[v] = {"bytes": os.path.getsize(path), "sha256": sl.sha256_file(path)}
    return {"id": catalog_id(name), "ep": ep, "ok": ok, "variants": status, "rows": rows, "files": files}


def write_manifest(name, out_root):
    """``embedder.json`` in the entry folder, from the CPU and (when present) DirectML reports."""
    folder_name, layers, final_proj = EMBEDDERS[name]
    reports = {}
    for ep, fname in (("cpu", "embedders.json"), ("gpu", "embedders_dml.json")):
        path = os.path.join(REPORTS, fname)
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                rep = json.load(f).get("embedders", {}).get(name)
            if rep:
                reports[ep] = rep
    out = os.path.join(out_root, catalog_id(name))
    variants = {}
    for v in ("fp32", "fp16", "int8"):
        if not os.path.isfile(os.path.join(out, f"model.{v}.onnx")):
            continue
        par = {}
        for cls in ("cpu", "gpu"):
            status = reports.get(cls, {}).get("variants", {}).get(v)
            par[cls] = status if status in ("pass", "fail") else "untested"
        variants[v] = {"file": f"model.{v}.onnx", "parity": par}
    rec = {
        "cpu": "fp32",
        "gpu": "fp16" if variants.get("fp16", {}).get("parity", {}).get("gpu") == "pass" else "fp32",
    }
    manifest = {
        "format": "sigil-vc-embedder",
        "format_version": 1,
        "id": catalog_id(name),
        "embedder": name,
        "source": {"repo": "IAHispano/Applio", "revision": APPLIO_HF_REVISION, "path": f"Resources/embedders/{folder_name}"},
        "io": {"input": {"name": "audio", "shape": [1, "N"], "dtype": "float32", "sample_rate": 16000, "normalized": False},
               "output": {"name": "feats", "shape": [1, "T", 256 if final_proj else 768], "dtype": "float32"},
               "frames": "T = (N - 400) // 320 + 1"},
        "frontend": {"receptive_field": RECEPTIVE_FIELD, "hop": HOP},
        "output_layer": layers or 12,
        "final_proj": bool(final_proj),
        "variants": variants,
        "recommended": rec,
        "int8_note": "int8 is an opt-in CPU speed lever for VNNI CPUs (int8_gate); shipped only where it passes its gate",
        "gates": GATES,
    }
    sl.write_json(os.path.join(out, "embedder.json"), manifest)
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", action="append", choices=sorted(EMBEDDERS))
    ap.add_argument("--src", default=SRC, help="folder of the Applio HF checkpoints (<name>/config.json + pytorch_model.bin)")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--no-export", action="store_true", help="only check existing exports")
    ap.add_argument("--ep", default="cpu", choices=("cpu", "dml"))
    args = ap.parse_args()
    names = args.only or list(EMBEDDERS)
    report_path = os.path.join(REPORTS, f"embedders{'' if args.ep == 'cpu' else '_' + args.ep}.json")
    report = {}
    if os.path.isfile(report_path):
        with open(report_path, encoding="utf-8") as f:
            report = json.load(f).get("embedders", {})
    for name in names:
        if not args.no_export and args.ep == "cpu":
            export_embedder(name, args.out, args.src)
        report[name] = check_embedder(name, args.out, args.src, args.ep)
    os.makedirs(REPORTS, exist_ok=True)
    sl.write_json(report_path, {"gates": GATES, "source_revision": APPLIO_HF_REVISION, "embedders": report})
    for name in names:
        m = write_manifest(name, args.out)
        print(f"  {m['id']}: recommended {m['recommended']}, variants {sorted(m['variants'])}")
    bad = [n for n in names if not report[n]["ok"]]
    if args.ep != "cpu":
        # a GPU-class run measures parity classes; it does not fail the export
        print("GPU class: fp16 passes for every embedder" if not bad else f"GPU class: fp16 misses the gate for {', '.join(bad)} (they run fp32 on GPUs)")
        raise SystemExit(0)
    print("EMBEDDERS OK" if not bad else f"FAILED: {', '.join(bad)}")
    raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
