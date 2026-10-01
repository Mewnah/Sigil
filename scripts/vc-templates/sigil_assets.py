# SPDX-License-Identifier: AGPL-3.0-or-later
"""Helpers shared by the base-asset scripts (export_pitch.py, export_denoise.py, export_bench.py).

Every script stages its catalog entries' files in the SIGIL_LOCAL_ASSETS layout
(``<stage>/<entry id>/<catalog path>``; default ``%LOCALAPPDATA%/sigil-dev-cache/vc-assets``), checks
each model's contract in ONNX Runtime on signals whose answer is known, and writes a report under
``scripts/vc-templates/reports``. Upstream files are read from ``upstream-assets`` in the dev cache
(fetched once, pinned by sha256 in the catalog fragments); nothing here downloads.
"""

import hashlib
import json
import os
import shutil

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
UPSTREAM = os.path.join(DEV_CACHE, "upstream-assets")
STAGE = os.path.join(DEV_CACHE, "vc-assets")
SAMPLES = os.path.join(DEV_CACHE, "models", "transcribe", "samples")
REPORTS = os.path.join(HERE, "reports")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        f.write("\n")


def upstream(rel, sha256=None):
    """Path of a fetched upstream file, verified against its pinned sha256."""
    path = os.path.join(UPSTREAM, rel)
    if not os.path.isfile(path):
        raise SystemExit(f"missing {path}: fetch the pinned upstream file first (see the catalog fragment's source URL)")
    if sha256 and sha256_file(path) != sha256:
        raise SystemExit(f"{path}: sha256 does not match the pin {sha256}")
    return path


def stage(entry_id, catalog_path, src, stage_root=STAGE):
    """Copy ``src`` to ``<stage>/<entry id>/<catalog path>``; returns (``<entry id>/<catalog path>``,
    bytes, sha256): reports record the staged location relative to the stage, never a local path."""
    dst = os.path.join(stage_root, *entry_id.split("/"), *catalog_path.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.abspath(src) != os.path.abspath(dst):
        shutil.copyfile(src, dst)
    return f"{entry_id}/{catalog_path}", os.path.getsize(dst), sha256_file(dst)


def session(path, ep="cpu"):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    providers = ["CPUExecutionProvider"]
    if ep == "dml":
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        providers = [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    return ort.InferenceSession(path, so, providers=providers)


def io_contract(path):
    """{inputs, outputs, opset} of an ONNX file (names, element types, shapes with symbolic dims)."""
    import onnx

    m = onnx.load(path, load_external_data=False)

    def one(v):
        tt = v.type.tensor_type
        return {"name": v.name, "dtype": onnx.helper.tensor_dtype_to_np_dtype(tt.elem_type).name,
                "shape": [d.dim_param if d.dim_param else int(d.dim_value) for d in tt.shape.dim]}

    inits = {i.name for i in m.graph.initializer}
    return {
        "inputs": [one(v) for v in m.graph.input if v.name not in inits],
        "outputs": [one(v) for v in m.graph.output],
        "opset": max(o.version for o in m.opset_import if o.domain in ("", "ai.onnx")),
    }


def harmonic_tone(f0, seconds=2.0, sr=16000, harmonics=10, noise=0.0, seed=0):
    """A voiced, speech-like harmonic tone (1/k amplitudes) at ``f0`` Hz (scalar or per-sample)."""
    n = int(seconds * sr)
    f = np.broadcast_to(np.asarray(f0, dtype=np.float64), (n,))
    ph = 2 * np.pi * np.cumsum(f) / sr
    x = sum(np.sin(k * ph) / k for k in range(1, harmonics + 1))
    x = 0.3 * x / np.abs(x).max()
    if noise:
        x = x + noise * np.random.default_rng(seed).standard_normal(n)
    return x.astype(np.float32)


def speech(name="jfk.wav", seconds=None):
    import soundfile as sf

    path = os.path.join(SAMPLES, name)
    if not os.path.isfile(path):
        return None
    audio, sr = sf.read(path, dtype="float32")
    if sr != 16000:
        raise SystemExit(f"{path}: {sr} Hz, expected 16 kHz")
    return audio if seconds is None else audio[: int(seconds * 16000)]


def cents(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return 1200 * np.log2(a / b)


def si_sdr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    alpha = np.dot(est, ref) / np.dot(ref, ref)
    target = alpha * ref
    return 10 * np.log10(np.sum(target**2) / np.sum((est - target) ** 2))


def report(name, obj):
    write_json(os.path.join(REPORTS, f"{name}.json"), obj)
