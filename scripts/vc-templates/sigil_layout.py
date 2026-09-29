# SPDX-License-Identifier: AGPL-3.0-or-later
"""Layout, weights and tensor-bundle helpers shared by export_rvc.py and parity_test.py.

``write_weights`` is the Python reference of the Rust importer's weights writer
(``src-tauri/src/services/neural_audio/vc/import/writer.rs``): both must produce byte-identical
``weights.f32.bin`` files for the same checkpoint (parity_test.py checks this when given the
Rust output).
"""

import hashlib
import json
import os
import re

import numpy as np

LAYOUT_FORMAT = "sigil-rvc-layout"
LAYOUT_VERSION = 1
ALIGN = 64
WEIGHTS_F32 = "weights.f32.bin"
WEIGHTS_F16 = "weights.f16.bin"
SPEAKERS_F32 = "speakers.f32.bin"

BUNDLE_FORMAT = "sigil-tensor-bundle"

_PARAMETRIZATION = re.compile(r"^(.*)\.parametrizations\.(\w+)\.original([01])$")


def align_up(n, a=ALIGN):
    return (n + a - 1) // a * a


def normalize_key(key):
    """``X.parametrizations.weight.original0/1`` (torch.nn.utils.parametrizations.weight_norm, used
    by some trainers) -> ``X.weight_g`` / ``X.weight_v`` (torch.nn.utils.weight_norm, RVC's)."""
    m = _PARAMETRIZATION.match(key)
    if not m:
        return key
    base, name, which = m.groups()
    return f"{base}.{name}_{'g' if which == '0' else 'v'}"


def normalized_state(state):
    out = {}
    for k, v in state.items():
        nk = normalize_key(k)
        if nk in out:
            raise ValueError(f"duplicate tensor after key normalisation: {nk}")
        out[nk] = v
    return out


def fold_weight_norm(g, v, dim):
    """w = g * v / ||v||, the norm taken over every axis except ``dim`` (-1: the whole tensor).

    float64 math with the Rust importer's exact summation order (``fold.rs``: a sequential sum
    over each contiguous block, then over the leading axes), so both produce identical f32 bits.
    """
    v64 = np.asarray(v, dtype=np.float64)
    g64 = np.asarray(g, dtype=np.float64).reshape(-1)
    shape = v64.shape
    if dim == -1:
        sq = np.cumsum((v64 * v64).reshape(-1))[-1:]
        scale = g64 / np.sqrt(sq)
        return (v64 * scale[0]).astype(np.float32)
    n = shape[dim]
    outer = int(np.prod(shape[:dim], dtype=np.int64))
    inner = int(np.prod(shape[dim + 1 :], dtype=np.int64))
    v3 = v64.reshape(outer, n, inner)
    block = np.cumsum(v3 * v3, axis=2)[:, :, -1]  # sequential within each block
    sq = np.cumsum(block, axis=0)[-1]  # then sequential over the leading axes
    scale = g64 / np.sqrt(sq)
    return (v3 * scale[None, :, None]).reshape(shape).astype(np.float32)


def tensor_value(entry, state):
    """The f32 value of one layout tensor from a (normalised) checkpoint state dict of float32 arrays."""
    src = entry["source"]
    if "key" in src:
        arr = np.asarray(state[src["key"]], dtype=np.float32)
    else:
        wn = src["weight_norm"]
        if wn["g"] in state and wn["v"] in state:
            arr = fold_weight_norm(state[wn["g"]], state[wn["v"]], wn["dim"])
        elif wn["folded"] in state:
            arr = np.asarray(state[wn["folded"]], dtype=np.float32)
        else:
            raise KeyError(f"checkpoint has neither {wn['g']}/{wn['v']} nor {wn['folded']}")
    if list(arr.shape) != entry["shape"]:
        raise ValueError(f"{entry['name']}: shape {list(arr.shape)} != template {entry['shape']}")
    return arr


def write_weights(layout, state):
    """weights.f32.bin bytes for a checkpoint (see the module docs)."""
    w = layout["weights"]
    buf = bytearray(w["bytes_f32"])
    for entry in w["tensors"]:
        arr = tensor_value(entry, state)
        data = arr.astype("<f4").tobytes()
        if len(data) != entry["bytes"]:
            raise ValueError(f"{entry['name']}: {len(data)} bytes != layout {entry['bytes']}")
        buf[entry["offset"] : entry["offset"] + entry["bytes"]] = data
    return bytes(buf)


def f16_weights(f32_bytes):
    """weights.f16.bin: every f32 word of weights.f32.bin cast to f16 (offsets halve)."""
    return np.frombuffer(f32_bytes, dtype="<f4").astype("<f2").tobytes()


def speaker_table(layout, state):
    spk = layout["speakers"]
    arr = np.asarray(state[spk["key"]], dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != spk["dim"]:
        raise ValueError(f"speaker table {list(arr.shape)} does not match [n, {spk['dim']}]")
    return arr


def tensor_table_id(tensors):
    """Identity of a weights layout: sha256 of its canonical JSON (embedded in every template graph
    as the ``sigil.weights_id`` metadata property, so a graph and a layout cannot be mixed up)."""
    canon = [
        {k: t[k] for k in ("name", "shape", "offset", "bytes")}
        for t in sorted(tensors, key=lambda t: t["offset"])
    ]
    blob = json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, obj):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------------------------
# Tensor bundles: <name>.json + <name>.bin (raw little-endian arrays, 64-byte aligned); read by the
# Rust tests (import/tests.rs) for reference inputs and outputs.
# ---------------------------------------------------------------------------------------------

_DTYPES = {"float32": "<f4", "int64": "<i8", "float16": "<f2"}


def write_bundle(stem, arrays, meta=None):
    entries, blob = [], bytearray()
    for name, arr in arrays.items():
        arr = np.ascontiguousarray(arr)
        dtype = {np.dtype("float32"): "float32", np.dtype("int64"): "int64", np.dtype("float16"): "float16"}[arr.dtype]
        data = arr.astype(_DTYPES[dtype]).tobytes()
        off = align_up(len(blob))
        blob.extend(b"\0" * (off - len(blob)))
        blob.extend(data)
        entries.append({"name": name, "dtype": dtype, "shape": list(arr.shape), "offset": off, "bytes": len(data)})
    os.makedirs(os.path.dirname(os.path.abspath(stem)), exist_ok=True)
    with open(stem + ".bin", "wb") as f:
        f.write(bytes(blob))
    write_json(stem + ".json", {"format": BUNDLE_FORMAT, "version": 1, "meta": meta or {}, "tensors": entries})


def read_bundle(stem):
    with open(stem + ".json", encoding="utf-8") as f:
        desc = json.load(f)
    blob = open(stem + ".bin", "rb").read()
    out = {}
    for t in desc["tensors"]:
        raw = blob[t["offset"] : t["offset"] + t["bytes"]]
        out[t["name"]] = np.frombuffer(raw, dtype=_DTYPES[t["dtype"]]).reshape(t["shape"]).copy()
    return out, desc.get("meta", {})
