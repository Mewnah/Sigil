#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Pitch-bank assets (``vc/pitch-*``): stage, adapt and check them.

| id | file (model.onnx) | source | Sigil work |
| --- | --- | --- | --- |
| vc/pitch-swiftf0 | SwiftF0 ``model.onnx`` (MIT, 135 KB) | lars76/swift-f0 (= Applio's ``swift.onnx``) | contract check |
| vc/pitch-crepe-tiny | CREPE tiny (MIT, 2 MB) | yqzhishen/onnxcrepe v1.1.0 ``tiny.onnx`` | contract check |
| vc/pitch-fcpe | FCPE (MIT, 44 MB) | gzivdo/fcpe-onnx (torchfcpe 0.0.4 weights) | **adapter**: the Sigil FCPE contract |
| vc/pitch-rmvpe | RMVPE (MIT, 362 MB) | lj1995/VoiceConversionWebUI ``rmvpe.onnx`` | contract check (``rmvpe.rs``'s mel and decoding) |
| vc/pitch-pesto | PESTO streaming (LGPL-3.0) | SonyCSLParis/pesto ``realtime/export_onnx.py`` | manual export (see below) |

Contracts (what P3-D's extractors feed; all 16 kHz mono):
- **SwiftF0**: ``audio`` f32 [1, N], ``fmin`` / ``fmax`` f32 scalars (Hz; SwiftF0's range is
  46.875-2093.75) -> ``pitch`` f64 [1, F] Hz and ``confidence`` f32 [1, F]; F = N // 256 (a 256-sample
  hop, 16 ms; measured by the check, reports/pitch.json). Voicing = confidence >= 0.9 (SwiftF0's
  default). Resample the contour to the hop-160 grid.
- **CREPE tiny**: ``frames`` f32 [n, 1024] (1024-sample windows at 16 kHz, hop 160 = 10 ms, the
  signal zero-padded by 512 on both sides; each frame minus its mean, divided by max(std, 1e-10)) ->
  ``probabilities`` f32 [n, 360] (sigmoid). Decode: local weighted mean of 9 bins around the argmax,
  cents = 1997.3794084376191 + 20 * bin, f = 10 * 2^(cents / 1200); voicing: peak >= 0.1 (torchcrepe's
  periodicity) is a reasonable start.
- **FCPE** (Sigil adapter, the contract ``fcpe.rs`` already implements): ``audio`` f32 [1, N] ->
  ``f0`` f32 [1, N // 160 + 1] Hz, 0 = unvoiced. The community export takes [1, N, 1], returns
  [1, T, 1] and can emit NaN on quiet frames (internal log(0)); the adapter reshapes both and maps NaN
  to 0 (unvoiced), and is otherwise the upstream graph byte for byte (weights and nodes untouched).
- **RMVPE**: RVC's ``rmvpe.onnx``: log-mel [1, 128, T] (T a multiple of 32) -> salience [1, T, 360]
  (``rmvpe.rs`` computes the mel and decodes).
- **PESTO** (manual): export PESTO's streaming model with its own ``realtime/export_onnx.py``
  (pesto commit 62bc0c97, ``mir-1k_g7.ckpt``) and import it as ``vc/pitch-pesto`` ``model.onnx``;
  running a downloaded project's export code was not permitted to the agent that built this, so the
  catalog entry is import-only until a maintainer runs it.

Usage (repo root): ``python scripts/vc-templates/export_pitch.py [--ep dml]``.

RMVPE graph templates (P3B-E; ``--rmvpe-templates``): weightless graphs that ``rmvpe.rs`` fills at
load with the installed ``rmvpe.onnx``'s own weights (no new hosted asset), written to
``src-tauri/src/services/neural_audio/vc/graphs/``:

- ``rmvpe_t32.graph.onnx``: exact kernel-row pruning for a 32-frame mel. After the U-Net's five
  AvgPool(2) stages, a 32-frame mel reaches the Intermediate block as a ``[1, C, 1, 4]`` map, so rows 0
  and 2 of each of the block's 32 3x3 kernels (31 of 512->512 plus the 256->512 entry) only ever
  multiply the convolution's zero padding. Those convs become ``[1, 3]`` kernels with pads
  ``[0, 1, 0, 1]`` (the middle row of the original weights, sliced by Rust at load); every other
  tensor is the original. 49,545,216 fewer parameters (-189 MiB). The mel input is fixed at
  ``[1, 128, 32]`` (the graph is only valid at that length).
- ``rmvpe_t32_fp16.graph.onnx`` / ``rmvpe_fp16.graph.onnx``: the t32 and the full graph with the
  U-Net and the ``cnn`` conv in fp16 (weights rounded by Rust at load) and the reshape, the BiGRU,
  the final Linear, the Sigmoid and the I/O kept fp32 (one Cast after the mel input, one after
  ``/cnn/Conv``; ``rmvpe_fp16_model``). Gated (``rmvpe.rs`` docs): offered only under an explicit
  fp16 precision until its stream-path pitch gate (``rmvpe.rs`` ``rmvpe_fp16_gate``, DirectML and
  CUDA) passes.

Every initializer at or above 1 KB is external data in ``rmvpe.weights`` (offsets recorded in the
graph); smaller ones stay inline. Rust fills the image from the installed ``rmvpe.onnx`` by name
(``neural_audio::onnx_proto::external_weights_image``). The check
(``reports/pitch_rmvpe_templates.json``): ONNX Runtime CPU against the full model at 32 mel frames on
20 random and speech mels (max |salience| difference, argmax agreement, F0 cents, V/UV flips); for
t32 it must be <= 1e-5 with identical argmax. The fp16 rows are informational: ONNX Runtime's x86
CPU EP has no fp16 Conv and inserts Casts, so they measure the fp16 weight rounding alone.
``--rmvpe-census`` records the fp32 per-layer max |activation| over speech, tones and random mels at
32 and 64 frames (``reports/pitch_rmvpe_census.json``): a layer above ~8,000 would need an fp32
island before fp16 (measured: at most ~1,030, a decoder ConvTranspose).
"""

import argparse
import os
import shutil
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigil_assets as sa  # noqa: E402

PINS = {
    "swiftf0": ("vc/pitch-swiftf0", "swiftf0/model.onnx", "6385e8c2ebc3872e82c9ff5946228de44cd3be77a750ea53698b7dbfe94b0a22"),
    "crepe-tiny": ("vc/pitch-crepe-tiny", "crepe/tiny.onnx", "91fc2a0fd10f965dbf7775995daf50e99273caedd7efd00001f23be649da1bc3"),
    "fcpe": ("vc/pitch-fcpe", "fcpe/fcpe.onnx", "b7e4f3871b10641869b7ac5a2d56ed94deb37552c0336d77e17ad6e66760adf0"),
}
SWIFTF0_FMIN, SWIFTF0_FMAX = 46.875, 2093.75
TONES = (98.0, 220.0, 440.0, 880.0)
CENTS_GATE = 25.0  # median error on a clean harmonic tone
CREPE_CENTS = 1997.3794084376191 + 20 * np.arange(360)


# ---------------------------------------------------------------------------------------------
# SwiftF0
# ---------------------------------------------------------------------------------------------


def swiftf0_run(sess, audio):
    pitch, conf = sess.run(None, {"audio": audio[None].astype(np.float32), "fmin": np.array(SWIFTF0_FMIN, np.float32), "fmax": np.array(SWIFTF0_FMAX, np.float32)})
    return pitch[0].astype(np.float64), conf[0]


def check_swiftf0(path, ep):
    sess = sa.session(path, ep)
    rows = []
    lengths = {}
    for n in (16000, 16000 + 777, 17850, 17920, 32000):
        p, _ = swiftf0_run(sess, sa.harmonic_tone(220, n / 16000))
        lengths[n] = int(p.shape[0])
    for f0 in TONES:
        p, c = swiftf0_run(sess, sa.harmonic_tone(f0))
        voiced = c >= 0.9
        err = float(np.median(np.abs(sa.cents(p[voiced], f0)))) if voiced.any() else float("inf")
        rows.append({"tone_hz": f0, "voiced_frames": int(voiced.sum()), "frames": int(p.size), "median_cents": err, "ok": err <= CENTS_GATE and voiced.mean() > 0.8})
    _, c = swiftf0_run(sess, np.zeros(32000, np.float32))
    silence_ok = bool((c < 0.9).mean() > 0.95)
    # hop from the frame counts: the hop h with F = N // h for every probed length
    hop = next((h for h in range(64, 1025) if all(f == n // h for n, f in lengths.items())), None)
    if hop is None:
        raise SystemExit(f"SwiftF0 frame counts {lengths} fit no N // hop rule")
    return {"rows": rows, "silence_unvoiced": silence_ok, "frames_by_samples": lengths, "hop": int(hop),
            "ok": all(r["ok"] for r in rows) and silence_ok}


# ---------------------------------------------------------------------------------------------
# CREPE tiny
# ---------------------------------------------------------------------------------------------


def crepe_frames(audio, hop=160):
    x = np.pad(audio.astype(np.float32), 512)
    n = 1 + (len(x) - 1024) // hop
    idx = np.arange(1024)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)
    fr = fr / np.maximum(fr.std(1, keepdims=True), 1e-10)
    return fr.astype(np.float32)


def crepe_decode(prob):
    out = np.zeros(prob.shape[0])
    for i, row in enumerate(prob):
        c = int(np.argmax(row))
        lo, hi = max(0, c - 4), min(360, c + 5)
        w = row[lo:hi]
        out[i] = 10 * 2 ** (np.sum(w * CREPE_CENTS[lo:hi]) / np.sum(w) / 1200)
    return out, prob.max(1)


def check_crepe(path, ep):
    sess = sa.session(path, ep)
    rows = []
    for f0 in TONES:
        (prob,) = sess.run(None, {"frames": crepe_frames(sa.harmonic_tone(f0))})
        f, peak = crepe_decode(prob)
        voiced = peak >= 0.5
        err = float(np.median(np.abs(sa.cents(f[voiced], f0)))) if voiced.any() else float("inf")
        rows.append({"tone_hz": f0, "voiced_frames": int(voiced.sum()), "frames": int(f.size), "median_cents": err, "ok": err <= CENTS_GATE and voiced.mean() > 0.8})
    return {"rows": rows, "ok": all(r["ok"] for r in rows)}


# ---------------------------------------------------------------------------------------------
# RMVPE (RVC's rmvpe.onnx; the mel and the decoding are rmvpe.rs's, mirrored here)
# ---------------------------------------------------------------------------------------------

RMVPE_SHA256 = "5370e71ac80af8b4b7c793d27efd51fd8bf962de3a7ede0766dac0befa3660fd"


def rmvpe_mel(audio):
    """RVC's MelSpectrogram for RMVPE: 16 kHz, n_fft = win = 1024, hop 160, periodic Hann, centred
    reflect padding, 128 HTK mels over 30-8000 Hz, log(clamp(x, 1e-5)); T padded to a multiple of 32."""
    import librosa
    import torch

    x = torch.from_numpy(audio.astype(np.float32))
    spec = torch.stft(x, 1024, 160, 1024, torch.hann_window(1024), center=True, return_complex=True).abs().numpy()
    basis = librosa.filters.mel(sr=16000, n_fft=1024, n_mels=128, fmin=30, fmax=8000, htk=True)
    mel = np.log(np.maximum(basis @ spec, 1e-5)).astype(np.float32)
    t = mel.shape[1]
    pad = 32 * ((t - 1) // 32 + 1) - t
    return np.pad(mel, ((0, 0), (0, pad)))[None], t


def rmvpe_decode(hidden, frames, threshold=0.03):
    cents_map = 20 * np.arange(360) + 1997.3794084376191
    out = np.zeros(frames)
    for i, row in enumerate(hidden[:frames]):
        c = int(np.argmax(row))
        if row[c] <= threshold:
            continue
        lo, hi = max(0, c - 4), min(360, c + 5)
        w = row[lo:hi]
        out[i] = 10 * 2 ** (np.sum(w * cents_map[lo:hi]) / np.sum(w) / 1200)
    return out


def check_rmvpe(path, ep):
    sess = sa.session(path, ep)
    rows = []
    for f0 in TONES:
        mel, t = rmvpe_mel(sa.harmonic_tone(f0))
        (hidden,) = sess.run(None, {sess.get_inputs()[0].name: mel})
        f = rmvpe_decode(hidden[0], t)
        v = f > 0
        err = float(np.median(np.abs(sa.cents(f[v], f0)))) if v.any() else float("inf")
        rows.append({"tone_hz": f0, "voiced_frames": int(v.sum()), "frames": int(t), "median_cents": err, "ok": err <= CENTS_GATE and v.mean() > 0.8})
    mel, t = rmvpe_mel(np.zeros(32000, np.float32))
    (hidden,) = sess.run(None, {sess.get_inputs()[0].name: mel})
    silence_ok = bool((rmvpe_decode(hidden[0], t) == 0).mean() > 0.95)
    return {"rows": rows, "silence_unvoiced": silence_ok, "ok": all(r["ok"] for r in rows) and silence_ok}


# ---------------------------------------------------------------------------------------------
# RMVPE graph templates (P3B-E): t32 kernel-row pruning and gated fp16
# ---------------------------------------------------------------------------------------------

REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
GRAPHS_DIR = os.path.join(REPO, "src-tauri", "src", "services", "neural_audio", "vc", "graphs")
RMVPE_WEIGHTS_FILE = "rmvpe.weights"
RMVPE_T32_FRAMES = 32
RMVPE_MEL_FLOOR = float(np.log(1e-5))
# Initializers below this size stay inline in the template (biases, BatchNorm statistics); the rest
# are external data that Rust fills from the installed model at load.
TEMPLATE_INLINE_BYTES = 1024


def rmvpe_intermediate_convs(model):
    """The 32 3x3 convolutions of RMVPE's Intermediate block: (node, weight initializer)."""
    import onnx

    inits = {i.name: i for i in model.graph.initializer}
    out = []
    for n in model.graph.node:
        if n.op_type != "Conv" or not n.name.startswith("/unet/intermediate/"):
            continue
        w = inits.get(n.input[1])
        if w is None or list(w.dims[2:]) != [3, 3]:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
        if list(attrs.get("pads", [])) != [1, 1, 1, 1] or list(attrs.get("strides", [1, 1])) != [1, 1] \
                or list(attrs.get("dilations", [1, 1])) != [1, 1] or attrs.get("group", 1) != 1:
            raise SystemExit(f"RMVPE {n.name}: unexpected conv attributes {attrs}")
        out.append((n, w))
    if len(out) != 32:
        raise SystemExit(f"RMVPE: expected 32 Intermediate 3x3 convs, found {len(out)}")
    return out


def rmvpe_check_one_row(model, frames=RMVPE_T32_FRAMES):
    """Shape inference at a fixed mel length: the Intermediate convs' input height (time rows)."""
    import onnx

    m = onnx.ModelProto()
    m.CopyFrom(model)
    for d, v in zip(m.graph.input[0].type.tensor_type.shape.dim, (1, 128, frames)):
        d.ClearField("dim_param")
        d.dim_value = v
    inferred = onnx.shape_inference.infer_shapes(m)
    shapes = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in inferred.graph.value_info}
    rows = sorted({tuple(shapes.get(n.input[0], [])) for n, _ in rmvpe_intermediate_convs(model)})
    return [list(r) for r in rows]


def rmvpe_t32_model(full):
    """The t32 model (with weights): the Intermediate convs' kernels cut to their middle row."""
    import onnx
    from onnx import numpy_helper

    m = onnx.ModelProto()
    m.CopyFrom(full)
    convs = rmvpe_intermediate_convs(m)
    inits = {i.name: i for i in m.graph.initializer}
    for node, w in convs:
        arr = numpy_helper.to_array(inits[w.name])
        inits[w.name].CopyFrom(numpy_helper.from_array(np.ascontiguousarray(arr[:, :, 1:2, :]), w.name))
        for a in node.attribute:
            if a.name == "pads":
                del a.ints[:]
                a.ints.extend([0, 1, 0, 1])
            if a.name == "kernel_shape":
                del a.ints[:]
                a.ints.extend([1, 3])
    for d, v in zip(m.graph.input[0].type.tensor_type.shape.dim, (1, 128, RMVPE_T32_FRAMES)):
        d.ClearField("dim_param")
        d.dim_value = v
    for d, v in zip(m.graph.output[0].type.tensor_type.shape.dim, (1, RMVPE_T32_FRAMES, 360)):
        d.ClearField("dim_param")
        d.dim_value = v
    onnx.checker.check_model(m)
    return m


def rmvpe_fp16_model(model):
    """The U-Net and the ``cnn`` conv in fp16 (every node up to and including ``/cnn/Conv``, and the
    initializers only they read), everything after it (reshape, BiGRU, Linear, Sigmoid) and the
    graph I/O in fp32: one Cast to fp16 after the mel input and one back to fp32 after ``/cnn/Conv``.
    Written by hand because onnxconverter_common's converter round-trips the blocked nodes through
    fp16 Casts and casts the graph output to fp16. Values are rounded to nearest (NumPy, like Rust's
    ``half::f16::from_f32``); a weight that overflows fp16 is an error (none does: max |w| ~19,110)."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    m = onnx.ModelProto()
    m.CopyFrom(model)
    g = m.graph
    names = [n.name for n in g.node]
    if "/cnn/Conv" not in names:
        raise SystemExit("RMVPE: no /cnn/Conv node")
    if len(g.value_info):
        raise SystemExit("RMVPE: unexpected value_info (the fp16 split would leave stale fp32 types)")
    cut = names.index("/cnn/Conv")
    region_a, region_b = list(g.node[:cut + 1]), list(g.node[cut + 1:])
    used_a = {i for n in region_a for i in n.input}
    used_b = {i for n in region_b for i in n.input}
    for n in region_a:
        for a in n.attribute:
            if a.type == onnx.AttributeProto.TENSOR and a.t.data_type == TensorProto.FLOAT:
                raise SystemExit(f"RMVPE fp16: {n.name} has a float tensor attribute; convert it too")
    in_name = g.input[0].name
    for n in region_a:
        for k, i in enumerate(n.input):
            if i == in_name:
                n.input[k] = in_name + "_fp16"
    cnn = region_a[-1]
    out = cnn.output[0]
    cnn.output[0] = out + "_fp16"
    cast_in = helper.make_node("Cast", [in_name], [in_name + "_fp16"], name="/sigil/input_to_fp16", to=TensorProto.FLOAT16)
    cast_out = helper.make_node("Cast", [out + "_fp16"], [out], name="/sigil/cnn_to_fp32", to=TensorProto.FLOAT)
    del g.node[:]
    g.node.extend([cast_in] + region_a + [cast_out] + region_b)
    converted = 0
    for init in g.initializer:
        if init.name in used_a and init.name in used_b:
            raise SystemExit(f"RMVPE fp16: {init.name} is read on both sides of the split")
        if init.name in used_a and init.data_type == TensorProto.FLOAT:
            arr = numpy_helper.to_array(init)
            half = arr.astype(np.float16)
            if not np.all(np.isfinite(half)):
                raise SystemExit(f"RMVPE fp16: {init.name} overflows fp16 (max |w| {np.abs(arr).max()})")
            init.CopyFrom(numpy_helper.from_array(half, init.name))
            converted += 1
    onnx.checker.check_model(m)
    onnx.shape_inference.infer_shapes(m, strict_mode=True)
    print(f"fp16: {converted} initializers converted, {len(region_b)} tail nodes kept fp32")
    return m


def save_weightless(model, dst, variant, source_sha, frames):
    """Write ``model``'s graph to ``dst`` with every initializer >= 1 KB as external data in
    ``rmvpe.weights`` (offsets recorded, the data file itself written to a temporary folder and
    returned, so the parity checks can load the template exactly as Rust fills it)."""
    import tempfile

    import onnx

    m = onnx.ModelProto()
    m.CopyFrom(model)
    m.producer_name = "sigil-vc-templates"
    meta = {"sigil.format": "sigil-rmvpe-template", "sigil.variant": variant, "sigil.source_sha256": source_sha,
            "sigil.weights": RMVPE_WEIGHTS_FILE, "sigil.mel_frames": str(frames or ""),
            "sigil.source": "lj1995/VoiceConversionWebUI rmvpe.onnx (MIT), sliced / cast by scripts/vc-templates/export_pitch.py"}
    del m.metadata_props[:]
    for k, v in sorted(meta.items()):
        p = m.metadata_props.add()
        p.key, p.value = k, v
    tmp = tempfile.mkdtemp(prefix="sigil-rmvpe-template-")
    tmp_graph = os.path.join(tmp, "graph.onnx")
    onnx.save_model(m, tmp_graph, save_as_external_data=True, all_tensors_to_one_file=True, location=RMVPE_WEIGHTS_FILE,
                    size_threshold=TEMPLATE_INLINE_BYTES, convert_attribute=False)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(tmp_graph, "rb") as f:
        graph_bytes = f.read()
    with open(dst, "wb") as f:
        f.write(graph_bytes)
    return tmp_graph, len(graph_bytes), sa.sha256_file(dst)


def rmvpe_parity_mels(n_random=10, n_speech=10, seed=20261003, frames=RMVPE_T32_FRAMES):
    """``n_random`` random and ``n_speech`` speech mels of exactly ``frames`` frames (31 hops of audio:
    the incremental call a 32-frame plan holds)."""
    rng = np.random.default_rng(seed)
    mels = []
    for k in range(n_random):
        if k % 2 == 0:
            m = rng.uniform(RMVPE_MEL_FLOOR, 3.0, size=(128, frames))
        else:  # smooth random field around a spectral tilt (speech-like dynamics)
            base = np.linspace(1.5, -6.0, 128)[:, None] + rng.normal(0, 1.5, size=(128, 1))
            m = base + np.cumsum(rng.normal(0, 0.4, size=(128, frames)), axis=1)
            m = np.clip(m, RMVPE_MEL_FLOOR, 4.0)
        mels.append(("random", m.astype(np.float32)))
    clips = []
    for name in sorted(os.listdir(SAMPLES_DIR)) if os.path.isdir(SAMPLES_DIR) else []:
        if name.endswith(".wav"):
            clips.append(os.path.join(SAMPLES_DIR, name))
    clip_dir = os.path.join(sa.DEV_CACHE, "sherpa-onnx", "clips")
    if os.path.isdir(clip_dir):
        clips += [os.path.join(clip_dir, n) for n in sorted(os.listdir(clip_dir)) if n.endswith(".wav")]
    n_audio = (frames - 1) * 160
    speech_audio = [a for a in (load_16k(c) for c in clips) if a is not None and len(a) > n_audio + 160]
    # Windows of active speech: at least a quarter of the clip's loud-window RMS (clips differ in
    # level), at most 50 tries per clip (a quiet clip is skipped instead of looping).
    k, tries, clip = 0, 0, 0
    while k < n_speech and speech_audio and clip < 4 * len(speech_audio):
        audio = speech_audio[clip % len(speech_audio)]
        hop = n_audio // 2
        levels = [float(np.sqrt(np.mean(audio[i:i + n_audio] ** 2))) for i in range(0, len(audio) - n_audio, hop)]
        floor = 0.25 * float(np.percentile(levels, 90)) if levels else 1.0
        start = int(rng.integers(0, len(audio) - n_audio))
        seg = audio[start:start + n_audio]
        tries += 1
        if np.sqrt(np.mean(seg ** 2)) < max(floor, 1e-4):
            if tries >= 50:
                clip, tries = clip + 1, 0
            continue
        mel, t = rmvpe_mel(seg)
        assert t == frames, (t, frames)
        mels.append(("speech", mel[0, :, :frames]))
        k, clip, tries = k + 1, clip + 1, 0
    return mels


SAMPLES_DIR = sa.SAMPLES


def load_16k(path):
    import soundfile as sf
    from math import gcd

    try:
        audio, sr = sf.read(path, dtype="float32")
    except Exception:
        return None
    if audio.ndim > 1:
        audio = audio.mean(1)
    if sr != 16000:
        from scipy.signal import resample_poly

        g = gcd(sr, 16000)
        audio = resample_poly(audio, 16000 // g, sr // g).astype(np.float32)
    return audio


def salience_compare(a, b, threshold=0.03):
    """max |a - b|, the rows whose argmax differs (among rows voiced in either), and the decoded
    F0 difference in cents on rows voiced in both."""
    diff = float(np.abs(a - b).max())
    voiced = (a.max(1) > threshold) | (b.max(1) > threshold)
    argmax_diff = int(((a.argmax(1) != b.argmax(1)) & voiced).sum())
    fa, fb = rmvpe_decode(a, a.shape[0], threshold), rmvpe_decode(b, b.shape[0], threshold)
    both = (fa > 0) & (fb > 0)
    cents = float(np.abs(sa.cents(fa[both], fb[both])).max()) if both.any() else 0.0
    vuv = int(((fa > 0) != (fb > 0)).sum())
    return diff, argmax_diff, int(voiced.sum()), cents, vuv


def build_rmvpe_templates(full_path):
    """Write the three weightless templates and check them (module docs)."""
    import onnx
    import onnxruntime as ort

    full = onnx.load(full_path)
    rows32 = rmvpe_check_one_row(full, 32)
    rows64 = rmvpe_check_one_row(full, 64)
    if any(r[2] != 1 for r in rows32):
        raise SystemExit(f"RMVPE: at 32 frames the Intermediate convs see {rows32}, not one row")
    t32 = rmvpe_t32_model(full)
    removed = sum(int(np.prod(w.dims)) for _, w in rmvpe_intermediate_convs(full)) * 2 // 3
    out = {"source_sha256": RMVPE_SHA256, "intermediate_inputs_at_32": rows32, "intermediate_inputs_at_64": rows64,
           "params_removed": removed, "templates": {}}
    variants = [("t32", t32, RMVPE_T32_FRAMES), ("t32_fp16", rmvpe_fp16_model(t32), RMVPE_T32_FRAMES),
                ("fp16", rmvpe_fp16_model(full), None)]
    so = ort.SessionOptions()
    so.log_severity_level = 3
    ref = ort.InferenceSession(full_path, so, providers=["CPUExecutionProvider"])
    mels = rmvpe_parity_mels()
    for variant, model, frames in variants:
        dst = os.path.join(GRAPHS_DIR, f"rmvpe_{variant}.graph.onnx")
        tmp_graph, size, sha = save_weightless(model, dst, variant, RMVPE_SHA256, frames)
        try:
            row = {"file": os.path.relpath(dst, REPO).replace(os.sep, "/"), "bytes": size, "sha256": sha,
                   "weights_bytes": os.path.getsize(os.path.join(os.path.dirname(tmp_graph), RMVPE_WEIGHTS_FILE))}
            sess = ort.InferenceSession(tmp_graph, so, providers=["CPUExecutionProvider"])
            worst, worst_cents, flips, argmax, rows = 0.0, 0.0, 0, 0, 0
            per = []
            for kind, mel in mels:
                (a,) = ref.run(None, {"input": mel[None]})
                (b,) = sess.run(None, {"input": mel[None]})
                d, am, nv, c, v = salience_compare(a[0], b[0])
                worst, worst_cents, flips, argmax, rows = max(worst, d), max(worst_cents, c), flips + v, argmax + am, rows + nv
                per.append({"kind": kind, "max_abs": d, "argmax_diff_rows": am, "voiced_rows": nv, "max_cents": c, "vuv_flips": v})
            parity = {"mels": len(mels), "max_abs_salience_diff": worst, "argmax_diff_rows": argmax, "voiced_rows": rows,
                      "max_f0_cents": worst_cents, "vuv_flips": flips, "per_mel": per}
            if variant == "t32":
                parity["ok"] = worst <= 1e-5 and argmax == 0
            else:
                # ONNX Runtime's CPU EP has no fp16 Conv kernel on x86: it inserts Casts and computes in
                # fp32, so this row is the fp16 weight rounding alone (informational). The gate (fp16
                # arithmetic on DirectML / CUDA, the stream path) is rmvpe.rs's `rmvpe_fp16_gate`.
                parity["note"] = "CPU: fp16 weights, fp32 arithmetic (weight rounding only); the gate runs on the GPU"
            row["cpu_parity"] = parity
            print(f"{variant} vs full (CPU, {len(mels)} mels at 32 frames): max |d| {worst:.3g}, argmax rows differing {argmax}, "
                  f"max {worst_cents:.4f} cents, V/UV flips {flips}")
            del sess
        finally:
            shutil.rmtree(os.path.dirname(tmp_graph), ignore_errors=True)
        out["templates"][variant] = row
        print(f"wrote {row['file']} ({size} bytes; weights {row['weights_bytes']} bytes)")
    return out


def rmvpe_activation_census(full_path, seconds=4.0):
    """fp32 per-layer max |activation| (CPU) over speech windows, loud tones and random mels at 32
    and 64 frames: the fp16 overflow check (any layer above ~8,000 needs an fp32 island)."""
    import onnx
    import onnxruntime as ort

    m = onnx.load(full_path)
    watched = [o for n in m.graph.node if n.op_type in ("Conv", "ConvTranspose", "Relu", "Add", "BatchNormalization",
                                                        "AveragePool", "Concat", "Pad") for o in n.output]
    existing = {o.name for o in m.graph.output}
    for name in watched:
        if name not in existing:
            m.graph.output.append(onnx.helper.make_empty_tensor_value_info(name))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    so.intra_op_num_threads = 2
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    peak = {n: 0.0 for n in names}
    inputs = []
    for frames in (32, 64):
        for kind, mel in rmvpe_parity_mels(n_random=6, n_speech=12, seed=7 + frames, frames=frames):
            inputs.append((f"{kind}-{frames}", mel))
        for f0 in (55.0, 98.0, 220.0, 440.0, 880.0, 1500.0):
            for amp in (0.3, 1.0):
                tone = sa.harmonic_tone(f0, frames * 160 / 16000)[: (frames - 1) * 160] * (amp / 0.3)
                mel, t = rmvpe_mel(tone)
                assert t == frames, (t, frames)
                inputs.append((f"tone{f0:g}x{amp}-{frames}", mel[0, :, :frames]))
        noise = np.random.default_rng(frames).standard_normal((frames - 1) * 160).astype(np.float32)
        mel, t = rmvpe_mel(noise)
        assert t == frames, (t, frames)
        inputs.append((f"white-noise-{frames}", mel[0, :, :frames]))
    for label, mel in inputs:
        outs = sess.run(None, {"input": mel[None].astype(np.float32)})
        for n, o in zip(names, outs):
            if o.dtype.kind == "f":
                peak[n] = max(peak[n], float(np.abs(o).max()))
    top = sorted(peak.items(), key=lambda kv: -kv[1])
    over = [n for n, v in top if v > 8000.0]
    print(f"census: {len(inputs)} inputs, {len(names)} tensors; top: " + ", ".join(f"{n} {v:.0f}" for n, v in top[:5]))
    w_max = max(float(np.abs(onnx.numpy_helper.to_array(i)).max()) for i in m.graph.initializer)
    return {"inputs": len(inputs), "tensors": len(names), "over_8000": over, "max_weight_abs": w_max,
            "top": [{"tensor": n, "max_abs": v} for n, v in top[:25]], "ok": not over}


# ---------------------------------------------------------------------------------------------
# FCPE adapter
# ---------------------------------------------------------------------------------------------


def build_fcpe_adapter(src, dst):
    """The community FCPE export with Sigil's contract: audio [1, N] -> f0 [1, T], NaN -> 0."""
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(src)
    g = m.graph
    if len(g.input) != 1 or len(g.output) != 1 or g.input[0].name != "audio" or g.output[0].name != "f0_hz":
        raise SystemExit("unexpected FCPE export I/O")
    inner_in, inner_out = "audio_n1", "f0_hz"
    for node in g.node:
        for i, name in enumerate(node.input):
            if name == "audio":
                node.input[i] = inner_in
    axes = helper.make_tensor("sigil_axis2", TensorProto.INT64, [1], [2])
    zero = helper.make_tensor("sigil_zero", TensorProto.FLOAT, [], [0.0])
    g.initializer.extend([axes, zero])
    pre = helper.make_node("Unsqueeze", ["audio", "sigil_axis2"], [inner_in], name="sigil_in_unsqueeze")
    post = [
        helper.make_node("Squeeze", [inner_out, "sigil_axis2"], ["sigil_f0_raw"], name="sigil_out_squeeze"),
        helper.make_node("IsNaN", ["sigil_f0_raw"], ["sigil_f0_nan"], name="sigil_nan"),
        helper.make_node("Where", ["sigil_f0_nan", "sigil_zero", "sigil_f0_raw"], ["f0"], name="sigil_nan_to_unvoiced"),
    ]
    nodes = [pre] + list(g.node) + post
    del g.node[:]
    g.node.extend(nodes)
    del g.input[:]
    g.input.append(helper.make_tensor_value_info("audio", TensorProto.FLOAT, [1, "n_samples"]))
    del g.output[:]
    g.output.append(helper.make_tensor_value_info("f0", TensorProto.FLOAT, [1, "n_frames"]))
    m.producer_name = "sigil-vc-assets"
    meta = {p.key: p.value for p in m.metadata_props}
    meta.update({"sigil.format": "sigil-pitch-fcpe", "sigil.io": "audio [1,N] f32 16 kHz -> f0 [1, N//160+1] Hz (0 = unvoiced)",
                 "sigil.source": "gzivdo/fcpe-onnx@5800a2b1 fcpe.onnx (torchfcpe 0.0.4 weights), reshaped I/O + NaN -> 0"})
    del m.metadata_props[:]
    for k, v in sorted(meta.items()):
        p = m.metadata_props.add()
        p.key, p.value = k, v
    onnx.checker.check_model(m)
    onnx.save_model(m, dst)


def check_fcpe(upstream_path, adapter_path, ep):
    up = sa.session(upstream_path, ep)
    ad = sa.session(adapter_path, ep)
    rows = []
    worst = 0.0
    signals = [(f"tone {f0:g}", sa.harmonic_tone(f0)) for f0 in TONES]
    sp = sa.speech("jfk.wav", 4)
    if sp is not None:
        signals.append(("jfk.wav[:4 s]", sp))
    signals.append(("silence", np.zeros(16000, np.float32)))
    for label, audio in signals:
        (want,) = up.run(None, {"audio": audio[None, :, None].astype(np.float32)})
        want = np.nan_to_num(want[0, :, 0], nan=0.0)
        (got,) = ad.run(None, {"audio": audio[None].astype(np.float32)})
        got = got[0]
        if got.shape[0] != len(audio) // 160 + 1:
            raise SystemExit(f"FCPE adapter: {got.shape[0]} frames for {len(audio)} samples")
        worst = max(worst, float(np.abs(got - want).max()))
        row = {"signal": label, "frames": int(got.size), "voiced": int((got > 0).sum())}
        if label.startswith("tone"):
            f0 = float(label.split()[1])
            v = got > 0
            row["median_cents"] = float(np.median(np.abs(sa.cents(got[v], f0)))) if v.any() else float("inf")
            row["ok"] = row["median_cents"] <= CENTS_GATE and v.mean() > 0.8
        elif label == "silence":
            row["ok"] = bool((got == 0).mean() > 0.95)
        rows.append(row)
    return {"rows": rows, "adapter_vs_upstream_max_abs": worst, "ok": worst == 0.0 and all(r.get("ok", True) for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", default="cpu", choices=("cpu", "dml"))
    ap.add_argument("--stage", default=sa.STAGE)
    ap.add_argument("--rmvpe-templates", action="store_true",
                    help="write RMVPE's t32 / t32_fp16 / fp16 weightless graph templates and their CPU parity report")
    ap.add_argument("--rmvpe-census", action="store_true",
                    help="record fp32 RMVPE's per-layer max |activation| (the fp16 overflow check)")
    args = ap.parse_args()
    if args.rmvpe_templates or args.rmvpe_census:
        rm = sa.upstream("rmvpe/rmvpe.onnx", RMVPE_SHA256)
        ok = True
        if args.rmvpe_templates:
            rep = build_rmvpe_templates(rm)
            sa.report("pitch_rmvpe_templates", rep)
            ok &= rep["templates"]["t32"]["cpu_parity"]["ok"]
        if args.rmvpe_census:
            rep = rmvpe_activation_census(rm)
            sa.report("pitch_rmvpe_census", rep)
            ok &= rep["ok"]
        raise SystemExit(0 if ok else 1)
    out = {"ep": args.ep}
    sw = sa.upstream(PINS["swiftf0"][1], PINS["swiftf0"][2])
    out["swiftf0"] = check_swiftf0(sw, args.ep)
    out["swiftf0"]["contract"] = sa.io_contract(sw)
    out["swiftf0"]["file"] = dict(zip(("path", "bytes", "sha256"), sa.stage(PINS["swiftf0"][0], "model.onnx", sw, args.stage)))
    cr = sa.upstream(PINS["crepe-tiny"][1], PINS["crepe-tiny"][2])
    out["crepe-tiny"] = check_crepe(cr, args.ep)
    out["crepe-tiny"]["contract"] = sa.io_contract(cr)
    out["crepe-tiny"]["file"] = dict(zip(("path", "bytes", "sha256"), sa.stage(PINS["crepe-tiny"][0], "model.onnx", cr, args.stage)))
    fc = sa.upstream(PINS["fcpe"][1], PINS["fcpe"][2])
    dst = os.path.join(args.stage, "vc", "pitch-fcpe", "model.onnx")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if args.ep == "cpu":
        build_fcpe_adapter(fc, dst)
    out["fcpe"] = check_fcpe(fc, dst, args.ep)
    out["fcpe"]["contract"] = sa.io_contract(dst)
    out["fcpe"]["file"] = dict(zip(("path", "bytes", "sha256"), sa.stage("vc/pitch-fcpe", "model.onnx", dst, args.stage)))
    rm = os.path.join(sa.UPSTREAM, "rmvpe", "rmvpe.onnx")
    if os.path.isfile(rm):  # 362 MB: checked when fetched, pinned either way
        sa.upstream("rmvpe/rmvpe.onnx", RMVPE_SHA256)
        out["rmvpe"] = check_rmvpe(rm, args.ep)
        out["rmvpe"]["contract"] = sa.io_contract(rm)
    for k in [k for k in ("swiftf0", "crepe-tiny", "fcpe", "rmvpe") if k in out]:
        print(f"{'PASS' if out[k]['ok'] else 'FAIL'}  {k}: " + ", ".join(
            f"{r.get('tone_hz', r.get('signal'))}: {r.get('median_cents', float('nan')):.1f} c" for r in out[k]["rows"] if "median_cents" in r))
    print(f"SwiftF0 hop {out['swiftf0']['hop']} samples; frames {out['swiftf0']['frames_by_samples']}")
    print(f"FCPE adapter == upstream: max |d| {out['fcpe']['adapter_vs_upstream_max_abs']}")
    sa.report("pitch" if args.ep == "cpu" else f"pitch_{args.ep}", out)
    raise SystemExit(0 if all(out[k]["ok"] for k in ("swiftf0", "crepe-tiny", "fcpe", "rmvpe") if k in out) else 1)


if __name__ == "__main__":
    main()
