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
"""

import argparse
import os
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
    args = ap.parse_args()
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
