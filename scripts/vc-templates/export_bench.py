#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark-only assets (kind ``bench``; never needed by normal users): stage, build and check them.

| id | files | source |
| --- | --- | --- |
| bench/dnsmos-p835 | model.onnx (DNSMOS P.835 ``sig_bak_ovr.onnx``), p808.onnx (``model_v8.onnx``) | microsoft/DNS-Challenge@591184a9 DNSMOS/DNSMOS (MIT) |
| bench/campplus | model.onnx (CAM++ speaker embedding, 3D-Speaker zh-cn common, 192-d) | k2-fsa/sherpa-onnx release ``speaker-recongition-models`` (Apache-2.0) |
| bench/gpu-load | model.onnx (Sigil's GPU hammer: 8 x Conv 3x3 32 -> 32 + ReLU) | built here (deterministic) |

Contracts (what P3-F's ``bench/quality.rs`` and ``contention.rs`` feed):
- **DNSMOS P.835**: ``input_1`` f32 [N, 144160] (9.01 s of 16 kHz audio; longer clips are scored in
  9.01 s windows hopped by 1 s and averaged) -> ``Identity:0`` [N, 3] raw SIG, BAK, OVRL, mapped with
  ``DNSMOS_POLY`` (DNS-Challenge ``dnsmos_local.py``, non-personalised). ``p808.onnx``: ``input_1``
  [N, 900, 120] = (librosa power_to_db(mel(n_fft 321, hop 160, 120 mels), ref=max) + 40) / 40 of the
  same window minus its last 160 samples -> P.808 MOS.
- **CAM++**: ``x`` f32 [N, T, 80] = Kaldi fbank (80 bins, 25 ms / 10 ms, 16 kHz, no dither) minus its
  mean over time -> ``embedding`` [N, 192]; speaker similarity is the cosine of two embeddings.
- **gpu-load**: ``x`` f32 [1, 32, H, W] -> ``y`` (same shape); the contention mode sizes H x W for
  its duty cycle (256 x 256 is ~4.8 GMAC per run).

Checks: DNSMOS scores clean speech above the same speech at 0 dB SNR (OVRL by >= 0.5); CAM++ gives
two halves of one speaker a higher cosine than two different speakers (by >= 0.2); gpu-load runs and
keeps activations finite. Report: reports/bench.json.

Usage (repo root): ``python scripts/vc-templates/export_bench.py``.
"""

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigil_assets as sa  # noqa: E402

PINS = {
    "dnsmos": ("bench/dnsmos-p835", "dnsmos/sig_bak_ovr.onnx", "269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd"),
    "dnsmos-p808": ("bench/dnsmos-p835", "dnsmos/model_v8.onnx", "9246480c58567bc6affd4200938e77eef49468c8bc7ed3776d109c07456f6e91"),
    "campplus": ("bench/campplus", "campplus/3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx", "f682b514c05d947ee3fa91cd6ec6c5c7543479a128373fa29b1faedccd21fd11"),
}
# DNS-Challenge dnsmos_local.py get_polyfit_val (is_personalized_MOS = False)
DNSMOS_POLY = {
    "ovr": [-0.06766283, 1.11546468, 0.04602535],
    "sig": [-0.08397278, 1.22083953, 0.0052439],
    "bak": [-0.13166888, 1.60915514, -0.39604546],
}
DNSMOS_LEN = 144160
GPU_LOAD_LAYERS, GPU_LOAD_CH = 8, 32


def dnsmos(sess, audio):
    x = audio[:DNSMOS_LEN]
    if x.size < DNSMOS_LEN:
        x = np.pad(x, (0, DNSMOS_LEN - x.size), mode="wrap")
    (raw,) = sess.run(None, {"input_1": x[None].astype(np.float32)})
    sig, bak, ovr = raw[0]
    return {k: float(np.poly1d(DNSMOS_POLY[k])(v)) for k, v in (("sig", sig), ("bak", bak), ("ovr", ovr))}


def check_dnsmos(path, ep):
    sess = sa.session(path, ep)
    clean = sa.speech("jfk.wav")
    if clean is None:
        raise SystemExit("DNSMOS check needs the jfk.wav sample in the dev cache")
    noise = np.random.default_rng(4).standard_normal(clean.size).astype(np.float32)
    noise *= np.sqrt(np.mean(clean**2) / np.mean(noise**2))
    a, b = dnsmos(sess, clean), dnsmos(sess, clean + noise)
    return {"clean": a, "noisy_0db": b, "ok": bool(a["ovr"] - b["ovr"] >= 0.5 and 1.0 <= a["ovr"] <= 5.0)}


def campplus_embed(sess, audio):
    import torchaudio

    feats = torchaudio.compliance.kaldi.fbank(torch.from_numpy(audio)[None] * 32768.0, num_mel_bins=80, sample_frequency=16000, dither=0.0)
    feats = feats - feats.mean(0, keepdim=True)
    (emb,) = sess.run(None, {"x": feats[None].numpy()})
    e = emb[0]
    return e / np.linalg.norm(e)


def check_campplus(path, ep):
    sess = sa.session(path, ep)
    jfk = sa.speech("jfk.wav")
    others = [sa.speech(n) for n in ("ja-short.wav", "ko-short.wav")]
    if jfk is None or any(o is None for o in others):
        raise SystemExit("CAM++ check needs the jfk / ja-short / ko-short samples in the dev cache")
    a, b = campplus_embed(sess, jfk[: jfk.size // 2]), campplus_embed(sess, jfk[jfk.size // 2 :])
    same = float(a @ b)
    diff = [float(a @ campplus_embed(sess, o)) for o in others]
    return {"same_speaker_cosine": same, "different_speaker_cosine": diff, "ok": bool(same - max(diff) >= 0.2)}


def build_gpu_load(dst):
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(20260929)
    nodes, inits = [], []
    x = "x"
    for i in range(GPU_LOAD_LAYERS):
        w = (rng.standard_normal((GPU_LOAD_CH, GPU_LOAD_CH, 3, 3)) * np.sqrt(2.0 / (GPU_LOAD_CH * 9))).astype(np.float32)
        b = np.zeros(GPU_LOAD_CH, np.float32)
        inits += [numpy_helper.from_array(w, f"w{i}"), numpy_helper.from_array(b, f"b{i}")]
        conv = f"c{i}"
        out = "y" if i == GPU_LOAD_LAYERS - 1 else f"h{i}"
        nodes.append(helper.make_node("Conv", [x, f"w{i}", f"b{i}"], [conv], name=f"conv{i}", pads=[1, 1, 1, 1], kernel_shape=[3, 3]))
        nodes.append(helper.make_node("Relu", [conv], [out], name=f"relu{i}"))
        x = out
    graph = helper.make_graph(
        nodes, "sigil_bench_gpu_load",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, GPU_LOAD_CH, "H", "W"])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, GPU_LOAD_CH, "H", "W"])],
        inits,
    )
    model = helper.make_model(graph, producer_name="sigil-bench", opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    model.doc_string = "Sigil benchmark GPU load: 8 x (Conv 3x3 32->32 + ReLU), He-initialised, seed 20260929"
    onnx.checker.check_model(model)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    onnx.save_model(model, dst)


def check_gpu_load(path, ep):
    sess = sa.session(path, ep)
    x = np.random.default_rng(1).standard_normal((1, GPU_LOAD_CH, 64, 64)).astype(np.float32)
    (y,) = sess.run(None, {"x": x})
    return {"out_rms": float(np.sqrt(np.mean(y**2))), "ok": bool(np.isfinite(y).all() and y.shape == x.shape and np.abs(y).max() > 0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", default="cpu", choices=("cpu", "dml"))
    ap.add_argument("--stage", default=sa.STAGE)
    args = ap.parse_args()
    out = {"ep": args.ep}
    d = sa.upstream(PINS["dnsmos"][1], PINS["dnsmos"][2])
    p808 = sa.upstream(PINS["dnsmos-p808"][1], PINS["dnsmos-p808"][2])
    out["dnsmos"] = check_dnsmos(d, args.ep)
    out["dnsmos"]["contract"] = {"model.onnx": sa.io_contract(d), "p808.onnx": sa.io_contract(p808)}
    out["dnsmos"]["files"] = {
        "model.onnx": dict(zip(("path", "bytes", "sha256"), sa.stage(PINS["dnsmos"][0], "model.onnx", d, args.stage))),
        "p808.onnx": dict(zip(("path", "bytes", "sha256"), sa.stage(PINS["dnsmos"][0], "p808.onnx", p808, args.stage))),
    }
    c = sa.upstream(PINS["campplus"][1], PINS["campplus"][2])
    out["campplus"] = check_campplus(c, args.ep)
    out["campplus"]["contract"] = sa.io_contract(c)
    out["campplus"]["file"] = dict(zip(("path", "bytes", "sha256"), sa.stage(PINS["campplus"][0], "model.onnx", c, args.stage)))
    g = os.path.join(args.stage, "bench", "gpu-load", "model.onnx")
    if args.ep == "cpu":
        build_gpu_load(g)
    out["gpu-load"] = check_gpu_load(g, args.ep)
    out["gpu-load"]["contract"] = sa.io_contract(g)
    out["gpu-load"]["file"] = dict(zip(("path", "bytes", "sha256"), sa.stage("bench/gpu-load", "model.onnx", g, args.stage)))
    dn = out["dnsmos"]
    print(f"{'PASS' if dn['ok'] else 'FAIL'}  dnsmos: OVRL clean {dn['clean']['ovr']:.2f} vs 0 dB noisy {dn['noisy_0db']['ovr']:.2f}")
    cp = out["campplus"]
    print(f"{'PASS' if cp['ok'] else 'FAIL'}  campplus: same speaker {cp['same_speaker_cosine']:.3f}, others {[round(x, 3) for x in cp['different_speaker_cosine']]}")
    print(f"{'PASS' if out['gpu-load']['ok'] else 'FAIL'}  gpu-load: {out['gpu-load']['file']['bytes']} bytes, sha256 {out['gpu-load']['file']['sha256'][:16]}")
    sa.report("bench" if args.ep == "cpu" else f"bench_{args.ep}", out)
    raise SystemExit(0 if all(out[k]["ok"] for k in ("dnsmos", "campplus", "gpu-load")) else 1)


if __name__ == "__main__":
    main()
