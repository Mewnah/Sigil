#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Benchmark-only assets (kind ``bench``; never needed by normal users): stage, build and check them.

| id | files | source |
| --- | --- | --- |
| bench/dnsmos-p835 | model.onnx (DNSMOS P.835 ``sig_bak_ovr.onnx``), p808.onnx (``model_v8.onnx``) | microsoft/DNS-Challenge@591184a9 DNSMOS/DNSMOS (MIT) |
| bench/campplus | model.onnx (CAM++ speaker embedding, 3D-Speaker zh-cn common, 192-d) | k2-fsa/sherpa-onnx release ``speaker-recongition-models`` (Apache-2.0) |
| bench/campplus-exact | model.onnx (the same CAM++ checkpoint re-exported with exact segment pooling; P3B-B) | built here from ``campplus_cn_common.bin`` (3D-Speaker zh-cn common, Apache-2.0; the seed-vc archive's copy) |
| bench/gpu-load | model.onnx (Sigil's GPU hammer: 8 x Conv 3x3 32 -> 32 + ReLU) | built here (deterministic) |

Why ``bench/campplus-exact`` (P3B-B, ADR 0009): the sherpa-onnx export keeps CAM++'s 52
``AveragePool(ceil_mode=1)`` segment poolings, and ONNX Runtime divides the last, partial 100-frame
segment by the full kernel where PyTorch divides by its valid frames, so the graph computes another
function than upstream on any length that is not a whole number of segments (ADR 0006: 4.4 dB SNR,
cosine 0.925). The re-export runs the upstream ``CAMPPlus`` with Seed-VC's ``_exact_seg_pooling``
(segment means as a one-hot matmul; imported read-only from ``scripts/vc-families/seed_vc/export.py``)
and must match upstream PyTorch at >= 100 dB SNR on real fbanks of any length. Its I/O contract is
``bench/campplus``'s, so the benchmark runs either file; ``bench/campplus`` stays selectable as the
legacy scorer for one release.

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
keeps activations finite. ``campplus-exact``: >= 100 dB SNR against upstream PyTorch on the fbanks
of real clips at lengths that do and do not fill whole segments; the legacy file checked against
the same upstream at whole-segment lengths (where AveragePool is exact) tells whether it carries the
same weights; identical input embeds bit-identically twice; a matched-length calibration (two
different ~3.8 s utterances of one speaker) next to the half-clip one. Report: reports/bench.json.

Usage (repo root): ``python scripts/vc-templates/export_bench.py [--only campplus-exact]``. Run it
with the ONNX Runtime the engine uses (``sigil-dev-cache/venv-ort128-cuda``, CPU EP) for the SNR gate.
"""

import argparse
import copy
import importlib.util
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sigil_assets as sa  # noqa: E402

SEED_VC_EXPORT = os.path.join(os.path.dirname(HERE), "vc-families", "seed_vc", "export.py")
CAMPPLUS_EXACT_ID = "bench/campplus-exact"
CAMPPLUS_EXACT_GATE_DB = 100.0

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


# ---------------------------------------------------------------------------------------------
# bench/campplus-exact (P3B-B)
# ---------------------------------------------------------------------------------------------

def _seed_vc_export():
    """``scripts/vc-families/seed_vc/export.py`` as a module (read-only: its upstream CAM++ code,
    checkpoint path and exact segment pooling). Importing it only registers its stub modules."""
    spec = importlib.util.spec_from_file_location("sigil_seed_vc_export", SEED_VC_EXPORT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def upstream_campplus():
    """(upstream CAMPPlus as 3D-Speaker / Seed-VC load it (fp32, eval, PyTorch's own pooling), the
    seed_vc export module)."""
    sv = _seed_vc_export()
    from modules.campplus.DTDNN import CAMPPlus  # on sys.path through the seed_vc export module

    if not os.path.isfile(sv.CAMPPLUS_CKPT):
        raise SystemExit(f"missing {sv.CAMPPLUS_CKPT}: the seed-vc repository archive's campplus_cn_common.bin")
    m = CAMPPlus(feat_dim=80, embedding_size=192)
    m.load_state_dict(torch.load(sv.CAMPPLUS_CKPT, map_location="cpu"))
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, sv


class _BenchCampPlus(torch.nn.Module):
    """``x`` [1, T, 80] (mean-normalised Kaldi fbank) -> ``embedding`` [1, 192]: the bench contract."""

    def __init__(self, exact):
        super().__init__()
        self.exact = exact

    def forward(self, x):
        return self.exact(x)


def export_campplus_exact(dst):
    """Export the upstream CAM++ with exact segment pooling (opset 17, dynamic T). Returns the
    checkpoint's sha256."""
    import onnx

    up, sv = upstream_campplus()
    model = _BenchCampPlus(sv.CampPlus(copy.deepcopy(up))).eval()
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(model, (torch.randn(1, 300, 80),), dst, input_names=["x"], output_names=["embedding"],
                          dynamic_axes={"x": {1: "T"}}, opset_version=17, do_constant_folding=True)
    m = onnx.load(dst)
    onnx.checker.check_model(m)
    del m.metadata_props[:]
    ckpt_sha = sa.sha256_file(sv.CAMPPLUS_CKPT)
    for k, v in (("sigil.source", "3D-Speaker campplus_sv_zh-cn_16k-common (campplus_cn_common.bin, sha256 " + ckpt_sha + ")"),
                 ("sigil.graph", "bench/campplus-exact: upstream CAMPPlus, segment pooling as exact segment means (P3B-B)")):
        m.metadata_props.append(onnx.StringStringEntryProto(key=k, value=v))
    onnx.save(m, dst)
    return ckpt_sha


def _fbank(audio):
    import torchaudio

    f = torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(audio, dtype=np.float32))[None] * 32768.0,
                                          num_mel_bins=80, sample_frequency=16000, dither=0.0)
    return (f - f.mean(0, keepdim=True))[None]


def _snr_db(ref, x):
    ref = np.asarray(ref, np.float64).ravel()
    err = np.asarray(x, np.float64).ravel() - ref
    e = float(np.sum(err * err))
    return float("inf") if e == 0.0 else float(10.0 * np.log10(np.sum(ref * ref) / e))


def _unit(e):
    e = np.asarray(e, np.float64).ravel()
    return e / np.linalg.norm(e)


def _initializer_overlap(a_path, b_path):
    """How many float initializers of ``a`` occur bit-identically in ``b`` (informational: an export
    may fold batch norms differently, so the functional check below is the evidence)."""
    import hashlib

    import onnx
    from onnx import numpy_helper

    def digests(path):
        m = onnx.load(path)
        out = {}
        for t in m.graph.initializer:
            arr = numpy_helper.to_array(t)
            if arr.dtype.kind == "f" and arr.size > 1:
                out[hashlib.sha256(np.ascontiguousarray(arr, np.float32).tobytes()).hexdigest()] = arr.size
        return out

    a, b = digests(a_path), digests(b_path)
    shared = set(a) & set(b)
    return {"float_initializers": len(a), "bit_identical_in_other": len(shared),
            "values": int(sum(a.values())), "values_bit_identical": int(sum(a[k] for k in shared))}


def check_campplus_exact(path, legacy_path, ep):
    """SNR of the exact export and of the legacy file against upstream PyTorch on real fbanks, at
    whole-segment lengths (multiples of 200 fbank frames: the 2x time stride makes the segments
    whole) and at every other length; determinism; calibration cosines."""
    up, _ = upstream_campplus()
    exact = sa.session(path, ep)
    legacy = sa.session(legacy_path, ep) if legacy_path and os.path.isfile(legacy_path) else None
    jfk = sa.speech("jfk.wav")
    ja_long = sa.speech("ja-long.wav")
    others = {n: sa.speech(n) for n in ("ja-short.wav", "ko-short.wav", "ru-short.wav")}
    if jfk is None or ja_long is None or any(o is None for o in others.values()):
        raise SystemExit("CAM++ exact check needs the jfk / ja-long / ja-short / ko-short / ru-short samples in the dev cache")
    sr = 16000
    # Clips of real speech at whole-segment and ragged lengths (fbank frames = (n - 400) // 160 + 1).
    cuts = []
    for name, clip in (("jfk", jfk), ("ja-long", ja_long)):
        for frames in (400, 600, 1000, 237, 381, 512, 777, 1043):
            n = (frames - 1) * 160 + 400
            if n <= clip.size:
                cuts.append((f"{name}[0:{frames}f]", clip[:n], frames))
    rows = []
    for label, audio, frames in cuts:
        x = _fbank(audio)
        assert x.shape[1] == frames, (label, x.shape)
        with torch.no_grad():
            ref = up(x).numpy()
        e = exact.run(None, {"x": x.numpy()})[0]
        row = {"clip": label, "frames": frames, "whole_segments": frames % 200 == 0, "exact_snr_db": _snr_db(ref, e),
               "exact_cosine_to_upstream": float(_unit(ref) @ _unit(e))}
        if legacy is not None:
            g = legacy.run(None, {"x": x.numpy()})[0]
            row["legacy_snr_db"] = _snr_db(ref, g)
            row["legacy_cosine_to_upstream"] = float(_unit(ref) @ _unit(g))
        rows.append(row)
    exact_min = min(r["exact_snr_db"] for r in rows)
    whole = [r for r in rows if r["whole_segments"] and "legacy_snr_db" in r]
    ragged = [r for r in rows if not r["whole_segments"] and "legacy_snr_db" in r]
    # Determinism: the same input twice is bit-identical.
    x = _fbank(jfk[: 3 * sr])
    e1 = exact.run(None, {"x": x.numpy()})[0]
    e2 = exact.run(None, {"x": x.numpy()})[0]

    def emb(sess, audio):
        return _unit(sess.run(None, {"x": _fbank(audio).numpy()})[0])

    def calib(sess):
        half = jfk.size // 2
        # Matched length: two different ~3.8 s utterances of one speaker (different words), against
        # the half-clip pair the P3 calibration used (~5.5 s each here) and other speakers.
        n = int(3.8 * sr)
        jfk_a, jfk_b = jfk[int(0.3 * sr): int(0.3 * sr) + n], jfk[int(6.0 * sr): int(6.0 * sr) + n]
        ja_a, ja_b = ja_long[int(2.0 * sr): int(2.0 * sr) + n], ja_long[int(15.0 * sr): int(15.0 * sr) + n]
        out = {
            "same_speaker_halves": float(emb(sess, jfk[:half]) @ emb(sess, jfk[half:])),
            "same_speaker_matched_3_8s_en": float(emb(sess, jfk_a) @ emb(sess, jfk_b)),
            "same_speaker_matched_3_8s_ja": float(emb(sess, ja_a) @ emb(sess, ja_b)),
            "different_speakers": {k: float(emb(sess, jfk_a) @ emb(sess, v[:n])) for k, v in others.items()},
        }
        out["margin"] = min(out["same_speaker_matched_3_8s_en"], out["same_speaker_matched_3_8s_ja"]) - max(out["different_speakers"].values())
        return out

    out = {
        "rows": rows,
        "exact_min_snr_db": exact_min,
        "gate_db": CAMPPLUS_EXACT_GATE_DB,
        "deterministic": bool(np.array_equal(e1, e2)),
        "calibration": {"exact": calib(exact)},
    }
    if legacy is not None:
        out["legacy_min_snr_db_whole_segments"] = min(r["legacy_snr_db"] for r in whole) if whole else None
        out["legacy_min_snr_db_ragged"] = min(r["legacy_snr_db"] for r in ragged) if ragged else None
        out["legacy_mean_snr_db_ragged"] = float(np.mean([r["legacy_snr_db"] for r in ragged])) if ragged else None
        # Same checkpoint = the legacy file is exact wherever its AveragePool is (whole segments).
        out["legacy_same_checkpoint"] = bool(whole and out["legacy_min_snr_db_whole_segments"] >= 60.0)
        out["initializers"] = _initializer_overlap(path, legacy_path)
        out["calibration"]["legacy"] = calib(legacy)
    out["ok"] = bool(exact_min >= CAMPPLUS_EXACT_GATE_DB and out["deterministic"] and out["calibration"]["exact"]["margin"] > 0.2)
    return out


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


def run_campplus_exact(args):
    """Export, check and stage ``bench/campplus-exact``; returns its report section."""
    dst = os.path.join(sa.DEV_CACHE, "vc-assets-build", "campplus-exact", "model.onnx")
    if args.ep == "cpu":
        ckpt_sha = export_campplus_exact(dst)
    else:
        ckpt_sha = sa.sha256_file(_seed_vc_export().CAMPPLUS_CKPT)
    legacy = sa.upstream(PINS["campplus"][1], PINS["campplus"][2])
    import onnxruntime

    sec = check_campplus_exact(dst, legacy, args.ep)
    sec["onnxruntime"] = onnxruntime.__version__
    sec["checkpoint_sha256"] = ckpt_sha
    sec["contract"] = sa.io_contract(dst)
    sec["file"] = dict(zip(("path", "bytes", "sha256"), sa.stage(CAMPPLUS_EXACT_ID, "model.onnx", dst, args.stage)))
    return sec


def print_campplus_exact(ce):
    c = ce["calibration"]["exact"]
    print(f"{'PASS' if ce['ok'] else 'FAIL'}  campplus-exact: min {ce['exact_min_snr_db']:.1f} dB SNR vs upstream (gate {ce['gate_db']:.0f}), "
          f"deterministic {ce['deterministic']}, matched 3.8 s same speaker en {c['same_speaker_matched_3_8s_en']:.3f} "
          f"ja {c['same_speaker_matched_3_8s_ja']:.3f}, halves {c['same_speaker_halves']:.3f}, sha256 {ce['file']['sha256'][:16]}")
    if "legacy_same_checkpoint" in ce:
        print(f"      legacy bench/campplus: whole segments {ce['legacy_min_snr_db_whole_segments']:.1f} dB, ragged "
              f"{ce['legacy_mean_snr_db_ragged']:.1f} dB mean (min {ce['legacy_min_snr_db_ragged']:.1f}); same checkpoint: "
              f"{ce['legacy_same_checkpoint']}; initializers bit-identical {ce['initializers']['bit_identical_in_other']}/{ce['initializers']['float_initializers']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", default="cpu", choices=("cpu", "dml"))
    ap.add_argument("--stage", default=sa.STAGE)
    ap.add_argument("--only", default="", help="comma list of sections: dnsmos, campplus, campplus-exact, gpu-load (default: all)")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only - {"dnsmos", "campplus", "campplus-exact", "gpu-load"}:
        raise SystemExit(f"--only: unknown sections {sorted(only - {'dnsmos', 'campplus', 'campplus-exact', 'gpu-load'})}")
    if only == {"campplus-exact"}:
        # Merge into the existing report (the other sections keep their last results).
        name = "bench" if args.ep == "cpu" else f"bench_{args.ep}"
        report_path = os.path.join(sa.REPORTS, f"{name}.json")
        import json

        out = json.load(open(report_path, encoding="utf-8")) if os.path.isfile(report_path) else {"ep": args.ep}
        out["campplus-exact"] = run_campplus_exact(args)
        print_campplus_exact(out["campplus-exact"])
        sa.report(name, out)
        raise SystemExit(0 if out["campplus-exact"]["ok"] else 1)
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
    out["campplus-exact"] = run_campplus_exact(args)
    dn = out["dnsmos"]
    print(f"{'PASS' if dn['ok'] else 'FAIL'}  dnsmos: OVRL clean {dn['clean']['ovr']:.2f} vs 0 dB noisy {dn['noisy_0db']['ovr']:.2f}")
    cp = out["campplus"]
    print(f"{'PASS' if cp['ok'] else 'FAIL'}  campplus: same speaker {cp['same_speaker_cosine']:.3f}, others {[round(x, 3) for x in cp['different_speaker_cosine']]}")
    print(f"{'PASS' if out['gpu-load']['ok'] else 'FAIL'}  gpu-load: {out['gpu-load']['file']['bytes']} bytes, sha256 {out['gpu-load']['file']['sha256'][:16]}")
    print_campplus_exact(out["campplus-exact"])
    sa.report("bench" if args.ep == "cpu" else f"bench_{args.ep}", out)
    raise SystemExit(0 if all(out[k]["ok"] for k in ("dnsmos", "campplus", "gpu-load", "campplus-exact")) else 1)


if __name__ == "__main__":
    main()
