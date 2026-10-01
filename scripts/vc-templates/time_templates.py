#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Indicative per-call cost of the templates at real-time shapes (not the benchmark).

For each template: ``enc`` over a 1 s context (T = 100 frames) and ``flowdec`` returning R frames
(default 4 = 40 ms; F = flow_margin + dec_margin + R, the latents over F - latent_offset), the
recommended variant per device class, 3 warm-up calls then the median of 20, host tensors every call
(no IOBinding, no CUDA graphs, Python ONNX Runtime 1.19). The engine's numbers come from
``Sigil.exe --headless bench`` (P3-F); these only compare the vocoders and the fp16 mixes. Writes
reports/template_timing.json.

Usage (repo root; DirectML in the onnxruntime-directml environment):
    python scripts/vc-templates/time_templates.py --ep cpu --template v2-40k-f0-hifigan
    .venv-dml/Scripts/python scripts/vc-templates/time_templates.py --ep dml
"""

import argparse
import json
import os
import statistics
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import parity_test as pt  # noqa: E402
import sigil_layout as sl  # noqa: E402
from rvc_arch import configs  # noqa: E402


def time_calls(fn, warm=3, runs=20):
    for _ in range(warm):
        fn()
    t = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        t.append((time.perf_counter() - t0) * 1000)
    return statistics.median(t)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", default="cpu", choices=sorted(pt.EPS))
    ap.add_argument("--template", action="append", choices=configs.MATRIX)
    ap.add_argument("--templates", default=pt.DEFAULT_TEMPLATES_DIR)
    ap.add_argument("--return-frames", type=int, default=4)
    args = ap.parse_args()
    pt.EP_PROVIDERS = pt.EPS[args.ep]
    cls = "cpu" if args.ep == "cpu" else "gpu"
    ids = args.template or ["v2-40k-f0-hifigan", "v2-48k-f0-hifigan", "v2-40k-nof0-hifigan", "v2-40k-f0-mrf-hifigan", "v2-32k-f0-refinegan", "v2-48k-f0-refinegan"]
    rows = {}
    rng = np.random.default_rng(0)
    for tid in ids:
        spec = configs.TEMPLATES[tid]
        tdir = os.path.join(args.templates, tid)
        layout = json.load(open(os.path.join(tdir, "layout.json"), encoding="utf-8"))
        w32 = bytes(layout["weights"]["bytes_f32"])  # zero weights: timing only
        w16 = sl.f16_weights(w32)
        st, arch = layout["streaming"], layout["arch"]
        upp, c, gin, emb = arch["upp"], arch["inter_channels"], arch["gin_channels"], arch["emb_channels"]
        pool = np.zeros(st["adain"]["pool"]["count"], np.float32) if "adain" in st else None
        rec = {g: layout["graphs"][g]["recommended"][cls] for g in ("enc", "flowdec")}
        enc, _k1 = pt.ort_session(tdir, layout, "enc", rec["enc"], w32, w16)
        fd, _k2 = pt.ort_session(tdir, layout, "flowdec", rec["flowdec"], w32, w16, pool)
        t = 100
        enc_feeds = {"phone": rng.standard_normal((1, t, emb)).astype(np.float32)}
        if spec["f0"]:
            enc_feeds["pitch"] = np.full((1, t), 60, np.int64)
        f = st["flow_margin"] + st["dec_margin"] + args.return_frames
        fz = f - st.get("latent_offset", 0)  # RefineGAN's latent window (m_p, logs_p, z_noise)
        feeds = {"m_p": np.zeros((1, c, fz), np.float32), "logs_p": np.zeros((1, c, fz), np.float32),
                 "z_noise": np.zeros((1, c, fz), np.float32), "g": np.zeros((1, gin, 1), np.float32)}
        names = [i.name for i in fd.get_inputs()]
        if "f0" in names:
            feeds["f0"] = np.full((1, f), 150.0, np.float32)
        if "phase0" in names:
            feeds["phase0"] = np.zeros((1, 1), np.float32)
        if "phase" in names:
            feeds["phase"] = np.zeros((1, f), np.float32)
        if "src_noise" in names:
            shape = [1, f * upp] + ([st["excitation"]["harmonics"]] if st["excitation"].get("harmonics", 1) > 1 else [])
            feeds["src_noise"] = np.zeros(shape, np.float32)
        if "noise_offsets" in names:
            feeds["noise_offsets"] = np.zeros((1, f), np.int64)
        e_ms = time_calls(lambda s=enc, x=enc_feeds: pt.run(s, x))
        d_ms = time_calls(lambda s=fd, x=feeds: pt.run(s, x))
        rows[tid] = {"ep": args.ep, "enc_variant": rec["enc"], "flowdec_variant": rec["flowdec"], "enc_ms_T100": round(e_ms, 2),
                     "flowdec_frames": f, "latent_frames": fz, "return_frames": args.return_frames, "flowdec_ms": round(d_ms, 2)}
        print(f"{tid:<24} {args.ep}: enc {rec['enc']} {e_ms:6.2f} ms (T=100) | flowdec {rec['flowdec']} {d_ms:6.2f} ms "
              f"(F={f}, Fz={fz}, R={args.return_frames})")
    path = os.path.join(HERE, "reports", "template_timing.json")
    data = json.load(open(path, encoding="utf-8")) if os.path.isfile(path) else {}
    data.setdefault(args.ep, {}).update(rows)
    sl.write_json(path, data)


if __name__ == "__main__":
    main()
