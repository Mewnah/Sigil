#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Export the whole RVC template matrix with its parity gate, and write the matrix index.

For every template of ``configs.MATRIX`` (v1/v2 x 24/32/40/48 kHz x f0/nof0 x HiFi-GAN / RefineGAN /
MRF HiFi-GAN, minus the cells no trainer produces, ``configs.UNSUPPORTED``):

1. export it (``export_rvc.export_template``) with the fp16 flowdec mix ``MIXES[0]`` (decoder stage 0
   in fp32, the ADR 0004 rule);
2. run the parity gate on the CPU (``parity_test.parity``, in process) and, with ``--dml-python``,
   on DirectML (the GPU device class; DirectML is the least accurate GPU EP for fp16, ADR 0004);
3. when the fp16 flowdec misses the GPU gate (with a ``GPU_MARGIN`` safety factor), re-export with
   the next mix (more stages, then the flow, in fp32) and measure again; a template whose fp16
   flowdec misses with every mix keeps the first mix, marked ``fail`` on GPUs (Sigil then runs its
   fp32 flowdec there);
4. write the measured parity and the resulting recommendation into its ``layout.json``, with the
   evidence: the weights, the worst rows per class and the no-lookahead seam (``right_edge``: how
   many frames before a mid-utterance window's end the CPU fp32 graph differs from the offline
   decode by more than 1e-4, parity_test.py's ``edge`` rows; T5 evidence, not a gate).

Then it writes ``matrix.json`` (the template index with the unsupported cells and their reasons,
and per template the streaming numbers an engine or autotuner sizes lookahead and crossfades from)
and the reports in ``scripts/vc-templates/reports/``, and checks the bundled size budget (30 MB).

Usage (repo root; the DirectML Python is a separate environment with onnxruntime-directml==1.19.2,
see README.md):

    python scripts/vc-templates/export_matrix.py --dml-python <venv>/Scripts/python.exe
    python scripts/vc-templates/export_matrix.py --template v2-32k-f0-refinegan --dml-python ...
    python scripts/vc-templates/export_matrix.py --index-only        # rewrite matrix.json only
"""

import argparse
import datetime
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)

import export_rvc  # noqa: E402
import parity_test  # noqa: E402
import sigil_layout as sl  # noqa: E402
from rvc_arch import configs  # noqa: E402

BUDGET_BYTES = 30 * 1024 * 1024
REPORTS = os.path.join(HERE, "reports")
# fp16 flowdec mixes, cheapest first (see export_rvc.FLOWDEC_FP16).
MIXES = [
    {"fp32_flow": False, "fp32_stages": (0,), "fp32_ends": False},
    {"fp32_flow": False, "fp32_stages": (0, 1), "fp32_ends": False},
    # RefineGAN on DirectML also needs its main path's ends (mel conv / cond, conv_post) in fp32; the
    # other vocoders ignore `fp32_ends`.
    {"fp32_flow": True, "fp32_stages": (0, 1), "fp32_ends": True},
]
# A GPU fp16 row must stay under this share of the 5e-3 gate on the measurement weights: most matrix
# cells are measured on donor weights, and the gate has to hold for real voices too.
GPU_MARGIN = 0.8
FP16_FLOWDEC_ROWS = ("flowdec", "chain-mix", "partial")


def rows_ok(rep, variant, checks, scale=1.0):
    rows = [r for r in rep["rows"] if r["variant"] == variant and r["check"] in checks]
    return bool(rows) and all(r["max_abs"] <= r["tol"] * scale for r in rows)


def class_parity(rep, margin=1.0):
    """{graph: {variant: pass|fail}} for one device class. fp16 flowdec is judged on the flowdec,
    chain-mix (enc fp32 -> flowdec fp16) and partial rows with ``margin``; everything else at the gate."""
    enc = ("enc.m_p", "enc.logs_p")
    out = {"enc": {}, "flowdec": {}, "knn": {}}
    for variant in ("fp32", "fp16"):
        # enc fp16 also needs its chain row (enc fp16 -> flowdec) under the gate
        enc_ok = rows_ok(rep, variant, enc) and (variant == "fp32" or rows_ok(rep, variant, ("chain",)))
        out["enc"][variant] = "pass" if enc_ok else "fail"
        if variant == "fp32":
            fd = rows_ok(rep, variant, ("flowdec", "partial", "fallback"))
        else:
            fd = rows_ok(rep, variant, FP16_FLOWDEC_ROWS, margin)
        out["flowdec"][variant] = "pass" if fd else "fail"
        out["knn"][variant] = "pass" if rows_ok(rep, variant, ("knn",)) else "fail"
    return out


def gpu_report(dml_python, template_id, templates_dir):
    fd, path = tempfile.mkstemp(suffix=".json", prefix="sigil-dml-")
    os.close(fd)
    cmd = [dml_python, os.path.join(HERE, "parity_test.py"), "--ep", "dml", "--template", template_id,
           "--templates", templates_dir, "--report", path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        raise SystemExit(f"DirectML parity of {template_id} did not report:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}") from None
    finally:
        os.remove(path)


def cpu_report(template_id, templates_dir):
    spec = configs.TEMPLATES[template_id]
    tdir = os.path.join(templates_dir, template_id)
    with open(os.path.join(tdir, "layout.json"), encoding="utf-8") as f:
        layout = json.load(f)
    donors = matrix_donors()
    state, note = parity_test.matrix_weights(template_id, spec, donors)
    frames = 300 if spec["vocoder"] == "refinegan" else 200
    rep = parity_test.parity(tdir, layout, state, spec, frames, 20, 5, weights_note=note, quiet=True)
    rep["ep"] = "cpu"
    return rep


_DONORS = None


def matrix_donors():
    global _DONORS
    if _DONORS is None:
        _DONORS = {}
        default_state = parity_test._state_of(parity_test.DEFAULT_PTH)
        if default_state is not None:
            _DONORS["hifigan"] = ("MIT default voice", default_state)
            _DONORS[parity_test.DEFAULT_TEMPLATE] = ("MIT default voice", {k: v.numpy() for k, v in default_state.items()})
        refine_state = parity_test._state_of(parity_test.REFINEGAN_PTH)
        if refine_state is not None:
            _DONORS["refinegan"] = ("Applio RefineGAN 32k pretrain", refine_state)
            _DONORS["v2-32k-f0-refinegan"] = ("Applio RefineGAN 32k pretrain", {k: v.numpy() for k, v in refine_state.items()})
    return _DONORS


def worst(rep, variant, checks):
    vals = [r["max_abs"] for r in rep["rows"] if r["variant"] == variant and r["check"] in checks]
    return max(vals) if vals else None


EDGE_KEYS = ("window_end", "kind", "returned_frames", "tol", "seam_frames", "saturated", "last_frame_max_abs", "last_frame_rel_rms")


def right_edge(rep, variant="fp32"):
    """The ``edge`` windows of a parity report (per-frame errors left out), [] when it has none."""
    return [{k: w[k] for k in EDGE_KEYS} for w in (rep.get("right_edge") or {}).get(variant, [])]


def slim_report(rep):
    """A parity report for reports/matrix_parity.json: the edge windows without their per-frame
    error profiles (the summary is the evidence; the profiles would triple the file)."""
    if rep is None:
        return None
    return dict(rep, right_edge={v: right_edge(rep, v) for v in (rep.get("right_edge") or {})})


def seam_frames(edges):
    """The longest no-lookahead seam of ``right_edge`` windows (None without any)."""
    return max((w["seam_frames"] for w in edges), default=None)


def export_one(template_id, out_root, dml_python):
    """Export, gate and finalise one template. Returns its summary."""
    spec = configs.TEMPLATES[template_id]
    tried = []
    chosen = None
    for mix in MIXES:
        stages = tuple(s for s in mix["fp32_stages"] if s < len(configs.config_dict(spec["config"])["upsample_rates"]))
        mix = {"fp32_flow": mix["fp32_flow"], "fp32_stages": stages, "fp32_ends": mix["fp32_ends"] and spec["vocoder"] == "refinegan"}
        if any(m == mix for m, _, _ in tried):
            continue
        export_rvc.FLOWDEC_FP16.clear()
        export_rvc.FLOWDEC_FP16.update(mix)
        t0 = time.time()
        export_rvc.export_template(template_id, out_root, verbose=False)
        cpu = cpu_report(template_id, out_root)
        gpu = gpu_report(dml_python, template_id, out_root) if dml_python else None
        tried.append((mix, cpu, gpu))
        ok_gpu = gpu is not None and class_parity(gpu, GPU_MARGIN)["flowdec"]["fp16"] == "pass"
        print(f"  {template_id}: mix {mix} -> cpu {'ok' if cpu['ok'] else 'FAIL'}, gpu fp16 flowdec "
              f"{'n/a' if gpu is None else f'{worst(gpu, 'fp16', FP16_FLOWDEC_ROWS):.2e}'} ({time.time() - t0:.0f} s)")
        if gpu is None or ok_gpu:
            chosen = (mix, cpu, gpu)
            break
    if chosen is None:
        # No mix passes on GPUs: ship the cheapest mix, marked fail there (the fp32 flowdec runs).
        mix = tried[0][0]
        if tried[-1][0] != mix:
            export_rvc.FLOWDEC_FP16.clear()
            export_rvc.FLOWDEC_FP16.update(mix)
            export_rvc.export_template(template_id, out_root, verbose=False)
        chosen = tried[0]
    mix, cpu, gpu = chosen
    finalise_layout(template_id, out_root, mix, cpu, gpu, tried)
    return {"template": template_id, "mix": mix, "cpu": cpu, "gpu": gpu, "tried": [m for m, _, _ in tried]}


def finalise_layout(template_id, out_root, mix, cpu, gpu, tried):
    path = os.path.join(out_root, template_id, "layout.json")
    with open(path, encoding="utf-8") as f:
        layout = json.load(f)
    pc = class_parity(cpu)
    pg = class_parity(gpu, GPU_MARGIN) if gpu is not None else None
    for graph, g in layout["graphs"].items():
        for variant, v in g["variants"].items():
            v["parity"] = {"cpu": pc[graph][variant], "gpu": pg[graph][variant] if pg else "untested"}
        rec = {"cpu": "fp32"}
        gpu16 = g["variants"]["fp16"]["parity"]["gpu"] == "pass"
        rec["gpu"] = "fp16" if graph in ("flowdec", "knn") and gpu16 else "fp32"
        g["recommended"] = rec
        for cls, variant in rec.items():
            if g["variants"][variant]["parity"][cls] == "fail":
                raise SystemExit(f"{template_id}: recommended {graph} {cls} variant {variant} fails parity")
    layout["exporter"]["flowdec_fp16"] = {"fp32_flow": mix["fp32_flow"], "fp32_stages": list(mix["fp32_stages"]), "fp32_ends": mix["fp32_ends"]}
    layout["parity_evidence"] = {
        "weights": cpu.get("weights", ""),
        "cpu": "ONNX Runtime 1.19.2 CPU (parity_test.py)",
        "gpu": ("DirectML via onnxruntime-directml 1.19.2 (the least accurate GPU EP for fp16, ADR 0004); "
                f"fp16 flowdec rows must stay under {GPU_MARGIN:g} x the gate") if gpu is not None else "not measured",
        "gate": {"fp32": parity_test.TOL["fp32"], "fp16": parity_test.TOL["fp16"]},
        "worst": {
            "cpu": {k: worst(cpu, v, c) for k, v, c in (
                ("flowdec_fp32", "fp32", ("flowdec", "partial")), ("flowdec_fp16", "fp16", FP16_FLOWDEC_ROWS),
                ("enc_fp32", "fp32", ("enc.m_p", "enc.logs_p")), ("enc_fp16", "fp16", ("enc.m_p", "enc.logs_p")))},
            "gpu": None if gpu is None else {k: worst(gpu, v, c) for k, v, c in (
                ("flowdec_fp32", "fp32", ("flowdec", "partial")), ("flowdec_fp16", "fp16", FP16_FLOWDEC_ROWS),
                ("enc_fp32", "fp32", ("enc.m_p", "enc.logs_p")), ("enc_fp16", "fp16", ("enc.m_p", "enc.logs_p")))},
        },
        "fp16_mixes_tried": [{"fp32_flow": m["fp32_flow"], "fp32_stages": list(m["fp32_stages"]), "fp32_ends": m["fp32_ends"]} for m, _, _ in tried],
        # T5 evidence: CPU fp32 windows ending mid-utterance with no lookahead vs the offline decode
        # (the encoder output of the whole utterance, so this is the flow's and the decoder's seam)
        "right_edge": right_edge(cpu),
    }
    sl.write_json(path, layout)


def dir_bytes(path):
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total


def write_index(out_root):
    templates = []
    for tid in configs.MATRIX:
        lp = os.path.join(out_root, tid, "layout.json")
        if not os.path.isfile(lp):
            continue
        with open(lp, encoding="utf-8") as f:
            layout = json.load(f)
        arch, st = layout["arch"], layout["streaming"]
        templates.append({
            "id": tid,
            "version": arch["version"],
            "sample_rate": arch["sample_rate"],
            "f0": arch["f0"],
            "vocoder": arch["vocoder"],
            "layout_version": layout["format_version"],
            "features": layout.get("features", []),
            "dec_margin": st["dec_margin"],
            # Streaming numbers to size lookahead / crossfades per template: the latent window offset,
            # the structural right receptive fields (frames after a frame its audio depends on) and
            # the measured no-lookahead seam (frames over 1e-4 vs offline, CPU fp32, evidence weights).
            "latent_offset": st.get("latent_offset", 0),
            "flow_receptive_right": st.get("flow_receptive_right"),
            "dec_receptive_right_frames": st.get("dec_receptive_right_frames"),
            "f0_lookahead_frames": st.get("excitation", {}).get("f0_lookahead_frames", 0),
            "seam_frames_fp32": seam_frames(layout.get("parity_evidence", {}).get("right_edge", [])),
            "recommended": {g: v["recommended"] for g, v in layout["graphs"].items()},
            "bytes": dir_bytes(os.path.join(out_root, tid)),
        })
    unsupported = [
        {"id": tid, "version": v, "sample_rate": sr, "f0": f0, "vocoder": voc, "reason": why}
        for tid, v, sr, f0, voc, why in configs.matrix_cells() if why
    ]
    index = {
        "format": "sigil-rvc-template-matrix",
        "format_version": 1,
        "templates": templates,
        "unsupported": unsupported,
        "total_bytes": sum(t["bytes"] for t in templates),
    }
    sl.write_json(os.path.join(out_root, "matrix.json"), index)
    return index


def write_reports(results):
    os.makedirs(REPORTS, exist_ok=True)
    slim = {}
    for tid, r in results.items():
        slim[tid] = {"mix": {"fp32_flow": r["mix"]["fp32_flow"], "fp32_stages": list(r["mix"]["fp32_stages"]), "fp32_ends": r["mix"]["fp32_ends"]},
                     "cpu": slim_report(r["cpu"]), "gpu": slim_report(r["gpu"])}
    stamp = datetime.date.today().isoformat()
    sl.write_json(os.path.join(REPORTS, "matrix_parity.json"), {"date": stamp, "templates": slim})
    lines = [
        f"# Template matrix parity ({stamp})",
        "",
        "Generated by `scripts/vc-templates/export_matrix.py`. Max |error| against the reference decode",
        "(RVC upstream for HiFi-GAN, float64 Applio arithmetic for RefineGAN / MRF); gate fp32 <= 1e-4,",
        f"fp16 <= 5e-3 (GPU fp16 flowdec rows <= {GPU_MARGIN:g} x the gate). GPU = DirectML (onnxruntime-directml 1.19.2, RTX 4090).",
        "Seam (T5 evidence, not gated): frames before the end of a mid-utterance window, decoded with no",
        "lookahead, over which the CPU fp32 graph differs from the offline decode by more than 1e-4 (voiced end /",
        "end at a voicing offset; the flow's and the decoder's missing right context, and RefineGAN's f0",
        "lookahead); `>=` = the whole returned window.",
        "",
        "| template | weights | fp16 mix (fp32 parts) | CPU flowdec fp32 | CPU fp16 | GPU flowdec fp32 | GPU fp16 | GPU runs | seam (frames) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for tid, r in results.items():
        cpu, gpu, mix = r["cpu"], r["gpu"], r["mix"]
        parts = (["flow"] if mix["fp32_flow"] else []) + [f"stage {s}" for s in mix["fp32_stages"]] + (["ends"] if mix.get("fp32_ends") else [])
        g16 = "n/a"
        run = "fp32"
        if gpu is not None:
            g16 = f"{worst(gpu, 'fp16', FP16_FLOWDEC_ROWS):.1e}"
            run = "fp16" if class_parity(gpu, GPU_MARGIN)["flowdec"]["fp16"] == "pass" else "fp32"
        seams = " / ".join((">=" if w["saturated"] else "") + str(w["seam_frames"]) for w in right_edge(cpu)) or "n/a"
        lines.append(
            f"| {tid} | {cpu.get('weights', '')} | {', '.join(parts)} | {worst(cpu, 'fp32', ('flowdec', 'partial')):.1e} | "
            f"{worst(cpu, 'fp16', FP16_FLOWDEC_ROWS):.1e} | {'n/a' if gpu is None else f'{worst(gpu, 'fp32', ('flowdec', 'partial')):.1e}'} | {g16} | {run} | {seams} |"
        )
    with open(os.path.join(REPORTS, "matrix_parity.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=export_rvc.DEFAULT_OUT)
    ap.add_argument("--template", action="append", choices=configs.MATRIX)
    ap.add_argument("--dml-python", default=None, help="python.exe of an environment with onnxruntime-directml (GPU parity)")
    ap.add_argument("--index-only", action="store_true", help="only rewrite matrix.json")
    args = ap.parse_args()
    if not args.index_only:
        results = {}
        existing = os.path.join(REPORTS, "matrix_parity.json")
        if args.template and os.path.isfile(existing):
            with open(existing, encoding="utf-8") as f:
                prev = json.load(f).get("templates", {})
            for tid, r in prev.items():
                results[tid] = {"mix": {"fp32_flow": r["mix"]["fp32_flow"], "fp32_stages": tuple(r["mix"]["fp32_stages"]), "fp32_ends": r["mix"].get("fp32_ends", False)},
                                "cpu": r["cpu"], "gpu": r["gpu"]}
        for tid in args.template or configs.MATRIX:
            print(f"[{tid}]")
            results[tid] = export_one(tid, args.out, args.dml_python)
        results = {t: results[t] for t in configs.MATRIX if t in results}
        write_reports(results)
        bad = [t for t, r in results.items() if not r["cpu"]["ok"]]
        if bad:
            raise SystemExit(f"CPU parity failed: {', '.join(bad)}")
    index = write_index(args.out)
    total = dir_bytes(args.out)
    print(f"{len(index['templates'])} templates, {len(index['unsupported'])} unsupported cells, "
          f"{total / 1048576:.1f} MiB bundled (budget {BUDGET_BYTES / 1048576:.0f} MiB)")
    if total > BUDGET_BYTES:
        raise SystemExit("the bundled templates exceed the 30 MB budget")


if __name__ == "__main__":
    main()
