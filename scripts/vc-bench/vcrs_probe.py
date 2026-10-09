#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Same-rig compute rows for plan T6's normalized-RTF row (Sigil against vc-rs, rvc-realtime,
RVC WebUI, Applio): per-block compute p50 / p95 / p99 and p95 / block period, with a verdict only
between rows measured on the SAME GPU.

Published numbers come from other GPUs and other measurements: vc-rs's 13.9 ms p95 per 100 ms chunk
is trtexec model-only time (no host transfers) with TensorRT 11 on an RTX 3060 Ti, and
rvc-realtime's figure is p50 15.7 / p95 21.2 ms per 60 ms block (FCPE, fp16) on an RTX 3060 Ti. Between
an RTX 4090 and a 3060 Ti the speed ratio of small kernels is anywhere from about 1.5x (clock) to 5x
(SMs x clock), so no cross-rig ranking holds: such rows are listed as references, never judged.

Inputs (any mix):
- ``--sigil report.json``: a Sigil benchmark report; every offline / paced run with ``call_ms_*``
  becomes a row (GPU: the report's environment). Its per-component rows (content, F0, enc,
  flowdec) are labelled "model step incl. host I/O" (Sigil's stages run with their host copies,
  unlike trtexec).
- ``--vcrs file.csv``: a vc-rs ``latency_probe`` CSV (a header row; the per-chunk time column is
  found by name: ``total`` / ``infer`` / ``ms``, or given with ``--vcrs-column``; ``--vcrs-chunk-ms``,
  default 100, and ``--vcrs-gpu`` describe the run).
- ``--sidecars DIR``: the JSON sidecars of the ``scripts/vc-bench`` drivers (``*.wav.json``): each
  operating point's per-block compute, GPU and block.

    python scripts/vc-bench/vcrs_probe.py --sigil out/report.json --vcrs vcrs_latency.csv --vcrs-gpu "NVIDIA GeForce RTX 4090" \\
        --sidecars corpus/out/rvc_webui --json rows.json

The JSON output holds every row and the same-GPU verdicts (for the integrator's scorecard).
``--self-test`` checks the ingestion on synthetic inputs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import vcbench_common as vb  # noqa: E402

# Published reference rows (another GPU, another measurement: never judged against this rig).
REFERENCES = [
    {"system": "vc_rs", "label": "vc-rs, TensorRT 11, trtexec model-only (no host transfers), 100 ms chunk + 100 ms extra",
     "gpu": "NVIDIA GeForce RTX 3060 Ti", "block_ms": 100.0, "p95": 13.9,
     "components_p95": {"contentvec": 2.51, "rmvpe": 5.32, "rvc": 6.07}, "source": "vc-rs docs/tensorrt_performance_ja.md"},
    {"system": "rvc_realtime", "label": "rvc-realtime, FCPE, fp16, 60 ms block", "gpu": "NVIDIA GeForce RTX 3060 Ti",
     "block_ms": 60.0, "p50": 15.7, "p95": 21.2, "source": "rvc-realtime README benchmark"},
]


def norm_gpu(name: str) -> str:
    return " ".join((name or "").lower().replace("nvidia", "").replace("geforce", "").split())


def row(system, label, gpu, block_ms, stats, source, kind="per-block compute", f0=""):
    p95 = stats.get("p95")
    return {
        "system": system, "label": label, "gpu": gpu, "block_ms": block_ms, "f0": str(f0 or "").strip().lower(),
        "p50": stats.get("p50"), "p95": p95, "p99": stats.get("p99"),
        "rtf_p95": (p95 / block_ms) if (p95 is not None and block_ms) else None,
        "kind": kind, "source": source,
    }


def sigil_rows(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        rep = json.load(f)
    gpus = [g for g in (rep.get("environment") or {}).get("gpus", []) if not g.get("software")]
    gpu = gpus[0]["name"] if gpus else ""
    eng = (rep.get("engine") or {})
    out = []
    for r in rep.get("runs", []):
        m = r.get("metrics") or {}
        if "call_ms_p95" not in m:
            continue
        plan = eng.get("plan") or {}
        block = None
        if isinstance(plan, dict):
            block = plan.get("block_ms") or (plan.get("block_16k", 0) / 16.0 if plan.get("block_16k") else None)
        block = block or (r.get("details") or {}).get("block_ms")
        ep = eng.get("execution_provider_active") or eng.get("execution_provider") or ""
        label = f"Sigil {r.get('name')} ({ep}, {eng.get('pitch_extractor', '')})"
        # A CPU run is compared with CPU runs of the same processor, never with a GPU's.
        dev = f"CPU {(rep.get('environment') or {}).get('cpu', '')}".strip() if ep == "cpu" else gpu
        stats = {"p50": m.get("call_ms_p50"), "p95": m.get("call_ms_p95"), "p99": m.get("call_ms_p99")}
        out.append(row("sigil", label, dev, block, stats, os.path.abspath(path), "per-call compute (whole engine, host I/O included)", f0=eng.get("pitch_extractor", "")))
        comps = {k: m.get(f"{k}_ms_p95") for k in ("content", "f0", "enc", "flowdec") if f"{k}_ms_p95" in m}
        if comps:
            out[-1]["components_p95"] = comps
            out[-1]["components_label"] = "model step incl. host I/O"
    return out


def vcrs_rows(path: str, column: str, chunk_ms: float, gpu: str) -> list:
    with open(path, encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        fields = rd.fieldnames or []
        col = column
        if not col:
            for key in ("total_ms", "total", "infer_ms", "inference_ms", "ms", "latency_ms"):
                hit = [c for c in fields if c.strip().lower() == key]
                if hit:
                    col = hit[0]
                    break
            if not col:
                hit = [c for c in fields if any(t in c.lower() for t in ("total", "infer", "ms"))]
                col = hit[0] if hit else ""
        if not col:
            raise SystemExit(f"{path}: no per-chunk time column found in {fields}; give --vcrs-column")
        values = []
        for rec in rd:
            try:
                values.append(float(rec[col]))
            except (TypeError, ValueError, KeyError):
                continue
    if not values:
        raise SystemExit(f"{path}: column {col!r} has no numbers")
    return [row("vc_rs", f"vc-rs latency_probe ({col}, {len(values)} chunks)", gpu, chunk_ms, vb.percentiles(values), os.path.abspath(path))]


def sidecar_rows(folder: str) -> list:
    groups = {}
    for p in sorted(glob.glob(os.path.join(folder, "**", "*.wav.json"), recursive=True)):
        with open(p, encoding="utf-8") as f:
            s = json.load(f)
        if s.get("schema") != vb.SIDECAR_SCHEMA or not s.get("call_ms"):
            continue
        cfg = s.get("config") or {}
        device = "CPU" if str(cfg.get("ep", "")).lower() == "cpu" else s.get("gpu", "")
        key = (s.get("system", ""), cfg.get("block_ms"), device, cfg.get("f0_method", ""))
        groups.setdefault(key, []).append(s["call_ms"])
    out = []
    for (system, block, gpu, f0), stats in groups.items():
        # Worst p95 / p99 over the clips, median p50 (as Sigil's compare aggregates its own rows).
        p50s = sorted(x["p50"] for x in stats if x.get("p50") is not None)
        agg = {
            "p50": p50s[len(p50s) // 2] if p50s else None,
            "p95": max((x["p95"] for x in stats if x.get("p95") is not None), default=None),
            "p99": max((x["p99"] for x in stats if x.get("p99") is not None), default=None),
        }
        out.append(row(system, f"{system} driver, {f0}, {len(stats)} clip(s)", gpu, block, agg, folder, f0=f0))
    return out


def same_block(a, b) -> bool:
    return a is not None and b is not None and abs(float(a) - float(b)) < 0.5


def verdicts(rows: list) -> list:
    """Sigil against every other row of the same GPU, the same block and (where both state it) the
    same F0 method: Sigil's p95 RTF at most theirs.

    Only Sigil rows at the competitor's own operating point count. Taking Sigil's best row over any
    block, F0 method or run would pit a 250 ms block's RTF against a 40 ms block's and pick the
    most favourable of many; several Sigil rows at the point (runs, configurations) are judged by
    their upper median, not their best."""
    out = []
    sig = [r for r in rows if r["system"] == "sigil" and r.get("rtf_p95") is not None]
    for o in rows:
        if o["system"] == "sigil":
            continue
        if o.get("kind", "").startswith("published") or o.get("rtf_p95") is None:
            out.append({"other": o["label"], "verdict": "none", "why": "a published reference (another GPU and measurement): listed, never judged"})
            continue
        same = [s for s in sig if norm_gpu(s["gpu"]) and norm_gpu(s["gpu"]) == norm_gpu(o["gpu"])]
        if not same:
            out.append({"other": o["label"], "verdict": "none", "why": f"no Sigil row on {o['gpu'] or 'an unnamed GPU'}: a cross-GPU ratio is not judged"})
            continue
        blocks = [s for s in same if same_block(s["block_ms"], o["block_ms"])]
        if not blocks:
            have = sorted({s["block_ms"] for s in same if s["block_ms"] is not None})
            out.append({"other": o["label"], "verdict": "none",
                        "why": f"no Sigil row at the competitor's {o['block_ms']} ms block on this GPU (Sigil has {have}): a ratio across blocks is not judged"})
            continue
        # Where both name an F0 method it must be the same one (RMVPE costs more than FCPE).
        f0s = [s for s in blocks if not (s.get("f0") and o.get("f0")) or s["f0"] == o["f0"]]
        if not f0s:
            have = sorted({s.get("f0") or "?" for s in blocks})
            out.append({"other": o["label"], "verdict": "none",
                        "why": f"no Sigil row with the competitor's F0 method '{o.get('f0')}' at that block (Sigil has {have})"})
            continue
        ranked = sorted(f0s, key=lambda s: s["rtf_p95"])
        chosen = ranked[len(ranked) // 2]  # the upper median of the matching rows
        out.append({
            "other": o["label"], "sigil": chosen["label"], "gpu": o["gpu"],
            "sigil_rtf_p95": chosen["rtf_p95"], "other_rtf_p95": o["rtf_p95"],
            "verdict": "sigil_faster_or_equal" if chosen["rtf_p95"] <= o["rtf_p95"] else "sigil_slower",
            "note": f"same GPU and block ({o['block_ms']:g} ms)" + (f", F0 {o['f0']}" if o.get("f0") and chosen.get("f0") else ", F0 method not compared")
                    + f"; judged on the upper median of {len(f0s)} Sigil row(s)",
        })
    return out


def self_test() -> bool:
    import tempfile

    d = tempfile.mkdtemp()
    csv_path = os.path.join(d, "probe.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("chunk,total_ms,host_ms\n" + "".join(f"{i},{10 + (i % 5)},{1}\n" for i in range(100)))
    rows = vcrs_rows(csv_path, "", 100.0, "NVIDIA GeForce RTX 4090")
    ok = abs(rows[0]["p50"] - 12.0) < 1e-9 and abs(rows[0]["rtf_p95"] - 0.14) < 1e-9
    os.makedirs(os.path.join(d, "side"))
    for i, p in enumerate((5.0, 7.0)):
        side = {"schema": vb.SIDECAR_SCHEMA, "system": "rvc_webui", "gpu": "RTX 4090", "config": {"block_ms": 40, "f0_method": "rmvpe"},
                "call_ms": {"p50": p, "p95": p + 1, "p99": p + 2}}
        with open(os.path.join(d, "side", f"c{i}.wav.json"), "w", encoding="utf-8") as f:
            json.dump(side, f)
    srows = sidecar_rows(os.path.join(d, "side"))
    ok &= len(srows) == 1 and srows[0]["p95"] == 8.0 and srows[0]["p99"] == 9.0
    sig = row("sigil", "Sigil offline", "NVIDIA GeForce RTX 4090", 40.0, {"p50": 4, "p95": 6, "p99": 7}, "test")
    refs = [dict(r, rtf_p95=r["p95"] / r["block_ms"], kind="published reference") for r in REFERENCES]
    v = verdicts([sig, srows[0], rows[0]] + refs)
    by = {x["other"]: x for x in v}
    ok &= by[srows[0]["label"]]["verdict"] == "sigil_faster_or_equal"
    # vc-rs ran 100 ms chunks and Sigil's only row is a 40 ms block: no Sigil row at its operating
    # point, so no verdict (a ratio across blocks used to read "sigil_slower" here).
    ok &= by[rows[0]["label"]]["verdict"] == "none" and "100" in by[rows[0]["label"]]["why"]
    ok &= all(by[r["label"]]["verdict"] == "none" for r in refs)
    # A row of another GPU gets no verdict.
    other = row("vc_rs", "vc-rs elsewhere", "NVIDIA GeForce RTX 3060", 100.0, {"p95": 5.0}, "test")
    ok &= verdicts([sig, other])[0]["verdict"] == "none"
    # Sigil's best row over other blocks is not borrowed: a 250 ms row (RTF 0.05) must not decide a
    # 40 ms competitor (RTF 0.10) when Sigil's own 40 ms row (RTF 0.15) is slower.
    gpu = "NVIDIA GeForce RTX 4090"
    s250 = row("sigil", "Sigil 250", gpu, 250.0, {"p95": 12.5}, "t", f0="rmvpe")
    s40 = row("sigil", "Sigil 40", gpu, 40.0, {"p95": 6.0}, "t", f0="rmvpe")
    comp = row("rvc_webui", "RVC 40", gpu, 40.0, {"p95": 4.0}, "t", f0="rmvpe")
    v = verdicts([s250, comp])[0]
    ok &= v["verdict"] == "none" and "250" in v["why"]
    v = verdicts([s250, s40, comp])[0]
    ok &= v["verdict"] == "sigil_slower" and v["sigil"] == "Sigil 40"
    # The F0 method must match where both state it; several runs are judged by the upper median.
    fcpe = row("sigil", "Sigil fcpe", gpu, 40.0, {"p95": 2.0}, "t", f0="fcpe")
    ok &= verdicts([fcpe, comp])[0]["verdict"] == "none"
    runs = [row("sigil", f"run{i}", gpu, 40.0, {"p95": p}, "t", f0="rmvpe") for i, p in enumerate((3.0, 3.5, 5.0))]
    v = verdicts(runs + [comp])[0]
    ok &= v["sigil"] == "run1" and v["verdict"] == "sigil_faster_or_equal"  # median 3.5 ms (0.0875) against 4.0 ms (0.1)
    print("self-test", "passed" if ok else "FAILED")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigil", action="append", default=[], help="a Sigil report.json (repeatable)")
    ap.add_argument("--vcrs", action="append", default=[], help="a vc-rs latency_probe CSV (repeatable)")
    ap.add_argument("--vcrs-column", default="")
    ap.add_argument("--vcrs-chunk-ms", type=float, default=100.0)
    ap.add_argument("--vcrs-gpu", default="", help="the GPU the vc-rs CSV was measured on")
    ap.add_argument("--sidecars", action="append", default=[], help="a folder of driver sidecars (repeatable)")
    ap.add_argument("--json", help="write rows and verdicts here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    rows = []
    for p in args.sigil:
        rows += sigil_rows(p)
    for p in args.vcrs:
        rows += vcrs_rows(p, args.vcrs_column, args.vcrs_chunk_ms, args.vcrs_gpu)
    for p in args.sidecars:
        rows += sidecar_rows(p)
    refs = [dict(r, rtf_p95=(r["p95"] / r["block_ms"]), kind="published reference (another GPU: no verdict)") for r in REFERENCES]
    v = verdicts(rows + refs)
    fmt = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    print(f"{'system':14s} {'GPU':28s} {'block':>6s} {'p50':>7s} {'p95':>7s} {'p99':>7s} {'RTF p95':>8s}  label")
    for r in rows + refs:
        print(f"{r['system']:14s} {r['gpu'][:28]:28s} {fmt(r['block_ms']):>6s} {fmt(r.get('p50')):>7s} {fmt(r.get('p95')):>7s} {fmt(r.get('p99')):>7s} {fmt(r.get('rtf_p95')):>8s}  {r['label']}")
        if r.get("components_p95"):
            print(f"{'':14s} components p95 ({r.get('components_label', 'model-only')}): " + ", ".join(f"{k} {fmt(x)}" for k, x in r["components_p95"].items()))
    print()
    for x in v:
        print(json.dumps(x))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({"rows": rows, "references": refs, "verdicts": v}, f, indent=1)


if __name__ == "__main__":
    main()
