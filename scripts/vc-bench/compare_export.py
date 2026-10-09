#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the files of a compare v2 run next to its ``report.json`` (standard library only):

- ``<run>.make_test.json``: the blind A/B manifest for ``tools/listening-test/make_test.py`` (the
  aligned, trimmed, loudness-matched renditions ``<run>.pair.<clip>.<point>.wav`` the run wrote
  when the spec had ``output.write_audio``);
- ``<run>.pareto.svg``: mean DNSMOS (95 % CI) against latency, one panel per tier and latency
  kind, Sigil's rows and the other systems' points, the Pareto front joined;
- ``<run>.t6.md``: the T6 verdicts per competitor point, the corpus sizing and the ablations.

    python scripts/vc-bench/compare_export.py <out>/report.json [--run h2h]
    python tools/listening-test/make_test.py <out>/h2h.make_test.json --out listening-1

``--self-test`` renders a synthetic report (no files needed).
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys


def svg_pareto(points: list, title: str) -> str:
    """One panel per (tier, latency kind); x = latency (ms), y = mean DNSMOS with its CI."""
    groups = {}
    for p in points:
        if p.get("latency_ms") is None or p.get("dnsmos_mean") is None:
            continue
        groups.setdefault((p.get("tier", ""), p.get("latency_kind", "")), []).append(p)
    if not groups:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="480" height="60"><text x="10" y="35" font-family="sans-serif" font-size="14">{html.escape(title)}: no point has both a latency and a DNSMOS mean</text></svg>'
    w, h, pad = 520, 340, 56
    panels = []
    for gi, ((tier, kind), pts) in enumerate(sorted(groups.items())):
        xs = [p["latency_ms"] for p in pts]
        ys = [v for p in pts for v in (p.get("dnsmos_ci95") or [p["dnsmos_mean"]] * 2)] + [p["dnsmos_mean"] for p in pts]
        x0, x1 = 0.0, max(xs) * 1.15 + 1
        y0, y1 = min(ys) - 0.05, max(ys) + 0.05
        sx = lambda x: pad + (x - x0) / (x1 - x0) * (w - 2 * pad)  # noqa: E731
        sy = lambda y: h - pad - (y - y0) / (y1 - y0) * (h - 2 * pad)  # noqa: E731
        g = [f'<g transform="translate(0,{gi * (h + 10)})">',
             f'<rect x="0" y="0" width="{w}" height="{h}" fill="#fff" stroke="#ccc"/>',
             f'<text x="{w / 2}" y="20" text-anchor="middle" font-size="14">{html.escape(title)}: {html.escape(tier)} tier, {html.escape(kind or "unknown")} latency</text>',
             f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#333"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#333"/>',
             f'<text x="{w / 2}" y="{h - 14}" text-anchor="middle" font-size="12">latency (ms)</text>',
             f'<text x="16" y="{h / 2}" transform="rotate(-90 16 {h / 2})" text-anchor="middle" font-size="12">mean DNSMOS OVRL</text>']
        for t in range(5):
            xv = x0 + (x1 - x0) * t / 4
            yv = y0 + (y1 - y0) * t / 4
            g.append(f'<text x="{sx(xv):.1f}" y="{h - pad + 16}" text-anchor="middle" font-size="10">{xv:.0f}</text>')
            g.append(f'<text x="{pad - 6}" y="{sy(yv) + 3:.1f}" text-anchor="end" font-size="10">{yv:.2f}</text>')
        front = sorted((p for p in pts if p.get("front")), key=lambda p: p["latency_ms"])
        if len(front) > 1:
            g.append('<polyline fill="none" stroke="#999" stroke-dasharray="4 3" points="' + " ".join(f"{sx(p['latency_ms']):.1f},{sy(p['dnsmos_mean']):.1f}" for p in front) + '"/>')
        for p in pts:
            x, y = sx(p["latency_ms"]), sy(p["dnsmos_mean"])
            sigil = p.get("role") in ("sigil", "ablation")
            color = "#3b5bdb" if p.get("role") == "sigil" else ("#868e96" if p.get("role") == "ablation" else "#e8590c")
            ci = p.get("dnsmos_ci95")
            if ci:
                g.append(f'<line x1="{x:.1f}" y1="{sy(ci[0]):.1f}" x2="{x:.1f}" y2="{sy(ci[1]):.1f}" stroke="{color}"/>')
            shape = f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>' if sigil else f'<rect x="{x - 5:.1f}" y="{y - 5:.1f}" width="10" height="10" fill="{color}"/>'
            g.append(shape)
            g.append(f'<text x="{x + 8:.1f}" y="{y - 6:.1f}" font-size="10">{html.escape(p["name"])} ({p.get("dnsmos_clips", 0)} clips)</text>')
        g.append("</g>")
        panels.append("\n".join(g))
    total_h = len(panels) * (h + 10)
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{total_h}" font-family="sans-serif">\n' + "\n".join(panels) + "\n</svg>\n"


def t6_markdown(run: str, v2: dict) -> str:
    out = [f"# T6 head-to-head: {run}", ""]
    t6 = v2.get("t6") or {}
    out.append(f"T6: **{t6.get('status', '?')}**")
    for r in (t6.get("reasons") or [])[:40]:
        out.append(f"- {r}")
    caveats = t6.get("caveats") or []
    if caveats:
        out += ["", "Caveats (read these next to the status):"] + [f"- {c}" for c in caveats[:40]]
    out += ["", "| point | system | tier | DNSMOS | GPE | MCD | latency |", "|---|---|---|---|---|---|---|"]
    for v in v2.get("verdicts", []):
        cell = lambda r: f"{r['status']}" + (f" ({r['by']})" if r.get("by") else "")  # noqa: E731
        out.append(f"| {v['point']} | {v['system']} | {v['tier']} | {cell(v['dnsmos'])} | {cell(v['gpe'])} | {cell(v['mcd'])} | {cell(v['latency'])} |")
    s = v2.get("sizing")
    if s:
        out += ["", f"Corpus sizing: paired DNSMOS SD {s['dnsmos_paired_sd']:.3f}; the +0.05 row needs {s['clips_needed_50']} clips (50 % power) / {s['clips_needed_80']} (80 %); the corpus has {s['clips_with_dnsmos']} of at least 9.01 s. {s['note']}."]
    abl = v2.get("ablations") or []
    if abl:
        out += ["", "## Ablations (never competitors)", "", "| ablation | versus | DNSMOS diff (versus − ablation) | MCD diff (upstream refs) | MCD diff (stand-in refs) | artifacts/min diff | seam clicks diff | clips for +0.05 |", "|---|---|---|---|---|---|---|---|"]
        for a in abl:
            p = a["pairing"]
            d = lambda x: "-" if not x else f"{x['mean']:+.3f} [{x['t_lo']:+.3f}, {x['t_hi']:+.3f}] sd {x['sd']:.3f} n {x['n']}"  # noqa: E731
            need = a.get("clips_needed_dnsmos") or a.get("clips_needed_dnsmos_tiled")
            out.append(f"| {a['ablation']} | {a['versus']} | {d(p.get('dnsmos')) if p.get('dnsmos') else d(p.get('dnsmos_tiled')) + ' (tiled, advisory)'} | {d(p.get('mcd'))} | {d(p.get('mcd_stand_in'))} | {d(p.get('artifacts'))} | {d(p.get('seam_clicks'))} | {need} |")
    pts = v2.get("points", [])
    if pts:
        out += ["", "## Points", "", "| point | system | role | tier | latency (ms) | source | feasible | clips not scored | refusals | flags |", "|---|---|---|---|---|---|---|---|---|---|"]
        for p in pts:
            lat = "-" if p.get("latency_ms") is None else f"{p['latency_ms']:.1f} {p.get('latency_kind', '')}"
            out.append(f"| {p['name']} | {p['system']} | {p['role']} | {p['tier']} | {lat} | {p.get('latency_source', '')} | {p.get('feasible')} | {len(p.get('unscored_clips') or [])} | {'; '.join(p.get('refusals', []))} | {'; '.join(p.get('flags', []))} |")
    skipped = (v2.get("make_test") or {}).get("skipped") or []
    if skipped:
        out += ["", "## Left out of the listening test (unfair to hear)", ""] + [f"- {x['point']} (against {x['sigil']}): {x['why']}" for x in skipped]
    return "\n".join(out) + "\n"


def export(report_path: str, run_name: str = "") -> list:
    with open(report_path, encoding="utf-8") as f:
        rep = json.load(f)
    out_dir = os.path.dirname(os.path.abspath(report_path))
    written = []
    for r in rep.get("runs", []):
        if r.get("mode") != "compare" or (run_name and r.get("name") != run_name):
            continue
        v2 = (r.get("details") or {}).get("v2")
        if not v2:
            continue
        name = r["name"]
        mt = dict(v2.get("make_test") or {})
        mt.pop("$comment", None)
        mt.pop("skipped", None)  # listed in the t6.md instead
        present = [t for t in mt.get("trials", []) if os.path.isfile(os.path.join(out_dir, t["a"])) and os.path.isfile(os.path.join(out_dir, t["b"]))]
        if len(present) < len(mt.get("trials", [])):
            print(f"{name}: {len(mt.get('trials', [])) - len(present)} trial(s) without their WAVs (run the spec with output.write_audio)")
        mt["trials"] = present
        paths = {
            f"{name}.make_test.json": json.dumps(mt, indent=1),
            f"{name}.pareto.svg": svg_pareto(v2.get("pareto", []), name),
            f"{name}.t6.md": t6_markdown(name, v2),
        }
        for fname, text in paths.items():
            p = os.path.join(out_dir, fname)
            with open(p, "w", encoding="utf-8") as f:
                f.write(text)
            written.append(p)
    return written


def self_test() -> bool:
    import tempfile

    d = tempfile.mkdtemp()
    v2 = {
        "t6": {"status": "undecided", "reasons": ["x latency: Undecided"], "caveats": ["rvc: operating point: index_rate 0.75 here, the Sigil rows use 0"]},
        "verdicts": [{"point": "rvc", "system": "rvc_webui", "tier": "offline", **{k: {"status": "pass", "by": "sve", "detail": ""} for k in ("dnsmos", "gpe", "mcd", "latency")}}],
        "sizing": {"dnsmos_paired_sd": 0.21, "clips_needed_50": 68, "clips_needed_80": 139, "clips_with_dnsmos": 11, "note": "too small"},
        "ablations": [{"ablation": "abl", "versus": "sve", "pairing": {"dnsmos": {"mean": 0.03, "t_lo": -0.04, "t_hi": 0.1, "sd": 0.15, "n": 11}}, "clips_needed_dnsmos": [35, 72]}],
        "points": [{"name": "sve", "system": "sigil", "role": "sigil", "tier": "offline", "latency_ms": 60.0, "latency_kind": "algorithmic", "refusals": [], "flags": [], "unscored_clips": ["c3"]}],
        "pareto": [
            {"name": "sve", "role": "sigil", "tier": "offline", "latency_ms": 60.0, "latency_kind": "algorithmic", "dnsmos_mean": 3.1, "dnsmos_ci95": [3.0, 3.2], "dnsmos_clips": 11, "front": True},
            {"name": "rvc", "role": "competitor", "tier": "offline", "latency_ms": 300.0, "latency_kind": "algorithmic", "dnsmos_mean": 3.0, "dnsmos_ci95": [2.9, 3.1], "dnsmos_clips": 11, "front": False},
        ],
        "make_test": {"title": "t", "question": "q", "trials": [{"clip": "c", "a": "h.pair.c.sve.wav", "system_a": "sve", "b": "h.pair.c.rvc.wav", "system_b": "rvc"}],
                      "skipped": [{"point": "bad", "sigil": "sve", "why": "refused: transpose 12"}]},
    }
    rep = {"runs": [{"name": "h", "mode": "compare", "details": {"v2": v2}}]}
    p = os.path.join(d, "report.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f)
    for n in ("h.pair.c.sve.wav", "h.pair.c.rvc.wav"):
        open(os.path.join(d, n), "wb").close()
    files = export(p)
    ok = len(files) == 3
    svg = open(os.path.join(d, "h.pareto.svg"), encoding="utf-8").read()
    ok &= svg.startswith("<svg") and "sve" in svg and "polyline" not in svg  # one front point: no line
    mt = json.load(open(os.path.join(d, "h.make_test.json"), encoding="utf-8"))
    ok &= len(mt["trials"]) == 1 and "$comment" not in mt and "skipped" not in mt
    md = open(os.path.join(d, "h.t6.md"), encoding="utf-8").read()
    ok &= "undecided" in md and "| rvc | rvc_webui |" in md and "68 clips" in md
    ok &= "Caveats" in md and "index_rate 0.75" in md and "bad (against sve): refused: transpose 12" in md
    print("self-test", "passed" if ok else "FAILED")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?")
    ap.add_argument("--run", default="")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        sys.exit(0 if self_test() else 1)
    if not args.report:
        ap.error("a report.json (or --self-test)")
    for p in export(args.report, args.run):
        print(f"-> {p}")


if __name__ == "__main__":
    main()
