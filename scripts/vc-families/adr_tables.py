#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""ADR 0006's evidence tables, built from ``reports/*.json`` (P3-H).

Prints the tables; ``--write`` replaces each ``<!-- tables:NAME -->`` ... ``<!-- /tables:NAME -->``
block of docs/adr/0006-vc-zero-shot-singing.md with them, so every number in the ADR's tables comes
from a checked-in report. Blocks: ``parity``, ``join``, ``cost``, ``t10``, ``profile``.

Usage (repo root): ``python scripts/vc-families/adr_tables.py [--write]``
"""

from __future__ import annotations

import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORTS = os.path.join(HERE, "reports")
ADR = os.path.join(HERE, "..", "..", "docs", "adr", "0006-vc-zero-shot-singing.md")
H1_FACTORS = (3, 5)   # H1 (RTX 3060 / 4060) per-call time as a multiple of the 4090's: the ADR's assumption and a pessimistic bound
T10_P95_MS = 300


def load(name):
    p = os.path.join(REPORTS, name)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def snr(r):
    if r is None:
        return "—"
    s = r["snr_db"]
    lm = get(r, "logmel", "snr_db")
    return f"{s}" + (f" ({lm})" if lm is not None else "")


def best(row):
    """(mode, timing) of the cheapest measured way to run a call: graph > bound > host."""
    for m in ("graph", "bound", "host"):
        if m in row and "median_ms" in row[m]:
            return m, row[m]
    raise ValueError(row)


REFUSED = "¹"
REFUSED_NOTE = ("¹ DirectML refused the pinned dimensions; the session runs with dynamic shapes "
                "(`session.dims_refused` in the report).")


def fmt(row):
    m, t = best(row)
    refused = bool(get(row, "session", "dims_refused"))
    return f"{t['median_ms']:.2f}" + ("" if m == "host" else f" {m[0]}") + (f" {REFUSED}" if refused else "")


def conservative(row):
    """A per-call figure for p95 estimates: the larger of the kept block's p90 and the slowest
    timing block's median (other processes on the shared rig only ever slow a block down)."""
    _, t = best(row)
    return max(t["p90_ms"], max(t.get("block_medians_ms", [t["median_ms"]])))


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------------------------

def parity_md():
    sv = {ep: load(f"seed_vc_{ep}.json") for ep in ("cpu", "cuda", "dml")}
    cols = [("CPU fp32", "cpu", "fp32"), ("CUDA fp32", "cuda", "fp32"), ("DirectML fp32", "dml", "fp32"),
            ("CUDA hift16", "cuda", "hift16"), ("CUDA xlsr16", "cuda", "xlsr16"), ("CUDA fp16 (all)", "cuda", "fp16")]
    names = {"xlsr_window": "xlsr (tick window)", "campplus": "campplus", "mel": "mel (prompt)", "lr_tick": "lr (tick)",
             "dit_estimator": "dit (estimator, CFG batch)", "dit_step": "dit_step", "dit_step1": "dit_step1",
             "hift": "hift (whole tick, on upstream's mel)", "hift_f0": "hift_f0", "hift_partial": "hift_f0 + hift_nsf, partial decode (returned audio)",
             "chain_tick_mel": "chain: tick mel", "chain_tick_audio": "chain: tick audio, `hift`",
             "chain_tick_audio_partial": "chain: tick audio, partial decode", "lean_chain_tick_audio": "chain: tick audio, `sigil_lean`, `hift`",
             "lean_chain_tick_audio_partial": "chain: tick audio, `sigil_lean`, partial decode"}
    out = ["*Seed-VC* (preset `upstream_readme` unless noted; stage rows feed each graph upstream's own inputs, chain rows run the "
           "whole reference branch and one tick):\n"]
    rows = []
    for k, label in names.items():
        rows.append([label] + [snr(get(sv[ep], "variants", v, "parity", k)) for _, ep, v in cols])
    out.append(table(["stage"] + [c[0] for c in cols], rows))
    fr = get(sv["cuda"], "variants", "fp32", "upstream_realtime_fp16_vs_fp32")
    if fr:
        out.append(f"\nFor scale: upstream's own fp16 real-time mode against upstream fp32 on CUDA (same draws) gives tick audio "
                   f"{fr['tick_audio']['snr_db']} dB (log-mel {fr['tick_audio']['logmel']['snr_db']} dB) and tick mel "
                   f"{fr['tick_mel']['snr_db']} dB.")

    dd = {ep: load(f"ddsp_svc_{ep}.json") for ep in ("cpu", "cuda", "dml")}
    names = {"units_tta2x_shared_embedder": "units: Sigil ContentVec ONNX × 2 (tta2x) vs upstream encoder", "ddsp_cond": "ddsp: cond (DDSP mel)",
             "ddsp_x": "ddsp: x (normalised)", "chain_mel": "chain: mel after 20 flow steps", "chain_audio": "chain: tick audio (formant +2 st)",
             "chain_audio_host_update": "chain: tick audio, update on the host (`velocity_f`)", "chain_audio_formant0": "chain: tick audio (formant 0)",
             "chain_audio_volume_gate": "chain: tick audio + `SvcDDSP.infer`'s volume gate",
             "vocoder_sweep_E2_Ds7_4s": "vocoder alone: 4 s real mel, F0 sweep E2 → D#7"}
    out.append("\n*DDSP-SVC 6.3* (preset `upstream_gui`: 2.5 s window, 20 steps from t_start 0.7; F0 sweeps E2 → D#7 over the "
               "decoded frames):\n")
    eps = ("cpu", "cuda", "dml")
    rows = [[label] + [snr(get(dd[ep], "parity", k)) for ep in eps] for k, label in names.items()]
    gate = get(dd["cpu"], "parity", "chain_audio_volume_gate")
    ue = {ep: get(dd[ep], "upstream_exporter") or {} for ep in eps}
    rows.append(["upstream exporter: `encoder.onnx` cond over the decoded frames"]
                + [str(get(ue[ep], "encoder_cond_tail_vs_upstream", "snr_db")) for ep in eps])
    rows.append(["upstream exporter: `velocity.onnx`, int64 t rounded / truncated, tick audio"]
                + [f"{get(ue[ep], 'velocity_int64_t_int_round_chain_audio', 'snr_db')} / {get(ue[ep], 'velocity_int64_t_int_trunc_chain_audio', 'snr_db')}"
                   for ep in eps])
    rows.append(["upstream exporter: `encoder.onnx` output NaN at f0 = 0"] + ["yes" if ue[ep].get("encoder_nan_at_f0_0") else "no" for ep in eps])
    out.append(table(["stage", "CPU", "CUDA", "DirectML"], rows))
    if gate:
        out.append(f"\nThe gate row mutes {round(100 * gate['muted_fraction'])} % of the decoded audio (threshold {gate['threshold_db']:g} dBFS, "
                   f"the median frame volume).")

    mv = {ep: load(f"meanvc_{ep}.json") for ep in ("cpu", "cuda")}
    names = {"fbank_utterance": "fbank (whole utterance) vs torchaudio Kaldi", "asr_12_calls": "asr: 12 streaming calls (bn)",
             "asr_att_cache_after": "asr: attention cache after 12 calls", "gtm": "gtm",
             "dit_5_calls": "dit (bare, 5 streaming calls, fixed cache vs upstream's growing one)", "chain_bn_up": "chain: upsampled BN (69 chunks)",
             "chain_mel": "chain: mel (552 DiT calls)", "chain_audio": "chain: output audio (11 s)", "vocos_24_frames": "vocos alone (24 frames)"}
    out.append("\n*MeanVC2* (upstream `process_file` on the whole 11 s utterance, 2 ODE steps):\n")
    rows = [[label] + [snr(get(mv[ep], "parity", k)) for ep in ("cpu", "cuda")] for k, label in names.items()]
    rows.append(["crossfade `linspace` bit-exact with torch"] + ["yes" if get(mv[ep], "parity", "crossfade_linspace_bitexact", "pass") else "no"
                                                              for ep in ("cpu", "cuda")])
    out.append(table(["stage", "CPU", "CUDA"], rows))
    return "\n".join(out)


def join_md():
    sv = {ep: load(f"seed_vc_{ep}.json") for ep in ("cpu", "cuda")}
    labels = {"upstream": "upstream: fresh z, harmonic phases and sine noise every tick",
              "z_anchored": "z and sine noise anchored on the mel-frame grid",
              "z_and_phase": "… and the NSF phase anchored at the join"}
    joins = {ep: get(sv[ep], "variants", "fp32", "stream_join") for ep in sv}
    ep = "cuda" if joins["cuda"] else "cpu"
    sj = joins[ep]
    if not sj:
        return "(no stream_join in the reports)"
    setup = sj["setup"]
    rows = []
    for m, label in labels.items():
        r = sj[m]
        offs = ", ".join(str(j["best_offset"]) for j in r["joins"])
        rows.append([label, r["mean_mel_overlap_snr_db"], f"{r['mean_best_corr']} ({offs})", r["mean_corr_at_expected_offset"],
                     r["mean_sola_range_best_corr"]])
    head = (f"*Seed-VC joins* ({'CUDA' if ep == 'cuda' else 'CPU'} fp32, `{setup['preset']}`, {setup['joins']} consecutive joins from "
            f"{setup['window_start_s']:g} s of `jfk.wav`; the content moves {setup['frames_per_block']} mel frames per block, so the grid "
            f"advances {setup['grid_frames_per_tick']} frames and the ticks line up {setup['expected_offset']} samples from upstream's "
            f"join):\n")
    out = head + "\n" + table(["levers", "mel overlap SNR (dB)", "best correlation (offsets)", "at the grid offset", "best in SOLA's 0…441"], rows)
    other = joins["cpu"] if ep == "cuda" else None
    if other:
        d = max(abs(other[m][k] - sj[m][k]) for m in labels for k in ("mean_best_corr", "mean_corr_at_expected_offset", "mean_sola_range_best_corr"))
        out += f"\n\nThe CPU fp32 run gives the same means to within {d:.3f}."
    return out


# ---------------------------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------------------------

def cost_md():
    sv = {ep: load(f"seed_vc_{ep}.json") for ep in ("cpu", "cuda", "dml")}
    out = ["*Seed-VC* (ms per call at each preset's shapes; `g` = replayed CUDA graph, `b` = IOBinding, plain = host tensors; "
           "tick = xlsr + lr + steps × step + vocoder, with `hift` over the whole tick or `hift_f0` + `hift_nsf` over the returned "
           "frames plus 12 frames of context):\n"]
    rows = []
    for p in ("upstream_readme", "sigil_low", "sigil_lean"):
        for label, ep, v in (("CUDA fp32", "cuda", "fp32"), ("CUDA hift16", "cuda", "hift16"), ("CUDA fp16 (all)", "cuda", "fp16"),
                             ("DirectML fp32", "dml", "fp32"), ("CPU fp32, 2 threads", "cpu", "fp32")):
            rr = get(sv[ep], "variants", v, "cost", p)
            if not rr:
                continue
            sk = "dit_step" if "dit_step" in rr else "dit_step1"
            t_full = min(rr["tick"].values(), key=lambda t: t["tick_ms"])
            t_part = min(rr["tick_partial"].values(), key=lambda t: t["tick_ms"])
            rows.append([f"`{p}` ({rr['block_ms']:.0f} ms, {rr['preset']['steps']})", label, fmt(rr["xlsr"]), fmt(rr["lr"]), fmt(rr[sk]),
                         fmt(rr["hift"]), f"{fmt(rr['hift_f0'])} + {fmt(rr['hift_nsf'])}", t_full["tick_ms"], t_part["tick_ms"],
                         f"{t_full['rtf']} / {t_part['rtf']}"])
    out.append(table(["preset (block, steps)", "EP / variant", "xlsr", "lr", "step", "hift", "hift_f0 + hift_nsf", "tick, hift",
                      "tick, partial", "RTF"], rows))
    if any(REFUSED in str(c) for r in rows for c in r):
        out.append("\n" + REFUSED_NOTE)
    o, oc = get(sv["cuda"], "variants", "fp32", "cost", "once_per_voice"), get(sv["cpu"], "variants", "fp32", "cost", "once_per_voice")
    if o and oc:
        out.append(f"\nOnce per voice (3 s reference): CUDA fp32 xlsr {fmt(o['xlsr'])}, campplus {fmt(o['campplus'])}, mel {fmt(o['mel'])} ms; "
                   f"CPU xlsr {fmt(oc['xlsr'])}, campplus {fmt(oc['campplus'])}, mel {fmt(oc['mel'])} ms.")

    dd = {ep: load(f"ddsp_svc_{ep}.json") for ep in ("cpu", "cuda", "dml")}
    out.append("\n*DDSP-SVC 6.3* (tick = 2 × ContentVec (tta2x) + ddsp + steps × reflow_step + vocoder; F0 extraction and resampling "
               "not included):\n")
    rows = []
    for p in ("upstream_gui", "sigil_singing", "sigil_singing_fast"):
        for label, ep in (("CUDA", "cuda"), ("DirectML", "dml"), ("CPU, 2 threads", "cpu")):
            rr = get(dd[ep], "cost", p)
            if not rr:
                continue
            tick = min(rr["tick"].values(), key=lambda t: t["tick_ms"])
            rows.append([f"`{p}` ({rr['block_ms']:.0f} ms, {rr['preset']['steps']})", label, fmt(rr["contentvec"]),
                         fmt(rr["ddsp"]), fmt(rr["reflow_step"]), fmt(rr["vocoder"]), tick["tick_ms"], tick["rtf"]])
    out.append(table(["preset (block, steps)", "EP", "ContentVec", "ddsp", "step", "vocoder", "tick", "RTF"], rows))
    if any(REFUSED in str(c) for r in rows for c in r):
        out.append("\n" + REFUSED_NOTE)

    mv = {ep: load(f"meanvc_{ep}.json") for ep in ("cpu", "cuda")}
    out.append("\n*MeanVC2* (per 40 ms output chunk = fbank/4 + asr/2 + steps × dit_step + vocos):\n")
    rows = []
    for label, ep, th in (("CPU, 1 thread", "cpu", "threads1"), ("CPU, 2 threads", "cpu", "threads2"), ("CUDA", "cuda", "threads2")):
        rr = get(mv[ep], "cost", th)
        if not rr:
            continue
        pc = rr["per_chunk"]
        b1 = min((v for k, v in pc.items() if k.endswith("steps1")), key=lambda t: t["per_40ms_chunk_ms"])
        b2 = min((v for k, v in pc.items() if k.endswith("steps2")), key=lambda t: t["per_40ms_chunk_ms"])
        rows.append([label, fmt(rr["fbank"]), fmt(rr["asr"]), fmt(rr["dit_step"]), fmt(rr["vocos"]),
                     f"{b1['per_40ms_chunk_ms']} ({b1['rtf']})", f"{b2['per_40ms_chunk_ms']} ({b2['rtf']})"])
    out.append(table(["EP", "fbank", "asr", "dit_step", "vocos", "1 step: ms (RTF)", "2 steps: ms (RTF)"], rows))
    hb = get(mv["cuda"], "cost", "threads2", "per_chunk", "bound_steps2")
    if hb:
        out.append(f"\nMeanVC2 on CUDA without graphs (IOBinding, 2 steps): {hb['per_40ms_chunk_ms']} ms per chunk (RTF {hb['rtf']}).")
    return "\n".join(out)


def t10_md():
    """Sigil-ZS against T10 (p95 <= 300 ms end to end on H1 idle), from the CUDA fp32 cost rows."""
    sv = load("seed_vc_cuda.json")
    rows = []
    for p in ("upstream_readme", "sigil_low", "sigil_lean"):
        rr = get(sv, "variants", "fp32", "cost", p)
        if not rr:
            continue
        pr = rr["preset"]
        sh = rr["shapes"]
        block = pr["block"] * 1000
        cf_units = min(round(pr["crossfade"] * 50), 4)
        algo = block + sh["skip_tail"] * 20 + cf_units * 20 + 20                      # + right context + SOLA buffer + SOLA search (worst)
        sk = "dit_step" if "dit_step" in rr else "dit_step1"
        fixed = conservative(rr["xlsr"]) + conservative(rr["lr"])
        tick_full = fixed + pr["steps"] * conservative(rr[sk]) + conservative(rr["hift"])
        tick_part = fixed + pr["steps"] * conservative(rr[sk]) + conservative(rr["hift_f0"]) + conservative(rr["hift_nsf"])
        cells = [f"`{p}`", f"{block:.0f} + {sh['skip_tail'] * 20} + {cf_units * 20} + 20 = {algo:.0f}",
                 f"{tick_full:.1f} / {tick_part:.1f}"]
        for fct in H1_FACTORS:
            tot = algo + fct * tick_part
            cells.append(f"{tot:.0f} ({T10_P95_MS - tot:+.0f})")
        rows.append(cells)
    head = ("*Sigil-ZS against T10* (p95 ≤ 300 ms end to end on H1 idle). Algorithmic latency, worst case: block + right context + "
            "SOLA crossfade + SOLA search. Tick on the 4090: per call the larger of the kept block's p90 and the slowest block's median, "
            f"CUDA fp32 (graphs where captured), whole-tick `hift` / partial decode. H1 = the 4090 tick × {H1_FACTORS[0]} (the "
            f"assumption) or × {H1_FACTORS[1]} (pessimistic); the bracket is what is left of 300 ms for the terms not counted (see "
            "below):\n")
    return head + "\n" + table(["preset", "algorithmic (ms)", "4090 tick, conservative (ms)",
                                f"H1 ×{H1_FACTORS[0]}, partial (left of 300)", f"H1 ×{H1_FACTORS[1]}, partial (left of 300)"], rows)


def profile_md():
    dd = load("ddsp_svc_dml.json")
    prof = get(dd, "profile")
    if not prof:
        return "(no DirectML profile in reports/ddsp_svc_dml.json)"
    rows = []
    for p in ("upstream_gui", "sigil_singing", "sigil_singing_fast"):
        for g in ("ddsp", "contentvec", "reflow_step", "vocoder"):
            r = get(prof, p, g)
            if not r:
                continue
            nodes = ", ".join(f"{k.replace('ExecutionProvider', '')} {v}" for k, v in sorted(r["nodes_by_provider"].items()))
            refused = get(r, "session", "dims_refused")
            gpu = get(r, "gpu_util_pct_during", "median") if (get(r, "gpu_util_pct_during", "samples") or 0) >= 4 else None
            rows.append([f"`{p}`", g, f"{r['wall_ms_per_run']:.1f}", f"{r['kernel_ms_per_run']:.2f}",
                         f"{r['process_cpu_ms_per_run']:.0f}" if "process_cpu_ms_per_run" in r else "—",
                         "—" if gpu is None else f"{gpu:g}", nodes,
                         "no (refused)" if refused else ("yes" if get(r, "session", "dims_pinned") else "no")])
    head = (f"*DDSP-SVC on DirectML, ORT's own profile* (`ddsp_svc/parity.py --ep dml --profile`; median of "
            f"{get(prof, 'upstream_gui', 'ddsp', 'runs')} runs after 2 warm-up runs. Node time = the summed per-node times ORT records, "
            "which on DirectML is the CPU-side dispatch of each node, not its GPU execution; process CPU = this process's CPU time per "
            "run; GPU = median utilisation sampled every 0.25 s during the runs, shown when the runs last long enough for 4 "
            "samples):\n")
    out = head + "\n" + table(["preset", "graph", "wall (ms)", "node time (ms)", "process CPU (ms)", "GPU (%)", "nodes per EP",
                               "dims pinned"], rows)
    cpu_ops = get(prof, "upstream_gui", "ddsp", "cpu_placed_ops")
    if cpu_ops:
        cpu_ms = get(prof, "upstream_gui", "ddsp", "kernel_ms_per_run_by_provider", "CPUExecutionProvider")
        out += ("\n\nOps the DirectML session placed on the CPU (`ddsp`): " + ", ".join(f"{k} × {v}" for k, v in cpu_ops.items())
                + (f", {cpu_ms:.2f} ms per run in all." if cpu_ms is not None else "."))
    return out


BLOCKS = {"parity": parity_md, "join": join_md, "cost": cost_md, "t10": t10_md, "profile": profile_md}


def main():
    md = {k: f() for k, f in BLOCKS.items()}
    if "--write" not in sys.argv:
        for k, v in md.items():
            print(f"<!-- tables:{k} -->\n{v}\n<!-- /tables:{k} -->\n")
        return
    with open(ADR, encoding="utf-8") as f:
        s = f.read()
    for k, v in md.items():
        pat = re.compile(rf"(<!-- tables:{k} -->\n)(.*?)(<!-- /tables:{k} -->)", re.S)
        if not pat.search(s):
            raise SystemExit(f"{ADR}: no <!-- tables:{k} --> block")
        s = pat.sub(lambda m, v=v: m.group(1) + v + "\n" + m.group(3), s)
    with open(ADR, "w", encoding="utf-8", newline="\n") as f:
        f.write(s)
    print("wrote", os.path.normpath(ADR))


if __name__ == "__main__":
    main()
