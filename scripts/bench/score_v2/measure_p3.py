# SPDX-License-Identifier: AGPL-3.0-or-later
"""P3B-B measure-first (rule R1): scorer v2 prototypes on the saved P3 stream / ceiling WAV pairs.

CPU only (ONNX Runtime CPU EP, NumPy); no GPU, nothing timed. For every saved quality run under
``--root`` (default: the P3 final gate and lookahead runs of the session scratchpad):

(a) CAM++: the run's v1 cosine (report) next to this replica's legacy cosine, the exact export's
    cosine over the same region, and the exact export on speech frames only (frames within 40 dB of
    the ceiling's loudest, the same frames in both renditions); plus the identical-stream pairs
    across builds, where only the ceiling's near-silence differs.
(b) F0: the run's RMSE split into fine pitch error (non-gross voiced frames) and gross frames
    (count, positions, size), with the replica's whole-clip RMVPE analyzer.
(c) Clicks: the v1 statistic and the conditional binomial (97.5 %) from the saved join details.
(d) MCD: the STFT MCD replica next to the envelope MCD, and the different-seed ceiling floor (the
    same clip's ceilings from runs with a different lead, so different anchored-noise frames on the
    same 200 ms block phase).

Usage: ``python scripts/bench/score_v2/measure_p3.py --scratch <scratchpad> [--out result.json]``
(run it with ``sigil-dev-cache/venv-ort128-cuda``'s Python: ONNX Runtime 1.28, the engine's).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as cm  # noqa: E402

LEGACY = os.path.join(cm.DEV_CACHE, "upstream-assets", "campplus", "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx")
EXACT = os.path.join(cm.DEV_CACHE, "vc-assets", "bench", "campplus-exact", "model.onnx")


def session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.intra_op_num_threads = 2
    so.log_severity_level = 3
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def fbank(x16):
    import torch
    import torchaudio

    return torchaudio.compliance.kaldi.fbank(torch.from_numpy(np.ascontiguousarray(x16, np.float32))[None] * 32768.0,
                                             num_mel_bins=80, sample_frequency=16000, dither=0.0).numpy()


def frame_energy_db(x16):
    starts = cm.frame_starts(len(x16), 400, 160)
    return np.array([10 * np.log10(np.mean(np.asarray(x16[s:s + 400], np.float64) ** 2) + 1e-12) for s in starts])


def embed(sess, fb, mask=None):
    f = fb if mask is None else fb[mask[: len(fb)]]
    f = f - f.mean(0, keepdims=True)
    e = sess.run(None, {"x": f[None].astype(np.float32)})[0].ravel().astype(np.float64)
    return e / np.linalg.norm(e)


def speaker_rows(legacy, exact, cs, ss):
    fc, fs = fbank(cs), fbank(ss)
    n = min(len(fc), len(fs))
    fc, fs = fc[:n], fs[:n]
    en = frame_energy_db(cs)[:n]
    mask = en >= en.max() - 40.0
    return {
        "legacy_full": float(embed(legacy, fc) @ embed(legacy, fs)),
        "exact_full": float(embed(exact, fc) @ embed(exact, fs)),
        "exact_speech": float(embed(exact, fc, mask) @ embed(exact, fs, mask)),
        "legacy_speech": float(embed(legacy, fc, mask) @ embed(legacy, fs, mask)),
        "speech_frames": int(mask.sum()), "frames": int(n),
    }


def clicks(details):
    j = details.get("joins") or {}
    at, cov = j.get("artifacts_in_windows"), j.get("window_coverage")
    total = (details.get("quality") or {}).get("artifact_events")
    if at is None or cov is None or total is None:
        return None
    exp = j.get("expected_from_off_join_rate", 0.0)
    v1 = max(0.0, round(at - exp - 2 * np.sqrt(exp)))
    q = cm.binom_quantile(0.975, int(total), float(cov))
    from scipy.stats import binom

    p_tail = float(binom.sf(at - 1, int(total), float(cov)))
    return {"events": int(total), "in_windows": int(at), "coverage": float(cov), "v1_clicks": v1,
            "binomial_q975": q, "binomial_clicks": max(0, int(at) - q), "p_at_least": p_tail,
            "artifact_phase_clicks": (details.get("artifact_phase") or {}).get("clicks")}


def score_run(run_dir, legacy, exact):
    r = cm.load_run(run_dir)
    m, d = r["row"]["metrics"], r["row"]["details"]
    cs, ss = r["cs"], r["ss"]
    out = {"run": run_dir, "lead_s": d.get("lead_s"), "report": {k: m.get(k) for k in
           ("speaker_cosine", "mcd_db", "f0_rmse_cents", "gross_pitch_error", "vuv_error", "clicks", "dnsmos_ovrl")}}
    out["speaker"] = speaker_rows(legacy, exact, cs, ss)
    out["mcd_replica"] = cm.mcd_db(cs, ss)
    fc, fs = cm.rmvpe_f0(cs), cm.rmvpe_f0(ss)
    mask = cm.speech_frames(cs, 40.0)
    n = min(len(fc), len(fs), len(mask))
    out["f0"] = cm.f0_errors_v2(fc[:n], fs[:n], mask[:n])
    out["envelope_mcd"] = cm.envelope_mcd_db(cs, ss, fc)
    out["clicks"] = clicks(d)
    return out, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--quick", action="store_true", help="only the gate-after rows and the identical-stream pairs")
    args = ap.parse_args()
    cm.SCRATCH = args.scratch
    legacy, exact = session(LEGACY), session(EXACT)
    sc = args.scratch
    dirs = sorted(glob.glob(os.path.join(sc, "p3gate", "gate-after", "sve-*"))) + sorted(glob.glob(os.path.join(sc, "p3gate", "gate-before", "sve-*")))
    if not args.quick:
        dirs += sorted(glob.glob(os.path.join(sc, "p3gate", "fid", "*"))) + sorted(glob.glob(os.path.join(sc, "p3i0", "la", "*")))
    dirs = [x for x in dirs if os.path.isfile(os.path.join(x, "fidelity.ceiling.wav"))]
    rows, runs = [], {}
    for dr in dirs:
        try:
            row, r = score_run(dr, legacy, exact)
        except Exception as e:  # a run without WAVs or with another layout
            print(f"skip {dr}: {e}")
            continue
        rows.append(row)
        runs[dr] = r
        sp, f0 = row["speaker"], row["f0"]
        cl = row["clicks"] or {}
        print(f"{os.path.relpath(dr, sc):45s} cos v1 {row['report']['speaker_cosine']:.4f} (replica {sp['legacy_full']:.4f}) "
              f"exact {sp['exact_full']:.4f} exact/speech {sp['exact_speech']:.4f} | MCD {row['report']['mcd_db']:.3f} "
              f"(replica {row['mcd_replica']:.3f}) env {row['envelope_mcd']:.3f} | F0 {row['report']['f0_rmse_cents']:.1f} c -> "
              f"fine {f0['fine_rmse_cents']:.1f} c + {f0['gross_frames']} gross {[round(c) for c in f0['gross_cents']]} | "
              f"clicks v1 {cl.get('v1_clicks')} binom {cl.get('binomial_clicks')} ({cl.get('in_windows')}/{cl.get('events')}, "
              f"cov {cl.get('coverage', 0):.2f})")
    # Identical streams across builds (the ceiling re-rendered): what moves the cosine.
    ident = []
    for name in ("sve-dml", "sve-cpu", "sve-cuda"):
        a = os.path.join(sc, "p3i0", "gate-after", name)
        b = os.path.join(sc, "p3gate", "gate-before", name)
        if not (os.path.isdir(a) and os.path.isdir(b)):
            continue
        ra, rb = cm.load_run(a), cm.load_run(b)
        same_stream = bool(np.array_equal(ra["stream"], rb["stream"]))
        diff = np.abs(ra["cs"] - rb["cs"]) if len(ra["cs"]) == len(rb["cs"]) else None
        e = {"name": name, "stream_identical": same_stream,
             "ceiling_region_max_abs_diff": float(diff.max()) if diff is not None else None,
             "a": speaker_rows(legacy, exact, ra["cs"], ra["ss"]), "b": speaker_rows(legacy, exact, rb["cs"], rb["ss"]),
             "report_a": ra["row"]["metrics"].get("speaker_cosine"), "report_b": rb["row"]["metrics"].get("speaker_cosine")}
        if diff is not None:
            loud = np.array([np.sqrt(np.mean(ra["cs"][i:i + 160].astype(np.float64) ** 2)) for i in range(0, len(diff), 160)])
            dd = np.array([diff[i:i + 160].max() for i in range(0, len(diff), 160)])
            e["differing_10ms_frames"] = int((dd > 0).sum())
            e["differing_frames_max_rms_dbfs"] = float(20 * np.log10(loud[dd > 0].max() + 1e-12)) if (dd > 0).any() else None
        ident.append(e)
        print(f"identical stream {name}: stream identical {same_stream}, ceiling diff max {e['ceiling_region_max_abs_diff']}, "
              f"in {e.get('differing_10ms_frames')} frames up to {e.get('differing_frames_max_rms_dbfs')} dBFS; cos v1 report "
              f"{e['report_a']:.4f} vs {e['report_b']:.4f}; exact/speech {e['a']['exact_speech']:.6f} vs {e['b']['exact_speech']:.6f}")
    # Different-seed floor: the lead-1 and lead-4 ceilings of the same configuration (same lag).
    floors = []
    for one, four in (("sve-cuda-la20", "sve-cuda-la20-lead4"),):
        p1, p4 = os.path.join(sc, "p3i0", "la", one), os.path.join(sc, "p3i0", "la", four)
        if p1 in runs and p4 in runs:
            r1, r4 = runs[p1], runs[p4]
            if r1["lag"] == r4["lag"] and r1["region"] == r4["region"]:
                a, b = r1["region"]
                c1, c4 = r1["c16"][a:b], r4["c16"][a:b]
                f1, f4 = cm.rmvpe_f0(c1), cm.rmvpe_f0(c4)
                fl = {"pair": [one, four], "stft_mcd": cm.mcd_db(c1, c4), "envelope_mcd": cm.envelope_mcd_db(c1, c4, f1),
                      "stream_stft_mcd": [rows[[x["run"] for x in rows].index(p1)]["mcd_replica"], rows[[x["run"] for x in rows].index(p4)]["mcd_replica"]],
                      "stream_envelope_mcd": [rows[[x["run"] for x in rows].index(p1)]["envelope_mcd"], rows[[x["run"] for x in rows].index(p4)]["envelope_mcd"]]}
                floors.append(fl)
                print(f"different-seed ceiling floor {one} vs {four}: STFT MCD {fl['stft_mcd']:.3f} dB, envelope MCD {fl['envelope_mcd']:.3f} dB "
                      f"(stream vs ceiling: STFT {fl['stream_stft_mcd']}, envelope {fl['stream_envelope_mcd']})")
            else:
                print(f"floor {one}/{four}: lag or region differ ({r1['lag']} {r4['lag']} {r1['region']} {r4['region']})")
    # (c) on every saved report (the WAVs are not needed): the binomial against the v1 statistic.
    click_rows = []
    for rep_path in sorted(glob.glob(os.path.join(sc, "p3gate", "*", "*", "report.json")) + glob.glob(os.path.join(sc, "p3i0", "*", "*", "report.json"))):
        try:
            rep = json.load(open(rep_path, encoding="utf-8"))
        except Exception:
            continue
        for run in rep.get("runs", []):
            if run.get("mode") != "quality":
                continue
            c = clicks(run.get("details") or {})
            if c:
                c["run"] = os.path.relpath(os.path.dirname(rep_path), sc)
                c["lead_s"] = (run.get("details") or {}).get("lead_s")
                click_rows.append(c)
                if c["v1_clicks"] or c["binomial_clicks"]:
                    print(f"clicks {c['run']:40s} lead {c['lead_s']}: {c['in_windows']}/{c['events']} in windows (coverage {c['coverage']:.2f}): "
                          f"v1 {c['v1_clicks']:.0f}, binomial {c['binomial_clicks']} (P(X >= {c['in_windows']}) = {c['p_at_least']:.4f}), "
                          f"phase-locked {c['artifact_phase_clicks']}")
    print(f"clicks: {len(click_rows)} saved quality runs; v1 nonzero on {sum(1 for c in click_rows if c['v1_clicks'])}, "
          f"binomial nonzero on {sum(1 for c in click_rows if c['binomial_clicks'])}")
    result = {"rows": rows, "identical_streams": ident, "different_seed_floors": floors, "clicks": click_rows}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1, default=float)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
