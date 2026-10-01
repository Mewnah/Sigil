#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MeanVC2 (40 ms model): offline parity of the exported graphs against upstream PyTorch, and their
per-call cost (P3-H, ADR 0006).

Reference: upstream ``runtime/run_rt.py`` ``VCRunner.process_file`` run unchanged in PyTorch fp32 on
the CPU (its ``__init__`` too; only the speaker-embedding extractor is replaced by a fixed
embedding, since the WavLM + ECAPA checkpoint is a manual Google Drive download), with every random
draw recorded (``common.recorded_random``) and every stage observed (``_encode_chunk``, ``_vc_step``,
``_decode_mel`` outputs). Candidate: ``sampler.SigilMeanVC`` on the ONNX graphs, fed the same audio
(``jfk.wav``, 11 s) and the same draws. Gate: SNR >= 35 dB on the output audio and on every stage.

The speaker embedding is a seeded random unit vector (``--spk-seed``): the parity of the graphs does
not depend on which voice is chosen, but listening quality does, so no quality claim is made here.

Cost: per graph at the streaming shapes (fbank over 2880 samples, one ASR call, one DiT step, Vocos
over 6 mel frames), and per 40 ms output chunk = fbank/4 + asr/2 + steps x dit_step + vocos, which is
what upstream spends per 160 ms audio chunk divided by 4. CPU at 1 and 2 intra-op threads (the
real-time policy); CUDA as host tensors, IOBinding and a captured CUDA graph.

Every parity and cost section records the sha256 of the graphs it measured (``graphs``).

Usage (repo root):
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-cuda/Scripts/python scripts/vc-families/meanvc/parity.py   # CPU EP of ORT 1.28
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-cuda/Scripts/python scripts/vc-families/meanvc/parity.py --ep cuda
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-dml/Scripts/python scripts/vc-families/meanvc/parity.py --ep dml
Writes ``reports/meanvc_<ep>.json``.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import common as cm  # noqa: E402
import export as ex  # noqa: E402
import sampler as sp  # noqa: E402

SOURCE_WAV = "jfk.wav"
GRAPHS = ("fbank", "asr", "gtm", "dit", "dit_step", "vocos")


def ex_cache_shape(sess):
    """The fixed KV-cache shape of a DiT graph ([layers, key/value, heads, frames, dim])."""
    return tuple(int(d) for d in next(i for i in sess.get_inputs() if i.name == "cache").shape)


def speaker_embedding(seed):
    v = np.random.default_rng(seed).standard_normal((1, 256)).astype(np.float64)
    return (v / np.linalg.norm(v)).astype(np.float32)


def read_source():
    """The input exactly as ``process_file`` reads it (soundfile float64 -> mono -> float32)."""
    import soundfile as sf
    path = cm.need(os.path.join(cm.SAMPLES, SOURCE_WAV), "the STT speech samples (models/transcribe/samples)")
    wav, sr = sf.read(path)
    if wav.ndim == 2:
        wav = np.mean(wav, axis=1)
    if sr != 16000:
        import librosa
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    return path, wav.astype(np.float32)


# ---------------------------------------------------------------------------------------------
# Upstream reference
# ---------------------------------------------------------------------------------------------

def upstream_runner(spks):
    """``VCRunner(model="40ms")`` built by its own ``__init__``: the checkpoint paths point at the dev
    cache (and the torch 2.3 copy of the ASR archive), the speaker extractor returns ``spks``."""
    import run_rt
    ex.load_asr()  # writes the patched archive if needed
    paths = run_rt.MODEL_PATHS["40ms"]
    paths["ckpt"] = ex.DIT_CKPT
    paths["config"] = ex.CONFIG
    paths["asr_ckpt"] = os.path.join(ex.OUT, "_torch23", "fastu2pp_80ms.pt")
    run_rt.VOCODER_PATH = ex.VOCOS_CKPT
    run_rt.init_speaker_model = lambda *a, **k: None
    run_rt.extract_embedding = lambda *a, **k: torch.from_numpy(spks)
    threads = torch.get_num_threads()
    runner = run_rt.VCRunner(target_wav="(fixed embedding)", device="cpu", model="40ms")
    torch.set_num_threads(threads)  # __init__ pins torch to 1 thread; values do not depend on it
    return runner


def run_upstream(runner, path, seed):
    """``process_file`` unchanged; stage outputs observed through instance wrappers, the result
    captured from its ``soundfile.write`` call (the file itself would be 16-bit PCM)."""
    import soundfile
    trace = {"bn_up": [], "mel": [], "wav": []}

    def observe(name, key):
        fn = getattr(runner, name)

        def g(*a, **k):
            out = fn(*a, **k)
            if out is not None:
                trace[key].append(out.detach().numpy().copy() if torch.is_tensor(out) else np.array(out, copy=True))
            return out
        setattr(runner, name, g)

    observe("_encode_chunk", "bn_up")
    observe("_vc_step", "mel")
    observe("_decode_mel", "wav")
    captured = {}
    orig_write = soundfile.write
    soundfile.write = lambda file, data, *a, **k: captured.setdefault("wav", np.asarray(data, np.float32).copy())
    try:
        with torch.no_grad(), cm.recorded_random(seed) as rec:
            runner.process_file(path, os.path.join(ex.OUT, "_unused.wav"), seed=seed)
    finally:
        soundfile.write = orig_write
        for name in ("_encode_chunk", "_vc_step", "_decode_mel"):
            delattr(runner, name)
    # _vc_step's noise; other randn calls in the record (meta-device placeholders some torch
    # internals draw during first use) are not part of the computation
    draws = [t.numpy().copy() for k, t in rec if k == "randn" and t.device.type == "cpu" and tuple(t.shape) == (1, 8, 80)]
    report_n = sum(1 for k, t in rec if k == "randn")
    if len(draws) != len(trace["mel"]):
        raise SystemExit(f"recorded {len(draws)} noise draws for {len(trace['mel'])} DiT chunks ({report_n} randn calls)")
    return captured["wav"], trace, draws


# ---------------------------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------------------------

def parity(args, sessions, up, report, files=None):
    import torchaudio.compliance.kaldi as kaldi
    spks = speaker_embedding(args.spk_seed)
    path, wav = read_source()
    rows = {}

    # isolated graphs
    fb_ref = kaldi.fbank(torch.from_numpy(wav * 32768)[None], frame_length=25, frame_shift=10, snip_edges=True,
                         num_mel_bins=80, energy_floor=0.0, dither=0.0, sample_frequency=16000).numpy()
    rows["fbank_utterance"] = cm.gate(cm.compare(fb_ref, cm.run(sessions["fbank"], {"wav16k": wav[None]})[0][0]))
    att = torch.zeros(ex.ATT_CACHE)
    cnn = torch.zeros(ex.CNN_CACHE)
    att_o, cnn_o = att.numpy(), cnn.numpy()
    bn_ref, bn_got = [], []
    for i in range(12):
        fb = torch.from_numpy(fb_ref[i * 8: i * 8 + 11][None].copy())
        off = ex.ASR_OFFSET_INIT + ex.ASR_OFFSET_STEP * i
        with torch.no_grad():
            o, att, cnn = up["asr"](fb, torch.tensor(off), torch.tensor(ex.RCS), att, cnn)
        g, att_o, cnn_o = cm.run(sessions["asr"], {"fbank": fb.numpy(), "offset": np.array(off, np.int64), "rcs": np.array(ex.RCS, np.int64),
                                                   "att_cache": att_o, "cnn_cache": cnn_o})
        bn_ref.append(o.numpy())
        bn_got.append(g)
    rows["asr_12_calls"] = cm.gate(cm.compare(np.concatenate(bn_ref, 1), np.concatenate(bn_got, 1)))
    rows["asr_att_cache_after"] = cm.gate(cm.compare(att.numpy(), att_o))
    with torch.no_grad():
        k_ref, v_ref = up["dit"].gtm(torch.from_numpy(spks))
    k_got, v_got = cm.run(sessions["gtm"], {"spks": spks})
    rows["gtm"] = cm.gate(cm.compare(np.concatenate([k_ref.numpy(), v_ref.numpy()]), np.concatenate([k_got, v_got])))
    # the bare DiT graph (host-side update) over five streaming calls: upstream's growing KV cache vs
    # the fixed padded one, fed each call's cache from the previous call of the same side
    g = torch.Generator().manual_seed(7)
    kv, cache, u_ref, u_got = None, np.zeros(ex_cache_shape(sessions["dit"]), np.float32), [], []
    for call in range(5):
        x, cond = torch.randn(1, 8, 80, generator=g), torch.randn(1, 8, 256, generator=g)
        offset = ex.CHUNK * call
        with torch.no_grad():
            u, kv = up["dit"](x, torch.tensor([1.0]), torch.tensor([0.5]), cache=None, cond=cond, spks=torch.from_numpy(spks),
                              offset=offset, is_inference=True, kv_cache=kv, gtm_kv=(k_ref, v_ref))
        o, cache = cm.run(sessions["dit"], {"x": x.numpy(), "t": np.array([1.0], np.float32), "r": np.array([0.5], np.float32),
                                            "cond": cond.numpy(), "spks": spks, "k_mem": k_got, "v_mem": v_got,
                                            "offset": np.array([offset], np.int64), "n_valid": np.array([min(offset, ex.CAP)], np.int64),
                                            "cache": cache})
        u_ref.append(u.numpy())
        u_got.append(o)
    rows["dit_5_calls"] = cm.gate(cm.compare(np.concatenate(u_ref), np.concatenate(u_got)))

    # the whole upstream run and the Sigil sampler on the same draws
    runner = upstream_runner(spks)
    out_u, tr_u, draws = run_upstream(runner, path, args.seed)
    zs = sp.SigilMeanVC(sessions, spks, steps=2)
    out_s = zs.convert(wav, draws)
    tr_s = zs.trace
    rows["chain_bn_up"] = cm.gate(cm.compare(np.concatenate(tr_u["bn_up"], 1), np.concatenate(tr_s["bn_up"], 1)))
    rows["chain_mel"] = cm.gate(cm.compare(np.concatenate(tr_u["mel"], -1), np.concatenate(tr_s["mel"], -1)))
    rows["chain_audio"] = cm.gate(cm.compare_audio(out_u, out_s, 16000))
    # every intermediate call of the chain, isolated: the DiT step and Vocos on upstream's inputs
    with torch.no_grad():
        mel_ref = torch.from_numpy(np.concatenate(tr_u["mel"][:6], -1))
        voc_ref = up["vocos_jit"].decode((mel_ref + 1) / 2).numpy()
    rows["vocos_24_frames"] = cm.gate(cm.compare_audio(voc_ref.ravel(), cm.run(sessions["vocos"], {"mel": ((mel_ref.numpy() + 1) / 2)})[0].ravel(), 16000))
    lin = sp.linspace_f32(1.0, 0.0, sp.WAV_OVERLAP)
    rows["crossfade_linspace_bitexact"] = {"equal": bool(np.array_equal(lin, torch.linspace(1, 0, steps=sp.WAV_OVERLAP).numpy())),
                                           "pass": bool(np.array_equal(lin, torch.linspace(1, 0, steps=sp.WAV_OVERLAP).numpy()))}
    report["parity"] = rows
    report["parity_setup"] = {"source": SOURCE_WAV, "seconds": round(len(wav) / 16000, 2), "seed": args.seed,
                              "speaker": f"seeded random unit vector (numpy default_rng({args.spk_seed}))",
                              "reference_impl": "upstream run_rt.VCRunner.process_file, PyTorch fp32 CPU (40ms model, 2 ODE steps)",
                              "chunks": len(tr_u["bn_up"]), "dit_calls": len(tr_u["mel"]) * 2, "output_samples": int(out_u.size)}
    if files:
        report["parity_setup"]["graphs"] = cm.graph_stamp(files)
    del runner
    return report


# ---------------------------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------------------------

def cost(args, files, report):
    report["timing_conditions"] = cm.timing_conditions(args.ep)
    rng = np.random.default_rng(0)
    f = lambda *shape: rng.standard_normal(shape).astype(np.float32)  # noqa: E731
    spks = speaker_embedding(args.spk_seed)
    cache = f(4, 2, 2, ex.CAP, 64) * 0.5
    dit_feeds = {"x": f(1, 8, 80), "t": np.array([1.0], np.float32), "r": np.array([0.5], np.float32), "dt": np.array([0.5], np.float32),
                 "cond": f(1, 8, 256), "spks": spks, "k_mem": f(1, 32, 256), "v_mem": f(1, 32, 256),
                 "offset": np.array([400], np.int64), "n_valid": np.array([ex.CAP], np.int64), "cache": cache}
    calls = {
        "fbank": ({"N": 2880, "F": 16}, {"wav16k": f(1, 2880) * 0.1}),
        "asr": ({}, {"fbank": f(1, 11, 80) + 5.0, "offset": np.array(40, np.int64), "rcs": np.array(ex.RCS, np.int64),
                     "att_cache": f(*ex.ATT_CACHE) * 0.1, "cnn_cache": f(*ex.CNN_CACHE) * 0.1}),
        "dit_step": ({}, dit_feeds),
        "vocos": ({"M": 6, "S": 800}, {"mel": np.clip(f(1, 80, 6) * 0.2 + 0.5, 0, 1)}),
    }
    out = {}
    thread_list = args.threads_list if args.ep == "cpu" else [args.threads]
    for threads in thread_list:
        rows = {}
        for g, (dims, feeds) in calls.items():
            sess = cm.session(files[g], args.ep, threads, dims=dims)
            rows[g] = cm.timed_session_call(sess, feeds, args.ep, args.warm, args.runs, path=files[g], dims=dims, threads=threads)
            del sess
        per_chunk = {}
        for mode in ("host", "bound", "graph"):
            if not any(mode in r and "median_ms" in r[mode] for r in rows.values()):
                continue
            pick = lambda r, mode=mode: r[mode]["median_ms"] if mode in r and "median_ms" in r[mode] else cm.best_ms(r)[0]  # noqa: E731
            for steps in (1, 2):
                ms = pick(rows["fbank"]) / 4 + pick(rows["asr"]) / 2 + steps * pick(rows["dit_step"]) + pick(rows["vocos"])
                per_chunk[f"{mode}_steps{steps}"] = {"per_40ms_chunk_ms": round(ms, 3), "rtf": round(ms / 40.0, 3)}
        rows["per_chunk"] = per_chunk
        out[f"threads{threads}"] = rows
        print(f"[{args.ep} {threads} thr] " + "; ".join(f"{k} {v['per_40ms_chunk_ms']} ms (RTF {v['rtf']})" for k, v in per_chunk.items()))
    sess = cm.session(files["gtm"], args.ep, args.threads)
    out["once_per_voice"] = {"gtm": cm.timed_session_call(sess, {"spks": spks}, args.ep, 2, 10)}
    out["note"] = ("per 40 ms output chunk = fbank(2880 samples)/4 + asr/2 + steps x dit_step + vocos(6 frames); upstream "
                   "processes 160 ms audio chunks (2 ASR calls, 4 DiT chunks). The speaker encoder (WavLM-Large + ECAPA) "
                   "runs once per reference voice and is not included.")
    out["graphs"] = cm.graph_stamp(files)
    report["cost"] = out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", default="cpu", choices=sorted(cm.EPS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--threads-list", default="1,2", help="CPU cost at each of these intra-op thread counts")
    ap.add_argument("--seed", type=int, default=42, help="upstream process_file's own default seed")
    ap.add_argument("--spk-seed", type=int, default=1234)
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--warm", type=int, default=10)
    ap.add_argument("--runs", type=int, default=100)
    args = ap.parse_args()
    args.threads_list = [int(x) for x in args.threads_list.split(",") if x.strip()]

    cm.import_ort(args.ep)
    files = {n: cm.need(os.path.join(ex.OUT, f"{n}.onnx"), "run export.py first") for n in GRAPHS}
    report = {"family": "meanvc", "model": "MeanVC2 40ms_40ms (HF ASLP-lab/MeanVC2 @39cdd195)", "env": cm.env_info(args.ep),
              "threads": args.threads, "files": {n: os.path.basename(p) for n, p in files.items()}}
    if not args.no_parity:
        up = ex.load_upstream()
        sessions = {n: cm.session(p, args.ep, args.threads) for n, p in files.items()}
        parity(args, sessions, up, report, files)
        bad = [k for k, r in report["parity"].items() if not r["pass"]]
        print(f"[{args.ep}] parity: {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}; audio {report['parity']['chain_audio']['snr_db']} dB")
        for k, r in report["parity"].items():
            print(cm.row_line(k, r, 28))
        del sessions, up
    if not args.no_time:
        cost(args, files, report)
    # a --no-parity / --no-time rerun keeps the other half of an existing report
    path = cm.write_report(f"meanvc_{args.ep}.json", report, merge=args.no_parity or args.no_time)
    print("wrote", path)


if __name__ == "__main__":
    main()
