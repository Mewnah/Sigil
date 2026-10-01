#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Seed-VC realtime: offline parity of the exported graphs against upstream PyTorch, and their
per-call cost (P3-H, ADR 0006).

Reference: upstream ``real-time-gui.py`` ``custom_infer`` (the prompt branch and one real-time
tick) run on the vendored upstream modules in PyTorch fp32 on the CPU, with every random draw
recorded (``common.recorded_random``). Candidate: ``sampler.SigilZS`` on the ONNX graphs, fed the
same audio and the same draws. Source speech: ``jfk.wav``; reference voice: ``ja-short.wav``
(the STT samples in the dev cache). Gate: SNR >= 35 dB on the tick's output audio (and on every
stage). ``--ep cuda`` also runs upstream's own real-time precision (XLS-R in fp16, the DiT and HiFT
under fp16 autocast, as ``real-time-gui.py --fp16 true``, its default) against the same fp32
reference, for scale.

Precision variants (``VARIANTS``): fp32; fp16 (every graph with an fp16 export); hift16 (only
HiFT's decoder, in ``hift`` and ``hift_nsf``); xlsr16 and xlsr16_hift16. A reduced-precision variant
is judged on its stage rows and on the log-mel of its audio next to upstream's own fp16 mode: HiFT
predicts its F0 from the mel, so any mel difference drifts the waveform's phase (ADR 0006).

The vocoder runs both ways (``sampler.SigilZS.vocode``): ``full`` (the ``hift`` graph over the
whole tick, upstream) and ``partial`` (``hift_f0`` over the tick, ``hift_nsf`` over the returned
frames plus ``hift_margin`` frames), each gated against upstream. ``stream_join`` (not gated)
measures what the time-anchored levers buy over consecutive ticks: at each join, how well the two
ticks' overlapping mels agree and how their waveforms line up over the SOLA crossfade, with
upstream's fresh randomness, with z and the sine noise anchored on the mel-frame grid, and with the
NSF phase anchored at the join as well.

Cost: per graph at the shapes of each preset in ``sampler.PRESETS``, and per tick
(xlsr + lr + steps x dit_step (dit_step1 when cfg = 0) + vocoder, ``tick`` with ``hift`` and
``tick_partial`` with ``hift_f0`` + ``hift_nsf``); on CUDA as host-tensor ``run()``, IOBinding with
device-resident tensors, and a replayed CUDA graph. Every parity and cost section records the
sha256 of the graphs it measured (``graphs``).

Usage (repo root):
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-cuda/Scripts/python scripts/vc-families/seed_vc/parity.py   # CPU EP of ORT 1.28
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-cuda/Scripts/python scripts/vc-families/seed_vc/parity.py --ep cuda
  %LOCALAPPDATA%/sigil-dev-cache/venv-ort128-dml/Scripts/python scripts/vc-families/seed_vc/parity.py --ep dml --variants fp32
Writes ``reports/seed_vc_<ep>.json`` (``--no-parity``, ``--no-time`` and ``--variants`` merge into it).
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

SOURCE_WAV, REFERENCE_WAV = "jfk.wav", "ja-short.wav"
WINDOW_START_S = 1.0  # the tick's 16 kHz window starts here in the source


VARIANTS = {
    "fp32": set(),
    # fp16 inside / fp32 I/O for every graph that has an fp16 export
    "fp16": {"xlsr", "dit", "dit_step", "dit_step1", "hift"},
    # only HiFT's decoder in fp16 (``hift`` and ``hift_nsf``): it runs after the F0 predictor, so it
    # cannot move the NSF phase
    "hift16": {"hift"},
    # XLS-R in fp16 (half the download and VRAM of the largest graph), the DiT in fp32
    "xlsr16": {"xlsr"},
    "xlsr16_hift16": {"xlsr", "hift"},
}


GRAPHS = ("xlsr", "lr", "dit", "dit_step", "dit_step1", "hift", "hift_f0", "hift_nsf", "campplus", "mel")


def variant_files(variant):
    half = set(VARIANTS[variant])
    if "hift" in half:
        half.add("hift_nsf")   # the same decoder; hift_f0 is fp32 only
    return {n: cm.need(os.path.join(ex.OUT, f"{n}{'.fp16' if n in half else ''}.onnx"), "run export.py first") for n in GRAPHS}


def kaldi_fbank(wav16k, scale=1.0):
    import torchaudio.compliance.kaldi as kaldi
    fb = kaldi.fbank(torch.from_numpy(wav16k * scale)[None], num_mel_bins=80, dither=0, sample_frequency=16000)
    return (fb - fb.mean(dim=0, keepdim=True)).numpy()


# ---------------------------------------------------------------------------------------------
# Upstream reference (custom_infer, fp32 unless ``fp16``)
# ---------------------------------------------------------------------------------------------

class Upstream:
    def __init__(self, up, device="cpu", fp16=False):
        from transformers import Wav2Vec2FeatureExtractor
        from modules.audio import mel_spectrogram
        self.up, self.device, self.fp16 = up, device, fp16
        self.fe = Wav2Vec2FeatureExtractor.from_pretrained(ex.XLSR_DIR)
        self.to_mel = lambda x: mel_spectrogram(x, **up["mel_args"])
        self.xlsr = up["xlsr"].half() if fp16 else up["xlsr"]

    def semantic_fn(self, waves_16k):
        inputs = self.fe([waves_16k[b].cpu().numpy() for b in range(len(waves_16k))], return_tensors="pt",
                         return_attention_mask=True, padding=True, sampling_rate=16000).to(self.device)
        with torch.no_grad():
            iv = inputs.input_values.half() if self.fp16 else inputs.input_values
            return self.xlsr(iv).last_hidden_state.float()

    @torch.no_grad()
    def reference(self, ref22k, max_prompt):
        import torchaudio
        sr = sp.SR
        ref = torch.from_numpy(ref22k[: int(sr * max_prompt)]).to(self.device)
        ori_16k = torchaudio.functional.resample(ref, sr, 16000)
        s_ori = self.semantic_fn(ori_16k.unsqueeze(0))
        feat2 = torchaudio.compliance.kaldi.fbank(ori_16k.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
        feat2 = feat2 - feat2.mean(dim=0, keepdim=True)
        style2 = self.up["campplus"](feat2.unsqueeze(0))
        mel2 = self.to_mel(ref.unsqueeze(0))
        prompt_condition = self.up["model"].length_regulator(s_ori, ylens=torch.LongTensor([mel2.size(2)]).to(self.device),
                                                             n_quantizers=3, f0=None)[0]
        return {"ref16k": ori_16k.cpu().numpy(), "fbank": feat2.cpu().numpy(), "s_ori": s_ori, "style": style2,
                "prompt_mel": mel2, "prompt_cond": prompt_condition}

    @torch.no_grad()
    def tick(self, ref, window16k, preset, seed=0):
        u = preset.units()
        sr, hop = sp.SR, sp.HOP
        s_alt = self.semantic_fn(torch.from_numpy(window16k).to(self.device).unsqueeze(0))
        ce = u["ce_frames"]
        s_alt = s_alt[:, ce:]
        target_lengths = torch.LongTensor([(u["skip_head"] + u["return_length"] + u["skip_tail"] - ce) / 50 * sr // hop]).to(self.device)
        cond = self.up["model"].length_regulator(s_alt, ylens=target_lengths, n_quantizers=3, f0=None)[0]
        cat_condition = torch.cat([ref["prompt_cond"], cond], dim=1)
        mel2 = ref["prompt_mel"]
        with cm.recorded_random(seed) as rec:
            with torch.autocast(device_type=self.device, dtype=torch.float16 if self.fp16 else torch.float32, enabled=self.fp16):
                vc_target = self.up["model"].cfm.inference(cat_condition, torch.LongTensor([cat_condition.size(1)]).to(self.device),
                                                           mel2, ref["style"], None, n_timesteps=preset.steps,
                                                           inference_cfg_rate=preset.cfg_rate)
                vc_target = vc_target[:, :, mel2.size(-1):]
                vc_wave = self.up["hift"](vc_target).squeeze()
        out_len, tail = u["return_length"] * sr // 50, u["skip_tail"] * sr // 50
        out = vc_wave[-out_len - tail: -tail]
        draws = {"z": next(t for k, t in rec if k == "randn"), "harm_phase": next(t for k, t in rec if k == "uniform"),
                 "sine_noise": next(t for k, t in rec if k == "randn_like")}
        return out.float().cpu().numpy(), {"s_alt": s_alt, "cond": cond, "mel_out": vc_target.float(),
                                           "draws": {k: v.float().cpu().numpy() for k, v in draws.items()}}


# ---------------------------------------------------------------------------------------------
# Parity
# ---------------------------------------------------------------------------------------------

def np_(t):
    return t.detach().float().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)


def returned(wav, dec_start, n_mel, preset):
    """The returned audio (block + SOLA margins) of a vocoded slice that starts at frame ``dec_start``."""
    u = preset.units()
    out_len, tail = u["return_length"] * sp.ZC_22K, u["skip_tail"] * sp.ZC_22K
    end = n_mel * sp.HOP - tail - dec_start * sp.HOP
    return wav[end - out_len: end]


def join_coherence(prev, cur, preset, expect, span=512):
    """One join between consecutive ticks' returned audio (``prev``, ``cur``; whole decoded slices
    with the returned audio at ``out0``): the previous tick's samples after its block are SOLA's
    crossfade buffer. Normalised correlation of ``cur`` against that buffer at every offset in
    [-span, span]: the best one and where it is, the best within SOLA's own search range
    [0, sola_search], and the value at ``expect`` (the offset where the ticks' noise grids align)."""
    blk = int(np.round(preset.block * 50)) * sp.ZC_22K
    sb, ss = min(int(np.round(preset.crossfade * 50)), 4) * sp.ZC_22K, sp.ZC_22K
    out0 = prev["out0"]
    buf = prev["wav"][out0 + blk: out0 + blk + sb].astype(np.float64)
    nb = float(np.linalg.norm(buf)) + 1e-12
    offs = [o for o in range(-span, span + 1) if 0 <= out0 + o and out0 + o + sb <= cur["wav"].size]
    cor = {}
    for o in offs:
        seg = cur["wav"][out0 + o: out0 + o + sb].astype(np.float64)
        cor[o] = float(seg @ buf) / (nb * (float(np.linalg.norm(seg)) + 1e-12))
    best = max(cor, key=cor.get)
    sola = [o for o in range(0, ss + 1) if o in cor]
    return {"best_corr": round(cor[best], 3), "best_offset": best, "sola_range_best_corr": round(max(cor[o] for o in sola), 3),
            "corr_at_expected_offset": round(cor.get(expect, float("nan")), 3)}


def stream_join(zs, ref_s, src16k, preset, s0, seed, n_joins=8):
    """Consecutive ticks (the window advanced by one block each) with three sets of levers:
    ``upstream`` (a fresh z, fresh harmonic phases and fresh sine noise every tick), ``z_anchored``
    (z and the sine noise anchored on the mel-frame grid of the content time base, F frames per
    tick, ``Preset.frames_per_block``) and ``z_and_phase`` (plus the NSF phase anchored at the
    join, ``Preset.join_anchor(F * 256)``). Per join: how well the overlapping mel frames of the two
    ticks agree (SNR), and how the waveforms line up over the SOLA crossfade (``join_coherence``)."""
    sh = preset.shapes()
    blk_units = int(np.round(preset.block * 50))
    blk16, blk22 = blk_units * sp.ZC_16K, blk_units * sp.ZC_22K
    n_mel, big_l, p = sh["n_mel"], sh["L"], sh["prompt_frames"]
    fpb = preset.frames_per_block()
    f = int(round(fpb))
    n_this, n_prev = preset.join_anchor(f * sp.HOP)
    out0 = sh["out_start"] - sh["dec_start"] * sp.HOP
    expect = blk22 - f * sp.HOP                                         # where the anchored grids put the previous buffer
    rng = np.random.default_rng(seed)
    z_grid = rng.standard_normal((1, 80, n_mel + n_joins * f)).astype(np.float32)
    n_grid = rng.standard_normal((1, sp.N_HARM, (n_mel + n_joins * f) * sp.HOP)).astype(np.float32)
    rows = {m: [] for m in ("upstream", "z_anchored", "z_and_phase")}
    prev = {}
    for k in range(n_joins + 1):
        w = np.ascontiguousarray(src16k[s0 + k * blk16: s0 + k * blk16 + sh["n16"]])
        z_fresh = rng.standard_normal((1, 80, big_l)).astype(np.float32)
        z_anch = z_fresh.copy()
        z_anch[..., p:] = z_grid[..., k * f: k * f + n_mel]
        hp = rng.uniform(-np.pi, np.pi, (1, sp.N_HARM, 1)).astype(np.float32)
        hp[:, 0] = 0.0
        sn_fresh = rng.standard_normal((1, sp.N_HARM, n_mel * sp.HOP)).astype(np.float32)
        sn_anch = n_grid[..., k * f * sp.HOP: (k * f + n_mel) * sp.HOP]
        mels = {}
        for mode in rows:
            z, sn = (z_fresh, sn_fresh) if mode == "upstream" else (z_anch, sn_anch)
            if mode != "z_and_phase":
                _, st = zs.tick(ref_s, w, preset, z, hp, sn, hift_mode="partial")
                mels[mode] = st["mel_out"]
            anchor = None
            if mode == "z_and_phase" and mode in prev:
                anchor = (n_this, sp.nsf_cycles_at(prev[mode]["f0_dec"], prev[mode]["harm_phase"], n_prev))
            wav, info = zs.vocode(mels[mode if mode != "z_and_phase" else "z_anchored"], hp, sn, preset, "partial", anchor)
            cur = {"wav": wav, "out0": out0, "mel": mels[mode if mode != "z_and_phase" else "z_anchored"], **info}
            if mode in prev:
                r = join_coherence(prev[mode], cur, preset, expect)
                r["mel_overlap_snr_db"] = cm.compare(prev[mode]["mel"][..., f:], cur["mel"][..., : n_mel - f])["snr_db"]
                rows[mode].append(r)
            prev[mode] = cur
    res = {}
    for mode, rs in rows.items():
        mean = lambda key: round(float(np.mean([r[key] for r in rs])), 3)  # noqa: E731
        res[mode] = {"mean_mel_overlap_snr_db": mean("mel_overlap_snr_db"), "mean_best_corr": mean("best_corr"),
                     "mean_corr_at_expected_offset": mean("corr_at_expected_offset"),
                     "mean_sola_range_best_corr": mean("sola_range_best_corr"), "joins": rs}
    res["setup"] = {"preset": preset.name, "joins": n_joins, "window_start_s": s0 / 16000, "frames_per_block": round(fpb, 3),
                    "grid_frames_per_tick": f, "expected_offset": expect, "anchor": {"n_this": n_this, "n_prev": n_prev},
                    "note": "upstream: fresh z / phases / sine noise per tick; z_anchored: z and sine noise on the content-time "
                            "mel-frame grid; z_and_phase: plus the NSF phase anchored at the join. Offsets are samples at 22.05 kHz "
                            "relative to upstream's join (SOLA searches 0..441)."}
    return res


def parity(args, sessions, up, ref22k, src16k, report, files=None):
    preset = sp.PRESETS[args.preset]
    ups = Upstream(up)
    ref_u = ups.reference(ref22k, preset.max_prompt)
    n16 = preset.shapes()["n16"]
    s0 = int(WINDOW_START_S * 16000)
    window = np.ascontiguousarray(src16k[s0: s0 + n16])
    out_u, st = ups.tick(ref_u, window, preset, seed=args.seed)
    zs = sp.SigilZS(sessions)
    rows = {}

    # isolated graphs: each fed upstream's own inputs to that stage
    rows["xlsr_prompt"] = cm.gate(cm.compare(np_(ref_u["s_ori"]), zs._run("xlsr", {"wav16k": ref_u["ref16k"][None]})[0]))
    rows["xlsr_window"] = cm.gate(cm.compare(np_(ups.semantic_fn(torch.from_numpy(window)[None])),
                                             zs._run("xlsr", {"wav16k": window[None]})[0]))
    rows["campplus"] = cm.gate(cm.compare(np_(ref_u["style"]), zs._run("campplus", {"fbank": ref_u["fbank"][None]})[0]))
    ref22_cut = ref22k[: int(sp.SR * preset.max_prompt)]
    rows["mel"] = cm.gate(cm.compare(np_(ref_u["prompt_mel"]), zs._run("mel", {"wav22k": ref22_cut[None]})[0]))
    p = ref_u["prompt_mel"].shape[-1]
    rows["lr_prompt"] = cm.gate(cm.compare(np_(ref_u["prompt_cond"]), zs._run("lr", {"feats": np_(ref_u["s_ori"]), "n_frames": np.array([p], np.int64)})[0]))
    n_mel = st["cond"].shape[1]
    rows["lr_tick"] = cm.gate(cm.compare(np_(st["cond"]), zs._run("lr", {"feats": np_(st["s_alt"]), "n_frames": np.array([n_mel], np.int64)})[0]))
    ref_o = {"style": np_(ref_u["style"]), "prompt_mel": np_(ref_u["prompt_mel"]), "prompt_cond": np_(ref_u["prompt_cond"])}
    mu = np.concatenate([ref_o["prompt_cond"], np_(st["cond"])], axis=1)
    d = st["draws"]
    # one estimator call and one fused step at mid-trajectory inputs
    x_mid = d["z"].copy()
    x_mid[..., :p] = 0
    prompt_x = np.zeros_like(x_mid)
    prompt_x[..., :p] = ref_o["prompt_mel"]
    with torch.no_grad():
        est = up["model"].cfm.estimator
        big_l = mu.shape[1]
        v_ref = est(torch.from_numpy(np.concatenate([x_mid, x_mid])), torch.from_numpy(np.concatenate([prompt_x, 0 * prompt_x])),
                    torch.LongTensor([big_l, big_l]), torch.tensor([0.3, 0.3]),
                    torch.from_numpy(np.concatenate([ref_o["style"], 0 * ref_o["style"]])),
                    torch.from_numpy(np.concatenate([mu, 0 * mu]))).numpy()
    v_got = zs._run("dit", {"x": np.concatenate([x_mid, x_mid]), "prompt_x": np.concatenate([prompt_x, 0 * prompt_x]),
                            "x_lens": np.array([big_l, big_l], np.int64), "t": np.array([0.3, 0.3], np.float32),
                            "style": np.concatenate([ref_o["style"], 0 * ref_o["style"]]), "cond": np.concatenate([mu, 0 * mu])})[0]
    rows["dit_estimator"] = cm.gate(cm.compare(v_ref, v_got))
    cfg, dt = np.float32(preset.cfg_rate), np.float32(0.1)
    step_ref = x_mid + dt * ((1 + cfg) * v_ref[0:1] - cfg * v_ref[1:2])
    step_ref[..., :p] = 0
    step_got = zs._run("dit_step", {"x": x_mid, "prompt_x": prompt_x, "cond": mu, "style": ref_o["style"], "t": np.array([0.3], np.float32),
                                    "dt": np.array([dt], np.float32), "cfg_rate": np.array([cfg], np.float32), "prompt_len": np.array([p], np.int64)})[0]
    rows["dit_step"] = cm.gate(cm.compare(step_ref, step_got))
    with torch.no_grad():
        v1 = est(torch.from_numpy(x_mid), torch.from_numpy(prompt_x), torch.LongTensor([big_l]), torch.tensor([0.3]),
                 torch.from_numpy(ref_o["style"]), torch.from_numpy(mu)).numpy()
    step1_ref = x_mid + dt * v1
    step1_ref[..., :p] = 0
    step1_got = zs._run("dit_step1", {"x": x_mid, "prompt_x": prompt_x, "cond": mu, "style": ref_o["style"], "t": np.array([0.3], np.float32),
                                      "dt": np.array([dt], np.float32), "prompt_len": np.array([p], np.int64)})[0]
    rows["dit_step1"] = cm.gate(cm.compare(step1_ref, step1_got))
    with torch.no_grad(), cm.recorded_random(args.seed + 1) as rec:
        wav_ref = up["hift"](st["mel_out"]).numpy()
    hp = next(t for k, t in rec if k == "uniform").numpy()
    sn = next(t for k, t in rec if k == "randn_like").numpy()
    rows["hift"] = cm.gate(cm.compare_audio(wav_ref.ravel(), zs._run("hift", {"mel": np_(st["mel_out"]), "harm_phase": hp, "sine_noise": sn})[0].ravel(), sp.SR))
    # the split graphs on the same mel and draws: hift_f0, the whole window (split), and the partial
    # decode of the returned frames (hift_margin frames of context, the phase carried to the slice)
    mel_u = np_(st["mel_out"])
    with torch.no_grad():
        f0_ref = up["hift"].f0_predictor(st["mel_out"]).numpy()
    rows["hift_f0"] = cm.gate(cm.compare(f0_ref, zs._run("hift_f0", {"mel": mel_u})[0]))
    w_split, _ = zs.vocode(mel_u, hp, sn, preset, "split")
    rows["hift_split"] = cm.gate(cm.compare_audio(wav_ref.ravel(), w_split, sp.SR))
    w_part, vi = zs.vocode(mel_u, hp, sn, preset, "partial")
    n_mel_u = mel_u.shape[-1]
    rows["hift_partial"] = cm.gate(cm.compare_audio(returned(wav_ref.ravel(), 0, n_mel_u, preset), returned(w_part, vi["dec_start"], n_mel_u, preset), sp.SR))

    # the whole chain: reference branch and one tick, Sigil sampler end to end
    ref_s = zs.prepare_reference(ref22_cut, ref_u["ref16k"], ref_u["fbank"])
    rows["chain_style"] = cm.gate(cm.compare(np_(ref_u["style"]), ref_s["style"]))
    rows["chain_prompt_cond"] = cm.gate(cm.compare(np_(ref_u["prompt_cond"]), ref_s["prompt_cond"]))
    out_s, st_s = zs.tick(ref_s, window, preset, d["z"], d["harm_phase"], d["sine_noise"])
    rows["chain_tick_cond"] = cm.gate(cm.compare(np_(st["cond"]), st_s["cond"]))
    rows["chain_tick_mel"] = cm.gate(cm.compare(np_(st["mel_out"]), st_s["mel_out"]))
    rows["chain_tick_audio"] = cm.gate(cm.compare_audio(out_u, out_s, sp.SR))
    out_b, _ = zs.tick(ref_s, window, preset, d["z"], d["harm_phase"], d["sine_noise"], use_step_graph=False)
    rows["chain_tick_audio_host_update"] = cm.gate(cm.compare_audio(out_u, out_b, sp.SR))
    out_p, _ = zs.tick(ref_s, window, preset, d["z"], d["harm_phase"], d["sine_noise"], hift_mode="partial")
    rows["chain_tick_audio_partial"] = cm.gate(cm.compare_audio(out_u, out_p, sp.SR))
    # a second preset without CFG (dit_step1), a shorter prompt, window and step count
    lean = sp.PRESETS["sigil_lean"]
    ref_ul = ups.reference(ref22k, lean.max_prompt)
    wl = np.ascontiguousarray(src16k[s0: s0 + lean.shapes()["n16"]])
    out_ul, st_l = ups.tick(ref_ul, wl, lean, seed=args.seed + 2)
    ref_sl = zs.prepare_reference(ref22k[: int(sp.SR * lean.max_prompt)], ref_ul["ref16k"], ref_ul["fbank"])
    dl = st_l["draws"]
    out_sl, st_sl = zs.tick(ref_sl, wl, lean, dl["z"], dl["harm_phase"], dl["sine_noise"])
    rows["lean_chain_tick_mel"] = cm.gate(cm.compare(np_(st_l["mel_out"]), st_sl["mel_out"]))
    rows["lean_chain_tick_audio"] = cm.gate(cm.compare_audio(out_ul, out_sl, sp.SR))
    out_slp, _ = zs.tick(ref_sl, wl, lean, dl["z"], dl["harm_phase"], dl["sine_noise"], hift_mode="partial")
    rows["lean_chain_tick_audio_partial"] = cm.gate(cm.compare_audio(out_ul, out_slp, sp.SR))
    report["parity"] = rows
    report["parity_setup"] = {"preset": dataclasses_asdict(preset), "shapes": preset.shapes(), "source": SOURCE_WAV,
                              "reference": REFERENCE_WAV, "window_start_s": WINDOW_START_S, "seed": args.seed,
                              "reference_impl": "upstream custom_infer, PyTorch fp32 CPU"}
    if files:
        report["parity_setup"]["graphs"] = cm.graph_stamp(files)
    # the phase-continuous NSF lever over consecutive ticks (not gated: upstream has no such path)
    report["stream_join"] = stream_join(zs, ref_s, src16k, preset, s0, args.seed + 3)
    if files:
        report["stream_join"]["graphs"] = cm.graph_stamp({n: files[n] for n in ("xlsr", "lr", "dit_step", "dit_step1", "hift_f0", "hift_nsf")})

    # upstream's own real-time precision (fp16) vs the same fp32 reference, for scale
    # (both on CUDA with the same seed, so the draws are identical; CPU and CUDA draws differ)
    if args.ep == "cuda" and torch.cuda.is_available() and not getattr(args, "fp16_ref_done", False):
        args.fp16_ref_done = True
        up32c = ex.load_upstream("cuda")
        u32c = Upstream(up32c, "cuda", fp16=False)
        out32c, st32 = u32c.tick(u32c.reference(ref22k, preset.max_prompt), window, preset, seed=args.seed)
        u16 = Upstream(up32c, "cuda", fp16=True)   # halves the XLS-R copy in place; loaded last
        out16, st16 = u16.tick(u16.reference(ref22k, preset.max_prompt), window, preset, seed=args.seed)
        report["upstream_realtime_fp16_vs_fp32"] = {"tick_audio": cm.compare_audio(out32c, out16, sp.SR),
                                                    "tick_mel": cm.compare(np_(st32["mel_out"]), np_(st16["mel_out"])),
                                                    "note": "upstream real-time-gui --fp16 true (its default) vs upstream fp32, both on CUDA, same draws"}
        del up32c, u32c, u16
        torch.cuda.empty_cache()

    # CAM++ front ends: Rust bench dsp::kaldi_fbank scales by 32768; the bench asset is sherpa's export
    fb_scaled = kaldi_fbank(ref_u["ref16k"], 32768.0)
    e_scaled = zs._run("campplus", {"fbank": fb_scaled[None]})[0]
    fe = {"fbank_x32768_vs_upstream": cm.compare(np_(ref_u["style"]), e_scaled)}
    bench = os.path.join(cm.DEV_CACHE, "vc-assets", "bench", "campplus", "model.onnx")
    stamp = {"campplus": files["campplus"]} if files else {}
    if os.path.exists(bench):
        sb = cm.session(bench, "cpu")
        name = sb.get_inputs()[0].name
        e_bench = sb.run(None, {name: fb_scaled[None].astype(np.float32)})[0]
        fe["bench_campplus_asset_vs_upstream"] = cm.compare(np_(ref_u["style"]).ravel(), e_bench.ravel())
        stamp["bench_campplus"] = bench
    if stamp:
        fe["graphs"] = cm.graph_stamp(stamp)
    report["campplus_front_end"] = fe
    return report


def dataclasses_asdict(p):
    import dataclasses
    return dataclasses.asdict(p)


# ---------------------------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------------------------

def cost(args, files, report):
    """Per-call cost at each preset's shapes. Sessions pin every symbolic dimension to the preset
    (as the engine does); on CUDA, ``graph`` rows replay a captured CUDA graph."""
    report["timing_conditions"] = cm.timing_conditions(args.ep)
    rng = np.random.default_rng(0)
    f = lambda *shape: rng.standard_normal(shape).astype(np.float32)  # noqa: E731
    out = {}
    for name, preset in sp.PRESETS.items():
        s = preset.shapes()
        big_l, p, m = s["L"], s["prompt_frames"], s["n_mel"]
        calls = {
            "xlsr": ("xlsr", {"N": s["n16"], "T": s["xlsr_frames"]}, {"wav16k": f(1, s["n16"]) * 0.1}),
            "lr": ("lr", {"T": s["lr_in"], "n": m}, {"feats": f(1, s["lr_in"], 1024), "n_frames": np.array([m], np.int64)}),
            "dit_step": ("dit_step", {"L": big_l}, {"x": f(1, 80, big_l), "prompt_x": f(1, 80, big_l), "cond": f(1, big_l, 384), "style": f(1, 192),
                                                     "t": np.array([0.3], np.float32), "dt": np.array([0.1], np.float32),
                                                     "cfg_rate": np.array([preset.cfg_rate], np.float32), "prompt_len": np.array([p], np.int64)}),
            "dit_step1": ("dit_step1", {"L": big_l}, {"x": f(1, 80, big_l), "prompt_x": f(1, 80, big_l), "cond": f(1, big_l, 384), "style": f(1, 192),
                                                       "t": np.array([0.3], np.float32), "dt": np.array([0.1], np.float32),
                                                       "prompt_len": np.array([p], np.int64)}),
            "hift": ("hift", {"M": m, "S": m * 256}, {"mel": f(1, 80, m) - 5.0, "harm_phase": np.zeros((1, 9, 1), np.float32),
                                                     "sine_noise": f(1, 9, m * 256)}),
            "hift_f0": ("hift_f0", {"M": m}, {"mel": f(1, 80, m) - 5.0}),
            "hift_nsf": ("hift_nsf", {"W": s["dec_frames"], "S": s["dec_samples"]},
                         {"mel": f(1, 80, s["dec_frames"]) - 5.0, "f0": np.full((1, s["dec_frames"]), 180.0, np.float32),
                          "harm_phase": np.zeros((1, 9, 1), np.float32), "sine_noise": f(1, 9, s["dec_samples"])}),
        }
        calls.pop("dit_step1" if preset.cfg_rate > 0 else "dit_step")
        rows = {}
        for g, (graph, dims, feeds) in calls.items():
            sess = cm.session(files[graph], args.ep, args.threads, dims=dims)
            rows[g] = cm.timed_session_call(sess, feeds, args.ep, args.warm, args.runs, path=files[graph], dims=dims, threads=args.threads)
            del sess
        step_key = "dit_step" if preset.cfg_rate > 0 else "dit_step1"
        for tick_key, vocoder in (("tick", ("hift",)), ("tick_partial", ("hift_f0", "hift_nsf"))):
            totals = {}
            for mode in ("host", "bound", "graph"):
                try:
                    pick = lambda r, mode=mode: r[mode]["median_ms"] if mode in r and "median_ms" in r[mode] else cm.best_ms(r)[0]  # noqa: E731
                    if not any(mode in r and "median_ms" in r[mode] for r in rows.values() if isinstance(r, dict)):
                        continue
                    step = pick(rows[step_key])
                    fixed = pick(rows["xlsr"]) + pick(rows["lr"]) + sum(pick(rows[v]) for v in vocoder)
                    totals[mode] = {"tick_ms": round(fixed + preset.steps * step, 2), "per_step_ms": step,
                                    "rtf": round((fixed + preset.steps * step) / (preset.block * 1000), 3)}
                except KeyError:
                    pass
            rows[tick_key] = totals
        rows["block_ms"] = round(preset.block * 1000, 1)
        rows["preset"] = dataclasses_asdict(preset)
        rows["shapes"] = s
        out[name] = rows
        for tick_key in ("tick", "tick_partial"):
            print(f"{name} {tick_key}: block {rows['block_ms']} ms; " + "; ".join(
                f"{k} {v['tick_ms']} ms (step {v['per_step_ms']} x {preset.steps}, RTF {v['rtf']})" for k, v in rows[tick_key].items()))
    once = {}
    ref_n, fb_n, mel_n = int(16000 * 3.0), 298, int(22050 * 3.0)
    for g, dims, feeds in (("xlsr", {"N": ref_n}, {"wav16k": f(1, ref_n) * 0.1}), ("campplus", {"F": fb_n}, {"fbank": f(1, fb_n, 80)}),
                           ("mel", {"S": mel_n}, {"wav22k": f(1, mel_n) * 0.1})):
        sess = cm.session(files[g], args.ep, args.threads, dims=dims)
        once[g] = cm.timed_session_call(sess, feeds, args.ep, 2, 10)
    out["once_per_voice"] = once
    out["graphs"] = cm.graph_stamp(files)
    report["cost"] = out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ep", default="cpu", choices=sorted(cm.EPS))
    ap.add_argument("--variants", default=None, help="comma list of VARIANTS keys (default: fp32 on cpu, all on gpus); a subset merges into the existing report")
    ap.add_argument("--preset", default="upstream_readme", choices=sorted(sp.PRESETS))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-parity", action="store_true")
    ap.add_argument("--no-time", action="store_true")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--runs", type=int, default=30)
    args = ap.parse_args()
    variants = args.variants.split(",") if args.variants else (["fp32"] if args.ep == "cpu" else list(VARIANTS))

    cm.import_ort(args.ep)
    report = {"family": "seed_vc", "model": "seed-uvit-tat-xlsr-tiny", "env": cm.env_info(args.ep), "threads": args.threads, "variants": {}}
    up = ex.load_upstream("cpu") if not args.no_parity else None
    src16k = cm.load_wav16k(SOURCE_WAV)
    ref22k = cm.resample(cm.load_wav16k(REFERENCE_WAV), 16000, sp.SR)
    for v in variants:
        files = variant_files(v)
        sessions = {n: cm.session(p, args.ep, args.threads) for n, p in files.items()}
        rep = {"files": {n: os.path.basename(p) for n, p in files.items()}}
        if not args.no_parity:
            parity(args, sessions, up, ref22k, src16k, rep, files)
            bad = [k for k, r in rep["parity"].items() if not r["pass"]]
            print(f"[{args.ep} {v}] parity: {'PASS' if not bad else 'FAIL ' + ', '.join(bad)}; "
                  f"tick audio {rep['parity']['chain_tick_audio']['snr_db']} dB")
            for k, r in rep["parity"].items():
                print(cm.row_line(k, r))
            sj = rep["stream_join"]
            for m in ("upstream", "z_anchored", "z_and_phase"):
                print(f"   stream_join {m:12s} mel overlap {sj[m]['mean_mel_overlap_snr_db']} dB; best corr {sj[m]['mean_best_corr']} "
                      f"(offsets {[r['best_offset'] for r in sj[m]['joins']]}); at the grid offset {sj[m]['mean_corr_at_expected_offset']}; "
                      f"in SOLA's range {sj[m]['mean_sola_range_best_corr']}")
        if not args.no_time:
            del sessions
            cost(args, files, rep)
        report["variants"][v] = rep
    # a --no-parity / --no-time rerun keeps the other half of an existing report
    path = cm.write_report(f"seed_vc_{args.ep}.json", report, merge=args.no_parity or args.no_time or bool(args.variants))
    print("wrote", path)


if __name__ == "__main__":
    main()
