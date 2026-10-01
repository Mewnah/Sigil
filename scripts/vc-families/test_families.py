# SPDX-License-Identifier: AGPL-3.0-or-later
"""Focused checks of the pieces of the P3-H family exports that Sigil wrote itself (ADR 0006), the
parts a Rust port copies: the fixed-shape KV-cache DiT wrapper, the convolutional STFT / iSTFT and
Kaldi fbank, the samplers' bit-exact torch formulas, and the upstream host-side bookkeeping they
mirror. No checkpoints are needed (random weights, synthetic signals); the whole-model parity runs
are ``<family>/parity.py``.

Run from the repo root: ``python -m pytest scripts/vc-families/test_families.py -q``
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import common as cm  # noqa: E402


def _load(family, module):
    """Import ``<family>/<module>.py`` under a unique name (every family has export/sampler)."""
    name = f"{family}_{module}"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, family, f"{module}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------------------------

def test_compare_snr_and_gate():
    ref = np.ones(1000, np.float32)
    row = cm.compare(ref, ref * np.float32(1.01))
    assert abs(row["snr_db"] - 40.0) < 1e-3
    assert cm.gate(dict(row))["pass"]
    assert not cm.gate(cm.compare(ref, ref * np.float32(1.1)))["pass"]      # 20 dB
    assert cm.gate(cm.compare(ref, ref))["pass"]                             # identical: inf
    assert cm.compare(ref, np.full_like(ref, np.nan))["snr_db"] == "nan"


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_compare_fails_on_a_non_finite_reference_or_candidate(bad):
    ok = np.array([0.0, 1.0, 2.0, 3.0], np.float32)
    ref = ok.copy()
    ref[1] = bad
    row = cm.gate(cm.compare(ref, ok))                       # a reference that went non-finite
    assert row["snr_db"] == "nan" and row["ref_non_finite"] == 1 and not row["pass"]
    assert not cm.gate(cm.compare(ref, ref.copy()))["pass"]  # "identical" but non-finite: never a pass
    row = cm.gate(cm.compare(ok, ref))                       # a candidate that went non-finite
    assert row["snr_db"] == "nan" and row["non_finite"] == 1 and not row["pass"]


def test_gate_passes_only_finite_identical_or_high_snr():
    zero = np.zeros(8)
    assert cm.compare(zero, zero)["snr_db"] == "inf" and cm.gate(cm.compare(zero, zero))["pass"]
    assert not cm.gate(cm.compare(zero, zero + 1e-6))["pass"]        # silence vs anything: -inf-like SNR
    assert not cm.gate({"snr_db": float("nan")})["pass"]
    assert not cm.gate({"snr_db": "inf", "ref_non_finite": 2})["pass"]
    assert cm.gate({"snr_db": 35.0})["pass"] and not cm.gate({"snr_db": 34.99})["pass"]
    assert "FAIL" in cm.row_line("x", cm.gate(cm.compare(np.ones(3), np.array([1.0, np.nan, 1.0]))))


def _identity_model(path):
    import onnx
    from onnx import TensorProto, helper
    g = helper.make_graph([helper.make_node("Identity", ["x"], ["y"])], "g",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, "N"])],
                          [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, "N"])])
    onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)]), str(path))
    return str(path)


def test_session_info_records_pinned_and_refused_dimensions(tmp_path):
    path = _identity_model(tmp_path / "id.onnx")
    assert cm.session_info(cm.session(path, "cpu", 1, dims={"N": 4})) == {"dims_pinned": True}
    assert cm.session_info(cm.session(path, "cpu", 1)) == {"dims_pinned": False}

    class Fallback:  # what session() leaves on a DirectML session that fell back to dynamic shapes
        sigil_dims = {}
        sigil_dims_refused = {"dims": {"N": 4}, "error": "The parameter is incorrect."}
    info = cm.session_info(Fallback())
    assert info["dims_pinned"] is False and info["dims_refused"]["dims"] == {"N": 4}
    row = cm.timed_session_call(cm.session(path, "cpu", 1, dims={"N": 4}), {"x": np.ones((1, 4), np.float32)}, "cpu", 1, 3)
    assert row["session"] == {"dims_pinned": True} and "median_ms" in row["host"]


def test_profile_session_call_summarises_ort_profile(tmp_path):
    path = _identity_model(tmp_path / "id.onnx")
    p = cm.profile_session_call(path, {"x": np.ones((1, 16), np.float32)}, "cpu", 1, dims={"N": 16}, warm=1, runs=3)
    assert p["runs"] == 3 and p["nodes_by_provider"] == {"CPUExecutionProvider": 1}
    assert p["kernel_ms_per_run"] >= 0 and p["wall_ms_per_run"] > 0 and p["session"]["dims_pinned"]
    assert p["process_cpu_ms_per_run"] >= 0 and "gpu_util_pct_during" not in p      # no GPU sampling on the CPU EP
    assert p["graphs"]["id"]["sha256"] == cm.graph_sha256(path)
    assert not [f for f in os.listdir(os.path.join(cm.OUT_ROOT, "_profiles")) if f.startswith("id_")]   # parsed and deleted


def test_report_flags_sections_measured_on_another_export(tmp_path, monkeypatch):
    monkeypatch.setattr(cm, "REPORTS", str(tmp_path / "reports"))
    g = tmp_path / "g.onnx"
    g.write_bytes(b"first export")
    cm.write_report("r.json", {"parity_setup": {"graphs": cm.graph_stamp({"g": str(g)})}, "parity": {"a": {"snr_db": 50.0}},
                               "cost": {"graphs": cm.graph_stamp({"g": str(g)})}})
    assert "graph_mismatch" not in json.load(open(tmp_path / "reports" / "r.json", encoding="utf-8"))
    g.write_bytes(b"a re-export of the graph")
    cm.write_report("r.json", {"cost": {"graphs": cm.graph_stamp({"g": str(g)})}}, merge=True)      # a --no-parity rerun
    rep = json.load(open(tmp_path / "reports" / "r.json", encoding="utf-8"))
    assert [m["file"] for m in rep["graph_mismatch"]] == ["g.onnx"]
    assert set(rep["graph_mismatch"][0]["sha256"]) == {"parity_setup", "cost"}
    cm.write_report("r.json", {"parity_setup": {"graphs": cm.graph_stamp({"g": str(g)})}}, merge=True)  # parity rerun too
    assert "graph_mismatch" not in json.load(open(tmp_path / "reports" / "r.json", encoding="utf-8"))


def test_report_merge_keeps_what_was_not_recomputed():
    old = {"parity": {"a": 1}, "cost": {"x": {"host": 1}}, "env": {"ep": "cuda"}}
    new = {"cost": {"x": {"host": 2}}, "env": {"ep": "cuda", "onnxruntime": "1.28.0"}}
    assert cm._merge(old, new) == {"parity": {"a": 1}, "cost": {"x": {"host": 2}}, "env": {"ep": "cuda", "onnxruntime": "1.28.0"}}


# ---------------------------------------------------------------------------------------------
# MeanVC2
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def mv():
    return _load("meanvc", "export"), _load("meanvc", "sampler")


def test_meanvc_linspace_is_torch_bit_for_bit(mv):
    _, sp = mv
    for start, end, n in ((1.0, 0.0, 160), (0.0, 1.0, 160), (0.0, 1.0, 7), (1.0, 0.0, 2)):
        assert np.array_equal(sp.linspace_f32(start, end, n), torch.linspace(start, end, steps=n).numpy())


@pytest.mark.parametrize("t_in,t_out", [(5, 17), (2, 17), (3, 9), (9, 33)])
def test_meanvc_bn_upsampling_matches_torch(mv, t_in, t_out):
    _, sp = mv
    x = np.random.default_rng(t_in).standard_normal((1, t_in, 256)).astype(np.float32)
    ref = F.interpolate(torch.from_numpy(x).transpose(1, 2), size=t_out, mode="linear", align_corners=True).transpose(1, 2).numpy()
    np.testing.assert_allclose(sp.upsample_linear_align_corners(x, t_out), ref, rtol=0, atol=2e-6)


def test_meanvc_fbank_module_matches_kaldi(mv):
    ex, _ = mv
    import torchaudio.compliance.kaldi as kaldi
    t = np.arange(32000) / 16000.0
    wav = (0.3 * np.sin(2 * np.pi * 220 * t) + 0.05 * np.random.default_rng(0).standard_normal(t.size)).astype(np.float32)
    ref = kaldi.fbank(torch.from_numpy(wav * 32768)[None], frame_length=25, frame_shift=10, snip_edges=True, num_mel_bins=80,
                      energy_floor=0.0, dither=0.0, sample_frequency=16000).numpy()
    got = ex.KaldiFbank()(torch.from_numpy(wav)[None])[0].numpy()
    assert got.shape == ref.shape
    assert cm.compare(ref, got)["snr_db"] > 90


def test_meanvc_conv_istft_matches_torch_istft(mv):
    ex, _ = mv
    window = torch.hann_window(ex.N_FFT)
    basis, win_sq = ex.istft_kernels(ex.N_FFT, window)
    g = torch.Generator().manual_seed(0)
    spec = torch.complex(torch.randn(1, ex.N_FFT // 2 + 1, 12, generator=g), torch.randn(1, ex.N_FFT // 2 + 1, 12, generator=g))
    ref = torch.istft(spec, ex.N_FFT, ex.HOP, ex.N_FFT, window)
    got = ex.istft_conv(torch.cat([spec.real, spec.imag], 1), basis, win_sq, ex.HOP)
    assert got.shape == ref.shape
    assert cm.compare(ref.numpy(), got.numpy())["snr_db"] > 100


@pytest.mark.parametrize("t_p,t_f", [(2, 1), (2, 0), (1, 0), (0, 0)])
def test_meanvc_chunk_mask_matches_upstream(mv, t_p, t_f):
    ex, _ = mv
    from src.modules import BlockAttnProcessor
    for n in (8, 12, 20):
        up = BlockAttnProcessor(ex.CHUNK, ex.BLOCK, t_p, t_f).try_cached_mask(n, "cpu")
        assert torch.equal(ex._chunk_mask(n, t_p, t_f), up)


def _tiny_dit():
    """The upstream DiT class at a small width with seeded weights; the parameters upstream
    zero-initialises (adaLN, output projection) get small random values, or the output is 0."""
    from src.dit import DiT
    torch.manual_seed(0)
    dit = DiT(dim=64, depth=4, heads=2, dim_head=16, dropout=0.0, ff_mult=2, bn_dim=256, qk_norm="rms_norm",
              conv_layers=4, chunk_size=4, block_size=4).eval()
    with torch.no_grad():
        for m in [b.attn_norm.linear for b in dit.transformer_blocks] + [dit.norm_out.linear, dit.proj_out]:
            m.weight.normal_(0.0, 0.05)
            m.bias.normal_(0.0, 0.05)
    return dit


def test_meanvc_fixed_cache_dit_matches_upstream_kv_cache(mv):
    """Eight streaming chunks (the caches fill, then slide) with the upstream DiT and its growing,
    per-layer-trimmed KV cache vs the fixed 12-frame padded cache of the exported wrapper."""
    ex, _ = mv
    dit = _tiny_dit()
    wrap = ex.DitStream(dit).eval()
    heads, dh = 2, 16
    g = torch.Generator().manual_seed(1)
    spk = F.normalize(torch.randn(1, 256, generator=g), dim=-1)
    k_mem, v_mem = dit.gtm(spk)
    kv, cache, offset = None, torch.zeros(4, 2, heads, ex.CAP, dh), 0
    worst = math.inf
    with torch.no_grad():
        for _chunk in range(8):
            x = torch.randn(1, 8, 80, generator=g)
            cond = torch.randn(1, 8, 256, generator=g)
            for step, (tv, rv) in enumerate(((1.0, 0.5), (0.5, 0.0))):
                t, r = torch.tensor([tv]), torch.tensor([rv])
                u_ref, kv_new = dit(x, t, r, cache=None, cond=cond, spks=spk, offset=offset, is_inference=True, kv_cache=kv, gtm_kv=(k_mem, v_mem))
                u_got, cache_new = wrap(x, t, r, cond, spk, k_mem, v_mem, torch.tensor([offset]), torch.tensor([min(offset, ex.CAP)]), cache)
                worst = min(worst, cm.compare(u_ref.numpy(), u_got.numpy())["snr_db"] if not torch.equal(u_ref, u_got) else math.inf)
                x = x - 0.5 * u_ref
                if step == 1:
                    kv, cache = kv_new, cache_new
                    for i, (k, v) in enumerate(kv):  # upstream's (trimmed) cache is the tail of the fixed one
                        n = k.shape[2]
                        np.testing.assert_allclose(cache[i, 0, :, -n:].numpy(), k[0].numpy(), atol=1e-5)
                        np.testing.assert_allclose(cache[i, 1, :, -n:].numpy(), v[0].numpy(), atol=1e-5)
            offset += ex.CHUNK
    assert worst > 90


# ---------------------------------------------------------------------------------------------
# Seed-VC
# ---------------------------------------------------------------------------------------------

def test_seed_vc_linspace_is_torch_bit_for_bit():
    sp = _load("seed_vc", "sampler")
    for n in range(1, 31):
        assert np.array_equal(sp.linspace_f32(n), torch.linspace(0, 1, n + 1).numpy()), n


def test_seed_vc_preset_shapes_are_consistent():
    sp = _load("seed_vc", "sampler")
    for preset in sp.PRESETS.values():
        s = preset.shapes()
        assert s["L"] == s["prompt_frames"] + s["n_mel"]
        assert s["hift_samples"] == 256 * s["n_mel"]
        assert s["n16"] == 320 * s["window_units"]
        assert s["out_samples"] <= s["hift_samples"]
        # the partial decode keeps hift_margin whole frames before the first returned sample
        assert s["out_start"] - 256 * s["dec_start"] >= 256 * preset.hift_margin
        assert s["dec_samples"] < s["hift_samples"]
        n_this, n_prev = preset.join_anchor()
        assert s["out_start"] - 256 * s["dec_start"] <= n_this < n_prev < s["dec_samples"]


@pytest.fixture(scope="module")
def sv():
    return _load("seed_vc", "export"), _load("seed_vc", "sampler")


def _random_hift(ex, seed=0):
    """CosyVoice HiFT (the upstream class and config) with seeded random weights; Snake alphas
    spread over 1e-5..2 like trained ones (most of them are not representable in fp16)."""
    import yaml
    from modules.hifigan.f0_predictor import ConvRNNF0Predictor
    from modules.hifigan.generator import HiFTGenerator
    cfg = yaml.safe_load(open(ex.HIFT_CONFIG, encoding="utf-8"))
    torch.manual_seed(seed)
    h = HiFTGenerator(**cfg["hift"], f0_predictor=ConvRNNF0Predictor(**cfg["f0_predictor"])).eval()
    with torch.no_grad():
        for m in h.modules():
            if type(m).__name__ == "Snake":
                m.alpha.copy_(torch.exp(torch.empty_like(m.alpha).uniform_(math.log(1e-5), math.log(2.0))))
    return cm.freeze(h)


def _voiced_f0(m, base=150.0, seed=0):
    k = np.arange(m)
    return torch.from_numpy((base + 40 * np.sin(2 * np.pi * k / 23.0) + np.random.default_rng(seed).normal(0, 2, m)).astype(np.float32))[None]


def test_seed_vc_nsf_phase_helpers_match_the_graph_arithmetic(sv):
    ex, sp = sv
    hift = ex.Hift(_random_hift(ex))
    f0 = _voiced_f0(41)
    with torch.no_grad():
        ph = hift.nsf_phase(f0)[0].double().numpy()                     # [9, 256 M] cycles
    hp = np.random.default_rng(1).uniform(-np.pi, np.pi, (1, 9, 1)).astype(np.float32)
    for n in (0, 1, 255, 256, 3000, 41 * 256 - 1):
        want = (ph[:, n] + hp.reshape(-1).astype(np.float64) / (2 * np.pi)) % 1.0
        got = sp.nsf_cycles_at(f0.numpy(), hp, n)
        # float32 arithmetic on both sides (torch divides by a scalar through its reciprocal): 1e-5
        # cycles is 6e-5 rad
        assert np.max(np.abs((got - want + 0.5) % 1.0 - 0.5)) < 1e-5, n
    assert np.all(sp.nsf_start_cycles(f0.numpy(), 0) == 0)
    for d in (1, 5, 13, 40):
        # the phase before frame d is the phase the window reached at its last sample before d
        # (upstream's cumsum is inclusive: sample n carries n + 1 increments)
        assert np.max(np.abs((sp.nsf_start_cycles(f0.numpy(), d) - ph[:, 256 * d - 1] + 0.5) % 1.0 - 0.5)) < 2e-5, d


def test_seed_vc_harm_phase_for_anchor_sets_the_phase_at_the_anchor(sv):
    _, sp = sv
    rng = np.random.default_rng(2)
    f0 = _voiced_f0(28, 210.0, seed=3).numpy()
    for n in (0, 100, 3640, 7000):
        target = rng.uniform(0, 1, 9)
        hp = sp.harm_phase_for_anchor(f0, n, target)
        assert hp.shape == (1, 9, 1) and np.all(np.abs(hp) <= np.pi + 1e-6)
        got = sp.nsf_cycles_at(f0, hp, n)
        assert np.max(np.abs((got - target + 0.5) % 1.0 - 0.5)) < 2e-6


@pytest.mark.parametrize("name", ["upstream_readme", "sigil_low"])
def test_seed_vc_join_anchor_keeps_the_phase_continuous_where_slice_start_carry_drifts(sv, name):
    """Consecutive ticks whose F0 estimates differ by 1 Hz (180 vs 181 Hz): carrying the phase to
    the next slice's first sample leaves harmonic h 2 pi h (1 Hz) n_this / sr apart at the join (the
    drift the plan's SOLA seams come from); anchoring at the join leaves every harmonic aligned."""
    _, sp = sv
    preset = sp.PRESETS[name]
    w = preset.shapes()["dec_frames"]
    blk22 = int(round(preset.block * 50)) * sp.ZC_22K
    n_this, n_prev = preset.join_anchor()
    f0_k, f0_k1 = np.full((1, w), 180.0, np.float32), np.full((1, w), 181.0, np.float32)
    hp_k = np.random.default_rng(5).uniform(-np.pi, np.pi, (1, 9, 1)).astype(np.float32)
    target = sp.nsf_cycles_at(f0_k, hp_k, n_prev)                         # where tick k's excitation is at the join

    def err(hp):  # radians, per harmonic, wrapped to [-pi, pi)
        return ((sp.nsf_cycles_at(f0_k1, hp, n_this) - target + 0.5) % 1.0 - 0.5) * 2 * np.pi

    assert np.max(np.abs(err(sp.harm_phase_for_anchor(f0_k1, n_this, target)))) < 1e-4
    drift = err(sp.harm_phase_for_anchor(f0_k1, 0, sp.nsf_cycles_at(f0_k, hp_k, blk22)))
    h = np.arange(1, 10)
    want = ((h * 1.0 * n_this / sp.SR + 0.5) % 1.0 - 0.5) * 2 * np.pi
    np.testing.assert_allclose(drift, want, atol=2e-3)
    assert abs(drift[0]) > 0.5                                            # > 0.5 rad on the fundamental already


def test_seed_vc_hift_split_and_partial_decode_match_the_full_decode(sv):
    ex, sp = sv
    h = _random_hift(ex, seed=1)
    full, f0_graph, nsf = ex.Hift(h).eval(), ex.HiftF0(h).eval(), ex.HiftNsf(h).eval()
    for name in ("upstream_readme", "sigil_low"):
        preset = sp.PRESETS[name]
        s = preset.shapes()
        m = s["n_mel"]
        g = torch.Generator().manual_seed(m)
        mel = torch.randn(1, 80, m, generator=g) * 0.7 - 5.0
        hp = (torch.rand(1, 9, 1, generator=g) * 2 - 1) * math.pi
        hp[:, 0] = 0
        sn = torch.randn(1, 9, m * 256, generator=g)
        with torch.no_grad():
            y_full, f0 = full(mel, hp, sn)
            assert torch.equal(f0, f0_graph(mel))
            assert cm.compare(y_full.numpy(), nsf(mel, f0, hp, sn).numpy())["snr_db"] == "inf"     # split = whole graph
            f0v = _voiced_f0(m, seed=m)                                    # voiced, so the phase matters
            y_ref = nsf(mel, f0v, hp, sn)[0].numpy()
            d = s["dec_start"]
            phase = torch.from_numpy((hp.numpy().astype(np.float64) + 2 * np.pi * sp.nsf_start_cycles(f0v.numpy(), d).reshape(1, 9, 1)).astype(np.float32))
            y_part = nsf(mel[..., d:], f0v[:, d:], phase, sn[..., 256 * d:])[0].numpy()
        a, b = s["out_start"], s["hift_samples"] - s["skip_tail"] * sp.ZC_22K
        assert cm.compare(y_ref[a:b], y_part[a - 256 * d: b - 256 * d])["snr_db"] > 90, name
        # without the receptive-field margin the slice does not reproduce the window
        d0 = s["out_start"] // 256
        phase0 = torch.from_numpy((hp.numpy().astype(np.float64) + 2 * np.pi * sp.nsf_start_cycles(f0v.numpy(), d0).reshape(1, 9, 1)).astype(np.float32))
        with torch.no_grad():
            y0 = nsf(mel[..., d0:], f0v[:, d0:], phase0, sn[..., 256 * d0:])[0].numpy()
        assert cm.compare(y_ref[a:b], y0[a - 256 * d0: b - 256 * d0])["snr_db"] < 40, name


def test_seed_vc_fp16_hift_keeps_exact_fp32_snake_alpha(sv):
    ex, _ = sv
    h = _random_hift(ex, seed=2)
    exact = [m.alpha.detach().clone() for m in h.modules() if type(m).__name__ == "Snake"]
    assert any(not torch.equal(a.half().float(), a) for a in exact)       # fp16 would round them
    for cls in (ex.Hift, ex.HiftNsf):
        mod = cls(h, half=True)
        got = [m.alpha for m in mod.dec.modules() if type(m).__name__ == "Snake"]
        assert len(got) == len(exact)
        for a, b in zip(got, exact):
            assert a.dtype == torch.float32 and torch.equal(a, b)
        assert mod.dec.conv_pre.weight.dtype == torch.float16


def test_seed_vc_fp16_dit_keeps_the_timestep_sinusoid_exact(sv):
    """The DiT's fp16 export computes the timestep sinusoid from the exact fp32 ``freqs`` buffer and
    the exact fp32 ``t`` (upstream's autocast keeps both fp32); only the MLP after it is fp16."""
    ex, _ = sv
    from modules.diffusion_transformer import TimestepEmbedder

    class Dit(torch.nn.Module):
        style_as_token, time_as_token = True, True

        def __init__(self):
            super().__init__()
            self.t_embedder = TimestepEmbedder(64)

        def forward(self, x, prompt_x, x_lens, t, style, cond):
            self.seen_t = t
            return self.t_embedder(t)

    torch.manual_seed(0)
    dit = Dit()
    ref = copy_module(dit)
    exact_freqs = dit.t_embedder.freqs.clone()
    est = ex.Estimator(dit, half=True)
    te = est.dit.t_embedder
    assert te.freqs.dtype == torch.float32 and torch.equal(te.freqs, exact_freqs)
    assert te.mlp[0].weight.dtype == torch.float16
    t = torch.tensor([0.123456789, 0.987654321, 1.0 / 3.0], dtype=torch.float32)
    with torch.no_grad():
        assert torch.equal(te.timestep_embedding(t), ref.t_embedder.timestep_embedding(t))      # bit-exact sinusoid
        out = est(torch.zeros(3, 80, 4), torch.zeros(3, 80, 4), torch.tensor([4, 4, 4]), t, torch.zeros(3, 192), torch.zeros(3, 4, 384))
    assert est.dit.seen_t.dtype == torch.float32 and torch.equal(est.dit.seen_t, t)
    want = ref.t_embedder.mlp.half()(ref.t_embedder.timestep_embedding(t).half()).float()
    assert torch.equal(out, want)


def copy_module(m):
    import copy
    return copy.deepcopy(m)


# ---------------------------------------------------------------------------------------------
# DDSP-SVC
# ---------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def dd():
    return _load("ddsp_svc", "export"), _load("ddsp_svc", "sampler")


def test_ddsp_conv_stft_matches_torch_stft(dd):
    ex, _ = dd
    window = torch.hann_window(2048)
    x = torch.randn(1, 512 * 30, generator=torch.Generator().manual_seed(0))
    ref = torch.stft(x, 2048, 512, 2048, window, center=True, pad_mode="reflect", return_complex=True)
    y = F.conv1d(F.pad(x[:, None], (1024, 1024), mode="reflect"), ex._stft_basis(2048, window), stride=512)
    got = torch.complex(y[:, :1025], y[:, 1025:])
    assert got.shape == ref.shape
    assert cm.compare(torch.view_as_real(ref).numpy(), torch.view_as_real(got).numpy())["snr_db"] > 100


def test_ddsp_safe_sinc_matches_torch_sinc_without_nan(dd):
    ex, _ = dd
    x = torch.cat([torch.zeros(3), torch.linspace(-4, 4, 1001), torch.tensor([1e-30, -1e-30])])
    got = ex._safe_sinc(x)
    assert not torch.isnan(got).any()
    np.testing.assert_allclose(got.numpy(), torch.sinc(x).numpy(), atol=1e-6)


def test_ddsp_tta2x_matches_upstream_interleave(dd):
    _, sp = dd
    from ddsp.vocoder import Audio2ContentVec768L12TTA2X
    feats_for = {}

    def fake_hubert(audio):  # frames of a 400-sample window, hop 320, like ContentVec
        n = (audio.shape[-1] - 400) // 320 + 1
        feats_for.setdefault(n, torch.randn(1, n, 768, generator=torch.Generator().manual_seed(n)))
        return {"last_hidden_state": feats_for[n]}

    enc = Audio2ContentVec768L12TTA2X.__new__(Audio2ContentVec768L12TTA2X)
    enc.hubert = fake_hubert
    for n in (16000, 16160, 17600, 40000, 40100):
        ref = enc(torch.zeros(1, n)).numpy()
        a = fake_hubert(torch.zeros(1, n))["last_hidden_state"].numpy()
        b = fake_hubert(torch.zeros(1, n + 160))["last_hidden_state"].numpy()
        np.testing.assert_array_equal(sp.tta2x(a, b), ref)


@pytest.mark.parametrize("threshold_db", [-60.0, -45.0, -30.0, -12.0])
def test_ddsp_volume_mask_matches_upstream_gate(dd, threshold_db):
    _, sp = dd
    from ddsp.core import upsample
    from ddsp.vocoder import Volume_Extractor
    rng = np.random.default_rng(0)
    t = np.arange(44100 * 2) / 44100.0
    env = np.where((t % 0.5) < 0.2, 0.3, 0.003)                           # bursts and gaps
    audio = (env * np.sin(2 * np.pi * 220 * t) + 1e-4 * rng.standard_normal(t.size)).astype(np.float32)
    volume = Volume_Extractor(512, 1024).extract(audio)
    mask = (volume > 10 ** (threshold_db / 20)).astype("float")
    ref = upsample(torch.from_numpy(mask).float().unsqueeze(-1).unsqueeze(0), 512).squeeze(-1)
    n = 512 * 60
    got = sp.volume_mask(sp.volume_envelope(audio)[None], threshold_db, n)
    np.testing.assert_array_equal(got, ref[0, -n:].numpy())
    assert got.dtype == np.float32 and got.size == n


def test_ddsp_sampler_gates_by_default():
    sp = _load("ddsp_svc", "sampler")
    import inspect
    assert inspect.signature(sp.DdspSvc.tick).parameters["threshold_db"].default == -45.0   # SvcDDSP.infer's GUI default


@pytest.mark.parametrize("cfg", [
    dict(sampling_rate=44100, num_mels=128, n_fft=2048, win_size=2048, hop_size=512, fmin=40, fmax=16000),   # pc_nsf_hifigan_44.1k
    dict(sampling_rate=44100, num_mels=128, n_fft=2048, win_size=1024, hop_size=512, fmin=0, fmax=22050),
    dict(sampling_rate=44100, num_mels=80, n_fft=1024, win_size=1024, hop_size=512, fmin=40, fmax=16000),
])
def test_ddsp_mel_follows_the_vocoder_config(dd, cfg):
    """The ``ddsp`` graph's DDSP mel is upstream ``vocoder.extract`` (nvSTFT.get_mel with the
    vocoder's own config), whatever that config is, not the exporter's hard-coded STFT."""
    ex, _ = dd
    from nsf_hifigan.nvSTFT import STFT
    from reflow.vocoder import DotDict
    h = DotDict(cfg)
    x = torch.randn(1, 512 * 40, generator=torch.Generator().manual_seed(0)) * 0.1
    ref = STFT(h.sampling_rate, h.num_mels, h.n_fft, h.win_size, h.hop_size, h.fmin, h.fmax).get_mel(x)
    got = ex.VocoderMel(h)(x)
    assert got.shape == ref.shape
    assert cm.compare(ref.numpy(), got.numpy())["snr_db"] > 90
    shared = ex.VocoderMel(h, share=(2048, torch.hann_window(2048)))
    assert shared.shared == (h.n_fft == 2048 and h.win_size == 2048)


def test_ddsp_export_refuses_a_vocoder_that_does_not_fit_the_model(dd):
    ex, _ = dd
    from reflow.vocoder import DotDict
    args = DotDict({"data": {"sampling_rate": 44100, "block_size": 512}})
    ok = DotDict(dict(sampling_rate=44100, hop_size=512, n_fft=2048, win_size=2048))
    ex.check_vocoder_config(args, ok)
    for bad in (dict(sampling_rate=48000), dict(hop_size=480), dict(win_size=4096)):
        with pytest.raises(SystemExit):
            ex.check_vocoder_config(args, DotDict({**ok, **bad}))


def test_ddsp_mel2ph_matches_upstream_alignment(dd):
    _, sp = dd
    from ddsp.vocoder import Units_Encoder
    enc = Units_Encoder.__new__(Units_Encoder)
    enc.device, enc.resample_kernel = "cpu", {}
    enc.encoder_sample_rate, enc.encoder_hop_size = 16000, 160
    for n44 in (48510, 110250, 44100):
        n16 = math.ceil(n44 * 160 / 441)
        units = 2 * ((max(n16, 400) - 400) // 320 + 1)
        enc.model = lambda audio, u=units: torch.arange(u, dtype=torch.float32)[None, :, None].repeat(1, 1, 3)
        idx = enc.encode(torch.zeros(1, n44), 44100, 512)[0, :, 0].long().numpy()
        np.testing.assert_array_equal(sp.mel2ph(n44 // 512 + 1, units)[0], idx)
