# SPDX-License-Identifier: AGPL-3.0-or-later
"""Checks of the template pipeline's own logic: the RefineGAN latent window, the measured receptive
fields, the no-lookahead seam rows, the parity bookkeeping export_rvc / export_matrix write into the
bundled layouts, and the Applio cross-check's precision and key handling (against a stand-in that
runs Sigil's port through the same harness; the vendored Applio code itself is never imported here).

No downloads: the tiny test configurations, the bundled templates and fixtures, and the dev-cache
voices where present (those tests skip otherwise). The whole-matrix gate is ``parity_test.py --all``
/ ``export_matrix.py``.

Run from the repo root: ``python -m pytest scripts/vc-templates/test_templates.py -q``
"""

import copy
import json
import os
import sys
import types
import warnings

import numpy as np
import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import export_matrix  # noqa: E402
import export_rvc  # noqa: E402
import parity_test as pt  # noqa: E402
import sigil_layout as sl  # noqa: E402
from rvc_arch import configs, split, vocoders  # noqa: E402

warnings.filterwarnings("ignore", message=".*weight_norm is deprecated.*")

BUNDLED = pt.DEFAULT_TEMPLATES_DIR
TINY_V2 = os.path.join(pt.REPO, "src-tauri", "test-fixtures", "vc", "tiny", "v2")


def _layout(root, tid):
    with open(os.path.join(root, tid, "layout.json"), encoding="utf-8") as f:
        return json.load(f)


def _tiny_net(tid, seed=1234):
    """The exporter's net for a template (random weights, weight_norm folded) and its margins."""
    spec = configs.TEMPLATES[tid]
    torch.manual_seed(seed)
    net = export_rvc.build_net(spec)
    export_rvc.perturb_(net, seed, 0.02)
    export_rvc.remove_weight_norm_(net)
    return spec, net, export_rvc.measure_margins(net, spec)


# ---------------------------------------------------------------------------------------------
# RefineGAN latent window
# ---------------------------------------------------------------------------------------------


def _refinegan_inputs(spec, net, module, r, gen):
    f = split.FLOW_MARGIN + module.dec_margin + r
    _, args, _ = export_rvc.flowdec_example(spec, net, f, gen, "cpu", spec["adain_pool"], module.per_frame, module.latent_offset)
    return f, args


def test_refinegan_reads_every_latent_frame_it_takes():
    """The latents cover only the latent window, and every frame of it reaches the returned audio
    (the old contract fed F frames of which the first dec_margin - main_margin were never read)."""
    spec, net, margins = _tiny_net("v2-16k-f0-refinegan")
    module = export_rvc.flowdec_module(spec, net, margins, torch.float32, "cpu", None)
    assert module.latent_offset == margins["dec_margin"] - margins["main_margin"] > 0
    gen = torch.Generator().manual_seed(3)
    f, args = _refinegan_inputs(spec, net, module, 4, gen)
    fz = f - module.latent_offset
    assert args[0].shape[2] == fz == split.FLOW_MARGIN + margins["main_margin"] + 4
    assert args[4].shape[1] == f  # f0 keeps the source window
    m_p = args[0].clone().requires_grad_(True)
    out = module(m_p, *args[1:])
    assert out.shape[1] == 4 * configs.upp(spec["config"])
    out.sum().backward()
    per_frame = m_p.grad.abs().sum(dim=(0, 1))
    assert bool((per_frame > 0).all()), f"latent frames never read: {torch.nonzero(per_frame == 0).flatten().tolist()}"


def test_refinegan_window_mismatch_fails_loudly():
    """Latents over the source window (F frames, the old contract) are an error, not a silent skip."""
    spec, net, margins = _tiny_net("v2-16k-f0-refinegan")
    module = export_rvc.flowdec_module(spec, net, margins, torch.float32, "cpu", None)
    gen = torch.Generator().manual_seed(3)
    f, args = _refinegan_inputs(spec, net, module, 4, gen)
    pad = torch.zeros(1, args[0].shape[1], module.latent_offset)
    old = [torch.cat([pad, a], dim=2) for a in args[:3]] + list(args[3:])
    with pytest.raises(RuntimeError):
        module(*old)


def test_tiny_refinegan_fixture_graph_takes_the_latent_window():
    """The exported fixture graph (ONNX Runtime): Fz-frame latents run, F-frame latents are refused."""
    tid = "v2-16k-f0-refinegan"
    layout = _layout(TINY_V2, tid)
    st = layout["streaming"]
    assert st["latent_offset"] == st["dec_margin"] - st["main_margin"] > 0
    ins = {i["name"]: i["shape"] for i in layout["graphs"]["flowdec"]["variants"]["fp32"]["inputs"]}
    assert ins["m_p"][2] == ins["logs_p"][2] == ins["z_noise"][2] == "Fz"
    assert ins["f0"][1] == ins["phase"][1] == ins["noise_offsets"][1] == "F"
    w32 = bytes(layout["weights"]["bytes_f32"])
    pool = np.zeros(st["adain"]["pool"]["count"], np.float32)
    sess, _keep = pt.ort_session(os.path.join(TINY_V2, tid), layout, "flowdec", "fp32", w32, sl.f16_weights(w32), pool)
    arch, r = layout["arch"], 3
    f = st["flow_margin"] + st["dec_margin"] + r
    c = arch["inter_channels"]

    def feeds(fz):
        return {"m_p": np.zeros((1, c, fz), np.float32), "logs_p": np.zeros((1, c, fz), np.float32),
                "z_noise": np.zeros((1, c, fz), np.float32), "g": np.zeros((1, arch["gin_channels"], 1), np.float32),
                "f0": np.full((1, f), 150.0, np.float32), "phase": np.zeros((1, f), np.float32),
                "src_noise": np.zeros((1, f * arch["upp"]), np.float32), "noise_offsets": np.zeros((1, f), np.int64)}

    (audio,) = sess.run(None, feeds(f - st["latent_offset"]))
    assert audio.shape == (1, r * arch["upp"])
    with pytest.raises(Exception):
        sess.run(None, feeds(f))


def test_bundled_layouts_declare_the_latent_window():
    for tid in configs.MATRIX:
        layout = _layout(BUNDLED, tid)
        st = layout["streaming"]
        ins = {i["name"]: i["shape"] for i in layout["graphs"]["flowdec"]["variants"]["fp32"]["inputs"]}
        if layout["arch"]["vocoder"] == "refinegan":
            assert st["latent_offset"] == st["dec_margin"] - st["main_margin"] > 0, tid
            assert ins["m_p"][2] == "Fz" and ins["f0"][1] == "F", tid
            assert st["excitation"]["f0_lookahead_frames"] == 1, tid
        else:
            assert st["latent_offset"] == 0, tid
            assert ins["m_p"][2] == "F", tid
            if layout["arch"]["f0"]:
                assert st["excitation"]["f0_lookahead_frames"] == 0, tid


# ---------------------------------------------------------------------------------------------
# Receptive fields
# ---------------------------------------------------------------------------------------------


def test_right_receptive_field_is_exact_for_the_latents():
    """dec_receptive_right_frames bounds the latents' reach: a change just past it leaves the centre
    frame's audio unchanged, a change at it does not (f0-less: the latents are the only input)."""
    spec, net, m = _tiny_net("v2-16k-nof0-hifigan")
    right = m["dec_receptive_right_frames"]
    assert right > 0 and m["dec_receptive_right_samples"] == right * configs.upp(spec["config"])
    dec = copy.deepcopy(net.dec).double()
    upp = configs.upp(spec["config"])
    frames, c = 64, 32
    gen = torch.Generator().manual_seed(1)
    z = torch.randn(1, configs.config_dict(spec["config"])["inter_channels"], frames, generator=gen, dtype=torch.float64)
    g = torch.randn(1, configs.config_dict(spec["config"])["gin_channels"], 1, generator=gen, dtype=torch.float64)

    def centre(zz):
        with torch.no_grad():
            return split._decoder_nof0(dec, zz, g)[0, 0, c * upp : (c + 1) * upp]

    base = centre(z)
    for k, changes in ((right, True), (right + 1, False)):
        zz = z.clone()
        zz[:, :, c + k] += 1.0
        assert bool((centre(zz) != base).any()) == changes, f"frame c + {k}"


@pytest.mark.parametrize("tid", ["v2-16k-f0-hifigan", "v2-16k-f0-mrf-hifigan", "v2-16k-f0-refinegan"])
def test_tiny_layouts_record_right_fields(tid):
    root = TINY_V2 if tid in configs.TINY_V2 else os.path.dirname(TINY_V2)
    st = _layout(root, tid)["streaming"]
    assert st["dec_receptive_right_frames"] >= 1
    assert st["latent_offset"] == (st["dec_margin"] - st["main_margin"] if tid.endswith("refinegan") else 0)
    assert st["flow_receptive_right"] <= st["flow_margin"]


# ---------------------------------------------------------------------------------------------
# Seam rows
# ---------------------------------------------------------------------------------------------


def test_edge_windows_pick_a_voiced_end_and_a_voicing_offset():
    f0 = np.full(200, 150.0)
    f0[61:80] = 0.0
    f0[151:158] = 0.0
    ends = pt.edge_windows(f0, 35)
    assert [(e, k) for e, k, _ in ends] == [(124, "voiced"), (151, "voicing offset")]
    assert all(r == pt.EDGE_RETURN_FRAMES for _, _, r in ends)
    for e, _, _ in ends:
        assert f0[e - 1] > 0  # the window's last frame is voiced
    # too little room: the voiced end is dropped, the offset keeps what fits (>= 4 frames)
    short = pt.edge_windows(f0, 140)
    assert short == [(151, "voicing offset", 11)]
    assert pt.edge_windows(f0, 150) == []


def test_edge_profile_counts_the_seam_from_the_window_end():
    upp, tol = 4, 1e-4
    want = np.zeros((1, 6 * upp))
    got = want.copy()
    assert pt.edge_profile(got, want, upp, tol)[1] == 0
    got[0, 3 * upp + 1] = 5e-4  # frame 3 of 6 off: the seam reaches 3 frames back from the end
    got[0, 5 * upp] = 1e-3
    d, seam = pt.edge_profile(got, want, upp, tol)
    assert seam == 3 and d.shape == (6,) and d[3] == 5e-4
    got[0, 0] = 1.0
    assert pt.edge_profile(got, want, upp, tol)[1] == 6  # saturated: the whole window


def test_parity_reports_ungated_seam_rows_on_the_refinegan_fixture():
    tid = "v2-16k-f0-refinegan"
    layout = _layout(TINY_V2, tid)
    state = pt.model_state(pt.load_checkpoint(os.path.join(TINY_V2, "tiny-refinegan.pth")))
    rep = pt.parity(os.path.join(TINY_V2, tid), layout, state, configs.TEMPLATES[tid], 200, 12, 5,
                    knn_rows=64, knn_queries=8, fp16_scaled=True, quiet=True)
    assert rep["ok"]
    assert rep["latent_offset"] == layout["streaming"]["latent_offset"]
    edge = [r for r in rep["rows"] if r["check"] == "edge"]
    assert edge and not any(r["gate"] for r in edge)
    windows = rep["right_edge"]["fp32"]
    assert windows and all(w["seam_frames"] >= 1 for w in windows)  # no lookahead never matches offline
    assert all(len(w["per_frame_max_abs"]) == w["returned_frames"] for w in windows)
    # the seam rows never decide parity
    assert pt.cpu_parity_by_variant(rep)["flowdec"]["fp32"] == "pass"
    assert export_matrix.class_parity(rep)["flowdec"]["fp32"] == "pass"
    # what export_matrix records: the summary, without the per-frame profiles
    slim = export_matrix.slim_report(rep)
    assert all("per_frame_max_abs" not in w for ws in slim["right_edge"].values() for w in ws)
    assert export_matrix.seam_frames(export_matrix.right_edge(slim)) == max(w["seam_frames"] for w in windows)
    assert export_matrix.slim_report(slim) == slim and export_matrix.slim_report(None) is None


# ---------------------------------------------------------------------------------------------
# Parity bookkeeping
# ---------------------------------------------------------------------------------------------


def test_export_rvc_claims_measured_parity_only_for_the_p1_template():
    parity, rec = export_rvc.measured_parity(export_rvc.P1_TEMPLATE)
    assert parity is export_rvc.PARITY and rec is export_rvc.RECOMMENDED
    for tid in configs.MATRIX:
        if tid == export_rvc.P1_TEMPLATE:
            continue
        parity, rec = export_rvc.measured_parity(tid)
        assert all(c == "untested" for g in parity.values() for v in g.values() for c in v.values()), tid
        assert all(v == "fp32" for g in rec.values() for v in g.values()), tid
    # the test-only fixtures stand in for a shipped template in the Rust precision-policy tests
    assert export_rvc.measured_parity("v2-16k-f0-hifigan")[0] is export_rvc.PARITY


def test_export_rvc_refuses_unmeasured_bundled_exports(monkeypatch):
    called = []
    monkeypatch.setattr(export_rvc, "export_template", lambda *a, **k: called.append(a))
    monkeypatch.setattr(sys, "argv", ["export_rvc.py", "--template", "v2-48k-f0-hifigan"])
    with pytest.raises(SystemExit, match="export_matrix.py --template v2-48k-f0-hifigan"):
        export_rvc.main()
    assert not called
    monkeypatch.setattr(sys, "argv", ["export_rvc.py", "--template", "v2-48k-f0-hifigan", "--out", "scratch-dir"])
    export_rvc.main()
    assert called and called[0][1] == "scratch-dir"


def test_bundled_layouts_carry_measured_parity_and_seam_evidence():
    """Every bundled layout's parity comes from a gate run (export_matrix), never the P1 table copied
    by a bare export; the recommendation never points at a failing variant; the streaming numbers to
    size lookahead are recorded, in matrix.json too."""
    with open(os.path.join(BUNDLED, "matrix.json"), encoding="utf-8") as f:
        index = {t["id"]: t for t in json.load(f)["templates"]}
    assert sorted(index) == sorted(configs.MATRIX)
    for tid in configs.MATRIX:
        layout = _layout(BUNDLED, tid)
        ev = layout.get("parity_evidence")
        assert ev and ev["cpu"] != "not measured", tid
        for graph, g in layout["graphs"].items():
            for cls, variant in g["recommended"].items():
                assert g["variants"][variant]["parity"][cls] == "pass", (tid, graph, cls)
        st = layout["streaming"]
        assert st["dec_receptive_right_frames"] >= 1, tid
        assert ev["right_edge"] and all(w["tol"] == pt.TOL["fp32"] for w in ev["right_edge"]), tid
        row = index[tid]
        assert row["latent_offset"] == st["latent_offset"], tid
        assert row["dec_receptive_right_frames"] == st["dec_receptive_right_frames"], tid
        assert row["seam_frames_fp32"] == max(w["seam_frames"] for w in ev["right_edge"]), tid


def test_layouts_match_the_schema():
    jsonschema = pytest.importorskip("jsonschema")
    with open(os.path.join(HERE, "layout_schema.json"), encoding="utf-8") as f:
        schema = json.load(f)
    roots = [(BUNDLED, t) for t in configs.MATRIX] + [(TINY_V2, t) for t in configs.TINY_V2]
    for root, tid in roots:
        jsonschema.validate(_layout(root, tid), schema)


# ---------------------------------------------------------------------------------------------
# The Applio cross-check
# ---------------------------------------------------------------------------------------------


def test_applio_state_maps_weight_norm_pairs_to_parametrizations():
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.old = torch.nn.utils.weight_norm(torch.nn.Conv1d(2, 3, 3))
            self.new = torch.nn.utils.parametrizations.weight_norm(torch.nn.Conv1d(3, 2, 3))

    ref = torch.nn.Module()
    ref.old = torch.nn.utils.weight_norm(torch.nn.Conv1d(2, 3, 3))
    ref.new = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 2, 3))
    state = {k: v.detach().clone() for k, v in ref.state_dict().items()}
    toy = Toy()
    mapped = pt.applio_state(state, set(toy.state_dict()))
    assert "new.parametrizations.weight.original0" in mapped and "new.parametrizations.weight.original1" in mapped
    assert "old.weight_g" in mapped and "old.weight_v" in mapped  # targets with the old hooks keep their names
    pt.load_exact(toy, mapped)
    x = torch.randn(1, 3, 9)
    assert torch.allclose(toy.new(x), ref.new(x))
    with pytest.raises(SystemExit, match="unexpected"):
        pt.load_exact(toy, dict(mapped, stray=torch.zeros(1)))
    missing = dict(mapped)
    missing.pop("old.bias")
    with pytest.raises(SystemExit, match="old.bias"):
        pt.load_exact(toy, missing)
    pt.load_exact(toy, mapped, ignore=("enc_q.",))  # ignored prefixes may be missing
    toy.enc_q = torch.nn.Linear(1, 1)
    pt.load_exact(toy, mapped)


class _PortAsApplio(torch.nn.Module):
    """A stand-in for Applio's ``Synthesizer``: Sigil's port behind Applio's constructor and ``infer``
    (noise drawn with ``torch.randn_like`` / ``torch.rand``, the generator's conv_post under torch's
    parametrized weight_norm, a training-only ``enc_q``). Built like Applio's code: in whatever
    default dtype is active."""

    def __init__(self, *config, use_f0, text_enc_hidden_dim, vocoder):
        super().__init__()
        voc = {"RefineGAN": "refinegan", "MRF HiFi-GAN": "mrf-hifigan"}[vocoder]
        spec = {"version": "v2", "f0": use_f0, "vocoder": voc, "config": list(config), "emb_channels": text_enc_hidden_dim}
        net = export_rvc.build_net(spec)
        self.enc_p, self.flow, self.dec, self.emb_g = net.enc_p, net.flow, net.dec, net.emb_g
        torch.nn.utils.remove_weight_norm(self.dec.conv_post)
        torch.nn.utils.parametrizations.weight_norm(self.dec.conv_post)
        self.enc_q = torch.nn.Linear(1, 1)
        self.voc = voc

    def infer(self, phone, lengths, pitch, nsff0, sid):
        g = self.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.enc_p(phone, pitch, lengths)
        z_p = (m_p + torch.exp(logs_p) * torch.randn_like(m_p) * 0.66666) * x_mask
        z = self.flow(z_p, x_mask, g=g, reverse=True)
        n = z.shape[-1] * self.dec.upp
        if self.voc == "refinegan":
            noise = torch.randn_like(torch.zeros(1, n, 1)).reshape(1, -1)
            out = self.dec.reference(z * x_mask, nsff0, g, noise, lambda i, layer, y: torch.randn_like(y))
        else:
            noise = torch.randn_like(torch.zeros(1, n, vocoders.MRF_HARMONICS + 1))
            ini = torch.rand(1, len(vocoders.MRF_PHASE_INI))[0]
            out = self.dec.reference(z * x_mask, nsff0, g, noise, ini)
        return out, x_mask, None


@pytest.mark.skipif(not os.path.isfile(pt.REFINEGAN_PTH), reason="Applio's RefineGAN pretrain is not in the dev cache")
def test_upstream_check_compares_at_matched_precision(monkeypatch):
    """An exact port passes the cross-check, although its own fp32 decode is beyond the gate from its
    float64 decode (the fp32-vs-float64 comparison this check used to make)."""
    pkg = types.ModuleType("applio_arch")
    mod = types.ModuleType("applio_arch.synthesizers")
    mod.Synthesizer = _PortAsApplio
    pkg.synthesizers = mod
    monkeypatch.setitem(sys.modules, "applio_arch", pkg)
    monkeypatch.setitem(sys.modules, "applio_arch.synthesizers", mod)
    err, floor = pt.upstream_applio_check("v2-48k-f0-refinegan")
    assert err < 1e-9, err
    assert floor > pt.APPLIO_TOL, floor  # why the comparison has to be float64 against float64
    assert torch.get_default_dtype() == torch.float32  # restored
