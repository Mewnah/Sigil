# SPDX-License-Identifier: AGPL-3.0-or-later
"""Self-checks of the T6 tooling (no network, no competitor install, no GPU):

    python -m pytest scripts/vc-bench/tests -q
    python scripts/vc-bench/tests/test_vcbench.py          # without pytest

Covers the shared helpers (percentiles, sidecar schema, sweep shapes, corpus reading, BS.1770
loudness), the vc-rs / driver-timing ingestion, the EGG GCI extractor on a synthetic EGG, the FCPE
check's arithmetic, the export of a compare report, and fetch-public-corpus's dry run and its
synthetic-archive self-test.
"""

import json
import math
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
VC_BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(VC_BENCH))
sys.path.insert(0, VC_BENCH)

import vcbench_common as vb  # noqa: E402


def test_percentiles_match_numpy_linear():
    v = [5.0, 1.0, 3.0, 2.0, 4.0]
    p = vb.percentiles(v)
    assert p["p50"] == 3.0
    assert abs(p["p95"] - 4.8) < 1e-9 and abs(p["p99"] - 4.96) < 1e-9
    assert vb.percentiles([])["p50"] is None


def test_sweep_shapes_cap_the_crossfade():
    assert vb.parse_sweep("40, 60,100;250,40") == [40.0, 60.0, 100.0, 250.0]
    shapes = vb.sweep_shapes([40.0, 250.0], 50.0, 2500.0)
    assert shapes[0] == {"block_ms": 40.0, "crossfade_ms": 20.0, "extra_ms": 2500.0}
    assert shapes[1]["crossfade_ms"] == 50.0
    for bad in ("", "0", "-5", "abc"):
        try:
            vb.parse_sweep(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad!r} must be refused")


def test_sidecars_carry_the_schema_hashes_and_feasibility():
    d = tempfile.mkdtemp()
    inp = os.path.join(d, "in.wav")
    with open(inp, "wb") as f:
        f.write(b"RIFF....fake")
    side = vb.make_sidecar(system="rvc_webui", input_path=inp, voice_path=inp, config={"block_ms": 40.0, "transpose": 0},
                           call_ms={"p50": 10, "p95": 30, "p99": 41, "mean": 12}, latency_ms=95.0, latency_kind="algorithmic")
    assert side["schema"] == vb.SIDECAR_SCHEMA
    assert side["input_sha256"] == vb.sha256_file(inp) and side["voice_sha256"] == side["input_sha256"]
    assert set(side["call_ms"]) == {"p50", "p95", "p99"}, "only the keys compare.rs reads"
    assert side["feasible"] is False, "p99 41 ms on a 40 ms block is not real time"
    path = vb.write_sidecar(os.path.join(d, "out.wav"), side)
    assert path.endswith("out.wav.json") and json.load(open(path))["latency_kind"] == "algorithmic"
    assert vb.feasible({"p99": 39.9}, 40.0) is True and vb.feasible({}, 40.0) is None


def test_corpus_clips_and_git_commit():
    d = tempfile.mkdtemp()
    m = {"schema": vb.CORPUS_SCHEMA, "name": "t", "clips": [{"id": "a", "input": "clips/a.wav"}, {"id": "b", "input": os.path.join(d, "b.wav")}]}
    p = os.path.join(d, "corpus.json")
    json.dump(m, open(p, "w"))
    clips = vb.corpus_clips(p)
    assert clips[0] == ("a", os.path.join(d, "clips/a.wav")) and clips[1][1] == os.path.join(d, "b.wav")
    g = os.path.join(d, "repo", ".git", "refs", "heads")
    os.makedirs(g)
    open(os.path.join(d, "repo", ".git", "HEAD"), "w").write("ref: refs/heads/main\n")
    open(os.path.join(g, "main"), "w").write("0123abcd\n")
    assert vb.git_commit(os.path.join(d, "repo")) == "0123abcd"
    assert vb.git_commit(d) == ""


def test_loudness_is_bs1770():
    for rate in (16000, 48000):
        x = [0.1 * math.sin(2 * math.pi * 997 * i / rate) for i in range(rate * 2)]
        assert abs(vb.loudness_lufs(x, rate) + 23.01) < 0.05
    assert vb.loudness_lufs([0.0] * 100, 16000) is None
    shelf, rlb = vb.k_weighting(48000)
    assert abs(shelf[0] - 1.53512485958697) < 1e-8 and abs(rlb[3] + 1.99004745483398) < 1e-8


def _run(script, *args):
    r = subprocess.run([sys.executable, os.path.join(*script), *args], capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def test_vcrs_probe_ingestion():
    assert "passed" in _run((VC_BENCH, "vcrs_probe.py"), "--self-test")


def test_egg_gci_extractor_on_a_synthetic_egg():
    assert "passed" in _run((VC_BENCH, "egg_f0.py"), "--self-test")


def test_fcpe_check_arithmetic():
    assert "passed" in _run((VC_BENCH, "fcpe_check.py"), "--self-test")


def test_compare_export():
    assert "passed" in _run((VC_BENCH, "compare_export.py"), "--self-test")


def test_rvc_driver_paths_and_latency():
    import importlib.util

    spec = importlib.util.spec_from_file_location("rvc_webui_offline", os.path.join(VC_BENCH, "rvc_webui_offline.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    a = mod.parse_args(["--rvc-root", "r", "--pth", "v.pth", "--input", "in.wav", "--out", "o/x.wav"])
    assert mod.out_path(a, "c", 250.0, False) == "o/x.wav"
    assert mod.out_path(a, "c", 40.0, True) == "o/x.b40.wav"
    a = mod.parse_args(["--rvc-root", "r", "--pth", "v.pth", "--corpus", "c.json", "--out-dir", "o"])
    assert mod.out_path(a, "clip1", 60.0, True) == os.path.join("o", "b60", "clip1.wav")
    shape = {"block_frame": 12000, "sola_buffer_frame": 1920, "sola_search_frame": 480}
    # 250 ms + 40 ms + 10 ms - a mean 5 ms offset at 48 kHz.
    assert abs(mod.algorithmic_latency_ms(shape, [240, 240], 48000) - 295.0) < 1e-9


def _load(name):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, os.path.join(VC_BENCH, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_flush_blocks_and_padding():
    # A 250 ms loop delays by block + 40 ms SOLA buffer + 10 ms search: 2 blocks, plus the margin.
    assert vb.flush_blocks(300.0, 250.0) == 3
    assert vb.flush_blocks(90.0, 40.0) == 4          # ceil(2.25) + 1
    assert vb.flush_blocks(80.0, 40.0, margin=0) == 2  # an exact multiple is not rounded up
    for bad in ((10.0, 0.0), (-1.0, 40.0), (float("nan"), 40.0)):
        try:
            vb.flush_blocks(*bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} must be refused")
    import numpy as np

    audio = np.ones(25, dtype="float32")
    padded, n = vb.pad_blocks(audio, 10, 2)
    assert n == 3 and len(padded) == 50 and padded[:25].all() and not padded[25:].any()


class _FakeVoiceChanger:
    """A stateful stand-in for Applio's VoiceChanger: a rolling buffer that delays audio by
    ``delay`` samples (its crossfade / SOLA latency). ``created`` counts the instances."""

    created = 0

    def __init__(self, block_frame, delay=15):
        type(self).created += 1
        self.block, self.delay = block_frame, delay
        import numpy as np

        self.buf = np.zeros(delay, dtype="float32")

    def on_request(self, audio_input, f0_up_key, index_rate, protect, volume_envelope):
        import numpy as np

        both = np.concatenate([self.buf, audio_input])
        out, self.buf = both[: len(audio_input)], both[len(audio_input):]
        return out, 0.0, {}


def test_applio_driver_flushes_the_tail_and_starts_each_clip_cold():
    import numpy as np

    mod = _load("applio_offline")
    args = mod.parse_args(["--applio-root", "r", "--pth", "v.pth", "--input", "in.wav", "--out", "o.wav"])
    block, delay = 10, 15  # the loop delays audio by a block and a half
    _FakeVoiceChanger.created = 0
    make = lambda: _FakeVoiceChanger(block_frame=block, delay=delay)  # noqa: E731

    def clip(seed, n=40):
        return (np.random.default_rng(seed).uniform(0.2, 1.0, n)).astype("float32")

    a, b = clip(1), clip(2)
    flush = vb.flush_blocks(15.0, 10.0)  # a 15 ms delay on 10 ms blocks (the unit is arbitrary)
    assert flush == 3
    out_a, times_a, n_blocks, n_total = mod.convert_clip(make, a, block, flush, args, [])
    assert (n_blocks, n_total, len(times_a)) == (4, 7, 4), "only the clip's own blocks are timed"
    # The whole clip comes out, delayed; without the silence after it the last 15 samples never would.
    assert np.allclose(out_a[delay : delay + len(a)], a)
    short = mod.convert_clip(make, a, block, 0, args, [])[0]
    assert len(short) - delay < len(a), "without the flush the clip's last samples stay in the loop"
    # Every clip gets its own instance, and nothing of clip A is in clip B's rendition.
    out_b = mod.convert_clip(make, b, block, flush, args, [])[0]
    assert _FakeVoiceChanger.created == 3
    assert not out_b[:delay].any(), "clip B opened with the tail of clip A"
    assert np.allclose(out_b[delay : delay + len(b)], b)


def test_rvc_driver_builds_a_fresh_object_per_clip_from_the_last_one():
    mod = _load("rvc_webui_offline")
    args = mod.parse_args(["--rvc-root", "r", "--pth", "v.pth", "--input", "in.wav", "--out", "o.wav"])

    class RVC:  # RVC WebUI's infer/lib/rtrvc.py signature (last_rvc reuses the loaded models)
        def __init__(self, key, formant, pth_path, index_path, index_rate, n_cpu, inp_q, opt_q, config, last_rvc=None):
            self.last, self.cache_pitch = last_rvc, [0] * 4

    rtrvc = type("rtrvc", (), {"RVC": RVC})
    first = mod.make_rvc(rtrvc, args, config=object())
    assert first.last is None
    first.cache_pitch[0] = 99  # clip 1 leaves its pitch in the cache
    second = mod.make_rvc(rtrvc, args, config=object(), last=first)
    assert second is not first and second.last is first and second.cache_pitch == [0] * 4


def test_fetch_public_corpus_dry_run_and_self_test():
    out = _run((REPO, "scripts", "bench", "fetch-public-corpus.py"), "--dry-run")
    assert "test-clean.tar.gz" in out and "cmu_us_bdl_arctic" in out and "one at a time" in out
    assert "passed" in _run((REPO, "scripts", "bench", "fetch-public-corpus.py"), "--self-test")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)
