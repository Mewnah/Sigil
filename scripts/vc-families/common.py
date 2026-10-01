# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers of the zero-shot / singing family spike (P3-H, ADR 0006).

Each family folder (``seed_vc``, ``ddsp_svc``, ``meanvc``) has an ``export.py`` (upstream PyTorch ->
ONNX graphs) and a ``parity.py`` (the same input through the upstream PyTorch code and through the
exported graphs, error in dB SNR, and per-call cost). This module holds what they share:

- paths: inputs are read from ``%LOCALAPPDATA%/sigil-dev-cache`` (fetched once, one file at a
  time; nothing here downloads); exported graphs go to ``sigil-dev-cache/vc-families/<family>``
  (they carry fixed upstream weights and are large, so they are never committed); reports go to
  ``scripts/vc-families/reports`` (checked in: the evidence behind ADR 0006);
- stub modules for upstream imports that only training / GUI / other tokenizers use;
- ``recorded_random``: runs upstream code unchanged while recording every random tensor it draws,
  so the graphs can be fed the exact same noise;
- ONNX Runtime sessions per execution provider (CPU, CUDA through ORT 1.28 with CUDA 13.2 +
  cuDNN 9, DirectML), timing with and without IOBinding, error metrics and reports.

Runs use separate virtual environments (``--system-site-packages``) whose ``onnxruntime`` module
is the build under test: ``sigil-dev-cache/venv-ort128-cuda`` (onnxruntime-gpu 1.28.0, the ORT
version of the Rust engine; its CPU EP is the engine's CPU EP, so CPU evidence is measured there
too) and ``sigil-dev-cache/venv-ort128-dml`` (onnxruntime-directml 1.24.4, the newest DirectML
wheel). The system Python (torch 2.3.1+cu121, onnxruntime 1.19.2 CPU, onnx 1.17.0) exports the
graphs; ``env_info`` warns when a report is measured on another ORT version than the engine's.

Every report section that a run writes (parity, cost, profile) carries ``graphs``: the sha256 of
each graph file it used (``graph_stamp``). ``write_report(merge=True)`` keeps sections a run did not
recompute and records in ``graph_mismatch`` any graph whose hash differs between sections, so rows
measured on an older export are visible as such.
"""

from __future__ import annotations

import collections
import contextlib
import hashlib
import json
import math
import os
import statistics
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEV_CACHE = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache")
OUT_ROOT = os.path.join(DEV_CACHE, "vc-families")
REPORTS = os.path.join(HERE, "reports")
SAMPLES = os.path.join(DEV_CACHE, "models", "transcribe", "samples")
ENGINE_ORT = "1.28"   # ort =2.0.0-rc.13 (ADR 0001): the ONNX Runtime the Rust engine runs

# CUDA 13.2 runtime (toolkit) and cuDNN 9.26 for CUDA 13 (the P1 NVIDIA runtime pack as installed by
# the P2 dev app data); onnxruntime-gpu 1.28 links cudart64_13 / cublasLt64_13 / cudnn64_9 / cufft64_12.
CUDA_DLL_DIRS = [
    os.path.join(os.environ.get("CUDA_PATH_V13_2", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2"), "bin", "x64"),
    os.path.join(DEV_CACHE, "appdata-p2d", "runtimes", "nvidia-cuda13", "cuda-13.2.0_cudnn-9.26.0", "cuda13", "bin", "x64"),
]

EPS = {
    "cpu": ["CPUExecutionProvider"],
    # TF32 off: ORT's CUDA EP enables TF32 by default, which costs fp32 graphs ~10x their error (ADR 0004).
    "cuda": [("CUDAExecutionProvider", {"device_id": 0, "use_tf32": 0, "cudnn_conv_algo_search": "EXHAUSTIVE"}), "CPUExecutionProvider"],
    "dml": [("DmlExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"],
}


def family_out(family):
    path = os.path.join(OUT_ROOT, family)
    os.makedirs(path, exist_ok=True)
    return path


def need(path, hint):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path}: {hint}")
    return path


# ---------------------------------------------------------------------------------------------
# Upstream import stubs
# ---------------------------------------------------------------------------------------------

class Munch(dict):
    """Attribute access over a dict (what upstream's ``munch.Munch`` is used for)."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v


def install_stubs(spec):
    """Register placeholder modules: ``spec`` maps a dotted module name to a dict of attributes.
    Only for imports whose code the inference path never calls (training quantizers, GUI-side
    pitch trackers); calling a stubbed attribute raises."""

    def _unavailable(name):
        def f(*_a, **_k):
            raise RuntimeError(f"{name} is stubbed in the P3-H spike (not on the inference path)")
        return f

    for dotted, attrs in spec.items():
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):
            name = ".".join(parts[:i])
            if name not in sys.modules:
                mod = types.ModuleType(name)
                mod.__path__ = []  # a package, so submodules resolve
                sys.modules[name] = mod
                if i > 1:
                    setattr(sys.modules[".".join(parts[: i - 1])], parts[i - 1], mod)
        mod = sys.modules[dotted]
        for k, v in (attrs or {}).items():
            setattr(mod, k, v if v is not None else _unavailable(f"{dotted}.{k}"))


def freeze(module):
    """Inference only: no parameter needs grad, and the weight each old-style weight_norm hook
    cached at registration (a non-leaf tensor that blocks ``copy.deepcopy``) is recomputed as a
    leaf. The hook recomputes it identically on every forward, so values do not change."""
    import torch
    from torch.nn.utils.weight_norm import WeightNorm
    mods = module.values() if isinstance(module, dict) else [module]
    for mod in mods:
        for p in mod.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            for m in mod.modules():
                for hook in m._forward_pre_hooks.values():
                    if isinstance(hook, WeightNorm):
                        setattr(m, hook.name, hook.compute_weight(m).detach())
    return module


def strip_weight_norm(module):
    """Fold every weight_norm (hook or parametrization) of an export copy into plain weights."""
    import torch
    from torch.nn.utils import parametrize
    for m in module.modules():
        if parametrize.is_parametrized(m):
            for name in list(m.parametrizations.keys()):
                parametrize.remove_parametrizations(m, name, leave_parametrized=True)
        elif hasattr(m, "weight_g") and hasattr(m, "weight_v"):
            torch.nn.utils.remove_weight_norm(m)
    return module


# ---------------------------------------------------------------------------------------------
# Randomness capture
# ---------------------------------------------------------------------------------------------

@contextlib.contextmanager
def recorded_random(seed=0):
    """Record every tensor drawn through ``torch.randn``, ``torch.randn_like``, ``torch.rand``,
    ``torch.rand_like`` and ``torch.distributions.Uniform.sample`` while upstream code runs
    unchanged. Yields the list of ``(kind, tensor)`` in call order. The tensors are the objects
    upstream receives, so later in-place edits (e.g. SineGen zeroing the fundamental's random
    phase) are visible in the record."""
    import torch
    from torch.distributions import Uniform

    torch.manual_seed(seed)
    record = []
    orig = {n: getattr(torch, n) for n in ("randn", "randn_like", "rand", "rand_like")}
    orig_sample = Uniform.sample

    def wrap(name):
        fn = orig[name]

        def g(*a, **k):
            t = fn(*a, **k)
            record.append((name, t))
            return t
        return g

    def sample(self, sample_shape=torch.Size()):  # noqa: B008 (torch.distributions.Uniform.sample's own signature)
        t = orig_sample(self, sample_shape)
        record.append(("uniform", t))
        return t

    for n in orig:
        setattr(torch, n, wrap(n))
    Uniform.sample = sample
    try:
        yield record
    finally:
        for n, f in orig.items():
            setattr(torch, n, f)
        Uniform.sample = orig_sample


# ---------------------------------------------------------------------------------------------
# ONNX Runtime
# ---------------------------------------------------------------------------------------------

_DLL_DIRS_ADDED = set()


def import_ort(ep):
    """Import onnxruntime for ``ep`` (CUDA DLL folders go on the search path first)."""
    if ep == "cuda":
        for d in CUDA_DLL_DIRS:
            if os.path.isdir(d) and d not in _DLL_DIRS_ADDED:  # once per process (PATH is capped at 32767 chars)
                _DLL_DIRS_ADDED.add(d)
                os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")
                os.add_dll_directory(d)
    import onnxruntime as ort
    avail = ort.get_available_providers()
    want = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider", "dml": "DmlExecutionProvider"}[ep]
    if want not in avail:
        raise SystemExit(f"onnxruntime {ort.__version__} at {ort.__file__} has no {want}; run this EP in its "
                         f"venv (sigil-dev-cache/venv-ort128-cuda or venv-ort128-dml)")
    return ort


def session(path, ep="cpu", threads=2, dims=None, cuda_graph=False, profile_prefix=None):
    """An inference session the way the engine builds one: sequential execution, spinning off,
    ``threads`` intra-op threads on the CPU (the real-time policy is 1-2). ``dims`` pins symbolic
    dimensions (``add_free_dimension_override_by_name``, the engine's ``with_dimension_override``):
    shape arithmetic then folds away, which is what lets the CUDA EP capture a CUDA graph.
    ``profile_prefix`` turns on ORT's profiler (``profile_session_call``).

    The session records how it was built: ``sigil_dims`` (the pinned dimensions, {} if none) and,
    when DirectML refused the pinned graph and the session fell back to dynamic shapes,
    ``sigil_dims_refused`` (the requested dimensions and the error); ``session_info`` reads them
    into every cost row."""
    ort = import_ort(ep)
    so = ort.SessionOptions()
    if profile_prefix:
        so.enable_profiling = True
        so.profile_file_prefix = profile_prefix
    for k, v in (dims or {}).items():
        so.add_free_dimension_override_by_name(k, int(v))
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    so.add_session_config_entry("session.inter_op.allow_spinning", "0")
    so.log_severity_level = 3
    if ep == "dml":
        so.enable_mem_pattern = False
    providers = EPS[ep]
    if cuda_graph:
        opts = dict(EPS["cuda"][0][1], enable_cuda_graph="1")
        providers = [("CUDAExecutionProvider", opts)]
    try:
        s = ort.InferenceSession(path, sess_options=so, providers=providers)
    except Exception as e:  # noqa: BLE001
        if ep != "dml" or not dims:
            raise
        # DirectML's graph fusion refuses some graphs once their dimensions are pinned
        # ("The parameter is incorrect", DmlGraphFusionHelper); the unpinned graph runs.
        reason = str(e).splitlines()[0][-160:]
        print(f"note: {os.path.basename(path)}: DirectML refused pinned dimensions ({reason[-60:]}); unpinned")
        s = session(path, ep, threads, None, cuda_graph, profile_prefix)
        s.sigil_dims_refused = {"dims": dict(dims), "error": reason}
        return s
    s.sigil_dims = dict(dims or {})
    used = s.get_providers()[0]
    if ep != "cpu" and used == "CPUExecutionProvider":
        raise SystemExit(f"{path}: {ep} requested but the session runs on the CPU")
    if ep == "dml":
        # onnxruntime-directml 1.24.4: running a session after another DirectML session of the same
        # process was released crashed with an access violation inside run(), so these scripts
        # never release one (a finding for the engine's session lifetime, ADR 0006).
        _DML_KEEPALIVE.append(s)
    return s


_DML_KEEPALIVE = []


def session_info(sess):
    """How a session was built, for cost rows: ``dims_pinned`` (False when no dimension is pinned,
    including a DirectML session that fell back to dynamic shapes) and ``dims_refused``."""
    info = {"dims_pinned": bool(getattr(sess, "sigil_dims", {}))}
    refused = getattr(sess, "sigil_dims_refused", None)
    if refused:
        info["dims_pinned"] = False
        info["dims_refused"] = refused
    return info


def run(sess, feeds):
    names = {i.name for i in sess.get_inputs()}
    return sess.run(None, {k: v for k, v in feeds.items() if k in names})


TIMING_REPEATS = int(os.environ.get("SIGIL_FAMILY_TIMING_REPEATS", "3"))


def time_calls(fn, warm=5, runs=30, repeats=None):
    """Median / p90 / min of ``runs`` timed calls after ``warm`` untimed ones. The block is measured
    ``repeats`` times (default 3, ``SIGIL_FAMILY_TIMING_REPEATS``) and the block with the lowest
    median is kept: other processes on the shared dev rig only ever make a block slower, so the
    quickest block is the closest to this call's own cost. Every block's median is kept too."""
    for _ in range(warm):
        fn()
    blocks = []
    for _ in range(max(1, repeats or TIMING_REPEATS)):
        t = []
        for _ in range(runs):
            t0 = time.perf_counter()
            fn()
            t.append((time.perf_counter() - t0) * 1000.0)
        t.sort()
        blocks.append(t)
    t = min(blocks, key=statistics.median)
    return {"median_ms": round(statistics.median(t), 3), "p90_ms": round(t[int(0.9 * (len(t) - 1))], 3),
            "min_ms": round(t[0], 3), "runs": runs, "block_medians_ms": [round(statistics.median(b), 3) for b in blocks]}


def _bind(sess, feeds, out_shapes, host_ints=False):
    """IOBinding with device-resident inputs and outputs. ``host_ints`` binds integer inputs from
    host memory instead (the fallback when a device-bound int64 input is refused)."""
    import onnxruntime as ort
    io = sess.io_binding()
    keep = []
    for k, v in feeds.items():
        v = np.asarray(v)
        if not v.flags.c_contiguous:
            v = np.array(v, order="C")  # (np.ascontiguousarray would turn a 0-d scalar into shape [1])
        if host_ints and np.issubdtype(v.dtype, np.integer):
            keep.append(v)
            io.bind_cpu_input(k, v)
            continue
        ov = ort.OrtValue.ortvalue_from_numpy(v, "cuda", 0)
        keep.append(ov)
        io.bind_ortvalue_input(k, ov)
    for o in sess.get_outputs():
        if out_shapes is not None:
            ov = ort.OrtValue.ortvalue_from_shape_and_type(out_shapes[o.name], np.float32, "cuda", 0)
            keep.append(ov)
            io.bind_ortvalue_output(o.name, ov)
        else:
            io.bind_output(o.name, "cuda", 0)
    return io, keep


def timed_session_call(sess, feeds, ep, warm=5, runs=30, path=None, dims=None, threads=2):
    """Per-call cost: ``host`` = ``run()`` with numpy in and out (inputs copied to the device and
    outputs back every call); on CUDA also ``bound`` = IOBinding with device-resident tensors, and
    ``graph`` = the same with dimensions pinned (``dims``) and a captured CUDA graph replayed (the
    engine's plan for fixed shapes), or the reason capture failed."""
    names = {i.name for i in sess.get_inputs()}
    feeds = {k: v for k, v in feeds.items() if k in names}
    outs = sess.run(None, feeds)
    quiet = wait_for_quiet_gpu(max_wait_s=120, settle=2) if ep in ("cuda", "dml") else None
    out = {"host": time_calls(lambda: sess.run(None, feeds), warm, runs), "session": session_info(sess)}
    if quiet is not None:
        out["gpu_before"] = quiet
    if ep == "cuda":
        host_ints = False
        try:
            io, keep = _bind(sess, feeds, None)
            sess.run_with_iobinding(io)
        except Exception:  # noqa: BLE001 - an int64 shape input must stay on the host
            host_ints = True
            io, keep = _bind(sess, feeds, None, host_ints=True)
        out["bound"] = time_calls(lambda: sess.run_with_iobinding(io), warm, runs)
        if host_ints:
            out["bound"]["int_inputs_on_host"] = True
        if path is not None:
            try:
                sg = session(path, "cuda", threads, dims=dims, cuda_graph=True)
                shapes = {o.name: outs[i].shape for i, o in enumerate(sess.get_outputs())}
                iog, keep_g = _bind(sg, feeds, shapes, host_ints)
                sg.run_with_iobinding(iog)  # capture
                out["graph"] = time_calls(lambda: sg.run_with_iobinding(iog), warm, runs)
                del sg, iog, keep_g
            except Exception as e:  # noqa: BLE001 - recorded as the result
                out["graph"] = {"error": str(e).splitlines()[0][:200]}
    return out


def profile_session_call(path, feeds, ep, threads=2, dims=None, warm=2, runs=5):
    """ORT's own profile of ``runs`` calls (after ``warm``): wall time per run, the summed node
    ("kernel") time per run, its split by execution provider and by op type, how many graph nodes
    each provider was assigned, and the ops placed on the CPU; plus this process's CPU time per run
    and the GPU utilisation sampled during the runs. On a GPU EP a node's time is its dispatch on
    the CPU (DirectML records and queues work asynchronously), not its execution, so wall − kernel
    holds the GPU execution as well as any per-run work outside the nodes; the CPU time and the GPU
    utilisation tell the two apart. The profile file is parsed and deleted."""
    prof_dir = os.path.join(OUT_ROOT, "_profiles")
    os.makedirs(prof_dir, exist_ok=True)
    before = set(os.listdir(prof_dir))
    sess = session(path, ep, threads, dims=dims, profile_prefix=os.path.join(prof_dir, os.path.splitext(os.path.basename(path))[0]))
    names = {i.name for i in sess.get_inputs()}
    feeds = {k: v for k, v in feeds.items() if k in names}
    for _ in range(warm):
        sess.run(None, feeds)
    # Where a run's time goes when ORT's node times do not add up (a GPU EP's node time is its
    # dispatch on the CPU, not its execution): this process's CPU time per run, and the GPU's
    # utilisation sampled while the timed runs execute (NVIDIA tooling; None elsewhere).
    import threading
    samples, stop = [], threading.Event()

    def sample_gpu():
        while not stop.is_set():
            u = gpu_utilization()
            if u is None:
                return
            samples.append(u)
            stop.wait(0.25)

    sampler = threading.Thread(target=sample_gpu, daemon=True) if ep != "cpu" else None
    if sampler:
        sampler.start()
    wall = []
    cpu0 = time.process_time()
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, feeds)
        wall.append((time.perf_counter() - t0) * 1000.0)
    cpu_ms = (time.process_time() - cpu0) * 1000.0 / runs
    if sampler:
        stop.set()
        sampler.join()
    pf = sess.end_profiling()
    with open(pf, encoding="utf-8") as f:
        events = json.load(f)
    # this call's profile, and the one a session that failed to build (DirectML refusing pinned
    # dimensions) left behind
    for name in set(os.listdir(prof_dir)) - before:
        with contextlib.suppress(OSError):
            os.remove(os.path.join(prof_dir, name))
    # events of the timed runs only: the last ``runs`` model_run spans
    model_runs = sorted((e for e in events if e.get("cat") == "Session" and e.get("name") == "model_run"), key=lambda e: e["ts"])
    timed = model_runs[-runs:]
    t_lo = timed[0]["ts"] if timed else 0
    nodes = collections.defaultdict(set)
    k_ep, k_op = collections.Counter(), collections.Counter()
    cpu_ops = collections.Counter()
    for e in events:
        if e.get("cat") != "Node" or not e["name"].endswith("_kernel_time"):
            continue
        a = e.get("args", {})
        prov, op = a.get("provider", "?"), a.get("op_name", "?")
        node = e["name"][: -len("_kernel_time")]
        if node not in nodes[prov]:
            nodes[prov].add(node)
            if prov == "CPUExecutionProvider" and ep != "cpu":
                cpu_ops[op] += 1
        if e["ts"] >= t_lo:
            k_ep[prov] += e["dur"] / 1000.0 / max(1, len(timed))
            k_op[(op, prov)] += e["dur"] / 1000.0 / max(1, len(timed))
    out = {
        "runs": runs,
        "wall_ms_per_run": round(statistics.median(wall), 3),
        "model_run_ms_per_run": round(statistics.median(e["dur"] / 1000.0 for e in timed), 3) if timed else None,
        "kernel_ms_per_run": round(sum(k_ep.values()), 3),
        "kernel_ms_per_run_by_provider": {k: round(v, 3) for k, v in k_ep.items()},
        "nodes_by_provider": {k: len(v) for k, v in nodes.items()},
        "top_ops_ms_per_run": [{"op": op, "provider": pv, "ms": round(ms, 3)} for (op, pv), ms in k_op.most_common(12)],
        "session": session_info(sess),
        "graphs": graph_stamp({os.path.splitext(os.path.basename(path))[0]: path}),
    }
    if cpu_ops:
        out["cpu_placed_ops"] = dict(cpu_ops.most_common())
    out["overhead_ms_per_run"] = round(out["wall_ms_per_run"] - out["kernel_ms_per_run"], 3)
    out["process_cpu_ms_per_run"] = round(cpu_ms, 3)
    if samples:
        out["gpu_util_pct_during"] = {"median": statistics.median(samples), "max": max(samples), "samples": len(samples)}
    return out


def best_ms(row):
    """The cheapest measured way to run a call (graph > bound > host), in ms."""
    for k in ("graph", "bound", "host"):
        if k in row and "median_ms" in row[k]:
            return row[k]["median_ms"], k
    raise ValueError(row)


# ---------------------------------------------------------------------------------------------
# Metrics and reports
# ---------------------------------------------------------------------------------------------

def compare(ref, out):
    """``snr_db`` = 10 log10(sum ref^2 / sum (ref - out)^2), plus max |d|, relative L2 and cosine.

    A non-finite value on either side makes the comparison meaningless: ``snr_db`` is then "nan"
    (which ``gate`` fails) with the counts in ``non_finite`` (candidate) and ``ref_non_finite``
    (reference). "inf" means only that both sides are finite and identical."""
    ref = np.asarray(ref, np.float64).ravel()
    out = np.asarray(out, np.float64).ravel()
    if ref.shape != out.shape:
        raise ValueError(f"shape mismatch {ref.shape} vs {out.shape}")
    bad_out, bad_ref = int(np.sum(~np.isfinite(out))), int(np.sum(~np.isfinite(ref)))
    if bad_out or bad_ref:
        fin = ref[np.isfinite(ref)]
        return {"snr_db": "nan", "max_abs": "nan", "rel_l2": "nan", "cosine": "nan",
                "non_finite": bad_out, "ref_non_finite": bad_ref,
                "ref_rms": float(f"{np.sqrt(np.mean(fin * fin)) if fin.size else float('nan'):.4g}")}
    d = ref - out
    num, den = float(np.sum(ref * ref)), float(np.sum(d * d))
    snr = math.inf if den == 0 else 10.0 * math.log10(max(num, 1e-300) / den)
    nr, no = float(np.linalg.norm(ref)), float(np.linalg.norm(out))
    return {
        "snr_db": "inf" if snr == math.inf else (round(snr, 2) if math.isfinite(snr) else "nan"),
        "max_abs": float(f"{np.max(np.abs(d)):.3g}"),
        "rel_l2": float(f"{math.sqrt(den) / max(nr, 1e-300):.3g}"),
        "cosine": float(f"{float(ref @ out) / max(nr * no, 1e-300):.9f}"),
        "ref_rms": float(f"{nr / math.sqrt(ref.size):.4g}"),
    }


def gate(row, snr_min=35.0):
    """``pass`` = finite on both sides and SNR >= ``snr_min`` dB ("inf": identical)."""
    s = row["snr_db"]
    finite = not row.get("non_finite") and not row.get("ref_non_finite")
    if s == "inf":
        ok = True
    elif isinstance(s, (int, float)) and not isinstance(s, bool):
        ok = math.isfinite(s) and s >= snr_min
    else:
        ok = False
    row["pass"] = bool(finite and ok)
    return row


def row_line(name, row, width=30):
    """One parity row for the console (robust to "nan" / "inf" and to rows without an SNR)."""
    if "snr_db" not in row:
        return f"   {name:{width}s} {row}"
    mx = row.get("max_abs")
    mx = f"{mx:.3g}" if isinstance(mx, float) else str(mx)
    lm = row.get("logmel", {}).get("snr_db", "")
    bad = "".join(f"  {k} {row[k]}" for k in ("non_finite", "ref_non_finite") if row.get(k))
    verdict = "" if "pass" not in row else ("ok" if row["pass"] else "FAIL")
    return f"   {name:{width}s} snr {row['snr_db']!s:>8} dB  max|d| {mx}  {verdict}" + (f"  logmel {lm}" if lm != "" else "") + bad


def logmel(wav, sr, n_fft=1024, hop=256, n_mels=80):
    """log(1e-5 + mel power) of a waveform: a phase-insensitive view for audio rows (a small F0
    difference drifts the NSF phase over a window and decorrelates waveforms that sound the same)."""
    import librosa
    m = librosa.feature.melspectrogram(y=np.asarray(wav, np.float32), sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels, power=2.0)
    return np.log(m + 1e-5)


def compare_audio(ref, out, sr):
    """Waveform comparison plus the same comparison on log-mel spectra (``logmel``)."""
    row = compare(ref, out)
    if row["snr_db"] != "nan":
        row["logmel"] = compare(logmel(ref, sr), logmel(out, sr))
    return row


def load_wav16k(name="jfk.wav", seconds=None):
    """A speech sample from the dev cache (the STT samples), mono float32 at 16 kHz."""
    import soundfile as sf
    path = need(os.path.join(SAMPLES, name), "the STT speech samples (models/transcribe/samples) are fetched by P2")
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=16000).astype(np.float32)
    if seconds is not None:
        x = x[: int(seconds * 16000)]
    return np.ascontiguousarray(x, np.float32)


def resample(x, sr_from, sr_to):
    import librosa
    return np.ascontiguousarray(librosa.resample(np.asarray(x, np.float32), orig_sr=sr_from, target_sr=sr_to), np.float32)


def env_info(ep):
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception:
        pass
    try:
        ort = import_ort(ep)
        info["onnxruntime"] = ort.__version__
        if ep != "dml" and not ort.__version__.startswith(ENGINE_ORT + "."):
            info["warning"] = f"measured on onnxruntime {ort.__version__}; the engine runs {ENGINE_ORT}"
            print(f"WARNING: onnxruntime {ort.__version__} is not the engine's {ENGINE_ORT}: run this in "
                  f"sigil-dev-cache/venv-ort128-cuda (its CPU EP is the engine's) for engine evidence")
    except SystemExit:
        pass
    info["ep"] = ep
    return info


_SHA = {}


def graph_sha256(path):
    """sha256 of a graph file (cached per path, size and mtime within a run)."""
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 23), b""):
                h.update(chunk)
        _SHA[key] = h.hexdigest()
    return _SHA[key]


def graph_stamp(files):
    """{graph name: {file, bytes, sha256}} for the graph files a report section measured."""
    return {name: {"file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": graph_sha256(p)}
            for name, p in sorted(files.items())}


def _stamps(obj, where=""):
    """Every ``graphs`` stamp in a report: (section path, stamp)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            here = f"{where}/{k}" if where else k
            if k == "graphs" and isinstance(v, dict) and all(isinstance(x, dict) and "sha256" in x for x in v.values()):
                yield where or "/", v
            else:
                yield from _stamps(v, here)


def graph_mismatches(report):
    """Graph files whose sha256 differs between the sections of one report (a merged report whose
    older sections were measured on another export), as [{file, sha256: {section: hash}}]."""
    seen = collections.defaultdict(dict)
    for where, stamp in _stamps(report):
        for v in stamp.values():
            seen[v["file"]][where] = v["sha256"]
    return [{"file": f, "sha256": secs} for f, secs in sorted(seen.items()) if len(set(secs.values())) > 1]


def _merge(old, new):
    for k, v in new.items():
        if isinstance(v, dict) and isinstance(old.get(k), dict):
            _merge(old[k], v)
        else:
            old[k] = v
    return old


def gpu_utilization():
    """Current NVIDIA GPU utilisation in percent (nvidia-smi), or None when it cannot be read."""
    import subprocess
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
        return max(int(x) for x in out.split() if x.strip().isdigit())
    except Exception:  # noqa: BLE001 - no NVIDIA tooling: unknown
        return None


def wait_for_quiet_gpu(max_wait_s=600, threshold=40, settle=3):
    """Before GPU timing: wait (up to ``max_wait_s``) until utilisation stays under ``threshold`` %
    for ``settle`` consecutive one-second samples, so another process's benchmark (which holds the
    GPU near 100 %) does not skew the numbers, and ours does not skew theirs. The threshold leaves
    room for the desktop's own background load (browser and compositor, 25-30 % on the dev rig).
    Returns what was seen, for the report."""
    t0 = time.time()
    quiet = 0
    seen = None
    while time.time() - t0 < max_wait_s:
        seen = gpu_utilization()
        if seen is None:
            return {"gpu_util_pct": None, "waited_s": 0}
        quiet = quiet + 1 if seen < threshold else 0
        if quiet >= settle:
            return {"gpu_util_pct": seen, "waited_s": round(time.time() - t0, 1)}
        time.sleep(1.0)
    return {"gpu_util_pct": seen, "waited_s": round(time.time() - t0, 1), "note": "GPU still busy; numbers may be skewed"}


def cpu_utilization(interval=1.0):
    """Whole-machine CPU utilisation in percent over ``interval`` seconds (Windows GetSystemTimes),
    or None elsewhere."""
    try:
        import ctypes
        from ctypes import wintypes

        def times():
            idle, kernel, user = wintypes.FILETIME(), wintypes.FILETIME(), wintypes.FILETIME()
            if not ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
                raise OSError("GetSystemTimes failed")
            return [(ft.dwHighDateTime << 32) | ft.dwLowDateTime for ft in (idle, kernel, user)]

        i0, k0, u0 = times()
        time.sleep(interval)
        i1, k1, u1 = times()
        total = (k1 - k0) + (u1 - u0)  # kernel time includes idle time
        return round(100.0 * (total - (i1 - i0)) / max(total, 1), 1)
    except Exception:  # noqa: BLE001 - not Windows: unknown
        return None


def timing_conditions(ep):
    """Machine load around a cost run (other agents and apps share the dev rig): on GPU EPs first
    wait for a quiet GPU; record CPU and GPU utilisation."""
    out = wait_for_quiet_gpu() if ep in ("cuda", "dml") else {"gpu_util_pct": gpu_utilization()}
    out["cpu_util_pct"] = cpu_utilization()
    out["cpu_threads"] = os.cpu_count()
    return out


def write_report(name, obj, merge=False):
    """Write ``reports/<name>``. ``merge``: keep what the existing report holds and this run did not
    recompute (a ``--no-parity`` cost rerun keeps the parity rows), recursively, new values winning.
    Sections whose ``graphs`` stamps disagree are listed in ``graph_mismatch`` (and warned about)."""
    os.makedirs(REPORTS, exist_ok=True)
    path = os.path.join(REPORTS, name)
    if merge and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            obj = _merge(json.load(f), obj)
    obj.pop("graph_mismatch", None)
    mism = graph_mismatches(obj)
    if mism:
        obj["graph_mismatch"] = mism
        print(f"WARNING: {name}: sections measured on different exports of " + ", ".join(m["file"] for m in mism)
              + " (see graph_mismatch); rerun them")
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        f.write("\n")
    return path
