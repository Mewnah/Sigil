# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers of the competitor drivers in ``scripts/vc-bench`` (standard library only, so the
drivers can import it from any Python environment, the user's RVC WebUI or Applio venv included).

- ``sha256_file``: the hashes Sigil's compare mode checks (input and voice).
- ``git_commit``: the commit of a checkout, read from ``.git`` without the git binary.
- ``percentiles``: per-block compute p50 / p95 / p99.
- ``Sidecar`` / ``write_sidecar``: the JSON a driver writes next to each WAV
  (``<wav>.json``, schema ``sigil-vc-bench-sidecar/1``, read by ``bench/compare.rs``).
- ``parse_sweep`` / ``sweep_shapes``: the ``--sweep 40,60,100,250`` operating points.
- ``corpus_clips``: the clips of a ``sigil-compare-corpus/1`` manifest.
- ``loudness_lufs``: ITU-R BS.1770-4 integrated loudness (libebur128's K-weighting design).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time

SIDECAR_SCHEMA = "sigil-vc-bench-sidecar/1"
CORPUS_SCHEMA = "sigil-compare-corpus/1"


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root: str) -> str:
    """The checked-out commit of ``root`` (``.git/HEAD``, following one ref; packed refs too);
    empty when it is not a git checkout."""
    git = os.path.join(root, ".git")
    try:
        if os.path.isfile(git):  # a worktree: "gitdir: <path>"
            with open(git, encoding="utf-8") as f:
                line = f.read().strip()
            if line.startswith("gitdir:"):
                git = os.path.normpath(os.path.join(root, line[7:].strip()))
        with open(os.path.join(git, "HEAD"), encoding="utf-8") as f:
            head = f.read().strip()
        if not head.startswith("ref:"):
            return head
        ref = head[4:].strip()
        p = os.path.join(git, *ref.split("/"))
        if os.path.isfile(p):
            with open(p, encoding="utf-8") as f:
                return f.read().strip()
        packed = os.path.join(git, "packed-refs")
        if os.path.isfile(packed):
            with open(packed, encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split(" ")
                    if len(parts) == 2 and parts[1] == ref:
                        return parts[0]
    except OSError:
        pass
    return ""


def percentiles(values, qs=(50, 95, 99)) -> dict:
    """Linear-interpolated percentiles (numpy's default), as ``{"p50": .., "p95": .., "p99": ..}``."""
    v = sorted(float(x) for x in values)
    out = {}
    for q in qs:
        if not v:
            out[f"p{q}"] = None
            continue
        pos = (len(v) - 1) * q / 100.0
        lo = int(math.floor(pos))
        hi = min(lo + 1, len(v) - 1)
        out[f"p{q}"] = v[lo] + (v[hi] - v[lo]) * (pos - lo)
    return out


def feasible(call_ms: dict, block_ms: float):
    """Real-time feasibility of an offline operating point: per-block p99 compute below the block
    period (``None`` when unknown)."""
    p99 = call_ms.get("p99") if call_ms else None
    if p99 is None or not block_ms:
        return None
    return bool(p99 < block_ms)


def gpu_name() -> str:
    """The CUDA device torch sees (empty without torch or CUDA)."""
    try:
        import torch  # noqa: WPS433 (the driver's environment has it)

        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    return ""


def make_sidecar(
    *,
    system: str,
    app_version: str = "",
    commit: str = "",
    input_path: str,
    voice_path: str = "",
    config: dict,
    call_ms: dict | None = None,
    latency_ms: float | None = None,
    latency_kind: str = "",
    latency_note: str = "",
    gpu: str = "",
    output_rate: int | None = None,
    notes: list | None = None,
    extra: dict | None = None,
) -> dict:
    """The sidecar of one rendition (see the module docs)."""
    block = config.get("block_ms")
    side = {
        "schema": SIDECAR_SCHEMA,
        "system": system,
        "app_version": app_version,
        "commit": commit,
        "input": os.path.abspath(input_path),
        "input_sha256": sha256_file(input_path),
        "voice": os.path.abspath(voice_path) if voice_path else "",
        "voice_sha256": sha256_file(voice_path) if voice_path and os.path.isfile(voice_path) else "",
        "config": config,
        "call_ms": {k: (None if v is None else round(float(v), 4)) for k, v in (call_ms or {}).items() if k in ("p50", "p95", "p99")},
        "gpu": gpu,
        "latency_ms": latency_ms,
        "latency_kind": latency_kind,
        "latency_note": latency_note,
        "feasible": feasible(call_ms or {}, block) if block else None,
        "output_rate": output_rate,
        "host": platform.node(),
        "created_unix": int(time.time()),
        "notes": list(notes or []),
    }
    if extra:
        side.update(extra)
    return side


def write_sidecar(wav_path: str, sidecar: dict) -> str:
    path = wav_path + ".json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sidecar, f, indent=1)
    return path


def parse_sweep(text: str) -> list:
    """``"40,60,100,250"`` -> ``[40.0, 60.0, 100.0, 250.0]`` (ms, positive, unique, in order)."""
    out = []
    for part in (text or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if not (v > 0 and math.isfinite(v)):
            raise ValueError(f"sweep block {part!r} is not a positive number of ms")
        if v not in out:
            out.append(v)
    if not out:
        raise ValueError("an empty sweep")
    return out


def sweep_shapes(blocks_ms, crossfade_ms: float, extra_ms: float) -> list:
    """The operating points of a sweep: each block with the crossfade capped at half the block (a
    crossfade cannot exceed what one block returns) and the extra context as given (the same
    context for every point, so the sweep changes one thing)."""
    return [
        {"block_ms": b, "crossfade_ms": min(crossfade_ms, b / 2.0), "extra_ms": extra_ms}
        for b in blocks_ms
    ]


def flush_blocks(latency_ms: float, block_ms: float, margin: int = 1) -> int:
    """Silent blocks to feed after a clip so the loop emits the last of its speech.

    A block loop delays audio by up to ``latency_ms`` (block + crossfade / SOLA buffers): without
    feeding anything after the clip, the last ``latency_ms`` of speech never leaves the loop, and a
    250 ms-block render loses about half a second the 40 ms one keeps. ``ceil(latency / block)``
    blocks bring the final sample out, plus ``margin`` for rounding."""
    if not (block_ms > 0 and math.isfinite(block_ms) and latency_ms >= 0 and math.isfinite(latency_ms)):
        raise ValueError(f"flush_blocks({latency_ms!r}, {block_ms!r}): needs a positive block and a non-negative latency")
    return int(math.ceil(latency_ms / block_ms - 1e-9)) + max(0, int(margin))


def pad_blocks(audio, block_frame: int, flush: int):
    """``(padded, n_blocks)``: ``audio`` (mono) zero-padded to whole blocks of ``block_frame``
    samples plus ``flush`` silent blocks; ``n_blocks`` counts the blocks that hold the clip."""
    import numpy as np  # the drivers run where numpy is (the competitor's environment)

    n_blocks = (len(audio) + block_frame - 1) // block_frame
    padded = np.zeros((n_blocks + max(0, flush)) * block_frame, dtype="float32")
    padded[: len(audio)] = audio
    return padded, n_blocks


def corpus_clips(manifest_path: str) -> list:
    """``[(clip_id, absolute input path), ...]`` of a corpus manifest."""
    with open(manifest_path, encoding="utf-8") as f:
        m = json.load(f)
    if m.get("schema") != CORPUS_SCHEMA:
        raise SystemExit(f"{manifest_path}: schema {m.get('schema')!r} (expected {CORPUS_SCHEMA})")
    base = os.path.dirname(os.path.abspath(manifest_path))
    return [(c["id"], c["input"] if os.path.isabs(c["input"]) else os.path.join(base, c["input"])) for c in m["clips"]]


def point_snippet(name: str, system: str, wav_pattern: str, config: dict, app_version: str = "", tier: str = "offline") -> dict:
    """A ``points`` entry for the corpus manifest (metadata comes from the sidecars)."""
    return {"name": name, "system": system, "app_version": app_version, "tier": tier, "wav": wav_pattern, "config": config}


# ---- loudness (BS.1770-4; the same design as Sigil's bench/compare.rs) ----

def k_weighting(rate: int):
    """The two K-weighting biquads ``(b0, b1, b2, a1, a2)`` at ``rate`` (libebur128's design)."""
    f0, g, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = math.tan(math.pi * f0 / rate)
    vh = 10 ** (g / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    shelf = ((vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0)
    f0, q = 38.13547087602444, 0.5003270373238773
    k = math.tan(math.pi * f0 / rate)
    d = 1.0 + k / q + k * k
    rlb = (1.0, -2.0, 1.0, 2.0 * (k * k - 1.0) / d, (1.0 - k / q + k * k) / d)
    return shelf, rlb


def _biquad(x, c):
    try:
        from scipy.signal import lfilter  # fast path

        return lfilter([c[0], c[1], c[2]], [1.0, c[3], c[4]], x)
    except Exception:  # noqa: BLE001
        b0, b1, b2, a1, a2 = c
        y = [0.0] * len(x)
        x1 = x2 = y1 = y2 = 0.0
        for i, v in enumerate(x):
            y0 = b0 * v + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, v, y1, y0
            y[i] = y0
        return y


def loudness_lufs(x, rate: int):
    """Integrated loudness (LUFS) of a mono signal (a sequence of floats); ``None`` when shorter
    than one 400 ms block or silent."""
    block = int(round(0.4 * rate))
    step = int(round(0.1 * rate))
    if len(x) < block:
        return None
    y = list(x) if not hasattr(x, "dtype") else x
    for c in k_weighting(rate):
        y = _biquad(y, c)
    sq = [float(v) * float(v) for v in y]
    prefix = [0.0]
    acc = 0.0
    for v in sq:
        acc += v
        prefix.append(acc)
    z = [(prefix[i + block] - prefix[i]) / block for i in range(0, len(sq) - block + 1, step)]
    lufs = lambda ms: -0.691 + 10.0 * math.log10(max(ms, 1e-30))  # noqa: E731
    gated = [m for m in z if lufs(m) > -70.0]
    if not gated:
        return None
    rel = lufs(sum(gated) / len(gated)) - 10.0
    kept = [m for m in gated if lufs(m) > rel]
    return lufs(sum(kept) / len(kept)) if kept else None
