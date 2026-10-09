# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared helpers of the P7 scorers: JSON / JSON-lines I/O, file hashes, report traversal, tables."""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from typing import Iterable, Iterator


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list[dict]:
    """JSON lines (blank lines and ``#`` comment lines skipped)."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{n}: {e}") from e
    return rows


def write_json(path: str, value) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp, path)


def write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value) -> str:
    """sha256 of the canonical JSON of ``value`` (sorted keys, no whitespace, UTF-8): the
    ``spec_sha256`` rule of the speech-bench report."""
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def now_ms() -> int:
    return int(time.time() * 1000)


def iter_results(report: dict, task: str | None = None, modes: Iterable[str] | None = None) -> Iterator[tuple[dict, dict, dict]]:
    """``(run, result, engine_info)`` of a speech-bench report, optionally only engines of ``task``
    and runs of ``modes``."""
    engines = {e["id"]: e for e in report.get("engines", [])}
    wanted = set(modes) if modes else None
    for run in report.get("runs", []):
        if wanted is not None and run.get("mode") not in wanted:
            continue
        for result in run.get("results", []):
            info = engines.get(result.get("engine"), {"id": result.get("engine"), "task": None})
            if task is not None and info.get("task") != task:
                continue
            yield run, result, info


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile (the report's rule): ``sorted[ceil(p / 100 * n) - 1]``."""
    v = sorted(x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x)))
    if not v:
        return None
    k = max(1, math.ceil(p / 100.0 * len(v)))
    return v[min(k, len(v)) - 1]


def fmt_pct(rate: float | None, lo: float | None = None, hi: float | None = None, digits: int = 1) -> str:
    """``12.3 [10.9-13.8]`` (percent) or ``-``."""
    if rate is None:
        return "-"
    s = f"{rate * 100:.{digits}f}"
    if lo is not None and hi is not None:
        s += f" [{lo * 100:.{digits}f}-{hi * 100:.{digits}f}]"
    return s


def fmt_num(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c).replace("|", "\\|") for c in r) + " |")
    return "\n".join(out) + "\n"
