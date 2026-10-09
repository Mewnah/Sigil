# SPDX-License-Identifier: AGPL-3.0-or-later
"""Bootstrap confidence intervals for corpus-level error rates and paired system comparisons.

Items (utterances, lines) are resampled with replacement ``B`` times; each resample's statistic is
the corpus rate ``sum(errors) / sum(units)`` of the drawn items (never a mean of per-item rates).
The 95 % interval is the 2.5th / 97.5th percentile of the resampled statistic (percentile
bootstrap). A paired comparison draws the same items for both systems in every resample, so
utterance difficulty cancels; it reports the observed difference, its interval, the share of
resamples where system A is better (lower), and a two-sided p-value
``2 * min(P(delta* <= 0), P(delta* >= 0))`` (capped at 1). Everything is seeded: the same inputs
and seed give the same numbers. Needs numpy.
"""

from __future__ import annotations

import numpy as np

DEFAULT_B = 1000
DEFAULT_SEED = 20261002


def _rate(err: np.ndarray, units: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(units > 0, err / np.maximum(units, 1), np.nan)


def _indices(n: int, b: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(b, n))


def corpus_rate(errors, units) -> float | None:
    e = float(np.sum(errors))
    u = float(np.sum(units))
    return None if u == 0 else e / u


def bootstrap_ci(errors, units, b: int = DEFAULT_B, seed: int = DEFAULT_SEED, alpha: float = 0.05):
    """``(rate, lo, hi)`` of the corpus rate of per-item ``errors`` / ``units``; ``(None, None,
    None)`` for an empty set or no reference units."""
    e = np.asarray(errors, dtype=np.float64)
    u = np.asarray(units, dtype=np.float64)
    if e.size == 0 or u.sum() == 0:
        return None, None, None
    rate = float(e.sum() / u.sum())
    if e.size == 1 or b <= 0:
        return rate, rate, rate
    idx = _indices(e.size, b, seed)
    stats = _rate(e[idx].sum(axis=1), u[idx].sum(axis=1))
    stats = stats[np.isfinite(stats)]
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return rate, float(lo), float(hi)


def paired_bootstrap(a_errors, a_units, b_errors, b_units, b: int = DEFAULT_B, seed: int = DEFAULT_SEED, alpha: float = 0.05) -> dict:
    """Paired comparison of systems A and B on the same items (lists aligned item by item).

    ``units`` may differ between A and B only when the systems were scored with different
    references; normally they are equal. Lower is better (error rates)."""
    ae = np.asarray(a_errors, dtype=np.float64)
    au = np.asarray(a_units, dtype=np.float64)
    be = np.asarray(b_errors, dtype=np.float64)
    bu = np.asarray(b_units, dtype=np.float64)
    if not (ae.size == au.size == be.size == bu.size):
        raise ValueError("paired bootstrap needs item-aligned arrays of equal length")
    n = ae.size
    out = {"items": int(n), "a": None, "b": None, "delta": None, "ci95": [None, None], "prob_a_better": None, "p_two_sided": None, "resamples": int(b), "seed": int(seed)}
    if n == 0 or au.sum() == 0 or bu.sum() == 0:
        return out
    ra, rb = float(ae.sum() / au.sum()), float(be.sum() / bu.sum())
    out.update(a=ra, b=rb, delta=ra - rb)
    if n == 1 or b <= 0:
        out.update(ci95=[ra - rb, ra - rb], prob_a_better=float(ra < rb), p_two_sided=1.0)
        return out
    idx = _indices(n, b, seed)
    da = _rate(ae[idx].sum(axis=1), au[idx].sum(axis=1))
    db = _rate(be[idx].sum(axis=1), bu[idx].sum(axis=1))
    delta = da - db
    delta = delta[np.isfinite(delta)]
    lo, hi = np.percentile(delta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    p_le = float(np.mean(delta <= 0))
    p_ge = float(np.mean(delta >= 0))
    out.update(
        ci95=[float(lo), float(hi)],
        prob_a_better=float(np.mean(delta < 0)),
        p_two_sided=min(1.0, 2 * min(p_le, p_ge)),
    )
    return out


def paired_bootstrap_stat(stat_fn, a_stats, b_stats, b: int = DEFAULT_B, seed: int = DEFAULT_SEED, alpha: float = 0.05, higher_is_better: bool = True) -> dict:
    """Paired bootstrap of any corpus statistic computed from summable per-item sufficient
    statistics (chrF's n-gram counts): ``a_stats`` / ``b_stats`` are arrays [items, k] and
    ``stat_fn(summed [k] vector) -> float``. ``prob_a_better`` follows ``higher_is_better``."""
    sa = np.asarray(a_stats, dtype=np.float64)
    sb = np.asarray(b_stats, dtype=np.float64)
    if sa.shape != sb.shape or sa.ndim != 2:
        raise ValueError("paired statistics need equal [items, k] arrays")
    n = sa.shape[0]
    va, vb = float(stat_fn(sa.sum(axis=0))), float(stat_fn(sb.sum(axis=0)))
    out = {"items": int(n), "a": va, "b": vb, "delta": va - vb, "ci95": [va - vb, va - vb], "prob_a_better": None, "p_two_sided": 1.0, "resamples": int(b), "seed": int(seed)}
    if n < 2 or b <= 0:
        out["prob_a_better"] = float(va > vb) if higher_is_better else float(va < vb)
        return out
    idx = _indices(n, b, seed)
    da = np.array([stat_fn(sa[i].sum(axis=0)) for i in idx])
    db = np.array([stat_fn(sb[i].sum(axis=0)) for i in idx])
    delta = da - db
    lo, hi = np.percentile(delta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    better = np.mean(delta > 0) if higher_is_better else np.mean(delta < 0)
    out.update(
        ci95=[float(lo), float(hi)],
        prob_a_better=float(better),
        p_two_sided=min(1.0, 2 * min(float(np.mean(delta <= 0)), float(np.mean(delta >= 0)))),
    )
    return out


def bootstrap_stat_ci(stat_fn, stats, b: int = DEFAULT_B, seed: int = DEFAULT_SEED, alpha: float = 0.05):
    """``(value, lo, hi)`` of a corpus statistic of summable per-item statistics [items, k]."""
    s = np.asarray(stats, dtype=np.float64)
    if s.ndim != 2 or s.shape[0] == 0:
        return None, None, None
    v = float(stat_fn(s.sum(axis=0)))
    if s.shape[0] == 1 or b <= 0:
        return v, v, v
    idx = _indices(s.shape[0], b, seed)
    vals = np.array([stat_fn(s[i].sum(axis=0)) for i in idx])
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return v, float(lo), float(hi)
