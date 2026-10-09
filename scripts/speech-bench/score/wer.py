# SPDX-License-Identifier: AGPL-3.0-or-later
"""Edit-distance alignment for WER / CER (P7 speech-bench).

``align_counts(ref, hyp)`` returns substitutions, deletions and insertions of a minimum-edit
(Levenshtein) alignment of two token lists; the error rate of a corpus is
``sum(S + D + I) / sum(N_ref)`` over its items (never the mean of per-item rates). Among
alignments of equal cost the breakdown prefers substitutions, then deletions (the total, and so
the rate, does not depend on the tie-break).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .normalize import metric_for, tokens


@dataclass
class EditCounts:
    """Edit operations of one or more aligned items."""

    sub: int = 0
    dele: int = 0
    ins: int = 0
    ref_len: int = 0
    hyp_len: int = 0
    items: int = field(default=0)

    @property
    def errors(self) -> int:
        return self.sub + self.dele + self.ins

    @property
    def hits(self) -> int:
        return self.ref_len - self.sub - self.dele

    def rate(self) -> float | None:
        """Errors per reference unit; ``None`` when there is no reference unit."""
        if self.ref_len == 0:
            return None
        return self.errors / self.ref_len

    def add(self, other: "EditCounts") -> "EditCounts":
        self.sub += other.sub
        self.dele += other.dele
        self.ins += other.ins
        self.ref_len += other.ref_len
        self.hyp_len += other.hyp_len
        self.items += other.items
        return self

    def as_dict(self) -> dict:
        return {
            "sub": self.sub,
            "del": self.dele,
            "ins": self.ins,
            "ref_units": self.ref_len,
            "hyp_units": self.hyp_len,
            "items": self.items,
            "errors": self.errors,
            "rate": self.rate(),
        }


def align_counts(ref: list[str], hyp: list[str]) -> EditCounts:
    """Minimum-edit alignment counts of token lists ``ref`` and ``hyp``."""
    n, m = len(ref), len(hyp)
    if n == 0:
        return EditCounts(ins=m, ref_len=0, hyp_len=m, items=1)
    if m == 0:
        return EditCounts(dele=n, ref_len=n, hyp_len=0, items=1)
    # Row j of the DP holds (cost, sub, del, ins) of aligning ref[:i] with hyp[:j].
    prev = [(j, 0, 0, j) for j in range(m + 1)]
    for i in range(1, n + 1):
        r = ref[i - 1]
        cur = [(i, 0, i, 0)]
        for j in range(1, m + 1):
            d = prev[j - 1]
            if hyp[j - 1] == r:
                best = d
            else:
                best = (d[0] + 1, d[1] + 1, d[2], d[3])
            up = prev[j]  # delete ref[i-1]
            if up[0] + 1 < best[0]:
                best = (up[0] + 1, up[1], up[2] + 1, up[3])
            left = cur[j - 1]  # insert hyp[j-1]
            if left[0] + 1 < best[0]:
                best = (left[0] + 1, left[1], left[2], left[3] + 1)
            cur.append(best)
        prev = cur
    _, s, d, ins = prev[m]
    return EditCounts(sub=s, dele=d, ins=ins, ref_len=n, hyp_len=m, items=1)


def score_pair(reference: str | None, hypothesis: str | None, lang: str | None, unit: str | None = None) -> EditCounts:
    """Counts of one item after normalization (``unit``: ``word`` / ``char``; default: the
    language's metric unit)."""
    return align_counts(tokens(reference, lang, unit), tokens(hypothesis, lang, unit))


def unit_for(lang: str | None) -> str:
    """``word`` for WER languages, ``char`` for CER languages."""
    return "char" if metric_for(lang) == "cer" else "word"
