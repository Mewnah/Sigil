# SPDX-License-Identifier: AGPL-3.0-or-later
"""Per-language text normalization for STT scoring (P7 speech-bench). Frozen as ``sigil-norm/1``.

Every hypothesis and every reference goes through the same function before alignment, so the
choice only has to be consistent and documented, not "right". The rules:

| language | metric | normalization |
|---|---|---|
| en | WER | Whisper's ``EnglishTextNormalizer`` (vendored in ``whisper_english.py``): lower-case, bracketed / parenthesized spans and ``hmm / mm / mhm / mmm / uh / um`` removed, contractions expanded, spelled numbers to digits (``twenty five`` -> ``25``, ``five percent`` -> ``5%``), punctuation and diacritics removed, British -> American spelling |
| ru | WER | Whisper's ``BasicTextNormalizer`` (NFKC, lower-case, brackets removed, every punctuation / symbol / mark to a space, diacritics kept), then ``ё`` -> ``е`` |
| other space-delimited (es, de, fr, pt, it, ...) | WER | Whisper's ``BasicTextNormalizer`` |
| ja, zh, yue | CER | NFKC (full-width ASCII and digits to half-width, half-width katakana to full-width), lower-case, bracketed spans removed, then only letters (Unicode L*, which keeps ``ー`` and ``々``) and digits (N*) are kept: punctuation, symbols and all spaces go; zh / yue: Traditional -> Simplified with OpenCC when ``opencc`` is importable (recorded in ``describe()``) |
| ko | CER | as ja (syllables; spaces removed) |

Common to all: Sigil / SenseVoice / Whisper special tags (``<|ja|>``, ``<|HAPPY|>``) and
``[music]``-style events are removed first. CER units are extended grapheme clusters (``regex``'s
``\\X``; code points without it). Known limitation, shared by every engine (so paired comparisons
stay fair): digits vs number words outside English (``1990`` vs ``mil novecientos noventa``,
``3`` vs ``三``) count as errors.

``describe()`` returns the version stamp the scorer writes into every scores file.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata
from functools import lru_cache

from .whisper_english import BasicTextNormalizer, EnglishTextNormalizer

try:
    import regex as _regex  # type: ignore
except ImportError:  # pragma: no cover
    _regex = None

NORMALIZER_VERSION = "sigil-norm/1"

# Languages scored by character error rate (no word spaces, or CER by convention: Korean).
CER_LANGUAGES = frozenset({"ja", "zh", "yue", "ko", "th", "lo", "my", "km", "bo"})
# Chinese varieties that get the optional Traditional -> Simplified conversion.
CHINESE = frozenset({"zh", "yue"})

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_TAG = re.compile(r"<\|[^|>]*\|>")
_BRACKETS = re.compile(r"[<\[][^>\]]*[>\]]|\(([^)]+?)\)|（[^）]*）|【[^】]*】")


def base_language(lang: str | None) -> str:
    """``ja-JP`` / ``zh-Hans`` / ``pt_BR`` -> ``ja`` / ``zh`` / ``pt`` (lower-case base subtag)."""
    if not lang:
        return ""
    return re.split(r"[-_]", lang.strip().lower(), maxsplit=1)[0]


def metric_for(lang: str | None) -> str:
    """``"cer"`` for :data:`CER_LANGUAGES`, else ``"wer"``."""
    return "cer" if base_language(lang) in CER_LANGUAGES else "wer"


@lru_cache(maxsize=1)
def _spelling() -> tuple[dict, str, str]:
    """(mapping, source name, sha256): Whisper's ``english.json`` when present, else Sigil's table."""
    for name in ("english.json", "english_spelling_sigil.json"):
        path = os.path.join(_DATA, name)
        if os.path.isfile(path):
            with open(path, "rb") as f:
                raw = f.read()
            mapping = json.loads(raw.decode("utf-8"))
            mapping = {k: v for k, v in mapping.items() if not k.startswith("$")}
            return mapping, name, hashlib.sha256(raw).hexdigest()
    return {}, "none", ""


@lru_cache(maxsize=1)
def _english() -> EnglishTextNormalizer:
    return EnglishTextNormalizer(_spelling()[0])


_BASIC = BasicTextNormalizer(remove_diacritics=False)


@lru_cache(maxsize=1)
def _opencc():
    try:
        import opencc  # type: ignore  # noqa: PLC0415

        return opencc.OpenCC("t2s")
    except Exception:  # noqa: BLE001 - optional dependency (any import / config failure)
        return None


def _strip_tags(text: str) -> str:
    return _TAG.sub(" ", text)


def _keep_letters_digits(text: str) -> str:
    return "".join(c for c in text if unicodedata.category(c)[0] in "LN")


def normalize(text: str | None, lang: str | None) -> str:
    """The normalized form of ``text`` in ``lang`` (see the module table)."""
    if not text:
        return ""
    base = base_language(lang)
    s = _strip_tags(text)
    if base in CER_LANGUAGES:
        s = unicodedata.normalize("NFKC", s).lower()
        s = _BRACKETS.sub("", s)
        if base in CHINESE:
            cc = _opencc()
            if cc is not None:
                s = cc.convert(s)
        return _keep_letters_digits(s)
    if base == "en":
        return _english()(s).strip()
    s = _BASIC(s).strip()
    if base == "ru":
        s = s.replace("ё", "е")
    return s


def graphemes(text: str) -> list[str]:
    """Extended grapheme clusters of ``text`` (code points when ``regex`` is missing), spaces excluded."""
    if _regex is not None:
        return [g for g in _regex.findall(r"\X", text) if not g.isspace()]
    return [c for c in text if not c.isspace()]


def tokens(text: str | None, lang: str | None, unit: str | None = None) -> list[str]:
    """Scoring units of ``text``: words (``unit="word"``) or characters (``"char"``); by default the
    language's metric unit. The text is normalized first."""
    norm = normalize(text, lang)
    unit = unit or ("char" if metric_for(lang) == "cer" else "word")
    if unit == "word":
        return norm.split()
    if unit == "char":
        return graphemes(norm)
    raise ValueError(f"unknown unit {unit!r}")


def describe() -> dict:
    """The version stamp of this normalizer (written into scores files)."""
    mapping, source, sha = _spelling()
    return {
        "version": NORMALIZER_VERSION,
        "english": "whisper EnglishTextNormalizer (transformers 4.39.3 copy)",
        "english_spelling": {"source": source, "entries": len(mapping), "sha256": sha},
        "basic": "whisper BasicTextNormalizer (diacritics kept); ru: ё -> е",
        "cer_languages": sorted(CER_LANGUAGES),
        "cer_units": "grapheme clusters" if _regex is not None else "code points",
        "zh_traditional_to_simplified": "opencc t2s" if _opencc() is not None else "none (opencc not installed)",
    }
