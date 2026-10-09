# SPDX-License-Identifier: AGPL-3.0-or-later
"""Real VRChat clips: the drop-in folder that joins the same manifest as the synthetic items.

Layout (inside the corpus folder; never committed, never uploaded)::

    real/
      <session>/                    one recording session, e.g. 2026-10-05-alice-world-music
        session.json                optional defaults for every clip of the session
        <clip>.wav | .flac | .ogg | .mp3   any rate / channels (converted to 16 kHz mono PCM 16-bit)
        <clip>.txt                  the verbatim transcript, UTF-8 (required)
        <clip>.json                 optional per-clip overrides

``session.json`` / ``<clip>.json`` fields (clip wins)::

    {
      "language": "ja",                       required (here or in session.json)
      "consent": {"given": true, "date": "2026-10-05", "scope": "local benchmarking only"},   required
      "condition": "real_world_music",        default "real"; becomes a manifest condition
      "speaker": "alice",                     an alias, never a real name if the speaker prefers
      "gender": null, "mic": "headset", "bleed": "VRChat world music through speakers",
      "tags": ["casual", "fast"], "notes": "..."
    }

A clip without a transcript, without a language or without ``consent.given == true`` is skipped
with a message. Clips are the raw microphone before any voice changer (what Sigil's STT hears).
Item ids are ``real/<session>/<clip>``; ``rank`` orders the clips of a language by id.
"""

from __future__ import annotations

import json
import os

import numpy as np

from . import dsp

AUDIO_EXT = (".wav", ".flac", ".ogg", ".mp3")
README = """Real VRChat clips for the P7 STT benchmark (see scripts/speech-bench/corpus/real.py).

real/<session>/<clip>.wav  + <clip>.txt (verbatim transcript, UTF-8)
real/<session>/session.json  {"language": "ja", "consent": {"given": true, "date": "YYYY-MM-DD",
                               "scope": "local benchmarking only"}, "condition": "real",
                               "speaker": "alias", "mic": "...", "bleed": "...", "tags": []}
Optional <clip>.json overrides session.json per clip. Record the raw microphone (before any voice
changer), with the world audio you normally have. Then run:
  python scripts/speech-bench/corpus/build_corpus.py --real-only
These files stay on this PC: never commit or upload them.
"""


def _read_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _decode(path: str) -> tuple[np.ndarray, int, int]:
    import soundfile as sf  # noqa: PLC0415

    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), int(sr), int(x.shape[1])


def ingest(real_dir: str, out: str, write_pcm16, log) -> tuple[list[dict], list[str]]:
    """Convert every valid clip under ``real_dir`` into ``audio/real/...`` and return its items."""
    items, notes = [], []
    if not os.path.isdir(real_dir):
        return items, notes
    for session in sorted(os.listdir(real_dir)):
        sdir = os.path.join(real_dir, session)
        if not os.path.isdir(sdir):
            continue
        base = _read_json(os.path.join(sdir, "session.json"))
        for name in sorted(os.listdir(sdir)):
            stem, ext = os.path.splitext(name)
            if ext.lower() not in AUDIO_EXT:
                continue
            meta = {**base, **_read_json(os.path.join(sdir, stem + ".json"))}
            txt = os.path.join(sdir, stem + ".txt")
            where = f"real/{session}/{name}"
            if not os.path.isfile(txt):
                notes.append(f"{where}: no transcript ({stem}.txt), skipped")
                continue
            with open(txt, encoding="utf-8-sig") as f:
                reference = " ".join(f.read().split())
            consent = meta.get("consent") or {}
            if consent.get("given") is not True:
                notes.append(f"{where}: no consent (consent.given true in session.json or {stem}.json), skipped")
                continue
            lang = (meta.get("language") or "").strip().lower()
            if not lang or not reference:
                notes.append(f"{where}: no language or empty transcript, skipped")
                continue
            x, sr, ch = _decode(os.path.join(sdir, name))
            y = dsp.resample(x, sr)
            y, post = dsp.headroom(y)
            rel = f"audio/real/{session}/{stem}.wav"
            digest = write_pcm16(os.path.join(out, rel), y)
            items.append({
                "id": f"real/{session}/{stem}",
                "utterance": f"real/{session}/{stem}",
                "kind": "real",
                "language": lang,
                "condition": meta.get("condition") or "real",
                "audio": rel,
                "seconds": round(len(y) / dsp.SR, 4),
                "sha256": digest,
                "reference": reference,
                "rank": 0,
                "speaker": {"id": meta.get("speaker"), "gender": meta.get("gender")},
                "source": {"kind": "real", "session": session, "file": name, "original_rate": sr, "original_channels": ch,
                           "mic": meta.get("mic"), "bleed": meta.get("bleed")},
                "mix": {"seed": 0, "snr_db": None, "post_gain_db": round(post, 3), "components": [], "chain": []} if post else None,
                "consent": {"given": True, "date": consent.get("date"), "scope": consent.get("scope", "local benchmarking only")},
                "tags": list(meta.get("tags") or []),
                "notes": meta.get("notes", ""),
            })
    by_lang: dict[str, list[dict]] = {}
    for it in items:
        by_lang.setdefault(it["language"], []).append(it)
    for lst in by_lang.values():
        for rank, it in enumerate(sorted(lst, key=lambda i: i["id"])):
            it["rank"] = rank
    for n in notes:
        log(n)
    if items:
        log(f"real clips: {len(items)} in {len(by_lang)} language(s)")
    return items, notes


def write_readme(real_dir: str) -> None:
    os.makedirs(real_dir, exist_ok=True)
    path = os.path.join(real_dir, "README.txt")
    if not os.path.isfile(path):
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            f.write(README)
