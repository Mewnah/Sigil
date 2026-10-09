# SPDX-License-Identifier: AGPL-3.0-or-later
"""The corpus conditions: turn one clean utterance into its condition variant, deterministically.

``synthesize(kind, params, snr_db, speech, rng, ctx)`` returns ``(signal, mix)``: the float64
signal at 16 kHz (same length as ``speech``) and the provenance written into the manifest item's
``mix`` (components with the exact bank files, offsets and gains; the processing chain). ``ctx``
is a :class:`Banks` (music segments, effect events, babble pools); see ``dsp.py`` for the level and
room definitions.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np

from . import dsp

SR = dsp.SR


@dataclass
class Talker:
    """A babble pool utterance."""

    lang: str
    file: str  # FLEURS file name
    sentence_id: int
    audio: np.ndarray  # trimmed, float32


@dataclass
class Banks:
    music: list[dict] = field(default_factory=list)  # {file (abs), rel, seconds, ...provenance}
    sfx: list[dict] = field(default_factory=list)  # {rel, audio (float32), category, ...}
    pools: dict[str, list[Talker]] = field(default_factory=dict)  # lang -> talkers
    languages: list[str] = field(default_factory=list)

    def music_segment(self, idx: int, offset: int, n: int) -> np.ndarray:
        import soundfile as sf  # noqa: PLC0415

        x, _ = sf.read(self.music[idx]["file"], start=offset, frames=n, dtype="float64", always_2d=True)
        return dsp.fit_length(x.mean(axis=1), n)


def _music(rng, n, banks: Banks):
    if not banks.music:
        raise RuntimeError("the music bank is empty")
    idx = int(rng.integers(len(banks.music)))
    seg = banks.music[idx]
    total = int(seg["seconds"] * SR)
    offset = int(rng.integers(0, max(1, total - n)))
    x = banks.music_segment(idx, offset, n)
    return x, [{"kind": "music", "ref": seg["rel"], "offset_s": round(offset / SR, 4)}]


def _sfx(rng, n, banks: Banks, events_per_10s: float, jitter_db: float):
    if not banks.sfx:
        raise RuntimeError("the effects bank is empty")
    y = np.zeros(n, dtype=np.float64)
    events = []
    interval = 10.0 / max(events_per_10s, 0.1)
    t = float(rng.uniform(0.0, min(1.0, n / SR)))
    while t < n / SR:
        ev = banks.sfx[int(rng.integers(len(banks.sfx)))]
        g_db = float(rng.uniform(-jitter_db, jitter_db))
        start = int(t * SR)
        a = ev["audio"].astype(np.float64) * 10 ** (g_db / 20.0)
        end = min(n, start + len(a))
        y[start:end] += a[: end - start]
        events.append({"kind": "sfx", "ref": ev["rel"], "category": ev["category"], "at_s": round(start / SR, 4), "gain_db": round(g_db, 3)})
        t += max(len(a) / SR + 0.1, float(rng.exponential(interval)))
    return y, events


def _babble(rng, n, banks: Banks, lang: str, talkers: int, language: str, exclude_sentence: int | None):
    if language == "same":
        langs = [lang] * talkers
    else:
        others = [x for x in banks.languages if x != lang and banks.pools.get(x)]
        if not others:
            raise RuntimeError("no other-language babble pool")
        pick = rng.permutation(len(others))
        langs = [others[int(pick[i % len(others)])] for i in range(talkers)]
    y = np.zeros(n, dtype=np.float64)
    comps = []
    used: set[str] = set()
    for tl in langs:
        pool = [t for t in banks.pools.get(tl, []) if not (tl == lang and t.sentence_id == exclude_sentence)]
        if not pool:
            raise RuntimeError(f"empty babble pool for {tl}")
        pieces, files, total = [], [], 0
        offset = None
        while total < n + SR:
            choices = [t for t in pool if t.file not in used] or pool
            talker = choices[int(rng.integers(len(choices)))]
            used.add(talker.file)
            a = talker.audio.astype(np.float64)
            pieces.append(a)
            pieces.append(np.zeros(int(0.15 * SR)))
            files.append(talker.file)
            total += len(a) + int(0.15 * SR)
        track = np.concatenate(pieces)
        offset = int(rng.integers(0, max(1, min(len(track) - n, SR))))
        seg = track[offset: offset + n]
        p = dsp.active_power(seg)
        if p > 0:
            seg = seg / math.sqrt(p)
        y += dsp.fit_length(seg, n)
        comps.append({"kind": "talker", "lang": tl, "refs": files, "offset_s": round(offset / SR, 4)})
    return y, comps


def _mic(x, params, rng):
    chain = []
    y = x
    if params.get("highpass_hz"):
        y = dsp.butter_filter(y, "highpass", params["highpass_hz"], order=4)
        chain.append({"op": "highpass", "hz": params["highpass_hz"], "order": 4})
    if params.get("lowpass_hz"):
        y = dsp.butter_filter(y, "lowpass", params["lowpass_hz"], order=8)
        chain.append({"op": "lowpass", "hz": params["lowpass_hz"], "order": 8})
    if params.get("hiss_dbfs") is not None:
        y = y + dsp.hiss(rng, len(y), params["hiss_dbfs"])
        chain.append({"op": "hiss", "rms_dbfs": params["hiss_dbfs"]})
    if params.get("mu_law_bits"):
        y = dsp.mu_law(y, int(params["mu_law_bits"]))
        chain.append({"op": "mu_law", "bits": int(params["mu_law_bits"])})
    return y, chain


def _scaled(speech_power, noise, snr_db):
    p = dsp.active_power(noise)
    g = dsp.gain_for_snr(speech_power, p, snr_db)
    return noise * g, dsp.power_to_db(p * g * g) if p > 0 else None


def synthesize(kind: str, params: dict, snr_db: float | None, speech: np.ndarray, rng: np.random.Generator, banks: Banks,
               lang: str, sentence_id: int | None) -> tuple[np.ndarray, dict]:
    s = np.asarray(speech, dtype=np.float64)
    n = len(s)
    ps = dsp.active_power(s)
    mix = {"snr_db": snr_db, "speech_active_dbfs": round(dsp.power_to_db(ps), 3), "components": [], "chain": []}
    if kind == "clean":
        y = s
    elif kind == "music":
        m, comps = _music(rng, n, banks)
        m, lvl = _scaled(ps, m, snr_db)
        y = s + m
        mix.update(components=comps, interferer_active_dbfs=lvl)
    elif kind == "sfx":
        e, comps = _sfx(rng, n, banks, params.get("events_per_10s", 4), params.get("event_gain_jitter_db", 4))
        e, lvl = _scaled(ps, e, snr_db)
        y = s + e
        mix.update(components=comps, interferer_active_dbfs=lvl)
    elif kind == "babble":
        b, comps = _babble(rng, n, banks, lang, int(params.get("talkers", 3)), params.get("language", "same"), sentence_id)
        b, lvl = _scaled(ps, b, snr_db)
        y = s + b
        mix.update(components=comps, interferer_active_dbfs=lvl)
    elif kind == "room":
        h = dsp.synthetic_rir(rng, params["rt60_s"], params["drr_db"])
        y = dsp.convolve_same(s, h)
        mix["chain"].append({"op": "room", "rt60_s": params["rt60_s"], "drr_db": params["drr_db"], "rir_samples": len(h)})
    elif kind == "mic":
        y, chain = _mic(s, params, rng)
        mix["chain"].extend(chain)
    elif kind == "composite":
        sr_p = params["speech_room"]
        h = dsp.synthetic_rir(rng, sr_p["rt60_s"], sr_p["drr_db"])
        sw = dsp.convolve_same(s, h)
        mix["chain"].append({"op": "room", "target": "speech", "rt60_s": sr_p["rt60_s"], "drr_db": sr_p["drr_db"]})
        pw = dsp.active_power(sw)
        y = sw.copy()
        if params.get("music"):
            mp = params["music"]
            m, comps = _music(rng, n, banks)
            if mp.get("room"):
                hm = dsp.synthetic_rir(rng, mp["room"]["rt60_s"], mp["room"]["drr_db"])
                m = dsp.convolve_same(m, hm)
            m, lvl = _scaled(pw, m, mp["snr_db"])
            y += m
            mix["components"] += [dict(c, snr_db=mp["snr_db"], room=mp.get("room"), active_dbfs=lvl) for c in comps]
        if params.get("sfx"):
            ep = params["sfx"]
            e, comps = _sfx(rng, n, banks, ep.get("events_per_10s", 3), ep.get("event_gain_jitter_db", 4))
            e, lvl = _scaled(pw, e, ep["snr_db"])
            y += e
            mix["components"] += [dict(c, snr_db=ep["snr_db"]) for c in comps]
        if params.get("babble"):
            bp = params["babble"]
            b, comps = _babble(rng, n, banks, lang, int(bp.get("talkers", 2)), bp.get("language", "same"), sentence_id)
            b, lvl = _scaled(pw, b, bp["snr_db"])
            y += b
            mix["components"] += [dict(c, snr_db=bp["snr_db"], active_dbfs=lvl) for c in comps]
        if params.get("mic"):
            y, chain = _mic(y, params["mic"], rng)
            mix["chain"].extend(chain)
    else:
        raise ValueError(f"unknown condition kind {kind!r}")
    return y, mix


def bank_rel(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace("\\", "/")
