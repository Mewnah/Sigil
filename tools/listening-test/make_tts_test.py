#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build a blind TTS listening test (P7: Qwen3-TTS vs Kokoro vs Magpie, or any engines) from
speech-bench ``tts_listening`` renderings.

    python tools/listening-test/make_tts_test.py --report <out>/report.json [--report ...]
        --prompts corpus/tts/prompts.jsonl --engines kokoro,qwen3-tts-0.6b,magpie
        [--languages en,es] [--per-language 8] [--kinds casual,read,numbers]
        [--mode mushra|pairwise] [--rate 24000] [--loudness -23] [--seed 7] --out tts-1

- ``mushra`` (default): every trial plays one prompt by every chosen engine that rendered it (two
  or more), in a random order, rated 0-100 (MUSHRA-lite: no hidden reference or anchor exists for
  synthetic speech); ``pairwise``: every pair of engines per prompt, A / B / no preference.
- Prompts: per language, those every chosen engine supporting the language rendered without error,
  ``--kinds`` first, ``--per-language`` of them by a seeded order.
- Audio: every rendering is decoded, mixed to mono, resampled to one rate (``--rate``, default the
  highest among them), brought to the same active level (``--loudness`` dBFS, the corpus's
  active-level measure; a louder voice otherwise wins), limited to a -1 dBFS peak by lowering that
  rendering only when needed (recorded in key.json), and written as 16-bit PCM WAV under random
  names. Needs numpy, soundfile and scipy (the P7 venv has them).

Give ``<out>/test/`` to the listeners; keep ``<out>/key.json``; score with ``aggregate.py``.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import secrets
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from listening_common import new_test_id, ranked, rng_for, write_test  # noqa: E402

DEFAULT_INSTRUCTIONS = ("Each sentence is read by several synthetic voices. Rate how natural, clear and correctly pronounced "
                        "each one sounds, as if a friend's stream used it.")


def active_dbfs(x, sr: int) -> float:
    """Active level: mean power of the 20 ms frames within 30 dB of the 95th-percentile frame."""
    import numpy as np  # noqa: PLC0415

    n = max(1, int(0.02 * sr))
    frames = len(x) // n
    if frames == 0:
        p = np.array([float(np.mean(x * x))]) if len(x) else np.zeros(1)
    else:
        f = x[: frames * n].reshape(frames, n)
        p = (f * f).mean(axis=1)
    p = p[p > 1e-12]
    if p.size == 0:
        return -120.0
    ref = float(np.percentile(p, 95))
    act = p[p >= ref * 10 ** (-3.0)]
    return 10 * math.log10(max(float(act.mean()), 1e-12))


def load_renderings(reports: list[str]) -> dict[str, dict[str, dict]]:
    """label -> prompt id -> {path, lang, ...} of successful renderings."""
    out: dict[str, dict[str, dict]] = {}
    multi = len(reports) > 1
    for path in reports:
        with open(path, encoding="utf-8") as f:
            report = json.load(f)
        base = os.path.dirname(os.path.abspath(path))
        engines = {e["id"]: e for e in report.get("engines", [])}
        rname = report.get("name") or os.path.basename(base)
        for run in report.get("runs", []):
            for res in run.get("results", []):
                if (engines.get(res["engine"]) or {}).get("task") != "tts":
                    continue
                label = f"{rname}:{res['engine']}" if multi else res["engine"]
                for it in res.get("items", []):
                    if it.get("error") is None and it.get("audio"):
                        p = it["audio"] if os.path.isabs(it["audio"]) else os.path.join(base, it["audio"])
                        if os.path.isfile(p):
                            out.setdefault(label, {})[it["item"]] = {"path": p, "lang": it.get("language")}
    return out


def build(renderings, prompts, engines, languages, per_language, kinds, mode, seed):
    rng, seed = rng_for(seed)
    trials, key_trials, selection = [], [], {}
    langs = languages or sorted({p["lang"] for p in prompts.values()})
    for lang in langs:
        cands = []
        for pid, p in prompts.items():
            if p["lang"] != lang or (kinds and p["kind"] not in kinds):
                continue
            have = [e for e in engines if pid in renderings.get(e, {})]
            supporting = [e for e in engines if any(prompts.get(q, {}).get("lang") == lang for q in renderings.get(e, {}))]
            if len(have) >= 2 and set(have) == set(supporting):
                cands.append((pid, have))
        order = {k: i for i, k in enumerate(kinds or [])}
        cands = ranked(cands, seed, "tts", lang)
        cands.sort(key=lambda c: order.get(prompts[c[0]]["kind"], len(order)))
        chosen = cands[:per_language]
        selection[lang] = {"eligible": len(cands), "chosen": len(chosen), "engines": sorted({e for _, h in chosen for e in h})}
        for pid, have in chosen:
            if mode == "mushra":
                systems = list(have)
                rng.shuffle(systems)
                trials.append({"lang": lang, "text": prompts[pid]["text"], "_item": pid, "_systems": systems})
            else:
                for a, b in itertools.combinations(have, 2):
                    if rng.random() < 0.5:
                        a, b = b, a
                    trials.append({"lang": lang, "text": prompts[pid]["text"], "_item": pid, "_systems": [a, b]})
    rng.shuffle(trials)
    return trials, selection, seed


def render_audio(trials, renderings, audio_dir: str, rate: int | None, loudness: float):
    """Write every needed rendering once (blinded); returns {(label, prompt): (file name, gain info)}."""
    import numpy as np  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415
    from scipy.signal import resample_poly  # noqa: PLC0415

    needed = sorted({(s, t["_item"]) for t in trials for s in t["_systems"]})
    decoded = {}
    for label, pid in needed:
        x, sr = sf.read(renderings[label][pid]["path"], dtype="float64", always_2d=True)
        decoded[(label, pid)] = (x.mean(axis=1), int(sr))
    target_rate = rate or max(sr for _, sr in decoded.values())
    os.makedirs(audio_dir, exist_ok=True)
    out, used = {}, set()
    for k, (x, sr) in decoded.items():
        if sr != target_rate:
            g = math.gcd(sr, target_rate)
            x = resample_poly(x, target_rate // g, sr // g)
        level = active_dbfs(x, target_rate)
        gain_db = loudness - level
        y = x * 10 ** (gain_db / 20)
        peak = float(np.max(np.abs(y))) if len(y) else 0.0
        limit_db = 0.0
        if peak > 10 ** (-1 / 20):
            limit_db = 20 * math.log10(10 ** (-1 / 20) / peak)
            y = y * 10 ** (limit_db / 20)
        while True:
            name = secrets.token_hex(8) + ".wav"
            if name not in used:
                used.add(name)
                break
        sf.write(os.path.join(audio_dir, name), y, target_rate, subtype="PCM_16")
        out[k] = (name, {"level_dbfs": round(level, 2), "gain_db": round(gain_db, 2), "peak_limit_db": round(limit_db, 2)})
    return out, target_rate


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="append", required=True)
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--engines", required=True, help="comma-separated labels (two or more)")
    ap.add_argument("--languages", help="comma-separated (default: every prompt language)")
    ap.add_argument("--per-language", type=int, default=8)
    ap.add_argument("--kinds", default="casual,read,numbers,long", help="prompt kinds, in preference order")
    ap.add_argument("--mode", choices=("mushra", "pairwise"), default="mushra")
    ap.add_argument("--rate", type=int)
    ap.add_argument("--loudness", type=float, default=-23.0, help="active level of every rendering (dBFS)")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--title", default="Voice listening test")
    ap.add_argument("--instructions", default=DEFAULT_INSTRUCTIONS)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    if len(engines) < 2:
        ap.error("--engines needs two or more labels")
    with open(args.prompts, encoding="utf-8") as f:
        prompts = {p["id"]: p for p in (json.loads(l) for l in f if l.strip())}
    renderings = load_renderings(args.report)
    missing = [e for e in engines if e not in renderings]
    if missing:
        ap.error(f"no successful renderings for {missing}; the reports have {sorted(renderings)}")
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    languages = [x.strip() for x in args.languages.split(",")] if args.languages else None
    trials, selection, seed = build(renderings, prompts, engines, languages, args.per_language, kinds, args.mode, args.seed)
    if not trials:
        print("no trial: no prompt was rendered by two or more of these engines", file=sys.stderr)
        return 1
    files, rate = render_audio(trials, renderings, os.path.join(args.out, "test", "audio"), args.rate, args.loudness)
    page_trials, key_trials = [], []
    for n, t in enumerate(trials, 1):
        tid = "t%03d" % n
        if args.mode == "mushra":
            stimuli, smap, fmap = [], {}, {}
            for k, label in enumerate(t["_systems"], 1):
                name, _ = files[(label, t["_item"])]
                stimuli.append({"id": f"s{k}", "audio": "audio/" + name})
                smap[f"s{k}"] = label
                fmap[name] = renderings[label][t["_item"]]["path"]
            page_trials.append({"id": tid, "lang": t["lang"], "text": t["text"], "stimuli": stimuli})
            key_trials.append({"id": tid, "item": t["_item"], "lang": t["lang"], "stimuli": smap, "files": fmap})
        else:
            a, b = t["_systems"]
            na, nb = files[(a, t["_item"])][0], files[(b, t["_item"])][0]
            page_trials.append({"id": tid, "lang": t["lang"], "text": t["text"], "a": "audio/" + na, "b": "audio/" + nb})
            key_trials.append({"id": tid, "item": t["_item"], "lang": t["lang"], "A": a, "B": b,
                               "files": {na: renderings[a][t["_item"]]["path"], nb: renderings[b][t["_item"]]["path"]}})
    test_id = new_test_id()
    manifest = {"schema": "sigil-listening-tts/1", "test_id": test_id, "title": args.title, "instructions": args.instructions,
                "mode": args.mode, "scale": {"min": 0, "max": 100, "labels": ["Bad", "Poor", "Fair", "Good", "Excellent"]},
                "require_play": True, "trials": page_trials}
    key = {"schema": "sigil-listening-key/1", "test_id": test_id, "kind": "tts", "mode": args.mode, "title": args.title, "seed": seed,
           "systems": engines, "sources": [os.path.basename(p) for p in args.report] + [os.path.basename(args.prompts)], "selection": selection,
           "audio": {"rate": rate, "channels": 1, "subtype": "PCM_16", "loudness_dbfs": args.loudness, "peak_limit_dbfs": -1.0,
                     "per_file": {name: info for (name, info) in files.values()}},
           "trials": key_trials}
    test_dir = write_test(args.out, "tts-listen.html", manifest, key)
    print(f"{len(page_trials)} trials ({args.mode}, {rate} Hz) -> {test_dir} (give this folder to listeners); key -> {os.path.join(args.out, 'key.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
