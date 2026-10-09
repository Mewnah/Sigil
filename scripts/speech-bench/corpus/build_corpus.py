#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the P7 STT corpus (``sigil-speech-corpus/1``) from the prefetched datasets, outside the repo.

    python scripts/speech-bench/corpus/build_corpus.py [--recipe corpus/recipe.json]
        [--p7-root %LOCALAPPDATA%/sigil-dev-cache/p7] [--out <p7-root>/corpus]
        [--languages en,ja] [--per-language N] [--conditions clean,vrchat_mix]
        [--workers 2] [--priority idle|below_normal|normal] [--resume] [--no-real] [--verify]

Steps (each is skipped when its output already exists, so an interrupted build resumes):
1. select the utterances of every language from the FLEURS test TSV (deterministic, see
   ``sources.select_fleurs``) and extract only those WAVs (plus the babble pool) into ``_cache/``;
2. build the banks: ``_bank/music`` (30 s segments of FMA tracks, decoded from MP3) and
   ``_bank/sfx`` (ESC-50 events resampled to 16 kHz and trimmed), with ``_bank/bank.json``;
3. synthesize every (utterance, condition) into ``audio/<lang>/<condition>/<file>.wav``
   (16 kHz mono PCM 16-bit, written with the standard ``wave`` module so the bytes are
   reproducible), one worker process per language at a time (``--workers``, default 2);
4. add the real drop-in clips of ``real/`` (``real.py``; consent required);
5. write ``corpus.json`` (validated against schemas/speech-corpus.schema.json when ``jsonschema``
   is importable).

``--per-language``, ``--languages`` and ``--conditions`` override the recipe: the effective recipe
(and so ``recipe_sha256``) changes, so a reduced corpus never passes for the full one. The process
lowers its own priority (``--priority``, default ``below_normal``; the workers inherit it) and
limits math libraries to one thread. ``--verify`` re-hashes every file of an existing manifest.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np  # noqa: E402

from corpus import dsp, real, sources  # noqa: E402
from corpus.conditions import Banks, Talker, bank_rel, synthesize  # noqa: E402

BUILDER_VERSION = "3"
MANIFEST_SCHEMA = "sigil-speech-corpus/1"
DEFAULT_P7 = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache", "p7")


# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------

def set_priority(level: str) -> None:
    """Lower this process's priority class (Windows; ``nice`` elsewhere). Children inherit it."""
    if level == "normal":
        return
    if os.name == "nt":
        import ctypes  # noqa: PLC0415

        cls = {"idle": 0x40, "below_normal": 0x4000}[level]
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), cls)
    else:  # pragma: no cover
        os.nice(19 if level == "idle" else 10)


def canonical_sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_pcm16(path: str, y: np.ndarray, sr: int = dsp.SR) -> str:
    """Write mono PCM 16-bit WAV atomically; returns the sha256 of the file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pcm = dsp.to_pcm16(y)
    tmp = path + ".part"
    with wave.open(tmp, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    os.replace(tmp, path)
    return sha256_file(path)


def read_pcm16(path: str) -> np.ndarray:
    with wave.open(path, "rb") as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float64) / 32768.0


def effective_recipe(recipe: dict, languages: list[str] | None, per_language: int | None, conditions: list[str] | None) -> dict:
    r = copy.deepcopy(recipe)
    r.pop("$comment", None)
    if languages:
        known = {x["lang"] for x in r["languages"]}
        missing = set(languages) - known
        if missing:
            raise SystemExit(f"unknown languages {sorted(missing)} (recipe has {sorted(known)})")
        r["languages"] = [x for x in r["languages"] if x["lang"] in languages]
    if per_language:
        r["selection"]["per_language"] = int(per_language)
    if conditions:
        known = {c["id"] for c in r["conditions"]}
        missing = set(conditions) - known
        if missing:
            raise SystemExit(f"unknown conditions {sorted(missing)}")
        r["conditions"] = [c for c in r["conditions"] if c["id"] in conditions]
        r["subsets"] = {k: [c for c in v if c in conditions or c == "real"] for k, v in r["subsets"].items()}
    return r


def content_sha256(items: list[dict]) -> str:
    """The corpus identity by content: sha256 of the canonical JSON of sorted [id, file sha256]."""
    return canonical_sha256(sorted([it["id"], it["sha256"]] for it in items))


def builder_stamp() -> dict:
    pkgs = {}
    for name in ("numpy", "scipy", "soundfile", "pyarrow"):
        try:
            mod = __import__(name)
            pkgs[name] = getattr(mod, "__version__", "?")
        except ImportError:
            pass
    stamp = {"script": "scripts/speech-bench/corpus/build_corpus.py", "version": BUILDER_VERSION, "python": platform.python_version(), "packages": pkgs}
    try:
        import soundfile  # noqa: PLC0415

        stamp["libsndfile"] = soundfile.__libsndfile_version__
    except Exception:  # noqa: BLE001
        pass
    return stamp


# ---------------------------------------------------------------------------------------------
# step 1: selection + extraction
# ---------------------------------------------------------------------------------------------

def select_all(recipe: dict, p7_root: str, out: str, log) -> dict:
    sel = recipe["selection"]
    result = {}
    for entry in recipe["languages"]:
        lang, cfg = entry["lang"], entry["fleurs"]
        d = sources.fleurs_dir(p7_root, cfg)
        rows = sources.read_fleurs_tsv(os.path.join(d, "test.tsv"))
        chosen, pool = sources.select_fleurs(rows, lang, recipe["seed"], sel["per_language"], sel["min_seconds"], sel["max_seconds"],
                                             sel.get("one_recording_per_sentence", True), sel.get("balance_gender", True), sel.get("babble_pool", 24))
        if len(chosen) < sel["per_language"]:
            log(f"warning: {lang}: only {len(chosen)} utterances match the selection")
        cache = os.path.join(out, "_cache", "fleurs", lang)
        names = {r["file_name"] for r in chosen + pool}
        t = time.time()
        paths = sources.extract_fleurs(os.path.join(d, "audio", "test.tar.gz"), names, cache)
        log(f"{lang}: {len(chosen)} utterances + {len(pool)} babble talkers ({time.time() - t:.0f} s)")
        result[lang] = {"config": cfg, "chosen": chosen, "pool": pool, "paths": paths}
    return result


# ---------------------------------------------------------------------------------------------
# step 2: banks
# ---------------------------------------------------------------------------------------------

def build_music_bank(recipe: dict, p7_root: str, out: str, log) -> list[dict]:
    cfg = recipe["banks"]["music"]
    bank_dir = os.path.join(out, "_bank", "music")
    shards = sources.fma_shards(p7_root)
    if not shards:
        raise SystemExit(f"no FMA parquet shard under {p7_root}/bleed/fma-commercial-16khz/data")
    tracks = []
    for shard in shards:
        for t in sources.fma_tracks(shard):
            t["_shard_path"] = shard
            t["_key"] = sources.seeded_key(recipe["seed"], "fma", t["shard"], t["path"])
            tracks.append(t)
    tracks.sort(key=lambda t: t["_key"])
    need = cfg["tracks"]
    per = cfg["segments_per_track"]
    seg_n = int(cfg["segment_seconds"] * dsp.SR)
    accepted: list[dict] = []
    cursor = 0
    while len(accepted) < need * per and cursor < len(tracks):
        batch = tracks[cursor: cursor + max(8, (need - len(accepted) // per) * 2)]
        cursor += len(batch)
        by_group: dict[tuple[str, int], set[int]] = {}
        for t in batch:
            by_group.setdefault((t["_shard_path"], t["row_group"]), set()).add(t["row"])
        decoded = {}
        for (shard, g), rows in sorted(by_group.items()):
            for row, val in sources.fma_decode_group(shard, g, rows).items():
                decoded[(shard, g, row)] = val
        for t in batch:
            if len(accepted) >= need * per:
                break
            x, sr = decoded[(t["_shard_path"], t["row_group"], t["row"])]
            if sr != dsp.SR:
                x = dsp.resample(x, sr)
            margin = 10 * dsp.SR
            if len(x) < seg_n + 2 * margin:
                continue
            offsets = list(range(margin, len(x) - seg_n - margin, dsp.SR))
            offsets.sort(key=lambda o: sources.seeded_key(recipe["seed"], "fma-seg", t["path"], o))
            segs = []
            for o in offsets:
                if any(abs(o - s) < seg_n for s, _ in segs):
                    continue
                seg = x[o: o + seg_n]
                if dsp.active_dbfs(seg) < cfg.get("min_active_dbfs", -38.0) or dsp.silent_fraction(seg) > cfg.get("max_silent_fraction", 0.1):
                    continue
                segs.append((o, seg))
                if len(segs) == per:
                    break
            if len(segs) < per:
                continue
            stem = os.path.splitext(os.path.basename(t["path"]))[0]
            for k, (o, seg) in enumerate(segs):
                path = os.path.join(bank_dir, f"fma-{stem}-{k}.wav")
                if not os.path.isfile(path):
                    write_pcm16(path, seg)
                accepted.append({"file": path, "rel": bank_rel(path, out), "seconds": len(seg) / dsp.SR, "track": t["path"], "shard": t["shard"],
                                 "title": t["title"], "artist": t["artist"], "license_code": t["license"], "url": t["url"],
                                 "offset_s": o / dsp.SR, "active_dbfs": round(dsp.active_dbfs(seg), 2)})
        log(f"music bank: {len(accepted) // per} / {need} tracks")
    if len(accepted) < need * per:
        log(f"warning: music bank has only {len(accepted)} segments")
    return accepted


def build_sfx_bank(recipe: dict, p7_root: str, out: str, log) -> list[dict]:
    cfg = recipe["banks"]["sfx"]
    zip_path = sources.esc50_zip(p7_root)
    if zip_path is None:
        raise SystemExit(f"no ESC-50 zip under {p7_root}/bleed/esc50")
    rows, root = sources.esc50_rows(zip_path)
    bank = []
    max_n = int(cfg.get("max_event_seconds", 3.0) * dsp.SR)
    for cat in cfg["categories"]:
        clips = [r for r in rows if r["category"] == cat]
        if not clips:
            raise SystemExit(f"ESC-50 has no category {cat!r}")
        clips.sort(key=lambda r: sources.seeded_key(recipe["seed"], "esc50", r["filename"]))
        for r in clips[: cfg["clips_per_category"]]:
            path = os.path.join(out, "_bank", "sfx", cat, os.path.splitext(r["filename"])[0] + ".wav")
            if not os.path.isfile(path):
                x, sr = sources.esc50_read(zip_path, root, r["filename"])
                ev = dsp.trim_active(dsp.resample(x, sr))[:max_n]
                if len(ev) < int(0.05 * dsp.SR):
                    continue
                write_pcm16(path, ev)
            bank.append({"file": path, "rel": bank_rel(path, out), "category": cat, "esc50_file": r["filename"], "fold": int(r["fold"]),
                         "freesound_id": r.get("src_file"), "take": r.get("take")})
    log(f"effects bank: {len(bank)} events in {len(cfg['categories'])} categories")
    return bank


# ---------------------------------------------------------------------------------------------
# step 3: synthesis (one language per task)
# ---------------------------------------------------------------------------------------------

def _load_banks(music: list[dict], sfx: list[dict], pools: dict[str, list[dict]], languages: list[str]) -> Banks:
    b = Banks(music=music, languages=languages)
    for ev in sfx:
        b.sfx.append(dict(ev, audio=read_pcm16(ev["file"]).astype(np.float32)))
    for lang, talkers in pools.items():
        b.pools[lang] = []
        for t in talkers:
            x, sr = sources.read_wav_mono(t["path"])
            b.pools[lang].append(Talker(lang=lang, file=t["file_name"], sentence_id=t["sentence_id"], audio=dsp.trim_active(x).astype(np.float32)))
    return b


def synthesize_language(task: dict) -> list[dict]:
    """Every (utterance, condition) item of one language. Progress is appended to
    ``_state/<lang>.jsonl`` (one item per line), which ``--resume`` reads back so a finished item
    whose audio file still exists is not synthesized again."""
    set_priority(task["priority"])
    recipe, lang, out = task["recipe"], task["lang"], task["out"]
    state_path = os.path.join(out, "_state", f"{lang}.jsonl")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    done: dict[str, dict] = {}
    if task["resume"] and os.path.isfile(state_path):
        with open(state_path, encoding="utf-8") as f:
            for line in f:
                try:
                    it = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut by an interrupted build
                if it.get("recipe_sha256") == task["recipe_sha256"] and it.get("builder") == BUILDER_VERSION:
                    done[it["item"]["id"]] = it["item"]
    banks = None
    items = []
    with open(state_path, "a" if task["resume"] else "w", encoding="utf-8", newline=chr(10)) as state:
        for utt in task["chosen"]:
            stem = os.path.splitext(utt["file_name"])[0]
            speech = None
            for cond in recipe["conditions"]:
                item_id = f"{lang}/{stem}/{cond['id']}"
                rel = f"audio/{lang}/{cond['id']}/{stem}.wav"
                path = os.path.join(out, rel)
                prev = done.get(item_id)
                if prev is not None and os.path.isfile(path):
                    items.append(prev)
                    continue
                if banks is None:
                    banks = _load_banks(task["music"], task["sfx"], task["pools"], [x["lang"] for x in recipe["languages"]])
                if speech is None:
                    speech, sr = sources.read_wav_mono(utt["path"])
                    if sr != dsp.SR:
                        speech = dsp.resample(speech, sr)
                    level = recipe.get("speech_level") or {}
                    speech_gain_db = 0.0
                    if level.get("active_dbfs") is not None and dsp.active_power(speech) > 0:
                        jitter = float(level.get("jitter_db", 0.0))
                        offset = np.random.default_rng(sources.seed_int(recipe["seed"], lang, stem, "level")).uniform(-jitter, jitter) if jitter else 0.0
                        speech_gain_db = level["active_dbfs"] + offset - dsp.active_dbfs(speech)
                        speech = speech * 10 ** (speech_gain_db / 20.0)
                seed = sources.seed_int(recipe["seed"], lang, stem, cond["id"])
                rng = np.random.default_rng(seed)
                y, mix = synthesize(cond["kind"], cond.get("params") or {}, cond.get("snr_db"), speech, rng, banks, lang, utt["sentence_id"])
                y, post = dsp.headroom(y)
                mix = {"seed": seed, "speech_gain_db": round(speech_gain_db, 3), **mix, "post_gain_db": round(post, 3)}
                digest = write_pcm16(path, y)
                item = {
                    "id": item_id,
                    "utterance": f"{lang}/{stem}",
                    "kind": "synthetic",
                    "language": lang,
                    "condition": cond["id"],
                    "audio": rel,
                    "seconds": round(len(y) / dsp.SR, 4),
                    "sha256": digest,
                    "reference": utt["raw_transcription"],
                    "rank": utt["rank"],
                    "speaker": {"id": None, "gender": utt.get("gender")},
                    "source": {"kind": "fleurs", "dataset": sources.FLEURS["dataset"], "revision": sources.FLEURS["revision"], "config": task["config"],
                               "split": "test", "file": utt["file_name"], "sentence_id": utt["sentence_id"],
                               "transcription_normalized": utt["transcription"]},
                    "mix": mix,
                }
                state.write(json.dumps({"recipe_sha256": task["recipe_sha256"], "builder": BUILDER_VERSION, "item": item}, ensure_ascii=False) + chr(10))
                state.flush()
                items.append(item)
    return items


# ---------------------------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------------------------

def verify(out: str) -> int:
    with open(os.path.join(out, "corpus.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    bad = 0
    for it in manifest["items"]:
        p = os.path.join(out, it["audio"])
        if not os.path.isfile(p) or sha256_file(p) != it["sha256"]:
            bad += 1
            print(f"mismatch: {it['id']}")
    print(f"{len(manifest['items']) - bad} / {len(manifest['items'])} files match")
    return 1 if bad else 0


def build(args) -> dict:
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f} s] {msg}", flush=True)

    with open(args.recipe, encoding="utf-8") as f:
        recipe = effective_recipe(json.load(f), args.languages, args.per_language, args.conditions)
    recipe_sha = canonical_sha256(recipe)
    out = args.out
    os.makedirs(out, exist_ok=True)
    log(f"recipe {recipe['name']} ({recipe_sha[:12]}): {len(recipe['languages'])} languages x {recipe['selection']['per_language']} x {len(recipe['conditions'])} conditions -> {out}")
    items: list[dict] = []
    src_meta = {}
    if args.real_only:
        existing_path = os.path.join(out, "corpus.json")
        if not os.path.isfile(existing_path):
            raise SystemExit(f"--real-only needs an existing {existing_path}")
        with open(existing_path, encoding="utf-8") as f:
            existing = json.load(f)
        recipe, recipe_sha = existing["recipe"], existing["recipe_sha256"]
        items = [it for it in existing["items"] if it["kind"] != "real"]
        src_meta = {k: v for k, v in existing.get("sources", {}).items() if k != "real"}
    else:
        selection = select_all(recipe, args.p7_root, out, log)
        kinds = {c["kind"] for c in recipe["conditions"]}
        comp = [c.get("params") or {} for c in recipe["conditions"] if c["kind"] == "composite"]
        needs_music = "music" in kinds or any("music" in p for p in comp)
        needs_sfx = "sfx" in kinds or any("sfx" in p for p in comp)
        music = build_music_bank(recipe, args.p7_root, out, log) if needs_music else []
        sfx = build_sfx_bank(recipe, args.p7_root, out, log) if needs_sfx else []
        bank_doc = {"music": [{k: v for k, v in m.items() if k != "file"} for m in music], "sfx": [{k: v for k, v in e.items() if k != "file"} for e in sfx]}
        os.makedirs(os.path.join(out, "_bank"), exist_ok=True)
        with open(os.path.join(out, "_bank", "bank.json"), "w", encoding="utf-8") as f:
            json.dump(bank_doc, f, ensure_ascii=False, indent=1)
        pools = {lang: [dict(t, path=s["paths"][t["file_name"]]) for t in s["pool"]] for lang, s in selection.items()}
        tasks = []
        for lang, s in selection.items():
            chosen = [dict(u, path=s["paths"][u["file_name"]]) for u in s["chosen"]]
            needed_pools = pools if any((c.get("params") or {}).get("language") == "other" for c in recipe["conditions"]) else {lang: pools[lang]}
            tasks.append({"recipe": recipe, "recipe_sha256": recipe_sha, "lang": lang, "config": s["config"], "out": out, "chosen": chosen,
                          "pools": needed_pools, "music": music, "sfx": sfx, "resume": args.resume, "priority": args.priority})
        log(f"synthesizing {sum(len(t['chosen']) for t in tasks) * len(recipe['conditions'])} items with {args.workers} worker(s)")
        if args.workers <= 1:
            for t in tasks:
                items += synthesize_language(t)
                log(f"{t['lang']}: done")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                for t, res in zip(tasks, ex.map(synthesize_language, tasks)):
                    items += res
                    log(f"{t['lang']}: done")
        langs_cfg = {s: v["config"] for s, v in selection.items()}
        src_meta = {
            "fleurs": {**sources.FLEURS, "files": [f for c in langs_cfg.values() for f in (
                {"config": c, "path": os.path.join(sources.fleurs_dir(args.p7_root, c), "test.tsv")},
                {"config": c, "path": os.path.join(sources.fleurs_dir(args.p7_root, c), "audio", "test.tar.gz")})]},
            "fma": {**sources.FMA, "files": [{"path": p} for p in sources.fma_shards(args.p7_root)], "note": "bank: _bank/bank.json"} if music else None,
            "esc50": {**sources.ESC50, "files": [{"path": sources.esc50_zip(args.p7_root)}], "note": "bank: _bank/bank.json"} if sfx else None,
        }
        src_meta = {k: v for k, v in src_meta.items() if v}
        if args.hash_sources:
            for v in src_meta.values():
                for f in v.get("files", []):
                    p = f.get("path")
                    if p and os.path.isfile(p):
                        f["sha256"] = sha256_file(p)
    if not args.no_real:
        real_items, real_notes = real.ingest(os.path.join(out, "real"), out, write_pcm16, log)
        items += real_items
        if real_items:
            src_meta["real"] = {"dataset": "local drop-in (consented VRChat clips)", "license": "private; never committed or uploaded", "note": "; ".join(real_notes)}
        real.write_readme(os.path.join(out, "real"))
    items.sort(key=lambda it: (it["kind"], it["language"], it["rank"], it["condition"]))
    langs = {}
    for it in items:
        d = langs.setdefault(it["language"], {"items": 0, "utterances": set(), "seconds": 0.0})
        d["items"] += 1
        d["utterances"].add(it["utterance"])
        d["seconds"] += it["seconds"]
    conditions = [{"id": c["id"], "kind": c["kind"], "description": c["description"], "snr_db": c.get("snr_db"), "params": c.get("params") or {}} for c in recipe["conditions"]]
    if any(it["kind"] == "real" for it in items):
        for cid in sorted({it["condition"] for it in items if it["kind"] == "real"}):
            conditions.append({"id": cid, "kind": "real", "description": "Real recordings from the drop-in folder (corpus/real).", "snr_db": None, "params": {}})
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "name": recipe["name"],
        "description": recipe.get("description", ""),
        "created_unix_ms": int(time.time() * 1000),
        "sample_rate": dsp.SR,
        "channels": 1,
        "format": "wav_pcm16",
        "recipe": recipe,
        "recipe_sha256": recipe_sha,
        "content_sha256": content_sha256(items),
        "builder": builder_stamp(),
        "sources": src_meta,
        "conditions": conditions,
        "subsets": recipe["subsets"],
        "languages": {k: {"items": v["items"], "utterances": len(v["utterances"]), "hours": round(v["seconds"] / 3600, 3)} for k, v in sorted(langs.items())},
        "items": items,
    }
    tmp = os.path.join(out, "corpus.json.part")
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=1)
    os.replace(tmp, os.path.join(out, "corpus.json"))
    log(f"corpus.json: {len(items)} items, {sum(it['seconds'] for it in items) / 3600:.2f} h")
    return manifest


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default=os.path.join(HERE, "recipe.json"))
    ap.add_argument("--p7-root", default=DEFAULT_P7)
    ap.add_argument("--out")
    ap.add_argument("--languages", type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    ap.add_argument("--per-language", type=int)
    ap.add_argument("--conditions", type=lambda s: [x.strip() for x in s.split(",") if x.strip()])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--priority", choices=("idle", "below_normal", "normal"), default="below_normal")
    ap.add_argument("--resume", action="store_true", help="keep audio files a previous build wrote (same recipe)")
    ap.add_argument("--no-real", action="store_true", help="skip the real/ drop-in folder")
    ap.add_argument("--real-only", action="store_true", help="only (re)ingest real/ clips (no synthetic items)")
    ap.add_argument("--hash-sources", action="store_true", help="record the sha256 of the source archives (reads about 5 GB)")
    ap.add_argument("--verify", action="store_true", help="re-hash the files of an existing corpus.json")
    args = ap.parse_args(argv)
    args.out = args.out or os.path.join(args.p7_root, "corpus")
    set_priority(args.priority)
    if args.verify:
        return verify(args.out)
    manifest = build(args)
    try:
        sys.path.insert(0, os.path.dirname(HERE))
        from validate import schema_errors  # noqa: PLC0415

        errs = schema_errors(manifest)
        if errs:
            print("manifest does not validate:", *errs[:20], sep="\n  ")
            return 1
        print("manifest validates against sigil-speech-corpus/1")
    except ImportError:
        print("jsonschema not importable: manifest not validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
