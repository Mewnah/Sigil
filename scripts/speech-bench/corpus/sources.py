# SPDX-License-Identifier: AGPL-3.0-or-later
"""Readers of the prefetched P7 datasets (FLEURS test sets, the FMA 16 kHz shard, ESC-50) and the
deterministic selection of the corpus utterances. Nothing here downloads anything.

Layout under the P7 root (default %LOCALAPPDATA%/sigil-dev-cache/p7, from the prefetch):
- ``eval/fleurs/<config>/test.tsv`` + ``eval/fleurs/<config>/audio/test.tar.gz``
- ``bleed/fma-commercial-16khz/data/train-*.parquet`` (``audio`` = {bytes (MP3), path}, title, artist, license, url)
- ``bleed/esc50/ESC-50-*.zip`` (``meta/esc50.csv`` + ``audio/*.wav`` at 44.1 kHz)
"""

from __future__ import annotations

import csv
import glob
import hashlib
import io
import os
import tarfile
import zipfile

import numpy as np

# Dataset identities of the P7 prefetch (p7_prefetch_manifest.json); recorded in every manifest.
FLEURS = {"dataset": "google/fleurs", "revision": "70bb2e84b976b7e960aa89f1c648e09c59f894dd", "license": "CC-BY-4.0"}
FMA = {"dataset": "benjamin-paine/free-music-archive-commercial-16khz-full", "revision": "4fad7a587e80adca671184bd017d1d6ae8ea9a1d",
       "license": "per-track Creative Commons licences that allow commercial use (FMA); shard 0 of 13"}
ESC50 = {"dataset": "karolpiczak/ESC-50", "revision": "33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6", "license": "CC BY-NC 3.0 (clips from freesound.org)"}

FLEURS_SR = 16000


def seeded_key(*parts) -> str:
    """Deterministic sort key of ``parts`` (sha256 hex)."""
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def seed_int(*parts) -> int:
    return int(seeded_key(*parts)[:15], 16)


# ---------------------------------------------------------------------------------------------
# FLEURS
# ---------------------------------------------------------------------------------------------

def fleurs_dir(p7_root: str, config: str) -> str:
    return os.path.join(p7_root, "eval", "fleurs", config)


def read_fleurs_tsv(path: str) -> list[dict]:
    """Rows of a FLEURS split TSV: id, file_name, raw_transcription, transcription, phonemes,
    num_samples, gender."""
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for rec in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            if len(rec) < 7:
                continue
            try:
                rows.append({
                    "sentence_id": int(rec[0]),
                    "file_name": rec[1],
                    "raw_transcription": rec[2].strip(),
                    "transcription": rec[3].strip(),
                    "num_samples": int(rec[5]),
                    "gender": rec[6].strip().lower() or None,
                })
            except ValueError:
                continue
    return rows


def _interleave_by_gender(rows: list[dict], count: int) -> list[dict]:
    """Take ``count`` rows in key order, alternating genders while both remain."""
    by_gender: dict[str, list[dict]] = {}
    for r in rows:
        by_gender.setdefault(r.get("gender") or "unknown", []).append(r)
    taken: dict[str, int] = {g: 0 for g in by_gender}
    pos: dict[str, int] = {g: 0 for g in by_gender}
    out = []
    while len(out) < count:
        candidates = [g for g in by_gender if pos[g] < len(by_gender[g])]
        if not candidates:
            break
        g = min(candidates, key=lambda g: (taken[g], by_gender[g][pos[g]]["_key"]))
        out.append(by_gender[g][pos[g]])
        pos[g] += 1
        taken[g] += 1
    return out


def select_fleurs(rows: list[dict], lang: str, seed: int, per_language: int, min_s: float, max_s: float,
                  one_per_sentence: bool = True, balance_gender: bool = True, babble_pool: int = 24) -> tuple[list[dict], list[dict]]:
    """(selected utterances in rank order, babble pool) of one language. Deterministic: every
    choice is ordered by sha256(seed | lang | file name). Sentences are taken in the order of their
    first recording's key; with ``balance_gender`` each sentence contributes its first recording
    of the gender chosen least so far (FLEURS often has two or three speakers per sentence), else
    its first recording."""
    ok = [dict(r, _key=seeded_key(seed, lang, r["file_name"])) for r in rows
          if r["raw_transcription"] and min_s <= r["num_samples"] / FLEURS_SR <= max_s]
    ok.sort(key=lambda r: r["_key"])
    if one_per_sentence:
        by_sentence: dict[int, list[dict]] = {}
        for r in ok:
            by_sentence.setdefault(r["sentence_id"], []).append(r)
        genders = sorted({r.get("gender") or "unknown" for r in ok})
        counts = {g: 0 for g in genders}
        chosen = []
        for sid in sorted(by_sentence, key=lambda s: by_sentence[s][0]["_key"]):
            if len(chosen) >= per_language:
                break
            recs = by_sentence[sid]
            pick = recs[0]
            if balance_gender:
                for g in sorted(genders, key=lambda g: (counts[g], g)):
                    match = next((r for r in recs if (r.get("gender") or "unknown") == g), None)
                    if match is not None:
                        pick = match
                        break
            counts[pick.get("gender") or "unknown"] += 1
            chosen.append(pick)
    else:
        chosen = _interleave_by_gender(ok, per_language) if balance_gender else ok[:per_language]
    chosen_sentences = {r["sentence_id"] for r in chosen}
    rest = [r for r in ok if r["sentence_id"] not in chosen_sentences]
    pool = _interleave_by_gender(rest, babble_pool) if balance_gender else rest[:babble_pool]
    for rank, r in enumerate(chosen):
        r["rank"] = rank
    return chosen, pool


def extract_fleurs(tar_path: str, names: set[str], dest: str) -> dict[str, str]:
    """Extract the WAVs whose base names are in ``names`` from a FLEURS ``test.tar.gz`` (one
    streaming pass); files already present are kept. Returns {name: path}."""
    os.makedirs(dest, exist_ok=True)
    found = {n: os.path.join(dest, n) for n in names if os.path.isfile(os.path.join(dest, n))}
    missing = set(names) - set(found)
    if not missing:
        return found
    with tarfile.open(tar_path, "r|gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            base = os.path.basename(member.name)
            if base in missing:
                data = tar.extractfile(member).read()
                tmp = os.path.join(dest, base + ".part")
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, os.path.join(dest, base))
                found[base] = os.path.join(dest, base)
                missing.discard(base)
                if not missing:
                    break
    if missing:
        raise FileNotFoundError(f"{tar_path}: {len(missing)} selected files not in the archive (e.g. {sorted(missing)[:3]})")
    return found


def read_wav_mono(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415

    x, sr = sf.read(path, dtype="float64", always_2d=True)
    return x.mean(axis=1), int(sr)


# ---------------------------------------------------------------------------------------------
# FMA music shard
# ---------------------------------------------------------------------------------------------

def fma_shards(p7_root: str) -> list[str]:
    return sorted(glob.glob(os.path.join(p7_root, "bleed", "fma-commercial-16khz", "data", "*.parquet")))


def fma_tracks(shard: str) -> list[dict]:
    """Track metadata of one shard (no audio): {row_group, row, path, title, artist, license, url}."""
    import pyarrow.parquet as pq  # noqa: PLC0415

    pf = pq.ParquetFile(shard)
    out = []
    for g in range(pf.metadata.num_row_groups):
        cols = [c for c in pf.schema_arrow.names if c != "audio"]
        meta = pf.read_row_group(g, columns=cols).to_pylist()
        paths = pf.read_row_group(g, columns=["audio"]).column("audio").combine_chunks().field("path").to_pylist()
        for i, (m, p) in enumerate(zip(meta, paths)):
            out.append({"shard": os.path.basename(shard), "row_group": g, "row": i, "path": p, "title": m.get("title"), "artist": m.get("artist"),
                        "license": m.get("license"), "url": m.get("url")})
    return out


def fma_decode_group(shard: str, group: int, rows: set[int]) -> dict[int, tuple[np.ndarray, int]]:
    """Decode the MP3s of ``rows`` of a row group: {row: (mono float64, rate)}."""
    import pyarrow.parquet as pq  # noqa: PLC0415
    import soundfile as sf  # noqa: PLC0415

    pf = pq.ParquetFile(shard)
    col = pf.read_row_group(group, columns=["audio"]).column("audio").combine_chunks().field("bytes")
    out = {}
    for r in sorted(rows):
        data = col[r].as_py()
        x, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
        out[r] = (x.mean(axis=1), int(sr))
    return out


# ---------------------------------------------------------------------------------------------
# ESC-50
# ---------------------------------------------------------------------------------------------

def esc50_zip(p7_root: str) -> str | None:
    found = sorted(glob.glob(os.path.join(p7_root, "bleed", "esc50", "ESC-50-*.zip")))
    return found[0] if found else None


def esc50_rows(zip_path: str) -> tuple[list[dict], str]:
    """(meta rows {filename, fold, target, category, src_file, take}, archive root prefix)."""
    with zipfile.ZipFile(zip_path) as z:
        meta = next(n for n in z.namelist() if n.endswith("meta/esc50.csv"))
        root = meta[: -len("meta/esc50.csv")]
        with z.open(meta) as raw:
            rows = list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8")))
    return rows, root


def esc50_read(zip_path: str, root: str, filename: str) -> tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415

    with zipfile.ZipFile(zip_path) as z:
        data = z.read(root + "audio/" + filename)
    x, sr = sf.read(io.BytesIO(data), dtype="float64", always_2d=True)
    return x.mean(axis=1), int(sr)
