# SPDX-License-Identifier: AGPL-3.0-or-later
"""Test helpers: paths and a tiny synthetic P7 root (FLEURS, FMA, ESC-50 look-alikes)."""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sys
import tarfile
import wave
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(BENCH))
LISTENING = os.path.join(REPO, "tools", "listening-test")
FIXTURES = os.path.join(HERE, "fixtures")
EXAMPLES = os.path.join(BENCH, "examples")

# scripts/speech-bench first: its `score` package must win over tools/listening-test/score.py (the
# voice-changer A/B scorer), so the package is imported before the listening folder joins the path.
if BENCH not in sys.path:
    sys.path.insert(0, BENCH)
import score  # noqa: E402,F401  (pins the package in sys.modules)

if LISTENING not in sys.path:
    sys.path.append(LISTENING)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)


def write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def wav_bytes(x: np.ndarray, sr: int = 16000) -> bytes:
    pcm = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def voice_like(seconds: float, seed: int, sr: int = 16000, level: float = 0.1) -> np.ndarray:
    """A speech-like test signal: harmonic bursts (syllables) with pauses."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 110 + 60 * rng.random()
    x = sum(np.sin(2 * math.pi * f0 * k * t + rng.random() * 6) / k for k in range(1, 6))
    env = np.zeros(n)
    pos = int(0.2 * sr)
    while pos < n - int(0.3 * sr):
        length = int((0.15 + 0.2 * rng.random()) * sr)
        env[pos: pos + length] = np.hanning(length)
        pos += length + int((0.05 + 0.25 * rng.random()) * sr)
    return level * x * env / 3


def make_fake_p7(root: str, languages=(("en", "en_us"), ("ja", "ja_jp")), per_lang: int = 8) -> str:
    """A P7 root with FLEURS-like test sets, an FMA-like parquet shard and an ESC-50-like zip."""
    for lang, cfg in languages:
        d = os.path.join(root, "eval", "fleurs", cfg)
        os.makedirs(os.path.join(d, "audio"), exist_ok=True)
        rows = []
        with tarfile.open(os.path.join(d, "audio", "test.tar.gz"), "w:gz") as tar:
            for i in range(per_lang):
                sid = 100 + i // 2  # two recordings per sentence
                name = f"{lang}{i:04d}.wav"
                x = voice_like(3.5 + 0.3 * i, seed=sum(map(ord, lang)) * 100 + i, level=0.02 if i % 3 == 0 else 0.2)
                data = wav_bytes(x)
                info = tarfile.TarInfo(name=f"test/{name}")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                text = f"sentence {sid} in {lang}" if lang != "ja" else f"文{sid}です。"
                rows.append([str(sid), name, text, text.lower(), "x", str(len(x)), "MALE" if i % 2 else "FEMALE"])
        with open(os.path.join(d, "test.tsv"), "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
            w.writerows(rows)
    # FMA-like parquet: audio {bytes (WAV here), path}, title, artist, license, url
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    fma = os.path.join(root, "bleed", "fma-commercial-16khz", "data")
    os.makedirs(fma, exist_ok=True)
    rng = np.random.default_rng(7)
    audios, titles = [], []
    for k in range(6):
        sr = 16000
        t = np.arange(int(80 * sr)) / sr
        x = 0.2 * np.sin(2 * math.pi * (220 + 40 * k) * t) * (0.6 + 0.4 * np.sin(2 * math.pi * 0.5 * t)) + 0.02 * rng.standard_normal(len(t))
        audios.append({"bytes": wav_bytes(x), "path": f"{k:06d}_16khz_mono.mp3"})
        titles.append(f"track {k}")
    table = pa.table({"audio": audios, "title": titles, "artist": ["a"] * 6, "license": [4] * 6, "url": [""] * 6})
    pq.write_table(table, os.path.join(fma, "train-00000-of-00013.parquet"), row_group_size=3)
    # ESC-50-like zip
    esc = os.path.join(root, "bleed", "esc50")
    os.makedirs(esc, exist_ok=True)
    zroot = "ESC-50-test/"
    meta = io.StringIO()
    w = csv.writer(meta)
    w.writerow(["filename", "fold", "target", "category", "esc10", "src_file", "take"])
    with zipfile.ZipFile(os.path.join(esc, "ESC-50-test.zip"), "w") as z:
        for cat in ("clapping", "keyboard_typing"):
            for j in range(3):
                fn = f"1-{cat}-{j}.wav"
                sr = 44100
                x = np.zeros(5 * sr)
                x[sr: sr + 4000] = 0.5 * rng.standard_normal(4000) * np.hanning(4000)
                z.writestr(zroot + "audio/" + fn, wav_bytes(x, sr))
                w.writerow([fn, "1", "0", cat, "False", "123", "A"])
        z.writestr(zroot + "meta/esc50.csv", meta.getvalue())
    return root


def small_recipe(languages=("en", "ja"), per_language: int = 3) -> dict:
    """The real recipe, shrunk for tests (small banks, two languages)."""
    r = load_json(os.path.join(BENCH, "corpus", "recipe.json"))
    r["name"] = "test"
    r["languages"] = [x for x in r["languages"] if x["lang"] in languages]
    r["selection"]["per_language"] = per_language
    r["selection"]["babble_pool"] = 2
    r["banks"]["music"]["tracks"] = 2
    r["banks"]["music"]["segments_per_track"] = 1
    r["banks"]["music"]["min_active_dbfs"] = -60.0
    r["banks"]["sfx"]["categories"] = ["clapping", "keyboard_typing"]
    r["banks"]["sfx"]["clips_per_category"] = 2
    return r
