#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Build the P7 text sets (light, text only) from the prefetched corpora, outside the repo.

    python scripts/speech-bench/corpus/build_text_sets.py [--p7-root %LOCALAPPDATA%/sigil-dev-cache/p7]
        [--out <p7-root>/corpus] [--per-direction 200] [--seed 20261002]

MT sets (``<out>/mt/<set>.jsonl``; the mt-bench corpus format plus ``set``, ``domain``, ``doc``,
``license``, which mt-bench ignores; validate with ``validate.py --jsonl mt_line``):
- ``wmt24pp.jsonl``: English -> ja, zh, zh-Hant, ko, de, fr, es, ru, pt from WMT24++ (post-edited
  references ``target``), social and speech domains first, bad sources and the canary dropped,
  ``--per-direction`` lines per direction; ``doc`` = document id (previous segment = context);
- ``flores200.jsonl``: ja, zh, ko, de, fr, es, ru, pt -> English from FLORES-200 devtest (the same
  sentences in every direction);
- ``flores200-eu.jsonl``: the same devtest sentences as ``flores200.jsonl`` in the directions it lacks for the European
  pairs (English -> de, es, fr, pt, it, nl and it, nl -> English), so Italian and Dutch have a row (WMT24++ has neither);
- ``bsd.jsonl``: Japanese <-> English business dialogues (BSD test), whole dialogues;
- ``sigil-casual.jsonl``: the 24-line stream / VRChat smoke corpus of P4 (scripts/mt-bench), both
  directions (a smoke check, not a quality benchmark).

TTS prompts (``<out>/tts/prompts.jsonl``; validate with ``validate.py --jsonl tts_prompt``), per
language (en, ja, zh, ko, es, de, fr, ru, pt): 8 ``casual`` lines (the smoke corpus, or
``data/casual_tts.json``), 8 ``read`` FLEURS sentences, 2 ``numbers`` FLEURS sentences with digits
and 1 ``long`` passage (three read sentences). Every choice is seeded (sha256 order).
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(HERE))

from corpus.sources import read_fleurs_tsv, seeded_key  # noqa: E402

DEFAULT_P7 = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache", "p7")
WMT = {"ja_JP": "ja", "zh_CN": "zh", "zh_TW": "zh-Hant", "ko_KR": "ko", "de_DE": "de", "fr_FR": "fr", "es_MX": "es", "ru_RU": "ru", "pt_BR": "pt"}
FLORES = {"ja": "jpn_Jpan", "zh": "zho_Hans", "ko": "kor_Hang", "de": "deu_Latn", "fr": "fra_Latn", "es": "spa_Latn", "ru": "rus_Cyrl", "pt": "por_Latn"}
FLORES_EU = {"de": "deu_Latn", "es": "spa_Latn", "fr": "fra_Latn", "pt": "por_Latn", "it": "ita_Latn", "nl": "nld_Latn"}
TTS_LANGS = {"en": "en_us", "ja": "ja_jp", "zh": "cmn_hans_cn", "ko": "ko_kr", "es": "es_419", "de": "de_de", "fr": "fr_fr", "ru": "ru_ru", "pt": "pt_br"}
LICENSES = {"wmt24pp": "Apache-2.0", "flores200": "CC-BY-SA-4.0", "flores200-eu": "CC-BY-SA-4.0", "bsd": "CC BY-NC-SA 4.0", "sigil-casual": "AGPL-3.0-or-later (Sigil)",
            "fleurs": "CC-BY-4.0"}


def read_jsonl(path: str) -> list[dict]:
    """JSON lines (blank and ``#`` lines skipped)."""
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip() and not line.lstrip().startswith("#")]


def write_jsonl(path: str, rows: list[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".part", "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(path + ".part", path)


def wmt24pp(p7: str, n: int, seed: int) -> list[dict]:
    out = []
    folder = os.path.join(p7, "eval", "wmt24pp")
    for code, tgt in WMT.items():
        path = os.path.join(folder, f"en-{code}.jsonl")
        if not os.path.isfile(path):
            continue
        rows = read_jsonl(path)
        rows = [r for r in rows if not r.get("is_bad_source") and r.get("domain") != "canary" and r.get("source") and r.get("target")]
        preferred = [r for r in rows if r["domain"] in ("social", "speech")]
        rest = [r for r in rows if r["domain"] not in ("social", "speech")]
        key = lambda r: seeded_key(seed, "wmt24pp", r["document_id"], r["segment_id"])  # noqa: E731 - same lines in every direction
        chosen = sorted(preferred, key=key)[:n]
        if len(chosen) < n:
            chosen += sorted(rest, key=key)[: n - len(chosen)]
        chosen.sort(key=lambda r: (r["document_id"], r["segment_id"]))
        for i, r in enumerate(chosen, 1):
            out.append({"id": f"wmt24pp:en-{tgt}:{i}", "source": "en", "target": tgt, "src": r["source"], "ref": r["target"], "set": "wmt24pp",
                        "domain": r["domain"], "doc": f"{r['document_id']}#{r['segment_id']}", "license": LICENSES["wmt24pp"]})
    return out


def flores200(p7: str, n: int, seed: int) -> list[dict]:
    path = os.path.join(p7, "eval", "flores200", "flores200_dataset.tar.gz")
    if not os.path.isfile(path):
        return []
    want = {"eng_Latn", *FLORES.values()}
    texts, meta = {}, []
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            base = os.path.basename(m.name)
            if m.isfile() and base.endswith(".devtest") and base[: -len(".devtest")] in want:
                texts[base[: -len(".devtest")]] = tar.extractfile(m).read().decode("utf-8").splitlines()
            elif m.isfile() and base == "metadata_devtest.tsv":
                meta = list(csv.DictReader(io.TextIOWrapper(tar.extractfile(m), encoding="utf-8"), delimiter="\t"))
    eng = texts.get("eng_Latn", [])
    idx = sorted(range(len(eng)), key=lambda i: seeded_key(seed, "flores200", i))[:n]
    idx.sort()
    out = []
    for lang, code in FLORES.items():
        src = texts.get(code)
        if not src:
            continue
        for k, i in enumerate(idx, 1):
            topic = (meta[i].get("topic") if i < len(meta) else None) or None
            out.append({"id": f"flores200:{lang}-en:{k}", "source": lang, "target": "en", "src": src[i], "ref": eng[i], "set": "flores200",
                        "domain": topic, "doc": f"devtest#{i}", "license": LICENSES["flores200"]})
    return out


def flores200_eu(p7: str, n: int, seed: int) -> list[dict]:
    """European directions of FLORES-200 devtest: the sentences ``flores200()`` picks (same seed, same ranking), English ->
    de / es / fr / pt / it / nl and it / nl -> English (the other ``X -> en`` directions are in ``flores200.jsonl``)."""
    path = os.path.join(p7, "eval", "flores200", "flores200_dataset.tar.gz")
    if not os.path.isfile(path):
        return []
    want = {"eng_Latn", *FLORES_EU.values()}
    texts, meta = {}, []
    with tarfile.open(path) as tar:
        for m in tar.getmembers():
            base = os.path.basename(m.name)
            if m.isfile() and base.endswith(".devtest") and base[: -len(".devtest")] in want:
                texts[base[: -len(".devtest")]] = tar.extractfile(m).read().decode("utf-8").splitlines()
            elif m.isfile() and base == "metadata_devtest.tsv":
                meta = list(csv.DictReader(io.TextIOWrapper(tar.extractfile(m), encoding="utf-8"), delimiter="\t"))
    eng = texts.get("eng_Latn", [])
    idx = sorted(range(len(eng)), key=lambda i: seeded_key(seed, "flores200", i))[:n]  # the same pick as flores200()
    idx.sort()
    out = []

    def rows(source: str, target: str, src_lines: list[str], ref_lines: list[str]) -> None:
        for k, i in enumerate(idx, 1):
            topic = (meta[i].get("topic") if i < len(meta) else None) or None
            out.append({"id": f"flores200-eu:{source}-{target}:{k}", "source": source, "target": target, "src": src_lines[i], "ref": ref_lines[i],
                        "set": "flores200-eu", "domain": topic, "doc": f"devtest#{i}", "license": LICENSES["flores200-eu"]})

    for lang, code in FLORES_EU.items():
        if texts.get(code):
            rows("en", lang, eng, texts[code])
    for lang in ("it", "nl"):
        if texts.get(FLORES_EU[lang]):
            rows(lang, "en", texts[FLORES_EU[lang]], eng)
    return out


def bsd(p7: str, n: int, seed: int) -> list[dict]:
    path = os.path.join(p7, "eval", "bsd-ja-en", "test.json")
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        dialogues = json.load(f)
    dialogues.sort(key=lambda d: seeded_key(seed, "bsd", d["id"]))
    chosen, total = [], 0
    for d in dialogues:
        if total >= n:
            break
        chosen.append(d)
        total += len(d["conversation"])
    chosen.sort(key=lambda d: d["id"])
    out = []
    for direction, (sk, tk) in {"ja-en": ("ja_sentence", "en_sentence"), "en-ja": ("en_sentence", "ja_sentence")}.items():
        k = 0
        for d in chosen:
            for turn in d["conversation"]:
                if not turn.get(sk) or not turn.get(tk):
                    continue
                k += 1
                s, t = direction.split("-")
                out.append({"id": f"bsd:{direction}:{k}", "source": s, "target": t, "src": turn[sk], "ref": turn[tk], "set": "bsd",
                            "domain": d.get("tag"), "doc": f"{d['id']}#{turn.get('no')}", "license": LICENSES["bsd"]})
    return out


def sigil_casual() -> list[dict]:
    path = os.path.join(REPO, "scripts", "mt-bench", "corpus.sample.jsonl")
    out = []
    for r in read_jsonl(path):
        out.append({"id": f"sigil-casual:{r['id']}", "source": r["source"], "target": r["target"], "src": r["src"], "ref": r["ref"],
                    "set": "sigil-casual", "domain": "casual", "doc": None, "license": LICENSES["sigil-casual"]})
    return out


def _words(text: str, lang: str) -> int:
    if lang in ("ja", "zh"):
        return len(re.sub(r"\s", "", text)) // 2  # rough word equivalent for length filters
    return len(text.split())


def tts_prompts(p7: str, seed: int) -> list[dict]:
    casual = {}
    for r in read_jsonl(os.path.join(REPO, "scripts", "mt-bench", "corpus.sample.jsonl")):
        n = int(r["id"].rsplit(":", 1)[1])
        if r["source"] == "en" and n <= 8:
            casual.setdefault("en", {})[n] = r["src"]
            casual.setdefault(r["target"], {})[n] = r["ref"]
    with open(os.path.join(HERE, "data", "casual_tts.json"), encoding="utf-8") as f:
        extra = json.load(f)["lines"]
    for lang, lines in extra.items():
        casual[lang] = {i: t for i, t in enumerate(lines, 1)}
    out = []
    for lang, cfg in TTS_LANGS.items():
        for i in sorted(casual.get(lang, {})):
            out.append({"id": f"{lang}-casual-{i}", "lang": lang, "text": casual[lang][i], "kind": "casual",
                        "source": "sigil-casual" + ("" if lang in ("en", "ja", "de", "es", "zh", "ko") else ":p7"), "license": LICENSES["sigil-casual"]})
        tsv = os.path.join(p7, "eval", "fleurs", cfg, "test.tsv")
        if not os.path.isfile(tsv):
            continue
        rows = {r["sentence_id"]: r for r in read_fleurs_tsv(tsv)}  # one row per sentence
        ranked = sorted(rows.values(), key=lambda r: seeded_key(seed, "tts", lang, r["sentence_id"]))
        plain = [r for r in ranked if 8 <= _words(r["raw_transcription"], lang) <= 25 and not re.search(r"\d", r["raw_transcription"])]
        digits = [r for r in ranked if re.search(r"\d", r["raw_transcription"]) and _words(r["raw_transcription"], lang) <= 30]
        for i, r in enumerate(plain[:8], 1):
            out.append({"id": f"{lang}-read-{i}", "lang": lang, "text": r["raw_transcription"], "kind": "read",
                        "source": f"fleurs:{cfg}:{r['file_name']}", "license": LICENSES["fleurs"]})
        for i, r in enumerate(digits[:2], 1):
            out.append({"id": f"{lang}-numbers-{i}", "lang": lang, "text": r["raw_transcription"], "kind": "numbers",
                        "source": f"fleurs:{cfg}:{r['file_name']}", "license": LICENSES["fleurs"]})
        long_parts = plain[8:11]
        if len(long_parts) == 3:
            sep = "" if lang in ("ja", "zh") else " "
            out.append({"id": f"{lang}-long-1", "lang": lang, "text": sep.join(r["raw_transcription"] for r in long_parts), "kind": "long",
                        "source": "fleurs:" + cfg + ":" + "+".join(r["file_name"] for r in long_parts), "license": LICENSES["fleurs"]})
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p7-root", default=DEFAULT_P7)
    ap.add_argument("--out")
    ap.add_argument("--per-direction", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261002)
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.p7_root, "corpus")
    sets = {
        "wmt24pp": wmt24pp(args.p7_root, args.per_direction, args.seed),
        "flores200": flores200(args.p7_root, args.per_direction, args.seed),
        "flores200-eu": flores200_eu(args.p7_root, args.per_direction, args.seed),
        "bsd": bsd(args.p7_root, args.per_direction, args.seed),
        "sigil-casual": sigil_casual(),
    }
    for name, rows in sets.items():
        if rows:
            write_jsonl(os.path.join(out, "mt", f"{name}.jsonl"), rows)
        dirs = sorted({f"{r['source']}-{r['target']}" for r in rows})
        print(f"mt/{name}.jsonl: {len(rows)} lines, {len(dirs)} directions ({', '.join(dirs)})")
    prompts = tts_prompts(args.p7_root, args.seed)
    write_jsonl(os.path.join(out, "tts", "prompts.jsonl"), prompts)
    print(f"tts/prompts.jsonl: {len(prompts)} prompts in {len({p['lang'] for p in prompts})} languages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
