"""
Generate locale JSON from English using Google Translate (deep-translator).
Preserves {{interpolation}} and <code>...</code> segments.
Uses one API call per string for reliable results (slow but deterministic).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator

ROOT = Path(__file__).resolve().parents[1]
EN_PATH = ROOT / "public" / "i18n" / "en" / "translation.json"

# Folder name -> Google Translate target code (phase 1 + phase 2 “commonly spoken”)
LOCALES = {
    "es": "es",
    "hi": "hi",
    "ar": "ar",
    "fr": "fr",
    "pt_br": "pt",
    "ru": "ru",
    "ja": "ja",
    "de": "de",
    "ko": "ko",
    "it": "it",
    "id": "id",
    "tr": "tr",
    "vi": "vi",
    "pl": "pl",
    "th": "th",
    "bn": "bn",
    "ur": "ur",
}

PH_INTERP = re.compile(r"\{\{[^}]+\}\}")
PH_CODE = re.compile(r"<code>[^<]*</code>")


def protect(s: str) -> tuple[str, list[str]]:
    placeholders: list[str] = []

    def sub_code(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"⟦{len(placeholders) - 1}⟧"

    def sub_interp(m: re.Match[str]) -> str:
        placeholders.append(m.group(0))
        return f"⟦{len(placeholders) - 1}⟧"

    out = PH_CODE.sub(sub_code, s)
    out = PH_INTERP.sub(sub_interp, out)
    return out, placeholders


def restore(s: str, placeholders: list[str]) -> str:
    for i, p in enumerate(placeholders):
        s = s.replace(f"⟦{i}⟧", p)
    return s


def collect_strings(obj: object, acc: list[str]) -> None:
    if isinstance(obj, dict):
        for _k, v in obj.items():
            collect_strings(v, acc)
    elif isinstance(obj, str):
        acc.append(obj)


def apply_strings(obj: object, it: list[str], idx: list[int]) -> object:
    if isinstance(obj, dict):
        return {k: apply_strings(v, it, idx) for k, v in obj.items()}
    if isinstance(obj, str):
        out = it[idx[0]]
        idx[0] += 1
        return out
    return obj


def translate_flat(flat: list[str], translator: GoogleTranslator, delay: float) -> list[str]:
    out: list[str] = []
    for n, s in enumerate(flat):
        prot, ph = protect(s)
        translated: str | None = None
        for attempt in range(3):
            try:
                translated = restore(translator.translate(prot), ph)
                break
            except Exception as e:
                print(f"  [{n}] attempt {attempt + 1} fail: {e!s}", file=sys.stderr)
                time.sleep(delay * 4)
        if translated is None:
            print(f"  [{n}] using EN fallback", file=sys.stderr)
            translated = s
        out.append(translated)
        time.sleep(delay)
        if (n + 1) % 40 == 0:
            print(f"  … {n + 1}/{len(flat)}", flush=True)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        help="Comma-separated folder names (e.g. ru,ja). Default: all LOCALES.",
    )
    parser.add_argument("--delay", type=float, default=0.08, help="Seconds between API calls.")
    args = parser.parse_args()

    data = json.loads(EN_PATH.read_text(encoding="utf-8"))
    flat: list[str] = []
    collect_strings(data, flat)

    todo = LOCALES
    if args.only:
        keys = {x.strip() for x in args.only.split(",") if x.strip()}
        todo = {k: v for k, v in LOCALES.items() if k in keys}
        missing = keys - set(todo)
        if missing:
            print(f"Unknown locale folders: {missing}", file=sys.stderr)
            sys.exit(1)

    for folder, tgt in todo.items():
        out_dir = ROOT / "public" / "i18n" / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "translation.json"
        print(f"Translating -> {folder} ({tgt}) … ({len(flat)} strings)", flush=True)
        translator = GoogleTranslator(source="en", target=tgt)
        translated_flat = translate_flat(flat, translator, args.delay)
        if len(translated_flat) != len(flat):
            print(f"  FATAL count mismatch", file=sys.stderr)
            sys.exit(1)
        rebuilt = apply_strings(json.loads(json.dumps(data)), translated_flat, [0])
        out_path.write_text(json.dumps(rebuilt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        time.sleep(0.5)
    print("Done.")


if __name__ == "__main__":
    main()
