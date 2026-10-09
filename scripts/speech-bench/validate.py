#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Validate P7 campaign files against their frozen JSON Schemas (draft 2020-12).

    python scripts/speech-bench/validate.py <file.json> [...]          # schema chosen by the file's "schema" field
    python scripts/speech-bench/validate.py --jsonl mt_line corpus/mt/wmt24pp.jsonl
    python scripts/speech-bench/validate.py --jsonl tts_prompt corpus/tts/prompts.jsonl

Known schema ids: sigil-speech-bench-spec/1, sigil-speech-bench-report/1, sigil-speech-bench-export/1,
sigil-speech-corpus/1, sigil-speech-corpus-recipe/1, and the listening-test formats
(sigil-listening-mt/1, sigil-listening-tts/1, sigil-listening-key/1, sigil-listening-results/1).
Exports also pass the privacy rules (no absolute paths, no audio, no text of real clips, text
only when the export says include_text). A spec also gets the cross-reference checks serde cannot
express (unique ids, runs naming existing engines whose task the mode takes). Exit 0 when every
file is valid, 1 otherwise. Needs ``jsonschema`` (in the P7 venv).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMAS = os.path.join(HERE, "schemas")
LISTENING_SCHEMAS = os.path.join(os.path.dirname(os.path.dirname(HERE)), "tools", "listening-test", "schemas")

SCHEMA_FILES = {
    "sigil-speech-bench-spec/1": os.path.join(SCHEMAS, "speech-bench-spec.schema.json"),
    "sigil-speech-bench-report/1": os.path.join(SCHEMAS, "speech-bench-report.schema.json"),
    "sigil-speech-bench-export/1": os.path.join(SCHEMAS, "speech-bench-export.schema.json"),
    "sigil-speech-corpus/1": os.path.join(SCHEMAS, "speech-corpus.schema.json"),
    "sigil-speech-corpus-recipe/1": os.path.join(SCHEMAS, "speech-corpus-recipe.schema.json"),
    "sigil-listening-mt/1": os.path.join(LISTENING_SCHEMAS, "listening-mt.schema.json"),
    "sigil-listening-tts/1": os.path.join(LISTENING_SCHEMAS, "listening-tts.schema.json"),
    "sigil-listening-key/1": os.path.join(LISTENING_SCHEMAS, "listening-key.schema.json"),
    "sigil-listening-results/1": os.path.join(LISTENING_SCHEMAS, "listening-results.schema.json"),
}
TEXT_SETS = os.path.join(SCHEMAS, "speech-text-sets.schema.json")

# Which tasks each run mode takes (the runner's rule; frozen with the spec).
MODE_TASKS = {
    "accuracy": {"stt"},
    "cpu_speed": {"stt", "mt", "tts"},
    "gpu_under_load": {"stt", "mt", "tts"},
    "mt_quality": {"mt"},
    "tts_listening": {"tts"},
}
BACKENDS = {
    "stt": {"whisper", "http", "sherpa-online", "sherpa-offline", "transcribe", "pipeline", "crispasr", "hypotheses"},
    "mt": {"nllb", "openai", "hypotheses", "bergamot", "llama-server"},
    "tts": {"local-tts", "tts-helper", "openai-speech", "rendered"},
}
ABS_PATH = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\|/(?:Users|home|mnt|tmp)/)|[\\/](?:Users|home)[\\/][^\\/]+[\\/]")


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def registry():
    """A referencing registry with every schema (cross-file $refs resolve by $id)."""
    from referencing import Registry, Resource  # noqa: PLC0415

    resources = []
    for path in list(SCHEMA_FILES.values()) + [TEXT_SETS]:
        if os.path.isfile(path):
            doc = _load(path)
            resources.append((doc["$id"], Resource.from_contents(doc)))
    return Registry().with_resources(resources)


def validator_for(schema_id: str):
    import jsonschema  # noqa: PLC0415

    path = SCHEMA_FILES.get(schema_id)
    if path is None or not os.path.isfile(path):
        raise KeyError(f"unknown schema {schema_id!r}")
    schema = _load(path)
    return jsonschema.Draft202012Validator(schema, registry=registry())


def schema_errors(doc, schema_id: str | None = None) -> list[str]:
    schema_id = schema_id or (doc.get("schema") if isinstance(doc, dict) else None)
    if not schema_id:
        return ["no \"schema\" field"]
    try:
        v = validator_for(schema_id)
    except KeyError as e:
        return [str(e)]
    out = []
    for err in sorted(v.iter_errors(doc), key=lambda e: list(e.absolute_path)):
        where = "/".join(str(p) for p in err.absolute_path) or "(root)"
        out.append(f"{where}: {err.message}")
    return out


def spec_errors(spec: dict) -> list[str]:
    """Cross-reference rules of sigil-speech-bench-spec/1 beyond the schema."""
    out = []
    engines = {}
    for e in spec.get("engines", []):
        if e["id"] in engines:
            out.append(f"engines: duplicate id {e['id']!r}")
        engines[e["id"]] = e
        if e.get("backend") not in BACKENDS.get(e.get("task"), set()):
            out.append(f"engine {e['id']}: backend {e.get('backend')!r} is not a {e.get('task')} backend ({', '.join(sorted(BACKENDS.get(e.get('task'), [])))})")
        opts = e.get("options") or {}
        for k in opts:
            if re.search(r"(api[_-]?key|token|secret|password)$", k, re.I) and k != "apiKeyEnv":
                out.append(f"engine {e['id']}: options.{k} looks like a secret (name an environment variable with apiKeyEnv instead)")
    names = set()
    for r in spec.get("runs", []):
        if r["name"] in names:
            out.append(f"runs: duplicate name {r['name']!r}")
        names.add(r["name"])
        tasks = MODE_TASKS.get(r.get("mode"), set())
        chosen = r.get("engines") or [e for e, v in engines.items() if v.get("task") in tasks]
        if not chosen:
            out.append(f"run {r['name']}: no engine of task {sorted(tasks)} in the spec")
        for eid in chosen:
            e = engines.get(eid)
            if e is None:
                out.append(f"run {r['name']}: unknown engine {eid!r}")
            elif e.get("task") not in tasks:
                out.append(f"run {r['name']}: engine {eid} is {e.get('task')}, mode {r.get('mode')} takes {sorted(tasks)}")
        if r.get("mode") in ("accuracy", "cpu_speed", "gpu_under_load") and any((engines.get(x) or {}).get("task") == "stt" for x in chosen) and "corpus" not in spec:
            out.append(f"run {r['name']}: STT runs need the spec's corpus")
        tts_intel = ((r.get("tts") or {}).get("intelligibility") or {}).get("engine")
        if tts_intel and (engines.get(tts_intel) or {}).get("task") != "stt":
            out.append(f"run {r['name']}: tts.intelligibility.engine {tts_intel!r} is not an stt engine of the spec")
    return out


def _walk_strings(value, path=""):
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _walk_strings(v, f"{path}/{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _walk_strings(v, f"{path}/{i}")
    elif isinstance(value, str):
        yield path, value


def privacy_errors(export: dict) -> list[str]:
    """The friend-export rules (sigil-speech-bench-export/1)."""
    out = []
    for path, s in _walk_strings(export):
        if ABS_PATH.search(s):
            out.append(f"{path}: looks like a local path ({s[:60]!r})")
    env = export.get("environment") or {}
    for k in env:
        if k.lower() in ("hostname", "host", "computer_name", "user", "username", "user_name"):
            out.append(f"environment/{k}: host or user names never go into an export")
    include_text = bool((export.get("export") or {}).get("include_text"))
    for ri, run in enumerate(export.get("runs", [])):
        if run.get("files"):
            out.append(f"runs/{ri}/files: an export lists no files")
        for si, res in enumerate(run.get("results", [])):
            for ii, it in enumerate(res.get("items", [])):
                where = f"runs/{ri}/results/{si}/items/{ii}"
                if it.get("audio") is not None:
                    out.append(f"{where}/audio: an export carries no audio")
                for key in ("text", "asr_text"):
                    if it.get(key) is not None:
                        if str(it.get("item", "")).startswith("real/"):
                            out.append(f"{where}/{key}: text of a real clip never goes into an export")
                        elif not include_text:
                            out.append(f"{where}/{key}: text present although export.include_text is false")
                for seg in it.get("segments") or []:
                    if seg.get("text") and not include_text:
                        out.append(f"{where}/segments: text present although export.include_text is false")
                        break
    return out


def validate_file(path: str) -> list[str]:
    doc = _load(path)
    errors = schema_errors(doc)
    sid = doc.get("schema") if isinstance(doc, dict) else None
    if not errors and sid == "sigil-speech-bench-spec/1":
        errors += spec_errors(doc)
    if sid == "sigil-speech-bench-export/1":
        errors += privacy_errors(doc)
    return errors


def validate_jsonl(kind: str, path: str) -> list[str]:
    import jsonschema  # noqa: PLC0415

    schema = _load(TEXT_SETS)
    sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], "$ref": f"#/$defs/{kind}"}
    v = jsonschema.Draft202012Validator(sub)
    out, ids = [], set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            row = json.loads(line)
            for err in v.iter_errors(row):
                out.append(f"line {n}: {err.message}")
            if row.get("id") in ids:
                out.append(f"line {n}: duplicate id {row.get('id')!r}")
            ids.add(row.get("id"))
    return out


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*")
    ap.add_argument("--jsonl", nargs=2, action="append", metavar=("KIND", "FILE"), default=[], help="KIND: mt_line or tts_prompt")
    args = ap.parse_intermixed_args(argv)
    bad = 0
    for path in args.files:
        errs = validate_file(path)
        print(f"{'ok  ' if not errs else 'FAIL'} {path}")
        for e in errs[:50]:
            print(f"     {e}")
        bad += bool(errs)
    for kind, path in args.jsonl:
        errs = validate_jsonl(kind, path)
        print(f"{'ok  ' if not errs else 'FAIL'} {path} ({kind})")
        for e in errs[:50]:
            print(f"     {e}")
        bad += bool(errs)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
