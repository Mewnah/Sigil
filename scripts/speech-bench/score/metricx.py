#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
"""MetricX-24 runner (reference-based MT quality; lower is better, 0-25). Runs only when its
dependencies are present: it never installs anything.

Needs, in the interpreter given by ``--python`` (default: this one):
- ``torch`` and ``transformers`` (mT5) and ``sentencepiece``: CPU torch is enough for the large
  model (about 5 GB of RAM in bfloat16). The P7 venv deliberately has none of them; the dev PC's
  system Python 3.12 has torch 2.3.1 + transformers 4.39.3 + sentencepiece 0.2.0. To add them to a
  venv (one package at a time, about 200 MB for CPU torch):
  ``python -m pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu``, then
  ``python -m pip install transformers==4.39.3`` and ``python -m pip install sentencepiece==0.2.0``;
- the checkpoint folder (``config.json`` + ``pytorch_model.bin``): google/metricx-24-hybrid-large-v2p6-bfloat16;
- the mT5-XL tokenizer folder (``spiece.model`` + ``tokenizer_config.json``);
- the MetricX code (google-research/metricx, the ``metricx24`` package; the source archive
  ``metricx-<rev>.tar.gz`` is extracted next to itself on first use).

Defaults are the P7 prefetch locations under %LOCALAPPDATA%/sigil-dev-cache/p7/metrics/.

    python score/metricx.py --check                      # detection only (exit 0 ready, 3 missing)
    python score/metricx.py --pairs pairs.en-ja.jsonl [--pairs ...] --out metricx.jsonl [--device cpu|cuda] [--python <exe>]
    python score/metricx.py --self-test                  # 2 lines: a good translation must score lower than a wrong one

Input: the ``pairs.<direction>.jsonl`` files of ``score_mt.py --export-pairs`` ({id, src, ref,
systems: {label: hyp}}). Output: JSON lines ``{system, id, direction, score}`` for ``score_mt.py
--metricx``. Inference follows ``metricx24/predict.py`` (input ``source: ... candidate: ...
reference: ...``, max 1536 tokens, EOS removed, prediction clamped to 0-25) without its
``datasets`` / ``Trainer`` dependencies; ``--device cpu`` (the default) hides CUDA from the worker
(``CUDA_VISIBLE_DEVICES=-1`` and ``torch.cuda.is_available`` patched to False before transformers
loads: MetricX's model code moves its decoder input to CUDA whenever CUDA is available). GPU runs
belong under the GPU timing lock.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
P7 = os.path.join(os.environ.get("LOCALAPPDATA", os.path.expanduser("~")), "sigil-dev-cache", "p7", "metrics")
DEFAULT_MODEL = os.path.join(P7, "metricx-24-hybrid-large-v2p6-bfloat16")
DEFAULT_TOKENIZER = os.path.join(P7, "mt5-xl-tokenizer")
DEFAULT_CODE = os.path.join(P7, "metricx-code")
EXIT_MISSING = 3

# Runs inside the chosen interpreter (may differ from this one): detection or inference.
WORKER = r'''
import json, os, sys
mode, code_dir, model_dir, tok_dir, device, max_len, inp, outp, threads = sys.argv[1:10]
report = {"python": sys.version.split()[0], "executable": sys.executable}
try:
    import torch
    if device != "cuda":
        # MetricX's model code moves its decoder input to CUDA whenever CUDA is available: hide it
        # before anything (transformers included) can initialize the driver.
        torch.cuda.is_available = lambda: False
    import transformers, sentencepiece  # noqa: F401,E401
    report.update(torch=torch.__version__, transformers=transformers.__version__)
except Exception as e:  # noqa: BLE001
    report["error"] = f"import: {e}"
    print(json.dumps(report)); sys.exit(3)
sys.path.insert(0, code_dir)
try:
    from metricx24 import models
except Exception as e:  # noqa: BLE001
    report["error"] = f"metricx24: {e}"
    print(json.dumps(report)); sys.exit(3)
if mode == "check":
    report["ok"] = True
    print(json.dumps(report)); sys.exit(0)
torch.set_num_threads(max(1, int(threads)))
tok = transformers.AutoTokenizer.from_pretrained(tok_dir)
model = models.MT5ForRegression.from_pretrained(model_dir, torch_dtype="auto")
dev = torch.device("cuda" if device == "cuda" else "cpu")
model.to(dev); model.eval()
with open(inp, encoding="utf-8") as f:
    rows = [json.loads(l) for l in f if l.strip()]
with open(outp, "w", encoding="utf-8") as out, torch.no_grad():
    for r in rows:
        text = "source: " + r["src"] + " candidate: " + r["hyp"] + " reference: " + r["ref"]
        enc = tok(text, max_length=int(max_len), truncation=True, padding=False)
        ids = torch.tensor([enc["input_ids"][:-1]], device=dev)
        mask = torch.tensor([enc["attention_mask"][:-1]], device=dev)
        pred = model(input_ids=ids, attention_mask=mask).predictions
        out.write(json.dumps({"system": r["system"], "id": r["id"], "direction": r.get("direction"), "score": float(pred.reshape(-1)[0])}) + "\n")
report["ok"] = True; report["lines"] = len(rows)
print(json.dumps(report))
'''


def code_dir(path: str) -> str | None:
    """A folder holding the ``metricx24`` package (extracting a ``metricx-*.tar.gz`` found in ``path``)."""
    if os.path.isdir(os.path.join(path, "metricx24")):
        return path
    if not os.path.isdir(path):
        return None
    for name in sorted(os.listdir(path)):
        sub = os.path.join(path, name)
        if os.path.isdir(os.path.join(sub, "metricx24")):
            return sub
    for name in sorted(os.listdir(path)):
        if name.startswith("metricx") and name.endswith(".tar.gz"):
            with tarfile.open(os.path.join(path, name)) as tar:
                members = [m for m in tar.getmembers() if m.isfile() and ".." not in m.name.split("/") and not m.name.startswith("/")]
                try:
                    tar.extractall(path, members=members, filter="data")  # noqa: S202 - members filtered above
                except TypeError:  # Python without extraction filters (members are filtered above)
                    tar.extractall(path, members=members)  # noqa: S202
            return code_dir(path)
    return None


def files_status(model: str, tokenizer: str, code: str) -> dict:
    return {
        "model": os.path.isfile(os.path.join(model, "config.json")) and os.path.isfile(os.path.join(model, "pytorch_model.bin")),
        "tokenizer": os.path.isfile(os.path.join(tokenizer, "spiece.model")),
        "code": code_dir(code) is not None,
    }


def run_worker(python: str, mode: str, code: str, model: str, tokenizer: str, device: str, inp: str = "-", outp: str = "-", max_len: int = 1536,
               threads: int = 4) -> tuple[int, dict]:
    env = dict(os.environ)
    if device != "cuda":
        env["CUDA_VISIBLE_DEVICES"] = "-1"  # an empty value is the same as unset on Windows
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    proc = subprocess.run([python, "-c", WORKER, mode, code, model, tokenizer, device, str(max_len), inp, outp, str(threads)],
                          capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    last = (proc.stdout.strip().splitlines() or ["{}"])[-1]
    try:
        info = json.loads(last)
    except json.JSONDecodeError:
        info = {"error": (proc.stderr or proc.stdout)[-2000:]}
    if proc.returncode != 0 and "error" not in info:
        info["error"] = (proc.stderr or "")[-2000:]
    return proc.returncode, info


def detect(python: str, model: str, tokenizer: str, code: str) -> dict:
    status = {"files": files_status(model, tokenizer, code), "python": python}
    if not all(status["files"].values()):
        status["ready"] = False
        status["missing"] = [k for k, v in status["files"].items() if not v]
        return status
    rc, info = run_worker(python, "check", code_dir(code), model, tokenizer, "cpu")
    status.update(interpreter=info, ready=rc == 0)
    if rc != 0:
        status["missing"] = ["torch / transformers / sentencepiece or metricx24 import: " + str(info.get("error"))]
    return status


def flatten_pairs(paths: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                p = json.loads(line)
                direction = f"{p.get('source')}-{p.get('target')}"
                for system, hyp in (p.get("systems") or {}).items():
                    rows.append({"system": system, "id": p["id"], "direction": direction, "src": p["src"], "ref": p["ref"], "hyp": hyp or ""})
    return rows


def main(argv=None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover
        pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    ap.add_argument("--code", default=DEFAULT_CODE)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--max-input-length", type=int, default=1536)
    ap.add_argument("--threads", type=int, default=4, help="CPU threads of torch (default 4)")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--pairs", action="append", default=[])
    ap.add_argument("--out")
    args = ap.parse_args(argv)
    status = detect(args.python, args.model, args.tokenizer, args.code)
    if args.check or not status["ready"]:
        print(json.dumps(status, indent=2))
        if not status["ready"]:
            print("MetricX is not ready: see the header of score/metricx.py for the install.", file=sys.stderr)
            return EXIT_MISSING
        return 0
    import tempfile  # noqa: PLC0415

    if args.self_test:
        rows = [
            {"system": "good", "id": "t1", "direction": "en-de", "src": "The cat sits on the mat.", "ref": "Die Katze sitzt auf der Matte.", "hyp": "Die Katze sitzt auf der Matte."},
            {"system": "bad", "id": "t1", "direction": "en-de", "src": "The cat sits on the mat.", "ref": "Die Katze sitzt auf der Matte.", "hyp": "Der Hund schläft im Auto."},
        ]
    else:
        if not args.pairs or not args.out:
            ap.error("give --pairs and --out (or --check / --self-test)")
        rows = flatten_pairs(args.pairs)
    with tempfile.TemporaryDirectory() as tmp:
        inp, outp = os.path.join(tmp, "in.jsonl"), os.path.join(tmp, "out.jsonl")
        with open(inp, "w", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        rc, info = run_worker(args.python, "predict", code_dir(args.code), args.model, args.tokenizer, args.device, inp, outp, args.max_input_length, args.threads)
        if rc != 0:
            print(json.dumps(info, indent=2), file=sys.stderr)
            return 1
        with open(outp, encoding="utf-8") as f:
            scored = [json.loads(l) for l in f if l.strip()]
    if args.self_test:
        good = next(r["score"] for r in scored if r["system"] == "good")
        bad = next(r["score"] for r in scored if r["system"] == "bad")
        print(json.dumps({"good": good, "bad": bad, "passed": good < bad}))
        return 0 if good < bad else 1
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in scored)
    print(f"{len(scored)} scores -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
