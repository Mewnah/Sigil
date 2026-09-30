# MT benchmark and NLLB tooling (P4-D)

Machine-translation measurement for Sigil's translation engines, plus the reference tooling of the
in-process NLLB-200 engine (`src-tauri/src/services/mt_local/nllb.rs`).

| File | What it is |
|---|---|
| `spec.example.json` | Example spec for `Sigil.exe --headless mt-bench` |
| `corpus.sample.jsonl` | 24 casual stream / VRChat lines, English ↔ ja, de, es, zh, ko (240 test lines). A smoke corpus for the harness, with single references written for it; **not** a quality benchmark |
| `nllb_reference.py` | Writes `fixtures/nllb_parity.json`: Python sentencepiece token ids and Python onnxruntime greedy outputs that the Rust tokenizer and decoder must reproduce |
| `fixtures/nllb_parity.json` | 27 lines in 20 languages (normalization edge cases included); checked by the ignored Rust tests `mt_local::tokenizer::tests::matches_python_sentencepiece` and `mt_local::nllb::tests::real_model_matches_the_python_reference` |
| `export_nllb.py` | Builds the unpublished `mt/nllb-200-distilled-1.3b-int8` zip (Optimum export, 8-bit quantization, smoke test, sha256) |

## Running the benchmark

```powershell
# 1. Install the model into a scratch app-data folder (one download, ~0.9 GB)
$env:SIGIL_APP_DATA_DIR = "$env:LOCALAPPDATA\sigil-dev-cache\appdata-p4d"
src-tauri\target\debug\sigil.exe --headless fetch mt/nllb-200-distilled-600m-int8

# 2. Run the spec (reports go to bench-results/mt/<timestamp>/ unless --out is given)
src-tauri\target\debug\sigil.exe --headless mt-bench scripts\mt-bench\spec.example.json --out bench-results\mt\nllb
```

Outputs: `report.json` (per engine and direction: lines, errors, corpus chrF, latency p50 / p95 / mean /
max, source characters per second, the engine configuration), `report.md` (the same as a table) and
`hyps/<engine>.<src>-<tgt>.jsonl` (source, hypothesis, reference per line, for blind comparisons).
`bench-results/` is git-ignored; curated tables go to `docs/benchmarks/`.

### Spec

- `corpus`: JSON lines `{"id"?, "source", "target", "src", "ref" | "refs": [...]}` with canonical
  BCP-47 codes (`en`, `ja`, `zh`, `zh-Hant`, `pt-PT`). Your own corpus (real caption logs) stays local
  and is never committed.
- `directions`: optional filter, e.g. `["en-ja", "ja-en"]`; `limit`: lines per direction; `warmup`:
  untimed lines per engine and direction (default 1).
- Engines:
  - `nllb`: the app's engine in process. `model` (catalog id, from the model store of
    `SIGIL_APP_DATA_DIR`) or `path` (a folder with the ONNX files), `threads` (default 2), `beam`
    (1 = greedy, up to 4).
  - `openai`: any OpenAI-compatible server. `baseUrl` (e.g. `http://127.0.0.1:8080/v1` for a
    `llama-server -m hy-mt2-1.8b-q4_k_m.gguf --port 8080` you started, or Ollama / LM Studio), `model`,
    `prompt` (`generic`, `hy-mt2`, or a template with `{source_lang}`, `{target_lang}`, `{text}`),
    `maxTokens`, `temperature`, `timeoutMs`. A key, when needed, comes from the environment variable
    named by `apiKeyEnv`. `<think>` blocks are removed from answers.
  - `hypotheses`: a JSON-lines file `{"id", "hyp"}` made elsewhere (for example by the Bergamot
    helper, or a competitor), scored without latency.

An engine that fails to open (model not installed, server down) is reported in the table; the others
still run.

### chrF

Corpus-level chrF2 as sacrebleu 2.x computes it by default (`CHRF()`: character n-grams 1–6,
β = 2, whitespace removed, average over the orders present, best reference per line). Failed lines
count as empty hypotheses and appear in `errors`.

## NLLB parity fixtures

```powershell
$m = "$env:LOCALAPPDATA\sigil-dev-cache\appdata-p4d\models\mt\nllb-200-distilled-600m-int8\261c31d"
# sentencepiece.bpe.model of the same revision (4.8 MB):
# https://huggingface.co/Xenova/nllb-200-distilled-600M/resolve/261c31d1a5732c67cdd16d80e8d6088507c7ccea/sentencepiece.bpe.model
python scripts\mt-bench\nllb_reference.py --tokenizer-dir $m --spm <path>\sentencepiece.bpe.model --model-dir $m

# Rust side (model tests are ignored by default)
$env:SIGIL_APP_DATA_DIR = "$env:LOCALAPPDATA\sigil-dev-cache\appdata-p4d"
cargo test --manifest-path src-tauri/Cargo.toml mt_local -- --ignored
```

The slow Python `NllbTokenizer` built from `sentencepiece.bpe.model` alone numbers the FLORES language
tokens one higher than the checkpoint (transformers 4.38–4.39); the script takes the pieces from
sentencepiece and the language id from `tokenizer.json`, which is what the model was exported with.

## NLLB 1.3B

No maintained ONNX export of NLLB-200 1.3B exists on Hugging Face, so its catalog entry is
`unpublished`: build the zip with `export_nllb.py` (needs `optimum[exporters]` and a 5.5 GB checkpoint
download), then install it by local import in Models & Runtimes or through `SIGIL_LOCAL_ASSETS`.
