# Making a GGUF for the local translation server

Sigil's local translation models run in llama.cpp's `llama-server` (the `runtime/llama-server-*` packs,
build **b11146**; see `src-tauri/model-catalog/runtime-llama.json` and ADR 0005). Most models in
`src-tauri/model-catalog/llm-gguf.json` use published GGUF files. This guide covers the other cases:

- a model with no GGUF at all (when the P4 catalog was written, MiLMMT-46 12B v1.0 had none; mradermacher
  has published one since, and the catalog uses it; CAT-Translate 3.3B final still has none);
- checking a community quantization against the original weights before pinning it;
- a model you want to try without adding it to the catalog.

Everything here is manual maintainer work on one PC. Download one model at a time (the dev network
blocks bursts of connections), and never clone model repositories: download the files you need.

## 1. Tools

1. llama.cpp at the **same build as the runtime pack** (b11146, commit
   `7fe450e19305b828c199d602c23a8337aaa1f03b`), so the converter writes metadata the server
   understands:
   - the source archive of tag `b11146` (`https://github.com/ggml-org/llama.cpp/archive/refs/tags/b11146.zip`),
     unpacked to e.g. `C:\dev\llama.cpp-b11146`, for `convert_hf_to_gguf.py` and `gguf-py`;
   - the Windows CPU release zip of the same tag (`llama-b11146-bin-win-cpu-x64.zip`) for
     `llama-quantize.exe` and `llama-perplexity.exe`. Sigil's pack installs only the server from that zip,
     so unpack a second copy for the tools.
2. Python 3.10+ in a virtual environment:

   ```powershell
   py -3.11 -m venv C:\dev\gguf-venv
   C:\dev\gguf-venv\Scripts\Activate.ps1
   pip install -r C:\dev\llama.cpp-b11146\requirements\requirements-convert_hf_to_gguf.txt
   pip install "huggingface_hub[cli]"
   ```

## 2. Download the original weights

Download only the files the converter reads (config, tokenizer and the safetensors shards), with one
worker:

```powershell
$env:HF_HUB_ENABLE_HF_TRANSFER = "0"
huggingface-cli download xiaomi-research/MiLMMT-46-12B-v1.0 `
  --include "*.json" "*.safetensors" "tokenizer*" "*.model" `
  --local-dir C:\dev\hf\MiLMMT-46-12B-v1.0 --max-workers 1
```

Gated repositories need `huggingface-cli login` first. Note the commit hash the download used
(`huggingface-cli download` prints it, and `.cache\huggingface` in the folder records it): the catalog
entry's `version` and its provenance comment refer to it.

## 3. Convert and quantize

```powershell
cd C:\dev\llama.cpp-b11146
# Full-precision text model. Gemma 3 based models (MiLMMT, TranslateGemma) also carry a vision tower;
# the converter writes the text model here, and the vision projector would be a separate --mmproj run
# that Sigil never needs.
python convert_hf_to_gguf.py C:\dev\hf\MiLMMT-46-12B-v1.0 --outtype bf16 --outfile C:\dev\gguf\MiLMMT-46-12B-v1.0-BF16.gguf

# Q4_K_M: the size class the catalog uses for every model (quality close to Q6 at about 60% of the size).
C:\dev\llama-b11146-cpu\llama-quantize.exe C:\dev\gguf\MiLMMT-46-12B-v1.0-BF16.gguf `
  C:\dev\gguf\MiLMMT-46-12B-v1.0.Q4_K_M.gguf Q4_K_M
```

Notes:

- **Encoder-decoder models** (MADLAD-400, other T5 models): the converter supports them, but whether
  `llama-server` serves them must be checked (step 5) before a catalog entry loses its
  `experimental` availability.
- **Models without a chat template** (Seed-X, VNTL, MADLAD) are driven by a raw prompt through
  `/completion`; the prompt builder picks that from the catalog `promptFormat`, not from the GGUF.
- **Very low-bit builds** (Hy-MT2's official 1.25-bit and 2-bit files) use quantization kernels that
  upstream llama.cpp may not have (the 1.25-bit build needs STQ1_0, llama.cpp PR #22836). They cannot be
  reproduced with `llama-quantize`; their catalog entries stay `needs_runtime_update` until the runtime
  pin includes the kernel.

## 4. Check a quantization against the original

For a self-made or community Q4_K_M, compare its token distributions with the BF16 file on a few
hundred lines of the language pairs you care about (plain text, one caption per line):

```powershell
$tools = "C:\dev\llama-b11146-cpu"
# Reference logits from BF16 (slow on the CPU; add -ngl 99 with a GPU build of the tools).
& "$tools\llama-perplexity.exe" -m C:\dev\gguf\MiLMMT-46-12B-v1.0-BF16.gguf -f C:\dev\eval\captions.txt `
  --kl-divergence-base C:\dev\eval\milmmt12b-bf16.kld
# The quantization against it.
& "$tools\llama-perplexity.exe" -m C:\dev\gguf\MiLMMT-46-12B-v1.0.Q4_K_M.gguf `
  --kl-divergence-base C:\dev\eval\milmmt12b-bf16.kld --kl-divergence
```

A healthy Q4_K_M shows a mean KL divergence of a few hundredths and "same top token" above ~90%. A
broken conversion (wrong tokenizer, missing chat template tokens) is far off. Then translate the
same lines through Sigil (step 5) and read them.

## 5. Try it in Sigil

1. Translation → provider "AI model (LLM)" → "Local model server" → Model → **A GGUF file on this PC…**,
   and pick the file. The plan shows the video memory estimate from the file's header; **Test load**
   starts the server with it and the server log shows llama.cpp's messages.
2. For a model that needs its own prompt format, set Translation → prompt format accordingly (the
   catalog's `promptFormat` is used only for catalog models).
3. `cargo test --manifest-path src-tauri/Cargo.toml real_model_end_to_end -- --ignored --nocapture`
   with `SIGIL_TEST_LLM_RUNTIME` (the installed runtime folder) and `SIGIL_TEST_LLM_GGUF` runs the whole
   server path (device pin, readiness, one translation, the measured video memory, stop) outside the UI.

## 6. Add it to the catalog

1. Add an entry to `src-tauri/model-catalog/llm-gguf.json`:
   - family `mt-llm`, kind `model`, an id `mt/<name>-<quant>` (lower case, never renamed later);
   - `min_vram_mb`: the whole server at 4096 tokens of context (weights, KV cache, compute buffers).
     Start from the `llama-server` load log (`<device> model/KV/compute buffer size`) plus 100 MB for
     Vulkan or 400 MB for CUDA; Sigil resizes its claim to those buffers at run time anyway;
   - a `routing.models` row: `promptFormat`, `role` (`specialist` for pair- or style-specific models,
     which the automatic choice never picks), `languages`, `directions` when narrower, `kvBytesPerToken`
     (f16 K+V bytes per token: layers × KV heads × (key + value head size) × 2) and the authors'
     recommended `sampling`.
2. A **published** file (a Hugging Face repo): a `huggingface` source pinned to a commit, then
   `node scripts/models/pin-catalog.mjs --fragment src-tauri/model-catalog/llm-gguf.json --write`
   fills in size and sha256 (sequential, at least 2 s between requests, cached).
3. A **self-made** file that is not hosted anywhere: status `unpublished` with `size: 0` and a
   placeholder `github-release` source, like the Sigil-built runtime packs. It installs through Models &
   Runtimes → Import (or `SIGIL_LOCAL_ASSETS`), and `sha256` / `size` are pinned from the file when it is
   published.
4. Run `cargo test --manifest-path src-tauri/Cargo.toml llm_server` (the catalog tests check that every
   entry has a routing row, a hash-pinned GGUF and a footprint, and that the ids the resource governor
   recommends exist with its footprints).
