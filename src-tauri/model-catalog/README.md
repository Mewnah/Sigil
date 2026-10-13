# Model and runtime catalog

Every downloadable model, runtime pack, template and benchmark asset Sigil can install is described
by a catalog entry. Entries live in per-area **fragment files** so workstreams never edit a shared
list:

```
src-tauri/model-catalog/<area>.json     e.g. stt-legacy.json, base.json, runtime-nvidia.json, runtimes.json
src-tauri/model-catalog/_schema.json    JSON Schema every fragment must satisfy
```

`build.rs` validates every fragment (the rules of `_schema.json`, unique ids across fragments, known
`requires`, no entry folder shared with or nested inside another's, and in release builds a `sha256`
or `git_oid` on every file of a `published` entry) and embeds the merged catalog as
`model_store::EMBEDDED_CATALOG_JSON`. At run time the model store
(`src-tauri/src/services/model_store/`, P1-A) validates it again and also refuses dependency cycles. Adding a fragment is a data-only change: no Rust edit and
no hotspot. Files whose name starts with `_` are not fragments.

Current fragments: `base.json` (the tiny `base/test-small` self-test entry), `stt-legacy.json`
(`whisper/<plugin id>` for every Whisper model the Whisper plugin offers, and
`vosk/small-en-us-0.15`), `runtime-nvidia.json` (`runtime/nvidia-cuda13`), `runtimes.json` (the
ONNX Runtime CUDA provider pack and the echo test engine), `runtime-sherpa.json`
(`runtime/sherpa-onnx`), `runtime-transcribe.json` (`runtime/sigil-transcribe-*` and two bench
models) and `runtime-llama.json` (`runtime/llama-server-*`).

P2 fragments:

- `stt-sherpa.json` (P2-B): the sherpa-onnx CPU speech models, all published and hash-pinned, each
  requiring `runtime/sherpa-onnx`: `stt/nemotron-3.5-streaming-560ms-int8` (the default streaming
  model), `stt/nemotron-3.5-streaming-160ms-int8`, `stt/parakeet-ultra-int8`,
  `stt/parakeet-tdt-0.6b-v3-int8`, `stt/qwen3-asr-0.6b-int8`, `stt/sense-voice-small-int8`,
  `stt/moonshine-base-en-int8` and `stt/lid-whisper-base-int8` (spoken-language ID). How each one
  runs (family, languages, capabilities) is `speech_rt::sherpa::catalog`; a unit test keeps the two
  in step.
- `stt-gguf.json` (P2-D): the transcribe.cpp GPU-tier GGUF models, pinned to Hugging Face commits
  with LFS sha256 (`stt/qwen3-asr-1.7b-q8_0`, `-bf16`, `stt/cohere-transcribe-q4_k`,
  `stt/granite-speech-5.0-turboctc-q8_0`, `stt/voxtral-mini-4b-realtime-q4_k_m`), `min_vram_mb`
  from measurements.
- `stt-routing.json` (P2-C): no entries; its top-level `routing` object is the STT routing table,
  read by `speech::router` (and the TS mirror `src/core/services/stt/router.ts`) with
  `include_str!`. `build.rs` only checks that it is an object; it is not part of the embedded
  catalog. Keep `vramMb` in step with `resources::policy`.
- `smart-turn.json` (P2-C): `vad/smart-turn-v3.2` (Smart Turn v3.2 int8, 8.7 MB).
- `tts-local.json` (P2-H): the in-process TTS voices (Kokoro v1.0 and v1.1-zh, Kitten 0.8
  mini / micro / nano, 36 Piper voices), each requiring `runtime/sherpa-onnx`.
- `stt-whisper.json` (P2-A): empty; the `whisper/*` and `vosk/*` entries stay in `stt-legacy.json`
  (ids are forever, so they do not move).

P3 fragments (ids and the in-entry file layout, `model.onnx` or `model.<fp32|fp16|int8>.onnx`, are
frozen in `src/services/neural_audio/vc/assets.rs`; the four P3-A fragments are written by
`scripts/assets/vc_catalog.py`, whose `--check` verifies them):

- `vc-bases.json` (P3-A): pitch models `vc/pitch-rmvpe`, `vc/pitch-swiftf0`, `vc/pitch-crepe-tiny`
  (published, hash-pinned), `vc/pitch-fcpe` (Sigil's adapter around a community export,
  `unpublished`) and `vc/pitch-pesto` (import only, unpinned: its ONNX comes from PESTO's own export
  script), and the MIT default voice `vc/voice-default` (`default.pth` plus its
  `added_IVF511_Flat_nprobe_1_default_v2.index`, adopted by the importer as the sibling index).
- `vc-embedders.json` (P3-A): content encoders
  `vc/embedder-{contentvec,contentvec-256,spin-v2,spin,hubert-ja,hubert-ko,hubert-zh}` (Sigil
  exports, `unpublished`), each holding `model.fp32.onnx`, `model.fp16.onnx` (ContentVec also
  `model.int8.onnx`) and `embedder.json` (the export contract, parity per variant and EP class, the
  recommended variant). Installed together an entry is 575–867 MB; split the precision variants into
  separate entries before publishing.
- `vc-denoise.json` (P3-A): `vc/denoise-gtcrn`, `vc/denoise-dfn3` (the upstream tar.gz, 2 leading
  folders stripped: `enc.onnx`, `erb_dec.onnx`, `df_dec.onnx`, `config.ini`) and `vc/denoise-dpdfnet`.
- `bench-models.json` (P3-A): benchmark-only (kind `bench`) `bench/dnsmos-p835` (`model.onnx` plus
  `p808.onnx`), `bench/campplus` and `bench/gpu-load` (Sigil-built, `unpublished`).
- `runtime-trt-rtx.json` (P3-C2): `runtime/tensorrt-rtx` (a local pack: a TensorRT-RTX plugin
  execution provider plus the runtime DLLs its imports name; `unpublished`, unpinned).

`scripts/assets/check-local-assets.mjs` verifies a `SIGIL_LOCAL_ASSETS` folder against these entries.

Sigil-exported files start `unpublished` (local import or `SIGIL_LOCAL_ASSETS`) and become `published`
once they are on the `assets-v1` release (see "Publishing the Sigil-built assets" below).

P4 fragments (translation):

- `llm-gguf.json` (P4-B): 15 family `mt-llm` GGUF translation models for the bundled llama-server,
  every file pinned to a Hugging Face commit with its LFS sha256: Hy-MT2 1.8B Q4_K_M (the default,
  `mt/hy-mt2-1.8b-q4_k_m`) and 1.25-bit, Hy-MT2 7B and 30B-A3B, MiLMMT-46 4B and 12B, TranslateGemma
  4B and 12B, Seed-X-PPO-7B, LMT-60 4B, and the opt-in specialists CAT-Translate 7B, Shisa V2.1 8B,
  VNTL, GalTransl v4 4B and MADLAD-400 3B. Its top-level `routing.models` table (prompt format, role,
  availability, languages, directions, KV size, recommended sampler) is read by
  `llm_server::catalog`; the footprints of the ids `resources::policy` recommends equal the policy's
  values (a test keeps them in step).
- `mt-nllb.json` (P4-D): `mt/nllb-200-distilled-600m-int8` (published, Xenova's ONNX export at
  commit `261c31d`, 912 MB, LFS-pinned) and `mt/nllb-200-distilled-1.3b-int8` (unpublished until its
  zip is on the release: built by `scripts/mt-bench/export_nllb.py`, pinned, 1.56 GiB).
  Engine: `mt_local::nllb`. Dry-run the pin script with `--only mt/nllb-200-distilled-600m`.
- `mt-bergamot.json` (P4-E): **generated** by `scripts/runtime-packs/bergamot/gen-catalog.mjs` from
  Mozilla's model registry (106 directions to and from English, pinned with the registry's size and
  sha256, published). Ids are `mt-bergamot/<src>_<tgt>` in Sigil's canonical codes (`zh` Simplified,
  `zh-hant` Traditional), family `mt-bergamot`; every entry requires `runtime/bergamot`. Regenerate
  with `gen-catalog.mjs --fetch --write`; `--check` verifies the committed file.
- `runtime-bergamot.json` (P4-E): `runtime/bergamot`, unpublished: build
  `sigil-bergamot-<commit>-s1-cpu.zip` with `scripts/runtime-packs/bergamot/build.ps1` and import it
  (ADR 0008).

P6 fragments (streamer features; created empty by P6-S0, each owned by one workstream):

- `runtime-openvr.json` (P6-B): `runtime/openvr` (`openvr_api.dll` v2.15.6, pinned by git blob id,
  loaded lazily by `vr_overlay`; `SIGIL_OPENVR_DIR` is a development override).
- `tts-qwen3.json` and `runtime-qwentts.json` (P6-E): the Qwen3-TTS talkers
  (`tts/qwen3-tts-{0.6b,1.7b}` with `-base` / `-voicedesign`) and tokenizer, and the Sigil-built
  `runtime/qwentts-server-{vulkan,cuda13,cpu}` packs (unpublished; built by
  `scripts/runtime-packs/qwentts/build.ps1` and imported).
- `tts-ggml-experimental.json` (P6-F): 24 entries: 7 third-party runtime packs (NeMo-Speech.cpp,
  Breeze-TTS-2.cpp, CrispASR official zips, pinned), 3 Sigil-built s2 packs (unpublished,
  `scripts/runtime-packs/s2/build.ps1`) and 14 models (the Magpie tokenizers are Sigil-hosted
  `assets-v1` files rebuilt by `scripts/runtime-packs/nemo-speech/fetch-magpie-tokenizer.ps1`; the
  v2607 checkpoint is NVIDIA's own GGUF on Hugging Face, pinned to its commit).
- `runtime-sbv2.json` (P6-G): not a runtime pack: the Style-Bert-VITS2 Japanese BERT
  (`tts/sbv2-bert-ja`) and the NAIST-JDIC dictionary `text/jpreprocess-naist-jdic`, which is
  shared by text readings (P6-L), Style-Bert-VITS2 and Kokoro's Japanese reader. Never add a second
  entry for that dictionary (catalog ids are unique).
- `tts-cloning.json` (P6-I): `tts/pocket-tts-en-int8` (default) and `tts/pocket-tts-en` (sherpa
  release archives, sha256 from the GitHub release digests, require `runtime/sherpa-onnx`);
  `tts/chatterbox-mtl-v3` (KitsuMate split-speed V3 export at commit 3f44461, 19 files, sha256 from
  its `manifest.json`) and the optional `tts/chatterbox-mtl-v3-gpu-flow` (the fp16 flow step;
  requires the main entry).
- `readings.json` (P6-L): no entries of its own (the dictionary is the shared entry above).
- `firered.json` (P6-L): `vad/firered-aed`, the FireRedVAD AED ONNX plus `cmvn.ark` as `direct`
  GitHub raw sources pinned to commit c30ec49e (`speech::music_gate::MODEL_ID`).

## Fragment format

```json
{
  "$schema": "./_schema.json",
  "entries": [
    {
      "id": "stt/silero-vad-v6.2",
      "family": "stt",
      "kind": "model",
      "version": "6.2",
      "license": "MIT",
      "status": "published",
      "files": [
        {
          "path": "silero_vad.onnx",
          "size": 2327524,
          "sha256": "<64 hex chars>",
          "sources": [
            { "kind": "github-release", "url": "https://github.com/<owner>/<repo>/releases/download/<tag>/silero_vad.onnx", "revision": "<tag>" }
          ]
        }
      ]
    }
  ]
}
```

(The example values are placeholders, not a real pin.)

## Rules

- **Ids are forever.** Never rename or reuse a published id; publish a new version or a new id.
- **Pin everything.** Source URLs point at an immutable revision (commit hash or release tag). Release
  catalogs are fully hash-pinned (`sha256`, or `git_oid` for small non-LFS files). Trust-on-first-use
  is allowed only for non-executable files in development builds.
- **Mirrors in order.** List sources in the order to try (Hugging Face, hf-mirror, ModelScope, Sigil
  GitHub release, custom base URL). The store fails over on connect errors, 403/404/429/5xx, HTML
  bodies or hash mismatch.
- **Licenses are facts.** `license` is metadata for the Models tab; the UI and docs never show
  license warnings.
- **Unpublished assets** (Sigil-built packs, exported embedders, templates) start `"unpublished"`.
  They are built by local scripts (for example `scripts/native/build-runtime-packs.ps1`) and installed
  through the Models tab's Import or `SIGIL_LOCAL_ASSETS`, and through a custom mirror only for files
  pinned by `sha256` or `git_oid`. Unpinned unpublished packs are import-only in every build. The
  store never downloads an unpublished entry from its catalog URLs. Their sources name flat assets of
  the `assets-v1` release; `scripts/models/publish-entries.mjs` flips an entry to `published` after
  the uploaded bytes were proven (next section).
- **Paths.** Catalog `path` values use only `[A-Za-z0-9._/+-]`. Content paths inside archives may use
  any name extraction accepts (spaces, `!`, non-ASCII and so on).
- **One folder per entry.** No entry's folder (`models/<family>/<id rest>`, or `runtimes/<id rest>`
  for runtime packs) may equal or sit inside another entry's: `t/x` next to `t/x/y` is rejected.
- **Network gentleness.** Downloads are one file at a time (at most 2 in parallel, never two from one
  host). Pinning lookups run sequentially with at least 2 s spacing (see "Pinning" below).
- **Hugging Face mirrors are automatic.** Every `huggingface.co` source gets an `hf-mirror.com` twin;
  do not list it.
- **Unknown sizes.** `"size": 0` on an unpinned file means "size unknown" (progress then has no total).

## Install layout

Consumers locate files only with `model_store::installed_path(id)` (and `model_store::ensure(&[ids])`,
awaited on a tokio runtime, to install on demand); never build these paths by hand.

| Kind | Folder under `app_data/` |
|---|---|
| ordinary entries (`model`, `template`, `bench`) | `models/<family>/<id without "<family>/">/<version>/` |
| runtime packs (`kind: runtime`) | `runtimes/<id without "runtime/">/<version>/<variant or "default">/` |
| family `whisper` | the legacy `whisper/` folder the Whisper plugin reads |
| family `vosk` | the legacy `vosk-models/` folder the Vosk plugin reads |

The install record is `models/<family>/<id rest>/installed.json`; it is written atomically after the
files were committed with one rename, so a present record means a complete install. Existing files in
`whisper/` and `vosk-models/` are adopted after a sha256 check, without downloading again (the Models
tab marks them "Found on this PC").

## Mirrors and local assets

- **Mirror preference** (`--mirror`, the Models tab's download source): `auto` (catalog order plus the
  hf-mirror twins), or one family moved first: `huggingface`, `hf-mirror`, `modelscope`,
  `github-release`, `custom`.
- **Custom mirror layout:** `<custom_url>/<entry id>/<file path>`; https anywhere, http only on the
  local network or loopback. Files from any mirror are still checked against the catalog pins.
- **`SIGIL_LOCAL_ASSETS=<dir>`** is searched in this order: `<dir>/<id>/<path>`, then `<dir>/<path>`,
  then `<dir>/<source file name>`.
- **`SIGIL_MODEL_CATALOG_OVERRIDE=<file or folder of fragments>`** replaces the embedded catalog in
  development (validated like the embedded one; http sources only on loopback).

## Pinning

```
node scripts/models/pin-catalog.mjs [--fragment <file>] [--only <id prefix>] [--write] [--hash-by-download]
```

It reads the Hugging Face tree API (branch to commit, LFS sha256 or git blob id) and GitHub release
asset digests. Requests are sequential, at least 2 s apart, back off on 429/5xx, and immutable
responses are cached in `%LOCALAPPDATA%\sigil-dev-cache\pin-catalog`. `--hash-by-download` (opt-in,
needs consent for large files) hashes hosts that publish no digests, such as alphacephei (Vosk).

## Publishing the Sigil-built assets (`assets-v1`)

The Models tab downloads only `published` entries. Sigil-built files list
`https://github.com/Mewnah/Sigil-assets/releases/download/assets-v1/<asset name>` as their source (a release
has no folders: the asset name is one flat file name, never `folder/file`), so turning an
`unpublished` entry into a download button is: stage the files under their asset names, pin them,
upload them, prove the upload, flip the status. Nothing uploads or flips by itself.

```
node scripts/models/assets-v1.mjs plan --staging <dir> [--hash]         # what the catalog expects vs what is staged
node scripts/models/pin-staged.mjs --staging <dir> [--only <prefix>] [--force] [--write]
                                                                         # size + sha256 from the local files, in pin-catalog's format
pwsh scripts/release/publish-assets.ps1 -Staging <dir> [-Verify] [-Execute]   # upload plan; -Execute uploads with gh
node scripts/models/publish-entries.mjs --ids <id,id|all-staged> [--staging <dir>] [--digest-only] [--write]
                                                                         # download + hash every release URL, then status -> published
```

- `pin-staged.mjs` is the offline twin of `pin-catalog.mjs` (which reads a host's metadata and so cannot
  pin a file that is not uploaded yet). The pin of an archive entry is the archive itself, as downloaded.
  It edits only the lines it means to, so hand-formatted fragments keep their layout.
- `publish-entries.mjs` checks one URL at a time (at least 2 s apart); a file that also has an upstream
  source is checked against that host's metadata, and its `assets-v1` URL is then only a mirror. It
  refuses an unpinned file and an entry whose `requires` would stay unpublished (the button would not
  work). `--assets-base http://127.0.0.1:<port>/` points it at a local copy for a dry run.
- The voice changer fragments (`vc-bases`, `vc-embedders`, `vc-denoise`, `bench-models`) are generated by
  `scripts/assets/vc_catalog.py`, which keeps the status already in the committed fragment.
- When you re-upload rebuilt bytes, bump the entry's `version`: the Models tab compares versions to offer
  an update (the store itself also re-installs when an install record no longer matches a pin).
- The full walk-through, with the upload prerequisites, is in `scripts/release/README.md`.

