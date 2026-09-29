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
- **Unpublished assets** (Sigil-built packs, exported embedders, templates) stay `"unpublished"`.
  Asset hosting is local only for now: they are built by local scripts (for example
  `scripts/native/build-runtime-packs.ps1`) and installed through the Models tab's Import or
  `SIGIL_LOCAL_ASSETS`, and through a custom mirror only for files pinned by `sha256` or `git_oid`.
  Unpinned unpublished packs are import-only in every build. The store never downloads an
  unpublished entry from its catalog URLs.
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
