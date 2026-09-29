# Model and runtime catalog

Every downloadable model, runtime pack, template and benchmark asset Sigil can install is described
by a catalog entry. Entries live in per-area **fragment files** so workstreams never edit a shared
list:

```
src-tauri/model-catalog/<area>.json     e.g. stt.json, tts.json, vc.json, mt.json, runtime.json, bench.json
src-tauri/model-catalog/_schema.json    JSON Schema every fragment must satisfy
```

Status (P0-S0): the directory, schema and conventions exist; the model store that reads the catalog
(download, resume, verify, install, `build.rs` validation and embedding) is P1-A. Adding a fragment
is a data-only change: no Rust edit and no hotspot.

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
- **Unpublished assets** (Sigil-built packs, exported embedders, templates) stay `"unpublished"`
  until the maintainer runs the publish workflow; they are satisfied by local import or
  `SIGIL_LOCAL_ASSETS`.
- **Network gentleness.** Hash lookups for pinning (`scripts/models/pin-catalog.mjs`, P1-A) run
  sequentially with at least 2 s spacing; downloads are one file at a time.
- Installed files land in `app_data/models/<family>/<id>/` with an `installed.json`.

Development overrides: `SIGIL_MODEL_CATALOG_OVERRIDE=<file or dir>` replaces the embedded catalog;
`SIGIL_LOCAL_ASSETS=<dir>` serves files from a local folder instead of the network.
