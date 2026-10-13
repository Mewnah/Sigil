# Sigil-built assets: catalog and local staging

Sigil-built files (the voice changer's exported content encoders, the FCPE adapter, the benchmark's GPU
load graph, the RVC templates) are built by local scripts and installed through the Models tab's
Import, `SIGIL_LOCAL_ASSETS` or a custom mirror. Their catalog entries start `unpublished` and pinned by
sha256, so any copy is verified; the store never downloads an unpublished entry from its catalog URLs.
The `github-release` URLs in those entries name the files of the `assets-v1` release. Asset hosting was
local only until the user decided to publish (2026-10-13): the files go there with
`scripts/release/publish-assets.ps1` and `node scripts/models/publish-entries.mjs --write` flips the
entries to `published`, which is what gives them a download button (see `scripts/release/README.md`,
"The assets-v1 release"). `vc_catalog.py` keeps the status already in the committed fragments, so
regenerating them never takes it back.

| File | Role |
| --- | --- |
| `vc_catalog.py` | Writes the voice changer's catalog fragments (`vc-bases.json`, `vc-embedders.json`, `vc-denoise.json`, `bench-models.json`) from one declaration table: upstream pins (published) plus the staged Sigil-built files' sizes and hashes (unpublished). `--check` fails when a fragment is stale. |
| `check-local-assets.mjs` | Verifies a `SIGIL_LOCAL_ASSETS` folder (`<dir>/<entry id>/<catalog path>`) against the catalog before Sigil uses it. |

## Workflow

```bash
# 1. Build and check the assets (see scripts/vc-templates/README.md); they are staged in
#    %LOCALAPPDATA%/sigil-dev-cache/vc-assets
python scripts/vc-templates/export_embedders.py
python scripts/vc-templates/export_pitch.py
python scripts/vc-templates/export_denoise.py
python scripts/vc-templates/export_bench.py

# 2. Pin the catalog to what was built
python scripts/assets/vc_catalog.py

# 3. Verify the staged folder, then point Sigil at it
node scripts/assets/check-local-assets.mjs
set SIGIL_LOCAL_ASSETS=%LOCALAPPDATA%\sigil-dev-cache\vc-assets
Sigil.exe --headless fetch vc/embedder-contentvec
```

The exports are deterministic for the pinned tool versions (torch 2.3.1, onnx 1.17.0, onnxruntime
1.19.2, transformers 4.39.3): re-running them reproduces the catalog hashes. A different version
changes the bytes; re-run `vc_catalog.py` (and commit the fragments) after re-exporting.
