# Local speech package (Sigil)

When you use **ONNX (local)** text-to-speech and choose **Folder on this PC**, Sigil loads a **speech package**: a directory that contains everything the in-process engine needs, not a single arbitrary `.onnx` file.

## Format: `speech_package_v1` (current)

The supported layout matches what the app downloads from a Hugging Face **model repository** for this backend: a small JSON manifest, one ONNX graph, and a voice data archive.

| File | Role |
|------|------|
| `config.json` | Declares model type (`ONNX1` / `ONNX2`), filenames for the ONNX model and voice data, optional speed priors and voice aliases. |
| ONNX file | Name given by `model_file` in `config.json`. |
| Voices archive | Name given by `voices` in `config.json` (NPZ with per-voice style data). |

Hub download and a local folder use the **same** layout, so you can copy an extracted repo or point at a cache directory that contains these three pieces.

## Hub download

With **Model from → Hugging Face repo**, the first run may download and cache that same package automatically.

## Other formats

Support for additional on-disk layouts (for example lighter single-file or two-file voice kits) may be added later. Unsupported formats will be rejected with a clear error.
