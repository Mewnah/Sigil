# RVC templates (maintainer pipeline)

Sigil runs community RVC voices without Python: the importer (Rust,
`src-tauri/src/services/neural_audio/vc/import/`) writes a voice's weights into a flat file, and the
voice engine hands that file to ONNX Runtime, in memory, as the external initializers of
architecture-only ONNX **templates**. This folder produces and verifies those templates. The design
and the measurements are in [ADR 0004](../../docs/adr/0004-rvc-templates.md).

| File | Role |
| --- | --- |
| `rvc_arch/models.py`, `modules.py`, `attentions.py`, `commons.py`, `transforms.py` | Upstream RVC model code, vendored unchanged except for package-relative imports (MIT, `rvc_arch/LICENSE`, commit in each header). |
| `rvc_arch/split.py` | Sigil's split graphs: `enc`, `flowdec` (partial decode, anchored noise, `phase0`), `knn`. |
| `rvc_arch/configs.py` | Template configurations (the RVC `config` list per template). |
| `export_rvc.py` | Exports a template: graph-only ONNX (fp32 and fp16-inside / fp32-I/O) + `layout.json`. |
| `sigil_layout.py` | Python reference of the Rust weights writer; tensor bundles for the Rust tests. |
| `parity_test.py` | The parity gate (fp32 <= 1e-4, fp16 <= 5e-3, partial decode == full-decode tail). |
| `layout_schema.json` | JSON schema of `layout.json`. |

## Setup

Python 3.10+ with `requirements.txt` (the dev rig has torch 2.3.1+cu121 and onnxruntime 1.19.2;
`pip install onnx==1.17.0` was the only addition). Install packages one at a time.

## Commands (repo root)

```bash
# Re-export the bundled template (deterministic; commit the result).
python scripts/vc-templates/export_rvc.py

# Parity on the MIT default voice (PhoenixStormJr/RVC-V2-default-voice default.pth, 55 MB, in
# %LOCALAPPDATA%/sigil-dev-cache/rvc-default-voice/ or --pth); writes the reference bundle the Rust
# acceptance test reads.
python scripts/vc-templates/parity_test.py

# The Rust side: import with Rust, run the templates in ONNX Runtime 1.28 (CPU + DirectML), compare
# with the same upstream reference; leaves the Rust-written slot in vc-parity/<template>/rust.
SIGIL_TEST_MODELS_DIR="$LOCALAPPDATA/sigil-dev-cache" cargo test --manifest-path src-tauri/Cargo.toml default_voice_parity -- --ignored --nocapture

# Byte-compare the Rust importer's weights with the Python writer.
python scripts/vc-templates/parity_test.py --rust-weights "$LOCALAPPDATA/sigil-dev-cache/vc-parity/v2-40k-f0-hifigan/rust"

# Regenerate the tiny test fixture used by `cargo test vc::import` (random weights; commit it).
python scripts/vc-templates/parity_test.py --write-fixture src-tauri/test-fixtures/vc/tiny
```

## Adding a template (P3-A)

1. Add its `config` to `rvc_arch/configs.py` (id `<v1|v2>-<sr>k-<f0|nof0>-<vocoder>`; the importer
   derives the same id from a checkpoint).
2. RefineGAN / MRF vocoders need their generator code vendored (Applio, MIT) and a `_decoder`
   variant in `split.py`; f0-less (nono) voices need a flowdec without the NSF source.
3. Export, then run `parity_test.py --template <id> --pth <a voice of that kind>`; a template ships
   only when every row passes.
4. Measure the GPU EPs with `default_voice_parity` (DirectML, CUDA) and record the result per
   variant and device class in `export_rvc.py`'s `PARITY` table (written to `layout.json` as
   `parity`). Sigil refuses a variant marked `fail` on that device class, and `RECOMMENDED` may not
   point at one.
