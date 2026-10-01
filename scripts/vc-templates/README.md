# Voice-changer templates and base assets (maintainer pipeline)

Sigil runs community RVC voices without Python: the importer (Rust,
`src-tauri/src/services/neural_audio/vc/import/`) writes a voice's weights into a flat file, and the
voice engine hands that file to ONNX Runtime, in memory, as the external initializers of
architecture-only ONNX **templates**. This folder produces and verifies those templates (bundled
under `src-tauri/resources/vc-templates/`) and the voice changer's base assets (content encoders,
pitch models, denoisers, benchmark models; model-store entries). The design and the P1 measurements
are in [ADR 0004](../../docs/adr/0004-rvc-templates.md).

| File | Role |
| --- | --- |
| `rvc_arch/models.py`, `modules.py`, `attentions.py`, `commons.py`, `transforms.py` | Upstream RVC model code, vendored unchanged except for package-relative imports (MIT, `rvc_arch/LICENSE`, commit in each header). |
| `rvc_arch/vocoders.py` | Sigil's ports of Applio's RefineGAN and MRF HiFi-GAN generators (Applio's module and parameter names, Applio's arithmetic, randomness as inputs). |
| `applio_arch/` | Applio's model code, vendored unchanged except for imports (MIT, `applio_arch/LICENSE`); only the maintainer cross-check `parity_test.py --upstream-applio` imports it. |
| `rvc_arch/split.py` | Sigil's split graphs: `enc`, `flowdec` per vocoder (partial decode, anchored noise, phase-continuous source), `knn`. |
| `rvc_arch/configs.py` | The template matrix (the RVC `config` per template) and the cells no trainer produces. |
| `export_rvc.py` | Exports one template: graph-only ONNX (fp32 and fp16-inside / fp32-I/O) + `layout.json` (parity "untested", fp32 recommended, except the P1-measured v2-40k-f0-hifigan; bundled templates go through `export_matrix.py`). |
| `export_matrix.py` | Exports the whole matrix with the parity gate (CPU and DirectML), picks each template's fp16 mix, writes `matrix.json` and `reports/matrix_parity.*`. |
| `parity_test.py` | The template parity gate (fp32 <= 1e-4, fp16 <= 5e-3, partial decode == full-decode tail), plus the ungated no-lookahead seam rows (`edge`). |
| `test_templates.py` | pytest checks of the pipeline's own logic (latent window, receptive fields, seam rows, parity bookkeeping, the Applio cross-check's precision and key handling); no downloads, the dev-cache voices only where present. |
| `export_embedders.py` | Content encoders `vc/embedder-*` (ContentVec 768 / 256, SPIN, SPIN v2, ja / ko / zh HuBERT) to ONNX fp32 / fp16 / int8, with their gate and `embedder.json`. |
| `export_pitch.py` | Pitch assets `vc/pitch-*` (SwiftF0, CREPE tiny, the FCPE adapter; RMVPE and PESTO pins). |
| `export_denoise.py` | Denoisers `vc/denoise-*` (GTCRN, DeepFilterNet3, DPDFNet). |
| `export_bench.py` | Benchmark models `bench/*` (DNSMOS P.835, CAM++) and the GPU load graph `bench/gpu-load`. |
| `sigil_layout.py`, `sigil_assets.py` | Shared helpers (the Python reference of the Rust weights writer; staging and checks). |
| `layout_schema.json` | JSON schema of `layout.json` (versions 1 and 2). |
| `reports/` | Measurements written by the scripts (checked in: they are the evidence behind every `parity` field). |

## Setup

Python 3.10+ with `requirements.txt` (the dev rig: torch 2.3.1+cu121, onnx 1.17.0, onnxruntime
1.19.2, transformers 4.39.3, torchaudio, soundfile). GPU parity runs in a *separate* environment
with `onnxruntime-directml==1.19.2` (it replaces the `onnxruntime` module):

```bash
python -m venv --system-site-packages .venv-dml
.venv-dml/Scripts/python -m pip install --no-deps onnxruntime-directml==1.19.2
```

Inputs live in `%LOCALAPPDATA%/sigil-dev-cache/` (fetched once, one file at a time):
`rvc-default-voice/default.pth` (the MIT default voice), `applio-hf/refinegan/f0G32k.pth` (Applio's
RefineGAN 32 kHz pretrain), `applio-hf/embedders/<name>/{config.json,pytorch_model.bin}` (Applio's
embedder checkpoints, `IAHispano/Applio@70ed5638`), `upstream-assets/…` (the pinned pitch, denoiser
and benchmark files; URLs and hashes in `scripts/assets/vc_catalog.py`) and
`models/transcribe/samples/*.wav` (speech for the checks).

## The template matrix

Template id: `<v1|v2>-<sr>k-<f0|nof0>-<vocoder>`; the importer derives the same id from a checkpoint.
22 templates cover every cell a trainer produces:

| | HiFi-GAN (RVC, Applio) | RefineGAN (Applio) | MRF HiFi-GAN (Applio) |
| --- | --- | --- | --- |
| v1, 32 / 40 / 48 kHz | f0 and f0-less (RVC's v1 configs) | not produced (Applio trains v2 only) | not produced |
| v2, 24 / 32 / 40 / 48 kHz | f0 and f0-less | f0 | f0 |

`matrix.json` (bundled next to the templates) lists the templates and every unsupported cell with its
reason (f0-less RefineGAN / MRF: Applio refuses to train them; v1 24 kHz: no trainer config).

Layout versions: `format_version` 1 is the P1 contract (f0 HiFi-GAN). Version 2 marks templates whose
flowdec inputs differ; `features` names them and `streaming.excitation` / `streaming.adain` describe
them:
- **f0-less** (`nof0`): flowdec takes `m_p`, `logs_p`, `z_noise`, `g` only; enc takes `phone` only.
- **MRF HiFi-GAN** (`nsf-harmonics`): 9 sines; `phase` [1, F] (the frame-level NSF phase at every
  window frame, computed by the engine in float64: the HiFi-GAN `phase0` carry applied per frame) and
  `src_noise` [1, S, 9]; fixed initial phases per harmonic (`streaming.excitation.phase_ini`).
- **RefineGAN** (`linear-f0`, `adain-noise-pool`): `phase` [1, F] as MRF; its per-layer AdaIN noise
  comes from the `adain_noise_pool` initializer (external file `noise.f32.bin`: iid N(0, 1) values the
  engine generates once) indexed by `noise_offsets` [1, F] int64 (one pool offset per absolute frame),
  so overlapping ticks see the same noise. Its source path needs ~0.9 s of left context
  (`dec_margin` ~89 frames at 32 kHz: a 64-zero-crossing sinc resampler), its main path only
  `main_margin` (~12). So the flowdec has two windows ending at the same frame: the *source window*
  F = `flow_margin + dec_margin + R` (`f0`, `phase`, `src_noise`, `noise_offsets`) and the *latent
  window* Fz = F - `latent_offset` = `flow_margin + main_margin + R` (`m_p`, `logs_p`, `z_noise`;
  `streaming.latent_offset` = `dec_margin - main_margin`, 0 for every other vocoder). The text encoder
  therefore runs over Fz frames, not F (40 instead of 117 at 32 kHz with R = 4: ContentVec on the
  CPU 48.6 -> 22.8 ms per tick); the cheap source inputs (f0 from the F0 cache, phase, noise) carry
  the long history. The f0 interpolation reads the next frame
  (`excitation.f0_lookahead_frames` = 1): decode one frame more than is returned, or the excitation
  phase steps at tick boundaries when f0 changes (up to ~0.2 cycle at a voicing offset).
  Linear interpolation (f0 and the upsamplers) is computed in closed form, not with ONNX `Resize`,
  whose coordinate arithmetic differs between execution providers.

Every layout also records the right receptive fields (`flow_receptive_right`,
`dec_receptive_right_frames` / `_samples`: frames after a frame that its output depends on, from input
gradients; structural bounds) and `parity_evidence.right_edge` (the measured no-lookahead seam, below);
`matrix.json` repeats them per template so the engine and the autotuner can size lookahead and
crossfades per template.

## Parity gate

`parity_test.py` compares every graph with the whole-utterance reference decode: upstream RVC
(`infer`, PyTorch fp32, fixed noise) for HiFi-GAN, the float64 Applio arithmetic of
`rvc_arch/vocoders.py` for RefineGAN / MRF (Applio's per-sample fp32 phase sum itself drifts by
~1e-5 cycles over seconds, which a trained RefineGAN amplifies to the size of the fp32 gate).
Weights: the MIT default voice (v2-40k-f0-hifigan), Applio's RefineGAN pretrain (v2-32k-f0-refinegan,
rounded to fp16 as exported voices are), and otherwise a donor build (every tensor of the architecture
taken from the nearest real voice where its shape matches; the rest LeCun-normal), so fp16 rows see
trained magnitudes.

The `edge` rows (not gated; T5 evidence) decode windows that end mid-utterance (a voiced end and one at
a voicing offset) with no lookahead and compare their returned frames with the offline decode of the
same frames: the last frames lack the flow's and the decoder's right context. `right_edge.seam_frames`
is how many final frames exceed the variant's tolerance; `export_matrix.py` records the CPU fp32 seam
per template (`parity_evidence.right_edge`, `matrix.json` `seam_frames_fp32`,
`reports/matrix_parity.md`).

`export_matrix.py` runs the gate on the CPU and on DirectML (the GPU class; the least accurate GPU EP
for fp16 in ADR 0004) and writes each variant's `parity` per class into `layout.json`. The fp16
flowdec mix (which parts compute in fp32 while the weights stay fp16) is chosen per template: decoder
stage 0 (the ADR 0004 rule), then stages 0-1, then the flow and (RefineGAN) the main path's ends too;
the first whose DirectML rows stay under 0.8 x the gate ships, otherwise fp16 is marked `fail` on GPUs
and Sigil runs the fp32 flowdec there. `enc` always runs fp32 (fp16 activations cost it accuracy on
every GPU EP). Results: `reports/matrix_parity.md`.

## Commands (repo root)

```bash
# The whole matrix (exports into src-tauri/resources/vc-templates; deterministic)
python scripts/vc-templates/export_matrix.py --dml-python .venv-dml/Scripts/python.exe
python scripts/vc-templates/export_matrix.py --template v2-32k-f0-refinegan --dml-python ...

# One template's parity (CPU; --ep dml in the DirectML environment)
python scripts/vc-templates/parity_test.py --template v2-48k-f0-mrf-hifigan
python scripts/vc-templates/parity_test.py --all

# The pipeline's own checks (fast; the dev-cache voices only where present)
python -m pytest scripts/vc-templates/test_templates.py -q

# A scratch export of any template (parity "untested", fp32 recommended; bundled ones: export_matrix.py)
python scripts/vc-templates/export_rvc.py --template v2-32k-f0-refinegan --out <scratch dir>

# P1 checks, unchanged: the default voice against the bundled v2-40k-f0-hifigan (writes the reference
# bundle the Rust test `default_voice_parity` reads), the Rust weights byte check, the tiny fixture
python scripts/vc-templates/parity_test.py
python scripts/vc-templates/parity_test.py --rust-weights "$LOCALAPPDATA/sigil-dev-cache/vc-parity/v2-40k-f0-hifigan/rust"
python scripts/vc-templates/parity_test.py --write-fixture src-tauri/test-fixtures/vc/tiny

# Base assets (staged in %LOCALAPPDATA%/sigil-dev-cache/vc-assets, the SIGIL_LOCAL_ASSETS layout)
python scripts/vc-templates/export_embedders.py            # then --no-export --ep dml in the DirectML env
python scripts/vc-templates/export_pitch.py
python scripts/vc-templates/export_denoise.py
python scripts/vc-templates/export_bench.py
python scripts/assets/vc_catalog.py                        # rewrite the catalog fragments from the staged files
node scripts/assets/check-local-assets.mjs                 # verify the staged folder against the catalog
```

## Maintainer steps the build agent could not run

- `parity_test.py --upstream-applio`: compares the RefineGAN / MRF reference decodes with Applio's own
  `Synthesizer.infer` (vendored `applio_arch`) on the same weights and noise, both in float64 (Applio's
  model is built under a float64 default dtype, so its internal buffers are float64 too) and with every
  checkpoint key accounted for (weight_norm pairs mapped to Applio's `parametrizations` names; any
  missing or unexpected key stops the check and is named). In fp32 the port's own rounding is already
  2.2e-4 from float64 at 48 kHz, which is why the comparison is float64; each line also prints that
  fp32 floor. Running downloaded project code was not permitted to the agent, so the ports were checked
  line by line against the source, by their real-weight behaviour (the Applio RefineGAN pretrain's key
  names and shapes load with no leftovers) and by `test_templates.py` against a stand-in that runs the
  port through the same harness. Run it once; it must print max |d| <= 1e-4 for every template.
- PESTO: export its streaming model with PESTO's own `realtime/export_onnx.py` (SonyCSLParis/pesto
  `62bc0c97`, `mir-1k_g7.ckpt`) and import it as `vc/pitch-pesto` `model.onnx`.

## Adding a template

1. Add the cell to `rvc_arch/configs.py` (`UPSAMPLING`, or a new vocoder with its port in
   `vocoders.py` and a flowdec wrapper in `split.py`).
2. `python scripts/vc-templates/export_matrix.py --template <id> --dml-python ...`; a template ships
   only when its CPU rows pass; its GPU classes decide the recommendation.
3. Commit the template folder, `matrix.json` and `reports/`.
