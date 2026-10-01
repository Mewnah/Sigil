# Voice-changer families: export spike (P3-H)

Maintainer tooling behind [ADR 0006](../../docs/adr/0006-vc-zero-shot-singing.md). It answers one question
per family: can the upstream PyTorch model run inside Sigil's own ONNX Runtime engine with the same
output, and how fast? Nothing here is needed to build or run Sigil.

| Folder | Family | Upstream (vendored unchanged under `upstream/`) |
| --- | --- | --- |
| `seed_vc/` | Sigil-ZS: Seed-VC real-time zero-shot (`seed-uvit-tat-xlsr-tiny`) | Plachtaa/seed-vc @51383efd |
| `ddsp_svc/` | DDSP-SVC 6.3 singing (DDSP + rectified flow, 44.1 kHz) | yxlllc/DDSP-SVC branch 6.3 @d6dd52fc |
| `meanvc/` | MeanVC2 CPU zero-shot (40 ms model) | ASLP-lab/MeanVC2 @13acf84c |

Every vendored file starts with a one-line provenance header; everything below it is upstream's.

Per family:
- `export.py` loads the upstream model the way upstream's own real-time code does and writes the ONNX
  graphs to `%LOCALAPPDATA%/sigil-dev-cache/vc-families/<family>/`. The graphs carry weights, so
  they are never committed.
- `sampler.py` is the reference streaming loop on those graphs, with numpy between graph calls. It
  is the spec a Rust port is checked against.
- `parity.py` runs upstream PyTorch and the sampler on the same audio and the same recorded random
  draws, then reports SNR per stage. It also measures per-call and per-tick cost. It writes
  `reports/<family>_<ep>.json`; those reports are checked in, since they are the ADR's evidence.
  Every section of a report (parity, cost, profile) records the sha256 of each graph it measured
  (`graphs`).

Shared pieces:
- `common.py` holds the session builder (matching the engine: sequential, no spinning, 1–2 threads,
  pinned dimensions), IOBinding and CUDA-graph timing, ORT's profiler (`profile_session_call`), the
  comparisons, recorded randomness, graph hashes and report writing.
- `adr_tables.py` builds the ADR's evidence tables from the reports (`--write` splices them into
  ADR 0006), so every table number comes from a checked-in report.
- `test_families.py` runs fast pytest checks of the Sigil-written parts. It needs no checkpoints.

## Inputs

Each input is fetched once into `%LOCALAPPDATA%/sigil-dev-cache`, one file at a time. No script
here downloads anything.

- **`seed-vc/`**
  - `Plachta_Seed-VC/DiT_uvit_tat_xlsr_ema.pth`
  - `CosyVoice-300M/hift.pt`
  - `wav2vec2-xls-r-300m/{config.json,preprocessor_config.json,pytorch_model.bin}`
- **`seed-vc-src/seed-vc-main/campplus_cn_common.bin`**, from the seed-vc source archive.
- **`ddsp-svc/openvpi/`**: `pc_nsf_hifigan_44.1k_hop512_128bin_2025.02/{model.ckpt,config.json}`,
  unzipped from the openvpi release.
  - There is no public DDSP-SVC 6.3 checkpoint, so the DDSP and flow networks use seeded random
    weights.
  - The reference units come from `applio-hf/embedders/contentvec/pytorch_model.bin`, and Sigil's
    own units come from the `vc/embedder-contentvec` ONNX (P3-A).
- **`meanvc2/ASLP-lab_MeanVC2/`**
  - `meanvc2_40ms_40ms.safetensors`, `fastu2pp_80ms.pt`, `vocos.pt`.
  - The speaker encoder is not included. It is WavLM-Large plus a fine-tuned ECAPA-TDNN, and the
    fine-tuned weights are a Google Drive download, so parity uses a fixed embedding.
- **Speech samples**: `models/transcribe/samples/{jfk,ja-short}.wav`.

## Running (repo root)

```bash
pip install -r scripts/vc-families/requirements.txt        # one package at a time
python scripts/vc-families/seed_vc/export.py                # likewise ddsp_svc/, meanvc/ (system Python exports)
CUDA=$LOCALAPPDATA/sigil-dev-cache/venv-ort128-cuda/Scripts/python   # ORT 1.28, the engine's version
$CUDA scripts/vc-families/seed_vc/parity.py                 # CPU EP: parity + cost -> reports/seed_vc_cpu.json
$CUDA scripts/vc-families/seed_vc/parity.py --ep cuda
$LOCALAPPDATA/sigil-dev-cache/venv-ort128-dml/Scripts/python  scripts/vc-families/seed_vc/parity.py --ep dml --variants fp32
$LOCALAPPDATA/sigil-dev-cache/venv-ort128-dml/Scripts/python  scripts/vc-families/ddsp_svc/parity.py --ep dml --profile
python scripts/vc-families/adr_tables.py --write            # refresh ADR 0006's tables from the reports
python -m pytest scripts/vc-families/test_families.py -q
```

CPU evidence is measured on the CPU EP of ORT 1.28 (the `venv-ort128-cuda` build), the engine's
runtime; a report measured on another ORT version carries `env.warning`.

`--no-parity`, `--no-time` (so also `--no-parity --no-time --profile`, a profile alone) and a
`--variants` subset merge into the existing report instead of replacing it. When the merged sections were measured on different exports of a
graph (their `graphs` hashes differ), the report lists them under `graph_mismatch` and the script
warns: rerun those sections.

## Conventions

- **Gate.** An fp32 graph passes at SNR ≥ 35 dB against upstream PyTorch fp32, on every stage and
  on the output audio. A reduced-precision variant must pass on its stage outputs. A non-finite
  value on either side (the reference too) fails the row (`snr_db` "nan", with `non_finite` /
  `ref_non_finite` counts); "inf" means finite and identical.
  - Its audio is compared on log-mel too. That comparison also runs against upstream's own
    reduced-precision mode, where one exists.
  - The reason: a vocoder that predicts its own F0 (Seed-VC's HiFT) turns a tiny mel difference
    into a phase drift. The waveforms then decorrelate while sounding the same.
- **Randomness.** Upstream's random draws (`torch.randn`, `rand`, `Uniform`) are recorded while the
  unchanged upstream code runs, then fed to the graphs as inputs. The engine will feed its own
  counter-based, time-anchored noise instead; Seed-VC's `stream_join` measures what that buys at
  the joins between ticks (anchored DiT noise and an NSF phase anchored at the join).
- **Seed-VC vocoder.** `hift` decodes the whole tick (upstream). `hift_f0` + `hift_nsf` split it
  at upstream's own `forward(x, f0)` seam: the F0 is known before synthesis (so the NSF phase can
  be anchored), and `hift_nsf` decodes only the returned frames plus 12 frames of context, which
  reproduces the whole-tick decode (`sampler.HIFT_MARGIN`). Both paths are gated.
- **DDSP-SVC gate.** `sampler.DdspSvc.tick` applies `SvcDDSP.infer`'s volume gate by default
  (threshold −45 dBFS, the GUI default); the parity rows of the bare model call pass
  `threshold_db=None`, and `chain_audio_volume_gate` checks the gate itself.
- **Timing.** The CPU, the GPU and the dev rig are shared, and another process only ever makes a
  call slower.
  - Each call is timed in 3 blocks, and the block with the lowest median is kept. All three
    medians are recorded.
  - Before each GPU measurement, the scripts wait for GPU utilisation to stay under 40 %. The
    desktop alone holds 25–30 %; another benchmark holds the GPU near 100 %. The load they saw is
    recorded (`timing_conditions`, `gpu_before`).
  - Run one timing job at a time.
- **DirectML.** The scripts never release a DirectML session.
  - onnxruntime-directml 1.24.4 crashed with an access violation when a session ran after another
    one had been released in the same process.
  - Pinned dimensions that DirectML's graph fusion refuses fall back to unpinned ones; the cost and
    profile rows record it (`session.dims_pinned: false`, `session.dims_refused` with the error).
