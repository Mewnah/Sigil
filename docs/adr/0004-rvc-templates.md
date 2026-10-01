# ADR 0004: RVC voices run as Sigil templates with imported weights

- **Status:** accepted (P1 spike; P3 built the template matrix, the production importer and the engine on it: see [P3 outcomes](#p3-outcomes))
- **Date:** 2026-09-29
- **Workstream:** P1-G3
- **Plan decisions:** "How RVC importer templates are produced and shipped", "Parsing untrusted .pth/.zip", "Voice-changer engine architecture (main engine)"
- **Research items:** §4.3#1 (import the checkpoints users actually have), §4.2 (fp16 main engine feasibility)

## Context

Community RVC voices ship as `.pth` checkpoints, usually zipped with an `added_*.index` (research
§4.3#1); almost none are ONNX. Today Sigil only runs bring-your-own fp32 ONNX exports, so most
voices need Python and a manual export first. The main engine also has to beat every other engine
(§ "VC best engine"), which requires:
- partial decode (decoding only the frames returned each tick);
- fixed shapes for CUDA graphs;
- fp16 on the GPU (§4.2);
- time-anchored noise and phase-continuous NSF excitation.

A monolithic w-okada-style generator graph supports none of these.

Constraints:
- No Python on users' PCs.
- No protobuf code at runtime.
- Untrusted pickles never run as code.
- Nothing downloads at build time.
- Templates are bundled, within 30 MB for the whole P3 matrix.
- AGPL, with vendored MIT code keeping its attribution.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A. Export each voice to ONNX with PyTorch at import time | Exact upstream graph | Needs Python and torch (GBs) on every PC; slow; still one monolithic graph | Rejected |
| B. Parse `.pth` with `pthrs`, patch weights into ONNX protobufs in Rust | No Python | A young crate as the production parser; protobuf code at runtime; still monolithic | Rejected (`pthrs` stays a test cross-check, P3-B) |
| C. Architecture-only split **templates** (graphs with external initializers), plus a Rust importer that writes the weights file, fed to ONNX Runtime in memory | No Python, no protobuf; graphs designed for streaming (partial decode, anchored noise, phase0, fixed shapes, fp16); tiny bundled files | One template per architecture; the exporter and parity gate must be maintained | **Chosen**; measured below |
| D. Monolithic full-decode template (RVC `infer` as one graph) | Simplest | Decodes the whole context every tick, which is the competitors' weakness | Kept only as the fallback for a variant that fails parity |

## Decision

**Templates** (`scripts/vc-templates`, bundled under `src-tauri/resources/vc-templates/<id>/`):
- One folder per architecture. The id is `<v1|v2>-<sr>k-<f0|nof0>-<vocoder>`; P1 ships
  `v2-40k-f0-hifigan`.
- Each folder is built by a maintainer pipeline from upstream RVC model code, vendored unchanged
  (MIT, commit `81eed5e8`). Sigil's wrappers are in `rvc_arch/split.py`.
- Weights are random-initialised and weight_norm is removed before export.
- Exports use torch 2.3.1 at opset 17 with `do_constant_folding=False`, so every weight stays a
  named, untransformed initializer; ONNX Runtime folds the transposes at load.
- All length dimensions are dynamic. The engine pins them per preset with free-dimension overrides
  (`with_dimension_override`).
- Three graphs, each exported in fp32 and in fp16-inside with fp32 I/O:

| graph | inputs | output |
| --- | --- | --- |
| `enc` | `phone` f32 [1,T,768], `pitch` i64 [1,T] | `m_p`, `logs_p` f32 [1,192,T] |
| `flowdec` | `m_p`, `logs_p`, `z_noise` f32 [1,192,F]; `g` f32 [1,256,1]; `f0` f32 [1,F] (Hz); `phase0` f32 [1,1] (cycles); `src_noise` f32 [1,F·upp] | `audio` f32 [1,(F−flow_margin−dec_margin)·upp] |
| `knn` | `feats` f32 [T,768]; `index` [N,768] (f32, or f16 in fp16); `index_sq` f32 [N]; `index_rate` f32 [1] | `feats_out` f32 [T,768] |

- **flowdec windows.** Every input covers the same window of F frames:
  - the first `flow_margin` frames (24) feed only the flow;
  - the next `dec_margin` frames (11 at 40 kHz) feed only the decoder; the exporter measures this
    margin from float64 input gradients rather than assuming it;
  - the rest is returned, equal within fp error to the same frames of a full decode.
- **Random parts are inputs,** so the engine can anchor them to absolute time (P3-C1):
  - `z_noise` = N(0,1) × `noise_scale` (0.66666 in RVC);
  - `src_noise` = N(0,1) per output sample;
  - `phase0` is the NSF sine phase at the window's first sample:
    `phase0' = frac(phase0 + Σ wrap(upp·f0/sr))` over the frames that slide out.
  - The NSF excitation is computed in fp32 in every variant.
- **Speaker vector.** `g` is computed in Rust from the speaker table, which makes speaker blending
  free and lets one template serve any speaker count.
- **Retrieval.** `knn` is an exact retrieval blend (RVC's k = 8, weights (1/d²)² normalised,
  `index_rate` mix):
  - A coarse top-64 on `|x|² − 2q·x` (fp16 in the fp16 variant), then an exact fp32 re-rank on
    `|q − x|²`. An fp16 index returns fp32 neighbours at half the VRAM.
  - The index needs at least 64 rows; the engine pads it or turns retrieval off below that.

**Weights.**
- **Storage.** Every template weight is an ONNX external initializer named after its RVC
  parameter, pointing into `weights.f32.bin` at a 64-byte-aligned offset. The `enc` tensors come
  first, so a session gets only the prefix it needs.
- **fp16.** The fp16 graphs point into `weights.f16.bin`: the f32 file cast word by word, so every
  offset halves. Regions that compute in fp32 cast their fp16 weights inside the graph.
- **Mapping.** `layout.json` (schema: `scripts/vc-templates/layout_schema.json`) maps each
  initializer to its checkpoint source (`key`, or `weight_norm` {g, v, dim, folded}), plus the
  margins, the architecture and the precision recommendation.
- **Integrity.** Its `weights_id` (sha256 of the canonical tensor table) is stamped into every graph
  as `sigil.weights_id`. A graph and a layout from different exports are refused at session
  creation.
- **At run time** the engine passes the image with `with_external_initializer_file_in_memory`.
  ONNX Runtime copies each tensor, so the buffer is freed with the builder (tested).
- The speaker table goes to `speakers.f32.bin`.

**Importer** (`src-tauri/src/services/neural_audio/vc/import/`):
1. **Zip guard.**
   - The directory is pre-scanned before the zip crate parses it: ≤ 64 entries for a voice `.zip`,
     ≤ 8192 for the torch format.
   - The pre-scan picks the end record the way the zip crate does: the last signature, zip64 when
     the zip32 total or offset is saturated.
   - It refuses anything that could make the two readers disagree:
     - a record whose comment does not end the file;
     - several disks, or "entries on this disk" ≠ "total";
     - a central directory that does not sit right before the end records at its declared offset.
   - It then walks the central directory headers itself, so the count and duplicate names never
     rely on declared numbers. After parsing, the zip crate's directory offset and entry count must
     match the pre-scan.
   - Names must be safe and unique. No links, no encryption, stored or deflated only.
   - Sizes: ≤ 2 GB in total, compression ratio ≤ 200×.
   - A read enforces the declared size and the CRC.
   - In a voice archive, OS metadata is ignored when choosing the `.pth` and `.index` entries:
     `__MACOSX/`, `._name` AppleDouble files, other hidden entries, `Thumbs.db`, `desktop.ini`.
2. **Allow-list pickle VM** (arena-based, bounded).
   - Allowed globals: OrderedDict, `_rebuild_tensor_v2`, `_rebuild_parameter`,
     Float/Half/BFloat16/Long/Int storages, `_codecs.encode`. Anything else is refused.
   - NEWOBJ/BUILD/INST, the text protocol and out-of-band buffers are refused.
   - Legacy (pre-1.6) pickles and TorchScript archives are refused with their own codes.
   - Tensor views are bounds-checked against their storage.
   - **Memory.** Strings and bytes are shared (`Arc`), so a memo reference, a dict key or a repeated
     use never copies them.
     - Storage keys and locations are at most 64 plain characters.
     - Everything the VM materialises counts against a 256 MB budget: arena slots, payloads,
       container slots, dict indexes and memo entries. This bounds copies such as repeated
       `_codecs.encode` calls.
   - **Metadata.** Only the top-level entries the importer reads are converted to JSON (`config`,
     `sr`, `f0`, `version`, `info`, `vocoder`, `embedder_model`, `model_name`, `author`).
     - They share one budget per checkpoint: 10,000 nodes and 1 MB, counting string bytes.
     - Strings are cut to 4 KB before they are copied.
     - A checkpoint holds at most 16,384 weights.
3. **Metadata.** The weights are the truth:
   - version from `emb_phone` (256 = v1, 768 = v2);
   - f0 from `emb_pitch`;
   - sample rate from `config`;
   - vocoder and embedder from Applio's fields.

   Declared fields that disagree become warnings.
4. **Template match.** The checkpoint's architecture fields must equal the template's.
5. **Weights writer.**
   - weight_norm is folded in f64 with a fixed summation order, byte-identical to the Python
     reference writer.
   - Keys like `parametrizations.weight.original0/1` are accepted.
   - Every checkpoint tensor must be used or on the ignore list (`enc_q.*`), so an unknown
     architecture is refused instead of playing broken.
6. **Smoke decode.** A CPU fp32 decode runs before the voice is accepted. An fp16 smoke decode is
   opt-in (`smoke_fp16`), because ORT's CPU fp16 path takes about 3 s; the engine checks its first
   GPU fp16 output anyway.
7. **Error codes:** `io`, `job`, `not_a_checkpoint`, `legacy_format`, `torchscript`,
   `archive_rejected`, `ambiguous_archive`, `pickle_rejected`, `tensor_invalid`, `missing_tensor`,
   `shape_mismatch`, `non_finite`, `unsupported_model`, `no_template`, `layout_invalid`, `session`,
   `time_limit` (the worker's watchdog), `precision_refused` (a variant that fails parity on the
   target's device class).

**Worker contract** (frozen by P1-S0b): `Sigil.exe --headless vc-import-worker <job.json>`.
- The job file: `{version: 1, source, out_dir, templates_dir?, time_limit_s?, smoke?, smoke_fp16?}`.
- On success it writes `weights.f32.bin`, `speakers.f32.bin` and `sigil-import.json`, and prints
  that import record on stdout.
- **The slot is replaced as one unit.**
  - The three files are written and flushed into the hidden sibling `.<slot>.sigil-import-new`.
  - An existing slot folder is renamed to `.<slot>.sigil-import-old`, and the new folder is renamed
    into place.
  - The old folder's other files (an adopted index, a manifest) are moved over, then it is deleted.
  - A failure before the swap leaves the previous voice untouched. A worker killed during the swap
    leaves the old folder beside the slot; the next import into that slot restores or finishes the
    swap first.
  - The parent runs one import per slot folder at a time.
- On failure it prints `<code>: <message>` on stderr and exits 1.
- It enforces its own time limit (default 120 s), reported as `time_limit`. The watchdog lets a slot
  swap in progress finish first, waiting up to 10 s. P3-B adds the Job Object and memory cap on the
  parent side.
- The parent should always pass `templates_dir`, because it has Tauri's `resource_dir()`. Without
  it the worker probes Tauri's resource folders:
  - Windows: the executable's folder;
  - Linux: `../lib/Sigil`, `$APPDIR/usr/lib/Sigil` or `/usr/lib/Sigil`;
  - macOS: `../Resources`.

  In each it looks under `resources/vc-templates`, then `vc-templates`.

**Precision policy** (from the measurements below).
- `layout.json` records the policy as `graphs.*.recommended`. It records each variant's gate
  result per device class as `graphs.*.variants.*.parity` = {`cpu`, `gpu`: `pass` | `fail`}, where
  `gpu` is the worst measured GPU EP.
- `session::template_session` refuses a variant marked `fail` for the target's class with
  `precision_refused`, unless the target is explicitly `experimental` (parity runs and benchmarks).
  A layout whose `recommended` variant fails parity on that class is refused as `layout_invalid`.
- The P3 autotuner may override the recommendation per PC, but only among the variants that pass on
  that class. Today the only `fail` is `enc.fp16` on GPUs.
- `enc` always runs fp32. It is a small share of the compute, and fp16 activations cost it accuracy
  on every GPU EP.
- `flowdec` runs fp16 on GPUs (CUDA and DirectML) and fp32 on the CPU.
- The shipped fp16 flowdec is mixed precision: decoder stage 0 computes in fp32, from fp16 weights
  cast in the graph.
- `knn` runs fp16 (index storage) on GPUs.

## Evidence

**Setup.**
- Rig H2: Ryzen 7 7800X3D, RTX 4090 (driver 616.56) plus the Radeon iGPU, Windows 11. DirectML
  device 0 is the RTX 4090.
- CUDA runs used the CUDA 13.2 cuBLAS and a cuDNN 9.8 build from DaVinci Resolve as a dev stand-in.
- Voice: the MIT default voice `PhoenixStormJr/RVC-V2-default-voice` `default.pth` (sha256
  `c9d6b0ac…7257fbfccc`, revision `22e4cd2b`): v2, 40k, f0, fp16 weights, 457 tensors, 109
  speakers.
- Input: 200 frames (2 s) of smooth synthetic content features and an f0 contour with unvoiced
  gaps, with fixed noise.
- Reference: upstream `SynthesizerTrnMs768NSFsid.infer` in PyTorch fp32 with weight_norm hooks
  active.
- The partial decode returns the last 20 frames from a 55-frame window.

Values are max |error|, with rel. L2 where it matters.

**fp32 (gate 1e-4):**

| check | Python ORT 1.19 CPU | Rust ORT 1.28 CPU | Rust DirectML | Rust CUDA (ORT default: TF32 on) |
| --- | --- | --- | --- | --- |
| enc m_p / logs_p | 5.9e-6 / 2.3e-6 | 6.2e-6 / 2.3e-6 | 6.2e-6 / 2.6e-6 | 4.1e-3 / 1.3e-3 |
| flowdec, whole window | 8.1e-6 | 8.3e-6 | 7.2e-5 | 1.2e-3 |
| chain enc → flowdec | 7.9e-6 | 8.4e-6 | 7.2e-5 | 1.2e-3 |
| partial vs upstream tail | 3.5e-6 | 3.7e-6 | 1.6e-5 | 5.2e-4 |
| partial vs the graph's own full decode | 1.0e-5 | 1.0e-5 | 5.6e-5 | 4.6e-4 |
| knn (N 2048, T 96) | 5.1e-7 | 4.8e-7 | 7.2e-7 | 4.8e-7 |
| weights file on disk vs in memory | 0.0 | | | |

**fp16 inside, fp32 I/O (gate 5e-3):**
- The CPU columns are ORT's CPU fp16 path: casts inserted around kernels that have no fp16 CPU
  implementation.
- "all-fp16" is the first export.
- "shipped" computes decoder stage 0 in fp32.
- ✗ marks a miss; each miss decides the policy.

| check | Python CPU | Rust CPU | DirectML all-fp16 | DirectML shipped | CUDA all-fp16 | CUDA shipped |
| --- | --- | --- | --- | --- | --- | --- |
| enc m_p (fp16 enc) | 2.1e-3 | 3.1e-3 | 2.1e-2 ✗ | 2.1e-2 ✗ | 6.0e-3 ✗ | 6.0e-3 ✗ |
| flowdec | 4.3e-4 | 4.3e-4 | 9.1e-3 ✗ (rel 1.0e-2) | **2.9e-3** (rel 4.1e-3) | 2.2e-3 (rel 2.1e-3) | **1.3e-3** (rel 1.6e-3) |
| chain, enc fp16 | 1.6e-3 | 2.4e-3 | 1.0e-2 ✗ | 9.4e-3 ✗ | 3.5e-3 | 3.1e-3 |
| chain, enc fp32 → flowdec fp16 (recommended) | 4.6e-4 | 4.6e-4 | 1.1e-2 ✗ | **2.9e-3** | 1.9e-3 | **1.4e-3** |
| partial vs upstream tail | 2.4e-4 | 2.4e-4 | 4.0e-3 | 1.8e-3 | 8.6e-4 | 4.7e-4 |
| partial vs own full decode | 1.2e-4 | 1.2e-4 | 4.7e-3 | 1.3e-3 | 7.3e-4 | 5.8e-4 |
| knn (fp16 index) | 6.0e-7 | 7.2e-7 | 4.8e-7 | 4.8e-7 | 4.8e-7 | 4.8e-7 |

For reference, PyTorch on CUDA (cuDNN, TF32 off) gives:
- upstream RVC's own `.half()` inference: 1.8e-3 (rel. L2 3.1e-3);
- Sigil's fp16 flowdec: 1.6e-3;
- fp16 enc: 7.3e-3, with or without LayerNorm and softmax in fp32.

The text encoder's fp16 error comes from fp16 activations, which is why `enc` stays fp32.

**Where DirectML loses fp16 accuracy.** flowdec fp16 on DirectML, by variant. The variants were
exported with `export_rvc.py --fp32-flow / --fp32-stages` and measured by `default_voice_parity`
with `SIGIL_VC_TEMPLATES_DIR`:

| variant | max abs | rel. L2 | flowdec ms (F = 43) |
| --- | --- | --- | --- |
| all fp16 | 9.1e-3 | 1.0e-2 | 1.7 |
| flow fp32 | 9.0e-3 | 1.0e-2 | 1.8 |
| **decoder stage 0 fp32 (shipped)** | **2.9e-3** | **4.1e-3** | **2.7–2.9** |
| flow + stage 0 fp32 | 3.7e-3 | 4.2e-3 | 2.8 |
| stages 0–1 fp32 | 1.4e-3 | 1.9e-3 | 4.2 |
| fp32 graph | 7.2e-5 | 9.2e-5 | 4.9–9.0 |

- **Stage 0 carries the fp16 error.** It holds the 256-channel resblocks with kernels up to 11, the
  largest fan-in, and about 18% of the decoder's MACs.
- **The fp16 weights stay usable.** They remain fp16 in storage, cast in the graph, so the fp16
  graphs keep a single `weights.f16.bin`.

**Rust weights = Python weights.** `weights.f32.bin` written by the Rust importer is byte-identical
to the Python reference writer, 109,947,264 bytes. This is checked by sha256 in the tiny test and in
`default_voice_parity`, and byte-for-byte by `parity_test.py --rust-weights`.

**Import time.**
- Parse, fold and image: 2.8–3.0 s in a debug build of Sigil, including the sha256 of the 55 MB
  source.
- fp32 CPU smoke decode: 0.34 s. The opt-in fp16 smoke decode: 3.1–3.3 s.
- `Sigil.exe --headless vc-import-worker job.json`: exit 0, 7.2 s wall in a debug build, including
  process start and writing 110 MB with fsync. That is inside T8's 10 s.

**Templates.**
- The six graphs total 0.54 MB (enc 106 KB, flowdec 143–171 KB, knn 6 KB, each ×2), so the P3
  matrix of about 36 folders stays far under 30 MB.
- Export takes 15 s and is deterministic: re-exporting produced byte-identical graphs.

**Per-call time at real-time shapes.** Enc over a 1 s context (T = 100). flowdec returning 80 ms
(F = 43). Averages of 20 calls after 3 warm-ups, with host copies every call (no IOBinding, no CUDA
graphs). The machine was shared with other build jobs, so these are ranges over runs, not a
benchmark (P1-F / P3-F own that):

| EP | enc fp32 | flowdec fp32 | flowdec fp16 (shipped) | enc + flowdec session build |
| --- | --- | --- | --- | --- |
| DirectML, RTX 4090 | 3.4–4.2 ms | 4.9–9.0 ms | 2.7–7.2 ms | 0.6–1.0 s |
| CUDA 13, RTX 4090 | 5.1–6.8 ms | 6.7–8.1 ms | 7.3–7.5 ms | 0.2–0.35 s |
| CPU, 2 threads | 9.9–12.0 ms | 125–141 ms | 156–169 ms (emulated) | 0.17–0.30 s |

CUDA is not faster than DirectML at these small shapes. That is per-call overhead (many small
kernels, host↔device copies), which P3-C2's IOBinding and CUDA graphs target.

**Commands.**
- `python scripts/vc-templates/parity_test.py`
- `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache SIGIL_TEST_CUDA_BIN=<cuda 13 bin\x64> SIGIL_TEST_CUDNN_BIN=<cudnn 9 bin> cargo test --manifest-path src-tauri/Cargo.toml default_voice_parity -- --ignored --nocapture`
- `python scripts/vc-templates/parity_test.py --rust-weights %LOCALAPPDATA%\sigil-dev-cache\vc-parity\v2-40k-f0-hifigan\rust`
- `cargo test --manifest-path src-tauri/Cargo.toml vc::import`: 48 tests, plus the ignored
  acceptance test, on the committed tiny fixture. They cover:
  - the pickle VM: forbidden globals and opcodes, memo blow-up, shared strings as dict keys and
    locations, the memory budget, byte-bounded JSON conversion, truncation, 4,000 random mutations;
  - the zip guard: traversal, bombs, links, duplicates, entry counts from the directory walk,
    forged or appended end records, disk-count mismatches, zip64 records, CRC, 3,000 random
    mutations of the directory and end records;
  - pth views, metadata limited to the used keys, weight_norm fold and layout tamper checks;
  - checkpoints that differ from their template: an extra tensor, each architecture field of the
    config, a reshaped weight, the speaker width, missing weights, an all-zero `weight_v`, NaNs;
  - macOS metadata in voice archives, the atomic slot swap and its recovery, the template probe
    locations, and the parity refusal;
  - end-to-end fp32/fp16 parity and in-memory buffer lifetime;
  - the worker job and archive unwrapping.

## Consequences

- **Every voice kind needs a template.** P3-A exports the v1/v2 × 32/40/48k × f0/nof0 ×
  HiFiGAN/RefineGAN/MRF matrix with the same pipeline and gate:
  - RefineGAN and MRF need Applio's generator code vendored.
  - nof0 voices need a flowdec without the NSF source.
  - The stage-0-fp32 rule has to be re-measured per vocoder.

  A cell that fails parity is refused with `no_template` or `unsupported_model`, never run
  approximately. A variant that fails on one device class ships marked `fail` there and is
  refused there (`precision_refused`). BYO ONNX keeps working.
- **P3-B** productionises the importer:
  - the Job Object parent. Its memory cap is also the backstop for one case the zip pre-scan cannot
    bound: when the zip crate rejects the checked directory on its own per-header checks, it falls
    back to an earlier end record. The post-parse check then refuses the file, but only after the
    zip crate has allocated for that record's entry count, which is at most one entry per byte
    before it;
  - the slot manifest `sigil-rvc-1` built from `sigil-import.json`;
  - `.index` adoption (`SourceInfo.index_entries` already lists the index files);
  - the sha256 import cache (the source hash is recorded);
  - the 200-file fuzz corpus;
  - auto-import of the default voice.
- **P3-C1** builds the SVE core on these contracts:
  - windows of F frames with the margins from `layout.json`;
  - counter-based noise (`z_noise`, `src_noise`) keyed by absolute frame and sample;
  - `phase0` carried in f64;
  - `g` computed from `speakers.f32.bin`;
  - `enc` in fp32.
- **P3-C2** needs the EP ladder to accept in-memory external initializers and dimension overrides.
  `ort_session::build_session` takes no builder hook today; P1 builds template sessions through
  `builder_for_ep` on one explicit EP. P3-C2 also adds IOBinding and CUDA graphs over the pinned
  shapes.
- **TF32 on CUDA.** ONNX Runtime's CUDA EP enables TF32 by default, which puts fp32 graphs 10× over
  the fp32 gate (about 1e-3; still better than fp16). `ort_session::cuda_provider` should set
  `use_tf32 = 0` for voice-changer sessions, or the autotuner should treat CUDA fp32 as
  "TF32 quality".
- **Packaging.** `tauri.conf.json` bundle resources must include `resources/vc-templates/**`
  (P3-S0). The parent passes `templates_dir` from Tauri's `resource_dir()`. Without it, the importer
  probes Tauri's resource folders per platform (see the worker contract), then the
  source tree (debug builds only).
- **Fallbacks.**
  - If in-memory external data misbehaves on some EP, place the graph next to the weights file (the
    ONNX external-data convention); this is verified equal to in-memory (difference 0.0).
  - If a variant fails parity, use a full-decode fp32 template (option D).
  - If the worker process causes packaging problems, parse in-process under the same limits.

## P3 outcomes

Recorded by P3-I0 (2026-09-30) from the P3 workstream results.

**Template matrix (P3-A).** 22 templates cover every cell a trainer produces: RVC v1 at 32/40/48 kHz and v2 at 24/32/40/48 kHz, HiFi-GAN NSF with and without F0, plus Applio's RefineGAN and MRF HiFi-GAN with F0 on v2 (15.5 MiB bundled, budget 30 MB). `resources/vc-templates/matrix.json` lists the 26 cells no trainer produces, with the reason (f0-less RefineGAN / MRF, v1 RefineGAN / MRF, v1 at 24 kHz). The `v2-40k-f0-hifigan` graphs are byte-identical to P1; exports are deterministic. Parity: all 22 pass the CPU gate (worst fp32 flowdec 1.0e-5); partial decode equals the full-decode tail. Table: `scripts/vc-templates/reports/matrix_parity.md`.

- **Layout version 2** (15 templates) lists what a reader must support in `features`: `flowdec-inputs-by-layout`, `nof0` (no pitch or source inputs), `nsf-harmonics` (MRF: 9 harmonics, `src_noise` `[1,S,9]`, fixed `phase_ini`), `adain-noise-pool` (RefineGAN's per-layer noise from a pool stored as the external initializer `noise.f32.bin`, indexed per absolute frame by `noise_offsets` `[1,F]` int64, so overlapping ticks get identical noise) and `linear-f0`. Both Applio vocoders take `phase` `[1,F]`, the phase0 carry at every window frame computed by the engine in f64, so nothing accumulates in fp32 inside the graph. RefineGAN decodes its main path only over `main_margin` (about 11 frames) + R although its source path needs `dec_margin` 85–89 frames (a 64-zero-crossing sinc resampler), about 5× less compute per tick. Linear interpolation is closed-form: ONNX Resize moved one RefineGAN fp32 flowdec on DirectML from 2.7e-5 to 3.3e-4.
- **fp16 per template**, chosen on DirectML with a 0.8 × gate margin (decoder stage 0 in fp32, then stages 0–1, then also the flow and RefineGAN's main-path ends): 16 of 22 run fp16 flowdec on GPUs; RefineGAN (all four), v2-48k HiFi-GAN and v2-24k f0-less run fp32. `enc` stays fp32 everywhere. v1-40k-f0-hifigan's DirectML fp32 whole-window row is 1.02e-4 (2 % over the gate; streaming-shape rows 1.1e-5), so it is marked `fail` for fp32 on GPUs and recommends fp16.
- **References:** Sigil's own RefineGAN / MRF ports (`rvc_arch/vocoders.py`, Applio's parameter names; Applio's RefineGAN 32 kHz pretrain loads with no missing or extra keys) evaluated in float64. The cross-check against Applio's own code (`parity_test.py --upstream-applio`) is a maintainer step still to run. Most cells were measured on "donor" weights; real voices only for v2-40k HiFi-GAN (the MIT default voice) and v2-32k RefineGAN (Applio's pretrain). The single `gpu` parity class is DirectML's; CUDA is more accurate for fp16, so a per-EP class would let CUDA run fp16 on the fp32-only templates.
- **Consumers read layout version 1 only.** The importer refuses version-2 templates as a voice kind it does not run yet (`unsupported_model`, P3-I0), and the engine has no version-2 inputs yet: f0-less, RefineGAN and MRF voices wait on that follow-up (fixtures in `test-fixtures/vc/tiny/v2`).

**Importer (P3-B).** Done as planned: the Job Object parent (3072 MB, 120 s + 20 s grace, BelowNormal), manifest `sigil-rvc-1`, `.index` adoption (`added_*` only), the sha256 import cache, the 200-file fuzz corpus (0 crashes or hangs, also through 200 worker processes) and one-click default-voice import. Training checkpoints, bare / DataParallel state dicts and multi-`.pth` archives are now accepted (the P1 refusal is resolved); the vocoder is detected from the decoder's weight names. The pickle VM gained CPU amplification bounds (cached dict-key hashes, cheap `_codecs.encode`): the corpus runs in 1.3 s instead of 29.5 s. The default voice imports in 7.4 s wall through the isolated debug worker (0.84 s in the release benchmark) with weights byte-identical to the Python writer. Still open: the `pthrs` cross-check (needs the crate as a dev-dependency) and the T8 corpus report on real community voices.

**Engine and GPU layer (P3-C1, P3-C2).** The contracts above hold: windows of F frames with the layout's margins, counter-based noise keyed by absolute frame and sample, phase0 carried exactly like the graph, `g` from `speakers.f32.bin` (with blending), `enc` fp32. Measured on the default voice: partial decode vs the full-decode tail 1.1e-6 (CPU / DirectML fp32), 2.4e-4 (fp16), 1.1e-4 (CUDA fp32 with TF32). `ort_session::build_session_with` (P3-S0) gives template sessions the full EP ladder with in-memory external initializers and dimension overrides. **TF32 (supersedes the consequence above):** the CUDA sessions of the voice engine use TF32 unless `fp32` precision is requested (`GpuLevers::tf32`): turning it off made enc + flowdec 2.3× slower (3.43 → 7.85 ms p50) for an error (about 1e-3) inside the fp16 budget the GPU path already uses; an `fp32` request gets bit-level parity (4.8e-6 vs CPU). CUDA graphs are captured by Sigil in thread-local mode, not through ONNX Runtime's `enable_cuda_graph` (its global capture broke concurrent CUDA work in the process). Numbers: [docs/benchmarks/VC_BENCHMARKS.md](../benchmarks/VC_BENCHMARKS.md).
