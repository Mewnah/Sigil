# P3B-D specs

Content encoder (fixed-shape plan, F0 overlap, content window, running normalisation).

Benchmark specs (`sigil-bench-spec/1`) and spec templates of workstream P3B-D live here (P3B rule R7:
each workstream keeps its specs under `scripts/bench/specs/p3b/<letter>/`; only P3B-A edits
`scripts/bench/profiles/`, the others send profile rows to P3B-I0).

Every timing run holds the GPU timing lock (`pwsh scripts/bench/gpu-lock.ps1 -Acquire <tag>` /
`-Release`, or `run-perf-gate.ps1`, which takes it per run; rule R2), and before/after claims come
from ABBA-interleaved reruns (rule R3). Record each spec's command and numbers in the workstream
report.

## Templates

| template | what |
|---|---|
| `content-stages.template.json` | The engine built directly (`build: direct`: `content_ms` / `f0_ms` / `enc_ms` / `flowdec_ms` per call) on `${EP}`: offline, paced and the H2 load simulator (`h2-load70`). Variables `CROSSFADE_MS` and `LOOKAHEAD_MS` give the DirectML shape cases (10 / 15 ms crossfade, 20 ms lookahead). |
| `content-fidelity.template.json` | One quality run against the offline ceiling (`${CEILING_MODE}`) at the lead offsets `${OFFSETS_JSON}` with the `${ANALYZER}` scorer, audio written for rescoring against one fixed ceiling. `CONTENT_VARIANT` pins the ceiling's content encoder (precision checks). |

Variables both need: `NAME`, `EP`, `CLIP`, `SLOT`, `CONTENT_ENCODER` (empty: the store's variant),
`BLOCK_FRAMES`, `EXTRA_MS`, `PITCH`, `LEVERS_JSON`, `CROSSFADE_MS`, `LOOKAHEAD_MS`; the fidelity
template also `PRESET`, `BLOCK_MS`, `CONTENT_VARIANT`, `CEILING_MODE`, `OFFSETS_JSON`, `ANALYZER`.

## Switches the A/B runs flip (one executable, `-CompareEnv`)

| switch | default | off / on |
|---|---|---|
| `SIGIL_VC_CONTENT_PLAN` | on | `0`: the content encoder's dynamic session (the P3 path) |
| `SIGIL_VC_F0_OVERLAP` | on | `0`: F0 after the content encoder on the worker thread |
| `SIGIL_VC_CONTENT_GRAPH` | on | `0`: the content plan without its CUDA graph (IOBinding only; the graph's speed against its VRAM) |
| `SIGIL_VC_CONTENT_CHECK` | on | `0`: a deferred plan (the production GPU path, no dynamic session) is not cross-checked against a CPU session of its file; only the finite / not-silent test remains (review fix, see "The deferred plan's cross-check") |
| `SIGIL_VC_CONTENT_CONTEXT_MS` | unset | the content context (ms) of the **benchmark's stream instance** only (sessions under the `bench` component prefix and a stream geometry) whose request leaves it automatic; the benchmark spec has no field for it yet, the offline ceiling keeps its own shapes, and a production voice never reads it (review fix: the voice changer's factory also plans with a stream geometry, so the geometry alone was not a bench test) |
| lever `content_norm` | off | `LEVERS_JSON={"content_norm": "on"}`: running GroupNorm statistics (needs the encoder's `stream` variant: `export_embedders.py --stream`) |
| `SIGIL_VC_CONTENT_NORM_ONSET` | on | `0`: with `content_norm`, the running statistics alone also at speech onsets after a long pause (no glide from the window's own statistics) |

`-CompareEnv` sets side B's environment; side A inherits the gate's. To compare *off* (A) against
*on* (B), run the gate with the switch off in its environment and `-CompareEnv` turning it on:

```powershell
$env:SIGIL_VC_CONTENT_PLAN = '0'; $env:SIGIL_VC_F0_OVERLAP = '0'
pwsh scripts/bench/run-perf-gate.ps1 -Exe <exe> -Profile <profile with the content-stages templates> `
    -AppData <app data> -CompareEnv 'SIGIL_VC_CONTENT_PLAN=1;SIGIL_VC_F0_OVERLAP=1' -Rounds 7 `
    -Metrics '*_p50,*_p95,*_p99,cpu_cores,engine_load_ms,vram_engine_mb'
```

## Dev-rig tests (ignored; hold the lock)

```powershell
$env:SIGIL_APP_DATA_DIR = '<app data with the NVIDIA runtime pack>'
$env:SIGIL_TEST_CONTENT_ENCODER_DIR = '<app data>/models/vc/embedder-contentvec/1'
$env:SIGIL_TEST_RMVPE = '<app data>/models/vc/pitch-rmvpe/1/model.onnx'
cargo test --manifest-path src-tauri/Cargo.toml content_plan_dev_rig -- --ignored --nocapture
cargo test --manifest-path src-tauri/Cargo.toml content_and_f0_concurrency_bench -- --ignored --nocapture
cargo test --manifest-path src-tauri/Cargo.toml content_fp16_stream_parity -- --ignored --nocapture
cargo test --manifest-path src-tauri/Cargo.toml content_plan_cpu_reference_dev_rig -- --ignored --nocapture
cargo test --manifest-path src-tauri/Cargo.toml content_cold_load_dev_rig -- --ignored --nocapture
```

`content_plan_dev_rig` measures the dynamic session at the 20 ms- and 10 ms-crossfade windows (first
shape, fresh session, alternating), the plans at 7040 / 6720 / 7360 samples (plus
`SIGIL_TEST_CONTENT_LENGTHS`, e.g. `16000,32000`), their parity against the dynamic session and their
VRAM; `content_and_f0_concurrency_bench` times content alone, RMVPE alone and both from two threads;
`content_fp16_stream_parity` compares fp16 with fp32 ContentVec at stream window lengths;
`content_plan_cpu_reference_dev_rig` calibrates and times the deferred plan's CPU cross-check;
`content_cold_load_dev_rig` times the eager load (dynamic session, then the plan) against the deferred
one (`content::acquire_for_engine`: the plan straight from the export contract) with the load's VRAM
peak.

The CPU feature experiment that decided the running-normalisation design is
`scripts/vc-templates/content_feature_probe.py` (CPU only, ~20 min).

## Results (P3B-D, 2026-10-01/03, RTX 4090 dev rig, ORT 1.28)

Every timing below held the GPU timing lock. The machine was never quiet: phase P6 compiled the
crate concurrently (one or two `rustc.exe`, 20-100 % CPU, logged per run), so absolute numbers are
provisional and only the interleaved (ABBA) before/after ratios are evidence. Executables:
`exe-s0` (P3B-S0's baseline) for the measurement-first runs, `exe-d2` (this workstream's engine
code, release build of 2026-10-01 22:28, sha256 `0fd7b926...`) for the first A/B runs and the
plan-fidelity A/A, and `exe-d3` (the same engine code rebuilt on 2026-10-02 18:01 with the other
workstreams' benchmark changes of that day, sha256 `a0089580...`) for the runs marked "r4",
`exe-d4` (the final engine code except the onset glide, which it has in its first, moment-blend
form; 2026-10-03 00:36, sha256 `0562ef4d...`) for the runs marked "r5", and `exe-d5` (the final
code, 01:26, sha256 `ed2d8a71...`) for the `content_norm` onset-glide batch (C0g). Phase P6 had finished by r5 (no concurrent compiles; the
background load is logged per run). Every A/B compares the two sides inside one executable
(`-CompareEnv` / `-CompareVars`), except the family check (`-CompareExe` against `exe-s0`). App data: a private copy of the
P3 dev app data with the `stream` ContentVec variant added (`model.stream.fp16/fp32.onnx`,
`conv0.f32.bin`). Voice: the default RVC v2 40 kHz voice; RMVPE; Low latency, 40 ms blocks,
64 ms extra context, 20 ms crossfade unless a row says otherwise. The run folders and scripts
named below (`runs/...`, `r4/...`, `r5/...`, `*.profile.json`) are in the workstream's scratchpad
(`p3b/d/` of the P3B session), not in the repository.

### R1 measurement first (exe-s0, BUILD=direct, one run each)

Per-call stage times (ms), JFK clip:

| EP / run | call p50 | call p95 | content p50 | content p95 | F0 p50 | enc p50 | flowdec p50 |
|---|---|---|---|---|---|---|---|
| CUDA offline | 11.94 | 13.61 | 5.18 | 6.49 | 3.30 | 0.80 | 2.17 |
| CUDA paced | 15.66 | 17.37 | 5.85 | 7.17 | 5.15 | 1.19 | 2.91 |
| CUDA h2-load70 | 16.36 | 18.17 | 8.29 | 9.76 | 3.87 | 1.10 | 2.63 |
| DML offline | 10.51 | 13.48 | 3.85 | 5.04 | 2.47 | 0.83 | 2.77 |
| DML paced | 15.38 | 18.41 | 5.52 | 6.90 | 4.07 | 1.24 | 4.05 |
| DML h2-load70 | 14.59 | 16.39 | 5.94 | 7.52 | 3.65 | 1.17 | 3.82 |

Hypothesis confirmed: content is the largest stage of the call on both EPs, idle and under load.

**DirectML shapes (measurement 2, `content_plan_dev_rig`).** The dynamic fp16 session: first run
at 7040 samples 246 ms, then 4.25 ms per call; the same session at 6720 samples (the 10 ms
crossfade): first call 225 ms, then **67 ms per call for good**; a fresh session that starts at
6720: 6.4 ms; alternating 7040 / 6720: 34 ms. So the 34-49 ms are not callers alternating: a
DirectML session runs fast only at the shape it first ran, and slowly at every other one. CUDA
has no such penalty (3.9-4.3 ms at either length; 13 ms alternating). Fixed-shape plans at each
length cost 1.1-1.3 ms (fp16) on DirectML.

**ORT profile of the dynamic ContentVec (measurement 3, `SIGIL_ORT_PROFILE`, 100 runs).** 562 node
dispatches per run on CUDA (430 CUDA + 132 CPU) and 550 on DirectML (418 + 132): the CPU nodes
are the graph's shape arithmetic (Concat 60, Unsqueeze 36, Gather 24, Mul 12 per run), each a host
round trip between GPU kernels. At a fixed shape ORT folds them away and its Level-3 optimizer
fuses SkipLayerNormalization: 391 nodes instead of 561 (see the fusion check below).

**Concurrency (measurement 4, `content_and_f0_concurrency_bench`, live shapes).**

| EP | content | content alone p50 | RMVPE alone p50 | both on two threads p50 | hidden share of the shorter |
|---|---|---|---|---|---|
| CUDA | plan | 1.71 | 7.13 | 7.50 | 78 % |
| CUDA | dynamic | 4.97 | 7.53 | 7.87 | 93 % |
| DirectML | plan | 2.12 | 4.98 | 5.00 | 99 % |
| DirectML | dynamic | 4.74 | 4.96 | 6.82 | 61 % |

Concurrent is close to the longer of the two on both EPs: DirectML's queues do overlap here, so the
COMPUTE-queue idea is not needed (not deferred, dropped).

### The fixed-shape plan (dev rig, `content_plan_dev_rig`; p50 / p95 ms per call)

| EP, precision | dynamic p50 | plan 7040 (20 ms xf) | 6720 (10 ms xf) | 7360 (20 ms lookahead) | 16000 (1 s) | 32000 (2 s) | plan VRAM (7040) | parity vs dynamic (max rel L2) |
|---|---|---|---|---|---|---|---|---|
| CUDA fp16, graph | 4.31 | 1.60 / 1.99 | 1.69 / 2.16 | 1.62 / 2.08 | 2.04 / 2.74 | 2.12 / 2.73 | 328 MB | 3.4e-3 - 1.0e-2 |
| CUDA fp16, no graph | 4.31 | 4.98 / 5.75 | 4.93 / 5.74 | 4.99 / 5.51 | 5.02 / 5.96 | 4.43 / 4.90 | 286 MB | same |
| CUDA fp32, graph | 5.03 | 2.58 / 2.96 | 2.55 / 3.04 | 2.53 / 3.01 | 2.59 / 2.87 | 3.85 / 4.45 | 498 MB | 1.2e-3 - 4.1e-3 |
| DirectML fp16 | 4.25 | 1.12 / 1.65 | 1.29 / 1.76 | 1.17 / 1.64 | 1.37 / 1.91 | 1.79 / 2.61 | 208 MB | 0 (bit-identical) |
| DirectML fp32 | 5.75 | 1.66 / 2.18 | 2.19 / 2.72 | 1.85 / 2.45 | 2.97 / 3.56 | 3.77 / 14.5 | 381 MB | 0 - 2.3e-6 |

- On CUDA the gain is the CUDA graph (the plan without it is no faster than the dynamic session):
  the ~430 kernels replay in one launch. Every CUDA graph passed `check_graph`.
- DirectML static vs dynamic parity is exact in fp16 and <= 2.3e-6 in fp32 (acceptance <= 1e-3).
- The plan build costs 330-450 ms (CUDA) and 650-890 ms (DirectML), once per voice and length.
- A content context costs little at a fixed shape: +0.4 ms (1 s) / +0.5 ms (2 s) on CUDA fp16 and
  +0.25 / +0.7 ms on DirectML fp16, well under the planned +1.5 / +4 ms.

**Cold load (`content_cold_load_dev_rig`, fp16, after the EP's process state exists).** The load
path today (`acquire_for_slot`: dynamic session, a reference run, the plan, then the dynamic session
released) against the deferred path (`acquire_for_engine`: the plan straight from the export
contract):

| EP | eager: to the first planned call | eager: load VRAM peak | deferred: to the first planned call | deferred: load VRAM peak |
|---|---|---|---|---|
| CUDA | 525-557 ms | 566 MB | 297-316 ms | 292 MB |
| DirectML | 1089-1134 ms | 557 MB | 707-725 ms | 340 MB |

The eager path holds the dynamic session and the plan at once during the load (+274 MB on CUDA,
+217 MB on DirectML above the steady state), which alone breaks T3's new "cold-load peak <= steady +
150 MB" row; the engine's load switches to the deferred path with a one-line change in `load.rs`
(requested from P3B-H / P3B-I0).

### The deferred plan's cross-check (review fix)

The production GPU path pools Sigil store variants as *deferred* encoders (`acquire_for_engine`): no
dynamic session exists, so until this fix those plans were checked only for finite, not-silent output
(the "every plan is checked against the dynamic session" wording held for eager encoders only: the
CPU, explicit or legacy files, the kill-switch path). A driver that compiles the pinned-length graph
wrongly (ADR 0006 finding 5 documents the DirectML kind) would have shipped garbled features, with
`SIGIL_VC_CONTENT_PLAN=0` the only way out. Now a plan with no dynamic session to compare with is
checked against a **CPU-EP session of the same file**, run on a helper thread while the plan builds
(`content::ReferenceJob`, 3 intra-op threads, no EP outcome of its own, no VRAM): relative L2 at most
`PLAN_CPU_CHECK_REL_L2` = 0.1, else the plan is refused with a note, the dynamic session serves the
voice, and the refusal is remembered for the process (the broken plan is not built again). A passing
verdict is remembered per (file with size and time stamp, rung, device, window length, levers), so
only the first load of a shape in a process pays; a reference that cannot run (no memory, an op the
CPU lacks) leaves the plan on its finite test with a load warning and records no verdict. Kill switch
`SIGIL_VC_CONTENT_CHECK=0`. The same check now also covers an eager encoder whose dynamic session was
released and that gets a plan of a new shape.

The limit was calibrated with `content_plan_cpu_reference_dev_rig` (RTX 4090, CUDA and DirectML,
ContentVec fp16 and fp32 store variants, 8 speech windows per length, 2026-10-09, `rustc.exe` x0, CPU
11-13 %). The CPU runs the fp16 file in fp32, so an fp16 plan differs from it by fp16's own error; the
two EPs read the same numbers:

| window | fp16 worst rel L2 (CUDA = DirectML) | fp32 worst rel L2 | a 30 % gain error reads |
|---|---|---|---|
| 7040 (0.44 s, 20 ms crossfade) | 1.23e-2 | 1.48e-5 | 3.0e-1 |
| 6720 (10 ms crossfade) | 1.07e-2 | 2.46e-5 | 3.0e-1 |
| 7360 (20 ms lookahead) | 9.61e-3 | 1.62e-5 | 3.0e-1 |
| 16000 (1 s content window) | 6.10e-3 | 7.87e-6 | 3.0e-1 |
| 32000 (2 s content window) | 4.93e-3 | 6.29e-6 | 3.0e-1 |

The limit (0.1) is 8x above the live shapes' noise (and above the 7.6e-2 that fp16 reads against fp32
at 0.1 s, a window no preset uses), and a third of what a 30 % error reads; real compile bugs read
order 1. The same-EP dynamic check of eager encoders keeps its 2e-2 limit: CUDA's fp16 graph differs
from the dynamic session by 3.4e-3 to 1.0e-2 (above), so a tighter limit would refuse healthy CUDA
plans; DirectML's exactness (fp16 bit-identical, fp32 <= 2.3e-6) is asserted at <= 1e-3 for both
precisions by `content_plan_dev_rig`.

Cost (cold, the verdict not on record, a debug test binary so the plan build is slower than the app's,
20 deferred first planned calls: check off against on): median +20 ms, mean +43 ms, range -79 to
+180 ms (noise about 100 ms). The CPU session builds in 230-450 ms and runs a window in 40 ms (fp32) to
75 ms (fp16) at 0.44 s, all hidden behind the plan's own build (330-890 ms), inside T9's +0.3 s.

Tests: `content::tests::deferred_plans_are_cross_checked_against_a_cpu_session` (a correct plan is
accepted and its verdict kept; a plan checked against another graph is refused, falls back to the
dynamic session and is refused again from the record without a new reference; a reference that cannot
run leaves the plan standing with a note and no verdict).

### Level-3 fusion check (T2/T1-W8 part a)

ORT's Level-3 optimized ContentVec graphs (saved for the CPU EP; ORT's transformer fusions are
graph transformers shared by the CPU, CUDA and DirectML EPs) and ORT's transformer optimizer
(`onnxruntime.transformers`, `model_type=bert`, 12 heads, 768 hidden) on the fixed-shape graph:

| graph | nodes | LayerNorm | SkipLayerNorm | BiasGelu / Gelu | FusedMatMul | Attention / MHA | MatMul | Softmax |
|---|---|---|---|---|---|---|---|---|
| dynamic, ORT Level 3 | 561 | 26 | 0 | 12 / 8 | 12 | 0 | 85 | 12 |
| fixed 7040, ORT Level 3 | 391 | 1 | 25 | 12 / 8 | 12 | 0 | 85 | 12 |
| fixed 7040, transformer optimizer | 1055 | 1 | 25 | 12 / 8 | 0 | 0 | 97 | 12 |

LayerNorm and GELU are fused (and at a fixed shape the residual adds fold into SkipLayerNorm);
attention is not, and the transformer optimizer cannot fuse HuBERT's attention pattern either (0
Attention / MultiHeadAttention nodes). The remaining unfused attention is ~15 small kernels per layer
that the CUDA graph replays in one launch and DirectML compiles with the rest of the fixed-shape
graph, so a re-export with a hand-written fused attention is not worth its parity risk: no
re-export (decision recorded; the fused ContentVec engine stays with the deferred TensorRT work).

### Precision: fp16 ContentVec at stream lengths (`content_fp16_stream_parity`)

fp16 against fp32, both as fixed-shape plans, 10 speech windows per length; gate (the export's
fp16 gate): rel L2 <= 1e-2 and mean frame cosine >= 0.9995.

| length | CUDA worst rel L2 / cosine | DirectML worst rel L2 / cosine |
|---|---|---|
| 0.10 s | 4.4e-2 / 0.99903 FAIL | 7.6e-2 / 0.99708 FAIL |
| 0.20 s | 7.0e-3 / 0.99998 pass | 1.9e-2 / 0.99982 FAIL |
| 0.30 s | 9.0e-3 / 0.99996 pass | 1.8e-2 / 0.99979 FAIL |
| 0.44 s (the live window) | 3.5e-3 / 0.99999 pass | 1.2e-2 / 0.99993 FAIL (rel L2 only) |
| 0.50 s | 2.0e-2 / 0.99980 FAIL | 8.3e-3 / 0.99997 pass |
| 1.00 s | 6.9e-3 / 0.99994 pass | 4.3e-2 / 0.99905 FAIL |
| 2.00 s | 4.6e-3 / 0.99999 pass | 4.8e-3 / 0.99998 pass |

fp16 misses the export gate at several stream lengths on both EPs (DirectML at the live 0.44 s
window by rel L2, with the cosine passing). The plan's rule says fp32 on the GPU then; the
downstream effect (fp16 against fp32 content in the engine, scored against the offline ceiling)
is in the fidelity section below, and the default is P3B-I0's call (rule R6).

### CPU feature experiment (T5-P3; `scripts/vc-templates/content_feature_probe.py`)

PyTorch fp32 ContentVec on 8 clips (ja, ko, ru, en, es, fr and JFK; active speech at -20 dBFS),
each with a 1.2 s pause filled with digital zeros or a -60 dBFS noise floor. Mean cosine to the
whole-utterance features (`ret`: the 4 newest frames a tick returns; `enc`: every frame of the
0.44 s `enc` window; `agree`: the same frame from consecutive ticks; `onset`: the returned frames in
the first 200 ms after the pause), -60 dBFS floor (the zero floor reads the same):

| condition | ret | enc | agree | onset cos | onset norm |
|---|---|---|---|---|---|
| 0.44 s window, own statistics (today) | 0.435 | 0.494 | 0.743 | 0.170 | 0.862 |
| 0.44 s, training-slice statistics | 0.441 | 0.504 | 0.755 | 0.218 | 0.849 |
| 0.44 s, whole-utterance statistics (oracle) | 0.443 | 0.505 | 0.756 | 0.213 | 0.848 |
| 1 s window, own statistics | 0.432 | 0.654 | 0.853 | 0.161 | 0.810 |
| 1 s window, training-slice statistics | 0.437 | 0.661 | 0.854 | 0.205 | 0.828 |
| 2 s window, own statistics | 0.462 | 0.711 | 0.855 | 0.269 | 0.829 |
| 2 s window, training-slice statistics | 0.463 | 0.714 | 0.853 | 0.288 | 0.822 |

Per clip: training-slice statistics beat the window's own on `ret` for 8/8 clips (0.44 s) and 8/8
(1 s), and on the onset cosine for 7/7 clips with a measurable onset; 1 s of left context improves
`enc` and `agree` on 8/8 clips but loses `ret` on 5/8 and the onset level on 6/7 (the GroupNorm
muting the plan expected); at 2 s the window's own statistics are already close to the
utterance's and the slice statistics stop helping (3-5/8). Hence the engine A/B at 1 s of content
context, window against running statistics, below.

**T5-P7 (frame grid), decided: dropped.** The newest frame on an end-aligned grid agrees *less*
with the utterance features than on the current grid (cosine 0.223 vs 0.242; better on 1 of 8
clips), while features 5 ms apart keep a cosine of 0.969, so the 5 ms content/F0 misregistration
costs little and the lever is not built.

### Interleaved A/B on exe-d2 (rule R3; `run-perf-gate.ps1 -CompareEnv`)

**Both P3B-D paths (A: `SIGIL_VC_CONTENT_PLAN=0 SIGIL_VC_F0_OVERLAP=0`, B: both on, the default),
9 ABBA pairs, 22:28-22:59** (background: one `rustc.exe`, 12-34 % CPU). Paired median change with its
interval and the gate's verdict:

| EP / run | call p50 | call p95 | call p99 | content p50 | content p95 | F0 p50 (helper's wall time) |
|---|---|---|---|---|---|---|
| CUDA offline | 10.38 -> 6.51 (-37.6 %) | **11.66 -> 7.40** (-36.9 %, -39.5..-34.8) IMPROVED | 12.56 -> 7.74 (-37.5 %) | **4.13 -> 1.98** (-51.3 %) | 5.13 -> 2.63 (-49.5 %) | 3.08 -> 3.40 |
| CUDA paced | 14.00 -> 11.79 (-14.0 %) | 16.62 -> 15.39 (-6.7 %) IMPROVED | 17.92 -> 17.03 NEUTRAL | 4.93 -> 4.23 (-11.3 %) | 6.07 -> 5.83 NEUTRAL | 4.68 -> 6.24 |
| CUDA h2-load70 | 15.83 -> 7.89 (-50.7 %) | **19.23 -> 9.60** (-51.4 %, -52.8..-47.5) IMPROVED | 20.82 -> 11.02 (-49.1 %) | 8.17 -> 2.42 (-70.5 %) | 10.57 -> 3.33 (-68.7 %) | 3.76 -> 3.97 |
| DML offline | 10.63 -> 7.75 (-25.4 %) | **12.17 -> 8.98** (-26.7 %, -31.6..-17.7) IMPROVED | 13.23 -> 13.67 NEUTRAL | **4.07 -> 3.07** (-23.4 %) | 5.27 -> 3.71 (-31.3 %) | 2.54 -> 4.02 |
| DML paced | 15.83 -> 12.53 (-20.6 %) | 18.98 -> 14.69 (-22.6 %) IMPROVED | 19.85 -> 16.57 NEUTRAL | 5.71 -> 4.89 (-14.2 %) | 7.07 -> 5.90 (-16.6 %) | 4.30 -> 6.85 |
| DML h2-load70 | 14.68 -> 10.39 (-26.6 %) | 17.05 -> 11.56 (-32.7 %) IMPROVED | 18.52 -> 12.13 NEUTRAL | 5.80 -> 4.07 (-29.5 %) | 7.38 -> 4.79 (-34.8 %) | 3.82 -> 5.35 |

Blocks over budget: 0 on both sides of every run. Engine load: CUDA 2214 -> 2343 ms, DirectML
2962 -> 3023 ms (both NEUTRAL; +129 / +61 ms, within the +0.3 s budget). VRAM (engine): CUDA
1306 -> 1374 MB (+68 MB, REGRESSED: the plan's fixed buffers and its CUDA graph against the dynamic
session; see the graph row below), DirectML 861 -> 853 MB (-8 MB). CPU: CUDA offline 0.99 -> 1.32
cores (REGRESSED: two threads now wait on CUDA syncs, which spin until P3B-G's blocking waits land),
DirectML -43 to -55 %.

- `F0 p50` with the overlap is the helper's wall time, which runs while the GPU also runs the
  content encoder (it is not on the call's critical path any more; the call times are what count).
- **Paced on CUDA keeps little of the gain** (content 4.23 ms paced against 1.98 offline; enc and
  flowdec +24-28 % REGRESSED in that run only): between 40 ms blocks the RTX 4090 drops its clocks,
  and a block that keeps the GPU busy for less time runs at lower clocks. The offline and
  h2-load70 runs keep the GPU busy and show the full gain. This is a GPU power-state effect for
  P3B-G / P3B-H (clock floor while streaming), not a property of the plan.

**DirectML at a 10 ms crossfade (6720-sample window; A: plan off, B: plan on; the overlap on in
both), 7 pairs, 22:59-23:42:**

| run | call p95 | content p50 | content p95 | blocks over budget |
|---|---|---|---|---|
| offline | 86.05 -> 8.88 (-88.9 %) IMPROVED | **56.1 -> 3.19** (-94.2 %) IMPROVED | 78.5 -> 3.73 | 98.3 % -> 0 |
| paced | 71.33 -> 15.17 (-78.4 %) IMPROVED | 54.5 -> 5.43 (-90.1 %) IMPROVED | 62.6 -> 6.44 | 98.7 % -> 0 |
| h2-load70 | 71.69 -> 11.73 (-83.0 %) IMPROVED | 56.3 -> 4.20 (-92.7 %) IMPROVED | 66.3 -> 4.93 | 99.1 % -> 0 |

### Fidelity: the content plan against the dynamic session (exe-d2, scorer v2)

`content-fidelity.template.json` on 5 clips x 3 lead offsets (1.0 / 1.5 / 2.0 s) per GPU EP (Low
latency, 40 ms blocks, RMVPE, long-shape ceiling), three batches in a row: P0a plan off, P1 plan on,
P0b plan off again (the A/A). Per (clip, offset) pair, 15 pairs per EP (`offset_ab.py`):

| metric | CUDA plain (median) | CUDA plan - plain (median) | pairs better | A/A abs diff (median) | offset spread (median) |
|---|---|---|---|---|---|
| speaker cosine | 0.9282 | +0.0012 | 9/15 | 0.0005 | 0.021 |
| MCD dB | 1.968 | +0.0027 | 4/15 | 0.0004 | 0.032 |
| envelope MCD dB | 2.024 | +0.0046 | 4/15 | 0.0004 | 0.039 |
| F0 RMSE cents | 39.52 | +0.63 | 2/15 | 0.015 | 10.7 |
| V/UV error | 0.060 | 0 | 5/15 (3 equal) | 0 | 0.018 |
| DNSMOS OVRL | 2.683 | +0.027 | 10/15 | 0.008 | 0.157 |
| clicks | 0 | 0 | 15 equal | 0 | 0 |

- **DirectML: bit-identical.** All 15 pairs give the same metrics and the same output samples
  (the fixed-shape fp16 session computes exactly what the dynamic one does).
- **CUDA: not bit-identical, but far inside the noise that matters.** The plan's fixed-shape fp16
  kernels and its CUDA graph differ from the dynamic session by a feature rel L2 of 3e-3 to 1e-2,
  so its differences exceed the run-to-run A/A (CUDA is nearly deterministic here), but they are
  mixed in direction (speaker cosine and DNSMOS a little better, MCD and output F0 RMSE a little
  worse) and 5-17 % of how much the same metrics move when the clip is merely started at another
  offset. For scale against precision, see the fp16 / fp32 rows below.

**Against one fixed ceiling (rescore, r5).** `SIGIL_VC_CONTENT_PLAN` also switches the quality run's
own ceiling instance, so the three batches were rescored (P3B-B's `rescore` mode, scorer v2) against
P0a's ceilings (`runs/rescore-plan`, 15 pairs per EP):

| CUDA metric | plain (median) | plan - plain (median) | pairs better | A/A abs diff (median) | offset spread (median) |
|---|---|---|---|---|---|
| speaker cosine | 0.9282 | -0.0003 | 7/15 | 0.0003 | 0.021 |
| MCD dB | 1.968 | +0.0026 | 3/15 | 0.0002 | 0.032 |
| envelope MCD dB | 2.024 | +0.0049 | 3/15 | 0.0003 | 0.039 |
| F0 RMSE cents | 39.52 | +0.60 | 4/15 | 0.008 | 10.7 |
| DNSMOS OVRL | 2.683 | +0.027 | 10/15 | 0.008 | 0.157 |
| seam flux ratio | 1.264 | -0.009 | 9/15 | 0.003 | - |
| clicks | 0 | 0 | 15 equal | 0 | 0 |

DirectML: all 15 pairs identical again. On CUDA the speaker cosine is inside the A/A; MCD (+0.003
dB), output F0 RMSE (+0.6 cents) and DNSMOS (+0.03) move beyond the A/A but by 1/10 or less of the
offset spread and in mixed directions: the plan's fp16 kernels at a fixed shape are a different but
not a worse rendition (feature rel L2 3e-3 to 1e-2 against the dynamic session, the fp16 noise
level; fp16 against fp32 differs by more, see the precision rows).

### Content context and running normalisation (T5-P3 engine A/B, exe-d3, scorer v2)

`content-fidelity.template.json` on the 8 corpus clips (each with a 1.2 s pause at a -60 dBFS noise
floor) per GPU EP, Low latency, 40 ms blocks, `enc` held at 44 frames (0.44 s). Conditions: **C0**
today (the content window is the context, each window's own statistics); **C1w** 1 s of content
context (`SIGIL_VC_CONTENT_CONTEXT_MS=1000`), window statistics; **C1r** 1 s with running statistics
(`content_norm` on); **C0r** running statistics at today's window. `content_norm` also switches the
quality run's ceiling instance, so every condition is rescored against C0's ceilings
(`runs/rescore-ctx`; the quality runs' own scores are not comparable across that lever). Median paired
difference against C0 and the clips that moved in the better direction (CUDA / DirectML):

| metric | C0 (CUDA / DML) | C1w - C0 | C1r - C0 | C0r - C0 |
|---|---|---|---|---|
| speaker cosine | 0.909 / 0.912 | +0.001 (4/8) / -0.001 (4/8) | +0.008 (5/8) / +0.007 (5/8) | **+0.008 (7/8) / +0.010 (8/8)** |
| MCD dB | 1.941 / 1.941 | **+0.088 (1/8) / +0.084 (1/8)** | +0.102 (1/8) / +0.096 (1/8) | +0.045 (2/8) / +0.038 (2/8) |
| envelope MCD dB | 2.045 / 2.046 | +0.087 (1/8) / +0.088 (1/8) | +0.103 (1/8) / +0.100 (1/8) | +0.044 (2/8) / +0.044 (2/8) |
| F0 RMSE cents | 37.4 / 41.0 | -1.5 (5/8) / -0.4 (4/8) | -0.4 (4/8) / +0.4 (4/8) | -2.0 (6/8) / -2.0 (6/8) |
| V/UV error | 0.063 / 0.058 | +0.004 (2/8) / +0.010 (2/8) | +0.004 (3/8) / +0.008 (2/8) | 0 (3/8) / 0 (3/8) |
| DNSMOS OVRL | 2.761 / 2.643 | -0.111 (2/8) / -0.040 (3/8) | -0.044 (4/8) / +0.029 (4/8) | +0.022 (4/8) / +0.032 (5/8) |
| seam flux ratio | 1.263 / 1.267 | +0.094 (1/8) / +0.092 (2/8) | +0.112 (1/8) / +0.044 (2/8) | +0.009 (3/8) / -0.012 (5/8) |
| artifact events | 30 / 28.5 | -3 (6/8) / -3.5 (6/8) | -6.5 (6/8) / -3 (5/8) | -1.5 (5/8) / -2.5 (5/8) |
| clicks | 0 / 0 | 0 | 0 | 0 |

- **Real left context does not pay at the 0.44 s `enc` window**: 1 s of content context costs
  +0.08-0.10 dB MCD on 7 of 8 clips, with or without running statistics, and DNSMOS falls on CUDA.
  The content window stays the context (`content_context_ms` automatic) and no preset gets a content
  context.
- **Running statistics at today's window (C0r) help the speaker and F0 but not the spectrum**:
  speaker cosine +0.008 / +0.010 on 7/8 and 8/8 clips, F0 RMSE -2 cents (6/8), DNSMOS +0.02 / +0.03,
  but MCD +0.04 dB (2/8 better).
- **Onsets after a long pause (the "GroupNorm vs `enc_p` muting" check).** Level of the first 200 ms
  after >= 1 s of silence relative to the output's speech level (`onset_1s.py`, 14 onsets per EP:
  the 1 s lead and the 1.2 s pause of each clip): median C0 -3.45 dB, C1w -3.73, C1r -4.18, C0r
  -3.16 dB, but with running statistics 2 of the 14 onsets after the 1.2 s pause lose their first
  ~50 ms (ja-long -20.8 -> -54.9 dB, JFK -7.4 -> -48.6 dB; identical on both EPs), which the offline
  ceiling renders as speech (10 ms frames at -19 / -25 / -28 / -15 dB at the JFK onset; C0 keeps the
  first 30 ms, C0r mutes 50 ms). The first onset of each clip (no history yet: the window's own
  statistics) is unaffected. So the onset muting the design attributed to the window's GroupNorm is
  `enc_p`'s (its 0.44 s window is almost all silence there), and the window's own statistics
  partly *mask* it by lifting an onset the window barely holds; running statistics remove that lift.

**The onset glide (r5).** Two attempts to keep running statistics without losing those onsets, both
switched by `SIGIL_VC_CONTENT_NORM_ONSET` (on by default under `content_norm`): while a long silence
(the kind the training slice drops) still fills part of the content window, the statistics move from
the window's own to the running ones as the silence leaves the window, weighted by the share of the
window it fills.

- **C0b, moments blended linearly** (exe-d4): changed almost nothing (JFK onset -54.3 dB, ja-long
  -29.0 dB). A CPU check of the features (`r5/onset_feats.py`, ContentVec fp32 against the stock
  graph at the JFK onset) shows why: speech sits ~40 dB above the noise floor, so even a 5 % share
  of the running (speech) variance dominates a silent window's variance, and the onset's features
  stay as far from the stock graph's as with running statistics alone (cosine 0.26 at +40 ms).
- **C0g, means blended linearly and inverse standard deviations geometrically** (exe-d5, the shipped
  code): the onset's features follow the stock graph (cosine 0.96-0.995 for the first 80 ms, then
  gliding to the running statistics' by the time the pause has left the 0.44 s window).

Rescored against C0's ceilings (`runs/rescore-ctx2`, the C0 / C0r rows reproduce the first rescore
exactly):

| metric | C0 (CUDA / DML) | C0r - C0 | C0g - C0 | C0g - C0r |
|---|---|---|---|---|
| speaker cosine | 0.909 / 0.912 | +0.008 (7/8) / +0.010 (8/8) | **+0.009 (8/8) / +0.012 (6/8)** | +0.001 (5/8) / -0.001 (1/8) |
| MCD dB | 1.941 / 1.941 | +0.045 (2/8) / +0.038 (2/8) | **+0.047 (2/8) / +0.037 (2/8)** | +0.003 (2/8) / +0.001 (3/8) |
| envelope MCD dB | 2.045 / 2.046 | +0.044 (2/8) / +0.044 (2/8) | +0.048 (2/8) / +0.040 (2/8) | +0.007 / +0.001 |
| F0 RMSE cents | 37.4 / 41.0 | -2.0 (6/8) / -2.0 (6/8) | -1.2 (6/8) / -0.6 (6/8) | +0.3 / +0.1 |
| V/UV error | 0.063 / 0.058 | 0 / 0 | -0.002 (5/8) / +0.003 (3/8) | 0 / 0 |
| DNSMOS OVRL | 2.761 / 2.643 | +0.022 (4/8) / +0.032 (5/8) | **+0.053 (5/8) / +0.013 (5/8)** | -0.004 / -0.029 |
| seam flux ratio | 1.263 / 1.267 | +0.009 / -0.012 | +0.062 (3/8) / +0.004 (4/8) | +0.016 / -0.011 |
| artifact events | 30 / 28.5 | -1.5 (5/8) / -2.5 (5/8) | -0.5 (4/8) / -0.5 (4/8) | +1 / +0.5 |
| clicks | 0 | 0 | 0 | 0 |

Onsets after >= 1 s of silence (median of 28; the two after the 1.2 s pause in brackets, CUDA, DML
identical within 0.5 dB): C0 -3.45 dB (ja-long -20.8, JFK -7.4), C0r -3.16 (-54.9, -48.6), C0b -2.78
(-29.0, -54.3), **C0g -2.77 (-23.3, -17.8)**. The glide brings the ja-long onset back to C0's level
and recovers most of the JFK onset (its first 20 ms at -37 / -46 dB against C0's -27 / -23 dB, then
30 ms that C0 also partly loses).

**Decision (T5-P3).** `content_norm` (running statistics with the onset glide) raises speaker
similarity on 14 of 16 clip-EP pairs (+0.009 / +0.012) and lowers output F0 error, at no DNSMOS
cost, but every condition with running statistics is +0.04 dB MCD from the ceiling on 6 of 8 clips,
and the JFK onset still loses ~10 dB of its first 50 ms. It does not meet the flip rule ("running >=
window on >= 70 % of clips"; MCD fails it), so the lever ships **off** (the default stays the
window's own statistics) and is recommended to P3B-I0 as an opt-in "speaker similarity" switch, not
as a default; a blind listening A/B is what would settle it. Its T2 cost is the statistics update
(the first convolution over each 20 ms of new audio, ~0.65 M multiply-adds per 40 ms tick, and
the aggregation, ~0.2 M adds: an estimated 0.1-0.3 ms of CPU per tick, not benchmarked here) and a
`stream` encoder variant in place of the stock one (the same graph with two [1, 512, 1] inputs).
**Content context: no.** Real left context (1 s) costs +0.08-0.10 dB MCD on 7 of 8 clips; the
`content_context_ms` param stays automatic and no preset sets one.

### DirectML with 20 ms of lookahead (exe-d4, r5; A: `SIGIL_VC_CONTENT_PLAN=0`, B: plan on; the overlap on in both)

`d2-stages` profile, `stages-dml` with `LOOKAHEAD_MS=20` (7360-sample content window), 7 ABBA pairs,
00:52-01:27 (background: 0 `rustc.exe` at the start, CPU 17-32 %; pairs 3-7 overlapped this
workstream's own idle-priority builds, a 4-job test build at 01:09-01:11 and a 6-job release build at
01:11-01:26, which load both sides of a pair alike; the effects below are 20x their noise). Paired
median change, 95 % interval, verdict:

| run | call p50 | call p95 | content p50 | content p95 | RTF p99 | underruns / dropped (median) |
|---|---|---|---|---|---|---|
| offline | 41.09 -> 7.95 (-81.2 %) | **46.91 -> 8.74** (-81.8 %, -82.9..-80.8) IMPROVED | **34.39 -> 3.13** (-91.2 %) IMPROVED | 39.06 -> 3.50 (-91.3 %) | 1.27 -> 0.23 | - |
| paced | 41.82 -> 14.25 (-65.3 %) | 47.33 -> 15.94 (-65.5 %) IMPROVED | 35.06 -> 5.42 (-83.9 %) | 39.63 -> 6.37 (-84.0 %) | 1.33 -> 0.42 | 110 / 315 ms -> 0 / 0 |
| h2-load70 | 41.86 -> 10.47 (-74.9 %) | 46.56 -> 11.39 (-75.7 %) IMPROVED | 36.75 -> 4.07 (-88.9 %) | 40.88 -> 4.60 (-88.4 %) | 1.26 -> 0.29 | 127 / 470 ms -> 0 / 0 |

CPU 0.83 -> 0.21 cores (offline), engine VRAM 863 -> 852 MB, engine load 3345 -> 3421 ms (NEUTRAL,
+76 ms). Without the plan a 20 ms lookahead does not run in real time on DirectML (RTF 1.26-1.33,
100+ underruns); with it the content call costs 3.1 ms (acceptance: <= 5 ms). The gate's
"REGRESSED" rows are the overlapped F0 helper's own wall time (`f0_ms`, `overlap_ms`: F0 now
shares the DirectML queue with a content encoder that is busy for a shorter, denser time; not on
the call's critical path) and the paced T1 latency split (`t1_*`), which on side A describes a
stream that dropped 315-470 ms of audio and is not comparable. A second, independent 7-pair run of
the same gate (01:05-01:28, the duplicated r5 queue, see the known issues of the report) replicated
it: call p95 -81.6 / -65.5 / -75.3 % and content p50 -91.3 / -85.6 / -88.8 % (offline / paced /
h2-load70).

### Precision downstream (exe-d4, r5, scorer v2; the ceiling's content encoder pinned to fp32)

`d2-fid-gpuprec` (Low latency, 40 ms blocks, RMVPE; 5 clips per GPU EP, both sides on the
fixed-shape plan) and `d2-fid-cpu-int8` (Streamer, 120 ms blocks, SwiftF0, CPU; 5 clips). Median of
the per-clip differences against the fp32 stream and the clips that moved in the better direction:

| metric | GPU fp16 - fp32, CUDA | GPU fp16 - fp32, DirectML | CPU int8 - fp32 |
|---|---|---|---|
| speaker cosine | +0.0008 (3/5) | +0.0001 (3/5) | **-0.0026 (2/5)** |
| MCD dB | +0.0045 (1/5) | -0.0062 (4/5) | -0.0123 (3/5) |
| envelope MCD dB | +0.0042 (2/5) | -0.0051 (4/5) | -0.0128 (3/5) |
| F0 RMSE cents | +0.46 (2/5) | +0.72 (2/5) | +10.5 (2/5) |
| V/UV error | -0.0014 (3/5) | -0.0014 (4/5) | -0.0058 (4/5) |
| DNSMOS OVRL delta | +0.017 (3/5) | -0.037 (1/5) | -0.041 (2/5) |
| seam flux ratio | +0.020 (2/5) | -0.007 (3/5) | +0.047 (1/5) |
| clicks | 0 | 0 | 0 |

- **GPU fp16 ContentVec: no downstream loss.** Although fp16 misses the export's feature gate at
  several stream lengths (above), the converted voice moves by less than 0.001 speaker cosine and
  0.007 dB MCD in either direction, the size of the plan-vs-plain differences on CUDA and far inside
  the int8 rule's limits (0.002 / 0.03 dB). Recommendation: keep fp16 on the GPU (fp32 would cost
  +170 MB VRAM per plan and +1.0 ms (CUDA) / +0.5 ms (DirectML) per content call for no measurable
  fidelity).
- **CPU int8 ContentVec: the rule fires.** int8 loses 0.0026 speaker cosine against fp32 (3 of 5
  clips worse) and DNSMOS -0.04, with MCD a little better (-0.012 dB); output F0 RMSE moves +10 cents
  (mixed: 2/5 better). Δcosine > 0.002, so by the stated rule `variant_order` should put fp32 first
  on the CPU and leave int8 as a CPU-saver option (`SIGIL_VC_INT8=1`). Recommendation to P3B-I0 (the
  default is its flip, rule R6): fp32 first for the CPU Streamer when its T4 budget allows the
  fp32 encoder's extra CPU time; the CPU cost difference was not re-measured here.
  **Decision (review fix).** The two rules are *waived*, not applied, and the waiver is on record:
  GPU fp16 costs no measurable fidelity (the table above) and fp32 would add ~170 MB VRAM per plan
  (T3 is already a miss) and 0.5-1.0 ms per content call; the CPU rule fired by 0.0006 cosine with
  MCD and V/UV better, and the fp32 encoder's extra CPU time on a Streamer that already misses its RTF
  target is unmeasured, so P3B-I0 kept int8 (ADR 0010) and P7-B measures both. The
  waiver text and these numbers are in the doc comment of `content::variant_order` (next to the
  order it explains) and in ADR 0011; the rules' outcome is one setting away (`precision = fp32` puts
  fp32 first on both classes, `SIGIL_VC_INT8=0` turns the CPU's int8 off).

### Quality's 20 ms lookahead now runs on DirectML (exe-d4, r5, scorer v2)

`d2-fid-la20`: the Quality preset with 20 ms of lookahead, 40 ms blocks, RMVPE, 5 clips per GPU EP
(before P3B-D the factory dropped the lookahead on DirectML, where it cost 35-49 ms per content
call; with the plan it lifts, `content::runs_fixed_shape("dml")`). Medians per EP and the paired
DirectML - CUDA difference:

| metric | CUDA | DirectML | DML - CUDA (median) | DML better |
|---|---|---|---|---|
| speaker cosine | 0.9564 | 0.9571 | -0.0004 | 2/5 |
| MCD dB | 1.742 | 1.743 | +0.0009 | 2/5 |
| envelope MCD dB | 1.857 | 1.862 | -0.0003 | 3/5 |
| F0 RMSE cents | 34.33 | 34.14 | +0.52 | 1/5 |
| V/UV error | 0.048 | 0.049 | -0.0014 | 3/5 |
| DNSMOS OVRL | 2.806 | 2.927 | +0.035 | 3/5 |
| seam flux ratio | 1.141 | 1.126 | -0.059 | 4/5 |
| onset level (dB) | -15.05 | -14.57 | +0.04 | 4/5 |
| clicks | 0 | 0 | 0 | - |

DirectML renders Quality's lookahead as well as CUDA does (and runs it in real time: the call p95
table above).

### Attribution: the plan alone (exe-d4, r5; the F0 overlap off on both sides, A: plan off, B: plan on)

`d2-stages`, 7 ABBA pairs per EP, 03:32-04:25 (background: 0 `rustc.exe`, CPU 22-35 %); a second,
independent 7-pair run of the same gate (the duplicated queue, to 04:27) in brackets:

| EP / run | call p95 | content p50 | content p95 | engine load |
|---|---|---|---|---|
| CUDA offline | 14.57 -> 11.75 (-18.6 %) [-24.0 %] IMPROVED | **4.91 -> 1.82** (-62.9 %) [4.80 -> 1.85] | 5.77 -> 2.88 (-49.7 %) | 2201 -> 2439 ms (+12.5 %) [+14.0 %] REGRESSED |
| CUDA paced | 15.61 -> 11.45 (-24.1 %) IMPROVED | 4.97 -> 1.88 (-62.9 %) | 6.12 -> 2.92 (-54.0 %) | |
| CUDA h2-load70 | 21.77 -> 14.48 (-34.0 %) [-32.5 %] IMPROVED | 8.37 -> 1.93 (-76.8 %) | 10.77 -> 3.44 (-68.3 %) | |
| DML offline | 14.63 -> 11.18 (-21.3 %) [-26.2 %] IMPROVED | **4.27 -> 1.49** (-64.9 %) [4.11 -> 1.44] | 5.45 -> 2.47 (-55.5 %) | 2834 -> 3228 ms (+14.2 %) [+12.2 %] REGRESSED |
| DML paced | 16.06 -> 12.05 (-23.4 %) IMPROVED | 4.68 -> 1.54 (-67.2 %) | 6.01 -> 2.48 (-59.0 %) | |
| DML h2-load70 | 19.70 -> 15.87 (-18.8 %) [-21.4 %] IMPROVED | 5.60 -> 2.39 (-57.4 %) | 7.59 -> 3.79 (-50.0 %) | |

CPU: CUDA offline 0.99 -> 0.99 cores, paced / h2-load70 -23 %; DirectML -50 to -59 %. VRAM: CUDA
1306 -> 1332 MB (+26 MB), DirectML 861 -> 853 MB.

- The plan alone meets T2's content targets on both EPs (CUDA 1.82-1.85 ms <= 2.0; DirectML
  1.44-1.49 ms <= 2.5). With the F0 overlap on (the shipped default) DirectML's content p50 reads
  3.07 ms because RMVPE runs on the same DirectML queue during the content call; the call itself is
  shorter with the overlap than without it (call p95 8.98 ms against 11.18 ms here), so the overlap
  stays on.
- Cold load with the plan: +240-350 ms (CUDA) and +390 ms (DirectML) with the overlap off, against
  +129 / +61 ms (A/B both paths, exe-d2) and +76 ms (DirectML 20 ms lookahead, exe-d4) with it on:
  at the load's warm-up tick the overlap runs RMVPE's first (slow) call on the helper thread while the
  worker builds the content plan, which hides most of the plan's build. Shipped defaults are inside
  T9's +0.3 s; the overlap-off path is not, and the deferred acquisition
  (`content::acquire_for_engine`, a one-line change in `load.rs`, requested from P3B-H / P3B-I0)
  takes another 230 ms (CUDA) / 380 ms (DirectML) off by never building the dynamic session (dev rig,
  `content_cold_load_dev_rig`).
- End-to-end latency (T1): paced 132.6 ms on both sides (NEUTRAL), CUDA h2-load70 142.6 ->
  132.6 ms (IMPROVED: one block less behind under load), DirectML h2-load70 NEUTRAL. The gate's
  other REGRESSED rows are the T1 split's engine wait / hold (+5-21 %: a block that is computed
  sooner waits longer for its output slot, at the same end-to-end latency) and DirectML's F0 onset
  call (+29 %, the RMVPE call after a silence skip).

### A P5 family voice on the shared code (exe-s0 against exe-d4, r5)

`ddsp-family-cuda.spec.json` (DDSP-SVC 6.3, the synthetic GUI voice: it shares the content-encoder
pool and the GPU plan code), `-CompareExe` with exe-s0 as A, 3 ABBA pairs, 03:32-03:49: PASSED, no
row REGRESSED or INCONCLUSIVE; offline call p95 57.7 -> 51.2 ms (-10.1 %, IMPROVED), model p95
-9.4 % (IMPROVED), paced and H1 rows NEUTRAL, VRAM 1206 -> 1204 MB, load NEUTRAL. (The family's own
pooled encoder runs the dynamic path, so the plan code is not on its tick; the check guards the
shared pool, `acquire_for_slot` and the GPU layer.)

### Acceptance against the plan (P3B-D)

| criterion | result | status |
|---|---|---|
| T2 content p50, CUDA 4.4-4.6 -> <= 2.0 ms | **offline** plan alone 4.91 -> 1.82 ms (replicated 4.80 -> 1.85); with the overlap 4.13 -> 1.98 ms. **Paced** (40 ms blocks, the live shape) plan alone 4.97 -> 1.88 ms but **with the shipped overlap 4.93 -> 4.23 ms**; h2-load70 plan alone 8.37 -> 1.93, with the overlap 8.17 -> 2.42 ms | met offline and under load; **missed as streamed (paced, shipped defaults): 4.23 ms against 2.0**. Reason: between 40 ms blocks the RTX 4090 drops its clocks, and a block that keeps the GPU busy for less time runs at lower clocks (a GPU power-state effect for P3B-G / P3B-H, not a property of the plan). The P7-B scorecard judges T2 on the paced and h2-load70 rows, not on the offline ones |
| T2 content p50, DirectML 3.7-3.9 -> <= 2.5 ms | **offline** plan alone 4.27 -> 1.49 ms (replicated 4.11 -> 1.44); with the F0 overlap on 4.07 -> 3.07 ms. **Paced** plan alone 4.68 -> 1.54 ms, with the overlap 5.71 -> 4.89 ms; h2-load70 plan alone 5.60 -> 2.39, with the overlap 5.80 -> 4.07 ms | met by the plan alone on every row; **missed at the shipped defaults** (3.07 offline, 4.89 paced, 4.07 under load against 2.5): RMVPE shares the DirectML queue during the content call, and the call itself is shorter that way (offline call p95 8.98 ms against 11.18 ms without the overlap), so the overlap stays on and the row stays a miss |
| sve-cuda offline call p95 10.9 -> <= 8.0 ms | 11.66 -> 7.40 ms (-36.9 %, IMPROVED, 9 pairs); **paced 16.62 -> 15.39 ms (-6.7 %)**, plan alone paced 15.61 -> 11.45 ms (-24.1 %) | met offline; paced improves only 6.7 % with the overlap (the same clock-drop reason as the content row) |
| sve-dml offline call p95 11.1 -> <= 9.5 ms | 12.17 -> 8.98 ms (-26.7 %, IMPROVED); paced 18.98 -> 14.69 ms (-22.6 %) | met |
| sve-cuda h2-load70 call p95 25.5 -> <= 18 ms | 19.23 -> 9.60 ms (-51.4 %, IMPROVED) | met |
| over-budget stays 0 % | 0 on every run with the plan (A/B both paths); DirectML 10 ms crossfade 98-99 % -> 0; 20 ms lookahead 100+ underruns -> 0 | met |
| DirectML 10 / 15 ms crossfade or 20 ms lookahead <= 5 ms per content call | 10 ms crossfade 56.1 -> 3.19 ms; 20 ms lookahead 34.4 -> 3.13 ms (15 ms crossfade not run: same mechanism, any fixed length) | met |
| Quality's 20 ms lookahead runs on DirectML, fidelity rows next to CUDA's | runs (the factory skip lifts); speaker 0.957 vs 0.956, MCD 1.743 vs 1.742 dB | met |
| CUDA graphs pass check_graph | every content plan at every length (dev rig) | met |
| DirectML static vs dynamic rel L2 <= 1e-3 | fp16 0 (bit-identical), fp32 <= 2.3e-6 | met |
| plan path T5 rows within A/A noise of the plain path | DirectML bit-identical; CUDA speaker cosine inside the A/A, MCD +0.003 dB and output F0 +0.6 cents beyond the A/A but <= 1/10 of the offset spread, mixed direction | met on DirectML; CUDA: a different, not worse, fp16 rendition |
| VRAM per plan and the DML dynamic vs fixed delta recorded; plan adds <= 0.3 s to cold load | plan 208-498 MB by EP / precision; engine VRAM CUDA +26-68 MB (the graph), DirectML -8 MB; load +129 ms CUDA / +61-76 ms DirectML with the shipped overlap, +240-350 / +390 ms with the overlap off | met with the defaults; the overlap-off path misses (the requested `load.rs` deferred acquisition saves 230 / 380 ms) |
| zero heap allocations per tick in the content step | `bench::hot_path_tests::sve_content_step_is_allocation_free_on_its_plan` (plan, 1 s content window, F0 hand-over) | met |
| P3 feature experiment and engine A/B published; lever ships off with a recommendation | published; `content_norm` off; recommendation: keep it off by default, opt-in | met |
| P7 decision recorded | dropped (the end-aligned grid agrees less with the utterance) | met |
| precision: int8 and fp16 stream-length results, defaults per the rule | fp16 fails the feature gate at stream lengths but is neutral downstream (keep fp16, waiver with the per-EP numbers in `content::variant_order` and ADR 0011); CPU int8 fires the rule by 0.0006 speaker cosine (kept: the fp32 encoder's CPU cost is unmeasured and the Streamer already misses its RTF target; ADR 0010 / 0011, P7-B measures both). `precision = fp32` puts fp32 first on both classes; `SIGIL_VC_INT8=0` turns the CPU's int8 off | met as a recorded waiver, not as the rule's outcome |

### Recommendations for P3B-I0 (defaults are its flips, rule R6)

- **Content plan: on** (already the default on CUDA / DirectML through the `io_binding` /
  `cuda_graphs` auto table; kill switch `SIGIL_VC_CONTENT_PLAN=0`). Keep its CUDA graph
  (`SIGIL_VC_CONTENT_GRAPH`): without it the CUDA plan is no faster than the dynamic session.
- **F0 overlap: on** (default; `SIGIL_VC_F0_OVERLAP=0`). Its CPU cost on CUDA (+0.34 cores offline:
  two threads in CUDA's spin-waits) goes with P3B-G's blocking waits.
- **`load.rs`: `content::acquire_for_engine`** instead of `acquire_for_slot` (same arguments; it falls
  back to it on the CPU, for explicit / legacy files and for files without an export contract).
- **`content_norm`: stays off**, offered as an opt-in (speaker similarity +0.01 at +0.04 dB MCD);
  publish the `stream` variants only if it is ever adopted. **`content_context_ms`: automatic**, no
  preset content contexts.
- **GPU content precision: fp16 stays** (neutral downstream). **CPU: int8 stays** as the decision
  of P3B-I0 (ADR 0010: the fp32 encoder's CPU cost is unmeasured and the Streamer already misses its
  RTF target; P7-B measures both); the rule fired by 0.0006 speaker cosine. Both are recorded with the
  per-EP numbers as waivers in `content::variant_order` and ADR 0011; `precision = fp32` or
  `SIGIL_VC_INT8=0` give the rule's outcome. The embedder export's own `recommended: fp32` (and the
  catalog comment "those run fp32 on GPUs") describe the export gate, not the engine's decision.
- The factory's DirectML lookahead skip should read the loaded voice's
  `SigilRvcEngine::content_runs_fixed_shape()` (a voice whose plan was refused, e.g. an fp16-I/O
  explicit file, still pays 35-49 ms per call with lookahead), not the process switch
  `content::runs_fixed_shape("dml")`.
