# ADR 0010: Sigil Voice Engine optimization (P3B)

- **Status:** accepted (P3B-I0, 2026-10-09). Defaults were flipped only from interleaved evidence; the
  final single scorecard of the shipped defaults is P7-B's.
- **Date:** 2026-10-09
- **Workstream:** P3B-I0 (the ADR belongs to its workstream; INTEGRATE links it from `docs/adr/README.md`)
- **Plan decision:** Voice-changer engine architecture (main engine); P3B target revisions
  ([plan → P3B outcomes](../SPEECH_STACK_IMPLEMENTATION_PLAN.md#p3b-outcomes-binding-amendments-for-p7))
- **Related:** [ADR 0004](0004-rvc-templates.md) (templates), [ADR 0009](0009-vc-scorer-v2.md) (the scorer v2 that
  judged fidelity here), [VC_BENCHMARKS.md → P3B](../benchmarks/VC_BENCHMARKS.md#p3b-voice-engine-optimization-p3b-i0)

## Context

P3 missed several of its own targets on the dev rig: the per-block compute (T2: CUDA 10.9 ms, DirectML
11.1 ms idle against 6 ms), the Low-latency end-to-end delay (T1: 132.6 ms against 85 ms), VRAM with
RMVPE (T3: DirectML 861 MB, CUDA 1312 MB) and the CPU generator (T4). P3B measured first (rule R1) and
found that the per-tick cost is overhead, not arithmetic:

- about 1 ms of GPU math against 9.6 ms measured per call: roughly a thousand small launches, four
  serial CPU-GPU sync points per call, and host round trips between `enc` and `flowdec`;
- ContentVec still ran as an eager dynamic-shape session outside the session-plan layer (51-60 % of the
  p95 under the 70 % load: CUDA 12.2 of 20.4 ms, DirectML 8.3 of 16.3 ms), and F0 ran after it, serially;
- RMVPE's plan was sized for the first, whole-window call (64 mel frames) although every later call is an
  incremental 24-frame one, and a second, plain copy of RMVPE was built first (the load-time VRAM transient);
- the CUDA wait spun a core for the whole GPU time of a run (0.40 cores paced against DirectML's 0.12);
- the cold load built the generator, the content encoder, RMVPE and the index one after the other, and
  built the generator twice for any non-default lookahead or crossfade.

The benchmark itself also did not resolve what the targets ask for: the pair SD of a `call_ms_p95` was
11-19 % offline, the GPU clock governor held the SM clock low for whole sessions, the CAM++ asset computed a
different function from upstream, the click statistic was ill-conditioned and the competitor comparison scored one clip.

## Options considered

| Option | Pros | Cons | Evidence |
| --- | --- | --- | --- |
| A. Ship every lever on | the biggest single numbers | several levers trade fidelity, memory or safety for speed; a bad default is invisible to users | the filled F0 lead (speaker cosine -0.02), the 32-frame RMVPE plan (gross pitch error 4.9 -> 8.2 % on noisy and creaky speech), the adaptive cushion (underruns under the 70 % loads) |
| B. Ship everything off, switches only | no risk | leaves lossless wins on the table (a 45 % faster cold load, a content call 2.7x cheaper, 84-94 % less CUDA CPU) | the interleaved gates below |
| **C. Lossless or measured-better paths on, everything that can lower fidelity a switch** | keeps quality first (rule R6), every path keeps its kill switch (R5) | some target rows stay missed at the shipped defaults | this ADR |

## Decision

**Option C.** Every item has a kill switch; the ones marked *on* are the shipped defaults.

On (the integrated build keeps these unless the switch says otherwise):

| Path | Switch | Interleaved evidence (ABBA, one exe) |
| --- | --- | --- |
| Content encoder as a fixed-shape session plan, plus the F0 overlap helper (P3B-D) | `SIGIL_VC_CONTENT_PLAN=0`, `SIGIL_VC_CONTENT_GRAPH=0`, `SIGIL_VC_F0_OVERLAP=0` | content p50 CUDA 4.91 -> 1.82 ms, DirectML 4.27 -> 1.49 ms; offline call p95 CUDA 11.66 -> 7.40 ms, DirectML 12.17 -> 8.98 ms; CUDA h2-load70 call p95 19.23 -> 9.60 ms; lookahead on DirectML (content 34.4 -> 3.13 ms) |
| Final shapes at load, parallel cold start, `enc` / `flowdec` on two threads, single-pass fp16 image, background loads sequential (P3B-H, P3B-G) | `SIGIL_VC_FINAL_SHAPES=0`, `SIGIL_VC_PARALLEL_LOAD=0` | cold load CUDA 1837 -> 1009 ms (-45 %), DirectML 2504 -> 1300 ms (-49 %), Quality CUDA 2343 -> 1485 ms; 200 loads per EP without a crash |
| RMVPE plan built once at load from the file's metadata, shared across voices, split long calls (P3B-E) | `SIGIL_RMVPE_SHARE=0` | no plain second copy (the load transient), second hot voice +130 / +136 MB with a shared plan |
| Device-resident generator chain (P3B-G) | `SIGIL_VC_DEVICE_CHAIN=0` | audio bit-identical; syncs per call 4 -> 3; the `enc` -> `flowdec` gap 0.35 -> 0.08 ms |
| Hybrid CUDA wait (P3B-G) | `SIGIL_CUDA_TUNING=wait=spin` | worker CPU paced 0.629 -> 0.042 cores (-92.9 %), offline 1.306 -> 0.215, under load 0.45 -> 0.05; call p95 / p99 NEUTRAL; over budget 0 = 0 |
| Arena shrink before the CUDA graph capture (P3B-G) | `SIGIL_CUDA_TUNING=shrink=0` | -40 MB on every pair; call p95 NEUTRAL; graphs kept |
| CPU session arena (P3B-G) | `SIGIL_VC_CPU_ARENA=0` | Streamer shape RTF p50 -5.9 %, paced p99 -9.4 %, cores -4.6 % |
| Parallel SwiftF0 / RMVPE fusion lane: the voicing extractor on a persistent thread beside RMVPE (P3B-F) | `SIGIL_VC_FUSION_PARALLEL=0` | bit-identical to the serial path; F0 stage on a quiet GPU CUDA 4.66 -> 3.12 ms, DirectML 4.15 -> 3.94 ms |

Off, as a switch or an opt-in setting (each with the reason; nothing here was dropped):

| Path | Reason it is not the default |
| --- | --- |
| 32-frame RMVPE plan, the "VRAM saver" (`SIGIL_RMVPE_PLAN_FRAMES=32`) | DirectML 854 -> 654 MB and F0 -21 % per steady call, but gross pitch error 4.9 -> 8.2 % on noisy and creaky synthetic speech and slower on CUDA (steady +7 %, onset p99 +93 %). The real-speech fidelity A/B favoured it (fed fine pitch error 22.4 -> 19.5 cents), so the evidence is mixed and the trade is the user's. The setting is not exposed in the UI yet |
| RMVPE fp16 under `auto` (`FP16_UNDER_AUTO`) | its gate failed on DirectML at the default plan (gross pitch error +0.057 points against +0.05 allowed) |
| F0 cache priming and onset continuity (P3B-F; opt-in with `SIGIL_VC_F0_PRIME=1`, `SIGIL_VC_F0_CONTINUITY=1` or `SIGIL_VC_F0_GEOMETRY=primed`; `F0Geometry::DEFAULT` is P3's cache) | what they buy is structural (no extractor call longer than its plan, no whole-window call at an onset; a no-op on the default 64-frame plan at Low latency), not quality, and the evidence is not "no regression": on pause clips 3 to 22 of about 5,000 fed frames differ from P3's cache (one by an octave), `multi-3` at the 64-frame plan gross errors 2.012 -> 2.117 %; the engine A/B that read bit-identical compares per-clip medians, which hide isolated frames; nothing was measured with FCPE. Rule R6 keeps them off until P3B-I0 has per-frame evidence for RMVPE and FCPE |
| Filled F0 calls (`SIGIL_VC_F0_GEOMETRY=filled`, the measured combination with priming and continuity; `SIGIL_VC_F0_LEAD=fill` alone changes only the lead) | FPE 23.6 -> 17.8 cents and V/UV 4.0 -> 2.3 %, but speaker cosine -0.021 at Low latency (within noise, mostly tonal clips) |
| Cross-tick F0 decoding (`f0_decode` lever) | did not improve pitch on real speech, lowered speaker cosine (-0.014 / -0.016, CI excludes 0) |
| Running content normalisation (`content_norm` lever) | speaker cosine +0.01 but MCD +0.04 dB; needs an unpublished encoder variant |
| 1 s of content context | MCD +0.08-0.10 dB on 7 of 8 clips |
| Shared DirectML queue (P3B-G; opt-in with `SIGIL_VC_DML_SHARED=1`) | cold load Hodges-Lehmann -93 ms (CI -269..-37, 8 of 8 pairs faster) and call p95 -7.1 % paced, but the content stage's p95 REGRESSED on the one queue (+6.1 % paced, +21.9 % under the 70 % load, rule R3), the gain's CI bound (37 ms) is under T9's 50 ms and the integrated build did not reproduce it (1246 against 1254 ms). P7-B re-measures on a quiet rig |
| Adaptive output cushion (`output_cushion=adaptive`) | -18 ms on a calm machine but it underran on the 70 % and out-of-process loads (fixed: 0); experimental label |
| 10 ms device block (`device_block=model_divisor`) | -17.4 ms paced on CUDA with 0 underruns in 7 pairs, but not yet measured on DirectML, the CPU, every load or a live WASAPI device. P7-B confirms, then it can become the default |
| Low-latency lookahead 10 ms | +0.032 set-sim, -0.13 dB env MCD, but it spends 10 ms of the latency the preset exists for |
| 10 ms crossfade, frame-aligned SOLA tail, narrower SOLA search | the 10 ms crossfade fails the T5 v2 gate (cosine v2 -0.040, DNSMOS -0.16); the aligned tail lowers cosine v2 by 0.014 (CI excludes 0); narrowing hurt fidelity |
| GPU prewarm (`SIGIL_VC_PREWARM=1`) | no first-load gain (1644 vs 1669 ms) and it holds a 338 MB CUDA context for the whole session |
| fp16 kNN storage on CUDA | saves 44 MB at 2.7e-4 blend rounding; the exact blend stays (fidelity-adjacent) |
| GPU process priority, denormals as zero, `dynamic_block_base`, context-wide blocking waits | NEUTRAL or unmeasured against STT / TTS (rejected or opt-in experiments) |
| GPU fp16 ContentVec -> fp32 first | the parity rule fired at stream lengths, but the downstream scorer-v2 numbers are neutral (CUDA +0.0008 cosine / +0.0045 dB MCD, DirectML +0.0001 / -0.0062) and fp32 costs about 170 MB per plan; fp16 stays first on the GPU rungs ([ADR 0011](0011-vc-content-encoder-precision.md)) |
| CPU int8 ContentVec -> fp32 first | the downstream rule marginally fired (speaker cosine -0.0026 against 0.002; MCD and V/UV better), but the fp32 encoder's CPU cost was not measured and the Streamer is already over its RTF target. P7-B measures both ([ADR 0011](0011-vc-content-encoder-precision.md)) |
| Streamer preset on 6 threads at below-normal priority | best measured CPU policy (RTF p99 0.74-0.83, 2.8 cores) but the app has no thread-count setting yet, and it is not H3-tested |

**Target revisions (accepted by the user, measured against honestly):** T1 keeps 85 ms for the reference
config and reports 95 ms as achieved; T1 Streamer p99 280 ms at 120 ms blocks; T2 splits into an idle row
(p95 6 ms) and a 70 %-load row (p95 10 ms, p99 15 ms, 0.1 % over budget); T3 per EP with the CUDA context
reported separately and a new load-peak row; T4 fp32 RTF p99 0.6 at 3.5 cores; T5 re-defined under scorer
v2 ([ADR 0009](0009-vc-scorer-v2.md)); T6 statistically defined (corpus, paired statistics, Pareto latency,
same-GPU RTF only).

## Evidence

Every number is an ABBA-interleaved comparison inside one executable (`run-perf-gate.ps1 -CompareEnv`,
`-CompareExe`), with the paired log-ratio interval and a verdict (REGRESSED, INCONCLUSIVE, IMPROVED,
NEUTRAL). The gate's own resolution was measured with two A/A sessions: 62-64 of 64 rows NEUTRAL, no false
REGRESSED, `call_ms_p95` interval half-width 1.3-4.3 % offline and under the 70 % hammer, 8-13 % on the
paced rows (the GPU governor drops clocks between ticks at a real-time cadence). Per-workstream commands and
tables: `scripts/bench/specs/p3b/<a..h>/README.md`; the summary is in
[VC_BENCHMARKS.md → P3B](../benchmarks/VC_BENCHMARKS.md#p3b-voice-engine-optimization-p3b-i0). Per user scope
change (2026-10-09), P3B-I0 ran no final regression scorecard: it ran one functional smoke per execution
provider on the integrated build (the section lists what it proves) and P7-B runs the single final scorecard
of the shipped defaults on a quiet machine.

## Consequences

- The shipped defaults get cheaper (CUDA CPU, cold load, content and F0 stages) without changing a sample of
  audio, except through the gated levers that stay off. The scorer v2 baseline for the shipped defaults is
  therefore the P5 baseline in ADR 0009 for every T5 row: the F0 cache is P3's (priming and continuity are opt-in; when on they changed no per-clip median).
- Misses stay misses: T1 (132.6 ms at the shipped defaults; 115.2 ms with the 10 ms device block), T2 on
  DirectML under load (11.6 ms against 10), T3 DirectML (853 MB against 650; 654 with the saver) and CUDA
  (about 1340 MB against 800, 1156 with the saver), T4 Streamer RTF (best 0.69 / 0.74-0.83 against 0.6; the
  arithmetic needs at least 4.0 cores at perfect scaling) and most T5 v2 rows. T6 is undecided until the
  competitor renditions exist.
- Deferred with reasons: TensorRT rungs and fused engines (no runtime pack, over the 50 MB rule, needs live
  measurement); the int8 CPU decoder (the only route to T4's original 1.5 cores, lowest priority); the
  DirectML merged `enc` + `flowdec` graph; an app setting for the VRAM saver and for the CPU thread count;
  moving the BS.1770 loudness helper from `compare.rs` into `dsp.rs`; lifting the workstreams' profile rows
  into `scripts/bench/profiles/p3b.json` (each runs today with `-Profile scripts/bench/specs/p3b/<x>/*.profile.json`).
- Manual checks stay open: the H1 and H3 live records, the e2e loopback over a virtual cable, competitor
  installs and the public corpus fetch, a listening check of DirectML `ja-long`'s seams (one narrow-window
  click), a live WASAPI run of `device_block=model_divisor`, and the blind A/B of the head-to-head ablations.
