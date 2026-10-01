# Voice changer benchmarks: the Sigil Voice Engine scorecard

This is the scorecard of the Sigil Voice Engine (SVE) against the plan's targets T1–T10
([SPEECH_STACK_IMPLEMENTATION_PLAN.md → VC best engine → Targets](../SPEECH_STACK_IMPLEMENTATION_PLAN.md#targets)).
Every number comes from Sigil's own harness (`Sigil.exe --headless bench`, [README.md](README.md)),
driving the production stream harness and the voice changer's engine factory on a real voice. No
number is a vendor figure.

Status words:
- **met** / **missed**: measured on the dev rig (H2) under the stated conditions.
- **provisional**: measured with the load simulator standing in for H1 (RTX 3060/4060 + VRChat +
  OBS) or H3 (6-core CPU + VRChat). It becomes authoritative when a friend sends a live-record
  export from a real H1 or H3 session.
- **not measured**: needs a manual step ([Still to run](#still-to-run-manual)), or belongs to a
  later phase.

## Setup

| | |
|---|---|
| Rig (H2) | AMD Ryzen 7 7800X3D (8 cores, 16 threads), NVIDIA RTX 4090 24 GB (driver 32.0.16.1714), Windows 11 Pro 26200 |
| Build | Sigil 0.4.0-1, release (`lto`, one codegen unit), ONNX Runtime 1.28.0 (ort 2.0.0-rc.13), CUDA 13 runtime pack `runtime/nvidia-cuda13` |
| Voice | The MIT default voice (`vc/voice-default`, `default.pth`: RVC v2, 40 kHz, f0, HiFiGAN-NSF, ContentVec 768), imported by Sigil's `.pth` importer into the `v2-40k-f0-hifigan` template (`sigil-rvc-1`) |
| Models | `vc/embedder-contentvec`, `vc/pitch-rmvpe`, `vc/pitch-fcpe`, `vc/pitch-swiftf0`, `vc/pitch-crepe-tiny`, `bench/campplus`, `bench/dnsmos-p835` |
| Clip | `en-jfk.wav` (5.8 s of public-domain speech, 16 kHz), resampled to the 48 kHz device rate. The plan's 20-clip consented corpus does not exist yet, so the fidelity rows describe this one clip. |
| Stream | The voice changer's defaults: 48 kHz mono in, stereo out, 512-frame device block (10.7 ms), 64 ms extra context, 20 ms crossfade, 35 / 70 ms output cushion, 300 ms latency budget |
| Reference configuration | The plan's: 40 ms engine blocks, RMVPE on the GPU, the Low-latency preset with the `auto` levers. In P3-F's runs below `auto` meant the algorithmic levers on (partial decode, anchored noise, phase0, F0 cache) and the GPU levers off; since P3-I0 it also turns IOBinding (CUDA, DirectML) and CUDA graphs (CUDA) on ([P3 integration](#p3-integration-p3-i0-the-benchmark-decided-defaults)). On the CPU, the Streamer preset: 80 ms blocks, SwiftF0 |
| Conditions | 2026-09-30. Runs one after another; each waited until no other build, test or benchmark ran on the PC. The desktop stayed in use (a browser playing video: about 15 % of the video-decode engine, under 5 % 3D). The same spec repeated gives up to ±20 % on a p95 and more on CUDA under load; the tables say where runs disagreed. |

## Scorecard

Per-block compute is `call_ms`: the worker time of the device blocks that ran the model (a 40 ms
engine block runs once every 3.75 device blocks of 10.7 ms). The target rows come from
`run-perf-gate.ps1 -Profile p3` unless the row says otherwise; "+levers" means IOBinding, CUDA
graphs (CUDA) and the priority stream / queue switched on. The P3 exit rows (T2, T3, T5, T7, T9 on
H2 with the load simulator, T4 streamer) are the profile's `gate: true` targets; T2's per-block
compute is gated on the `h2-load70` run (H2 is the dev rig under the load simulator), and the idle
`offline` numbers are advisory.

| ID | Target | Measured | Status |
|---|---|---|---|
| T1 | Low latency, engine-added p99 ≤ 85 ms (paced harness, H2) | reference config 132.6 ms (p50 = p99); CUDA +levers, SwiftF0, 20 ms blocks, 10 ms cushion and crossfade: **80.6 ms**, but at a fidelity cost (cosine 0.77 vs the ceiling) | missed at the reference quality |
| T1 | H1 under load: end to end p50 ≤ 80 ms, p99 ≤ 110 ms | engine-added p99 132.6–142.6 ms under the H1 simulation (DirectML) | missed (provisional) |
| T1 | H2 end to end p50 ≤ 55 ms | the e2e-loopback meter needs a virtual cable | not measured |
| T1 | Streamer (CPU) p99 ≤ 230 ms on H3 with a game | Streamer preset (one thread) 784.9 ms under the H3 simulation; the autotuned CPU preset 380.6 ms idle | missed (provisional) |
| T2 | Per-block compute p95 ≤ 6 ms (H2: the dev rig under the 70 % load simulator, the gated row) | under the H2 load: DirectML 17.1 ms; CUDA 48.0 ms (+levers 27.7). **P3-I0 defaults:** CUDA 25.5 ms | missed |
| T2 | Per-block compute p95 ≤ 6 ms, idle (advisory) | DirectML 12.4–15.5 ms (+levers 11.6); CUDA 23.8–32.3 ms (+levers 17.5–19.8); CUDA +levers with SwiftF0 10.1 ms. **P3-I0 defaults:** CUDA 10.9 ms, DirectML 11.1 ms | missed |
| T2 | H1 under load: p99 ≤ 20 ms, ≤ 0.1 % of blocks over budget | DirectML 18.5 ms, 0 % (H1 simulation) | met (provisional) |
| T2 | H2 load simulator (70 %): ≤ 0.1 % of blocks over budget | DirectML 0 % in every run; CUDA 2–62 % of calls in three runs (+levers 0.4 %). **P3-I0 defaults:** CUDA 0 % (H2 load and H1 simulation) | met (DirectML; CUDA since P3-I0) |
| T2 | Context 64 → 512 ms raises the per-block p95 by ≤ 30 % | DirectML: −16 % (within noise; context costs nothing measurable) | met |
| T2 | Zero heap allocations per block on the GPU hot path | counting-allocator tests (`bench::hot_path_tests`): 0 per block on the real-time I/O path and in the generator's bound `enc` / `flowdec` runs; the content encoder and F0 steps are not covered yet | met (I/O, generator) |
| T3 | Low latency, one voice ≤ 650 MB (process delta) | DirectML 861 MB with RMVPE, 484–537 MB with SwiftF0, FCPE, CREPE-tiny or DIO; CUDA 1312 MB (about 360 MB of it the CUDA context; 776 MB with SwiftF0) | missed with RMVPE |
| T3 | +150 MB per extra hot voice; Quality with the index on the GPU ≤ 900 MB | hot slots are P3-E's, and the default voice has no index | not measured |
| T4 | GPU presets ≤ 0.25 cores | DirectML 0.12–0.16 cores; CUDA 0.57–0.65 (+levers 0.43; +levers with SwiftF0 0.25). **P3-I0 defaults:** CUDA 0.40 | met (DirectML) |
| T4 | Streamer ≤ 1.5 cores at RTF ≤ 0.6 (80–128 ms blocks) | Streamer preset: 0.99 cores at RTF 3.35 (one ONNX Runtime thread); the default thread pool: RTF 0.92 at 120 ms with 4.6 cores, 0.64 at 200 ms with 3.4 cores | missed |
| T4 | Enabled and silent ≤ 0.02 cores, no GPU inference | DirectML 0.0095 cores; CUDA +levers 0.022 cores; the model is not run while silent | met (DirectML) |
| T4 | ONNX Runtime never spins | every session is built through `ort_session` with spinning off | met (by construction) |
| T5 | CAM++ speaker cosine vs the offline ceiling ≥ 0.97 | DirectML 0.935; CUDA 0.924–0.929; CPU 0.951; the autotuned CPU preset (200 ms blocks) 0.972. **With 20 ms of lookahead (Quality, P3-I0): CUDA 0.975** | met with lookahead on CUDA; missed at the Low-latency shape |
| T5 | MCD within 0.5 dB | 1.70 dB (CUDA 1.72, CPU 1.57). **P3 final gate:** DirectML 1.68, CUDA 1.66 | missed |
| T5 | Voiced F0 RMSE ≤ 15 cents; V/UV error ≤ 3 % | RMVPE on DirectML 68 cents, 8.0 %; RMVPE on CUDA 49 cents, 6.8 % (68 cents, 9.2 % on its CUDA-graph plan since P3-I0); SwiftF0 on the CPU 53 cents, 8.4 %; CUDA with 20 ms lookahead 63 cents, 3.5 %. **P3 final gate (RMVPE's mel end-aligned, [below](#p3-final-gate-pitch-track-fixes)):** DirectML 47 cents, 3.8 %; CUDA 53 cents, 3.5 % (part of the RMSE gain is the scorer's own RMVPE) | missed (V/UV within 0.8 pt) |
| T5 | 0 block-boundary clicks | 0 in every run at 40 ms blocks and longer | met |
| T5 | CER of the converted speech ≤ offline + 1 pt | stream, ceiling and the unconverted input transcribe identically (Whisper tiny.en): +0 pt | met |
| T6 | Head to head: DNSMOS +0.05, GPE ≤ each, A/B ≥ 60 %, latency ≤ 0.7× | needs the competitors' outputs (`scripts/vc-bench`) | not measured |
| T6 | Normalized RTF below vc-rs (13.9 ms p95 per 100 ms) and rvc-realtime (15.7 ms per 60 ms) | RMVPE: 13.6 ms per 100 ms, 15.9 ms per 60 ms (DirectML); SwiftF0 with CUDA graphs: 10.1 ms per 40 ms, 10.6 ms per 80 ms; on a faster GPU than rvc-realtime's | not met on equal terms |
| T7 | Backlog never above budget + one block | 0 ms over in every run (stub, DirectML, CUDA, CPU, 3 s stalls) | met |
| T7 | Back within budget ≤ 2 blocks after a 3 s stall | 0 blocks (stub, DirectML, CUDA) | met |
| T7 | 30 min soak with 0 underruns; EP failure shown within 1 s; a returning device resumes within 2 s | not measured here (the P0 / P1 tests cover the EP report and the device watcher) | not measured |
| T8 | ≥ 95 % of the 40-voice community corpus imports | needs the user's corpus | not measured |
| T8 | 200-file malformed corpus: 0 crashes or hangs | P3-B's fuzz corpus test passes | met |
| T8 | Import ≤ 10 s per voice | the MIT default voice (55 MB `.pth`): 0.84 s including the smoke inference | met |
| T8 | Every bring-your-own ONNX slot keeps working | the legacy `RvcModel` slot runs through the harness in the bench tests | met (plumbing) |
| T9 | Hot switch within one block, with a crossfade | P3-E's factory tests (functional; not timed here) | not measured |
| T9 | Cold slot load ≤ 2.5 s including warm-up / graph capture | DirectML 2.7–3.2 s; CUDA 1.9–3.1 s (1.94 s with graph capture, P3-I0 defaults); CPU 1.0–1.1 s | missed (DirectML) |
| T9 | TensorRT-RTX first build ≤ 60 s, cache hit ≤ 3 s | the `runtime/tensorrt-rtx` pack is not installed on the dev rig | not measured |
| T9, T10 | Zero-shot slot ≤ 5 s; Sigil-ZS, DDSP-SVC singing; TTS through the voice | zero-shot and singing are phase P5; TTS routing is covered by the P2 / P3-E tests | later phase |

In one paragraph: the engine's real-time machinery is sound (0 clicks, the budget and stall
recovery hold under every load, 0.01 cores when silent, context is free, DirectML shrugs off a game
at 70 % of the GPU), and it runs every F0 method within about 10–15 ms per 40 ms block. What misses
is the absolute speed of the per-call fixed costs (T2), latency at the reference block (T1), VRAM
with RMVPE (T3), the CPU generator (T4) and the streaming pitch track (T5).
[What closes the gaps](#what-closes-the-gaps) lists the fixes the numbers point to.

## P3 integration (P3-I0): the benchmark-decided defaults

What P3-I0 changed, from the workstreams' measurements: the `auto` levers now turn on IOBinding on
CUDA and DirectML and CUDA graphs on CUDA (`engine::presets::auto_lever`; priority, GPU kNN and the
governor stay off); the engine factory builds the pitch extractor with the voice's GPU levers, so
RMVPE runs as the pitch bank's CUDA-graph plan on CUDA; the pitch bank reports RMVPE's and FCPE's EP;
and the new `lookahead_ms` parameter gives Quality and Studio-TRT 20 ms of lookahead.

**Before / after** (`run-perf-gate.ps1 -Profile p3`, release builds, run back to back on 2026-09-30:
before = the P3 workstreams' final build, after = the integration build; the default voice, the JFK
clip, 40 ms blocks, RMVPE, the Low-latency preset with `auto` levers):

| Spec / run | Before | After |
|---|---|---|
| CUDA offline: per-call p50 / p95 / p99 | 20.0 / 21.7 / 37.9 ms | **9.6 / 10.9 / 11.3 ms** |
| CUDA offline: RTF p99 | 0.95 | **0.28** |
| CUDA paced: CPU | 0.55 cores | 0.40 cores |
| CUDA, H2 load (70 %): per-call p95, calls over budget | 35.9 ms, 0.4 % | 25.5 ms, **0 %** |
| CUDA, H1 simulation: per-call p99, calls over budget, latency p99 | 44.0 ms, 3.6 %, 225.9 ms | 28.5 ms, **0 %**, 148.2 ms |
| CUDA fidelity run: per-call p99 (RMVPE spikes) | 133.8 ms | **18.2 ms** |
| CUDA cold load | 1.86 s | 1.94 s |
| CUDA fidelity: cosine / MCD / F0 RMSE / V/UV / DNSMOS | 0.927 / 1.73 dB / 48 cents / 6.5 % / 3.12 | 0.935 / 1.71 dB / 68 cents / 9.2 % / 3.08 |
| CUDA with every GPU lever (`sve-cuda-levers`) offline p95 | 16.0 ms | 11.4 ms (RMVPE now on its graph plan) |
| DirectML offline: per-call p50 / p95 | 10.6 / 11.7 ms | 9.7 / 11.1 ms (fidelity identical: deterministic) |
| CPU (Streamer) offline RTF p99 | 3.38 | 3.22 (unchanged path) |

- **CUDA becomes predictable.** The CUDA-graph generator and RMVPE plan halve the per-call cost,
  remove RMVPE's 130–150 ms p99 spikes, and keep every call inside its budget under both load
  simulations. CUDA now ties DirectML per call (10.9 vs 11.1 ms p95).
- **The p95 gate:** the run flagged 56 p95 "regressions", all in the specs that ran while the GPU sat
  in a slower state (`sve-dml-ctx512`, `sve-dml-levers`, the paced runs of `sve-dml` and
  `sve-cuda-levers`: every call at 16–17 ms p50 instead of 10 ms). Interleaved reruns of the same
  specs with both builds show the state, not the build: `sve-dml-ctx512` offline p50 16.85 / 17.07 ms
  (before / after) in one round and 10.67 / 10.60 ms in the next; `sve-dml-levers` 16.59 / 16.18 ms
  and 10.30 / 10.36 ms. No regression is attributable to the changes; the DirectML path changed
  only by IOBinding, whose output is bit-identical.
- **RMVPE's fixed-shape plan tracks the stream's F0 further from the ceiling** (68 vs 48–50 cents
  RMSE, V/UV 9.2 vs 6.5–7.4 % on CUDA; DirectML always ran the plan and scores 68 cents / 8.0 %),
  while speaker cosine and MCD are equal or slightly better and the p99 spikes disappear. The plan
  pads every call to the first call's mel length; whether that padding or the plan's shapes cost the
  F0 accuracy is a P3-D follow-up (both paths are bit-stable, so the comparison is exact).

**Lookahead** (CUDA, RMVPE, 40 ms blocks, `auto` levers; fidelity with a 4 s lead):

| Lookahead | Speaker cosine | MCD dB | F0 RMSE cents | V/UV error | DNSMOS | Clicks | Per-call p95 | Engine-added latency |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.940 | 1.66 | 69.5 | 7.4 % | 3.12 | 0 | 11.8 ms | 132.6 ms |
| 20 ms | **0.975** | **1.40** | 63.3 | **3.5 %** | **3.29** | 0 | 12.5 ms | 152.6 ms |

- 20 ms of lookahead meets T5's cosine row (≥ 0.97) and nearly its V/UV row (≤ 3 %) on this voice,
  for 20 ms of latency; Quality and Studio-TRT use it. With the plain RMVPE session (P3-D's run) the
  same change gave 0.947 → 0.969, V/UV 5.3 → 1.8 %, F0 58 → 38 cents.
- **Not on DirectML yet:** with lookahead the DirectML content encoder costs 35 ms per call (about
  4 ms without; the same DirectML shape problem as a crossfade other than 20 ms), so a 40 ms block
  runs at 45 ms. The engine factory skips lookahead while the content encoder runs on DirectML, with
  a lever note, until that encoder runs a fixed shape there (P3-C1 follow-up).
- **Scoring note:** fidelity runs with lookahead need `lead_s` ≥ 4 in the quality run. With the P3
  profile's 1 s lead the scorer reported 2700–2900 clicks per 10 minutes and 130 cents F0 RMSE for
  the same audio (an alignment artefact; P3-F follow-up).

## P3 final gate: pitch-track fixes

What the final gate changed and measured (2026-10-01, release builds of the tree before and after
the change, `run-perf-gate.ps1 -Profile p3` back to back, then interleaved reruns; the default
voice; RMVPE; 40 ms blocks; the Low-latency preset with `auto` levers unless a row says otherwise):

- **RMVPE's mel is end-aligned** (`vc/rmvpe.rs`): the audio's frames end the padded mel and the
  padding before them is the log-mel floor (silence). RVC's `mel2hidden` appends zero log-mel
  frames after the audio, a loud flat spectrum that RMVPE's BiGRU carried into the newest frames of
  every short incremental call (the F0 cache's 24-frame slices had 7 such frames after the newest
  one on the plain path, 39 on the 64-frame plan). Same padded size, same cost.
- **Not adopted: a 20-frame FCPE lead-in.** It improved FCPE's pitch track against whole-clip F0
  (V/UV 3.31 → 2.85 % in the F0-cache sweep) but not the converted audio (table below), so FCPE keeps
  the cache's default 12 frames.
- **Low latency keeps 0 ms of lookahead** (table below): 10 ms buys fidelity for latency, a trade the
  Low-latency preset does not make while T1 is missed; Quality keeps 20 ms.

**Gate run, the JFK clip** (stream against the offline ceiling; the ceiling also changed, it runs the
same RMVPE):

| Run | Speaker cosine | MCD dB | F0 RMSE cents | V/UV error | Gross pitch error | DNSMOS | Clicks | Seam flux ratio |
|---|---|---|---|---|---|---|---|---|
| DirectML before | 0.929 | 1.703 | 68.1 | 8.0 % | 1.28 % | 3.06 | 0 | 1.53 |
| DirectML after | 0.925 | **1.679** | **46.9** | **3.8 %** | **0.39 %** | 3.01 | 0 | **1.42** |
| CUDA before | 0.929 | 1.709 | 67.8 | 9.2 % | 1.28 % | 3.08 | 0 | 1.54 |
| CUDA after | 0.924 | **1.662** | **52.6** | **3.5 %** | **0.39 %** | 3.08 | 0 | **1.39** |
| CPU (SwiftF0) before / after | 0.944 / 0.944 | 1.567 / 1.567 | 52.6 / 38.6 | 8.4 / 8.1 % | 1.22 / 0 % | 2.87 / 2.87 | 0 / 0 | 1.65 / 1.65 |

- The CPU row's audio is identical (SwiftF0); its F0 rows moved because the scorer's F0 analyzer is
  RMVPE over the whole clip, and the zero padding had put octave errors at the clip's end there too.
  So part of the GPU rows' F0 RMSE gain is the analyzer; the V/UV gain (8–9 % → 3.5–3.8 %) is almost
  all the stream.
- **Four more clips** (en, ja, uk, zh from `sherpa-onnx/clips`, 1 s lead, DirectML and CUDA): speaker
  cosine up on 8 of 8 (+0.005 to +0.036, mean +0.02), F0 RMSE down on 8 of 8, V/UV down on 6 of 8,
  MCD within ±0.03 dB, DNSMOS mixed (−0.34 to +0.26, mean −0.02; the ceilings moved as much).
- **With Quality's 20 ms lookahead** (CUDA, five clips): neutral. Speaker cosine −0.020 to +0.020
  (mean 0.00), V/UV down on 4 of 5, MCD within ±0.07 dB, DNSMOS mean −0.05.
- **Clicks:** 0 on the gated rows. The `clicks` statistic (artifacts in the join windows beyond the
  rate away from the joins plus two standard deviations) read 2–10 on five runs (the ctx512 DirectML
  run, the uk clip on both EPs, two 20 ms lookahead runs) where the artifacts away from the joins
  fell while those in the windows changed little (at 20 ms lookahead on the JFK clip: 36 → 30 events
  in all, 28 → 26 in the windows; the uk clip on DirectML: 16 → 20 in the windows). The phase-locked
  analysis (`artifact_phase`) found 0 seam clicks in every run, before and after. A listening check
  of those runs is part of the P7 campaign.
- **Cost:** no p95 regression reproduces. The gate run flagged 24 p95 rows (all under the load
  simulators or the ctx512 spec, which moved ±20 % run to run); interleaved reruns (before / after,
  then after / before) give median p95 changes of +0.6 % (CUDA H1 simulation, 7 pairs), +2.1 % (CUDA
  offline, 7 pairs, individual pairs −8.0 to +8.8 %), −0.4 % (CUDA with every lever, H1 simulation,
  7 pairs), −3.2 % (DirectML offline), −0.5 % (DirectML H1 simulation) and +2.4 % (ctx512 under the
  H2 load, whose first pair read +104 %, a slower GPU state).

**Lookahead on Low latency** (CUDA, RMVPE, after the fix, the JFK clip with a 4 s lead):

| Lookahead | Speaker cosine | MCD dB | F0 RMSE cents | V/UV error | DNSMOS | Engine-added latency |
|---|---|---|---|---|---|---|
| 0 | 0.931 | 1.644 | 51.4 | 5.6 % | 2.98 | 132.6 ms |
| 10 ms | 0.951 | 1.534 | 39.2 | 4.4 % | 2.96 | 142.6 ms |
| 20 ms | 0.948 | 1.333 | 119.3 | 3.3 % | 3.19 | 152.6 ms |

The 20 ms row's F0 RMSE is a few octave-error frames (gross pitch error 0.8 %). One clip; the P7
corpus decides whether Low latency takes 10 ms.

**FCPE with a 20-frame lead-in** (five clips each, the change against the 12-frame default):

| Run | Speaker cosine | MCD dB | V/UV error | DNSMOS |
|---|---|---|---|---|
| CUDA, 40 ms blocks | mean −0.009 (−0.027 to +0.015) | ±0.03 | down on 5 of 5 (−0.2 to −4.1 pt) | −0.06 to −0.18 on every clip |
| CPU Streamer, 80 ms blocks | mean +0.001 | −0.07 to +0.03 | mixed (−1.0 to +0.4 pt) | −0.05 to +0.13 |

## Where the time goes

### Per model call, by F0 method and EP

The default voice, 40 ms engine blocks on the 512-frame device block, 64 ms of extra context,
`build: direct` (the same parts as the factory, plus the component breakdown). Offline `model_ms`
per model call and the component medians (content encoder / F0 / `enc` / `flowdec`); VRAM is the
process delta after the load; CPU is the paced run (virtual clock).

| EP | F0 | levers | call p50 / p95 / p99 ms | content / F0 / enc / flowdec p50 ms | VRAM MB | CPU cores | load s |
|---|---|---|---|---|---|---|---|
| DirectML | DIO | auto | 7.8 / 9.2 / 9.8 | 3.7 / 0.1 / 0.9 / 2.7 | 484 | 0.08 | 1.9 |
| DirectML | SwiftF0 | auto | 9.4 / 10.6 / 11.1 | 3.7 / 1.7 / 0.8 / 2.8 | 484 | 0.12 | 1.9 |
| DirectML | FCPE | auto | 9.3 / 10.4 / 10.8 | 3.8 / 1.6 / 0.8 / 2.7 | 537 | 0.10 | 2.0 |
| DirectML | CREPE-tiny | auto | 9.6 / 11.0 / 11.7 | 3.9 / 1.7 / 0.9 / 2.9 | 513 | 0.09 | 1.9 |
| DirectML | RMVPE | auto | 10.7 / 12.1 / 12.8 | 3.9 / 2.6 / 0.8 / 2.6 | 861 | 0.10 | 2.7 |
| DirectML | fusion (SwiftF0 + RMVPE) | auto | 12.7 / 14.9 / 15.6 | 3.9 / 4.5 / 0.9 / 2.7 | 861 | 0.15 | 2.8 |
| DirectML | SwiftF0 | IOBinding, priority | 9.1 / 10.4 / 11.8 | 3.7 / 1.7 / 0.8 / 2.6 | 481 | 0.13 | 1.8 |
| DirectML | RMVPE | IOBinding, priority | 10.4 / 11.5 / 11.7 | 3.8 / 2.6 / 0.9 / 2.5 | 857 | 0.10 | 2.6 |
| CUDA | SwiftF0 | auto | 14.9 / 16.2 / 16.7 | 4.0 / 1.7 / 2.5 / 6.5 | 776 | 0.37 | 1.4 |
| CUDA | RMVPE | auto | 22.1 / 24.8 / 133.2 | 4.3 / 8.0 / 2.6 / 7.0 | 1310 | 0.54 | 1.9 |
| CUDA | CREPE-tiny | IOBinding, graphs, priority | 8.3 / 9.5 / 12.3 | 4.4 / 0.6 / 0.9 / 2.3 | 836 | 0.22 | 1.4 |
| CUDA | SwiftF0 | IOBinding, graphs, priority | 9.6 / 10.8 / 11.0 | 4.4 / 1.7 / 0.9 / 2.3 | 800 | 0.25 | 1.4 |
| CUDA | RMVPE | IOBinding, graphs, priority | 16.3 / 19.5 / 149.0 | 4.6 / 8.5 / 0.8 / 2.1 | 1334 | 0.44 | 2.0 |

What this says:
- **The generator is fast; the fixed per-call costs are not.** With CUDA graphs the split generator
  (`enc` + partial `flowdec`) costs 3.2 ms per 40 ms block (3.5 ms on DirectML). The content encoder
  (ContentVec over the whole context window) costs 3.7–4.6 ms per call on every EP and is the largest
  single cost; F0 adds 0.1–2.6 ms (8 ms for RMVPE on CUDA).
- **CUDA needs its levers.** Without CUDA graphs the generator costs 9 ms per call on CUDA (kernel
  launches), and under GPU load the default stream misses the budget on 2 % to 62 % of its calls
  (below). With IOBinding + graphs + the priority stream, CUDA matches DirectML (8.3–9.6 ms with
  CREPE-tiny or SwiftF0) at a quarter of a core.
- **RMVPE is the expensive F0 method.** It holds about 380 MB of VRAM (861 vs 484 MB on DirectML), and
  on CUDA it costs 8 ms per call with p99 spikes of 130–150 ms (DirectML: 2.6 ms, p99 12.8 ms). The
  engine factory did not use the pitch bank's CUDA-graph RMVPE plan in these runs (it built the pitch
  extractor without the voice's GPU levers; fixed in P3-I0: 10.9 ms per call p95 with no spikes). It
  is also the only F0 method that breaks T3.

### Block length and context

SwiftF0, 512-frame device block; per model call (offline) and the engine-added latency of the paced
run. Per-call compute is almost flat in the block length, so short blocks cost RTF, not milliseconds.

| Engine block | DirectML call p50 / p95, 64 · 512 ms context | CUDA (auto) call p50 / p95, 64 · 512 ms context | Algorithmic latency | Engine-added latency (paced) |
|---|---|---|---|---|
| 20 ms | 11.1 / 12.4 · 9.5 / 10.9 | 14.8 / 16.5 · 15.4 / 16.4 | 58.7 ms | 123.7 ms |
| 40 ms | 9.4 / 10.6 · 9.0 / 10.2 | 15.1 / 16.6 · 14.8 / 16.2 | 67.6 ms | 132.6 ms |
| 60 ms | 9.3 / 10.6 · 10.1 / 11.4 | 15.0 / 16.2 · 15.0 / 16.6 | 89.7 ms | 144.7 ms |
| 80 ms | 9.8 / 11.8 · 9.7 / 11.2 | 15.2 / 17.6 · 15.4 / 17.2 | 104.9 ms | 159.9 ms |

- **Context is free.** 512 ms of extra context costs nothing measurable over 64 ms: partial decode
  and the incremental F0 cache work as designed.
- **20 ms blocks** save 9 ms of latency at the default cushions but put the p99 RTF at about 0.6 on
  DirectML and 0.9 on CUDA without graphs: the per-call floor of about 10 ms leaves no headroom.

### Latency

The engine-added latency of the paced harness (a captured frame to its playout, the devices' own
buffers excluded) is 132.6 ms at the reference configuration, constant (p50 = p99). Of that, 67.6 ms
is the harness's algorithmic delay at 40 ms engine blocks (block accumulation, the 48 → 16 kHz and
40 → 48 kHz resamplers, SOLA with the 20 ms crossfade, priming), 35 ms the output cushion (prefill),
10 ms each the capture and playback periods, and the rest the call's compute. Compute is not the
problem; the block and the cushions are. What the stream settings buy:

| Engine block | Prefill / max queue | Crossfade | EP, F0 | Algorithmic | Engine-added p50 / p99 | H1 simulation p50 / p99 | vs ceiling: cos, V/UV, clicks |
|---|---|---|---|---|---|---|---|
| 40 ms | 35 / 70 ms | 20 ms | DirectML, RMVPE | 67.6 ms | 132.6 / 132.6 ms | 124.3 / 132.6 ms | 0.935, 8.0 %, 0 |
| 20 ms | 35 / 70 ms | 20 ms | DirectML, SwiftF0 | 58.7 ms | 123.7 / 123.7 ms | 133.7 / 217.7 ms | 0.879, 12 %, 25 |
| 20 ms | 10 / 30 ms | 10 ms | CUDA +levers, SwiftF0 | 46.2 ms | **80.6 / 80.6 ms** | 87.2 / 132.5 ms | 0.772, 15 %, 0 |
| 20 ms | 10 / 30 ms | 10 ms | DirectML, SwiftF0 | 46.2 ms | 346.6 / 449.9 ms | 363.9 / 470.6 ms | not meaningful (below) |

- **T1's 85 ms is reachable**, with 20 ms blocks and a 10 ms cushion and crossfade: CUDA with its
  levers holds 80.6 ms (one 9 ms underrun in 20 s). But 20 ms blocks cost fidelity (cosine 0.77,
  15 % voicing errors against the ceiling), and a 20 ms crossfade on 20 ms blocks clicks.
- **DirectML cannot run that shape yet.** With a 10 or 15 ms crossfade instead of the default 20 ms,
  the DirectML content encoder costs 34–49 ms per call instead of 4–5 ms (at 40 ms blocks too:
  49 ms), while the engine plan is unchanged and CUDA is unaffected: the encoder's input length
  evidently varies from call to call and DirectML re-plans every new shape. Repro: any template-voice
  spec on DirectML with `stream.crossfade_ms` of 10 or 15 and `engine.build: direct` (the
  `content_ms_*` metrics).

### Under load (T2, T7)

The reference configuration under the load simulator (`h2_load70`: a conv hammer at 70 % duty on the
same GPU; `h1_vrchat_obs`: the same plus three CPU threads at 50 % and a 3 s stall), from the final
gate run; the CUDA rows varied most between runs:

| Run | DirectML | CUDA (auto) | CUDA +levers |
|---|---|---|---|
| H2 load: call p95 / p99, calls over budget | 17.1 / 18.6 ms, 0 % | 48.0 / 55.5 ms, 46 % (other runs: 2 %, 62 %) | 27.7 / 30.7 ms, 0.4 % (the same in the other runs) |
| H2 load: engine-added latency p50 / p99 | 142.6 / 142.6 ms | 310.6 / 340.8 ms | 143.3 / 143.3 ms |
| H1 simulation: call p99, calls over budget | 18.5 ms, 0 % | 46.1 ms, 8.4 % | 34.4 ms, 0.3 % |
| H1 simulation: latency p50 / p99 | 124.3 / 132.6 ms | 140.6 / 262.6 ms | 122.6 / 143.3 ms |
| 3 s stall: blocks to recover, backlog over budget | 0, 0 ms | 0, 0 ms | 0, 0 ms |
| 3 s stall: latency back to its pre-stall median | 165 ms | 725 ms | 255 ms |

DirectML's queue barely notices a game at 70 % of the GPU; CUDA's default stream shares the GPU with
the game's work and loses blocks by the dozen, differently every run. The graphs and the priority
stream make it predictable. The budget holds everywhere: after a 3 s stall the waiting capture is
back within budget at the next wake, and the output queue trims back to the prefill (P3-E's fix of
the P1 finding).

## Fidelity against the offline ceiling (T5)

| Configuration | CAM++ cos | MCD dB | F0 RMSE cents | GPE | V/UV | clicks | DNSMOS (ceiling) | seam flux (ceiling) |
|---|---|---|---|---|---|---|---|---|
| DirectML, RMVPE, 40 ms, 64 ms context | 0.935 | 1.70 | 68.1 | 1.3 % | 8.0 % | 0 | 3.06 (3.31) | 1.53 (1.75) |
| DirectML, RMVPE, 40 ms, 512 ms context | 0.952 | 1.67 | 77.5 | 1.3 % | 7.5 % | 0 | 3.02 (3.27) | 1.27 (1.55) |
| CUDA, RMVPE, 40 ms | 0.929 | 1.72 | 48.8 | 0.9 % | 6.8 % | 0 | 3.03 (3.33) | 1.43 (1.77) |
| CPU, SwiftF0, 80 ms (Streamer) | 0.951 | 1.57 | 52.6 | 1.2 % | 8.4 % | 0 | 2.87 (3.17) | 1.65 (1.77) |
| Autotuned GPU: CUDA, SwiftF0, 60 ms, 512 ms context, IOBinding + graphs | 0.940 | 1.69 | 44.0 | 0 % | 7.3 % | 0 | 3.16 | |
| Autotuned CPU: SwiftF0, 200 ms, thread pool | 0.972 | 1.37 | 54.0 | 0.6 % | 6.6 % | 0 | 3.14 | |

- **The joins are clean.** No run at 40 ms or longer has a seam click, and the streamed output's
  seam flux ratio (1.3–1.7) is lower than the ceiling's own back-to-back joins: time-anchored noise,
  phase0 and SOLA do their job.
- **The gap is the streaming pitch track.** The streamed F0 is 41–78 cents from the offline F0 with
  6–10 % voicing disagreements; the rest follows from it (MCD 1.4–1.7 dB, DNSMOS 0.2–0.3 below the
  ceiling). It narrows with the block: at 200 ms the voice reaches a cosine of 0.972. The F0 cache
  computes each frame's F0 from a short window when it first appears; 20 ms of lookahead (P3-I0,
  the Quality preset) brings the cosine to 0.975 and the V/UV error to 3.5 % on CUDA.
- **The cosine target is strict.** CAM++ scores two sentences of one speaker at 0.918 (below), so
  0.97 between two renditions of one sentence is a high bar.

## Component measurements (P3 workstreams)

These come from the workstreams' dev-rig tests (debug test builds on the same RTX 4090 unless a row
says otherwise; ONNX Runtime's share of each number is representative, Rust-side costs are inflated
by the debug build). They isolate one part of the engine; the tables above measure the whole engine.

### Engine core: context scaling and joins (P3-C1)

`engine::tests::sve_context_scaling_bench` (default voice, v2 40k, 40 ms blocks, all four
algorithmic levers on, 150 blocks): block p95 from 64 ms to 512 ms of extra context.

| Content encoder | DirectML | CUDA |
|---|---|---|
| stand-in | 17.89 → 18.43 ms (+3 %) | 33.10 → 32.00 ms (−3 %) |
| ContentVec (fp16 on the GPU) | 33.2 → 36.4 ms (+10 %) | 43.5 → 37.9 ms (−13 %) |

With the F0 cache off, the growth is 13–56 % at 512 ms and 3–4× at 2 s. With partial decode off,
flowdec grows from 3.6 to 6.6 ms (DirectML) and from 124 to 1060 ms (CPU) at 2 s of context; with
it on it stays at 3.0–3.3 ms and 142 ms. F0 frames analysed per tick: 24 with the cache, 44 / 56 /
204 without it (64 ms / 512 ms / 2 s of context).

`engine::tests::sve_render_default_voice` (DirectML, two speech clips): the normalized
cross-correlation of the SOLA overlap at the chosen join (1.0 would be a perfect overlap).

| Configuration | en | ja |
|---|---|---|
| legacy paths (all four levers off) | 0.692 | 0.686 |
| all levers but anchored noise | 0.691 | 0.686 |
| all levers but phase0 | 0.744 | 0.738 |
| all four levers | 0.767 | 0.756 |
| all levers + 10 ms lookahead | 0.834 | 0.827 |
| all levers + 20 ms lookahead | 0.878 | 0.875 |
| all levers + 40 ms lookahead | 0.915 | 0.921 |
| legacy paths + 20 ms lookahead | 0.779 | 0.776 |

Partial decode against the full-decode tail on the default voice: CPU / DirectML fp32 1.1e-6,
DirectML / CUDA fp16 2.4e-4, CUDA fp32 with TF32 1.1e-4 (ADR 0004 gates: 1e-4 fp32, 5e-3 fp16).

### GPU layer: session-plan levers (P3-C2)

`gpu::tests::gpu_lever_bench` (default voice, 40 ms block, 64 ms extra context, partial decode;
`enc` + `flowdec` per tick, 600 ticks; flowdec fp16).

| EP | Levers | Idle p50 / p95 / p99 ms | 70 % GPU load p50 / p95 / p99 ms | Host allocations per tick | Cold load |
|---|---|---|---|---|---|
| CUDA | legacy | 12.17 / 16.09 / 26.07 | 17.51 / 21.19 / 24.77 | 38 | 261 ms |
| CUDA | IOBinding | 12.67 / 25.99 / 29.41 | 17.65 / 22.06 / 25.94 | 0 | 515 ms |
| CUDA | IOBinding + CUDA graphs | 3.29 / 4.01 / 4.25 | 4.98 / 6.67 / 7.89 | 0 | 610 ms |
| CUDA | IOBinding + priority stream | 12.08 / 13.60 / 15.34 | 17.01 / 19.82 / 22.21 | 0 | 270 ms |
| CUDA | all three | 3.37 / 4.00 / 4.18 | 5.10 / 7.51 / 8.53 | 0 | 558 ms |
| DirectML | legacy | 4.17 / 4.94 / 5.12 | 5.54 / 8.90 / 9.30 | 38 | 1843 ms |
| DirectML | IOBinding | 4.13 / 4.83 / 5.14 | 5.33 / 8.76 / 9.18 | 0 | 1414 ms |
| DirectML | IOBinding + high-priority queue | 3.93 / 4.59 / 4.80 | 5.07 / 8.85 / 9.37 | 0 | 759 ms |

- CUDA graphs are what makes CUDA fast (3.3 ms instead of 12 ms, and 6.7 ms p95 instead of 21 ms at
  70 % load); IOBinding alone makes the tick allocation-free but does not speed CUDA up (its one
  idle p95 of 26 ms is an outlier the other runs do not repeat). The priority stream is in effect
  (priority −5 vs 0) but gains nothing against the in-process load simulator.
- Process VRAM: CUDA context 356 MB, enc + flowdec 524 MB in total with every lever (502 MB legacy);
  DirectML about 653 MB.
- Accuracy against CPU fp32 (every lever bit-identical to the legacy path): `auto` precision (fp16 +
  TF32) relative L2 3.8e-3 on CUDA, 9.2e-3 on DirectML; `fp32` (TF32 off) 4.8e-6 and 1.7e-5. TF32 off
  costs 2.3× on CUDA (3.43 → 7.85 ms p50).

### Pitch bank, retrieval and post stage (P3-D)

Pitch A/B (`f0_fusion::tests::pitch_ab_report`: 27 synthetic clips × 6 s with exact truth, speech and
singing under noise, music bleed, a missing fundamental and creak; 40 ms blocks through the
production F0 cache). GPE = gross pitch error (> 20 %), VDE = voicing decision error.

| Extractor (EP) | Offline GPE / VDE | Streaming GPE / VDE | Streaming + 20 ms lookahead GPE / VDE |
|---|---|---|---|
| Fusion SwiftF0 + RMVPE (CUDA) | 0.09 % / 1.15 % | 0.35 % / 10.92 % | 0.29 % / 6.11 % |
| SwiftF0 (CPU) | 0.21 % / 12.44 % | 0.45 % / 21.98 % | 0.43 % / 16.73 % |
| RMVPE (CUDA) | 4.23 % / 3.34 % | 4.61 % / 11.54 % | 4.03 % / 7.91 % |
| Fusion SwiftF0 + FCPE (CPU) | 1.04 % / 6.63 % | 2.03 % / 8.55 % | |
| FCPE (CPU) | 14.53 % / 8.62 % | 12.69 % / 10.39 % | |
| CREPE tiny (CUDA) | 11.33 % / 14.01 % | 12.18 % / 12.75 % | |
| DIO | 6.00 % / 13.18 % | 5.81 % / 12.47 % | |

On DirectML, fusion streams at 0.24 % GPE / 10.8 % VDE. SwiftF0's end alignment (front padding to
whole 256-sample frames) took its streaming GPE from 0.72 % to 0.45 %. The missed voicing of the
newest frames of each tick is what lookahead removes (fusion 23 % → 8 %).

Per-tick costs: RMVPE 8.0–9.5 ms on CUDA, 3.25–3.4 ms as the pitch bank's CUDA-graph plan, 2.4–2.8 ms
as the DirectML plan (5.5 ms before), 13.8 ms on the CPU (24-frame slice); the post stage with every
control on 0.42 ms p95 at 40 kHz (0.87 ms at 48 kHz), gate + limiter 0.014 ms. GPU kNN index: 88 MB
(fp32) or 44 MB (fp16) for 30 000 × 768 rows.

Real voice (MIT default voice, `en-jfk.wav`, Low latency, 40 ms blocks, against the offline ceiling;
`Sigil.exe --headless bench`):

| F0 (EP) | Lookahead | Speaker cosine | MCD dB | F0 RMSE cents | V/UV error | DNSMOS OVRL | Clicks |
|---|---|---|---|---|---|---|---|
| RMVPE (CUDA) | 0 | 0.947 | 1.70 | 58.3 | 5.3 % | 3.08 | 0 |
| RMVPE (CUDA) | 20 ms | **0.969** | **1.41** | **37.7** | **1.8 %** | **3.25** | 0 |
| Fusion (CUDA) | 0 | 0.917 | 1.71 | 64.1 | 4.8 % | 3.01 | 0 |
| Fusion (CUDA) | 20 ms | 0.961 | 1.41 | 51.5 | 2.1 % | 3.20 | 0 |
| RMVPE (DirectML) | 0 | 0.941 | 1.66 | 67.4 | 7.7 % | 3.11 | 0 |
| Fusion (DirectML) | 0 | 0.925 | 1.73 | 65.8 | 7.7 % | 3.02 | 0 |
| FCPE (CUDA rung; FCPE on the GPU) | 0 | 0.938 | 1.70 | 103 | 8.2 % | 2.93 | 0 |
| SwiftF0 (CUDA rung; SwiftF0 on the CPU) | 0 | 0.892 | 1.78 | 44.8 | 9.0 % | 2.60 | 0 |

20 ms of lookahead brings the stream to within 0.001 of T5's cosine target and meets its V/UV row
(≤ 3 %); MCD and F0 RMSE stay over their targets. This is why Quality and Studio-TRT use 20 ms (P3-I0)
and keep RMVPE: on this voice RMVPE with lookahead beats fusion with lookahead, while fusion's
advantage shows on the synthetic octave traps.

### Runtime loop and denoisers (P3-E)

- Stall recovery (stub-smoke paced run): after a 3 s stall the standing latency is back to 74.4 ms
  within 95 ms (before P3-E: 94.4 ms, never recovered in the 2 s window); backlog over budget 0 ms,
  recovery 0 blocks; the denoiser-off end-to-end latency is unchanged (74.37 ms).
- Denoisers (real models, debug build): GTCRN SI-SDR 5.1 → 13.2 dB at 1.6 ms per 16 ms hop (about
  10 % of a core), 16 ms of overlap-add plus up to one hop; DPDFNet SI-SDR 5.0 → 16.3 dB at 2.6 ms per
  10 ms hop (about 26 % of a core), about 50 ms plus hop fill.

## The autotuner

Plan acceptance: *the autotuner returns a valid preset on CPU and GPU*. Both runs below are full
sweeps (`quick: false`) of the default voice on the final code: the request's EPs, the synthetic
tuning voice for the F0 ranking (it has an exact F0) and the JFK clip for timing.

| | GPU (DirectML and CUDA offered) | CPU |
|---|---|---|
| Chosen | CUDA, SwiftF0, 60 ms blocks, 512 ms context, IOBinding + CUDA graphs | CPU, SwiftF0, 200 ms blocks, 64 ms context, the default thread pool (`rvc_preset: low_latency`, `worker_priority: auto`) |
| p99 RTF (limit 0.5) | 0.47: met | 0.64: not met; the fastest configuration found (nothing up to 200 ms reaches 0.5 with this voice) |
| Per-call p95 | 21.5 ms (9.9 ms at 40 ms) | 125 ms |
| Engine-added latency p50 / p99 | 144.7 / 144.7 ms | 363.3 / 380.6 ms |
| Fidelity vs the ceiling | cos 0.940, MCD 1.69 dB, F0 44 cents, V/UV 7.3 %, 0 clicks | cos 0.972, MCD 1.37 dB, F0 54 cents, V/UV 6.6 %, 0 clicks |
| Measured | 37 configurations in 103 s | 21 configurations in 114 s |

How it got there:
- **GPU:** SwiftF0 ranks first on the synthetic voice (no gross and no voicing errors; RMVPE and FCPE
  0.5 % voicing errors, DIO 1.4 %) and meets the limit at 40 ms on both EPs; 20 ms blocks miss it
  (RTF 0.62 on DirectML, 0.75 on CUDA); the context is free up to 512 ms. The levers are measured
  under the H2 load: on CUDA the base configuration's p99 RTF of 0.69 falls to 0.45 with IOBinding +
  graphs (kept) and the priority stream adds nothing on top; on DirectML IOBinding lowers it from
  0.37 to 0.34 (kept). CUDA with graphs then ties DirectML on latency (132.6 ms) at a lower RTF (0.26
  vs 0.27) and wins. The fidelity check moves the block to 60 ms, which cuts the T5 shortfall by a
  third (80 ms does not beat it by a quarter).
- **CPU:** the thread stage measures one thread (RTF 3.37), two (3.03) and the pool (1.20). Nothing
  meets the limit at 80 ms, so the most accurate F0 method within 20 % of the fastest one stays
  (SwiftF0: 1.25 vs DIO's 1.17), and the block sweep continues past 100 ms (1.03) to 120 ms (0.92),
  160 ms (0.77) and 200 ms (0.64). The result keeps up with 36 % headroom, at the latency a 200 ms
  block costs.
- **Between runs** on this shared desktop the GPU choice moved among near-equal options (CUDA +
  SwiftF0 at 60 or 80 ms, once DirectML + FCPE at 40 ms when a whole SwiftF0 measurement ran twice
  as slow; that run led to re-measuring a candidate that misses the limit while its median call meets
  it). Every run returned a working preset: p99 RTF 0.14–0.47, 132.6–159.9 ms, cosine 0.94–0.955.
  The CPU choice was stable: 200 ms and the thread pool in every run (RTF 0.63–0.68), with SwiftF0
  except once, before the 20 % slack, when a slow SwiftF0 sample let FCPE through.
- **What a tune measures with** (since the P3-F review): every candidate runs with the voice
  changer's own settings besides the swept knobs (`TuneConditions`: device rates and block,
  crossfade, cushion, budget, worker priority, retrieval, voice controls, F0 range, denoiser,
  lookahead, lever settings). The runs above used the voice changer's defaults, which is what the
  conditions are on a fresh install. The GPU levers are written only when the lever stage measured
  them; TensorRT-RTX counts as the CUDA rung's accelerator (it could never win before); a pitch
  model that falls back drops only its F0 method; and `voice_changer_get_autotune` reports `stale`
  when a setting the tune measured with has changed since.

## Competitors (T6)

The head to head needs the competitors' outputs of the same voice and clip (manual, below). Until
then, the published per-chunk figures are reference rows. They come from other PCs, so they bound
rather than rank:

| System | Per-chunk compute p95 | Chunk | Compute / chunk | GPU |
|---|---|---|---|---|
| vc-rs (TensorRT, `trtexec`) | 13.9 ms | 100 ms | 0.139 | not stated |
| rvc-realtime | 15.7 ms | 60 ms | 0.262 | RTX 3060 Ti |
| SVE, DirectML, RMVPE, IOBinding + priority | 13.6 ms | 100 ms | 0.136 | RTX 4090 |
| SVE, DirectML, RMVPE, IOBinding + priority | 15.9 ms | 60 ms | 0.265 | RTX 4090 |
| SVE, CUDA + levers, SwiftF0 | 10.6 ms | 80 ms | 0.132 | RTX 4090 |
| SVE, CUDA + levers, SwiftF0 | 10.1 ms | 40 ms | 0.252 | RTX 4090 |

With RMVPE, SVE's per-call cost ties both references while running on a faster GPU than
rvc-realtime's, so T6's "beats both on normalized RTF" is not shown. The per-call floor (the content
encoder and F0, above) is what has to fall; the SwiftF0 rows show the direction.

## Scorers and how to read them

- **The offline ceiling** (T5) is the same voice converting the recorded 16 kHz model stream offline.
  The Sigil Voice Engine is fixed-shape, so the ceiling is a second instance with long shapes: 200 ms
  blocks, 500 ms of extra context, 80 ms of lookahead, full decode (partial decode off), converted
  back to back. Its time-anchored noise is keyed to the same absolute frames as the stream's, so the
  two renditions differ only by what streaming changes. See [The offline ceiling](#the-offline-ceiling)
  for why these shapes.
- **Placement** of the stream against the ceiling comes from the timeline (the ceiling's absolute
  start, where the input entered the model stream, the probed harness latency), not from a search.
  Before P3-F fixed this, an envelope search locked onto the wrong syllable and reported MCD
  16–55 dB and 17 "clicks" for a stream that is 1.7 dB from its ceiling with no seam clicks.
- **CAM++ calibration** (`bench/campplus`; the ignored `real_scorers_rank_like_the_export_checks`
  test): the two halves of the 5.8 s JFK clip score **0.918**, the JFK clip against a Japanese
  speaker **0.089**.
- **DNSMOS P.835** (`bench/dnsmos-p835`): the clean JFK clip scores OVRL **3.36**; with white noise
  at 0 dB SNR it scores **1.68**.
- **F0** metrics use one analyzer for both clips (offline RMVPE over the whole clip), on the ceiling's
  speech frames. They measure how far the streamed pitch contour is from the offline one, which is
  mostly the streaming pitch tracker's error (P3-D's A/B: RMVPE's voicing decision error is 3.3 %
  offline and 11.5 % in streaming mode).
- **Clicks** are impulses of the stream that the ceiling does not have, in the join windows, in
  excess of the artifact rate away from the joins (plus two standard deviations). The stub identity
  model at its own ceiling scores MCD 0.006 dB, F0 RMSE 0.07 cents, 0 clicks (`stub-p3`).
- **CER** transcribes the unconverted input, the stream and the ceiling with a Sigil STT engine
  (here Whisper tiny.en) and scores the stream and ceiling transcripts against the input's.

## The offline ceiling

Why the ceiling has these shapes (measured on the default voice, CUDA, RMVPE, the same clip):

- **Context.** With a context of one second or more, the voice mutes the start of an utterance that
  follows silence: with 1 s of context the first 200 ms of the JFK clip after 4 s of silence come out
  15–35 dB quieter than the 64 ms-context stream, and with the longest context (about 4 s) there are
  drop-outs to −68 / −75 dBFS inside the utterance. The silent part of a long window dominates what
  the content encoder and generator see. The ceiling therefore uses 500 ms, where the onset envelope
  matches the stream's to a few dB. For P3-C1: long-context presets should not exceed about 0.5 s
  until the voice handles silence-dominated windows.
- **Back-to-back blocks need lookahead.** A ceiling of 40 ms blocks with no lookahead and no
  crossfade (the stream's own shapes) scores DNSMOS 2.17 against the stream's 3.12 and a seam flux
  ratio of 4.3: time-anchored noise and phase0 keep the excitation continuous, but a 40 ms decode
  window with no future context still leaves a step at every join. With 200 ms blocks and 80 ms of
  lookahead the seam ratio is 1.7 (the stream's crossfaded joins: 1.4).
- **Full decode, cached F0.** `ceiling.exact` (default) turns partial decode off in the ceiling. It
  keeps the F0 cache on: recomputing F0 over every block's window tracks the input's pitch slightly
  better (69 vs 80 cents RMSE against the input's own F0) but makes adjacent blocks disagree at their
  joins: 531 vs 343 artifact events per minute and DNSMOS 3.19 vs 3.32.

## What closes the gaps

In order of what they buy, with the workstream that owns each piece:

1. **Done in P3-I0: the GPU levers are on for CUDA by default** (`engine::presets::auto_lever`):
   IOBinding + CUDA graphs on CUDA, IOBinding on DirectML (allocation-free ticks); the priority
   stream stays off (no gain measured in-process). CUDA per call 21.7 → 10.9 ms p95, 0 % of calls over
   budget under both load simulations.
2. **Latency (T1):** 85 ms needs 20 ms blocks with a 10 ms cushion and crossfade, which today costs
   fidelity (cosine 0.77 against the ceiling) and does not run on DirectML (next item). The engine
   has to hold quality at 20 ms (the content and F0 windows, P3-C1 / P3-D), and the voice changer can
   then offer that stream as Low latency (P3-E / P3-G).
3. **DirectML content encoder with a crossfade other than 20 ms** (P3-C1): 34–49 ms per call instead
   of 4–5 ms; its input shape must be fixed like the generator's.
4. **The content encoder** is the largest per-call cost (3.7–4.6 ms of a 9–11 ms call on every EP).
   T2's 6 ms needs it on the plan path too: an fp16 ContentVec session with IOBinding and a CUDA
   graph (P3-C1 / P3-C2).
5. **RMVPE:** done in P3-I0, the engine factory builds the pitch extractor with the voice's GPU
   levers, so RMVPE runs as its CUDA-graph plan (the 130–150 ms p99 spikes are gone). Done at the P3
   final gate: the plan's F0 accuracy (the zero log-mel padding after short calls; V/UV 9.2 → 3.5 %
   on CUDA, [P3 final gate](#p3-final-gate-pitch-track-fixes)). Still open: its VRAM (about 380 MB;
   SwiftF0 or CREPE-tiny keep a DirectML voice under T3's 650 MB), and sizing the plan for the
   incremental 24-frame call (32 mel frames) instead of the first whole-window call (64), which
   would halve its compute (the whole-window call would then need its own path).
6. **The CPU tier (T4):** with one ONNX Runtime thread (the Streamer preset) the 40 kHz HiFiGAN
   decoder alone costs about 205 ms per 80 ms block, and the voice runs 3.4× slower than real time.
   With the default thread pool, 120 ms blocks are just real time and 200 ms blocks leave headroom,
   which is what the autotuner picks. T4's 1.5 cores at RTF 0.6 needs a cheaper CPU generator (a
   smaller decoder or int8 weights, P3-A / P3-C1), not more threads.
7. **Fidelity (T5):** 20 ms of lookahead (the `lookahead_ms` parameter, P3-I0) meets the cosine row
   and nearly the V/UV row on CUDA; it needs the DirectML content encoder fixed (item 3) to run there.
   The streaming pitch track is most of the remaining gap (F0 RMSE).
8. **Cold load (T9):** 2.7–3.2 s on DirectML is session creation for four graphs; build them in
   parallel or cache ONNX Runtime's optimized models (P3-C2).
9. **Diagnostics:** done in P3-I0, the pitch bank returns the EP outcome of RMVPE and FCPE too, and
   a bring-your-own ONNX voice lists its generator's outcome first.

## Reproducing

1. Build a release executable: `cargo build --release --manifest-path src-tauri/Cargo.toml`. Keep
   `sigil.exe` next to `DirectML.dll`, the `onnxruntime_providers_*.dll` files and
   `resources/vc-templates/` (as in `src-tauri/target/release/`).
2. Make an app-data folder for the benchmark (`SIGIL_APP_DATA_DIR`) and install the models with the
   store, one at a time: `sigil.exe --headless fetch vc/voice-default`, `vc/pitch-rmvpe`,
   `vc/pitch-swiftf0`, `vc/pitch-crepe-tiny`, `bench/campplus`, `bench/dnsmos-p835` and
   `runtime/nvidia-cuda13` for CUDA; `vc/embedder-contentvec` and `vc/pitch-fcpe` are Sigil exports
   (unpublished: `SIGIL_LOCAL_ASSETS`). Import the voice with
   `sigil.exe --headless vc-import-worker job.json`, where `job.json` is `{"version": 1, "source":
   ".../default.pth", "out_dir": "<app data>/voice-changer/models/default", "smoke": true, "name":
   "MIT default voice"}` (the app's Import does the same).
3. Run the P3 profile:

   ```powershell
   pwsh scripts/bench/run-perf-gate.ps1 -Profile p3 -Exe <folder>\sigil.exe -AppData <app data> `
       -Slot <app data>\voice-changer\models\default -Clip C:\clips\speech.wav -OutRoot bench-results\p3
   ```

   `gate-summary.md` lists every target row. `-Var 'LEVERS_JSON={...}'`, `-Var PITCH=swiftf0` and
   `-Var BLOCK_FRAMES=1920` try other configurations.
4. The matrices: `pwsh scripts/bench/run-matrix.ps1 -Exe ... -AppData ... -Slot ... -Clip ...
   -Eps dml,cuda -Pitch rmvpe,fcpe,swiftf0,crepe,fusion,dio -Build direct`, adding
   `-Levers io_binding=on,cuda_graphs=on,gpu_priority=on`, `-BlockMs 20,40,60,80 -ExtraMs 64,512`, or
   `-Eps cpu -Preset streamer`.
5. The autotuner: a spec with one run `{"mode": "autotune", "name": "tune", "request":
   {"execution_providers": ["dml", "cuda"]}}` on the voice's `rvc` engine, or Autotune in the app.
6. Latency and silence: paced runs with `stream.output_prefill_ms`, `output_max_queue_ms` and
   `crossfade_ms` set, and a `{"kind": "synthetic", "signal": "silence"}` input.

Run nothing else heavy on the PC meanwhile: another GPU job changes every number, which is exactly
what the contention runs measure on purpose.

## Still to run (manual)

- **H1 evidence (T1, T2, T7 under real load):** a friend with an RTX 3060 / 4060 runs the voice
  changer in VRChat while OBS records, starts a live recording from the voice changer's benchmark
  panel (`voice_changer_live_record_start`), and sends the `app_data/bench/results/live-<ms>/`
  folder (no audio, no paths). Until then the H1 rows are provisional (the `h1_vrchat_obs` load
  simulator).
- **True end-to-end latency (T1):** the e2e-loopback meter through a virtual cable (`output_device`
  = the cable into Sigil, `input_device` = the cable out) with `calibration_ms` from a direct cable
  run. The dev PC's virtual-cable endpoints are disabled, so the paced harness's engine-added latency
  stands in (it excludes the devices' own buffers).
- **Head to head (T6):** RVC WebUI realtime and Applio realtime through `scripts/vc-bench` (offline
  drivers in the user's own Python environments), w-okada VCClient through `play_record.py`, the same
  voice and clips; score with a `compare` run and the blind A/B kit (`tools/listening-test`, 5+
  listeners × 20 pairs). Neither app is installed on the dev rig, and the plan keeps competitor
  installs manual.
- **Corpus (T5, T8):** the 20 consented speech clips (plus game-music bleed and singing) for the
  fidelity rows, and the user's 40-voice community corpus for the importer's T8 row.
- **TensorRT-RTX (T9):** install `runtime/tensorrt-rtx` and run a `studio_trt` spec twice (the first
  engine build, then the cache hit).
