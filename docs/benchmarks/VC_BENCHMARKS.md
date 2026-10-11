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
| Models | `vc/embedder-contentvec`, `vc/pitch-rmvpe`, `vc/pitch-fcpe`, `vc/pitch-swiftf0`, `vc/pitch-crepe-tiny`, `bench/campplus` (the legacy scorer), `bench/campplus-exact` (scorer v2; unpublished, local), `bench/dnsmos-p835` |
| Clip | `en-jfk.wav` (3.85 s of public-domain speech, 24 kHz mono; P3B-B corrected the earlier "5.8 s at 16 kHz"), resampled to the 48 kHz device rate. The plan's 20-clip consented corpus does not exist, so the P3 fidelity rows describe this one clip; scorer v2 rows (P3B) use the 18 local dev-cache clips (13 sherpa-onnx and 5 transcribe samples, ten languages), each at three offsets, as a corpus mean with a bootstrap CI ([ADR 0009](../adr/0009-vc-scorer-v2.md)). |
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
| T5 | CAM++ speaker cosine vs the offline ceiling ≥ 0.97 | DirectML 0.935; CUDA 0.924–0.929; CPU 0.951; the autotuned CPU preset (200 ms blocks) 0.972. **With 20 ms of lookahead (Quality, P3-I0): CUDA 0.975**, not demonstrated after the P3 final gate (the same shape scored 0.948); re-baselined as an advisory row under scorer v2 ([P3B](#p3b-voice-engine-optimization-p3b-i0)) | advisory; missed at the Low-latency shape |
| T5 | MCD within 0.5 dB | 1.70 dB (CUDA 1.72, CPU 1.57). **P3 final gate:** DirectML 1.68, CUDA 1.66 | missed |
| T5 | Voiced F0 RMSE ≤ 15 cents; V/UV error ≤ 3 % | RMVPE on DirectML 68 cents, 8.0 %; RMVPE on CUDA 49 cents, 6.8 % (68 cents, 9.2 % on its CUDA-graph plan since P3-I0); SwiftF0 on the CPU 53 cents, 8.4 %; CUDA with 20 ms lookahead 63 cents, 3.5 %. **P3 final gate (RMVPE's mel end-aligned, [below](#p3-final-gate-pitch-track-fixes)):** DirectML 47 cents, 3.8 %; CUDA 53 cents, 3.5 % (part of the RMSE gain is the scorer's own RMVPE) | missed (V/UV within 0.8 pt) |
| T5 | 0 block-boundary clicks | 0 in every run at 40 ms blocks and longer | met |
| T5 | CER of the converted speech ≤ offline + 1 pt | stream, ceiling and the unconverted input transcribe identically (Whisper tiny.en): +0 pt | met |
| T6 | Head to head: DNSMOS +0.05, GPE ≤ each, A/B ≥ 60 %, latency ≤ 0.7× | needs the competitors' outputs (`scripts/vc-bench`) | not measured |
| T6 | Normalized RTF below vc-rs and rvc-realtime, decided only from same-GPU runs (P3B) | the published rows are other GPUs ([Competitors](#competitors-t6)); SVE at vc-rs's shape on the 4090 (P3B-C, one run): CUDA call p50 / p95 / p99 7.98 / 10.98 / 11.66 ms per 100 ms (RTF p95 0.11), DirectML p95 17.09 ms | undecided (no same-GPU run) |
| T7 | Backlog never above budget + one block | 0 ms over in every run (stub, DirectML, CUDA, CPU, 3 s stalls) | met |
| T7 | Back within budget ≤ 2 blocks after a 3 s stall | 0 blocks (stub, DirectML, CUDA) | met |
| T7 | 30 min soak with 0 underruns; EP failure shown within 1 s; a returning device resumes within 2 s | P7-B: the soak ran 30 min under a game-like load with 0 underruns ([below](#p7-b-final-scorecard-of-the-shipped-defaults)); the P0 / P1 tests cover the EP report and the device watcher | met (P7-B) |
| T8 | ≥ 95 % of the 40-voice community corpus imports | needs the user's corpus | not measured |
| T8 | 200-file malformed corpus: 0 crashes or hangs | P3-B's fuzz corpus test passes | met |
| T8 | Import ≤ 10 s per voice | the MIT default voice (55 MB `.pth`): 0.84 s including the smoke inference | met |
| T8 | Every bring-your-own ONNX slot keeps working | the legacy `RvcModel` slot runs through the harness in the bench tests | met (plumbing) |
| T9 | Hot switch within one block, with a crossfade | P3-E's factory tests (functional; not timed here) | not measured |
| T9 | Cold slot load ≤ 2.5 s including warm-up / graph capture | DirectML 2.7–3.2 s; CUDA 1.9–3.1 s (1.94 s with graph capture, P3-I0 defaults); CPU 1.0–1.1 s | missed (DirectML) |
| T9 | TensorRT-RTX first build ≤ 60 s, cache hit ≤ 3 s | the `runtime/tensorrt-rtx` pack is not installed on the dev rig | not measured |
| T9 | Zero-shot slot from a 10 s reference in ≤ 5 s | Sigil-ZS: reference 1.2–1.4 s (cached per library voice afterwards) plus sessions 1.5 s on CUDA, 2.0 s on DirectML ([P5](#sigil-zs-zero-shot-p5-b)) | met |
| T10 | Sigil-ZS end to end p95 ≤ 300 ms on H1 idle, p99 ≤ 450 ms under load, speaker similarity ≥ upstream offline | H2, `sigil_low` with its 80 ms buffer: p95 252–256 ms idle, p99 248–280 ms under the H2 load simulator, similarity 0.582 vs 0.559 ([P5](#sigil-zs-zero-shot-p5-b)) | met on H2 (H1 scheduled) |
| T10 | DDSP-SVC singing over E2–D#7, gross pitch error ≤ 2 % | 0.00 % with CREPE, SwiftF0 or fusion F0 on CUDA and DirectML (RMVPE 2.36 %) ([P5](#ddsp-svc-singing-p5-a)) | met |
| T10 | TTS through the voice plays intact at any length, mic ducked | covered by the P0 / P2 / P3-E tests | met (tests) |

In one paragraph: the engine's real-time machinery is sound (0 clicks, the budget and stall
recovery hold under every load, 0.01 cores when silent, context is free, DirectML shrugs off a game
at 70 % of the GPU), and it runs every F0 method within about 10–15 ms per 40 ms block. What misses
is the absolute speed of the per-call fixed costs (T2), latency at the reference block (T1), VRAM
with RMVPE (T3), the CPU generator (T4) and the streaming pitch track (T5).
[What closes the gaps](#what-closes-the-gaps) lists the fixes the numbers point to.

**P3B update (2026-10-09).** The rows above are the P3 / P5 values. P3B cut the content and F0 cost, the CUDA CPU and the cold load, and re-defined T5 and T6; its before / after rows, the targets as revised and the misses are in [P3B voice-engine optimization](#p3b-voice-engine-optimization-p3b-i0).

**P7-B update (2026-10-10).** The single final scorecard of the shipped defaults, run after the P3B commit, is in [P7-B final scorecard of the shipped defaults](#p7-b-final-scorecard-of-the-shipped-defaults): every T1-T10 row has a measured value or an explicit pending note, T7's 30-minute soak is measured (0 underruns) and every miss carries its reason. The scorecard itself changed no default. P7-I0 then applied one of its two recommendations, the CPU Streamer thread policy ([Defaults](#defaults-and-what-the-numbers-recommend)); the other, the 10 ms device block on the GPU presets, stays an opt-in until a live-device record exists. The rows below measure the build at `38f426d`, whose Streamer preset ran one thread. **The review of the scorecard corrected one claim: no CPU row is real time as shipped.** At the default 35 / 70 ms output queue the 120 ms CPU blocks lose 37-52 % of the converted audio in the benchmark's paced and game-load runs (one thread, the default pool and 6 threads alike), so the pool's and the 6-thread policy's latency and core figures describe the audio that survived ([CPU rows lose audio](#cpu-rows-lose-audio-at-the-shipped-output-queue)); the 4-6 thread policy of P7-I0 was not shown to be real time until the output queue was sized for 120 ms blocks and the rows re-run. **That was done afterwards** ([the output queue floor](#the-fix-an-output-queue-floor-for-template-voices-on-long-blocks)): the cause was the output queue's recovery trim, not the 70 ms cap, and with the floor all six CPU cells (the three configurations, paced and under the game stand-in) play 99.1-99.8 % of the converted audio (worst single pass 98.8 %) at 265-278 ms median end to end; the T1 latency target (280 ms p99 under the game) is **missed** with all of the audio played: 305 ms for the Streamer policy, 288 ms for the pool, 295 ms for an explicit 6 threads. The in-app Streamer was given the 120 ms block by itself afterwards ([The Streamer button and the automatic block](#the-streamer-button-and-the-automatic-block)): before, its Block size Auto derived a 20 ms block, which the CPU cannot keep up with.

## P7-B final scorecard of the shipped defaults

The single final scorecard of the voice changer's shipped defaults, as the user's scope change of 2026-10-09 asked for it: VRChat streamers running STT captions are the main audience and the voice changer is secondary, so this is one measurement of what ships, not a tuning round. No preset was swept, no autotuner prior was tuned and **no default was changed** (see [Defaults](#defaults-and-what-the-numbers-recommend)). Same rig as above (H2: Ryzen 7 7800X3D, RTX 4090 24 GB, driver 617.14, Windows 11 Pro 26200, Balanced power plan), 2026-10-09 to 2026-10-10. T1-T4, T7 and T9 come from `run-perf-gate.ps1 -Profile final` ([`scripts/bench/profiles/final.json`](../../scripts/bench/profiles/final.json): the P3B profile plus the app's own shape, the CPU thread-policy row, the silent rows and the soak), T5 from scorer v2 over the 18 local clips ([ADR 0009](../adr/0009-vc-scorer-v2.md)). The campaign is driven by `scripts/vc-bench/final_scorecard.ps1` and tabulated by `final_scorecard.py` ([scripts/vc-bench/README.md → The final scorecard](../../scripts/vc-bench/README.md#the-final-scorecard-p7-b)); the complete generated tables, all 67 target rows with their per-pass values, 14 rows derived from the reports (the audio each run delivered, the H1 simulation's over-budget half), the delivery table of every paced and game-load run and the load log of every step, are in [`scripts/vc-bench/final/p7b-scorecard-2026-10.md`](../../scripts/vc-bench/final/p7b-scorecard-2026-10.md).

**Method.**
- **Executable.** A release build of the tree at `38f426d` (`sigil.exe` sha256 `f1132368dc0c73a4…`, built at below-normal priority in its own target folder and copied with `DirectML.dll`, the `onnxruntime_providers_*.dll` files and `resources/` into a folder no rebuild can touch). No voice-changer code differs between that tree and the P3B commit `8baef6f`, and none changed since (the later commits touch STT, TTS, the new speech-bench tool and the main window's start-up only), so this is the P3B build with the P3B defaults. Voice: the MIT default voice; clip `en-jfk.wav`; models as in [Setup](#setup).
- **Passes.** Every timing row ran three times (the app-shape rows twice), each pass in a different order of the three execution-provider families (DirectML, CUDA, CPU), because a slow GPU or host state lasts minutes and must not land on one family only. A target is judged on the **median of its passes**, never on one pass; the cells give min..max. The GPU timing lock was held for every run, and each row started only after a quiet check.
- **GPU clocks.** An otherwise idle 4090 lowers its SM clock for work that arrives as one burst per block, and the same run reads 1.5-2x slower at 1.0-1.5 GHz than at 2.8 GHz (DirectML offline call p95: 8.9 ms at 2790 MHz, 13.8 at 1440, 15.3 at 1515). A game keeps the clocks up, so each row also gives the median over the passes that ran at or above 2384 MHz (85 % of the highest clock seen), the figure a streamer sees, with the number of such passes in brackets ("none": no pass reached it). The status is always the all-pass median; where the high-clock median would flip it, the last column says so.
- **Machine.** 54 logged steps and sub-steps (every pass, every corpus run, every session, and the 6-thread FCPE session added by the review of this scorecard), sampled every 2 s: no compiler was seen in any sample (rustc and cargo maximum 0 / 0); total CPU before a row 11-25 % (the gate waits for under 25 %); the desktop's own load (total minus Sigil's cores) averaged 11-25 % per step with a 90th percentile of at most 33 %; no step was flagged disturbed; 5 of the records exited 1: the two decision sessions whose candidate regressed (`cpu-f0`, `cpu-int8`: the gate's verdict, which is the result of a session) with the two holds that contained them, and the FCPE corpus's first attempt, which could not take its private lock (left behind by a stopped run) and was rerun (the driver now removes such a lock itself before a corpus step). The PC was in use throughout (a browser, Discord, the editor, a file indexer, Steam and, during the soak and the decision sessions, another program using CPU and GPU). In the GPU rows of passes 1 and 2 and the DirectML rows of pass 3, other processes (a browser, an RGB-lighting service) used 10-24 % of the GPU's SMs (median over a run; the video decoder up to 11 %); the full tables flag these rows ("GPU shared"), the medians stand, and the decision sessions are paired inside one hold so they cancel it.
- **Decision sessions.** The only other runs: interleaved (ABBA) A/B sessions inside this one executable, to settle the questions P3B left open ([below](#decision-sessions)); the review of the scorecard added one (`cpu-t6-f0`, run on the same executable, rig and day as the rest).

### Scorecard

Status words as above. Targets are those as revised by P3B ([plan → P3B outcomes](../SPEECH_STACK_IMPLEMENTATION_PLAN.md#p3b-outcomes-binding-amendments-for-p7)); the v1 rows of the first scorecard stay above for continuity. "Provisional" rows stand in for H1 (RTX 3060/4060 + VRChat + OBS) or H3 (6-core CPU + VRChat) through the load simulator.

| ID | Target (as revised) | Measured: median of the passes (min..max), high-clock median | Status | Why, or what it needs |
|---|---|---|---|---|
| T1 | Low latency, engine-added p99 <= 85 ms (paced, 40 ms blocks, RMVPE; 95 ms counts as achieved if the 10 ms crossfade fails its fidelity gate) | CUDA **132.6** ms; DirectML **142.6** ms (132.6..142.6) | missed | Fixed parts, CUDA at p50 (the sum is exact, residual 0 ms): capture 7.6 + block wait 2.5 + queue 0.7 + compute 3.7 + engine wait 17.6 + model 10.4 + hold 39.4 (SOLA tail, resamplers, fixed-block hold-back) + output 50.6 ms (output cushion and callback period). DirectML reads 10 ms more when the GPU clock is low (model step 17.4 against 10.4 ms; one pass at 1837 MHz read 132.6). The 10 ms crossfade stays out (it fails the T5 v2 gate); the device block is the lever left ([decision sessions](#decision-sessions)). |
| T1 | The same as the app starts the preset: `block_ms` 0, the engine block follows the 512-frame device block (rounded up to 20 ms), RMVPE, 96 ms context | CUDA **114.7** ms; DirectML **114.7** ms (2 passes) | missed | The shipped app shape is 18 ms under the reference config, still 30 ms over 85 and 20 over the 95 ms fallback. |
| T1 | Engine-added p99 <= 110 ms under the H1 simulation (VRChat + OBS load; provisional until a live H1 record) | CUDA **132.6** ms (132.6..143.3); DirectML **132.6** ms (132.6..145.3); app shape CUDA **114.7** ms | missed (provisional) | The load adds little: the budget drops stale audio instead of queueing it (p50 = p99 in most runs; the 3 s injected stall belongs to T7). The app shape misses 110 by 4.7 ms. |
| T1 | The T1 decomposition sums to the latency p50 within +-1 ms | CUDA **0** ms; DirectML **0** ms | met | Every millisecond is accounted for (the table above). |
| T1 | Streamer (CPU, 120 ms blocks, SwiftF0) p99 <= 280 ms with a game (H3 simulation; provisional until an H3 record), and the share of the converted audio the output played | one-thread preset **758.6** ms (737.9..768.6), **48 %** of the audio played; ONNX Runtime default pool **277.9** ms (270.6..285.3), **56 %** played (55.6..58.2); 6 threads, below normal **345.3** ms (310.6..465.7), **51 %** played | missed (provisional) for all three: the pool's latency number meets 280 ms only for the audio that survived | The latency is measured on the audio that was played. Under the game load the output queue trimmed 11.9 s (one thread), 26.4 s (pool) and 29.4 s (6 threads) of the 60 s run and the output ran dry for 48.8, 26.4 and 29.6 s (5,020, 2,992 and 3,309 underruns). Under the game load the pool converts in time (RTF p99 0.94, median 0 % of blocks over budget, 3.95-3.99 cores) and 6 threads do not (RTF p99 1.18 [1.13, 1.31], 7.0 % of blocks over budget [3.4, 9.9 %], 3.6 cores); both lose audio, because the 35 / 70 ms queue cuts 47-59 ms from each 120 ms block, so the T1 number of a row that "meets" 280 ms is the latency of the surviving 56 %. The benchmark's CPU rows run SwiftF0; the Streamer button applies FCPE ([T5](#t5-scorer-v2)). Detail: [CPU rows lose audio](#cpu-rows-lose-audio-at-the-shipped-output-queue). P7-I0 then changed the preset to 4-6 threads ([Defaults](#defaults-and-what-the-numbers-recommend)); an H3 live record is still pending. **After the output queue floor (re-run, [below](#re-run-with-the-floor)): the three rows play 99.5-99.6 % of the converted audio under the game load and read 305 ms (the 4-6 thread Streamer policy, 6 threads here), 288 ms (pool) and 295 ms (6 threads) p99, with every sample of the audio counted: all three missed by 8-25 ms. The 277.9 ms "met" of the pool above was never a meeting of the target.** |
| T1 | H2 end to end p50 <= 55 ms (e2e-loopback meter) | not measured | pending | Needs a virtual cable pair; the dev PC's endpoints are disabled. Steps in [Still to run](#still-to-run-manual). |
| T2a | H2 idle per-block compute p95 <= 6 ms (offline, 40 ms blocks) | CUDA **7.99** ms (7.496..8.662); DirectML **13.76** ms (8.941..15.32), at high clock 8.941 (1) | missed | CUDA reads 7.5-8.7 ms (the workstream builds of P3B read 7.4-7.6, its integrated smoke 7.2); DirectML is 8.9 ms at a full clock and 14-15 at 1.4-1.5 GHz. 6 ms is TensorRT-class work (deferred: no runtime pack). |
| T2a | The same as the app starts the preset (block_ms 0, 20 ms engine blocks), reported | CUDA **8.068** ms (7.929..8.206); DirectML **9.77** ms (9.596..9.944) (2 passes) | missed | On CUDA the 20 ms block costs the same per call as the 40 ms reference (8.07 against 7.99 ms); on DirectML it reads 9.77 ms, against 8.94 ms for the reference at a full clock. The call runs twice as often, so the worker does about twice the work per second (0.07-0.09 cores paced). |
| T2a | Per-block compute p95 on the real-time cadence (paced; the idle GPU drops its clocks between blocks), reported | CUDA **16.7** ms (13.35..28.3); DirectML **21.99** ms (14.54..25.95) | missed | No pass reached a high clock. NVIDIA's *Prefer maximum performance* for Sigil.exe removes it (Sigil never changes driver settings); a game keeps the clocks up on its own. |
| T2b | Under the in-process 70 % GPU hammer (the worst case: it shares the engine's context): p95 <= 10, p99 <= 15 ms, <= 0.1 % of blocks over budget | CUDA p95 **10.01** ms (9.93..17.6), p99 **10.63** ms (10.4..20.48), over budget 0 %; DirectML p95 **24.38** ms (12.51..27.23), at high clock 12.51 (1), p99 **25.57** ms (12.93..29.13), at high clock 12.93 (1), over budget 0 % | CUDA p99 and over-budget met, p95 missed by 0.01 ms (met at high clock); DirectML missed | The passes at 1.0-1.1 GHz (DirectML passes 1 and 3, CUDA pass 3) pull the medians up; at a high clock CUDA reads p95 9.97 / p99 10.5 ms (met) and DirectML 12.5 / 12.9 ms (p99 met, p95 2.5 ms over). |
| T2b | Under the out-of-process 70 % load (a game's own process and queue; duty held at 0.70): p95 <= 10, p99 <= 15 ms | CUDA p95 **11.07** ms (10.56..23.49), at high clock 10.81 (2), p99 **11.6** ms (11.11..26.35), at high clock 11.36 (2); DirectML p95 **24.69** ms (12.4..25.75), at high clock 19.07 (2), p99 **26.01** ms (12.72..29.33), at high clock 21.03 (2); 0 blocks over budget on both | CUDA p99 met, p95 missed; DirectML missed (over budget met) | The same clock story: the passes at 2790 MHz read CUDA 10.6-11.1 / 11.1-11.6 ms and DirectML 12.4-25.7 / 12.7-29.3 ms (a high clock is no guarantee on DirectML here); only CUDA pass 3, at 975 MHz, read 23.5 / 26.4 ms. The app shape under the same load reads CUDA 11 ms and DirectML 12.59 ms (p95). |
| T2b | H1 simulation: per-block compute p99 <= 20 ms, <= 0.1 % over budget (provisional) | p99: CUDA **10.71** ms (10.67..18.68); DirectML **13.21** ms (12.74..41.68). Blocks over budget (derived from the reports; the profile gates only the p99 here): CUDA **0 %** median (0..0.14 %), DirectML **0 %** median (0..1.19 %) | p99 met (provisional); over budget met on the median and **missed in the low-clock pass 3** (DirectML 1.19 %, CUDA 0.14 % against 0.1 %) | The profile has no over-budget target for this run, so the gate never judged that half; it is computed from the reports now ([generated tables](../../scripts/vc-bench/final/p7b-scorecard-2026-10.md), derived rows). Pass 3 ran at about 1.0-1.1 GHz with the GPU shared (DirectML p99 41.7 ms). In the app's shape (20 ms engine blocks) the 10 ms-block decision session ran DirectML at 1.3-1.6 GHz and read 30 % of blocks over budget (call p95 22 ms against the 20 ms engine block) on both sides; the gate's own app-shape rows at 2.79 GHz read 0 % ([decision sessions](#decision-sessions)). |
| T2c | Extra context 64 -> 512 ms raises the per-block p95 by <= 30 % (one executable, ABBA, 6 rounds) | DirectML +1.4 % (15.64 -> 15.91 ms; interval -2.9..+9.2 %); CUDA +1.7 % (8.98 -> 9.097 ms; interval -7.2..+11.0 %) | met | Partial decode and the F0 cache make context free. The upper bound of both intervals is far below 30 %. |
| T2d | Zero heap allocations per block on the GPU hot path | counting-allocator tests (`bench::hot_path_tests`): 0 per block on the real-time I/O path and in the generator's bound runs; the content encoder and F0 steps are not covered | met (tests, I/O and generator) | Carried from P3 / P3B (no voice-changer code changed since 8baef6f); not re-run by the scorecard. |
| T3 | One voice, Low latency <= 650 MB process delta (DirectML, RMVPE); CUDA <= 800 MB including the context (reported separately) | DirectML **853** MB; CUDA **1330** MB | missed | DirectML ledger: content 208 + encoder 154 + flowdec 101 + RMVPE 376 MB (P3B); the CUDA context is about 340 MB of the CUDA figure. The opt-in VRAM saver (32-frame RMVPE plan) reads 654 MB on DirectML, 1156 on CUDA; it costs pitch accuracy on noisy speech. The bench ledger of this build over-counts (its parts sum to 1924 MB against the 853 MB process delta), so the process delta is the number to read. |
| T3 | Cold-load peak <= steady + 150 MB | DirectML +0 MB; CUDA +0 MB (peak = steady) | met | P3B-I0 recorded +200 (DirectML) / +241 MB (CUDA) from workstream builds; this build reads none at 20 ms, and a 2 ms sampler (375 samples over the DirectML load) read the same: the peak is the steady level. |
| T3 | Each extra hot voice <= +150 MB (hot_slots = 2); its background load peaks <= +50 MB | CUDA +142 MB (load overshoot 0); DirectML +131 MB | met | Content encoder and RMVPE plan are shared; the second voice adds its generator. |
| T3 | Quality with the index resident on the GPU <= 900 MB (CUDA, synthetic 30 000-row index) | CUDA **1440** MB | missed | The CUDA context and the weight floor; the index itself is about 106 MB (fp32). One run (not repeated). |
| T4 | GPU presets <= 0.25 cores on average (paced) | CUDA **0.0411** cores; DirectML **0.0504** cores | met | The CUDA wait is hybrid (spin, then block): 0.2-0.4 cores in P3, 0.04 now. |
| T4 | Voice changer enabled and the user silent <= 0.02 cores | CUDA **0.0087** cores; DirectML **0.0091** cores; CPU Streamer **0.018** cores | met | The model is not run while silent; the CPU row has 10 % headroom only (one run each). |
| T4 | Streamer RTF p99 <= 0.6 at <= 3.5 cores (fp32, 120 ms blocks, SwiftF0) | shipped one-thread preset RTF **2.681** (2.587..3.116) at 0.9949 cores; default pool RTF **0.9054** (0.792..1.006) at 3.598 cores; 6 threads below normal RTF **0.877** (0.7461..0.9693) at 3.016 cores. Share of the converted audio played in the paced run: **48 % / 61 % / 63 %** (underruns 4,944 / 2,685 / 2,585) | missed on RTF (all three), met on cores for the preset and the 6-thread policy on compute only: none of the three delivered its audio at the shipped output queue (the floor fixes that: see the re-run) | Per call on the preset (p50 290 ms): flowdec 224, content 55, encoder 7.7 ms. At 6 threads the call is 86 ms (flowdec 61). RTF 0.6 at 120 ms needs about 4 cores at perfect scaling; the routes are longer blocks, the int8 decoder (deferred) or a thread-count setting ([What closes the gaps](#what-closes-the-gaps)). **After the output queue floor (re-run, [below](#re-run-with-the-floor)): RTF p99 0.79-0.86 offline (the compute is unchanged; still missed), 2.8-3.2 cores paced (met), and the paced run now plays 99.1-99.8 % of the converted audio with 0-0.42 s of dry output per 60 s (medians of three passes).** |
| T4 | ONNX Runtime never spins | every session is built through `ort_session` with spinning off; the CUDA wait is hybrid | met (by construction) |  |
| T5 | Fidelity against the whole-utterance ceiling, scorer v2 ([ADR 0009](../adr/0009-vc-scorer-v2.md)): identity delta >= -0.02, fed fine pitch error <= 15 cents, fed gross pitch error <= 0.3 %, V/UV error <= 3 %, envelope MCD over its floor <= 0.6 dB, 0 clicks | CUDA and DirectML (18 clips): -0.05553 / 22.85 cents / 1.422 % / 5.071 % / 0.6011 dB / 0 clicks (DirectML: one narrow-window click, `ja-long`); the CPU Streamer rows are in the T5 table below | missed (clicks met) | Identical to the P5 engine's numbers in ADR 0009: P3B changed no default that moves them (priming, continuity, lookahead and the 32-frame plan are opt-in). The content encoder's context is the largest piece of the gap. |
| T5 | The same rows for the CPU Streamer preset: as the Streamer button starts it (FCPE) and as the benchmark CPU rows run it (SwiftF0) | FCPE (9 clips): -0.04452 / 31.95 cents / 3.396 % / 7.981 % / 0.653 dB / 0 clicks; SwiftF0 (18 clips): -0.05938 / 13.16 cents / 0.1154 % / 44.89 % / 0.8269 dB / 0 clicks | missed (clicks met) | The preset runs FCPE, so the FCPE row is the one a CPU user gets. On the same 9 clips, paired per clip, FCPE has a higher speaker cosine v2 (+0.0482 [+0.0252, +0.0748]), a higher DNSMOS (+0.261 [+0.122, +0.434], higher on 9 of 9 clips), a lower V/UV error (-11.7 points [-17.5, -6.72]) and a lower envelope MCD (-0.218 dB [-0.29, -0.158]) than SwiftF0, and a worse fed fine pitch error (+19.2 cents [+13.8, +24.6]): SwiftF0 voices fewer frames, easier ones ([comparison below](#t5-scorer-v2)). A CPU ceiling is the same engine on the whole utterance, so the gap to it is the comparable figure. |
| T5 | CER of the converted speech <= offline + 1 pt | stream, ceiling and the unconverted input transcribe identically (Whisper tiny.en, one clip): +0 pt | met | Carried from P3; scorer v2 does not compute CER. |
| T6 | Head to head against w-okada VCClient, RVC WebUI realtime and Applio realtime: DNSMOS +0.05 at matched latency, GPE <= each, blind A/B >= 60 %, latency <= 0.7x; normalized RTF below vc-rs and rvc-realtime on the same GPU | no competitor is installed on the dev rig and no renditions exist | pending (manual) | Every Sigil-side tool is built and dry-run (compare v2, the live tier, the drivers, the public-corpus fetch, the listening kit). The exact steps, in order, are the [T6 checklist](../../scripts/vc-bench/README.md#t6-checklist-what-is-left-to-do-in-order-p7-b); the verdict goes into the [Competitors](#competitors-t6) table. |
| T7 | Backlog never above budget + one block; back within budget <= 2 blocks after an injected 3 s stall (H1 simulation) | DirectML and CUDA: 0 ms over, 0 blocks to recover, in all 3 passes | met | The stall costs the 297 underrun blocks it must (the output is empty for 3 s); the budget drops stale audio and playback resumes on the next block. |
| T7 | 30 minute soak under a game-like load: 0 underruns, backlog never above the budget (CUDA) | 30.0 min (168 749 blocks): **0** underruns, backlog over budget **0** ms (largest backlog 47 ms against the 300 ms budget), 0 model errors, 0.007 % of blocks over budget, call p50 / p95 / p99 23.2 / 27.3 / 29.5 ms, latency 132.6 ms throughout, worker 0.027 cores (0.044 for the process) | met | Under the game-like load of the plan: the 70 % DirectML hammer in its own process (the queue of a game) and three CPU threads at 50 % (render, game, OBS), with the desktop in use beside it (a browser, Discord and another program kept CPU and the GPU busy: other processes used up to 82 % of the SMs, the hammer included). The SM clock of the GPU sat at 1170 MHz with a P-state change, so the per-call times are about three times the full-clock figures; the run proves stability, not speed. |
| T7 | EP failure surfaced within 1 s; a returning device resumes within 2 s | covered by the P0 / P1 tests (EP report, device watcher); not a benchmark row | met (tests) | Carried from P0 / P1. |
| T8 | >= 95 % of the 40-voice community corpus imports; the 200-file malformed corpus: 0 crashes or hangs; import <= 10 s; every BYO ONNX slot works | fuzz corpus test passes; the MIT default voice (55 MB `.pth`) imports in 0.84 s with its smoke inference; the legacy slot runs through the harness in the bench tests; the 40-voice corpus does not exist | met (fuzz, import time, BYO); the 95 % row pending | Carried from P3 (no importer change since). The 95 % row needs the user's corpus ([Still to run](#still-to-run-manual)). |
| T9 | Cold slot load <= 2.5 s including warm-up and graph capture (5 fresh processes, median) | CUDA **972.4** ms; DirectML **1190** ms; CPU Streamer **900.2** ms; Quality + 20 ms lookahead, CUDA **1468** ms | met | Parallel cold start; 99 % of the load is attributed to named stages (DirectML 0.992, CUDA 0.990). |
| T9 | Hot switch within one block, with a crossfade | P3-E's factory tests (functional); not timed | not measured | Needs a live device with two hot slots; listed in [Still to run](#still-to-run-manual). |
| T9 | TensorRT-RTX first build <= 60 s, cache hit <= 3 s | the `runtime/tensorrt-rtx` pack is not installed on the dev rig | not measured | Opt-in (Studio preset); steps in [Still to run](#still-to-run-manual). |
| T9 | Zero-shot slot from a 10 s reference <= 5 s | Sigil-ZS: reference 1.2-1.4 s plus sessions 1.5 s (CUDA), 2.0 s (DirectML) (P5) | met (P5) | Not re-measured: no code change since. |
| T10 | Sigil-ZS end to end p95 <= 300 ms idle, p99 <= 450 ms under load; DDSP-SVC singing over E2-D#7 with gross pitch error <= 2 %; TTS through the voice plays intact | H2: Sigil-ZS p95 252-256 ms idle, p99 248-280 ms under the load simulator, similarity 0.582 against 0.559; DDSP-SVC 0.00 % with CREPE, SwiftF0 or fusion F0 (RMVPE 2.36 %); TTS by the P0 / P2 / P3-E tests | met on H2 (P5; H1 scheduled) | Not re-run: no family code changed since P5. Not part of the shipped defaults of the main audience. |

In one paragraph: the engine's real-time machinery holds on the main audience's shape. A 30-minute soak under a game-like load ran with 0 underruns, an injected 3 s stall recovers in 0 blocks, a voice that is enabled and silent costs 0.009 cores, the engine takes 0.04-0.05 cores while converting on a GPU, a second hot voice adds 131-142 MB, a cold load takes 1.0-1.2 s, and 512 ms of context costs nothing measurable. What misses is what P3B left: latency (engine-added p99 115-133 ms against 85), per-call compute at an idle GPU clock (T2a) and, on DirectML, under a game's load (T2b: 12.5 ms at a full clock against 10), VRAM with RMVPE (T3), the CPU preset (T4: one thread cannot convert 120 ms in 120 ms; and at the default 35 / 70 ms output queue no CPU row, the pool and 6 threads included, played more than 63 % of the audio it converted, as the median of its passes; the output queue floor added afterwards fixed that, all six CPU cells now play 99.1-99.8 %, and the latency target under a game, 280 ms, is still missed at 305 / 288 / 295 ms) and T5 v2's identity and pitch rows. Most spreads between passes are the GPU's clock state, not the engine.

### T5 (scorer v2)

| Row (limit) | CUDA, RMVPE (18 clips) | DirectML, RMVPE (18 clips) | CPU Streamer, SwiftF0 (18 clips) | CPU Streamer, FCPE (9 clips) |
|---|---|---|---|---|
| identity delta (speaker similarity to the target set, stream minus whole ceiling) (limit -0.02) | -0.0555 [-0.077, -0.0346] (missed) | -0.0534 [-0.0794, -0.0262] (missed) | -0.0594 [-0.0792, -0.0398] (missed) | -0.0445 [-0.0909, 0.00335] (missed) |
| fed fine pitch error (cents) (limit 15.0) | 22.9 [20.6, 25.4] (missed) | 22.9 [20.6, 25.4] (missed) | 13.2 [11.4, 15.2] (met) | 32.0 [25.8, 37.7] (missed) |
| fed gross pitch error (pooled) (limit 0.003) | 0.0142 [0.0142, 0.0142] (missed) | 0.0142 [0.0142, 0.0142] (missed) | 0.00115 [0.00115, 0.00115] (met) | 0.034 [0.034, 0.034] (missed) |
| fed V/UV error (limit 0.03) | 0.0507 [0.0344, 0.0703] (missed) | 0.0507 [0.0344, 0.0703] (missed) | 0.449 [0.38, 0.519] (missed) | 0.0798 [0.0514, 0.116] (missed) |
| envelope MCD over its floor (dB) (limit 0.6) | 0.601 [0.545, 0.653] (missed) | 0.602 [0.546, 0.653] (missed) | 0.827 [0.771, 0.884] (missed) | 0.653 [0.592, 0.715] (missed) |
| clicks (binomial, model-delay windows) (limit 0) | 0 [0, 0] (met) | 0 [0, 0] (met) | 0 [0, 0] (met) | 0 [0, 0] (met) |
| clicks (narrow crossfade windows) (limit 0) | 0 [0, 0] (met) | 0.0556 [0, 0.167] (missed) | 0 [0, 0] (met) | 0 [0, 0] (met) |

| Mean over the clips [95 % CI] | CUDA, RMVPE | DirectML, RMVPE | CPU Streamer, SwiftF0 | CPU Streamer, FCPE |
|---|---|---|---|---|
| speaker cosine v2 | 0.886 [0.873, 0.899] | 0.88 [0.861, 0.897] | 0.855 [0.834, 0.876] | 0.892 [0.86, 0.915] |
| DNSMOS OVRL | 2.78 [2.63, 2.92] | 2.79 [2.64, 2.92] | 2.4 [2.21, 2.59] | 2.8 [2.52, 3.04] |
| DNSMOS OVRL of the whole-utterance ceiling | 3.08 [2.94, 3.21] | 3.07 [2.92, 3.19] | 2.6 [2.37, 2.82] | 3.13 [2.94, 3.29] |
| MCD (dB) | 1.91 [1.83, 1.99] | 1.91 [1.83, 1.99] | 2.18 [2.09, 2.27] | 1.88 [1.77, 2] |
| envelope MCD (dB) | 2.03 [1.97, 2.08] | 2.03 [1.97, 2.08] | 2.28 [2.22, 2.35] | 2.01 [1.93, 2.1] |
| output F0 RMSE (cents) | 43.8 [34.4, 55.6] | 43.0 [34.6, 53.5] | 135 [83.4, 211] | 105 [43.3, 178] |
| output V/UV error | 0.0468 [0.0418, 0.0524] | 0.0476 [0.0418, 0.0541] | 0.166 [0.137, 0.197] | 0.0501 [0.0396, 0.0608] |
| output gross pitch error | 0.0024 [0.000885, 0.00413] | 0.00182 [0.000479, 0.00334] | 0.037 [0.0129, 0.0696] | 0.0173 [0.00329, 0.0346] |

Reading the CPU columns: the benchmark's CPU rows use SwiftF0, the engine's `auto` F0 for the CPU, while the **Streamer button applies FCPE** (`presets.ts`; it is also the default `pitch_extractor`), so the FCPE column is the one a CPU user sees; it ran on the first 9 of the 18 clips (2-3 minutes a clip on the CPU, so the hold on the shared machine stayed short). SwiftF0 voices only part of the speech frames its whole-clip analyzer sees (fed V/UV error 0.21-0.70 per clip against 0.006-0.17 with RMVPE), which is why its "fed fine pitch error" is a mean over fewer, easier frames and is not comparable with the GPU columns. A CPU ceiling is the same engine converting the whole utterance, so the identity delta and the DNSMOS gap to the ceiling are the comparable rows.

SwiftF0 against FCPE on the same 9 clips (CPU Streamer preset, paired per clip, bootstrap 95 % interval of the difference, FCPE minus SwiftF0):

| Metric | SwiftF0 | FCPE | Difference | Clips where FCPE is higher |
|---|---|---|---|---|
| speaker cosine v2 | 0.8433 | 0.8915 | +0.0482 [+0.0252, +0.0748] | 8 of 9 |
| DNSMOS OVRL | 2.543 | 2.804 | +0.261 [+0.122, +0.434] | 9 of 9 |
| envelope MCD (dB) | 2.231 | 2.012 | -0.218 [-0.29, -0.158] | 0 of 9 |
| output V/UV error (points) | 16.71 | 5.015 | -11.7 [-17.5, -6.72] | 0 of 9 |
| fed V/UV error (points) | 41.93 | 7.981 | -34 [-46.7, -22.6] | 0 of 9 |
| fed fine pitch error (cents) | 12.76 | 31.95 | +19.2 [+13.8, +24.6] | 9 of 9 |
| fed gross pitch error, mean per clip (points) | 0.3175 | 3.415 | +3.1 [+0.884, +5.66] | 6 of 9 |
| output F0 RMSE (cents) | 95.91 | 104.8 | +8.94 [-43, +71.7] | 3 of 9 |

FCPE sounds closer to the target voice and is voiced where the speech is (a V/UV error of 5 % against 17 %), at the price of more frames with a wrong pitch: its fed fine pitch error is 32 cents against 13, and 3.4 % of its fed frames (pooled over the clips) are more than 20 % off the analyzer's pitch against 0.14 % of SwiftF0's (octave slips, over the fewer frames SwiftF0 voices). The audible output F0 error is not different within the noise (the output F0 RMSE interval spans zero). Neither meets the T5 pitch rows; the preset keeps FCPE because the audio scores better on every automatic audio metric measured here (a blind listening check of the two is still open) and the F0 stage costs 12 ms against 2 ms (decision sessions below).

### Decision sessions

Each session is an ABBA run of this one executable against itself (A the shipped default, B one switch), 4 rounds (6 for the context), both sides inside one lock hold, so the GPU's clock state and the desktop's load cancel in the pairs. Intervals are 95 %; "neutral" means the interval spans zero or the gate's resolution. The gate exits 1 when a candidate REGRESSED a gated row; for a decision session that is the answer, not a failure (the driver counts a session as done when this run wrote its paired summary and the summary lists every round of every spec; a summary left by an earlier run is deleted before the gate starts), so the two sessions whose candidate regressed under the first driver (`cpu-int8`, `cpu-f0`) show `exit 1` in the step log of the campaign although they ran to their end; `cpu-t6-f0` regressed too and is recorded with exit 0 by the fixed driver.

| Question (B) | Result | Decision |
|---|---|---|
| **Context 64 → 512 ms** (`EXTRA_MS=512`; offline, 6 rounds) | per-call p95 DirectML 15.64 → 15.91 ms (+1.4 %, [-2.9, +9.2]), CUDA 8.98 → 9.10 ms (+1.7 %, [-7.2, +11.0]); p50 +1.7 % / +0.4 % | T2 context row **met** (the interval's upper bound is under the 30 % limit on both EPs) |
| **Device block 480 frames** (10 ms; the `device_block = model_divisor` shape at 48 kHz) **against the shipped 512**, the app shape (`block_ms` 0), CUDA | paced latency p50 114.7 → 96.0 ms (-15.9 %, [-20.3, -7.3]), p99 114.7 → 101.0 (-16.3 %, [-21.0, -7.2]); out-of-process 70 % load p50 119.7 → 106.0 (-12.0 %, [-19.9, -3.1]); H1 simulation p50 125.4 → 108.5 (-8.4 %, [-19.5, -1.7]); call p95 and worker CPU neutral. **Underruns** (A → B, median of 4 rounds): paced 1.5 → 0.5, out-of-process load 1.0 → 0.5, H1 simulation 302.5 → 297.5 (297 of each are the injected 3 s stall): too few events to resolve (intervals -97..+282 %, -97..+615 %, -2.4..+0.1 %), so the verdicts are neutral and say nothing either way. **Blocks over budget** (A → B): paced 1.2 → 1.9 %, load 7.7 → 7.2 %, H1 simulation 3.4 → 1.4 %; the gate judges these on a linear scale with a 0.25 floor, which makes them "neutral" whatever they do, so read the medians: all are far above the 0.1 % limit on both sides, because rounds 3 and 4 ran at 1.2-1.4 GHz (call p95 18.7-21.4 ms against the 20 ms engine block) while rounds 1 and 2 at 2.8 GHz read 8.7-10.9 ms and 0 % | **improves T1** on CUDA; underruns and over-budget unresolved; stays opt-in for now ([below](#defaults-and-what-the-numbers-recommend)) |
| the same, DirectML | paced p50 124.7 → 113.5 ms (-9.0 %, [-13.0, -5.1]), p99 -9.5 %; out-of-process load p50 125.7 → 116.0 (-7.7 %); H1 simulation p50 126.0 → 116.0 (-8.0 %); call p95 neutral (21.6 → 21.3 ms paced). **Underruns** (A → B): paced 2 → 0 (neutral, too few), **out-of-process 70 % load 26.5 → 6.5 (IMPROVED, -75 %, [-87, -55])**, H1 simulation 338 → 315 (IMPROVED, -5.1 %, [-8.7, -2.2]; 297 of each are the stall). **Blocks over budget** (A → B): paced 8.0 → 7.8 %, load 31.8 → 32.7 %, H1 simulation 30.4 → 31.6 %: neutral, and large. **This session ran DirectML at a low GPU clock (1.3-1.6 GHz in every round)**, where the shipped 512-frame shape already overruns the 20 ms engine block (call p95 21.6-22.3 ms on both sides); the 10 ms device block does not change that. At 2.79 GHz the gate's own app-shape rows read call p95 9.8 ms paced and 0 % over budget | improves T1 on DirectML, and the underruns under load |
| the same, CPU Streamer | p50 647.9 → 635.2 ms (-2.0 %); RTF p99 2.51 → 2.51 | no information: the preset is not real time, so the latency is the backlog |
| **Shared DirectML queue** (`SIGIL_VC_DML_SHARED=1`; offline and 70 % out-of-process load) | call p95 offline 9.57 → 10.90 ms (neutral, [-26, +40]); under the load 17.29 → 18.79 ms (neutral); content stage p95 under load 6.07 → 8.25 ms (+11.9 %, [-12.8, +64.2], inconclusive); 13 GPU-state flags | **stays off** (as the P3B review decided; nothing here argues for it) |
| **CPU content encoder fp32 instead of int8** (`SIGIL_VC_INT8=0`), Streamer preset (1 thread) and 6 threads below normal | content stage p50 55.1 → 63.3 ms (+15.0 %, [14.1, 15.8]) on one thread, 17.5 → 19.9 ms (+13.3 %) on six; whole-call p95 neutral on one thread (303 → 311 ms), +7.7 % ([2.7, 10.8]) on six; RTF p99 on six 0.85 → 0.93 (+10.2 %, [1.4, 22.6]) | **int8 stays first** on VNNI CPUs: the fp32 encoder's CPU cost is +15 % of the content stage and +8-10 % of the call once the generator has threads |
| **CPU F0: FCPE** (what the Streamer button applies) **against SwiftF0** (the benchmark's CPU F0), Streamer preset | F0 stage p50 2.05 → 12.07 ms (+489 %, regressed), whole call p95 301 → 309 ms (+1.5 %, [-17, +15], neutral), RTF p99 2.56 → 2.61 (neutral), cores 0.993 both | the F0 stage costs 10 ms more but the call does not notice **on one thread** (it is 3 % of a 300 ms call; the 6-thread policy is the next row); on the same 9 clips FCPE gives the better audio (higher speaker cosine and DNSMOS, V/UV error 5.0 against 16.7 %) and the worse fed pitch (32 against 13 cents), so **FCPE stays** the Streamer F0 ([T5](#t5-scorer-v2)) |
| **CPU F0: FCPE against SwiftF0 on the 6-thread policy** (`cpu-t6-f0`: 6 threads below normal, 120 ms blocks; offline, paced 45 s and the H3 game simulation 45 s; added by the review of this scorecard) | F0 stage p50 2.05 → 4.36 ms offline, 2.06 → 4.39 paced, 2.08 → 5.45 under the game (+113 %, +113 %, +162 %, regressed): FCPE's cost shrinks with the threads (12.1 ms on one). Whole call p95 96.6 → 105.1 ms offline (+8.6 %, [-8.2, +26.1], inconclusive), 103.0 → 107.2 paced (+6.4 %, inconclusive), 116.3 → 126.8 under the game (+6.2 %, [+0.9, +13.7], regressed). RTF p99 0.886 → 0.970 offline (+8.5 %, inconclusive), 0.948 → 0.965 paced (neutral), 1.148 → 1.228 under the game (+5.0 %, neutral). Cores 2.95 → 3.12 paced (+6.1 %, regressed), 3.55 → 3.72 under the game. Blocks over budget under the game 4.0 → 9.1 % (regressed, interval 0.03..298 %). Latency p99 under the game 323 → 403 ms (+15 %, [-8, +51], inconclusive). Audio played 53 → 51 % (game) and 64 → 62 % (paced); trimmed 21.1 → 22.0 s and 16.2 → 17.1 s of the 45 s runs. 17 GPU / host state flags; 18 rows with an interval wider than ±5 % after 4 rounds | FCPE costs the 6-thread policy about 8-10 ms per call (+6-9 %) and 0.1-0.2 cores, on a margin that is thin already: under the game load the SwiftF0 side is over its budget at RTF p99 1.15 and FCPE adds 5 % to it. **FCPE stays** the Streamer F0 for the audio it gives ([T5](#t5-scorer-v2)) with this cost recorded; it does not move the thread-policy decision, which the unsized output queue ([above](#cpu-rows-lose-audio-at-the-shipped-output-queue)) and the game-load miss decide first |

### CPU rows lose audio at the shipped output queue

The tables above read the CPU rows on call time (RTF), latency and cores. The review of the scorecard found that none of them says whether the audio comes out: every paced and game-load run also counts the frames the engine converted and the frames the output queue played, and the latency of a run is measured on the played frames only. The benchmark's harness uses the production output queue (`OutputPlayback`) with the voice changer's default 35 ms prefill and 70 ms cap, which the Streamer button does not change, against 120 ms engine blocks. "Played" is the frames the output played over the frames the engine handed to the output queue (the one-thread preset also dropped 35 s of input at the 300 ms latency budget because it could not keep up, so its figure mixes the queue's trimming with that overload; the pool rows dropped none and the 6-thread rows at most 0.5 s). Medians of the three passes, 60 s runs (the campaign's `delivery` table, [generated tables](../../scripts/vc-bench/final/p7b-scorecard-2026-10.md)). This table is the P7-B measurement at the default 35 / 70 ms; the [diagnosis](#diagnosis-the-recovery-never-ends), the [fix](#the-fix-an-output-queue-floor-for-template-voices-on-long-blocks) and the [re-run](#re-run-with-the-floor) follow it:

| Run | Played | Trimmed by the output queue | Output dry | Underruns | RTF p99 | Latency p99, surviving audio |
|---|---|---|---|---|---|---|
| one thread (the `38f426d` preset), paced | 48.5 % | 12.7 s | 48.0 s | 4,944 | 2.62 | 709 ms |
| one thread, game load | 48.5 % | 11.9 s | 48.8 s | 5,020 | 2.82 | 759 ms |
| default pool, paced | 61.0 % | 23.3 s | 23.4 s | 2,685 | 0.92 | 275 ms |
| default pool, game load | 55.9 % | 26.4 s | 26.4 s | 2,992 | 0.94 | 278 ms |
| 6 threads below normal, paced | 62.7 % | 22.3 s | 22.3 s | 2,585 | 0.90 | 275 ms |
| 6 threads below normal, game load | 50.6 % | 29.4 s | 29.6 s | 3,309 | 1.18 | 345 ms |
| for comparison: CUDA and DirectML, 40 ms blocks, paced | 99.9 % | 0 | 0 | 0 | 0.69 / 0.72 | 133 / 143 ms |

What this means:
- **At 35 / 70 ms no CPU row was real time as shipped (fixed below), the pool and the 6-thread policy included.** The pool converts a block in time (RTF p99 0.92-0.94) and still plays 56-61 % of the audio; its "277.9 ms, met" is the latency of the audio that survived. The T1 row above no longer reads "met" for it, and the T4 row says its core figure is compute only.
- Per converted block the queue cuts 47 ms (pool, paced), 45 ms (6 threads, paced), 53 and 59 ms (pool and 6 threads under the game) and 62 ms (one thread) from each 120 ms block. **It is not the 70 ms cap.** The review compared these figures with the difference between the block and the cap and suggested the cap as the cause; the mechanism, found afterwards, is the output queue's recovery trim to the ring target ([diagnosis](#diagnosis-the-recovery-never-ends)), and raising only the cap changes nothing.
- The GPU rows are not affected: they play 99.7-99.9 % in every run (the 0.1-0.3 % is the end of the run and, in the H1 simulation, the catch-up after the injected 3 s stall).
- A user would hear 37-52 % less output than the engine produced (48-63 % played), whatever the RTF says. The T4 compute rows (RTF, cores) still describe the compute honestly; they are not a statement about the output.

**Follow-up (done).** The review asked for the output queue to be sized for the Streamer's 120 ms blocks and for the three CPU rows to be re-run against the 98 % `played_ratio` requirement. The diagnosis below corrected the proposed lever (the cap) and the fix and the re-run follow it.

### Diagnosis: the recovery never ends

The review's first reading was that the 70 ms cap cuts each block (the 45-62 ms lost per block is of the order of the block minus the cap). That is wrong. The loss is the output queue's **recovery trim**:

- The worker is synchronous. With 120 ms engine blocks on 512-frame device chunks (10.67 ms), eleven of every twelve ticks cost about 0.01 ms and push one chunk each; the twelfth runs the model call (on 6 threads: p50 74-81 ms with the PC idle and 88-98 ms under the game stand-in, p99 90-106 and 98-130 ms, maximum 98-130 and 105-147 ms, in the final campaign's reports) and pushes nothing, so the device drains its ring for that long.
- The ring holds at most the ring target (the gate level, the last chunk, one capture burst and 1.3 ms: 57 ms at 35 / 70), which is shorter than the call. The device runs dry, `push_interleaved` starts a recovery, and while recovering every push trims the queue (deque and ring) to the ring target, which discards the catch-up burst the worker pushes when the call returns (about 45 ms of each 120 ms block). The next call dries the ring again, so the recovery never ends.
- All of the trimmed audio is recovery trim; the cap rule and the drift rule trim nothing. **Raising only `output_max_queue_ms` therefore does nothing.** Replaying the recorded tick timelines of the 36 CPU runs of the P7-B campaign through a Python port of the output queue (a scratch analysis, not part of the repository: its figures are replayed, not run, and cannot be re-derived from the repository; the committed simulations in `playback.rs` pin the mechanism and the knee instead) reproduces each run's recorded played ratio, trims, dry time and latency, and the same replay at 35 / 170 and at 35 / 300 gives exactly the numbers of 35 / 70 (62.7 %, 50.6 %, 61.0 % and 55.9 % for 6 threads paced and game, pool paced and game). The prefill is the lever, because the gate level and the ring target follow it. Replayed on 120 ms blocks (6 threads, paced / game): 60 / 120 plays 84 / 71 %, the **Stable** preset's 80 / 150 plays 98 / 88 % (not enough under a game), and only 100 / 125 and above reached 98 % in every recorded pass (worst 98.3 %).
- A second constraint: the prefill gate must open before the first model call. A prefill above (pushes before the first call) x chunk opens it after that call, and the queue then stands 20-40 ms higher for the whole run (replayed: a 120 ms prefill gives a 299 ms median where 110 ms gives 278). At 120 ms blocks on 512-frame chunks the first call comes after 11 pushes (117.3 ms), at 40 ms blocks after 3.

Committed tests pin it: `playback.rs` `a_burst_workers_loss_is_a_recovery_trim_that_only_the_prefill_removes` (a burst worker loses 37.7 % of its audio at 35 / 70 and at 35 / 600 alike, every frame of it a recovery trim, and 0.1 % at 110 / 136), `bench/paced.rs` `a_burst_engine_loses_audio_at_the_default_buffer_and_plays_it_with_the_floor` (a stub with 120 ms blocks and 90 ms calls through the paced run's output queue: 62.3 % played and 22.6 s trimmed at 35 / 70, 99.8 % and nothing trimmed with the floor) and the unit tests of the rule.

### The fix: an output queue floor for template voices on long blocks

`voice_changer/block_buffer.rs` derives, per voice, the smallest output buffer that covers a block's model call. The voice carries it as `CreatedEngine::min_output_buffer_ms`; the worker (`port_limits`) and the benchmark (`EngineInfo::output_buffer_ms`) play with the larger of it and the stored values, so `sve-cpu`, `sve-cpu-pool`, `sve-cpu-t6`, the H3 run and the autotuner size the queue exactly like the app. For an engine block B, a device chunk D and `n = ceil((B - 0.4) / D) - 1` pushes before the first call:

`prefill = min(ceil5(0.9 B), floor5(n D - 2))` (no floor when that is not above the 35 ms default), `queue = ceil(prefill + 2 D + 4)`.

The first term covers the call (an assumed real-time factor of 0.9; this rig measures 0.74-0.89 idle), the second keeps the gate before the first call. On 512-frame chunks at 48 kHz:

| Engine block | 40 ms or less | 60 ms | 80 ms | 100 ms | 120 ms | 160 ms | 200 ms |
|---|---|---|---|---|---|---|---|
| Prefill / max queue | none | 50 / 76 | 70 / 96 | 90 / 116 | **110 / 136** | 145 / 171 | 180 / 206 |

- **Who gets it.** Sigil template voices whose generator really runs on the CPU (the requested rung, or a GPU rung that fell back) with an engine block of about 50 ms or more on the run's device chunk. **Nobody else:** every GPU rung at every block (the 40 ms reference shape has n = 3 and a 30 ms limit, below the default; the app's 20 ms block likewise), bring-your-own ONNX voices, the P5 voice families (they keep their own minimums), Classic, ONNX and sidecar modes. `SIGIL_VC_BLOCK_BUFFER=0` switches it off.
- **Stored values** are never rewritten, and a larger stored value always wins. The two named buffers under Playback feel (**Low latency** 35 / 70 ms, **Stable** 80 / 150 ms) are templates for short blocks and are raised to the floor (35 / 70 becomes 110 / 136, **Stable** 80 / 150 becomes 110 / 150). A buffer the user typed (the preset reads **Custom**) is the user's choice and is **kept** however far under the floor it is, until the run proves that it cannot play: the worker watches the model calls (`block_buffer::StoredBufferProof`), and once the median of the last 33 calls (at least 16 seen, about two seconds of 120 ms blocks) is longer than the stored prefill, the output runs dry in more than half of the blocks and the floor applies from then on for that value (latched; the log says so; 60 / 65 becomes 110 / 136, 100 / 200 becomes 110 / 200). The rule the earlier text of this section gave, "a stored value that cannot play the voice's blocks is raised", is therefore what the code does now; before, every value under the floor was raised, a faster CPU's hand-tuned 70 / 100 included. The criterion is the knee of the burst-worker simulation (`playback::tests::the_proof_criterion_is_the_knee_of_the_burst_worker`): across 70-110 ms calls and five buffers, wherever it says the buffer cannot play the simulated output loses at least 1 % of the audio and runs dry in at least 100 callbacks of the minute, and wherever it does not at least 98 % is played (80 / 150 against 78 ms calls loses 1.3 %: borderline values just under the call are left to the user). It also agrees with the replay above (60 / 120 plays 84 / 71 % paced / game, **Stable**'s 80 / 150 98 / 88 %, 100 / 125 at least 98.3 % in every pass; replayed, not run). The benchmark has no Custom buffer (its stream's numbers stand for the stock one), so it always plays with the floor.
- **In the app** `block_ms` defaults to 0, which derives a 20 ms block from the 512-frame chunk: no floor, and not a block the CPU can keep up with. The Streamer on the processor therefore takes the 120 ms block by itself when Block size is on Auto, and gets this floor with it ([The Streamer button and the automatic block](#the-streamer-button-and-the-automatic-block)); a Block size you set or Autotune's pick wins.

### Re-run with the floor

The three CPU rows of the `final` profile re-run on the release build of the working tree on top of `0ce30a6` (sha256 `1b41d6f73a2f…`; `sve-cpu` now measures the 4-6 thread Streamer policy of P7-I0, which is 6 threads on this 8-core rig), three passes in rotated order, 60 s paced and H3 game-load runs at 120 ms blocks on 512-frame chunks, SwiftF0, a quiet check before each row (total CPU under 25 %, no compiler), the machine lock held, at below-normal priority class (the P7-B campaign ran at the normal class). Besides Sigil the desktop used 1.3-1.9 cores in pass 1 and 2.0-2.2 cores in passes 2 and 3; the rows with more desktop load show more dry output. Medians of the three passes (minimum..maximum where a spread matters); the last column is the P7-B result at 35 / 70 ms for the same row, whose first two rows are the one-thread preset that P7-I0 replaced:

| Run | Played | Trimmed by the queue | Output dry | Underruns | Dropped input | RTF p99 | Latency p50 / p99 | Before (35 / 70): played, p50 / p99 |
|---|---|---|---|---|---|---|---|---|
| Streamer policy (6 threads), paced | 99.63 % (99.17..99.80) | 0.12 s | 111 ms (8..386) | 17 | 0 ms | 0.88 | 268 / 275 ms | 48.5 %, 649 / 709 ms |
| Streamer policy (6 threads), game load | 99.58 % (99.44..99.61) | 0.22 s | 229 ms (141..241) | 35 | 0 ms | 1.04 | 275 / 305 ms | 48.5 %, 688 / 759 ms |
| default pool, paced | 99.82 % (98.82..99.82) | 0.00 s | 0 ms (0..598) | 0 | 0 ms | 0.79 | 273 / 273 ms | 61.0 %, 255 / 275 ms |
| default pool, game load | 99.58 % (99.44..99.67) | 0.15 s | 145 ms (89..229) | 27 | 0 ms | 0.94 | 273 / 288 ms | 55.9 %, 265 / 278 ms |
| 6 threads, paced | 99.12 % (98.90..99.82) | 0.44 s | 420 ms (0..549) | 74 | 0 ms | 0.88 | 265 / 273 ms | 62.7 %, 253 / 275 ms |
| 6 threads, game load | 99.49 % (99.33..99.80) | 0.19 s | 195 ms (8..295) | 29 | 0 ms | 1.04 | 278 / 295 ms | 50.6 %, 275 / 345 ms |
| for comparison: CUDA, 40 ms blocks, paced (four A/B samples, two per build) | 99.92 % | 0 s | 0 ms | 0 | 0 ms | 0.36 | 133 / 133 ms | 99.92 %, 133 / 133 ms |

What this shows:
- **The target is reached.** Every one of the six CPU cells plays at least 99.1 % of the converted audio (the 98 % requirement of `final_scorecard.py` holds for the median and for every single pass: the worst pass is 98.82 %), the queue trims at most 0.6 s in any 60 s run, no row drops input, no model call failed, and the dry output is 0-0.42 s per run in the medians (the worst single pass 598 ms). The kill-switch control (`SIGIL_VC_BLOCK_BUFFER=0`, same build, `sve-cpu` paced) plays 65.8 % with 20.5 s trimmed and 2,402 underruns and reports no minimum buffer, so the difference is the floor and only the floor. The replay had predicted 99.4-99.8 % for these rows.
- **Compute did not change.** The offline RTF p99 medians are 0.855 (Streamer policy), 0.817 (pool) and 0.790 (6 threads), against 0.905 and 0.877 for the pool and 6 threads in P7-B, within the rig's spread; the buffer cannot change them. Under the game stand-in the 6-thread rows still go over budget in 2.4-2.6 % of the blocks (RTF p99 1.04), the pool in 0.2 % (0.94).
- **What it costs.** The queue holds 110 ms where it held 35 ms (75 ms more), and a continuous stream now leaves the engine 265-278 ms after it entered (p50), 273-305 ms at p99. The earlier 253-275 ms figures were the latency of the 49-63 % of the audio that survived. One control sample of the same build with the switch off (`sve-cpu` paced, n = 1, surviving audio only) reads 245 ms p50 against 268 ms with the floor (+23 ms): that is not the 75 ms of queue, because the 35 / 70 ms run dropped the audio that would have waited in it. A user hears a delayed voice instead of a dropped one.
- **The GPU rows do not take this path.** `engine.min_output_buffer_ms` is absent from every CUDA report, and no code path of the GPU rungs changed (the binary did). The evidence is one row: a paced-only copy of the P7-B materialized `sve-cuda` spec (so that no GPU load process starts on the PC), CUDA at 40 ms blocks, five samples (the first in pass 1, then two of this build and two of the P7-B build, alternating). Four read 99.92 % with 0 underruns and 132.6 ms p99, identical to the P7-B result (its three passes: 99.9 %, 0 underruns, 132.6 ms). The fifth, the first of the campaign, read 99.90 %, 2 underruns and 13 ms dry, latency 131 / 133 ms; its RTF p99 of 0.75 lies inside P7-B's own CUDA range (0.36, 0.69 and 0.80 over its three passes, against 0.36 in the four later samples), so it is a spread of the GPU's clock state, not a measured effect, and its two underruns remain unexplained (they did not repeat in either build). DirectML, the app's 20 ms shape and the load runs were not re-run.

**T1 and T4 restated.** The T1 target for the Streamer (engine-added latency under the H3 game stand-in, p99 at most 280 ms, provisional) was "met" for the pool in P7-B (277.9 ms) only on the audio that survived. Measured on all of the audio it is **missed**: 305 ms for the Streamer policy, 288 ms for the pool and 295 ms for 6 threads (8-25 ms over; before the floor 6 threads read 345 ms on the surviving 51 %). The T4 compute rows are unchanged (RTF p99 0.79-0.86 offline against 0.6: missed; paced cores 2.8-3.2, game 3.3-3.7), and what T4 now adds is that the audio is delivered: `played_ratio` of at least 98 % and the dry output are the first things to read in a CPU row, because the latency is measured on the played frames.

**What the floor does not do.** It does not make the Streamer real time under a game: the 6-thread policy still misses the budget in about 2.5 % of the blocks there, and a CPU whose call is longer than the block cannot be helped by any buffer (the 300 ms latency budget then drops input; the one-thread **Protect game FPS** Streamer, RTF 2.6, is the example). The benchmark's CPU rows run SwiftF0 and the Streamer button applies FCPE (+6-9 % on the call), which this campaign did not re-run, nor the 4-thread policy of a 6-core PC (RTF p99 0.96 paced): with a call longer than the 110 ms prefill the output runs dry in the slowest blocks (the trap guard keeps the prefill at the first-call limit, see `block_buffer.rs`; the simulation puts every call 10 ms over the prefill at 1.4 % of the audio, `a_call_ten_ms_past_the_prefill_costs_a_little_audio_not_all_of_it`). The headless benchmark and the replay cannot see WASAPI or Bluetooth timing, so a live listening check on a real device is still open; a longer device callback raises the standing queue by itself (`playback::tests::a_longer_device_callback_needs_no_larger_floor`: 90-110 ms calls at 110 / 136 ms on 10-40 ms callbacks, no dry callback). Faster CPUs and autotuned configurations (RTF 0.5-0.6) are over-provisioned by 30-50 ms at long blocks for the stock buffers, because the cover term assumes 0.9; a typed buffer that plays is left alone (`SIGIL_VC_BLOCK_BUFFER=0` switches the whole rule off). A hot switch between voices with different floors re-gates once.

### The Streamer button and the automatic block

The re-run above sets the 120 ms block in its specs (`block_ms` 120). The app does not: its **Block size** is on Auto (`block_ms` 0), which derives the engine block from the device chunk, 10.67 ms at the default 512 frames at 48 kHz, planned to 20 ms. That was the first thing the review of the floor found still open: with a 20 ms block the voice gets no floor (a 20 ms block is not a burst worker) and, on the processor, nothing to keep up with. Before this fix the document had no measurement at 20 ms on the processor (a straight line through the 120 ms rows, p99 call 89-107 ms on 6 threads, and the 200 ms row, about 124 ms, put a call for a 20 ms block at 45-85 ms); one headless paced run of that shape measured it (explicit `block_ms` 20, the 4-6 thread Streamer policy, 6 threads here, SwiftF0, 40 s, the machine lock held, the desktop in use): the call takes p50 57 ms and p99 76 ms for a 20 ms block, **RTF p99 3.8**, 2,699 underruns, 26.0 s of the 40 s of input dropped by the 300 ms latency budget, latency p50 399 / p99 515 ms. The old default of the in-app Streamer on the processor was therefore not a voice that works, and no output buffer can change that.

What changed: for the **Streamer** profile on the CPU rung with no block set, a derived block shorter than 60 ms becomes 120 ms (`presets::cpu_streamer_block`, called by `create_sigil` and by the benchmark's `build_sve`; the worker's `run_chunk_frames` treats it as an explicit block for `device_block = model_divisor`). The 120 ms shape then gets the floor above (110 / 136 ms), so a fresh in-app Streamer on a CPU-only PC runs the configuration this document measured: 99.1-99.8 % of the converted audio played, 265-278 ms from microphone to speaker (median), 305 ms p99 under the game stand-in. What it costs is delay, not fidelity: the block is only how much new audio each call takes, and the alternative at 20 ms is a voice that falls behind.

What it leaves alone: a Block size you set (any number), Autotune's pick, a derived block of 60 ms or more (a deliberately big processing chunk: 2,880 frames or more at 48 kHz), every other preset, and every GPU rung. `SIGIL_VC_STREAMER_BLOCK=0` restores the old rule. A bench spec that leaves `block_ms` out now measures this 120 ms block (the 2026-10 specs all set it).

Check on the final build (release build of the working tree on `0ce30a6` with this change, sha256 `0331ba98…`; the campaign above ran on `1b41d6f7…`, and the later changes leave the explicit-block paced path as it was): the `sve-cpu` spec with `block_ms` left out, paced 60 s, the machine lock held at below-normal priority. The PC was in use (about 36 % total CPU against under 25 % in the campaign), so this is one noisier run, not a new measurement of the rows above. The engine reported `min_output_buffer_ms` [110, 136], so the 120 ms block was chosen by itself; **99.3 % of the converted audio played** (2,860,544 of 2,879,488 frames), 0.29 s trimmed, 0.29 s dry (48 underruns), nothing dropped, no model error; latency p50 273 ms, p99 335 ms; calls p50 83 ms, p99 128 ms (RTF p99 1.07, against 0.88 in the quiet campaign run). The H3 game-load run was not repeated on this build. A listening check on a real device is still open.

### Defaults, and what the numbers recommend

**No default changed in P7-B.** No measurement showed a shipped default clearly wrong, and the two candidates with a clear gain needed a decision outside that workstream's files. P7-I0 decided them: item 2 is applied, item 1 is not.

1. **`device_block = model_divisor` for the GPU presets (recommended).** A 10 ms device block removes the 512-frame block's hold-back and lowers the engine-added latency by 9-16 % on both GPU execution providers with no change in compute or CPU. Underruns fall sharply on DirectML under the out-of-process load (26.5 → 6.5, IMPROVED, interval -87..-55 %) and in the H1 simulation (338 → 315); on CUDA they are too few to resolve (1.5 → 0.5, 1.0 → 0.5; [decision sessions](#decision-sessions)). The blocks over budget do not change (DirectML 8 / 32 / 30 % on both sides at the session's 1.3-1.6 GHz clock): the shipped 512-frame shape already overruns its 20 ms engine block at a low GPU clock, which the device block does not fix. P3B asked for three confirmations before flipping it: DirectML (done here), the CPU (not applicable while the preset is not real time) and a live WASAPI device (it needs a person at the PC with a real microphone and headphones, and the e2e meter needs the virtual-cable endpoints that are disabled on the dev PC). The default lives in `params.rs` (`DeviceBlockMode::default`), `contract.ts` (`VC_ENUM_DEFAULTS`) and `params_contract.json`, not in the preset tables `presets.rs` / `presets.ts` (whose settings type has no device-block entry, and whose test `hotkeys.test.ts` pins them), so flipping it is a change for the integrator or the voice-changer owner, best made together with a live record on a real microphone and headphones. Until then it stays the opt-in "device block" row.
2. **The CPU Streamer preset is not real time on one thread and needs a thread policy (applied by P7-I0).** One ONNX Runtime thread converts a 120 ms block in 290 ms (RTF p99 2.7; about 4,900 underrun blocks in a 60 s paced run; latency p50 649 ms, p99 709 ms paced and 759 ms under the game load) on this Ryzen 7 7800X3D. The ONNX Runtime default pool reaches RTF 0.91 and p99 278 ms at 3.6 cores; 6 threads at below-normal priority reach RTF 0.88 at 3.0 cores (p99 345 ms with the game load). Neither meets T4's 0.6 and both take cores a game wants, so the choice is a product decision (a thread-count setting, a longer block, or sending users to the autotuner), made together with an H3 live record. The fix is in `ort_session.rs`'s Streamer profile, not in the preset tables. **P7-I0 applied it:** `ort_session::vc_threads` gives the Streamer preset on the CPU rung `streamer_threads_for(physical cores)` threads, all physical cores but two within 4 to 6, on the worker priority policy's pool (below normal under `Automatic`); the worker and the benchmark both read it. P3B-G's sweep at 120 ms with a below-normal pool is the evidence for the count (RTF p99 offline / paced, cores paced): 2 threads 1.39 / 1.43 (1.8), 3 threads 1.14 / 1.08 (2.5), **4 threads 0.89 / 0.96 (2.6)**, **6 threads 0.83 / 0.74 (2.8)**, the default pool of 8 0.80 / 0.90 (3.3). Four is the smallest count that keeps up in both runs, but with a margin of only a few percent, so an 8-core PC (this rig) takes six; a 6-core PC takes four and leaves two cores to the game. A 4- or 5-core PC also takes four (the count never drops below four: three threads are not real time), so it keeps no spare core (4 cores) or one (5 cores); that was not measured, and the autotuner's longer block is its way to a lighter load. On the dev rig the six threads read RTF p99 0.88 (0.75..0.97) in the idle offline run and 3.0 cores in the paced run (this scorecard's `sve-cpu-t6` row); **under the H3 game stand-in the same configuration reads RTF p99 1.13 / 1.18 / 1.31 (3.4, 7.0 and 9.9 % of blocks over budget) and a latency p99 of 345 ms (311..466), which misses the 280 ms H3 target that the default pool meets (278 ms)**, so it is not real time under a game load on this rig. The default pool (8 threads here) meets the H3 latency and RTF (278 ms, 0.94) at 3.95-3.99 cores under the game, against 3.6 cores for six threads: the two extra threads are what the game-load margin costs on this CPU, so the weighing is cores against margin, not one free option against another, and the H3 record has to settle it. The benchmark's CPU rows run **SwiftF0**; the Streamer button applies FCPE, so the shipped preset has less margin than these rows: the paired `cpu-t6-f0` session ([decision sessions](#decision-sessions)) measures FCPE on six threads at +2.3 ms in the F0 stage offline and +3.4 ms under the game, +6-9 % in the call p95, +5 % in the RTF p99 (1.15 → 1.23 under the game) and 4.0 → 9.1 % of blocks over budget under the game; the one-thread cost of +10 ms in the F0 stage does not carry over to six threads. `Protect game FPS` keeps the one-thread profile (the user asked for the least load, and it is not real time), a GPU rung keeps it, and `SIGIL_VC_STREAMER_THREADS=1` restores the old behaviour; the thread count never changes the audio (rule R6). Not re-measured after the change (no campaign, no H3 record): the `sve-cpu` row of the `final` profile now measures the policy, so the next scorecard has the figure, and a slower CPU than this rig needs the autotuner's longer block. **The review also found that none of the three configurations plays its audio at the shipped 35 / 70 ms output queue** ([CPU rows lose audio](#cpu-rows-lose-audio-at-the-shipped-output-queue): 48 %, 56-61 % and 51-63 % played), so the thread policy removed the compute bottleneck but did not by itself make the rows real time. The output queue was sized for 120 ms blocks afterwards (the floor) and the three CPU rows re-run ([Re-run with the floor](#re-run-with-the-floor)): they now play 99.1-99.8 % of the converted audio in the paced and game-load runs; the game-load RTF p99 (1.04 on 6 threads) and the T1 latency (288-305 ms against 280) are still missed.
3. **The shared DirectML queue and the fp32 CPU encoder stay off / second** (decided above); the 10 ms Low-latency lookahead and the 10 ms crossfade were not re-run (P3B's numbers stand: +10 ms for +0.03 speaker similarity; the crossfade fails the T5 v2 gate).

### What the numbers mean for the main audience

A streamer running STT captions who uses the voice changer now and then: it costs nothing while silent (0.009 cores, no GPU inference), 0.04-0.05 cores while converting on a GPU, 0.85-1.3 GB of VRAM (853 MB DirectML, 1330 MB CUDA including its context), loads in about a second, and held 30 minutes under a game-like load (a 70 % GPU hammer in another process and three busy CPU threads, with the desktop in use) without one underrun. The converted voice leaves the engine 115-133 ms after it entered on the reference shapes (96-116 ms with the 10 ms device block), plus the audio devices' own buffers. On a CPU-only PC the Streamer preset of the scorecard's build could not keep up on one thread; P7-I0 gives it 4-6 threads (item 2 above), which makes the conversion itself fast enough on this rig, and with the output queue floor of 110 / 136 ms for 120 ms blocks the benchmark plays 99.1-99.8 % of the converted audio at 265-278 ms median end to end ([Re-run with the floor](#re-run-with-the-floor)); the in-app Streamer does not get that floor at its default 20 ms block (a 120 ms block comes from the benchmark, Autotune or a manual block size); the per-PC autotuner finds a longer block for a slower CPU (P3: 380.6 ms idle at 200 ms blocks, measured at the old 35 / 70 ms output queue; a 200 ms CPU block now holds a 180 / 206 ms floor, 145 ms more standing queue, and 80 ms blocks 70 / 96 ms, 35 ms more; neither was re-run, so expect more than that figure).

### Misses and what each needs

| Miss | Reason | Next step |
|---|---|---|
| T1 115-133 ms against 85 (95) | The fixed parts sum to it (hold 39 ms, output cushion 51 ms, engine wait 18 ms, model 10 ms); the 10 ms crossfade fails T5 v2 | the 10 ms device block (96-113 ms), then a model that sounds as good with less right context |
| T2a CUDA 8.0 ms, DirectML 8.9-15 ms against 6 | per-call fixed costs; the idle GPU lowers its clock | TensorRT-class work (deferred); NVIDIA *Prefer maximum performance* for Sigil.exe |
| T2b DirectML 12.5 ms at a full clock, CUDA 10.0 against 10 | the 70 % load | a merged DirectML `enc` + `flowdec` graph (deferred); an H1 live record |
| T3 853 / 1330 MB against 650 / 800 | weights plus the CUDA context | the opt-in VRAM saver (654 / 1156 MB) once its pitch cost is decided |
| T4 Streamer RTF 2.7 (1 thread, the scorecard's build), 0.88-0.91 (6 threads, pool) against 0.6 | the generator needs about 4 cores for 0.6; under the game load 6 threads read RTF 1.13-1.31 | item 2 above: the preset now runs 4-6 threads (0.74-0.89 idle on this rig); 0.6 needs a longer block (200 ms reads 0.62) |
| CPU Streamer rows play 37-52 % less audio than they convert (one thread 48 %, pool 56-61 %, 6 threads 51-63 % played) | the default 35 / 70 ms output queue against 120 ms blocks; the latency and core figures describe the surviving audio | **Fixed afterwards:** the output queue floor for template voices on long blocks (110 / 136 ms at 120 ms blocks; [diagnosis and re-run](#diagnosis-the-recovery-never-ends)): all six CPU cells now play 99.1-99.8 % (worst single pass 98.8 %) |
| T5 v2 identity -0.055, pitch 22.9 cents, V/UV 5.1 %, envelope 0.60 dB | the content encoder's context; the streaming F0 tracker | the gated levers (lookahead, filled F0) and the blind check |
| T5 v2 on the CPU Streamer preset (FCPE, 9 clips): identity -0.045, fed pitch 32 cents, 3.4 % gross, V/UV 8.0 %, envelope 0.65 dB | the same content-encoder gap, plus FCPE's octave slips and its weaker pitch track against RMVPE's | a better CPU F0 (P3-D's synthetic pitch A/B had the SwiftF0 + FCPE fusion at 2.0 % streaming gross pitch error against FCPE's 12.7 %, at the cost of running both trackers; it was not run through this corpus) and the blind check |
| T1 end to end, T6, T8 (95 % row), TRT-RTX, hot-switch timing, H1/H3 | need hardware, installs or files the dev rig lacks | [Still to run](#still-to-run-manual) |

## P3B voice-engine optimization (P3B-I0)

P3B was designed after P3 missed several targets. It measured first, fixed what the measurements named, re-defined the targets that could not be met as written (accepted by the user: [plan → P3B outcomes](../SPEECH_STACK_IMPLEMENTATION_PLAN.md#p3b-outcomes-binding-amendments-for-p7)) and rebuilt the benchmark so a before/after claim means something ([ADR 0009](../adr/0009-vc-scorer-v2.md), [ADR 0010](../adr/0010-sve-optimization-p3b.md), [README → Interleaved A/B](README.md#interleaved-ab-p3b-a-the-evidence-a-beforeafter-claim-needs)). Same rig (H2: Ryzen 7 7800X3D, RTX 4090), 2026-10-01 to 2026-10-09.

**How to read the numbers.** Every before/after below is an ABBA-interleaved comparison inside one executable (a switch), with the paired interval; absolute figures come from one workstream's executable and are provisional (some runs overlapped other builds; each workstream's README records the background load). By the user's scope change of 2026-10-09 (VRChat streamers using STT captions are the main audience, the voice changer is secondary), P3B-I0 ran **no final regression scorecard**: it ran one functional smoke per execution provider on the integrated build ([below](#integrated-smoke-p3b-i0)), and **P7-B runs the single final scorecard of the shipped defaults** on a quiet machine. Rows P3B did not re-measure say so.

### What was wrong, and what changed

The per-call cost was overhead, not arithmetic: about 1 ms of GPU math against 9.6 ms measured, roughly a thousand small launches, four serial CPU-GPU syncs per call, ContentVec still an eager dynamic-shape session (51-60 % of the p95 under load: CUDA 12.2 of 20.4 ms, DirectML 8.3 of 16.3 ms), an RMVPE plan sized for the first whole-window call, F0 serial behind the content encoder and a CUDA wait that spun a core. The shipped fixes are in [ADR 0010](../adr/0010-sve-optimization-p3b.md#decision); the levers that were measured but trade fidelity, memory or safety for speed are switches and are listed there with the reason.

### Scorecard v2 (targets as revised; v1 rows above stay for continuity)

"Before" is the P5 baseline (exe-s0, behaviour identical to P5) unless the row says otherwise; "after" is the workstream's interleaved measurement of the shipped default (unless it says *opt-in*).

| Target (revised) | Before | After (P3B) | Status |
|---|---|---|---|
| **T1** Low latency, engine-added p99 ≤ 85 ms (95 ms reported as achieved if the 10 ms crossfade fails its gate), paced, 40 ms blocks | 132.6 ms (p50 = p99). Decomposition at p50: capture 7.6, block wait 2.5, queue 1.0, compute 4.2, engine wait 17.0, model 11.8, hold-back 38.6, output 49.9 ms | shipped defaults unchanged, 132.6 ms. Opt-in 10 ms device block: **115.2 ms** (-17.4 ms, -13.1 %, CI -14.8..-13.1 %; 0 underruns in 7 pairs, a 30-minute soak and one out-of-process-load run); probed algorithmic latency 67.6 → 60.2 ms. Adaptive cushion (experimental) read 97.2 ms on a calm PC once but underran under the 70 % loads. The 10 ms crossfade (83-86 ms) fails the T5 v2 gate (cosine v2 -0.040, DNSMOS -0.16) | **missed** (95 ms not reached by a default; 115.2 ms opt-in) |
| **T1** Streamer (CPU), p99 with a game on H3 ≤ 280 ms at 120 ms blocks | 784.9 ms (one thread) | shipped Streamer preset (one ONNX Runtime thread) 719 ms; the default thread pool 280.6 ms at 4.7 cores offline | **missed** for the preset, borderline for the pool (provisional until an H3 live record) |
| **T2a** H2 idle per-call p95 ≤ 6 ms | CUDA 10.9, DirectML 11.1 ms | offline p95 CUDA **7.4-7.6**, DirectML **9.0-9.1** (content p50 4.9 → 1.8 / 4.3 → 1.5 ms; F0 overlapped); the integrated build's smoke read CUDA 7.2 and DirectML 7.8 ms (one run, [below](#integrated-smoke-p3b-i0)) | **missed** (expected 6.5-7.5 / 7-8.5: CUDA at the edge, DirectML just above) |
| **T2b** under the 70 % load: p95 ≤ 10, p99 ≤ 15 ms, ≤ 0.1 % over budget | CUDA 19.2-25.5, DirectML 17.1 ms | in-process hammer: CUDA p95 **9.35-9.60**, p99 9.96; DirectML 11.6 / 11.9. Out-of-process load (duty 0.70 held): CUDA 10.2 / 10.6, DirectML 11.3 / 11.6; H1 simulation CUDA 9.7, DirectML 11.5. 0 % over budget everywhere | **met** on CUDA in process, **missed** by 0.2-1.6 ms elsewhere |
| **T3** DirectML one voice with RMVPE ≤ 650 MB | 861 MB | 853 MB (clean ledger: content 208, enc 154, flowdec 101, RMVPE 376); opt-in fp16 680, VRAM saver 654 | **missed** (654 with the opt-in saver) |
| **T3** CUDA on H2 ≤ 800 MB incl. the context (reported separately) | 1312 MB (context 334-356) | 1374 MB, 1340 with the shrink default; opt-in fp16 1186, saver 1156 | **missed** (the weight floor makes 800 hard: 331 MiB of fp16 weights plus the context) |
| **T3** load-peak overshoot ≤ +150 MB | +345 MB (a second RMVPE copy) | DirectML +200, CUDA +241 MB (was +348-379 / +520-680): RMVPE no longer has a transient, what is left is the content encoder's session init | **missed** |
| **T3** extra hot voice ≤ +150 MB | +508 (DirectML), +620-660 (CUDA) | **+130 / +136 MB** with the shared RMVPE plan; second load 780 / 636 ms | **met** |
| **T3** Quality with the GPU index ≤ 900 MB | not measured | kNN on 30 000 rows: footprint CUDA 106 MB fp32 (62 MB fp16, off), DirectML 68 MB; process 1526 MB | **missed** (CUDA context) |
| **T4** GPU presets ≤ 0.25 cores; silent ≤ 0.02 | CUDA 0.40-0.63 paced; silent 0.0088 | CUDA paced **0.042** (-92.9 %), offline 0.215, under load 0.05; silent **0.0081**. DirectML 0.12 | **met** |
| **T4** Streamer (fp32) RTF p99 ≤ 0.6 at ≤ 3.5 cores, 120 ms blocks | preset RTF 2.52 (0.99 cores); pool 0.92 (4.6 cores) | pool: RTF p99 0.79-0.90 at 3.3 cores (before the arena's −5.9 %); best measured policy (6 threads below normal, arena) p50 0.69 / p99 0.74-0.83 at 2.8 cores. The arithmetic needs at least 4.0 cores at perfect scaling for 0.6 | **missed** on RTF, met on cores (opt-in policy; no thread-count setting yet) |
| **T5** identity, fed-F0 error, GPE, V/UV, envelope MCD, clicks (scorer v2) | see [ADR 0009](../adr/0009-vc-scorer-v2.md): at the P5 engine −0.054 / 22.9 cents / 1.42 % / 5.1 % / 0.601 dB over floor / 0 clicks | priming and continuity (opt-in since the P3B review: `SIGIL_VC_F0_PRIME=1`, `SIGIL_VC_F0_CONTINUITY=1`; rule R6, as they change 3 to 22 of about 5,000 fed frames on the pause clips and have no per-frame or FCPE evidence) change no per-clip median, so the shipped defaults read as before. Opt-in: lookahead 10 / 20 ms raises set-sim by +0.032 / +0.046 (env MCD −0.13 / −0.18 dB); the 32-frame plan 18.7 cents and V/UV 3.0 %; filled F0 17.8 cents and 2.3 % | **missed** (clicks met; DirectML `ja-long` has one narrow-window click to listen to) |
| **T6** head to head | tooling absent | compare v2, the live tier, the drivers and the public-corpus fetch exist and are dry-run (10 local clips); no competitor renditions exist | **undecided** (manual) |
| **T9** cold load ≤ 2.5 s | DirectML 2.7-3.2 s, CUDA 1.9-3.1 s | CUDA **1.0 s** (1837 → 1009 ms, -45 %), DirectML **1.1-1.3 s** (2504 → 1300 ms, -49 %), Quality CUDA 1485 ms, CPU 1.1 → 0.6 s; on the integrated build CUDA 967 ms and DirectML 1045 ms (5 ABBA pairs against the sequential load: -44 % / -51 %) | **met** |
| T7, T8, T10 | see the P3 / P5 rows above | not re-measured | unchanged |

### Method notes

- **The gate's resolution** (two A/A sessions, 64 rows each): 62-64 NEUTRAL, no false REGRESSED. `call_ms_p95` interval half-width 1.3-4.3 % offline and under the hammer once clock-regime-mismatched pairs are set aside, 8-13 % paced. The slow runs were GPU clock states, confirmed with NVML clock and P-state evidence (SM clock held at 1170 of 2790 MHz with reason `gpu_idle` only; in another session 2790 → 1455 MHz offline, memory clock halved, 223 → 122 W, while another process used up to 68 % of the GPU's SMs; the host probe and CPU clock stayed steady). In the manual rounds p95 also tracked host CPU utilisation (Spearman +0.94, DirectML offline), so the gate flags both.
- **A real-time cadence runs the GPU slower than offline** (about 1.1-1.8 GHz SM against 2.8 GHz): paced p95 is 14 ms against 7.6 offline. NVIDIA's *Prefer maximum performance* for Sigil.exe removes it (documented; Sigil never changes driver settings).
- **The app's EcoQoS opt-out is worth about 30 %** on the CPU path under a simulated game (5 ABBA pairs: call p50 +29 % [+24, +36], core-seconds per audio-second +29 % without it, REGRESSED). P3's headless H3 rows lacked it, so they were pessimistic by about that much.
- **Under-load share of the p95 (DML and CUDA)**: F0 52-54 % (it now overlaps content), content 33-40 %, flowdec 33-38 %, encoder 8-13 %, glue about 1 %, harness 0.19-0.28 ms; syncs per call CUDA 4 → 3 with the device chain (DirectML's plan waits are counted since P3B-G).
- **RMVPE plan size** (stream path, 27 synthetic clips with exact truth): gross pitch error 4.89 % (64 frames, P3's size), 8.23 % (32 frames), 5.67 % (32 frames filled with real audio), 4.69 % (64 frames, lead 51); under noise and creak the 32-frame plan is much worse (5 dB SNR 12.2 against 5.0 %, creak 17.2 against 7.2 %). On real speech through the engine it is better (fed fine pitch error 19.5 against 22.4 cents, V/UV 3 against 4 %). Mixed evidence, so it is the opt-in "VRAM saver"; t32 kernel pruning is exact (|Δsalience| 3.3e-7 in Python, 0.0 in Rust on the real model).
- **fp16 RMVPE** passes its pitch gate at the saver's plan on both EPs and on CUDA at the default plan; on DirectML at the default plan gross pitch error rose 4.893 → 4.950 % (+0.057 points against +0.05 allowed), so `auto` stays fp32.
- **Content encoder precision**: GPU fp16 ContentVec misses the export's feature gate at some stream lengths but is neutral downstream (|Δcosine| < 0.001, MCD ±0.007 dB); CPU int8 vs fp32: cosine −0.0026 (2 of 5 better), DNSMOS −0.04, MCD −0.012 dB, V/UV −0.006.
- **hot_slots = 2**: the content encoder is shared; the second voice's own RMVPE plan (+368-420 MB) and its transient plain session (+351-520 MB) dominated the P5 cost, which the plan-at-load change removed.
- **Cold versus warm loads** (`load_twice`): CUDA 2318-2689 → 1281-1748 ms, DirectML 2973-3438 → 1620-1722 ms before the load changes; the cold loads above are after them.
- **Aligned SOLA tail** (480 against 481 samples at a 20 ms crossfade): set-sim +0.0002, cosine v2 −0.0141 [−0.0247, −0.0052], so it stays legacy; **SOLA search** narrowing lowered fidelity (4 ms window: set-sim −0.012, cosine v2 −0.031); **crossfade**: 15 ms cosine v2 −0.018 / DNSMOS −0.103, 10 ms −0.040 / −0.162, 5 ms DNSMOS −0.44.
- **Lookahead** (8 clips, paired CI against 0 ms, A/A noise ±0.001): 10 ms set-sim +0.032 [+0.013, +0.055], cosine v2 +0.029, env MCD −0.13 dB, F0 −3.2 cents, DNSMOS +0.053; 20 ms +0.046 [+0.023, +0.078], +0.036, −0.18 dB, −6.1 cents, +0.136 [+0.007, +0.27]. Clicks 0 at 10 ms, +0.25 at 20 ms. Quality's 20 ms now also runs on DirectML at CUDA's fidelity (cosine 0.957 against 0.956, MCD 1.743 against 1.742 dB).
- **Content context and normalisation**: 1 s of content context costs +0.08-0.10 dB MCD on 7 of 8 clips; running normalisation with the onset glide raises cosine +0.009 / +0.012 (8/8, 6/8 clips) but MCD +0.04 dB on 6 of 8 and still mutes the onset after a pause (−17.8 against −7.4 dB). The onset muting comes from the generator's `enc_p`; the window's own GroupNorm partly masks it.
- **Missed-target honesty**: the T5 v2 rows, T1, T3 and the Streamer RTF are *misses*; nothing in this section is a claim that they are met.

### Integrated smoke (P3B-I0)

What it proves: the integrated engine (every P3B workstream plus the defaults flipped by P3B-I0) loads, converts and plays on each execution provider, does not crash, and gives sane output. It is not a timing claim (one run per row, machine otherwise idle: no `rustc`, CPU about 7 % before the first run, GPU timing lock held). Executable: release build of the integrated tree (`sigil.exe` sha256 `a7cc2a05…`, 0 compiler warnings), the MIT default voice, Low latency / 40 ms blocks / RMVPE on the GPUs, the Streamer preset / 120 ms blocks / SwiftF0 on the CPU, 512-frame device blocks, the shipped defaults, clips `en-jfk` (3.85 s) and `de` (2.75 s).

| | CUDA | DirectML | CPU (Streamer, one thread) |
|---|---|---|---|
| Runs | jfk offline + 30 s paced, de offline: all exit 0, 0 model errors | the same | jfk offline + 20 s paced, de offline: all exit 0, 0 model errors |
| EPs in use (every component) | content, enc, flowdec, pitch: `cuda`, no fallback, no warnings | `dml`, no fallback, no warnings | `cpu` |
| Cold load (first / second process) | 2060 / 981 ms (the first read the models from a cold file cache); generator built once | 1125 / 1048 ms | 634 / 625 ms |
| Offline call p50 / p95 / p99 | 6.3 / 7.2 / 7.9 ms | 6.5 / 7.8 / 8.0 ms | 275 / 280 / 282 ms (RTF p99 2.35: the one-thread preset is not real time at 120 ms blocks, as measured before) |
| Stage p50 (content / F0 / enc / flowdec) | 2.0 / 3.3 / 0.2 / 2.5 ms | 1.1 / 3.1 / 0.7 / 2.3 ms | 54 / 2.0 / 7.5 / 211 ms |
| Syncs per call | 3 | 4 | 0 |
| Paced (wall clock): latency p50 = p99, underruns | 132.6 ms, 0 | 132.6 ms, 0 | 638 / 811 ms, 1641 (expected: RTF > 1) |
| Paced call p95 (idle-clock cadence) | 16.4 ms | 15.8 ms | 290 ms |
| Worker CPU, offline / paced | 0.22 / 0.038 cores | 0.23 / 0.045 cores | 1.0 / 0.99 cores |
| VRAM (process delta) | 1332 MB | 842 MB | 0 |
| F0 extractor calls / over the plan's bound | 98 / 0 and 737 / 0 | 98 / 0 and 737 / 0 | 34 / 0 |
| Output | 48 kHz stereo, length equals the input, finite, rms 0.107-0.126, peak 0.59-0.71 | the same (rms 0.107-0.126, peak 0.59-0.73) | the same (offline rms 0.107-0.109; the paced file includes the muted underruns) |

Two integrated-build confirmations of flipped defaults (interleaved, one executable, the switch on side B; GPU timing lock held; 2026-10-09):

- **Parallel cold load** (`SIGIL_VC_PARALLEL_LOAD=0` against the default, 5 ABBA rounds, `load-cuda` and `load-dml`): CUDA `engine_load_ms` 967 against 1719 ms (-44 %, all 5 pairs faster by 732-816 ms), DirectML 1045 against 2130 ms (-51 %, all 5 pairs faster by 1077-1109 ms); `generator_builds` 1; VRAM CUDA 1330 against 1300 MB (the parallel build's transient, +30 MB), DirectML equal.
- **Hybrid CUDA wait** (`SIGIL_CUDA_TUNING=wait=spin` against the default, 4 ABBA rounds, jfk offline + 30 s paced): worker CPU offline 0.198 against 1.292 cores (-85 %), paced 0.041 against 0.463 (-91 %) (the spin side REGRESSED); call p95 8.22 against 8.24 ms offline and 17.29 against 17.09 ms paced, p99 NEUTRAL, latency p99 identical, 0 underruns. The GPU changed clock regime in every paced run (SM clock 855-1935 MHz), so absolute paced numbers are not comparable across rows.
- **Shared DirectML queue** (`SIGIL_VC_DML_SHARED=0` against the default, `load-dml`, 6 ABBA rounds): offline call p50 7.20 against 7.91 ms (the unshared side is 12.3 % slower, CI 2.9..17.8 %), p95 7.84 against 8.64 ms (INCONCLUSIVE, unshared +11.4 %), VRAM 841 against 853 MB (-12 MB); the cold-load gain P3B-G measured is not reproduced on the integrated build (1246 against 1254 ms, NEUTRAL: the parallel cold start already hides it) and the worker CPU is higher with the shared queue (0.225 against 0.196 cores offline, +14 %, about 0.03 cores). A first, noisy run of the paced row (3 usable pairs, the GPU changed clock regime in 5 of 8 runs) agreed on the sign of the call p50. **Review fix (2026-10-09): the default is now off.** The integrated build measured it as the default; P3B-G's own gates showed the content stage's p95 REGRESSED on the one queue (+6.1 % [1.13, 7.26] paced, +21.9 % [19.46, 28.94] under the 70 % GPU load, where a game shares the GPU), the cold-load gain's CI bound (37 ms) is under T9's 0.05 s, the integrated build did not reproduce it, and the default also reaches sessions the gates never ran (kNN, the legacy bridge, P5's families). A call-level gain does not waive a regressed stage under load (rule R3). `SIGIL_VC_DML_SHARED=1` opts in; P7-B re-measures on a quiet rig.

Measurements overlapped nothing else. The smoke specs and logs are in the P3B-I0 scratch folder; the CPU row is the shipped preset, so its underruns describe that preset on this PC (the per-PC autotuner exists for it), not a regression.

## P5 voice families

P5 added voice families that run behind the same harness as RVC voices ([ADR 0006 → P5 outcomes](../adr/0006-vc-zero-shot-singing.md#p5-outcomes), [DEVELOPER_GUIDE → Voice families](../DEVELOPER_GUIDE.md#voice-families-p5)). The rule for P5 was that no family may make the main SigilRvc path worse; the integrated build's A/B is the first table. Same rig as above, 2026-10-01; each workstream ran on a GPU shared with the others, so their absolute numbers are pessimistic, while P5-I0's runs had the PC to themselves.

**Idle-GPU clocks (P5-B finding).** An otherwise idle NVIDIA GPU lowers its clocks (to about 900 MHz) for work that arrives as one burst per block, so a paced run on an idle 4090 reads a bursty voice's ticks 1.4–2.4× slower than offline, with spikes to 80–105 ms for Sigil-ZS; a game keeps the clocks up. The idle paced and contention rows above (P3) likely carry the same effect. NVIDIA's **Prefer maximum performance** for Sigil.exe removes it (documented for users; Sigil never changes driver settings).

### The main path after P5 (P5-I0)

The P3 reference specs (the MIT default voice, RMVPE, 40 ms blocks, Low latency, `auto` levers; `offline`, 2 repeats) on the release exe of HEAD (P3) and of the integrated P5 tree, interleaved in 4 rounds (A B, B A, …), medians, the GPU otherwise idle:

| Spec | call p50 (ms) | call p95 (ms) | call p99 (ms) | CPU (cores) | VRAM (MB) |
|---|---|---|---|---|---|
| CUDA, `auto` levers | 9.35 → 9.34 (−0.1 %) | 9.92 → 9.87 (−0.4 %) | 10.43 → 10.49 (+0.6 %) | 0.99 → 0.99 | 1309 → 1308 |
| DirectML, `auto` levers | 9.25 → 9.32 (+0.7 %) | 9.87 → 9.99 (+1.2 %) | 10.07 → 10.45 (+3.8 %) | 0.38 → 0.38 | 861 → 861 |
| CUDA, `gpu_priority` on (P5-S0's flag) | 9.37 → 9.36 (−0.1 %) | 9.90 → 9.92 (+0.2 %) | 10.13 → 10.40 (+2.6 %) | 0.99 → 0.99 | 1314 → 1312 |

No p95 moved by more than the run-to-run spread (the largest single-round swing was 9.73–12.10 ms on DirectML), far inside the 5 % rule; P5-S0's +7 % with `gpu_priority` on was rig noise. The P5 code on the main path is load-time only (the content encoder's output chosen by width, foreign-graph checks) plus one `Option` per loop for the output-buffer minimum. Note that the idle GPU now gives CUDA p95 9.9 ms against P3-I0's 10.9 ms on a shared rig: the T2 status above is unchanged (target 6 ms).

### DDSP-SVC singing (P5-A)

T10 asks for gross pitch error ≤ 2 % over E2–D#7. The input is a sung E2 → D#6 sweep with vibrato (`scripts/bench/singing_sweep.py`) at +0 and +12 semitones, so the output reaches D#7. Every figure uses a seeded random voice (`export_templates.py --voice`: no public DDSP-SVC 6.3 checkpoint exists), which does not change the pitch path. "Pitch path" is the F0 the vocoder receives; "output" is the converted audio through a pYIN-class tracker.

| F0 method | Pitch-path GPE | Accuracy | Output GPE (CREPE run) | T10 |
|---|---|---|---|---|
| CREPE | 0.00 % | 6.9 cents | 0.00 % on CUDA and DirectML, incl. the frames ≥ 2 kHz | **met** |
| SwiftF0 | 0.00 % | 5.9 cents | | met |
| Fusion | 0.00 % | 4.5 cents | | met |
| FCPE | 0.00 % | 56 cents (flat at the top) | | met on GPE, off-key at the top |
| RMVPE | 2.36 % | 13.1 cents | | missed (notes above about 1.15 kHz input come out an octave low) |

The offline ceiling (the same voice over the whole sweep) scores 0.00 % too; stream against ceiling: F0 16–17 cents, 0 % gross errors, MCD 2.3–2.4 dB (100 ms presets) and 1.0–1.5 dB (`upstream_gui`). New DDSP-SVC voices start with CREPE in the singing range (P5-I0).

| Preset · EP | Tick per block (ms) | Call p50 / p95 (bench) | VRAM | Delay (paced, engine-added) |
|---|---|---|---|---|
| `sigil_singing` (100 ms, 10 steps) · CUDA | 27.9 (content encoder × 2 18.1, F0 0.6, ddsp 2.5, flow 4.7, vocoder 1.8) | 27 / 29–33 | 1028 MB (1540 with RMVPE) | 185 ms p50 at a 35 ms prefill (0–94 underruns per 9 s); 215 ms at 70 ms, 0 underruns |
| `sigil_singing` · DirectML | 28.4 (content encoder 12.8, F0 1.7, ddsp 4.2, flow 6.5, vocoder 3.0) | 27 / 48–53 | 703 MB | as CUDA |
| `sigil_singing_fast` (5 steps) · CUDA | 25.6 | | | |
| `upstream_gui` (500 ms, 20 steps) · CUDA / DirectML | 35.8 / 103.6 per 500 ms | | | about 614 ms |

H1 simulation p99 207–243 ms, 0 blocks over budget; paced CPU 0.26 cores (CUDA), 0.13 (DirectML); load 1.2 s (CUDA), 1.8–2.3 s (DirectML). DirectML went from 2.0–4.6 s per `ddsp` call (ADR 0006) to 4.2 ms (the inverse STFT rebuilt as MatMul + overlap-add). Since P5-I0 the family plays with at least a 70 / 140 ms output buffer.

### Sigil-ZS zero-shot (P5-B)

Input `en-jfk`, reference `ja-short` (one reference, three sources for similarity), 512-frame device blocks; release exe, `--headless bench` waiting for a free GPU before each run.

| Preset · EP | Offline tick p50 | Algorithmic latency | Paced, 80 / 150 ms buffer: latency p50 / p95 / p99 | Underruns per 20 s | H2 70 % load: p99 | H1 sim: p99 |
|---|---|---|---|---|---|---|
| `sigil_low` · CUDA | 24.6 ms | 153 ms | 248 / 256 / 256 ms | 75 (idle-clock spikes) | 248 ms | 248 ms |
| `sigil_low` · DirectML | 29.8 ms | 153 ms | 248 / 252 / 252 ms | 2 | 280 ms | 248 ms |
| `sigil_lean` · CUDA | 16.0 ms | 153 ms | 248 ms flat | 0 | 248 ms | 248 ms |
| `sigil_lean` · DirectML | 19.9 ms | 153 ms | 248 ms flat | 0 | 252 ms | 248 ms |
| `upstream_readme` · CUDA | 34.1 ms | 232 ms | p95 353 ms | 179 | | 387 ms |
| `upstream_readme` · DirectML | 41.6 ms | 232 ms | p95 351 ms | 190 | | 392 ms |

With the voice changer's default 35 / 70 ms buffer the latency read 201–229 ms but underran 145–527 times in 20 s, which is why the family now plays with at least 80 / 150 ms (P5-I0). The ~300 underruns of each H1-sim run are its injected 3 s stall.

| Speaker similarity (exact CAM++ export) | jfk | ru | ko | mean |
|---|---|---|---|---|
| Upstream offline (`inference.py` defaults through our graphs: whole clip as prompt, 30 steps) | 0.507 | 0.522 | 0.647 | 0.559 |
| `sigil_low`, 4 s prompt (default) | 0.556 | 0.561 | 0.629 | **0.582** |
| `sigil_low`, 5 s prompt | 0.556 | 0.565 | 0.639 | 0.587 |
| `sigil_lean`, 3 s prompt | 0.425 | 0.466 | 0.563 | 0.485 |
| `upstream_readme` | 0.520 | 0.583 | 0.629 | 0.577 |

VRAM per voice: CUDA +1358 MB (`sigil_low`) / +1317 MB (`sigil_lean`) including the CUDA context, DirectML +1117 / +1075 MB. Load (T9: ≤ 5 s): reference 1.2–1.4 s (cached afterwards per library voice) plus sessions 1.5 s (CUDA) / 2.0 s (DirectML). The explicit fp16 request gives 15.8 ms on CUDA but moves the waveform, so it stays opt-in; `gpu_priority` and the governor showed no gain.

**T10 (zero-shot):** `sigil_low` with the 80 ms buffer: H2 idle p95 252–256 ms (target 300), H2 load p99 248–280 ms (target 450), similarity 0.582 ≥ 0.559: **met on H2**. H1 is scheduled; the 4090 × 3 estimate is a 74 ms tick on CUDA (about 273 ms end to end) and 89 ms on DirectML, where `sigil_lean` would be chosen. Upstream's live GUI runs 430–690 ms.

### MeanVC2 against the RVC CPU tier (P5-C)

Release exe (LTO off), two interleaved rounds, `en-jfk`, 48 kHz / 512-frame device blocks, 64 ms extra context, on a shared rig.

| Voice | Paced CPU (cores) | Engine-added latency p50 / p99 | Underruns per 20 s (paced / H3) | H3 call p99 |
|---|---|---|---|---|
| MeanVC2 2 steps, 80 ms tick as two 40 ms blocks (default) | 0.28 | 292.5 / 292.5 ms | 0 / 3 | 19.5 ms |
| MeanVC2 2 steps, whole 80 ms blocks (before the split) | 0.26 | 272.5 / 282.5 ms | 105 / 243 | 40.7 ms |
| MeanVC2 1 step, split | 0.19 | 292.5 ms | 0 / 0 | 16.0 ms |
| MeanVC2 160 ms tick | 0.23 | 350 / 360 ms | 265 / 162 | 62.4 ms |
| RVC Streamer (default voice, one thread) | 0.98 | 646 / 741 ms, not real time (RTF 3.4–4.4; about 14 s of every 20 s dropped) | | |
| RVC autotuned CPU (thread pool, 200 ms) | 2.85 | 359 / 397 ms | 966 / 1072 | 201 ms |

Algorithmic latency: MeanVC2 227.5 ms (80 ms tick), 299.5 ms (160 ms); RVC autotuned 227.6 ms. Silence: 0.0098 cores over 10 s (T4 silent target met). Quality smoke (compare mode, a synthetic speaker embedding): DNSMOS OVRL / SIG / BAK MeanVC2 3.02 / 3.30 / 3.97 (2 steps), 3.03 / 3.33 / 3.96 (1 step), 3.01 / 3.33 / 3.88 (160 ms) against RVC Streamer 3.22 / 3.49 / 4.05 and the autotuned CPU preset 3.11 / 3.39 / 3.99; MeanVC2 GPE 5–16 %, F0 RMSE 225–267 cents (it follows the reference's intonation) against RVC's 0–1 %, 47–96 cents. Speaker similarity waits on the real speaker encoder (manual download). Against T4 Streamer (≤ 1.5 cores, RTF ≤ 0.6): MeanVC2 0.28 cores, model RTF about 0.22–0.29. Against T1 Streamer (engine-added p99 ≤ 230 ms): missed at 292–298 ms, but 100–500 ms better than the RVC CPU options. Also: the RVC CPU presets underrun heavily at the default 35 / 70 ms buffer; P7 pairs them with the Stable buffer.

### so-vits-svc and w-okada compatibility (P5-D)

P5-D's own A/B (the P5-S0 exe against its tree, 4 interleaved ABBA rounds, medians, a rig loaded by the other agents): the P3 reference spec on CUDA p50 12.63 → 12.64 ms, p95 16.72 → 14.64 ms (noise); FCPE on CUDA p95 17.01 → 16.22 ms; FCPE CPU Streamer (80 ms) p95 300.3 → 298.7 ms. No regression. No real so-vits-svc export was run (none exists locally); the family is checked against stand-in exports of the exact `onnx_export.py` I/O and stays experimental. Real-model checks: the community FCPE export matches Sigil's adapter exactly (201 frames, 0 voicing disagreements, worst 0 Hz) and mel-input FCPE graphs driven by Sigil's front end match the in-graph pipeline to 0.0024 Hz; 24 real dev-cache graphs are identified correctly.

### Sidecar protocol v2 (P5-E)

The real Seed-VC model under the process host and the production harness, 600 ms budget, delay measured by matching input and output loudness, idle GPU:

| Seed-VC preset (sidecar) | Heard delay | Compute per frame | Muted blocks |
|---|---|---|---|
| `sigil_low` (100 ms frames), CUDA / DirectML | 300–340 / 300–320 ms | CUDA 58 / 66 ms p50 / p95; DirectML 36 / 45 ms | 0 of 650 |
| `sigil_lean` | 280–340 ms | | 0 |
| upstream default (180 ms frames, the recipe's default) | 400–440 ms | CUDA 78 / 96 ms; DirectML 127 / 151 ms | 0 |

Pinning the graphs' shapes made DirectML real time (168–243 → 36–64 ms per 100 ms frame). No `convert` call took more than about 20 ms in any run (debug build). Under 57–98 % GPU load from other jobs, late frames were muted at the budget and nothing blocked. The echo recipe through the real Python helper: ready in 0.56 s, first audio after 380 ms, 0 muted, longest call 3.8 ms.

### Family fidelity against the offline ceiling (P5-I0)

The integrated build's release exe, the GPU otherwise idle, one run per spec: the `-Profile p5` templates' runs without the H1 simulation (Sigil-ZS: `sigil_low`, the `en-jfk` input converted toward `ja-short`; DDSP-SVC: P5-A's imported seeded random voice on the sung sweep at +12 st). Each stream asked for the voice changer's default 35 / 70 ms output buffer; the paced runs played with the family's minimum, as the voice changer now does. The `quality` run scores the stream against the family's own offline ceiling (Sigil-ZS: the one-off conversion of the whole recording, 5.2 s of CPU for the clip; DDSP-SVC: a second instance converting the whole recording in one call).

| Sigil-ZS `sigil_low` (`en-jfk` → `ja-short`) | CUDA | DirectML |
|---|---|---|
| Offline call p50 / p95 | 24.4 / 24.7 ms | 29.9 / 30.5 ms |
| Paced, idle GPU (80 / 150 ms buffer): latency p50 / p95 / p99, underruns per 20 s | 214.6 / 247.9 / 249.3 ms, 154 (idle-clock spikes: call p99 72 ms) | 247.9 / 254.6 / 254.6 ms, 28 (call p99 80 ms) |
| H2 load simulator (70 %): latency p99, underruns | 247.9 ms, 0 | 247.9 ms, 0 |
| VRAM (after the harness build, since P5-I0) / load | 1418 MB / 1.23 s | 1116 MB / 1.34 s |
| Stream vs its offline ceiling | CAM++ 0.83, MCD 2.24 dB, F0 RMSE 156 cents, GPE 1.1 %, V/UV 9.5 %, 0 clicks, DNSMOS 2.60 (ceiling 3.06) | the same to three digits (DNSMOS 2.59) |
| Stream vs the reference clip (bench CAM++ asset) | 0.53 | 0.53 |

T10 on H2 again on the integrated build: p95 247.9–254.6 ms idle and p99 247.9 ms under load, against 300 / 450 ms. The idle-GPU clock dips still underrun on CUDA at the 80 ms minimum (154 per 20 s here, 75 in P5-B's run); under load the clocks stay up and nothing underruns. An adaptive buffer sized from the measured tick is a P7 item. The stream is 0.47 DNSMOS below its whole-utterance rendition and its F0 differs by 156 cents RMS: HiFT predicts its own F0 from each tick's mel, so two renditions of a generative voice differ more than two RVC renditions do; the speaker similarity to the reference (P5-B's exact CAM++: 0.582) is what T10 asks for.

| DDSP-SVC `sigil_singing`, seeded random voice, sweep at +12 st, CREPE | CUDA | DirectML |
|---|---|---|
| Offline call p50 / p95 (per 100 ms block) | 27.3 / 28.9 ms | 26.6 / 27.6 ms |
| Paced (70 / 140 ms buffer): latency, underruns | 214.9 ms flat, 0 | 214.9 ms flat, 0 |
| VRAM / load | 1100 MB / 1.20 s | 820 MB / 1.71 s |
| **T10:** output against the intended notes (pYIN, `score_singing.py`) | GPE **0.00 %**, 9.7 cents, 96 % voiced; frames ≥ 2 kHz: 106, GPE 0.00 % | GPE **0.00 %**, 9.7 cents; ≥ 2 kHz 0.00 % |
| The offline ceiling against the intended notes | GPE 0.00 % | |
| Stream vs its offline ceiling (CREPE analyzer) | GPE 0.85 %, F0 RMSE 102 cents, V/UV 19 %, MCD 3.87 dB, 0 clicks, DNSMOS 1.04 (ceiling 1.05) | GPE 1.33 %, 108 cents, V/UV 19 %, MCD 3.87 dB, DNSMOS 1.05 (1.05) |

The random voice renders a near-noise timbre (DNSMOS about 1.05 for stream and ceiling alike, partly unvoiced at low notes), so its MCD, F0 RMSE and V/UV against the ceiling describe that voice rather than a trained one (P5-A's pitch-path tracker measured 16–17 cents stream against ceiling); the gross pitch error, which T10 asks for, is 0.00 % against the intended notes on both EPs.

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
- **Scoring note (corrected by P3B-B, [ADR 0009](../adr/0009-vc-scorer-v2.md)):** fidelity runs with
  lookahead need `lead_s` ≥ 4 in the quality run. With the P3 profile's 1 s lead the scorer reported
  2700–2900 clicks per 10 minutes and 130 cents F0 RMSE for the same audio. The clicks came from v1's
  click statistic, which estimated the background rate from the 3–8 events outside windows covering
  about 70 % of a 40 ms block and placed those windows 20 ms late at 20 ms of lookahead (the returned
  span's start in input coordinates); the F0 RMSE came from a few gross octave frames. Scorer v2
  places the windows for the model delay, uses a conditional binomial and reads 0 on the saved
  cases and 2 on the two lead-1 lookahead runs.

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
- **The gap is mostly the content encoder's context, not the pitch track (P3B-B, [ADR 0009](../adr/0009-vc-scorer-v2.md)).**
  Through the output analyzer the streamed F0 is 41–78 cents from the offline F0 with 6–10 % voicing
  disagreements, but that figure is dominated by the analyzer's own error and a few gross frames. On the
  engine's fed F0, oracle F0 recovers at most 0.03 dB of the envelope MCD, while the content features'
  limited context (the residual ablation) recovers 0.31–0.32 dB, about half of the 0.60–0.64 dB the
  stream sits over its floor; the rest is the generator's right context and the joins. Lookahead helps
  for that reason (P3B-H, scorer v2: 20 ms raises the set similarity by +0.046 [+0.023, +0.078]).
- **The cosine target was strict and the asset was not exact.** The legacy CAM++ asset scored two
  sentences of one speaker at 0.918 over every frame; with the exact export on speech frames two
  different 3.8 s utterances of one speaker score 0.778 (English) and 0.654 (Japanese), and the same
  pair scored twice is bit-identical. T5's identity row is therefore a difference to the whole-utterance
  ceiling against a target speaker set (ADR 0009).

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

T6 is defined statistically since P3B ([README → compare v2](README.md#modes)): a corpus of loudness-normalised clips, paired per-clip statistics, DNSMOS +0.05 at matched latency with the lower confidence bound above 0, GPE and MCD non-inferiority under every analyzer against EGG or consensus truth, and latency at most 0.7x as a Pareto row against each competitor's operating points, its minimum stable chunk on the same rig included. It is **undecided**: no competitor renditions exist (the installs and captures are manual, [Still to run](#still-to-run-manual)). P7-B left it pending; the exact steps, in order, are the [T6 checklist](../../scripts/vc-bench/README.md#t6-checklist-what-is-left-to-do-in-order-p7-b). The published per-chunk figures are reference rows from other PCs, so they bound rather than rank:

| System | Per-chunk compute | Chunk | Compute / chunk | GPU and method |
|---|---|---|---|---|
| vc-rs | 13.9 ms p95 | 100 ms | 0.139 | RTX 3060 Ti, `trtexec` with TensorRT 11, model only (transfers excluded): ContentVec 2.51, RMVPE 5.32, RVC 6.07 ms p95 |
| rvc-realtime | p50 15.7 / p95 21.2 ms | 60 ms | 0.262 / 0.353 | RTX 3060 Ti, FCPE, fp16 |
| SVE, CUDA, RMVPE (P3B-C, one run, vc-rs's shape) | p50 7.98 / p95 10.98 / p99 11.66 ms | 100 ms | 0.080 / 0.110 | RTX 4090; "model step incl. host I/O": components p95 content 3.78, F0 5.64, enc 1.73, flowdec 3.68 ms |
| SVE, DirectML, RMVPE (same run) | p95 17.09 ms | 100 ms | 0.171 | RTX 4090 |
| SVE, CUDA + levers, SwiftF0 (P3) | p95 10.6 / 10.1 ms | 80 / 40 ms | 0.132 / 0.252 | RTX 4090 |

The 4090 and the 3060 Ti differ by roughly 1.5x to 5x depending on the workload, so no cross-rig ranking holds: `scripts/vc-bench/vcrs_probe.py` gives a normalized-RTF verdict only when the GPU names match. P3B-C's ablation dry run (10 local clips, SVE minus time anchoring at matched shapes) found no automatic metric that separates the time-anchoring levers (DNSMOS −0.011 [−0.120, +0.098] at 40 ms, MCD to the reference +0.007 dB, GPE 0), while the same metrics do separate block and context (DNSMOS 3.11–3.14 at a GUI-like 200 ms shape against 2.71–2.79 at 40 ms); a blind listening check comes before any conclusion, and the ablation margins are never published as competitor results. The +0.05 DNSMOS row needs 31–36 clips (50 % power) or 63–73 (80 %) at the measured paired SD of 0.14–0.15, so the public corpus (92 clips) can carry it and the 10-clip local corpus cannot (the blind A/B row carries the quality claim there). Inputs must be loudness-normalised: a −65 dBFS clip was silence-gated in both renditions. As a whole-clip analyzer SwiftF0 voices only 2–47 % of the speech frames and CREPE-tiny 78–100 %, so the strict four-analyzer consensus truth covers 2–44 % of speech frames: use EGG truth or `consensus_min_agree: 3`.

## Scorers and how to read them

**Scorer v2** (P3B-B, [ADR 0009](../adr/0009-vc-scorer-v2.md); `scoring.analyzer: v2`, the v1 metrics are printed next to it):

- **CAM++:** the legacy asset divides the last, partial 100-frame segment by the full kernel (`AveragePool` with `ceil_mode` in ONNX Runtime: 6.5–25.5 dB SNR against upstream on ragged lengths); `bench/campplus-exact` reads 104.5–117.4 dB, and is applied to speech frames only, so identical audio scores bit-identically across builds. The identity row is SIM(stream, target set) − SIM(whole ceiling, same set), the set being the other clips' whole-ceiling renditions.
- **F0:** the engine's fed F0 (the probe taps the shifted pitch on the absolute 10 ms grid) against whole-clip RMVPE of the input: fine pitch error, gross pitch error, V/UV and onset lag. The RMVPE analyzer floor is 5.7 cents [3.8, 8.0], below 10, so the fine-pitch limit stays 15 cents. The output-analyzer RMSE is advisory.
- **Clicks:** two conditional binomial tests (97.5 %): v1's windows placed for the model delay, and narrow windows at both crossfade boundaries; validated by planted pops and a null test.
- **MCD:** the envelope MCD (F0-adaptive smoothed spectrum) over its same-seed floor (primary: the same seed with blocks shifted by 100 ms, 1.41–1.43 dB; a different seed reads 1.22, ceiling B against A about the same), and the offset-corrected MCD (SOLA timing accounts for 0.04–0.05 dB of the 1.91–1.93).
- **Ablations** (oracle F0, oracle content, both, stream F0 in the ceiling, residual) say where a gap lives; **rescore** judges old WAVs with the current scorer.

**Scorer v1** (kept next to v2 for one release):

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
  test): the two halves of the 3.85 s JFK clip score **0.918**, the JFK clip against a Japanese
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
- **Other ceilings (P3B-B).** Ceiling B (400 ms blocks, 160 ms lookahead, 500 ms context) and the whole-utterance ceiling (the input region plus 1 s of reflect padding, one pass on a dynamic-shape CPU path, run precision plus an fp32 row) bracket the primary. The stream's first 200 ms after an onset sit −4.9 to −5.0 dB under its speech level, against A −3.7, B −2.4 to −3.3 and whole −2.1 to −2.2: no ceiling mutes onsets, and the stream mutes them a little more.
- **Full decode, cached F0.** `ceiling.exact` (default) turns partial decode off in the ceiling. It
  keeps the F0 cache on: recomputing F0 over every block's window tracks the input's pitch slightly
  better (69 vs 80 cents RMSE against the input's own F0) but makes adjacent blocks disagree at their
  joins: 531 vs 343 artifact events per minute and DNSMOS 3.19 vs 3.32.

## What closes the gaps

P3B shipped what the measurements named and measured the rest ([P3B section](#p3b-voice-engine-optimization-p3b-i0)). What is left, in order of what it buys:

1. **Latency (T1: 132.6 ms against 85–95 ms; 114.7 ms as the app starts the preset).** The 10 ms device block (−17.4 ms in P3B; P7-B's interleaved sessions read −16 % on CUDA and −9 % on DirectML at the app shape, opt-in) once it is confirmed on a live WASAPI device; the adaptive cushion if it can hold 0 underruns under every load (−18 ms on a calm PC); a lookahead other than 0 spends the delay it would save. The 10 ms crossfade (−8.7 ms) fails its fidelity gate, so 85 ms needs a model that sounds as good with less right context.
2. **Per-call cost (T2).** CUDA is at 7.4–7.6 ms idle (6 ms is TensorRT-class work, deferred: no runtime pack) and 9.4 ms under load; DirectML 9.0 idle and 11.6 under load. What remains is RMVPE (3–5 ms) and the generator's launches; a merged DirectML `enc` + `flowdec` graph and F0 on a DirectML compute queue are the deferred follow-ups.
3. **VRAM (T3).** The weight floor plus the CUDA context make 800 MB hard on H2; the VRAM saver (DirectML 654 MB, CUDA 1156) needs an app setting and a decision about its pitch cost; the content encoder's session-init transient (200–241 MB) is the largest peak left.
4. **The CPU tier (T4).** RTF 0.6 at 120 ms blocks needs at least 4.0 cores at perfect scaling; the best measured policy (6 threads at below-normal priority, arena on) reaches 0.74–0.83 at 2.8 cores. The routes to the original 1.5 cores are the opt-in int8 CPU decoder (deferred) or longer blocks (200 ms: 0.62 at 2.8 cores). The Streamer thread policy was applied by P7-I0: all physical cores but two, within 4 to 6 threads, where P7-B measured RTF 2.7 on one thread, 0.88 on six threads at below-normal priority and 0.91 on the default pool; `SIGIL_VC_STREAMER_THREADS` overrides the count, and an H3 live record on a real 4 to 6-core PC is still pending.
5. **Fidelity (T5 v2).** The content encoder's context is the largest piece of the gap; lookahead (10–20 ms) and the gated levers recover parts of it at a cost. DirectML `ja-long`'s seam needs a listening check.
6. **Head to head (T6).** Needs the competitors' renditions (manual).

## Reproducing

1. Build a release executable: `cargo build --release --manifest-path src-tauri/Cargo.toml`. Keep
   `sigil.exe` next to `DirectML.dll`, the `onnxruntime_providers_*.dll` files and
   `resources/vc-templates/` (as in `src-tauri/target/release/`).
2. Make an app-data folder for the benchmark (`SIGIL_APP_DATA_DIR`) and install the models with the
   store, one at a time: `sigil.exe --headless fetch vc/voice-default`, `vc/pitch-rmvpe`,
   `vc/pitch-swiftf0`, `vc/pitch-crepe-tiny`, `bench/campplus`, `bench/dnsmos-p835` and
   `runtime/nvidia-cuda13` for CUDA (scorer v2 also needs `bench/campplus-exact`, unpublished: `SIGIL_LOCAL_ASSETS`); `vc/embedder-contentvec` and `vc/pitch-fcpe` are Sigil exports
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

7. Before / after claims (P3B) come from interleaved runs, not single pairs: `run-perf-gate.ps1 -CompareExe | -CompareVars | -CompareEnv`, holding `gpu-lock.ps1`'s lock ([README → Interleaved A/B](README.md#interleaved-ab-p3b-a-the-evidence-a-beforeafter-claim-needs)). Fidelity: `scoring.analyzer: v2`, the corpus runner `scripts/bench/specs/p3b/b/run_corpus.ps1`, and `rescore` for old WAVs.

Run nothing else heavy on the PC meanwhile: another GPU job changes every number, which is exactly
what the contention runs measure on purpose.

## Still to run (manual)

- **P7-B follow-ups** (the final scorecard [ran](#p7-b-final-scorecard-of-the-shipped-defaults); these are what it leaves, decided by its numbers; the Streamer thread policy was applied by P7-I0, the rest are not): flip `device_block` to `model_divisor` for the GPU presets after a live record on a real microphone and headphones (the interleaved sessions show -9 to -16 % latency on DirectML and CUDA with compute and CPU unchanged, and, on DirectML under load, 75 % fewer underruns); the CPU Streamer thread policy is applied (P7-I0, [Defaults](#defaults-and-what-the-numbers-recommend)), an H3 record on a real 6-core PC would confirm it (it was not real time on one thread: RTF 2.7). **Closed since the review of the scorecard:** the output queue floor for template voices on long blocks (the cap was not the lever; [diagnosis](#diagnosis-the-recovery-never-ends)), with `sve-cpu`, `sve-cpu-pool` and `sve-cpu-t6` re-run: their derived `played_ratio` rows are 99.1-99.8 % (>= 98 % required), and a live listening check of the 110 / 136 ms floor on a real audio device is still open; whether the Low-latency preset takes a 10 ms lookahead (unchanged from P3B: +10 ms for +0.03 speaker similarity); a blind listen of FCPE against SwiftF0 on the CPU Streamer preset (the automatic metrics favour FCPE's audio, its pitch track is worse; the SwiftF0 + FCPE fusion is the candidate if the pitch is audible). Measured and closed: the CPU int8 content encoder stays first (fp32 costs +15 % of the content stage), the shared DirectML queue stays off, context is free, FCPE stays the Streamer F0.
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
  drivers in the user's own Python environments, `--corpus` and `--sweep`), w-okada VCClient and other
  live apps through `play_record.py` (the e2e play-and-record tier), the same voice and clips; fetch the
  public corpus (`scripts/bench/fetch-public-corpus.py`, about 646 MB, one download at a time), then
  score with a `compare` run (compare v2) and the blind A/B kit (`tools/listening-test`, 5+ listeners ×
  20 pairs; a 20-trial page for the ablation dry run is already prepared). None of the apps is installed
  on the dev rig, and the plan keeps competitor installs manual. Also listen to DirectML `ja-long`'s
  seams (one narrow-window click) and publish `bench/campplus-exact` (local-only until then).
- **Corpus (T5, T8):** the 20 consented speech clips (plus game-music bleed and singing) for the
  fidelity rows, and the user's 40-voice community corpus for the importer's T8 row.
- **TensorRT-RTX (T9):** install `runtime/tensorrt-rtx` and run a `studio_trt` spec twice (the first
  engine build, then the cache hit).
- **P5 families:** the H1 live-record of Sigil-ZS and DDSP-SVC; a real trained DDSP-SVC 6.3 voice
  (every DDSP figure above is a seeded random voice); a real so-vits-svc export against upstream
  inference; MeanVC2's speaker encoder (a manual Google Drive download) for its speaker and
  listening A/B; X-VC and StreamVoiceAnon with their checkpoints; a blind listen of Sigil-ZS's
  joins. `pwsh scripts/bench/run-perf-gate.ps1 -Profile p5` reruns the family specs.
