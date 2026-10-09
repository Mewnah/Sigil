# P3B-G specs

Execution layer (device-resident generator, CUDA waits, workspace, DirectML sharing, kNN, CPU sessions).

Benchmark specs (`sigil-bench-spec/1`) and spec templates of workstream P3B-G live here (P3B rule R7:
each workstream keeps its specs under `scripts/bench/specs/p3b/<letter>/`; only P3B-A edits
`scripts/bench/profiles/`, the others send profile rows to P3B-I0).

Every timing run holds the GPU timing lock (`pwsh scripts/bench/gpu-lock.ps1 -Acquire <tag>` /
`-Release`, or `run-perf-gate.ps1`, which takes it per run; rule R2), and before/after claims come
from ABBA-interleaved reruns (rule R3). Record each spec's command and numbers in the workstream
report.

## Files

| File | What |
|---|---|
| `engine.template.json` | One template for every P3B-G row: EP, F0, preset, block, levers, precision, the CPU session options (`INTRA_THREADS_JSON`, `CPU_ARENA_JSON`, `DYNAMIC_BLOCK_BASE_JSON`), worker priority, retrieval index and `RUNS_JSON` (the runs). |
| `silent.template.json` | The same on silent synthetic input (T4's silent row). |
| `autotune.template.json` | The same without thresholds, for one `autotune` run (the CPU autotuner end to end). |
| `g.profile.json` | The working profile (`run-perf-gate.ps1 -Profile scripts/bench/specs/p3b/g/g.profile.json`); rows below. |

| Row | Runs | Used for |
|---|---|---|
| `g-cuda`, `g-cuda-levers` | offline 8, paced 60 s, h2-load70, h2-load70-oop, h1-sim | the CUDA wait modes (T4 cores, T2 under load) |
| `g-cuda-core` | offline 8, paced 60 s, h2-load70 | the device chain on / off, block waits |
| `g-cuda-paced`, `g-cuda-silent` | paced 60 s / silent 30 s | the spin hypothesis, T4's silent row |
| `g-cuda-sat`, `g-cuda-sat-hard` | one 40 s contention run: 10 burners at 90 % duty (simulated clock), or 12 at 100 % (wall clock), plus the 70 % in-process GPU hammer | the CUDA wait modes with the CPU saturated: pin the process tree first (`ProcessorAffinity` 0xFFF / 0x3F; the review-fix driver is `scratchpad/p3b/gfix/sat.ps1`) |
| `g-oop-cuda`, `g-oop-dml` | h2-load70-oop | process GPU priority (with the load's achieved duty) |
| `g-vram-cuda`, `g-vram-cuda-levers` | offline 8 | the `SIGIL_CUDA_TUNING` VRAM sweep, the arena shrink |
| `g-knn-cuda`, `g-knn-dml`, `g-knn-probe-*` | offline (+ paced) with `INDEX` | the GPU kNN footprint, arena growth, optimized-graph dumps and profiles |
| `g-load-dml`, `g-load-cuda` | one cold load per process | the shared DirectML queue A/B, concurrent `enc` / `flowdec` builds |
| `g-cpu`, `g-cpu-offline`, `g-sve-cpu` | 120 ms CPU blocks | the CPU arena, denormals, `dynamic_block_base`, threads x priority |
| `g-cpu-autotune` | one `autotune` run (CPU rung, SwiftF0, RTF limit 0.6, scorer-v2 verify) | the autotuner's thread x priority stage and block choice |
| `g-dml-core` | offline 8, paced 60 s, h2-load70 | per-call effects of the shared DirectML queue |

## Switches (P3B-G; every change keeps a way back, rule R5)

| Variable | Effect |
|---|---|
| `SIGIL_CUDA_TUNING=wait=spin\|block\|hybrid` | How a CUDA plan's thread waits for its run (`gpu::cuda_wait`): `spin` = `cuStreamSynchronize` (P3, spins under the driver's default scheduling), `block` = a `CU_EVENT_BLOCKING_SYNC` event, `hybrid` = poll while the run is expected to end within the spin budget (`spinus=<µs>`, default 250), then block. Default: `cuda_wait::WaitMode::DEFAULT`. |
| `SIGIL_CUDA_TUNING=shrink=1` | A CUDA-graph plan's first warm-up run shrinks its arena (`memory.enable_memory_arena_shrinkage`), the second runs with the cached algorithms, then the capture. Default: `plan::SHRINK_BEFORE_CAPTURE`. |
| `SIGIL_VC_DEVICE_CHAIN=0` | `enc`'s `m_p` / `logs_p` take the host path again (download, host copy, re-upload, one more host wait). Default: the device chain. |
| `SIGIL_VC_PARALLEL_LOAD=1` | `enc` and `flowdec` build on two threads. Default: `generator::PARALLEL_LOAD_DEFAULT`. |
| `SIGIL_GPU_PROCESS_PRIORITY=normal\|above_normal\|high` | The process's GPU scheduling priority class (`gpu::sched_priority`, `D3DKMTSetProcessSchedulingPriorityClass`; never realtime, never admin). Default: unchanged. |
| `SIGIL_CUDA_CTX_SCHED=auto\|spin\|yield\|blocking` | The CUDA primary context's scheduling flags, process-wide (ONNX Runtime's own syncs too). An experiment only. |
| `SIGIL_VC_CPU_ARENA=0\|1` | The voice engine's CPU sessions with ONNX Runtime's CPU arena (`ort_session::vc_cpu_arena_default`). |
| `SIGIL_VC_DENORMAL_ZERO=0\|1` | `session.set_denormal_as_zero` on the voice engine's CPU sessions. |
| `SIGIL_VC_DYNAMIC_BLOCK_BASE=<n>` | `session.dynamic_block_base` on the voice engine's CPU sessions without an explicit value. |
| `SIGIL_VC_DML_SHARED=0\|1` | Voice-engine DirectML sessions (without the high-priority queue) on one normal-priority device and queue per adapter (`gpu::dml_queue::DmlQueue::shared_normal`; weak references, so the objects go with the last plan that holds them; device-removed check on every hand-out). Default: off (`ort_session::VC_DML_SHARED_DEFAULT`; see "Review fixes"). |
| `SIGIL_VC_KNN_STORAGE=fp16\|fp32` | The GPU kNN index storage under `auto` precision (default: `retrieval::KNN_FP16_ON_CUDA_DEFAULT` on CUDA, fp16 on DirectML). |
| `SIGIL_VC_KNN_ARENA=exact\|pow2` | The GPU kNN session's CUDA arena growth (default exact, `retrieval::KNN_EXACT_ARENA_DEFAULT`). |
| `SIGIL_ORT_OPTIMIZED_DIR=<dir>` | Tooling: every matching session writes its optimized graph (`<component>-<ep>.optimized.onnx`; `SIGIL_ORT_PROFILE_FILTER` filters). |
| `SIGIL_AUTOTUNE_VERIFY=v1` | The autotuner's fidelity check without scorer v2's offsets. |
| `SIGIL_BENCH_GPU_TIMING=1` | (P3B-A's switch, now filled) GPU time per plan run from CUDA events: `gpu_<content\|pitch\|enc\|flowdec>_ms_*`. |

## Results (dev rig H2, RTX 4090 / Ryzen 7800X3D, exes built from the P3B tree; background load logged)

All before/after rows are ABBA-interleaved on one exe with the switch in `-CompareEnv` (rule R3); the
interval is the gate's (t below 6 pairs, exact Wilcoxon / Hodges-Lehmann from 6). The GPU sat in a
low clock regime for most sessions (paced SM clock 840-870 MHz; another process used up to 62 % SM),
so absolute numbers are provisional; the ratios are the evidence.

### CUDA waits (T4 GPU preset, T2)
Measure first (spin hypothesis, `g-cuda-paced`, spin waits): the worker spent 15.0-15.5 ms of CPU per
18.7-19.2 ms call (worker 0.38 cores) and the F0 helper ~0.24 more: CPU time per tick = GPU-busy time.
`wait=hybrid` vs spin, `g-cuda`, 9 pairs:

| Run | CPU cores spin -> hybrid | call p95 | call p99 |
|---|---|---|---|
| offline | 1.306 -> 0.215 (-83.6 %) | +1.93 % [+0.35, +5.07] | +1.94 % [-0.06, +4.65] |
| paced | 0.629 -> 0.042 (-92.9 %) | -0.54 % (regime-matched IMPROVED) | +0.06 % |
| h2-load70 | 0.449 -> 0.050 (-88.8 %) | -0.45 % | -1.17 % [-2.97, +0.10] |
| h2-load70-oop | 0.723 -> 0.047 (-93.4 %) | +1.44 % | +0.99 % |
| h1-sim | 0.463 -> 0.047 (-88.9 %) | +0.99 % | +1.71 % |

Over budget 0 everywhere; the out-of-process load held its 0.70 duty. Recommendation:
`WaitMode::DEFAULT = Hybrid` (costs ~0.15 ms of wake-ups on back-to-back offline calls, nothing on the
real-time cadence or under load).

### CUDA waits with the CPU saturated (review check, `g-cuda-sat`, `g-cuda-sat-hard`)
The review asked for the hybrid default under a saturated CPU (the dev rig has 16 idle threads; the first series
loaded 3). Same exe (the integrated `exe-i0`), default (hybrid) against `SIGIL_CUDA_TUNING=wait=spin`, ABBA, the
whole process tree pinned to fewer logical CPUs (pwsh `ProcessorAffinity`, inherited by the gate and the exe) and
the 70 % in-process GPU hammer running (`scripts/bench/specs/p3b/g/g.profile.json`, driver
`scratchpad/p3b/gfix/sat.ps1`); the worker is NORMAL priority like the burners; background: 0-1 rustc.

| Row | Load | Pairs | call p50 / p95 / p99 (hybrid vs spin, medians) | over budget | cores (hybrid vs spin) |
|---|---|---|---|---|---|
| `g-cuda-sat` (simulated clock, 40 s) | 10 burner threads at 90 % duty on 12 logical CPUs | 3 | 7.61 vs 7.52 / 8.63 vs 8.63 / 9.31 vs 9.38 ms (all NEUTRAL) | 0 = 0 | 0.066 vs 0.179 (spin +172 %, REGRESSED) |
| `g-cuda-sat-hard` (wall clock, 40 s) | 12 burner threads at 100 % duty on 6 logical CPUs (2x oversubscribed) | 3 | 7.24 vs 7.36 (NEUTRAL) / 11.1 vs 57.5 (spin higher in each pair: 10.7 vs 61.9, 29.9 vs 35.7, 11.1 vs 57.5; INCONCLUSIVE at 3 pairs, REGRESSED for spin without the pair across GPU clock regimes) / 83.0 vs 87.3 (NEUTRAL) ms | 3.5 % (3.3-4.5) vs 7.6 % (4.0-9.0) | 0.058 vs 0.155 (spin +191 %, REGRESSED) |

Under the 2x oversubscription spin was worse, not better: its call p95 was 5-6 times the hybrid's in the two pairs
at matching GPU clocks (1.2 times in the third), it went over budget about twice as often and underran more in
every pair (607 / 282 / 296 underruns against 254 / 248 / 285 for hybrid). With 3 pairs the intervals are wide, so the claim is the direction and, more
to the point, that hybrid did not regress: a worker that spins holds its core and its quantum while it waits for
the GPU, and a blocked worker gives the core to the game and is woken with a priority boost when the run ends.
The default stays hybrid. Not run: a 20-pair series and the 4-thread `h3_cpu_game` profile pinned below 6
logical CPUs (P7-B re-measures the shipped defaults on a quiet rig).

### Sigil-ZS on the hybrid wait (a P5 family; the review asked for one family voice)
`real_model_streams_on_the_gpu` (the real seed_vc graphs from the dev cache, `sigil_low`, CUDA, 110 blocks of
100 ms through the production stream harness; every plan wait goes through the same `cuda_wait`), default (hybrid)
against `SIGIL_CUDA_TUNING=wait=spin`, 12 runs in two ABBA series under the GPU lock, 0 rustc, the first run (a
cold GPU, hybrid 38.75 / 40.10 ms) left out of the medians. Tick p50 / p95 (ms), hybrid: 33.92 / 36.87, 33.71 /
34.36, 33.92 / 37.11, 33.98 / 38.90, 36.40 / 39.65 (median 33.92 / 37.11); spin: 33.36 / 38.08, 33.43 / 34.82,
33.63 / 36.55, 33.34 / 34.62, 33.33 / 34.81, 33.93 / 37.49 (median 33.40 / 35.68). The blocking wait costs
Sigil-ZS about 0.5 ms per tick at p50 (+1.5 %: a handful of plan waits per tick, each pays an OS wake-up of
tens of microseconds) and 1.4 ms at p95 (+4 %, inside the 34.4-39.7 ms spread of either side), against a
100 ms block budget; the output is unchanged (speaker cosine 0.556-0.558 stream / 0.494 offline in every run), VRAM
+1334 MB on both sides. Kept: the family's real-time margin is large, and the CPU the spinning wait gives
back goes to the game. A family that wants the spin back has `SIGIL_CUDA_TUNING=wait=spin`.

### Device-resident generator (T2)
Chain on (default) vs `SIGIL_VC_DEVICE_CHAIN=0`, `g-cuda-core`, 5 pairs: host waits per call 3 vs 4;
device gap from `enc`'s end to `flowdec`'s start 0.08-0.10 vs 0.11-0.12 ms idle and 0.075-0.08 vs
0.33-0.37 ms under the 70 % hammer; without the chain call p50 is +0.56 % [+0.31, +0.89] offline and
+2.24 % [+0.28, +3.30] under the hammer (p95 +0.34 % / +2.10 %, NEUTRAL). Outputs bit-identical
(`generator::tests::device_chain_matches_the_host_path_on_cuda`).

### CUDA VRAM workspace (T3)
`SIGIL_CUDA_TUNING` `maxws=0` / `algo=heuristic` / both vs default (`g-vram-cuda-levers`, 3 pairs each):
VRAM 1380 -> 1380 MB in every case (0 %); call p95 -5.1 % / +0.25 % / +0.65 %. The plans' CUDA arenas
(reserved / in use MiB, identical to the byte on both sides): content 210.8 / 188.1, enc 31.6 / 25.4,
flowdec 73.0 / 60.0, RMVPE 418.1 / 353.2; the DXGI ledger (312 / 42 / 108 / 546 MB) holds 101 / 10 / 35 /
128 MB outside the arenas, untouched by the search knobs. The cuDNN search workspace is not arena-held
(ONNX Runtime allocates it as a transient buffer), so there is nothing for a shrink before the capture
to give back (`shrink=1` kept as a switch, measured below).

### GPU kNN (T3, T2)
ONNX Runtime fuses `Transpose(index) -> MatMul` into `FusedMatMul` on CUDA fp32, CUDA fp16 and DirectML
(optimized-graph dumps): no re-export. Measured footprint, 30k x 768 synthetic index, the kNN loading alone: CUDA fp32 106 MB
(index 88 MiB), fp16 62 MB, DirectML (fp16) 68 MB. The DXGI delta is process-wide, so the engine books it as
`vc.knn` only when no other probe's measurement window overlapped it (`arena::VramDelta::exclusive`); the
default parallel cold start builds the pitch bank, the content encoder and the generator beside the kNN, and
that load holds the computed size instead (index + norms + the session's fixed cost, ~18 MB on CUDA and ~24 MB
on DirectML, the constant these four measurements share) and does not report it as measured
(`Retrieval::vram_measured`). The benchmark's kNN probe rows run with `SIGIL_VC_PARALLEL_LOAD=0`, where the
measurement is exclusive. Profile on CUDA: fp32 index 1.04 ms per run, fp16
0.94 ms (FusedMatMul 0.043 / 0.035 ms): the P3 "5.0 vs 1.1 ms" no longer reproduces. fp16 storage keeps
the exact blend (max |blend - exact| 2.7e-4 at feature scale 2.5 vs 3.0e-7 for fp32; no neighbour swap).
fp16 vs fp32 index storage on CUDA (`g-knn-cuda`, 30k rows, 6 ABBA pairs): `vram_engine_mb` 1482 -> 1438 (-44
MB), retrieval p50 offline -1.7 % [-5.7, +0.8], paced -11.8 % [-15.3, -9.3]; p95 -4.7 % / -7.1 %, p99 +4.9 % /
-9.4 % (intervals include 0); call p95 -0.9 % / -1.6 %. Equal speed at half the index bytes, so
`KNN_FP16_ON_CUDA_DEFAULT = true` is recommended (the blend carries fp16 rounding of the rows, 2.7e-4 at
feature scale 2.5, so it stays P3B-I0's decision).

### Margin trim (T4, experiment)
`margin_probe.py`, float64, default voice, 80 windows: trimming 1 frame changes the returned audio by
up to 1.06e-5 relative (gate 1e-5), 2 frames 8.9e-3. No frame survives: dropped.

### 2-D (1 x k) convolutions on the CPU (T4, experiment)
Per-node profile at one thread (120 ms block): flowdec 227 ms per run, convs 91 % (57.6 GMAC/s, ~72-75 %
of a Zen 4 core's fp32 FMA peak), ConvTranspose 6 %, elementwise < 3 %. `cpu_conv2d_bench.py` (parity
1.05e-6; 42 + 31 NCHWc reorders): 1 thread 1.14-1.26x faster, 2 threads 1.09-1.16x, 3 threads 0.87x,
4 threads 0.75x. Not adopted (the gate wants >= 15 % at every thread count; the CPU engine runs on
several).


### Shared normal-priority DirectML device and queue (T9)
`SIGIL_VC_DML_SHARED=1` vs off, same exe, fresh processes (`g-load-dml`, one cold load per process, 8 ABBA
pairs): `engine_load_ms` 2704 -> 2520 (-4.0 % [-8.3, -0.9]), every pair faster (B - A ms: -410, -78, -128,
-35, -49, -26, -141, -133; median -103 ms, Hodges-Lehmann -93 ms), VRAM 854 -> 841 MB (the sessions stop
creating a device and queue each). The zero-code A/B that decided to build it (IOBinding on, `gpu_priority`
off vs on, 10 pairs): HL -99 ms, 95 % CI -253..-10. Per call (`g-dml-core`, 7 pairs): offline call p95
-7.3 % [-18.0, -3.2], paced p50 -6.4 %, p95 -7.1 % [-10.6, -4.9], p99 -9.4 % [-13.7, -5.7] (all IMPROVED),
h2-load70 p95 -1.4 % [-3.4, -0.3], p99 -3.3 % [-5.6, +0.8]; over budget 0 = 0. One stage row regresses:
`content_ms_p95` +6.1 % paced and +21.9 % under the hammer (the content session now queues behind the
other stages' work; the F0 helper gets faster, -10.9 % paced). The call-level numbers improve, but R3 does
not let them waive a REGRESSED stage row (the CI lower bounds, 1.13 and 19.46, are above 0), and T9 wants a load
gain of at least 0.05 s at the CI (the 8-pair bound was 37 ms; the integrated build did not reproduce it:
1246 against 1254 ms). P3B-G recommended `true`; the review pass found the gates did not clear R3 / T9 and the
default also reaches sessions the gates never ran (kNN, the legacy `vc` bridge, P5's families), so the default
is **off** (`VC_DML_SHARED_DEFAULT = false`) and `SIGIL_VC_DML_SHARED=1` is the opt-in until P7-B re-measures.

### Concurrent `enc` / `flowdec` builds (T9)
`SIGIL_VC_PARALLEL_LOAD=1` vs off, fresh processes: DirectML `engine_load_ms` 2689 -> 2471 (-7.7 %
[-15.3, -4.1] IMPROVED, 7 pairs, B - A ms: -672, -190, -107, -205, -309, -229, -90), VRAM equal (854 MB);
CUDA 2159 -> 2026 (-7.6 % [-28.5, +2.8], 7 pairs; the first pair's cold A load, 3401 ms, is the outlier
that widens the interval), VRAM equal (1374 MB). Outputs are bit-identical to the serial build
(`generator::tests::concurrent_builds_render_the_same_audio`, and on CUDA the device-chain test).

### Cross-process GPU priority (T2 / T7 W9)
`SIGIL_GPU_PROCESS_PRIORITY=high` (`D3DKMTSetProcessSchedulingPriorityClass`, accepted without admin on the
dev rig: `sched_priority::tests::the_class_changes_without_admin_or_refuses_cleanly` reads `normal -> high`
back) vs off, `h2-load70-oop` (the 70 % duty load in a separate process), 5 ABBA pairs, one exe:

| EP | call p95 | call p99 | over budget | load duty |
|---|---|---|---|---|
| CUDA | +1.57 % [-0.12, +2.45] | +1.12 % [-1.57, +2.94] | 0 = 0 | 0.70 = 0.70 |
| DirectML | +1.07 % [-1.05, +2.50] | +2.88 % [-0.67, +4.30] | 0 = 0 | 0.70 = 0.70 |

Every per-stage p99 (content, F0, `enc`, `flowdec`) is NEUTRAL too. The acceptance rule wants a p99 gain
(CI) with a duty drop of at most 1 %: there is no gain, so the lever is rejected and stays an opt-in
experiment (`gpu::sched_priority::DEFAULT = None`).

### CPU session knobs (T4, Streamer / low-latency CPU at 120 ms blocks, SwiftF0, default pool)
All ABBA on one exe (`g-cpu`: offline x4 and paced 60 s; QoS `app` on both sides; the CPU clock regime
shifted between pairs on this rig, so several pairs are reported as regime-mismatched).

| Knob | RTF p50 | RTF p99 | cores | Verdict |
|---|---|---|---|---|
| CPU arena (`CPU_ARENA_JSON=true`; 5 pairs) | -5.9 % [-10.7, -2.6] | -3.6 % [-8.7, +3.0]; paced p99 -9.4 % [-15.6, -2.2] | paced -4.6 % [-6.2, -3.5] | IMPROVED: recommend on |
| `SIGIL_VC_DENORMAL_ZERO=1` (9 pairs) | +4.2 % [-4.0, +13.7] | +3.6 % [-2.5, +12.0] | +0.9 % | NEUTRAL, not adopted |
| `session.dynamic_block_base=4` (7 pairs) | +3.8 % [-6.1, +9.0] | -0.9 % [-8.8, +5.0]; paced +4.8 % [-2.1, +9.0] | +0.1 % | NEUTRAL, not adopted |

### Streamer operating point: threads x priority at 120 ms (T4)
Single gate runs, two passes (the second in reverse order), `g-cpu` with `INTRA_THREADS_JSON` /
`WORKER_PRIORITY` (`auto` = BELOW_NORMAL pool threads, `protect_voice` = NORMAL), exe-g2, quiet machine
(no rustc, CPU 24 % mean). Medians of the two passes:

| Priority | Threads | RTF p50 | RTF p99 (offline) | Offline cores | Paced cores | Paced RTF p99 |
|---|---|---|---|---|---|---|
| auto | 2 | 1.312 | 1.391 | 1.85 | 1.83 | 1.430 |
| auto | 3 | 0.987 | 1.135 | 2.56 | 2.52 | 1.077 |
| auto | 4 | 0.818 | 0.886 | 3.15 | 2.60 | 0.961 |
| auto | 6 | 0.687 | 0.831 | 4.15 | 2.78 | 0.744 |
| auto | pool (8) | 0.667 | 0.795 | 4.69 | 3.34 | 0.902 |
| protect_voice | 2 | 1.322 | 1.493 | 1.85 | 1.84 | 1.451 |
| protect_voice | 3 | 0.991 | 1.119 | 2.56 | 2.52 | 1.140 |
| protect_voice | 4 | 0.835 | 0.943 | 3.13 | 2.62 | 0.949 |
| protect_voice | 6 | 0.695 | 0.819 | 4.08 | 2.88 | 0.883 |
| protect_voice | pool (8) | 0.718 | 0.881 | 4.62 | 3.30 | 0.868 |

Scaling stops at six threads on this 8-core / 16-thread Zen 4: the default pool (8) costs 0.5-0.6 more
cores for no RTF gain, and NORMAL priority buys nothing on an idle machine (its value is under contention
with a game, which this table does not simulate). The T4 average-cores target (<= 3.5) holds at 4-6
threads (paced 2.6-2.9); the RTF p99 <= 0.6 target does not at 120 ms: the best configuration (6 threads,
below normal) measures 0.69 p50 / 0.74-0.83 p99, 0.92 -> 0.74-0.83 against the P3 baseline at 4.6 cores.
With the CPU arena (-5.9 % RTF) the estimate is 0.65 p50, still above 0.6. The arithmetic: 138 GMAC per
audio second at 120 ms and the measured ~57 GMAC/s per core need 2.4 core-seconds per audio second even at
perfect scaling, i.e. at least 4.0 cores for RTF 0.6.

### Arena shrink before the graph capture (T3)
`SIGIL_CUDA_TUNING=shrink=1` vs off, `g-vram-cuda-levers` (IOBinding + graphs + GPU priority, offline), 7
ABBA pairs, one exe: `vram_engine_mb` 1380 -> 1340 (-2.9 % [-2.90, -2.75], every pair), call p50 +0.9 %
[-11.6, +6.2], p95 +2.1 % [-8.8, +7.5], p99 +2.5 % [-7.8, +8.0] (NEUTRAL; the pair intervals are wide because
the GPU clock regime changed between pairs). Graphs stay on in both arms (levers `cuda_graphs: true`, no
plan notes). Per component ledger (MB): content 312 -> 300, `enc` 44 -> 38, `flowdec` 108 -> 104, RMVPE
544 -> 526; arena reserved (MiB): content 210.8 -> 202.2, `enc` 31.6 -> 28.0, `flowdec` 73.0 -> 70.5, RMVPE
418.1 -> 397.9, in use unchanged. So the first warm-up's search workspace is mostly not arena-held (the
`maxws` / `algo` knobs moved nothing), but ~40 MB of the arena's slack is: the shrink returns it at no
measurable per-call cost. Recommendation: `plan::SHRINK_BEFORE_CAPTURE = true` (the test
`shrink_before_capture_keeps_graphs` checks the capture still verifies and the replays match). ONNX Runtime
validates the request against the session's arena allocator ("Did not find an arena based allocator registered
for device-id combination in the memory arena shrink list: gpu:7" for a device it has none for, reproduced by
`a_refused_arena_shrink_keeps_the_graph`); a refused first warm-up is repeated without the shrink, so a session
on an allocator that is not an ONNX Runtime arena (the registered provider pack) keeps its graph and the plan
gets a note (`arena shrink: ...`).

### Concurrent `enc` / `flowdec` builds, second CUDA series
9 more ABBA pairs (`m6/pload-cuda2`): `engine_load_ms` 2065 -> 2006 (-4.0 % [-25.1, -2.3] IMPROVED, median
-81 ms, all nine pairs faster); together with the first series CUDA gains 4-8 %, DirectML 7.7 %.

### CUDA waits on the levers row (`g-cuda-levers-core`, 9 pairs)
`wait=hybrid` vs `spin` with IOBinding + graphs + GPU priority (the sve-cuda-levers configuration):
`cpu_cores` paced 0.417 -> 0.045 (-89.2 % [-92.7, -88.2]), h2-load70 0.234 -> 0.052 (-76.0 %), h1-sim 0.238
-> 0.050 (-76.9 %), all IMPROVED (T4: <= 0.25 met, and the silent row 0.0088 -> 0.0081). Call p50 / p95 /
p99 are NEUTRAL in all three runs, but this session's GPU clock regime swung between pairs (paced call p95
A 21.6 vs B 15.5 ms medians with intervals of +-20 %), so the intervals are wide; the first series
(`g-cuda`, 9 pairs, tighter) is the call-time evidence. Over budget 0 = 0, syncs per call 3 = 3, no row
REGRESSED except `enc_ms_p95` +8.9 % (INCONCLUSIVE, 0.33 -> 0.37 ms: the enc span is now its enqueue).

### The CPU autotuner end to end (T4, `g-cpu-autotune`)
One `autotune` run (exe-g3, CPU rung, SwiftF0, `low_latency` preset, RTF limit 0.6, scorer-v2 verify over three
lead offsets; 225 s, quiet machine): the thread stage (reference block 80 ms) measured 1-6 threads and the
pool at BELOW_NORMAL and NORMAL priority (13 measurements). RTF p99 at 80 ms: pool 1.26 / 1.14 (below normal /
normal), 6 threads 1.44 / 1.13, 4 threads 1.82 / 1.30; cores per audio second: 6 threads at normal 3.82, pool at
normal 4.54. Nothing meets the limit, so the pick is the fastest expressible policy (pool at normal; the stage
notes that 6 threads at normal measured best but the params cannot set a thread count yet). The block stage:
120 ms 0.948, 160 ms 0.729, 200 ms 0.624 RTF p99 (2.8 cores): no block up to 200 ms meets 0.6, the tuner returns
the 200 ms block as the fastest and says so. The verify stage ran scorer v2 (shortfall 10.3 +- 0.0 on this clip:
fed-input V/UV error 0.34 and fine F0 error 41 cents against the RMVPE reference; the autotuner's fidelity
note reports it and keeps the block). `reports/autotune_cpu.json`. That shortfall is almost all block-independent
(the tracker disagrees with the reference at every block; each seam click adds only 1.0), so the first rule,
"a larger block must cut the shortfall by a quarter", could never fire there: 10.3 -> 8.3 (two clicks
removed) is a 19 % cut. `pick_verified` now also keeps a larger block that cuts the shortfall by
`VERIFY_ABSOLUTE_GAIN` (1.0, one click's worth) beyond the spread over the offsets.

### Recommendations for P3B-I0 (defaults live in constants of P3B-G's files; nothing was flipped here, rule R6)
| Constant (file) | Now | Recommend | Evidence |
|---|---|---|---|
| `cuda_wait::WaitMode::DEFAULT` | `Spin` | `Hybrid` | CUDA cores -84 to -94 % on every row incl. levers (0.417 -> 0.045) and silent (0.0088 -> 0.0081); call p95/p99 NEUTRAL; costs ~0.15 ms of wake-ups on back-to-back offline calls only |
| `ort_session::VC_CPU_ARENA_DEFAULT` | false | true | RTF p50 -5.9 % [-10.7, -2.6], paced p99 -9.4 % [-15.6, -2.2], cores -4.6 % |
| `ort_session::VC_DML_SHARED_DEFAULT` | false | **false** (the review pass kept it off) | load HL -93 ms (CI -269..-37: the bound is under T9's 0.05 s; not reproduced on the integrated build), VRAM -12 MB, call p95 -7 % paced; but the content stage p95 REGRESSED +6.1 % [1.13, 7.26] paced and +21.9 % [19.46, 28.94] under the 70 % load (R3) |
| `generator::PARALLEL_LOAD_DEFAULT` | false | true | load: DirectML HL -209 ms (CI -451..-107), CUDA 16 pairs HL -114 ms (CI -621..-64), VRAM equal, bit-identical audio |
| `plan::SHRINK_BEFORE_CAPTURE` | false | true | VRAM -40 MB (-2.9 %), graphs kept, call p95 +2.1 % [-8.8, +7.5] NEUTRAL |
| `retrieval::KNN_FP16_ON_CUDA_DEFAULT` | false | true | VRAM -44 MB, retrieval p50 equal or faster; blend rounding 2.7e-4 (a fidelity-adjacent switch: P3B-I0's call) |
| `sched_priority::DEFAULT` | None | None (reject) | call p99 CUDA +1.1 % [-1.6, +2.9], DirectML +2.9 % [-0.7, +4.3] under the out-of-process load, duty 0.70 |
| `VC_DENORMAL_ZERO_DEFAULT`, `SIGIL_VC_DYNAMIC_BLOCK_BASE` | off | off (reject) | NEUTRAL (+3.6 % / -0.9 % RTF p99) |
| `autotune::CPU_THREADS_PARAM` | `None` | `Some("rvc_cpu_threads")` once `VoiceChangerParams` has the field (`voice_changer/params.rs` + the worker's `intra_threads`) | 6 threads at BELOW_NORMAL is the best measured Streamer policy on this 8-core rig (0.69 p50 / 0.74-0.83 p99, 2.8 paced cores) vs the pool (8): 0.67 / 0.79-0.90, 3.3 cores |
| Streamer preset | 1 thread (the 120 ms RTF 3.4) | 120 ms blocks, 6 threads below normal, CPU arena on, presented as a latency change | RTF p99 0.69-0.83 (miss vs 0.6), 2.8 paced cores (met vs <= 3.5); 200 ms blocks reach 0.62 |
| `SIGIL_CUDA_CTX_SCHED` | unset | unset | an opt-in experiment; not measured against STT / TTS p95, so never a default |

## Review fixes (fixer pass after the P3B-G review, 2026-10-09)
Seven findings, all confirmed in the code and fixed; each fix has a test that fails without it.

| Finding | Fix | Test |
|---|---|---|
| `VC_DML_SHARED_DEFAULT = true` against R3 / T9 | default **off**; `SIGIL_VC_DML_SHARED=1` opts in; `presets.rs` lists it with the other measured-but-off switches | `presets::p3b_defaults_are_the_decided_ones` |
| kNN footprint = a process-wide DXGI delta over a window the parallel cold start shares | `VramProbe` registers its measurement window; `VramProbe::delta()` says whether another probe overlapped it; `knn_footprint` books the DXGI delta only when exclusive, else index + norms + the session's fixed cost (not reported as measured, no telemetry mark) | `arena::tests::measurement_windows_report_overlap`, `an_abandoned_window_expires`, `measurement_windows_are_bounded`, `nested_probe_windows_are_not_exclusive`; `retrieval::tests::knn_footprint_is_measured_only_when_exclusive` |
| `vc.knn` diagnostics row recorded before the CPU-rung refusal | `GpuExtras::skip_outcome_record`; `GpuKnn::load` records the row once the session is accepted | `ort_session::tests::skip_outcome_record_leaves_the_diagnostics_row_alone`; `retrieval::tests::a_refused_cpu_rung_leaves_no_knn_diagnostics_row` |
| `pick_verified`'s relative 25 % rule on a shortfall with a large block-independent part | `VERIFY_ABSOLUTE_GAIN` (one click's worth) beyond the spread | `autotune::tests::a_large_block_independent_shortfall_does_not_hide_a_click_free_block` |
| Hybrid wait default never run with the CPU saturated, nor on a P5 family | two saturated A/Bs (12 and 6 logical CPUs) and Sigil-ZS on CUDA, above: hybrid did not regress (spin was worse at 2x oversubscription); Sigil-ZS +0.5 ms p50 per 100 ms block; default unchanged | rows `g-cuda-sat`, `g-cuda-sat-hard` in `g.profile.json`; `sigil_zs::tests::real_model_streams_on_the_gpu` |
| the shared DirectML device held for the life of the process | `DmlQueue::shared_normal` uses the weak cache of the high-priority queue (released with the last plan) | `dml_queue::cache_tests::per_priority_weak_caches_release_with_their_last_holder`, `dml_queue::tests::shared_normal_queue_is_released_with_its_last_holder` |
| a shrink run ONNX Runtime refuses fails the capture (no graphs, for 40 MB) | `first_warmup` repeats the warm-up plain and notes it | `plan::tests::a_refused_shrink_run_is_repeated_plain`, `gpu::tests::a_refused_arena_shrink_keeps_the_graph` (dev rig; the refusal is real ONNX Runtime) |

The review checks' evidence: `reports/gates/wait-hybrid-saturated-12cpu.md`, `wait-hybrid-saturated-6cpu.md`,
`wait-hybrid-sigil-zs.txt`.

## Reports
`reports/margin_probe.json`, `reports/cpu_conv2d_bench.json`, `reports/autotune_cpu.json`,
`reports/cpu_sweep_threads_priority.json`, and every gate summary quoted above under `reports/gates/`.
