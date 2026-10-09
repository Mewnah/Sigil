# P3B-A specs

Bench trust I (perf gate, attribution, loads, VRAM).

Benchmark specs (`sigil-bench-spec/1`) and spec templates of workstream P3B-A live here (P3B rule R7:
each workstream keeps its specs under `scripts/bench/specs/p3b/<letter>/`; only P3B-A edits
`scripts/bench/profiles/`, the others send profile rows to P3B-I0).

Every timing run holds the GPU timing lock (`pwsh scripts/bench/gpu-lock.ps1 -Acquire <tag>` /
`-Release`, or `run-perf-gate.ps1`, which takes it per run; rule R2), and before/after claims come
from ABBA-interleaved reruns (rule R3). Record each spec's command and numbers in the workstream
report.

The templates are materialized by `run-perf-gate.ps1 -Profile p3b` (variables in
`scripts/bench/profiles/p3b.json`; `-Slot`, `-Clip`, `-Var NAME=value`):

| Template | Profile rows | What it measures |
|---|---|---|
| `sve-gpu.template.json` | `sve-dml`, `sve-cuda`, `sve-cuda-levers`, `sve-dml-ctx512` | The scorecard spec: offline (T2 idle), paced (T1 and its decomposition, T4 GPU CPU), the in-process 70 % hammer, the out-of-process DirectML load (`h2_load70_oop`, a game's process and queue) and the H1 simulation. `-Var BUILD=direct` adds the per-component breakdown. |
| `dml-on-cuda.template.json` | `sve-cuda-dmlload` | Advisory: the CUDA engine under the conv hammer on DirectML in the same process (`h2_load70_dml`, a D3D12 queue beside the CUDA context), between the same-context hammer and the separate process. |
| `sve-cpu.template.json` | `sve-cpu`, `sve-cpu-pool` | The CPU engine at 120 ms blocks (T4, T1 Streamer): `sve-cpu` is the shipped Streamer preset (ONNX Runtime on one thread), `sve-cpu-pool` the low_latency preset on ONNX Runtime's default pool (`CPU_PRESET`, the T4 reference rows). QoS: compare `QOS=app` (the app's EcoQoS opt-out, default) against `QOS=none` with `-CompareVars QOS=none`. |
| `aa.template.json` | `aa-dml`, `aa-cuda` | The gate's resolution: run with `-CompareExe <the same exe>` in at least two sessions; every row should be NEUTRAL; the session's `gate-summary.json` feeds `-AASummary`. |
| `load.template.json` | `load-dml`, `load-cuda`, `load-cuda-fcpe`, `load-cpu`, `load-quality-cuda` | One cold load per process with its wall-clock breakdown by stage (T9 / T4-WS0); five gate runs give five fresh processes. |
| `vram.template.json` | `vram-dml`, `vram-cuda`, `vram-cuda-levers` | The load-peak sampler, the measured ledger, the CUDA context, `cuMemGetInfo` / `cuCtxGetLimit`, then `load_twice` (`hot_slots = 2`). Steady state: the session plans' holds (`ledger`) plus the CUDA context against the steady delta (`ledger_unattributed_mb`). Load time: what each component's spans allocated (`measured`), transient copies included (`measured_over_steady_mb` is about the load overshoot). Both are process-wide DXGI deltas: with the F0 helper on, the RMVPE plan builds beside the content plan and the two read each other's allocations (`measured.*.concurrent`); run once more with `SIGIL_VC_F0_OVERLAP=0` in the environment for a clean per-component split. |
| `knn30k.template.json` | `knn30k-cuda` | The GPU kNN with a synthetic 30k-row index (`make-synthetic-index.py`, written locally, never committed): the Quality-with-index VRAM row and retrieval's per-call cost. |

Examples:

    # A/A (the gate's resolution), the same exe on both sides
    pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <exe> -CompareExe <exe> -Only aa-dml,aa-cuda -Slot <voice> -Clip <wav>
    # Per-stage attribution under every load (direct build)
    pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <exe> -Only sve-dml,sve-cuda -Var BUILD=direct -Slot <voice> -Clip <wav> -TargetsAdvisory
    # Whether the app's EcoQoS opt-out matters on the CPU (B = Windows-managed QoS)
    pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <exe> -CompareExe <exe> -CompareVars QOS=none -Only sve-cpu -Slot <voice> -Clip <wav>
    # A change against the P3B baseline exe (exe-s0 predates engine.qos: QOS=none on both sides)
    pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <new> -CompareExe <exe-s0> -Var QOS=none -AASummary <aa>/gate-summary.json -Only sve-dml

## Environment switches of the benchmark (P3B-A)

| Variable | Effect |
|---|---|
| `SIGIL_BENCH_STATE` | `off`: no GPU / host state probes (an A/B against an exe without them); `light`: the probes without the GPU speed index; default: full (NVML clocks / P-state / reasons / other processes at run start, end and 1 Hz, the speed index on DirectML or without NVML, the host probe, power plan and QoS, the CPU's effective clock). |
| `SIGIL_BENCH_GPU_TIMING=1` | Session plans time each run with CUDA events (P3B-G) while the spec records: `gpu_<content\|pitch\|enc\|flowdec\|knn>_ms_*`. Off by default (the events add a little work per run). |
| `SIGIL_ORT_PROFILE=<dir>` | ONNX Runtime's per-node profiler on every rung of the matching sessions; each session writes `<dir>/<component>-<ep>_<timestamp>.json` when it is dropped. `SIGIL_ORT_PROFILE_FILTER=<prefix>` limits it to components starting with the prefix (`vc.enc`, `bench.vc.content`, ...; empty = every session). Tooling only: profiling slows every run, so never time with it on. |
| `SIGIL_VC_F0_OVERLAP=0` | The engine's F0 helper off (P3B-D's kill switch): F0 runs after the content encoder on the worker. The VRAM specs use it for a clean per-component ledger (the RMVPE plan then builds on the loading thread instead of beside the content plan). |

Every spec records `neural_audio::telemetry` while it runs: stage spans per call (`content_ms`, `f0_ms`,
`enc_ms`, `flowdec_ms`, `retrieval_ms`, `post_ms`, plus `glue_ms` = the call minus the union of its
stages, `overlap_ms` = stage time that ran beside another stage, `harness_ms` = the tick's worker time
minus its model calls), host waits on the GPU (`syncs_per_call`), onset calls apart
(`call_ms_onset_*` / `call_ms_steady_*`), the SOLA joins (`join_offset_ms_*`, `join_step_ms_*`,
`join_ncc_*`), the paced T1 decomposition (`t1_<part>_ms_*`, `t1_at_p50_<part>_ms`,
`t1_residual_p50_ms`), the engine load's wall-clock breakdown (`load_<stage>_ms`, `load_other_ms`,
`load_attributed_ratio`, `generator_builds`; `report.engine.attribution.load_breakdown_ms`), the VRAM
figures (`vram_load_peak_mb`, `vram_load_overshoot_mb`, `vram_cuda_context_mb`,
`vram_cuda_device_delta_mb`, `cuda_stack_limit_kb`; `report.engine.attribution.vram`), worker and
process cycle counts (`details.cycles`: effective GHz, core-seconds per audio-second) and each run's
GPU / host state (`state_*` metrics, `details.state` with its warnings, naming the other process that
used the GPU). The per-session ORT arena statistics on CUDA need the session owners'
`telemetry::arena_mark_session` call (see the P3B-A report's hotspot requests).

## Reading an interleaved gate summary

Every paired row carries the verdict over all pairs and, when some pairs ran with their two sides in
different GPU clock regimes (the run's median SM clock, or the speed index without NVML, more than
`-RegimeTolerancePct` = 25 % apart: the driver held the GPU at a lower clock for a while, typically
while another process used it), the same statistics without them (`matched_*`, information only).
A wide interval with a regime-mismatched pair means the session saw the GPU change state, not that
the builds differ: check `state_flags` and rerun the session on a quiet GPU. On this rig the A/A
sessions resolved call_ms_p95 to +-1.3..4.3 % offline and under the 70 % hammer once such pairs are
set aside; paced rows (the real-time cadence lets the GPU drop clocks between ticks) resolve to
+-3..13 %, so size their rounds from an A/A session (`-AASummary`).

Reading the verdicts:

- IMPROVED needs an interval entirely below 0 and a median at least `-ImprovedFloorPct` (1 %) better:
  the A/A's false IMPROVED was a -0.87 % shift, a statistically clear but negligible one, which now
  reads NEUTRAL. REGRESSED and INCONCLUSIVE are unchanged (median above +5 %).
- The gate warns for every spec with gated rows whose interval is wider than the regression limit
  (half-width above `-MaxRegressionPct`): it cannot tell a 5 % change from noise there. Paced rows
  and stage metrics (`enc_ms_p95`, `f0_ms_p95`, gated through `-Metrics`) are the usual ones: size
  their rounds from an A/A session (`-AASummary`), or pass `-AllowInconclusive` so noise cannot fail
  the gate at `-MaxRounds`, and read those rows as unresolved.
- The T-targets of an interleaved session read the **median of B's rounds** (per metric), never one
  round: a single round's p95 carries the rig's 11-19 % per-pair spread.
- Onset calls (`onset_calls`, `call_ms_onset_*`, `call_ms_steady_*`) are the calls that analyse
  clearly more F0 frames than the run's steady calls (the run's most common frame count plus 4 frames
  or 25 %; `details.attribution.onset_above_f0_frames`), so the split holds at any block size and
  extractor lead.

The GPU timing lock's path is `-LockPath`, else `SIGIL_GPU_LOCK`, else the first line of
`%LOCALAPPDATA%\sigil-bench\gpu_lock_path.txt` (how the agents of a shared rig agree on one lock
without a path in the repository), else `%LOCALAPPDATA%\sigil-bench\gpu_timing.lock`;
`gpu-lock.ps1 -PrintPath` prints the resolved one.

