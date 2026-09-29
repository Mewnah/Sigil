# Sigil benchmark harness

Sigil measures its own real-time voice pipeline: per-block compute, end-to-end latency, dropouts,
the latency budget's behaviour under stalls and load, CPU and VRAM. Defaults and "this engine beats
that one" claims come from these numbers, not from vendor figures (speech-stack plan, Principles
and §8).

The harness core lives in `src-tauri/src/services/bench/` (P1-F). Quality metrics, competitor
comparison, live-record and the autotuner build on it in P3.

## Entry points

| Entry point | Use |
|---|---|
| `Sigil.exe --headless bench <spec.json> [--out <dir>]` | CLI. Runs a spec, writes `report.json`, `report.md` and per-run files, prints a summary. Exit code 0 = every run passed; 1 = a run errored or a threshold failed (reports are still written); 2 = usage error. No window, no server, no port. |
| IPC `plugin:bench\|run_benchmark({ spec, outDir? })` | "Benchmark this PC" and the autotuner. Returns the report; events `bench:progress` and `bench:done`. `plugin:bench\|cancel_benchmark`, `plugin:bench\|get_last_report`. Relative paths resolve against `app_data/bench/`, and `outDir` must stay inside it (default `app_data/bench/results/<name>-<unix ms>/`). One benchmark at a time. Every accepted call ends in one `bench:done`, including errors and panics; a call refused because a benchmark is already running emits none. |
| `cargo test --manifest-path src-tauri/Cargo.toml bench` | Stub-model and fake-RVC plumbing, the latency tracer, the budget, the meters. No downloads, no devices. |
| `scripts/bench/run-perf-gate.ps1` | Dev-rig perf gate: runs specs and compares against a baseline (a p95 regression over 5 % fails). |

Quick start (from the repo root):

```powershell
cargo build --release --manifest-path src-tauri/Cargo.toml
src-tauri\target\release\sigil.exe --headless bench scripts\bench\specs\stub-smoke.json
# -> bench-results\stub-smoke-<unix ms>\report.md
```

Release builds give representative numbers. A debug build runs the harness unoptimized, adding
about 2.5 ms per 10.7 ms block on the dev rig, so its numbers only compare with other debug runs.
`bench-results/` is git-ignored. Curated scorecards go to `docs/benchmarks/*.md`.

## What runs

Every offline and paced run drives the production `VcStreamHarness`, in the worker's block size,
through the same per-block steps as a live voice-changer block. Those steps are input gain, harness
conversion, failed-block muting (`MuteConceal`), formant tilt and output gain. Paced runs also use
the voice changer's own capture ring and latency budget (`InputBacklog`), output queue
(`OutputPlayback`) and device playout (`RingPlayout`). The bench compiles these directly from
`voice_changer/{budget,playback,fallback}.rs`, so it measures the shipped policy and never a copy
that could drift.

### Engines

- `stub`: a built-in stand-in with no files. It returns identity or silence at any output rate, and
  its compute cost is configurable and seeded. It can `sleep` (a GPU wait) or `spin` (CPU work),
  and it can fail every n-th block. Use it for plumbing, latency accounting and budget behaviour.
- `rvc`: a real slot through the production `RvcModel` loader. The voice changer's rules apply:
  - the preset's default EP;
  - rmvpe runs as fcpe on the CPU EP;
  - ONNX Runtime pool threads get the worker priority (`protect_game` caps them at 2);
  - HuBERT, FCPE and RMVPE paths auto-resolve from `app_data/voice-changer/base` and then the slot.

  The EP the sessions actually landed on is reported, including fallbacks.

  Benchmark sessions record their EP outcomes as `bench.vc`, `bench.vc.content` and
  `bench.vc.pitch`, separate from the voice changer's.

The engine loads once per spec and is shared by its runs.

### Modes

**offline** feeds the whole input as fast as the engine goes. It reports the per-block compute
distribution (`block_ms_*`), `rtf_*` (compute / block duration), `over_budget_ratio`,
`throughput_x` (audio seconds per compute second) and `cpu_cores`.

**paced** is the worker loop on a simulated device clock:
- Capture callbacks deliver `capture_callback_ms` of input at their period.
- The playback device pulls `playback_callback_ms` at its period, half a period out of phase.
- On every wake the worker runs the production sequence: capture ring, then backlog, then the
  latency budget, then one block, then the output queue, ring and device.

The clock is one of three kinds:
- `wall` (default): real time. The worker sleeps until the next capture callback, injected stalls
  really sleep, and load generators contend for real. This is the authoritative mode.
- `virtual`: time advances by each block's measured compute, plus a stub's declared cost and
  injected stalls. Those are accounted but not slept. It is fast and free of sleep and scheduling
  effects, but compute times are real, so results vary with the PC's load. Use it for quick
  what-ifs with real engines.
- `simulated`: only a stub's declared compute and injected stalls advance time. The harness's and
  engine's real compute are not charged, and `block_ms` reports the declared cost. Every timeline
  metric is then reproducible on any PC under any load. Use it to test the latency, budget and
  queue logic (the cargo tests and the smoke's stall run do); it does not measure performance, so
  it reports no CPU figures.

`latency_ms_*` is the engine-added latency a listener hears. The tracer follows every sample the
device plays back to the captured sample it came from:
1. output ring frame to emitted frame (queue trims accounted);
2. emitted frame to harness input frame (minus the harness's algorithmic delay, measured by an
   identity probe);
3. harness input frame to worker block;
4. worker block to capture ring frame (budget drops accounted);
5. capture ring frame to capture frame (ring overflow accounted).

The value is *play time − capture time* of the first frame of every playback callback. It includes
the capture period, queueing, compute, the harness's algorithmic delay and the output cushion. It
excludes the devices' internal buffers, which the loopback meter measures. A cargo test checks the
tracer against the cross-correlation of what the simulated device actually played, and they agree
to within 0.001 ms.

The run also reports:
- underruns, and `underrun_ms`;
- the budget's `dropped_ms` and `drop_events`;
- `backlog_max_ms`, and `backlog_over_budget_ms`: backlog beyond budget + one block, which should
  always be 0;
- `block_age_ms_*`;
- `output_trimmed_ms`;
- the stall-recovery metrics below.

**e2e_loopback** measures real devices. An irregular train of 40 ms chirps (200 Hz to 4 kHz) is
played to `output_device` and recorded from `input_device`, and every burst is located in the
recording. Both devices must be named; a name matches exactly or by a unique part. The literal
`default` picks the system default device, which plays the chirps on your real speakers or
headset. The recording is analysed with one of two detectors:
- `xcorr` detector: a matched filter with sub-sample peak interpolation.
- `onset` detector: the envelope rise, for paths that change the timbre, such as a voice changer.

Both streams are mapped onto one time base using the drivers' stream timestamps (WASAPI QPC), with
a line fitted per stream. The residual is reported as `e2e_timebase_jitter_ms`. WASAPI's playback
timestamp is an estimate with a constant bias. To remove it, measure a direct cable once with the
same devices (for example, output `CABLE Input`, input `CABLE Output`) and pass its median as
`calibration_ms`. Results are then within ±1 ms. This mode can time any app, competitors included,
by routing its input and output through virtual cables.

> The meter plays audio. Point it only at devices you chose. If a virtual cable is also a game's or
> Discord's microphone, close those apps first: the chirps would go out on voice chat.

### Load and stalls (paced)

- `load.gpu`: a chain of `depth` square MatMuls (`size` × `size`, fp32), built in memory and run
  through Sigil's own ONNX Runtime session factory on `execution_provider` at `duty`. It stands in
  for a game rendering on the same GPU. When no GPU EP loads, the CPU fallback is reported and the
  matrices shrink to at most 256 × 256.
- `load.cpu`: `threads` busy threads at `duty` of every `period_ms`.
- Duty is controlled on accumulated busy time, so it holds on average. The achieved duty is
  reported as `gpu_load_duty` and `cpu_load_duty`.
- CPU attributed to the voice changer (`cpu_cores`) is process CPU minus the load threads' own CPU
  time. `worker_cpu_cores` is the worker thread alone. CPU time comes from Windows' cycle counters
  (`QueryProcessCycleTime` / `QueryThreadCycleTime` over the calibrated TSC rate). Tick-based
  `GetProcessTimes` only charges the thread running at each ~15.6 ms scheduler tick, so it reads
  about 0 for a worker that computes 0.3 ms and then sleeps. That matters for the plan's
  0.02-core and 0.25-core targets (T4).
- `stalls[]`: the block starting at `at_s` takes `ms` longer (a frozen GPU). The run reports two
  recovery metrics:
  - `stall_recovery_blocks`: blocks until the waiting capture is back within the budget.
  - `stall_latency_recovery_ms`: time until output latency returns to within max(10 ms, one block)
    of its pre-stall median.

  An unrecovered stall counts the rest of the run. Plan T7 wants the first metric at 2 blocks or
  fewer. A stall must start at least 1 s after `warmup_s`, so the pre-stall median has samples.
  When no latency was measured before the stall (for example, audio never started playing),
  `stall_latency_recovery_ms` is absent rather than reported as the worst case.

## Spec reference (`sigil-bench-spec/1`)

Unknown fields are errors. Threshold keys must name a metric the run produces, so a typo never
disables a gate. Relative paths resolve against the spec file's folder (CLI) or `app_data/bench/`
(IPC).

```json
{
  "name": "my-bench",
  "description": "free text",
  "seed": 7,
  "input":  { "kind": "synthetic", "signal": "voice", "seconds": 8, "level_dbfs": -12 },
  "stream": { "input_rate": 48000, "output_rate": 48000, "input_channels": 1, "output_channels": 2,
              "block_frames": 512, "extra_context_ms": 64, "crossfade_ms": 20,
              "capture_callback_ms": 10, "playback_callback_ms": 10,
              "output_prefill_ms": 35, "output_max_queue_ms": 70, "latency_budget_ms": 300,
              "pitch_semitones": 0, "index_rate": 0, "protect": 0.33, "speaker_id": 0,
              "formant_shift": 0, "input_gain": 1, "output_gain": 1, "worker_priority": "auto" },
  "engine": { "kind": "stub", "behavior": "passthrough", "output_rate": 40000,
              "min_context_samples": 720, "compute_ms": 2, "compute_jitter_ms": 0.5,
              "compute": "sleep", "fail_every": 0 },
  "runs": [
    { "mode": "offline", "name": "offline", "repeat": 1 },
    { "mode": "paced", "name": "paced", "seconds": 10, "clock": "wall", "warmup_s": 0.5,
      "load": { "gpu": { "duty": 0.7, "period_ms": 20, "execution_provider": "cuda",
                         "device_index": 0, "size": 1024, "depth": 4 },
                "cpu": { "threads": 4, "duty": 0.5, "period_ms": 10 } },
      "stalls": [ { "at_s": 4, "ms": 3000 } ],
      "thresholds": { "latency_ms_p99": { "max": 85 }, "underruns": { "max": 0 } } },
    { "mode": "e2e_loopback", "name": "cable", "output_device": "CABLE Input",
      "input_device": "CABLE Output", "input_channel": null, "bursts": 20, "interval_ms": 600,
      "burst_ms": 40, "level_dbfs": -12, "detector": "xcorr", "max_latency_ms": 400,
      "calibration_ms": 0, "warmup_ms": 500, "min_snr": 8 }
  ],
  "thresholds": { "block_ms_p95": { "max": 12 } },
  "output": { "write_timeline": true, "write_audio": false }
}
```

- The `stream` defaults are the voice changer's defaults. `block_frames` must be one of its chunk
  sizes: 160 … 48000. For example, 512 is 10.7 ms and 1536 is 32 ms at 48 kHz.
- `input.signal` is one of the following:
  - `voice`: speech-like, voiced, with pauses;
  - `tones`;
  - `chirp_train`;
  - `impulse_train`;
  - `noise`;
  - `silence`.

  `{"kind": "wav", "path": ...}` reads any PCM or float WAV and resamples it to `input_rate`.
- `rvc` engine fields:
  - `slot_dir` (required);
  - `hubert`, `fcpe`, `rmvpe` (empty = auto);
  - `pitch_extractor`: `fcpe`, `rmvpe` (alias `crepe`) or `dio`;
  - `execution_provider`: `cpu`, `cuda`, `dml`, or empty for the preset's default;
  - `preset`: `streamer`, `low_latency` or `quality`;
  - `gpu_device_index`;
  - `intra_threads`: for the plan's "1-2 threads for real-time sessions" measurement;
  - `vram_cap_mb`;
  - `silence_skip`: default true, as in the voice changer.
- Paced `clock` is `wall` (default), `virtual` or `simulated` (see Modes). `warmup_s` (default
  0.5) excludes start-up priming from `latency_ms_*` and must be below `seconds`.
- A stall must start at least 1 s after `warmup_s` and end at least 1 s before its run ends, so
  there is a baseline and recovery is observable.
- `e2e_loopback` needs `output_device` and `input_device`. Use `"default"` only if you mean the
  system default device.
- Thresholds follow these rules:
  - Which runs produce a metric depends on the mode and on the run's configuration. The stall
    metrics need `stalls`, the load duties need that load, and `cpu_cores` / `worker_cpu_cores`
    need a `wall` or `virtual` clock.
  - A run-level threshold must name a metric that run produces, or the spec is rejected.
  - A spec-level threshold must apply to at least one run of the spec. It applies to every run
    that produces the metric and is skipped for the others.
  - A run that should produce a metric but did not report it fails the threshold as "not
    measured", at either level. For example, a paced run that never played audio has no
    `latency_ms_*`, so a latency gate cannot pass by omission.
  - A run-level threshold overrides a spec-level one on the same metric.
  - Limits are inclusive.

## Metrics

Distribution metrics are `<prefix>_<suffix>` with suffixes `mean`, `stddev`, `min`, `p50`, `p90`,
`p95`, `p99`, `p999` and `max`. Percentiles are linearly interpolated between the closest ranks
(NumPy's default). All times are in ms. Lower is better except for `throughput_x`,
`e2e_detected` and the load duties.

| Mode | Distributions | Scalars |
|---|---|---|
| offline | `block_ms`, `model_ms`, `rtf` | `blocks`, `over_budget_ratio`, `model_errors`, `algorithmic_latency_ms`, `engine_load_ms`, `harness_init_ms`, `cpu_cores`, `worker_cpu_cores`, `vram_engine_mb`, `throughput_x` |
| paced | the offline ones, plus `latency_ms`, `block_age_ms` | the offline scalars except `throughput_x`, plus `underruns`, `underrun_ms`, `dropped_ms`, `drop_events`, `overflow_ms`, `backlog_max_ms`, `backlog_over_budget_ms`, `stall_recovery_blocks`, `stall_latency_recovery_ms`, `output_trimmed_ms`, `gpu_load_duty`, `cpu_load_duty` |
| e2e_loopback | `e2e_ms` | `e2e_detected`, `e2e_missed`, `e2e_timebase_jitter_ms` |

Notes on specific metrics:
- `algorithmic_latency_ms` is the harness's own delay: block accumulation, the input resampler,
  the context ring, SOLA, the output resampler and the priming cushion. An identity model with the
  run's rates and context measures it by cross-correlation. Model-internal delay is not included.
- `vram_engine_mb` is this process's DXGI local VRAM after the engine loaded, minus before.
- A metric that the PC cannot measure is absent: for example, VRAM without DXGI, or CPU time
  outside Windows. A threshold on it then fails as "not measured".

## Reports

Each report folder contains:
- `report.json` (`sigil-bench-report/1`), which holds:
  - the pass/fail result and the failures;
  - the spec, and `spec_sha256` (the same spec gives the same hash);
  - the environment: Sigil version and build, OS, CPU, logical cores, ONNX Runtime version, GPUs,
    and whether the voice changer was running;
  - the engine (description, rates, requested and active EP with the fallback reason, load time);
  - per run: status (`passed`, `failed`, `error`, `cancelled`), metrics, the threshold results,
    mode-specific `details` (stall records, load report, algorithmic-latency probe, devices), and
    the files written.
- `report.md`: the same content, readable.
- `<run>.timeline.csv`: one row per block, with columns `tick`, `t_ms`, `age_ms`, `compute_ms`,
  `model_ms`, `stall_ms`, `waiting_ms`, `dropped_ms`, `queued_out_ms` and `error`.
- `<run>.output.wav` (with `output.write_audio`): the converted audio. For offline runs this is
  the harness output; for paced runs it is what the simulated device played, underrun holds
  included.

## Perf gate

```powershell
pwsh scripts/bench/run-perf-gate.ps1 -OutRoot bench-results/baseline   # before a change
pwsh scripts/bench/run-perf-gate.ps1 -Baseline bench-results/baseline  # after it
```

The gate runs every spec in `scripts/bench/specs/`, or the ones passed with `-Specs`. It fails
when either of these happens:
- the harness exits non-zero, because a run errored or a spec threshold failed;
- a gated metric regressed by more than `-MaxRegressionPct`.

The details:
- Gated metrics default to every `*_p95`. `-Metrics` takes wildcard patterns, as an array or as
  one comma-separated string. A baseline gate whose patterns match no metric fails.
- `-MaxRegressionPct` defaults to 5 %, the plan's rule for voice-changer changes. A regression
  counts only when it is also at least `-MinDelta` (default 0.25) in absolute terms.
- A metric whose baseline is 0 has no percentage, so any worsening of at least `-MinDelta`
  regresses (0 underruns before and 50 now fails).
- Every gated metric in the baseline must still be in the current report. A metric that is gone
  fails as MISSING, whether its run errored, measured nothing, or was removed.
- The gate warns when the spec or the build type (debug or release) differs from the baseline, and
  about runs that are new since the baseline.
- It writes `gate-summary.md` and `gate-summary.json`.

## Inputs: synthetic clips and the corpus

- `python scripts/bench/make-test-clips.py --out <dir>` writes the synthetic signals as WAV files.
  It uses the standard library only. `--mix voice music_bleed:-20` writes a voice clip with world
  music bleed at -20 dB, and `--float` writes 32-bit float.
- The real corpus is about 20 consented recordings from you and friends, the same clips with game
  or music bleed, and a singing set. It lives in `app_data/bench/corpus/`
  (`%APPDATA%\com.sigil.app\bench\corpus`) and is never committed.
  `scripts/bench/corpus.example.json` is the manifest template. IPC specs reference clips as
  `{"kind": "wav", "path": "corpus/clips/speech-01.wav"}`.

## Specs in the repo

| Spec | What it checks |
|---|---|
| `scripts/bench/specs/stub-smoke.json` | Plumbing: stub offline, a wall-clock paced run under a light CPU and ONNX Runtime load, and a 3 s stall on the virtual clock. Its thresholds only catch broken plumbing, and they hold on a busy PC and in debug builds. |

## First measurements (dev rig, 2026-09-28)

These are the harness's first numbers, the baseline on ONNX Runtime 1.28.0 (rc.13). They were taken
on the dev rig: an AMD Zen 4 CPU with 16 logical cores and an RTX 4090. The build was release, and
the stream used the defaults (48 kHz mono in, stereo out, 512-frame blocks = 10.7 ms, 64 ms extra
context, 20 ms crossfade, 35 / 70 ms output cushion, 300 ms budget). The engine was the stub:
identity at 40 kHz.

| What | Result |
|---|---|
| Harness cost per block (resamplers, ring, SOLA, post, zero-cost model) | 0.045 ms p50, 0.06 ms p99 (RTF 0.005). A debug build adds about 2.5 ms. |
| Algorithmic latency (identity probe) | 19.37 ms |
| Engine-added latency, wall clock, 2 ± 0.5 ms model, light CPU + ORT load | 74.37 ms, constant (p50 = p99), 0 underruns |
| Voice-changer CPU in that run (process minus load threads) | 0.013 cores |
| 3 s stall (plan T7) | Waiting capture 3003 ms → 150 ms at the very next wake (0 blocks); 2853 ms dropped in one step; backlog never above budget + one block |

Repeat runs agree within about 3 % on `*_p95` (`run-perf-gate.ps1` comparison), and the simulated
stall run is bit-identical. No real RVC voice was benchmarked, because none is on the dev rig. The
real `RvcModel` path is covered by cargo tests with a hand-encoded stand-in slot on the CPU EP. The
first real-voice run on CUDA, DirectML and CPU becomes the baseline for P3's engine work.

**Finding (for the P3 runtime loop):** after a stall, output latency settles about 20 ms *higher*,
at 94.4 ms against 74.4 ms, and stays there. The catch-up blocks refill the output queue up to
`output_max_queue_ms` (70 ms) rather than the prefill (35 ms), and input and output run at the same
rate, so the queue never drains back. With `output_max_queue_ms` = prefill + 5, latency recovers
within a fraction of a second. `stall_latency_recovery_ms` reports this, and `latency_end_p50_ms` in
the stall record shows where latency settled.
