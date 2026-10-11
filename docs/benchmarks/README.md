# Sigil benchmark harness

Sigil measures its own real-time voice pipeline: per-block compute, end-to-end latency, dropouts,
the latency budget's behaviour under stalls and load, CPU and VRAM. Defaults and "this engine beats
that one" claims come from these numbers, not from vendor figures (speech-stack plan, Principles
and §8).

The harness core lives in `src-tauri/src/services/bench/` (P1-F). P3-F added real voices through
the voice changer's engine factory, fidelity against the offline ceiling, named contention
profiles, competitor comparison, in-app live recording and the per-PC autotuner. The voice-engine
scorecard against the plan's targets T1–T10 is [VC_BENCHMARKS.md](VC_BENCHMARKS.md).

## Entry points

| Entry point | Use |
|---|---|
| `Sigil.exe --headless bench <spec.json> [--out <dir>]` | CLI. Runs a spec, writes `report.json`, `report.md` and per-run files, prints a summary. Exit code 0 = every run passed; 1 = a run errored or a threshold failed (reports are still written); 2 = usage error. No window, no server, no port. |
| IPC `plugin:bench\|run_benchmark({ spec, outDir? })` | "Benchmark this PC" and the autotuner. Returns the report; events `bench:progress` and `bench:done`. `plugin:bench\|cancel_benchmark`, `plugin:bench\|get_last_report`. Relative paths resolve against `app_data/bench/`, and `outDir` must stay inside it (default `app_data/bench/results/<name>-<unix ms>/`). One benchmark at a time. Every accepted call ends in one `bench:done`, including errors and panics; a call refused because a benchmark is already running emits none. |
| `cargo test --manifest-path src-tauri/Cargo.toml bench` | Stub-model and fake-RVC plumbing, the latency tracer, the budget, the meters. No downloads, no devices. |
| `scripts/bench/run-perf-gate.ps1` | Dev-rig perf gate: runs specs and compares against a baseline (a p95 regression over 5 % fails). `-Profile p3` also materializes the real-voice templates and checks the plan's T-targets. |
| `scripts/bench/gpu-lock.ps1` | The dev rig's GPU timing lock (P3B rule R2): `-Acquire <tag>` / `-Release` / `-Run <tag> -Command ...` / `-Status`. Every timing-sensitive run (any `--headless bench` on a GPU or the CPU, ignored GPU cargo tests, ORT profiling, Python microbenchmarks) holds it; `run-perf-gate.ps1` takes it for each run (`-NoGpuLock`, `-GpuLockWait`, `-GpuLockTag`). A fair queue keeps an ABBA series from starving other waiters. Path: `-LockPath`, else `SIGIL_GPU_LOCK`, else the session lock, else `%LOCALAPPDATA%\sigil-bench\gpu_timing.lock`. Never hold it across a build. |
| `Sigil.exe --headless gpu-load <spec.json>` | The benchmark's out-of-process GPU load (a DirectML convolution at a set duty, in its own process in a KILL_ON_JOB_CLOSE job, reporting the duty it achieved). The `h2_load70_oop` and `h2_load70_dml` contention profiles start it; you do not run it by hand. |
| `scripts/bench/run-matrix.ps1` | One voice through a matrix of EP × F0 method × block × precision × context × levers; writes `matrix.md` / `matrix.csv`. |
| IPC `plugin:voice-changer\|voice_changer_autotune({ request })` | The per-PC autotuner (the voice changer must be stopped). Progress on `voice_changer:autotune_progress`; the result is stored per GPU and driver in `app_data/voice-changer/autotune.json` and its report goes to `app_data/bench/results/autotune-<ms>/`. Also `voice_changer_cancel_autotune`, `voice_changer_get_autotune`. |
| IPC `plugin:voice-changer\|voice_changer_live_record_start({ request })` / `voice_changer_live_record_stop` | Records the running voice changer's per-tick timeline for up to 10 minutes into `app_data/bench/results/live-<ms>/` (no audio, no paths: shareable as it is). |

Quick start (from the repo root):

```powershell
cargo build --release --manifest-path src-tauri/Cargo.toml
src-tauri\target\release\sigil.exe --headless bench scripts\bench\specs\stub-smoke.json
# -> bench-results\stub-smoke-<unix ms>\report.md
```

Release builds give representative numbers. A debug build runs the harness unoptimized, adding
about 2.5 ms per 10.7 ms block on the dev rig, so its numbers only compare with other debug runs.
`bench-results/` is git-ignored. Curated scorecards go to `docs/benchmarks/*.md` (speech: [SPEECH_BENCHMARKS.md](SPEECH_BENCHMARKS.md)).

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
- `rvc`: a real voice, resolved with the voice changer's own rules (`voice_changer::params` and
  `capabilities`, never a copy) and built by the voice changer's engine factory, so the benchmark
  measures exactly what the user runs:
  - template voices (`sigil-rvc-1`, imported from `.pth` / `.zip`) run on the Sigil Voice Engine;
    bring-your-own ONNX slots (`rvc`) run on the legacy `RvcModel`;
  - the preset's default EP; accelerator ids run on their ladder rung (`tensorrt_rtx` on CUDA,
    `dml_hp` on DirectML);
  - an F0 method this build cannot run runs its stand-in, and GPU-only methods fall back on the
    CPU EP, with the voice changer's warning;
  - the levers resolve through `resolve_levers` (request, availability, the preset's `auto` table);
  - ONNX Runtime pool threads get the worker priority (`protect_game` caps them at 2);
  - empty model paths resolve like the voice changer's: a template voice's content encoder and
    pitch models from the model store (its embedder's entry, `vc/pitch-*`), and the legacy
    candidates in `app_data/voice-changer/base`, then the slot folder.

  The EP every session actually landed on is reported per component, including fallbacks.
  Benchmark sessions record their EP outcomes under the `bench.` prefix (`bench.vc`,
  `bench.vc.content`, `bench.vc.pitch`, `bench.vc.enc`, `bench.vc.flowdec`, ...), separate from the
  voice changer's. `build: direct` builds a template voice with the benchmark's own builder (the
  same parts) and adds the per-component compute breakdown (`content_ms`, `f0_ms`, `enc_ms`,
  `flowdec_ms`) and the `lookahead_ms` knob.

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

**quality** (plan T5) streams the spec's input, led by `lead_s` of silence (default 1 s), through
the harness exactly like an offline run while recording the 16 kHz stream the model received at
its absolute positions. The same voice then converts that recorded stream offline: the **offline
ceiling**. Dynamic-shape models (bring-your-own ONNX) convert it in one call (20 s pieces beyond
30 s); a template voice is fixed-shape, so a second instance with long shapes converts it block
after block with no crossfade: `ceiling.block_ms` (200), `ceiling.extra_context_ms` (500, never
less than the run's), `ceiling.lookahead_ms` (80) and `ceiling.exact` (true: full decode instead
of partial decode; the F0 cache, anchored noise and phase0 stay, see
[VC_BENCHMARKS.md](VC_BENCHMARKS.md#the-offline-ceiling)). Its time-anchored noise is keyed to the
same absolute frames, so the two renditions differ only by what streaming changes. Voice families
(P5-I0) use their own ceiling: Sigil-ZS and so-vits-svc voices convert the whole recording as a
one-off call of the stream's model, DDSP-SVC through a second instance configured for the whole
recording; MeanVC2 (its offline path is the streaming loop) and sidecar voices have none, and the
run says why. A family voice's paced runs play with at least the family's output buffer
(`engine.min_output_buffer_ms` in the report: Sigil-ZS 80 / 150 ms, DDSP-SVC 70 / 140 ms), as the
voice changer does, and its `vram_engine_mb` is read after one harness build. The stream is
placed against the ceiling from the timeline (the ceiling's absolute start, where the input
entered the recorded stream, the probed harness latency) and cross-checked on energy envelopes;
a search is used only when the timeline cannot be derived. Scored over the ceiling's speech (100 ms
margins):
- `mcd_db`: mel-cepstral distortion (c1..c24 of 40 mel bands, 25 / 10 ms frames, speech frames);
- `f0_rmse_cents`, `gross_pitch_error` (> 20 %), `vuv_error`: one F0 analyzer for both (offline
  RMVPE when installed, else DIO; `scoring.f0_method`);
- `speaker_cosine`: CAM++ (store entry `bench/campplus`); `speaker_cosine_reference` against a
  clip of the target voice (`scoring.speaker_reference`);
- `dnsmos_ovrl` / `_sig` / `_bak` (DNSMOS P.835, `bench/dnsmos-p835`) of the stream,
  `ceiling_dnsmos_ovrl` and `dnsmos_ovrl_delta`;
- `clicks`: impulsive events of the stream that the ceiling does not have, in the join windows
  (a call's first returned samples plus the crossfade), in excess of the artifact rate away from
  the joins plus two standard deviations; `clicks_per_10min`, `artifact_events`;
- `seam_flux_ratio` / `ceiling_seam_flux_ratio`: spectral flux at the block period's phases, max
  over median (about 1 when joins are inaudible);
- with `stt` (any Sigil STT engine, e.g. Whisper with a local model): `cer` of the stream,
  `cer_ceiling`, `cer_delta_pts` (T5: ≤ 1 pt), against `stt.reference_text` or the input's own
  transcript;
- `ceiling_compute_ms`, `ceiling_rtf`, `offset_ms`, `quality_seconds`, plus the offline run's
  compute metrics.

A scorer whose model is not installed is skipped with a note (its metrics are then absent, and a
threshold on them fails as "not measured"). `Sigil.exe --headless fetch bench/campplus` and
`bench/dnsmos-p835` install them (about 30 MB together); normal users never need them.

**contention** is a paced run under a named load profile, with the paced run's metric names:

| profile | GPU load | CPU load | stall |
|---|---|---|---|
| `h1_vrchat_obs` (default) | conv hammer 256 × 256 × 8 layers, 70 % duty, 16.7 ms period (a 60 fps frame) | 3 threads at 50 % (game, render, OBS) | 3 s at 40 % of the run |
| `h2_load70` | conv hammer, 70 % duty (plan H2) | - | - |
| `h3_cpu_game` | - | 4 threads at 60 % (VRChat on a 6-core CPU) | - |
| `stall` | - | - | 3 s at 40 % of the run |
| `custom` | `load` only | `load` only | `stalls` only |

`load` / `stalls` in the run replace the profile's. The GPU load runs on the engine's own GPU EP
(DirectML when the engine runs on the CPU) unless `gpu_execution_provider` names one;
`gpu_load_kind` is `conv` (default: the in-memory `bench/gpu-load` convolution chain) or `matmul`.
The H1 rows of the scorecard come from here until a friend sends a live-record export.

**compare** (plan T6) scores competitor WAVs of the same input (RVC WebUI realtime, Applio
realtime, w-okada, ...; drivers and procedure in `scripts/vc-bench/README.md`) with the same
metrics as Sigil's own output. Each candidate is aligned to the input on energy envelopes and
scored over the input's speech: DNSMOS, F0 against the input's F0 shifted by `semitones`, CAM++
against `scoring.speaker_reference`, artifact events per minute, and the latency given for it
(`latency_ms`, from an `e2e_loopback` run through that app). Sigil's own row (`sigil.include`,
default on) converts the input with the spec's engine; its latency comes from `sigil.latency_ms`
or another run of the spec (`sigil.latency_from`, `sigil.latency_metric`). Metrics: Sigil's row
(`sigil_*`) and the T6 margins: `dnsmos_ovrl_margin` = Sigil − the best competitor (T6: ≥ 0.05),
`gross_pitch_error_margin` = the best competitor − Sigil (≥ 0), `latency_ratio` = Sigil / the
fastest competitor (≤ 0.7). The blind A/B test of T6 uses the same WAVs: `tools/listening-test`.

**compare v2** (P3B-C; T6 as the P3B plan defines it) scores a corpus instead of one clip, with
`compare.corpus` pointing at a `sigil-compare-corpus/1` manifest (template `scripts/bench/corpus.example.json`,
parsed by a test; `compare.candidates` are ignored then). Every compare run also reports the v2 rows in
`details.v2`. The manifest lists clips (with `sha256`, optional EGG F0 truth and neutral references),
the operating points of other systems (one WAV per clip, `{clip}` in the path, plus a driver sidecar
`sigil-vc-bench-sidecar/1`), live captures, Sigil's own rows (`sigil.configs`: `shipping_default`,
`autotuned`, `low_latency`, `quality`, built in-process), ablations, pre-registered margins, `min_clips`,
the F0 analyzers, the loudness target (-26 LUFS, BS.1770-4), `consensus_min_agree`, stand-in references
and `dump_f0`. Fairness: a rendition with another transpose, another voice or an input hash that does not
match is refused; missing metadata is flagged; latencies compare only within a tier and latency kind (a
live offset nobody calibrated is `uncalibrated` and compares with nothing);
offline points count only when per-block p99 is below the block on this run's GPU (unknown is undecided,
never a silent pass); Sigil's rows meet the same rules. The sidecars of all of a point's clips are
merged (they must agree on the voice, transpose and block, or the point is refused; the per-block
compute is the worst p95 / p99 over the clips). A live capture's sidecar carries only the voice the
manifest asserts for the point. A point that misses clips (a missing, refused or muted rendition, an
alignment failure, a conversion error, no DNSMOS where the clip needs one) is incomplete
(`unscored_clips`, `points_incomplete`, `clips_unscored`) and decides nothing, a Sigil row included.
A competitor whose index rate, protect, loudness mix or F0 method differs from the Sigil rows' is
flagged (`operating point: ...`, listed in `t6.caveats`). Stand-in references are scored apart
(`mcd_stand_in_db`; ablations only), so the T6 MCD row needs upstream references. A Sigil row's GPU is
the adapter its engine ran on. The blind A/B manifest leaves out refused and real-time-infeasible
points (`make_test.skipped`). Scoring: DNSMOS on loudness-normalised clips of at
least 9.01 s (shorter clips get an advisory tiled score), F0 against EGG truth or the analyzers' consensus
under every output analyzer, envelope and STFT MCD to the neutral references, CAM++ exact similarity to a
leave-one-clip-out target set, artifacts per minute, seam clicks on each point's own block grid and the
onset level after 300 ms of silence. Statistics are per-clip paired differences with the Student t 95 %
interval deciding (Wilcoxon / Hodges-Lehmann and a clip bootstrap cross-check; a disagreement is flagged
fragile). Per competitor point: DNSMOS +0.05 superiority at matched latency, GPE (under every analyzer)
and MCD non-inferiority, and latency at most 0.7x as a Pareto row with non-inferiority on every quality
row. A row is Pass, NotShown, Fail, Undecided, Refused or Excluded, and `t6_pass` is 1 only if every
counted point passes every row. Metrics: `corpus_clips`, `corpus_clips_dnsmos`, `points_*`,
`renditions_refused`, `t6_pass`, `t6_undecided`, `t6_rows_*`, `dnsmos_paired_sd`, `clips_needed_dnsmos_50` /
`_80` and `ablation_<name>_<metric>_diff_mean|_sd|_lo|_hi|_clips`. `scripts/vc-bench/compare_export.py` writes
`make_test.json` (aligned, trimmed, loudness-matched pairs for the blind A/B), `pareto.svg` and `t6.md`;
`scripts/bench/fetch-public-corpus.py` fetches LibriSpeech test-clean and CMU ARCTIC (EGG truth) one
download at a time (about 646 MB, resumable; `--dry-run`); `make-test-clips.py --local-corpus` builds a
10-clip corpus from the dev cache. The exact manual commands for the competitors are in
`scripts/vc-bench/README.md`; nothing installs a competitor.

**autotune** runs the per-PC autotuner on the spec's `rvc` engine (`request`: the
`AutotuneRequest` of `voice_changer_autotune`) and reports the chosen configuration
(`autotune_*` metrics; `details` holds the result with every candidate measured). The search, per
EP and then across EPs:
1. EPs: those the request names, else every EP the build offers. The voice's generator sessions
   decide (`vc.enc` / `vc.flowdec` of template voices, `vc` of ONNX voices): one on another rung
   drops the EP with its reason, and so does `tensorrt_rtx` when no generator session runs
   TensorRT (the CUDA rung is tuned as `cuda`; classic TensorRT counts, with a note). A pitch model
   that falls back drops only its F0 method; another component that falls back (e.g. the content
   encoder) is measured as it runs, with a note.
   On the CPU, the thread policy comes next: the reference configuration with one ONNX Runtime
   thread (pinned explicitly: the Streamer preset under `protect_game` also runs one, but on `auto` it
   takes its own 4-6 thread policy since P7-I0), two (`worker_priority: protect_game`) and the default pool (one
   per physical core, below-normal priority); the fewest threads that meet the limit win (least
   load next to a game), else the fastest. An explicit priority is kept: `protect_game` never
   gets the pool, `protect_voice` is never demoted to two threads. The result then carries
   `rvc_preset` and `worker_priority`.
2. F0 method: every method the build runs on the rung, ranked by accuracy (gross pitch error +
   V/UV error against the synthetic voice's exact F0, measured as each method runs on the rung;
   a user clip only times the engine), then the most accurate one that meets the RTF limit at
   the reference block (40 ms on a GPU, 80 ms on the CPU). When none does, the most accurate one
   within 20 % of the fastest one's RTF (the F0 methods cost about the same next to the voice, and
   one measurement varies by about that much).
3. Block, ascending (template voices: 20 / 40 / 60 / 80 / 100 ms engine blocks; ONNX voices: the
   voice changer's chunk sizes from 20 to 100 ms): the first that meets the limit. On the CPU the
   sweep continues to 120 / 160 / 200 ms (ONNX voices: chunk sizes up to 200 ms) when nothing
   shorter meets it: a voice that keeps up at a higher latency beats one that cannot.
4. Context: the largest extra context (64–512 ms) that still meets the limit.
5. Precision (template voices on a GPU): fp32 when it is within 5 % of fp16.
6. GPU levers (template voices on a GPU), measured under the H2 load (the conv hammer at 70 % on
   the same GPU), where they matter: from every lever off, IOBinding with CUDA graphs (CUDA;
   graphs need IOBinding), else IOBinding alone, then the priority stream / queue, each set kept
   when it lowers the loaded p99 RTF by 5 % or more; the winner is then measured idle again. Only
   the levers this stage measured go into `params`: stages 2–5 measure with the voice changer's
   own lever settings, which a lever keeps when the stage does not run (out of time) or never
   tried it.
7. Fidelity: the winner is scored against its offline ceiling (T5: CAM++ ≥ 0.97 when installed,
   F0 RMSE ≤ 15 cents, V/UV ≤ 3 %, no seam clicks). A failing block steps up, twice at most; a
   larger block is kept only when it passes or cuts the T5 shortfall by a quarter (never when it
   misses the RTF limit the smaller one met).

"Meets the limit" = p99 RTF ≤ `rtf_limit` (default 0.5: headroom for a game), at most 0.1 % of
the model calls over budget and no model errors, measured on the virtual clock (real compute, no
sleeps). A configuration that misses the limit while its median call meets it is measured once
more and the better run counts: a PC busy for a few seconds must not decide the preset.
Across EPs the lowest p50 latency wins. When nothing meets the limit, the fastest configuration is
returned with a note, so the result is always a valid set of parameters.

Every candidate runs with the voice changer's own settings besides the knobs the search sweeps:
`request.conditions` (`TuneConditions`: device rates and block, preset, crossfade, output cushion
and latency budget, worker priority, retrieval and the hot voice controls, F0 range, denoiser,
lookahead, the lever settings, GPU index and VRAM cap; default: the voice changer's defaults, and
in a spec, the spec's `stream` and `rvc` engine). A `gtcrn` denoiser runs inside the measured
engine, as in the voice changer's RVC mode; `dfn3` / `dpdfnet` run on the capture copy, outside
it, and the result says so. The result records the conditions.

In the app, the `voice_changer_autotune` command fills the voice, models, device block, preset and
conditions from the voice changer (the device rates from its saved devices' formats), refuses to
start while the voice changer runs, claims the one measurement slot it shares with "Benchmark this
PC" in one atomic step (a benchmark and a tune never run together), and fails without storing
anything when the voice changer starts during the tune. It stores the result per GPU and driver,
and the UI applies its `params` with `set_voice_changer_params`. `voice_changer_get_autotune`
returns the stored result with `stale`: what changed since the tune (another voice, a different
crossfade, cushion, budget, priority, preset, retrieval on / off, F0 range, denoiser, lookahead,
lever, device rate or block), compared with the values the tune itself wrote.

**live_record** (in-app only: `voice_changer_live_record_start { request: { max_seconds?, note? } }`
and `voice_changer_live_record_stop`) records the running voice changer's per-tick timeline for up
to 10 minutes during a real session (VRChat + OBS): the authoritative H1 evidence for T1 / T2 / T7.
A recorder thread drains the timeline every 100 ms (bounded, drop-oldest: it never slows the
worker) and samples the voice changer's counters, process CPU and VRAM. It stops at `max_seconds`,
when the voice changer stops for more than 2 s, or on stop. The folder holds `report.json` /
`report.md` (one run of mode `live_record`), `live.timeline.csv` and `live.samples.csv`, and no
audio or file paths, so it can be shared as it is. Metrics: `latency_ms_*` (capture of a block's
newest frame to the estimated playout of its first converted sample), `block_ms_*`, `model_ms_*`,
`backlog_ms_*`, `queued_out_ms_*`, `ticks`, `live_seconds`, `underrun_ticks`, `budget_drop_ticks`,
`gap_ticks`, `error_ticks`, `silent_ratio`, `governor_steps`, `governor_level_max`, `dropped_ms`,
`backlog_max_ms`, `over_budget_ratio`, `cpu_cores`, `vram_mb_peak`, `timeline_dropped`,
`underruns`.

### Fidelity scorer v2, rescore, load_twice and attribution (P3B)

- **`scoring.analyzer`** is `v1` (default, the P3 metrics), `v2` or `v2+ablations` ([ADR 0009](../adr/0009-vc-scorer-v2.md)).
  A v2 quality run adds the exact CAM++ (`bench/campplus-exact`, speech frames only), the fed-F0 errors (the
  engine's own F0 captured on the absolute frame grid against whole-clip RMVPE of the input: fine pitch error,
  gross pitch error, V/UV, onset lag), two conditional binomial click tests (wide windows placed for the model
  delay, narrow windows at the crossfade boundaries), the envelope MCD over its same-seed floor, the
  offset-corrected MCD and the onset levels, with `quality.fidelity.offsets_s` (default 3 offsets, medians and
  `_spread`), the ceilings A (`long_shape`, the primary), B and `whole` (`ceiling.mode`), the speaker target set
  (`fidelity.reference_clips`) and, with `v2+ablations`, the oracle ablations. The v1 metrics are printed next to
  them. `scripts/bench/specs/p3b/b/run_corpus.ps1` runs the whole local corpus (18 clips x 3 offsets), the
  leave-one-clip-out rescore and `scripts/bench/score_v2/aggregate.py` (corpus means with bootstrap CIs and the
  T5 v2 rows).
- **`rescore`** scores saved stream / ceiling WAV pairs (`pairs[]` of `{name, stream, ceiling, input}`) with the
  current scorer without converting anything, so an old executable's WAVs and a new one's are judged by one
  scorer. Reports `clips`, `pairs` and the quality metrics; the offsets of one v2 run count as one clip.
- **`load_twice`** loads a second voice while the first streams (the `hot_slots = 2` row) and reports
  `second_load_ms`, `vram_second_voice_mb`, `vram_second_load_peak_mb` and `background_call_ms_p99`.
- **Attribution, on every engine run:** per-call `content_ms`, `f0_ms`, `enc_ms`, `flowdec_ms`, `retrieval_ms`,
  `post_ms`, `glue_ms` (the call minus the union of its stages), `overlap_ms`, `harness_ms`, `syncs_per_call`, the
  onset calls apart (`call_ms_onset_*`), the SOLA joins (`join_offset_ms_*`, `join_step_ms_*`, `join_ncc_*`),
  `f0_calls` / `f0_long_calls` (extractor calls over the run, and those longer than the extractor's bound: 0
  with priming on a bounded plan), the paced T1 decomposition (`t1_<part>_ms_*`, `t1_at_p50_<part>_ms`,
  `t1_residual_p50_ms`: a telescoping partition of each sample's path whose median sample adds up to
  `latency_ms_p50`), `algorithmic_latency_table` (480 against 512 frames at 20/40/60/80 ms engine blocks), the
  load window split by stage (`load_<stage>_ms`, `load_other_ms`, `load_attributed_ratio`, `generator_builds`),
  VRAM (`vram_load_peak_mb`, `vram_load_overshoot_mb`, `vram_cuda_context_mb`, the per-component ledger and
  the measured spans) and the run's GPU and host state (`state_*`: NVML clocks, P-state, clock-event reasons,
  other processes using the GPU, a speed index, a fixed host micro-benchmark, power plan, process QoS).
  `SIGIL_BENCH_GPU_TIMING=1` adds CUDA-event GPU time per plan (`gpu_<component>_ms_*`).
- **New spec fields:** `engine.qos` (`app`: the voice changer's EcoQoS opt-out, or `none`), `engine.cpu_arena`,
  `engine.dynamic_block_base`, `engine.content_variant` (`auto`, `fp32`, `fp16`, `int8`), `stream.output_cushion`
  (`fixed`, `adaptive`), `stream.device_block` (`fixed`, `model_divisor`), `stream.sola_search_ms`, `vram.sample_ms`
  (the load-peak sampler), the contention profiles `h2_load70_oop` (the out-of-process DirectML load) and
  `h2_load70_dml`, and `load.gpu.out_of_process`.
- **Environment switches of the benchmark:** `SIGIL_BENCH_STATE` (`off`, `light`, default full),
  `SIGIL_BENCH_GPU_TIMING=1`, `SIGIL_BENCH_NVML_DEVICE`, `SIGIL_BENCH_ABLATIONS=1`, `SIGIL_ORT_PROFILE=<dir>` with
  `SIGIL_ORT_PROFILE_FILTER=<component prefix>` (tooling only; never time with it on) and
  `SIGIL_VC_F0_OVERLAP=0` for a clean per-component VRAM ledger. The engine's own kill switches are listed in
  [DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md#voice-engine-optimization-p3b).

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

- The `stream` defaults are the voice changer's defaults. `block_frames` is one of its chunk
  sizes: 160 … 48000. For example, 512 is 10.7 ms and 1536 is 32 ms at 48 kHz. A whole number of
  10 ms up to 200 ms also runs (e.g. 1920 = 40 ms at 48 kHz, where a template voice's device block
  equals its engine block), but the voice changer itself runs only its chunk sizes (any other
  setting becomes 512): such a run measures the engine's floor, and its report says so (an engine
  warning in `report.json` / `report.md`).
- `input.signal` is one of the following:
  - `voice`: speech-like, voiced, with pauses;
  - `tones`;
  - `chirp_train`;
  - `impulse_train`;
  - `noise`;
  - `silence`.

  `{"kind": "wav", "path": ...}` reads any PCM or float WAV and resamples it to `input_rate`.
- `rvc` engine fields:
  - `slot_dir` (required): a template voice (`sigil-rvc-1`) or a bring-your-own ONNX slot;
  - `hubert` (the content encoder), `fcpe`, `rmvpe` (empty = auto), `index` (empty = the slot's);
  - `pitch_extractor`: `rmvpe`, `fcpe`, `swiftf0`, `crepe`, `pesto`, `fusion` or `dio`;
  - `execution_provider`: `cpu`, `cuda`, `dml`, `tensorrt_rtx`, `dml_hp`, or empty for the
    preset's default;
  - `preset`: `streamer`, `low_latency`, `quality` or `studio_trt`;
  - `gpu_device_index`, `intra_threads` (the plan's "1-2 threads for real-time sessions"
    measurement), `vram_cap_mb`, `silence_skip` (default true, as in the voice changer);
  - template voices: `block_ms` (engine block, 0 = the device block), `precision` (`auto`, `fp32`,
    `fp16`), `levers` (`{"io_binding": "on", ...}`: each of `partial_decode`, `anchored_noise`,
    `nsf_phase0`, `f0_cache`, `gpu_knn`, `io_binding`, `cuda_graphs`, `gpu_priority`, `governor`
    is `auto`, `on` or `off`), `f0_range` (`speech`, `singing`), `lookahead_ms` (0–80, direct
    build), `noise_seed`;
  - `build`: `auto` (the engine factory; the direct builder when `lookahead_ms` is set),
    `factory` or `direct`;
  - `denoiser`: `off` (default), `gtcrn`, `dfn3` or `dpdfnet`, resolved like the voice changer's
    (a kind this build cannot run runs its stand-in, noted). `gtcrn` runs inside the engine on
    factory builds, as in the voice changer's RVC mode, and its latency counts; the others run on
    the voice changer's capture copy, outside the engine, so the run excludes them with a warning.
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
(NumPy's default). All times are in ms. Lower is better except for the metrics a report lists in
a run's `higher_is_better`: `throughput_x`, `e2e_detected`, the load duties, speaker similarity,
DNSMOS, the compare margins and `autotune_fidelity`.

| Mode | Distributions | Scalars |
|---|---|---|
| offline | `block_ms`, `call_ms`, `model_ms`, `rtf` | `blocks`, `over_budget_ratio`, `model_errors`, `algorithmic_latency_ms`, `engine_load_ms`, `harness_init_ms`, `cpu_cores`, `worker_cpu_cores`, `vram_engine_mb`, `throughput_x` |
| paced | the offline ones, plus `latency_ms`, `block_age_ms` | the offline scalars except `throughput_x`, plus `underruns`, `underrun_ms`, `dropped_ms`, `drop_events`, `overflow_ms`, `backlog_max_ms`, `backlog_over_budget_ms`, `stall_recovery_blocks`, `stall_latency_recovery_ms`, `output_trimmed_ms`, `gpu_load_duty`, `cpu_load_duty` |
| e2e_loopback | `e2e_ms` | `e2e_detected`, `e2e_missed`, `e2e_timebase_jitter_ms` |
| quality | the offline ones | see **quality** above |
| contention | the paced ones | the paced ones |
| compare | - | `sigil_dnsmos_ovrl` / `_sig` / `_bak`, `sigil_gross_pitch_error`, `sigil_f0_rmse_cents`, `sigil_vuv_error`, `sigil_speaker_cosine`, `sigil_latency_ms`, `sigil_artifact_events_per_min`, `best_competitor_dnsmos_ovrl`, `dnsmos_ovrl_margin`, `min_competitor_gross_pitch_error`, `gross_pitch_error_margin`, `best_competitor_latency_ms`, `latency_ratio`, `candidates`, `candidates_scored` |
| autotune | - | `autotune_block_ms`, `autotune_extra_context_ms`, `autotune_rtf_p99`, `autotune_compute_ms_p95`, `autotune_latency_ms_p50` / `_p99`, `autotune_fidelity`, `autotune_gpu`, `autotune_candidates`, `autotune_seconds` |
| live_record | `latency_ms`, `block_ms`, `model_ms`, `backlog_ms`, `queued_out_ms` | see **live_record** above |

Template voices built directly (`build: direct`) add `content_ms_*`, `f0_ms_*`, `enc_ms_*` and
`flowdec_ms_*` to offline and paced runs.

Notes on specific metrics:
- `block_ms` covers every device block; `call_ms` only the device blocks that ran the model, and
  `model_ms` is per model call. `rtf_*` (compute over the real-time budget: the longer of the
  device block and the engine block) and `over_budget_ratio` are judged on the device blocks that
  ran the model. All of these are the same for an engine that converts every device block. A
  template voice converts one engine block (for example 40 ms) every few device blocks (10.7 ms),
  so its per-block compute (T2) is `call_ms`: `block_ms_p95` would be about the 80th percentile of
  the calls. The timeline CSV has a `calls` column.
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
  `model_ms`, `calls`, `stall_ms`, `waiting_ms`, `dropped_ms`, `queued_out_ms` and `error`.
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

`-Profile p3` (`scripts/bench/profiles/p3.json`) runs the stub specs as they are and materializes
the real-voice templates in `scripts/bench/specs/p3/` with `-Slot` (an imported voice folder),
`-Clip` (a speech WAV) and `-AppData` (the store with the models), then checks the plan's
T-targets: `gate: true` rows fail the gate when missed or unmeasured (the P3 exit rows: T2, T3,
T5, T7, T9 on H2 with the load simulator, T4 streamer), `gate: false` rows are reported only
(provisional H1 / H3 rows, CUDA with the GPU levers on). Templates whose variables are missing are
skipped with a warning (`-Strict`: an error). `-Var NAME=value` overrides a template variable
(`-Var 'LEVERS_JSON={"cuda_graphs":"on"}'`, `-Var PITCH=swiftf0`).

```powershell
pwsh scripts/bench/run-perf-gate.ps1 -Profile p3 -Exe <release sigil.exe> -AppData <app data> `
    -Slot <app data>/voice-changer/models/default -Clip C:/clips/speech.wav -OutRoot bench-results/p3
```

`-Profile p5` (`scripts/bench/profiles/p5.json`) materializes the voice-family templates in
`scripts/bench/specs/p5/` (Sigil-ZS with `-Var SLOT_ZS=<slot> -Var REFERENCE=<clip> -Clip <input>`,
DDSP-SVC singing with `-Var SLOT_DDSP=<slot> -Var SWEEP=<wav from scripts/bench/singing_sweep.py>`)
and checks T10's delay rows and the families' fidelity; score the singing output against the
intended notes with `python scripts/bench/score_singing.py <out>/offline.output.wav 12`.

`scripts/bench/run-matrix.ps1` runs one voice through EP × F0 method × block × precision × context
× levers cells (one after the other, never in parallel) and tabulates compute, RTF, latency, VRAM,
CPU, load time and the per-component breakdown in `matrix.md` / `matrix.csv`.

### Interleaved A/B (P3B-A): the evidence a before/after claim needs

On this rig a single before/after pair is noise: the GPU clock governor holds the SM clock low for whole
stretches, and the pair SD of `call_ms_p95` was 11-19 % offline. A claim therefore comes from interleaved runs
of the two sides (A B, B A, A B, ...), whose paired log-ratios the gate summarises:

```powershell
# one executable, a switch (an environment variable on side B)
pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <exe> -CompareExe <exe> -CompareEnv 'SIGIL_VC_DML_SHARED=1' -Only sve-dml -Slot <voice> -Clip <wav>
# two executables, or two values of a profile variable
pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <new> -CompareExe <old> -CompareVars 'BUILD=direct' -Only load-cuda -Slot <voice> -Clip <wav>
# the gate's own resolution: the same exe on both sides, then feed -AASummary
pwsh scripts/bench/run-perf-gate.ps1 -Profile p3b -Exe <exe> -CompareExe <exe> -Only aa-dml,aa-cuda -Slot <voice> -Clip <wav>
```

- `-Rounds` (default 7) ABBA rounds, grown by two while a row is INCONCLUSIVE up to `-MaxRounds` (15);
  `-Only` selects profile rows, `-Resume` continues a session, `-AASummary <gate-summary.json>` with
  `-TargetHalfWidthPct` sizes the rounds from an A/A session, `-AllowInconclusive` keeps the exit code 0.
- Statistics are paired log-ratios with the exact Wilcoxon / Hodges-Lehmann interval from 6 pairs (Student t
  below that; no bootstrap). The Rust mirror is `bench::metrics::paired`, with golden tests; `-SelfTest`
  checks the gate itself in PowerShell 7 and 5.1.
- Verdicts: REGRESSED (median above +5 % and the interval's lower bound above 0), INCONCLUSIVE (median above
  +5 %, lower bound at or below 0), IMPROVED (upper bound below 0), NEUTRAL, and REFUSED for F0 metrics across
  different analyzer stamps (`analyzer_version`). Only a REGRESSED p95 blocks a change.
- Runs more than 10 % off the session median of their GPU or host state (`state_sm_clock_mhz`,
  `state_mem_clock_mhz`, `state_speed_index_ms`, `state_host_probe_ms`, `state_cpu_effective_mhz`) are flagged in
  `report.json` (`gate_state_flags`). Each paired row also carries `matched_*` statistics without the pairs whose
  two sides ran in different GPU clock regimes (`-RegimeTolerancePct`, default 25): information only, the
  verdict always uses every pair.
- Measured resolution (two A/A sessions): 62-64 of 64 rows NEUTRAL, no false REGRESSED; `call_ms_p95` interval
  half-width 1.3-4.3 % offline and under the 70 % hammer once regime-mismatched pairs are set aside, 8-13 % on
  the paced rows (a real-time cadence lets the GPU drop clocks between ticks), so size paced rows with
  `-AASummary` (about 15 rounds for 6 %).
- The `p3b` profile (`scripts/bench/profiles/p3b.json`) holds the scorecard rows and their variables; each
  workstream keeps its own rows and spec templates under `scripts/bench/specs/p3b/<a..h>/` (`*.profile.json`,
  run with `-Profile <path>`; their READMEs list the commands and numbers).

## Inputs: synthetic clips and the corpus

- `python scripts/bench/make-test-clips.py --out <dir>` writes the synthetic signals as WAV files.
  It uses the standard library only. `--mix voice music_bleed:-20` writes a voice clip with world
  music bleed at -20 dB, and `--float` writes 32-bit float.
- The real corpus is about 20 consented recordings from you and friends, the same clips with game
  or music bleed, and a singing set. It lives in `app_data/bench/corpus/`
  (`%APPDATA%\com.sigil.app\bench\corpus`) and is never committed.
  `scripts/bench/corpus.example.json` is the manifest template. IPC specs reference clips as
  `{"kind": "wav", "path": "corpus/clips/speech-01.wav"}`.

## Competitors and listening tests

`scripts/vc-bench/` holds offline drivers for RVC WebUI realtime and Applio realtime (they run in
the user's own Python environment for those apps; nothing installs a competitor; both take `--corpus` and
`--sweep 40,60,100,250` and write driver sidecars) and a play-and-record helper for live apps such as w-okada
VCClient, whose output a `compare` run scores (see **compare v2** above and `scripts/vc-bench/README.md`). The
`e2e_loopback` mode also has a play-and-record run (`PlayRecordSpec`) that plays a clip and records the response on
one time base, reporting the response delay (envelope, about 1 ms), driver glitches and dropouts inside speech;
corpus captures use it, Sigil's live voice changer included.
`tools/listening-test/` builds a blind A/B page from the same WAVs (`make_test.py`) and scores the
answers with Wilson intervals (`score.py`): plan T6's preference test.

## Specs in the repo

| Spec | What it checks |
|---|---|
| `scripts/bench/specs/stub-smoke.json` | Plumbing: stub offline, a wall-clock paced run under a light CPU and ONNX Runtime load, and a 3 s stall on the virtual clock. Its thresholds only catch broken plumbing, and they hold on a busy PC and in debug builds. |
| `scripts/bench/specs/p3/stub-p3.json` | P3 plumbing without models or devices: a fixed-block stub (40 ms, like the voice engine) against its own offline ceiling (an identity model is at its ceiling: the MCD, F0 and click gates must hold), and the stall profile on the simulated clock (recovery within two blocks, backlog within budget). |
| `scripts/bench/specs/p3/sve-gpu.template.json` | Profile template: a template voice on a GPU EP (40 ms engine blocks on the 512-frame device block, 64 ms context, Low latency): offline, paced (wall clock), `h2_load70`, `h1_vrchat_obs` and fidelity. |
| `scripts/bench/specs/p3/sve-cpu.template.json` | Profile template: the same voice on the CPU (Streamer, 80 ms blocks): offline, paced, `h3_cpu_game` and fidelity. |
| `scripts/bench/specs/p3/matrix.template.json` | One `run-matrix.ps1` cell: offline plus a virtual-clock paced run. |
| `scripts/bench/specs/p3b/<a..h>/` | P3B workstreams: the scorecard rows (`a`: GPU, CPU, A/A, loads, VRAM, kNN-30k), scorer v2 and the corpus (`b`), compare v2 (`c`), content stages and fidelity (`d`), RMVPE plan size and onsets (`e`), F0 geometry (`f`), execution layer (`g`) and load and pacing (`h`). Each README lists the commands and the measured numbers. |
| `scripts/bench/specs/p5/zs-gpu.template.json` | Profile template: a Sigil-ZS voice with a reference clip on a GPU EP: offline, paced (wall clock), `h2_load70`, `h1_vrchat_obs` and fidelity against its one-off ceiling. |
| `scripts/bench/specs/p5/ddsp-singing.template.json` | Profile template: a DDSP-SVC voice on the sung sweep at `SEMITONES` (default +12, so the output reaches D#7), CREPE in the singing range: offline, paced (virtual clock), `h1_vrchat_obs` and fidelity (CREPE analyzer). |

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

**Fixed in P3 (P3-E):** the output queue now trims back toward the prefill with a crossfade after
an underrun or a budget drop. The same stub-smoke stall run (debug build before, release after;
the stall run is on the simulated clock, so the timeline is build-independent) now settles at
67.0 ms (74.4 ms before the stall) and recovers in 95 ms; before the fix it stayed at 94.4 ms for
the rest of the run (`stall_latency_recovery_ms` 2000 = never).
