//! `bench` plugin: the benchmark harness core. Spec parsing, offline and paced drivers over the real
//! `VcStreamHarness` and engine factory, the e2e-loopback latency meter, the GPU / CPU load
//! simulator and stall injection, CPU / VRAM metrics, and JSON + Markdown reports. The same code
//! runs from the CLI, from IPC ("Benchmark this PC", the autotuner) and from cargo tests.
//!
//! Owned by P1-F; scaffolded by P1-S0b (plugin registered in main.rs, capability
//! `capabilities/bench.json`, commands in `plugin-commands/bench.txt`, which P1-F owns). The VC v2
//! benchmark (P3-F) builds on these modules: real voices through the engine factory (template
//! voices on the Sigil Voice Engine), quality against the offline ceiling, competitor compare,
//! named contention profiles, live recording of the running voice changer and the per-PC
//! autotuner.
//!
//! # Layout
//! - [`spec`]: the `sigil-bench-spec/1` document (engine, stream, input, runs, thresholds).
//! - [`engine`]: engine factory (stub, RVC) and the identity probe for the harness's algorithmic
//!   latency.
//! - [`offline`]: throughput over the whole input. [`paced`]: the worker loop on a simulated
//!   device clock with the production budget and output path ([`vc_rt`]), load and stalls.
//! - [`e2e_loopback`]: real-device latency meter and the cross-correlation core.
//! - [`load_sim`]: GPU MatMul load (through `ort_session`) and CPU burners at a duty cycle.
//! - [`metrics`]: percentiles, CPU and VRAM meters. [`report`]: metric names, thresholds, JSON /
//!   Markdown / CSV. [`signals`]: synthetic clips, WAV I/O, resampling. [`cli`]: the headless body.
//! - P3-F: `rvc_build` (real voices with the voice changer's rules and factory), [`quality`]
//!   (metric functions) with [`dsp`] and [`scorers`] (CAM++, DNSMOS, STT), [`ceiling`] (the offline
//!   ceiling) and `fidelity` (the `quality` mode), [`compare`], [`contention`], [`live_record`],
//!   [`autotune`].
//! - P3B-A: [`attribution`] (per-call stage / glue / harness / sync / GPU-time metrics, onset calls,
//!   SOLA join histograms, the paced T1 decomposition, the load breakdown, the algorithmic latency
//!   table and the `load_twice` run), [`gpu_state`] (NVML clocks / P-state / reasons, the GPU speed
//!   index, the host probe, power plan and QoS, the VRAM load-peak sampler and the CUDA context,
//!   memory and arena probes), [`gpu_load_child`] (the out-of-process GPU load,
//!   `--headless gpu-load`), [`metrics::paired`] (the perf gate's paired statistics). Every spec
//!   records `neural_audio::telemetry` while it runs; the engine load's breakdown, VRAM ledger and
//!   CUDA figures go to `report.engine.attribution` and into every engine run's metrics
//!   (`load_<stage>_ms`, `vram_load_peak_mb`, `vram_cuda_context_mb`, `generator_builds`), each
//!   run's GPU / host state to `details.state` (`state_*` metrics). `engine.qos = app` (default)
//!   applies the app's process EcoQoS opt-out before the load, as the voice changer runs.
//!
//! # Frozen contract (P1-S0b; change only through INTEGRATE)
//! - [`headless_bench`]`(spec, out) -> Result<String, String>`: body of
//!   `Sigil.exe --headless bench <spec.json> [--out <dir>]`; writes `report.json` and `report.md`
//!   into `out` (default: `bench-results/<spec file stem>-<unix ms>/` under the working directory) and
//!   returns the summary printed to stdout. `Err` (bad spec, failed run, or a failed threshold in
//!   the spec) exits with code 1.
//! - IPC `plugin:bench|run_benchmark({ spec, outDir? }) -> report` (spec and report are JSON
//!   values whose schema P1-F owns in spec.rs / report.rs), `plugin:bench|cancel_benchmark`,
//!   `plugin:bench|get_last_report -> report | null`.
//! - Events [`PROGRESS_EVENT`] `"bench:progress"` ([`Progress`]) and [`DONE_EVENT`] `"bench:done"`
//!   (`{ ok, report | null, error | null }`).
//!
//! IPC specs resolve relative paths against `app_data/bench/`, and their `outDir` must stay inside
//! `app_data/bench/` (default `app_data/bench/results/<name>-<unix ms>/`). Every accepted
//! `run_benchmark` call ends in exactly one `bench:done`, errors and panics included; a call
//! refused because another benchmark is running emits none. RVC benchmark sessions record their
//! EP outcomes as `bench.vc`, `bench.vc.content` and `bench.vc.pitch`, apart from the voice
//! changer's rows, so an in-app RVC run may share the PC with a running voice changer.

/// Oracle ablations of the fidelity scorer v2 (P3B-B; stub by P3B-S0).
pub mod ablation;
/// Per-stage attribution and the `load_twice` run (P3B-A; stub by P3B-S0).
pub mod attribution;
/// Per-PC autotuner (P3-F; scaffolded by P3-S0).
pub mod autotune;
/// The offline ceiling (P3-F).
pub mod ceiling;
mod cli;
/// Competitor WAV scoring (P3-F; scaffolded by P3-S0).
pub mod compare;
/// GPU / CPU contention mode (P3-F; scaffolded by P3-S0).
pub mod contention;
/// Offline DSP of the quality metrics (P3-F).
pub mod dsp;
pub mod e2e_loopback;
pub mod engine;
/// The `quality` run mode (P3-F).
mod fidelity;
/// The out-of-process GPU load, `--headless gpu-load` (P3B-A; stub by P3B-S0).
pub mod gpu_load_child;
/// GPU clocks / P-state and host state of a run (P3B-A; stub by P3B-S0).
pub mod gpu_state;
#[cfg(test)]
mod hot_path_tests;
/// In-app live recording of the running voice changer (P3-F; scaffolded by P3-S0).
pub mod live_record;
pub mod load_sim;
pub mod metrics;
/// Multi-clip scoring and the `rescore` run (P3B-B; stub by P3B-S0).
pub mod multi_clip;
mod offline;
mod paced;
/// Quality metrics against the offline ceiling (P3-F; scaffolded by P3-S0).
pub mod quality;
pub mod report;
/// Real voices with the voice changer's rules and engine factory (P3-F).
pub(crate) mod rvc_build;
/// CAM++, DNSMOS P.835 and STT scorers (P3-F).
pub mod scorers;
pub mod signals;
pub mod spec;
/// The P7 speech benchmark runner, `--headless speech-bench` (P7-A1).
pub mod speech;
mod vc_rt;

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use crate::services::neural_audio::telemetry;

use serde::Serialize;
use sha2::{Digest, Sha256};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Runtime,
};

pub use cli::headless_bench;

use engine::{build_engine, probe_algorithmic_latency, AlgorithmicLatency, Engine};
use metrics::process_vram_mb;
use report::{evaluate, run_failures, BenchReport, Environment, RunOutcome, RunReport, RunStatus, REPORT_SCHEMA};
use spec::{BenchSpec, RunSpec};

/// Event: benchmark progress.
pub const PROGRESS_EVENT: &str = "bench:progress";

/// Stack of every benchmark thread (the headless CLI body, IPC runs, the autotuner, the offline and
/// paced workers). Engine builds, CUDA graph capture and the autotuner's nested measurements need
/// more than the 1 MB of Windows' main thread (a stack overflow there ended a GPU autotune run);
/// the reservation is virtual and only the pages used are committed.
pub(crate) const BENCH_STACK_BYTES: usize = 16 * 1024 * 1024;
/// Event: a benchmark finished (payload: the report, or an error).
pub const DONE_EVENT: &str = "bench:done";

/// Progress of a running benchmark.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    pub run: String,
    /// 0-based index of the run, and the number of runs.
    pub run_index: usize,
    pub run_count: usize,
    /// Fraction of the current run (0..1).
    pub fraction: f64,
    pub message: String,
}

/// Cancellation and progress for [`run_spec`].
pub struct BenchContext<'a> {
    pub cancel: &'a AtomicBool,
    pub progress: &'a (dyn Fn(&Progress) + Sync),
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Where and on what the benchmark runs.
pub(crate) fn environment() -> Environment {
    let sys = crate::services::sysinfo::cached();
    let cpu = Some(sys.cpu.brand.trim().to_string())
        .filter(|b| !b.is_empty())
        .or_else(|| std::env::var("PROCESSOR_IDENTIFIER").ok());
    let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0);
    let gpus = crate::services::diagnostics::gpu_mem::adapters()
        .into_iter()
        .filter(|a| !a.software)
        .filter_map(|a| serde_json::to_value(a).ok())
        .collect();
    Environment {
        sigil_version: env!("CARGO_PKG_VERSION").to_string(),
        build: if cfg!(debug_assertions) { "debug" } else { "release" }.to_string(),
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu,
        logical_cores: logical,
        onnx_runtime: crate::services::neural_audio::ort_session::runtime_version().to_string(),
        gpus,
        voice_changer_running: crate::services::resources::vc_running(),
    }
}

/// The engine, the harness latency probe and the input clip shared by a spec's engine runs.
pub(crate) struct Prepared {
    pub engine: Engine,
    pub probe: AlgorithmicLatency,
    pub input: Vec<f32>,
    pub vram_engine_mb: Option<f64>,
}

/// What `prepare` measured besides the [`Prepared`] engine (P3B-A): the engine-level metrics every
/// engine run carries, the report's `engine.attribution`, and the CUDA context the benchmark
/// retained (released when the spec ends).
#[derive(Default)]
struct PrepareExtras {
    metrics: metrics::Metrics,
    attribution: serde_json::Map<String, serde_json::Value>,
    cuda: Option<gpu_state::cuda::ContextProbe>,
}

/// The run-state probes of a spec (`SIGIL_BENCH_STATE`: `off`, `light` = no GPU speed index,
/// anything else = full).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StateMode {
    Full,
    Light,
    Off,
}

impl StateMode {
    fn from_env() -> StateMode {
        match std::env::var("SIGIL_BENCH_STATE").map(|v| v.trim().to_ascii_lowercase()).as_deref() {
            Ok("off" | "0") => StateMode::Off,
            Ok("light") => StateMode::Light,
            _ => StateMode::Full,
        }
    }
}

/// Whether an engine's active rung runs on the CUDA device context: CUDA itself and the TensorRT
/// accelerators of the CUDA rung (`EngineInfo::gpu`'s rule). A CUDA plan that fell back to
/// DirectML or the CPU is not.
fn cuda_rung_active(active: Option<&str>) -> bool {
    matches!(active, Some("cuda" | "tensorrt_rtx" | "tensorrt"))
}

/// The engine's process VRAM delta in MiB (`after - before`) without the CUDA context the
/// benchmark retained for an engine that did not end up on the CUDA rung (`unused_context_mb`):
/// that context is the benchmark's, not the engine's.
fn engine_vram_mb(before: Option<u64>, after: Option<u64>, unused_context_mb: Option<f64>) -> Option<f64> {
    match (before, after) {
        (Some(a), Some(b)) => Some((b.saturating_sub(a) as f64 - unused_context_mb.unwrap_or(0.0)).max(0.0)),
        _ => None,
    }
}

/// The CUDA ordinal of a spec's engine when its sessions are planned on the CUDA rung (a plan, not
/// an outcome: the CUDA pack may be missing, so `prepare` checks the loaded engine's active rung
/// and drops the context it retained when the engine did not end up there).
fn cuda_ordinal(spec: &BenchSpec) -> Option<i32> {
    let spec::EngineSpec::Rvc(r) = &spec.engine else {
        return None;
    };
    let priority = crate::services::rt_thread::WorkerPriority::from_str_lossy(&spec.stream.worker_priority);
    let plan = rvc_build::plan(r, &spec.stream, priority).ok()?;
    (plan.ep == "cuda").then_some(r.gpu_device_index)
}

/// The process QoS of the run (`engine.qos`): `app` applies the app's EcoQoS opt-out (the
/// headless bench never ran the diagnostics plugin that does it), `none` leaves the process as
/// Windows manages it. Returns what was asked.
fn apply_qos(spec: &BenchSpec) -> Option<&'static str> {
    let spec::EngineSpec::Rvc(r) = &spec.engine else {
        return None;
    };
    if r.qos.trim().eq_ignore_ascii_case("none") {
        Some("none")
    } else {
        let _ = crate::services::diagnostics::qos::opt_out_power_throttling();
        Some("app")
    }
}

fn prepare(spec: &BenchSpec) -> Result<(Prepared, PrepareExtras), String> {
    let s = &spec.stream;
    let input = signals::render_input(&spec.input, s.input_rate, usize::from(s.input_channels), spec.seed)?;
    if input.len() / usize::from(s.input_channels) < s.block_frames as usize {
        return Err("the input clip is shorter than one block".into());
    }
    let mut extras = PrepareExtras::default();
    let qos = apply_qos(spec);
    let vram0 = process_vram_mb();
    // P3B-A: the load-peak sampler, the load spans and the VRAM marks around the load; on CUDA the
    // device's primary context is retained first (its VRAM measured once per process). The app
    // creates that context inside the voice load, so the retain is timed as a load stage
    // (`cuda_context`) and added to `engine_load_ms`: the load stays the app's cold load.
    let sampler = (spec.vram.sample_ms > 0.0).then(|| gpu_state::VramSampler::start(Duration::from_secs_f64(spec.vram.sample_ms / 1000.0)));
    let load_mark = telemetry::load_mark();
    let vram_marks = telemetry::vram_mark_count();
    let ledger0 = crate::services::neural_audio::gpu::arena::ledger().snapshot();
    let w0 = telemetry::now_ns();
    let t_ctx = Instant::now();
    let mut cuda = cuda_ordinal(spec).and_then(|d| {
        let _span = telemetry::load_span(telemetry::LoadStage::CudaContext);
        gpu_state::cuda::ContextProbe::retain(d).ok()
    });
    let context_ms = t_ctx.elapsed().as_secs_f64() * 1000.0;
    let priority = crate::services::rt_thread::WorkerPriority::from_str_lossy(&s.worker_priority);
    // The load window starts before the context retain; it moves behind it when that context turns
    // out not to belong to the engine (below).
    let w_after_context = telemetry::now_ns();
    let mut engine = build_engine(&spec.engine, s, priority, spec.seed)?;
    let w1 = telemetry::now_ns();
    let peak = sampler.map(gpu_state::VramSampler::stop);
    if engine.rvc.as_ref().is_some_and(|p| p.format.family().is_some()) {
        configure_family_stream(&mut engine, s);
    }
    let vram1 = process_vram_mb();
    // A CUDA plan whose engine ended on DirectML or the CPU (no `runtime/nvidia-cuda13` pack, an
    // old driver) never created a CUDA context of its own: the one retained above is the
    // benchmark's. Release it and leave it out of the load time, the load breakdown and the VRAM
    // figures, so a DirectML engine is not charged a CUDA context in the report or the T3/T9 rows.
    let mut w0 = w0;
    let mut unused_context_mb = None;
    if cuda.is_some() && !cuda_rung_active(engine.info.execution_provider_active.as_deref()) {
        let unused = cuda.take();
        unused_context_mb = unused.as_ref().and_then(|c| c.context_mb);
        w0 = w_after_context;
        engine.info.warnings.push(format!(
            "the plan asked for the CUDA rung but the engine runs on {}: the benchmark's own CUDA context ({} MB, {:.0} ms) is left out of the load time and the VRAM figures",
            engine.info.execution_provider_active.as_deref().unwrap_or("the CPU"),
            unused_context_mb.map_or_else(|| "unmeasured".to_string(), |mb| format!("{mb:.0}")),
            context_ms
        ));
        drop(unused);
    }
    if cuda.is_some() {
        engine.info.load_ms += context_ms;
        extras.metrics.insert("cuda_context_ms".into(), context_ms);
    }
    if let Some(note) = spec::device_block_note(s) {
        engine.info.warnings.push(note);
    }
    let probe = probe_algorithmic_latency(
        s,
        engine.info.output_rate,
        engine.info.min_context_samples,
        engine.fixed_block(),
        engine.info.model_delay_ms,
    )?;
    let vram_engine_mb = engine_vram_mb(vram0, vram1, unused_context_mb);

    // ---- load breakdown ----
    let load = attribution::LoadBreakdown::of(&telemetry::load_events_since(load_mark), w0, w1);
    load.insert_metrics("load", &mut extras.metrics);
    extras.metrics.insert("generator_builds".into(), f64::from(load.generator_builds));
    extras.attribution.insert("load".into(), serde_json::to_value(&load).unwrap_or_default());
    // The plan's `load_breakdown_ms`: wall-clock ms by stage plus `other`, summing to the load
    // window (the engine load, `engine_load_ms`, up to the timer's placement).
    let mut breakdown: serde_json::Map<String, serde_json::Value> = load.stages_ms.iter().map(|(k, v)| ((*k).to_string(), (*v).into())).collect();
    breakdown.insert("other".into(), load.other_ms.into());
    extras.attribution.insert("load_breakdown_ms".into(), breakdown.into());

    // ---- VRAM ----
    let mut vram = serde_json::Map::new();
    if let Some(mb) = vram_engine_mb {
        vram.insert("engine_mb".into(), mb.into());
    }
    if let (Some(a), Some((Some(pk), n))) = (vram0, peak) {
        let peak_mb = (pk.saturating_sub(a) as f64 - unused_context_mb.unwrap_or(0.0)).max(0.0);
        extras.metrics.insert("vram_load_peak_mb".into(), peak_mb);
        vram.insert("load_peak_mb".into(), peak_mb.into());
        // The load transient above the steady state (T3: a cold load peaks <= steady + 150 MB).
        if let Some(steady) = vram_engine_mb {
            extras.metrics.insert("vram_load_overshoot_mb".into(), (peak_mb - steady).max(0.0));
            vram.insert("load_overshoot_mb".into(), (peak_mb - steady).max(0.0).into());
        }
        vram.insert("load_peak_samples".into(), n.into());
        vram.insert("load_peak_sample_ms".into(), spec.vram.sample_ms.into());
    }
    // The ledger's change over the load (session plans' measured holds) and the spans' marks
    // (the non-plan sessions, with the VRAM probe on).
    let ledger1 = crate::services::neural_audio::gpu::arena::ledger().snapshot();
    let mut ledger = serde_json::Map::new();
    for (k, v) in &ledger1.components {
        let d = v - ledger0.components.get(k).copied().unwrap_or(0.0);
        if d.abs() > 0.0 {
            ledger.insert(k.clone(), d.into());
        }
    }
    vram.insert("ledger".into(), ledger.into());
    // `measured`: what each component's load spans allocated themselves (DXGI process deltas with
    // the VRAM probe on; nested spans count once; `concurrent` when another thread's span
    // overlapped, whose allocations the reading may include). `marks`: the components' own marks
    // (`telemetry::vram_mark`, e.g. a content plan's hold) and arena statistics
    // (`telemetry::arena_mark`, as `<component>/<source>`), kept apart from the spans.
    let mut measured = serde_json::Map::new();
    let by_component = load.vram_by_component();
    for (component, (mb, concurrent)) in &by_component {
        measured.insert((*component).to_string(), serde_json::json!({"mb": mb, "concurrent": concurrent}));
    }
    vram.insert("measured".into(), measured.into());
    // The closures. Steady state: the session plans' holds (`ledger`, held while the plans live)
    // plus the CUDA context against the steady process delta: `ledger_unattributed_mb` is what no
    // plan holds (eager sessions that stay, activations outside a plan). Load time: the spans
    // (`measured`) are what each component allocated while it loaded, transient copies included
    // (an eager session that a plan replaces and releases), so their sum exceeds the steady delta
    // by about the load overshoot. Both read process-wide DXGI deltas: sessions built at the same
    // time on two threads count each other's allocations (`measured.*.concurrent`, and a plan's
    // hold measured meanwhile); for a clean per-component split load with the F0 helper off
    // (`SIGIL_VC_F0_OVERLAP=0`: the same sessions, built one after another).
    let ledger_sum: f64 = ledger1
        .components
        .iter()
        .map(|(k, v)| v - ledger0.components.get(k).copied().unwrap_or(0.0))
        .filter(|d| *d > 0.0)
        .sum();
    if let Some(engine_mb) = vram_engine_mb {
        if ledger_sum > 0.0 {
            let context = cuda.as_ref().and_then(|c| c.context_mb).unwrap_or(0.0);
            vram.insert("ledger_sum_mb".into(), ledger_sum.into());
            vram.insert("ledger_unattributed_mb".into(), (engine_mb - context - ledger_sum).into());
        }
    }
    if !by_component.is_empty() {
        let sum: f64 = by_component.values().map(|(mb, _)| *mb).sum();
        let any_concurrent = by_component.values().any(|(_, c)| *c);
        vram.insert("measured_sum_mb".into(), sum.into());
        vram.insert("measured_concurrent".into(), any_concurrent.into());
        if let Some(engine_mb) = vram_engine_mb {
            // Load-time allocations above the steady delta (transients, or concurrent spans).
            vram.insert("measured_over_steady_mb".into(), (sum - engine_mb).into());
        }
    }
    let mut marks: BTreeMap<String, f64> = BTreeMap::new();
    for m in telemetry::vram_marks_since(vram_marks).into_iter().filter(|m| m.source != "span") {
        let key = match m.source {
            "explicit" => m.component.to_string(),
            other => format!("{}/{other}", m.component),
        };
        *marks.entry(key).or_insert(0.0) += m.mb;
    }
    vram.insert("marks".into(), serde_json::to_value(&marks).unwrap_or_default());
    if let Some(c) = &cuda {
        if let Some(mb) = c.context_mb {
            extras.metrics.insert("vram_cuda_context_mb".into(), mb);
            vram.insert("cuda_context_mb".into(), mb.into());
        }
        if let (Some(before), Some(after)) = (c.device_used_before_mb, c.memory()) {
            extras.metrics.insert("vram_cuda_device_delta_mb".into(), after.used_mb - before);
            vram.insert("cuda_device_delta_mb".into(), (after.used_mb - before).into());
            vram.insert("cuda_device_used_mb".into(), after.used_mb.into());
            vram.insert("cuda_device_total_mb".into(), after.total_mb.into());
        }
        if let Some(b) = c.stack_limit_bytes() {
            extras.metrics.insert("cuda_stack_limit_kb".into(), b as f64 / 1024.0);
            vram.insert("cuda_stack_limit_kb".into(), (b as f64 / 1024.0).into());
        }
        if let Some(stats) = gpu_state::cuda::env_arena_stats(c.ordinal()) {
            vram.insert("cuda_env_arena".into(), serde_json::to_value(stats).unwrap_or_default());
        }
    }
    extras.attribution.insert("vram".into(), vram.into());
    extras.cuda = cuda;

    // ---- host, QoS and the algorithmic latency table ----
    let mut host = serde_json::to_value(gpu_state::host_snapshot()).unwrap_or_default();
    gpu_state::insert_json(&mut host, "qos_requested", qos);
    extras.attribution.insert("host".into(), host);
    if engine.fixed_block().is_some() {
        let table = attribution::algorithmic_latency_table(s, engine.info.output_rate, engine.info.min_context_samples, &[20.0, 40.0, 60.0, 80.0]);
        extras
            .attribution
            .insert("algorithmic_latency_table".into(), serde_json::to_value(table).unwrap_or_default());
    }
    Ok((
        Prepared {
            engine,
            probe,
            input,
            vram_engine_mb,
        },
        extras,
    ))
}

/// Family voices build their sessions when a harness configures their stream (P5-I0): build one
/// harness at the stream's shape (warm-up included) before the VRAM is read, as the voice changer
/// measures a voice after its harness, and report what the model runs after it (a family can
/// re-plan its sessions there). A failure is left to the runs, which build the same harness.
fn configure_family_stream(engine: &mut Engine, s: &spec::StreamSpec) {
    let times = std::sync::Arc::new(std::sync::Mutex::new(engine::ModelTimes::default()));
    let built = crate::services::neural_audio::vc::VcStreamHarness::new(
        engine.handle(times),
        s.input_rate,
        s.output_rate,
        usize::from(s.input_channels),
        usize::from(s.output_channels),
        s.extra_samples_16k(),
        s.crossfade_samples(engine.info.output_rate),
        s.block_frames as usize,
    );
    if built.is_err() {
        return;
    }
    drop(built);
    if let Some(report) = engine.with_model(|m| m.stream_report()) {
        // As the voice changer folds a model's stream report into its voice (`factory.rs`).
        engine.info.ep_outcomes.extend(report.outcomes);
        let levers = engine.info.levers.get_or_insert_with(Default::default);
        levers.levers = report.levers.levers;
        levers.accelerator = report.levers.accelerator;
        for note in report.levers.notes {
            if !levers.notes.contains(&note) {
                levers.notes.push(note);
            }
        }
        for warning in report.warnings {
            if !engine.info.warnings.contains(&warning) {
                engine.info.warnings.push(warning);
            }
        }
        if let Some(rung) = engine.rvc.as_ref().map(|p| p.ep) {
            rvc_build::refresh_active_ep(&mut engine.info, rung);
        }
    }
}

/// Write a run's extra files; returns their names.
fn write_run_files(dir: &Path, name: &str, outcome: &RunOutcome, spec: &BenchSpec) -> Result<Vec<String>, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let mut files = Vec::new();
    if spec.output.write_timeline && !outcome.timeline.is_empty() {
        let file = format!("{name}.timeline.csv");
        report::write_atomic(&dir.join(&file), report::timeline_csv(&outcome.timeline).as_bytes())?;
        files.push(file);
    }
    if let Some((audio, rate, ch)) = &outcome.audio {
        let file = format!("{name}.output.wav");
        signals::write_wav(&dir.join(&file), audio, *rate, *ch)?;
        files.push(file);
    }
    for (suffix, audio, rate, ch) in &outcome.extra_audio {
        let file = format!("{name}.{suffix}.wav");
        signals::write_wav(&dir.join(&file), audio, *rate, *ch)?;
        files.push(file);
    }
    Ok(files)
}

/// The process-wide state a spec switches on, restored on every exit of [`run_spec`] (a panic
/// that unwinds it included): the telemetry's GPU-timing and VRAM-probe flags, and the GPU speed
/// index's session (a DirectML or CUDA session with its own device and queue), which must not
/// outlive the benchmark in the running app.
struct SpecProcessState {
    gpu_timing: bool,
    vram_probe: bool,
}

impl SpecProcessState {
    fn enter(gpu_timing: bool, vram_probe: bool) -> SpecProcessState {
        if gpu_timing {
            telemetry::set_gpu_timing(true);
        }
        if vram_probe {
            telemetry::set_vram_probe(true);
        }
        SpecProcessState { gpu_timing, vram_probe }
    }
}

impl Drop for SpecProcessState {
    fn drop(&mut self) {
        if self.vram_probe {
            telemetry::set_vram_probe(false);
        }
        if self.gpu_timing {
            telemetry::set_gpu_timing(false);
        }
        gpu_state::release_speed_index();
    }
}

/// Run every run of `spec`, write the reports into `out_dir` (when given) and return the report.
/// Never fails as a whole: an engine that does not load, a device that does not open or a
/// cancellation become per-run statuses, so a report always describes what happened.
pub fn run_spec(spec: &BenchSpec, out_dir: Option<&Path>, ctx: &BenchContext<'_>) -> BenchReport {
    let started = Instant::now();
    let started_unix_ms = unix_ms_now();
    metrics::prepare_cpu_meter();
    let spec_value = serde_json::to_value(spec).unwrap_or_default();
    let spec_sha256 = hex::encode(Sha256::digest(spec_value.to_string().as_bytes()));
    let run_count = spec.runs.len();
    // P3B-A: telemetry (stage and load spans, syncs, VRAM marks) for the whole spec; GPU time per
    // plan run on request (`SIGIL_BENCH_GPU_TIMING=1`: CUDA events, P3B-G); the spans' VRAM
    // samples only for VRAM runs (`vram.sample_ms`), since they cost load time.
    let _recording = telemetry::start_recording();
    let gpu_timing = std::env::var("SIGIL_BENCH_GPU_TIMING").is_ok_and(|v| v.trim() == "1");
    let vram_probe = spec.vram.sample_ms > 0.0;
    let process_state = SpecProcessState::enter(gpu_timing, vram_probe);

    let prepared_all: Option<Result<(Prepared, PrepareExtras), String>> = spec.runs.iter().any(RunSpec::uses_engine).then(|| {
        (ctx.progress)(&Progress {
            run: spec.runs[0].name().to_string(),
            run_index: 0,
            run_count,
            fraction: 0.0,
            message: "loading the engine".into(),
        });
        prepare(spec)
    });
    let (prepared, extras): (Option<Result<Prepared, String>>, Option<PrepareExtras>) = match prepared_all {
        Some(Ok((p, x))) => (Some(Ok(p)), Some(x)),
        Some(Err(e)) => (Some(Err(e)), None),
        None => (None, None),
    };
    let state_ctx = gpu_state::StateContext {
        engine_ep: match &prepared {
            Some(Ok(p)) => p.engine.info.execution_provider_active.clone(),
            _ => None,
        },
        device: match &spec.engine {
            spec::EngineSpec::Rvc(r) => r.gpu_device_index,
            spec::EngineSpec::Stub(_) => 0,
        },
        period_ms: 1_000,
        speed_index: true,
    };
    let state_mode = StateMode::from_env();

    let mut runs: Vec<RunReport> = Vec::with_capacity(run_count);
    for (index, run) in spec.runs.iter().enumerate() {
        let run_started = Instant::now();
        let name = run.name().to_string();
        // P3B-A: the run's GPU / host state (start, 1 Hz, end) and load spans (re-plans, the
        // ceiling's generator builds). `SIGIL_BENCH_STATE=off` skips the state probes (an A/B
        // against an exe without them), `light` keeps them without the GPU speed index.
        let state_probe = (!ctx.cancel.load(Ordering::Relaxed) && state_mode != StateMode::Off).then(|| {
            let mut c = state_ctx.clone();
            if state_mode == StateMode::Light {
                c.speed_index = false;
            }
            gpu_state::RunStateProbe::start(c)
        });
        let run_load_mark = telemetry::load_mark();
        // P3B-F: the F0 caches' extractor calls over this run (process-wide counters).
        let f0_calls_mark = crate::services::neural_audio::vc::engine::f0_cache::process_call_counts();
        let run_w0 = telemetry::now_ns();
        let progress = |fraction: f64, message: &str| {
            (ctx.progress)(&Progress {
                run: name.clone(),
                run_index: index,
                run_count,
                fraction,
                message: message.to_string(),
            });
        };
        progress(0.0, "starting");
        let result: Result<RunOutcome, String> = if ctx.cancel.load(Ordering::Relaxed) {
            Err("cancelled".into())
        } else {
            match run {
                RunSpec::E2eLoopback(r) => e2e_loopback::run_live(r, spec.seed, ctx.cancel, &progress).map(|o| {
                    let details = serde_json::json!({
                        "output_device": o.output_device,
                        "input_device": o.input_device,
                        "output_rate": o.output_rate,
                        "input_rate": o.input_rate,
                        "timebase_jitter_ms": [o.timebase_jitter_ms.0, o.timebase_jitter_ms.1],
                        "recorded_seconds": o.recorded_seconds,
                        "calibration_ms": r.calibration_ms,
                        "detector": r.detector,
                        "bursts": o.bursts,
                    });
                    RunOutcome {
                        metrics: o.metrics,
                        details,
                        ..RunOutcome::default()
                    }
                }),
                RunSpec::Offline(r) => match &prepared {
                    Some(Ok(p)) => offline::run_offline(&offline::OfflineInput {
                        stream: &spec.stream,
                        run: r,
                        engine: &p.engine,
                        input: &p.input,
                        probe: p.probe,
                        record_audio: spec.output.write_audio && out_dir.is_some(),
                        cancel: ctx.cancel,
                        progress: &progress,
                        recorder: None,
                    }),
                    Some(Err(e)) => Err(e.clone()),
                    None => Err("no engine".into()),
                },
                RunSpec::Paced(r) => match &prepared {
                    Some(Ok(p)) => paced::run_paced(&paced::PacedInput {
                        stream: &spec.stream,
                        run: r,
                        engine: &p.engine,
                        input: &p.input,
                        probe: p.probe,
                        record_audio: spec.output.write_audio && out_dir.is_some(),
                        cancel: ctx.cancel,
                        progress: &progress,
                    }),
                    Some(Err(e)) => Err(e.clone()),
                    None => Err("no engine".into()),
                },
                RunSpec::Contention(c) => match &prepared {
                    Some(Ok(p)) => contention::run(c, spec, p, out_dir.is_some(), ctx.cancel, &progress),
                    Some(Err(e)) => Err(e.clone()),
                    None => Err("no engine".into()),
                },
                RunSpec::Quality(q) => match &prepared {
                    Some(Ok(p)) => fidelity::run_quality(q, spec, p, out_dir.is_some(), ctx.cancel, &progress),
                    Some(Err(e)) => Err(e.clone()),
                    None => Err("no engine".into()),
                },
                RunSpec::Compare(c) => {
                    let engine = match &prepared {
                        Some(Ok(p)) => Some(p),
                        Some(Err(e)) if c.sigil.include => {
                            runs.push(error_report(run, run_started, e.clone()));
                            progress(1.0, "done");
                            continue;
                        }
                        _ => None,
                    };
                    compare::run_compare(c, spec, engine, &runs, out_dir.is_some(), ctx.cancel, &progress)
                }
                RunSpec::Autotune(a) => autotune::run_in_spec(a, spec, ctx.cancel, &progress),
                // P3B-S0 stubs (P3B-B, P3B-A).
                RunSpec::Rescore(r) => multi_clip::run_rescore(r, spec, ctx.cancel, &progress),
                RunSpec::LoadTwice(r) => match &prepared {
                    Some(Ok(p)) => attribution::run_load_twice_with(r, spec, Some(p), ctx.cancel, &progress),
                    Some(Err(e)) => Err(e.clone()),
                    None => attribution::run_load_twice(r, spec, ctx.cancel, &progress),
                },
            }
        };
        let run_load = attribution::LoadBreakdown::of(&telemetry::load_events_since(run_load_mark), run_w0, telemetry::now_ns());
        let state = state_probe.map(gpu_state::RunStateProbe::finish);
        let report = match result {
            Ok(mut outcome) => {
                if let Some(Ok(p)) = &prepared {
                    if run.uses_engine() {
                        if let Some(mb) = p.vram_engine_mb {
                            outcome.metrics.insert("vram_engine_mb".into(), mb);
                        }
                        // P3B-A: the engine load's breakdown, VRAM and CUDA figures, and this
                        // run's own generator builds (a quality run's ceiling builds its own).
                        if let Some(x) = &extras {
                            for (k, v) in &x.metrics {
                                outcome.metrics.entry(k.clone()).or_insert(*v);
                            }
                        }
                        outcome.metrics.insert("run_generator_builds".into(), f64::from(run_load.generator_builds));
                        // Extractor calls of this run, and those longer than the extractor's bound
                        // (calls it had to split: 0 with priming on a bounded plan).
                        let (calls, long_calls) = crate::services::neural_audio::vc::engine::f0_cache::process_call_counts();
                        outcome.metrics.insert("f0_calls".into(), calls.saturating_sub(f0_calls_mark.0) as f64);
                        outcome.metrics.insert("f0_long_calls".into(), long_calls.saturating_sub(f0_calls_mark.1) as f64);
                        // The run's own load work (the harness's re-plan to the live shape and its
                        // warm-up, a ceiling's builds): part of a cold start in the app.
                        outcome.metrics.insert("run_load_spans_ms".into(), run_load.stages_ms.values().sum::<f64>());
                    }
                }
                if (state.is_some() || !run_load.spans.is_empty()) && outcome.details.is_null() {
                    outcome.details = serde_json::json!({});
                }
                if !run_load.spans.is_empty() {
                    gpu_state::insert_json(&mut outcome.details, "run_load", &run_load);
                }
                if let Some(st) = &state {
                    for (k, v) in st.metrics() {
                        outcome.metrics.insert(k.into(), v);
                    }
                    gpu_state::insert_json(&mut outcome.details, "state", st);
                }
                let thresholds = evaluate(&outcome.metrics, &spec.thresholds, run);
                let (files, file_error) = match out_dir {
                    Some(dir) => match write_run_files(dir, run.name(), &outcome, spec) {
                        Ok(f) => (f, None),
                        Err(e) => (Vec::new(), Some(e)),
                    },
                    None => (Vec::new(), None),
                };
                let status = if file_error.is_some() {
                    RunStatus::Error
                } else if thresholds.iter().all(|t| t.passed) {
                    RunStatus::Passed
                } else {
                    RunStatus::Failed
                };
                RunReport {
                    name: run.name().to_string(),
                    mode: run.mode(),
                    status,
                    error: file_error,
                    duration_ms: run_started.elapsed().as_secs_f64() * 1000.0,
                    higher_is_better: report::higher_is_better_of(&outcome.metrics),
                    metrics: outcome.metrics,
                    thresholds,
                    details: outcome.details,
                    files,
                }
            }
            Err(e) => error_report(run, run_started, e),
        };
        progress(1.0, "done");
        runs.push(report);
    }

    let failures: Vec<String> = runs.iter().flat_map(run_failures).collect();
    drop(process_state);
    BenchReport {
        schema: REPORT_SCHEMA.to_string(),
        name: spec.name.clone(),
        description: spec.description.clone(),
        passed: failures.is_empty() && runs.iter().all(|r| r.status == RunStatus::Passed),
        failures,
        started_unix_ms,
        duration_ms: started.elapsed().as_secs_f64() * 1000.0,
        spec_sha256,
        spec: spec_value,
        environment: environment(),
        engine: match &prepared {
            Some(Ok(p)) => serde_json::to_value(&p.engine.info).ok().map(|mut v| {
                if let (Some(obj), Some(x)) = (v.as_object_mut(), &extras) {
                    let mut a = x.attribution.clone();
                    a.insert("gpu_timing".into(), gpu_timing.into());
                    let (load_dropped, gpu_dropped) = telemetry::global().dropped();
                    a.insert("telemetry_dropped".into(), serde_json::json!({"load_events": load_dropped, "gpu_times": gpu_dropped}));
                    obj.insert("attribution".into(), serde_json::Value::Object(a));
                }
                v
            }),
            _ => None,
        },
        runs,
    }
}

/// The report of a run that could not complete.
fn error_report(run: &RunSpec, started: Instant, e: String) -> RunReport {
    RunReport {
        name: run.name().to_string(),
        mode: run.mode(),
        status: if e == "cancelled" { RunStatus::Cancelled } else { RunStatus::Error },
        error: Some(e),
        duration_ms: started.elapsed().as_secs_f64() * 1000.0,
        metrics: Default::default(),
        thresholds: Vec::new(),
        details: serde_json::Value::Null,
        files: Vec::new(),
        higher_is_better: Vec::new(),
    }
}

// ---------------------------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------------------------

/// What holds the one in-app measurement slot. A benchmark and the autotuner both measure what this
/// PC can do, so they never run together: the slot is claimed with one compare-and-swap
/// ([`claim_measurement`]), never checked and then claimed (two near-simultaneous IPC calls cannot
/// both pass).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Measurement {
    Benchmark = 1,
    Autotune = 2,
}

impl Measurement {
    fn from_u8(v: u8) -> Option<Measurement> {
        match v {
            1 => Some(Measurement::Benchmark),
            2 => Some(Measurement::Autotune),
            _ => None,
        }
    }

    /// Why `wanted` cannot start while `self` holds the slot.
    pub(crate) fn busy_message(self, wanted: Measurement) -> String {
        match (self, wanted) {
            (Measurement::Benchmark, Measurement::Benchmark) => "a benchmark is already running".into(),
            (Measurement::Autotune, Measurement::Autotune) => "the autotuner is already running".into(),
            (Measurement::Benchmark, Measurement::Autotune) => "a benchmark is running; tune when it has finished".into(),
            (Measurement::Autotune, Measurement::Benchmark) => "the autotuner is running; benchmark when it has finished".into(),
        }
    }
}

static MEASURING: AtomicU8 = AtomicU8::new(0);

/// Holds the measurement slot; frees it when dropped.
#[derive(Debug)]
pub(crate) struct MeasurementGuard(Measurement);

impl Drop for MeasurementGuard {
    fn drop(&mut self) {
        let _ = MEASURING.compare_exchange(self.0 as u8, 0, Ordering::SeqCst, Ordering::SeqCst);
    }
}

/// Claim the measurement slot for `what` (`Err`: what already holds it).
pub(crate) fn claim_measurement(what: Measurement) -> Result<MeasurementGuard, Measurement> {
    match MEASURING.compare_exchange(0, what as u8, Ordering::SeqCst, Ordering::SeqCst) {
        Ok(_) => Ok(MeasurementGuard(what)),
        Err(held) => Err(Measurement::from_u8(held).unwrap_or(what)),
    }
}

static CANCEL: AtomicBool = AtomicBool::new(false);
static LAST_REPORT: Mutex<Option<serde_json::Value>> = Mutex::new(None);

/// `app_data/bench/`: IPC specs' base folder.
fn ipc_base_dir() -> Result<PathBuf, String> {
    crate::services::diagnostics::app_data_dir()
        .map(|d| d.join("bench"))
        .ok_or_else(|| "no app data folder".to_string())
}

/// An IPC `outDir`: relative to `base`, and never outside it.
fn ipc_out_dir(base: &Path, out_dir: Option<&str>, name: &str) -> Result<PathBuf, String> {
    let default = || base.join("results").join(format!("{name}-{}", unix_ms_now()));
    let Some(raw) = out_dir.map(str::trim).filter(|s| !s.is_empty()) else {
        return Ok(default());
    };
    let p = Path::new(raw);
    let rel = if p.is_absolute() {
        p.strip_prefix(base)
            .map_err(|_| format!("outDir must be inside {}", base.display()))?
            .to_path_buf()
    } else {
        p.to_path_buf()
    };
    if rel.components().any(|c| !matches!(c, Component::Normal(_))) {
        return Err("outDir must not contain '..' or a drive / root".into());
    }
    Ok(base.join(rel))
}

#[derive(Serialize, Clone)]
struct DonePayload {
    ok: bool,
    report: Option<serde_json::Value>,
    error: Option<String>,
}

/// Text of a panic payload.
fn panic_text(payload: &(dyn std::any::Any + Send)) -> String {
    payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_string())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".to_string())
}

/// Run `f`, turning a panic into an error, so the IPC caller and `bench:done` always get an answer.
fn catch_panic<T>(f: impl FnOnce() -> Result<T, String>) -> Result<T, String> {
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))
        .unwrap_or_else(|p| Err(format!("the benchmark failed unexpectedly: {}", panic_text(p.as_ref()))))
}

/// Validate, start and await one in-app benchmark. `guard` frees the one-benchmark slot when the
/// benchmark ends (or when it never starts).
async fn execute_benchmark<R: Runtime>(
    app: &AppHandle<R>,
    spec: serde_json::Value,
    out_dir: Option<String>,
    guard: MeasurementGuard,
) -> Result<serde_json::Value, String> {
    let base = ipc_base_dir()?;
    let mut spec = BenchSpec::from_value(spec)?;
    spec.resolve_paths(&base);
    let out = ipc_out_dir(&base, out_dir.as_deref(), &spec.name)?;
    let (tx, rx) = tokio::sync::oneshot::channel();
    let emitter = app.clone();
    std::thread::Builder::new()
        .name("sigil-bench".into())
        .stack_size(BENCH_STACK_BYTES)
        .spawn(move || {
            let last = Mutex::new(Instant::now() - std::time::Duration::from_secs(1));
            let progress = |p: &Progress| {
                // At most 4 events per second, plus every run's first and last.
                let mut t = last.lock().unwrap_or_else(PoisonError::into_inner);
                if p.fraction <= 0.0 || p.fraction >= 1.0 || t.elapsed().as_millis() >= 250 {
                    *t = Instant::now();
                    let _ = emitter.emit(PROGRESS_EVENT, p);
                }
            };
            let result = catch_panic(|| {
                let report = run_spec(&spec, Some(&out), &BenchContext { cancel: &CANCEL, progress: &progress });
                report::write_report(&out, &report)?;
                serde_json::to_value(&report).map_err(|e| e.to_string())
            });
            // Free the slot before answering, so a caller reacting to bench:done can start the next.
            drop(guard);
            let _ = tx.send(result);
        })
        .map_err(|e| format!("cannot start the benchmark thread: {e}"))?;
    rx.await.map_err(|_| "the benchmark thread stopped unexpectedly".to_string())?
}

#[command]
async fn run_benchmark<R: Runtime>(
    app: AppHandle<R>,
    spec: serde_json::Value,
    out_dir: Option<String>,
) -> Result<serde_json::Value, String> {
    // Refused without bench:done: that event belongs to the benchmark already running (or the
    // autotuner owns the slot).
    let guard = claim_measurement(Measurement::Benchmark).map_err(|held| held.busy_message(Measurement::Benchmark))?;
    CANCEL.store(false, Ordering::SeqCst);
    // Every other outcome (bad spec, blocked, thread failure, panic, report) ends in bench:done.
    let result = execute_benchmark(&app, spec, out_dir, guard).await;
    let payload = match &result {
        Ok(v) => {
            *LAST_REPORT.lock().unwrap_or_else(PoisonError::into_inner) = Some(v.clone());
            DonePayload { ok: true, report: Some(v.clone()), error: None }
        }
        Err(e) => DonePayload { ok: false, report: None, error: Some(e.clone()) },
    };
    let _ = app.emit(DONE_EVENT, payload);
    result
}

#[command]
async fn cancel_benchmark() -> Result<(), String> {
    CANCEL.store(true, Ordering::SeqCst);
    Ok(())
}

#[command]
async fn get_last_report() -> Result<Option<serde_json::Value>, String> {
    Ok(LAST_REPORT.lock().unwrap_or_else(PoisonError::into_inner).clone())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("bench")
        .invoke_handler(tauri::generate_handler![run_benchmark, cancel_benchmark, get_last_report])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet() -> impl Fn(&Progress) + Sync {
        |_: &Progress| {}
    }

    /// Only the CUDA rung (and the TensorRT accelerators on it) uses the context the benchmark
    /// retains before the load: a CUDA plan that fell back is not charged it.
    #[test]
    fn only_the_cuda_rung_uses_the_retained_context() {
        for active in ["cuda", "tensorrt_rtx", "tensorrt"] {
            assert!(cuda_rung_active(Some(active)), "{active}");
        }
        for active in ["dml", "dml_hp", "cpu", ""] {
            assert!(!cuda_rung_active(Some(active)), "{active}");
        }
        assert!(!cuda_rung_active(None));
    }

    /// An engine that ended on DirectML is not charged the CUDA context the benchmark retained: the
    /// VRAM delta loses it, and never goes below zero.
    #[test]
    fn an_unused_cuda_context_is_not_charged_to_the_engine() {
        assert_eq!(engine_vram_mb(Some(100), Some(930), None), Some(830.0));
        assert_eq!(engine_vram_mb(Some(100), Some(930), Some(330.0)), Some(500.0));
        assert_eq!(engine_vram_mb(Some(100), Some(150), Some(330.0)), Some(0.0));
        assert_eq!(engine_vram_mb(None, Some(1), None), None);
        assert_eq!(engine_vram_mb(Some(1), None, Some(5.0)), None);
    }

    /// A finished spec leaves no GPU speed-index session in the process (the in-app benchmark would
    /// otherwise keep a DirectML device and queue until the app exits).
    #[test]
    fn a_finished_spec_releases_the_speed_index() {
        let _serial = gpu_state::SPEED_CACHE_TEST_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        gpu_state::seed_cpu_speed_session_for_tests();
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "release", "engine": {"kind": "stub"},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 2},
            "runs": [{"mode": "offline", "name": "o"}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        assert_eq!(r.runs[0].status, RunStatus::Passed, "{:?}", r.runs[0].error);
        assert!(!gpu_state::speed_index_cached(), "run_spec releases the speed index at its end");
    }

    /// The same when the spec unwinds: the guard's drop runs during a panic.
    #[test]
    fn a_spec_that_unwinds_still_releases_the_speed_index() {
        let _serial = gpu_state::SPEED_CACHE_TEST_LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        gpu_state::seed_cpu_speed_session_for_tests();
        let unwound = std::panic::catch_unwind(|| {
            let _state = SpecProcessState::enter(false, false);
            panic!("a run panicked");
        });
        assert!(unwound.is_err());
        assert!(!gpu_state::speed_index_cached(), "the process state was restored during the unwind");
    }

    /// A benchmark and a tune share one slot, claimed atomically: whoever holds it refuses the
    /// other (and itself) until its guard drops.
    #[test]
    fn benchmark_and_autotune_share_one_slot() {
        let tune = autotune::begin().expect("free");
        assert_eq!(claim_measurement(Measurement::Benchmark).unwrap_err(), Measurement::Autotune);
        assert!(autotune::begin().unwrap_err().contains("already running"));
        let msg = Measurement::Autotune.busy_message(Measurement::Benchmark);
        assert!(msg.contains("autotuner is running"), "{msg}");
        drop(tune);
        let bench = claim_measurement(Measurement::Benchmark).expect("freed");
        assert!(autotune::begin().unwrap_err().contains("a benchmark is running"));
        drop(bench);
        // Many racing claims: exactly one wins.
        let wins = std::sync::atomic::AtomicUsize::new(0);
        let guards = Mutex::new(Vec::new());
        std::thread::scope(|s| {
            for i in 0..8 {
                let (wins, guards) = (&wins, &guards);
                s.spawn(move || {
                    let what = if i % 2 == 0 { Measurement::Benchmark } else { Measurement::Autotune };
                    if let Ok(g) = claim_measurement(what) {
                        wins.fetch_add(1, Ordering::SeqCst);
                        guards.lock().unwrap().push(g);
                    }
                });
            }
        });
        assert_eq!(wins.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn ipc_out_dirs_stay_inside_the_bench_folder() {
        let base = std::env::temp_dir().join("sigil-app").join("bench");
        let d = ipc_out_dir(&base, None, "x").unwrap();
        assert!(d.starts_with(base.join("results")) && d.file_name().unwrap().to_string_lossy().starts_with("x-"));
        assert_eq!(ipc_out_dir(&base, Some("runs/a"), "x").unwrap(), base.join("runs/a"));
        assert_eq!(ipc_out_dir(&base, Some(&base.join("b").to_string_lossy()), "x").unwrap(), base.join("b"));
        assert!(ipc_out_dir(&base, Some("../escape"), "x").is_err());
        assert!(ipc_out_dir(&base, Some(&std::env::temp_dir().join("elsewhere").to_string_lossy()), "x").is_err());
    }

    /// A whole spec: every mode's run lands in the report with its status, thresholds and files.
    #[test]
    fn run_spec_reports_passes_failures_and_errors() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "unit",
            "engine": {"kind": "stub", "compute_ms": 1.0},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 2},
            "output": {"write_audio": true},
            "thresholds": {"block_ms_p99": {"max": 1000}},
            "runs": [
                // A floor any PC meets even while a build saturates it (the mechanism is under test).
                {"mode": "offline", "name": "off", "thresholds": {"throughput_x": {"min": 0.01}}},
                {"mode": "paced", "name": "paced", "seconds": 2, "clock": "simulated",
                 "thresholds": {"underruns": {"max": 0}, "latency_ms_p99": {"max": 0.001}}},
                {"mode": "e2e_loopback", "name": "loop", "output_device": "no such device 1f9c",
                 "input_device": "no such device 1f9c"}
            ]
        }))
        .unwrap();
        let dir = std::env::temp_dir().join(format!("sigil-bench-runspec-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, Some(&dir), &BenchContext { cancel: &cancel, progress: &progress });
        assert_eq!(r.runs.len(), 3);
        assert_eq!(r.runs[0].status, RunStatus::Passed, "{:?}", r.runs[0]);
        assert!(r.runs[0].metrics["throughput_x"] > 0.01);
        assert!(r.runs[0].thresholds.iter().any(|t| t.metric == "block_ms_p99" && t.scope == "spec"));
        assert_eq!(r.runs[1].status, RunStatus::Failed, "latency cannot be under 1 us");
        assert!(r.failures.iter().any(|f| f.starts_with("paced: latency_ms_p99")), "{:?}", r.failures);
        // No such device (or no audio devices at all on a CI runner): an error, never a hang.
        assert_eq!(r.runs[2].status, RunStatus::Error, "{:?}", r.runs[2].error);
        assert!(!r.passed);
        assert_eq!(r.engine.as_ref().unwrap()["kind"], "stub");
        assert_eq!(r.spec_sha256.len(), 64);
        for f in ["off.timeline.csv", "off.output.wav", "paced.timeline.csv", "paced.output.wav"] {
            assert!(dir.join(f).is_file(), "{f} missing");
        }
        report::write_report(&dir, &r).unwrap();
        let back: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("report.json")).unwrap()).unwrap();
        assert_eq!(back["schema"], REPORT_SCHEMA);
        assert_eq!(back["runs"][1]["status"], "failed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A paced run that never plays any audio (the first block outlasts the run) measures no
    /// latency: a spec-wide latency gate fails it as "not measured" instead of skipping it, while
    /// `underruns` alone (0: playout never started) would have passed.
    #[test]
    fn a_spec_wide_gate_fails_a_run_that_measured_nothing() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "silent", "engine": {"kind": "stub", "compute_ms": 2000},
            "runs": [{"mode": "paced", "name": "p", "seconds": 1, "clock": "simulated", "warmup_s": 0}],
            "thresholds": {"latency_ms_p99": {"max": 1000}, "underruns": {"max": 0}}
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        let run = &r.runs[0];
        assert!(!run.metrics.contains_key("latency_ms_p50"), "{:?}", run.metrics);
        assert_eq!(run.metrics["underruns"], 0.0);
        assert_eq!(run.status, RunStatus::Failed, "{:?}", run.thresholds);
        let lat = run.thresholds.iter().find(|t| t.metric == "latency_ms_p99").expect("applied");
        assert!(!lat.passed && lat.value.is_none() && lat.scope == "spec", "{lat:?}");
        assert!(r.failures.iter().any(|f| f == "p: latency_ms_p99 = not measured (limit <= 1000)"), "{:?}", r.failures);
        assert!(!r.passed);
    }

    /// Blocks of 2 s and more at low device rates (valid chunk sizes): the latency probe sizes its
    /// input from the block, so offline and paced runs work instead of failing in `prepare`.
    #[test]
    fn long_blocks_at_low_rates_run() {
        for (rate, block) in [(16_000, 32_768), (8_000, 48_000)] {
            let spec = BenchSpec::from_value(serde_json::json!({
                "name": "long", "engine": {"kind": "stub"},
                "input": {"kind": "synthetic", "signal": "voice", "seconds": 8},
                "stream": {"input_rate": rate, "output_rate": rate, "block_frames": block},
                "runs": [{"mode": "offline", "name": "off"},
                         {"mode": "paced", "name": "p", "seconds": 20, "clock": "simulated", "warmup_s": 0}]
            }))
            .unwrap();
            let cancel = AtomicBool::new(false);
            let progress = quiet();
            let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
            for run in &r.runs {
                assert_eq!(run.status, RunStatus::Passed, "{rate}/{block} {}: {:?} {:?}", run.name, run.error, run.thresholds);
            }
            let alg = r.runs[0].metrics["algorithmic_latency_ms"];
            let block_ms = f64::from(block) * 1000.0 / f64::from(rate);
            assert!(alg > 0.0 && alg < block_ms + 1_000.0, "{rate}/{block}: {alg} ms");
            assert!(r.runs[1].metrics.contains_key("latency_ms_p50"), "{rate}/{block}");
        }
    }

    /// A fixed-block engine converts once every few device blocks: `call_ms` is the worker time of
    /// the blocks that ran the model (its per-block compute), `model_ms` is per call in both modes
    /// and `rtf` is judged on the calls, while `block_ms` over every device block is dominated by
    /// the accumulating ones.
    #[test]
    fn fixed_block_engines_report_their_per_call_compute() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "calls", "engine": {"kind": "stub", "compute_ms": 3.0, "compute": "sleep", "fixed_block_ms": 40},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 3},
            "stream": {"block_frames": 512},
            "runs": [{"mode": "offline", "name": "off"},
                     {"mode": "paced", "name": "p", "seconds": 3, "clock": "simulated", "warmup_s": 0.5}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        for run in &r.runs {
            let m = &run.metrics;
            assert_eq!(run.status, RunStatus::Passed, "{}: {:?}", run.name, run.error);
            assert!(m["call_ms_p50"] >= 2.5, "{}: {m:?}", run.name);
            assert!(m["model_ms_p50"] >= 2.5, "{}: per call, not per device block: {m:?}", run.name);
            assert!(m["block_ms_p50"] < 1.0, "{}: most device blocks only accumulate: {m:?}", run.name);
            // The real-time budget is judged on the calls: 3 ms of a 40 ms block, not 3 ms spread
            // over four device blocks.
            assert!(m["rtf_p50"] > 0.05, "{}: {m:?}", run.name);
        }
    }

    /// P3B-A on a stub: the engine's attribution (load window, host, the 10 ms vs 512-frame
    /// algorithmic latency of a fixed-block engine), the per-run state and load spans, the
    /// attribution metrics of every engine run, and a `load_twice` run that loads a second engine
    /// while the first streams.
    #[test]
    fn specs_report_attribution_state_and_load_twice() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "attr", "engine": {"kind": "stub", "compute_ms": 2.0, "compute": "sleep", "fixed_block_ms": 40},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 3},
            "stream": {"block_frames": 1920, "output_prefill_ms": 80, "output_max_queue_ms": 150},
            "vram": {"sample_ms": 10},
            "runs": [
                {"mode": "offline", "name": "off"},
                {"mode": "paced", "name": "p", "seconds": 3, "clock": "simulated", "warmup_s": 0.5},
                {"mode": "load_twice", "name": "twice", "thresholds": {"second_load_ms": {"max": 60000}}}
            ]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        for run in &r.runs {
            assert_eq!(run.status, RunStatus::Passed, "{}: {:?} {:?}", run.name, run.error, run.thresholds);
            assert!(run.details.get("state").is_some(), "{}: state", run.name);
            assert!(run.metrics.contains_key("state_host_probe_ms"), "{}", run.name);
            assert!(run.metrics.contains_key("load_other_ms") && run.metrics["generator_builds"] == 0.0, "{}: {:?}", run.name, run.metrics);
        }
        let a = &r.engine.as_ref().unwrap()["attribution"];
        assert!(a["load"]["window_ms"].as_f64().unwrap() >= 0.0, "{a}");
        assert!(a["load_breakdown_ms"].get("other").is_some(), "{a}");
        assert!(r.runs.iter().all(|run| run.metrics.contains_key("run_load_spans_ms")));
        assert!(a["host"].get("process_qos").is_some(), "{a}");
        let table = a["algorithmic_latency_table"].as_array().expect("a fixed-block engine gets the table");
        assert_eq!(table.len(), 8, "20/40/60/80 ms x 480/512 frames");
        let off = &r.runs[0];
        // The stub has no stages: the harness time of its calls is still reported.
        assert!(off.metrics.contains_key("harness_ms_p50"), "{:?}", off.metrics);
        assert!(off.details.get("cycles").is_some() && off.details.get("worker_thread").is_some());
        let paced = &r.runs[1];
        assert!(paced.metrics.contains_key("t1_hold_ms_p50") && paced.metrics.contains_key("t1_residual_p50_ms"), "{:?}", paced.metrics);
        // A fixed-block engine: the model calls map to its blocks, the parts add up to the p50. (The
        // paced run charges the stub's nominal compute: with measured sleeps the residual follows
        // the machine's load, which made this test fail under a loaded build machine.)
        assert!(paced.details["t1"]["engine_block_offset_frames"].is_number(), "{}", paced.details["t1"]);
        assert!(paced.metrics["t1_at_p50_engine_wait_ms"] >= 0.0 && paced.metrics["t1_residual_p50_ms"].abs() <= 1.0, "{:?}", paced.metrics);
        // The parts' means add up to the mean latency exactly.
        let t1 = &paced.details["t1"];
        assert!((t1["sum_of_means_ms"].as_f64().unwrap() - t1["latency_mean_ms"].as_f64().unwrap()).abs() < 1e-6, "{t1}");
        assert!(paced.metrics.contains_key("joins_silent"));
        let twice = &r.runs[2];
        assert!(twice.metrics["second_load_ms"] >= 0.0 && twice.metrics.contains_key("engine_load_ms"), "{:?}", twice.metrics);
        assert_eq!(twice.metrics["model_errors"], 0.0, "the background voice's errors (a spec-wide model_errors threshold applies)");
        // The first voice streamed (a stub loads too fast for calls during the load itself).
        assert!(twice.details["background"]["calls"].as_u64().unwrap() >= 1, "{}", twice.details);
        assert!(twice.details["load_breakdown"].is_object());
    }

    #[test]
    fn panics_become_errors() {
        assert_eq!(catch_panic(|| Ok::<_, String>(7)), Ok(7));
        assert_eq!(catch_panic(|| Err::<u8, _>("plain".to_string())), Err("plain".to_string()));
        let e = catch_panic::<u8>(|| panic!("probe exploded")).unwrap_err();
        assert!(e.contains("probe exploded"), "{e}");
        let e = catch_panic::<u8>(|| std::panic::panic_any(format!("formatted {}", 42))).unwrap_err();
        assert!(e.contains("formatted 42"), "{e}");
    }

    #[test]
    fn a_cancelled_spec_reports_cancelled_runs() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "c", "engine": {"kind": "stub"},
            "runs": [{"mode": "offline", "name": "a"}, {"mode": "paced", "name": "b", "clock": "virtual", "seconds": 2}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(true);
        let progress = quiet();
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        assert!(r.runs.iter().all(|x| x.status == RunStatus::Cancelled));
        assert!(!r.passed);
    }

    /// A spec on the real `RvcModel` path: the hand-encoded fake slot (content-encoder conv and
    /// an F0-following generator) on the CPU EP, through offline and paced runs.
    #[test]
    fn run_spec_drives_a_real_rvc_slot() {
        let (root, slot, hubert) = crate::services::neural_audio::vc::test_models::write_slot("bench-runspec");
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "fake-rvc",
            "engine": {"kind": "rvc", "slot_dir": slot, "hubert": hubert.to_string_lossy(),
                       "pitch_extractor": "dio", "execution_provider": "cpu", "preset": "streamer",
                       "silence_skip": false},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 2},
            "stream": {"output_prefill_ms": 80, "output_max_queue_ms": 150},
            "runs": [
                {"mode": "offline", "name": "off", "thresholds": {"model_errors": {"max": 0}}},
                {"mode": "paced", "name": "paced", "seconds": 2, "clock": "simulated",
                 "thresholds": {"model_errors": {"max": 0}, "backlog_over_budget_ms": {"max": 0}}}
            ]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        let engine = r.engine.as_ref().expect("engine info");
        assert_eq!(engine["kind"], "rvc");
        assert_eq!(engine["execution_provider_active"], "cpu");
        assert_eq!(engine["output_rate"], 40_000);
        for run in &r.runs {
            assert_eq!(run.status, RunStatus::Passed, "{}: {:?} {:?}", run.name, run.error, run.thresholds);
            assert!(run.metrics["blocks"] > 100.0, "{}", run.metrics["blocks"]);
        }
        assert!(r.runs[0].metrics["model_ms_p50"] > 0.0, "the real model's convert time is measured");
        assert!(r.runs[1].metrics.contains_key("latency_ms_p50"));
        let _ = std::fs::remove_dir_all(root);
    }

    #[test]
    fn an_engine_that_does_not_load_fails_its_runs_only() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "e", "engine": {"kind": "rvc", "slot_dir": "Z:/nope"},
            "runs": [{"mode": "offline", "name": "a"}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        assert_eq!(r.runs[0].status, RunStatus::Error);
        assert!(r.runs[0].error.as_deref().unwrap().contains("slot folder not found"));
        assert!(r.engine.is_none());
    }
}
