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
#[cfg(test)]
mod hot_path_tests;
/// In-app live recording of the running voice changer (P3-F; scaffolded by P3-S0).
pub mod live_record;
pub mod load_sim;
pub mod metrics;
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
mod vc_rt;

use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU8, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

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

fn prepare(spec: &BenchSpec) -> Result<Prepared, String> {
    let s = &spec.stream;
    let input = signals::render_input(&spec.input, s.input_rate, usize::from(s.input_channels), spec.seed)?;
    if input.len() / usize::from(s.input_channels) < s.block_frames as usize {
        return Err("the input clip is shorter than one block".into());
    }
    let vram0 = process_vram_mb();
    let priority = crate::services::rt_thread::WorkerPriority::from_str_lossy(&s.worker_priority);
    let mut engine = build_engine(&spec.engine, s, priority, spec.seed)?;
    let vram1 = process_vram_mb();
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
    Ok(Prepared {
        engine,
        probe,
        input,
        vram_engine_mb: match (vram0, vram1) {
            (Some(a), Some(b)) => Some(b.saturating_sub(a) as f64),
            _ => None,
        },
    })
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

    let prepared: Option<Result<Prepared, String>> = spec.runs.iter().any(RunSpec::uses_engine).then(|| {
        (ctx.progress)(&Progress {
            run: spec.runs[0].name().to_string(),
            run_index: 0,
            run_count,
            fraction: 0.0,
            message: "loading the engine".into(),
        });
        prepare(spec)
    });

    let mut runs: Vec<RunReport> = Vec::with_capacity(run_count);
    for (index, run) in spec.runs.iter().enumerate() {
        let run_started = Instant::now();
        let name = run.name().to_string();
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
            }
        };
        let report = match result {
            Ok(mut outcome) => {
                if let Some(Ok(p)) = &prepared {
                    if run.uses_engine() {
                        if let Some(mb) = p.vram_engine_mb {
                            outcome.metrics.insert("vram_engine_mb".into(), mb);
                        }
                    }
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
            Some(Ok(p)) => serde_json::to_value(&p.engine.info).ok(),
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
