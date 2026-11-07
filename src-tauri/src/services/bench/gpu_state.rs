//! GPU and host state of a benchmark run (P3B-A): NVML clocks, P-state, clocks-event reasons and
//! other processes' GPU use (through its own lazily loaded NVML), a GPU speed index for DirectML /
//! non-NVIDIA GPUs, and a host probe (a fixed CPU micro-benchmark, the power plan, the process QoS
//! and thread priority), sampled at run start and end and about once a second on a non-real-time
//! thread. Runs more than 10 % off the session median are flagged in `report.json`.
//!
//! # Frozen by P3B-S0 (signatures; P3B-A implements)
//! - [`GpuState`] and [`sample`]`() -> GpuState`.
//!
//! # Added by P3B-A
//! - [`RunStateProbe`]: a run's state at its start and end (with the host probe and, on DirectML
//!   or without NVML, the speed index) plus a 1 Hz series and the CPU's effective clock over the
//!   run (PDH `% Processor Performance`), summarised as [`RunState`] (`details.state` of a run).
//! - [`host_probe`], [`speed_index`], [`thread_state`], [`process_qos`], [`power_plan`].
//! - VRAM: [`VramSampler`] (the load-peak sampler around the engine load), [`cuda`] (the CUDA
//!   context's VRAM, `cuMemGetInfo`, `cuCtxGetLimit(STACK_SIZE)`, the shared ONNX Runtime CUDA
//!   arena's statistics).
//!
//! NVML, PDH and `powrprof` are loaded from System32 only, on first use, and only by a benchmark;
//! a missing library or entry point leaves its fields `None` (NVML absent: the speed index stands
//! in for the clocks). Nothing here runs inside a timed region: the sampler thread runs at normal
//! priority beside the worker, and every probe runs before or after a run.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use serde::Serialize;

use super::metrics::{percentile_sorted, process_vram_mb};

/// One sample of the GPU and host state (every field `None` when not measurable).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct GpuState {
    /// SM / memory clocks (MHz).
    pub sm_clock_mhz: Option<u32>,
    pub mem_clock_mhz: Option<u32>,
    /// Performance state (P0 = highest).
    pub pstate: Option<u32>,
    /// NVML clocks-event reasons (bit mask).
    pub clocks_event_reasons: Option<u64>,
    /// GPU utilization of other processes (%; the highest SM share of another process over the
    /// last sampling window, where the driver reports per-process samples).
    pub other_gpu_util_pct: Option<f64>,
    /// The GPU speed index (conv load p50, ms; DirectML / non-NVIDIA GPUs).
    pub speed_index_ms: Option<f64>,
    /// The fixed CPU micro-benchmark (ms).
    pub host_probe_ms: Option<f64>,
    /// The active power plan's name.
    pub power_plan: Option<String>,
    // ---- P3B-A ----
    /// Unix time of the sample (ms).
    pub at_unix_ms: u64,
    /// The NVML device sampled (name).
    pub device: Option<String>,
    pub graphics_clock_mhz: Option<u32>,
    /// Device-wide utilization (%): SM busy time, memory controller, video decoder / encoder.
    pub gpu_util_pct: Option<u32>,
    pub mem_util_pct: Option<u32>,
    pub decoder_util_pct: Option<u32>,
    pub encoder_util_pct: Option<u32>,
    pub temperature_c: Option<u32>,
    pub power_w: Option<f64>,
    /// The highest video-decoder share of another process (%).
    pub other_decoder_util_pct: Option<f64>,
    /// Other processes with GPU samples in the window.
    pub other_processes: Option<u32>,
    /// The other process with the highest SM share in the window: `name (pid N)` (its image
    /// name where Windows lets the benchmark read it), with that share.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub other_top_process: Option<String>,
    /// Names of the set [`clocks_event_reasons`](Self::clocks_event_reasons) bits.
    pub clocks_event_reason_names: Vec<&'static str>,
    pub power_plan_guid: Option<String>,
    /// Windows 11 power mode overlay (`best_power_efficiency`, `balanced`, `better_performance`,
    /// `best_performance`, or its GUID).
    pub power_mode: Option<String>,
    /// This process's EcoQoS state (`SetProcessInformation(ProcessPowerThrottling)`).
    pub process_qos: Option<QosState>,
    /// This process's priority class (`GetPriorityClass`, e.g. 0x20 = NORMAL).
    pub process_priority_class: Option<u32>,
    /// The micro-benchmark as an effective core clock (GHz, an estimate: 4 cycles per step).
    pub host_probe_ghz_est: Option<f64>,
    /// Runs the speed index timed.
    pub speed_index_runs: Option<u32>,
}

/// An EcoQoS (power throttling) state: `control_mask` = policies set explicitly (0 = system
/// managed), `state_mask` = those of them that are on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct QosState {
    pub control_mask: u32,
    pub state_mask: u32,
    /// Execution-speed throttling: `Some(false)` = opted out (HighQoS, the app's state),
    /// `Some(true)` = forced on (EcoQoS), `None` = left to Windows.
    pub execution_speed_throttled: Option<bool>,
}

impl QosState {
    fn new(control_mask: u32, state_mask: u32) -> QosState {
        const EXECUTION_SPEED: u32 = 0x1;
        QosState {
            control_mask,
            state_mask,
            execution_speed_throttled: (control_mask & EXECUTION_SPEED != 0).then_some(state_mask & EXECUTION_SPEED != 0),
        }
    }
}

/// The calling thread's scheduling state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct ThreadState {
    /// `GetThreadPriority` (0 = NORMAL, -1 = BELOW_NORMAL, 15 = TIME_CRITICAL).
    pub priority: Option<i32>,
    pub qos: Option<QosState>,
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Names of NVML clocks-event reason bits (`nvmlClocksEventReason*`; bits this table does not
/// know read as `other`).
pub fn reason_names(mask: u64) -> Vec<&'static str> {
    const NAMES: [(u64, &str); 9] = [
        (0x1, "gpu_idle"),
        (0x2, "applications_clocks_setting"),
        (0x4, "sw_power_cap"),
        (0x8, "hw_slowdown"),
        (0x10, "sync_boost"),
        (0x20, "sw_thermal_slowdown"),
        (0x40, "hw_thermal_slowdown"),
        (0x80, "hw_power_brake_slowdown"),
        (0x100, "display_clock_setting"),
    ];
    let mut names: Vec<&'static str> = NAMES.iter().filter(|(bit, _)| mask & bit != 0).map(|(_, n)| *n).collect();
    if mask & !0x1ff != 0 {
        names.push("other");
    }
    names
}

/// Reasons that slow the GPU below what the workload asks for (power, thermal, hardware).
pub const THROTTLE_REASONS: u64 = 0x4 | 0x8 | 0x20 | 0x40 | 0x80;

/// Sample the GPU and host state now: NVML (when an NVIDIA driver is present), the power plan,
/// this process's QoS and priority class. Cheap (no micro-benchmark; see [`RunStateProbe`]).
pub fn sample() -> GpuState {
    let mut s = GpuState {
        at_unix_ms: unix_ms(),
        ..GpuState::default()
    };
    if let Some(n) = nvml() {
        n.fill(nvml_device_index(), &mut s);
    }
    let plan = power_plan();
    s.power_plan = plan.name;
    s.power_plan_guid = plan.guid;
    s.power_mode = plan.mode;
    s.process_qos = process_qos();
    s.process_priority_class = imp::process_priority_class();
    s
}

/// The NVML device the benchmark samples: the first (the dev rig and most PCs have one NVIDIA
/// GPU); `SIGIL_BENCH_NVML_DEVICE` picks another.
fn nvml_device_index() -> u32 {
    std::env::var("SIGIL_BENCH_NVML_DEVICE")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------------------------
// Host
// ---------------------------------------------------------------------------------------------

/// The calling thread's priority and EcoQoS state.
pub fn thread_state() -> ThreadState {
    imp::thread_state()
}

/// This process's EcoQoS state.
pub fn process_qos() -> Option<QosState> {
    imp::process_qos()
}

/// The active power plan.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct PowerPlan {
    pub name: Option<String>,
    pub guid: Option<String>,
    pub mode: Option<String>,
}

/// The active power plan (name and GUID) and the Windows 11 power mode overlay.
pub fn power_plan() -> PowerPlan {
    imp::power_plan()
}

/// Steps of the host probe's dependent multiply-add chain (fewer in debug builds, whose numbers are
/// never compared with a release build's).
const HOST_PROBE_STEPS: u64 = if cfg!(debug_assertions) { 400_000 } else { 4_000_000 };

/// The fixed CPU micro-benchmark: a dependent 64-bit multiply-add chain (about 4 core cycles per
/// step on current x86: `imul` 3 + `add` 1) on the calling thread, best of 3. Returns `(ms, GHz
/// estimate)`. About 3-5 ms in a release build.
pub fn host_probe() -> (f64, f64) {
    let mut best = f64::INFINITY;
    for _ in 0..3 {
        let k = std::hint::black_box(6_364_136_223_846_793_005u64);
        let c = std::hint::black_box(1_442_695_040_888_963_407u64);
        let mut x = std::hint::black_box(0x9e37_79b9_7f4a_7c15u64);
        let t0 = Instant::now();
        for _ in 0..HOST_PROBE_STEPS {
            x = x.wrapping_mul(k).wrapping_add(c);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        std::hint::black_box(x);
        best = best.min(ms);
    }
    let ghz = if best > 0.0 { HOST_PROBE_STEPS as f64 * 4.0 / (best * 1e6) } else { 0.0 };
    (best, ghz)
}

/// CPU performance over a window (PDH `\Processor Information(_Total)\% Processor Performance`:
/// the effective clock relative to the nominal one, averaged over the logical processors).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub struct CpuPerf {
    pub perf_pct: f64,
    pub nominal_mhz: f64,
    pub effective_mhz: f64,
    /// `% Processor Utility` over the window (busy share of the whole CPU, frequency-scaled).
    pub utility_pct: Option<f64>,
}

/// A PDH query open over a window ([`CpuPerfMeter::start`] .. [`CpuPerfMeter::finish`]).
pub struct CpuPerfMeter(imp::Pdh);

impl CpuPerfMeter {
    /// Open the query and take the first sample (`None` without PDH).
    pub fn start() -> Option<CpuPerfMeter> {
        imp::Pdh::start().map(CpuPerfMeter)
    }

    /// The window's averages (`None` when PDH could not compute them, e.g. a window under 1 ms).
    pub fn finish(self) -> Option<CpuPerf> {
        self.0.finish()
    }
}

// ---------------------------------------------------------------------------------------------
// GPU speed index
// ---------------------------------------------------------------------------------------------

/// Size and depth of the speed index graph: the `bench/gpu-load` conv hammer (`h2_load70`'s).
const SPEED_SIZE: usize = 256;
const SPEED_DEPTH: usize = 8;
/// Timed runs (fewer when they would take longer than [`SPEED_BUDGET`]).
const SPEED_RUNS: u32 = 200;
const SPEED_MIN_RUNS: u32 = 20;
const SPEED_BUDGET: Duration = Duration::from_millis(1_500);

/// A GPU speed index: the conv hammer's p50 on one EP.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SpeedIndex {
    pub p50_ms: f64,
    pub p90_ms: f64,
    pub runs: u32,
    pub execution_provider: String,
    pub execution_provider_active: String,
}

struct SpeedSession {
    key: (String, i32),
    session: ort::session::Session,
    active: String,
    input: ort::value::Tensor<f32>,
}

/// The speed index's session (built on first use per EP and device). It owns its own D3D12 device
/// and queue on DirectML (and a CUDA context reference on CUDA) plus an 8 MB input tensor, so a
/// benchmark releases it when its spec ends ([`release_speed_index`]): the in-app benchmark must
/// not leave a GPU session behind in the running app.
static SPEED_CACHE: Mutex<Option<SpeedSession>> = Mutex::new(None);

/// Drop the cached speed-index session (its GPU device, queue and memory). The next
/// [`speed_index`] builds a new one. A no-op when none is cached.
pub fn release_speed_index() {
    clear_speed_cache(&SPEED_CACHE);
}

fn clear_speed_cache(cache: &Mutex<Option<SpeedSession>>) {
    *cache.lock().unwrap_or_else(PoisonError::into_inner) = None;
}

/// Whether a speed-index session is cached (tests, diagnostics).
#[allow(dead_code)]
pub fn speed_index_cached() -> bool {
    SPEED_CACHE.lock().unwrap_or_else(PoisonError::into_inner).is_some()
}

/// Tests that seed or inspect the speed-index cache hold this (a spec running in a parallel test
/// releases the cache at its end, which would empty a seeded one under them).
#[cfg(test)]
pub(super) static SPEED_CACHE_TEST_LOCK: Mutex<()> = Mutex::new(());

/// A small CPU session in a speed-index slot, as if a GPU speed index had run (no GPU is needed to
/// test the cache's lifetime).
#[cfg(test)]
fn cpu_speed_session() -> SpeedSession {
    use crate::services::neural_audio::ort_session::{build_session, ModelSource, VcSessionOptions};
    let bytes = super::load_sim::conv_chain_model(16, 2, 7);
    let (session, outcome) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::Memory(&bytes)).expect("a CPU session");
    let c = super::load_sim::CONV_CHANNELS;
    let input = ort::value::Tensor::from_array(([1usize, c, 16, 16], vec![0.25f32; c * 16 * 16])).expect("a tensor");
    SpeedSession {
        key: ("cpu".into(), 0),
        session,
        active: outcome.active,
        input,
    }
}

/// Seed the process-wide cache with [`cpu_speed_session`] (checked while the lock is held).
#[cfg(test)]
pub(super) fn seed_cpu_speed_session_for_tests() {
    let mut cache = SPEED_CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    *cache = Some(cpu_speed_session());
    assert!(cache.is_some());
}

/// The conv hammer run [`SPEED_RUNS`] times on `ep` (`dml` / `cuda`) and device `device`, after 5
/// untimed runs; its p50 (ms). The session is built once per spec and EP and recorded under no
/// diagnostics component (a probe's EP row would linger in Settings after the benchmark; its
/// active EP is in the result); [`release_speed_index`] drops it. `None` when the EP or the graph
/// cannot run.
pub fn speed_index(ep: &str, device: i32) -> Option<SpeedIndex> {
    use crate::services::neural_audio::ort_session::{build_session, normalize_ep, ModelSource, VcSessionOptions};
    let ep = normalize_ep(ep);
    if ep == "cpu" {
        return None;
    }
    let mut cache = SPEED_CACHE.lock().unwrap_or_else(PoisonError::into_inner);
    let key = (ep.to_string(), device);
    if cache.as_ref().is_none_or(|c| c.key != key) {
        *cache = None;
        let bytes = super::load_sim::conv_chain_model(SPEED_SIZE, SPEED_DEPTH, 0x5eed);
        let opts = VcSessionOptions {
            cuda_device_index: device,
            intra_threads: Some(1),
            ..VcSessionOptions::default()
        };
        let (session, outcome) = build_session(None, ep, &opts, ModelSource::Memory(&bytes)).ok()?;
        if outcome.active == "cpu" {
            // A CPU fallback is no GPU index.
            return None;
        }
        let c = super::load_sim::CONV_CHANNELS;
        let data = vec![0.25f32; c * SPEED_SIZE * SPEED_SIZE];
        let input = ort::value::Tensor::from_array(([1usize, c, SPEED_SIZE, SPEED_SIZE], data)).ok()?;
        *cache = Some(SpeedSession {
            key,
            session,
            active: outcome.active,
            input,
        });
    }
    let s = cache.as_mut()?;
    for _ in 0..5 {
        s.session.run(ort::inputs!["x" => &s.input]).ok()?;
    }
    let mut times = Vec::with_capacity(SPEED_RUNS as usize);
    let t_all = Instant::now();
    for i in 0..SPEED_RUNS {
        if i >= SPEED_MIN_RUNS && t_all.elapsed() > SPEED_BUDGET {
            break;
        }
        let t0 = Instant::now();
        s.session.run(ort::inputs!["x" => &s.input]).ok()?;
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    times.sort_by(f64::total_cmp);
    Some(SpeedIndex {
        p50_ms: percentile_sorted(&times, 50.0),
        p90_ms: percentile_sorted(&times, 90.0),
        runs: times.len() as u32,
        execution_provider: ep.to_string(),
        execution_provider_active: s.active.clone(),
    })
}

// ---------------------------------------------------------------------------------------------
// Run state
// ---------------------------------------------------------------------------------------------

/// What a run's state probe needs to know about the engine.
#[derive(Clone, Debug, Default)]
pub struct StateContext {
    /// The EP the engine's sessions run on (`cuda`, `dml`, `cpu`, ...; `None` = no engine).
    pub engine_ep: Option<String>,
    /// The engine's GPU device (DirectML adapter / CUDA ordinal).
    pub device: i32,
    /// Sampling period of the series (ms; 0 = start and end only).
    pub period_ms: u64,
    /// Time the GPU speed index at the run boundaries (DirectML, or a GPU EP without NVML).
    pub speed_index: bool,
}

/// Summary of a run's 1 Hz series.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct StateSummary {
    pub samples: usize,
    pub sm_clock_min_mhz: Option<u32>,
    pub sm_clock_median_mhz: Option<f64>,
    pub sm_clock_max_mhz: Option<u32>,
    pub mem_clock_median_mhz: Option<f64>,
    /// Every P-state seen.
    pub pstates: Vec<u32>,
    /// Every clocks-event reason seen.
    pub reasons: Vec<&'static str>,
    pub gpu_util_median_pct: Option<f64>,
    pub decoder_util_max_pct: Option<u32>,
    pub other_gpu_util_max_pct: Option<f64>,
    pub other_decoder_util_max_pct: Option<f64>,
    pub other_processes_max: Option<u32>,
    /// The distinct other processes that used at least 5 % SM in a sample (the evidence of a
    /// flagged run).
    pub other_top_processes: Vec<String>,
    pub temperature_max_c: Option<u32>,
    pub power_median_w: Option<f64>,
}

/// A run's state (`details.state`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct RunState {
    pub start: GpuState,
    pub end: GpuState,
    pub series: Vec<GpuState>,
    pub summary: StateSummary,
    pub cpu: Option<CpuPerf>,
    /// Things that make the run's timing suspect (GPU throttling, another process decoding video
    /// or using the GPU, a P-state change).
    pub warnings: Vec<String>,
}

impl RunState {
    /// The scalars a report's `metrics` carry (`state_*`; never gated by default).
    pub fn metrics(&self) -> Vec<(&'static str, f64)> {
        let mut m = Vec::new();
        let s = &self.summary;
        if let Some(v) = s.sm_clock_median_mhz {
            m.push(("state_sm_clock_mhz", v));
        }
        if let Some(v) = s.mem_clock_median_mhz {
            m.push(("state_mem_clock_mhz", v));
        }
        if let Some(v) = self.start.speed_index_ms {
            m.push(("state_speed_index_ms", v));
        }
        if let Some(v) = self.start.host_probe_ms {
            m.push(("state_host_probe_ms", v));
        }
        if let Some(c) = &self.cpu {
            m.push(("state_cpu_effective_mhz", c.effective_mhz));
        }
        if let Some(v) = s.decoder_util_max_pct {
            m.push(("state_decoder_util_max_pct", f64::from(v)));
        }
        m
    }
}

/// The 1 Hz sampler thread of a [`RunStateProbe`]. Stopped and joined when dropped, so a probe
/// that is abandoned (a run skipped before `finish`, an unwinding panic) leaves no thread sampling
/// NVML for the rest of the process.
struct SeriesThread {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<Vec<GpuState>>>,
}

impl SeriesThread {
    /// Sample with `sample_fn` every `period` (at most 4 h of samples are kept).
    fn spawn(period: Duration, sample_fn: impl Fn() -> GpuState + Send + 'static) -> Option<SeriesThread> {
        let stop = Arc::new(AtomicBool::new(false));
        let thread_stop = stop.clone();
        let handle = std::thread::Builder::new()
            .name("sigil-bench-state".into())
            .spawn(move || {
                let mut out = Vec::new();
                let mut next = Instant::now() + period;
                while !thread_stop.load(Ordering::Relaxed) {
                    let now = Instant::now();
                    if now < next {
                        std::thread::sleep((next - now).min(Duration::from_millis(50)));
                        continue;
                    }
                    next += period;
                    if out.len() < 4 * 3600 {
                        out.push(sample_fn());
                    }
                }
                out
            })
            .ok()?;
        Some(SeriesThread { stop, handle: Some(handle) })
    }

    /// Stop the thread and take its samples.
    fn finish(mut self) -> Vec<GpuState> {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.take().and_then(|h| h.join().ok()).unwrap_or_default()
    }
}

impl Drop for SeriesThread {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// A run's state probe: [`start`](Self::start) before the run, [`finish`](Self::finish) after.
/// Dropping it without `finish` stops its sampler thread and closes its CPU query.
pub struct RunStateProbe {
    ctx: StateContext,
    start: GpuState,
    cpu: Option<CpuPerfMeter>,
    series: Option<SeriesThread>,
}

/// The full sample of a run boundary: [`sample`] plus the host probe and, on DirectML (or a GPU
/// EP without NVML), the speed index.
fn boundary_sample(ctx: &StateContext) -> GpuState {
    let mut s = sample();
    let (ms, ghz) = host_probe();
    s.host_probe_ms = Some(ms);
    s.host_probe_ghz_est = Some(ghz);
    let ep = ctx.engine_ep.as_deref().unwrap_or("cpu");
    let dml = matches!(ep, "dml" | "dml_hp");
    let gpu = dml || matches!(ep, "cuda" | "tensorrt_rtx" | "tensorrt");
    if ctx.speed_index && (dml || (gpu && s.sm_clock_mhz.is_none())) {
        let rung = if dml { "dml" } else { "cuda" };
        if let Some(idx) = speed_index(rung, ctx.device) {
            s.speed_index_ms = Some(idx.p50_ms);
            s.speed_index_runs = Some(idx.runs);
        }
    }
    s
}

impl RunStateProbe {
    pub fn start(ctx: StateContext) -> RunStateProbe {
        let start = boundary_sample(&ctx);
        let series = if ctx.period_ms > 0 && nvml().is_some() {
            SeriesThread::spawn(Duration::from_millis(ctx.period_ms), sample)
        } else {
            None
        };
        RunStateProbe {
            cpu: CpuPerfMeter::start(),
            ctx,
            start,
            series,
        }
    }

    pub fn finish(self) -> RunState {
        let cpu = self.cpu.and_then(CpuPerfMeter::finish);
        let series = self.series.map(SeriesThread::finish).unwrap_or_default();
        let end = boundary_sample(&self.ctx);
        let mut all: Vec<&GpuState> = vec![&self.start];
        all.extend(series.iter());
        all.push(&end);
        let summary = summarize(&all);
        let warnings = state_warnings(&summary, self.ctx.engine_ep.as_deref());
        RunState {
            start: self.start,
            end,
            series,
            summary,
            cpu,
            warnings,
        }
    }
}

fn median_f(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(percentile_sorted(&v, 50.0))
}

/// Summary of a series of samples.
pub fn summarize(samples: &[&GpuState]) -> StateSummary {
    let sm: Vec<u32> = samples.iter().filter_map(|s| s.sm_clock_mhz).collect();
    let mut pstates: Vec<u32> = samples.iter().filter_map(|s| s.pstate).collect();
    pstates.sort_unstable();
    pstates.dedup();
    let mask = samples.iter().filter_map(|s| s.clocks_event_reasons).fold(0u64, |a, b| a | b);
    let max_f = |f: fn(&GpuState) -> Option<f64>| samples.iter().filter_map(|s| f(s)).fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b))));
    StateSummary {
        samples: samples.len(),
        sm_clock_min_mhz: sm.iter().copied().min(),
        sm_clock_median_mhz: median_f(sm.iter().map(|&v| f64::from(v)).collect()),
        sm_clock_max_mhz: sm.iter().copied().max(),
        mem_clock_median_mhz: median_f(samples.iter().filter_map(|s| s.mem_clock_mhz).map(f64::from).collect()),
        pstates,
        reasons: reason_names(mask),
        gpu_util_median_pct: median_f(samples.iter().filter_map(|s| s.gpu_util_pct).map(f64::from).collect()),
        decoder_util_max_pct: samples.iter().filter_map(|s| s.decoder_util_pct).max(),
        other_gpu_util_max_pct: max_f(|s| s.other_gpu_util_pct),
        other_decoder_util_max_pct: max_f(|s| s.other_decoder_util_pct),
        other_processes_max: samples.iter().filter_map(|s| s.other_processes).max(),
        other_top_processes: {
            let mut v: Vec<String> = samples
                .iter()
                .filter(|s| s.other_gpu_util_pct.is_some_and(|u| u >= 5.0))
                .filter_map(|s| s.other_top_process.as_deref())
                .map(|t| t.split(" (pid").next().unwrap_or(t).to_string())
                .collect();
            v.sort();
            v.dedup();
            v
        },
        temperature_max_c: samples.iter().filter_map(|s| s.temperature_c).max(),
        power_median_w: median_f(samples.iter().filter_map(|s| s.power_w).collect()),
    }
}

/// Why a run's timing may be suspect.
pub fn state_warnings(s: &StateSummary, engine_ep: Option<&str>) -> Vec<String> {
    let mut w = Vec::new();
    let throttles: Vec<&str> = s
        .reasons
        .iter()
        .copied()
        .filter(|r| reason_names(THROTTLE_REASONS).contains(r))
        .collect();
    if !throttles.is_empty() {
        w.push(format!("the GPU throttled during the run ({})", throttles.join(", ")));
    }
    if s.other_decoder_util_max_pct.is_some_and(|v| v > 0.0) || (s.other_decoder_util_max_pct.is_none() && s.decoder_util_max_pct.is_some_and(|v| v > 0)) {
        w.push(format!(
            "the GPU's video decoder was busy (up to {} %; a browser or player in another process)",
            s.other_decoder_util_max_pct.map_or_else(|| s.decoder_util_max_pct.unwrap_or(0).to_string(), |v| format!("{v:.0}"))
        ));
    }
    if s.other_gpu_util_max_pct.is_some_and(|v| v >= 5.0) {
        let who = if s.other_top_processes.is_empty() { String::new() } else { format!(": {}", s.other_top_processes.join(", ")) };
        w.push(format!("another process used the GPU (up to {:.0} % SM{who})", s.other_gpu_util_max_pct.unwrap_or(0.0)));
    }
    let gpu_engine = matches!(engine_ep, Some("cuda" | "dml" | "dml_hp" | "tensorrt_rtx" | "tensorrt"));
    if gpu_engine && s.pstates.len() > 1 && s.pstates.iter().any(|&p| p > 2) {
        w.push(format!("the GPU changed P-state during the run ({:?})", s.pstates));
    }
    w
}

/// Values more than `tolerance` (0.10 = 10 %) off the median of `values` (`None` entries are
/// skipped): their indices. The session-level slow-state flag (the gate applies it across runs;
/// its `-SelfTest` mirrors this function's test).
#[allow(dead_code)]
pub fn off_median(values: &[Option<f64>], tolerance: f64) -> Vec<usize> {
    let Some(med) = median_f(values.iter().flatten().copied().collect()) else {
        return Vec::new();
    };
    if med == 0.0 {
        return Vec::new();
    }
    values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_some_and(|v| ((v - med) / med).abs() > tolerance))
        .map(|(i, _)| i)
        .collect()
}

// ---------------------------------------------------------------------------------------------
// VRAM
// ---------------------------------------------------------------------------------------------

/// A thread sampling this process's VRAM every `period` until [`stop`](Self::stop): the peak
/// around an engine load (`vram_load_peak_mb`; DXGI, about 0.1-1 ms per sample, off the loading
/// thread).
pub struct VramSampler {
    stop: Arc<AtomicBool>,
    handle: Option<JoinHandle<(Option<u64>, usize)>>,
}

impl VramSampler {
    pub fn start(period: Duration) -> VramSampler {
        let stop = Arc::new(AtomicBool::new(false));
        let s2 = stop.clone();
        let period = period.max(Duration::from_millis(1));
        let handle = std::thread::Builder::new()
            .name("sigil-bench-vram".into())
            .spawn(move || {
                let mut peak: Option<u64> = None;
                let mut n = 0usize;
                loop {
                    if let Some(mb) = process_vram_mb() {
                        peak = Some(peak.map_or(mb, |p| p.max(mb)));
                        n += 1;
                    }
                    if s2.load(Ordering::Relaxed) {
                        break;
                    }
                    std::thread::sleep(period);
                }
                (peak, n)
            })
            .ok();
        VramSampler { stop, handle }
    }

    /// Stop; returns `(peak MiB, samples)`. The thread takes one last sample after the stop.
    pub fn stop(mut self) -> (Option<u64>, usize) {
        self.stop.store(true, Ordering::Relaxed);
        self.handle.take().and_then(|h| h.join().ok()).unwrap_or((None, 0))
    }
}

impl Drop for VramSampler {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

/// CUDA-side VRAM probes (the CUDA rung only).
pub mod cuda {
    use std::collections::BTreeMap;

    use serde::Serialize;

    use crate::services::neural_audio::gpu::cuda_driver::PrimaryContext;

    /// The primary context of the engine's CUDA device, retained by the benchmark before the load
    /// (so its own VRAM is measured once: `vram_cuda_context_mb`) and released when the spec ends.
    pub struct ContextProbe {
        ctx: PrimaryContext,
        /// What creating the context added to this process's VRAM (MiB; `None` when it already
        /// existed or DXGI cannot tell).
        pub context_mb: Option<f64>,
        /// Device memory in use before the load (`cuMemGetInfo`, every process; MiB).
        pub device_used_before_mb: Option<f64>,
        pub device_total_mb: Option<f64>,
    }

    /// Device memory snapshot of a [`ContextProbe`].
    #[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
    pub struct DeviceMemory {
        pub used_mb: f64,
        pub total_mb: f64,
    }

    impl ContextProbe {
        /// Retain device `ordinal`'s primary context (creating it when this process has none yet).
        pub fn retain(ordinal: i32) -> Result<ContextProbe, String> {
            let before = super::process_vram_mb();
            let ctx = PrimaryContext::retain(ordinal)?;
            let after = super::process_vram_mb();
            let mut p = ContextProbe {
                ctx,
                context_mb: match (before, after) {
                    (Some(a), Some(b)) if b > a => Some((b - a) as f64),
                    _ => None,
                },
                device_used_before_mb: None,
                device_total_mb: None,
            };
            if let Some(m) = p.memory() {
                p.device_used_before_mb = Some(m.used_mb);
                p.device_total_mb = Some(m.total_mb);
            }
            Ok(p)
        }

        /// `cuMemGetInfo` now.
        pub fn memory(&self) -> Option<DeviceMemory> {
            let (free, total) = self.ctx.mem_info().ok()?;
            const MIB: f64 = 1024.0 * 1024.0;
            Some(DeviceMemory {
                used_mb: (total.saturating_sub(free)) as f64 / MIB,
                total_mb: total as f64 / MIB,
            })
        }

        /// `cuCtxGetLimit(CU_LIMIT_STACK_SIZE)` of the context (bytes per thread). The driver grows
        /// the per-thread local-memory reservation with it and never shrinks it.
        pub fn stack_limit_bytes(&self) -> Option<u64> {
            let f = ctx_get_limit()?;
            let _scope = self.ctx.enter().ok()?;
            let mut v: usize = 0;
            // SAFETY: `CUresult cuCtxGetLimit(size_t*, CUlimit)` on the current (pushed) context;
            // valid out pointer; CU_LIMIT_STACK_SIZE = 0.
            let r = unsafe { f(&mut v, 0) };
            (r == 0).then_some(v as u64)
        }

        pub fn ordinal(&self) -> i32 {
            self.ctx.ordinal()
        }
    }

    type FnCtxGetLimit = unsafe extern "system" fn(*mut usize, i32) -> i32;

    fn ctx_get_limit() -> Option<FnCtxGetLimit> {
        static F: std::sync::OnceLock<Option<(libloading::Library, FnCtxGetLimit)>> = std::sync::OnceLock::new();
        F.get_or_init(|| {
            #[cfg(windows)]
            {
                use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
                // SAFETY: the NVIDIA driver's user-mode library from System32 only (already loaded
                // by the CUDA layer; this takes another reference).
                let lib: libloading::Library = unsafe { Library::load_with_flags("nvcuda.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }.ok()?.into();
                // SAFETY: the export's signature per cuda.h.
                let f: FnCtxGetLimit = unsafe { lib.get::<FnCtxGetLimit>(b"cuCtxGetLimit\0") }.ok().map(|s| *s)?;
                Some((lib, f))
            }
            #[cfg(not(windows))]
            {
                None
            }
        })
        .as_ref()
        .map(|(_, f)| *f)
    }

    /// Statistics of the shared ONNX Runtime CUDA arena of `device` (`AllocatorGetStats`:
    /// `Limit`, `InUse`, `TotalAllocated`, `MaxInUse`, `NumAllocs`, `NumReserves`,
    /// `NumArenaExtensions`, ...), when the voice engine registered it
    /// (`gpu::arena::ensure_cuda_env_arena`). Sessions with their own arenas (CUDA graphs, the
    /// default) are not reachable from here; DirectML has no arena (DXGI deltas only).
    pub fn env_arena_stats(device: i32) -> Option<BTreeMap<String, String>> {
        use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
        use ort::AsPointer;

        if !crate::services::neural_audio::gpu::arena::cuda_env_arena_registered(device) {
            return None;
        }
        let env = ort::environment::Environment::current().ok()?;
        let mem_info = MemoryInfo::new(AllocationDevice::CUDA, device.max(0), AllocatorType::Device, MemoryType::Default).ok()?;
        let api = ort::api();
        let mut allocator: *mut ort::sys::OrtAllocator = std::ptr::null_mut();
        // SAFETY: live environment and memory info; the shared allocator stays owned by the env.
        let status = unsafe { (api.GetSharedAllocator)(env.ptr().cast_mut(), mem_info.ptr(), &mut allocator) };
        // SAFETY: a status just returned by ORT.
        unsafe { ort::Error::result_from_status(status) }.ok()?;
        // SAFETY: the env's shared allocator lives as long as the environment (held above).
        unsafe { crate::services::neural_audio::telemetry::allocator_stats_raw(allocator) }
    }
}

// ---------------------------------------------------------------------------------------------
// NVML
// ---------------------------------------------------------------------------------------------

/// The benchmark's own NVML (loaded once per process; `None` without an NVIDIA driver).
fn nvml() -> Option<&'static imp::Nvml> {
    static NVML: OnceLock<Option<imp::Nvml>> = OnceLock::new();
    NVML.get_or_init(|| {
        let nvidia = crate::services::diagnostics::gpu_mem::adapters()
            .iter()
            .any(|a| a.vendor_id == crate::services::diagnostics::gpu_mem::VENDOR_NVIDIA && !a.software);
        if !nvidia {
            return None;
        }
        imp::Nvml::load().ok()
    })
    .as_ref()
}

/// Whether NVML loaded (an NVIDIA driver is present).
pub fn nvml_available() -> bool {
    nvml().is_some()
}

#[cfg(windows)]
mod imp {
    use super::{GpuState, PowerPlan, QosState, ThreadState};
    use libloading::os::windows::{Library, LOAD_LIBRARY_SEARCH_SYSTEM32};
    use std::ffi::{c_char, c_void};
    use std::sync::Mutex;
    use windows::core::GUID;
    use windows::Win32::System::Threading::{
        GetCurrentProcess, GetCurrentThread, GetPriorityClass, GetProcessInformation, GetThreadInformation, GetThreadPriority, ProcessPowerThrottling,
        ThreadPowerThrottling, PROCESS_POWER_THROTTLING_CURRENT_VERSION, PROCESS_POWER_THROTTLING_STATE, THREAD_POWER_THROTTLING_CURRENT_VERSION,
        THREAD_POWER_THROTTLING_STATE,
    };

    // ---- NVML ----------------------------------------------------------------------------------

    const NVML_SUCCESS: i32 = 0;
    const NVML_ERROR_INSUFFICIENT_SIZE: i32 = 7;
    const CLOCK_GRAPHICS: i32 = 0;
    const CLOCK_SM: i32 = 1;
    const CLOCK_MEM: i32 = 2;

    type Device = *mut c_void;
    type FnVoid = unsafe extern "C" fn() -> i32;
    type FnHandle = unsafe extern "C" fn(u32, *mut Device) -> i32;
    type FnName = unsafe extern "C" fn(Device, *mut c_char, u32) -> i32;
    type FnClock = unsafe extern "C" fn(Device, i32, *mut u32) -> i32;
    type FnU32 = unsafe extern "C" fn(Device, *mut u32) -> i32;
    type FnI32 = unsafe extern "C" fn(Device, *mut i32) -> i32;
    type FnU64 = unsafe extern "C" fn(Device, *mut u64) -> i32;
    type FnUtil = unsafe extern "C" fn(Device, *mut Utilization) -> i32;
    type FnCodec = unsafe extern "C" fn(Device, *mut u32, *mut u32) -> i32;
    type FnTemp = unsafe extern "C" fn(Device, i32, *mut u32) -> i32;
    type FnProc = unsafe extern "C" fn(Device, *mut ProcessSample, *mut u32, u64) -> i32;

    #[repr(C)]
    #[derive(Default)]
    struct Utilization {
        gpu: u32,
        memory: u32,
    }

    /// `nvmlProcessUtilizationSample_t`.
    #[repr(C)]
    #[derive(Clone, Copy, Default)]
    struct ProcessSample {
        pid: u32,
        time_stamp: u64,
        sm_util: u32,
        mem_util: u32,
        enc_util: u32,
        dec_util: u32,
    }

    pub struct Nvml {
        shutdown: FnVoid,
        handle: FnHandle,
        name: Option<FnName>,
        clock: Option<FnClock>,
        pstate: Option<FnI32>,
        reasons: Option<FnU64>,
        util: Option<FnUtil>,
        decoder: Option<FnCodec>,
        encoder: Option<FnCodec>,
        temperature: Option<FnTemp>,
        power: Option<FnU32>,
        processes: Option<FnProc>,
        /// Last per-process sample time (µs) per device: the next query returns newer samples.
        last_seen: Mutex<std::collections::BTreeMap<u32, u64>>,
        _lib: Library,
    }

    // SAFETY: function pointers into a library that lives as long as the struct; NVML is
    // thread-safe; the timestamp map is behind a mutex.
    unsafe impl Send for Nvml {}
    unsafe impl Sync for Nvml {}

    fn sym<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
        // SAFETY: every call site names a documented NVML export with its exact C signature.
        unsafe { lib.get::<T>(name) }.ok().map(|s| *s)
    }

    impl Nvml {
        pub fn load() -> Result<Nvml, String> {
            // SAFETY: System32-only search: the NVIDIA driver installs nvml.dll there.
            let lib = unsafe { Library::load_with_flags("nvml.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }.map_err(|e| format!("nvml.dll: {e}"))?;
            let init: FnVoid = sym(&lib, b"nvmlInit_v2\0").ok_or("nvml.dll has no nvmlInit_v2")?;
            let shutdown: FnVoid = sym(&lib, b"nvmlShutdown\0").ok_or("nvml.dll has no nvmlShutdown")?;
            let handle: FnHandle = sym(&lib, b"nvmlDeviceGetHandleByIndex_v2\0").ok_or("nvml.dll has no nvmlDeviceGetHandleByIndex_v2")?;
            // SAFETY: `nvmlReturn_t nvmlInit_v2(void)`; reference counted (sysinfo may hold one too).
            let rc = unsafe { init() };
            if rc != NVML_SUCCESS {
                return Err(format!("nvmlInit failed (code {rc})"));
            }
            Ok(Nvml {
                shutdown,
                handle,
                name: sym(&lib, b"nvmlDeviceGetName\0"),
                clock: sym(&lib, b"nvmlDeviceGetClockInfo\0"),
                pstate: sym(&lib, b"nvmlDeviceGetPerformanceState\0"),
                reasons: sym(&lib, b"nvmlDeviceGetCurrentClocksEventReasons\0").or_else(|| sym(&lib, b"nvmlDeviceGetCurrentClocksThrottleReasons\0")),
                util: sym(&lib, b"nvmlDeviceGetUtilizationRates\0"),
                decoder: sym(&lib, b"nvmlDeviceGetDecoderUtilization\0"),
                encoder: sym(&lib, b"nvmlDeviceGetEncoderUtilization\0"),
                temperature: sym(&lib, b"nvmlDeviceGetTemperature\0"),
                power: sym(&lib, b"nvmlDeviceGetPowerUsage\0"),
                processes: sym(&lib, b"nvmlDeviceGetProcessUtilization\0"),
                last_seen: Mutex::new(std::collections::BTreeMap::new()),
                _lib: lib,
            })
        }

        fn device(&self, index: u32) -> Option<Device> {
            let mut d: Device = std::ptr::null_mut();
            // SAFETY: valid out pointer; NVML validates the index.
            (unsafe { (self.handle)(index, &mut d) } == NVML_SUCCESS && !d.is_null()).then_some(d)
        }

        /// Fill `s` with device `index`'s state.
        pub fn fill(&self, index: u32, s: &mut GpuState) {
            let Some(d) = self.device(index) else {
                return;
            };
            // SAFETY (every call below): a valid device handle and valid out pointers, with the
            // export's C signature.
            unsafe {
                if let Some(f) = self.name {
                    let mut buf = [0 as c_char; 96];
                    if f(d, buf.as_mut_ptr(), buf.len() as u32) == NVML_SUCCESS {
                        let bytes: Vec<u8> = buf.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
                        s.device = Some(String::from_utf8_lossy(&bytes).trim().to_string());
                    }
                }
                if let Some(f) = self.clock {
                    let read = |kind: i32| {
                        let mut v = 0u32;
                        (f(d, kind, &mut v) == NVML_SUCCESS).then_some(v)
                    };
                    s.sm_clock_mhz = read(CLOCK_SM);
                    s.mem_clock_mhz = read(CLOCK_MEM);
                    s.graphics_clock_mhz = read(CLOCK_GRAPHICS);
                }
                if let Some(f) = self.pstate {
                    let mut v = 0i32;
                    if f(d, &mut v) == NVML_SUCCESS && (0..=15).contains(&v) {
                        s.pstate = Some(v as u32);
                    }
                }
                if let Some(f) = self.reasons {
                    let mut v = 0u64;
                    if f(d, &mut v) == NVML_SUCCESS {
                        s.clocks_event_reasons = Some(v);
                        s.clocks_event_reason_names = super::reason_names(v);
                    }
                }
                if let Some(f) = self.util {
                    let mut u = Utilization::default();
                    if f(d, &mut u) == NVML_SUCCESS {
                        s.gpu_util_pct = Some(u.gpu);
                        s.mem_util_pct = Some(u.memory);
                    }
                }
                let codec = |f: Option<FnCodec>| {
                    let f = f?;
                    let (mut v, mut period) = (0u32, 0u32);
                    (f(d, &mut v, &mut period) == NVML_SUCCESS).then_some(v)
                };
                s.decoder_util_pct = codec(self.decoder);
                s.encoder_util_pct = codec(self.encoder);
                if let Some(f) = self.temperature {
                    let mut v = 0u32;
                    if f(d, 0, &mut v) == NVML_SUCCESS {
                        s.temperature_c = Some(v);
                    }
                }
                if let Some(f) = self.power {
                    let mut mw = 0u32;
                    if f(d, &mut mw) == NVML_SUCCESS {
                        s.power_w = Some(f64::from(mw) / 1000.0);
                    }
                }
            }
            self.fill_processes(index, d, s);
        }

        /// Other processes' SM and decoder shares since the last query (where the driver keeps
        /// per-process samples; WDDM drivers may not).
        fn fill_processes(&self, index: u32, d: Device, s: &mut GpuState) {
            let Some(f) = self.processes else {
                return;
            };
            let mut seen = self.last_seen.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            let last = seen.get(&index).copied().unwrap_or(0);
            let mut count = 0u32;
            // SAFETY: a null buffer with a count asks for the needed size.
            let rc = unsafe { f(d, std::ptr::null_mut(), &mut count, last) };
            if rc != NVML_ERROR_INSUFFICIENT_SIZE && rc != NVML_SUCCESS {
                return;
            }
            let mut buf = vec![ProcessSample::default(); count.max(1) as usize + 8];
            let mut n = buf.len() as u32;
            // SAFETY: `n` entries of the documented sample struct.
            if unsafe { f(d, buf.as_mut_ptr(), &mut n, last) } != NVML_SUCCESS {
                return;
            }
            let me = std::process::id();
            let samples = &buf[..(n as usize).min(buf.len())];
            if let Some(newest) = samples.iter().map(|p| p.time_stamp).max() {
                seen.insert(index, newest);
            }
            let others: Vec<&ProcessSample> = samples.iter().filter(|p| p.pid != me && p.pid != 0).collect();
            let mut pids: Vec<u32> = others.iter().map(|p| p.pid).collect();
            pids.sort_unstable();
            pids.dedup();
            s.other_processes = Some(pids.len() as u32);
            s.other_gpu_util_pct = Some(others.iter().map(|p| f64::from(p.sm_util)).fold(0.0, f64::max));
            s.other_decoder_util_pct = Some(others.iter().map(|p| f64::from(p.dec_util)).fold(0.0, f64::max));
            if let Some(top) = others.iter().filter(|p| p.sm_util > 0).max_by_key(|p| p.sm_util) {
                let name = process_image_name(top.pid).unwrap_or_else(|| "?".into());
                s.other_top_process = Some(format!("{name} (pid {}, {} % SM)", top.pid, top.sm_util));
            }
        }
    }

    /// The image file name of process `pid` (`dwm.exe`), where this process may query it.
    pub(super) fn process_image_name(pid: u32) -> Option<String> {
        use windows::core::PWSTR;
        use windows::Win32::Foundation::CloseHandle;
        use windows::Win32::System::Threading::{OpenProcess, QueryFullProcessImageNameW, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION};
        // SAFETY: plain OpenProcess for a limited query; the handle is closed below.
        let h = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
        let mut buf = [0u16; 520];
        let mut len = buf.len() as u32;
        // SAFETY: a valid process handle and a buffer of `len` UTF-16 units.
        let ok = unsafe { QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, PWSTR(buf.as_mut_ptr()), &mut len) }.is_ok();
        // SAFETY: opened above, closed once.
        unsafe {
            let _ = CloseHandle(h);
        }
        if !ok {
            return None;
        }
        let path = String::from_utf16_lossy(&buf[..(len as usize).min(buf.len())]);
        path.rsplit(['\\', '/']).next().map(str::to_string).filter(|n| !n.is_empty())
    }

    impl Drop for Nvml {
        fn drop(&mut self) {
            // SAFETY: balances the successful nvmlInit_v2 in `load`.
            unsafe {
                (self.shutdown)();
            }
        }
    }

    // ---- threads and QoS ---------------------------------------------------------------------

    pub fn thread_state() -> ThreadState {
        let mut state = THREAD_POWER_THROTTLING_STATE {
            Version: THREAD_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: 0,
            StateMask: 0,
        };
        // SAFETY: the calling thread's pseudo-handle; pointer and size describe `state`.
        unsafe {
            let priority = GetThreadPriority(GetCurrentThread());
            let qos = GetThreadInformation(
                GetCurrentThread(),
                ThreadPowerThrottling,
                &mut state as *mut _ as *mut c_void,
                std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
            )
            .ok()
            .map(|_| QosState::new(state.ControlMask, state.StateMask));
            ThreadState {
                priority: (priority != 0x7fff_ffff).then_some(priority),
                qos,
            }
        }
    }

    pub fn process_qos() -> Option<QosState> {
        let mut state = PROCESS_POWER_THROTTLING_STATE {
            Version: PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            ControlMask: 0,
            StateMask: 0,
        };
        // SAFETY: the current process's pseudo-handle; pointer and size describe `state`.
        unsafe {
            GetProcessInformation(
                GetCurrentProcess(),
                ProcessPowerThrottling,
                &mut state as *mut _ as *mut c_void,
                std::mem::size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
            )
        }
        .ok()
        .map(|_| QosState::new(state.ControlMask, state.StateMask))
    }

    pub fn process_priority_class() -> Option<u32> {
        // SAFETY: the current process's pseudo-handle.
        let c = unsafe { GetPriorityClass(GetCurrentProcess()) };
        (c != 0).then_some(c)
    }

    // ---- power plan --------------------------------------------------------------------------

    #[link(name = "kernel32")]
    extern "system" {
        fn LocalFree(mem: *mut c_void) -> *mut c_void;
    }

    type FnActiveScheme = unsafe extern "system" fn(*mut c_void, *mut *mut GUID) -> u32;
    type FnFriendlyName = unsafe extern "system" fn(*mut c_void, *const GUID, *const GUID, *const GUID, *mut u8, *mut u32) -> u32;
    type FnOverlay = unsafe extern "system" fn(*mut GUID) -> u32;

    fn guid_text(g: &GUID) -> String {
        format!(
            "{:08x}-{:04x}-{:04x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
            g.data1, g.data2, g.data3, g.data4[0], g.data4[1], g.data4[2], g.data4[3], g.data4[4], g.data4[5], g.data4[6], g.data4[7]
        )
    }

    fn powrprof() -> Option<&'static Library> {
        static LIB: std::sync::OnceLock<Option<Library>> = std::sync::OnceLock::new();
        // SAFETY: a Windows system DLL from System32 only.
        LIB.get_or_init(|| unsafe { Library::load_with_flags("powrprof.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }.ok())
            .as_ref()
    }

    pub fn power_plan() -> PowerPlan {
        let mut plan = PowerPlan::default();
        let Some(lib) = powrprof() else {
            return plan;
        };
        let active: Option<FnActiveScheme> = sym(lib, b"PowerGetActiveScheme\0");
        let friendly: Option<FnFriendlyName> = sym(lib, b"PowerReadFriendlyName\0");
        let overlay: Option<FnOverlay> = sym(lib, b"PowerGetActualOverlayScheme\0");
        if let Some(active) = active {
            let mut p: *mut GUID = std::ptr::null_mut();
            // SAFETY: `DWORD PowerGetActiveScheme(HKEY, GUID**)`; NULL root key; the GUID is
            // allocated by the system and freed with LocalFree below.
            if unsafe { active(std::ptr::null_mut(), &mut p) } == 0 && !p.is_null() {
                // SAFETY: a valid GUID returned above.
                let guid = unsafe { *p };
                plan.guid = Some(guid_text(&guid));
                if let Some(friendly) = friendly {
                    let mut size = 0u32;
                    // SAFETY: a size query (NULL buffer), then a buffer of that size.
                    unsafe {
                        let _ = friendly(std::ptr::null_mut(), &guid, std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), &mut size);
                        if size > 0 && size < 4096 {
                            let mut buf = vec![0u8; size as usize];
                            if friendly(std::ptr::null_mut(), &guid, std::ptr::null(), std::ptr::null(), buf.as_mut_ptr(), &mut size) == 0 {
                                let wide: Vec<u16> = buf.as_chunks::<2>().0.iter().map(|c| u16::from_le_bytes([c[0], c[1]])).take_while(|&c| c != 0).collect();
                                plan.name = Some(String::from_utf16_lossy(&wide));
                            }
                        }
                    }
                }
                // SAFETY: memory allocated by PowerGetActiveScheme.
                unsafe {
                    LocalFree(p.cast());
                }
            }
        }
        if let Some(overlay) = overlay {
            let mut g = GUID::zeroed();
            // SAFETY: valid out pointer.
            if unsafe { overlay(&mut g) } == 0 {
                let text = guid_text(&g);
                plan.mode = Some(
                    match text.as_str() {
                        "961cc777-2547-4f9d-8174-7d86181b8a7a" => "best_power_efficiency",
                        "00000000-0000-0000-0000-000000000000" => "balanced",
                        "3af9b8d9-7c97-431d-ad78-34a8bfea439f" => "better_performance",
                        "ded574b5-45a0-4f42-8737-46345c09c238" => "best_performance",
                        _ => text.as_str(),
                    }
                    .to_string(),
                );
            }
        }
        plan
    }

    // ---- PDH ---------------------------------------------------------------------------------

    type FnOpen = unsafe extern "system" fn(*const u16, usize, *mut isize) -> u32;
    type FnAdd = unsafe extern "system" fn(isize, *const u16, usize, *mut isize) -> u32;
    type FnCollect = unsafe extern "system" fn(isize) -> u32;
    type FnFormat = unsafe extern "system" fn(isize, u32, *mut u32, *mut FmtValue) -> u32;
    type FnClose = unsafe extern "system" fn(isize) -> u32;

    /// `PDH_FMT_COUNTERVALUE` with the double member.
    #[repr(C)]
    #[derive(Default)]
    struct FmtValue {
        status: u32,
        value: f64,
    }

    const PDH_FMT_DOUBLE: u32 = 0x0000_0200;

    struct PdhFns {
        open: FnOpen,
        add: FnAdd,
        collect: FnCollect,
        format: FnFormat,
        close: FnClose,
    }

    fn pdh() -> Option<&'static PdhFns> {
        static FNS: std::sync::OnceLock<Option<(Library, PdhFns)>> = std::sync::OnceLock::new();
        FNS.get_or_init(|| {
            // SAFETY: a Windows system DLL from System32 only.
            let lib = unsafe { Library::load_with_flags("pdh.dll", LOAD_LIBRARY_SEARCH_SYSTEM32) }.ok()?;
            let fns = PdhFns {
                open: sym(&lib, b"PdhOpenQueryW\0")?,
                add: sym(&lib, b"PdhAddEnglishCounterW\0")?,
                collect: sym(&lib, b"PdhCollectQueryData\0")?,
                format: sym(&lib, b"PdhGetFormattedCounterValue\0")?,
                close: sym(&lib, b"PdhCloseQuery\0")?,
            };
            Some((lib, fns))
        })
        .as_ref()
        .map(|(_, f)| f)
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub struct Pdh {
        query: isize,
        perf: isize,
        freq: isize,
        utility: Option<isize>,
        started: std::time::Instant,
    }

    /// Queries closed so far (monotonic; tests check that an abandoned meter closes its query).
    #[cfg(test)]
    pub(super) static CLOSED_QUERIES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

    impl Drop for Pdh {
        fn drop(&mut self) {
            if let Some(f) = pdh() {
                // SAFETY: closes the query opened in `start`; `Drop` runs once per `Pdh`.
                unsafe { (f.close)(self.query) };
                #[cfg(test)]
                CLOSED_QUERIES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            }
        }
    }

    impl Pdh {
        pub fn start() -> Option<Pdh> {
            let f = pdh()?;
            let mut query = 0isize;
            // SAFETY: NULL data source (live data); valid out pointer.
            if unsafe { (f.open)(std::ptr::null(), 0, &mut query) } != 0 {
                return None;
            }
            let add = |path: &str| -> Option<isize> {
                let mut c = 0isize;
                let w = wide(path);
                // SAFETY: an open query and a NUL-terminated counter path.
                (unsafe { (f.add)(query, w.as_ptr(), 0, &mut c) } == 0).then_some(c)
            };
            let (Some(perf), Some(freq)) = (
                add(r"\Processor Information(_Total)\% Processor Performance"),
                add(r"\Processor Information(_Total)\Processor Frequency"),
            ) else {
                // SAFETY: the query opened above.
                unsafe { (f.close)(query) };
                return None;
            };
            let utility = add(r"\Processor Information(_Total)\% Processor Utility");
            // SAFETY: an open query.
            unsafe { (f.collect)(query) };
            Some(Pdh {
                query,
                perf,
                freq,
                utility,
                started: std::time::Instant::now(),
            })
        }

        pub fn finish(self) -> Option<super::CpuPerf> {
            // The query is closed when `self` drops, on every return path.
            let f = pdh()?;
            // Rate counters need two samples at least a few ms apart.
            if self.started.elapsed() < std::time::Duration::from_millis(20) {
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            // SAFETY: the query is open until `self` drops.
            unsafe { (f.collect)(self.query) };
            let read = |c: isize| -> Option<f64> {
                let mut ty = 0u32;
                let mut v = FmtValue::default();
                // SAFETY: a counter of the open query; valid out pointers.
                let ok = unsafe { (f.format)(c, PDH_FMT_DOUBLE, &mut ty, &mut v) } == 0 && v.status <= 1;
                (ok && v.value.is_finite()).then_some(v.value)
            };
            let perf = read(self.perf);
            let freq = read(self.freq);
            let utility = self.utility.and_then(read);
            let (perf, freq) = (perf?, freq?);
            Some(super::CpuPerf {
                perf_pct: perf,
                nominal_mhz: freq,
                effective_mhz: freq * perf / 100.0,
                utility_pct: utility,
            })
        }
    }
}

#[cfg(not(windows))]
mod imp {
    use super::{GpuState, PowerPlan, QosState, ThreadState};

    pub struct Nvml;

    impl Nvml {
        pub fn load() -> Result<Nvml, String> {
            Err("NVML is read on Windows only".into())
        }

        pub fn fill(&self, _index: u32, _s: &mut GpuState) {}
    }

    pub fn thread_state() -> ThreadState {
        ThreadState::default()
    }

    pub fn process_qos() -> Option<QosState> {
        None
    }

    pub fn process_priority_class() -> Option<u32> {
        None
    }

    pub fn power_plan() -> PowerPlan {
        PowerPlan::default()
    }

    pub struct Pdh;

    impl Pdh {
        pub fn start() -> Option<Pdh> {
            None
        }

        pub fn finish(self) -> Option<super::CpuPerf> {
            None
        }
    }
}

/// Merge `extra` into a JSON object (helpers for `details`).
pub(super) fn insert_json(details: &mut serde_json::Value, key: &str, value: impl Serialize) {
    if let Some(obj) = details.as_object_mut() {
        obj.insert(key.to_string(), serde_json::to_value(value).unwrap_or_default());
    }
}

/// A process-level host snapshot for the report (`attribution.host`).
pub fn host_snapshot() -> BTreeMap<&'static str, serde_json::Value> {
    let mut m = BTreeMap::new();
    let plan = power_plan();
    m.insert("power_plan", serde_json::to_value(&plan).unwrap_or_default());
    m.insert("process_qos", serde_json::to_value(process_qos()).unwrap_or_default());
    m.insert("process_priority_class", serde_json::to_value(imp::process_priority_class()).unwrap_or_default());
    m.insert("nvml", serde_json::Value::Bool(nvml_available()));
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reason_bits_have_names() {
        assert_eq!(reason_names(0), Vec::<&str>::new());
        assert_eq!(reason_names(0x1), vec!["gpu_idle"]);
        assert_eq!(reason_names(0x4 | 0x40), vec!["sw_power_cap", "hw_thermal_slowdown"]);
        assert_eq!(reason_names(THROTTLE_REASONS).len(), 5);
    }

    #[test]
    fn qos_states_read_as_opted_out_forced_or_system_managed() {
        assert_eq!(QosState::new(0, 0).execution_speed_throttled, None);
        assert_eq!(QosState::new(1, 0).execution_speed_throttled, Some(false));
        assert_eq!(QosState::new(1, 1).execution_speed_throttled, Some(true));
    }

    /// A sample never fails: without NVML (CI, AMD) the GPU fields stay `None`, NVML absent
    /// included; the host side reads on any Windows PC.
    #[test]
    fn sampling_is_best_effort() {
        let s = sample();
        assert!(s.at_unix_ms > 0);
        if !nvml_available() {
            assert!(s.sm_clock_mhz.is_none() && s.pstate.is_none() && s.device.is_none());
        } else if let Some(p) = s.pstate {
            assert!(p <= 15);
        }
        #[cfg(windows)]
        {
            assert!(s.process_qos.is_some(), "GetProcessInformation(ProcessPowerThrottling)");
            assert!(s.process_priority_class.is_some());
            let t = thread_state();
            assert!(t.priority.is_some());
        }
        let json = serde_json::to_value(&s).unwrap();
        assert!(json.get("sm_clock_mhz").is_some());
    }

    #[test]
    fn the_host_probe_measures_a_fixed_workload() {
        let (ms, ghz) = host_probe();
        assert!(ms > 0.0 && ms.is_finite(), "{ms}");
        assert!(ghz > 0.0, "{ghz}");
    }

    /// The summary and its warnings: throttling, a busy video decoder and a P-state change of a
    /// GPU run are flagged; an idle CPU run's P-state is not.
    #[test]
    fn summaries_flag_suspect_runs() {
        let a = GpuState {
            sm_clock_mhz: Some(2520),
            mem_clock_mhz: Some(10_501),
            pstate: Some(2),
            clocks_event_reasons: Some(0x1),
            decoder_util_pct: Some(0),
            ..GpuState::default()
        };
        let b = GpuState {
            sm_clock_mhz: Some(1_100),
            pstate: Some(8),
            clocks_event_reasons: Some(0x4),
            decoder_util_pct: Some(12),
            ..GpuState::default()
        };
        let s = summarize(&[&a, &b]);
        assert_eq!((s.sm_clock_min_mhz, s.sm_clock_max_mhz), (Some(1_100), Some(2520)));
        assert_eq!(s.sm_clock_median_mhz, Some(1_810.0));
        assert_eq!(s.pstates, vec![2, 8]);
        assert_eq!(s.reasons, vec!["gpu_idle", "sw_power_cap"]);
        let w = state_warnings(&s, Some("cuda"));
        assert_eq!(w.len(), 3, "{w:?}");
        assert!(w[0].contains("sw_power_cap") && w[1].contains("video decoder") && w[2].contains("P-state"));
        assert_eq!(state_warnings(&summarize(&[&a]), Some("cpu")), Vec::<String>::new());
        let cpu_run = state_warnings(&summarize(&[&a, &GpuState { pstate: Some(8), ..GpuState::default() }]), Some("cpu"));
        assert!(cpu_run.is_empty(), "{cpu_run:?}");
        // Another process's GPU use names the busiest process (below 5 % SM it is not evidence).
        let busy = |pct: f64, who: &str| GpuState {
            other_gpu_util_pct: Some(pct),
            other_top_process: Some(format!("{who} (pid 7, {pct} % SM)")),
            ..GpuState::default()
        };
        let s = summarize(&[&busy(40.0, "game.exe"), &busy(3.0, "dwm.exe"), &busy(12.0, "game.exe"), &busy(9.0, "obs64.exe")]);
        assert_eq!(s.other_top_processes, vec!["game.exe".to_string(), "obs64.exe".to_string()]);
        let w = state_warnings(&s, Some("cuda"));
        assert!(w.iter().any(|x| x.contains("up to 40 % SM: game.exe, obs64.exe")), "{w:?}");
    }

    /// The image name of a process the benchmark may query (this one).
    #[cfg(windows)]
    #[test]
    fn process_names_resolve() {
        let me = imp::process_image_name(std::process::id()).expect("this process");
        assert!(me.to_ascii_lowercase().ends_with(".exe") && !me.contains('\\'), "{me}");
        assert!(imp::process_image_name(u32::MAX - 3).is_none());
    }

    #[test]
    fn values_off_the_median_are_flagged() {
        let v = [Some(10.0), Some(10.4), None, Some(9.8), Some(12.0), Some(10.1)];
        assert_eq!(off_median(&v, 0.10), vec![4]);
        assert!(off_median(&[None, None], 0.1).is_empty());
        assert!(off_median(&[Some(0.0), Some(1.0), Some(0.0)], 0.1).is_empty(), "a zero median has no percentage");
    }

    /// The VRAM sampler takes at least one sample and stops promptly.
    #[test]
    fn the_vram_sampler_stops() {
        let s = VramSampler::start(Duration::from_millis(5));
        std::thread::sleep(Duration::from_millis(30));
        let t0 = Instant::now();
        let (peak, n) = s.stop();
        assert!(t0.elapsed() < Duration::from_millis(500));
        if process_vram_mb().is_some() {
            assert!(n >= 1 && peak.is_some());
        }
    }

    /// A probe that is dropped without `finish` (a run skipped before it ended, a panic) stops its
    /// sampler thread: no sample is taken after the drop returns.
    #[test]
    fn an_abandoned_sampler_thread_stops() {
        use std::sync::atomic::AtomicUsize;
        let taken = Arc::new(AtomicUsize::new(0));
        let counter = taken.clone();
        let series = SeriesThread::spawn(Duration::from_millis(5), move || {
            counter.fetch_add(1, Ordering::SeqCst);
            GpuState::default()
        })
        .expect("a thread");
        std::thread::sleep(Duration::from_millis(60));
        let t0 = Instant::now();
        drop(series);
        assert!(t0.elapsed() < Duration::from_millis(500), "the drop joins promptly");
        let at_drop = taken.load(Ordering::SeqCst);
        assert!(at_drop >= 1, "the thread sampled while it lived");
        std::thread::sleep(Duration::from_millis(60));
        assert_eq!(taken.load(Ordering::SeqCst), at_drop, "no sample after the drop");
        // `finish` hands the samples over and leaves nothing for the drop.
        let counter = Arc::new(AtomicUsize::new(0));
        let c2 = counter.clone();
        let series = SeriesThread::spawn(Duration::from_millis(5), move || {
            c2.fetch_add(1, Ordering::SeqCst);
            GpuState::default()
        })
        .expect("a thread");
        std::thread::sleep(Duration::from_millis(40));
        let samples = series.finish();
        assert_eq!(samples.len(), counter.load(Ordering::SeqCst));
        assert!(!samples.is_empty());
    }

    /// An abandoned probe closes its PDH query too (the count of closed queries only grows, so
    /// parallel tests cannot break the check).
    #[cfg(windows)]
    #[test]
    fn an_abandoned_cpu_meter_closes_its_query() {
        use std::sync::atomic::Ordering::SeqCst;
        let before = imp::CLOSED_QUERIES.load(SeqCst);
        let Some(meter) = CpuPerfMeter::start() else {
            return; // no PDH on this PC
        };
        drop(meter);
        assert!(imp::CLOSED_QUERIES.load(SeqCst) > before, "dropping the meter closed its query");
        let before = imp::CLOSED_QUERIES.load(SeqCst);
        let meter = CpuPerfMeter::start().expect("PDH worked a moment ago");
        std::thread::sleep(Duration::from_millis(25));
        let _ = meter.finish();
        assert!(imp::CLOSED_QUERIES.load(SeqCst) > before, "finish closes it as well");
    }

    /// The speed index's session is released with the spec, and releasing again is harmless. A CPU
    /// session stands in for the GPU one (no GPU is needed to test the cache).
    #[test]
    fn the_speed_index_session_is_released() {
        // A private slot: the process-wide cache is released by every spec that finishes in a
        // parallel test.
        let slot = Mutex::new(Some(cpu_speed_session()));
        assert!(slot.lock().unwrap().is_some());
        clear_speed_cache(&slot);
        assert!(slot.lock().unwrap().is_none(), "the session, its device and its tensor are dropped");
        clear_speed_cache(&slot);
        // The public entry point empties the process-wide one (and is harmless when it is empty).
        let _serial = SPEED_CACHE_TEST_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
        seed_cpu_speed_session_for_tests();
        release_speed_index();
        assert!(!speed_index_cached());
        release_speed_index();
    }

    /// The run probe's boundaries carry the host probe; its CPU window reads where PDH exists.
    #[test]
    fn run_probes_have_start_and_end() {
        let p = RunStateProbe::start(StateContext {
            engine_ep: Some("cpu".into()),
            device: 0,
            period_ms: 20,
            speed_index: true,
        });
        std::thread::sleep(Duration::from_millis(60));
        let r = p.finish();
        assert!(r.start.host_probe_ms.is_some() && r.end.host_probe_ms.is_some());
        assert!(r.start.speed_index_ms.is_none(), "no speed index for a CPU engine");
        let m = r.metrics();
        assert!(m.iter().any(|(k, _)| *k == "state_host_probe_ms"), "{m:?}");
        #[cfg(windows)]
        if let Some(c) = r.cpu {
            assert!(c.nominal_mhz > 0.0 && c.effective_mhz > 0.0, "{c:?}");
        }
    }
}
