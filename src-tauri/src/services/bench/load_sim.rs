//! Contention for paced runs: a GPU load generator and CPU burner threads, each at a duty cycle.
//!
//! - GPU: a chain of `depth` square MatMuls (`size` x `size`, fp32) built in memory and run through
//!   `ort_session::build_session` (the one session factory: EP ladder, `error_on_failure`,
//!   spinning off) on a normal-priority thread, like a game's compute on the same GPU. When no GPU
//!   EP loads, the ladder's CPU fallback is reported (never hidden) and the graph shrinks to at most
//!   256 x 256 so it stays a load rather than a multi-second stall.
//! - CPU: `threads` busy threads, each spinning `duty` of every `period_ms`.
//!
//! Duty is controlled on accumulated busy time (sleep until `busy / duty` has elapsed), so it is
//! exact on average even when one GPU run is longer than the period. Every load thread adds its own
//! CPU time to a shared [`LoadCpu`] tally as it goes (per GPU run / burner period), so the paced run
//! can subtract the load's CPU over its own measurement window from the process CPU it attributes
//! to the voice changer.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use ort::session::Session;
use ort::value::Tensor;
use serde::Serialize;

use crate::services::neural_audio::ort_session::{build_session, normalize_ep, ModelSource, VcSessionOptions};

use super::engine::spin_for;
use super::metrics::{thread_cpu_time, Summary};
use super::signals::SplitMix64;
use super::spec::{CpuLoadSpec, GpuLoadSpec, LoadSpec};

/// Largest matrix used when the GPU load fell back to the CPU.
const CPU_FALLBACK_MAX_SIZE: usize = 256;

// ---- minimal ONNX protobuf writer (field numbers from onnx.proto) ----------------------------

fn varint(out: &mut Vec<u8>, mut v: u64) {
    while v >= 0x80 {
        out.push((v as u8) | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn field_varint(out: &mut Vec<u8>, field: u32, v: u64) {
    varint(out, u64::from(field) << 3);
    varint(out, v);
}

fn field_bytes(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    varint(out, (u64::from(field) << 3) | 2);
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn value_info(name: &str, dims: &[u64]) -> Vec<u8> {
    let mut shape = Vec::new();
    for &d in dims {
        let mut dim = Vec::new();
        field_varint(&mut dim, 1, d); // Dimension.dim_value
        field_bytes(&mut shape, 1, &dim); // TensorShapeProto.dim
    }
    let mut tensor = Vec::new();
    field_varint(&mut tensor, 1, 1); // elem_type = FLOAT
    field_bytes(&mut tensor, 2, &shape);
    let mut ty = Vec::new();
    field_bytes(&mut ty, 1, &tensor); // TypeProto.tensor_type
    let mut v = Vec::new();
    field_bytes(&mut v, 1, name.as_bytes());
    field_bytes(&mut v, 2, &ty);
    v
}

/// `Y = X · W · W · … · W` (`depth` MatMuls), X and Y `[size, size]`, W a fixed random matrix
/// scaled so activations keep their magnitude (entries uniform with variance `1 / size`).
pub fn matmul_chain_model(size: usize, depth: usize, seed: u64) -> Vec<u8> {
    let n = size as u64;
    let a = (3.0 / size as f64).sqrt();
    let mut rng = SplitMix64::new(seed);
    let raw: Vec<u8> = (0..size * size)
        .flat_map(|_| (rng.range(-a, a) as f32).to_le_bytes())
        .collect();
    let mut weight = Vec::new();
    let mut dims = Vec::new();
    varint(&mut dims, n);
    varint(&mut dims, n);
    field_bytes(&mut weight, 1, &dims); // dims (packed)
    field_varint(&mut weight, 2, 1); // data_type = FLOAT
    field_bytes(&mut weight, 8, b"W"); // name
    field_bytes(&mut weight, 9, &raw); // raw_data

    let mut graph = Vec::new();
    let depth = depth.max(1);
    for i in 0..depth {
        let input = if i == 0 { "X".to_string() } else { format!("h{}", i - 1) };
        let output = if i + 1 == depth { "Y".to_string() } else { format!("h{i}") };
        let mut node = Vec::new();
        field_bytes(&mut node, 1, input.as_bytes());
        field_bytes(&mut node, 1, b"W");
        field_bytes(&mut node, 2, output.as_bytes());
        field_bytes(&mut node, 3, format!("mm{i}").as_bytes());
        field_bytes(&mut node, 4, b"MatMul");
        field_bytes(&mut graph, 1, &node);
    }
    field_bytes(&mut graph, 2, b"sigil_bench_gpu_load");
    field_bytes(&mut graph, 5, &weight);
    field_bytes(&mut graph, 11, &value_info("X", &[n, n]));
    field_bytes(&mut graph, 12, &value_info("Y", &[n, n]));

    let mut opset = Vec::new();
    field_varint(&mut opset, 2, 13);
    let mut model = Vec::new();
    field_varint(&mut model, 1, 7); // ir_version
    field_bytes(&mut model, 2, b"sigil-bench");
    field_bytes(&mut model, 7, &graph);
    field_bytes(&mut model, 8, &opset);
    model
}

// ---- CPU tally --------------------------------------------------------------------------------

/// Running total of the load threads' CPU time, readable while they run.
#[derive(Debug, Clone, Default)]
pub struct LoadCpu(Arc<AtomicU64>);

impl LoadCpu {
    /// CPU time (ms) the load threads used so far (current to within one GPU run / burner period).
    pub fn ms(&self) -> f64 {
        self.0.load(Ordering::Relaxed) as f64 / 1e6
    }
}

/// One load thread's feed into the tally.
struct CpuFeed {
    last: Option<Duration>,
    tally: LoadCpu,
    total: Duration,
}

impl CpuFeed {
    fn start(tally: LoadCpu) -> Self {
        Self {
            last: thread_cpu_time(),
            tally,
            total: Duration::ZERO,
        }
    }

    fn update(&mut self) {
        if let (Some(last), Some(now)) = (self.last, thread_cpu_time()) {
            let d = now.saturating_sub(last);
            self.total += d;
            self.tally.0.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
            self.last = Some(now);
        }
    }
}

// ---- GPU load ---------------------------------------------------------------------------------

/// What the GPU load generator did.
#[derive(Debug, Clone, Serialize)]
pub struct GpuLoadReport {
    pub execution_provider: String,
    pub execution_provider_active: String,
    pub reason: Option<String>,
    pub size: usize,
    pub depth: usize,
    pub duty_target: f64,
    /// Busy fraction of the load thread's wall time.
    pub duty: f64,
    pub runs: u64,
    pub run_ms: Option<Summary>,
    /// Achieved throughput while busy.
    pub gflops: f64,
    pub cpu_ms: f64,
    pub error: Option<String>,
}

struct GpuSession {
    session: Session,
    active: String,
    reason: Option<String>,
    size: usize,
}

fn open_gpu_session(spec: &GpuLoadSpec) -> Result<GpuSession, String> {
    let requested = normalize_ep(&spec.execution_provider);
    let opts = VcSessionOptions {
        cuda_device_index: spec.device_index,
        // The calling thread only: its CPU time is then the whole load's CPU time.
        intra_threads: Some(1),
        ..VcSessionOptions::default()
    };
    let bytes = matmul_chain_model(spec.size, spec.depth, 0x10ad);
    let (session, outcome) = build_session(Some("bench.load"), requested, &opts, ModelSource::Memory(&bytes))?;
    if outcome.active == "cpu" && requested != "cpu" && spec.size > CPU_FALLBACK_MAX_SIZE {
        let small = CPU_FALLBACK_MAX_SIZE;
        let bytes = matmul_chain_model(small, spec.depth, 0x10ad);
        let (session, _) = build_session(None, "cpu", &opts, ModelSource::Memory(&bytes))?;
        let reason = format!(
            "{}; on the CPU the matrices shrink to {small} x {small}",
            outcome.reason.unwrap_or_else(|| "no GPU execution provider".into())
        );
        return Ok(GpuSession {
            session,
            active: "cpu".into(),
            reason: Some(reason),
            size: small,
        });
    }
    Ok(GpuSession {
        session,
        active: outcome.active.clone(),
        reason: outcome.reason.clone(),
        size: spec.size,
    })
}

fn gpu_thread(
    spec: GpuLoadSpec,
    stop: Arc<AtomicBool>,
    ready: mpsc::Sender<Result<String, String>>,
    tally: LoadCpu,
) -> GpuLoadReport {
    let mut report = GpuLoadReport {
        execution_provider: normalize_ep(&spec.execution_provider).to_string(),
        execution_provider_active: String::new(),
        reason: None,
        size: spec.size,
        depth: spec.depth,
        duty_target: spec.duty,
        duty: 0.0,
        runs: 0,
        run_ms: None,
        gflops: 0.0,
        cpu_ms: 0.0,
        error: None,
    };
    let mut gs = match open_gpu_session(&spec) {
        Ok(s) => s,
        Err(e) => {
            let _ = ready.send(Err(format!("GPU load: {e}")));
            report.error = Some(e);
            return report;
        }
    };
    report.execution_provider_active = gs.active.clone();
    report.reason = gs.reason.clone();
    report.size = gs.size;
    let n = gs.size;
    let mut rng = SplitMix64::new(0x10ad_0001);
    let data: Vec<f32> = (0..n * n).map(|_| rng.range(-1.0, 1.0) as f32).collect();
    let x = match Tensor::from_array(([n, n], data)) {
        Ok(x) => x,
        Err(e) => {
            let _ = ready.send(Err(format!("GPU load input: {e}")));
            report.error = Some(e.to_string());
            return report;
        }
    };
    // One run before "ready" so kernel selection / allocation is not part of the measured load.
    if let Err(e) = gs.session.run(ort::inputs!["X" => &x]) {
        let msg = format!("GPU load run on {}: {e}", gs.active);
        let _ = ready.send(Err(msg.clone()));
        report.error = Some(msg);
        return report;
    }
    let _ = ready.send(Ok(gs.active.clone()));

    let mut feed = CpuFeed::start(tally);
    let start = Instant::now();
    let period = Duration::from_secs_f64(spec.period_ms / 1000.0);
    let mut busy = Duration::ZERO;
    let mut run_ms = Vec::new();
    while !stop.load(Ordering::Relaxed) {
        let r0 = Instant::now();
        if let Err(e) = gs.session.run(ort::inputs!["X" => &x]) {
            report.error = Some(format!("GPU load run: {e}"));
            break;
        }
        let r = r0.elapsed();
        busy += r;
        run_ms.push(r.as_secs_f64() * 1000.0);
        feed.update();
        let target = busy.div_f64(spec.duty);
        while !stop.load(Ordering::Relaxed) {
            let elapsed = start.elapsed();
            if elapsed >= target {
                break;
            }
            std::thread::sleep((target - elapsed).min(period));
        }
    }
    let wall = start.elapsed();
    report.runs = run_ms.len() as u64;
    report.run_ms = Summary::of(&run_ms);
    report.duty = if wall.is_zero() { 0.0 } else { busy.as_secs_f64() / wall.as_secs_f64() };
    let flops = 2.0 * (n as f64).powi(3) * spec.depth as f64 * run_ms.len() as f64;
    report.gflops = if busy.is_zero() { 0.0 } else { flops / busy.as_secs_f64() / 1e9 };
    feed.update();
    report.cpu_ms = feed.total.as_secs_f64() * 1000.0;
    report
}

// ---- CPU burners ------------------------------------------------------------------------------

/// What the CPU burners did.
#[derive(Debug, Clone, Serialize)]
pub struct CpuLoadReport {
    pub threads: usize,
    pub duty_target: f64,
    /// Mean over threads of busy (spinning) wall time / wall time.
    pub duty: f64,
    /// Mean over threads of CPU time / wall time (below `duty` when the OS preempted the
    /// burners, e.g. under heavier load).
    pub cpu_share: f64,
    pub cpu_ms: f64,
}

/// One burner thread's result: (CPU ms, busy / wall, CPU / wall).
type BurnerResult = (f64, f64, f64);

fn burner(spec: CpuLoadSpec, stop: Arc<AtomicBool>, tally: LoadCpu) -> BurnerResult {
    let mut feed = CpuFeed::start(tally);
    let start = Instant::now();
    let period = Duration::from_secs_f64(spec.period_ms / 1000.0);
    let busy_per_period = period.mul_f64(spec.duty);
    let mut busy = Duration::ZERO;
    while !stop.load(Ordering::Relaxed) {
        let p0 = Instant::now();
        spin_for(busy_per_period);
        busy += p0.elapsed();
        feed.update();
        // Sleep until `busy / duty` has elapsed: sleep overshoot is paid back next period.
        let target = busy.div_f64(spec.duty);
        let elapsed = start.elapsed();
        if target > elapsed {
            std::thread::sleep(target - elapsed);
        }
    }
    let wall = start.elapsed().as_secs_f64();
    feed.update();
    let cpu = feed.total.as_secs_f64();
    if wall > 0.0 {
        (cpu * 1000.0, busy.as_secs_f64() / wall, cpu / wall)
    } else {
        (cpu * 1000.0, 0.0, 0.0)
    }
}

// ---- Handle -----------------------------------------------------------------------------------

/// Everything the load generators did during a run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct LoadReport {
    pub gpu: Option<GpuLoadReport>,
    pub cpu: Option<CpuLoadReport>,
}

impl LoadReport {
    /// CPU time the load threads used over their whole run.
    #[cfg(test)]
    pub fn cpu_ms(&self) -> f64 {
        self.gpu.as_ref().map_or(0.0, |g| g.cpu_ms) + self.cpu.as_ref().map_or(0.0, |c| c.cpu_ms)
    }
}

/// Running load generators; [`LoadHandle::stop`] joins them.
pub struct LoadHandle {
    stop: Arc<AtomicBool>,
    gpu: Option<JoinHandle<GpuLoadReport>>,
    cpu: Vec<JoinHandle<BurnerResult>>,
    cpu_spec: Option<CpuLoadSpec>,
    tally: LoadCpu,
}

impl LoadHandle {
    /// Start the generators of `load`; returns once the GPU load has its session and first run
    /// done (or failed: then nothing keeps running and the error is returned).
    pub fn start(load: &LoadSpec) -> Result<LoadHandle, String> {
        let stop = Arc::new(AtomicBool::new(false));
        let tally = LoadCpu::default();
        let gpu = match &load.gpu {
            None => None,
            Some(spec) => {
                let (tx, rx) = mpsc::channel();
                let (spec, stop2, tally2) = (spec.clone(), stop.clone(), tally.clone());
                let handle = std::thread::Builder::new()
                    .name("sigil-bench-gpu-load".into())
                    .spawn(move || gpu_thread(spec, stop2, tx, tally2))
                    .map_err(|e| format!("cannot start the GPU load thread: {e}"))?;
                match rx.recv() {
                    Ok(Ok(_active)) => Some(handle),
                    Ok(Err(e)) => {
                        let _ = handle.join();
                        return Err(e);
                    }
                    Err(_) => {
                        let _ = handle.join();
                        return Err("the GPU load thread exited during start-up".into());
                    }
                }
            }
        };
        let mut cpu = Vec::new();
        if let Some(spec) = &load.cpu {
            for i in 0..spec.threads {
                let (spec, stop2, tally2) = (spec.clone(), stop.clone(), tally.clone());
                match std::thread::Builder::new()
                    .name(format!("sigil-bench-cpu-load-{i}"))
                    .spawn(move || burner(spec, stop2, tally2))
                {
                    Ok(h) => cpu.push(h),
                    Err(e) => {
                        stop.store(true, Ordering::Relaxed);
                        return Err(format!("cannot start CPU load thread {i}: {e}"));
                    }
                }
            }
        }
        Ok(LoadHandle {
            stop,
            gpu,
            cpu,
            cpu_spec: load.cpu.clone(),
            tally,
        })
    }

    /// The load threads' running CPU tally.
    pub fn cpu_tally(&self) -> LoadCpu {
        self.tally.clone()
    }

    #[cfg(test)]
    pub fn is_idle(&self) -> bool {
        self.gpu.is_none() && self.cpu.is_empty()
    }

    /// Stop and join every generator.
    pub fn stop(self) -> LoadReport {
        self.stop.store(true, Ordering::Relaxed);
        let gpu = self.gpu.and_then(|h| h.join().ok());
        let results: Vec<BurnerResult> = self.cpu.into_iter().filter_map(|h| h.join().ok()).collect();
        let mean = |f: fn(&BurnerResult) -> f64| {
            if results.is_empty() {
                0.0
            } else {
                results.iter().map(f).sum::<f64>() / results.len() as f64
            }
        };
        let cpu = self.cpu_spec.map(|spec| CpuLoadReport {
            threads: spec.threads,
            duty_target: spec.duty,
            duty: mean(|r| r.1),
            cpu_share: mean(|r| r.2),
            cpu_ms: results.iter().map(|r| r.0).sum(),
        });
        LoadReport { gpu, cpu }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_chain_runs_on_the_cpu_and_keeps_magnitude() {
        let bytes = matmul_chain_model(32, 3, 1);
        let (mut session, outcome) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::Memory(&bytes))
            .expect("the load graph builds on the CPU EP");
        assert_eq!(outcome.active, "cpu");
        let x = Tensor::from_array(([32usize, 32], vec![0.5f32; 32 * 32])).unwrap();
        let out = session.run(ort::inputs!["X" => &x]).unwrap();
        let (shape, y) = out["Y"].try_extract_tensor::<f32>().unwrap();
        assert_eq!(shape.iter().copied().collect::<Vec<i64>>(), vec![32, 32]);
        assert!(y.iter().all(|v| v.is_finite()));
        let rms = (y.iter().map(|v| v * v).sum::<f32>() / y.len() as f32).sqrt();
        assert!(rms > 1e-3 && rms < 10.0, "activations stay in range: {rms}");
    }

    #[test]
    fn load_generators_hold_their_duty_cycle() {
        let spec: LoadSpec = serde_json::from_value(serde_json::json!({
            "gpu": {"duty": 0.5, "period_ms": 10, "execution_provider": "cpu", "size": 64, "depth": 2},
            "cpu": {"threads": 1, "duty": 0.5, "period_ms": 10}
        }))
        .unwrap();
        let h = LoadHandle::start(&spec).unwrap();
        assert!(!h.is_idle());
        let tally = h.cpu_tally();
        std::thread::sleep(Duration::from_millis(400));
        let running = tally.ms();
        let r = h.stop();
        // The tally is live, and ends at the threads' own totals.
        assert!(running > 0.0 && running <= r.cpu_ms() + 1e-9, "{running} vs {}", r.cpu_ms());
        assert!((tally.ms() - r.cpu_ms()).abs() < 1e-6, "{} vs {}", tally.ms(), r.cpu_ms());
        let g = r.gpu.expect("gpu report");
        assert_eq!(g.execution_provider_active, "cpu");
        assert!(g.error.is_none(), "{:?}", g.error);
        assert!(g.runs > 5, "{}", g.runs);
        assert!(g.duty > 0.3 && g.duty < 0.7, "gpu duty {}", g.duty);
        let c = r.cpu.expect("cpu report");
        assert_eq!(c.threads, 1);
        assert!(c.duty > 0.4 && c.duty < 0.65, "cpu duty {}", c.duty);
        // CPU time is only what the OS granted (quantum accounting, other tests running).
        assert!(c.cpu_ms > 0.0 && c.cpu_share <= c.duty + 0.1, "{c:?}");
    }

    #[test]
    fn an_empty_load_spec_starts_nothing() {
        let h = LoadHandle::start(&LoadSpec::default()).unwrap();
        assert!(h.is_idle());
        let r = h.stop();
        assert!(r.gpu.is_none() && r.cpu.is_none());
        assert_eq!(r.cpu_ms(), 0.0);
    }
}
