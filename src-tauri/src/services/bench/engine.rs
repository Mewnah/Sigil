//! Engine factory for benchmark runs, and the identity probe that measures the stream harness's
//! algorithmic latency.
//!
//! Every offline / paced run drives the production [`VcStreamHarness`] with a model built here:
//! - `stub`: a built-in stand-in (identity or silence) with a configurable, seeded compute cost
//!   that either really sleeps / spins or, on the virtual clock, is only accounted.
//! - `rvc`: a real slot through the production [`RvcModel`] loader, with the voice changer's rules
//!   for the preset EP, the GPU-only F0 fallback, ONNX Runtime thread policy and model auto paths
//!   (mirrored from `voice_changer/params.rs` and `worker.rs`).
//!
//! The model is loaded once per spec and shared by the runs ([`Engine::handle`]); each handle
//! times every conversion into its run's [`ModelTimes`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::Serialize;

use crate::services::diagnostics;
use crate::services::neural_audio::ort_session::{normalize_ep, VcSessionOptions};
use crate::services::neural_audio::vc::bases::{resolve_fcpe, resolve_hubert, resolve_rmvpe};
use crate::services::neural_audio::vc::traits::ConversionCtx;
use crate::services::neural_audio::{
    DioPitchExtractor, FcpePitchExtractor, PitchExtractor, RmvpePitchExtractor, RvcModel, RvcPresetProfile,
    VcStreamHarness, VoiceConversionModel,
};
use crate::services::rt_thread::WorkerPriority;

use super::e2e_loopback::estimate_delay;
use super::signals::{channel, to_interleaved, BurstTrain, SplitMix64};
use super::spec::{ComputeKind, EngineSpec, RvcSpec, StreamSpec, StubBehavior, StubSpec};

/// Model input rate of every voice-changer engine (HuBERT / ContentVec domain).
pub const MODEL_INPUT_RATE: u32 = 16_000;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// What was loaded, and where it runs.
#[derive(Debug, Clone, Serialize)]
pub struct EngineInfo {
    pub kind: String,
    pub description: String,
    pub input_rate: u32,
    pub output_rate: u32,
    pub min_context_samples: usize,
    /// EP asked for (after the preset default), e.g. `cuda`; `None` for engines without ORT.
    pub execution_provider: Option<String>,
    /// EP the sessions actually run on (a GPU request can fall back).
    pub execution_provider_active: Option<String>,
    pub execution_provider_reason: Option<String>,
    pub pitch_extractor: Option<String>,
    pub load_ms: f64,
    pub warnings: Vec<String>,
}

impl EngineInfo {
    /// Whether the engine runs on a GPU execution provider (the worker priority `auto` rule).
    pub fn gpu(&self) -> bool {
        matches!(self.execution_provider_active.as_deref(), Some("cuda") | Some("dml"))
    }
}

/// Virtual-clock support: a stub in simulate mode reports its compute cost here instead of
/// spending it.
#[derive(Clone, Default)]
pub struct SimControl {
    simulate: Arc<AtomicBool>,
    cost_ns: Arc<AtomicU64>,
}

/// Per-run model timings, filled by [`Engine::handle`] handles.
#[derive(Debug, Default)]
pub struct ModelTimes {
    pub convert_ms: Vec<f64>,
    pub warmup_ms: f64,
    pub errors: u64,
    pub first_error: Option<String>,
}

/// A loaded engine, shared by a spec's runs.
pub struct Engine {
    model: Arc<Mutex<Box<dyn VoiceConversionModel>>>,
    pub info: EngineInfo,
    sim: SimControl,
}

impl Engine {
    /// A model for one harness: forwards to the shared engine and times every call into `times`.
    pub fn handle(&self, times: Arc<Mutex<ModelTimes>>) -> Box<dyn VoiceConversionModel> {
        Box::new(ModelHandle {
            model: self.model.clone(),
            times,
            in_sr: self.info.input_rate,
            out_sr: self.info.output_rate,
            min_ctx: self.info.min_context_samples,
        })
    }

    /// Account simulated compute instead of spending it (virtual clock); stubs only.
    pub fn set_simulated(&self, on: bool) {
        self.sim.simulate.store(on, Ordering::SeqCst);
        self.sim.cost_ns.store(0, Ordering::SeqCst);
    }

    /// Simulated compute accounted since the last call.
    pub fn take_simulated_cost(&self) -> Duration {
        Duration::from_nanos(self.sim.cost_ns.swap(0, Ordering::SeqCst))
    }
}

struct ModelHandle {
    model: Arc<Mutex<Box<dyn VoiceConversionModel>>>,
    times: Arc<Mutex<ModelTimes>>,
    in_sr: u32,
    out_sr: u32,
    min_ctx: usize,
}

impl VoiceConversionModel for ModelHandle {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let t0 = Instant::now();
        let result = lock(&self.model).convert(ctx);
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let mut times = lock(&self.times);
        times.convert_ms.push(ms);
        if let Err(e) = &result {
            times.errors += 1;
            times.first_error.get_or_insert_with(|| e.clone());
        }
        result
    }

    fn input_sample_rate(&self) -> u32 {
        self.in_sr
    }

    fn output_sample_rate(&self) -> u32 {
        self.out_sr
    }

    fn min_context_samples(&self) -> usize {
        self.min_ctx
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        let t0 = Instant::now();
        let r = lock(&self.model).warmup_at(ctx);
        lock(&self.times).warmup_ms += t0.elapsed().as_secs_f64() * 1000.0;
        r
    }
}

/// Load the spec's engine. `priority` is the worker policy the run applies (it also sets ONNX
/// Runtime's pool-thread priority, as the voice changer does).
pub fn build_engine(spec: &EngineSpec, priority: WorkerPriority, seed: u64) -> Result<Engine, String> {
    let sim = SimControl::default();
    let started = Instant::now();
    let (model, mut info): (Box<dyn VoiceConversionModel>, EngineInfo) = match spec {
        EngineSpec::Stub(s) => (
            Box::new(StubModel::new(s.clone(), seed, sim.clone())),
            EngineInfo {
                kind: "stub".into(),
                description: format!(
                    "stub {:?}, {} Hz out, {} ms {:?} compute (+-{} ms)",
                    s.behavior, s.output_rate, s.compute_ms, s.compute, s.compute_jitter_ms
                ),
                input_rate: MODEL_INPUT_RATE,
                output_rate: s.output_rate,
                min_context_samples: s.min_context_samples,
                execution_provider: None,
                execution_provider_active: None,
                execution_provider_reason: None,
                pitch_extractor: None,
                load_ms: 0.0,
                warnings: Vec::new(),
            },
        ),
        EngineSpec::Rvc(r) => build_rvc(r, priority)?,
    };
    info.load_ms = started.elapsed().as_secs_f64() * 1000.0;
    Ok(Engine {
        model: Arc::new(Mutex::new(model)),
        info,
        sim,
    })
}

// ---------------------------------------------------------------------------------------------
// Stub model
// ---------------------------------------------------------------------------------------------

/// Built-in stand-in model (see [`StubSpec`]).
pub struct StubModel {
    spec: StubSpec,
    rng: SplitMix64,
    calls: u64,
    sim: SimControl,
}

impl StubModel {
    pub fn new(spec: StubSpec, seed: u64, sim: SimControl) -> Self {
        Self {
            spec,
            rng: SplitMix64::new(seed ^ 0x57ab_0001),
            calls: 0,
            sim,
        }
    }

    /// An identity model with no compute cost (the latency probe).
    fn identity(output_rate: u32, min_context_samples: usize) -> Self {
        Self::new(
            StubSpec {
                behavior: StubBehavior::Passthrough,
                output_rate,
                min_context_samples,
                compute_ms: 0.0,
                compute_jitter_ms: 0.0,
                compute: ComputeKind::Sleep,
                fail_every: 0,
            },
            0,
            SimControl::default(),
        )
    }

    fn spend(&mut self) {
        let jitter = self.spec.compute_jitter_ms * (2.0 * self.rng.unit() - 1.0);
        let ms = (self.spec.compute_ms + jitter).max(0.0);
        if ms <= 0.0 {
            return;
        }
        let d = Duration::from_secs_f64(ms / 1000.0);
        if self.sim.simulate.load(Ordering::Relaxed) {
            self.sim.cost_ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
            return;
        }
        match self.spec.compute {
            ComputeKind::Sleep => std::thread::sleep(d),
            ComputeKind::Spin => spin_for(d),
        }
    }
}

/// Busy CPU work for `d` on the calling thread.
pub fn spin_for(d: Duration) {
    let start = Instant::now();
    let mut x = 1.0f64;
    while start.elapsed() < d {
        for _ in 0..64 {
            x = std::hint::black_box(x * 1.000_000_1 + 1e-9).sqrt() + 0.5;
        }
    }
    std::hint::black_box(x);
}

/// The trailing `n` samples of `x` (16 kHz) as `round(n * out_sr / 16k)` samples at `out_sr`,
/// aligned by the `VoiceConversionModel` contract: the last output sample is the last input
/// sample. Catmull-Rom interpolation (linear extrapolation past the edges).
pub fn identity_tail(x: &[f32], n: usize, out_sr: u32) -> Vec<f32> {
    let n = n.min(x.len());
    let len = ((n as u64 * u64::from(out_sr) + u64::from(MODEL_INPUT_RATE / 2)) / u64::from(MODEL_INPUT_RATE)) as usize;
    if out_sr == MODEL_INPUT_RATE {
        return x[x.len() - n..].to_vec();
    }
    if len == 0 || x.is_empty() {
        return Vec::new();
    }
    let step = f64::from(MODEL_INPUT_RATE) / f64::from(out_sr);
    let last = (x.len() - 1) as f64;
    let n_x = x.len() as isize;
    let at = |i: isize| -> f32 {
        if n_x < 2 {
            x[0]
        } else if i < 0 {
            x[0] + i as f32 * (x[1] - x[0])
        } else if i >= n_x {
            x[(n_x - 1) as usize] + (i - n_x + 1) as f32 * (x[(n_x - 1) as usize] - x[(n_x - 2) as usize])
        } else {
            x[i as usize]
        }
    };
    (0..len)
        .map(|k| {
            let pos = last - (len - 1 - k) as f64 * step;
            let i = pos.floor() as isize;
            let t = (pos - i as f64) as f32;
            let (p0, p1, p2, p3) = (at(i - 1), at(i), at(i + 1), at(i + 2));
            p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
        })
        .collect()
}

impl VoiceConversionModel for StubModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        self.calls += 1;
        self.spend();
        if self.spec.fail_every > 0 && self.calls.is_multiple_of(u64::from(self.spec.fail_every)) {
            return Err(format!("stub failure (call {}, fail_every {})", self.calls, self.spec.fail_every));
        }
        Ok(match self.spec.behavior {
            StubBehavior::Passthrough => identity_tail(ctx.mono_16k, ctx.new_frames, self.spec.output_rate),
            StubBehavior::Silence => {
                let len = (ctx.new_frames as u64 * u64::from(self.spec.output_rate) + 8_000) / 16_000;
                vec![0.0; len as usize]
            }
        })
    }

    fn input_sample_rate(&self) -> u32 {
        MODEL_INPUT_RATE
    }

    fn output_sample_rate(&self) -> u32 {
        self.spec.output_rate
    }

    fn min_context_samples(&self) -> usize {
        self.spec.min_context_samples
    }
}

// ---------------------------------------------------------------------------------------------
// Algorithmic latency probe
// ---------------------------------------------------------------------------------------------

/// The harness's algorithmic latency for a stream configuration and model rates.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct AlgorithmicLatency {
    /// Output-device frames between an input sample and its converted copy.
    pub frames: f64,
    pub ms: f64,
    /// Correlation of the probe's output with the input (quality of the measurement).
    pub correlation: f64,
}

/// Probe input length: whole blocks covering the 2.5 s chirp train, 1.5 s for its delayed copy
/// and two blocks of harness hold-back.
fn probe_input_frames(r_in: u32, block: usize) -> usize {
    let block = block.max(1);
    let need = ((2.5 + 1.5) * f64::from(r_in)).ceil() as usize + 2 * block;
    need.div_ceil(block) * block
}

/// Measure the latency the harness itself adds (block accumulation, input resampler, context
/// ring, SOLA, output resampler and priming cushion) by running an identity model with the same
/// rates and context through a harness built exactly like the run's, and locating a chirp train
/// in its output. Model-internal delay is not included (an identity model has none).
pub fn probe_algorithmic_latency(stream: &StreamSpec, model_out_sr: u32, min_context: usize) -> Result<AlgorithmicLatency, String> {
    let (r_in, r_out) = (stream.input_rate, stream.output_rate);
    let (ch_in, ch_out) = (usize::from(stream.input_channels), usize::from(stream.output_channels));
    let block = stream.block_frames as usize;
    let mut harness = VcStreamHarness::new(
        Box::new(StubModel::identity(model_out_sr, min_context)),
        r_in,
        r_out,
        ch_in,
        ch_out,
        stream.extra_samples_16k(),
        stream.crossfade_samples(model_out_sr),
        block,
    )?;
    let mut rng = SplitMix64::new(0x9e0b_e5a7);
    let train = BurstTrain::irregular(2.5, 0.03, 0.12, &mut rng);
    // The train, then 1.5 s of silence and two blocks, in whole blocks, so its delayed copy is
    // complete in the output whatever the block (up to 48000 frames: 6 s at 8 kHz).
    let total_in = probe_input_frames(r_in, block);
    let input = to_interleaved(&train.render(r_in, total_in), ch_in);
    let mut out = Vec::with_capacity(total_in * r_out as usize / r_in as usize * ch_out + 4096);
    for tick in input.chunks_exact(block * ch_in) {
        match harness.push_and_convert(tick) {
            Ok(v) => out.extend(v),
            Err(e) => return Err(format!("latency probe: {e}")),
        }
    }
    let out = channel(&out, ch_out, 0);
    let reference = train.render(r_out, (2.5 * f64::from(r_out)) as usize);
    let max_lag = out.len().saturating_sub(reference.len());
    let est = estimate_delay(&reference, &out, max_lag).ok_or("latency probe: the harness produced no output")?;
    if est.correlation < 0.9 {
        return Err(format!(
            "latency probe: the identity model's output does not match its input (correlation {:.3})",
            est.correlation
        ));
    }
    Ok(AlgorithmicLatency {
        frames: est.lag,
        ms: est.lag * 1000.0 / f64::from(r_out),
        correlation: est.correlation,
    })
}

// ---------------------------------------------------------------------------------------------
// RVC
// ---------------------------------------------------------------------------------------------

/// `rmvpe` / `crepe` → `rmvpe`, `dio`, else `fcpe` (the voice changer's `normalize_pitch_extractor`).
fn normalize_pitch(s: &str) -> &'static str {
    match s.trim().to_ascii_lowercase().as_str() {
        "rmvpe" | "crepe" => "rmvpe",
        "dio" => "dio",
        _ => "fcpe",
    }
}

fn explicit_or_auto(
    label: &str,
    explicit: &str,
    auto: impl Fn(&Path) -> Option<PathBuf>,
    app_data: Option<&Path>,
) -> Result<PathBuf, String> {
    let t = explicit.trim();
    if !t.is_empty() {
        let p = PathBuf::from(t);
        return if p.is_file() {
            Ok(p)
        } else {
            Err(format!("{label} not found: {}", p.display()))
        };
    }
    app_data
        .and_then(auto)
        .ok_or_else(|| format!("{label} not found: set its path in the spec, or place it under app_data/voice-changer/base"))
}

/// Diagnostics components the RVC loaders record the benchmark's sessions under: generator,
/// content encoder, pitch extractor. `build_rvc` sets `VcSessionOptions::component_prefix` to
/// `bench`, so these stay separate from the live voice changer's `vc` / `vc.*` rows.
const BENCH_COMPONENT_PREFIX: &str = "bench";
const RVC_COMPONENTS: [&str; 3] = ["bench.vc", "bench.vc.content", "bench.vc.pitch"];

/// Load an RVC slot with the voice changer's rules; returns the model and its info.
fn build_rvc(spec: &RvcSpec, priority: WorkerPriority) -> Result<(Box<dyn VoiceConversionModel>, EngineInfo), String> {
    let slot = spec.slot_dir.clone();
    if !slot.is_dir() {
        return Err(format!("RVC slot folder not found: {}", slot.display()));
    }
    let app_data = diagnostics::app_data_dir();
    let preset = RvcPresetProfile::from_str(&spec.preset);
    let ep = if spec.execution_provider.trim().is_empty() {
        preset.default_ep()
    } else {
        normalize_ep(&spec.execution_provider)
    };
    let mut warnings = Vec::new();
    let mut pitch = normalize_pitch(&spec.pitch_extractor);
    if ep == "cpu" && pitch == "rmvpe" {
        warnings.push("rmvpe is GPU-only in Sigil; the voice changer runs fcpe on the CPU EP, and so does this run".into());
        pitch = "fcpe";
    }
    let hubert = explicit_or_auto(
        "HuBERT / ContentVec ONNX",
        &spec.hubert,
        |ad| resolve_hubert("", ad, &slot),
        app_data.as_deref(),
    )?;
    let opts = VcSessionOptions {
        cuda_device_index: spec.gpu_device_index,
        streamer_profile: preset == RvcPresetProfile::Streamer,
        intra_threads: spec
            .intra_threads
            .or((priority == WorkerPriority::ProtectGame).then_some(2)),
        thread_priority: Some(priority),
        vram_cap_mb: if ep == "cuda" { spec.vram_cap_mb } else { None },
        component_prefix: Some(BENCH_COMPONENT_PREFIX),
        // Same CPU allocator as the voice changer it measures.
        cpu_arena: false,
    };
    let since = unix_ms_now();
    let extractor: Box<dyn PitchExtractor> = match pitch {
        "rmvpe" => {
            let p = explicit_or_auto("RMVPE ONNX", &spec.rmvpe, |ad| resolve_rmvpe("", ad, &slot), app_data.as_deref())?;
            Box::new(RmvpePitchExtractor::load(&p.to_string_lossy(), ep, &opts)?)
        }
        "dio" => Box::new(DioPitchExtractor::new()),
        _ => {
            let p = explicit_or_auto("FCPE ONNX", &spec.fcpe, |ad| resolve_fcpe("", ad, &slot), app_data.as_deref())?;
            Box::new(FcpePitchExtractor::load(&p.to_string_lossy(), ep, &opts)?)
        }
    };
    let model = RvcModel::load(&slot, &hubert.to_string_lossy(), extractor, ep, &opts, spec.silence_skip)?;
    warnings.extend(model.warnings().iter().cloned());

    // The EP the sessions landed on: a component that fell back wins (as the worker reports it).
    let fresh: Vec<_> = diagnostics::ep_outcomes()
        .into_iter()
        .filter(|e| e.at_unix_ms >= since && RVC_COMPONENTS.contains(&e.component.as_str()))
        .collect();
    let outcome = fresh
        .iter()
        .find(|e| e.outcome.fell_back())
        .or_else(|| fresh.iter().find(|e| e.component == RVC_COMPONENTS[0]))
        .map(|e| e.outcome.clone());
    let (active, reason) = match outcome {
        Some(o) => (o.active.clone(), o.reason.clone()),
        None => (ep.to_string(), None),
    };
    let info = EngineInfo {
        kind: "rvc".into(),
        description: format!(
            "RVC {} ({} preset, {pitch}, {ep})",
            slot.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            spec.preset
        ),
        input_rate: model.input_sample_rate(),
        output_rate: model.output_sample_rate(),
        min_context_samples: model.min_context_samples(),
        execution_provider: Some(ep.to_string()),
        execution_provider_active: Some(active),
        execution_provider_reason: reason,
        pitch_extractor: Some(pitch.to_string()),
        load_ms: 0.0,
        warnings,
    };
    Ok((Box::new(model), info))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::bench::spec::BenchSpec;

    fn stream(v: serde_json::Value) -> StreamSpec {
        serde_json::from_value(v).unwrap()
    }

    #[test]
    fn identity_tail_follows_the_alignment_contract() {
        let x: Vec<f32> = (0..1_000).map(|i| i as f32).collect();
        assert_eq!(identity_tail(&x, 160, 16_000), x[840..].to_vec());
        let y = identity_tail(&x, 160, 40_000);
        assert_eq!(y.len(), 400);
        // Last output sample = last input sample; a ramp stays a ramp (step 0.4 input samples).
        assert!((y[399] - 999.0).abs() < 1e-3, "{}", y[399]);
        assert!((y[398] - 998.6).abs() < 1e-3, "{}", y[398]);
        assert!((y[0] - (999.0 - 399.0 * 0.4)).abs() < 1e-3, "{}", y[0]);
    }

    #[test]
    fn stub_fails_on_schedule_and_simulates_cost() {
        let spec: StubSpec = serde_json::from_value(serde_json::json!({
            "compute_ms": 2.0, "compute_jitter_ms": 1.0, "fail_every": 3, "behavior": "silence"
        }))
        .unwrap();
        let sim = SimControl::default();
        sim.simulate.store(true, Ordering::SeqCst);
        let mut m = StubModel::new(spec, 1, sim.clone());
        let audio = vec![0.1f32; 2_000];
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: 400,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
        };
        let t0 = Instant::now();
        let results: Vec<_> = (0..6).map(|_| m.convert(&ctx)).collect();
        assert!(t0.elapsed() < Duration::from_millis(5), "simulated cost must not be spent");
        assert!(results[2].is_err() && results[5].is_err());
        assert_eq!(results[0].as_ref().unwrap().len(), 1_000);
        assert!(results[0].as_ref().unwrap().iter().all(|&v| v == 0.0));
        let cost = Duration::from_nanos(sim.cost_ns.load(Ordering::SeqCst));
        assert!(cost >= Duration::from_millis(6) && cost <= Duration::from_millis(18), "{cost:?}");
    }

    #[test]
    fn engine_handles_share_one_model_and_time_each_call() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "t", "engine": {"kind": "stub"}, "runs": [{"mode": "offline", "name": "o"}]
        }))
        .unwrap();
        let engine = build_engine(&spec.engine, WorkerPriority::Auto, 1).unwrap();
        assert_eq!((engine.info.input_rate, engine.info.output_rate), (16_000, 40_000));
        assert!(!engine.info.gpu());
        let times = Arc::new(Mutex::new(ModelTimes::default()));
        let mut h = engine.handle(times.clone());
        let audio = vec![0.0f32; 1_000];
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: 320,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
        };
        h.warmup_at(&ctx).unwrap();
        assert_eq!(h.convert(&ctx).unwrap().len(), 800);
        assert_eq!(h.convert(&ctx).unwrap().len(), 800);
        let t = times.lock().unwrap();
        assert_eq!(t.convert_ms.len(), 2);
        assert_eq!(t.errors, 0);
    }

    /// The probe measures a constant harness delay that grows with the block and the extra
    /// context exactly as the harness accounting says, at matching and differing device rates.
    #[test]
    fn probe_measures_the_harness_latency() {
        let base = probe_algorithmic_latency(&stream(serde_json::json!({})), 40_000, 720).unwrap();
        assert!(base.correlation > 0.95, "{base:?}");
        // 512-frame blocks at 48 kHz: at least one block plus the SOLA / resampler delays, and
        // well under 100 ms.
        assert!(base.ms > 10.0 && base.ms < 100.0, "{base:?}");
        let bigger = probe_algorithmic_latency(&stream(serde_json::json!({"block_frames": 1024})), 40_000, 720).unwrap();
        assert!(bigger.ms > base.ms + 5.0, "{bigger:?} vs {base:?}");
        let rates = probe_algorithmic_latency(
            &stream(serde_json::json!({"input_rate": 44_100, "output_rate": 48_000, "input_channels": 2})),
            48_000,
            720,
        )
        .unwrap();
        assert!(rates.correlation > 0.95 && rates.ms > 10.0 && rates.ms < 100.0, "{rates:?}");
        // Deterministic: the same configuration measures the same latency.
        let again = probe_algorithmic_latency(&stream(serde_json::json!({})), 40_000, 720).unwrap();
        assert!((again.frames - base.frames).abs() < 1e-9);
    }

    /// Every valid block works at the lowest device rates: the probe feeds whole blocks covering
    /// the train, its delayed copy and the hold-back (a 2 s or 6 s block used to leave no output).
    #[test]
    fn probe_handles_blocks_longer_than_its_train() {
        assert_eq!(probe_input_frames(48_000, 512) % 512, 0);
        assert!(probe_input_frames(48_000, 512) >= 4 * 48_000 + 2 * 512);
        assert_eq!(probe_input_frames(8_000, 48_000), 3 * 48_000);
        for (rate, block) in [(16_000u32, 32_768u32), (8_000, 48_000), (8_000, 20_480)] {
            let s = stream(serde_json::json!({"input_rate": rate, "output_rate": rate, "block_frames": block}));
            let p = probe_algorithmic_latency(&s, 40_000, 720).unwrap_or_else(|e| panic!("{rate}/{block}: {e}"));
            // At 8 kHz the train's 4 kHz top sits at Nyquist, so the match is a little weaker.
            assert!(p.correlation > 0.9, "{rate}/{block}: {p:?}");
            let block_ms = f64::from(block) * 1000.0 / f64::from(rate);
            assert!(p.ms > 0.0 && p.ms < block_ms + 1_000.0, "{rate}/{block}: {p:?}");
        }
    }

    #[test]
    fn rvc_engine_reports_missing_files_clearly() {
        let spec: RvcSpec = serde_json::from_value(serde_json::json!({"slot_dir": "Z:/no/such/slot"})).unwrap();
        let err = build_rvc(&spec, WorkerPriority::Auto).err().unwrap();
        assert!(err.contains("slot folder not found"), "{err}");
        assert_eq!(normalize_pitch("CREPE"), "rmvpe");
        assert_eq!(normalize_pitch("harvest"), "fcpe");
    }

    /// The real `RvcModel` path (hand-encoded content encoder + generator, dio F0) on the CPU EP.
    #[test]
    fn rvc_engine_loads_a_slot_on_the_cpu() {
        let (root, slot, hubert) = crate::services::neural_audio::vc::test_models::write_slot("bench-engine");
        let spec: RvcSpec = serde_json::from_value(serde_json::json!({
            "slot_dir": slot, "hubert": hubert.to_string_lossy(), "pitch_extractor": "dio",
            "execution_provider": "cpu", "preset": "streamer", "silence_skip": false
        }))
        .unwrap();
        let engine = build_engine(&EngineSpec::Rvc(spec), WorkerPriority::Auto, 1).unwrap();
        assert_eq!(engine.info.output_rate, 40_000);
        assert_eq!(engine.info.execution_provider_active.as_deref(), Some("cpu"));
        assert_eq!(engine.info.pitch_extractor.as_deref(), Some("dio"));
        assert!(engine.info.load_ms > 0.0);
        drop(engine);
        let _ = std::fs::remove_dir_all(root);
    }
}
