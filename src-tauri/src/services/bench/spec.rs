//! Benchmark spec (`sigil-bench-spec/1`): what to run, on which engine and stream configuration,
//! under which load, and the thresholds that decide pass / fail.
//!
//! A spec is a JSON document (a file for `Sigil.exe --headless bench <spec.json>`, a value for IPC
//! `plugin:bench|run_benchmark`). Unknown fields are errors, so a typo never silently disables a
//! setting, and threshold keys must name a metric the run actually produces (see
//! [`super::report::run_produces`]). Relative paths resolve against the spec file's folder (CLI)
//! or `app_data/bench/` (IPC).
//!
//! Stream defaults mirror the voice changer's own defaults (`voice_changer/params.rs`: 512-frame
//! blocks, 64 ms extra context, 20 ms crossfade, 35 / 70 ms output prefill / queue, 300 ms latency
//! budget), so a spec that only names an engine measures what users run by default.
//!
//! The full field reference with examples is in `docs/benchmarks/README.md`.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::services::neural_audio::chunk_processor::ALLOWED_CHUNK_FRAMES;

use super::report::{known_metric, run_produces, unproduced_reason};

/// Schema id of the spec format this module reads.
pub const SPEC_SCHEMA: &str = "sigil-bench-spec/1";

/// A whole benchmark: one engine, one stream configuration, one input, several runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchSpec {
    #[serde(default = "default_schema")]
    pub schema: String,
    /// Short id; also names the default output folder.
    pub name: String,
    #[serde(default)]
    pub description: String,
    /// Seed of every pseudo-random choice (synthetic input, compute jitter, burst spacing).
    #[serde(default = "default_seed")]
    pub seed: u64,
    #[serde(default)]
    pub input: InputSpec,
    #[serde(default)]
    pub stream: StreamSpec,
    /// The engine every offline / paced run drives through the real stream harness.
    pub engine: EngineSpec,
    pub runs: Vec<RunSpec>,
    /// Thresholds applied to every run whose mode and configuration produce the metric (the others
    /// skip it); such a run that did not report the metric fails it as "not measured".
    #[serde(default)]
    pub thresholds: Thresholds,
    #[serde(default)]
    pub output: OutputSpec,
}

fn default_schema() -> String {
    SPEC_SCHEMA.to_string()
}

fn default_seed() -> u64 {
    0x5167_11be
}

/// Extra files written next to `report.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutputSpec {
    /// `<run>.output.wav`: the converted audio (offline: the harness output; paced: what the
    /// simulated output device played, underrun holds included).
    #[serde(default)]
    pub write_audio: bool,
    /// `<run>.timeline.csv`: one row per block (paced and offline runs).
    #[serde(default = "default_true")]
    pub write_timeline: bool,
}

fn default_true() -> bool {
    true
}

impl Default for OutputSpec {
    fn default() -> Self {
        Self {
            write_audio: false,
            write_timeline: true,
        }
    }
}

/// Input audio fed to offline and paced runs (looped when a paced run is longer).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum InputSpec {
    /// Generated in-process (no files; CI-safe). See `signals.rs`.
    Synthetic {
        #[serde(default)]
        signal: SignalKind,
        #[serde(default = "default_input_seconds")]
        seconds: f64,
        /// Peak level of the signal.
        #[serde(default = "default_level_dbfs")]
        level_dbfs: f64,
    },
    /// A WAV file (any PCM / float format), resampled to the stream's input rate.
    Wav { path: PathBuf },
}

impl Default for InputSpec {
    fn default() -> Self {
        InputSpec::Synthetic {
            signal: SignalKind::default(),
            seconds: default_input_seconds(),
            level_dbfs: default_level_dbfs(),
        }
    }
}

fn default_input_seconds() -> f64 {
    8.0
}

fn default_level_dbfs() -> f64 {
    -12.0
}

/// Synthetic input signals.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SignalKind {
    /// Voiced speech-like source: gliding F0 with vibrato through moving formants, syllable
    /// envelope and pauses (voiced for every pitch extractor).
    #[default]
    Voice,
    /// Three incommensurate tones (311, 523, 797 Hz).
    Tones,
    /// Log chirps (200 Hz - 4 kHz) at irregular intervals.
    ChirpTrain,
    /// Single-sample clicks at irregular intervals.
    ImpulseTrain,
    /// White noise.
    Noise,
    Silence,
}

/// Device and harness configuration (what the voice changer would open and build).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StreamSpec {
    #[serde(default = "default_rate")]
    pub input_rate: u32,
    #[serde(default = "default_rate")]
    pub output_rate: u32,
    #[serde(default = "default_input_channels")]
    pub input_channels: u16,
    #[serde(default = "default_output_channels")]
    pub output_channels: u16,
    /// Worker block (device frames per tick); one of the voice changer's allowed chunk sizes.
    #[serde(default = "default_block_frames")]
    pub block_frames: u32,
    #[serde(default = "default_extra_context_ms")]
    pub extra_context_ms: f64,
    #[serde(default = "default_crossfade_ms")]
    pub crossfade_ms: f64,
    /// Capture device period (WASAPI shared mode: 10 ms).
    #[serde(default = "default_callback_ms")]
    pub capture_callback_ms: f64,
    /// Playback device period.
    #[serde(default = "default_callback_ms")]
    pub playback_callback_ms: f64,
    #[serde(default = "default_prefill_ms")]
    pub output_prefill_ms: f64,
    #[serde(default = "default_max_queue_ms")]
    pub output_max_queue_ms: f64,
    /// Input latency budget (stale audio beyond it is dropped, oldest first).
    #[serde(default = "default_budget_ms")]
    pub latency_budget_ms: f64,
    #[serde(default)]
    pub pitch_semitones: f32,
    #[serde(default)]
    pub index_rate: f32,
    #[serde(default = "default_protect")]
    pub protect: f32,
    #[serde(default)]
    pub speaker_id: i64,
    /// Formant tilt applied after conversion, as the voice changer does for RVC output.
    #[serde(default)]
    pub formant_shift: f32,
    #[serde(default = "default_gain")]
    pub input_gain: f32,
    #[serde(default = "default_gain")]
    pub output_gain: f32,
    /// Worker thread policy: `auto`, `protect_game`, `protect_voice` (as the voice changer).
    #[serde(default = "default_worker_priority")]
    pub worker_priority: String,
}

impl Default for StreamSpec {
    fn default() -> Self {
        // Round-trip through serde so the defaults live in exactly one place (the attributes).
        serde_json::from_value(serde_json::json!({})).expect("stream defaults")
    }
}

fn default_rate() -> u32 {
    48_000
}
fn default_input_channels() -> u16 {
    1
}
fn default_output_channels() -> u16 {
    2
}
fn default_block_frames() -> u32 {
    crate::services::neural_audio::DEFAULT_CHUNK_FRAMES as u32
}
fn default_extra_context_ms() -> f64 {
    64.0
}
fn default_crossfade_ms() -> f64 {
    20.0
}
fn default_callback_ms() -> f64 {
    10.0
}
fn default_prefill_ms() -> f64 {
    35.0
}
fn default_max_queue_ms() -> f64 {
    70.0
}
fn default_budget_ms() -> f64 {
    300.0
}
fn default_protect() -> f32 {
    0.33
}
fn default_gain() -> f32 {
    1.0
}
fn default_worker_priority() -> String {
    "auto".to_string()
}

impl StreamSpec {
    /// Device frames per capture callback.
    pub fn capture_callback_frames(&self) -> usize {
        ms_to_frames(self.capture_callback_ms, self.input_rate).max(1)
    }

    /// Device frames per playback callback.
    pub fn playback_callback_frames(&self) -> usize {
        ms_to_frames(self.playback_callback_ms, self.output_rate).max(1)
    }

    /// Duration of one worker block.
    pub fn block_ms(&self) -> f64 {
        f64::from(self.block_frames) * 1000.0 / f64::from(self.input_rate.max(1))
    }

    /// Extra context in model-input (16 kHz) samples, as the worker computes it.
    pub fn extra_samples_16k(&self) -> usize {
        ((self.extra_context_ms / 1000.0) * 16_000.0).round().max(0.0) as usize
    }

    /// Crossfade in model-output samples, as the worker computes it.
    pub fn crossfade_samples(&self, model_out_sr: u32) -> usize {
        ((self.crossfade_ms / 1000.0) * f64::from(model_out_sr)).round().max(0.0) as usize
    }
}

/// Frames in `ms` at `rate`, rounded.
pub fn ms_to_frames(ms: f64, rate: u32) -> usize {
    (ms.max(0.0) * f64::from(rate) / 1000.0).round() as usize
}

/// The engine behind the harness.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum EngineSpec {
    /// Built-in stand-in model (no files): plumbing, latency accounting and load behaviour.
    Stub(StubSpec),
    /// A real RVC slot through the production `RvcModel` (HuBERT + generator + pitch sessions).
    Rvc(RvcSpec),
}

/// Built-in stand-in model.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StubSpec {
    #[serde(default)]
    pub behavior: StubBehavior,
    #[serde(default = "default_stub_output_rate")]
    pub output_rate: u32,
    /// `VoiceConversionModel::min_context_samples` (16 kHz samples); RVC's HuBERT needs 720.
    #[serde(default = "default_stub_min_context")]
    pub min_context_samples: usize,
    /// Simulated inference time per block.
    #[serde(default)]
    pub compute_ms: f64,
    /// Uniform jitter (+-) on `compute_ms`.
    #[serde(default)]
    pub compute_jitter_ms: f64,
    #[serde(default)]
    pub compute: ComputeKind,
    /// Every n-th conversion fails (0 = never): exercises the error path and concealment.
    #[serde(default)]
    pub fail_every: u32,
}

fn default_stub_output_rate() -> u32 {
    40_000
}

fn default_stub_min_context() -> usize {
    crate::services::neural_audio::vc::rvc::MIN_CONTEXT_16K
}

/// What the stub returns.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StubBehavior {
    /// Identity: the new block (and SOLA tail), resampled to the output rate and time-aligned by
    /// the `VoiceConversionModel` contract.
    #[default]
    Passthrough,
    Silence,
}

/// How the stub spends `compute_ms`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComputeKind {
    /// Sleep (a GPU wait: no CPU used).
    #[default]
    Sleep,
    /// Busy CPU work on the worker thread (a CPU execution provider).
    Spin,
}

/// A real RVC voice (same load path and rules as the voice changer's RVC mode).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RvcSpec {
    /// Slot folder (generator `.onnx`, optional `manifest.json` / `.index`).
    pub slot_dir: PathBuf,
    /// Explicit model paths; empty = the voice changer's auto rule (app_data/voice-changer/base,
    /// then the slot folder).
    #[serde(default)]
    pub hubert: String,
    #[serde(default)]
    pub fcpe: String,
    #[serde(default)]
    pub rmvpe: String,
    /// `fcpe`, `rmvpe` (`crepe` is its alias) or `dio`; rmvpe runs as fcpe on the CPU EP.
    #[serde(default = "default_pitch_extractor")]
    pub pitch_extractor: String,
    /// `cpu`, `cuda` or `dml`; empty = the preset's default.
    #[serde(default)]
    pub execution_provider: String,
    /// `streamer`, `low_latency` or `quality`.
    #[serde(default = "default_rvc_preset")]
    pub preset: String,
    #[serde(default)]
    pub gpu_device_index: i32,
    /// ONNX Runtime intra-op threads per session (default: the voice changer's rule).
    #[serde(default)]
    pub intra_threads: Option<usize>,
    /// CUDA arena cap in MiB (CUDA only).
    #[serde(default)]
    pub vram_cap_mb: Option<u32>,
    #[serde(default = "default_true")]
    pub silence_skip: bool,
}

fn default_pitch_extractor() -> String {
    "fcpe".to_string()
}

fn default_rvc_preset() -> String {
    "low_latency".to_string()
}

/// One measurement.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum RunSpec {
    /// Throughput: the whole input through the harness as fast as possible.
    Offline(OfflineRun),
    /// A simulated device clock: capture and playback callbacks at their periods, the production
    /// latency budget and output queue, optional load and stalls.
    Paced(PacedRun),
    /// Real devices: an impulse / chirp train played to an output and recorded from an input
    /// (e.g. through a virtual cable and Sigil's voice changer, or any other app).
    E2eLoopback(LoopbackRun),
}

impl RunSpec {
    pub fn name(&self) -> &str {
        match self {
            RunSpec::Offline(r) => &r.name,
            RunSpec::Paced(r) => &r.name,
            RunSpec::E2eLoopback(r) => &r.name,
        }
    }

    pub fn mode(&self) -> RunMode {
        match self {
            RunSpec::Offline(_) => RunMode::Offline,
            RunSpec::Paced(_) => RunMode::Paced,
            RunSpec::E2eLoopback(_) => RunMode::E2eLoopback,
        }
    }

    pub fn thresholds(&self) -> &Thresholds {
        match self {
            RunSpec::Offline(r) => &r.thresholds,
            RunSpec::Paced(r) => &r.thresholds,
            RunSpec::E2eLoopback(r) => &r.thresholds,
        }
    }

    /// Whether the run drives the engine through the harness.
    pub fn uses_engine(&self) -> bool {
        !matches!(self, RunSpec::E2eLoopback(_))
    }
}

/// Run modes (report / metric scoping).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunMode {
    Offline,
    Paced,
    E2eLoopback,
}

impl RunMode {
    pub const ALL: [RunMode; 3] = [RunMode::Offline, RunMode::Paced, RunMode::E2eLoopback];

    pub fn as_str(self) -> &'static str {
        match self {
            RunMode::Offline => "offline",
            RunMode::Paced => "paced",
            RunMode::E2eLoopback => "e2e_loopback",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OfflineRun {
    pub name: String,
    /// Passes over the input.
    #[serde(default = "default_repeat")]
    pub repeat: u32,
    #[serde(default)]
    pub thresholds: Thresholds,
}

fn default_repeat() -> u32 {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PacedRun {
    pub name: String,
    /// Simulated capture time.
    #[serde(default = "default_paced_seconds")]
    pub seconds: f64,
    #[serde(default)]
    pub clock: ClockKind,
    /// Latency samples before this time (start-up priming) are not counted; below `seconds`.
    #[serde(default = "default_warmup_s")]
    pub warmup_s: f64,
    #[serde(default)]
    pub load: LoadSpec,
    /// Injected stalls (a frozen GPU / worker): the block starting at `at_s` takes `ms` longer.
    /// `at_s` is at least [`STALL_BASELINE_S`] after `warmup_s` (the pre-stall latency baseline).
    #[serde(default)]
    pub stalls: Vec<StallSpec>,
    #[serde(default)]
    pub thresholds: Thresholds,
}

fn default_paced_seconds() -> f64 {
    10.0
}

fn default_warmup_s() -> f64 {
    0.5
}

/// Measured run before the first stall: `stall_latency_recovery_ms` compares the latency after a
/// stall with its median over this window (after `warmup_s`).
pub const STALL_BASELINE_S: f64 = 1.0;

/// Clock of a paced run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ClockKind {
    /// Real time: callbacks at wall-clock cadence, real sleeps, real contention (the default).
    #[default]
    Wall,
    /// Simulated time advanced by each block's measured compute (plus a stub's declared cost and
    /// injected stalls, which are accounted, not slept): fast, and free of sleep / scheduling
    /// effects, but compute times are still real (they vary with the PC's load).
    Virtual,
    /// Simulated time advanced only by a stub's declared compute and injected stalls; the real
    /// compute of the harness and the engine is not charged (and `block_ms` reports the declared
    /// cost). Every timeline metric is reproducible on any PC under any load: for tests of the
    /// latency, budget and queue logic, not for performance.
    Simulated,
}

impl ClockKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ClockKind::Wall => "wall",
            ClockKind::Virtual => "virtual",
            ClockKind::Simulated => "simulated",
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StallSpec {
    pub at_s: f64,
    pub ms: f64,
}

/// Contention during a paced run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoadSpec {
    #[serde(default)]
    pub gpu: Option<GpuLoadSpec>,
    #[serde(default)]
    pub cpu: Option<CpuLoadSpec>,
}

/// An ONNX Runtime MatMul chain run at a duty cycle (a game rendering on the same GPU).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GpuLoadSpec {
    /// Busy fraction of wall time, (0, 1].
    pub duty: f64,
    /// Duty-cycle control period.
    #[serde(default = "default_gpu_period_ms")]
    pub period_ms: f64,
    /// `cuda`, `dml` or `cpu` (the fallback when no GPU EP loads is reported, not hidden).
    #[serde(default = "default_gpu_ep")]
    pub execution_provider: String,
    #[serde(default)]
    pub device_index: i32,
    /// Square matrix size.
    #[serde(default = "default_gpu_size")]
    pub size: usize,
    /// MatMuls per run.
    #[serde(default = "default_gpu_depth")]
    pub depth: usize,
}

fn default_gpu_period_ms() -> f64 {
    20.0
}
fn default_gpu_ep() -> String {
    "cuda".to_string()
}
fn default_gpu_size() -> usize {
    1024
}
fn default_gpu_depth() -> usize {
    4
}

/// Busy CPU threads at a duty cycle (a game's CPU load).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CpuLoadSpec {
    pub threads: usize,
    #[serde(default = "default_cpu_duty")]
    pub duty: f64,
    #[serde(default = "default_cpu_period_ms")]
    pub period_ms: f64,
}

fn default_cpu_duty() -> f64 {
    1.0
}
fn default_cpu_period_ms() -> f64 {
    10.0
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LoopbackRun {
    pub name: String,
    /// Output device name (exact, or a unique part of it); required. The literal `default` picks
    /// the system default output, which plays the chirps on the real speakers or headset.
    #[serde(default)]
    pub output_device: String,
    /// Input device name (exact, or a unique part of it); required. The literal `default` picks
    /// the system default input.
    #[serde(default)]
    pub input_device: String,
    /// Recorded input channel (0-based); `None` = mix all channels.
    #[serde(default)]
    pub input_channel: Option<u16>,
    #[serde(default = "default_bursts")]
    pub bursts: u32,
    /// Mean burst spacing (each spacing is jittered by +-20 %).
    #[serde(default = "default_interval_ms")]
    pub interval_ms: f64,
    #[serde(default = "default_burst_ms")]
    pub burst_ms: f64,
    #[serde(default = "default_burst_level")]
    pub level_dbfs: f64,
    #[serde(default)]
    pub detector: Detector,
    /// Longest latency searched for after each burst.
    #[serde(default = "default_max_latency_ms")]
    pub max_latency_ms: f64,
    /// Subtracted from every measurement: the latency of the bare loop (e.g. a direct virtual
    /// cable) measured once with the same devices, which also cancels the drivers' timestamp bias.
    #[serde(default)]
    pub calibration_ms: f64,
    /// Silence before the first burst (device start-up).
    #[serde(default = "default_loopback_warmup_ms")]
    pub warmup_ms: f64,
    /// Detection strength (matched-filter peak over the window's median) below which a burst
    /// counts as missed.
    #[serde(default = "default_min_snr")]
    pub min_snr: f64,
    #[serde(default)]
    pub thresholds: Thresholds,
}

fn default_bursts() -> u32 {
    20
}
fn default_interval_ms() -> f64 {
    600.0
}
fn default_burst_ms() -> f64 {
    40.0
}
fn default_burst_level() -> f64 {
    -12.0
}
fn default_max_latency_ms() -> f64 {
    400.0
}
fn default_loopback_warmup_ms() -> f64 {
    500.0
}
fn default_min_snr() -> f64 {
    8.0
}

/// How bursts are found in the recording.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Detector {
    /// Matched filter against the emitted chirp (sample-accurate; needs an unaltered path).
    #[default]
    Xcorr,
    /// Envelope onset (for paths that change the timbre, e.g. a voice changer), ~1-2 ms.
    Onset,
}

/// Metric name -> limit.
pub type Thresholds = BTreeMap<String, Limit>;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limit {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
}

impl BenchSpec {
    /// Parse and validate a spec document.
    pub fn from_json_str(text: &str) -> Result<Self, String> {
        let spec: BenchSpec = serde_json::from_str(text).map_err(|e| format!("invalid bench spec: {e}"))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Parse and validate a spec value (IPC).
    pub fn from_value(value: serde_json::Value) -> Result<Self, String> {
        let spec: BenchSpec = serde_json::from_value(value).map_err(|e| format!("invalid bench spec: {e}"))?;
        spec.validate()?;
        Ok(spec)
    }

    /// Read a spec file; relative paths inside it are resolved against its folder.
    pub fn load(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        let mut spec = Self::from_json_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let base = path.parent().map(Path::to_path_buf).unwrap_or_default();
        spec.resolve_paths(&base);
        Ok(spec)
    }

    /// Make every relative file path absolute against `base`.
    pub fn resolve_paths(&mut self, base: &Path) {
        let fix = |p: &mut PathBuf| {
            if !p.as_os_str().is_empty() && p.is_relative() {
                *p = base.join(&*p);
            }
        };
        let fix_str = |s: &mut String| {
            if !s.trim().is_empty() && Path::new(s.trim()).is_relative() {
                *s = base.join(s.trim()).to_string_lossy().into_owned();
            }
        };
        if let InputSpec::Wav { path } = &mut self.input {
            fix(path);
        }
        if let EngineSpec::Rvc(r) = &mut self.engine {
            fix(&mut r.slot_dir);
            fix_str(&mut r.hubert);
            fix_str(&mut r.fcpe);
            fix_str(&mut r.rmvpe);
        }
    }

    /// Check every value against the ranges the harness and the voice changer support.
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != SPEC_SCHEMA {
            return Err(format!("unsupported spec schema '{}' (expected '{SPEC_SCHEMA}')", self.schema));
        }
        check_id("name", &self.name)?;
        if self.runs.is_empty() {
            return Err("the spec has no runs".into());
        }
        let mut names = HashSet::new();
        for run in &self.runs {
            check_id("run name", run.name())?;
            if !names.insert(run.name().to_string()) {
                return Err(format!("duplicate run name '{}'", run.name()));
            }
        }
        if self.runs.iter().any(RunSpec::uses_engine) {
            self.input.validate()?;
            self.stream.validate()?;
            self.engine.validate()?;
        }
        for run in &self.runs {
            run.validate().map_err(|e| format!("run '{}': {e}", run.name()))?;
        }
        // Spec-wide thresholds must name a metric some run of this spec produces (typos and
        // thresholds that would gate nothing never pass silently).
        for (metric, limit) in &self.thresholds {
            check_limit(metric, limit)?;
            if !RunMode::ALL.iter().any(|&m| known_metric(m, metric)) {
                return Err(format!("threshold '{metric}' is not a metric any run produces"));
            }
            if !self.runs.iter().any(|r| run_produces(r, metric)) {
                return Err(format!("threshold '{metric}' applies to no run of this spec"));
            }
        }
        Ok(())
    }
}

/// Ids become file names: letters, digits, `-`, `_`, `.`; at most 64 characters.
fn check_id(what: &str, id: &str) -> Result<(), String> {
    let ok = !id.is_empty()
        && id.len() <= 64
        && !id.starts_with('.')
        && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
    if ok {
        Ok(())
    } else {
        Err(format!(
            "{what} '{id}' must be 1-64 characters of letters, digits, '-', '_' or '.' (not starting with '.')"
        ))
    }
}

fn check_range(what: &str, v: f64, lo: f64, hi: f64) -> Result<(), String> {
    if v.is_finite() && v >= lo && v <= hi {
        Ok(())
    } else {
        Err(format!("{what} = {v} is outside {lo}..={hi}"))
    }
}

fn check_limit(metric: &str, limit: &Limit) -> Result<(), String> {
    if limit.min.is_none() && limit.max.is_none() {
        return Err(format!("threshold '{metric}' needs a min or a max"));
    }
    for v in [limit.min, limit.max].into_iter().flatten() {
        if !v.is_finite() {
            return Err(format!("threshold '{metric}' has a non-finite limit"));
        }
    }
    if let (Some(lo), Some(hi)) = (limit.min, limit.max) {
        if lo > hi {
            return Err(format!("threshold '{metric}': min {lo} is above max {hi}"));
        }
    }
    Ok(())
}

impl InputSpec {
    fn validate(&self) -> Result<(), String> {
        match self {
            InputSpec::Synthetic { seconds, level_dbfs, .. } => {
                check_range("input.seconds", *seconds, 0.5, 3600.0)?;
                check_range("input.level_dbfs", *level_dbfs, -80.0, 0.0)
            }
            InputSpec::Wav { path } => {
                if path.as_os_str().is_empty() {
                    Err("input.path is empty".into())
                } else {
                    Ok(())
                }
            }
        }
    }
}

impl StreamSpec {
    fn validate(&self) -> Result<(), String> {
        for (what, rate) in [("stream.input_rate", self.input_rate), ("stream.output_rate", self.output_rate)] {
            check_range(what, f64::from(rate), 8_000.0, 384_000.0)?;
        }
        for (what, ch) in [
            ("stream.input_channels", self.input_channels),
            ("stream.output_channels", self.output_channels),
        ] {
            check_range(what, f64::from(ch), 1.0, 8.0)?;
        }
        if !ALLOWED_CHUNK_FRAMES.contains(&self.block_frames) {
            return Err(format!(
                "stream.block_frames = {} is not a voice-changer chunk size (the voice changer would run 512 instead); allowed: {:?}",
                self.block_frames, ALLOWED_CHUNK_FRAMES
            ));
        }
        check_range("stream.extra_context_ms", self.extra_context_ms, 0.0, 2_000.0)?;
        check_range("stream.crossfade_ms", self.crossfade_ms, 0.0, 500.0)?;
        check_range("stream.capture_callback_ms", self.capture_callback_ms, 1.0, 500.0)?;
        check_range("stream.playback_callback_ms", self.playback_callback_ms, 1.0, 500.0)?;
        // The voice changer's own clamp (`clamp_output_buffer_ms`).
        check_range("stream.output_prefill_ms", self.output_prefill_ms, 5.0, 300.0)?;
        check_range(
            "stream.output_max_queue_ms",
            self.output_max_queue_ms,
            self.output_prefill_ms + 5.0,
            600.0,
        )?;
        // The voice changer's budget range (`clamp_latency_budget_ms`).
        check_range("stream.latency_budget_ms", self.latency_budget_ms, 100.0, 2_000.0)?;
        check_range("stream.pitch_semitones", f64::from(self.pitch_semitones), -48.0, 48.0)?;
        check_range("stream.index_rate", f64::from(self.index_rate), 0.0, 1.0)?;
        check_range("stream.protect", f64::from(self.protect), 0.0, 0.5)?;
        check_range("stream.formant_shift", f64::from(self.formant_shift), -1.0, 1.0)?;
        check_range("stream.input_gain", f64::from(self.input_gain), 0.0, 16.0)?;
        check_range("stream.output_gain", f64::from(self.output_gain), 0.0, 16.0)?;
        match self.worker_priority.as_str() {
            "auto" | "protect_game" | "protect_voice" => Ok(()),
            other => Err(format!(
                "stream.worker_priority '{other}' must be auto, protect_game or protect_voice"
            )),
        }
    }
}

impl EngineSpec {
    fn validate(&self) -> Result<(), String> {
        match self {
            EngineSpec::Stub(s) => {
                check_range("engine.output_rate", f64::from(s.output_rate), 8_000.0, 96_000.0)?;
                check_range("engine.min_context_samples", s.min_context_samples as f64, 0.0, 160_000.0)?;
                check_range("engine.compute_ms", s.compute_ms, 0.0, 10_000.0)?;
                check_range("engine.compute_jitter_ms", s.compute_jitter_ms, 0.0, s.compute_ms.max(0.0))
            }
            EngineSpec::Rvc(r) => {
                if r.slot_dir.as_os_str().is_empty() {
                    return Err("engine.slot_dir is empty".into());
                }
                match r.execution_provider.trim().to_ascii_lowercase().as_str() {
                    "" | "cpu" | "cuda" | "gpu" | "dml" | "directml" => {}
                    other => return Err(format!("engine.execution_provider '{other}' must be cpu, cuda or dml")),
                }
                match r.preset.trim().to_ascii_lowercase().as_str() {
                    "streamer" | "low_latency" | "quality" => {}
                    other => return Err(format!("engine.preset '{other}' must be streamer, low_latency or quality")),
                }
                match r.pitch_extractor.trim().to_ascii_lowercase().as_str() {
                    "fcpe" | "rmvpe" | "crepe" | "dio" => {}
                    other => return Err(format!("engine.pitch_extractor '{other}' must be fcpe, rmvpe or dio")),
                }
                if let Some(n) = r.intra_threads {
                    check_range("engine.intra_threads", n as f64, 1.0, 64.0)?;
                }
                Ok(())
            }
        }
    }
}

impl RunSpec {
    fn validate(&self) -> Result<(), String> {
        for (metric, limit) in self.thresholds() {
            check_limit(metric, limit)?;
            if !known_metric(self.mode(), metric) {
                return Err(format!(
                    "threshold '{metric}' is not a metric of {} runs",
                    self.mode().as_str()
                ));
            }
            if let Some(why) = unproduced_reason(self, metric) {
                return Err(format!("threshold '{metric}' can never be measured: {why}"));
            }
        }
        match self {
            RunSpec::Offline(r) => check_range("repeat", f64::from(r.repeat), 1.0, 1_000.0),
            RunSpec::Paced(r) => {
                check_range("seconds", r.seconds, 0.5, 4.0 * 3600.0)?;
                check_range("warmup_s", r.warmup_s, 0.0, r.seconds)?;
                if r.warmup_s >= r.seconds {
                    return Err(format!(
                        "warmup_s = {} must be below seconds = {} (latency is only measured after it)",
                        r.warmup_s, r.seconds
                    ));
                }
                for s in &r.stalls {
                    check_range("stalls[].at_s", s.at_s, 0.0, r.seconds)?;
                    check_range("stalls[].ms", s.ms, 1.0, 60_000.0)?;
                    // Latency recovery needs a pre-stall baseline: measured latency between the
                    // warm-up and the stall.
                    if s.at_s < r.warmup_s + STALL_BASELINE_S {
                        return Err(format!(
                            "stalls[].at_s: the stall at {} s must start at least {STALL_BASELINE_S} s after warmup_s ({} s), so its latency recovery has a baseline",
                            s.at_s, r.warmup_s
                        ));
                    }
                    // Recovery must be observable: at least 1 s of run after the stall ends.
                    if s.at_s + s.ms / 1000.0 + 1.0 > r.seconds {
                        return Err(format!(
                            "stalls[].at_s: the stall at {} s ({} ms) must end at least 1 s before the run's {} s",
                            s.at_s, s.ms, r.seconds
                        ));
                    }
                }
                if let Some(g) = &r.load.gpu {
                    check_range("load.gpu.duty", g.duty, 0.01, 1.0)?;
                    check_range("load.gpu.period_ms", g.period_ms, 1.0, 1_000.0)?;
                    check_range("load.gpu.size", g.size as f64, 16.0, 8_192.0)?;
                    check_range("load.gpu.depth", g.depth as f64, 1.0, 64.0)?;
                    match g.execution_provider.trim().to_ascii_lowercase().as_str() {
                        "cpu" | "cuda" | "gpu" | "dml" | "directml" => {}
                        other => return Err(format!("load.gpu.execution_provider '{other}' must be cuda, dml or cpu")),
                    }
                }
                if let Some(c) = &r.load.cpu {
                    check_range("load.cpu.threads", c.threads as f64, 1.0, 256.0)?;
                    check_range("load.cpu.duty", c.duty, 0.01, 1.0)?;
                    check_range("load.cpu.period_ms", c.period_ms, 1.0, 1_000.0)?;
                }
                Ok(())
            }
            RunSpec::E2eLoopback(r) => {
                // The meter plays loud chirps: never on a device the author did not name.
                for (what, device) in [("output_device", &r.output_device), ("input_device", &r.input_device)] {
                    if device.trim().is_empty() {
                        return Err(format!(
                            "{what} is required: name the device (e.g. a virtual cable), or write \"default\" for the system default, which plays the chirps on your real speakers or headset"
                        ));
                    }
                }
                check_range("bursts", f64::from(r.bursts), 1.0, 10_000.0)?;
                check_range("burst_ms", r.burst_ms, 5.0, 500.0)?;
                check_range("level_dbfs", r.level_dbfs, -60.0, 0.0)?;
                check_range("max_latency_ms", r.max_latency_ms, 5.0, 5_000.0)?;
                check_range("calibration_ms", r.calibration_ms, -1_000.0, 5_000.0)?;
                check_range("warmup_ms", r.warmup_ms, 0.0, 10_000.0)?;
                check_range("min_snr", r.min_snr, 1.0, 1_000.0)?;
                // Each burst's search window (up to max_latency after it, plus the burst itself)
                // must end before the earliest possible next burst (spacing jittered by -20 %).
                let min_gap = r.max_latency_ms + r.burst_ms + 20.0;
                if !(r.interval_ms.is_finite() && r.interval_ms * 0.8 > min_gap && r.interval_ms <= 60_000.0) {
                    return Err(format!(
                        "interval_ms = {} is too short: 0.8 x interval must exceed max_latency_ms + burst_ms + 20 ms = {min_gap}",
                        r.interval_ms
                    ));
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMOKE: &str = include_str!("../../../../scripts/bench/specs/stub-smoke.json");

    fn minimal() -> serde_json::Value {
        serde_json::json!({
            "name": "t",
            "engine": { "kind": "stub" },
            "runs": [ { "mode": "offline", "name": "off" } ]
        })
    }

    #[test]
    fn checked_in_smoke_spec_is_valid() {
        let spec = BenchSpec::from_json_str(SMOKE).expect("stub-smoke.json parses and validates");
        assert_eq!(spec.name, "stub-smoke");
        assert!(spec.runs.iter().any(|r| r.mode() == RunMode::Paced));
        assert!(spec.runs.iter().any(|r| r.mode() == RunMode::Offline));
    }

    #[test]
    fn defaults_mirror_the_voice_changer() {
        let spec = BenchSpec::from_value(minimal()).unwrap();
        let s = &spec.stream;
        assert_eq!((s.input_rate, s.output_rate), (48_000, 48_000));
        assert_eq!(s.block_frames, 512);
        assert_eq!((s.extra_context_ms, s.crossfade_ms), (64.0, 20.0));
        assert_eq!((s.output_prefill_ms, s.output_max_queue_ms), (35.0, 70.0));
        assert_eq!(s.latency_budget_ms, 300.0);
        assert_eq!(s.capture_callback_frames(), 480);
        assert_eq!(s.extra_samples_16k(), 1024);
        assert_eq!(s.crossfade_samples(40_000), 800);
        assert!(matches!(spec.input, InputSpec::Synthetic { signal: SignalKind::Voice, .. }));
        match spec.engine {
            EngineSpec::Stub(st) => {
                assert_eq!(st.output_rate, 40_000);
                assert_eq!(st.min_context_samples, 720);
                assert_eq!(st.behavior, StubBehavior::Passthrough);
            }
            other => panic!("{other:?}"),
        }
        assert!(spec.output.write_timeline && !spec.output.write_audio);
    }

    #[test]
    fn unknown_fields_are_errors_everywhere() {
        for (path, patch) in [
            ("/", serde_json::json!({"nmae": "x"})),
            ("/output", serde_json::json!({"write_wav": true})),
            ("/input", serde_json::json!({"secs": 3})),
            ("/stream", serde_json::json!({"block_frame": 512})),
            ("/engine", serde_json::json!({"compute_msec": 1})),
            ("/runs/0", serde_json::json!({"repeats": 2})),
        ] {
            let mut v = minimal();
            for key in ["stream", "output", "input"] {
                v.as_object_mut().unwrap().insert(key.into(), serde_json::json!({}));
            }
            v["input"] = serde_json::json!({"kind": "synthetic"});
            let target = if path == "/" { &mut v } else { v.pointer_mut(path).unwrap() };
            for (k, val) in patch.as_object().unwrap() {
                target.as_object_mut().unwrap().insert(k.clone(), val.clone());
            }
            let err = BenchSpec::from_value(v).unwrap_err();
            assert!(err.contains("unknown field"), "{path}: {err}");
        }
    }

    #[test]
    fn invalid_values_are_rejected_with_the_field_name() {
        let cases: Vec<(serde_json::Value, &str)> = vec![
            (serde_json::json!({"stream": {"block_frames": 1920}}), "block_frames"),
            (serde_json::json!({"stream": {"latency_budget_ms": 50}}), "latency_budget_ms"),
            (serde_json::json!({"stream": {"output_prefill_ms": 40, "output_max_queue_ms": 42}}), "output_max_queue_ms"),
            (serde_json::json!({"stream": {"worker_priority": "realtime"}}), "worker_priority"),
            (serde_json::json!({"name": "bad name"}), "name"),
            (serde_json::json!({"runs": []}), "no runs"),
            (
                serde_json::json!({"runs": [{"mode": "offline", "name": "a"}, {"mode": "offline", "name": "a"}]}),
                "duplicate",
            ),
            (serde_json::json!({"thresholds": {"block_ms_p95x": {"max": 1}}}), "block_ms_p95x"),
            (serde_json::json!({"thresholds": {"block_ms_p95": {}}}), "min or a max"),
            (
                serde_json::json!({"runs": [{"mode": "offline", "name": "a", "thresholds": {"latency_ms_p99": {"max": 1}}}]}),
                "not a metric of offline runs",
            ),
            (
                serde_json::json!({"runs": [{"mode": "paced", "name": "p", "seconds": 2, "stalls": [{"at_s": 5, "ms": 10}]}]}),
                "stalls[].at_s",
            ),
            // A stall inside the warm-up (or right after it) leaves latency recovery no baseline.
            (
                serde_json::json!({"runs": [{"mode": "paced", "name": "p", "seconds": 6, "warmup_s": 0.5,
                                             "stalls": [{"at_s": 0.2, "ms": 200}]}]}),
                "after warmup_s",
            ),
            (
                serde_json::json!({"runs": [{"mode": "paced", "name": "p", "seconds": 6, "warmup_s": 2,
                                             "stalls": [{"at_s": 2.5, "ms": 200}]}]}),
                "after warmup_s",
            ),
            (
                serde_json::json!({"runs": [{"mode": "paced", "name": "p", "seconds": 2, "warmup_s": 2}]}),
                "warmup_s = 2 must be below seconds",
            ),
            // Thresholds on metrics the run's configuration can never produce.
            (
                serde_json::json!({"runs": [{"mode": "paced", "name": "p", "thresholds": {"stall_recovery_blocks": {"max": 2}}}]}),
                "no stalls",
            ),
            (
                serde_json::json!({"runs": [{"mode": "paced", "name": "p", "clock": "simulated",
                                             "thresholds": {"cpu_cores": {"max": 1}}}]}),
                "simulated clock",
            ),
            (
                serde_json::json!({"runs": [{"mode": "paced", "name": "p", "thresholds": {"gpu_load_duty": {"min": 0.5}}}]}),
                "no load.gpu",
            ),
            (serde_json::json!({"thresholds": {"latency_ms_p99": {"max": 80}}}), "applies to no run"),
            // The loopback meter plays chirps: both devices must be named.
            (
                serde_json::json!({"runs": [{"mode": "e2e_loopback", "name": "l"}]}),
                "output_device is required",
            ),
            (
                serde_json::json!({"runs": [{"mode": "e2e_loopback", "name": "l", "output_device": "CABLE Input", "input_device": " "}]}),
                "input_device is required",
            ),
            (
                serde_json::json!({"runs": [{"mode": "e2e_loopback", "name": "l", "output_device": "CABLE Input",
                                             "input_device": "CABLE Output", "interval_ms": 300, "max_latency_ms": 400}]}),
                "interval_ms",
            ),
            (serde_json::json!({"schema": "sigil-bench-spec/9"}), "schema"),
        ];
        for (patch, needle) in cases {
            let mut v = minimal();
            for (k, val) in patch.as_object().unwrap() {
                v.as_object_mut().unwrap().insert(k.clone(), val.clone());
            }
            let err = BenchSpec::from_value(v).unwrap_err();
            assert!(err.contains(needle), "expected '{needle}' in: {err}");
        }
    }

    #[test]
    fn loopback_only_specs_skip_engine_checks_and_paths_resolve() {
        let v = serde_json::json!({
            "name": "loop",
            "engine": { "kind": "rvc", "slot_dir": "voices/a", "hubert": "base/hubert.onnx" },
            "input": { "kind": "wav", "path": "clips/a.wav" },
            "runs": [ { "mode": "e2e_loopback", "name": "cable", "output_device": "CABLE Input", "input_device": "default" } ]
        });
        let mut spec = BenchSpec::from_value(v).unwrap();
        let base = std::env::temp_dir().join("bench-spec-base");
        spec.resolve_paths(&base);
        match (&spec.input, &spec.engine) {
            (InputSpec::Wav { path }, EngineSpec::Rvc(r)) => {
                assert_eq!(path, &base.join("clips/a.wav"));
                assert_eq!(r.slot_dir, base.join("voices/a"));
                assert_eq!(r.hubert, base.join("base/hubert.onnx").to_string_lossy());
                assert!(r.fcpe.is_empty(), "empty paths stay empty (= auto)");
            }
            other => panic!("{other:?}"),
        }
    }
}
