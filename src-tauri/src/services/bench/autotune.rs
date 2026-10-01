//! Per-PC autotuner (P3-F): sweeps EP x F0 method x block (20-100 ms) x context x precision x GPU
//! levers on a clip through the real harness and engine factory, picks the lowest-latency
//! configuration with p99 RTF <= 0.5 and fidelity >= T5, and stores it per GPU and driver
//! (`app_data/voice-changer/autotune.json`). Plan acceptance: a valid preset on CPU and on GPU.
//!
//! # The search (per EP, then across EPs)
//! 1. **EP**: every EP id the request names or this build offers (`capabilities`). The voice's
//!    generator sessions decide whether the EP runs (`vc.enc` / `vc.flowdec` of template voices,
//!    `vc` of bring-your-own ONNX voices): one that lands on another rung drops the EP with its
//!    reason, and so does `tensorrt_rtx` when no generator session runs TensorRT (the CUDA rung is
//!    tuned as `cuda`; classic TensorRT counts, with a note). A pitch model that falls back drops only
//!    that F0 method; any other component that falls back is measured as it runs, with a note.
//! 1b. **CPU threads** (the CPU rung): the reference configuration with one ONNX Runtime thread
//!    (the Streamer preset), two (`protect_game`) and the default pool; the fewest threads that
//!    meet the limit win (least load next to a game, T4), else the fastest. The result then carries
//!    `rvc_preset` / `worker_priority`, the voice changer's own thread policy. An explicit priority
//!    is kept: `protect_game` never gets the pool, `protect_voice` is never demoted to two threads.
//! 2. **F0 method**: every method the build runs on the rung (after the voice changer's stand-in and
//!    GPU-only rules), ranked by accuracy (gross pitch error + V/UV error against the synthetic
//!    voice's exact F0; a user clip only times the engine, it has no F0 truth), then the most
//!    accurate one whose engine meets the RTF limit at the reference block (40 ms on a GPU, 80 ms on
//!    the CPU). When none does, the most accurate one within 20 % of the fastest one's RTF (single
//!    measurements vary by about that much): the F0 methods cost about the same next to the voice,
//!    and the block sweep finds a block that keeps up.
//! 3. **Block**: ascending (template voices: 20 / 40 / 60 / 80 / 100 ms engine blocks on the user's
//!    device block; bring-your-own ONNX voices: the voice changer's chunk sizes from 20 to 100 ms);
//!    the first that meets the limit has the lowest latency. On the CPU the sweep continues to
//!    200 ms when nothing shorter meets it.
//! 4. **Context**: the largest extra context (64-512 ms) that still meets the limit (context costs
//!    compute, not latency, and improves quality).
//! 5. **Precision** (template voices on a GPU): fp16 vs fp32; fp32 when it is within 5 % of fp16.
//! 6. **GPU levers** (template voices on a GPU, when this build runs them), measured under the H2
//!    load (the conv hammer at 70 % duty on the same GPU: a game rendering), where they matter:
//!    from all of them off, IOBinding with CUDA graphs (CUDA; graphs need IOBinding), else IOBinding
//!    alone, then the priority stream / queue, each set kept when it lowers the loaded p99 RTF by
//!    5 % or more; the winner is measured idle again. Only the levers this stage measured are
//!    written to the params: stages 2-5 measure with the voice changer's own lever settings, which
//!    every lever keeps when the stage does not run (out of time) or never tried it.
//! 7. **Fidelity**: the winner is scored against its offline ceiling (the `quality` mode): CAM++
//!    cosine >= 0.97 (when installed), voiced F0 RMSE <= 15 cents, V/UV error <= 3 %, no clicks at
//!    the joins; a failing block steps up (twice at most).
//!
//! "Meets the limit": p99 RTF (compute over the real-time budget of a block) <= `rtf_limit`
//! (default 0.5, headroom for a game), at most 0.1 % of blocks over budget, no model errors,
//! measured on the virtual clock (real compute, no sleeps). Across EPs the lowest p50 latency
//! wins. When nothing meets the limit (a slow CPU), the fastest configuration is returned with a
//! note: the result is always a valid set of parameters.
//!
//! # Conditions
//! Everything the voice changer runs with besides the knobs the tuner sweeps ([`TuneConditions`]:
//! device rates and block, crossfade, output cushion and budget, worker priority, retrieval and the
//! hot voice controls, F0 range, denoiser, lookahead, the lever settings, GPU index and VRAM cap)
//! is put on every candidate's stream and voice; the IPC path fills it from the voice changer's
//! params and last device format (`cmd_bench`). The result records it, and [`stale_reasons`] lists
//! what changed since, so a tune measured for other settings is never presented as current.
//!
//! # Frozen by P3-S0 (signatures; P3-F implements)
//! [`available`], [`AUTOTUNE_PROGRESS_EVENT`], [`AutotuneRequest`], [`AutotuneResult`] (its
//! `params` is a snake_case `VoiceChangerParams` patch the UI sends with
//! `set_voice_changer_params`), [`AutotuneProgress`], [`run_autotune`], [`stored_result`]. The
//! structs gained fields (P3-F); nothing was renamed.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Deserializer, Serialize};

use super::engine::{build_engine, probe_algorithmic_latency, EngineInfo};
use super::paced::{run_paced, PacedInput};
use super::report::RunOutcome;
use super::signals::{read_wav, resample, synth, to_interleaved, to_mono, voice_f0_truth};
use super::spec::{
    AutotuneRun, BenchSpec, ClockKind, EngineSpec, GpuLoadKind, InputSpec, LoadSpec, PacedRun, QualityRun, RvcSpec, ScoringSpec, SignalKind, StreamSpec,
};
use super::{claim_measurement, BenchContext, Measurement, MeasurementGuard, Prepared, Progress};
use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::chunk_processor::ALLOWED_CHUNK_FRAMES;
use crate::services::neural_audio::denoise::DenoiserKind;
use crate::services::neural_audio::ort_session::{VcSessionOptions, EP_TRT_CLASSIC};
use crate::services::neural_audio::vc::engine::options::{F0Range, Lever, LeverMode};
use crate::services::neural_audio::vc::f0_fusion::{create_pitch_extractor, PitchBankRequest};
use crate::services::neural_audio::vc::manifest::SlotFormat;
use crate::services::rt_thread::WorkerPriority;
use crate::services::voice_changer::capabilities::{capabilities, lever_applies, lever_available};
use crate::services::voice_changer::params::{
    base_ep, effective_pitch_extractor, normalize_execution_provider, normalize_pitch_extractor, normalize_rvc_preset, VoiceChangerParams,
};

/// Event: autotune progress (payload [`AutotuneProgress`]).
pub const AUTOTUNE_PROGRESS_EVENT: &str = "voice_changer:autotune_progress";
/// Schema of the stored results file.
pub const RESULTS_SCHEMA: &str = "sigil-autotune/1";
/// File of the stored results under `app_data/voice-changer/`.
pub const RESULTS_FILE: &str = "autotune.json";

/// Whether the autotuner runs in this build.
pub fn available() -> bool {
    true
}

/// Held while an in-app autotune runs ([`begin`]); frees the measurement slot it shares with the
/// in-app benchmark when dropped.
#[derive(Debug)]
pub struct RunGuard {
    _slot: MeasurementGuard,
}

/// Claim the measurement slot for a tune, in one atomic step (`Err`: a benchmark or another tune
/// holds it, with the message to show).
pub fn begin() -> Result<RunGuard, String> {
    claim_measurement(Measurement::Autotune)
        .map(|g| RunGuard { _slot: g })
        .map_err(|held| held.busy_message(Measurement::Autotune))
}

/// What to tune (snake_case).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AutotuneRequest {
    /// Voice slot to tune with (default: the active voice).
    #[serde(default)]
    pub slot_dir: Option<String>,
    /// EP ids to try (`cpu`, `cuda`, `dml`, `tensorrt_rtx`, `dml_hp`); empty = every one available.
    #[serde(default)]
    pub execution_providers: Vec<String>,
    /// A short sweep (fewer blocks and candidates).
    #[serde(default)]
    pub quick: bool,
    /// Stop after this long and return the best so far.
    #[serde(default)]
    pub max_seconds: Option<f64>,
    // ---- P3-F ----
    /// Explicit base models (default: the voice changer's automatic rules).
    #[serde(default)]
    pub hubert: Option<String>,
    #[serde(default)]
    pub fcpe: Option<String>,
    #[serde(default)]
    pub rmvpe: Option<String>,
    /// Preset whose `auto` lever table applies (default `low_latency`).
    #[serde(default)]
    pub preset: Option<String>,
    /// F0 methods to try; empty = every method this build runs.
    #[serde(default)]
    pub pitch_extractors: Vec<String>,
    /// A WAV to tune on; default: the synthetic voice (exact F0 known).
    #[serde(default)]
    pub clip: Option<String>,
    /// Device rate and block (the voice changer's `chunk_frames`) the voice will run with.
    #[serde(default)]
    pub device_rate: Option<u32>,
    #[serde(default)]
    pub chunk_frames: Option<u32>,
    /// p99 RTF limit (default 0.5).
    #[serde(default)]
    pub rtf_limit: Option<f64>,
    /// Score the winner against its offline ceiling (default true).
    #[serde(default)]
    pub verify_fidelity: Option<bool>,
    /// The voice changer settings to measure with (default: the voice changer's defaults; the IPC
    /// path fills the current ones). `device_rate` / `chunk_frames` above override theirs.
    #[serde(default)]
    pub conditions: Option<TuneConditions>,
}

impl AutotuneRequest {
    /// Check the fields a spec may set.
    pub fn validate(&self) -> Result<(), String> {
        for ep in &self.execution_providers {
            match ep.trim().to_ascii_lowercase().as_str() {
                "cpu" | "cuda" | "gpu" | "dml" | "tensorrt_rtx" | "dml_hp" => {}
                other => return Err(format!("request.execution_providers: '{other}' is not an EP id")),
            }
        }
        for p in &self.pitch_extractors {
            match p.trim().to_ascii_lowercase().as_str() {
                "fcpe" | "rmvpe" | "dio" | "swiftf0" | "crepe" | "pesto" | "fusion" => {}
                other => return Err(format!("request.pitch_extractors: '{other}' is not an F0 method")),
            }
        }
        if let Some(r) = self.rtf_limit {
            if !(r.is_finite() && (0.05..=1.0).contains(&r)) {
                return Err(format!("request.rtf_limit = {r} is outside 0.05..=1"));
            }
        }
        if let Some(s) = self.max_seconds {
            if !(s.is_finite() && (10.0..=3_600.0).contains(&s)) {
                return Err(format!("request.max_seconds = {s} is outside 10..=3600"));
            }
        }
        if let Some(r) = self.device_rate {
            if !(8_000..=192_000).contains(&r) {
                return Err(format!("request.device_rate = {r} is outside 8000..=192000"));
            }
        }
        if let Some(c) = self.chunk_frames {
            if !ALLOWED_CHUNK_FRAMES.contains(&c) {
                return Err(format!("request.chunk_frames = {c} is not a voice-changer chunk size"));
            }
        }
        if let Some(c) = &self.conditions {
            c.validate().map_err(|e| format!("request.conditions: {e}"))?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Conditions
// ---------------------------------------------------------------------------------------------

/// The voice changer settings a tune measures with, besides the knobs it sweeps (snake_case; see
/// the module docs). Rates and the chunk size at 0 mean unknown (the stream's defaults).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TuneConditions {
    /// Capture (device) rate, output rate and output channels of the voice changer's devices.
    pub device_rate: u32,
    pub output_rate: u32,
    pub output_channels: u16,
    /// Device block (the voice changer's `chunk_frames`).
    pub chunk_frames: u32,
    /// `rvc_preset` (its `auto` lever table; the CPU thread stage may set another).
    pub preset: String,
    pub crossfade_ms: f64,
    pub output_prefill_ms: f64,
    pub output_max_queue_ms: f64,
    pub latency_budget_ms: f64,
    /// `auto` / `protect_game` / `protect_voice` (the CPU thread stage may set another).
    pub worker_priority: String,
    pub pitch_semitones: f32,
    /// Retrieval mix (0 = the index is not searched).
    pub index_rate: f32,
    pub protect: f32,
    pub speaker_id: i64,
    pub formant_shift: f32,
    pub input_gain: f32,
    pub output_gain: f32,
    pub rms_mix: f32,
    pub formant_semitones: f32,
    pub noise_scale: f32,
    pub silence_db: f32,
    /// `speech` / `singing`.
    pub f0_range: String,
    /// `off` / `gtcrn` / `dfn3` / `dpdfnet` (`gtcrn` runs inside the measured engine, as in the
    /// voice changer's RVC mode; the others run on the capture copy, outside it: noted).
    pub denoiser: String,
    pub lookahead_ms: f64,
    /// Lever requests (`auto` / `on` / `off`) by name, as the voice changer has them.
    pub levers: BTreeMap<String, String>,
    pub gpu_device_index: i32,
    /// CUDA arena cap in MiB (0 = none).
    pub vram_cap_mb: u32,
}

fn default_rvc_spec() -> RvcSpec {
    serde_json::from_value(serde_json::json!({ "slot_dir": "" })).expect("rvc defaults")
}

impl Default for TuneConditions {
    /// The voice changer's defaults (the stream's and the voice's).
    fn default() -> Self {
        TuneConditions::of(&default_rvc_spec(), &StreamSpec::default())
    }
}

fn worker_priority_id(s: &str) -> String {
    WorkerPriority::from_str_lossy(s).as_str().to_string()
}

fn lever_mode_id(s: &str) -> String {
    LeverMode::from_str_lossy(s).as_str().to_string()
}

impl TuneConditions {
    /// What a voice and stream measure with.
    pub(crate) fn of(base: &RvcSpec, s: &StreamSpec) -> TuneConditions {
        TuneConditions {
            device_rate: s.input_rate,
            output_rate: s.output_rate,
            output_channels: s.output_channels,
            chunk_frames: s.block_frames,
            preset: normalize_rvc_preset(&base.preset).to_string(),
            crossfade_ms: s.crossfade_ms,
            output_prefill_ms: s.output_prefill_ms,
            output_max_queue_ms: s.output_max_queue_ms,
            latency_budget_ms: s.latency_budget_ms,
            worker_priority: worker_priority_id(&s.worker_priority),
            pitch_semitones: s.pitch_semitones,
            index_rate: s.index_rate,
            protect: s.protect,
            speaker_id: s.speaker_id,
            formant_shift: s.formant_shift,
            input_gain: s.input_gain,
            output_gain: s.output_gain,
            rms_mix: s.rms_mix,
            formant_semitones: s.formant_semitones,
            noise_scale: s.noise_scale,
            silence_db: s.silence_db,
            f0_range: F0Range::from_str_lossy(&base.f0_range).as_str().to_string(),
            denoiser: DenoiserKind::from_str_lossy(&base.denoiser).as_str().to_string(),
            lookahead_ms: base.lookahead_ms,
            levers: Lever::ALL
                .iter()
                .map(|l| (l.as_str().to_string(), lever_mode_id(base.levers.get(l.as_str()).map_or("auto", String::as_str))))
                .collect(),
            gpu_device_index: base.gpu_device_index,
            vram_cap_mb: base.vram_cap_mb.unwrap_or(0),
        }
    }

    /// Put the conditions on a voice and stream (unknown rates / block keep the stream's).
    pub(crate) fn apply(&self, base: &mut RvcSpec, s: &mut StreamSpec) {
        if self.device_rate > 0 {
            s.input_rate = self.device_rate;
        }
        if self.output_rate > 0 {
            s.output_rate = self.output_rate;
        }
        if self.output_channels > 0 {
            s.output_channels = self.output_channels;
        }
        if ALLOWED_CHUNK_FRAMES.contains(&self.chunk_frames) {
            s.block_frames = self.chunk_frames;
        }
        if !self.preset.trim().is_empty() {
            base.preset = normalize_rvc_preset(&self.preset).to_string();
        }
        s.crossfade_ms = self.crossfade_ms;
        s.output_prefill_ms = self.output_prefill_ms;
        s.output_max_queue_ms = self.output_max_queue_ms;
        s.latency_budget_ms = self.latency_budget_ms;
        s.worker_priority = worker_priority_id(&self.worker_priority);
        s.pitch_semitones = self.pitch_semitones;
        s.index_rate = self.index_rate;
        s.protect = self.protect;
        s.speaker_id = self.speaker_id;
        s.formant_shift = self.formant_shift;
        s.input_gain = self.input_gain;
        s.output_gain = self.output_gain;
        s.rms_mix = self.rms_mix;
        s.formant_semitones = self.formant_semitones;
        s.noise_scale = self.noise_scale;
        s.silence_db = self.silence_db;
        base.f0_range = F0Range::from_str_lossy(&self.f0_range).as_str().to_string();
        base.denoiser = DenoiserKind::from_str_lossy(&self.denoiser).as_str().to_string();
        base.lookahead_ms = self.lookahead_ms;
        base.levers = self.levers.iter().map(|(k, v)| (k.trim().to_string(), lever_mode_id(v))).collect();
        base.gpu_device_index = self.gpu_device_index;
        base.vram_cap_mb = (self.vram_cap_mb > 0).then_some(self.vram_cap_mb);
    }

    /// The voice changer's params (as `get_voice_changer_params` returns them) and the device
    /// format of its last run (0 = unknown: not started in this session).
    pub fn from_params(p: &VoiceChangerParams, capture_rate: u32, output_rate: u32, output_channels: u16) -> TuneConditions {
        let d = TuneConditions::default();
        TuneConditions {
            device_rate: capture_rate,
            output_rate,
            output_channels,
            chunk_frames: p.chunk_frames,
            preset: normalize_rvc_preset(&p.rvc_preset).to_string(),
            crossfade_ms: p.crossfade_ms,
            output_prefill_ms: p.output_prefill_ms,
            output_max_queue_ms: p.output_max_queue_ms,
            latency_budget_ms: p.latency_budget_ms.unwrap_or(d.latency_budget_ms),
            worker_priority: worker_priority_id(p.worker_priority.as_deref().unwrap_or("auto")),
            pitch_semitones: p.pitch_semitones,
            index_rate: p.index_rate,
            protect: p.protect,
            speaker_id: p.speaker_id,
            formant_shift: p.formant_shift,
            input_gain: p.input_gain,
            output_gain: p.output_gain,
            rms_mix: p.rms_mix.unwrap_or(d.rms_mix),
            formant_semitones: p.formant_semitones.unwrap_or(d.formant_semitones),
            noise_scale: p.noise_scale.unwrap_or(d.noise_scale),
            silence_db: p.silence_db.unwrap_or(d.silence_db),
            f0_range: F0Range::from_str_lossy(p.f0_range.as_deref().unwrap_or("speech")).as_str().to_string(),
            denoiser: DenoiserKind::from_str_lossy(p.denoiser.as_deref().unwrap_or("off")).as_str().to_string(),
            lookahead_ms: p.lookahead_ms.unwrap_or(0.0),
            levers: Lever::ALL
                .iter()
                .map(|&l| (l.as_str().to_string(), lever_mode_id(p.lever(l).unwrap_or("auto"))))
                .collect(),
            gpu_device_index: p.gpu_device_index,
            vram_cap_mb: p.vram_cap_mb,
        }
    }

    /// A benchmark spec's stream checks on these values, plus the voice's.
    fn validate(&self) -> Result<(), String> {
        for (what, rate) in [("device_rate", self.device_rate), ("output_rate", self.output_rate)] {
            if rate != 0 && !(8_000..=192_000).contains(&rate) {
                return Err(format!("{what} = {rate} is outside 8000..=192000"));
            }
        }
        if self.chunk_frames != 0 && !ALLOWED_CHUNK_FRAMES.contains(&self.chunk_frames) {
            return Err(format!("chunk_frames = {} is not a voice-changer chunk size", self.chunk_frames));
        }
        for (name, mode) in &self.levers {
            if Lever::from_name(name.trim()).is_none() {
                return Err(format!("levers: '{name}' is not a lever"));
            }
            if !matches!(mode.trim().to_ascii_lowercase().as_str(), "auto" | "on" | "off") {
                return Err(format!("levers.{name} '{mode}' must be auto, on or off"));
            }
        }
        if !(self.lookahead_ms.is_finite() && (0.0..=80.0).contains(&self.lookahead_ms)) {
            return Err(format!("lookahead_ms = {} is outside 0..=80", self.lookahead_ms));
        }
        let mut base = default_rvc_spec();
        let mut s = StreamSpec::default();
        self.apply(&mut base, &mut s);
        s.validate()
    }
}

/// What changed since `result` was tuned that affects its cost or latency (empty: the stored tune
/// still describes this configuration). `now`: the current conditions
/// ([`TuneConditions::from_params`]; unknown rates are not compared); `current_slot`: the voice
/// selected now. The settings the tune itself writes (`rvc_preset`, `worker_priority`,
/// `chunk_frames`, the levers it measured) are compared with the value it wrote.
pub fn stale_reasons(result: &AutotuneResult, now: &TuneConditions, current_slot: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let norm = |s: &str| s.trim().replace('\\', "/").trim_end_matches('/').to_ascii_lowercase();
    if let Some(slot) = result.slot.as_deref() {
        if !current_slot.trim().is_empty() && norm(slot) != norm(current_slot) {
            out.push("another voice is selected".into());
        }
    }
    let Some(then) = &result.conditions else {
        out.push("the tune does not record the settings it was measured with".into());
        return out;
    };
    let wrote_str = |key: &str, measured: &str| -> String {
        result.params.get(key).and_then(|v| v.as_str()).unwrap_or(measured).to_string()
    };
    let chunk = result
        .params
        .get("chunk_frames")
        .and_then(|v| v.as_f64())
        .unwrap_or(f64::from(then.chunk_frames));
    let mut nums: Vec<(&str, f64, f64)> = vec![
        ("chunk_frames", chunk, f64::from(now.chunk_frames)),
        ("crossfade_ms", then.crossfade_ms, now.crossfade_ms),
        ("output_prefill_ms", then.output_prefill_ms, now.output_prefill_ms),
        ("output_max_queue_ms", then.output_max_queue_ms, now.output_max_queue_ms),
        ("latency_budget_ms", then.latency_budget_ms, now.latency_budget_ms),
        ("lookahead_ms", then.lookahead_ms, now.lookahead_ms),
        ("gpu_device_index", f64::from(then.gpu_device_index), f64::from(now.gpu_device_index)),
        ("vram_cap_mb", f64::from(then.vram_cap_mb), f64::from(now.vram_cap_mb)),
    ];
    if now.device_rate > 0 {
        nums.push(("device_rate", f64::from(then.device_rate), f64::from(now.device_rate)));
    }
    if now.output_rate > 0 {
        nums.push(("output_rate", f64::from(then.output_rate), f64::from(now.output_rate)));
    }
    for (name, was, is) in nums {
        if (was - is).abs() > 1e-6 {
            out.push(format!("{name}: {was} -> {is}"));
        }
    }
    let on_off = |rate: f32| if rate > 0.0 { "on" } else { "off" }.to_string();
    let mut texts: Vec<(String, String, String)> = vec![
        ("rvc_preset".into(), wrote_str("rvc_preset", &then.preset), now.preset.clone()),
        ("worker_priority".into(), wrote_str("worker_priority", &then.worker_priority), now.worker_priority.clone()),
        ("f0_range".into(), then.f0_range.clone(), now.f0_range.clone()),
        ("denoiser".into(), then.denoiser.clone(), now.denoiser.clone()),
        ("retrieval".into(), on_off(then.index_rate), on_off(now.index_rate)),
    ];
    for lever in Lever::ALL {
        let name = lever.as_str();
        let measured = then.levers.get(name).map_or("auto", String::as_str);
        let is = now.levers.get(name).map_or("auto", String::as_str);
        texts.push((name.to_string(), wrote_str(name, measured), is.to_string()));
    }
    for (name, was, is) in texts {
        if was != is {
            out.push(format!("{name}: {was} -> {is}"));
        }
    }
    out
}

/// One measured configuration.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Candidate {
    /// `pitch`, `ep`, `block`, `context`, `precision`, `lever`, `verify`.
    pub stage: String,
    pub ep: String,
    pub pitch: String,
    /// Engine block (template voices) or device block (bring-your-own ONNX), ms.
    pub block_ms: f64,
    pub chunk_frames: u32,
    pub extra_context_ms: f64,
    pub precision: String,
    /// Levers switched on (`io_binding,cuda_graphs`).
    pub levers: String,
    pub rtf_p99: Option<f64>,
    pub compute_ms_p95: Option<f64>,
    pub latency_ms_p50: Option<f64>,
    pub latency_ms_p99: Option<f64>,
    pub over_budget_ratio: Option<f64>,
    /// Pitch stage: accuracy of the F0 method on the tuning clip.
    pub gross_pitch_error: Option<f64>,
    pub vuv_error: Option<f64>,
    pub f0_rmse_cents: Option<f64>,
    pub fidelity: Option<f64>,
    pub ok: bool,
    pub note: String,
    /// CPU thread policy (`1`, `2`, `pool`; empty = the request's preset and priority).
    #[serde(default)]
    pub threads: String,
    /// Process CPU of the run (cores; the virtual clock's compute time).
    #[serde(default)]
    pub cpu_cores: Option<f64>,
    /// Measured under the H2 load (the lever stage).
    #[serde(default)]
    pub under_load: bool,
}

/// The chosen configuration (snake_case; stored per GPU and driver).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AutotuneResult {
    pub gpu: Option<String>,
    pub driver: Option<String>,
    pub created_unix_ms: u64,
    /// `VoiceChangerParams` fields to apply (e.g. `rvc_execution_provider`, `block_ms`,
    /// `pitch_extractor`, `precision`, `extra_context_ms`, the levers).
    pub params: serde_json::Map<String, serde_json::Value>,
    /// The winner's p99 RTF and per-block compute p95 (JSON `null` reads back as NaN: a record
    /// never fails to parse over a missing number).
    #[serde(default = "nan", deserialize_with = "f64_or_nan")]
    pub rtf_p99: f64,
    #[serde(default = "nan", deserialize_with = "f64_or_nan")]
    pub block_ms_p95: f64,
    pub latency_ms_p50: Option<f64>,
    /// Fidelity against the offline ceiling (speaker cosine), when measured.
    pub fidelity: Option<f64>,
    /// Benchmark report folder of the winning run.
    pub report_dir: Option<String>,
    pub notes: Vec<String>,
    // ---- P3-F ----
    #[serde(default)]
    pub schema: String,
    /// Whether the winner met the RTF limit (false: the fastest configuration found).
    #[serde(default)]
    pub met_limit: bool,
    #[serde(default)]
    pub slot: Option<String>,
    /// Slot format (`rvc`, `sigil-rvc-1`).
    #[serde(default)]
    pub format: Option<String>,
    #[serde(default)]
    pub execution_provider_active: Option<String>,
    #[serde(default)]
    pub latency_ms_p99: Option<f64>,
    /// The winner's fidelity metrics against its ceiling (quality-mode names).
    #[serde(default)]
    pub fidelity_metrics: BTreeMap<String, f64>,
    #[serde(default)]
    pub candidates: Vec<Candidate>,
    #[serde(default)]
    pub seconds: f64,
    #[serde(default)]
    pub sigil_version: String,
    /// The voice changer settings every candidate was measured with.
    #[serde(default)]
    pub conditions: Option<TuneConditions>,
    /// `voice_changer_get_autotune` only (never stored): what changed since the tune
    /// ([`stale_reasons`]); empty while the stored tune still describes the current settings.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub stale: Vec<String>,
}

fn nan() -> f64 {
    f64::NAN
}

fn f64_or_nan<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(d)?.unwrap_or(f64::NAN))
}

/// Payload of [`AUTOTUNE_PROGRESS_EVENT`].
#[derive(Clone, Debug, Default, Serialize)]
pub struct AutotuneProgress {
    /// `preparing`, `sweeping`, `verifying`, `done` or `failed`.
    pub stage: String,
    pub fraction: f64,
    /// The configuration being measured.
    pub candidate: Option<String>,
    pub message: String,
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The GPU and driver results are stored under: the non-software adapter with the most VRAM.
pub fn gpu_identity() -> (Option<String>, Option<String>) {
    crate::services::diagnostics::gpu_mem::adapters()
        .into_iter()
        .filter(|a| !a.software)
        .max_by_key(|a| a.dedicated_vram_mb)
        .map_or((None, None), |a| (Some(a.name.trim().to_string()), a.nvidia_driver.or(a.driver_version)))
}

// ---------------------------------------------------------------------------------------------
// Configurations
// ---------------------------------------------------------------------------------------------

/// A configuration under test.
#[derive(Clone, Debug, PartialEq)]
struct Config {
    ep: String,
    pitch: String,
    /// Template voices: engine block (ms). Bring-your-own ONNX: 0 (the device block is the block).
    block_ms: f64,
    chunk_frames: u32,
    extra_ms: f64,
    precision: String,
    levers: BTreeMap<String, String>,
    threads: Threads,
    /// GPU levers the lever stage measured both ways (written to the params; every other lever
    /// keeps the voice changer's own setting, which is what the configuration measured with).
    tuned: Vec<&'static str>,
}

/// The GPU levers the lever stage decides.
const GPU_LEVERS: [Lever; 3] = [Lever::IoBinding, Lever::CudaGraphs, Lever::GpuPriority];

/// ONNX Runtime thread policy of a CPU configuration, through the voice changer's own params: the
/// Streamer preset caps every session at one thread, `protect_game` at two (below-normal priority),
/// and the other presets with the `auto` priority keep ONNX Runtime's default pool (one thread per
/// physical core, below-normal priority on the CPU EP).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Threads {
    /// The request's preset and priority (GPU rungs, and the CPU before the thread stage).
    #[default]
    AsRequested,
    One,
    Two,
    Pool,
}

impl Threads {
    /// The policies the CPU stage sweeps, fewest threads first, for the voice changer's worker
    /// priority. An explicit choice is kept: `protect_game` never gets the pool at normal priority,
    /// and `protect_voice` is never demoted to `protect_game`'s two threads.
    fn sweep(worker_priority: &str) -> &'static [Threads] {
        match worker_priority {
            "protect_game" => &[Threads::One, Threads::Two],
            "protect_voice" => &[Threads::One, Threads::Pool],
            _ => &[Threads::One, Threads::Two, Threads::Pool],
        }
    }

    fn label(self) -> &'static str {
        match self {
            Threads::AsRequested => "",
            Threads::One => "1",
            Threads::Two => "2",
            Threads::Pool => "pool",
        }
    }

    /// `(rvc_preset, worker_priority)` of the policy for a request preset and the voice changer's
    /// worker priority (`None` = unchanged).
    fn params(self, preset: &str, worker_priority: &str) -> (Option<String>, Option<&'static str>) {
        let pooled = || {
            let p = normalize_rvc_preset(preset);
            Some(if p == "streamer" { "low_latency".to_string() } else { p.to_string() })
        };
        match self {
            Threads::AsRequested => (None, None),
            Threads::One => (Some("streamer".into()), None),
            Threads::Two => (pooled(), Some("protect_game")),
            Threads::Pool => (pooled(), Some(if worker_priority == "protect_voice" { "protect_voice" } else { "auto" })),
        }
    }
}

impl Config {
    fn levers_on(&self) -> String {
        self.levers
            .iter()
            .filter(|(_, m)| m.as_str() == "on")
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(",")
    }

    fn lever_is_on(&self, lever: Lever) -> bool {
        self.levers.get(lever.as_str()).map(String::as_str) == Some("on")
    }
}

/// The ladder rung an active EP id runs on (the TensorRT accelerators on the CUDA rung).
fn rung_of_active(active: &str) -> &'static str {
    match active.trim().to_ascii_lowercase().as_str() {
        "tensorrt_rtx" | "tensorrt" => "cuda",
        other => base_ep(other),
    }
}

/// Session components of the voice's generator (template voices: `vc.enc`, `vc.flowdec`;
/// bring-your-own ONNX voices: `vc`), without the benchmark's `bench.` prefix.
const GENERATOR_COMPONENTS: [&str; 3] = ["vc", "vc.enc", "vc.flowdec"];

/// Where a built voice's sessions landed, as the tuner judges an EP.
#[derive(Clone, Debug, PartialEq)]
enum Placement {
    /// The generator runs on the rung asked for (`active`: its EP, e.g. `tensorrt_rtx`); `notes`:
    /// other components that fell back (measured as they run).
    OnRung { active: String, notes: Vec<String> },
    /// The generator runs on another rung, or the accelerator asked for did not run: the EP is
    /// dropped.
    EpFellBack { active: String, reason: String },
    /// The F0 method's model fell off its rung: that method is skipped on this EP.
    PitchFellBack { active: String, reason: String },
}

/// How a voice built for `ep` landed (see [`Placement`]): decided by the generator sessions only;
/// a pitch model that falls back fails its F0 method; a component asked for on the CPU (the small
/// pitch models) is never a fallback.
fn placement(info: &EngineInfo, ep: &str) -> Placement {
    let rung = base_ep(ep);
    let wants_trt = normalize_execution_provider(ep) == "tensorrt_rtx";
    let reason_of = |o: &EpOutcome| o.reason.clone().unwrap_or_else(|| "fallback".to_string());
    let mut generator: Vec<(&str, &EpOutcome)> = Vec::new();
    let mut pitch: Option<&EpOutcome> = None;
    let mut notes = Vec::new();
    for (key, o) in &info.ep_outcomes {
        let bare = key.strip_prefix("bench.").unwrap_or(key);
        let off_rung = rung_of_active(&o.active) != rung_of_active(&o.requested);
        if GENERATOR_COMPONENTS.contains(&bare) {
            generator.push((bare, o));
        } else if bare == "vc.pitch" {
            if off_rung {
                pitch = Some(o);
            }
        } else if off_rung {
            notes.push(format!("{bare} runs on {} ({})", o.active, reason_of(o)));
        }
    }
    let active = if generator.is_empty() {
        // No per-session rows: the engine's own summary.
        let active = info.execution_provider_active.clone().unwrap_or_else(|| rung.to_string());
        if rung_of_active(&active) != rung {
            let reason = info.execution_provider_reason.clone().unwrap_or_else(|| "fallback".to_string());
            return Placement::EpFellBack { active, reason };
        }
        active
    } else {
        if let Some((_, o)) = generator.iter().find(|(_, o)| rung_of_active(&o.active) != rung) {
            return Placement::EpFellBack {
                active: o.active.clone(),
                reason: reason_of(o),
            };
        }
        // An accelerated session names the EP (TensorRT on one plan), else the decoder's.
        generator
            .iter()
            .find(|(_, o)| o.active.starts_with("tensorrt"))
            .or_else(|| generator.iter().find(|(k, _)| *k == "vc.flowdec"))
            .unwrap_or(&generator[0])
            .1
            .active
            .clone()
    };
    if wants_trt {
        if !active.starts_with("tensorrt") {
            let why = generator
                .iter()
                .find_map(|(_, o)| o.reason.clone())
                .or_else(|| info.execution_provider_reason.clone())
                .unwrap_or_else(|| "it did not apply to the voice's sessions".to_string());
            return Placement::EpFellBack {
                active,
                reason: format!("TensorRT-RTX did not run: {why}; the CUDA rung is tuned as cuda"),
            };
        }
        if active == EP_TRT_CLASSIC {
            notes.push("classic TensorRT runs instead of TensorRT-RTX".to_string());
        }
    }
    if let Some(o) = pitch {
        return Placement::PitchFellBack {
            active: o.active.clone(),
            reason: reason_of(o),
        };
    }
    Placement::OnRung { active, notes }
}

/// The tuner's fixed context.
struct Tuner<'a> {
    base: RvcSpec,
    stream: StreamSpec,
    template: bool,
    clip: Vec<f32>,
    /// The synthetic voice (device rate, mono) and its exact F0: the F0 methods' accuracy.
    truth_clip: Vec<f32>,
    truth: Vec<Option<f32>>,
    rtf_limit: f64,
    measure_s: f64,
    quick: bool,
    deadline: Instant,
    cancel: &'a AtomicBool,
    progress: &'a (dyn Fn(&Progress) + Sync),
    candidates: Vec<Candidate>,
    notes: Vec<String>,
    measured: usize,
    planned: usize,
    /// The voice changer's worker priority (the CPU thread stage keeps an explicit one).
    worker_priority: String,
}

/// Why a configuration did not run where it was asked to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FellBack {
    /// The generator (or the accelerator asked for) is not on the EP: drop the EP.
    Ep,
    /// The F0 method's model is not: skip the method.
    Pitch,
}

/// A measured configuration.
#[derive(Clone, Debug)]
struct Measured {
    cfg: Config,
    cand: Candidate,
    /// The generator's EP (e.g. `tensorrt_rtx`).
    active: Option<String>,
    fell_back: Option<FellBack>,
}

impl Measured {
    fn rtf(&self) -> f64 {
        self.cand.rtf_p99.unwrap_or(f64::INFINITY)
    }
    fn latency(&self) -> f64 {
        self.cand.latency_ms_p50.unwrap_or(f64::INFINITY)
    }
}

impl Tuner<'_> {
    fn check(&self) -> Result<(), String> {
        if self.cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        Ok(())
    }

    fn out_of_time(&self) -> bool {
        Instant::now() >= self.deadline
    }

    fn report(&self, stage: &str, message: &str) {
        (self.progress)(&Progress {
            run: "autotune".into(),
            run_index: 0,
            run_count: 1,
            fraction: (self.measured as f64 / self.planned.max(1) as f64).min(0.95),
            message: format!("{stage}: {message}"),
        });
    }

    fn variant(&self, cfg: &Config) -> (RvcSpec, StreamSpec) {
        let mut r = self.base.clone();
        r.execution_provider = cfg.ep.clone();
        r.pitch_extractor = cfg.pitch.clone();
        r.precision = cfg.precision.clone();
        r.levers = cfg.levers.clone();
        r.block_ms = cfg.block_ms;
        // Every tick converts (worst case; the silence skip only ever saves work).
        r.silence_skip = false;
        let mut s = self.stream.clone();
        s.block_frames = cfg.chunk_frames;
        s.extra_context_ms = cfg.extra_ms;
        let (preset, priority) = cfg.threads.params(&self.base.preset, &self.worker_priority);
        if let Some(p) = preset {
            r.preset = p;
        }
        if let Some(p) = priority {
            s.worker_priority = p.to_string();
        }
        (r, s)
    }

    /// Engine block (ms) of a configuration.
    fn block_of(&self, cfg: &Config) -> f64 {
        if self.template {
            cfg.block_ms
        } else {
            f64::from(cfg.chunk_frames) * 1000.0 / f64::from(self.stream.input_rate)
        }
    }

    /// Build and run one configuration on the virtual clock.
    fn measure(&mut self, stage: &str, cfg: &Config) -> Result<Measured, String> {
        self.measure_with(stage, cfg, None)
    }

    /// [`Tuner::measure`] with a load running on the side (the lever stage).
    fn measure_with(&mut self, stage: &str, cfg: &Config, load: Option<&LoadSpec>) -> Result<Measured, String> {
        self.check()?;
        self.measured += 1;
        let label = format!(
            "{} {} {:.0} ms ctx {:.0} {}{}{}",
            cfg.ep,
            cfg.pitch,
            self.block_of(cfg),
            cfg.extra_ms,
            cfg.precision,
            if cfg.levers_on().is_empty() { String::new() } else { format!(" +{}", cfg.levers_on()) },
            if cfg.threads == Threads::AsRequested { String::new() } else { format!(" threads {}", cfg.threads.label()) }
        );
        self.report(stage, &label);
        let mut cand = Candidate {
            stage: stage.into(),
            ep: cfg.ep.clone(),
            pitch: cfg.pitch.clone(),
            block_ms: self.block_of(cfg),
            chunk_frames: cfg.chunk_frames,
            extra_context_ms: cfg.extra_ms,
            precision: cfg.precision.clone(),
            levers: cfg.levers_on(),
            threads: cfg.threads.label().to_string(),
            under_load: load.is_some(),
            ..Candidate::default()
        };
        let (rvc, stream) = self.variant(cfg);
        let priority = WorkerPriority::from_str_lossy(&stream.worker_priority);
        let engine = match build_engine(&EngineSpec::Rvc(rvc), &stream, priority, 0x5167_7e57) {
            Ok(e) => e,
            Err(e) => {
                cand.note = e;
                self.candidates.push(cand.clone());
                return Ok(Measured {
                    cfg: cfg.clone(),
                    cand,
                    active: None,
                    fell_back: None,
                });
            }
        };
        let (active, fell_back, placement_notes) = match placement(&engine.info, &cfg.ep) {
            Placement::OnRung { active, notes } => (active, None, notes),
            Placement::EpFellBack { active, reason } => {
                cand.note = format!("the voice runs on {active} ({reason})");
                (active, Some(FellBack::Ep), Vec::new())
            }
            Placement::PitchFellBack { active, reason } => {
                cand.note = format!("the {} pitch model runs on {active} ({reason})", cfg.pitch);
                (active, Some(FellBack::Pitch), Vec::new())
            }
        };
        let active = Some(active);
        if fell_back.is_some() {
            self.candidates.push(cand.clone());
            return Ok(Measured {
                cfg: cfg.clone(),
                cand,
                active,
                fell_back,
            });
        }
        let probe = match probe_algorithmic_latency(&stream, engine.info.output_rate, engine.info.min_context_samples, engine.fixed_block(), engine.info.model_delay_ms) {
            Ok(p) => p,
            Err(e) => {
                cand.note = e;
                self.candidates.push(cand.clone());
                return Ok(Measured {
                    cfg: cfg.clone(),
                    cand,
                    active,
                    fell_back: None,
                });
            }
        };
        let input = to_interleaved(&self.clip, usize::from(stream.input_channels));
        let run = PacedRun {
            name: "autotune".into(),
            seconds: self.measure_s,
            clock: ClockKind::Virtual,
            warmup_s: 0.5,
            load: load.cloned().unwrap_or_default(),
            gpu_load_kind: GpuLoadKind::Conv,
            stalls: Vec::new(),
            thresholds: Default::default(),
        };
        let measure_once = || {
            run_paced(&PacedInput {
                stream: &stream,
                run: &run,
                engine: &engine,
                input: &input,
                probe,
                record_audio: false,
                cancel: self.cancel,
                progress: &|_, _| {},
            })
        };
        let mut outcome = measure_once();
        // A miss on the tail while the typical call is fast is measured once more, and the better
        // run counts: a transient hiccup of the PC must not decide the preset (a real problem
        // misses twice).
        let mut remeasured = false;
        if let Ok(first) = &outcome {
            if worth_remeasuring(&first.metrics, self.rtf_limit) {
                self.check()?;
                let again = measure_once();
                if let Ok(second) = &again {
                    remeasured = true;
                    if judge(&second.metrics, self.rtf_limit).better_than(&judge(&first.metrics, self.rtf_limit)) {
                        outcome = again;
                    }
                }
            }
        }
        match outcome {
            Ok(o) => {
                let m = &o.metrics;
                let v = judge(m, self.rtf_limit);
                cand.rtf_p99 = v.rtf_p99;
                cand.compute_ms_p95 = m.get("call_ms_p95").or_else(|| m.get("block_ms_p95")).copied();
                cand.latency_ms_p50 = m.get("latency_ms_p50").copied();
                cand.latency_ms_p99 = m.get("latency_ms_p99").copied();
                cand.over_budget_ratio = m.get("over_budget_ratio").copied();
                cand.cpu_cores = m.get("cpu_cores").copied();
                cand.ok = v.ok;
                if v.errors > 0.0 {
                    cand.note = format!("{} model errors", v.errors);
                } else if !cand.ok {
                    cand.note = format!("p99 RTF above {}", self.rtf_limit);
                }
                if remeasured {
                    cand.note = format!("{} (measured twice, the better run kept)", cand.note).trim_start().to_string();
                }
            }
            Err(e) if e == "cancelled" => return Err(e),
            Err(e) => cand.note = e,
        }
        for n in placement_notes {
            cand.note = if cand.note.is_empty() { n } else { format!("{}; {n}", cand.note) };
        }
        self.candidates.push(cand.clone());
        Ok(Measured {
            cfg: cfg.clone(),
            cand,
            active,
            fell_back: None,
        })
    }

    /// F0 methods to try on `ep`, most accurate first (pitch-stage rows recorded).
    fn rank_pitch(&mut self, ep: &str, wanted: &[String]) -> Vec<String> {
        let rung = base_ep(ep);
        let caps = capabilities();
        let pool: Vec<String> = if wanted.is_empty() {
            caps.pitch_extractors.iter().map(|p| p.to_string()).collect()
        } else {
            wanted.iter().map(|p| normalize_pitch_extractor(p).to_string()).collect()
        };
        let mut methods: Vec<String> = Vec::new();
        for p in pool {
            let (run, _) = effective_pitch_extractor(&p, rung);
            if !methods.contains(&run) {
                methods.push(run);
            }
        }
        // The pitch bank's quality order (P3-D's A/B) breaks ties; the accuracy on the synthetic
        // voice below decides (it measures each method as it runs on this rung, e.g. the fusion's
        // FCPE on the CPU).
        let prior = crate::services::neural_audio::vc::f0_fusion::ab::METHODS;
        methods.sort_by_key(|m| prior.iter().position(|p| p == m).unwrap_or(prior.len()));
        let truth = self.truth.clone();
        let clip16 = resample(&self.truth_clip, self.stream.input_rate, 16_000).unwrap_or_default();
        let opts = VcSessionOptions {
            intra_threads: Some(2),
            component_prefix: Some("bench"),
            ..VcSessionOptions::default()
        };
        let app_data = crate::services::diagnostics::app_data_dir();
        let mut scored: Vec<(f64, f64, String)> = Vec::new();
        for m in &methods {
            let mut cand = Candidate {
                stage: "pitch".into(),
                ep: ep.to_string(),
                pitch: m.clone(),
                ..Candidate::default()
            };
            let fcpe = self.base.fcpe.trim();
            let rmvpe = self.base.rmvpe.trim();
            let req = PitchBankRequest {
                method: m,
                ep: rung,
                opts: &opts,
                range: F0Range::from_str_lossy(&self.base.f0_range),
                fcpe_path: (!fcpe.is_empty()).then(|| Path::new(fcpe)),
                rmvpe_path: (!rmvpe.is_empty()).then(|| Path::new(rmvpe)),
                app_data: app_data.as_deref(),
                small_models_on_cpu: true,
            };
            match create_pitch_extractor(&req).and_then(|mut b| b.extractor.f0(&clip16)) {
                Ok(f0) => {
                    let acc = pitch_accuracy(&truth, &f0);
                    cand.gross_pitch_error = Some(acc.0);
                    cand.vuv_error = Some(acc.1);
                    cand.f0_rmse_cents = Some(acc.2);
                    cand.ok = true;
                    scored.push((acc.0 + acc.1, acc.2, m.clone()));
                }
                Err(e) => cand.note = e,
            }
            self.candidates.push(cand);
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.total_cmp(&b.1)));
        scored.into_iter().map(|s| s.2).collect()
    }
}

/// How one measurement meets the limit (`p99 RTF <= limit`, at most 0.1 % of the calls over
/// budget, no model errors, latency measured).
#[derive(Clone, Copy, Debug)]
struct Judgement {
    ok: bool,
    rtf_p99: Option<f64>,
    errors: f64,
}

impl Judgement {
    /// Meeting the limit wins, then the lower p99 RTF.
    fn better_than(&self, other: &Judgement) -> bool {
        match (self.ok, other.ok) {
            (true, false) => true,
            (false, true) => false,
            _ => self.rtf_p99.unwrap_or(f64::INFINITY) < other.rtf_p99.unwrap_or(f64::INFINITY),
        }
    }
}

fn judge(m: &super::metrics::Metrics, rtf_limit: f64) -> Judgement {
    let rtf_p99 = m.get("rtf_p99").copied();
    let errors = m.get("model_errors").copied().unwrap_or(0.0);
    let ok = errors == 0.0
        && rtf_p99.is_some_and(|r| r <= rtf_limit)
        && m.get("over_budget_ratio").is_some_and(|&r| r <= 0.001)
        && m.contains_key("latency_ms_p50");
    Judgement { ok, rtf_p99, errors }
}

/// A measurement that missed the limit while its median call met it (and the model never failed):
/// worth one more run. A persistent tail misses twice; a PC busy for a few seconds does not decide
/// the preset (on the shared dev rig one run in five measured a whole candidate 2x slow).
fn worth_remeasuring(m: &super::metrics::Metrics, rtf_limit: f64) -> bool {
    let j = judge(m, rtf_limit);
    !j.ok && j.errors == 0.0 && m.get("rtf_p50").is_some_and(|&p50| p50 <= rtf_limit)
}

/// `(gross pitch error, V/UV error, voiced RMSE cents)` of a contour against the synthetic truth
/// (frames the truth leaves undecided are skipped).
fn pitch_accuracy(truth: &[Option<f32>], f0: &[f32]) -> (f64, f64, f64) {
    let (mut both, mut gross, mut vuv, mut n, mut sq) = (0usize, 0usize, 0usize, 0usize, 0.0f64);
    for (t, &f) in truth.iter().zip(f0) {
        let Some(t) = *t else { continue };
        n += 1;
        let (tv, fv) = (t > 0.0, f > 0.0 && f.is_finite());
        if tv != fv {
            vuv += 1;
        }
        if tv && fv {
            both += 1;
            let ratio = f64::from(f) / f64::from(t);
            if (ratio - 1.0).abs() > super::quality::GROSS_PITCH_RATIO {
                gross += 1;
            }
            let c = 1200.0 * ratio.log2();
            sq += c * c;
        }
    }
    (
        gross as f64 / both.max(1) as f64,
        vuv as f64 / n.max(1) as f64,
        (sq / both.max(1) as f64).sqrt(),
    )
}

/// The tuning clips (mono, at the device rate): the one the engine is timed on (the request's
/// clip, else the synthetic voice), and the synthetic voice with its exact F0 (the F0 ranking).
struct TuningClip {
    clip: Vec<f32>,
    truth_clip: Vec<f32>,
    truth: Vec<Option<f32>>,
}

fn tuning_clip(req: &AutotuneRequest, rate: u32, quick: bool) -> Result<TuningClip, String> {
    let secs = if quick { 5.0 } else { 8.0 };
    let seed = 0x5167_a070;
    let peak = super::signals::db_to_amp(-12.0);
    let truth_clip = synth(SignalKind::Voice, rate, secs, peak, seed);
    let clip = match req.clip.as_deref().map(str::trim).filter(|c| !c.is_empty()) {
        Some(path) => {
            let (x, sr, ch) = read_wav(Path::new(path))?;
            resample(&to_mono(&x, ch), sr, rate)?
        }
        None => truth_clip.clone(),
    };
    Ok(TuningClip {
        clip,
        truth_clip,
        truth: voice_f0_truth(secs, seed),
    })
}

/// Slack of [`pick_pitch_without_limit`]: one measurement's p99 RTF varies by about this much.
const PITCH_RTF_SLACK: f64 = 1.2;

/// The F0 method to go on with when none met the limit at the reference block: the most accurate
/// (the first of `rtfs`, in accuracy order) within 20 % of the fastest one's p99 RTF.
fn pick_pitch_without_limit(rtfs: &[f64]) -> Option<usize> {
    let fastest = rtfs.iter().copied().filter(|r| r.is_finite()).fold(f64::INFINITY, f64::min);
    if !fastest.is_finite() {
        return None;
    }
    rtfs.iter().position(|&r| r.is_finite() && r <= fastest * PITCH_RTF_SLACK)
}

/// Blocks to sweep, ascending: engine blocks (ms) of template voices, or device chunk sizes of
/// legacy ones, from 20 to 100 ms; on the CPU rung up to 200 ms (the engine's longest block), which
/// the ascending search reaches only when no shorter block meets the limit (a slow CPU: a working
/// voice at a higher latency beats one that cannot keep up).
fn block_sweep(template: bool, quick: bool, rate: u32, cpu: bool) -> Vec<(f64, Option<u32>)> {
    let max_ms = if cpu { 201.0 } else { 101.0 };
    if template {
        let mut ms: Vec<f64> = if quick { vec![20.0, 40.0, 80.0] } else { vec![20.0, 40.0, 60.0, 80.0, 100.0] };
        if cpu {
            ms.extend_from_slice(if quick { &[120.0, 200.0] } else { &[120.0, 160.0, 200.0] });
        }
        ms.into_iter().map(|m| (m, None)).collect()
    } else {
        let chunks: Vec<u32> = ALLOWED_CHUNK_FRAMES
            .iter()
            .copied()
            .filter(|&c| {
                let ms = f64::from(c) * 1000.0 / f64::from(rate);
                (19.0..=max_ms).contains(&ms)
            })
            .collect();
        let pick: Vec<u32> = if quick {
            // Every other size, the smallest and the largest included.
            let mut v: Vec<u32> = chunks.iter().copied().step_by(2).collect();
            if let Some(&last) = chunks.last() {
                if v.last() != Some(&last) {
                    v.push(last);
                }
            }
            v
        } else {
            chunks
        };
        pick.into_iter().map(|c| (0.0, Some(c))).collect()
    }
}

/// The reference block of an EP rung (template ms / legacy chunk).
fn reference_block(template: bool, rung: &str, rate: u32) -> (f64, Option<u32>) {
    let ms = if rung == "cpu" { 80.0 } else { 40.0 };
    if template {
        (ms, None)
    } else {
        let want = ms * f64::from(rate) / 1000.0;
        let c = ALLOWED_CHUNK_FRAMES
            .iter()
            .copied()
            .min_by(|a, b| (f64::from(*a) - want).abs().total_cmp(&(f64::from(*b) - want).abs()))
            .unwrap_or(2048);
        (0.0, Some(c))
    }
}

/// The best measured configuration of one EP.
fn tune_ep(t: &mut Tuner<'_>, ep: &str, req: &AutotuneRequest) -> Result<Option<Measured>, String> {
    let rung = base_ep(ep);
    let rate = t.stream.input_rate;
    let device_chunk = req.chunk_frames.unwrap_or(t.stream.block_frames);
    let (ref_ms, ref_chunk) = reference_block(t.template, rung, rate);
    let base_extra = 64.0;
    let mut threads = Threads::AsRequested;
    let mk = |pitch: &str, block_ms: f64, chunk: Option<u32>, extra: f64, precision: &str, levers: &BTreeMap<String, String>, threads: Threads| Config {
        ep: ep.to_string(),
        pitch: pitch.to_string(),
        block_ms,
        chunk_frames: chunk.unwrap_or(device_chunk),
        extra_ms: extra,
        precision: precision.to_string(),
        levers: levers.clone(),
        threads,
        tuned: Vec::new(),
    };
    // Every lever as the voice changer has it (`auto` / `on` / `off`): stages 2-5 measure what runs;
    // the lever stage then decides the GPU levers it measures.
    let levers: BTreeMap<String, String> = t.base.levers.clone();
    // 2. F0 method: most accurate first, the first whose engine meets the limit.
    let ranked = t.rank_pitch(ep, &req.pitch_extractors);
    // 1b. CPU threads: the fewest that meet the limit on the reference configuration (with the most
    // accurate F0 method), else the fastest policy.
    if rung == "cpu" {
        if let Some(pitch) = ranked.first() {
            let mut fastest: Option<Measured> = None;
            let mut met = None;
            for &policy in Threads::sweep(&t.worker_priority) {
                if t.out_of_time() {
                    break;
                }
                let m = t.measure("threads", &mk(pitch, ref_ms, ref_chunk, base_extra, "auto", &levers, policy))?;
                if m.fell_back == Some(FellBack::Ep) {
                    t.notes.push(format!("{ep}: {}", m.cand.note));
                    return Ok(None);
                }
                if m.cand.ok {
                    met = Some(policy);
                    break;
                }
                if m.cand.rtf_p99.is_some() && fastest.as_ref().is_none_or(|f| m.rtf() < f.rtf()) {
                    fastest = Some(m);
                }
            }
            if let Some(policy) = met.or_else(|| fastest.map(|f| f.cfg.threads)) {
                threads = policy;
            }
        }
    }
    let mut chosen: Option<Measured> = None;
    let mut measured: Vec<Measured> = Vec::new();
    for pitch in &ranked {
        if t.out_of_time() {
            break;
        }
        let m = t.measure("pitch", &mk(pitch, ref_ms, ref_chunk, base_extra, "auto", &levers, threads))?;
        match m.fell_back {
            Some(FellBack::Ep) => {
                // The EP itself fell back: nothing on this EP is worth measuring.
                t.notes.push(format!("{ep}: {}", m.cand.note));
                return Ok(None);
            }
            Some(FellBack::Pitch) => {
                // This F0 method cannot run here; the next most accurate one may.
                t.notes.push(format!("{ep}: {pitch} skipped: {}", m.cand.note));
                continue;
            }
            None => {}
        }
        if m.cand.ok {
            chosen = Some(m);
            break;
        }
        if m.cand.rtf_p99.is_some() {
            measured.push(m);
        }
    }
    let fallback = || {
        let rtfs: Vec<f64> = measured.iter().map(Measured::rtf).collect();
        pick_pitch_without_limit(&rtfs).map(|i| measured[i].clone())
    };
    let Some(mut best) = chosen.or_else(fallback) else {
        t.notes.push(format!("{ep}: no configuration ran"));
        return Ok(None);
    };
    let pitch = best.cfg.pitch.clone();
    // 3. Blocks, ascending: the first that meets the limit.
    let mut last_block: Option<Measured> = None;
    for (block_ms, chunk) in block_sweep(t.template, t.quick, rate, rung == "cpu") {
        if t.out_of_time() {
            break;
        }
        let cfg = mk(&pitch, block_ms, chunk, base_extra, "auto", &levers, threads);
        let m = if cfg == best.cfg { best.clone() } else { t.measure("block", &cfg)? };
        if m.cand.ok {
            best = m;
            break;
        }
        if m.cand.rtf_p99.is_some() {
            last_block = Some(m);
        }
    }
    if !best.cand.ok {
        if let Some(l) = last_block.filter(|l| l.rtf() < best.rtf()) {
            best = l;
        }
    }
    // 4. Context: the largest that still meets the limit.
    let extras: &[f64] = if t.quick { &[256.0] } else { &[128.0, 256.0, 512.0] };
    for &extra in extras {
        if t.out_of_time() || !best.cand.ok {
            break;
        }
        let mut cfg = best.cfg.clone();
        cfg.extra_ms = extra;
        let m = t.measure("context", &cfg)?;
        if m.cand.ok {
            best = m;
        } else {
            break;
        }
    }
    // 5. Precision (template voices on a GPU): fp32 when within 5 % of fp16.
    if t.template && rung != "cpu" && !t.out_of_time() {
        let mut cfg = best.cfg.clone();
        cfg.precision = "fp32".into();
        let m = t.measure("precision", &cfg)?;
        if m.cand.ok && m.rtf() <= best.rtf() * 1.05 {
            best = m;
        }
    }
    // 6. GPU levers under the H2 load.
    if t.template && rung != "cpu" && !t.out_of_time() {
        best = tune_levers(t, ep, best)?;
    }
    Ok(Some(best))
}

/// Stage 6 (template voices on a GPU): from every usable GPU lever off, each lever set kept when
/// it lowers the loaded p99 RTF by 5 % or more. The levers the stage measured both ways are then
/// decided by it, and that configuration is measured idle again like every other; a lever it never
/// tried keeps the setting `best` measured with (the voice changer's own).
fn tune_levers(t: &mut Tuner<'_>, ep: &str, best: Measured) -> Result<Measured, String> {
    let rung = base_ep(ep);
    let usable: Vec<Lever> = GPU_LEVERS
        .into_iter()
        .filter(|&l| lever_available(l) && lever_applies(l, rung))
        .collect();
    let sets = lever_sets(|l| usable.contains(&l));
    if sets.is_empty() {
        return Ok(best);
    }
    let load = super::contention::h2_load(rung);
    let mut baseline = best.cfg.clone();
    for l in &usable {
        baseline.levers.insert(l.as_str().into(), "off".into());
    }
    let mut kept = t.measure_with("lever", &baseline, Some(&load))?;
    let mut tried: Vec<Lever> = Vec::new();
    for set in sets {
        if t.out_of_time() {
            break;
        }
        // IOBinding alone only after the pair with CUDA graphs was not kept.
        if set == [Lever::IoBinding] && kept.cfg.lever_is_on(Lever::IoBinding) {
            continue;
        }
        let mut cfg = kept.cfg.clone();
        for l in &set {
            cfg.levers.insert(l.as_str().into(), "on".into());
        }
        let m = t.measure_with("lever", &cfg, Some(&load))?;
        for l in set {
            if !tried.contains(&l) {
                tried.push(l);
            }
        }
        if m.cand.rtf_p99.is_some() && m.rtf() <= kept.rtf() * 0.95 {
            kept = m;
        }
    }
    if tried.is_empty() {
        t.notes.push(format!("{ep}: the GPU levers were not tuned (out of time); they keep their current settings"));
        return Ok(best);
    }
    let mut cfg = kept.cfg.clone();
    for l in &usable {
        if !tried.contains(l) {
            match best.cfg.levers.get(l.as_str()) {
                Some(v) => cfg.levers.insert(l.as_str().into(), v.clone()),
                None => cfg.levers.remove(l.as_str()),
            };
        }
    }
    cfg.tuned = usable.iter().filter(|l| tried.contains(l)).map(|l| l.as_str()).collect();
    if cfg.levers == best.cfg.levers {
        // The settings it measured with already are the stage's choice.
        let mut same = best;
        same.cfg.tuned = cfg.tuned;
        return Ok(same);
    }
    let m = t.measure("lever", &cfg)?;
    if m.cand.rtf_p99.is_some() && (m.cand.ok || !best.cand.ok) {
        Ok(m)
    } else {
        t.notes.push(format!(
            "{ep}: the GPU levers chosen under load ({}) did not hold idle; they keep their current settings",
            if cfg.levers_on().is_empty() { "all off".to_string() } else { cfg.levers_on() }
        ));
        Ok(best)
    }
}

/// The GPU lever sets the lever stage tries, in order (those `usable` on the rung): IOBinding with
/// CUDA graphs (graphs need IOBinding), IOBinding alone, the priority stream / queue.
fn lever_sets(usable: impl Fn(Lever) -> bool) -> Vec<Vec<Lever>> {
    let mut sets = Vec::new();
    if usable(Lever::IoBinding) {
        if usable(Lever::CudaGraphs) {
            sets.push(vec![Lever::IoBinding, Lever::CudaGraphs]);
        }
        sets.push(vec![Lever::IoBinding]);
    }
    if usable(Lever::GpuPriority) {
        sets.push(vec![Lever::GpuPriority]);
    }
    sets
}

/// A configuration's fidelity against its offline ceiling.
struct Verdict {
    /// Every measured T5 check passed.
    pass: bool,
    /// CAM++ cosine, when CAM++ is installed.
    cosine: Option<f64>,
    metrics: BTreeMap<String, f64>,
}

/// How far fidelity metrics fall short of T5, each in units of its own limit (0 = every measured
/// check passes): speaker cosine below 0.97 per 0.03, voiced F0 RMSE above 15 cents per 15 cents,
/// V/UV error above 3 % per 3 %, one per seam click. Unmeasured metrics count 0.
fn t5_shortfall(m: &BTreeMap<String, f64>) -> f64 {
    let get = |k: &str| m.get(k).copied().filter(|v| v.is_finite());
    get("speaker_cosine").map_or(0.0, |c| ((0.97 - c) / 0.03).max(0.0))
        + get("f0_rmse_cents").map_or(0.0, |v| ((v - 15.0) / 15.0).max(0.0))
        + get("vuv_error").map_or(0.0, |v| ((v - 0.03) / 0.03).max(0.0))
        + get("clicks").map_or(0.0, |v| v.max(0.0))
}

/// One verified configuration, in ascending block order.
#[derive(Clone, Copy, Debug)]
struct VerifiedAttempt {
    pass: bool,
    shortfall: f64,
    /// It met the RTF limit.
    meets_limit: bool,
}

/// Which verified attempt to keep: never one that misses the RTF limit when the first met it; then
/// the first that passes T5; else the smallest shortfall, where a larger block must cut the kept
/// one's shortfall by a quarter to be worth its latency.
fn pick_verified(attempts: &[VerifiedAttempt]) -> usize {
    let first_ok = attempts.first().is_some_and(|a| a.meets_limit);
    let eligible = |a: &VerifiedAttempt| a.meets_limit || !first_ok;
    if let Some(i) = attempts.iter().position(|a| eligible(a) && a.pass) {
        return i;
    }
    let mut best = 0;
    for (i, a) in attempts.iter().enumerate().skip(1) {
        if eligible(a) && a.shortfall < attempts[best].shortfall * 0.75 {
            best = i;
        }
    }
    best
}

/// Fidelity of a configuration against its offline ceiling (T5 checks).
fn verify(t: &mut Tuner<'_>, cfg: &Config) -> Result<Verdict, String> {
    t.check()?;
    t.report("verifying", &format!("{} {:.0} ms", cfg.ep, t.block_of(cfg)));
    let (rvc, stream) = t.variant(cfg);
    let spec = BenchSpec {
        schema: super::spec::SPEC_SCHEMA.into(),
        name: "autotune-verify".into(),
        description: String::new(),
        seed: 0x5167_7e57,
        input: InputSpec::default(),
        stream: stream.clone(),
        engine: EngineSpec::Rvc(rvc.clone()),
        runs: Vec::new(),
        thresholds: Default::default(),
        output: Default::default(),
    };
    let priority = WorkerPriority::from_str_lossy(&stream.worker_priority);
    let engine = build_engine(&spec.engine, &stream, priority, spec.seed)?;
    let probe = probe_algorithmic_latency(&stream, engine.info.output_rate, engine.info.min_context_samples, engine.fixed_block(), engine.info.model_delay_ms)?;
    let prepared = Prepared {
        engine,
        probe,
        input: to_interleaved(&t.clip, usize::from(stream.input_channels)),
        vram_engine_mb: None,
    };
    let q = QualityRun {
        name: "verify".into(),
        lead_s: 1.0,
        ceiling: Default::default(),
        scoring: ScoringSpec::default(),
        stt: None,
        thresholds: Default::default(),
    };
    let o = super::fidelity::run_quality(&q, &spec, &prepared, false, t.cancel, &|_, _| {})?;
    let m = o.metrics;
    let get = |k: &str| m.get(k).copied();
    let cos = get("speaker_cosine");
    let pass = cos.is_none_or(|c| c >= 0.97)
        && get("f0_rmse_cents").is_none_or(|v| v <= 15.0)
        && get("vuv_error").is_none_or(|v| v <= 0.03)
        && get("clicks").is_none_or(|v| v == 0.0);
    let keep = ["speaker_cosine", "mcd_db", "f0_rmse_cents", "gross_pitch_error", "vuv_error", "clicks", "artifact_events", "dnsmos_ovrl"];
    let metrics: BTreeMap<String, f64> = keep.iter().filter_map(|k| get(k).map(|v| (k.to_string(), v))).collect();
    let mut cand = Candidate {
        stage: "verify".into(),
        ep: cfg.ep.clone(),
        pitch: cfg.pitch.clone(),
        block_ms: t.block_of(cfg),
        chunk_frames: cfg.chunk_frames,
        extra_context_ms: cfg.extra_ms,
        precision: cfg.precision.clone(),
        levers: cfg.levers_on(),
        fidelity: cos,
        ok: pass,
        ..Candidate::default()
    };
    if !pass {
        cand.note = format!("fidelity below T5: {metrics:?}");
    }
    t.candidates.push(cand);
    Ok(Verdict {
        pass,
        cosine: cos,
        metrics,
    })
}

/// The params patch of a configuration (`preset` / `worker_priority`: the voice changer's, for the
/// CPU thread policy). GPU levers only when the lever stage measured them (`cfg.tuned`); every other
/// lever keeps the voice changer's setting, which is what the configuration measured with.
fn params_of(cfg: &Config, template: bool, preset: &str, worker_priority: &str) -> serde_json::Map<String, serde_json::Value> {
    let mut p = serde_json::Map::new();
    let (rvc_preset, priority) = cfg.threads.params(preset, worker_priority);
    if let Some(v) = rvc_preset {
        p.insert("rvc_preset".into(), v.into());
    }
    if let Some(v) = priority {
        p.insert("worker_priority".into(), v.into());
    }
    p.insert("rvc_execution_provider".into(), cfg.ep.clone().into());
    p.insert("pitch_extractor".into(), cfg.pitch.clone().into());
    p.insert("extra_context_ms".into(), cfg.extra_ms.into());
    if template {
        p.insert("block_ms".into(), cfg.block_ms.into());
        p.insert("precision".into(), cfg.precision.clone().into());
        for &lever in &cfg.tuned {
            if let Some(v) = cfg.levers.get(lever) {
                p.insert(lever.to_string(), v.clone().into());
            }
        }
    } else {
        p.insert("chunk_frames".into(), cfg.chunk_frames.into());
    }
    p
}

/// The sweep over a base voice (`base.slot_dir`) and stream.
fn tune(req: &AutotuneRequest, mut base: RvcSpec, mut stream: StreamSpec, cancel: &AtomicBool, progress: &(dyn Fn(&Progress) + Sync)) -> Result<AutotuneResult, String> {
    req.validate()?;
    let started = Instant::now();
    if let Some(r) = req.device_rate {
        // A device running at one rate both ways moves both.
        if stream.output_rate == stream.input_rate {
            stream.output_rate = r;
        }
        stream.input_rate = r;
    }
    if let Some(c) = req.chunk_frames {
        stream.block_frames = c;
    }
    stream.input_channels = 1;
    for (field, v) in [(&mut base.hubert, &req.hubert), (&mut base.fcpe, &req.fcpe), (&mut base.rmvpe, &req.rmvpe)] {
        if let Some(v) = v.as_deref().filter(|s| !s.trim().is_empty()) {
            *field = v.to_string();
        }
    }
    if let Some(p) = req.preset.as_deref().filter(|s| !s.trim().is_empty()) {
        base.preset = p.to_string();
    }
    if !base.slot_dir.is_dir() {
        return Err(format!("voice folder not found: {}", base.slot_dir.display()));
    }
    let format = crate::services::voice_changer::factory::slot_format(&base.slot_dir)?;
    let template = format == SlotFormat::SigilRvc;
    if let SlotFormat::Unknown(f) = &format {
        return Err(format!("slot format '{f}' cannot be tuned"));
    }
    let quick = req.quick;
    // What every candidate measures with besides the swept knobs (recorded in the result).
    let conditions = TuneConditions::of(&base, &stream);
    let mut notes: Vec<String> = Vec::new();
    match DenoiserKind::from_str_lossy(&conditions.denoiser) {
        DenoiserKind::Off | DenoiserKind::Gtcrn => {}
        other => notes.push(format!(
            "the {} denoiser runs on the capture copy, outside the measured engine: its cost and latency are not in these numbers",
            other.as_str()
        )),
    }
    let TuningClip { clip, truth_clip, truth } = tuning_clip(req, stream.input_rate, quick)?;
    let eps: Vec<String> = if req.execution_providers.is_empty() {
        capabilities().rvc_execution_providers.iter().map(|e| e.to_string()).collect()
    } else {
        let mut v: Vec<String> = Vec::new();
        for e in &req.execution_providers {
            let n = normalize_execution_provider(e);
            if !n.is_empty() && !v.contains(&n) {
                v.push(n);
            }
        }
        v
    };
    let budget = req.max_seconds.unwrap_or(if quick { 180.0 } else { 600.0 });
    let per_ep = if template { if quick { 9 } else { 14 } } else if quick { 7 } else { 12 };
    let worker_priority = worker_priority_id(&stream.worker_priority);
    let cpu_threads = if eps.iter().any(|e| base_ep(e) == "cpu") { Threads::sweep(&worker_priority).len() } else { 0 };
    let mut t = Tuner {
        base,
        stream,
        template,
        clip,
        truth_clip,
        truth,
        rtf_limit: req.rtf_limit.unwrap_or(0.5),
        measure_s: if quick { 3.0 } else { 6.0 },
        quick,
        deadline: started + std::time::Duration::from_secs_f64(budget),
        cancel,
        progress,
        candidates: Vec::new(),
        notes,
        measured: 0,
        planned: per_ep * eps.len().max(1) + cpu_threads + 2,
        worker_priority,
    };
    let mut winners: Vec<Measured> = Vec::new();
    for ep in &eps {
        if t.out_of_time() {
            t.notes.push(format!("stopped after {budget:.0} s: {ep} and later EPs were not measured"));
            break;
        }
        if let Some(w) = tune_ep(&mut t, ep, req)? {
            winners.push(w);
        }
    }
    // Across EPs: limit met first, then the lowest latency, then the lowest RTF.
    winners.sort_by(|a, b| {
        b.cand
            .ok
            .cmp(&a.cand.ok)
            .then(a.latency().total_cmp(&b.latency()))
            .then(a.rtf().total_cmp(&b.rtf()))
    });
    let Some(mut best) = winners.first().cloned() else {
        return Err(format!(
            "no configuration could run on this PC ({})",
            if t.notes.is_empty() { "every build failed".to_string() } else { t.notes.join("; ") }
        ));
    };
    // 7. Fidelity: step the block up (twice at most) when the winner fails T5; keep a larger
    // block only when it passes or measurably closes the gap (`pick_verified`).
    let mut fidelity = None;
    let mut fidelity_metrics = BTreeMap::new();
    if req.verify_fidelity.unwrap_or(true) {
        let sweep = block_sweep(template, false, t.stream.input_rate, base_ep(&best.cfg.ep) == "cpu");
        let mut attempts: Vec<(Measured, Verdict)> = Vec::new();
        let mut current = best.clone();
        for attempt in 0..3 {
            match verify(&mut t, &current.cfg) {
                Ok(v) => {
                    let pass = v.pass;
                    attempts.push((current.clone(), v));
                    if pass || attempt == 2 {
                        break;
                    }
                    // Next larger block.
                    let cur = t.block_of(&current.cfg);
                    let next = sweep.iter().find(|(ms, chunk)| {
                        let b = if template { *ms } else { f64::from(chunk.unwrap_or(0)) * 1000.0 / f64::from(t.stream.input_rate) };
                        b > cur + 1e-6
                    });
                    let Some(&(ms, chunk)) = next else { break };
                    let mut cfg = current.cfg.clone();
                    if template {
                        cfg.block_ms = ms;
                    } else if let Some(c) = chunk {
                        cfg.chunk_frames = c;
                    }
                    let m = t.measure("block", &cfg)?;
                    if m.cand.rtf_p99.is_none() {
                        break;
                    }
                    current = m;
                }
                Err(e) if e == "cancelled" => return Err(e),
                Err(e) => {
                    t.notes.push(format!("fidelity check: {e}"));
                    break;
                }
            }
        }
        if !attempts.is_empty() {
            let scores: Vec<VerifiedAttempt> = attempts
                .iter()
                .map(|(m, v)| VerifiedAttempt {
                    pass: v.pass,
                    shortfall: t5_shortfall(&v.metrics),
                    meets_limit: m.cand.ok,
                })
                .collect();
            let i = pick_verified(&scores);
            if !scores[i].pass {
                let tried: Vec<String> = attempts
                    .iter()
                    .zip(&scores)
                    .map(|((m, _), s)| format!("{:.0} ms: {:.2}", t.block_of(&m.cfg), s.shortfall))
                    .collect();
                t.notes.push(format!(
                    "fidelity: below T5 at every block tried (shortfall {}); kept the {:.0} ms block (a larger one must cut the shortfall by a quarter to be worth its latency)",
                    tried.join(", "),
                    t.block_of(&attempts[i].0.cfg)
                ));
            }
            let (m, v) = attempts.swap_remove(i);
            best = m;
            fidelity = v.cosine;
            fidelity_metrics = v.metrics;
        }
    }
    if !best.cand.ok {
        t.notes.push(format!(
            "no configuration met p99 RTF <= {} on this PC; this is the fastest one found",
            t.rtf_limit
        ));
    }
    let gpu_rung = base_ep(&best.cfg.ep) != "cpu";
    if template && gpu_rung && best.cfg.tuned.is_empty() && !t.notes.iter().any(|n| n.contains("GPU levers")) {
        t.notes.push("the GPU levers were not tuned; they keep their current settings".into());
    }
    let (gpu, driver) = gpu_identity();
    (t.progress)(&Progress {
        run: "autotune".into(),
        run_index: 0,
        run_count: 1,
        fraction: 1.0,
        message: "done".into(),
    });
    Ok(AutotuneResult {
        gpu,
        driver,
        created_unix_ms: unix_ms(),
        params: params_of(&best.cfg, template, &t.base.preset, &t.worker_priority),
        rtf_p99: best.cand.rtf_p99.unwrap_or(f64::NAN),
        block_ms_p95: best.cand.compute_ms_p95.unwrap_or(f64::NAN),
        latency_ms_p50: best.cand.latency_ms_p50,
        fidelity,
        report_dir: None,
        notes: t.notes,
        schema: RESULTS_SCHEMA.into(),
        met_limit: best.cand.ok,
        slot: Some(t.base.slot_dir.display().to_string()),
        format: Some(if template { "sigil-rvc-1" } else { "rvc" }.into()),
        execution_provider_active: best.active.clone(),
        latency_ms_p99: best.cand.latency_ms_p99,
        fidelity_metrics,
        candidates: t.candidates,
        seconds: started.elapsed().as_secs_f64(),
        sigil_version: env!("CARGO_PKG_VERSION").into(),
        conditions: Some(conditions),
        stale: Vec::new(),
    })
}

/// Run the sweep for the voice in `request.slot_dir` (the IPC path) with `request.conditions`
/// (the voice changer's settings; its defaults when absent) on the voice changer's own engine
/// factory.
pub fn run_autotune(request: &AutotuneRequest, ctx: &BenchContext<'_>) -> Result<AutotuneResult, String> {
    let slot = request
        .slot_dir
        .as_deref()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .ok_or("no voice to tune: select a voice first")?;
    let mut base: RvcSpec = serde_json::from_value(serde_json::json!({ "slot_dir": slot, "build": "factory" }))
        .map_err(|e| format!("autotune: {e}"))?;
    let mut stream = StreamSpec::default();
    if let Some(c) = &request.conditions {
        c.apply(&mut base, &mut stream);
    }
    tune(request, base, stream, ctx.cancel, ctx.progress)
}

/// An `autotune` run of a spec: the spec's `rvc` engine is the base voice.
pub(super) fn run_in_spec(a: &AutotuneRun, spec: &BenchSpec, cancel: &AtomicBool, progress: &(dyn Fn(f64, &str) + Sync)) -> Result<RunOutcome, String> {
    let EngineSpec::Rvc(base) = &spec.engine else {
        return Err("an autotune run needs an rvc engine".into());
    };
    let mut req = a.request.clone();
    let (mut base, mut stream) = (base.clone(), spec.stream.clone());
    if let Some(c) = &req.conditions {
        c.apply(&mut base, &mut stream);
    }
    if req.clip.is_none() {
        if let InputSpec::Wav { path } = &spec.input {
            req.clip = Some(path.to_string_lossy().into_owned());
        }
    }
    let forward = |p: &Progress| progress(p.fraction, &p.message);
    let result = tune(&req, base, stream.clone(), cancel, &forward)?;
    let m = result_metrics(&result, req.device_rate.unwrap_or(stream.input_rate));
    let details = serde_json::json!({
        "result": result,
        "candidates": result.candidates,
        "notes": result.notes,
    });
    Ok(RunOutcome {
        metrics: m,
        details,
        ..RunOutcome::default()
    })
}

/// The report metrics of a result (the `autotune` run mode's names).
pub fn result_metrics(result: &AutotuneResult, device_rate: u32) -> super::metrics::Metrics {
    let mut m = super::metrics::Metrics::new();
    let block = result.params.get("block_ms").and_then(|v| v.as_f64()).or_else(|| {
        result
            .params
            .get("chunk_frames")
            .and_then(|v| v.as_f64())
            .map(|c| c * 1000.0 / f64::from(device_rate.max(1)))
    });
    let mut put = |k: &str, v: Option<f64>| {
        if let Some(v) = v.filter(|v| v.is_finite()) {
            m.insert(k.into(), v);
        }
    };
    put("autotune_block_ms", block);
    put("autotune_extra_context_ms", result.params.get("extra_context_ms").and_then(|v| v.as_f64()));
    put("autotune_rtf_p99", Some(result.rtf_p99));
    put("autotune_compute_ms_p95", Some(result.block_ms_p95));
    put("autotune_latency_ms_p50", result.latency_ms_p50);
    put("autotune_latency_ms_p99", result.latency_ms_p99);
    put("autotune_fidelity", result.fidelity);
    put(
        "autotune_gpu",
        Some(if result.execution_provider_active.as_deref().is_some_and(|e| e != "cpu") { 1.0 } else { 0.0 }),
    );
    put("autotune_candidates", Some(result.candidates.len() as f64));
    put("autotune_seconds", Some(result.seconds));
    m
}

/// A benchmark report folder (`report.json`, `report.md`) of an in-app tune.
pub fn write_result_report(result: &AutotuneResult, dir: &Path) -> Result<(), String> {
    use super::report::{write_report, BenchReport, RunReport, RunStatus, REPORT_SCHEMA};
    let request = serde_json::json!({ "slot": result.slot, "format": result.format });
    let run = RunReport {
        name: "autotune".into(),
        mode: super::spec::RunMode::Autotune,
        status: RunStatus::Passed,
        error: None,
        duration_ms: result.seconds * 1000.0,
        metrics: result_metrics(result, 48_000),
        thresholds: Vec::new(),
        details: serde_json::json!({ "result": result, "candidates": result.candidates, "notes": result.notes }),
        files: Vec::new(),
        higher_is_better: super::report::higher_is_better_of(&result_metrics(result, 48_000)),
    };
    let report = BenchReport {
        schema: REPORT_SCHEMA.into(),
        name: "autotune".into(),
        description: "In-app autotune".into(),
        passed: true,
        failures: Vec::new(),
        started_unix_ms: result.created_unix_ms,
        duration_ms: result.seconds * 1000.0,
        spec_sha256: String::new(),
        spec: request,
        environment: super::environment(),
        engine: None,
        runs: vec![run],
    };
    write_report(dir, &report).map(|_| ())
}

// ---------------------------------------------------------------------------------------------
// Stored results
// ---------------------------------------------------------------------------------------------

#[derive(Debug, Default, Serialize, Deserialize)]
struct ResultsFile {
    schema: String,
    results: Vec<AutotuneResult>,
}

fn results_path(app_data: &Path) -> PathBuf {
    app_data.join("voice-changer").join(RESULTS_FILE)
}

/// The stored results, record by record: one that does not parse is dropped, never the others.
fn read_results(app_data: &Path) -> ResultsFile {
    let Some(v) = std::fs::read(results_path(app_data))
        .ok()
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
    else {
        return ResultsFile::default();
    };
    let results = v
        .get("results")
        .and_then(|r| r.as_array())
        .map(|a| a.iter().filter_map(|r| serde_json::from_value::<AutotuneResult>(r.clone()).ok()).collect())
        .unwrap_or_default();
    ResultsFile {
        schema: v.get("schema").and_then(|s| s.as_str()).unwrap_or(RESULTS_SCHEMA).to_string(),
        results,
    }
}

/// Store `result` for its GPU and driver (replacing an earlier one), atomically.
pub fn store_result(app_data: &Path, result: &AutotuneResult) -> Result<(), String> {
    let mut file = read_results(app_data);
    file.schema = RESULTS_SCHEMA.into();
    file.results.retain(|r| !(r.gpu == result.gpu && r.driver == result.driver));
    let mut stored = result.clone();
    // The candidate table stays in the report folder; the stored record is the decision.
    stored.candidates.clear();
    stored.stale.clear();
    file.results.push(stored);
    let path = results_path(app_data);
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let json = serde_json::to_vec_pretty(&file).map_err(|e| e.to_string())?;
    super::report::write_atomic(&path, &json)
}

/// The stored result for this PC's GPU and driver, if any.
pub fn stored_result(app_data: &Path) -> Option<AutotuneResult> {
    let (gpu, driver) = gpu_identity();
    read_results(app_data)
        .results
        .into_iter()
        .rev()
        .find(|r| r.gpu == gpu && r.driver == driver)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quiet() -> impl Fn(&Progress) + Sync {
        |_: &Progress| {}
    }

    #[test]
    fn requests_validate_and_accept_the_frozen_shape() {
        // The P3-S0 shape (what the UI sends) still parses.
        let r: AutotuneRequest = serde_json::from_value(serde_json::json!({
            "slot_dir": "C:/voices/a", "execution_providers": ["cuda", "cpu"], "quick": true, "max_seconds": 60
        }))
        .unwrap();
        assert!(r.validate().is_ok());
        let bad: AutotuneRequest = serde_json::from_value(serde_json::json!({"execution_providers": ["vulkan"]})).unwrap();
        assert!(bad.validate().is_err());
        let bad: AutotuneRequest = serde_json::from_value(serde_json::json!({"rtf_limit": 3.0})).unwrap();
        assert!(bad.validate().is_err());
        let bad: AutotuneRequest = serde_json::from_value(serde_json::json!({"chunk_frames": 1000})).unwrap();
        assert!(bad.validate().is_err());
        assert!(available());
    }

    #[test]
    fn block_sweeps_cover_20_to_100_ms() {
        let t = block_sweep(true, false, 48_000, false);
        assert_eq!(t.iter().map(|b| b.0).collect::<Vec<_>>(), vec![20.0, 40.0, 60.0, 80.0, 100.0]);
        let l = block_sweep(false, false, 48_000, false);
        assert!(l.iter().all(|b| {
            let ms = f64::from(b.1.unwrap()) / 48.0;
            (19.0..=101.0).contains(&ms)
        }));
        assert_eq!(l.first().unwrap().1, Some(1024));
        let q = block_sweep(false, true, 48_000, false);
        assert!(q.len() < l.len() && q.last() == l.last());
        // The CPU rung continues up to the engine's longest block, after the 20-100 ms ones.
        let tc = block_sweep(true, false, 48_000, true);
        assert_eq!(tc.iter().map(|b| b.0).collect::<Vec<_>>(), vec![20.0, 40.0, 60.0, 80.0, 100.0, 120.0, 160.0, 200.0]);
        assert_eq!(block_sweep(true, true, 48_000, true).last().unwrap().0, 200.0);
        let lc = block_sweep(false, false, 48_000, true);
        assert!(lc.starts_with(&l) && lc.len() > l.len());
        assert!(lc.iter().all(|b| f64::from(b.1.unwrap()) / 48.0 <= 201.0));
        assert_eq!(reference_block(false, "cpu", 48_000).1, Some(3584), "80 ms at 48 kHz: the nearest chunk");
        assert_eq!(reference_block(true, "dml", 48_000), (40.0, None));
    }

    /// A larger block is kept only when it passes T5 or closes a quarter of the gap, and never
    /// when it misses the RTF limit the smaller one met.
    #[test]
    fn verification_keeps_the_smallest_block_worth_its_latency() {
        let a = |pass, shortfall, meets_limit| VerifiedAttempt { pass, shortfall, meets_limit };
        let t5 = |cos: f64, rmse: f64, vuv: f64, clicks: f64| {
            let m: BTreeMap<String, f64> = [("speaker_cosine", cos), ("f0_rmse_cents", rmse), ("vuv_error", vuv), ("clicks", clicks)]
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect();
            t5_shortfall(&m)
        };
        assert_eq!(t5(0.98, 10.0, 0.01, 0.0), 0.0);
        assert!((t5(0.94, 30.0, 0.06, 1.0) - 4.0).abs() < 1e-9);
        assert_eq!(t5_shortfall(&BTreeMap::new()), 0.0, "unmeasured metrics count 0");
        // The first passing block wins.
        assert_eq!(pick_verified(&[a(false, 2.0, true), a(true, 0.0, true), a(true, 0.0, true)]), 1);
        // Nothing passes: a larger block that barely helps is not worth its latency ...
        assert_eq!(pick_verified(&[a(false, 2.0, true), a(false, 1.8, true), a(false, 1.7, true)]), 0);
        // ... one that closes a quarter of the gap is.
        assert_eq!(pick_verified(&[a(false, 2.0, true), a(false, 1.4, true), a(false, 1.2, true)]), 1);
        assert_eq!(pick_verified(&[a(false, 2.0, true), a(false, 1.4, true), a(false, 1.0, true)]), 2);
        // A larger block that misses the RTF limit is never kept over one that met it.
        assert_eq!(pick_verified(&[a(false, 2.0, true), a(true, 0.0, false)]), 0);
        // When the first missed it too, the limit does not decide.
        assert_eq!(pick_verified(&[a(false, 2.0, false), a(true, 0.0, false)]), 1);
        assert_eq!(pick_verified(&[a(true, 0.0, true)]), 0);
    }

    /// CUDA tries IOBinding with graphs as a pair (graphs need IOBinding) before IOBinding alone;
    /// DirectML has no graphs; a rung without any lever tries nothing.
    #[test]
    fn lever_sets_pair_graphs_with_io_binding() {
        let cuda = lever_sets(|l| matches!(l, Lever::IoBinding | Lever::CudaGraphs | Lever::GpuPriority));
        assert_eq!(cuda, vec![vec![Lever::IoBinding, Lever::CudaGraphs], vec![Lever::IoBinding], vec![Lever::GpuPriority]]);
        let dml = lever_sets(|l| matches!(l, Lever::IoBinding | Lever::GpuPriority));
        assert_eq!(dml, vec![vec![Lever::IoBinding], vec![Lever::GpuPriority]]);
        assert!(lever_sets(|_| false).is_empty());
    }

    /// A tail miss with a median inside the limit is re-measured; a slow median or a model error is not.
    #[test]
    fn only_transient_misses_are_measured_again() {
        let m = |pairs: &[(&str, f64)]| -> super::super::metrics::Metrics { pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect() };
        let fast_tail = m(&[("rtf_p50", 0.2), ("rtf_p99", 0.6), ("over_budget_ratio", 0.0), ("latency_ms_p50", 140.0), ("model_errors", 0.0)]);
        assert!(worth_remeasuring(&fast_tail, 0.5));
        let slow_tail = m(&[("rtf_p50", 0.4), ("rtf_p99", 0.6), ("over_budget_ratio", 0.0), ("latency_ms_p50", 140.0), ("model_errors", 0.0)]);
        assert!(worth_remeasuring(&slow_tail, 0.5), "the median still meets the limit");
        let slow = m(&[("rtf_p50", 0.6), ("rtf_p99", 0.9), ("over_budget_ratio", 0.0), ("latency_ms_p50", 140.0), ("model_errors", 0.0)]);
        assert!(!worth_remeasuring(&slow, 0.5));
        let failing = m(&[("rtf_p50", 0.1), ("rtf_p99", 0.6), ("over_budget_ratio", 0.0), ("latency_ms_p50", 140.0), ("model_errors", 3.0)]);
        assert!(!worth_remeasuring(&failing, 0.5));
        let fine = m(&[("rtf_p50", 0.1), ("rtf_p99", 0.3), ("over_budget_ratio", 0.0), ("latency_ms_p50", 140.0), ("model_errors", 0.0)]);
        assert!(judge(&fine, 0.5).ok && !worth_remeasuring(&fine, 0.5));
        assert!(judge(&fine, 0.5).better_than(&judge(&fast_tail, 0.5)));
        assert!(!judge(&fast_tail, 0.5).better_than(&judge(&fine, 0.5)));
    }

    /// With nothing meeting the limit, the most accurate F0 method within 20 % of the fastest.
    #[test]
    fn a_slow_pc_keeps_the_most_accurate_affordable_f0_method() {
        // Accuracy order swiftf0, fcpe, fusion, crepe, dio (a CPU run of the dev rig where one
        // SwiftF0 sample came out slow): SwiftF0 stays.
        assert_eq!(pick_pitch_without_limit(&[1.502, 1.336, 1.322, 1.329, 1.309]), Some(0));
        assert_eq!(pick_pitch_without_limit(&[2.0, 1.5, 1.2]), Some(2));
        assert_eq!(pick_pitch_without_limit(&[f64::INFINITY, 1.5, 1.4]), Some(1));
        assert_eq!(pick_pitch_without_limit(&[]), None);
        assert_eq!(pick_pitch_without_limit(&[f64::INFINITY]), None);
    }

    /// A user clip times the engine; the F0 ranking always uses the synthetic voice's truth.
    #[test]
    fn a_user_clip_still_ranks_f0_on_the_synthetic_truth() {
        let dir = std::env::temp_dir().join(format!("sigil-autotune-clip-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("clip.wav");
        crate::services::bench::signals::write_wav(&wav, &vec![0.1f32; 16_000], 16_000, 1).unwrap();
        let req = AutotuneRequest {
            clip: Some(wav.to_string_lossy().into_owned()),
            ..AutotuneRequest::default()
        };
        let c = tuning_clip(&req, 48_000, true).unwrap();
        assert_eq!(c.clip.len(), 48_000, "the user clip, resampled to the device rate");
        assert_eq!(c.truth_clip.len(), 5 * 48_000);
        assert!(c.truth.iter().any(|t| t.is_some_and(|f| f > 0.0)));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn pitch_accuracy_scores_against_the_truth() {
        let truth = vec![Some(100.0f32), Some(100.0), Some(0.0), None, Some(200.0)];
        let (gpe, vuv, rmse) = pitch_accuracy(&truth, &[100.0, 200.0, 0.0, 999.0, 0.0]);
        assert!((gpe - 0.5).abs() < 1e-9, "one of two voiced-in-both frames is an octave off");
        assert!((vuv - 0.25).abs() < 1e-9, "the last frame lost its voicing; the undecided one is skipped");
        assert!((rmse - (1200.0f64 * 1200.0 / 2.0).sqrt()).abs() < 1e-6);
    }

    #[test]
    fn params_patches_name_voice_changer_fields() {
        let mut levers = BTreeMap::new();
        levers.insert("io_binding".to_string(), "on".to_string());
        levers.insert("cuda_graphs".to_string(), "off".to_string());
        levers.insert("gpu_priority".to_string(), "auto".to_string());
        levers.insert("partial_decode".to_string(), "off".to_string());
        let cfg = Config {
            ep: "cuda".into(),
            pitch: "rmvpe".into(),
            block_ms: 40.0,
            chunk_frames: 512,
            extra_ms: 256.0,
            precision: "auto".into(),
            levers,
            threads: Threads::AsRequested,
            tuned: vec!["io_binding", "cuda_graphs"],
        };
        let p = params_of(&cfg, true, "low_latency", "auto");
        for k in ["rvc_execution_provider", "pitch_extractor", "extra_context_ms", "block_ms", "precision", "io_binding", "cuda_graphs"] {
            assert!(p.contains_key(k), "{k}");
        }
        assert_eq!((p["io_binding"].as_str(), p["cuda_graphs"].as_str()), (Some("on"), Some("off")));
        // A lever the lever stage never measured, and every algorithmic lever, keeps the voice
        // changer's own setting (what the configuration measured with): not in the patch.
        assert!(!p.contains_key("gpu_priority") && !p.contains_key("partial_decode"), "{p:?}");
        let untuned = Config { tuned: Vec::new(), ..cfg.clone() };
        let q = params_of(&untuned, true, "low_latency", "auto");
        assert!(GPU_LEVERS.iter().all(|l| !q.contains_key(l.as_str())), "levers never pinned when not tuned: {q:?}");
        // Every key is a VoiceChangerParams field: the patch round-trips through it.
        let mut full = serde_json::to_value(crate::services::voice_changer::params::VoiceChangerParams::default()).unwrap();
        for (k, v) in &p {
            assert!(full.get(k).is_some(), "{k} is not a params field");
            full[k] = v.clone();
        }
        assert!(serde_json::from_value::<crate::services::voice_changer::params::VoiceChangerParams>(full).is_ok());
        let legacy = params_of(&cfg, false, "low_latency", "auto");
        assert!(legacy.contains_key("chunk_frames") && !legacy.contains_key("block_ms"));
        assert!(!p.contains_key("rvc_preset") && !p.contains_key("worker_priority"), "GPU: the request's policy");
        // The CPU thread policies, through the voice changer's own params.
        let cpu = |threads: Threads, preset: &str| {
            let c = Config { ep: "cpu".into(), threads, tuned: Vec::new(), ..cfg.clone() };
            params_of(&c, true, preset, "auto")
        };
        assert_eq!(cpu(Threads::One, "low_latency")["rvc_preset"], "streamer");
        assert!(!cpu(Threads::One, "low_latency").contains_key("worker_priority"));
        let two = cpu(Threads::Two, "streamer");
        assert_eq!((two["rvc_preset"].as_str(), two["worker_priority"].as_str()), (Some("low_latency"), Some("protect_game")));
        let pool = cpu(Threads::Pool, "quality");
        assert_eq!((pool["rvc_preset"].as_str(), pool["worker_priority"].as_str()), (Some("quality"), Some("auto")));
        for patch in [cpu(Threads::One, "streamer"), two, pool] {
            let mut full = serde_json::to_value(crate::services::voice_changer::params::VoiceChangerParams::default()).unwrap();
            for (k, v) in &patch {
                assert!(full.get(k).is_some(), "{k} is not a params field");
                full[k] = v.clone();
            }
            let back: crate::services::voice_changer::params::VoiceChangerParams = serde_json::from_value(full).unwrap();
            assert_eq!(back.rvc_execution_provider, "cpu");
        }
    }

    /// The CPU thread stage keeps an explicit worker priority: `protect_game` never gets the pool,
    /// `protect_voice` is never demoted to two threads, and its pool keeps `protect_voice`.
    #[test]
    fn the_cpu_thread_stage_keeps_an_explicit_priority() {
        assert_eq!(Threads::sweep("auto"), &[Threads::One, Threads::Two, Threads::Pool]);
        assert_eq!(Threads::sweep("protect_game"), &[Threads::One, Threads::Two]);
        assert_eq!(Threads::sweep("protect_voice"), &[Threads::One, Threads::Pool]);
        assert_eq!(Threads::Pool.params("quality", "protect_voice").1, Some("protect_voice"));
        assert_eq!(Threads::Pool.params("quality", "auto").1, Some("auto"));
        assert_eq!(Threads::One.params("quality", "protect_game"), (Some("streamer".to_string()), None));
    }

    fn outcomes(rows: &[(&str, &str, &str)]) -> EngineInfo {
        let mut info = EngineInfo::new("rvc", 16_000, 40_000, 720);
        for (k, requested, active) in rows {
            let reason = (requested != active).then(|| format!("{requested} failed"));
            info.ep_outcomes.insert(k.to_string(), EpOutcome::new(*requested, *active, reason));
        }
        info
    }

    /// The generator decides the EP; TensorRT counts for the CUDA rung; a pitch model that falls
    /// back fails its method only; a CPU-placed small model is never a fallback.
    #[test]
    fn placement_is_decided_by_the_generator() {
        let on = |p: &Placement| matches!(p, Placement::OnRung { .. });
        // A template voice on CUDA with the small pitch model on the CPU by design.
        let p = placement(&outcomes(&[("vc.enc", "cuda", "cuda"), ("vc.flowdec", "cuda", "cuda"), ("vc.pitch", "cpu", "cpu")]), "cuda");
        assert!(on(&p), "{p:?}");
        // TensorRT-RTX on the decoder: the tensorrt_rtx EP runs (and is the active EP).
        let p = placement(
            &outcomes(&[("vc.enc", "cuda", "cuda"), ("vc.flowdec", "tensorrt_rtx", "tensorrt_rtx"), ("vc.content", "cuda", "cuda")]),
            "tensorrt_rtx",
        );
        assert_eq!(p, Placement::OnRung { active: "tensorrt_rtx".into(), notes: Vec::new() });
        // Classic TensorRT: still the accelerator, with a note.
        let p = placement(&outcomes(&[("vc.enc", "cuda", "cuda"), ("vc.flowdec", "tensorrt_rtx", "tensorrt")]), "tensorrt_rtx");
        assert!(matches!(&p, Placement::OnRung { active, notes } if active == "tensorrt" && notes.len() == 1), "{p:?}");
        // TensorRT did not run: the tensorrt_rtx EP is the CUDA rung's tune, dropped.
        let p = placement(&outcomes(&[("vc.enc", "cuda", "cuda"), ("vc.flowdec", "tensorrt_rtx", "cuda")]), "tensorrt_rtx");
        assert!(matches!(&p, Placement::EpFellBack { active, reason } if active == "cuda" && reason.contains("TensorRT-RTX")), "{p:?}");
        // The generator on the CPU: the EP is dropped.
        let p = placement(&outcomes(&[("vc.enc", "dml", "cpu"), ("vc.flowdec", "dml", "dml")]), "dml");
        assert!(matches!(&p, Placement::EpFellBack { active, .. } if active == "cpu"), "{p:?}");
        // The pitch model fell back: that method fails, the EP stays.
        let p = placement(&outcomes(&[("vc.enc", "dml", "dml"), ("vc.flowdec", "dml", "dml"), ("vc.pitch", "dml", "cpu")]), "dml_hp");
        assert!(matches!(&p, Placement::PitchFellBack { active, .. } if active == "cpu"), "{p:?}");
        // The content encoder fell back: measured as it runs, with a note.
        let p = placement(&outcomes(&[("vc.enc", "dml", "dml"), ("vc.flowdec", "dml", "dml"), ("vc.content", "dml", "cpu")]), "dml");
        assert!(matches!(&p, Placement::OnRung { notes, .. } if notes.len() == 1 && notes[0].contains("vc.content")), "{p:?}");
        // A bring-your-own ONNX voice: `bench.vc` is the generator.
        let p = placement(&outcomes(&[("bench.vc", "cuda", "cuda"), ("bench.vc.pitch", "cuda", "cpu")]), "cuda");
        assert!(matches!(p, Placement::PitchFellBack { .. }), "{p:?}");
        let p = placement(&outcomes(&[("bench.vc", "cuda", "dml"), ("bench.vc.content", "cuda", "cuda")]), "cuda");
        assert!(matches!(p, Placement::EpFellBack { .. }), "{p:?}");
        // No rows: the engine's summary.
        let mut info = EngineInfo::new("rvc", 16_000, 40_000, 720);
        info.execution_provider_active = Some("cpu".into());
        assert!(on(&placement(&info, "cpu")));
        assert!(matches!(placement(&info, "cuda"), Placement::EpFellBack { .. }));
    }

    /// The conditions come from the voice changer's params, reach every candidate's stream and
    /// voice, and survive the trip.
    #[test]
    fn conditions_carry_the_voice_changers_settings() {
        use crate::services::voice_changer::params::VoiceChangerParams;
        let p = VoiceChangerParams {
            crossfade_ms: 10.0,
            index_rate: 0.75,
            worker_priority: Some("protect_game".into()),
            f0_range: Some("singing".into()),
            denoiser: Some("gtcrn".into()),
            output_prefill_ms: 50.0,
            output_max_queue_ms: 120.0,
            gpu_knn: Some("on".into()),
            ..VoiceChangerParams::default()
        };
        let c = TuneConditions::from_params(&p, 44_100, 48_000, 2);
        assert!(c.validate().is_ok(), "{:?}", c.validate());
        let mut base = default_rvc_spec();
        let mut stream = StreamSpec::default();
        c.apply(&mut base, &mut stream);
        assert_eq!((stream.input_rate, stream.output_rate, stream.output_channels), (44_100, 48_000, 2));
        assert_eq!((stream.crossfade_ms, stream.index_rate, stream.worker_priority.as_str()), (10.0, 0.75, "protect_game"));
        assert_eq!((stream.output_prefill_ms, stream.output_max_queue_ms), (50.0, 120.0));
        assert_eq!((base.f0_range.as_str(), base.denoiser.as_str()), ("singing", "gtcrn"));
        assert_eq!(base.levers["gpu_knn"], "on");
        assert_eq!(TuneConditions::of(&base, &stream), c, "of(apply(c)) == c");
        // Unknown rates keep the stream's; the defaults are the stream's defaults.
        let mut s2 = StreamSpec::default();
        TuneConditions::from_params(&p, 0, 0, 0).apply(&mut default_rvc_spec(), &mut s2);
        assert_eq!((s2.input_rate, s2.output_rate), (48_000, 48_000));
        let d = TuneConditions::default();
        assert_eq!((d.crossfade_ms, d.worker_priority.as_str(), d.denoiser.as_str()), (20.0, "auto", "off"));
        // Bad values are refused like a spec's.
        let bad = TuneConditions { crossfade_ms: 900.0, ..TuneConditions::default() };
        assert!(bad.validate().is_err());
        let bad = TuneConditions { levers: [("nope".to_string(), "on".to_string())].into_iter().collect(), ..TuneConditions::default() };
        assert!(bad.validate().is_err());
        let req = AutotuneRequest { conditions: Some(TuneConditions { device_rate: 7, ..TuneConditions::default() }), ..AutotuneRequest::default() };
        assert!(req.validate().is_err());
    }

    /// A stored tune is stale when what it measured with changed, but not over the values it wrote
    /// itself.
    #[test]
    fn stale_tunes_name_what_changed() {
        let then = TuneConditions { device_rate: 48_000, output_rate: 48_000, chunk_frames: 512, ..TuneConditions::default() };
        let mut params = serde_json::Map::new();
        params.insert("worker_priority".into(), "protect_game".into());
        params.insert("io_binding".into(), "on".into());
        let r = AutotuneResult {
            slot: Some("C:\\voices\\a".into()),
            conditions: Some(then.clone()),
            params,
            ..AutotuneResult::default()
        };
        // The tune applied: nothing changed.
        let mut now = then.clone();
        now.worker_priority = "protect_game".into();
        now.levers.insert("io_binding".into(), "on".into());
        assert!(stale_reasons(&r, &now, "c:/voices/a/").is_empty(), "{:?}", stale_reasons(&r, &now, "c:/voices/a/"));
        // Unknown device rates are not compared.
        let unknown = TuneConditions { device_rate: 0, output_rate: 0, ..now.clone() };
        assert!(stale_reasons(&r, &unknown, "C:/voices/a").is_empty());
        // What the tune measured with changed.
        let changed = TuneConditions { crossfade_ms: 10.0, index_rate: 0.5, device_rate: 44_100, ..now.clone() };
        let why = stale_reasons(&r, &changed, "C:/voices/b");
        for needle in ["another voice", "crossfade_ms", "retrieval", "device_rate"] {
            assert!(why.iter().any(|w| w.contains(needle)), "{needle}: {why:?}");
        }
        let mut lever = now.clone();
        lever.levers.insert("partial_decode".into(), "off".into());
        assert_eq!(stale_reasons(&r, &lever, "").len(), 1);
        // A tune that predates the record is stale.
        let old = AutotuneResult { conditions: None, ..r.clone() };
        assert_eq!(stale_reasons(&old, &now, "").len(), 1);
    }

    #[test]
    fn results_are_stored_per_gpu_and_driver() {
        let dir = std::env::temp_dir().join(format!("sigil-autotune-store-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(stored_result(&dir).is_none());
        let (gpu, driver) = gpu_identity();
        let mut r = AutotuneResult {
            gpu: gpu.clone(),
            driver: driver.clone(),
            rtf_p99: 0.2,
            candidates: vec![Candidate::default()],
            ..AutotuneResult::default()
        };
        store_result(&dir, &r).unwrap();
        r.rtf_p99 = 0.3;
        store_result(&dir, &r).unwrap();
        let other = AutotuneResult {
            gpu: Some("Some Other GPU".into()),
            ..AutotuneResult::default()
        };
        store_result(&dir, &other).unwrap();
        let back = stored_result(&dir).expect("stored for this GPU");
        assert_eq!(back.rtf_p99, 0.3, "replaced, not duplicated");
        assert!(back.candidates.is_empty());
        assert_eq!(read_results(&dir).results.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A record with a missing number (NaN is written as `null`) or a broken one never drops the
    /// file's other records.
    #[test]
    fn one_bad_record_never_drops_the_others() {
        let dir = std::env::temp_dir().join(format!("sigil-autotune-lenient-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let nan = AutotuneResult {
            gpu: Some("GPU A".into()),
            rtf_p99: f64::NAN,
            block_ms_p95: f64::NAN,
            ..AutotuneResult::default()
        };
        store_result(&dir, &nan).unwrap();
        let other = AutotuneResult {
            gpu: Some("GPU B".into()),
            rtf_p99: 0.25,
            ..AutotuneResult::default()
        };
        store_result(&dir, &other).unwrap();
        let file = read_results(&dir);
        assert_eq!(file.results.len(), 2, "the null record still parses");
        assert!(file.results[0].rtf_p99.is_nan());
        // A record that does not parse at all is dropped alone.
        let path = results_path(&dir);
        let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&path).unwrap()).unwrap();
        v["results"].as_array_mut().unwrap().push(serde_json::json!({"gpu": 5}));
        std::fs::write(&path, serde_json::to_vec(&v).unwrap()).unwrap();
        let file = read_results(&dir);
        assert_eq!(file.results.len(), 2);
        let third = AutotuneResult {
            gpu: Some("GPU C".into()),
            rtf_p99: 0.3,
            ..AutotuneResult::default()
        };
        store_result(&dir, &third).unwrap();
        assert_eq!(read_results(&dir).results.len(), 3, "storing keeps every good record");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Plan acceptance: the autotuner returns a valid preset on the CPU (the fake RVC slot:
    /// hand-encoded content encoder + generator, DIO pitch).
    #[test]
    fn autotune_returns_a_valid_cpu_preset() {
        let (root, slot, hubert) = crate::services::neural_audio::vc::test_models::write_slot("bench-autotune");
        let req: AutotuneRequest = serde_json::from_value(serde_json::json!({
            "slot_dir": slot, "execution_providers": ["cpu"], "quick": true, "hubert": hubert.to_string_lossy(),
            "pitch_extractors": ["dio"], "max_seconds": 120
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_autotune(&req, &BenchContext { cancel: &cancel, progress: &progress }).unwrap();
        assert_eq!(r.params["rvc_execution_provider"], "cpu");
        assert_eq!(r.params["pitch_extractor"], "dio");
        let chunk = r.params["chunk_frames"].as_u64().unwrap() as u32;
        assert!(ALLOWED_CHUNK_FRAMES.contains(&chunk), "{chunk}");
        assert!(r.rtf_p99.is_finite() && r.latency_ms_p50.is_some(), "{r:?}");
        assert_eq!(r.format.as_deref(), Some("rvc"));
        assert!(r.candidates.iter().any(|c| c.stage == "pitch" && c.gross_pitch_error.is_some()));
        assert!(r.candidates.iter().any(|c| c.stage == "verify"), "{:?}", r.notes);
        // The measured settings ride along (the voice changer's defaults here).
        assert_eq!(r.conditions.as_ref().map(|c| c.crossfade_ms), Some(20.0));
        assert!(r.params.keys().all(|k| GPU_LEVERS.iter().all(|l| l.as_str() != k.as_str())), "no lever pinned on the CPU: {:?}", r.params);
        // The CPU thread policy was measured, and the patch carries the one chosen.
        let threads: Vec<&Candidate> = r.candidates.iter().filter(|c| c.stage == "threads").collect();
        assert!(!threads.is_empty() && threads.iter().all(|c| c.cpu_cores.is_some()), "{threads:?}");
        assert!(r.params.contains_key("rvc_preset"), "{:?}", r.params);
        // The patch applies to the voice changer's params.
        let mut full = serde_json::to_value(crate::services::voice_changer::params::VoiceChangerParams::default()).unwrap();
        for (k, v) in &r.params {
            full[k] = v.clone();
        }
        assert!(serde_json::from_value::<crate::services::voice_changer::params::VoiceChangerParams>(full).is_ok());
        let _ = std::fs::remove_dir_all(root);
    }

    /// Plan acceptance on the GPU (dev rig): the MIT default voice imported into a template slot
    /// (`SIGIL_BENCH_SLOT`, a `sigil-rvc-1` folder; `SIGIL_TEST_CONTENT_ENCODER` optional) tuned on
    /// DirectML (and CUDA when `SIGIL_BENCH_EPS` includes it).
    #[test]
    #[ignore = "dev rig: needs SIGIL_BENCH_SLOT (an imported template voice) and a GPU"]
    fn autotune_returns_a_valid_gpu_preset() {
        let Some(slot) = std::env::var_os("SIGIL_BENCH_SLOT") else {
            return;
        };
        let eps: Vec<String> = std::env::var("SIGIL_BENCH_EPS")
            .unwrap_or_else(|_| "dml".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .collect();
        let req = AutotuneRequest {
            slot_dir: Some(slot.to_string_lossy().into_owned()),
            execution_providers: eps,
            quick: true,
            hubert: std::env::var("SIGIL_TEST_CONTENT_ENCODER").ok(),
            ..AutotuneRequest::default()
        };
        let cancel = AtomicBool::new(false);
        let progress = |p: &Progress| eprintln!("{:>3.0}% {}", p.fraction * 100.0, p.message);
        let r = run_autotune(&req, &BenchContext { cancel: &cancel, progress: &progress }).unwrap();
        eprintln!("{}", serde_json::to_string_pretty(&r).unwrap());
        assert_ne!(r.params["rvc_execution_provider"], "cpu");
        assert!(r.params.contains_key("block_ms"));
    }
}
