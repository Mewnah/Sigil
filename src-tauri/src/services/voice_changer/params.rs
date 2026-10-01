//! Voice-changer IPC parameters: defaults, clamps and normalization, plus the RVC path /
//! execution-provider / F0 resolution rules shared by the start preflight and the worker.
//!
//! # P3 parameters (P3-S0; frozen, change only through INTEGRATE)
//! The Sigil Voice Engine params travel like the P0 ones: `Option`s on `VoiceChangerParams`
//! (absent or null on `set` keeps the stored value; `get` always returns them), listed with their
//! ranges and enums in `params_contract.json` (`params_p3_new_fields`). They are stored and
//! persisted but no-ops until their workstream lands: every lever falls back to the proven legacy
//! path, an F0 method this build cannot run yet runs its stand-in (`pitch_stand_in`), and an
//! accelerator EP (`tensorrt_rtx`, `dml_hp`) runs on its base rung (`base_ep`). Hot fields
//! (`rms_mix`, `formant_semitones`, `noise_scale`, `silence_db`) live in `VoiceChangerHotParams`
//! and reach the model as `VoiceControls`; the rest are read when the engine starts. What this
//! build can run is `capabilities.rs`.

use super::budget::{clamp_latency_budget_ms, DEFAULT_LATENCY_BUDGET_MS};
use crate::services::neural_audio::vc::engine::options::{
    Accelerator, DenoiserKind, F0Range, Lever, LeverMode, LeverRequest, PrecisionMode,
};
use crate::services::neural_audio::vc::traits::{VoiceControls, DEFAULT_NOISE_SCALE, DEFAULT_SILENCE_DB};
use crate::services::neural_audio::{preflight_rvc, RvcPresetProfile, DEFAULT_CHUNK_FRAMES};
use crate::services::rt_thread::WorkerPriority;
use serde::{Deserialize, Serialize};
use std::path::Path;

pub(super) const DEFAULT_PITCH_SEMITONES: f32 = 0.0;
pub(super) const DEFAULT_FORMANT_SHIFT: f32 = 0.0;
const DEFAULT_VOCODER_WINDOW_MS: u32 = 45;
pub(super) const DEFAULT_VOCODER_OVERSAMPLE: u32 = 8;

/// Output prefill before starting playback (latency cushion) — overridden by latency preset.
const PREFILL_MS_LOW: f64 = 35.0;
const MAX_QUEUE_MS_LOW: f64 = 70.0;
/// Matches TS `OUTPUT_BUFFER_STABLE` / low-latency vs stable buffer templates.
pub(super) const PREFILL_MS_STABLE: f64 = 80.0;
pub(super) const MAX_QUEUE_MS_STABLE: f64 = 150.0;

// ---- P3 ranges (params_contract.json `numbers`, `p3_new`) ----
/// SVE block length (ms); 0 = derived from `chunk_frames` (the legacy harness).
pub(crate) const BLOCK_MS_RANGE: (f64, f64) = (0.0, 200.0);
/// SVE lookahead (ms; P3-I0): future context decoded into every returned sample, at exactly that
/// much latency (0 = off; the engine rounds up to its 10 ms frames and allows at most 80).
pub(crate) const LOOKAHEAD_MS_RANGE: (f64, f64) = (0.0, 80.0);
pub(crate) const RMS_MIX_RANGE: (f32, f32) = (0.0, 1.0);
pub(crate) const FORMANT_SEMITONES_RANGE: (f32, f32) = (-12.0, 12.0);
pub(crate) const NOISE_SCALE_RANGE: (f32, f32) = (0.0, 1.0);
pub(crate) const SILENCE_DB_RANGE: (f32, f32) = (-90.0, -10.0);
pub(crate) const HOT_SLOTS_RANGE: (u32, u32) = (1, 8);
/// VRAM budget of the voice engine's hot slots (MiB); 0 = automatic (resources governor).
pub(crate) const VRAM_BUDGET_MB_RANGE: (u32, u32) = (0, 65_536);
const DEFAULT_RMS_MIX: f32 = 1.0;
const DEFAULT_HOT_SLOTS: u32 = 1;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VoiceChangerMode {
    #[default]
    Classic,
    Onnx,
    Sidecar,
    Rvc,
}

fn default_param_vocoder_window_ms() -> u32 {
    DEFAULT_VOCODER_WINDOW_MS
}

fn default_param_vocoder_oversample() -> u32 {
    DEFAULT_VOCODER_OVERSAMPLE
}

fn default_mode() -> VoiceChangerMode {
    VoiceChangerMode::Classic
}

fn default_sidecar_ws_url() -> String {
    String::new()
}

fn default_onnx_model_path() -> String {
    String::new()
}

fn default_onnx_execution_provider() -> String {
    "cpu".to_string()
}

/// RVC's own execution provider (`cpu` / `cuda` / `dml`); ONNX chunk mode keeps
/// `onnx_execution_provider`.
fn default_rvc_execution_provider() -> String {
    "cpu".to_string()
}

/// What happens to the microphone while aux (TTS) audio plays through the voice changer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuxMicGate {
    /// Mic and aux are mixed as they are.
    #[default]
    None,
    /// The mic is lowered (about -12 dB) while aux audio plays.
    Duck,
    /// The mic is silenced while aux audio plays.
    Mute,
}

impl AuxMicGate {
    /// Every accepted value, in UI order.
    #[cfg(test)]
    pub const ALL: [AuxMicGate; 3] = [AuxMicGate::None, AuxMicGate::Duck, AuxMicGate::Mute];

    /// The persisted / IPC string.
    pub fn as_str(self) -> &'static str {
        match self {
            AuxMicGate::None => "none",
            AuxMicGate::Duck => "duck",
            AuxMicGate::Mute => "mute",
        }
    }

    /// Parse a persisted / IPC string; anything unknown maps to [`AuxMicGate::None`].
    pub fn from_str_lossy(s: &str) -> Self {
        match s.trim().to_ascii_lowercase().as_str() {
            "duck" => AuxMicGate::Duck,
            "mute" => AuxMicGate::Mute,
            _ => AuxMicGate::None,
        }
    }
}

/// Output-buffer preset (`low_latency` / `stable` / `custom`); anything else is `low_latency`.
pub(super) fn normalize_latency_preset(s: &str) -> &'static str {
    match s.trim().to_ascii_lowercase().as_str() {
        "stable" => "stable",
        "custom" => "custom",
        _ => "low_latency",
    }
}

/// RVC profiles, in contract order (`studio_trt`: P3, TensorRT-RTX on the CUDA rung).
pub(crate) const RVC_PRESETS: [&str; 4] = ["streamer", "low_latency", "quality", "studio_trt"];

/// RVC profile (`streamer` / `low_latency` / `quality` / `studio_trt`); anything else is
/// `streamer`.
pub(crate) fn normalize_rvc_preset(s: &str) -> &'static str {
    match s.trim().to_ascii_lowercase().as_str() {
        "low_latency" => "low_latency",
        "quality" => "quality",
        "studio_trt" => "studio_trt",
        _ => "streamer",
    }
}

/// F0 methods, in contract order. P3 added `swiftf0`, a real `crepe` (CREPE-tiny; it was an alias
/// of RMVPE until P3), `pesto` and `fusion`.
pub(crate) const PITCH_EXTRACTORS: [&str; 7] = ["fcpe", "rmvpe", "dio", "swiftf0", "crepe", "pesto", "fusion"];
/// F0 methods every build runs (no P3 engine support needed).
#[allow(dead_code)] // P3-S0 frozen API; its P3 consumer lands with the owning workstream.
pub(crate) const BASELINE_PITCH_EXTRACTORS: [&str; 3] = ["fcpe", "rmvpe", "dio"];
/// F0 methods that need a GPU EP; on the CPU EP they run as fcpe.
pub(crate) const GPU_ONLY_PITCH_EXTRACTORS: [&str; 1] = ["rmvpe"];

/// F0 method (see [`PITCH_EXTRACTORS`]); anything unknown is `fcpe`.
pub(crate) fn normalize_pitch_extractor(s: &str) -> &'static str {
    match s.trim().to_ascii_lowercase().as_str() {
        "rmvpe" => "rmvpe",
        "dio" => "dio",
        "swiftf0" => "swiftf0",
        "crepe" => "crepe",
        "pesto" => "pesto",
        "fusion" => "fusion",
        _ => "fcpe",
    }
}

/// The baseline method an F0 method runs as while this build cannot run it (contract
/// `enums.pitch_extractor.stand_in`): the closest one by device class (`crepe` / `fusion` -> rmvpe,
/// which is what `crepe` ran before P3; `swiftf0` / `pesto` -> fcpe).
pub(crate) fn pitch_stand_in(pitch: &str) -> &'static str {
    match normalize_pitch_extractor(pitch) {
        "swiftf0" | "pesto" => "fcpe",
        "crepe" | "fusion" => "rmvpe",
        p => p,
    }
}

/// Whether this build runs the F0 method `pitch` (baseline methods always; the P3-D pitch bank
/// through each module's `available()`).
pub(crate) fn pitch_extractor_available(pitch: &str) -> bool {
    use crate::services::neural_audio::vc::{crepe, f0_fusion, pesto, swiftf0};
    match normalize_pitch_extractor(pitch) {
        "swiftf0" => swiftf0::available(),
        "crepe" => crepe::available(),
        "pesto" => pesto::available(),
        "fusion" => f0_fusion::available(),
        _ => true,
    }
}

/// RVC execution providers, in contract order: the ladder rungs plus the P3 accelerators
/// (`tensorrt_rtx` on the CUDA rung, `dml_hp` on the DirectML rung).
pub(crate) const RVC_EXECUTION_PROVIDERS: [&str; 5] = ["cpu", "cuda", "dml", "tensorrt_rtx", "dml_hp"];

/// Normalize an EP string from the UI (`gpu` is the legacy spelling of `cuda`; `""` = the preset's
/// default; anything unknown is `cpu`).
pub(crate) fn normalize_execution_provider(ep: &str) -> String {
    match ep.trim().to_lowercase().as_str() {
        "cuda" | "gpu" => "cuda".to_string(),
        "dml" => "dml".to_string(),
        "tensorrt_rtx" => "tensorrt_rtx".to_string(),
        "dml_hp" => "dml_hp".to_string(),
        "" => String::new(),
        _ => "cpu".to_string(),
    }
}

/// The EP ladder rung an EP id runs on (contract `enums.rvc_execution_provider.base`): sessions are
/// always built on `cpu` / `cuda` / `dml`.
pub(crate) fn base_ep(ep: &str) -> &'static str {
    match normalize_execution_provider(ep).as_str() {
        "cuda" | "tensorrt_rtx" => "cuda",
        "dml" | "dml_hp" => "dml",
        _ => "cpu",
    }
}

/// The accelerator an EP id asks for on top of its base rung.
pub(crate) fn accelerator_of(ep: &str) -> Accelerator {
    match normalize_execution_provider(ep).as_str() {
        "tensorrt_rtx" => Accelerator::TensorRtRtx,
        "dml_hp" => Accelerator::DmlHighPriority,
        _ => Accelerator::None,
    }
}

/// The EP a preset asks for when none is stored (contract `enums.rvc_preset.default_ep`).
pub(crate) fn preset_default_ep(preset: RvcPresetProfile) -> &'static str {
    match preset {
        RvcPresetProfile::StudioTrt => "tensorrt_rtx",
        p => p.default_ep(),
    }
}

/// RVC EP id a run asks for: the stored (normalized) one, or the preset default only when it is
/// unset. May be an accelerator (`tensorrt_rtx`, `dml_hp`).
pub(crate) fn requested_rvc_ep(execution_provider: &str, preset: RvcPresetProfile) -> &str {
    if execution_provider.is_empty() {
        preset_default_ep(preset)
    } else {
        execution_provider
    }
}

/// EP ladder rung a run's sessions are built on (`cpu` / `cuda` / `dml`): [`base_ep`] of
/// [`requested_rvc_ep`]. Shared by the start preflight and the worker so both resolve the same EP;
/// the accelerator is [`rvc_accelerator`].
pub(crate) fn effective_rvc_ep(execution_provider: &str, preset: RvcPresetProfile) -> &'static str {
    base_ep(requested_rvc_ep(execution_provider, preset))
}

/// Accelerator a run asks for on top of [`effective_rvc_ep`] (applied only when available).
#[allow(dead_code)] // P3-S0 frozen API; its P3 consumer lands with the owning workstream.
pub(crate) fn rvc_accelerator(execution_provider: &str, preset: RvcPresetProfile) -> Accelerator {
    accelerator_of(requested_rvc_ep(execution_provider, preset))
}

/// `(chosen, runnable, run)`: the normalized choice, what this build can run of it (its stand-in
/// while unavailable), and what runs on `ep` (a GPU-only method is fcpe on the CPU rung).
fn pitch_resolution(pitch_extractor: &str, ep: &str) -> (&'static str, &'static str, &'static str) {
    let chosen = normalize_pitch_extractor(pitch_extractor);
    let runnable = if pitch_extractor_available(chosen) { chosen } else { pitch_stand_in(chosen) };
    let run = if base_ep(ep) == "cpu" && GPU_ONLY_PITCH_EXTRACTORS.contains(&runnable) {
        "fcpe"
    } else {
        runnable
    };
    (chosen, runnable, run)
}

/// F0 extractor a run uses on `ep`: a method this build cannot run yet runs its stand-in, and the
/// GPU-only extractor (rmvpe) runs as fcpe on the CPU EP, matching the inspector, which greys it
/// out there. Returns the normalized name and whether it fell back. The stored choice itself is
/// never rewritten.
pub(crate) fn effective_pitch_extractor(pitch_extractor: &str, ep: &str) -> (String, bool) {
    let (chosen, _, run) = pitch_resolution(pitch_extractor, ep);
    (run.to_string(), run != chosen)
}

/// Why [`effective_pitch_extractor`] fell back, for the `rvc_warning` the worker shows (`None`
/// when the chosen method runs). The GPU-only case keeps its P0 wording.
pub(crate) fn pitch_fallback_reason(pitch_extractor: &str, ep: &str) -> Option<String> {
    let (chosen, runnable, run) = pitch_resolution(pitch_extractor, ep);
    if run == chosen {
        return None;
    }
    Some(if runnable == chosen {
        format!("{chosen} is GPU-only in Sigil; using fcpe on the CPU execution provider")
    } else if run == runnable {
        format!("{chosen} is not available in this build yet; using {run}")
    } else {
        format!(
            "{chosen} is not available in this build yet and {runnable} is GPU-only in Sigil; using fcpe on the CPU execution provider"
        )
    })
}

/// Graph precision request (`auto` / `fp32` / `fp16`); anything unknown is `auto`.
#[allow(dead_code)] // P3-S0 frozen API; its P3 consumer lands with the owning workstream.
pub(crate) fn normalize_precision(s: &str) -> &'static str {
    PrecisionMode::from_str_lossy(s).as_str()
}

/// Lever request mode (`auto` / `on` / `off`); anything unknown is `auto`.
#[allow(dead_code)] // P3-S0 frozen API; its P3 consumer lands with the owning workstream.
pub(crate) fn normalize_lever_mode(s: &str) -> &'static str {
    LeverMode::from_str_lossy(s).as_str()
}

/// Input denoiser (`off` / `gtcrn` / `dfn3` / `dpdfnet`); anything unknown is `off`.
#[allow(dead_code)] // P3-S0 frozen API; its P3 consumer lands with the owning workstream.
pub(crate) fn normalize_denoiser(s: &str) -> &'static str {
    DenoiserKind::from_str_lossy(s).as_str()
}

/// F0 search range (`speech` / `singing`); anything unknown is `speech`.
#[allow(dead_code)] // P3-S0 frozen API; its P3 consumer lands with the owning workstream.
pub(crate) fn normalize_f0_range(s: &str) -> &'static str {
    F0Range::from_str_lossy(s).as_str()
}

/// Resolve the model files a run loads (empty = auto) with the effective EP / F0 rule above and
/// write them into `rvc`; the F0 choice itself is never rewritten. Shared by the start preflight
/// and the worker, which re-resolves from its own snapshot because a params sync in between puts
/// the UI's raw (usually empty) paths back. After a GPU-only F0 fell back, the error names that,
/// since the file it could not find is then fcpe's rather than the chosen extractor's.
pub(super) fn resolve_rvc_paths(rvc: &mut RvcParamsState, app_data: &Path) -> Result<(), String> {
    let preset = RvcPresetProfile::from_str(&rvc.preset);
    let ep = effective_rvc_ep(&rvc.execution_provider, preset);
    let (chosen, runnable, pitch) = pitch_resolution(&rvc.pitch_extractor, ep);
    let (slot, hubert, fcpe, rmvpe) = preflight_rvc(
        &rvc.slot_dir,
        &rvc.hubert_path,
        &rvc.fcpe_path,
        &rvc.rmvpe_path,
        pitch,
        app_data,
    )
    .map_err(|e| {
        if pitch == chosen {
            e
        } else if runnable == chosen {
            format!("{chosen} runs as fcpe on the CPU execution provider: {e}")
        } else {
            format!("{chosen} runs as {pitch} in this build: {e}")
        }
    })?;
    rvc.slot_dir = slot.to_string_lossy().into_owned();
    rvc.hubert_path = hubert.to_string_lossy().into_owned();
    if !fcpe.as_os_str().is_empty() {
        rvc.fcpe_path = fcpe.to_string_lossy().into_owned();
    }
    if let Some(rm) = rmvpe {
        rvc.rmvpe_path = rm.to_string_lossy().into_owned();
    }
    Ok(())
}

fn default_latency_preset() -> String {
    "low_latency".to_string()
}

fn default_chunk_frames() -> u32 {
    DEFAULT_CHUNK_FRAMES as u32
}

fn default_input_gain() -> f32 {
    1.0
}

fn default_output_gain() -> f32 {
    1.0
}

fn default_output_prefill_ms() -> f64 {
    PREFILL_MS_LOW
}

fn default_output_max_queue_ms() -> f64 {
    MAX_QUEUE_MS_LOW
}

fn default_buffer_truncate_enabled() -> bool {
    false
}

fn default_buffer_truncate_rms() -> f32 {
    0.0005
}

fn default_buffer_truncate_chunks() -> u32 {
    32
}

fn default_monitor_output_device() -> String {
    String::new()
}

fn default_monitor_gain() -> f32 {
    1.0
}

fn default_rvc_slot_dir() -> String {
    String::new()
}

fn default_rvc_hubert_path() -> String {
    String::new()
}

fn default_rvc_fcpe_path() -> String {
    String::new()
}

fn default_rvc_rmvpe_path() -> String {
    String::new()
}

fn default_rvc_preset() -> String {
    "streamer".to_string()
}

fn default_pitch_extractor() -> String {
    "fcpe".to_string()
}

fn default_extra_context_ms() -> f64 {
    64.0
}

fn default_crossfade_ms() -> f64 {
    20.0
}

fn default_index_rate() -> f32 {
    0.0
}

fn default_protect() -> f32 {
    0.33
}

fn default_speaker_id() -> i64 {
    0
}

fn default_gpu_device_index() -> i32 {
    0
}

fn default_silence_skip() -> bool {
    true
}

fn default_vram_cap_mb() -> u32 {
    768
}

fn default_enabled() -> bool {
    false
}

fn default_pitch_semitones() -> f32 {
    DEFAULT_PITCH_SEMITONES
}

fn default_formant_shift() -> f32 {
    DEFAULT_FORMANT_SHIFT
}

fn default_output_device() -> String {
    "default".to_string()
}

#[derive(Clone, Debug)]
pub(super) struct RvcParamsState {
    pub(super) slot_dir: String,
    pub(super) hubert_path: String,
    pub(super) fcpe_path: String,
    pub(super) rmvpe_path: String,
    pub(super) preset: String,
    pub(super) pitch_extractor: String,
    pub(super) extra_context_ms: f64,
    pub(super) crossfade_ms: f64,
    pub(super) index_rate: f32,
    pub(super) protect: f32,
    pub(super) speaker_id: i64,
    pub(super) gpu_device_index: i32,
    pub(super) silence_skip: bool,
    pub(super) vram_cap_mb: u32,
    /// Normalized RVC EP (`cpu` / `cuda` / `dml` / `tensorrt_rtx` / `dml_hp`; empty = preset
    /// default).
    pub(super) execution_provider: String,
    // ---- P3 (read when the engine starts) ----
    /// Slot-store id of the selected voice (`""` = a folder chosen directly, `slot_dir` rules).
    pub(super) slot_id: String,
    /// SVE block length (ms); 0 = derived from `chunk_frames`.
    pub(super) block_ms: f64,
    /// SVE lookahead (ms, 0 = off; template voices only).
    pub(super) lookahead_ms: f64,
    pub(super) precision: PrecisionMode,
    /// F0 search range of the voice (per voice, `slot.json`).
    pub(super) f0_range: F0Range,
    /// Requested mode of every engine lever.
    pub(super) levers: LeverRequest,
    pub(super) denoiser: DenoiserKind,
    /// Voices kept loaded for instant switching.
    pub(super) hot_slots: u32,
    /// VRAM budget of the hot slots (MiB, 0 = automatic).
    pub(super) vram_budget_mb: u32,
}

impl Default for RvcParamsState {
    fn default() -> Self {
        Self {
            slot_dir: default_rvc_slot_dir(),
            hubert_path: default_rvc_hubert_path(),
            fcpe_path: default_rvc_fcpe_path(),
            rmvpe_path: default_rvc_rmvpe_path(),
            preset: default_rvc_preset(),
            pitch_extractor: default_pitch_extractor(),
            extra_context_ms: default_extra_context_ms(),
            crossfade_ms: default_crossfade_ms(),
            index_rate: default_index_rate(),
            protect: default_protect(),
            speaker_id: default_speaker_id(),
            gpu_device_index: default_gpu_device_index(),
            silence_skip: default_silence_skip(),
            vram_cap_mb: default_vram_cap_mb(),
            execution_provider: default_rvc_execution_provider(),
            slot_id: String::new(),
            block_ms: BLOCK_MS_RANGE.0,
            lookahead_ms: LOOKAHEAD_MS_RANGE.0,
            precision: PrecisionMode::default(),
            f0_range: F0Range::default(),
            levers: LeverRequest::default(),
            denoiser: DenoiserKind::default(),
            hot_slots: DEFAULT_HOT_SLOTS,
            vram_budget_mb: VRAM_BUDGET_MB_RANGE.0,
        }
    }
}

impl RvcParamsState {
    /// Apply the P3 cold fields of `params` (present fields only, clamped / normalized).
    pub(super) fn apply_p3(&mut self, params: &VoiceChangerParams) {
        if let Some(v) = params.slot_id.as_deref() {
            self.slot_id = v.trim().to_string();
        }
        if let Some(v) = params.block_ms.filter(|v| v.is_finite()) {
            self.block_ms = v.clamp(BLOCK_MS_RANGE.0, BLOCK_MS_RANGE.1);
        }
        if let Some(v) = params.lookahead_ms.filter(|v| v.is_finite()) {
            self.lookahead_ms = v.clamp(LOOKAHEAD_MS_RANGE.0, LOOKAHEAD_MS_RANGE.1);
        }
        if let Some(v) = params.precision.as_deref() {
            self.precision = PrecisionMode::from_str_lossy(v);
        }
        if let Some(v) = params.f0_range.as_deref() {
            self.f0_range = F0Range::from_str_lossy(v);
        }
        for lever in Lever::ALL {
            if let Some(v) = params.lever(lever) {
                self.levers.set(lever, LeverMode::from_str_lossy(v));
            }
        }
        if let Some(v) = params.denoiser.as_deref() {
            self.denoiser = DenoiserKind::from_str_lossy(v);
        }
        if let Some(v) = params.hot_slots {
            self.hot_slots = v.clamp(HOT_SLOTS_RANGE.0, HOT_SLOTS_RANGE.1);
        }
        if let Some(v) = params.vram_budget_mb {
            self.vram_budget_mb = v.clamp(VRAM_BUDGET_MB_RANGE.0, VRAM_BUDGET_MB_RANGE.1);
        }
    }
}

/// IPC params (`get_voice_changer_params` / `set_voice_changer_params`, snake_case) and the
/// `params` object of the persisted `settings.json`. Every field has a serde default, so an older
/// settings file or a sender that omits fields still parses.
///
/// The P0 fields at the end are `Option`s: on `set`, an absent (or null) field keeps the stored
/// value, so a webview that predates them cannot reset them; `get` always returns them.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct VoiceChangerParams {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default = "default_pitch_semitones")]
    pub pitch_semitones: f32,
    #[serde(default = "default_formant_shift")]
    pub formant_shift: f32,
    #[serde(default = "default_param_vocoder_window_ms")]
    pub vocoder_window_ms: u32,
    #[serde(default = "default_param_vocoder_oversample")]
    pub vocoder_oversample: u32,
    #[serde(default = "default_chunk_frames")]
    pub chunk_frames: u32,
    #[serde(default = "default_input_gain")]
    pub input_gain: f32,
    #[serde(default = "default_output_gain")]
    pub output_gain: f32,
    #[serde(default = "default_output_prefill_ms")]
    pub output_prefill_ms: f64,
    #[serde(default = "default_output_max_queue_ms")]
    pub output_max_queue_ms: f64,
    #[serde(default = "default_buffer_truncate_enabled")]
    pub buffer_truncate_enabled: bool,
    #[serde(default = "default_buffer_truncate_rms")]
    pub buffer_truncate_rms: f32,
    #[serde(default = "default_buffer_truncate_chunks")]
    pub buffer_truncate_chunks: u32,
    #[serde(default = "default_output_device")]
    pub output_device: String,
    #[serde(default = "default_mode")]
    pub mode: VoiceChangerMode,
    #[serde(default = "default_sidecar_ws_url")]
    pub sidecar_ws_url: String,
    #[serde(default = "default_onnx_model_path")]
    pub onnx_model_path: String,
    #[serde(default = "default_onnx_execution_provider")]
    pub onnx_execution_provider: String,
    #[serde(default = "default_latency_preset")]
    pub latency_preset: String,
    /// Empty = monitor off. Otherwise device name (or `"default"` for system default output).
    #[serde(default = "default_monitor_output_device")]
    pub monitor_output_device: String,
    #[serde(default = "default_monitor_gain")]
    pub monitor_gain: f32,
    #[serde(default = "default_rvc_slot_dir")]
    pub rvc_slot_dir: String,
    #[serde(default = "default_rvc_hubert_path")]
    pub rvc_hubert_path: String,
    #[serde(default = "default_rvc_fcpe_path")]
    pub rvc_fcpe_path: String,
    #[serde(default = "default_rvc_rmvpe_path")]
    pub rvc_rmvpe_path: String,
    #[serde(default = "default_rvc_preset")]
    pub rvc_preset: String,
    #[serde(default = "default_pitch_extractor")]
    pub pitch_extractor: String,
    #[serde(default = "default_extra_context_ms")]
    pub extra_context_ms: f64,
    #[serde(default = "default_crossfade_ms")]
    pub crossfade_ms: f64,
    #[serde(default = "default_index_rate")]
    pub index_rate: f32,
    #[serde(default = "default_protect")]
    pub protect: f32,
    #[serde(default = "default_speaker_id")]
    pub speaker_id: i64,
    #[serde(default = "default_gpu_device_index")]
    pub gpu_device_index: i32,
    #[serde(default = "default_silence_skip")]
    pub silence_skip: bool,
    /// CUDA arena cap for RVC sessions (applies only when `rvc_execution_provider` is `cuda`).
    #[serde(default = "default_vram_cap_mb")]
    pub vram_cap_mb: u32,
    #[serde(default = "default_rvc_execution_provider")]
    pub rvc_execution_provider: String,
    /// Backlog of captured audio (ms) above which the oldest audio is dropped (100-2000).
    #[serde(default)]
    pub latency_budget_ms: Option<f64>,
    /// `none` / `duck` / `mute`: the mic while aux (TTS) audio plays.
    #[serde(default)]
    pub aux_mic_gate: Option<String>,
    /// `auto` / `protect_game` / `protect_voice` (see `rt_thread::WorkerPriority`).
    #[serde(default)]
    pub worker_priority: Option<String>,
    /// cpal input device name; `""` = system default. `start_voice_changer`'s `inputDevice`
    /// argument, when given, replaces it.
    #[serde(default)]
    pub input_device: Option<String>,
    /// The RVC consent dialog was acknowledged.
    #[serde(default)]
    pub rvc_consent_acknowledged: Option<bool>,

    // ---- P3 (params_contract.json `params_p3_new_fields`; absent / null on `set` keeps) ----
    /// Slot-store id of the selected voice; `""` = a folder chosen directly (`rvc_slot_dir`).
    #[serde(default)]
    pub slot_id: Option<String>,
    /// SVE block length in ms (0-200; 0 = derived from `chunk_frames`).
    #[serde(default)]
    pub block_ms: Option<f64>,
    /// SVE lookahead in ms (0-80, 0 = off): template voices decode that much future context into
    /// every returned sample (cleaner joins and voicing) at exactly that much latency (P3-I0).
    #[serde(default)]
    pub lookahead_ms: Option<f64>,
    /// `auto` / `fp32` / `fp16`.
    #[serde(default)]
    pub precision: Option<String>,
    /// `speech` / `singing` (per voice).
    #[serde(default)]
    pub f0_range: Option<String>,
    /// Loudness envelope mix, 0-1 (1 = off). Hot, per voice.
    #[serde(default)]
    pub rms_mix: Option<f32>,
    /// True formant shift of the RVC path in semitones, -12..12 (0 = off). Hot, per voice.
    #[serde(default)]
    pub formant_semitones: Option<f32>,
    /// Flow prior noise scale of template voices, 0-1. Hot.
    #[serde(default)]
    pub noise_scale: Option<f32>,
    /// Silence-gate threshold (RMS dBFS), -90..-10. Hot.
    #[serde(default)]
    pub silence_db: Option<f32>,
    /// Engine levers: `auto` / `on` / `off` each (`engine::options::Lever`).
    #[serde(default)]
    pub partial_decode: Option<String>,
    #[serde(default)]
    pub anchored_noise: Option<String>,
    #[serde(default)]
    pub nsf_phase0: Option<String>,
    #[serde(default)]
    pub f0_cache: Option<String>,
    #[serde(default)]
    pub gpu_knn: Option<String>,
    #[serde(default)]
    pub io_binding: Option<String>,
    #[serde(default)]
    pub cuda_graphs: Option<String>,
    #[serde(default)]
    pub gpu_priority: Option<String>,
    #[serde(default)]
    pub governor: Option<String>,
    /// `off` / `gtcrn` / `dfn3` / `dpdfnet`.
    #[serde(default)]
    pub denoiser: Option<String>,
    /// Voices kept loaded (1-8).
    #[serde(default)]
    pub hot_slots: Option<u32>,
    /// VRAM budget of the hot slots in MiB (0 = automatic).
    #[serde(default)]
    pub vram_budget_mb: Option<u32>,
}

impl VoiceChangerParams {
    /// The IPC field of a lever.
    pub(crate) fn lever(&self, lever: Lever) -> Option<&str> {
        match lever {
            Lever::PartialDecode => self.partial_decode.as_deref(),
            Lever::AnchoredNoise => self.anchored_noise.as_deref(),
            Lever::NsfPhase0 => self.nsf_phase0.as_deref(),
            Lever::F0Cache => self.f0_cache.as_deref(),
            Lever::GpuKnn => self.gpu_knn.as_deref(),
            Lever::IoBinding => self.io_binding.as_deref(),
            Lever::CudaGraphs => self.cuda_graphs.as_deref(),
            Lever::GpuPriority => self.gpu_priority.as_deref(),
            Lever::Governor => self.governor.as_deref(),
        }
    }

    /// Set the IPC field of a lever.
    pub(crate) fn set_lever(&mut self, lever: Lever, mode: Option<LeverMode>) {
        let v = mode.map(|m| m.as_str().to_string());
        match lever {
            Lever::PartialDecode => self.partial_decode = v,
            Lever::AnchoredNoise => self.anchored_noise = v,
            Lever::NsfPhase0 => self.nsf_phase0 = v,
            Lever::F0Cache => self.f0_cache = v,
            Lever::GpuKnn => self.gpu_knn = v,
            Lever::IoBinding => self.io_binding = v,
            Lever::CudaGraphs => self.cuda_graphs = v,
            Lever::GpuPriority => self.gpu_priority = v,
            Lever::Governor => self.governor = v,
        }
    }

    /// Fill every P3 field from the stored state (what `get` returns).
    pub(super) fn set_p3_from(&mut self, rvc: &RvcParamsState, hot: &VoiceChangerHotParams) {
        self.slot_id = Some(rvc.slot_id.clone());
        self.block_ms = Some(rvc.block_ms);
        self.lookahead_ms = Some(rvc.lookahead_ms);
        self.precision = Some(rvc.precision.as_str().to_string());
        self.f0_range = Some(rvc.f0_range.as_str().to_string());
        self.rms_mix = Some(hot.rms_mix);
        self.formant_semitones = Some(hot.formant_semitones);
        self.noise_scale = Some(hot.noise_scale);
        self.silence_db = Some(hot.silence_db);
        for lever in Lever::ALL {
            self.set_lever(lever, Some(rvc.levers.get(lever)));
        }
        self.denoiser = Some(rvc.denoiser.as_str().to_string());
        self.hot_slots = Some(rvc.hot_slots);
        self.vram_budget_mb = Some(rvc.vram_budget_mb);
    }

    /// Clear every P3 field (as parsed from a sender that predates them).
    #[cfg(test)]
    pub(super) fn clear_p3(&mut self) {
        self.slot_id = None;
        self.block_ms = None;
        self.lookahead_ms = None;
        self.precision = None;
        self.f0_range = None;
        self.rms_mix = None;
        self.formant_semitones = None;
        self.noise_scale = None;
        self.silence_db = None;
        for lever in Lever::ALL {
            self.set_lever(lever, None);
        }
        self.denoiser = None;
        self.hot_slots = None;
        self.vram_budget_mb = None;
    }
}

impl Default for VoiceChangerParams {
    fn default() -> Self {
        let rvc_defaults = RvcParamsState::default();
        let rvc = rvc_defaults.clone();
        let mut p = Self {
            enabled: default_enabled(),
            pitch_semitones: default_pitch_semitones(),
            formant_shift: default_formant_shift(),
            vocoder_window_ms: default_param_vocoder_window_ms(),
            vocoder_oversample: default_param_vocoder_oversample(),
            chunk_frames: default_chunk_frames(),
            input_gain: default_input_gain(),
            output_gain: default_output_gain(),
            output_prefill_ms: default_output_prefill_ms(),
            output_max_queue_ms: default_output_max_queue_ms(),
            buffer_truncate_enabled: default_buffer_truncate_enabled(),
            buffer_truncate_rms: default_buffer_truncate_rms(),
            buffer_truncate_chunks: default_buffer_truncate_chunks(),
            output_device: default_output_device(),
            mode: default_mode(),
            sidecar_ws_url: default_sidecar_ws_url(),
            onnx_model_path: default_onnx_model_path(),
            onnx_execution_provider: default_onnx_execution_provider(),
            latency_preset: default_latency_preset(),
            monitor_output_device: default_monitor_output_device(),
            monitor_gain: default_monitor_gain(),
            rvc_slot_dir: rvc.slot_dir,
            rvc_hubert_path: rvc.hubert_path,
            rvc_fcpe_path: rvc.fcpe_path,
            rvc_rmvpe_path: rvc.rmvpe_path,
            rvc_preset: rvc.preset,
            pitch_extractor: rvc.pitch_extractor,
            extra_context_ms: rvc.extra_context_ms,
            crossfade_ms: rvc.crossfade_ms,
            index_rate: rvc.index_rate,
            protect: rvc.protect,
            speaker_id: rvc.speaker_id,
            gpu_device_index: rvc.gpu_device_index,
            silence_skip: rvc.silence_skip,
            vram_cap_mb: rvc.vram_cap_mb,
            rvc_execution_provider: rvc.execution_provider,
            latency_budget_ms: Some(DEFAULT_LATENCY_BUDGET_MS),
            aux_mic_gate: Some(AuxMicGate::default().as_str().to_string()),
            worker_priority: Some(WorkerPriority::default().as_str().to_string()),
            input_device: Some(String::new()),
            rvc_consent_acknowledged: Some(false),
            slot_id: None,
            block_ms: None,
            lookahead_ms: None,
            precision: None,
            f0_range: None,
            rms_mix: None,
            formant_semitones: None,
            noise_scale: None,
            silence_db: None,
            partial_decode: None,
            anchored_noise: None,
            nsf_phase0: None,
            f0_cache: None,
            gpu_knn: None,
            io_binding: None,
            cuda_graphs: None,
            gpu_priority: None,
            governor: None,
            denoiser: None,
            hot_slots: None,
            vram_budget_mb: None,
        };
        // Concrete P3 defaults, like the P0 options (a default `set` is a full `get`).
        p.set_p3_from(&rvc_defaults, &VoiceChangerHotParams::default());
        p
    }
}

/// Live parameters read every processing chunk (single lock vs. many mutexes).
#[derive(Clone, Copy, Debug)]
pub(super) struct VoiceChangerHotParams {
    pub(super) pitch_semitones: f32,
    pub(super) formant_shift: f32,
    pub(super) vocoder_window_ms: u32,
    pub(super) vocoder_oversample: u32,
    pub(super) chunk_frames: u32,
    pub(super) input_gain: f32,
    pub(super) output_gain: f32,
    pub(super) output_prefill_ms: f64,
    pub(super) output_max_queue_ms: f64,
    pub(super) buffer_truncate_enabled: bool,
    pub(super) buffer_truncate_rms: f32,
    pub(super) buffer_truncate_chunks: u32,
    pub(super) monitor_gain: f32,
    /// Clamped latency budget (ms).
    pub(super) latency_budget_ms: f64,
    pub(super) aux_mic_gate: AuxMicGate,
    pub(super) worker_priority: WorkerPriority,
    // ---- P3 hot per-voice controls (`VoiceControls`) ----
    pub(super) rms_mix: f32,
    pub(super) formant_semitones: f32,
    pub(super) noise_scale: f32,
    pub(super) silence_db: f32,
}

impl VoiceChangerHotParams {
    /// The per-voice controls handed to the model (`VcStreamHarness::set_voice_controls`).
    pub(super) fn voice_controls(&self) -> VoiceControls {
        VoiceControls {
            rms_mix: self.rms_mix,
            formant_semitones: self.formant_semitones,
            noise_scale: self.noise_scale,
            silence_db: self.silence_db,
        }
    }

    /// Apply the P3 hot fields of `params` (present, finite fields only; clamped).
    pub(super) fn apply_p3(&mut self, params: &VoiceChangerParams) {
        if let Some(v) = params.rms_mix.filter(|v| v.is_finite()) {
            self.rms_mix = v.clamp(RMS_MIX_RANGE.0, RMS_MIX_RANGE.1);
        }
        if let Some(v) = params.formant_semitones.filter(|v| v.is_finite()) {
            self.formant_semitones = v.clamp(FORMANT_SEMITONES_RANGE.0, FORMANT_SEMITONES_RANGE.1);
        }
        if let Some(v) = params.noise_scale.filter(|v| v.is_finite()) {
            self.noise_scale = v.clamp(NOISE_SCALE_RANGE.0, NOISE_SCALE_RANGE.1);
        }
        if let Some(v) = params.silence_db.filter(|v| v.is_finite()) {
            self.silence_db = v.clamp(SILENCE_DB_RANGE.0, SILENCE_DB_RANGE.1);
        }
    }
}

impl Default for VoiceChangerHotParams {
    fn default() -> Self {
        Self {
            pitch_semitones: DEFAULT_PITCH_SEMITONES,
            formant_shift: DEFAULT_FORMANT_SHIFT,
            vocoder_window_ms: DEFAULT_VOCODER_WINDOW_MS,
            vocoder_oversample: DEFAULT_VOCODER_OVERSAMPLE,
            chunk_frames: DEFAULT_CHUNK_FRAMES as u32,
            input_gain: 1.0,
            output_gain: 1.0,
            output_prefill_ms: PREFILL_MS_LOW,
            output_max_queue_ms: MAX_QUEUE_MS_LOW,
            buffer_truncate_enabled: false,
            buffer_truncate_rms: 0.0005,
            buffer_truncate_chunks: 32,
            monitor_gain: 1.0,
            latency_budget_ms: DEFAULT_LATENCY_BUDGET_MS,
            aux_mic_gate: AuxMicGate::None,
            worker_priority: WorkerPriority::Auto,
            rms_mix: DEFAULT_RMS_MIX,
            formant_semitones: 0.0,
            noise_scale: DEFAULT_NOISE_SCALE,
            silence_db: DEFAULT_SILENCE_DB,
        }
    }
}

/// Clamp a latency budget from IPC (`None` = keep `current`).
pub(super) fn latency_budget_or(current: f64, requested: Option<f64>) -> f64 {
    requested.map(clamp_latency_budget_ms).unwrap_or(current)
}

pub(super) fn clamp_output_buffer_ms(prefill: f64, max_q: f64) -> (f64, f64) {
    let prefill = prefill.clamp(5.0, 300.0);
    let max_q = max_q.clamp(prefill + 5.0, 600.0);
    (prefill, max_q)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    /// Scratch app-data and slot folders, removed on drop.
    struct TempRvcDirs(PathBuf);

    impl Drop for TempRvcDirs {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// The worker re-resolves raw UI paths ("" = auto) with the preflight's rule, and an error
    /// after a GPU-only F0 fell back to fcpe says so.
    #[test]
    fn rvc_paths_resolve_from_raw_and_name_the_fcpe_fallback() {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let root = std::env::temp_dir().join(format!("sigil-vc-resolve-{}-{nanos}", std::process::id()));
        let _dirs = TempRvcDirs(root.clone());
        let app_data = root.join("app");
        let base = app_data.join("voice-changer").join("base");
        let slot = root.join("slot");
        std::fs::create_dir_all(&base).unwrap();
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::write(slot.join("gen.onnx"), b"").unwrap();
        std::fs::write(base.join("hubert_base.onnx"), b"").unwrap();
        std::fs::write(base.join("fcpe.onnx"), b"").unwrap();
        let raw = |pitch: &str, ep: &str| RvcParamsState {
            slot_dir: slot.to_string_lossy().into_owned(),
            pitch_extractor: pitch.to_string(),
            execution_provider: ep.to_string(),
            ..RvcParamsState::default()
        };

        // rmvpe on the CPU EP runs as fcpe: the empty paths resolve to the base files and the
        // stored F0 choice is kept.
        let mut rvc = raw("rmvpe", "cpu");
        resolve_rvc_paths(&mut rvc, &app_data).unwrap();
        assert_eq!(PathBuf::from(&rvc.slot_dir), slot);
        assert_eq!(PathBuf::from(&rvc.hubert_path), base.join("hubert_base.onnx"));
        assert_eq!(PathBuf::from(&rvc.fcpe_path), base.join("fcpe.onnx"));
        assert_eq!(rvc.rmvpe_path, "");
        assert_eq!(rvc.pitch_extractor, "rmvpe");

        std::fs::remove_file(base.join("fcpe.onnx")).unwrap();
        // After the fallback the missing file is fcpe's; the error names the fallback.
        let mut rvc = raw("rmvpe", "cpu");
        let err = resolve_rvc_paths(&mut rvc, &app_data).unwrap_err();
        assert!(
            err.starts_with("rmvpe runs as fcpe on the CPU execution provider: FCPE ONNX not found"),
            "{err}"
        );
        assert_eq!(rvc.hubert_path, "");
        // A chosen fcpe, or rmvpe on a GPU EP, has nothing to name.
        let err = resolve_rvc_paths(&mut raw("fcpe", "cpu"), &app_data).unwrap_err();
        assert!(err.starts_with("FCPE ONNX not found"), "{err}");
        let err = resolve_rvc_paths(&mut raw("rmvpe", "cuda"), &app_data).unwrap_err();
        assert!(err.starts_with("RMVPE ONNX not found"), "{err}");
    }

    /// Start preflight and worker share one rule: the stored EP wins (preset default only when
    /// unset), and only the CPU EP turns a GPU-only F0 into fcpe.
    #[test]
    fn rvc_effective_ep_and_pitch_extractor() {
        // Low latency / Quality default to CUDA but honour an explicit CPU choice.
        assert_eq!(effective_rvc_ep("", RvcPresetProfile::LowLatency), "cuda");
        assert_eq!(effective_rvc_ep("cpu", RvcPresetProfile::Quality), "cpu");
        // Streamer defaults to CPU but honours an explicit GPU choice.
        assert_eq!(effective_rvc_ep("", RvcPresetProfile::Streamer), "cpu");
        assert_eq!(effective_rvc_ep("cuda", RvcPresetProfile::Streamer), "cuda");
        assert_eq!(effective_rvc_ep("dml", RvcPresetProfile::Streamer), "dml");

        // Low latency / Quality with rmvpe on the CPU EP: preflight and run as fcpe.
        assert_eq!(effective_pitch_extractor("rmvpe", "cpu"), ("fcpe".to_string(), true));
        // Streamer on CUDA / DML keeps the chosen rmvpe.
        assert_eq!(effective_pitch_extractor("rmvpe", "cuda"), ("rmvpe".to_string(), false));
        assert_eq!(effective_pitch_extractor("RMVPE", "dml"), ("rmvpe".to_string(), false));
        // CPU-capable extractors are untouched everywhere.
        assert_eq!(effective_pitch_extractor("dio", "cpu"), ("dio".to_string(), false));
        assert_eq!(effective_pitch_extractor("fcpe", "cuda"), ("fcpe".to_string(), false));
        // The accelerators build their sessions on the base rung (an accelerator EP is a GPU EP).
        assert_eq!(effective_rvc_ep("tensorrt_rtx", RvcPresetProfile::Streamer), "cuda");
        assert_eq!(effective_rvc_ep("dml_hp", RvcPresetProfile::Quality), "dml");
        assert_eq!(effective_pitch_extractor("rmvpe", "tensorrt_rtx"), ("rmvpe".to_string(), false));
    }

    /// P3 F0 methods: a real value each (`crepe` is no longer an alias of rmvpe); while this build
    /// cannot run one, its stand-in runs and the warning says so. Written against `available()`,
    /// so it keeps holding when P3-D enables an extractor.
    #[test]
    fn p3_pitch_methods_run_their_stand_in_until_available() {
        for p in PITCH_EXTRACTORS {
            assert_eq!(normalize_pitch_extractor(p), p);
            assert_eq!(normalize_pitch_extractor(&format!(" {} ", p.to_uppercase())), p);
        }
        assert_eq!(normalize_pitch_extractor("harvest"), "fcpe");
        for p in BASELINE_PITCH_EXTRACTORS {
            assert!(pitch_extractor_available(p));
            assert_eq!(pitch_stand_in(p), p);
        }
        for (p, stand_in) in [("swiftf0", "fcpe"), ("crepe", "rmvpe"), ("pesto", "fcpe"), ("fusion", "rmvpe")] {
            assert_eq!(pitch_stand_in(p), stand_in);
            for ep in ["cpu", "cuda", "dml"] {
                let (run, fell_back) = effective_pitch_extractor(p, ep);
                let reason = pitch_fallback_reason(p, ep);
                if pitch_extractor_available(p) {
                    assert_eq!((run.as_str(), fell_back), (p, false));
                    assert_eq!(reason, None);
                    continue;
                }
                let expect = if ep == "cpu" && stand_in == "rmvpe" { "fcpe" } else { stand_in };
                assert_eq!((run.as_str(), fell_back), (expect, true), "{p} on {ep}");
                let reason = reason.unwrap();
                assert!(reason.starts_with(&format!("{p} is not available in this build yet")), "{reason}");
                assert!(reason.contains(&format!("using {expect}")), "{reason}");
            }
        }
        // The GPU-only fallback keeps its P0 wording.
        assert_eq!(
            pitch_fallback_reason("rmvpe", "cpu").as_deref(),
            Some("rmvpe is GPU-only in Sigil; using fcpe on the CPU execution provider")
        );
        assert_eq!(pitch_fallback_reason("rmvpe", "dml"), None);
    }

    #[test]
    fn enums_normalize() {
        assert_eq!(normalize_pitch_extractor("Dio"), "dio");
        assert_eq!(normalize_latency_preset("STABLE"), "stable");
        assert_eq!(normalize_latency_preset("whatever"), "low_latency");
        assert_eq!(normalize_rvc_preset("Quality"), "quality");
        assert_eq!(normalize_rvc_preset(" Studio_TRT "), "studio_trt");
        assert_eq!(normalize_rvc_preset(""), "streamer");
        for p in RVC_PRESETS {
            assert_eq!(normalize_rvc_preset(p), p);
        }
        assert_eq!(normalize_precision("FP16"), "fp16");
        assert_eq!(normalize_precision("int4"), "auto");
        assert_eq!(normalize_lever_mode("Off"), "off");
        assert_eq!(normalize_lever_mode("yes"), "auto");
        assert_eq!(normalize_denoiser("DFN3"), "dfn3");
        assert_eq!(normalize_denoiser("rnnoise"), "off");
        assert_eq!(normalize_f0_range("singing"), "singing");
        assert_eq!(normalize_f0_range("opera"), "speech");
        for g in AuxMicGate::ALL {
            assert_eq!(AuxMicGate::from_str_lossy(g.as_str()), g);
            assert_eq!(serde_json::to_string(&g).unwrap(), format!("\"{}\"", g.as_str()));
        }
        assert_eq!(AuxMicGate::from_str_lossy("bogus"), AuxMicGate::None);
    }

    /// Every field has a serde default (older settings files and partial senders still parse),
    /// and `Default` agrees with the serde defaults.
    #[test]
    fn params_parse_from_an_empty_object_like_default() {
        let parsed: VoiceChangerParams = serde_json::from_str("{}").unwrap();
        let d = VoiceChangerParams::default();
        // The P0 options are absent (keep the stored value) when parsed, concrete in `Default`.
        assert_eq!(parsed.latency_budget_ms, None);
        assert_eq!(d.latency_budget_ms, Some(300.0));
        assert_eq!(parsed.rms_mix, None);
        assert_eq!(d.rms_mix, Some(1.0));
        assert_eq!(d.partial_decode.as_deref(), Some("auto"));
        let strip = |mut p: VoiceChangerParams| {
            p.latency_budget_ms = None;
            p.aux_mic_gate = None;
            p.worker_priority = None;
            p.input_device = None;
            p.rvc_consent_acknowledged = None;
            p.clear_p3();
            p
        };
        assert_eq!(strip(parsed), strip(d));
    }

    #[test]
    fn execution_provider_normalization() {
        assert_eq!(normalize_execution_provider("CUDA"), "cuda");
        assert_eq!(normalize_execution_provider("gpu"), "cuda");
        assert_eq!(normalize_execution_provider(" dml "), "dml");
        // P3 accelerators are values of their own; classic TensorRT is not one (yet).
        assert_eq!(normalize_execution_provider("TensorRT_RTX"), "tensorrt_rtx");
        assert_eq!(normalize_execution_provider("dml_hp"), "dml_hp");
        assert_eq!(normalize_execution_provider("tensorrt"), "cpu");
        assert_eq!(normalize_execution_provider(""), "");
        for ep in RVC_EXECUTION_PROVIDERS {
            assert_eq!(normalize_execution_provider(ep), ep);
        }
        assert_eq!(base_ep("tensorrt_rtx"), "cuda");
        assert_eq!(base_ep("dml_hp"), "dml");
        assert_eq!(base_ep("gpu"), "cuda");
        assert_eq!(base_ep("bogus"), "cpu");
        assert_eq!(accelerator_of("tensorrt_rtx"), Accelerator::TensorRtRtx);
        assert_eq!(accelerator_of("dml_hp"), Accelerator::DmlHighPriority);
        assert_eq!(accelerator_of("cuda"), Accelerator::None);
        // Studio-TRT asks for TensorRT-RTX and builds on CUDA.
        assert_eq!(requested_rvc_ep("", RvcPresetProfile::StudioTrt), "tensorrt_rtx");
        assert_eq!(effective_rvc_ep("", RvcPresetProfile::StudioTrt), "cuda");
        assert_eq!(rvc_accelerator("", RvcPresetProfile::StudioTrt), Accelerator::TensorRtRtx);
        assert_eq!(rvc_accelerator("cuda", RvcPresetProfile::StudioTrt), Accelerator::None);
    }

    /// P3 fields: present values are clamped / normalized into the state, absent ones keep it.
    #[test]
    fn p3_fields_apply_with_clamps_and_absent_keeps() {
        let mut rvc = RvcParamsState::default();
        let mut hot = VoiceChangerHotParams::default();
        let mut p = VoiceChangerParams::default();
        p.clear_p3();
        p.slot_id = Some(" alto ".into());
        p.block_ms = Some(500.0);
        p.lookahead_ms = Some(-5.0);
        p.precision = Some("FP16".into());
        p.f0_range = Some("singing".into());
        p.rms_mix = Some(2.0);
        p.formant_semitones = Some(-40.0);
        p.noise_scale = Some(f32::NAN);
        p.silence_db = Some(-5.0);
        p.cuda_graphs = Some("ON".into());
        p.governor = Some("off".into());
        p.denoiser = Some("gtcrn".into());
        p.hot_slots = Some(0);
        p.vram_budget_mb = Some(1_000_000);
        rvc.apply_p3(&p);
        hot.apply_p3(&p);
        let mut got = VoiceChangerParams::default();
        got.set_p3_from(&rvc, &hot);
        assert_eq!(got.slot_id.as_deref(), Some("alto"));
        assert_eq!(got.block_ms, Some(200.0));
        assert_eq!(got.lookahead_ms, Some(0.0));
        assert_eq!(got.precision.as_deref(), Some("fp16"));
        assert_eq!(got.f0_range.as_deref(), Some("singing"));
        assert_eq!(got.rms_mix, Some(1.0));
        assert_eq!(got.formant_semitones, Some(-12.0));
        assert_eq!(got.noise_scale, Some(DEFAULT_NOISE_SCALE), "non-finite keeps the stored value");
        assert_eq!(got.silence_db, Some(-10.0));
        assert_eq!(got.cuda_graphs.as_deref(), Some("on"));
        assert_eq!(got.governor.as_deref(), Some("off"));
        assert_eq!(got.partial_decode.as_deref(), Some("auto"), "absent keeps");
        assert_eq!(got.denoiser.as_deref(), Some("gtcrn"));
        assert_eq!(got.hot_slots, Some(1));
        assert_eq!(got.vram_budget_mb, Some(65_536));
        assert_eq!(hot.voice_controls().formant_semitones, -12.0);
        // A sender without the P3 fields changes nothing.
        let before = (rvc.clone(), hot);
        let mut old = VoiceChangerParams::default();
        old.clear_p3();
        rvc.apply_p3(&old);
        hot.apply_p3(&old);
        assert_eq!(rvc.levers, before.0.levers);
        assert_eq!(rvc.slot_id, before.0.slot_id);
        assert_eq!(hot.voice_controls(), before.1.voice_controls());
    }
}
