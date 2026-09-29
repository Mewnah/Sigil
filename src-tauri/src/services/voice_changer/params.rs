//! Voice-changer IPC parameters: defaults, clamps and normalization, plus the RVC path /
//! execution-provider / F0 resolution rules shared by the start preflight and the worker.

use super::budget::{clamp_latency_budget_ms, DEFAULT_LATENCY_BUDGET_MS};
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

/// RVC profile (`streamer` / `low_latency` / `quality`); anything else is `streamer`.
pub(super) fn normalize_rvc_preset(s: &str) -> &'static str {
    match s.trim().to_ascii_lowercase().as_str() {
        "low_latency" => "low_latency",
        "quality" => "quality",
        _ => "streamer",
    }
}

/// F0 extractor (`fcpe` / `rmvpe` / `dio`). `crepe` is a legacy alias that always ran RMVPE, so it
/// is stored (and reported) as `rmvpe`; anything unknown is `fcpe`.
pub(super) fn normalize_pitch_extractor(s: &str) -> &'static str {
    match s.trim().to_ascii_lowercase().as_str() {
        "rmvpe" | "crepe" => "rmvpe",
        "dio" => "dio",
        _ => "fcpe",
    }
}

/// Normalize an EP string from the UI to one the session builder understands.
pub(super) fn normalize_execution_provider(ep: &str) -> String {
    match ep.trim().to_lowercase().as_str() {
        "cuda" | "gpu" => "cuda".to_string(),
        "dml" => "dml".to_string(),
        "" => String::new(),
        _ => "cpu".to_string(),
    }
}

/// RVC execution provider a run uses: the stored (normalized) one, or the preset default only
/// when it is unset. Shared by the start preflight and the worker so both resolve the same EP.
pub(super) fn effective_rvc_ep(execution_provider: &str, preset: RvcPresetProfile) -> &str {
    if execution_provider.is_empty() {
        preset.default_ep()
    } else {
        execution_provider
    }
}

/// F0 extractor a run uses on `ep`: the GPU-only extractor (rmvpe, and its legacy alias crepe) runs
/// as fcpe on the CPU EP, matching the inspector, which greys it out there. Returns the normalized
/// name and whether it fell back. The stored choice itself is never rewritten.
pub(super) fn effective_pitch_extractor(pitch_extractor: &str, ep: &str) -> (String, bool) {
    let p = normalize_pitch_extractor(pitch_extractor);
    if ep == "cpu" && p == "rmvpe" {
        ("fcpe".to_string(), true)
    } else {
        (p.to_string(), false)
    }
}

/// Resolve the model files a run loads (empty = auto) with the effective EP / F0 rule above and
/// write them into `rvc`; the F0 choice itself is never rewritten. Shared by the start preflight
/// and the worker, which re-resolves from its own snapshot because a params sync in between puts
/// the UI's raw (usually empty) paths back. After a GPU-only F0 fell back, the error names that,
/// since the file it could not find is then fcpe's rather than the chosen extractor's.
pub(super) fn resolve_rvc_paths(rvc: &mut RvcParamsState, app_data: &Path) -> Result<(), String> {
    let preset = RvcPresetProfile::from_str(&rvc.preset);
    let ep = effective_rvc_ep(&rvc.execution_provider, preset);
    let (pitch, fell_back) = effective_pitch_extractor(&rvc.pitch_extractor, ep);
    let (slot, hubert, fcpe, rmvpe) = preflight_rvc(
        &rvc.slot_dir,
        &rvc.hubert_path,
        &rvc.fcpe_path,
        &rvc.rmvpe_path,
        &pitch,
        app_data,
    )
    .map_err(|e| {
        if fell_back {
            format!(
                "{} runs as fcpe on the CPU execution provider: {e}",
                rvc.pitch_extractor.trim().to_lowercase()
            )
        } else {
            e
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
    /// Normalized RVC EP (`cpu` / `cuda` / `dml`; empty = preset default).
    pub(super) execution_provider: String,
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
}

impl Default for VoiceChangerParams {
    fn default() -> Self {
        let rvc = RvcParamsState::default();
        Self {
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
        }
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
    use super::{
        effective_pitch_extractor, effective_rvc_ep, normalize_execution_provider, normalize_latency_preset,
        normalize_pitch_extractor, normalize_rvc_preset, resolve_rvc_paths, AuxMicGate, RvcParamsState,
        RvcPresetProfile, VoiceChangerParams,
    };
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
        assert_eq!(effective_pitch_extractor("crepe", "cpu"), ("fcpe".to_string(), true));
        // Streamer on CUDA / DML keeps the chosen rmvpe.
        assert_eq!(effective_pitch_extractor("rmvpe", "cuda"), ("rmvpe".to_string(), false));
        assert_eq!(effective_pitch_extractor("RMVPE", "dml"), ("rmvpe".to_string(), false));
        // CPU-capable extractors are untouched everywhere.
        assert_eq!(effective_pitch_extractor("dio", "cpu"), ("dio".to_string(), false));
        assert_eq!(effective_pitch_extractor("fcpe", "cuda"), ("fcpe".to_string(), false));
    }

    /// `crepe` always ran RMVPE: it is accepted as input but never stored or reported.
    #[test]
    fn crepe_alias_is_hidden_and_enums_normalize() {
        assert_eq!(normalize_pitch_extractor("crepe"), "rmvpe");
        assert_eq!(normalize_pitch_extractor(" CREPE "), "rmvpe");
        assert_eq!(normalize_pitch_extractor("Dio"), "dio");
        assert_eq!(normalize_pitch_extractor("swiftf0"), "fcpe");
        assert_eq!(effective_pitch_extractor("crepe", "cuda"), ("rmvpe".to_string(), false));
        assert_eq!(normalize_latency_preset("STABLE"), "stable");
        assert_eq!(normalize_latency_preset("whatever"), "low_latency");
        assert_eq!(normalize_rvc_preset("Quality"), "quality");
        assert_eq!(normalize_rvc_preset(""), "streamer");
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
        let strip = |mut p: VoiceChangerParams| {
            p.latency_budget_ms = None;
            p.aux_mic_gate = None;
            p.worker_priority = None;
            p.input_device = None;
            p.rvc_consent_acknowledged = None;
            p
        };
        assert_eq!(strip(parsed), strip(d));
    }

    #[test]
    fn execution_provider_normalization() {
        assert_eq!(normalize_execution_provider("CUDA"), "cuda");
        assert_eq!(normalize_execution_provider("gpu"), "cuda");
        assert_eq!(normalize_execution_provider(" dml "), "dml");
        assert_eq!(normalize_execution_provider("tensorrt"), "cpu");
        assert_eq!(normalize_execution_provider(""), "");
    }
}
