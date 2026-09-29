//! Real-time voice changer: capture → **classic** phase vocoder (DSP), optional **WebSocket** converter,
//! optional **ONNX** chunk model, or **RVC** neural VC → shared post chain / harness → playback.

use crate::services::audio::{resolve_input_device, resolve_output_device};
use crate::services::neural_audio::chunk_processor::{apply_formant_interleaved, FormantLpState};
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::{
    clamp_chunk_frames, ClassicPhaseVocoderProcessor, DioPitchExtractor, FcpePitchExtractor,
    OnnxVoiceBridge, PitchExtractor, preflight_rvc, RmvpePitchExtractor, RvcModel, RvcPresetProfile,
    SidecarVoiceBridge, VcStreamHarness, VoiceChunkPipeline, VoiceConversionModel,
    DEFAULT_CHUNK_FRAMES,
};
use base64::{engine::general_purpose::STANDARD as B64_ENGINE, Engine as _};
use cpal::traits::{DeviceTrait, StreamTrait};
use hound::WavReader;
use rtrb::RingBuffer;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};

/// IPC body for [`voice_changer_push_aux_wav`].
/// Prefer `wavBytesB64` from the webview (compact over IPC vs. JSON number arrays).
#[derive(Deserialize)]
struct PushAuxWavRequest {
    #[serde(default, rename = "wavBytes", alias = "wav_bytes")]
    wav_bytes: Option<Vec<u8>>,
    #[serde(default, rename = "wavBytesB64", alias = "wav_bytes_b64")]
    wav_bytes_b64: Option<String>,
}

fn decode_push_aux_wav_payload(req: PushAuxWavRequest) -> Result<Vec<u8>, String> {
    if let Some(s) = req.wav_bytes_b64.as_ref().map(|x| x.trim()).filter(|x| !x.is_empty()) {
        return B64_ENGINE
            .decode(s.as_bytes())
            .map_err(|e| format!("Invalid base64 WAV: {}", e));
    }
    req.wav_bytes
        .filter(|b| !b.is_empty())
        .ok_or_else(|| "Missing WAV payload (wavBytes or wavBytesB64).".into())
}

/// Decode WAV PCM and mix into the running voice changer (used by ONNX/sidecar TTS in-process).
pub fn push_aux_wav_bytes(state: &VoiceChangerState, raw: Vec<u8>) -> Result<(), String> {
    if raw.is_empty() {
        return Ok(());
    }
    let cursor = Cursor::new(raw);
    let mut reader = WavReader::new(cursor).map_err(|e| format!("WAV: {}", e))?;
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let ch = spec.channels.max(1);
    let samples: Result<Vec<f32>, String> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map_err(|e| e.to_string()))
            .collect(),
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            reader
                .samples::<i32>()
                .map(|s| {
                    s.map_err(|e| e.to_string()).map(|v| match bits {
                        16 => v as f32 / 32768.0,
                        24 => v as f32 / 8388608.0,
                        _ => v as f32 / (1i32 << (bits.min(31) - 1)) as f32,
                    })
                })
                .collect()
        }
    };
    let samples = samples?;
    voice_changer_push_aux_pcm_impl(state, samples, sr, ch)
}

const DEFAULT_PITCH_SEMITONES: f32 = 0.0;
const DEFAULT_FORMANT_SHIFT: f32 = 0.0;
const DEFAULT_VOCODER_WINDOW_MS: u32 = 45;
const DEFAULT_VOCODER_OVERSAMPLE: u32 = 8;

/// Output prefill before starting playback (latency cushion) — overridden by latency preset.
const PREFILL_MS_LOW: f64 = 35.0;
const MAX_QUEUE_MS_LOW: f64 = 70.0;
/// Matches TS `OUTPUT_BUFFER_STABLE` / low-latency vs stable buffer templates.
const PREFILL_MS_STABLE: f64 = 80.0;
const MAX_QUEUE_MS_STABLE: f64 = 150.0;

/// Throttle repeated `voice_changer:error` emissions while conversion keeps failing each chunk.
const VC_ERROR_EMIT_THROTTLE: Duration = Duration::from_secs(2);

#[derive(Clone, Serialize)]
struct VoiceChangerErrorPayload {
    code: String,
    message: String,
}

/// Emit one `voice_changer:error` event (codes: `onnx_infer`, `sidecar_length`,
/// `sidecar_exchange`, `rvc_load`, `rvc_convert`, `rvc_warning`, `worker_panic`).
fn emit_vc_error<R: Runtime>(app: &AppHandle<R>, code: &'static str, message: impl Into<String>) {
    let _ = app.emit(
        "voice_changer:error",
        VoiceChangerErrorPayload {
            code: code.to_string(),
            message: message.into(),
        },
    );
}

fn maybe_emit_vc_error<R: Runtime>(
    app: &AppHandle<R>,
    code: &'static str,
    message: impl Into<String>,
    last_emit: &mut Option<Instant>,
) {
    let now = Instant::now();
    if let Some(t) = *last_emit {
        if now.duration_since(t) < VC_ERROR_EMIT_THROTTLE {
            return;
        }
    }
    *last_emit = Some(now);
    emit_vc_error(app, code, message);
}

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

/// Normalize an EP string from the UI to one the session builder understands.
fn normalize_execution_provider(ep: &str) -> String {
    match ep.trim().to_lowercase().as_str() {
        "cuda" | "gpu" => "cuda".to_string(),
        "dml" => "dml".to_string(),
        "" => String::new(),
        _ => "cpu".to_string(),
    }
}

/// RVC execution provider a run uses: the stored (normalized) one, or the preset default only
/// when it is unset. Shared by the start preflight and the worker so both resolve the same EP.
fn effective_rvc_ep(execution_provider: &str, preset: RvcPresetProfile) -> &str {
    if execution_provider.is_empty() {
        preset.default_ep()
    } else {
        execution_provider
    }
}

/// F0 extractor a run uses on `ep`: GPU-only extractors (rmvpe / crepe) run as fcpe on the CPU
/// EP, matching the inspector, which greys them out there. Returns the lower-cased name and
/// whether it fell back. The stored choice itself is never rewritten.
fn effective_pitch_extractor(pitch_extractor: &str, ep: &str) -> (String, bool) {
    let p = pitch_extractor.trim().to_lowercase();
    if ep == "cpu" && (p == "rmvpe" || p == "crepe") {
        ("fcpe".to_string(), true)
    } else {
        (p, false)
    }
}

/// Resolve the model files a run loads (empty = auto) with the effective EP / F0 rule above and
/// write them into `rvc`; the F0 choice itself is never rewritten. Shared by the start preflight
/// and the worker, which re-resolves from its own snapshot because a params sync in between puts
/// the UI's raw (usually empty) paths back. After a GPU-only F0 fell back, the error names that,
/// since the file it could not find is then fcpe's rather than the chosen extractor's.
fn resolve_rvc_paths(rvc: &mut RvcParamsState, app_data: &Path) -> Result<(), String> {
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

#[derive(Clone, Debug)]
struct RvcParamsState {
    slot_dir: String,
    hubert_path: String,
    fcpe_path: String,
    rmvpe_path: String,
    preset: String,
    pitch_extractor: String,
    extra_context_ms: f64,
    crossfade_ms: f64,
    index_rate: f32,
    protect: f32,
    speaker_id: i64,
    gpu_device_index: i32,
    silence_skip: bool,
    vram_cap_mb: u32,
    /// Normalized RVC EP (`cpu` / `cuda` / `dml`; empty = preset default).
    execution_provider: String,
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

#[derive(Serialize, Deserialize, Debug)]
pub struct VoiceChangerParams {
    pub enabled: bool,
    pub pitch_semitones: f32,
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
}

/// Live parameters read every processing chunk (single lock vs. many mutexes).
#[derive(Clone, Copy, Debug)]
struct VoiceChangerHotParams {
    pitch_semitones: f32,
    formant_shift: f32,
    vocoder_window_ms: u32,
    vocoder_oversample: u32,
    chunk_frames: u32,
    input_gain: f32,
    output_gain: f32,
    output_prefill_ms: f64,
    output_max_queue_ms: f64,
    buffer_truncate_enabled: bool,
    buffer_truncate_rms: f32,
    buffer_truncate_chunks: u32,
    monitor_gain: f32,
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
        }
    }
}

#[derive(Serialize, Clone, Debug)]
pub struct VoiceChangerMetrics {
    pub buffer_ms: f64,
    pub process_ms: f64,
    pub chunk_ms: f64,
}

/// Extra ring top-up (frames per channel) beyond one device callback + one chunk, absorbing
/// scheduling jitter between the worker tick and the device callback.
const RING_TOPUP_MARGIN_FRAMES: usize = 64;

/// Floor of the output and monitor rings (interleaved samples). The ring must hold its top-up
/// target: prefill (or one device callback) + one chunk + one capture callback + margin. With a
/// 16384 floor, 48 kHz stereo, a 35 ms prefill and 512-frame chunks left room for capture
/// callbacks of only ~124 ms; 32768 covers ~290 ms.
const OUT_RING_MIN_SAMPLES: usize = 32_768;
const OUT_RING_MAX_SAMPLES: usize = 262_144;

/// After an underrun, blend this many **frames** (per channel) from the held sample back to live
/// audio so playback resumes without a click.
const RECOVER_FRAMES: usize = 16;

struct OutputPlayback {
    deque: VecDeque<f32>,
    prefill_target: usize,
    max_queue_samples: usize,
    prefilled: bool,
    truncate_silent_streak: u32,
    /// Length of the most recently pushed chunk (ring top-up allowance beyond prefill).
    last_chunk_len: usize,
    /// Interleaved channel count: trims and ring hand-off move whole frames only, so the
    /// device callback never sees a split frame (which would swap channels).
    ch: usize,
    /// Samples requested by the device's latest output callback (written by [`RingPlayout`];
    /// 0 before the first callback).
    callback_len: Arc<AtomicUsize>,
    /// Capture-side burst: interleaved samples of the latest input callback (0 before the first,
    /// or when not wired). The worker produces nothing between two capture callbacks, so the
    /// ring must also bridge one capture period.
    in_burst_len: Arc<AtomicUsize>,
    /// Converts `in_burst_len` into this output's samples: input channels, and the output/input
    /// sample rates (`r_out / r_in`).
    in_burst_ch: usize,
    in_burst_r_in: u64,
    in_burst_r_out: u64,
}

fn rms_interleaved(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|x| x * x).sum();
    (sum / samples.len() as f32).sqrt()
}

impl OutputPlayback {
    fn new(
        prefill_target: usize,
        max_queue_samples: usize,
        ch_out: usize,
        callback_len: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            deque: VecDeque::with_capacity(max_queue_samples.max(prefill_target) + 8192),
            prefill_target,
            max_queue_samples: max_queue_samples.max(prefill_target + 1),
            prefilled: false,
            truncate_silent_streak: 0,
            last_chunk_len: 0,
            ch: ch_out.max(1),
            callback_len,
            in_burst_len: Arc::new(AtomicUsize::new(0)),
            in_burst_ch: 1,
            in_burst_r_in: 1,
            in_burst_r_out: 1,
        }
    }

    /// Wire the capture callback length (`in_burst_len`, interleaved samples at `ch_in` / `r_in`)
    /// into the ring target of this output (`r_out`).
    fn with_capture_burst(
        mut self,
        in_burst_len: Arc<AtomicUsize>,
        ch_in: usize,
        r_in: u32,
        r_out: u32,
    ) -> Self {
        self.in_burst_len = in_burst_len;
        self.in_burst_ch = ch_in.max(1);
        self.in_burst_r_in = u64::from(r_in.max(1));
        self.in_burst_r_out = u64::from(r_out.max(1));
        self
    }

    fn queue_len(&self) -> usize {
        self.deque.len()
    }

    /// Latest capture callback converted to this output's interleaved samples (whole frames,
    /// rounded up).
    fn in_burst_samples(&self) -> usize {
        let frames_in = (self.in_burst_len.load(Ordering::Relaxed) / self.in_burst_ch) as u64;
        let frames_out = (frames_in * self.in_burst_r_out + self.in_burst_r_in - 1) / self.in_burst_r_in;
        (frames_out as usize).saturating_mul(self.ch)
    }

    /// Level at which the prefill gate opens: the prefill, or one full device callback when that
    /// is longer (a gate below one callback would underrun the very first period).
    fn gate_target(&self) -> usize {
        self.prefill_target.max(self.callback_len.load(Ordering::Relaxed))
    }

    /// Samples the ring should hold after a top-up: one full device callback (which can be
    /// longer than the prefill, e.g. a 10 ms WASAPI period vs. a 5 ms prefill) or the prefill,
    /// whichever is larger, plus a small jitter margin, plus the longest stretch with no top-up.
    /// Top-ups happen only when a chunk is produced, and chunks are produced only when capture
    /// delivers: the gap between two top-ups is below one chunk plus one capture callback (e.g.
    /// a 20 ms capture period feeding 3.3 ms chunks, or a capture period just short of a chunk).
    fn ring_target(&self) -> usize {
        self.gate_target()
            .max(1)
            .saturating_add(self.last_chunk_len)
            .saturating_add(self.in_burst_samples())
            .saturating_add(RING_TOPUP_MARGIN_FRAMES * self.ch)
    }

    /// Queue cap over deque + ring. Never below the ring target: a cap smaller than one device
    /// callback plus a chunk plus one capture callback would otherwise drop audio on every tick.
    fn queue_cap(&self) -> usize {
        self.max_queue_samples.max(self.ring_target())
    }

    /// Drop at least `n` samples from the front of the deque, rounded up to whole frames and
    /// capped at the deque length (the ring part cannot be trimmed from here).
    fn drop_front(&mut self, n: usize) {
        let ch = self.ch;
        let n = ((n + ch - 1) / ch * ch).min(self.deque.len());
        if n > 0 {
            self.deque.drain(0..n);
        }
    }

    /// Apply new prefill/max targets from hot params without restarting cpal (may reset prefill gating).
    /// `ring_used` = samples already handed to the lock-free output ring (still queued for playback).
    fn update_limits(&mut self, prefill_target: usize, max_queue_samples: usize, ring_used: usize) {
        let max_queue_samples = max_queue_samples.max(prefill_target.saturating_add(1));
        if self.prefill_target == prefill_target && self.max_queue_samples == max_queue_samples {
            return;
        }
        self.prefill_target = prefill_target;
        self.max_queue_samples = max_queue_samples;
        self.drop_front((self.deque.len() + ring_used).saturating_sub(self.queue_cap()));
        if self.deque.len() + ring_used < self.prefill_target {
            self.prefilled = false;
        }
    }

    fn push_interleaved(
        &mut self,
        samples: &[f32],
        truncate_enabled: bool,
        truncate_rms: f32,
        truncate_chunks: u32,
        ring_used: usize,
    ) {
        self.last_chunk_len = samples.len();
        self.deque.extend(samples.iter().copied());
        // The queue cap covers deque + ring; only the deque part can be trimmed here.
        self.drop_front((self.deque.len() + ring_used).saturating_sub(self.queue_cap()));

        if truncate_enabled && truncate_chunks > 0 && !samples.is_empty() {
            let rms = rms_interleaved(samples);
            if rms < truncate_rms {
                self.truncate_silent_streak = self.truncate_silent_streak.saturating_add(1);
            } else {
                self.truncate_silent_streak = 0;
            }
            if self.truncate_silent_streak >= truncate_chunks {
                self.drop_front((self.deque.len() + ring_used).saturating_sub(self.prefill_target));
                self.truncate_silent_streak = 0;
            }
        } else {
            self.truncate_silent_streak = 0;
        }
    }

    /// Hand whole frames to the lock-free output ring once the prefill gate opens. The ring is
    /// only topped up to [`Self::ring_target`], so any backlog stays in the deque where the
    /// queue cap and silence truncation can still trim it.
    fn flush_to_ring(&mut self, producer: &mut rtrb::Producer<f32>, ring_used: usize) {
        if !self.prefilled {
            if self.deque.len() + ring_used >= self.gate_target() {
                self.prefilled = true;
            } else {
                return;
            }
        }
        let ch = self.ch;
        let budget = self.ring_target().saturating_sub(ring_used);
        let n = (budget / ch)
            .min(self.deque.len() / ch)
            .min(producer.slots() / ch)
            * ch;
        if n == 0 {
            return;
        }
        // One commit for the whole run of frames: the consumer never observes a partial frame.
        if let Ok(chunk) = producer.write_chunk_uninit(n) {
            chunk.fill_from_iter(self.deque.drain(..n));
        }
    }
}

/// Real-time side of an output device: pulls whole frames from the lock-free ring, holds the
/// last frame on underrun and ramps back to live audio after a starve. All state is
/// closure-local (no locks, no allocation per callback).
struct RingPlayout {
    consumer: rtrb::Consumer<f32>,
    /// Last emitted frame (one value per channel).
    hold: Vec<f32>,
    starving: bool,
    /// Frames left in the post-underrun recovery ramp.
    recover_left: usize,
    /// Length (samples) of the latest callback, read by the worker's ring top-up.
    callback_len: Arc<AtomicUsize>,
}

impl RingPlayout {
    fn new(consumer: rtrb::Consumer<f32>, ch: usize, callback_len: Arc<AtomicUsize>) -> Self {
        Self {
            consumer,
            hold: vec![0.0; ch.max(1)],
            // Start "starved" so the first audio after prefill fades in from silence.
            starving: true,
            recover_left: 0,
            callback_len,
        }
    }

    fn fill(&mut self, data: &mut [f32]) {
        self.callback_len.store(data.len(), Ordering::Relaxed);
        let ch = self.hold.len();
        let avail = (self.consumer.slots() / ch).min(data.len() / ch);
        let mut filled = 0usize;
        if avail > 0 {
            if let Ok(chunk) = self.consumer.read_chunk(avail * ch) {
                let (a, b) = chunk.as_slices();
                let mut t = 1.0f32;
                for (i, &v) in a.iter().chain(b.iter()).enumerate() {
                    let c = i % ch;
                    if c == 0 {
                        if self.starving {
                            self.starving = false;
                            self.recover_left = RECOVER_FRAMES;
                        }
                        t = if self.recover_left > 0 {
                            let done = RECOVER_FRAMES - self.recover_left;
                            self.recover_left -= 1;
                            (done + 1) as f32 / RECOVER_FRAMES as f32
                        } else {
                            1.0
                        };
                    }
                    let out = if t < 1.0 {
                        self.hold[c] + (v - self.hold[c]) * t
                    } else {
                        v
                    };
                    data[i] = out;
                    self.hold[c] = out;
                }
                chunk.commit_all();
                filled = avail * ch;
            }
        }
        if filled < data.len() {
            self.starving = true;
            for (i, s) in data.iter_mut().enumerate().skip(filled) {
                *s = self.hold[i % ch];
            }
        }
    }
}

#[derive(Clone)]
pub struct VoiceChangerState {
    enabled: Arc<Mutex<bool>>,
    hot_params: Arc<Mutex<VoiceChangerHotParams>>,
    output_device: Arc<Mutex<String>>,
    mode: Arc<Mutex<VoiceChangerMode>>,
    sidecar_ws_url: Arc<Mutex<String>>,
    onnx_model_path: Arc<Mutex<String>>,
    onnx_execution_provider: Arc<Mutex<String>>,
    latency_preset: Arc<Mutex<String>>,
    monitor_output_device: Arc<Mutex<String>>,
    rvc: Arc<Mutex<RvcParamsState>>,
    stop_sender: Arc<Mutex<Option<Sender<()>>>>,
    is_running: Arc<AtomicBool>,
    /// Bumped per `start_voice_changer`; a worker only clears `is_running` / emits `stopped`
    /// when it is still the current generation (a restarted worker must not kill its successor).
    run_generation: Arc<AtomicU64>,
    /// Previous worker thread; the next worker waits (bounded, 3 s) for it to finish before
    /// opening audio devices and detaches it otherwise.
    worker: Arc<Mutex<Option<thread::JoinHandle<()>>>>,
    /// Mono/stereo float32 interleaved at capture rate, mixed before processing.
    aux_feed: Arc<Mutex<Vec<f32>>>,
    /// Capture sample rate while the loop runs (0 = not running).
    capture_sample_rate: Arc<AtomicU32>,
    capture_channels: Arc<AtomicU32>,
    output_sample_rate: Arc<AtomicU32>,
    output_channels: Arc<AtomicU32>,
    metrics_process_us: Arc<AtomicU32>,
    /// Exponential moving average of per-chunk processing time (µs) for stable UI.
    metrics_process_ema_us: Arc<AtomicU32>,
    metrics_buffer_samples: Arc<AtomicU32>,
}

impl VoiceChangerState {
    fn new() -> Self {
        Self {
            enabled: Arc::new(Mutex::new(false)),
            hot_params: Arc::new(Mutex::new(VoiceChangerHotParams::default())),
            output_device: Arc::new(Mutex::new("default".to_string())),
            mode: Arc::new(Mutex::new(VoiceChangerMode::Classic)),
            sidecar_ws_url: Arc::new(Mutex::new(String::new())),
            onnx_model_path: Arc::new(Mutex::new(String::new())),
            onnx_execution_provider: Arc::new(Mutex::new("cpu".to_string())),
            latency_preset: Arc::new(Mutex::new("low_latency".to_string())),
            monitor_output_device: Arc::new(Mutex::new(String::new())),
            rvc: Arc::new(Mutex::new(RvcParamsState::default())),
            stop_sender: Arc::new(Mutex::new(None)),
            is_running: Arc::new(AtomicBool::new(false)),
            run_generation: Arc::new(AtomicU64::new(0)),
            worker: Arc::new(Mutex::new(None)),
            aux_feed: Arc::new(Mutex::new(Vec::new())),
            capture_sample_rate: Arc::new(AtomicU32::new(0)),
            capture_channels: Arc::new(AtomicU32::new(0)),
            output_sample_rate: Arc::new(AtomicU32::new(0)),
            output_channels: Arc::new(AtomicU32::new(0)),
            metrics_process_us: Arc::new(AtomicU32::new(0)),
            metrics_process_ema_us: Arc::new(AtomicU32::new(0)),
            metrics_buffer_samples: Arc::new(AtomicU32::new(0)),
        }
    }

    /// Clear what a running worker publishes (device rates, metrics, pending aux audio), so a
    /// stopped voice changer reports "not running" to aux/TTS pushes and the metrics UI.
    fn reset_run_outputs(&self) {
        self.capture_sample_rate.store(0, Ordering::Relaxed);
        self.capture_channels.store(0, Ordering::Relaxed);
        self.output_sample_rate.store(0, Ordering::Relaxed);
        self.output_channels.store(0, Ordering::Relaxed);
        self.metrics_process_us.store(0, Ordering::Relaxed);
        self.metrics_process_ema_us.store(0, Ordering::Relaxed);
        self.metrics_buffer_samples.store(0, Ordering::Relaxed);
        if let Ok(mut aux) = self.aux_feed.lock() {
            aux.clear();
        }
    }
}

fn linear_resample_interleaved_into(
    input: &[f32],
    channels: usize,
    from_sr: u32,
    to_sr: u32,
    out: &mut Vec<f32>,
) {
    if from_sr == to_sr || channels == 0 {
        out.clear();
        out.extend_from_slice(input);
        return;
    }
    let frames_in = input.len() / channels;
    let frames_out = ((frames_in as f64) * (to_sr as f64) / (from_sr as f64)).ceil() as usize;
    if frames_out == 0 {
        out.clear();
        return;
    }
    let len = frames_out * channels;
    out.clear();
    out.resize(len, 0.0f32);
    for fo in 0..frames_out {
        let src_pos = fo as f64 * from_sr as f64 / to_sr as f64;
        let base = src_pos.floor() as usize;
        let frac = (src_pos - base as f64) as f32;
        for c in 0..channels {
            let idx = base.saturating_mul(channels).saturating_add(c);
            let a = input.get(idx).copied().unwrap_or(0.0);
            let b = input
                .get(idx.saturating_add(channels))
                .copied()
                .unwrap_or(a);
            out[fo * channels + c] = a + (b - a) * frac;
        }
    }
}

fn linear_resample_interleaved(input: &[f32], channels: usize, from_sr: u32, to_sr: u32) -> Vec<f32> {
    let mut out = Vec::new();
    linear_resample_interleaved_into(input, channels, from_sr, to_sr, &mut out);
    out
}

fn mix_aux_into_chunk(chunk: &mut [f32], aux: &mut Vec<f32>) {
    let take = chunk.len().min(aux.len());
    for i in 0..take {
        chunk[i] = (chunk[i] + aux[i]).clamp(-1.0, 1.0);
    }
    if take > 0 {
        aux.drain(..take);
    }
}

fn adapt_channels_interleaved_into(data: &[f32], from_ch: usize, to_ch: usize, out: &mut Vec<f32>) {
    out.clear();
    if from_ch == 0 || to_ch == 0 || data.is_empty() || data.len() % from_ch != 0 {
        return;
    }
    let frames = data.len() / from_ch;
    out.reserve(frames * to_ch);
    for f in 0..frames {
        let base = f * from_ch;
        let frame = &data[base..base + from_ch];
        match (from_ch, to_ch) {
            (a, b) if a == b => out.extend_from_slice(frame),
            (_, 1) => {
                let s: f32 = frame.iter().sum::<f32>() / from_ch as f32;
                out.push(s);
            }
            (1, _) => {
                for _ in 0..to_ch {
                    out.push(frame[0]);
                }
            }
            _ => {
                for c in 0..to_ch {
                    out.push(frame[c % from_ch]);
                }
            }
        }
    }
}

#[command]
fn get_voice_changer_params(state: State<'_, VoiceChangerState>) -> Result<VoiceChangerParams, String> {
    let h = *state.hot_params.lock().map_err(|_| "Lock failed")?;
    let rvc = state.rvc.lock().map_err(|_| "Lock failed")?.clone();
    Ok(VoiceChangerParams {
        enabled: *state.enabled.lock().map_err(|_| "Lock failed")?,
        pitch_semitones: h.pitch_semitones,
        formant_shift: h.formant_shift,
        vocoder_window_ms: h.vocoder_window_ms,
        vocoder_oversample: h.vocoder_oversample,
        chunk_frames: h.chunk_frames,
        input_gain: h.input_gain,
        output_gain: h.output_gain,
        output_prefill_ms: h.output_prefill_ms,
        output_max_queue_ms: h.output_max_queue_ms,
        buffer_truncate_enabled: h.buffer_truncate_enabled,
        buffer_truncate_rms: h.buffer_truncate_rms,
        buffer_truncate_chunks: h.buffer_truncate_chunks,
        output_device: state
            .output_device
            .lock()
            .map_err(|_| "Lock failed")?
            .clone(),
        mode: state.mode.lock().map_err(|_| "Lock failed")?.clone(),
        sidecar_ws_url: state
            .sidecar_ws_url
            .lock()
            .map_err(|_| "Lock failed")?
            .clone(),
        onnx_model_path: state
            .onnx_model_path
            .lock()
            .map_err(|_| "Lock failed")?
            .clone(),
        onnx_execution_provider: state
            .onnx_execution_provider
            .lock()
            .map_err(|_| "Lock failed")?
            .clone(),
        latency_preset: state
            .latency_preset
            .lock()
            .map_err(|_| "Lock failed")?
            .clone(),
        monitor_output_device: state
            .monitor_output_device
            .lock()
            .map_err(|_| "Lock failed")?
            .clone(),
        monitor_gain: h.monitor_gain,
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
    })
}

fn clamp_output_buffer_ms(prefill: f64, max_q: f64) -> (f64, f64) {
    let prefill = prefill.clamp(5.0, 300.0);
    let max_q = max_q.clamp(prefill + 5.0, 600.0);
    (prefill, max_q)
}

#[command]
fn set_voice_changer_params(state: State<'_, VoiceChangerState>, params: VoiceChangerParams) -> Result<(), String> {
    *state.enabled.lock().map_err(|_| "Lock failed")? = params.enabled;
    let lp = params.latency_preset.to_lowercase();
    let (pf, mq) = if lp == "stable" {
        clamp_output_buffer_ms(PREFILL_MS_STABLE, MAX_QUEUE_MS_STABLE)
    } else {
        clamp_output_buffer_ms(params.output_prefill_ms, params.output_max_queue_ms)
    };
    {
        let mut h = state.hot_params.lock().map_err(|_| "Lock failed")?;
        h.pitch_semitones = params.pitch_semitones.clamp(-24.0, 24.0);
        h.formant_shift = params.formant_shift.clamp(-1.0, 1.0);
        h.vocoder_window_ms = params.vocoder_window_ms.clamp(30, 60);
        h.vocoder_oversample = match params.vocoder_oversample {
            1 | 2 | 4 | 8 | 16 | 32 => params.vocoder_oversample,
            _ => DEFAULT_VOCODER_OVERSAMPLE,
        };
        h.chunk_frames = clamp_chunk_frames(params.chunk_frames) as u32;
        h.input_gain = params.input_gain.clamp(0.0, 4.0);
        h.output_gain = params.output_gain.clamp(0.0, 4.0);
        h.output_prefill_ms = pf;
        h.output_max_queue_ms = mq;
        h.buffer_truncate_enabled = params.buffer_truncate_enabled;
        h.buffer_truncate_rms = params.buffer_truncate_rms.clamp(1e-8, 1.0);
        h.buffer_truncate_chunks = params.buffer_truncate_chunks.clamp(1, 500);
        h.monitor_gain = params.monitor_gain.clamp(0.0, 4.0);
    }
    *state.output_device.lock().map_err(|_| "Lock failed")? = params.output_device;
    *state.mode.lock().map_err(|_| "Lock failed")? = params.mode;
    *state.sidecar_ws_url.lock().map_err(|_| "Lock failed")? = params.sidecar_ws_url;
    *state.onnx_model_path.lock().map_err(|_| "Lock failed")? = params.onnx_model_path;
    *state.onnx_execution_provider.lock().map_err(|_| "Lock failed")? = params
        .onnx_execution_provider
        .to_lowercase();
    *state.monitor_output_device.lock().map_err(|_| "Lock failed")? =
        params.monitor_output_device;
    *state.latency_preset.lock().map_err(|_| "Lock failed")? = if lp == "stable" {
        "stable".into()
    } else if lp == "custom" {
        "custom".into()
    } else {
        "low_latency".into()
    };
    {
        let mut rvc = state.rvc.lock().map_err(|_| "Lock failed")?;
        rvc.slot_dir = params.rvc_slot_dir;
        rvc.hubert_path = params.rvc_hubert_path;
        rvc.fcpe_path = params.rvc_fcpe_path;
        rvc.rmvpe_path = params.rvc_rmvpe_path;
        rvc.preset = {
            let p = params.rvc_preset.to_lowercase();
            match p.as_str() {
                "low_latency" | "quality" | "streamer" => p,
                _ => "streamer".into(),
            }
        };
        rvc.pitch_extractor = {
            let p = params.pitch_extractor.to_lowercase();
            match p.as_str() {
                "fcpe" | "rmvpe" | "crepe" | "dio" => p,
                _ => "fcpe".into(),
            }
        };
        rvc.extra_context_ms = params.extra_context_ms.clamp(0.0, 2000.0);
        rvc.crossfade_ms = params.crossfade_ms.clamp(0.0, 200.0);
        rvc.index_rate = params.index_rate.clamp(0.0, 1.0);
        rvc.protect = params.protect.clamp(0.0, 0.5);
        rvc.speaker_id = params.speaker_id.max(0);
        rvc.gpu_device_index = params.gpu_device_index.max(0);
        rvc.silence_skip = params.silence_skip;
        rvc.vram_cap_mb = params.vram_cap_mb.clamp(256, 8192);
        rvc.execution_provider = normalize_execution_provider(&params.rvc_execution_provider);
    }
    Ok(())
}

#[command]
fn get_voice_changer_metrics(state: State<'_, VoiceChangerState>) -> Result<VoiceChangerMetrics, String> {
    let r = state.output_sample_rate.load(Ordering::Relaxed);
    let ch = state.output_channels.load(Ordering::Relaxed).max(1) as f64;
    let buf_n = state.metrics_buffer_samples.load(Ordering::Relaxed) as f64;
    let buffer_ms = if r > 0 {
        buf_n / ch / r as f64 * 1000.0
    } else {
        0.0
    };
    let ema = state.metrics_process_ema_us.load(Ordering::Relaxed);
    let raw = state.metrics_process_us.load(Ordering::Relaxed);
    let proc_us = if ema > 0 { ema } else { raw } as f64;
    let chunk_frames = state
        .hot_params
        .lock()
        .map(|h| h.chunk_frames as f64)
        .unwrap_or(DEFAULT_CHUNK_FRAMES as f64);
    let mut chunk_sr = state.capture_sample_rate.load(Ordering::Relaxed) as f64;
    if chunk_sr <= 0.0 {
        chunk_sr = state.output_sample_rate.load(Ordering::Relaxed) as f64;
    }
    let chunk_ms = if chunk_sr > 0.0 {
        chunk_frames / chunk_sr * 1000.0
    } else {
        0.0
    };
    Ok(VoiceChangerMetrics {
        buffer_ms,
        process_ms: proc_us / 1000.0,
        chunk_ms,
    })
}

#[command]
fn set_voice_changer_enabled(state: State<'_, VoiceChangerState>, enabled: bool) -> Result<(), String> {
    *state.enabled.lock().map_err(|_| "Lock failed")? = enabled;
    Ok(())
}

fn voice_changer_push_aux_pcm_impl(
    state: &VoiceChangerState,
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    let r_target = state.capture_sample_rate.load(Ordering::Relaxed);
    if r_target == 0 {
        return Err("Voice changer is not running".into());
    }
    let ch = channels.max(1) as usize;
    let ch_cap = state.capture_channels.load(Ordering::Relaxed).max(1) as usize;
    let resampled = if sample_rate != r_target || ch != ch_cap {
        let frames = samples.len() / ch;
        let mut planar = Vec::with_capacity(frames * ch_cap);
        for f in 0..frames {
            for c in 0..ch_cap {
                let v = if c < ch {
                    samples[f * ch + c]
                } else {
                    samples[f * ch + ch - 1]
                };
                planar.push(v);
            }
        }
        linear_resample_interleaved(&planar, ch_cap, sample_rate, r_target)
    } else {
        samples
    };
    let mut aux = state.aux_feed.lock().map_err(|_| "Lock failed")?;
    aux.extend_from_slice(&resampled);
    const MAX_AUX_SAMPLES: usize = 48000 * 2 * 2;
    if aux.len() > MAX_AUX_SAMPLES {
        let drop = aux.len() - MAX_AUX_SAMPLES;
        aux.drain(..drop);
    }
    Ok(())
}

/// Push decoded f32 PCM (interleaved) to be mixed into the live capture stream at `sample_rate` Hz.
#[command]
fn voice_changer_push_aux_pcm(
    state: State<'_, VoiceChangerState>,
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    voice_changer_push_aux_pcm_impl(&state, samples, sample_rate, channels)
}

/// WAV (PCM) from TTS etc., mixed into the voice changer input while running.
#[command]
fn voice_changer_push_aux_wav(
    state: State<'_, VoiceChangerState>,
    req: PushAuxWavRequest,
) -> Result<(), String> {
    let raw = decode_push_aux_wav_payload(req)?;
    push_aux_wav_bytes(&state, raw)
}

/// A worker's run generation: it is current while no later `start_voice_changer` has claimed a
/// newer one. Only the current worker may change shared state or show the user anything.
#[derive(Clone)]
struct RunGeneration {
    counter: Arc<AtomicU64>,
    mine: u64,
}

impl RunGeneration {
    fn is_current(&self) -> bool {
        self.counter.load(Ordering::SeqCst) == self.mine
    }
}

#[command]
async fn start_voice_changer<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoiceChangerState>,
    input_device: Option<String>,
) -> Result<(), String> {
    if state.is_running.load(Ordering::Acquire) {
        return Err("Voice changer already running".to_string());
    }

    let mode = state.mode.lock().map_err(|_| "Lock failed")?.clone();
    if mode == VoiceChangerMode::Onnx {
        let path = state
            .onnx_model_path
            .lock()
            .map_err(|_| "Lock failed")?
            .clone();
        let p = path.trim();
        if p.is_empty() {
            return Err("Choose an ONNX model file for ONNX voice mode.".into());
        }
        if !Path::new(p).is_file() {
            return Err(format!("ONNX model file not found: {}", p));
        }
    }
    if mode == VoiceChangerMode::Sidecar {
        let url = state.sidecar_ws_url.lock().map_err(|_| "Lock failed")?.clone();
        if url.trim().is_empty() {
            return Err("Enter a WebSocket URL for external voice conversion (ws:// or wss://).".into());
        }
    }
    if mode == VoiceChangerMode::Rvc {
        // Same rule as the worker and the inspector: resolve the effective RVC EP (preset default
        // only when unset); on the CPU EP a GPU-only F0 runs as fcpe, so preflight checks the fcpe
        // file. The stored choice is kept: the worker falls back and emits `rvc_warning`, and a
        // later switch to CUDA/DML uses the chosen extractor again.
        let mut resolved = state.rvc.lock().map_err(|_| "Lock failed")?.clone();
        let app_data = app.path().app_data_dir().map_err(|e| e.to_string())?;
        resolve_rvc_paths(&mut resolved, &app_data)?;
        // Persist resolved paths so the UI sees the files in use (never the F0 choice). The worker
        // does not rely on them: it resolves again from its own snapshot.
        {
            let mut r = state.rvc.lock().map_err(|_| "Lock failed")?;
            r.slot_dir = resolved.slot_dir;
            r.hubert_path = resolved.hubert_path;
            r.fcpe_path = resolved.fcpe_path;
            r.rmvpe_path = resolved.rmvpe_path;
        }
    }

    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }

    let (tx, rx) = channel();
    *state.stop_sender.lock().unwrap() = Some(tx);
    // Claim the generation before publishing `is_running`, so a still-live predecessor can
    // never pass its exit-guard compare after we set the flag.
    let my_generation = state.run_generation.fetch_add(1, Ordering::SeqCst) + 1;
    let generation = RunGeneration {
        counter: state.run_generation.clone(),
        mine: my_generation,
    };
    state.is_running.store(true, Ordering::Release);
    *state.enabled.lock().unwrap() = true;
    let previous_worker = state.worker.lock().unwrap().take();

    let hot_params = state.hot_params.clone();
    let is_running = state.is_running.clone();
    let mode_arc = state.mode.clone();
    let sidecar_url = state.sidecar_ws_url.clone();
    let onnx_model_path = state.onnx_model_path.clone();
    let onnx_execution_provider = state.onnx_execution_provider.clone();
    let rvc_params = state.rvc.clone();
    let aux_feed = state.aux_feed.clone();
    let capture_sr = state.capture_sample_rate.clone();
    let capture_ch = state.capture_channels.clone();
    let output_sr = state.output_sample_rate.clone();
    let output_ch = state.output_channels.clone();
    let metrics_process_us = state.metrics_process_us.clone();
    let metrics_process_ema_us = state.metrics_process_ema_us.clone();
    let metrics_buffer_samples = state.metrics_buffer_samples.clone();
    let output_device_name = state
        .output_device
        .lock()
        .map_err(|_| "Lock failed")?
        .clone();
    let monitor_output_device = state
        .monitor_output_device
        .lock()
        .map_err(|_| "Lock failed")?
        .clone();
    let input_device_name = input_device.clone();
    let app_for_thread = app.clone();
    let run_state = state.inner().clone();

    let handle = thread::spawn(move || {
        #[cfg(windows)]
        {
            use windows::Win32::System::Threading::{
                GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_BELOW_NORMAL,
            };
            unsafe {
                let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL);
            }
        }

        // Wait (bounded) for the previous worker to release its devices/sessions before opening
        // ours. A worker blocked in an untimed sidecar read must not pin every future start, so
        // give up after the deadline and leave it detached instead of joining forever.
        if let Some(prev) = previous_worker {
            let deadline = Instant::now() + Duration::from_secs(3);
            while !prev.is_finished() && Instant::now() < deadline {
                thread::sleep(Duration::from_millis(5));
            }
            if prev.is_finished() {
                let _ = prev.join();
            } else {
                eprintln!("[VoiceChanger] previous worker still busy after 3s; continuing without it");
            }
        }

        // Superseded while waiting: do not open devices or load models.
        if !generation.is_current() {
            return;
        }
        // Stopped while waiting: honour it here. The queued message is only read at the loop head,
        // which is after the devices are opened, the rates republished and (RVC) the models loaded
        // — i.e. seconds of live microphone after the user pressed Stop.
        if rx.try_recv().is_ok() {
            if generation.is_current() {
                if let Ok(mut e) = run_state.enabled.lock() {
                    *e = false;
                }
                is_running.store(false, Ordering::Release);
                let _ = app_for_thread.emit("voice_changer:stopped", ());
            }
            return;
        }
        // `join()` orders the predecessor's final stores before this point; re-assert the flags to
        // close the remaining check-then-store window in its exit guard (which clears both).
        if let Ok(mut e) = run_state.enabled.lock() {
            *e = true;
        }
        is_running.store(true, Ordering::Release);

        // A panic anywhere in the loop (model, DSP, device glue) must still reach the epilogue
        // below; otherwise `is_running` stays true with no worker, the rates stay published and
        // the UI never sees `stopped`. The cpal streams and sessions are dropped by the unwind.
        let app_for_loop = app_for_thread.clone();
        let is_running_for_loop = is_running.clone();
        let generation_for_loop = generation.clone();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            run_voice_changer_loop(
                app_for_loop,
                generation_for_loop,
                rx,
                hot_params,
                is_running_for_loop,
                mode_arc,
                sidecar_url,
                onnx_model_path,
                onnx_execution_provider,
                rvc_params,
                aux_feed,
                capture_sr,
                capture_ch,
                output_sr,
                output_ch,
                metrics_process_us,
                metrics_process_ema_us,
                metrics_buffer_samples,
                input_device_name,
                output_device_name,
                monitor_output_device,
            )
        }))
        .map_err(|payload| {
            format!(
                "worker panicked: {}; voice changer stopped",
                panic_message(payload.as_ref())
            )
        });
        let panicked = result.is_err();
        let result = result.and_then(|r| r);

        if let Err(e) = &result {
            eprintln!("[VoiceChanger] Error: {}", e);
        }

        // Epilogue for Ok, Err and a caught panic alike (an early `return Err` after the rates
        // were published must not leave them non-zero while stopped). A superseded worker must
        // not clear the flags, zero the rates/metrics its successor published (zeroed rates make
        // `voice_changer_push_aux_pcm_impl` reject every TTS/aux push) or report "stopped".
        if generation.is_current() {
            if panicked {
                // Own code so the UI does not show the WebSocket-sidecar toast for a crash.
                if let Err(e) = &result {
                    emit_vc_error(&app_for_thread, "worker_panic", e.clone());
                }
            }
            run_state.reset_run_outputs();
            // A worker that stopped itself (load error, device error, panic) leaves `enabled`
            // false too, as `stop_voice_changer` does, so `get_voice_changer_params` (e.g. after a
            // webview reload) does not report a stopped voice changer as switched on.
            if let Ok(mut e) = run_state.enabled.lock() {
                *e = false;
            }
            is_running.store(false, Ordering::Release);
            let _ = app_for_thread.emit("voice_changer:stopped", ());
        }
    });
    *state.worker.lock().unwrap() = Some(handle);

    let _ = app.emit("voice_changer:started", ());
    Ok(())
}

/// Frames of lead a run of classic-fallback ticks keeps in hand. Per tick, the classic resampler's
/// output and the harness's owed frames follow the same average rate but round differently (a
/// frame or two either way when the device rates differ); sizing each tick on its own padded a
/// gap into every short tick and cut a frame from every long one (crackle in degraded mode).
const FALLBACK_LEAD_FRAMES: usize = 4;
/// Frames held beyond the lead that count as a real excess (dropped, oldest first).
const FALLBACK_SLACK_FRAMES: usize = 8;

/// Joins the classic chain's output across a run of failed RVC ticks so each tick plays exactly
/// the device frames the harness owed (`last_owed_frames()`), as a converted tick does, and pads
/// or drops only on a real shortage or excess.
struct FallbackFifo {
    buf: VecDeque<f32>,
    ch: usize,
    in_run: bool,
}

impl FallbackFifo {
    fn new(ch: usize) -> Self {
        let ch = ch.max(1);
        Self {
            buf: VecDeque::with_capacity((FALLBACK_LEAD_FRAMES + FALLBACK_SLACK_FRAMES + 1) * ch),
            ch,
            in_run: false,
        }
    }

    /// Whether the previous tick was a fallback tick too.
    fn in_run(&self) -> bool {
        self.in_run
    }

    /// A converted tick ends the run; the next failure starts a new one from silence.
    fn end_run(&mut self) {
        self.in_run = false;
        self.buf.clear();
    }

    /// Queue one tick of classic output and return exactly `owed_frames` frames. A run starts
    /// with the lead as silence. A real shortage (the classic chain's start-up delay on a run's
    /// first tick, or a tick that produced too little) is padded with leading silence that also
    /// restores the lead; more than lead + slack left over is a real excess, dropped oldest first.
    fn take(&mut self, produced: &[f32], owed_frames: usize) -> Vec<f32> {
        let ch = self.ch;
        let lead = FALLBACK_LEAD_FRAMES * ch;
        if !self.in_run {
            self.in_run = true;
            self.buf.clear();
            self.buf.extend(std::iter::repeat(0.0).take(lead));
        }
        self.buf
            .extend(produced[..produced.len() / ch * ch].iter().copied());
        let target = owed_frames * ch;
        if self.buf.len() < target {
            let pad = target + lead - self.buf.len();
            for _ in 0..pad {
                self.buf.push_front(0.0);
            }
        }
        let out: Vec<f32> = self.buf.drain(..target).collect();
        if self.buf.len() > lead + FALLBACK_SLACK_FRAMES * ch {
            let excess = self.buf.len() - lead;
            self.buf.drain(..excess);
        }
        out
    }
}

/// Message of a caught worker panic (`&str` / `String` payloads; anything else is opaque).
fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

#[allow(clippy::too_many_arguments)]
fn run_voice_changer_loop<R: Runtime>(
    app: AppHandle<R>,
    generation: RunGeneration,
    rx: std::sync::mpsc::Receiver<()>,
    hot_params: Arc<Mutex<VoiceChangerHotParams>>,
    is_running: Arc<AtomicBool>,
    mode_arc: Arc<Mutex<VoiceChangerMode>>,
    sidecar_url: Arc<Mutex<String>>,
    onnx_model_path: Arc<Mutex<String>>,
    onnx_execution_provider: Arc<Mutex<String>>,
    rvc_params: Arc<Mutex<RvcParamsState>>,
    aux_feed: Arc<Mutex<Vec<f32>>>,
    capture_sr: Arc<AtomicU32>,
    capture_ch: Arc<AtomicU32>,
    output_sr: Arc<AtomicU32>,
    output_ch: Arc<AtomicU32>,
    metrics_process_us: Arc<AtomicU32>,
    metrics_process_ema_us: Arc<AtomicU32>,
    metrics_buffer_samples: Arc<AtomicU32>,
    input_device_name: Option<String>,
    output_device_name: String,
    monitor_output_device: String,
) -> Result<(), String> {
    let mut last_vc_error_emit: Option<Instant> = None;
    let host = cpal::default_host();

    let input = resolve_input_device(&host, input_device_name.as_deref())?;
    let output = resolve_output_device(&host, &output_device_name)?;

    let input_config = input.default_input_config().map_err(|e| e.to_string())?;
    let output_config = output.default_output_config().map_err(|e| e.to_string())?;

    let r_in = input_config.sample_rate().0;
    let r_out = output_config.sample_rate().0;
    let ch_in = input_config.channels() as usize;
    let ch_out = output_config.channels() as usize;

    capture_sr.store(r_in, Ordering::Relaxed);
    capture_ch.store(ch_in.max(1) as u32, Ordering::Relaxed);

    #[cfg(debug_assertions)]
    println!(
        "[VoiceChanger] Input:  {} Hz, {} ch | Output: {} Hz, {} ch",
        r_in, ch_in, r_out, ch_out
    );

    if !matches!(output_config.sample_format(), cpal::SampleFormat::F32) {
        return Err(format!(
            "Output sample format {:?} not supported (need F32)",
            output_config.sample_format()
        ));
    }

    output_sr.store(r_out, Ordering::Relaxed);
    output_ch.store(ch_out as u32, Ordering::Relaxed);

    let h_init = *hot_params.lock().map_err(|_| "Lock failed")?;
    let (prefill_ms, max_queue_ms) =
        clamp_output_buffer_ms(h_init.output_prefill_ms, h_init.output_max_queue_ms);
    let prefill_samples = ((r_out as f64) * prefill_ms / 1000.0).ceil() as usize * ch_out;
    let max_queue_samples = ((r_out as f64) * max_queue_ms / 1000.0).ceil() as usize * ch_out;

    let chunk_frames_usize = clamp_chunk_frames(h_init.chunk_frames);
    let need = chunk_frames_usize * ch_in;
    if need == 0 {
        return Err("Invalid chunk size or channel count.".into());
    }

    let ring_cap = need.saturating_mul(64).clamp(16_384, 262_144);
    let (mut audio_producer, mut audio_consumer) = RingBuffer::<f32>::new(ring_cap);
    let is_running_input = is_running.clone();

    let out_ring_cap = max_queue_samples
        .saturating_mul(2)
        .clamp(OUT_RING_MIN_SAMPLES, OUT_RING_MAX_SAMPLES)
        .max(prefill_samples.saturating_add(1));
    let (mut out_producer, out_consumer) = RingBuffer::<f32>::new(out_ring_cap);
    let out_callback_len = Arc::new(AtomicUsize::new(0));
    // Interleaved samples of the latest capture callback: output top-ups only happen when capture
    // delivers, so each playback's ring must also bridge one capture period.
    let in_burst_len = Arc::new(AtomicUsize::new(0));
    let mut playback = OutputPlayback::new(
        prefill_samples,
        max_queue_samples,
        ch_out,
        out_callback_len.clone(),
    )
    .with_capture_burst(in_burst_len.clone(), ch_in, r_in, r_out);

    let input_stream = if matches!(input_config.sample_format(), cpal::SampleFormat::F32) {
        let in_burst_f32 = in_burst_len.clone();
        input
            .build_input_stream(
                &input_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !is_running_input.load(Ordering::Acquire) {
                        return;
                    }
                    in_burst_f32.store(data.len(), Ordering::Relaxed);
                    let (_, rest) = audio_producer.push_partial_slice(data);
                    if !rest.is_empty() {
                        // Ring full: remainder dropped (real-time safe).
                    }
                },
                |e| eprintln!("[VoiceChanger] Input error: {}", e),
                None,
            )
            .map_err(|e| e.to_string())?
    } else if matches!(input_config.sample_format(), cpal::SampleFormat::I16) {
        let is_running_i16 = is_running.clone();
        let in_burst_i16 = in_burst_len.clone();
        input
            .build_input_stream(
                &input_config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !is_running_i16.load(Ordering::Acquire) {
                        return;
                    }
                    in_burst_i16.store(data.len(), Ordering::Relaxed);
                    for &s in data {
                        let f = s as f32 / 32768.0;
                        if audio_producer.push(f).is_err() {
                            break;
                        }
                    }
                },
                |e| eprintln!("[VoiceChanger] Input error: {}", e),
                None,
            )
            .map_err(|e| e.to_string())?
    } else {
        return Err("Unsupported input sample format".to_string());
    };

    let mut out_playout = RingPlayout::new(out_consumer, ch_out, out_callback_len);
    let output_stream = output
        .build_output_stream(
            &output_config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| out_playout.fill(data),
            |e| eprintln!("[VoiceChanger] Output error: {}", e),
            None,
        )
        .map_err(|e| e.to_string())?;

    input_stream.play().map_err(|e| e.to_string())?;
    output_stream.play().map_err(|e| e.to_string())?;

    let (_monitor_stream, mut monitor_playback, mut mon_producer, r_mon_out, ch_mon_out, mon_ring_cap) =
        if monitor_output_device.trim().is_empty() {
            (
                None::<cpal::Stream>,
                None,
                None,
                0u32,
                0usize,
                0usize,
            )
        } else {
            let mon_dev = resolve_output_device(&host, &monitor_output_device)?;
            let mc = mon_dev.default_output_config().map_err(|e| e.to_string())?;
            if !matches!(mc.sample_format(), cpal::SampleFormat::F32) {
                return Err(format!(
                    "Monitor output needs F32 (got {:?})",
                    mc.sample_format()
                ));
            }
            let r_mon = mc.sample_rate().0;
            let ch_mon = mc.channels().max(1) as usize;
            let prefill_m = ((r_mon as f64) * prefill_ms / 1000.0).ceil() as usize * ch_mon;
            let max_q_m = ((r_mon as f64) * max_queue_ms / 1000.0).ceil() as usize * ch_mon;
            let mon_ring_cap = max_q_m
                .saturating_mul(2)
                .clamp(OUT_RING_MIN_SAMPLES, OUT_RING_MAX_SAMPLES)
                .max(prefill_m.saturating_add(1));
            let (mon_prod, mon_cons) = RingBuffer::<f32>::new(mon_ring_cap);
            let mon_callback_len = Arc::new(AtomicUsize::new(0));
            let mon_pb = OutputPlayback::new(prefill_m, max_q_m, ch_mon, mon_callback_len.clone())
                .with_capture_burst(in_burst_len.clone(), ch_in, r_in, r_mon);
            let mut mon_playout = RingPlayout::new(mon_cons, ch_mon, mon_callback_len);
            let m_stream = mon_dev
                .build_output_stream(
                    &mc.into(),
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| mon_playout.fill(data),
                    |e| eprintln!("[VoiceChanger] Monitor output error: {}", e),
                    None,
                )
                .map_err(|e| e.to_string())?;
            m_stream.play().map_err(|e| e.to_string())?;
            (
                Some(m_stream),
                Some(mon_pb),
                Some(mon_prod),
                r_mon,
                ch_mon,
                mon_ring_cap,
            )
        };

    let mode = mode_arc.lock().map_err(|_| "Lock failed")?.clone();
    let mut classic = ClassicPhaseVocoderProcessor::new(
        r_in,
        r_out,
        ch_in,
        ch_out,
        h_init.vocoder_window_ms,
        chunk_frames_usize,
    )?;

    let mut sidecar = if mode == VoiceChangerMode::Sidecar {
        let url = sidecar_url.lock().map_err(|_| "Lock failed")?.clone();
        Some(SidecarVoiceBridge::connect(
            &url,
            r_in,
            ch_in,
            chunk_frames_usize,
        )?)
    } else {
        None
    };

    let mut onnx_vc = if mode == VoiceChangerMode::Onnx {
        let path = onnx_model_path.lock().map_err(|_| "Lock failed")?.clone();
        let ep = onnx_execution_provider
            .lock()
            .map_err(|_| "Lock failed")?
            .clone();
        Some(OnnxVoiceBridge::new(path.trim(), need, ch_in, ep.trim())?)
    } else {
        None
    };

    let mut rvc_harness = if mode == VoiceChangerMode::Rvc {
        // Every RVC load failure is reported once as `rvc_load` before stopping the worker, unless
        // a newer start has superseded this worker (its own load reports what the user asked for).
        let rvc_load_err = |e: String| {
            if generation.is_current() {
                emit_vc_error(&app, "rvc_load", e.clone());
            }
            e
        };
        let mut settings = rvc_params.lock().map_err(|_| "Lock failed")?.clone();
        // The start preflight's resolved paths may already be gone: a params sync between it and
        // this snapshot (pitch slider, field blur) puts the UI's raw paths back, usually "" = auto,
        // and loading "" fails. Resolve again with the same rule as the preflight.
        let app_data = app
            .path()
            .app_data_dir()
            .map_err(|e| rvc_load_err(e.to_string()))?;
        resolve_rvc_paths(&mut settings, &app_data).map_err(&rvc_load_err)?;
        let preset = RvcPresetProfile::from_str(&settings.preset);
        // RVC has its own EP (set by the profile or the expert select); ONNX chunk mode keeps
        // `onnx_execution_provider`. Unset falls back to the preset default.
        let ep_hint = effective_rvc_ep(&settings.execution_provider, preset);
        let session_opts = VcSessionOptions {
            cuda_device_index: settings.gpu_device_index,
            streamer_profile: matches!(preset, RvcPresetProfile::Streamer),
            // The arena cap is a CUDA EP option; DirectML/CPU ignore it.
            vram_cap_mb: if ep_hint == "cuda" {
                Some(settings.vram_cap_mb)
            } else {
                None
            },
        };
        let _int8_ok = crate::services::neural_audio::vc::supports_int8_content_encoder();
        if matches!(preset, RvcPresetProfile::Streamer) && _int8_ok {
            eprintln!("[VoiceChanger] CPU supports int8 content-encoder path (gated for Phase 3 quant)");
        }

        // Same rule as the inspector (which greys rmvpe/crepe out on the CPU EP) and the start
        // preflight: a stale GPU-only choice falls back to fcpe, and the user is told instead of
        // it happening silently.
        let (pitch_name, fell_back) = effective_pitch_extractor(&settings.pitch_extractor, ep_hint);
        if fell_back {
            let msg = format!(
                "{} is GPU-only in Sigil; using fcpe on the CPU execution provider",
                settings.pitch_extractor.trim().to_lowercase()
            );
            eprintln!("[VoiceChanger] {msg}");
            if generation.is_current() {
                emit_vc_error(&app, "rvc_warning", msg);
            }
        }

        let pitch: Box<dyn PitchExtractor> = match pitch_name.as_str() {
            "rmvpe" | "crepe" => Box::new(
                RmvpePitchExtractor::load(&settings.rmvpe_path, ep_hint, &session_opts)
                    .map_err(&rvc_load_err)?,
            ),
            "dio" => Box::new(DioPitchExtractor::new()),
            _ => Box::new(
                FcpePitchExtractor::load(&settings.fcpe_path, ep_hint, &session_opts)
                    .map_err(&rvc_load_err)?,
            ),
        };

        let model = RvcModel::load(
            Path::new(&settings.slot_dir),
            &settings.hubert_path,
            pitch,
            ep_hint,
            &session_opts,
            settings.silence_skip,
        )
        .map_err(&rvc_load_err)?;
        // Non-fatal load issues (manifest inferred, index unreadable, ...) surface as warnings.
        for warning in model.warnings() {
            eprintln!("[VoiceChanger] RVC warning: {warning}");
            if generation.is_current() {
                emit_vc_error(&app, "rvc_warning", warning.clone());
            }
        }
        // No separate warm-up here: `VcStreamHarness::new` warms the model once at the live
        // steady-state input shape (`warmup_at`), so GPU EPs pick kernels for the shape the first
        // spoken tick uses instead of an unused one.
        let model_out_sr = model.output_sample_rate();
        let extra_samples =
            ((settings.extra_context_ms / 1000.0) * 16_000.0).round().max(0.0) as usize;
        let crossfade_samples = ((settings.crossfade_ms / 1000.0) * model_out_sr as f64)
            .round()
            .max(0.0) as usize;

        let mut harness = VcStreamHarness::new(
            Box::new(model),
            r_in,
            r_out,
            ch_in,
            ch_out,
            extra_samples,
            crossfade_samples,
            chunk_frames_usize,
        )
        .map_err(&rvc_load_err)?;
        // The profile sets index rate in the UI (Quality 0.5); an explicit 0 is honoured, as the
        // per-tick set_controls below already does.
        harness.set_controls(
            h_init.pitch_semitones,
            settings.index_rate,
            settings.protect,
            settings.speaker_id,
        );
        Some(harness)
    } else {
        None
    };
    // Formant tilt for RVC output (the model has no formant control), per output channel at r_out.
    let mut rvc_formant_states = vec![FormantLpState::default(); ch_out.max(1)];
    // RVC degraded mode: the classic chain's output across a run of failed ticks, and whether that
    // chain has run since it was built (its delay line then holds the end of an earlier run).
    let mut fallback_fifo = FallbackFifo::new(ch_out);
    let mut classic_used = false;

    let mut work_chunk = Vec::with_capacity(need);
    let mut scratch_mon = Vec::with_capacity(need);
    let mut scratch_adapt = Vec::with_capacity(need * 2);
    let mut scratch_resample = Vec::with_capacity(need * 2);

    'vc_main: loop {
        if rx.try_recv().is_ok() || !is_running.load(Ordering::Acquire) {
            break;
        }

        let h = *hot_params.lock().unwrap();
        let pitch = h.pitch_semitones;
        let formant = h.formant_shift;
        let window_ms = h.vocoder_window_ms;
        let oversample = h.vocoder_oversample;
        let g_in = h.input_gain;
        let g_out = h.output_gain;
        let g_mon = h.monitor_gain;
        let te = h.buffer_truncate_enabled;
        let thr = h.buffer_truncate_rms;
        let n_trunc = h.buffer_truncate_chunks;

        let (pf_ms, mq_ms) = clamp_output_buffer_ms(h.output_prefill_ms, h.output_max_queue_ms);
        let prefill_samples = ((r_out as f64) * pf_ms / 1000.0).ceil() as usize * ch_out;
        let max_queue_samples = ((r_out as f64) * mq_ms / 1000.0).ceil() as usize * ch_out;
        let ring_used = out_ring_cap.saturating_sub(out_producer.slots());
        playback.update_limits(prefill_samples, max_queue_samples, ring_used);
        if let (Some(mon_pb), Some(mon_prod)) = (monitor_playback.as_mut(), mon_producer.as_ref()) {
            if r_mon_out > 0 && ch_mon_out > 0 {
                let prefill_m = ((r_mon_out as f64) * pf_ms / 1000.0).ceil() as usize * ch_mon_out;
                let max_q_m = ((r_mon_out as f64) * mq_ms / 1000.0).ceil() as usize * ch_mon_out;
                let mon_ring_used = mon_ring_cap.saturating_sub(mon_prod.slots());
                mon_pb.update_limits(prefill_m, max_q_m, mon_ring_used);
            }
        }

        work_chunk.clear();
        while work_chunk.len() < need {
            match audio_consumer.pop() {
                Ok(s) => work_chunk.push(s),
                Err(_) => {
                    thread::sleep(std::time::Duration::from_millis(2));
                    if rx.try_recv().is_ok() || !is_running.load(Ordering::Acquire) {
                        break 'vc_main;
                    }
                }
            }
        }

        if (g_in - 1.0).abs() > 1e-6 {
            for s in work_chunk.iter_mut() {
                *s = (*s * g_in).clamp(-1.0, 1.0);
            }
        }
        {
            let mut aux = aux_feed.lock().unwrap();
            mix_aux_into_chunk(&mut work_chunk, &mut aux);
        }

        let t0 = Instant::now();
        let processed = if let Some(ref mut ob) = onnx_vc {
            let converted = match ob.infer(&work_chunk) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("[VoiceChanger] ONNX: {}", e);
                    maybe_emit_vc_error(
                        &app,
                        "onnx_infer",
                        e.clone(),
                        &mut last_vc_error_emit,
                    );
                    work_chunk.clone()
                }
            };
            VoiceChunkPipeline::process_chunk(
                &mut classic,
                &converted,
                pitch,
                formant,
                window_ms,
                oversample,
            )?
        } else if let Some(ref mut bridge) = sidecar {
            let neural = match bridge.exchange(&work_chunk) {
                Ok(v) if v.len() == need => v,
                Ok(v) => {
                    eprintln!(
                        "[VoiceChanger] Sidecar: expected {} samples, got {}",
                        need,
                        v.len()
                    );
                    maybe_emit_vc_error(
                        &app,
                        "sidecar_length",
                        format!("expected {} samples, got {}", need, v.len()),
                        &mut last_vc_error_emit,
                    );
                    work_chunk.clone()
                }
                Err(e) => {
                    eprintln!("[VoiceChanger] Sidecar: {}", e);
                    maybe_emit_vc_error(&app, "sidecar_exchange", e.clone(), &mut last_vc_error_emit);
                    work_chunk.clone()
                }
            };
            VoiceChunkPipeline::process_chunk(
                &mut classic,
                &neural,
                pitch,
                formant,
                window_ms,
                oversample,
            )?
        } else if let Some(ref mut harness) = rvc_harness {
            let (index_rate, protect, speaker_id) = {
                let s = rvc_params.lock().unwrap();
                (s.index_rate, s.protect, s.speaker_id)
            };
            // Pitch is applied inside the RVC model; skip classic phase vocoder.
            harness.set_controls(pitch, index_rate, protect, speaker_id);
            let mut out = match harness.push_and_convert(&work_chunk) {
                Ok(v) => {
                    fallback_fifo.end_run();
                    v
                }
                Err(e) => {
                    eprintln!("[VoiceChanger] RVC: {}", e);
                    maybe_emit_vc_error(&app, "rvc_convert", e.clone(), &mut last_vc_error_emit);
                    if !fallback_fifo.in_run() && classic_used {
                        // New run of failed ticks: start the dry chain from silence instead of
                        // replaying the end of the previous run from its resampler/vocoder
                        // history (a rebuild costs a few ms, once per run, in degraded mode only).
                        match ClassicPhaseVocoderProcessor::new(
                            r_in,
                            r_out,
                            ch_in,
                            ch_out,
                            window_ms,
                            chunk_frames_usize,
                        ) {
                            Ok(fresh) => classic = fresh,
                            Err(err) => eprintln!("[VoiceChanger] classic fallback rebuild: {err}"),
                        }
                    }
                    let produced = VoiceChunkPipeline::process_chunk(
                        &mut classic,
                        &work_chunk,
                        0.0,
                        0.0,
                        window_ms,
                        oversample,
                    )?;
                    classic_used = true;
                    // Play exactly the device frames the harness owed for this tick, as on
                    // success, so a failed tick neither shortens nor lengthens the output stream;
                    // the FIFO carries the per-tick rounding across a run of failed ticks.
                    fallback_fifo.take(&produced, harness.last_owed_frames())
                }
            };
            // The model has no formant control: apply the classic spectral tilt to its output
            // (same per-channel filter state on the fallback tick, so the tilt stays continuous).
            apply_formant_interleaved(
                &mut out,
                ch_out,
                formant,
                &mut rvc_formant_states,
                r_out as f64,
            );
            out
        } else {
            VoiceChunkPipeline::process_chunk(
                &mut classic,
                &work_chunk,
                pitch,
                formant,
                window_ms,
                oversample,
            )?
        };

        if let (Some(ref mut mon_pb), Some(ref mut mon_prod)) =
            (monitor_playback.as_mut(), mon_producer.as_mut())
        {
            if g_mon > 1e-6 && ch_mon_out > 0 {
                let mon_src: &[f32] = if (g_mon - 1.0).abs() > 1e-6 {
                    scratch_mon.clear();
                    scratch_mon.reserve(processed.len());
                    for &s in &processed {
                        scratch_mon.push((s * g_mon).clamp(-1.0, 1.0));
                    }
                    scratch_mon.as_slice()
                } else {
                    processed.as_slice()
                };
                adapt_channels_interleaved_into(mon_src, ch_out, ch_mon_out, &mut scratch_adapt);
                let final_mon: &[f32] = if r_mon_out != r_out && !scratch_adapt.is_empty() {
                    linear_resample_interleaved_into(
                        &scratch_adapt,
                        ch_mon_out,
                        r_out,
                        r_mon_out,
                        &mut scratch_resample,
                    );
                    scratch_resample.as_slice()
                } else {
                    scratch_adapt.as_slice()
                };
                if !final_mon.is_empty() {
                    let mon_ring_used = mon_ring_cap.saturating_sub(mon_prod.slots());
                    mon_pb.push_interleaved(final_mon, false, 0.0, 0, mon_ring_used);
                    mon_pb.flush_to_ring(mon_prod, mon_ring_used);
                }
            }
        }

        let mut processed = processed;
        if (g_out - 1.0).abs() > 1e-6 {
            for s in processed.iter_mut() {
                *s = (*s * g_out).clamp(-1.0, 1.0);
            }
        }

        let elapsed_us = t0.elapsed().as_micros().min(u128::from(u32::MAX)) as u32;
        metrics_process_us.store(elapsed_us, Ordering::Relaxed);
        let prev_ema = metrics_process_ema_us.load(Ordering::Relaxed);
        let next_ema = if prev_ema == 0 {
            elapsed_us
        } else {
            ((prev_ema as u64 * 3 + elapsed_us as u64) / 4) as u32
        };
        metrics_process_ema_us.store(next_ema, Ordering::Relaxed);

        let ring_used = out_ring_cap.saturating_sub(out_producer.slots());
        playback.push_interleaved(&processed, te, thr, n_trunc, ring_used);
        playback.flush_to_ring(&mut out_producer, ring_used);
        let ring_used = out_ring_cap.saturating_sub(out_producer.slots());
        let ql = (playback.queue_len() + ring_used).min(u32::MAX as usize) as u32;
        metrics_buffer_samples.store(ql, Ordering::Relaxed);
    }

    // Published rates/metrics are cleared by the worker epilogue in `start_voice_changer`
    // (generation-guarded), which also covers the early `return Err` paths above.
    #[cfg(debug_assertions)]
    println!("[VoiceChanger] Stopped audio streams");
    Ok(())
}

#[command]
fn stop_voice_changer<R: Runtime>(_app: AppHandle<R>, state: State<'_, VoiceChangerState>) -> Result<(), String> {
    *state.enabled.lock().unwrap() = false;
    state.is_running.store(false, Ordering::Release);
    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }
    state.reset_run_outputs();
    Ok(())
}

#[command]
fn is_voice_changer_running(state: State<'_, VoiceChangerState>) -> bool {
    state.is_running.load(Ordering::Acquire)
}

#[derive(Serialize, Clone, Debug)]
pub struct VoiceChangerDebugSnapshot {
    pub mode: String,
    pub buffer_ms: f64,
    pub process_ms: f64,
    pub chunk_ms: f64,
    pub capture_sample_rate: u32,
    pub output_sample_rate: u32,
    pub rvc_slot_dir: String,
    pub rvc_preset: String,
    pub rvc_execution_provider: String,
    pub pitch_extractor: String,
    pub int8_encoder_supported: bool,
}

#[command]
fn capture_voice_changer_debug(
    state: State<'_, VoiceChangerState>,
) -> Result<VoiceChangerDebugSnapshot, String> {
    let r = state.output_sample_rate.load(Ordering::Relaxed);
    let ch = state.output_channels.load(Ordering::Relaxed).max(1) as f64;
    let buf_n = state.metrics_buffer_samples.load(Ordering::Relaxed) as f64;
    let buffer_ms = if r > 0 {
        buf_n / ch / r as f64 * 1000.0
    } else {
        0.0
    };
    let ema = state.metrics_process_ema_us.load(Ordering::Relaxed);
    let raw = state.metrics_process_us.load(Ordering::Relaxed);
    let proc_us = if ema > 0 { ema } else { raw } as f64;
    let chunk_frames = state
        .hot_params
        .lock()
        .map_err(|_| "Lock failed")?
        .chunk_frames
        .max(1) as f64;
    let capture_sr = state.capture_sample_rate.load(Ordering::Relaxed);
    let chunk_sr = if capture_sr > 0 {
        capture_sr as f64
    } else if r > 0 {
        r as f64
    } else {
        0.0
    };
    let chunk_ms = if chunk_sr > 0.0 {
        chunk_frames / chunk_sr * 1000.0
    } else {
        0.0
    };

    let mode = state.mode.lock().map_err(|_| "Lock failed")?.clone();
    let mode_s = match mode {
        VoiceChangerMode::Classic => "classic",
        VoiceChangerMode::Onnx => "onnx",
        VoiceChangerMode::Sidecar => "sidecar",
        VoiceChangerMode::Rvc => "rvc",
    };
    let rvc = state.rvc.lock().map_err(|_| "Lock failed")?.clone();
    Ok(VoiceChangerDebugSnapshot {
        mode: mode_s.into(),
        buffer_ms,
        process_ms: proc_us / 1000.0,
        chunk_ms,
        capture_sample_rate: capture_sr,
        output_sample_rate: r,
        rvc_slot_dir: rvc.slot_dir,
        rvc_preset: rvc.preset,
        rvc_execution_provider: rvc.execution_provider,
        pitch_extractor: rvc.pitch_extractor,
        int8_encoder_supported: crate::services::neural_audio::vc::supports_int8_content_encoder(),
    })
}

#[derive(Serialize, Clone, Debug)]
pub struct VoiceChangerModelSlot {
    pub id: String,
    pub path: String,
    pub has_manifest: bool,
}

#[command]
fn list_voice_changer_model_slots<R: Runtime>(
    app: AppHandle<R>,
) -> Result<Vec<VoiceChangerModelSlot>, String> {
    let dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?
        .join("voice-changer")
        .join("models");
    if !dir.is_dir() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        let id = entry.file_name().to_string_lossy().to_string();
        let has_manifest = path.join("manifest.json").is_file();
        out.push(VoiceChangerModelSlot {
            id,
            path: path.to_string_lossy().to_string(),
            has_manifest,
        });
    }
    out.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(out)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("voice-changer")
        .setup(|app, _api| {
            app.manage(VoiceChangerState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_voice_changer_params,
            set_voice_changer_params,
            set_voice_changer_enabled,
            start_voice_changer,
            stop_voice_changer,
            is_voice_changer_running,
            get_voice_changer_metrics,
            voice_changer_push_aux_pcm,
            voice_changer_push_aux_wav,
            capture_voice_changer_debug,
            list_voice_changer_model_slots,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::{
        adapt_channels_interleaved_into, effective_pitch_extractor, effective_rvc_ep,
        normalize_execution_provider, panic_message, resolve_rvc_paths,
        ClassicPhaseVocoderProcessor, FallbackFifo, OutputPlayback, RingPlayout, RvcParamsState,
        RvcPresetProfile, VoiceChunkPipeline, FALLBACK_LEAD_FRAMES, FALLBACK_SLACK_FRAMES,
        RECOVER_FRAMES,
    };
    use rtrb::RingBuffer;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::{SystemTime, UNIX_EPOCH};

    const RING_CAP: usize = 16_384;

    fn ring_used(p: &rtrb::Producer<f32>) -> usize {
        RING_CAP - p.slots()
    }

    /// 5 ms prefill + 160-frame chunks vs. a 10 ms (960-sample stereo) WASAPI callback: the
    /// ring must cover a whole callback, so after start-up no callback runs dry.
    #[test]
    fn ring_top_up_covers_long_device_callback() {
        let ch = 2usize;
        let prefill = 240 * ch; // 5 ms @ 48 kHz
        let max_queue = 480 * ch; // 10 ms
        let chunk = vec![0.25f32; 160 * ch];
        let cb_len = Arc::new(AtomicUsize::new(0));
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(prefill, max_queue, ch, cb_len.clone());
        let mut playout = RingPlayout::new(cons, ch, cb_len.clone());
        let mut data = vec![0.0f32; 480 * ch];
        let mut underrun_periods = 0usize;
        for period in 0..300 {
            // One 10 ms input burst = three 160-frame chunks, then the output callback.
            for _ in 0..3 {
                let used = ring_used(&prod);
                pb.push_interleaved(&chunk, false, 0.0, 0, used);
                let used = ring_used(&prod);
                pb.flush_to_ring(&mut prod, used);
            }
            playout.fill(&mut data);
            if period > 0 && playout.starving {
                underrun_periods += 1;
            }
        }
        assert_eq!(cb_len.load(Ordering::Relaxed), 480 * ch);
        assert_eq!(underrun_periods, 0);
        // Nothing was dropped: produced - consumed stays bounded (= what is still queued).
        assert!(pb.queue_len() + ring_used(&prod) < 2 * 480 * ch);
    }

    /// A 20 ms capture callback (six 160-frame chunks back to back) against 10 ms output
    /// callbacks: nothing tops the ring up between two capture bursts, so the ring must bridge a
    /// whole capture period. Without the capture-burst term the second callback of every period
    /// ran dry however much backlog sat in the deque.
    #[test]
    fn ring_top_up_bridges_long_capture_bursts() {
        let ch = 2usize;
        let prefill = 240 * ch; // 5 ms @ 48 kHz
        let max_queue = 480 * ch; // 10 ms
        let chunk = vec![0.25f32; 160 * ch];
        let cb_len = Arc::new(AtomicUsize::new(0));
        let burst = Arc::new(AtomicUsize::new(0));
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(prefill, max_queue, ch, cb_len.clone())
            .with_capture_burst(burst.clone(), 1, 48_000, 48_000);
        let mut playout = RingPlayout::new(cons, ch, cb_len.clone());
        let mut data = vec![0.0f32; 480 * ch];
        let mut underruns = 0usize;
        for period in 0..300 {
            // Mono capture: 960 interleaved samples = 960 frames = 960 stereo output frames.
            burst.store(960, Ordering::Relaxed);
            for _ in 0..6 {
                let used = ring_used(&prod);
                pb.push_interleaved(&chunk, false, 0.0, 0, used);
                let used = ring_used(&prod);
                pb.flush_to_ring(&mut prod, used);
            }
            for _ in 0..2 {
                playout.fill(&mut data);
                if period > 0 && playout.starving {
                    underruns += 1;
                }
            }
        }
        assert_eq!(pb.in_burst_samples(), 960 * ch);
        assert_eq!(underruns, 0);
        assert!(pb.queue_len() + ring_used(&prod) < 3 * 960 * ch);
    }

    /// The capture burst is converted to the output's rate and channel count.
    #[test]
    fn capture_burst_converts_to_output_samples() {
        let burst = Arc::new(AtomicUsize::new(441 * 2)); // 10 ms stereo @ 44.1 kHz
        let pb = OutputPlayback::new(0, 0, 1, Arc::new(AtomicUsize::new(0)))
            .with_capture_burst(burst, 2, 44_100, 48_000);
        assert_eq!(pb.in_burst_samples(), 480); // 10 ms mono @ 48 kHz
        let unwired = OutputPlayback::new(0, 0, 2, Arc::new(AtomicUsize::new(0)));
        assert_eq!(unwired.in_burst_samples(), 0);
    }

    /// The prefill gate opens at one full device callback when that exceeds the prefill.
    #[test]
    fn prefill_gate_waits_for_one_device_callback() {
        let ch = 2usize;
        let cb_len = Arc::new(AtomicUsize::new(480 * ch)); // 10 ms callback already seen
        let (mut prod, _cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(240 * ch, 960 * ch, ch, cb_len);
        let chunk = vec![0.1f32; 160 * ch];
        for _ in 0..2 {
            pb.push_interleaved(&chunk, false, 0.0, 0, 0);
            pb.flush_to_ring(&mut prod, 0);
            // 160 / 320 frames: past the 240-frame prefill at the second push, still gated.
            assert_eq!(ring_used(&prod), 0);
        }
        pb.push_interleaved(&chunk, false, 0.0, 0, 0);
        pb.flush_to_ring(&mut prod, 0);
        assert_eq!(ring_used(&prod), 480 * ch);
    }

    /// Harness owed frames after `k` ticks of `frames` input frames: the rounded running total,
    /// as `scale_with_carry` (half-unit start) yields it.
    fn owed_total(k: u64, frames: u64, r_in: u64, r_out: u64) -> u64 {
        (k * frames * r_out + r_in / 2) / r_in
    }

    /// Stereo frames numbered `from..from + n` (L = n, R = -n), so order and gaps are visible.
    fn numbered_frames(from: u32, n: u32) -> Vec<f32> {
        (from..from + n).flat_map(|v| [v as f32, -(v as f32)]).collect()
    }

    /// A run of fallback ticks whose classic output rounds differently from the owed frames
    /// (floor vs. round of the same rate, first tick short by the chain's start-up delay) plays
    /// every produced frame once and in order: the only silence is at the start of the run.
    #[test]
    fn fallback_fifo_joins_mismatched_rates_without_gaps() {
        let ch = 2usize;
        let lead = FALLBACK_LEAD_FRAMES;
        for &(r_in, r_out, frames) in &[
            (44_100u64, 48_000u64, 160u64),
            (48_000, 44_100, 160),
            (44_100, 48_000, 512),
            (48_000, 48_000, 512),
            (16_000, 48_000, 160),
        ] {
            let delay = 128 * r_out / r_in;
            let mut fifo = FallbackFifo::new(ch);
            let mut produced_total = 0u64;
            let mut played: Vec<f32> = Vec::new();
            for k in 1..=400u64 {
                let p = (k * frames * r_out / r_in).saturating_sub(delay);
                let produced = numbered_frames(produced_total as u32 + 1, (p - produced_total) as u32);
                produced_total = p;
                let owed = (owed_total(k, frames, r_in, r_out) - owed_total(k - 1, frames, r_in, r_out)) as usize;
                let out = fifo.take(&produced, owed);
                assert_eq!(out.len(), owed * ch, "{r_in}->{r_out} tick {k}");
                for f in out.chunks_exact(ch) {
                    assert_eq!(f[1], -f[0], "{r_in}->{r_out} tick {k}");
                    played.push(f[0]);
                }
            }
            // Leading silence: the lead, plus the start-up shortage padded once on tick 1.
            let (o1, p1) = (owed_total(1, frames, r_in, r_out) as usize, (frames * r_out / r_in - delay) as usize);
            let silence = if o1 > p1 + lead { o1 - p1 + lead } else { lead };
            assert!(played[..silence].iter().all(|&v| v == 0.0), "{r_in}->{r_out}");
            // Then 1, 2, 3, ... with no gap (a pad) and no skip (a drop) anywhere in the run.
            for (i, &v) in played[silence..].iter().enumerate() {
                assert_eq!(v, (i + 1) as f32, "{r_in}->{r_out} frame {}", silence + i);
            }
            // Everything produced was played, except the few frames still held as the lead.
            let held = fifo.buf.len() / ch;
            assert_eq!(played.len() - silence + held, produced_total as usize, "{r_in}->{r_out}");
            assert!(held.abs_diff(lead) <= 1, "{r_in}->{r_out}: {held} held");
        }
    }

    /// The real classic chain against the harness's owed frames: after the run's first tick
    /// (the chain's start-up delay, padded once) no tick is short or over the slack.
    #[test]
    fn fallback_fifo_with_classic_chain_pads_only_the_start_up_delay() {
        for &(r_in, r_out, frames) in &[
            (44_100u32, 48_000u32, 160usize),
            (48_000, 44_100, 160),
            (48_000, 48_000, 512),
        ] {
            let mut classic = ClassicPhaseVocoderProcessor::new(r_in, r_out, 1, 1, 45, frames).unwrap();
            let mut fifo = FallbackFifo::new(1);
            let input: Vec<f32> = (0..frames).map(|i| 0.1 * (i as f32 * 0.05).sin()).collect();
            let (ri, ro, fr) = (u64::from(r_in), u64::from(r_out), frames as u64);
            for k in 1..=200u64 {
                let produced =
                    VoiceChunkPipeline::process_chunk(&mut classic, &input, 0.0, 0.0, 45, 1).unwrap();
                let owed = (owed_total(k, fr, ri, ro) - owed_total(k - 1, fr, ri, ro)) as usize;
                let held = if fifo.in_run() { fifo.buf.len() } else { FALLBACK_LEAD_FRAMES };
                let avail = held + produced.len();
                if k == 1 {
                    assert!(avail < owed, "{r_in}->{r_out}: expected a start-up shortage");
                } else {
                    assert!(avail >= owed, "{r_in}->{r_out} tick {k}: short, {avail} < {owed}");
                    assert!(
                        avail - owed <= FALLBACK_LEAD_FRAMES + FALLBACK_SLACK_FRAMES,
                        "{r_in}->{r_out} tick {k}: excess, {avail} - {owed}"
                    );
                }
                assert_eq!(fifo.take(&produced, owed).len(), owed);
            }
        }
    }

    #[test]
    fn fallback_fifo_pads_real_shortage_and_drops_real_excess() {
        let ch = 2usize;
        let lead = FALLBACK_LEAD_FRAMES;
        let slack = FALLBACK_SLACK_FRAMES;
        let mut fifo = FallbackFifo::new(ch);
        assert!(!fifo.in_run());
        // Run start: the lead as silence, then the produced frames; the lead stays held.
        let out = fifo.take(&numbered_frames(1, 10), 10);
        assert!(fifo.in_run());
        assert_eq!(out.len(), 10 * ch);
        assert!(out[..lead * ch].iter().all(|&v| v == 0.0));
        assert_eq!(out[lead * ch], 1.0);
        assert_eq!(Vec::from(fifo.buf.clone()), numbered_frames(7, 4));
        // Nothing produced: a real shortage, padded with leading silence; the held frames stay
        // held (in order) as the lead.
        let out = fifo.take(&[], 10);
        assert!(out.iter().all(|&v| v == 0.0));
        assert_eq!(Vec::from(fifo.buf.clone()), numbered_frames(7, 4));
        // Twenty frames too many: a real excess, dropped oldest first down to the lead.
        let out = fifo.take(&numbered_frames(11, 30), 10);
        assert_eq!(out, numbered_frames(7, 10));
        assert_eq!(Vec::from(fifo.buf.clone()), numbered_frames(37, 4));
        // Up to lead + slack held over is jitter, not excess: nothing dropped.
        let out = fifo.take(&numbered_frames(41, 10 + slack as u32), 10);
        assert_eq!(out, numbered_frames(37, 10));
        assert_eq!(fifo.buf.len(), (lead + slack) * ch);
        // Owed zero plays nothing; a partial trailing frame is ignored.
        let before = fifo.buf.len();
        assert!(fifo.take(&[0.5], 0).is_empty());
        assert_eq!(fifo.buf.len(), before);
        // A converted tick ends the run: the next failure starts again from the lead's silence.
        fifo.end_run();
        assert!(!fifo.in_run());
        let out = fifo.take(&numbered_frames(100, 10), 10);
        assert!(out[..lead * ch].iter().all(|&v| v == 0.0));
        assert_eq!(out[lead * ch], 100.0);
    }

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

    #[test]
    fn caught_panic_payload_is_reported() {
        let p = std::panic::catch_unwind(|| panic!("boom {}", 7)).unwrap_err();
        assert_eq!(panic_message(p.as_ref()), "boom 7");
        let p = std::panic::catch_unwind(|| panic!("static")).unwrap_err();
        assert_eq!(panic_message(p.as_ref()), "static");
    }

    #[test]
    fn ring_hand_off_and_trims_keep_whole_frames() {
        let ch = 2usize;
        let cb_len = Arc::new(AtomicUsize::new(0));
        let (mut prod, _cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(4 * ch, 8 * ch, ch, cb_len);
        let chunk: Vec<f32> = (0..10 * ch).map(|i| i as f32).collect();
        pb.push_interleaved(&chunk, false, 0.0, 0, 0);
        pb.drop_front(3);
        assert_eq!(pb.queue_len() % ch, 0);
        pb.flush_to_ring(&mut prod, 0);
        assert_eq!(ring_used(&prod) % ch, 0);
        assert_eq!(pb.queue_len() % ch, 0);
    }

    #[test]
    fn playout_ramps_back_after_underrun_per_channel() {
        let ch = 2usize;
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut playout = RingPlayout::new(cons, ch, Arc::new(AtomicUsize::new(0)));
        // Starved callback: holds silence.
        let mut data = vec![1.0f32; 8 * ch];
        playout.fill(&mut data);
        assert!(data.iter().all(|&v| v == 0.0));
        assert!(playout.starving);
        // Live audio L = +1, R = -1 resumes through the ramp, channel by channel.
        for _ in 0..(2 * RECOVER_FRAMES) {
            prod.push(1.0).unwrap();
            prod.push(-1.0).unwrap();
        }
        let mut data = vec![0.0f32; 2 * RECOVER_FRAMES * ch];
        playout.fill(&mut data);
        assert!(!playout.starving);
        let first_l = data[0];
        assert!((first_l - 1.0 / RECOVER_FRAMES as f32).abs() < 1e-6);
        assert!((data[1] + first_l).abs() < 1e-6);
        for f in 1..RECOVER_FRAMES {
            assert!(data[f * ch] >= data[(f - 1) * ch]);
            assert!(data[f * ch] > 0.0 && data[f * ch + 1] < 0.0);
        }
        // Fully recovered at the end of the ramp and afterwards.
        for f in (RECOVER_FRAMES - 1)..(2 * RECOVER_FRAMES) {
            assert_eq!(data[f * ch], 1.0);
            assert_eq!(data[f * ch + 1], -1.0);
        }
    }

    #[test]
    fn execution_provider_normalization() {
        assert_eq!(normalize_execution_provider("CUDA"), "cuda");
        assert_eq!(normalize_execution_provider("gpu"), "cuda");
        assert_eq!(normalize_execution_provider(" dml "), "dml");
        assert_eq!(normalize_execution_provider("tensorrt"), "cpu");
        assert_eq!(normalize_execution_provider(""), "");
    }

    #[test]
    fn device_resample_ratio_standard() {
        let r_in = 44100u32;
        let r_out = 48000u32;
        let ratio = r_out as f64 / r_in as f64;
        assert!((ratio - 48000.0 / 44100.0).abs() < 1e-6);
    }

    #[test]
    fn adapt_channels_mono_to_stereo() {
        let mut o = Vec::new();
        adapt_channels_interleaved_into(&[0.5f32], 1, 2, &mut o);
        assert_eq!(o, vec![0.5, 0.5]);
    }

    #[test]
    fn adapt_channels_stereo_to_mono() {
        let mut o = Vec::new();
        adapt_channels_interleaved_into(&[0.2f32, 0.4f32], 2, 1, &mut o);
        assert!((o[0] - 0.3).abs() < 1e-6);
    }
}
