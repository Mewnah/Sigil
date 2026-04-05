//! Real-time voice changer: capture → **classic** phase vocoder (DSP), optional **WebSocket** converter,
//! or optional **ONNX** chunk model → shared post chain (phase-vocoder pitch, [`rubato`] device match, formant tilt) → playback.

use crate::services::neural_audio::{
    clamp_chunk_frames, ClassicPhaseVocoderProcessor, OnnxVoiceBridge, SidecarVoiceBridge,
    VoiceChunkPipeline, DEFAULT_CHUNK_FRAMES,
};
use base64::{engine::general_purpose::STANDARD as B64_ENGINE, Engine as _};
use std::path::Path;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound::WavReader;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::io::Cursor;
use std::time::Instant;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use rtrb::RingBuffer;
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
#[allow(dead_code)]
const PREFILL_MS_STABLE: f64 = 80.0;
#[allow(dead_code)]
const MAX_QUEUE_MS_STABLE: f64 = 150.0;

/// After an underrun, blend this many **frames** (per channel) from hold → new audio (stereo = 2×).
const RECOVER_FRAMES: usize = 16;

#[derive(Clone, Serialize, Deserialize, Debug, PartialEq, Default)]
#[serde(rename_all = "snake_case")]
pub enum VoiceChangerMode {
    #[default]
    Classic,
    Onnx,
    Sidecar,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VoiceChangerPreset {
    pub id: String,
    pub name: String,
    pub pitch: f32,
    pub formant: f32,
}

static VOICE_CHANGER_PRESETS: OnceLock<Vec<VoiceChangerPreset>> = OnceLock::new();

fn voice_changer_presets() -> &'static [VoiceChangerPreset] {
    VOICE_CHANGER_PRESETS
        .get_or_init(|| {
            vec![
                VoiceChangerPreset {
                    id: "default".to_string(),
                    name: "Default".to_string(),
                    pitch: 0.0,
                    formant: 0.0,
                },
                VoiceChangerPreset {
                    id: "deeper".to_string(),
                    name: "Deeper".to_string(),
                    pitch: -4.0,
                    formant: -0.2,
                },
                VoiceChangerPreset {
                    id: "higher".to_string(),
                    name: "Higher".to_string(),
                    pitch: 5.0,
                    formant: 0.2,
                },
                VoiceChangerPreset {
                    id: "chipmunk".to_string(),
                    name: "Chipmunk".to_string(),
                    pitch: 10.0,
                    formant: 0.4,
                },
                VoiceChangerPreset {
                    id: "deep_voice".to_string(),
                    name: "Deep Voice".to_string(),
                    pitch: -8.0,
                    formant: -0.3,
                },
                VoiceChangerPreset {
                    id: "subtle_lower".to_string(),
                    name: "Subtle Lower".to_string(),
                    pitch: -2.0,
                    formant: 0.0,
                },
                VoiceChangerPreset {
                    id: "subtle_higher".to_string(),
                    name: "Subtle Higher".to_string(),
                    pitch: 2.0,
                    formant: 0.0,
                },
            ]
        })
        .as_slice()
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
}

struct OutputPlayback {
    deque: VecDeque<f32>,
    prefill_target: usize,
    max_queue_samples: usize,
    ch_out: usize,
    prefilled: bool,
    hold: Vec<f32>,
    starving: bool,
    recover_left: usize,
    truncate_silent_streak: u32,
}

fn rms_interleaved(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|x| x * x).sum();
    (sum / samples.len() as f32).sqrt()
}

impl OutputPlayback {
    fn new(prefill_target: usize, max_queue_samples: usize, ch_out: usize) -> Self {
        Self {
            deque: VecDeque::with_capacity(max_queue_samples.max(prefill_target) + 8192),
            prefill_target,
            max_queue_samples: max_queue_samples.max(prefill_target + 1),
            ch_out,
            prefilled: false,
            hold: vec![0.0f32; ch_out.max(1)],
            starving: false,
            recover_left: 0,
            truncate_silent_streak: 0,
        }
    }

    fn queue_len(&self) -> usize {
        self.deque.len()
    }

    fn push_interleaved(
        &mut self,
        samples: &[f32],
        truncate_enabled: bool,
        truncate_rms: f32,
        truncate_chunks: u32,
    ) {
        self.deque.extend(samples.iter().copied());
        let excess = self.deque.len().saturating_sub(self.max_queue_samples);
        if excess > 0 {
            self.deque.drain(0..excess);
        }

        if truncate_enabled && truncate_chunks > 0 && !samples.is_empty() {
            let rms = rms_interleaved(samples);
            if rms < truncate_rms {
                self.truncate_silent_streak = self.truncate_silent_streak.saturating_add(1);
            } else {
                self.truncate_silent_streak = 0;
            }
            if self.truncate_silent_streak >= truncate_chunks {
                let drop_n = self.deque.len().saturating_sub(self.prefill_target);
                if drop_n > 0 {
                    self.deque.drain(0..drop_n);
                }
                self.truncate_silent_streak = 0;
            }
        } else {
            self.truncate_silent_streak = 0;
        }
    }

    fn fill_callback(&mut self, data: &mut [f32]) {
        let ch_out = self.ch_out.max(1);
        debug_assert_eq!(data.len() % ch_out, 0);

        if !self.prefilled {
            if self.deque.len() >= self.prefill_target {
                self.prefilled = true;
            } else {
                data.fill(0.0);
                return;
            }
        }

        let recover_span = (RECOVER_FRAMES * ch_out).max(1);

        for i in 0..data.len() {
            let ch = i % ch_out;
            if let Some(v) = self.deque.pop_front() {
                if self.starving {
                    self.recover_left = recover_span;
                    self.starving = false;
                }

                let out_v = if self.recover_left > 0 {
                    let done = recover_span - self.recover_left;
                    let t = (done as f32 + 1.0) / recover_span as f32;
                    self.recover_left -= 1;
                    self.hold[ch].mul_add(1.0 - t, v * t)
                } else {
                    v
                };

                data[i] = out_v;
                self.hold[ch] = out_v;
            } else {
                self.starving = true;
                data[i] = self.hold[ch];
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
    stop_sender: Arc<Mutex<Option<Sender<()>>>>,
    is_running: Arc<AtomicBool>,
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
            stop_sender: Arc::new(Mutex::new(None)),
            is_running: Arc::new(AtomicBool::new(false)),
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
fn get_voice_changer_presets() -> Vec<VoiceChangerPreset> {
    voice_changer_presets().to_vec()
}

#[command]
fn get_voice_changer_params(state: State<'_, VoiceChangerState>) -> Result<VoiceChangerParams, String> {
    let h = *state.hot_params.lock().map_err(|_| "Lock failed")?;
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
    let (pf, mq) = clamp_output_buffer_ms(params.output_prefill_ms, params.output_max_queue_ms);
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
    let lp = params.latency_preset.to_lowercase();
    *state.latency_preset.lock().map_err(|_| "Lock failed")? = if lp == "stable" {
        "stable".into()
    } else if lp == "custom" {
        "custom".into()
    } else {
        "low_latency".into()
    };
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
    Ok(VoiceChangerMetrics {
        buffer_ms,
        process_ms: proc_us / 1000.0,
    })
}

#[command]
fn set_voice_changer_enabled(state: State<'_, VoiceChangerState>, enabled: bool) -> Result<(), String> {
    *state.enabled.lock().map_err(|_| "Lock failed")? = enabled;
    Ok(())
}

#[command]
fn apply_voice_changer_preset(state: State<'_, VoiceChangerState>, preset_id: String) -> Result<(), String> {
    let preset = voice_changer_presets()
        .iter()
        .find(|p| p.id == preset_id)
        .ok_or_else(|| format!("Preset '{}' not found", preset_id))?;

    {
        let mut h = state.hot_params.lock().map_err(|_| "Lock failed")?;
        h.pitch_semitones = preset.pitch;
        h.formant_shift = preset.formant;
    }
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

    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }

    let (tx, rx) = channel();
    *state.stop_sender.lock().unwrap() = Some(tx);
    state.is_running.store(true, Ordering::Release);
    *state.enabled.lock().unwrap() = true;

    let hot_params = state.hot_params.clone();
    let is_running = state.is_running.clone();
    let mode_arc = state.mode.clone();
    let sidecar_url = state.sidecar_ws_url.clone();
    let onnx_model_path = state.onnx_model_path.clone();
    let onnx_execution_provider = state.onnx_execution_provider.clone();
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
    let app_clone = app.clone();

    thread::spawn(move || {
        let result = run_voice_changer_loop(
            rx,
            hot_params,
            is_running.clone(),
            mode_arc,
            sidecar_url,
            onnx_model_path,
            onnx_execution_provider,
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
        );

        if let Err(e) = result {
            eprintln!("[VoiceChanger] Error: {}", e);
        }

        is_running.store(false, Ordering::Release);
        let _ = app_clone.emit("voice_changer:stopped", ());
    });

    let _ = app.emit("voice_changer:started", ());
    Ok(())
}

fn run_voice_changer_loop(
    rx: std::sync::mpsc::Receiver<()>,
    hot_params: Arc<Mutex<VoiceChangerHotParams>>,
    is_running: Arc<AtomicBool>,
    mode_arc: Arc<Mutex<VoiceChangerMode>>,
    sidecar_url: Arc<Mutex<String>>,
    onnx_model_path: Arc<Mutex<String>>,
    onnx_execution_provider: Arc<Mutex<String>>,
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
    let host = cpal::default_host();

    let input = if let Some(ref name) = input_device_name {
        if !name.is_empty() {
            host.input_devices()
                .map_err(|e| e.to_string())?
                .find(|d| d.name().unwrap_or_default() == *name)
                .ok_or_else(|| format!("Input device '{}' not found", name))?
        } else {
            host.default_input_device().ok_or("No default input device")?
        }
    } else {
        host.default_input_device().ok_or("No default input device")?
    };

    let output = if output_device_name.is_empty() || output_device_name == "default" {
        host.default_output_device().ok_or("No default output device")?
    } else {
        host.output_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.name().unwrap_or_default() == output_device_name)
            .ok_or_else(|| format!("Output device '{}' not found", output_device_name))?
    };

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

    let ring_cap = need
        .saturating_mul(64)
        .max(16_384)
        .min(262_144);
    let (mut audio_producer, mut audio_consumer) = RingBuffer::<f32>::new(ring_cap);
    let is_running_input = is_running.clone();

    let playback: Arc<Mutex<OutputPlayback>> = Arc::new(Mutex::new(OutputPlayback::new(
        prefill_samples,
        max_queue_samples,
        ch_out,
    )));
    let playback_clone = playback.clone();

    let input_stream = if matches!(input_config.sample_format(), cpal::SampleFormat::F32) {
        input
            .build_input_stream(
                &input_config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !is_running_input.load(Ordering::Acquire) {
                        return;
                    }
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
        input
            .build_input_stream(
                &input_config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !is_running_i16.load(Ordering::Acquire) {
                        return;
                    }
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

    let output_stream = output
        .build_output_stream(
            &output_config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut pb = playback_clone.lock().unwrap();
                pb.fill_callback(data);
            },
            |e| eprintln!("[VoiceChanger] Output error: {}", e),
            None,
        )
        .map_err(|e| e.to_string())?;

    input_stream.play().map_err(|e| e.to_string())?;
    output_stream.play().map_err(|e| e.to_string())?;

    let (_monitor_stream, monitor_playback, r_mon_out, ch_mon_out) =
        if monitor_output_device.trim().is_empty() {
            (None::<cpal::Stream>, None, 0u32, 0usize)
        } else {
            let mon_dev = if monitor_output_device == "default" {
                host.default_output_device()
                    .ok_or_else(|| "No default output device for monitor".to_string())?
            } else {
                host.output_devices()
                    .map_err(|e| e.to_string())?
                    .find(|d| d.name().unwrap_or_default() == monitor_output_device)
                    .ok_or_else(|| {
                        format!("Monitor output device '{}' not found", monitor_output_device)
                    })?
            };
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
            let mon_pb = Arc::new(Mutex::new(OutputPlayback::new(prefill_m, max_q_m, ch_mon)));
            let mon_cb = mon_pb.clone();
            let m_stream = mon_dev
                .build_output_stream(
                    &mc.into(),
                    move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                        mon_cb.lock().unwrap().fill_callback(data);
                    },
                    |e| eprintln!("[VoiceChanger] Monitor output error: {}", e),
                    None,
                )
                .map_err(|e| e.to_string())?;
            m_stream.play().map_err(|e| e.to_string())?;
            (Some(m_stream), Some(mon_pb), r_mon, ch_mon)
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
                    work_chunk.clone()
                }
            };
            VoiceChunkPipeline::process_chunk(
                &mut classic,
                &converted,
                pitch,
                formant,
                window_ms,
                oversample as u32,
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
                    work_chunk.clone()
                }
                Err(e) => {
                    eprintln!("[VoiceChanger] Sidecar: {}", e);
                    work_chunk.clone()
                }
            };
            VoiceChunkPipeline::process_chunk(
                &mut classic,
                &neural,
                pitch,
                formant,
                window_ms,
                oversample as u32,
            )?
        } else {
            VoiceChunkPipeline::process_chunk(
                &mut classic,
                &work_chunk,
                pitch,
                formant,
                window_ms,
                oversample as u32,
            )?
        };

        if let Some(ref mon_pb) = monitor_playback {
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
                    mon_pb
                        .lock()
                        .unwrap()
                        .push_interleaved(final_mon, false, 0.0, 0);
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

        {
            let mut pb = playback.lock().unwrap();
            pb.push_interleaved(&processed, te, thr, n_trunc);
            let ql = pb.queue_len().min(u32::MAX as usize) as u32;
            metrics_buffer_samples.store(ql, Ordering::Relaxed);
        }
    }

    capture_sr.store(0, Ordering::Relaxed);
    capture_ch.store(0, Ordering::Relaxed);
    output_sr.store(0, Ordering::Relaxed);
    output_ch.store(0, Ordering::Relaxed);
    metrics_process_us.store(0, Ordering::Relaxed);
    metrics_process_ema_us.store(0, Ordering::Relaxed);
    metrics_buffer_samples.store(0, Ordering::Relaxed);
    if let Ok(mut aux) = aux_feed.lock() {
        aux.clear();
    }

    #[cfg(debug_assertions)]
    println!("[VoiceChanger] Stopped audio streams");
    Ok(())
}

#[command]
fn stop_voice_changer<R: Runtime>(app: AppHandle<R>, state: State<'_, VoiceChangerState>) -> Result<(), String> {
    *state.enabled.lock().unwrap() = false;
    state.is_running.store(false, Ordering::Release);
    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }
    state.capture_sample_rate.store(0, Ordering::Relaxed);
    state.capture_channels.store(0, Ordering::Relaxed);
    state.output_sample_rate.store(0, Ordering::Relaxed);
    state.output_channels.store(0, Ordering::Relaxed);
    state.metrics_process_us.store(0, Ordering::Relaxed);
    state.metrics_process_ema_us.store(0, Ordering::Relaxed);
    state.metrics_buffer_samples.store(0, Ordering::Relaxed);
    if let Ok(mut aux) = state.aux_feed.lock() {
        aux.clear();
    }
    let _ = app.emit("voice_changer:stopped", ());
    Ok(())
}

#[command]
fn is_voice_changer_running(state: State<'_, VoiceChangerState>) -> bool {
    state.is_running.load(Ordering::Acquire)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("voice-changer")
        .setup(|app, _api| {
            app.manage(VoiceChangerState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_voice_changer_presets,
            get_voice_changer_params,
            set_voice_changer_params,
            set_voice_changer_enabled,
            apply_voice_changer_preset,
            start_voice_changer,
            stop_voice_changer,
            is_voice_changer_running,
            get_voice_changer_metrics,
            voice_changer_push_aux_pcm,
            voice_changer_push_aux_wav,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::adapt_channels_interleaved_into;

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
