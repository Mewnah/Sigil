//! Debug snapshot IPC (`capture_voice_changer_debug`).

use super::metrics::collect_metrics;
use super::params::VoiceChangerMode;
use super::state::{lock, MmcssFlag, VoiceChangerState};
use crate::services::diagnostics::EpOutcome;
use serde::Serialize;
use std::sync::atomic::Ordering;
use tauri::{command, State};

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
    pub process_p99_ms: f64,
    pub backlog_ms: f64,
    pub dropped_ms: f64,
    pub underruns: u64,
    pub latency_budget_ms: f64,
    pub aux_mic_gate: String,
    pub worker_priority: String,
    /// Queued aux (TTS) audio in ms.
    pub aux_queued_ms: f64,
    pub ep: Option<EpOutcome>,
    /// MMCSS "Pro Audio" of the capture / output / monitor callbacks: `registered`, `failed` or
    /// `none` (no callback yet, or the stream is off).
    pub mmcss_capture: String,
    pub mmcss_output: String,
    pub mmcss_monitor: String,
}

fn mmcss(flag: &MmcssFlag) -> String {
    match flag.get() {
        None => "none",
        Some(true) => "registered",
        Some(false) => "failed",
    }
    .to_string()
}

#[command]
pub(super) fn capture_voice_changer_debug(
    state: State<'_, VoiceChangerState>,
) -> Result<VoiceChangerDebugSnapshot, String> {
    let m = collect_metrics(&state);
    let h = *lock(&state.hot_params);
    let capture_sr = state.capture_sample_rate.load(Ordering::Relaxed);
    let capture_ch = state.capture_channels.load(Ordering::Relaxed).max(1);
    let aux_queued_ms = if capture_sr > 0 {
        state.aux_feed.lock().queued_samples() as f64 / f64::from(capture_ch) / f64::from(capture_sr) * 1000.0
    } else {
        0.0
    };
    let mode_s = match *lock(&state.mode) {
        VoiceChangerMode::Classic => "classic",
        VoiceChangerMode::Onnx => "onnx",
        VoiceChangerMode::Sidecar => "sidecar",
        VoiceChangerMode::Rvc => "rvc",
    };
    let rvc = lock(&state.rvc).clone();
    Ok(VoiceChangerDebugSnapshot {
        mode: mode_s.into(),
        buffer_ms: m.buffer_ms,
        process_ms: m.process_ms,
        chunk_ms: m.chunk_ms,
        capture_sample_rate: capture_sr,
        output_sample_rate: state.output_sample_rate.load(Ordering::Relaxed),
        rvc_slot_dir: rvc.slot_dir,
        rvc_preset: rvc.preset,
        rvc_execution_provider: rvc.execution_provider,
        pitch_extractor: rvc.pitch_extractor,
        int8_encoder_supported: crate::services::neural_audio::vc::supports_int8_content_encoder(),
        process_p99_ms: m.process_p99_ms,
        backlog_ms: m.backlog_ms,
        dropped_ms: m.dropped_ms,
        underruns: m.underruns,
        latency_budget_ms: h.latency_budget_ms,
        aux_mic_gate: h.aux_mic_gate.as_str().into(),
        worker_priority: h.worker_priority.as_str().into(),
        aux_queued_ms,
        ep: m.ep,
        mmcss_capture: mmcss(&state.mmcss.capture),
        mmcss_output: mmcss(&state.mmcss.output),
        mmcss_monitor: mmcss(&state.mmcss.monitor),
    })
}
