//! Voice-changer metrics IPC (`get_voice_changer_metrics`) and the worker's p99 window.

use super::state::{lock, VoiceChangerState};
use crate::services::diagnostics::EpOutcome;
use serde::Serialize;
use std::sync::atomic::Ordering;
use tauri::{command, State};

/// Blocks in the p99 window (about 5 s of 10 ms blocks).
const P99_WINDOW: usize = 512;
/// The p99 is recomputed every this many blocks (a copy + select of 512 values).
const P99_EVERY: usize = 16;

/// Recent per-block processing times and their 99th percentile. Worker-owned; allocation-free
/// after construction.
pub(super) struct ProcessTimeWindow {
    times_us: Vec<u32>,
    next: usize,
    filled: usize,
    since_update: usize,
    scratch: Vec<u32>,
    p99_us: u32,
}

impl ProcessTimeWindow {
    pub(super) fn new() -> Self {
        Self {
            times_us: vec![0; P99_WINDOW],
            next: 0,
            filled: 0,
            since_update: 0,
            scratch: Vec::with_capacity(P99_WINDOW),
            p99_us: 0,
        }
    }

    /// Record one block; returns the current p99 (µs), refreshed every [`P99_EVERY`] blocks and
    /// on the first ones.
    pub(super) fn push(&mut self, us: u32) -> u32 {
        self.times_us[self.next] = us;
        self.next = (self.next + 1) % P99_WINDOW;
        self.filled = (self.filled + 1).min(P99_WINDOW);
        self.since_update += 1;
        if self.since_update >= P99_EVERY || self.filled < P99_EVERY {
            self.since_update = 0;
            self.scratch.clear();
            self.scratch.extend_from_slice(&self.times_us[..self.filled]);
            // Nearest-rank p99: the smallest value with at least 99 % of the samples at or below.
            let rank = (self.filled * 99).div_ceil(100).max(1) - 1;
            let (_, v, _) = self.scratch.select_nth_unstable(rank);
            self.p99_us = *v;
        }
        self.p99_us
    }
}

/// `get_voice_changer_metrics`. The P0 fields (`process_p99_ms`, `backlog_ms`, `dropped_ms`,
/// `underruns`, `ep`) are defined in `params_contract.json`.
#[derive(Serialize, Clone, Debug)]
pub struct VoiceChangerMetrics {
    /// Output queue (deque + ring) in ms.
    pub buffer_ms: f64,
    /// EMA of the per-block processing time.
    pub process_ms: f64,
    /// Block length in ms.
    pub chunk_ms: f64,
    /// p99 of the per-block processing time over the recent window.
    pub process_p99_ms: f64,
    /// Captured audio waiting to be converted (capture ring + pre-accumulator) in ms.
    pub backlog_ms: f64,
    /// Audio dropped by the latency budget (and capture-ring overflow) since start, in ms.
    pub dropped_ms: f64,
    /// Output callbacks that ran dry since start (after playback first began).
    pub underruns: u64,
    /// Execution provider of the running model (`null` for classic / sidecar, or when stopped).
    pub ep: Option<EpOutcome>,
}

/// Frames at `rate` Hz in ms (0 while not running).
pub(super) fn frames_to_ms(frames: f64, rate: u32) -> f64 {
    if rate > 0 {
        frames / f64::from(rate) * 1000.0
    } else {
        0.0
    }
}

pub(super) fn collect_metrics(state: &VoiceChangerState) -> VoiceChangerMetrics {
    let r = state.output_sample_rate.load(Ordering::Relaxed);
    let ch = state.output_channels.load(Ordering::Relaxed).max(1) as f64;
    let buf_n = state.metrics_buffer_samples.load(Ordering::Relaxed) as f64;
    let buffer_ms = frames_to_ms(buf_n / ch, r);
    let ema = state.metrics_process_ema_us.load(Ordering::Relaxed);
    let raw = state.metrics_process_us.load(Ordering::Relaxed);
    let proc_us = if ema > 0 { ema } else { raw } as f64;
    let chunk_frames = lock(&state.hot_params).chunk_frames.max(1) as f64;
    let capture_sr = state.capture_sample_rate.load(Ordering::Relaxed);
    let chunk_sr = if capture_sr > 0 { capture_sr } else { r };
    let chunk_ms = frames_to_ms(chunk_frames, chunk_sr);
    VoiceChangerMetrics {
        buffer_ms,
        process_ms: proc_us / 1000.0,
        chunk_ms,
        process_p99_ms: f64::from(state.metrics_process_p99_us.load(Ordering::Relaxed)) / 1000.0,
        backlog_ms: frames_to_ms(f64::from(state.metrics_backlog_frames.load(Ordering::Relaxed)), capture_sr),
        dropped_ms: frames_to_ms(state.metrics_dropped_frames.load(Ordering::Relaxed) as f64, capture_sr),
        underruns: state.metrics_underruns.load(Ordering::Relaxed),
        ep: lock(&state.run_ep).clone(),
    }
}

#[command]
pub(super) fn get_voice_changer_metrics(state: State<'_, VoiceChangerState>) -> Result<VoiceChangerMetrics, String> {
    Ok(collect_metrics(&state))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p99_tracks_the_slow_tail_of_the_window() {
        let mut w = ProcessTimeWindow::new();
        // 1000 blocks: 98 % at 5 ms, 2 % at 40 ms (spread out): p99 is 40 ms.
        let mut last = 0;
        for i in 0..1_000u32 {
            last = w.push(if i % 50 == 0 { 40_000 } else { 5_000 });
        }
        assert_eq!(last, 40_000);
        // A long calm stretch pushes the spikes out of the window.
        for _ in 0..P99_WINDOW {
            last = w.push(5_000);
        }
        assert_eq!(last, 5_000);
        // First block: the p99 is that block.
        let mut w = ProcessTimeWindow::new();
        assert_eq!(w.push(1_234), 1_234);
    }

    /// Every P0 metrics field of the contract is serialized under its contract name.
    #[test]
    fn metrics_serialize_the_contract_fields() {
        let c: serde_json::Value = serde_json::from_str(include_str!("params_contract.json")).unwrap();
        let st = VoiceChangerState::new();
        let m = serde_json::to_value(collect_metrics(&st)).unwrap();
        for section in ["metrics_existing_fields", "metrics_p0_new_fields"] {
            for key in c[section].as_object().unwrap().keys() {
                assert!(m.get(key).is_some(), "metrics field {key} missing");
            }
        }
        assert!(m["ep"].is_null(), "no EP while stopped");
    }
}
