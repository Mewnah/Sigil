//! Voice-changer metrics IPC (`get_voice_changer_metrics`) and the worker's percentile window.
//!
//! The P3 fields (`process_p50_ms`, `process_p95_ms`, `e2e_ms`, `vram_mb`, `governor_level`,
//! `levers`) are filled by the worker while it runs (P3-E) and are `null` while stopped or before
//! the worker measured them. `e2e_ms` is the median estimated mic-to-speaker latency of the recent
//! blocks from the per-tick timeline (capture time of a block's newest frame to the estimated
//! playout of its converted copy: waiting input, processing, the harness's algorithmic delay, a
//! denoiser's delay, the output queue and one device period).

use super::state::{lock, VoiceChangerState};
use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::vc::engine::options::EffectiveLevers;
use serde::Serialize;
use std::sync::atomic::Ordering;
use tauri::{command, State};

/// Blocks in the p99 window (about 5 s of 10 ms blocks).
const P99_WINDOW: usize = 512;
/// The p99 is recomputed every this many blocks (a copy + select of 512 values).
const P99_EVERY: usize = 16;
/// Sentinel of the P3 metrics atomics that have no value yet (`null` in the metrics).
pub(super) const METRIC_UNSET: u32 = u32::MAX;

/// Recent per-block processing times and their percentiles (p50 / p95 / p99). Worker-owned;
/// allocation-free after construction.
pub(super) struct ProcessTimeWindow {
    times_us: Vec<u32>,
    next: usize,
    filled: usize,
    since_update: usize,
    scratch: Vec<u32>,
    p50_us: u32,
    p95_us: u32,
    p99_us: u32,
}

/// Nearest-rank percentile `q` (0..=100) of `len` values: the index of the smallest value with at
/// least `q` % of the samples at or below it.
fn rank(len: usize, q: usize) -> usize {
    (len * q).div_ceil(100).max(1) - 1
}

impl ProcessTimeWindow {
    pub(super) fn new() -> Self {
        Self {
            times_us: vec![0; P99_WINDOW],
            next: 0,
            filled: 0,
            since_update: 0,
            scratch: Vec::with_capacity(P99_WINDOW),
            p50_us: 0,
            p95_us: 0,
            p99_us: 0,
        }
    }

    /// Record one block; returns the current p99 (us), refreshed every [`P99_EVERY`] blocks and
    /// on the first ones (p50 / p95 refresh with it).
    pub(super) fn push(&mut self, us: u32) -> u32 {
        self.times_us[self.next] = us;
        self.next = (self.next + 1) % P99_WINDOW;
        self.filled = (self.filled + 1).min(P99_WINDOW);
        self.since_update += 1;
        if self.since_update >= P99_EVERY || self.filled < P99_EVERY {
            self.since_update = 0;
            self.scratch.clear();
            self.scratch.extend_from_slice(&self.times_us[..self.filled]);
            let n = self.filled;
            // Select the highest rank first; the lower ranks then lie in the left part.
            let (left, v99, _) = self.scratch.select_nth_unstable(rank(n, 99));
            self.p99_us = *v99;
            let r95 = rank(n, 95);
            self.p95_us = if r95 < left.len() {
                *left.select_nth_unstable(r95).1
            } else {
                self.p99_us
            };
            let r50 = rank(n, 50);
            self.p50_us = if r50 < left.len() {
                *left.select_nth_unstable(r50).1
            } else {
                self.p99_us
            };
        }
        self.p99_us
    }

    pub(super) fn p50(&self) -> u32 {
        self.p50_us
    }

    pub(super) fn p95(&self) -> u32 {
        self.p95_us
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
    // ---- P3 (params_contract.json `metrics_p3_new_fields`; `null` until P3-E reports them) ----
    /// p50 / p95 of the per-block processing time over the recent window.
    pub process_p50_ms: Option<f64>,
    pub process_p95_ms: Option<f64>,
    /// Estimated mic-to-speaker latency from the per-tick timeline (capture end to playout).
    pub e2e_ms: Option<f64>,
    /// Process VRAM held by the voice engine (MiB; the VRAM ledger / NVML / DXGI).
    pub vram_mb: Option<f64>,
    /// Governor quality level (0 = full quality).
    pub governor_level: Option<u32>,
    /// Levers, accelerator and precision the running engine really uses.
    pub levers: Option<EffectiveLevers>,
    /// The output cushion the queue holds now in ms (P3B-H): the adaptive controller's target, else the
    /// fixed prefill (`null` while stopped or before the output runs).
    pub cushion_ms: Option<f64>,
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
    // The panel asks for the metrics when it opens, also while stopped: a good moment to warm the GPU
    // path the saved settings point at (P3B-H; once per process, idle priority, a no-op while running).
    if !state.is_running.load(Ordering::Relaxed) {
        state.prewarm("panel", std::time::Duration::ZERO);
    }
    let r = state.output_sample_rate.load(Ordering::Relaxed);
    let ch = state.output_channels.load(Ordering::Relaxed).max(1) as f64;
    let buf_n = state.metrics_buffer_samples.load(Ordering::Relaxed) as f64;
    let buffer_ms = frames_to_ms(buf_n / ch, r);
    let ema = state.metrics_process_ema_us.load(Ordering::Relaxed);
    let raw = state.metrics_process_us.load(Ordering::Relaxed);
    let proc_us = if ema > 0 { ema } else { raw } as f64;
    // The device block the run uses (a template voice's `model_divisor` block differs from the
    // setting), else the setting.
    let run_chunk = state.metrics_chunk_frames.load(Ordering::Relaxed);
    let chunk_frames = if run_chunk > 0 { f64::from(run_chunk) } else { lock(&state.hot_params).chunk_frames.max(1) as f64 };
    let capture_sr = state.capture_sample_rate.load(Ordering::Relaxed);
    let chunk_sr = if capture_sr > 0 { capture_sr } else { r };
    let chunk_ms = frames_to_ms(chunk_frames, chunk_sr);
    // The P3 fields are reported while the worker runs and has measured them (`null` otherwise).
    let running = capture_sr > 0 && state.is_running.load(Ordering::Relaxed);
    let measured_us = |a: &std::sync::atomic::AtomicU32| {
        let v = a.load(Ordering::Relaxed);
        (running && v > 0).then(|| f64::from(v) / 1000.0)
    };
    let vram = state.metrics_vram_mb.load(Ordering::Relaxed);
    let level = state.metrics_governor_level.load(Ordering::Relaxed);
    VoiceChangerMetrics {
        buffer_ms,
        process_ms: proc_us / 1000.0,
        chunk_ms,
        process_p99_ms: f64::from(state.metrics_process_p99_us.load(Ordering::Relaxed)) / 1000.0,
        backlog_ms: frames_to_ms(f64::from(state.metrics_backlog_frames.load(Ordering::Relaxed)), capture_sr),
        dropped_ms: frames_to_ms(state.metrics_dropped_frames.load(Ordering::Relaxed) as f64, capture_sr),
        underruns: state.metrics_underruns.load(Ordering::Relaxed),
        ep: lock(&state.run_ep).clone(),
        process_p50_ms: measured_us(&state.metrics_process_p50_us),
        process_p95_ms: measured_us(&state.metrics_process_p95_us),
        e2e_ms: measured_us(&state.metrics_e2e_us),
        vram_mb: (running && vram != METRIC_UNSET).then_some(f64::from(vram)),
        governor_level: (running && level != METRIC_UNSET).then_some(level),
        levers: if running { lock(&state.run_levers).clone() } else { None },
        cushion_ms: {
            let c = state.metrics_cushion_us.load(Ordering::Relaxed);
            (running && c != METRIC_UNSET).then(|| f64::from(c) / 1000.0)
        },
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

    /// p50 / p95 / p99 of one window (nearest rank).
    #[test]
    fn percentiles_of_the_window() {
        let mut w = ProcessTimeWindow::new();
        // 112 blocks (a refresh boundary): 1..=100 ms shuffled, then 12 x 50 ms.
        let mut last = 0;
        for i in 0..100u32 {
            last = w.push(((i * 37) % 100 + 1) * 1_000);
        }
        for _ in 0..12 {
            last = w.push(50_000);
        }
        assert!(last >= 98_000, "{last}");
        assert!((w.p95() as i64 - 94_000).abs() <= 3_000, "{}", w.p95());
        assert!((w.p50() as i64 - 50_000).abs() <= 3_000, "{}", w.p50());
        let mut one = ProcessTimeWindow::new();
        one.push(7);
        assert_eq!((one.p50(), one.p95()), (7, 7));
    }

    /// While running, the P3 fields report the worker's values; stopped, they are null.
    #[test]
    fn p3_metrics_are_reported_while_running() {
        let st = VoiceChangerState::new();
        st.metrics_process_p50_us.store(4_000, Ordering::Relaxed);
        st.metrics_process_p95_us.store(6_000, Ordering::Relaxed);
        st.metrics_e2e_us.store(74_000, Ordering::Relaxed);
        st.metrics_vram_mb.store(512, Ordering::Relaxed);
        st.metrics_governor_level.store(1, Ordering::Relaxed);
        let m = collect_metrics(&st);
        assert!(m.process_p50_ms.is_none() && m.e2e_ms.is_none() && m.governor_level.is_none(), "stopped");
        st.capture_sample_rate.store(48_000, Ordering::Relaxed);
        st.is_running.store(true, Ordering::Relaxed);
        let m = collect_metrics(&st);
        assert_eq!((m.process_p50_ms, m.process_p95_ms, m.e2e_ms), (Some(4.0), Some(6.0), Some(74.0)));
        assert_eq!((m.vram_mb, m.governor_level), (Some(512.0), Some(1)));
        st.metrics_vram_mb.store(METRIC_UNSET, Ordering::Relaxed);
        assert_eq!(collect_metrics(&st).vram_mb, None);
    }

    /// Every P0 metrics field of the contract is serialized under its contract name.
    #[test]
    fn metrics_serialize_the_contract_fields() {
        let c: serde_json::Value = serde_json::from_str(include_str!("params_contract.json")).unwrap();
        let st = VoiceChangerState::new();
        let m = serde_json::to_value(collect_metrics(&st)).unwrap();
        for section in ["metrics_existing_fields", "metrics_p0_new_fields", "metrics_p3_new_fields"] {
            for key in c[section].as_object().unwrap().keys() {
                assert!(m.get(key).is_some(), "metrics field {key} missing");
            }
        }
        assert!(m["ep"].is_null(), "no EP while stopped");
    }
}
