//! The voice-changer worker: model load, audio streams, the capture → convert → playback loop,
//! run generations and `voice_changer:error` emission.
//!
//! Real-time structure (plan "Real-time hygiene"; P3-E moved the capture onto the hub):
//! - Models and bridges load **before** any audio flows, so the first ticks never run on a backlog
//!   captured during a multi-second model load.
//! - Capture is the capture hub's device-rate consumer (`devices::HubInput`): the hub owns the one
//!   stream per microphone (MMCSS callback threads, device-follow, restarts), STT keeps its own raw
//!   16 kHz consumer of it, and the voice changer conditions only its own copy. The worker parks
//!   on the consumer (woken per device block; no sleep-poll) and moves every read into its backlog.
//!   A microphone the hub keeps reconnecting is reported (`device_lost`, then `device_restored`),
//!   and a named one that comes back renumbered by Windows is followed (`devices::InputFollow`).
//! - The worker enforces the latency budget (oldest audio dropped with a crossfaded splice, see
//!   `budget.rs`) before taking a block, in every mode. A capture gap (device restart, mute edge)
//!   is a splice too, and resets the model's stream state; any skipped input makes the outputs trim
//!   the catch-up back to their steady level (`playback.rs`), so a stall never leaves standing
//!   latency (plan T7).
//! - VC-only conditioning, in order: input gain, the device-rate denoiser (DPDFNet at 48 kHz, or
//!   GTCRN resampled for the non-RVC modes; RVC mode runs GTCRN inside the harness instead), then
//!   the TTS aux mix. STT never sees any of it.
//! - RVC voices come from the engine factory (`factory.rs`, dispatching on `manifest.format`) as a
//!   [`VoiceSet`]: switching to another voice (a params change while running) happens without a
//!   restart, crossfaded over one block after the incoming voice is primed with the latest input
//!   (`LoadedVoice::prime_blocks`; spread over the next blocks when one tick has no room for it,
//!   [`sync_prime_blocks`]); a voice that is not loaded yet loads in the background while the
//!   current one keeps playing. The per-voice values (pitch, index, speaker, ...) belong to the
//!   selected voice: while a switch is pending the playing voice keeps its own, and a switch that
//!   fails puts the selection back on the playing voice (`voice_changer:params`), so what the UI
//!   shows is what is heard. The worker reads the selection only as a consistent view
//!   (`VoiceChangerState::try_voice_view`), never half of an update. The classic tilt applies to
//!   converted audio only while the voice's true formant shift is off
//!   (`chunk_processor::rvc_output_tilt`).
//! - The adaptive governor (the `governor` lever) steps quality down under load and back up
//!   (`engine::governor`); each tick is published on the timeline (`timeline.rs`) and the metrics
//!   carry p50 / p95 / p99, the estimated mic-to-speaker latency, VRAM and the levers in use.
//! - Output and monitor are `devices::OutputPort`s: a device that disconnects no longer stops the
//!   voice changer; the port reopens it when it returns (and follows the system default).
//! - A block whose conversion fails (ONNX error, sidecar timeout / bad reply, RVC error) is muted
//!   with click-free edges (`fallback.rs`), never replaced by the unconverted microphone.
//! - A worker that stops by itself (a saved device is gone at start, a model or the sidecar does not
//!   load, a panic) always tells the user through `voice_changer:error` before
//!   `voice_changer:stopped` ([`WorkerError`]). A missing monitor device is not fatal.

use super::aux_feed::AuxMixer;
use super::block_buffer::{self, StoredBufferProof};
use super::budget::{capture_ring_capacity, effective_latency_budget_ms, push_capture, InputBacklog};
use super::capabilities::resolve_levers;
use super::commands::PARAMS_EVENT;
use super::devices::{
    resolve_input, resolve_output, CaptureSink, DeviceRole, HubInput, InputFollow, InputTarget, OutputPort, PortEvent, PortLimits,
    PortSetup,
};
use super::factory::{crossfade_into, EngineKey, EngineRequest, HarnessSpec, LoadedVoice, SwitchPlan, VoiceSet};
use super::fallback::MuteConceal;
use super::metrics::{engine_block_ms_of, frames_to_ms, CallTimes, ProcessTimeWindow, METRIC_UNSET};
use super::params::{
    clamp_output_buffer_ms, effective_pitch_extractor, effective_rvc_ep, normalize_execution_provider, pitch_fallback_reason,
    requested_rvc_ep, resolve_rvc_paths, DeviceBlockMode, OutputCushion, RvcParamsState, VoiceChangerHotParams, VoiceChangerMode,
};
use super::resample::PolyphaseResampler;
use super::state::{lock, MmcssFlag, VoiceChangerState, VoiceSelection, VoiceTuning};
use super::timeline::{self, tick_flags, VcTick};
use crate::services::audio::hub::{self, now_qpc_ns, MuteGroup};
use crate::services::diagnostics::{ep_outcomes, EpOutcome};
use crate::services::neural_audio::chunk_processor::{apply_formant_interleaved, model_divisor_chunk, model_divisor_chunk_auto, rvc_output_tilt, FormantLpState};
use crate::services::neural_audio::denoise::{self, Denoiser, DenoiserKind};
use crate::services::neural_audio::gpu::arena::VramProbe;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::engine::governor::{self, Governor, GovernorConfig, QualityLevel, TickLoad};
use crate::services::neural_audio::vc::engine::presets;
use crate::services::neural_audio::vc::engine::{EffectiveLevers, Lever};
use crate::services::neural_audio::vc::import;
use crate::services::neural_audio::vc::manifest::SlotFormat;
use crate::services::neural_audio::vc::traits::VoiceControls;
use crate::services::neural_audio::{
    clamp_chunk_frames, ClassicPhaseVocoderProcessor, OnnxVoiceBridge, RvcPresetProfile, SidecarVoiceBridge, VoiceChunkPipeline,
};
use crate::services::resources::{self, Subsystem, VramClaim};
use crate::services::rt_thread::{self, WorkerPriority};
use serde::Serialize;
use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, Runtime};

/// Throttle repeated `voice_changer:error` emissions while conversion keeps failing each chunk.
const VC_ERROR_EMIT_THROTTLE: Duration = Duration::from_secs(2);
/// Longest park without capture (stop requests also unpark the worker at once).
const PARK_TIMEOUT: Duration = Duration::from_millis(100);
/// When the MMCSS status of the output callbacks is logged (once per run).
const MMCSS_LOG_AFTER: Duration = Duration::from_secs(2);
/// Budget drops are summarized in the log at most this often.
const DROP_LOG_EVERY: Duration = Duration::from_secs(5);
/// A microphone the hub has been reconnecting for this long is reported as disconnected (a quick
/// stream restart is not worth a notice).
const INPUT_LOST_NOTICE_AFTER: Duration = Duration::from_secs(1);
/// Input history kept for priming a voice that is switched to (ms).
const SWITCH_HISTORY_MS: f64 = 500.0;


#[derive(Clone, Serialize)]
struct VoiceChangerErrorPayload {
    code: String,
    message: String,
}

/// Emit one `voice_changer:error` event. Codes while running: `onnx_infer`, `sidecar_length`,
/// `sidecar_exchange`, `rvc_convert` (the block was muted), `rvc_warning`, `rvc_load` (a voice
/// switch failed; the current voice keeps playing and the selection goes back to it), `denoiser_warning` (the denoiser did not load or
/// failed; the run goes on without it), `monitor_unavailable` (the run continues without the
/// monitor), `device_lost` / `device_restored` (the input, output or monitor device disconnected /
/// came back; the run continues). Codes of a worker that stopped ([`WorkerError`]): `device`,
/// `sidecar_connect`, `onnx_load`, `rvc_load`, `worker_error`, `worker_panic`.
pub(super) fn emit_vc_error<R: Runtime>(app: &AppHandle<R>, code: &'static str, message: impl Into<String>) {
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
    let message = message.into();
    tracing::warn!("voice changer {code}: {message}");
    emit_vc_error(app, code, message);
}

/// Why a worker stopped by itself: the `voice_changer:error` code and the message the epilogue in
/// `start_voice_changer` emits (only while the run is still current). A plain `String` error (a
/// `?` on any helper) becomes `worker_error`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct WorkerError {
    pub(super) code: &'static str,
    pub(super) message: String,
}

impl WorkerError {
    pub(super) fn new(code: &'static str, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    /// A device problem (missing, unusable format, stream failure, disconnected).
    fn device(message: impl Into<String>) -> Self {
        Self::new("device", message)
    }
}

impl From<String> for WorkerError {
    fn from(message: String) -> Self {
        Self::new("worker_error", message)
    }
}

impl From<&str> for WorkerError {
    fn from(message: &str) -> Self {
        Self::new("worker_error", message)
    }
}

impl std::fmt::Display for WorkerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({})", self.message, self.code)
    }
}

/// A worker's run generation: it is current while no later `start_voice_changer` has claimed a
/// newer one. Only the current worker may change shared state or show the user anything.
#[derive(Clone)]
pub(super) struct RunGeneration {
    pub(super) counter: Arc<AtomicU64>,
    pub(super) mine: u64,
}

impl RunGeneration {
    pub(super) fn is_current(&self) -> bool {
        self.counter.load(Ordering::SeqCst) == self.mine
    }
}

/// Message of a caught worker panic (`&str` / `String` payloads; anything else is opaque).
pub(super) fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

/// Stop requests: the stop message (remembered once read), the running flag, or a newer run.
struct StopSignal {
    rx: Receiver<()>,
    is_running: Arc<std::sync::atomic::AtomicBool>,
    generation: RunGeneration,
    stopped: bool,
}

impl StopSignal {
    fn requested(&mut self) -> bool {
        if !self.stopped && self.rx.try_recv().is_ok() {
            self.stopped = true;
        }
        self.stopped || !self.is_running.load(Ordering::Acquire) || !self.generation.is_current()
    }
}

/// Sets `flag` once the run is stopped (`is_running` cleared by Stop) or superseded (a newer
/// generation), checking every [`LOAD_WATCH_PERIOD`] on a helper thread until dropped. The worker's
/// first voice load runs before the loop reads its stop message; it hands this flag to the voice
/// (`EngineRequest::cancel`, `LoadedVoice::load_cancellable`), so Stop during a minutes-long
/// sidecar start takes effect at once.
struct LoadStopWatch {
    done: Arc<AtomicBool>,
    thread: Option<std::thread::Thread>,
}

const LOAD_WATCH_PERIOD: Duration = Duration::from_millis(100);

impl LoadStopWatch {
    fn start(is_running: Arc<AtomicBool>, generation: RunGeneration, flag: Arc<AtomicBool>) -> LoadStopWatch {
        let done = Arc::new(AtomicBool::new(false));
        let watching = done.clone();
        let thread = std::thread::Builder::new()
            .name("sigil-vc-load-watch".into())
            .spawn(move || {
                while !watching.load(Ordering::Acquire) {
                    if !is_running.load(Ordering::Acquire) || !generation.is_current() {
                        // Not once the load has ended (a Stop right after it is the loop's).
                        if !watching.load(Ordering::Acquire) {
                            flag.store(true, Ordering::Relaxed);
                        }
                        return;
                    }
                    std::thread::park_timeout(LOAD_WATCH_PERIOD);
                }
            })
            .ok()
            .map(|h| h.thread().clone());
        LoadStopWatch { done, thread }
    }
}

impl Drop for LoadStopWatch {
    fn drop(&mut self) {
        self.done.store(true, Ordering::Release);
        if let Some(t) = &self.thread {
            t.unpark();
        }
    }
}

fn unix_ms_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis().min(u128::from(u64::MAX)) as u64)
        .unwrap_or(0)
}

fn component_label(component: &str) -> &str {
    match component {
        "vc.content" => "content encoder",
        "vc.pitch" | "vc.pitch.voicing" => "F0 model",
        "vc.onnx" => "ONNX model",
        "vc.enc" => "text encoder",
        "vc.flowdec" => "decoder",
        "vc.knn" => "retrieval",
        "vc.denoise" => "denoiser",
        other => other,
    }
}

/// Session components of an RVC run, main generator first (legacy `vc`, template `vc.flowdec`).
const RVC_COMPONENTS: [&str; 8] = ["vc", "vc.flowdec", "vc.enc", "vc.content", "vc.pitch", "vc.pitch.voicing", "vc.knn", "vc.denoise"];

/// The EP this run's model actually uses, from the outcomes the session builder recorded since
/// `since_ms` (`components[0]` is the main one). A component that fell back wins (the user must
/// see it); a CPU request is trivially granted; otherwise unknown (`None`).
fn run_ep_outcome(components: &[&str], requested: &str, since_ms: u64) -> Option<EpOutcome> {
    let fresh: Vec<_> = ep_outcomes()
        .into_iter()
        .filter(|e| e.at_unix_ms >= since_ms && components.contains(&e.component.as_str()))
        .collect();
    if let Some(e) = fresh.iter().find(|e| e.outcome.fell_back()) {
        let mut o = e.outcome.clone();
        if e.component != components[0] {
            let why = o.reason.take().unwrap_or_else(|| "fell back".to_string());
            o.reason = Some(format!("{}: {why}", component_label(&e.component)));
        }
        return Some(o);
    }
    for c in components {
        if let Some(e) = fresh.iter().find(|e| e.component == *c) {
            return Some(e.outcome.clone());
        }
    }
    (requested == "cpu").then(|| EpOutcome::granted("cpu"))
}

/// The run EP of a voice from its own outcome list (a fallback wins, then the generator's).
fn voice_ep_outcome(outcomes: &[(String, EpOutcome)]) -> Option<EpOutcome> {
    if let Some((c, o)) = outcomes.iter().find(|(_, o)| o.fell_back()) {
        let mut o = o.clone();
        if c != "vc.flowdec" && c != "vc" {
            let why = o.reason.take().unwrap_or_else(|| "fell back".to_string());
            o.reason = Some(format!("{}: {why}", component_label(c)));
        }
        return Some(o);
    }
    RVC_COMPONENTS
        .iter()
        .find_map(|want| outcomes.iter().find(|(c, _)| c == want).map(|(_, o)| o.clone()))
}

fn apply_worker_priority(priority: WorkerPriority, gpu_ep: bool) {
    let ok = rt_thread::set_worker_priority(priority, gpu_ep);
    tracing::info!(
        "voice changer worker priority: {} on a {} execution provider{}",
        priority.as_str(),
        if gpu_ep { "GPU" } else { "CPU" },
        if ok { "" } else { " (not applied: the OS refused)" }
    );
}

fn mmcss_label(flag: &MmcssFlag) -> &'static str {
    match flag.get() {
        None => "no callback yet",
        Some(true) => "registered",
        Some(false) => "failed",
    }
}

fn warn_monitor_unavailable<R: Runtime>(app: &AppHandle<R>, generation: &RunGeneration, why: &str) {
    let message = format!("{}. The voice changer runs without the monitor.", why.trim_end_matches('.'));
    tracing::warn!("voice changer: {message}");
    if generation.is_current() {
        emit_vc_error(app, "monitor_unavailable", message);
    }
}

// ---------------------------------------------------------------------------------------------
// Input side: the capture ring and the latency budget
// ---------------------------------------------------------------------------------------------

/// The worker's input side: what the hub consumer reads goes through a local lock-free ring into
/// the latency-budget backlog (`budget.rs`, the proven accounting the benchmark also runs).
pub(super) struct CaptureBuffer {
    producer: rtrb::Producer<f32>,
    consumer: rtrb::Consumer<f32>,
    overflow: AtomicU64,
    backlog: InputBacklog,
    ch: usize,
    block: usize,
    /// Frames dropped by the budget (or refused by a full ring) since the last [`Self::wake`].
    dropped_since: u64,
    /// Gaps noted since the last [`Self::take_gaps`].
    gaps_since: u32,
}

impl CaptureBuffer {
    pub(super) fn new(rate: u32, ch: usize, block: usize, budget_ms: f64) -> CaptureBuffer {
        let ch = ch.max(1);
        let cap = capture_ring_capacity(rate, ch, block);
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(cap);
        CaptureBuffer {
            producer,
            consumer,
            overflow: AtomicU64::new(0),
            backlog: InputBacklog::new(ch, rate, budget_ms, cap + cap / 2 + block * ch),
            ch,
            block: block.max(1),
            dropped_since: 0,
            gaps_since: 0,
        }
    }

    pub(super) fn set_budget_ms(&mut self, ms: f64) {
        self.backlog.set_budget_ms(ms);
    }

    /// Everything received, then the budget. Returns the frames dropped at this wake.
    pub(super) fn wake(&mut self) -> u64 {
        self.backlog.ingest(&mut self.consumer, &self.overflow);
        if let Some(d) = self.backlog.enforce(self.block) {
            self.dropped_since += d.frames as u64 + d.overflow_frames;
        }
        std::mem::take(&mut self.dropped_since)
    }

    /// Gaps noted since the last call.
    pub(super) fn take_gaps(&mut self) -> u32 {
        std::mem::take(&mut self.gaps_since)
    }

    pub(super) fn frames_waiting(&self) -> usize {
        self.backlog.frames() + self.consumer.slots() / self.ch
    }

    pub(super) fn dropped_total(&self) -> u64 {
        self.backlog.dropped_frames()
    }

    pub(super) fn drop_events(&self) -> u64 {
        self.backlog.drop_events()
    }

    pub(super) fn take_block(&mut self, out: &mut Vec<f32>, frames: usize) -> bool {
        self.backlog.take_block(out, frames)
    }
}

impl CaptureSink for CaptureBuffer {
    fn audio(&mut self, frames: &[f32]) {
        let mut data = frames;
        if self.producer.slots() < data.len() {
            // A catch-up read larger than the ring's free room (a long stall on a large device
            // format): move what is queued into the backlog first, whose budget drops the oldest
            // audio. A read longer than the whole ring keeps only its newest part (the older part
            // is a hole right after the backlog). The newest audio is never the part that is lost.
            self.backlog.ingest(&mut self.consumer, &self.overflow);
            let room = self.producer.slots() / self.ch;
            let frames_in = data.len() / self.ch;
            if frames_in > room {
                let skip = frames_in - room;
                self.backlog.note_overflow(skip as u64);
                data = &data[skip * self.ch..];
            }
            if let Some(d) = self.backlog.enforce(self.block) {
                self.dropped_since += d.frames as u64 + d.overflow_frames;
            }
        }
        let lost = push_capture(&mut self.producer, data, self.ch);
        if lost > 0 {
            self.overflow.fetch_add(lost as u64, Ordering::Relaxed);
        }
    }

    /// A discontinuity right after what was pushed: the backlog fades out into it and fades the
    /// next audio in (`InputBacklog`'s hole, counted as at least one lost frame).
    fn gap(&mut self, lost_frames: u64) {
        self.overflow.fetch_add(lost_frames.max(1), Ordering::Relaxed);
        self.backlog.ingest(&mut self.consumer, &self.overflow);
        // Only a budget drop counts for the governor; the frames lost at the gap are the device's.
        if let Some(d) = self.backlog.enforce(self.block) {
            self.dropped_since += d.frames as u64;
        }
        self.gaps_since += 1;
    }
}

// ---------------------------------------------------------------------------------------------
// VC-only conditioning: the device-rate denoiser stage
// ---------------------------------------------------------------------------------------------

/// Where the denoise stage is in its bypass transitions (the governor's light-denoiser step).
#[derive(Clone, Copy, Debug, PartialEq)]
enum StageMode {
    Wet,
    Dry,
    /// Denoiser warming up after a bypass (output dry) for this many more frames.
    Warming(usize),
    /// Crossfading dry -> wet / wet -> dry; frames done.
    ToWet(usize),
    ToDry(usize),
}

/// The denoiser on the capture's device-rate copy: mono, resampled to the model's rate when the
/// device runs at another one, a constant delay (the denoiser's plus the resamplers'), written back
/// to every channel. A bypassed stage (governor) plays the input through a matching delay line, so
/// its timing never jumps; switching crossfades over [`STAGE_FADE_MS`].
pub(super) struct DenoiseStage {
    den: Box<dyn Denoiser>,
    kind: DenoiserKind,
    ch: usize,
    rate: u32,
    den_rate: u32,
    /// Zeros the wet queue starts with (the resamplers' count deficit).
    prime: usize,
    down: Option<PolyphaseResampler>,
    up: Option<PolyphaseResampler>,
    mono: Vec<f32>,
    at_rate: Vec<f32>,
    back: Vec<f32>,
    wet_fifo: VecDeque<f32>,
    dry: VecDeque<f32>,
    wet: Vec<f32>,
    latency: usize,
    mode: StageMode,
    fade: usize,
    failures: u64,
    last_error: Option<String>,
}

const STAGE_FADE_MS: f64 = 10.0;

impl DenoiseStage {
    pub(super) fn new(den: Box<dyn Denoiser>, kind: DenoiserKind, rate: u32, ch: usize, block: usize) -> DenoiseStage {
        let den_rate = den.sample_rate();
        let resample = den_rate != rate;
        // Count deficit of the two resamplers (their look-ahead in output frames, plus rounding).
        let prime = if resample {
            32 + (32 * u64::from(rate)).div_ceil(u64::from(den_rate.max(1))) as usize + 4
        } else {
            0
        };
        let den_latency = (den.latency_samples() as u64 * u64::from(rate)).div_ceil(u64::from(den_rate.max(1))) as usize;
        let latency = prime + den_latency;
        let mut wet_fifo = VecDeque::with_capacity(prime + block * 4 + 4096);
        wet_fifo.extend(std::iter::repeat_n(0.0, prime));
        DenoiseStage {
            den,
            kind,
            ch: ch.max(1),
            rate,
            den_rate,
            prime,
            down: resample.then(|| PolyphaseResampler::new(rate, den_rate, 1)),
            up: resample.then(|| PolyphaseResampler::new(den_rate, rate, 1)),
            mono: Vec::with_capacity(block),
            at_rate: Vec::with_capacity(block * 4),
            back: Vec::with_capacity(block * 4),
            wet_fifo,
            dry: std::iter::repeat_n(0.0, latency).collect(),
            wet: Vec::with_capacity(block),
            latency,
            mode: StageMode::Wet,
            fade: ((STAGE_FADE_MS * f64::from(rate)) / 1000.0).round().max(1.0) as usize,
            failures: 0,
            last_error: None,
        }
    }

    pub(super) fn kind(&self) -> DenoiserKind {
        self.kind
    }

    /// Constant delay of the stage (device frames).
    pub(super) fn latency_frames(&self) -> usize {
        self.latency
    }

    /// Model failures (their frames passed unprocessed) and the last error, since the last call.
    pub(super) fn take_failure(&mut self) -> Option<String> {
        (self.failures > 0).then(|| {
            self.failures = 0;
            self.last_error.take().unwrap_or_default()
        })
    }

    /// Governor: bypass (`true`) or run the denoiser. Returns whether that changes anything.
    pub(super) fn set_light(&mut self, light: bool) -> bool {
        let running = matches!(self.mode, StageMode::Wet | StageMode::ToWet(_) | StageMode::Warming(_));
        match (light, running) {
            (true, true) => {
                self.mode = StageMode::ToDry(0);
                true
            }
            (false, false) => {
                // The wet path restarts from silence (fresh denoiser state, resampler history and
                // queue) and warms up for its latency while the dry path still plays.
                self.den.reset();
                if self.down.is_some() {
                    self.down = Some(PolyphaseResampler::new(self.rate, self.den_rate, 1));
                    self.up = Some(PolyphaseResampler::new(self.den_rate, self.rate, 1));
                }
                self.wet_fifo.clear();
                self.wet_fifo.extend(std::iter::repeat_n(0.0, self.prime));
                self.mode = StageMode::Warming(self.latency + self.fade);
                true
            }
            _ => false,
        }
    }

    /// Capture gap / restart: the denoiser's recurrent state is stale.
    pub(super) fn reset(&mut self) {
        self.den.reset();
    }

    fn run_wet(&mut self, frames: usize) {
        self.at_rate.clear();
        match self.down.as_mut() {
            Some(d) => d.process(&self.mono, &mut self.at_rate),
            None => self.at_rate.extend_from_slice(&self.mono),
        }
        if let Err(e) = self.den.process(&mut self.at_rate) {
            self.failures += 1;
            self.last_error = Some(e);
        }
        match self.up.as_mut() {
            Some(u) => {
                self.back.clear();
                u.process(&self.at_rate, &mut self.back);
                self.wet_fifo.extend(self.back.iter().copied());
            }
            None => self.wet_fifo.extend(self.at_rate.iter().copied()),
        }
        self.wet.clear();
        for _ in 0..frames {
            self.wet.push(self.wet_fifo.pop_front().unwrap_or(0.0));
        }
    }

    /// Denoise one block in place (interleaved, the capture format).
    pub(super) fn process(&mut self, block: &mut [f32]) {
        let ch = self.ch;
        let frames = block.len() / ch;
        self.mono.clear();
        self.mono
            .extend(block.chunks_exact(ch).map(|f| f.iter().sum::<f32>() / ch as f32));
        let needs_wet = !matches!(self.mode, StageMode::Dry);
        if needs_wet {
            self.run_wet(frames);
        }
        let fade = self.fade;
        for (f, frame) in block.chunks_exact_mut(ch).enumerate() {
            self.dry.push_back(self.mono[f]);
            let dry = self.dry.pop_front().unwrap_or(0.0);
            let wet = if needs_wet { self.wet[f] } else { 0.0 };
            let v = match self.mode {
                StageMode::Wet => wet,
                StageMode::Dry => dry,
                StageMode::Warming(left) => {
                    self.mode = if left <= 1 { StageMode::ToWet(0) } else { StageMode::Warming(left - 1) };
                    dry
                }
                StageMode::ToWet(done) => {
                    let t = (done as f32 + 0.5) / fade as f32;
                    self.mode = if done + 1 >= fade { StageMode::Wet } else { StageMode::ToWet(done + 1) };
                    wet * t + dry * (1.0 - t)
                }
                StageMode::ToDry(done) => {
                    let t = (done as f32 + 0.5) / fade as f32;
                    self.mode = if done + 1 >= fade { StageMode::Dry } else { StageMode::ToDry(done + 1) };
                    dry * t + wet * (1.0 - t)
                }
            };
            frame.iter_mut().for_each(|s| *s = v);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// RVC: voices, switching, the algorithmic latency
// ---------------------------------------------------------------------------------------------

/// What a run needs to build any voice's request (fixed for the run).
#[derive(Clone)]
struct RvcBase {
    app_data: PathBuf,
    templates_dir: Option<PathBuf>,
    ep: String,
    session: VcSessionOptions,
    preset: RvcPresetProfile,
    chunk_frames: u32,
    device_rate: u32,
    in_harness_denoiser: DenoiserKind,
    /// The latency budget when the run started (ms; P5-S0: a family's default stall timeout, e.g.
    /// a sidecar's; a later change applies at the next voice load).
    latency_budget_ms: f64,
}

/// The voice a selection names (the key a switch compares), with its effective F0 method.
fn selection_key(sel: &VoiceSelection, base: &RvcBase) -> EngineKey {
    let (pitch, _) = effective_pitch_extractor(&sel.pitch_extractor, &base.ep);
    EngineKey {
        slot_dir: PathBuf::from(&sel.slot_dir),
        pitch_extractor: pitch,
        f0_range: sel.f0_range,
        denoiser: base.in_harness_denoiser,
        reference_voice: sel.reference_voice.clone(),
    }
}

/// The voice the params currently select.
fn desired_key(s: &RvcParamsState, base: &RvcBase) -> EngineKey {
    selection_key(&VoiceSelection::of(s), base)
}

/// Point empty FCPE / RMVPE paths at the model store's pitch models when installed (the legacy
/// `base/` preflight only looks in `app_data/voice-changer/base`; the pitch bank reads the store
/// first), so a voice whose pitch model came from Models & Runtimes starts.
pub(super) fn fill_store_pitch_paths(s: &mut RvcParamsState, app_data: &std::path::Path) {
    use crate::services::neural_audio::vc::{assets, f0_fusion};
    for (method, path) in [("fcpe", &mut s.fcpe_path), ("rmvpe", &mut s.rmvpe_path)] {
        if path.trim().is_empty() {
            let store = f0_fusion::pitch_model_id(method).and_then(|id| assets::model_file(id, None));
            if let Some(p) = store.or_else(|| f0_fusion::resolve_pitch_model(method, Some(app_data))) {
                *path = p.to_string_lossy().into_owned();
            }
        }
    }
}

/// Build the factory request for the voice `settings` selects (resolving its base models like the
/// start preflight does).
fn voice_request(mut s: RvcParamsState, base: &RvcBase, levers: &EffectiveLevers) -> Result<EngineRequest, String> {
    if !s.slot_dir.trim().is_empty() {
        // An unreadable or unsupported manifest is refused with its own message.
        super::factory::slot_format(std::path::Path::new(s.slot_dir.trim()))?;
    }
    fill_store_pitch_paths(&mut s, &base.app_data);
    resolve_rvc_paths(&mut s, &base.app_data)?;
    let (pitch, _) = effective_pitch_extractor(&s.pitch_extractor, &base.ep);
    Ok(EngineRequest {
        slot_dir: PathBuf::from(s.slot_dir.trim()),
        app_data: Some(base.app_data.clone()),
        templates_dir: base.templates_dir.clone(),
        hubert_path: s.hubert_path,
        fcpe_path: s.fcpe_path,
        rmvpe_path: s.rmvpe_path,
        pitch_extractor: pitch,
        ep: base.ep.clone(),
        session: base.session.clone(),
        levers: levers.clone(),
        anchored_noise_mode: s.levers.get(Lever::AnchoredNoise),
        silence_skip: s.silence_skip,
        preset: base.preset,
        precision: s.precision,
        block_ms: s.block_ms,
        chunk_frames: base.chunk_frames,
        device_rate: base.device_rate,
        extra_context_ms: s.extra_context_ms,
        lookahead_ms: s.lookahead_ms,
        f0_range: s.f0_range,
        speaker_id: s.speaker_id,
        reference_voice: s.reference_voice_id.clone(),
        reference_clip: None,
        latency_budget_ms: base.latency_budget_ms,
        // A background load sets its own flag (`VoiceSet`); the first load sets the run's.
        cancel: None,
        // P3B-S0 (no-op until P3B-D).
        content_context_ms: s.content_context_ms,
    })
}

/// Measure the harness's algorithmic latency for this stream (an identity model through the same
/// rates, blocks, fixed model block, context and crossfade: the benchmark's probe) on a helper
/// thread. The result lands in `out` as `(generation << 32) | us`, only while `out` still carries
/// `generation` (the probes of quick consecutive switches can finish out of order: an older voice's
/// latency must never overwrite the newer one's).
fn probe_algorithmic_latency(
    spec: HarnessSpec,
    model_out_sr: u32,
    min_context: usize,
    fixed_block: Option<usize>,
    out: Arc<AtomicU64>,
    generation: u64,
) {
    let _ = std::thread::Builder::new().name("sigil-vc-latency-probe".into()).spawn(move || {
        let stream = serde_json::json!({
            "input_rate": spec.r_in,
            "output_rate": spec.r_out,
            "input_channels": spec.ch_in,
            "output_channels": spec.ch_out,
            "block_frames": spec.chunk_frames,
            "extra_context_ms": spec.extra_context_ms,
            "crossfade_ms": spec.crossfade_ms,
        });
        let Ok(stream) = serde_json::from_value::<crate::services::bench::spec::StreamSpec>(stream) else {
            return;
        };
        match crate::services::bench::engine::probe_algorithmic_latency(&stream, model_out_sr, min_context, fixed_block, 0.0) {
            Ok(lat) => {
                let us = (lat.ms * 1000.0).round().clamp(1.0, f64::from(u32::MAX - 1)) as u64;
                store_probe(&out, generation, us);
            }
            Err(e) => tracing::debug!("voice changer: algorithmic latency probe failed: {e}"),
        }
    });
}

/// Store a latency probe's result (us) unless a newer probe superseded `generation`.
fn store_probe(out: &AtomicU64, generation: u64, us: u64) -> bool {
    out.try_update(Ordering::Relaxed, Ordering::Relaxed, |cur| {
        (cur >> 32 == generation).then_some((generation << 32) | (us & 0xffff_ffff))
    })
    .is_ok()
}

/// History blocks (device blocks of `block_ms`) a voice switch primes on its first tick, of `need`.
/// A model call comes every `blocks_per_call` device blocks (1 for a voice that converts every
/// block; a fixed-block voice converts once per model block), so the pipeline absorbs one call per
/// such period. All of them when the model calls they trigger, the switch block's and the outgoing
/// voice's crossfade call fit in three quarters of that period (the switch then happens at once,
/// plan T9); otherwise what fits beside the switch's own two calls, and the rest primes on the next
/// live blocks (one extra call per period instead of one tick several periods long, which would
/// starve the output on a slow execution provider). `call_ms` is the recent per-call cost (0 =
/// unknown: all at once).
fn sync_prime_blocks(need: usize, call_ms: f64, block_ms: f64, blocks_per_call: f64) -> usize {
    if !(call_ms > 0.0 && block_ms > 0.0) {
        return need;
    }
    let per_call = if blocks_per_call.is_finite() { blocks_per_call.max(1.0) } else { 1.0 };
    let fits = (0.75 * block_ms * per_call / call_ms).floor().min(1e6);
    let calls = |blocks: usize| (blocks as f64 / per_call).ceil();
    if calls(need) + 2.0 <= fits {
        need
    } else {
        ((fits - 2.0).max(0.0) * per_call).floor().min(need as f64) as usize
    }
}

/// A voice being primed before it takes over.
struct Priming {
    key: EngineKey,
    /// Device blocks it was fed, and needs.
    fed: usize,
    need: usize,
}

/// RVC mode's running state.
struct RvcRun {
    voices: VoiceSet,
    base: RvcBase,
    levers: EffectiveLevers,
    spec: HarnessSpec,
    /// The voice fading out during a switch block.
    outgoing: Option<LoadedVoice>,
    /// Recent conditioned input (interleaved capture format), for priming a switched-to voice.
    history: VecDeque<f32>,
    history_cap: usize,
    /// Interleaved samples of one device block.
    need: usize,
    /// The selection the voice set last acted on (switches happen when it changes).
    selection: VoiceSelection,
    /// The selection the active voice was chosen under (a failed switch goes back to it).
    active_selection: VoiceSelection,
    /// The newest consistent per-voice values of the selection.
    live: VoiceTuning,
    /// The per-voice values the active voice plays with: `live` while the selection is its slot, its
    /// own while a switch to another voice is pending (the incoming voice's pitch, speaker or index
    /// never reach the voice that still plays, also when that switch fails).
    tuning: VoiceTuning,
    /// Voice controls last given to the active voice (`None`: give them at the next tick).
    applied_controls: Option<VoiceControls>,
    /// Switch priming of the ready voice.
    priming: Option<Priming>,
    /// Recent per-call model cost (ms; a slowly decaying peak; 0 = unknown), for spreading the
    /// switch priming.
    call_ms: f64,
    formant_states: Vec<FormantLpState>,
    /// The active voice's algorithmic latency: `(probe generation << 32) | us` (us 0 until measured).
    algo_latency: Arc<AtomicU64>,
    probe_generation: u64,
    /// Model input carried silence since the last model reset (restored output etc.).
    needs_reset: bool,
}

impl RvcRun {
    /// Latency on top of the harness's probed algorithmic delay: the in-harness denoiser and the
    /// model's own lookahead.
    fn active_latency_extra_ns(&self) -> i64 {
        let v = self.voices.active();
        ((v.denoiser_latency_16k + v.model_delay_16k) as i64) * 1_000_000_000 / 16_000
    }

    fn start_probe(&mut self) {
        self.probe_generation = (self.probe_generation + 1) & 0xffff_ffff;
        self.algo_latency.store(self.probe_generation << 32, Ordering::Relaxed);
        let v = self.voices.active();
        probe_algorithmic_latency(
            self.spec,
            v.output_sample_rate,
            crate::services::neural_audio::vc::rvc::MIN_CONTEXT_16K,
            v.harness.fixed_shape().map(|(block, _)| block),
            self.algo_latency.clone(),
            self.probe_generation,
        );
    }

    /// The active voice's algorithmic latency (us; 0 until its probe finished).
    fn algo_latency_us(&self) -> u32 {
        (self.algo_latency.load(Ordering::Relaxed) & 0xffff_ffff) as u32
    }

    fn push_history(&mut self, block: &[f32]) {
        self.history.extend(block.iter().copied());
        let over = self.history.len().saturating_sub(self.history_cap);
        if over > 0 {
            self.history.drain(..over);
        }
    }

    /// Note one model call's cost (ms).
    fn note_call_ms(&mut self, ms: f64) {
        if ms.is_finite() {
            self.call_ms = ms.max(self.call_ms * 0.98);
        }
    }

    /// Whether the selection is the active voice's slot (its per-voice values are the live ones).
    fn selects_active(&self) -> bool {
        Path::new(&self.selection.slot_dir) == self.voices.active().key.slot_dir.as_path()
    }

    /// A consistent view of the params: the newest per-voice values, and the RVC settings when the
    /// selection changed (a voice switch: it starts here).
    fn on_view(&mut self, live: VoiceTuning, changed: Option<RvcParamsState>) {
        self.live = live;
        let Some(settings) = changed else { return };
        self.selection = VoiceSelection::of(&settings);
        // Whatever is ready next primes from scratch (a voice that went idle in between missed
        // blocks; its context would not continue).
        self.priming = None;
        let want = desired_key(&settings, &self.base);
        if want == self.voices.active().key {
            // Back to the playing voice: a load in flight is abandoned.
            self.voices.abandon();
            self.active_selection = self.selection.clone();
            return;
        }
        // The request (manifest, base models) is built on the loader thread: no file I/O here.
        let (base, levers) = (self.base.clone(), self.levers.clone());
        if self.voices.request(want.clone(), self.spec, move || voice_request(settings, &base, &levers)) == SwitchPlan::Loading {
            tracing::info!("voice changer: loading {} in the background", want.slot_dir.display());
        }
    }

    /// A finished background load. A failure returns its message and, when it is the current
    /// selection's, that selection (it should go back to the playing voice).
    fn on_poll(&mut self) -> Option<(String, Option<VoiceSelection>)> {
        match self.voices.poll()? {
            Ok((key, took)) => {
                tracing::info!("voice changer: {} loaded in {:.0} ms", key.slot_dir.display(), took.as_secs_f64() * 1000.0);
                None
            }
            Err((key, e)) => {
                let msg = format!("Could not load the voice {}: {e}. The current voice keeps playing.", key.slot_dir.display());
                let failed = (selection_key(&self.selection, &self.base) == key).then(|| self.selection.clone());
                Some((msg, failed))
            }
        }
    }

    /// The active voice's controls, every tick (cheap; the voice controls only when they change): the
    /// live per-voice values while the selection is its slot, else its own.
    fn apply_tuning(&mut self, h: &VoiceChangerHotParams) {
        if self.selects_active() {
            self.tuning = self.live;
        }
        let t = self.tuning;
        let controls = t.controls(h);
        let a = self.voices.active_mut();
        a.harness.set_controls(t.pitch_semitones, t.index_rate, t.protect, t.speaker_id);
        if self.applied_controls != Some(controls) {
            a.harness.set_voice_controls(&controls);
            self.applied_controls = Some(controls);
        }
    }

    /// Prime the ready voice (if any) for this block: on the first block, with its own controls, the
    /// governor's level, a fresh stream and as much recent input as the tick has room for
    /// ([`sync_prime_blocks`]); then with each live block until it has had enough. Returns whether it
    /// takes over at this block (it then converts `block` itself; the outputs of priming are
    /// discarded).
    fn prime_ready(&mut self, block: &[f32], level: &QualityLevel, h: &VoiceChangerHotParams, block_ms: f64) -> bool {
        let need_samples = self.need.max(1);
        let spec = self.spec;
        let Some(ready) = self.voices.ready_mut() else {
            self.priming = None;
            return false;
        };
        // The selection is the ready voice's slot: its per-voice values are the live ones.
        let t = self.live;
        ready.harness.set_controls(t.pitch_semitones, t.index_rate, t.protect, t.speaker_id);
        if self.priming.as_ref().is_none_or(|p| p.key != ready.key) {
            ready.harness.set_voice_controls(&t.controls(h));
            ready.harness.reset_model_stream();
            // Every level, full quality included: an idle voice keeps the level it had when it was
            // retired.
            ready.harness.apply_quality(level);
            let need = ready.prime_blocks(&spec);
            // Device blocks per model call of the incoming voice (a fixed-block voice calls its model
            // once per model block).
            let blocks_per_call = ready.harness.fixed_shape().map_or(1.0, |(block, _)| {
                block as f64 * f64::from(spec.r_in) / f64::from(ready.model_input_rate.max(1)) / spec.chunk_frames.max(1) as f64
            });
            let k = need
                .min(self.history.len() / need_samples)
                .min(sync_prime_blocks(need, self.call_ms, block_ms, blocks_per_call));
            let history = self.history.make_contiguous();
            for b in history[history.len() - k * need_samples..].chunks_exact(need_samples) {
                let _ = ready.harness.push_and_convert(b);
            }
            self.priming = Some(Priming {
                key: ready.key.clone(),
                fed: k,
                need,
            });
        }
        let p = self.priming.as_mut().expect("set above");
        if p.fed >= p.need {
            self.priming = None;
            return true;
        }
        let _ = ready.harness.push_and_convert(block);
        p.fed += 1;
        false
    }

    /// The ready voice becomes the active one (the previous one fades out over this block,
    /// `outgoing`). Returns whether there was one.
    fn switch_in(&mut self, level: &QualityLevel, h: &VoiceChangerHotParams) -> bool {
        let Some(old) = self.voices.take_ready() else { return false };
        self.active_selection = self.selection.clone();
        self.tuning = self.live;
        self.applied_controls = None;
        // The governor may have stepped while the voice was primed.
        self.voices.active_mut().harness.apply_quality(level);
        self.apply_tuning(h);
        self.outgoing = Some(old);
        self.start_probe();
        true
    }
}

/// A voice switch failed: put the selection back on the playing voice (with the per-voice values it
/// plays with) off the worker thread (entering a voice reads its `slot.json`), and tell the UI.
fn revert_failed_switch<R: Runtime>(
    app: &AppHandle<R>,
    generation: &RunGeneration,
    st: &VoiceChangerState,
    failed: VoiceSelection,
    playing: VoiceSelection,
    tuning: VoiceTuning,
) {
    let (app, generation, st) = (app.clone(), generation.clone(), st.clone());
    let spawned = std::thread::Builder::new().name("sigil-vc-revert".into()).spawn(move || {
        if !generation.is_current() {
            return;
        }
        if let Some(effective) = st.revert_voice_selection(&failed, &playing, &tuning) {
            tracing::info!("voice changer: the selection went back to {}", playing.slot_dir);
            if generation.is_current() {
                let _ = app.emit(PARAMS_EVENT, &effective);
            }
        }
    });
    if let Err(e) = spawned {
        tracing::warn!("voice changer: could not put the selection back on the playing voice ({e})");
    }
}

/// Publish the slot folders the run holds (`VoiceChangerState::held_slots`; the import commands
/// refuse to replace or delete them) when they changed since `published`: compared without
/// allocating, so the lock and the copy happen only on a change of the voice set (a request, a
/// finished or abandoned load, a switch, a retired voice). Non-RVC engines hold no voice.
fn publish_held_slots(st: &VoiceChangerState, generation: &RunGeneration, engine: &Engine, published: &mut Option<Vec<PathBuf>>) {
    let same = published.as_ref().is_some_and(|p| match engine {
        Engine::Rvc(run) => run.voices.slot_dirs_iter().eq(p.iter().map(PathBuf::as_path)),
        _ => p.is_empty(),
    });
    if same {
        return;
    }
    let dirs = match engine {
        Engine::Rvc(run) => run.voices.slot_dirs(),
        _ => Vec::new(),
    };
    *published = Some(dirs.clone());
    if generation.is_current() {
        *lock(&st.held_slots) = Some(dirs);
    }
}

/// Everything a voice switch reports to the user / metrics (its load warnings only on its first
/// activation, not on every switch back to a hot voice).
fn publish_voice<R: Runtime>(app: &AppHandle<R>, generation: &RunGeneration, st: &VoiceChangerState, voice: &mut LoadedVoice, levers_notes: bool) {
    if !generation.is_current() {
        return;
    }
    if !std::mem::replace(&mut voice.warnings_shown, true) {
        for warning in &voice.warnings {
            tracing::warn!("voice changer RVC: {warning}");
            emit_vc_error(app, "rvc_warning", warning.clone());
        }
    }
    if levers_notes {
        for note in &voice.levers.notes {
            tracing::info!("voice changer lever: {note}");
        }
    }
    *lock(&st.run_levers) = Some(voice.levers.clone());
    if let Some(o) = voice_ep_outcome(&voice.outcomes) {
        *lock(&st.run_ep) = Some(o);
    }
    // The block this voice converts per model call (what the Chunk size setting resolved to), shown by the
    // inspector next to the setting.
    st.metrics_engine_block_us
        .store(super::metrics::engine_block_us(voice.harness.fixed_shape().map(|(block, _)| block)), Ordering::Relaxed);
}

// ---------------------------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------------------------

/// The mode-specific engine of a run.
enum Engine {
    Classic,
    Onnx(Box<OnnxVoiceBridge>),
    Sidecar(Box<SidecarVoiceBridge>),
    Rvc(Box<RvcRun>),
}

/// What the active voice needs of the output queue (`CreatedEngine::min_output_buffer_ms`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct VoiceFloor {
    prefill_ms: f64,
    max_queue_ms: f64,
    /// A template voice's floor for blocks it converts in one call (`block_buffer`): the app's stock
    /// buffers are raised to it, a buffer the user typed only once the run proves that it cannot hold
    /// the voice's calls. Otherwise a voice family's minimum, which raises every stored value.
    block: bool,
}

/// The output ports' prefill and queue: the user's, raised to the active voice's floor. The timeline
/// and the end-to-end metric measure the real queue, so the raised buffer is counted in the latency.
///
/// - **No floor** (GPU voices, short blocks, bring-your-own voices, the other modes): the stored values.
/// - **A voice family's minimum** (`families::min_output_buffer_ms`; P5-I0): the larger of each stored
///   value and the minimum, whatever was stored.
/// - **A template voice's burst floor** (`block_buffer`): the same for the stock buffers (Low latency
///   35 / 70 ms, Stable 80 / 150 ms: `output_buffer_stock`); a buffer the user typed (Custom) stays
///   unless the run proved it cannot hold the voice's model calls (`custom_cannot_play`, from
///   `block_buffer::StoredBufferProof`).
///
/// Nothing stored is rewritten: the settings keep the user's numbers.
fn port_limits(h: &VoiceChangerHotParams, floor: Option<VoiceFloor>, custom_cannot_play: bool) -> PortLimits {
    let stored = || clamp_output_buffer_ms(h.output_prefill_ms, h.output_max_queue_ms);
    let Some(f) = floor else {
        let (prefill_ms, max_queue_ms) = stored();
        return PortLimits { prefill_ms, max_queue_ms, min_prefill_ms: 0.0 };
    };
    if f.block && !h.output_buffer_stock && !custom_cannot_play {
        // The user's own buffer: kept (the adaptive cushion gets no voice floor either).
        let (prefill_ms, max_queue_ms) = stored();
        return PortLimits { prefill_ms, max_queue_ms, min_prefill_ms: 0.0 };
    }
    let (prefill_ms, max_queue_ms) =
        clamp_output_buffer_ms(h.output_prefill_ms.max(f.prefill_ms), h.output_max_queue_ms.max(f.max_queue_ms));
    PortLimits {
        prefill_ms,
        max_queue_ms,
        // The adaptive cushion's floor: what the active voice needs.
        min_prefill_ms: f.prefill_ms,
    }
}

/// The device block of a run (frames): the setting, or under `device_block = model_divisor` a divisor
/// of a template voice's engine block (`chunk_processor::model_divisor_chunk`: 10 ms at the device
/// rate, so the harness holds back `block - chunk` instead of a burst cushion). With an explicit
/// `block_ms` that is a divisor of that block; with the block on automatic (the default) the engine
/// derives its block from the device block, so the divisor is taken from the block the configured chunk
/// derives and only when it derives the same one (`model_divisor_chunk_auto`). The mode applies to RVC
/// runs whose voice is a template voice; every other voice keeps the setting. The second value says why
/// the mode did not apply (for the log).
fn run_chunk_frames(st: &VoiceChangerState, setting: u32, rate: u32) -> (usize, Option<&'static str>) {
    let configured = clamp_chunk_frames(setting);
    if *lock(&st.mode) != VoiceChangerMode::Rvc {
        return (configured, None);
    }
    let rvc = lock(&st.rvc).clone();
    if rvc.device_block != DeviceBlockMode::ModelDivisor {
        return (configured, None);
    }
    let slot = rvc.slot_dir.trim();
    if slot.is_empty() || !matches!(super::factory::slot_format(Path::new(slot)), Ok(SlotFormat::SigilRvc)) {
        return (configured, Some("the voice is not a Sigil template voice"));
    }
    // The Streamer on the processor with the block on automatic runs the block it can keep up with, by
    // this PC's physical cores: 120, 160 or 200 ms (`presets::cpu_streamer_block`): for the divisor that is
    // an explicit block.
    let profile = RvcPresetProfile::from_str(&rvc.preset);
    let ep = effective_rvc_ep(&rvc.execution_provider, profile);
    let block_ms = (rvc.block_ms > 0.0)
        .then_some(rvc.block_ms)
        .or_else(|| presets::cpu_streamer_auto_block_ms(profile, ep, configured as u32, rate));
    let explicit = block_ms.is_some();
    let chunk = match block_ms {
        Some(block_ms) => model_divisor_chunk(block_ms, rate),
        None => model_divisor_chunk_auto(configured, rate),
    };
    match chunk {
        Some(chunk) => (chunk, None),
        None if explicit => (configured, Some("the sample rate has no whole-frame 10 ms")),
        None => (
            configured,
            Some("the engine block follows the device block here (set an engine block to divide), or the sample rate has no whole-frame 10 ms"),
        ),
    }
}

/// The active voice's output-buffer floor (RVC mode). A template voice's is the burst floor of
/// `block_buffer`; any other voice's is its family's minimum.
fn voice_output_floor(engine: &Engine) -> Option<VoiceFloor> {
    match engine {
        Engine::Rvc(run) => {
            let voice = run.voices.active();
            voice.min_output_buffer_ms.map(|(prefill_ms, max_queue_ms)| VoiceFloor {
                prefill_ms,
                max_queue_ms,
                block: voice.format == SlotFormat::SigilRvc,
            })
        }
        _ => None,
    }
}

fn ns_of_frames(frames: f64, rate: u32) -> i64 {
    (frames * 1e9 / f64::from(rate.max(1))) as i64
}

pub(super) fn run_voice_changer_loop<R: Runtime>(
    app: AppHandle<R>,
    generation: RunGeneration,
    rx: Receiver<()>,
    st: VoiceChangerState,
    input_device_name: Option<String>,
    output_device_name: String,
    monitor_output_device: String,
) -> Result<(), WorkerError> {
    let mut last_vc_error_emit: Option<Instant> = None;
    let mut stop = StopSignal {
        rx,
        is_running: st.is_running.clone(),
        generation: generation.clone(),
        stopped: false,
    };
    st.reset_run_metrics();
    let host = cpal::default_host();

    // ---- Devices: formats first (the models and rings are sized for them). ----
    let mut input: InputTarget = resolve_input(&host, input_device_name.as_deref().unwrap_or("")).map_err(WorkerError::device)?;
    if let Some(from) = &input.renamed_from {
        tracing::info!("voice changer: input device '{from}' was renumbered by Windows; using '{}'", input.name);
    }
    let (out_device, out_name, out_config) = resolve_output(&host, DeviceRole::Output, &output_device_name).map_err(WorkerError::device)?;
    let monitor_target = if monitor_output_device.trim().is_empty() {
        None
    } else {
        match resolve_output(&host, DeviceRole::Monitor, &monitor_output_device) {
            Ok(t) => Some(t),
            Err(e) => {
                warn_monitor_unavailable(&app, &generation, &e);
                None
            }
        }
    };
    let r_in = input.rate;
    let ch_in = input.ch;
    let r_out = out_config.sample_rate().0;
    let ch_out = usize::from(out_config.channels()).max(1);
    tracing::info!(
        "voice changer: input '{}' {} Hz {} ch (capture hub source '{}'), output '{}' {} Hz {} ch",
        input.name,
        r_in,
        ch_in,
        input.spec.key(),
        out_name,
        r_out,
        ch_out
    );

    // Rates are published before the (possibly slow) model load, so TTS pushed meanwhile is
    // queued in the capture format and plays once the loop runs.
    st.capture_sample_rate.store(r_in, Ordering::Relaxed);
    st.capture_channels.store(ch_in as u32, Ordering::Relaxed);
    st.output_sample_rate.store(r_out, Ordering::Relaxed);
    st.output_channels.store(ch_out as u32, Ordering::Relaxed);
    st.aux_feed.reset(r_in, ch_in);

    let h_init = *lock(&st.hot_params);
    let (chunk_frames, device_block_note) = run_chunk_frames(&st, h_init.chunk_frames, r_in);
    st.metrics_chunk_frames.store(chunk_frames as u32, Ordering::Relaxed);
    if chunk_frames != clamp_chunk_frames(h_init.chunk_frames) {
        tracing::info!(
            "voice changer: device block {chunk_frames} frames ({:.2} ms), a divisor of the engine block (device_block = model_divisor)",
            chunk_frames as f64 * 1000.0 / f64::from(r_in.max(1))
        );
    } else if let Some(why) = device_block_note {
        tracing::info!("voice changer: device_block = model_divisor does not apply: {why}");
    }
    let need = chunk_frames * ch_in;
    if need == 0 {
        return Err("Invalid chunk size or channel count.".into());
    }
    let block_ms = chunk_frames as f64 * 1000.0 / f64::from(r_in.max(1));

    // ---- Engines: every model and bridge loads before any audio flows. ----
    let mode = lock(&st.mode).clone();
    let load_started_ms = unix_ms_now();
    let mut classic = ClassicPhaseVocoderProcessor::new(r_in, r_out, ch_in, ch_out, h_init.vocoder_window_ms, chunk_frames)?;

    // ONNX Runtime's pool threads start with the worker priority policy (`auto` on the CPU EP and
    // `protect_game` are BELOW_NORMAL), so its intra-op pool never does the CPU work at NORMAL
    // priority beside the game. `protect_game` also caps the pool at 2 threads (the calling worker
    // plus one pool thread). Sessions are built once per run, so a live switch applies from the
    // next start.
    let ort_priority = h_init.worker_priority;
    let ort_intra_threads = (ort_priority == WorkerPriority::ProtectGame).then_some(2);

    let rvc_settings = lock(&st.rvc).clone();
    // The denoiser: where it runs depends on the mode (see the module docs).
    let (den_kind, den_note) = denoise::effective_kind(rvc_settings.denoiser);
    if let Some(note) = &den_note {
        tracing::warn!("voice changer denoiser: {note}");
        if generation.is_current() {
            emit_vc_error(&app, "denoiser_warning", note.clone());
        }
    }
    let in_harness_denoiser = if mode == VoiceChangerMode::Rvc && den_kind == DenoiserKind::Gtcrn {
        DenoiserKind::Gtcrn
    } else {
        DenoiserKind::Off
    };

    let budget_timeout = |ms: f64| Duration::from_secs_f64(ms.max(1.0) / 1000.0);
    let mut sidecar_timeout_ms = h_init.latency_budget_ms;
    let onnx_ep = normalize_execution_provider(&lock(&st.onnx_execution_provider));
    let mut rvc_ep = String::new();
    let mut vram_claim: Option<VramClaim> = None;
    let mut engine = match mode {
        VoiceChangerMode::Classic => Engine::Classic,
        VoiceChangerMode::Sidecar => {
            let url = lock(&st.sidecar_ws_url).clone();
            Engine::Sidecar(Box::new(
                SidecarVoiceBridge::connect(&url, r_in, ch_in, chunk_frames, budget_timeout(sidecar_timeout_ms))
                    .map_err(|e| WorkerError::new("sidecar_connect", e))?,
            ))
        }
        VoiceChangerMode::Onnx => {
            let path = lock(&st.onnx_model_path).clone();
            let opts = VcSessionOptions {
                intra_threads: ort_intra_threads,
                thread_priority: Some(ort_priority),
                ..VcSessionOptions::default()
            };
            let probe = VramProbe::for_rung(&onnx_ep, 0);
            let bridge = OnnxVoiceBridge::new_with_options(path.trim(), need, ch_in, &onnx_ep, &opts)
                .map_err(|e| WorkerError::new("onnx_load", e))?;
            let mb = probe.and_then(|p| p.delta_mb()).unwrap_or(0.0);
            vram_claim = Some(resources::claim_vram(Subsystem::Vc, "vc: onnx", if onnx_ep == "cpu" { 0 } else { mb.round() as u64 }));
            st.metrics_vram_mb.store(mb.round() as u32, Ordering::Relaxed);
            Engine::Onnx(Box::new(bridge))
        }
        VoiceChangerMode::Rvc => {
            // Every RVC load failure stops the worker as `rvc_load` (the epilogue shows it once,
            // unless a newer start has superseded this worker).
            let rvc_load_err = |e: String| WorkerError::new("rvc_load", e);
            let app_data = app.path().app_data_dir().map_err(|e| rvc_load_err(e.to_string()))?;
            let preset = RvcPresetProfile::from_str(&rvc_settings.preset);
            // RVC has its own EP (set by the profile or the expert select); ONNX chunk mode keeps
            // `onnx_execution_provider`. Unset falls back to the preset default. Sessions run on
            // the rung (`cpu` / `cuda` / `dml`); an accelerator is a lever on top.
            let requested = requested_rvc_ep(&rvc_settings.execution_provider, preset).to_string();
            let ep_hint = effective_rvc_ep(&rvc_settings.execution_provider, preset).to_string();
            // The Streamer preset on the CPU runs several threads on the below-normal pool (P7-I0: one
            // thread is not real time); every other preset keeps the caps above.
            let threads = crate::services::neural_audio::ort_session::vc_threads(preset, &ep_hint, ort_priority, None);
            let session = VcSessionOptions {
                cuda_device_index: rvc_settings.gpu_device_index,
                streamer_profile: threads.streamer_profile,
                intra_threads: threads.intra_threads,
                thread_priority: Some(ort_priority),
                // The arena cap is a CUDA EP option; DirectML/CPU ignore it.
                vram_cap_mb: (ep_hint == "cuda").then_some(rvc_settings.vram_cap_mb),
                // The voice changer's rows stay the plain `vc` / `vc.*` keys.
                component_prefix: None,
                // Benchmarked configuration (ort rc.13 default; P3B-G decides per preset).
                cpu_arena: crate::services::neural_audio::ort_session::vc_cpu_arena_default(preset),
                dynamic_block_base: None,
            };
            // Same rule as the inspector and the start preflight: a method this build cannot run
            // (or a GPU-only one on the CPU) runs its stand-in, and the user is told.
            if let Some(msg) = pitch_fallback_reason(&rvc_settings.pitch_extractor, &ep_hint) {
                tracing::warn!("voice changer: {msg}");
                if generation.is_current() {
                    emit_vc_error(&app, "rvc_warning", msg);
                }
            }
            let levers = resolve_levers(&rvc_settings.levers, preset, &requested, rvc_settings.precision);
            let base = RvcBase {
                app_data: app_data.clone(),
                templates_dir: app
                    .path()
                    .resource_dir()
                    .ok()
                    .map(|r| r.join("resources").join("vc-templates"))
                    .filter(|d| d.is_dir())
                    .or_else(import::templates_dir),
                ep: ep_hint.clone(),
                session,
                preset,
                chunk_frames: chunk_frames as u32,
                device_rate: r_in,
                in_harness_denoiser,
                latency_budget_ms: h_init.latency_budget_ms,
            };
            let spec = HarnessSpec {
                r_in,
                r_out,
                ch_in,
                ch_out,
                extra_context_ms: rvc_settings.extra_context_ms,
                crossfade_ms: rvc_settings.crossfade_ms,
                chunk_frames,
                sola_search_ms: rvc_settings.sola_search_ms,
            };
            let key = desired_key(&rvc_settings, &base);
            let mut request = voice_request(rvc_settings.clone(), &base, &levers).map_err(rvc_load_err)?;
            // Stop (or a newer start) during this load calls it off: the family stops waiting for
            // a sidecar helper, the factory at its next stage.
            let cancel = Arc::new(AtomicBool::new(false));
            request.cancel = Some(cancel.clone());
            let loaded = {
                let _watch = LoadStopWatch::start(st.is_running.clone(), generation.clone(), cancel.clone());
                LoadedVoice::load_cancellable(key, request, &spec, &cancel)
            };
            let mut voice = match loaded {
                Ok(v) => v,
                // Called off by Stop or a newer start: nothing to report (the epilogue says
                // `stopped` when this run is still current).
                Err(_) if cancel.load(Ordering::Relaxed) => return Ok(()),
                Err(e) => return Err(rvc_load_err(e)),
            };
            tracing::info!(
                "voice changer: loaded {} ({:?}) in {:.0} ms, {:.0} MB VRAM",
                voice.key.slot_dir.display(),
                voice.format,
                voice.load_ms,
                voice.vram_mb
            );
            publish_voice(&app, &generation, &st, &mut voice, true);
            let voices = VoiceSet::new(voice, rvc_settings.hot_slots, rvc_settings.vram_budget_mb);
            st.metrics_vram_mb.store(voices.vram_mb().round() as u32, Ordering::Relaxed);
            rvc_ep = ep_hint;
            // Voices of one run share the preset (block, context), so the first voice's priming
            // need stands for every voice switched to.
            let prime_blocks = voices.active().prime_blocks(&spec);
            let history_cap = ((SWITCH_HISTORY_MS * f64::from(r_in) / 1000.0) as usize).max(chunk_frames * (prime_blocks + 1)) * ch_in;
            let tuning = VoiceTuning::of(&rvc_settings, &h_init);
            let mut run = RvcRun {
                voices,
                base,
                levers,
                spec,
                outgoing: None,
                history: VecDeque::with_capacity(history_cap + need),
                history_cap,
                need,
                selection: VoiceSelection::of(&rvc_settings),
                active_selection: VoiceSelection::of(&rvc_settings),
                live: tuning,
                tuning,
                applied_controls: None,
                priming: None,
                call_ms: 0.0,
                formant_states: vec![FormantLpState::default(); ch_out],
                algo_latency: Arc::new(AtomicU64::new(0)),
                probe_generation: 0,
                needs_reset: false,
            };
            run.apply_tuning(&h_init);
            run.start_probe();
            Engine::Rvc(Box::new(run))
        }
    };
    if stop.requested() {
        return Ok(());
    }
    // The voices this run holds (the import commands refuse to replace or delete them).
    let mut held_published: Option<Vec<PathBuf>> = None;
    publish_held_slots(&st, &generation, &engine, &mut held_published);

    // The device-rate denoiser stage (every mode; RVC mode runs GTCRN inside the harness instead).
    let mut den_stage: Option<DenoiseStage> = None;
    if den_kind != DenoiserKind::Off && in_harness_denoiser == DenoiserKind::Off {
        let opts = VcSessionOptions {
            intra_threads: Some(if den_kind == DenoiserKind::Gtcrn { 1 } else { 2 }),
            thread_priority: Some(ort_priority),
            ..VcSessionOptions::default()
        };
        let den_rate = if den_kind == DenoiserKind::Gtcrn { denoise::gtcrn::SAMPLE_RATE } else { denoise::dfn::SAMPLE_RATE };
        let chunk = (den_rate == r_in).then_some(chunk_frames);
        match denoise::create_with(den_kind, "cpu", &opts, chunk) {
            Ok(d) => {
                let stage = DenoiseStage::new(d, den_kind, r_in, ch_in, chunk_frames);
                tracing::info!(
                    "voice changer denoiser: {} on the capture copy, {:.1} ms of latency",
                    den_kind.as_str(),
                    frames_to_ms(stage.latency_frames() as f64, r_in)
                );
                den_stage = Some(stage);
            }
            Err(e) => {
                let msg = format!("{e}; the voice changer runs without the denoiser");
                tracing::warn!("voice changer denoiser: {msg}");
                if generation.is_current() {
                    emit_vc_error(&app, "denoiser_warning", msg);
                }
            }
        }
    }

    // ---- The EP actually in use, and the worker priority that depends on it. ----
    let run_ep = match &engine {
        Engine::Rvc(run) => voice_ep_outcome(&run.voices.active().outcomes)
            .or_else(|| run_ep_outcome(&RVC_COMPONENTS, &rvc_ep, load_started_ms)),
        Engine::Onnx(_) => run_ep_outcome(&["vc.onnx", "vc"], &onnx_ep, load_started_ms),
        Engine::Classic | Engine::Sidecar(_) => None,
    };
    let gpu_ep = match (&run_ep, &engine) {
        (Some(o), _) => o.is_gpu(),
        (None, Engine::Rvc(_)) => rvc_ep != "cpu",
        (None, Engine::Onnx(_)) => onnx_ep == "cuda" || onnx_ep == "dml",
        _ => false,
    };
    if let Some(o) = &run_ep {
        match &o.reason {
            Some(why) if o.fell_back() => tracing::warn!("voice changer: requested {} but running on {}: {why}", o.requested, o.active),
            _ => tracing::info!("voice changer: running on the {} execution provider", o.active),
        }
    }
    // Resource governor: report the device the models landed on (DSP-only and sidecar modes run
    // no local model, so `false`). A superseded worker never reports over its successor.
    let report_gpu = match &engine {
        Engine::Rvc(_) | Engine::Onnx(_) => run_ep.as_ref().map(EpOutcome::is_gpu),
        Engine::Classic | Engine::Sidecar(_) => Some(false),
    };
    if let Some(on_gpu) = report_gpu.filter(|_| generation.is_current()) {
        resources::set_vc_gpu(on_gpu);
    }
    *lock(&st.run_ep) = run_ep;
    let mut applied_priority = h_init.worker_priority;
    apply_worker_priority(applied_priority, gpu_ep);

    // ---- The adaptive governor (RVC mode, the `governor` lever). ----
    let mut governor = match &engine {
        Engine::Rvc(run) if run.levers.levers.get(Lever::Governor) && governor::available() => {
            st.metrics_governor_level.store(0, Ordering::Relaxed);
            Some(Governor::new(GovernorConfig::for_run(block_ms, h_init.latency_budget_ms)))
        }
        _ => None,
    };

    // ---- Capture (the hub) and outputs. ----
    let hub = hub::global().map_err(WorkerError::device)?;
    let mut capture = HubInput::attach(hub, &input, Box::new(move || hub.is_muted(MuteGroup::VoiceChanger)))
        .map_err(|e| WorkerError::device(format!("The input device '{}' could not be started: {e}", input.name)))?;
    let mut capbuf = CaptureBuffer::new(r_in, ch_in, chunk_frames, h_init.latency_budget_ms);
    rt_thread::preload();
    let limits = port_limits(&h_init, voice_output_floor(&engine), false);
    let port_setup =|role: DeviceRole, saved: &str, underruns: Option<Arc<AtomicU64>>, which: fn(&super::state::MmcssStatus) -> &MmcssFlag| PortSetup {
        role,
        saved: saved.to_string(),
        run_rate: r_out,
        run_ch: ch_out,
        limits,
        underruns,
        mmcss: st.mmcss.clone(),
        which,
        burst: capture.burst(),
        burst_ch: ch_in,
        burst_rate: r_in,
        // The adaptive cushion applies to the main output of an RVC run; the monitor stays fixed.
        cushion: if role == DeviceRole::Output && mode == VoiceChangerMode::Rvc {
            rvc_settings.output_cushion
        } else {
            OutputCushion::Fixed
        },
    };
    let mut out = OutputPort::start(
        port_setup(DeviceRole::Output, &output_device_name, Some(st.metrics_underruns.clone()), |m| &m.output),
        out_device,
        out_name,
        out_config,
    )
    .map_err(|e| WorkerError::device(format!("The output device could not be started: {e}")))?;
    let mut monitor = match monitor_target {
        None => None,
        Some((dev, name, config)) => match OutputPort::start(port_setup(DeviceRole::Monitor, &monitor_output_device, None, |m| &m.monitor), dev, name, config) {
            Ok(p) => Some(p),
            Err(e) => {
                warn_monitor_unavailable(&app, &generation, &e);
                None
            }
        },
    };
    let streams_started = Instant::now();
    tracing::info!("voice changer: capture attached to '{}', output '{}' playing", capture.source(), out.name());

    // A block whose conversion fails is muted (never the dry microphone): ONNX / sidecar blocks
    // before the post chain (capture format), RVC ticks at the output.
    let mut conceal_in = MuteConceal::new(ch_in, r_in);
    let mut conceal_out = MuteConceal::new(ch_out, r_out);
    let mut concealed = Vec::with_capacity(need);
    let mut work_chunk = Vec::with_capacity(need);
    let mut scratch_gain = Vec::with_capacity(need * 4);
    let mut aux_mixer = AuxMixer::new(r_in, ch_in);
    let mut times = ProcessTimeWindow::new();
    let mut e2e = ProcessTimeWindow::new();
    // The model calls of a voice that converts long blocks, timed apart from the ticks between them.
    let mut call_times = CallTimes::new();
    // The latency budget in force last time it was above the stored one (to log a change once).
    let mut budget_raised_to = 0.0f64;
    // The evidence for raising a typed output buffer (`block_buffer::StoredBufferProof`), the floor it
    // belongs to and whether the raise was logged.
    let mut buffer_proof = StoredBufferProof::default();
    let mut proof_floor: Option<VoiceFloor> = None;
    let mut proof_logged = false;
    let mut mmcss_logged = false;
    let mut drop_log_at = Instant::now();
    let (mut logged_dropped, mut logged_events) = (0u64, 0u64);
    let mut seq = 0u64;
    let mut last_underruns = 0u64;
    let mut pending_flags = 0u8;
    let mut pending_drop_frames = 0u64;
    let mut den_warned_at: Option<Instant> = None;
    let mut input_follow = InputFollow::new(input_device_name.as_deref().unwrap_or(""));
    let mut input_lost_since: Option<Instant> = None;
    let mut input_lost_told = false;
    // The name the input had when it was lost, once the capture followed it to a new one.
    let mut input_renamed_from: Option<String> = None;

    loop {
        if stop.requested() {
            break;
        }

        // ---- Device follow (outputs; the hub follows the input). ----
        let mut restored = false;
        for (port, is_monitor) in [(Some(&mut out), false), (monitor.as_mut(), true)] {
            let Some(port) = port else { continue };
            match port.maintain() {
                None => {}
                Some(PortEvent::Lost(msg)) => {
                    let message = format!("{msg}; the voice changer keeps running and resumes when it is back.");
                    tracing::warn!("voice changer: {message}");
                    if generation.is_current() {
                        emit_vc_error(&app, "device_lost", message);
                    }
                }
                Some(event) => {
                    let message = match &event {
                        PortEvent::Restored {
                            name,
                            renamed_from: Some(old),
                        } => format!("Playing on '{name}' ('{old}' came back under this name)."),
                        PortEvent::Restored { name, .. } => format!("Playing on '{name}' again."),
                        PortEvent::Moved(name) => format!("Playing on '{name}', the new default device."),
                        PortEvent::Lost(_) => unreachable!("handled above"),
                    };
                    tracing::info!("voice changer {}: {message}", if is_monitor { "monitor" } else { "output" });
                    if generation.is_current() {
                        emit_vc_error(&app, "device_restored", message);
                    }
                    restored |= !is_monitor;
                }
            }
        }

        // ---- Input follow: the hub reconnects the microphone; tell the user when it stays away,
        // and follow a named one that came back renumbered. ----
        if capture.recovering() {
            let since = *input_lost_since.get_or_insert_with(Instant::now);
            if !input_lost_told && since.elapsed() >= INPUT_LOST_NOTICE_AFTER {
                input_lost_told = true;
                let message = format!(
                    "The input device '{}' was disconnected; the voice changer keeps running and resumes when it is back.",
                    input.name
                );
                tracing::warn!("voice changer: {message}");
                if generation.is_current() {
                    emit_vc_error(&app, "device_lost", message);
                }
            }
        } else if input_lost_since.take().is_some() {
            let renamed = input_renamed_from.take();
            if std::mem::take(&mut input_lost_told) {
                let message = match renamed {
                    Some(old) => format!("Capturing from '{}' ('{old}' came back under this name).", input.name),
                    None => format!("Capturing from '{}' again.", input.name),
                };
                tracing::info!("voice changer: {message}");
                if generation.is_current() {
                    emit_vc_error(&app, "device_restored", message);
                }
            }
        }
        if let Some(moved) = input_follow.poll(capture.recovering(), &input, Instant::now()) {
            match capture.reattach(hub, &moved) {
                Ok(()) => {
                    tracing::info!("voice changer: the input device '{}' came back as '{}'", input.name, moved.name);
                    let old = std::mem::replace(&mut input, moved).name;
                    input_renamed_from.get_or_insert(old);
                }
                Err(e) => tracing::debug!("voice changer: moving the capture to '{}' failed: {e}", moved.name),
            }
        }

        // ---- Hot params. ----
        let h = *lock(&st.hot_params);
        let floor = voice_output_floor(&engine);
        if floor != proof_floor {
            // Another voice (or none) plays: the calls remembered are not its calls.
            buffer_proof.reset();
            proof_floor = floor;
            proof_logged = false;
        }
        let custom_cannot_play =
            floor.is_some_and(|f| f.block) && !h.output_buffer_stock && buffer_proof.judge(h.output_prefill_ms);
        if custom_cannot_play && !proof_logged {
            proof_logged = true;
            tracing::info!(
                "voice changer: the output prefill of {:.0} ms cannot hold this voice's model calls (median {:.0} ms of the last {}); playing with at least {:.0} / {:.0} ms",
                h.output_prefill_ms,
                buffer_proof.median_call_ms().unwrap_or(0.0),
                block_buffer::CALL_WINDOW,
                floor.map_or(0.0, |f| f.prefill_ms),
                floor.map_or(0.0, |f| f.max_queue_ms),
            );
        } else if !custom_cannot_play {
            proof_logged = false;
        }
        let limits = port_limits(&h, floor, custom_cannot_play);
        st.metrics_limit_prefill_us
            .store((limits.prefill_ms * 1000.0).clamp(0.0, f64::from(u32::MAX - 1)) as u32, Ordering::Relaxed);
        st.metrics_limit_queue_us
            .store((limits.max_queue_ms * 1000.0).clamp(0.0, f64::from(u32::MAX - 1)) as u32, Ordering::Relaxed);
        out.set_limits(limits);
        if let Some(m) = monitor.as_mut() {
            m.set_limits(limits);
        }
        // A voice that converts a long block in one call needs a budget that holds a call's worth of capture.
        let budget_ms = effective_latency_budget_ms(
            h.latency_budget_ms,
            engine_block_ms_of(st.metrics_engine_block_us.load(Ordering::Relaxed)),
            block_ms,
        );
        if (budget_ms - h.latency_budget_ms).abs() > 0.5 {
            if (budget_ms - budget_raised_to).abs() > 0.5 {
                budget_raised_to = budget_ms;
                tracing::info!(
                    "voice changer: the latency budget of {:.0} ms is raised to {:.0} ms while this voice converts {:.0} ms chunks",
                    h.latency_budget_ms,
                    budget_ms,
                    engine_block_ms_of(st.metrics_engine_block_us.load(Ordering::Relaxed)).unwrap_or(0.0),
                );
            }
        } else {
            budget_raised_to = 0.0;
        }
        capbuf.set_budget_ms(budget_ms);
        if let Engine::Sidecar(b) = &mut engine {
            if h.latency_budget_ms != sidecar_timeout_ms {
                sidecar_timeout_ms = h.latency_budget_ms;
                b.set_exchange_timeout(budget_timeout(sidecar_timeout_ms));
            }
        }
        if h.worker_priority != applied_priority {
            applied_priority = h.worker_priority;
            apply_worker_priority(applied_priority, gpu_ep);
            let uses_ort = matches!(engine, Engine::Onnx(_) | Engine::Rvc(_));
            if uses_ort && applied_priority != ort_priority {
                tracing::info!(
                    "voice changer: the ONNX Runtime thread policy for {} applies from the next start",
                    applied_priority.as_str()
                );
            }
        }

        // ---- Intake: everything the hub has, then the latency budget. ----
        let intake = capture.pump(&mut capbuf, now_qpc_ns(), chunk_frames);
        if let Some(why) = intake.ended {
            return Err(WorkerError::device(format!(
                "The input device '{}' stopped ({why}); the voice changer stopped.",
                input.name
            )));
        }
        let dropped_now = capbuf.wake();
        let gaps = capbuf.take_gaps();
        if gaps > 0 || restored {
            // The input (or the output timeline) is discontinuous: stale model / denoiser state.
            pending_flags |= tick_flags::GAP;
            if let Engine::Rvc(run) = &mut engine {
                run.needs_reset = true;
            }
            if let Some(d) = den_stage.as_mut() {
                d.reset();
            }
        }
        if dropped_now > 0 || gaps > 0 || intake.hub_dropped > 0 {
            out.note_discontinuity();
            if let Some(m) = monitor.as_mut() {
                m.note_discontinuity();
            }
        }
        if dropped_now > 0 {
            pending_flags |= tick_flags::BUDGET_DROP;
            pending_drop_frames += dropped_now;
        }
        let waiting = capbuf.frames_waiting();
        st.metrics_backlog_frames.store(waiting.min(u32::MAX as usize) as u32, Ordering::Relaxed);
        st.metrics_dropped_frames.store(capbuf.dropped_total(), Ordering::Relaxed);
        if capbuf.dropped_total() > logged_dropped && drop_log_at.elapsed() >= DROP_LOG_EVERY {
            tracing::info!(
                "voice changer: latency budget ({:.0} ms) dropped {:.0} ms of stale input in {} step(s) since the last report",
                h.latency_budget_ms,
                frames_to_ms((capbuf.dropped_total() - logged_dropped) as f64, r_in),
                capbuf.drop_events() - logged_events
            );
            logged_dropped = capbuf.dropped_total();
            logged_events = capbuf.drop_events();
            drop_log_at = Instant::now();
        }
        if !mmcss_logged && streams_started.elapsed() >= MMCSS_LOG_AFTER {
            mmcss_logged = true;
            tracing::info!(
                "voice changer MMCSS Pro Audio: capture by the capture hub, output {}, monitor {}",
                mmcss_label(&st.mmcss.output),
                if monitor.is_some() { mmcss_label(&st.mmcss.monitor) } else { "off" }
            );
            let backend = out.backend_errors() + monitor.as_ref().map_or(0, |m| m.backend_errors());
            if backend > 0 {
                tracing::warn!("voice changer: {backend} audio backend error(s) reported by the output devices");
            }
        }

        // ---- RVC: the selection (a change while running is a voice switch), finished loads, and the
        // active voice's controls. ----
        if let Engine::Rvc(run) = &mut engine {
            let view = st.try_voice_view(|s, hp| (VoiceTuning::of(s, hp), (!run.selection.matches(s)).then(|| s.clone())));
            if let Some((live, changed)) = view {
                run.on_view(live, changed);
            }
            if let Some((msg, failed)) = run.on_poll() {
                tracing::warn!("voice changer: {msg}");
                if generation.is_current() {
                    emit_vc_error(&app, "rvc_load", msg);
                }
                if let Some(failed) = failed {
                    revert_failed_switch(&app, &generation, &st, failed, run.active_selection.clone(), run.tuning);
                }
            }
            run.apply_tuning(&h);
        }
        // A voice-set change of this tick's selection and poll, or of the last block's switch.
        publish_held_slots(&st, &generation, &engine, &mut held_published);

        // ---- A block. ----
        if !capbuf.take_block(&mut work_chunk, chunk_frames) {
            // Woken by the hub's next block (or a stop request); while silence is synthesized the
            // worker runs on its own clock, a block at a time.
            let wait = if capture.filling() || capture.recovering() {
                Duration::from_secs_f64(block_ms / 1000.0).min(PARK_TIMEOUT)
            } else {
                PARK_TIMEOUT
            };
            capture.wait(wait);
            continue;
        }
        if out.is_lost() {
            // Nothing can play: consume the input (the budget keeps it fresh), convert nothing, and
            // restart the model's stream when the device is back.
            if let Engine::Rvc(run) = &mut engine {
                run.needs_reset = true;
            }
            continue;
        }
        let t_start = now_qpc_ns();
        let t0 = Instant::now();
        let filled_block = capture.filling();
        // Capture time of this block's newest frame (what is still waiting came after it).
        let capture_end_ns = capture
            .newest_capture_ns()
            .filter(|_| !filled_block)
            .map(|newest| newest - ns_of_frames(capbuf.frames_waiting() as f64, r_in));

        // VC-only conditioning: gain, denoiser, TTS aux mix.
        let g_in = h.input_gain;
        if (g_in - 1.0).abs() > 1e-6 {
            for s in work_chunk.iter_mut() {
                *s = (*s * g_in).clamp(-1.0, 1.0);
            }
        }
        if let Some(d) = den_stage.as_mut() {
            d.process(&mut work_chunk);
            if let Some(e) = d.take_failure() {
                if den_warned_at.is_none_or(|t| t.elapsed() >= Duration::from_secs(30)) {
                    den_warned_at = Some(Instant::now());
                    let msg = format!("the {} denoiser failed ({e}); that audio passed un-denoised", d.kind().as_str());
                    tracing::warn!("voice changer: {msg}");
                    if generation.is_current() {
                        emit_vc_error(&app, "denoiser_warning", msg);
                    }
                }
            }
        }
        st.aux_feed.mix_into(&mut work_chunk, &mut aux_mixer, h.aux_mic_gate);

        let pitch = h.pitch_semitones;
        let formant = h.formant_shift;
        let window_ms = h.vocoder_window_ms;
        let oversample = h.vocoder_oversample;
        let mut model_ms = 0.0f64;
        let mut tick_error = false;
        let mut owed_frames = 0usize;
        let processed = match &mut engine {
            Engine::Onnx(ob) => {
                let tm = Instant::now();
                let converted: &[f32] = match ob.infer(&work_chunk) {
                    Ok(mut v) => {
                        conceal_in.pass(&mut v);
                        concealed = v;
                        &concealed
                    }
                    Err(e) => {
                        tick_error = true;
                        maybe_emit_vc_error(&app, "onnx_infer", e, &mut last_vc_error_emit);
                        conceal_in.mute_into(&mut concealed, chunk_frames);
                        &concealed
                    }
                };
                model_ms = tm.elapsed().as_secs_f64() * 1000.0;
                VoiceChunkPipeline::process_chunk(&mut classic, converted, pitch, formant, window_ms, oversample)?
            }
            Engine::Sidecar(bridge) => {
                let tm = Instant::now();
                let neural: &[f32] = match bridge.exchange(&work_chunk) {
                    Ok(mut v) if v.len() == need => {
                        conceal_in.pass(&mut v);
                        concealed = v;
                        &concealed
                    }
                    Ok(v) => {
                        tick_error = true;
                        maybe_emit_vc_error(&app, "sidecar_length", format!("expected {} samples, got {}", need, v.len()), &mut last_vc_error_emit);
                        conceal_in.mute_into(&mut concealed, chunk_frames);
                        &concealed
                    }
                    Err(e) => {
                        tick_error = true;
                        maybe_emit_vc_error(&app, "sidecar_exchange", e, &mut last_vc_error_emit);
                        conceal_in.mute_into(&mut concealed, chunk_frames);
                        &concealed
                    }
                };
                model_ms = tm.elapsed().as_secs_f64() * 1000.0;
                VoiceChunkPipeline::process_chunk(&mut classic, neural, pitch, formant, window_ms, oversample)?
            }
            Engine::Classic => VoiceChunkPipeline::process_chunk(&mut classic, &work_chunk, pitch, formant, window_ms, oversample)?,
            Engine::Rvc(run) => {
                if run.needs_reset {
                    run.needs_reset = false;
                    run.voices.active_mut().harness.reset_model_stream();
                    // The input before the discontinuity does not continue: a switch primes again
                    // from what comes after it.
                    run.history.clear();
                    run.priming = None;
                }
                // A voice ready to switch to: primed with the latest input, then crossfaded in over
                // this block.
                let level = governor.as_ref().map(|g| g.level()).unwrap_or(QualityLevel::FULL);
                if run.prime_ready(&work_chunk, &level, &h, block_ms) && run.switch_in(&level, &h) {
                    publish_voice(&app, &generation, &st, run.voices.active_mut(), true);
                    tracing::info!("voice changer: switched to {}", run.voices.active().key.slot_dir.display());
                    st.metrics_vram_mb.store(run.voices.vram_mb().round() as u32, Ordering::Relaxed);
                }
                run.push_history(&work_chunk);
                let tm = Instant::now();
                let active = run.voices.active_mut();
                let mut out_block = match active.harness.push_and_convert(&work_chunk) {
                    Ok(mut v) => {
                        conceal_out.pass(&mut v);
                        v
                    }
                    Err(e) => {
                        tick_error = true;
                        maybe_emit_vc_error(&app, "rvc_convert", e, &mut last_vc_error_emit);
                        // Exactly the device frames the harness owed for this tick, as on success, so
                        // a failed tick neither shortens nor lengthens the output stream; muted, never
                        // the dry microphone.
                        let mut muted = Vec::new();
                        conceal_out.mute_into(&mut muted, active.harness.last_owed_frames());
                        muted
                    }
                };
                owed_frames = active.harness.last_owed_frames();
                run.note_call_ms(tm.elapsed().as_secs_f64() * 1000.0);
                if let Some(mut old) = run.outgoing.take() {
                    if let Ok(old_block) = old.harness.push_and_convert(&work_chunk) {
                        crossfade_into(&old_block, &mut out_block, ch_out);
                    }
                    run.voices.retire(old);
                    st.metrics_vram_mb.store(run.voices.vram_mb().round() as u32, Ordering::Relaxed);
                }
                model_ms = tm.elapsed().as_secs_f64() * 1000.0;
                // The true formant shift (`formant_semitones`) runs in the voice's post stage; while
                // it is off, a voice set up before P3 keeps its classic spectral tilt (same
                // per-channel filter state on the fallback tick, so the tilt stays continuous).
                let tilt = rvc_output_tilt(run.tuning.formant_shift, run.tuning.formant_semitones);
                apply_formant_interleaved(&mut out_block, ch_out, tilt, &mut run.formant_states, r_out as f64);
                out_block
            }
        };

        if matches!(engine, Engine::Rvc(_)) {
            buffer_proof.note_tick(model_ms, block_ms);
        }
        if let Some(m) = monitor.as_mut() {
            // Always fed (silence at gain 0), so the monitor's queue keeps an unbroken timeline.
            let g_mon = h.monitor_gain;
            if (g_mon - 1.0).abs() > 1e-6 {
                scratch_gain.clear();
                scratch_gain.extend(processed.iter().map(|&s| (s * g_mon).clamp(-1.0, 1.0)));
                m.push(&scratch_gain, (false, 0.0, 0));
            } else {
                m.push(&processed, (false, 0.0, 0));
            }
        }

        let mut processed = processed;
        let g_out = h.output_gain;
        if (g_out - 1.0).abs() > 1e-6 {
            for s in processed.iter_mut() {
                *s = (*s * g_out).clamp(-1.0, 1.0);
            }
        }

        // ---- Timing metrics. ----
        let elapsed_us = t0.elapsed().as_micros().min(u128::from(u32::MAX)) as u32;
        st.metrics_process_us.store(elapsed_us, Ordering::Relaxed);
        let prev_ema = st.metrics_process_ema_us.load(Ordering::Relaxed);
        let next_ema = if prev_ema == 0 {
            elapsed_us
        } else {
            ((prev_ema as u64 * 3 + elapsed_us as u64) / 4) as u32
        };
        st.metrics_process_ema_us.store(next_ema, Ordering::Relaxed);
        st.metrics_process_p99_us.store(times.push(elapsed_us), Ordering::Relaxed);
        st.metrics_process_p50_us.store(times.p50().max(1), Ordering::Relaxed);
        st.metrics_process_p95_us.store(times.p95().max(1), Ordering::Relaxed);
        if let Engine::Rvc(run) = &engine {
            call_times.note_tick(
                run.probe_generation,
                elapsed_us,
                st.metrics_engine_block_us.load(Ordering::Relaxed),
                block_ms,
            );
            let (p50, p99) = call_times.stats_us().unwrap_or((METRIC_UNSET, METRIC_UNSET));
            st.metrics_call_p50_us.store(p50, Ordering::Relaxed);
            st.metrics_call_p99_us.store(p99, Ordering::Relaxed);
        }
        let t_end = now_qpc_ns();

        // ---- Output. ----
        let (queued_before, dev_rate) = out.queued_frames();
        out.push(&processed, (h.buffer_truncate_enabled, h.buffer_truncate_rms, h.buffer_truncate_chunks));
        let enqueue_ns = now_qpc_ns();
        st.metrics_buffer_samples
            .store(out.queued_samples().min(u32::MAX as usize) as u32, Ordering::Relaxed);
        st.metrics_cushion_us
            .store((out.cushion_ms() * 1000.0).clamp(0.0, f64::from(u32::MAX - 1)) as u32, Ordering::Relaxed);

        // ---- Governor. ----
        let backlog_ms = frames_to_ms(capbuf.frames_waiting() as f64, r_in);
        let dropped_ms = frames_to_ms(std::mem::take(&mut pending_drop_frames) as f64, r_in);
        let mut governor_level = 0u8;
        if let (Some(g), Engine::Rvc(run)) = (governor.as_mut(), &mut engine) {
            // The budget is a hot param: the backlog-pressure rule follows it.
            g.set_budget_ms(budget_ms);
            let tick = TickLoad {
                compute_ms: f64::from(elapsed_us) / 1000.0,
                block_ms,
                backlog_ms,
                dropped_ms,
            };
            let stage = &mut den_stage;
            let changed = g.observe_with(&tick, |level| {
                let model = run.voices.active_mut().harness.apply_quality(level);
                let den = stage.as_mut().is_some_and(|d| d.set_light(level.light_denoiser));
                model || den
            });
            governor_level = g.level().level;
            if let Some(level) = changed {
                pending_flags |= tick_flags::GOVERNOR_STEP;
                st.metrics_governor_level.store(u32::from(level.level), Ordering::Relaxed);
                tracing::info!(
                    "voice changer governor: quality level {} (retrieval {}, context {:.0} %, light F0 {}, light denoiser {}) after {:.1} ms compute / {:.1} ms blocks, {:.0} ms backlog",
                    level.level,
                    level.retrieval,
                    level.context_scale * 100.0,
                    level.light_f0,
                    level.light_denoiser,
                    tick.compute_ms,
                    block_ms,
                    backlog_ms
                );
            }
        }

        // ---- Latency estimate and the timeline. ----
        let underruns = st.metrics_underruns.load(Ordering::Relaxed);
        if underruns != last_underruns {
            last_underruns = underruns;
            pending_flags |= tick_flags::UNDERRUN;
        }
        let algo_ns = match &engine {
            Engine::Rvc(run) => {
                let us = run.algo_latency_us();
                (us > 0).then(|| i64::from(us) * 1_000 + run.active_latency_extra_ns())
            }
            _ => Some(0),
        };
        let den_ns = den_stage.as_ref().map_or(0, |d| ns_of_frames(d.latency_frames() as f64, r_in));
        let first_play_ns = enqueue_ns + ns_of_frames(queued_before as f64, dev_rate);
        let owed = if owed_frames > 0 { owed_frames } else { processed.len() / ch_out };
        let e2e_ns = match (capture_end_ns, algo_ns) {
            (Some(cap), Some(algo)) if !out.is_lost() => {
                let v = first_play_ns + ns_of_frames(owed as f64, r_out) + algo + den_ns - cap;
                (v > 0).then_some(v)
            }
            _ => None,
        };
        if let Some(v) = e2e_ns {
            e2e.push((v / 1_000).clamp(1, i64::from(u32::MAX)) as u32);
            st.metrics_e2e_us.store(e2e.p50().max(1), Ordering::Relaxed);
        }
        let mut flags = std::mem::take(&mut pending_flags);
        if tick_error {
            flags |= tick_flags::ERROR;
        }
        if filled_block {
            flags |= tick_flags::SILENT;
        }
        // Only the current run publishes (a superseded worker still finishing a block must not mix
        // its ticks into a live-record of its successor).
        if timeline::has_subscribers() && generation.is_current() {
            timeline::publish(&VcTick {
                seq,
                capture_end_ns: capture_end_ns.unwrap_or(0),
                start_ns: t_start,
                end_ns: t_end,
                enqueue_ns,
                est_playout_ns: first_play_ns,
                block_frames: chunk_frames as u32,
                backlog_ms: backlog_ms as f32,
                dropped_ms: dropped_ms as f32,
                queued_out_ms: frames_to_ms(out.queued_frames().0 as f64, dev_rate) as f32,
                model_ms: model_ms as f32,
                governor_level,
                flags,
            });
        }
        seq += 1;
    }

    drop(vram_claim);
    // Published rates/metrics are cleared by the worker epilogue in `start_voice_changer`
    // (generation-guarded), which also covers the early `return Err` paths above.
    let (trimmed, recoveries) = out.trims();
    tracing::info!(
        "voice changer: audio streams stopped ({} blocks; output latency trimmed {:.0} ms in {} recovery step(s))",
        seq,
        frames_to_ms(trimmed as f64, r_out),
        recoveries
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::super::devices::tests::{target, test_hub, ScriptedMic, STEREO_48K};
    use super::*;
    use crate::services::diagnostics::record_ep_outcome;

    /// The first load's watch calls the load off on Stop or a newer start, and never after it is
    /// dropped (the load finished).
    #[test]
    fn the_first_load_is_called_off_by_stop_or_a_newer_start() {
        let wait_for = |flag: &AtomicBool| {
            let deadline = Instant::now() + Duration::from_secs(5);
            while !flag.load(Ordering::Relaxed) && Instant::now() < deadline {
                std::thread::sleep(Duration::from_millis(5));
            }
            flag.load(Ordering::Relaxed)
        };
        let generation = RunGeneration { counter: Arc::new(AtomicU64::new(1)), mine: 1 };
        // Stop clears `is_running`.
        let (running, flag) = (Arc::new(AtomicBool::new(true)), Arc::new(AtomicBool::new(false)));
        let watch = LoadStopWatch::start(running.clone(), generation.clone(), flag.clone());
        std::thread::sleep(Duration::from_millis(30));
        assert!(!flag.load(Ordering::Relaxed), "a running load is left alone");
        running.store(false, Ordering::Release);
        assert!(wait_for(&flag), "Stop calls the load off");
        drop(watch);
        // A newer start supersedes the run.
        let flag = Arc::new(AtomicBool::new(false));
        let _watch = LoadStopWatch::start(Arc::new(AtomicBool::new(true)), generation.clone(), flag.clone());
        generation.counter.store(2, Ordering::SeqCst);
        assert!(wait_for(&flag), "a newer start calls the load off");
        // Dropped (the load ended): a later Stop changes nothing.
        let (running, flag) = (Arc::new(AtomicBool::new(true)), Arc::new(AtomicBool::new(false)));
        let current = RunGeneration { counter: Arc::new(AtomicU64::new(1)), mine: 1 };
        drop(LoadStopWatch::start(running.clone(), current, flag.clone()));
        running.store(false, Ordering::Release);
        std::thread::sleep(LOAD_WATCH_PERIOD * 2);
        assert!(!flag.load(Ordering::Relaxed), "the watch ends with the load");
    }

    /// Every way a worker stops by itself carries a code the epilogue shows: a plain `?` on a
    /// helper's `String` error is `worker_error`, device problems are `device`.
    #[test]
    fn worker_errors_carry_a_code() {
        fn helper() -> Result<(), String> {
            Err("boom".into())
        }
        fn run() -> Result<(), WorkerError> {
            helper()?;
            Ok(())
        }
        assert_eq!(run().unwrap_err(), WorkerError::new("worker_error", "boom"));
        let e = WorkerError::device("The input device 'Mic' is not connected.");
        assert_eq!(e.code, "device");
        assert_eq!(e.to_string(), "The input device 'Mic' is not connected. (device)");
        assert_eq!(WorkerError::from("x").code, "worker_error");
    }

    /// A voice family's minimum (P5-I0) and a template voice's burst floor (`block_buffer`) at 120 ms blocks.
    const FAMILY_FLOOR: Option<VoiceFloor> = Some(VoiceFloor { prefill_ms: 80.0, max_queue_ms: 150.0, block: false });
    const BLOCK_FLOOR: Option<VoiceFloor> = Some(VoiceFloor { prefill_ms: 110.0, max_queue_ms: 136.0, block: true });

    /// Hot params with the given stored output buffer; `stock` says it is one of the app's templates.
    fn buffer(prefill: f64, queue: f64, stock: bool) -> VoiceChangerHotParams {
        VoiceChangerHotParams {
            output_prefill_ms: prefill,
            output_max_queue_ms: queue,
            output_buffer_stock: stock,
            ..VoiceChangerHotParams::default()
        }
    }

    /// The limits and the adaptive cushion's floor of a stored buffer (`port_limits` takes the params by
    /// reference: the stored numbers are the user's and are never rewritten).
    fn limits_of(h: VoiceChangerHotParams, floor: Option<VoiceFloor>, custom_cannot_play: bool) -> (f64, f64, f64) {
        let l = port_limits(&h, floor, custom_cannot_play);
        (l.prefill_ms, l.max_queue_ms, l.min_prefill_ms)
    }

    /// A burst family's minimum raises every stored buffer, typed or not, and never lowers one (P5-I0).
    #[test]
    fn a_voice_familys_minimum_raises_the_output_buffer_never_lowers_it() {
        for stock in [true, false] {
            assert_eq!(limits_of(buffer(35.0, 70.0, stock), None, false), (35.0, 70.0, 0.0), "no floor: the stored values");
            assert_eq!(limits_of(buffer(35.0, 70.0, stock), FAMILY_FLOOR, false), (80.0, 150.0, 80.0), "stock {stock}");
            assert_eq!(limits_of(buffer(120.0, 240.0, stock), FAMILY_FLOOR, false), (120.0, 240.0, 80.0), "a larger user buffer stays");
            assert_eq!(limits_of(buffer(60.0, 65.0, stock), FAMILY_FLOOR, false), (80.0, 150.0, 80.0));
        }
    }

    /// The stock buffers (Low latency 35 / 70 ms, Stable 80 / 150 ms) are raised to the floor of a CPU
    /// template voice on 120 ms blocks (110 / 136 ms, `block_buffer`): each field is the larger of the
    /// stored value and the floor.
    #[test]
    fn the_stock_buffers_are_raised_to_the_block_floor() {
        assert_eq!(limits_of(buffer(35.0, 70.0, true), BLOCK_FLOOR, false), (110.0, 136.0, 110.0), "the low-latency default");
        assert_eq!(limits_of(buffer(80.0, 150.0, true), BLOCK_FLOOR, false), (110.0, 150.0, 110.0), "the stable preset");
        // The run's evidence is about typed buffers; it changes nothing for a stock one.
        assert_eq!(limits_of(buffer(35.0, 70.0, true), BLOCK_FLOOR, true), (110.0, 136.0, 110.0));
        // No floor on the voice: exactly the stored values.
        assert_eq!(limits_of(buffer(35.0, 70.0, true), None, false), (35.0, 70.0, 0.0));
        assert_eq!(limits_of(buffer(80.0, 150.0, true), None, false), (80.0, 150.0, 0.0));
    }

    /// What happens to a buffer the user typed (the stored-value rule): it is kept, whatever its
    /// distance from the floor, until the run proves that it cannot hold the voice's model calls; from
    /// then on each field is the larger of it and the floor. A larger typed value is never touched.
    #[test]
    fn a_typed_buffer_stays_unless_the_run_proves_it_cannot_play() {
        let cases = [
            // (stored, kept as typed, raised once proven)
            ((35.0, 70.0), (35.0, 70.0), (110.0, 136.0)),
            ((5.0, 10.0), (5.0, 10.0), (110.0, 136.0)), // the smallest the sliders allow
            ((60.0, 65.0), (60.0, 65.0), (110.0, 136.0)),
            ((80.0, 150.0), (80.0, 150.0), (110.0, 150.0)),
            ((100.0, 200.0), (100.0, 200.0), (110.0, 200.0)),
            ((115.0, 120.0), (115.0, 120.0), (115.0, 136.0)),
            ((60.0, 400.0), (60.0, 400.0), (110.0, 400.0)),
            ((110.0, 136.0), (110.0, 136.0), (110.0, 136.0)),
            ((120.0, 240.0), (120.0, 240.0), (120.0, 240.0)),
            ((300.0, 600.0), (300.0, 600.0), (300.0, 600.0)),
        ];
        for ((prefill, queue), kept, raised) in cases {
            let (p, q, floor) = limits_of(buffer(prefill, queue, false), BLOCK_FLOOR, false);
            assert_eq!((p, q), kept, "typed {prefill} / {queue}, no evidence");
            assert_eq!(floor, 0.0, "a typed buffer gets no voice floor for the adaptive cushion");
            let (p, q, floor) = limits_of(buffer(prefill, queue, false), BLOCK_FLOOR, true);
            assert_eq!((p, q), raised, "typed {prefill} / {queue}, proven");
            assert_eq!(floor, 110.0);
            // Never lowered, in either state.
            assert!(p >= prefill && q >= queue);
        }
        // The same numbers as a stock buffer are raised without any evidence.
        assert_eq!(limits_of(buffer(60.0, 65.0, true), BLOCK_FLOOR, false), (110.0, 136.0, 110.0));
        // Evidence about a typed buffer changes nothing on a voice without a floor.
        assert_eq!(limits_of(buffer(60.0, 65.0, false), None, true), (60.0, 65.0, 0.0));
        // A family's minimum is not a template voice's: it raises a typed buffer without any evidence.
        assert_eq!(limits_of(buffer(60.0, 65.0, false), FAMILY_FLOOR, false), (80.0, 150.0, 80.0));
    }

    #[test]
    fn caught_panic_payload_is_reported() {
        let p = std::panic::catch_unwind(|| panic!("boom {}", 7)).unwrap_err();
        assert_eq!(panic_message(p.as_ref()), "boom 7");
        let p = std::panic::catch_unwind(|| panic!("static")).unwrap_err();
        assert_eq!(panic_message(p.as_ref()), "static");
    }

    /// The run's EP comes from outcomes recorded during its own load; a fallen-back component
    /// wins and is named; a CPU request needs no record; a GPU request without one is unknown.
    #[test]
    fn run_ep_prefers_fresh_fallbacks() {
        // Unique component names so parallel tests never see each other's rows.
        let (main, pitch) = ("test.worker.vc", "test.worker.vc.pitch");
        let stale_since = super::unix_ms_now();
        record_ep_outcome(main, EpOutcome::granted("cuda"));
        let since = super::unix_ms_now();
        assert_eq!(run_ep_outcome(&[main, pitch], "cuda", since + 60_000), None, "stale rows are ignored");
        assert_eq!(run_ep_outcome(&[main, pitch], "cpu", since + 60_000), Some(EpOutcome::granted("cpu")));
        assert_eq!(run_ep_outcome(&[main, pitch], "cuda", stale_since), Some(EpOutcome::granted("cuda")));
        record_ep_outcome(pitch, EpOutcome::fallback("cuda", "cpu", "cudnn64_9.dll not found"));
        let got = run_ep_outcome(&[main, pitch], "cuda", stale_since).unwrap();
        assert!(got.fell_back());
        assert_eq!(got.reason.as_deref(), Some("test.worker.vc.pitch: cudnn64_9.dll not found"));
    }

    /// A voice's own outcomes: a fallback anywhere wins (named), else the generator's.
    #[test]
    fn voice_ep_prefers_fallbacks_then_the_generator() {
        let ok = vec![
            ("vc.content".to_string(), EpOutcome::granted("cuda")),
            ("vc.flowdec".to_string(), EpOutcome::granted("dml")),
        ];
        assert_eq!(voice_ep_outcome(&ok), Some(EpOutcome::granted("dml")));
        let mut bad = ok.clone();
        bad.push(("vc.pitch".to_string(), EpOutcome::fallback("dml", "cpu", "no D3D12")));
        let got = voice_ep_outcome(&bad).unwrap();
        assert!(got.fell_back());
        assert_eq!(got.reason.as_deref(), Some("F0 model: no D3D12"));
        assert_eq!(voice_ep_outcome(&[]), None);
    }

    fn now() -> i64 {
        now_qpc_ns()
    }

    fn wait_until(what: &str, mut cond: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !cond() {
            assert!(Instant::now() < deadline, "timed out waiting for: {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// Plan T7 through the real capture path (hub consumer, capture ring, latency budget): a 2.5 s
    /// worker stall is back within the budget at the first wake, the backlog never exceeds the
    /// budget plus one block after a wake, and every captured frame is accounted for (consumed,
    /// dropped or waiting).
    #[test]
    fn a_stalled_worker_recovers_within_the_budget_through_the_hub() {
        let mic = ScriptedMic::new(STEREO_48K);
        let hub = test_hub(&mic);
        let mut capture = HubInput::attach(&hub, &target(STEREO_48K), Box::new(|| false)).unwrap();
        wait_until("stream live", || mic.live());
        let (block, budget_ms) = (512usize, 300.0);
        let mut cap = CaptureBuffer::new(48_000, 2, block, budget_ms);
        let limit = (budget_ms * 48.0) as usize + block;
        let mut pushed = 0u64;
        let mut out = Vec::new();
        let mut consumed = 0u64;
        let mut wake = |cap: &mut CaptureBuffer, capture: &mut HubInput, consumed: &mut u64| {
            let intake = capture.pump(cap, now(), block);
            assert_eq!(intake.filled, 0, "{intake:?}");
            cap.wake();
            let waiting = cap.frames_waiting();
            while cap.take_block(&mut out, block) {
                *consumed += block as u64;
            }
            waiting
        };
        // 1 s of normal operation.
        for _ in 0..100 {
            assert!(mic.push(&vec![0.1f32; 480 * 2]));
            pushed += 480;
            assert!(wake(&mut cap, &mut capture, &mut consumed) <= limit);
        }
        // The worker stalls for 2.5 s while capture continues.
        for _ in 0..250 {
            assert!(mic.push(&vec![0.2f32; 480 * 2]));
            pushed += 480;
        }
        let first = wake(&mut cap, &mut capture, &mut consumed);
        assert!(first <= limit, "back within the budget at the first wake: {first} > {limit}");
        assert!(cap.dropped_total() >= (2_500 * 48 - limit) as u64, "{}", cap.dropped_total());
        for _ in 0..50 {
            assert!(mic.push(&vec![0.1f32; 480 * 2]));
            pushed += 480;
            assert!(wake(&mut cap, &mut capture, &mut consumed) <= limit);
        }
        assert_eq!(cap.backlog.captured_frames(), pushed, "every captured frame reached the backlog");
        assert_eq!(
            consumed + cap.dropped_total() + cap.frames_waiting() as u64,
            pushed,
            "captured == consumed + dropped + waiting"
        );
        assert_eq!(cap.backlog.consumed_frames(), consumed);
    }

    /// A pure-delay "denoiser" at `rate` (for the stage's timing tests).
    struct Delay {
        rate: u32,
        line: VecDeque<f32>,
    }

    impl Delay {
        fn new(rate: u32, d: usize) -> Delay {
            Delay {
                rate,
                line: std::iter::repeat_n(0.0, d).collect(),
            }
        }
    }

    impl Denoiser for Delay {
        fn sample_rate(&self) -> u32 {
            self.rate
        }
        fn latency_samples(&self) -> usize {
            self.line.len()
        }
        fn process(&mut self, audio: &mut [f32]) -> Result<(), String> {
            for s in audio.iter_mut() {
                self.line.push_back(*s);
                *s = self.line.pop_front().unwrap_or(0.0);
            }
            Ok(())
        }
        fn reset(&mut self) {}
    }

    fn sine(frames: usize, ch: usize, hz: f32, rate: u32) -> Vec<f32> {
        (0..frames)
            .flat_map(|i| std::iter::repeat_n((2.0 * std::f32::consts::PI * hz * i as f32 / rate as f32).sin() * 0.5, ch))
            .collect()
    }

    /// At the device rate the stage is the denoiser's exact delay on the mono mix, written to every
    /// channel; switching the governor's light step on and off never moves that timeline (the dry
    /// path runs through a matching delay, and transitions crossfade).
    #[test]
    fn the_denoise_stage_keeps_a_constant_delay_through_bypass_switches() {
        let (rate, ch, block) = (48_000u32, 2usize, 480usize);
        let mut stage = DenoiseStage::new(Box::new(Delay::new(rate, 700)), DenoiserKind::Dpdfnet, rate, ch, block);
        assert_eq!(stage.latency_frames(), 700);
        let input = sine(block * 200, ch, 440.0, rate);
        let mut out = Vec::new();
        for (k, b) in input.chunks(block * ch).enumerate() {
            match k {
                50 => assert!(stage.set_light(true)),
                51 => assert!(!stage.set_light(true), "already bypassed"),
                120 => assert!(stage.set_light(false)),
                _ => {}
            }
            let mut b = b.to_vec();
            stage.process(&mut b);
            out.extend_from_slice(&b);
        }
        let d = stage.latency_frames();
        for f in d..input.len() / ch {
            assert!((out[f * ch] - input[(f - d) * ch]).abs() < 1e-6, "frame {f}");
            assert_eq!(out[f * ch], out[f * ch + 1]);
        }
    }

    /// A 16 kHz denoiser on a 48 kHz device: resampled both ways, still one constant delay (the
    /// band-limited copy of the input).
    #[test]
    fn the_denoise_stage_resamples_to_the_model_rate() {
        let (rate, ch, block) = (48_000u32, 1usize, 512usize);
        let mut stage = DenoiseStage::new(Box::new(Delay::new(16_000, 100)), DenoiserKind::Gtcrn, rate, ch, block);
        let d = stage.latency_frames();
        assert!((300..500).contains(&d), "{d}");
        let input = sine(block * 100, ch, 300.0, rate);
        let mut out = Vec::new();
        for b in input.chunks(block) {
            let mut b = b.to_vec();
            stage.process(&mut b);
            assert_eq!(b.len(), block);
            out.extend_from_slice(&b);
        }
        // Find the delay by correlation and check it is the reported one (within the resamplers'
        // fractional alignment).
        let n = 8_000;
        let best = (d.saturating_sub(10)..d + 10)
            .map(|lag| {
                let e: f32 = (0..n).map(|i| (out[20_000 + i] - input[20_000 + i - lag]).abs()).sum();
                (lag, e)
            })
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert!(best.0.abs_diff(d) <= 2, "delay {} vs reported {d}", best.0);
        assert!(best.1 / (n as f32) < 0.01, "mean error {}", best.1 / n as f32);
    }

    /// A catch-up read longer than the whole capture ring (a 3 s stall on a 192 kHz 8-channel
    /// interface: the ring holds 2.7 s) keeps the newest audio; the budget then drops the oldest, and
    /// every frame is accounted for.
    #[test]
    fn an_oversized_catch_up_read_keeps_the_newest_audio() {
        let (rate, ch, block) = (192_000u32, 8usize, 512usize);
        let mut cap = CaptureBuffer::new(rate, ch, block, 300.0);
        let frames = 3 * rate as usize;
        assert!(capture_ring_capacity(rate, ch, block) / ch < frames, "the read must exceed the ring");
        let value = |i: usize| i as f32 * 1e-6;
        let read: Vec<f32> = (0..frames).flat_map(|i| std::iter::repeat_n(value(i), ch)).collect();
        cap.audio(&read);
        cap.wake();
        let waiting = cap.frames_waiting();
        assert!(waiting <= (0.3 * f64::from(rate)) as usize + block, "within the budget: {waiting}");
        let mut out = Vec::new();
        assert!(cap.take_block(&mut out, waiting));
        assert_eq!(out[out.len() - ch], value(frames - 1), "the newest frame is kept");
        assert_eq!(cap.backlog.captured_frames(), frames as u64);
        assert_eq!(cap.backlog.consumed_frames() + cap.dropped_total() + cap.frames_waiting() as u64, frames as u64);
    }

    /// The hub's gap reaches the budget as a splice: the audio before it fades out, the audio after
    /// it fades in, and the gap is counted.
    #[test]
    fn capture_gaps_are_spliced_in_the_backlog() {
        let mut cap = CaptureBuffer::new(48_000, 1, 480, 2_000.0);
        cap.audio(&vec![0.5f32; 4_800]);
        cap.gap(0);
        cap.audio(&vec![0.5f32; 4_800]);
        cap.wake();
        assert_eq!(cap.take_gaps(), 1);
        let mut out = Vec::new();
        assert!(cap.take_block(&mut out, 4_800));
        assert_eq!(out[0], 0.5);
        assert!(out[4_799].abs() < 0.01, "faded out into the gap: {}", out[4_799]);
        assert!(cap.take_block(&mut out, 4_800));
        assert!(out[0].abs() < 0.01 && out[4_799] == 0.5, "faded back in");
        assert_eq!(cap.dropped_total(), 1, "a gap counts as one lost frame");
    }

    // ---- voice switching (RvcRun) ----

    use super::super::factory::{VoiceLoader, LOAD_CANCELLED, SWITCH_PRIME_BLOCKS};
    use crate::services::neural_audio::vc::test_models;
    use super::super::params::VoiceChangerParams;
    use crate::services::neural_audio::vc::traits::{ConversionCtx, VoiceConversionModel};
    use std::sync::Mutex;

    /// What a stand-in voice was asked: the controls of each conversion, and its quality level.
    #[derive(Default)]
    struct Seen {
        /// (pitch, index rate, protect, speaker) per conversion.
        calls: Vec<(f32, f32, f32, i64)>,
        level: Option<QualityLevel>,
    }

    /// A stand-in voice that records what it is asked (16 kHz in and out).
    struct Probe(Arc<Mutex<Seen>>);

    impl VoiceConversionModel for Probe {
        fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
            self.0
                .lock()
                .unwrap()
                .calls
                .push((ctx.pitch_shift_semitones, ctx.index_rate, ctx.protect, ctx.speaker_id));
            Ok(vec![0.1; ctx.new_frames])
        }
        fn input_sample_rate(&self) -> u32 {
            16_000
        }
        fn output_sample_rate(&self) -> u32 {
            16_000
        }
        fn apply_quality(&mut self, level: &QualityLevel) -> bool {
            self.0.lock().unwrap().level.replace(*level) != Some(*level)
        }
    }

    const BLOCK: usize = 480;

    fn spec() -> HarnessSpec {
        HarnessSpec {
            r_in: 48_000,
            r_out: 48_000,
            ch_in: 1,
            ch_out: 1,
            extra_context_ms: 32.0,
            crossfade_ms: 10.0,
            chunk_frames: BLOCK,
            sola_search_ms: None,
        }
    }

    fn base() -> RvcBase {
        RvcBase {
            app_data: std::env::temp_dir(),
            templates_dir: None,
            ep: "cpu".into(),
            session: VcSessionOptions::default(),
            preset: RvcPresetProfile::LowLatency,
            chunk_frames: BLOCK as u32,
            device_rate: 48_000,
            in_harness_denoiser: DenoiserKind::Off,
            latency_budget_ms: 300.0,
        }
    }

    /// RVC settings selecting the voice folder `name` with these per-voice values.
    fn voice_settings(name: &str, speaker_id: i64, index_rate: f32) -> RvcParamsState {
        RvcParamsState {
            slot_dir: format!("C:/voices/{name}"),
            pitch_extractor: "fcpe".into(),
            speaker_id,
            index_rate,
            ..RvcParamsState::default()
        }
    }

    fn hot(pitch: f32) -> VoiceChangerHotParams {
        VoiceChangerHotParams {
            pitch_semitones: pitch,
            ..VoiceChangerHotParams::default()
        }
    }

    fn probe_voice(settings: &RvcParamsState) -> (LoadedVoice, Arc<Mutex<Seen>>) {
        let seen = Arc::new(Mutex::new(Seen::default()));
        let key = desired_key(settings, &base());
        (LoadedVoice::for_test(key, Box::new(Probe(seen.clone())), &spec(), 0.0), seen)
    }

    fn run_with(voices: VoiceSet, settings: &RvcParamsState, h: &VoiceChangerHotParams) -> RvcRun {
        let tuning = VoiceTuning::of(settings, h);
        let mut run = RvcRun {
            voices,
            base: base(),
            levers: EffectiveLevers::default(),
            spec: spec(),
            outgoing: None,
            history: VecDeque::new(),
            history_cap: BLOCK * 50,
            need: BLOCK,
            selection: VoiceSelection::of(settings),
            active_selection: VoiceSelection::of(settings),
            live: tuning,
            tuning,
            applied_controls: None,
            priming: None,
            call_ms: 0.0,
            formant_states: vec![FormantLpState::default(); 1],
            algo_latency: Arc::new(AtomicU64::new(0)),
            probe_generation: 0,
            needs_reset: false,
        };
        run.apply_tuning(h);
        run
    }

    /// One worker tick of the RVC path: the params view, finished loads, the controls, a possible
    /// switch and the active voice's conversion. Returns a failed switch (message, selection).
    fn tick(
        run: &mut RvcRun,
        view: Option<(&RvcParamsState, &VoiceChangerHotParams)>,
        level: &QualityLevel,
    ) -> Option<(String, Option<VoiceSelection>)> {
        let h = view.map(|v| *v.1).unwrap_or_else(|| hot(0.0));
        if let Some((s, hp)) = view {
            let changed = (!run.selection.matches(s)).then(|| s.clone());
            run.on_view(VoiceTuning::of(s, hp), changed);
        }
        let failed = run.on_poll();
        run.apply_tuning(&h);
        let block = vec![0.1f32; BLOCK];
        if run.prime_ready(&block, level, &h, 10.0) {
            assert!(run.switch_in(level, &h));
        }
        run.push_history(&block);
        run.voices.active_mut().harness.push_and_convert(&block).unwrap();
        if let Some(old) = run.outgoing.take() {
            run.voices.retire(old);
        }
        failed
    }

    fn wait_for<T>(what: &str, mut f: impl FnMut() -> Option<T>) -> T {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            if let Some(v) = f() {
                return v;
            }
            assert!(Instant::now() < deadline, "timed out waiting for: {what}");
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    /// The incoming voice's per-voice values never reach the voice that still plays: while voice B
    /// (speaker 2, index 0.9, 0 st) loads, voice A keeps its +12 st, speaker 0 and index 0.3. B fails
    /// to load: A still plays as A, the failure names B's selection, and the selection goes back to A
    /// with A's values (what the UI then shows is what is heard); selecting B again retries the load.
    #[test]
    fn a_pending_or_failed_switch_never_retunes_the_playing_voice() {
        let a = voice_settings("a", 0, 0.3);
        let (h_a, h_b) = (hot(12.0), hot(0.0));
        let (voice_a, seen_a) = probe_voice(&a);
        // B is a real (tiny) voice folder, so its request builds and the load itself runs.
        let (root, slot_b, hubert) = test_models::write_slot("worker-failed-switch");
        let b = RvcParamsState {
            slot_dir: slot_b.to_string_lossy().into_owned(),
            hubert_path: hubert.to_string_lossy().into_owned(),
            pitch_extractor: "dio".into(),
            speaker_id: 2,
            index_rate: 0.9,
            ..RvcParamsState::default()
        };
        let release = Arc::new(Mutex::new(false));
        let loads = Arc::new(AtomicU64::new(0));
        let loader: VoiceLoader = {
            let (release, loads) = (release.clone(), loads.clone());
            Arc::new(move |_key, _req, _spec, _cancel| {
                loads.fetch_add(1, Ordering::SeqCst);
                while !*release.lock().unwrap() {
                    std::thread::sleep(Duration::from_millis(1));
                }
                Err("the voice is broken".into())
            })
        };
        let set = VoiceSet::with_loader(voice_a, 2, 0, loader);
        let mut run = run_with(set, &a, &h_a);
        let full = QualityLevel::FULL;
        for _ in 0..3 {
            assert!(tick(&mut run, Some((&a, &h_a)), &full).is_none());
        }
        assert_eq!(seen_a.lock().unwrap().calls.last().copied(), Some((12.0, 0.3, a.protect, 0)));

        // Select B (cold): it loads in the background while A keeps playing as A.
        for _ in 0..5 {
            assert!(tick(&mut run, Some((&b, &h_b)), &full).is_none());
            assert_eq!(seen_a.lock().unwrap().calls.last().copied(), Some((12.0, 0.3, a.protect, 0)), "B's values reached A");
        }
        wait_for("B's load to start", || (loads.load(Ordering::SeqCst) == 1).then_some(()));
        // B's request was built on the loader thread and its load runs; A still plays as A.
        tick(&mut run, Some((&b, &h_b)), &full);
        assert_eq!(seen_a.lock().unwrap().calls.last().copied(), Some((12.0, 0.3, a.protect, 0)));
        *release.lock().unwrap() = true;
        let (msg, failed) = wait_for("the failed load", || tick(&mut run, Some((&b, &h_b)), &full));
        assert!(msg.contains("the voice is broken"), "{msg}");
        assert_eq!(failed, Some(VoiceSelection::of(&b)));
        assert_eq!(run.voices.active().key, desired_key(&a, &base()), "A keeps playing");
        tick(&mut run, Some((&b, &h_b)), &full);
        assert_eq!(seen_a.lock().unwrap().calls.last().copied(), Some((12.0, 0.3, a.protect, 0)), "and stays A");

        // The selection goes back to A, with A's values.
        let st = VoiceChangerState::new();
        *lock(&st.mode) = VoiceChangerMode::Rvc;
        *lock(&st.rvc) = b.clone();
        *lock(&st.hot_params) = h_b;
        let failed = failed.unwrap();
        let reverted = st.revert_voice_selection(&failed, &run.active_selection, &run.tuning).expect("reverted");
        assert_eq!(reverted.rvc_slot_dir, a.slot_dir);
        assert_eq!((reverted.pitch_semitones, reverted.speaker_id, reverted.index_rate), (12.0, 0, 0.3));
        assert!(VoiceSelection::of(&lock(&st.rvc)).matches(&a));
        assert!(st.revert_voice_selection(&failed, &run.active_selection, &run.tuning).is_none(), "already moved on");
        let (a_now, h_now) = (lock(&st.rvc).clone(), *lock(&st.hot_params));
        tick(&mut run, Some((&a_now, &h_now)), &full);
        assert!(run.voices.loading().is_none());
        assert_eq!(seen_a.lock().unwrap().calls.last().copied(), Some((12.0, 0.3, a.protect, 0)));

        // Selecting B again retries it.
        let before = loads.load(Ordering::SeqCst);
        tick(&mut run, Some((&b, &h_b)), &full);
        wait_for("a second load", || (loads.load(Ordering::SeqCst) > before).then_some(()));
        wait_for("its failure", || tick(&mut run, Some((&b, &h_b)), &full));
        let _ = std::fs::remove_dir_all(&root);
    }

    /// While a switch is pending to another F0 method of the same voice, the per-voice values are
    /// that voice's own: the playing voice follows them.
    #[test]
    fn the_same_voice_keeps_following_its_values_during_a_switch() {
        let a = voice_settings("a", 0, 0.3);
        let (voice_a, seen_a) = probe_voice(&a);
        let loader: VoiceLoader = Arc::new(|_key, _req, _spec, cancel: &std::sync::atomic::AtomicBool| {
            while !cancel.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(1));
            }
            Err(LOAD_CANCELLED.to_string())
        });
        let mut run = run_with(VoiceSet::with_loader(voice_a, 2, 0, loader), &a, &hot(3.0));
        let other_f0 = RvcParamsState {
            pitch_extractor: "dio".into(),
            ..a.clone()
        };
        for _ in 0..3 {
            tick(&mut run, Some((&other_f0, &hot(5.0))), &QualityLevel::FULL);
        }
        assert_ne!(desired_key(&other_f0, &base()), run.voices.active().key, "another F0 method is another voice");
        assert_eq!(seen_a.lock().unwrap().calls.last().map(|c| c.0), Some(5.0));
        run.voices.abandon();
    }

    /// The governor's level reaches every voice that takes over, full quality included: a hot voice
    /// retired while the governor was degraded gets full quality back when it returns after the
    /// governor recovered (it used to keep retrieval off and the light F0 extractor for good).
    #[test]
    fn a_returning_voice_gets_the_current_quality_level() {
        let (a, b) = (voice_settings("a", 0, 0.3), voice_settings("b", 0, 0.3));
        let h = hot(0.0);
        let (voice_a, seen_a) = probe_voice(&a);
        let (voice_b, seen_b) = probe_voice(&b);
        let mut set = VoiceSet::new(voice_a, 3, 0);
        set.retire(voice_b);
        let mut run = run_with(set, &a, &h);
        let degraded = QualityLevel::at(3);
        // The governor steps A down under load.
        run.voices.active_mut().harness.apply_quality(&degraded);
        for _ in 0..5 {
            tick(&mut run, Some((&a, &h)), &degraded);
        }
        // Switch to B (hot): it takes over at the current level; A goes idle degraded.
        tick(&mut run, Some((&b, &h)), &degraded);
        assert_eq!(run.voices.active().key, desired_key(&b, &base()));
        assert_eq!(seen_b.lock().unwrap().level, Some(degraded));
        // The governor recovers on B.
        run.voices.active_mut().harness.apply_quality(&QualityLevel::FULL);
        tick(&mut run, Some((&b, &h)), &QualityLevel::FULL);
        // Back to A: full quality again.
        tick(&mut run, Some((&a, &h)), &QualityLevel::FULL);
        assert_eq!(run.voices.active().key, desired_key(&a, &base()));
        assert_eq!(seen_a.lock().unwrap().level, Some(QualityLevel::FULL), "A came back degraded");
    }

    /// The switch priming runs at once while the calls fit in the block (plan T9: the switch happens
    /// within one block), else it is spread over the next blocks, one extra call per tick, and the
    /// voice still gets exactly the blocks it needs before it takes over.
    #[test]
    fn switch_priming_spreads_when_one_tick_has_no_room() {
        assert_eq!(sync_prime_blocks(3, 0.0, 40.0, 1.0), 3, "unknown cost: at once");
        assert_eq!(sync_prime_blocks(3, 6.0, 40.0, 1.0), 3, "3 + 2 calls of 6 ms fit in 30 ms");
        assert_eq!(sync_prime_blocks(3, 32.0, 40.0, 1.0), 0, "slow: spread");
        assert_eq!(sync_prime_blocks(3, 9.0, 40.0, 1.0), 1);
        // A template voice: 40 ms model blocks on 10.67 ms device blocks (512 frames at 48 kHz).
        assert_eq!(sync_prime_blocks(7, 6.0, 10.67, 3.75), 7, "4090: 2 model calls + 2 fit in 30 ms: at once");
        assert_eq!(sync_prime_blocks(7, 32.0, 10.67, 3.75), 0, "DirectML at 32 ms a call: spread");
        assert_eq!(sync_prime_blocks(7, 9.0, 10.67, 3.75), 3, "room for one priming call (3 device blocks)");
        let (a, b) = (voice_settings("a", 0, 0.3), voice_settings("b", 0, 0.3));
        let h = hot(0.0);
        for (call_ms, ticks_to_switch) in [(0.5, 1usize), (15.0, 3)] {
            let (voice_a, _) = probe_voice(&a);
            let (voice_b, seen_b) = probe_voice(&b);
            let mut set = VoiceSet::new(voice_a, 3, 0);
            set.retire(voice_b);
            let mut run = run_with(set, &a, &h);
            for _ in 0..10 {
                tick(&mut run, Some((&a, &h)), &QualityLevel::FULL);
            }
            run.call_ms = call_ms;
            let need = SWITCH_PRIME_BLOCKS;
            let mut ticks = 0;
            while run.voices.active().key != desired_key(&b, &base()) {
                let before = seen_b.lock().unwrap().calls.len();
                tick(&mut run, Some((&b, &h)), &QualityLevel::FULL);
                ticks += 1;
                let calls = seen_b.lock().unwrap().calls.len() - before;
                if run.voices.active().key != desired_key(&b, &base()) {
                    assert!(calls <= 1, "a spread priming tick runs one extra call, not {calls}");
                }
                assert!(ticks < 20);
            }
            assert_eq!(ticks, ticks_to_switch, "call {call_ms} ms");
            // Primed blocks plus the switch block.
            assert_eq!(seen_b.lock().unwrap().calls.len(), need + 1, "call {call_ms} ms");
        }
    }

    /// Latency probes of quick consecutive switches can finish out of order: only the newest
    /// voice's probe stores its result.
    #[test]
    fn a_stale_latency_probe_never_overwrites_the_current_one() {
        let out = AtomicU64::new(0);
        out.store(1 << 32, Ordering::Relaxed);
        out.store(2 << 32, Ordering::Relaxed); // a newer switch started its probe
        assert!(!store_probe(&out, 1, 4_000), "the older voice's probe lands late");
        assert_eq!(out.load(Ordering::Relaxed) & 0xffff_ffff, 0);
        assert!(store_probe(&out, 2, 7_500));
        assert_eq!(out.load(Ordering::Relaxed), (2 << 32) | 7_500);
    }

    /// Selections and their per-voice values: a consistent view of the params is never half of an
    /// update (`update_params` holds the commit lock; the worker's view does not wait for it).
    #[test]
    fn the_voice_view_never_waits_and_never_sees_half_an_update() {
        let st = VoiceChangerState::new();
        let got = st.try_voice_view(|s, h| (s.slot_dir.clone(), h.pitch_semitones));
        assert!(got.is_some());
        let held = lock(&st.params_commit);
        let t = Instant::now();
        assert!(st.try_voice_view(|_, _| ()).is_none(), "an update in progress: no view");
        assert!(t.elapsed() < Duration::from_millis(50), "and no wait");
        drop(held);
        let mut p = VoiceChangerParams {
            rvc_slot_dir: "C:/voices/x".into(),
            pitch_semitones: 7.0,
            ..st.snapshot_params()
        };
        p.mode = VoiceChangerMode::Rvc;
        st.update_params(p);
        let (slot, pitch) = st.try_voice_view(|s, h| (s.slot_dir.clone(), h.pitch_semitones)).unwrap();
        assert_eq!((slot.as_str(), pitch), ("C:/voices/x", 7.0));
    }

    /// P3-E acceptance "with the denoiser off, e2e latency is unchanged (±2 ms)": the part of the
    /// chain P3-E changed on the input side is the capture (the hub's device-rate consumer instead
    /// of the voice changer's own stream). At real-time pacing (10 ms device blocks, 10 ms worker
    /// blocks) the time from a device callback to the worker holding the block it completes is
    /// measured through the hub path (`HubInput` → `CaptureBuffer`) and through the pre-P3-E path
    /// (callback → lock-free ring → unpark), and the hub path's median may be at most 2 ms worse. A
    /// timing measurement (about 4 s), run with `--ignored`; it prints both.
    #[test]
    #[ignore = "timing measurement at real-time pacing"]
    fn the_hub_capture_path_adds_no_latency() {
        const BLOCKS: usize = 200;
        let period = Duration::from_millis(10);
        let pct = |v: &mut Vec<f64>, p: f64| {
            v.sort_by(f64::total_cmp);
            v[((v.len() - 1) as f64 * p).round() as usize]
        };
        // The hub path.
        let mic = ScriptedMic::new(STEREO_48K);
        let hub = test_hub(&mic);
        let pushed: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let hub_lat = std::thread::scope(|s| {
            let worker = s.spawn(|| {
                let mut capture = HubInput::attach(&hub, &target(STEREO_48K), Box::new(|| false)).unwrap();
                let mut cap = CaptureBuffer::new(48_000, 2, BLOCK, 300.0);
                let mut out = Vec::new();
                let mut lat = Vec::new();
                while lat.len() < BLOCKS {
                    capture.wait(Duration::from_millis(100));
                    capture.pump(&mut cap, now(), BLOCK);
                    cap.wake();
                    while cap.take_block(&mut out, BLOCK) {
                        let t = now();
                        if let Some(&p) = pushed.lock().unwrap().get(lat.len()) {
                            lat.push((t - p) as f64 / 1e6);
                        }
                    }
                }
                lat
            });
            wait_until("stream live", || mic.live());
            std::thread::sleep(Duration::from_millis(50));
            let start = Instant::now();
            for k in 0..BLOCKS {
                let due = start + period * k as u32;
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
                pushed.lock().unwrap().push(now());
                assert!(mic.push(&vec![0.1f32; BLOCK * 2]));
            }
            worker.join().unwrap()
        });
        // The pre-P3-E path: the callback pushes into the ring and unparks the worker.
        let (mut producer, mut consumer) = rtrb::RingBuffer::<f32>::new(BLOCK * 64);
        let pushed: Arc<Mutex<Vec<i64>>> = Arc::new(Mutex::new(Vec::new()));
        let direct_lat = std::thread::scope(|s| {
            let worker = s.spawn(|| {
                let mut lat = Vec::new();
                let mut have = 0usize;
                while lat.len() < BLOCKS {
                    std::thread::park_timeout(Duration::from_millis(100));
                    let n = consumer.slots();
                    if let Ok(chunk) = consumer.read_chunk(n) {
                        chunk.commit_all();
                    }
                    have += n;
                    while have >= BLOCK {
                        have -= BLOCK;
                        let t = now();
                        if let Some(&p) = pushed.lock().unwrap().get(lat.len()) {
                            lat.push((t - p) as f64 / 1e6);
                        }
                    }
                }
                lat
            });
            let handle = worker.thread().clone();
            std::thread::sleep(Duration::from_millis(50));
            let start = Instant::now();
            for k in 0..BLOCKS {
                let due = start + period * k as u32;
                if let Some(wait) = due.checked_duration_since(Instant::now()) {
                    std::thread::sleep(wait);
                }
                pushed.lock().unwrap().push(now());
                assert_eq!(push_capture(&mut producer, &[0.1f32; BLOCK], 1), 0);
                handle.unpark();
            }
            worker.join().unwrap()
        });
        let (mut hub_lat, mut direct_lat) = (hub_lat, direct_lat);
        let (hub_p50, hub_p95) = (pct(&mut hub_lat, 0.5), pct(&mut hub_lat, 0.95));
        let (dir_p50, dir_p95) = (pct(&mut direct_lat, 0.5), pct(&mut direct_lat, 0.95));
        eprintln!("capture to worker block: hub p50 {hub_p50:.3} ms p95 {hub_p95:.3} ms; pre-P3-E path p50 {dir_p50:.3} ms p95 {dir_p95:.3} ms");
        assert!(hub_p50 <= dir_p50 + 2.0, "the hub path adds {:.3} ms", hub_p50 - dir_p50);
    }
}
