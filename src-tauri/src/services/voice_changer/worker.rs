//! The voice-changer worker: model load, audio streams, the capture → convert → playback loop,
//! run generations and `voice_changer:error` emission.
//!
//! Real-time structure (plan "Real-time hygiene"):
//! - Models and bridges load **before** any audio stream opens, so the first ticks never run on a
//!   backlog captured during a multi-second model load.
//! - The capture stream opens in the device's native sample format through
//!   `pcm_convert::build_input_stream_f32` (allocation-free f32 conversion, shared with STT); the
//!   callback copies whole frames into a lock-free ring and unparks the worker; the worker parks when it has less than a block (no
//!   sleep-poll). Capture, output and monitor callbacks register with MMCSS "Pro Audio"; the worker
//!   never does, it gets the `worker_priority` policy instead.
//! - The worker drains the ring into its backlog on every wake and enforces the latency budget
//!   (oldest audio dropped, see `budget.rs`) before taking a block, in every mode.
//! - Callbacks never lock, allocate or log: stream errors and MMCSS results go through atomics and
//!   the worker logs them.
//! - A block whose conversion fails (ONNX error, sidecar timeout / bad reply, RVC error) is muted
//!   with click-free edges (`fallback.rs`), never replaced by the unconverted microphone.
//! - A worker that stops by itself (a saved device is gone, a model or the sidecar does not load,
//!   a device disconnects, a panic) always tells the user through `voice_changer:error` before
//!   `voice_changer:stopped` ([`WorkerError`]). A missing monitor device is not fatal: the run goes
//!   on without the monitor and says so.

use super::aux_feed::AuxMixer;
use super::budget::{capture_ring_capacity, push_capture, InputBacklog};
use super::devices::{open_saved_device, DeviceRole, OpenedDevice};
use super::fallback::MuteConceal;
use super::metrics::{frames_to_ms, ProcessTimeWindow};
use super::params::{
    clamp_output_buffer_ms, effective_pitch_extractor, effective_rvc_ep, normalize_execution_provider,
    resolve_rvc_paths, VoiceChangerMode,
};
use super::playback::{
    adapt_channels_interleaved_into, OutputPlayback, RingPlayout, OUT_RING_MAX_SAMPLES, OUT_RING_MIN_SAMPLES,
};
use super::resample::PolyphaseResampler;
use super::state::{lock, MmcssFlag, MmcssStatus, VoiceChangerState};
use crate::services::diagnostics::{ep_outcomes, EpOutcome};
use crate::services::neural_audio::chunk_processor::{apply_formant_interleaved, FormantLpState};
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::{
    clamp_chunk_frames, ClassicPhaseVocoderProcessor, DioPitchExtractor, FcpePitchExtractor, OnnxVoiceBridge,
    PitchExtractor, RmvpePitchExtractor, RvcModel, RvcPresetProfile, SidecarVoiceBridge, VcStreamHarness,
    VoiceChunkPipeline, VoiceConversionModel,
};
use crate::services::pcm_convert::build_input_stream_f32;
use crate::services::rt_thread::{self, WorkerPriority};
use cpal::traits::{DeviceTrait, StreamTrait};
use cpal::{SampleFormat, StreamError};
use rtrb::RingBuffer;
use serde::Serialize;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::Receiver;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use tauri::{AppHandle, Emitter, Manager, Runtime};

/// Throttle repeated `voice_changer:error` emissions while conversion keeps failing each chunk.
const VC_ERROR_EMIT_THROTTLE: Duration = Duration::from_secs(2);
/// Longest park without a capture callback (stop requests also unpark the worker at once).
const PARK_TIMEOUT: Duration = Duration::from_millis(100);
/// When the MMCSS status of the audio callbacks is logged (once per run).
const MMCSS_LOG_AFTER: Duration = Duration::from_secs(2);
/// Budget drops are summarized in the log at most this often.
const DROP_LOG_EVERY: Duration = Duration::from_secs(5);

#[derive(Clone, Serialize)]
struct VoiceChangerErrorPayload {
    code: String,
    message: String,
}

/// Emit one `voice_changer:error` event. Codes while running: `onnx_infer`, `sidecar_length`,
/// `sidecar_exchange`, `rvc_convert` (the block was muted), `rvc_warning`, `monitor_unavailable`
/// (the run continues without the monitor). Codes of a worker that stopped ([`WorkerError`]):
/// `device`, `sidecar_connect`, `onnx_load`, `rvc_load`, `worker_error`, `worker_panic`.
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
    is_running: Arc<AtomicBool>,
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

/// Device-lost flags set by cpal error callbacks (real-time safe), logged by the worker.
#[derive(Default)]
struct StreamErrors {
    device_lost: AtomicBool,
    backend_errors: AtomicU64,
}

impl StreamErrors {
    fn note(&self, e: StreamError) {
        match e {
            StreamError::DeviceNotAvailable => self.device_lost.store(true, Ordering::Relaxed),
            _ => {
                self.backend_errors.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

/// Everything the capture callback owns.
struct CaptureCallback {
    producer: rtrb::Producer<f32>,
    ch: usize,
    is_running: Arc<AtomicBool>,
    in_burst: Arc<AtomicUsize>,
    overflow: Arc<AtomicU64>,
    worker: thread::Thread,
    mmcss: Arc<MmcssStatus>,
}

impl CaptureCallback {
    /// One capture buffer (already f32): MMCSS once, whole frames into the ring, wake the worker.
    #[inline]
    fn on_data(&mut self, data: &[f32]) {
        self.mmcss.capture.record(rt_thread::register_mmcss_pro_audio_once());
        if !self.is_running.load(Ordering::Acquire) {
            return;
        }
        self.in_burst.store(data.len(), Ordering::Relaxed);
        let lost = push_capture(&mut self.producer, data, self.ch);
        if lost > 0 {
            self.overflow.fetch_add(lost as u64, Ordering::Relaxed);
        }
        self.worker.unpark();
    }
}

/// Open the capture stream in the device's native sample format (every format cpal defines;
/// converted to f32 by `pcm_convert` without allocating).
fn build_capture_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut cb: CaptureCallback,
    errors: Arc<StreamErrors>,
) -> Result<cpal::Stream, String> {
    build_input_stream_f32(device, config, move |data: &[f32]| cb.on_data(data), move |e| errors.note(e))
}

/// Open an f32 output stream playing `playout`, registering its callback thread with MMCSS.
fn build_output_stream(
    device: &cpal::Device,
    config: &cpal::SupportedStreamConfig,
    mut playout: RingPlayout,
    mmcss: Arc<MmcssStatus>,
    which: fn(&MmcssStatus) -> &MmcssFlag,
    errors: Arc<StreamErrors>,
) -> Result<cpal::Stream, String> {
    device
        .build_output_stream(
            &config.config(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                which(&mmcss).record(rt_thread::register_mmcss_pro_audio_once());
                playout.fill(data);
            },
            move |e| errors.note(e),
            None,
        )
        .map_err(|e| format!("Failed to open the output stream: {e}"))
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
        "vc.pitch" => "F0 model",
        "vc.onnx" => "ONNX model",
        other => other,
    }
}

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

/// The optional monitor output: its stream, queue, ring and the stateful rate converter from the
/// main output rate.
struct Monitor {
    _stream: cpal::Stream,
    playback: OutputPlayback,
    producer: rtrb::Producer<f32>,
    ring_cap: usize,
    rate: u32,
    ch: usize,
    resampler: PolyphaseResampler,
    errors: Arc<StreamErrors>,
}

fn ms_to_samples(ms: f64, rate: u32, ch: usize) -> usize {
    ((f64::from(rate)) * ms / 1000.0).ceil() as usize * ch
}

fn ring_capacity(prefill: usize, max_queue: usize) -> usize {
    max_queue
        .saturating_mul(2)
        .clamp(OUT_RING_MIN_SAMPLES, OUT_RING_MAX_SAMPLES)
        .max(prefill.saturating_add(1))
}

/// Open a saved device, logging when it was found under a renumbered name.
fn open_device(host: &cpal::Host, role: DeviceRole, saved: &str) -> Result<OpenedDevice, String> {
    let opened = open_saved_device(host, role, saved)?;
    if let Some(saved) = &opened.renamed_from {
        tracing::info!(
            "voice changer: {role:?} device '{saved}' was renumbered by Windows; using '{}'",
            opened.name
        );
    }
    Ok(opened)
}

/// The monitor device and its config, or `None` (monitor off, or unusable: the run goes on
/// without it and the user is told once).
fn open_monitor<R: Runtime>(
    app: &AppHandle<R>,
    generation: &RunGeneration,
    host: &cpal::Host,
    saved: &str,
) -> Option<(cpal::Device, cpal::SupportedStreamConfig)> {
    if saved.trim().is_empty() {
        return None;
    }
    let opened = open_device(host, DeviceRole::Monitor, saved).and_then(|d| {
        let config = d
            .device
            .default_output_config()
            .map_err(|e| format!("The monitor device '{}' cannot be opened: {e}", d.name))?;
        if !matches!(config.sample_format(), SampleFormat::F32) {
            return Err(format!(
                "The monitor device '{}' uses {:?} samples (32-bit float is needed)",
                d.name,
                config.sample_format()
            ));
        }
        Ok((d.device, config))
    });
    match opened {
        Ok(pair) => Some(pair),
        Err(e) => {
            warn_monitor_unavailable(app, generation, &e);
            None
        }
    }
}

fn warn_monitor_unavailable<R: Runtime>(app: &AppHandle<R>, generation: &RunGeneration, why: &str) {
    let message = format!("{}. The voice changer runs without the monitor.", why.trim_end_matches('.'));
    tracing::warn!("voice changer: {message}");
    if generation.is_current() {
        emit_vc_error(app, "monitor_unavailable", message);
    }
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

    let input = open_device(&host, DeviceRole::Input, input_device_name.as_deref().unwrap_or(""))
        .map_err(WorkerError::device)?;
    let output = open_device(&host, DeviceRole::Output, &output_device_name).map_err(WorkerError::device)?;
    let input_config = input
        .device
        .default_input_config()
        .map_err(|e| WorkerError::device(format!("The input device '{}' cannot be opened: {e}", input.name)))?;
    let output_config = output
        .device
        .default_output_config()
        .map_err(|e| WorkerError::device(format!("The output device '{}' cannot be opened: {e}", output.name)))?;

    let r_in = input_config.sample_rate().0;
    let r_out = output_config.sample_rate().0;
    let ch_in = (input_config.channels() as usize).max(1);
    let ch_out = (output_config.channels() as usize).max(1);

    if !matches!(output_config.sample_format(), SampleFormat::F32) {
        return Err(WorkerError::device(format!(
            "The output device '{}' uses {:?} samples (32-bit float is needed)",
            output.name,
            output_config.sample_format()
        )));
    }
    let monitor_target = open_monitor(&app, &generation, &host, &monitor_output_device);

    tracing::info!(
        "voice changer: input {} Hz {} ch {:?}, output {} Hz {} ch",
        r_in,
        ch_in,
        input_config.sample_format(),
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
    let chunk_frames = clamp_chunk_frames(h_init.chunk_frames);
    let need = chunk_frames * ch_in;
    if need == 0 {
        return Err("Invalid chunk size or channel count.".into());
    }

    // ---- Engines: every model and bridge loads before any audio stream opens. ----
    let mode = lock(&st.mode).clone();
    let load_started_ms = unix_ms_now();
    let mut classic = ClassicPhaseVocoderProcessor::new(
        r_in,
        r_out,
        ch_in,
        ch_out,
        h_init.vocoder_window_ms,
        chunk_frames,
    )?;

    // ONNX Runtime's pool threads start with the worker priority policy (`auto` on the CPU EP and
    // `protect_game` are BELOW_NORMAL), so its intra-op pool never does the CPU work at NORMAL
    // priority beside the game. `protect_game` also caps the pool at 2 threads (the calling worker
    // plus one pool thread). Sessions are built once per run, so a live switch applies from the
    // next start.
    let ort_priority = h_init.worker_priority;
    let ort_intra_threads = (ort_priority == WorkerPriority::ProtectGame).then_some(2);

    let budget_timeout = |ms: f64| Duration::from_secs_f64(ms.max(1.0) / 1000.0);
    let mut sidecar_timeout_ms = h_init.latency_budget_ms;
    let mut sidecar = if mode == VoiceChangerMode::Sidecar {
        let url = lock(&st.sidecar_ws_url).clone();
        Some(
            SidecarVoiceBridge::connect(&url, r_in, ch_in, chunk_frames, budget_timeout(sidecar_timeout_ms))
                .map_err(|e| WorkerError::new("sidecar_connect", e))?,
        )
    } else {
        None
    };

    let onnx_ep = normalize_execution_provider(&lock(&st.onnx_execution_provider));
    let mut onnx_vc = if mode == VoiceChangerMode::Onnx {
        let path = lock(&st.onnx_model_path).clone();
        let opts = VcSessionOptions {
            intra_threads: ort_intra_threads,
            thread_priority: Some(ort_priority),
            ..VcSessionOptions::default()
        };
        Some(
            OnnxVoiceBridge::new_with_options(path.trim(), need, ch_in, &onnx_ep, &opts)
                .map_err(|e| WorkerError::new("onnx_load", e))?,
        )
    } else {
        None
    };

    let mut rvc_ep = String::new();
    let mut rvc_harness = if mode == VoiceChangerMode::Rvc {
        // Every RVC load failure stops the worker as `rvc_load` (the epilogue shows it once,
        // unless a newer start has superseded this worker).
        let rvc_load_err = |e: String| WorkerError::new("rvc_load", e);
        let mut settings = lock(&st.rvc).clone();
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
        let ep_hint = effective_rvc_ep(&settings.execution_provider, preset).to_string();
        let session_opts = VcSessionOptions {
            cuda_device_index: settings.gpu_device_index,
            streamer_profile: matches!(preset, RvcPresetProfile::Streamer),
            intra_threads: ort_intra_threads,
            thread_priority: Some(ort_priority),
            // The arena cap is a CUDA EP option; DirectML/CPU ignore it.
            vram_cap_mb: if ep_hint == "cuda" {
                Some(settings.vram_cap_mb)
            } else {
                None
            },
            // The voice changer's rows stay the plain `vc` / `vc.*` keys.
            component_prefix: None,
        };
        if matches!(preset, RvcPresetProfile::Streamer)
            && crate::services::neural_audio::vc::supports_int8_content_encoder()
        {
            tracing::info!("voice changer: CPU supports the int8 content-encoder path (gated for Phase 3 quant)");
        }

        // Same rule as the inspector (which greys rmvpe out on the CPU EP) and the start
        // preflight: a stale GPU-only choice falls back to fcpe, and the user is told instead of
        // it happening silently.
        let (pitch_name, fell_back) = effective_pitch_extractor(&settings.pitch_extractor, &ep_hint);
        if fell_back {
            let msg = format!(
                "{} is GPU-only in Sigil; using fcpe on the CPU execution provider",
                settings.pitch_extractor.trim().to_lowercase()
            );
            tracing::warn!("voice changer: {msg}");
            if generation.is_current() {
                emit_vc_error(&app, "rvc_warning", msg);
            }
        }

        let pitch: Box<dyn PitchExtractor> = match pitch_name.as_str() {
            "rmvpe" => Box::new(
                RmvpePitchExtractor::load(&settings.rmvpe_path, &ep_hint, &session_opts).map_err(&rvc_load_err)?,
            ),
            "dio" => Box::new(DioPitchExtractor::new()),
            _ => Box::new(
                FcpePitchExtractor::load(&settings.fcpe_path, &ep_hint, &session_opts).map_err(&rvc_load_err)?,
            ),
        };

        let model = RvcModel::load(
            Path::new(&settings.slot_dir),
            &settings.hubert_path,
            pitch,
            &ep_hint,
            &session_opts,
            settings.silence_skip,
        )
        .map_err(&rvc_load_err)?;
        // Non-fatal load issues (manifest inferred, index unreadable, ...) surface as warnings.
        for warning in model.warnings() {
            tracing::warn!("voice changer RVC: {warning}");
            if generation.is_current() {
                emit_vc_error(&app, "rvc_warning", warning.clone());
            }
        }
        // No separate warm-up here: `VcStreamHarness::new` warms the model once at the live
        // steady-state input shape (`warmup_at`), so GPU EPs pick kernels for the shape the first
        // spoken tick uses instead of an unused one.
        let model_out_sr = model.output_sample_rate();
        let extra_samples = ((settings.extra_context_ms / 1000.0) * 16_000.0).round().max(0.0) as usize;
        let crossfade_samples = ((settings.crossfade_ms / 1000.0) * model_out_sr as f64).round().max(0.0) as usize;

        let mut harness = VcStreamHarness::new(
            Box::new(model),
            r_in,
            r_out,
            ch_in,
            ch_out,
            extra_samples,
            crossfade_samples,
            chunk_frames,
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
        rvc_ep = ep_hint;
        Some(harness)
    } else {
        None
    };
    if stop.requested() {
        return Ok(());
    }

    // ---- The EP actually in use, and the worker priority that depends on it. ----
    let run_ep = match mode {
        VoiceChangerMode::Rvc => run_ep_outcome(&["vc", "vc.content", "vc.pitch"], &rvc_ep, load_started_ms),
        VoiceChangerMode::Onnx => run_ep_outcome(&["vc.onnx", "vc"], &onnx_ep, load_started_ms),
        VoiceChangerMode::Classic | VoiceChangerMode::Sidecar => None,
    };
    let gpu_ep = match (&run_ep, &mode) {
        (Some(o), _) => o.is_gpu(),
        (None, VoiceChangerMode::Rvc) => rvc_ep != "cpu",
        (None, VoiceChangerMode::Onnx) => onnx_ep == "cuda" || onnx_ep == "dml",
        _ => false,
    };
    if let Some(o) = &run_ep {
        match &o.reason {
            Some(why) if o.fell_back() => tracing::warn!(
                "voice changer: requested {} but running on {}: {why}",
                o.requested,
                o.active
            ),
            _ => tracing::info!("voice changer: running on the {} execution provider", o.active),
        }
    }
    // Resource governor: report the device the models landed on (DSP-only and sidecar modes run
    // no local model, so `false`). Without a recorded outcome, resources infers it from the
    // `vc` / `vc.*` rows. A superseded worker never reports over its successor.
    let report_gpu = match mode {
        VoiceChangerMode::Rvc | VoiceChangerMode::Onnx => run_ep.as_ref().map(EpOutcome::is_gpu),
        VoiceChangerMode::Classic | VoiceChangerMode::Sidecar => Some(false),
    };
    if let Some(on_gpu) = report_gpu.filter(|_| generation.is_current()) {
        crate::services::resources::set_vc_gpu(on_gpu);
    }
    *lock(&st.run_ep) = run_ep;
    let mut applied_priority = h_init.worker_priority;
    apply_worker_priority(applied_priority, gpu_ep);

    // ---- Rings and queues (sized once, before the streams). ----
    let (prefill_ms, max_queue_ms) = clamp_output_buffer_ms(h_init.output_prefill_ms, h_init.output_max_queue_ms);
    let prefill_samples = ms_to_samples(prefill_ms, r_out, ch_out);
    let max_queue_samples = ms_to_samples(max_queue_ms, r_out, ch_out);

    let in_ring_cap = capture_ring_capacity(r_in, ch_in, chunk_frames);
    let (audio_producer, mut audio_consumer) = RingBuffer::<f32>::new(in_ring_cap);
    let mut backlog = InputBacklog::new(ch_in, r_in, h_init.latency_budget_ms, in_ring_cap + in_ring_cap / 2 + need);
    let overflow = Arc::new(AtomicU64::new(0));

    let out_ring_cap = ring_capacity(prefill_samples, max_queue_samples);
    let (mut out_producer, out_consumer) = RingBuffer::<f32>::new(out_ring_cap);
    let out_callback_len = Arc::new(AtomicUsize::new(0));
    // Interleaved samples of the latest capture callback: output top-ups only happen when capture
    // delivers, so each playback's ring must also bridge one capture period.
    let in_burst_len = Arc::new(AtomicUsize::new(0));
    let mut playback = OutputPlayback::new(prefill_samples, max_queue_samples, ch_out, out_callback_len.clone())
        .with_capture_burst(in_burst_len.clone(), ch_in, r_in, r_out);

    // ---- Streams. ----
    rt_thread::preload();
    let input_errors = Arc::new(StreamErrors::default());
    let output_errors = Arc::new(StreamErrors::default());
    let input_stream = build_capture_stream(
        &input.device,
        &input_config,
        CaptureCallback {
            producer: audio_producer,
            ch: ch_in,
            is_running: st.is_running.clone(),
            in_burst: in_burst_len.clone(),
            overflow: overflow.clone(),
            worker: thread::current(),
            mmcss: st.mmcss.clone(),
        },
        input_errors.clone(),
    )
    .map_err(|e| WorkerError::device(format!("The input device '{}' could not be started: {e}", input.name)))?;
    let out_playout = RingPlayout::new(out_consumer, ch_out, out_callback_len)
        .with_underrun_counter(st.metrics_underruns.clone());
    let output_stream = build_output_stream(
        &output.device,
        &output_config,
        out_playout,
        st.mmcss.clone(),
        |m| &m.output,
        output_errors.clone(),
    )
    .map_err(|e| WorkerError::device(format!("The output device '{}' could not be started: {e}", output.name)))?;

    let mut monitor = match monitor_target {
        None => None,
        Some((dev, mc)) => {
            let r_mon = mc.sample_rate().0;
            let ch_mon = (mc.channels() as usize).max(1);
            let prefill_m = ms_to_samples(prefill_ms, r_mon, ch_mon);
            let max_q_m = ms_to_samples(max_queue_ms, r_mon, ch_mon);
            let ring_cap = ring_capacity(prefill_m, max_q_m);
            let (producer, consumer) = RingBuffer::<f32>::new(ring_cap);
            let callback_len = Arc::new(AtomicUsize::new(0));
            let playback = OutputPlayback::new(prefill_m, max_q_m, ch_mon, callback_len.clone())
                .with_capture_burst(in_burst_len.clone(), ch_in, r_in, r_mon);
            let errors = Arc::new(StreamErrors::default());
            match build_output_stream(
                &dev,
                &mc,
                RingPlayout::new(consumer, ch_mon, callback_len),
                st.mmcss.clone(),
                |m| &m.monitor,
                errors.clone(),
            ) {
                Ok(stream) => Some(Monitor {
                    _stream: stream,
                    playback,
                    producer,
                    ring_cap,
                    rate: r_mon,
                    ch: ch_mon,
                    resampler: PolyphaseResampler::new(r_out, r_mon, ch_mon),
                    errors,
                }),
                Err(e) => {
                    warn_monitor_unavailable(&app, &generation, &e);
                    None
                }
            }
        }
    };

    input_stream
        .play()
        .map_err(|e| WorkerError::device(format!("The input device '{}' could not be started: {e}", input.name)))?;
    output_stream
        .play()
        .map_err(|e| WorkerError::device(format!("The output device '{}' could not be started: {e}", output.name)))?;
    if let Some(Err(e)) = monitor.as_ref().map(|m| m._stream.play()) {
        warn_monitor_unavailable(&app, &generation, &format!("The monitor device could not be started: {e}"));
        monitor = None;
    }
    let streams_started = Instant::now();

    // Formant tilt for RVC output (the model has no formant control), per output channel at r_out.
    let mut rvc_formant_states = vec![FormantLpState::default(); ch_out];
    // A block whose conversion fails is muted (never the dry microphone): ONNX / sidecar blocks
    // before the post chain (capture format), RVC ticks at the output.
    let mut conceal_in = MuteConceal::new(ch_in, r_in);
    let mut conceal_out = MuteConceal::new(ch_out, r_out);
    let mut concealed = Vec::with_capacity(need);

    let mut work_chunk = Vec::with_capacity(need);
    let mut scratch_mon = Vec::with_capacity(need * 2);
    let mut scratch_adapt = Vec::with_capacity(need * 2);
    let mut scratch_resample = Vec::with_capacity(need * 4);
    let mut aux_mixer = AuxMixer::new(r_in, ch_in);
    let mut p99 = ProcessTimeWindow::new();
    let mut mmcss_logged = false;
    let mut monitor_lost_logged = false;
    let mut drop_log_at = Instant::now();
    let (mut logged_dropped, mut logged_events) = (0u64, 0u64);

    loop {
        if stop.requested() {
            break;
        }
        if input_errors.device_lost.load(Ordering::Relaxed) {
            return Err(WorkerError::device(format!(
                "The input device '{}' was disconnected; the voice changer stopped.",
                input.name
            )));
        }
        if output_errors.device_lost.load(Ordering::Relaxed) {
            return Err(WorkerError::device(format!(
                "The output device '{}' was disconnected; the voice changer stopped.",
                output.name
            )));
        }

        let h = *lock(&st.hot_params);
        let (pf_ms, mq_ms) = clamp_output_buffer_ms(h.output_prefill_ms, h.output_max_queue_ms);
        let ring_used = out_ring_cap.saturating_sub(out_producer.slots());
        playback.update_limits(
            ms_to_samples(pf_ms, r_out, ch_out),
            ms_to_samples(mq_ms, r_out, ch_out),
            ring_used,
        );
        if let Some(m) = monitor.as_mut() {
            let used = m.ring_cap.saturating_sub(m.producer.slots());
            m.playback.update_limits(
                ms_to_samples(pf_ms, m.rate, m.ch),
                ms_to_samples(mq_ms, m.rate, m.ch),
                used,
            );
        }
        backlog.set_budget_ms(h.latency_budget_ms);
        if let Some(b) = sidecar.as_mut() {
            if h.latency_budget_ms != sidecar_timeout_ms {
                sidecar_timeout_ms = h.latency_budget_ms;
                b.set_exchange_timeout(budget_timeout(sidecar_timeout_ms));
            }
        }
        if h.worker_priority != applied_priority {
            applied_priority = h.worker_priority;
            apply_worker_priority(applied_priority, gpu_ep);
            let uses_ort = onnx_vc.is_some() || rvc_harness.is_some();
            if uses_ort && applied_priority != ort_priority {
                tracing::info!(
                    "voice changer: the ONNX Runtime thread policy for {} applies from the next start",
                    applied_priority.as_str()
                );
            }
        }

        // Everything captured so far, then the budget (oldest audio dropped, every mode).
        backlog.ingest(&mut audio_consumer, &overflow);
        backlog.enforce(chunk_frames);
        let waiting = backlog.frames() + audio_consumer.slots() / ch_in;
        st.metrics_backlog_frames
            .store(waiting.min(u32::MAX as usize) as u32, Ordering::Relaxed);
        st.metrics_dropped_frames.store(backlog.dropped_frames(), Ordering::Relaxed);
        if backlog.dropped_frames() > logged_dropped && drop_log_at.elapsed() >= DROP_LOG_EVERY {
            tracing::info!(
                "voice changer: latency budget ({:.0} ms) dropped {:.0} ms of stale input in {} step(s) since the last report",
                h.latency_budget_ms,
                frames_to_ms((backlog.dropped_frames() - logged_dropped) as f64, r_in),
                backlog.drop_events() - logged_events
            );
            logged_dropped = backlog.dropped_frames();
            logged_events = backlog.drop_events();
            drop_log_at = Instant::now();
        }
        if !mmcss_logged && streams_started.elapsed() >= MMCSS_LOG_AFTER {
            mmcss_logged = true;
            tracing::info!(
                "voice changer MMCSS Pro Audio: capture {}, output {}, monitor {}",
                mmcss_label(&st.mmcss.capture),
                mmcss_label(&st.mmcss.output),
                if monitor.is_some() { mmcss_label(&st.mmcss.monitor) } else { "off" }
            );
            let backend = input_errors.backend_errors.load(Ordering::Relaxed)
                + output_errors.backend_errors.load(Ordering::Relaxed);
            if backend > 0 {
                tracing::warn!("voice changer: {backend} audio backend error(s) reported by the devices");
            }
        }
        if !monitor_lost_logged && monitor.as_ref().is_some_and(|m| m.errors.device_lost.load(Ordering::Relaxed)) {
            monitor_lost_logged = true;
            tracing::warn!("voice changer: the monitor device was disconnected; the main output continues");
        }

        if !backlog.take_block(&mut work_chunk, chunk_frames) {
            // Woken by the next capture callback (or a stop request).
            thread::park_timeout(PARK_TIMEOUT);
            continue;
        }

        let pitch = h.pitch_semitones;
        let formant = h.formant_shift;
        let window_ms = h.vocoder_window_ms;
        let oversample = h.vocoder_oversample;
        let g_in = h.input_gain;
        if (g_in - 1.0).abs() > 1e-6 {
            for s in work_chunk.iter_mut() {
                *s = (*s * g_in).clamp(-1.0, 1.0);
            }
        }
        st.aux_feed.mix_into(&mut work_chunk, &mut aux_mixer, h.aux_mic_gate);

        let t0 = Instant::now();
        let processed = if let Some(ref mut ob) = onnx_vc {
            let converted: &[f32] = match ob.infer(&work_chunk) {
                Ok(mut v) => {
                    conceal_in.pass(&mut v);
                    concealed = v;
                    &concealed
                }
                Err(e) => {
                    maybe_emit_vc_error(&app, "onnx_infer", e, &mut last_vc_error_emit);
                    conceal_in.mute_into(&mut concealed, chunk_frames);
                    &concealed
                }
            };
            VoiceChunkPipeline::process_chunk(&mut classic, converted, pitch, formant, window_ms, oversample)?
        } else if let Some(ref mut bridge) = sidecar {
            let neural: &[f32] = match bridge.exchange(&work_chunk) {
                Ok(mut v) if v.len() == need => {
                    conceal_in.pass(&mut v);
                    concealed = v;
                    &concealed
                }
                Ok(v) => {
                    maybe_emit_vc_error(
                        &app,
                        "sidecar_length",
                        format!("expected {} samples, got {}", need, v.len()),
                        &mut last_vc_error_emit,
                    );
                    conceal_in.mute_into(&mut concealed, chunk_frames);
                    &concealed
                }
                Err(e) => {
                    maybe_emit_vc_error(&app, "sidecar_exchange", e, &mut last_vc_error_emit);
                    conceal_in.mute_into(&mut concealed, chunk_frames);
                    &concealed
                }
            };
            VoiceChunkPipeline::process_chunk(&mut classic, neural, pitch, formant, window_ms, oversample)?
        } else if let Some(ref mut harness) = rvc_harness {
            let (index_rate, protect, speaker_id) = {
                let s = lock(&st.rvc);
                (s.index_rate, s.protect, s.speaker_id)
            };
            // Pitch is applied inside the RVC model; skip classic phase vocoder.
            harness.set_controls(pitch, index_rate, protect, speaker_id);
            let mut out = match harness.push_and_convert(&work_chunk) {
                Ok(mut v) => {
                    conceal_out.pass(&mut v);
                    v
                }
                Err(e) => {
                    maybe_emit_vc_error(&app, "rvc_convert", e, &mut last_vc_error_emit);
                    // Exactly the device frames the harness owed for this tick, as on success, so
                    // a failed tick neither shortens nor lengthens the output stream; muted, never
                    // the dry microphone.
                    let mut muted = Vec::new();
                    conceal_out.mute_into(&mut muted, harness.last_owed_frames());
                    muted
                }
            };
            // The model has no formant control: apply the classic spectral tilt to its output
            // (same per-channel filter state on the fallback tick, so the tilt stays continuous).
            apply_formant_interleaved(&mut out, ch_out, formant, &mut rvc_formant_states, r_out as f64);
            out
        } else {
            VoiceChunkPipeline::process_chunk(&mut classic, &work_chunk, pitch, formant, window_ms, oversample)?
        };

        if let Some(m) = monitor.as_mut() {
            // Always fed (silence at gain 0), so the stateful resampler and the monitor queue
            // keep an unbroken timeline.
            let g_mon = h.monitor_gain;
            let mon_src: &[f32] = if (g_mon - 1.0).abs() > 1e-6 {
                scratch_mon.clear();
                scratch_mon.extend(processed.iter().map(|&s| (s * g_mon).clamp(-1.0, 1.0)));
                &scratch_mon
            } else {
                &processed
            };
            adapt_channels_interleaved_into(mon_src, ch_out, m.ch, &mut scratch_adapt);
            scratch_resample.clear();
            m.resampler.process(&scratch_adapt, &mut scratch_resample);
            if !scratch_resample.is_empty() {
                let used = m.ring_cap.saturating_sub(m.producer.slots());
                m.playback.push_interleaved(&scratch_resample, false, 0.0, 0, used);
                m.playback.flush_to_ring(&mut m.producer, used);
            }
        }

        let mut processed = processed;
        let g_out = h.output_gain;
        if (g_out - 1.0).abs() > 1e-6 {
            for s in processed.iter_mut() {
                *s = (*s * g_out).clamp(-1.0, 1.0);
            }
        }

        let elapsed_us = t0.elapsed().as_micros().min(u128::from(u32::MAX)) as u32;
        st.metrics_process_us.store(elapsed_us, Ordering::Relaxed);
        let prev_ema = st.metrics_process_ema_us.load(Ordering::Relaxed);
        let next_ema = if prev_ema == 0 {
            elapsed_us
        } else {
            ((prev_ema as u64 * 3 + elapsed_us as u64) / 4) as u32
        };
        st.metrics_process_ema_us.store(next_ema, Ordering::Relaxed);
        st.metrics_process_p99_us.store(p99.push(elapsed_us), Ordering::Relaxed);

        let ring_used = out_ring_cap.saturating_sub(out_producer.slots());
        playback.push_interleaved(
            &processed,
            h.buffer_truncate_enabled,
            h.buffer_truncate_rms,
            h.buffer_truncate_chunks,
            ring_used,
        );
        playback.flush_to_ring(&mut out_producer, ring_used);
        let ring_used = out_ring_cap.saturating_sub(out_producer.slots());
        let ql = (playback.queue_len() + ring_used).min(u32::MAX as usize) as u32;
        st.metrics_buffer_samples.store(ql, Ordering::Relaxed);
    }

    // Published rates/metrics are cleared by the worker epilogue in `start_voice_changer`
    // (generation-guarded), which also covers the early `return Err` paths above.
    tracing::info!("voice changer: audio streams stopped");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{panic_message, run_ep_outcome, WorkerError};
    use crate::services::diagnostics::{record_ep_outcome, EpOutcome};

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
}
