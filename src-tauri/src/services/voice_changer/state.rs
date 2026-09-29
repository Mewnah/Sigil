//! [`VoiceChangerState`]: the Tauri-managed state shared by the IPC commands and the worker, the
//! params snapshot / apply rules used by IPC and persistence, and the persistence wiring.

use super::aux_feed::AuxFeed;
use super::params::{
    clamp_output_buffer_ms, latency_budget_or, normalize_execution_provider, normalize_latency_preset,
    normalize_pitch_extractor, normalize_rvc_preset, AuxMicGate, RvcParamsState, VoiceChangerHotParams,
    VoiceChangerMode, VoiceChangerParams, DEFAULT_VOCODER_OVERSAMPLE, MAX_QUEUE_MS_STABLE, PREFILL_MS_STABLE,
};
use super::persist::{
    load_json, parse_doc, settings_path, slot_settings_path, to_json_bytes, LoadSource, OutsideRvcVoice,
    SettingsFile, SettingsStore, SlotSettingsFile, SETTINGS_VERSION,
};
use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::clamp_chunk_frames;
use crate::services::rt_thread::WorkerPriority;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicU8, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How long after Rust replaced the per-voice values the UI sent a sync that still carries them
/// counts as sent before the UI adopted the reply (the IPC round trip is milliseconds).
const STALE_SYNC_WINDOW: Duration = Duration::from_secs(2);

/// Per-voice values Rust put in place of the ones the UI sent (a voice's `slot.json` on entering
/// it, or the other modes' pitch on leaving RVC mode). Syncs the UI sent before it adopted the
/// reply still carry the replaced values; they must not overwrite the voice.
#[derive(Clone, Debug)]
pub(super) struct ReplacedVoice {
    /// What the UI sent when Rust replaced it.
    stale: SlotSettingsFile,
    /// What Rust applied instead.
    effective: SlotSettingsFile,
    at: Instant,
}

/// Lock a state mutex, recovering the data if a panic poisoned it: every value behind these locks
/// is plain data, and a wedged lock would make every later command fail.
pub(super) fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A worker thread plus a receiver that disconnects when the thread exits (its sender is dropped
/// at the end of the thread), so a successor can wait for it with a timeout instead of polling.
pub(super) struct WorkerHandle {
    pub(super) join: JoinHandle<()>,
    pub(super) done: Receiver<()>,
}

/// Dropped at the end of a worker thread (also when it unwinds): disconnects [`WorkerHandle::done`].
pub(super) struct WorkerDoneGuard(#[allow(dead_code)] pub(super) Sender<()>);

/// MMCSS registration outcome of one audio callback thread.
#[derive(Default)]
pub(super) struct MmcssFlag(AtomicU8);

impl MmcssFlag {
    /// Record the first registration attempt of the stream's callback thread (real-time safe).
    #[inline]
    pub(super) fn record(&self, ok: bool) {
        if self.0.load(Ordering::Relaxed) == 0 {
            self.0.store(if ok { 1 } else { 2 }, Ordering::Relaxed);
        }
    }

    /// `None` before the first callback, else whether MMCSS registration succeeded.
    pub(super) fn get(&self) -> Option<bool> {
        match self.0.load(Ordering::Relaxed) {
            0 => None,
            v => Some(v == 1),
        }
    }

    pub(super) fn reset(&self) {
        self.0.store(0, Ordering::Relaxed);
    }
}

/// MMCSS "Pro Audio" status of the capture, output and monitor callback threads.
#[derive(Default)]
pub(super) struct MmcssStatus {
    pub(super) capture: MmcssFlag,
    pub(super) output: MmcssFlag,
    pub(super) monitor: MmcssFlag,
}

impl MmcssStatus {
    fn reset(&self) {
        self.capture.reset();
        self.output.reset();
        self.monitor.reset();
    }
}

#[derive(Clone)]
pub struct VoiceChangerState {
    pub(super) enabled: Arc<Mutex<bool>>,
    pub(super) hot_params: Arc<Mutex<VoiceChangerHotParams>>,
    pub(super) output_device: Arc<Mutex<String>>,
    /// cpal input device name (`""` = system default).
    pub(super) input_device: Arc<Mutex<String>>,
    pub(super) mode: Arc<Mutex<VoiceChangerMode>>,
    pub(super) sidecar_ws_url: Arc<Mutex<String>>,
    pub(super) onnx_model_path: Arc<Mutex<String>>,
    pub(super) onnx_execution_provider: Arc<Mutex<String>>,
    pub(super) latency_preset: Arc<Mutex<String>>,
    pub(super) monitor_output_device: Arc<Mutex<String>>,
    pub(super) rvc: Arc<Mutex<RvcParamsState>>,
    pub(super) rvc_consent_acknowledged: Arc<AtomicBool>,
    pub(super) stop_sender: Arc<Mutex<Option<Sender<()>>>>,
    pub(super) is_running: Arc<AtomicBool>,
    /// Bumped per `start_voice_changer`; a worker only clears `is_running` / emits `stopped`
    /// when it is still the current generation (a restarted worker must not kill its successor).
    pub(super) run_generation: Arc<AtomicU64>,
    /// Previous worker thread; the next worker waits (bounded, 3 s) for it to finish before
    /// opening audio devices and detaches it otherwise.
    pub(super) worker: Arc<Mutex<Option<WorkerHandle>>>,
    /// Aux (TTS) clips mixed into the capture stream before conversion.
    pub(super) aux_feed: Arc<AuxFeed>,
    /// Capture sample rate while the loop runs (0 = not running).
    pub(super) capture_sample_rate: Arc<AtomicU32>,
    pub(super) capture_channels: Arc<AtomicU32>,
    pub(super) output_sample_rate: Arc<AtomicU32>,
    pub(super) output_channels: Arc<AtomicU32>,
    pub(super) metrics_process_us: Arc<AtomicU32>,
    /// Exponential moving average of per-chunk processing time (µs) for stable UI.
    pub(super) metrics_process_ema_us: Arc<AtomicU32>,
    pub(super) metrics_buffer_samples: Arc<AtomicU32>,
    /// p99 of the per-block processing time over the recent window (µs).
    pub(super) metrics_process_p99_us: Arc<AtomicU32>,
    /// Captured frames waiting to be converted (capture ring + backlog).
    pub(super) metrics_backlog_frames: Arc<AtomicU32>,
    /// Capture frames dropped by the latency budget / ring overflow since start.
    pub(super) metrics_dropped_frames: Arc<AtomicU64>,
    /// Output callbacks that ran dry since start.
    pub(super) metrics_underruns: Arc<AtomicU64>,
    /// EP outcome of the running model (`None` for classic / sidecar or when stopped).
    pub(super) run_ep: Arc<Mutex<Option<EpOutcome>>>,
    pub(super) mmcss: Arc<MmcssStatus>,
    pub(super) store: Arc<SettingsStore>,
    /// Pitch and formant of the non-RVC modes while RVC mode (whose voices bring their own) is
    /// active; restored on leaving RVC mode. Persisted in `settings.json`.
    pub(super) outside_rvc: Arc<Mutex<Option<OutsideRvcVoice>>>,
    /// The last replacement of per-voice values (see [`ReplacedVoice`]).
    pub(super) replaced: Arc<Mutex<Option<ReplacedVoice>>>,
}

impl VoiceChangerState {
    pub(super) fn new() -> Self {
        Self {
            enabled: Arc::new(Mutex::new(false)),
            hot_params: Arc::new(Mutex::new(VoiceChangerHotParams::default())),
            output_device: Arc::new(Mutex::new("default".to_string())),
            input_device: Arc::new(Mutex::new(String::new())),
            mode: Arc::new(Mutex::new(VoiceChangerMode::Classic)),
            sidecar_ws_url: Arc::new(Mutex::new(String::new())),
            onnx_model_path: Arc::new(Mutex::new(String::new())),
            onnx_execution_provider: Arc::new(Mutex::new("cpu".to_string())),
            latency_preset: Arc::new(Mutex::new("low_latency".to_string())),
            monitor_output_device: Arc::new(Mutex::new(String::new())),
            rvc: Arc::new(Mutex::new(RvcParamsState::default())),
            rvc_consent_acknowledged: Arc::new(AtomicBool::new(false)),
            stop_sender: Arc::new(Mutex::new(None)),
            is_running: Arc::new(AtomicBool::new(false)),
            run_generation: Arc::new(AtomicU64::new(0)),
            worker: Arc::new(Mutex::new(None)),
            aux_feed: Arc::new(AuxFeed::default()),
            capture_sample_rate: Arc::new(AtomicU32::new(0)),
            capture_channels: Arc::new(AtomicU32::new(0)),
            output_sample_rate: Arc::new(AtomicU32::new(0)),
            output_channels: Arc::new(AtomicU32::new(0)),
            metrics_process_us: Arc::new(AtomicU32::new(0)),
            metrics_process_ema_us: Arc::new(AtomicU32::new(0)),
            metrics_buffer_samples: Arc::new(AtomicU32::new(0)),
            metrics_process_p99_us: Arc::new(AtomicU32::new(0)),
            metrics_backlog_frames: Arc::new(AtomicU32::new(0)),
            metrics_dropped_frames: Arc::new(AtomicU64::new(0)),
            metrics_underruns: Arc::new(AtomicU64::new(0)),
            run_ep: Arc::new(Mutex::new(None)),
            mmcss: Arc::new(MmcssStatus::default()),
            store: Arc::new(SettingsStore::default()),
            outside_rvc: Arc::new(Mutex::new(None)),
            replaced: Arc::new(Mutex::new(None)),
        }
    }

    /// Zero the per-run counters (a new run reports "since start").
    pub(super) fn reset_run_metrics(&self) {
        self.metrics_process_us.store(0, Ordering::Relaxed);
        self.metrics_process_ema_us.store(0, Ordering::Relaxed);
        self.metrics_buffer_samples.store(0, Ordering::Relaxed);
        self.metrics_process_p99_us.store(0, Ordering::Relaxed);
        self.metrics_backlog_frames.store(0, Ordering::Relaxed);
        self.metrics_dropped_frames.store(0, Ordering::Relaxed);
        self.metrics_underruns.store(0, Ordering::Relaxed);
        self.mmcss.reset();
        *lock(&self.run_ep) = None;
    }

    /// Clear what a running worker publishes (device rates, metrics, EP, pending aux audio), so a
    /// stopped voice changer reports "not running" to aux/TTS pushes and the metrics UI.
    pub(super) fn reset_run_outputs(&self) {
        self.capture_sample_rate.store(0, Ordering::Relaxed);
        self.capture_channels.store(0, Ordering::Relaxed);
        self.output_sample_rate.store(0, Ordering::Relaxed);
        self.output_channels.store(0, Ordering::Relaxed);
        self.reset_run_metrics();
        self.aux_feed.reset(0, 1);
    }

    /// The current params, as `get_voice_changer_params` returns them (every P0 field set).
    pub(super) fn snapshot_params(&self) -> VoiceChangerParams {
        let h = *lock(&self.hot_params);
        let rvc = lock(&self.rvc).clone();
        VoiceChangerParams {
            enabled: *lock(&self.enabled),
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
            output_device: lock(&self.output_device).clone(),
            mode: lock(&self.mode).clone(),
            sidecar_ws_url: lock(&self.sidecar_ws_url).clone(),
            onnx_model_path: lock(&self.onnx_model_path).clone(),
            onnx_execution_provider: lock(&self.onnx_execution_provider).clone(),
            latency_preset: lock(&self.latency_preset).clone(),
            monitor_output_device: lock(&self.monitor_output_device).clone(),
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
            latency_budget_ms: Some(h.latency_budget_ms),
            aux_mic_gate: Some(h.aux_mic_gate.as_str().to_string()),
            worker_priority: Some(h.worker_priority.as_str().to_string()),
            input_device: Some(lock(&self.input_device).clone()),
            rvc_consent_acknowledged: Some(self.rvc_consent_acknowledged.load(Ordering::Relaxed)),
        }
    }

    /// Store `params`, clamped and normalized (the rules of `set_voice_changer_params`). Absent
    /// P0 options keep their stored values.
    pub(super) fn apply_params(&self, params: VoiceChangerParams) {
        let lp = normalize_latency_preset(&params.latency_preset);
        let (pf, mq) = if lp == "stable" {
            clamp_output_buffer_ms(PREFILL_MS_STABLE, MAX_QUEUE_MS_STABLE)
        } else {
            clamp_output_buffer_ms(params.output_prefill_ms, params.output_max_queue_ms)
        };
        {
            let mut h = lock(&self.hot_params);
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
            h.latency_budget_ms = latency_budget_or(h.latency_budget_ms, params.latency_budget_ms);
            if let Some(g) = params.aux_mic_gate.as_deref() {
                h.aux_mic_gate = AuxMicGate::from_str_lossy(g);
            }
            if let Some(w) = params.worker_priority.as_deref() {
                h.worker_priority = WorkerPriority::from_str_lossy(w);
            }
        }
        *lock(&self.enabled) = params.enabled;
        *lock(&self.output_device) = params.output_device;
        *lock(&self.mode) = params.mode;
        *lock(&self.sidecar_ws_url) = params.sidecar_ws_url;
        *lock(&self.onnx_model_path) = params.onnx_model_path;
        *lock(&self.onnx_execution_provider) = params.onnx_execution_provider.to_lowercase();
        *lock(&self.monitor_output_device) = params.monitor_output_device;
        *lock(&self.latency_preset) = lp.to_string();
        if let Some(d) = params.input_device {
            *lock(&self.input_device) = d.trim().to_string();
        }
        if let Some(c) = params.rvc_consent_acknowledged {
            self.rvc_consent_acknowledged.store(c, Ordering::Relaxed);
        }
        let mut rvc = lock(&self.rvc);
        rvc.slot_dir = params.rvc_slot_dir;
        rvc.hubert_path = params.rvc_hubert_path;
        rvc.fcpe_path = params.rvc_fcpe_path;
        rvc.rmvpe_path = params.rvc_rmvpe_path;
        rvc.preset = normalize_rvc_preset(&params.rvc_preset).to_string();
        rvc.pitch_extractor = normalize_pitch_extractor(&params.pitch_extractor).to_string();
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

    /// The selected RVC slot folder (trimmed; empty = none).
    pub(super) fn current_slot_dir(&self) -> String {
        lock(&self.rvc).slot_dir.trim().to_string()
    }

    /// The per-voice values, as stored in the slot's `slot.json`.
    pub(super) fn slot_settings(&self) -> SlotSettingsFile {
        let h = *lock(&self.hot_params);
        let rvc = lock(&self.rvc);
        SlotSettingsFile {
            version: SETTINGS_VERSION,
            pitch_semitones: Some(h.pitch_semitones),
            formant_shift: Some(h.formant_shift),
            index_rate: Some(rvc.index_rate),
            protect: Some(rvc.protect),
            pitch_extractor: Some(rvc.pitch_extractor.clone()),
            speaker_id: Some(rvc.speaker_id),
        }
    }

    /// Apply a voice's `slot.json` (present fields only, clamped like IPC params).
    pub(super) fn apply_slot_settings(&self, s: &SlotSettingsFile) {
        {
            let mut h = lock(&self.hot_params);
            if let Some(v) = s.pitch_semitones.filter(|v| v.is_finite()) {
                h.pitch_semitones = v.clamp(-24.0, 24.0);
            }
            if let Some(v) = s.formant_shift.filter(|v| v.is_finite()) {
                h.formant_shift = v.clamp(-1.0, 1.0);
            }
        }
        let mut rvc = lock(&self.rvc);
        if let Some(v) = s.index_rate.filter(|v| v.is_finite()) {
            rvc.index_rate = v.clamp(0.0, 1.0);
        }
        if let Some(v) = s.protect.filter(|v| v.is_finite()) {
            rvc.protect = v.clamp(0.0, 0.5);
        }
        if let Some(v) = s.pitch_extractor.as_deref() {
            rvc.pitch_extractor = normalize_pitch_extractor(v).to_string();
        }
        if let Some(v) = s.speaker_id {
            rvc.speaker_id = v.max(0);
        }
    }

    /// `set_voice_changer_params`: store `params`, then apply voice rules and persist. Returns
    /// the effective params and whether they differ from what was sent (the UI must then adopt
    /// them).
    ///
    /// - Entering an RVC voice (another slot, or switching into RVC mode) applies that voice's
    ///   `slot.json` in place of the incoming per-voice values.
    /// - Pitch and formant are kept apart for RVC mode and the other modes: switching into RVC
    ///   mode sets the other modes' values aside, and leaving it restores them (unless the same
    ///   sync changed them itself).
    /// - Within one voice, changed per-voice values are saved to its `slot.json`, except for a
    ///   sync that still carries exactly the values Rust just replaced (sent before the UI adopted
    ///   the reply): that one keeps the replacement instead of overwriting the voice with the
    ///   previous voice's values.
    pub(super) fn update_params(&self, params: VoiceChangerParams) -> (VoiceChangerParams, bool) {
        let before_slot = self.current_slot_dir();
        let before_mode = lock(&self.mode).clone();
        let before_voice = self.slot_settings();
        self.apply_params(params);
        let sent = self.slot_settings();
        let slot = self.current_slot_dir();
        let mode = lock(&self.mode).clone();
        let (was_rvc, in_rvc) = (before_mode == VoiceChangerMode::Rvc, mode == VoiceChangerMode::Rvc);
        let transition = before_mode != mode || (in_rvc && !same_dir(&before_slot, &slot));
        let last_replacement = lock(&self.replaced).take();
        let mut replaced = None;

        if !transition {
            let stale = last_replacement.filter(|r| {
                r.at.elapsed() < STALE_SYNC_WINDOW && sent == r.stale && sent != r.effective
            });
            if let Some(r) = stale {
                self.apply_slot_settings(&r.effective);
                replaced = Some(r);
            } else if in_rvc && !slot.is_empty() && sent != before_voice {
                self.save_slot_settings();
            }
        } else {
            if in_rvc && !was_rvc {
                *lock(&self.outside_rvc) = Some(OutsideRvcVoice {
                    pitch_semitones: sent.pitch_semitones.unwrap_or_default(),
                    formant_shift: sent.formant_shift.unwrap_or_default(),
                });
            }
            if in_rvc && !slot.is_empty() {
                if let Some(file) = self.load_slot_settings(Path::new(&slot)) {
                    self.apply_slot_settings(&file);
                }
            }
            if was_rvc && !in_rvc {
                let kept = lock(&self.outside_rvc).take();
                let untouched = (sent.pitch_semitones, sent.formant_shift)
                    == (before_voice.pitch_semitones, before_voice.formant_shift);
                if let Some(v) = kept.filter(|_| untouched) {
                    self.apply_slot_settings(&SlotSettingsFile {
                        pitch_semitones: Some(v.pitch_semitones),
                        formant_shift: Some(v.formant_shift),
                        ..SlotSettingsFile::default()
                    });
                }
            }
            let effective = self.slot_settings();
            if effective != sent {
                replaced = Some(ReplacedVoice {
                    stale: sent.clone(),
                    effective,
                    at: Instant::now(),
                });
            }
        }
        let adjusted = self.slot_settings() != sent;
        *lock(&self.replaced) = replaced;
        self.save_settings();
        (self.snapshot_params(), adjusted)
    }

    /// Load `settings.json` (or its backup) under `app_data` and enable saving there. A file that
    /// does not parse is kept aside; the voice changer then starts from defaults.
    pub(super) fn attach_persistence(&self, app_data: &Path) {
        self.store.attach(app_data);
        let path = settings_path(app_data);
        let loaded = load_json::<SettingsFile>(&path);
        for note in &loaded.notes {
            tracing::warn!("voice changer settings: {note}");
        }
        let Some((file, source)) = loaded.value else {
            return;
        };
        if file.version > SETTINGS_VERSION {
            tracing::warn!(
                "voice changer settings: {} is from a newer Sigil (version {}); unknown settings are ignored and will not be kept",
                path.display(),
                file.version
            );
        }
        let mut params = file.params;
        params.enabled = false;
        self.apply_params(params);
        *lock(&self.outside_rvc) = file.outside_rvc.filter(|v| v.pitch_semitones.is_finite() && v.formant_shift.is_finite());
        tracing::info!("voice changer settings restored from {}", path.display());
        if source == LoadSource::Backup {
            if loaded.main_unreadable {
                // The main file exists and is probably newer than its backup: never write the
                // backup over it. The next change the user makes is saved as usual.
                tracing::warn!(
                    "voice changer settings: {} could not be read; using its backup for this session",
                    path.display()
                );
            } else {
                // Put a good main file back.
                self.save_settings();
            }
        }
    }

    /// The voice's current `slot.json`: a save still waiting in the debounced writer (the user
    /// tuned the voice, switched away and back within the quiet period), else the file on disk.
    fn load_slot_settings(&self, slot_dir: &Path) -> Option<SlotSettingsFile> {
        self.store.settings_path()?;
        let path = slot_settings_path(slot_dir);
        self.store.writer.with_latest(&path, |pending| {
            if let Some(bytes) = pending {
                return parse_doc::<SlotSettingsFile>(bytes);
            }
            let loaded = load_json::<SlotSettingsFile>(&path);
            for note in &loaded.notes {
                tracing::warn!("voice changer voice settings: {note}");
            }
            loaded.value.map(|(v, _)| v)
        })
    }

    /// Schedule a save of `settings.json` (no-op until persistence is attached).
    pub(super) fn save_settings(&self) {
        let Some(path) = self.store.settings_path() else {
            return;
        };
        let mut file = SettingsFile::new(self.snapshot_params());
        file.outside_rvc = *lock(&self.outside_rvc);
        if let Some(bytes) = to_json_bytes(&file) {
            self.store.writer.schedule(path.to_path_buf(), bytes);
        }
    }

    /// Schedule a save of the selected voice's `slot.json` (only an existing slot folder).
    pub(super) fn save_slot_settings(&self) {
        if self.store.settings_path().is_none() {
            return;
        }
        let slot = self.current_slot_dir();
        if slot.is_empty() {
            return;
        }
        let dir = PathBuf::from(&slot);
        if !dir.is_dir() {
            return;
        }
        if let Some(bytes) = to_json_bytes(&self.slot_settings()) {
            self.store.writer.schedule(slot_settings_path(&dir), bytes);
        }
    }

    /// Write pending settings now (app exit).
    pub(super) fn flush_settings(&self) {
        self.store.writer.flush();
    }
}

/// Whether two slot folder strings name the same folder (separator / trailing-slash insensitive).
fn same_dir(a: &str, b: &str) -> bool {
    Path::new(a.trim()) == Path::new(b.trim())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(tag: &str) -> TempDir {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let p = std::env::temp_dir().join(format!("sigil-vc-state-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    /// get(set(x)) == x for in-range values, and absent P0 options keep what is stored.
    #[test]
    fn params_round_trip_and_absent_options_keep_values() {
        let st = VoiceChangerState::new();
        let mut p = VoiceChangerParams {
            pitch_semitones: 7.0,
            output_device: "Line 1".into(),
            mode: VoiceChangerMode::Rvc,
            pitch_extractor: "crepe".into(),
            chunk_frames: 1024,
            ..VoiceChangerParams::default()
        };
        p.latency_budget_ms = Some(800.0);
        p.aux_mic_gate = Some("mute".into());
        p.worker_priority = Some("protect_game".into());
        p.input_device = Some(" USB Mic ".into());
        p.rvc_consent_acknowledged = Some(true);
        st.apply_params(p.clone());
        let got = st.snapshot_params();
        assert_eq!(got.pitch_extractor, "rmvpe", "the crepe alias is stored as rmvpe");
        assert_eq!(got.input_device.as_deref(), Some("USB Mic"));
        let mut expect = p.clone();
        expect.pitch_extractor = "rmvpe".into();
        expect.input_device = Some("USB Mic".into());
        assert_eq!(got, expect);

        // A sender that predates the P0 fields cannot reset them.
        let old_sender: VoiceChangerParams = serde_json::from_str(r#"{"pitch_semitones":1,"output_device":"Line 1"}"#).unwrap();
        st.apply_params(old_sender);
        let got = st.snapshot_params();
        assert_eq!(got.latency_budget_ms, Some(800.0));
        assert_eq!(got.aux_mic_gate.as_deref(), Some("mute"));
        assert_eq!(got.worker_priority.as_deref(), Some("protect_game"));
        assert_eq!(got.input_device.as_deref(), Some("USB Mic"));
        assert_eq!(got.rvc_consent_acknowledged, Some(true));
        // Out-of-range / unknown values are clamped / normalized.
        st.apply_params(VoiceChangerParams {
            latency_budget_ms: Some(10.0),
            aux_mic_gate: Some("loud".into()),
            worker_priority: Some("realtime".into()),
            ..VoiceChangerParams::default()
        });
        let got = st.snapshot_params();
        assert_eq!(got.latency_budget_ms, Some(100.0));
        assert_eq!(got.aux_mic_gate.as_deref(), Some("none"));
        assert_eq!(got.worker_priority.as_deref(), Some("auto"));
    }

    /// Settings and devices survive a restart: save through the store, flush, load into a fresh
    /// state (as the plugin setup does), and the params match (the voice changer stays off).
    #[test]
    fn persistence_round_trips_through_a_restart() {
        let dir = temp_dir("restart");
        let st = VoiceChangerState::new();
        st.attach_persistence(&dir.0);
        let mut p = VoiceChangerParams {
            enabled: true,
            output_device: "Speakers (Realtek)".into(),
            monitor_output_device: "Headset".into(),
            mode: VoiceChangerMode::Rvc,
            rvc_execution_provider: "dml".into(),
            chunk_frames: 768,
            ..VoiceChangerParams::default()
        };
        p.input_device = Some("Microphone (Yeti)".into());
        p.latency_budget_ms = Some(500.0);
        p.worker_priority = Some("protect_voice".into());
        p.aux_mic_gate = Some("duck".into());
        p.rvc_consent_acknowledged = Some(true);
        st.update_params(p.clone());
        st.flush_settings();

        let restarted = VoiceChangerState::new();
        restarted.attach_persistence(&dir.0);
        let got = restarted.snapshot_params();
        p.enabled = false;
        assert_eq!(got, p);
    }

    /// Per-voice settings travel with the voice: changes are saved to the slot's `slot.json`, and
    /// entering a voice applies its values (reported as adjusted so the UI re-hydrates).
    #[test]
    fn per_voice_settings_follow_the_slot() {
        let dir = temp_dir("slots");
        let (a, b) = (dir.0.join("voice-a"), dir.0.join("voice-b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let st = VoiceChangerState::new();
        st.attach_persistence(&dir.0.join("app"));
        let rvc = |slot: &Path, pitch: f32, protect: f32| VoiceChangerParams {
            mode: VoiceChangerMode::Rvc,
            rvc_slot_dir: slot.to_string_lossy().into_owned(),
            pitch_semitones: pitch,
            protect,
            ..VoiceChangerParams::default()
        };
        // Enter A (no slot.json yet), then tune it: saved to A/slot.json.
        let (_, adjusted) = st.update_params(rvc(&a, 0.0, 0.33));
        assert!(!adjusted);
        st.update_params(rvc(&a, 5.0, 0.2));
        st.flush_settings();
        let saved: SlotSettingsFile = serde_json::from_slice(&std::fs::read(slot_settings_path(&a)).unwrap()).unwrap();
        assert_eq!((saved.pitch_semitones, saved.protect), (Some(5.0), Some(0.2)));
        // Switch to B with A's values in the UI: B has no file, so they stay (and are not saved
        // just by switching); tuning B saves B.
        let (eff, adjusted) = st.update_params(rvc(&b, 5.0, 0.2));
        assert!(!adjusted && eff.pitch_semitones == 5.0);
        st.flush_settings();
        assert!(!slot_settings_path(&b).exists());
        st.update_params(rvc(&b, -2.0, 0.2));
        // Back to A while the UI still shows B's values: A's own values win.
        let (eff, adjusted) = st.update_params(rvc(&a, -2.0, 0.2));
        assert!(adjusted);
        assert_eq!((eff.pitch_semitones, eff.protect), (5.0, 0.2));
        // Outside RVC mode nothing is written into the voice.
        st.flush_settings();
        let before = std::fs::read(slot_settings_path(&a)).unwrap();
        let classic = VoiceChangerParams {
            mode: VoiceChangerMode::Classic,
            rvc_slot_dir: a.to_string_lossy().into_owned(),
            pitch_semitones: 12.0,
            ..VoiceChangerParams::default()
        };
        st.update_params(classic);
        st.flush_settings();
        assert_eq!(std::fs::read(slot_settings_path(&a)).unwrap(), before);
    }

    fn rvc_params(slot: &Path, pitch: f32, protect: f32) -> VoiceChangerParams {
        VoiceChangerParams {
            mode: VoiceChangerMode::Rvc,
            rvc_slot_dir: slot.to_string_lossy().into_owned(),
            pitch_semitones: pitch,
            protect,
            ..VoiceChangerParams::default()
        }
    }

    fn saved_slot(slot: &Path) -> SlotSettingsFile {
        serde_json::from_slice(&std::fs::read(slot_settings_path(slot)).unwrap()).unwrap()
    }

    /// A sync the UI sent before it adopted the reply that entered voice B still carries voice A's
    /// values; it keeps B's (reported as adjusted) and never overwrites B's `slot.json`. Once the
    /// UI sends B's values, edits are saved to B again.
    #[test]
    fn in_flight_sync_after_entering_a_voice_does_not_overwrite_it() {
        let dir = temp_dir("stale-sync");
        let (a, b) = (dir.0.join("voice-a"), dir.0.join("voice-b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let st = VoiceChangerState::new();
        st.attach_persistence(&dir.0.join("app"));
        st.update_params(rvc_params(&b, 0.0, 0.33));
        st.update_params(rvc_params(&b, -2.0, 0.33)); // B tuned: saved
        st.update_params(rvc_params(&a, -2.0, 0.33));
        st.update_params(rvc_params(&a, 5.0, 0.33)); // A tuned: saved
        st.flush_settings();
        assert_eq!(saved_slot(&b).pitch_semitones, Some(-2.0));

        // Select B (sync #1 still carries A's pitch 5): B's own values win.
        let (eff, adjusted) = st.update_params(rvc_params(&b, 5.0, 0.33));
        assert!(adjusted && eff.pitch_semitones == -2.0);
        // Sync #2 was sent before the UI adopted #1's reply: still A's pitch 5.
        let (eff, adjusted) = st.update_params(rvc_params(&b, 5.0, 0.33));
        assert!(adjusted, "the UI must adopt B's values");
        assert_eq!(eff.pitch_semitones, -2.0);
        assert_eq!(lock(&st.hot_params).pitch_semitones, -2.0, "B plays with its own pitch");
        st.flush_settings();
        assert_eq!(saved_slot(&b).pitch_semitones, Some(-2.0), "B's slot.json is untouched");
        // The UI adopted B's values; then the user tunes B, even to A's old value: saved.
        let (_, adjusted) = st.update_params(rvc_params(&b, -2.0, 0.33));
        assert!(!adjusted);
        let (eff, adjusted) = st.update_params(rvc_params(&b, 5.0, 0.33));
        assert!(!adjusted && eff.pitch_semitones == 5.0);
        st.flush_settings();
        assert_eq!(saved_slot(&b).pitch_semitones, Some(5.0));
    }

    /// Switching back to a voice tuned moments ago (its `slot.json` save still waiting in the
    /// debounced writer) applies what was just tuned, not the older file on disk.
    #[test]
    fn returning_to_a_voice_before_its_save_lands_uses_the_pending_save() {
        let dir = temp_dir("pending-slot");
        let (a, b) = (dir.0.join("voice-a"), dir.0.join("voice-b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        std::fs::write(slot_settings_path(&b), br#"{"version":1,"pitch_semitones":1}"#).unwrap();
        let st = VoiceChangerState::new();
        st.attach_persistence(&dir.0.join("app"));
        st.update_params(rvc_params(&b, 1.0, 0.33));
        st.update_params(rvc_params(&b, 7.0, 0.33)); // tuned; the save is pending
        st.update_params(rvc_params(&a, 7.0, 0.33));
        let (eff, _) = st.update_params(rvc_params(&b, 7.0, 0.33));
        assert_eq!(eff.pitch_semitones, 7.0);
        st.update_params(rvc_params(&a, 7.0, 0.33));
        st.update_params(rvc_params(&a, 0.0, 0.33));
        let (eff, adjusted) = st.update_params(rvc_params(&b, 0.0, 0.33));
        assert!(adjusted && eff.pitch_semitones == 7.0);
    }

    /// A stale-looking sync long after the replacement is a user edit (the guard expires).
    #[test]
    fn stale_sync_guard_expires() {
        let dir = temp_dir("stale-expiry");
        let (a, b) = (dir.0.join("voice-a"), dir.0.join("voice-b"));
        std::fs::create_dir_all(&a).unwrap();
        std::fs::create_dir_all(&b).unwrap();
        let st = VoiceChangerState::new();
        st.attach_persistence(&dir.0.join("app"));
        st.update_params(rvc_params(&b, 0.0, 0.33));
        st.update_params(rvc_params(&b, -2.0, 0.33));
        st.update_params(rvc_params(&a, -2.0, 0.33));
        st.update_params(rvc_params(&a, 5.0, 0.33));
        let (_, adjusted) = st.update_params(rvc_params(&b, 5.0, 0.33));
        assert!(adjusted);
        lock(&st.replaced).as_mut().unwrap().at -= STALE_SYNC_WINDOW;
        let (eff, adjusted) = st.update_params(rvc_params(&b, 5.0, 0.33));
        assert!(!adjusted && eff.pitch_semitones == 5.0);
    }

    /// Visiting RVC mode does not change the other modes' pitch and formant: they are set aside on
    /// entering it and restored on leaving it, across a restart too; a pitch changed in the same
    /// sync that leaves RVC mode is kept.
    #[test]
    fn leaving_rvc_mode_restores_the_other_modes_pitch() {
        let dir = temp_dir("outside-rvc");
        let voice = dir.0.join("voice");
        std::fs::create_dir_all(&voice).unwrap();
        std::fs::write(slot_settings_path(&voice), br#"{"version":1,"pitch_semitones":12,"formant_shift":0.5}"#).unwrap();
        let app = dir.0.join("app");
        let st = VoiceChangerState::new();
        st.attach_persistence(&app);
        let classic = |pitch: f32, formant: f32| VoiceChangerParams {
            mode: VoiceChangerMode::Classic,
            rvc_slot_dir: voice.to_string_lossy().into_owned(),
            pitch_semitones: pitch,
            formant_shift: formant,
            ..VoiceChangerParams::default()
        };
        let rvc = |pitch: f32, formant: f32| VoiceChangerParams {
            mode: VoiceChangerMode::Rvc,
            ..classic(pitch, formant)
        };
        st.update_params(classic(3.0, -0.2));
        let (eff, adjusted) = st.update_params(rvc(3.0, -0.2));
        assert!(adjusted && (eff.pitch_semitones, eff.formant_shift) == (12.0, 0.5));
        // Restart while in RVC mode: the classic values survive in settings.json.
        st.flush_settings();
        let st = VoiceChangerState::new();
        st.attach_persistence(&app);
        assert_eq!(st.snapshot_params().pitch_semitones, 12.0);
        let (eff, adjusted) = st.update_params(classic(12.0, 0.5));
        assert!(adjusted);
        assert_eq!((eff.pitch_semitones, eff.formant_shift), (3.0, -0.2));
        // A sync sent before the UI adopted that reply keeps the restored values.
        let (eff, adjusted) = st.update_params(classic(12.0, 0.5));
        assert!(adjusted && eff.pitch_semitones == 3.0);
        // Classic tuning is not written into the voice.
        st.update_params(classic(4.0, -0.2));
        st.flush_settings();
        assert_eq!(saved_slot(&voice).pitch_semitones, Some(12.0));
        // Leaving RVC mode with a new pitch in the same sync keeps that pitch.
        st.update_params(rvc(4.0, -0.2));
        let (eff, adjusted) = st.update_params(classic(-5.0, 0.5));
        assert!(!adjusted && eff.pitch_semitones == -5.0);
    }

    /// A main settings file that cannot be read is not overwritten by the (older) backup at
    /// startup.
    #[test]
    fn unreadable_settings_are_not_repaired_from_the_backup() {
        let dir = temp_dir("unreadable");
        let path = settings_path(&dir.0);
        std::fs::create_dir_all(&path).unwrap(); // stands in for a file another process holds
        let backup = to_json_bytes(&SettingsFile::new(VoiceChangerParams {
            output_device: "Old device".into(),
            ..VoiceChangerParams::default()
        }))
        .unwrap();
        std::fs::write(crate::services::voice_changer::persist::backup_path(&path), backup).unwrap();
        let st = VoiceChangerState::new();
        st.attach_persistence(&dir.0);
        assert_eq!(st.snapshot_params().output_device, "Old device");
        assert_eq!(st.store.writer.pending_len(), 0, "no repair write was scheduled");
    }

    #[test]
    fn slot_folders_compare_by_path() {
        assert!(same_dir("C:\\voices\\a", "C:\\voices\\a\\"));
        assert!(same_dir(" C:\\voices\\a ", "C:\\voices\\a"));
        assert!(!same_dir("C:\\voices\\a", "C:\\voices\\b"));
    }
}
