//! `audio` plugin: device lists, the capture hub (one stream per physical source, fanned out to
//! STT and other consumers), device-follow events, the Rust mute gate, and `play_async`.
//!
//! Capture (P1-C):
//! - [`hub`]: sources, per-consumer rtrb rings with QPC-stamped blocks, drop-oldest per consumer,
//!   16 kHz mono or device-rate output, mute gate; [`watchdog`]: restarts under the same epoch,
//!   stall detection, default-device follow; [`stream`]: the cpal/WASAPI backend (MMCSS callback
//!   threads); [`loopback`]: source ids, output devices as loopback sources; [`device_watch`]:
//!   `IMMNotificationClient` with a polling fallback.
//! - Consumers: Whisper (`whisper::start_recording`), utterance capture (HTTP STT), and the legacy
//!   `audio:chunk` stream (Vosk in the webview; 16 kHz mono when `sampleRate: 16000` is requested).
//!
//! Events: `audio:devices_changed` ([`device_watch::DevicesChanged`]), `audio:stream_error`,
//! `audio:stream_recovered` and `audio:stream_closed` ([`hub::StreamEvent`]), `audio:chunk`,
//! `audio:utterance`.
//! Source ids accepted by every `deviceName`: see [`loopback`] (`"loopback:<output>"` captures
//! what an output device plays).

use std::io::Cursor;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread::{self, Thread};
use std::time::Duration;

use cpal::traits::{DeviceTrait, HostTrait};
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};
use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};
use tokio::sync::oneshot;

pub mod consumer;
pub mod device_resolve;
pub mod device_watch;
pub mod hub;
pub mod loopback;
pub mod speech_capture;
pub mod stream;
pub mod utterance_capture;
mod watchdog;

#[cfg(test)]
mod hub_tests;

pub use device_resolve::resolve_output_device;

use hub::{
    CaptureConsumer, ConsumerFormat, ConsumerOptions, HubEventSink, HubStatus, MuteGroup, SourceSpec, StreamEvent,
    STREAM_CLOSED_EVENT, STREAM_ERROR_EVENT, STREAM_RECOVERED_EVENT,
};
use utterance_capture::UtteranceCaptureState;

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
}

impl AudioDevice {
    /// A cpal device row (id == name).
    pub fn named(name: &str) -> Self {
        Self {
            id: name.to_string(),
            name: name.to_string(),
        }
    }
}

/// Idle wake-up of the `audio:chunk` emitter (the hub wakes it every ~20 ms of audio).
const CHUNK_IDLE_WAIT: Duration = Duration::from_millis(100);
/// How far the `audio:chunk` emitter may fall behind before it drops its oldest audio.
const CHUNK_MAX_BACKLOG_MS: u32 = 1_000;
/// Bound on waiting for the emitter to release its consumer.
const CHUNK_STOP_TIMEOUT: Duration = Duration::from_secs(3);

/// The raw `audio:chunk` capture (Vosk and legacy polling engines).
pub struct AudioCaptureState {
    /// The stop lock: the running capture. Chunks are emitted while holding it with the epoch current.
    active: Arc<Mutex<Option<ChunkCapture>>>,
    /// Bumped (under `active`'s lock) by every start and stop. A start re-checks it under the same lock
    /// before publishing, so a stop that lands while the device is still opening cancels that start. The
    /// emitter compares it with its own start's epoch, so a superseded capture never emits.
    capture_epoch: Arc<AtomicU64>,
}

struct ChunkCapture {
    stop: Arc<AtomicBool>,
    emitter: Thread,
    /// Resolves once the emitter released its hub consumer (and the device, if it was the last).
    done: Option<oneshot::Receiver<()>>,
}

impl ChunkCapture {
    fn signal_stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.emitter.unpark();
    }

    async fn stop(mut self) {
        self.signal_stop();
        if let Some(done) = self.done.take() {
            let _ = tokio::time::timeout(CHUNK_STOP_TIMEOUT, done).await;
        }
    }
}

impl AudioCaptureState {
    fn new() -> Self {
        Self {
            active: Arc::new(Mutex::new(None)),
            capture_epoch: Arc::new(AtomicU64::new(0)),
        }
    }
}

fn lock_active(active: &Mutex<Option<ChunkCapture>>) -> MutexGuard<'_, Option<ChunkCapture>> {
    active.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Mute group of a consumer of `spec`: the user's STT for inputs, "others" for loopback.
pub(crate) fn stt_mute_group(spec: &SourceSpec) -> MuteGroup {
    if spec.is_loopback() {
        MuteGroup::Others
    } else {
        MuteGroup::Stt
    }
}

/// Attach to the global hub off the async runtime (a first consumer opens the device).
pub(crate) async fn attach_async(spec: SourceSpec, opts: ConsumerOptions) -> Result<CaptureConsumer, String> {
    tauri::async_runtime::spawn_blocking(move || hub::global()?.attach(spec, opts))
        .await
        .map_err(|e| format!("Audio capture task failed: {e}"))?
}

fn list_devices(output: bool) -> Result<Vec<AudioDevice>, String> {
    let host = cpal::default_host();
    let devices = if output {
        host.output_devices()
    } else {
        host.input_devices()
    }
    .map_err(|e| e.to_string())?;
    Ok(devices.filter_map(|d| d.name().ok()).map(|n| AudioDevice::named(n.trim())).collect())
}

#[command]
pub async fn list_input_devices() -> Result<Vec<AudioDevice>, String> {
    tauri::async_runtime::spawn_blocking(|| list_devices(false))
        .await
        .map_err(|e| e.to_string())?
}

#[command]
pub async fn list_output_devices() -> Result<Vec<AudioDevice>, String> {
    tauri::async_runtime::spawn_blocking(|| list_devices(true))
        .await
        .map_err(|e| e.to_string())?
}

/// Inputs, outputs and the current defaults in one call (same shape as `audio:devices_changed`,
/// reason `snapshot`). Loopback sources are the outputs as `"loopback:<name>"`.
#[command]
pub async fn get_device_snapshot() -> Result<device_watch::DevicesChanged, String> {
    let snapshot = tauri::async_runtime::spawn_blocking(device_watch::snapshot_cpal)
        .await
        .map_err(|e| e.to_string())??;
    Ok(device_watch::DevicesChanged {
        reason: "snapshot".to_string(),
        snapshot,
    })
}

/// The Rust mute gate: while a group is muted its consumers receive no audio at all.
#[command]
pub fn set_capture_mute(group: MuteGroup, muted: bool) -> Result<(), String> {
    hub::global()?.set_muted(group, muted);
    Ok(())
}

/// Sources, streams and mute state of the capture hub (diagnostics).
#[command]
pub async fn get_capture_status() -> Result<HubStatus, String> {
    tauri::async_runtime::spawn_blocking(|| hub::global()?.status())
        .await
        .map_err(|e| e.to_string())?
}

#[derive(Clone, Serialize)]
struct AudioChunkPayload {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

/// `audio:chunk` stream for webview engines (Vosk). `sampleRate: 16000` yields 16 kHz mono
/// (band-limited, a sixth of the IPC of 48 kHz stereo); anything else the device format. Every
/// payload states its own format. Served by the capture hub, so it shares the device with Whisper
/// and utterance capture and follows device changes.
#[command]
pub async fn start_audio_capture<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AudioCaptureState>,
    device_name: Option<String>,
    sample_rate: Option<u32>,
) -> Result<(), String> {
    // Stop any existing capture and claim an epoch before the (slow) device open.
    let (my_epoch, previous) = {
        let mut active = lock_active(&state.active);
        let epoch = state.capture_epoch.fetch_add(1, Ordering::SeqCst) + 1;
        (epoch, active.take())
    };
    if let Some(previous) = previous {
        previous.stop().await;
    }

    let spec = SourceSpec::parse(device_name.as_deref());
    let format = if sample_rate == Some(hub::StreamFormat::STT.sample_rate) {
        ConsumerFormat::Raw16kMono
    } else {
        ConsumerFormat::DeviceRate
    };
    let opts = ConsumerOptions {
        format,
        max_backlog_ms: CHUNK_MAX_BACKLOG_MS,
        wake_ms: 20,
        mute_group: Some(stt_mute_group(&spec)),
        label: "audio-chunk",
    };
    let consumer = attach_async(spec, opts).await?;

    let stop = Arc::new(AtomicBool::new(false));
    let (done_tx, done_rx) = oneshot::channel();
    // Emitter: batches the consumer's audio into `audio:chunk` events. Emits only while holding the
    // stop lock with this start's epoch current, so once `stop_audio_capture` returns nothing more is
    // emitted, and a superseded capture's chunks never reach a newer one.
    let spawned = {
        let stop = Arc::clone(&stop);
        let active = Arc::clone(&state.active);
        let epoch = Arc::clone(&state.capture_epoch);
        thread::Builder::new().name("sigil-audio-chunk".into()).spawn(move || {
            let mut consumer = consumer;
            let mut samples = Vec::with_capacity(8192);
            loop {
                if stop.load(Ordering::Acquire) {
                    break;
                }
                samples.clear();
                let info = consumer.read(&mut samples);
                if let Some(format) = info.format.filter(|_| !samples.is_empty()) {
                    let _guard = lock_active(&active);
                    if epoch.load(Ordering::SeqCst) != my_epoch || stop.load(Ordering::Acquire) {
                        break;
                    }
                    let _ = app.emit(
                        "audio:chunk",
                        AudioChunkPayload {
                            samples: samples.clone(),
                            sample_rate: format.sample_rate,
                            channels: format.channels,
                        },
                    );
                    continue;
                }
                if !info.gap_after {
                    consumer.wait(CHUNK_IDLE_WAIT);
                }
            }
            consumer.close_and_wait();
            let _ = done_tx.send(());
        })
    };
    let emitter = match spawned {
        Ok(handle) => handle.thread().clone(),
        // The consumer moved into the failed closure and was dropped (detached).
        Err(e) => return Err(format!("Failed to start the audio chunk emitter: {e}")),
    };

    let mut capture = ChunkCapture {
        stop,
        emitter,
        done: Some(done_rx),
    };
    let cancelled = {
        let mut active = lock_active(&state.active);
        if state.capture_epoch.load(Ordering::SeqCst) == my_epoch {
            *active = Some(capture);
            None
        } else {
            capture.signal_stop();
            Some(capture)
        }
    };
    if let Some(capture) = cancelled {
        // A stop (or a newer start) landed while the device was opening.
        capture.stop().await;
        return Err("cancelled".to_string());
    }
    Ok(())
}

#[command]
pub fn stop_audio_capture(state: State<'_, AudioCaptureState>) -> Result<(), String> {
    let mut active = lock_active(&state.active);
    // Cancels a start that is still opening its device (it re-checks the epoch under this lock).
    state.capture_epoch.fetch_add(1, Ordering::SeqCst);
    if let Some(mut capture) = active.take() {
        capture.signal_stop();
    }
    Ok(())
}

fn get_output_stream(device_name: &str) -> Result<(OutputStream, OutputStreamHandle), String> {
    let name = device_name.trim();
    if name.is_empty() || name.eq_ignore_ascii_case("default") {
        // rodio's `try_default` falls back to any other output device when the default one
        // cannot be opened (exclusive-mode hold, unplugged endpoint, unsupported config).
        return OutputStream::try_default().map_err(|e| e.to_string());
    }
    let host = cpal::default_host();
    let device = resolve_output_device(&host, device_name)?;
    OutputStream::try_from_device(&device).map_err(|e| e.to_string())
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcAudioPlayAsync {
    pub device_name: String,
    pub data: Vec<u8>,
    pub volume: f32,
    pub rate: f32,
}

#[command]
pub async fn play_async(data: RpcAudioPlayAsync) -> Result<(), String> {
    let (_stream, stream_handle) = get_output_stream(&data.device_name)?;
    let sink = Sink::try_new(&stream_handle).map_err(|e| e.to_string())?;
    sink.set_volume(data.volume);
    sink.set_speed(data.rate);

    let source = Decoder::new(Cursor::new(data.data)).map_err(|e| e.to_string())?;
    sink.append(source);
    sink.sleep_until_end();
    Ok(())
}

/// Forwards hub stream events to the webview.
struct TauriHubEvents<R: Runtime> {
    app: AppHandle<R>,
}

impl<R: Runtime> HubEventSink for TauriHubEvents<R> {
    fn stream_error(&self, event: &StreamEvent) {
        let _ = self.app.emit(STREAM_ERROR_EVENT, event.clone());
    }

    fn stream_recovered(&self, event: &StreamEvent) {
        let _ = self.app.emit(STREAM_RECOVERED_EVENT, event.clone());
    }

    fn stream_closed(&self, event: &StreamEvent) {
        let _ = self.app.emit(STREAM_CLOSED_EVENT, event.clone());
    }
}

/// Keeps the device watcher alive for the app's lifetime.
struct DeviceWatchState(#[allow(dead_code)] Mutex<Option<device_watch::DeviceWatcher>>);

fn start_capture_services<R: Runtime>(app: &AppHandle<R>) {
    match hub::global() {
        Ok(hub) => hub.set_event_sink(Arc::new(TauriHubEvents { app: app.clone() })),
        Err(e) => tracing::error!(target: "sigil::audio", "{e}"),
    }
    let emitter = app.clone();
    let watcher = device_watch::start(
        device_watch::snapshot_cpal,
        move |notice| {
            if let device_watch::WatchNotice::Changed(event) = &notice {
                let _ = emitter.emit(device_watch::DEVICES_CHANGED_EVENT, event.clone());
            }
            if let Ok(hub) = hub::global() {
                hub.devices_changed();
            }
        },
        device_watch::WatchConfig::default(),
    );
    match watcher {
        Ok(watcher) => {
            app.manage(DeviceWatchState(Mutex::new(Some(watcher))));
        }
        Err(e) => tracing::error!(target: "sigil::audio", "{e}"),
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("audio")
        .setup(|app, _api| {
            app.manage(AudioCaptureState::new());
            app.manage(UtteranceCaptureState::new());
            start_capture_services(app);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            play_async,
            list_input_devices,
            list_output_devices,
            get_device_snapshot,
            start_audio_capture,
            stop_audio_capture,
            set_capture_mute,
            get_capture_status,
            utterance_capture::start_utterance_capture,
            utterance_capture::stop_utterance_capture,
            utterance_capture::take_utterance
        ])
        .build()
}
