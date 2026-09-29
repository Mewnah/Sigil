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

pub mod device_resolve;
pub mod speech_capture;
pub mod stream;
pub mod utterance_capture;

pub use device_resolve::{resolve_input_device, resolve_output_device};

use stream::{open_input_stream, InputStream};
use utterance_capture::UtteranceCaptureState;

#[derive(Serialize, Debug)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
}

/// `audio:chunk` batching period (the callback only fills a ring; a worker emits every ~20 ms).
const CHUNK_EMIT_PERIOD: Duration = Duration::from_millis(20);
/// Ring capacity of the raw chunk capture, in seconds of interleaved device audio.
const CHUNK_RING_SECONDS: usize = 1;

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
    stream: InputStream,
    stop: Arc<AtomicBool>,
    emitter: Thread,
}

impl ChunkCapture {
    fn signal_stop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.emitter.unpark();
        self.stream.signal_stop();
    }

    async fn stop(self) {
        self.stop.store(true, Ordering::Release);
        self.emitter.unpark();
        let ChunkCapture { stream, .. } = self;
        stream.stop().await;
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

#[command]
pub fn list_input_devices() -> Result<Vec<AudioDevice>, String> {
    let host = cpal::default_host();
    let devices = host.input_devices().map_err(|e| e.to_string())?;

    let mut result = Vec::new();
    for device in devices {
        if let Ok(name) = device.name() {
            result.push(AudioDevice { id: name.clone(), name });
        }
    }
    Ok(result)
}

#[command]
pub fn list_output_devices() -> Result<Vec<AudioDevice>, String> {
    let host = cpal::default_host();
    let devices = host.output_devices().map_err(|e| e.to_string())?;

    let mut result = Vec::new();
    for device in devices {
        if let Ok(name) = device.name() {
            result.push(AudioDevice { id: name.clone(), name });
        }
    }
    Ok(result)
}

#[derive(Clone, Serialize)]
struct AudioChunkPayload {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

#[command]
pub async fn start_audio_capture<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, AudioCaptureState>,
    device_name: Option<String>,
    _sample_rate: Option<u32>,
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

    let host = cpal::default_host();
    let device = resolve_input_device(&host, device_name.as_deref())?;
    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let sample_rate = config.sample_rate().0;
    let channels = config.channels();
    let frame = usize::from(channels.max(1));
    let capacity = (sample_rate as usize * frame * CHUNK_RING_SECONDS).max(8192);
    let (mut producer, mut consumer) = rtrb::RingBuffer::<f32>::new(capacity);
    let stop = Arc::new(AtomicBool::new(false));

    // Emitter: batches the ring into `audio:chunk` events. Emits only while holding the stop lock with
    // this start's epoch current, so once `stop_audio_capture` returns nothing more is emitted, and a
    // superseded capture's chunks (labelled with its device's rate/channels) never reach a newer one.
    let emitter = {
        let stop = Arc::clone(&stop);
        let active = Arc::clone(&state.active);
        let epoch = Arc::clone(&state.capture_epoch);
        thread::Builder::new()
            .name("sigil-audio-chunk".into())
            .spawn(move || loop {
                if stop.load(Ordering::Acquire) {
                    return;
                }
                let available = consumer.slots() / frame * frame;
                if available > 0 {
                    let mut samples = vec![0.0f32; available];
                    let _ = consumer.pop_entire_slice(&mut samples);
                    let _guard = lock_active(&active);
                    if epoch.load(Ordering::SeqCst) != my_epoch || stop.load(Ordering::Acquire) {
                        return;
                    }
                    let _ = app.emit(
                        "audio:chunk",
                        AudioChunkPayload {
                            samples,
                            sample_rate,
                            channels,
                        },
                    );
                } else if consumer.is_abandoned() {
                    return;
                }
                thread::park_timeout(CHUNK_EMIT_PERIOD);
            })
            .map_err(|e| format!("Failed to start the audio chunk emitter: {e}"))?
            .thread()
            .clone()
    };

    let stream = open_input_stream(
        device,
        config,
        move |data: &[f32]| {
            // Whole frames only; when the emitter lags a full second the newest audio is dropped.
            let fits = if data.len() <= producer.slots() {
                data.len()
            } else {
                producer.slots() / frame * frame
            };
            if fits > 0 {
                let _ = producer.push_entire_slice(&data[..fits]);
            }
        },
        |e| tracing::warn!(target: "sigil::audio", "[AudioCapture] stream error: {e}"),
        "AudioCapture",
    )
    .await;
    let stream = match stream {
        Ok(stream) => stream,
        Err(e) => {
            stop.store(true, Ordering::Release);
            emitter.unpark();
            return Err(e);
        }
    };

    let mut capture = ChunkCapture { stream, stop, emitter };
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

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("audio")
        .setup(|app, _api| {
            app.manage(AudioCaptureState::new());
            app.manage(UtteranceCaptureState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            play_async,
            list_input_devices,
            list_output_devices,
            start_audio_capture,
            stop_audio_capture,
            utterance_capture::start_utterance_capture,
            utterance_capture::stop_utterance_capture,
            utterance_capture::take_utterance
        ])
        .build()
}
