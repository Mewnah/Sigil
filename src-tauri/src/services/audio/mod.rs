use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::thread;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rodio::{Decoder, OutputStream, OutputStreamHandle, Sink};
use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};

pub mod device_resolve;

pub use device_resolve::{resolve_input_device, resolve_output_device};

use crate::services::pcm_convert::build_input_stream_f32;

#[derive(Serialize, Debug)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
}

// Shared state for audio capture
pub struct AudioCaptureState {
    stop_sender: Arc<Mutex<Option<Sender<()>>>>,
    is_capturing: Arc<Mutex<bool>>,
    /// Bumped (under `stop_sender`'s lock) by every start and stop. A start re-checks it under the same lock
    /// before publishing, so a stop that lands while the device is still opening cancels that start. Each capture
    /// callback also compares it with its own start's epoch, so a superseded stream never emits.
    capture_epoch: Arc<AtomicU64>,
}

impl AudioCaptureState {
    fn new() -> Self {
        Self {
            stop_sender: Arc::new(Mutex::new(None)),
            is_capturing: Arc::new(Mutex::new(false)),
            capture_epoch: Arc::new(AtomicU64::new(0)),
        }
    }
}

/// Build an input stream for `device` in any cpal sample format on a dedicated thread (cpal streams are not
/// `Send`), start it, and keep it alive until `stop` receives a message or its sender is dropped.
/// Resolves once the stream is playing, or with the build/play error.
pub async fn spawn_input_stream<F>(
    device: cpal::Device,
    config: cpal::SupportedStreamConfig,
    stop: Receiver<()>,
    on_data: F,
    log_tag: &'static str,
) -> Result<(), String>
where
    F: FnMut(Vec<f32>) + Send + 'static,
{
    let (ready_tx, ready_rx) = tokio::sync::oneshot::channel::<Result<(), String>>();
    thread::spawn(move || {
        let started = build_input_stream_f32(&device, &config, on_data, move |e| eprintln!("[{log_tag}] Stream error: {e}"))
            .and_then(|stream| stream.play().map(|()| stream).map_err(|e| e.to_string()));
        match started {
            Ok(stream) => {
                if ready_tx.send(Ok(())).is_ok() {
                    let _ = stop.recv(); // Block until stop signal (or the sender is dropped)
                }
                drop(stream);
            }
            Err(e) => {
                let _ = ready_tx.send(Err(e));
            }
        }
    });
    ready_rx
        .await
        .map_err(|_| "Audio capture thread exited before the stream started".to_string())?
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
    let my_epoch = {
        let mut sender = state.stop_sender.lock().unwrap();
        *state.is_capturing.lock().unwrap() = false;
        if let Some(old) = sender.take() {
            let _ = old.send(());
        }
        state.capture_epoch.fetch_add(1, Ordering::SeqCst) + 1
    };

    let host = cpal::default_host();

    // Select device
    let device = resolve_input_device(&host, device_name.as_deref())?;

    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let config_sample_rate = config.sample_rate().0;
    let config_channels = config.channels();

    let (tx, rx) = channel();
    let is_capturing = state.is_capturing.clone();
    let capture_epoch = state.capture_epoch.clone();
    let app_clone = app.clone();

    spawn_input_stream(
        device,
        config,
        rx,
        move |samples| {
            // Muted until this capture is published below, and for good once a stop or a newer start bumps the
            // epoch: this stream can outlive that briefly while its thread drops it, and its chunks (labelled with
            // this device's rate/channels) must not reach a newer capture's listeners. Checked and emitted while
            // holding `is_capturing`, which a stop takes after bumping the epoch, so once `stop_audio_capture`
            // returns, nothing more is emitted from this stream.
            let Ok(capturing) = is_capturing.lock() else { return };
            if !*capturing || capture_epoch.load(Ordering::SeqCst) != my_epoch {
                return;
            }
            let payload = AudioChunkPayload {
                samples,
                sample_rate: config_sample_rate,
                channels: config_channels,
            };
            let _ = app_clone.emit("audio:chunk", payload);
        },
        "AudioCapture",
    )
    .await?;

    let mut sender = state.stop_sender.lock().unwrap();
    if state.capture_epoch.load(Ordering::SeqCst) != my_epoch {
        // A stop (or a newer start) landed while the device was opening. Dropping `tx` on return ends the
        // capture thread, which drops the stream and releases the microphone.
        return Err("cancelled".to_string());
    }
    if let Some(old) = sender.replace(tx) {
        let _ = old.send(());
    }
    *state.is_capturing.lock().unwrap() = true;
    Ok(())
}

#[command]
pub fn stop_audio_capture(state: State<'_, AudioCaptureState>) -> Result<(), String> {
    let mut sender = state.stop_sender.lock().unwrap();
    // Cancels a start that is still opening its device (it re-checks the epoch under this lock).
    state.capture_epoch.fetch_add(1, Ordering::SeqCst);
    *state.is_capturing.lock().unwrap() = false;
    if let Some(sender) = sender.take() {
        let _ = sender.send(());
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
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            play_async,
            list_input_devices,
            list_output_devices,
            start_audio_capture,
            stop_audio_capture
        ])
        .build()
}
