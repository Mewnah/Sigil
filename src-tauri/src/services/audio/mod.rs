use std::io::Cursor;
use std::sync::mpsc::{channel, Sender};
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

#[derive(Serialize, Debug)]
pub struct AudioDevice {
    pub id: String,
    pub name: String,
}

// Shared state for audio capture
pub struct AudioCaptureState {
    stop_sender: Arc<Mutex<Option<Sender<()>>>>,
    is_capturing: Arc<Mutex<bool>>,
}

impl AudioCaptureState {
    fn new() -> Self {
        Self {
            stop_sender: Arc::new(Mutex::new(None)),
            is_capturing: Arc::new(Mutex::new(false)),
        }
    }
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
    // Stop any existing capture
    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }

    let host = cpal::default_host();

    // Select device
    let device = if let Some(ref name) = device_name {
        if !name.is_empty() {
            host.input_devices()
                .map_err(|e| e.to_string())?
                .find(|d| d.name().unwrap_or_default() == *name)
                .ok_or_else(|| format!("Input device '{}' not found", name))?
        } else {
            host.default_input_device()
                .ok_or("No default input device")?
        }
    } else {
        host.default_input_device()
            .ok_or("No default input device")?
    };

    let config = device.default_input_config().map_err(|e| e.to_string())?;
    let config_sample_rate = config.sample_rate().0;
    let config_channels = config.channels();

    let (tx, rx) = channel();
    *state.stop_sender.lock().unwrap() = Some(tx);
    *state.is_capturing.lock().unwrap() = true;

    let is_capturing = state.is_capturing.clone();
    let app_clone = app.clone();

    thread::spawn(move || {
        let stream = match config.sample_format() {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !*is_capturing.lock().unwrap() {
                        return;
                    }
                    let payload = AudioChunkPayload {
                        samples: data.to_vec(),
                        sample_rate: config_sample_rate,
                        channels: config_channels,
                    };
                    let _ = app_clone.emit("audio:chunk", payload);
                },
                |e| eprintln!("[AudioCapture] Stream error: {}", e),
                None,
            ),
            cpal::SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !*is_capturing.lock().unwrap() {
                        return;
                    }
                    // Convert i16 to f32
                    let samples: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let payload = AudioChunkPayload {
                        samples,
                        sample_rate: config_sample_rate,
                        channels: config_channels,
                    };
                    let _ = app_clone.emit("audio:chunk", payload);
                },
                |e| eprintln!("[AudioCapture] Stream error: {}", e),
                None,
            ),
            _ => return,
        };

        if let Ok(stream) = stream {
            if stream.play().is_ok() {
                let _ = rx.recv(); // Block until stop signal
            }
        }
    });

    Ok(())
}

#[command]
pub fn stop_audio_capture(state: State<'_, AudioCaptureState>) -> Result<(), String> {
    *state.is_capturing.lock().unwrap() = false;
    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }
    Ok(())
}

fn get_output_stream(device_name: &str) -> Result<(OutputStream, OutputStreamHandle), String> {
    if device_name == "default" || device_name.is_empty() {
        OutputStream::try_default().map_err(|e| e.to_string())
    } else {
        let host = cpal::default_host();
        let devices = host.output_devices().map_err(|e| e.to_string())?;
        let device = devices
            .into_iter()
            .find(|d| d.name().unwrap_or_default() == device_name)
            .ok_or("Device not found")?;
        OutputStream::try_from_device(&device).map_err(|e| e.to_string())
    }
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
