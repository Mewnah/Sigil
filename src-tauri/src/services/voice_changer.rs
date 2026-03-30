use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use rubato::{FftFixedIn, Resampler};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};

const DEFAULT_PITCH_SEMITONES: f32 = 0.0;
const DEFAULT_FORMANT_SHIFT: f32 = 0.0;

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VoiceChangerPreset {
    pub id: String,
    pub name: String,
    pub pitch: f32,
    pub formant: f32,
}

fn get_presets() -> Vec<VoiceChangerPreset> {
    vec![
        VoiceChangerPreset {
            id: "default".to_string(),
            name: "Default".to_string(),
            pitch: 0.0,
            formant: 0.0,
        },
        VoiceChangerPreset {
            id: "deeper".to_string(),
            name: "Deeper".to_string(),
            pitch: -4.0,
            formant: -0.2,
        },
        VoiceChangerPreset {
            id: "higher".to_string(),
            name: "Higher".to_string(),
            pitch: 5.0,
            formant: 0.2,
        },
        VoiceChangerPreset {
            id: "chipmunk".to_string(),
            name: "Chipmunk".to_string(),
            pitch: 10.0,
            formant: 0.4,
        },
        VoiceChangerPreset {
            id: "deep_voice".to_string(),
            name: "Deep Voice".to_string(),
            pitch: -8.0,
            formant: -0.3,
        },
        VoiceChangerPreset {
            id: "subtle_lower".to_string(),
            name: "Subtle Lower".to_string(),
            pitch: -2.0,
            formant: 0.0,
        },
        VoiceChangerPreset {
            id: "subtle_higher".to_string(),
            name: "Subtle Higher".to_string(),
            pitch: 2.0,
            formant: 0.0,
        },
    ]
}

#[derive(Clone)]
pub struct VoiceChangerState {
    enabled: Arc<Mutex<bool>>,
    pitch_semitones: Arc<Mutex<f32>>,
    formant_shift: Arc<Mutex<f32>>,
    output_device: Arc<Mutex<String>>,
    stop_sender: Arc<Mutex<Option<Sender<()>>>>,
    is_running: Arc<Mutex<bool>>,
}

impl VoiceChangerState {
    fn new() -> Self {
        Self {
            enabled: Arc::new(Mutex::new(false)),
            pitch_semitones: Arc::new(Mutex::new(DEFAULT_PITCH_SEMITONES)),
            formant_shift: Arc::new(Mutex::new(DEFAULT_FORMANT_SHIFT)),
            output_device: Arc::new(Mutex::new("default".to_string())),
            stop_sender: Arc::new(Mutex::new(None)),
            is_running: Arc::new(Mutex::new(false)),
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct VoiceChangerParams {
    pub enabled: bool,
    pub pitch_semitones: f32,
    pub formant_shift: f32,
    pub output_device: String,
}

/// Get available presets
#[command]
fn get_voice_changer_presets() -> Vec<VoiceChangerPreset> {
    get_presets()
}

/// Get current voice changer parameters
#[command]
fn get_voice_changer_params(state: State<'_, VoiceChangerState>) -> Result<VoiceChangerParams, String> {
    Ok(VoiceChangerParams {
        enabled: *state.enabled.lock().map_err(|_| "Lock failed")?,
        pitch_semitones: *state.pitch_semitones.lock().map_err(|_| "Lock failed")?,
        formant_shift: *state.formant_shift.lock().map_err(|_| "Lock failed")?,
        output_device: state
            .output_device
            .lock()
            .map_err(|_| "Lock failed")?
            .clone(),
    })
}

/// Set voice changer parameters
#[command]
fn set_voice_changer_params(state: State<'_, VoiceChangerState>, params: VoiceChangerParams) -> Result<(), String> {
    *state.enabled.lock().map_err(|_| "Lock failed")? = params.enabled;
    *state.pitch_semitones.lock().map_err(|_| "Lock failed")? = params.pitch_semitones.clamp(-12.0, 12.0);
    *state.formant_shift.lock().map_err(|_| "Lock failed")? = params.formant_shift.clamp(-1.0, 1.0);
    *state.output_device.lock().map_err(|_| "Lock failed")? = params.output_device;
    Ok(())
}

/// Set enabled state
#[command]
fn set_voice_changer_enabled(state: State<'_, VoiceChangerState>, enabled: bool) -> Result<(), String> {
    *state.enabled.lock().map_err(|_| "Lock failed")? = enabled;
    Ok(())
}

/// Apply a preset
#[command]
fn apply_voice_changer_preset(state: State<'_, VoiceChangerState>, preset_id: String) -> Result<(), String> {
    let presets = get_presets();
    let preset = presets
        .iter()
        .find(|p| p.id == preset_id)
        .ok_or_else(|| format!("Preset '{}' not found", preset_id))?;

    *state.pitch_semitones.lock().map_err(|_| "Lock failed")? = preset.pitch;
    *state.formant_shift.lock().map_err(|_| "Lock failed")? = preset.formant;
    Ok(())
}

/// Calculate resample ratio from semitones
#[allow(dead_code)]
fn semitones_to_ratio(semitones: f32) -> f64 {
    2.0_f64.powf(semitones as f64 / 12.0)
}

/// Start the voice changer (capture -> process -> playback loop)
#[command]
async fn start_voice_changer<R: Runtime>(app: AppHandle<R>, state: State<'_, VoiceChangerState>, input_device: Option<String>) -> Result<(), String> {
    // Check if already running
    if *state.is_running.lock().map_err(|_| "Lock failed")? {
        return Err("Voice changer already running".to_string());
    }

    // Stop any existing
    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }

    // Setup stop signal
    let (tx, rx) = channel();
    *state.stop_sender.lock().unwrap() = Some(tx);
    *state.is_running.lock().unwrap() = true;

    // Clone state for the audio thread
    let pitch_semitones = state.pitch_semitones.clone();
    let enabled = state.enabled.clone();
    let is_running = state.is_running.clone();
    let output_device_name = state
        .output_device
        .lock()
        .map_err(|_| "Lock failed")?
        .clone();
    let input_device_name = input_device.clone();
    let app_clone = app.clone();

    // Spawn a single thread that owns all audio streams (they're not Send)
    thread::spawn(move || {
        let result = run_voice_changer_loop(rx, pitch_semitones, enabled, is_running.clone(), input_device_name, output_device_name);

        if let Err(e) = result {
            eprintln!("[VoiceChanger] Error: {}", e);
        }

        *is_running.lock().unwrap() = false;
        let _ = app_clone.emit("voice_changer:stopped", ());
    });

    let _ = app.emit("voice_changer:started", ());
    Ok(())
}

/// The main audio processing loop - runs in a dedicated thread
fn run_voice_changer_loop(
    rx: std::sync::mpsc::Receiver<()>,
    pitch_semitones: Arc<Mutex<f32>>,
    enabled: Arc<Mutex<bool>>,
    is_running: Arc<Mutex<bool>>,
    input_device_name: Option<String>,
    output_device_name: String,
) -> Result<(), String> {
    let host = cpal::default_host();

    // Get input device
    let input = if let Some(ref name) = input_device_name {
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

    // Get output device
    let output = if output_device_name.is_empty() || output_device_name == "default" {
        host.default_output_device()
            .ok_or("No default output device")?
    } else {
        host.output_devices()
            .map_err(|e| e.to_string())?
            .find(|d| d.name().unwrap_or_default() == output_device_name)
            .ok_or_else(|| format!("Output device '{}' not found", output_device_name))?
    };

    let input_config = input.default_input_config().map_err(|e| e.to_string())?;
    let output_config = output.default_output_config().map_err(|e| e.to_string())?;

    let sample_rate = input_config.sample_rate().0 as usize;
    let channels = input_config.channels() as usize;

    // Shared buffer for audio data
    let buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let buffer_clone = buffer.clone();

    // Clone for input callback
    let is_running_input = is_running.clone();

    // Output buffer for processed audio
    let output_buffer: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let output_buffer_clone = output_buffer.clone();

    // Input stream - capture audio to buffer
    let input_stream = match input_config.sample_format() {
        cpal::SampleFormat::F32 => input.build_input_stream(
            &input_config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                if !*is_running_input.lock().unwrap() {
                    return;
                }
                let mut buf = buffer_clone.lock().unwrap();
                buf.extend_from_slice(data);
            },
            |e| eprintln!("[VoiceChanger] Input error: {}", e),
            None,
        ),
        cpal::SampleFormat::I16 => {
            let is_running_i16 = is_running.clone();
            let buffer_i16 = buffer.clone();
            input.build_input_stream(
                &input_config.into(),
                move |data: &[i16], _: &cpal::InputCallbackInfo| {
                    if !*is_running_i16.lock().unwrap() {
                        return;
                    }
                    let samples: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                    let mut buf = buffer_i16.lock().unwrap();
                    buf.extend_from_slice(&samples);
                },
                |e| eprintln!("[VoiceChanger] Input error: {}", e),
                None,
            )
        }
        _ => return Err("Unsupported sample format".to_string()),
    }
    .map_err(|e| e.to_string())?;

    // Output stream - play processed audio
    let output_stream = output
        .build_output_stream(
            &output_config.into(),
            move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                let mut buf = output_buffer_clone.lock().unwrap();
                for sample in data.iter_mut() {
                    *sample = if !buf.is_empty() { buf.remove(0) } else { 0.0 };
                }
            },
            |e| eprintln!("[VoiceChanger] Output error: {}", e),
            None,
        )
        .map_err(|e| e.to_string())?;

    // Start streams
    input_stream.play().map_err(|e| e.to_string())?;
    output_stream.play().map_err(|e| e.to_string())?;

    println!("[VoiceChanger] Started audio streams");

    // Main processing loop
    let chunk_size = 1024;
    let mut current_pitch = 0.0_f32;
    let mut resampler: Option<FftFixedIn<f32>> = None;

    loop {
        // Check stop signal
        if rx.try_recv().is_ok() || !*is_running.lock().unwrap() {
            break;
        }

        // Get pitch
        let pitch = *pitch_semitones.lock().unwrap();
        let is_enabled = *enabled.lock().unwrap();

        // Check if we need to recreate resampler
        if (pitch - current_pitch).abs() > 0.01 || resampler.is_none() {
            current_pitch = pitch;
            if pitch.abs() > 0.01 && is_enabled {
                if let Ok(r) = FftFixedIn::<f32>::new(sample_rate, sample_rate, chunk_size, 2, channels) {
                    resampler = Some(r);
                }
            } else {
                resampler = None;
            }
        }

        // Get input samples
        let samples: Vec<f32> = {
            let mut buf = buffer.lock().unwrap();
            if buf.len() < chunk_size * channels {
                drop(buf);
                thread::sleep(std::time::Duration::from_millis(5));
                continue;
            }
            buf.drain(..chunk_size * channels).collect()
        };

        // Process
        let processed = if is_enabled && resampler.is_some() && pitch.abs() > 0.01 {
            // Convert to channel-separated format for rubato
            let mut channel_data: Vec<Vec<f32>> = vec![Vec::new(); channels];
            for (i, sample) in samples.iter().enumerate() {
                channel_data[i % channels].push(*sample);
            }

            // Apply resampling for pitch shift
            if let Some(ref mut r) = resampler {
                match r.process(&channel_data, None) {
                    Ok(resampled) => {
                        // Interleave channels back
                        let len = resampled[0].len();
                        let mut output = Vec::with_capacity(len * channels);
                        for i in 0..len {
                            for ch in 0..channels {
                                output.push(resampled[ch].get(i).copied().unwrap_or(0.0));
                            }
                        }
                        output
                    }
                    Err(_) => samples,
                }
            } else {
                samples
            }
        } else {
            samples
        };

        // Push to output buffer
        {
            let mut out_buf = output_buffer.lock().unwrap();
            out_buf.extend(processed);
        }
    }

    println!("[VoiceChanger] Stopped audio streams");
    Ok(())
}

/// Stop the voice changer
#[command]
fn stop_voice_changer<R: Runtime>(app: AppHandle<R>, state: State<'_, VoiceChangerState>) -> Result<(), String> {
    *state.is_running.lock().unwrap() = false;
    if let Some(sender) = state.stop_sender.lock().unwrap().take() {
        let _ = sender.send(());
    }
    let _ = app.emit("voice_changer:stopped", ());
    Ok(())
}

/// Check if voice changer is running
#[command]
fn is_voice_changer_running(state: State<'_, VoiceChangerState>) -> bool {
    *state.is_running.lock().unwrap()
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("voice-changer")
        .setup(|app, _api| {
            app.manage(VoiceChangerState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_voice_changer_presets,
            get_voice_changer_params,
            set_voice_changer_params,
            set_voice_changer_enabled,
            apply_voice_changer_preset,
            start_voice_changer,
            stop_voice_changer,
            is_voice_changer_running
        ])
        .build()
}
