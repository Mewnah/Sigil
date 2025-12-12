use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";
const CHUNK_DURATION_SECS: u64 = 5;
const MODEL_HASH: &str = "a03779c86df3323075f5e796cb2ce5029f00ec8869eee3fdfb897afe36c6d002";

#[derive(Clone, Debug)]
struct VadConfig {
    enabled: bool,
    silence_threshold_db: f32,
    silence_duration_ms: u64,
    min_chunk_duration_ms: u64,
}

#[derive(Clone)]
pub struct WhisperState {
    stop_sender: Arc<Mutex<Option<Sender<()>>>>,
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: Arc<Mutex<u32>>,
    channels: Arc<Mutex<u16>>,
    is_recording: Arc<Mutex<bool>>,
    vad_config: Arc<Mutex<VadConfig>>,
    consecutive_silent_frames: Arc<Mutex<u32>>,
    last_transcription_time: Arc<Mutex<Instant>>,
    ctx: Arc<Mutex<Option<WhisperContext>>>,
}

impl WhisperState {
    fn new() -> Self {
        Self {
            stop_sender: Arc::new(Mutex::new(None)),
            audio_buffer: Arc::new(Mutex::new(Vec::new())),
            sample_rate: Arc::new(Mutex::new(16000)),
            channels: Arc::new(Mutex::new(1)),
            is_recording: Arc::new(Mutex::new(false)),
            vad_config: Arc::new(Mutex::new(VadConfig {
                enabled: true,
                silence_threshold_db: -40.0,
                silence_duration_ms: 1500,
                min_chunk_duration_ms: 1000,
            })),
            consecutive_silent_frames: Arc::new(Mutex::new(0)),
            last_transcription_time: Arc::new(Mutex::new(Instant::now())),
            ctx: Arc::new(Mutex::new(None)),
        }
    }
}

#[derive(Clone, serde::Serialize)]
struct ProgressPayload {
    file: String,
    progress: f64,
}

fn calculate_rms_db(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return -100.0;
    }
    let sum_squares: f32 = samples.iter().map(|s| s * s).sum();
    let rms = (sum_squares / samples.len() as f32).sqrt();
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -100.0
    }
}

fn verify_file(path: &std::path::Path, expected_hash: &str) -> Result<(), String> {
    let mut file = File::open(path).map_err(|e| format!("Failed to open: {}", e))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 4096];
    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    let hash_hex = hex::encode(hasher.finalize());
    if hash_hex.to_lowercase() != expected_hash.to_lowercase() {
        return Err(format!("Hash mismatch! Expected {}, got {}", expected_hash, hash_hex));
    }
    Ok(())
}

#[command]
async fn ensure_dependencies<R: Runtime>(app: AppHandle<R>, state: State<'_, WhisperState>) -> Result<(), String> {
    let app_data_dir = app
        .path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;
    let whisper_dir = app_data_dir.join("whisper");
    let model_path = whisper_dir.join("ggml-base.en.bin");

    if !whisper_dir.exists() {
        fs::create_dir_all(&whisper_dir).map_err(|e| e.to_string())?;
    }

    if !model_path.exists() {
        let response = reqwest::get(MODEL_URL)
            .await
            .map_err(|e| format!("Model download failed: {}", e))?;
        let total = response.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();
        let mut file = File::create(&model_path).map_err(|e| e.to_string())?;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| e.to_string())?;
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            downloaded += chunk.len() as u64;
            if total > 0 {
                let _ = app.emit_all(
                    "whisper:download_progress",
                    ProgressPayload {
                        file: "ggml-base.en.bin".to_string(),
                        progress: (downloaded as f64 / total as f64) * 100.0,
                    },
                );
            }
        }
        if let Err(e) = verify_file(&model_path, MODEL_HASH) {
            fs::remove_file(&model_path).ok();
            return Err(e);
        }
    }

    let mut ctx_guard = state.ctx.lock().map_err(|_| "Failed to lock ctx")?;
    if ctx_guard.is_none() {
        let path_str = model_path.to_str().ok_or("Invalid model path")?;
        let ctx =
            WhisperContext::new_with_params(path_str, WhisperContextParameters::default()).map_err(|e| format!("Failed to load context: {}", e))?;
        *ctx_guard = Some(ctx);
    }
    Ok(())
}

fn transcribe_chunk(state: &WhisperState, audio_data: Vec<f32>, sample_rate: u32, channels: u16) -> Result<String, String> {
    let mono_data = if channels == 2 {
        audio_data
            .chunks_exact(2)
            .map(|c| (c[0] + c[1]) / 2.0)
            .collect()
    } else {
        audio_data
    };

    let resampled = if sample_rate != 16000 {
        let ratio = 16000.0 / sample_rate as f32;
        let target = (mono_data.len() as f32 * ratio) as usize;
        let mut v = Vec::with_capacity(target);
        for i in 0..target {
            let idx = (i as f32 / ratio) as usize;
            if idx < mono_data.len() {
                v.push(mono_data[idx]);
            }
        }
        v
    } else {
        mono_data
    };

    let mut ctx_guard = state.ctx.lock().map_err(|_| "Failed to lock ctx")?;
    let ctx = ctx_guard.as_mut().ok_or("Model not loaded")?;

    // Create state (ephemeral)
    let mut w_state = ctx
        .create_state()
        .map_err(|e| format!("Failed to create state: {}", e))?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(4);
    params.set_language(Some("en"));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    w_state
        .full(params, &resampled)
        .map_err(|e| format!("Inference failed: {}", e))?;

    let num_segments = w_state.full_n_segments().map_err(|e| e.to_string())?;
    let mut text = String::new();
    for i in 0..num_segments {
        let segment = w_state
            .full_get_segment_text(i)
            .map_err(|e| e.to_string())?;
        text.push_str(&segment);
        text.push(' ');
    }
    Ok(text.trim().to_string())
}

#[command]
async fn start_recording<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, WhisperState>,
    vad_enabled: bool,
    silence_threshold_db: f32,
    silence_duration_ms: u64,
    min_chunk_duration_ms: u64,
    capture_local: bool,
) -> Result<(), String> {
    let mut config_sample_rate = 16000;
    let mut config_channels = 1;
    let device_opt = if capture_local {
        let host = cpal::default_host();
        let device = host.default_input_device().ok_or("No input device")?;
        let config = device.default_input_config().map_err(|e| e.to_string())?;
        config_sample_rate = config.sample_rate().0;
        config_channels = config.channels();
        Some((device, config))
    } else {
        None
    };

    {
        *state.sample_rate.lock().unwrap() = config_sample_rate;
        *state.channels.lock().unwrap() = config_channels;
        *state.vad_config.lock().unwrap() = VadConfig {
            enabled: vad_enabled,
            silence_threshold_db,
            silence_duration_ms,
            min_chunk_duration_ms,
        };
        state.audio_buffer.lock().unwrap().clear();
        *state.consecutive_silent_frames.lock().unwrap() = 0;
        *state.last_transcription_time.lock().unwrap() = Instant::now();
        *state.is_recording.lock().unwrap() = true;
    }

    let (tx, rx) = channel();
    *state.stop_sender.lock().unwrap() = Some(tx);

    if capture_local {
        let (device, config) = device_opt.unwrap();
        let state_clone = state.inner().clone();
        let app_clone = app.clone();
        thread::spawn(move || {
            let stream = match config.sample_format() {
                cpal::SampleFormat::F32 => device.build_input_stream(
                    &config.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| process_audio_chunk(&state_clone, &app_clone, data.to_vec()),
                    |e| eprintln!("Stream error: {}", e),
                    None,
                ),
                _ => return,
            };
            if let Ok(stream) = stream {
                if stream.play().is_ok() {
                    let _ = rx.recv();
                }
            }
        });
    } else {
        thread::spawn(move || {
            let _ = rx.recv();
        });
    }
    Ok(())
}

pub fn process_audio_chunk<R: Runtime>(state: &WhisperState, app: &AppHandle<R>, data: Vec<f32>) {
    if !*state.is_recording.lock().unwrap() {
        return;
    }
    state.audio_buffer.lock().unwrap().extend_from_slice(&data);

    let vad_config = state.vad_config.lock().unwrap().clone();
    let sample_rate = *state.sample_rate.lock().unwrap();
    let channels = *state.channels.lock().unwrap();

    let should_process = if vad_config.enabled {
        let rms = calculate_rms_db(&data);
        if rms < vad_config.silence_threshold_db {
            *state.consecutive_silent_frames.lock().unwrap() += 1;
        } else {
            *state.consecutive_silent_frames.lock().unwrap() = 0;
        }

        let silent_frames = *state.consecutive_silent_frames.lock().unwrap();
        let chunk_dur = (data.len() as f32 / channels as f32 / sample_rate as f32 * 1000.0) as u64;
        let silent_dur = silent_frames as u64 * chunk_dur;
        let last = state.last_transcription_time.lock().unwrap();
        let time = last.elapsed().as_millis() as u64;
        drop(last);

        if silent_dur >= vad_config.silence_duration_ms && time >= vad_config.min_chunk_duration_ms {
            *state.consecutive_silent_frames.lock().unwrap() = 0;
            *state.last_transcription_time.lock().unwrap() = Instant::now();
            true
        } else {
            false
        }
    } else {
        let last = state.last_transcription_time.lock().unwrap();
        if last.elapsed().as_secs() >= CHUNK_DURATION_SECS {
            drop(last);
            *state.last_transcription_time.lock().unwrap() = Instant::now();
            true
        } else {
            false
        }
    };

    if should_process {
        if !*state.is_recording.lock().unwrap() {
            return;
        }
        let mut buf = state.audio_buffer.lock().unwrap();
        let chunk = buf.clone();
        buf.clear();
        drop(buf);

        if chunk.len() < 3200 {
            return;
        }
        let app = app.clone();
        let state_clone = state.clone();

        thread::spawn(move || {
            if let Ok(text) = transcribe_chunk(&state_clone, chunk, sample_rate, channels) {
                if !text.trim().is_empty() {
                    let _ = app.emit_all("whisper:partial_result", text);
                }
            }
        });
    }
}

#[command]
async fn feed_audio_chunk<R: Runtime>(app: AppHandle<R>, state: State<'_, WhisperState>, chunk: Vec<f32>) -> Result<(), String> {
    process_audio_chunk(&state, &app, chunk);
    Ok(())
}

#[command]
async fn stop_recording<R: Runtime>(_app: AppHandle<R>, state: State<'_, WhisperState>) -> Result<String, String> {
    *state.is_recording.lock().unwrap() = false;
    if let Some(tx) = state.stop_sender.lock().unwrap().take() {
        let _ = tx.send(());
    } else {
        return Err("Not recording".to_string());
    }
    thread::sleep(std::time::Duration::from_millis(200));
    state.audio_buffer.lock().unwrap().clear();
    Ok(String::new())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("whisper")
        .setup(|app| {
            app.manage(WhisperState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            ensure_dependencies,
            start_recording,
            stop_recording,
            feed_audio_chunk
        ])
        .build()
}
