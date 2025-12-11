use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::os::windows::process::CommandExt;
use std::process::Command;
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};
use zip::ZipArchive;

const WHISPER_VERSION: &str = "v1.5.4";
const MODEL_URL: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin";
const CHUNK_DURATION_SECS: u64 = 5;

// Trusted SHA256 hashes (Trust On First Use verified)
const WHISPER_ZIP_HASH: &str = "9cb13bbe167e0947afedd7ff9766575c4324b3cd01b4267be2ba9648dc7e8cc9";
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
    let mut file = File::open(path).map_err(|e| format!("Failed to open file for verification: {}", e))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0; 4096];

    loop {
        let count = file.read(&mut buffer).map_err(|e| e.to_string())?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }

    let result = hasher.finalize();
    let hash_hex = hex::encode(result);

    if hash_hex.to_lowercase() != expected_hash.to_lowercase() {
        return Err(format!("Hash mismatch! Expected {}, got {}", expected_hash, hash_hex));
    }
    Ok(())
}

#[command]
async fn ensure_dependencies<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    let app_data_dir = app
        .path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;

    let whisper_dir = app_data_dir.join("whisper");
    let whisper_exe = whisper_dir.join("main.exe");
    let model_path = whisper_dir.join("ggml-base.en.bin");

    // Create dir if missing
    if !whisper_dir.exists() {
        fs::create_dir_all(&whisper_dir).map_err(|e| e.to_string())?;
    }

    if !whisper_exe.exists() {
        eprintln!("Downloading whisper.cpp...");
        let whisper_url = format!(
            "https://github.com/ggerganov/whisper.cpp/releases/download/{}/whisper-bin-x64.zip",
            WHISPER_VERSION
        );

        let response = reqwest::get(&whisper_url)
            .await
            .map_err(|e| format!("Failed to download whisper: {}", e))?;

        let bytes = response
            .bytes()
            .await
            .map_err(|e| format!("Failed to read whisper download: {}", e))?;

        let zip_path = whisper_dir.join("whisper.zip");
        let mut file = File::create(&zip_path).map_err(|e| e.to_string())?;
        file.write_all(&bytes).map_err(|e| e.to_string())?;

        // VERIFY ZIP HASH
        if let Err(e) = verify_file(&zip_path, WHISPER_ZIP_HASH) {
            fs::remove_file(&zip_path).ok();
            return Err(format!("Security check failed for Whisper binary: {}", e));
        }

        let file = File::open(&zip_path).map_err(|e| e.to_string())?;
        let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
        archive.extract(&whisper_dir).map_err(|e| e.to_string())?;

        fs::remove_file(&zip_path).ok();
        eprintln!("Whisper.cpp extracted and verified successfully");
    }

    if !model_path.exists() {
        eprintln!("Downloading Whisper model...");

        let response = reqwest::get(MODEL_URL)
            .await
            .map_err(|e| format!("Failed to download model: {}", e))?;

        let total_size = response.content_length().unwrap_or(0);
        let mut downloaded: u64 = 0;
        let mut stream = response.bytes_stream();
        let mut file = File::create(&model_path).map_err(|e| e.to_string())?;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.map_err(|e| format!("Failed to read chunk: {}", e))?;
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            downloaded += chunk.len() as u64;

            if total_size > 0 {
                let progress = (downloaded as f64 / total_size as f64) * 100.0;
                app.emit_all(
                    "whisper:download_progress",
                    ProgressPayload {
                        file: "ggml-base.en.bin".to_string(),
                        progress,
                    },
                )
                .ok();
            }
        }

        // VERIFY MODEL HASH
        if let Err(e) = verify_file(&model_path, MODEL_HASH) {
            fs::remove_file(&model_path).ok();
            return Err(format!("Security check failed for Whisper model: {}", e));
        }

        eprintln!("Model downloaded successfully");
    }

    Ok(())
}

fn transcribe_chunk<R: Runtime>(app: &AppHandle<R>, audio_data: Vec<f32>, sample_rate: u32, channels: u16) -> Result<String, String> {
    let app_data_dir = app
        .path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;

    let whisper_dir = app_data_dir.join("whisper");
    let whisper_exe = whisper_dir.join("main.exe");
    let model_path = whisper_dir.join("ggml-base.en.bin");
    let temp_wav = whisper_dir.join("temp_chunk.wav");

    // Convert stereo to mono if needed
    let mono_data = if channels == 2 {
        // eprintln!("[Whisper] Converting stereo to mono ({} samples)", audio_data.len());
        audio_data
            .chunks_exact(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect::<Vec<f32>>()
    } else {
        audio_data
    };

    // eprintln!("[Whisper] Mono data: {} samples", mono_data.len());

    // Resample to 16kHz if needed
    let resampled_data = if sample_rate != 16000 {
        // eprintln!("[Whisper] Resampling from {}Hz to 16000Hz", sample_rate);
        let ratio = 16000.0 / sample_rate as f32;
        let target_len = (mono_data.len() as f32 * ratio) as usize;
        let mut resampled = Vec::with_capacity(target_len);

        for i in 0..target_len {
            let src_idx = (i as f32 / ratio) as usize;
            if src_idx < mono_data.len() {
                resampled.push(mono_data[src_idx]);
            }
        }
        resampled
    } else {
        mono_data
    };

    // eprintln!("[Whisper] Final audio: {} samples at 16kHz mono", resampled_data.len());

    // Check audio levels
    let final_rms = calculate_rms_db(&resampled_data);
    // eprintln!("[Whisper] Final audio RMS: {:.2} dB", final_rms);

    if final_rms < -60.0 {
        // eprintln!("[Whisper] WARNING: Audio is very quiet (< -60dB), likely silence!");
    }

    // Save as WAV (16-bit PCM as required by Whisper)
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(&temp_wav, spec).map_err(|e| e.to_string())?;
    for sample in resampled_data {
        // Convert f32 to i16 (clamp to [-1.0, 1.0] and scale to i16 range)
        let sample_clamped = sample.max(-1.0).min(1.0);
        let sample_i16 = (sample_clamped * 32767.0) as i16;
        writer.write_sample(sample_i16).map_err(|e| e.to_string())?;
    }
    writer.finalize().map_err(|e| e.to_string())?;

    // Run whisper
    let whisper_dir = whisper_exe.parent().ok_or("Failed to get whisper dir")?;
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    let output = Command::new(&whisper_exe)
        .current_dir(whisper_dir)
        .args(&[
            "-m",
            model_path.to_str().ok_or("Invalid model path")?,
            "-f",
            temp_wav.to_str().ok_or("Invalid wav path")?,
            "-nt",
            "-l",
            "en",
        ])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("Failed to run whisper: {}", e))?;

    fs::remove_file(&temp_wav).ok();

    // eprintln!("[Whisper] Exit status: {}", output.status);
    // eprintln!("[Whisper] Stdout: {}", String::from_utf8_lossy(&output.stdout));
    // eprintln!("[Whisper] Stderr: {}", String::from_utf8_lossy(&output.stderr));

    if !output.status.success() {
        return Err(format!("Whisper failed: {}", String::from_utf8_lossy(&output.stderr)));
    }

    let text = String::from_utf8_lossy(&output.stdout).trim().to_string();
    Ok(text)
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
    // 1. Initialize State (Common for both local and remote)

    // Store default sample rate/channels for remote if local is skipped.
    // Ideally, remote sources tells us this, but for now we default to 16k/mono or keep existing.
    // If capture_local is true, we get it from the device.

    let mut config_sample_rate = 16000;
    let mut config_channels = 1;

    let device_opt = if capture_local {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .ok_or("No input device available")?;
        let config = device.default_input_config().map_err(|e| e.to_string())?;

        config_sample_rate = config.sample_rate().0;
        config_channels = config.channels();
        Some((device, config))
    } else {
        None
    };

    // Store sample rate and channels
    {
        let mut sr = state
            .sample_rate
            .lock()
            .expect("Failed to lock sample_rate");
        *sr = config_sample_rate;
    }
    {
        let mut ch = state.channels.lock().expect("Failed to lock channels");
        *ch = config_channels;
    }

    // Update VAD config
    {
        let mut vad_cfg = state.vad_config.lock().expect("Failed to lock vad_config");
        *vad_cfg = VadConfig {
            enabled: vad_enabled,
            silence_threshold_db,
            silence_duration_ms,
            min_chunk_duration_ms,
        };
    }

    // Clear buffer and reset VAD state
    {
        let mut buffer = state
            .audio_buffer
            .lock()
            .expect("Failed to lock audio_buffer");
        buffer.clear();
    }
    {
        let mut silent_frames = state
            .consecutive_silent_frames
            .lock()
            .expect("Failed to lock silent_frames");
        *silent_frames = 0;
    }
    {
        let mut last_trans = state
            .last_transcription_time
            .lock()
            .expect("Failed to lock last_transcription_time");
        *last_trans = Instant::now();
    }

    // Set recording flag to true
    {
        let mut is_rec = state
            .is_recording
            .lock()
            .expect("Failed to lock is_recording");
        *is_rec = true;
    }

    // Create stop channel
    let (tx, rx) = channel();
    {
        let mut stop_sender = state
            .stop_sender
            .lock()
            .expect("Failed to lock stop_sender");
        *stop_sender = Some(tx);
    }

    // 2. Start Local Capture (if requested)
    if capture_local {
        let (device, config) = device_opt.unwrap();
        let err_fn = move |err| {
            eprintln!("[Whisper] Audio stream error: {}", err);
        };

        let state_clone_for_callback = state.inner().clone();
        let app_clone_for_callback = app.clone();

        // Spawn a thread to manage the stream lifetime
        thread::spawn(move || {
            let stream = match config.sample_format() {
                cpal::SampleFormat::F32 => device.build_input_stream(
                    &config.into(),
                    move |data: &[f32], _: &cpal::InputCallbackInfo| {
                        process_audio_chunk(&state_clone_for_callback, &app_clone_for_callback, data.to_vec());
                    },
                    err_fn,
                    None,
                ),
                _ => {
                    eprintln!("[Whisper] Unsupported sample format");
                    return;
                }
            };

            if let Ok(stream) = stream {
                if let Err(e) = stream.play() {
                    eprintln!("[Whisper] Failed to play stream: {}", e);
                    return;
                }
                // Wait for stop signal
                let _ = rx.recv();
                // Stream is dropped here, stopping recording
            }
        });
    } else {
        // For remote mode, we just spawn a thread to wait for the stop signal
        // so we can clean up cleanly when stop_recording is called.
        thread::spawn(move || {
            let _ = rx.recv();
            // Just exit thread when stopped
        });
    }

    Ok(())
}

// Public helper to process audio chunks (used by both local and remote)
pub fn process_audio_chunk<R: Runtime>(state: &WhisperState, app: &AppHandle<R>, data: Vec<f32>) {
    // eprintln!("[Whisper] Processing chunk: {} samples", data.len());
    let is_recording = *state
        .is_recording
        .lock()
        .expect("Failed to lock is_recording");

    if !is_recording {
        // Ignore chunks if we are not explicitly recording (state initialized)
        return;
    }

    {
        let mut buffer = state
            .audio_buffer
            .lock()
            .expect("Failed to lock audio_buffer");
        buffer.extend_from_slice(&data);
    }

    let vad_config = state
        .vad_config
        .lock()
        .expect("Failed to lock vad_config")
        .clone();
    let sample_rate = *state
        .sample_rate
        .lock()
        .expect("Failed to lock sample_rate");
    let channels = *state.channels.lock().expect("Failed to lock channels");

    let should_process = if vad_config.enabled {
        let rms_db = calculate_rms_db(&data);
        let is_silent = rms_db < vad_config.silence_threshold_db;

        eprintln!(
            "[Whisper] RMS: {:.2} dB, Silent: {}, Threshold: {:.2} dB",
            rms_db, is_silent, vad_config.silence_threshold_db
        );

        if is_silent {
            let mut silent_frames = state
                .consecutive_silent_frames
                .lock()
                .expect("Failed to lock silent_frames");
            *silent_frames += 1;
        } else {
            let mut silent_frames = state
                .consecutive_silent_frames
                .lock()
                .expect("Failed to lock silent_frames");
            *silent_frames = 0;
        }

        let silent_frames_count = *state
            .consecutive_silent_frames
            .lock()
            .expect("Failed to lock silent_frames");
        // Estimate duration based on chunk size. data.len() is samples.
        // Duration = samples / channels / sample_rate
        let chunk_duration_ms = (data.len() as f32 / channels as f32 / sample_rate as f32 * 1000.0) as u64;

        // We approximate silent duration by counting consecutive silent chunks.
        // This assumes chunks are roughly equal size, which is true for cpal but might vary for remote.
        // A better approach would be to accumulate actual time.
        // For now, we'll use the current chunk's duration * count.
        let silent_duration_ms = silent_frames_count as u64 * chunk_duration_ms;

        let last_trans_time = state
            .last_transcription_time
            .lock()
            .expect("Failed to lock last_transcription_time");
        let time_since_last_trans = last_trans_time.elapsed().as_millis() as u64;
        drop(last_trans_time);

        eprintln!(
            "[Whisper] Silent Duration: {} ms, Time Since Last: {} ms",
            silent_duration_ms, time_since_last_trans
        );

        if silent_duration_ms >= vad_config.silence_duration_ms && time_since_last_trans >= vad_config.min_chunk_duration_ms {
            eprintln!("[Whisper] Triggering transcription (VAD)");
            let mut silent_frames = state
                .consecutive_silent_frames
                .lock()
                .expect("Failed to lock silent_frames");
            *silent_frames = 0;

            let mut last_trans = state
                .last_transcription_time
                .lock()
                .expect("Failed to lock last_transcription_time");
            *last_trans = Instant::now();
            true
        } else {
            false
        }
    } else {
        // For non-VAD, we need a different trigger.
        // The original code used a timer in the closure.
        // Here we can check buffer size or time since last transcription.
        let last_trans_time = state
            .last_transcription_time
            .lock()
            .expect("Failed to lock last_transcription_time");
        if last_trans_time.elapsed().as_secs() >= CHUNK_DURATION_SECS {
            drop(last_trans_time);
            let mut last_trans = state
                .last_transcription_time
                .lock()
                .expect("Failed to lock last_transcription_time");
            *last_trans = Instant::now();
            true
        } else {
            false
        }
    };

    if should_process {
        let is_still_recording = *state
            .is_recording
            .lock()
            .expect("Failed to lock is_recording");
        if !is_still_recording {
            return;
        }

        let chunk = {
            let mut buffer = state
                .audio_buffer
                .lock()
                .expect("Failed to lock audio_buffer");
            let chunk = buffer.clone();
            buffer.clear();
            chunk
        };

        if chunk.len() < 3200 {
            // 0.2s at 16kHz
            return;
        }

        let app_handle = app.clone();

        thread::spawn(move || match transcribe_chunk(&app_handle, chunk, sample_rate, channels) {
            Ok(text) if !text.trim().is_empty() => {
                let _ = app_handle.emit_all("whisper:partial_result", text);
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[Whisper] Transcription error: {}", e);
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
    // eprintln!("[Whisper] Stopping recording...");

    {
        let mut is_rec = state
            .is_recording
            .lock()
            .expect("Failed to lock is_recording");
        *is_rec = false;
    }

    {
        let mut stop_sender = state
            .stop_sender
            .lock()
            .expect("Failed to lock stop_sender");
        if let Some(tx) = stop_sender.take() {
            let _ = tx.send(());
        } else {
            return Err("Not recording".to_string());
        }
    }

    thread::sleep(std::time::Duration::from_millis(200));

    {
        let mut buffer = state
            .audio_buffer
            .lock()
            .expect("Failed to lock audio_buffer");
        buffer.clear();
    }

    // eprintln!("[Whisper] Recording stopped");
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
