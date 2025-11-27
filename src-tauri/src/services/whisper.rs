use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use futures::StreamExt;
use std::fs::{self, File};
use std::io::Write;
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

#[derive(Clone, Debug)]
struct VadConfig {
    enabled: bool,
    silence_threshold_db: f32,
    silence_duration_ms: u64,
    min_chunk_duration_ms: u64,
}

pub struct WhisperState {
    stop_sender: Mutex<Option<Sender<()>>>,
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
            stop_sender: Mutex::new(None),
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

#[command]
async fn ensure_dependencies<R: Runtime>(app: AppHandle<R>) -> Result<(), String> {
    let app_data_dir = app
        .path_resolver()
        .app_data_dir()
        .ok_or("Failed to get app data directory")?;

    let whisper_dir = app_data_dir.join("whisper");
    let whisper_exe = whisper_dir.join("main.exe");
    let model_path = whisper_dir.join("ggml-base.en.bin");

    if whisper_exe.exists() && model_path.exists() {
        eprintln!("Whisper dependencies already exist");
        return Ok(());
    }

    fs::create_dir_all(&whisper_dir).map_err(|e| e.to_string())?;

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

        let file = File::open(&zip_path).map_err(|e| e.to_string())?;
        let mut archive = ZipArchive::new(file).map_err(|e| e.to_string())?;
        archive.extract(&whisper_dir).map_err(|e| e.to_string())?;

        fs::remove_file(&zip_path).ok();
        eprintln!("Whisper.cpp extracted successfully");
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
        eprintln!("[Whisper] Converting stereo to mono ({} samples)", audio_data.len());
        audio_data
            .chunks_exact(2)
            .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
            .collect::<Vec<f32>>()
    } else {
        audio_data
    };

    eprintln!("[Whisper] Mono data: {} samples", mono_data.len());

    // Resample to 16kHz if needed
    let resampled_data = if sample_rate != 16000 {
        eprintln!("[Whisper] Resampling from {}Hz to 16000Hz", sample_rate);
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

    eprintln!("[Whisper] Final audio: {} samples at 16kHz mono", resampled_data.len());

    // Check audio levels
    let final_rms = calculate_rms_db(&resampled_data);
    eprintln!("[Whisper] Final audio RMS: {:.2} dB", final_rms);

    if final_rms < -60.0 {
        eprintln!("[Whisper] WARNING: Audio is very quiet (< -60dB), likely silence!");
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
    let whisper_dir = whisper_exe.parent().unwrap();
    const CREATE_NO_WINDOW: u32 = 0x08000000;
    let output = Command::new(&whisper_exe)
        .current_dir(whisper_dir)
        .args(&["-m", model_path.to_str().unwrap(), "-f", temp_wav.to_str().unwrap(), "-nt", "-l", "en"])
        .creation_flags(CREATE_NO_WINDOW)
        .output()
        .map_err(|e| format!("Failed to run whisper: {}", e))?;

    fs::remove_file(&temp_wav).ok();

    eprintln!("[Whisper] Exit status: {}", output.status);
    eprintln!("[Whisper] Stdout: {}", String::from_utf8_lossy(&output.stdout));
    eprintln!("[Whisper] Stderr: {}", String::from_utf8_lossy(&output.stderr));

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
) -> Result<(), String> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or("No input device available")?;

    let config = device.default_input_config().map_err(|e| e.to_string())?;
    eprintln!("[Whisper] Default input config: {:?}", config);

    // Store the actual sample rate and channels
    {
        let mut sr = state.sample_rate.lock().unwrap();
        *sr = config.sample_rate().0;
    }
    {
        let mut ch = state.channels.lock().unwrap();
        *ch = config.channels();
    }

    // Update VAD config
    {
        let mut vad_cfg = state.vad_config.lock().unwrap();
        *vad_cfg = VadConfig {
            enabled: vad_enabled,
            silence_threshold_db,
            silence_duration_ms,
            min_chunk_duration_ms,
        };
    }

    // Clear buffer and reset VAD state
    {
        let mut buffer = state.audio_buffer.lock().unwrap();
        buffer.clear();
    }
    {
        let mut silent_frames = state.consecutive_silent_frames.lock().unwrap();
        *silent_frames = 0;
    }
    {
        let mut last_trans = state.last_transcription_time.lock().unwrap();
        *last_trans = Instant::now();
    }

    // Set recording flag to true
    {
        let mut is_rec = state.is_recording.lock().unwrap();
        *is_rec = true;
    }

    // Create stop channel
    let (tx, rx) = channel();
    {
        let mut stop_sender = state.stop_sender.lock().unwrap();
        *stop_sender = Some(tx);
    }

    let buffer_clone = state.audio_buffer.clone();
    let sample_rate_clone = state.sample_rate.clone();
    let channels_clone = state.channels.clone();
    let is_recording_clone = state.is_recording.clone();
    let vad_config_clone = state.vad_config.clone();
    let silent_frames_clone = state.consecutive_silent_frames.clone();
    let last_transcription_clone = state.last_transcription_time.clone();
    let app_clone = app.clone();

    let last_chunk_time = Arc::new(Mutex::new(Instant::now()));
    let last_chunk_time_clone = last_chunk_time.clone();

    let err_fn = move |err| {
        eprintln!("[Whisper] Audio stream error: {}", err);
    };

    let stream = match config.sample_format() {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                let is_recording = *is_recording_clone.lock().unwrap();
                if !is_recording {
                    return;
                }

                {
                    let mut buffer = buffer_clone.lock().unwrap();
                    buffer.extend_from_slice(data);
                }

                let vad_config = vad_config_clone.lock().unwrap().clone();
                let sample_rate = *sample_rate_clone.lock().unwrap();
                let channels = *channels_clone.lock().unwrap();

                let should_process = if vad_config.enabled {
                    let rms_db = calculate_rms_db(data);
                    let is_silent = rms_db < vad_config.silence_threshold_db;

                    if is_silent {
                        let mut silent_frames = silent_frames_clone.lock().unwrap();
                        *silent_frames += 1;
                    } else {
                        let mut silent_frames = silent_frames_clone.lock().unwrap();
                        *silent_frames = 0;
                    }

                    let silent_frames_count = *silent_frames_clone.lock().unwrap();
                    let callback_duration_ms = (data.len() as f32 / channels as f32 / sample_rate as f32 * 1000.0) as u64;
                    let silent_duration_ms = silent_frames_count as u64 * callback_duration_ms;

                    let last_trans_time = last_transcription_clone.lock().unwrap();
                    let time_since_last_trans = last_trans_time.elapsed().as_millis() as u64;
                    drop(last_trans_time);

                    if silent_duration_ms >= vad_config.silence_duration_ms && time_since_last_trans >= vad_config.min_chunk_duration_ms {
                        let mut silent_frames = silent_frames_clone.lock().unwrap();
                        *silent_frames = 0;

                        let mut last_trans = last_transcription_clone.lock().unwrap();
                        *last_trans = Instant::now();

                        eprintln!(
                            "[Whisper VAD] Silence detected after {}ms, triggering transcription",
                            time_since_last_trans
                        );
                        true
                    } else {
                        false
                    }
                } else {
                    let mut last_time = last_chunk_time_clone.lock().unwrap();
                    if last_time.elapsed().as_secs() >= CHUNK_DURATION_SECS {
                        *last_time = Instant::now();
                        true
                    } else {
                        false
                    }
                };

                if should_process {
                    let is_still_recording = *is_recording_clone.lock().unwrap();
                    if !is_still_recording {
                        eprintln!("[Whisper] Recording stopped, not processing chunk");
                        return;
                    }

                    let chunk = {
                        let mut buffer = buffer_clone.lock().unwrap();
                        let chunk = buffer.clone();
                        buffer.clear();
                        chunk
                    };

                    if chunk.len() < 3200 {
                        eprintln!("[Whisper] Skipping small chunk ({} samples)", chunk.len());
                        return;
                    }

                    let app_handle = app_clone.clone();
                    let sample_rate = *sample_rate_clone.lock().unwrap();
                    let channels = *channels_clone.lock().unwrap();

                    thread::spawn(move || match transcribe_chunk(&app_handle, chunk, sample_rate, channels) {
                        Ok(text) if !text.trim().is_empty() => {
                            eprintln!("[Whisper] Emitting partial result: {}", text);
                            let _ = app_handle.emit_all("whisper:partial_result", text);
                        }
                        Ok(_) => {
                            eprintln!("[Whisper] Empty transcription");
                        }
                        Err(e) => {
                            eprintln!("[Whisper] Transcription error: {}", e);
                        }
                    });
                }
            },
            err_fn,
            None,
        ),
        _ => {
            return Err("Unsupported sample format".to_string());
        }
    }
    .map_err(|e| e.to_string())?;

    stream.play().map_err(|e| e.to_string())?;

    thread::spawn(move || {
        let _ = rx.recv();
        eprintln!("[Whisper] Stop signal received");
    });

    std::mem::forget(stream);

    eprintln!("[Whisper] Recording started");
    Ok(())
}

#[command]
async fn stop_recording<R: Runtime>(_app: AppHandle<R>, state: State<'_, WhisperState>) -> Result<String, String> {
    eprintln!("[Whisper] Stopping recording...");

    {
        let mut is_rec = state.is_recording.lock().unwrap();
        *is_rec = false;
    }

    {
        let mut stop_sender = state.stop_sender.lock().unwrap();
        if let Some(tx) = stop_sender.take() {
            let _ = tx.send(());
        } else {
            return Err("Not recording".to_string());
        }
    }

    thread::sleep(std::time::Duration::from_millis(200));

    {
        let mut buffer = state.audio_buffer.lock().unwrap();
        buffer.clear();
    }

    eprintln!("[Whisper] Recording stopped");
    Ok(String::new())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("whisper")
        .setup(|app| {
            app.manage(WhisperState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![ensure_dependencies, start_recording, stop_recording])
        .build()
}
