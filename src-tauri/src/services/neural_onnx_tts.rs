//! In-process ONNX speech TTS: Hub download or local **speech package** folder (see
//! `docs/OPERATOR_REFERENCE.md#local-speech-package-onnx`).
//! Implementation uses the `kittentts` crate for the supported package format.

use crate::services::neural_audio::ort_session::ensure_ort_environment;

use crate::services::voice_changer::{push_aux_wav_bytes, VoiceChangerState};
use hound::{SampleFormat, WavSpec, WavWriter};
use kittentts::download::{self, ModelConfig};
use serde::{Deserialize, Serialize};
use std::io::Cursor;
use std::path::Path;
use std::sync::Mutex;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_HUB_ID: &str = "KittenML/kitten-tts-mini-0.8";

pub struct NeuralOnnxTtsState {
    hub_id: std::sync::Arc<Mutex<String>>,
    voice: std::sync::Arc<Mutex<String>>,
    speed: std::sync::Arc<Mutex<f32>>,
    model: std::sync::Arc<Mutex<Option<kittentts::KittenTTS>>>,
    /// Last EP hint from the UI (`cpu` / `cuda` / `dml`). Recorded only: `kittentts` builds its own
    /// ORT session with the default (CPU) provider, so the hint does not select a device yet.
    execution_provider_hint: std::sync::Arc<Mutex<String>>,
    /// When true, load from `local_model_dir` instead of Hugging Face.
    use_local: std::sync::Arc<Mutex<bool>>,
    local_model_dir: std::sync::Arc<Mutex<String>>,
    /// `speech_package_v1` today; other values reserved for future formats.
    local_package_format: std::sync::Arc<Mutex<String>>,
}

impl NeuralOnnxTtsState {
    fn new() -> Self {
        Self {
            hub_id: std::sync::Arc::new(Mutex::new(String::from(DEFAULT_HUB_ID))),
            voice: std::sync::Arc::new(Mutex::new("Jasper".to_string())),
            speed: std::sync::Arc::new(Mutex::new(1.0)),
            model: std::sync::Arc::new(Mutex::new(None)),
            execution_provider_hint: std::sync::Arc::new(Mutex::new("cpu".to_string())),
            use_local: std::sync::Arc::new(Mutex::new(false)),
            local_model_dir: std::sync::Arc::new(Mutex::new(String::new())),
            local_package_format: std::sync::Arc::new(Mutex::new(String::from("speech_package_v1"))),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct NeuralOnnxVoice {
    pub name: String,
}

fn load_from_local_dir(dir: &Path) -> Result<kittentts::KittenTTS, String> {
    let config_path = dir.join("config.json");
    let config_bytes = std::fs::read(&config_path).map_err(|e| {
        format!(
            "Cannot read {}: {}",
            config_path.display(),
            e
        )
    })?;
    let config: ModelConfig = serde_json::from_slice(&config_bytes)
        .map_err(|e| format!("Invalid config.json in {}: {}", dir.display(), e))?;
    if !matches!(config.model_type.as_str(), "ONNX1" | "ONNX2") {
        return Err(format!(
            "Unsupported model type '{}' in config.json (expected ONNX1 or ONNX2)",
            config.model_type
        ));
    }
    let model_path = dir.join(&config.model_file);
    let voices_path = dir.join(&config.voices);
    kittentts::KittenTTS::load(
        &model_path,
        &voices_path,
        config.speed_priors,
        config.voice_aliases,
    )
    .map_err(|e| format!("Failed to load local speech model from folder: {}", e))
}

fn load_model_for_state(state: &NeuralOnnxTtsState) -> Result<kittentts::KittenTTS, String> {
    // Commit the app-owned ORT environment before kittentts creates its first session.
    ensure_ort_environment();
    let use_local = *state.use_local.lock().map_err(|_| "Lock failed")?;
    if use_local {
        let fmt = state
            .local_package_format
            .lock()
            .map_err(|_| "Lock failed")?
            .trim()
            .to_lowercase();
        if fmt != "speech_package_v1" {
            return Err(format!(
                "Unsupported local speech folder format '{}'. Only speech_package_v1 is available (see docs/OPERATOR_REFERENCE.md#local-speech-package-onnx).",
                fmt
            ));
        }
        let dir = state
            .local_model_dir
            .lock()
            .map_err(|_| "Lock failed")?
            .clone();
        let trimmed = dir.trim();
        if trimmed.is_empty() {
            return Err("Local model folder path is empty. Choose a folder or switch to Hugging Face.".into());
        }
        load_from_local_dir(Path::new(trimmed))
    } else {
        let hub_id = state.hub_id.lock().map_err(|_| "Lock failed")?.clone();
        download::load_from_hub(&hub_id).map_err(|e| format!("Failed to load speech model from Hub: {}", e))
    }
}

#[command]
fn neural_onnx_tts_set_execution_hint<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    hint: String,
) -> Result<(), String> {
    let h = hint.trim().to_lowercase();
    if h.is_empty() {
        return Err("execution hint must not be empty".into());
    }
    *state
        .execution_provider_hint
        .lock()
        .map_err(|_| "Lock failed")? = h;
    Ok(())
}

#[command]
fn neural_onnx_tts_set_hub_id<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    hub_id: String,
) -> Result<(), String> {
    let id = hub_id.trim().to_string();
    if id.is_empty() {
        return Err("hub_id must not be empty".into());
    }
    *state.hub_id.lock().map_err(|_| "Lock failed")? = id;
    *state.use_local.lock().map_err(|_| "Lock failed")? = false;
    *state.model.lock().map_err(|_| "Lock failed")? = None;
    Ok(())
}

/// `source`: `"hub"` or `"local"`. For `local`, pass the absolute folder path (may be empty until user picks a folder).
#[command]
fn neural_onnx_tts_set_model_source<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    source: String,
    local_dir: Option<String>,
) -> Result<(), String> {
    let s = source.trim().to_lowercase();
    match s.as_str() {
        "hub" => {
            *state.use_local.lock().map_err(|_| "Lock failed")? = false;
            *state.local_model_dir.lock().map_err(|_| "Lock failed")? = String::new();
        }
        "local" => {
            *state.use_local.lock().map_err(|_| "Lock failed")? = true;
            if let Some(d) = local_dir {
                *state.local_model_dir.lock().map_err(|_| "Lock failed")? = d;
            }
        }
        _ => return Err("model source must be 'hub' or 'local'".into()),
    }
    *state.model.lock().map_err(|_| "Lock failed")? = None;
    Ok(())
}

#[command]
fn neural_onnx_tts_set_local_package_format<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    format: String,
) -> Result<(), String> {
    let f = format.trim().to_lowercase();
    if f.is_empty() {
        return Err("local package format must not be empty".into());
    }
    *state
        .local_package_format
        .lock()
        .map_err(|_| "Lock failed")? = f;
    *state.model.lock().map_err(|_| "Lock failed")? = None;
    Ok(())
}

#[command]
fn neural_onnx_tts_set_voice_speed<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    voice: String,
    speed: f32,
) -> Result<(), String> {
    *state.voice.lock().map_err(|_| "Lock failed")? = if voice.trim().is_empty() {
        "Jasper".to_string()
    } else {
        voice
    };
    *state.speed.lock().map_err(|_| "Lock failed")? = speed.clamp(0.5, 2.0);
    Ok(())
}

#[command]
fn neural_onnx_tts_load_model<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    execution_provider: Option<String>,
) -> Result<(), String> {
    if let Some(ref ep) = execution_provider {
        *state
            .execution_provider_hint
            .lock()
            .map_err(|_| "Lock failed")? = ep.to_lowercase();
    }
    let tts = load_model_for_state(&state)?;
    *state.model.lock().map_err(|_| "Lock failed")? = Some(tts);
    Ok(())
}

#[command]
fn neural_onnx_tts_list_voices<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
) -> Result<Vec<NeuralOnnxVoice>, String> {
    let guard = state.model.lock().map_err(|_| "Lock failed")?;
    let Some(tts) = guard.as_ref() else {
        return Ok(vec![]);
    };
    Ok(tts
        .available_voices
        .iter()
        .map(|name| NeuralOnnxVoice {
            name: name.clone(),
        })
        .collect())
}

fn synthesize_wav_bytes(state: &NeuralOnnxTtsState, text: &str) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let voice = state.voice.lock().map_err(|_| "Lock failed")?.clone();
    let speed = *state.speed.lock().map_err(|_| "Lock failed")?;
    let mut model_guard = state.model.lock().map_err(|_| "Lock failed")?;
    if model_guard.is_none() {
        let tts = load_model_for_state(state)?;
        *model_guard = Some(tts);
    }
    let tts = model_guard.as_ref().ok_or("Model not loaded")?;
    let audio = tts
        .generate(text, &voice, speed, true)
        .map_err(|e| format!("Synthesis failed: {}", e))?;

    let spec = WavSpec {
        channels: 1,
        sample_rate: kittentts::SAMPLE_RATE,
        bits_per_sample: 32,
        sample_format: SampleFormat::Float,
    };
    let mut cursor = Cursor::new(Vec::new());
    {
        let mut writer = WavWriter::new(&mut cursor, spec).map_err(|e| e.to_string())?;
        for s in &audio {
            writer.write_sample(*s).map_err(|e| e.to_string())?;
        }
        writer.finalize().map_err(|e| e.to_string())?;
    }
    Ok(cursor.into_inner())
}

#[command]
fn neural_onnx_tts_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
    text: String,
) -> Result<Vec<u8>, String> {
    synthesize_wav_bytes(&state, &text)
}

/// One IPC hop: synthesize WAV and mix into the voice changer (no base64 / webview round-trip).
#[command]
fn neural_onnx_tts_speak_into_voice_changer<R: Runtime>(
    _app: AppHandle<R>,
    onnx: State<'_, NeuralOnnxTtsState>,
    vc: State<'_, VoiceChangerState>,
    text: String,
) -> Result<(), String> {
    let wav = synthesize_wav_bytes(&onnx, &text)?;
    push_aux_wav_bytes(&vc, wav)
}

#[command]
fn neural_onnx_tts_is_loaded<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralOnnxTtsState>,
) -> Result<bool, String> {
    Ok(state
        .model
        .lock()
        .map_err(|_| "Lock failed")?
        .is_some())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("neural-onnx-tts")
        .setup(|app, _api| {
            app.manage(NeuralOnnxTtsState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            neural_onnx_tts_set_execution_hint,
            neural_onnx_tts_set_hub_id,
            neural_onnx_tts_set_model_source,
            neural_onnx_tts_set_local_package_format,
            neural_onnx_tts_set_voice_speed,
            neural_onnx_tts_load_model,
            neural_onnx_tts_list_voices,
            neural_onnx_tts_speak,
            neural_onnx_tts_speak_into_voice_changer,
            neural_onnx_tts_is_loaded,
        ])
        .build()
}
