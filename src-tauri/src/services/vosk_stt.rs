use crate::services::http_client::http_client;
use serde::{Deserialize, Serialize};
use std::fs;
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

// Note: vosk-rs requires pre-built Vosk libraries to be available.
// STT uses vosk-browser (WASM); models are stored as zip under app data and loaded via Blob URL.

const MODEL_CDN_BASE: &str = "https://alphacephei.com/vosk/models";

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VoskModel {
    pub id: String,
    pub name: String,
    pub language: String,
    pub size: String,
    pub url: String,
}

fn validate_model_id(model_id: &str) -> Result<(), String> {
    if model_id.is_empty() {
        return Err("Model id is empty.".to_string());
    }
    if model_id.contains("..") || model_id.contains('/') || model_id.contains('\\') {
        return Err("Invalid model id.".to_string());
    }
    Ok(())
}

/// Catalog: single source of truth for the STT inspector (must match files on alphacephei CDN).
fn get_available_models() -> Vec<VoskModel> {
    vec![
        VoskModel {
            id: "vosk-model-small-en-us-0.15".to_string(),
            name: "English (US)".to_string(),
            language: "en".to_string(),
            size: "40 MB".to_string(),
            url: format!("{}/vosk-model-small-en-us-0.15.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-en-in-0.4".to_string(),
            name: "English (India)".to_string(),
            language: "en".to_string(),
            size: "36 MB".to_string(),
            url: format!("{}/vosk-model-small-en-in-0.4.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-cn-0.22".to_string(),
            name: "Chinese".to_string(),
            language: "zh".to_string(),
            size: "42 MB".to_string(),
            url: format!("{}/vosk-model-small-cn-0.22.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-ru-0.22".to_string(),
            name: "Russian".to_string(),
            language: "ru".to_string(),
            size: "45 MB".to_string(),
            url: format!("{}/vosk-model-small-ru-0.22.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-de-0.15".to_string(),
            name: "German".to_string(),
            language: "de".to_string(),
            size: "45 MB".to_string(),
            url: format!("{}/vosk-model-small-de-0.15.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-fr-0.22".to_string(),
            name: "French".to_string(),
            language: "fr".to_string(),
            size: "41 MB".to_string(),
            url: format!("{}/vosk-model-small-fr-0.22.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-es-0.42".to_string(),
            name: "Spanish".to_string(),
            language: "es".to_string(),
            size: "39 MB".to_string(),
            url: format!("{}/vosk-model-small-es-0.42.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-pt-0.3".to_string(),
            name: "Portuguese".to_string(),
            language: "pt".to_string(),
            size: "31 MB".to_string(),
            url: format!("{}/vosk-model-small-pt-0.3.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-it-0.22".to_string(),
            name: "Italian".to_string(),
            language: "it".to_string(),
            size: "48 MB".to_string(),
            url: format!("{}/vosk-model-small-it-0.22.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-ja-0.22".to_string(),
            name: "Japanese".to_string(),
            language: "ja".to_string(),
            size: "48 MB".to_string(),
            url: format!("{}/vosk-model-small-ja-0.22.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-ko-0.22".to_string(),
            name: "Korean".to_string(),
            language: "ko".to_string(),
            size: "82 MB".to_string(),
            url: format!("{}/vosk-model-small-ko-0.22.zip", MODEL_CDN_BASE),
        },
    ]
}

#[derive(Clone)]
pub struct VoskSttState {
    model_path: Arc<Mutex<Option<String>>>,
    is_available: Arc<Mutex<bool>>,
}

impl VoskSttState {
    fn new() -> Self {
        Self {
            model_path: Arc::new(Mutex::new(None)),
            is_available: Arc::new(Mutex::new(false)),
        }
    }
}

#[command]
fn get_vosk_models() -> Vec<VoskModel> {
    get_available_models()
}

#[command]
fn check_vosk_native_availability(state: State<'_, VoskSttState>) -> bool {
    *state.is_available.lock().unwrap() = false;
    false
}

#[command]
fn set_vosk_model_path(state: State<'_, VoskSttState>, path: String) -> Result<(), String> {
    *state.model_path.lock().map_err(|_| "Lock failed")? = Some(path);
    Ok(())
}

#[command]
fn get_vosk_model_path(state: State<'_, VoskSttState>) -> Option<String> {
    state.model_path.lock().ok()?.clone()
}

#[command]
async fn vosk_native_transcribe(
    _state: State<'_, VoskSttState>,
    _audio_data: Vec<i16>,
    _sample_rate: u32,
) -> Result<String, String> {
    Err("Native Vosk not available. Using browser-based Vosk (WASM) instead.".to_string())
}

/// Relative path from app data dir: `vosk-models/{id}.zip`, if present.
#[command]
fn vosk_model_zip_relative_path<R: Runtime>(
    app: AppHandle<R>,
    model_id: String,
) -> Result<Option<String>, String> {
    validate_model_id(&model_id)?;
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let zip_path = app_dir.join("vosk-models").join(format!("{}.zip", model_id));
    if zip_path.is_file() {
        Ok(Some(format!("vosk-models/{}.zip", model_id)))
    } else {
        Ok(None)
    }
}

/// Download `model_id` to `{app_data}/vosk-models/{model_id}.zip` when missing.
/// `url_override`: full zip URL when using "Custom model URL" in the UI.
#[command]
async fn download_vosk_model<R: Runtime>(
    app: AppHandle<R>,
    model_id: String,
    url_override: Option<String>,
) -> Result<String, String> {
    validate_model_id(&model_id)?;

    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let models_dir = app_dir.join("vosk-models");
    fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;

    let zip_path = models_dir.join(format!("{}.zip", model_id));
    let rel = format!("vosk-models/{}.zip", model_id);

    if zip_path.is_file() {
        return Ok(rel);
    }

    let url = if let Some(u) = url_override {
        let t = u.trim().to_string();
        if t.is_empty() {
            return Err("Custom model URL is empty.".to_string());
        }
        t
    } else {
        get_available_models()
            .into_iter()
            .find(|m| m.id == model_id)
            .map(|m| m.url)
            .ok_or_else(|| {
                format!(
                    "Unknown model '{}'. Choose a listed model or set a custom zip URL.",
                    model_id
                )
            })?
    };

    let response = http_client()
        .get(&url)
        .timeout(Duration::from_secs(7200))
        .send()
        .await
        .map_err(|e| format!("Failed to download model: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Download failed: HTTP {}", response.status()));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    const MIN_ZIP: usize = 10 * 1024;
    if bytes.len() < MIN_ZIP {
        return Err("Downloaded file too small to be a valid model zip.".to_string());
    }
    if bytes.first() == Some(&b'<') {
        return Err("Download returned HTML instead of a zip.".to_string());
    }

    let partial_path = models_dir.join(format!("{}.zip.partial", model_id));
    let _ = fs::remove_file(&partial_path);

    fs::write(&partial_path, &bytes).map_err(|e| e.to_string())?;

    if let Err(e) = fs::rename(&partial_path, &zip_path) {
        let _ = fs::remove_file(&partial_path);
        let _ = fs::remove_file(&zip_path);
        return Err(e.to_string());
    }

    Ok(rel)
}

/// Model ids that have a `vosk-models/{id}.zip` on disk.
#[command]
async fn list_downloaded_vosk_models<R: Runtime>(app: AppHandle<R>) -> Result<Vec<String>, String> {
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let models_dir = app_dir.join("vosk-models");
    if !models_dir.exists() {
        return Ok(vec![]);
    }

    let mut models = vec![];
    for entry in fs::read_dir(&models_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        if path.is_file() {
            if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
                if let Some(id) = name.strip_suffix(".zip") {
                    if !id.ends_with(".partial") && id.starts_with("vosk-model-") {
                        models.push(id.to_string());
                    }
                }
            }
        }
    }

    models.sort();
    Ok(models)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("vosk-stt")
        .setup(|app, _api| {
            app.manage(VoskSttState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_vosk_models,
            check_vosk_native_availability,
            set_vosk_model_path,
            get_vosk_model_path,
            vosk_native_transcribe,
            download_vosk_model,
            vosk_model_zip_relative_path,
            list_downloaded_vosk_models
        ])
        .build()
}
