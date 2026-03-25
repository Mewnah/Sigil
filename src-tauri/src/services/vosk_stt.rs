use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

// Note: vosk-rs requires pre-built Vosk libraries to be available.
// If the vosk crate fails to link, the WASM fallback (vosk-browser) remains available.
//
// To enable native Vosk:
// 1. Download Vosk library from https://github.com/alphacep/vosk-api/releases
// 2. Set VOSK_LIB_PATH environment variable or add to system library path
// 3. Rebuild the application

const MODEL_CDN_BASE: &str = "https://alphacephei.com/vosk/models";

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct VoskModel {
    pub id: String,
    pub name: String,
    pub language: String,
    pub size: String,
    pub url: String,
}

/// Get available Vosk models for download
fn get_available_models() -> Vec<VoskModel> {
    vec![
        VoskModel {
            id: "vosk-model-small-en-us-0.15".to_string(),
            name: "English (US) - Small".to_string(),
            language: "en".to_string(),
            size: "40 MB".to_string(),
            url: format!("{}/vosk-model-small-en-us-0.15.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-cn-0.22".to_string(),
            name: "Chinese - Small".to_string(),
            language: "zh".to_string(),
            size: "42 MB".to_string(),
            url: format!("{}/vosk-model-small-cn-0.22.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-de-0.15".to_string(),
            name: "German - Small".to_string(),
            language: "de".to_string(),
            size: "45 MB".to_string(),
            url: format!("{}/vosk-model-small-de-0.15.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-fr-0.22".to_string(),
            name: "French - Small".to_string(),
            language: "fr".to_string(),
            size: "41 MB".to_string(),
            url: format!("{}/vosk-model-small-fr-0.22.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-es-0.42".to_string(),
            name: "Spanish - Small".to_string(),
            language: "es".to_string(),
            size: "39 MB".to_string(),
            url: format!("{}/vosk-model-small-es-0.42.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-ja-0.22".to_string(),
            name: "Japanese - Small".to_string(),
            language: "ja".to_string(),
            size: "48 MB".to_string(),
            url: format!("{}/vosk-model-small-ja-0.22.zip", MODEL_CDN_BASE),
        },
        VoskModel {
            id: "vosk-model-small-ru-0.22".to_string(),
            name: "Russian - Small".to_string(),
            language: "ru".to_string(),
            size: "45 MB".to_string(),
            url: format!("{}/vosk-model-small-ru-0.22.zip", MODEL_CDN_BASE),
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

/// Get list of available Vosk models for download
#[command]
fn get_vosk_models() -> Vec<VoskModel> {
    get_available_models()
}

/// Check if native Vosk is available (requires vosk library)
#[command]
fn check_vosk_native_availability(state: State<'_, VoskSttState>) -> bool {
    // Native Vosk requires pre-built libraries to be installed
    // For now, report as unavailable - frontend will use vosk-browser WASM
    *state.is_available.lock().unwrap() = false;
    false
}

/// Set the path to a downloaded Vosk model
#[command]
fn set_vosk_model_path(state: State<'_, VoskSttState>, path: String) -> Result<(), String> {
    *state.model_path.lock().map_err(|_| "Lock failed")? = Some(path);
    Ok(())
}

/// Get the current model path
#[command]
fn get_vosk_model_path(state: State<'_, VoskSttState>) -> Option<String> {
    state.model_path.lock().ok()?.clone()
}

/// Transcribe audio using native Vosk (if available)
/// Falls back to returning an error if native Vosk isn't compiled in
#[command]
async fn vosk_native_transcribe(_state: State<'_, VoskSttState>, _audio_data: Vec<i16>, _sample_rate: u32) -> Result<String, String> {
    // Native Vosk transcription would go here when vosk-native feature is enabled
    // For now, this returns an error indicating to use the WASM fallback
    Err("Native Vosk not available. Using browser-based Vosk (WASM) instead.".to_string())
}

/// Download a Vosk model to the app data directory
#[command]
async fn download_vosk_model<R: Runtime>(app: AppHandle<R>, model_id: String) -> Result<String, String> {
    let models = get_available_models();
    let model = models
        .iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| format!("Model '{}' not found", model_id))?;

    // Get app data directory
    let app_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;

    let models_dir = app_dir.join("vosk-models");
    std::fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;

    let model_path = models_dir.join(&model.id);

    // Check if already downloaded
    if model_path.exists() {
        return Ok(model_path.to_string_lossy().to_string());
    }

    // Download the model
    let response = reqwest::get(&model.url)
        .await
        .map_err(|e| format!("Failed to download model: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Download failed: {}", response.status()));
    }

    let bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read response: {}", e))?;

    // Save the zip file
    let zip_path = models_dir.join(format!("{}.zip", model.id));
    std::fs::write(&zip_path, &bytes).map_err(|e| format!("Failed to save model: {}", e))?;

    // Extract the zip file
    let file = std::fs::File::open(&zip_path).map_err(|e| e.to_string())?;
    let mut archive = zip::ZipArchive::new(file).map_err(|e| e.to_string())?;

    for i in 0..archive.len() {
        let mut file = archive.by_index(i).map_err(|e| e.to_string())?;
        let outpath = models_dir.join(file.name());

        if file.name().ends_with('/') {
            std::fs::create_dir_all(&outpath).map_err(|e| e.to_string())?;
        } else {
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
                }
            }
            let mut outfile = std::fs::File::create(&outpath).map_err(|e| e.to_string())?;
            std::io::copy(&mut file, &mut outfile).map_err(|e| e.to_string())?;
        }
    }

    // Clean up zip file
    let _ = std::fs::remove_file(&zip_path);

    Ok(model_path.to_string_lossy().to_string())
}

/// List downloaded Vosk models
#[command]
async fn list_downloaded_vosk_models<R: Runtime>(app: AppHandle<R>) -> Result<Vec<String>, String> {
    let app_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;

    let models_dir = app_dir.join("vosk-models");
    if !models_dir.exists() {
        return Ok(vec![]);
    }

    let mut models = vec![];
    for entry in std::fs::read_dir(&models_dir).map_err(|e| e.to_string())? {
        let entry = entry.map_err(|e| e.to_string())?;
        if entry.path().is_dir() {
            if let Some(name) = entry.file_name().to_str() {
                if name.starts_with("vosk-model-") {
                    models.push(name.to_string());
                }
            }
        }
    }

    Ok(models)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("vosk_stt")
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
            list_downloaded_vosk_models
        ])
        .build()
}
