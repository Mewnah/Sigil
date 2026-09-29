use crate::services::http_client::{describe_failed_response, http_download_get};
use crate::services::whisper::{progress_percent, sweep_stale_partials, ProgressThrottle};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};

// Note: vosk-rs requires pre-built Vosk libraries to be available.
// STT uses vosk-browser (WASM); models are stored as zip under app data and loaded via convertFileSrc.

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

const MIN_VOSK_ZIP_BYTES: usize = 10 * 1024;

const DOWNLOAD_CANCELLED: &str = "Vosk model download cancelled.";

/// Per-call suffix for import temp files so overlapping imports never share (or delete) each other's file.
static IMPORT_SEQ: AtomicU64 = AtomicU64::new(0);

fn validate_vosk_zip_bytes(bytes: &[u8]) -> Result<(), String> {
    if bytes.len() < MIN_VOSK_ZIP_BYTES {
        return Err("File too small to be a valid Vosk model zip.".to_string());
    }
    if bytes.first() == Some(&b'<') {
        return Err("File looks like HTML, not a zip.".to_string());
    }
    Ok(())
}

fn copy_vosk_zip_to_cache(src: &Path, dest: &Path) -> Result<(), String> {
    let bytes = fs::read(src).map_err(|e| e.to_string())?;
    validate_vosk_zip_bytes(&bytes)?;
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let seq = IMPORT_SEQ.fetch_add(1, Ordering::Relaxed);
    let partial_path = dest.with_extension(format!("zip.{}.{seq}.partial_import", std::process::id()));
    let _ = fs::remove_file(&partial_path);
    fs::write(&partial_path, &bytes).map_err(|e| {
        let _ = fs::remove_file(&partial_path);
        e.to_string()
    })?;
    // `rename` replaces an existing destination in one step (Windows and Unix), so `dest` is never missing.
    fs::rename(&partial_path, dest).map_err(|e| {
        let _ = fs::remove_file(&partial_path);
        e.to_string()
    })?;
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
    /// Bumped on each download start and on cancel (cancel = bump). Stale runs stop streaming and discard their
    /// own partial file; they never touch another run's partial or delete a committed zip.
    download_epoch: Arc<AtomicU64>,
}

impl VoskSttState {
    fn new() -> Self {
        Self {
            model_path: Arc::new(Mutex::new(None)),
            is_available: Arc::new(Mutex::new(false)),
            download_epoch: Arc::new(AtomicU64::new(0)),
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

fn download_stale(state: &VoskSttState, my_epoch: u64) -> bool {
    state.download_epoch.load(Ordering::SeqCst) != my_epoch
}

/// `vosk:download_progress` payload, same shape as `whisper:download_progress`.
#[derive(Clone, Serialize)]
struct ProgressPayload {
    file: String,
    /// Percent 0-100, or -1 while the total length is unknown.
    progress: f64,
}

fn emit_progress<R: Runtime>(app: &AppHandle<R>, file: &str, progress: f64) {
    let _ = app.emit(
        "vosk:download_progress",
        ProgressPayload {
            file: file.to_string(),
            progress,
        },
    );
}

/// Stream `url` into this run's own `partial_path` and validate it, emitting `vosk:download_progress` for
/// `file_label` at most ~5 times per second (nothing once the run is stale). The caller removes the partial on `Err`.
async fn fetch_zip_to_partial<R: Runtime>(
    app: &AppHandle<R>,
    state: &VoskSttState,
    my_epoch: u64,
    url: &str,
    file_label: &str,
    partial_path: &Path,
) -> Result<(), String> {
    let response = http_download_get(url)
        .send()
        .await
        .map_err(|e| format!("Failed to download model: {}", e))?;

    if !response.status().is_success() {
        let detail = describe_failed_response(response).await;
        return Err(format!("Download failed: {detail}"));
    }

    let total = response.content_length().unwrap_or(0);
    let mut stream = response.bytes_stream();
    let mut downloaded: u64 = 0;
    let mut first_byte: Option<u8> = None;
    let mut throttle = ProgressThrottle::new();

    {
        let mut file = File::create(partial_path).map_err(|e| e.to_string())?;
        while let Some(chunk) = stream.next().await {
            if download_stale(state, my_epoch) {
                return Err(DOWNLOAD_CANCELLED.to_string());
            }
            let chunk = chunk.map_err(|e| format!("Failed to read response: {}", e))?;
            if first_byte.is_none() && !chunk.is_empty() {
                first_byte = Some(chunk[0]);
            }
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            downloaded += chunk.len() as u64;
            if throttle.ready() {
                emit_progress(app, file_label, progress_percent(downloaded, total).unwrap_or(-1.0));
            }
        }
    }
    if !download_stale(state, my_epoch) {
        emit_progress(app, file_label, 100.0);
    }

    if downloaded < MIN_VOSK_ZIP_BYTES as u64 {
        return Err("Downloaded file too small to be a valid model zip.".to_string());
    }
    if first_byte == Some(b'<') {
        return Err("Download returned HTML instead of a zip.".to_string());
    }
    Ok(())
}

/// Download `model_id` to `{app_data}/vosk-models/{model_id}.zip` when missing.
/// `url_override`: full zip URL when using "Custom model URL" in the UI.
#[command]
async fn download_vosk_model<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, VoskSttState>,
    model_id: String,
    url_override: Option<String>,
    local_zip_path: Option<String>,
) -> Result<String, String> {
    validate_model_id(&model_id)?;
    // A newer run or `cancel_vosk_model_download` bumps the epoch; from then on this run is stale.
    let my_epoch = state.download_epoch.fetch_add(1, Ordering::SeqCst) + 1;

    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let models_dir = app_dir.join("vosk-models");
    fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;

    let zip_path = models_dir.join(format!("{}.zip", model_id));
    let rel = format!("vosk-models/{}.zip", model_id);

    if zip_path.is_file() {
        return Ok(rel);
    }

    if let Some(ref lp) = local_zip_path {
        let t = lp.trim();
        if !t.is_empty() {
            copy_vosk_zip_to_cache(Path::new(t), &zip_path)?;
            return Ok(rel);
        }
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

    // Per-run name (process id + epoch): overlapping runs, and a second Sigil instance whose epoch counter
    // also restarted at launch, never share, rename or delete each other's partial file.
    // (`list_downloaded_vosk_models` only lists names ending in `.zip`, so partials never show up.)
    let partial_path = models_dir.join(format!("{}.zip.{}.{}.partial", model_id, std::process::id(), my_epoch));
    let _ = fs::remove_file(&partial_path);

    let file_label = format!("{}.zip", model_id);
    if let Err(e) = fetch_zip_to_partial(&app, &state, my_epoch, &url, &file_label, &partial_path).await {
        let _ = fs::remove_file(&partial_path);
        return Err(e);
    }
    if download_stale(&state, my_epoch) {
        let _ = fs::remove_file(&partial_path);
        return Err(DOWNLOAD_CANCELLED.to_string());
    }
    if zip_path.is_file() {
        // An overlapping run committed this model first; keep its zip and drop ours.
        let _ = fs::remove_file(&partial_path);
        return Ok(rel);
    }
    if let Err(e) = fs::rename(&partial_path, &zip_path) {
        let _ = fs::remove_file(&partial_path);
        return Err(e.to_string());
    }

    // The committed zip is valid: a cancel that lands after the rename must never delete it.
    if download_stale(&state, my_epoch) {
        return Err(DOWNLOAD_CANCELLED.to_string());
    }
    Ok(rel)
}

#[command]
fn cancel_vosk_model_download(state: State<'_, VoskSttState>) -> Result<(), String> {
    // Makes every in-flight download stale; each run then removes only its own partial file.
    state.download_epoch.fetch_add(1, Ordering::SeqCst);
    Ok(())
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

/// Copy a user-selected `.zip` into app data for the selected catalog model id.
#[command]
async fn import_vosk_model_zip<R: Runtime>(
    app: AppHandle<R>,
    model_id: String,
    source_path: String,
) -> Result<(), String> {
    validate_model_id(&model_id)?;
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let models_dir = app_dir.join("vosk-models");
    fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;
    let zip_path = models_dir.join(format!("{}.zip", model_id));
    copy_vosk_zip_to_cache(Path::new(source_path.trim()), &zip_path)?;
    Ok(())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("vosk-stt")
        .setup(|app, _api| {
            app.manage(VoskSttState::new());
            if let Ok(dir) = app.path().app_data_dir() {
                sweep_stale_partials(&dir.join("vosk-models"));
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_vosk_models,
            check_vosk_native_availability,
            set_vosk_model_path,
            get_vosk_model_path,
            vosk_native_transcribe,
            download_vosk_model,
            cancel_vosk_model_download,
            import_vosk_model_zip,
            vosk_model_zip_relative_path,
            list_downloaded_vosk_models
        ])
        .build()
}
