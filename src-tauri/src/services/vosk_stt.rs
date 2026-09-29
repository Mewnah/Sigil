//! `vosk-stt` plugin: model files for the Vosk backend. Recognition itself runs in the webview
//! (vosk-browser, WASM), which loads `<app_data>/vosk-models/<id>.zip` through the asset protocol
//! (scope `$APPDATA/vosk-models/**`); this plugin only puts the zip there.
//!
//! Every download and import goes through the model store (P1-A):
//! - every listed model has a catalog entry (`vosk/<id without "vosk-model-">`, family `vosk`, all
//!   11 sha256-pinned in `model-catalog/stt-legacy.json`) and is installed by the store: pinned
//!   sha256 and size, Range resume, and a zip already on disk is adopted after a hash check instead
//!   of being fetched again. alphacephei publishes no digests: only the `small-en-us-0.15` pin has
//!   been checked against a real download; see the fragment's `$comment` for the other ten;
//! - "Custom model URL" (and an id missing from the catalog) has no hash: the store's unpinned
//!   fetch still gives it resume (the partial is kept on cancel), the stall timeout, HTML
//!   detection and the one-download-at-a-time slot, and the zip is fully CRC-checked before it
//!   replaces anything;
//! - a local zip (import, or "use local zip") is streamed to disk with its CRCs checked, never read
//!   into memory whole (models reach 1.8 GB), then renamed into place atomically.
//!
//! `vosk:download_progress` (`{ file, progress }`, percent or -1, a few per second) is unchanged.

use crate::services::model_store::{
    self as store, register_legacy_file, validate_zip, vosk_entry_id, CancelToken, Digest, DownloadError,
    ExtractLimits, Origin, Store,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};
use tokio::sync::broadcast::error::RecvError;

// Note: vosk-rs requires pre-built Vosk libraries to be available.
// STT uses vosk-browser (WASM); models are stored as zip under app data and loaded via convertFileSrc.

const MODEL_CDN_BASE: &str = "https://alphacephei.com/vosk/models";

/// Folder under the app data folder that the asset protocol may read.
const MODELS_FOLDER: &str = "vosk-models";

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

const MIN_VOSK_ZIP_BYTES: u64 = 10 * 1024;

const DOWNLOAD_CANCELLED: &str = "Vosk model download cancelled.";

/// Per-call suffix for import temp files so overlapping imports never share (or delete) each other's file.
static IMPORT_SEQ: AtomicU64 = AtomicU64::new(0);

/// Chunk size of local zip imports; progress and cancellation are checked once per chunk.
const COPY_CHUNK_BYTES: usize = 1 << 20;

/// Minimum spacing of `vosk:download_progress` events.
const PROGRESS_EMIT_INTERVAL: Duration = Duration::from_millis(200);

/// Catalog: single source of truth for the STT inspector (must match files on alphacephei CDN).
fn get_available_models() -> Vec<VoskModel> {
    let m = |id: &str, name: &str, language: &str, size: &str| VoskModel {
        id: id.to_string(),
        name: name.to_string(),
        language: language.to_string(),
        size: size.to_string(),
        url: format!("{MODEL_CDN_BASE}/{id}.zip"),
    };
    vec![
        m("vosk-model-small-en-us-0.15", "English (US)", "en", "40 MB"),
        m("vosk-model-small-en-in-0.4", "English (India)", "en", "36 MB"),
        m("vosk-model-small-cn-0.22", "Chinese", "zh", "42 MB"),
        m("vosk-model-small-ru-0.22", "Russian", "ru", "45 MB"),
        m("vosk-model-small-de-0.15", "German", "de", "45 MB"),
        m("vosk-model-small-fr-0.22", "French", "fr", "41 MB"),
        m("vosk-model-small-es-0.42", "Spanish", "es", "39 MB"),
        m("vosk-model-small-pt-0.3", "Portuguese", "pt", "31 MB"),
        m("vosk-model-small-it-0.22", "Italian", "it", "48 MB"),
        m("vosk-model-small-ja-0.22", "Japanese", "ja", "48 MB"),
        m("vosk-model-small-ko-0.22", "Korean", "ko", "82 MB"),
    ]
}

pub struct VoskSttState {
    model_path: Arc<Mutex<Option<String>>>,
    is_available: Arc<Mutex<bool>>,
    /// Cancels every download / import started before the next `cancel_vosk_model_download`
    /// (which cancels this token and installs a fresh one).
    cancel: Mutex<CancelToken>,
}

impl VoskSttState {
    fn new() -> Self {
        Self {
            model_path: Arc::new(Mutex::new(None)),
            is_available: Arc::new(Mutex::new(false)),
            cancel: Mutex::new(CancelToken::default()),
        }
    }

    fn token(&self) -> CancelToken {
        self.cancel.lock().unwrap_or_else(|p| p.into_inner()).clone()
    }

    fn cancel_all(&self) {
        let mut t = self.cancel.lock().unwrap_or_else(|p| p.into_inner());
        t.cancel();
        *t = CancelToken::default();
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

fn zip_rel_path(model_id: &str) -> String {
    format!("{MODELS_FOLDER}/{model_id}.zip")
}

/// Relative path from app data dir: `vosk-models/{id}.zip`, if present.
#[command]
fn vosk_model_zip_relative_path<R: Runtime>(
    app: AppHandle<R>,
    model_id: String,
) -> Result<Option<String>, String> {
    validate_model_id(&model_id)?;
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let zip_path = app_dir.join(MODELS_FOLDER).join(format!("{}.zip", model_id));
    if zip_path.is_file() {
        Ok(Some(zip_rel_path(&model_id)))
    } else {
        Ok(None)
    }
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

fn percent(done: u64, total: Option<u64>) -> f64 {
    match total {
        Some(t) if t > 0 => (done as f64 / t as f64 * 100.0).min(100.0),
        _ => -1.0,
    }
}

/// Forward the store's `models:progress` for `progress_id` as `vosk:download_progress`.
fn bridge_progress<R: Runtime>(
    app: &AppHandle<R>,
    store: &Store,
    progress_id: String,
    file_label: String,
) -> tauri::async_runtime::JoinHandle<()> {
    let mut rx = store.subscribe_progress();
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(e) if e.id == progress_id => emit_progress(&app, &file_label, percent(e.bytes_done, e.bytes_total)),
                Ok(_) | Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => break,
            }
        }
    })
}

/// A complete, undamaged zip of plausible size (checked before it replaces anything).
fn validate_vosk_zip_file(path: &Path, limits: &ExtractLimits) -> Result<(), String> {
    let len = fs::metadata(path).map_err(|e| e.to_string())?.len();
    if len < MIN_VOSK_ZIP_BYTES {
        return Err("File too small to be a valid Vosk model zip.".to_string());
    }
    let mut first = [0u8; 1];
    if File::open(path).and_then(|mut f| f.read_exact(&mut first)).is_ok() && first[0] == b'<' {
        return Err("File looks like HTML, not a zip.".to_string());
    }
    validate_zip(path, limits, &|| false)
        .map(|_| ())
        .map_err(|e| format!("Not a valid Vosk model zip: {e}"))
}

/// Stream `src` into `dest` through a per-call temp file next to it: sha256 while copying, the zip
/// checked (every CRC) before the atomic rename. `on_progress(copied, total)` runs per chunk;
/// returning `false` cancels. Never holds the whole file in memory.
fn copy_vosk_zip_to_cache(
    src: &Path,
    dest: &Path,
    limits: &ExtractLimits,
    on_progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<String, String> {
    let total = fs::metadata(src).map_err(|e| e.to_string())?.len();
    if total < MIN_VOSK_ZIP_BYTES {
        return Err("File too small to be a valid Vosk model zip.".to_string());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent).map_err(|e| e.to_string())?;
    }
    let seq = IMPORT_SEQ.fetch_add(1, Ordering::Relaxed);
    let partial_path = dest.with_extension(format!("zip.{}.{seq}.partial_import", std::process::id()));
    let result = (|| -> Result<String, String> {
        let mut input = File::open(src).map_err(|e| e.to_string())?;
        let mut output = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&partial_path)
            .map_err(|e| e.to_string())?;
        let mut hasher = Sha256::new();
        let mut buf = vec![0u8; COPY_CHUNK_BYTES];
        let mut copied: u64 = 0;
        loop {
            let n = match input.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.to_string()),
            };
            if copied == 0 && buf[0] == b'<' {
                return Err("File looks like HTML, not a zip.".to_string());
            }
            hasher.update(&buf[..n]);
            output.write_all(&buf[..n]).map_err(|e| e.to_string())?;
            copied += n as u64;
            if !on_progress(copied, total) {
                return Err(DOWNLOAD_CANCELLED.to_string());
            }
        }
        output.sync_all().map_err(|e| e.to_string())?;
        drop(output);
        validate_vosk_zip_file(&partial_path, limits)?;
        // `rename` replaces an existing destination in one step (Windows and Unix), so `dest` is never missing.
        fs::rename(&partial_path, dest).map_err(|e| e.to_string())?;
        Ok(hex::encode(hasher.finalize()))
    })();
    if result.is_err() {
        let _ = fs::remove_file(&partial_path);
    }
    result
}

/// Register `vosk-models/<id>.zip` with the store when it is the catalog's file (so the Models tab
/// shows it installed).
fn register_if_catalog(store: &Store, model_id: &str, sha256: &str, origin: Origin) {
    if let Some(entry) = store.catalog.get(&vosk_entry_id(model_id)) {
        if register_legacy_file(store, entry, &Digest::Sha256(sha256.to_string()), origin) {
            store.set_state(&entry.id, None);
        }
    }
}

/// Stream a user-picked zip into the cache on a worker thread (progress + cancel).
async fn import_zip<R: Runtime>(
    app: &AppHandle<R>,
    model_id: &str,
    src: PathBuf,
    dest: PathBuf,
    cancel: CancelToken,
) -> Result<(), String> {
    let app2 = app.clone();
    let file_label = format!("{model_id}.zip");
    let store = store::global_store().ok();
    let limits = store
        .as_ref()
        .map(|s| s.settings.limits.clone())
        .unwrap_or_default();
    let sha = tauri::async_runtime::spawn_blocking(move || {
        let mut last: Option<Instant> = None;
        let mut on_progress = |copied: u64, total: u64| {
            if cancel.is_cancelled() {
                return false;
            }
            if last.is_none_or(|t| t.elapsed() >= PROGRESS_EMIT_INTERVAL) {
                last = Some(Instant::now());
                emit_progress(&app2, &file_label, percent(copied, Some(total)));
            }
            true
        };
        copy_vosk_zip_to_cache(&src, &dest, &limits, &mut on_progress)
    })
    .await
    .map_err(|e| format!("Import task failed: {e}"))??;
    if let Some(store) = store {
        register_if_catalog(&store, model_id, &sha, Origin::Import);
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
    let cancel = state.token();
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let models_dir = app_dir.join(MODELS_FOLDER);
    fs::create_dir_all(&models_dir).map_err(|e| e.to_string())?;
    let zip_path = models_dir.join(format!("{}.zip", model_id));
    let rel = zip_rel_path(&model_id);

    if zip_path.is_file() {
        return Ok(rel);
    }

    if let Some(local) = local_zip_path.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        import_zip(&app, &model_id, PathBuf::from(local), zip_path, cancel).await?;
        return Ok(rel);
    }

    let store = store::global_store()?;
    let file_label = format!("{}.zip", model_id);
    let progress_id = vosk_entry_id(&model_id);

    let custom_url = match url_override {
        Some(u) => {
            let t = u.trim().to_string();
            if t.is_empty() {
                return Err("Custom model URL is empty.".to_string());
            }
            Some(t)
        }
        None => None,
    };

    if custom_url.is_none() && store.catalog.get(&progress_id).is_some() {
        // Catalog model: pinned, mirrored and resumable; an existing zip is adopted, not refetched.
        let queued = store.enqueue(std::slice::from_ref(&progress_id))?;
        if let Some(job) = queued.into_iter().next().and_then(|q| q.job.map(|j| (j, q.created))) {
            let (job, created) = job;
            let bridge = bridge_progress(&app, &store, progress_id.clone(), file_label.clone());
            let result = tokio::select! {
                r = job.wait() => r,
                _ = cancel.cancelled() => {
                    // Stop only what this call started; a download from the Models tab continues.
                    if created {
                        job.cancel.cancel();
                    }
                    Err("cancelled".to_string())
                }
            };
            bridge.abort();
            match result {
                Ok(_) => emit_progress(&app, &file_label, 100.0),
                Err(e) if e == "cancelled" => return Err(DOWNLOAD_CANCELLED.to_string()),
                Err(e) => return Err(format!("Download failed: {e}")),
            }
        }
        return if zip_path.is_file() {
            Ok(rel)
        } else {
            Err("The model was installed but its zip is missing.".to_string())
        };
    }

    let url = match custom_url {
        Some(u) => u,
        None => get_available_models()
            .into_iter()
            .find(|m| m.id == model_id)
            .map(|m| m.url)
            .ok_or_else(|| {
                format!(
                    "Unknown model '{}'. Choose a listed model or set a custom zip URL.",
                    model_id
                )
            })?,
    };

    let bridge = bridge_progress(&app, &store, progress_id.clone(), file_label.clone());
    let result = store
        .fetch_unpinned(&progress_id, &file_label, &url, &zip_path, validate_vosk_zip_file, &cancel)
        .await;
    bridge.abort();
    match result {
        Ok(digest) => {
            emit_progress(&app, &file_label, 100.0);
            if let Some(sha) = digest.sha256() {
                register_if_catalog(&store, &model_id, sha, Origin::Download);
            }
            Ok(rel)
        }
        Err(DownloadError::Cancelled) => Err(DOWNLOAD_CANCELLED.to_string()),
        Err(DownloadError::Failed(m)) => Err(format!("Download failed: {m}")),
    }
}

#[command]
fn cancel_vosk_model_download(state: State<'_, VoskSttState>) -> Result<(), String> {
    // Every download or import in flight stops; partial downloads are kept for resume.
    state.cancel_all();
    Ok(())
}

/// Model ids that have a `vosk-models/{id}.zip` on disk.
#[command]
async fn list_downloaded_vosk_models<R: Runtime>(app: AppHandle<R>) -> Result<Vec<String>, String> {
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let models_dir = app_dir.join(MODELS_FOLDER);
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
    state: State<'_, VoskSttState>,
    model_id: String,
    source_path: String,
) -> Result<(), String> {
    validate_model_id(&model_id)?;
    let app_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let zip_path = app_dir.join(MODELS_FOLDER).join(format!("{}.zip", model_id));
    import_zip(&app, &model_id, PathBuf::from(source_path.trim()), zip_path, state.token()).await
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("vosk-stt")
        .setup(|app, _api| {
            app.manage(VoskSttState::new());
            if let Ok(dir) = app.path().app_data_dir() {
                // Import temps (and partials of older Sigil versions) left by a crash.
                crate::services::whisper::sweep_stale_partials(&dir.join(MODELS_FOLDER));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::model_store::Catalog;

    fn tmp(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let d = std::env::temp_dir().join(format!(
            "sigil-vosk-{tag}-{}-{}",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    /// A small but real zip laid out like a Vosk model.
    fn model_zip() -> Vec<u8> {
        let mut w = zip::ZipWriter::new(io::Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);
        w.add_directory("vosk-model-test/", opts).unwrap();
        w.start_file("vosk-model-test/am/final.mdl", opts).unwrap();
        let body: Vec<u8> = (0..40_000u32).flat_map(|i| i.to_le_bytes()).collect();
        w.write_all(&body).unwrap();
        w.start_file("vosk-model-test/README", opts).unwrap();
        w.write_all(b"test model").unwrap();
        w.finish().unwrap().into_inner()
    }

    #[test]
    fn model_ids_are_validated() {
        assert!(validate_model_id("vosk-model-small-en-us-0.15").is_ok());
        for bad in ["", "../x", "a/b", "a\\b", "x..y"] {
            assert!(validate_model_id(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn local_import_streams_verifies_and_replaces_atomically() {
        let d = tmp("import");
        let src = d.join("model.zip");
        let bytes = model_zip();
        fs::write(&src, &bytes).unwrap();
        let dest = d.join("vosk-models").join("vosk-model-test.zip");
        let mut calls = 0;
        let sha = copy_vosk_zip_to_cache(&src, &dest, &ExtractLimits::default(), &mut |c, t| {
            calls += 1;
            assert!(c <= t);
            true
        })
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), bytes);
        assert_eq!(sha, hex::encode(Sha256::digest(&bytes)));
        assert!(calls >= 1);

        // A second import replaces the file; a failed one leaves it (and no temp) behind.
        fs::write(&src, b"<!DOCTYPE html><html>".repeat(1000)).unwrap();
        let err = copy_vosk_zip_to_cache(&src, &dest, &ExtractLimits::default(), &mut |_, _| true).unwrap_err();
        assert!(err.contains("HTML"), "{err}");
        let mut truncated = bytes.clone();
        truncated.truncate(bytes.len() - 200);
        fs::write(&src, &truncated).unwrap();
        assert!(copy_vosk_zip_to_cache(&src, &dest, &ExtractLimits::default(), &mut |_, _| true).is_err());
        fs::write(&src, b"tiny").unwrap();
        assert!(copy_vosk_zip_to_cache(&src, &dest, &ExtractLimits::default(), &mut |_, _| true).is_err());
        assert_eq!(fs::read(&dest).unwrap(), bytes, "the good zip survives failed imports");
        let leftovers: Vec<_> = fs::read_dir(dest.parent().unwrap())
            .unwrap()
            .flatten()
            .filter(|e| e.file_name().to_string_lossy().contains("partial"))
            .collect();
        assert!(leftovers.is_empty());

        // Cancel stops the copy and leaves the destination untouched.
        fs::write(&src, &bytes).unwrap();
        let err = copy_vosk_zip_to_cache(&src, &dest, &ExtractLimits::default(), &mut |_, _| false).unwrap_err();
        assert_eq!(err, DOWNLOAD_CANCELLED);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn catalog_vosk_entries_match_the_plugin_list() {
        let catalog = Catalog::embedded().unwrap();
        let models = get_available_models();
        let mut n = 0;
        for e in catalog.entries.iter().filter(|e| e.family == "vosk") {
            let m = models
                .iter()
                .find(|m| vosk_entry_id(&m.id) == e.id)
                .unwrap_or_else(|| panic!("{} is not a listed Vosk model", e.id));
            assert_eq!(e.files.len(), 1);
            assert_eq!(e.files[0].path, format!("{}.zip", m.id));
            assert!(e.files[0].sources.iter().any(|s| s.url == m.url), "{}", e.id);
            n += 1;
        }
        assert!(n >= 1, "the default Vosk model is in the catalog");
    }
}
