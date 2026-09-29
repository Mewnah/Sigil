use crate::services::audio::{resolve_input_device, spawn_input_stream};
use crate::services::http_client::{describe_failed_response, http_download_get};
use cpal::traits::DeviceTrait;
use futures::StreamExt;
use sha2::{Digest, Sha256};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{channel, Sender};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const CHUNK_DURATION_SECS: u64 = 5;

fn get_model_info(model: &str) -> (&'static str, &'static str) {
    // Returns (url, filename)
    // Quantized models (Q5_1, Q8_0) provide ~2-3x speedup with minimal quality loss
    match model {
        // Standard models
        "tiny.en" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin",
            "ggml-tiny.en.bin",
        ),
        "tiny" => ("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin", "ggml-tiny.bin"),
        "base.en" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            "ggml-base.en.bin",
        ),
        "base" => ("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin", "ggml-base.bin"),
        "small.en" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en.bin",
            "ggml-small.en.bin",
        ),
        "small" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
            "ggml-small.bin",
        ),
        "medium.en" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en.bin",
            "ggml-medium.en.bin",
        ),
        "medium" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
            "ggml-medium.bin",
        ),
        // Quantized Q5_1 models (best balance of speed and quality)
        "tiny.en-q5_1" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en-q5_1.bin",
            "ggml-tiny.en-q5_1.bin",
        ),
        "tiny-q5_1" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin",
            "ggml-tiny-q5_1.bin",
        ),
        "base.en-q5_1" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin",
            "ggml-base.en-q5_1.bin",
        ),
        "base-q5_1" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base-q5_1.bin",
            "ggml-base-q5_1.bin",
        ),
        "small.en-q5_1" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q5_1.bin",
            "ggml-small.en-q5_1.bin",
        ),
        "small-q5_1" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small-q5_1.bin",
            "ggml-small-q5_1.bin",
        ),
        "medium.en-q5_1" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q5_1.bin",
            "ggml-medium.en-q5_1.bin",
        ),
        "medium-q5_1" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium-q5_1.bin",
            "ggml-medium-q5_1.bin",
        ),
        // Quantized Q8_0 models (higher quality, slightly slower than Q5_1)
        "small.en-q8_0" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.en-q8_0.bin",
            "ggml-small.en-q8_0.bin",
        ),
        "medium.en-q8_0" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.en-q8_0.bin",
            "ggml-medium.en-q8_0.bin",
        ),
        // Large V3 Turbo (optimized for speed)
        "large-v3-turbo" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin",
            "ggml-large-v3-turbo.bin",
        ),
        "large-v3-turbo-q5_0" => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q5_0.bin",
            "ggml-large-v3-turbo-q5_0.bin",
        ),
        _ => (
            "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
            "ggml-base.en.bin",
        ),
    }
}

const MIN_WHISPER_BIN_BYTES: u64 = 100 * 1024;

const DOWNLOAD_CANCELLED: &str = "Model download cancelled.";

/// Per-call suffix for import temp files so overlapping imports never share (or delete) each other's file.
static IMPORT_SEQ: AtomicU64 = AtomicU64::new(0);

/// A `*.partial` / `*.partial_import` untouched for this long belongs to no live download or copy.
const STALE_PARTIAL_AGE: Duration = Duration::from_secs(300);

/// Minimum spacing of `*:download_progress` events (at most ~5 per second).
const PROGRESS_EMIT_INTERVAL: Duration = Duration::from_millis(200);

/// Chunk size of the local-model copy; progress and cancellation are checked once per chunk.
const COPY_CHUNK_BYTES: usize = 1 << 20;

/// Remove `*.partial` / `*.partial_import` left by a previous process (exit or crash mid-download or mid-copy).
/// Per-run names carry the process id plus an epoch/sequence, so no other run or instance reuses them.
/// Files touched in the last 5 minutes are kept (a second Sigil instance may be writing them).
pub(crate) fn sweep_stale_partials(dir: &Path) {
    sweep_partials_older_than(dir, STALE_PARTIAL_AGE);
}

fn sweep_partials_older_than(dir: &Path, max_age: Duration) {
    let entries = match fs::read_dir(dir) {
        Ok(rd) => rd,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !(name.ends_with(".partial") || name.ends_with(".partial_import")) {
            continue;
        }
        // Stat the file itself: on Windows `DirEntry::metadata` is the cached directory entry, whose mtime NTFS
        // refreshes only when the writer closes its handle, so another instance's live download would look old.
        let age = fs::metadata(entry.path())
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok());
        // Unknown age (no mtime, or an mtime in the future after a clock change) counts as stale; otherwise a
        // leftover could survive every sweep.
        let stale = match age {
            Some(age) => age >= max_age,
            None => true,
        };
        if stale && entry.file_type().map(|t| t.is_file()).unwrap_or(false) {
            let _ = fs::remove_file(entry.path());
        }
    }
}

/// Lets the first `*:download_progress` event through at once, then at most one per [`PROGRESS_EMIT_INTERVAL`].
pub(crate) struct ProgressThrottle {
    last: Option<Instant>,
}

impl ProgressThrottle {
    pub(crate) fn new() -> Self {
        Self { last: None }
    }

    /// True when an event may be emitted now (and records it as emitted).
    pub(crate) fn ready(&mut self) -> bool {
        let now = Instant::now();
        match self.last {
            Some(prev) if now.duration_since(prev) < PROGRESS_EMIT_INTERVAL => false,
            _ => {
                self.last = Some(now);
                true
            }
        }
    }
}

/// Percent (0-100) of `done` out of `total`, or `None` when the total length is unknown.
pub(crate) fn progress_percent(done: u64, total: u64) -> Option<f64> {
    if total == 0 {
        None
    } else {
        Some((done as f64 / total as f64 * 100.0).min(100.0))
    }
}

/// Copy `src` to `dest` in [`COPY_CHUNK_BYTES`] chunks. `on_progress(copied, total)` runs after every chunk;
/// returning `false` stops the copy with [`DOWNLOAD_CANCELLED`]. The caller removes `dest` on `Err`.
fn copy_with_progress(src: &Path, dest: &Path, on_progress: &mut dyn FnMut(u64, u64) -> bool) -> Result<(), String> {
    let mut input = File::open(src).map_err(|e| e.to_string())?;
    let total = input.metadata().map(|m| m.len()).unwrap_or(0);
    let mut output = File::create(dest).map_err(|e| e.to_string())?;
    let mut buf = vec![0u8; COPY_CHUNK_BYTES];
    let mut copied: u64 = 0;
    loop {
        let n = match input.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(e.to_string()),
        };
        output.write_all(&buf[..n]).map_err(|e| e.to_string())?;
        copied += n as u64;
        if !on_progress(copied, total) {
            return Err(DOWNLOAD_CANCELLED.to_string());
        }
    }
    output.flush().map_err(|e| e.to_string())
}

fn huggingface_model_url(base: &str) -> String {
    if base.contains('?') {
        base.to_string()
    } else {
        format!("{base}?download=true")
    }
}

fn validate_whisper_bin_src(path: &Path) -> Result<(), String> {
    let meta = fs::metadata(path).map_err(|e| e.to_string())?;
    if meta.len() < MIN_WHISPER_BIN_BYTES {
        return Err(
            "File is too small to be a valid Whisper model (expected a .bin GGML file)."
                .to_string(),
        );
    }
    let mut sniff = [0u8; 8];
    let mut f = File::open(path).map_err(|e| e.to_string())?;
    let n = f.read(&mut sniff).map_err(|e| e.to_string())?;
    if n > 0 && sniff[0] == b'<' {
        return Err("File looks like HTML, not a GGML Whisper model.".to_string());
    }
    Ok(())
}

fn copy_whisper_bin_to_model_path(
    src: &Path,
    dest: &Path,
    on_progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<(), String> {
    validate_whisper_bin_src(src)?;
    if let Some(parent) = dest.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let seq = IMPORT_SEQ.fetch_add(1, Ordering::Relaxed);
    let partial_path = dest.with_extension(format!("bin.{}.{seq}.partial_import", std::process::id()));
    let _ = fs::remove_file(&partial_path);
    if let Err(e) = copy_with_progress(src, &partial_path, on_progress) {
        let _ = fs::remove_file(&partial_path);
        return Err(e);
    }
    // `rename` replaces an existing destination in one step (Windows and Unix), so `dest` is never missing.
    fs::rename(&partial_path, dest).map_err(|e| {
        let _ = fs::remove_file(&partial_path);
        e.to_string()
    })?;
    Ok(())
}

/// Copy a user-picked `.bin` to `dest` on a blocking thread (a medium/large model is 0.5-1.6 GB), emitting
/// `whisper:download_progress` for `file` at most ~5 times per second. `live()` is polled once per chunk: when it
/// returns false the copy stops with [`DOWNLOAD_CANCELLED`] and nothing more is emitted.
async fn copy_whisper_bin_with_progress<R: Runtime, L>(
    app: &AppHandle<R>,
    src: PathBuf,
    dest: PathBuf,
    file: &'static str,
    live: L,
) -> Result<(), String>
where
    L: Fn() -> bool + Send + 'static,
{
    let app = app.clone();
    tauri::async_runtime::spawn_blocking(move || {
        let mut throttle = ProgressThrottle::new();
        let mut on_progress = |copied: u64, total: u64| {
            if !live() {
                return false;
            }
            if let Some(progress) = progress_percent(copied, total) {
                if copied >= total || throttle.ready() {
                    let _ = app.emit(
                        "whisper:download_progress",
                        ProgressPayload {
                            file: file.to_string(),
                            progress,
                        },
                    );
                }
            }
            true
        };
        copy_whisper_bin_to_model_path(&src, &dest, &mut on_progress)
    })
    .await
    .map_err(|e| format!("Model copy task failed: {}", e))?
}

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
    /// Bumped (under `stop_sender`'s lock) by every `start_recording` and `stop_recording`. A start re-checks it
    /// under the same lock before publishing, so a stop that lands while the device is still opening cancels it.
    recording_epoch: Arc<AtomicU64>,
    audio_buffer: Arc<Mutex<Vec<f32>>>,
    sample_rate: Arc<Mutex<u32>>,
    channels: Arc<Mutex<u16>>,
    is_recording: Arc<Mutex<bool>>,
    vad_config: Arc<Mutex<VadConfig>>,
    consecutive_silent_frames: Arc<Mutex<u32>>,
    last_transcription_time: Arc<Mutex<Instant>>,
    ctx: Arc<Mutex<Option<WhisperContext>>>,
    /// Model file whose weights are in `ctx` (updated together with `ctx`, lock order ctx -> this).
    loaded_model_file: Arc<Mutex<Option<String>>>,
    /// Language used by `transcribe_chunk`; written only by a current (non-stale) `ensure_dependencies` run.
    current_language: Arc<Mutex<String>>,
    /// Bumped on each `ensure_dependencies` start and on cancel; stale runs must not mutate state or emit,
    /// and only ever touch their own temp files.
    ensure_deps_epoch: Arc<AtomicU64>,
    /// Held across a model load (build + publish), so a restart during a load waits for it instead of building a
    /// second `WhisperContext` alongside it (2x model RAM). Async so a waiting run does not block a runtime thread.
    load_lock: Arc<tokio::sync::Mutex<()>>,
}

impl WhisperState {
    fn new() -> Self {
        Self {
            stop_sender: Arc::new(Mutex::new(None)),
            recording_epoch: Arc::new(AtomicU64::new(0)),
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
            loaded_model_file: Arc::new(Mutex::new(None)),
            current_language: Arc::new(Mutex::new("en".to_string())),
            ensure_deps_epoch: Arc::new(AtomicU64::new(0)),
            load_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }
}

#[derive(Clone, serde::Serialize)]
struct ProgressPayload {
    file: String,
    progress: f64,
}

#[derive(Clone, serde::Serialize)]
struct DownloadCompletePayload {
    file: String,
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

#[allow(dead_code)]
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

fn ensure_deps_stale(state: &WhisperState, my_epoch: u64) -> bool {
    state.ensure_deps_epoch.load(Ordering::SeqCst) != my_epoch
}

/// Stream `model_url` into this run's own `partial_path` and validate it. The caller removes the partial on `Err`.
async fn fetch_model_to_partial<R: Runtime>(
    app: &AppHandle<R>,
    state: &WhisperState,
    my_epoch: u64,
    model_url: &str,
    model_filename: &str,
    partial_path: &Path,
) -> Result<(), String> {
    let response = http_download_get(model_url)
        .send()
        .await
        .map_err(|e| format!("Model download failed: {}", e))?;

    if !response.status().is_success() {
        let detail = describe_failed_response(response).await;
        return Err(format!("Model download failed: {detail}"));
    }

    let total = response.content_length().unwrap_or(0);
    let mut downloaded: u64 = 0;
    let mut stream = response.bytes_stream();
    let mut throttle = ProgressThrottle::new();

    {
        let mut file = File::create(partial_path).map_err(|e| e.to_string())?;
        while let Some(chunk) = stream.next().await {
            if ensure_deps_stale(state, my_epoch) {
                return Err(DOWNLOAD_CANCELLED.to_string());
            }
            let chunk = chunk.map_err(|e| e.to_string())?;
            file.write_all(&chunk).map_err(|e| e.to_string())?;
            downloaded += chunk.len() as u64;
            // One event per network chunk would flood IPC (thousands per second); ~5 per second is plenty.
            if downloaded > 0 && throttle.ready() {
                let _ = app.emit(
                    "whisper:download_progress",
                    ProgressPayload {
                        file: model_filename.to_string(),
                        // Unknown total length: 0 still shows activity (indeterminate % in the UI).
                        progress: progress_percent(downloaded, total).unwrap_or(0.0),
                    },
                );
            }
        }
        if total > 0 && !ensure_deps_stale(state, my_epoch) {
            let _ = app.emit(
                "whisper:download_progress",
                ProgressPayload {
                    file: model_filename.to_string(),
                    progress: 100.0,
                },
            );
        }
    }

    let meta = fs::metadata(partial_path).map_err(|e| e.to_string())?;
    if meta.len() < MIN_WHISPER_BIN_BYTES {
        return Err("Downloaded file is too small to be a valid Whisper model (got an error page?)".to_string());
    }

    let mut sniff = [0u8; 8];
    let mut sniff_file = File::open(partial_path).map_err(|e| e.to_string())?;
    let n = sniff_file.read(&mut sniff).map_err(|e| e.to_string())?;
    if n > 0 && sniff[0] == b'<' {
        return Err("Download returned HTML instead of a model file.".to_string());
    }
    Ok(())
}

#[command]
async fn ensure_dependencies<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, WhisperState>,
    model: String,
    language: String,
    local_model_path: Option<String>,
) -> Result<(), String> {
    // A newer run or `cancel_whisper_download` bumps the epoch; from then on this run is stale.
    let my_epoch = state.ensure_deps_epoch.fetch_add(1, Ordering::SeqCst) + 1;

    let (model_url_base, model_filename) = get_model_info(&model);
    let model_url = huggingface_model_url(model_url_base);

    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    let whisper_dir = app_data_dir.join("whisper");
    let model_path = whisper_dir.join(model_filename);

    if !whisper_dir.exists() {
        fs::create_dir_all(&whisper_dir).map_err(|e| e.to_string())?;
    }

    // Reload when nothing is loaded, a different model is loaded, or the file still has to be fetched.
    let need_reload = {
        let ctx_guard = state.ctx.lock().map_err(|_| "Failed to lock ctx")?;
        let loaded = state.loaded_model_file.lock().map_err(|_| "Lock failed")?;
        ctx_guard.is_none() || loaded.as_deref() != Some(model_filename) || !model_path.exists()
    };

    // Set when this run put the model file in place (download or local copy): the inspector's progress row then
    // gets `whisper:download_complete` instead of staying on its last progress value.
    let mut fetched_model = false;
    if !model_path.exists() {
        if let Some(ref lp) = local_model_path {
            let trimmed = lp.trim();
            if !trimmed.is_empty() {
                if ensure_deps_stale(&state, my_epoch) {
                    return Err(DOWNLOAD_CANCELLED.to_string());
                }
                let _ = app.emit("whisper:load_status", "Copying local Whisper model...");
                // Cancellable like a download: a stop or a newer run ends the copy at the next 1 MiB chunk.
                let epoch = state.ensure_deps_epoch.clone();
                copy_whisper_bin_with_progress(
                    &app,
                    PathBuf::from(trimmed),
                    model_path.clone(),
                    model_filename,
                    move || epoch.load(Ordering::SeqCst) == my_epoch,
                )
                .await?;
                fetched_model = true;
            }
        }

        if !model_path.exists() {
            fetched_model = true;
            // Per-run name (process id + epoch): overlapping runs, and a second Sigil instance whose epoch
            // counter also restarted at launch, never share, rename or delete each other's partial file.
            let partial_path =
                whisper_dir.join(format!("{}.{}.{}.partial", model_filename, std::process::id(), my_epoch));
            let _ = fs::remove_file(&partial_path);

            if let Err(e) = fetch_model_to_partial(&app, &state, my_epoch, &model_url, model_filename, &partial_path).await {
                let _ = fs::remove_file(&partial_path);
                return Err(e);
            }
            if ensure_deps_stale(&state, my_epoch) {
                let _ = fs::remove_file(&partial_path);
                return Err(DOWNLOAD_CANCELLED.to_string());
            }
            if model_path.exists() {
                // An overlapping run committed this model first; keep its file and drop ours.
                let _ = fs::remove_file(&partial_path);
            } else if let Err(e) = fs::rename(&partial_path, &model_path) {
                let _ = fs::remove_file(&partial_path);
                return Err(e.to_string());
            }
            // From here on the committed model file is valid and is never deleted, even if this run goes stale.
        }
    }

    if need_reload {
        if ensure_deps_stale(&state, my_epoch) {
            return Err(DOWNLOAD_CANCELLED.to_string());
        }
        let _ = app.emit("whisper:load_status", "Loading Whisper model into memory...");
        // One load at a time: a restart during a load waits here for the stale run to finish and drop its context,
        // instead of holding two contexts at once. Declared before `ctx`, so on every return path the context is
        // dropped before this guard is released.
        let _load_guard = state.load_lock.lock().await;
        if ensure_deps_stale(&state, my_epoch) {
            return Err("Model load cancelled.".to_string());
        }
        let path_str = model_path.to_string_lossy().into_owned();
        let ctx = tauri::async_runtime::spawn_blocking(move || {
            WhisperContext::new_with_params(&path_str, WhisperContextParameters::default())
        })
        .await
        .map_err(|e| format!("Model load task failed: {}", e))?
        .map_err(|e| format!("Failed to load context: {}", e))?;
        let previous = {
            let mut ctx_guard = state.ctx.lock().map_err(|_| "Failed to lock ctx")?;
            // Checked under the lock `transcribe_chunk` holds across inference (seconds): a cancel or restart that
            // lands while this run waits for it is seen here, and the stale context is never published.
            if ensure_deps_stale(&state, my_epoch) {
                return Err("Model load cancelled.".to_string());
            }
            let mut loaded = state.loaded_model_file.lock().map_err(|_| "Lock failed")?;
            *loaded = Some(model_filename.to_string());
            ctx_guard.replace(ctx)
        };
        // Free the replaced model outside the ctx lock so transcription is not blocked while it is released.
        drop(previous);
        if !ensure_deps_stale(&state, my_epoch) {
            let _ = app.emit("whisper:load_status", "");
        }
    }

    {
        // Same as above: `transcribe_chunk` holds this lock across inference, so check staleness only once held.
        let mut lang = state.current_language.lock().map_err(|_| "Lock failed")?;
        if ensure_deps_stale(&state, my_epoch) {
            return Err(DOWNLOAD_CANCELLED.to_string());
        }
        *lang = language;
    }

    if fetched_model {
        let _ = app.emit(
            "whisper:download_complete",
            DownloadCompletePayload {
                file: model_filename.to_string(),
            },
        );
    }
    Ok(())
}

/// Copy a user-selected `.bin` into app data for the given catalog model id (same layout as download).
#[command]
async fn import_whisper_model<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, WhisperState>,
    model: String,
    source_path: String,
) -> Result<(), String> {
    let (_, model_filename) = get_model_info(&model);
    let app_data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let whisper_dir = app_data_dir.join("whisper");
    if !whisper_dir.exists() {
        fs::create_dir_all(&whisper_dir).map_err(|e| e.to_string())?;
    }
    let dest = whisper_dir.join(model_filename);
    // The inspector shows the copy on its download row; the caller's toast reports the result, so the row is
    // dismissed (not marked complete) once the copy ends either way.
    let copied = copy_whisper_bin_with_progress(
        &app,
        PathBuf::from(source_path.trim()),
        dest,
        model_filename,
        || true,
    )
    .await;
    let _ = app.emit("whisper:download_dismiss", ());
    copied?;
    // The file on disk changed: make the next `ensure_dependencies` reload it instead of reusing `ctx`.
    let mut loaded = state.loaded_model_file.lock().map_err(|_| "Lock failed")?;
    if loaded.as_deref() == Some(model_filename) {
        *loaded = None;
    }
    Ok(())
}

fn transcribe_chunk(state: &WhisperState, audio_data: Vec<f32>, sample_rate: u32, channels: u16) -> Result<String, String> {
    let mono_data = if channels > 1 {
        let ch = channels as usize;
        audio_data
            .chunks_exact(ch)
            .map(|c| c.iter().sum::<f32>() / ch as f32)
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
    // Get language from state (use None for auto-detect)
    let lang = state.current_language.lock().map_err(|_| "Lock failed")?;
    let lang_opt = if lang.as_str() == "auto" { None } else { Some(lang.as_str()) };
    params.set_language(lang_opt);
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);

    w_state
        .full(params, &resampled)
        .map_err(|e| format!("Inference failed: {}", e))?;

    let num_segments = w_state.full_n_segments();
    let mut text = String::new();
    for i in 0..num_segments {
        let segment = w_state
            .get_segment(i)
            .ok_or_else(|| format!("Missing segment {}", i))?;
        let segment_text = segment.to_str().map_err(|e| e.to_string())?;
        text.push_str(segment_text);
        text.push(' ');
    }
    Ok(text.trim().to_string())
}

// The argument list is the IPC contract with `whisper.ts` (one named field per setting).
#[allow(clippy::too_many_arguments)]
#[command]
async fn start_recording<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, WhisperState>,
    device_name: Option<String>,
    vad_enabled: bool,
    silence_threshold_db: f32,
    silence_duration_ms: u64,
    min_chunk_duration_ms: u64,
    capture_local: bool,
) -> Result<(), String> {
    // Stop any previous recording and claim an epoch before the (slow) device open.
    let my_epoch = {
        let mut sender = state.stop_sender.lock().unwrap();
        *state.is_recording.lock().unwrap() = false;
        if let Some(old) = sender.take() {
            let _ = old.send(());
        }
        state.recording_epoch.fetch_add(1, Ordering::SeqCst) + 1
    };

    let (tx, rx) = channel();
    let (config_sample_rate, config_channels) = if capture_local {
        let host = cpal::default_host();

        let device = resolve_input_device(&host, device_name.as_deref())?;

        let config = device.default_input_config().map_err(|e| e.to_string())?;
        let rate_and_channels = (config.sample_rate().0, config.channels());
        let state_clone = state.inner().clone();
        let app_clone = app.clone();
        // Drops data until `is_recording` is published below, and for good once a stop or a newer start
        // supersedes `my_epoch` (this stream may outlive that briefly while its thread shuts it down).
        spawn_input_stream(
            device,
            config,
            rx,
            move |data| process_audio_chunk_for(&state_clone, &app_clone, data, Some(my_epoch)),
            "Whisper",
        )
        .await?;
        rate_and_channels
    } else {
        // Audio arrives through `feed_audio_chunk`; this thread only mirrors the recording lifecycle.
        thread::spawn(move || {
            let _ = rx.recv();
        });
        (16000, 1)
    };

    let mut sender = state.stop_sender.lock().unwrap();
    if state.recording_epoch.load(Ordering::SeqCst) != my_epoch {
        // A stop (or a newer start) landed while the device was opening. Dropping `tx` on return ends the
        // capture thread, which drops the stream and releases the microphone.
        return Err("cancelled".to_string());
    }
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
    if let Some(old) = sender.replace(tx) {
        let _ = old.send(());
    }
    Ok(())
}

/// Feed samples that do not come from a Whisper capture stream (`feed_audio_chunk`, the web pubsub bridge).
pub fn process_audio_chunk<R: Runtime>(state: &WhisperState, app: &AppHandle<R>, data: Vec<f32>) {
    process_audio_chunk_for(state, app, data, None);
}

/// `capture_epoch`: the `recording_epoch` of the start that opened the calling capture stream (`None` for fed
/// audio). Samples are tagged with the epoch they arrived under; once a stop or a newer start bumps it, a
/// superseded stream adds nothing and a segment still being transcribed for the old epoch is never emitted.
fn process_audio_chunk_for<R: Runtime>(state: &WhisperState, app: &AppHandle<R>, data: Vec<f32>, capture_epoch: Option<u64>) {
    // The session these samples belong to. If a start or stop bumps the epoch after this read, the re-checks under
    // the buffer lock below drop the samples (a start also clears the buffer only after bumping).
    let epoch = state.recording_epoch.load(Ordering::SeqCst);
    if capture_epoch.is_some_and(|mine| mine != epoch) {
        return;
    }
    if !*state.is_recording.lock().unwrap() {
        return;
    }
    {
        let mut buf = state.audio_buffer.lock().unwrap();
        // Re-checked under the buffer lock: a start bumps the epoch before it clears this buffer, so samples from a
        // superseded session are either rejected here or cleared by that start.
        if state.recording_epoch.load(Ordering::SeqCst) != epoch {
            return;
        }
        buf.extend_from_slice(&data);
    }

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
        let chunk = {
            let mut buf = state.audio_buffer.lock().unwrap();
            if state.recording_epoch.load(Ordering::SeqCst) != epoch {
                return;
            }
            std::mem::take(&mut *buf)
        };

        if chunk.len() < 3200 {
            return;
        }
        let app = app.clone();
        let state_clone = state.clone();

        thread::spawn(move || {
            if let Ok(text) = transcribe_chunk(&state_clone, chunk, sample_rate, channels) {
                if !text.trim().is_empty() {
                    // Starts and stops bump the epoch under `stop_sender`'s lock; holding it here means that once a
                    // restart or stop has returned, no segment of the session it ended can still be emitted (it
                    // would reach the next session's listener as a final).
                    let Ok(_sender) = state_clone.stop_sender.lock() else { return };
                    if state_clone.recording_epoch.load(Ordering::SeqCst) != epoch {
                        return;
                    }
                    // One completed VAD/time segment (not a streaming partial). Frontend maps this to STT `onFinal` per segment.
                    let _ = app.emit("whisper:segment_transcript", text);
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
fn cancel_whisper_download(state: State<'_, WhisperState>) -> Result<(), String> {
    // Makes every in-flight `ensure_dependencies` run stale; each one then removes only its own partial file.
    state.ensure_deps_epoch.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

#[command]
async fn stop_recording<R: Runtime>(_app: AppHandle<R>, state: State<'_, WhisperState>) -> Result<String, String> {
    let tx = {
        let mut sender = state.stop_sender.lock().unwrap();
        // Cancels a `start_recording` still opening its device (it re-checks the epoch under this lock).
        state.recording_epoch.fetch_add(1, Ordering::SeqCst);
        *state.is_recording.lock().unwrap() = false;
        // Samples are epoch-checked under the buffer lock, so nothing from the ended session lands after this
        // clear. (Clearing here, not after the wait below, also never wipes a session started meanwhile.)
        state.audio_buffer.lock().unwrap().clear();
        sender.take()
    };
    match tx {
        Some(tx) => {
            let _ = tx.send(());
        }
        None => return Err("Not recording".to_string()),
    }
    // Give the capture thread a moment to drop its stream (release the microphone) before the caller moves on;
    // an async sleep so no runtime worker is blocked meanwhile.
    tokio::time::sleep(Duration::from_millis(200)).await;
    Ok(String::new())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("whisper")
        .setup(|app, _api| {
            app.manage(WhisperState::new());
            if let Ok(dir) = app.path().app_data_dir() {
                sweep_stale_partials(&dir.join("whisper"));
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            ensure_dependencies,
            import_whisper_model,
            cancel_whisper_download,
            start_recording,
            stop_recording,
            feed_audio_chunk
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "sigil-whisper-test-{tag}-{}-{}",
            std::process::id(),
            IMPORT_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn sweep_removes_only_old_partials() {
        let dir = temp_dir("sweep");
        for name in [
            "ggml-base.en.bin.3.partial",
            "ggml-base.en.bin.0.partial_import",
            "vosk-model-small-en-us-0.15.zip.1.partial",
            "ggml-base.en.bin",
            "vosk-model-small-en-us-0.15.zip",
        ] {
            fs::write(dir.join(name), b"x").unwrap();
        }
        // Fresh files are kept (another instance may be writing them).
        sweep_stale_partials(&dir);
        assert_eq!(fs::read_dir(&dir).unwrap().count(), 5);
        // Once old enough, only the partials go.
        sweep_partials_older_than(&dir, Duration::ZERO);
        let mut left: Vec<String> = fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        left.sort();
        assert_eq!(left, vec!["ggml-base.en.bin", "vosk-model-small-en-us-0.15.zip"]);
        // A missing directory is not an error.
        sweep_stale_partials(&dir.join("missing"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn progress_throttle_lets_first_through_then_rate_limits() {
        let mut t = ProgressThrottle::new();
        assert!(t.ready());
        assert!(!t.ready());
        t.last = Instant::now().checked_sub(PROGRESS_EMIT_INTERVAL);
        assert!(t.ready());
        assert!(!t.ready());
    }

    #[test]
    fn progress_percent_handles_unknown_and_overshoot() {
        assert_eq!(progress_percent(5, 0), None);
        assert_eq!(progress_percent(50, 200), Some(25.0));
        assert_eq!(progress_percent(300, 200), Some(100.0));
    }

    #[test]
    fn local_model_copy_reports_progress_and_can_be_cancelled() {
        let dir = temp_dir("copy");
        let src = dir.join("src.bin");
        let data: Vec<u8> = (0..(COPY_CHUNK_BYTES * 2 + 1234)).map(|i| (i % 251) as u8).collect();
        fs::write(&src, &data).unwrap();

        let dest = dir.join("ggml-base.en.bin");
        let mut calls = Vec::new();
        copy_whisper_bin_to_model_path(&src, &dest, &mut |copied, total| {
            calls.push((copied, total));
            true
        })
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data);
        let total = data.len() as u64;
        assert_eq!(calls.last(), Some(&(total, total)));
        assert!(calls.len() >= 3);

        // Cancelled after the first chunk: error, no partial left, and the committed file is untouched.
        let dest2 = dir.join("ggml-tiny.bin");
        let err = copy_whisper_bin_to_model_path(&src, &dest2, &mut |_, _| false).unwrap_err();
        assert_eq!(err, DOWNLOAD_CANCELLED);
        assert!(!dest2.exists());
        let partials = fs::read_dir(&dir)
            .unwrap()
            .filter(|e| e.as_ref().unwrap().file_name().to_string_lossy().ends_with(".partial_import"))
            .count();
        assert_eq!(partials, 0);
        assert_eq!(fs::read(&dest).unwrap(), data);

        // HTML (an error page saved as .bin) is rejected before any copy.
        let html = dir.join("page.bin");
        let mut page = b"<!DOCTYPE html>".to_vec();
        page.resize(MIN_WHISPER_BIN_BYTES as usize, b' ');
        fs::write(&html, &page).unwrap();
        assert!(copy_whisper_bin_to_model_path(&html, &dir.join("ggml-small.bin"), &mut |_, _| true).is_err());
        let _ = fs::remove_dir_all(&dir);
    }
}
