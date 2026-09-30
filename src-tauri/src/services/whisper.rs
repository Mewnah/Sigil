//! `whisper` plugin: local Whisper (whisper-rs / whisper.cpp, CPU) model management and the legacy
//! recording commands.
//!
//! Since P2 the recording runs through the speech pipeline (`speech::pipeline`): capture hub (raw
//! 16 kHz mono, before any voice conversion; device loss recovered; the STT mute gate upstream) →
//! Silero v6.2 segmenter → the ordered utterance worker (latency budget, drop-oldest backlog, merge
//! on backlog) → the Whisper engine (`speech::engines::whisper`: one reused `WhisperState`,
//! no-speech drop, token cap, prompt / hotwords) → hallucination / repeat filter. `start_recording`
//! keeps its IPC contract and events; it starts the crate-internal pipeline key `legacy-whisper`
//! with an engine over the model `ensure_dependencies` loaded, and maps the pipeline's events back
//! to the `whisper:*` names below. Silence never reaches Whisper, and no PCM crosses IPC.
//!
//! Models: catalog models download through the model store (`model_store` entries
//! `whisper/<id>`: pinned sha256, resume, mirrors, one download at a time), which installs them in
//! the same `<app_data>/whisper/` folder and adopts files already there. A local `.bin`
//! ("Use local model file") or an imported one is copied in and registered with the store when it
//! matches the catalog. Anime-Whisper is bring-your-own.
//!
//! Events (the legacy recording's, all emitted under the pipeline's epoch discipline):
//! - `whisper:segment_transcript` (string): one final transcript per utterance (or merged batch).
//! - `whisper:speech_start` (`{ seq, start_ms }`): speech detected (`seq` is the pipeline segment id).
//! - `whisper:utterance_dropped` (`{ seq, audio_ms, reason }`): `reason` is `"backlog"` (the model
//!   runs slower than real time and the oldest waiting utterance was discarded) or
//!   `"latency_budget"` (an utterance waited longer than 20 s).
//! - `whisper:capture_error` (`{ message }`): the recording stopped itself (speech detection failed,
//!   the model failed repeatedly). A lost or unplugged device is *not* an error: the capture hub
//!   reconnects it (`audio:stream_error` / `audio:stream_recovered`) and the recording continues.
//! - `whisper:download_progress`, `whisper:load_status`, `whisper:download_complete`,
//!   `whisper:download_dismiss`: model fetch / load (unchanged).

use crate::services::model_store::{self, register_legacy_file, CancelToken, Digest, Origin};
use crate::services::speech::engine_api::{AsrEngine, EngineKind, EngineSpec, PipelineConfig, RouteRequest};
use crate::services::speech::engines::whisper::{self as whisper_engine, LoadedModel, SharedModel, WhisperEngine};
use crate::services::speech::events::{
    SpeechFinal, SpeechPartial, SpeechState, SpeechStatus, SpeechTurnEnd, SpeechUtterance,
};
use crate::services::speech::pipeline::{self, EngineFactory, EventSink, SessionDeps, INTERNAL_PREFIX};
use crate::services::speech::utterance::reasons;
use crate::services::speech_core::VadConfig;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use futures::future::BoxFuture;
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};
use tokio::sync::broadcast::error::RecvError;

/// Log target of the Whisper plugin (file log via `diagnostics`).
const LOG: &str = "sigil::stt::whisper";

/// Folder under the app data folder that holds the ggml models (shared with the model store).
pub(crate) const WHISPER_FOLDER: &str = "whisper";

/// Pipeline key of the legacy recording (crate-internal).
pub(crate) const LEGACY_PIPELINE_ID: &str = "legacy-whisper";
const _: () = assert!(LEGACY_PIPELINE_ID.len() > INTERNAL_PREFIX.len());

/// Model id → ggml file (the ids the inspector persists; the model store's `whisper/<id>`
/// entries, `model-catalog/stt-legacy.json`, carry the URLs and hashes). Checked against the
/// repository's file list (revision 5359861c, 2026-09).
const CATALOG: &[(&str, &str)] = &[
    ("tiny.en", "ggml-tiny.en.bin"),
    ("tiny", "ggml-tiny.bin"),
    ("base.en", "ggml-base.en.bin"),
    ("base", "ggml-base.bin"),
    ("small.en", "ggml-small.en.bin"),
    ("small", "ggml-small.bin"),
    ("medium.en", "ggml-medium.en.bin"),
    ("medium", "ggml-medium.bin"),
    // Q5_1 (tiny/base/small) and Q5_0 (medium/large): ~2-3x faster, minimal quality loss.
    ("tiny.en-q5_1", "ggml-tiny.en-q5_1.bin"),
    ("tiny-q5_1", "ggml-tiny-q5_1.bin"),
    ("base.en-q5_1", "ggml-base.en-q5_1.bin"),
    ("base-q5_1", "ggml-base-q5_1.bin"),
    ("small.en-q5_1", "ggml-small.en-q5_1.bin"),
    ("small-q5_1", "ggml-small-q5_1.bin"),
    ("medium.en-q5_0", "ggml-medium.en-q5_0.bin"),
    ("medium-q5_0", "ggml-medium-q5_0.bin"),
    // Q8_0: closest to full precision.
    ("tiny.en-q8_0", "ggml-tiny.en-q8_0.bin"),
    ("tiny-q8_0", "ggml-tiny-q8_0.bin"),
    ("base.en-q8_0", "ggml-base.en-q8_0.bin"),
    ("base-q8_0", "ggml-base-q8_0.bin"),
    ("small.en-q8_0", "ggml-small.en-q8_0.bin"),
    ("small-q8_0", "ggml-small-q8_0.bin"),
    ("medium.en-q8_0", "ggml-medium.en-q8_0.bin"),
    ("medium-q8_0", "ggml-medium-q8_0.bin"),
    // Large v3 Turbo (multilingual, fastest large model) and Large v3.
    ("large-v3-turbo", "ggml-large-v3-turbo.bin"),
    ("large-v3-turbo-q5_0", "ggml-large-v3-turbo-q5_0.bin"),
    ("large-v3-turbo-q8_0", "ggml-large-v3-turbo-q8_0.bin"),
    ("large-v3", "ggml-large-v3.bin"),
    ("large-v3-q5_0", "ggml-large-v3-q5_0.bin"),
];

/// Legacy ids that never existed upstream (they 404ed) → the real quantization.
const ALIASES: &[(&str, &str)] = &[("medium-q5_1", "medium-q5_0"), ("medium.en-q5_1", "medium.en-q5_0")];

/// Bring-your-own Anime-Whisper (litagin/anime-whisper, a Japanese fine-tune of Whisper large-v3
/// turbo for expressive, anime-style speech). There is no official ggml build: convert it with
/// whisper.cpp's `models/convert-h5-to-ggml.py` and select the .bin as a local model (or import it).
pub(crate) const ANIME_WHISPER_ID: &str = "anime-whisper";

/// A Whisper model the plugin can load.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct WhisperModelInfo {
    /// Canonical id (aliases resolved).
    pub id: &'static str,
    /// File name under `<app_data>/whisper/`.
    pub filename: &'static str,
    /// Must be supplied by the user (local file or import); never downloaded.
    pub bring_your_own: bool,
    /// Language forced for this model (Japanese-only fine-tunes).
    pub language: Option<&'static str>,
    /// Whether the initial prompt / hotwords are fed to this model (Anime-Whisper's authors warn
    /// that prompts make it hallucinate).
    pub accepts_prompt: bool,
}

/// Resolve a model id; unknown ids are an error (they used to fall back to base.en silently).
pub(crate) fn get_model_info(model: &str) -> Result<WhisperModelInfo, String> {
    let id = model.trim();
    if id == ANIME_WHISPER_ID {
        return Ok(WhisperModelInfo {
            id: ANIME_WHISPER_ID,
            filename: "ggml-anime-whisper.bin",
            bring_your_own: true,
            language: Some("ja"),
            accepts_prompt: false,
        });
    }
    let id = ALIASES.iter().find(|(from, _)| *from == id).map_or(id, |(_, to)| *to);
    CATALOG
        .iter()
        .find(|(cid, _)| *cid == id)
        .map(|&(id, filename)| WhisperModelInfo {
            id,
            filename,
            bring_your_own: false,
            language: None,
            accepts_prompt: true,
        })
        .ok_or_else(|| format!("Unknown Whisper model '{}'. Pick a model from the list.", model.trim()))
}

/// Message for a bring-your-own model whose file is missing.
pub(crate) fn bring_your_own_missing(info: &WhisperModelInfo) -> String {
    if info.id == ANIME_WHISPER_ID {
        "Anime-Whisper is bring-your-own: convert litagin/anime-whisper to a ggml .bin with whisper.cpp's \
         models/convert-h5-to-ggml.py, then choose that file with \"Use local model file\" (or import it)."
            .to_string()
    } else {
        format!("Model '{}' is bring-your-own: choose its .bin file as a local model.", info.id)
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

/// How often a store download checks whether this `ensure_dependencies` run went stale.
const STALE_POLL: Duration = Duration::from_millis(250);

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

/// Copy `src` to `dest` in [`COPY_CHUNK_BYTES`] chunks, hashing it (sha256, hex). `on_progress(copied, total)`
/// runs after every chunk; returning `false` stops the copy with [`DOWNLOAD_CANCELLED`]. The caller removes
/// `dest` on `Err`.
fn copy_with_progress(src: &Path, dest: &Path, on_progress: &mut dyn FnMut(u64, u64) -> bool) -> Result<String, String> {
    let mut input = File::open(src).map_err(|e| e.to_string())?;
    let total = input.metadata().map(|m| m.len()).unwrap_or(0);
    let mut output = File::create(dest).map_err(|e| e.to_string())?;
    let mut hasher = Sha256::new();
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
        hasher.update(&buf[..n]);
        copied += n as u64;
        if !on_progress(copied, total) {
            return Err(DOWNLOAD_CANCELLED.to_string());
        }
    }
    output.flush().map_err(|e| e.to_string())?;
    Ok(hex::encode(hasher.finalize()))
}

/// A ggml Whisper model file of plausible size that is not an HTML error page.
pub(crate) fn validate_whisper_bin_src(path: &Path) -> Result<(), String> {
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

/// Copy a validated `.bin` into place through a per-call temp file; returns its sha256.
fn copy_whisper_bin_to_model_path(
    src: &Path,
    dest: &Path,
    on_progress: &mut dyn FnMut(u64, u64) -> bool,
) -> Result<String, String> {
    validate_whisper_bin_src(src)?;
    if let Some(parent) = dest.parent() {
        if !parent.exists() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
    }
    let seq = IMPORT_SEQ.fetch_add(1, Ordering::Relaxed);
    let partial_path = dest.with_extension(format!("bin.{}.{seq}.partial_import", std::process::id()));
    let _ = fs::remove_file(&partial_path);
    let sha = match copy_with_progress(src, &partial_path, on_progress) {
        Ok(sha) => sha,
        Err(e) => {
            let _ = fs::remove_file(&partial_path);
            return Err(e);
        }
    };
    // `rename` replaces an existing destination in one step (Windows and Unix), so `dest` is never missing.
    fs::rename(&partial_path, dest).map_err(|e| {
        let _ = fs::remove_file(&partial_path);
        e.to_string()
    })?;
    Ok(sha)
}

/// Copy a user-picked `.bin` to `dest` on a blocking thread (a medium/large model is 0.5-1.6 GB), emitting
/// `whisper:download_progress` for `file` at most ~5 times per second. `live()` is polled once per chunk: when it
/// returns false the copy stops with [`DOWNLOAD_CANCELLED`] and nothing more is emitted. Returns the sha256.
async fn copy_whisper_bin_with_progress<R: Runtime, L>(
    app: &AppHandle<R>,
    src: PathBuf,
    dest: PathBuf,
    file: &'static str,
    live: L,
) -> Result<String, String>
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
                    emit_progress(&app, file, progress);
                }
            }
            true
        };
        copy_whisper_bin_to_model_path(&src, &dest, &mut on_progress)
    })
    .await
    .map_err(|e| format!("Model copy task failed: {}", e))?
}

/// Register `whisper/<file>` with the model store when it is the catalog's file (the Models tab
/// then shows it installed); a custom model under a catalog name is left unregistered.
fn register_with_store(model_id: &str, sha256: &str, origin: Origin) {
    let Ok(store) = model_store::global_store() else { return };
    if let Some(entry) = store.catalog.get(&model_store::whisper_entry_id(model_id)) {
        if register_legacy_file(&store, entry, &Digest::Sha256(sha256.to_string()), origin) {
            store.set_state(&entry.id, None);
        }
    }
}

/// Decoder options of `start_recording` (`decode`); every field optional, snake_case like
/// `VadConfig`. TS mirror: `RustWhisperDecodeOptions` in `src/types.ts`.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
#[serde(default)]
pub struct WhisperDecodeOptions {
    /// CPU threads for inference (default: half the logical CPUs, 1..=4; VRChat is CPU-bound).
    pub n_threads: Option<u32>,
    /// Decoder prompt: style / vocabulary hint (ignored by presets that do not take prompts).
    pub initial_prompt: Option<String>,
    /// Words to bias recognition towards; appended to the prompt as a comma-separated list.
    pub hotwords: Option<Vec<String>>,
    /// Drop an utterance whose no-speech probability exceeds this (default 0.6; 1 disables).
    pub no_speech_threshold: Option<f32>,
    /// Curated hallucination and repetition filter (default on).
    pub hallucination_filter: Option<bool>,
    /// Utterances waiting for the model before the oldest is dropped (default 3, 1..=10).
    pub max_backlog: Option<u32>,
}

/// VAD settings for a recording: the explicit `vad` object, else defaults. The legacy
/// `vadEnabled: false` ("fixed 5 s windows") maps to 5 s forced splits; silence is still never sent.
fn resolve_vad(vad: Option<VadConfig>, vad_enabled: Option<bool>) -> VadConfig {
    match vad {
        Some(v) => v,
        None if vad_enabled == Some(false) => VadConfig {
            max_utterance_ms: 5_000,
            ..VadConfig::default()
        },
        None => VadConfig::default(),
    }
}

/// The pipeline config of a legacy recording.
fn legacy_pipeline_config(
    device_name: Option<String>,
    vad: VadConfig,
    decode: &WhisperDecodeOptions,
    language: &str,
    model: &str,
) -> PipelineConfig {
    let mut options = serde_json::Map::new();
    if let Some(v) = decode.no_speech_threshold.filter(|v| v.is_finite()) {
        options.insert("noSpeechThreshold".into(), serde_json::json!(v));
    }
    PipelineConfig {
        pipeline_id: LEGACY_PIPELINE_ID.to_string(),
        source: device_name.unwrap_or_default(),
        vad,
        language: language.to_string(),
        route: RouteRequest::default(),
        streaming: None,
        final_engine: Some(EngineSpec {
            engine: EngineKind::Whisper,
            model: model.to_string(),
            threads: decode.n_threads.filter(|&n| n > 0),
            variant: None,
            options: serde_json::Value::Object(options),
        }),
        smart_turn: false,
        hotwords: decode.hotwords.clone().unwrap_or_default(),
        prompt: decode.initial_prompt.clone().unwrap_or_default(),
        hallucination_filter: decode.hallucination_filter.unwrap_or(true),
        latency_budget_ms: 0,
        max_backlog: decode.max_backlog.unwrap_or(3),
        retain_pcm: false,
        capture_only: false,
    }
}

#[derive(Clone)]
pub struct WhisperState {
    /// The loaded model and its reused `WhisperState`, shared with the recording's engine.
    model: SharedModel,
    /// Preset of the model whose weights are in `model` (updated together with it, lock order
    /// model -> this).
    loaded: Arc<Mutex<Option<WhisperModelInfo>>>,
    /// Language used for transcription; written only by a current (non-stale) `ensure_dependencies` run.
    current_language: Arc<Mutex<String>>,
    /// Bumped on each `ensure_dependencies` start and on cancel; stale runs must not mutate state or emit,
    /// and only ever touch their own temp files.
    ensure_deps_epoch: Arc<AtomicU64>,
    /// Cancels the store downloads and copies started before the next `cancel_whisper_download`
    /// (which cancels this token and installs a fresh one).
    cancel: Arc<Mutex<CancelToken>>,
    /// Store entries whose install job a Whisper run cancelled, and when (it created the job,
    /// then went stale or was cancelled). A newer run that had joined that job queues it again
    /// instead of failing; a cancel from the Models tab is not in here and ends the waiting run.
    store_cancels: Arc<Mutex<HashMap<String, Instant>>>,
    /// Held across a model load (build + publish), so a restart during a load waits for it instead of building a
    /// second `WhisperContext` alongside it (2x model RAM). Async so a waiting run does not block a runtime thread.
    load_lock: Arc<tokio::sync::Mutex<()>>,
}

impl WhisperState {
    fn new() -> Self {
        Self {
            model: Arc::new(Mutex::new(None)),
            loaded: Arc::new(Mutex::new(None)),
            current_language: Arc::new(Mutex::new("en".to_string())),
            ensure_deps_epoch: Arc::new(AtomicU64::new(0)),
            cancel: Arc::new(Mutex::new(CancelToken::default())),
            store_cancels: Arc::new(Mutex::new(HashMap::new())),
            load_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    fn token(&self) -> CancelToken {
        self.cancel.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn cancel_all(&self) {
        let mut t = self.cancel.lock().unwrap_or_else(PoisonError::into_inner);
        t.cancel();
        *t = CancelToken::default();
    }

    fn loaded(&self) -> Option<WhisperModelInfo> {
        *self.loaded.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

#[derive(Clone, Serialize)]
struct SpeechStartPayload {
    seq: u64,
    start_ms: u64,
}

#[derive(Clone, Serialize)]
struct UtteranceDroppedPayload {
    seq: u64,
    audio_ms: u64,
    reason: String,
}

#[derive(Clone, Serialize)]
struct CaptureErrorPayload {
    message: String,
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

fn emit_progress<R: Runtime>(app: &AppHandle<R>, file: &str, progress: f64) {
    let _ = app.emit(
        "whisper:download_progress",
        ProgressPayload {
            file: file.to_string(),
            progress,
        },
    );
}

fn ensure_deps_stale(state: &WhisperState, my_epoch: u64) -> bool {
    state.ensure_deps_epoch.load(Ordering::SeqCst) != my_epoch
}

/// Forward the store's `models:progress` for `entry_id` as `whisper:download_progress` (percent, or
/// -1 while the size is unknown).
fn bridge_progress<R: Runtime>(
    app: &AppHandle<R>,
    store: &model_store::Store,
    entry_id: String,
    file: &'static str,
) -> tauri::async_runtime::JoinHandle<()> {
    let mut rx = store.subscribe_progress();
    let app = app.clone();
    tauri::async_runtime::spawn(async move {
        loop {
            match rx.recv().await {
                Ok(e) if e.id == entry_id => {
                    let progress = e
                        .bytes_total
                        .and_then(|t| progress_percent(e.bytes_done, t))
                        .unwrap_or(-1.0);
                    emit_progress(&app, file, progress);
                }
                Ok(_) | Err(RecvError::Lagged(_)) => {}
                Err(RecvError::Closed) => break,
            }
        }
    })
}

/// Error text of a cancelled store job (`model_store`'s).
const STORE_CANCELLED: &str = "cancelled";
/// How often one run queues its model again after another Whisper run cancelled the job it joined.
const MAX_REQUEUES: u32 = 3;
/// A Whisper run's cancel of a store job explains a joined job's `cancelled` only this long.
const STORE_CANCEL_WINDOW: Duration = Duration::from_secs(30);

/// A store install job as [`wait_for_install`] sees it (closures over the store's job, so the
/// waiting logic is testable without a store).
struct InstallJob {
    wait: BoxFuture<'static, Result<PathBuf, String>>,
    cancel: Box<dyn Fn() + Send + Sync>,
    /// This run created the job (vs. joined one already queued or running).
    created: bool,
}

/// How [`wait_for_install`] ended.
#[derive(Debug, PartialEq)]
enum InstallEnd {
    /// Installed (or adopted) already: nothing was downloaded.
    AlreadyInstalled,
    Installed,
    /// This run was cancelled or went stale (or someone cancelled the job in the Models tab).
    Cancelled,
    /// Queueing or the install failed (the error text).
    Failed(String),
}

/// Queue `entry_id` and wait for its install job. A cancel or staleness of this run stops the wait
/// and cancels the job when this run created it (recorded in `store_cancels`); a job that ends
/// `cancelled` while this run still wants the model is queued again when a Whisper run cancelled
/// it (bounded), since this run may have joined it just before that run gave up.
async fn wait_for_install(
    entry_id: &str,
    mut enqueue: impl FnMut() -> Result<Option<InstallJob>, String>,
    cancel: &CancelToken,
    stale: impl Fn() -> bool,
    store_cancels: &Mutex<HashMap<String, Instant>>,
) -> InstallEnd {
    let mut requeues = 0u32;
    let mut waited = false;
    loop {
        let job = match enqueue() {
            Ok(Some(job)) => job,
            Ok(None) if waited => return InstallEnd::Installed,
            Ok(None) => return InstallEnd::AlreadyInstalled,
            Err(e) => return InstallEnd::Failed(e),
        };
        waited = true;
        if job.created {
            // A fresh job: an older cancel of this entry concerns nobody any more.
            store_cancels.lock().unwrap_or_else(PoisonError::into_inner).remove(entry_id);
            tracing::info!(target: LOG, "downloading {entry_id} through the model store");
        }
        let stale_wait = async {
            while !stale() {
                tokio::time::sleep(STALE_POLL).await;
            }
        };
        let result = tokio::select! {
            r = job.wait => r,
            _ = cancel.cancelled() => Err(STORE_CANCELLED.to_string()),
            _ = stale_wait => Err(STORE_CANCELLED.to_string()),
        };
        match result {
            Ok(_) => return InstallEnd::Installed,
            Err(e) if e == STORE_CANCELLED => {
                if cancel.is_cancelled() || stale() {
                    if job.created {
                        store_cancels
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .insert(entry_id.to_string(), Instant::now());
                        (job.cancel)();
                    }
                    return InstallEnd::Cancelled;
                }
                let ours = store_cancels
                    .lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .remove(entry_id)
                    .is_some_and(|at| at.elapsed() <= STORE_CANCEL_WINDOW);
                if !ours || requeues >= MAX_REQUEUES {
                    return InstallEnd::Cancelled;
                }
                requeues += 1;
                tracing::info!(target: LOG, "{entry_id}: the download this run joined was cancelled by an earlier run; queueing it again");
            }
            Err(e) => return InstallEnd::Failed(format!("Model download failed: {e}")),
        }
    }
}

/// Install a catalog model through the model store (pinned, resumable, mirrored; a file already in
/// `whisper/` is adopted instead of downloaded). A cancel or a newer `ensure_dependencies` run stops
/// waiting, and cancels the download when this call started it (one started from the Models tab
/// continues); the partial download is kept for resume. See [`wait_for_install`].
async fn download_through_store<R: Runtime>(
    app: &AppHandle<R>,
    state: &WhisperState,
    my_epoch: u64,
    cancel: &CancelToken,
    info: &WhisperModelInfo,
) -> Result<(), String> {
    let store = model_store::global_store()?;
    let entry_id = model_store::whisper_entry_id(info.id);
    if store.catalog.get(&entry_id).is_none() {
        return Err(format!("Whisper model '{}' is not in the model catalog.", info.id));
    }
    let bridge = bridge_progress(app, &store, entry_id.clone(), info.filename);
    let enqueue = || -> Result<Option<InstallJob>, String> {
        let queued = store.enqueue(std::slice::from_ref(&entry_id))?;
        Ok(queued.into_iter().next().and_then(|q| {
            let created = q.created;
            q.job.map(|job| {
                let waiter = Arc::clone(&job);
                InstallJob {
                    wait: Box::pin(async move { waiter.wait().await }),
                    cancel: Box::new(move || job.cancel.cancel()),
                    created,
                }
            })
        }))
    };
    let end = wait_for_install(
        &entry_id,
        enqueue,
        cancel,
        || ensure_deps_stale(state, my_epoch),
        &state.store_cancels,
    )
    .await;
    bridge.abort();
    match end {
        InstallEnd::AlreadyInstalled => Ok(()),
        InstallEnd::Installed => {
            emit_progress(app, info.filename, 100.0);
            Ok(())
        }
        InstallEnd::Cancelled => Err(DOWNLOAD_CANCELLED.to_string()),
        InstallEnd::Failed(e) => Err(e),
    }
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
    let cancel = state.token();

    let info = get_model_info(&model)?;
    let model_filename = info.filename;

    let app_data_dir = app
        .path()
        .app_data_dir()
        .map_err(|e| e.to_string())?;
    let whisper_dir = app_data_dir.join(WHISPER_FOLDER);
    let model_path = whisper_dir.join(model_filename);

    if !whisper_dir.exists() {
        fs::create_dir_all(&whisper_dir).map_err(|e| e.to_string())?;
    }

    // Reload when nothing is loaded, a different model is loaded, or the file still has to be fetched.
    let need_reload = {
        let model_guard = state.model.lock().map_err(|_| "Failed to lock the model")?;
        model_guard.is_none() || state.loaded().map(|l| l.filename) != Some(model_filename) || !model_path.exists()
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
                let token = cancel.clone();
                let sha = copy_whisper_bin_with_progress(
                    &app,
                    PathBuf::from(trimmed),
                    model_path.clone(),
                    model_filename,
                    move || epoch.load(Ordering::SeqCst) == my_epoch && !token.is_cancelled(),
                )
                .await?;
                register_with_store(info.id, &sha, Origin::Import);
                fetched_model = true;
            }
        }

        if !model_path.exists() {
            if info.bring_your_own {
                return Err(bring_your_own_missing(&info));
            }
            if ensure_deps_stale(&state, my_epoch) {
                return Err(DOWNLOAD_CANCELLED.to_string());
            }
            fetched_model = true;
            download_through_store(&app, &state, my_epoch, &cancel, &info).await?;
            if ensure_deps_stale(&state, my_epoch) {
                return Err(DOWNLOAD_CANCELLED.to_string());
            }
            if !model_path.exists() {
                return Err("The Whisper model was installed, but its file is missing.".to_string());
            }
            // From here on the installed model file is valid and is never deleted, even if this run goes stale.
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
        let path = model_path.clone();
        let ctx = tauri::async_runtime::spawn_blocking(move || whisper_engine::load_context(&path))
            .await
            .map_err(|e| format!("Model load task failed: {}", e))??;
        let previous = {
            let mut model_guard = state.model.lock().map_err(|_| "Failed to lock the model")?;
            // Checked under the lock the recording's engine holds across inference (seconds): a cancel or restart
            // that lands while this run waits for it is seen here, and the stale context is never published.
            if ensure_deps_stale(&state, my_epoch) {
                return Err("Model load cancelled.".to_string());
            }
            *state.loaded.lock().map_err(|_| "Lock failed")? = Some(info);
            model_guard.replace(LoadedModel::new(ctx, info.accepts_prompt))
        };
        // Free the replaced model outside the model lock so transcription is not blocked while it is released.
        drop(previous);
        if !ensure_deps_stale(&state, my_epoch) {
            let _ = app.emit("whisper:load_status", "");
        }
    }

    {
        let mut lang = state.current_language.lock().map_err(|_| "Lock failed")?;
        if ensure_deps_stale(&state, my_epoch) {
            return Err(DOWNLOAD_CANCELLED.to_string());
        }
        // Presets for single-language fine-tunes (Anime-Whisper: Japanese) override the setting.
        *lang = info.language.map_or(language, str::to_string);
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
    let info = get_model_info(&model)?;
    let model_filename = info.filename;
    let app_data_dir = app.path().app_data_dir().map_err(|e| e.to_string())?;
    let whisper_dir = app_data_dir.join(WHISPER_FOLDER);
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
    let sha = copied?;
    register_with_store(info.id, &sha, Origin::Import);
    // The file on disk changed: make the next `ensure_dependencies` reload it instead of reusing the model.
    let mut loaded = state.loaded.lock().map_err(|_| "Lock failed")?;
    if loaded.map(|l| l.filename) == Some(model_filename) {
        *loaded = None;
    }
    Ok(())
}

/// Maps the legacy recording's pipeline events to the `whisper:*` events `whisper.ts` listens to.
struct LegacyWhisperSink<R: Runtime> {
    app: AppHandle<R>,
}

/// A legacy event: name and payload.
#[derive(Debug, PartialEq)]
enum LegacyEvent {
    Transcript(String),
    SpeechStart { seq: u64, start_ms: u64 },
    Dropped { seq: u64, audio_ms: u64, reason: String },
    CaptureError(String),
}

impl LegacyEvent {
    fn from_final(e: &SpeechFinal) -> Option<Self> {
        (!e.text.trim().is_empty()).then(|| LegacyEvent::Transcript(e.text.clone()))
    }

    fn from_status(e: &SpeechStatus) -> Option<Self> {
        (e.state == SpeechState::Error).then(|| {
            LegacyEvent::CaptureError(if e.detail.is_empty() { e.code.clone() } else { e.detail.clone() })
        })
    }

    fn from_utterance(e: &SpeechUtterance, audio_ms: u64) -> Option<Self> {
        match e {
            SpeechUtterance::SpeechStart {
                segment_id, start_ms, ..
            } => Some(LegacyEvent::SpeechStart {
                seq: *segment_id,
                start_ms: *start_ms,
            }),
            // Only real losses were ever reported (a merged utterance's text is in the previous final).
            SpeechUtterance::Dropped {
                segment_id, reason, ..
            } if reason == reasons::BACKLOG || reason == reasons::LATENCY_BUDGET => Some(LegacyEvent::Dropped {
                seq: *segment_id,
                audio_ms,
                reason: reason.clone(),
            }),
            _ => None,
        }
    }

    fn emit<R: Runtime>(self, app: &AppHandle<R>) {
        let _ = match self {
            LegacyEvent::Transcript(text) => app.emit("whisper:segment_transcript", text),
            LegacyEvent::SpeechStart { seq, start_ms } => {
                app.emit("whisper:speech_start", SpeechStartPayload { seq, start_ms })
            }
            LegacyEvent::Dropped { seq, audio_ms, reason } => app.emit(
                "whisper:utterance_dropped",
                UtteranceDroppedPayload { seq, audio_ms, reason },
            ),
            LegacyEvent::CaptureError(message) => {
                app.emit("whisper:capture_error", CaptureErrorPayload { message })
            }
        };
    }
}

impl<R: Runtime> EventSink for LegacyWhisperSink<R> {
    fn partial(&self, _event: &SpeechPartial) {}

    fn final_transcript(&self, event: &SpeechFinal) {
        if let Some(e) = LegacyEvent::from_final(event) {
            e.emit(&self.app);
        }
    }

    fn status(&self, event: &SpeechStatus) {
        if let Some(e) = LegacyEvent::from_status(event) {
            e.emit(&self.app);
        }
    }

    fn utterance(&self, event: &SpeechUtterance, audio_ms: u64) {
        if let Some(e) = LegacyEvent::from_utterance(event, audio_ms) {
            e.emit(&self.app);
        }
    }

    fn turn_end(&self, _event: &SpeechTurnEnd) {}
}

// The argument list is the IPC contract with `whisper.ts` (one named field per setting). Legacy RMS
// fields (`silenceThresholdDb`, `silenceDurationMs`, `minChunkDurationMs`) are still accepted by the
// frontend call and ignored: Silero VAD (`vad`) replaced the RMS gate. `deviceName` takes any
// capture-hub source id (`null`/"default", an input name, or "loopback:<output>"). `captureLocal`
// must be absent or true: the JSON-PCM feed (`captureLocal: false` + `feed_audio_chunk`) was
// removed with the capture hub (no PCM over IPC).
#[command]
async fn start_recording<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, WhisperState>,
    device_name: Option<String>,
    vad_enabled: Option<bool>,
    capture_local: Option<bool>,
    vad: Option<VadConfig>,
    decode: Option<WhisperDecodeOptions>,
) -> Result<(), String> {
    if capture_local == Some(false) {
        return Err("Whisper no longer accepts audio over IPC (captureLocal: false); it captures through the audio hub.".to_string());
    }
    let loaded_model = {
        let model = state.model.lock().unwrap_or_else(PoisonError::into_inner);
        model.as_ref().and(state.loaded())
    };
    let Some(info) = loaded_model else {
        return Err("The Whisper model is not loaded yet (ensure_dependencies must run first).".to_string());
    };
    let decode = decode.unwrap_or_default();
    let language = state
        .current_language
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let label = if info.bring_your_own {
        info.id.to_string()
    } else {
        model_store::whisper_entry_id(info.id)
    };
    let config = legacy_pipeline_config(device_name, resolve_vad(vad, vad_enabled), &decode, &language, &label);
    let engine_info = whisper_engine::engine_info(&label, info.accepts_prompt, info.language);
    let model = Arc::clone(&state.model);
    let threads = decode.n_threads;
    let no_speech = decode.no_speech_threshold;
    let factory: EngineFactory = Arc::new(move |_spec, _ctx| {
        Ok(Box::new(WhisperEngine::shared(
            Arc::clone(&model),
            engine_info.clone(),
            threads,
            no_speech,
            info.language,
        )) as Box<dyn AsrEngine>)
    });
    let sink: Arc<dyn EventSink> = Arc::new(LegacyWhisperSink { app });
    let started = pipeline::start_with(config, SessionDeps::production(sink).with_engines(factory)).await?;
    tracing::info!(
        target: LOG,
        "recording started through pipeline {} (session {}, model {label})",
        started.pipeline_id,
        started.epoch
    );
    Ok(())
}

#[command]
fn cancel_whisper_download(state: State<'_, WhisperState>) -> Result<(), String> {
    // Makes every in-flight `ensure_dependencies` run stale and stops what it waits for; downloads this
    // plugin started are cancelled (their partial files are kept for resume).
    state.ensure_deps_epoch.fetch_add(1, Ordering::SeqCst);
    state.cancel_all();
    Ok(())
}

#[command]
async fn stop_recording<R: Runtime>(_app: AppHandle<R>, _state: State<'_, WhisperState>) -> Result<String, String> {
    // After this returns nothing of the ended recording is emitted and the microphone was released
    // (bounded wait); an in-flight decode is aborted.
    if pipeline::stop_pipeline_key(LEGACY_PIPELINE_ID).await {
        tracing::info!(target: LOG, "recording stopped");
        Ok(String::new())
    } else {
        Err("Not recording".to_string())
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("whisper")
        .setup(|app, _api| {
            app.manage(WhisperState::new());
            if let Ok(dir) = app.path().app_data_dir() {
                sweep_stale_partials(&dir.join(WHISPER_FOLDER));
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            ensure_dependencies,
            import_whisper_model,
            cancel_whisper_download,
            start_recording,
            stop_recording
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::speech::events::FinalPass;

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
    fn local_model_copy_hashes_reports_progress_and_can_be_cancelled() {
        let dir = temp_dir("copy");
        let src = dir.join("src.bin");
        let data: Vec<u8> = (0..(COPY_CHUNK_BYTES * 2 + 1234)).map(|i| (i % 251) as u8).collect();
        fs::write(&src, &data).unwrap();

        let dest = dir.join("ggml-base.en.bin");
        let mut calls = Vec::new();
        let sha = copy_whisper_bin_to_model_path(&src, &dest, &mut |copied, total| {
            calls.push((copied, total));
            true
        })
        .unwrap();
        assert_eq!(fs::read(&dest).unwrap(), data);
        assert_eq!(sha, hex::encode(Sha256::digest(&data)), "the sha256 registers imports with the store");
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

    #[test]
    fn catalog_resolves_every_id_and_rejects_unknown_ones() {
        for (id, file) in CATALOG {
            let info = get_model_info(id).unwrap();
            assert_eq!((info.id, info.filename), (*id, *file));
            assert!(file.starts_with("ggml-") && file.ends_with(".bin"));
            assert!(!info.bring_your_own && info.accepts_prompt);
        }
        // The previously missing ids.
        for id in ["small-q8_0", "base.en-q8_0", "tiny-q5_1", "large-v3-turbo", "large-v3-turbo-q8_0", "medium-q5_0"] {
            assert!(get_model_info(id).is_ok(), "{id}");
        }
        // Legacy ids that 404ed upstream map to the real quantization.
        assert_eq!(get_model_info("medium.en-q5_1").unwrap().filename, "ggml-medium.en-q5_0.bin");
        assert_eq!(get_model_info(" medium-q5_1 ").unwrap().id, "medium-q5_0");
        // Unknown ids are an error, never a silent base.en.
        let err = get_model_info("huge-v9").unwrap_err();
        assert!(err.contains("huge-v9"), "{err}");
        assert!(get_model_info("").is_err());
    }

    /// Every downloadable model is a store entry, so downloads go through the store (pinned,
    /// resumable) and land in the folder this plugin loads from.
    #[test]
    fn every_catalog_model_downloads_through_the_store() {
        let catalog = model_store::Catalog::embedded().unwrap();
        for (id, file) in CATALOG {
            let entry = catalog
                .get(&model_store::whisper_entry_id(id))
                .unwrap_or_else(|| panic!("whisper/{id} is missing from the model catalog"));
            assert_eq!(entry.family, WHISPER_FOLDER);
            assert_eq!(entry.files.len(), 1);
            assert_eq!(entry.files[0].path, *file);
        }
    }

    #[test]
    fn anime_whisper_is_a_japanese_bring_your_own_preset() {
        let info = get_model_info(ANIME_WHISPER_ID).unwrap();
        assert!(info.bring_your_own);
        assert_eq!(info.language, Some("ja"));
        assert!(!info.accepts_prompt);
        assert!(bring_your_own_missing(&info).contains("convert-h5-to-ggml.py"));
    }

    #[test]
    fn legacy_vad_toggle_maps_to_bounded_windows() {
        assert_eq!(resolve_vad(None, None), VadConfig::default());
        assert_eq!(resolve_vad(None, Some(true)), VadConfig::default());
        assert_eq!(resolve_vad(None, Some(false)).max_utterance_ms, 5_000);
        let explicit = VadConfig {
            threshold: 0.7,
            ..VadConfig::default()
        };
        assert_eq!(resolve_vad(Some(explicit.clone()), Some(false)), explicit);
    }

    #[test]
    fn decode_options_map_onto_the_pipeline_config() {
        let json: WhisperDecodeOptions = serde_json::from_str(r#"{"n_threads":2,"hotwords":["a"]}"#).unwrap();
        assert_eq!(json.n_threads, Some(2));
        let decode = WhisperDecodeOptions {
            n_threads: Some(3),
            initial_prompt: Some("Chat in VRChat.".into()),
            hotwords: Some(vec!["Sigil".into()]),
            no_speech_threshold: Some(0.7),
            hallucination_filter: Some(false),
            max_backlog: Some(5),
        };
        let c = legacy_pipeline_config(Some("Mic".into()), VadConfig::default(), &decode, "ja", "whisper/small-q5_1");
        assert_eq!(c.pipeline_id, LEGACY_PIPELINE_ID);
        assert!(c.pipeline_id.starts_with(INTERNAL_PREFIX), "not reachable from start_pipeline");
        assert_eq!((c.source.as_str(), c.language.as_str()), ("Mic", "ja"));
        assert_eq!((c.prompt.as_str(), c.hotwords.clone()), ("Chat in VRChat.", vec!["Sigil".to_string()]));
        assert_eq!((c.hallucination_filter, c.max_backlog, c.smart_turn, c.retain_pcm), (false, 5, false, false));
        let spec = c.final_engine.unwrap();
        assert_eq!((spec.engine, spec.model.as_str(), spec.threads), (EngineKind::Whisper, "whisper/small-q5_1", Some(3)));
        assert_eq!(spec.options["noSpeechThreshold"].as_f64().map(|v| (v * 10.0).round()), Some(7.0));
        assert!(c.streaming.is_none());
        let defaults = legacy_pipeline_config(None, VadConfig::default(), &WhisperDecodeOptions::default(), "auto", "m");
        assert_eq!((defaults.source.as_str(), defaults.max_backlog, defaults.hallucination_filter), ("", 3, true));
        assert!(defaults.final_engine.unwrap().options.as_object().unwrap().is_empty());
    }

    #[test]
    fn pipeline_events_map_to_the_legacy_whisper_events() {
        let fin = SpeechFinal {
            pipeline_id: LEGACY_PIPELINE_ID.into(),
            epoch: 1,
            segment_id: 3,
            text: "hello".into(),
            pass: FinalPass::First,
            ..SpeechFinal::default()
        };
        assert_eq!(LegacyEvent::from_final(&fin), Some(LegacyEvent::Transcript("hello".into())));
        assert_eq!(LegacyEvent::from_final(&SpeechFinal { text: " ".into(), ..fin }), None);

        let start = SpeechUtterance::SpeechStart {
            pipeline_id: "p".into(),
            epoch: 1,
            segment_id: 4,
            start_ms: 20,
        };
        assert_eq!(
            LegacyEvent::from_utterance(&start, 0),
            Some(LegacyEvent::SpeechStart { seq: 4, start_ms: 20 })
        );
        let dropped = |reason: &str| SpeechUtterance::Dropped {
            pipeline_id: "p".into(),
            epoch: 1,
            segment_id: 5,
            reason: reason.into(),
        };
        assert_eq!(
            LegacyEvent::from_utterance(&dropped("backlog"), 3100),
            Some(LegacyEvent::Dropped { seq: 5, audio_ms: 3100, reason: "backlog".into() })
        );
        assert!(LegacyEvent::from_utterance(&dropped("latency_budget"), 1).is_some());
        for quiet in ["merged", "no_speech", "muted", "engine_error"] {
            assert_eq!(LegacyEvent::from_utterance(&dropped(quiet), 1), None, "{quiet}");
        }
        let closed = SpeechUtterance::Utterance {
            pipeline_id: "p".into(),
            epoch: 1,
            segment_id: 5,
            start_ms: 0,
            end_ms: 900,
            samples: 14_400,
            forced: false,
        };
        assert_eq!(LegacyEvent::from_utterance(&closed, 900), None);

        let error = SpeechStatus {
            state: SpeechState::Error,
            code: "capture_error".into(),
            detail: "speech detection failed: x".into(),
            ..SpeechStatus::default()
        };
        assert_eq!(
            LegacyEvent::from_status(&error),
            Some(LegacyEvent::CaptureError("speech detection failed: x".into()))
        );
        let listening = SpeechStatus {
            state: SpeechState::Listening,
            ..SpeechStatus::default()
        };
        assert_eq!(LegacyEvent::from_status(&listening), None);
    }

    /// A fake store queue: one job per entry until it finishes, like `model_store`'s queue (which
    /// forgets a job before it publishes the job's result).
    #[derive(Default)]
    struct FakeQueue {
        current: Mutex<Option<Arc<FakeJob>>>,
        created: AtomicU64,
        enqueued: AtomicU64,
    }

    struct FakeJob {
        done: tokio::sync::watch::Sender<Option<Result<PathBuf, String>>>,
        cancelled: std::sync::atomic::AtomicBool,
    }

    impl FakeQueue {
        fn enqueue(&self) -> Result<Option<InstallJob>, String> {
            self.enqueued.fetch_add(1, Ordering::SeqCst);
            let (job, created) = {
                let mut current = self.current.lock().unwrap();
                match current.as_ref() {
                    Some(job) => (Arc::clone(job), false),
                    None => {
                        let job = Arc::new(FakeJob {
                            done: tokio::sync::watch::channel(None).0,
                            cancelled: std::sync::atomic::AtomicBool::new(false),
                        });
                        *current = Some(Arc::clone(&job));
                        self.created.fetch_add(1, Ordering::SeqCst);
                        (job, true)
                    }
                }
            };
            let waiter = Arc::clone(&job);
            Ok(Some(InstallJob {
                wait: Box::pin(async move {
                    let mut rx = waiter.done.subscribe();
                    loop {
                        if let Some(r) = rx.borrow_and_update().clone() {
                            return r;
                        }
                        if rx.changed().await.is_err() {
                            return Err("gone".into());
                        }
                    }
                }),
                cancel: Box::new(move || job.cancelled.store(true, Ordering::SeqCst)),
                created,
            }))
        }

        fn job(&self) -> Option<Arc<FakeJob>> {
            self.current.lock().unwrap().clone()
        }

        /// The store finishes the current job.
        fn finish(&self, result: Result<PathBuf, String>) {
            let job = self.current.lock().unwrap().take().expect("a job");
            job.done.send_replace(Some(result));
        }
    }

    async fn until(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(10);
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    }

    type Cancels = Arc<Mutex<HashMap<String, Instant>>>;

    fn spawn_run(
        queue: &Arc<FakeQueue>,
        epoch: &Arc<AtomicU64>,
        cancels: &Cancels,
        mine: u64,
    ) -> tokio::task::JoinHandle<InstallEnd> {
        let (queue, epoch, cancels) = (Arc::clone(queue), Arc::clone(epoch), Arc::clone(cancels));
        tokio::spawn(async move {
            let token = CancelToken::default();
            wait_for_install(
                "whisper/base",
                || queue.enqueue(),
                &token,
                || epoch.load(Ordering::SeqCst) != mine,
                &cancels,
            )
            .await
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_run_that_joined_a_job_an_older_run_cancelled_queues_it_again() {
        let queue = Arc::new(FakeQueue::default());
        let epoch = Arc::new(AtomicU64::new(1));
        let cancels: Cancels = Arc::default();
        let run1 = spawn_run(&queue, &epoch, &cancels, 1);
        until("the first job", || queue.created.load(Ordering::SeqCst) == 1).await;
        let first = queue.job().unwrap();
        // A restart: the new run makes the first stale and joins its job (which the store has not
        // finished yet, cancelled or not).
        epoch.store(2, Ordering::SeqCst);
        let run2 = spawn_run(&queue, &epoch, &cancels, 2);
        until("the join", || queue.enqueued.load(Ordering::SeqCst) == 2).await;
        until("the stale run's cancel", || first.cancelled.load(Ordering::SeqCst)).await;
        assert_eq!(queue.created.load(Ordering::SeqCst), 1, "the second run joined");
        queue.finish(Err(STORE_CANCELLED.into()));
        assert_eq!(run1.await.unwrap(), InstallEnd::Cancelled);
        // The newer run still wants the model: it queues it again instead of failing.
        until("the new job", || queue.created.load(Ordering::SeqCst) == 2).await;
        queue.finish(Ok(PathBuf::from("whisper/base")));
        assert_eq!(run2.await.unwrap(), InstallEnd::Installed);
        assert!(cancels.lock().unwrap().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_cancel_from_the_models_tab_ends_the_waiting_run() {
        let queue = Arc::new(FakeQueue::default());
        let epoch = Arc::new(AtomicU64::new(1));
        let cancels: Cancels = Arc::default();
        let run = spawn_run(&queue, &epoch, &cancels, 1);
        until("the job", || queue.created.load(Ordering::SeqCst) == 1).await;
        queue.finish(Err(STORE_CANCELLED.into()));
        assert_eq!(run.await.unwrap(), InstallEnd::Cancelled);
        assert_eq!(queue.created.load(Ordering::SeqCst), 1, "never queued again");

        // A failure is reported as such; an installed model needs no job.
        let run = spawn_run(&queue, &epoch, &cancels, 1);
        until("the job", || queue.created.load(Ordering::SeqCst) == 2).await;
        queue.finish(Err("sha256 mismatch".into()));
        assert_eq!(
            run.await.unwrap(),
            InstallEnd::Failed("Model download failed: sha256 mismatch".into())
        );
        let token = CancelToken::default();
        let end = wait_for_install("whisper/base", || Ok(None), &token, || false, &cancels).await;
        assert_eq!(end, InstallEnd::AlreadyInstalled);
    }
}
