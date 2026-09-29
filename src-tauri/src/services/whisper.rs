//! `whisper` plugin: local Whisper (whisper-rs / whisper.cpp, CPU) behind a Silero VAD gate.
//!
//! Pipeline (P0-C): microphone → `audio::speech_capture` (rubato to 16 kHz, Silero v6.2 segmenter)
//! → VAD-closed utterances → one ordered worker per recording (bounded, drop-oldest backlog) →
//! whisper.cpp with one reused `WhisperState` → no-speech drop + hallucination/repeat filter →
//! `whisper:segment_transcript` (one final per utterance). Silence never reaches Whisper.
//!
//! Events (all emitted under the recording's stop lock with its epoch still current):
//! - `whisper:segment_transcript` (string): one final transcript per utterance (unchanged contract).
//! - `whisper:speech_start` (`{ seq, start_ms }`): speech detected (after `min_speech_ms`; a short
//!   voiced tail after an 18 s forced split gets it right before its transcript).
//! - `whisper:utterance_dropped` (`{ seq, audio_ms, reason }`): `reason` is `"backlog"` when the model
//!   runs slower than real time and the oldest waiting utterance was discarded.
//! - `whisper:capture_error` (`{ message }`): the input device failed; the recording is over.
//! - `whisper:download_progress`, `whisper:load_status`, `whisper:download_complete`,
//!   `whisper:download_dismiss`: model fetch / load (unchanged).

use crate::services::audio::speech_capture::{start_speech_capture, SpeechCapture, SpeechCaptureEvent};
use crate::services::http_client::{describe_failed_response, http_download_get};
use crate::services::speech_core::hallucination::clean_transcript;
use crate::services::speech_core::utterance::{samples_to_ms, BoundedQueue, SpeechFrontEnd};
use crate::services::speech_core::{SegmenterEvent, VadConfig, SAMPLE_RATE};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::ffi::{c_int, c_void};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime, State,
};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Log target of the Whisper pipeline (file log via `diagnostics`).
const LOG: &str = "sigil::stt::whisper";

/// Download base of the official whisper.cpp ggml conversions.
const HF_WHISPER_BASE: &str = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/";

/// Model id → ggml file on `HF_WHISPER_BASE` (the ids the inspector persists). Checked against the
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

impl WhisperModelInfo {
    /// Download URL, `None` for bring-your-own presets.
    pub fn download_url(&self) -> Option<String> {
        (!self.bring_your_own).then(|| huggingface_model_url(&format!("{HF_WHISPER_BASE}{}", self.filename)))
    }
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
fn bring_your_own_missing(info: &WhisperModelInfo) -> String {
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

/// Decoder options of `start_recording` (`decode`); every field optional, snake_case like
/// `VadConfig`. TS mirror: `RustWhisperDecodeOptions` (INTEGRATE adds it to `src/types.ts`).
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

/// Longest prompt kept, in bytes (whisper.cpp keeps at most the last 223 tokens anyway).
const MAX_PROMPT_BYTES: usize = 1024;
/// Prompt tokens fed to the decoder (`n_text_ctx / 2 - 1`).
const MAX_PROMPT_TOKENS: usize = 223;

/// `WhisperDecodeOptions` resolved to concrete values.
#[derive(Clone, Debug, PartialEq)]
struct DecodeSettings {
    n_threads: c_int,
    prompt: String,
    no_speech_threshold: f32,
    hallucination_filter: bool,
    max_backlog: usize,
}

/// Default inference threads: half the logical CPUs (about the physical cores), 1..=4.
fn default_whisper_threads() -> u32 {
    let logical = thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4);
    (logical / 2).clamp(1, 4)
}

impl DecodeSettings {
    fn resolve(opts: &WhisperDecodeOptions) -> Self {
        let mut prompt = opts.initial_prompt.as_deref().unwrap_or("").trim().to_string();
        let hotwords: Vec<&str> = opts
            .hotwords
            .iter()
            .flatten()
            .map(|w| w.trim())
            .filter(|w| !w.is_empty())
            .collect();
        if !hotwords.is_empty() {
            if !prompt.is_empty() {
                prompt.push(' ');
            }
            prompt.push_str(&hotwords.join(", "));
            prompt.push('.');
        }
        // NUL would abort tokenization; cap on a char boundary.
        prompt.retain(|c| c != '\0');
        if prompt.len() > MAX_PROMPT_BYTES {
            let mut cut = MAX_PROMPT_BYTES;
            while !prompt.is_char_boundary(cut) {
                cut -= 1;
            }
            prompt.truncate(cut);
        }
        let threads = opts.n_threads.filter(|&n| n > 0).unwrap_or_else(default_whisper_threads);
        let available = thread::available_parallelism().map(|n| n.get() as u32).unwrap_or(4);
        Self {
            n_threads: threads.clamp(1, available.max(1)).min(16) as c_int,
            prompt,
            no_speech_threshold: opts
                .no_speech_threshold
                .filter(|v| v.is_finite())
                .map_or(0.6, |v| v.clamp(0.05, 1.0)),
            hallucination_filter: opts.hallucination_filter.unwrap_or(true),
            max_backlog: opts.max_backlog.map_or(3, |n| n.clamp(1, 10) as usize),
        }
    }
}

/// Whisper's per-pass token cap for an utterance of `samples` (16 kHz): generous for fast speech
/// (about 12 tokens/s + 32), far below a runaway repetition loop (224).
fn max_tokens_for(samples: usize) -> c_int {
    let secs = samples as f32 / SAMPLE_RATE as f32;
    ((secs * 12.0).ceil() as c_int + 32).clamp(48, 220)
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

/// The loaded model plus the one `WhisperState` reused for every utterance (created lazily; both
/// are replaced together on a model switch, so the old model's memory is freed at once).
struct LoadedModel {
    ctx: WhisperContext,
    state: Option<whisper_rs::WhisperState>,
    accepts_prompt: bool,
    /// `(prompt text, its tokens)` for this model's vocabulary.
    prompt_tokens: Option<(String, Vec<c_int>)>,
}

impl LoadedModel {
    fn new(ctx: WhisperContext, info: &WhisperModelInfo) -> Self {
        Self {
            ctx,
            state: None,
            accepts_prompt: info.accepts_prompt,
            prompt_tokens: None,
        }
    }
}

/// One VAD-closed utterance waiting for the model.
struct WhisperJob {
    seq: u64,
    samples: Vec<f32>,
}

/// Where a recording's audio comes from.
enum AudioSource {
    /// Microphone through `audio::speech_capture`.
    Capture(SpeechCapture),
    /// 16 kHz mono pushed by `feed_audio_chunk` (segmented synchronously).
    Fed(Arc<Mutex<SpeechFrontEnd>>),
}

/// A running recording.
struct RecordingSession {
    epoch: u64,
    source: AudioSource,
    queue: Arc<BoundedQueue<WhisperJob>>,
}

impl RecordingSession {
    /// Discard waiting utterances, wake the worker and release the device (bounded wait).
    async fn shutdown(self) {
        self.queue.close();
        if let AudioSource::Capture(capture) = self.source {
            capture.stop().await;
        }
    }
}

#[derive(Clone)]
pub struct WhisperState {
    /// The stop lock and the running recording. Starts and stops bump `recording_epoch` while holding
    /// it; every event of a recording is emitted while holding it with that recording's epoch current,
    /// so once a stop (or restart) returns nothing of the ended recording is emitted.
    session: Arc<Mutex<Option<RecordingSession>>>,
    recording_epoch: Arc<AtomicU64>,
    ctx: Arc<Mutex<Option<LoadedModel>>>,
    /// Model file whose weights are in `ctx` (updated together with `ctx`, lock order ctx -> this).
    loaded_model_file: Arc<Mutex<Option<String>>>,
    /// Language used for transcription; written only by a current (non-stale) `ensure_dependencies` run.
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
            session: Arc::new(Mutex::new(None)),
            recording_epoch: Arc::new(AtomicU64::new(0)),
            ctx: Arc::new(Mutex::new(None)),
            loaded_model_file: Arc::new(Mutex::new(None)),
            current_language: Arc::new(Mutex::new("en".to_string())),
            ensure_deps_epoch: Arc::new(AtomicU64::new(0)),
            load_lock: Arc::new(tokio::sync::Mutex::new(())),
        }
    }

    fn lock_session(&self) -> MutexGuard<'_, Option<RecordingSession>> {
        self.session.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn epoch_is(&self, epoch: u64) -> bool {
        self.recording_epoch.load(Ordering::SeqCst) == epoch
    }

    /// Emit `event` only while recording `epoch` is current (under the stop lock).
    fn emit_for<R: Runtime, S: Serialize + Clone>(&self, app: &AppHandle<R>, epoch: u64, event: &str, payload: S) {
        let _guard = self.lock_session();
        if self.epoch_is(epoch) {
            let _ = app.emit(event, payload);
        }
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
    reason: &'static str,
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

    let info = get_model_info(&model)?;
    let model_filename = info.filename;

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
            let Some(model_url) = info.download_url() else {
                return Err(bring_your_own_missing(&info));
            };
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
            // Checked under the lock the transcription worker holds across inference (seconds): a cancel or restart that
            // lands while this run waits for it is seen here, and the stale context is never published.
            if ensure_deps_stale(&state, my_epoch) {
                return Err("Model load cancelled.".to_string());
            }
            let mut loaded = state.loaded_model_file.lock().map_err(|_| "Lock failed")?;
            *loaded = Some(model_filename.to_string());
            ctx_guard.replace(LoadedModel::new(ctx, &info))
        };
        // Free the replaced model outside the ctx lock so transcription is not blocked while it is released.
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
    let model_filename = get_model_info(&model)?.filename;
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

/// Abort flag handed to whisper.cpp: inference stops as soon as the recording's epoch moves on
/// (stop / restart), so a stop never waits for a long decode and a model reload is not held up.
struct AbortToken {
    epoch: Arc<AtomicU64>,
    mine: u64,
}

impl AbortToken {
    fn aborted(&self) -> bool {
        self.epoch.load(Ordering::Relaxed) != self.mine
    }
}

/// `ggml_abort_callback`: `user_data` is the worker's `AbortToken`.
unsafe extern "C" fn abort_trampoline(user_data: *mut c_void) -> bool {
    // SAFETY: set only by `transcribe_utterance`, which passes a pointer to an `AbortToken` that
    // outlives the `full()` call that invokes this callback.
    let token = unsafe { &*(user_data as *const AbortToken) };
    token.aborted()
}

/// Result of one utterance.
struct Transcript {
    /// Final text, `None` when nothing survived the no-speech / hallucination filters.
    text: Option<String>,
    /// Language Whisper decoded (configured or detected).
    language: Option<&'static str>,
    /// No-speech probability of the utterance ([`sot_no_speech_prob`]; 0 when unavailable).
    no_speech_prob: f32,
}

/// OpenAI's no-speech probability: P(`<|nospeech|>`) at the `<|startoftranscript|>` position, from one
/// extra decoder step over the encoder output `full()` left in `wstate` (a single 30 s window for a
/// VAD utterance).
///
/// whisper.cpp's own per-segment value cannot be used: it reads row 0 of the logits buffer, but a
/// multi-token prompt decode (initial prompt, or the language/task tokens of every multilingual
/// model) writes only the last row, so row 0 holds stale logits of an earlier decode. Measured on
/// base.en: this probe gives 0.94 on silence and 0.0006 on speech with or without a prompt, while
/// whisper.cpp's value simply repeated the previous utterance's number.
fn sot_no_speech_prob(ctx: &WhisperContext, wstate: &mut whisper_rs::WhisperState, threads: c_int) -> Option<f32> {
    wstate.decode(&[ctx.token_sot()], 0, threads.max(1) as usize).ok()?;
    let logits = wstate.get_logits().ok()?;
    let nosp = usize::try_from(ctx.token_nosp()).ok()?;
    let target = *logits.get(nosp)?;
    let max = logits.iter().copied().filter(|v| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    if !target.is_finite() || !max.is_finite() {
        return None;
    }
    let sum: f64 = logits.iter().filter(|v| v.is_finite()).map(|&v| f64::from(v - max).exp()).sum();
    (sum > 0.0).then(|| (f64::from(target - max).exp() / sum) as f32)
}

/// Whisper language code for `configured` (`auto`, empty or unknown codes: detect).
fn whisper_language(configured: &str) -> Option<&'static str> {
    let code = configured.trim().split(['-', '_']).next().unwrap_or("").to_ascii_lowercase();
    if code.is_empty() || code == "auto" {
        return None;
    }
    whisper_rs::get_lang_id(&code).and_then(whisper_rs::get_lang_str)
}

/// Transcribe one 16 kHz mono utterance on the loaded model, reusing its `WhisperState`.
fn transcribe_utterance(
    state: &WhisperState,
    settings: &DecodeSettings,
    samples: &[f32],
    abort: &AbortToken,
) -> Result<Transcript, String> {
    if samples.len() < SAMPLE_RATE as usize / 5 {
        // whisper.cpp needs >= 100 ms; the VAD never closes anything this short.
        return Ok(Transcript {
            text: None,
            language: None,
            no_speech_prob: 1.0,
        });
    }
    let configured = state
        .current_language
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .clone();
    let language = whisper_language(&configured);

    let mut guard = state.ctx.lock().unwrap_or_else(PoisonError::into_inner);
    let loaded = guard.as_mut().ok_or("Whisper model is not loaded")?;
    let LoadedModel {
        ctx,
        state: wstate_slot,
        accepts_prompt,
        prompt_tokens,
    } = loaded;

    // Prompt tokens for this model's vocabulary, tokenized once per (model, prompt).
    let use_prompt = *accepts_prompt && !settings.prompt.is_empty();
    if use_prompt && prompt_tokens.as_ref().is_none_or(|(text, _)| *text != settings.prompt) {
        // Every token covers at least one byte, so `len + 8` can never overflow whisper-rs' buffer.
        match ctx.tokenize(&settings.prompt, settings.prompt.len() + 8) {
            Ok(mut tokens) => {
                if tokens.len() > MAX_PROMPT_TOKENS {
                    tokens.drain(..tokens.len() - MAX_PROMPT_TOKENS);
                }
                *prompt_tokens = Some((settings.prompt.clone(), tokens));
            }
            Err(e) => {
                tracing::warn!(target: LOG, "initial prompt ignored (tokenize failed: {e})");
                *prompt_tokens = Some((settings.prompt.clone(), Vec::new()));
            }
        }
    }

    if wstate_slot.is_none() {
        *wstate_slot = Some(ctx.create_state().map_err(|e| format!("Failed to create Whisper state: {e}"))?);
    }
    let wstate = wstate_slot.as_mut().expect("state created above");

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_n_threads(settings.n_threads);
    params.set_translate(false);
    params.set_language(language);
    // Each utterance stands alone: carrying the previous text as context propagates hallucinations.
    params.set_no_context(true);
    // One decode window per utterance (utterances are <= 18 s by default): whisper.cpp would otherwise
    // re-encode a short tail after the last timestamp, which is where "you" / "Thank you." hallucinations
    // come from, and the no-speech probe below must see the utterance's own encoder output.
    params.set_single_segment(true);
    params.set_suppress_nst(true);
    params.set_suppress_blank(true);
    // Disables whisper.cpp's internal no-speech skip, which acts on its stale value (see
    // `sot_no_speech_prob`) and could discard real speech after a noisy utterance; the probe below
    // replaces it. Temperature fallback still runs on low average log-probability.
    params.set_no_speech_thold(1.0);
    params.set_max_tokens(max_tokens_for(samples.len()));
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    if use_prompt {
        if let Some((_, tokens)) = prompt_tokens.as_ref().filter(|(_, t)| !t.is_empty()) {
            params.set_tokens(tokens);
        }
    }
    // SAFETY: `abort` outlives `full()` below, the only call that uses the callback.
    unsafe {
        params.set_abort_callback(Some(abort_trampoline));
        params.set_abort_callback_user_data(abort as *const AbortToken as *mut c_void);
    }

    if let Err(e) = wstate.full(params, samples) {
        // Never reuse a state whose decode failed or was aborted midway.
        *wstate_slot = None;
        return Err(if abort.aborted() {
            ABORTED.to_string()
        } else {
            format!("Whisper inference failed: {e}")
        });
    }

    // Concatenate raw bytes: a multi-byte character can be split across segments, and segment text
    // already carries its own leading space where the script uses spaces (none for CJK).
    let mut bytes = Vec::new();
    for segment in wstate.as_iter() {
        if let Ok(b) = segment.to_bytes() {
            bytes.extend_from_slice(b);
        }
    }
    let detected = whisper_rs::get_lang_str(wstate.full_lang_id_from_state());
    // After the segments are read: this extra decode reuses the state's decoder buffers. It sees the
    // last encoded window only, so it is meaningful for utterances that fit one 30 s window (all of
    // them at the default 18 s VAD limit); longer ones are never dropped on it.
    let no_speech_prob = if samples.len() <= WHISPER_WINDOW_SAMPLES {
        sot_no_speech_prob(ctx, wstate, settings.n_threads).unwrap_or(0.0)
    } else {
        0.0
    };
    drop(guard);
    if no_speech_prob > settings.no_speech_threshold {
        // The utterance as a whole is judged non-speech (breath, keyboard, music the VAD let through).
        bytes.clear();
    }

    let raw = String::from_utf8_lossy(&bytes);
    let language = detected.or(language);
    let text = if settings.hallucination_filter {
        clean_transcript(&raw, language, no_speech_prob)
    } else {
        Some(raw.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|t| !t.is_empty())
    };
    Ok(Transcript {
        text,
        language,
        no_speech_prob,
    })
}

const ABORTED: &str = "aborted";

/// One whisper.cpp decode window (30 s at 16 kHz).
const WHISPER_WINDOW_SAMPLES: usize = 30 * SAMPLE_RATE as usize;

/// The ordered worker of one recording: transcribes utterances strictly in order, one at a time.
fn run_whisper_worker<R: Runtime>(
    app: AppHandle<R>,
    state: WhisperState,
    epoch: u64,
    queue: Arc<BoundedQueue<WhisperJob>>,
    settings: DecodeSettings,
) {
    let abort = AbortToken {
        epoch: Arc::clone(&state.recording_epoch),
        mine: epoch,
    };
    let mut last_error_log: Option<Instant> = None;
    while let Some(job) = queue.pop() {
        if abort.aborted() {
            break;
        }
        let audio_ms = samples_to_ms(job.samples.len() as u64);
        let started = Instant::now();
        tracing::info!(target: LOG, "transcribe utterance seq={} audio_ms={audio_ms} backlog={}", job.seq, queue.len());
        match transcribe_utterance(&state, &settings, &job.samples, &abort) {
            Ok(t) => {
                let took = started.elapsed().as_millis();
                match t.text {
                    Some(text) => {
                        tracing::info!(
                            target: LOG,
                            "utterance seq={} done in {took} ms (lang={})",
                            job.seq,
                            t.language.unwrap_or("?")
                        );
                        // One final per utterance. `whisper.ts` maps it to STT `onFinal`.
                        state.emit_for(&app, epoch, "whisper:segment_transcript", text);
                    }
                    None => tracing::info!(
                        target: LOG,
                        "utterance seq={} produced no text in {took} ms (no-speech {:.2}, or filtered)",
                        job.seq,
                        t.no_speech_prob
                    ),
                }
            }
            Err(e) if e == ABORTED => break,
            Err(e) => {
                // Rate-limit: a missing model fails every utterance the same way.
                if last_error_log.is_none_or(|t| t.elapsed() >= Duration::from_secs(10)) {
                    tracing::warn!(target: LOG, "utterance seq={} failed: {e}", job.seq);
                    last_error_log = Some(Instant::now());
                }
            }
        }
    }
    tracing::debug!(target: LOG, "worker for recording {epoch} exited");
}

/// Route a VAD event of recording `epoch` (capture worker or fed audio).
fn handle_vad_event<R: Runtime>(
    state: &WhisperState,
    app: &AppHandle<R>,
    epoch: u64,
    queue: &BoundedQueue<WhisperJob>,
    event: SegmenterEvent,
) {
    match event {
        SegmenterEvent::SpeechStart { seq, start_sample } => {
            state.emit_for(
                app,
                epoch,
                "whisper:speech_start",
                SpeechStartPayload {
                    seq,
                    start_ms: samples_to_ms(start_sample),
                },
            );
        }
        SegmenterEvent::Segment(segment) => {
            if !state.epoch_is(epoch) {
                return;
            }
            let dropped = queue.push(WhisperJob {
                seq: segment.seq,
                samples: segment.samples,
            });
            for old in dropped {
                let audio_ms = samples_to_ms(old.samples.len() as u64);
                tracing::warn!(
                    target: LOG,
                    "utterance seq={} ({audio_ms} ms) dropped: Whisper is slower than real time (backlog cap {})",
                    old.seq,
                    queue.capacity()
                );
                state.emit_for(
                    app,
                    epoch,
                    "whisper:utterance_dropped",
                    UtteranceDroppedPayload {
                        seq: old.seq,
                        audio_ms,
                        reason: "backlog",
                    },
                );
            }
        }
    }
}

// The argument list is the IPC contract with `whisper.ts` (one named field per setting). Legacy RMS
// fields (`silenceThresholdDb`, `silenceDurationMs`, `minChunkDurationMs`) are still accepted by the
// frontend call and ignored: Silero VAD (`vad`) replaced the RMS gate.
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
    // Stop any previous recording and claim an epoch before the (slow) device open.
    let (my_epoch, previous) = {
        let mut session = state.lock_session();
        let epoch = state.recording_epoch.fetch_add(1, Ordering::SeqCst) + 1;
        (epoch, session.take())
    };
    if let Some(previous) = previous {
        previous.shutdown().await;
    }

    let vad = resolve_vad(vad, vad_enabled);
    let settings = DecodeSettings::resolve(&decode.unwrap_or_default());
    let queue = Arc::new(BoundedQueue::new(settings.max_backlog));
    let whisper = state.inner().clone();

    let source = if capture_local.unwrap_or(true) {
        let handler = {
            let whisper = whisper.clone();
            let app = app.clone();
            let queue = Arc::clone(&queue);
            move |event| match event {
                SpeechCaptureEvent::Vad(ev) => handle_vad_event(&whisper, &app, my_epoch, &queue, ev),
                SpeechCaptureEvent::Error(message) => {
                    whisper.emit_for(&app, my_epoch, "whisper:capture_error", CaptureErrorPayload { message });
                }
            }
        };
        AudioSource::Capture(start_speech_capture(device_name, vad.clone(), "Whisper", handler).await?)
    } else {
        // Audio arrives through `feed_audio_chunk` as 16 kHz mono.
        let vad = vad.clone();
        let front_end = tauri::async_runtime::spawn_blocking(move || SpeechFrontEnd::new(SAMPLE_RATE, 1, &vad))
            .await
            .map_err(|e| format!("Speech detector task failed: {e}"))??;
        AudioSource::Fed(Arc::new(Mutex::new(front_end)))
    };

    let session = RecordingSession {
        epoch: my_epoch,
        source,
        queue: Arc::clone(&queue),
    };
    let cancelled = {
        let mut current = state.lock_session();
        if state.epoch_is(my_epoch) {
            *current = Some(session);
            None
        } else {
            Some(session)
        }
    };
    if let Some(session) = cancelled {
        // A stop (or a newer start) landed while the device was opening.
        session.shutdown().await;
        return Err("cancelled".to_string());
    }

    tracing::info!(
        target: LOG,
        "recording {my_epoch} started (threads={}, prompt={}B, backlog={}, vad={:?})",
        settings.n_threads,
        settings.prompt.len(),
        settings.max_backlog,
        vad.sanitized()
    );
    let spawned = thread::Builder::new().name("sigil-whisper".into()).spawn({
        let app = app.clone();
        let queue = Arc::clone(&queue);
        move || run_whisper_worker(app, whisper, my_epoch, queue, settings)
    });
    if let Err(e) = spawned {
        let session = {
            let mut current = state.lock_session();
            state.recording_epoch.fetch_add(1, Ordering::SeqCst);
            current.take()
        };
        if let Some(session) = session {
            session.shutdown().await;
        }
        return Err(format!("Failed to start the Whisper worker: {e}"));
    }
    Ok(())
}

/// Feed 16 kHz mono samples that do not come from Whisper's own capture (the `feed_audio_chunk`
/// command). Ignored unless a recording was started with `captureLocal: false`; the samples go
/// through the same Silero gate as microphone audio, so silence never reaches Whisper. The web
/// pubsub binary-frame path that used to call this is gone (LAN security stage 1).
fn process_audio_chunk<R: Runtime>(state: &WhisperState, app: &AppHandle<R>, data: Vec<f32>) {
    let (front_end, queue, epoch) = {
        let session = state.lock_session();
        match session.as_ref() {
            Some(RecordingSession {
                source: AudioSource::Fed(front_end),
                queue,
                epoch,
            }) => (Arc::clone(front_end), Arc::clone(queue), *epoch),
            _ => return,
        }
    };
    let mut events = Vec::new();
    {
        let mut front_end = front_end.lock().unwrap_or_else(PoisonError::into_inner);
        if let Err(e) = front_end.push_interleaved(&data, &mut events) {
            tracing::warn!(target: LOG, "fed audio rejected: {e}");
            return;
        }
    }
    for event in events {
        handle_vad_event(state, app, epoch, &queue, event);
    }
}

#[command]
async fn feed_audio_chunk<R: Runtime>(app: AppHandle<R>, state: State<'_, WhisperState>, chunk: Vec<f32>) -> Result<(), String> {
    let state = state.inner().clone();
    // Silero runs here; keep it off the async runtime threads.
    tauri::async_runtime::spawn_blocking(move || process_audio_chunk(&state, &app, chunk))
        .await
        .map_err(|e| e.to_string())
}

#[command]
fn cancel_whisper_download(state: State<'_, WhisperState>) -> Result<(), String> {
    // Makes every in-flight `ensure_dependencies` run stale; each one then removes only its own partial file.
    state.ensure_deps_epoch.fetch_add(1, Ordering::SeqCst);
    Ok(())
}

#[command]
async fn stop_recording<R: Runtime>(_app: AppHandle<R>, state: State<'_, WhisperState>) -> Result<String, String> {
    let previous = {
        let mut session = state.lock_session();
        // Cancels a `start_recording` still opening its device (it re-checks the epoch under this lock) and
        // aborts an in-flight decode; nothing of the ended recording is emitted after this block.
        state.recording_epoch.fetch_add(1, Ordering::SeqCst);
        session.take()
    };
    match previous {
        Some(session) => {
            // Waits (bounded) until the microphone is released before the caller moves on.
            session.shutdown().await;
            tracing::info!(target: LOG, "recording stopped");
            Ok(String::new())
        }
        None => Err("Not recording".to_string()),
    }
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

    #[test]
    fn catalog_resolves_every_id_and_rejects_unknown_ones() {
        for (id, file) in CATALOG {
            let info = get_model_info(id).unwrap();
            assert_eq!((info.id, info.filename), (*id, *file));
            assert!(file.starts_with("ggml-") && file.ends_with(".bin"));
            assert_eq!(
                info.download_url().unwrap(),
                format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{file}?download=true")
            );
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

    #[test]
    fn anime_whisper_is_a_japanese_bring_your_own_preset() {
        let info = get_model_info(ANIME_WHISPER_ID).unwrap();
        assert!(info.bring_your_own);
        assert_eq!(info.download_url(), None);
        assert_eq!(info.language, Some("ja"));
        assert!(!info.accepts_prompt);
        assert!(bring_your_own_missing(&info).contains("convert-h5-to-ggml.py"));
    }

    #[test]
    fn decode_options_resolve_to_safe_values() {
        let d = DecodeSettings::resolve(&WhisperDecodeOptions::default());
        assert!((1..=4).contains(&d.n_threads));
        assert_eq!((d.no_speech_threshold, d.hallucination_filter, d.max_backlog), (0.6, true, 3));
        assert!(d.prompt.is_empty());

        let d = DecodeSettings::resolve(&WhisperDecodeOptions {
            n_threads: Some(1),
            initial_prompt: Some("  Chat in VRChat. ".into()),
            hotwords: Some(vec!["Sigil".into(), " ".into(), "Kokoro".into()]),
            no_speech_threshold: Some(f32::NAN),
            hallucination_filter: Some(false),
            max_backlog: Some(99),
        });
        assert_eq!(d.n_threads, 1);
        assert_eq!(d.prompt, "Chat in VRChat. Sigil, Kokoro.");
        assert_eq!((d.no_speech_threshold, d.hallucination_filter, d.max_backlog), (0.6, false, 10));

        // NUL bytes are removed and long prompts are capped on a char boundary.
        let d = DecodeSettings::resolve(&WhisperDecodeOptions {
            initial_prompt: Some(format!("a\0b{}", "語".repeat(600))),
            ..Default::default()
        });
        assert!(!d.prompt.contains('\0') && d.prompt.starts_with("ab"));
        assert!(d.prompt.len() <= MAX_PROMPT_BYTES);
        let json: WhisperDecodeOptions = serde_json::from_str(r#"{"n_threads":2,"hotwords":["a"]}"#).unwrap();
        assert_eq!(json.n_threads, Some(2));
    }

    #[test]
    fn token_cap_scales_with_utterance_length() {
        assert_eq!(max_tokens_for(8_000), 48);
        assert_eq!(max_tokens_for(16_000 * 5), 92);
        assert_eq!(max_tokens_for(16_000 * 18), 220);
        assert_eq!(max_tokens_for(16_000 * 60), 220);
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
    fn whisper_language_codes_are_validated() {
        assert_eq!(whisper_language("auto"), None);
        assert_eq!(whisper_language(""), None);
        assert_eq!(whisper_language("en"), Some("en"));
        assert_eq!(whisper_language("zh-TW"), Some("zh"));
        assert_eq!(whisper_language("pt_BR"), Some("pt"));
        assert_eq!(whisper_language("xx"), None, "unknown codes auto-detect instead of failing");
    }

    /// Real-model check (manual): `SIGIL_TEST_MODELS_DIR` must hold `ggml-tiny.en.bin`,
    /// `ggml-base.en.bin` or `ggml-base.bin`. Run with `cargo test real_model -- --ignored`.
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with a ggml Whisper model"]
    fn real_model_transcribes_speech_and_filters_silence() {
        let Ok(dir) = std::env::var("SIGIL_TEST_MODELS_DIR") else { return };
        let dir = PathBuf::from(dir);
        let Some(model) = ["ggml-tiny.en.bin", "ggml-base.en.bin", "ggml-base.bin"]
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
        else {
            return;
        };
        let ctx = WhisperContext::new_with_params(model.to_str().unwrap(), WhisperContextParameters::default()).unwrap();
        let state = WhisperState::new();
        let info = get_model_info("base.en").unwrap();
        *state.ctx.lock().unwrap() = Some(LoadedModel::new(ctx, &info));
        let settings = DecodeSettings::resolve(&WhisperDecodeOptions {
            hotwords: Some(vec!["weather".into()]),
            ..Default::default()
        });
        let abort = AbortToken {
            epoch: Arc::clone(&state.recording_epoch),
            mine: state.recording_epoch.load(Ordering::SeqCst),
        };

        let speech = crate::services::speech_core::silero::tests::speech_fixture();
        let t = transcribe_utterance(&state, &settings, &speech, &abort).unwrap();
        let text = t.text.expect("speech transcribed").to_lowercase();
        assert!(text.contains("weather"), "{text}");
        // The same WhisperState is reused for the next utterance.
        assert!(state.ctx.lock().unwrap().as_ref().unwrap().state.is_some());
        let again = transcribe_utterance(&state, &settings, &speech, &abort).unwrap();
        assert_eq!(again.text.map(|t| t.to_lowercase()), Some(text));

        assert!(t.no_speech_prob < 0.1, "speech no-speech prob {}", t.no_speech_prob);

        // Silence or noise that slipped past the VAD produces no caption, even with a prompt (whisper.cpp
        // alone would emit "you" / "Thank you." here on multilingual models).
        let silence = vec![0.0f32; 16_000 * 3];
        let s = transcribe_utterance(&state, &settings, &silence, &abort).unwrap();
        assert_eq!(s.text, None);
        assert!(s.no_speech_prob > 0.6, "silence no-speech prob {}", s.no_speech_prob);
        let mut seed = 7u32;
        let noise: Vec<f32> = (0..16_000 * 3)
            .map(|_| {
                seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((seed >> 8) as f32 / 16_777_216.0 - 0.5) * 0.01
            })
            .collect();
        assert_eq!(transcribe_utterance(&state, &settings, &noise, &abort).unwrap().text, None);
        // Speech right after noise is still transcribed (no stale no-speech value carries over).
        assert!(transcribe_utterance(&state, &settings, &speech, &abort).unwrap().text.is_some());

        // A bumped epoch aborts inference promptly and the state is discarded.
        state.recording_epoch.fetch_add(1, Ordering::SeqCst);
        let long: Vec<f32> = speech.iter().cycle().take(16_000 * 15).copied().collect();
        let started = Instant::now();
        assert_eq!(transcribe_utterance(&state, &settings, &long, &abort).err().as_deref(), Some(ABORTED));
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
