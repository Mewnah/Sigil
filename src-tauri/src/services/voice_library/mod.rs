//! `voice-library` plugin: the shared reference-voice recorder and library (P5-F; scaffolded by
//! P5-S0). One clean 3–10 s clip per voice, recorded through the capture hub (or imported from a
//! file), trimmed by Silero VAD, checked for SNR and clipping, loudness-normalised, transcribed by
//! the local STT, and stored under `app_data/voices/<id>/{ref.wav, meta.json, cache/<backend>.bin}`.
//! The voice changer's zero-shot families (Sigil-ZS, MeanVC2, sidecar v2) and TTS cloning (P6:
//! Qwen3-TTS, Chatterbox) read the same voices; each backend caches its embedding in
//! `cache/<backend>.bin` (`vc::families::reference::cached_embedding` writes the voice changer's).
//!
//! Scaffolded by P5-S0 (plugin registered in main.rs, capability `capabilities/voice-library.json`,
//! commands in `plugin-commands/voice-library.txt`). Everything in this folder is P5-F's; the
//! signatures below are frozen (change only through INTEGRATE; structs may gain fields).
//!
//! # Frozen contract (P5-S0)
//! **Rust API** (other modules; the voice changer's factory and families use it):
//! - [`voices_dir`]`(app_data)`, [`valid_voice_id`]`(id)`, [`voice_dir`]`(app_data, id)`.
//! - [`list_voices`]`(app_data) -> Result<Vec<VoiceMeta>, String>`.
//! - [`reference_clip`]`(app_data, id) -> Result<ReferenceClip, String>`: the voice's `ref.wav` as a
//!   `vc::families::reference::ReferenceClip` (its `sha256` is the clip file's, so a re-recorded
//!   clip invalidates every cached embedding).
//! - [`read_cache`]`(app_data, id, backend) -> Option<Vec<u8>>` and
//!   [`write_cache`]`(app_data, id, backend, &[u8]) -> Result<(), String>`: `cache/<backend>.bin`
//!   (`backend` is a safe file stem, `reference::valid_backend`; written atomically; a cache is
//!   only an optimisation and may be deleted at any time).
//!
//! **IPC** (payloads and arguments camelCase; errors are strings; TS mirror
//! `src/core/services/voice_library/contract.ts`):
//! - `voice_library_list -> VoiceMeta[]`, `voice_library_get { id } -> VoiceMeta`.
//! - `voice_library_record_start { request: RecordRequest } -> ()`: capture from the hub (the raw
//!   16 kHz feed is not used: the take is recorded at the device rate), progress events
//!   [`RECORD_PROGRESS_EVENT`] (≤ 20 Hz, [`RecordProgress`]).
//! - `voice_library_record_stop -> VoiceTake` (trimmed, checked and normalised; not saved yet),
//!   `voice_library_record_cancel -> ()`.
//! - `voice_library_import_clip { request: ImportClipRequest } -> VoiceTake`: a WAV / audio file
//!   through the same trim, checks and normalisation.
//! - `voice_library_read_take { takeId }` / `voice_library_read_clip { id }` -> WAV bytes (binary
//!   IPC, `tauri::ipc::Response`; never PCM over JSON) for preview playback.
//! - `voice_library_discard_take { takeId } -> ()`.
//! - `voice_library_save_take { request: SaveTakeRequest } -> VoiceMeta` (transcribes with the local
//!   STT when asked).
//! - `voice_library_update { request: UpdateVoiceRequest } -> VoiceMeta` (name, transcript,
//!   language), `voice_library_delete { id } -> ()`.
//! - Event [`CHANGED_EVENT`] `"voice_library:changed"` (no payload) after save, update and delete.
//!
//! # Implementation (P5-F)
//! - [`process`]: one pipeline for recorded and imported takes: rumble high-pass, Silero VAD
//!   ([`trim`]: speech plus pads, long pauses spliced down, at most 10 s), clipping, peak and
//!   speech-to-noise checks, BS.1770 loudness normalisation to −20 LUFS under a −1 dBFS peak
//!   ([`dsp`]), stored as 16-bit PCM mono WAV at the source rate (≤ 48 kHz). Issues `no_speech` and
//!   `too_short` block saving; `too_long` (cut to 10 s), `noisy` (< 15 dB), `clipped` and
//!   `too_quiet` are warnings.
//! - [`recorder`]: one take at a time on its own thread, a hub consumer of its own; it stops by
//!   itself at `maxSeconds` or when the device changes / drops out (the last progress event then
//!   has `stopped` and `stopReason`). `RecordRequest::device` `""` is the voice changer's input
//!   device (its saved settings), else the system default.
//! - [`takes`]: unsaved takes live in memory (the newest six).
//! - [`store`]: the folder layout, atomic writes, one library lock; ids are a slug of the name plus
//!   a random suffix and are never reused.
//! - [`transcribe`]: after a save, the local STT (installed models only, never a download) writes
//!   the transcript in the background (`VoiceMeta::transcribing`, then `voice_library:changed`); a
//!   user edit made meanwhile wins. Without an installed model, `transcriptError` is
//!   `no_stt_model`.
//! - P5-F additions to the frozen types (fields only): `VoiceMeta::{transcribing,
//!   transcript_error}`, `VoiceTake::sample_rate`, `RecordProgress::{stopped, stop_reason}`,
//!   `UpdateVoiceRequest::{take_id, transcribe}` (replace the clip, keeping the id; re-run the STT).

use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Runtime,
};

use crate::services::neural_audio::vc::families::reference::ReferenceClip;
use takes::{TakeStore, TAKES};

pub mod dsp;
pub mod process;
pub mod recorder;
pub mod store;
pub mod takes;
pub mod transcribe;
pub mod trim;

const LOG: &str = "sigil::voice_library";

/// Plugin name (`plugin:voice-library|...`, capability `voice-library:default`).
pub const PLUGIN_NAME: &str = "voice-library";
/// Event: the library changed (no payload).
pub const CHANGED_EVENT: &str = "voice_library:changed";
/// Event: recording progress ([`RecordProgress`], at most 20 per second).
pub const RECORD_PROGRESS_EVENT: &str = "voice_library:record_progress";
/// Folder under app data.
pub const VOICES_DIR: &str = "voices";
/// The clip of a voice.
pub const CLIP_FILE: &str = "ref.wav";
/// The metadata of a voice ([`VoiceMeta`]).
pub const META_FILE: &str = "meta.json";
/// Per-backend embedding caches (`cache/<backend>.bin`).
pub const CACHE_DIR: &str = "cache";
/// Shortest and longest saved clip (seconds, after trimming).
pub const CLIP_SECONDS: (f64, f64) = (3.0, 10.0);

/// One library voice (`meta.json`; camelCase on the wire and on disk).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VoiceMeta {
    /// Folder name under `voices/` (stable id, [`valid_voice_id`]).
    pub id: String,
    pub name: String,
    pub created_unix_ms: u64,
    pub duration_ms: u64,
    /// `ref.wav`'s sample rate (Hz).
    pub sample_rate: u32,
    /// Integrated loudness of the saved clip (LUFS, after normalisation).
    pub lufs: f32,
    /// Speech-to-noise ratio estimate (dB), when measurable.
    pub snr_db: Option<f32>,
    /// Sample peak of the saved clip (dBFS, after normalisation).
    pub peak_dbfs: f32,
    /// Some samples were clipped in the original take.
    pub clipped: bool,
    /// What is said in the clip (local STT, or edited by the user).
    pub transcript: Option<String>,
    /// BCP-47 language of the transcript.
    pub language: Option<String>,
    /// `recorded` or `imported`.
    pub source: String,
    /// sha256 (hex) of `ref.wav`.
    pub sha256: String,
    /// Backends with a cached embedding (`cache/<backend>.bin`).
    pub cached_backends: Vec<String>,
    /// The local STT is writing the transcript (runtime state, not stored).
    pub transcribing: bool,
    /// Why the last automatic transcript failed: `no_stt_model` (no local speech-to-text model is
    /// installed), `failed`, or `empty` (nothing recognisable). Cleared by a transcript.
    pub transcript_error: Option<String>,
}

/// The checks of a take ([`VoiceTake::quality`]).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ClipQuality {
    /// Length after trimming (ms).
    pub duration_ms: u64,
    /// Audio trimmed off (ms): silence at the ends, long pauses spliced down, anything past 10 s.
    pub trimmed_ms: u64,
    pub snr_db: Option<f32>,
    /// Sample peak of the kept audio before normalisation (dBFS).
    pub peak_dbfs: f32,
    /// Fraction of clipped samples (0..1).
    pub clipped_ratio: f32,
    /// Loudness before and after normalisation (LUFS).
    pub lufs_before: f32,
    pub lufs_after: f32,
    /// Usable as a reference (no blocking issue).
    pub ok: bool,
    /// Issue codes: `too_short`, `too_long`, `no_speech`, `noisy`, `clipped`, `too_quiet`
    /// (`no_speech` and `too_short` block saving; the others are warnings).
    pub issues: Vec<String>,
}

/// A recorded or imported clip waiting to be saved (or discarded).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct VoiceTake {
    pub take_id: String,
    pub quality: ClipQuality,
    /// `recorded` or `imported`.
    pub source: String,
    /// Suggested name (the imported file's stem).
    pub suggested_name: Option<String>,
    /// Sample rate of the processed clip (Hz).
    pub sample_rate: u32,
}

/// `voice_library_record_start { request }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RecordRequest {
    /// Capture device name (`""` = the voice changer's input device, else the system default).
    pub device: String,
    /// Recording stops by itself after this many seconds (default 12, at most 30).
    pub max_seconds: f32,
}

impl Default for RecordRequest {
    fn default() -> Self {
        RecordRequest { device: String::new(), max_seconds: 12.0 }
    }
}

/// Payload of [`RECORD_PROGRESS_EVENT`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RecordProgress {
    /// Audio recorded so far (ms).
    pub elapsed_ms: u64,
    /// RMS level of the last ~50 ms (dBFS).
    pub level_db: f32,
    /// Sample peak of the last ~50 ms (dBFS).
    pub peak_db: f32,
    /// A clipped sample was recorded in this take (sticky).
    pub clipping: bool,
    /// Speech heard so far (Silero), ms.
    pub voiced_ms: u64,
    /// The recorder stopped by itself (the last event of a take): call `record_stop` for the take.
    pub stopped: bool,
    /// Why it stopped: `max_length`, `device_changed`, `interrupted`, `device_lost`, `no_audio`.
    pub stop_reason: Option<String>,
}

/// `voice_library_import_clip { request }`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct ImportClipRequest {
    pub path: String,
}

/// `voice_library_save_take { request }`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SaveTakeRequest {
    pub take_id: String,
    pub name: String,
    /// Transcribe with the local STT (default true).
    pub transcribe: bool,
    /// Language hint for the transcript (BCP-47; `None` = detect).
    pub language: Option<String>,
}

impl Default for SaveTakeRequest {
    fn default() -> Self {
        SaveTakeRequest { take_id: String::new(), name: String::new(), transcribe: true, language: None }
    }
}

/// `voice_library_update { request }` (absent fields are kept).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct UpdateVoiceRequest {
    pub id: String,
    pub name: Option<String>,
    /// `""` clears it.
    pub transcript: Option<String>,
    /// `""` / `"auto"` clears it.
    pub language: Option<String>,
    /// Replace the clip with this take (re-record a voice and keep its id, so voices and presets
    /// that use it follow). Clears the cached embeddings and, unless `transcript` is given, the
    /// transcript (re-transcribed unless `transcribe` is `false`).
    pub take_id: Option<String>,
    /// `true`: run the local STT again and replace the transcript.
    pub transcribe: Option<bool>,
}

/// `app_data/voices`.
pub fn voices_dir(app_data: &Path) -> PathBuf {
    app_data.join(VOICES_DIR)
}

/// A voice id: 1–64 characters of `[a-z0-9-]`, not starting or ending with `-`.
pub fn valid_voice_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && !id.starts_with('-')
        && !id.ends_with('-')
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// `app_data/voices/<id>` (the id is validated; the folder may not exist).
pub fn voice_dir(app_data: &Path, id: &str) -> Result<PathBuf, String> {
    if !valid_voice_id(id) {
        return Err(format!("invalid voice id {id:?}"));
    }
    Ok(voices_dir(app_data).join(id))
}

/// Every library voice, sorted by name.
pub fn list_voices(app_data: &Path) -> Result<Vec<VoiceMeta>, String> {
    store::list(app_data)
}

/// The clip of the library voice `id`.
pub fn reference_clip(app_data: &Path, id: &str) -> Result<ReferenceClip, String> {
    store::reference_clip(app_data, id)
}

/// The cached embedding `cache/<backend>.bin` of voice `id`, if any.
pub fn read_cache(app_data: &Path, id: &str, backend: &str) -> Option<Vec<u8>> {
    store::read_cache(app_data, id, backend)
}

/// Store `cache/<backend>.bin` of voice `id` (atomically).
pub fn write_cache(app_data: &Path, id: &str, backend: &str, bytes: &[u8]) -> Result<(), String> {
    if store::write_cache(app_data, id, backend, bytes)? {
        notify_changed();
    }
    Ok(())
}

// ---- Events ---------------------------------------------------------------------------------

static CHANGED: OnceLock<Box<dyn Fn() + Send + Sync>> = OnceLock::new();

/// Emit [`CHANGED_EVENT`] (no-op before the plugin is set up, e.g. headless or in tests).
fn notify_changed() {
    if let Some(emit) = CHANGED.get() {
        emit();
    }
}

// ---- Operations (pure over `app_data`; the commands below run them off the async runtime) -----

use transcribe::Transcriber;

/// The app's transcriber.
fn local_stt() -> Transcriber {
    Arc::new(transcribe::transcribe)
}

fn app_data() -> Result<PathBuf, String> {
    crate::services::diagnostics::app_data_dir().ok_or_else(|| "the voice library is not available (no app data folder)".to_string())
}

/// The capture device of a take: the request's, else the voice changer's saved input device.
fn record_device(requested: &str, app_data: Option<&Path>) -> String {
    let d = requested.trim();
    if !d.is_empty() {
        return d.to_string();
    }
    app_data
        .and_then(|a| std::fs::read(a.join("voice-changer").join("settings.json")).ok())
        .and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok())
        .and_then(|v| v.get("params")?.get("input_device")?.as_str().map(|s| s.trim().to_string()))
        .unwrap_or_default()
}

/// A language as stored: `None` for empty / `auto`, else the normalised code (an error when it is
/// not a language tag).
fn stored_language(language: Option<&str>) -> Result<Option<String>, String> {
    match language.map(str::trim) {
        None => Ok(None),
        Some(l) if crate::services::speech::lang::is_auto(l) => Ok(None),
        Some(l) => crate::services::speech::lang::normalize(l)
            .map(Some)
            .ok_or_else(|| format!("{l:?} is not a language code")),
    }
}

/// Process raw mono audio into a held take.
fn make_take(mono: Vec<f32>, rate: u32, source: &str, suggested_name: Option<String>, vad: &mut dyn process::Vad) -> Result<takes::Take, String> {
    let processed = process::process_take(mono, rate, vad)?;
    let wav = process::encode_wav(&processed.clip, processed.rate)?;
    let info = VoiceTake {
        take_id: takes::new_take_id(),
        quality: processed.quality.clone(),
        source: source.to_string(),
        suggested_name,
        sample_rate: processed.rate,
    };
    Ok(takes::Take { info, processed, wav })
}

fn hold(store: &TakeStore, take: takes::Take) -> VoiceTake {
    let info = take.info.clone();
    store.insert(take);
    info
}

/// Import an audio file as a take.
fn import_clip(held: &TakeStore, path: &str, vad: &mut dyn process::Vad) -> Result<VoiceTake, String> {
    let path = PathBuf::from(path.trim());
    if path.as_os_str().is_empty() {
        return Err("no file was given".into());
    }
    let (mono, rate) = process::decode_file(&path)?;
    let suggested = path.file_stem().and_then(|s| store::clean_name(&s.to_string_lossy()));
    Ok(hold(held, make_take(mono, rate, "imported", suggested, vad)?))
}

fn blocking_issues(q: &ClipQuality) -> String {
    q.issues
        .iter()
        .filter(|i| process::BLOCKING_ISSUES.contains(&i.as_str()))
        .cloned()
        .collect::<Vec<_>>()
        .join(", ")
}

/// The take as 16 kHz mono (what the STT reads).
fn audio16k(take: &process::ProcessedTake) -> Vec<f32> {
    crate::services::bench::signals::resample(&take.clip, take.rate, 16_000).unwrap_or_default()
}

fn queue_transcript(app_data: &Path, meta: &VoiceMeta, audio: Vec<f32>, transcriber: Transcriber) {
    if audio.is_empty() {
        return;
    }
    let job = transcribe::Job::new(app_data, &meta.id, &meta.sha256, audio, meta.language.clone());
    transcribe::spawn(job, transcriber, Arc::new(notify_changed));
}

/// Save a take as a new voice.
fn save_take(app_data: &Path, held: &TakeStore, request: &SaveTakeRequest, transcriber: Transcriber) -> Result<VoiceMeta, String> {
    // The take is claimed for the whole save, so a second save of it (a double Enter, a second
    // click) finds it gone instead of storing a second voice; a refused or failed save puts it back.
    let take = held.take(request.take_id.trim())?;
    let saved = match create_voice(app_data, &take, request) {
        Ok(saved) => saved,
        Err(e) => {
            held.insert(take);
            return Err(e);
        }
    };
    tracing::info!(target: LOG, "voice {} saved ({} ms, {:.1} LUFS)", saved.id, saved.duration_ms, saved.lufs);
    if request.transcribe {
        queue_transcript(app_data, &saved, audio16k(&take.processed), transcriber);
    }
    notify_changed();
    store::get(app_data, &saved.id)
}

/// Check a claimed take and store it as a new voice.
fn create_voice(app_data: &Path, take: &takes::Take, request: &SaveTakeRequest) -> Result<VoiceMeta, String> {
    if !take.info.quality.ok {
        return Err(format!("this take cannot be saved ({})", blocking_issues(&take.info.quality)));
    }
    let name = store::clean_name(&request.name).ok_or("the voice needs a name")?;
    let language = stored_language(request.language.as_deref())?;
    let q = &take.info.quality;
    let meta = VoiceMeta {
        name,
        created_unix_ms: store::now_unix_ms(),
        duration_ms: q.duration_ms,
        sample_rate: take.processed.rate,
        lufs: q.lufs_after,
        snr_db: q.snr_db,
        peak_dbfs: dsp::amp_to_db(f64::from(dsp::peak(&take.processed.clip))),
        clipped: take.processed.clipped,
        language,
        source: take.info.source.clone(),
        ..VoiceMeta::default()
    };
    store::create(app_data, &take.wav, meta)
}

/// Apply an update request.
fn update_voice(app_data: &Path, held: &TakeStore, request: &UpdateVoiceRequest, transcriber: Transcriber) -> Result<VoiceMeta, String> {
    let id = request.id.trim();
    store::get(app_data, id)?;
    // Validate everything before changing anything.
    let name = match &request.name {
        Some(n) => Some(store::clean_name(n).ok_or("the voice needs a name")?),
        None => None,
    };
    let language = match &request.language {
        Some(l) => Some(stored_language(Some(l))?),
        None => None,
    };
    // Claimed like a save's take (two Replace clicks never both apply it); put back on failure.
    let take = match request.take_id.as_deref().map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => {
            let take = held.take(t)?;
            if !take.info.quality.ok {
                let e = format!("this take cannot be used ({})", blocking_issues(&take.info.quality));
                held.insert(take);
                return Err(e);
            }
            Some(take)
        }
        None => None,
    };
    let replaced = take.is_some();
    let mut new_audio: Option<Vec<f32>> = None;
    if let Some(take) = take {
        let q = take.info.quality.clone();
        let rate = take.processed.rate;
        let peak = dsp::amp_to_db(f64::from(dsp::peak(&take.processed.clip)));
        let clipped = take.processed.clipped;
        let source = take.info.source.clone();
        let keep_transcript = request.transcript.is_some();
        let result = store::replace_clip(app_data, id, &take.wav, |m| {
            m.duration_ms = q.duration_ms;
            m.sample_rate = rate;
            m.lufs = q.lufs_after;
            m.snr_db = q.snr_db;
            m.peak_dbfs = peak;
            m.clipped = clipped;
            m.source = source;
            if !keep_transcript {
                m.transcript = None;
                m.transcript_error = None;
            }
        });
        if let Err(e) = result {
            held.insert(take);
            return Err(e);
        }
        new_audio = Some(audio16k(&take.processed));
    }
    let edits_transcript = request.transcript.is_some();
    let meta = store::update(app_data, id, |m| {
        if let Some(n) = &name {
            m.name = n.clone();
        }
        if let Some(t) = &request.transcript {
            m.transcript = store::clean_transcript(t);
            m.transcript_error = None;
            // Under the library lock, like a job's apply: a job that finishes later sees the edit.
            transcribe::note_user_edit(id);
        }
        if let Some(l) = &language {
            m.language = l.clone();
        }
        Ok(())
    })?;
    let rerun = match request.transcribe {
        Some(t) => t,
        None => replaced && !edits_transcript,
    };
    if rerun {
        let audio = match new_audio {
            Some(a) => a,
            None => {
                let wav = std::fs::read(store::clip_path(&voice_dir(app_data, id)?)).map_err(|e| format!("cannot read the clip of voice {id:?}: {e}"))?;
                let (mono, rate) = process::decode_wav_bytes(&wav)?;
                crate::services::bench::signals::resample(&mono, rate, 16_000)?
            }
        };
        queue_transcript(app_data, &meta, audio, transcriber);
    }
    notify_changed();
    store::get(app_data, id)
}

fn delete_voice(app_data: &Path, id: &str) -> Result<(), String> {
    store::delete(app_data, id.trim())?;
    tracing::info!(target: LOG, "voice {} deleted", id.trim());
    notify_changed();
    Ok(())
}

/// The finished take's audio as a held take (blocking: Silero and the processing).
fn finish_recording(held: &TakeStore, captured: recorder::Captured) -> Result<VoiceTake, String> {
    if captured.mono.is_empty() || captured.rate == 0 {
        return Err(match (captured.stop_reason, captured.error) {
            (_, Some(e)) => format!("the microphone stopped delivering audio: {e}"),
            (Some(recorder::STOP_NO_AUDIO), None) => "no audio arrived from the microphone (is it muted or used exclusively by another program?)".into(),
            _ => "nothing was recorded".into(),
        });
    }
    let mut vad = process::silero()?;
    Ok(hold(held, make_take(captured.mono, captured.rate, "recorded", None, &mut vad)?))
}

async fn blocking<T: Send + 'static>(f: impl FnOnce() -> Result<T, String> + Send + 'static) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(f)
        .await
        .map_err(|e| format!("voice library task failed: {e}"))?
}

// ---- IPC ----------------------------------------------------------------------------------------

#[command]
async fn voice_library_list() -> Result<Vec<VoiceMeta>, String> {
    blocking(|| list_voices(&app_data()?)).await
}

#[command]
async fn voice_library_get(id: String) -> Result<VoiceMeta, String> {
    blocking(move || store::get(&app_data()?, id.trim())).await
}

#[command]
async fn voice_library_record_start<R: Runtime>(app: AppHandle<R>, request: RecordRequest) -> Result<(), String> {
    use crate::services::audio::hub::SourceSpec;
    let max_seconds = recorder::sanitize_max_seconds(request.max_seconds);
    let device = record_device(&request.device, app_data().ok().as_deref());
    let (epoch, previous) = recorder::RECORDER.begin();
    if let Some(previous) = previous {
        // A take was still running: it is replaced (its audio is dropped).
        tauri::async_runtime::spawn_blocking(move || drop(previous.finish()));
    }
    let spec = SourceSpec::parse(Some(&device));
    let consumer = crate::services::audio::attach_async(spec, recorder::consumer_options())
        .await
        .map_err(|e| format!("the microphone could not be opened: {e}"))?;
    let sink: recorder::ProgressSink = Arc::new(move |p: &RecordProgress| {
        let _ = app.emit(RECORD_PROGRESS_EVENT, p);
    });
    let make_vad: recorder::VadFactory = Box::new(|| -> Option<Box<dyn process::Vad + Send>> {
        match process::silero() {
            Ok(v) => Some(Box::new(v)),
            Err(e) => {
                tracing::warn!(target: LOG, "recorder: no live speech detection: {e}");
                None
            }
        }
    });
    recorder::RECORDER.start(epoch, Box::new(recorder::HubSource(consumer)), max_seconds, sink, make_vad)?;
    tracing::info!(target: LOG, "recording a reference take from {:?} (at most {max_seconds} s)", if device.is_empty() { "default" } else { device.as_str() });
    Ok(())
}

#[command]
async fn voice_library_record_stop() -> Result<VoiceTake, String> {
    let session = recorder::RECORDER.end().ok_or("nothing is being recorded")?;
    blocking(move || finish_recording(&TAKES, session.finish())).await
}

#[command]
async fn voice_library_record_cancel() -> Result<(), String> {
    if let Some(session) = recorder::RECORDER.end() {
        blocking(move || {
            drop(session.finish());
            Ok(())
        })
        .await?;
    }
    Ok(())
}

#[command]
async fn voice_library_import_clip(request: ImportClipRequest) -> Result<VoiceTake, String> {
    blocking(move || {
        let mut vad = process::silero()?;
        import_clip(&TAKES, &request.path, &mut vad)
    })
    .await
}

#[command]
async fn voice_library_read_take(take_id: String) -> Result<tauri::ipc::Response, String> {
    TAKES.wav(take_id.trim()).map(tauri::ipc::Response::new)
}

#[command]
async fn voice_library_read_clip(id: String) -> Result<tauri::ipc::Response, String> {
    blocking(move || {
        let dir = voice_dir(&app_data()?, id.trim())?;
        std::fs::read(store::clip_path(&dir)).map_err(|e| format!("cannot read the clip of voice {:?}: {e}", id.trim()))
    })
    .await
    .map(tauri::ipc::Response::new)
}

#[command]
async fn voice_library_discard_take(take_id: String) -> Result<(), String> {
    TAKES.remove(take_id.trim());
    Ok(())
}

#[command]
async fn voice_library_save_take(request: SaveTakeRequest) -> Result<VoiceMeta, String> {
    blocking(move || save_take(&app_data()?, &TAKES, &request, local_stt())).await
}

#[command]
async fn voice_library_update(request: UpdateVoiceRequest) -> Result<VoiceMeta, String> {
    blocking(move || update_voice(&app_data()?, &TAKES, &request, local_stt())).await
}

#[command]
async fn voice_library_delete(id: String) -> Result<(), String> {
    blocking(move || delete_voice(&app_data()?, &id)).await
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .setup(|app, _api| {
            let handle = app.clone();
            let _ = CHANGED.set(Box::new(move || {
                let _ = handle.emit(CHANGED_EVENT, ());
            }));
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            voice_library_list,
            voice_library_get,
            voice_library_record_start,
            voice_library_record_stop,
            voice_library_record_cancel,
            voice_library_import_clip,
            voice_library_read_take,
            voice_library_read_clip,
            voice_library_discard_take,
            voice_library_save_take,
            voice_library_update,
            voice_library_delete,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::voice_library::process::tests::{noise, vad as synthetic_vad, voice};
    use crate::services::voice_library::store::tests::Scratch;
    use crate::services::voice_library::transcribe::{TranscribeError, Transcript};
    use std::sync::atomic::AtomicBool;

    #[test]
    fn voice_ids_are_safe_folder_names() {
        for ok in ["alice", "voice-2", "a"] {
            assert!(valid_voice_id(ok), "{ok}");
        }
        for bad in ["", "-a", "a-", "A", "a/b", "..", "a b", &"x".repeat(65)] {
            assert!(!valid_voice_id(bad), "{bad}");
        }
        let app = Path::new("C:/app");
        assert_eq!(voice_dir(app, "alice").unwrap(), app.join("voices").join("alice"));
        assert!(voice_dir(app, "../x").is_err());
    }

    /// Every command of `generate_handler!` is listed for the ACL (build.rs reads the file).
    #[test]
    fn the_command_list_matches_the_handler() {
        let listed: Vec<&str> = include_str!("../../../plugin-commands/voice-library.txt")
            .lines()
            .map(|l| l.split('#').next().unwrap_or("").trim())
            .filter(|l| !l.is_empty())
            .collect();
        let handled = [
            "voice_library_list",
            "voice_library_get",
            "voice_library_record_start",
            "voice_library_record_stop",
            "voice_library_record_cancel",
            "voice_library_import_clip",
            "voice_library_read_take",
            "voice_library_read_clip",
            "voice_library_discard_take",
            "voice_library_save_take",
            "voice_library_update",
            "voice_library_delete",
        ];
        assert_eq!(listed, handled);
    }

    #[test]
    fn wire_payloads_are_camel_case() {
        let v = serde_json::to_value(VoiceMeta { created_unix_ms: 1, ..Default::default() }).unwrap();
        assert!(v.get("createdUnixMs").is_some() && v.get("cachedBackends").is_some());
        assert!(v.get("transcriptError").is_some() && v.get("transcribing").is_some());
        let r: RecordRequest = serde_json::from_value(serde_json::json!({ "maxSeconds": 8 })).unwrap();
        assert_eq!(r.max_seconds, 8.0);
        let s: SaveTakeRequest = serde_json::from_value(serde_json::json!({ "takeId": "t", "name": "n" })).unwrap();
        assert!(s.transcribe, "transcribing is the default");
        let p = serde_json::to_value(RecordProgress { stopped: true, stop_reason: Some("max_length".into()), ..Default::default() }).unwrap();
        assert_eq!(p["stopReason"], "max_length");
        let u: UpdateVoiceRequest = serde_json::from_value(serde_json::json!({ "id": "a", "takeId": "take-0", "transcribe": true })).unwrap();
        assert_eq!((u.take_id.as_deref(), u.transcribe), (Some("take-0"), Some(true)));
        let t = serde_json::to_value(VoiceTake { sample_rate: 48_000, ..Default::default() }).unwrap();
        assert_eq!(t["sampleRate"], 48_000);
    }

    #[test]
    fn languages_and_record_devices() {
        assert_eq!(stored_language(None), Ok(None));
        assert_eq!(stored_language(Some("auto")), Ok(None));
        assert_eq!(stored_language(Some(" ")), Ok(None));
        assert_eq!(stored_language(Some("ja-JP")), Ok(Some("ja".into())));
        assert!(stored_language(Some("not a language")).is_err());
        let s = Scratch::new("device");
        assert_eq!(record_device(" USB Mic ", Some(&s.0)), "USB Mic");
        assert_eq!(record_device("", Some(&s.0)), "", "no voice-changer settings: the default device");
        let vc = s.0.join("voice-changer");
        std::fs::create_dir_all(&vc).unwrap();
        std::fs::write(vc.join("settings.json"), br#"{"version":1,"params":{"input_device":" Yeti "}}"#).unwrap();
        assert_eq!(record_device("", Some(&s.0)), "Yeti");
        assert_eq!(record_device("", None), "");
    }

    fn take_audio(rate: u32, speech: f64) -> Vec<f32> {
        let r = f64::from(rate);
        let mut x = vec![0.0f32; r as usize];
        x.extend(voice(rate, speech, 0.1));
        x.extend(vec![0.0f32; r as usize]);
        for (v, n) in x.iter_mut().zip(noise(((speech + 2.0) * r) as usize + 8, 0.0002, 11)) {
            *v += n;
        }
        x
    }

    fn fake_stt(text: &'static str) -> Transcriber {
        Arc::new(move |_: &[f32], language: Option<&str>, _: &AtomicBool| {
            Ok(Transcript { text: text.into(), language: language.map(str::to_string).or(Some("en".into())), model: "fake".into() })
        })
    }

    fn wait_transcribed(app: &Path, id: &str) -> VoiceMeta {
        for _ in 0..200 {
            let m = store::get(app, id).unwrap();
            if !m.transcribing {
                return m;
            }
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        panic!("the transcript job did not finish");
    }

    /// The acceptance path: a clean take is trimmed, stored and listed (what both pickers read),
    /// with its transcript arriving in the background.
    #[test]
    fn a_clean_take_is_trimmed_saved_listed_and_transcribed() {
        let s = Scratch::new("flow");
        let held = TakeStore::new();
        let take = hold(&held, make_take(take_audio(48_000, 5.0), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        assert!(take.quality.ok, "{:?}", take.quality);
        assert!(take.quality.trimmed_ms >= 1_400, "silence was trimmed: {:?}", take.quality);
        assert_eq!(take.sample_rate, 48_000);
        // Preview bytes.
        let wav = held.wav(&take.take_id).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
        // A nameless save is refused and keeps the take.
        let nameless = SaveTakeRequest { take_id: take.take_id.clone(), name: "  ".into(), ..Default::default() };
        assert!(save_take(&s.0, &held, &nameless, fake_stt("x")).is_err());
        let req = SaveTakeRequest { take_id: take.take_id.clone(), name: "  My  voice ".into(), language: Some("en-US".into()), ..Default::default() };
        let saved = save_take(&s.0, &held, &req, fake_stt("hello world")).unwrap();
        assert_eq!((saved.name.as_str(), saved.source.as_str(), saved.language.as_deref()), ("My voice", "recorded", Some("en")));
        assert!(saved.id.starts_with("my-voice-"));
        assert!((saved.lufs - process::TARGET_LUFS).abs() < 0.2, "{}", saved.lufs);
        assert!(saved.peak_dbfs <= process::PEAK_CEILING_DBFS + 0.01);
        assert!(held.get(&take.take_id).is_err(), "a saved take is released");
        // Listed for every picker, with its clip readable by the voice changer.
        let listed = list_voices(&s.0).unwrap();
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].id, saved.id);
        let clip = reference_clip(&s.0, &saved.id).unwrap();
        assert_eq!(clip.sha256, saved.sha256);
        let mono = clip.load_mono(16_000).unwrap();
        assert!((mono.len() as u64 * 1000 / 16_000).abs_diff(saved.duration_ms) <= 2);
        // The transcript arrives.
        let done = wait_transcribed(&s.0, &saved.id);
        assert_eq!(done.transcript.as_deref(), Some("hello world"));
    }

    /// The same path from the microphone: a take recorded through the capture hub by the app's
    /// recorder (its session thread, consumer options and live speech count) stops at its length,
    /// then is trimmed, saved and listed.
    #[test]
    fn a_take_recorded_through_the_capture_hub_is_trimmed_saved_and_listed() {
        use crate::services::audio::hub::SourceSpec;
        use crate::services::voice_library::recorder::tests::OneMic;
        static R: recorder::Recorder = recorder::Recorder::new();
        let s = Scratch::new("hub-flow");
        let mic = OneMic::default();
        let hub = mic.hub();
        let consumer = hub.attach(SourceSpec::DefaultInput, recorder::consumer_options()).unwrap();
        // 1 s of room noise, 5 s of voice, 1 s of room noise.
        let audio = take_audio(48_000, 5.0);
        let seconds = audio.len() as f32 / 48_000.0;
        let (tx, rx) = std::sync::mpsc::channel::<RecordProgress>();
        let recorded_ms = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let sink: recorder::ProgressSink = {
            let recorded_ms = Arc::clone(&recorded_ms);
            Arc::new(move |p: &RecordProgress| {
                recorded_ms.store(p.elapsed_ms, std::sync::atomic::Ordering::Release);
                let _ = tx.send(p.clone());
            })
        };
        let live_vad: recorder::VadFactory = Box::new(|| -> Option<Box<dyn process::Vad + Send>> { Some(Box::new(synthetic_vad())) });
        let (epoch, _) = R.begin();
        R.start(epoch, Box::new(recorder::HubSource(consumer)), seconds, sink, live_vad).unwrap();
        mic.play(&audio, &recorded_ms);
        let last = loop {
            let p = rx.recv_timeout(std::time::Duration::from_secs(20)).expect("the recorder reports progress");
            if p.stopped {
                break p;
            }
        };
        assert_eq!(last.stop_reason.as_deref(), Some(recorder::STOP_MAX_LENGTH), "{last:?}");
        assert!((4_500..=5_600).contains(&last.voiced_ms), "the live speech count heard the voice: {last:?}");
        let captured = R.end().expect("the session waits for record_stop").finish();
        assert_eq!((captured.rate, captured.mono.len()), (48_000, audio.len()));

        let held = TakeStore::new();
        let take = hold(&held, make_take(captured.mono, captured.rate, "recorded", None, &mut synthetic_vad()).unwrap());
        assert!(take.quality.ok && take.quality.issues.is_empty(), "{:?}", take.quality);
        assert!((5_300..=5_700).contains(&take.quality.duration_ms), "speech plus its pads: {:?}", take.quality);
        let saved = save_take(&s.0, &held, &SaveTakeRequest { take_id: take.take_id, name: "Mic take".into(), transcribe: false, ..Default::default() }, fake_stt("x")).unwrap();
        let listed = list_voices(&s.0).unwrap();
        assert_eq!(listed.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), vec![saved.id.as_str()]);
        assert_eq!((listed[0].sample_rate, listed[0].duration_ms), (48_000, take.quality.duration_ms));
        assert!(reference_clip(&s.0, &saved.id).unwrap().load_mono(16_000).unwrap().len() > 5 * 16_000);
    }

    /// The voice changer's zero-shot families cache their embedding per library voice through the
    /// frozen `reference::cached_embedding`: computed once per clip, recomputed after a re-record.
    #[test]
    fn family_embeddings_are_cached_per_voice_and_clip() {
        use crate::services::neural_audio::vc::families::reference::{cached_embedding, ReferenceEmbedding, ReferenceTensor};
        let s = Scratch::new("embed");
        let held = TakeStore::new();
        let t = hold(&held, make_take(take_audio(48_000, 4.0), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        let v = save_take(&s.0, &held, &SaveTakeRequest { take_id: t.take_id, name: "Eve".into(), transcribe: false, ..Default::default() }, fake_stt("x")).unwrap();
        let computed = std::cell::Cell::new(0);
        let embed = |clip: &ReferenceClip| -> Result<ReferenceEmbedding, String> {
            computed.set(computed.get() + 1);
            Ok(ReferenceEmbedding { tensors: vec![ReferenceTensor::new("style", vec![1, 2], vec![clip.duration_ms as f32, 1.0])], ..Default::default() })
        };
        let clip = reference_clip(&s.0, &v.id).unwrap();
        let a = cached_embedding(Some(&s.0), &clip, "test-zs-1", embed).unwrap();
        let b = cached_embedding(Some(&s.0), &clip, "test-zs-1", embed).unwrap();
        assert_eq!(a, b);
        assert_eq!(computed.get(), 1, "the second load reads cache/test-zs-1.bin");
        assert_eq!(list_voices(&s.0).unwrap()[0].cached_backends, vec!["test-zs-1"]);
        // Re-recording the voice invalidates the cache (new clip, new sha256).
        let t2 = hold(&held, make_take(take_audio(48_000, 5.0), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        update_voice(&s.0, &held, &UpdateVoiceRequest { id: v.id.clone(), take_id: Some(t2.take_id), transcribe: Some(false), ..Default::default() }, fake_stt("x")).unwrap();
        let clip2 = reference_clip(&s.0, &v.id).unwrap();
        assert_ne!(clip2.sha256, clip.sha256);
        let c = cached_embedding(Some(&s.0), &clip2, "test-zs-1", embed).unwrap();
        assert_eq!(computed.get(), 2);
        assert_eq!(c.clip_sha256, clip2.sha256);
    }

    #[test]
    fn unusable_takes_are_never_saved() {
        let s = Scratch::new("unusable");
        let held = TakeStore::new();
        let silent = hold(&held, make_take(noise(3 * 48_000, 0.0002, 5), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        assert!(!silent.quality.ok && silent.quality.issues == vec!["no_speech".to_string()]);
        let req = SaveTakeRequest { take_id: silent.take_id.clone(), name: "x".into(), ..Default::default() };
        let err = save_take(&s.0, &held, &req, fake_stt("x")).unwrap_err();
        assert!(err.contains("no_speech"), "{err}");
        let short = hold(&held, make_take(take_audio(48_000, 1.2), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        let req = SaveTakeRequest { take_id: short.take_id.clone(), name: "x".into(), ..Default::default() };
        assert!(save_take(&s.0, &held, &req, fake_stt("x")).unwrap_err().contains("too_short"));
        assert!(list_voices(&s.0).unwrap().is_empty());
        assert!(save_take(&s.0, &held, &SaveTakeRequest { take_id: "take-000000000000".into(), name: "x".into(), ..Default::default() }, fake_stt("x")).is_err());
    }

    /// A take is stored once: a second save of it (a double Enter while the first one writes, a
    /// second Replace click) finds it gone instead of storing a second voice, and a save that fails
    /// keeps the take for another try.
    #[test]
    fn a_take_is_saved_once_and_kept_when_a_save_fails() {
        let s = Scratch::new("once");
        let held = TakeStore::new();
        let fresh = |seconds: f64| hold(&held, make_take(take_audio(48_000, seconds), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        let t = fresh(4.0);
        let req = SaveTakeRequest { take_id: t.take_id.clone(), name: "My voice".into(), transcribe: false, ..Default::default() };
        let results: Vec<Result<VoiceMeta, String>> = std::thread::scope(|sc| {
            let a = sc.spawn(|| save_take(&s.0, &held, &req, fake_stt("x")));
            let b = sc.spawn(|| save_take(&s.0, &held, &req, fake_stt("x")));
            vec![a.join().unwrap(), b.join().unwrap()]
        });
        assert_eq!(results.iter().filter(|r| r.is_ok()).count(), 1, "{results:?}");
        assert!(results.iter().any(|r| r.as_ref().is_err_and(|e| e.contains("gone"))), "{results:?}");
        assert_eq!(list_voices(&s.0).unwrap().len(), 1, "one voice, not two");

        // A save that fails (no library folder can be made) keeps the take; the next try works.
        let t2 = fresh(4.5);
        let req2 = SaveTakeRequest { take_id: t2.take_id.clone(), name: "Second".into(), transcribe: false, ..Default::default() };
        let blocked = s.0.join("not-a-folder");
        std::fs::write(&blocked, b"a file where the app data folder should be").unwrap();
        assert!(save_take(&blocked, &held, &req2, fake_stt("x")).is_err());
        assert!(held.get(&t2.take_id).is_ok(), "a failed save keeps the take");
        let second = save_take(&s.0, &held, &req2, fake_stt("x")).unwrap();

        // Replacing a clip twice with one take applies it once.
        let t3 = fresh(5.0);
        let replace = UpdateVoiceRequest { id: second.id.clone(), take_id: Some(t3.take_id.clone()), transcribe: Some(false), ..Default::default() };
        let first = update_voice(&s.0, &held, &replace, fake_stt("x")).unwrap();
        assert_ne!(first.sha256, second.sha256);
        assert!(update_voice(&s.0, &held, &replace, fake_stt("x")).unwrap_err().contains("gone"));
        assert_eq!(store::get(&s.0, &second.id).unwrap().sha256, first.sha256);
        // A replace that is refused keeps its take: an unknown voice, an unusable take.
        let t4 = fresh(4.0);
        let unknown = UpdateVoiceRequest { id: "gone-000000".into(), take_id: Some(t4.take_id.clone()), ..Default::default() };
        assert!(update_voice(&s.0, &held, &unknown, fake_stt("x")).is_err());
        assert!(held.get(&t4.take_id).is_ok());
        let silent = hold(&held, make_take(noise(3 * 48_000, 0.0002, 9), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        let unusable = UpdateVoiceRequest { id: second.id.clone(), take_id: Some(silent.take_id.clone()), ..Default::default() };
        assert!(update_voice(&s.0, &held, &unusable, fake_stt("x")).unwrap_err().contains("no_speech"));
        assert!(held.get(&silent.take_id).is_ok());
        assert_eq!(list_voices(&s.0).unwrap().len(), 2);
    }

    #[test]
    fn updates_rename_edit_and_re_record_a_voice_in_place() {
        let s = Scratch::new("update");
        let held = TakeStore::new();
        let first = hold(&held, make_take(take_audio(44_100, 4.0), 44_100, "recorded", None, &mut synthetic_vad()).unwrap());
        let saved = save_take(&s.0, &held, &SaveTakeRequest { take_id: first.take_id, name: "Ann".into(), transcribe: false, ..Default::default() }, fake_stt("x")).unwrap();
        assert!(!saved.transcribing && saved.transcript.is_none());
        write_cache(&s.0, &saved.id, "sigil-zs-1", b"embedding").unwrap();
        // Rename + transcript + language.
        let u = update_voice(
            &s.0,
            &held,
            &UpdateVoiceRequest { id: saved.id.clone(), name: Some("Anna".into()), transcript: Some(" typed text ".into()), language: Some("de".into()), ..Default::default() },
            fake_stt("x"),
        )
        .unwrap();
        assert_eq!((u.name.as_str(), u.transcript.as_deref(), u.language.as_deref()), ("Anna", Some("typed text"), Some("de")));
        assert_eq!(u.cached_backends, vec!["sigil-zs-1"]);
        assert!(update_voice(&s.0, &held, &UpdateVoiceRequest { id: saved.id.clone(), name: Some(" ".into()), ..Default::default() }, fake_stt("x")).is_err());
        assert!(update_voice(&s.0, &held, &UpdateVoiceRequest { id: saved.id.clone(), language: Some("??".into()), ..Default::default() }, fake_stt("x")).is_err());
        // Re-record: same id, new clip and hash, caches gone, transcript redone.
        let second = hold(&held, make_take(take_audio(48_000, 6.0), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        let r = update_voice(&s.0, &held, &UpdateVoiceRequest { id: saved.id.clone(), take_id: Some(second.take_id.clone()), ..Default::default() }, fake_stt("new words")).unwrap();
        assert_eq!(r.id, saved.id);
        assert_ne!(r.sha256, saved.sha256);
        assert!(r.cached_backends.is_empty());
        assert_eq!(r.sample_rate, 48_000);
        let done = wait_transcribed(&s.0, &saved.id);
        assert_eq!(done.transcript.as_deref(), Some("new words"));
        assert_eq!(done.language.as_deref(), Some("de"), "the user's language is kept");
        // Re-transcribe on request.
        update_voice(&s.0, &held, &UpdateVoiceRequest { id: saved.id.clone(), transcribe: Some(true), ..Default::default() }, fake_stt("again")).unwrap();
        assert_eq!(wait_transcribed(&s.0, &saved.id).transcript.as_deref(), Some("again"));
        // Clearing the transcript.
        let c = update_voice(&s.0, &held, &UpdateVoiceRequest { id: saved.id.clone(), transcript: Some(String::new()), language: Some("auto".into()), ..Default::default() }, fake_stt("x")).unwrap();
        assert_eq!((c.transcript, c.language), (None, None));
        // Delete.
        delete_voice(&s.0, &saved.id).unwrap();
        assert!(list_voices(&s.0).unwrap().is_empty());
        assert!(update_voice(&s.0, &held, &UpdateVoiceRequest { id: saved.id.clone(), ..Default::default() }, fake_stt("x")).is_err());
    }

    #[test]
    fn missing_stt_is_reported_on_the_voice() {
        let s = Scratch::new("nostt");
        let held = TakeStore::new();
        let t = hold(&held, make_take(take_audio(48_000, 4.0), 48_000, "recorded", None, &mut synthetic_vad()).unwrap());
        let none: Transcriber = Arc::new(|_: &[f32], _: Option<&str>, _: &AtomicBool| Err(TranscribeError::NoModel));
        let saved = save_take(&s.0, &held, &SaveTakeRequest { take_id: t.take_id, name: "Zed".into(), ..Default::default() }, none).unwrap();
        let done = wait_transcribed(&s.0, &saved.id);
        assert_eq!((done.transcript, done.transcript_error.as_deref()), (None, Some(transcribe::ERROR_NO_STT_MODEL)));
    }

    #[test]
    fn imports_go_through_the_same_pipeline() {
        let s = Scratch::new("import");
        let held = TakeStore::new();
        let wav = s.0.join("Studio Take.wav");
        crate::services::bench::signals::write_wav(&wav, &take_audio(22_050, 4.0), 22_050, 1).unwrap();
        let t = import_clip(&held, wav.to_str().unwrap(), &mut synthetic_vad()).unwrap();
        assert_eq!((t.source.as_str(), t.suggested_name.as_deref(), t.sample_rate), ("imported", Some("Studio Take"), 22_050));
        assert!(t.quality.ok, "{:?}", t.quality);
        assert!(import_clip(&held, "", &mut synthetic_vad()).is_err());
        assert!(import_clip(&held, s.0.join("nope.wav").to_str().unwrap(), &mut synthetic_vad()).is_err());
    }

    /// The real Silero VAD on the bundled speech fixture (SAPI, 16 kHz): the take is trimmed to
    /// the spoken words and usable once padded to the 3 s minimum with silence around it.
    #[test]
    fn silero_trims_real_speech() {
        let bytes: &[u8] = include_bytes!("../speech_core/testdata/speech_16k.wav");
        let (speech, rate) = process::decode_wav_bytes(bytes).unwrap();
        let mut x = vec![0.0f32; 2 * rate as usize];
        x.extend(&speech);
        x.extend(&speech);
        x.extend(vec![0.0f32; 2 * rate as usize]);
        let mut vad = process::silero().unwrap();
        let p = process::process_take(x.clone(), rate, &mut vad).unwrap();
        assert!(p.quality.ok, "{:?}", p.quality);
        let spoken_ms = (speech.len() as u64 * 2 * 1000) / u64::from(rate);
        assert!(p.quality.duration_ms <= spoken_ms + 600, "{:?} vs {spoken_ms} ms of speech", p.quality);
        assert!(p.quality.trimmed_ms >= 3_000, "the 4 s of silence went: {:?}", p.quality);
        assert!(p.quality.lufs_after <= process::TARGET_LUFS + 0.1 && p.quality.lufs_after > -30.0, "{:?}", p.quality);
    }
}
