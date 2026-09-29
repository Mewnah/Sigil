//! Voice-changer settings persistence (owned by P0-A; scaffolded by P0-S0).
//!
//! Plan decision "Voice-changer settings persistence": Rust is authoritative. Global settings
//! (mode, presets, input/output/monitor devices, EP, block size, latency budget, aux mic gate,
//! worker priority, consent) live in `<app_data>/voice-changer/settings.json`; per-voice settings
//! (pitch, index, protect, formant, F0 method, speaker) in `<slot dir>/slot.json`. The TS service
//! stays non-persisted (no `patchService`) and hydrates from `get_voice_changer_params`.
//!
//! Frozen by P0-S0 (implementers may add, not change): [`settings_path`], [`slot_settings_path`],
//! [`backup_path`], [`write_atomic`] and [`read_with_backup`]. The on-disk schema (a versioned
//! serde struct) and load/save wiring are P0-A's. Loading never overwrites a file it cannot
//! parse: it is kept aside as `<file>.corrupt-<unix-ms>` and loading falls back to `.bak`, then to
//! defaults.
//!
//! On-disk schema (version 1):
//! - `settings.json`: `{ "version": 1, "params": VoiceChangerParams, "outside_rvc"? }` — every IPC
//!   param except that `enabled` is always stored `false` (the voice changer never auto-starts
//!   from disk); `outside_rvc` (`{ "pitch_semitones", "formant_shift" }`) holds the other modes'
//!   pitch and formant while RVC mode (whose voices bring their own) is active.
//! - `slot.json`: `{ "version": 1, "pitch_semitones", "formant_shift", "index_rate", "protect",
//!   "pitch_extractor", "speaker_id" }`, every field optional.
//!
//! Loading is lenient per field: a field this build does not understand (a newer build's enum
//! value such as a new `mode`, or a changed type) is dropped and logged, and everything else is
//! kept. Only a file that is not a JSON object at all counts as corrupt. A main file that exists
//! but cannot be read (another process holds it, e.g. an antivirus scan at startup) is retried
//! briefly; if it stays unreadable the backup is used for the session without rewriting the main
//! file from it.
//!
//! Writes go through a [`DebouncedWriter`] (a slider drag is one write, not one per tick) on its
//! own thread; `flush` writes synchronously (app exit).

use super::params::VoiceChangerParams;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Condvar, Mutex, OnceLock, PoisonError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Folder under app data shared with `neural_audio::vc::bases::voice_changer_root`.
pub(crate) const SETTINGS_DIR: &str = "voice-changer";
/// Global voice-changer settings file name.
pub(crate) const SETTINGS_FILE: &str = "settings.json";
/// Per-voice settings file name inside a model slot directory.
pub(crate) const SLOT_SETTINGS_FILE: &str = "slot.json";
/// Current on-disk schema version of both files.
pub(crate) const SETTINGS_VERSION: u32 = 1;
/// Quiet period before a scheduled write happens.
const WRITE_DEBOUNCE: Duration = Duration::from_millis(400);
/// Waits before re-reading a settings file that exists but could not be read (a sharing
/// violation lasts a moment); about a quarter second in total.
const READ_RETRY_DELAYS: [Duration; 3] = [Duration::from_millis(20), Duration::from_millis(60), Duration::from_millis(180)];

/// `<app_data>/voice-changer/settings.json`.
pub(crate) fn settings_path(app_data: &Path) -> PathBuf {
    app_data.join(SETTINGS_DIR).join(SETTINGS_FILE)
}

/// `<slot_dir>/slot.json`.
pub(crate) fn slot_settings_path(slot_dir: &Path) -> PathBuf {
    slot_dir.join(SLOT_SETTINGS_FILE)
}

/// `<file>.bak`: the previous contents kept by [`write_atomic`].
pub(crate) fn backup_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".bak");
    path.with_file_name(name)
}

/// `<file>.tmp`: the staging file of [`write_atomic`].
fn temp_path(path: &Path) -> PathBuf {
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(".tmp");
    path.with_file_name(name)
}

/// Atomically replace `path` with `bytes`: write and fsync `<file>.tmp`, copy the current file (if
/// any) to `<file>.bak`, then rename the temp file over `path` (`MoveFileExW` with
/// `MOVEFILE_REPLACE_EXISTING` on Windows), so a crash leaves either the old or the new file, never
/// a truncated one. Creates the parent directory. A failed backup copy does not block the save.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let tmp = temp_path(path);
    {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    if path.is_file() {
        let _ = fs::copy(path, backup_path(path));
    }
    fs::rename(&tmp, path).inspect_err(|_| {
        let _ = fs::remove_file(&tmp);
    })
}

/// Read `path`, or its `.bak` when `path` is missing or empty. `Ok(None)` when neither has data.
/// Parsing (and keeping an unparsable file aside) is the caller's job.
#[allow(dead_code)] // Frozen P0-S0 API; the loaders below also need to tell a corrupt file apart.
pub(crate) fn read_with_backup(path: &Path) -> io::Result<Option<Vec<u8>>> {
    for candidate in [path.to_path_buf(), backup_path(path)] {
        match fs::read(&candidate) {
            Ok(bytes) if !bytes.is_empty() => return Ok(Some(bytes)),
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::NotFound => {}
            Err(e) => return Err(e),
        }
    }
    Ok(None)
}

fn unix_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

/// Rename an unparsable file to `<file>.corrupt-<unix-ms>` so nothing ever overwrites it.
fn keep_aside(path: &Path) -> Result<PathBuf, io::Error> {
    let mut name = path.file_name().map(|n| n.to_os_string()).unwrap_or_default();
    name.push(format!(".corrupt-{}", unix_ms()));
    let target = path.with_file_name(name);
    fs::rename(path, &target)?;
    Ok(target)
}

/// Pitch and formant of the non-RVC modes, kept aside while RVC mode is active.
#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq)]
pub(super) struct OutsideRvcVoice {
    pub(super) pitch_semitones: f32,
    pub(super) formant_shift: f32,
}

/// Global settings file (`settings.json`).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub(super) struct SettingsFile {
    #[serde(default)]
    pub(super) version: u32,
    #[serde(default)]
    pub(super) params: VoiceChangerParams,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub(super) outside_rvc: Option<OutsideRvcVoice>,
}

impl SettingsFile {
    /// The file contents for `params` (`enabled` is never persisted as on).
    pub(super) fn new(mut params: VoiceChangerParams) -> Self {
        params.enabled = false;
        Self {
            version: SETTINGS_VERSION,
            params,
            outside_rvc: None,
        }
    }
}

/// Parse `value` as `T`, dropping the top-level fields that do not parse on their own (every
/// field of `T` has a serde default, so the rest still makes a value). Returns the value and the
/// dropped field names. Fails only when `value` is not an object that way.
fn salvage<T: DeserializeOwned>(value: Value) -> Result<(T, Vec<String>), serde_json::Error> {
    let first_error = match serde_json::from_value::<T>(value.clone()) {
        Ok(v) => return Ok((v, Vec::new())),
        Err(e) => e,
    };
    let Value::Object(map) = value else {
        return Err(first_error);
    };
    let mut kept = Map::new();
    let mut dropped = Vec::new();
    for (key, v) in map {
        let mut one = Map::new();
        one.insert(key.clone(), v.clone());
        if serde_json::from_value::<T>(Value::Object(one)).is_ok() {
            kept.insert(key, v);
        } else {
            dropped.push(key);
        }
    }
    let value = serde_json::from_value::<T>(Value::Object(kept))?;
    Ok((value, dropped))
}

/// A settings document [`load_json`] can read leniently (see the module docs).
pub(super) trait SettingsDoc: Sized {
    /// The document, and the names of the fields dropped because they did not parse.
    fn from_value_lenient(value: Value) -> Result<(Self, Vec<String>), serde_json::Error>;
}

impl SettingsDoc for SlotSettingsFile {
    fn from_value_lenient(value: Value) -> Result<(Self, Vec<String>), serde_json::Error> {
        salvage(value)
    }
}

impl SettingsDoc for SettingsFile {
    /// `params` is salvaged field by field on its own, so one unknown value (a newer build's
    /// `mode`) costs that field only, not every setting.
    fn from_value_lenient(value: Value) -> Result<(Self, Vec<String>), serde_json::Error> {
        let Value::Object(mut map) = value else {
            return salvage(value);
        };
        let params = map.remove("params");
        let (mut file, mut dropped) = salvage::<SettingsFile>(Value::Object(map))?;
        match params.map(salvage::<VoiceChangerParams>) {
            None => {}
            Some(Ok((params, bad))) => {
                file.params = params;
                dropped.extend(bad.into_iter().map(|k| format!("params.{k}")));
            }
            Some(Err(_)) => dropped.push("params".to_string()),
        }
        Ok((file, dropped))
    }
}

/// Per-voice settings file (`slot.json`); absent fields keep the current value when applied.
#[derive(Serialize, Deserialize, Debug, Clone, Default, PartialEq)]
pub(super) struct SlotSettingsFile {
    #[serde(default)]
    pub(super) version: u32,
    #[serde(default)]
    pub(super) pitch_semitones: Option<f32>,
    #[serde(default)]
    pub(super) formant_shift: Option<f32>,
    #[serde(default)]
    pub(super) index_rate: Option<f32>,
    #[serde(default)]
    pub(super) protect: Option<f32>,
    #[serde(default)]
    pub(super) pitch_extractor: Option<String>,
    #[serde(default)]
    pub(super) speaker_id: Option<i64>,
}

/// Where a loaded value came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum LoadSource {
    Main,
    Backup,
}

/// Result of [`load_json`]: the value (if any file parsed), its source, and what went wrong on the
/// way (for the log).
#[derive(Debug)]
pub(super) struct Loaded<T> {
    pub(super) value: Option<(T, LoadSource)>,
    pub(super) notes: Vec<String>,
    /// The main file exists but could not be read even after retrying (not missing, blank or
    /// corrupt). A value from the backup must then not be written back over it: the main file is
    /// probably newer.
    pub(super) main_unreadable: bool,
}

/// Parse settings bytes leniently (see [`SettingsDoc`]); `None` when they are not a JSON object.
pub(super) fn parse_doc<T: SettingsDoc>(bytes: &[u8]) -> Option<T> {
    serde_json::from_slice::<Value>(bytes)
        .and_then(T::from_value_lenient)
        .ok()
        .map(|(v, _)| v)
}

/// Read `path`, retrying briefly when it exists but cannot be read (a sharing violation).
fn read_with_retry(path: &Path) -> io::Result<Vec<u8>> {
    let mut delays = READ_RETRY_DELAYS.iter();
    loop {
        match fs::read(path) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => match delays.next() {
                Some(d) => thread::sleep(*d),
                None => return Err(e),
            },
            other => return other,
        }
    }
}

/// Parse `path`, falling back to its `.bak`. A file that exists but does not parse is kept aside
/// (renamed, never overwritten) before trying the next candidate; unreadable files are left alone.
/// Fields that do not parse are dropped (listed in the notes), not the whole file.
pub(super) fn load_json<T: SettingsDoc>(path: &Path) -> Loaded<T> {
    let mut notes = Vec::new();
    let mut main_unreadable = false;
    for (candidate, source) in [(path.to_path_buf(), LoadSource::Main), (backup_path(path), LoadSource::Backup)] {
        let bytes = match read_with_retry(&candidate) {
            Ok(b) if b.iter().all(|c| c.is_ascii_whitespace()) => {
                if !b.is_empty() {
                    notes.push(format!("{} is blank", candidate.display()));
                }
                continue;
            }
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => continue,
            Err(e) => {
                notes.push(format!("cannot read {}: {e}", candidate.display()));
                main_unreadable |= source == LoadSource::Main;
                continue;
            }
        };
        let parsed = serde_json::from_slice::<Value>(&bytes).and_then(T::from_value_lenient);
        match parsed {
            Ok((v, dropped)) => {
                if !dropped.is_empty() {
                    notes.push(format!(
                        "{}: ignored {} (not understood by this version of Sigil)",
                        candidate.display(),
                        dropped.join(", ")
                    ));
                }
                return Loaded {
                    value: Some((v, source)),
                    notes,
                    main_unreadable,
                };
            }
            Err(e) => {
                let kept = match keep_aside(&candidate) {
                    Ok(p) => format!("kept as {}", p.display()),
                    Err(re) => format!("could not keep it aside: {re}"),
                };
                notes.push(format!("{} does not parse ({e}); {kept}", candidate.display()));
            }
        }
    }
    Loaded {
        value: None,
        notes,
        main_unreadable,
    }
}

struct WriterQueue {
    pending: BTreeMap<PathBuf, Vec<u8>>,
    due: Option<Instant>,
    thread_started: bool,
}

struct WriterShared {
    queue: Mutex<WriterQueue>,
    wake: Condvar,
    /// Held while taking and writing a batch, so batches hit the disk in the order they were taken
    /// (the writer thread and `flush` never interleave two versions of one file).
    write_lock: Mutex<()>,
    delay: Duration,
}

/// Coalescing background writer: [`schedule`](Self::schedule) replaces the pending bytes of a
/// path and (re)starts a quiet-period timer; after it expires the writer thread writes every
/// pending file with [`write_atomic`]. [`flush`](Self::flush) writes immediately.
#[derive(Clone)]
pub(crate) struct DebouncedWriter {
    shared: Arc<WriterShared>,
}

impl Default for DebouncedWriter {
    fn default() -> Self {
        Self::new(WRITE_DEBOUNCE)
    }
}

impl DebouncedWriter {
    pub(crate) fn new(delay: Duration) -> Self {
        Self {
            shared: Arc::new(WriterShared {
                queue: Mutex::new(WriterQueue {
                    pending: BTreeMap::new(),
                    due: None,
                    thread_started: false,
                }),
                wake: Condvar::new(),
                write_lock: Mutex::new(()),
                delay,
            }),
        }
    }

    /// Queue `bytes` for `path` (replacing an older pending version) and restart the quiet period.
    pub(crate) fn schedule(&self, path: PathBuf, bytes: Vec<u8>) {
        let mut q = self.shared.queue.lock().unwrap_or_else(PoisonError::into_inner);
        q.pending.insert(path, bytes);
        q.due = Some(Instant::now() + self.shared.delay);
        if !q.thread_started {
            let shared = self.shared.clone();
            let spawned = thread::Builder::new()
                .name("sigil-vc-settings".into())
                .spawn(move || writer_thread(shared));
            match spawned {
                Ok(_) => q.thread_started = true,
                Err(e) => tracing::warn!("voice changer: settings writer thread failed to start ({e}); writing on flush only"),
            }
        }
        drop(q);
        self.shared.wake.notify_one();
    }

    /// Write everything pending now, on the calling thread.
    pub(crate) fn flush(&self) {
        write_batch(&self.shared);
    }

    /// Run `f` with writes paused, passing the bytes still waiting to be written to `path` (if
    /// any). A file saved moments ago may not be on disk yet; with writes paused, what `f` reads
    /// from disk otherwise is not about to be replaced by a batch already under way.
    pub(crate) fn with_latest<T>(&self, path: &Path, f: impl FnOnce(Option<&[u8]>) -> T) -> T {
        let _order = self.shared.write_lock.lock().unwrap_or_else(PoisonError::into_inner);
        let pending = self
            .shared
            .queue
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .pending
            .get(path)
            .cloned();
        f(pending.as_deref())
    }

    #[cfg(test)]
    pub(super) fn pending_len(&self) -> usize {
        self.shared.queue.lock().unwrap_or_else(PoisonError::into_inner).pending.len()
    }
}

fn write_batch(shared: &WriterShared) {
    let _order = shared.write_lock.lock().unwrap_or_else(PoisonError::into_inner);
    let batch = {
        let mut q = shared.queue.lock().unwrap_or_else(PoisonError::into_inner);
        q.due = None;
        std::mem::take(&mut q.pending)
    };
    for (path, bytes) in batch {
        if let Err(e) = write_atomic(&path, &bytes) {
            tracing::warn!("voice changer: could not save {}: {e}", path.display());
        }
    }
}

fn writer_thread(shared: Arc<WriterShared>) {
    let mut q = shared.queue.lock().unwrap_or_else(PoisonError::into_inner);
    loop {
        match q.due {
            None => {
                q = shared.wake.wait(q).unwrap_or_else(PoisonError::into_inner);
            }
            Some(due) => {
                let now = Instant::now();
                if now < due {
                    q = shared
                        .wake
                        .wait_timeout(q, due - now)
                        .unwrap_or_else(PoisonError::into_inner)
                        .0;
                } else {
                    drop(q);
                    write_batch(&shared);
                    q = shared.queue.lock().unwrap_or_else(PoisonError::into_inner);
                }
            }
        }
    }
}

/// Where the global settings live (set once when the plugin starts) and the writer for both files.
#[derive(Default)]
pub(crate) struct SettingsStore {
    settings_path: OnceLock<PathBuf>,
    pub(crate) writer: DebouncedWriter,
}

impl SettingsStore {
    /// Enable saving of `settings.json` under `app_data`. Only the first call has an effect.
    pub(crate) fn attach(&self, app_data: &Path) {
        let _ = self.settings_path.set(settings_path(app_data));
    }

    pub(crate) fn settings_path(&self) -> Option<&Path> {
        self.settings_path.get().map(PathBuf::as_path)
    }
}

/// Serialize `value` as pretty JSON (settings files are meant to be readable in a bug report).
pub(super) fn to_json_bytes<T: Serialize>(value: &T) -> Option<Vec<u8>> {
    match serde_json::to_vec_pretty(value) {
        Ok(mut b) => {
            b.push(b'\n');
            Some(b)
        }
        Err(e) => {
            tracing::warn!("voice changer: settings do not serialize: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDir(PathBuf);

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp_dir(tag: &str) -> TempDir {
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        TempDir(std::env::temp_dir().join(format!("sigil-vc-persist-{tag}-{}-{nanos}", std::process::id())))
    }

    #[test]
    fn settings_paths_are_stable() {
        let app = Path::new("app");
        assert_eq!(settings_path(app), Path::new("app").join("voice-changer").join("settings.json"));
        assert_eq!(slot_settings_path(Path::new("slot")), Path::new("slot").join("slot.json"));
        assert_eq!(backup_path(Path::new("a/settings.json")), Path::new("a/settings.json.bak"));
    }

    #[test]
    fn write_atomic_keeps_previous_as_backup_and_reads_fall_back() {
        let dir = temp_dir("atomic");
        let file = settings_path(&dir.0);
        assert_eq!(read_with_backup(&file).unwrap(), None);

        write_atomic(&file, b"one").unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"one");
        assert!(!backup_path(&file).exists());

        write_atomic(&file, b"two").unwrap();
        assert_eq!(std::fs::read(&file).unwrap(), b"two");
        assert_eq!(std::fs::read(backup_path(&file)).unwrap(), b"one");

        // A truncated main file falls back to the backup.
        std::fs::write(&file, b"").unwrap();
        assert_eq!(read_with_backup(&file).unwrap().as_deref(), Some(&b"one"[..]));
        // No staging file is left behind.
        assert!(!file.with_file_name("settings.json.tmp").exists());
    }

    /// Settings survive a save/load cycle unchanged (except `enabled`, which never persists on).
    #[test]
    fn settings_file_round_trips() {
        let dir = temp_dir("roundtrip");
        let file = settings_path(&dir.0);
        let mut params = VoiceChangerParams {
            enabled: true,
            pitch_semitones: 4.5,
            output_device: "Speakers (Test)".into(),
            monitor_output_device: "Headphones".into(),
            chunk_frames: 1024,
            rvc_slot_dir: "C:\\voices\\alto".into(),
            pitch_extractor: "rmvpe".into(),
            rvc_execution_provider: "cuda".into(),
            ..VoiceChangerParams::default()
        };
        params.latency_budget_ms = Some(450.0);
        params.aux_mic_gate = Some("duck".into());
        params.worker_priority = Some("protect_voice".into());
        params.input_device = Some("Mic (USB)".into());
        params.rvc_consent_acknowledged = Some(true);
        let bytes = to_json_bytes(&SettingsFile::new(params.clone())).unwrap();
        write_atomic(&file, &bytes).unwrap();

        let loaded = load_json::<SettingsFile>(&file);
        let (got, source) = loaded.value.unwrap();
        assert_eq!(source, LoadSource::Main);
        assert!(loaded.notes.is_empty(), "{:?}", loaded.notes);
        assert_eq!(got.version, SETTINGS_VERSION);
        params.enabled = false;
        assert_eq!(got.params, params);
    }

    /// A corrupt main file is kept aside (never overwritten) and the backup is used; a corrupt
    /// backup too leaves defaults, with both files kept.
    #[test]
    fn corrupt_settings_are_kept_and_the_backup_is_used() {
        let dir = temp_dir("corrupt");
        let file = settings_path(&dir.0);
        let good = to_json_bytes(&SettingsFile::new(VoiceChangerParams {
            pitch_semitones: -3.0,
            ..VoiceChangerParams::default()
        }))
        .unwrap();
        write_atomic(&file, &good).unwrap();
        write_atomic(&file, b"{\"version\":1,\"params\":{\"pitch_semit").unwrap(); // torn write

        let loaded = load_json::<SettingsFile>(&file);
        let (got, source) = loaded.value.unwrap();
        assert_eq!(source, LoadSource::Backup);
        assert_eq!(got.params.pitch_semitones, -3.0);
        assert_eq!(loaded.notes.len(), 1, "{:?}", loaded.notes);
        let aside: Vec<_> = std::fs::read_dir(file.parent().unwrap())
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("settings.json.corrupt-"))
            .collect();
        assert_eq!(aside.len(), 1, "{aside:?}");
        assert!(!file.exists(), "the corrupt main file was moved, not rewritten");

        // Backup corrupt as well: nothing loads, and nothing is deleted.
        std::fs::write(backup_path(&file), b"not json").unwrap();
        let loaded = load_json::<SettingsFile>(&file);
        assert!(loaded.value.is_none());
        let kept = std::fs::read_dir(file.parent().unwrap()).unwrap().count();
        assert_eq!(kept, 2, "both corrupt files are kept aside");
    }

    /// Older / partial files parse with defaults for what is missing; unknown fields are ignored.
    #[test]
    fn partial_and_future_files_parse() {
        let dir = temp_dir("partial");
        let file = settings_path(&dir.0);
        write_atomic(&file, br#"{"version":7,"params":{"chunk_frames":2048,"some_future_field":[1,2]},"extra":true}"#).unwrap();
        let (got, _) = load_json::<SettingsFile>(&file).value.unwrap();
        assert_eq!(got.version, 7);
        assert_eq!(got.params.chunk_frames, 2048);
        assert_eq!(got.params.pitch_extractor, "fcpe");
        assert_eq!(got.params.latency_budget_ms, None);

        let slot = dir.0.join("slot");
        write_atomic(&slot_settings_path(&slot), br#"{"version":1,"protect":0.2}"#).unwrap();
        let (s, _) = load_json::<SlotSettingsFile>(&slot_settings_path(&slot)).value.unwrap();
        assert_eq!(s.protect, Some(0.2));
        assert_eq!(s.pitch_semitones, None);
    }

    /// A settings file from a newer build keeps everything this build understands: an unknown
    /// `mode` value (or any field of an unexpected type) costs that field only, and nothing is
    /// set aside as corrupt.
    #[test]
    fn unknown_values_cost_only_their_field() {
        let dir = temp_dir("lenient");
        let file = settings_path(&dir.0);
        write_atomic(
            &file,
            br#"{"version":2,"params":{"mode":"engine","output_device":"CABLE Input","chunk_frames":"big",
                "latency_budget_ms":450,"input_device":"Mic (USB)","pitch_extractor":"swiftf0"},
                "outside_rvc":{"pitch_semitones":3,"formant_shift":0.1},"future":{"x":1}}"#,
        )
        .unwrap();
        let loaded = load_json::<SettingsFile>(&file);
        let (got, source) = loaded.value.unwrap();
        assert_eq!(source, LoadSource::Main);
        assert_eq!(got.version, 2);
        assert_eq!(got.params.mode, VoiceChangerParams::default().mode, "unknown mode -> default");
        assert_eq!(got.params.chunk_frames, VoiceChangerParams::default().chunk_frames);
        assert_eq!(got.params.output_device, "CABLE Input");
        assert_eq!(got.params.latency_budget_ms, Some(450.0));
        assert_eq!(got.params.input_device.as_deref(), Some("Mic (USB)"));
        assert_eq!(
            got.outside_rvc,
            Some(OutsideRvcVoice {
                pitch_semitones: 3.0,
                formant_shift: 0.1
            })
        );
        assert_eq!(loaded.notes.len(), 1, "{:?}", loaded.notes);
        assert!(loaded.notes[0].contains("params.mode") && loaded.notes[0].contains("params.chunk_frames"));
        assert!(!loaded.main_unreadable);
        assert!(file.exists(), "nothing was set aside");
        // A per-voice file with one bad field keeps the others.
        let slot = dir.0.join("slot");
        write_atomic(&slot_settings_path(&slot), br#"{"version":1,"protect":0.2,"pitch_semitones":"high"}"#).unwrap();
        let loaded = load_json::<SlotSettingsFile>(&slot_settings_path(&slot));
        let (s, _) = loaded.value.unwrap();
        assert_eq!((s.protect, s.pitch_semitones), (Some(0.2), None));
        // Not an object at all: corrupt (kept aside), as before.
        write_atomic(&slot_settings_path(&slot), b"[1,2,3]").unwrap();
        assert!(load_json::<SlotSettingsFile>(&slot_settings_path(&slot)).value.is_some(), "the backup");
    }

    /// A main file that exists but cannot be read (here: a folder in its place) is retried, then
    /// the backup is used and reported, so the caller does not write the backup over it.
    #[test]
    fn unreadable_main_file_uses_the_backup_without_repair() {
        let dir = temp_dir("unreadable");
        let file = settings_path(&dir.0);
        let good = to_json_bytes(&SettingsFile::new(VoiceChangerParams {
            pitch_semitones: 2.0,
            ..VoiceChangerParams::default()
        }))
        .unwrap();
        std::fs::create_dir_all(&file).unwrap();
        std::fs::write(backup_path(&file), &good).unwrap();
        let t0 = Instant::now();
        let loaded = load_json::<SettingsFile>(&file);
        assert!(t0.elapsed() >= Duration::from_millis(200), "retried before giving up");
        let (got, source) = loaded.value.unwrap();
        assert_eq!((source, got.params.pitch_semitones), (LoadSource::Backup, 2.0));
        assert!(loaded.main_unreadable);
        assert!(file.is_dir(), "left alone");
        // A missing main file is not "unreadable".
        std::fs::remove_dir(&file).unwrap();
        let loaded = load_json::<SettingsFile>(&file);
        assert_eq!(loaded.value.unwrap().1, LoadSource::Backup);
        assert!(!loaded.main_unreadable);
    }

    /// Scheduled writes coalesce per path (last version wins) and `flush` writes them at once.
    #[test]
    fn debounced_writer_coalesces_and_flushes() {
        let dir = temp_dir("writer");
        let a = dir.0.join("a.json");
        let b = dir.0.join("sub").join("b.json");
        // A long quiet period, so only `flush` writes here.
        let w = DebouncedWriter::new(Duration::from_secs(3_600));
        w.schedule(a.clone(), b"1".to_vec());
        w.schedule(a.clone(), b"2".to_vec());
        w.schedule(b.clone(), b"x".to_vec());
        assert_eq!(w.pending_len(), 2);
        assert!(!a.exists());
        w.flush();
        assert_eq!(w.pending_len(), 0);
        assert_eq!(std::fs::read(&a).unwrap(), b"2");
        assert_eq!(std::fs::read(&b).unwrap(), b"x");
        assert!(!backup_path(&a).exists(), "one write per path, not one per schedule");

        // A reader sees what is pending before it reaches the disk, and the disk otherwise.
        w.schedule(a.clone(), b"3".to_vec());
        assert_eq!(w.with_latest(&a, |p| p.map(<[u8]>::to_vec)), Some(b"3".to_vec()));
        assert_eq!(w.with_latest(&b, |p| p.map(<[u8]>::to_vec)), None);
        w.flush();
        assert!(w.with_latest(&a, |p| p.is_none()));

        // The writer thread writes on its own once the quiet period ends.
        let w = DebouncedWriter::new(Duration::from_millis(20));
        let c = dir.0.join("c.json");
        w.schedule(c.clone(), b"late".to_vec());
        let deadline = Instant::now() + Duration::from_secs(10);
        while !c.exists() && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(std::fs::read(&c).unwrap(), b"late");
    }
}
