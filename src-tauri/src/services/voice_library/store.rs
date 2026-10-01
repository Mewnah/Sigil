//! The library on disk: `app_data/voices/<id>/{ref.wav, meta.json, cache/<backend>.bin}`.
//!
//! - Every write of a library file is atomic (a hidden temp file in the same folder, then a rename
//!   over the target), and every change of the library is serialized by one process-wide lock,
//!   so a voice is never seen half written.
//! - A new voice is assembled in a hidden `voices/.new-<n>` folder and renamed into place; a deleted
//!   voice is first renamed to `voices/.trash-<n>` and then removed, so a delete that fails half
//!   way (a file held open) never leaves a voice without its clip. Hidden folders are never listed
//!   and stale ones are swept.
//! - `meta.json` that cannot be parsed does not lose the voice: it is listed with its id as the
//!   name and the clip's facts read from `ref.wav`.
//! - `cached_backends` is read from the `cache` folder, never trusted from `meta.json`; the
//!   `sha256` given to the voice changer is always hashed from `ref.wav` itself.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};

use super::{valid_voice_id, voice_dir, voices_dir, VoiceMeta, CACHE_DIR, CLIP_FILE, META_FILE};
use crate::services::neural_audio::vc::families::reference::{self, ReferenceClip};

const LOG: &str = "sigil::voice_library";
/// Longest display name (characters).
pub const MAX_NAME_CHARS: usize = 64;
/// Longest transcript kept (characters).
pub const MAX_TRANSCRIPT_CHARS: usize = 2_000;
/// Largest cache file read or written (bytes): the biggest embedding the codec allows plus room.
const MAX_CACHE_BYTES: u64 = (reference::MAX_EMBEDDING_VALUES as u64) * 4 + (2 << 20);
/// Hidden staging and trash folders older than this are swept.
const STALE_AFTER: Duration = Duration::from_secs(600);

static LIBRARY: Mutex<()> = Mutex::new(());
static COUNTER: AtomicU64 = AtomicU64::new(0);

/// The library's write lock.
pub(super) fn lock() -> MutexGuard<'static, ()> {
    LIBRARY.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(super) fn now_unix_ms() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// A short unique token (hex) for ids and temp names.
pub(super) fn token(len: usize) -> String {
    let mut h = Sha256::new();
    h.update(now_unix_ms().to_le_bytes());
    h.update(COUNTER.fetch_add(1, Ordering::Relaxed).to_le_bytes());
    h.update(std::process::id().to_le_bytes());
    if let Ok(d) = SystemTime::now().duration_since(UNIX_EPOCH) {
        h.update(d.subsec_nanos().to_le_bytes());
    }
    h.finalize().iter().map(|b| format!("{b:02x}")).collect::<String>()[..len.min(64)].to_string()
}

pub(super) fn clip_path(dir: &Path) -> PathBuf {
    dir.join(CLIP_FILE)
}

fn meta_path(dir: &Path) -> PathBuf {
    dir.join(META_FILE)
}

fn cache_path(dir: &Path, backend: &str) -> PathBuf {
    dir.join(CACHE_DIR).join(format!("{backend}.bin"))
}

/// Write `bytes` to `path` atomically (temp file in the same folder, flushed, renamed over).
pub(super) fn write_atomic(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let dir = path.parent().ok_or_else(|| format!("{} has no folder", path.display()))?;
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let tmp = dir.join(format!(".{name}.tmp-{}-{}", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed)));
    let result = (|| -> std::io::Result<()> {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        std::fs::rename(&tmp, path)
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_file(&tmp);
        return Err(format!("cannot write {}: {e}", path.display()));
    }
    Ok(())
}

/// A display name: trimmed, inner whitespace collapsed, control characters removed, at most
/// [`MAX_NAME_CHARS`]. `None` when nothing is left.
pub fn clean_name(name: &str) -> Option<String> {
    let collapsed = name
        .split_whitespace()
        .map(|w| w.chars().filter(|c| !c.is_control()).collect::<String>())
        .filter(|w| !w.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    let cut: String = collapsed.chars().take(MAX_NAME_CHARS).collect();
    let cut = cut.trim().to_string();
    (!cut.is_empty()).then_some(cut)
}

/// A transcript: trimmed, at most [`MAX_TRANSCRIPT_CHARS`]; `None` when empty.
pub fn clean_transcript(text: &str) -> Option<String> {
    let t: String = text.trim().chars().filter(|c| !c.is_control() || *c == '\n').take(MAX_TRANSCRIPT_CHARS).collect();
    let t = t.trim().to_string();
    (!t.is_empty()).then_some(t)
}

/// The id stem of a name: lower-case ASCII letters and digits, other runs become `-` (empty for a
/// name without any, e.g. Japanese).
pub fn slug(name: &str) -> String {
    let mut out = String::new();
    let mut dash = false;
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
            dash = false;
        } else if !out.is_empty() && !dash {
            out.push('-');
            dash = true;
        }
        if out.len() >= 40 {
            break;
        }
    }
    out.trim_matches('-').to_string()
}

/// A new, unused voice id for `name`: its [`slug`] (or `voice`) plus a random suffix, so an id is
/// never reused after a delete (a voice-changer slot that named a deleted voice must not silently
/// get another one).
pub(super) fn new_voice_id(voices: &Path, name: &str) -> String {
    let stem = match slug(name) {
        s if s.is_empty() => "voice".to_string(),
        s => s,
    };
    loop {
        let id = format!("{stem}-{}", token(6));
        if valid_voice_id(&id) && !voices.join(&id).exists() {
            return id;
        }
    }
}

/// Backends with a cache file, sorted.
fn cached_backends(dir: &Path) -> Vec<String> {
    let mut out: Vec<String> = std::fs::read_dir(dir.join(CACHE_DIR))
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            let stem = name.strip_suffix(".bin")?;
            (reference::valid_backend(stem) && e.path().is_file()).then(|| stem.to_string())
        })
        .collect();
    out.sort();
    out
}

/// The voice in `dir` (folder name `id`), with its runtime fields filled. Errors when the folder
/// holds no clip.
pub(super) fn read_voice(dir: &Path, id: &str) -> Result<VoiceMeta, String> {
    let clip = clip_path(dir);
    if !clip.is_file() {
        return Err(format!("voice {id:?} has no clip"));
    }
    let mut meta = match std::fs::read(meta_path(dir)).map_err(|e| e.to_string()).and_then(|b| {
        serde_json::from_slice::<VoiceMeta>(&b).map_err(|e| e.to_string())
    }) {
        Ok(m) => m,
        Err(e) => {
            tracing::warn!(target: LOG, "voice {id}: meta.json unreadable ({e}); listing it from its clip");
            let mut m = VoiceMeta { name: id.to_string(), source: "imported".into(), ..VoiceMeta::default() };
            if let Ok(r) = hound::WavReader::open(&clip) {
                let spec = r.spec();
                m.sample_rate = spec.sample_rate;
                m.duration_ms = u64::from(r.duration()) * 1000 / u64::from(spec.sample_rate.max(1));
            }
            m.created_unix_ms = std::fs::metadata(&clip)
                .and_then(|md| md.modified())
                .ok()
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0);
            m
        }
    };
    meta.id = id.to_string();
    if clean_name(&meta.name).is_none() {
        meta.name = id.to_string();
    }
    meta.cached_backends = cached_backends(dir);
    meta.transcribing = super::transcribe::is_pending(id);
    Ok(meta)
}

/// Persist `meta` (runtime fields cleared) as `dir/meta.json`.
pub(super) fn write_meta(dir: &Path, meta: &VoiceMeta) -> Result<(), String> {
    let mut stored = meta.clone();
    stored.cached_backends = Vec::new();
    stored.transcribing = false;
    let json = serde_json::to_vec_pretty(&stored).map_err(|e| e.to_string())?;
    write_atomic(&meta_path(dir), &json)
}

/// Remove hidden staging / trash folders left by an interrupted operation.
fn sweep(voices: &Path) {
    for e in std::fs::read_dir(voices).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let stale = e
            .metadata()
            .and_then(|m| m.modified())
            .ok()
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|age| age > STALE_AFTER);
        if name.starts_with(".trash-") || (name.starts_with(".new-") && stale) {
            if let Err(err) = std::fs::remove_dir_all(e.path()) {
                tracing::debug!(target: LOG, "cannot sweep {}: {err}", e.path().display());
            }
        }
    }
}

/// Every voice, sorted by name (case-insensitive), then id.
pub fn list(app_data: &Path) -> Result<Vec<VoiceMeta>, String> {
    let voices = voices_dir(app_data);
    let entries = match std::fs::read_dir(&voices) {
        Ok(e) => e,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("cannot read the voice library ({}): {e}", voices.display())),
    };
    let mut out = Vec::new();
    for e in entries.flatten() {
        let id = e.file_name().to_string_lossy().into_owned();
        if !valid_voice_id(&id) || !e.path().is_dir() {
            continue;
        }
        match read_voice(&e.path(), &id) {
            Ok(m) => out.push(m),
            Err(err) => tracing::debug!(target: LOG, "skipping {}: {err}", e.path().display()),
        }
    }
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()).then_with(|| a.id.cmp(&b.id)));
    {
        let _guard = lock();
        sweep(&voices);
    }
    Ok(out)
}

/// One voice.
pub fn get(app_data: &Path, id: &str) -> Result<VoiceMeta, String> {
    let dir = voice_dir(app_data, id)?;
    if !dir.is_dir() {
        return Err(format!("voice {id:?} is not in the library"));
    }
    read_voice(&dir, id)
}

/// The clip of voice `id` for the voice changer (see `voice_library::reference_clip`).
pub fn reference_clip(app_data: &Path, id: &str) -> Result<ReferenceClip, String> {
    let meta = get(app_data, id).map_err(|e| format!("reference voice {id:?}: {e}"))?;
    let dir = voice_dir(app_data, id)?;
    let mut clip = ReferenceClip::from_wav(&clip_path(&dir)).map_err(|e| format!("reference voice {id:?}: {e}"))?;
    clip.voice_id = id.to_string();
    clip.name = meta.name;
    clip.transcript = meta.transcript;
    clip.language = meta.language;
    Ok(clip)
}

/// `cache/<backend>.bin` of voice `id`, if it exists and is readable.
pub fn read_cache(app_data: &Path, id: &str, backend: &str) -> Option<Vec<u8>> {
    if !reference::valid_backend(backend) {
        return None;
    }
    let path = cache_path(&voice_dir(app_data, id).ok()?, backend);
    let len = std::fs::metadata(&path).ok()?.len();
    if len > MAX_CACHE_BYTES {
        tracing::debug!(target: LOG, "{} is too large to be a reference cache ({len} bytes)", path.display());
        return None;
    }
    std::fs::read(&path).ok()
}

/// Store `cache/<backend>.bin` of voice `id` atomically. The voice must exist (a cache is never
/// written for a deleted voice); returns whether the set of cached backends changed.
pub fn write_cache(app_data: &Path, id: &str, backend: &str, bytes: &[u8]) -> Result<bool, String> {
    if !reference::valid_backend(backend) {
        return Err(format!("invalid reference cache backend {backend:?}"));
    }
    if bytes.len() as u64 > MAX_CACHE_BYTES {
        return Err(format!("the reference cache is too large ({} bytes)", bytes.len()));
    }
    let dir = voice_dir(app_data, id)?;
    let _guard = lock();
    if !clip_path(&dir).is_file() {
        return Err(format!("voice {id:?} is not in the library"));
    }
    let path = cache_path(&dir, backend);
    let existed = path.is_file();
    std::fs::create_dir_all(dir.join(CACHE_DIR)).map_err(|e| format!("cannot create {}: {e}", dir.join(CACHE_DIR).display()))?;
    write_atomic(&path, bytes)?;
    Ok(!existed)
}

/// Create a voice from a clip (WAV bytes) and its metadata (`id` is assigned here). Returns it.
pub(super) fn create(app_data: &Path, wav: &[u8], mut meta: VoiceMeta) -> Result<VoiceMeta, String> {
    let voices = voices_dir(app_data);
    std::fs::create_dir_all(&voices).map_err(|e| format!("cannot create the voice library ({}): {e}", voices.display()))?;
    let _guard = lock();
    let id = new_voice_id(&voices, &meta.name);
    let staging = voices.join(format!(".new-{}", token(12)));
    let result = (|| -> Result<(), String> {
        std::fs::create_dir_all(&staging).map_err(|e| format!("cannot create {}: {e}", staging.display()))?;
        write_atomic(&clip_path(&staging), wav)?;
        meta.id = id.clone();
        meta.sha256 = sha256_bytes(wav);
        write_meta(&staging, &meta)?;
        std::fs::rename(&staging, voices.join(&id)).map_err(|e| format!("cannot add voice {id:?} to the library: {e}"))
    })();
    if let Err(e) = result {
        let _ = std::fs::remove_dir_all(&staging);
        return Err(e);
    }
    read_voice(&voices.join(&id), &id)
}

/// Change voice `id`'s metadata under the library lock (`change` may refuse). Returns the voice.
pub(super) fn update(app_data: &Path, id: &str, change: impl FnOnce(&mut VoiceMeta) -> Result<(), String>) -> Result<VoiceMeta, String> {
    let dir = voice_dir(app_data, id)?;
    let _guard = lock();
    if !dir.is_dir() {
        return Err(format!("voice {id:?} is not in the library"));
    }
    let mut meta = read_voice(&dir, id)?;
    change(&mut meta)?;
    write_meta(&dir, &meta)?;
    read_voice(&dir, id)
}

/// Replace voice `id`'s clip (WAV bytes) and its clip facts; every cached embedding is removed
/// (they belong to the old clip). `change` adjusts the rest of the metadata.
pub(super) fn replace_clip(
    app_data: &Path,
    id: &str,
    wav: &[u8],
    change: impl FnOnce(&mut VoiceMeta),
) -> Result<VoiceMeta, String> {
    let dir = voice_dir(app_data, id)?;
    let _guard = lock();
    if !dir.is_dir() {
        return Err(format!("voice {id:?} is not in the library"));
    }
    let mut meta = read_voice(&dir, id)?;
    // Caches first: a failure here leaves the old clip with its valid caches.
    let cache = dir.join(CACHE_DIR);
    if cache.exists() {
        std::fs::remove_dir_all(&cache).map_err(|e| format!("cannot clear the cached embeddings of voice {id:?}: {e}"))?;
    }
    write_atomic(&clip_path(&dir), wav)?;
    meta.sha256 = sha256_bytes(wav);
    change(&mut meta);
    write_meta(&dir, &meta)?;
    read_voice(&dir, id)
}

/// Delete voice `id` (its folder, clip, metadata and caches).
pub(super) fn delete(app_data: &Path, id: &str) -> Result<(), String> {
    let voices = voices_dir(app_data);
    let dir = voice_dir(app_data, id)?;
    let _guard = lock();
    if !dir.is_dir() {
        return Err(format!("voice {id:?} is not in the library"));
    }
    let trash = voices.join(format!(".trash-{}", token(12)));
    std::fs::rename(&dir, &trash).map_err(|e| format!("cannot delete voice {id:?} (is its clip open in another program?): {e}"))?;
    if let Err(e) = std::fs::remove_dir_all(&trash) {
        tracing::warn!(target: LOG, "voice {id} deleted; its files stay in {} until the next listing: {e}", trash.display());
    }
    Ok(())
}

/// sha256 (lower-case hex) of bytes (equals `reference::sha256_file` of the written file).
pub(super) fn sha256_bytes(bytes: &[u8]) -> String {
    Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A scratch app-data folder, removed on drop.
    pub(crate) struct Scratch(pub PathBuf);

    impl Scratch {
        pub(crate) fn new(tag: &str) -> Scratch {
            let dir = std::env::temp_dir().join(format!("sigil-vl-{tag}-{}-{}", std::process::id(), token(8)));
            std::fs::create_dir_all(&dir).unwrap();
            Scratch(dir)
        }
    }

    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    pub(crate) fn wav(seconds: f64) -> Vec<u8> {
        let clip = crate::services::voice_library::process::tests::voice(24_000, seconds, 0.3);
        crate::services::voice_library::process::encode_wav(&clip, 24_000).unwrap()
    }

    fn meta(name: &str) -> VoiceMeta {
        VoiceMeta {
            name: name.into(),
            created_unix_ms: now_unix_ms(),
            duration_ms: 4_000,
            sample_rate: 24_000,
            lufs: -20.0,
            snr_db: Some(42.0),
            peak_dbfs: -3.0,
            source: "recorded".into(),
            ..VoiceMeta::default()
        }
    }

    #[test]
    fn names_slugs_and_ids() {
        assert_eq!(clean_name("  Alice \t  Smith \n"), Some("Alice Smith".into()));
        assert_eq!(clean_name(" \u{7} "), None);
        assert_eq!(clean_name(&"x".repeat(100)).unwrap().chars().count(), MAX_NAME_CHARS);
        assert_eq!(clean_name("あかり"), Some("あかり".into()));
        assert_eq!(slug("Alice Smith!"), "alice-smith");
        assert_eq!(slug("--Ünïcode__voice--"), "n-code-voice");
        assert_eq!(slug("あかり"), "");
        let s = Scratch::new("ids");
        let a = new_voice_id(&s.0, "Alice Smith");
        let b = new_voice_id(&s.0, "Alice Smith");
        assert!(a.starts_with("alice-smith-") && valid_voice_id(&a) && a != b, "{a} {b}");
        let j = new_voice_id(&s.0, "あかり");
        assert!(j.starts_with("voice-") && valid_voice_id(&j), "{j}");
        let long = new_voice_id(&s.0, &"a".repeat(200));
        assert!(valid_voice_id(&long), "{long}");
        assert_eq!(clean_transcript("  hello\u{0}  world \n"), Some("hello  world".into()));
        assert_eq!(clean_transcript("   "), None);
    }

    #[test]
    fn create_list_update_and_delete() {
        let s = Scratch::new("crud");
        assert!(list(&s.0).unwrap().is_empty(), "a missing library is empty");
        let wav_b = wav(4.0);
        let b = create(&s.0, &wav_b, meta("bob")).unwrap();
        let a = create(&s.0, &wav(3.5), meta("Alice")).unwrap();
        assert!(b.id.starts_with("bob-") && a.id.starts_with("alice-"));
        assert_eq!(b.sha256, sha256_bytes(&wav_b));
        let names: Vec<String> = list(&s.0).unwrap().into_iter().map(|v| v.name).collect();
        assert_eq!(names, vec!["Alice", "bob"], "sorted by name, case-insensitive");
        // The clip handed to the voice changer.
        let clip = reference_clip(&s.0, &b.id).unwrap();
        assert_eq!((clip.voice_id.as_str(), clip.name.as_str(), clip.sample_rate), (b.id.as_str(), "bob", 24_000));
        assert_eq!(clip.sha256, b.sha256, "hashed from the file");
        assert_eq!(clip.duration_ms, 4_000);
        // Update.
        let u = update(&s.0, &b.id, |m| {
            m.name = "Bobby".into();
            m.transcript = Some("hi".into());
            Ok(())
        })
        .unwrap();
        assert_eq!((u.name.as_str(), u.transcript.as_deref()), ("Bobby", Some("hi")));
        assert_eq!(get(&s.0, &b.id).unwrap().name, "Bobby");
        assert!(update(&s.0, &b.id, |_| Err("no".into())).is_err());
        assert_eq!(get(&s.0, &b.id).unwrap().name, "Bobby", "a refused change writes nothing");
        // Delete.
        delete(&s.0, &a.id).unwrap();
        assert!(get(&s.0, &a.id).is_err());
        assert!(delete(&s.0, &a.id).is_err());
        assert_eq!(list(&s.0).unwrap().len(), 1);
        assert!(get(&s.0, "../x").is_err() && get(&s.0, "nope").is_err());
        // Nothing hidden is left behind.
        let hidden: Vec<String> = std::fs::read_dir(voices_dir(&s.0))
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with('.'))
            .collect();
        assert!(hidden.is_empty(), "{hidden:?}");
    }

    #[test]
    fn caches_are_per_backend_and_cleared_with_a_new_clip() {
        let s = Scratch::new("cache");
        let v = create(&s.0, &wav(4.0), meta("carol")).unwrap();
        assert_eq!(read_cache(&s.0, &v.id, "sigil-zs-1"), None);
        assert!(write_cache(&s.0, &v.id, "sigil-zs-1", b"abc").unwrap(), "a new backend");
        assert!(!write_cache(&s.0, &v.id, "sigil-zs-1", b"abcd").unwrap(), "replaced");
        write_cache(&s.0, &v.id, "meanvc2.v1", b"x").unwrap();
        assert_eq!(read_cache(&s.0, &v.id, "sigil-zs-1").as_deref(), Some(&b"abcd"[..]));
        assert_eq!(get(&s.0, &v.id).unwrap().cached_backends, vec!["meanvc2.v1", "sigil-zs-1"]);
        assert!(write_cache(&s.0, &v.id, "../evil", b"x").is_err());
        assert_eq!(read_cache(&s.0, &v.id, "../evil"), None);
        assert!(write_cache(&s.0, "ghost-voice", "sigil-zs-1", b"x").is_err(), "never for a missing voice");
        assert!(!voice_dir(&s.0, "ghost-voice").unwrap().exists());
        // A new clip invalidates every cache and changes the hash.
        let new_wav = wav(5.0);
        let r = replace_clip(&s.0, &v.id, &new_wav, |m| m.duration_ms = 5_000).unwrap();
        assert!(r.cached_backends.is_empty());
        assert_ne!(r.sha256, v.sha256);
        assert_eq!(reference_clip(&s.0, &v.id).unwrap().sha256, sha256_bytes(&new_wav));
        assert_eq!(r.duration_ms, 5_000);
    }

    #[test]
    fn a_damaged_meta_file_keeps_the_voice_and_incomplete_folders_are_skipped() {
        let s = Scratch::new("damaged");
        let v = create(&s.0, &wav(4.0), meta("dave")).unwrap();
        let dir = voice_dir(&s.0, &v.id).unwrap();
        std::fs::write(dir.join(META_FILE), b"{ not json").unwrap();
        let got = get(&s.0, &v.id).unwrap();
        assert_eq!((got.name.as_str(), got.sample_rate, got.duration_ms), (v.id.as_str(), 24_000, 4_000));
        // A folder without a clip, a hidden folder and an invalid name are not voices.
        std::fs::create_dir_all(voices_dir(&s.0).join("empty-voice")).unwrap();
        std::fs::create_dir_all(voices_dir(&s.0).join(".new-abc")).unwrap();
        std::fs::create_dir_all(voices_dir(&s.0).join("Bad Name")).unwrap();
        std::fs::create_dir_all(voices_dir(&s.0).join(".trash-old")).unwrap();
        let all = list(&s.0).unwrap();
        assert_eq!(all.len(), 1);
        assert!(!voices_dir(&s.0).join(".trash-old").exists(), "trash is swept");
        assert!(voices_dir(&s.0).join(".new-abc").exists(), "fresh staging is left alone");
    }

    #[test]
    fn atomic_writes_replace_and_leave_no_temp_files() {
        let s = Scratch::new("atomic");
        let p = s.0.join("f.json");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"two");
        let names: Vec<String> = std::fs::read_dir(&s.0).unwrap().flatten().map(|e| e.file_name().to_string_lossy().into_owned()).collect();
        assert_eq!(names, vec!["f.json"]);
        assert!(write_atomic(&s.0.join("missing").join("f"), b"x").is_err());
    }
}
