//! On-disk layout of the store and the per-entry install record (`installed.json`).
//!
//! ```text
//! <app_data>/models/
//!   .downloads/<key>.partial (+ .partial.json)  resumable downloads, kept on cancel
//!   .staging/<entry>.<pid>.<nonce>/             an install being assembled (renamed into place)
//!   .trash/                                     folders on their way out (swept at startup)
//!   <family>/<rest of id>/installed.json        the install record of every entry (all layouts)
//!   <family>/<rest of id>/<version>/...         files of ordinary entries
//! <app_data>/runtimes/<rest of id>/<version>/<variant>/  files of runtime packs (global rule)
//! <app_data>/whisper/, <app_data>/vosk-models/  files of the legacy families `whisper` and `vosk`
//! ```
//!
//! `<rest of id>` is the id without its first segment when that segment equals the family
//! (`stt/silero-vad` in family `stt` → `models/stt/silero-vad/`). The record is the only commit
//! marker: an entry is installed when its record parses, names this entry and every recorded file
//! is present with its recorded size. The record is written last, atomically (temp + rename).
//! Consumers never recompute any of this; they call `model_store::installed_path(id)`.

use super::legacy;
use super::{CatalogEntry, EntryKind, INSTALLED_MANIFEST, MODELS_DIR};
use serde::{Deserialize, Serialize};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

pub(crate) const DOWNLOADS_DIR: &str = ".downloads";
pub(crate) const STAGING_DIR: &str = ".staging";
pub(crate) const TRASH_DIR: &str = ".trash";
pub(crate) const RUNTIMES_DIR: &str = "runtimes";
/// Schema of `installed.json`.
pub(crate) const RECORD_SCHEMA: u32 = 1;

/// A process-unique suffix for temp names (`<ms>-<counter>`).
pub(crate) fn nonce() -> String {
    static SEQ: AtomicU64 = AtomicU64::new(0);
    format!("{}-{}", unix_ms(), SEQ.fetch_add(1, Ordering::Relaxed))
}

pub(crate) fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Folder-name form of a version or id segment: `[A-Za-z0-9._+-]`, anything else becomes `_`.
pub(crate) fn sanitize_dir_name(s: &str) -> String {
    let mut out: String = s
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '+' | '-') { c } else { '_' })
        .collect();
    while out.ends_with('.') {
        out.pop();
    }
    if out.is_empty() || out.chars().all(|c| c == '.') {
        out = "_".into();
    }
    out
}

/// The id segments below the family folder.
pub(crate) fn id_rest(entry: &CatalogEntry) -> Vec<String> {
    let segs: Vec<&str> = entry.id.split('/').collect();
    let rest = if segs.len() > 1 && segs[0] == entry.family { &segs[1..] } else { &segs[..] };
    rest.iter().map(|s| sanitize_dir_name(s)).collect()
}

/// Validate a catalog file path (relative, `/`-separated, no `..`, storable on Windows). Catalog
/// paths are also restricted to `[A-Za-z0-9._/+-]`; recorded content paths use the wider
/// [`valid_member_path`] rule instead.
pub(crate) fn valid_rel_path(rel: &str) -> bool {
    valid_member_path(rel)
        && rel
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '/' | '+' | '-'))
}

/// `base` joined with a validated catalog file path.
pub(crate) fn join_rel(base: &Path, rel: &str) -> Result<PathBuf, String> {
    if !valid_rel_path(rel) {
        return Err(format!("unsafe file path {rel:?} in the catalog"));
    }
    Ok(super::extract::join_parts(base, rel))
}

/// Validate a recorded content path: the rule archive extraction applies to member names
/// (`extract::member_path`), so every extracted file can be found again. Relative, `/`-separated,
/// every segment storable on Windows (spaces, `!`, `(`, `@` and non-ASCII are fine; `..`, `:`,
/// `\`, device names, empty segments and trailing dots or spaces are not).
pub(crate) fn valid_member_path(rel: &str) -> bool {
    !rel.is_empty() && rel.split('/').all(super::extract::valid_component)
}

/// `base` joined with a validated content path (an installed file of a record, or an extracted
/// archive member). Confined to `base` like [`join_rel`], with the extraction rule for names.
pub(crate) fn join_member(base: &Path, rel: &str) -> Result<PathBuf, String> {
    if !valid_member_path(rel) {
        return Err(format!("unsafe file path {rel:?} in the install record"));
    }
    Ok(super::extract::join_parts(base, rel))
}

/// Paths of one store (app data folder).
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    pub app_data: PathBuf,
    pub models: PathBuf,
}

impl Layout {
    pub fn new(app_data: &Path) -> Self {
        Self {
            app_data: app_data.to_path_buf(),
            models: app_data.join(MODELS_DIR),
        }
    }

    pub fn downloads(&self) -> PathBuf {
        self.models.join(DOWNLOADS_DIR)
    }

    pub fn staging(&self) -> PathBuf {
        self.models.join(STAGING_DIR)
    }

    pub fn trash(&self) -> PathBuf {
        self.models.join(TRASH_DIR)
    }

    /// `models/<family>/<rest of id>`: holds the record (and the version folders of ordinary entries).
    pub fn entry_dir(&self, entry: &CatalogEntry) -> PathBuf {
        let mut p = self.models.join(sanitize_dir_name(&entry.family));
        for s in id_rest(entry) {
            p.push(s);
        }
        p
    }

    /// `/`-separated form of [`Self::entry_dir`] relative to the models folder (uniqueness checks).
    pub fn entry_key(entry: &CatalogEntry) -> String {
        let mut parts = vec![sanitize_dir_name(&entry.family)];
        parts.extend(id_rest(entry));
        parts.join("/").to_lowercase()
    }

    pub fn record_path(&self, entry: &CatalogEntry) -> PathBuf {
        self.entry_dir(entry).join(INSTALLED_MANIFEST)
    }

    /// Where the files of `entry` go on a fresh install (before collision handling).
    pub fn preferred_content_dir(&self, entry: &CatalogEntry) -> PathBuf {
        if let Some(folder) = legacy::legacy_folder(&entry.family) {
            return self.app_data.join(folder);
        }
        if entry.kind == EntryKind::Runtime {
            let mut p = self.app_data.join(RUNTIMES_DIR);
            for s in id_rest(entry) {
                p.push(s);
            }
            p.push(sanitize_dir_name(&entry.version));
            p.push(sanitize_dir_name(entry.variant.as_deref().unwrap_or("default")));
            return p;
        }
        self.entry_dir(entry).join(sanitize_dir_name(&entry.version))
    }

    /// `path` relative to the app data folder, `/`-separated (for records).
    pub fn rel_to_app_data(&self, path: &Path) -> Option<String> {
        let rel = path.strip_prefix(&self.app_data).ok()?;
        let mut parts = Vec::new();
        for c in rel.components() {
            match c {
                Component::Normal(s) => parts.push(s.to_str()?.to_string()),
                _ => return None,
            }
        }
        (!parts.is_empty()).then(|| parts.join("/"))
    }

    /// Resolve a record's `root`. Only folders the store may own are accepted: something under
    /// `models/` or `runtimes/`, or a legacy folder. A tampered record can never point `remove`
    /// at an arbitrary folder.
    pub fn resolve_root(&self, rel: &str) -> Option<PathBuf> {
        let parts: Vec<&str> = rel.split('/').collect();
        if parts.is_empty() || parts.iter().any(|p| !super::extract::valid_component(p)) {
            return None;
        }
        let allowed = match parts[0] {
            MODELS_DIR | RUNTIMES_DIR => parts.len() >= 3 && !parts[1].starts_with('.'),
            first => parts.len() == 1 && legacy::is_legacy_folder(first),
        };
        allowed.then(|| super::extract::join_parts(&self.app_data, rel))
    }

    /// Whether `root` is one of the shared legacy folders (files are managed one by one there).
    pub fn is_legacy_root(&self, root: &Path) -> bool {
        self.rel_to_app_data(root).is_some_and(|r| legacy::is_legacy_folder(&r))
    }

    /// Whether `root` is a folder the store creates and may delete as a whole: under `models/` or
    /// `runtimes/` (never a shared legacy folder, never the app data folder itself).
    pub fn is_store_root(&self, root: &Path) -> bool {
        (root.starts_with(&self.models) || root.starts_with(self.app_data.join(RUNTIMES_DIR)))
            && self.rel_to_app_data(root).is_some_and(|r| self.resolve_root(&r).is_some())
            && !self.is_legacy_root(root)
    }

    /// `models/.pending-trash.json`: install folders that could not be removed yet.
    pub fn pending_trash_path(&self) -> PathBuf {
        self.models.join(PENDING_TRASH)
    }

    /// The folders every entry of the catalog owns, as lower-case `/`-separated keys relative to
    /// the app data folder: `models/<family>/<id rest>` (record and version folders) and, for
    /// runtime packs, `runtimes/<id rest>`. No key may equal or contain another entry's key.
    pub fn owned_dir_keys(entry: &CatalogEntry) -> Vec<String> {
        let mut keys = vec![format!("{MODELS_DIR}/{}", Self::entry_key(entry))];
        if entry.kind == EntryKind::Runtime && legacy::legacy_folder(&entry.family).is_none() {
            keys.push(format!("{RUNTIMES_DIR}/{}", id_rest(entry).join("/").to_lowercase()));
        }
        keys
    }
}

// ---------------------------------------------------------------------------------------------
// Pending trash
// ---------------------------------------------------------------------------------------------

/// File name of the pending-trash list under `models/`.
pub(crate) const PENDING_TRASH: &str = ".pending-trash.json";

/// An install folder that could not be moved to the trash (a file inside was in use, typically a
/// loaded DLL of a runtime pack). Retried by `Store::sweep` at the next start.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PendingRoot {
    /// Relative to the app data folder, `/`-separated (checked with [`Layout::resolve_root`]).
    pub root: String,
    /// Catalog family (disk usage).
    pub family: String,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct PendingTrash {
    #[serde(default)]
    pub roots: Vec<PendingRoot>,
}

pub(crate) fn read_pending_trash(layout: &Layout) -> PendingTrash {
    fs::read(layout.pending_trash_path())
        .ok()
        .and_then(|b| serde_json::from_slice(&b).ok())
        .unwrap_or_default()
}

/// Write the list (removes the file when it is empty). Call with the registry lock held.
pub(crate) fn write_pending_trash(layout: &Layout, pending: &PendingTrash) -> io::Result<()> {
    let path = layout.pending_trash_path();
    if pending.roots.is_empty() {
        return match fs::remove_file(&path) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e),
            _ => Ok(()),
        };
    }
    let json = serde_json::to_vec_pretty(pending).map_err(io::Error::other)?;
    write_atomic(&path, &json)
}

/// Move a store-owned install folder to the trash, or, when that fails (a file inside is in use),
/// remember it in [`PENDING_TRASH`] so the next start removes it. Returns whether it is gone now.
/// Call with the registry lock held.
pub(crate) fn trash_or_defer(layout: &Layout, dir: &Path, family: &str) -> bool {
    let err = match trash_dir(layout, dir) {
        Ok(()) => return true,
        Err(e) => e,
    };
    let Some(root) = layout.rel_to_app_data(dir).filter(|_| layout.is_store_root(dir)) else {
        tracing::warn!(target: "sigil::models", "{} could not be removed ({err})", dir.display());
        return false;
    };
    tracing::info!(target: "sigil::models", "{} is in use; it is removed at the next start ({err})", dir.display());
    let mut pending = read_pending_trash(layout);
    if !pending.roots.iter().any(|p| p.root.eq_ignore_ascii_case(&root)) {
        pending.roots.push(PendingRoot {
            root,
            family: family.to_string(),
        });
        if let Err(e) = write_pending_trash(layout, &pending) {
            tracing::warn!(target: "sigil::models", "could not record {} for removal: {e}", dir.display());
        }
    }
    false
}

/// Bytes of the regular files under `dir` (links are not followed).
pub(crate) fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(rd) = fs::read_dir(&d) else { continue };
        for e in rd.flatten() {
            match e.path().symlink_metadata() {
                Ok(m) if m.is_dir() => stack.push(e.path()),
                Ok(m) if m.is_file() => total += m.len(),
                _ => {}
            }
        }
    }
    total
}

// ---------------------------------------------------------------------------------------------
// Install record
// ---------------------------------------------------------------------------------------------

/// How an entry got onto this PC.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub(crate) enum Origin {
    Download,
    Import,
    Adopted,
    LocalAssets,
}

/// A catalog file as installed (the downloaded artifact; for archives, the archive itself).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RecordedFile {
    pub path: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_oid: Option<String>,
    #[serde(default)]
    pub archive: bool,
}

/// A file on disk under the root that belongs to the entry (plain files, or extracted members).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct RecordedContent {
    pub path: String,
    pub size: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub git_oid: Option<String>,
}

/// `installed.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct InstalledRecord {
    pub schema: u32,
    pub id: String,
    pub family: String,
    pub kind: EntryKind,
    pub version: String,
    /// Folder holding the files, relative to the app data folder, `/`-separated.
    pub root: String,
    pub files: Vec<RecordedFile>,
    pub contents: Vec<RecordedContent>,
    pub origin: Origin,
    /// Every file was checked against a catalog hash (false: trust on first use / local import).
    pub pinned: bool,
    pub installed_at_unix_ms: u64,
}

/// Read a record; `None` when missing, unreadable, from a newer schema or for another id.
pub(crate) fn read_record(path: &Path, id: &str) -> Option<InstalledRecord> {
    let text = fs::read_to_string(path).ok()?;
    match serde_json::from_str::<InstalledRecord>(&text) {
        Ok(r) if r.schema == RECORD_SCHEMA && r.id == id => Some(r),
        Ok(_) => None,
        Err(e) => {
            tracing::warn!(target: "sigil::models", "ignoring unreadable install record {}: {e}", path.display());
            None
        }
    }
}

/// Write `bytes` to `path` atomically: temp file in the same folder, flushed, then renamed over.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> io::Result<()> {
    let dir = path.parent().ok_or_else(|| io::Error::other("no parent folder"))?;
    fs::create_dir_all(dir)?;
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp = dir.join(format!("{name}.{}.{}.tmp", std::process::id(), nonce()));
    let result = (|| {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        drop(f);
        fs::rename(&tmp, path)
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

pub(crate) fn write_record(path: &Path, record: &InstalledRecord) -> io::Result<()> {
    let json = serde_json::to_vec_pretty(record).map_err(io::Error::other)?;
    write_atomic(path, &json)
}

/// Stat every recorded content file under `root`: `Ok(bytes)` when all are present with their
/// recorded sizes, else the first problem.
pub(crate) fn contents_present(root: &Path, record: &InstalledRecord) -> Result<u64, String> {
    let mut total = 0;
    for c in &record.contents {
        let p = join_member(root, &c.path)?;
        match fs::metadata(&p) {
            Ok(m) if m.is_file() && m.len() == c.size => total += c.size,
            Ok(m) if m.is_file() => {
                return Err(format!("{} has {} bytes, expected {}", c.path, m.len(), c.size));
            }
            _ => return Err(format!("{} is missing", c.path)),
        }
    }
    Ok(total)
}

// ---------------------------------------------------------------------------------------------
// File helpers
// ---------------------------------------------------------------------------------------------

/// Move `src` to `dst` (replacing a file there): a rename, or copy + rename across volumes.
pub(crate) fn move_file(src: &Path, dst: &Path) -> io::Result<()> {
    if let Some(parent) = dst.parent() {
        fs::create_dir_all(parent)?;
    }
    match fs::rename(src, dst) {
        Ok(()) => Ok(()),
        // ERROR_NOT_SAME_DEVICE (17) / EXDEV (18): copy next to the target, then rename.
        Err(e) if matches!(e.raw_os_error(), Some(17) | Some(18)) => {
            let tmp = dst.with_extension(format!("{}.{}.tmp", std::process::id(), nonce()));
            let result = fs::copy(src, &tmp).and_then(|_| {
                OpenOptions::new().write(true).open(&tmp)?.sync_all()?;
                fs::rename(&tmp, dst)
            });
            if result.is_err() {
                let _ = fs::remove_file(&tmp);
                return result;
            }
            let _ = fs::remove_file(src);
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Move a folder out of the way into the trash, then try to delete it. Returns an error only when
/// the move itself fails (typically: a file inside is in use).
pub(crate) fn trash_dir(layout: &Layout, dir: &Path) -> io::Result<()> {
    if !dir.exists() {
        return Ok(());
    }
    let trash = layout.trash();
    fs::create_dir_all(&trash)?;
    let name = dir.file_name().and_then(|n| n.to_str()).unwrap_or("dir");
    let target = trash.join(format!("{name}.{}.{}", std::process::id(), nonce()));
    fs::rename(dir, &target)?;
    let _ = super::extract::remove_tree(&target);
    Ok(())
}

/// Human-readable form of an I/O error, naming the common Windows cases.
pub(crate) fn describe_io(what: &str, e: &io::Error) -> String {
    match e.raw_os_error() {
        Some(32) | Some(33) => format!("{what}: the file is in use by another program (or another Sigil process)"),
        Some(112) | Some(39) => format!("{what}: not enough disk space"),
        Some(5) => format!("{what}: access denied"),
        _ => format!("{what}: {e}"),
    }
}

/// Free bytes for the current user on the volume holding `path` (or its nearest existing parent).
pub(crate) fn free_space(path: &Path) -> Option<u64> {
    let mut p = path;
    while !p.exists() {
        p = p.parent()?;
    }
    free_space_impl(p)
}

#[cfg(windows)]
fn free_space_impl(dir: &Path) -> Option<u64> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(
            directory: *const u16,
            free_bytes_available_to_caller: *mut u64,
            total_bytes: *mut u64,
            total_free_bytes: *mut u64,
        ) -> i32;
    }
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let mut free: u64 = 0;
    // SAFETY: `wide` is NUL-terminated and outlives the call; the out pointer is valid, the other
    // two are optional (null) per the API contract.
    let ok = unsafe { GetDiskFreeSpaceExW(wide.as_ptr(), &mut free, std::ptr::null_mut(), std::ptr::null_mut()) };
    (ok != 0).then_some(free)
}

#[cfg(not(windows))]
fn free_space_impl(_dir: &Path) -> Option<u64> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::model_store::{Catalog, EntryKind};

    fn entry(json: &str) -> CatalogEntry {
        let c = Catalog::from_json(&format!(r#"{{"entries":[{json}]}}"#)).unwrap();
        c.entries[0].clone()
    }

    fn plain(id: &str, family: &str, kind: &str, version: &str, variant: Option<&str>) -> CatalogEntry {
        let variant = variant.map(|v| format!(r#","variant":"{v}""#)).unwrap_or_default();
        entry(&format!(
            r#"{{"id":"{id}","family":"{family}","kind":"{kind}","version":"{version}","license":"MIT","status":"published"{variant},
            "files":[{{"path":"a.bin","size":1,"sources":[{{"kind":"direct","url":"https://example.invalid/a"}}]}}]}}"#
        ))
    }

    #[test]
    fn layout_places_each_kind() {
        let l = Layout::new(Path::new("C:/AppData/com.sigil.app"));
        let m = plain("stt/silero-vad", "stt", "model", "6.2", None);
        assert_eq!(l.entry_dir(&m), Path::new("C:/AppData/com.sigil.app/models/stt/silero-vad"));
        assert_eq!(l.preferred_content_dir(&m), l.entry_dir(&m).join("6.2"));
        let r = plain("runtime/sherpa-onnx", "runtime", "runtime", "1.13.8", Some("cpu"));
        assert_eq!(
            l.preferred_content_dir(&r),
            Path::new("C:/AppData/com.sigil.app/runtimes/sherpa-onnx/1.13.8/cpu")
        );
        assert_eq!(r.kind, EntryKind::Runtime);
        let w = plain("whisper/base.en", "whisper", "model", "1", None);
        assert_eq!(l.preferred_content_dir(&w), Path::new("C:/AppData/com.sigil.app/whisper"));
        assert_eq!(l.entry_dir(&w), Path::new("C:/AppData/com.sigil.app/models/whisper/base.en"));
        let odd = plain("base.test-small", "base", "bench", "v 1/2", None);
        assert_eq!(l.entry_dir(&odd), Path::new("C:/AppData/com.sigil.app/models/base/base.test-small"));
        assert!(l.preferred_content_dir(&odd).ends_with("v_1_2"));
    }

    #[test]
    fn record_roots_are_confined() {
        let l = Layout::new(Path::new("C:/AppData/app"));
        assert!(l.resolve_root("models/stt/x/1").is_some());
        assert!(l.resolve_root("runtimes/sherpa/1.0/cpu").is_some());
        assert!(l.resolve_root("whisper").is_some());
        assert!(l.resolve_root("vosk-models").is_some());
        for bad in ["", "..", "models/../..", "models", "models/.downloads/x", "user/settings", "C:/Windows", "whisper/..", "/abs"] {
            assert!(l.resolve_root(bad).is_none(), "{bad}");
        }
        assert_eq!(l.rel_to_app_data(Path::new("C:/AppData/app/models/stt/x/1")).as_deref(), Some("models/stt/x/1"));
        assert_eq!(l.rel_to_app_data(Path::new("C:/elsewhere")), None);
    }

    #[test]
    fn rel_paths_are_validated() {
        assert!(valid_rel_path("a/b.onnx"));
        assert!(valid_rel_path("vosk-model-small-en-us-0.15.zip"));
        for bad in ["", "/a", "a/../b", "a//b", "a\\b", "con.txt", "a/nul", "x:y", "a/b."] {
            assert!(!valid_rel_path(bad), "{bad}");
        }
        assert_eq!(sanitize_dir_name(".."), "_");
        assert_eq!(sanitize_dir_name("1.0 beta"), "1.0_beta");
    }

    #[test]
    fn content_paths_follow_the_extraction_rule() {
        // Names real archives ship (sherpa espeak-ng-data) that catalog paths may not use.
        for ok in ["espeak-ng-data/voices/!v/Mr serious", "data/(x)@y~z,w.txt", "données/модель.onnx", "a/b.onnx"] {
            assert!(valid_member_path(ok), "{ok}");
            assert!(!valid_rel_path(ok) || ok == "a/b.onnx", "{ok}");
            let p = join_member(Path::new("C:/root"), ok).unwrap();
            assert!(p.starts_with("C:/root"), "{ok}");
        }
        for bad in ["", "/a", "a/../b", "a//b", "a\\b", "con.txt", "a/nul", "x:y", "a/b.", "a/b ", "..", "C:/x"] {
            assert!(!valid_member_path(bad), "{bad}");
            assert!(join_member(Path::new("C:/root"), bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn owned_folders_and_store_roots() {
        let l = Layout::new(Path::new("C:/AppData/app"));
        let r = plain("runtime/sherpa-onnx", "runtime", "runtime", "1.13.8", Some("cpu"));
        assert_eq!(Layout::owned_dir_keys(&r), ["models/runtime/sherpa-onnx", "runtimes/sherpa-onnx"]);
        let m = plain("stt/Silero", "stt", "model", "1", None);
        assert_eq!(Layout::owned_dir_keys(&m), ["models/stt/silero"]);
        assert!(l.is_store_root(&l.preferred_content_dir(&r)));
        assert!(l.is_store_root(&l.preferred_content_dir(&m)));
        assert!(!l.is_store_root(Path::new("C:/AppData/app/whisper")));
        assert!(!l.is_store_root(Path::new("C:/AppData/app/runtimes/nvidia-cu13")), "too shallow to be an install");
        assert!(!l.is_store_root(Path::new("C:/AppData/app/models/.trash/x/y")));
        assert!(!l.is_store_root(Path::new("C:/elsewhere/runtimes/a/b/c")));
    }

    #[test]
    fn folders_in_use_are_remembered_for_the_next_start() {
        let d = std::env::temp_dir().join(format!("sigil-ms-pending-{}-{}", std::process::id(), nonce()));
        let l = Layout::new(&d);
        let busy = d.join("runtimes").join("x").join("1").join("cpu");
        fs::create_dir_all(&busy).unwrap();
        fs::write(busy.join("a.dll"), b"12345").unwrap();
        assert_eq!(dir_size(&busy), 5);
        // A free folder goes to the trash at once and is not remembered.
        assert!(trash_or_defer(&l, &busy, "runtime"));
        assert!(!busy.exists());
        assert!(read_pending_trash(&l).roots.is_empty());
        // Hold a file open without delete sharing (a loaded DLL) so the move fails.
        fs::create_dir_all(&busy).unwrap();
        fs::write(busy.join("a.dll"), b"12345").unwrap();
        #[cfg(windows)]
        {
            use std::os::windows::fs::OpenOptionsExt;
            let _held = OpenOptions::new().read(true).share_mode(0x1).open(busy.join("a.dll")).unwrap();
            assert!(!trash_or_defer(&l, &busy, "runtime"));
            assert!(!trash_or_defer(&l, &busy, "runtime"), "listed once");
            let pending = read_pending_trash(&l);
            assert_eq!(
                pending.roots,
                [PendingRoot {
                    root: "runtimes/x/1/cpu".into(),
                    family: "runtime".into()
                }]
            );
        }
        write_pending_trash(&l, &PendingTrash::default()).unwrap();
        assert!(!l.pending_trash_path().exists());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn atomic_write_replaces_and_leaves_no_temp() {
        let d = std::env::temp_dir().join(format!("sigil-ms-layout-{}-{}", std::process::id(), nonce()));
        fs::create_dir_all(&d).unwrap();
        let p = d.join("installed.json");
        write_atomic(&p, b"one").unwrap();
        write_atomic(&p, b"two").unwrap();
        assert_eq!(fs::read(&p).unwrap(), b"two");
        assert_eq!(fs::read_dir(&d).unwrap().count(), 1);
        assert!(free_space(&d.join("not/yet/created")).is_some() || cfg!(not(windows)));
        let _ = fs::remove_dir_all(&d);
    }
}
