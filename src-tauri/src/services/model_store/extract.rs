//! Safe archive extraction (zip, tar.gz, tar.bz2) into a staging folder.
//!
//! Every member name is validated before anything is written: no absolute paths or drive letters,
//! no `..`, no Windows device names (`CON`, `NUL`, `COM1`, ...), no `:` (alternate data streams) or
//! other characters Windows cannot store, and no two members that differ only in case. Symbolic
//! links, hard links, devices and sparse files are refused. Entry count, per-entry size, total size
//! and the compression ratio are capped (zip bombs), the ratio being measured on the bytes actually
//! produced, not on what the headers claim. Files are created with `create_new`, so nothing is
//! overwritten, and every extracted file is hashed (sha256) while it is written so the install
//! record can verify it later.

use super::verify::{Digest, FileHasher, Expected};
use super::{ArchiveFormat, ArchiveSpec};
use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Caps applied to every archive.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExtractLimits {
    pub max_entries: usize,
    pub max_entry_bytes: u64,
    pub max_total_bytes: u64,
    /// Uncompressed bytes per compressed byte of the whole archive.
    pub max_ratio: u64,
    /// Below this many uncompressed bytes the ratio is not enforced (tiny archives of text files).
    pub ratio_floor_bytes: u64,
}

impl Default for ExtractLimits {
    fn default() -> Self {
        Self {
            max_entries: 50_000,
            max_entry_bytes: 32 << 30,
            max_total_bytes: 64 << 30,
            max_ratio: 200,
            ratio_floor_bytes: 64 << 20,
        }
    }
}

impl ExtractLimits {
    fn ratio_cap(&self, compressed: u64) -> u64 {
        compressed.saturating_mul(self.max_ratio).max(self.ratio_floor_bytes)
    }
}

/// One extracted file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ExtractedFile {
    /// `/`-separated path relative to the destination.
    pub path: String,
    pub size: u64,
    pub sha256: String,
}

/// Why an extraction stopped.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ExtractError {
    Cancelled,
    /// The archive is unsafe or damaged (never retried from the same bytes).
    Rejected(String),
    /// Disk or other I/O failure while writing.
    Io(String),
}

impl std::fmt::Display for ExtractError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExtractError::Cancelled => write!(f, "cancelled"),
            ExtractError::Rejected(m) | ExtractError::Io(m) => write!(f, "{m}"),
        }
    }
}

fn rejected(msg: impl Into<String>) -> ExtractError {
    ExtractError::Rejected(msg.into())
}

// ---------------------------------------------------------------------------------------------
// Member names
// ---------------------------------------------------------------------------------------------

const WINDOWS_DEVICE_NAMES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8",
    "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
];

/// Whether `component` can be stored as one path component on Windows (and is not a device name).
pub(crate) fn valid_component(component: &str) -> bool {
    if component.is_empty() || component == "." || component == ".." || component.len() > 255 {
        return false;
    }
    if component.ends_with('.') || component.ends_with(' ') {
        return false;
    }
    if component
        .chars()
        .any(|c| c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '|' | '?' | '*' | '\\' | '/'))
    {
        return false;
    }
    let stem = component.split('.').next().unwrap_or(component).trim_end();
    !WINDOWS_DEVICE_NAMES.iter().any(|d| d.eq_ignore_ascii_case(stem))
}

/// Validate an archive member name and apply `strip`: `Ok(None)` when nothing is left (a stripped
/// top-level folder), `Err` for anything unsafe.
pub(crate) fn member_path(raw: &str, strip: u32) -> Result<Option<Vec<String>>, ExtractError> {
    let name = raw.replace('\\', "/");
    if name.starts_with('/') || name.as_bytes().get(1) == Some(&b':') || name.contains(':') {
        return Err(rejected(format!("archive member {raw:?} has an absolute path")));
    }
    let mut parts = Vec::new();
    for comp in name.split('/') {
        match comp {
            "" | "." => continue,
            ".." => return Err(rejected(format!("archive member {raw:?} escapes the install folder (..)"))),
            c if !valid_component(c) => {
                return Err(rejected(format!("archive member {raw:?} has a name Windows cannot store")))
            }
            c => parts.push(c.to_string()),
        }
    }
    if parts.len() > 64 {
        return Err(rejected(format!("archive member {raw:?} is nested too deeply")));
    }
    let strip = strip as usize;
    if parts.len() <= strip {
        return Ok(None);
    }
    Ok(Some(parts.split_off(strip)))
}

/// `*` matches within one path segment, `?` one character, `**` any number of whole segments.
pub(crate) fn glob_match(pattern: &str, path: &str) -> bool {
    fn seg_match(p: &[char], s: &[char]) -> bool {
        match (p.first(), s.first()) {
            (None, None) => true,
            (Some('*'), _) => seg_match(&p[1..], s) || (!s.is_empty() && seg_match(p, &s[1..])),
            (Some('?'), Some(_)) => seg_match(&p[1..], &s[1..]),
            (Some(a), Some(b)) if a == b => seg_match(&p[1..], &s[1..]),
            _ => false,
        }
    }
    fn match_segs(p: &[&str], s: &[&str]) -> bool {
        match p.first() {
            None => s.is_empty(),
            Some(&"**") => match_segs(&p[1..], s) || (!s.is_empty() && match_segs(p, &s[1..])),
            Some(ps) => {
                !s.is_empty() && {
                    let pc: Vec<char> = ps.chars().collect();
                    let sc: Vec<char> = s[0].chars().collect();
                    seg_match(&pc, &sc) && match_segs(&p[1..], &s[1..])
                }
            }
        }
    }
    let p: Vec<&str> = pattern.split('/').filter(|s| !s.is_empty()).collect();
    let s: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();
    match_segs(&p, &s)
}

fn included(spec: &ArchiveSpec, rel: &str) -> bool {
    spec.include.is_empty() || spec.include.iter().any(|g| glob_match(g, rel))
}

// ---------------------------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------------------------

/// Tracks totals and duplicate names across one archive.
struct Budget<'a> {
    limits: &'a ExtractLimits,
    compressed: Arc<AtomicU64>,
    written: u64,
    entries: usize,
    names: HashSet<String>,
}

impl Budget<'_> {
    fn count_entry(&mut self, raw: &str) -> Result<(), ExtractError> {
        self.entries += 1;
        if self.entries > self.limits.max_entries {
            return Err(rejected(format!(
                "the archive has more than {} entries (at {raw:?})",
                self.limits.max_entries
            )));
        }
        Ok(())
    }

    fn claim_name(&mut self, rel: &str) -> Result<(), ExtractError> {
        if !self.names.insert(rel.to_lowercase()) {
            return Err(rejected(format!("the archive contains {rel:?} twice (names differ only in case?)")));
        }
        Ok(())
    }

    fn add(&mut self, n: u64) -> Result<(), ExtractError> {
        self.written += n;
        if self.written > self.limits.max_total_bytes {
            return Err(rejected(format!(
                "the archive expands to more than {} GiB",
                self.limits.max_total_bytes >> 30
            )));
        }
        let compressed = self.compressed.load(Ordering::Relaxed).max(1);
        if self.written > self.limits.ratio_cap(compressed) {
            return Err(rejected(format!(
                "the archive expands more than {}x (zip bomb?)",
                self.limits.max_ratio
            )));
        }
        Ok(())
    }
}

fn io_err(what: &str, e: io::Error) -> ExtractError {
    ExtractError::Io(format!("{what}: {e}"))
}

/// Stream `reader` into a new file at `dest/rel`, hashing it. `declared` caps the entry.
fn write_member(
    reader: &mut dyn Read,
    dest: &Path,
    parts: &[String],
    declared: u64,
    budget: &mut Budget<'_>,
    cancelled: &dyn Fn() -> bool,
) -> Result<ExtractedFile, ExtractError> {
    let rel = parts.join("/");
    if declared > budget.limits.max_entry_bytes {
        return Err(rejected(format!("archive member {rel:?} is larger than the per-file limit")));
    }
    let mut out_path = dest.to_path_buf();
    for p in parts {
        out_path.push(p);
    }
    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| io_err(&format!("create folder for {rel}"), e))?;
    }
    let mut out = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&out_path)
        .map_err(|e| io_err(&format!("create {rel}"), e))?;
    let mut hasher = FileHasher::new(&Expected::None, declared);
    let mut buf = vec![0u8; 256 * 1024];
    let mut size: u64 = 0;
    loop {
        if cancelled() {
            return Err(ExtractError::Cancelled);
        }
        let n = match reader.read(&mut buf) {
            Ok(0) => break,
            Ok(n) => n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(rejected(format!("archive member {rel:?} is damaged: {e}"))),
        };
        size += n as u64;
        if size > declared || size > budget.limits.max_entry_bytes {
            return Err(rejected(format!("archive member {rel:?} is larger than its header says")));
        }
        budget.add(n as u64)?;
        hasher.update(&buf[..n]);
        out.write_all(&buf[..n]).map_err(|e| io_err(&format!("write {rel}"), e))?;
    }
    if size != declared {
        return Err(rejected(format!("archive member {rel:?} is truncated ({size} of {declared} bytes)")));
    }
    out.sync_all().map_err(|e| io_err(&format!("flush {rel}"), e))?;
    let sha256 = match hasher.finish() {
        Digest::Sha256(h) => h,
        Digest::GitOid(_) => unreachable!("extraction hashes with sha256"),
    };
    Ok(ExtractedFile { path: rel, size, sha256 })
}

/// Counts bytes read from the compressed archive (for the ratio cap on tar streams).
struct CountingReader<R> {
    inner: R,
    count: Arc<AtomicU64>,
}

impl<R: Read> Read for CountingReader<R> {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.count.fetch_add(n as u64, Ordering::Relaxed);
        Ok(n)
    }
}

// ---------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------

/// Extract `archive` into `dest` (which must exist and should be empty) following `spec`.
pub(crate) fn extract_archive(
    archive: &Path,
    spec: &ArchiveSpec,
    dest: &Path,
    limits: &ExtractLimits,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<ExtractedFile>, ExtractError> {
    match spec.format {
        ArchiveFormat::Zip => extract_zip(archive, spec, dest, limits, cancelled),
        ArchiveFormat::TarGz | ArchiveFormat::TarBz2 => extract_tar(archive, spec, dest, limits, cancelled),
    }
}

fn open_zip(archive: &Path) -> Result<(zip::ZipArchive<BufReader<File>>, u64), ExtractError> {
    let file = File::open(archive).map_err(|e| io_err("open archive", e))?;
    let len = file.metadata().map_err(|e| io_err("read archive size", e))?.len();
    let zip = zip::ZipArchive::new(BufReader::new(file))
        .map_err(|e| rejected(format!("not a valid zip archive: {e}")))?;
    Ok((zip, len))
}

fn extract_zip(
    archive: &Path,
    spec: &ArchiveSpec,
    dest: &Path,
    limits: &ExtractLimits,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<ExtractedFile>, ExtractError> {
    let (mut zip, archive_len) = open_zip(archive)?;
    if zip.len() > limits.max_entries {
        return Err(rejected(format!("the archive has more than {} entries", limits.max_entries)));
    }
    let mut budget = Budget {
        limits,
        compressed: Arc::new(AtomicU64::new(archive_len)),
        written: 0,
        entries: 0,
        names: HashSet::new(),
    };
    // Plan first (no bytes written): names, links and the declared totals.
    let mut plan: Vec<(usize, Vec<String>, u64)> = Vec::new();
    let mut dirs: Vec<Vec<String>> = Vec::new();
    let mut declared_total: u64 = 0;
    for i in 0..zip.len() {
        let f = zip
            .by_index_raw(i)
            .map_err(|e| rejected(format!("damaged zip entry #{i}: {e}")))?;
        let raw = f.name().to_string();
        budget.count_entry(&raw)?;
        let Some(parts) = member_path(&raw, spec.strip)? else { continue };
        let rel = parts.join("/");
        if f.is_symlink() {
            if included(spec, &rel) {
                return Err(rejected(format!("the archive contains a symbolic link ({raw:?})")));
            }
            continue;
        }
        if f.is_dir() {
            if included(spec, &rel) {
                dirs.push(parts);
            }
            continue;
        }
        if !included(spec, &rel) {
            continue;
        }
        budget.claim_name(&rel)?;
        if f.size() > limits.max_entry_bytes {
            return Err(rejected(format!("archive member {raw:?} is larger than the per-file limit")));
        }
        declared_total = declared_total.saturating_add(f.size());
        plan.push((i, parts, f.size()));
    }
    if declared_total > limits.max_total_bytes {
        return Err(rejected(format!("the archive expands to more than {} GiB", limits.max_total_bytes >> 30)));
    }
    if declared_total > limits.ratio_cap(archive_len.max(1)) {
        return Err(rejected(format!("the archive expands more than {}x (zip bomb?)", limits.max_ratio)));
    }
    for parts in dirs {
        let mut p = dest.to_path_buf();
        for c in &parts {
            p.push(c);
        }
        fs::create_dir_all(&p).map_err(|e| io_err("create folder", e))?;
    }
    let mut out = Vec::with_capacity(plan.len());
    for (i, parts, declared) in plan {
        if cancelled() {
            return Err(ExtractError::Cancelled);
        }
        let mut f = zip
            .by_index(i)
            .map_err(|e| rejected(format!("cannot read zip entry {:?}: {e}", parts.join("/"))))?;
        out.push(write_member(&mut f, dest, &parts, declared, &mut budget, cancelled)?);
    }
    Ok(out)
}

fn extract_tar(
    archive: &Path,
    spec: &ArchiveSpec,
    dest: &Path,
    limits: &ExtractLimits,
    cancelled: &dyn Fn() -> bool,
) -> Result<Vec<ExtractedFile>, ExtractError> {
    let file = File::open(archive).map_err(|e| io_err("open archive", e))?;
    let compressed = Arc::new(AtomicU64::new(0));
    let counting = BufReader::new(CountingReader {
        inner: file,
        count: compressed.clone(),
    });
    let decoder: Box<dyn Read> = match spec.format {
        ArchiveFormat::TarGz => Box::new(flate2::read::GzDecoder::new(counting)),
        ArchiveFormat::TarBz2 => Box::new(bzip2::read::BzDecoder::new(counting)),
        ArchiveFormat::Zip => unreachable!("zip handled separately"),
    };
    let mut tar = tar::Archive::new(decoder);
    let mut budget = Budget {
        limits,
        compressed,
        written: 0,
        entries: 0,
        names: HashSet::new(),
    };
    let mut out = Vec::new();
    let entries = tar
        .entries()
        .map_err(|e| rejected(format!("not a valid tar archive: {e}")))?;
    for entry in entries {
        if cancelled() {
            return Err(ExtractError::Cancelled);
        }
        let mut entry = entry.map_err(|e| rejected(format!("damaged tar archive: {e}")))?;
        let raw = String::from_utf8(entry.path_bytes().into_owned())
            .map_err(|_| rejected("the archive has a member name that is not UTF-8"))?;
        budget.count_entry(&raw)?;
        let Some(parts) = member_path(&raw, spec.strip)? else { continue };
        let rel = parts.join("/");
        let kind = entry.header().entry_type();
        use tar::EntryType as T;
        match kind {
            T::Regular | T::Continuous => {}
            T::Directory => {
                if included(spec, &rel) {
                    let mut p = dest.to_path_buf();
                    for c in &parts {
                        p.push(c);
                    }
                    fs::create_dir_all(&p).map_err(|e| io_err("create folder", e))?;
                }
                continue;
            }
            T::Symlink | T::Link => {
                if included(spec, &rel) {
                    return Err(rejected(format!("the archive contains a link ({raw:?})")));
                }
                continue;
            }
            T::XGlobalHeader | T::XHeader | T::GNULongName | T::GNULongLink => continue,
            other => {
                if included(spec, &rel) {
                    return Err(rejected(format!("unsupported archive member type {other:?} ({raw:?})")));
                }
                continue;
            }
        }
        if !included(spec, &rel) {
            continue;
        }
        budget.claim_name(&rel)?;
        let declared = entry.size();
        out.push(write_member(&mut entry, dest, &parts, declared, &mut budget, cancelled)?);
    }
    Ok(out)
}

/// Summary of [`validate_zip`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct ZipSummary {
    pub entries: usize,
    pub uncompressed_bytes: u64,
}

/// Check that `path` is a complete, undamaged zip without extracting it: the central directory
/// parses and every member decompresses with a matching CRC-32, within the limits.
pub(crate) fn validate_zip(path: &Path, limits: &ExtractLimits, cancelled: &dyn Fn() -> bool) -> Result<ZipSummary, ExtractError> {
    let (mut zip, archive_len) = open_zip(path)?;
    if zip.len() > limits.max_entries {
        return Err(rejected(format!("the zip has more than {} entries", limits.max_entries)));
    }
    if zip.is_empty() {
        return Err(rejected("the zip is empty"));
    }
    let cap = limits.ratio_cap(archive_len.max(1)).min(limits.max_total_bytes);
    let mut total: u64 = 0;
    let mut buf = vec![0u8; 256 * 1024];
    for i in 0..zip.len() {
        let mut f = zip
            .by_index(i)
            .map_err(|e| rejected(format!("damaged zip entry #{i}: {e}")))?;
        let name = f.name().to_string();
        loop {
            if cancelled() {
                return Err(ExtractError::Cancelled);
            }
            let n = match f.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(rejected(format!("zip member {name:?} is damaged: {e}"))),
            };
            total += n as u64;
            if total > cap {
                return Err(rejected("the zip expands too much (zip bomb?)"));
            }
        }
    }
    Ok(ZipSummary {
        entries: zip.len(),
        uncompressed_bytes: total,
    })
}

/// Remove a folder tree, tolerating it being gone already.
pub(crate) fn remove_tree(path: &Path) -> io::Result<()> {
    match fs::remove_dir_all(path) {
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(()),
        other => other,
    }
}

/// `dest/<parts>` as a path.
pub(crate) fn join_parts(dest: &Path, rel: &str) -> PathBuf {
    let mut p = dest.to_path_buf();
    for c in rel.split('/').filter(|c| !c.is_empty()) {
        p.push(c);
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn tmp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "sigil-ms-extract-{tag}-{}-{}",
            std::process::id(),
            super::super::layout::nonce()
        ));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    fn zip_bytes(members: &[(&str, &[u8])]) -> Vec<u8> {
        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, data) in members {
            if name.ends_with('/') {
                w.add_directory(*name, opts).unwrap();
            } else {
                w.start_file(*name, opts).unwrap();
                w.write_all(data).unwrap();
            }
        }
        w.finish().unwrap().into_inner()
    }

    fn spec(format: ArchiveFormat, strip: u32, include: &[&str]) -> ArchiveSpec {
        ArchiveSpec {
            format,
            strip,
            include: include.iter().map(|s| s.to_string()).collect(),
        }
    }

    fn write(dir: &Path, name: &str, bytes: &[u8]) -> PathBuf {
        let p = dir.join(name);
        fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn member_names_are_validated() {
        assert_eq!(member_path("a/b/c.txt", 0).unwrap(), Some(vec!["a".into(), "b".into(), "c.txt".into()]));
        assert_eq!(member_path("./top/x", 1).unwrap(), Some(vec!["x".into()]));
        assert_eq!(member_path("top/", 1).unwrap(), None);
        assert_eq!(member_path("top\\sub\\f.bin", 1).unwrap(), Some(vec!["sub".into(), "f.bin".into()]));
        for bad in [
            "../evil.txt",
            "a/../../evil",
            "/etc/passwd",
            "C:/Windows/evil.dll",
            "c:evil",
            "a/NUL.txt",
            "a/con",
            "a/b:stream",
            "a/trailing.",
            "a/bad*name",
        ] {
            assert!(member_path(bad, 0).is_err(), "{bad} must be rejected");
        }
    }

    #[test]
    fn globs_match_segments() {
        assert!(glob_match("bin/*.dll", "bin/cudnn64_9.dll"));
        assert!(!glob_match("bin/*.dll", "bin/x64/cudnn64_9.dll"));
        assert!(glob_match("bin/**/*.dll", "bin/x64/cudnn64_9.dll"));
        assert!(glob_match("bin/**/*.dll", "bin/a.dll"));
        assert!(glob_match("**", "anything/at/all"));
        assert!(glob_match("model.?nnx", "model.onnx"));
        assert!(!glob_match("*.onnx", "dir/model.onnx"));
    }

    #[test]
    fn zip_extracts_with_strip_include_and_hashes() {
        let d = tmp("zip-ok");
        let bytes = zip_bytes(&[
            ("pack/", b""),
            ("pack/bin/", b""),
            ("pack/bin/a.dll", b"AAAA"),
            ("pack/bin/b.txt", b"BB"),
            ("pack/LICENSE", b"L"),
        ]);
        let archive = write(&d, "a.zip", &bytes);
        let dest = d.join("out");
        fs::create_dir_all(&dest).unwrap();
        let files = extract_archive(
            &archive,
            &spec(ArchiveFormat::Zip, 1, &["bin/*.dll", "LICENSE"]),
            &dest,
            &ExtractLimits::default(),
            &|| false,
        )
        .unwrap();
        let mut names: Vec<_> = files.iter().map(|f| f.path.as_str()).collect();
        names.sort();
        assert_eq!(names, ["LICENSE", "bin/a.dll"]);
        assert_eq!(fs::read(dest.join("bin").join("a.dll")).unwrap(), b"AAAA");
        assert!(!dest.join("bin").join("b.txt").exists());
        let a = files.iter().find(|f| f.path == "bin/a.dll").unwrap();
        assert_eq!(a.sha256, super::super::verify::sha256_hex(b"AAAA"));
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn zip_slip_absolute_and_symlinks_are_rejected() {
        let d = tmp("zip-bad");
        let dest = d.join("out");
        fs::create_dir_all(&dest).unwrap();
        for (i, name) in ["../evil.txt", "ok/../../evil.txt", "/abs.txt", "C:/x.txt"].iter().enumerate() {
            let archive = write(&d, &format!("slip{i}.zip"), &zip_bytes(&[(name, b"x")]));
            let err = extract_archive(&archive, &spec(ArchiveFormat::Zip, 0, &[]), &dest, &ExtractLimits::default(), &|| false)
                .unwrap_err();
            assert!(matches!(err, ExtractError::Rejected(_)), "{name}: {err:?}");
        }
        assert!(!d.join("evil.txt").exists());

        let mut w = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let opts = zip::write::SimpleFileOptions::default();
        w.start_file("real.txt", opts).unwrap();
        w.write_all(b"data").unwrap();
        w.add_symlink("link", "../../outside", opts).unwrap();
        let archive = write(&d, "link.zip", &w.finish().unwrap().into_inner());
        let err = extract_archive(&archive, &spec(ArchiveFormat::Zip, 0, &[]), &dest, &ExtractLimits::default(), &|| false)
            .unwrap_err();
        assert!(matches!(&err, ExtractError::Rejected(m) if m.contains("symbolic link")), "{err:?}");
        // A link outside the include set is ignored.
        let ok = extract_archive(&archive, &spec(ArchiveFormat::Zip, 0, &["real.txt"]), &dest, &ExtractLimits::default(), &|| false)
            .unwrap();
        assert_eq!(ok.len(), 1);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn ratio_entry_count_and_case_duplicates_are_capped() {
        let d = tmp("zip-caps");
        let dest = d.join("out");
        fs::create_dir_all(&dest).unwrap();
        let zeros = vec![0u8; 4 << 20];
        let archive = write(&d, "bomb.zip", &zip_bytes(&[("zeros.bin", &zeros)]));
        let strict = ExtractLimits {
            ratio_floor_bytes: 0,
            max_ratio: 50,
            ..ExtractLimits::default()
        };
        let err = extract_archive(&archive, &spec(ArchiveFormat::Zip, 0, &[]), &dest, &strict, &|| false).unwrap_err();
        assert!(matches!(&err, ExtractError::Rejected(m) if m.contains("zip bomb")), "{err:?}");
        assert!(validate_zip(&archive, &strict, &|| false).is_err());
        // The default floor lets small, highly compressible archives through.
        assert!(extract_archive(&archive, &spec(ArchiveFormat::Zip, 0, &[]), &dest, &ExtractLimits::default(), &|| false).is_ok());

        let many = write(&d, "many.zip", &zip_bytes(&[("a", b"1"), ("b", b"2"), ("c", b"3")]));
        let few = ExtractLimits {
            max_entries: 2,
            ..ExtractLimits::default()
        };
        let dest2 = d.join("out2");
        fs::create_dir_all(&dest2).unwrap();
        assert!(extract_archive(&many, &spec(ArchiveFormat::Zip, 0, &[]), &dest2, &few, &|| false).is_err());

        let dup = write(&d, "dup.zip", &zip_bytes(&[("Model.bin", b"1"), ("model.BIN", b"2")]));
        let dest3 = d.join("out3");
        fs::create_dir_all(&dest3).unwrap();
        let err = extract_archive(&dup, &spec(ArchiveFormat::Zip, 0, &[]), &dest3, &ExtractLimits::default(), &|| false)
            .unwrap_err();
        assert!(matches!(&err, ExtractError::Rejected(m) if m.contains("twice")), "{err:?}");
        let _ = fs::remove_dir_all(&d);
    }

    fn tar_with(entries: &[(&str, tar::EntryType, &[u8], Option<&str>)]) -> Vec<u8> {
        let mut b = tar::Builder::new(Vec::new());
        for (name, kind, data, link) in entries {
            let mut h = tar::Header::new_gnu();
            h.set_entry_type(*kind);
            h.set_size(data.len() as u64);
            h.set_mode(0o644);
            // Write the name bytes directly so hostile names ("../x") survive the builder's checks.
            let gnu = h.as_gnu_mut().unwrap();
            gnu.name[..name.len()].copy_from_slice(name.as_bytes());
            if let Some(l) = link {
                gnu.linkname[..l.len()].copy_from_slice(l.as_bytes());
            }
            h.set_cksum();
            b.append(&h, *data).unwrap();
        }
        b.into_inner().unwrap()
    }

    fn gz(data: &[u8]) -> Vec<u8> {
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    fn bz2(data: &[u8]) -> Vec<u8> {
        let mut e = bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
        e.write_all(data).unwrap();
        e.finish().unwrap()
    }

    #[test]
    fn tar_gz_and_bz2_extract_and_reject_links_and_traversal() {
        let d = tmp("tar");
        let good = tar_with(&[
            ("model/", tar::EntryType::Directory, b"", None),
            ("model/tokens.txt", tar::EntryType::Regular, b"a b c", None),
            ("model/sub/m.onnx", tar::EntryType::Regular, b"onnx!", None),
        ]);
        for (fmt, bytes) in [(ArchiveFormat::TarGz, gz(&good)), (ArchiveFormat::TarBz2, bz2(&good))] {
            let archive = write(&d, "m.tar", &bytes);
            let dest = d.join(format!("{fmt:?}"));
            fs::create_dir_all(&dest).unwrap();
            let files = extract_archive(&archive, &spec(fmt, 1, &[]), &dest, &ExtractLimits::default(), &|| false).unwrap();
            assert_eq!(files.len(), 2);
            assert_eq!(fs::read(dest.join("sub").join("m.onnx")).unwrap(), b"onnx!");
        }
        let cases: [(&str, Vec<u8>); 3] = [
            ("symlink", tar_with(&[("x/link", tar::EntryType::Symlink, b"", Some("/etc/passwd"))])),
            ("hardlink", tar_with(&[("x/hard", tar::EntryType::Link, b"", Some("x/other"))])),
            ("traversal", tar_with(&[("../escape.txt", tar::EntryType::Regular, b"x", None)])),
        ];
        for (what, raw) in cases {
            let archive = write(&d, "bad.tar.gz", &gz(&raw));
            let dest = d.join(format!("bad-{what}"));
            fs::create_dir_all(&dest).unwrap();
            let err = extract_archive(&archive, &spec(ArchiveFormat::TarGz, 0, &[]), &dest, &ExtractLimits::default(), &|| false)
                .unwrap_err();
            assert!(matches!(err, ExtractError::Rejected(_)), "{what}: {err:?}");
        }
        assert!(!d.join("escape.txt").exists());

        // A gzip bomb is stopped by the running ratio check, not by the declared sizes.
        let zeros = vec![0u8; 8 << 20];
        let bomb = gz(&tar_with(&[("z.bin", tar::EntryType::Regular, &zeros, None)]));
        let archive = write(&d, "bomb.tar.gz", &bomb);
        let dest = d.join("bomb");
        fs::create_dir_all(&dest).unwrap();
        let strict = ExtractLimits {
            ratio_floor_bytes: 0,
            max_ratio: 20,
            ..ExtractLimits::default()
        };
        let err = extract_archive(&archive, &spec(ArchiveFormat::TarGz, 0, &[]), &dest, &strict, &|| false).unwrap_err();
        assert!(matches!(&err, ExtractError::Rejected(m) if m.contains("zip bomb")), "{err:?}");
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn validate_zip_detects_truncation_and_corruption() {
        let d = tmp("zip-validate");
        let data: Vec<u8> = (0..200_000u32).flat_map(|i| i.to_le_bytes()).collect();
        let bytes = zip_bytes(&[("model/am/final.mdl", &data), ("model/README", b"readme")]);
        let ok = write(&d, "ok.zip", &bytes);
        let s = validate_zip(&ok, &ExtractLimits::default(), &|| false).unwrap();
        assert_eq!(s.entries, 2);
        assert_eq!(s.uncompressed_bytes, data.len() as u64 + 6);
        let truncated = write(&d, "trunc.zip", &bytes[..bytes.len() / 2]);
        assert!(validate_zip(&truncated, &ExtractLimits::default(), &|| false).is_err());
        let mut flipped = bytes.clone();
        flipped[200] ^= 0xFF;
        let corrupt = write(&d, "corrupt.zip", &flipped);
        assert!(validate_zip(&corrupt, &ExtractLimits::default(), &|| false).is_err());
        let html = write(&d, "page.zip", b"<!DOCTYPE html><html><body>Not found</body></html>");
        assert!(validate_zip(&html, &ExtractLimits::default(), &|| false).is_err());
        let _ = fs::remove_dir_all(&d);
    }
}
