//! Installing, importing, verifying and removing entries.
//!
//! An install never exposes a partial or unverified file:
//! 1. every file is obtained and verified first (adopted in place from a legacy folder after a hash
//!    check, copied from [`super::LOCAL_ASSETS_ENV`], or downloaded), into `models/.downloads/`;
//! 2. archives are extracted with the safety caps into `models/.staging/<entry>.<pid>.<nonce>/`,
//!    plain files are moved there;
//! 3. the staging folder is renamed into place in one step (legacy families: each file is renamed
//!    into the shared legacy folder), and only then is `installed.json` written (atomically).
//!
//! A failure before step 3 leaves the old install (if any) untouched; the staging folder is
//! removed, and only a cancel or a transient network error keeps the partial download.

use super::download::{self, DownloadError, DownloadRequest};
use super::extract::{self, ExtractError};
use super::layout::{
    self, describe_io, join_member, join_rel, move_file, nonce, read_record, sanitize_dir_name, trash_dir, trash_or_defer,
    unix_ms, InstalledRecord, Origin, RecordedContent, RecordedFile, RECORD_SCHEMA,
};
use super::legacy;
use super::mirrors::{self, ResolvedSource};
use super::queue::{CancelToken, Job, JobResult, CANCELLED};
use super::store::{partial_key, Store};
use super::verify::{hash_file, Digest, Expected, FileHasher, HashError};
use super::{CatalogEntry, CatalogFile, EntryState, EntryStatus, PublishStatus, StoreConfig};
use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Why an install stopped.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum InstallError {
    Cancelled,
    Failed(String),
}

impl From<DownloadError> for InstallError {
    fn from(e: DownloadError) -> Self {
        match e {
            DownloadError::Cancelled => InstallError::Cancelled,
            DownloadError::Failed(m) => InstallError::Failed(m),
        }
    }
}

fn failed(msg: impl Into<String>) -> InstallError {
    InstallError::Failed(msg.into())
}

/// File extensions never fetched without a pinned hash (trust on first use is for data only).
const EXECUTABLE_EXTENSIONS: &[&str] = &[
    "exe", "dll", "sys", "com", "bat", "cmd", "ps1", "psm1", "vbs", "vbe", "js", "jse", "wsf", "msi", "msp", "scr",
    "cpl", "so", "dylib", "sh", "jar", "lnk", "node",
];

pub(crate) fn is_executable_name(path: &str) -> bool {
    path.rsplit('.')
        .next()
        .filter(|ext| *ext != path)
        .is_some_and(|ext| EXECUTABLE_EXTENSIONS.iter().any(|x| x.eq_ignore_ascii_case(ext)))
}

/// Whether an install record still matches the catalog: every pinned file has the recorded hash;
/// unpinned files and archive entries also need the same version.
pub(crate) fn record_matches(entry: &CatalogEntry, rec: &InstalledRecord) -> bool {
    if rec.files.len() != entry.files.len() {
        return false;
    }
    let has_archive = entry.files.iter().any(|f| f.archive.is_some());
    for f in &entry.files {
        let Some(r) = rec.files.iter().find(|r| r.path == f.path) else {
            return false;
        };
        if r.archive != f.archive.is_some() {
            return false;
        }
        let ok = match Expected::of(f) {
            Expected::Sha256(h) => r.sha256.as_deref() == Some(h.as_str()),
            Expected::GitOid(o) => r.git_oid.as_deref() == Some(o.as_str()),
            Expected::None => rec.version == entry.version && (f.size == 0 || r.size == f.size),
        };
        if !ok {
            return false;
        }
    }
    !has_archive || rec.version == entry.version
}

fn title(entry: &CatalogEntry) -> &str {
    entry.title.as_deref().unwrap_or(&entry.id)
}

/// Expected size of a catalog file (`size` 0 on an unpinned file means "unknown").
pub(crate) fn expected_size(file: &CatalogFile) -> Option<u64> {
    (Expected::of(file).is_pinned() || file.size > 0).then_some(file.size)
}

// ---------------------------------------------------------------------------------------------
// Job entry point
// ---------------------------------------------------------------------------------------------

/// Body of an install job: wait for requirements, then a slot, then install.
pub(crate) async fn run_job(store: &Arc<Store>, job: &Arc<Job>, entry: &CatalogEntry) -> JobResult {
    for dep in &job.deps {
        let r = tokio::select! {
            r = dep.wait() => r,
            _ = job.cancel.cancelled() => return Err(CANCELLED.into()),
        };
        match r {
            Ok(_) => {}
            Err(e) if e == CANCELLED => return Err(format!("{} (required) was cancelled", dep.id)),
            Err(e) => return Err(format!("{} (required) could not be installed: {e}", dep.id)),
        }
    }
    let Some(_slot) = store.queue.acquire(job.seq, &job.cancel).await else {
        return Err(CANCELLED.into());
    };
    match install_entry(store, entry, &job.cancel).await {
        Ok(p) => Ok(p),
        Err(InstallError::Cancelled) => Err(CANCELLED.into()),
        Err(InstallError::Failed(m)) => Err(m),
    }
}

// ---------------------------------------------------------------------------------------------
// Planning and acquiring files
// ---------------------------------------------------------------------------------------------

struct FilePlan {
    expected: Expected,
    size: Option<u64>,
    partial: PathBuf,
    sources: Vec<ResolvedSource>,
    local: Option<PathBuf>,
    legacy_existing: Option<PathBuf>,
}

/// A verified file, ready to commit.
enum Artifact {
    /// Already at its final place in a legacy folder (adopted).
    InPlace { size: u64, digest: Digest },
    /// A verified file in `models/.downloads/` (or an import temp) to move or extract.
    Staged { path: PathBuf, size: u64, digest: Digest, origin: Origin },
}

impl Artifact {
    fn digest(&self) -> &Digest {
        match self {
            Artifact::InPlace { digest, .. } | Artifact::Staged { digest, .. } => digest,
        }
    }
    fn size(&self) -> u64 {
        match self {
            Artifact::InPlace { size, .. } | Artifact::Staged { size, .. } => *size,
        }
    }
}

/// `<local assets>/<id>/<path>`, then `<local assets>/<path>`, then the source file name.
fn find_local_asset(root: &Path, entry: &CatalogEntry, file: &CatalogFile) -> Option<PathBuf> {
    let mut candidates = Vec::new();
    if let Ok(p) = join_rel(&extract::join_parts(root, &entry.id), &file.path) {
        candidates.push(p);
    }
    if let Ok(p) = join_rel(root, &file.path) {
        candidates.push(p);
    }
    if let Some(name) = file
        .sources
        .first()
        .and_then(|s| s.url.rsplit('/').next())
        .filter(|n| layout::valid_rel_path(n))
    {
        candidates.push(root.join(name));
    }
    candidates.into_iter().find(|p| p.is_file())
}

fn plan_file(store: &Store, entry: &CatalogEntry, file: &CatalogFile, config: &StoreConfig) -> Result<FilePlan, InstallError> {
    let expected = Expected::of(file);
    let size = expected_size(file);
    let partial = store.layout.downloads().join(format!("{}.partial", partial_key(entry, file)));
    let local = store
        .settings
        .local_assets
        .as_deref()
        .and_then(|root| find_local_asset(root, entry, file));
    let legacy_existing = legacy::legacy_folder(&entry.family)
        .and_then(|folder| join_rel(&store.layout.app_data.join(folder), &file.path).ok())
        .filter(|p| fs::metadata(p).is_ok_and(|m| m.is_file() && size.is_none_or(|s| m.len() == s)));
    let sources = mirrors::sources_for(entry, file, config, store.settings.validate);
    if local.is_none() && legacy_existing.is_none() {
        if sources.is_empty() {
            return Err(failed(if entry.status == PublishStatus::Unpublished {
                // mirrors::sources_for offers the custom mirror only for hash-pinned files.
                let mirror_note = if config.mirror == super::MirrorPreference::Custom && !expected.is_pinned() {
                    "; a custom mirror serves only files the catalog pins by hash"
                } else {
                    ""
                };
                format!(
                    "{} is not published for download yet: import it from a file in the Models tab (or set {} to a folder that has it){mirror_note}",
                    title(entry),
                    super::LOCAL_ASSETS_ENV
                )
            } else {
                format!("{} has no usable download source for {}", title(entry), file.path)
            }));
        }
        if !expected.is_pinned() {
            if !store.settings.allow_tofu_network {
                return Err(failed(format!(
                    "{} is not hash-pinned in this catalog, and release builds only download pinned files",
                    file.path
                )));
            }
            if is_executable_name(&file.path) {
                return Err(failed(format!(
                    "refusing to download {} without a pinned hash (executable files must be pinned)",
                    file.path
                )));
            }
        }
    }
    Ok(FilePlan {
        expected,
        size,
        partial,
        sources,
        local,
        legacy_existing,
    })
}

/// Hash a file on a worker thread.
async fn hash_blocking(path: PathBuf, expected: Expected, cancel: &CancelToken) -> Result<Digest, InstallError> {
    let c = cancel.clone();
    let label = path.display().to_string();
    tokio::task::spawn_blocking(move || hash_file(&path, &expected, &|| c.is_cancelled()))
        .await
        .map_err(|e| failed(format!("hash task failed: {e}")))?
        .map_err(|e| match e {
            HashError::Cancelled => InstallError::Cancelled,
            HashError::Io(e) => failed(describe_io(&format!("read {label}"), &e)),
        })
}

/// Stream `src` into `dst` (created or truncated) while hashing; check size and hash.
fn copy_verified(
    src: &Path,
    dst: &Path,
    expected: &Expected,
    size: Option<u64>,
    cancelled: &dyn Fn() -> bool,
) -> Result<(u64, Digest), InstallError> {
    let mut input = File::open(src).map_err(|e| failed(describe_io(&format!("open {}", src.display()), &e)))?;
    let len = input.metadata().map_err(|e| failed(describe_io("read the file size", &e)))?.len();
    if let Some(s) = size {
        if len != s {
            return Err(failed(format!(
                "{} has {len} bytes, the catalog expects {s} (a different file or version?)",
                src.display()
            )));
        }
    }
    if let Some(dir) = dst.parent() {
        fs::create_dir_all(dir).map_err(|e| failed(describe_io("create the downloads folder", &e)))?;
    }
    let mut out = download::open_partial(dst).map_err(|e| failed(describe_io("create the import copy", &e)))?;
    let result = (|| -> Result<(u64, Digest), InstallError> {
        out.set_len(0).map_err(|e| failed(describe_io("reset the import copy", &e)))?;
        out.seek(SeekFrom::Start(0)).map_err(|e| failed(describe_io("reset the import copy", &e)))?;
        let mut hasher = FileHasher::new(expected, len);
        let mut buf = vec![0u8; super::verify::HASH_CHUNK];
        let mut copied: u64 = 0;
        loop {
            if cancelled() {
                return Err(InstallError::Cancelled);
            }
            let n = match input.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(failed(describe_io(&format!("read {}", src.display()), &e))),
            };
            hasher.update(&buf[..n]);
            out.write_all(&buf[..n]).map_err(|e| failed(describe_io("write the import copy", &e)))?;
            copied += n as u64;
        }
        out.sync_all().map_err(|e| failed(describe_io("flush the import copy", &e)))?;
        let digest = hasher.finish();
        if !expected.matches(&digest) {
            return Err(failed(format!(
                "{} does not match the catalog ({} expected, got {})",
                src.display(),
                expected.describe(),
                digest.describe()
            )));
        }
        Ok((copied, digest))
    })();
    drop(out);
    if result.is_err() {
        download::remove_partial(dst);
    }
    result
}

async fn copy_verified_blocking(
    src: PathBuf,
    dst: PathBuf,
    expected: Expected,
    size: Option<u64>,
    cancel: &CancelToken,
) -> Result<(u64, Digest), InstallError> {
    let c = cancel.clone();
    tokio::task::spawn_blocking(move || copy_verified(&src, &dst, &expected, size, &|| c.is_cancelled()))
        .await
        .map_err(|e| failed(format!("copy task failed: {e}")))?
}

async fn acquire_file(
    store: &Store,
    entry: &CatalogEntry,
    file: &CatalogFile,
    plan: FilePlan,
    cancel: &CancelToken,
) -> Result<Artifact, InstallError> {
    if let Some(existing) = &plan.legacy_existing {
        store.set_state(&entry.id, Some((EntryState::Verifying, None)));
        let digest = hash_blocking(existing.clone(), plan.expected.clone(), cancel).await?;
        if plan.expected.matches(&digest) {
            let size = fs::metadata(existing).map(|m| m.len()).unwrap_or(0);
            return Ok(Artifact::InPlace { size, digest });
        }
        tracing::info!(
            target: "sigil::models",
            "{}: existing {} does not match the catalog ({}), downloading it",
            entry.id,
            existing.display(),
            digest.describe()
        );
    }
    if let Some(local) = &plan.local {
        store.set_state(&entry.id, Some((EntryState::Verifying, None)));
        let (size, digest) =
            copy_verified_blocking(local.clone(), plan.partial.clone(), plan.expected.clone(), plan.size, cancel)
                .await
                .map_err(|e| match e {
                    InstallError::Failed(m) => failed(format!("{} (from {}): {m}", file.path, super::LOCAL_ASSETS_ENV)),
                    other => other,
                })?;
        return Ok(Artifact::Staged {
            path: plan.partial,
            size,
            digest,
            origin: Origin::LocalAssets,
        });
    }
    if plan.sources.is_empty() {
        return Err(failed(format!("{} has no usable download source for {}", title(entry), file.path)));
    }
    store.set_state(&entry.id, Some((EntryState::Downloading, None)));
    let req = DownloadRequest {
        entry_id: &entry.id,
        file_label: &file.path,
        expected: &plan.expected,
        size: plan.size,
        sources: &plan.sources,
        partial: plan.partial.clone(),
        cross_mirror_resume: plan.expected.is_pinned(),
        allow_html: file.path.ends_with(".html") || file.path.ends_with(".htm"),
    };
    let done = download::download(store, &req, cancel).await?;
    tracing::info!(
        target: "sigil::models",
        "{}: {} downloaded and verified ({} bytes via {:?})",
        entry.id,
        file.path,
        done.size,
        done.mirror
    );
    Ok(Artifact::Staged {
        path: done.path,
        size: done.size,
        digest: done.digest,
        origin: Origin::Download,
    })
}

fn check_free_space(store: &Store, entry: &CatalogEntry, plans: &[FilePlan]) -> Result<(), InstallError> {
    let mut need: u64 = 0;
    for (f, p) in entry.files.iter().zip(plans) {
        if p.legacy_existing.is_some() {
            continue;
        }
        let have = fs::metadata(&p.partial).map(|m| m.len()).unwrap_or(0);
        let size = p.size.unwrap_or(f.size);
        need = need.saturating_add(size.saturating_sub(have));
        if f.archive.is_some() {
            // Room to extract next to the archive (the ratio cap bounds the real figure).
            need = need.saturating_add(size);
        }
    }
    if need == 0 {
        return Ok(());
    }
    let need = need.saturating_add(store.settings.free_space_margin);
    match layout::free_space(&store.layout.models) {
        Some(free) if free < need => Err(failed(format!(
            "not enough disk space for {}: needs {}, {} free",
            title(entry),
            super::events::human_bytes(need),
            super::events::human_bytes(free)
        ))),
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------------------------
// Install
// ---------------------------------------------------------------------------------------------

/// Install `entry` if it is not installed and up to date; returns its folder.
pub(crate) async fn install_entry(store: &Arc<Store>, entry: &CatalogEntry, cancel: &CancelToken) -> Result<PathBuf, InstallError> {
    if let Some((root, rec, _)) = store.installed(entry) {
        if record_matches(entry, &rec) {
            return Ok(root);
        }
    }
    let config = store.config();
    let plans = entry
        .files
        .iter()
        .map(|f| plan_file(store, entry, f, &config))
        .collect::<Result<Vec<_>, _>>()?;
    check_free_space(store, entry, &plans)?;
    let mut artifacts = Vec::with_capacity(plans.len());
    for (f, plan) in entry.files.iter().zip(plans) {
        if cancel.is_cancelled() {
            return Err(InstallError::Cancelled);
        }
        artifacts.push(acquire_file(store, entry, f, plan, cancel).await?);
    }
    let origin = if artifacts.iter().all(|a| matches!(a, Artifact::InPlace { .. })) {
        Origin::Adopted
    } else if artifacts
        .iter()
        .any(|a| matches!(a, Artifact::Staged { origin: Origin::LocalAssets, .. }))
    {
        Origin::LocalAssets
    } else {
        Origin::Download
    };
    commit(store, entry, artifacts, origin, cancel).await
}

/// Assemble and publish an install (see the module docs).
async fn commit(
    store: &Arc<Store>,
    entry: &CatalogEntry,
    artifacts: Vec<Artifact>,
    origin: Origin,
    cancel: &CancelToken,
) -> Result<PathBuf, InstallError> {
    if entry.files.iter().any(|f| f.archive.is_some()) {
        store.set_state(&entry.id, Some((EntryState::Extracting, None)));
    }
    let store2 = store.clone();
    let entry2 = entry.clone();
    let c = cancel.clone();
    tokio::task::spawn_blocking(move || commit_blocking(&store2, &entry2, artifacts, origin, &|| c.is_cancelled()))
        .await
        .map_err(|e| failed(format!("install task failed: {e}")))?
}

fn commit_blocking(
    store: &Store,
    entry: &CatalogEntry,
    artifacts: Vec<Artifact>,
    origin: Origin,
    cancelled: &dyn Fn() -> bool,
) -> Result<PathBuf, InstallError> {
    let layout = &store.layout;
    let staging = layout.staging().join(format!(
        "{}.{}.{}",
        sanitize_dir_name(&entry.id.replace('/', "--")),
        std::process::id(),
        nonce()
    ));
    fs::create_dir_all(&staging).map_err(|e| failed(describe_io("create the staging folder", &e)))?;
    let result = assemble_and_publish(store, entry, &artifacts, origin, &staging, cancelled);
    if result.is_err() {
        // Verified plain files were moved into staging: put them back so a retry needs no network.
        for (f, a) in entry.files.iter().zip(&artifacts) {
            if let (None, Artifact::Staged { path, .. }) = (&f.archive, a) {
                if let Ok(staged) = join_rel(&staging, &f.path) {
                    if staged.is_file() {
                        let _ = move_file(&staged, path);
                    }
                }
            }
        }
    }
    let _ = extract::remove_tree(&staging);
    if result.is_ok() {
        // Archives were extracted: their downloads are no longer needed.
        for (f, a) in entry.files.iter().zip(&artifacts) {
            if let (Some(_), Artifact::Staged { path, .. }) = (&f.archive, a) {
                download::remove_partial(path);
            }
        }
    }
    result
}

fn assemble_and_publish(
    store: &Store,
    entry: &CatalogEntry,
    artifacts: &[Artifact],
    origin: Origin,
    staging: &Path,
    cancelled: &dyn Fn() -> bool,
) -> Result<PathBuf, InstallError> {
    let layout = &store.layout;
    let legacy_root = legacy::legacy_folder(&entry.family).map(|f| layout.app_data.join(f));
    let mut files = Vec::with_capacity(entry.files.len());
    let mut contents: Vec<RecordedContent> = Vec::new();
    // Files to move into a legacy folder: (staged path, relative path).
    let mut staged_rel: Vec<String> = Vec::new();
    for (f, a) in entry.files.iter().zip(artifacts) {
        if cancelled() {
            return Err(InstallError::Cancelled);
        }
        let digest = a.digest();
        files.push(RecordedFile {
            path: f.path.clone(),
            size: a.size(),
            sha256: digest.sha256().map(str::to_string),
            git_oid: digest.git_oid().map(str::to_string),
            archive: f.archive.is_some(),
        });
        match (&f.archive, a) {
            (Some(spec), Artifact::Staged { path, .. }) => {
                let extracted = extract::extract_archive(path, spec, staging, &store.settings.limits, cancelled)
                    .map_err(|e| match e {
                        ExtractError::Cancelled => InstallError::Cancelled,
                        ExtractError::Rejected(m) => {
                            // Unsafe or damaged: never retry from these bytes.
                            download::remove_partial(path);
                            failed(format!("{}: {m}", f.path))
                        }
                        ExtractError::Io(m) => failed(format!("{}: {m}", f.path)),
                    })?;
                if extracted.is_empty() {
                    return Err(failed(format!("{}: the archive has no files to install", f.path)));
                }
                // Trust on first use covers data only: an unpinned archive fetched from the
                // network may not carry executables (local imports are the user's own choice).
                let from_network = matches!(a, Artifact::Staged { origin: Origin::Download, .. });
                if from_network && !Expected::of(f).is_pinned() {
                    if let Some(bad) = extracted.iter().find(|x| is_executable_name(&x.path)) {
                        download::remove_partial(path);
                        return Err(failed(format!(
                            "{}: refusing to install {} from an archive without a pinned hash",
                            f.path, bad.path
                        )));
                    }
                }
                for x in extracted {
                    staged_rel.push(x.path.clone());
                    contents.push(RecordedContent {
                        path: x.path,
                        size: x.size,
                        sha256: Some(x.sha256),
                        git_oid: None,
                    });
                }
            }
            // Adopted in place: legacy folders hold plain files only.
            (Some(_), Artifact::InPlace { .. }) => contents.push(content_of(f, a)),
            (None, Artifact::Staged { path, .. }) => {
                let dst = join_rel(staging, &f.path).map_err(failed)?;
                move_file(path, &dst).map_err(|e| failed(describe_io(&format!("stage {}", f.path), &e)))?;
                staged_rel.push(f.path.clone());
                contents.push(content_of(f, a));
            }
            (None, Artifact::InPlace { .. }) => contents.push(content_of(f, a)),
        }
    }
    if cancelled() {
        return Err(InstallError::Cancelled);
    }

    let _registry = store.registry_lock();
    let record_path = layout.record_path(entry);
    let old = read_record(&record_path, &entry.id);
    let old_root = old.as_ref().and_then(|r| layout.resolve_root(&r.root));
    let root = match &legacy_root {
        Some(root) => {
            for rel in &staged_rel {
                // Extracted member names follow the extraction rule, not the catalog path rule.
                let src = join_member(staging, rel).map_err(failed)?;
                let dst = join_member(root, rel).map_err(failed)?;
                move_file(&src, &dst).map_err(|e| failed(describe_io(&format!("install {rel}"), &e)))?;
            }
            root.clone()
        }
        None => {
            let preferred = layout.preferred_content_dir(entry);
            let target = choose_content_dir(store, entry, &preferred, old_root.as_deref());
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent).map_err(|e| failed(describe_io("create the install folder", &e)))?;
            }
            fs::rename(staging, &target).map_err(|e| failed(describe_io("move the install into place", &e)))?;
            target
        }
    };
    let rel_root = layout
        .rel_to_app_data(&root)
        .ok_or_else(|| failed("the install folder is outside the app data folder"))?;
    let pinned = entry.files.iter().all(|f| Expected::of(f).is_pinned());
    let record = InstalledRecord {
        schema: RECORD_SCHEMA,
        id: entry.id.clone(),
        family: entry.family.clone(),
        kind: entry.kind,
        version: entry.version.clone(),
        root: rel_root,
        files,
        contents,
        origin,
        pinned,
        installed_at_unix_ms: unix_ms(),
    };
    layout::write_record(&record_path, &record).map_err(|e| failed(describe_io("write installed.json", &e)))?;
    // The previous install of an ordinary entry, now unreferenced. When a file inside is in use (a
    // loaded runtime DLL) it is listed in models/.pending-trash.json and removed at the next start.
    // Never a folder holding the new install or its record (a hand-edited record naming the entry
    // folder itself).
    let old_root = old_root.filter(|r| !root.starts_with(r) && !record_path.starts_with(r) && !layout.is_legacy_root(r));
    if let Some(old_root) = old_root {
        trash_or_defer(layout, &old_root, &entry.family);
    }
    // Legacy folder: files of the old version that the new one no longer has.
    if let (Some(old), true) = (&old, legacy_root.is_some()) {
        for c in &old.contents {
            if !record.contents.iter().any(|n| n.path == c.path) {
                if let Ok(p) = join_member(&root, &c.path) {
                    let _ = fs::remove_file(p);
                }
            }
        }
    }
    Ok(root)
}

fn content_of(f: &CatalogFile, a: &Artifact) -> RecordedContent {
    RecordedContent {
        path: f.path.clone(),
        size: a.size(),
        sha256: a.digest().sha256().map(str::to_string),
        git_oid: a.digest().git_oid().map(str::to_string),
    }
}

/// The folder to rename the staging folder to: the preferred one when free (or when it is an
/// unreferenced leftover under `models/` or `runtimes/` that can be moved to the trash), else a
/// fresh sibling (`<name>~<nonce>`). A leftover that is in use (a loaded runtime DLL) is listed for
/// removal at the next start. The current install is never touched here: it stays usable until the
/// new record is written, then it is trashed. Folders the store does not own, and folders any other
/// entry's record points into, are never touched. Call with the registry lock held.
fn choose_content_dir(store: &Store, entry: &CatalogEntry, preferred: &Path, old_root: Option<&Path>) -> PathBuf {
    if !preferred.exists() {
        return preferred.to_path_buf();
    }
    let leftover = old_root != Some(preferred)
        && store.layout.is_store_root(preferred)
        && !store
            .record_roots()
            .iter()
            .any(|(id, r)| *id != entry.id && (r.starts_with(preferred) || preferred.starts_with(r)));
    if leftover && trash_or_defer(&store.layout, preferred, &entry.family) {
        return preferred.to_path_buf();
    }
    let name = preferred.file_name().and_then(|n| n.to_str()).unwrap_or("install");
    preferred.with_file_name(format!("{name}~{}", nonce()))
}

// ---------------------------------------------------------------------------------------------
// Remove, verify, import
// ---------------------------------------------------------------------------------------------

/// Delete an entry's files, record and partial downloads. A running install is cancelled first.
pub(crate) async fn remove(store: &Arc<Store>, id: &str) -> Result<EntryStatus, String> {
    let entry = store.entry(id)?.clone();
    if let Some(job) = store.queue.get(&entry.id) {
        job.cancel.cancel();
        let _ = tokio::time::timeout(std::time::Duration::from_secs(15), job.wait()).await;
    }
    let s = store.clone();
    let e = entry.clone();
    tokio::task::spawn_blocking(move || remove_blocking(&s, &e))
        .await
        .map_err(|e| format!("remove task failed: {e}"))??;
    store.set_state(&entry.id, None);
    store.status(&entry.id)
}

fn remove_blocking(store: &Store, entry: &CatalogEntry) -> Result<(), String> {
    let layout = &store.layout;
    let _g = store.registry_lock();
    let record_path = layout.record_path(entry);
    if let Some(rec) = read_record(&record_path, &entry.id) {
        if let Some(root) = layout.resolve_root(&rec.root) {
            if layout.is_legacy_root(&root) {
                for c in &rec.contents {
                    let p = join_member(&root, &c.path)?;
                    match fs::remove_file(&p) {
                        Ok(()) => {}
                        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                        Err(e) => return Err(describe_io(&format!("delete {}", c.path), &e)),
                    }
                }
            } else {
                trash_dir(layout, &root).map_err(|e| describe_io(&format!("delete {}", root.display()), &e))?;
            }
        }
    }
    match fs::remove_file(&record_path) {
        Ok(()) => {}
        Err(e) if e.kind() == io::ErrorKind::NotFound => {}
        Err(e) => return Err(describe_io("delete installed.json", &e)),
    }
    // Empty record folders up to (not including) models/.
    let mut dir = layout.entry_dir(entry);
    while dir.starts_with(&layout.models) && dir != layout.models && fs::remove_dir(&dir).is_ok() {
        match dir.parent() {
            Some(p) => dir = p.to_path_buf(),
            None => break,
        }
    }
    for f in &entry.files {
        download::remove_partial(&layout.downloads().join(format!("{}.partial", partial_key(entry, f))));
    }
    Ok(())
}

/// Re-hash every installed file. On a mismatch the entry is unregistered (never left "installed"
/// while corrupt) and the returned status is `failed` with the reason.
pub(crate) async fn verify(store: &Arc<Store>, id: &str) -> Result<EntryStatus, String> {
    let entry = store.entry(id)?.clone();
    if store.queue.get(&entry.id).is_some() {
        return Err(format!("{} is being installed right now", title(&entry)));
    }
    let Some((root, rec, _)) = store.installed(&entry) else {
        return Err(format!("{} is not installed", title(&entry)));
    };
    store.set_state(&entry.id, Some((EntryState::Verifying, None)));
    let s = store.clone();
    let e = entry.clone();
    let outcome = tokio::task::spawn_blocking(move || verify_blocking(&s, &e, &root, &rec))
        .await
        .map_err(|e| format!("verify task failed: {e}"))?;
    match outcome {
        Ok(()) => store.set_state(&entry.id, None),
        Err(problem) => {
            tracing::warn!(target: "sigil::models", "{}: verification failed: {problem}", entry.id);
            store.set_state(
                &entry.id,
                Some((EntryState::Failed, Some(format!("{problem}. Install it again to repair it.")))),
            );
        }
    }
    store.status(&entry.id)
}

fn verify_blocking(store: &Store, entry: &CatalogEntry, root: &Path, rec: &InstalledRecord) -> Result<(), String> {
    let problem = (|| {
        for c in &rec.contents {
            let expected = match (&c.sha256, &c.git_oid) {
                (Some(h), _) => Expected::Sha256(h.clone()),
                (None, Some(o)) => Expected::GitOid(o.clone()),
                (None, None) => continue,
            };
            let p = join_member(root, &c.path)?;
            match hash_file(&p, &expected, &|| false) {
                Ok(d) if expected.matches(&d) => {}
                Ok(d) => return Err(format!("{} is corrupt ({} expected, got {})", c.path, expected.describe(), d.describe())),
                Err(HashError::Io(e)) => return Err(describe_io(&format!("read {}", c.path), &e)),
                Err(HashError::Cancelled) => return Err("cancelled".into()),
            }
        }
        Ok(())
    })();
    if problem.is_err() {
        let _g = store.registry_lock();
        let _ = fs::remove_file(store.layout.record_path(entry));
        // Store-owned folders (models/ and runtimes/) go; a pack whose DLL is loaded goes at the next
        // start. Shared legacy folders keep the user's other files.
        if store.layout.is_store_root(root) {
            trash_or_defer(&store.layout, root, &entry.family);
        }
    }
    problem
}

/// Candidates for catalog file `f` inside an import folder.
fn import_candidate(dir: &Path, f: &CatalogFile) -> Option<PathBuf> {
    let mut c = Vec::new();
    if let Ok(p) = join_rel(dir, &f.path) {
        c.push(p);
    }
    if let Some(name) = f.path.rsplit('/').next() {
        c.push(dir.join(name));
    }
    if let Some(name) = f.sources.first().and_then(|s| s.url.rsplit('/').next()) {
        if layout::valid_rel_path(name) {
            c.push(dir.join(name));
        }
    }
    c.into_iter().find(|p| p.is_file())
}

/// Import an entry from a local file (single-file entries) or a folder holding its files. Pinned
/// files must match the catalog hash; unpinned ones (unpublished Sigil-built packs) are trusted
/// because the user picked them.
pub(crate) async fn import_local(store: &Arc<Store>, id: &str, path: &str) -> Result<EntryStatus, String> {
    let entry = store.entry(id)?.clone();
    let src = PathBuf::from(path.trim());
    if !src.exists() {
        return Err(format!("{} does not exist", src.display()));
    }
    let (job, created) = store.queue.get_or_create(&entry.id, Vec::new());
    if !created {
        return Err(format!("{} is being installed right now; cancel that first", title(&entry)));
    }
    store.set_state(&entry.id, Some((EntryState::Verifying, None)));
    let result = import_inner(store, &entry, &src, &job.cancel).await;
    let job_result: JobResult = match &result {
        Ok(p) => Ok(p.clone()),
        Err(InstallError::Cancelled) => Err(CANCELLED.into()),
        Err(InstallError::Failed(m)) => Err(m.clone()),
    };
    store.finish_job(&job, job_result.clone());
    job_result?;
    store.status(&entry.id)
}

async fn import_inner(store: &Arc<Store>, entry: &CatalogEntry, src: &Path, cancel: &CancelToken) -> Result<PathBuf, InstallError> {
    let mut picks: Vec<PathBuf> = Vec::with_capacity(entry.files.len());
    if src.is_dir() {
        let mut missing = Vec::new();
        for f in &entry.files {
            match import_candidate(src, f) {
                Some(p) => picks.push(p),
                None => missing.push(f.path.clone()),
            }
        }
        if !missing.is_empty() {
            return Err(failed(format!("{} is missing {}", src.display(), missing.join(", "))));
        }
    } else if entry.files.len() == 1 {
        picks.push(src.to_path_buf());
    } else {
        return Err(failed(format!(
            "{} has {} files: choose the folder that contains them",
            title(entry),
            entry.files.len()
        )));
    }
    let legacy_root = legacy::legacy_folder(&entry.family).map(|f| store.layout.app_data.join(f));
    let mut artifacts = Vec::with_capacity(picks.len());
    for (f, pick) in entry.files.iter().zip(picks) {
        let expected = Expected::of(f);
        let size = expected_size(f);
        // Importing the very file that already sits in the legacy folder: verify it in place.
        let in_place = legacy_root
            .as_ref()
            .and_then(|r| join_rel(r, &f.path).ok())
            .is_some_and(|dst| same_file(&dst, &pick));
        if in_place {
            let digest = hash_blocking(pick.clone(), expected.clone(), cancel).await?;
            if !expected.matches(&digest) {
                return Err(failed(format!(
                    "{} does not match the catalog ({} expected, got {})",
                    pick.display(),
                    expected.describe(),
                    digest.describe()
                )));
            }
            let size = fs::metadata(&pick).map(|m| m.len()).unwrap_or(0);
            artifacts.push(Artifact::InPlace { size, digest });
            continue;
        }
        let tmp = store
            .layout
            .downloads()
            .join(format!("import-{}.{}.partial", partial_key(entry, f), nonce()));
        let (size, digest) = copy_verified_blocking(pick, tmp.clone(), expected, size, cancel).await?;
        artifacts.push(Artifact::Staged {
            path: tmp,
            size,
            digest,
            origin: Origin::Import,
        });
    }
    let temps: Vec<PathBuf> = artifacts
        .iter()
        .filter_map(|a| match a {
            Artifact::Staged { path, .. } => Some(path.clone()),
            Artifact::InPlace { .. } => None,
        })
        .collect();
    let result = commit(store, entry, artifacts, Origin::Import, cancel).await;
    // Import copies are never resumed: drop whatever is left of them.
    for t in &temps {
        download::remove_partial(t);
    }
    result
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => false,
    }
}

/// Register a file that already sits in a legacy folder (Vosk zip import, adoption) as the install
/// of single-file legacy `entry`, when `digest` matches the catalog. Returns whether it did.
pub(crate) fn register_legacy_file(store: &Store, entry: &CatalogEntry, digest: &Digest, origin: Origin) -> bool {
    let [file] = entry.files.as_slice() else { return false };
    let Some(folder) = legacy::legacy_folder(&entry.family) else { return false };
    let expected = Expected::of(file);
    if !expected.is_pinned() || !expected.matches(digest) {
        return false;
    }
    let root = store.layout.app_data.join(folder);
    let Ok(path) = join_rel(&root, &file.path) else { return false };
    let Ok(meta) = fs::metadata(&path) else { return false };
    if meta.len() != file.size {
        return false;
    }
    let record = InstalledRecord {
        schema: RECORD_SCHEMA,
        id: entry.id.clone(),
        family: entry.family.clone(),
        kind: entry.kind,
        version: entry.version.clone(),
        root: folder.to_string(),
        files: vec![RecordedFile {
            path: file.path.clone(),
            size: file.size,
            sha256: digest.sha256().map(str::to_string),
            git_oid: digest.git_oid().map(str::to_string),
            archive: false,
        }],
        contents: vec![RecordedContent {
            path: file.path.clone(),
            size: file.size,
            sha256: digest.sha256().map(str::to_string),
            git_oid: digest.git_oid().map(str::to_string),
        }],
        origin,
        pinned: true,
        installed_at_unix_ms: unix_ms(),
    };
    let _g = store.registry_lock();
    layout::write_record(&store.layout.record_path(entry), &record).is_ok()
}
