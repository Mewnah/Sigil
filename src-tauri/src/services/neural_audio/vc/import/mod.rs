//! RVC importer: turns an untrusted community `.pth` / `.zip` voice into the weights of a Sigil
//! template slot (decision record: `docs/adr/0004-rvc-templates.md`).
//!
//! Pipeline ([`import_file`]):
//! 1. [`zipguard`]: the file is a zip (a `.pth` is the torch zip format; a community voice is a
//!    `.zip` holding one `.pth` and usually an `added_*.index`). Entry count, names, links,
//!    encryption, sizes and compression ratio are checked before anything is decompressed
//!    (voice archives: <= 64 entries, <= 2 GB, <= 200x). Legacy (pre-1.6) torch pickles and
//!    TorchScript archives are refused with their own codes.
//! 2. [`pickle`]: `data.pkl` runs on an allow-list pickle VM (OrderedDict, `_rebuild_tensor_v2` /
//!    `_rebuild_parameter`, float / half / bfloat16 / long / int storages, `_codecs.encode`);
//!    anything else is refused, nothing is executed. [`pth`] then bounds-checks every tensor view
//!    against its storage.
//! 3. [`meta`]: version / f0 / sample rate / vocoder / embedder from the weights and metadata,
//!    which select the template (`<templates>/<version>-<sr>k-<f0|nof0>-<vocoder>/layout.json`);
//!    the checkpoint's architecture fields must equal the template's.
//! 4. [`writer`] + [`fold`]: every template weight is written at its [`layout`] offset in
//!    `weights.f32.bin` (weight_norm folded in f64), plus the speaker table; every checkpoint tensor
//!    must be accounted for.
//! 5. [`session`]: the engine hands `weights.f32.bin` (or its f16 cast) to ONNX Runtime in memory
//!    for the template graphs; no protobuf code runs in Sigil.
//!
//! Untrusted files are meant to be parsed only inside the isolated worker process
//! ([`run_import_worker`], `Sigil.exe --headless vc-import-worker <job.json>`), which P3-B starts in
//! a Job Object with a 2-4 GB memory cap; the worker also enforces its own time limit.
//!
//! [`write_slot`] replaces a slot folder as a whole: the three result files are written into a
//! hidden sibling folder and swapped in with directory renames, so a failed or killed import leaves
//! either the previous voice or the new one, never a mix (see [`write_slot`]).
//!
//! # Frozen contract (P1-S0b)
//! - [`run_import_worker`]`(job_path) -> Result<String, String>`: body of
//!   `Sigil.exe --headless vc-import-worker <job.json>`; reads the job file ([`ImportJob`]), runs
//!   the import and returns the result JSON printed to stdout (`Err("<code>: <message>")` -> exit 1).

pub mod fold;
pub mod layout;
pub mod meta;
pub mod pickle;
pub mod pth;
pub mod session;
pub mod writer;
pub mod zipguard;

#[cfg(test)]
mod tests;

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use layout::Template;
use meta::CheckpointMeta;
use pth::TorchCheckpoint;
use writer::ImportedWeights;
use zipguard::{GuardedZip, ZipLimits};

/// Result files written into a slot folder.
pub const WEIGHTS_FILE: &str = "weights.f32.bin";
pub const SPEAKERS_FILE: &str = "speakers.f32.bin";
pub const IMPORT_RECORD: &str = "sigil-import.json";
pub const IMPORT_RECORD_FORMAT: &str = "sigil-rvc-import";

/// Largest source file accepted.
const MAX_SOURCE_BYTES: u64 = 2 << 30;
/// Largest `.pth` extracted from a voice archive.
const MAX_PTH_IN_ARCHIVE: u64 = 1 << 30;
const MAX_JOB_BYTES: u64 = 1 << 20;
const DEFAULT_TIME_LIMIT_S: u64 = 120;

/// Why an import failed (stable, machine-readable; the UI maps codes to messages).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    Io,
    Job,
    NotACheckpoint,
    LegacyFormat,
    TorchScript,
    ArchiveRejected,
    AmbiguousArchive,
    PickleRejected,
    TensorInvalid,
    MissingTensor,
    ShapeMismatch,
    NonFinite,
    UnsupportedModel,
    NoTemplate,
    LayoutInvalid,
    Session,
    /// The worker's wall-clock limit was reached (printed by its watchdog).
    TimeLimit,
    /// A template graph variant that fails the parity gate on the requested device class
    /// (`layout.json` `parity`); the engine uses the fp32 variant instead.
    PrecisionRefused,
}

impl ErrorCode {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorCode::Io => "io",
            ErrorCode::Job => "job",
            ErrorCode::NotACheckpoint => "not_a_checkpoint",
            ErrorCode::LegacyFormat => "legacy_format",
            ErrorCode::TorchScript => "torchscript",
            ErrorCode::ArchiveRejected => "archive_rejected",
            ErrorCode::AmbiguousArchive => "ambiguous_archive",
            ErrorCode::PickleRejected => "pickle_rejected",
            ErrorCode::TensorInvalid => "tensor_invalid",
            ErrorCode::MissingTensor => "missing_tensor",
            ErrorCode::ShapeMismatch => "shape_mismatch",
            ErrorCode::NonFinite => "non_finite",
            ErrorCode::UnsupportedModel => "unsupported_model",
            ErrorCode::NoTemplate => "no_template",
            ErrorCode::LayoutInvalid => "layout_invalid",
            ErrorCode::Session => "session",
            ErrorCode::TimeLimit => "time_limit",
            ErrorCode::PrecisionRefused => "precision_refused",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ImportError {
    pub code: ErrorCode,
    pub message: String,
}

impl ImportError {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self { code, message: message.into() }
    }

    pub fn io(e: std::io::Error) -> Self {
        Self::new(ErrorCode::Io, e.to_string())
    }
}

impl fmt::Display for ImportError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.code.as_str(), self.message)
    }
}

impl std::error::Error for ImportError {}

/// Where the checkpoint came from.
#[derive(Clone, Debug, Serialize)]
pub struct SourceInfo {
    pub path: PathBuf,
    pub bytes: u64,
    /// sha256 of the source file (the import cache key).
    pub sha256: String,
    /// `pth` or `zip` (a voice archive).
    pub kind: &'static str,
    /// The `.pth` entry inside a voice archive.
    pub pth_entry: Option<String>,
    /// FAISS index entries found next to it (adopted by the slot store, P3-B).
    pub index_entries: Vec<String>,
}

/// A checkpoint converted to a template's weights.
pub struct ImportedVoice {
    pub template: Template,
    pub meta: CheckpointMeta,
    pub weights: ImportedWeights,
    pub source: SourceInfo,
    pub tensors: usize,
    /// Storage type of the checkpoint's weights (`float16` for RVC's own exports).
    pub checkpoint_dtype: &'static str,
}

fn sha256_of(r: &mut (impl Read + Seek)) -> Result<(String, u64), ImportError> {
    r.seek(SeekFrom::Start(0)).map_err(ImportError::io)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = r.read(&mut buf).map_err(ImportError::io)?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    r.seek(SeekFrom::Start(0)).map_err(ImportError::io)?;
    Ok((hex::encode(h.finalize()), n))
}

fn is_torch_zip<R: Read + Seek>(zip: &GuardedZip<R>) -> bool {
    zip.entries().iter().any(|e| e.name.ends_with("/data.pkl") && e.name.matches('/').count() == 1)
}

/// Archive entries that are operating-system metadata rather than content: macOS Finder's
/// `__MACOSX/` folder and `._name` AppleDouble files, other hidden files and folders (`.DS_Store`,
/// ...), and Windows' `Thumbs.db` / `desktop.ini`. A voice archive's `.pth` and `.index` are chosen
/// among the other entries.
pub fn is_metadata_entry(name: &str) -> bool {
    let base = name.rsplit('/').find(|p| !p.is_empty()).unwrap_or("");
    name.split('/').any(|part| part == "__MACOSX" || part.starts_with('.'))
        || base.eq_ignore_ascii_case("thumbs.db")
        || base.eq_ignore_ascii_case("desktop.ini")
}

fn convert<R: Read + Seek>(
    mut ckpt: TorchCheckpoint<R>,
    templates_dir: &Path,
    source: SourceInfo,
) -> Result<ImportedVoice, ImportError> {
    let meta = meta::detect(&ckpt.meta, |k| ckpt.shape(k))?;
    let template = Template::load(templates_dir, &meta.template_id())?;
    meta::check_template(&meta, &template.layout)?;
    let weights = writer::build(&template.layout, &mut ckpt)?;
    let checkpoint_dtype = ckpt.dtype("enc_p.emb_phone.weight").map(|d| d.name()).unwrap_or("unknown");
    Ok(ImportedVoice { template, meta, weights, source, tensors: ckpt.len(), checkpoint_dtype })
}

/// Import a `.pth` or a voice `.zip` against the templates in `templates_dir` (no files written).
pub fn import_file(source: &Path, templates_dir: &Path) -> Result<ImportedVoice, ImportError> {
    let file = File::open(source).map_err(|e| ImportError::new(ErrorCode::Io, format!("{}: {e}", source.display())))?;
    let len = file.metadata().map_err(ImportError::io)?.len();
    if len > MAX_SOURCE_BYTES {
        return Err(ImportError::new(ErrorCode::ArchiveRejected, format!("{len} bytes (at most {MAX_SOURCE_BYTES})")));
    }
    let mut reader = BufReader::new(file);
    let (sha256, bytes) = sha256_of(&mut reader)?;
    let mut head = [0u8; 4];
    let got = reader.read(&mut head).map_err(ImportError::io)?;
    reader.seek(SeekFrom::Start(0)).map_err(ImportError::io)?;
    if !zipguard::looks_like_zip(&head[..got]) {
        return Err(pth::reject_non_zip(&head[..got]));
    }
    let mut info = SourceInfo { path: source.to_path_buf(), bytes, sha256, kind: "pth", pth_entry: None, index_entries: Vec::new() };
    // A .pth is itself a zip; tell it from a voice archive by its data.pkl.
    let probe = GuardedZip::open(&mut reader, ZipLimits::TORCH_CHECKPOINT);
    if matches!(&probe, Ok(z) if is_torch_zip(z)) {
        drop(probe);
        reader.seek(SeekFrom::Start(0)).map_err(ImportError::io)?;
        return convert(TorchCheckpoint::open(reader)?, templates_dir, info);
    }
    drop(probe);
    reader.seek(SeekFrom::Start(0)).map_err(ImportError::io)?;
    let mut archive = GuardedZip::open(reader, ZipLimits::VOICE_ARCHIVE)?;
    let content = || archive.entries().iter().filter(|e| !is_metadata_entry(&e.name));
    let with_ext = |ext: &str| -> Vec<String> {
        content().filter(|e| e.name.to_ascii_lowercase().ends_with(ext)).map(|e| e.name.clone()).collect()
    };
    let pths = with_ext(".pth");
    info.index_entries = with_ext(".index");
    let pth_name = match pths.as_slice() {
        [one] => one.clone(),
        [] => return Err(ImportError::new(ErrorCode::NotACheckpoint, "the archive holds no .pth voice")),
        many => {
            return Err(ImportError::new(
                ErrorCode::AmbiguousArchive,
                format!("the archive holds {} .pth files ({}); import one of them directly", many.len(), many.join(", ")),
            ))
        }
    };
    let bytes = archive.read(&pth_name, MAX_PTH_IN_ARCHIVE)?;
    if !zipguard::looks_like_zip(&bytes) {
        return Err(pth::reject_non_zip(&bytes));
    }
    info.kind = "zip";
    info.pth_entry = Some(pth_name);
    convert(TorchCheckpoint::open(Cursor::new(bytes))?, templates_dir, info)
}

fn io_at(path: &Path, e: std::io::Error) -> ImportError {
    ImportError::new(ErrorCode::Io, format!("{}: {e}", path.display()))
}

/// Write `data` to `dir/name` and flush it to disk.
fn write_synced(dir: &Path, name: &str, data: &[u8]) -> Result<(), ImportError> {
    let path = dir.join(name);
    let mut f = File::create(&path).map_err(|e| io_at(&path, e))?;
    f.write_all(data).and_then(|_| f.sync_all()).map_err(|e| io_at(&path, e))
}

/// The files an import writes into a slot folder (everything else in the folder is kept).
const SLOT_FILES: [&str; 3] = [WEIGHTS_FILE, SPEAKERS_FILE, IMPORT_RECORD];

/// Imports in their commit step (the watchdog lets a swap finish before it exits).
static COMMITTING: AtomicUsize = AtomicUsize::new(0);

struct CommitGuard;

impl CommitGuard {
    fn new() -> CommitGuard {
        COMMITTING.fetch_add(1, Ordering::SeqCst);
        CommitGuard
    }
}

impl Drop for CommitGuard {
    fn drop(&mut self) {
        COMMITTING.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Hidden siblings of a slot folder: the new contents being written, and the previous folder
/// while it is being replaced.
fn swap_dirs(out_dir: &Path) -> Result<(PathBuf, PathBuf), ImportError> {
    let name = out_dir
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| ImportError::new(ErrorCode::Job, format!("{}: not a folder name", out_dir.display())))?;
    let parent = out_dir.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    Ok((parent.join(format!(".{name}.sigil-import-new")), parent.join(format!(".{name}.sigil-import-old"))))
}

/// Move what `from` holds besides the import's own files into `to` (an adopted index, a manifest).
fn carry_over(from: &Path, to: &Path) -> Result<(), ImportError> {
    for entry in std::fs::read_dir(from).map_err(|e| io_at(from, e))? {
        let entry = entry.map_err(|e| io_at(from, e))?;
        let name = entry.file_name();
        let ours = name.to_str().is_some_and(|n| SLOT_FILES.contains(&n) || n.ends_with(".partial"));
        if !ours && !to.join(&name).exists() {
            std::fs::rename(entry.path(), to.join(&name)).map_err(|e| io_at(&entry.path(), e))?;
        }
    }
    Ok(())
}

/// Finish or undo a swap a killed import left behind, and drop an uncommitted new folder.
fn recover_slot(out_dir: &Path, staging: &Path, backup: &Path) -> Result<(), ImportError> {
    if backup.is_dir() {
        if !out_dir.exists() {
            // Killed between the two renames: the previous slot is intact in the backup.
            std::fs::rename(backup, out_dir).map_err(|e| io_at(out_dir, e))?;
        } else {
            // Killed while the extra files were moved over.
            carry_over(backup, out_dir)?;
            std::fs::remove_dir_all(backup).map_err(|e| io_at(backup, e))?;
        }
    }
    if staging.exists() {
        std::fs::remove_dir_all(staging).map_err(|e| io_at(staging, e))?;
    }
    Ok(())
}

/// Result of an import written to a slot folder (also the worker's stdout JSON).
#[derive(Debug, Serialize)]
pub struct ImportRecord {
    pub format: &'static str,
    pub format_version: u32,
    pub template: String,
    pub weights_id: String,
    pub weights: FileRecord,
    pub speakers: SpeakerRecord,
    pub meta: CheckpointMeta,
    pub source: SourceInfo,
    pub tensors: usize,
    pub checkpoint_dtype: &'static str,
    /// Whether the fp16 graphs can run this voice: its weights fit half precision and, when the
    /// job asked for the fp16 smoke decode, it stayed finite. Otherwise the engine uses fp32.
    pub fp16_ok: bool,
    /// CPU smoke decodes of the imported voice (fp32, then fp16 when requested; empty when the
    /// job skipped them).
    pub smoke: Vec<session::SmokeReport>,
    /// Why the fp16 smoke decode failed, when it did.
    pub fp16_issue: Option<String>,
    pub elapsed_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct FileRecord {
    pub file: &'static str,
    pub bytes: u64,
    pub sha256: String,
}

#[derive(Debug, Serialize)]
pub struct SpeakerRecord {
    pub file: &'static str,
    pub count: usize,
    pub dim: usize,
    pub sha256: String,
}

/// Smoke decodes of an imported voice (see [`session::smoke_inference`]).
#[derive(Clone, Debug, Default)]
pub struct Smoke {
    pub reports: Vec<session::SmokeReport>,
    pub fp16_issue: Option<String>,
}

impl Smoke {
    /// fp32 must pass (else the import fails); an fp16 failure only disables the fp16 graphs.
    /// The fp16 decode runs on ONNX Runtime's slow CPU fp16 path (~3 s for a v2 40k voice), so it
    /// is opt-in: the engine checks its first GPU fp16 output anyway.
    pub fn run(voice: &ImportedVoice, fp16: bool) -> Result<Smoke, ImportError> {
        let mut smoke = Smoke { reports: vec![session::smoke_inference(voice, session::Precision::Fp32)?], fp16_issue: None };
        if fp16 {
            match session::smoke_inference(voice, session::Precision::Fp16) {
                Ok(r) => smoke.reports.push(r),
                Err(e) => smoke.fp16_issue = Some(e.to_string()),
            }
        }
        Ok(smoke)
    }
}

/// Write `weights.f32.bin`, `speakers.f32.bin` and `sigil-import.json` into the slot folder
/// `out_dir`, replacing the three as one unit:
/// 1. they are written (and flushed) into the hidden sibling `.<slot>.sigil-import-new`;
/// 2. an existing slot folder is renamed to `.<slot>.sigil-import-old`, the new folder is renamed
///    into place, and the old folder's other files (an adopted index, a manifest) are moved over
///    before it is deleted.
///
/// A failure before step 2 leaves the slot untouched. A process killed in step 2 leaves the old
/// folder next to the slot, and the next `write_slot` for the slot restores or finishes the swap
/// before it writes. One import per slot folder at a time (the parent serialises them). On Windows
/// the swap fails, changing nothing, while another process holds a file in the slot open: the
/// engine releases a slot before it is re-imported.
pub fn write_slot(out_dir: &Path, voice: &ImportedVoice, smoke: &Smoke, elapsed: Duration) -> Result<ImportRecord, ImportError> {
    let (staging, backup) = swap_dirs(out_dir)?;
    if out_dir.exists() && !out_dir.is_dir() {
        return Err(ImportError::new(ErrorCode::Io, format!("{} is not a folder", out_dir.display())));
    }
    if let Some(parent) = staging.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io_at(parent, e))?;
    }
    recover_slot(out_dir, &staging, &backup)?;
    std::fs::create_dir(&staging).map_err(|e| io_at(&staging, e))?;
    let record = write_slot_files(&staging, voice, smoke, elapsed).and_then(|record| {
        let _commit = CommitGuard::new();
        swap_in(out_dir, &staging, &backup)?;
        Ok(record)
    });
    if record.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    record
}

/// Step 2 of [`write_slot`].
fn swap_in(out_dir: &Path, staging: &Path, backup: &Path) -> Result<(), ImportError> {
    if !out_dir.exists() {
        return std::fs::rename(staging, out_dir).map_err(|e| io_at(out_dir, e));
    }
    std::fs::rename(out_dir, backup).map_err(|e| io_at(out_dir, e))?;
    if let Err(e) = std::fs::rename(staging, out_dir) {
        // Put the previous voice back.
        let _ = std::fs::rename(backup, out_dir);
        return Err(io_at(out_dir, e));
    }
    // The new voice is in place; a failure below only delays moving the other files (the next
    // write_slot finishes it), so it does not fail the import.
    if carry_over(backup, out_dir).is_ok() {
        let _ = std::fs::remove_dir_all(backup);
    }
    Ok(())
}

fn write_slot_files(dir: &Path, voice: &ImportedVoice, smoke: &Smoke, elapsed: Duration) -> Result<ImportRecord, ImportError> {
    let w = &voice.weights;
    let speakers = writer::f32_bytes(&w.speakers);
    write_synced(dir, WEIGHTS_FILE, &w.f32_image)?;
    write_synced(dir, SPEAKERS_FILE, &speakers)?;
    let record = ImportRecord {
        format: IMPORT_RECORD_FORMAT,
        format_version: 1,
        template: voice.template.layout.template.clone(),
        weights_id: voice.template.layout.weights.weights_id.clone(),
        weights: FileRecord { file: WEIGHTS_FILE, bytes: w.f32_image.len() as u64, sha256: hex::encode(Sha256::digest(&w.f32_image)) },
        speakers: SpeakerRecord { file: SPEAKERS_FILE, count: w.n_speakers, dim: w.speaker_dim, sha256: hex::encode(Sha256::digest(&speakers)) },
        meta: voice.meta.clone(),
        source: voice.source.clone(),
        tensors: voice.tensors,
        checkpoint_dtype: voice.checkpoint_dtype,
        fp16_ok: writer::f16_image(&w.f32_image).is_ok() && smoke.fp16_issue.is_none(),
        smoke: smoke.reports.clone(),
        fp16_issue: smoke.fp16_issue.clone(),
        elapsed_ms: elapsed.as_millis() as u64,
    };
    let json = serde_json::to_vec_pretty(&record).map_err(|e| ImportError::new(ErrorCode::Io, e.to_string()))?;
    write_synced(dir, IMPORT_RECORD, &json)?;
    Ok(record)
}

/// The job file of `Sigil.exe --headless vc-import-worker <job.json>`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportJob {
    /// Job schema version (1).
    #[serde(default = "job_version")]
    pub version: u32,
    /// The `.pth` or voice `.zip` to import.
    pub source: PathBuf,
    /// Slot folder receiving `weights.f32.bin`, `speakers.f32.bin` and `sigil-import.json`.
    pub out_dir: PathBuf,
    /// Folder holding `<template id>/layout.json`. The parent should always pass it (it has
    /// Tauri's `resource_dir()`); when it is missing the worker probes the bundled locations
    /// ([`default_templates_dir`]).
    #[serde(default)]
    pub templates_dir: Option<PathBuf>,
    /// Wall-clock limit (seconds, 5..=600, default 120); the worker exits when it is exceeded.
    #[serde(default)]
    pub time_limit_s: Option<u64>,
    /// Decode a short input on the CPU (fp32) before accepting the voice (default true).
    #[serde(default = "yes")]
    pub smoke: bool,
    /// Also smoke-decode with the fp16 graphs (default false, see [`Smoke::run`]).
    #[serde(default)]
    pub smoke_fp16: bool,
}

fn yes() -> bool {
    true
}

fn job_version() -> u32 {
    1
}

/// `productName` in `tauri.conf.json`: Tauri's Linux resource folder is named after it.
const PRODUCT_NAME: &str = "Sigil";

/// Where the bundled `vc-templates` can be, for an executable in `exe_dir`: under each of Tauri's
/// resource folders (`resource_dir()`: the executable's folder on Windows; `../lib/<product>`,
/// `$APPDIR/usr/lib/<product>` (AppImage) or `/usr/lib/<product>` on Linux; `../Resources` in a
/// macOS bundle), as `resources/vc-templates` (the bundle's source path) or `vc-templates`.
fn template_dir_candidates(exe_dir: &Path) -> Vec<PathBuf> {
    #[allow(unused_mut)]
    let mut roots = vec![exe_dir.to_path_buf()];
    #[cfg(target_os = "linux")]
    {
        roots.push(exe_dir.join("..").join("lib").join(PRODUCT_NAME));
        if let Some(appdir) = std::env::var_os("APPDIR") {
            roots.push(PathBuf::from(appdir).join("usr").join("lib").join(PRODUCT_NAME));
        }
        roots.push(Path::new("/usr/lib").join(PRODUCT_NAME));
    }
    #[cfg(target_os = "macos")]
    roots.push(exe_dir.join("..").join("Resources"));
    let _ = PRODUCT_NAME;
    roots
        .into_iter()
        .flat_map(|root| [root.join("resources").join("vc-templates"), root.join("vc-templates")])
        .collect()
}

/// The bundled templates ([`template_dir_candidates`] of the running executable), then, in debug
/// builds only, the source tree (release binaries never embed the build machine's paths).
pub fn default_templates_dir() -> Option<PathBuf> {
    let exe_dir = std::env::current_exe().ok().and_then(|p| p.parent().map(Path::to_path_buf));
    let mut candidates: Vec<PathBuf> = exe_dir.as_deref().map(template_dir_candidates).unwrap_or_default();
    #[cfg(debug_assertions)]
    candidates.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("resources").join("vc-templates"));
    candidates.into_iter().find(|d| d.is_dir())
}

fn run_job(job: &ImportJob) -> Result<String, ImportError> {
    if job.version != 1 {
        return Err(ImportError::new(ErrorCode::Job, format!("job version {} is not supported", job.version)));
    }
    let templates = match &job.templates_dir {
        Some(d) => d.clone(),
        None => default_templates_dir().ok_or_else(|| ImportError::new(ErrorCode::NoTemplate, "the bundled vc-templates folder was not found"))?,
    };
    let started = Instant::now();
    let voice = import_file(&job.source, &templates)?;
    let smoke = if job.smoke { Smoke::run(&voice, job.smoke_fp16)? } else { Smoke::default() };
    let record = write_slot(&job.out_dir, &voice, &smoke, started.elapsed())?;
    serde_json::to_string_pretty(&record).map_err(|e| ImportError::new(ErrorCode::Io, e.to_string()))
}

/// Body of `Sigil.exe --headless vc-import-worker <job.json>`.
pub fn run_import_worker(job_path: &Path) -> Result<String, String> {
    let read = || -> Result<ImportJob, ImportError> {
        let len = std::fs::metadata(job_path).map_err(|e| ImportError::new(ErrorCode::Job, format!("{}: {e}", job_path.display())))?.len();
        if len > MAX_JOB_BYTES {
            return Err(ImportError::new(ErrorCode::Job, "job file too large"));
        }
        let bytes = std::fs::read(job_path).map_err(ImportError::io)?;
        serde_json::from_slice(&bytes).map_err(|e| ImportError::new(ErrorCode::Job, format!("{}: {e}", job_path.display())))
    };
    let job = read().map_err(|e| e.to_string())?;
    let limit = Duration::from_secs(job.time_limit_s.unwrap_or(DEFAULT_TIME_LIMIT_S).clamp(5, 600));
    // Watchdog: a hostile file must not keep the worker alive (the parent's Job Object is the
    // backstop). Dropping `done` (success, error or unwind) releases it. A slot swap in progress
    // (a few renames) is allowed to finish first, so the slot is never left half-swapped.
    let (done, wait) = mpsc::channel::<()>();
    let _ = std::thread::Builder::new().name("vc-import-watchdog".into()).spawn(move || {
        if let Err(mpsc::RecvTimeoutError::Timeout) = wait.recv_timeout(limit) {
            let grace = Instant::now();
            while COMMITTING.load(Ordering::SeqCst) > 0 && grace.elapsed() < Duration::from_secs(10) {
                if wait.recv_timeout(Duration::from_millis(20)) != Err(mpsc::RecvTimeoutError::Timeout) {
                    return;
                }
            }
            let e = ImportError::new(ErrorCode::TimeLimit, format!("the import took longer than {} s", limit.as_secs()));
            eprintln!("vc-import-worker: {e}");
            std::process::exit(1);
        }
    });
    let result = run_job(&job).map_err(|e| e.to_string());
    drop(done);
    result
}
