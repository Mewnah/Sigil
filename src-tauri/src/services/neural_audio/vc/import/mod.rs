//! RVC importer: turns an untrusted community `.pth` / `.zip` voice into a Sigil template slot
//! (decision record: `docs/adr/0004-rvc-templates.md`).
//!
//! Pipeline ([`import_file`]):
//! 1. [`zipguard`]: the file is a zip (a `.pth` is the torch zip format; a community voice is a
//!    `.zip` holding one `.pth` and usually an `added_*.index`). Entry count, names, links,
//!    encryption, sizes and compression ratio are checked before anything is decompressed
//!    (voice archives: <= 64 entries, <= 2 GB, <= 200x). Legacy (pre-1.6) torch pickles and
//!    TorchScript archives are refused with their own codes. In an archive, OS metadata is
//!    ignored, training by-products (`G_*.pth`, `D_*.pth`, `f0G40k.pth`, ...) give way to the
//!    exported voice, and an ONNX-only archive is refused with a pointer to BYO ONNX slots.
//! 2. [`pickle`]: `data.pkl` runs on an allow-list pickle VM (OrderedDict, `_rebuild_tensor_v2` /
//!    `_rebuild_parameter`, float / half / bfloat16 / long / int storages, `_codecs.encode`);
//!    anything else is refused, nothing is executed. [`pth`] then bounds-checks every tensor view
//!    against its storage. Inference exports, training checkpoints and bare state dicts are read.
//! 3. [`meta`]: version / f0 / vocoder from the weights, sample rate and architecture from the
//!    config, embedder from the metadata ([`meta::canonical_embedder`]).
//! 4. [`select_template`]: the template folder `<templates>/<version>-<sr>k-<f0|nof0>-<vocoder>`
//!    when it exists and matches, else every template under `<templates>` is scanned: the one whose
//!    architecture fields equal the checkpoint's config (or, without a config, whose tensor shapes
//!    the weights fit, [`layout::Layout::fit`]).
//! 5. [`writer`] + [`fold`]: every template weight is written at its [`layout`] offset in
//!    `weights.f32.bin` (weight_norm folded in f64), plus the speaker table; every checkpoint tensor
//!    must be accounted for.
//! 6. [`session`]: a CPU smoke decode proves the voice produces finite, non-silent audio.
//! 7. [`load_index`]: the retrieval `.index` (the archive's, or an explicit file) is parsed and
//!    its width checked before it is adopted; a bad index is a warning, never a failed import.
//! 8. [`write_slot`]: `weights.f32.bin`, `speakers.f32.bin`, the adopted index (`added.index`),
//!    `sigil-import.json` and the `sigil-rvc-1` `manifest.json` ([`slot::write_manifest`]) replace
//!    the slot folder as one unit.
//!
//! Untrusted files are parsed only inside the isolated worker process ([`run_import_worker`],
//! `Sigil.exe --headless vc-import-worker <job.json>`), which [`worker::run_isolated`] starts in a
//! Job Object with a memory cap and a wall-clock limit; the worker also enforces its own time
//! limit. The parent keeps the slot store ([`slot`]): listing, the sha256 import cache, rename,
//! delete, duplicate and merge.
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
/// Slot store: manifests, index adoption, import cache, list / rename / delete / duplicate / merge.
pub mod slot;
/// Parent side of the isolated import worker.
pub mod worker;
pub mod writer;
pub mod zipguard;

#[cfg(test)]
mod corpus_tests;
#[cfg(test)]
mod tests;

use std::fmt;
use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, RwLock};
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
/// The slot's manifest (`manifest::RvcManifest`, format `sigil-rvc-1`).
pub const MANIFEST_FILE: &str = "manifest.json";
/// The retrieval index adopted into a slot.
pub const ADOPTED_INDEX_FILE: &str = "added.index";
/// Version of the importer's output. A slot imported by an older version is not reused by the
/// import cache (its file is imported again).
pub const IMPORTER_VERSION: u32 = 2;

/// Largest source file accepted.
const MAX_SOURCE_BYTES: u64 = 2 << 30;
/// Largest `.pth` extracted from a voice archive.
const MAX_PTH_IN_ARCHIVE: u64 = 1 << 30;
/// Largest retrieval index adopted (RVC's k-means reduced indexes are well below this).
pub const MAX_INDEX_BYTES: u64 = 1 << 30;
const MAX_JOB_BYTES: u64 = 1 << 20;
const DEFAULT_TIME_LIMIT_S: u64 = 120;
/// Longest display name kept (characters).
pub const MAX_NAME_CHARS: usize = 80;

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
    /// Every code (the order of `params_contract.json` `import_error_codes`).
    pub const ALL: [ErrorCode; 18] = [
        ErrorCode::Io,
        ErrorCode::Job,
        ErrorCode::NotACheckpoint,
        ErrorCode::LegacyFormat,
        ErrorCode::TorchScript,
        ErrorCode::ArchiveRejected,
        ErrorCode::AmbiguousArchive,
        ErrorCode::PickleRejected,
        ErrorCode::TensorInvalid,
        ErrorCode::MissingTensor,
        ErrorCode::ShapeMismatch,
        ErrorCode::NonFinite,
        ErrorCode::UnsupportedModel,
        ErrorCode::NoTemplate,
        ErrorCode::LayoutInvalid,
        ErrorCode::Session,
        ErrorCode::TimeLimit,
        ErrorCode::PrecisionRefused,
    ];

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

    /// The code named `s` (the inverse of [`ErrorCode::as_str`]).
    pub fn parse(s: &str) -> Option<ErrorCode> {
        ErrorCode::ALL.into_iter().find(|c| c.as_str() == s)
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

    /// Parse `"<code>: <message>"` (the worker's stderr, the IPC error string); `None` when the
    /// text does not start with a known code.
    pub fn parse(text: &str) -> Option<ImportError> {
        let (code, message) = text.split_once(':')?;
        let code = ErrorCode::parse(code.trim())?;
        Some(ImportError::new(code, message.trim()))
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
    /// FAISS index entries found next to it.
    pub index_entries: Vec<String>,
    /// The index entry chosen for adoption ([`choose_index`]).
    pub index_entry: Option<String>,
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

/// sha256 of a file on disk (the import cache key).
pub fn sha256_file(path: &Path) -> Result<(String, u64), ImportError> {
    let file = File::open(path).map_err(|e| io_at(path, e))?;
    sha256_of(&mut BufReader::new(file))
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

/// The file name of an archive entry or path (after the last `/` or `\`).
fn base_name(name: &str) -> &str {
    name.rsplit(['/', '\\']).find(|p| !p.is_empty()).unwrap_or(name)
}

/// The lower-case stem of a file name (`MyVoice.pth` -> `myvoice`).
fn stem_lower(name: &str) -> String {
    let base = base_name(name);
    base.rsplit_once('.').map_or(base, |(s, _)| s).to_ascii_lowercase()
}

/// Step number of a training checkpoint name (`G_2333.pth` -> 2333; `G_latest.pth` -> max), or
/// `None` for anything else.
fn generator_step(name: &str) -> Option<u64> {
    let stem = stem_lower(name);
    let rest = stem.strip_prefix("g_")?;
    if rest == "latest" {
        return Some(u64::MAX);
    }
    rest.parse().ok()
}

/// A training by-product or a pretrained base model rather than an exported voice: `G_2333.pth`,
/// `D_2333.pth`, `G_latest.pth`, `f0G40k.pth`, `f0D48k.pth`, `G32k.pth`, `D40k.pth`.
pub fn is_training_artifact(name: &str) -> bool {
    if !base_name(name).to_ascii_lowercase().ends_with(".pth") {
        return false;
    }
    let stem = stem_lower(name);
    let stem = stem.strip_prefix("f0").unwrap_or(&stem);
    let Some(rest) = stem.strip_prefix('g').or_else(|| stem.strip_prefix('d')) else {
        return false;
    };
    let rest = rest.strip_prefix('_').unwrap_or(rest);
    if rest == "latest" {
        return true;
    }
    let digits = rest.strip_suffix('k').unwrap_or(rest);
    !digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit())
}

/// A weights file saved during training, `(voice name, epoch, step)` (lower case): RVC WebUI and
/// Mangio's `<name>_e<epoch>_s<step>.pth`, Applio's `<name>_<epoch>e_<step>s.pth` (and its
/// `..._best_epoch.pth`). `None` for any other name.
fn epoch_save(name: &str) -> Option<(String, u64, u64)> {
    let stem = stem_lower(name);
    let stem = stem.strip_suffix("_best_epoch").unwrap_or(&stem);
    let (rest, step) = stem.rsplit_once('_')?;
    let (voice, epoch) = rest.rsplit_once('_')?;
    let number = |s: &str| (!s.is_empty() && s.len() <= 12 && s.bytes().all(|b| b.is_ascii_digit())).then(|| s.parse::<u64>().ok()).flatten();
    let (e, s) = match (epoch.strip_suffix('e'), step.strip_suffix('s')) {
        (Some(e), Some(s)) if number(e).is_some() && number(s).is_some() => (e, s),
        _ => (epoch.strip_prefix('e')?, step.strip_prefix('s')?),
    };
    if voice.is_empty() {
        return None;
    }
    Some((voice.to_string(), number(e)?, number(s)?))
}

/// The voice's name in a file name: the lower-case stem without a training save's epoch / step
/// suffix ([`epoch_save`]). Index names are matched against it.
pub(crate) fn voice_stem(name: &str) -> String {
    epoch_save(name).map_or_else(|| stem_lower(name), |(voice, _, _)| voice)
}

/// Several `.pth` files that are one voice saved at different training stages (the same voice
/// name, [`epoch_save`] suffixes, at most one file without a suffix): the final export (the one
/// without a suffix) or else the save with the highest step, and a note saying which. `None` when
/// the files are not one voice or no single file is the latest.
fn latest_save(voices: &[&String]) -> Option<(String, String)> {
    let mut name: Option<String> = None;
    let mut finals = Vec::new();
    let mut saves = Vec::new();
    for v in voices {
        let (voice, save) = match epoch_save(v) {
            Some((voice, e, s)) => (voice, Some((s, e, !stem_lower(v).ends_with("_best_epoch")))),
            None => (stem_lower(v), None),
        };
        if name.get_or_insert_with(|| voice.clone()) != &voice {
            return None;
        }
        match save {
            Some(rank) => saves.push((rank, *v)),
            None => finals.push(*v),
        }
    }
    let total = voices.len();
    let chosen = match finals.as_slice() {
        [one] => (*one).clone(),
        [] => {
            saves.sort();
            let (best, file) = saves.last()?;
            if saves.iter().filter(|(r, _)| r == best).count() > 1 {
                return None;
            }
            (*file).clone()
        }
        _ => return None,
    };
    let what = match epoch_save(&chosen) {
        Some((_, e, s)) => format!("the latest save (epoch {e}, step {s})"),
        None => "the final export".to_string(),
    };
    let note = format!("the archive holds {total} saves of the voice {}; imported {what}, {}", name.unwrap_or_default(), base_name(&chosen));
    Some((chosen, note))
}

/// The voice among the `.pth` entries of an archive, and a note when the choice needs one: the one
/// exported voice when training by-products sit next to it; when one voice was saved at several
/// training stages, its final export or latest save ([`latest_save`]); else the latest generator
/// training checkpoint (`G_<step>.pth`).
pub fn choose_pth(pths: &[String]) -> Result<(String, Option<String>), ImportError> {
    let voices: Vec<&String> = pths.iter().filter(|p| !is_training_artifact(p)).collect();
    match voices.as_slice() {
        [one] => return Ok(((*one).clone(), None)),
        [] => {}
        many => {
            if let Some((chosen, note)) = latest_save(many) {
                return Ok((chosen, Some(note)));
            }
            let names: Vec<&str> = many.iter().map(|s| s.as_str()).collect();
            return Err(ImportError::new(
                ErrorCode::AmbiguousArchive,
                format!("the archive holds {} voices ({}); import one of them directly", many.len(), names.join(", ")),
            ));
        }
    }
    if let Some(latest) = pths.iter().filter_map(|p| generator_step(p).map(|s| (s, p))).max_by_key(|(s, _)| *s) {
        return Ok((latest.1.clone(), None));
    }
    match pths {
        [] => Err(ImportError::new(ErrorCode::NotACheckpoint, "the archive holds no .pth voice")),
        _ => Err(ImportError::new(
            ErrorCode::NotACheckpoint,
            format!("the archive holds only training by-products ({}), not a voice", pths.join(", ")),
        )),
    }
}

/// The retrieval index to adopt among `names` (archive entries or file names) for the voice
/// `pth_name`: RVC's `added_*` index (a `trained_*` index holds no vectors and is never adopted),
/// preferring one whose name contains the voice's name. Returns the choice and a warning when the
/// choice was not obvious.
pub fn choose_index(names: &[String], pth_name: &str) -> (Option<String>, Option<String>) {
    let rank = |n: &str| {
        let b = base_name(n).to_ascii_lowercase();
        if b.starts_with("trained") {
            None
        } else if b.starts_with("added") {
            Some(0)
        } else {
            Some(1)
        }
    };
    let Some(best) = names.iter().filter_map(|n| rank(n)).min() else {
        return (None, None);
    };
    let mut pool: Vec<&String> = names.iter().filter(|n| rank(n) == Some(best)).collect();
    pool.sort();
    if let [one] = pool.as_slice() {
        return (Some((*one).clone()), None);
    }
    let stem = voice_stem(pth_name);
    let matching: Vec<&&String> = if stem.len() >= 3 {
        pool.iter().filter(|n| base_name(n).to_ascii_lowercase().contains(&stem)).collect()
    } else {
        Vec::new()
    };
    let chosen = match matching.as_slice() {
        [one] => (**one).clone(),
        _ => pool[0].clone(),
    };
    let warning = format!("{} retrieval indexes found; adopted {}", pool.len(), base_name(&chosen));
    (Some(chosen), Some(warning))
}

fn convert<R: Read + Seek>(
    mut ckpt: TorchCheckpoint<R>,
    templates_dir: &Path,
    source: SourceInfo,
) -> Result<ImportedVoice, ImportError> {
    let mut meta = meta::detect(&ckpt.meta, |k| ckpt.shape(k), ckpt.tensor_names(), ckpt.kind)?;
    let template = select_template(templates_dir, &mut meta, &ckpt)?;
    let weights = writer::build(&template.layout, &mut ckpt)?;
    let checkpoint_dtype = ckpt.dtype("enc_p.emb_phone.weight").map(|d| d.name()).unwrap_or("unknown");
    Ok(ImportedVoice { template, meta, weights, source, tensors: ckpt.len(), checkpoint_dtype })
}

/// The template of a checkpoint (see the module docs, step 4). A checkpoint without a config gets
/// the sample rate of the template its weights fit.
pub fn select_template<R: Read + Seek>(
    templates_dir: &Path,
    meta: &mut CheckpointMeta,
    ckpt: &TorchCheckpoint<R>,
) -> Result<Template, ImportError> {
    // Fast path: the canonical folder (every standard RVC / Applio config has its own id).
    if meta.has_config() && meta.sample_rate != 0 {
        let id = meta.template_id();
        if templates_dir.join(&id).join(layout::LAYOUT_FILE).is_file() {
            let t = Template::load(templates_dir, &id)?;
            if meta::check_template(meta, &t.layout).is_ok() {
                return Ok(t);
            }
        }
    }
    let (all, errors) = Template::scan(templates_dir);
    let candidates: Vec<Template> = all.into_iter().filter(|t| meta::covers(meta, &t.layout)).collect();
    if candidates.is_empty() {
        let unreadable = match errors.first() {
            Some(e) => format!("; {} template folder(s) could not be read ({e})", errors.len()),
            None => String::new(),
        };
        return Err(ImportError::new(
            ErrorCode::NoTemplate,
            format!("Sigil has no template for a {} voice yet (looked in {}){unreadable}", meta.template_id(), templates_dir.display()),
        ));
    }
    if meta.has_config() {
        let mut first_err = None;
        for t in candidates {
            match meta::check_template(meta, &t.layout) {
                Ok(()) => return Ok(t),
                Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
        return Err(first_err.expect("at least one candidate was checked"));
    }
    // No config: the tensor shapes decide (a declared sr already narrowed the candidates).
    let mut misfit = None;
    let mut fitting = Vec::new();
    for t in candidates {
        match t.layout.fit(ckpt.tensor_names(), |k| ckpt.shape(k)) {
            Ok(()) => fitting.push(t),
            Err(e) => {
                misfit.get_or_insert(e);
            }
        }
    }
    match fitting.len() {
        0 => Err(ImportError::new(
            ErrorCode::UnsupportedModel,
            format!("the weights fit none of Sigil's {} templates ({})", meta.template_id(), misfit.unwrap_or_default()),
        )),
        1 => {
            let t = fitting.remove(0);
            if meta.sample_rate == 0 {
                meta.sample_rate = t.layout.arch.sample_rate;
                meta.warnings.push(format!("sample rate {} Hz inferred from the weights", meta.sample_rate));
            }
            meta::check_template(meta, &t.layout)?;
            Ok(t)
        }
        _ => {
            let ids: Vec<&str> = fitting.iter().map(|t| t.layout.template.as_str()).collect();
            Err(ImportError::new(
                ErrorCode::UnsupportedModel,
                format!(
                    "the checkpoint does not say its sample rate and its weights fit several templates ({}); import the voice RVC / Applio exported to its weights folder instead",
                    ids.join(", ")
                ),
            ))
        }
    }
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
    let mut info = SourceInfo {
        path: source.to_path_buf(),
        bytes,
        sha256,
        kind: "pth",
        pth_entry: None,
        index_entries: Vec::new(),
        index_entry: None,
    };
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
    if pths.is_empty() {
        if let Some(onnx) = with_ext(".onnx").first() {
            return Err(ImportError::new(
                ErrorCode::NotACheckpoint,
                format!(
                    "the archive holds an ONNX voice ({}), not a .pth; unzip it into a folder under voice-changer/models and pick that folder as a bring-your-own ONNX voice",
                    base_name(onnx)
                ),
            ));
        }
    }
    let (pth_name, note) = choose_pth(&pths)?;
    let bytes = archive.read(&pth_name, MAX_PTH_IN_ARCHIVE)?;
    if !zipguard::looks_like_zip(&bytes) {
        return Err(pth::reject_non_zip(&bytes));
    }
    info.kind = "zip";
    info.index_entry = choose_index(&info.index_entries, &pth_name).0;
    info.pth_entry = Some(pth_name);
    let mut voice = convert(TorchCheckpoint::open(Cursor::new(bytes))?, templates_dir, info)?;
    voice.meta.warnings.extend(note);
    Ok(voice)
}

// ---- retrieval index ------------------------------------------------------------------------------

/// An index checked for adoption into the slot.
pub struct AdoptedIndex {
    pub bytes: Vec<u8>,
    /// `archive:<entry>` or `file:<name>`.
    pub origin: String,
    pub dim: usize,
}

/// Check an index: FAISS layout (`vc::faiss`) and the voice's feature width. A parser panic is
/// contained (the index is optional; the voice imports without it).
fn check_index(bytes: &[u8], emb_channels: usize) -> Result<usize, String> {
    let parsed = std::panic::catch_unwind(|| crate::services::neural_audio::vc::faiss::FaissIndex::from_bytes(bytes))
        .map_err(|_| "the index parser failed".to_string())??;
    let dim = parsed.dim();
    if dim != emb_channels {
        return Err(format!("it holds {dim}-channel features, the voice needs {emb_channels}"));
    }
    Ok(dim)
}

/// The retrieval index for `voice`: the file `explicit` when given, else the archive entry chosen
/// at import ([`SourceInfo::index_entry`]). Every problem is a warning (the voice works without an
/// index), returned with the index.
pub fn load_index(voice: &ImportedVoice, explicit: Option<&Path>) -> (Option<AdoptedIndex>, Vec<String>) {
    let emb = voice.meta.emb_channels();
    let mut warnings = Vec::new();
    let (bytes, origin) = if let Some(path) = explicit {
        let name = base_name(&path.to_string_lossy()).to_string();
        let read = std::fs::metadata(path)
            .map_err(|e| e.to_string())
            .and_then(|m| {
                if !m.is_file() {
                    Err("not a file".to_string())
                } else if m.len() > MAX_INDEX_BYTES {
                    Err(format!("{} bytes (at most {MAX_INDEX_BYTES})", m.len()))
                } else {
                    std::fs::read(path).map_err(|e| e.to_string())
                }
            });
        match read {
            Ok(b) => (b, format!("file:{name}")),
            Err(e) => {
                warnings.push(format!("the retrieval index {name} was not adopted: {e}"));
                return (None, warnings);
            }
        }
    } else {
        let (Some(entry), "zip") = (voice.source.index_entry.as_deref(), voice.source.kind) else {
            return (None, warnings);
        };
        let read = File::open(&voice.source.path)
            .map_err(ImportError::io)
            .and_then(|f| GuardedZip::open(BufReader::new(f), ZipLimits::VOICE_ARCHIVE))
            .and_then(|mut z| z.read(entry, MAX_INDEX_BYTES));
        match read {
            Ok(b) => (b, format!("archive:{entry}")),
            Err(e) => {
                warnings.push(format!("the retrieval index {} was not adopted: {}", base_name(entry), e.message));
                return (None, warnings);
            }
        }
    };
    if let (_, Some(w)) = choose_index(&voice.source.index_entries, voice.source.pth_entry.as_deref().unwrap_or("")) {
        if explicit.is_none() {
            warnings.push(w);
        }
    }
    match check_index(&bytes, emb) {
        Ok(dim) => (Some(AdoptedIndex { bytes, origin, dim }), warnings),
        Err(e) => {
            let name = origin.split_once(':').map_or(origin.as_str(), |(_, n)| base_name(n));
            warnings.push(format!("the retrieval index {name} was not adopted: {e}"));
            (None, warnings)
        }
    }
}

// ---- slot folder ----------------------------------------------------------------------------------

fn io_at(path: &Path, e: std::io::Error) -> ImportError {
    ImportError::new(ErrorCode::Io, format!("{}: {e}", path.display()))
}

/// Write `data` to `dir/name` and flush it to disk.
fn write_synced(dir: &Path, name: &str, data: &[u8]) -> Result<(), ImportError> {
    let path = dir.join(name);
    let mut f = File::create(&path).map_err(|e| io_at(&path, e))?;
    f.write_all(data).and_then(|_| f.sync_all()).map_err(|e| io_at(&path, e))
}

/// Write `data` to `path` atomically: a flushed sibling temp file renamed over it.
pub fn write_atomic(path: &Path, data: &[u8]) -> Result<(), ImportError> {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let dir = path.parent().filter(|p| !p.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("file");
    let tmp_name = format!(".{name}.{}-{}.partial", std::process::id(), COUNTER.fetch_add(1, Ordering::Relaxed));
    let tmp = dir.join(&tmp_name);
    let result = write_synced(dir, &tmp_name, data).and_then(|_| std::fs::rename(&tmp, path).map_err(|e| io_at(path, e)));
    if result.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    result
}

/// The files an import regenerates in a slot folder. Everything else in the folder (the voice's
/// `slot.json` settings, an index kept from the previous import, user files) is carried over.
const SLOT_FILES: [&str; 4] = [WEIGHTS_FILE, SPEAKERS_FILE, IMPORT_RECORD, MANIFEST_FILE];

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

/// Move what `from` holds besides the import's regenerated files and `skip` into `to` (the voice's
/// settings, a kept index), unless `to` already has a file of that name.
fn carry_over(from: &Path, to: &Path, skip: &[&str]) -> Result<(), ImportError> {
    for entry in std::fs::read_dir(from).map_err(|e| io_at(from, e))? {
        let entry = entry.map_err(|e| io_at(from, e))?;
        let name = entry.file_name();
        let ours = name.to_str().is_some_and(|n| SLOT_FILES.contains(&n) || n.ends_with(".partial") || skip.contains(&n));
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
            carry_over(backup, out_dir, &[])?;
            std::fs::remove_dir_all(backup).map_err(|e| io_at(backup, e))?;
        }
    }
    if staging.exists() {
        std::fs::remove_dir_all(staging).map_err(|e| io_at(staging, e))?;
    }
    Ok(())
}

/// Restore the slot folder `out_dir` after a worker that was killed or failed mid-import (the
/// parent calls it; the next import into the slot would do the same).
pub fn recover_slot_folder(out_dir: &Path) -> Result<(), ImportError> {
    let (staging, backup) = swap_dirs(out_dir)?;
    recover_slot(out_dir, &staging, &backup)
}

/// Result of an import written to a slot folder (also the worker's stdout JSON).
#[derive(Debug, Serialize)]
pub struct ImportRecord {
    pub format: &'static str,
    pub format_version: u32,
    /// [`IMPORTER_VERSION`] of the importer that wrote the slot.
    pub importer_version: u32,
    /// Display name written to the manifest.
    pub name: String,
    pub template: String,
    pub weights_id: String,
    pub weights: FileRecord,
    pub speakers: SpeakerRecord,
    /// The slot's retrieval index, if it has one.
    pub index: Option<IndexRecord>,
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
    /// Checkpoint metadata warnings plus index adoption notes.
    pub warnings: Vec<String>,
    pub created_unix_ms: u64,
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

#[derive(Debug, Serialize)]
pub struct IndexRecord {
    /// File in the slot folder.
    pub file: String,
    /// `archive:<entry>`, `file:<name>`, or `kept` (from the slot's previous import).
    pub origin: String,
    pub bytes: u64,
    /// Of a newly adopted index (a kept one is not re-hashed).
    pub sha256: Option<String>,
    pub dim: Option<usize>,
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

/// Everything [`write_slot`] writes.
pub struct SlotContents<'a> {
    pub voice: &'a ImportedVoice,
    pub smoke: &'a Smoke,
    /// A newly adopted index (else the slot's previous one, if any, is kept).
    pub index: Option<&'a AdoptedIndex>,
    /// Display name ([`display_name`]).
    pub name: &'a str,
    /// Notes beyond the checkpoint's own warnings (index adoption).
    pub warnings: &'a [String],
    pub elapsed: Duration,
}

/// Milliseconds since the Unix epoch.
pub fn unix_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// A display name: control characters removed, whitespace collapsed, at most [`MAX_NAME_CHARS`]
/// characters; `None` when nothing is left.
pub fn clean_name(raw: &str) -> Option<String> {
    let collapsed = raw.split(|c: char| c.is_whitespace() || c.is_control()).filter(|p| !p.is_empty()).collect::<Vec<_>>().join(" ");
    let cut: String = collapsed.chars().take(MAX_NAME_CHARS).collect();
    let cut = cut.trim().to_string();
    (!cut.is_empty()).then_some(cut)
}

/// The voice's display name: the requested one, else the checkpoint's `model_name` (Applio), else
/// the source file's name.
pub fn display_name(requested: Option<&str>, voice: &ImportedVoice) -> String {
    requested
        .and_then(clean_name)
        .or_else(|| voice.meta.model_name.as_deref().and_then(clean_name))
        .or_else(|| clean_name(&stem_of(&voice.source.path)))
        .unwrap_or_else(|| "Imported voice".to_string())
}

/// The file stem of a path, as text.
fn stem_of(path: &Path) -> String {
    path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
}

/// What a re-import without an index of its own does with the slot's previous index (the
/// `sigil-rvc-1` manifest's `index`, when that file is present).
#[derive(Debug, PartialEq, Eq)]
enum PreviousIndex {
    /// No previous index.
    None,
    /// Kept: `(file, bytes)`.
    Keep(String, u64),
    /// Removed with the previous voice: `(file, warning)`.
    Drop(String, String),
}

/// Longest previous `sigil-import.json` read back.
const MAX_RECORD_BYTES: u64 = 4 << 20;

/// The slot's previous import record, as JSON (`None` when missing or unreadable).
fn previous_record(out_dir: &Path) -> Option<serde_json::Value> {
    let path = out_dir.join(IMPORT_RECORD);
    if std::fs::metadata(&path).ok()?.len() > MAX_RECORD_BYTES {
        return None;
    }
    serde_json::from_slice::<serde_json::Value>(&std::fs::read(path).ok()?)
        .ok()
        .filter(|r| r["format"] == IMPORT_RECORD_FORMAT)
}

/// Whether the slot's previous import was the same voice as `voice`: the same source file
/// (sha256), the same Applio `model_name`, or the same voice name in the source file's name
/// ([`voice_stem`]: `Kira.pth` and a retrained `Kira_e300_s6000.pth` are one voice).
fn same_voice(previous: &serde_json::Value, voice: &ImportedVoice) -> bool {
    let src = &previous["source"];
    if src["sha256"].as_str().is_some_and(|s| s.eq_ignore_ascii_case(&voice.source.sha256)) {
        return true;
    }
    let named = |v: Option<&str>| v.map(str::trim).filter(|s| !s.is_empty()).map(str::to_lowercase);
    if let (Some(a), Some(b)) = (named(previous["meta"]["model_name"].as_str()), named(voice.meta.model_name.as_deref())) {
        if a == b {
            return true;
        }
    }
    let new_name = match &voice.source.pth_entry {
        Some(entry) => entry.clone(),
        None => voice.source.path.to_string_lossy().into_owned(),
    };
    let old_name = src["pth_entry"].as_str().or(src["path"].as_str());
    old_name.is_some_and(|old| voice_stem(old) == voice_stem(&new_name))
}

/// The previous index of the slot `out_dir` for a re-import of `voice` without an index of its own.
/// It is kept only when the previous import was the same voice ([`same_voice`]) and its features
/// fit this voice; otherwise it is removed with a warning (another speaker's index would leak that
/// speaker's timbre into this voice, and an index of the wrong width does nothing).
fn previous_index(out_dir: &Path, voice: &ImportedVoice) -> PreviousIndex {
    let Ok(m) = crate::services::neural_audio::vc::manifest::load_slot_manifest(out_dir) else {
        return PreviousIndex::None;
    };
    let Some(name) = m.index else {
        return PreviousIndex::None;
    };
    if !slot::safe_file_name(&name) {
        return PreviousIndex::None;
    }
    let Some(meta) = std::fs::metadata(out_dir.join(&name)).ok().filter(|m| m.is_file()) else {
        return PreviousIndex::None;
    };
    let removed = "the retrieval index of the voice previously in this slot was removed";
    let width = voice.meta.emb_channels();
    if m.emb_channels as usize != width {
        return PreviousIndex::Drop(name, format!("{removed}: it holds {}-channel features, this voice needs {width}", m.emb_channels));
    }
    match previous_record(out_dir) {
        Some(previous) if same_voice(&previous, voice) => PreviousIndex::Keep(name, meta.len()),
        Some(_) => PreviousIndex::Drop(name, format!("{removed}: it belongs to another voice (add this voice's own .index)")),
        None => PreviousIndex::Drop(name, format!("{removed}: the previous import record is missing, so it could not be matched to this voice")),
    }
}

/// Write the slot's files into `out_dir`, replacing the folder's imported contents as one unit:
/// 1. `weights.f32.bin`, `speakers.f32.bin`, the adopted `added.index`, `sigil-import.json` and
///    `manifest.json` are written (and flushed) into the hidden sibling `.<slot>.sigil-import-new`;
/// 2. an existing slot folder is renamed to `.<slot>.sigil-import-old`, the new folder is renamed
///    into place, and the old folder's other files (the voice's `slot.json`, an index kept from
///    the previous import) are moved over before it is deleted. A previous index is kept only for
///    the same voice ([`previous_index`]); another voice's goes with the old folder.
///
/// A failure before step 2 leaves the slot untouched. A process killed in step 2 leaves the old
/// folder next to the slot, and the next `write_slot` (or [`recover_slot_folder`]) for the slot
/// restores or finishes the swap. One import per slot folder at a time (the parent serialises
/// them). On Windows the swap fails, changing nothing, while another process holds a file in the
/// slot open.
pub fn write_slot(out_dir: &Path, contents: &SlotContents<'_>) -> Result<ImportRecord, ImportError> {
    let (staging, backup) = swap_dirs(out_dir)?;
    if out_dir.exists() && !out_dir.is_dir() {
        return Err(ImportError::new(ErrorCode::Io, format!("{} is not a folder", out_dir.display())));
    }
    if let Some(parent) = staging.parent() {
        std::fs::create_dir_all(parent).map_err(|e| io_at(parent, e))?;
    }
    recover_slot(out_dir, &staging, &backup)?;
    let previous = if contents.index.is_none() { previous_index(out_dir, contents.voice) } else { PreviousIndex::None };
    let (kept_index, dropped, note) = match previous {
        PreviousIndex::None => (None, None, None),
        PreviousIndex::Keep(file, bytes) => (Some((file, bytes)), None, None),
        PreviousIndex::Drop(file, warning) => (None, Some(file), Some(warning)),
    };
    std::fs::create_dir(&staging).map_err(|e| io_at(&staging, e))?;
    let record = write_slot_files(&staging, contents, kept_index, note).and_then(|record| {
        let _commit = CommitGuard::new();
        swap_in(out_dir, &staging, &backup, dropped.as_deref())?;
        Ok(record)
    });
    if record.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    record
}

/// Step 2 of [`write_slot`]; `dropped` (a previous index of another voice) is not carried over.
fn swap_in(out_dir: &Path, staging: &Path, backup: &Path, dropped: Option<&str>) -> Result<(), ImportError> {
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
    if carry_over(backup, out_dir, dropped.as_slice()).is_ok() {
        let _ = std::fs::remove_dir_all(backup);
    }
    Ok(())
}

fn write_slot_files(
    dir: &Path,
    c: &SlotContents<'_>,
    kept_index: Option<(String, u64)>,
    note: Option<String>,
) -> Result<ImportRecord, ImportError> {
    let voice = c.voice;
    let w = &voice.weights;
    let speakers = writer::f32_bytes(&w.speakers);
    write_synced(dir, WEIGHTS_FILE, &w.f32_image)?;
    write_synced(dir, SPEAKERS_FILE, &speakers)?;
    let index = match (c.index, kept_index) {
        (Some(ix), _) => {
            write_synced(dir, ADOPTED_INDEX_FILE, &ix.bytes)?;
            Some(IndexRecord {
                file: ADOPTED_INDEX_FILE.into(),
                origin: ix.origin.clone(),
                bytes: ix.bytes.len() as u64,
                sha256: Some(hex::encode(Sha256::digest(&ix.bytes))),
                dim: Some(ix.dim),
            })
        }
        (None, Some((file, bytes))) => Some(IndexRecord { file, origin: "kept".into(), bytes, sha256: None, dim: None }),
        (None, None) => None,
    };
    let mut warnings = voice.meta.warnings.clone();
    warnings.extend(c.warnings.iter().cloned());
    warnings.extend(note);
    let record = ImportRecord {
        format: IMPORT_RECORD_FORMAT,
        format_version: 1,
        importer_version: IMPORTER_VERSION,
        name: c.name.to_string(),
        template: voice.template.layout.template.clone(),
        weights_id: voice.template.layout.weights.weights_id.clone(),
        weights: FileRecord { file: WEIGHTS_FILE, bytes: w.f32_image.len() as u64, sha256: hex::encode(Sha256::digest(&w.f32_image)) },
        speakers: SpeakerRecord { file: SPEAKERS_FILE, count: w.n_speakers, dim: w.speaker_dim, sha256: hex::encode(Sha256::digest(&speakers)) },
        index,
        meta: voice.meta.clone(),
        source: voice.source.clone(),
        tensors: voice.tensors,
        checkpoint_dtype: voice.checkpoint_dtype,
        fp16_ok: writer::f16_image(&w.f32_image).is_ok() && c.smoke.fp16_issue.is_none(),
        smoke: c.smoke.reports.clone(),
        fp16_issue: c.smoke.fp16_issue.clone(),
        warnings,
        created_unix_ms: unix_ms(),
        elapsed_ms: c.elapsed.as_millis() as u64,
    };
    let json = serde_json::to_vec_pretty(&record).map_err(|e| ImportError::new(ErrorCode::Io, e.to_string()))?;
    write_synced(dir, IMPORT_RECORD, &json)?;
    slot::write_manifest(dir, &record, c.name)?;
    Ok(record)
}

// ---- the worker -----------------------------------------------------------------------------------

/// The job file of `Sigil.exe --headless vc-import-worker <job.json>`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ImportJob {
    /// Job schema version (1).
    #[serde(default = "job_version")]
    pub version: u32,
    /// The `.pth` or voice `.zip` to import.
    pub source: PathBuf,
    /// Slot folder receiving the voice (see [`write_slot`]).
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
    /// Display name for the manifest (default: [`display_name`]).
    #[serde(default)]
    pub name: Option<String>,
    /// A retrieval index file to adopt (default: the voice archive's own `added_*.index`).
    #[serde(default)]
    pub index: Option<PathBuf>,
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
    let candidates: Vec<PathBuf> = exe_dir.as_deref().map(template_dir_candidates).unwrap_or_default();
    #[cfg(debug_assertions)]
    let candidates = {
        let mut c = candidates;
        c.push(Path::new(env!("CARGO_MANIFEST_DIR")).join("resources").join("vc-templates"));
        c
    };
    candidates.into_iter().find(|d| d.is_dir())
}

/// The templates folder the app registered (Tauri's `resource_dir()/resources/vc-templates`).
static TEMPLATES_DIR: RwLock<Option<PathBuf>> = RwLock::new(None);

/// Register the bundled templates folder (the voice-changer plugin does, from Tauri's resource
/// folder); [`templates_dir`] returns it from then on.
pub fn set_templates_dir(dir: PathBuf) {
    *TEMPLATES_DIR.write().unwrap_or_else(|e| e.into_inner()) = Some(dir);
}

/// The templates folder: the registered one ([`set_templates_dir`]) when it exists, else
/// [`default_templates_dir`]. The engine factory and the slot store use it.
pub fn templates_dir() -> Option<PathBuf> {
    let set = TEMPLATES_DIR.read().unwrap_or_else(|e| e.into_inner()).clone();
    set.filter(|d| d.is_dir()).or_else(default_templates_dir)
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
    let (index, warnings) = load_index(&voice, job.index.as_deref());
    let name = display_name(job.name.as_deref(), &voice);
    let contents = SlotContents { voice: &voice, smoke: &smoke, index: index.as_ref(), name: &name, warnings: &warnings, elapsed: started.elapsed() };
    let record = write_slot(&job.out_dir, &contents)?;
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
            // The worker protocol: `<code>: <message>` on stderr, exit 1 (the parent parses it).
            let e = ImportError::new(ErrorCode::TimeLimit, format!("the import took longer than {} s", limit.as_secs()));
            eprintln!("vc-import-worker: {e}");
            std::process::exit(1);
        }
    });
    let result = run_job(&job).map_err(|e| e.to_string());
    drop(done);
    result
}
