//! Slot store: Sigil's voice slots under `app_data/voice-changer/models/<id>/`.
//!
//! - **Imported voices** (`manifest.format == "sigil-rvc-1"`): the isolated worker writes
//!   `weights.f32.bin`, `speakers.f32.bin`, the adopted `added.index`, `sigil-import.json` and the
//!   manifest ([`write_manifest`]) as one unit (`import::write_slot`).
//! - **Bring-your-own ONNX folders** (format `rvc`, with or without `manifest.json`) are listed
//!   too, and can be renamed, duplicated and deleted like any voice.
//!
//! The store also keeps the sha256 import cache ([`find_cached`]: importing the same file again
//! reuses its slot), the weighted merge of two voices of one template ([`merge_slots`], done in
//! the parent: both inputs are Sigil's own files), and the display names (the manifest's `name`;
//! a manifest-less BYO folder keeps it in `sigil-slot.json`, because writing a `manifest.json`
//! there would change how its generator is loaded).
//!
//! Every write is atomic (a flushed temp file renamed into place; new folders are built hidden
//! and renamed into place), so a crash never leaves a half-written voice in the list. Hidden
//! folders (`.x...`) are never listed: they are staging or trash folders ([`recover_all`] tidies
//! them while no import runs).
//!
//! # Frozen by P3-S0 (signatures)
//! [`available`], [`SlotInfo`] (TS mirror `VcSlotInfo` in `src/core/services/voice_changer/ipc.ts`),
//! [`validate_slot_id`], [`slot_dir`], [`list_slots`], [`slot_info`], [`rename_slot`],
//! [`delete_slot`], [`duplicate_slot`], [`find_cached`], [`MergeSpec`], [`merge_slots`],
//! [`write_manifest`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::services::neural_audio::vc::bases::{find_generator_onnx, find_index_file};
use crate::services::neural_audio::vc::manifest::{
    load_slot_manifest, RvcManifest, RvcVersion, SlotFormat, LEGACY_RVC_FORMAT, SIGIL_RVC_FORMAT,
};

use super::meta::manifest_vocoder;
use super::{
    base_name, choose_index, clean_name, unix_ms, voice_stem, write_atomic, ErrorCode, ImportError, ImportRecord,
    IMPORTER_VERSION, IMPORT_RECORD, IMPORT_RECORD_FORMAT, MANIFEST_FILE, SPEAKERS_FILE, WEIGHTS_FILE,
};

/// Display name of a manifest-less BYO folder.
pub const SLOT_NAME_FILE: &str = "sigil-slot.json";
/// Longest slot id accepted (bytes); generated ids are much shorter.
pub const MAX_SLOT_ID_BYTES: usize = 120;
/// Slot folders listed at most.
const MAX_SLOTS: usize = 2000;
/// Longest generated id (characters, before a `-N` suffix).
const MAX_GENERATED_ID: usize = 40;
/// Largest weights image merged or copied (the templates' is ~110 MB).
const MAX_WEIGHTS_BYTES: u64 = 4 << 30;
/// Files copied at most by [`duplicate_slot`], and folder depth.
const MAX_COPY_FILES: usize = 10_000;
const MAX_COPY_DEPTH: usize = 4;

/// Whether the slot store (and the `.pth` / `.zip` import commands) run in this build.
pub fn available() -> bool {
    true
}

/// One voice slot as the UI lists it (snake_case).
#[derive(Clone, Debug, Default, Serialize)]
pub struct SlotInfo {
    /// Folder name under `models/` (stable id).
    pub id: String,
    /// Display name (manifest `name`, else the id).
    pub name: String,
    /// Absolute folder path (what `rvc_slot_dir` stores).
    pub path: String,
    /// `rvc` (bring-your-own ONNX) or `sigil-rvc-1` (imported template voice).
    pub format: String,
    pub template: Option<String>,
    /// `v1` / `v2`.
    pub version: Option<String>,
    pub sample_rate: Option<u32>,
    pub f0: Option<bool>,
    pub embedder: Option<String>,
    pub vocoder: Option<String>,
    pub has_index: bool,
    pub speaker_count: Option<u32>,
    pub fp16_ok: Option<bool>,
    pub source_sha256: Option<String>,
    pub created_unix_ms: Option<u64>,
    /// Why the slot cannot run in this build (unknown format, missing template), if it cannot.
    pub problem: Option<String>,
}

fn job(msg: impl Into<String>) -> String {
    ImportError::new(ErrorCode::Job, msg).to_string()
}

fn io_err(path: &Path, e: impl std::fmt::Display) -> String {
    ImportError::new(ErrorCode::Io, format!("{}: {e}", path.display())).to_string()
}

/// A single file name that stays inside its folder on every platform.
pub fn safe_file_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 255
        && name != "."
        && name != ".."
        && !name.chars().any(|c| c.is_control() || matches!(c, '/' | '\\' | ':' | '*' | '?' | '"' | '<' | '>' | '|'))
}

/// Windows device names that cannot be folder names (with or without an extension).
fn reserved_windows_name(id: &str) -> bool {
    let stem = id.split('.').next().unwrap_or(id).trim_end().to_ascii_uppercase();
    matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$")
        || ((stem.starts_with("COM") || stem.starts_with("LPT"))
            && stem.len() == 4
            && (stem.as_bytes()[3].is_ascii_digit()))
        || ((stem.starts_with("COM") || stem.starts_with("LPT")) && matches!(&stem[3..], "¹" | "²" | "³"))
}

/// A slot id is a single safe folder name: no separators, no `..`, not hidden, no reserved Windows
/// name, no trailing dot or space, at most [`MAX_SLOT_ID_BYTES`] bytes. Unicode is fine (a BYO
/// folder may be named anything the user likes).
pub fn validate_slot_id(id: &str) -> Result<(), String> {
    let ok = safe_file_name(id)
        && id.len() <= MAX_SLOT_ID_BYTES
        && id.trim() == id
        && !id.starts_with('.')
        && !id.ends_with('.')
        && !reserved_windows_name(id);
    if ok {
        Ok(())
    } else {
        Err(job(format!("{:?} is not a voice slot id", id.chars().take(64).collect::<String>())))
    }
}

/// `<models_dir>/<id>` after [`validate_slot_id`].
pub fn slot_dir(models_dir: &Path, id: &str) -> Result<PathBuf, String> {
    validate_slot_id(id)?;
    Ok(models_dir.join(id))
}

/// `<models_dir>/<id>` of an existing slot folder.
fn existing_slot(models_dir: &Path, id: &str) -> Result<PathBuf, String> {
    let dir = slot_dir(models_dir, id)?;
    if !dir.is_dir() {
        return Err(job(format!("there is no voice slot {id:?}")));
    }
    Ok(dir)
}

/// A fresh id for a voice named `name`: its ASCII letters and digits in lower case with dashes
/// (`My Voice (v2)` -> `my-voice-v2`), else `voice`; `-2`, `-3`, ... when taken.
pub fn new_slot_id(models_dir: &Path, name: &str) -> String {
    let mut slug = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
        } else if !slug.ends_with('-') && !slug.is_empty() {
            slug.push('-');
        }
        if slug.len() >= MAX_GENERATED_ID {
            break;
        }
    }
    let slug = slug.trim_matches('-').to_string();
    let base = if slug.is_empty() || validate_slot_id(&slug).is_err() { "voice".to_string() } else { slug };
    let taken = |id: &str| {
        let p = models_dir.join(id);
        p.exists() || std::fs::symlink_metadata(&p).is_ok()
    };
    if !taken(&base) {
        return base;
    }
    (2..100_000).map(|n| format!("{base}-{n}")).find(|id| !taken(id)).unwrap_or_else(|| format!("{base}-{}", nonce()))
}

/// A short random-enough suffix for hidden work folders.
fn nonce() -> String {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    format!("{:x}{:x}{:x}", std::process::id(), unix_ms(), COUNTER.fetch_add(1, Ordering::Relaxed))
}

// ---- reading slots --------------------------------------------------------------------------------

/// What the slot store reads back from `sigil-import.json` (tolerant: every field optional).
#[derive(Debug, Default, Deserialize)]
struct RecordSummary {
    #[serde(default)]
    format: String,
    #[serde(default)]
    importer_version: u32,
    #[serde(default)]
    weights_id: String,
    #[serde(default)]
    weights: FileSummary,
    #[serde(default)]
    speakers: SpeakerSummary,
    #[serde(default)]
    fp16_ok: Option<bool>,
    #[serde(default)]
    smoke: Vec<SmokeSummary>,
    #[serde(default)]
    fp16_issue: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
struct SmokeSummary {
    #[serde(default)]
    precision: String,
}

#[derive(Debug, Default, Deserialize)]
struct FileSummary {
    #[serde(default)]
    bytes: u64,
    #[serde(default)]
    sha256: String,
}

#[derive(Debug, Default, Deserialize)]
struct SpeakerSummary {
    #[serde(default)]
    dim: usize,
}

fn read_record(dir: &Path) -> Option<RecordSummary> {
    let path = dir.join(IMPORT_RECORD);
    let meta = std::fs::metadata(&path).ok()?;
    if meta.len() > 4 << 20 {
        return None;
    }
    serde_json::from_slice::<RecordSummary>(&std::fs::read(path).ok()?).ok().filter(|r| r.format == IMPORT_RECORD_FORMAT)
}

/// Whether the import of the slot `dir` already settled fp16 for the voice: an fp16 smoke decode
/// ran (passed or failed), or its weights do not fit half precision. An import that asks for the
/// fp16 check is served from the cache only then.
pub fn fp16_checked(dir: &Path) -> bool {
    read_record(dir).is_some_and(|r| r.fp16_ok == Some(false) || r.fp16_issue.is_some() || r.smoke.iter().any(|s| s.precision == "fp16"))
}

/// `weights.weights_id` of a template's `layout.json` (read without the full validation).
fn template_weights_id(templates: &Path, template: &str) -> Option<String> {
    #[derive(Deserialize)]
    struct Head {
        weights: HeadWeights,
    }
    #[derive(Deserialize)]
    struct HeadWeights {
        weights_id: String,
    }
    if !template.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'-') || template.is_empty() {
        return None;
    }
    let path = templates.join(template).join(super::layout::LAYOUT_FILE);
    let bytes = std::fs::read(path).ok()?;
    serde_json::from_slice::<Head>(&bytes).ok().map(|h| h.weights.weights_id)
}

/// Template weights ids read during one listing (every slot of a template shares one read).
#[derive(Default)]
struct TemplateIds(std::collections::HashMap<String, Option<String>>);

impl TemplateIds {
    fn get(&mut self, templates: &Path, template: &str) -> Option<String> {
        self.0.entry(template.to_string()).or_insert_with(|| template_weights_id(templates, template)).clone()
    }
}

/// The display name a manifest-less folder keeps in [`SLOT_NAME_FILE`].
fn sidecar_name(dir: &Path) -> Option<String> {
    #[derive(Deserialize)]
    struct Sidecar {
        name: String,
    }
    let path = dir.join(SLOT_NAME_FILE);
    if std::fs::metadata(&path).ok()?.len() > 64 << 10 {
        return None;
    }
    let s: Sidecar = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
    clean_name(&s.name)
}

/// The `format` field of a manifest that does not load (for the listing).
fn raw_format(dir: &Path) -> Option<String> {
    let bytes = std::fs::read(dir.join(MANIFEST_FILE)).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    Some(v.get("format").and_then(|f| f.as_str()).unwrap_or(LEGACY_RVC_FORMAT).chars().take(64).collect())
}

fn version_name(v: RvcVersion) -> &'static str {
    match v {
        RvcVersion::V1 => "v1",
        RvcVersion::V2 => "v2",
        RvcVersion::Nono => "nono",
    }
}

/// What is wrong with an imported voice's files, if anything.
fn template_voice_problem(dir: &Path, m: &RvcManifest, templates: Option<&Path>, ids: &mut TemplateIds) -> Option<String> {
    let weights = m.weights.as_deref().unwrap_or(WEIGHTS_FILE);
    let speakers = m.speakers.as_deref().unwrap_or(SPEAKERS_FILE);
    for f in [weights, speakers] {
        if !safe_file_name(f) || !dir.join(f).is_file() {
            return Some(format!("{f} is missing; import the voice again"));
        }
    }
    let template = m.template.as_deref().unwrap_or_default();
    if let Some(templates) = templates {
        match ids.get(templates, template) {
            None => return Some(format!("it needs the {template} template, which this version of Sigil does not have")),
            Some(id) => {
                if let Some(r) = read_record(dir) {
                    if !r.weights_id.is_empty() && r.weights_id != id {
                        return Some(format!("it was imported for another version of the {template} template; import it again"));
                    }
                    if r.weights.bytes > 0 && std::fs::metadata(dir.join(weights)).map(|md| md.len()).ok() != Some(r.weights.bytes) {
                        return Some(format!("{weights} is damaged; import the voice again"));
                    }
                }
            }
        }
    }
    None
}

/// Why no imported voice can run in this build, if none can (a property of the build, not of the
/// slot: the import cache still reuses such a slot).
fn engine_problem() -> Option<String> {
    (!crate::services::neural_audio::vc::engine::available())
        .then(|| "imported voices run on the Sigil Voice Engine, which this build cannot run yet".to_string())
}

/// The listing of one slot folder (never fails: a broken folder is listed with its problem).
fn info_for(models_dir: &Path, id: &str, templates: Option<&Path>, ids: &mut TemplateIds) -> SlotInfo {
    let dir = models_dir.join(id);
    let mut info = SlotInfo {
        id: id.to_string(),
        name: id.to_string(),
        path: dir.to_string_lossy().into_owned(),
        format: LEGACY_RVC_FORMAT.to_string(),
        ..SlotInfo::default()
    };
    if !dir.join(MANIFEST_FILE).is_file() {
        if let Some(n) = sidecar_name(&dir) {
            info.name = n;
        }
        info.has_index = find_index_file(&dir).is_some();
        if find_generator_onnx(&dir).is_none() {
            info.problem = Some("no manifest.json and no generator .onnx in the folder".into());
        }
        return info;
    }
    let m = match load_slot_manifest(&dir) {
        Ok(m) => m,
        Err(e) => {
            info.format = raw_format(&dir).unwrap_or_else(|| "unknown".into());
            info.name = sidecar_name(&dir).unwrap_or(info.name);
            info.problem = Some(e);
            return info;
        }
    };
    info.format = m.format.trim().to_string();
    info.name = m.name.as_deref().and_then(clean_name).or_else(|| sidecar_name(&dir)).unwrap_or(info.name);
    info.version = Some(version_name(m.version).to_string());
    info.sample_rate = Some(m.sample_rate);
    info.f0 = Some(m.wants_f0());
    info.embedder = m.embedder.clone();
    info.vocoder = m.vocoder.clone();
    info.speaker_count = m.speaker_count;
    info.fp16_ok = m.fp16_ok;
    info.source_sha256 = m.source_sha256.clone();
    info.created_unix_ms = m.created_unix_ms;
    info.template = m.template.clone();
    info.has_index = match &m.index {
        Some(ix) => safe_file_name(ix) && dir.join(ix).is_file(),
        None => m.slot_format() == SlotFormat::LegacyOnnxRvc && find_index_file(&dir).is_some(),
    };
    info.problem = match m.slot_format() {
        SlotFormat::SigilRvc => template_voice_problem(&dir, &m, templates, ids).or_else(engine_problem),
        SlotFormat::LegacyOnnxRvc => {
            let g = m.generator.trim();
            (!safe_file_name(g) || !dir.join(g).is_file()).then(|| format!("the generator {g} is missing"))
        }
        SlotFormat::Unknown(f) => Some(format!("unsupported voice format {f:?}")),
    };
    info
}

fn slot_ids(models_dir: &Path) -> Vec<String> {
    let Ok(rd) = std::fs::read_dir(models_dir) else {
        return Vec::new();
    };
    let mut ids: Vec<String> = rd
        .flatten()
        .take(MAX_SLOTS * 2)
        .filter(|e| e.path().is_dir())
        .filter_map(|e| e.file_name().to_str().map(str::to_string))
        .filter(|id| validate_slot_id(id).is_ok())
        .collect();
    ids.sort();
    ids.truncate(MAX_SLOTS);
    ids
}

/// Every slot folder under `models_dir`, sorted by display name (case-insensitive), then id.
pub fn list_slots(models_dir: &Path) -> Result<Vec<SlotInfo>, String> {
    list_slots_with(models_dir, super::templates_dir().as_deref())
}

/// [`list_slots`] against an explicit templates folder.
pub fn list_slots_with(models_dir: &Path, templates: Option<&Path>) -> Result<Vec<SlotInfo>, String> {
    if !models_dir.exists() {
        return Ok(Vec::new());
    }
    if !models_dir.is_dir() {
        return Err(io_err(models_dir, "not a folder"));
    }
    let mut ids = TemplateIds::default();
    let mut out: Vec<SlotInfo> = slot_ids(models_dir).iter().map(|id| info_for(models_dir, id, templates, &mut ids)).collect();
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()).then_with(|| a.id.cmp(&b.id)));
    Ok(out)
}

pub fn slot_info(models_dir: &Path, id: &str) -> Result<SlotInfo, String> {
    slot_info_with(models_dir, id, super::templates_dir().as_deref())
}

/// [`slot_info`] against an explicit templates folder.
pub fn slot_info_with(models_dir: &Path, id: &str, templates: Option<&Path>) -> Result<SlotInfo, String> {
    existing_slot(models_dir, id)?;
    Ok(info_for(models_dir, id, templates, &mut TemplateIds::default()))
}

// ---- names ----------------------------------------------------------------------------------------

/// Set the display name in `dir`: the manifest's `name` (patched as JSON, so every other field,
/// known or not, is kept as it was), or [`SLOT_NAME_FILE`] for a manifest-less folder.
fn set_name(dir: &Path, name: &str) -> Result<(), String> {
    let manifest = dir.join(MANIFEST_FILE);
    if manifest.is_file() {
        let bytes = std::fs::read(&manifest).map_err(|e| io_err(&manifest, e))?;
        let mut v: serde_json::Value = serde_json::from_slice(&bytes).map_err(|e| io_err(&manifest, e))?;
        let obj = v.as_object_mut().ok_or_else(|| io_err(&manifest, "not a JSON object"))?;
        obj.insert("name".into(), serde_json::Value::String(name.to_string()));
        let out = serde_json::to_vec_pretty(&v).map_err(|e| io_err(&manifest, e))?;
        write_atomic(&manifest, &out).map_err(|e| e.to_string())
    } else {
        let out = serde_json::to_vec_pretty(&serde_json::json!({ "name": name })).map_err(|e| io_err(dir, e))?;
        write_atomic(&dir.join(SLOT_NAME_FILE), &out).map_err(|e| e.to_string())
    }
}

/// Change the display name (the folder id stays).
pub fn rename_slot(models_dir: &Path, id: &str, name: &str) -> Result<SlotInfo, String> {
    let dir = existing_slot(models_dir, id)?;
    let name = clean_name(name).ok_or_else(|| job("the name is empty"))?;
    set_name(&dir, &name)?;
    slot_info(models_dir, id)
}

// ---- delete / duplicate -------------------------------------------------------------------------------

/// Delete a slot folder (the caller stops or switches the voice changer away from it first). The
/// folder is first renamed to a hidden trash name, which fails (changing nothing) while a file in
/// it is open; a link to a folder elsewhere is removed without touching its target.
pub fn delete_slot(models_dir: &Path, id: &str) -> Result<(), String> {
    let dir = slot_dir(models_dir, id)?;
    let meta = std::fs::symlink_metadata(&dir).map_err(|_| job(format!("there is no voice slot {id:?}")))?;
    // A link (symlink or junction) to a folder elsewhere: remove the link, never its target.
    if meta.file_type().is_symlink() {
        return std::fs::remove_dir(&dir).or_else(|_| std::fs::remove_file(&dir)).map_err(|e| io_err(&dir, e));
    }
    if !meta.is_dir() {
        return Err(job(format!("there is no voice slot {id:?}")));
    }
    let trash = models_dir.join(format!(".{id}.sigil-delete-{}", nonce()));
    std::fs::rename(&dir, &trash).map_err(|e| {
        io_err(&dir, format!("{e} (is the voice in use? stop the voice changer or close the folder, then try again)"))
    })?;
    if let Err(e) = std::fs::remove_dir_all(&trash) {
        // The voice is gone from the list either way; the hidden rest is removed later.
        tracing::warn!("voice slot {id}: could not remove {} completely: {e}", trash.display());
    }
    Ok(())
}

/// Copy the regular files of `from` into the new folder `to` (links and hidden work files are
/// skipped; bounded depth and file count).
fn copy_tree(from: &Path, to: &Path, depth: usize, files: &mut usize) -> Result<(), String> {
    std::fs::create_dir(to).map_err(|e| io_err(to, e))?;
    for entry in std::fs::read_dir(from).map_err(|e| io_err(from, e))? {
        let entry = entry.map_err(|e| io_err(from, e))?;
        let ft = entry.file_type().map_err(|e| io_err(&entry.path(), e))?;
        let name = entry.file_name();
        let hidden_work = name.to_str().is_some_and(|n| n.ends_with(".partial") || (n.starts_with('.') && n.contains(".sigil-")));
        if ft.is_symlink() || hidden_work {
            continue;
        }
        let target = to.join(&name);
        if ft.is_dir() {
            if depth + 1 < MAX_COPY_DEPTH {
                copy_tree(&entry.path(), &target, depth + 1, files)?;
            }
        } else if ft.is_file() {
            *files += 1;
            if *files > MAX_COPY_FILES {
                return Err(job(format!("the voice folder holds more than {MAX_COPY_FILES} files")));
            }
            std::fs::copy(entry.path(), &target).map_err(|e| io_err(&entry.path(), e))?;
        }
    }
    Ok(())
}

/// Build a new slot in a hidden folder with `fill`, then rename it into place as `id`.
fn create_slot(models_dir: &Path, id: &str, fill: impl FnOnce(&Path) -> Result<(), String>) -> Result<(), String> {
    validate_slot_id(id)?;
    std::fs::create_dir_all(models_dir).map_err(|e| io_err(models_dir, e))?;
    let staging = models_dir.join(format!(".{id}.sigil-new-{}", nonce()));
    let result = fill(&staging).and_then(|_| {
        let dest = models_dir.join(id);
        if dest.exists() {
            return Err(job(format!("a voice slot {id:?} already exists")));
        }
        std::fs::rename(&staging, &dest).map_err(|e| io_err(&dest, e))
    });
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result
}

pub fn duplicate_slot(models_dir: &Path, id: &str, name: Option<&str>) -> Result<SlotInfo, String> {
    let src = existing_slot(models_dir, id)?;
    let source = slot_info(models_dir, id)?;
    let name = name.and_then(clean_name).unwrap_or_else(|| {
        let base: String = source.name.chars().take(super::MAX_NAME_CHARS.saturating_sub(7)).collect();
        format!("{base} (copy)")
    });
    let new_id = new_slot_id(models_dir, &name);
    create_slot(models_dir, &new_id, |staging| {
        copy_tree(&src, staging, 0, &mut 0)?;
        set_name(staging, &name)
    })?;
    slot_info(models_dir, &new_id)
}

// ---- import cache ---------------------------------------------------------------------------------

/// The slot an earlier import of the file with this sha256 produced, if it still matches: same
/// importer version, weights intact (size), its template present with the same weights layout, and
/// nothing else wrong with it.
pub fn find_cached(models_dir: &Path, source_sha256: &str) -> Option<SlotInfo> {
    find_cached_with(models_dir, source_sha256, super::templates_dir().as_deref())
}

/// [`find_cached`] against an explicit templates folder.
pub fn find_cached_with(models_dir: &Path, source_sha256: &str, templates: Option<&Path>) -> Option<SlotInfo> {
    let sha = source_sha256.trim().to_ascii_lowercase();
    if sha.len() != 64 || !sha.bytes().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    let templates = templates?;
    let mut ids = TemplateIds::default();
    for id in slot_ids(models_dir) {
        let dir = models_dir.join(&id);
        let Ok(m) = load_slot_manifest(&dir) else {
            continue;
        };
        if m.slot_format() != SlotFormat::SigilRvc || m.source_sha256.as_deref() != Some(sha.as_str()) {
            continue;
        }
        let Some(record) = read_record(&dir) else {
            continue;
        };
        let template = m.template.clone().unwrap_or_default();
        let current = ids.get(templates, &template);
        let intact = std::fs::metadata(dir.join(WEIGHTS_FILE)).map(|md| md.len()).ok() == Some(record.weights.bytes);
        if record.importer_version != IMPORTER_VERSION || current.as_deref() != Some(record.weights_id.as_str()) || !intact {
            continue;
        }
        // Reusable unless something is wrong with the slot itself (the engine's availability is a
        // property of the build, not of the slot).
        if template_voice_problem(&dir, &m, Some(templates), &mut ids).is_none() {
            return Some(info_for(models_dir, &id, Some(templates), &mut ids));
        }
    }
    None
}

// ---- retrieval index next to a .pth ---------------------------------------------------------------

/// The retrieval index to adopt with a `.pth` imported from a folder: RVC's `added_*.index` next to
/// it when that choice is safe (its name contains the voice's name, or it is the folder's only
/// index for its only voice). A Downloads folder full of voices adopts nothing by guesswork.
pub fn sibling_index(pth: &Path) -> Option<PathBuf> {
    let dir = pth.parent()?;
    let pth_name = pth.file_name()?.to_string_lossy().into_owned();
    let mut indexes = Vec::new();
    let mut voices = 0usize;
    for e in std::fs::read_dir(dir).ok()?.flatten().take(5000) {
        if !e.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        let name = e.file_name().to_string_lossy().into_owned();
        let lower = name.to_ascii_lowercase();
        if lower.ends_with(".index") {
            indexes.push(name);
        } else if lower.ends_with(".pth") && !super::is_training_artifact(&name) {
            voices += 1;
        }
    }
    let (choice, ambiguous) = choose_index(&indexes, &pth_name);
    let choice = choice?;
    let stem = voice_stem(&pth_name);
    let named = stem.len() >= 3 && base_name(&choice).to_ascii_lowercase().contains(&stem);
    (named || (voices <= 1 && ambiguous.is_none())).then(|| dir.join(choice))
}

// ---- merge ------------------------------------------------------------------------------------------

/// Weighted merge of two same-template voices into a new slot.
#[derive(Clone, Debug)]
pub struct MergeSpec {
    pub a: String,
    pub b: String,
    /// Weight of `a` (0..1); `b` gets `1 - weight_a`.
    pub weight_a: f32,
    pub name: String,
}

/// `sigil-import.json` of a merged voice.
#[derive(Serialize)]
struct MergeRecord<'a> {
    format: &'static str,
    format_version: u32,
    importer_version: u32,
    name: &'a str,
    template: &'a str,
    weights_id: &'a str,
    weights: MergeFile,
    speakers: MergeSpeakers,
    fp16_ok: bool,
    merge: MergeSources<'a>,
    created_unix_ms: u64,
}

#[derive(Serialize)]
struct MergeFile {
    file: &'static str,
    bytes: u64,
    sha256: String,
}

#[derive(Serialize)]
struct MergeSpeakers {
    file: &'static str,
    count: usize,
    dim: usize,
    sha256: String,
}

#[derive(Serialize)]
struct MergeSources<'a> {
    a: MergeSource<'a>,
    b: MergeSource<'a>,
    weight_a: f32,
    /// How the voices were combined.
    method: &'static str,
}

#[derive(Serialize)]
struct MergeSource<'a> {
    id: &'a str,
    name: &'a str,
    weights_sha256: String,
    source_sha256: Option<&'a str>,
}

/// A merge input: an imported voice with its weights and speaker table read.
struct MergeInput {
    id: String,
    name: String,
    manifest: RvcManifest,
    record: RecordSummary,
    weights: Vec<u8>,
    speakers: Vec<u8>,
}

fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>, String> {
    let len = std::fs::metadata(path).map_err(|e| io_err(path, e))?.len();
    if len > max {
        return Err(io_err(path, format!("{len} bytes (at most {max})")));
    }
    std::fs::read(path).map_err(|e| io_err(path, e))
}

fn merge_input(models_dir: &Path, id: &str) -> Result<MergeInput, String> {
    let dir = existing_slot(models_dir, id)?;
    let not_imported = || job(format!("{id} is not an imported voice (only imported voices can be merged)"));
    if !dir.join(MANIFEST_FILE).is_file() {
        return Err(not_imported());
    }
    let manifest = load_slot_manifest(&dir).map_err(|e| job(format!("{id}: {e}")))?;
    if manifest.slot_format() != SlotFormat::SigilRvc {
        return Err(not_imported());
    }
    let record = read_record(&dir).ok_or_else(|| job(format!("{id}: its {IMPORT_RECORD} is missing or unreadable; import it again")))?;
    let wf = manifest.weights.clone().unwrap_or_else(|| WEIGHTS_FILE.into());
    let sf = manifest.speakers.clone().unwrap_or_else(|| SPEAKERS_FILE.into());
    if !safe_file_name(&wf) || !safe_file_name(&sf) {
        return Err(job(format!("{id}: bad file names in its manifest")));
    }
    let weights = read_bounded(&dir.join(wf), MAX_WEIGHTS_BYTES)?;
    let speakers = read_bounded(&dir.join(sf), 64 << 20)?;
    if weights.len() as u64 != record.weights.bytes || !weights.len().is_multiple_of(4) {
        return Err(job(format!("{id}: {WEIGHTS_FILE} is damaged; import the voice again")));
    }
    let name = manifest.name.as_deref().and_then(clean_name).unwrap_or_else(|| id.to_string());
    Ok(MergeInput { id: id.to_string(), name, manifest, record, weights, speakers })
}

fn f32s(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
}

/// Weighted merge of two imported voices of the same template and embedder into a new slot (done in
/// the parent: both inputs are Sigil's own files). Every weight is `w·a + (1 − w)·b` of the folded
/// weights; the speaker tables are blended row by row over the rows both have (RVC's merge rule).
/// The retrieval index is not carried (it describes one voice's training data).
pub fn merge_slots(models_dir: &Path, spec: &MergeSpec) -> Result<SlotInfo, String> {
    if !spec.weight_a.is_finite() || !(0.0..=1.0).contains(&spec.weight_a) {
        return Err(job("the merge weight must be between 0 and 1"));
    }
    if spec.a == spec.b {
        return Err(job("choose two different voices to merge"));
    }
    let a = merge_input(models_dir, &spec.a)?;
    let b = merge_input(models_dir, &spec.b)?;
    let (ta, tb) = (a.manifest.template.as_deref().unwrap_or(""), b.manifest.template.as_deref().unwrap_or(""));
    if ta != tb || a.record.weights_id != b.record.weights_id || a.weights.len() != b.weights.len() {
        return Err(ImportError::new(
            ErrorCode::UnsupportedModel,
            format!("{} ({ta}) and {} ({tb}) have different architectures; only voices of one template can be merged", a.name, b.name),
        )
        .to_string());
    }
    if a.manifest.embedder != b.manifest.embedder {
        return Err(ImportError::new(
            ErrorCode::UnsupportedModel,
            format!(
                "{} and {} were trained with different embedders ({} / {}); their merge would not work",
                a.name,
                b.name,
                a.manifest.embedder.as_deref().unwrap_or("?"),
                b.manifest.embedder.as_deref().unwrap_or("?")
            ),
        )
        .to_string());
    }
    let dim = a.record.speakers.dim;
    if dim == 0 || dim != b.record.speakers.dim || a.speakers.len() % (dim * 4) != 0 || b.speakers.len() % (dim * 4) != 0 {
        return Err(job("the voices' speaker tables do not match"));
    }
    let wa = spec.weight_a as f64;
    let blend = |x: &[u8], y: &[u8]| -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(x.len());
        for (p, q) in f32s(x).zip(f32s(y)) {
            let v = (wa * p as f64 + (1.0 - wa) * q as f64) as f32;
            if !v.is_finite() {
                return Err(ImportError::new(ErrorCode::NonFinite, "the merge produced a non-finite weight").to_string());
            }
            out.extend_from_slice(&v.to_le_bytes());
        }
        Ok(out)
    };
    let weights = blend(&a.weights, &b.weights)?;
    let rows = (a.speakers.len() / (dim * 4)).min(b.speakers.len() / (dim * 4));
    let speakers = blend(&a.speakers[..rows * dim * 4], &b.speakers[..rows * dim * 4])?;
    let fp16_ok = super::writer::f16_image(&weights).is_ok() && a.manifest.fp16_ok != Some(false) && b.manifest.fp16_ok != Some(false);
    let name = clean_name(&spec.name).unwrap_or_else(|| {
        let n: String = format!("{} + {}", a.name, b.name).chars().take(super::MAX_NAME_CHARS).collect();
        n
    });
    let new_id = new_slot_id(models_dir, &name);
    let created = unix_ms();
    let record = MergeRecord {
        format: IMPORT_RECORD_FORMAT,
        format_version: 1,
        importer_version: IMPORTER_VERSION,
        name: &name,
        template: ta,
        weights_id: &a.record.weights_id,
        weights: MergeFile { file: WEIGHTS_FILE, bytes: weights.len() as u64, sha256: hex::encode(Sha256::digest(&weights)) },
        speakers: MergeSpeakers { file: SPEAKERS_FILE, count: rows, dim, sha256: hex::encode(Sha256::digest(&speakers)) },
        fp16_ok,
        merge: MergeSources {
            a: MergeSource { id: &a.id, name: &a.name, weights_sha256: a.record.weights.sha256.clone(), source_sha256: a.manifest.source_sha256.as_deref() },
            b: MergeSource { id: &b.id, name: &b.name, weights_sha256: b.record.weights.sha256.clone(), source_sha256: b.manifest.source_sha256.as_deref() },
            weight_a: spec.weight_a,
            method: "linear blend of the folded weights; speaker rows blended over the rows both voices have",
        },
        created_unix_ms: created,
    };
    let manifest = RvcManifest {
        name: Some(name.clone()),
        index: None,
        speaker_id: 0,
        speaker_count: u32::try_from(rows).ok(),
        fp16_ok: Some(fp16_ok),
        source_sha256: None,
        weights: Some(WEIGHTS_FILE.into()),
        speakers: Some(SPEAKERS_FILE.into()),
        import_record: Some(IMPORT_RECORD.into()),
        created_unix_ms: Some(created),
        ..a.manifest.clone()
    };
    create_slot(models_dir, &new_id, |staging| {
        std::fs::create_dir(staging).map_err(|e| io_err(staging, e))?;
        let write = |name: &str, data: &[u8]| write_atomic(&staging.join(name), data).map_err(|e| e.to_string());
        write(WEIGHTS_FILE, &weights)?;
        write(SPEAKERS_FILE, &speakers)?;
        write(IMPORT_RECORD, &serde_json::to_vec_pretty(&record).map_err(|e| io_err(staging, e))?)?;
        write(MANIFEST_FILE, &serde_json::to_vec_pretty(&manifest).map_err(|e| io_err(staging, e))?)
    })?;
    slot_info(models_dir, &new_id)
}

// ---- manifest ---------------------------------------------------------------------------------------

/// Write the `sigil-rvc-1` manifest of a freshly imported slot from its import record.
pub fn write_manifest(slot_dir: &Path, record: &ImportRecord, name: &str) -> Result<RvcManifest, ImportError> {
    let meta = &record.meta;
    let v1 = meta.version == "v1";
    let m = RvcManifest {
        format: SIGIL_RVC_FORMAT.into(),
        version: if v1 { RvcVersion::V1 } else { RvcVersion::V2 },
        sample_rate: meta.sample_rate,
        if_f0: meta.f0,
        emb_channels: if v1 { 256 } else { 768 },
        emb_layer: if v1 { 9 } else { 12 },
        use_final_proj: v1,
        generator: String::new(),
        index: record.index.as_ref().map(|i| i.file.clone()),
        speaker_id: 0,
        hubert: None,
        name: Some(clean_name(name).unwrap_or_else(|| "Imported voice".into())),
        template: Some(record.template.clone()),
        weights: Some(record.weights.file.to_string()),
        speakers: Some(record.speakers.file.to_string()),
        speaker_count: u32::try_from(record.speakers.count).ok(),
        embedder: Some(meta.embedder.clone()),
        vocoder: Some(manifest_vocoder(&meta.vocoder).to_string()),
        fp16_ok: Some(record.fp16_ok),
        source_sha256: Some(record.source.sha256.clone()).filter(|s| !s.is_empty()),
        import_record: Some(IMPORT_RECORD.into()),
        created_unix_ms: Some(record.created_unix_ms),
    };
    let json = serde_json::to_vec_pretty(&m).map_err(|e| ImportError::new(ErrorCode::Io, e.to_string()))?;
    write_atomic(&slot_dir.join(MANIFEST_FILE), &json)?;
    Ok(m)
}

// ---- housekeeping -----------------------------------------------------------------------------------

/// Finish or undo the slot swaps of imports that were killed (`.X.sigil-import-old` / `-new`), and
/// remove leftover trash and staging folders. Call it only while no import runs (a running
/// import's staging folder would be removed).
pub fn recover_all(models_dir: &Path) {
    let Ok(rd) = std::fs::read_dir(models_dir) else {
        return;
    };
    let mut hidden = Vec::new();
    for e in rd.flatten().take(MAX_SLOTS * 2) {
        let Some(name) = e.file_name().to_str().map(str::to_string).filter(|n| n.starts_with('.')) else {
            continue;
        };
        if e.path().is_dir() {
            hidden.push(name);
        } else if name.starts_with(".sigil-import-job-") && name.ends_with(".json") {
            // A job file whose parent died before removing it.
            let _ = std::fs::remove_file(e.path());
        }
    }
    for name in hidden {
        let path = models_dir.join(&name);
        let slot = name
            .strip_prefix('.')
            .and_then(|n| n.strip_suffix(".sigil-import-old").or_else(|| n.strip_suffix(".sigil-import-new")));
        if let Some(id) = slot.filter(|id| validate_slot_id(id).is_ok()) {
            if let Err(e) = super::recover_slot_folder(&models_dir.join(id)) {
                tracing::warn!("voice slot {id}: could not recover an interrupted import: {e}");
            }
        } else if name.contains(".sigil-delete-") || name.contains(".sigil-new-") {
            let _ = std::fs::remove_dir_all(&path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) struct TempDir(pub PathBuf);

    impl TempDir {
        pub fn new(tag: &str) -> TempDir {
            let d = std::env::temp_dir().join(format!("sigil-vc-slot-{tag}-{}-{}", std::process::id(), nonce()));
            std::fs::create_dir_all(&d).unwrap();
            TempDir(d)
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    #[test]
    fn slot_ids_are_single_safe_folder_names() {
        for ok in ["voice", "my-voice-2", "My Voice", "こえ", "a.b", "v2_40k"] {
            assert!(validate_slot_id(ok).is_ok(), "{ok}");
        }
        for bad in ["", ".", "..", ".hidden", "a/b", "a\\b", "C:", "con", "CON.txt", "com1", "LPT9.x", "nul ", " x", "x.", "a*b", "tab\t", &"x".repeat(121)] {
            assert!(validate_slot_id(bad).is_err(), "{bad:?}");
        }
        let err = validate_slot_id("../evil").unwrap_err();
        assert!(err.starts_with("job:"), "{err}");
    }

    #[test]
    fn new_ids_are_slugs_and_unique() {
        let t = TempDir::new("ids");
        assert_eq!(new_slot_id(&t.0, "My Voice (v2)!"), "my-voice-v2");
        assert_eq!(new_slot_id(&t.0, "こえ"), "voice");
        assert_eq!(new_slot_id(&t.0, "CON"), "voice");
        std::fs::create_dir(t.0.join("voice")).unwrap();
        std::fs::create_dir(t.0.join("voice-2")).unwrap();
        assert_eq!(new_slot_id(&t.0, "  "), "voice-3");
        let long = new_slot_id(&t.0, &"abc ".repeat(40));
        assert!(long.len() <= MAX_GENERATED_ID && !long.ends_with('-'), "{long}");
    }

    #[test]
    fn byo_folders_are_listed_renamed_duplicated_and_deleted() {
        let t = TempDir::new("byo");
        let models = t.0.join("models");
        std::fs::create_dir_all(models.join("zeta")).unwrap();
        std::fs::write(models.join("zeta").join("gen.onnx"), b"x").unwrap();
        std::fs::write(models.join("zeta").join("added_IVF1_Flat.index"), b"x").unwrap();
        std::fs::create_dir_all(models.join("alpha")).unwrap();
        std::fs::write(models.join("alpha").join("manifest.json"), br#"{"generator":"g.onnx","sample_rate":48000,"w_okada_extra":7}"#).unwrap();
        std::fs::create_dir_all(models.join("empty")).unwrap();
        std::fs::create_dir_all(models.join(".alpha.sigil-import-new")).unwrap();
        let list = list_slots_with(&models, None).unwrap();
        let ids: Vec<&str> = list.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["alpha", "empty", "zeta"], "hidden folders are not listed");
        let zeta = &list[2];
        assert_eq!((zeta.format.as_str(), zeta.has_index, zeta.problem.as_deref()), ("rvc", true, None));
        assert!(list[1].problem.as_deref().unwrap().contains("no generator"));
        let alpha = &list[0];
        assert_eq!(alpha.sample_rate, Some(48_000));
        assert!(alpha.problem.as_deref().unwrap().contains("g.onnx"), "{alpha:?}");

        // Renaming a manifest-less folder writes the sidecar, never a manifest.
        let renamed = rename_slot(&models, "zeta", "  Zeta\u{7}  Prime ").unwrap();
        assert_eq!(renamed.name, "Zeta Prime");
        assert!(!models.join("zeta").join(MANIFEST_FILE).exists());
        // Renaming a manifest folder patches only its name (unknown keys are kept).
        rename_slot(&models, "alpha", "Alpha").unwrap();
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(models.join("alpha").join(MANIFEST_FILE)).unwrap()).unwrap();
        assert_eq!((v["name"].as_str(), v["w_okada_extra"].as_i64(), v["sample_rate"].as_i64()), (Some("Alpha"), Some(7), Some(48_000)));
        assert!(rename_slot(&models, "alpha", " \t ").unwrap_err().starts_with("job:"));
        assert!(rename_slot(&models, "missing", "x").unwrap_err().starts_with("job:"));

        let copy = duplicate_slot(&models, "zeta", None).unwrap();
        assert_eq!((copy.id.as_str(), copy.name.as_str()), ("zeta-prime-copy", "Zeta Prime (copy)"));
        assert!(models.join("zeta-prime-copy").join("gen.onnx").is_file());
        let copy2 = duplicate_slot(&models, "zeta", Some("Second")).unwrap();
        assert_eq!(copy2.id, "second");

        delete_slot(&models, "zeta").unwrap();
        assert!(!models.join("zeta").exists());
        assert!(delete_slot(&models, "zeta").unwrap_err().starts_with("job:"));
        assert!(delete_slot(&models, "../models").is_err());
        let ids: Vec<String> = list_slots_with(&models, None).unwrap().into_iter().map(|s| s.id).collect();
        assert_eq!(ids, ["alpha", "empty", "second", "zeta-prime-copy"]);
    }

    #[test]
    fn sibling_indexes_are_adopted_only_when_unambiguous() {
        let t = TempDir::new("sib");
        let one = t.0.join("one");
        std::fs::create_dir_all(&one).unwrap();
        std::fs::write(one.join("Kira.pth"), b"x").unwrap();
        std::fs::write(one.join("added_IVF300_Flat_nprobe_1_Kira_v2.index"), b"x").unwrap();
        std::fs::write(one.join("trained_IVF300_Flat_nprobe_1_Kira_v2.index"), b"x").unwrap();
        assert_eq!(sibling_index(&one.join("Kira.pth")).unwrap(), one.join("added_IVF300_Flat_nprobe_1_Kira_v2.index"));
        // A Downloads folder: two voices, their indexes named after them, plus an unrelated one.
        let dl = t.0.join("dl");
        std::fs::create_dir_all(&dl).unwrap();
        for f in ["Kira.pth", "Other.pth", "added_Kira.index", "added_Other.index", "added_misc.index"] {
            std::fs::write(dl.join(f), b"x").unwrap();
        }
        assert_eq!(sibling_index(&dl.join("Kira.pth")).unwrap(), dl.join("added_Kira.index"));
        std::fs::write(dl.join("Zed.pth"), b"x").unwrap();
        assert_eq!(sibling_index(&dl.join("Zed.pth")), None, "no guessing among several indexes");
        // Only a trained index: nothing to adopt.
        let tr = t.0.join("tr");
        std::fs::create_dir_all(&tr).unwrap();
        std::fs::write(tr.join("v.pth"), b"x").unwrap();
        std::fs::write(tr.join("trained_v.index"), b"x").unwrap();
        assert_eq!(sibling_index(&tr.join("v.pth")), None);
    }

    #[test]
    fn recovery_restores_interrupted_swaps_and_removes_trash() {
        let t = TempDir::new("recover");
        let m = &t.0;
        // Killed between the renames: the voice is only in the backup.
        std::fs::create_dir_all(m.join(".v1.sigil-import-old")).unwrap();
        std::fs::write(m.join(".v1.sigil-import-old").join("weights.f32.bin"), b"w").unwrap();
        std::fs::create_dir_all(m.join(".v1.sigil-import-new")).unwrap();
        std::fs::create_dir_all(m.join(".v2.sigil-delete-abc")).unwrap();
        std::fs::create_dir_all(m.join(".v3.sigil-new-abc")).unwrap();
        std::fs::write(m.join(".sigil-import-job-1-2-3.json"), b"{}").unwrap();
        std::fs::write(m.join(".keep-me.json"), b"{}").unwrap();
        recover_all(m);
        assert!(!m.join(".sigil-import-job-1-2-3.json").exists());
        assert!(m.join(".keep-me.json").exists());
        assert!(m.join("v1").join("weights.f32.bin").is_file());
        for gone in [".v1.sigil-import-old", ".v1.sigil-import-new", ".v2.sigil-delete-abc", ".v3.sigil-new-abc"] {
            assert!(!m.join(gone).exists(), "{gone}");
        }
    }
}
