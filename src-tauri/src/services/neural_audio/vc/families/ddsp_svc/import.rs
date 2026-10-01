//! Importing a DDSP-SVC 6.3 voice (P5-A): a user's training checkpoint (`model_*.pt`) and its
//! `config.yaml` → the slot's `weights.f32.bin` laid out for Sigil's template graphs
//! ([`super::layout`]), an optional speaker table, an import record and the manifest.
//!
//! Sources: a `.pt` with `config.yaml` beside it (DDSP-SVC writes both into the experiment folder),
//! an experiment folder, or a `.zip` holding both. Several `model_*.pt` pick the highest step
//! (`model_best.pt` / a single checkpoint otherwise); anything else ambiguous is refused by name.
//!
//! Every weight is checked against the template: the config's architecture fields, the exact set
//! of checkpoint tensors (one the template does not know means another architecture), each source
//! shape before its values are read, and finite values. A weight-norm pair may also come folded
//! (`X.weight`, a checkpoint saved after `remove_weight_norm`). The voice's speaker table
//! (`n_spk > 1`, at most [`MAX_SPEAKER_TABLE_BYTES`]) becomes `speakers.f32.bin`; a voice trained
//! without pitch augmentation gets a zero formant embedding.

use std::fs::File;
use std::io::{BufReader, Cursor, Read, Seek};
use std::path::{Path, PathBuf};
use std::time::Instant;

use serde::Serialize;
use sha2::{Digest, Sha256};

use super::checkpoint::DdspCheckpoint;
use super::config::{DdspConfig, MAX_CONFIG_BYTES};
use super::layout::{DdspLayout, Transform};
use crate::services::neural_audio::vc::import::fold::weight_norm_fold;
use crate::services::neural_audio::vc::import::zipguard::{looks_like_zip, GuardedZip, ZipLimits};
use crate::services::neural_audio::vc::import::{self, ErrorCode, ImportError};

/// The slot's weights image (as RVC template voices).
pub const WEIGHTS_FILE: &str = "weights.f32.bin";
/// The slot's speaker table (`count × dim` f32, multi-speaker voices only).
pub const SPEAKERS_FILE: &str = "speakers.f32.bin";
/// The import record of a DDSP-SVC voice (not RVC's `sigil-import.json`: other code reads that one
/// as an RVC record).
pub const RECORD_FILE: &str = "sigil-ddsp-import.json";
pub const RECORD_FORMAT: &str = "sigil-ddsp-import";
/// Bumped when the importer writes slots differently.
pub const IMPORTER_VERSION: u32 = 1;
/// Largest checkpoint read from inside a zip (into memory).
const MAX_ZIPPED_CHECKPOINT: u64 = 1536 << 20;
/// Largest speaker table (bytes of f32) a voice may have: the loader's limit too (32768 speakers
/// of 512 values).
pub const MAX_SPEAKER_TABLE_BYTES: usize = 64 << 20;

/// Where the checkpoint came from.
#[derive(Clone, Debug, Serialize)]
pub struct SourceRecord {
    pub path: PathBuf,
    /// `pt`, `folder` or `zip`.
    pub kind: &'static str,
    /// The checkpoint's file (or zip entry) name.
    pub checkpoint: String,
    pub bytes: u64,
    /// sha256 of the checkpoint file (inside a zip: of the zip).
    pub sha256: String,
}

/// A checkpoint converted to the template's weights.
pub struct ImportedDdsp {
    pub config: DdspConfig,
    pub image: Vec<u8>,
    /// `(rows, count)` of a multi-speaker voice.
    pub speakers: Option<(Vec<f32>, usize)>,
    pub source: SourceRecord,
    pub tensors: usize,
    pub checkpoint_dtype: &'static str,
    /// `training` (`{"global_step", "model", ...}`) or `state_dict` (a bare state dict).
    pub checkpoint_kind: &'static str,
    pub global_step: Option<i64>,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Serialize)]
pub struct FileRecord {
    pub file: &'static str,
    pub bytes: u64,
    pub sha256: String,
}

/// `sigil-ddsp-import.json` (and the worker's record).
#[derive(Clone, Debug, Serialize, serde::Deserialize)]
pub struct DdspRecord {
    pub format: String,
    pub format_version: u32,
    pub importer_version: u32,
    pub family: String,
    pub name: String,
    pub template: String,
    pub weights_id: String,
    pub weights: serde_json::Value,
    pub speakers: Option<serde_json::Value>,
    pub speaker_count: u32,
    pub config: serde_json::Value,
    pub source: serde_json::Value,
    pub tensors: usize,
    pub checkpoint_dtype: String,
    /// `training` or `state_dict` (see [`ImportedDdsp::checkpoint_kind`]).
    #[serde(default)]
    pub checkpoint_kind: String,
    pub global_step: Option<i64>,
    pub warnings: Vec<String>,
    pub created_unix_ms: u64,
    pub elapsed_ms: u64,
}

impl DdspRecord {
    /// Read a slot's record (bounded; `None` when missing or not a DDSP-SVC record).
    pub fn read(slot_dir: &Path) -> Option<DdspRecord> {
        let path = slot_dir.join(RECORD_FILE);
        if std::fs::metadata(&path).ok()?.len() > 4 << 20 {
            return None;
        }
        let r: DdspRecord = serde_json::from_slice(&std::fs::read(path).ok()?).ok()?;
        (r.format == RECORD_FORMAT).then_some(r)
    }

    /// The voice's typed config fields.
    pub fn config(&self) -> Option<DdspConfig> {
        serde_json::from_value(self.config.clone()).ok()
    }
}

fn io_at(path: &Path, e: std::io::Error) -> ImportError {
    ImportError::new(ErrorCode::Io, format!("{}: {e}", path.display()))
}

/// The training step of `model_<n>.pt` (`None` for other names).
fn step_of(name: &str) -> Option<u64> {
    let stem = Path::new(name).file_stem()?.to_str()?;
    stem.strip_prefix("model_")?.parse().ok()
}

/// The checkpoint to import among `names` (`.pt` files): the only one, else the highest
/// `model_<step>.pt`, else `model_best.pt`.
pub fn choose_checkpoint(names: &[String]) -> Result<String, ImportError> {
    match names {
        [] => return Err(ImportError::new(ErrorCode::NotACheckpoint, "no DDSP-SVC checkpoint (model_*.pt) found")),
        [one] => return Ok(one.clone()),
        _ => {}
    }
    if let Some(best) = names.iter().filter_map(|n| step_of(n).map(|s| (s, n))).max_by_key(|(s, _)| *s) {
        return Ok(best.1.clone());
    }
    if let Some(b) = names.iter().find(|n| Path::new(n.as_str()).file_name().is_some_and(|f| f == "model_best.pt")) {
        return Ok(b.clone());
    }
    Err(ImportError::new(
        ErrorCode::AmbiguousArchive,
        format!("several checkpoints and none is a model_<step>.pt: {}", names.join(", ")),
    ))
}

fn sha256_reader(r: &mut (impl Read + Seek)) -> Result<(String, u64), ImportError> {
    r.seek(std::io::SeekFrom::Start(0)).map_err(ImportError::io)?;
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
    r.seek(std::io::SeekFrom::Start(0)).map_err(ImportError::io)?;
    Ok((hex::encode(h.finalize()), n))
}

fn read_config_file(path: &Path) -> Result<DdspConfig, ImportError> {
    let meta = std::fs::metadata(path).map_err(|_| {
        ImportError::new(
            ErrorCode::NotACheckpoint,
            format!(
                "the voice's config.yaml is missing: put the config.yaml DDSP-SVC wrote next to {} (it is in the same experiment folder)",
                path.parent().map(|p| p.display().to_string()).unwrap_or_default()
            ),
        )
    })?;
    if meta.len() > MAX_CONFIG_BYTES as u64 {
        return Err(ImportError::new(ErrorCode::NotACheckpoint, "config.yaml is too large"));
    }
    let text = std::fs::read_to_string(path).map_err(|e| io_at(path, e))?;
    DdspConfig::parse(&text).map_err(|e| ImportError::new(ErrorCode::NotACheckpoint, e))
}

/// Convert a voice into the template's weights (see the module docs).
pub fn import_voice(source: &Path, layout: &DdspLayout) -> Result<ImportedDdsp, ImportError> {
    let meta = std::fs::metadata(source).map_err(|e| io_at(source, e))?;
    if meta.is_dir() {
        let mut pts: Vec<String> = std::fs::read_dir(source)
            .map_err(|e| io_at(source, e))?
            .filter_map(Result::ok)
            .filter(|e| e.path().is_file())
            .filter_map(|e| e.file_name().to_str().map(str::to_string))
            .filter(|n| n.to_ascii_lowercase().ends_with(".pt"))
            .collect();
        pts.sort();
        let chosen = choose_checkpoint(&pts)?;
        let config = read_config_file(&source.join("config.yaml"))?;
        let path = source.join(&chosen);
        let mut f = BufReader::new(File::open(&path).map_err(|e| io_at(&path, e))?);
        let (sha256, bytes) = sha256_reader(&mut f)?;
        let src = SourceRecord { path: source.to_path_buf(), kind: "folder", checkpoint: chosen, bytes, sha256 };
        return convert(f, config, src, layout);
    }
    let mut f = BufReader::new(File::open(source).map_err(|e| io_at(source, e))?);
    let mut head = [0u8; 4];
    let got = f.read(&mut head).map_err(ImportError::io)?;
    f.seek(std::io::SeekFrom::Start(0)).map_err(ImportError::io)?;
    let (sha256, bytes) = sha256_reader(&mut f)?;
    let is_pt = source.extension().is_some_and(|e| e.eq_ignore_ascii_case("pt") || e.eq_ignore_ascii_case("pth"));
    if is_pt {
        if !looks_like_zip(&head[..got]) {
            return Err(import::pth::reject_non_zip(&head[..got]));
        }
        let config = read_config_file(&source.parent().unwrap_or(Path::new(".")).join("config.yaml"))?;
        let name = source.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let src = SourceRecord { path: source.to_path_buf(), kind: "pt", checkpoint: name, bytes, sha256 };
        return convert(f, config, src, layout);
    }
    if !looks_like_zip(&head[..got]) {
        return Err(ImportError::new(ErrorCode::NotACheckpoint, "not a DDSP-SVC checkpoint (.pt) or voice archive (.zip)"));
    }
    // A voice archive: a torch checkpoint itself is a zip whose entries are `<name>/data.pkl`...
    let mut zip = GuardedZip::open(f, ZipLimits::VOICE_ARCHIVE)?;
    if zip.entries().iter().any(|e| e.name.ends_with("/data.pkl")) {
        return Err(ImportError::new(
            ErrorCode::NotACheckpoint,
            "this is a bare checkpoint without its config.yaml: import the .pt from its experiment folder, or a .zip holding both",
        ));
    }
    let pts: Vec<String> = zip.entries().iter().filter(|e| e.name.to_ascii_lowercase().ends_with(".pt")).map(|e| e.name.clone()).collect();
    let chosen = choose_checkpoint(&pts)?;
    let dir = chosen.rsplit_once('/').map(|(d, _)| format!("{d}/")).unwrap_or_default();
    let configs: Vec<String> = zip
        .entries()
        .iter()
        .filter(|e| e.name.rsplit('/').next() == Some("config.yaml"))
        .map(|e| e.name.clone())
        .collect();
    let config_name = configs
        .iter()
        .find(|n| **n == format!("{dir}config.yaml"))
        .or_else(|| (configs.len() == 1).then(|| &configs[0]))
        .cloned()
        .ok_or_else(|| {
            ImportError::new(
                if configs.is_empty() { ErrorCode::NotACheckpoint } else { ErrorCode::AmbiguousArchive },
                if configs.is_empty() { "the archive has no config.yaml for the voice".to_string() } else { format!("several config.yaml files: {}", configs.join(", ")) },
            )
        })?;
    let text = zip.read(&config_name, MAX_CONFIG_BYTES as u64)?;
    let config = DdspConfig::parse(&String::from_utf8_lossy(&text)).map_err(|e| ImportError::new(ErrorCode::NotACheckpoint, e))?;
    let pt = zip.read(&chosen, MAX_ZIPPED_CHECKPOINT)?;
    let src = SourceRecord { path: source.to_path_buf(), kind: "zip", checkpoint: chosen, bytes, sha256 };
    convert(Cursor::new(pt), config, src, layout)
}

/// One weight of the layout from the checkpoint, in the initializer's layout.
fn weight_values<R: Read + Seek>(
    ckpt: &mut DdspCheckpoint<R>,
    t: &super::layout::TensorEntry,
    layout: &DdspLayout,
    warnings: &mut Vec<String>,
) -> Result<Vec<f32>, ImportError> {
    let shape_err = |key: &str, got: Vec<usize>, want: &[usize]| {
        ImportError::new(ErrorCode::ShapeMismatch, format!("{key} is {got:?}, Sigil's DDSP-SVC 6.3 template needs {want:?}"))
    };
    let values = match (&t.source.key, &t.source.weight_norm) {
        (Some(key), _) => {
            if !ckpt.has(key) && layout.optional_zero.iter().any(|k| k == key) {
                warnings.push(format!("{key}: not in the checkpoint (trained without it); its contribution is zero"));
                return Ok(vec![0.0; t.shape.iter().product()]);
            }
            let got = ckpt.shape(key).ok_or_else(|| ImportError::new(ErrorCode::MissingTensor, format!("the checkpoint has no {key}")))?;
            if got != t.source_shape {
                return Err(shape_err(key, got, &t.source_shape));
            }
            ckpt.read_f32(key)?
        }
        (None, Some(wn)) => {
            let folded = t.source.folded_key().unwrap_or(&wn.g).to_string();
            if !ckpt.has(&wn.g) && !ckpt.has(&wn.v) && ckpt.has(&folded) {
                // A checkpoint saved with the weight norm already removed.
                let got = ckpt.shape(&folded).unwrap_or_default();
                if got != t.source_shape {
                    return Err(shape_err(&folded, got, &t.source_shape));
                }
                ckpt.read_f32(&folded)?
            } else {
                let v_shape = ckpt.shape(&wn.v).ok_or_else(|| ImportError::new(ErrorCode::MissingTensor, format!("the checkpoint has no {}", wn.v)))?;
                if v_shape != t.source_shape {
                    return Err(shape_err(&wn.v, v_shape, &t.source_shape));
                }
                // The magnitude: one value per slice along `dim` (one in all for dim -1), checked
                // before it is read.
                let g_shape = ckpt.shape(&wn.g).ok_or_else(|| ImportError::new(ErrorCode::MissingTensor, format!("the checkpoint has no {}", wn.g)))?;
                let want_g = usize::try_from(wn.dim).ok().and_then(|d| t.source_shape.get(d).copied()).unwrap_or(1);
                if g_shape.iter().try_fold(1usize, |a, &d| a.checked_mul(d)) != Some(want_g) {
                    return Err(ImportError::new(
                        ErrorCode::ShapeMismatch,
                        format!("{} is {g_shape:?}, Sigil's DDSP-SVC 6.3 template needs {want_g} values", wn.g),
                    ));
                }
                let g = ckpt.read_f32(&wn.g)?;
                let v = ckpt.read_f32(&wn.v)?;
                weight_norm_fold(&g, &v, &t.source_shape, wn.dim).map_err(|e| ImportError::new(ErrorCode::TensorInvalid, format!("{folded}: {e}")))?
            }
        }
        _ => return Err(ImportError::new(ErrorCode::LayoutInvalid, format!("{}: no source", t.name))),
    };
    if let Some(bad) = values.iter().position(|v| !v.is_finite()) {
        return Err(ImportError::new(ErrorCode::NonFinite, format!("{}: a non-finite value at {bad}", t.source.label())));
    }
    Ok(match t.transform {
        Transform::Identity | Transform::Reshape => values,
        Transform::Transpose => {
            let (rows, cols) = (t.source_shape[0], t.source_shape[1]);
            let mut out = vec![0f32; values.len()];
            for r in 0..rows {
                for c in 0..cols {
                    out[c * rows + r] = values[r * cols + c];
                }
            }
            out
        }
    })
}

fn convert<R: Read + Seek>(reader: R, config: DdspConfig, source: SourceRecord, layout: &DdspLayout) -> Result<ImportedDdsp, ImportError> {
    if let Some(why) = config.mismatch(&layout.arch, &layout.vocoder) {
        return Err(ImportError::new(ErrorCode::UnsupportedModel, format!("this DDSP-SVC voice cannot run on Sigil's 6.3 template: {why}")));
    }
    let mut ckpt = DdspCheckpoint::open(reader)?;
    let known = layout.known_keys();
    let unknown: Vec<String> = ckpt.keys().filter(|k| !known.contains(k)).map(str::to_string).collect();
    if !unknown.is_empty() {
        return Err(ImportError::new(
            ErrorCode::UnsupportedModel,
            format!(
                "the checkpoint has {} weights Sigil's DDSP-SVC 6.3 template does not have ({}{}): it is another DDSP-SVC version or architecture",
                unknown.len(),
                unknown.iter().take(4).cloned().collect::<Vec<_>>().join(", "),
                if unknown.len() > 4 { ", ..." } else { "" }
            ),
        ));
    }
    // The speaker table's shape, before any weight is read: `[speakers, dim]`, at most what the
    // loader takes (`MAX_SPEAKER_TABLE_BYTES`).
    let speaker_count = match ckpt.shape(&layout.speakers.key) {
        Some(shape) => {
            let dim = layout.speakers.dim;
            if shape.len() != 2 || shape[1] != dim || shape[0] == 0 {
                return Err(ImportError::new(ErrorCode::ShapeMismatch, format!("{} is {shape:?}, expected [speakers, {dim}]", layout.speakers.key)));
            }
            if shape[0].checked_mul(dim * 4).is_none_or(|b| b > MAX_SPEAKER_TABLE_BYTES) {
                return Err(ImportError::new(
                    ErrorCode::ShapeMismatch,
                    format!("{} holds {} speakers; Sigil takes at most {}", layout.speakers.key, shape[0], MAX_SPEAKER_TABLE_BYTES / (dim * 4)),
                ));
            }
            Some(shape[0])
        }
        None if config.n_spk > 1 => {
            return Err(ImportError::new(
                ErrorCode::MissingTensor,
                format!("config.yaml says n_spk = {} but the checkpoint has no speaker table", config.n_spk),
            ));
        }
        None => None,
    };
    let mut warnings = Vec::new();
    let mut image = vec![0u8; layout.weights.bytes as usize];
    let mut dtype = "float32";
    for t in &layout.weights.tensors {
        if let Some(d) = t.source.keys().into_iter().chain(t.source.folded_key()).find_map(|k| ckpt.dtype(k)) {
            if d.name() != "float32" {
                dtype = d.name();
            }
        }
        let values = weight_values(&mut ckpt, t, layout, &mut warnings)?;
        if values.len() * 4 != t.bytes as usize {
            return Err(ImportError::new(ErrorCode::ShapeMismatch, format!("{}: {} values for {} bytes", t.name, values.len(), t.bytes)));
        }
        let section = layout.section(&t.graph).ok_or_else(|| ImportError::new(ErrorCode::LayoutInvalid, format!("no section {}", t.graph)))?;
        let at = (section.offset + t.offset) as usize;
        for (dst, v) in image[at..at + t.bytes as usize].chunks_exact_mut(4).zip(&values) {
            dst.copy_from_slice(&v.to_le_bytes());
        }
        ckpt.release_storages();
    }
    let speakers = match speaker_count {
        Some(count) => {
            let rows = ckpt.read_f32(&layout.speakers.key)?;
            if rows.len() != count * layout.speakers.dim {
                return Err(ImportError::new(ErrorCode::ShapeMismatch, format!("{}: {} values for {count} speakers", layout.speakers.key, rows.len())));
            }
            if rows.iter().any(|v| !v.is_finite()) {
                return Err(ImportError::new(ErrorCode::NonFinite, format!("{}: non-finite values", layout.speakers.key)));
            }
            if count as u64 != u64::from(config.n_spk) {
                warnings.push(format!("config.yaml says n_spk = {}, the checkpoint has {count} speakers", config.n_spk));
            }
            Some((rows, count))
        }
        None => None,
    };
    if config.infer_method != "euler" {
        warnings.push(format!("the voice's config asks for {} sampling; Sigil runs the Euler steps", config.infer_method));
    }
    if config.t_start > super::DEFAULT_T_START {
        warnings.push(format!(
            "the voice was trained with t_start {} (config.yaml model.t_start): its flow starts there, above the presets' {}",
            config.t_start,
            super::DEFAULT_T_START
        ));
    }
    Ok(ImportedDdsp {
        config,
        image,
        speakers,
        source,
        tensors: ckpt.len(),
        checkpoint_dtype: dtype,
        checkpoint_kind: ckpt.kind,
        global_step: ckpt.global_step,
        warnings,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Write the voice into `dir` (an empty staging folder): weights, speakers, record, manifest.
pub fn write_slot(dir: &Path, voice: &ImportedDdsp, layout: &DdspLayout, name: &str, options: serde_json::Value, started: Instant) -> Result<DdspRecord, ImportError> {
    use super::super::{family_manifest, write_family_manifest, FamilyKind};
    std::fs::create_dir_all(dir).map_err(|e| io_at(dir, e))?;
    import::write_atomic(&dir.join(WEIGHTS_FILE), &voice.image)?;
    let weights = FileRecord { file: WEIGHTS_FILE, bytes: voice.image.len() as u64, sha256: sha256_hex(&voice.image) };
    let speakers = match &voice.speakers {
        Some((rows, count)) => {
            let bytes: Vec<u8> = rows.iter().flat_map(|v| v.to_le_bytes()).collect();
            import::write_atomic(&dir.join(SPEAKERS_FILE), &bytes)?;
            Some(serde_json::json!({"file": SPEAKERS_FILE, "count": count, "dim": layout.speakers.dim, "sha256": sha256_hex(&bytes)}))
        }
        None => None,
    };
    let record = DdspRecord {
        format: RECORD_FORMAT.into(),
        format_version: 1,
        importer_version: IMPORTER_VERSION,
        family: FamilyKind::DdspSvc.as_str().into(),
        name: name.to_string(),
        template: layout.template.clone(),
        weights_id: layout.weights.weights_id.clone(),
        weights: serde_json::to_value(&weights).unwrap_or_default(),
        speakers,
        speaker_count: voice.speakers.as_ref().map(|(_, n)| *n as u32).unwrap_or(1),
        config: serde_json::to_value(&voice.config).unwrap_or_default(),
        source: serde_json::to_value(&voice.source).unwrap_or_default(),
        tensors: voice.tensors,
        checkpoint_dtype: voice.checkpoint_dtype.to_string(),
        checkpoint_kind: voice.checkpoint_kind.to_string(),
        global_step: voice.global_step,
        warnings: voice.warnings.clone(),
        created_unix_ms: import::unix_ms(),
        elapsed_ms: started.elapsed().as_millis() as u64,
    };
    let json = serde_json::to_vec_pretty(&record).map_err(|e| ImportError::new(ErrorCode::Io, e.to_string()))?;
    import::write_atomic(&dir.join(RECORD_FILE), &json)?;
    let mut m = family_manifest(FamilyKind::DdspSvc, name, options);
    m.sample_rate = layout.arch.sampling_rate;
    m.if_f0 = true;
    m.emb_channels = layout.arch.encoder_out_channels;
    m.embedder = Some("contentvec".into());
    m.template = Some(layout.template.clone());
    m.weights = Some(WEIGHTS_FILE.into());
    m.speakers = voice.speakers.as_ref().map(|_| SPEAKERS_FILE.to_string());
    m.speaker_count = Some(record.speaker_count);
    m.vocoder = Some("pc-nsf-hifigan".into());
    m.source_sha256 = Some(voice.source.sha256.clone());
    m.import_record = Some(RECORD_FILE.into());
    write_family_manifest(dir, &m).map_err(|e| ImportError::new(ErrorCode::Io, e))?;
    Ok(record)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checkpoint_choice() {
        let v = |s: &[&str]| s.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(choose_checkpoint(&v(&["a.pt"])).unwrap(), "a.pt");
        assert_eq!(choose_checkpoint(&v(&["model_100.pt", "model_9000.pt", "model_best.pt"])).unwrap(), "model_9000.pt");
        assert_eq!(choose_checkpoint(&v(&["exp/model_best.pt", "x.pt"])).unwrap(), "exp/model_best.pt");
        assert_eq!(choose_checkpoint(&v(&["a.pt", "b.pt"])).unwrap_err().code, ErrorCode::AmbiguousArchive);
        assert_eq!(choose_checkpoint(&[]).unwrap_err().code, ErrorCode::NotACheckpoint);
    }
}
