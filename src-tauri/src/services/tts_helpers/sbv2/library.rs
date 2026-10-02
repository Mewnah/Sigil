//! Imported voice models of the `sbv2` engine: `app_data/tts/sbv2/<id>/` holding the model file
//! (`model.aivmx`, or `model.onnx` with `style_vectors.{json,npy}` and an optional `config.json`)
//! and `sigil-sbv2.json` (what the file says about itself, so listing never re-reads a model).
//!
//! Imports are copies (the user's file stays where it is), written into a hidden staging folder
//! and renamed into place, so a half-copied model never lists. A model already in the library
//! (same content hash) is returned instead of being copied again.
//!
//! Every change of the library (an import, a delete, the sweep of leftovers) holds one
//! process-wide lock, so two drops in a row import one after the other: the second one finds the
//! first in the library instead of racing it, and its sweep never removes the first one's staging
//! folder. Staging and trash folders carry the process id, so a sweep removes this process's
//! leftovers at once and another process's only once they have been idle for [`STALE_AFTER`].

use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::format::{self, ModelDescription, SpeakerInfo, StyleVectors};

/// Folder under app data.
pub const LIBRARY_DIR: &str = "tts/sbv2";
/// Description file of an imported model.
pub const INDEX_FILE: &str = "sigil-sbv2.json";
/// Model file of an imported `.aivmx`.
pub const AIVMX_FILE: &str = "model.aivmx";
/// Model file of an imported ONNX export.
pub const ONNX_FILE: &str = "model.onnx";
/// Largest model file imported.
pub const MAX_MODEL_BYTES: u64 = 4 * 1024 * 1024 * 1024;
/// Largest style-vector or config file read.
const MAX_SIDE_FILE_BYTES: u64 = 64 * 1024 * 1024;

/// A staging or trash folder of another process is a leftover once nothing in it changed for this long.
const STALE_AFTER: Duration = Duration::from_secs(10 * 60);
/// Prefix of a staging folder (`.import-<pid>-<n>-<sha12>`).
const STAGING_PREFIX: &str = ".import-";
/// Prefix of a deleted model on its way out (`.trash-<pid>-<id>-<ms>`).
const TRASH_PREFIX: &str = ".trash-";

/// Held by every change of the library (see the module docs).
static LIBRARY_LOCK: Mutex<()> = Mutex::new(());
/// Makes every staging folder of this process unique.
static STAGING_SEQ: AtomicU64 = AtomicU64::new(0);

fn lock() -> MutexGuard<'static, ()> {
    LIBRARY_LOCK.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Wire value of `Sbv2Model::format` for AivisSpeech voices.
pub const FORMAT_AIVMX: &str = "aivmx";
/// Wire value of `Sbv2Model::format` for plain Style-Bert-VITS2 ONNX exports.
pub const FORMAT_SBV2: &str = "sbv2";

/// `app_data/tts/sbv2`.
pub fn library_dir(app_data: &Path) -> PathBuf {
    app_data.join(LIBRARY_DIR)
}

/// A model id: 1–64 characters of `[a-z0-9-]`, not starting or ending with `-`.
pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && !id.starts_with('-')
        && !id.ends_with('-')
        && id.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-')
}

/// What `sigil-sbv2.json` records about an imported model.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct StoredModel {
    pub id: String,
    pub name: String,
    /// [`FORMAT_AIVMX`] or [`FORMAT_SBV2`].
    pub format: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub creators: Vec<String>,
    #[serde(default)]
    pub version: String,
    #[serde(default)]
    pub uuid: String,
    pub architecture: String,
    pub jp_extra: bool,
    pub sample_rate: u32,
    pub speakers: Vec<SpeakerInfo>,
    /// Model file name inside the folder.
    pub model_file: String,
    /// Style vectors file inside the folder (exports only; `.aivmx` carries its own).
    #[serde(default)]
    pub styles_file: Option<String>,
    /// The imported file's name.
    #[serde(default)]
    pub source_name: String,
    pub size_bytes: u64,
    pub sha256: String,
    pub imported_unix_ms: u64,
}

impl StoredModel {
    fn from_description(id: String, d: &ModelDescription, format: &str, model_file: &str, styles_file: Option<&str>) -> Self {
        Self {
            id,
            name: d.name.clone(),
            format: format.to_string(),
            description: d.description.clone(),
            creators: d.creators.clone(),
            version: d.version.clone(),
            uuid: d.uuid.clone(),
            architecture: d.architecture.clone(),
            jp_extra: d.jp_extra,
            sample_rate: d.sample_rate,
            speakers: d.speakers.clone(),
            model_file: model_file.to_string(),
            styles_file: styles_file.map(str::to_string),
            source_name: String::new(),
            size_bytes: 0,
            sha256: String::new(),
            imported_unix_ms: 0,
        }
    }
}

/// The style vectors of an imported model (from the `.aivmx` metadata or its styles file).
pub fn load_styles(dir: &Path, m: &StoredModel) -> Result<StyleVectors, String> {
    match m.styles_file.as_deref() {
        Some(f) => read_styles_file(&dir.join(f)),
        None => Ok(format::describe_aivmx_file(&dir.join(&m.model_file))?.styles),
    }
}

fn read_capped(path: &Path, cap: u64) -> Result<Vec<u8>, String> {
    let len = std::fs::metadata(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?.len();
    if len > cap {
        return Err(format!("{} is too large ({} MB)", path.display(), len / (1024 * 1024)));
    }
    std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

fn read_styles_file(path: &Path) -> Result<StyleVectors, String> {
    let bytes = read_capped(path, MAX_SIDE_FILE_BYTES)?;
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("npy")) {
        format::parse_npy(&bytes)
    } else {
        format::parse_style_json(&bytes)
    }
}

// -------------------------------------------------------------------------------------------------
// Listing
// -------------------------------------------------------------------------------------------------

/// Every imported model, sorted by name, then id.
pub fn list(root: &Path) -> Vec<StoredModel> {
    let mut out: Vec<StoredModel> = std::fs::read_dir(root)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| {
            let id = e.file_name().to_string_lossy().into_owned();
            if !valid_id(&id) {
                return None;
            }
            get(root, &id).ok()
        })
        .collect();
    out.sort_by(|a, b| a.name.to_lowercase().cmp(&b.name.to_lowercase()).then_with(|| a.id.cmp(&b.id)));
    out
}

/// One imported model (its description file, checked against its folder).
pub fn get(root: &Path, id: &str) -> Result<StoredModel, String> {
    if !valid_id(id) {
        return Err(format!("invalid voice model id {id:?}"));
    }
    let dir = root.join(id);
    let bytes = std::fs::read(dir.join(INDEX_FILE)).map_err(|_| format!("the voice model {id:?} is not imported"))?;
    let mut m: StoredModel = serde_json::from_slice(&bytes).map_err(|e| format!("the voice model {id:?} is damaged ({e})"))?;
    m.id = id.to_string();
    let safe = |f: &str| !f.is_empty() && !f.contains(['/', '\\']) && f != "." && f != "..";
    if !safe(&m.model_file) || !m.styles_file.as_deref().is_none_or(safe) {
        return Err(format!("the voice model {id:?} is damaged (bad file name)"));
    }
    if !dir.join(&m.model_file).is_file() {
        return Err(format!("the voice model {id:?} is missing its model file"));
    }
    Ok(m)
}

/// The folder of model `id`.
pub fn model_dir(root: &Path, id: &str) -> Result<PathBuf, String> {
    if !valid_id(id) {
        return Err(format!("invalid voice model id {id:?}"));
    }
    Ok(root.join(id))
}

// -------------------------------------------------------------------------------------------------
// Import
// -------------------------------------------------------------------------------------------------

/// What a dropped path resolved to.
#[derive(Debug)]
enum Source {
    Aivmx(PathBuf),
    Export { model: PathBuf, styles: PathBuf, config: Option<PathBuf> },
}

fn ext_of(p: &Path) -> String {
    p.extension().map(|e| e.to_string_lossy().to_ascii_lowercase()).unwrap_or_default()
}

fn refuse_by_extension(p: &Path) -> Option<String> {
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    match ext_of(p).as_str() {
        "aivm" => Some(format!(
            "{name} is an .aivm voice (PyTorch weights). Sigil runs the ONNX version: download the .aivmx file of this voice from AivisHub, or convert it with AIVM Generator."
        )),
        "sbv2" => Some(format!(
            "{name} is an sbv2-api bundle, which this version cannot open. Import the voice's .aivmx file, or its ONNX export (model .onnx with style_vectors.json)."
        )),
        "safetensors" | "pth" | "pt" | "ckpt" => Some(format!(
            "{name} holds PyTorch weights. Convert the Style-Bert-VITS2 voice to .aivmx with AIVM Generator (or to ONNX), then import that."
        )),
        _ => None,
    }
}

/// Style vectors and config next to an ONNX export `model` (`style_vectors_<name>.json` for
/// `model_<name>.onnx`, else `style_vectors.json` / `.npy`; `config.json`).
fn export_side_files(model: &Path) -> Option<(PathBuf, Option<PathBuf>)> {
    let dir = model.parent()?;
    let stem = model.file_stem()?.to_string_lossy().into_owned();
    let mut candidates = Vec::new();
    if let Some(name) = stem.strip_prefix("model_") {
        candidates.push(dir.join(format!("style_vectors_{name}.json")));
        candidates.push(dir.join(format!("style_vectors_{name}.npy")));
    }
    candidates.push(dir.join(format!("{stem}.style_vectors.json")));
    candidates.push(dir.join("style_vectors.json"));
    candidates.push(dir.join("style_vectors.npy"));
    let styles = candidates.into_iter().find(|p| p.is_file())?;
    let config = Some(dir.join("config.json")).filter(|p| p.is_file());
    Some((styles, config))
}

fn files_with_ext(dir: &Path, ext: &str) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && ext_of(p) == ext)
        .collect();
    v.sort();
    v
}

/// Has AIVM metadata (an `.aivmx` under another name).
fn is_aivmx(path: &Path) -> bool {
    std::fs::File::open(path)
        .ok()
        .and_then(|f| format::read_onnx_metadata(f).ok())
        .is_some_and(|m| m.contains_key("aivm_manifest"))
}

fn resolve_source(path: &Path) -> Result<Source, String> {
    let meta = std::fs::metadata(path).map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    if meta.is_dir() {
        let aivmx = files_with_ext(path, "aivmx");
        match aivmx.len() {
            1 => return Ok(Source::Aivmx(aivmx.into_iter().next().expect("one"))),
            n if n > 1 => return Err(format!("{} holds {n} .aivmx voices; drop one file at a time.", path.display())),
            _ => {}
        }
        let onnx = files_with_ext(path, "onnx");
        for model in &onnx {
            if is_aivmx(model) {
                return Ok(Source::Aivmx(model.clone()));
            }
            if let Some((styles, config)) = export_side_files(model) {
                return Ok(Source::Export { model: model.clone(), styles, config });
            }
        }
        if let Some(p) = ["aivm", "safetensors", "sbv2", "pth"].iter().flat_map(|e| files_with_ext(path, e)).next() {
            if let Some(why) = refuse_by_extension(&p) {
                return Err(why);
            }
        }
        return Err(format!(
            "{} holds no Style-Bert-VITS2 voice (an .aivmx file, or an ONNX model with style_vectors.json).",
            path.display()
        ));
    }
    if let Some(why) = refuse_by_extension(path) {
        return Err(why);
    }
    match ext_of(path).as_str() {
        "aivmx" => Ok(Source::Aivmx(path.to_path_buf())),
        "onnx" => {
            if is_aivmx(path) {
                return Ok(Source::Aivmx(path.to_path_buf()));
            }
            let (styles, config) = export_side_files(path).ok_or_else(|| {
                format!(
                    "{} has no style vectors next to it (style_vectors.json or style_vectors.npy). Drop the whole folder of the export.",
                    path.display()
                )
            })?;
            Ok(Source::Export { model: path.to_path_buf(), styles, config })
        }
        "json" | "npy" => Err("Drop the voice's model file (.aivmx or .onnx) or its folder, not the style vectors.".into()),
        _ => Err(format!("{} is not a Style-Bert-VITS2 voice (.aivmx, or an ONNX export).", path.display())),
    }
}

/// `name` as an id stem: lower-case ASCII letters and digits, other runs as `-` ("voice" when
/// nothing is left, e.g. a Japanese name).
fn slug(name: &str) -> String {
    let mut out = String::new();
    for c in name.chars() {
        if c.is_ascii_alphanumeric() {
            out.push(c.to_ascii_lowercase());
        } else if !out.ends_with('-') && !out.is_empty() {
            out.push('-');
        }
        if out.len() >= 40 {
            break;
        }
    }
    let out = out.trim_matches('-').to_string();
    if out.is_empty() {
        "voice".into()
    } else {
        out
    }
}

fn sha256_file(path: &Path) -> Result<(String, u64), String> {
    let mut f = std::fs::File::open(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    let mut n = 0u64;
    loop {
        let k = f.read(&mut buf).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if k == 0 {
            break;
        }
        h.update(&buf[..k]);
        n += k as u64;
    }
    Ok((hex::encode(h.finalize()), n))
}

fn now_ms() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_millis() as u64).unwrap_or(0)
}

/// Import the voice at `path` (a file or a folder) into `root`. `check` validates the copied model
/// file before it is committed (the engine loads it once to read its inputs); a model with the
/// same content already in the library is returned as is. Imports run one at a time (see the
/// module docs), each starting with a sweep of leftovers.
pub fn import(root: &Path, path: &Path, check: &dyn Fn(&Path, &StoredModel) -> Result<(), String>) -> Result<StoredModel, String> {
    let _guard = lock();
    sweep_locked(root, SystemTime::now());
    import_locked(root, path, check)
}

fn import_locked(root: &Path, path: &Path, check: &dyn Fn(&Path, &StoredModel) -> Result<(), String>) -> Result<StoredModel, String> {
    let source = resolve_source(path)?;
    let (model_src, mut desc, format, styles_src) = match &source {
        Source::Aivmx(p) => (p.clone(), format::describe_aivmx_file(p)?, FORMAT_AIVMX, None),
        Source::Export { model, styles, config } => {
            let style_vectors = read_styles_file(styles)?;
            let cfg = match config {
                Some(c) => Some(read_capped(c, MAX_SIDE_FILE_BYTES)?),
                None => None,
            };
            let folder = model
                .parent()
                .and_then(|d| d.file_name())
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default();
            let stem = model.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
            let fallback = stem.strip_prefix("model_").map(str::to_string).filter(|s| !s.is_empty()).unwrap_or(if folder.is_empty() { stem } else { folder });
            (model.clone(), format::describe_export(style_vectors, cfg.as_deref(), &fallback)?, FORMAT_SBV2, Some(styles.clone()))
        }
    };
    let size = std::fs::metadata(&model_src).map_err(|e| e.to_string())?.len();
    if size > MAX_MODEL_BYTES {
        return Err(format!("the voice model is too large ({} MB)", size / (1024 * 1024)));
    }
    let (sha, _) = sha256_file(&model_src)?;
    if let Some(existing) = list(root).into_iter().find(|m| m.sha256 == sha) {
        return Ok(existing);
    }
    std::fs::create_dir_all(root).map_err(|e| format!("cannot create {}: {e}", root.display()))?;
    let id = {
        let stem = slug(&desc.name);
        let mut id = format!("{stem}-{}", &sha[..6]);
        let mut n = 2;
        while root.join(&id).exists() {
            id = format!("{stem}-{}-{n}", &sha[..6]);
            n += 1;
        }
        id
    };
    desc.name = if desc.name.is_empty() { id.clone() } else { desc.name.clone() };

    let seq = STAGING_SEQ.fetch_add(1, Ordering::Relaxed);
    let staging = root.join(format!("{STAGING_PREFIX}{}-{seq}-{}", std::process::id(), &sha[..12]));
    let _ = std::fs::remove_dir_all(&staging);
    std::fs::create_dir_all(&staging).map_err(|e| format!("cannot create {}: {e}", staging.display()))?;
    let result = (|| -> Result<StoredModel, String> {
        let model_file = if format == FORMAT_AIVMX { AIVMX_FILE } else { ONNX_FILE };
        std::fs::copy(&model_src, staging.join(model_file)).map_err(|e| format!("cannot copy the voice model: {e}"))?;
        let styles_file = match &styles_src {
            Some(s) => {
                let name = if ext_of(s) == "npy" { "style_vectors.npy" } else { "style_vectors.json" };
                std::fs::copy(s, staging.join(name)).map_err(|e| format!("cannot copy the style vectors: {e}"))?;
                Some(name)
            }
            None => None,
        };
        let mut stored = StoredModel::from_description(id.clone(), &desc, format, model_file, styles_file);
        stored.source_name = model_src.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        stored.size_bytes = size;
        stored.sha256 = sha.clone();
        stored.imported_unix_ms = now_ms();
        check(&staging, &stored)?;
        let json = serde_json::to_vec_pretty(&stored).map_err(|e| e.to_string())?;
        std::fs::write(staging.join(INDEX_FILE), json).map_err(|e| format!("cannot write the voice description: {e}"))?;
        std::fs::rename(&staging, root.join(&id)).map_err(|e| format!("cannot add the voice to the library: {e}"))?;
        Ok(stored)
    })();
    if result.is_err() {
        let _ = std::fs::remove_dir_all(&staging);
    }
    result
}

/// Delete model `id` (renamed away first, so a failed delete never leaves a half model listed).
pub fn remove(root: &Path, id: &str) -> Result<(), String> {
    let dir = model_dir(root, id)?;
    let _guard = lock();
    if !dir.is_dir() {
        return Err(format!("the voice model {id:?} is not imported"));
    }
    let trash = root.join(format!("{TRASH_PREFIX}{}-{id}-{}", std::process::id(), now_ms()));
    std::fs::rename(&dir, &trash).map_err(|e| format!("cannot remove the voice model (is it in use?): {e}"))?;
    if let Err(e) = std::fs::remove_dir_all(&trash) {
        tracing::warn!(target: "sigil::tts::sbv2", "removed voice model {id}, but its files stay in {}: {e}", trash.display());
    }
    Ok(())
}

/// How long nothing in folder `dir` (or the folder itself) has changed, at `now`.
fn idle_for(dir: &Path, now: SystemTime) -> Option<Duration> {
    let mut newest = std::fs::metadata(dir).and_then(|m| m.modified()).ok()?;
    for e in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        if let Ok(t) = e.metadata().and_then(|m| m.modified()) {
            newest = newest.max(t);
        }
    }
    Some(now.duration_since(newest).unwrap_or(Duration::ZERO))
}

/// Remove staging and trash folders left by an interrupted import or delete. Call with the library
/// lock held: then no import or delete of this process is running, so every folder of this process
/// is a leftover; another process's are removed only once idle for [`STALE_AFTER`] at `now`.
fn sweep_locked(root: &Path, now: SystemTime) {
    let pid = std::process::id().to_string();
    for e in std::fs::read_dir(root).into_iter().flatten().flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(STAGING_PREFIX).or_else(|| name.strip_prefix(TRASH_PREFIX)) else {
            continue;
        };
        let ours = rest.split('-').next() == Some(pid.as_str());
        if ours || idle_for(&e.path(), now).is_some_and(|idle| idle >= STALE_AFTER) {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::tts_helpers::sbv2::format::{npy_f32, test_manifest, with_metadata};
    use base64::Engine as _;

    pub(crate) struct Scratch(pub PathBuf);
    impl Scratch {
        pub(crate) fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!("sigil-sbv2-lib-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&p);
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Scratch {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tiny_onnx() -> Vec<u8> {
        use crate::services::neural_audio::test_onnx::{Dim, Graph, FLOAT};
        let mut g = Graph::new();
        g.input("x", FLOAT, &[Dim::Fixed(1)]).output("y", FLOAT, &[Dim::Fixed(1)]).node("Identity", &["x"], &["y"], &[]);
        g.model_bytes(13)
    }

    fn aivmx(name: &str) -> Vec<u8> {
        let npy = npy_f32(2, 4, &[0.5; 8]);
        with_metadata(
            tiny_onnx(),
            &[
                ("aivm_manifest", &test_manifest(name, &[(name, &["ノーマル", "Happy"])], true)),
                ("aivm_hyper_parameters", r#"{"version":"2.0-JP-Extra","data":{"sampling_rate":44100}}"#),
                ("aivm_style_vectors", &base64::engine::general_purpose::STANDARD.encode(npy)),
            ],
        )
    }

    fn ok(_: &Path, _: &StoredModel) -> Result<(), String> {
        Ok(())
    }

    #[test]
    fn aivmx_files_import_list_and_remove() {
        let s = Scratch::new("aivmx");
        let src = s.0.join("src");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("Mao.aivmx"), aivmx("まお")).unwrap();
        let root = s.0.join("lib");

        let m = import(&root, &src.join("Mao.aivmx"), &ok).unwrap();
        assert_eq!(m.name, "まお");
        assert_eq!(m.format, FORMAT_AIVMX);
        assert!(m.id.starts_with("voice-"), "{}", m.id);
        assert!(valid_id(&m.id));
        assert_eq!(m.speakers[0].styles.len(), 2);
        assert_eq!(m.source_name, "Mao.aivmx");
        assert!(root.join(&m.id).join(AIVMX_FILE).is_file());
        assert_eq!(load_styles(&root.join(&m.id), &m).unwrap().rows, 2);

        // The same file again (or its folder) is the same model.
        assert_eq!(import(&root, &src, &ok).unwrap().id, m.id);
        assert_eq!(list(&root).len(), 1);
        assert_eq!(get(&root, &m.id).unwrap(), m);

        remove(&root, &m.id).unwrap();
        assert!(list(&root).is_empty());
        assert!(remove(&root, &m.id).is_err());
        assert!(get(&root, "../x").is_err());
    }

    #[test]
    fn exports_import_with_their_side_files() {
        let s = Scratch::new("export");
        let src = s.0.join("tsukuyomi");
        std::fs::create_dir_all(&src).unwrap();
        std::fs::write(src.join("model_tsukuyomi.onnx"), tiny_onnx()).unwrap();
        std::fs::write(src.join("style_vectors_tsukuyomi.json"), br#"{"shape":[2,2],"data":[[0,0],[1,1]]}"#).unwrap();
        std::fs::write(
            src.join("config.json"),
            br#"{"model_name":"Tsukuyomi","version":"2.0-JP-Extra","data":{"sampling_rate":44100,"spk2id":{"tsukuyomi":0},"style2id":{"Neutral":0,"Joy":1}}}"#,
        )
        .unwrap();
        let root = s.0.join("lib");
        let m = import(&root, &src.join("model_tsukuyomi.onnx"), &ok).unwrap();
        assert_eq!(m.format, FORMAT_SBV2);
        assert_eq!(m.name, "Tsukuyomi");
        assert!(m.id.starts_with("tsukuyomi-"));
        assert_eq!(m.styles_file.as_deref(), Some("style_vectors.json"));
        assert_eq!(m.speakers[0].styles[1].name, "Joy");
        assert_eq!(load_styles(&root.join(&m.id), &m).unwrap().data, [0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn unsupported_files_are_refused_with_a_reason() {
        let s = Scratch::new("refuse");
        let root = s.0.join("lib");
        for (name, needle) in [("v.aivm", "AivisHub"), ("v.safetensors", "PyTorch"), ("v.sbv2", "sbv2-api"), ("v.txt", "not a Style-Bert-VITS2")] {
            let p = s.0.join(name);
            std::fs::write(&p, b"x").unwrap();
            let e = import(&root, &p, &ok).unwrap_err();
            assert!(e.contains(needle), "{name}: {e}");
        }
        // An ONNX without style vectors, and one that is not ONNX at all.
        let p = s.0.join("lonely.onnx");
        std::fs::write(&p, tiny_onnx()).unwrap();
        assert!(import(&root, &p, &ok).unwrap_err().contains("style vectors"));
        let p = s.0.join("fake.aivmx");
        std::fs::write(&p, b"not a model").unwrap();
        assert!(import(&root, &p, &ok).is_err());
        // A failed check leaves nothing behind.
        let good = s.0.join("good.aivmx");
        std::fs::write(&good, aivmx("x")).unwrap();
        let e = import(&root, &good, &|_: &Path, _: &StoredModel| Err("broken graph".into())).unwrap_err();
        assert_eq!(e, "broken graph");
        assert!(list(&root).is_empty());
        assert_eq!(std::fs::read_dir(&root).unwrap().count(), 0, "no staging folder left");
    }

    #[test]
    fn a_second_drop_waits_for_the_running_import() {
        // Voice A is in its import check (the ORT probe takes seconds) when voice B is dropped: B
        // must neither sweep A's staging folder nor fail; both end up in the library.
        let s = Scratch::new("concurrent");
        let root = s.0.join("lib");
        let a = s.0.join("a.aivmx");
        let b = s.0.join("b.aivmx");
        std::fs::write(&a, aivmx("a")).unwrap();
        std::fs::write(&b, aivmx("b")).unwrap();
        let (in_check_tx, in_check_rx) = std::sync::mpsc::channel::<()>();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let (ma, mb) = std::thread::scope(|sc| {
            let (root, a, b) = (&root, &a, &b);
            let ha = sc.spawn(move || {
                import(root, a, &move |staging: &Path, _: &StoredModel| {
                    in_check_tx.send(()).unwrap();
                    let _ = go_rx.recv_timeout(Duration::from_secs(5));
                    if staging.join(AIVMX_FILE).is_file() {
                        Ok(())
                    } else {
                        Err("A's staging folder was swept".into())
                    }
                })
            });
            in_check_rx.recv().unwrap();
            let hb = sc.spawn(move || import(root, b, &ok));
            // B is waiting for the library lock; without it, its sweep would run now.
            std::thread::sleep(Duration::from_millis(300));
            go_tx.send(()).unwrap();
            (ha.join().unwrap(), hb.join().unwrap())
        });
        let (ma, mb) = (ma.unwrap(), mb.unwrap());
        assert_ne!(ma.id, mb.id);
        assert_eq!(list(&root).len(), 2);
        let hidden = std::fs::read_dir(&root).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().starts_with('.')).count();
        assert_eq!(hidden, 0, "no staging folder left");
    }

    #[test]
    fn the_same_file_dropped_twice_at_once_is_one_model() {
        let s = Scratch::new("twice");
        let root = s.0.join("lib");
        let a = s.0.join("a.aivmx");
        std::fs::write(&a, aivmx("a")).unwrap();
        let ids: Vec<String> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..3).map(|_| sc.spawn(|| import(&root, &a, &ok).map(|m| m.id))).collect();
            hs.into_iter().map(|h| h.join().unwrap().unwrap()).collect()
        });
        assert!(ids.iter().all(|id| *id == ids[0]), "{ids:?}");
        assert_eq!(list(&root).len(), 1);
    }

    #[test]
    fn sweeps_remove_only_leftovers() {
        let s = Scratch::new("sweep");
        let root = s.0.join("lib");
        let pid = std::process::id();
        let other = pid.wrapping_add(1);
        let folder = |name: String| {
            let d = root.join(name);
            std::fs::create_dir_all(&d).unwrap();
            std::fs::write(d.join("f"), b"x").unwrap();
            d
        };
        let own = [folder(format!(".import-{pid}-0-0123456789ab")), folder(format!(".trash-{pid}-voice-1-1"))];
        let foreign = [folder(format!(".import-{other}-0-0123456789ab")), folder(format!(".trash-{other}-voice-1-1"))];
        let model = folder("voice-abc123".into());
        let now = SystemTime::now();
        // Under the lock nothing of this process is running: its folders are leftovers. Another
        // process's stay while they may still be in use.
        sweep_locked(&root, now);
        assert!(own.iter().all(|d| !d.exists()));
        assert!(foreign.iter().all(|d| d.exists()));
        assert!(model.exists());
        sweep_locked(&root, now + STALE_AFTER + Duration::from_secs(1));
        assert!(foreign.iter().all(|d| !d.exists()), "idle ones are leftovers");
        assert!(model.exists(), "models are never swept");
    }

    #[test]
    fn ids_are_safe_slugs() {
        assert_eq!(slug("Tsukuyomi-chan v2"), "tsukuyomi-chan-v2");
        assert_eq!(slug("まお"), "voice");
        assert_eq!(slug("--A__B--"), "a-b");
        assert!(valid_id("abc-123"));
        assert!(!valid_id("-abc"));
        assert!(!valid_id("Abc"));
        assert!(!valid_id("a/b"));
        assert!(!valid_id(""));
    }
}
