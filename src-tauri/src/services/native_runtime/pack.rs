//! Runtime packs of sigil_engine DLLs: the pack manifest, hash verification, catalog discovery,
//! variant selection with fallback, and the `list_runtime_packs` rows.
//!
//! # Catalog convention
//! A Sigil-built engine pack is a catalog entry of kind `runtime` with the id
//! `runtime/sigil-<engine>-<variant>` and `variant: "<variant>"` (one entry per build variant, e.g.
//! `runtime/sigil-echo-cpu-avx2`). Its folder comes from `model_store::installed_path(id)` and holds
//! `sigil_<engine>.dll` (or the `library` named by the manifest) plus `sigil-pack.json`. A `cuda13`
//! variant lists `runtime/nvidia-cuda13` in `requires`; the NVIDIA runtime is preloaded before the
//! DLL is loaded.
//!
//! # Pack manifest (`sigil-pack.json`)
//! Written by `scripts/native/build-runtime-packs.ps1` from the built DLL (its own
//! `sigil_engine_get_info`, read by `sigil_engine_check.exe`):
//! `{schema: 1, engine, version, variant, abi_version, library, kind, capabilities, build_info,
//! files: [{path, size, sha256}]}`. Before a DLL from the folder is loaded, every listed file is
//! checked against its size and sha256 (cached per file size and modification time), and after the
//! load the DLL's own info must match the manifest. The archive the pack was installed from is
//! hash-pinned in the catalog for published packs, so the manifest is trustworthy at install time;
//! the load-time check catches later corruption and half-copied folders. An installed pack must
//! carry its manifest (a pack without one is refused, never loaded unverified); only an explicit
//! development folder (`--runtime-dir`, [`OpenRequest::runtime_dir`]) may lack one, and it is then
//! loaded logged as not hash-verified.
//!
//! The manifest's `variant` must equal the catalog variant the pack is installed as, and this CPU
//! must have the instruction sets of both, before the DLL is loaded
//! ([`VariantSupport::cpu_supports`]): a mislabeled `cpu-avx512` DLL would otherwise load as `cpu`
//! and crash the process on its first vector loop.
//!
//! # Selection
//! [`open_engine`] tries the installed variants in [`RuntimeVariant::PREFERENCE`] order among those
//! this PC supports ([`VariantSupport::detect`]); a variant whose DLL fails to load, or whose backend
//! is unavailable at create time, falls through to the next one. Every attempt is reported. A forced
//! variant ([`OpenRequest::variant`]) or folder is tried even without its GPU, never without its
//! instruction sets.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::SystemTime;

use serde::{Deserialize, Serialize};

use super::abi;
use super::loader::{valid_engine_name, EngineInfo, EngineInstance, EngineLibrary};
use super::variant::{candidates, RuntimeVariant, VariantSupport};
use crate::services::model_store::{Catalog, EntryKind, PublishStatus};

/// File name of the pack manifest.
pub const PACK_MANIFEST: &str = "sigil-pack.json";
/// Schema of [`PackManifest`].
pub const PACK_MANIFEST_SCHEMA: u32 = 1;
/// Id prefix of Sigil-built engine packs (`runtime/sigil-<engine>-<variant>`).
pub const ENGINE_PACK_PREFIX: &str = "runtime/sigil-";
/// Largest manifest accepted (it lists a handful of files).
const MAX_MANIFEST_BYTES: u64 = 256 * 1024;

/// Default DLL name of an engine.
pub fn engine_dll_name(engine: &str) -> String {
    format!("sigil_{engine}.dll")
}

/// Catalog id of an engine pack variant.
#[allow(dead_code)] // For engine wrappers and the Models tab (P1-G2, P2).
pub fn engine_pack_id(engine: &str, variant: RuntimeVariant) -> String {
    format!("{ENGINE_PACK_PREFIX}{engine}-{}", variant.as_str())
}

// ---------------------------------------------------------------------------------------------
// Manifest
// ---------------------------------------------------------------------------------------------

/// One file of a pack.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackFile {
    /// Path relative to the pack folder (`/`-separated, no `..`).
    pub path: String,
    pub size: u64,
    /// Lower-case hex sha256.
    pub sha256: String,
}

/// `sigil-pack.json`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PackManifest {
    pub schema: u32,
    pub engine: String,
    pub version: String,
    pub variant: String,
    pub abi_version: u32,
    /// The engine DLL (one of `files`).
    pub library: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub capabilities: u64,
    #[serde(default)]
    pub build_info: String,
    pub files: Vec<PackFile>,
}

fn is_sha256(s: &str) -> bool {
    s.len() == 64 && s.chars().all(|c| c.is_ascii_digit() || ('a'..='f').contains(&c))
}

/// A safe relative file path: `/`-separated plain components, no drive, root or `..`.
pub(crate) fn safe_rel_path(p: &str) -> bool {
    !p.is_empty()
        && p.len() <= 260
        && !p.starts_with('/')
        && !p.contains('\\')
        && !p.contains(':')
        && p.split('/').all(|c| {
            !c.is_empty()
                && c != "."
                && c != ".."
                && !c.ends_with('.')
                && !c.ends_with(' ')
                && c.chars().all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '.' | '_' | '-' | '+'))
        })
}

impl PackManifest {
    /// Read `dir/sigil-pack.json`: `Ok(None)` when there is none.
    pub fn read(dir: &Path) -> Result<Option<PackManifest>, String> {
        let path = dir.join(PACK_MANIFEST);
        let meta = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        };
        if meta.len() > MAX_MANIFEST_BYTES {
            return Err(format!("{} is too large ({} bytes)", path.display(), meta.len()));
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let manifest: PackManifest =
            serde_json::from_str(&text).map_err(|e| format!("{} is not a valid pack manifest: {e}", path.display()))?;
        manifest.validate().map_err(|e| format!("{}: {e}", path.display()))?;
        Ok(Some(manifest))
    }

    /// Structural checks (schema, names, safe paths, hashes, library listed).
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != PACK_MANIFEST_SCHEMA {
            return Err(format!(
                "manifest schema {} is not supported (this Sigil reads schema {PACK_MANIFEST_SCHEMA})",
                self.schema
            ));
        }
        if !valid_engine_name(&self.engine) {
            return Err(format!("invalid engine name {:?}", self.engine));
        }
        if RuntimeVariant::parse(&self.variant).is_none() {
            return Err(format!("unknown variant {:?}", self.variant));
        }
        if self.version.trim().is_empty() {
            return Err("empty version".into());
        }
        if self.files.is_empty() {
            return Err("no files listed".into());
        }
        let mut seen = std::collections::HashSet::new();
        for f in &self.files {
            if !safe_rel_path(&f.path) {
                return Err(format!("unsafe file path {:?}", f.path));
            }
            if !is_sha256(&f.sha256) {
                return Err(format!("{}: sha256 must be 64 lower-case hex digits", f.path));
            }
            if !seen.insert(f.path.to_ascii_lowercase()) {
                return Err(format!("{} is listed twice", f.path));
            }
        }
        if !self.files.iter().any(|f| f.path.eq_ignore_ascii_case(&self.library)) {
            return Err(format!("the library {:?} is not listed in files", self.library));
        }
        Ok(())
    }

    /// Check every listed file's size and sha256 (hashes cached per size and modification time).
    pub fn verify_files(&self, dir: &Path) -> Result<(), String> {
        for f in &self.files {
            let path = dir.join(f.path.replace('/', std::path::MAIN_SEPARATOR_STR));
            let meta = std::fs::metadata(&path)
                .map_err(|e| format!("{} is missing from the pack ({e}); reinstall it", f.path))?;
            if meta.len() != f.size {
                return Err(format!(
                    "{} has {} bytes but the pack manifest says {}; reinstall the pack",
                    f.path,
                    meta.len(),
                    f.size
                ));
            }
            let sha = sha256_cached(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
            if sha != f.sha256 {
                return Err(format!(
                    "{} does not match its pack manifest (sha256 {sha}, expected {}); reinstall the pack",
                    f.path, f.sha256
                ));
            }
        }
        Ok(())
    }

    /// The engine info the manifest declares (for listing packs without loading them).
    pub fn info(&self) -> EngineInfo {
        EngineInfo {
            abi_version: self.abi_version,
            name: self.engine.clone(),
            version: self.version.clone(),
            variant: self.variant.clone(),
            kind: self.kind.clone(),
            build_info: self.build_info.clone(),
            capabilities: self.capabilities,
        }
    }

    /// Whether the DLL's own `sigil_engine_get_info` agrees with the manifest.
    pub(crate) fn check_matches(&self, info: &EngineInfo) -> Result<(), String> {
        let mut diffs = Vec::new();
        if self.engine != info.name {
            diffs.push(format!("engine {:?} vs {:?}", self.engine, info.name));
        }
        if self.version != info.version {
            diffs.push(format!("version {:?} vs {:?}", self.version, info.version));
        }
        if self.variant != info.variant {
            diffs.push(format!("variant {:?} vs {:?}", self.variant, info.variant));
        }
        if self.abi_version != info.abi_version {
            diffs.push(format!("ABI v{} vs v{}", self.abi_version, info.abi_version));
        }
        if diffs.is_empty() {
            Ok(())
        } else {
            Err(format!(
                "{PACK_MANIFEST} does not match the DLL ({}); rebuild or reinstall the pack",
                diffs.join(", ")
            ))
        }
    }
}

/// Before `dir/file` is loaded: when the folder has a manifest, it must list `file` and every listed
/// file must match its size and sha256. `Ok(None)`: no manifest (development folder).
pub(crate) fn verify_library(dir: &Path, file: &str) -> Result<Option<PackManifest>, String> {
    let Some(manifest) = PackManifest::read(dir)? else {
        return Ok(None);
    };
    if !manifest.files.iter().any(|f| f.path.eq_ignore_ascii_case(file)) {
        return Err(format!("{file} is not listed in {}", dir.join(PACK_MANIFEST).display()));
    }
    if manifest.abi_version != abi::SIGIL_ENGINE_ABI_VERSION {
        return Err(super::loader::abi_mismatch_message(file, manifest.abi_version));
    }
    manifest.verify_files(dir)?;
    Ok(Some(manifest))
}

// ---------------------------------------------------------------------------------------------
// Hashing
// ---------------------------------------------------------------------------------------------

type HashKey = (PathBuf, u64, Option<SystemTime>);
/// Canonical path → (size, modification time, sha256).
type HashCache = HashMap<PathBuf, (u64, Option<SystemTime>, String)>;

fn hash_cache() -> &'static Mutex<HashCache> {
    static C: OnceLock<Mutex<HashCache>> = OnceLock::new();
    C.get_or_init(|| Mutex::new(HashMap::new()))
}

fn hash_key(path: &Path) -> std::io::Result<HashKey> {
    let meta = std::fs::metadata(path)?;
    let canon = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    Ok((canon, meta.len(), meta.modified().ok()))
}

/// sha256 of `path` (lower-case hex), cached per canonical path, size and modification time, so a
/// DLL loaded again in the same process is not re-hashed.
pub(crate) fn sha256_cached(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let (canon, len, mtime) = hash_key(path)?;
    if let Some((l, m, sha)) = hash_cache().lock().unwrap_or_else(PoisonError::into_inner).get(&canon) {
        if *l == len && *m == mtime && mtime.is_some() {
            return Ok(sha.clone());
        }
    }
    let mut file = std::fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let sha = hex::encode(hasher.finalize());
    hash_cache()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .insert(canon, (len, mtime, sha.clone()));
    Ok(sha)
}

// ---------------------------------------------------------------------------------------------
// Catalog discovery
// ---------------------------------------------------------------------------------------------

/// The catalog in effect (embedded, or the development override), loaded once per process.
pub(crate) fn catalog() -> Result<&'static Catalog, String> {
    static CATALOG: OnceLock<Result<Catalog, String>> = OnceLock::new();
    CATALOG.get_or_init(Catalog::load).as_ref().map_err(Clone::clone)
}

/// One engine pack variant in the catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EnginePackEntry {
    pub id: String,
    pub engine: String,
    pub variant: RuntimeVariant,
    pub published: bool,
    pub requires: Vec<String>,
}

/// `(engine, variant)` of an engine pack id (`runtime/sigil-<engine>-<variant>` with the entry's
/// `variant` field).
pub(crate) fn parse_engine_pack_id(id: &str, variant: Option<&str>) -> Option<(String, RuntimeVariant)> {
    let rest = id.strip_prefix(ENGINE_PACK_PREFIX)?;
    let variant = RuntimeVariant::parse(variant?)?;
    let engine = rest.strip_suffix(variant.as_str())?.strip_suffix('-')?;
    valid_engine_name(engine).then(|| (engine.to_string(), variant))
}

/// Every engine pack of the catalog (optionally only `engine`'s), in catalog order.
pub fn engine_pack_entries(catalog: &Catalog, engine: Option<&str>) -> Vec<EnginePackEntry> {
    catalog
        .entries
        .iter()
        .filter(|e| e.kind == EntryKind::Runtime)
        .filter_map(|e| {
            let (name, variant) = parse_engine_pack_id(&e.id, e.variant.as_deref())?;
            if engine.is_some_and(|want| want != name) {
                return None;
            }
            Some(EnginePackEntry {
                id: e.id.clone(),
                engine: name,
                variant,
                published: e.status == PublishStatus::Published,
                requires: e.requires.clone(),
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Opening an engine
// ---------------------------------------------------------------------------------------------

/// How to open an engine.
#[derive(Clone, Debug, Default)]
pub struct OpenRequest {
    /// Only this variant (no fallback).
    pub variant: Option<RuntimeVariant>,
    /// Load from this folder instead of the installed packs (development, engine probes).
    pub runtime_dir: Option<PathBuf>,
    /// Also create an instance with this config; a backend failure then falls through to the next
    /// variant as well.
    pub config_json: Option<String>,
}

/// A variant that was skipped or failed while opening an engine.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Attempt {
    pub variant: Option<RuntimeVariant>,
    pub pack_id: Option<String>,
    pub dir: Option<PathBuf>,
    pub error: String,
}

/// A variant to try.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Candidate {
    pub variant: Option<RuntimeVariant>,
    pub pack_id: Option<String>,
    pub dir: PathBuf,
}

/// Result of [`open_engine`].
#[derive(Debug)]
pub struct OpenedEngine {
    pub library: EngineLibrary,
    /// Created when [`OpenRequest::config_json`] was given.
    pub instance: Option<EngineInstance>,
    pub variant: Option<RuntimeVariant>,
    pub pack_id: Option<String>,
    pub dir: PathBuf,
    /// Variants skipped or failed before this one (in order).
    pub attempts: Vec<Attempt>,
}

/// Plan the variants to try for `engine`: `(candidates in order, skipped installed variants)`.
/// `installed` maps a catalog id to its install folder.
pub(crate) fn plan(
    engine: &str,
    req: &OpenRequest,
    catalog: &Catalog,
    installed: &dyn Fn(&str) -> Option<PathBuf>,
    support: &VariantSupport,
) -> Result<(Vec<Candidate>, Vec<Attempt>), String> {
    if !valid_engine_name(engine) {
        return Err(format!("invalid engine name {engine:?}"));
    }
    if let Some(dir) = &req.runtime_dir {
        if !dir.is_dir() {
            return Err(format!("{} is not a folder", dir.display()));
        }
        // A manifest that cannot be read is reported by the load itself (`verify_library`).
        let manifest_variant = PackManifest::read(dir)
            .ok()
            .flatten()
            .and_then(|m| RuntimeVariant::parse(&m.variant));
        if let (Some(forced), Some(built)) = (req.variant, manifest_variant) {
            if forced != built {
                return Err(format!(
                    "{} holds the {} build of {engine}, not {}",
                    dir.display(),
                    built.as_str(),
                    forced.as_str()
                ));
            }
        }
        let variant = req.variant.or(manifest_variant);
        if let Some(v) = variant {
            if let Some(why) = support.cpu_unsupported_reason(v) {
                return Err(format!(
                    "the {} build of {engine} in {} cannot run on this PC: {why}",
                    v.as_str(),
                    dir.display()
                ));
            }
        }
        return Ok((
            vec![Candidate {
                variant,
                pack_id: None,
                dir: dir.clone(),
            }],
            Vec::new(),
        ));
    }
    let entries = engine_pack_entries(catalog, Some(engine));
    if entries.is_empty() {
        return Err(format!("the catalog has no runtime pack for the engine '{engine}'"));
    }
    let installed_entries: Vec<(EnginePackEntry, PathBuf)> = entries
        .iter()
        .filter_map(|e| installed(&e.id).map(|d| (e.clone(), d)))
        .collect();
    if installed_entries.is_empty() {
        let ids: Vec<&str> = entries.iter().map(|e| e.id.as_str()).collect();
        return Err(format!(
            "the {engine} runtime is not installed (install one of {} in Models & Runtimes)",
            ids.join(", ")
        ));
    }
    let available: Vec<RuntimeVariant> = installed_entries.iter().map(|(e, _)| e.variant).collect();
    if let Some(forced) = req.variant {
        if !available.contains(&forced) {
            return Err(format!(
                "the {} variant of {engine} is not installed (installed: {})",
                forced.as_str(),
                available.iter().map(|v| v.as_str()).collect::<Vec<_>>().join(", ")
            ));
        }
    }
    let order = candidates(&available, support, req.variant);
    let skipped: Vec<Attempt> = installed_entries
        .iter()
        .filter(|(e, _)| !order.contains(&e.variant))
        .map(|(e, d)| Attempt {
            variant: Some(e.variant),
            pack_id: Some(e.id.clone()),
            dir: Some(d.clone()),
            error: match req.variant {
                // A forced variant is only dropped for its instruction sets.
                Some(forced) if forced == e.variant => support
                    .cpu_unsupported_reason(e.variant)
                    .unwrap_or_else(|| "not selected".into()),
                Some(forced) => format!("only {} was requested", forced.as_str()),
                None => support
                    .unsupported_reason(e.variant)
                    .unwrap_or_else(|| "not selected".into()),
            },
        })
        .collect();
    if let (Some(forced), true) = (req.variant, order.is_empty()) {
        let why = support
            .cpu_unsupported_reason(forced)
            .unwrap_or_else(|| "not selected".into());
        return Err(format!(
            "the {} variant of {engine} cannot run on this PC: {why}",
            forced.as_str()
        ));
    }
    if order.is_empty() {
        let reasons: Vec<String> = skipped
            .iter()
            .map(|a| format!("{}: {}", a.variant.map_or("?", |v| v.as_str()), a.error))
            .collect();
        return Err(format!(
            "none of the installed {engine} variants can run on this PC ({})",
            reasons.join("; ")
        ));
    }
    let chosen = order
        .into_iter()
        .filter_map(|v| {
            installed_entries.iter().find(|(e, _)| e.variant == v).map(|(e, d)| Candidate {
                variant: Some(v),
                pack_id: Some(e.id.clone()),
                dir: d.clone(),
            })
        })
        .collect();
    Ok((chosen, skipped))
}

/// Checks of one candidate that need no code from its folder: this CPU runs the variant it is
/// installed as, an installed pack carries its manifest, and the manifest describes `engine` in
/// that same variant (a mislabeled pack is refused). Returns the DLL file name to load.
pub(crate) fn preflight(engine: &str, c: &Candidate, support: &VariantSupport) -> Result<String, String> {
    let reinstall = |id: &Option<String>| match id {
        Some(id) => format!("reinstall {id}"),
        None => "rebuild the folder".to_string(),
    };
    if let Some(v) = c.variant {
        if let Some(why) = support.cpu_unsupported_reason(v) {
            return Err(format!("the {} build cannot run on this PC: {why}", v.as_str()));
        }
    }
    let Some(manifest) = PackManifest::read(&c.dir)? else {
        return match &c.pack_id {
            // Only an explicit development folder may skip the hash check.
            Some(id) => Err(format!(
                "{} has no {PACK_MANIFEST}, so its files cannot be verified; reinstall {id}",
                c.dir.display()
            )),
            None => Ok(engine_dll_name(engine)),
        };
    };
    if manifest.engine != engine {
        return Err(format!(
            "{} holds the '{}' engine, not '{engine}'; {}",
            c.dir.display(),
            manifest.engine,
            reinstall(&c.pack_id)
        ));
    }
    if let Some(v) = c.variant {
        if manifest.variant != v.as_str() {
            return Err(format!(
                "{} holds the {} build but is installed as {}; {}",
                c.dir.display(),
                manifest.variant,
                v.as_str(),
                reinstall(&c.pack_id)
            ));
        }
    }
    support
        .check_build(&manifest.variant)
        .map_err(|e| format!("{}: {e}", c.dir.join(&manifest.library).display()))?;
    Ok(manifest.library)
}

/// Load (and optionally create) one candidate. `Err((message, fall_through))`.
fn try_candidate(
    engine: &str,
    c: &Candidate,
    req: &OpenRequest,
    support: &VariantSupport,
) -> Result<(EngineLibrary, Option<EngineInstance>), (String, bool)> {
    let file = preflight(engine, c, support).map_err(|e| (e, true))?;
    if c.variant == Some(RuntimeVariant::Cuda13) {
        super::nvidia::ensure_runtime_for_engines().map_err(|e| (e, true))?;
    }
    // The pack folder is a search location for DLLs the engine loads by name at run time.
    if let Err(e) = super::dll_dirs::add_dll_directory(&c.dir) {
        tracing::debug!(target: "sigil::runtime", "{e}");
    }
    let library = EngineLibrary::load_checked(&c.dir, &file, support, c.pack_id.is_some()).map_err(|e| (e, true))?;
    if library.info().name != engine {
        return Err((
            format!(
                "{} is the '{}' engine, not '{engine}'",
                library.path().display(),
                library.info().name
            ),
            true,
        ));
    }
    // With a manifest, `load_checked` already matched the DLL's variant to it (and `preflight` the
    // manifest to `c.variant`); this catches a manifest-less development folder.
    if let Some(v) = c.variant {
        if library.info().variant != v.as_str() {
            return Err((
                format!(
                    "{} reports the {} build, not {}",
                    library.path().display(),
                    library.info().variant,
                    v.as_str()
                ),
                true,
            ));
        }
    }
    let instance = match &req.config_json {
        None => None,
        Some(cfg) => match library.create_engine(cfg) {
            Ok(i) => Some(i),
            Err(e) => {
                let gpu = c.variant.is_some_and(RuntimeVariant::is_gpu) || library.info().has(abi::SIGIL_CAP_GPU);
                let fall = e.is_backend_failure(gpu);
                return Err((e.message, fall));
            }
        },
    };
    Ok((library, instance))
}

/// Open `engine` from its installed packs (or [`OpenRequest::runtime_dir`]) with variant fallback.
pub fn open_engine(engine: &str, req: &OpenRequest) -> Result<OpenedEngine, String> {
    // An explicit folder needs no variant choice (and no GPU probe).
    let support = if req.runtime_dir.is_some() {
        VariantSupport::detect_cpu()
    } else {
        VariantSupport::detect()
    };
    let installed = |id: &str| crate::services::model_store::installed_path(id);
    let (plan, mut attempts) = match &req.runtime_dir {
        Some(_) => plan(engine, req, &Catalog::default(), &installed, &support)?,
        None => plan(engine, req, catalog()?, &installed, &support)?,
    };
    for c in plan {
        match try_candidate(engine, &c, req, &support) {
            Ok((library, instance)) => {
                if !attempts.is_empty() {
                    tracing::info!(
                        target: "sigil::runtime",
                        "{engine}: using {} after {}",
                        c.variant.map_or("the given folder", |v| v.as_str()),
                        attempts
                            .iter()
                            .map(|a| format!("{}: {}", a.variant.map_or("?", |v| v.as_str()), a.error))
                            .collect::<Vec<_>>()
                            .join("; ")
                    );
                }
                return Ok(OpenedEngine {
                    library,
                    instance,
                    variant: c.variant,
                    pack_id: c.pack_id,
                    dir: c.dir,
                    attempts,
                });
            }
            Err((error, fall_through)) => {
                tracing::warn!(
                    target: "sigil::runtime",
                    "{engine} ({}) failed: {error}",
                    c.variant.map_or("given folder", |v| v.as_str())
                );
                attempts.push(Attempt {
                    variant: c.variant,
                    pack_id: c.pack_id.clone(),
                    dir: Some(c.dir.clone()),
                    error: error.clone(),
                });
                if !fall_through {
                    return Err(error);
                }
            }
        }
    }
    Err(format!(
        "the {engine} runtime could not be loaded: {}",
        attempts
            .iter()
            .map(|a| format!("{}: {}", a.variant.map_or("given folder", |v| v.as_str()), a.error))
            .collect::<Vec<_>>()
            .join("; ")
    ))
}
