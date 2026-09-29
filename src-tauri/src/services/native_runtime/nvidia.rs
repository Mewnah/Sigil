//! The NVIDIA GPU runtime (CUDA 13 + cuDNN 9) and the ONNX Runtime CUDA provider as model-store
//! runtime packs.
//!
//! - [`NVIDIA_PACK_ID`] (`model-catalog/runtime-nvidia.json`): cudart, cuBLAS / cuBLASLt, cuFFT and
//!   cuDNN 9 (`cuda13` build) from NVIDIA's pinned redistributable archives, installed by the model
//!   store (one download at a time, Range resume, sha256, safe extraction of the DLLs only) into
//!   `app_data/runtimes/nvidia-cuda13/<version>/cuda13/`. It replaces the interim P0 downloader in
//!   `diagnostics::nvidia_pack`, whose installs keep working: when the store pack is not installed,
//!   the interim pack is preloaded instead, so nobody downloads 1.9 GB twice.
//! - [`CUDA_PROVIDER_PACK_ID`] (`model-catalog/runtimes.json`): `onnxruntime_providers_cuda.dll`
//!   (ONNX Runtime 1.28.0, 62 MB), registered with `register_ep_library` by `ort_session` when it is
//!   not installed next to `Sigil.exe` (ADR 0007). Status `unpublished` until the maintainer
//!   publishes it; `scripts/native/build-runtime-packs.ps1 -Engine ort-cuda-provider` stages it for a
//!   local install.
//!
//! [`ensure_preloaded`] loads the runtime DLLs by full path in dependency order before the first
//! CUDA session (and before a `cuda13` engine DLL), registering their folders with
//! `AddDllDirectory` so the provider's and cuDNN's load-by-name calls bind to them. Once loaded, a
//! pack stays loaded for the life of the process (DLL names are process-wide); a pack removed while
//! Sigil runs is reported as removed, so CUDA falls back to DirectML until the next start.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};
use std::time::Instant;

use serde::Serialize;

use super::dll_dirs::{find_dlls, preload_ordered};
use super::pack::{catalog, sha256_cached};
use crate::services::diagnostics::{self, nvidia_pack};

/// Catalog id of the NVIDIA CUDA 13 / cuDNN 9 runtime pack.
pub const NVIDIA_PACK_ID: &str = "runtime/nvidia-cuda13";
/// Catalog id of the ONNX Runtime CUDA provider pack.
pub const CUDA_PROVIDER_PACK_ID: &str = "runtime/ort-cuda13-provider";
/// The provider library inside [`CUDA_PROVIDER_PACK_ID`].
pub const CUDA_PROVIDER_FILE: &str = "onnxruntime_providers_cuda.dll";
/// Folder depth searched for DLLs inside the pack (NVIDIA archives nest them in `bin/x64`).
const SEARCH_DEPTH: usize = 4;

/// Load order: dependencies first. `false` = optional (loaded when present). cudart is not needed
/// by ONNX Runtime 1.28 (its CUDA runtime is static) but by ggml-based `cuda13` engine packs.
pub const PRELOAD_ORDER: [(&str, bool); 12] = [
    ("cudart64_13.dll", true),
    ("cublasLt64_13.dll", true),
    ("cublas64_13.dll", true),
    ("cufft64_12.dll", true),
    ("cudnn64_9.dll", true),
    ("cudnn_graph64_9.dll", true),
    ("cudnn_ops64_9.dll", true),
    ("cudnn_heuristic64_9.dll", true),
    ("cudnn_cnn64_9.dll", true),
    ("cudnn_engines_precompiled64_9.dll", true),
    // Needed only for RNN/attention graphs and runtime-fused engines (NVRTC); RVC uses neither.
    ("cudnn_adv64_9.dll", false),
    ("cudnn_engines_runtime_compiled64_9.dll", false),
];

/// The load list for an installed pack: [`PRELOAD_ORDER`], then every other cuDNN 9 sub-library the
/// pack ships, as optional entries (name order). cuDNN loads its sub-libraries by module name at run
/// time, which only finds modules that are already loaded (`AddDllDirectory` folders are not
/// searched by a plain `LoadLibrary`), and newer releases add sub-libraries: cuDNN 9.26 ships
/// `cudnn_engines_tensor_ir64_9.dll` and `cudnn_ext64_9.dll`, and without them the cuDNN frontend
/// fails to initialize.
pub(crate) fn preload_list(index: &BTreeMap<String, PathBuf>) -> Vec<(String, bool)> {
    let mut list: Vec<(String, bool)> = PRELOAD_ORDER.iter().map(|(n, r)| ((*n).to_string(), *r)).collect();
    for name in index.keys() {
        let known = PRELOAD_ORDER.iter().any(|(n, _)| n.eq_ignore_ascii_case(name));
        if !known && name.starts_with("cudnn") && name.ends_with("64_9.dll") {
            list.push((name.clone(), false));
        }
    }
    list
}

/// Which install of the NVIDIA runtime this process uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RuntimeSource {
    /// [`NVIDIA_PACK_ID`] installed by the model store.
    ModelStore,
    /// The interim P0 pack (`diagnostics::nvidia_pack`, `runtimes/nvidia-cu13/<label>/bin`).
    Interim,
}

/// Result of [`ensure_preloaded`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Preload {
    /// No NVIDIA runtime pack is installed (CUDA may still work from a system install on `PATH`).
    NotInstalled,
    /// The runtime is loaded in this process.
    Loaded { source: RuntimeSource, root: PathBuf },
    /// Loading failed (reason); kept until Sigil restarts, like any partially loaded DLL set.
    Failed(String),
    /// The pack this process loaded was removed while Sigil runs.
    Removed(String),
}

struct Preloaded {
    source: RuntimeSource,
    root: PathBuf,
    result: Result<Vec<String>, String>,
}

static PRELOADED: Mutex<Option<Preloaded>> = Mutex::new(None);

fn store_root() -> Option<PathBuf> {
    crate::services::model_store::installed_path(NVIDIA_PACK_ID)
}

/// Whether the interim P0 pack is installed and complete (not marked for removal).
fn interim_installed() -> bool {
    nvidia_pack::status(diagnostics::app_data_dir().as_deref()).installed
}

const REMOVED: &str = "the NVIDIA GPU runtime pack was removed while Sigil was running; restart Sigil to finish removing it";

/// Preload the NVIDIA runtime (once per process; later calls report the first result). Prefers the
/// model-store pack, else the interim P0 pack. Call before registering the CUDA EP or loading a
/// `cuda13` engine DLL.
pub fn ensure_preloaded() -> Preload {
    let store = store_root();
    let mut state = PRELOADED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(p) = state.as_ref() {
        let still_installed = match p.source {
            RuntimeSource::ModelStore => store.as_deref() == Some(p.root.as_path()),
            RuntimeSource::Interim => interim_installed(),
        };
        return match (&p.result, still_installed) {
            (_, false) => Preload::Removed(REMOVED.into()),
            (Ok(_), true) => Preload::Loaded {
                source: p.source,
                root: p.root.clone(),
            },
            (Err(e), true) => Preload::Failed(e.clone()),
        };
    }
    if let Some(root) = store {
        let started = Instant::now();
        let index = find_dlls(&root, SEARCH_DEPTH);
        let list = preload_list(&index);
        let order: Vec<(&str, bool)> = list.iter().map(|(n, r)| (n.as_str(), *r)).collect();
        let result = preload_ordered(&order, &index);
        match &result {
            Ok(loaded) => tracing::info!(
                target: "sigil::runtime",
                "NVIDIA runtime pack preloaded from {} ({} DLLs) in {} ms",
                root.display(),
                loaded.len(),
                started.elapsed().as_millis()
            ),
            Err(e) => tracing::warn!(target: "sigil::runtime", "NVIDIA runtime pack in {} failed to load: {e}", root.display()),
        }
        let out = match &result {
            Ok(_) => Preload::Loaded {
                source: RuntimeSource::ModelStore,
                root: root.clone(),
            },
            Err(e) => Preload::Failed(e.clone()),
        };
        *state = Some(Preloaded {
            source: RuntimeSource::ModelStore,
            root,
            result,
        });
        return out;
    }
    // The interim pack keeps its own preload state; mirror its first result here.
    match nvidia_pack::ensure_preloaded() {
        None => Preload::NotInstalled,
        Some(result) => {
            let root = diagnostics::app_data_dir()
                .map(|d| nvidia_pack::bin_dir(&d))
                .unwrap_or_default();
            let out = match &result {
                Ok(()) => Preload::Loaded {
                    source: RuntimeSource::Interim,
                    root: root.clone(),
                },
                Err(e) => Preload::Failed(e.clone()),
            };
            *state = Some(Preloaded {
                source: RuntimeSource::Interim,
                root,
                result: result.map(|()| Vec::new()),
            });
            out
        }
    }
}

/// DLLs of the installed [`NVIDIA_PACK_ID`] by lower-case file name, without loading anything (the
/// EP probe locates the CUDA runtime there). `None` when the store pack is not installed.
pub fn store_dll_index() -> Option<BTreeMap<String, PathBuf>> {
    store_root().map(|root| find_dlls(&root, SEARCH_DEPTH))
}

/// What this process preloaded, without loading anything: `(source, root, error)`.
pub fn preload_state() -> Option<(RuntimeSource, PathBuf, Option<String>)> {
    PRELOADED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .as_ref()
        .map(|p| (p.source, p.root.clone(), p.result.as_ref().err().cloned()))
}

/// For `cuda13` engine packs: the NVIDIA runtime must be installed and loaded.
pub fn ensure_runtime_for_engines() -> Result<(), String> {
    match ensure_preloaded() {
        Preload::Loaded { .. } => Ok(()),
        Preload::NotInstalled => Err(format!(
            "the NVIDIA GPU runtime pack ({NVIDIA_PACK_ID}) is not installed; install it in Models & Runtimes"
        )),
        Preload::Failed(e) | Preload::Removed(e) => Err(e),
    }
}

/// Everything that decides which CUDA runtime and provider this process can use (compared by
/// `ort_session` to invalidate cached EP canary results when a pack is installed or removed).
pub fn fingerprint() -> String {
    format!(
        "{:?}|{:?}|{}",
        store_root(),
        crate::services::model_store::installed_path(CUDA_PROVIDER_PACK_ID),
        interim_installed()
    )
}

/// The provider library of the installed [`CUDA_PROVIDER_PACK_ID`], checked against the catalog's
/// pinned sha256 (cached per size and modification time). `None` when the pack is not installed.
pub fn provider_pack_library() -> Option<Result<PathBuf, String>> {
    let root = crate::services::model_store::installed_path(CUDA_PROVIDER_PACK_ID)?;
    Some(verify_provider(&root, pinned_provider_sha256().as_deref()))
}

/// The sha256 the catalog pins for the provider file.
fn pinned_provider_sha256() -> Option<String> {
    catalog()
        .ok()?
        .get(CUDA_PROVIDER_PACK_ID)?
        .files
        .iter()
        .find(|f| f.path.eq_ignore_ascii_case(CUDA_PROVIDER_FILE))?
        .sha256
        .clone()
}

/// `root/onnxruntime_providers_cuda.dll` when present and matching `pin` (when given).
pub(crate) fn verify_provider(root: &Path, pin: Option<&str>) -> Result<PathBuf, String> {
    let path = root.join(CUDA_PROVIDER_FILE);
    if !path.is_file() {
        return Err(format!(
            "{CUDA_PROVIDER_FILE} is missing from the CUDA provider pack in {}; reinstall it",
            root.display()
        ));
    }
    if let Some(pin) = pin {
        let sha = sha256_cached(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        if !sha.eq_ignore_ascii_case(pin) {
            return Err(format!(
                "{} does not match the pinned CUDA provider (sha256 {sha}); reinstall the CUDA provider pack",
                path.display()
            ));
        }
    }
    Ok(path)
}

// ---------------------------------------------------------------------------------------------
// Leftovers from earlier Sigil versions
// ---------------------------------------------------------------------------------------------

/// A runtime folder from an earlier Sigil that this version no longer uses (or duplicates).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LegacyRuntime {
    /// Pseudo id (`legacy/...`), accepted by [`remove_legacy`].
    pub id: &'static str,
    pub title: &'static str,
    pub dir: PathBuf,
    pub bytes: u64,
    pub note: String,
}

/// `legacy/nvidia-cu12`: the P0-era CUDA 12 pack (ONNX Runtime 1.23), never loaded by this Sigil.
pub const LEGACY_CU12_ID: &str = "legacy/nvidia-cu12";
/// `legacy/nvidia-cu13-interim`: the interim CUDA 13 pack when the model-store pack is installed too.
pub const LEGACY_INTERIM_ID: &str = "legacy/nvidia-cu13-interim";

fn folder_bytes(dir: &Path, depth: usize) -> u64 {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return 0;
    };
    entries
        .flatten()
        .map(|e| match e.file_type() {
            Ok(t) if t.is_file() => e.metadata().map(|m| m.len()).unwrap_or(0),
            Ok(t) if t.is_dir() && depth > 0 => folder_bytes(&e.path(), depth - 1),
            _ => 0,
        })
        .sum()
}

fn gb(bytes: u64) -> String {
    format!("{:.1} GB", bytes as f64 / 1e9)
}

/// Leftover runtime folders under `app_data` (see [`LegacyRuntime`]).
pub fn legacy_runtimes(app_data: &Path, store_pack_installed: bool) -> Vec<LegacyRuntime> {
    let mut out = Vec::new();
    let cu12 = app_data.join("runtimes").join("nvidia-cu12");
    if cu12.is_dir() {
        let bytes = folder_bytes(&cu12, 6);
        out.push(LegacyRuntime {
            id: LEGACY_CU12_ID,
            title: "NVIDIA runtime for CUDA 12 (earlier Sigil)",
            dir: cu12,
            bytes,
            note: format!(
                "no longer used: this Sigil runs CUDA 13; remove it to free {}",
                gb(bytes)
            ),
        });
    }
    let interim = nvidia_pack::install_dir(app_data);
    // `installed` is false for a pack already marked for removal (deferred because this process
    // loaded it): it goes at the next start, and offering Remove again would do nothing.
    if store_pack_installed && nvidia_pack::status(Some(app_data)).installed {
        let bytes = folder_bytes(&interim, 6);
        out.push(LegacyRuntime {
            id: LEGACY_INTERIM_ID,
            title: "NVIDIA runtime (previous installer)",
            dir: interim,
            bytes,
            note: format!(
                "a duplicate of {NVIDIA_PACK_ID}; remove it to free {}",
                gb(bytes)
            ),
        });
    }
    out
}

/// Remove a [`legacy_runtimes`] folder. The interim pack's removal is deferred to the next start
/// when this process loaded it. `store_pack_installed` is whether [`NVIDIA_PACK_ID`] is installed
/// right now: the interim pack is a leftover only while that pack exists, and is refused otherwise
/// (it is then the only NVIDIA runtime, whatever a stale list showed). Removing something already
/// gone, or already marked for removal, is a no-op.
pub fn remove_legacy(app_data: &Path, id: &str, store_pack_installed: bool) -> Result<(), String> {
    match id {
        LEGACY_CU12_ID => {
            let dir = app_data.join("runtimes").join("nvidia-cu12");
            if !dir.exists() {
                return Ok(());
            }
            // Rename first so a half-deleted folder never looks like a pack.
            let trash = app_data
                .join("runtimes")
                .join(format!(".trash-nvidia-cu12-{}", crate::services::diagnostics::logging::unix_ms_now()));
            std::fs::rename(&dir, &trash).map_err(|e| format!("cannot remove {}: {e}", dir.display()))?;
            std::fs::remove_dir_all(&trash).map_err(|e| format!("cannot remove {}: {e}", trash.display()))?;
            tracing::info!(target: "sigil::runtime", "removed the obsolete CUDA 12 runtime ({})", dir.display());
            Ok(())
        }
        LEGACY_INTERIM_ID => {
            if !nvidia_pack::status(Some(app_data)).installed {
                return Ok(()); // Already removed, or removal pending until the next start.
            }
            if !store_pack_installed {
                return Err(format!(
                    "the NVIDIA runtime from the previous installer is not a leftover: {NVIDIA_PACK_ID} is not installed, so it is the only NVIDIA GPU runtime"
                ));
            }
            nvidia_pack::remove(app_data).map(|_| ())
        }
        other => Err(format!("{other} is not a leftover runtime")),
    }
}

/// Sweep interrupted legacy removals (`runtimes/.trash-*`). Best effort.
pub fn sweep_legacy_trash(app_data: &Path) {
    let Ok(entries) = std::fs::read_dir(app_data.join("runtimes")) else {
        return;
    };
    for e in entries.flatten() {
        if e.file_name().to_string_lossy().starts_with(".trash-") && e.path().is_dir() {
            let _ = std::fs::remove_dir_all(e.path());
        }
    }
}
