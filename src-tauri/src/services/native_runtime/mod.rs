//! `runtime` plugin: lazily loaded native runtime DLLs (sigil_engine ABI engines such as echo and
//! transcribe.cpp, plus plain DLLs such as sherpa-onnx), variant selection, the NVIDIA GPU runtime
//! and CUDA provider packs, and the runtime-pack status. Nothing here is load-time linked: the app
//! starts, and every existing feature works, with no pack installed.
//!
//! Owned by P1-D1; scaffolded by P1-S0b (plugin registered in main.rs, capability
//! `capabilities/runtime.json`, commands in `plugin-commands/runtime.txt`, ABI header
//! `native/common/sigil_engine.h`). Decision record: `docs/adr/0007-native-runtime-packs.md`.
//!
//! # Modules
//! - [`abi`]: the Rust mirror of `native/common/sigil_engine.h` (ABI v1).
//! - `variant`: [`RuntimeVariant`], [`VariantSupport`] (CPU features, CUDA 13 eligibility from the
//!   system probe, Vulkan), [`best_variant`], [`candidates`].
//! - `dll_dirs`: [`load_library_in_dir`] (`LoadLibraryExW` with `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR |
//!   DEFAULT_DIRS`), [`add_dll_directory`], ordered preloads.
//! - `loader`: [`EngineLibrary`] → [`EngineInstance`] → [`EngineStream`], [`EngineCanceller`].
//! - `pack`: `sigil-pack.json` manifests and hash checks, catalog discovery
//!   (`runtime/sigil-<engine>-<variant>`), [`open_engine`] with variant fallback.
//! - `nvidia`: the NVIDIA CUDA 13 runtime pack and the ONNX Runtime CUDA provider pack.
//! - `probe`: `Sigil.exe --headless engine-probe <engine> ...`.
//!
//! # Frozen contract (P1-S0b; change only through an INTEGRATE step)
//! - [`abi`]: the Rust mirror of `native/common/sigil_engine.h` ABI v1 (constants, `#[repr(C)]`
//!   structs, function-pointer types, [`abi::SYMBOLS`]). Header and mirror change together.
//! - [`RuntimeVariant`] (`cuda13`, `vulkan`, `cpu-avx512`, `cpu-avx2`, `cpu`; serde kebab-case) with
//!   [`RuntimeVariant::PREFERENCE`], [`VariantSupport`] and [`best_variant`].
//! - [`load_library_in_dir`]`(dir, file_name) -> Result<libloading::Library, String>`: full-path
//!   `LoadLibraryExW(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`, so the
//!   DLL's own folder resolves its dependencies (used by speech_rt::sherpa for onnxruntime.dll then
//!   sherpa-onnx-c-api.dll). A missing file or DLL is an `Err`, never a panic.
//! - [`EngineLibrary`] (one loaded sigil_engine DLL: hash / ABI check, all v1 symbols resolved) →
//!   [`EngineInstance`] (`sigil_engine_create`) → `process` / `render` / `query` /
//!   [`EngineInstance::open_stream`] → [`EngineStream`], and [`EngineCanceller`]. `EngineInstance`
//!   and `EngineStream` are `Send` (not `Sync`); a stream keeps its instance and library alive.
//! - [`headless_probe`]: body of `Sigil.exe --headless engine-probe echo ...` (see speech_rt).
//! - IPC `plugin:runtime|list_runtime_packs -> RuntimePackStatus[]`.
//! - Pack location: packs are catalog entries of kind `runtime`; their folder comes from
//!   `model_store::installed_path(id)` (or an explicit directory in dev / probe runs), never from a
//!   recomputed layout.
//!
//! # Additions (P1-D1; additive)
//! - [`open_engine`]`(engine, &OpenRequest) -> OpenedEngine`: the high-level entry for engine
//!   wrappers (transcribe, P2): installed variants in preference order with fallback, `cuda13`
//!   variants preload the NVIDIA runtime first, optional create; every failed variant is reported.
//! - `EngineLibrary::create_engine`, `EngineInstance::{query_engine, process_audio, render_audio,
//!   open_engine_stream}`, `EngineStream::{push_audio, finish_stream}`: the same calls returning
//!   [`EngineError`] (status code + message).
//! - [`nvidia::ensure_preloaded`] / [`nvidia::provider_pack_library`]: used by
//!   `neural_audio::ort_session` for the CUDA execution provider.
//! - IPC `plugin:runtime|remove_legacy_runtime({ id })` for the `legacy/*` rows of
//!   `list_runtime_packs` (runtime folders of earlier Sigil versions that this one never loads).
//! - [`RuntimePackStatus`] gains `title` and `legacy`.

pub mod abi;
mod dll_dirs;
mod loader;
pub mod nvidia;
mod pack;
mod probe;
mod variant;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::model_store::{Catalog, CatalogEntry, EntryKind, PublishStatus};

// Public API for engine wrappers (sherpa P1-G1, transcribe P1-G2, the P2 speech plugin).
#[allow(unused_imports)]
pub use dll_dirs::{add_dll_directory, load_library_in_dir};
#[allow(unused_imports)]
pub use loader::{AudioBuffer, AudioView, EngineCanceller, EngineError, EngineInfo, EngineInstance, EngineLibrary, EngineStream};
#[allow(unused_imports)]
pub use pack::{
    engine_dll_name, engine_pack_entries, engine_pack_id, open_engine, Attempt, EnginePackEntry, OpenRequest, OpenedEngine,
    PackFile, PackManifest, ENGINE_PACK_PREFIX, PACK_MANIFEST,
};
pub use probe::headless_probe;
#[allow(unused_imports)]
pub use variant::{best_variant, candidates, RuntimeVariant, VariantSupport};

// ---------------------------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------------------------

/// Row of `list_runtime_packs`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimePackStatus {
    /// Catalog id of the pack (`legacy/...` for a leftover folder of an earlier Sigil).
    pub id: String,
    /// Installed variant, if any.
    pub variant: Option<RuntimeVariant>,
    /// Install directory, if installed.
    pub dir: Option<String>,
    /// Whether the pack loaded (or would load) successfully on this PC.
    pub loadable: bool,
    /// Engine info of an installed sigil_engine pack (from its hash-checked `sigil-pack.json`,
    /// which the DLL's own info must match when it loads; listing never loads a DLL).
    pub info: Option<EngineInfo>,
    /// Why the pack is not usable.
    pub error: Option<String>,
    /// Catalog title (or the leftover's description).
    #[serde(default)]
    pub title: Option<String>,
    /// A leftover folder of an earlier Sigil (`remove_legacy_runtime` deletes it).
    #[serde(default)]
    pub legacy: bool,
}

/// Inputs of [`pack_rows`], injectable for tests.
pub(crate) struct RowInputs<'a> {
    pub installed: &'a dyn Fn(&str) -> Option<PathBuf>,
    pub support: VariantSupport,
    /// Pinned sha256 check and ORT version check of the CUDA provider pack.
    pub check_provider: &'a dyn Fn(&Path) -> Result<(), String>,
    /// Preload failure of the NVIDIA runtime in this process (reason), if any.
    pub nvidia_error: Option<String>,
}

/// Why an installed engine pack cannot be used (`Ok(info)` from its manifest when it can).
fn engine_pack_state(dir: &Path) -> Result<EngineInfo, String> {
    let manifest = PackManifest::read(dir)?
        .ok_or_else(|| format!("the pack has no {PACK_MANIFEST}; reinstall it"))?;
    if manifest.abi_version != abi::SIGIL_ENGINE_ABI_VERSION {
        return Err(loader::abi_mismatch_message(&manifest.library, manifest.abi_version));
    }
    manifest.verify_files(dir)?;
    Ok(manifest.info())
}

/// One row per runtime entry of the catalog (catalog order).
pub(crate) fn pack_rows(catalog: &Catalog, inputs: &RowInputs<'_>) -> Vec<RuntimePackStatus> {
    catalog
        .entries
        .iter()
        .filter(|e| e.kind == EntryKind::Runtime)
        .map(|e| pack_row(catalog, e, inputs))
        .collect()
}

fn pack_row(catalog: &Catalog, e: &CatalogEntry, inputs: &RowInputs<'_>) -> RuntimePackStatus {
    let variant = e.variant.as_deref().and_then(RuntimeVariant::parse);
    let dir = (inputs.installed)(&e.id);
    let mut row = RuntimePackStatus {
        id: e.id.clone(),
        variant,
        dir: dir.as_ref().map(|d| d.display().to_string()),
        title: e.title.clone(),
        ..RuntimePackStatus::default()
    };
    let Some(dir) = dir else {
        row.error = Some(match e.status {
            PublishStatus::Published => "not installed".into(),
            PublishStatus::Unpublished => {
                "not installed (a Sigil-built pack: install it from a local file or SIGIL_LOCAL_ASSETS)".into()
            }
        });
        return row;
    };
    let mut problems: Vec<String> = Vec::new();
    let engine_pack = pack::parse_engine_pack_id(&e.id, e.variant.as_deref()).is_some();
    if let Some(v) = variant {
        // Only Sigil-built engines carry the cpu-avx2 baseline in their GPU variants; the NVIDIA
        // runtime, the CUDA provider and third-party runtimes need just their GPU.
        let why = if engine_pack {
            inputs.support.unsupported_reason(v)
        } else {
            inputs.support.runtime_unsupported_reason(v)
        };
        problems.extend(why);
    }
    for req in &e.requires {
        if (inputs.installed)(req).is_none() {
            let title = catalog.get(req).and_then(|r| r.title.clone()).unwrap_or_else(|| req.clone());
            problems.push(format!("needs {title} ({req})"));
        }
    }
    if engine_pack {
        match engine_pack_state(&dir) {
            Ok(info) => row.info = Some(info),
            Err(why) => problems.push(why),
        }
    } else if e.id == nvidia::NVIDIA_PACK_ID {
        if let Some(why) = &inputs.nvidia_error {
            problems.push(why.clone());
        }
    } else if e.id == nvidia::CUDA_PROVIDER_PACK_ID {
        if let Err(why) = (inputs.check_provider)(&dir) {
            problems.push(why);
        }
    }
    row.loadable = problems.is_empty();
    row.error = (!problems.is_empty()).then(|| problems.join("; "));
    row
}

fn legacy_rows(app_data: &Path, store_pack_installed: bool) -> Vec<RuntimePackStatus> {
    nvidia::legacy_runtimes(app_data, store_pack_installed)
        .into_iter()
        .map(|l| RuntimePackStatus {
            id: l.id.to_string(),
            variant: None,
            dir: Some(l.dir.display().to_string()),
            loadable: false,
            info: None,
            error: Some(l.note),
            title: Some(l.title.to_string()),
            legacy: true,
        })
        .collect()
}

/// The live rows: catalog runtime entries plus leftovers.
pub fn runtime_pack_statuses() -> Result<Vec<RuntimePackStatus>, String> {
    let catalog = pack::catalog()?;
    let installed = |id: &str| crate::services::model_store::installed_path(id);
    let check_provider = |dir: &Path| -> Result<(), String> {
        let pin = catalog
            .get(nvidia::CUDA_PROVIDER_PACK_ID)
            .and_then(|e| e.files.iter().find(|f| f.path.eq_ignore_ascii_case(nvidia::CUDA_PROVIDER_FILE)))
            .and_then(|f| f.sha256.clone());
        let path = nvidia::verify_provider(dir, pin.as_deref())?;
        match crate::services::neural_audio::ort_session::provider_version_mismatch(&path) {
            Some(why) => Err(why),
            None => Ok(()),
        }
    };
    let nvidia_error = nvidia::preload_state().and_then(|(_, _, e)| e);
    let inputs = RowInputs {
        installed: &installed,
        support: VariantSupport::detect(),
        check_provider: &check_provider,
        nvidia_error,
    };
    let mut rows = pack_rows(catalog, &inputs);
    if let Some(app_data) = crate::services::diagnostics::app_data_dir() {
        let store_pack = rows.iter().any(|r| r.id == nvidia::NVIDIA_PACK_ID && r.dir.is_some());
        rows.extend(legacy_rows(&app_data, store_pack));
    }
    Ok(rows)
}

#[command]
async fn list_runtime_packs() -> Result<Vec<RuntimePackStatus>, String> {
    tauri::async_runtime::spawn_blocking(runtime_pack_statuses)
        .await
        .map_err(|e| format!("runtime pack status task: {e}"))?
}

#[command]
async fn remove_legacy_runtime(id: String) -> Result<Vec<RuntimePackStatus>, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let app_data = crate::services::diagnostics::app_data_dir().ok_or("no app data folder")?;
        // Decided now, not from the (possibly stale) list the UI showed.
        let store_pack = crate::services::model_store::installed_path(nvidia::NVIDIA_PACK_ID).is_some();
        nvidia::remove_legacy(&app_data, id.trim(), store_pack)?;
        runtime_pack_statuses()
    })
    .await
    .map_err(|e| format!("remove task: {e}"))?
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("runtime")
        .invoke_handler(tauri::generate_handler![list_runtime_packs, remove_legacy_runtime])
        .setup(|_app, _api| {
            // Interrupted leftover removals; never blocks startup.
            std::thread::Builder::new()
                .name("sigil-runtime-sweep".into())
                .spawn(|| {
                    if let Some(app_data) = crate::services::diagnostics::app_data_dir() {
                        nvidia::sweep_legacy_trash(&app_data);
                    }
                })
                .map(|_| ())
                .map_err(|e| e.into())
        })
        .build()
}
