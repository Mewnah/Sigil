//! Loading `sigil_transcribe.dll` and running models on it.
//!
//! Selection (`TranscribeModel::open`):
//! - With a runtime folder (`ModelOptions::runtime_dir` or `SIGIL_TRANSCRIBE_RUNTIME_DIR`; development
//!   and engine probes) that one DLL is used.
//! - Otherwise the installed `runtime/sigil-transcribe-<variant>` packs are tried in
//!   `RuntimeVariant::PREFERENCE` order among those this PC supports (CUDA 13, Vulkan, AVX-512, AVX2,
//!   baseline CPU), each through `native_runtime::open_engine` (hash-verified pack, NVIDIA runtime
//!   preloaded for `cuda13`). A variant whose DLL or GPU backend fails falls through to the next one,
//!   and every failure is kept in [`RuntimeSelection::attempts`]. A failure that belongs to the
//!   model or the options (a corrupt or unsupported GGUF, a bad `n_ctx`) ends the open at once:
//!   every variant would fail the same way after reloading a multi-GB file.
//!
//! GPU failures are never silent: a GPU model that fails to load, or whose backend reports a
//! failure while running (`SIGIL_E_BACKEND_UNAVAILABLE` / out of device memory), is reloaded on the
//! CPU backend of the same DLL when `ModelOptions::cpu_fallback` is set (transcribe.h documents this
//! recovery), and [`TranscribeModel::fallback_reason`] says why; so does a GPU runtime that failed
//! before a CPU runtime took over.
//!
//! **In-process abort paths.** Only failures the engine returns as status codes are recoverable.
//! ggml-cuda handles CUDA errors it cannot return (`CUDA_CHECK` / `CU_CHECK` / `CUBLAS_CHECK`: its
//! memory pool failing to grow when the GPU is full, a device lost to a driver reset) with
//! `GGML_ABORT`, which ends Sigil.exe; the shim only logs the message first. The wrapper keeps runs
//! away from that path: GPU loads need the model's estimated VRAM plus
//! `ModelOptions::min_free_vram_mb` free, the warm-up decodes the longest utterance Sigil sends so
//! the pool is sized at load time, and every GPU run first checks the headroom (moving the model to
//! the CPU when it is gone). A driver reset remains fatal in-process; ADR 0003b names it as the
//! trigger for hosting the engine in a helper process.

use serde::Serialize;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};

use super::types::{
    create_config, default_threads, estimated_model_vram_mb, is_gpu_backend, select_device, vram_shortfall, warmup_seconds, BackendChoice,
    DeviceInfo, DeviceList, DeviceRef, ModelInfo, ModelOptions, StreamFinal, StreamUpdate, TranscribeOptions, Transcript,
};
use crate::services::model_store::Catalog;
use crate::services::native_runtime::{
    abi, add_dll_directory, candidates, engine_dll_name, engine_pack_entries, open_engine, Attempt, AudioView, EngineCanceller, EngineError,
    EngineInstance, EngineLibrary, EngineStream, OpenRequest, PackManifest, RuntimeVariant, VariantSupport,
};

/// sigil_engine name of transcribe.cpp (`sigil_transcribe.dll`, packs `runtime/sigil-transcribe-*`).
pub const ENGINE: &str = "transcribe";
/// Development override: load `sigil_transcribe.dll` from this folder instead of the installed packs.
pub const RUNTIME_DIR_ENV: &str = "SIGIL_TRANSCRIBE_RUNTIME_DIR";
/// Development only (runtime folders): `;`-separated folders with `cublas64_13.dll` /
/// `cublasLt64_13.dll` (e.g. the CUDA 13 toolkit's `bin`; its `bin\x64` is added too) for a `cuda13`
/// DLL when the NVIDIA runtime pack is not installed. `SIGIL_TEST_CUDA_BIN` (the tests' convention)
/// is used when this is not set. Never set in production.
pub const DEV_CUDA_BIN_ENV: &str = "SIGIL_TRANSCRIBE_CUDA_BIN";
/// Sample rate every transcribe.cpp model consumes.
pub const SAMPLE_RATE: u32 = 16_000;

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// What went wrong, for callers that react differently (retry, fall back, report).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// No usable runtime pack (not installed, or no variant runs on this PC).
    NotInstalled,
    /// The model or the DLL failed to load (missing / corrupt / unsupported model).
    Load,
    /// The requested GPU backend is unavailable or failed.
    Backend,
    OutOfMemory,
    /// Bad audio or options (wrong language code, audio longer than the model accepts, ...).
    InvalidInput,
    /// The model cannot do this (e.g. streaming on an offline model).
    Unsupported,
    /// `TranscribeCanceller::cancel` interrupted the call.
    Cancelled,
    Runtime,
}

/// A failed transcribe.cpp call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TranscribeError {
    pub kind: ErrorKind,
    pub message: String,
}

impl TranscribeError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for TranscribeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for TranscribeError {}

impl From<TranscribeError> for String {
    fn from(e: TranscribeError) -> String {
        e.message
    }
}

/// Error kind of a sigil_engine status code.
pub(crate) fn kind_of_status(status: abi::SigilStatus) -> ErrorKind {
    match status {
        abi::SIGIL_E_INVALID_ARGUMENT => ErrorKind::InvalidInput,
        abi::SIGIL_E_UNSUPPORTED => ErrorKind::Unsupported,
        abi::SIGIL_E_LOAD_FAILED => ErrorKind::Load,
        abi::SIGIL_E_OUT_OF_MEMORY => ErrorKind::OutOfMemory,
        abi::SIGIL_E_CANCELLED => ErrorKind::Cancelled,
        abi::SIGIL_E_BACKEND_UNAVAILABLE => ErrorKind::Backend,
        _ => ErrorKind::Runtime,
    }
}

impl From<EngineError> for TranscribeError {
    fn from(e: EngineError) -> Self {
        Self::new(kind_of_status(e.status), e.message)
    }
}

/// Whether a failed model load on a GPU (`gpu`) or CPU backend should fall through to the next
/// variant: the backend is unavailable, or a GPU backend ran out of memory or failed at run time.
/// Everything else (a missing, corrupt or unsupported model, bad options) is the model's fault and
/// ends the open.
pub(crate) fn falls_through(status: abi::SigilStatus, gpu: bool) -> bool {
    match status {
        abi::SIGIL_E_BACKEND_UNAVAILABLE => true,
        abi::SIGIL_E_OUT_OF_MEMORY | abi::SIGIL_E_RUNTIME => gpu,
        _ => false,
    }
}

/// Fallback reason of a model that loaded on `variant` (on a GPU backend: `gpu`) after an earlier
/// GPU runtime failed (`gpu_failure`; a GPU variant skipped because this PC cannot run it is not a
/// failure): a CPU runtime taking over is never silent.
pub(crate) fn takeover_reason(gpu_failure: Option<&str>, gpu: bool, variant: RuntimeVariant) -> Option<String> {
    if gpu {
        return None;
    }
    gpu_failure.map(|why| format!("{why}; running on the {} CPU runtime", variant.as_str()))
}

/// Error kind of an open whose every attempt failed: the kind they share; else `OutOfMemory` when
/// any attempt ran out of memory (a smaller model may fit); else the last attempt's.
pub(crate) fn combined_kind(kinds: &[ErrorKind]) -> ErrorKind {
    let Some(last) = kinds.last().copied() else {
        return ErrorKind::NotInstalled;
    };
    if kinds.iter().all(|k| *k == last) {
        last
    } else if kinds.contains(&ErrorKind::OutOfMemory) {
        ErrorKind::OutOfMemory
    } else {
        last
    }
}

fn parse_json<T: serde::de::DeserializeOwned>(what: &str, json: &str) -> Result<T, TranscribeError> {
    serde_json::from_str(json).map_err(|e| TranscribeError::new(ErrorKind::Runtime, format!("transcribe {what}: bad engine JSON ({e})")))
}

// ---------------------------------------------------------------------------------------------
// Variant planning
// ---------------------------------------------------------------------------------------------

/// The runtime a model was opened from.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct RuntimeSelection {
    pub variant: Option<RuntimeVariant>,
    pub pack_id: Option<String>,
    pub dir: PathBuf,
    /// `sigil_engine_get_info` version / build info of the DLL.
    pub version: String,
    pub build_info: String,
    /// Variants that were skipped or failed first (in order).
    pub attempts: Vec<Attempt>,
}

/// Variants to try for `choice` among the installed `available` ones (pure; `candidates` does the
/// support filtering and preference order).
pub(crate) fn plan_variants(
    choice: BackendChoice,
    forced: Option<RuntimeVariant>,
    available: &[RuntimeVariant],
    support: &VariantSupport,
) -> Vec<RuntimeVariant> {
    let order = candidates(available, support, forced);
    if forced.is_some() {
        return order;
    }
    match choice {
        BackendChoice::Auto => order,
        BackendChoice::Gpu => order.into_iter().filter(|v| v.is_gpu()).collect(),
        BackendChoice::Cpu => order.into_iter().filter(|v| !v.is_gpu()).collect(),
    }
}

/// `sigil_engine_create` backend for `variant` under `choice` (`None`: the variant's own backend).
pub(crate) fn backend_for(choice: BackendChoice, variant: Option<RuntimeVariant>) -> Option<&'static str> {
    match choice {
        BackendChoice::Cpu => Some("cpu"),
        BackendChoice::Gpu | BackendChoice::Auto => match variant {
            Some(RuntimeVariant::Cuda13) => Some("cuda"),
            Some(RuntimeVariant::Vulkan) => Some("vulkan"),
            Some(_) => Some("cpu"),
            None => None,
        },
    }
}

fn catalog() -> Result<&'static Catalog, TranscribeError> {
    static CATALOG: OnceLock<Result<Catalog, String>> = OnceLock::new();
    CATALOG
        .get_or_init(Catalog::load)
        .as_ref()
        .map_err(|e| TranscribeError::new(ErrorKind::NotInstalled, format!("model catalog: {e}")))
}

/// Installed transcribe packs: `(variant, catalog id, folder)` in catalog order.
pub fn installed_variants() -> Vec<(RuntimeVariant, String, PathBuf)> {
    let Ok(catalog) = catalog() else {
        return Vec::new();
    };
    engine_pack_entries(catalog, Some(ENGINE))
        .into_iter()
        .filter_map(|e| crate::services::model_store::installed_path(&e.id).map(|dir| (e.variant, e.id, dir)))
        .collect()
}

fn env_path(name: &str) -> Option<PathBuf> {
    std::env::var_os(name)
        .filter(|v| !v.is_empty())
        .map(PathBuf::from)
}

/// Development CUDA folders for runtime-folder loads of a `cuda13` DLL ([`DEV_CUDA_BIN_ENV`], then
/// `SIGIL_TEST_CUDA_BIN`; `;`-separated). A CUDA 13 toolkit `bin` also contributes its `bind`.
fn dev_cuda_dirs() -> Vec<PathBuf> {
    let Some(list) = std::env::var_os(DEV_CUDA_BIN_ENV)
        .filter(|v| !v.is_empty())
        .or_else(|| std::env::var_os("SIGIL_TEST_CUDA_BIN").filter(|v| !v.is_empty()))
    else {
        return Vec::new();
    };
    let mut dirs = Vec::new();
    for d in std::env::split_paths(&list).filter(|d| d.is_dir()) {
        let x64 = d.join("x64");
        if x64.join("cudart64_13.dll").is_file() || x64.join("cublas64_13.dll").is_file() {
            dirs.push(x64);
        }
        dirs.push(d);
    }
    dirs
}

/// Register the development CUDA folders as DLL search folders (no-op without the env variables).
/// Returns whether any folder was registered.
pub(crate) fn register_dev_cuda_dirs() -> bool {
    let dirs = dev_cuda_dirs();
    for d in &dirs {
        if let Err(e) = add_dll_directory(d) {
            tracing::debug!(target: "sigil::transcribe", "{e}");
        }
    }
    !dirs.is_empty()
}

fn threads_for(opts: &ModelOptions, gpu: bool) -> u32 {
    opts.threads
        .filter(|t| *t > 0)
        .unwrap_or_else(|| default_threads(gpu))
}

/// A failed model load on one backend.
#[derive(Debug)]
struct CreateFailure {
    error: TranscribeError,
    /// The failure belongs to the backend or the build ([`falls_through`]), so another variant or
    /// the CPU may succeed.
    falls_through: bool,
}

/// The devices of `library` as a model-less instance reports them (`None` when it cannot answer;
/// the real create then reports the failure).
fn library_devices(library: &EngineLibrary) -> Option<Vec<DeviceInfo>> {
    let mut probe = library.create_engine("{}").ok()?;
    let json = probe.query_engine(r#"{"op":"devices"}"#).ok()?;
    let list: DeviceList = serde_json::from_str(&json).ok()?;
    Some(list.devices.into_iter().flatten().collect())
}

/// Load the model on `library` with `backend` (`None`: the DLL's own backend).
///
/// On a GPU backend the device `device` names is looked up first. An index resolves to the
/// device's stable id, written back into `device` so a later GPU backend (whose device order
/// differs) targets the same GPU or fails over instead of picking another one. With
/// `opts.min_free_vram_mb` set, a device without the model's estimated VRAM plus that headroom
/// is refused as out of memory before anything loads (a load or first run that fills the GPU can
/// end the process inside ggml-cuda).
///
/// With `cpu_retry`, a GPU failure that belongs to the backend reloads the model on the same DLL's
/// CPU backend. Returns the instance and the fallback reason.
fn create_on(
    library: &EngineLibrary,
    opts: &ModelOptions,
    backend: Option<&str>,
    device: &mut Option<DeviceRef>,
    cpu_retry: bool,
) -> Result<(EngineInstance, Option<String>), CreateFailure> {
    let gpu = backend.map_or_else(|| library.info().has(abi::SIGIL_CAP_GPU), is_gpu_backend);
    let gpu_error = match create_gpu_or_cpu(library, opts, backend, device, gpu) {
        Ok(instance) => return Ok((instance, None)),
        Err(f) if gpu && cpu_retry && f.falls_through => f.error,
        Err(f) => return Err(f),
    };
    tracing::warn!(
        target: "sigil::transcribe",
        "GPU backend could not take the model ({}); loading it on the CPU",
        gpu_error.message
    );
    let cpu = create_config(opts, Some("cpu"), threads_for(opts, false), None);
    match library.create_engine(&cpu) {
        Ok(instance) => Ok((
            instance,
            Some(format!("the GPU backend could not load the model ({}); running on the CPU", gpu_error.message)),
        )),
        Err(cpu_err) => Err(CreateFailure {
            error: TranscribeError::new(
                kind_of_status(cpu_err.status),
                format!("{} (the CPU fallback failed too: {})", gpu_error.message, cpu_err.message),
            ),
            falls_through: falls_through(cpu_err.status, false),
        }),
    }
}

/// One create on `backend`, with the GPU device lookup and VRAM check of [`create_on`].
fn create_gpu_or_cpu(
    library: &EngineLibrary,
    opts: &ModelOptions,
    backend: Option<&str>,
    device: &mut Option<DeviceRef>,
    gpu: bool,
) -> Result<EngineInstance, CreateFailure> {
    if let Some(b) = backend.filter(|b| is_gpu_backend(b)) {
        if let Some(found) = library_devices(library).and_then(|all| select_device(&all, b, device.as_ref())) {
            if matches!(device, Some(DeviceRef::Index(_))) {
                if let Some(stable) = DeviceRef::stable_for(&found) {
                    *device = Some(stable);
                }
            }
            if opts.min_free_vram_mb > 0 {
                let file_bytes = std::fs::metadata(&opts.model).map(|m| m.len()).unwrap_or(0);
                let need = estimated_model_vram_mb(file_bytes) + u64::from(opts.min_free_vram_mb);
                if let Some(why) = vram_shortfall(&found, need) {
                    return Err(CreateFailure {
                        error: TranscribeError::new(ErrorKind::OutOfMemory, format!("{b}: {why} to load {}", opts.model.display())),
                        falls_through: true,
                    });
                }
            }
        }
    }
    let config = create_config(opts, backend, threads_for(opts, gpu), device.as_ref());
    library.create_engine(&config).map_err(|e| CreateFailure {
        falls_through: falls_through(e.status, gpu),
        error: e.into(),
    })
}

// ---------------------------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------------------------

/// Cancels the model's running call from any thread. It follows the model across a CPU fallback:
/// it always reaches the model's current instance, and with it the streams opened from that
/// instance (streams opened before a fallback are closed by the fallback; see
/// [`TranscribeModel::open_stream`]).
#[derive(Clone)]
pub struct TranscribeCanceller {
    current: Arc<Mutex<EngineCanceller>>,
}

impl TranscribeCanceller {
    /// Interrupt the running `transcribe` / warm-up / stream push or finish (between decode steps;
    /// the encoder pass in flight finishes first, and Qwen3-ASR 0.2.4 checks only before its
    /// single-shot decode). A call that starts afterwards is not affected.
    pub fn cancel(&self) {
        self.current
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .cancel();
    }
}

/// A stream's engine session; emptied when a CPU fallback closes it.
type StreamSlot = Arc<Mutex<Option<EngineStream>>>;

fn lock_slot(slot: &Mutex<Option<EngineStream>>) -> std::sync::MutexGuard<'_, Option<EngineStream>> {
    slot.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One loaded transcribe.cpp model. `Send` (move it into `spawn_blocking`), not `Sync`.
///
/// transcribe.cpp 0.x runs one compute per model at a time, and an open stream counts as running
/// from its first push until `finish` (or a failed push). While a stream of this model is in that
/// state, [`Self::transcribe`], the warm-up and the first push of any other stream of this model fail
/// with [`ErrorKind::InvalidInput`]; load a second model for parallel work.
pub struct TranscribeModel {
    library: EngineLibrary,
    instance: EngineInstance,
    canceller: Arc<Mutex<EngineCanceller>>,
    info: ModelInfo,
    runtime: RuntimeSelection,
    options: ModelOptions,
    fallback: Option<String>,
    /// Streams opened from `instance` (closed by a CPU fallback so they release the GPU copy).
    streams: Vec<Weak<Mutex<Option<EngineStream>>>>,
}

impl std::fmt::Debug for TranscribeModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscribeModel")
            .field("model", &self.options.model)
            .field("backend", &self.info.backend)
            .field("variant", &self.runtime.variant)
            .finish()
    }
}

impl TranscribeModel {
    /// Load the runtime (see the module docs) and the model; optionally warm it up.
    pub fn open(opts: &ModelOptions) -> Result<TranscribeModel, TranscribeError> {
        if !opts.model.is_file() {
            return Err(TranscribeError::new(
                ErrorKind::Load,
                format!("model file {} does not exist", opts.model.display()),
            ));
        }
        let runtime_dir = opts
            .runtime_dir
            .clone()
            .or_else(|| env_path(RUNTIME_DIR_ENV));
        let (library, instance, runtime, fallback) = match runtime_dir {
            Some(dir) => Self::open_from_dir(&dir, opts)?,
            None => Self::open_installed(opts)?,
        };
        let canceller = Arc::new(Mutex::new(instance.canceller()));
        let mut model = TranscribeModel {
            library,
            instance,
            canceller,
            info: ModelInfo::default(),
            runtime,
            options: opts.clone(),
            fallback,
            streams: Vec::new(),
        };
        model.refresh_info()?;
        tracing::info!(
            target: "sigil::transcribe",
            "loaded {} ({}) on {} with {} {} [{}]{}",
            opts.model.display(),
            model.info.arch.as_deref().unwrap_or("?"),
            model.info.backend,
            ENGINE,
            model.runtime.version,
            model.runtime.variant.map_or("runtime folder", |v| v.as_str()),
            model.fallback.as_deref().map(|f| format!("; {f}")).unwrap_or_default()
        );
        if opts.warmup {
            model.warmup()?;
        }
        Ok(model)
    }

    /// Development / probe path: the DLL in `dir`.
    fn open_from_dir(dir: &Path, opts: &ModelOptions) -> Result<(EngineLibrary, EngineInstance, RuntimeSelection, Option<String>), TranscribeError> {
        if !dir.is_dir() {
            return Err(TranscribeError::new(
                ErrorKind::NotInstalled,
                format!("{} is not a folder", dir.display()),
            ));
        }
        let manifest = PackManifest::read(dir).map_err(|e| TranscribeError::new(ErrorKind::Load, e))?;
        let variant = opts.variant.or_else(|| {
            manifest
                .as_ref()
                .and_then(|m| RuntimeVariant::parse(&m.variant))
        });
        let file = manifest
            .as_ref()
            .map(|m| m.library.clone())
            .unwrap_or_else(|| engine_dll_name(ENGINE));
        let library = if variant == Some(RuntimeVariant::Cuda13) && register_dev_cuda_dirs() {
            // Development: CUDA DLLs from a toolkit folder instead of the NVIDIA runtime pack.
            EngineLibrary::load(dir, &file).map_err(|e| TranscribeError::new(ErrorKind::Load, e))?
        } else {
            let opened = open_engine(
                ENGINE,
                &OpenRequest {
                    variant,
                    runtime_dir: Some(dir.to_path_buf()),
                    config_json: None,
                },
            )
            .map_err(|e| TranscribeError::new(ErrorKind::Load, e))?;
            opened.library
        };
        if library.info().name != ENGINE {
            return Err(TranscribeError::new(
                ErrorKind::Load,
                format!("{} is the '{}' engine, not '{ENGINE}'", library.path().display(), library.info().name),
            ));
        }
        let lib_variant = RuntimeVariant::parse(&library.info().variant).or(variant);
        if opts.backend == BackendChoice::Gpu && lib_variant.is_some_and(|v| !v.is_gpu()) {
            return Err(TranscribeError::new(
                ErrorKind::Backend,
                format!(
                    "{} is the {} build; a GPU backend needs the cuda13 or vulkan runtime",
                    library.path().display(),
                    library.info().variant
                ),
            ));
        }
        let mut device = opts.device_ref();
        let (instance, fallback) =
            create_on(&library, opts, backend_for(opts.backend, lib_variant), &mut device, opts.cpu_fallback).map_err(|f| f.error)?;
        let runtime = RuntimeSelection {
            variant: lib_variant,
            pack_id: None,
            dir: dir.to_path_buf(),
            version: library.info().version.clone(),
            build_info: library.info().build_info.clone(),
            attempts: Vec::new(),
        };
        Ok((library, instance, runtime, fallback))
    }

    /// Installed packs, best variant first.
    fn open_installed(opts: &ModelOptions) -> Result<(EngineLibrary, EngineInstance, RuntimeSelection, Option<String>), TranscribeError> {
        if opts.backend == BackendChoice::Gpu {
            if let Some(v) = opts.variant.filter(|v| !v.is_gpu()) {
                // Same answer as a runtime folder holding a CPU build: never a silent CPU run.
                return Err(TranscribeError::new(
                    ErrorKind::Backend,
                    format!(
                        "a GPU backend was requested, but only the {} runtime (a CPU build) was allowed",
                        v.as_str()
                    ),
                ));
            }
        }
        let installed = installed_variants();
        if installed.is_empty() {
            return Err(TranscribeError::new(
                ErrorKind::NotInstalled,
                "the transcribe.cpp runtime is not installed (install a runtime/sigil-transcribe-* pack in Models & Runtimes)",
            ));
        }
        let available: Vec<RuntimeVariant> = installed.iter().map(|(v, _, _)| *v).collect();
        let support = VariantSupport::detect();
        let order = plan_variants(opts.backend, opts.variant, &available, &support);
        let mut attempts: Vec<Attempt> = installed
            .iter()
            .filter(|(v, _, _)| !order.contains(v))
            .map(|(v, id, dir)| Attempt {
                variant: Some(*v),
                pack_id: Some(id.clone()),
                dir: Some(dir.clone()),
                error: support
                    .unsupported_reason(*v)
                    .unwrap_or_else(|| match (opts.variant, opts.backend) {
                        (Some(forced), _) => format!("only {} was requested", forced.as_str()),
                        (None, BackendChoice::Gpu) => "a GPU backend was requested".into(),
                        (None, BackendChoice::Cpu) => "the CPU backend was requested".into(),
                        (None, BackendChoice::Auto) => "not selected".into(),
                    }),
            })
            .collect();
        if order.is_empty() {
            return Err(TranscribeError::new(
                ErrorKind::NotInstalled,
                format!(
                    "no installed transcribe.cpp variant fits ({}): {}",
                    match opts.backend {
                        BackendChoice::Gpu => "GPU requested",
                        BackendChoice::Cpu => "CPU requested",
                        BackendChoice::Auto => "any",
                    },
                    attempts
                        .iter()
                        .map(|a| format!("{}: {}", a.variant.map_or("?", |v| v.as_str()), a.error))
                        .collect::<Vec<_>>()
                        .join("; ")
                ),
            ));
        }
        // Kinds of the failed tries (not the skips), and the first GPU runtime that failed: a CPU
        // runtime that takes over reports it as the fallback reason.
        let mut kinds: Vec<ErrorKind> = Vec::new();
        let mut gpu_failure: Option<String> = None;
        // The requested GPU; an index becomes the device's stable id once a GPU backend saw it.
        let mut device = opts.device_ref();
        for (i, variant) in order.iter().copied().enumerate() {
            let backend = backend_for(opts.backend, Some(variant));
            let gpu = variant.is_gpu() && backend != Some("cpu");
            let is_last = i + 1 == order.len();
            let pack = installed.iter().find(|(v, _, _)| *v == variant);
            let attempt = |error: String| Attempt {
                variant: Some(variant),
                pack_id: pack.map(|(_, id, _)| id.clone()),
                dir: pack.map(|(_, _, d)| d.clone()),
                error,
            };
            // 1. The DLL (hash-verified pack; the NVIDIA runtime for cuda13). A failure here is the
            //    variant's own, so the next variant is tried.
            let request = OpenRequest {
                variant: Some(variant),
                runtime_dir: None,
                config_json: None,
            };
            let opened = match open_engine(ENGINE, &request) {
                Ok(opened) => opened,
                Err(error) => {
                    tracing::warn!(target: "sigil::transcribe", "{} runtime failed: {error}", variant.as_str());
                    if gpu {
                        gpu_failure.get_or_insert_with(|| format!("the {} runtime failed ({error})", variant.as_str()));
                    }
                    attempts.push(attempt(error));
                    kinds.push(ErrorKind::Load);
                    continue;
                }
            };
            // 2. The model. The last candidate may fall back to its own CPU backend; earlier GPU
            //    candidates fall through to the next variant instead.
            let mut no_device = None;
            let device_slot = if gpu { &mut device } else { &mut no_device };
            match create_on(&opened.library, opts, backend, device_slot, is_last && gpu && opts.cpu_fallback) {
                Ok((instance, fallback)) => {
                    let fallback = fallback.or_else(|| takeover_reason(gpu_failure.as_deref(), gpu, variant));
                    // `opened.attempts` only lists the variants this forced request left out; the
                    // real skips and failures are tracked here.
                    let runtime = RuntimeSelection {
                        variant: opened.variant,
                        pack_id: opened.pack_id,
                        dir: opened.dir,
                        version: opened.library.info().version.clone(),
                        build_info: opened.library.info().build_info.clone(),
                        attempts,
                    };
                    return Ok((opened.library, instance, runtime, fallback));
                }
                Err(failure) => {
                    let error = failure.error;
                    tracing::warn!(target: "sigil::transcribe", "{} could not load the model: {error}", variant.as_str());
                    if gpu {
                        gpu_failure.get_or_insert_with(|| format!("the {} runtime could not load the model ({error})", variant.as_str()));
                    }
                    attempts.push(attempt(error.message.clone()));
                    kinds.push(error.kind);
                    if !failure.falls_through {
                        // The model or the options are at fault: every variant would fail the same
                        // way (after reloading the whole file).
                        return Err(TranscribeError::new(
                            error.kind,
                            format!("transcribe.cpp could not load {}: {}", opts.model.display(), error.message),
                        ));
                    }
                }
            }
        }
        Err(TranscribeError::new(
            combined_kind(&kinds),
            format!(
                "transcribe.cpp could not load {}: {}",
                opts.model.display(),
                attempts
                    .iter()
                    .map(|a| format!("{}: {}", a.variant.map_or("?", |v| v.as_str()), a.error))
                    .collect::<Vec<_>>()
                    .join("; ")
            ),
        ))
    }

    fn refresh_info(&mut self) -> Result<(), TranscribeError> {
        let json = self.instance.query_engine(r#"{"op":"info"}"#)?;
        self.info = parse_json("info", &json)?;
        Ok(())
    }

    /// What the engine reports about the loaded model (backend actually used, device, languages,
    /// limits, load time).
    pub fn info(&self) -> &ModelInfo {
        &self.info
    }

    /// Where the runtime came from and which variants failed first.
    pub fn runtime(&self) -> &RuntimeSelection {
        &self.runtime
    }

    /// Why the model runs on the CPU although a GPU backend was requested (`None` otherwise).
    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback.as_deref()
    }

    #[allow(dead_code)] // API for the P2 speech plugin.
    pub fn options(&self) -> &ModelOptions {
        &self.options
    }

    /// Tests: change the VRAM headroom of a loaded model (to force the pre-run check).
    #[cfg(test)]
    pub(crate) fn set_min_free_vram_mb(&mut self, mb: u32) {
        self.options.min_free_vram_mb = mb;
    }

    pub fn canceller(&self) -> TranscribeCanceller {
        TranscribeCanceller {
            current: Arc::clone(&self.canceller),
        }
    }

    /// Pay first-run costs now: graph allocation, GPU kernel / pipeline setup and, on a GPU,
    /// ggml-cuda's memory pool at the size of the longest utterance Sigil sends
    /// ([`ModelOptions::warmup_seconds`]). With `cpu_fallback`, a GPU left without
    /// [`ModelOptions::min_free_vram_mb`] of headroom afterwards moves the model to the CPU right away.
    /// Returns the wall ms of the warm-up decode.
    pub fn warmup(&mut self) -> Result<f64, TranscribeError> {
        let json = match self.run_warmup() {
            Ok(j) => j,
            Err(e) if self.can_fall_back(&e) => {
                self.fall_back_to_cpu(&format!("the backend failed ({})", e.message))?;
                self.run_warmup()?
            }
            Err(e) => return Err(e.into()),
        };
        if self.options.cpu_fallback {
            self.ensure_gpu_headroom()?;
        }
        let v: serde_json::Value = parse_json("warmup", &json)?;
        Ok(v.get("wall_ms")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0))
    }

    fn run_warmup(&mut self) -> Result<String, EngineError> {
        let seconds = warmup_seconds(self.options.warmup_seconds, self.info.on_gpu(), self.info.max_audio_ms());
        self.instance
            .query_engine(&serde_json::json!({ "op": "warmup", "seconds": seconds }).to_string())
    }

    /// Live memory of the model's device (a snapshot; device-wide for a discrete GPU).
    pub fn device_memory(&mut self) -> Result<Option<DeviceInfo>, TranscribeError> {
        let json = self.instance.query_engine(r#"{"op":"memory"}"#)?;
        #[derive(serde::Deserialize)]
        struct Memory {
            device: Option<DeviceInfo>,
        }
        Ok(parse_json::<Memory>("memory", &json)?.device)
    }

    /// Whether a failed call may be retried on the CPU copy: only failures the engine returns as
    /// status codes (`SIGIL_E_BACKEND_UNAVAILABLE`, out of device memory). A CUDA error that
    /// ggml-cuda cannot return aborts the process before any status exists; see
    /// [`Self::ensure_gpu_headroom`] for what keeps runs away from it.
    fn can_fall_back(&self, e: &EngineError) -> bool {
        self.options.cpu_fallback && self.info.on_gpu() && matches!(e.status, abi::SIGIL_E_BACKEND_UNAVAILABLE | abi::SIGIL_E_OUT_OF_MEMORY)
    }

    /// GPU only, before every run: the device must still have [`ModelOptions::min_free_vram_mb`]
    /// free. ggml-cuda grows its memory pool during a run and aborts the process when that fails,
    /// so a run never starts at the edge: the model moves to the CPU (`cpu_fallback`) or the call
    /// fails with [`ErrorKind::OutOfMemory`]. A device whose free memory is unknown is not checked.
    fn ensure_gpu_headroom(&mut self) -> Result<(), TranscribeError> {
        let need = u64::from(self.options.min_free_vram_mb);
        if need == 0 || !self.info.on_gpu() {
            return Ok(());
        }
        let Ok(Some(device)) = self.device_memory() else {
            return Ok(());
        };
        let Some(short) = vram_shortfall(&device, need) else {
            return Ok(());
        };
        let why = format!("{short} to run safely");
        if self.options.cpu_fallback {
            self.fall_back_to_cpu(&why)
        } else {
            Err(TranscribeError::new(
                ErrorKind::OutOfMemory,
                format!(
                    "{}: {why}; the run was not started (a failed VRAM allocation inside ggml-cuda ends the process)",
                    self.info.backend
                ),
            ))
        }
    }

    /// Close every stream opened from the current instance (their next call fails with
    /// [`ErrorKind::Backend`]), so they stop holding it. Returns how many were open.
    fn close_streams(&mut self) -> usize {
        let mut closed = 0;
        for weak in self.streams.drain(..) {
            if let Some(slot) = weak.upgrade() {
                let stream = lock_slot(&slot).take();
                if stream.is_some() {
                    closed += 1;
                }
                drop(stream);
            }
        }
        closed
    }

    /// Reload the model on the CPU backend of the same DLL after a GPU failure (or a GPU without
    /// headroom). The GPU copy's streams are closed first, and the GPU copy itself is released when
    /// the CPU copy replaces it, so both never stay resident.
    fn fall_back_to_cpu(&mut self, why: &str) -> Result<(), TranscribeError> {
        let gpu_backend = self.info.backend.clone();
        let closed = self.close_streams();
        tracing::warn!(
            target: "sigil::transcribe",
            "{gpu_backend}: {why}; reloading {} on the CPU{}",
            self.options.model.display(),
            if closed > 0 {
                format!(" ({closed} open stream(s) closed)")
            } else {
                String::new()
            }
        );
        let config = create_config(&self.options, Some("cpu"), threads_for(&self.options, false), None);
        let instance = self.library.create_engine(&config)?;
        *self
            .canceller
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = instance.canceller();
        self.instance = instance;
        self.fallback = Some(format!("{gpu_backend}: {why}; running on the CPU"));
        self.refresh_info()
    }

    /// Transcribe one utterance of 16 kHz mono audio. On a GPU the VRAM headroom is checked first
    /// ([`ModelOptions::min_free_vram_mb`]). Fails with [`ErrorKind::InvalidInput`] while a stream
    /// of this model is mid-utterance (one compute per model).
    pub fn transcribe(&mut self, pcm: &[f32], opts: &TranscribeOptions) -> Result<Transcript, TranscribeError> {
        if pcm.is_empty() {
            return Err(TranscribeError::new(ErrorKind::InvalidInput, "empty audio"));
        }
        let max_ms = self.info.max_audio_ms();
        let audio_ms = pcm.len() as i64 * 1000 / i64::from(SAMPLE_RATE);
        if max_ms > 0 && audio_ms > max_ms {
            return Err(TranscribeError::new(
                ErrorKind::InvalidInput,
                format!("{audio_ms} ms of audio is longer than this model accepts in one call ({max_ms} ms)"),
            ));
        }
        self.ensure_gpu_headroom()?;
        let options = opts.to_json();
        let view = AudioView {
            samples: pcm,
            sample_rate: SAMPLE_RATE,
            channels: 1,
        };
        let json = match self.instance.process_audio(view, &options) {
            Ok(j) => j,
            Err(e) if self.can_fall_back(&e) => {
                self.fall_back_to_cpu(&format!("the backend failed ({})", e.message))?;
                self.instance.process_audio(view, &options)?
            }
            Err(e) => return Err(e.into()),
        };
        let transcript: Transcript = parse_json("result", &json)?;
        if transcript.truncated {
            tracing::warn!(target: "sigil::transcribe", "transcript truncated at the model's generation limit");
        }
        Ok(transcript)
    }

    /// Transcribe interleaved audio at any rate / channel count (resampled to 16 kHz mono).
    #[allow(dead_code)] // API for the P2 speech plugin (the probe resamples once, up front).
    pub fn transcribe_audio(
        &mut self,
        samples: &[f32],
        sample_rate: u32,
        channels: u16,
        opts: &TranscribeOptions,
    ) -> Result<Transcript, TranscribeError> {
        let pcm = to_16k_mono(samples, sample_rate, channels)?;
        self.transcribe(&pcm, opts)
    }

    /// Open a streaming session (models whose capabilities say `streaming`, e.g. Nemotron,
    /// Parakeet streaming, Voxtral Realtime, Moonshine streaming).
    ///
    /// - One compute per model: from a stream's first push until its `finish` (or a failed push),
    ///   [`Self::transcribe`] and the first push of every other stream of this model fail with
    ///   [`ErrorKind::InvalidInput`]. Two live caption sources need two loaded models.
    /// - A CPU fallback of the model closes its open streams: their next push / finish fails with
    ///   [`ErrorKind::Backend`] ([`TranscribeStream::is_open`] turns false); open a new stream.
    /// - The VRAM headroom is checked here, not on every push (the warm-up sized the GPU pool).
    pub fn open_stream(&mut self, opts: &TranscribeOptions) -> Result<TranscribeStream, TranscribeError> {
        if !self.info.capabilities.streaming {
            return Err(TranscribeError::new(
                ErrorKind::Unsupported,
                format!(
                    "the {} model does not stream; transcribe each utterance instead",
                    self.info.arch.as_deref().unwrap_or("loaded")
                ),
            ));
        }
        self.ensure_gpu_headroom()?;
        let stream = self.instance.open_engine_stream(&opts.to_json())?;
        let slot: StreamSlot = Arc::new(Mutex::new(Some(stream)));
        self.streams.retain(|w| w.strong_count() > 0);
        self.streams.push(Arc::downgrade(&slot));
        Ok(TranscribeStream { slot })
    }
}

/// Interleaved audio at any rate → 16 kHz mono (exact length; rubato via `speech_core`).
pub fn to_16k_mono(samples: &[f32], sample_rate: u32, channels: u16) -> Result<Vec<f32>, TranscribeError> {
    if channels == 0 || sample_rate == 0 {
        return Err(TranscribeError::new(
            ErrorKind::InvalidInput,
            format!("unsupported audio format: {sample_rate} Hz, {channels} channels"),
        ));
    }
    if sample_rate == SAMPLE_RATE && channels == 1 {
        return Ok(samples.to_vec());
    }
    let mut rs = crate::services::speech_core::resample::StreamResampler::new(sample_rate, channels)
        .map_err(|e| TranscribeError::new(ErrorKind::InvalidInput, e))?;
    let mut out = Vec::with_capacity(samples.len() / usize::from(channels) * SAMPLE_RATE as usize / sample_rate as usize + 16);
    rs.push_interleaved(samples, &mut out)
        .and_then(|()| rs.flush(&mut out))
        .map_err(|e| TranscribeError::new(ErrorKind::Runtime, format!("resampling: {e}")))?;
    Ok(out)
}

/// A streaming session: push 16 kHz mono chunks, read partials, finish per utterance (the session
/// can then be pushed again for the next utterance). See [`TranscribeModel::open_stream`] for the
/// one-compute-per-model rule and what a CPU fallback does to it.
pub struct TranscribeStream {
    slot: StreamSlot,
}

impl std::fmt::Debug for TranscribeStream {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TranscribeStream")
            .field("open", &self.is_open())
            .finish_non_exhaustive()
    }
}

/// The error of a call on a stream that its model's CPU fallback closed.
pub(crate) fn stream_closed_error() -> TranscribeError {
    TranscribeError::new(
        ErrorKind::Backend,
        "this stream was closed when its model moved to the CPU after a GPU failure; open a new stream",
    )
}

impl TranscribeStream {
    /// Whether the stream still runs on its model's current instance (false after a CPU fallback
    /// of the model closed it).
    pub fn is_open(&self) -> bool {
        lock_slot(&self.slot).is_some()
    }

    /// Push audio; `Some(update)` when the hypothesis changed.
    pub fn push(&mut self, pcm: &[f32]) -> Result<Option<StreamUpdate>, TranscribeError> {
        let mut slot = lock_slot(&self.slot);
        let stream = slot.as_mut().ok_or_else(stream_closed_error)?;
        if pcm.is_empty() {
            return Ok(None);
        }
        let view = AudioView {
            samples: pcm,
            sample_rate: SAMPLE_RATE,
            channels: 1,
        };
        match stream.push_audio(view)? {
            Some(json) => Ok(Some(parse_json("stream update", &json)?)),
            None => Ok(None),
        }
    }

    /// Flush and return the final transcript of the current utterance.
    pub fn finish(&mut self) -> Result<StreamFinal, TranscribeError> {
        let mut slot = lock_slot(&self.slot);
        let stream = slot.as_mut().ok_or_else(stream_closed_error)?;
        let json = stream.finish_stream()?;
        parse_json("stream final", &json)
    }
}
