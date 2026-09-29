//! Loader for sigil_engine ABI v1 DLLs (`native/common/sigil_engine.h`).
//!
//! [`EngineLibrary::load`] is the only way an engine DLL enters the process:
//! 1. an already loaded copy of the same file is reused (one module per path);
//! 2. the files listed in the pack manifest (`sigil-pack.json`, written by
//!    `scripts/native/build-runtime-packs.ps1`) are checked against their recorded size and sha256
//!    before any code from the folder runs (`pack::verify_library`), and the manifest's variant
//!    must be one this CPU can execute (an installed pack must have a manifest:
//!    [`EngineLibrary::load_checked`]);
//! 3. the DLL is loaded by full path ([`super::load_library_in_dir`]);
//! 4. `sigil_engine_abi_version()` is resolved and called FIRST, and the DLL is refused (and unloaded)
//!    unless it returns [`abi::SIGIL_ENGINE_ABI_VERSION`];
//! 5. every other export of [`abi::SYMBOLS`] is resolved (a missing one refuses the DLL, naming it);
//! 6. `sigil_engine_get_info()` is validated (size, ABI field, UTF-8 strings) and compared with the
//!    pack manifest, and its variant must be one this CPU can execute (the only statement of the
//!    instruction sets in a folder without a manifest) before any engine call is made;
//! 7. engine logs are routed to `tracing` (target `sigil::engine`).
//!
//! Calls follow the header's rules: every call on one engine instance and its streams is
//! serialized by a per-instance lock (so an [`EngineInstance`] and its [`EngineStream`]s may live on
//! different threads), `sigil_engine_cancel` is the only call made without it
//! ([`EngineCanceller`]), strings and audio returned by the engine are copied and released with the
//! same DLL's `sigil_free_*`, and the thread-local `sigil_engine_last_error()` is read right after
//! the failing call on the same thread. Destroy order is guaranteed by reference counting: a stream
//! keeps its instance alive, an instance keeps its library loaded, and `sigil_engine_destroy` runs
//! only after the last stream was closed.
//!
//! Libraries are unloaded when the last handle drops, except GPU-capable ones
//! (`SIGIL_CAP_GPU`), which stay loaded for the rest of the process: GPU drivers and runtimes do not
//! tolerate being unloaded from under their own threads.

use std::collections::HashMap;
use std::ffi::{c_char, c_void, CStr, CString};
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};

use serde::{Deserialize, Serialize};

use super::abi::{self, SigilAudioBuffer, SigilAudioView, SigilEngine, SigilEngineInfo, SigilStatus, SigilStream};
use super::dll_dirs::load_library_in_dir;
use super::pack;
use super::variant::VariantSupport;

/// Longest engine-provided message kept in an error or log line.
const MAX_MESSAGE_CHARS: usize = 2000;
/// Largest audio buffer accepted from an engine (samples, all channels): 2^30 f32 = 4 GiB.
const MAX_OUTPUT_SAMPLES: u64 = 1 << 30;
/// Channel counts accepted from an engine.
const MAX_CHANNELS: u32 = 64;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn truncate(s: &str, max: usize) -> String {
    crate::services::diagnostics::truncate_chars(s, max)
}

// ---------------------------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------------------------

/// `sigil_engine_get_info`, copied into owned strings.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EngineInfo {
    pub abi_version: u32,
    pub name: String,
    pub version: String,
    pub variant: String,
    pub kind: String,
    pub build_info: String,
    /// `abi::SIGIL_CAP_*` bits.
    pub capabilities: u64,
}

impl EngineInfo {
    pub fn has(&self, capability: u64) -> bool {
        self.capabilities & capability == capability
    }

    /// Names of the capability bits (`process`, `stream`, `render`, `cancel`, `gpu`).
    pub fn capability_names(&self) -> Vec<&'static str> {
        [
            (abi::SIGIL_CAP_PROCESS, "process"),
            (abi::SIGIL_CAP_STREAM, "stream"),
            (abi::SIGIL_CAP_RENDER, "render"),
            (abi::SIGIL_CAP_CANCEL, "cancel"),
            (abi::SIGIL_CAP_GPU, "gpu"),
        ]
        .into_iter()
        .filter(|(bit, _)| self.has(*bit))
        .map(|(_, name)| name)
        .collect()
    }
}

/// Borrowed input audio (interleaved f32).
#[derive(Clone, Copy, Debug)]
pub struct AudioView<'a> {
    pub samples: &'a [f32],
    pub sample_rate: u32,
    pub channels: u16,
}

/// Owned output audio (interleaved f32), copied out of the engine buffer.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AudioBuffer {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

impl AudioBuffer {
    /// Frames (samples per channel).
    pub fn frames(&self) -> usize {
        if self.channels == 0 {
            0
        } else {
            self.samples.len() / usize::from(self.channels)
        }
    }
}

/// A failed engine call: the status code and a message (with the engine's last error).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EngineError {
    pub status: SigilStatus,
    pub message: String,
}

impl EngineError {
    pub(crate) fn host(status: SigilStatus, message: impl Into<String>) -> Self {
        Self {
            status,
            message: message.into(),
        }
    }

    /// Whether the failure belongs to the backend / build variant rather than to the request, so
    /// another variant of the same engine may succeed (`pack::open_engine` falls through). GPU
    /// variants also fall through on out-of-memory, load and runtime failures of their backend.
    pub fn is_backend_failure(&self, gpu_variant: bool) -> bool {
        self.status == abi::SIGIL_E_BACKEND_UNAVAILABLE
            || (gpu_variant
                && matches!(
                    self.status,
                    abi::SIGIL_E_OUT_OF_MEMORY | abi::SIGIL_E_LOAD_FAILED | abi::SIGIL_E_RUNTIME
                ))
    }

    pub fn is_cancelled(&self) -> bool {
        self.status == abi::SIGIL_E_CANCELLED
    }
}

impl std::fmt::Display for EngineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl From<EngineError> for String {
    fn from(e: EngineError) -> String {
        e.message
    }
}

// ---------------------------------------------------------------------------------------------
// Function table
// ---------------------------------------------------------------------------------------------

macro_rules! vtable {
    ($($field:ident : $ty:ty = $sym:literal),* $(,)?) => {
        /// Every ABI v1 export, resolved.
        #[derive(Clone, Copy)]
        pub(crate) struct Vtable {
            $(pub(crate) $field: $ty,)*
        }

        impl Vtable {
            /// Export names in declaration order (must equal [`abi::SYMBOLS`]).
            #[cfg_attr(not(test), allow(dead_code))]
            pub(crate) const NAMES: [&'static str; [$($sym),*].len()] = [$($sym),*];

            /// Resolve every export of `lib`; `Err` lists the missing names.
            ///
            /// # Safety
            /// The exports must have the ABI v1 signatures (the ABI version was checked first).
            unsafe fn resolve(lib: &libloading::Library) -> Result<Vtable, Vec<&'static str>> {
                let mut missing = Vec::new();
                $(
                    // SAFETY: see the function contract; the pointer is copied out and used only
                    // while the library stays loaded (the Vtable lives next to it).
                    let $field: Option<$ty> = unsafe { lib.get::<$ty>($sym.as_bytes()) }.ok().map(|s| *s);
                    if $field.is_none() {
                        missing.push($sym);
                    }
                )*
                match ($($field,)*) {
                    ($(Some($field),)*) => Ok(Vtable { $($field,)* }),
                    _ => Err(missing),
                }
            }
        }
    };
}

vtable! {
    abi_version: abi::AbiVersionFn = "sigil_engine_abi_version",
    get_info: abi::GetInfoFn = "sigil_engine_get_info",
    last_error: abi::LastErrorFn = "sigil_engine_last_error",
    set_log_callback: abi::SetLogCallbackFn = "sigil_engine_set_log_callback",
    create: abi::CreateFn = "sigil_engine_create",
    destroy: abi::DestroyFn = "sigil_engine_destroy",
    cancel: abi::CancelFn = "sigil_engine_cancel",
    query: abi::QueryFn = "sigil_engine_query",
    process: abi::ProcessFn = "sigil_engine_process",
    render: abi::RenderFn = "sigil_engine_render",
    stream_open: abi::StreamOpenFn = "sigil_stream_open",
    stream_push: abi::StreamPushFn = "sigil_stream_push",
    stream_finish: abi::StreamFinishFn = "sigil_stream_finish",
    stream_close: abi::StreamCloseFn = "sigil_stream_close",
    free_string: abi::FreeStringFn = "sigil_free_string",
    free_audio: abi::FreeAudioFn = "sigil_free_audio",
}

// ---------------------------------------------------------------------------------------------
// Log bridge
// ---------------------------------------------------------------------------------------------

/// A NUL-terminated engine name that lives for the rest of the process (the log callback's
/// `user_data`). One allocation per distinct engine name.
fn log_tag(name: &str) -> *mut c_void {
    static TAGS: OnceLock<Mutex<HashMap<String, &'static CStr>>> = OnceLock::new();
    let mut tags = lock(TAGS.get_or_init(|| Mutex::new(HashMap::new())));
    let tag: &'static CStr = tags.entry(name.to_string()).or_insert_with(|| {
        let owned = CString::new(name.replace('\0', "")).unwrap_or_default();
        Box::leak(owned.into_boxed_c_str())
    });
    tag.as_ptr().cast_mut().cast()
}

/// `sigil_log_fn` target: engine log lines go to `tracing` (`sigil::engine`), so WARN/ERROR reach
/// the file log and the in-app system log. Never unwinds into the engine.
unsafe extern "C" fn log_bridge(user_data: *mut c_void, level: i32, message: *const c_char) {
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        if message.is_null() {
            return;
        }
        // SAFETY: `user_data` is a leaked tag from `log_tag` (or NULL); `message` is a
        // NUL-terminated string valid for the duration of this call (header contract).
        let engine = if user_data.is_null() {
            "engine".into()
        } else {
            unsafe { CStr::from_ptr(user_data as *const c_char) }.to_string_lossy()
        };
        let text = unsafe { CStr::from_ptr(message) }.to_string_lossy();
        let text = truncate(text.trim_end(), MAX_MESSAGE_CHARS);
        match level {
            abi::SIGIL_LOG_TRACE => tracing::trace!(target: "sigil::engine", "[{engine}] {text}"),
            abi::SIGIL_LOG_DEBUG => tracing::debug!(target: "sigil::engine", "[{engine}] {text}"),
            abi::SIGIL_LOG_INFO => tracing::info!(target: "sigil::engine", "[{engine}] {text}"),
            abi::SIGIL_LOG_WARN => tracing::warn!(target: "sigil::engine", "[{engine}] {text}"),
            _ => tracing::error!(target: "sigil::engine", "[{engine}] {text}"),
        }
    }));
}

// ---------------------------------------------------------------------------------------------
// Library
// ---------------------------------------------------------------------------------------------

/// One loaded sigil_engine DLL (ABI checked, all [`abi::SYMBOLS`] resolved). Cheap to clone.
#[derive(Clone)]
pub struct EngineLibrary {
    inner: Arc<EngineLibraryInner>,
}

struct EngineLibraryInner {
    info: EngineInfo,
    path: PathBuf,
    vt: Vtable,
    /// Declared last so it is dropped (FreeLibrary) after everything that uses the vtable.
    _lib: Option<libloading::Library>,
}

impl std::fmt::Debug for EngineLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineLibrary")
            .field("name", &self.inner.info.name)
            .field("variant", &self.inner.info.variant)
            .field("path", &self.inner.path)
            .finish()
    }
}

/// Loaded libraries by normalized path (weak: unloaded when unused) and the GPU libraries pinned
/// for the process lifetime. The lock also serializes loads.
struct Registry {
    loaded: HashMap<String, Weak<EngineLibraryInner>>,
    pinned: Vec<EngineLibrary>,
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(|| {
        Mutex::new(Registry {
            loaded: HashMap::new(),
            pinned: Vec::new(),
        })
    })
}

fn registry_key(path: &Path) -> String {
    let p = std::fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    p.to_string_lossy().to_lowercase()
}

/// Copy a mandatory string field of `sigil_engine_info`.
///
/// # Safety
/// `ptr` is NULL or a NUL-terminated string that stays valid while the DLL is loaded.
unsafe fn info_string(ptr: *const c_char, field: &str) -> Result<String, String> {
    if ptr.is_null() {
        return Err(format!("sigil_engine_info.{field} is NULL"));
    }
    // SAFETY: see the function contract.
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map(|s| truncate(s, 512))
        .map_err(|_| format!("sigil_engine_info.{field} is not UTF-8"))
}

/// Validate and copy `sigil_engine_get_info()`.
///
/// # Safety
/// `p` is NULL or points to a static `sigil_engine_info` of at least `struct_size` bytes.
pub(crate) unsafe fn read_info(p: *const SigilEngineInfo) -> Result<EngineInfo, String> {
    if p.is_null() {
        return Err("sigil_engine_get_info returned NULL".into());
    }
    // SAFETY: `struct_size` is the first field; read before trusting the rest of the layout.
    let size = unsafe { std::ptr::read_unaligned(p.cast::<u32>()) };
    if size < abi::SIGIL_ENGINE_INFO_V1_SIZE {
        return Err(format!(
            "sigil_engine_info is {size} bytes, but ABI v1 needs at least {}",
            abi::SIGIL_ENGINE_INFO_V1_SIZE
        ));
    }
    // SAFETY: at least the v1 layout is present (checked above).
    let raw = unsafe { std::ptr::read_unaligned(p) };
    let info = EngineInfo {
        abi_version: raw.abi_version,
        name: unsafe { info_string(raw.name, "name") }?,
        version: unsafe { info_string(raw.version, "version") }?,
        variant: unsafe { info_string(raw.variant, "variant") }?,
        kind: unsafe { info_string(raw.kind, "kind") }?,
        build_info: unsafe { info_string(raw.build_info, "build_info") }?,
        capabilities: raw.capabilities,
    };
    if info.abi_version != abi::SIGIL_ENGINE_ABI_VERSION {
        return Err(format!(
            "sigil_engine_info reports ABI v{}, but sigil_engine_abi_version() returned v{}",
            info.abi_version,
            abi::SIGIL_ENGINE_ABI_VERSION
        ));
    }
    if !valid_engine_name(&info.name) {
        return Err(format!("invalid engine name {:?} (expected lower-case letters, digits, '-' or '_')", info.name));
    }
    Ok(info)
}

/// Engine ids: `[a-z0-9][a-z0-9_-]*`, at most 64 characters.
pub(crate) fn valid_engine_name(name: &str) -> bool {
    !name.is_empty()
        && name.len() <= 64
        && name.starts_with(|c: char| c.is_ascii_lowercase() || c.is_ascii_digit())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_')
}

/// Message for a DLL that implements another ABI version.
pub(crate) fn abi_mismatch_message(file: &str, found: u32) -> String {
    let expected = abi::SIGIL_ENGINE_ABI_VERSION;
    let advice = if found > expected {
        "it was built for a newer Sigil; update Sigil or install the pack that matches this version"
    } else {
        "it was built for an older Sigil; reinstall the runtime pack"
    };
    format!("{file} implements sigil_engine ABI v{found}, but this Sigil supports v{expected}: {advice}")
}

impl EngineLibrary {
    /// Load `dir/file_name` with [`load_library_in_dir`], refuse an ABI mismatch, resolve every
    /// v1 symbol and install the log bridge (engine logs → `tracing`). See the module docs for the
    /// full sequence (reuse, hash check, ABI first, exports, info, manifest match).
    pub fn load(dir: &Path, file_name: &str) -> Result<EngineLibrary, String> {
        Self::load_checked(dir, file_name, &VariantSupport::detect_cpu(), false)
    }

    /// [`Self::load`] against the given CPU features, optionally requiring a pack manifest (an
    /// installed pack: no unverified load). A build whose instruction sets `cpu` lacks is refused
    /// by its manifest's variant before the DLL is loaded, and by the DLL's own reported variant
    /// right after `sigil_engine_get_info` (a folder without a manifest), before any engine call:
    /// such a DLL would not fail, it would kill the process with an illegal instruction.
    pub(crate) fn load_checked(
        dir: &Path,
        file_name: &str,
        cpu: &VariantSupport,
        require_manifest: bool,
    ) -> Result<EngineLibrary, String> {
        let path = dir.join(file_name);
        if !path.is_file() {
            return Err(format!("{} does not exist", path.display()));
        }
        let key = registry_key(&path);
        let mut reg = lock(registry());
        if let Some(existing) = reg.loaded.get(&key).and_then(Weak::upgrade) {
            return Ok(EngineLibrary { inner: existing });
        }
        let manifest = pack::verify_library(dir, file_name)?;
        match &manifest {
            None if require_manifest => {
                return Err(format!(
                    "{} has no {}, so {file_name} cannot be verified; reinstall the pack",
                    dir.display(),
                    pack::PACK_MANIFEST
                ))
            }
            Some(m) => cpu.check_build(&m.variant).map_err(|e| format!("{}: {e}", path.display()))?,
            None => {}
        }
        let lib = load_library_in_dir(dir, file_name)?;
        // The ABI version is checked before anything else is resolved or called.
        // SAFETY: `sigil_engine_abi_version` takes nothing and returns a u32 in every ABI version.
        let abi_fn: abi::AbiVersionFn = unsafe { lib.get::<abi::AbiVersionFn>(b"sigil_engine_abi_version") }
            .map(|s| *s)
            .map_err(|_| format!("{file_name} is not a Sigil engine (it does not export sigil_engine_abi_version)"))?;
        // SAFETY: as above.
        let found = unsafe { abi_fn() };
        if found != abi::SIGIL_ENGINE_ABI_VERSION {
            return Err(abi_mismatch_message(file_name, found));
        }
        // SAFETY: the ABI version matches, so the exports have the v1 signatures.
        let vt = unsafe { Vtable::resolve(&lib) }
            .map_err(|missing| format!("{file_name} is missing sigil_engine exports: {}", missing.join(", ")))?;
        let engine = Self::from_parts(vt, path.clone(), Some(lib))?;
        if let Some(m) = &manifest {
            m.check_matches(engine.info())
                .map_err(|e| format!("{}: {e}", path.display()))?;
        }
        cpu.check_build(&engine.info().variant)
            .map_err(|e| format!("{}: {e}", path.display()))?;
        tracing::info!(
            target: "sigil::runtime",
            "loaded engine {} {} ({}; {}) from {}{}",
            engine.info().name,
            engine.info().version,
            engine.info().variant,
            engine.info().capability_names().join("+"),
            path.display(),
            if manifest.is_some() { " (hash-verified)" } else { " (no pack manifest; not hash-verified)" }
        );
        reg.loaded.retain(|_, w| w.strong_count() > 0);
        reg.loaded.insert(key, Arc::downgrade(&engine.inner));
        if engine.info().has(abi::SIGIL_CAP_GPU) {
            reg.pinned.push(engine.clone());
        }
        Ok(engine)
    }

    /// Wrap a resolved function table: checks the ABI version and `sigil_engine_info`, then
    /// installs the log bridge. `lib` is `None` for function tables implemented in-process (tests).
    pub(crate) fn from_parts(vt: Vtable, path: PathBuf, lib: Option<libloading::Library>) -> Result<EngineLibrary, String> {
        let file = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| path.display().to_string());
        // SAFETY: the vtable has the ABI v1 signatures (resolved after the version check, or
        // built from Rust functions of these types).
        let found = unsafe { (vt.abi_version)() };
        if found != abi::SIGIL_ENGINE_ABI_VERSION {
            return Err(abi_mismatch_message(&file, found));
        }
        // SAFETY: `get_info` returns NULL or a static struct (header contract), validated here.
        let info = unsafe { read_info((vt.get_info)()) }.map_err(|e| format!("{file}: {e}"))?;
        // SAFETY: the bridge is a plain function that lives as long as the process; the tag is a
        // leaked C string.
        unsafe { (vt.set_log_callback)(Some(log_bridge), log_tag(&info.name)) };
        Ok(EngineLibrary {
            inner: Arc::new(EngineLibraryInner { info, path, vt, _lib: lib }),
        })
    }

    pub fn info(&self) -> &EngineInfo {
        &self.inner.info
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    pub(crate) fn vt(&self) -> &Vtable {
        &self.inner.vt
    }

    /// The engine's `sigil_engine_last_error()` on this thread, bounded.
    fn last_error(&self) -> String {
        // SAFETY: returns NULL or a NUL-terminated thread-local string valid until the next call
        // on this thread; copied immediately.
        let p = unsafe { (self.vt().last_error)() };
        if p.is_null() {
            return String::new();
        }
        let s = unsafe { CStr::from_ptr(p) }.to_string_lossy();
        truncate(s.trim(), MAX_MESSAGE_CHARS)
    }

    /// Error for a failed call: must be built right after the call, before any other call on this
    /// thread (the last error is thread-local and reset by the next call).
    fn failure(&self, op: &str, status: SigilStatus) -> EngineError {
        let detail = self.last_error();
        let name = &self.inner.info.name;
        let code = abi::status_name(status);
        EngineError {
            status,
            message: if detail.is_empty() {
                format!("{name} {op} failed ({code})")
            } else {
                format!("{name} {op} failed ({code}): {detail}")
            },
        }
    }

    /// `sigil_engine_create(config_json)`.
    #[allow(dead_code)] // Frozen API (P1-S0b); engine wrappers use it.
    pub fn create(&self, config_json: &str) -> Result<EngineInstance, String> {
        self.create_engine(config_json).map_err(String::from)
    }

    /// [`Self::create`] with the status code of a failure (for variant fallback).
    pub fn create_engine(&self, config_json: &str) -> Result<EngineInstance, EngineError> {
        let config = json_arg("config", config_json)?;
        let mut raw: *mut SigilEngine = std::ptr::null_mut();
        // SAFETY: valid C string and out pointer for the duration of the call.
        let status = unsafe { (self.vt().create)(config.as_ptr(), &mut raw) };
        if status != abi::SIGIL_OK {
            let err = self.failure("create", status);
            if !raw.is_null() {
                // A misbehaving engine handed out an instance anyway: release it.
                // SAFETY: the pointer came from this engine's create.
                unsafe { (self.vt().destroy)(raw) };
            }
            return Err(err);
        }
        if raw.is_null() {
            return Err(EngineError::host(
                abi::SIGIL_E_RUNTIME,
                format!("{} create returned success without an engine", self.info().name),
            ));
        }
        Ok(EngineInstance {
            handle: Arc::new(EngineHandle {
                raw,
                library: self.clone(),
                calls: Mutex::new(()),
            }),
            _not_sync: PhantomData,
        })
    }
}

/// A JSON argument as a C string; blank means `{}` (the header's NULL).
fn json_arg(what: &str, json: &str) -> Result<CString, EngineError> {
    let json = if json.trim().is_empty() { "{}" } else { json };
    CString::new(json).map_err(|_| EngineError::host(abi::SIGIL_E_INVALID_ARGUMENT, format!("the {what} JSON contains a NUL byte")))
}

/// The C view of `audio` (validated: channels and rate non-zero, whole frames).
fn audio_view(audio: &AudioView<'_>) -> Result<SigilAudioView, EngineError> {
    let invalid = |m: String| EngineError::host(abi::SIGIL_E_INVALID_ARGUMENT, m);
    if audio.channels == 0 {
        return Err(invalid("audio has 0 channels".into()));
    }
    if audio.sample_rate == 0 {
        return Err(invalid("audio has a sample rate of 0".into()));
    }
    let channels = usize::from(audio.channels);
    if !audio.samples.len().is_multiple_of(channels) {
        return Err(invalid(format!(
            "{} samples are not a whole number of {}-channel frames",
            audio.samples.len(),
            audio.channels
        )));
    }
    Ok(SigilAudioView {
        samples: if audio.samples.is_empty() {
            std::ptr::null()
        } else {
            audio.samples.as_ptr()
        },
        frames: (audio.samples.len() / channels) as u64,
        sample_rate: audio.sample_rate,
        channels: u32::from(audio.channels),
    })
}

// ---------------------------------------------------------------------------------------------
// Instance, stream, canceller
// ---------------------------------------------------------------------------------------------

/// Owner of one `sigil_engine*`: destroyed when the instance and all of its streams are gone.
struct EngineHandle {
    raw: *mut SigilEngine,
    library: EngineLibrary,
    /// Serializes every call on this engine and its streams (header: calls on one instance must be
    /// serialized). `sigil_engine_cancel` does not take it.
    calls: Mutex<()>,
}

// SAFETY: the engine pointer is only used under `calls` (or for `sigil_engine_cancel`, which the
// ABI allows from any thread while the instance is alive; the Arc/Weak pair guarantees that).
unsafe impl Send for EngineHandle {}
unsafe impl Sync for EngineHandle {}

impl Drop for EngineHandle {
    fn drop(&mut self) {
        // SAFETY: last reference: every stream was closed first (they hold the handle) and no
        // call can be running.
        unsafe { (self.library.vt().destroy)(self.raw) };
    }
}

impl EngineHandle {
    /// Run a call that may return a JSON string, under the call lock; the string is copied and
    /// released. `Ok(None)` when the engine returned no string.
    fn call_json(
        &self,
        op: &str,
        f: impl FnOnce(*mut SigilEngine, *mut *mut c_char) -> SigilStatus,
    ) -> Result<Option<String>, EngineError> {
        let _serial = lock(&self.calls);
        let mut out: *mut c_char = std::ptr::null_mut();
        let status = f(self.raw, &mut out);
        // The error must be read before any other call (freeing the string is a call).
        let err = (status != abi::SIGIL_OK).then(|| self.library.failure(op, status));
        let text = take_string(&self.library, out);
        match err {
            Some(e) => Err(e),
            None => Ok(text),
        }
    }
}

/// Copy and release an engine-owned string (`None` for NULL).
fn take_string(library: &EngineLibrary, p: *mut c_char) -> Option<String> {
    if p.is_null() {
        return None;
    }
    // SAFETY: a NUL-terminated string allocated by this engine; freed with its own free function.
    let text = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { (library.vt().free_string)(p) };
    Some(text)
}

/// Copy an engine audio buffer (validated) into an [`AudioBuffer`].
fn copy_audio(engine: &str, buf: &SigilAudioBuffer) -> Result<AudioBuffer, EngineError> {
    let bad = |m: String| EngineError::host(abi::SIGIL_E_RUNTIME, format!("{engine} render returned {m}"));
    if buf.frames == 0 {
        return Ok(AudioBuffer {
            samples: Vec::new(),
            sample_rate: buf.sample_rate,
            channels: u16::try_from(buf.channels.min(MAX_CHANNELS)).unwrap_or(0),
        });
    }
    if buf.samples.is_null() {
        return Err(bad(format!("{} frames without samples", buf.frames)));
    }
    if buf.channels == 0 || buf.channels > MAX_CHANNELS {
        return Err(bad(format!("{} channels", buf.channels)));
    }
    if buf.sample_rate == 0 || buf.sample_rate > 1_536_000 {
        return Err(bad(format!("a sample rate of {}", buf.sample_rate)));
    }
    let total = buf
        .frames
        .checked_mul(u64::from(buf.channels))
        .filter(|&n| n <= MAX_OUTPUT_SAMPLES)
        .ok_or_else(|| bad(format!("an oversized buffer ({} frames x {} channels)", buf.frames, buf.channels)))?;
    // SAFETY: the engine owns `total` initialized f32 values at `samples` until sigil_free_audio.
    let samples = unsafe { std::slice::from_raw_parts(buf.samples, total as usize) }.to_vec();
    Ok(AudioBuffer {
        samples,
        sample_rate: buf.sample_rate,
        channels: buf.channels as u16,
    })
}

/// One engine instance. `Send`, not `Sync`: calls are serialized by `&mut self` (and by the
/// per-instance lock shared with its streams).
pub struct EngineInstance {
    handle: Arc<EngineHandle>,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

impl std::fmt::Debug for EngineInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineInstance").field("library", &self.handle.library).finish()
    }
}

impl EngineInstance {
    #[allow(dead_code)] // Frozen API (P1-S0b).
    pub fn library(&self) -> &EngineLibrary {
        &self.handle.library
    }

    /// `sigil_engine_query` → response JSON.
    pub fn query(&mut self, request_json: &str) -> Result<String, String> {
        self.query_engine(request_json).map_err(String::from)
    }

    /// [`Self::query`] with the status code of a failure.
    pub fn query_engine(&mut self, request_json: &str) -> Result<String, EngineError> {
        let request = json_arg("request", request_json)?;
        let vt = *self.handle.library.vt();
        self.handle
            .call_json("query", |raw, out| unsafe { (vt.query)(raw, request.as_ptr(), out) })
            .map(|s| s.unwrap_or_else(|| "{}".into()))
    }

    /// `sigil_engine_process` → result JSON (ASR: `{"text","language","segments"}`).
    pub fn process(&mut self, audio: AudioView<'_>, options_json: &str) -> Result<String, String> {
        self.process_audio(audio, options_json).map_err(String::from)
    }

    /// [`Self::process`] with the status code of a failure.
    pub fn process_audio(&mut self, audio: AudioView<'_>, options_json: &str) -> Result<String, EngineError> {
        let view = audio_view(&audio)?;
        let options = json_arg("options", options_json)?;
        let vt = *self.handle.library.vt();
        self.handle
            .call_json("process", |raw, out| unsafe { (vt.process)(raw, &view, options.as_ptr(), out) })
            .map(|s| s.unwrap_or_else(|| "{}".into()))
    }

    /// `sigil_engine_render` (input `None` for generators) → owned audio.
    pub fn render(&mut self, input: Option<AudioView<'_>>, options_json: &str) -> Result<AudioBuffer, String> {
        self.render_audio(input, options_json).map_err(String::from)
    }

    /// [`Self::render`] with the status code of a failure.
    pub fn render_audio(&mut self, input: Option<AudioView<'_>>, options_json: &str) -> Result<AudioBuffer, EngineError> {
        let view = input.as_ref().map(audio_view).transpose()?;
        let options = json_arg("options", options_json)?;
        let library = &self.handle.library;
        let vt = *library.vt();
        let _serial = lock(&self.handle.calls);
        let mut buf = SigilAudioBuffer::default();
        let view_ptr = view.as_ref().map_or(std::ptr::null(), |v| v as *const SigilAudioView);
        // SAFETY: valid (or NULL) view, C string and out buffer for the duration of the call.
        let status = unsafe { (vt.render)(self.handle.raw, view_ptr, options.as_ptr(), &mut buf) };
        if status != abi::SIGIL_OK {
            let err = library.failure("render", status);
            if !buf.samples.is_null() {
                // SAFETY: the buffer was filled by this engine; its own free releases it.
                unsafe { (vt.free_audio)(&mut buf) };
            }
            return Err(err);
        }
        let copied = copy_audio(&library.info().name, &buf);
        if !buf.samples.is_null() {
            // SAFETY: as above.
            unsafe { (vt.free_audio)(&mut buf) };
        }
        copied
    }

    /// `sigil_stream_open`. The stream keeps this instance's engine and library alive.
    pub fn open_stream(&mut self, options_json: &str) -> Result<EngineStream, String> {
        self.open_engine_stream(options_json).map_err(String::from)
    }

    /// [`Self::open_stream`] with the status code of a failure.
    pub fn open_engine_stream(&mut self, options_json: &str) -> Result<EngineStream, EngineError> {
        let options = json_arg("options", options_json)?;
        let library = &self.handle.library;
        let vt = *library.vt();
        let _serial = lock(&self.handle.calls);
        let mut raw: *mut SigilStream = std::ptr::null_mut();
        // SAFETY: valid C string and out pointer for the duration of the call.
        let status = unsafe { (vt.stream_open)(self.handle.raw, options.as_ptr(), &mut raw) };
        if status != abi::SIGIL_OK {
            let err = library.failure("stream open", status);
            if !raw.is_null() {
                // SAFETY: the pointer came from this engine's stream_open.
                unsafe { (vt.stream_close)(raw) };
            }
            return Err(err);
        }
        if raw.is_null() {
            return Err(EngineError::host(
                abi::SIGIL_E_RUNTIME,
                format!("{} stream open returned success without a stream", library.info().name),
            ));
        }
        Ok(EngineStream {
            raw,
            handle: self.handle.clone(),
            _not_sync: PhantomData,
        })
    }

    /// Thread-safe handle calling `sigil_engine_cancel` on this instance.
    pub fn canceller(&self) -> EngineCanceller {
        EngineCanceller {
            handle: Arc::downgrade(&self.handle),
        }
    }
}

/// Streaming session of one instance. `Send`, not `Sync`.
pub struct EngineStream {
    raw: *mut SigilStream,
    handle: Arc<EngineHandle>,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

// SAFETY: the stream pointer is only used under the owning engine's call lock.
unsafe impl Send for EngineStream {}

impl EngineStream {
    /// `sigil_stream_push` → partial result JSON, if the engine produced one.
    pub fn push(&mut self, audio: AudioView<'_>) -> Result<Option<String>, String> {
        self.push_audio(audio).map_err(String::from)
    }

    /// [`Self::push`] with the status code of a failure.
    pub fn push_audio(&mut self, audio: AudioView<'_>) -> Result<Option<String>, EngineError> {
        let view = audio_view(&audio)?;
        let vt = *self.handle.library.vt();
        let stream = self.raw;
        self.handle
            .call_json("stream push", |_, out| unsafe { (vt.stream_push)(stream, &view, out) })
    }

    /// `sigil_stream_finish` → final result JSON; the stream can be pushed again afterwards.
    pub fn finish(&mut self) -> Result<String, String> {
        self.finish_stream().map_err(String::from)
    }

    /// [`Self::finish`] with the status code of a failure.
    pub fn finish_stream(&mut self) -> Result<String, EngineError> {
        let vt = *self.handle.library.vt();
        let stream = self.raw;
        self.handle
            .call_json("stream finish", |_, out| unsafe { (vt.stream_finish)(stream, out) })
            .map(|s| s.unwrap_or_else(|| "{}".into()))
    }

    /// Cancel handle of the instance this stream belongs to.
    #[allow(dead_code)] // For streaming engine wrappers (P2 speech plugin).
    pub fn canceller(&self) -> EngineCanceller {
        EngineCanceller {
            handle: Arc::downgrade(&self.handle),
        }
    }
}

impl Drop for EngineStream {
    fn drop(&mut self) {
        let vt = *self.handle.library.vt();
        let _serial = lock(&self.handle.calls);
        // SAFETY: the stream belongs to this (still alive) engine and is closed exactly once.
        unsafe { (vt.stream_close)(self.raw) };
    }
}

/// `sigil_engine_cancel` from any thread (`Send + Sync + Clone`); a no-op after the instance is gone.
#[derive(Clone)]
pub struct EngineCanceller {
    handle: Weak<EngineHandle>,
}

impl EngineCanceller {
    pub fn cancel(&self) {
        if let Some(h) = self.handle.upgrade() {
            // SAFETY: the instance is alive (strong reference held for the call); cancel is
            // thread-safe by contract and does not take the call lock.
            unsafe { (h.library.vt().cancel)(h.raw) };
        }
    }

    /// Whether the instance still exists.
    pub fn is_alive(&self) -> bool {
        self.handle.strong_count() > 0
    }
}

impl std::fmt::Debug for EngineCanceller {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineCanceller").field("alive", &self.is_alive()).finish()
    }
}
