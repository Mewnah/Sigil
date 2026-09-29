//! Loading the sherpa-onnx runtime pack (`runtime/sherpa-onnx`, CPU only) and the FFI plumbing the
//! engine wrappers share.
//!
//! # Load order and isolation
//! 1. Refuse to load when a module named `onnxruntime.dll` from another folder is already in the
//!    process (Windows binds a DLL's imports to an already loaded module of the same base name, so
//!    sherpa would silently run against a foreign ONNX Runtime, e.g. `System32\onnxruntime.dll`
//!    1.17 that Windows ML ships).
//! 2. `onnxruntime.dll` of the pack by full path (`native_runtime::load_library_in_dir`, i.e.
//!    `LoadLibraryExW(LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`), then
//!    check it exposes the ORT C API level sherpa-onnx was built against.
//! 3. `sherpa-onnx-c-api.dll` the same way; its `onnxruntime.dll` import binds to step 2's module.
//!    Every allow-listed symbol must resolve ([`bindings::SherpaOnnxCApi::from_library`]).
//! 4. `SherpaOnnxGetVersionStr()` must equal [`EXPECTED_VERSION`] (the config structs in
//!    `bindings.rs` grow between releases, so any other version is refused, never guessed).
//! 5. The `onnxruntime.dll` module the process now resolves must be the pack's file.
//!
//! Sigil's own ONNX Runtime is linked statically into Sigil.exe and exports nothing, so the two
//! runtimes never share symbols; sherpa's copy stays CPU-only (the wrapper always passes
//! `provider = "cpu"`, and the pack ships no execution-provider DLLs).
//!
//! A loaded runtime is cached for the life of the process and **never unloaded** (ONNX Runtime
//! owns thread pools and thread-local state; unloading it under live threads is unsound). Failed
//! loads are not cached, so installing the pack later works without a restart. Loading a
//! different directory after one succeeded returns the loaded runtime (a pack update applies on
//! the next start).

use std::ffi::{c_char, c_void, CStr, CString};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};

use serde::Serialize;

use super::bindings::SherpaOnnxCApi;
use crate::services::native_runtime::load_library_in_dir;

/// Catalog id of the runtime pack (`model-catalog/runtime-sherpa.json`).
pub const RUNTIME_ID: &str = "runtime/sherpa-onnx";
/// The only sherpa-onnx version the vendored bindings describe.
pub const EXPECTED_VERSION: &str = "1.13.8";
/// ONNX Runtime C API level the pinned sherpa-onnx build needs (it was built against ORT 1.28.2).
pub const ORT_API_LEVEL: u32 = 28;
/// ONNX Runtime DLL of the pack (loaded first, by full path).
pub const ORT_DLL: &str = "onnxruntime.dll";
/// sherpa-onnx C API DLL of the pack.
pub const C_API_DLL: &str = "sherpa-onnx-c-api.dll";
/// Development override: load the runtime from this folder instead of the installed pack.
pub const RUNTIME_DIR_ENV: &str = "SIGIL_SHERPA_RUNTIME_DIR";
/// Default ONNX Runtime threads per sherpa session (VRChat is often CPU-bound).
pub const DEFAULT_THREADS: u32 = 2;
/// Default for streaming recognizers, which decode every chunk (see [`threads_or`]).
pub const STREAMING_DEFAULT_THREADS: u32 = 1;
/// Upper bound on threads per sherpa session.
pub const MAX_THREADS: u32 = 4;
/// Execution provider passed to every sherpa config: the pack is CPU-only by design.
pub(crate) const CPU_PROVIDER: &CStr = c"cpu";

/// What was loaded, for probes, logs and the Models tab.
#[derive(Clone, Debug, Serialize, PartialEq, Eq)]
pub struct RuntimeInfo {
    /// Folder both DLLs were loaded from.
    pub dir: String,
    /// `SherpaOnnxGetVersionStr()`.
    pub version: String,
    /// `SherpaOnnxGetGitSha1()`.
    pub git_sha1: String,
    /// `SherpaOnnxGetGitDate()`.
    pub git_date: String,
    /// `OrtGetApiBase()->GetVersionString()` of the pack's `onnxruntime.dll`.
    pub onnxruntime_version: String,
    /// Full path of the `onnxruntime.dll` module the process resolves after loading (Windows).
    pub onnxruntime_module: Option<String>,
}

/// A loaded sherpa-onnx runtime. Obtain it with [`SherpaRuntime::load`] or
/// [`SherpaRuntime::installed`]; share the `Arc` with every engine created from it.
pub struct SherpaRuntime {
    api: SherpaOnnxCApi,
    // Declared after `api` so a runtime dropped during a failed load releases the C API DLL
    // before the ONNX Runtime it imports. A cached runtime is never dropped.
    _ort: libloading::Library,
    info: RuntimeInfo,
}

// SAFETY: `SherpaOnnxCApi` holds a `libloading::Library` (Send + Sync) and plain `extern "C"` fn
// pointers; the functions themselves are called only through the engine wrappers, which serialize
// calls per native object (see `CallLock`).
unsafe impl Send for SherpaRuntime {}
unsafe impl Sync for SherpaRuntime {}

impl std::fmt::Debug for SherpaRuntime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SherpaRuntime")
            .field("info", &self.info)
            .finish()
    }
}

fn cache() -> MutexGuard<'static, Option<Arc<SherpaRuntime>>> {
    static CACHE: Mutex<Option<Arc<SherpaRuntime>>> = Mutex::new(None);
    CACHE.lock().unwrap_or_else(|e| e.into_inner())
}

impl SherpaRuntime {
    /// Load (once per process) the runtime from `dir`, which must hold [`ORT_DLL`] and
    /// [`C_API_DLL`]. A missing folder, DLL, symbol or a version mismatch is an `Err`.
    pub fn load(dir: &Path) -> Result<Arc<SherpaRuntime>, String> {
        let mut cached = cache();
        if let Some(rt) = cached.as_ref() {
            if !same_dir(Path::new(&rt.info.dir), dir) {
                tracing::info!(
                    target: "sigil::sherpa",
                    "sherpa-onnx is already loaded from {}; {} is used after a restart",
                    rt.info.dir,
                    dir.display()
                );
            }
            return Ok(rt.clone());
        }
        let rt = Arc::new(load_uncached(dir)?);
        tracing::info!(
            target: "sigil::sherpa",
            "loaded sherpa-onnx {} ({}) with ONNX Runtime {} from {}",
            rt.info.version,
            rt.info.git_sha1,
            rt.info.onnxruntime_version,
            rt.info.dir
        );
        *cached = Some(rt.clone());
        Ok(rt)
    }

    /// Load from [`RUNTIME_DIR_ENV`] when set, else from the installed `runtime/sherpa-onnx` pack.
    pub fn installed() -> Result<Arc<SherpaRuntime>, String> {
        if let Some(rt) = Self::loaded() {
            return Ok(rt);
        }
        let dir = runtime_dir().ok_or_else(|| {
            format!(
                "the sherpa-onnx runtime pack ({RUNTIME_ID}) is not installed; install it from Models & Runtimes \
                 or run `Sigil.exe --headless fetch {RUNTIME_ID}`"
            )
        })?;
        Self::load(&dir)
    }

    /// The runtime if one was loaded in this process.
    pub fn loaded() -> Option<Arc<SherpaRuntime>> {
        cache().clone()
    }

    pub fn info(&self) -> &RuntimeInfo {
        &self.info
    }

    pub(crate) fn api(&self) -> &SherpaOnnxCApi {
        &self.api
    }
}

/// Folder of the runtime to use: [`RUNTIME_DIR_ENV`] if set, else the installed pack's folder
/// (`model_store::installed_path`, the only way consumers locate pack files).
pub fn runtime_dir() -> Option<PathBuf> {
    if let Some(dir) = std::env::var_os(RUNTIME_DIR_ENV).filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(dir));
    }
    crate::services::model_store::installed_path(RUNTIME_ID)
}

/// Install the runtime pack through the model store if it is missing (one download, verified),
/// then load it.
#[allow(dead_code)] // For the P2 `speech` plugin (zero-setup CPU tier).
pub async fn ensure_installed() -> Result<Arc<SherpaRuntime>, String> {
    if let Some(rt) = SherpaRuntime::loaded() {
        return Ok(rt);
    }
    if std::env::var_os(RUNTIME_DIR_ENV)
        .filter(|v| !v.is_empty())
        .is_none()
    {
        crate::services::model_store::ensure(&[RUNTIME_ID]).await?;
    }
    tokio::task::spawn_blocking(SherpaRuntime::installed)
        .await
        .map_err(|e| format!("sherpa-onnx load task failed: {e}"))?
}

fn same_dir(a: &Path, b: &Path) -> bool {
    let canon = |p: &Path| std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf());
    let (a, b) = (canon(a), canon(b));
    a.to_string_lossy()
        .eq_ignore_ascii_case(&b.to_string_lossy())
}

pub(super) fn load_uncached(dir: &Path) -> Result<SherpaRuntime, String> {
    for dll in [ORT_DLL, C_API_DLL] {
        if !dir.join(dll).is_file() {
            return Err(format!("the sherpa-onnx runtime is incomplete: {} is missing", dir.join(dll).display()));
        }
    }
    let expected_ort = dir.join(ORT_DLL);
    if let Some(existing) = loaded_module_path(ORT_DLL) {
        if !same_file(&existing, &expected_ort) {
            return Err(format!(
                "another ONNX Runtime ({}) is already loaded in this process; sherpa-onnx would bind to it \
                 instead of {}",
                existing.display(),
                expected_ort.display()
            ));
        }
    }

    let ort = load_library_in_dir(dir, ORT_DLL)?;
    let onnxruntime_version = check_ort_api(&ort, &expected_ort)?;
    let c_api = load_library_in_dir(dir, C_API_DLL)?;
    // SAFETY: resolving symbols only; the signatures come from the pinned header, and the version
    // check below refuses every other build before any config struct crosses the boundary.
    let api = unsafe { SherpaOnnxCApi::from_library(c_api) }
        .map_err(|e| format!("{} is not a sherpa-onnx {EXPECTED_VERSION} C API ({e})", dir.join(C_API_DLL).display()))?;
    // SAFETY: argument-less getters returning static strings owned by the library.
    let (version, git_sha1, git_date) = unsafe {
        (
            owned_str((api.SherpaOnnxGetVersionStr)()),
            owned_str((api.SherpaOnnxGetGitSha1)()),
            owned_str((api.SherpaOnnxGetGitDate)()),
        )
    };
    if version.trim() != EXPECTED_VERSION {
        return Err(format!(
            "{} is sherpa-onnx {version}, but this Sigil build needs {EXPECTED_VERSION}; reinstall {RUNTIME_ID}",
            dir.join(C_API_DLL).display()
        ));
    }
    let onnxruntime_module = loaded_module_path(ORT_DLL);
    if let Some(module) = &onnxruntime_module {
        if !same_file(module, &expected_ort) {
            return Err(format!(
                "sherpa-onnx resolved ONNX Runtime from {} instead of {}",
                module.display(),
                expected_ort.display()
            ));
        }
    }
    Ok(SherpaRuntime {
        api,
        _ort: ort,
        info: RuntimeInfo {
            dir: dir.display().to_string(),
            version,
            git_sha1,
            git_date,
            onnxruntime_version,
            onnxruntime_module: onnxruntime_module.map(|p| p.display().to_string()),
        },
    })
}

/// `OrtApiBase` from `onnxruntime_c_api.h` (two function pointers, stable since ORT 1.0).
#[repr(C)]
struct OrtApiBase {
    get_api: unsafe extern "C" fn(version: u32) -> *const c_void,
    get_version_string: unsafe extern "C" fn() -> *const c_char,
}

/// The pack's `onnxruntime.dll` must be an ONNX Runtime serving [`ORT_API_LEVEL`]; returns its
/// version string.
fn check_ort_api(ort: &libloading::Library, path: &Path) -> Result<String, String> {
    // SAFETY: `OrtGetApiBase` has this signature in every ORT release; the returned struct is
    // static and immutable inside the DLL.
    unsafe {
        let get_base: libloading::Symbol<unsafe extern "C" fn() -> *const OrtApiBase> = ort
            .get(b"OrtGetApiBase\0")
            .map_err(|e| format!("{} is not ONNX Runtime ({e})", path.display()))?;
        let base = get_base();
        if base.is_null() {
            return Err(format!("{} returned no OrtApiBase", path.display()));
        }
        let version = owned_str(((*base).get_version_string)());
        if ((*base).get_api)(ORT_API_LEVEL).is_null() {
            return Err(format!(
                "{} is ONNX Runtime {version}, which lacks C API level {ORT_API_LEVEL} that sherpa-onnx \
                 {EXPECTED_VERSION} needs",
                path.display()
            ));
        }
        Ok(version)
    }
}

/// Copy a C string owned by the library (null → empty).
///
/// # Safety
/// `p` must be null or point to a NUL-terminated string that stays valid for the call.
pub(crate) unsafe fn owned_str(p: *const c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        CStr::from_ptr(p).to_string_lossy().into_owned()
    }
}

/// Like [`owned_str`] but `None` for null or empty strings.
///
/// # Safety
/// As [`owned_str`].
pub(crate) unsafe fn opt_str(p: *const c_char) -> Option<String> {
    let s = owned_str(p);
    (!s.is_empty()).then_some(s)
}

fn same_file(a: &Path, b: &Path) -> bool {
    same_dir(a, b)
}

/// Full path of the loaded module named `name`, if any.
#[cfg(windows)]
pub(crate) fn loaded_module_path(name: &str) -> Option<PathBuf> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    use windows::core::PCWSTR;
    use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
    let wide: Vec<u16> = std::ffi::OsStr::new(name)
        .encode_wide()
        .chain(Some(0))
        .collect();
    // SAFETY: plain Win32 calls with a NUL-terminated name and a sized buffer; the module handle is
    // only used for the file name query right away.
    unsafe {
        let module = GetModuleHandleW(PCWSTR(wide.as_ptr())).ok()?;
        let mut buf = vec![0u16; 32_768];
        let n = GetModuleFileNameW(Some(module), &mut buf) as usize;
        if n == 0 || n >= buf.len() {
            return None;
        }
        Some(PathBuf::from(std::ffi::OsString::from_wide(&buf[..n])))
    }
}

#[cfg(not(windows))]
pub(crate) fn loaded_module_path(_name: &str) -> Option<PathBuf> {
    None
}

/// Threads for one sherpa session: `None` → [`DEFAULT_THREADS`], clamped to `1..=MAX_THREADS`.
pub(crate) fn threads(requested: Option<u32>) -> i32 {
    threads_or(requested, DEFAULT_THREADS)
}

/// Threads for one sherpa session with an engine-specific default, clamped to `1..=MAX_THREADS`.
///
/// sherpa-onnx's C API cannot turn ONNX Runtime's spin-wait off: after every run each extra
/// intra-op worker spins for roughly 25 ms (measured, ADR 0002). That is noise for one decode per
/// utterance, but engines that run every few hundred milliseconds (streaming ASR) default to one
/// thread, which has no worker pool and never spins.
pub(crate) fn threads_or(requested: Option<u32>, default: u32) -> i32 {
    requested.unwrap_or(default).clamp(1, MAX_THREADS) as i32
}

/// What the wrapper passes for a deliberate 0 in a float option whose C default is not 0.
///
/// The C API converts those options with `SHERPA_ONNX_OR(x, default)` (`x ? x : default`), so an
/// exact 0 means "unset" and becomes sherpa's default (0.667 for `noise_scale`, 1.5 for
/// `hotwords_score`, ...). This is small enough to act as 0 (no audible noise, no boost). Not for
/// options sherpa range-checks: TTS `silence_scale` has its own floor (`tts::MIN_SILENCE_SCALE`).
pub(crate) const C_FLOAT_ZERO: f32 = 1e-6;

/// A float for a C option that `SHERPA_ONNX_OR` defaults: non-finite → `default`, an exact 0 →
/// [`C_FLOAT_ZERO`], anything else unchanged.
pub(crate) fn c_float(v: f32, default: f32) -> f32 {
    if !v.is_finite() {
        default
    } else if v == 0.0 {
        C_FLOAT_ZERO
    } else {
        v
    }
}

// ---------------------------------------------------------------------------------------------
// Strings and paths for config structs
// ---------------------------------------------------------------------------------------------

/// Owns the C strings a config struct points to while the `SherpaOnnxCreate*` call runs (the
/// C API copies every string into its own config, so the arena may be dropped afterwards).
/// A `CString`'s buffer never moves, so pointers stay valid as the arena grows.
#[derive(Default)]
pub(crate) struct CStrArena {
    strings: Vec<CString>,
}

impl CStrArena {
    pub fn str(&mut self, s: &str) -> Result<*const c_char, String> {
        let c = CString::new(s).map_err(|_| format!("{s:?} contains a NUL character"))?;
        let p = c.as_ptr();
        self.strings.push(c);
        Ok(p)
    }

    /// `None` or an empty string → null (the C API's "unset").
    pub fn opt(&mut self, s: Option<&str>) -> Result<*const c_char, String> {
        match s {
            Some(s) if !s.is_empty() => self.str(s),
            _ => Ok(std::ptr::null()),
        }
    }

    /// A file or folder path as the narrow string sherpa-onnx opens (see [`ffi_path`]).
    pub fn path(&mut self, p: &Path) -> Result<*const c_char, String> {
        let c = ffi_path(p)?;
        let ptr = c.as_ptr();
        self.strings.push(c);
        Ok(ptr)
    }

    pub fn opt_path(&mut self, p: Option<&Path>) -> Result<*const c_char, String> {
        match p {
            Some(p) if !p.as_os_str().is_empty() => self.path(p),
            _ => Ok(std::ptr::null()),
        }
    }

    /// Several paths joined with `,` (sherpa's list syntax for lexicons and rule FSTs).
    pub fn path_list(&mut self, paths: &[PathBuf]) -> Result<*const c_char, String> {
        if paths.is_empty() {
            return Ok(std::ptr::null());
        }
        let mut joined = String::new();
        for (i, p) in paths.iter().enumerate() {
            let c = ffi_path(p)?;
            let s = c
                .to_str()
                .map_err(|_| format!("{} is not a valid path", p.display()))?;
            if s.contains(',') {
                return Err(format!(
                    "{} contains a comma, which sherpa-onnx uses to separate list entries",
                    p.display()
                ));
            }
            if i > 0 {
                joined.push(',');
            }
            joined.push_str(s);
        }
        self.str(&joined)
    }
}

/// A path as the narrow `const char*` sherpa-onnx passes to `fopen` / `std::ifstream` and to
/// espeak-ng. MSVC interprets narrow paths in the ANSI code page, so a non-ASCII path (a user
/// name like `José` or `山田` inside `%APPDATA%`) is replaced by its 8.3 short form, which is
/// ASCII whenever short names exist on the volume (the default on system drives). Without a short
/// name the UTF-8 form is passed as a best effort.
pub(crate) fn ffi_path(p: &Path) -> Result<CString, String> {
    let utf8 = p.to_str();
    if let Some(s) = utf8 {
        if s.is_ascii() {
            return CString::new(s).map_err(|_| format!("{} contains a NUL character", p.display()));
        }
    }
    if let Some(short) = short_path(p).filter(|s| s.is_ascii()) {
        return CString::new(short).map_err(|_| format!("{} contains a NUL character", p.display()));
    }
    match utf8 {
        Some(s) => CString::new(s).map_err(|_| format!("{} contains a NUL character", p.display())),
        None => Err(format!("{} is not valid Unicode", p.display())),
    }
}

#[cfg(windows)]
fn short_path(p: &Path) -> Option<String> {
    use std::os::windows::ffi::{OsStrExt, OsStringExt};
    #[link(name = "kernel32")]
    extern "system" {
        fn GetShortPathNameW(long_path: *const u16, short_path: *mut u16, buffer_len: u32) -> u32;
    }
    let wide: Vec<u16> = p.as_os_str().encode_wide().chain(Some(0)).collect();
    // SAFETY: NUL-terminated input; the first call sizes the buffer, the second fills it.
    unsafe {
        let needed = GetShortPathNameW(wide.as_ptr(), std::ptr::null_mut(), 0);
        if needed == 0 {
            return None;
        }
        let mut buf = vec![0u16; needed as usize];
        let n = GetShortPathNameW(wide.as_ptr(), buf.as_mut_ptr(), needed) as usize;
        if n == 0 || n >= buf.len() {
            return None;
        }
        std::ffi::OsString::from_wide(&buf[..n]).into_string().ok()
    }
}

#[cfg(not(windows))]
fn short_path(_p: &Path) -> Option<String> {
    None
}

/// Every path must name an existing file (sherpa-onnx may abort the process on some missing
/// model files instead of returning an error, so they are checked before any create call).
pub(crate) fn require_files<'a>(what: &str, files: impl IntoIterator<Item = &'a Path>) -> Result<(), String> {
    for f in files {
        if !f.is_file() {
            return Err(format!("{what}: {} does not exist", f.display()));
        }
    }
    Ok(())
}

pub(crate) fn require_dir(what: &str, dir: &Path) -> Result<(), String> {
    if !dir.is_dir() {
        return Err(format!("{what}: folder {} does not exist", dir.display()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Call serialization and audio helpers
// ---------------------------------------------------------------------------------------------

/// Serializes every call into one native sherpa object. sherpa-onnx does not document its engines
/// as thread-safe, so each wrapper takes this lock around every call that touches the engine;
/// that makes the wrappers `Send + Sync` without relying on undocumented behaviour. The lock
/// ignores poisoning (a panic in Rust code never leaves the native object half-updated).
#[derive(Default)]
pub(crate) struct CallLock(Mutex<()>);

impl CallLock {
    pub fn lock(&self) -> MutexGuard<'_, ()> {
        self.0.lock().unwrap_or_else(|e| e.into_inner())
    }
}

/// Largest sample count passed in one C call (`int32_t n`); longer inputs are split.
pub(crate) const MAX_SAMPLES_PER_CALL: usize = i32::MAX as usize;

/// Sample rate as the C API's `int32_t`.
pub(crate) fn rate_i32(sample_rate: u32) -> Result<i32, String> {
    if sample_rate == 0 || sample_rate > 384_000 {
        return Err(format!("unsupported sample rate {sample_rate} Hz"));
    }
    Ok(sample_rate as i32)
}

/// Copy a C float array (null or `n <= 0` → empty).
///
/// # Safety
/// `p` must be null or point to at least `n` floats.
pub(crate) unsafe fn floats(p: *const f32, n: i32) -> Vec<f32> {
    if p.is_null() || n <= 0 {
        Vec::new()
    } else {
        std::slice::from_raw_parts(p, n as usize).to_vec()
    }
}

/// Copy an array of `n` C strings (null → empty).
///
/// # Safety
/// `p` must be null or point to `n` valid C string pointers.
pub(crate) unsafe fn strings(p: *const *const c_char, n: i32) -> Vec<String> {
    if p.is_null() || n <= 0 {
        return Vec::new();
    }
    std::slice::from_raw_parts(p, n as usize)
        .iter()
        .map(|&s| owned_str(s))
        .collect()
}
