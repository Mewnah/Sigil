//! Spec validation, `{port}` / `{api_key}` substitution, random keys and the low-level launch:
//! `CREATE_NO_WINDOW | CREATE_SUSPENDED`, piped stdio, join the Job Object, then resume.
//!
//! Children inherit the process error mode at creation, so every launch runs with
//! `SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX` in effect: a helper
//! whose DLL is missing ends at once with 0xC0000135 instead of sitting behind a modal "The code
//! execution cannot proceed" box on the user's desktop.

use super::job::{self, Job, JobLimits, MIN_MEMORY_LIMIT_MB};
use super::{HealthCheck, HelperSpec, PortRequest};
use std::ffi::OsString;
use std::os::windows::io::AsRawHandle;
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Mutex, OnceLock, PoisonError};
use windows::core::{s, w};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};
use windows::Win32::System::Threading::{CREATE_NO_WINDOW, CREATE_SUSPENDED};

/// Placeholder replaced by the chosen port.
pub const PORT_PLACEHOLDER: &str = "{port}";
/// Placeholder replaced by the generated API key.
pub const API_KEY_PLACEHOLDER: &str = "{api_key}";
/// Longest helper name (it appears in log lines, events and thread names).
const MAX_NAME_LEN: usize = 64;
/// Random bytes in a generated API key (hex encoded: 64 characters).
const API_KEY_BYTES: usize = 32;

// ---------------------------------------------------------------------------------------------
// Validation and substitution
// ---------------------------------------------------------------------------------------------

/// Helper names are short, lower-case ids (`llama-server`, `bergamot`, `qwentts.1`).
pub fn validate_name(name: &str) -> Result<(), String> {
    let ok = !name.is_empty()
        && name.len() <= MAX_NAME_LEN
        && name
            .bytes()
            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'-' | b'_' | b'.'))
        && name.as_bytes()[0].is_ascii_alphanumeric();
    if ok {
        Ok(())
    } else {
        Err(format!(
            "invalid helper name {name:?}: use 1-{MAX_NAME_LEN} characters from a-z, 0-9, '-', '_', '.', starting with a letter or digit"
        ))
    }
}

fn validate_health(check: &HealthCheck) -> Result<(), String> {
    let path = &check.path;
    let ok = path.starts_with('/')
        && !path.starts_with("//")
        && path.len() <= 512
        && path
            .bytes()
            .all(|b| b.is_ascii_graphic() && b != b'#' && b != b'\\');
    if !ok {
        return Err(format!(
            "invalid health path {path:?}: it must start with '/' and contain no spaces, '#' or '\\'"
        ));
    }
    if check.timeout_ms == 0 {
        return Err("the health check timeout must be greater than 0".into());
    }
    Ok(())
}

/// Reject specs that can never work, before anything is started.
pub fn validate(spec: &HelperSpec) -> Result<(), String> {
    validate_name(&spec.name)?;
    if spec.program.as_os_str().is_empty() {
        return Err(format!("{}: no program given", spec.name));
    }
    if let PortRequest::Fixed(0) = spec.port {
        return Err(format!("{}: port 0 is not a valid fixed port (use auto)", spec.name));
    }
    if let Some(check) = &spec.health {
        if spec.port == PortRequest::None {
            return Err(format!("{}: a health check needs a port (auto or fixed)", spec.name));
        }
        validate_health(check).map_err(|e| format!("{}: {e}", spec.name))?;
    }
    if let Some(mb) = spec.memory_limit_mb {
        if mb < MIN_MEMORY_LIMIT_MB {
            return Err(format!(
                "{}: memory limit {mb} MB is below the minimum of {MIN_MEMORY_LIMIT_MB} MB",
                spec.name
            ));
        }
    }
    let texts = spec.args.iter().chain(spec.env.iter().map(|(_, v)| v));
    for t in texts {
        if t.contains(PORT_PLACEHOLDER) && spec.port == PortRequest::None {
            return Err(format!("{}: {PORT_PLACEHOLDER} is used but the helper has no port", spec.name));
        }
        if t.contains(API_KEY_PLACEHOLDER) && !spec.api_key {
            return Err(format!("{}: {API_KEY_PLACEHOLDER} is used but api_key is false", spec.name));
        }
    }
    for (k, _) in &spec.env {
        if k.is_empty() || k.contains('=') || k.contains('\0') {
            return Err(format!("{}: invalid environment variable name {k:?}", spec.name));
        }
    }
    if spec.program.components().count() > 1 && !spec.program.is_file() {
        return Err(format!("{}: program not found: {}", spec.name, spec.program.display()));
    }
    if let Some(cwd) = &spec.cwd {
        if !cwd.is_dir() {
            return Err(format!("{}: working directory not found: {}", spec.name, cwd.display()));
        }
    }
    Ok(())
}

/// Replace `{port}` and `{api_key}`. [`validate`] already rejected placeholders without a value.
pub fn substitute(text: &str, port: Option<u16>, api_key: Option<&str>) -> String {
    let mut out = text.to_string();
    if let Some(p) = port {
        out = out.replace(PORT_PLACEHOLDER, &p.to_string());
    }
    if let Some(k) = api_key {
        out = out.replace(API_KEY_PLACEHOLDER, k);
    }
    out
}

/// Whether environment variable `name` matches a removal pattern (`NAME`, or `PREFIX*`).
/// Windows environment names are case-insensitive.
pub fn env_pattern_matches(pattern: &str, name: &str) -> bool {
    match pattern.strip_suffix('*') {
        Some(prefix) => name
            .get(..prefix.len())
            .is_some_and(|head| head.eq_ignore_ascii_case(prefix)),
        None => name.eq_ignore_ascii_case(pattern),
    }
}

// ---------------------------------------------------------------------------------------------
// Random API keys
// ---------------------------------------------------------------------------------------------

/// `ProcessPrng(PBYTE, SIZE_T) -> BOOL` (bcryptprimitives.dll, the RNG std itself uses on Windows 10+).
type ProcessPrngFn = unsafe extern "system" fn(*mut u8, usize) -> i32;
/// `RtlGenRandom` = `SystemFunction036(PVOID, ULONG) -> BOOLEAN` (advapi32), the fallback.
type RtlGenRandomFn = unsafe extern "system" fn(*mut u8, u32) -> u8;

fn process_prng() -> Option<ProcessPrngFn> {
    static CELL: OnceLock<Option<ProcessPrngFn>> = OnceLock::new();
    *CELL.get_or_init(|| unsafe {
        let module = LoadLibraryExW(w!("bcryptprimitives.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
        let proc = GetProcAddress(module, s!("ProcessPrng"))?;
        // SAFETY: documented signature; the module is never freed.
        Some(std::mem::transmute::<unsafe extern "system" fn() -> isize, ProcessPrngFn>(proc))
    })
}

fn rtl_gen_random() -> Option<RtlGenRandomFn> {
    static CELL: OnceLock<Option<RtlGenRandomFn>> = OnceLock::new();
    *CELL.get_or_init(|| unsafe {
        let module = LoadLibraryExW(w!("advapi32.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
        let proc = GetProcAddress(module, s!("SystemFunction036"))?;
        // SAFETY: documented signature (ntsecapi.h RtlGenRandom); the module is never freed.
        Some(std::mem::transmute::<unsafe extern "system" fn() -> isize, RtlGenRandomFn>(proc))
    })
}

/// Fill `buf` from the operating system's CSPRNG.
pub fn os_random(buf: &mut [u8]) -> Result<(), String> {
    if let Some(f) = process_prng() {
        // SAFETY: valid writable buffer of the given length.
        if unsafe { f(buf.as_mut_ptr(), buf.len()) } != 0 {
            return Ok(());
        }
    }
    if let Some(f) = rtl_gen_random() {
        let len = u32::try_from(buf.len()).map_err(|_| "random buffer too large".to_string())?;
        // SAFETY: valid writable buffer of the given length.
        if unsafe { f(buf.as_mut_ptr(), len) } != 0 {
            return Ok(());
        }
    }
    Err("the system random number generator is not available".into())
}

/// A fresh random API key: 256 bits, lower-case hex.
pub fn random_key() -> Result<String, String> {
    let mut bytes = [0u8; API_KEY_BYTES];
    os_random(&mut bytes)?;
    Ok(hex::encode(bytes))
}

// ---------------------------------------------------------------------------------------------
// Error mode
// ---------------------------------------------------------------------------------------------

// `SetErrorMode` / `GetErrorMode` live behind the `Win32_System_Diagnostics_Debug` feature of the
// windows crate, which is not enabled; both are plain kernel32 exports.
#[link(name = "kernel32")]
extern "system" {
    fn SetErrorMode(mode: u32) -> u32;
    fn GetErrorMode() -> u32;
}

/// No critical-error message box (e.g. the loader's "X.dll was not found" box): the call fails.
pub const SEM_FAILCRITICALERRORS: u32 = 0x0001;
/// No Windows Error Reporting fault UI.
pub const SEM_NOGPFAULTERRORBOX: u32 = 0x0002;
/// No `OpenFile` "file not found" box.
pub const SEM_NOOPENFILEERRORBOX: u32 = 0x8000;
/// Error-mode bits every helper and worker starts with.
pub const HELPER_ERROR_MODE: u32 = SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX | SEM_NOOPENFILEERRORBOX;

/// The error mode of this process.
pub fn process_error_mode() -> u32 {
    // SAFETY: reads this process's error mode; no arguments.
    unsafe { GetErrorMode() }
}

/// Set this process's error mode (tests: a fake Sigil with critical-error boxes on).
#[cfg(test)]
pub fn set_process_error_mode(mode: u32) {
    // SAFETY: sets this process's error mode; any combination of SEM_* bits is valid.
    unsafe { SetErrorMode(mode) };
}

/// Create the child with [`HELPER_ERROR_MODE`] added to the error mode it inherits. The process-wide
/// mode is widened only for the `CreateProcess` call and then restored (Sigil's own crash reporting
/// is unchanged); a lock keeps concurrent launches from restoring it under each other.
fn spawn_with_helper_error_mode(cmd: &mut Command) -> std::io::Result<Child> {
    static LOCK: Mutex<()> = Mutex::new(());
    let _guard = LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let previous = process_error_mode();
    // SAFETY: sets this process's error mode; any combination of SEM_* bits is valid.
    unsafe { SetErrorMode(previous | HELPER_ERROR_MODE) };
    let result = cmd.spawn();
    // SAFETY: as above, restoring the value read before.
    unsafe { SetErrorMode(previous) };
    result
}

// ---------------------------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------------------------

/// What to start (placeholders already substituted).
#[derive(Debug)]
pub struct Launch<'a> {
    pub name: &'a str,
    pub program: &'a Path,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    /// Inherited variables to drop first (`NAME` or `PREFIX*`).
    pub env_remove: &'a [String],
    pub cwd: Option<PathBuf>,
    pub limits: JobLimits,
    pub high_qos: bool,
}

/// A started process, already running inside its job, stdio pipes still attached to `child`.
pub struct Launched {
    pub child: Child,
    pub job: Job,
    pub pid: u32,
}

/// Default working directory: the program's folder when the program is a path.
pub fn default_cwd(program: &Path) -> Option<PathBuf> {
    if program.components().count() > 1 {
        program
            .parent()
            .filter(|p| p.is_dir())
            .map(Path::to_path_buf)
    } else {
        None
    }
}

/// Start `launch` without a console window, suspended, join a new job, then resume it. On any
/// failure after the process exists, it is killed before returning the error.
pub fn launch(launch: &Launch<'_>) -> Result<Launched, String> {
    let name = launch.name;
    let job = Job::create(launch.limits).map_err(|e| format!("{name}: cannot create the Job Object: {e}"))?;

    let mut cmd = Command::new(launch.program);
    cmd.args(&launch.args);
    if !launch.env_remove.is_empty() {
        let doomed: Vec<OsString> = std::env::vars_os()
            .map(|(k, _)| k)
            .filter(|k| {
                let k = k.to_string_lossy();
                launch.env_remove.iter().any(|p| env_pattern_matches(p, &k))
            })
            .collect();
        for k in doomed {
            cmd.env_remove(k);
        }
    }
    for (k, v) in &launch.env {
        cmd.env(k, v);
    }
    if let Some(dir) = launch.cwd.clone().or_else(|| default_cwd(launch.program)) {
        cmd.current_dir(dir);
    }
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    let suspended = job::can_resume_suspended();
    let mut flags = CREATE_NO_WINDOW.0;
    if suspended {
        flags |= CREATE_SUSPENDED.0;
    }
    cmd.creation_flags(flags);

    let mut child = spawn_with_helper_error_mode(&mut cmd).map_err(|e| format!("{name}: cannot start {}: {e}", launch.program.display()))?;
    let pid = child.id();
    let fail = |child: &mut Child, msg: String| -> String {
        let _ = child.kill();
        let _ = child.wait();
        format!("{name}: {msg}")
    };
    if let Err(e) = job.assign(child.as_raw_handle()) {
        return Err(fail(&mut child, e));
    }
    if launch.high_qos {
        // Best effort, before the first instruction runs.
        let _ = job::opt_out_eco_qos(child.as_raw_handle());
    }
    if suspended {
        if let Err(e) = job::resume_suspended(child.as_raw_handle()) {
            return Err(fail(&mut child, e));
        }
    }
    Ok(Launched { child, job, pid })
}
