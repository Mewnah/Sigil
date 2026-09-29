//! Shared ONNX Runtime session factory (voice changer, Silero VAD, Kitten TTS, the EP canary and
//! probe).
//!
//! # Runtime
//! Sigil links ONNX Runtime 1.28 statically (`ort` 2.0.0-rc.13, pyke's
//! `cuda13,tensorrt,nvrtx,directml` dist; ADR 0001). The CUDA provider library is built for CUDA 13
//! ([`CUDA_MAJOR`]): it load-time imports only `cublas64_13.dll` and `cublasLt64_13.dll` (cudart is
//! static) and loads cuDNN 9 when a graph needs it, so it needs an R580+ driver and a Turing or newer
//! GPU. A provider library from another ONNX Runtime release (an old copy left behind) is refused
//! before ORT loads it ([`provider_version_mismatch`]): the provider bridge has no ABI check of its
//! own.
//!
//! # Where the CUDA provider comes from ([`cuda_provider_source`], ADR 0007)
//! Decided once per process, at the first CUDA session (two copies of one provider must never be
//! loaded):
//! 1. `onnxruntime_providers_cuda.dll` next to `Sigil.exe` (the installer ships it while the provider
//!    pack is unpublished): ONNX Runtime loads it through the classic CUDA EP API.
//! 2. Otherwise the `runtime/ort-cuda13-provider` pack in app data (checked against its pinned
//!    sha256 and the linked ORT version) is registered with `register_ep_library`, and every CUDA
//!    session reaches it through its EP device (`with_devices`, device id and VRAM cap as provider
//!    options).
//!
//! `SIGIL_CUDA_PROVIDER=exe|pack` forces one of them (development). `onnxruntime_providers_shared.dll`
//! (10 KB, the provider bridge) always stays next to `Sigil.exe`. The NVIDIA CUDA 13 / cuDNN 9
//! runtime (the `runtime/nvidia-cuda13` model-store pack, or the interim P0 pack) is preloaded by
//! full path before the provider loads (`native_runtime::nvidia::ensure_preloaded`); installing or
//! removing either pack invalidates the cached EP canary results, and a pack removed while Sigil runs
//! makes the CUDA rung fall back until the next start.
//!
//! # Truthful execution providers
//! `ort` silently runs a session on the CPU when a GPU execution provider (EP) fails to register
//! unless `error_on_failure` is set, and some failures only surface when the session is created
//! (CUDA driver init) or first run (a missing cuDNN sub-library). Every session built here instead
//! walks an explicit ladder and reports what it really runs on:
//!
//! | requested | tried in order |
//! |-----------|----------------|
//! | `cuda`    | CUDA → DirectML → CPU |
//! | `dml`     | DirectML → CPU |
//! | `cpu`     | CPU |
//!
//! Each GPU rung first passes a preflight (CUDA: the provider bridge next to `Sigil.exe`, an NVIDIA
//! adapter, the NVIDIA runtime pack preloaded, a usable provider library), then the cached EP canary
//! ([`crate::services::diagnostics::canary`]: a real convolution on that EP and device), then
//! registration with `error_on_failure` and the commit. The first rung that succeeds wins; the
//! reasons of the rungs that failed become [`EpOutcome::reason`] (`outcome_for`), which is
//! recorded in the diagnostics registry under the session's component key (`vc`, `vc.content`,
//! `vc.pitch`, `vc.onnx`, …), emitted to the UI and written to the log.
//! [`VcSessionOptions::component_prefix`] moves that key under a prefix (`bench.vc`, …) so a
//! benchmark never overwrites the voice changer's rows.
//!
//! `vram_cap_mb` limits only the CUDA arena. A CUDA request that falls back to DirectML runs
//! uncapped on the same card, and its reason says so, so a user protecting a game's VRAM sees it.
//!
//! # Session options (unchanged from before)
//! Every session sleeps instead of busy-spinning (`allow_spinning=0`), runs sequentially with
//! Level3 graph optimization and the memory pattern (off on DirectML, which does not support it).
//! The intra/inter thread caps (= 1) are applied for the RVC "Streamer" preset
//! ([`VcSessionOptions::streamer_profile`]); [`VcSessionOptions::intra_threads`] sets another
//! intra-op count, and [`VcSessionOptions::thread_priority`] starts ONNX Runtime's pool threads
//! with the voice-changer worker priority policy. Without them a session keeps ONNX Runtime's
//! default pool.
//!
//! # API
//! - [`build_session`]: explicit component + model source → `(Session, EpOutcome)`.
//! - [`session_builder_for_vc`] / [`session_builder_with_ep`]: the older call shape
//!   (`session_builder_for_vc(ep, opts)?.commit_from_file(path)`), now returning a
//!   [`VcSessionBuilder`] that runs the same ladder. Without [`VcSessionBuilder::component`] the
//!   component is inferred from the model file name (`vc.pitch` for RMVPE/FCPE, `vc.content` for
//!   HuBERT/ContentVec, else `vc`); in-memory models without a component are not recorded.

use std::path::{Path, PathBuf};
use std::sync::{Mutex, Once, OnceLock, PoisonError};
use std::time::Instant;

use ort::ep::{ArenaExtendStrategy, CPU, CUDA};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;

use crate::services::diagnostics::{canary, gpu_mem, record_ep_outcome, truncate_chars, EpOutcome};
use crate::services::native_runtime::nvidia::{self, Preload};
use crate::services::rt_thread::{self, WorkerPriority};

pub const EP_CPU: &str = "cpu";
pub const EP_CUDA: &str = "cuda";
pub const EP_DML: &str = "dml";

/// File name of the CUDA provider library ONNX Runtime loads from `Sigil.exe`'s directory.
pub const CUDA_PROVIDER_DLL: &str = "onnxruntime_providers_cuda.dll";
/// Bridge library every provider library needs next to it.
pub const SHARED_PROVIDER_DLL: &str = "onnxruntime_providers_shared.dll";
/// CUDA major version the linked CUDA provider library is built for (`cublas64_13.dll`, ...).
pub const CUDA_MAJOR: u32 = 13;

static ORT_ENV_INIT: Once = Once::new();

/// Version string of the linked ONNX Runtime (`"1.28.0"`), from `OrtGetApiBase()->GetVersionString`.
pub fn runtime_version() -> &'static str {
    static VERSION: OnceLock<String> = OnceLock::new();
    VERSION.get_or_init(|| {
        // SAFETY: ONNX Runtime is statically linked; the API base and its version string are static
        // for the life of the process and never freed.
        unsafe {
            let base = ort::sys::OrtGetApiBase();
            if base.is_null() {
                return String::new();
            }
            let s = ((*base).GetVersionString)();
            if s.is_null() {
                return String::new();
            }
            std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned()
        }
    })
}

/// `(major, minor)` of a dotted version string (`"1.28.0"` → `(1, 28)`).
fn major_minor(v: &str) -> Option<(u32, u32)> {
    let mut parts = v.trim().split('.');
    Some((parts.next()?.parse().ok()?, parts.next()?.parse().ok()?))
}

/// Why a provider library at `path` must not be loaded into this ONNX Runtime: its file version
/// (e.g. `1.23.2`) has another major.minor than [`runtime_version`]. `None` when it matches or
/// either version cannot be read (then ORT's own load decides).
pub fn provider_version_mismatch(path: &Path) -> Option<String> {
    let file = file_version::of(path)?;
    let runtime = major_minor(runtime_version())?;
    if (u32::from(file[0]), u32::from(file[1])) == runtime {
        return None;
    }
    let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let next_to_exe = match (path.parent(), provider_dir()) {
        (Some(parent), Some(exe_dir)) => same_dir(parent, &exe_dir),
        _ => true,
    };
    let (location, advice) = if next_to_exe {
        ("next to Sigil.exe".to_string(), "an old copy was left behind, reinstall Sigil")
    } else {
        (
            format!("in {}", path.parent().map(|p| p.display().to_string()).unwrap_or_default()),
            "reinstall the CUDA provider pack in Models & Runtimes",
        )
    };
    Some(format!(
        "{name} {location} is from ONNX Runtime {}.{}.{}, but Sigil runs ONNX Runtime {}; {advice}",
        file[0],
        file[1],
        file[2],
        runtime_version()
    ))
}

/// Whether two folder paths name the same folder (canonical comparison, case-insensitive fallback).
fn same_dir(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a.to_string_lossy().eq_ignore_ascii_case(&b.to_string_lossy()),
    }
}

/// PE fixed file version through `version.dll`, resolved lazily from System32 (never a load-time
/// import, so `Sigil.exe`'s import table stays unchanged).
mod file_version {
    use std::path::Path;

    #[cfg(windows)]
    pub fn of(path: &Path) -> Option<[u16; 4]> {
        use std::ffi::c_void;
        use std::os::windows::ffi::OsStrExt;
        use std::sync::OnceLock;
        use windows::core::{s, w};
        use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32};

        type SizeFn = unsafe extern "system" fn(*const u16, *mut u32) -> u32;
        type InfoFn = unsafe extern "system" fn(*const u16, u32, u32, *mut c_void) -> i32;
        type QueryFn = unsafe extern "system" fn(*const c_void, *const u16, *mut *mut c_void, *mut u32) -> i32;
        struct Api {
            size: SizeFn,
            info: InfoFn,
            query: QueryFn,
        }
        static API: OnceLock<Option<Api>> = OnceLock::new();
        type Farproc = unsafe extern "system" fn() -> isize;
        let api = API
            .get_or_init(|| unsafe {
                let module = LoadLibraryExW(w!("version.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32).ok()?;
                let size = GetProcAddress(module, s!("GetFileVersionInfoSizeW"))?;
                let info = GetProcAddress(module, s!("GetFileVersionInfoW"))?;
                let query = GetProcAddress(module, s!("VerQueryValueW"))?;
                // SAFETY: the exports have exactly these signatures (winver.h); never freed.
                Some(Api {
                    size: std::mem::transmute::<Farproc, SizeFn>(size),
                    info: std::mem::transmute::<Farproc, InfoFn>(info),
                    query: std::mem::transmute::<Farproc, QueryFn>(query),
                })
            })
            .as_ref()?;
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(Some(0)).collect();
        let root: Vec<u16> = "\\".encode_utf16().chain(Some(0)).collect();
        // SAFETY: buffers are sized by GetFileVersionInfoSizeW; VerQueryValueW returns a pointer into
        // `buf` (checked for length) that is read unaligned before `buf` is dropped.
        unsafe {
            let mut ignored = 0u32;
            let len = (api.size)(wide.as_ptr(), &mut ignored);
            if len == 0 {
                return None;
            }
            let mut buf = vec![0u8; len as usize];
            if (api.info)(wide.as_ptr(), 0, len, buf.as_mut_ptr().cast()) == 0 {
                return None;
            }
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let mut plen = 0u32;
            if (api.query)(buf.as_ptr().cast(), root.as_ptr(), &mut ptr, &mut plen) == 0 || ptr.is_null() || plen < 16 {
                return None;
            }
            // VS_FIXEDFILEINFO { dwSignature, dwStrucVersion, dwFileVersionMS, dwFileVersionLS, ... }
            let words = ptr as *const u32;
            if std::ptr::read_unaligned(words) != 0xFEEF_04BD {
                return None;
            }
            let ms = std::ptr::read_unaligned(words.add(2));
            let ls = std::ptr::read_unaligned(words.add(3));
            Some([(ms >> 16) as u16, ms as u16, (ls >> 16) as u16, ls as u16])
        }
    }

    #[cfg(not(windows))]
    pub fn of(_path: &Path) -> Option<[u16; 4]> {
        None
    }
}

/// Ensure a process-wide ORT environment exists before the first session.
/// Safe to call repeatedly; configuration sticks on first commit.
pub fn ensure_ort_environment() {
    ORT_ENV_INIT.call_once(|| {
        // Application-owned init (libraries must not commit env themselves).
        if !ort::init().with_name("sigil").commit() {
            tracing::warn!(
                target: "sigil::ort",
                "ONNX Runtime environment was already configured before Sigil's init; the 'sigil' \
                 environment options were not applied"
            );
        }
    });
}

/// Per-session options for voice-changer ONNX graphs (RVC HuBERT/generator, pitch extractors).
#[derive(Clone, Debug, Default)]
pub struct VcSessionOptions {
    /// Device ordinal for the CUDA or DirectML execution provider (ignored on CPU).
    pub cuda_device_index: i32,
    /// RVC "Streamer" preset: cap intra/inter op threads to one each (wins over `intra_threads`).
    pub streamer_profile: bool,
    /// Intra-op threads (the calling thread plus `n - 1` pool threads); `None` keeps ONNX
    /// Runtime's default pool. Ignored with `streamer_profile`.
    pub intra_threads: Option<usize>,
    /// Start this session's ONNX Runtime pool threads with the voice-changer worker priority
    /// policy ([`rt_thread::set_worker_priority`], with the rung's EP as GPU or CPU), so they
    /// yield to a game like the worker that drives them. `None` keeps ONNX Runtime's own threads.
    pub thread_priority: Option<WorkerPriority>,
    /// CUDA arena memory limit in MiB (CUDA only; ignored for DirectML/CPU).
    pub vram_cap_mb: Option<u32>,
    /// Record EP outcomes under `<prefix>.<component>`, e.g. `Some("bench")` → `bench.vc` /
    /// `bench.vc.content` / `bench.vc.pitch`. This keeps a benchmark's sessions from overwriting
    /// the voice changer's diagnostics rows, firing its `diagnostics:ep_outcome` fallback toast, or
    /// being adopted by the worker's `run_ep_outcome`. `None` (or empty) records `<component>`.
    pub component_prefix: Option<&'static str>,
}

/// Canonical EP id for a UI / persisted hint (`gpu` is the legacy spelling of `cuda`).
pub fn normalize_ep(hint: &str) -> &'static str {
    match hint.trim().to_ascii_lowercase().as_str() {
        "cuda" | "gpu" => EP_CUDA,
        "dml" | "directml" => EP_DML,
        _ => EP_CPU,
    }
}

/// Human name of an EP id.
pub fn ep_label(ep: &str) -> &'static str {
    match normalize_ep(ep) {
        EP_CUDA => "CUDA",
        EP_DML => "DirectML",
        _ => "CPU",
    }
}

/// EPs tried, in order, for a requested EP (see the module docs).
pub fn ep_ladder(requested: &str) -> &'static [&'static str] {
    match normalize_ep(requested) {
        EP_CUDA if cfg!(windows) => &[EP_CUDA, EP_DML, EP_CPU],
        EP_CUDA => &[EP_CUDA, EP_CPU],
        EP_DML if cfg!(windows) => &[EP_DML, EP_CPU],
        _ => &[EP_CPU],
    }
}

/// Longest raw ONNX Runtime message kept in a user-facing reason.
const RAW_REASON_CHARS: usize = 300;

/// Drop ONNX Runtime's source-location prefix (`D:\a\...\provider_bridge_ort.cc:1844
/// onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : `) from an error message, and
/// the WIL prefix DirectML failures carry (`...\dml_provider_factory.cc(502)\Sigil.exe!00007FF6...:
/// (caller: 00007FF6...) Exception(1) tid(2a20) `), keeping the HRESULT and its text.
pub(crate) fn strip_ort_prefix(raw: &str) -> &str {
    const MARK: &str = "[ONNXRuntimeError]";
    let msg = match raw.find(MARK) {
        None => raw.trim(),
        Some(at) => {
            // `[ONNXRuntimeError] : <code> : <NAME> : <message>`
            let rest = &raw[at + MARK.len()..];
            let mut parts = rest.splitn(4, " : ");
            match (parts.next(), parts.next(), parts.next(), parts.next()) {
                (Some(_), Some(_), Some(_), Some(msg)) => msg.trim(),
                _ => rest.trim_start_matches([' ', ':']).trim(),
            }
        }
    };
    strip_wil_prefix(msg)
}

/// The text after WIL's `... Exception(<n>) tid(<hex>) ` prefix, when there is any.
fn strip_wil_prefix(msg: &str) -> &str {
    let Some(exception) = msg.find(" Exception(") else {
        return msg;
    };
    let after = &msg[exception..];
    match after.find(" tid(").and_then(|t| after[t..].find(") ").map(|c| t + c + 2)) {
        Some(start) if !after[start..].trim().is_empty() => after[start..].trim(),
        _ => msg,
    }
}

/// The DLL named in `... which depends on "cudnn64_9.dll"` (ORT's provider-load error) or in
/// `LoadLibrary failed for cudnn64_9.dll with error 2` (ORT 1.28's run-time load of cuDNN).
fn missing_dependency(raw: &str) -> Option<&str> {
    if let Some(at) = raw.find("depends on \"") {
        let tail = &raw[at + "depends on \"".len()..];
        return tail.split('"').next().filter(|n| !n.is_empty());
    }
    const LOAD_FAILED: &str = "LoadLibrary failed for ";
    let tail = &raw[raw.find(LOAD_FAILED)? + LOAD_FAILED.len()..];
    let name = tail.split_whitespace().next()?.trim_matches(['"', '\'', ',', ':']);
    name.to_ascii_lowercase().ends_with(".dll").then_some(name)
}

/// Whether `dll` (a bare file name) is part of the CUDA 13 / cuDNN 9 runtime the NVIDIA GPU runtime
/// pack installs (as opposed to Sigil's own files or the driver).
pub(crate) fn is_nvidia_runtime_dll(dll: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "cudart", "cublas", "cufft", "cudnn", "curand", "cusparse", "cusolver", "nvrtc", "nvjitlink",
    ];
    let name = dll.rsplit(['\\', '/']).next().unwrap_or(dll).to_ascii_lowercase();
    PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Reason used when a file of Sigil's own install (next to `Sigil.exe`) is missing.
fn incomplete_install(dll: &str) -> String {
    format!("the Sigil install is incomplete ({dll} is missing next to Sigil.exe); reinstall Sigil")
}

/// Turn a raw ORT / CUDA / DirectML failure into a short, actionable reason (cleaned raw text
/// appended for diagnosis).
pub fn explain_ep_error(ep: &str, raw: &str) -> String {
    let raw = strip_ort_prefix(raw);
    let lower = raw.to_lowercase();
    let hint: Option<String> = match normalize_ep(ep) {
        EP_CUDA => {
            if let Some(dll) = missing_dependency(raw) {
                Some(if is_nvidia_runtime_dll(dll) {
                    format!("{dll} (CUDA {CUDA_MAJOR} / cuDNN 9 runtime) was not found; install the NVIDIA GPU runtime pack in Models & Runtimes")
                } else if dll.eq_ignore_ascii_case("nvcuda.dll") {
                    "the NVIDIA driver is not installed correctly (nvcuda.dll was not found); reinstall the GPU driver".to_string()
                } else if dll.to_ascii_lowercase().starts_with("onnxruntime") {
                    incomplete_install(dll)
                } else {
                    format!("{dll}, which the CUDA provider needs, was not found")
                })
            } else if lower.contains(&SHARED_PROVIDER_DLL.to_ascii_lowercase()) {
                // ORT loads the provider bridge first; the pack cannot fix a missing Sigil file.
                Some(incomplete_install(SHARED_PROVIDER_DLL))
            } else if lower.contains("error 126") || lower.contains("cannot load") || lower.contains("could not be found") {
                Some(format!("the CUDA {CUDA_MAJOR} / cuDNN 9 runtime DLLs were not found; install the NVIDIA GPU runtime pack in Models & Runtimes"))
            } else if lower.contains("insufficient") || lower.contains("cuda failure 35") || lower.contains("driver version") {
                Some(format!("the NVIDIA driver is too old for CUDA {CUDA_MAJOR} (it needs driver R580 or newer); update the GPU driver"))
            } else if lower.contains("no cuda-capable device") || lower.contains("cuda failure 100") {
                Some("no CUDA-capable NVIDIA GPU was found".into())
            } else if lower.contains("out of memory") || lower.contains("cuda failure 2:") {
                Some("the GPU is out of memory".into())
            } else if lower.contains("cudnn") {
                Some("cuDNN failed to initialize; reinstall the NVIDIA GPU runtime pack".into())
            } else {
                None
            }
        }
        EP_DML => {
            if lower.contains("out of memory") || lower.contains("0x8007000e") {
                Some("the GPU is out of memory".into())
            } else if lower.contains("887a0005") || lower.contains("device removed") || lower.contains("device_removed") {
                Some("the GPU driver reset or the GPU was removed".into())
            } else if lower.contains("887a0002") {
                Some("the selected GPU does not exist on this PC (no graphics adapter with that number)".into())
            } else {
                None
            }
        }
        _ => None,
    };
    let raw = truncate_chars(raw, RAW_REASON_CHARS);
    match hint {
        Some(h) => format!("{h} ({raw})"),
        None => raw,
    }
}

/// Coexistence options for every session: sleep instead of busy-spin (preset-independent).
fn apply_common_session_options(builder: SessionBuilder, ep: &str) -> Result<SessionBuilder, String> {
    builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| format!("ONNX graph opt: {}", e))?
        .with_parallel_execution(false)
        .map_err(|e| format!("ONNX parallel exec: {}", e))?
        // DirectML does not support the memory pattern (ORT would disable it with a warning).
        .with_memory_pattern(ep != EP_DML)
        .map_err(|e| format!("ONNX memory pattern: {}", e))?
        .with_intra_op_spinning(false)
        .map_err(|e| format!("ONNX intra spinning: {}", e))?
        .with_inter_op_spinning(false)
        .map_err(|e| format!("ONNX inter spinning: {}", e))
}

/// Streamer-only thread caps: one thread per pool.
fn apply_streamer_thread_caps(builder: SessionBuilder) -> Result<SessionBuilder, String> {
    builder
        .with_intra_threads(1)
        .map_err(|e| format!("ONNX intra threads: {}", e))?
        .with_inter_threads(1)
        .map_err(|e| format!("ONNX inter threads: {}", e))
}

/// Spawns ONNX Runtime's pool threads with a worker priority policy (see
/// [`VcSessionOptions::thread_priority`]). A thread whose policy the OS refuses still runs.
struct PriorityThreads {
    priority: WorkerPriority,
    gpu_ep: bool,
}

impl ort::environment::ThreadManager for PriorityThreads {
    type Thread = std::thread::JoinHandle<()>;

    fn create(&self, work: impl FnOnce() + Send + 'static) -> ort::Result<Self::Thread> {
        let (priority, gpu_ep) = (self.priority, self.gpu_ep);
        std::thread::Builder::new()
            .name("sigil-ort-pool".into())
            .spawn(move || {
                let _ = rt_thread::set_worker_priority(priority, gpu_ep);
                work();
            })
            .map_err(|e| ort::Error::new(format!("ONNX Runtime pool thread: {e}")))
    }

    fn join(thread: Self::Thread) -> ort::Result<()> {
        let _ = thread.join();
        Ok(())
    }
}

/// Thread count and pool-thread priority from `opts` (see [`VcSessionOptions`]).
fn apply_thread_options(builder: SessionBuilder, ep: &str, opts: &VcSessionOptions) -> Result<SessionBuilder, String> {
    let builder = if opts.streamer_profile {
        apply_streamer_thread_caps(builder)?
    } else if let Some(n) = opts.intra_threads {
        builder
            .with_intra_threads(n.max(1))
            .map_err(|e| format!("ONNX intra threads: {}", e))?
    } else {
        builder
    };
    match opts.thread_priority {
        Some(priority) => builder
            .with_thread_manager(PriorityThreads {
                priority,
                gpu_ep: ep != EP_CPU,
            })
            .map_err(|e| format!("ONNX thread manager: {}", e)),
        None => Ok(builder),
    }
}

/// CUDA EP on `device_id`; with a VRAM cap, the arena is limited to that many bytes and grows only
/// by what is requested (power-of-two growth would overshoot the cap).
fn cuda_provider(device_id: i32, opts: &VcSessionOptions) -> CUDA {
    let cuda = CUDA::default().with_device_id(device_id.max(0));
    match opts.vram_cap_mb {
        Some(mb) if mb > 0 => cuda
            .with_memory_limit((mb as usize).saturating_mul(1024 * 1024))
            .with_arena_extend_strategy(ArenaExtendStrategy::SameAsRequested),
        _ => cuda,
    }
}

/// Provider options of a CUDA EP device session: the CUDA device ordinal (the legacy
/// [`CUDA::with_device_id`] meaning) and the VRAM cap exactly as [`cuda_provider`] sets them.
pub(crate) fn cuda_device_options(device_id: i32, opts: &VcSessionOptions) -> Vec<(String, String)> {
    const P: &str = "CUDAExecutionProvider.";
    let mut options = vec![(format!("{P}device_id"), device_id.max(0).to_string())];
    if let Some(mb) = opts.vram_cap_mb.filter(|&mb| mb > 0) {
        options.push((format!("{P}gpu_mem_limit"), (mb as usize).saturating_mul(1024 * 1024).to_string()));
        options.push((format!("{P}arena_extend_strategy"), "kSameAsRequested".to_string()));
    }
    options
}

/// Append the registered CUDA provider's EP device for CUDA ordinal `device_id` (ORT plugin-EP
/// path, ADR 0001 decision 5). One device per session: the CUDA EP drives exactly one GPU.
fn with_cuda_ep_device(builder: SessionBuilder, device_id: i32, opts: &VcSessionOptions) -> Result<SessionBuilder, String> {
    let env = ort::environment::Environment::current().map_err(|e| format!("ONNX Runtime environment: {e}"))?;
    let devices: Vec<_> = env
        .devices()
        .filter(|d| d.ep().ok() == Some("CUDAExecutionProvider"))
        .collect();
    let count = devices.len();
    let index = device_id.max(0) as usize;
    let Some(device) = devices.into_iter().nth(index) else {
        return Err(if count == 0 {
            "the CUDA provider found no usable NVIDIA GPU (driver R580 or newer and a Turing or newer GPU are needed)".to_string()
        } else {
            format!("CUDA device {index} does not exist ({count} CUDA device(s) found)")
        });
    };
    let options = cuda_device_options(device_id, opts);
    builder
        .with_devices([device], Some(&options))
        .map_err(|e| explain_ep_error(EP_CUDA, &e.to_string()))
}

/// A session builder with the common options and exactly one EP registered (`error_on_failure`
/// for GPU EPs, so a registration failure is an `Err`, never a silent CPU session).
pub(crate) fn builder_for_ep(ep: &str, device_id: i32, opts: &VcSessionOptions) -> Result<SessionBuilder, String> {
    ensure_ort_environment();
    let ep = normalize_ep(ep);
    let builder = Session::builder().map_err(|e| format!("ONNX session builder: {}", e))?;
    let builder = apply_common_session_options(builder, ep)?;
    let builder = apply_thread_options(builder, ep, opts)?;
    let registered = match ep {
        EP_CUDA => {
            // The provider's cuBLAS / cuDNN imports must resolve to the NVIDIA runtime pack (callers
            // outside `build_session`, e.g. the importer, reach this without the preflight).
            if let Preload::Removed(why) = nvidia::ensure_preloaded() {
                return Err(why);
            }
            match cuda_provider_source()? {
                CudaProviderSource::Registered(_) => return with_cuda_ep_device(builder, device_id, opts),
                CudaProviderSource::ExeDir(_) => {
                    builder.with_execution_providers([cuda_provider(device_id, opts).build().error_on_failure()])
                }
            }
        }
        #[cfg(windows)]
        EP_DML => builder.with_execution_providers([ort::ep::DirectML::default()
            .with_device_id(device_id.max(0))
            .build()
            .error_on_failure()]),
        _ => builder.with_execution_providers([CPU::default().build()]),
    };
    registered.map_err(|e| explain_ep_error(ep, &e.to_string()))
}

// ---------------------------------------------------------------------------------------------
// CUDA provider source (ADR 0007)
// ---------------------------------------------------------------------------------------------

/// Where the CUDA provider library comes from (decided once per process; see the module docs).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaProviderSource {
    /// `onnxruntime_providers_cuda.dll` next to `Sigil.exe`, loaded by ONNX Runtime through the
    /// classic CUDA EP API.
    ExeDir(PathBuf),
    /// The CUDA provider pack in app data, registered with `register_ep_library`; sessions reach it
    /// through its EP devices.
    Registered(PathBuf),
}

impl CudaProviderSource {
    pub fn path(&self) -> &Path {
        match self {
            CudaProviderSource::ExeDir(p) | CudaProviderSource::Registered(p) => p,
        }
    }
}

/// Development override of the provider choice: `exe` or `pack`.
pub const CUDA_PROVIDER_ENV: &str = "SIGIL_CUDA_PROVIDER";
/// Name the provider pack is registered under (`RegisterExecutionProviderLibrary`).
const CUDA_EP_LIBRARY_NAME: &str = "CUDA";

/// Which provider library to use, given whether each candidate is usable (pure; see the module
/// docs for the order): `exe_dir` is the provider next to `Sigil.exe` or why it cannot be used,
/// `pack` the provider pack's library (`None`: not installed).
pub(crate) fn choose_cuda_provider(
    force: Option<&str>,
    exe_dir: &Result<PathBuf, String>,
    pack: &Option<Result<PathBuf, String>>,
) -> Result<CudaProviderSource, String> {
    let pack_missing = || {
        format!(
            "the CUDA provider pack ({}) is not installed",
            nvidia::CUDA_PROVIDER_PACK_ID
        )
    };
    match force.map(|f| f.trim().to_ascii_lowercase()).as_deref() {
        Some("exe") => exe_dir.clone().map(CudaProviderSource::ExeDir),
        Some("pack") => match pack {
            Some(Ok(p)) => Ok(CudaProviderSource::Registered(p.clone())),
            Some(Err(e)) => Err(e.clone()),
            None => Err(format!("{CUDA_PROVIDER_ENV}=pack, but {}", pack_missing())),
        },
        _ => match (exe_dir, pack) {
            (Ok(p), _) => Ok(CudaProviderSource::ExeDir(p.clone())),
            (Err(_), Some(Ok(p))) => Ok(CudaProviderSource::Registered(p.clone())),
            (Err(exe), Some(Err(pack))) => Err(format!("{exe}; {pack}")),
            (Err(exe), None) => Err(format!("{exe}, and {}", pack_missing())),
        },
    }
}

/// The provider next to `Sigil.exe` when present and built for the linked ONNX Runtime.
fn exe_dir_provider() -> Result<PathBuf, String> {
    let dir = provider_dir().ok_or_else(|| "the Sigil.exe folder is unknown".to_string())?;
    let path = dir.join(CUDA_PROVIDER_DLL);
    if !path.is_file() {
        return Err(format!("the CUDA provider library ({CUDA_PROVIDER_DLL}) is not installed next to Sigil.exe"));
    }
    match provider_version_mismatch(&path) {
        Some(why) => Err(why),
        None => Ok(path),
    }
}

/// The provider pack's library (pinned hash and ORT version checked), when the pack is installed.
fn pack_provider() -> Option<Result<PathBuf, String>> {
    nvidia::provider_pack_library().map(|r| {
        r.and_then(|p| match provider_version_mismatch(&p) {
            Some(why) => Err(why),
            None => Ok(p),
        })
    })
}

static CUDA_PROVIDER: Mutex<Option<CudaProviderSource>> = Mutex::new(None);
/// The registration handle of the provider pack (kept for the life of the process: sessions may
/// exist until exit, and ORT must not unload the library under them).
static CUDA_EP_LIBRARY: Mutex<Option<ort::ep::ExecutionProviderLibrary>> = Mutex::new(None);

/// The CUDA provider this process uses, choosing (and for the pack, registering) it on first use.
/// Only a successful choice is kept, so installing the provider pack later still works.
pub fn cuda_provider_source() -> Result<CudaProviderSource, String> {
    let mut chosen = CUDA_PROVIDER.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(source) = chosen.as_ref() {
        return Ok(source.clone());
    }
    let force = std::env::var(CUDA_PROVIDER_ENV).ok().filter(|v| !v.trim().is_empty());
    let source = choose_cuda_provider(force.as_deref(), &exe_dir_provider(), &pack_provider())?;
    match &source {
        CudaProviderSource::Registered(path) => register_cuda_provider(path)?,
        CudaProviderSource::ExeDir(_) => tracing::info!(target: "sigil::ort", "CUDA provider: {}", source.path().display()),
    }
    *chosen = Some(source.clone());
    Ok(source)
}

/// Register the provider library at `path` with the ORT environment (plugin-EP path); the handle is
/// kept for the life of the process.
fn register_cuda_provider(path: &Path) -> Result<(), String> {
    ensure_ort_environment();
    let env = ort::environment::Environment::current().map_err(|e| format!("ONNX Runtime environment: {e}"))?;
    let started = Instant::now();
    let library = env
        .register_ep_library(CUDA_EP_LIBRARY_NAME, path)
        .map_err(|e| explain_ep_error(EP_CUDA, &e.to_string()))?;
    tracing::info!(
        target: "sigil::ort",
        "CUDA provider registered from {} in {} ms",
        path.display(),
        started.elapsed().as_millis()
    );
    *CUDA_EP_LIBRARY.lock().unwrap_or_else(PoisonError::into_inner) = Some(library);
    Ok(())
}

/// The provider already chosen by this process, if any (never chooses; for probes and the UI).
#[allow(dead_code)] // For the EP probe / diagnostics panel (INTEGRATE).
pub fn cuda_provider_in_use() -> Option<CudaProviderSource> {
    CUDA_PROVIDER.lock().unwrap_or_else(PoisonError::into_inner).clone()
}

/// Invalidate cached canary results when a runtime pack was installed or removed since the last
/// CUDA preflight (their failures are otherwise kept until a user retry).
fn refresh_runtime_fingerprint() {
    static LAST: Mutex<Option<String>> = Mutex::new(None);
    let now = nvidia::fingerprint();
    let mut last = LAST.lock().unwrap_or_else(PoisonError::into_inner);
    if last.as_deref() != Some(now.as_str()) {
        if last.is_some() {
            tracing::info!(target: "sigil::ort", "GPU runtime packs changed; cached GPU tests are discarded");
            canary::invalidate();
        }
        *last = Some(now);
    }
}

/// Where the model comes from.
#[derive(Clone, Copy, Debug)]
pub enum ModelSource<'a> {
    File(&'a Path),
    Memory(&'a [u8]),
}

impl ModelSource<'_> {
    fn describe(&self) -> String {
        match self {
            ModelSource::File(p) => p
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|| p.display().to_string()),
            ModelSource::Memory(b) => format!("<in-memory model, {} bytes>", b.len()),
        }
    }
}

fn commit(mut builder: SessionBuilder, source: ModelSource<'_>) -> Result<Session, String> {
    match source {
        ModelSource::File(p) => builder.commit_from_file(p),
        ModelSource::Memory(b) => builder.commit_from_memory(b),
    }
    .map_err(|e| e.to_string())
}

/// Directory ONNX Runtime loads provider libraries from (the executable's, as ORT is linked
/// statically into it).
pub fn provider_dir() -> Option<PathBuf> {
    std::env::current_exe().ok()?.parent().map(Path::to_path_buf)
}

/// Checks before a GPU rung is attempted; `Err` is the user-facing reason to skip it.
fn gpu_preflight(ep: &str, device_id: i32) -> Result<(), String> {
    if ep == EP_CUDA {
        if let Some(dir) = provider_dir() {
            // Without the bridge ORT fails with a bare "error 126", which reads like the NVIDIA
            // runtime is missing; installing the pack would not help. Both provider sources need it.
            if !dir.join(SHARED_PROVIDER_DLL).is_file() {
                return Err(incomplete_install(SHARED_PROVIDER_DLL));
            }
            // A bridge built for another ONNX Runtime release must never be loaded (no ABI check
            // there; loading it can crash the process). The provider is checked by its source.
            if let Some(why) = provider_version_mismatch(&dir.join(SHARED_PROVIDER_DLL)) {
                return Err(why);
            }
        }
        let adapters = gpu_mem::adapters();
        if !adapters.is_empty() && gpu_mem::nvidia_adapter_indices(&adapters).is_empty() {
            return Err("no NVIDIA GPU was found".to_string());
        }
        refresh_runtime_fingerprint();
        // Loads the NVIDIA runtime pack (if installed) so the provider's imports resolve. A load
        // failure is left to the canary, whose ORT error names the missing piece; CUDA may still
        // work from a system install on PATH.
        if let Preload::Removed(why) = nvidia::ensure_preloaded() {
            return Err(why);
        }
        // Chooses the provider library (and registers the app-data pack when that is the one).
        cuda_provider_source()?;
    }
    canary::run_cached(ep, device_id).map(|_| ())
}

/// Device index for `ep` when `requested` was asked for with `opts.cuda_device_index`. Falling
/// back from CUDA to DirectML targets the same NVIDIA card in DXGI order (DirectML's device id).
fn device_for(ep: &str, requested: &str, opts: &VcSessionOptions) -> i32 {
    let wanted = opts.cuda_device_index.max(0);
    if ep == EP_DML && requested == EP_CUDA {
        let nvidia = gpu_mem::nvidia_adapter_indices(&gpu_mem::adapters());
        return nvidia
            .get(wanted as usize)
            .or(nvidia.first())
            .map(|&i| i as i32)
            .unwrap_or(wanted);
    }
    wanted
}

/// The outcome of a session that `requested` one EP and landed on `active` after the rungs in
/// `reasons` failed (`"<EP>: <why>"` each).
///
/// - Same EP: granted (the ladder starts with the requested EP, so nothing failed before it).
/// - A CUDA request with a VRAM cap that lands on DirectML says the cap is not enforced there (the
///   cap only limits the CUDA arena; the user chose it to protect a game's VRAM on the same card).
/// - A fallback always carries a non-empty reason (an EP the platform does not offer at all).
pub(crate) fn outcome_for(requested: &str, active: &str, mut reasons: Vec<String>, opts: &VcSessionOptions) -> EpOutcome {
    if active == requested {
        // The first rung is always the requested EP, so nothing failed before it.
        return EpOutcome::granted(requested);
    }
    if active == EP_DML && requested == EP_CUDA {
        if let Some(mb) = opts.vram_cap_mb.filter(|&mb| mb > 0) {
            reasons.push(format!(
                "{}: the {mb} MB VRAM cap only applies to CUDA and is not enforced on DirectML",
                ep_label(EP_DML)
            ));
        }
    }
    if reasons.is_empty() {
        reasons.push(format!("{} is not available on this platform", ep_label(requested)));
    }
    EpOutcome::fallback(requested, active, reasons.join("; "))
}

/// Build a session on the best EP of the ladder for `ep_hint`, record the outcome under
/// `component` (when given) and return both. `Err` only when even the CPU fails (then the model
/// itself is the problem and the CPU error is returned).
pub fn build_session(
    component: Option<&str>,
    ep_hint: &str,
    opts: &VcSessionOptions,
    source: ModelSource<'_>,
) -> Result<(Session, EpOutcome), String> {
    ensure_ort_environment();
    if let ModelSource::File(p) = source {
        if !p.is_file() {
            return Err(format!("model file not found: {}", p.display()));
        }
    }
    let component: Option<String> = component.map(|c| match opts.component_prefix {
        Some(p) if !p.is_empty() => format!("{p}.{c}"),
        _ => c.to_string(),
    });
    let component = component.as_deref();
    let requested = normalize_ep(ep_hint);
    let started = Instant::now();
    let mut reasons: Vec<String> = Vec::new();
    for &ep in ep_ladder(requested) {
        let device = device_for(ep, requested, opts);
        if ep != EP_CPU {
            if let Err(why) = gpu_preflight(ep, device) {
                reasons.push(format!("{}: {why}", ep_label(ep)));
                continue;
            }
        }
        let attempt = builder_for_ep(ep, device, opts).and_then(|b| {
            commit(b, source).map_err(|e| if ep == EP_CPU { e } else { explain_ep_error(ep, &e) })
        });
        match attempt {
            Ok(session) => {
                let outcome = outcome_for(requested, ep, std::mem::take(&mut reasons), opts);
                tracing::info!(
                    target: "sigil::ort",
                    "session [{}] {} on {}{} in {} ms",
                    component.unwrap_or("-"),
                    source.describe(),
                    ep_label(ep),
                    if ep == EP_CPU { String::new() } else { format!(" (device {device})") },
                    started.elapsed().as_millis()
                );
                if let Some(c) = component {
                    record_ep_outcome(c, outcome.clone());
                }
                return Ok((session, outcome));
            }
            Err(e) if ep == EP_CPU => {
                if !reasons.is_empty() {
                    tracing::warn!(
                        target: "sigil::ort",
                        "session [{}] {}: GPU attempts failed before the CPU error: {}",
                        component.unwrap_or("-"),
                        source.describe(),
                        reasons.join("; ")
                    );
                }
                return Err(e);
            }
            Err(e) => reasons.push(format!("{}: {e}", ep_label(ep))),
        }
    }
    Err(format!("no execution provider could load {}", source.describe()))
}

/// Component key for a voice-changer model file when the caller did not name one.
pub(crate) fn infer_vc_component(path: &Path) -> &'static str {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    const PITCH: &[&str] = &["rmvpe", "fcpe", "crepe", "swiftf0", "pesto"];
    const CONTENT: &[&str] = &["hubert", "contentvec", "content_vec", "vec-768", "vec-256", "spin", "wavlm"];
    if PITCH.iter().any(|k| name.contains(k)) {
        "vc.pitch"
    } else if CONTENT.iter().any(|k| name.contains(k)) {
        "vc.content"
    } else {
        "vc"
    }
}

/// Deferred session build keeping the older `session_builder_for_vc(ep, opts)?.commit_from_file(p)`
/// call shape; the commit runs [`build_session`] (EP ladder, canary, CPU retry, outcome record).
#[derive(Clone, Debug)]
pub struct VcSessionBuilder {
    ep_hint: String,
    opts: VcSessionOptions,
    component: Option<String>,
}

impl VcSessionBuilder {
    /// Record the outcome under this component key (`vc`, `vc.content`, `vc.pitch`, `vc.onnx`, …).
    pub fn component(mut self, component: &str) -> Self {
        self.component = Some(component.to_string());
        self
    }

    /// Load a model file; the outcome is recorded (component inferred from the file name when unset).
    pub fn commit_from_file(self, path: impl AsRef<Path>) -> Result<Session, String> {
        let path = path.as_ref();
        let component = self.component.clone().unwrap_or_else(|| infer_vc_component(path).to_string());
        build_session(Some(&component), &self.ep_hint, &self.opts, ModelSource::File(path)).map(|(s, _)| s)
    }

    /// Load an in-memory model; recorded only when a component was set.
    pub fn commit_from_memory(self, bytes: &[u8]) -> Result<Session, String> {
        build_session(self.component.as_deref(), &self.ep_hint, &self.opts, ModelSource::Memory(bytes)).map(|(s, _)| s)
    }

    /// Like the commits above, also returning the EP outcome.
    #[allow(dead_code)] // For callers that surface the outcome themselves.
    pub fn commit_with_outcome(self, source: ModelSource<'_>) -> Result<(Session, EpOutcome), String> {
        let component = match (self.component.clone(), source) {
            (Some(c), _) => Some(c),
            (None, ModelSource::File(p)) => Some(infer_vc_component(p).to_string()),
            (None, ModelSource::Memory(_)) => None,
        };
        build_session(component.as_deref(), &self.ep_hint, &self.opts, source)
    }
}

/// Session builder for voice-changer graphs (see [`VcSessionBuilder`] and the module docs).
///
/// - `"cuda"` / `"gpu"`: CUDA on `cuda_device_index` (+ `vram_cap_mb` arena limit), else
///   DirectML on the same NVIDIA card (uncapped; the outcome reason says so), else CPU.
/// - `"dml"` (Windows): DirectML on `cuda_device_index`, else CPU (VRAM cap not applicable).
/// - anything else: CPU.
pub fn session_builder_for_vc(execution_provider_hint: &str, opts: &VcSessionOptions) -> Result<VcSessionBuilder, String> {
    ensure_ort_environment();
    Ok(VcSessionBuilder {
        ep_hint: normalize_ep(execution_provider_hint).to_string(),
        opts: opts.clone(),
        component: None,
    })
}

/// [`session_builder_for_vc`] with default options (device 0, no VRAM cap, default threading).
#[allow(dead_code)] // Pre-P0 call shape; onnx_voice now passes its options through session_builder_for_vc.
pub fn session_builder_with_ep(execution_provider_hint: &str) -> Result<VcSessionBuilder, String> {
    session_builder_for_vc(execution_provider_hint, &VcSessionOptions::default())
}

/// Whether a GPU EP could be used right now, without building a model session: the preflight and
/// canary for `ep` on `device_id`. Used by the probe; cached like the session path.
pub fn check_gpu_ep(ep: &str, device_id: i32) -> Result<(), String> {
    let ep = normalize_ep(ep);
    if ep == EP_CPU {
        return Ok(());
    }
    gpu_preflight(ep, device_id)
}

/// Test helper: preload the CUDA 13 cuBLAS (and optionally cuDNN 9) DLLs named by
/// `SIGIL_TEST_CUDA_BIN` / `SIGIL_TEST_CUDNN_BIN` by full path, like the NVIDIA runtime pack does.
/// `false` (skip the calling test) when `SIGIL_TEST_CUDA_BIN` is unset.
#[cfg(test)]
pub(crate) fn preload_test_cuda_runtime() -> bool {
    let Some(cuda_bin) = std::env::var_os("SIGIL_TEST_CUDA_BIN").map(PathBuf::from) else {
        eprintln!("SIGIL_TEST_CUDA_BIN is not set; skipped");
        return false;
    };
    for dll in ["cublasLt64_13.dll", "cublas64_13.dll"] {
        ort::util::preload_dylib(cuda_bin.join(dll)).unwrap_or_else(|e| panic!("preload {dll}: {e}"));
    }
    if let Some(cudnn_bin) = std::env::var_os("SIGIL_TEST_CUDNN_BIN").map(PathBuf::from) {
        for dll in ort::ep::cuda::CUDNN_DYLIBS {
            if let Err(e) = ort::util::preload_dylib(cudnn_bin.join(dll)) {
                eprintln!("cuDNN preload {dll}: {e}");
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::{canary, ep_outcome};

    #[test]
    fn ep_normalization_and_ladder() {
        assert_eq!(normalize_ep(" GPU "), EP_CUDA);
        assert_eq!(normalize_ep("DirectML"), EP_DML);
        assert_eq!(normalize_ep(""), EP_CPU);
        assert_eq!(normalize_ep("tensorrt"), EP_CPU, "unknown EPs never reach ORT unvalidated");
        assert_eq!(ep_ladder("cpu"), &[EP_CPU]);
        assert_eq!(*ep_ladder("cuda").last().unwrap(), EP_CPU, "every ladder ends on the CPU");
        assert_eq!(*ep_ladder("dml").last().unwrap(), EP_CPU);
        #[cfg(windows)]
        assert_eq!(ep_ladder("cuda"), &[EP_CUDA, EP_DML, EP_CPU]);
    }

    #[test]
    fn linked_runtime_is_onnx_runtime_1_28() {
        // ADR 0001: ort 2.0.0-rc.13 links ONNX Runtime 1.28. A different version here means the
        // wrong pyke dist was linked (e.g. a fallback rung of the ladder without updating the ADR).
        assert_eq!(major_minor(runtime_version()), Some((1, 28)), "{}", runtime_version());
        assert_eq!(major_minor(" 1.23.2 "), Some((1, 23)));
        assert_eq!(major_minor("1"), None);
        assert_eq!(major_minor("x.y"), None);
    }

    /// Research §4.3#6 prerequisite (ADR 0001): rc.13 exposes OrtDmlApi, whose
    /// `SessionOptionsAppendExecutionProvider_DML1(device, queue)` lets the P3 GPU layer hand
    /// DirectML its own high-priority D3D12 command queue.
    #[cfg(windows)]
    #[test]
    fn directml_api_for_a_custom_command_queue_is_exposed() {
        ensure_ort_environment();
        assert!(ort::ep::directml::api().is_some(), "OrtDmlApi must be available in this ORT build");
    }

    #[cfg(windows)]
    #[test]
    fn provider_libraries_from_another_runtime_are_refused() {
        let system = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        let kernel32 = system.join("System32").join("kernel32.dll");
        let v = file_version::of(&kernel32).expect("kernel32.dll has a version resource");
        assert!(v[0] >= 6, "{v:?}");
        // Any DLL whose version is not 1.28.x stands in for a stale 1.23 provider.
        let why = provider_version_mismatch(&kernel32).expect("mismatch");
        assert!(why.contains("kernel32.dll") && why.contains(runtime_version()), "{why}");
        // Outside Sigil.exe's folder (a runtime pack): reinstall the pack.
        assert!(why.contains("reinstall the CUDA provider pack"), "{why}");
        // Next to Sigil.exe (here: the test binary): an old copy of the install, reinstall Sigil.
        if let Some(dir) = provider_dir() {
            let stale = dir.join(format!("sigil-test-stale-provider-{}.dll", std::process::id()));
            if std::fs::copy(&kernel32, &stale).is_ok() {
                let why = provider_version_mismatch(&stale).expect("mismatch");
                let _ = std::fs::remove_file(&stale);
                assert!(why.contains("next to Sigil.exe") && why.contains("reinstall Sigil"), "{why}");
            }
        }
        // Unreadable or missing files: no opinion (ORT's own load decides).
        assert_eq!(provider_version_mismatch(Path::new(r"Z:\missing\onnxruntime_providers_cuda.dll")), None);
        // The providers ort-sys placed next to this binary (when present) match the runtime.
        if let Some(dir) = provider_dir() {
            for dll in [SHARED_PROVIDER_DLL, CUDA_PROVIDER_DLL] {
                let p = dir.join(dll);
                if p.is_file() {
                    assert_eq!(provider_version_mismatch(&p), None, "{}", p.display());
                }
            }
        }
    }

    /// ADR 0007: the installer's provider wins, the app-data pack is registered only when that one
    /// is missing or stale, and the override forces either (errors name both causes).
    #[test]
    fn cuda_provider_source_order_and_overrides() {
        let exe = PathBuf::from(r"C:\Sigil\onnxruntime_providers_cuda.dll");
        let pack = PathBuf::from(r"C:\data\runtimes\ort-cuda13-provider\1.28.0\cuda13\onnxruntime_providers_cuda.dll");
        let exe_ok: Result<PathBuf, String> = Ok(exe.clone());
        let exe_missing: Result<PathBuf, String> = Err("not installed next to Sigil.exe".into());
        let pack_ok = Some(Ok::<_, String>(pack.clone()));
        let pack_bad = Some(Err::<PathBuf, _>("does not match the pinned CUDA provider".to_string()));

        assert_eq!(choose_cuda_provider(None, &exe_ok, &pack_ok), Ok(CudaProviderSource::ExeDir(exe.clone())));
        assert_eq!(choose_cuda_provider(None, &exe_ok, &None), Ok(CudaProviderSource::ExeDir(exe.clone())));
        assert_eq!(choose_cuda_provider(None, &exe_missing, &pack_ok), Ok(CudaProviderSource::Registered(pack.clone())));
        let err = choose_cuda_provider(None, &exe_missing, &None).unwrap_err();
        assert!(err.contains("next to Sigil.exe") && err.contains("runtime/ort-cuda13-provider"), "{err}");
        let err = choose_cuda_provider(None, &exe_missing, &pack_bad).unwrap_err();
        assert!(err.contains("next to Sigil.exe") && err.contains("pinned"), "{err}");

        assert_eq!(choose_cuda_provider(Some("pack"), &exe_ok, &pack_ok), Ok(CudaProviderSource::Registered(pack.clone())));
        assert_eq!(choose_cuda_provider(Some(" PACK "), &exe_ok, &pack_ok), Ok(CudaProviderSource::Registered(pack)));
        assert!(choose_cuda_provider(Some("pack"), &exe_ok, &None).unwrap_err().contains("SIGIL_CUDA_PROVIDER=pack"));
        assert!(choose_cuda_provider(Some("pack"), &exe_ok, &pack_bad).unwrap_err().contains("pinned"));
        assert_eq!(choose_cuda_provider(Some("exe"), &exe_ok, &pack_ok), Ok(CudaProviderSource::ExeDir(exe)));
        assert!(choose_cuda_provider(Some("exe"), &exe_missing, &pack_ok).is_err());
        assert!(choose_cuda_provider(Some("other"), &exe_missing, &pack_ok).is_ok(), "unknown values mean auto");
    }

    #[test]
    fn cuda_device_options_match_the_classic_provider_options() {
        let plain = cuda_device_options(1, &VcSessionOptions::default());
        assert_eq!(plain, vec![("CUDAExecutionProvider.device_id".to_string(), "1".to_string())]);
        let capped = cuda_device_options(
            -3,
            &VcSessionOptions {
                vram_cap_mb: Some(768),
                ..VcSessionOptions::default()
            },
        );
        assert_eq!(
            capped,
            vec![
                ("CUDAExecutionProvider.device_id".to_string(), "0".to_string()),
                ("CUDAExecutionProvider.gpu_mem_limit".to_string(), (768usize * 1024 * 1024).to_string()),
                ("CUDAExecutionProvider.arena_extend_strategy".to_string(), "kSameAsRequested".to_string()),
            ]
        );
        let zero_cap = cuda_device_options(
            0,
            &VcSessionOptions {
                vram_cap_mb: Some(0),
                ..VcSessionOptions::default()
            },
        );
        assert_eq!(zero_cap.len(), 1, "a 0 MB cap means no cap");
    }

    #[cfg(windows)]
    #[test]
    fn folder_comparison_ignores_case() {
        let system = std::env::var_os("SystemRoot").map(PathBuf::from).unwrap_or_else(|| PathBuf::from(r"C:\Windows"));
        let upper = PathBuf::from(system.to_string_lossy().to_uppercase());
        assert!(same_dir(&system, &upper));
        assert!(!same_dir(&system, &system.join("System32")));
    }

    #[test]
    fn component_inference_from_model_names() {
        assert_eq!(infer_vc_component(Path::new("C:/x/base/rmvpe.onnx")), "vc.pitch");
        assert_eq!(infer_vc_component(Path::new("FCPE.onnx")), "vc.pitch");
        assert_eq!(infer_vc_component(Path::new("hubert_base.onnx")), "vc.content");
        assert_eq!(infer_vc_component(Path::new("vec-768-layer-12.onnx")), "vc.content");
        assert_eq!(infer_vc_component(Path::new("models/alice/model.onnx")), "vc");
    }

    #[test]
    fn explanations_are_actionable_and_bounded() {
        let e = explain_ep_error("cuda", "LoadLibrary failed with error 126 \"\" when trying to load onnxruntime_providers_cuda.dll");
        assert!(e.starts_with("the CUDA 13 / cuDNN 9 runtime DLLs were not found"), "{e}");
        // Real ORT 1.23 message on a PC with the CUDA toolkit but no cuDNN.
        let raw = r#"D:\a\ort-artifacts\onnxruntime\core\session\provider_bridge_ort.cc:1844 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : Error loading "C:\x\onnxruntime_providers_cuda.dll" which depends on "cudnn64_9.dll" which is missing. (Error 126: "The specified module could not be found.")"#;
        let e = explain_ep_error("cuda", raw);
        assert!(e.starts_with("cudnn64_9.dll (CUDA 13 / cuDNN 9 runtime) was not found"), "{e}");
        assert!(!e.contains("provider_bridge_ort.cc"), "ORT's source location is dropped: {e}");
        assert_eq!(strip_ort_prefix("plain message"), "plain message");
        let e = explain_ep_error("cuda", "CUDA failure 35: CUDA driver version is insufficient for CUDA runtime version");
        assert!(e.contains("driver is too old"), "{e}");
        let e = explain_ep_error("dml", &"x".repeat(5000));
        assert!(e.chars().count() <= RAW_REASON_CHARS + 1);
        // Real ORT 1.23 DirectML message for a device index that does not exist.
        let raw = r"D:\a\ort-artifacts\onnxruntime\core\providers\dml\dml_provider_factory.cc(502)\Sigil.exe!00007FF626232A1D: (caller: 00007FF62622EFE4) Exception(1) tid(2a20) 887A0002 The object was not found.";
        let e = explain_ep_error("dml", raw);
        assert_eq!(
            e,
            "the selected GPU does not exist on this PC (no graphics adapter with that number) (887A0002 The object was not found.)"
        );
        assert_eq!(strip_ort_prefix("x Exception(1) tid(2a20) "), "x Exception(1) tid(2a20)", "nothing after the prefix: kept");
    }

    #[test]
    fn missing_sigil_files_are_not_blamed_on_the_nvidia_runtime() {
        // ORT loads the provider bridge first; a partial copy of the install fails like this.
        let raw = r#"D:\a\ort\provider_bridge_ort.cc:1778 onnxruntime::ProviderSharedLibrary::Ensure [ONNXRuntimeError] : 1 : FAIL : LoadLibrary failed with error 126 "" when trying to load "C:\Sigil\onnxruntime_providers_shared.dll""#;
        let e = explain_ep_error("cuda", raw);
        assert!(e.starts_with("the Sigil install is incomplete (onnxruntime_providers_shared.dll"), "{e}");
        assert!(!e.contains("runtime pack"), "{e}");
        let raw = r#"Error loading "C:\Sigil\onnxruntime_providers_cuda.dll" which depends on "onnxruntime_providers_shared.dll" which is missing."#;
        assert!(explain_ep_error("cuda", raw).starts_with("the Sigil install is incomplete"));
        let raw = r#"Error loading "C:\Sigil\onnxruntime_providers_cuda.dll" which depends on "nvcuda.dll" which is missing."#;
        assert!(explain_ep_error("cuda", raw).contains("reinstall the GPU driver"));
        let raw = r#"Error loading "C:\Sigil\onnxruntime_providers_cuda.dll" which depends on "cublasLt64_13.dll" which is missing."#;
        assert!(explain_ep_error("cuda", raw).contains("install the NVIDIA GPU runtime pack"));
        // ORT 1.28 loads cuDNN at run time (real message from the dev rig without cuDNN).
        let raw = "cuDNN is unavailable or disabled for CUDA Execution Provider: LoadLibrary failed for cudnn64_9.dll with error 2";
        let e = explain_ep_error("cuda", raw);
        assert!(e.starts_with("cudnn64_9.dll (CUDA 13 / cuDNN 9 runtime) was not found; install the NVIDIA GPU runtime pack"), "{e}");
        assert_eq!(missing_dependency("LoadLibrary failed for something else"), None);
        assert!(is_nvidia_runtime_dll("cudnn_ops64_9.dll") && is_nvidia_runtime_dll(r"C:\x\CUDART64_12.DLL"));
        assert!(!is_nvidia_runtime_dll(SHARED_PROVIDER_DLL) && !is_nvidia_runtime_dll("nvcuda.dll"));
    }

    #[test]
    fn outcomes_join_reasons_and_flag_the_uncapped_directml_fallback() {
        let capped = VcSessionOptions {
            vram_cap_mb: Some(768),
            ..VcSessionOptions::default()
        };
        assert_eq!(outcome_for("cuda", "cuda", vec![], &capped), EpOutcome::granted("cuda"));
        let o = outcome_for("cuda", "dml", vec!["CUDA: cudnn64_9.dll was not found".into()], &capped);
        assert_eq!((o.requested.as_str(), o.active.as_str()), ("cuda", "dml"));
        let reason = o.reason.unwrap();
        assert!(reason.starts_with("CUDA: cudnn64_9.dll was not found; DirectML: the 768 MB VRAM cap"), "{reason}");
        assert!(reason.contains("not enforced on DirectML"), "{reason}");
        // No cap, or a CPU fallback (no VRAM at all): no cap note.
        let o = outcome_for("cuda", "dml", vec!["CUDA: x".into()], &VcSessionOptions::default());
        assert_eq!(o.reason.as_deref(), Some("CUDA: x"));
        let o = outcome_for("cuda", "cpu", vec!["CUDA: x".into(), "DirectML: y".into()], &capped);
        assert_eq!(o.reason.as_deref(), Some("CUDA: x; DirectML: y"));
        // A fallback never has an empty reason.
        let o = outcome_for("dml", "cpu", vec![], &VcSessionOptions::default());
        assert!(o.fell_back() && o.reason.as_deref().is_some_and(|r| r.contains("not available")));
    }

    #[test]
    fn cpu_session_is_granted_and_recorded() {
        let (mut _s, outcome) = build_session(
            Some("test.ort_session.cpu"),
            "cpu",
            &VcSessionOptions::default(),
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("CPU session");
        assert_eq!(outcome, EpOutcome::granted("cpu"));
        assert_eq!(ep_outcome("test.ort_session.cpu"), Some(EpOutcome::granted("cpu")));
    }

    /// A component prefix (the benchmark's `bench`) records `<prefix>.<component>` and leaves the
    /// unprefixed row (the voice changer's) untouched.
    #[test]
    fn component_prefix_records_under_its_own_key() {
        let opts = VcSessionOptions {
            component_prefix: Some("test.prefix"),
            ..VcSessionOptions::default()
        };
        let (_s, outcome) = build_session(
            Some("ort_session.vc"),
            "cpu",
            &opts,
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("CPU session");
        assert_eq!(outcome, EpOutcome::granted("cpu"));
        assert_eq!(ep_outcome("test.prefix.ort_session.vc"), Some(EpOutcome::granted("cpu")));
        assert_eq!(ep_outcome("ort_session.vc"), None);
        // An empty prefix records the plain component.
        let opts = VcSessionOptions {
            component_prefix: Some(""),
            ..VcSessionOptions::default()
        };
        build_session(
            Some("test.ort_session.empty_prefix"),
            "cpu",
            &opts,
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("CPU session");
        assert_eq!(ep_outcome("test.ort_session.empty_prefix"), Some(EpOutcome::granted("cpu")));
    }

    #[test]
    fn legacy_builder_shape_still_works() {
        let session = session_builder_for_vc(
            "cpu",
            &VcSessionOptions {
                streamer_profile: true,
                ..VcSessionOptions::default()
            },
        )
        .unwrap()
        .commit_from_memory(canary::model_bytes());
        assert!(session.is_ok());
        let missing = session_builder_with_ep("cpu").unwrap().commit_from_file("Z:/definitely/missing/model.onnx");
        assert!(missing.unwrap_err().contains("model file not found"));
    }

    /// Two intra-op threads with the worker priority policy on the pool thread: the session
    /// builds, runs on the custom pool thread, and computes the right result.
    #[test]
    fn priority_pool_threads_run_a_session() {
        let opts = VcSessionOptions {
            intra_threads: Some(2),
            thread_priority: Some(WorkerPriority::ProtectGame),
            ..VcSessionOptions::default()
        };
        let (mut session, outcome) = build_session(None, "cpu", &opts, ModelSource::Memory(canary::model_bytes()))
            .expect("CPU session with a thread manager");
        assert_eq!(outcome, EpOutcome::granted("cpu"));
        let input: Vec<f32> = (0..canary::CANARY_IN).map(|i| i as f32).collect();
        let x = ort::value::Tensor::from_array(([1usize, 1, canary::CANARY_IN], input)).unwrap();
        let outputs = session.run(ort::inputs!["X" => &x]).expect("inference");
        let (_, y) = outputs["Y"].try_extract_tensor::<f32>().unwrap();
        let expected = canary::expected_output();
        assert_eq!(y.len(), expected.len());
        assert!(y.iter().zip(expected.iter()).all(|(a, b)| (a - b).abs() < 1e-3), "{y:?}");
    }

    #[test]
    fn broken_models_fail_with_the_cpu_error() {
        let Err(err) = build_session(None, "cpu", &VcSessionOptions::default(), ModelSource::Memory(b"not an onnx model"))
        else {
            panic!("garbage must not load");
        };
        assert!(!err.is_empty());
    }

    /// A GPU rung that fails on any machine: DirectML adapter 63 does not exist (DXGI
    /// `EnumAdapters1` fails), so the canary fails, the session lands on the CPU and says why.
    #[cfg(windows)]
    #[test]
    fn failed_gpu_rung_falls_back_to_cpu_with_the_reason() {
        let opts = VcSessionOptions {
            cuda_device_index: 63,
            ..VcSessionOptions::default()
        };
        let (_session, outcome) = build_session(
            Some("test.ort_session.fallback"),
            "dml",
            &opts,
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("the CPU rung loads the canary");
        eprintln!("dml device 63 → {outcome:?}");
        assert_eq!(outcome.requested, "dml");
        assert_eq!(outcome.active, "cpu");
        assert!(outcome.fell_back() && !outcome.is_gpu());
        let reason = outcome.reason.clone().expect("a fallback carries its reason");
        assert!(reason.starts_with("DirectML: "), "{reason}");
        assert!(reason.len() > "DirectML: ".len(), "{reason}");
        assert_eq!(ep_outcome("test.ort_session.fallback"), Some(outcome), "the registry row matches");
    }

    /// Manual CUDA 13 smoke (ADR 0001). `SIGIL_TEST_CUDA_BIN` names a folder with
    /// `cublasLt64_13.dll` and `cublas64_13.dll` (e.g. the CUDA 13 toolkit's `bin\x64`);
    /// `SIGIL_TEST_CUDNN_BIN` optionally names a folder with the cuDNN 9 DLLs. The DLLs are preloaded
    /// by full path (what the NVIDIA runtime pack does), then the Conv canary must run on CUDA.
    /// `cargo test -- --ignored cuda13_runtime_smoke --nocapture`
    fn check_canary_output(session: &mut Session) {
        let input: Vec<f32> = (0..canary::CANARY_IN).map(|i| i as f32).collect();
        let x = ort::value::Tensor::from_array(([1usize, 1, canary::CANARY_IN], input)).unwrap();
        let outputs = session.run(ort::inputs!["X" => &x]).expect("inference");
        let (_, y) = outputs["Y"].try_extract_tensor::<f32>().unwrap();
        assert!(y.iter().zip(canary::expected_output().iter()).all(|(a, b)| (a - b).abs() < 1e-3), "{y:?}");
    }

    #[test]
    #[ignore = "needs an NVIDIA GPU and CUDA 13 DLLs (SIGIL_TEST_CUDA_BIN)"]
    fn cuda13_runtime_smoke() {
        if !preload_test_cuda_runtime() {
            return;
        }
        canary::invalidate();
        let started = Instant::now();
        let (mut session, outcome) = build_session(
            Some("test.ort_session.cuda13"),
            "cuda",
            &VcSessionOptions::default(),
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("some EP loads the canary");
        eprintln!("cuda13 smoke → {outcome:?} in {} ms (ORT {})", started.elapsed().as_millis(), runtime_version());
        assert_eq!(outcome.active, "cuda", "{outcome:?}");
        check_canary_output(&mut session);
    }

    /// P1-S0a gate (ADR 0001): the CUDA provider library registers from an app-data folder with
    /// `register_ep_library` (ORT plugin-EP registration) instead of loading from `Sigil.exe`'s
    /// folder, and a session reaches it through its EP device (`with_devices`). Same env vars as
    /// [`cuda13_runtime_smoke`]; run it alone so no other test loaded the exe-dir provider first:
    /// `cargo test -- --ignored cuda_provider_registers_from_an_app_data_folder --nocapture`
    #[cfg(windows)]
    #[test]
    #[ignore = "needs an NVIDIA GPU and CUDA 13 DLLs (SIGIL_TEST_CUDA_BIN)"]
    fn cuda_provider_registers_from_an_app_data_folder() {
        use windows::core::HSTRING;
        use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
        fn loaded_from(name: &str) -> Option<PathBuf> {
            // SAFETY: a module handle from GetModuleHandleW and a stack buffer.
            unsafe {
                let h = GetModuleHandleW(&HSTRING::from(name)).ok()?;
                let mut buf = [0u16; 1024];
                let n = GetModuleFileNameW(Some(h), &mut buf) as usize;
                (n > 0).then(|| PathBuf::from(String::from_utf16_lossy(&buf[..n])))
            }
        }
        if !preload_test_cuda_runtime() {
            return;
        }
        assert_eq!(loaded_from(CUDA_PROVIDER_DLL), None, "run this test alone: the exe-dir provider is already loaded");
        let src = provider_dir().unwrap().join(CUDA_PROVIDER_DLL);
        let folder = std::env::temp_dir()
            .join(format!("sigil-ep-library-{}", std::process::id()))
            .join("runtimes")
            .join("ort-cuda13");
        std::fs::create_dir_all(&folder).unwrap();
        let registered = folder.join(CUDA_PROVIDER_DLL);
        std::fs::copy(&src, &registered).expect("copy the provider into the app-data stand-in");

        ensure_ort_environment();
        let env = ort::environment::Environment::current().expect("ORT environment");
        let started = Instant::now();
        let library = env.register_ep_library("CUDA", &registered).expect("register_ep_library");
        let loaded = loaded_from(CUDA_PROVIDER_DLL).expect("the provider is loaded after registration");
        eprintln!("registered in {} ms; loaded from {}", started.elapsed().as_millis(), loaded.display());
        assert_eq!(
            std::fs::canonicalize(&loaded).unwrap(),
            std::fs::canonicalize(&registered).unwrap(),
            "the provider must come from the registered folder"
        );
        for d in env.devices() {
            let hw = d.hardware_device();
            eprintln!("EP device: {} ({}) on {:?} {} #{}", d.ep().unwrap_or("?"), d.ep_vendor().unwrap_or("?"), hw.ty(), hw.vendor().unwrap_or("?"), hw.id());
        }
        let cuda: Vec<_> = env.devices().filter(|d| d.ep().ok() == Some("CUDAExecutionProvider")).collect();
        assert!(!cuda.is_empty(), "the registered library exposes a CUDA EP device");
        let builder = Session::builder().unwrap();
        let builder = apply_common_session_options(builder, EP_CUDA).unwrap();
        let mut builder = builder.with_devices(cuda, None).expect("append the CUDA EP device");
        let mut session = builder.commit_from_memory(canary::model_bytes()).expect("CUDA session via the registered library");
        check_canary_output(&mut session);
        drop(session);
        library.unregister().expect("unregister");
        let _ = std::fs::remove_dir_all(folder.parent().unwrap().parent().unwrap());
    }

    /// ADR 0007: the CUDA provider as an app-data pack. The provider is copied into a runtime-pack
    /// style folder, registered through `register_cuda_provider` (what `cuda_provider_source` does
    /// for `runtime/ort-cuda13-provider`), and a real `build_session` on "cuda" (with a VRAM cap,
    /// so the provider options go through the EP-device path) must run the Conv canary on CUDA, with
    /// the provider module loaded from the pack folder. Same env vars as [`cuda13_runtime_smoke`];
    /// run it alone (a process may load only one copy of the provider):
    /// `cargo test -- --ignored registered_provider_pack_builds_cuda_sessions --nocapture`
    #[cfg(windows)]
    #[test]
    #[ignore = "needs an NVIDIA GPU and CUDA 13 DLLs (SIGIL_TEST_CUDA_BIN, SIGIL_TEST_CUDNN_BIN)"]
    fn registered_provider_pack_builds_cuda_sessions() {
        use windows::core::HSTRING;
        use windows::Win32::System::LibraryLoader::{GetModuleFileNameW, GetModuleHandleW};
        fn loaded_from(name: &str) -> Option<PathBuf> {
            // SAFETY: a module handle from GetModuleHandleW and a stack buffer.
            unsafe {
                let h = GetModuleHandleW(&HSTRING::from(name)).ok()?;
                let mut buf = [0u16; 1024];
                let n = GetModuleFileNameW(Some(h), &mut buf) as usize;
                (n > 0).then(|| PathBuf::from(String::from_utf16_lossy(&buf[..n])))
            }
        }
        if !preload_test_cuda_runtime() {
            return;
        }
        assert_eq!(loaded_from(CUDA_PROVIDER_DLL), None, "run this test alone: a provider is already loaded");
        let exe_dir = provider_dir().unwrap();
        let src = [exe_dir.clone(), exe_dir.parent().unwrap().to_path_buf()]
            .into_iter()
            .map(|d| d.join(CUDA_PROVIDER_DLL))
            .find(|p| p.is_file())
            .expect("ort-sys copied the provider into target/<profile>");
        // The bridge must sit next to the (test) executable, as in an install.
        if !exe_dir.join(SHARED_PROVIDER_DLL).is_file() {
            std::fs::copy(src.with_file_name(SHARED_PROVIDER_DLL), exe_dir.join(SHARED_PROVIDER_DLL)).unwrap();
        }
        let pack = std::env::temp_dir()
            .join(format!("sigil-provider-pack-{}", std::process::id()))
            .join("runtimes/ort-cuda13-provider/1.28.0/cuda13");
        std::fs::create_dir_all(&pack).unwrap();
        let copied = pack.join(CUDA_PROVIDER_DLL);
        std::fs::copy(&src, &copied).unwrap();
        assert_eq!(provider_version_mismatch(&copied), None);

        register_cuda_provider(&copied).expect("register the provider pack");
        *CUDA_PROVIDER.lock().unwrap() = Some(CudaProviderSource::Registered(copied.clone()));
        canary::invalidate();
        let opts = VcSessionOptions {
            vram_cap_mb: Some(512),
            ..VcSessionOptions::default()
        };
        let started = Instant::now();
        let (mut session, outcome) = build_session(
            Some("test.ort_session.cuda.registered"),
            "cuda",
            &opts,
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("some EP loads the canary");
        eprintln!("registered provider → {outcome:?} in {} ms", started.elapsed().as_millis());
        assert_eq!(outcome, EpOutcome::granted("cuda"), "{outcome:?}");
        check_canary_output(&mut session);
        let module = loaded_from(CUDA_PROVIDER_DLL).expect("the provider is loaded");
        assert_eq!(std::fs::canonicalize(module).unwrap(), std::fs::canonicalize(&copied).unwrap());
        assert!(matches!(cuda_provider_in_use(), Some(CudaProviderSource::Registered(_))));
        // A device that does not exist is a clean error on this path too.
        let err = builder_for_ep(EP_CUDA, 7, &opts).err().expect("no CUDA device 7");
        assert!(err.contains("CUDA device 7 does not exist"), "{err}");
    }

    /// ADR 0007 manual acceptance, in one process: with `runtime/nvidia-cuda13` installed by the
    /// model store in the app data folder `SIGIL_APP_DATA_DIR` names (e.g. `SIGIL_LOCAL_ASSETS=<dir
    /// with the four NVIDIA archives> Sigil.exe --headless fetch runtime/nvidia-cuda13`), a CUDA
    /// session runs on CUDA from the pack; after the pack's install record disappears (what "remove"
    /// in the Models tab does), the next CUDA session falls back to DirectML and says why. The record
    /// is restored afterwards. Run it alone:
    /// `SIGIL_APP_DATA_DIR=<scratch> cargo test -- --ignored nvidia_pack_install_and_removal --nocapture`
    #[cfg(windows)]
    #[test]
    #[ignore = "needs an NVIDIA GPU and runtime/nvidia-cuda13 installed under SIGIL_APP_DATA_DIR"]
    fn nvidia_pack_install_and_removal_switch_cuda_and_directml() {
        use crate::services::native_runtime::nvidia::{self, Preload, RuntimeSource};
        struct Restore(PathBuf, PathBuf);
        impl Drop for Restore {
            fn drop(&mut self) {
                let _ = std::fs::rename(&self.1, &self.0);
            }
        }
        let Some(app_data) = std::env::var_os("SIGIL_APP_DATA_DIR").map(PathBuf::from) else {
            eprintln!("SIGIL_APP_DATA_DIR is not set; skipped");
            return;
        };
        let record = app_data.join("models/runtime/nvidia-cuda13/installed.json");
        assert!(record.is_file(), "install runtime/nvidia-cuda13 into {} first", app_data.display());
        let opts = VcSessionOptions::default();
        let (mut session, outcome) = build_session(
            Some("test.ort_session.nvidia_pack"),
            "cuda",
            &opts,
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("a session");
        eprintln!("with the pack → {outcome:?}");
        assert_eq!(outcome, EpOutcome::granted("cuda"));
        check_canary_output(&mut session);
        assert!(matches!(nvidia::ensure_preloaded(), Preload::Loaded { source: RuntimeSource::ModelStore, .. }));
        drop(session);

        let hidden = record.with_extension("json.hidden");
        std::fs::rename(&record, &hidden).unwrap();
        let _restore = Restore(record.clone(), hidden);
        let (_session, outcome) = build_session(
            Some("test.ort_session.nvidia_pack"),
            "cuda",
            &opts,
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("a session");
        eprintln!("after removal → {outcome:?}");
        assert_eq!(outcome.active, "dml", "{outcome:?}");
        assert!(outcome.reason.as_deref().unwrap_or_default().contains("was removed"), "{outcome:?}");
    }

    /// Runs on whatever this machine has (CI: no GPU; dev rig: CUDA without cuDNN → DirectML; with
    /// the runtime pack → CUDA) and checks the invariants that hold everywhere.
    #[test]
    fn cuda_request_reports_what_it_runs_on() {
        let (_, outcome) = build_session(
            Some("test.ort_session.cuda"),
            "cuda",
            &VcSessionOptions::default(),
            ModelSource::Memory(canary::model_bytes()),
        )
        .expect("some EP must load the canary");
        eprintln!("cuda request → {outcome:?}");
        assert_eq!(outcome.requested, "cuda");
        assert_eq!(outcome.fell_back(), outcome.reason.is_some());
        assert!(outcome.reason.as_deref().is_none_or(|r| r.starts_with("CUDA: ")), "{outcome:?}");
        assert_eq!(ep_outcome("test.ort_session.cuda"), Some(outcome));
    }
}
