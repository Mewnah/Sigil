//! Shared ONNX Runtime session factory (voice changer, Silero VAD, the EP canary and probe).
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
//! Each GPU rung first passes a preflight (CUDA: the provider DLL and its bridge next to
//! `Sigil.exe`, an NVIDIA adapter, the NVIDIA runtime pack preloaded), then the cached EP canary
//! ([`crate::services::diagnostics::canary`]: a real convolution on that EP and device), then
//! registration with `error_on_failure` and the commit. The first rung that succeeds wins; the
//! reasons of the rungs that failed become [`EpOutcome::reason`] (`outcome_for`), which is
//! recorded in the diagnostics registry under the session's component key (`vc`, `vc.content`,
//! `vc.pitch`, `vc.onnx`, …), emitted to the UI and written to the log.
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
use std::sync::Once;
use std::time::Instant;

use ort::ep::{ArenaExtendStrategy, CPU, CUDA};
use ort::session::builder::{GraphOptimizationLevel, SessionBuilder};
use ort::session::Session;

use crate::services::diagnostics::{canary, gpu_mem, nvidia_pack, record_ep_outcome, truncate_chars, EpOutcome};
use crate::services::rt_thread::{self, WorkerPriority};

pub const EP_CPU: &str = "cpu";
pub const EP_CUDA: &str = "cuda";
pub const EP_DML: &str = "dml";

/// File name of the CUDA provider library ONNX Runtime loads from `Sigil.exe`'s directory.
pub const CUDA_PROVIDER_DLL: &str = "onnxruntime_providers_cuda.dll";
/// Bridge library every provider library needs next to it.
pub const SHARED_PROVIDER_DLL: &str = "onnxruntime_providers_shared.dll";

static ORT_ENV_INIT: Once = Once::new();

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

/// The DLL named in `... which depends on "cudnn64_9.dll"` (ORT's provider-load error).
fn missing_dependency(raw: &str) -> Option<&str> {
    let tail = &raw[raw.find("depends on \"")? + "depends on \"".len()..];
    tail.split('"').next().filter(|n| !n.is_empty())
}

/// Whether `dll` (a bare file name) is part of the CUDA 12 / cuDNN 9 runtime the NVIDIA GPU runtime
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
                    format!("{dll} (CUDA 12 / cuDNN 9 runtime) was not found; install the NVIDIA GPU runtime pack in Settings")
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
                Some("the CUDA 12 / cuDNN 9 runtime DLLs were not found; install the NVIDIA GPU runtime pack in Settings".into())
            } else if lower.contains("insufficient") || lower.contains("cuda failure 35") || lower.contains("driver version") {
                Some("the NVIDIA driver is too old for CUDA 12; update the GPU driver".into())
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

/// A session builder with the common options and exactly one EP registered (`error_on_failure`
/// for GPU EPs, so a registration failure is an `Err`, never a silent CPU session).
pub(crate) fn builder_for_ep(ep: &str, device_id: i32, opts: &VcSessionOptions) -> Result<SessionBuilder, String> {
    ensure_ort_environment();
    let ep = normalize_ep(ep);
    let builder = Session::builder().map_err(|e| format!("ONNX session builder: {}", e))?;
    let builder = apply_common_session_options(builder, ep)?;
    let builder = apply_thread_options(builder, ep, opts)?;
    let registered = match ep {
        EP_CUDA => builder.with_execution_providers([cuda_provider(device_id, opts).build().error_on_failure()]),
        #[cfg(windows)]
        EP_DML => builder.with_execution_providers([ort::ep::DirectML::default()
            .with_device_id(device_id.max(0))
            .build()
            .error_on_failure()]),
        _ => builder.with_execution_providers([CPU::default().build()]),
    };
    registered.map_err(|e| explain_ep_error(ep, &e.to_string()))
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

fn commit(builder: SessionBuilder, source: ModelSource<'_>) -> Result<Session, String> {
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
            if !dir.join(CUDA_PROVIDER_DLL).is_file() {
                return Err(format!(
                    "the CUDA provider library ({CUDA_PROVIDER_DLL}) is not installed next to Sigil.exe"
                ));
            }
            // Without the bridge ORT fails with a bare "error 126", which reads like the NVIDIA
            // runtime is missing; installing the pack would not help.
            if !dir.join(SHARED_PROVIDER_DLL).is_file() {
                return Err(incomplete_install(SHARED_PROVIDER_DLL));
            }
        }
        let adapters = gpu_mem::adapters();
        if !adapters.is_empty() && gpu_mem::nvidia_adapter_indices(&adapters).is_empty() {
            return Err("no NVIDIA GPU was found".to_string());
        }
        // Loads app_data/runtimes/nvidia-cu12 (if installed) so the provider's imports resolve.
        nvidia_pack::ensure_preloaded();
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
        assert!(e.starts_with("the CUDA 12 / cuDNN 9 runtime DLLs were not found"), "{e}");
        // Real ORT 1.23 message on a PC with the CUDA toolkit but no cuDNN.
        let raw = r#"D:\a\ort-artifacts\onnxruntime\core\session\provider_bridge_ort.cc:1844 onnxruntime::ProviderLibrary::Get [ONNXRuntimeError] : 1 : FAIL : Error loading "C:\x\onnxruntime_providers_cuda.dll" which depends on "cudnn64_9.dll" which is missing. (Error 126: "The specified module could not be found.")"#;
        let e = explain_ep_error("cuda", raw);
        assert!(e.starts_with("cudnn64_9.dll (CUDA 12 / cuDNN 9 runtime) was not found"), "{e}");
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
        let raw = r#"Error loading "C:\Sigil\onnxruntime_providers_cuda.dll" which depends on "cublasLt64_12.dll" which is missing."#;
        assert!(explain_ep_error("cuda", raw).contains("install the NVIDIA GPU runtime pack"));
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
