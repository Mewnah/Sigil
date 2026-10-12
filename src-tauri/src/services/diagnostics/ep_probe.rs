//! Execution-provider probe: what GPU acceleration this PC can use, and why not when it cannot.
//!
//! - **Light** ([`probe`] with `deep = false`): runs on a background thread at every start, and
//!   again after the NVIDIA runtime pack is installed or removed. Only cheap, side-effect-free
//!   checks: ORT's compiled EPs, the provider DLLs next to `Sigil.exe`, which `DirectML.dll` the
//!   process loaded, DXGI adapters and driver versions, where the CUDA 13 / cuDNN 9 DLLs would be
//!   found, and the NVIDIA runtime pack state. It never loads CUDA or creates a session (a CUDA
//!   context alone costs hundreds of MB of VRAM).
//! - **Deep** (`deep = true`): additionally preloads the runtime pack, reads the CUDA driver /
//!   runtime / cuDNN versions and runs the EP canary (a real convolution) on CUDA and on every
//!   hardware DirectML adapter. It runs in a child process (`Sigil.exe --headless ep-probe`,
//!   [`run_deep_probe_child`]) so the app never pays for the CUDA context; its report is kept in
//!   `app_data/diagnostics/ep-probe-deep.json`.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use super::gpu_mem::{self, GpuAdapter};
use super::nvidia_pack::{self, NvidiaPackStatus};
use crate::services::native_runtime::nvidia::{self, RuntimeSource};
use crate::services::neural_audio::ort_session::{
    self, CudaProviderSource, CUDA_PROVIDER_DLL, EP_CPU, EP_CUDA, EP_DML, SHARED_PROVIDER_DLL,
};

/// Tauri event carrying the startup (light) [`EpProbeReport`].
pub const PROBE_EVENT: &str = "diagnostics:probe";
pub const REPORT_SCHEMA: u32 = 1;
/// File (under `app_data/diagnostics/`) holding the last deep report.
pub const DEEP_REPORT_FILE: &str = "ep-probe-deep.json";
const DEEP_PROBE_TIMEOUT: Duration = Duration::from_secs(180);
/// Hardware DirectML adapters checked by the deep probe.
const MAX_DML_CHECKS: usize = 4;

/// Where a DLL was (or would be) found.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DllCheck {
    pub name: String,
    pub path: Option<String>,
    /// `loaded` (already in this process) / `pack` / `exe_dir` / `path` / `system` / `missing`.
    pub source: String,
    pub size_bytes: Option<u64>,
}

impl DllCheck {
    pub fn found(&self) -> bool {
        self.source != "missing"
    }
}

/// One EP canary run (deep probe only).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpCheck {
    pub ep: String,
    pub device_id: i32,
    /// DXGI adapter name for DirectML / the NVIDIA adapter for CUDA, when known.
    pub adapter: Option<String>,
    pub ok: bool,
    pub millis: Option<u64>,
    pub error: Option<String>,
}

/// The probe report (`get_ep_probe`, `run_deep_ep_probe`, `Sigil.exe --headless ep-probe`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpProbeReport {
    pub schema: u32,
    pub deep: bool,
    pub generated_at_unix_ms: u64,
    pub app_version: String,
    pub os_version: Option<String>,
    pub exe_dir: Option<String>,
    /// ONNX Runtime C API version the build links.
    pub ort_api_version: u32,
    /// EPs compiled into the linked ONNX Runtime (`cpu`, `cuda`, `dml`).
    pub ort_compiled_eps: Vec<String>,
    /// `onnxruntime_providers_shared.dll`, `onnxruntime_providers_cuda.dll`, `DirectML.dll`
    /// next to `Sigil.exe`.
    pub provider_dlls: Vec<DllCheck>,
    /// The `DirectML.dll` this process actually loaded (bundled redistributable vs. System32).
    pub directml_loaded_from: Option<String>,
    pub adapters: Vec<GpuAdapter>,
    pub nvidia_driver: Option<String>,
    /// Highest CUDA version the installed driver supports (deep).
    pub cuda_driver_version: Option<String>,
    /// CUDA 13 / cuDNN 9 runtime DLLs the CUDA provider needs.
    pub cuda_runtime_dlls: Vec<DllCheck>,
    /// Versions of the CUDA runtime / cuDNN actually loaded (deep).
    pub cudart_version: Option<String>,
    pub cudnn_version: Option<String>,
    pub nvidia_pack: NvidiaPackStatus,
    /// Which NVIDIA runtime install CUDA uses: `model_store` (`runtime/nvidia-cuda13` from Models &
    /// Runtimes, preferred) or `interim` (the Settings pack); `None` when neither is installed.
    #[serde(default)]
    pub nvidia_runtime_source: Option<String>,
    /// The CUDA provider library this process used (deep): `exe_dir` (next to `Sigil.exe`) or
    /// `registered` (the provider pack in app data, `register_ep_library`).
    #[serde(default)]
    pub cuda_provider_source: Option<String>,
    #[serde(default)]
    pub cuda_provider_path: Option<String>,
    /// `native_runtime::nvidia::fingerprint()` when the probe ran: a saved deep report goes stale
    /// once a runtime pack is installed or removed.
    #[serde(default)]
    pub runtime_fingerprint: String,
    /// Canary results (deep).
    pub ep_checks: Vec<EpCheck>,
    /// `cuda` / `dml` / `cpu`.
    pub recommended_ep: String,
    /// Findings in plain words (English; also written to the log).
    pub notes: Vec<String>,
}

/// DLLs the CUDA provider imports (plus the cuDNN sub-libraries it loads at run time).
pub fn cuda_runtime_dll_names() -> Vec<&'static str> {
    nvidia_pack::PRELOAD_ORDER.iter().map(|(n, _)| *n).collect()
}

fn required_cuda_runtime(name: &str) -> bool {
    nvidia_pack::PRELOAD_ORDER.iter().any(|(n, req)| *n == name && *req)
}

fn compiled_eps() -> Vec<String> {
    use ort::ep::ExecutionProvider;
    let mut eps = vec![EP_CPU.to_string()];
    if ort::ep::CUDA::default().is_available().unwrap_or(false) {
        eps.push(EP_CUDA.to_string());
    }
    #[cfg(windows)]
    if ort::ep::DirectML::default().is_available().unwrap_or(false) {
        eps.push(EP_DML.to_string());
    }
    eps
}

fn path_dirs() -> Vec<PathBuf> {
    std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).filter(|d| !d.as_os_str().is_empty()).collect())
        .unwrap_or_default()
}

fn system_dir() -> PathBuf {
    std::env::var_os("SystemRoot")
        .or_else(|| std::env::var_os("windir"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(r"C:\Windows"))
        .join("System32")
}

fn check_at(name: &str, path: PathBuf, source: &str) -> Option<DllCheck> {
    let meta = std::fs::metadata(&path).ok().filter(|m| m.is_file())?;
    Some(DllCheck {
        name: name.to_string(),
        path: Some(path.display().to_string()),
        source: source.to_string(),
        size_bytes: Some(meta.len()),
    })
}

/// Locate `name` the way the CUDA provider's imports would resolve it: already loaded, then the
/// runtime pack, the application directory, `PATH`, System32.
pub fn locate_dll(name: &str, pack_bin: Option<&Path>, exe_dir: Option<&Path>) -> DllCheck {
    if let Some(p) = imp::loaded_module_path(name) {
        return DllCheck {
            name: name.to_string(),
            size_bytes: std::fs::metadata(&p).ok().map(|m| m.len()),
            path: Some(p.display().to_string()),
            source: "loaded".to_string(),
        };
    }
    let mut candidates: Vec<(PathBuf, &str)> = Vec::new();
    if let Some(b) = pack_bin {
        candidates.push((b.join(name), "pack"));
    }
    if let Some(d) = exe_dir {
        candidates.push((d.join(name), "exe_dir"));
    }
    candidates.extend(path_dirs().into_iter().map(|d| (d.join(name), "path")));
    candidates.push((system_dir().join(name), "system"));
    candidates
        .into_iter()
        .find_map(|(p, src)| check_at(name, p, src))
        .unwrap_or(DllCheck {
            name: name.to_string(),
            path: None,
            source: "missing".to_string(),
            size_bytes: None,
        })
}

/// [`locate_dll`], but a DLL that is not loaded yet is looked up first in the model-store NVIDIA
/// pack (`store`, lower-case names), which the CUDA session factory preloads before the interim pack.
fn locate_runtime_dll(
    name: &str,
    store: Option<&BTreeMap<String, PathBuf>>,
    pack_bin: Option<&Path>,
    exe_dir: Option<&Path>,
) -> DllCheck {
    if imp::loaded_module_path(name).is_none() {
        if let Some(c) = store
            .and_then(|ix| ix.get(&name.to_ascii_lowercase()))
            .and_then(|p| check_at(name, p.clone(), "pack"))
        {
            return c;
        }
    }
    locate_dll(name, pack_bin, exe_dir)
}

fn runtime_source_name(source: RuntimeSource) -> &'static str {
    match source {
        RuntimeSource::ModelStore => "model_store",
        RuntimeSource::Interim => "interim",
    }
}

fn provider_dll_checks(exe_dir: Option<&Path>) -> Vec<DllCheck> {
    [SHARED_PROVIDER_DLL, CUDA_PROVIDER_DLL, "DirectML.dll"]
        .into_iter()
        .map(|name| {
            exe_dir
                .and_then(|d| check_at(name, d.join(name), "exe_dir"))
                .unwrap_or(DllCheck {
                    name: name.to_string(),
                    path: None,
                    source: "missing".to_string(),
                    size_bytes: None,
                })
        })
        .collect()
}

/// Pick the EP to recommend and explain the situation.
pub(crate) fn assess(r: &EpProbeReport) -> (String, Vec<String>) {
    let mut notes = Vec::new();
    let hardware: Vec<&GpuAdapter> = r.adapters.iter().filter(|a| !a.software).collect();
    let nvidia = hardware.iter().any(|a| a.vendor_id == gpu_mem::VENDOR_NVIDIA);
    // Provider libraries from another ONNX Runtime release are refused by the session factory.
    let stale_providers: Vec<String> = r
        .provider_dlls
        .iter()
        .filter(|d| d.name == CUDA_PROVIDER_DLL || d.name == SHARED_PROVIDER_DLL)
        .filter_map(|d| d.path.as_deref().and_then(|p| ort_session::provider_version_mismatch(Path::new(p))))
        .collect();
    let provider_ok = r.provider_dlls.iter().any(|d| d.name == CUDA_PROVIDER_DLL && d.found())
        && r.provider_dlls.iter().any(|d| d.name == SHARED_PROVIDER_DLL && d.found())
        && stale_providers.is_empty();
    notes.extend(stale_providers);
    let missing_runtime: Vec<&str> = r
        .cuda_runtime_dlls
        .iter()
        .filter(|d| !d.found() && required_cuda_runtime(&d.name))
        .map(|d| d.name.as_str())
        .collect();

    if hardware.is_empty() {
        notes.push("No hardware GPU adapter was found: models run on the CPU.".to_string());
    }
    if nvidia && !provider_ok {
        notes.push(format!(
            "This Sigil build has no CUDA provider ({CUDA_PROVIDER_DLL} next to Sigil.exe); NVIDIA GPUs use DirectML."
        ));
    }
    if nvidia && provider_ok && !missing_runtime.is_empty() {
        notes.push(format!(
            "The CUDA 13 / cuDNN 9 runtime is not installed (missing {}). Install the NVIDIA GPU runtime pack (about {:.1} GB) to use CUDA; DirectML is used until then.",
            missing_runtime.join(", "),
            nvidia_pack::download_bytes() as f64 / 1e9
        ));
    }
    if let Some(dml) = &r.directml_loaded_from {
        if !r.provider_dlls.iter().any(|d| d.name == "DirectML.dll" && d.found()) {
            notes.push(format!(
                "DirectML.dll is the Windows copy ({dml}), not the redistributable bundled with Sigil; DirectML may be slower or unavailable."
            ));
        }
    }
    if r.nvidia_pack.pending_removal {
        notes.push("The NVIDIA GPU runtime pack will be removed when Sigil restarts.".to_string());
    }
    if let (Some(driver), Some(cuda)) = (&r.nvidia_driver, &r.cuda_driver_version) {
        notes.push(format!("NVIDIA driver {driver} supports CUDA up to {cuda}."));
    }

    let recommended = if r.deep {
        let ok = |ep: &str| r.ep_checks.iter().any(|c| c.ep == ep && c.ok);
        for c in r.ep_checks.iter().filter(|c| !c.ok) {
            notes.push(format!(
                "{} test on device {} failed: {}",
                ort_session::ep_label(&c.ep),
                c.device_id,
                c.error.as_deref().unwrap_or("unknown error")
            ));
        }
        if ok(EP_CUDA) {
            EP_CUDA
        } else if ok(EP_DML) {
            EP_DML
        } else {
            EP_CPU
        }
    } else if nvidia && provider_ok && missing_runtime.is_empty() {
        EP_CUDA
    } else if !hardware.is_empty() && r.ort_compiled_eps.iter().any(|e| e == EP_DML) {
        EP_DML
    } else {
        EP_CPU
    };
    (recommended.to_string(), notes)
}

/// Run the probe. `deep` loads GPU runtimes and runs canaries: only call it in a headless child.
pub fn probe(deep: bool) -> EpProbeReport {
    let exe_dir = ort_session::provider_dir();
    let app_data = super::app_data_dir();
    let mut pack = nvidia_pack::status(app_data.as_deref());
    // The model-store pack (runtime/nvidia-cuda13) wins over the interim pack, as in
    // `native_runtime::nvidia::ensure_preloaded`.
    let store_dlls = nvidia::store_dll_index();
    let pack_bin = app_data
        .as_deref()
        .map(nvidia_pack::bin_dir)
        .filter(|_| pack.installed && store_dlls.is_none());

    let adapters = gpu_mem::adapters();
    let nvidia_driver = adapters
        .iter()
        .find(|a| a.vendor_id == gpu_mem::VENDOR_NVIDIA)
        .and_then(|a| a.nvidia_driver.clone());

    let mut report = EpProbeReport {
        schema: REPORT_SCHEMA,
        deep,
        generated_at_unix_ms: super::logging::unix_ms_now(),
        app_version: env!("CARGO_PKG_VERSION").to_string(),
        os_version: imp::os_version(),
        exe_dir: exe_dir.as_ref().map(|d| d.display().to_string()),
        ort_api_version: ort::MINOR_VERSION,
        ort_compiled_eps: compiled_eps(),
        provider_dlls: provider_dll_checks(exe_dir.as_deref()),
        directml_loaded_from: imp::loaded_module_path("DirectML.dll").map(|p| p.display().to_string()),
        adapters,
        nvidia_driver,
        cuda_driver_version: None,
        cuda_runtime_dlls: Vec::new(),
        cudart_version: None,
        cudnn_version: None,
        nvidia_pack: NvidiaPackStatus::default(),
        nvidia_runtime_source: None,
        cuda_provider_source: None,
        cuda_provider_path: None,
        runtime_fingerprint: nvidia::fingerprint(),
        ep_checks: Vec::new(),
        recommended_ep: EP_CPU.to_string(),
        notes: Vec::new(),
    };

    if deep {
        report.cuda_driver_version = imp::cuda_driver_version();
        let hardware: Vec<(u32, String, u32)> = report
            .adapters
            .iter()
            .filter(|a| !a.software)
            .map(|a| (a.index, a.name.clone(), a.vendor_id))
            .collect();
        let nvidia_name = hardware.iter().find(|(_, _, v)| *v == gpu_mem::VENDOR_NVIDIA).map(|(_, n, _)| n.clone());
        let mut checks = Vec::new();
        if report.ort_compiled_eps.iter().any(|e| e == EP_CUDA) {
            checks.push(run_check(EP_CUDA, 0, nvidia_name));
        }
        #[cfg(windows)]
        for (index, name, _) in hardware.iter().take(MAX_DML_CHECKS) {
            checks.push(run_check(EP_DML, *index as i32, Some(name.clone())));
        }
        report.ep_checks = checks;
        report.cudart_version = imp::cudart_version();
        report.cudnn_version = imp::cudnn_version();
        // The pack may have been preloaded by the CUDA check.
        pack = nvidia_pack::status(app_data.as_deref());
    }
    report.cuda_runtime_dlls = cuda_runtime_dll_names()
        .into_iter()
        .map(|n| locate_runtime_dll(n, store_dlls.as_ref(), pack_bin.as_deref(), exe_dir.as_deref()))
        .collect();
    report.nvidia_runtime_source = match nvidia::preload_state() {
        Some((source, _, _)) => Some(runtime_source_name(source)),
        None if store_dlls.is_some() => Some(runtime_source_name(RuntimeSource::ModelStore)),
        None if pack.installed => Some(runtime_source_name(RuntimeSource::Interim)),
        None => None,
    }
    .map(str::to_string);
    if let Some(provider) = ort_session::cuda_provider_in_use() {
        let source = match provider {
            CudaProviderSource::ExeDir(_) => "exe_dir",
            CudaProviderSource::Registered(_) => "registered",
        };
        report.cuda_provider_source = Some(source.to_string());
        report.cuda_provider_path = Some(provider.path().display().to_string());
    }
    report.nvidia_pack = pack;
    let (recommended, notes) = assess(&report);
    report.recommended_ep = recommended;
    report.notes = notes;
    report
}

fn run_check(ep: &str, device_id: i32, adapter: Option<String>) -> EpCheck {
    let started = Instant::now();
    let result = ort_session::check_gpu_ep(ep, device_id);
    EpCheck {
        ep: ep.to_string(),
        device_id,
        adapter,
        ok: result.is_ok(),
        millis: Some(started.elapsed().as_millis() as u64),
        error: result.err(),
    }
}

/// Log a one-paragraph summary of a report.
pub fn log_summary(r: &EpProbeReport) {
    let gpus: Vec<String> = r
        .adapters
        .iter()
        .filter(|a| !a.software)
        .map(|a| match (&a.nvidia_driver, &a.driver_version) {
            (Some(nv), _) => format!("{} (driver {nv}, {} MB)", a.name, a.dedicated_vram_mb),
            (None, Some(v)) => format!("{} (driver {v}, {} MB)", a.name, a.dedicated_vram_mb),
            _ => a.name.clone(),
        })
        .collect();
    tracing::info!(
        target: "sigil::diagnostics",
        "{} EP probe: recommended {}; compiled EPs [{}]; GPUs [{}]; NVIDIA pack {}; OS {}",
        if r.deep { "deep" } else { "startup" },
        ort_session::ep_label(&r.recommended_ep),
        r.ort_compiled_eps.join(", "),
        gpus.join("; "),
        if r.nvidia_pack.installed { "installed" } else { "not installed" },
        r.os_version.as_deref().unwrap_or("?")
    );
    for c in &r.ep_checks {
        tracing::info!(
            target: "sigil::diagnostics",
            "EP check {} device {}: {}",
            ort_session::ep_label(&c.ep),
            c.device_id,
            match &c.error {
                None => format!("ok in {} ms", c.millis.unwrap_or(0)),
                Some(e) => format!("failed: {e}"),
            }
        );
    }
    for n in &r.notes {
        tracing::info!(target: "sigil::diagnostics", "EP probe: {n}");
    }
}

/// Run the deep probe in a child `Sigil.exe --headless ep-probe --out <file>` and parse its report.
/// Blocking (call from a blocking task). The child inherits `SIGIL_APP_DATA_DIR`.
pub fn run_deep_probe_child(app_data: &Path) -> Result<EpProbeReport, String> {
    use std::process::{Command, Stdio};
    let exe = std::env::current_exe().map_err(|e| format!("cannot locate Sigil.exe: {e}"))?;
    let dir = app_data.join("diagnostics");
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let out = dir.join(DEEP_REPORT_FILE);
    let tmp = dir.join(format!("{DEEP_REPORT_FILE}.{}.tmp", std::process::id()));
    let _ = std::fs::remove_file(&tmp);
    let started_ms = super::logging::unix_ms_now();
    let mut cmd = Command::new(&exe);
    cmd.arg(crate::services::cli::HEADLESS_FLAG)
        .arg("ep-probe")
        .arg("--out")
        .arg(&tmp)
        .env(super::APP_DATA_ENV, app_data)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
    let mut child = cmd.spawn().map_err(|e| format!("cannot start the GPU check: {e}"))?;
    let deadline = Instant::now() + DEEP_PROBE_TIMEOUT;
    let status = loop {
        match child.try_wait() {
            Ok(Some(s)) => break Some(s),
            Ok(None) if Instant::now() > deadline => {
                let _ = child.kill();
                let _ = child.wait();
                break None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(100)),
            Err(e) => return Err(format!("GPU check: {e}")),
        }
    };
    // A driver can crash the child during teardown after the report was written: trust the file.
    let parsed = std::fs::read_to_string(&tmp)
        .ok()
        .and_then(|t| serde_json::from_str::<EpProbeReport>(&t).ok())
        .filter(|r| r.deep && r.generated_at_unix_ms >= started_ms);
    let _ = std::fs::remove_file(&tmp);
    match (parsed, status) {
        (Some(report), _) => {
            let json = serde_json::to_vec_pretty(&report).map_err(|e| e.to_string())?;
            let _ = std::fs::write(&out, json);
            Ok(report)
        }
        (None, None) => Err(format!(
            "the GPU check did not finish within {} s (a GPU driver may be hung)",
            DEEP_PROBE_TIMEOUT.as_secs()
        )),
        (None, Some(s)) => Err(format!("the GPU check exited without a report ({s})")),
    }
}

/// The last deep report saved by [`run_deep_probe_child`].
pub fn last_deep_report(app_data: &Path) -> Option<EpProbeReport> {
    let text = std::fs::read_to_string(app_data.join("diagnostics").join(DEEP_REPORT_FILE)).ok()?;
    serde_json::from_str::<EpProbeReport>(&text).ok().filter(|r| r.schema == REPORT_SCHEMA)
}

/// Whether a saved deep report still describes this install: same Sigil version, the same interim
/// NVIDIA runtime pack state (pinned label, installed, pending removal) and the same runtime packs
/// (`fingerprint_now` = `native_runtime::nvidia::fingerprint()`: model-store NVIDIA and CUDA
/// provider packs). A report from before a pack was installed or removed would contradict the pack
/// section, so the UI drops it.
pub fn deep_report_is_current(report: &EpProbeReport, pack_now: &NvidiaPackStatus, fingerprint_now: &str) -> bool {
    let then = &report.nvidia_pack;
    report.app_version == env!("CARGO_PKG_VERSION")
        && then.label == pack_now.label
        && then.installed == pack_now.installed
        && then.pending_removal == pack_now.pending_removal
        && report.runtime_fingerprint == fingerprint_now
}

#[cfg(windows)]
mod imp {
    use std::path::PathBuf;
    use windows::core::{s, w, PCSTR, PCWSTR};
    use windows::Win32::Foundation::HMODULE;
    use windows::Win32::System::LibraryLoader::{
        GetModuleFileNameW, GetModuleHandleW, GetProcAddress, LoadLibraryExW, LOAD_LIBRARY_SEARCH_SYSTEM32,
    };

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    fn module_handle(name: &str) -> Option<HMODULE> {
        let w = wide(name);
        // SAFETY: NUL-terminated name; GetModuleHandleW never loads anything.
        unsafe { GetModuleHandleW(PCWSTR(w.as_ptr())) }.ok()
    }

    /// Full path of `name` if this process already loaded it (no loading happens).
    pub fn loaded_module_path(name: &str) -> Option<PathBuf> {
        let h = module_handle(name)?;
        let mut buf = vec![0u16; 1024];
        // SAFETY: valid module handle and buffer.
        let n = unsafe { GetModuleFileNameW(Some(h), &mut buf) } as usize;
        (n > 0 && n < buf.len()).then(|| PathBuf::from(String::from_utf16_lossy(&buf[..n])))
    }

    fn proc_in(module: HMODULE, name: PCSTR) -> Option<unsafe extern "system" fn() -> isize> {
        // SAFETY: valid module; name is a NUL-terminated ASCII string.
        unsafe { GetProcAddress(module, name) }
    }

    fn cuda_style(v: i32) -> Option<String> {
        (v > 0).then(|| format!("{}.{}", v / 1000, (v % 1000) / 10))
    }

    /// `cuDriverGetVersion` from System32's nvcuda.dll (does not initialize CUDA).
    pub fn cuda_driver_version() -> Option<String> {
        // SAFETY: System32-only load of the NVIDIA driver DLL; kept loaded (process is short-lived).
        let module = unsafe { LoadLibraryExW(w!("nvcuda.dll"), None, LOAD_LIBRARY_SEARCH_SYSTEM32) }.ok()?;
        let f = proc_in(module, s!("cuDriverGetVersion"))?;
        let mut v = 0i32;
        // SAFETY: `CUresult cuDriverGetVersion(int*)`.
        let f: unsafe extern "system" fn(*mut i32) -> i32 = unsafe { std::mem::transmute(f) };
        if unsafe { f(&mut v) } == 0 {
            cuda_style(v)
        } else {
            None
        }
    }

    /// `cudaRuntimeGetVersion` of the loaded cudart64_13.dll (the pack's; ORT's own CUDA runtime is static).
    pub fn cudart_version() -> Option<String> {
        let f = proc_in(module_handle("cudart64_13.dll")?, s!("cudaRuntimeGetVersion"))?;
        let mut v = 0i32;
        // SAFETY: `cudaError_t cudaRuntimeGetVersion(int*)`.
        let f: unsafe extern "system" fn(*mut i32) -> i32 = unsafe { std::mem::transmute(f) };
        if unsafe { f(&mut v) } == 0 {
            cuda_style(v)
        } else {
            None
        }
    }

    /// `cudnnGetVersion` of the loaded cudnn64_9.dll (9.x: major*10000 + minor*100 + patch).
    pub fn cudnn_version() -> Option<String> {
        let f = proc_in(module_handle("cudnn64_9.dll")?, s!("cudnnGetVersion"))?;
        // SAFETY: `size_t cudnnGetVersion(void)`.
        let f: unsafe extern "system" fn() -> usize = unsafe { std::mem::transmute(f) };
        let v = unsafe { f() };
        (v > 0).then(|| format!("{}.{}.{}", v / 10_000, (v % 10_000) / 100, v % 100))
    }

    #[repr(C)]
    struct OsVersionInfoW {
        size: u32,
        major: u32,
        minor: u32,
        build: u32,
        platform: u32,
        csd: [u16; 128],
    }

    /// Real Windows version via `RtlGetVersion` (not subject to manifest-based lies).
    pub fn os_version() -> Option<String> {
        let ntdll = module_handle("ntdll.dll")?;
        let f = proc_in(ntdll, s!("RtlGetVersion"))?;
        // SAFETY: `NTSTATUS RtlGetVersion(PRTL_OSVERSIONINFOW)`.
        let f: unsafe extern "system" fn(*mut OsVersionInfoW) -> i32 = unsafe { std::mem::transmute(f) };
        let mut info = OsVersionInfoW {
            size: std::mem::size_of::<OsVersionInfoW>() as u32,
            major: 0,
            minor: 0,
            build: 0,
            platform: 0,
            csd: [0; 128],
        };
        (unsafe { f(&mut info) } == 0).then(|| format!("Windows {}.{}.{}", info.major, info.minor, info.build))
    }
}

#[cfg(not(windows))]
mod imp {
    use std::path::PathBuf;
    pub fn loaded_module_path(_name: &str) -> Option<PathBuf> {
        None
    }
    pub fn cuda_driver_version() -> Option<String> {
        None
    }
    pub fn cudart_version() -> Option<String> {
        None
    }
    pub fn cudnn_version() -> Option<String> {
        None
    }
    pub fn os_version() -> Option<String> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter(index: u32, vendor_id: u32, software: bool) -> GpuAdapter {
        GpuAdapter {
            index,
            name: format!("gpu{index}"),
            vendor: gpu_mem::vendor_name(vendor_id).into(),
            vendor_id,
            device_id: 0,
            luid: String::new(),
            dedicated_vram_mb: 8192,
            shared_memory_mb: 0,
            software,
            driver_version: None,
            nvidia_driver: None,
            vram: None,
        }
    }

    fn dll(name: &str, found: bool) -> DllCheck {
        DllCheck {
            name: name.into(),
            path: found.then(|| format!("C:/x/{name}")),
            source: if found { "exe_dir" } else { "missing" }.into(),
            size_bytes: found.then_some(1),
        }
    }

    fn base(adapters: Vec<GpuAdapter>, cuda_provider: bool, runtime: bool) -> EpProbeReport {
        EpProbeReport {
            schema: REPORT_SCHEMA,
            deep: false,
            generated_at_unix_ms: 1,
            app_version: "t".into(),
            os_version: None,
            exe_dir: None,
            ort_api_version: 0,
            ort_compiled_eps: vec!["cpu".into(), "cuda".into(), "dml".into()],
            provider_dlls: vec![
                dll(SHARED_PROVIDER_DLL, true),
                dll(CUDA_PROVIDER_DLL, cuda_provider),
                dll("DirectML.dll", true),
            ],
            directml_loaded_from: None,
            adapters,
            nvidia_driver: None,
            cuda_driver_version: None,
            cuda_runtime_dlls: cuda_runtime_dll_names().into_iter().map(|n| dll(n, runtime)).collect(),
            cudart_version: None,
            cudnn_version: None,
            nvidia_pack: NvidiaPackStatus::default(),
            nvidia_runtime_source: None,
            cuda_provider_source: None,
            cuda_provider_path: None,
            runtime_fingerprint: "fp".into(),
            ep_checks: vec![],
            recommended_ep: String::new(),
            notes: vec![],
        }
    }

    #[test]
    fn light_recommendations() {
        let nv = || vec![adapter(0, gpu_mem::VENDOR_NVIDIA, false)];
        assert_eq!(assess(&base(nv(), true, true)).0, "cuda");
        let (ep, notes) = assess(&base(nv(), true, false));
        assert_eq!(ep, "dml", "NVIDIA without the runtime pack uses DirectML");
        assert!(notes.iter().any(|n| n.contains("NVIDIA GPU runtime pack")), "{notes:?}");
        let (ep, notes) = assess(&base(nv(), false, true));
        assert_eq!(ep, "dml");
        assert!(notes.iter().any(|n| n.contains("no CUDA provider")));
        assert_eq!(assess(&base(vec![adapter(0, gpu_mem::VENDOR_AMD, false)], true, false)).0, "dml");
        let (ep, notes) = assess(&base(vec![adapter(0, gpu_mem::VENDOR_MICROSOFT, true)], true, true));
        assert_eq!(ep, "cpu");
        assert!(notes.iter().any(|n| n.contains("No hardware GPU")));
        // A clean Windows (Windows Sandbox) lists the Basic Render Driver three times, two without the DXGI software
        // flag: the startup log used to say "recommended DirectML" there while `--headless ep-probe` said CPU.
        let warp = |flagged| gpu_mem::is_software_adapter(flagged, gpu_mem::VENDOR_MICROSOFT, 0x8C, "Microsoft Basic Render Driver");
        let clean = vec![
            adapter(0, gpu_mem::VENDOR_MICROSOFT, warp(true)),
            adapter(1, gpu_mem::VENDOR_MICROSOFT, warp(false)),
            adapter(2, gpu_mem::VENDOR_MICROSOFT, warp(false)),
        ];
        let (ep, notes) = assess(&base(clean, true, true));
        assert_eq!(ep, "cpu");
        assert!(notes.iter().any(|n| n.contains("No hardware GPU")));
    }

    #[test]
    fn deep_recommendation_follows_canaries() {
        let mut r = base(vec![adapter(0, gpu_mem::VENDOR_NVIDIA, false)], true, true);
        r.deep = true;
        let check = |ep: &str, ok: bool| EpCheck {
            ep: ep.into(),
            device_id: 0,
            adapter: None,
            ok,
            millis: Some(5),
            error: (!ok).then(|| "boom".to_string()),
        };
        r.ep_checks = vec![check("cuda", false), check("dml", true)];
        let (ep, notes) = assess(&r);
        assert_eq!(ep, "dml");
        assert!(notes.iter().any(|n| n.contains("CUDA test on device 0 failed: boom")));
        r.ep_checks = vec![check("cuda", false), check("dml", false)];
        assert_eq!(assess(&r).0, "cpu");
    }

    #[test]
    fn light_probe_runs_and_round_trips() {
        let r = probe(false);
        assert!(!r.deep);
        assert!(r.ort_compiled_eps.contains(&"cpu".to_string()));
        assert_eq!(r.cuda_runtime_dlls.len(), cuda_runtime_dll_names().len());
        let json = serde_json::to_string(&r).unwrap();
        let back: EpProbeReport = serde_json::from_str(&json).unwrap();
        assert_eq!(back, r);
        #[cfg(windows)]
        assert!(r.os_version.as_deref().is_some_and(|v| v.starts_with("Windows 10.")));
    }

    #[test]
    fn deep_reports_go_stale_when_the_pack_changes() {
        let mut r = base(vec![adapter(0, gpu_mem::VENDOR_NVIDIA, false)], true, false);
        r.deep = true;
        r.app_version = env!("CARGO_PKG_VERSION").into();
        r.nvidia_pack = NvidiaPackStatus {
            label: nvidia_pack::LABEL.into(),
            ..NvidiaPackStatus::default()
        };
        let mut now = r.nvidia_pack.clone();
        assert!(deep_report_is_current(&r, &now, "fp"));
        // Progress fields (preloaded, installing, errors) do not make a report stale.
        now.preloaded = true;
        now.last_error = Some("x".into());
        assert!(deep_report_is_current(&r, &now, "fp"));
        // A model-store runtime or provider pack was installed or removed after the test.
        assert!(!deep_report_is_current(&r, &now, "fp-other"));
        now.installed = true;
        assert!(!deep_report_is_current(&r, &now, "fp"), "the pack was installed after the test");
        let mut removed = r.nvidia_pack.clone();
        r.nvidia_pack.installed = true;
        removed.pending_removal = true;
        assert!(!deep_report_is_current(&r, &removed, "fp"), "the pack was removed after the test");
        let mut other_build = r.clone();
        other_build.app_version = "0.0.0-older".into();
        assert!(!deep_report_is_current(&other_build, &r.nvidia_pack, "fp"));
        // Reports saved before the fingerprint existed are stale.
        let mut older = r.clone();
        older.runtime_fingerprint = String::new();
        assert!(!deep_report_is_current(&older, &r.nvidia_pack, "fp"));
    }

    #[test]
    fn reports_without_the_p1_fields_still_parse() {
        let r = base(vec![], true, true);
        let mut v = serde_json::to_value(&r).unwrap();
        let obj = v.as_object_mut().unwrap();
        for k in ["nvidia_runtime_source", "cuda_provider_source", "cuda_provider_path", "runtime_fingerprint"] {
            obj.remove(k);
        }
        let back: EpProbeReport = serde_json::from_value(v).unwrap();
        assert!(back.runtime_fingerprint.is_empty() && back.nvidia_runtime_source.is_none());
    }

    #[test]
    fn store_pack_dlls_are_located_before_the_interim_pack() {
        let dir = std::env::temp_dir().join(format!("sigil-ep-probe-store-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let dll = dir.join("cudnn_ops64_9.dll");
        std::fs::write(&dll, b"x").unwrap();
        let mut index = BTreeMap::new();
        index.insert("cudnn_ops64_9.dll".to_string(), dll.clone());
        let c = locate_runtime_dll("cudnn_ops64_9.dll", Some(&index), None, None);
        assert_eq!(c.source, "pack");
        assert_eq!(c.path.as_deref(), Some(dll.display().to_string().as_str()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn locate_reports_missing_dlls() {
        let c = locate_dll("definitely_not_a_real_sigil_dll_42.dll", None, None);
        assert!(!c.found() && c.path.is_none());
        #[cfg(windows)]
        assert_eq!(locate_dll("kernel32.dll", None, None).source, "loaded");
    }
}
