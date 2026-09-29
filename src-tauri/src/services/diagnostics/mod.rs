//! `diagnostics` plugin: execution-provider (EP) outcomes, the startup EP probe, the rotating file
//! log, EcoQoS opt-out, GPU memory and the interim NVIDIA runtime pack.
//!
//! Owned by P0-H; scaffolded by P0-S0 (plugin registered FIRST in main.rs so its setup hook runs
//! before every other plugin, capability `capabilities/diagnostics.json`, commands in
//! `plugin-commands/diagnostics.txt`).
//!
//! Submodules: [`logging`] (file log + UI forwarding), [`qos`] (EcoQoS opt-out), [`gpu_mem`]
//! (DXGI adapters, per-process VRAM), [`canary`] (GPU EP canary model), [`ep_probe`] (light and
//! deep probe), [`nvidia_pack`] (CUDA 13 / cuDNN 9 runtime pack).
//!
//! # Frozen contract (P0-S0; change only through an INTEGRATE step)
//! - [`EpOutcome`] `{ requested, active, reason }` — the EP a session asked for, the EP it actually
//!   runs on, and why they differ. EP ids are lower-case: `cpu`, `cuda`, `dml` (later `tensorrt`,
//!   `nvrtx`). `reason` is `None` when `requested == active`.
//! - Process-wide registry: [`record_ep_outcome`], [`ep_outcome`], [`ep_outcomes`]. Whoever builds a
//!   session (`neural_audio::ort_session`) records the outcome under a component key; consumers
//!   (voice changer metrics, UI) read it. Component keys: `vc` (voice-changer generator / main
//!   model), `vc.content` (content encoder), `vc.pitch` (F0 model), `vc.onnx` (ONNX chunk mode),
//!   `tts.kitten`, `stt.vad`; others are free-form dotted lower-case.
//! - Command `plugin:diagnostics|get_ep_outcomes` → `EpOutcomeEntry[]`
//!   (`{ component, requested, active, reason, at_unix_ms }`).
//! - Event [`EP_OUTCOME_EVENT`] (`"diagnostics:ep_outcome"`, payload `EpOutcomeEntry`), emitted by
//!   [`record_ep_outcome`].
//! - [`headless_ep_probe`]: body of `Sigil.exe --headless ep-probe`; returns the deep probe report
//!   as JSON text.
//!
//! # Commands added by P0-H (all in `plugin-commands/diagnostics.txt`)
//! `get_ep_probe`, `run_deep_ep_probe`, `get_last_deep_ep_probe`, `get_gpu_adapters`,
//! `get_log_info`, `get_recent_log_entries`, `log_from_frontend`, `open_log_dir`,
//! `get_nvidia_pack_status`, `install_nvidia_pack`, `cancel_nvidia_pack_install`,
//! `remove_nvidia_pack`. Events: [`EP_OUTCOME_EVENT`], [`logging::LOG_EVENT`],
//! [`ep_probe::PROBE_EVENT`], [`nvidia_pack::PROGRESS_EVENT`].

pub mod canary;
pub mod ep_probe;
pub mod gpu_mem;
pub mod logging;
pub mod nvidia_pack;
pub mod qos;

pub(crate) use logging::truncate_chars;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, Runtime,
};

/// Tauri event emitted when an EP outcome is recorded (payload [`EpOutcomeEntry`]).
pub const EP_OUTCOME_EVENT: &str = "diagnostics:ep_outcome";
/// Environment variable naming the app-data directory for headless children (and overrides).
pub const APP_DATA_ENV: &str = "SIGIL_APP_DATA_DIR";
/// Bundle identifier (tauri.conf.json) — the app-data folder name under `%APPDATA%`.
const APP_IDENTIFIER: &str = "com.sigil.app";

/// The execution provider a session requested vs. the one it actually runs on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpOutcome {
    /// Requested EP (`cpu` / `cuda` / `dml` / ...).
    pub requested: String,
    /// EP actually in use.
    pub active: String,
    /// Why `active` differs from `requested` (driver missing, DLL not found, ...).
    pub reason: Option<String>,
}

impl EpOutcome {
    /// Outcome with an explicit active EP and reason.
    #[allow(dead_code)] // Frozen P0-S0 contract; for session builders outside ort_session.
    pub fn new(requested: impl Into<String>, active: impl Into<String>, reason: Option<String>) -> Self {
        Self {
            requested: requested.into(),
            active: active.into(),
            reason,
        }
    }

    /// The requested EP is the active one.
    pub fn granted(ep: impl Into<String>) -> Self {
        let ep = ep.into();
        Self {
            requested: ep.clone(),
            active: ep,
            reason: None,
        }
    }

    /// A GPU EP failed and the session fell back to `active` (usually `cpu`).
    pub fn fallback(requested: impl Into<String>, active: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            requested: requested.into(),
            active: active.into(),
            reason: Some(reason.into()),
        }
    }

    /// Whether the session runs on a different EP than requested.
    pub fn fell_back(&self) -> bool {
        self.requested != self.active
    }

    /// Whether the active EP is a GPU EP.
    pub fn is_gpu(&self) -> bool {
        self.active != "cpu"
    }
}

/// One registry row, as returned by `get_ep_outcomes`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct EpOutcomeEntry {
    pub component: String,
    #[serde(flatten)]
    pub outcome: EpOutcome,
    /// When the outcome was recorded (Unix epoch milliseconds).
    pub at_unix_ms: u64,
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

fn registry() -> &'static Mutex<BTreeMap<String, EpOutcomeEntry>> {
    static REGISTRY: OnceLock<Mutex<BTreeMap<String, EpOutcomeEntry>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(BTreeMap::new()))
}

// ---------------------------------------------------------------------------------------------
// Event emission (installed in on_setup; a no-op in tests and headless runs)
// ---------------------------------------------------------------------------------------------

type EmitFn = Box<dyn Fn(&str, serde_json::Value) + Send + Sync>;
static EMITTER: OnceLock<EmitFn> = OnceLock::new();

/// Emit a Tauri event to the webview, if the app is running. Never blocks on the UI.
pub(crate) fn emit(event: &str, payload: &impl Serialize) {
    if let Some(emit) = EMITTER.get() {
        if let Ok(value) = serde_json::to_value(payload) {
            emit(event, value);
        }
    }
}

/// Record the EP outcome of `component`, replacing the previous one, and report it (log line and
/// [`EP_OUTCOME_EVENT`]). Never call it from an audio callback (it takes a lock and logs).
pub fn record_ep_outcome(component: &str, outcome: EpOutcome) {
    let entry = EpOutcomeEntry {
        component: component.to_string(),
        outcome,
        at_unix_ms: logging::unix_ms_now(),
    };
    lock(registry()).insert(component.to_string(), entry.clone());
    let o = &entry.outcome;
    if o.fell_back() {
        tracing::warn!(
            target: "sigil::ort",
            "[{component}] requested {} but running on {}: {}",
            o.requested,
            o.active,
            o.reason.as_deref().unwrap_or("no reason given")
        );
    } else {
        tracing::info!(target: "sigil::ort", "[{component}] running on {}", o.active);
    }
    emit(EP_OUTCOME_EVENT, &entry);
}

/// Latest outcome of `component`, if any session recorded one.
#[allow(dead_code)] // Frozen P0-S0 contract (consumers read single components; tests use it).
pub fn ep_outcome(component: &str) -> Option<EpOutcome> {
    lock(registry()).get(component).map(|e| e.outcome.clone())
}

/// Every recorded outcome, sorted by component.
pub fn ep_outcomes() -> Vec<EpOutcomeEntry> {
    lock(registry()).values().cloned().collect()
}

// ---------------------------------------------------------------------------------------------
// App data (Tauri's app_data_dir once set up; headless children get it through APP_DATA_ENV)
// ---------------------------------------------------------------------------------------------

static APP_DATA: OnceLock<PathBuf> = OnceLock::new();

/// Sigil's app-data directory: the running app's (set in setup), else `SIGIL_APP_DATA_DIR`, else
/// `%APPDATA%\com.sigil.app`.
pub fn app_data_dir() -> Option<PathBuf> {
    if let Some(d) = APP_DATA.get() {
        return Some(d.clone());
    }
    if let Some(d) = std::env::var_os(APP_DATA_ENV).filter(|v| !v.is_empty()) {
        return Some(PathBuf::from(d));
    }
    std::env::var_os("APPDATA")
        .filter(|v| !v.is_empty())
        .map(|d| PathBuf::from(d).join(APP_IDENTIFIER))
}

// ---------------------------------------------------------------------------------------------
// Probe state
// ---------------------------------------------------------------------------------------------

fn light_probe() -> &'static Mutex<Option<ep_probe::EpProbeReport>> {
    static P: OnceLock<Mutex<Option<ep_probe::EpProbeReport>>> = OnceLock::new();
    P.get_or_init(|| Mutex::new(None))
}

static DEEP_PROBE_RUNNING: AtomicBool = AtomicBool::new(false);

/// Body of `Sigil.exe --headless ep-probe`: the deep EP / driver / DXGI / CUDA probe as JSON text.
/// Loads GPU runtimes and runs canaries, which is why it runs in its own process.
pub fn headless_ep_probe() -> Result<String, String> {
    let report = ep_probe::probe(true);
    serde_json::to_string_pretty(&report).map_err(|e| format!("cannot serialize the report: {e}"))
}

fn require_app_data() -> Result<PathBuf, String> {
    app_data_dir().ok_or_else(|| "the app-data directory is unknown".to_string())
}

/// Recompute the light probe (the NVIDIA runtime pack was installed or removed), replace the
/// cached report and emit it as [`ep_probe::PROBE_EVENT`], so "Best available" and the notes
/// match the new pack state without a restart.
async fn refresh_light_probe() {
    match tauri::async_runtime::spawn_blocking(|| ep_probe::probe(false)).await {
        Ok(report) => {
            ep_probe::log_summary(&report);
            *lock(light_probe()) = Some(report.clone());
            emit(ep_probe::PROBE_EVENT, &report);
        }
        Err(e) => tracing::warn!(target: "sigil::diagnostics", "cannot refresh the EP probe: {e}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

#[command]
fn get_ep_outcomes() -> Vec<EpOutcomeEntry> {
    ep_outcomes()
}

/// The startup (light) probe; computed now if the startup thread has not finished yet. Recomputed
/// (and re-emitted) whenever the NVIDIA runtime pack is installed or removed.
#[command]
async fn get_ep_probe() -> Result<ep_probe::EpProbeReport, String> {
    if let Some(r) = lock(light_probe()).clone() {
        return Ok(r);
    }
    let report = tauri::async_runtime::spawn_blocking(|| ep_probe::probe(false))
        .await
        .map_err(|e| format!("probe task: {e}"))?;
    *lock(light_probe()) = Some(report.clone());
    Ok(report)
}

/// Run the deep probe in a headless child process (CUDA / DirectML canaries). One at a time.
#[command]
async fn run_deep_ep_probe() -> Result<ep_probe::EpProbeReport, String> {
    if DEEP_PROBE_RUNNING.swap(true, Ordering::AcqRel) {
        return Err("a GPU check is already running".to_string());
    }
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            DEEP_PROBE_RUNNING.store(false, Ordering::Release);
        }
    }
    let _reset = Reset;
    let app_data = require_app_data()?;
    let result = tauri::async_runtime::spawn_blocking(move || ep_probe::run_deep_probe_child(&app_data))
        .await
        .map_err(|e| format!("GPU check task: {e}"))?;
    // An explicit retry: the child's canaries never touch this process's cache, so forget the
    // failures cached here too (a game released the VRAM, the driver recovered); the next session
    // runs those canaries again.
    canary::forget_failures();
    let report = result?;
    ep_probe::log_summary(&report);
    Ok(report)
}

/// The last deep probe, unless it no longer describes this install (another Sigil version, or the
/// NVIDIA runtime pack was installed or removed since): then the UI offers a new test instead.
#[command]
fn get_last_deep_ep_probe() -> Option<ep_probe::EpProbeReport> {
    let app_data = app_data_dir()?;
    let report = ep_probe::last_deep_report(&app_data)?;
    let fingerprint = crate::services::native_runtime::nvidia::fingerprint();
    ep_probe::deep_report_is_current(&report, &nvidia_pack::status(Some(&app_data)), &fingerprint).then_some(report)
}

#[command]
async fn get_gpu_adapters() -> Result<Vec<gpu_mem::GpuAdapter>, String> {
    tauri::async_runtime::spawn_blocking(gpu_mem::adapters)
        .await
        .map_err(|e| format!("adapter task: {e}"))
}

#[command]
fn get_log_info() -> Option<logging::LogInfo> {
    logging::info()
}

#[command]
fn get_recent_log_entries() -> Vec<logging::LogEntry> {
    logging::recent_entries()
}

/// Write a system-log line from the webview into the file log (warnings and errors).
#[command]
fn log_from_frontend(level: String, tag: String, message: String) {
    logging::log_frontend(&level, &tag, &message);
}

#[command]
fn open_log_dir() -> Result<(), String> {
    let info = logging::info().ok_or_else(|| "the file log is not active".to_string())?;
    #[cfg(windows)]
    let opener = "explorer.exe";
    #[cfg(target_os = "macos")]
    let opener = "open";
    #[cfg(all(not(windows), not(target_os = "macos")))]
    let opener = "xdg-open";
    std::process::Command::new(opener)
        .arg(&info.dir)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("cannot open {}: {e}", info.dir))
}

#[command]
fn get_nvidia_pack_status() -> nvidia_pack::NvidiaPackStatus {
    nvidia_pack::status(app_data_dir().as_deref())
}

/// Download (~1.9 GB, one file at a time) and install the NVIDIA runtime pack. Progress arrives as
/// [`nvidia_pack::PROGRESS_EVENT`]; the result is the final status.
#[command]
async fn install_nvidia_pack() -> Result<nvidia_pack::NvidiaPackStatus, String> {
    let app_data = require_app_data()?;
    let status = nvidia_pack::install(app_data, |p| emit(nvidia_pack::PROGRESS_EVENT, &p)).await?;
    refresh_light_probe().await;
    Ok(status)
}

#[command]
fn cancel_nvidia_pack_install() {
    nvidia_pack::cancel();
}

#[command]
async fn remove_nvidia_pack() -> Result<nvidia_pack::NvidiaPackStatus, String> {
    let app_data = require_app_data()?;
    let status = tauri::async_runtime::spawn_blocking(move || nvidia_pack::remove(&app_data))
        .await
        .map_err(|e| format!("remove task: {e}"))??;
    refresh_light_probe().await;
    Ok(status)
}

// ---------------------------------------------------------------------------------------------
// Setup
// ---------------------------------------------------------------------------------------------

/// Runs in this plugin's setup. main.rs registers the diagnostics plugin FIRST, so this precedes
/// every other plugin's setup (and the webview): file log, UI forwarding, EcoQoS opt-out, then the
/// runtime-pack sweep and the light EP probe on a background thread. Never fails or blocks.
fn on_setup<R: Runtime>(app: &AppHandle<R>) {
    let paths = app.path();
    if let Ok(dir) = paths.app_data_dir() {
        let _ = APP_DATA.set(dir);
    }
    let log_dir = paths
        .app_log_dir()
        .ok()
        .or_else(|| app_data_dir().map(|d| d.join("logs")));
    match log_dir.as_deref().map(logging::init) {
        Some(Ok(info)) => tracing::info!(target: "sigil::diagnostics", "logging to {}", info.file),
        Some(Err(e)) => eprintln!("[diagnostics] file log disabled: {e}"),
        None => eprintln!("[diagnostics] file log disabled: no log directory"),
    }

    let handle = app.clone();
    let _ = EMITTER.set(Box::new(move |event, payload| {
        let _ = handle.emit(event, payload);
    }));
    logging::set_forwarder(|entry| emit(logging::LOG_EVENT, entry));

    qos::opt_out_power_throttling();

    let spawned = std::thread::Builder::new().name("sigil-ep-probe".into()).spawn(|| {
        if let Some(d) = app_data_dir() {
            nvidia_pack::sweep(&d);
        }
        let report = ep_probe::probe(false);
        ep_probe::log_summary(&report);
        *lock(light_probe()) = Some(report.clone());
        emit(ep_probe::PROBE_EVENT, &report);
    });
    if let Err(e) = spawned {
        tracing::warn!(target: "sigil::diagnostics", "cannot start the EP probe thread: {e}");
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("diagnostics")
        .setup(|app, _api| {
            on_setup(app);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            get_ep_outcomes,
            get_ep_probe,
            run_deep_ep_probe,
            get_last_deep_ep_probe,
            get_gpu_adapters,
            get_log_info,
            get_recent_log_entries,
            log_from_frontend,
            open_log_dir,
            get_nvidia_pack_status,
            install_nvidia_pack,
            cancel_nvidia_pack_install,
            remove_nvidia_pack
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::{app_data_dir, ep_outcome, ep_outcomes, headless_ep_probe, record_ep_outcome, EpOutcome};

    #[test]
    fn ep_outcome_registry_and_wire_format() {
        let o = EpOutcome::fallback("cuda", "cpu", "onnxruntime_providers_cuda.dll not found");
        assert!(o.fell_back() && !o.is_gpu());
        assert!(!EpOutcome::granted("dml").fell_back());

        record_ep_outcome("test.diagnostics", o.clone());
        assert_eq!(ep_outcome("test.diagnostics"), Some(o));
        let row = ep_outcomes().into_iter().find(|e| e.component == "test.diagnostics").unwrap();
        let json = serde_json::to_value(&row).unwrap();
        assert_eq!(json["component"], "test.diagnostics");
        assert_eq!(json["requested"], "cuda");
        assert_eq!(json["active"], "cpu");
        assert_eq!(json["reason"], "onnxruntime_providers_cuda.dll not found");
        assert!(json["at_unix_ms"].as_u64().unwrap() > 0);
    }

    #[test]
    fn app_data_resolves_without_tauri() {
        // Headless children and tests have no AppHandle; %APPDATA% (always set on Windows) or the
        // override must resolve.
        #[cfg(windows)]
        assert!(app_data_dir().is_some());
    }

    /// The deep probe runs canaries on real GPUs: `cargo test headless_probe -- --ignored`.
    #[test]
    #[ignore]
    fn headless_probe_emits_a_deep_report() {
        let json = headless_ep_probe().expect("probe");
        let v: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(v["deep"], true);
        eprintln!("{json}");
    }
}
