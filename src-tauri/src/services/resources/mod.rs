//! `resources` plugin: the VRAM budget and resource governor. The VRAM budget comes from DXGI
//! `QueryVideoMemoryInfo`; subsystems claim VRAM (vc, stt_final, llm, tts); the voice changer sets a
//! running flag; a deterministic, pure recommendation API picks models that fit (e.g. Hy-MT2 7B
//! only if the VC is off on 8 GB; Qwen3-TTS 0.6B while the VC runs).
//!
//! - [`ledger`]: the claims ledger (plain data).
//! - [`policy`]: the recommendation rules and the catalog ids they return ([`policy::ids`]).
//!
//! # For subsystems that load GPU models
//! - Before loading, `let claim = resources::claim_vram(Subsystem::Vc, "rvc: <voice>", estimate_mb)`
//!   (in-process sessions) or [`claim_external_vram`] (helper processes such as llama-server, whose
//!   memory is not this process's). Keep the handle alive while the model is loaded; dropping it
//!   releases the claim. Call [`VramClaim::resize`] once the real footprint is known.
//! - A model running on the CPU claims 0 MB (that tells the policy the subsystem is not on the GPU).
//! - The voice changer calls [`set_vc_running`]`(true)` when a session starts and `false` when it
//!   stops, and [`set_vc_gpu`] once its models are loaded (false for the CPU EP and DSP-only
//!   presets). While it runs on a GPU EP without a claim, the policy assumes
//!   [`policy::VC_ESTIMATE_MB`]; on the CPU (or not known to be on a GPU) it assumes 0 MB. Until
//!   the voice changer reports, [`vc_gpu`] infers the device from the EP outcomes its model
//!   sessions recorded since it started (components `vc`, `vc.*`).
//! - Ask [`recommend`] (or IPC `get_recommendation`) before choosing a model size.
//!
//! # Frozen contract (P1-S0b; change only through INTEGRATE; enums and structs may gain members)
//! - [`claim_vram`]`(subsystem, label, mb) -> VramClaim`: RAII claim, released on drop;
//!   [`VramClaim::resize`]. Never blocks on I/O (safe from worker threads, never from audio
//!   callbacks).
//! - [`set_vc_running`] / [`vc_running`]: set by the voice changer on start / stop.
//!   [`set_vc_gpu`] / [`vc_gpu`] (P1-D2 addition): whether its models run on a GPU EP.
//! - [`budget`]`() -> VramBudget`, [`snapshot`]`() -> ResourceSnapshot`.
//! - [`recommend`]`(topic, &SystemInfo, &ResourceSnapshot) -> Recommendation`: pure and
//!   deterministic (same inputs, same answer; unit-tested as a policy table).
//! - IPC `plugin:resources|get_resource_snapshot -> ResourceSnapshot` and
//!   `plugin:resources|get_recommendation({ topic }) -> Recommendation`.
//! - Event [`RESOURCES_EVENT`] `"resources:changed"` (payload [`ResourceSnapshot`]) when claims or
//!   the VC flag change (throttled to [`EVENT_THROTTLE`]; only while the app runs).

pub mod ledger;
pub mod policy;

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::diagnostics::{self, gpu_mem, logging, EpOutcomeEntry};
use crate::services::sysinfo::{self, nvml, GpuInfo, SystemInfo};

/// Event emitted when claims or the VC-running flag change (payload [`ResourceSnapshot`]).
pub const RESOURCES_EVENT: &str = "resources:changed";

/// Bursts of claim changes (a voice changer loading three sessions) coalesce into one event.
pub const EVENT_THROTTLE: Duration = Duration::from_millis(250);

/// Who holds a VRAM claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Subsystem {
    Vc,
    SttFinal,
    Llm,
    Tts,
    Other,
}

/// VRAM budget of the primary GPU, in MiB.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramBudget {
    pub adapter_index: Option<u32>,
    pub adapter_name: Option<String>,
    /// Dedicated VRAM.
    pub total_mb: u64,
    /// What the OS lets this process use (`QueryVideoMemoryInfo` Budget; the dedicated VRAM when
    /// the query is unavailable).
    pub budget_mb: u64,
    /// This process's current usage.
    pub process_usage_mb: u64,
    /// Sum of live claims (in-process and helper processes).
    pub claimed_mb: u64,
    /// Budget minus the larger of this process's usage and its in-process claims, minus the claims
    /// of helper processes (their memory never shows in this process's usage).
    pub free_for_claims_mb: u64,
    /// Part of `claimed_mb` held by helper processes.
    #[serde(default)]
    pub external_claimed_mb: u64,
    /// Device-wide use by every process (games, OBS, Sigil), from NVML; `None` for non-NVIDIA GPUs.
    #[serde(default)]
    pub device_used_mb: Option<u64>,
    /// Device-wide free memory, from NVML.
    #[serde(default)]
    pub device_free_mb: Option<u64>,
}

/// One live claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VramClaimInfo {
    pub id: u64,
    pub subsystem: Subsystem,
    pub label: String,
    pub mb: u64,
    /// Held by a helper process (llama-server, a TTS server), not by Sigil itself.
    #[serde(default)]
    pub external: bool,
}

/// Budget, claims and flags at one point in time.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceSnapshot {
    pub budget: VramBudget,
    pub claims: Vec<VramClaimInfo>,
    pub vc_running: bool,
    /// Whether the running voice changer's models use a GPU EP ([`vc_gpu`]); `None` while it is
    /// not running or not known yet (a DSP-only preset, models still loading).
    #[serde(default)]
    pub vc_gpu: Option<bool>,
}

// ---------------------------------------------------------------------------------------------
// Ledger, VC flag and change notification
// ---------------------------------------------------------------------------------------------

static LEDGER: Mutex<ledger::Ledger> = Mutex::new(ledger::Ledger::new());
static VC_RUNNING: AtomicBool = AtomicBool::new(false);
/// The voice changer's own device report ([`set_vc_gpu`]): one of the `VC_GPU_*` values.
static VC_GPU: AtomicU8 = AtomicU8::new(VC_GPU_UNREPORTED);
/// When the current voice-changer session started (Unix ms): EP outcomes older than this belong to
/// an earlier session.
static VC_STARTED_MS: AtomicU64 = AtomicU64::new(0);
const VC_GPU_UNREPORTED: u8 = 0;
const VC_GPU_NO: u8 = 1;
const VC_GPU_YES: u8 = 2;

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Set in the plugin's setup: events are only emitted while the app runs (tests and headless runs
/// never start the notifier thread).
static EVENTS_ENABLED: AtomicBool = AtomicBool::new(false);

struct Notify {
    dirty: bool,
    started: bool,
}

static NOTIFY: Mutex<Notify> = Mutex::new(Notify {
    dirty: false,
    started: false,
});
static NOTIFY_CV: Condvar = Condvar::new();

/// Something changed: wake the notifier (lock + flag + condvar only; never I/O).
fn changed() {
    if !EVENTS_ENABLED.load(Ordering::Acquire) {
        return;
    }
    let mut n = lock(&NOTIFY);
    n.dirty = true;
    if !n.started {
        n.started = std::thread::Builder::new()
            .name("sigil-resources-events".into())
            .spawn(notifier_loop)
            .map_err(|e| tracing::warn!(target: "sigil::resources", "cannot start the event thread: {e}"))
            .is_ok();
    }
    NOTIFY_CV.notify_one();
}

fn notifier_loop() {
    loop {
        {
            let mut n = lock(&NOTIFY);
            while !n.dirty {
                n = NOTIFY_CV.wait(n).unwrap_or_else(PoisonError::into_inner);
            }
        }
        // Coalesce a burst, then report the state as of now.
        std::thread::sleep(EVENT_THROTTLE);
        lock(&NOTIFY).dirty = false;
        diagnostics::emit(RESOURCES_EVENT, &snapshot());
    }
}

/// RAII VRAM claim: released when dropped.
#[derive(Debug)]
pub struct VramClaim {
    id: u64,
}

#[allow(dead_code)] // Public API for claim holders (P2/P3/P4/P6).
impl VramClaim {
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Change the claimed amount (e.g. after a session reports its real footprint).
    pub fn resize(&self, mb: u64) {
        let did_change = lock(&LEDGER).resize(self.id, mb);
        if did_change {
            changed();
        }
    }
}

impl Drop for VramClaim {
    fn drop(&mut self) {
        let removed = lock(&LEDGER).remove(self.id);
        if removed {
            changed();
        }
    }
}

fn claim(subsystem: Subsystem, label: &str, mb: u64, external: bool) -> VramClaim {
    let id = lock(&LEDGER).insert(subsystem, label, mb, external);
    changed();
    VramClaim { id }
}

/// Register a claim of `mb` MiB for `subsystem`, for memory this process allocates (ORT sessions,
/// in-process engines).
#[allow(dead_code)] // Public API: the voice changer (P3) and in-process STT/TTS engines (P2) claim.
pub fn claim_vram(subsystem: Subsystem, label: &str, mb: u64) -> VramClaim {
    claim(subsystem, label, mb, false)
}

/// Register a claim for memory a helper process allocates (llama-server, a TTS server): it counts
/// against the budget but never appears in this process's usage.
#[allow(dead_code)] // Public API: llama-server (P4-B) and TTS helper processes (P6) claim.
pub fn claim_external_vram(subsystem: Subsystem, label: &str, mb: u64) -> VramClaim {
    claim(subsystem, label, mb, true)
}

/// Set by the voice changer when a session starts / stops. Every call starts a new session for
/// [`vc_gpu`] (a start while running is a restart with possibly another model or EP).
pub fn set_vc_running(running: bool) {
    if running {
        VC_STARTED_MS.store(logging::unix_ms_now(), Ordering::Release);
    }
    let was_reported = VC_GPU.swap(VC_GPU_UNREPORTED, Ordering::AcqRel) != VC_GPU_UNREPORTED;
    if VC_RUNNING.swap(running, Ordering::AcqRel) != running || was_reported {
        changed();
    }
}

pub fn vc_running() -> bool {
    VC_RUNNING.load(Ordering::Acquire)
}

/// Reported by the voice changer once its models are loaded: whether they run on a GPU EP (the
/// CPU EP and DSP-only presets report `false`). Cleared by every [`set_vc_running`] call.
pub fn set_vc_gpu(on_gpu: bool) {
    let value = if on_gpu { VC_GPU_YES } else { VC_GPU_NO };
    if VC_GPU.swap(value, Ordering::AcqRel) != value {
        changed();
    }
}

/// EP components of voice-changer model sessions (`vc`, `vc.content`, `vc.pitch`, `vc.onnx`).
fn is_vc_component(component: &str) -> bool {
    component == "vc" || component.starts_with("vc.")
}

/// Whether voice-changer model sessions recorded since `since_ms` run on a GPU EP: `Some(true)` if
/// any does, `Some(false)` if all run on the CPU, `None` without such an outcome. Pure.
pub(crate) fn infer_vc_gpu(outcomes: &[EpOutcomeEntry], since_ms: u64) -> Option<bool> {
    let mut seen = false;
    for e in outcomes
        .iter()
        .filter(|e| e.at_unix_ms >= since_ms && is_vc_component(&e.component))
    {
        if e.outcome.is_gpu() {
            return Some(true);
        }
        seen = true;
    }
    seen.then_some(false)
}

/// Whether the running voice changer's models use a GPU EP: its own report ([`set_vc_gpu`]), else
/// inferred from the EP outcomes its sessions recorded since it started; `None` while it is not
/// running or when neither says (a DSP-only preset, models still loading).
pub fn vc_gpu() -> Option<bool> {
    if !vc_running() {
        return None;
    }
    match VC_GPU.load(Ordering::Acquire) {
        VC_GPU_YES => Some(true),
        VC_GPU_NO => Some(false),
        _ => infer_vc_gpu(&diagnostics::ep_outcomes(), VC_STARTED_MS.load(Ordering::Acquire)),
    }
}

// ---------------------------------------------------------------------------------------------
// Budget and snapshot
// ---------------------------------------------------------------------------------------------

/// Pure budget arithmetic (unit-tested): `usage` is this process's DXGI view of the primary GPU,
/// `device` NVML's device-wide view.
pub(crate) fn compute_budget(
    primary: Option<&GpuInfo>,
    usage: Option<gpu_mem::VramUsage>,
    device: Option<nvml::NvmlMemory>,
    in_process_claims_mb: u64,
    external_claims_mb: u64,
) -> VramBudget {
    let total_mb = primary.map(|g| g.adapter.dedicated_vram_mb).unwrap_or(0);
    let budget_mb = usage
        .map(|u| u.budget_mb)
        .filter(|&b| b > 0)
        .unwrap_or(total_mb);
    let process_usage_mb = usage.map(|u| u.process_usage_mb).unwrap_or(0);
    let held = process_usage_mb
        .max(in_process_claims_mb)
        .saturating_add(external_claims_mb);
    VramBudget {
        adapter_index: primary.map(|g| g.adapter.index),
        adapter_name: primary.map(|g| g.adapter.name.clone()),
        total_mb,
        budget_mb,
        process_usage_mb,
        claimed_mb: in_process_claims_mb.saturating_add(external_claims_mb),
        free_for_claims_mb: budget_mb.saturating_sub(held),
        external_claimed_mb: external_claims_mb,
        device_used_mb: device.map(|d| d.used_mb),
        device_free_mb: device.map(|d| d.free_mb),
    }
}

/// This process's live VRAM use on `gpu`, found by LUID (DXGI indices can shift when an adapter
/// appears or disappears), else by index.
fn live_usage(gpu: &GpuInfo) -> Option<gpu_mem::VramUsage> {
    let adapters = gpu_mem::adapters();
    adapters
        .iter()
        .find(|a| a.luid == gpu.adapter.luid)
        .or_else(|| {
            adapters
                .iter()
                .find(|a| a.index == gpu.adapter.index && a.name == gpu.adapter.name)
        })
        .and_then(|a| a.vram)
}

fn budget_for(sys: &SystemInfo) -> VramBudget {
    let primary = sys.primary();
    let usage = primary.and_then(live_usage);
    let device = primary
        .and_then(|g| g.nvml_index)
        .and_then(nvml::memory_if_loaded);
    let (inner, external) = lock(&LEDGER).totals();
    compute_budget(primary, usage, device, inner, external)
}

/// The primary GPU's VRAM budget right now (probes the system on first use).
#[allow(dead_code)] // Frozen P1-S0b API for claim holders; `snapshot()` shares `budget_for`.
pub fn budget() -> VramBudget {
    budget_for(&sysinfo::cached())
}

/// Budget, claims and the VC flags right now, for the GPU of `sys`.
fn snapshot_for(sys: &SystemInfo) -> ResourceSnapshot {
    ResourceSnapshot {
        budget: budget_for(sys),
        claims: lock(&LEDGER).list(),
        vc_running: vc_running(),
        vc_gpu: vc_gpu(),
    }
}

/// Budget, claims and the VC flags right now.
pub fn snapshot() -> ResourceSnapshot {
    snapshot_for(&sysinfo::cached())
}

// ---------------------------------------------------------------------------------------------
// Recommendations
// ---------------------------------------------------------------------------------------------

/// What to recommend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecommendationTopic {
    /// Local LLM translation model (Hy-MT2 1.8B / 7B, ...).
    MtLlm,
    /// Qwen3-TTS size (0.6B while the VC runs on small cards, 1.7B otherwise).
    Qwen3Tts,
    /// Second-pass STT finals model.
    SttFinal,
}

impl RecommendationTopic {
    pub const ALL: [RecommendationTopic; 3] = [RecommendationTopic::MtLlm, RecommendationTopic::Qwen3Tts, RecommendationTopic::SttFinal];
}

/// A deterministic recommendation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Recommendation {
    pub topic: RecommendationTopic,
    /// Catalog id (or preset value) to use ([`policy::ids`]).
    pub choice: String,
    /// Plain-English reason (logged; the UI translates by topic/choice and `reason_code`).
    pub reason: String,
    /// Other acceptable choices, best first (all fit right now).
    pub alternatives: Vec<String>,
    /// Machine-readable reason ([`policy::Reason`]): `fits`, `vc_running`, `shared_gpu`,
    /// `small_gpu`, `no_gpu`, `integrated_gpu`, `gpu_full`, `low_memory`, `no_gpu_runtime`,
    /// `gpu_unknown`.
    #[serde(default)]
    pub reason_code: String,
    /// Where the choice runs: `gpu` or `cpu`.
    #[serde(default)]
    pub device: String,
    /// Estimated VRAM the choice needs on the GPU (0 on the CPU), MiB.
    #[serde(default)]
    pub vram_mb: u64,
    /// VRAM the policy found available for this topic, MiB.
    #[serde(default)]
    pub available_mb: u64,
}

/// Pure, deterministic policy: the same inputs always give the same answer (see [`policy`]).
pub fn recommend(topic: RecommendationTopic, sys: &SystemInfo, snapshot: &ResourceSnapshot) -> Recommendation {
    policy::recommend(topic, sys, snapshot)
}

/// Body of `Sigil.exe --headless sysinfo`: system probe, VRAM budget, claims and every
/// recommendation as JSON.
pub fn headless_report() -> Result<String, String> {
    let system = sysinfo::cached();
    let snap = snapshot_for(&system);
    let recommendations: Vec<Recommendation> = RecommendationTopic::ALL
        .iter()
        .map(|&t| recommend(t, &system, &snap))
        .collect();
    serde_json::to_string_pretty(&serde_json::json!({
        "system": system,
        "resources": snap,
        "recommendations": recommendations,
    }))
    .map_err(|e| format!("cannot serialize the report: {e}"))
}

// ---------------------------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------------------------

#[command]
async fn get_resource_snapshot() -> Result<ResourceSnapshot, String> {
    tauri::async_runtime::spawn_blocking(snapshot)
        .await
        .map_err(|e| format!("resource snapshot task: {e}"))
}

#[command]
async fn get_recommendation(topic: RecommendationTopic) -> Result<Recommendation, String> {
    tauri::async_runtime::spawn_blocking(move || {
        let sys = sysinfo::cached();
        recommend(topic, &sys, &snapshot_for(&sys))
    })
    .await
    .map_err(|e| format!("recommendation task: {e}"))
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("resources")
        .setup(|_app, _api| {
            EVENTS_ENABLED.store(true, Ordering::Release);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_resource_snapshot, get_recommendation])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::gpu_mem::VENDOR_NVIDIA;
    use crate::services::sysinfo::test_support::gpu;

    #[test]
    fn budget_arithmetic() {
        let g = gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", 24_142, Some("8.9"), Some("ada"));
        let usage = gpu_mem::VramUsage {
            budget_mb: 22_000,
            process_usage_mb: 900,
            available_for_reservation_mb: 11_000,
        };
        let device = nvml::NvmlMemory {
            total_mb: 24_564,
            used_mb: 6_000,
            free_mb: 18_564,
        };
        // In-process claims above usage (a session still loading) count; helper claims add on top.
        let b = compute_budget(Some(&g), Some(usage), Some(device), 1_200, 5_300);
        assert_eq!(b.adapter_index, Some(0));
        assert_eq!(b.adapter_name.as_deref(), Some("NVIDIA GeForce RTX 4090"));
        assert_eq!(b.total_mb, 24_142);
        assert_eq!(b.budget_mb, 22_000);
        assert_eq!(b.process_usage_mb, 900);
        assert_eq!(b.claimed_mb, 6_500);
        assert_eq!(b.external_claimed_mb, 5_300);
        assert_eq!(b.free_for_claims_mb, 22_000 - 1_200 - 5_300);
        assert_eq!((b.device_used_mb, b.device_free_mb), (Some(6_000), Some(18_564)));

        // Usage above claims (unclaimed sessions) counts instead.
        let b = compute_budget(
            Some(&g),
            Some(gpu_mem::VramUsage {
                process_usage_mb: 3_000,
                ..usage
            }),
            None,
            1_200,
            0,
        );
        assert_eq!(b.free_for_claims_mb, 19_000);
        assert_eq!(b.device_used_mb, None);

        // No DXGI budget: the dedicated VRAM stands in. No GPU: all zero, never a panic.
        let b = compute_budget(Some(&g), None, None, 0, 0);
        assert_eq!((b.budget_mb, b.free_for_claims_mb), (24_142, 24_142));
        let b = compute_budget(None, None, None, 500, 0);
        assert_eq!((b.total_mb, b.budget_mb, b.free_for_claims_mb, b.claimed_mb), (0, 0, 0, 500));
        // Over-claimed never underflows.
        let b = compute_budget(Some(&g), Some(usage), None, 30_000, 30_000);
        assert_eq!(b.free_for_claims_mb, 0);
    }

    #[test]
    fn raii_claims_and_vc_flag() {
        let label = "test: raii claim";
        let find = |id: u64| lock(&LEDGER).list().into_iter().find(|c| c.id == id);
        let a = claim_vram(Subsystem::SttFinal, label, 2_700);
        let b = claim_external_vram(Subsystem::Llm, "test: helper", 5_300);
        assert_ne!(a.id(), b.id());
        assert_eq!(find(a.id()).map(|c| (c.mb, c.external)), Some((2_700, false)));
        assert_eq!(find(b.id()).map(|c| (c.subsystem, c.external)), Some((Subsystem::Llm, true)));
        a.resize(2_900);
        assert_eq!(find(a.id()).map(|c| c.mb), Some(2_900));
        let (ida, idb) = (a.id(), b.id());
        drop(a);
        assert!(find(ida).is_none());
        assert!(find(idb).is_some());
        drop(b);
        assert!(find(idb).is_none());

        // Claims from worker threads (the voice changer, helpers) are safe.
        let handles: Vec<_> = (0..8)
            .map(|i| std::thread::spawn(move || claim_vram(Subsystem::Other, "test: thread", i)))
            .collect();
        let claims: Vec<VramClaim> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let ids: Vec<u64> = claims.iter().map(VramClaim::id).collect();
        assert!(ids.iter().all(|&id| find(id).is_some()));
        drop(claims);
        assert!(ids.iter().all(|&id| find(id).is_none()));

        set_vc_running(true);
        assert!(vc_running());
        set_vc_gpu(true);
        assert_eq!(vc_gpu(), Some(true), "the voice changer's own report wins");
        set_vc_gpu(false);
        assert_eq!(vc_gpu(), Some(false));
        set_vc_running(true); // a restart clears the report
        assert_ne!(VC_GPU.load(Ordering::Acquire), VC_GPU_NO);
        set_vc_running(false);
        assert!(!vc_running());
        assert_eq!(vc_gpu(), None, "not running: no device");
    }

    #[test]
    fn vc_device_is_inferred_from_its_ep_outcomes() {
        use crate::services::diagnostics::EpOutcome;
        let entry = |component: &str, outcome: EpOutcome, at_unix_ms: u64| EpOutcomeEntry {
            component: component.into(),
            outcome,
            at_unix_ms,
        };
        let start = 1_000;
        // RVC on CUDA (content on the GPU, pitch fell back): on the GPU.
        let rvc_cuda = [
            entry("vc.content", EpOutcome::granted("cuda"), 1_200),
            entry("vc.pitch", EpOutcome::fallback("cuda", "cpu", "x"), 1_300),
            entry("stt.vad", EpOutcome::granted("cpu"), 1_100),
        ];
        assert_eq!(infer_vc_gpu(&rvc_cuda, start), Some(true));
        // RVC on the CPU EP: not on the GPU.
        let rvc_cpu = [
            entry("vc", EpOutcome::granted("cpu"), 1_200),
            entry("vc.content", EpOutcome::granted("cpu"), 1_200),
        ];
        assert_eq!(infer_vc_gpu(&rvc_cpu, start), Some(false));
        // A DirectML ONNX voice.
        assert_eq!(infer_vc_gpu(&[entry("vc.onnx", EpOutcome::granted("dml"), 1_001)], start), Some(true));
        // An earlier CUDA session and other subsystems' GPU sessions do not count; a DSP-only
        // preset records nothing: unknown.
        let stale_or_other = [
            entry("vc.content", EpOutcome::granted("cuda"), 999),
            entry("tts.kokoro", EpOutcome::granted("cuda"), 1_500),
            entry("vcx", EpOutcome::granted("cuda"), 1_500),
        ];
        assert_eq!(infer_vc_gpu(&stale_or_other, start), None);
        assert_eq!(infer_vc_gpu(&[], start), None);
    }

    #[test]
    fn wire_format_matches_the_ts_mirror() {
        let snap = ResourceSnapshot {
            budget: VramBudget {
                adapter_index: Some(0),
                total_mb: 8188,
                ..VramBudget::default()
            },
            claims: vec![VramClaimInfo {
                id: 1,
                subsystem: Subsystem::SttFinal,
                label: "x".into(),
                mb: 10,
                external: false,
            }],
            vc_running: true,
            vc_gpu: Some(true),
        };
        let v = serde_json::to_value(&snap).unwrap();
        assert_eq!(v["claims"][0]["subsystem"], "stt_final");
        assert_eq!(v["budget"]["total_mb"], 8188);
        assert_eq!(v["vc_running"], true);
        assert_eq!(v["vc_gpu"], true);
        assert_eq!(serde_json::from_value::<ResourceSnapshot>(v).unwrap(), snap);
        // A snapshot from before `vc_gpu` existed reads as "unknown".
        let old = serde_json::json!({"budget": serde_json::to_value(VramBudget::default()).unwrap(), "claims": [], "vc_running": true});
        assert_eq!(
            serde_json::from_value::<ResourceSnapshot>(old)
                .unwrap()
                .vc_gpu,
            None
        );
        // The frozen TS shape (no P1-D2 extras) still deserializes.
        let old = serde_json::json!({"id": 2, "subsystem": "llm", "label": "y", "mb": 5});
        assert!(
            !serde_json::from_value::<VramClaimInfo>(old)
                .unwrap()
                .external
        );
        assert_eq!(serde_json::to_value(RecommendationTopic::Qwen3Tts).unwrap(), "qwen3_tts");
        let topic: RecommendationTopic = serde_json::from_value(serde_json::json!("mt_llm")).unwrap();
        assert_eq!(topic, RecommendationTopic::MtLlm);
    }

    /// The acceptance check on the dev rig (`cargo test resources -- --nocapture` prints it): the
    /// primary GPU (the RTX 4090 there) is reported with its budget, and recommendations are
    /// deterministic. On a machine without a GPU the budget is empty and choices are CPU ones.
    #[test]
    fn headless_report_on_this_pc() {
        let text = headless_report().expect("report");
        println!("{text}");
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        let sys: SystemInfo = serde_json::from_value(v["system"].clone()).unwrap();
        let snap: ResourceSnapshot = serde_json::from_value(v["resources"].clone()).unwrap();
        if let Some(p) = sys.primary().filter(|p| !p.integrated) {
            assert_eq!(snap.budget.adapter_index, Some(p.adapter.index));
            assert_eq!(snap.budget.total_mb, p.adapter.dedicated_vram_mb);
            if cfg!(windows) {
                assert!(snap.budget.budget_mb > 0, "no DXGI budget for {}", p.adapter.name);
            }
        }
        let recs = v["recommendations"].as_array().unwrap();
        assert_eq!(recs.len(), 3);
        for (i, &topic) in RecommendationTopic::ALL.iter().enumerate() {
            let again = recommend(topic, &sys, &snap);
            assert_eq!(serde_json::to_value(&again).unwrap(), recs[i], "{topic:?} changed between calls");
        }
    }
}
