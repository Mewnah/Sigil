//! Shared allocators, the VRAM cap and the VRAM ledger (P3-C2).
//!
//! - **Ledger** ([`ledger`]): what each voice-engine component holds on the GPU, in MiB. Session
//!   plans measure their own footprint (this process's VRAM on the plan's adapter from before the
//!   plan's CUDA stream is created to after the session, its I/O buffers and its first run, i.e.
//!   activations, workspaces and the arena peak, through DXGI `QueryVideoMemoryInfo`) and hold it
//!   as a [`VramHold`] until they are dropped; several plans of one component (hot voices) add up.
//!   The first CUDA plan in the process includes the CUDA context (about 360 MB), like a legacy
//!   session would; a CUDA stream dropped after a fallback to DirectML gives back what it took.
//!   The governor reads [`VramLedger::snapshot`], and the voice changer resizes its
//!   `resources::claim_vram(Subsystem::Vc, ..)` to the measured process delta (P3-E; 0 MB on CPU).
//!   Measurements are process-wide deltas, so plans built concurrently on one adapter share them
//!   approximately.
//! - **Shared environment arena** ([`ensure_cuda_env_arena`]): one ONNX Runtime CUDA arena per
//!   device registered on the environment (`CreateAndRegisterAllocatorV2`) that sessions opt into
//!   (`session.use_env_allocators`, `ort_session::GpuExtras::cuda_env_arena`). Sessions that run one
//!   after another then reuse each other's activation memory instead of each keeping its own peak,
//!   and the VRAM cap bounds all of them together instead of each one. It grows only by what is
//!   requested. Only the classic CUDA provider (next to `Sigil.exe`) supports it; CUDA-graph
//!   sessions never use it (replays write into addresses captured from their own arena).
//!
//! # Frozen by P3-S0
//! - [`ledger`]`() -> &'static VramLedger`, [`VramLedger::record`], [`VramLedger::release`],
//!   [`VramLedger::snapshot`] → [`VramReport`].
//!
//! # Added by P3-C2
//! [`VramLedger::hold`] → [`VramHold`], [`VramProbe`], [`ensure_cuda_env_arena`],
//! [`cuda_env_arena_registered`].

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};

use serde::Serialize;

use crate::services::diagnostics::gpu_mem;

/// VRAM held by the voice engine, per component.
#[derive(Clone, Debug, Default, Serialize)]
pub struct VramReport {
    /// Sum of the components (MiB).
    pub total_mb: f64,
    /// `component -> MiB` (`vc.enc`, `vc.flowdec`, `vc.content`, `vc.index`, ...).
    pub components: BTreeMap<String, f64>,
}

#[derive(Default)]
struct Entries {
    /// `record` / `release` values (one per component).
    recorded: BTreeMap<String, f64>,
    /// Live holds: id → (component, MiB).
    holds: BTreeMap<u64, (String, f64)>,
}

/// The process-wide ledger.
#[derive(Default)]
pub struct VramLedger {
    entries: Mutex<Entries>,
    next_hold: AtomicU64,
}

/// The ledger.
pub fn ledger() -> &'static VramLedger {
    static LEDGER: OnceLock<VramLedger> = OnceLock::new();
    LEDGER.get_or_init(VramLedger::default)
}

impl VramLedger {
    fn lock(&self) -> std::sync::MutexGuard<'_, Entries> {
        self.entries.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Set what `component` holds (MiB), replacing an earlier value.
    pub fn record(&self, component: &str, mb: f64) {
        self.lock().recorded.insert(component.to_string(), sanitize(mb));
    }

    pub fn release(&self, component: &str) {
        self.lock().recorded.remove(component);
    }

    /// Add `mb` to `component` until the returned hold is dropped (several holds of one component
    /// add up: one per hot voice).
    pub fn hold(&'static self, component: &str, mb: f64) -> VramHold {
        let id = self.next_hold.fetch_add(1, Ordering::Relaxed);
        self.lock().holds.insert(id, (component.to_string(), sanitize(mb)));
        VramHold { ledger: self, id }
    }

    pub fn snapshot(&self) -> VramReport {
        let entries = self.lock();
        let mut components = entries.recorded.clone();
        for (component, mb) in entries.holds.values() {
            *components.entry(component.clone()).or_insert(0.0) += mb;
        }
        VramReport { total_mb: components.values().sum(), components }
    }
}

fn sanitize(mb: f64) -> f64 {
    if mb.is_finite() {
        mb.max(0.0)
    } else {
        0.0
    }
}

/// A ledger entry that lives as long as the holder (see [`VramLedger::hold`]).
pub struct VramHold {
    ledger: &'static VramLedger,
    id: u64,
}

impl VramHold {
    /// Replace the held amount (a re-measurement).
    pub fn update(&self, mb: f64) {
        if let Some(entry) = self.ledger.lock().holds.get_mut(&self.id) {
            entry.1 = sanitize(mb);
        }
    }

    pub fn mb(&self) -> f64 {
        self.ledger.lock().holds.get(&self.id).map(|e| e.1).unwrap_or(0.0)
    }
}

impl Drop for VramHold {
    fn drop(&mut self) {
        self.ledger.lock().holds.remove(&self.id);
    }
}

impl std::fmt::Debug for VramHold {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VramHold").field("id", &self.id).field("mb", &self.mb()).finish()
    }
}

/// This process's VRAM on one DXGI adapter at construction, for measuring what a build added.
#[derive(Clone, Copy, Debug)]
pub struct VramProbe {
    adapter: u32,
    before_mb: u64,
}

impl VramProbe {
    /// Start measuring on DXGI adapter `adapter` (`None` when DXGI cannot report it).
    pub fn start(adapter: u32) -> Option<VramProbe> {
        gpu_mem::process_vram_usage(adapter).map(|u| VramProbe { adapter, before_mb: u.process_usage_mb })
    }

    /// Start measuring on the adapter a session of `ep` on device `device` uses (the DXGI index of
    /// the `device`-th NVIDIA adapter for CUDA / TensorRT, the DXGI index itself for DirectML;
    /// `None` on the CPU).
    pub fn for_rung(ep: &str, device: i32) -> Option<VramProbe> {
        Self::start(dxgi_adapter_for(ep, device)?)
    }

    /// MiB this process gained on the adapter since [`start`](Self::start) (0 when it shrank).
    pub fn delta_mb(&self) -> Option<f64> {
        let now = gpu_mem::process_vram_usage(self.adapter)?.process_usage_mb;
        Some(now.saturating_sub(self.before_mb) as f64)
    }

    pub fn adapter(&self) -> u32 {
        self.adapter
    }
}

/// DXGI adapter index of a session on `ep` / `device` (see [`VramProbe::for_rung`]).
pub fn dxgi_adapter_for(ep: &str, device: i32) -> Option<u32> {
    match ep {
        "cuda" | "tensorrt_rtx" | "tensorrt" => {
            let nvidia = gpu_mem::nvidia_adapter_indices(&gpu_mem::adapters());
            nvidia.get(device.max(0) as usize).copied()
        }
        "dml" => Some(device.max(0) as u32),
        _ => None,
    }
}

// ---------------------------------------------------------------------------------------------
// Shared environment arena (CUDA)
// ---------------------------------------------------------------------------------------------

/// Devices with a registered environment arena, and the cap (MiB, 0 = none) it was registered with.
fn registered() -> &'static Mutex<BTreeMap<i32, u32>> {
    static REGISTERED: OnceLock<Mutex<BTreeMap<i32, u32>>> = OnceLock::new();
    REGISTERED.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Whether CUDA device `device` has a shared environment arena.
pub fn cuda_env_arena_registered(device: i32) -> bool {
    registered().lock().unwrap_or_else(PoisonError::into_inner).contains_key(&device.max(0))
}

/// The arena configuration keys and values (`CreateArenaCfgV2`): grow only by what is requested,
/// capped at `cap_mb` (0 / `None` = no cap).
pub(crate) fn arena_config(cap_mb: Option<u32>) -> [(&'static str, usize); 2] {
    let max_mem = cap_mb.filter(|&mb| mb > 0).map(|mb| (mb as usize).saturating_mul(1024 * 1024)).unwrap_or(0);
    // arena_extend_strategy: 0 = kNextPowerOfTwo, 1 = kSameAsRequested.
    [("max_mem", max_mem), ("arena_extend_strategy", 1)]
}

/// Register the shared CUDA arena of `device` on the ONNX Runtime environment (once per device and
/// process; ONNX Runtime cannot resize a registered arena, so a later call with another cap keeps
/// the first cap and logs it). Must run before a session that uses it is created.
pub fn ensure_cuda_env_arena(device: i32, cap_mb: Option<u32>) -> Result<(), String> {
    use std::ffi::{c_char, CString};

    use ort::memory::{AllocationDevice, AllocatorType, MemoryInfo, MemoryType};
    use ort::AsPointer;

    let device = device.max(0);
    let cap = cap_mb.unwrap_or(0);
    let mut reg = registered().lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(&first) = reg.get(&device) {
        if first != cap {
            tracing::info!(target: "sigil::gpu", "shared CUDA arena of device {device} keeps its {first} MB cap (asked {cap} MB)");
        }
        return Ok(());
    }
    crate::services::neural_audio::ort_session::ensure_ort_environment();
    let env = ort::environment::Environment::current().map_err(|e| format!("ONNX Runtime environment: {e}"))?;
    let mem_info = MemoryInfo::new(AllocationDevice::CUDA, device, AllocatorType::Device, MemoryType::Default)
        .map_err(|e| format!("CUDA memory info: {e}"))?;
    let api = ort::api();
    let config = arena_config(cap_mb);
    let keys: Vec<CString> = config.iter().map(|(k, _)| CString::new(*k).unwrap_or_default()).collect();
    let key_ptrs: Vec<*const c_char> = keys.iter().map(|k| k.as_ptr()).collect();
    let values: Vec<usize> = config.iter().map(|(_, v)| *v).collect();
    let mut cfg: *mut ort::sys::OrtArenaCfg = std::ptr::null_mut();
    // SAFETY: parallel key / value arrays of `num_keys` entries that outlive the call.
    let status = unsafe { (api.CreateArenaCfgV2)(key_ptrs.as_ptr(), values.as_ptr(), key_ptrs.len(), &mut cfg) };
    // SAFETY: a status just returned by ORT.
    unsafe { ort::Error::result_from_status(status) }.map_err(|e| format!("arena config: {e}"))?;
    let provider_keys = [CString::new("device_id").unwrap_or_default()];
    let provider_values = [CString::new(device.to_string()).unwrap_or_default()];
    let pk: Vec<*const c_char> = provider_keys.iter().map(|k| k.as_ptr()).collect();
    let pv: Vec<*const c_char> = provider_values.iter().map(|v| v.as_ptr()).collect();
    let provider = CString::new("CUDAExecutionProvider").unwrap_or_default();
    // SAFETY: live environment, memory info and arena config; the option arrays outlive the call.
    // ONNX Runtime copies what it keeps; the arena config is released right after.
    let status = unsafe {
        (api.CreateAndRegisterAllocatorV2)(env.ptr().cast_mut(), provider.as_ptr(), mem_info.ptr(), cfg, pk.as_ptr(), pv.as_ptr(), pk.len())
    };
    // SAFETY: created above, released once.
    unsafe { (api.ReleaseArenaCfg)(cfg) };
    // SAFETY: a status just returned by ORT.
    unsafe { ort::Error::result_from_status(status) }.map_err(|e| format!("shared CUDA arena of device {device}: {e}"))?;
    tracing::info!(
        target: "sigil::gpu",
        "shared CUDA arena registered on device {device} ({})",
        if cap > 0 { format!("{cap} MB cap") } else { "no cap".to_string() }
    );
    reg.insert(device, cap);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ledger_sums_components() {
        let l = VramLedger::default();
        l.record("vc.enc", 100.0);
        l.record("vc.flowdec", 50.5);
        l.record("vc.enc", 120.0);
        assert_eq!(l.snapshot().total_mb, 170.5);
        l.release("vc.enc");
        assert_eq!(l.snapshot().components.len(), 1);
        l.record("vc.x", f64::NAN);
        assert_eq!(l.snapshot().components["vc.x"], 0.0);
    }

    /// Holds of one component add up (two hot voices), update in place and vanish when dropped.
    #[test]
    fn holds_add_up_per_component_and_release_on_drop() {
        let l = ledger();
        let key = format!("test.arena.hold.{}", std::process::id());
        let a = l.hold(&key, 100.0);
        let b = l.hold(&key, 40.0);
        assert_eq!(l.snapshot().components[&key], 140.0);
        b.update(60.0);
        assert_eq!(b.mb(), 60.0);
        assert_eq!(l.snapshot().components[&key], 160.0);
        drop(a);
        assert_eq!(l.snapshot().components[&key], 60.0);
        drop(b);
        assert!(!l.snapshot().components.contains_key(&key));
        let neg = l.hold(&key, -5.0);
        assert_eq!(neg.mb(), 0.0);
    }

    #[test]
    fn arena_config_caps_and_grows_by_request() {
        assert_eq!(arena_config(None), [("max_mem", 0), ("arena_extend_strategy", 1)]);
        assert_eq!(arena_config(Some(0)), [("max_mem", 0), ("arena_extend_strategy", 1)]);
        assert_eq!(arena_config(Some(512))[0], ("max_mem", 512 * 1024 * 1024));
        assert_eq!(dxgi_adapter_for("cpu", 0), None);
        assert_eq!(dxgi_adapter_for("dml", 2), Some(2));
    }

    /// DXGI reports this process's VRAM on every hardware adapter (Windows); the delta is never
    /// negative.
    #[cfg(windows)]
    #[test]
    fn probe_measures_process_vram() {
        let Some(adapter) = gpu_mem::adapters().into_iter().find(|a| !a.software) else {
            eprintln!("no hardware GPU; skipped");
            return;
        };
        let probe = VramProbe::start(adapter.index).expect("DXGI VRAM usage");
        assert_eq!(probe.adapter(), adapter.index);
        assert!(probe.delta_mb().unwrap() >= 0.0);
    }
}
