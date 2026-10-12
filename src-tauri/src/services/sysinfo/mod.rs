//! `sysinfo` plugin: system probe for runtime and model choices. DXGI adapters and VRAM, the NVIDIA
//! driver version (nvcuda loaded dynamically), NVML (loaded dynamically), CPU features and
//! execution-provider eligibility (CUDA 13 needs Turing or newer and driver R580+; TensorRT-RTX
//! needs Ampere or newer; DirectML needs a D3D12 adapter). Never loads a GPU runtime or ORT and
//! never creates a CUDA context.
//!
//! - [`dxgi`]: adapters (via `diagnostics::gpu_mem`) plus a Direct3D 12 capability test.
//! - [`nvml`]: NVML driver / CUDA versions, compute capability, PCI ids, device-wide memory.
//! - [`cuda`]: `cuDriverGetVersion` from nvcuda.dll (no cuInit) and version helpers.
//! - [`arch`]: compute capability / PCI id → GPU generation, integrated-GPU heuristic.
//! - [`eps`]: the eligibility rules (pure, table-tested).
//! - [`cpu`]: brand, core counts, AVX2 / AVX-512 / VNNI, total RAM.
//!
//! The GPU part of a probe runs on its own thread with a timeout ([`GPU_PROBE_TIMEOUT`]), so a hung
//! graphics driver degrades the report instead of hanging the caller; only one GPU probe runs at a
//! time. The probe is not run at startup: the first [`cached`] call (the TS service asks right
//! after start, in the background) pays for it once.
//!
//! A degraded probe never sticks for the session ([`GpuProbe`]): a probe thread that answers after
//! the timeout publishes its result into the cache itself (and emits [`SYSINFO_EVENT`]); a probe
//! that got no GPU answer keeps the GPU facts of an earlier probe (`gpu_probe: previous`); and a
//! cached result with no GPU facts at all (`gpu_probe: missing`) is probed again by [`cached`] once
//! no GPU probe is running (at most every [`MISSING_RETRY`]).
//!
//! # Frozen contract (P1-S0b; change only through INTEGRATE; structs may gain fields)
//! - [`probe`]`() -> SystemInfo` (uncached; no GPU runtime loaded) and [`cached`]`() -> SystemInfo`
//!   (probed once per process, replaced by [`refresh`]).
//! - [`SystemInfo`] / [`GpuInfo`] (a `diagnostics::gpu_mem::GpuAdapter` flattened, plus `arch` and
//!   `compute_capability`) / [`EpEligibility`] / [`Eligibility`] / [`CpuInfo`]; TS mirrors
//!   `RustSystemInfo` etc. in src/types.ts (fields added by P1-D2 are mirrored in
//!   src/core/services/sysinfo/index.ts until INTEGRATE folds them into src/types.ts).
//! - IPC `plugin:sysinfo|get_system_info({ refresh?: boolean }) -> SystemInfo`.
//! - Event [`SYSINFO_EVENT`] `"sysinfo:changed"` (payload [`SystemInfo`]) after a refresh, a late
//!   GPU answer or a re-probe of a result without GPU facts.
//! - P1-D2 additions: `SystemInfo::gpu_probe` ([`GpuProbe`], serde default `complete`) and the
//!   eligibility code [`eps::GPU_UNKNOWN`].

pub mod arch;
pub mod cpu;
pub mod cuda;
pub mod dxgi;
pub mod eps;
pub mod nvml;

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::diagnostics::gpu_mem::{GpuAdapter, VENDOR_NVIDIA};
use crate::services::diagnostics::{self, logging};

/// Event emitted after the system info was re-probed (payload [`SystemInfo`]).
pub const SYSINFO_EVENT: &str = "sysinfo:changed";

/// Longest a probe waits for the graphics driver (DXGI, D3D12, NVML, nvcuda) before reporting
/// without GPU details.
pub const GPU_PROBE_TIMEOUT: Duration = Duration::from_secs(15);

/// A cached result without GPU facts is probed again after this long (only while no GPU probe is
/// running: a hung one publishes its own answer when it returns).
pub const MISSING_RETRY: Duration = Duration::from_secs(30);

/// How complete the GPU part of a probe is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GpuProbe {
    /// The graphics driver answered: the GPU facts are current.
    #[default]
    Complete,
    /// The driver did not answer this check; the GPU facts come from an earlier check in this
    /// session.
    Previous,
    /// The driver did not answer and no earlier check has GPU facts: `gpus` is empty although the
    /// PC may well have a GPU, and every GPU EP reports [`eps::GPU_UNKNOWN`].
    Missing,
}

/// One GPU: the DXGI adapter plus architecture details.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GpuInfo {
    #[serde(flatten)]
    pub adapter: GpuAdapter,
    /// Architecture family, lower-case: `pascal`, `turing`, `ampere`, `ada`, `blackwell`, `rdna2`,
    /// `rdna3`, `xe-hpg`, ... (`None` when unknown).
    pub arch: Option<String>,
    /// NVIDIA compute capability, e.g. `8.9` (from NVML; `None` without NVML).
    pub compute_capability: Option<String>,
    /// Shared-memory (integrated) GPU: tiny dedicated VRAM, an Intel iGPU or an AMD APU.
    #[serde(default)]
    pub integrated: bool,
    /// Direct3D 12 at feature level 11_0+ (what DirectML needs): `None` when not tested.
    #[serde(default)]
    pub d3d12: Option<bool>,
    /// Highest Direct3D 12 feature level, e.g. `12_2`.
    #[serde(default)]
    pub d3d12_feature_level: Option<String>,
    /// NVML device index (NVIDIA only, when NVML is available).
    #[serde(default)]
    pub nvml_index: Option<u32>,
    /// PCI bus id from NVML, e.g. `00000000:01:00.0`.
    #[serde(default)]
    pub pci_bus_id: Option<String>,
}

/// Whether an execution provider can run on this PC, and why not.
///
/// `eligible` covers the GPU, driver and OS. Runtime files (the NVIDIA runtime pack, provider DLLs)
/// are reported separately by the diagnostics probe; `eligible` means installing them would work.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Eligibility {
    pub eligible: bool,
    /// Plain-English explanation: why it is not eligible, or what could not be verified.
    pub reason: Option<String>,
    /// Machine-readable reason ([`eps::OK`], [`eps::UNVERIFIED`], [`eps::GPU_TOO_OLD`], ...); the
    /// UI translates by code.
    #[serde(default)]
    pub code: String,
}

/// EP eligibility on this PC (the CPU EP is always eligible).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct EpEligibility {
    pub cuda: Eligibility,
    pub tensorrt: Eligibility,
    pub tensorrt_rtx: Eligibility,
    pub directml: Eligibility,
}

/// CPU summary.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CpuInfo {
    pub brand: String,
    pub physical_cores: u32,
    pub logical_cores: u32,
    /// Cores in the highest efficiency class (P-cores on hybrid CPUs; all cores otherwise).
    #[serde(default)]
    pub performance_cores: u32,
    /// AVX2 with FMA (the `cpu-avx2` runtime variant).
    pub avx2: bool,
    /// AVX-512F (the `cpu-avx512` runtime variant).
    pub avx512: bool,
    /// AVX-VNNI or AVX512-VNNI (int8 content encoder on CPU).
    pub vnni: bool,
}

/// Result of the system probe.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemInfo {
    pub gpus: Vec<GpuInfo>,
    /// Index (DXGI `EnumAdapters1`) of the GPU Sigil uses by default ([`pick_primary`]): discrete
    /// first, then one the CUDA EP can use (CUDA is the preferred EP), then the most VRAM; `None`
    /// without a hardware GPU.
    pub primary_gpu: Option<u32>,
    /// NVIDIA driver number, e.g. `616.56`.
    pub nvidia_driver: Option<String>,
    /// CUDA driver API version, e.g. `13.2`.
    pub cuda_driver_version: Option<String>,
    pub eps: EpEligibility,
    pub cpu: CpuInfo,
    pub total_ram_mb: u64,
    pub probed_at_unix_ms: u64,
    /// NVML version, when NVML is available.
    #[serde(default)]
    pub nvml_version: Option<String>,
    /// Plain-English probe findings (what could not be read and why).
    #[serde(default)]
    pub notes: Vec<String>,
    /// Whether the GPU facts are current, from an earlier check, or missing.
    #[serde(default)]
    pub gpu_probe: GpuProbe,
}

impl SystemInfo {
    /// The primary GPU's entry.
    pub fn primary(&self) -> Option<&GpuInfo> {
        let index = self.primary_gpu?;
        self.gpus.iter().find(|g| g.adapter.index == index)
    }
}

// ---------------------------------------------------------------------------------------------
// Assembly (pure)
// ---------------------------------------------------------------------------------------------

/// Raw facts gathered from the OS and drivers; [`assemble`] turns them into a [`SystemInfo`].
#[derive(Clone, Debug, Default)]
pub(crate) struct RawProbe {
    pub adapters: Vec<GpuAdapter>,
    /// Direct3D 12 support by adapter LUID (missing = not tested).
    pub d3d12: BTreeMap<String, dxgi::D3d12Support>,
    pub nvml: Option<nvml::NvmlSystem>,
    /// Why NVML is unavailable (only reported when there is an NVIDIA GPU).
    pub nvml_error: Option<String>,
    /// From nvcuda's `cuDriverGetVersion`, e.g. `13.2`.
    pub cuda_driver_version: Option<String>,
    pub cpu: CpuInfo,
    pub total_ram_mb: u64,
    pub windows: bool,
    pub probed_at_unix_ms: u64,
    pub notes: Vec<String>,
    /// [`GpuProbe::Missing`] when the GPU part got no answer (the adapter fields are then empty).
    pub gpu_probe: GpuProbe,
}

/// The GPU Sigil uses by default: hardware only; discrete before integrated; then a GPU the CUDA
/// EP can use (NVIDIA Turing or newer, or an NVIDIA card of unknown generation; CUDA is the
/// preferred EP); then the most dedicated VRAM; then the lowest DXGI index. An NVIDIA card too old
/// for CUDA 13 (Pascal and older) ranks with the other vendors' cards (DirectML / Vulkan), so a
/// leftover GTX 1080 Ti never outranks the RTX 3070 that actually runs the models.
pub(crate) fn pick_primary(gpus: &[GpuInfo]) -> Option<u32> {
    gpus.iter()
        .filter(|g| !g.adapter.software)
        .max_by_key(|g| {
            (
                !g.integrated,
                eps::meets_cuda_floor(g) != Some(false),
                g.adapter.dedicated_vram_mb,
                std::cmp::Reverse(g.adapter.index),
            )
        })
        .map(|g| g.adapter.index)
}

pub(crate) fn assemble(raw: RawProbe) -> SystemInfo {
    let RawProbe {
        adapters,
        d3d12,
        nvml,
        nvml_error,
        cuda_driver_version,
        cpu,
        total_ram_mb,
        windows,
        probed_at_unix_ms,
        mut notes,
        gpu_probe,
    } = raw;
    let devices: &[nvml::NvmlDevice] = nvml.as_ref().map(|n| n.devices.as_slice()).unwrap_or(&[]);
    let keys: Vec<nvml::AdapterKey<'_>> = adapters
        .iter()
        .map(|a| nvml::AdapterKey {
            vendor_id: a.vendor_id,
            device_id: a.device_id,
            name: &a.name,
            software: a.software,
        })
        .collect();
    let matched = nvml::match_devices(&keys, devices);

    let gpus: Vec<GpuInfo> = adapters
        .iter()
        .zip(matched)
        .map(|(a, nvml_index)| {
            let dev = nvml_index.and_then(|i| devices.iter().find(|d| d.index == i));
            let cc = dev.and_then(|d| d.compute_capability);
            let arch = if a.software {
                None
            } else if a.vendor_id == VENDOR_NVIDIA {
                cc.and_then(|(x, y)| arch::nvidia_gen_from_cc(x, y))
                    .or_else(|| arch::nvidia_gen_from_device_id(a.device_id))
                    .map(|g| g.as_str().to_string())
            } else {
                arch::other_arch_from_device_id(a.vendor_id, a.device_id).map(str::to_string)
            };
            let support = d3d12.get(&a.luid);
            GpuInfo {
                arch,
                compute_capability: cc.map(|(x, y)| format!("{x}.{y}")),
                integrated: !a.software && arch::is_integrated(a.vendor_id, a.device_id, a.dedicated_vram_mb),
                d3d12: support.map(|s| matches!(s, dxgi::D3d12Support::Level(_))),
                d3d12_feature_level: match support {
                    Some(dxgi::D3d12Support::Level(l)) => Some((*l).to_string()),
                    _ => None,
                },
                nvml_index,
                pci_bus_id: dev.and_then(|d| d.pci_bus_id.clone()),
                adapter: a.clone(),
            }
        })
        .collect();

    let has_nvidia = gpus
        .iter()
        .any(|g| !g.adapter.software && g.adapter.vendor_id == VENDOR_NVIDIA);
    let nvidia_driver = nvml
        .as_ref()
        .and_then(|n| n.driver_version.clone())
        .or_else(|| gpus.iter().find_map(|g| g.adapter.nvidia_driver.clone()));
    let cuda_driver_version = cuda_driver_version.or_else(|| {
        nvml.as_ref()
            .and_then(|n| n.cuda_driver_version)
            .and_then(cuda::cuda_version_string)
    });
    if has_nvidia {
        if let Some(e) = nvml_error {
            notes.push(format!("NVML is unavailable ({e}); the GPU generation comes from its PCI id"));
        }
        if cuda_driver_version.is_none() {
            notes.push("the CUDA driver version could not be read".into());
        }
    }
    let eps = if gpu_probe == GpuProbe::Missing {
        eps::unknown()
    } else {
        eps::assess(&gpus, nvidia_driver.as_deref(), cuda_driver_version.as_deref(), windows)
    };
    SystemInfo {
        primary_gpu: pick_primary(&gpus),
        gpus,
        nvidia_driver,
        cuda_driver_version,
        eps,
        cpu,
        total_ram_mb,
        probed_at_unix_ms,
        nvml_version: nvml.and_then(|n| n.nvml_version),
        notes,
        gpu_probe,
    }
}

/// What to store when a probe finishes: a probe without GPU facts ([`GpuProbe::Missing`]) keeps
/// the GPU facts of `previous` (if it has any), marked [`GpuProbe::Previous`]; everything else
/// replaces `previous`. CPU facts and the timestamp always come from `new`. Pure.
pub(crate) fn keep_previous_gpu_facts(new: SystemInfo, previous: Option<&SystemInfo>) -> SystemInfo {
    if new.gpu_probe != GpuProbe::Missing {
        return new;
    }
    let Some(p) = previous.filter(|p| p.gpu_probe != GpuProbe::Missing) else {
        return new;
    };
    let mut notes = new.notes;
    notes.push("the GPU details are from an earlier check".into());
    SystemInfo {
        gpus: p.gpus.clone(),
        primary_gpu: p.primary_gpu,
        nvidia_driver: p.nvidia_driver.clone(),
        cuda_driver_version: p.cuda_driver_version.clone(),
        eps: p.eps.clone(),
        nvml_version: p.nvml_version.clone(),
        notes,
        gpu_probe: GpuProbe::Previous,
        ..new
    }
}

// ---------------------------------------------------------------------------------------------
// Probe
// ---------------------------------------------------------------------------------------------

/// GPU-side facts (the part that talks to the graphics driver).
struct GpuFacts {
    adapters: Vec<GpuAdapter>,
    d3d12: BTreeMap<String, dxgi::D3d12Support>,
    nvml: Option<nvml::NvmlSystem>,
    nvml_error: Option<String>,
    cuda_driver_version: Option<String>,
}

fn gather_gpu_facts(retry_nvml: bool) -> GpuFacts {
    let adapters = dxgi::adapters();
    let d3d12 = if adapters.iter().any(|a| !a.software) {
        dxgi::d3d12_support()
    } else {
        BTreeMap::new()
    };
    let has_nvidia = adapters
        .iter()
        .any(|a| !a.software && a.vendor_id == VENDOR_NVIDIA);
    let (nvml, nvml_error, cuda_driver_version) = if has_nvidia {
        let (sys, err) = match nvml::shared(retry_nvml) {
            Ok(h) => (Some(h.system()), None),
            Err(e) => (None, Some(e)),
        };
        (sys, err, cuda::driver_cuda_version())
    } else {
        // No NVIDIA GPU: never load NVML or nvcuda.
        (None, None, None)
    };
    GpuFacts {
        adapters,
        d3d12,
        nvml,
        nvml_error,
        cuda_driver_version,
    }
}

/// Set while a GPU probe thread runs (a hung driver keeps it set; later probes then skip the GPU
/// part instead of piling up threads).
static GPU_PROBE_BUSY: AtomicBool = AtomicBool::new(false);

/// Hand-off between a probe thread and the caller waiting for it.
struct Handoff<T> {
    answer: Option<Result<T, String>>,
    /// The caller stopped waiting: the thread hands its late answer to the `late` callback.
    abandoned: bool,
}

/// Run `gather` on its own thread and wait up to `timeout` for it. Only one runs at a time per
/// `busy` flag (a second call while one runs fails at once). An answer that arrives after the
/// timeout goes to `late` (with the time it took), on the probe thread, before `busy` clears; the
/// decision is made under one lock, so an answer is never lost between the timeout and the
/// hand-off.
fn answer_within<T, G, L>(busy: &'static AtomicBool, timeout: Duration, gather: G, late: L) -> Result<T, String>
where
    T: Send + 'static,
    G: FnOnce() -> T + Send + 'static,
    L: FnOnce(T, Duration) + Send + 'static,
{
    if busy.swap(true, Ordering::AcqRel) {
        return Err("an earlier GPU probe has not finished (the graphics driver may be busy or hung)".into());
    }
    let shared = Arc::new((
        Mutex::new(Handoff {
            answer: None,
            abandoned: false,
        }),
        Condvar::new(),
    ));
    let theirs = Arc::clone(&shared);
    let started = Instant::now();
    let spawned = std::thread::Builder::new()
        .name("sigil-sysinfo-gpu".into())
        .spawn(move || {
            struct Clear(&'static AtomicBool);
            impl Drop for Clear {
                fn drop(&mut self) {
                    self.0.store(false, Ordering::Release);
                }
            }
            let _clear = Clear(busy);
            let answer = std::panic::catch_unwind(std::panic::AssertUnwindSafe(gather)).map_err(|_| "the GPU probe stopped unexpectedly".to_string());
            let (m, cv) = &*theirs;
            let mut h = lock(m);
            if h.abandoned {
                drop(h);
                if let Ok(value) = answer {
                    late(value, started.elapsed());
                }
            } else {
                h.answer = Some(answer);
                cv.notify_all();
            }
        });
    if let Err(e) = spawned {
        busy.store(false, Ordering::Release);
        return Err(format!("cannot start the GPU probe thread: {e}"));
    }
    let (m, cv) = &*shared;
    let (mut h, _) = cv
        .wait_timeout_while(lock(m), timeout, |h| h.answer.is_none())
        .unwrap_or_else(PoisonError::into_inner);
    match h.answer.take() {
        Some(answer) => answer,
        None => {
            h.abandoned = true;
            Err(format!("the graphics driver did not answer within {} s", timeout.as_secs()))
        }
    }
}

fn raw_without_gpu() -> RawProbe {
    RawProbe {
        cpu: cpu::probe(),
        total_ram_mb: cpu::total_ram_mb(),
        windows: cfg!(windows),
        probed_at_unix_ms: logging::unix_ms_now(),
        ..RawProbe::default()
    }
}

fn with_gpu_facts(mut raw: RawProbe, f: GpuFacts) -> RawProbe {
    raw.adapters = f.adapters;
    raw.d3d12 = f.d3d12;
    raw.nvml = f.nvml;
    raw.nvml_error = f.nvml_error;
    raw.cuda_driver_version = f.cuda_driver_version;
    raw.gpu_probe = GpuProbe::Complete;
    raw
}

/// A GPU probe that timed out answered after all: its result is the newest GPU facts there are
/// (only one GPU probe runs at a time), so it replaces the cache and the UI hears about it.
fn publish_late(facts: GpuFacts, took: Duration) {
    let mut info = assemble(with_gpu_facts(raw_without_gpu(), facts));
    info.notes
        .push(format!("the graphics driver answered late (after {} s)", took.as_secs()));
    let stored = store(info);
    log_summary(&stored);
    diagnostics::emit(SYSINFO_EVENT, &stored);
}

fn probe_with(retry_nvml: bool) -> SystemInfo {
    let mut raw = raw_without_gpu();
    let answer = answer_within(&GPU_PROBE_BUSY, GPU_PROBE_TIMEOUT, move || gather_gpu_facts(retry_nvml), publish_late);
    match answer {
        Ok(f) => raw = with_gpu_facts(raw, f),
        Err(e) => {
            tracing::warn!(target: "sigil::sysinfo", "GPU probe skipped: {e}");
            raw.notes.push(format!("GPU details are missing: {e}"));
            raw.gpu_probe = GpuProbe::Missing;
        }
    }
    assemble(raw)
}

/// Probe now (no GPU runtime or ORT is loaded).
pub fn probe() -> SystemInfo {
    probe_with(false)
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

static CACHE: Mutex<Option<SystemInfo>> = Mutex::new(None);
/// Serializes probes so concurrent first callers share one probe.
static PROBING: Mutex<()> = Mutex::new(());

/// Store a finished probe ([`keep_previous_gpu_facts`] against the cache) and return what was
/// stored.
fn store(new: SystemInfo) -> SystemInfo {
    let mut cache = lock(&CACHE);
    let stored = keep_previous_gpu_facts(new, cache.as_ref());
    *cache = Some(stored.clone());
    stored
}

/// Whether [`cached`] may serve `info` instead of probing again: everything except a result
/// without GPU facts that is older than [`MISSING_RETRY`] while no GPU probe is running. Pure.
pub(crate) fn servable(info: &SystemInfo, gpu_probe_running: bool, now_unix_ms: u64) -> bool {
    info.gpu_probe != GpuProbe::Missing || gpu_probe_running || now_unix_ms.saturating_sub(info.probed_at_unix_ms) < MISSING_RETRY.as_millis() as u64
}

fn cached_servable() -> Option<SystemInfo> {
    let info = lock(&CACHE).clone()?;
    servable(&info, GPU_PROBE_BUSY.load(Ordering::Acquire), logging::unix_ms_now()).then_some(info)
}

fn log_summary(info: &SystemInfo) {
    let gpu = info
        .primary()
        .map(|g| {
            format!(
                "{} ({} MB{}{})",
                g.adapter.name,
                g.adapter.dedicated_vram_mb,
                g.arch
                    .as_deref()
                    .map(|a| format!(", {a}"))
                    .unwrap_or_default(),
                g.compute_capability
                    .as_deref()
                    .map(|c| format!(" {c}"))
                    .unwrap_or_default()
            )
        })
        .unwrap_or_else(|| "no hardware GPU".into());
    let ep = |e: &Eligibility| if e.eligible { "yes" } else { "no" };
    tracing::info!(
        target: "sigil::sysinfo",
        "system: {gpu}; driver {}; CUDA {}; eligible cuda={} trt={} trt-rtx={} dml={}; CPU {} ({}C/{}T); RAM {} MB",
        info.nvidia_driver.as_deref().unwrap_or("-"),
        info.cuda_driver_version.as_deref().unwrap_or("-"),
        ep(&info.eps.cuda),
        ep(&info.eps.tensorrt),
        ep(&info.eps.tensorrt_rtx),
        ep(&info.eps.directml),
        info.cpu.brand,
        info.cpu.physical_cores,
        info.cpu.logical_cores,
        info.total_ram_mb
    );
    if info.gpu_probe != GpuProbe::Complete {
        tracing::warn!(target: "sigil::sysinfo", "GPU facts: {:?}", info.gpu_probe);
    }
    for note in &info.notes {
        tracing::info!(target: "sigil::sysinfo", "{note}");
    }
}

/// The process-wide probe result (probed on first use; a result without GPU facts is probed again,
/// see [`servable`]).
pub fn cached() -> SystemInfo {
    if let Some(info) = cached_servable() {
        return info;
    }
    let _probing = lock(&PROBING);
    if let Some(info) = cached_servable() {
        return info; // another caller probed while this one waited
    }
    let replacing = lock(&CACHE).is_some();
    let info = store(probe());
    log_summary(&info);
    if replacing {
        diagnostics::emit(SYSINFO_EVENT, &info);
    }
    info
}

/// Re-probe (retrying NVML if it failed before), replace the cached result (keeping the earlier
/// GPU facts if the driver does not answer) and emit [`SYSINFO_EVENT`].
pub fn refresh() -> SystemInfo {
    let _probing = lock(&PROBING);
    let info = store(probe_with(true));
    log_summary(&info);
    diagnostics::emit(SYSINFO_EVENT, &info);
    info
}

#[command]
async fn get_system_info(refresh: Option<bool>) -> Result<SystemInfo, String> {
    let re_probe = refresh.unwrap_or(false);
    tauri::async_runtime::spawn_blocking(move || if re_probe { self::refresh() } else { cached() })
        .await
        .map_err(|e| format!("system probe task: {e}"))
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("sysinfo")
        .invoke_handler(tauri::generate_handler![get_system_info])
        .build()
}

/// Builders shared by the sysinfo and resources tests.
#[cfg(test)]
pub(crate) mod test_support {
    use super::{CpuInfo, GpuInfo, SystemInfo};
    use crate::services::diagnostics::gpu_mem::{vendor_name, GpuAdapter, VENDOR_MICROSOFT};

    /// A hardware GPU (D3D12 12_1) or, for the Microsoft vendor id, the software renderer.
    pub fn gpu(index: u32, vendor_id: u32, name: &str, vram_mb: u64, cc: Option<&str>, arch: Option<&str>) -> GpuInfo {
        let software = vendor_id == VENDOR_MICROSOFT;
        GpuInfo {
            adapter: GpuAdapter {
                index,
                name: name.into(),
                vendor: vendor_name(vendor_id).into(),
                vendor_id,
                device_id: 0,
                luid: format!("00000000:{index:08x}"),
                dedicated_vram_mb: vram_mb,
                shared_memory_mb: 16_384,
                software,
                driver_version: None,
                nvidia_driver: None,
                vram: None,
            },
            arch: arch.map(Into::into),
            compute_capability: cc.map(Into::into),
            integrated: false,
            d3d12: (!software).then_some(true),
            d3d12_feature_level: (!software).then(|| "12_1".to_string()),
            nvml_index: None,
            pci_bus_id: None,
        }
    }

    /// A system with these GPUs (primary by [`super::pick_primary`]; EP eligibility assessed with an
    /// R616 / CUDA 13.2 driver when there is an NVIDIA GPU), an 8-core AVX2 CPU and 32 GB RAM.
    pub fn system(gpus: Vec<GpuInfo>) -> SystemInfo {
        let primary_gpu = super::pick_primary(&gpus);
        let nvidia = gpus
            .iter()
            .any(|g| !g.adapter.software && g.adapter.vendor_id == super::VENDOR_NVIDIA);
        let (driver, cuda) = if nvidia { (Some("616.56"), Some("13.2")) } else { (None, None) };
        SystemInfo {
            primary_gpu,
            eps: super::eps::assess(&gpus, driver, cuda, true),
            nvidia_driver: driver.map(Into::into),
            cuda_driver_version: cuda.map(Into::into),
            gpus,
            cpu: CpuInfo {
                brand: "Test CPU".into(),
                physical_cores: 8,
                logical_cores: 16,
                performance_cores: 8,
                avx2: true,
                avx512: false,
                vnni: false,
            },
            total_ram_mb: 32_768,
            ..SystemInfo::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::gpu;
    use super::*;
    use crate::services::diagnostics::gpu_mem::{vendor_name, VENDOR_AMD, VENDOR_INTEL, VENDOR_MICROSOFT};

    fn adapter(index: u32, vendor_id: u32, device_id: u32, name: &str, vram_mb: u64, nvidia_driver: Option<&str>) -> GpuAdapter {
        GpuAdapter {
            index,
            name: name.into(),
            vendor: vendor_name(vendor_id).into(),
            vendor_id,
            device_id,
            luid: format!("00000000:{index:08x}"),
            dedicated_vram_mb: vram_mb,
            shared_memory_mb: 32_000,
            software: vendor_id == VENDOR_MICROSOFT,
            driver_version: None,
            nvidia_driver: nvidia_driver.map(Into::into),
            vram: None,
        }
    }

    fn nvml_dev(index: u32, device: u32, name: &str, cc: (u32, u32), total_mb: u64) -> nvml::NvmlDevice {
        nvml::NvmlDevice {
            index,
            name: name.into(),
            pci_device_id: device,
            pci_vendor_id: VENDOR_NVIDIA,
            pci_bus_id: Some(format!("00000000:0{}:00.0", index + 1)),
            compute_capability: Some(cc),
            memory: Some(nvml::NvmlMemory {
                total_mb,
                used_mb: 1500,
                free_mb: total_mb - 1500,
            }),
        }
    }

    /// The dev rig: RTX 4090 + WARP, NVML and nvcuda available.
    fn dev_rig_raw() -> RawProbe {
        let mut d3d12 = BTreeMap::new();
        d3d12.insert("00000000:00000000".to_string(), dxgi::D3d12Support::Level("12_2"));
        RawProbe {
            adapters: vec![
                adapter(0, VENDOR_NVIDIA, 0x2684, "NVIDIA GeForce RTX 4090", 24_142, Some("616.56")),
                adapter(1, VENDOR_MICROSOFT, 0x8C, "Microsoft Basic Render Driver", 0, None),
            ],
            d3d12,
            nvml: Some(nvml::NvmlSystem {
                driver_version: Some("616.56".into()),
                cuda_driver_version: Some(13020),
                nvml_version: Some("13.616.56".into()),
                devices: vec![nvml_dev(0, 0x2684, "NVIDIA GeForce RTX 4090", (8, 9), 24_564)],
            }),
            nvml_error: None,
            cuda_driver_version: Some("13.2".into()),
            cpu: CpuInfo::default(),
            total_ram_mb: 65_536,
            windows: true,
            probed_at_unix_ms: 1,
            notes: vec![],
            gpu_probe: GpuProbe::Complete,
        }
    }

    #[test]
    fn assembles_the_dev_rig() {
        let info = assemble(dev_rig_raw());
        assert_eq!(info.primary_gpu, Some(0));
        let g = info.primary().unwrap();
        assert_eq!(g.arch.as_deref(), Some("ada"));
        assert_eq!(g.compute_capability.as_deref(), Some("8.9"));
        assert_eq!(g.nvml_index, Some(0));
        assert_eq!(g.d3d12, Some(true));
        assert_eq!(g.d3d12_feature_level.as_deref(), Some("12_2"));
        assert!(!g.integrated);
        assert_eq!(info.nvidia_driver.as_deref(), Some("616.56"));
        assert_eq!(info.cuda_driver_version.as_deref(), Some("13.2"));
        assert_eq!(info.nvml_version.as_deref(), Some("13.616.56"));
        assert!(info.eps.cuda.eligible && info.eps.tensorrt_rtx.eligible && info.eps.directml.eligible);
        assert_eq!(info.eps.cuda.code, eps::OK);
        let warp = &info.gpus[1];
        assert!(warp.adapter.software && warp.arch.is_none() && warp.d3d12.is_none() && !warp.integrated);
        assert!(info.notes.is_empty(), "{:?}", info.notes);

        // Wire format: GpuAdapter fields are flattened next to the P1-D2 additions.
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["gpus"][0]["name"], "NVIDIA GeForce RTX 4090");
        assert_eq!(json["gpus"][0]["arch"], "ada");
        assert_eq!(json["gpus"][0]["dedicated_vram_mb"], 24_142);
        assert_eq!(json["eps"]["cuda"]["code"], "ok");
        let back: SystemInfo = serde_json::from_value(json).unwrap();
        assert_eq!(back, info);
    }

    /// Windows Sandbox with the virtual GPU off (found by the 0.5.0 packaged smoke): DXGI lists the Basic Render Driver
    /// three times (vendor 0x1414, device 0x8C), only the first with DXGI_ADAPTER_FLAG_SOFTWARE. The other two used to
    /// count as hardware GPUs: the smoke's no-gpu check failed and the startup probe recommended DirectML.
    #[test]
    fn a_clean_windows_with_unflagged_basic_render_drivers_has_no_hardware_gpu() {
        use crate::services::diagnostics::gpu_mem::is_software_adapter;
        const NAME: &str = "Microsoft Basic Render Driver";
        let warp = |index: u32, dxgi_flag: bool| GpuAdapter {
            software: is_software_adapter(dxgi_flag, VENDOR_MICROSOFT, 0x8C, NAME),
            ..adapter(index, VENDOR_MICROSOFT, 0x8C, NAME, 0, None)
        };
        let info = assemble(RawProbe {
            adapters: vec![warp(0, true), warp(1, false), warp(2, false)],
            windows: true,
            gpu_probe: GpuProbe::Complete,
            ..RawProbe::default()
        });
        assert_eq!(info.gpus.len(), 3);
        assert!(info.gpus.iter().all(|g| g.adapter.software && g.arch.is_none() && !g.integrated), "{:?}", info.gpus);
        assert_eq!(info.gpus.iter().filter(|g| !g.adapter.software).count(), 0, "no hardware GPU");
        assert_eq!(info.primary_gpu, None);
        assert!(!info.eps.directml.eligible);
        assert_eq!(info.eps.directml.code, eps::SOFTWARE_ONLY);
        assert!(!info.eps.cuda.eligible);
        // The regression itself: an entry without the DXGI flag is software too.
        assert!(warp(1, false).software);
    }

    #[test]
    fn falls_back_without_nvml_and_nvcuda() {
        let mut raw = dev_rig_raw();
        raw.nvml = None;
        raw.nvml_error = Some("nvml.dll was not found".into());
        raw.cuda_driver_version = None;
        let info = assemble(raw);
        let g = info.primary().unwrap();
        // Generation from the PCI id table; driver number from DXGI's UMD version.
        assert_eq!(g.arch.as_deref(), Some("ada"));
        assert_eq!(g.compute_capability, None);
        assert_eq!(info.nvidia_driver.as_deref(), Some("616.56"));
        assert!(info.eps.cuda.eligible);
        assert_eq!(info.eps.cuda.code, eps::OK);
        assert!(info.notes.iter().any(|n| n.contains("NVML")), "{:?}", info.notes);
    }

    #[test]
    fn primary_gpu_prefers_discrete_nvidia_then_vram() {
        // Laptop: Intel iGPU at index 0, NVIDIA dGPU at index 1.
        let mut igpu = gpu(0, VENDOR_INTEL, "Intel(R) UHD Graphics", 128, None, Some("xe-lp"));
        igpu.integrated = true;
        let dgpu = gpu(1, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4060 Laptop GPU", 8188, Some("8.9"), Some("ada"));
        assert_eq!(pick_primary(&[igpu.clone(), dgpu.clone()]), Some(1));
        // NVIDIA first even with less VRAM than an AMD card; then more VRAM; then lower index.
        let amd = gpu(0, VENDOR_AMD, "AMD Radeon RX 7900 XTX", 24_000, None, Some("rdna3"));
        assert_eq!(pick_primary(&[amd.clone(), dgpu.clone()]), Some(1));
        let big = gpu(2, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", 24_000, Some("8.9"), Some("ada"));
        assert_eq!(pick_primary(&[dgpu.clone(), big.clone()]), Some(2));
        let twin = gpu(3, VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", 24_000, Some("8.9"), Some("ada"));
        assert_eq!(pick_primary(&[twin, big]), Some(2));
        // A CUDA-capable NVIDIA card outranks an older NVIDIA card with more VRAM: RTX 3070 (8 GB)
        // next to a leftover GTX 1080 Ti (11 GB) used for OBS; RTX 4060 next to a Tesla P40.
        let rtx3070 = gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX 3070", 8192, Some("8.6"), Some("ampere"));
        let gtx1080ti = gpu(1, VENDOR_NVIDIA, "NVIDIA GeForce GTX 1080 Ti", 11_264, Some("6.1"), Some("pascal"));
        assert_eq!(pick_primary(&[rtx3070.clone(), gtx1080ti.clone()]), Some(0));
        assert_eq!(pick_primary(&[gtx1080ti.clone(), rtx3070.clone()]), Some(0), "order-independent");
        let p40 = gpu(0, VENDOR_NVIDIA, "Tesla P40", 24_576, Some("6.1"), Some("pascal"));
        assert_eq!(pick_primary(&[p40.clone(), dgpu.clone()]), Some(1));
        // Same rule when only the PCI-id generation is known (no NVML).
        let by_id_old = gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce GTX 1080 Ti", 11_264, None, Some("pascal"));
        let by_id_new = gpu(1, VENDOR_NVIDIA, "NVIDIA GeForce RTX 3070", 8192, None, Some("ampere"));
        assert_eq!(pick_primary(&[by_id_old, by_id_new]), Some(1));
        // An NVIDIA card of unknown generation is treated as CUDA-capable (like `unverified`).
        let unknown_gen = gpu(3, VENDOR_NVIDIA, "NVIDIA Future GPU", 8192, None, None);
        assert_eq!(pick_primary(&[amd.clone(), unknown_gen]), Some(3));
        // A card too old for CUDA ranks with the other vendors' cards: then VRAM decides.
        let amd_small = gpu(2, VENDOR_AMD, "AMD Radeon RX 6600", 8176, None, Some("rdna2"));
        assert_eq!(pick_primary(&[gtx1080ti.clone(), amd.clone()]), Some(0), "the 24 GB AMD card");
        assert_eq!(pick_primary(&[gtx1080ti, amd_small]), Some(1), "the 11 GB Pascal card");
        // Only integrated / software.
        assert_eq!(pick_primary(&[igpu]), Some(0));
        assert_eq!(
            pick_primary(&[gpu(0, VENDOR_MICROSOFT, "Microsoft Basic Render Driver", 0, None, None)]),
            None
        );
        assert_eq!(pick_primary(&[]), None);
    }

    fn skipped_raw(probed_at_unix_ms: u64) -> RawProbe {
        RawProbe {
            cpu: CpuInfo {
                brand: "X".into(),
                physical_cores: 6,
                logical_cores: 12,
                performance_cores: 6,
                ..CpuInfo::default()
            },
            windows: true,
            probed_at_unix_ms,
            notes: vec!["GPU details are missing: timeout".into()],
            gpu_probe: GpuProbe::Missing,
            ..RawProbe::default()
        }
    }

    #[test]
    fn a_skipped_gpu_probe_still_reports_the_cpu() {
        let info = assemble(skipped_raw(5));
        assert!(info.gpus.is_empty() && info.primary_gpu.is_none());
        assert_eq!(info.gpu_probe, GpuProbe::Missing);
        // Not "no graphics card": nothing is known about the GPUs.
        for e in [&info.eps.cuda, &info.eps.tensorrt, &info.eps.tensorrt_rtx, &info.eps.directml] {
            assert_eq!((e.eligible, e.code.as_str()), (false, eps::GPU_UNKNOWN));
        }
        assert_eq!(info.cpu.physical_cores, 6);
        assert_eq!(info.notes.len(), 1);
        let json = serde_json::to_value(&info).unwrap();
        assert_eq!(json["gpu_probe"], "missing");
        // Payloads from before the field existed read as complete.
        let mut old = serde_json::to_value(assemble(dev_rig_raw())).unwrap();
        old.as_object_mut().unwrap().remove("gpu_probe");
        assert_eq!(serde_json::from_value::<SystemInfo>(old).unwrap().gpu_probe, GpuProbe::Complete);
    }

    #[test]
    fn a_skipped_gpu_probe_keeps_the_earlier_gpu_facts() {
        let good = assemble(dev_rig_raw());
        let skipped = assemble(skipped_raw(99));

        // Refresh while the driver hangs: the 4090 stays, marked as from an earlier check; CPU facts
        // and the timestamp are the new probe's.
        let kept = keep_previous_gpu_facts(skipped.clone(), Some(&good));
        assert_eq!(kept.gpu_probe, GpuProbe::Previous);
        assert_eq!(kept.gpus, good.gpus);
        assert_eq!(kept.primary_gpu, Some(0));
        assert_eq!(kept.eps, good.eps);
        assert_eq!(kept.nvidia_driver, good.nvidia_driver);
        assert_eq!((kept.probed_at_unix_ms, kept.cpu.physical_cores), (99, 6));
        assert!(kept.notes.iter().any(|n| n.contains("earlier check")), "{:?}", kept.notes);
        // A second skipped refresh keeps them again.
        let again = keep_previous_gpu_facts(skipped.clone(), Some(&kept));
        assert_eq!((again.gpu_probe, again.gpus.len()), (GpuProbe::Previous, 2));

        // Nothing earlier (or nothing useful earlier): stays missing.
        assert_eq!(keep_previous_gpu_facts(skipped.clone(), None).gpu_probe, GpuProbe::Missing);
        assert_eq!(keep_previous_gpu_facts(skipped.clone(), Some(&skipped)).gpu_probe, GpuProbe::Missing);
        // A complete probe always replaces the cache, including a late answer.
        assert_eq!(keep_previous_gpu_facts(good.clone(), Some(&kept)), good);
    }

    #[test]
    fn a_missing_result_is_probed_again_once_the_driver_is_free() {
        let good = assemble(dev_rig_raw());
        let missing = assemble(skipped_raw(1_000_000));
        let retry = MISSING_RETRY.as_millis() as u64;
        assert!(servable(&good, false, u64::MAX), "complete results are served forever");
        assert!(servable(&missing, false, 1_000_000 + retry - 1), "not within the retry interval");
        assert!(!servable(&missing, false, 1_000_000 + retry), "retry once the interval passed");
        assert!(servable(&missing, true, u64::MAX), "never while a GPU probe still runs");
        let previous = keep_previous_gpu_facts(missing, Some(&good));
        assert!(servable(&previous, false, u64::MAX), "earlier facts are good enough");
    }

    #[test]
    fn a_late_gpu_answer_is_published_not_dropped() {
        use std::sync::mpsc;
        static BUSY: AtomicBool = AtomicBool::new(false);

        // Answers in time.
        let fast = answer_within(&BUSY, Duration::from_secs(5), || 7u32, |_, _| panic!("not late"));
        assert_eq!(fast, Ok(7));
        assert!(!BUSY.load(Ordering::Acquire));

        // Times out; a second probe meanwhile is refused; the late answer reaches `late`.
        let (release_tx, release_rx) = mpsc::channel::<()>();
        let (late_tx, late_rx) = mpsc::channel::<(u32, Duration)>();
        let slow = answer_within(
            &BUSY,
            Duration::from_millis(50),
            move || {
                let _ = release_rx.recv();
                42u32
            },
            move |v, took| {
                assert!(BUSY.load(Ordering::Acquire), "published before the busy flag clears");
                let _ = late_tx.send((v, took));
            },
        );
        assert!(slow.as_ref().is_err_and(|e| e.contains("did not answer")), "{slow:?}");
        let refused = answer_within(&BUSY, Duration::from_secs(5), || 1u32, |_, _| {});
        assert!(
            refused
                .as_ref()
                .is_err_and(|e| e.contains("has not finished")),
            "{refused:?}"
        );
        release_tx.send(()).unwrap();
        let (value, took) = late_rx
            .recv_timeout(Duration::from_secs(10))
            .expect("late answer");
        assert_eq!(value, 42);
        assert!(took >= Duration::from_millis(50));
        let deadline = Instant::now() + Duration::from_secs(10);
        while BUSY.load(Ordering::Acquire) && Instant::now() < deadline {
            std::thread::yield_now();
        }
        assert!(!BUSY.load(Ordering::Acquire), "the busy flag clears after the late answer");

        // A panicking probe reports at once instead of waiting for the timeout.
        let t0 = Instant::now();
        let crashed = answer_within(&BUSY, Duration::from_secs(30), || -> u32 { panic!("driver") }, |_, _| {});
        assert!(crashed.is_err() && t0.elapsed() < Duration::from_secs(10), "{crashed:?}");
    }

    /// Probes this PC and prints the report (`cargo test sysinfo -- --nocapture` shows it). On the
    /// dev rig the RTX 4090 is primary, Ada 8.9, and every GPU EP is eligible; on a CI runner there
    /// is no hardware GPU. Either way nothing panics and the cache serves the same result.
    #[test]
    fn probes_this_pc() {
        let info = cached();
        println!("{}", serde_json::to_string_pretty(&info).unwrap());
        assert!(info.cpu.logical_cores >= 1);
        assert!(info.probed_at_unix_ms > 0);
        if let Some(p) = info.primary() {
            assert!(!p.adapter.software);
            if p.adapter.vendor_id == VENDOR_NVIDIA && p.adapter.dedicated_vram_mb >= 1024 {
                assert!(p.arch.is_some(), "NVIDIA GPU without a generation: {p:?}");
                assert!(info.nvidia_driver.is_some());
            }
        }
        assert_eq!(cached(), info, "the cache must serve the first probe");
    }
}
