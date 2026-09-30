//! The resource governor for GPU speech models: whether a model may load on the GPU (admission),
//! the VRAM claims it holds while loaded, and the measured footprint the claims are resized to.
//!
//! # Admission ([`admit`], pure; [`admit_and_claim`] on live data)
//! The same inputs as `resources::policy` for the STT final (`RecommendationTopic::SttFinal`):
//! 1. A CPU request never touches the GPU.
//! 2. The primary GPU must be usable (`policy::room`: a discrete card of at least 3 GB whose probe
//!    answered) — else `no_gpu` / `gpu_unknown` / `integrated_gpu` / `small_gpu`.
//! 3. A GPU build for it: `cuda13` needs the CUDA EP to be eligible on the primary GPU
//!    (`policy::gpu_runtime_available`); `vulkan` is taken only when the spec asks for it (unbuilt and
//!    unmeasured, P1 outcomes) — else `no_gpu_runtime`.
//! 4. `auto` only: while the voice changer runs on the GPU of a card below the 16 GB class
//!    (`policy::LARGE_GPU_MB`), the GPU is the voice's (`vc_running`), as the router's
//!    `gpu_denied_vc_running`. A spec that names a GPU variant (the router's `gpu` tier, or an
//!    explicit choice) shares the GPU instead.
//! 5. VRAM: the model's footprint plus the runtime context (unless another engine already holds it)
//!    must fit the room the policy grants the STT final (card minus 30 % reserve, the voice changer
//!    and other subsystems) minus the STT claims already live (a second caption source's model) —
//!    else `gpu_full`.
//!
//! A denied model loads on the CPU (the engine reports why as its fallback reason) or, with
//! `cpuFallback: false`, fails (`OutOfMemory` for `gpu_full` / `vc_running`, `Unsupported` when the
//! PC has no GPU the engine can use). The live device check of the wrapper (free VRAM
//! for the load and before every run, ADR 0003b) still applies on top: the governor is about sharing
//! the card with the game, OBS and the voice changer; the live check keeps ggml-cuda away from its
//! abort paths.
//!
//! # Claims ([`GpuClaims`])
//! In-process engines hold `resources::claim_vram(Subsystem::SttFinal, ..)` for the model plus one
//! shared claim of [`GPU_CONTEXT_MB`] per GPU runtime. `native_runtime` pins GPU engine DLLs for
//! the process lifetime, and with them their CUDA context (measured: 378 MiB stay after the last
//! model is dropped), so once a model has run on a GPU runtime its context claim is pinned too
//! ([`GpuClaims::pin_context`]); before that it lives as long as the engines sharing it. An engine
//! in a worker process holds one `claim_external_vram` for its whole process (model + its own
//! context). Claims are taken under one lock with the admission, so two
//! engines loading at once cannot both take the last free VRAM. After the load the model claim is
//! resized to the measured process-VRAM delta, within ±25 % of the estimate ([`measured_model_mb`]),
//! unless another in-process claim changed during the load ([`ledger_fingerprint`]: the delta then
//! includes that load too, so the estimate stays). A model that ended on the CPU releases its model
//! claim; its context claim is pinned when the load initialized the GPU runtime anyway (the DLL
//! and its CUDA context stay), else released.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use crate::services::native_runtime::RuntimeVariant;
use crate::services::resources::{self, policy, RecommendationTopic, ResourceSnapshot, Subsystem, VramClaim};
use crate::services::sysinfo::{self, SystemInfo};

pub use super::models::GPU_CONTEXT_MB;
use super::types::same_pci_bus_id;
use super::DeviceInfo;

/// Machine codes of an [`Admission`] (the UI and logs map them; `fits` / `cpu_requested` carry no
/// fallback reason).
pub mod codes {
    pub const FITS: &str = "fits";
    pub const CPU_REQUESTED: &str = "cpu_requested";
    pub const NO_GPU: &str = "no_gpu";
    pub const GPU_UNKNOWN: &str = "gpu_unknown";
    pub const INTEGRATED_GPU: &str = "integrated_gpu";
    pub const SMALL_GPU: &str = "small_gpu";
    pub const NO_GPU_RUNTIME: &str = "no_gpu_runtime";
    pub const VC_RUNNING: &str = "vc_running";
    pub const GPU_FULL: &str = "gpu_full";
    /// Worker mode: the GPU worker crashed repeatedly, so it restarts on the CPU.
    pub const WORKER_CRASHED: &str = "worker_crashed";
}

/// What an engine spec asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Want {
    /// No variant named: the governor decides ("Best for my PC").
    Auto,
    /// A GPU variant named (the router's GPU tier, or an explicit choice): preferred when admitted.
    Gpu(RuntimeVariant),
    /// A CPU variant named, or the CPU requested: never the GPU.
    Cpu(Option<RuntimeVariant>),
}

impl Want {
    /// From an engine spec's `variant` (`None` / empty / unknown = auto).
    pub fn from_variant(variant: Option<&str>) -> Want {
        match variant.map(str::trim).filter(|v| !v.is_empty()) {
            None => Want::Auto,
            Some(v) if v.eq_ignore_ascii_case("gpu") => Want::Auto,
            Some(v) if v.eq_ignore_ascii_case("cpu-any") => Want::Cpu(None),
            Some(v) => match RuntimeVariant::parse(&v.to_ascii_lowercase()) {
                Some(rv) if rv.is_gpu() => Want::Gpu(rv),
                Some(rv) => Want::Cpu(Some(rv)),
                None => Want::Auto,
            },
        }
    }
}

/// The governor's answer for one model.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Admission {
    /// Load on this GPU variant (with the CPU variants as the fallback ladder); `None` = CPU only.
    pub gpu: Option<RuntimeVariant>,
    /// [`codes`] value.
    pub code: &'static str,
    /// Plain-English reason when a GPU was wanted but denied (the engine's fallback reason).
    pub reason: Option<String>,
    /// VRAM the load needs (model + runtime context when not held yet), MB.
    pub need_mb: u64,
    /// VRAM the governor found for STT models, MB.
    pub available_mb: u64,
}

impl Admission {
    fn cpu(code: &'static str, reason: Option<String>, need_mb: u64, available_mb: u64) -> Admission {
        Admission {
            gpu: None,
            code,
            reason,
            need_mb,
            available_mb,
        }
    }

    /// A CPU run chosen by the caller (worker restarts, tests).
    pub fn forced_cpu(code: &'static str, reason: impl Into<String>) -> Admission {
        Admission::cpu(code, Some(reason.into()), 0, 0)
    }
}

/// `5300` → `"5.2 GB"`.
fn gb(mb: u64) -> String {
    let tenths = mb.saturating_mul(10).saturating_add(512) / 1024;
    format!("{}.{} GB", tenths / 10, tenths % 10)
}

/// STT claims already live (other caption sources' models and the shared runtime context).
fn live_stt_claims_mb(snap: &ResourceSnapshot) -> u64 {
    snap.claims
        .iter()
        .filter(|c| c.subsystem == Subsystem::SttFinal)
        .fold(0u64, |acc, c| acc.saturating_add(c.mb))
}

/// Admission of a model of `model_mb` (pure and deterministic; see the module docs).
/// `context_held`: an engine of the same GPU runtime is loaded (its context claim is live and
/// counted in `snap`).
pub fn admit(want: Want, model_mb: u64, title: &str, context_held: bool, sys: &SystemInfo, snap: &ResourceSnapshot) -> Admission {
    let variant = match want {
        Want::Cpu(_) => return Admission::cpu(codes::CPU_REQUESTED, None, 0, 0),
        Want::Gpu(v) => v,
        Want::Auto => RuntimeVariant::Cuda13,
    };
    let topic = RecommendationTopic::SttFinal;
    let room = policy::room(topic, sys, snap);
    let context_mb = if context_held { 0 } else { GPU_CONTEXT_MB };
    let need_mb = model_mb.saturating_add(context_mb);
    let denied = |code: &'static str, why: String| Admission::cpu(code, Some(format!("{title} runs on the CPU: {why}")), need_mb, 0);
    match room.gpu {
        policy::GpuClass::Usable => {}
        policy::GpuClass::NoGpu => return denied(codes::NO_GPU, "no graphics card was found".into()),
        policy::GpuClass::Unknown => {
            return denied(
                codes::GPU_UNKNOWN,
                "the graphics driver did not answer, so the GPU could not be checked".into(),
            )
        }
        policy::GpuClass::Integrated => {
            return denied(
                codes::INTEGRATED_GPU,
                "the graphics card is integrated (it shares system memory)".into(),
            )
        }
        policy::GpuClass::TooSmall => return denied(codes::SMALL_GPU, format!("the graphics card has only {}", gb(room.total_mb))),
    }
    let runtime_ok = match variant {
        RuntimeVariant::Cuda13 => policy::gpu_runtime_available(topic, sys),
        // Only on request until the vulkan pack is built and measured (P1 outcomes).
        RuntimeVariant::Vulkan => matches!(want, Want::Gpu(RuntimeVariant::Vulkan)),
        _ => false,
    };
    if !runtime_ok {
        return denied(
            codes::NO_GPU_RUNTIME,
            "Sigil's GPU speech engine needs an NVIDIA GeForce GTX 16 / RTX 20 or newer with driver R580 or newer".into(),
        );
    }
    if want == Want::Auto && room.vc_on_gpu && room.total_mb < policy::LARGE_GPU_MB {
        return denied(
            codes::VC_RUNNING,
            format!(
                "the voice changer is using this {} graphics card (captions move to the GPU on 16 GB cards, or when the GPU tier is chosen)",
                gb(room.total_mb)
            ),
        );
    }
    let available_mb = room.available_mb.saturating_sub(live_stt_claims_mb(snap));
    if need_mb > available_mb {
        let mut why = format!("it needs {} of video memory and {} is free for it", gb(need_mb), gb(available_mb));
        if room.vc_mb > 0 {
            why.push_str(&format!(" ({} is kept for the voice changer)", gb(room.vc_mb)));
        }
        return Admission {
            available_mb,
            ..denied(codes::GPU_FULL, why)
        };
    }
    Admission {
        gpu: Some(variant),
        code: codes::FITS,
        reason: None,
        need_mb,
        available_mb,
    }
}

// ---------------------------------------------------------------------------------------------
// Claims
// ---------------------------------------------------------------------------------------------

/// Admission + claim are one step for every speech engine (two loads cannot both take the last
/// free VRAM).
static ADMISSION: Mutex<()> = Mutex::new(());
/// The shared runtime-context claim per GPU runtime (alive while any in-process engine of it is).
static CONTEXTS: Mutex<BTreeMap<RuntimeVariant, Weak<VramClaim>>> = Mutex::new(BTreeMap::new());
/// Context claims of GPU runtimes that ran a model: their DLL (and its CUDA context) stays loaded
/// for the process lifetime, so the claim does too.
static PINNED: Mutex<BTreeMap<RuntimeVariant, Arc<VramClaim>>> = Mutex::new(BTreeMap::new());

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The live context claim of `variant`, if an in-process engine holds it (or it is pinned).
fn context_claim(variant: RuntimeVariant) -> Option<Arc<VramClaim>> {
    if let Some(c) = lock(&PINNED).get(&variant) {
        return Some(Arc::clone(c));
    }
    lock(&CONTEXTS).get(&variant).and_then(Weak::upgrade)
}

/// The context claim of `variant`, created when no engine holds it (`true` = created now).
fn context_claim_or_new(variant: RuntimeVariant) -> (Arc<VramClaim>, bool) {
    if let Some(c) = lock(&PINNED).get(&variant) {
        return (Arc::clone(c), false);
    }
    let mut map = lock(&CONTEXTS);
    if let Some(c) = map.get(&variant).and_then(Weak::upgrade) {
        return (c, false);
    }
    let claim = Arc::new(resources::claim_vram(
        Subsystem::SttFinal,
        &format!("speech: {} runtime context (transcribe.cpp)", variant.label()),
        GPU_CONTEXT_MB,
    ));
    map.insert(variant, Arc::downgrade(&claim));
    (claim, true)
}

/// The VRAM claims of one speech engine; dropping it releases them (declare it after the model in
/// the engine so the model's memory is freed first).
#[derive(Debug, Default)]
pub struct GpuClaims {
    model: Option<VramClaim>,
    context: Option<Arc<VramClaim>>,
    /// The GPU runtime of `context`.
    variant: Option<RuntimeVariant>,
    /// The context claim was created by this engine (its load paid for the context).
    context_created: bool,
    model_mb: u64,
}

impl GpuClaims {
    /// No claims (a CPU engine).
    pub fn none() -> GpuClaims {
        GpuClaims::default()
    }

    /// Whether this engine created its runtime's context claim.
    pub fn context_created(&self) -> bool {
        self.context_created
    }

    /// The claimed model (or, for a worker process, the whole process) footprint in MB.
    pub fn model_mb(&self) -> u64 {
        if self.model.is_some() {
            self.model_mb
        } else {
            0
        }
    }

    /// Everything this engine holds: model plus the context it shares (MB).
    pub fn total_mb(&self) -> u64 {
        self.model_mb() + if self.context.is_some() { GPU_CONTEXT_MB } else { 0 }
    }

    pub fn is_empty(&self) -> bool {
        self.model.is_none() && self.context.is_none()
    }

    /// Resize the model (or worker process) claim to a measured footprint.
    pub fn resize_model(&mut self, mb: u64) {
        if let Some(c) = &self.model {
            c.resize(mb);
            self.model_mb = mb;
        }
    }

    /// The model ran on its GPU runtime: keep the runtime's context claim for the process lifetime
    /// (the loader keeps GPU DLLs, and so their CUDA context, loaded until Sigil exits).
    pub fn pin_context(&self) {
        if let (Some(c), Some(v)) = (&self.context, self.variant) {
            lock(&PINNED)
                .entry(v)
                .or_insert_with(|| Arc::clone(c));
        }
    }

    /// Release everything (the model moved to the CPU).
    pub fn release(&mut self) {
        self.model = None;
        self.context = None;
        self.model_mb = 0;
    }

    /// Tests: an external claim of `mb` on the real ledger (released on drop).
    #[cfg(test)]
    pub(crate) fn test_external(label: &str, mb: u64) -> GpuClaims {
        GpuClaims {
            model: Some(resources::claim_external_vram(Subsystem::SttFinal, label, mb)),
            context: None,
            variant: None,
            context_created: true,
            model_mb: mb,
        }
    }
}

/// Admit a model on live data and take its claims in the same step. `external`: the model runs in
/// a worker process (one external claim of model + context). `label` names the claim in the
/// resources view (no secrets). CPU admissions hold nothing.
pub fn admit_and_claim(want: Want, model_mb: u64, title: &str, label: &str, external: bool) -> (Admission, GpuClaims) {
    let _one_at_a_time = lock(&ADMISSION);
    let sys = sysinfo::cached();
    let snap = resources::snapshot();
    let variant = match want {
        Want::Gpu(v) => Some(v),
        Want::Auto => Some(RuntimeVariant::Cuda13),
        Want::Cpu(_) => None,
    };
    let context_held = !external && variant.and_then(context_claim).is_some();
    let admission = admit(want, model_mb, title, context_held, &sys, &snap);
    let Some(v) = admission.gpu else {
        return (admission, GpuClaims::none());
    };
    let claims = if external {
        GpuClaims {
            model: Some(resources::claim_external_vram(Subsystem::SttFinal, label, admission.need_mb)),
            context: None,
            variant: None,
            context_created: true,
            model_mb: admission.need_mb,
        }
    } else {
        let (context, created) = context_claim_or_new(v);
        GpuClaims {
            model: Some(resources::claim_vram(Subsystem::SttFinal, label, model_mb)),
            context: Some(context),
            variant: Some(v),
            context_created: created,
            model_mb,
        }
    };
    (admission, claims)
}

// ---------------------------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------------------------

/// This process's VRAM use on a DXGI adapter (a snapshot).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VramSample {
    pub adapter: u32,
    pub process_mb: u64,
}

/// The DXGI adapter the governor budgets for (the primary GPU), if any.
pub fn primary_adapter() -> Option<u32> {
    sysinfo::cached()
        .primary()
        .filter(|g| !g.integrated && !g.adapter.software)
        .map(|g| g.adapter.index)
}

/// Sample this process's VRAM on `adapter`.
pub fn sample(adapter: u32) -> Option<VramSample> {
    crate::services::diagnostics::gpu_mem::process_vram_usage(adapter).map(|u| VramSample {
        adapter,
        process_mb: u.process_usage_mb,
    })
}

/// The DXGI adapter of an engine device (PCI bus id, else a unique adapter name), else `None`.
pub fn adapter_of(device: &DeviceInfo, sys: &SystemInfo) -> Option<u32> {
    if let Some(id) = device.device_id.as_deref() {
        if let Some(g) = sys
            .gpus
            .iter()
            .find(|g| g.pci_bus_id.as_deref().is_some_and(|p| same_pci_bus_id(p, id)))
        {
            return Some(g.adapter.index);
        }
    }
    let name = device.description.as_deref()?.trim();
    let mut same = sys.gpus.iter().filter(|g| g.adapter.name.trim() == name);
    match (same.next(), same.next()) {
        (Some(g), None) => Some(g.adapter.index),
        _ => None,
    }
}

/// The model's own footprint from process-VRAM samples around its load (pure): the delta, minus
/// the runtime context when this load created it, kept within ±25 % of the estimate. The samples
/// cover the whole process, so a load that overlaps another one here (a second caption source, the
/// voice changer's ORT session) inflates the delta; the clamp bounds what such an overlap can add
/// to the ledger (the engine also keeps the estimate when other claims changed during its load,
/// [`ledger_fingerprint`]). `None` when the numbers are implausible (VRAM went down meanwhile, or
/// the delta is below a third or above three times the estimate).
pub fn measured_model_mb(before_mb: u64, after_mb: u64, context_created: bool, estimate_mb: u64) -> Option<u64> {
    let delta = after_mb.checked_sub(before_mb)?;
    let model = delta.saturating_sub(if context_created { GPU_CONTEXT_MB } else { 0 });
    let low = estimate_mb / 3;
    let high = estimate_mb.saturating_mul(3);
    if model == 0 || model < low || model > high {
        return None;
    }
    let quarter = estimate_mb / 4;
    Some(model.clamp(estimate_mb - quarter, estimate_mb + quarter).max(1))
}

/// The in-process VRAM claims of the ledger (pure; `(id, mb)` sorted): two equal fingerprints
/// around a load mean no other claim of this process was taken, resized or released meanwhile.
/// Claims of helper processes (`external`) are left out: their VRAM is not in this process's
/// samples.
pub fn ledger_fingerprint(snap: &ResourceSnapshot) -> Vec<(u64, u64)> {
    let mut claims: Vec<(u64, u64)> = snap
        .claims
        .iter()
        .filter(|c| !c.external)
        .map(|c| (c.id, c.mb))
        .collect();
    claims.sort_unstable();
    claims
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_NVIDIA};
    use crate::services::resources::VramClaimInfo;
    use crate::services::sysinfo::test_support::{gpu, system};
    use crate::services::sysinfo::GpuProbe;

    /// One GPU with an R616 / CUDA 13.2 driver when it is NVIDIA (EP eligibility assessed for real).
    fn pc(vendor: u32, name: &str, vram_mb: u64, cc: Option<&str>, arch: Option<&str>) -> SystemInfo {
        system(vec![gpu(0, vendor, name, vram_mb, cc, arch)])
    }

    fn rtx4060() -> SystemInfo {
        pc(VENDOR_NVIDIA, "NVIDIA GeForce RTX 4060", 8188, Some("8.9"), Some("ada"))
    }

    fn rtx4090() -> SystemInfo {
        pc(VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", 24_564, Some("8.9"), Some("ada"))
    }

    fn snap(vc_running: bool, vc_gpu: Option<bool>, claims: Vec<(Subsystem, u64)>) -> ResourceSnapshot {
        ResourceSnapshot {
            claims: claims
                .into_iter()
                .enumerate()
                .map(|(i, (subsystem, mb))| VramClaimInfo {
                    id: i as u64 + 1,
                    subsystem,
                    label: "test".into(),
                    mb,
                    external: false,
                })
                .collect(),
            vc_running,
            vc_gpu,
            ..ResourceSnapshot::default()
        }
    }

    const QWEN_Q8: u64 = 3100;

    #[test]
    fn wants_follow_the_spec_variant() {
        assert_eq!(Want::from_variant(None), Want::Auto);
        assert_eq!(Want::from_variant(Some(" ")), Want::Auto);
        assert_eq!(Want::from_variant(Some("cuda13")), Want::Gpu(RuntimeVariant::Cuda13));
        assert_eq!(Want::from_variant(Some("Vulkan")), Want::Gpu(RuntimeVariant::Vulkan));
        assert_eq!(Want::from_variant(Some("cpu-avx2")), Want::Cpu(Some(RuntimeVariant::CpuAvx2)));
        assert_eq!(Want::from_variant(Some("cpu-any")), Want::Cpu(None));
        assert_eq!(Want::from_variant(Some("gpu")), Want::Auto);
        assert_eq!(Want::from_variant(Some("tpu")), Want::Auto, "unknown: the governor decides");
    }

    /// The acceptance rule of P2-D: the GPU is used only when the governor allows it.
    #[test]
    fn the_gpu_is_admitted_only_when_the_governor_allows_it() {
        let idle = snap(false, None, vec![]);
        // 8 GB card, nothing else: 8188 - 2456 reserve = 5732 free; Qwen3 Q8 + context = 3500 fits.
        let a = admit(Want::Auto, QWEN_Q8, "Qwen3-ASR", false, &rtx4060(), &idle);
        assert_eq!((a.gpu, a.code, a.need_mb, a.available_mb), (Some(RuntimeVariant::Cuda13), codes::FITS, 3500, 5732));
        assert!(a.reason.is_none());
        // The context already held by another engine is not paid twice.
        assert_eq!(admit(Want::Auto, QWEN_Q8, "Qwen3-ASR", true, &rtx4060(), &idle).need_mb, QWEN_Q8);

        // Auto keeps an 8 GB card for a voice changer running on it...
        let vc = snap(true, Some(true), vec![]);
        let a = admit(Want::Auto, QWEN_Q8, "Qwen3-ASR", false, &rtx4060(), &vc);
        assert_eq!((a.gpu, a.code), (None, codes::VC_RUNNING));
        assert!(a.reason.as_deref().unwrap().contains("voice changer"), "{a:?}");
        // ... the GPU tier shares it when it fits (8188 - 2456 - 1200 assumed VC = 4532) ...
        let a = admit(Want::Gpu(RuntimeVariant::Cuda13), QWEN_Q8, "Qwen3-ASR", false, &rtx4060(), &vc);
        assert_eq!((a.gpu, a.available_mb), (Some(RuntimeVariant::Cuda13), 4532));
        // ... a voice changer on the CPU takes no GPU, and a 24 GB card has room for both.
        assert!(admit(Want::Auto, QWEN_Q8, "q", false, &rtx4060(), &snap(true, Some(false), vec![])).gpu.is_some());
        assert!(admit(Want::Auto, QWEN_Q8, "q", false, &rtx4090(), &vc).gpu.is_some());

        // Full: a measured voice changer claim plus an LLM leave too little.
        let busy = snap(true, Some(true), vec![(Subsystem::Vc, 900), (Subsystem::Llm, 3000)]);
        let a = admit(Want::Gpu(RuntimeVariant::Cuda13), QWEN_Q8, "Qwen3-ASR", false, &rtx4060(), &busy);
        assert_eq!((a.gpu, a.code), (None, codes::GPU_FULL));
        let why = a.reason.unwrap();
        assert!(why.contains("3.4 GB") && why.contains("1.8 GB") && why.contains("voice changer"), "{why}");

        // A live STT model of another caption source counts (the policy ignores its own
        // subsystem because a recommendation replaces it; a second engine does not).
        let other = snap(false, None, vec![(Subsystem::SttFinal, 3100), (Subsystem::SttFinal, 400)]);
        let a = admit(Want::Auto, QWEN_Q8, "q", true, &rtx4060(), &other);
        assert_eq!((a.gpu, a.available_mb), (None, 5732 - 3500));
        assert!(admit(Want::Auto, QWEN_Q8, "q", true, &rtx4090(), &other).gpu.is_some());

        // A CPU request never touches the GPU and gives no reason.
        let a = admit(Want::Cpu(None), QWEN_Q8, "q", false, &rtx4090(), &idle);
        assert_eq!((a.gpu, a.code, a.reason), (None, codes::CPU_REQUESTED, None));
    }

    #[test]
    fn gpus_the_speech_engine_cannot_use_are_denied_with_a_reason() {
        let idle = snap(false, None, vec![]);
        // AMD: no CUDA build; vulkan only when explicitly asked for.
        let amd = pc(VENDOR_AMD, "AMD Radeon RX 7800 XT", 16_368, None, Some("rdna3"));
        let a = admit(Want::Auto, QWEN_Q8, "q", false, &amd, &idle);
        assert_eq!((a.gpu, a.code), (None, codes::NO_GPU_RUNTIME));
        assert_eq!(admit(Want::Gpu(RuntimeVariant::Cuda13), QWEN_Q8, "q", false, &amd, &idle).gpu, None);
        assert_eq!(
            admit(Want::Gpu(RuntimeVariant::Vulkan), QWEN_Q8, "q", false, &amd, &idle).gpu,
            Some(RuntimeVariant::Vulkan)
        );
        // Pascal (CUDA not eligible), small, integrated, none, unknown.
        let pascal = pc(VENDOR_NVIDIA, "NVIDIA GeForce GTX 1080 Ti", 11_264, Some("6.1"), Some("pascal"));
        assert_eq!(admit(Want::Auto, QWEN_Q8, "q", false, &pascal, &idle).code, codes::NO_GPU_RUNTIME);
        let small = pc(VENDOR_NVIDIA, "NVIDIA GeForce GTX 1650", 2048, Some("7.5"), Some("turing"));
        assert_eq!(admit(Want::Auto, QWEN_Q8, "q", false, &small, &idle).code, codes::SMALL_GPU);
        let mut integrated = pc(VENDOR_AMD, "AMD Radeon(TM) Graphics", 512, None, None);
        integrated.gpus[0].integrated = true;
        integrated.primary_gpu = Some(0);
        assert_eq!(admit(Want::Auto, QWEN_Q8, "q", false, &integrated, &idle).code, codes::INTEGRATED_GPU);
        let none = SystemInfo::default();
        assert_eq!(admit(Want::Auto, QWEN_Q8, "q", false, &none, &idle).code, codes::NO_GPU);
        let unknown = SystemInfo {
            gpu_probe: GpuProbe::Missing,
            ..SystemInfo::default()
        };
        assert_eq!(admit(Want::Auto, QWEN_Q8, "q", false, &unknown, &idle).code, codes::GPU_UNKNOWN);
        // Every denial explains itself.
        for s in [&amd, &pascal, &small, &integrated, &none, &unknown] {
            let a = admit(Want::Auto, QWEN_Q8, "Qwen3-ASR", false, s, &idle);
            assert!(a.reason.as_deref().is_some_and(|r| r.starts_with("Qwen3-ASR runs on the CPU: ")), "{a:?}");
        }
    }

    #[test]
    fn measured_footprints_are_used_only_when_plausible() {
        // ADR 0003b: 0 -> 3470 MiB for Qwen3 Q8 with a fresh CUDA context (380 MiB).
        assert_eq!(measured_model_mb(0, 3470, true, 3100), Some(3070));
        assert_eq!(measured_model_mb(400, 3470, false, 3100), Some(3070));
        assert_eq!(measured_model_mb(3470, 3000, false, 3100), None, "VRAM went down meanwhile");
        assert_eq!(measured_model_mb(0, 300, true, 3100), None, "less than the context");
        assert_eq!(measured_model_mb(0, 20_000, false, 3100), None, "someone else loaded too");
        // Two Qwen3 loads at once on a 24 GB card: each sees both (3070 + 3070); the claim is
        // held near the estimate instead of doubling.
        assert_eq!(measured_model_mb(0, 6540, true, 3100), Some(3100 + 775));
        // A model lighter than its estimate (whisper-tiny: about its file) is kept within 25 %.
        assert_eq!(measured_model_mb(0, 400 + 100, true, 150), Some(113));
        assert_eq!(measured_model_mb(0, 400 + 170, true, 150), Some(170));
    }

    #[test]
    fn the_ledger_fingerprint_sees_other_in_process_claims_only() {
        let mut s = snap(false, None, vec![(Subsystem::SttFinal, 3100), (Subsystem::Llm, 3000)]);
        let a = ledger_fingerprint(&s);
        assert_eq!(a, vec![(1, 3100), (2, 3000)]);
        // A helper process loading (an external claim) does not touch this process's VRAM.
        s.claims.push(VramClaimInfo {
            id: 9,
            subsystem: Subsystem::Llm,
            label: "llama-server".into(),
            mb: 5000,
            external: true,
        });
        assert_eq!(ledger_fingerprint(&s), a);
        // Another engine of this process claiming or resizing does.
        s.claims[1].mb = 2800;
        assert_ne!(ledger_fingerprint(&s), a);
        s.claims.reverse();
        assert_eq!(ledger_fingerprint(&s), vec![(1, 3100), (2, 2800)], "order-independent");
    }

    #[test]
    fn engine_devices_map_to_dxgi_adapters() {
        let mut sys = rtx4090();
        sys.gpus[0].pci_bus_id = Some("00000000:01:00.0".into());
        sys.gpus
            .push(gpu(1, VENDOR_AMD, "AMD Radeon(TM) Graphics", 512, None, None));
        let cuda = DeviceInfo {
            description: Some("NVIDIA GeForce RTX 4090".into()),
            device_id: Some("0000:01:00.0".into()),
            kind: Some("cuda".into()),
            ..DeviceInfo::default()
        };
        assert_eq!(adapter_of(&cuda, &sys), Some(0), "NVML and CUDA spell the PCI id differently");
        let by_name = DeviceInfo {
            device_id: None,
            ..cuda.clone()
        };
        assert_eq!(adapter_of(&by_name, &sys), Some(0));
        let unknown = DeviceInfo {
            description: Some("Some GPU".into()),
            device_id: Some("0000:07:00.0".into()),
            ..DeviceInfo::default()
        };
        assert_eq!(adapter_of(&unknown, &sys), None);
    }

    #[test]
    fn claims_are_shared_per_runtime_and_released_with_the_engine() {
        // Uses the real ledger; unique labels keep parallel tests apart.
        let find = |label: &str| -> Vec<u64> {
            resources::snapshot()
                .claims
                .into_iter()
                .filter(|c| c.label == label)
                .map(|c| c.mb)
                .collect()
        };
        let (a, created_a) = context_claim_or_new(RuntimeVariant::Vulkan);
        let (b, created_b) = context_claim_or_new(RuntimeVariant::Vulkan);
        assert!(created_a || !created_b, "the second engine shares the first one's context");
        assert!(!created_b);
        assert_eq!(a.id(), b.id());
        let label = "speech: Vulkan runtime context (transcribe.cpp)";
        assert_eq!(find(label), vec![GPU_CONTEXT_MB]);
        drop(a);
        assert_eq!(find(label).len(), 1, "still held by the second engine");
        drop(b);
        assert!(find(label).is_empty());
        assert!(context_claim(RuntimeVariant::Vulkan).is_none());

        let mut claims = GpuClaims {
            model: Some(resources::claim_vram(Subsystem::SttFinal, "test: governor model", 3100)),
            context: None,
            variant: None,
            context_created: false,
            model_mb: 3100,
        };
        claims.resize_model(3070);
        assert_eq!(find("test: governor model"), vec![3070]);
        assert_eq!((claims.model_mb(), claims.total_mb()), (3070, 3070));
        claims.release();
        assert!(claims.is_empty() && find("test: governor model").is_empty());
    }

    #[test]
    fn a_context_that_ran_a_model_stays_claimed_like_its_pinned_dll() {
        // The baseline CPU variant stands in for a GPU runtime (no other test uses its context).
        let v = RuntimeVariant::Cpu;
        let label = "speech: CPU runtime context (transcribe.cpp)";
        let count = || resources::snapshot().claims.iter().filter(|c| c.label == label).count();
        let (context, created) = context_claim_or_new(v);
        assert!(created);
        let first = GpuClaims {
            context: Some(context),
            variant: Some(v),
            context_created: true,
            ..GpuClaims::default()
        };
        first.pin_context();
        drop(first);
        assert_eq!(count(), 1, "pinned: the DLL keeps its CUDA context after the last model");
        let (again, created) = context_claim_or_new(v);
        assert!(!created, "the next load does not pay for the context again");
        assert!(context_claim(v).is_some());
        drop(again);
        assert_eq!(count(), 1);
    }
}
