//! The server plan: which runtime variant, how many layers on the GPU, context, flash attention,
//! the VRAM estimate and whether the resource governor allows it. Pure and deterministic (the same
//! inputs always give the same plan), unit-tested as a table like `resources::policy`.
//!
//! # Rules
//! 1. **Room**: `resources::policy::room(MtLlm, ..)`, the same arithmetic the recommendation uses
//!    (dedicated VRAM minus the reserve for Windows, the game and OBS, minus the voice changer,
//!    minus other Sigil models; the server's own claim does not count against itself).
//! 2. **Footprint** at the chosen context: the catalog's `min_vram_mb` (at 4096 tokens) plus the
//!    KV cache of the extra context; for a user GGUF, weights + KV cache + compute buffers +
//!    runtime, from its header.
//! 3. **The 8 GB rule** (research §3.1, as in the policy): while the voice changer runs on the GPU
//!    of a card under [`policy::MID_GPU_MB`], models above [`VC_SMALL_CARD_MAX_MODEL_MB`] (the 7B
//!    class and up) are refused, whatever the arithmetic says: it protects the voice's GPU time as
//!    well as its memory. 1.8B and 4B-class models may run next to it when they fit.
//! 4. **Variant**: `auto` runs on the GPU when the model fits there completely, preferring CUDA 13
//!    when the primary GPU can use it and its pack is installed (Vulkan otherwise, which covers
//!    NVIDIA, AMD and Intel), and on the CPU pack otherwise. An explicit `vulkan` / `cuda13` never
//!    falls back to the CPU: it is refused when the model does not fit. `cpu` always runs on the
//!    CPU (when the model fits in RAM).
//! 5. `gpuLayers` set by the user offloads that many layers (partial offload), refused when the
//!    estimate exceeds the room. `allowOverBudget` starts a refused plan anyway, except when the
//!    runtime cannot run the model at all (a hard refusal, which also lists nothing to download).
//!    A refused plan reports the placement it refused (GPU variant, all layers or the hand-set
//!    count, the VRAM it needs), so an accepted override runs exactly that, with its VRAM claim.

use super::catalog::{Availability, DEFAULT_CTX, RUNTIME_CUDA13};
use super::LlmServerPlan;
use crate::services::diagnostics::gpu_mem::VENDOR_NVIDIA;
use crate::services::resources::policy::{self, GpuClass, Room};
use crate::services::resources::{RecommendationTopic, ResourceSnapshot};
use crate::services::sysinfo::{eps, SystemInfo};

/// `-ngl` meaning "every layer" (llama.cpp clamps it to the model's layer count).
pub const ALL_LAYERS: u32 = 999;
/// Largest model (MiB at the default context) allowed next to a voice changer on the GPU of an
/// 8 GB-class card: the 1.8B and 4B-class models, not the 7B class.
pub const VC_SMALL_CARD_MAX_MODEL_MB: u64 = 4608;
/// Largest context offered (tokens).
pub const MAX_CTX: u32 = 131_072;
pub const MIN_CTX: u32 = 512;
/// Runtime overhead of a user GGUF besides weights and KV cache (backend, scratch), MiB.
const BYO_RUNTIME_MB: u64 = 150;
/// Compute buffers of a user GGUF: a fixed part plus the logits of one 512-token micro-batch.
const BYO_COMPUTE_BASE_MB: u64 = 256;
/// RAM kept for Windows, the game and Sigil when a model runs on the CPU.
const CPU_RAM_RESERVE_MB: u64 = 4096;
const MIB: u64 = 1024 * 1024;

/// Reason codes beyond `resources::policy::Reason`.
pub mod reason {
    /// The runtime cannot run this model (e.g. the 1.25-bit build needs a newer llama.cpp).
    pub const NEEDS_RUNTIME_UPDATE: &str = "needs_runtime_update";
    /// The user chose the CPU variant.
    pub const CPU_SELECTED: &str = "cpu_selected";
    /// The user set the number of GPU layers (partial offload).
    pub const USER_LAYERS: &str = "user_layers";
    /// The model does not fit in the PC's memory.
    pub const LOW_MEMORY: &str = "low_memory";
}

/// What the plan knows about the model.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModelFacts {
    /// Catalog id or absolute path (what the plan reports as `model`).
    pub id: String,
    pub label: String,
    /// A catalog model (else a user file).
    pub catalog: bool,
    pub availability: Availability,
    /// GGUF bytes (download size for catalog models, file size for user files).
    pub size_bytes: u64,
    /// Catalog footprint at [`DEFAULT_CTX`] (`min_vram_mb`).
    pub vram_default_ctx_mb: Option<u64>,
    pub kv_bytes_per_token: Option<u64>,
    pub layers: Option<u32>,
    /// Trained context (caps `-c`).
    pub ctx_train: Option<u64>,
    pub vocab_size: Option<u64>,
    /// The model file is present (catalog: installed).
    pub installed: bool,
}

/// Normalized knobs of a config (see `LlmServerConfig`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Knobs {
    /// `auto`, `vulkan`, `cuda13`, `cpu`.
    pub variant: String,
    pub ctx_size: Option<u32>,
    pub gpu_layers: Option<u32>,
    /// `auto`, `on`, `off`.
    pub flash_attn: String,
    pub allow_over_budget: bool,
}

/// Inputs from the running system.
pub struct Env<'a> {
    pub sys: &'a SystemInfo,
    pub snap: &'a ResourceSnapshot,
    /// Whether a runtime pack id is installed.
    pub runtime_installed: &'a dyn Fn(&str) -> bool,
}

/// The plan plus the launch details that are not part of the wire struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Planned {
    pub plan: LlmServerPlan,
    /// `-t`.
    pub threads: u32,
    /// `--poll 0`: no busy-waiting CPU threads while the GPU works (full offload).
    pub no_poll: bool,
    /// Run the helper below normal priority (CPU inference next to a game).
    pub below_normal: bool,
    /// The refusal cannot be overridden with `allowOverBudget` (the runtime cannot run it).
    pub hard_refusal: bool,
}

/// The primary GPU can run llama.cpp's CUDA 13 build.
pub fn cuda_usable(sys: &SystemInfo) -> bool {
    sys.eps.cuda.eligible
        && sys
            .primary()
            .is_some_and(|g| g.adapter.vendor_id == VENDOR_NVIDIA && !g.integrated && eps::meets_cuda_floor(g) != Some(false))
}

/// Clamp the context to what the model and Sigil allow.
pub fn effective_ctx(knobs: &Knobs, model: &ModelFacts) -> u32 {
    let train_cap = model
        .ctx_train
        .filter(|&c| c >= u64::from(MIN_CTX))
        .map(|c| c.min(u64::from(MAX_CTX)) as u32)
        .unwrap_or(MAX_CTX);
    knobs
        .ctx_size
        .unwrap_or_else(super::catalog::ctx_default)
        .clamp(MIN_CTX, MAX_CTX.min(train_cap).max(MIN_CTX))
}

fn mb_ceil(bytes: u64) -> u64 {
    bytes.div_ceil(MIB)
}

/// KV cache (MiB) of `ctx` tokens.
fn kv_mb(model: &ModelFacts, ctx: u32) -> Option<u64> {
    model
        .kv_bytes_per_token
        .map(|b| mb_ceil(b.saturating_mul(u64::from(ctx))))
}

/// Estimated VRAM with every layer on the GPU at `ctx`.
pub fn full_vram_mb(model: &ModelFacts, ctx: u32) -> u64 {
    if let Some(base) = model.vram_default_ctx_mb {
        // Catalog footprint at 4096 tokens; add the KV cache of a longer context.
        let extra = if ctx > DEFAULT_CTX {
            model
                .kv_bytes_per_token
                .map(|b| mb_ceil(b.saturating_mul(u64::from(ctx - DEFAULT_CTX))))
                .unwrap_or_else(|| base.saturating_mul(u64::from(ctx - DEFAULT_CTX)) / (4 * u64::from(DEFAULT_CTX)))
        } else {
            0
        };
        return base.saturating_add(extra);
    }
    // A user GGUF: weights + KV cache + compute buffers + runtime. Without a KV shape, assume a
    // 7B-class 128 KiB per token.
    let weights = mb_ceil(model.size_bytes);
    let kv = kv_mb(model, ctx).unwrap_or_else(|| mb_ceil(131_072 * u64::from(ctx)));
    let vocab = model.vocab_size.unwrap_or(128_000);
    let logits = mb_ceil(vocab.saturating_mul(4 * 512));
    weights
        .saturating_add(kv)
        .saturating_add(BYO_COMPUTE_BASE_MB)
        .saturating_add(logits)
        .saturating_add(BYO_RUNTIME_MB)
}

/// VRAM with `layers` of the model's layers on the GPU (the fixed buffers stay whole).
fn partial_vram_mb(model: &ModelFacts, ctx: u32, layers: u32) -> u64 {
    let full = full_vram_mb(model, ctx);
    let Some(total) = model.layers.filter(|&n| n > 0) else {
        return full; // unknown depth: assume everything is offloaded (conservative)
    };
    if layers >= total {
        return full;
    }
    let fixed = (BYO_COMPUTE_BASE_MB + BYO_RUNTIME_MB).min(full);
    fixed.saturating_add(full.saturating_sub(fixed).saturating_mul(u64::from(layers)) / u64::from(total))
}

/// RAM a CPU run needs (weights are memory-mapped but read constantly).
fn ram_needed_mb(model: &ModelFacts, ctx: u32) -> u64 {
    mb_ceil(model.size_bytes)
        .saturating_add(kv_mb(model, ctx).unwrap_or(0))
        .saturating_add(1024)
}

fn cpu_threads(sys: &SystemInfo) -> u32 {
    let cores = if sys.cpu.performance_cores > 0 {
        sys.cpu.performance_cores
    } else {
        sys.cpu.physical_cores
    };
    let cores = if cores == 0 { 4 } else { cores };
    cores.saturating_sub(2).clamp(2, 8)
}

fn gb(mb: u64) -> String {
    let tenths = mb.saturating_mul(10).saturating_add(512) / 1024;
    format!("{}.{} GB", tenths / 10, tenths % 10)
}

/// Why a model that does not fit on the GPU does not fit (policy reason codes).
fn downgrade_code(need: u64, room: &Room) -> &'static str {
    if room.gpu != GpuClass::Usable {
        return cpu_code(room);
    }
    if room.vc_on_gpu && need <= room.available_mb.saturating_add(room.vc_mb) {
        "vc_running"
    } else if room.others_mb > 0
        && need
            <= room
                .available_mb
                .saturating_add(room.vc_mb)
                .saturating_add(room.others_mb)
    {
        "shared_gpu"
    } else if need <= room.total_mb.saturating_sub(room.reserve_mb) {
        "gpu_full"
    } else {
        "small_gpu"
    }
}

fn cpu_code(room: &Room) -> &'static str {
    match room.gpu {
        GpuClass::NoGpu => "no_gpu",
        GpuClass::Unknown => "gpu_unknown",
        GpuClass::Integrated => "integrated_gpu",
        GpuClass::TooSmall => "small_gpu",
        GpuClass::Usable => "gpu_full",
    }
}

fn room_text(room: &Room) -> String {
    match room.gpu {
        GpuClass::NoGpu => "no graphics card was found".into(),
        GpuClass::Unknown => "the graphics card details could not be read".into(),
        GpuClass::Integrated => "the graphics card is integrated".into(),
        GpuClass::TooSmall => format!("the graphics card has only {}", gb(room.total_mb)),
        GpuClass::Usable => {
            let mut parts = vec![format!("{} of {} free for it", gb(room.available_mb), gb(room.total_mb))];
            if room.vc_mb > 0 {
                parts.push(format!("{} for the voice changer", gb(room.vc_mb)));
            }
            if room.others_mb > 0 {
                parts.push(format!("{} for other Sigil models", gb(room.others_mb)));
            }
            parts.join(", ")
        }
    }
}

fn why(code: &str, label: &str, need: u64) -> String {
    match code {
        "fits" => "it fits".into(),
        "vc_running" => format!("{label} ({}) would leave too little for the voice changer running on the GPU", gb(need)),
        "shared_gpu" => format!("{label} ({}) does not fit next to the other models Sigil has loaded", gb(need)),
        "gpu_full" | "small_gpu" => format!("{label} needs {}", gb(need)),
        "no_gpu_runtime" => "llama.cpp's CUDA build needs an NVIDIA GeForce GTX 16 / RTX 20 or newer with driver 580 or newer".into(),
        reason::NEEDS_RUNTIME_UPDATE => format!("{label} needs a newer llama.cpp than Sigil ships"),
        reason::CPU_SELECTED => "the processor version was chosen".into(),
        reason::USER_LAYERS => "the number of GPU layers was set by hand".into(),
        reason::LOW_MEMORY => format!("{label} needs about {} of memory", gb(need)),
        "no_gpu" | "gpu_unknown" | "integrated_gpu" => "the GPU cannot be used".into(),
        _ => code.to_string(),
    }
}

/// Plan `model` with `knobs` on this system.
pub fn plan(knobs: &Knobs, model: &ModelFacts, env: &Env) -> Planned {
    let room = policy::room(RecommendationTopic::MtLlm, env.sys, env.snap);
    let ctx = effective_ctx(knobs, model);
    let flash_attn = match knobs.flash_attn.as_str() {
        "on" | "off" => knobs.flash_attn.clone(),
        _ => "auto".to_string(),
    };
    let full = full_vram_mb(model, ctx);
    let label = model.label.as_str();
    let cuda_ok = cuda_usable(env.sys);
    let gpu_variant = |forced: &str| -> String {
        match forced {
            "vulkan" | "cuda13" => forced.to_string(),
            _ if cuda_ok && (env.runtime_installed)(RUNTIME_CUDA13) => "cuda13".into(),
            _ => "vulkan".into(),
        }
    };

    // (variant, gpu layers, vram, allowed, code, hard). A refusal keeps the placement it refused
    // (all layers, or the hand-set count, on the GPU variant): `allowOverBudget` launches exactly
    // that, with its `-ngl`, device pin and VRAM claim, never a silent CPU run.
    struct D {
        variant: String,
        layers: u32,
        vram: u64,
        allowed: bool,
        code: &'static str,
        hard: bool,
    }
    let cpu = |code: &'static str| -> D {
        let need = ram_needed_mb(model, ctx);
        let ram_ok = env.sys.total_ram_mb == 0 || need <= env.sys.total_ram_mb.saturating_sub(CPU_RAM_RESERVE_MB);
        D {
            variant: "cpu".into(),
            layers: 0,
            vram: 0,
            allowed: ram_ok,
            code: if ram_ok { code } else { reason::LOW_MEMORY },
            hard: false,
        }
    };
    // A refused GPU placement: `layers` on `variant` (a GPU variant), needing `need` MiB.
    let refuse = |variant: String, layers: u32, code: &'static str, need: u64, hard: bool| D {
        variant,
        layers,
        vram: need,
        allowed: false,
        code,
        hard,
    };
    let vc_rule = room.vc_on_gpu && room.total_mb < policy::MID_GPU_MB && full > VC_SMALL_CARD_MAX_MODEL_MB;
    let cpu_chosen = knobs.variant == "cpu" || knobs.gpu_layers == Some(0);

    let d = if model.availability == Availability::NeedsRuntimeUpdate {
        if cpu_chosen {
            D {
                hard: true,
                allowed: false,
                code: reason::NEEDS_RUNTIME_UPDATE,
                ..cpu(reason::NEEDS_RUNTIME_UPDATE)
            }
        } else {
            refuse(gpu_variant(&knobs.variant), ALL_LAYERS, reason::NEEDS_RUNTIME_UPDATE, full, true)
        }
    } else if cpu_chosen {
        cpu(reason::CPU_SELECTED)
    } else if knobs.variant == "cuda13" && !cuda_ok {
        refuse("cuda13".into(), knobs.gpu_layers.unwrap_or(ALL_LAYERS).min(ALL_LAYERS), "no_gpu_runtime", full, true)
    } else if let Some(n) = knobs.gpu_layers {
        // Partial (or full) offload chosen by hand.
        let variant = gpu_variant(&knobs.variant);
        let layers = n.min(ALL_LAYERS);
        let need = partial_vram_mb(model, ctx, layers);
        if room.gpu != GpuClass::Usable && room.gpu != GpuClass::TooSmall {
            refuse(variant, layers, cpu_code(&room), need, false)
        } else if vc_rule && need > VC_SMALL_CARD_MAX_MODEL_MB {
            refuse(variant, layers, "vc_running", need, false)
        } else if need > room.available_mb {
            refuse(variant, layers, downgrade_code(need, &room), need, false)
        } else {
            D {
                variant,
                layers,
                vram: need,
                allowed: true,
                code: reason::USER_LAYERS,
                hard: false,
            }
        }
    } else {
        let forced_gpu = matches!(knobs.variant.as_str(), "vulkan" | "cuda13");
        if room.gpu != GpuClass::Usable {
            if forced_gpu {
                refuse(gpu_variant(&knobs.variant), ALL_LAYERS, cpu_code(&room), full, false)
            } else {
                cpu(cpu_code(&room))
            }
        } else if vc_rule {
            // Research §3.1: the 7B class only while the voice changer is off the GPU (8 GB cards).
            refuse(gpu_variant(&knobs.variant), ALL_LAYERS, "vc_running", full, false)
        } else if full <= room.available_mb {
            D {
                variant: gpu_variant(&knobs.variant),
                layers: ALL_LAYERS,
                vram: full,
                allowed: true,
                code: "fits",
                hard: false,
            }
        } else if forced_gpu {
            refuse(gpu_variant(&knobs.variant), ALL_LAYERS, downgrade_code(full, &room), full, false)
        } else {
            cpu(downgrade_code(full, &room))
        }
    };

    // Allowed or refused, `on_gpu` is the placement the plan is about (a refusal keeps its layers).
    let on_gpu = d.layers > 0 && d.variant != "cpu";
    let device = if on_gpu { "gpu" } else { "cpu" };
    // Downloads are offered only for a plan that can run (allowed, or refused but overridable): a
    // hard refusal never asks for a model or pack the runtime cannot use.
    let mut missing = Vec::new();
    if !d.hard {
        if model.catalog && !model.installed {
            missing.push(model.id.clone());
        }
        if let Some(rt) = super::catalog::runtime_id(&d.variant) {
            if !(env.runtime_installed)(rt) {
                missing.push(rt.to_string());
            }
        }
    }
    let what = if !d.allowed {
        label.to_string()
    } else if on_gpu {
        let layers = if d.layers >= ALL_LAYERS {
            "all layers".to_string()
        } else {
            format!("{} layers", d.layers)
        };
        format!("{label} on the GPU ({}, {}, {layers})", gb(d.vram), d.variant)
    } else {
        format!("{label} on the CPU")
    };
    let need_for_text = if d.code == reason::LOW_MEMORY { ram_needed_mb(model, ctx) } else { full };
    let verdict = if d.allowed { "" } else { "refused: " };
    let reason_text = format!("{what}: {verdict}{}; {}.", why(d.code, label, need_for_text), room_text(&room));
    let threads = if on_gpu && d.layers >= ALL_LAYERS { 2 } else { cpu_threads(env.sys) };
    Planned {
        plan: LlmServerPlan {
            model: model.id.clone(),
            variant: d.variant.clone(),
            gpu_layers: if on_gpu { d.layers } else { 0 },
            ctx_size: ctx,
            flash_attn,
            device: device.into(),
            vram_mb: if d.allowed && !on_gpu { 0 } else { d.vram },
            available_mb: room.available_mb,
            allowed: d.allowed,
            reason_code: d.code.into(),
            reason: reason_text,
            missing,
        },
        threads,
        no_poll: on_gpu && d.layers >= ALL_LAYERS,
        below_normal: !on_gpu || d.layers < ALL_LAYERS,
        hard_refusal: d.hard,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_INTEL, VENDOR_MICROSOFT};
    use crate::services::resources::{Subsystem, VramBudget, VramClaimInfo};
    use crate::services::sysinfo::test_support::{gpu, system};

    const GB8: u64 = 8188;
    const GB12: u64 = 12_282;
    const GB24: u64 = 24_142;

    fn pc(vram_mb: u64) -> SystemInfo {
        system(vec![gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX", vram_mb, Some("8.9"), Some("ada"))])
    }

    fn amd(vram_mb: u64) -> SystemInfo {
        system(vec![gpu(0, VENDOR_AMD, "AMD Radeon RX 7800 XT", vram_mb, None, Some("rdna3"))])
    }

    fn cpu_only() -> SystemInfo {
        system(vec![gpu(0, VENDOR_MICROSOFT, "Microsoft Basic Render Driver", 0, None, None)])
    }

    fn igpu() -> SystemInfo {
        let mut g = gpu(0, VENDOR_INTEL, "Intel(R) UHD Graphics", 128, None, Some("xe-lp"));
        g.integrated = true;
        system(vec![g])
    }

    fn snap(vc_gpu: bool, claims: Vec<VramClaimInfo>) -> ResourceSnapshot {
        ResourceSnapshot {
            budget: VramBudget::default(),
            claims,
            vc_running: vc_gpu,
            vc_gpu: vc_gpu.then_some(true),
        }
    }

    fn claim(subsystem: Subsystem, mb: u64) -> VramClaimInfo {
        VramClaimInfo {
            id: 1,
            subsystem,
            label: "x".into(),
            mb,
            external: true,
        }
    }

    fn model(id: &str, vram: u64, size_gb_tenths: u64) -> ModelFacts {
        ModelFacts {
            id: id.into(),
            label: id.rsplit('/').next().unwrap().into(),
            catalog: true,
            size_bytes: size_gb_tenths * 100 * MIB,
            vram_default_ctx_mb: Some(vram),
            kv_bytes_per_token: Some(131_072),
            layers: Some(32),
            installed: true,
            ..Default::default()
        }
    }

    fn hy18() -> ModelFacts {
        let mut m = model("mt/hy-mt2-1.8b-q4_k_m", 1600, 11);
        m.kv_bytes_per_token = Some(65_536);
        m
    }

    fn hy7() -> ModelFacts {
        model("mt/hy-mt2-7b-q4_k_m", 5300, 46)
    }

    fn knobs(variant: &str) -> Knobs {
        Knobs {
            variant: variant.into(),
            ctx_size: None,
            gpu_layers: None,
            flash_attn: "auto".into(),
            allow_over_budget: false,
        }
    }

    fn run(k: &Knobs, m: &ModelFacts, sys: &SystemInfo, s: &ResourceSnapshot, installed: &[&str]) -> Planned {
        let inst = |id: &str| installed.contains(&id);
        plan(
            k,
            m,
            &Env {
                sys,
                snap: s,
                runtime_installed: &inst,
            },
        )
    }

    const ALL_PACKS: [&str; 3] = [super::super::catalog::RUNTIME_VULKAN, RUNTIME_CUDA13, super::super::catalog::RUNTIME_CPU];

    struct Row {
        name: &'static str,
        k: Knobs,
        m: ModelFacts,
        sys: SystemInfo,
        s: ResourceSnapshot,
        allowed: bool,
        variant: &'static str,
        device: &'static str,
        code: &'static str,
    }

    #[test]
    fn plan_table() {
        let rows = vec![
            // --- the acceptance case: 7B refused on an 8 GB card while the voice changer runs ---
            Row {
                name: "8 GB, VC on the GPU: Hy-MT2 7B refused",
                k: knobs("auto"),
                m: hy7(),
                sys: pc(GB8),
                s: snap(true, vec![]),
                allowed: false,
                variant: "cuda13",
                device: "gpu",
                code: "vc_running",
            },
            Row {
                name: "8 GB, VC on the GPU with a small measured claim: still refused",
                k: knobs("auto"),
                m: hy7(),
                sys: pc(GB8),
                s: snap(true, vec![claim(Subsystem::Vc, 300)]),
                allowed: false,
                variant: "cuda13",
                device: "gpu",
                code: "vc_running",
            },
            Row {
                name: "8 GB, VC on the GPU, forced Vulkan: refused",
                k: knobs("vulkan"),
                m: hy7(),
                sys: pc(GB8),
                s: snap(true, vec![]),
                allowed: false,
                variant: "vulkan",
                device: "gpu",
                code: "vc_running",
            },
            Row {
                name: "8 GB, VC off: Hy-MT2 7B on the GPU",
                k: knobs("auto"),
                m: hy7(),
                sys: pc(GB8),
                s: snap(false, vec![]),
                allowed: true,
                variant: "cuda13",
                device: "gpu",
                code: "fits",
            },
            Row {
                name: "8 GB, VC on the GPU: Hy-MT2 1.8B fits next to it",
                k: knobs("auto"),
                m: hy18(),
                sys: pc(GB8),
                s: snap(true, vec![]),
                allowed: true,
                variant: "cuda13",
                device: "gpu",
                code: "fits",
            },
            Row {
                name: "8 GB, VC on the GPU: a 4B-class model may run when it fits",
                k: knobs("auto"),
                m: model("mt/milmmt-46-4b-q4_k_m", 3000, 25),
                sys: pc(GB8),
                s: snap(true, vec![]),
                allowed: true,
                variant: "cuda13",
                device: "gpu",
                code: "fits",
            },
            Row {
                name: "8 GB, VC on the CPU (0 MB claim): 7B fits",
                k: knobs("auto"),
                m: hy7(),
                sys: pc(GB8),
                s: ResourceSnapshot {
                    vc_running: true,
                    vc_gpu: Some(false),
                    claims: vec![claim(Subsystem::Vc, 0)],
                    ..Default::default()
                },
                allowed: true,
                variant: "cuda13",
                device: "gpu",
                code: "fits",
            },
            Row {
                name: "8 GB, VC on the GPU, CPU variant chosen: 7B on the CPU",
                k: knobs("cpu"),
                m: hy7(),
                sys: pc(GB8),
                s: snap(true, vec![]),
                allowed: true,
                variant: "cpu",
                device: "cpu",
                code: reason::CPU_SELECTED,
            },
            Row {
                name: "12 GB, VC on: 7B fits (above the 8 GB rule)",
                k: knobs("auto"),
                m: hy7(),
                sys: pc(GB12),
                s: snap(true, vec![]),
                allowed: true,
                variant: "cuda13",
                device: "gpu",
                code: "fits",
            },
            Row {
                name: "12 GB, VC + STT final: 7B does not fit → CPU",
                k: knobs("auto"),
                m: hy7(),
                sys: pc(GB12),
                s: snap(true, vec![claim(Subsystem::SttFinal, 3500)]),
                allowed: true,
                variant: "cpu",
                device: "cpu",
                code: "shared_gpu",
            },
            Row {
                name: "12 GB, VC + STT final, forced Vulkan: refused",
                k: knobs("vulkan"),
                m: hy7(),
                sys: pc(GB12),
                s: snap(true, vec![claim(Subsystem::SttFinal, 3500)]),
                allowed: false,
                variant: "vulkan",
                device: "gpu",
                code: "shared_gpu",
            },
            Row {
                name: "12 GB, STT final only: shared GPU → CPU",
                k: knobs("auto"),
                m: model("mt/milmmt-46-12b-q4_k_m", 8100, 73),
                sys: pc(GB12),
                s: snap(false, vec![claim(Subsystem::SttFinal, 3500)]),
                allowed: true,
                variant: "cpu",
                device: "cpu",
                code: "shared_gpu",
            },
            Row {
                name: "the server's own claim does not count against itself",
                k: knobs("auto"),
                m: hy7(),
                sys: pc(GB12),
                s: snap(true, vec![claim(Subsystem::Llm, 5300)]),
                allowed: true,
                variant: "cuda13",
                device: "gpu",
                code: "fits",
            },
            Row {
                name: "24 GB: 30B-A3B does not fit (19.6 GB > 16.9 GB)",
                k: knobs("auto"),
                m: model("mt/hy-mt2-30b-a3b-q4_k_m", 19_600, 182),
                sys: pc(GB24),
                s: snap(false, vec![]),
                allowed: true,
                variant: "cpu",
                device: "cpu",
                code: "small_gpu",
            },
            Row {
                name: "AMD: Vulkan",
                k: knobs("auto"),
                m: hy7(),
                sys: amd(16_368),
                s: snap(false, vec![]),
                allowed: true,
                variant: "vulkan",
                device: "gpu",
                code: "fits",
            },
            Row {
                name: "AMD, CUDA forced: refused for good",
                k: knobs("cuda13"),
                m: hy18(),
                sys: amd(16_368),
                s: snap(false, vec![]),
                allowed: false,
                variant: "cuda13",
                device: "gpu",
                code: "no_gpu_runtime",
            },
            Row {
                name: "no GPU: CPU",
                k: knobs("auto"),
                m: hy18(),
                sys: cpu_only(),
                s: snap(false, vec![]),
                allowed: true,
                variant: "cpu",
                device: "cpu",
                code: "no_gpu",
            },
            Row {
                name: "no GPU, Vulkan forced: refused",
                k: knobs("vulkan"),
                m: hy18(),
                sys: cpu_only(),
                s: snap(false, vec![]),
                allowed: false,
                variant: "vulkan",
                device: "gpu",
                code: "no_gpu",
            },
            Row {
                name: "integrated GPU: CPU",
                k: knobs("auto"),
                m: hy18(),
                sys: igpu(),
                s: snap(false, vec![]),
                allowed: true,
                variant: "cpu",
                device: "cpu",
                code: "integrated_gpu",
            },
            Row {
                name: "1.25-bit build: refused (llama.cpp lacks STQ1_0)",
                k: knobs("auto"),
                m: ModelFacts {
                    availability: Availability::NeedsRuntimeUpdate,
                    ..model("mt/hy-mt2-1.8b-1.25bit", 700, 4)
                },
                sys: pc(GB24),
                s: snap(false, vec![]),
                allowed: false,
                variant: "cuda13",
                device: "gpu",
                code: reason::NEEDS_RUNTIME_UPDATE,
            },
            Row {
                name: "hand-set layers: partial offload next to the VC",
                k: Knobs {
                    gpu_layers: Some(12),
                    ..knobs("auto")
                },
                m: hy7(),
                sys: pc(GB8),
                s: snap(true, vec![]),
                allowed: true,
                variant: "cuda13",
                device: "gpu",
                code: reason::USER_LAYERS,
            },
            Row {
                name: "hand-set layers that do not fit: refused",
                k: Knobs {
                    gpu_layers: Some(32),
                    ..knobs("vulkan")
                },
                m: hy7(),
                sys: pc(GB8),
                s: snap(true, vec![]),
                allowed: false,
                variant: "vulkan",
                device: "gpu",
                code: "vc_running",
            },
            Row {
                name: "gpu layers 0 = CPU",
                k: Knobs {
                    gpu_layers: Some(0),
                    ..knobs("vulkan")
                },
                m: hy18(),
                sys: pc(GB24),
                s: snap(false, vec![]),
                allowed: true,
                variant: "cpu",
                device: "cpu",
                code: reason::CPU_SELECTED,
            },
        ];
        for r in rows {
            let p = run(&r.k, &r.m, &r.sys, &r.s, &ALL_PACKS);
            assert_eq!(
                (
                    p.plan.allowed,
                    p.plan.variant.as_str(),
                    p.plan.device.as_str(),
                    p.plan.reason_code.as_str()
                ),
                (r.allowed, r.variant, r.device, r.code),
                "{}: {}",
                r.name,
                p.plan.reason
            );
            assert!(p.plan.reason.ends_with('.'), "{}", r.name);
            assert_eq!(p.plan.model, r.m.id);
            if p.plan.allowed && p.plan.device == "cpu" {
                assert_eq!((p.plan.gpu_layers, p.plan.vram_mb), (0, 0), "{}", r.name);
                assert!(p.below_normal && !p.no_poll, "{}", r.name);
            }
            if p.plan.allowed && p.plan.device == "gpu" {
                assert!(
                    p.plan.gpu_layers > 0 && p.plan.vram_mb > 0 && p.plan.vram_mb <= p.plan.available_mb,
                    "{}",
                    r.name
                );
            }
            // A refused GPU plan keeps the placement it refused (what an override launches).
            if !p.plan.allowed && p.plan.device == "gpu" {
                assert!(p.plan.gpu_layers > 0 && p.plan.vram_mb > 0, "{}: {:?}", r.name, p.plan);
                assert_eq!(p.no_poll, p.plan.gpu_layers >= ALL_LAYERS, "{}", r.name);
            }
            if p.hard_refusal {
                assert!(p.plan.missing.is_empty(), "{}: a hard refusal offers no download", r.name);
            }
        }
    }

    #[test]
    fn refusals_keep_the_placement_an_override_launches() {
        // The acceptance refusal: all layers on the GPU, the full estimate, full-offload threading.
        let p = run(&knobs("auto"), &hy7(), &pc(GB8), &snap(true, vec![]), &ALL_PACKS);
        assert!(!p.plan.allowed);
        assert_eq!((p.plan.gpu_layers, p.plan.vram_mb, p.plan.device.as_str()), (ALL_LAYERS, 5300, "gpu"));
        assert!(p.no_poll && !p.below_normal);
        assert_eq!(p.threads, 2);
        // Hand-set layers keep their count.
        let k = Knobs {
            gpu_layers: Some(32),
            ..knobs("vulkan")
        };
        let p = run(&k, &hy7(), &pc(GB8), &snap(true, vec![]), &ALL_PACKS);
        assert!(!p.plan.allowed && !p.hard_refusal);
        assert_eq!((p.plan.gpu_layers, p.plan.variant.as_str()), (32, "vulkan"));
        assert!(p.plan.vram_mb > 0);
        // A partial hand-set offload that is refused keeps its count and runs below normal priority.
        let k = Knobs {
            gpu_layers: Some(20),
            ..knobs("vulkan")
        };
        let crowded = snap(false, vec![claim(Subsystem::SttFinal, 5000)]);
        let p = run(&k, &hy7(), &pc(GB8), &crowded, &ALL_PACKS);
        assert!(!p.plan.allowed, "{}", p.plan.reason);
        assert_eq!(p.plan.gpu_layers, 20);
        assert!(!p.no_poll && p.below_normal);
        // A refused CPU plan (not enough memory) stays on the CPU.
        let mut sys = cpu_only();
        sys.total_ram_mb = 8192;
        let p = run(&knobs("auto"), &hy7(), &sys, &snap(false, vec![]), &ALL_PACKS);
        assert!(!p.plan.allowed);
        assert_eq!((p.plan.gpu_layers, p.plan.vram_mb, p.plan.device.as_str()), (0, 0, "cpu"));
    }

    #[test]
    fn hard_refusals_offer_no_downloads() {
        // AMD with CUDA forced: the CUDA pack can never run here, so it is not offered.
        let p = run(&knobs("cuda13"), &hy18(), &amd(16_368), &snap(false, vec![]), &[]);
        assert!(p.hard_refusal && p.plan.missing.is_empty(), "{:?}", p.plan.missing);
        // The 1.25-bit build, not installed: neither the model nor a pack.
        let tiny = ModelFacts {
            availability: Availability::NeedsRuntimeUpdate,
            installed: false,
            ..model("mt/hy-mt2-1.8b-1.25bit", 700, 4)
        };
        let p = run(&knobs("auto"), &tiny, &pc(GB24), &snap(false, vec![]), &[]);
        assert!(p.hard_refusal && p.plan.missing.is_empty(), "{:?}", p.plan.missing);
        let p = run(&knobs("cpu"), &tiny, &pc(GB24), &snap(false, vec![]), &[]);
        assert!(p.hard_refusal && !p.plan.allowed && p.plan.missing.is_empty());
        assert_eq!((p.plan.variant.as_str(), p.plan.reason_code.as_str()), ("cpu", reason::NEEDS_RUNTIME_UPDATE));
        // An overridable refusal still lists what it needs.
        let mut m = hy7();
        m.installed = false;
        let p = run(&knobs("auto"), &m, &pc(GB8), &snap(true, vec![]), &[]);
        assert!(!p.plan.allowed && !p.hard_refusal);
        assert_eq!(p.plan.missing, vec![m.id.clone(), super::super::catalog::RUNTIME_VULKAN.to_string()]);
    }

    #[test]
    fn the_acceptance_refusal_explains_itself_and_can_be_overridden() {
        let p = run(&knobs("auto"), &hy7(), &pc(GB8), &snap(true, vec![]), &ALL_PACKS);
        assert!(!p.plan.allowed && !p.hard_refusal);
        assert!(p.plan.reason.contains("voice changer"), "{}", p.plan.reason);
        assert_eq!(p.plan.vram_mb, 5300);
        // 8188 - 2456 reserve - 1200 voice changer = 4532 MiB.
        assert_eq!(p.plan.available_mb, 4532);
        let hard = run(&knobs("cuda13"), &hy18(), &amd(16_368), &snap(false, vec![]), &ALL_PACKS);
        assert!(hard.hard_refusal);
    }

    #[test]
    fn variant_prefers_cuda_only_when_installed_and_usable() {
        let installed = [super::super::catalog::RUNTIME_VULKAN];
        let p = run(&knobs("auto"), &hy18(), &pc(GB24), &snap(false, vec![]), &installed);
        assert_eq!(p.plan.variant, "vulkan", "the CUDA pack is not installed: Vulkan, no extra download");
        assert!(p.plan.missing.is_empty());
        let p = run(&knobs("auto"), &hy18(), &pc(GB24), &snap(false, vec![]), &[]);
        assert_eq!(p.plan.missing, vec![super::super::catalog::RUNTIME_VULKAN.to_string()]);
        let mut old = pc(GB24);
        old.eps = eps::assess(&old.gpus, Some("566.36"), Some("12.7"), true);
        let p = run(&knobs("auto"), &hy18(), &old, &snap(false, vec![]), &ALL_PACKS);
        assert_eq!(p.plan.variant, "vulkan", "pre-R580 driver: no CUDA 13");
        // A missing model is reported with the pack.
        let mut m = hy18();
        m.installed = false;
        let p = run(&knobs("cpu"), &m, &pc(GB24), &snap(false, vec![]), &[]);
        assert_eq!(p.plan.missing, vec![m.id.clone(), super::super::catalog::RUNTIME_CPU.to_string()]);
    }

    #[test]
    fn context_scales_the_estimate_and_is_clamped() {
        let m = hy7();
        assert_eq!(full_vram_mb(&m, 4096), 5300);
        assert_eq!(full_vram_mb(&m, 2048), 5300, "a shorter context never lowers the catalog footprint");
        // 128 KiB per token x 28672 extra tokens = 3.5 GiB.
        assert_eq!(full_vram_mb(&m, 32_768), 5300 + 3584);
        let k = Knobs {
            ctx_size: Some(1_000_000),
            ..knobs("auto")
        };
        assert_eq!(effective_ctx(&k, &m), MAX_CTX);
        let k = Knobs {
            ctx_size: Some(10),
            ..knobs("auto")
        };
        assert_eq!(effective_ctx(&k, &m), MIN_CTX);
        let short = ModelFacts {
            ctx_train: Some(2048),
            ..hy7()
        };
        assert_eq!(effective_ctx(&knobs("auto"), &short), 2048);
        // A long context no longer fits on 8 GB next to nothing else: the plan falls to the CPU.
        let k = Knobs {
            ctx_size: Some(32_768),
            ..knobs("auto")
        };
        let p = run(&k, &hy7(), &pc(GB8), &snap(false, vec![]), &ALL_PACKS);
        assert_eq!((p.plan.device.as_str(), p.plan.ctx_size), ("cpu", 32_768));
        // Without a KV shape, a quarter of the footprint per 4096 extra tokens.
        let no_kv = ModelFacts {
            kv_bytes_per_token: None,
            ..hy7()
        };
        assert_eq!(full_vram_mb(&no_kv, 8192), 5300 + 1325);
    }

    #[test]
    fn user_gguf_estimate_comes_from_its_header() {
        let byo = ModelFacts {
            id: r"D:\m\x.gguf".into(),
            label: "x.gguf".into(),
            catalog: false,
            size_bytes: 4 * 1024 * MIB,
            kv_bytes_per_token: Some(131_072),
            layers: Some(32),
            vocab_size: Some(128_000),
            installed: true,
            ..Default::default()
        };
        // 4096 weights + 512 KV + 256 compute + 250 logits (128k vocab x 512 x 4 B) + 150 runtime.
        assert_eq!(full_vram_mb(&byo, 4096), 4096 + 512 + 256 + 250 + 150);
        assert!(partial_vram_mb(&byo, 4096, 16) < full_vram_mb(&byo, 4096));
        assert_eq!(partial_vram_mb(&byo, 4096, 64), full_vram_mb(&byo, 4096));
        let p = run(&knobs("auto"), &byo, &pc(GB24), &snap(false, vec![]), &ALL_PACKS);
        assert!(p.plan.allowed && p.plan.missing.is_empty());
        assert_eq!(p.plan.device, "gpu");
    }

    #[test]
    fn cpu_runs_need_the_memory() {
        let mut sys = cpu_only();
        sys.total_ram_mb = 8192;
        let p = run(&knobs("auto"), &hy7(), &sys, &snap(false, vec![]), &ALL_PACKS);
        assert_eq!(
            (p.plan.allowed, p.plan.reason_code.as_str()),
            (false, reason::LOW_MEMORY),
            "{}",
            p.plan.reason
        );
        let p = run(&knobs("auto"), &hy18(), &sys, &snap(false, vec![]), &ALL_PACKS);
        assert!(p.plan.allowed, "{}", p.plan.reason);
        assert_eq!(p.threads, 6, "8 cores - 2");
    }

    #[test]
    fn plans_are_deterministic() {
        let s = snap(true, vec![claim(Subsystem::SttFinal, 2000)]);
        let first = run(&knobs("auto"), &hy7(), &pc(GB12), &s, &ALL_PACKS);
        for _ in 0..20 {
            assert_eq!(run(&knobs("auto"), &hy7(), &pc(GB12), &s, &ALL_PACKS), first);
        }
    }

    #[test]
    fn full_gpu_runs_do_not_spin_cpu_threads() {
        let p = run(&knobs("auto"), &hy18(), &pc(GB24), &snap(false, vec![]), &ALL_PACKS);
        assert!(p.no_poll && !p.below_normal);
        assert_eq!((p.threads, p.plan.gpu_layers), (2, ALL_LAYERS));
    }
}
