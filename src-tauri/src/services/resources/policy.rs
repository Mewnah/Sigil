//! Deterministic model recommendations (research §3.1 VRAM-aware model choice, §2.1 Qwen3-TTS
//! 0.6B while the voice changer runs, §8 GPU contention accounting).
//!
//! # How a choice is made
//! 1. **Room on the primary GPU**: its dedicated VRAM, minus a reserve for Windows, the game and
//!    OBS ([`reserve_mb`]: 30%, at least 2 GB), minus the voice changer (its claims, or
//!    [`VC_ESTIMATE_MB`] while it runs on a GPU EP without having claimed; nothing while it runs on
//!    the CPU or its device is not known), minus every other Sigil claim except the topic's own
//!    subsystem (the recommended model replaces it). Integrated GPUs and cards under
//!    [`MIN_GPU_MB`] run models on the CPU, and so does a PC whose GPU probe got no answer.
//! 2. **A GPU build for the primary GPU** ([`gpu_runtime`]): a topic whose engine has only a
//!    CUDA 13 build (the STT final pass, until transcribe.cpp's `vulkan` variant is built and
//!    measured) runs on the CPU unless the primary GPU can use the CUDA EP.
//! 3. **Candidates in quality order** with their GPU footprint (weights + KV cache + runtime
//!    buffers at the helper's default context). The first that fits wins; nothing fitting means
//!    the CPU variant.
//! 4. **Explicit rules** from the research: Hy-MT2 7B on 8 GB cards only while the voice changer
//!    is off on the GPU ([`MID_GPU_MB`]: a rule, not arithmetic, so a small measured VC claim
//!    cannot bring the 7B back; it protects the voice changer's GPU time as well as its VRAM);
//!    Qwen3-TTS 0.6B while the voice changer runs on the GPU of a card under 16 GB (GPU time for
//!    the voice first); Qwen3-ASR BF16 only on 16 GB+ cards.
//!
//! The live DXGI budget and device-wide usage are deliberately not inputs: they move with every
//! game frame, and a recommendation must not flip while the user reads it. Claims and the VC flag
//! are stable, explicit inputs. Integer arithmetic only: the same inputs always give the same
//! answer (unit-tested as a table). Footprints and thresholds are provisional until the §8
//! campaign (P7) measures them; ids follow the catalog convention `<family>/<name>` and are the
//! ids the MT (P4-B), Qwen3-TTS (P6-E) and GPU STT (P2-D) catalog fragments must use.

use super::{Recommendation, RecommendationTopic, ResourceSnapshot, Subsystem};
use crate::services::sysinfo::{eps, GpuProbe, SystemInfo};

/// Catalog ids the policy recommends.
pub mod ids {
    pub const MT_MILMMT_12B: &str = "mt/milmmt-46-12b-q4_k_m";
    pub const MT_HY_MT2_30B_A3B: &str = "mt/hy-mt2-30b-a3b-q4_k_m";
    pub const MT_HY_MT2_7B: &str = "mt/hy-mt2-7b-q4_k_m";
    pub const MT_MILMMT_4B: &str = "mt/milmmt-46-4b-q4_k_m";
    pub const MT_HY_MT2_1_8B: &str = "mt/hy-mt2-1.8b-q4_k_m";
    /// The 1.25-bit build (440 MB) for low-memory CPUs.
    pub const MT_HY_MT2_1_8B_TINY: &str = "mt/hy-mt2-1.8b-1.25bit";

    pub const TTS_QWEN3_1_7B: &str = "tts/qwen3-tts-1.7b";
    pub const TTS_QWEN3_0_6B: &str = "tts/qwen3-tts-0.6b";

    pub const STT_QWEN3_ASR_1_7B_BF16: &str = "stt/qwen3-asr-1.7b-bf16";
    pub const STT_QWEN3_ASR_1_7B_Q8: &str = "stt/qwen3-asr-1.7b-q8_0";
    /// English / European finals only (never ja/zh/ko; the P2 router enforces that).
    pub const STT_COHERE_Q4K: &str = "stt/cohere-transcribe-q4_k";
    pub const STT_QWEN3_ASR_0_6B: &str = "stt/qwen3-asr-0.6b-int8";
}

/// Share of the primary GPU's VRAM kept free for Windows, the game (VRChat) and OBS.
pub const RESERVE_PERCENT: u64 = 30;
/// The reserve never drops below this.
pub const MIN_RESERVE_MB: u64 = 2048;
/// What a running voice changer is assumed to hold until it claims its real footprint (community
/// RVC engines use about 1–1.2 GB per model).
pub const VC_ESTIMATE_MB: u64 = 1200;
/// Smaller GPUs (and integrated ones) run the recommended models on the CPU.
pub const MIN_GPU_MB: u64 = 3072;
/// Below this ("8 GB class" cards; 10 GB and 12 GB cards are above), a voice changer on the GPU
/// rules out Hy-MT2 7B, however small its claim (research §3.1: 7B on 8 GB only while RVC is off).
pub const MID_GPU_MB: u64 = 10_000;
/// "16 GB class" cards (DXGI reports ~16,300 MB for 16 GB).
pub const LARGE_GPU_MB: u64 = 15_000;
/// Machines below this ("8 GB class") get the smallest CPU variants.
pub const LOW_RAM_MB: u64 = 12_000;

/// Why a recommendation was made (`Recommendation::reason_code`; the UI translates by code).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reason {
    /// The best default fits on the GPU.
    Fits,
    /// A smaller model because the voice changer runs on the GPU.
    VcRunning,
    /// A smaller model because other Sigil models hold VRAM.
    SharedGpu,
    /// A smaller model because the GPU is small.
    SmallGpu,
    /// No hardware GPU: CPU.
    NoGpu,
    /// Only an integrated GPU: CPU.
    IntegratedGpu,
    /// Nothing fits on the GPU right now: CPU.
    GpuFull,
    /// The smallest CPU variant because of little RAM or few cores.
    LowMemory,
    /// The engine has no GPU build for the primary GPU (e.g. CUDA-only on an AMD or Pascal card): CPU.
    NoGpuRuntime,
    /// The graphics driver did not answer the probe, so the GPU is unknown: CPU for now.
    GpuUnknown,
}

impl Reason {
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::Fits => "fits",
            Reason::VcRunning => "vc_running",
            Reason::SharedGpu => "shared_gpu",
            Reason::SmallGpu => "small_gpu",
            Reason::NoGpu => "no_gpu",
            Reason::IntegratedGpu => "integrated_gpu",
            Reason::GpuFull => "gpu_full",
            Reason::LowMemory => "low_memory",
            Reason::NoGpuRuntime => "no_gpu_runtime",
            Reason::GpuUnknown => "gpu_unknown",
        }
    }
}

/// A candidate model: GPU footprint (MiB, including KV cache and runtime buffers) and a label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Model {
    id: &'static str,
    vram_mb: u64,
    label: &'static str,
}

const MILMMT_12B: Model = Model {
    id: ids::MT_MILMMT_12B,
    vram_mb: 8100,
    label: "MiLMMT-46 12B",
};
const HY_MT2_30B_A3B: Model = Model {
    id: ids::MT_HY_MT2_30B_A3B,
    vram_mb: 19_600,
    label: "Hy-MT2 30B-A3B",
};
const HY_MT2_7B: Model = Model {
    id: ids::MT_HY_MT2_7B,
    vram_mb: 5300,
    label: "Hy-MT2 7B",
};
const MILMMT_4B: Model = Model {
    id: ids::MT_MILMMT_4B,
    vram_mb: 3000,
    label: "MiLMMT-46 4B",
};
const HY_MT2_1_8B: Model = Model {
    id: ids::MT_HY_MT2_1_8B,
    vram_mb: 1600,
    label: "Hy-MT2 1.8B",
};
const HY_MT2_1_8B_TINY: Model = Model {
    id: ids::MT_HY_MT2_1_8B_TINY,
    vram_mb: 700,
    label: "Hy-MT2 1.8B (1.25-bit)",
};

const QWEN3_TTS_1_7B: Model = Model {
    id: ids::TTS_QWEN3_1_7B,
    vram_mb: 3200,
    label: "Qwen3-TTS 1.7B",
};
const QWEN3_TTS_0_6B: Model = Model {
    id: ids::TTS_QWEN3_0_6B,
    vram_mb: 1500,
    label: "Qwen3-TTS 0.6B",
};

const ASR_BF16: Model = Model {
    id: ids::STT_QWEN3_ASR_1_7B_BF16,
    vram_mb: 4600,
    label: "Qwen3-ASR 1.7B (BF16)",
};
const ASR_Q8: Model = Model {
    id: ids::STT_QWEN3_ASR_1_7B_Q8,
    vram_mb: 2700,
    label: "Qwen3-ASR 1.7B (Q8_0)",
};
const ASR_COHERE: Model = Model {
    id: ids::STT_COHERE_Q4K,
    vram_mb: 1900,
    label: "Cohere Transcribe (Q4_K)",
};
const ASR_0_6B: Model = Model {
    id: ids::STT_QWEN3_ASR_0_6B,
    vram_mb: 0,
    label: "Qwen3-ASR 0.6B (int8)",
};

/// MT defaults in order (the research's picks; the others are listed as alternatives only).
const MT_DEFAULTS: [Model; 2] = [HY_MT2_7B, HY_MT2_1_8B];
/// Every GPU MT model, best first (for alternatives).
const MT_BY_QUALITY: [Model; 5] = [MILMMT_12B, HY_MT2_30B_A3B, HY_MT2_7B, MILMMT_4B, HY_MT2_1_8B];

/// How the primary GPU can be used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuClass {
    NoGpu,
    /// The GPU probe got no answer ([`GpuProbe::Missing`]): the GPUs are unknown.
    Unknown,
    Integrated,
    TooSmall,
    Usable,
}

/// Which GPU builds a topic's engine has.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuRuntime {
    /// Vulkan (any vendor) and CUDA 13 builds: llama-server (MT), qwentts.cpp (Qwen3-TTS).
    VulkanOrCuda,
    /// Only a CUDA 13 build so far: transcribe.cpp (its `vulkan` variant is unbuilt and unmeasured;
    /// P1 outcomes), so non-NVIDIA and pre-Turing GPUs use the CPU variant. Switch the topic to
    /// [`GpuRuntime::VulkanOrCuda`] once the vulkan pack ships.
    Cuda13Only,
}

/// The GPU builds `topic`'s engine has.
pub fn gpu_runtime(topic: RecommendationTopic) -> GpuRuntime {
    match topic {
        RecommendationTopic::SttFinal => GpuRuntime::Cuda13Only,
        RecommendationTopic::MtLlm | RecommendationTopic::Qwen3Tts => GpuRuntime::VulkanOrCuda,
    }
}

/// Whether `topic`'s GPU build can run on the primary GPU: a CUDA-only engine needs the CUDA EP to
/// be eligible (driver R580+) and the primary GPU itself to be an NVIDIA Turing or newer card.
pub fn gpu_runtime_available(topic: RecommendationTopic, sys: &SystemInfo) -> bool {
    match gpu_runtime(topic) {
        GpuRuntime::VulkanOrCuda => true,
        GpuRuntime::Cuda13Only => {
            sys.eps.cuda.eligible
                && sys
                    .primary()
                    .is_some_and(|g| eps::meets_cuda_floor(g) != Some(false))
        }
    }
}

/// VRAM available to a topic's model on the primary GPU (MiB).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    pub gpu: GpuClass,
    pub total_mb: u64,
    pub reserve_mb: u64,
    /// Held by (or assumed for) the voice changer.
    pub vc_mb: u64,
    /// Held by other Sigil subsystems (not the voice changer, not the topic's own subsystem).
    pub others_mb: u64,
    pub available_mb: u64,
    /// The voice changer runs and holds (or is assumed to hold) VRAM.
    pub vc_on_gpu: bool,
}

/// VRAM kept free for Windows, the game and OBS on a card with `total_mb`.
pub fn reserve_mb(total_mb: u64) -> u64 {
    (total_mb.saturating_mul(RESERVE_PERCENT) / 100).max(MIN_RESERVE_MB)
}

fn topic_subsystem(topic: RecommendationTopic) -> Subsystem {
    match topic {
        RecommendationTopic::MtLlm => Subsystem::Llm,
        RecommendationTopic::Qwen3Tts => Subsystem::Tts,
        RecommendationTopic::SttFinal => Subsystem::SttFinal,
    }
}

/// The room for `topic`'s model.
pub fn room(topic: RecommendationTopic, sys: &SystemInfo, snap: &ResourceSnapshot) -> Room {
    let own = topic_subsystem(topic);
    let primary = sys.primary();
    let total_mb = primary
        .map(|g| g.adapter.dedicated_vram_mb)
        .unwrap_or(snap.budget.total_mb);
    let gpu = match primary {
        None if sys.gpu_probe == GpuProbe::Missing => GpuClass::Unknown,
        None => GpuClass::NoGpu,
        Some(g) if g.adapter.software => GpuClass::NoGpu,
        Some(g) if g.integrated => GpuClass::Integrated,
        Some(_) if total_mb < MIN_GPU_MB => GpuClass::TooSmall,
        Some(_) => GpuClass::Usable,
    };
    let vc_claims: Option<u64> = snap
        .claims
        .iter()
        .filter(|c| c.subsystem == Subsystem::Vc)
        .map(|c| c.mb)
        .reduce(u64::saturating_add);
    // Real claims win (0 MB = on the CPU). Without one, only a voice changer known to run on a
    // GPU EP is assumed to hold its estimate: the CPU EP, DSP-only presets and a device not
    // reported yet hold nothing.
    let vc_mb = match (snap.vc_running, vc_claims) {
        (_, Some(mb)) => mb,
        (true, None) if snap.vc_gpu == Some(true) => VC_ESTIMATE_MB,
        _ => 0,
    };
    let others_mb = snap
        .claims
        .iter()
        .filter(|c| c.subsystem != Subsystem::Vc && c.subsystem != own)
        .fold(0u64, |acc, c| acc.saturating_add(c.mb));
    let reserve = reserve_mb(total_mb);
    let available_mb = if gpu == GpuClass::Usable {
        total_mb
            .saturating_sub(reserve)
            .saturating_sub(vc_mb)
            .saturating_sub(others_mb)
    } else {
        0
    };
    Room {
        gpu,
        total_mb,
        reserve_mb: reserve,
        vc_mb,
        others_mb,
        available_mb,
        vc_on_gpu: snap.vc_running && vc_mb > 0,
    }
}

/// `5300` → `"5.2 GB"` (integer math: deterministic formatting).
fn gb(mb: u64) -> String {
    let tenths = mb.saturating_mul(10).saturating_add(512) / 1024;
    format!("{}.{} GB", tenths / 10, tenths % 10)
}

fn fits(m: &Model, room: &Room) -> bool {
    room.gpu == GpuClass::Usable && m.vram_mb <= room.available_mb
}

/// Why the preferred model `best` did not fit while a smaller one did.
fn downgrade_reason(best: &Model, room: &Room) -> Reason {
    let need = best.vram_mb;
    if room.vc_on_gpu && need <= room.available_mb.saturating_add(room.vc_mb) {
        Reason::VcRunning
    } else if room.others_mb > 0
        && need
            <= room
                .available_mb
                .saturating_add(room.vc_mb)
                .saturating_add(room.others_mb)
    {
        Reason::SharedGpu
    } else {
        Reason::SmallGpu
    }
}

/// Why a CPU variant was chosen.
fn cpu_reason(room: &Room) -> Reason {
    match room.gpu {
        GpuClass::NoGpu => Reason::NoGpu,
        GpuClass::Unknown => Reason::GpuUnknown,
        GpuClass::Integrated => Reason::IntegratedGpu,
        GpuClass::TooSmall => Reason::SmallGpu,
        GpuClass::Usable => Reason::GpuFull,
    }
}

fn room_text(room: &Room) -> String {
    match room.gpu {
        GpuClass::NoGpu => "no graphics card was found".into(),
        GpuClass::Unknown => "the graphics card details could not be read".into(),
        GpuClass::Integrated => "the graphics card is integrated (it shares system memory)".into(),
        GpuClass::TooSmall => format!("the graphics card has only {}", gb(room.total_mb)),
        GpuClass::Usable => {
            let mut parts = vec![format!("{} of {} free for it", gb(room.available_mb), gb(room.total_mb))];
            parts.push(format!("{} kept for Windows and games", gb(room.reserve_mb)));
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

fn reason_text(chosen: &Model, on_gpu: bool, reason: Reason, room: &Room, preferred: Option<&Model>) -> String {
    let what = if on_gpu {
        format!("{} on the GPU ({})", chosen.label, gb(chosen.vram_mb))
    } else {
        format!("{} on the CPU", chosen.label)
    };
    let why = match (reason, preferred) {
        (Reason::Fits, _) => "it fits".to_string(),
        (Reason::VcRunning, Some(p)) => format!("{} would leave too little for the running voice changer", p.label),
        (Reason::VcRunning, None) => "the voice changer is running on the GPU".to_string(),
        (Reason::SharedGpu, Some(p)) => format!("{} does not fit next to the other models Sigil has loaded", p.label),
        (Reason::SmallGpu, Some(p)) => format!("{} needs {}", p.label, gb(p.vram_mb)),
        (Reason::GpuFull, _) => "no GPU version fits right now".to_string(),
        (Reason::LowMemory, _) => "this PC has little memory or few cores".to_string(),
        (Reason::NoGpuRuntime, _) => {
            "Sigil has no GPU build of it for this graphics card yet (it needs an NVIDIA GeForce RTX 20 / GTX 16 or newer)".to_string()
        }
        (Reason::GpuUnknown, _) => "the graphics driver did not answer, so the GPU could not be checked".to_string(),
        (_, _) => "the GPU cannot be used".to_string(),
    };
    format!("{what}: {why}; {}.", room_text(room))
}

fn dedup_push(list: &mut Vec<String>, id: &str, chosen: &str) {
    if id != chosen && !list.iter().any(|x| x == id) {
        list.push(id.to_string());
    }
}

#[allow(clippy::too_many_arguments)]
fn make(
    topic: RecommendationTopic,
    chosen: &Model,
    on_gpu: bool,
    reason: Reason,
    room: &Room,
    preferred: Option<&Model>,
    alternatives: Vec<String>,
) -> Recommendation {
    Recommendation {
        topic,
        choice: chosen.id.to_string(),
        reason: reason_text(chosen, on_gpu, reason, room, preferred),
        alternatives,
        reason_code: reason.as_str().to_string(),
        device: if on_gpu { "gpu" } else { "cpu" }.to_string(),
        vram_mb: if on_gpu { chosen.vram_mb } else { 0 },
        available_mb: room.available_mb,
    }
}

fn mt_llm(sys: &SystemInfo, snap: &ResourceSnapshot) -> Recommendation {
    let topic = RecommendationTopic::MtLlm;
    let room = room(topic, sys, snap);
    let low_memory = sys.total_ram_mb < LOW_RAM_MB || sys.cpu.physical_cores < 4;
    let cpu_model = if low_memory { HY_MT2_1_8B_TINY } else { HY_MT2_1_8B };
    // Research §3.1: on 8 GB cards Hy-MT2 7B only while the voice changer is off the GPU.
    let protect_vc = room.vc_on_gpu && room.total_mb < MID_GPU_MB;
    let defaults: &[Model] = if protect_vc { &MT_DEFAULTS[1..] } else { &MT_DEFAULTS };
    if let Some(chosen) = defaults.iter().find(|m| fits(m, &room)).copied() {
        let (reason, preferred) = if chosen == MT_DEFAULTS[0] {
            (Reason::Fits, None)
        } else if protect_vc {
            (Reason::VcRunning, Some(&MT_DEFAULTS[0]))
        } else {
            (downgrade_reason(&MT_DEFAULTS[0], &room), Some(&MT_DEFAULTS[0]))
        };
        let mut alternatives = Vec::new();
        for m in MT_BY_QUALITY.iter().filter(|m| fits(m, &room)) {
            dedup_push(&mut alternatives, m.id, chosen.id);
        }
        return make(topic, &chosen, true, reason, &room, preferred, alternatives);
    }
    let reason = if low_memory { Reason::LowMemory } else { cpu_reason(&room) };
    let other = if low_memory { HY_MT2_1_8B } else { HY_MT2_1_8B_TINY };
    make(topic, &cpu_model, false, reason, &room, None, vec![other.id.to_string()])
}

fn qwen3_tts(sys: &SystemInfo, snap: &ResourceSnapshot) -> Recommendation {
    let topic = RecommendationTopic::Qwen3Tts;
    let room = room(topic, sys, snap);
    let protect_vc = room.vc_on_gpu && room.total_mb < LARGE_GPU_MB;
    let candidates: &[Model] = if protect_vc {
        &[QWEN3_TTS_0_6B]
    } else {
        &[QWEN3_TTS_1_7B, QWEN3_TTS_0_6B]
    };
    if let Some(chosen) = candidates.iter().find(|m| fits(m, &room)).copied() {
        let (reason, preferred) = if chosen == QWEN3_TTS_1_7B {
            (Reason::Fits, None)
        } else if protect_vc {
            (Reason::VcRunning, None)
        } else {
            (downgrade_reason(&QWEN3_TTS_1_7B, &room), Some(&QWEN3_TTS_1_7B))
        };
        let mut alternatives = Vec::new();
        for m in [QWEN3_TTS_1_7B, QWEN3_TTS_0_6B]
            .iter()
            .filter(|m| fits(m, &room))
        {
            dedup_push(&mut alternatives, m.id, chosen.id);
        }
        return make(topic, &chosen, true, reason, &room, preferred, alternatives);
    }
    make(topic, &QWEN3_TTS_0_6B, false, cpu_reason(&room), &room, None, Vec::new())
}

fn stt_final(sys: &SystemInfo, snap: &ResourceSnapshot) -> Recommendation {
    let topic = RecommendationTopic::SttFinal;
    let room = room(topic, sys, snap);
    if room.gpu == GpuClass::Usable && !gpu_runtime_available(topic, sys) {
        return make(topic, &ASR_0_6B, false, Reason::NoGpuRuntime, &room, None, Vec::new());
    }
    let large = room.total_mb >= LARGE_GPU_MB;
    let candidates: &[Model] = if large { &[ASR_BF16, ASR_Q8] } else { &[ASR_Q8] };
    if let Some(i) = candidates.iter().position(|m| fits(m, &room)) {
        let chosen = candidates[i];
        let (reason, preferred) = if i == 0 {
            (Reason::Fits, None)
        } else {
            (downgrade_reason(&candidates[0], &room), Some(&candidates[0]))
        };
        let mut alternatives = Vec::new();
        for m in [ASR_BF16, ASR_Q8, ASR_COHERE]
            .iter()
            .filter(|m| fits(m, &room))
        {
            // BF16 is only offered on 16 GB+ cards.
            if m.id != ASR_BF16.id || large {
                dedup_push(&mut alternatives, m.id, chosen.id);
            }
        }
        dedup_push(&mut alternatives, ASR_0_6B.id, chosen.id);
        return make(topic, &chosen, true, reason, &room, preferred, alternatives);
    }
    make(topic, &ASR_0_6B, false, cpu_reason(&room), &room, None, Vec::new())
}

/// Pure, deterministic policy: the same inputs always give the same answer.
pub fn recommend(topic: RecommendationTopic, sys: &SystemInfo, snap: &ResourceSnapshot) -> Recommendation {
    match topic {
        RecommendationTopic::MtLlm => mt_llm(sys, snap),
        RecommendationTopic::Qwen3Tts => qwen3_tts(sys, snap),
        RecommendationTopic::SttFinal => stt_final(sys, snap),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::diagnostics::gpu_mem::{VENDOR_AMD, VENDOR_INTEL, VENDOR_MICROSOFT, VENDOR_NVIDIA};
    use crate::services::resources::{VramBudget, VramClaimInfo};
    use crate::services::sysinfo::test_support::{gpu, system};

    /// A PC whose primary GPU is an NVIDIA card with `vram_mb` (DXGI-style numbers: 8 GB = 8188).
    fn pc(vram_mb: u64) -> SystemInfo {
        system(vec![
            gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce RTX", vram_mb, Some("8.9"), Some("ada")),
            gpu(1, VENDOR_MICROSOFT, "Microsoft Basic Render Driver", 0, None, None),
        ])
    }

    fn cpu_only() -> SystemInfo {
        system(vec![gpu(0, VENDOR_MICROSOFT, "Microsoft Basic Render Driver", 0, None, None)])
    }

    fn laptop_igpu() -> SystemInfo {
        let mut g = gpu(0, VENDOR_INTEL, "Intel(R) UHD Graphics", 128, None, Some("xe-lp"));
        g.integrated = true;
        system(vec![g])
    }

    /// An AMD-only PC (Vulkan / DirectML, no CUDA).
    fn amd_pc(vram_mb: u64) -> SystemInfo {
        system(vec![gpu(0, VENDOR_AMD, "AMD Radeon RX 7900 XTX", vram_mb, None, Some("rdna3"))])
    }

    /// A PC whose only NVIDIA card is a Pascal GTX 1080 Ti (too old for CUDA 13).
    fn pascal_pc() -> SystemInfo {
        system(vec![gpu(
            0,
            VENDOR_NVIDIA,
            "NVIDIA GeForce GTX 1080 Ti",
            11_264,
            Some("6.1"),
            Some("pascal"),
        )])
    }

    /// A streamer rig: RTX 3070 (8 GB) plus a leftover GTX 1080 Ti (11 GB) for OBS / NVENC.
    fn rtx3070_plus_1080ti() -> SystemInfo {
        system(vec![
            gpu(0, VENDOR_NVIDIA, "NVIDIA GeForce GTX 1080 Ti", 11_264, Some("6.1"), Some("pascal")),
            gpu(1, VENDOR_NVIDIA, "NVIDIA GeForce RTX 3070", GB8, Some("8.6"), Some("ampere")),
        ])
    }

    /// An RTX 4090 on driver R566 (CUDA 12.7: too old for the CUDA 13 EP).
    fn old_driver_pc() -> SystemInfo {
        let mut sys = pc(GB24);
        sys.eps = eps::assess(&sys.gpus, Some("566.36"), Some("12.7"), true);
        sys
    }

    /// The graphics driver did not answer the probe: no GPU facts at all.
    fn gpu_unknown_pc() -> SystemInfo {
        let mut sys = system(vec![]);
        sys.gpu_probe = GpuProbe::Missing;
        sys.eps = eps::unknown();
        sys
    }

    fn claim(id: u64, subsystem: Subsystem, mb: u64) -> VramClaimInfo {
        VramClaimInfo {
            id,
            subsystem,
            label: format!("{subsystem:?}"),
            mb,
            external: subsystem != Subsystem::Vc,
        }
    }

    /// `vc_running` means a voice changer on a GPU EP (the case the research rules are about).
    fn snap(vc_running: bool, claims: Vec<VramClaimInfo>) -> ResourceSnapshot {
        ResourceSnapshot {
            budget: VramBudget::default(),
            claims,
            vc_running,
            vc_gpu: vc_running.then_some(true),
        }
    }

    /// A running voice changer whose device is `vc_gpu` (`None`: DSP-only or not reported yet).
    fn snap_vc(vc_gpu: Option<bool>, claims: Vec<VramClaimInfo>) -> ResourceSnapshot {
        ResourceSnapshot {
            budget: VramBudget::default(),
            claims,
            vc_running: true,
            vc_gpu,
        }
    }

    const GB8: u64 = 8188;
    const GB10: u64 = 10_240;
    const GB12: u64 = 12_282;
    const GB16: u64 = 16_376;
    const GB24: u64 = 24_142;
    const GB6: u64 = 6144;
    const GB4: u64 = 4096;

    struct Row {
        name: &'static str,
        sys: SystemInfo,
        snap: ResourceSnapshot,
        topic: RecommendationTopic,
        choice: &'static str,
        device: &'static str,
        code: &'static str,
    }

    fn row(
        name: &'static str,
        sys: SystemInfo,
        snap: ResourceSnapshot,
        topic: RecommendationTopic,
        choice: &'static str,
        device: &'static str,
        code: Reason,
    ) -> Row {
        Row {
            name,
            sys,
            snap,
            topic,
            choice,
            device,
            code: code.as_str(),
        }
    }

    #[test]
    fn policy_table() {
        use RecommendationTopic::{MtLlm, Qwen3Tts, SttFinal};
        let rows = vec![
            // --- Translation LLM (§3.1) ---
            row(
                "8 GB, VC off: Hy-MT2 7B",
                pc(GB8),
                snap(false, vec![]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "8 GB, VC on: 7B only if RVC is off → 1.8B",
                pc(GB8),
                snap(true, vec![]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "8 GB, VC on with a real 700 MB claim: still 1.8B",
                pc(GB8),
                snap(true, vec![claim(1, Subsystem::Vc, 700)]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "8 GB, VC on the GPU with a small measured claim (300 MB): the rule keeps 1.8B",
                pc(GB8),
                snap(true, vec![claim(1, Subsystem::Vc, 300)]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "8 GB, VC on the GPU with a 430 MB claim (CUDA context shared): still 1.8B",
                pc(GB8),
                snap(true, vec![claim(1, Subsystem::Vc, 430)]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "8 GB, VC on the CPU EP without a claim: 7B",
                pc(GB8),
                snap_vc(Some(false), vec![]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "8 GB, DSP-only VC (device unknown) without a claim: 7B",
                pc(GB8),
                snap_vc(None, vec![]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "8 GB, VC on the CPU with a 0 MB claim: 7B",
                pc(GB8),
                snap_vc(Some(false), vec![claim(1, Subsystem::Vc, 0)]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "10 GB, VC on: above the 8 GB rule, 7B fits",
                pc(GB10),
                snap(true, vec![]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "RTX 3070 + GTX 1080 Ti, VC on: sized for the 3070 (8 GB) → 1.8B",
                rtx3070_plus_1080ti(),
                snap(true, vec![]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "AMD 24 GB: llama-server has a Vulkan build → 7B on the GPU",
                amd_pc(GB24),
                snap(false, vec![]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "Pascal 11 GB: Vulkan build → 7B on the GPU",
                pascal_pc(),
                snap(false, vec![]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "GPU probe got no answer: CPU, not 'no GPU'",
                gpu_unknown_pc(),
                snap(false, vec![]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "cpu",
                Reason::GpuUnknown,
            ),
            row(
                "12 GB, VC on: 7B is the 12 GB default",
                pc(GB12),
                snap(true, vec![]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "12 GB, VC + GPU STT final: 1.8B",
                pc(GB12),
                snap(true, vec![claim(2, Subsystem::SttFinal, 2700)]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "12 GB, GPU STT final, VC off: 7B",
                pc(GB12),
                snap(false, vec![claim(2, Subsystem::SttFinal, 2700)]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "12 GB, its own loaded LLM does not count",
                pc(GB12),
                snap(true, vec![claim(3, Subsystem::Llm, 5300)]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "6 GB: 1.8B on the GPU",
                pc(GB6),
                snap(false, vec![]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "gpu",
                Reason::SmallGpu,
            ),
            row(
                "4 GB, VC off: 1.8B on the GPU",
                pc(GB4),
                snap(false, vec![]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "gpu",
                Reason::SmallGpu,
            ),
            row(
                "4 GB, VC on: CPU",
                pc(GB4),
                snap(true, vec![]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "cpu",
                Reason::GpuFull,
            ),
            row(
                "24 GB, everything running: 7B",
                pc(GB24),
                snap(true, vec![claim(4, Subsystem::SttFinal, 4600), claim(5, Subsystem::Tts, 3200)]),
                MtLlm,
                ids::MT_HY_MT2_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "no GPU: 1.8B Q4 on the CPU",
                cpu_only(),
                snap(false, vec![]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "cpu",
                Reason::NoGpu,
            ),
            row(
                "integrated GPU: CPU",
                laptop_igpu(),
                snap(false, vec![]),
                MtLlm,
                ids::MT_HY_MT2_1_8B,
                "cpu",
                Reason::IntegratedGpu,
            ),
            // --- Qwen3-TTS (§2.1: 0.6B while the VC runs) ---
            row(
                "8 GB, VC on: 0.6B",
                pc(GB8),
                snap(true, vec![]),
                Qwen3Tts,
                ids::TTS_QWEN3_0_6B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "8 GB, VC off: 1.7B",
                pc(GB8),
                snap(false, vec![]),
                Qwen3Tts,
                ids::TTS_QWEN3_1_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "12 GB, VC on: 0.6B",
                pc(GB12),
                snap(true, vec![]),
                Qwen3Tts,
                ids::TTS_QWEN3_0_6B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "8 GB, VC on the CPU EP without a claim: 1.7B",
                pc(GB8),
                snap_vc(Some(false), vec![]),
                Qwen3Tts,
                ids::TTS_QWEN3_1_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "8 GB, VC on the GPU (reported), no claim: 0.6B",
                pc(GB8),
                snap_vc(Some(true), vec![]),
                Qwen3Tts,
                ids::TTS_QWEN3_0_6B,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "VC running on the CPU (0 MB claim): 1.7B",
                pc(GB8),
                snap(true, vec![claim(6, Subsystem::Vc, 0)]),
                Qwen3Tts,
                ids::TTS_QWEN3_1_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "24 GB, VC on: 1.7B",
                pc(GB24),
                snap(true, vec![]),
                Qwen3Tts,
                ids::TTS_QWEN3_1_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "8 GB, VC off, 7B LLM loaded: 0.6B does not fit either → CPU",
                pc(GB8),
                snap(false, vec![claim(7, Subsystem::Llm, 5300)]),
                Qwen3Tts,
                ids::TTS_QWEN3_0_6B,
                "cpu",
                Reason::GpuFull,
            ),
            row(
                "12 GB, VC off, 7B LLM loaded: 1.7B still fits",
                pc(GB12),
                snap(false, vec![claim(7, Subsystem::Llm, 5300)]),
                Qwen3Tts,
                ids::TTS_QWEN3_1_7B,
                "gpu",
                Reason::Fits,
            ),
            row(
                "8 GB, VC off, GPU STT final: 0.6B",
                pc(GB8),
                snap(false, vec![claim(7, Subsystem::SttFinal, 2700)]),
                Qwen3Tts,
                ids::TTS_QWEN3_0_6B,
                "gpu",
                Reason::SharedGpu,
            ),
            row(
                "no GPU: 0.6B on the CPU",
                cpu_only(),
                snap(false, vec![]),
                Qwen3Tts,
                ids::TTS_QWEN3_0_6B,
                "cpu",
                Reason::NoGpu,
            ),
            // --- STT final pass (§1.1: Q8_0 on mid GPUs, BF16 on 16 GB+) ---
            row(
                "8 GB, VC on: Qwen3-ASR Q8_0",
                pc(GB8),
                snap(true, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_1_7B_Q8,
                "gpu",
                Reason::Fits,
            ),
            row(
                "16 GB: BF16",
                pc(GB16),
                snap(false, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_1_7B_BF16,
                "gpu",
                Reason::Fits,
            ),
            row(
                "16 GB, VC + 7B LLM: BF16 still fits",
                pc(GB16),
                snap(true, vec![claim(8, Subsystem::Llm, 5300)]),
                SttFinal,
                ids::STT_QWEN3_ASR_1_7B_BF16,
                "gpu",
                Reason::Fits,
            ),
            row(
                "16 GB, 7B LLM + Qwen3-TTS 1.7B: Q8_0",
                pc(GB16),
                snap(false, vec![claim(8, Subsystem::Llm, 5300), claim(10, Subsystem::Tts, 3200)]),
                SttFinal,
                ids::STT_QWEN3_ASR_1_7B_Q8,
                "gpu",
                Reason::SharedGpu,
            ),
            row(
                "16 GB, VC + 7B LLM + Qwen3-TTS 0.6B: Q8_0 for the VC",
                pc(GB16),
                snap(true, vec![claim(8, Subsystem::Llm, 5300), claim(10, Subsystem::Tts, 1500)]),
                SttFinal,
                ids::STT_QWEN3_ASR_1_7B_Q8,
                "gpu",
                Reason::VcRunning,
            ),
            row(
                "8 GB, VC off, 7B LLM: CPU",
                pc(GB8),
                snap(false, vec![claim(9, Subsystem::Llm, 5300)]),
                SttFinal,
                ids::STT_QWEN3_ASR_0_6B,
                "cpu",
                Reason::GpuFull,
            ),
            row(
                "no GPU: Qwen3-ASR 0.6B",
                cpu_only(),
                snap(false, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_0_6B,
                "cpu",
                Reason::NoGpu,
            ),
            row(
                "AMD 24 GB: no Vulkan build of transcribe.cpp yet → CPU",
                amd_pc(GB24),
                snap(false, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_0_6B,
                "cpu",
                Reason::NoGpuRuntime,
            ),
            row(
                "Pascal only (CUDA 13 dropped it) → CPU",
                pascal_pc(),
                snap(false, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_0_6B,
                "cpu",
                Reason::NoGpuRuntime,
            ),
            row(
                "4090 on a pre-R580 driver: CUDA not eligible → CPU",
                old_driver_pc(),
                snap(false, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_0_6B,
                "cpu",
                Reason::NoGpuRuntime,
            ),
            row(
                "RTX 3070 + GTX 1080 Ti: the 3070 runs CUDA → Q8_0",
                rtx3070_plus_1080ti(),
                snap(false, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_1_7B_Q8,
                "gpu",
                Reason::Fits,
            ),
            row(
                "GPU probe got no answer → CPU",
                gpu_unknown_pc(),
                snap(false, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_0_6B,
                "cpu",
                Reason::GpuUnknown,
            ),
            row(
                "2 GB card: CPU",
                pc(2048),
                snap(false, vec![]),
                SttFinal,
                ids::STT_QWEN3_ASR_0_6B,
                "cpu",
                Reason::SmallGpu,
            ),
        ];
        for r in rows {
            let rec = recommend(r.topic, &r.sys, &r.snap);
            assert_eq!(
                (rec.choice.as_str(), rec.device.as_str(), rec.reason_code.as_str()),
                (r.choice, r.device, r.code),
                "{}: {}",
                r.name,
                rec.reason
            );
            assert_eq!(rec.topic, r.topic);
            assert!(
                !rec.alternatives.contains(&rec.choice),
                "{}: choice repeated in {:?}",
                r.name,
                rec.alternatives
            );
            assert_eq!(rec.vram_mb == 0, rec.device == "cpu", "{}", r.name);
            assert!(rec.vram_mb <= rec.available_mb || rec.device == "cpu", "{}", r.name);
            assert!(rec.reason.ends_with('.') && !rec.reason.is_empty(), "{}", r.name);
        }
    }

    #[test]
    fn recommendations_are_deterministic() {
        let cases = [
            (
                pc(GB8),
                snap(true, vec![claim(1, Subsystem::Vc, 900), claim(2, Subsystem::SttFinal, 2700)]),
            ),
            (pc(GB24), snap(false, vec![])),
            (cpu_only(), snap(false, vec![])),
        ];
        for (sys, s) in &cases {
            for topic in [RecommendationTopic::MtLlm, RecommendationTopic::Qwen3Tts, RecommendationTopic::SttFinal] {
                let first = recommend(topic, sys, s);
                for _ in 0..50 {
                    assert_eq!(recommend(topic, sys, s), first);
                }
                // Claim order and ids do not matter, only their subsystems and sizes.
                let mut reordered = s.clone();
                reordered.claims.reverse();
                for (i, c) in reordered.claims.iter_mut().enumerate() {
                    c.id = 100 + i as u64;
                }
                assert_eq!(recommend(topic, sys, &reordered), first);
            }
        }
    }

    #[test]
    fn live_budget_numbers_do_not_change_the_answer() {
        let sys = pc(GB8);
        let quiet = snap(true, vec![]);
        let mut busy = quiet.clone();
        busy.budget = VramBudget {
            adapter_index: Some(0),
            adapter_name: Some("NVIDIA GeForce RTX".into()),
            total_mb: GB8,
            budget_mb: 3000,
            process_usage_mb: 2500,
            claimed_mb: 0,
            free_for_claims_mb: 500,
            ..VramBudget::default()
        };
        for topic in [RecommendationTopic::MtLlm, RecommendationTopic::Qwen3Tts, RecommendationTopic::SttFinal] {
            assert_eq!(recommend(topic, &sys, &quiet), recommend(topic, &sys, &busy));
        }
    }

    #[test]
    fn alternatives_are_best_first_and_fit() {
        let rec = recommend(RecommendationTopic::MtLlm, &pc(GB24), &snap(false, vec![]));
        assert_eq!(rec.choice, ids::MT_HY_MT2_7B);
        // 24 GB - 7.1 GB reserve = 16.9 GB: the 12B fits, the 30B-A3B does not.
        assert_eq!(rec.alternatives, vec![ids::MT_MILMMT_12B, ids::MT_MILMMT_4B, ids::MT_HY_MT2_1_8B]);

        let rec = recommend(RecommendationTopic::MtLlm, &cpu_only(), &snap(false, vec![]));
        assert_eq!(rec.alternatives, vec![ids::MT_HY_MT2_1_8B_TINY]);

        let rec = recommend(RecommendationTopic::SttFinal, &pc(GB8), &snap(false, vec![]));
        assert_eq!(rec.alternatives, vec![ids::STT_COHERE_Q4K, ids::STT_QWEN3_ASR_0_6B]);

        let rec = recommend(RecommendationTopic::Qwen3Tts, &pc(GB8), &snap(true, vec![]));
        assert_eq!(
            rec.alternatives,
            vec![ids::TTS_QWEN3_1_7B],
            "1.7B fits in memory; it is only not preferred"
        );
    }

    #[test]
    fn low_memory_cpus_get_the_smallest_translation_model() {
        let mut sys = cpu_only();
        sys.total_ram_mb = 8000;
        let rec = recommend(RecommendationTopic::MtLlm, &sys, &snap(false, vec![]));
        assert_eq!((rec.choice.as_str(), rec.reason_code.as_str()), (ids::MT_HY_MT2_1_8B_TINY, "low_memory"));
        assert_eq!(rec.alternatives, vec![ids::MT_HY_MT2_1_8B]);
        let mut sys = cpu_only();
        sys.cpu.physical_cores = 2;
        assert_eq!(
            recommend(RecommendationTopic::MtLlm, &sys, &snap(false, vec![])).choice,
            ids::MT_HY_MT2_1_8B_TINY
        );
    }

    #[test]
    fn room_accounts_for_every_input() {
        let r = room(
            RecommendationTopic::MtLlm,
            &pc(GB12),
            &snap(
                true,
                vec![
                    claim(1, Subsystem::SttFinal, 2700),
                    claim(2, Subsystem::Llm, 5300),
                    claim(3, Subsystem::Other, 100),
                ],
            ),
        );
        assert_eq!(r.gpu, GpuClass::Usable);
        assert_eq!(r.reserve_mb, 3684);
        assert_eq!(r.vc_mb, VC_ESTIMATE_MB);
        assert_eq!(r.others_mb, 2800, "the topic's own LLM claim is excluded");
        assert_eq!(r.available_mb, GB12 - 3684 - VC_ESTIMATE_MB - 2800);
        assert!(r.vc_on_gpu);
        assert_eq!(reserve_mb(4096), MIN_RESERVE_MB);
        assert_eq!(gb(5300), "5.2 GB");
        assert_eq!(gb(0), "0.0 GB");
        // Garbage claim sizes saturate instead of overflowing.
        let huge = snap(true, vec![claim(1, Subsystem::Other, u64::MAX), claim(2, Subsystem::Tts, u64::MAX)]);
        let rec = recommend(RecommendationTopic::SttFinal, &pc(GB12), &huge);
        assert_eq!((rec.device.as_str(), rec.reason_code.as_str()), ("cpu", "gpu_full"));
        assert!(reserve_mb(u64::MAX) > 0);
    }

    #[test]
    fn the_8gb_vc_rule_holds_for_every_vc_claim_size() {
        // Whatever the measured VC footprint (a shared CUDA context makes it small), a voice
        // changer on the GPU of an 8 GB card never gets Hy-MT2 7B next to it.
        for mb in [1, 100, 300, 432, 433, 700, 1200] {
            let rec = recommend(RecommendationTopic::MtLlm, &pc(GB8), &snap(true, vec![claim(1, Subsystem::Vc, mb)]));
            assert_eq!(
                (rec.choice.as_str(), rec.reason_code.as_str()),
                (ids::MT_HY_MT2_1_8B, "vc_running"),
                "{mb} MB"
            );
            assert!(rec.reason.contains("Hy-MT2 7B"), "{}", rec.reason);
        }
    }

    #[test]
    fn gpu_runtime_needs_cuda_on_the_primary_gpu_only_for_cuda_only_engines() {
        use RecommendationTopic::{MtLlm, Qwen3Tts, SttFinal};
        assert_eq!(gpu_runtime(SttFinal), GpuRuntime::Cuda13Only);
        for sys in [amd_pc(GB24), pascal_pc(), old_driver_pc(), gpu_unknown_pc()] {
            assert!(!gpu_runtime_available(SttFinal, &sys));
            assert!(gpu_runtime_available(MtLlm, &sys) && gpu_runtime_available(Qwen3Tts, &sys));
        }
        assert!(gpu_runtime_available(SttFinal, &pc(GB8)));
        assert!(gpu_runtime_available(SttFinal, &rtx3070_plus_1080ti()));
    }

    #[test]
    fn reason_text_explains_the_numbers() {
        let rec = recommend(RecommendationTopic::MtLlm, &pc(GB8), &snap(true, vec![]));
        assert!(rec.reason.contains("Hy-MT2 7B") && rec.reason.contains("voice changer"), "{}", rec.reason);
        assert!(rec.reason.contains("8.0 GB"), "{}", rec.reason);
    }
}
