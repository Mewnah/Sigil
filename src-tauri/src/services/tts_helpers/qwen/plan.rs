//! What the Qwen3-TTS helper runs: model size, talker checkpoint, runtime pack, VRAM estimate and
//! what is missing. Pure and deterministic (the same inputs always give the same plan), unit-tested
//! as a table like `resources::policy` and `llm_server::plan`.
//!
//! # Rules
//! 1. **Checkpoint by mode** (qwentts.cpp loads one talker per server): `preset` runs a CustomVoice
//!    talker (9 named speakers; style instructions on 1.7B only), `clone` a Base talker (voice
//!    cloning from a library clip), `design` the 1.7B VoiceDesign talker (a voice described in
//!    words; there is no 0.6B VoiceDesign). Every talker loads with the shared 12 Hz tokenizer.
//! 2. **Size**: `auto` follows the resource governor (`resources::recommend(Qwen3Tts)`: 1.7B when it
//!    fits, 0.6B while the voice changer runs on the GPU of a card under 16 GB, 0.6B on the CPU):
//!    the recommended size when its files are installed, else the first installed alternative the
//!    governor also accepts, else the recommended size (to download). `design` is always 1.7B. An
//!    explicit size is the user's choice and is never overridden.
//! 3. **Runtime pack**: `auto` runs on the CPU pack when the governor found no usable GPU, else on
//!    CUDA 13 when the primary GPU can use it and the cuda13 pack and the NVIDIA runtime are
//!    installed, else on an installed Vulkan pack (NVIDIA, AMD, Intel), else on CUDA 13 when half of
//!    it is installed (its pack or the NVIDIA runtime), else on Vulkan (to install). An explicit
//!    `cuda13` on a GPU that cannot run CUDA 13 is refused (it can never work there).
//! 4. **VRAM**: the claim is the catalog footprint of the talker (`min_vram_mb`, kept equal to the
//!    governor's per-size estimate) on a GPU pack, 0 on the CPU pack; the server resizes it once it
//!    reports its buffers.

use super::{MODEL_0_6B, MODEL_1_7B};
use crate::services::llm_server::plan::cuda_usable;
use crate::services::resources::Recommendation;
use crate::services::sysinfo::SystemInfo;

/// The shared 12 Hz tokenizer (codec) every talker loads with.
pub const TOKENIZER_ID: &str = "tts/qwen3-tts-tokenizer";
/// Talker checkpoints besides the two size ids of `resources::policy::ids` (the CustomVoice ones).
pub const MODEL_1_7B_BASE: &str = "tts/qwen3-tts-1.7b-base";
pub const MODEL_0_6B_BASE: &str = "tts/qwen3-tts-0.6b-base";
pub const MODEL_1_7B_DESIGN: &str = "tts/qwen3-tts-1.7b-voicedesign";
/// Every talker id (catalog fragment `tts-qwen3.json`).
pub const TALKER_IDS: [&str; 5] = [MODEL_1_7B, MODEL_1_7B_BASE, MODEL_1_7B_DESIGN, MODEL_0_6B, MODEL_0_6B_BASE];
/// Runtime packs (catalog fragment `runtime-qwentts.json`, Sigil-built, local import).
pub const RUNTIME_VULKAN: &str = "runtime/qwentts-server-vulkan";
pub const RUNTIME_CUDA13: &str = "runtime/qwentts-server-cuda13";
pub const RUNTIME_CPU: &str = "runtime/qwentts-server-cpu";
/// The NVIDIA CUDA 13 runtime the cuda13 pack loads (cudart, cuBLAS).
pub const NVIDIA_RUNTIME_ID: &str = crate::services::native_runtime::nvidia::NVIDIA_PACK_ID;
/// The server program inside a runtime pack.
pub const SERVER_EXE: &str = "sigil-tts-server.exe";

/// Footprints used when the catalog has none (MiB; the governor's per-size estimates).
const FALLBACK_VRAM_1_7B: u64 = 3200;
const FALLBACK_VRAM_0_6B: u64 = 1500;

/// Talker size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Size {
    /// 1.7B: instructions, voice design, the quality pick.
    Large,
    /// 0.6B: for the GPU next to the voice changer, or the CPU.
    Small,
}

impl Size {
    pub fn as_str(self) -> &'static str {
        match self {
            Size::Large => "1.7b",
            Size::Small => "0.6b",
        }
    }

    /// The governor's id for this size (`resources::policy::ids`).
    pub fn policy_id(self) -> &'static str {
        match self {
            Size::Large => MODEL_1_7B,
            Size::Small => MODEL_0_6B,
        }
    }

    pub fn from_policy_id(id: &str) -> Option<Size> {
        match id {
            MODEL_1_7B => Some(Size::Large),
            MODEL_0_6B => Some(Size::Small),
            _ => None,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Size::Large => "Qwen3-TTS 1.7B",
            Size::Small => "Qwen3-TTS 0.6B",
        }
    }
}

/// What the voice is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Mode {
    /// One of the 9 built-in voices (CustomVoice), optionally with a style instruction (1.7B).
    Preset,
    /// A voice-library voice (Base).
    Clone,
    /// A voice described in words (VoiceDesign, 1.7B only).
    Design,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Preset => "preset",
            Mode::Clone => "clone",
            Mode::Design => "design",
        }
    }

    /// qwentts.cpp's `qt_model_type` of the talker this mode loads.
    pub fn model_type(self) -> &'static str {
        match self {
            Mode::Preset => "custom_voice",
            Mode::Clone => "base",
            Mode::Design => "voice_design",
        }
    }
}

/// Runtime pack variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Variant {
    Vulkan,
    Cuda13,
    Cpu,
}

impl Variant {
    pub fn as_str(self) -> &'static str {
        match self {
            Variant::Vulkan => "vulkan",
            Variant::Cuda13 => "cuda13",
            Variant::Cpu => "cpu",
        }
    }

    pub fn runtime_id(self) -> &'static str {
        match self {
            Variant::Vulkan => RUNTIME_VULKAN,
            Variant::Cuda13 => RUNTIME_CUDA13,
            Variant::Cpu => RUNTIME_CPU,
        }
    }

    pub fn on_gpu(self) -> bool {
        self != Variant::Cpu
    }
}

/// The talker checkpoint of `size` in `mode` (`None`: there is no 0.6B VoiceDesign).
pub fn talker_id(size: Size, mode: Mode) -> Option<&'static str> {
    match (size, mode) {
        (Size::Large, Mode::Preset) => Some(MODEL_1_7B),
        (Size::Large, Mode::Clone) => Some(MODEL_1_7B_BASE),
        (Size::Large, Mode::Design) => Some(MODEL_1_7B_DESIGN),
        (Size::Small, Mode::Preset) => Some(MODEL_0_6B),
        (Size::Small, Mode::Clone) => Some(MODEL_0_6B_BASE),
        (Size::Small, Mode::Design) => None,
    }
}

/// Size and mode of a talker id.
pub fn parse_talker_id(id: &str) -> Option<(Size, Mode)> {
    [Size::Large, Size::Small]
        .into_iter()
        .flat_map(|s| [Mode::Preset, Mode::Clone, Mode::Design].map(|m| (s, m)))
        .find(|&(s, m)| talker_id(s, m) == Some(id))
}

/// The engine config from TypeScript (`plugin:tts-helpers|tts_helper_start { engine: "qwen3", config }`),
/// the launch-relevant part of the `qwen3` settings slice. Unknown members are ignored.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct QwenConfig {
    /// `auto` (the governor), `1.7b` or `0.6b`.
    pub model: String,
    /// `auto`, `vulkan`, `cuda13` or `cpu`.
    pub variant: String,
    /// `preset`, `clone` or `design`.
    pub mode: String,
    /// The configured language: a qwentts language name, a BCP-47 tag, or `auto`. Only the warm-up
    /// request uses it (every speech request names its own); it never decides the server.
    pub language: String,
}

/// Normalized knobs of a [`QwenConfig`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Knobs {
    /// `None` = auto.
    pub size: Option<Size>,
    /// `None` = auto.
    pub variant: Option<Variant>,
    pub mode: Mode,
    /// A qwentts language name or `auto`.
    pub language: &'static str,
}

/// Check a config.
pub fn normalize(config: &QwenConfig) -> Result<Knobs, String> {
    let size = match config.model.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => None,
        "1.7b" | "large" => Some(Size::Large),
        "0.6b" | "small" => Some(Size::Small),
        other => return Err(format!("unknown Qwen3-TTS model size {other:?} (auto, 1.7b or 0.6b)")),
    };
    let variant = match config.variant.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => None,
        "vulkan" => Some(Variant::Vulkan),
        "cuda13" | "cuda" => Some(Variant::Cuda13),
        "cpu" => Some(Variant::Cpu),
        other => return Err(format!("unknown runtime variant {other:?} (auto, vulkan, cuda13 or cpu)")),
    };
    let mode = match config.mode.trim().to_ascii_lowercase().as_str() {
        "" | "preset" => Mode::Preset,
        "clone" => Mode::Clone,
        "design" => Mode::Design,
        other => return Err(format!("unknown Qwen3-TTS mode {other:?} (preset, clone or design)")),
    };
    Ok(Knobs {
        size,
        variant,
        mode,
        language: language_name(&config.language).unwrap_or("auto"),
    })
}

/// The languages Qwen3-TTS speaks: (qwentts.cpp name, BCP-47 base tag).
pub const LANGUAGES: [(&str, &str); 10] = [
    ("chinese", "zh"),
    ("english", "en"),
    ("japanese", "ja"),
    ("korean", "ko"),
    ("german", "de"),
    ("french", "fr"),
    ("russian", "ru"),
    ("portuguese", "pt"),
    ("spanish", "es"),
    ("italian", "it"),
];

/// The qwentts.cpp language name of a BCP-47 tag (`ja`, `pt-BR`, `zh-Hant`) or of a name it already
/// is (`Japanese`); `Some("auto")` for empty / `auto`; `None` for a language the model lacks.
pub fn language_name(tag: &str) -> Option<&'static str> {
    let t = tag.trim().to_ascii_lowercase().replace('_', "-");
    if t.is_empty() || t == "auto" {
        return Some("auto");
    }
    let base = t.split('-').next().unwrap_or("");
    // Cantonese and other Chinese tags share the Mandarin model voice.
    let base = match base {
        "cmn" | "yue" | "zho" => "zh",
        "jpn" => "ja",
        "eng" => "en",
        b => b,
    };
    LANGUAGES
        .iter()
        .find(|(name, code)| *name == t || *code == base)
        .map(|(name, _)| *name)
}

/// Inputs from the running system.
pub struct Env<'a> {
    pub sys: &'a SystemInfo,
    /// `resources::recommend(Qwen3Tts, ..)` now.
    pub recommendation: &'a Recommendation,
    /// Whether a catalog id is installed.
    pub installed: &'a dyn Fn(&str) -> bool,
    /// Catalog `min_vram_mb` of a talker id.
    pub footprint_mb: &'a dyn Fn(&str) -> Option<u64>,
}

/// The decision.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    pub size: Size,
    pub mode: Mode,
    pub variant: Variant,
    /// Talker catalog id.
    pub talker: &'static str,
    pub runtime: &'static str,
    /// VRAM claimed while it runs (0 on the CPU pack).
    pub vram_mb: u64,
    /// Catalog ids to install first (talker, tokenizer, runtime pack, NVIDIA runtime), in that order.
    pub missing: Vec<String>,
    /// Why it cannot run at all (never with an install).
    pub refused: Option<String>,
    /// Plain-English summary for the log.
    pub reason: String,
}

/// Plan `knobs` on this system.
pub fn plan(knobs: &Knobs, env: &Env) -> Plan {
    let rec = env.recommendation;
    let rec_size = Size::from_policy_id(&rec.choice).unwrap_or(Size::Small);
    let rec_cpu = rec.device == "cpu";
    let installed = |id: &str| (env.installed)(id);
    let ready = |size: Size| talker_id(size, knobs.mode).is_some_and(&installed) && installed(TOKENIZER_ID);

    let (size, size_why) = match (knobs.mode, knobs.size) {
        (Mode::Design, _) => (Size::Large, "voice design needs the 1.7B model".to_string()),
        (_, Some(s)) => (s, format!("{} chosen in the settings", s.as_str())),
        (_, None) => {
            let chosen = std::iter::once(rec.choice.as_str())
                .chain(rec.alternatives.iter().map(String::as_str))
                .filter_map(Size::from_policy_id)
                .find(|&s| ready(s));
            match chosen {
                Some(s) if s == rec_size => (s, format!("resource governor: {}", rec.reason)),
                Some(s) => (s, format!("{} is installed; the governor prefers {} ({})", s.label(), rec_size.label(), rec.reason)),
                None => (rec_size, format!("resource governor: {}", rec.reason)),
            }
        }
    };
    let talker = talker_id(size, knobs.mode).unwrap_or(MODEL_1_7B_DESIGN);

    let cuda_ok = cuda_usable(env.sys);
    let mut refused = None;
    let variant = match knobs.variant {
        Some(Variant::Cuda13) => {
            if !cuda_ok {
                refused = Some(
                    "the CUDA 13 build needs an NVIDIA GeForce GTX 16 / RTX 20 or newer with driver 580 or newer; choose Vulkan or Automatic".to_string(),
                );
            }
            Variant::Cuda13
        }
        Some(v) => v,
        None if rec_cpu => Variant::Cpu,
        None if cuda_ok && installed(RUNTIME_CUDA13) && installed(NVIDIA_RUNTIME_ID) => Variant::Cuda13,
        None if installed(RUNTIME_VULKAN) => Variant::Vulkan,
        // Half of the CUDA build is there already (its pack, or the NVIDIA runtime the voice
        // changer also uses): finish that one rather than ask for the Vulkan pack.
        None if cuda_ok && (installed(RUNTIME_CUDA13) || installed(NVIDIA_RUNTIME_ID)) => Variant::Cuda13,
        None => Variant::Vulkan,
    };

    let mut missing = Vec::new();
    if refused.is_none() {
        for id in [talker, TOKENIZER_ID, variant.runtime_id()] {
            if !installed(id) {
                missing.push(id.to_string());
            }
        }
        if variant == Variant::Cuda13 && !installed(NVIDIA_RUNTIME_ID) {
            missing.push(NVIDIA_RUNTIME_ID.to_string());
        }
    }
    let vram_mb = if variant.on_gpu() {
        (env.footprint_mb)(talker).unwrap_or(match size {
            Size::Large => FALLBACK_VRAM_1_7B,
            Size::Small => FALLBACK_VRAM_0_6B,
        })
    } else {
        0
    };
    let reason = format!(
        "{} ({}, {}) on {}: {size_why}",
        size.label(),
        knobs.mode.as_str(),
        talker,
        variant.as_str()
    );
    Plan {
        size,
        mode: knobs.mode,
        variant,
        talker,
        runtime: variant.runtime_id(),
        vram_mb,
        missing,
        refused,
        reason,
    }
}

/// Health budget for loading `bytes` of GGUF: a minute plus 30 s per GB, at most 10 minutes (the
/// first Vulkan run also builds its pipelines).
pub fn load_timeout_ms(bytes: u64) -> u64 {
    let gb = bytes.div_ceil(1 << 30);
    60_000u64.saturating_add(gb.saturating_mul(30_000)).min(600_000)
}
