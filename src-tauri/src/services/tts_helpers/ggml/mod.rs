//! Shared runner of the experimental ggml TTS helpers (P6-F): [`super::magpie`], [`super::breeze`],
//! [`super::omnivoice`] and [`super::s2`]. Everything in `tts_helpers/ggml/` is P6-F's; it is
//! declared in `tts_helpers/mod.rs` (`pub mod ggml;`, moved there from `magpie.rs` by P6-I0).
//!
//! # The four engines (time-boxed spikes; every one is an experimental preset)
//! | Engine | Runtime (helper) | Packs | Speech route | Key |
//! |---|---|---|---|---|
//! | `magpie` | NVIDIA NeMo-Speech.cpp `nemo-speech serve` | official v0.1.0 Windows zips (cpu, vulkan, cuda), pinned | OpenAI `/v1/audio/speech`, whole clip, 22.05 kHz PCM | `NEMO_SPEECH_HTTP_API_KEY` |
//! | `breeze` | Breeze-TTS-2.cpp `breeze-server` | official v0.1.0 Windows Vulkan zip (CPU with `--cpu`), pinned | native form `/v1/audio/speech` through Sigil's [`adapter`], streamed 24 kHz PCM | the adapter's (upstream has none) |
//! | `omnivoice` | CrispASR `crispasr --server --backend omnivoice` | official v0.8.40 Windows zips (cpu, vulkan, cuda13 without the CUDA runtime), pinned | OpenAI `/v1/audio/speech`, `stream: true`, 24 kHz PCM | `CRISPASR_API_KEYS` |
//! | `s2` | s2.cpp `s2 --server` (Fish Audio S2 Pro) | Sigil-built (`scripts/runtime-packs/s2/build.ps1`), unpublished | multipart `/generate` through the [`adapter`], 44.1 kHz PCM | the adapter's (upstream has none) |
//!
//! Deviations from the workstream entry, by decision (quality first, then local, then free):
//! - OmniVoice runs on CrispASR instead of omnivoice.cpp: official, hash-pinned Windows builds
//!   (CUDA 13 that uses Sigil's NVIDIA runtime pack, Vulkan, CPU) instead of a pack only a maintainer
//!   can build, an API key, a health route that waits for the model, and its own measurements put
//!   it about 2x faster than omnivoice.cpp on the same GGUF. CrispASR marks every clip with an
//!   inaudible spread-spectrum watermark and speaks an AI notice before *cloned* voices unless the
//!   user attests they disclose AI speech themselves (the panel's two consent switches).
//! - Magpie does not share a runtime with Nemotron STT: Sigil's Nemotron runs on sherpa-onnx (P2),
//!   so NeMo-Speech.cpp only serves TTS here. The official server profile has no Japanese or
//!   Mandarin frontend (`NEMO_SPEECH_TTS_WITH_JA` / `_ZH` are source-build options); Korean, Arabic
//!   and Brazilian Portuguese need the v2607 checkpoint, which NVIDIA publishes only as a `.nemo`
//!   (converted locally: `scripts/runtime-packs/nemo-speech/`).
//! - `tools/tts-shim/` is not used: a separate shim process would leave the upstream port as open as
//!   before, so the request translation for servers without Sigil's request shape lives in the
//!   in-process [`adapter`] (key-protected, loopback only, one request at a time).
//! - Fish S2 Pro stays reachable through the `fishSpeech` backend (the official fish-speech server).
//!
//! # How a start runs ([`runner`])
//! 1. The engine module plans a [`Launch`] from the TS config, the catalog and the installed
//!    packs and models (blocking: file checks), or names the downloads it needs (the UI asks for
//!    consent with `<ModelRequirement>` first; nothing is downloaded here).
//! 2. The runner reuses a running server with the same launch, else replaces it: VRAM claim
//!    (`resources::claim_external_vram`), `process_host::spawn` (127.0.0.1, auto port, random key,
//!    kill-on-close Job Object, restart on failure), readiness (the server's health route, or for
//!    s2 any HTTP answer), the [`adapter`] for engines that need one, a timed warm-up sentence.
//! 3. A watcher follows the helper (crash, restart, port change) and emits `tts_helpers:state`.
//!
//! Residual risk (documented in the threat model request): the Breeze and s2 upstream servers take
//! no key and check neither `Origin` nor `Host` (the adapter's checks cover only the adapter's own
//! port), so while they run their loopback port is reachable by
//! - any program on this PC, and
//! - web pages in the user's browser: a cross-origin `multipart/form-data` POST is a CORS "simple"
//!   request, sent without a preflight. CORS being off only hides the answer from the page; the
//!   request still runs. A page that finds the port (random per start, but scannable) can make the
//!   engine synthesize (tying up the GPU that TTS and the voice changer share), make s2.cpp read an
//!   `.s2voice` path it names, or use Breeze's voice routes, which write into Sigil's
//!   `tts-helpers/breeze/voices` folder.
//!
//! What limits it today: both bind 127.0.0.1 only (process_host's exposure audit stops a helper
//! that binds anything else), WebSocket and CORS stay off, the port is random per start, the
//! servers run only while the engine is in use (a preview's helper stops after two idle minutes),
//! and recent Chromium browsers ask before a public site may reach loopback addresses (Local
//! Network Access; other browsers may not). The fix is a Sigil-patched keyed server for both, as
//! `qwen` does for qwentts.cpp (planned for P7).

pub mod adapter;
pub mod launch;
pub mod runner;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};

use crate::services::sysinfo::SystemInfo;

pub use launch::{AdapterKind, Launch, PlanEnv};
pub use runner::Slot;

/// Runtime pack variant of a helper.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Variant {
    /// NVIDIA through CUDA 13 (the runtime DLLs come from `runtime/nvidia-cuda13`).
    Cuda13,
    /// NVIDIA through a CUDA build that carries its own runtime (NeMo-Speech.cpp's zip).
    Cuda,
    /// Any GPU through Vulkan.
    Vulkan,
    /// The processor.
    Cpu,
}

impl Variant {
    pub const ALL: [Variant; 4] = [Variant::Cuda13, Variant::Cuda, Variant::Vulkan, Variant::Cpu];

    pub fn as_str(self) -> &'static str {
        match self {
            Variant::Cuda13 => "cuda13",
            Variant::Cuda => "cuda",
            Variant::Vulkan => "vulkan",
            Variant::Cpu => "cpu",
        }
    }

    pub fn on_gpu(self) -> bool {
        self != Variant::Cpu
    }

    pub fn nvidia(self) -> bool {
        matches!(self, Variant::Cuda13 | Variant::Cuda)
    }

    /// Parse a config value; `None` = automatic.
    pub fn parse(value: &str) -> Result<Option<Variant>, String> {
        match value.trim().to_ascii_lowercase().as_str() {
            "" | "auto" => Ok(None),
            "cuda13" => Ok(Some(Variant::Cuda13)),
            "cuda" => Ok(Some(Variant::Cuda)),
            "vulkan" => Ok(Some(Variant::Vulkan)),
            "cpu" => Ok(Some(Variant::Cpu)),
            other => Err(format!(
                "unknown runtime variant {other:?} (auto, {})",
                Variant::ALL.map(Variant::as_str).join(", ")
            )),
        }
    }
}

/// The launch part of the `ggmlTts` settings (TS `GgmlHelperConfig`;
/// `plugin:tts-helpers|tts_helper_start { engine, config }`). Unknown members are ignored; every
/// engine reads the members it understands.
#[derive(Clone, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", default)]
pub struct GgmlConfig {
    /// The engine's model choice (`""` = its default; see each engine's `MODELS`).
    pub model: String,
    /// `auto`, `cuda13`, `cuda`, `vulkan` or `cpu` (each engine accepts the ones it has packs for).
    pub variant: String,
    /// Language when a request names none (BCP-47).
    pub language: String,
    /// Default voice when a request names none (Magpie speaker).
    pub voice: String,
    /// Voice-library voice the helper should be able to speak with (OmniVoice: copied into the
    /// server's voice folder at start; Breeze and s2 read it per request through the adapter).
    pub reference_voice_id: String,
    /// `high` (default) or `fast`: OmniVoice's refinement steps (32 / 16).
    pub quality: String,
    /// s2: keep part of the model and the codec on the processor (cards under 8 GB).
    pub low_vram: bool,
}

/// GPU facts a plan needs (from the system probe).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GpuFacts {
    /// A hardware GPU (not WARP) is present.
    pub any_gpu: bool,
    /// The primary GPU runs CUDA 13 (NVIDIA Turing or newer, driver 580 or newer).
    pub cuda: bool,
}

impl GpuFacts {
    pub fn of(sys: &SystemInfo) -> GpuFacts {
        GpuFacts {
            any_gpu: sys.primary().is_some_and(|g| !g.adapter.software),
            cuda: crate::services::llm_server::plan::cuda_usable(sys),
        }
    }
}

/// Run `f` with this PC's planning environment (blocking: catalog, system probe, installed files).
pub fn with_plan_env<T>(f: impl FnOnce(&PlanEnv<'_>) -> Result<T, String>) -> Result<T, String> {
    let catalog = crate::services::model_store::Catalog::load()?;
    let sys = crate::services::sysinfo::cached();
    let installed = |id: &str| crate::services::model_store::installed_path(id);
    let env = PlanEnv {
        catalog: &catalog,
        installed: &installed,
        gpu: GpuFacts::of(&sys),
        nvidia_dirs: &launch::nvidia_dll_dirs,
        app_data: crate::services::diagnostics::app_data_dir(),
    };
    f(&env)
}

/// Load timeout for `bytes` of model files (60 s plus 30 s per GiB, at most 10 min).
pub fn load_timeout_ms(bytes: u64) -> u64 {
    let gb = bytes.div_ceil(1 << 30);
    60_000u64.saturating_add(gb.saturating_mul(30_000)).min(600_000)
}

/// The BCP-47 base (`pt-BR` -> `pt`, `zh_Hant` -> `zh`), lower case; `None` for empty or `auto`.
pub fn language_base(tag: &str) -> Option<String> {
    let t = tag.trim().to_ascii_lowercase().replace('_', "-");
    if t.is_empty() || t == "auto" {
        return None;
    }
    let base = t.split('-').next().unwrap_or("").to_string();
    (!base.is_empty() && base.chars().all(|c| c.is_ascii_alphabetic())).then_some(base)
}
