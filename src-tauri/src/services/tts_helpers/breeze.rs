//! `breeze` helper TTS engine (P6-F, experimental): BreezeBlue Breeze TTS 2 (3B; English and
//! Mandarin) through Breeze-TTS-2.cpp's `breeze-server` (official v0.1.0 Windows Vulkan release,
//! Apache-2.0; the weights are BreezeBlue's research and non-commercial license).
//!
//! - **Pack** `runtime/breeze-server-vulkan`: the release zip, hash-pinned. Vulkan runs on NVIDIA,
//!   AMD and Intel GPUs; the processor variant starts the same program with `--cpu`.
//! - **Models** `tts/breeze-tts-2-q8_0` (recommended, about 4 GB of video memory) and
//!   `tts/breeze-tts-2-q4_k` (about 3 GB): one GGUF each, the vocoder inside.
//! - **Server**: `<model.gguf> --host 127.0.0.1 --port --ws-port -1 --voices-dir <app data>`: no
//!   web UI, no CORS, no WebSocket. It takes no key, so Sigil speaks to it through the in-process
//!   [`ggml::adapter`] (key-protected), which also turns requests into Breeze's native form: a
//!   voice designed or directed by `instructions`, a voice-library voice cloned from its clip and
//!   transcript (`ref_audio` + `ref_text`), `cfg_scale` (2-3 for inline events such as `(laugh)`),
//!   and a fixed `seed` for a steady voice. Streams 24 kHz PCM about every 2 s of audio.
//!
//! The four functions below are the frozen engine interface of [`super`].

use serde_json::{json, Value};
use tauri::{AppHandle, Runtime};

use super::ggml::{self, launch, load_timeout_ms, AdapterKind, GgmlConfig, Launch, PlanEnv, Slot, Variant};
use super::{TtsHelperEndpoint, TtsHelperEngine, TtsHelperInfo, TtsHelperStatus};
use launch::{pick_pack, PackChoice, Readiness};

/// This module's engine.
pub const ENGINE: TtsHelperEngine = TtsHelperEngine::Breeze;
const TITLE: &str = "Breeze TTS 2";
pub const PROGRAM: &str = "breeze-server.exe";
pub const RUNTIME_VULKAN: &str = "runtime/breeze-server-vulkan";
/// Sample rate of the server's PCM.
pub const PCM_RATE: u32 = 24_000;

const PACKS: [PackChoice; 2] = [
    PackChoice {
        variant: Variant::Vulkan,
        runtime: RUNTIME_VULKAN,
    },
    // The same pack, started with `--cpu`.
    PackChoice {
        variant: Variant::Cpu,
        runtime: RUNTIME_VULKAN,
    },
];

/// The models: (config value, catalog id). The first is the default.
pub const MODELS: [(&str, &str); 2] = [("q8_0", "tts/breeze-tts-2-q8_0"), ("q4_k", "tts/breeze-tts-2-q4_k")];

/// The catalog id of a config value (`""` = the default).
pub fn model_for(key: &str) -> Result<&'static str, String> {
    let k = key.trim().to_ascii_lowercase();
    if k.is_empty() || k == "auto" {
        return Ok(MODELS[0].1);
    }
    MODELS
        .iter()
        .find(|(name, id)| *name == k || *id == k)
        .map(|(_, id)| *id)
        .ok_or_else(|| format!("unknown Breeze TTS 2 model {key:?} (q8_0 or q4_k)"))
}

/// Plan the launch for `config`.
pub fn plan(config: &GgmlConfig, env: &PlanEnv<'_>) -> Result<Launch, String> {
    let model = model_for(&config.model)?;
    let pack = pick_pack(TITLE, &PACKS, Variant::parse(&config.variant)?, env)?;
    env.require(TITLE, &[model, pack.runtime])?;
    let gguf = env.file_of(model, None)?;
    let program = launch::pack_program(env, pack.runtime, PROGRAM)?;
    let voices = launch::helper_dir(env.app_data.as_deref(), ENGINE, "voices")?;
    let mut args: Vec<String> = vec![
        gguf.display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        launch::port_arg(),
        "--ws-port".into(),
        "-1".into(),
        "--voices-dir".into(),
        voices.display().to_string(),
    ];
    if pack.variant == Variant::Cpu {
        args.push("--cpu".into());
    }
    let vram_mb = if pack.variant.on_gpu() { env.footprint_mb(model).unwrap_or(4200) } else { 0 };
    Ok(Launch {
        engine: ENGINE,
        helper_name: ENGINE.helper_name().to_string(),
        program,
        args,
        env: Vec::new(),
        env_remove: Vec::new(),
        server_key: false,
        readiness: Readiness::Health("/health".into()),
        adapter: Some(AdapterKind::Breeze),
        model: model.to_string(),
        variant: pack.variant,
        pcm_rate: PCM_RATE,
        vram_mb,
        load_timeout_ms: load_timeout_ms(env.bytes_of(&[model])),
        dll_dirs: Vec::new(),
        warmup: Some(json!({ "input": "Hello there.", "instructions": "Speak clearly and naturally.", "seed": 42 })),
        summary: format!("{model} on {}", pack.variant.as_str()),
    })
}

pub(crate) static SLOT: Slot = Slot::new(ENGINE, TITLE);

/// Static facts about the engine (`available`: the pack and the default model are installed).
pub fn info() -> TtsHelperInfo {
    let installed = |id: &str| crate::services::model_store::installed_path(id).is_some();
    TtsHelperInfo {
        engine: ENGINE,
        title: "Breeze TTS 2 (Breeze-TTS-2.cpp)".into(),
        experimental: true,
        gpu: true,
        cloning: true,
        requires: vec![RUNTIME_VULKAN.into(), MODELS[0].1.into()],
        available: installed(RUNTIME_VULKAN) && installed(MODELS[0].1),
    }
}

/// Start (or reuse) the engine's helper and return where to reach it.
pub async fn start<R: Runtime>(app: &AppHandle<R>, config: Value) -> Result<TtsHelperEndpoint, String> {
    let _ = app;
    let config: GgmlConfig = serde_json::from_value(config).map_err(|e| format!("invalid Breeze TTS 2 settings: {e}"))?;
    SLOT.start(move || ggml::with_plan_env(|env| plan(&config, env))).await
}

/// Stop the helper (Ok when it is not running).
pub async fn stop() -> Result<(), String> {
    SLOT.stop(ENGINE.helper_name()).await
}

/// Current state.
pub fn status() -> TtsHelperStatus {
    SLOT.status()
}
