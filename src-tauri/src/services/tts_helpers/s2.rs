//! `s2` helper TTS engine (P6-F, experimental): Fish Audio S2 Pro (5B Dual-AR, 80+ languages,
//! inline tags such as `[laugh]`, zero-shot cloning) through s2.cpp's HTTP server (community ggml
//! port, early stage; code and weights under the Fish Audio Research License). Fish S2 Pro is also
//! reachable through the `fishSpeech` backend with the official fish-speech server.
//!
//! - **Packs** `runtime/s2-server-{cuda13,vulkan,cpu}`: built by Sigil from a pinned s2.cpp commit
//!   (`scripts/runtime-packs/s2/build.ps1`; upstream publishes no Windows binaries), installed by
//!   Import or `SIGIL_LOCAL_ASSETS` (`unpublished`). The build writes `sigil-pack.json`: the
//!   program's size and sha256 (checked before every start) and the GPU switches it found in
//!   `s2.exe --help` (`server_args`, `low_vram_args`). The CUDA 13 build loads NVIDIA's runtime
//!   from `runtime/nvidia-cuda13`.
//! - **Models** `tts/s2-pro-q4_k_m` (default, about 7.2 GB of video memory with the codec and the
//!   KV cache), `tts/s2-pro-q6_k`, `tts/s2-pro-q8_0`, plus `tts/s2-pro-tokenizer` (Qwen3 BPE).
//! - **Server**: `--model --tokenizer --server --host 127.0.0.1 --port` plus the pack's switches.
//!   It has no health route (ready once it answers HTTP), no key, one synthesis at a time and a
//!   multipart `POST /generate`: Sigil speaks to it through the in-process [`ggml::adapter`],
//!   which adds the key, sends whole-clip `pcm_s16le` requests (its live chunks re-decode the
//!   prefix) and attaches a voice-library clip plus transcript for cloning. Without a reference
//!   S2 Pro picks a voice per sentence; quality degrades past about 37 s of audio, far above a
//!   caption sentence.
//!
//! The four functions below are the frozen engine interface of [`super`].

use serde_json::{json, Value};
use tauri::{AppHandle, Runtime};

use super::ggml::{self, launch, load_timeout_ms, AdapterKind, GgmlConfig, Launch, PlanEnv, Slot, Variant};
use super::{TtsHelperEndpoint, TtsHelperEngine, TtsHelperInfo, TtsHelperStatus};
use launch::{pick_pack, PackChoice, Readiness};

/// This module's engine.
pub const ENGINE: TtsHelperEngine = TtsHelperEngine::S2;
const TITLE: &str = "Fish S2 Pro";
pub const PROGRAM: &str = "s2.exe";
/// Engine name in the packs' `sigil-pack.json`.
pub const PACK_ENGINE: &str = "s2-server";
pub const RUNTIME_CUDA13: &str = "runtime/s2-server-cuda13";
pub const RUNTIME_VULKAN: &str = "runtime/s2-server-vulkan";
pub const RUNTIME_CPU: &str = "runtime/s2-server-cpu";
pub const TOKENIZER_ID: &str = "tts/s2-pro-tokenizer";
/// Sample rate when the server's answer names none (S2 Pro's codec).
pub const PCM_RATE: u32 = 44_100;
/// Video memory with `low_vram_args` (part of the model and the codec on the processor), MiB.
const LOW_VRAM_MB: u64 = 5_500;

const PACKS: [PackChoice; 3] = [
    PackChoice {
        variant: Variant::Cuda13,
        runtime: RUNTIME_CUDA13,
    },
    PackChoice {
        variant: Variant::Vulkan,
        runtime: RUNTIME_VULKAN,
    },
    PackChoice {
        variant: Variant::Cpu,
        runtime: RUNTIME_CPU,
    },
];

/// The models: (config value, catalog id). The first is the default.
pub const MODELS: [(&str, &str); 3] = [
    ("q4_k_m", "tts/s2-pro-q4_k_m"),
    ("q6_k", "tts/s2-pro-q6_k"),
    ("q8_0", "tts/s2-pro-q8_0"),
];

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
        .ok_or_else(|| format!("unknown Fish S2 Pro model {key:?} (q4_k_m, q6_k or q8_0)"))
}

/// Plan the launch for `config` (checks the Sigil-built pack's manifest and program hash).
pub fn plan(config: &GgmlConfig, env: &PlanEnv<'_>) -> Result<Launch, String> {
    let model = model_for(&config.model)?;
    let pack = pick_pack(TITLE, &PACKS, Variant::parse(&config.variant)?, env)?;
    let mut ids = vec![model, TOKENIZER_ID];
    ids.extend(launch::pack_ids(pack));
    env.require(TITLE, &ids)?;
    let gguf = env.file_of(model, None)?;
    let tokenizer = env.file_of(TOKENIZER_ID, None)?;
    let pack_dir = env.dir_of(pack.runtime)?;
    let (program, manifest) = launch::verify_sigil_pack(&pack_dir, PACK_ENGINE, pack.variant, PROGRAM)?;
    let dll_dirs = if pack.variant == Variant::Cuda13 { (env.nvidia_dirs)() } else { Vec::new() };
    if pack.variant == Variant::Cuda13 && dll_dirs.is_empty() {
        return Err(format!(
            "the CUDA 13 build needs the NVIDIA GPU runtime ({}): install it in Models & Runtimes, or choose Vulkan",
            launch::NVIDIA_RUNTIME_ID
        ));
    }
    let mut args: Vec<String> = vec![
        "--model".into(),
        gguf.display().to_string(),
        "--tokenizer".into(),
        tokenizer.display().to_string(),
        "--server".into(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        launch::port_arg(),
    ];
    args.extend(manifest.server_args.iter().cloned());
    let low_vram = config.low_vram && pack.variant.on_gpu() && !manifest.low_vram_args.is_empty();
    if low_vram {
        args.extend(manifest.low_vram_args.iter().cloned());
    }
    let vram_mb = match (pack.variant.on_gpu(), low_vram) {
        (false, _) => 0,
        (true, true) => LOW_VRAM_MB,
        (true, false) => env.footprint_mb(model).unwrap_or(7_400),
    };
    Ok(Launch {
        engine: ENGINE,
        helper_name: ENGINE.helper_name().to_string(),
        program,
        args,
        env: Vec::new(),
        env_remove: Vec::new(),
        server_key: false,
        readiness: Readiness::AnyHttp,
        adapter: Some(AdapterKind::S2),
        model: model.to_string(),
        variant: pack.variant,
        pcm_rate: PCM_RATE,
        vram_mb,
        load_timeout_ms: load_timeout_ms(env.bytes_of(&[model, TOKENIZER_ID])),
        dll_dirs,
        warmup: Some(json!({ "input": "Hello there.", "max_new_tokens": 96 })),
        summary: format!(
            "{model} on {} ({}){}",
            pack.variant.as_str(),
            manifest.version,
            if low_vram { ", low video memory" } else { "" }
        ),
    })
}

pub(crate) static SLOT: Slot = Slot::new(ENGINE, TITLE);

/// Static facts about the engine (`available`: a pack, the default model and the tokenizer are
/// installed).
pub fn info() -> TtsHelperInfo {
    let installed = |id: &str| crate::services::model_store::installed_path(id).is_some();
    TtsHelperInfo {
        engine: ENGINE,
        title: "Fish Audio S2 Pro (s2.cpp)".into(),
        experimental: true,
        gpu: true,
        cloning: true,
        requires: vec![RUNTIME_VULKAN.into(), MODELS[0].1.into(), TOKENIZER_ID.into()],
        available: PACKS.iter().any(|p| installed(p.runtime)) && installed(MODELS[0].1) && installed(TOKENIZER_ID),
    }
}

/// Start (or reuse) the engine's helper and return where to reach it.
pub async fn start<R: Runtime>(app: &AppHandle<R>, config: Value) -> Result<TtsHelperEndpoint, String> {
    let _ = app;
    let config: GgmlConfig = serde_json::from_value(config).map_err(|e| format!("invalid Fish S2 Pro settings: {e}"))?;
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
