//! `omnivoice` helper TTS engine (P6-F, experimental): k2-fsa OmniVoice (Qwen3-0.6B backbone,
//! masked iterative 8-codebook TTS, 600+ languages, zero-shot cloning) through CrispASR's server
//! (official v0.8.40 Windows release, MIT). Why CrispASR rather than omnivoice.cpp: see
//! [`ggml`]'s module docs.
//!
//! - **Packs** `runtime/crispasr-{cuda13,vulkan,cpu}`: the release zips, hash-pinned. The CUDA 13
//!   zip is the build without NVIDIA's runtime DLLs: they come from `runtime/nvidia-cuda13` (put on
//!   the helper's PATH), the same pack the voice changer uses.
//! - **Models** `tts/omnivoice-q8_0` (default) or `tts/omnivoice-f16`, plus the HiggsAudio v2
//!   codec `tts/omnivoice-tokenizer-f16` (CrispASR's GGUF conversions, cstr/omnivoice-GGUF).
//! - **Server**: `--server --backend omnivoice -m <lm> --codec-model <codec> --host 127.0.0.1
//!   --port --voice-dir <app data> --tts-steps 32|16`, the key in `CRISPASR_API_KEYS`
//!   (the user's `CRISPASR_*` / `OMNIVOICE_*` variables are removed first). `GET /health` answers
//!   503 while loading, then 200.
//! - **Speech**: OpenAI `POST /v1/audio/speech` with `stream: true` and `response_format: pcm`
//!   (24 kHz, one chunk per sentence: OmniVoice is not autoregressive), `language` always set
//!   (the research's advice), a fixed `seed` for a steady automatic voice.
//! - **Cloning**: the chosen voice-library voice is copied into the server's voice folder as
//!   `<id>.wav` + `<id>.txt` at every start, and requests name it by id. CrispASR requires a
//!   consent attestation for cloned voices and prepends a spoken AI notice unless the request
//!   attests that the user discloses AI speech themselves (both are switches in the panel, off by
//!   default; the TS backend sends the attestations only when they are on).
//!
//! The four functions below are the frozen engine interface of [`super`].

use std::path::{Path, PathBuf};

use serde_json::{json, Value};
use tauri::{AppHandle, Runtime};

use super::ggml::{self, launch, load_timeout_ms, GgmlConfig, Launch, PlanEnv, Slot, Variant};
use super::{TtsHelperEndpoint, TtsHelperEngine, TtsHelperInfo, TtsHelperStatus};
use crate::services::voice_library;
use launch::{pick_pack, PackChoice, Readiness};

/// This module's engine.
pub const ENGINE: TtsHelperEngine = TtsHelperEngine::Omnivoice;
const TITLE: &str = "OmniVoice";
pub const PROGRAM: &str = "crispasr.exe";
pub const RUNTIME_CUDA13: &str = "runtime/crispasr-cuda13";
pub const RUNTIME_VULKAN: &str = "runtime/crispasr-vulkan";
pub const RUNTIME_CPU: &str = "runtime/crispasr-cpu";
/// The HiggsAudio v2 codec.
pub const CODEC_ID: &str = "tts/omnivoice-tokenizer-f16";
/// Sample rate of the server's PCM.
pub const PCM_RATE: u32 = 24_000;
/// Environment variable the server reads its keys from (comma-separated; Sigil sets one).
pub const API_KEY_ENV: &str = "CRISPASR_API_KEYS";

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
pub const MODELS: [(&str, &str); 2] = [("q8_0", "tts/omnivoice-q8_0"), ("f16", "tts/omnivoice-f16")];

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
        .ok_or_else(|| format!("unknown OmniVoice model {key:?} (q8_0 or f16)"))
}

/// Refinement steps of a quality setting (`high` = OmniVoice's default 32, `fast` = 16).
pub fn steps_for(quality: &str) -> Result<u32, String> {
    match quality.trim().to_ascii_lowercase().as_str() {
        "" | "high" => Ok(32),
        "fast" => Ok(16),
        other => Err(format!("unknown OmniVoice quality {other:?} (high or fast)")),
    }
}

/// A voice id CrispASR accepts as a voice-folder name (`[A-Za-z0-9_-]+`).
pub fn voice_file_stem(id: &str) -> Option<&str> {
    let id = id.trim();
    (!id.is_empty() && id.len() <= 96 && id.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')).then_some(id)
}

/// Copy library voice `id` into `dir` as `<id>.wav` + `<id>.txt` (blocking; rewritten only when
/// the clip or the transcript changed).
pub fn materialize_voice(app_data: &Path, dir: &Path, id: &str) -> Result<PathBuf, String> {
    let stem = voice_file_stem(id).ok_or_else(|| format!("the voice id {id:?} cannot be used as a file name"))?;
    let clip = voice_library::reference_clip(app_data, id)?;
    let transcript = clip.transcript.as_deref().map(str::trim).unwrap_or("");
    if transcript.is_empty() {
        return Err(format!("the library voice {:?} has no transcript yet: wait for it, or type it in the voice library", clip.name));
    }
    let wav = dir.join(format!("{stem}.wav"));
    let txt = dir.join(format!("{stem}.txt"));
    let wav_same = launch::sha256_cached(&wav).is_ok_and(|h| h.eq_ignore_ascii_case(&clip.sha256));
    if !wav_same {
        write_atomically(&wav, &std::fs::read(&clip.path).map_err(|e| format!("cannot read {}: {e}", clip.path.display()))?)?;
    }
    if std::fs::read_to_string(&txt).ok().as_deref() != Some(transcript) {
        write_atomically(&txt, transcript.as_bytes())?;
    }
    Ok(wav)
}

fn write_atomically(path: &Path, bytes: &[u8]) -> Result<(), String> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, bytes).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        let _ = std::fs::remove_file(&tmp);
        format!("cannot write {}: {e}", path.display())
    })
}

/// Plan the launch for `config` (copies the chosen library voice into the voice folder).
pub fn plan(config: &GgmlConfig, env: &PlanEnv<'_>) -> Result<Launch, String> {
    let model = model_for(&config.model)?;
    let steps = steps_for(&config.quality)?;
    let pack = pick_pack(TITLE, &PACKS, Variant::parse(&config.variant)?, env)?;
    let mut ids = vec![model, CODEC_ID];
    ids.extend(launch::pack_ids(pack));
    env.require(TITLE, &ids)?;
    let lm = env.file_of(model, None)?;
    let codec = env.file_of(CODEC_ID, None)?;
    let program = launch::pack_program(env, pack.runtime, PROGRAM)?;
    let voices = launch::helper_dir(env.app_data.as_deref(), ENGINE, "voices")?;
    let reference = config.reference_voice_id.trim();
    if !reference.is_empty() {
        let app_data = env.app_data.as_deref().ok_or("Sigil's app-data folder is unknown")?;
        materialize_voice(app_data, &voices, reference)?;
    }
    let dll_dirs = if pack.variant == Variant::Cuda13 { (env.nvidia_dirs)() } else { Vec::new() };
    if pack.variant == Variant::Cuda13 && dll_dirs.is_empty() {
        return Err(format!(
            "the CUDA 13 build needs the NVIDIA GPU runtime ({}): install it in Models & Runtimes, or choose Vulkan",
            launch::NVIDIA_RUNTIME_ID
        ));
    }
    let language = ggml::language_base(&config.language);
    // The language goes with every request (never a launch argument: changing it must not reload).
    let args: Vec<String> = vec![
        "--server".into(),
        "--backend".into(),
        "omnivoice".into(),
        "-m".into(),
        lm.display().to_string(),
        "--codec-model".into(),
        codec.display().to_string(),
        "--host".into(),
        "127.0.0.1".into(),
        "--port".into(),
        launch::port_arg(),
        "--voice-dir".into(),
        voices.display().to_string(),
        "--tts-steps".into(),
        steps.to_string(),
    ];
    let vram_mb = if pack.variant.on_gpu() { env.footprint_mb(model).unwrap_or(2200) } else { 0 };
    Ok(Launch {
        engine: ENGINE,
        helper_name: ENGINE.helper_name().to_string(),
        program,
        args,
        env: vec![(API_KEY_ENV.to_string(), launch::key_env())],
        env_remove: vec!["CRISPASR_*".to_string(), "OMNIVOICE_*".to_string()],
        server_key: true,
        readiness: Readiness::Health("/health".into()),
        adapter: None,
        model: model.to_string(),
        variant: pack.variant,
        pcm_rate: PCM_RATE,
        vram_mb,
        load_timeout_ms: load_timeout_ms(env.bytes_of(&[model, CODEC_ID])),
        dll_dirs,
        warmup: Some(json!({
            "input": "Hello there.",
            "language": language.as_deref().unwrap_or("en"),
            "response_format": "pcm",
            "stream": true,
            "seed": 7,
        })),
        summary: format!("{model} on {}, {steps} steps", pack.variant.as_str()),
    })
}

pub(crate) static SLOT: Slot = Slot::new(ENGINE, TITLE);

/// Static facts about the engine (`available`: a pack, the default model and the codec are
/// installed).
pub fn info() -> TtsHelperInfo {
    let installed = |id: &str| crate::services::model_store::installed_path(id).is_some();
    TtsHelperInfo {
        engine: ENGINE,
        title: "OmniVoice (CrispASR)".into(),
        experimental: true,
        gpu: true,
        cloning: true,
        requires: vec![RUNTIME_VULKAN.into(), MODELS[0].1.into(), CODEC_ID.into()],
        available: PACKS.iter().any(|p| installed(p.runtime)) && installed(MODELS[0].1) && installed(CODEC_ID),
    }
}

/// Start (or reuse) the engine's helper and return where to reach it.
pub async fn start<R: Runtime>(app: &AppHandle<R>, config: Value) -> Result<TtsHelperEndpoint, String> {
    let _ = app;
    let config: GgmlConfig = serde_json::from_value(config).map_err(|e| format!("invalid OmniVoice settings: {e}"))?;
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
