//! `magpie` helper TTS engine (P6-F, experimental): NVIDIA Magpie TTS Multilingual 357M through
//! NeMo-Speech.cpp's `nemo-speech serve` (official NVIDIA v0.2.0 Windows release, Apache-2.0).
//!
//! - **Packs** `runtime/nemo-speech-{cuda,vulkan,cpu}`: NVIDIA's release zips, hash-pinned
//!   (`model-catalog/tts-ggml-experimental.json`). The CUDA zip carries its own CUDA runtime (a
//!   cuBLAS shim, CUDA 13 build); the NVIDIA runtime pack's folders go on its PATH as a fallback.
//! - **Models**: the Magpie GGUF (`v2602`: en, es, de, fr, it, vi, hi; `v2607` adds ko, ar and
//!   pt-BR; both are NVIDIA's own GGUF files, used as they are), its tokenizer folder (members of
//!   the checkpoint's `.nemo` archive, extracted by `scripts/runtime-packs/nemo-speech/`; the
//!   server identifies the checkpoint from the GGUF and from the folder's `model_config.yaml` and
//!   refuses a mismatched pair, so each model has its own tokenizer), and the NanoCodec decoder.
//!   The v0.1.0 server could not read v2607 (frame stacking factor 2), hence [`MagpieModel::min_pack`].
//!   Mandarin and Japanese need a source build with `NEMO_SPEECH_TTS_WITH_ZH` / `_JA`; the official
//!   server profile has no frontend for them, so neither checkpoint lists them here (the TS
//!   backend refuses such text with a reason, as it does for any language Magpie cannot speak).
//! - **Server**: `serve --tts-model --codec-model --tokenizer-dir --host 127.0.0.1 --port --no-ui
//!   --device`, the key in `NEMO_SPEECH_HTTP_API_KEY` (every `NEMO_SPEECH_*` variable of the user's
//!   environment is removed first). `GET /health` answers 200 once the model is loaded. The voice
//!   and the language are never launch arguments: every request names both, so changing them never
//!   restarts the server (the server's own defaults, en-US and speaker 0 = John, are unused).
//! - **Speech**: OpenAI `POST /v1/audio/speech` (`input`, `voice` = one of [`SPEAKERS`],
//!   `language` = a [`language_code`], `response_format: pcm`): one buffered clip per sentence of
//!   22.05 kHz PCM; `speed` must be 1.
//!
//! The four functions below are the frozen engine interface of [`super`].

use std::path::Path;

use serde_json::{json, Value};
use tauri::{AppHandle, Runtime};

use super::ggml::launch::{self, pick_pack, PackChoice, PlanEnv, Readiness};
use super::ggml::{self, load_timeout_ms, GgmlConfig, Launch, Slot, Variant};
use super::{TtsHelperEndpoint, TtsHelperEngine, TtsHelperInfo, TtsHelperStatus};

/// This module's engine.
pub const ENGINE: TtsHelperEngine = TtsHelperEngine::Magpie;
const TITLE: &str = "Magpie TTS";
/// The NeMo-Speech.cpp program in the packs.
pub const PROGRAM: &str = "nemo-speech.exe";
pub const RUNTIME_CUDA: &str = "runtime/nemo-speech-cuda";
pub const RUNTIME_VULKAN: &str = "runtime/nemo-speech-vulkan";
pub const RUNTIME_CPU: &str = "runtime/nemo-speech-cpu";
/// The NanoCodec decoder both checkpoints use.
pub const CODEC_ID: &str = "tts/nemo-nano-codec-22khz";
/// Sample rate of the server's PCM (NanoCodec).
pub const PCM_RATE: u32 = 22_050;
/// The five built-in voices (speaker index order of the GGUF).
pub const SPEAKERS: [&str; 5] = ["John", "Sofia", "Aria", "Jason", "Leo"];
/// Environment variable the server reads its key from.
pub const API_KEY_ENV: &str = "NEMO_SPEECH_HTTP_API_KEY";

const PACKS: [PackChoice; 3] = [
    PackChoice {
        variant: Variant::Cuda,
        runtime: RUNTIME_CUDA,
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

/// A version of the NeMo-Speech.cpp server pack: (major, minor, patch).
pub type PackVersion = (u32, u32, u32);

/// One Magpie checkpoint.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MagpieModel {
    /// Config value.
    pub key: &'static str,
    pub model: &'static str,
    pub tokenizer: &'static str,
    /// Server language codes this checkpoint speaks with the official packs (TS mirror
    /// `magpieLanguages`; Mandarin and Japanese need a source build, so they are not listed).
    pub languages: &'static [&'static str],
    /// The oldest NeMo-Speech.cpp server that loads this checkpoint. Packs installed before the
    /// catalog moved to v0.2.0 still count as installed (the model store keeps an older install
    /// until it is updated), and v0.1.0 exits on v2607 with an unreadable message, so [`plan`]
    /// refuses that pairing with a message that names the update.
    pub min_pack: PackVersion,
}

/// The checkpoints (the first is the default: a saved profile that names none keeps v2602).
pub const MODELS: [MagpieModel; 2] = [
    MagpieModel {
        key: "v2602",
        model: "tts/magpie-tts-357m-v2602",
        tokenizer: "tts/magpie-tts-357m-v2602-tokenizer",
        languages: &["en-US", "es-ES", "de-DE", "fr-FR", "it-IT", "vi-VN", "hi-IN"],
        min_pack: (0, 1, 0),
    },
    MagpieModel {
        key: "v2607",
        model: "tts/magpie-tts-357m-v2607",
        tokenizer: "tts/magpie-tts-357m-v2607-tokenizer",
        languages: &["en-US", "es-ES", "de-DE", "fr-FR", "it-IT", "vi-VN", "hi-IN", "ko-KR", "ar-MSA", "pt-BR"],
        min_pack: (0, 2, 0),
    },
];

/// The version of an installed NeMo-Speech.cpp pack, read from its folder in the model store's
/// layout (`runtimes/nemo-speech-<variant>/<version>/<variant>`, so the folder named like the
/// runtime's id has the version folder below it). `None` for a folder outside that layout (a
/// development pack under `SIGIL_GGML_PACK_DIR`, a test folder) or a version that is not numeric:
/// only a positively identified old pack is refused.
pub fn pack_version(runtime: &str, dir: &Path) -> Option<PackVersion> {
    let id_rest = runtime.strip_prefix("runtime/").unwrap_or(runtime);
    dir.ancestors().find_map(|a| {
        let parent = a.parent()?.file_name()?.to_str()?;
        if !parent.eq_ignore_ascii_case(id_rest) {
            return None;
        }
        parse_version(a.file_name()?.to_str()?)
    })
}

/// `0.2.0` or `0.2.0-rc1` (digits up to the first non-digit of each part; at most three parts,
/// missing parts are 0), `None` when the first part is not a number.
fn parse_version(text: &str) -> Option<PackVersion> {
    let mut parts = text.trim().trim_start_matches(['v', 'V']).splitn(3, '.');
    let mut next = |first: bool| -> Option<u32> {
        let part = parts.next();
        let Some(part) = part else { return (!first).then_some(0) };
        let digits: String = part.chars().take_while(char::is_ascii_digit).collect();
        if digits.is_empty() {
            return (!first).then_some(0);
        }
        digits.parse().ok()
    };
    Some((next(true)?, next(false)?, next(false)?))
}

/// The checkpoint of a config value (`""` = the default).
pub fn model_for(key: &str) -> Result<MagpieModel, String> {
    let k = key.trim().to_ascii_lowercase();
    if k.is_empty() || k == "auto" {
        return Ok(MODELS[0]);
    }
    MODELS
        .iter()
        .copied()
        .find(|m| m.key == k || m.model == k)
        .ok_or_else(|| format!("unknown Magpie model {key:?} (v2602 or v2607)"))
}

/// The server language code of a BCP-47 tag (`ja` -> `ja-JP`, `pt` -> `pt-BR`, `ar` -> `ar-MSA`);
/// `None` for a language Magpie does not speak.
pub fn language_code(tag: &str) -> Option<&'static str> {
    let t = tag.trim().to_ascii_lowercase().replace('_', "-");
    match t.as_str() {
        "ar-ae" => return Some("ar-AE"),
        "ar-sa" => return Some("ar-SA"),
        _ => {}
    }
    Some(match ggml::language_base(&t)?.as_str() {
        "en" => "en-US",
        "es" => "es-ES",
        "de" => "de-DE",
        "fr" => "fr-FR",
        "it" => "it-IT",
        "vi" => "vi-VN",
        "hi" => "hi-IN",
        "zh" | "cmn" => "zh-CN",
        "ja" => "ja-JP",
        "ko" => "ko-KR",
        "ar" => "ar-MSA",
        "pt" => "pt-BR",
        _ => return None,
    })
}

/// The speaker name of a config value (`""` = John; case-insensitive), or an error.
pub fn speaker_for(voice: &str) -> Result<&'static str, String> {
    let v = voice.trim();
    if v.is_empty() || v.eq_ignore_ascii_case("default") {
        return Ok(SPEAKERS[0]);
    }
    SPEAKERS
        .iter()
        .copied()
        .find(|s| s.eq_ignore_ascii_case(v))
        .ok_or_else(|| format!("unknown Magpie voice {voice:?} ({})", SPEAKERS.join(", ")))
}

fn show_version(v: PackVersion) -> String {
    format!("{}.{}.{}", v.0, v.1, v.2)
}

/// Refuse a checkpoint that the installed pack is too old to load (see [`MagpieModel::min_pack`]).
fn check_pack_version(model: &MagpieModel, runtime: &str, dir: &Path) -> Result<(), String> {
    match pack_version(runtime, dir) {
        Some(found) if found < model.min_pack => Err(format!(
            "{TITLE} {} needs the NeMo-Speech.cpp server {} or newer, but {runtime} {} is installed: update it in Models & Runtimes",
            model.key,
            show_version(model.min_pack),
            show_version(found)
        )),
        _ => Ok(()),
    }
}

/// Plan the launch for `config` (blocking only through `env`).
pub fn plan(config: &GgmlConfig, env: &PlanEnv<'_>) -> Result<Launch, String> {
    let model = model_for(&config.model)?;
    let speaker = speaker_for(&config.voice)?;
    let pack = pick_pack(TITLE, &PACKS, Variant::parse(&config.variant)?, env)?;
    let mut ids = vec![model.model, model.tokenizer, CODEC_ID];
    ids.extend(launch::pack_ids(pack));
    env.require(TITLE, &ids)?;
    check_pack_version(&model, pack.runtime, &env.dir_of(pack.runtime)?)?;
    let gguf = env.file_of(model.model, None)?;
    let codec = env.file_of(CODEC_ID, None)?;
    let tokenizer_dir = env.dir_of(model.tokenizer)?;
    // The tokenizer folder is the extracted `.nemo` member set; model_config.yaml names the rest.
    let config_yaml = tokenizer_dir.join("model_config.yaml");
    if !config_yaml.is_file() {
        return Err(format!("{} is missing: install {} again", config_yaml.display(), model.tokenizer));
    }
    let program = launch::pack_program(env, pack.runtime, PROGRAM)?;
    // Only the warm-up uses them: every request names its voice and language, so they are not
    // launch arguments (a change must not restart the server; `Launch::same_server` compares the
    // arguments).
    let language = language_code(&config.language)
        .filter(|c| model.languages.contains(c))
        .unwrap_or("en-US");
    let device = match pack.variant {
        Variant::Cpu => "cpu",
        Variant::Vulkan => "vulkan",
        Variant::Cuda | Variant::Cuda13 => "cuda:0",
    };
    let dll_dirs = if pack.variant.nvidia() { (env.nvidia_dirs)() } else { Vec::new() };
    let vram_mb = if pack.variant.on_gpu() { env.footprint_mb(model.model).unwrap_or(1500) } else { 0 };
    let args: Vec<String> = [
        "serve",
        "--tts-model",
        &gguf.display().to_string(),
        "--codec-model",
        &codec.display().to_string(),
        "--tokenizer-dir",
        &tokenizer_dir.display().to_string(),
        "--host",
        "127.0.0.1",
        "--port",
        &launch::port_arg(),
        "--no-ui",
        "--device",
        device,
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    Ok(Launch {
        engine: ENGINE,
        helper_name: ENGINE.helper_name().to_string(),
        program,
        args,
        env: vec![(API_KEY_ENV.to_string(), launch::key_env())],
        env_remove: vec!["NEMO_SPEECH_*".to_string()],
        server_key: true,
        readiness: Readiness::Health("/health".into()),
        adapter: None,
        model: model.model.to_string(),
        variant: pack.variant,
        pcm_rate: PCM_RATE,
        vram_mb,
        load_timeout_ms: load_timeout_ms(env.bytes_of(&[model.model, model.tokenizer, CODEC_ID])),
        dll_dirs,
        warmup: Some(warmup_body(speaker, language)),
        summary: format!("{} ({}) on {}, {language}, voice {speaker}", model.model, model.key, pack.variant.as_str()),
    })
}

fn warmup_body(speaker: &str, language: &str) -> Value {
    json!({ "input": warmup_text(language), "voice": speaker, "language": language, "response_format": "pcm" })
}

/// The warm-up sentence: English, except in the two scripts of v2607 an English sentence is not
/// written in (Korean and Arabic), where the warm-up would otherwise tokenize Latin letters with
/// a tokenizer that has none.
fn warmup_text(language: &str) -> &'static str {
    match language {
        "ko-KR" => "안녕하세요.",
        "ar-MSA" | "ar-AE" | "ar-SA" => "مرحبا.",
        _ => "Hello there.",
    }
}

pub(crate) static SLOT: Slot = Slot::new(ENGINE, TITLE);

/// Static facts about the engine (`available`: a pack, the default checkpoint, its tokenizer and
/// the codec are installed).
pub fn info() -> TtsHelperInfo {
    let installed = |id: &str| crate::services::model_store::installed_path(id).is_some();
    let model = MODELS[0];
    let available = PACKS.iter().any(|p| installed(p.runtime)) && [model.model, model.tokenizer, CODEC_ID].iter().all(|id| installed(id));
    TtsHelperInfo {
        engine: ENGINE,
        title: "NVIDIA Magpie TTS Multilingual (NeMo-Speech.cpp)".into(),
        experimental: true,
        gpu: true,
        cloning: false,
        requires: vec![RUNTIME_VULKAN.into(), model.model.into(), model.tokenizer.into(), CODEC_ID.into()],
        available,
    }
}

/// Start (or reuse) the engine's helper and return where to reach it.
pub async fn start<R: Runtime>(app: &AppHandle<R>, config: Value) -> Result<TtsHelperEndpoint, String> {
    let _ = app;
    let config: GgmlConfig = serde_json::from_value(config).map_err(|e| format!("invalid Magpie settings: {e}"))?;
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
