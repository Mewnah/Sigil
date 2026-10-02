//! `qwen3` helper TTS engine: Qwen3-TTS 0.6B / 1.7B through qwentts.cpp (P6-E). Everything in this
//! file and in `tts_helpers/qwen/` is P6-E's.
//!
//! # How it runs
//! - **Server**: Sigil's build of qwentts.cpp's OpenAI-compatible server, `sigil-tts-server.exe`
//!   (`scripts/runtime-packs/qwentts/`: the pinned upstream plus `server/sigil-tts-server.cpp`).
//!   Packs `runtime/qwentts-server-{vulkan,cuda13,cpu}` (`model-catalog/runtime-qwentts.json`,
//!   Sigil-built, `unpublished`: Import or `SIGIL_LOCAL_ASSETS`); the cuda13 pack loads cuBLAS from
//!   `runtime/nvidia-cuda13` (put on the helper's PATH).
//! - **Models** (`model-catalog/tts-qwen3.json`, Q8_0 GGUF from Serveurperso/Qwen3-TTS-GGUF): one
//!   talker per mode plus the shared tokenizer `tts/qwen3-tts-tokenizer`. Preset voices run the
//!   CustomVoice talkers `tts/qwen3-tts-1.7b` / `tts/qwen3-tts-0.6b` (the ids of
//!   `resources::policy::ids`), cloning the Base talkers `tts/qwen3-tts-{1.7b,0.6b}-base`, voice
//!   design `tts/qwen3-tts-1.7b-voicedesign` (1.7B only). See [`plan`].
//! - **Size** follows the resource governor (`resources::recommend(Qwen3Tts)`) for `model: auto`:
//!   1.7B when it fits, 0.6B while the voice changer runs on the GPU of a card under 16 GB, 0.6B on
//!   the CPU. TypeScript asks again (and Rust re-plans) when the recommendation changes.
//! - **Process**: `process_host` helper `tts-qwen3` (127.0.0.1, auto port, a random key passed in
//!   `QWENTTS_API_KEY` and required by the server on every route but `/health`, kill-on-close Job
//!   Object, restart on failure), holding `resources::claim_external_vram(Subsystem::Tts, ..)` sized
//!   to the talker's catalog footprint, then to the buffers the server logs. See [`server`].
//! - **Speech**: TypeScript speaks through the OpenAI-compatible client
//!   (`plugin:openai-speech|openai_speech_speak`) with the endpoint [`start`] returns: raw 24 kHz
//!   PCM streamed frame by frame into the TTS sink, `language` / `instructions` as extra request
//!   members.
//! - **Clone registry**: [`qwen_tts_prepare_clone`] registers a voice-library voice with the running
//!   server; its speaker data is cached per voice in `voices/<id>/cache/qwen3-tts.bin`
//!   ([`CACHE_BACKEND`]). See [`clone`].
//!
//! The four engine functions are the frozen interface of [`super`]; [`qwen_tts_prepare_clone`] is
//! frozen too (`plugin:tts-helpers|qwen_tts_prepare_clone { voiceId } -> QwenCloneVoice`).

pub mod clone;
pub mod plan;
pub mod server;
#[cfg(test)]
mod tests;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{command, AppHandle, Runtime};

use super::{TtsHelperEndpoint, TtsHelperEngine, TtsHelperInfo, TtsHelperStatus};
use crate::services::model_store;

/// This module's engine.
pub const ENGINE: TtsHelperEngine = TtsHelperEngine::Qwen3;
/// Voice-library cache backend name of the clone registry (`voices/<id>/cache/qwen3-tts.bin`).
pub const CACHE_BACKEND: &str = "qwen3-tts";
/// Catalog ids of the models (the governor's size ids; the CustomVoice talkers).
pub const MODEL_1_7B: &str = "tts/qwen3-tts-1.7b";
pub const MODEL_0_6B: &str = "tts/qwen3-tts-0.6b";

/// A voice-library voice registered with the running server (TS `RustQwenCloneVoice`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct QwenCloneVoice {
    pub voice_id: String,
    /// Server-side voice name to send as the request's `voice`.
    pub voice: String,
    /// No extraction ran for this call: the voice was registered already, or from the voice
    /// library's cached speaker data.
    pub cached: bool,
}

/// Static facts about the engine (`available`: a runtime pack, the tokenizer and a talker are
/// installed).
pub fn info() -> TtsHelperInfo {
    let installed = |id: &str| model_store::installed_path(id).is_some();
    let runtime = [plan::RUNTIME_VULKAN, plan::RUNTIME_CUDA13, plan::RUNTIME_CPU]
        .iter()
        .any(|id| installed(id));
    let available = runtime && installed(plan::TOKENIZER_ID) && plan::TALKER_IDS.iter().any(|id| installed(id));
    TtsHelperInfo {
        engine: ENGINE,
        title: "Qwen3-TTS (qwentts.cpp)".into(),
        experimental: false,
        gpu: true,
        cloning: true,
        requires: vec![plan::RUNTIME_VULKAN.into(), MODEL_1_7B.into(), plan::TOKENIZER_ID.into()],
        available,
    }
}

/// Start (or reuse) the helper and return where to reach it. `config` is [`plan::QwenConfig`]
/// (`{ model, variant, mode, language }`, camelCase). Downloads nothing: a missing model or pack is
/// an error naming the catalog ids (the UI asks for consent with `<ModelRequirement>` first).
pub async fn start<R: Runtime>(app: &AppHandle<R>, config: Value) -> Result<TtsHelperEndpoint, String> {
    let _ = app;
    let config: plan::QwenConfig = serde_json::from_value(config).map_err(|e| format!("invalid Qwen3-TTS settings: {e}"))?;
    server::start(config).await
}

/// Stop the helper (Ok when it is not running).
pub async fn stop() -> Result<(), String> {
    server::stop().await
}

/// Current state.
pub fn status() -> TtsHelperStatus {
    server::status()
}

/// Register the voice-library voice `voice_id` with the running server (the helper must be
/// started in clone mode) and return the server voice name to request.
#[command]
pub async fn qwen_tts_prepare_clone<R: Runtime>(app: AppHandle<R>, voice_id: String) -> Result<QwenCloneVoice, String> {
    let _ = &app;
    clone::prepare(&voice_id).await
}
