//! `tts-helpers` plugin: TTS engines that run as helper processes under `process_host` (P6-S0
//! scaffold). This file (engine list, dispatch, IPC) is the scaffold's; each engine module belongs
//! to its workstream:
//!
//! | Module | Engine id | Owner | Engine |
//! |---|---|---|---|
//! | [`qwen`] | `qwen3` | P6-E | Qwen3-TTS 0.6B / 1.7B through qwentts.cpp's `tts-server` (Sigil-built packs: vulkan default, cuda13 optional); clone registry fed from the voice library |
//! | [`magpie`] | `magpie` | P6-F | NVIDIA Magpie TTS Multilingual through NeMo-Speech.cpp's official packs (experimental) |
//! | [`breeze`] | `breeze` | P6-F | Breeze-TTS-2.cpp's official Vulkan pack, behind [`ggml::adapter`] (experimental) |
//! | [`omnivoice`] | `omnivoice` | P6-F | OmniVoice through CrispASR (not omnivoice.cpp; experimental) |
//! | [`s2`] | `s2` | P6-F | Fish Audio S2 Pro through s2.cpp (Sigil-built packs), behind [`ggml::adapter`] (experimental) |
//! | [`sbv2`] | `sbv2` | P6-G | Style-Bert-VITS2 JP-Extra / AIVMX, **in-process** on Sigil's ONNX Runtime (a port of `sbv2_core` with jpreprocess OpenJTalk; no helper process, empty endpoint `base_url`). TypeScript speaks through `sbv2_speak` / `sbv2_cancel` |
//!
//! [`ggml`] is P6-F's shared runner of the four ggml engines (launch plans, the `Slot` lifecycle and
//! the key-protected request adapter).
//!
//! Every helper binds `127.0.0.1` only, gets a random API key, runs in a kill-on-close Job Object
//! and reports its GPU memory with `resources::claim_external_vram`. Once started, TypeScript speaks
//! through the OpenAI-compatible client (`openai_speech`, `plugin:openai-speech|openai_speech_speak`)
//! with the endpoint returned here (servers with another request shape use `openai_speech`'s
//! `HttpSpeech` / `speak_http` from their engine module and add their own command through
//! INTEGRATE). Like the bundled llama-server (ADR 0005, P4 update), the key is handed to the
//! webview for loopback calls; status payloads never carry it.
//!
//! # Frozen contract (P6-S0; change only through INTEGRATE; structs may gain fields)
//! - Engine ids: [`TtsHelperEngine`] (wire `qwen3`, `magpie`, `breeze`, `omnivoice`, `s2`, `sbv2`);
//!   process_host helper names [`TtsHelperEngine::helper_name`] (`tts-<id>`).
//! - Every engine module implements, with these signatures:
//!   - `pub fn info() -> TtsHelperInfo` (static facts plus whether its pack is installed),
//!   - `pub async fn start<R: Runtime>(app: &AppHandle<R>, config: Value) -> Result<TtsHelperEndpoint, String>`:
//!     ensure models / pack (installed ids only; downloads need the UI's consent first), spawn or
//!     reuse the helper, wait for health; `config` is the engine's own camelCase object (TS
//!     settings slice),
//!   - `pub async fn stop() -> Result<(), String>` (Ok when not running),
//!   - `pub fn status() -> TtsHelperStatus`.
//! - IPC (`plugin-commands/tts-helpers.txt`; wire camelCase; TS mirror `RustTtsHelper*` in
//!   `src/types.ts`):
//!   - `tts_helper_start { engine, config } -> TtsHelperEndpoint`, `tts_helper_stop { engine } -> ()`,
//!     `tts_helper_status { engine } -> TtsHelperStatus`, `tts_helper_list -> TtsHelperInfo[]`.
//!   - Engine-specific: `qwen_tts_prepare_clone { voiceId } -> QwenCloneVoice` ([`qwen`]);
//!     `sbv2_import_model { path } -> Sbv2Model`, `sbv2_list_models -> Sbv2Model[]`,
//!     `sbv2_remove_model { id } -> ()`, `sbv2_speak { request, utteranceId } -> SpeakReport`,
//!     `sbv2_cancel { utteranceId? } -> ()` ([`sbv2`]).
//! - Event [`STATE_EVENT`] `"tts_helpers:state"` (payload [`TtsHelperStatus`]) from the engine
//!   modules on every state change.
//! - Catalog fragments: `tts-qwen3.json` + `runtime-qwentts.json` (P6-E), `tts-ggml-experimental.json`
//!   (P6-F), `runtime-sbv2.json` (P6-G).
#![allow(dead_code)] // Scaffold (P6-S0): states and helpers the engine modules use once implemented.

pub mod breeze;
pub mod ggml;
pub mod magpie;
pub mod omnivoice;
pub mod qwen;
pub mod s2;
pub mod sbv2;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Runtime,
};

/// Plugin name (`plugin:tts-helpers|...`, capability `tts-helpers:default`).
pub const PLUGIN_NAME: &str = "tts-helpers";
/// Event emitted by the engine modules on every state change (payload [`TtsHelperStatus`]).
pub const STATE_EVENT: &str = "tts_helpers:state";

/// A helper-process TTS engine (TS `RustTtsHelperEngine`). Wire ids are persisted by the TS
/// backends: add values, never rename them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TtsHelperEngine {
    Qwen3,
    Magpie,
    Breeze,
    Omnivoice,
    S2,
    Sbv2,
}

impl TtsHelperEngine {
    pub const ALL: [TtsHelperEngine; 6] = [
        TtsHelperEngine::Qwen3,
        TtsHelperEngine::Magpie,
        TtsHelperEngine::Breeze,
        TtsHelperEngine::Omnivoice,
        TtsHelperEngine::S2,
        TtsHelperEngine::Sbv2,
    ];

    /// Wire id.
    pub fn as_str(self) -> &'static str {
        match self {
            TtsHelperEngine::Qwen3 => "qwen3",
            TtsHelperEngine::Magpie => "magpie",
            TtsHelperEngine::Breeze => "breeze",
            TtsHelperEngine::Omnivoice => "omnivoice",
            TtsHelperEngine::S2 => "s2",
            TtsHelperEngine::Sbv2 => "sbv2",
        }
    }

    /// `process_host` helper name of the engine's server.
    pub fn helper_name(self) -> &'static str {
        match self {
            TtsHelperEngine::Qwen3 => "tts-qwen3",
            TtsHelperEngine::Magpie => "tts-magpie",
            TtsHelperEngine::Breeze => "tts-breeze",
            TtsHelperEngine::Omnivoice => "tts-omnivoice",
            TtsHelperEngine::S2 => "tts-s2",
            TtsHelperEngine::Sbv2 => "tts-sbv2",
        }
    }
}

/// Where TypeScript reaches a started engine (TS `RustTtsHelperEndpoint`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TtsHelperEndpoint {
    pub engine: TtsHelperEngine,
    /// Base URL including the API prefix for the OpenAI-compatible client
    /// (`http://127.0.0.1:{port}/v1`); always loopback.
    pub base_url: String,
    /// Bearer key for loopback calls ("" when the server takes none). Never logged.
    pub api_key: String,
    /// Model the server loaded (catalog id or file name; informational).
    pub model: String,
    /// Sample rate of raw `pcm` responses; 0 = ask for WAV.
    pub pcm_sample_rate: u32,
}

/// Lifecycle of an engine's helper (TS `RustTtsHelperState`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum TtsHelperState {
    #[default]
    Stopped,
    Starting,
    Ready,
    Failed,
    /// The engine cannot run in this build or on this PC (pack not built, no GPU, ...).
    Unavailable,
}

/// Status of one engine (TS `RustTtsHelperStatus`; payload of [`STATE_EVENT`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TtsHelperStatus {
    pub engine: TtsHelperEngine,
    pub state: TtsHelperState,
    /// Loaded model (catalog id or file name).
    pub model: Option<String>,
    pub error: Option<String>,
}

/// Static facts about an engine (TS `RustTtsHelperInfo`).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TtsHelperInfo {
    pub engine: TtsHelperEngine,
    /// English display name (the UI localizes by engine id).
    pub title: String,
    pub experimental: bool,
    /// Needs (or strongly prefers) a GPU.
    pub gpu: bool,
    /// Clones voices from the voice library.
    pub cloning: bool,
    /// Catalog ids the engine needs (runtime pack and default model), for `<ModelRequirement>`.
    pub requires: Vec<String>,
    /// The engine can start in this build (its pack and a model are installed).
    pub available: bool,
}

/// Error of a stub engine (scaffold).
pub(crate) fn not_ready(engine: TtsHelperEngine) -> String {
    format!("The {} voice engine is not available in this build yet", engine.as_str())
}

/// `status()` of a stub engine (scaffold).
pub(crate) fn unavailable_status(engine: TtsHelperEngine) -> TtsHelperStatus {
    TtsHelperStatus {
        engine,
        state: TtsHelperState::Unavailable,
        model: None,
        error: None,
    }
}

async fn start_engine<R: Runtime>(app: &AppHandle<R>, engine: TtsHelperEngine, config: Value) -> Result<TtsHelperEndpoint, String> {
    match engine {
        TtsHelperEngine::Qwen3 => qwen::start(app, config).await,
        TtsHelperEngine::Magpie => magpie::start(app, config).await,
        TtsHelperEngine::Breeze => breeze::start(app, config).await,
        TtsHelperEngine::Omnivoice => omnivoice::start(app, config).await,
        TtsHelperEngine::S2 => s2::start(app, config).await,
        TtsHelperEngine::Sbv2 => sbv2::start(app, config).await,
    }
}

async fn stop_engine(engine: TtsHelperEngine) -> Result<(), String> {
    match engine {
        TtsHelperEngine::Qwen3 => qwen::stop().await,
        TtsHelperEngine::Magpie => magpie::stop().await,
        TtsHelperEngine::Breeze => breeze::stop().await,
        TtsHelperEngine::Omnivoice => omnivoice::stop().await,
        TtsHelperEngine::S2 => s2::stop().await,
        TtsHelperEngine::Sbv2 => sbv2::stop().await,
    }
}

fn engine_status(engine: TtsHelperEngine) -> TtsHelperStatus {
    match engine {
        TtsHelperEngine::Qwen3 => qwen::status(),
        TtsHelperEngine::Magpie => magpie::status(),
        TtsHelperEngine::Breeze => breeze::status(),
        TtsHelperEngine::Omnivoice => omnivoice::status(),
        TtsHelperEngine::S2 => s2::status(),
        TtsHelperEngine::Sbv2 => sbv2::status(),
    }
}

fn engine_info(engine: TtsHelperEngine) -> TtsHelperInfo {
    match engine {
        TtsHelperEngine::Qwen3 => qwen::info(),
        TtsHelperEngine::Magpie => magpie::info(),
        TtsHelperEngine::Breeze => breeze::info(),
        TtsHelperEngine::Omnivoice => omnivoice::info(),
        TtsHelperEngine::S2 => s2::info(),
        TtsHelperEngine::Sbv2 => sbv2::info(),
    }
}

#[command]
async fn tts_helper_start<R: Runtime>(app: AppHandle<R>, engine: TtsHelperEngine, config: Value) -> Result<TtsHelperEndpoint, String> {
    start_engine(&app, engine, config).await
}

#[command]
async fn tts_helper_stop(engine: TtsHelperEngine) -> Result<(), String> {
    stop_engine(engine).await
}

#[command]
async fn tts_helper_status(engine: TtsHelperEngine) -> Result<TtsHelperStatus, String> {
    Ok(engine_status(engine))
}

#[command]
async fn tts_helper_list() -> Result<Vec<TtsHelperInfo>, String> {
    Ok(TtsHelperEngine::ALL.into_iter().map(engine_info).collect())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            tts_helper_start,
            tts_helper_stop,
            tts_helper_status,
            tts_helper_list,
            qwen::qwen_tts_prepare_clone,
            sbv2::sbv2_import_model,
            sbv2::sbv2_list_models,
            sbv2::sbv2_remove_model,
            sbv2::sbv2_speak,
            sbv2::sbv2_cancel,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_ids_round_trip_and_helper_names_are_valid() {
        for engine in TtsHelperEngine::ALL {
            let wire = serde_json::to_value(engine).unwrap();
            assert_eq!(wire, Value::String(engine.as_str().to_string()));
            assert_eq!(serde_json::from_value::<TtsHelperEngine>(wire).unwrap(), engine);
            crate::services::process_host::spawn::validate_name(engine.helper_name()).unwrap();
            assert_eq!(engine_info(engine).engine, engine);
            assert_eq!(engine_status(engine).engine, engine);
        }
    }
}
