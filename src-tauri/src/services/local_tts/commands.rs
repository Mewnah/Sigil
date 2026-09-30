//! The `local-tts` plugin: in-process Kokoro / Kokoro v1.1-zh / Piper / Kitten through sherpa-onnx
//! `OfflineTts` (the vendored Kitten as fallback and for legacy speech packages). Owned by P2-H;
//! API frozen by P2-S0.
//!
//! Models come from the model store (`model_store::installed_path`; catalog fragment
//! `model-catalog/tts-local.json`, run as described by `local_tts::catalog`), from a
//! `speech_package_v1` or official sherpa package folder, or from a KittenML Hub repo
//! (`hub:<org>/<name>`); see `local_tts::models`. Nothing here downloads a catalog model: the TS
//! panel asks first (`<ModelRequirement>`). All inference runs on the `sigil-local-tts` worker
//! thread (`local_tts::engine`), never on the main thread; sherpa TTS generation is serialized
//! process-wide (espeak-ng). Output follows the PCM contract: `speak_local_tts` pushes into the
//! begun `tts_sink` utterance.
//!
//! # Frozen contract (P2-S0; change only through an INTEGRATE step)
//! Types may gain fields; names and wire formats stay (camelCase). TS mirrors: `RustLocalTts*` in
//! `src/types.ts`.
//! - `plugin:local-tts|list_local_tts_models` → [`LocalTtsModelInfo`]`[]` (catalog + installed
//!   state; cheap, no model load).
//! - `plugin:local-tts|load_local_tts { model, threads? }` → [`LocalTtsModelInfo`] (loads / warms
//!   up; `Err` names a missing catalog id). P2-H adds the optional `voice` and `language`
//!   arguments: a Kokoro model is configured for its voice's language when it loads, so passing
//!   the voice avoids a second load on the first sentence.
//! - `plugin:local-tts|speak_local_tts { request: LocalTtsRequest, utteranceId }` →
//!   [`SpeakReport`]: synthesizes into the begun `tts_sink` utterance; resolves after the last push;
//!   `Err("cancelled")` when the utterance was cancelled.
//! - `plugin:local-tts|synthesize_local_tts { request }` → `ArrayBuffer` (little-endian f32 mono
//!   at the model's `sampleRate`; previews and tests, not the playback path).
//! - `plugin:local-tts|cancel_local_tts { utteranceId?: string | null }` → `void`.
//! - `plugin:local-tts|unload_local_tts` → `void` (frees the model).
//!
//! Fields added by P2-H (additive): [`LocalTtsVoice::gender`], [`LocalTtsModelInfo::default_voice`].

use std::sync::Arc;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

pub use crate::services::tts_sink::SpeakReport;

use super::engine::{AppSink, Engine, IDLE_UNLOAD, WORKER_STOPPED};
use super::models::{self, StoreLoader};

/// Plugin name (`plugin:local-tts|...`, capability `local-tts:default`).
pub const PLUGIN_NAME: &str = "local-tts";

/// One synthesis request (one sentence).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LocalTtsRequest {
    /// Catalog id (`tts/...`) or a local package folder.
    pub model: String,
    /// Voice / speaker id of the model.
    pub voice: String,
    pub speed: f32,
    /// Language hint (normalized code) for multi-language models; `None` = the voice's language.
    pub language: Option<String>,
    pub text: String,
    /// Inference threads (2-4; `None` = default).
    pub threads: Option<u32>,
}

impl Default for LocalTtsRequest {
    fn default() -> Self {
        Self {
            model: String::new(),
            voice: String::new(),
            speed: 1.0,
            language: None,
            text: String::new(),
            threads: None,
        }
    }
}

/// A voice of a local model.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LocalTtsVoice {
    pub id: String,
    pub name: String,
    /// Normalized language code, if known.
    pub language: Option<String>,
    /// `female` / `male`, when the model says (P2-H).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gender: Option<String>,
}

/// A local TTS model as the UI lists it.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct LocalTtsModelInfo {
    /// Catalog id or local folder.
    pub id: String,
    /// `kokoro`, `piper` or `kitten`.
    pub family: String,
    pub title: String,
    pub languages: Vec<String>,
    pub voices: Vec<LocalTtsVoice>,
    pub sample_rate: u32,
    pub installed: bool,
    /// Voice id used when a request names none (P2-H).
    pub default_voice: String,
}

/// The app's engine (worker thread started on the first request).
static ENGINE: Lazy<Engine> = Lazy::new(|| Engine::new(Arc::new(StoreLoader::default()), Arc::new(AppSink), IDLE_UNLOAD));

/// The engine's answer, or the error of a dropped reply.
async fn answer<T>(rx: tokio::sync::oneshot::Receiver<Result<T, String>>) -> Result<T, String> {
    rx.await
        .map_err(|_| WORKER_STOPPED.to_string())?
}

#[command]
async fn list_local_tts_models() -> Result<Vec<LocalTtsModelInfo>, String> {
    tauri::async_runtime::spawn_blocking(models::list_models)
        .await
        .map_err(|e| format!("local speech model list failed: {e}"))
}

#[command]
async fn load_local_tts(
    model: String,
    threads: Option<u32>,
    voice: Option<String>,
    language: Option<String>,
) -> Result<LocalTtsModelInfo, String> {
    answer(ENGINE.load(model, voice.unwrap_or_default(), language, threads)).await
}

#[command]
async fn speak_local_tts(request: LocalTtsRequest, utterance_id: String) -> Result<SpeakReport, String> {
    if utterance_id.trim().is_empty() {
        return Err("utteranceId must not be empty".into());
    }
    answer(ENGINE.speak(request, utterance_id)).await
}

#[command]
async fn synthesize_local_tts(request: LocalTtsRequest) -> Result<tauri::ipc::Response, String> {
    let audio = answer(ENGINE.synthesize(request)).await?;
    let mut bytes = Vec::with_capacity(audio.samples.len() * 4);
    for s in &audio.samples {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    Ok(tauri::ipc::Response::new(bytes))
}

#[command]
async fn cancel_local_tts(utterance_id: Option<String>) -> Result<(), String> {
    ENGINE.cancel(utterance_id.as_deref());
    Ok(())
}

#[command]
async fn unload_local_tts() -> Result<(), String> {
    answer(ENGINE.unload()).await
}

/// Load the sherpa runtime early when a local voice is installed: `SherpaRuntime::load` refuses
/// once another `onnxruntime.dll` is in the process (ADR 0002), so the earlier the better. Cheap
/// (a few ms, two DLLs); never downloads.
fn preload_runtime() {
    let any_installed = super::catalog::all()
        .iter()
        .any(|m| crate::services::model_store::installed_path(&m.id).is_some());
    if !any_installed {
        return;
    }
    if let Err(e) = crate::services::speech_rt::sherpa::SherpaRuntime::installed() {
        tracing::warn!(target: "sigil::tts", "local TTS: the sherpa-onnx runtime did not load: {e}");
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .setup(|_app, _api| {
            let _ = std::thread::Builder::new()
                .name("sigil-local-tts-init".into())
                .spawn(preload_runtime);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            list_local_tts_models,
            load_local_tts,
            speak_local_tts,
            synthesize_local_tts,
            cancel_local_tts,
            unload_local_tts,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wire_names_are_frozen() {
        let r: LocalTtsRequest = serde_json::from_str(r#"{"model":"tts/kokoro","voice":"af_heart","text":"hi"}"#).unwrap();
        assert_eq!(r.speed, 1.0);
        assert!(r.language.is_none() && r.threads.is_none());
        let v = serde_json::to_value(LocalTtsModelInfo::default()).unwrap();
        for key in ["id", "family", "title", "languages", "voices", "sampleRate", "installed", "defaultVoice"] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        let voice = serde_json::to_value(LocalTtsVoice {
            id: "af_heart".into(),
            name: "Heart".into(),
            language: Some("en-US".into()),
            gender: Some("female".into()),
        })
        .unwrap();
        assert_eq!(voice["gender"], "female");
        let bare = serde_json::to_value(LocalTtsVoice::default()).unwrap();
        assert!(bare.get("gender").is_none(), "gender is left out when unknown");
        assert!(bare.get("language").is_some(), "language stays (null) as frozen");
    }
}
