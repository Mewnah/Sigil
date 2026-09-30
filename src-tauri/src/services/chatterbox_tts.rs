//! `chatterbox-tts` plugin: voice list and health check of an OpenAI-compatible Chatterbox server
//! (default `http://localhost:5555`).
//!
//! Synthesis goes through the generic OpenAI-compatible client (`openai-speech` plugin,
//! `openai_speech_speak`, P2-I): the `chatterbox` backend maps its saved settings onto it
//! (`{endpoint}/v1`, model `chatterbox`, WAV, `exaggeration` as an extra body member), streamed into
//! the TTS sink. Slow CPU servers get the long local budget of that client.
//!
//! Every command takes an optional per-call `endpoint` (blank = the one stored by
//! `set_chatterbox_endpoint`). Timeouts: voices 20 s, health 10 s.

use crate::services::http_client::{
    api_url, describe_api_error, describe_request_error, http_client, parse_named_list,
    pick_endpoint, probe_ok, LIST_TIMEOUT,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, PoisonError};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://localhost:5555";

#[derive(Clone)]
pub struct ChatterboxTTSState {
    endpoint: Arc<Mutex<String>>,
}

impl ChatterboxTTSState {
    fn new() -> Self {
        Self {
            endpoint: Arc::new(Mutex::new(DEFAULT_ENDPOINT.to_string())),
        }
    }

    fn endpoint_for(&self, call: Option<&str>) -> String {
        let stored = self.endpoint.lock().unwrap_or_else(PoisonError::into_inner);
        pick_endpoint(call, &stored)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct ChatterboxVoice {
    pub id: String,
    pub name: String,
}

/// Voices from `GET /v1/audio/voices`: `{"voices": [{"id","name"}]}`, a list of names, or a bare
/// array.
fn parse_voices(v: &serde_json::Value) -> Vec<ChatterboxVoice> {
    parse_named_list(
        v,
        &["voices", "data", "items"],
        &["id", "voice_id", "name"],
        &["name", "display_name", "id"],
    )
    .into_iter()
    .map(|e| ChatterboxVoice { id: e.id, name: e.name })
    .collect()
}

/// Set the Chatterbox API endpoint URL (the default for calls without `endpoint`).
#[command]
async fn set_chatterbox_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, ChatterboxTTSState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().unwrap_or_else(PoisonError::into_inner) = endpoint.trim().to_string();
    Ok(())
}

/// Get available voices from Chatterbox.
#[command]
async fn get_chatterbox_voices<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, ChatterboxTTSState>,
    endpoint: Option<String>,
) -> Result<Vec<ChatterboxVoice>, String> {
    let url = api_url(&state.endpoint_for(endpoint.as_deref()), "/v1/audio/voices")?;
    let what = "Chatterbox voice list";

    let response = http_client()
        .get(url)
        .timeout(LIST_TIMEOUT)
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, LIST_TIMEOUT))?;
    if !response.status().is_success() {
        return Err(describe_api_error(what, response).await);
    }
    let v: serde_json::Value = response
        .json()
        .await
        .map_err(|e| format!("{what}: unexpected response ({e})"))?;
    Ok(parse_voices(&v))
}

/// Check if the Chatterbox service answers (`GET /v1/audio/voices` within 10 s).
#[command]
async fn check_chatterbox_availability<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, ChatterboxTTSState>,
    endpoint: Option<String>,
) -> Result<bool, String> {
    let Ok(url) = api_url(&state.endpoint_for(endpoint.as_deref()), "/v1/audio/voices") else {
        return Ok(false);
    };
    Ok(probe_ok("Chatterbox availability check", http_client().get(url)).await)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("chatterbox-tts")
        .setup(|app, _api| {
            app.manage(ChatterboxTTSState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_chatterbox_endpoint,
            get_chatterbox_voices,
            check_chatterbox_availability
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chatterbox_voices_accept_known_shapes() {
        let a = parse_voices(&serde_json::json!({"voices": [{"id": "v1", "name": "Voice One"}]}));
        assert_eq!(a, vec![ChatterboxVoice { id: "v1".into(), name: "Voice One".into() }]);
        let b = parse_voices(&serde_json::json!({"voices": ["alice"]}));
        assert_eq!(b, vec![ChatterboxVoice { id: "alice".into(), name: "alice".into() }]);
        let c = parse_voices(&serde_json::json!([{"name": "bob"}]));
        assert_eq!(c[0].id, "bob");
    }
}
