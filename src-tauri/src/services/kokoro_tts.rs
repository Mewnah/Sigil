//! `kokoro-tts` plugin: voice list and health check of an OpenAI-compatible Kokoro server
//! (Kokoro-FastAPI, default `http://localhost:8880`).
//!
//! Synthesis goes through the generic OpenAI-compatible client (`openai-speech` plugin,
//! `openai_speech_speak`, P2-I): the `kokoro` backend maps its saved settings onto it
//! (`{endpoint}/v1`, model `kokoro`, raw 24 kHz PCM, streamed into the TTS sink), so the first audio
//! plays before the sentence is finished and the output device / voice-changer route apply.
//!
//! Every command takes an optional per-call `endpoint` (blank = the one stored by
//! `set_kokoro_endpoint`), so the inspector's voice list never re-points a running service.
//! Timeouts: voices 20 s, health 10 s.

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

const DEFAULT_ENDPOINT: &str = "http://localhost:8880";

#[derive(Clone)]
pub struct KokoroTTSState {
    endpoint: Arc<Mutex<String>>,
}

impl KokoroTTSState {
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
pub struct KokoroVoice {
    pub name: String,
}

/// Voice ids from `GET /v1/audio/voices`. Kokoro-FastAPI answers `{"voices": ["af_bella", ...]}`;
/// other servers answer `{"voices": [{"name": ...}]}`, `{"data": [...]}` or a bare array. All
/// shapes are accepted (the old strict parser failed on the string list and showed no voices).
///
/// `KokoroVoice::name` is the value sent back as `voice`, so for object items a real id field
/// (`id`, `voice_id`, `voice`) wins over a display `name` (`{"id":"af_heart","name":"Heart"}` must
/// send `af_heart`); `name` is used only when it is the only field.
fn parse_voices(v: &serde_json::Value) -> Vec<KokoroVoice> {
    parse_named_list(
        v,
        &["voices", "data", "items"],
        &["id", "voice_id", "voice", "name"],
        &["name"],
    )
    .into_iter()
    .map(|e| KokoroVoice { name: e.id })
    .collect()
}

/// Set the Kokoro API endpoint URL (the default for calls without `endpoint`).
#[command]
async fn set_kokoro_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, KokoroTTSState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().unwrap_or_else(PoisonError::into_inner) = endpoint.trim().to_string();
    Ok(())
}

/// Get available voices from Kokoro-FastAPI.
#[command]
async fn get_kokoro_voices<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, KokoroTTSState>,
    endpoint: Option<String>,
) -> Result<Vec<KokoroVoice>, String> {
    let url = api_url(&state.endpoint_for(endpoint.as_deref()), "/v1/audio/voices")?;
    let what = "Kokoro voice list";

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

/// Check if the Kokoro service answers (`GET /v1/audio/voices` within 10 s).
#[command]
async fn check_kokoro_availability<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, KokoroTTSState>,
    endpoint: Option<String>,
) -> Result<bool, String> {
    let Ok(url) = api_url(&state.endpoint_for(endpoint.as_deref()), "/v1/audio/voices") else {
        return Ok(false);
    };
    Ok(probe_ok("Kokoro availability check", http_client().get(url)).await)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("kokoro-tts")
        .setup(|app, _api| {
            app.manage(KokoroTTSState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_kokoro_endpoint,
            get_kokoro_voices,
            check_kokoro_availability
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(v: serde_json::Value) -> Vec<String> {
        parse_voices(&v).into_iter().map(|v| v.name).collect()
    }

    #[test]
    fn kokoro_voices_accept_string_list() {
        // Kokoro-FastAPI's actual shape.
        assert_eq!(
            names(serde_json::json!({"voices": ["af_bella", "jf_alpha", "zf_xiaobei"]})),
            ["af_bella", "jf_alpha", "zf_xiaobei"]
        );
    }

    #[test]
    fn kokoro_voices_accept_object_list_and_bare_array() {
        assert_eq!(names(serde_json::json!({"voices": [{"name": "af_sky"}]})), ["af_sky"]);
        assert_eq!(names(serde_json::json!([{"id": "am_adam"}, "bf_emma"])), ["am_adam", "bf_emma"]);
        assert_eq!(names(serde_json::json!({"data": [{"voice_id": "hf_alpha"}]})), ["hf_alpha"]);
        assert!(names(serde_json::json!({"detail": "Not Found"})).is_empty());
    }

    #[test]
    fn kokoro_voices_send_the_id_not_the_display_name() {
        // Review finding: with both fields present the display name used to win, so the server got
        // `voice: "Heart"` and rejected it.
        assert_eq!(
            names(serde_json::json!({"voices": [{"id": "af_heart", "name": "Heart"}, {"voice_id": "jf_alpha", "name": "Alpha"}]})),
            ["af_heart", "jf_alpha"]
        );
        assert_eq!(names(serde_json::json!({"voices": [{"voice": "bf_emma", "name": "Emma"}]})), ["bf_emma"]);
        // Legacy `{name}` objects still work.
        assert_eq!(names(serde_json::json!({"voices": [{"name": "af_sky", "language": "en-us"}]})), ["af_sky"]);
    }
}
