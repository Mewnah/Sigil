//! `chatterbox-tts` plugin: client for an OpenAI-compatible Chatterbox server (default
//! `http://localhost:5555`).
//!
//! Every command takes an optional per-call `endpoint` (blank = the one stored by
//! `set_chatterbox_endpoint`). Timeouts: voices 20 s, health 10 s, synthesis 60 s + 1 s per
//! character (max 10 min) on this PC or the LAN, where Chatterbox often runs on CPU and answers only
//! when the whole clip is done, else 30 s + 100 ms per character (max 120 s); see `tts_post`.

use crate::services::http_client::{
    api_url, content_type, describe_api_error, describe_request_error, ensure_audio_body,
    http_client, parse_named_list, pick_endpoint, probe_ok, read_body_capped, tts_post,
    LIST_TIMEOUT, MAX_AUDIO_BYTES, TTS_STALL_TIMEOUT,
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

#[derive(Serialize, Debug)]
struct SpeechRequest<'a> {
    model: &'static str,
    input: &'a str,
    voice: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exaggeration: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cfg_weight: Option<f32>,
    response_format: &'static str,
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

/// A finite, positive value, or `None` (server default).
fn finite_positive(v: Option<f32>) -> Option<f32> {
    v.filter(|x| x.is_finite() && *x >= 0.0)
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

/// Generate speech using Chatterbox TTS (with voice cloning); returns mp3 bytes.
#[command]
async fn chatterbox_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, ChatterboxTTSState>,
    text: String,
    voice: String,
    speed: Option<f32>,
    exaggeration: Option<f32>,
    endpoint: Option<String>,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let url = api_url(&state.endpoint_for(endpoint.as_deref()), "/v1/audio/speech")?;
    let what = "Chatterbox synthesis";
    let voice = voice.trim();

    let (req, timeout) = tts_post(url, &text);
    let response = req
        .json(&SpeechRequest {
            model: "chatterbox",
            input: &text,
            voice: if voice.is_empty() { "default" } else { voice },
            speed: finite_positive(speed).filter(|s| *s > 0.0),
            exaggeration: finite_positive(exaggeration),
            cfg_weight: None,
            response_format: "mp3",
        })
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, timeout))?;
    if !response.status().is_success() {
        return Err(describe_api_error(what, response).await);
    }
    let ct = content_type(&response);
    let audio = read_body_capped(what, response, MAX_AUDIO_BYTES, TTS_STALL_TIMEOUT).await?;
    ensure_audio_body(what, ct.as_deref(), &audio)?;
    Ok(audio)
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
            chatterbox_speak,
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

    #[test]
    fn chatterbox_numbers_are_sanitized() {
        assert_eq!(finite_positive(Some(f32::INFINITY)), None);
        assert_eq!(finite_positive(Some(-1.0)), None);
        assert_eq!(finite_positive(Some(0.5)), Some(0.5));
    }
}
