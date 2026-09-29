//! `kokoro-tts` plugin: client for an OpenAI-compatible Kokoro server (Kokoro-FastAPI, default
//! `http://localhost:8880`).
//!
//! Every command takes an optional per-call `endpoint` (blank = the one stored by
//! `set_kokoro_endpoint`), so the inspector's voice list never re-points a running service.
//! Timeouts: voices 20 s, health 10 s, synthesis 30 s + 100 ms per character (max 120 s), or
//! 60 s + 1 s per character (max 10 min) for a server on this PC or the LAN (`tts_post`).

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

const DEFAULT_ENDPOINT: &str = "http://localhost:8880";
const DEFAULT_VOICE: &str = "af_bella";

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

#[derive(Serialize, Debug)]
struct SpeechRequest<'a> {
    model: &'static str,
    input: &'a str,
    voice: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    response_format: &'static str,
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

/// A finite speed within Kokoro's accepted range, or `None` to use the server default.
fn sanitize_speed(speed: Option<f32>) -> Option<f32> {
    speed.filter(|s| s.is_finite() && *s > 0.0).map(|s| s.clamp(0.25, 4.0))
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

/// Generate speech using Kokoro TTS and return audio bytes (mp3).
#[command]
async fn kokoro_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, KokoroTTSState>,
    text: String,
    voice: String,
    speed: Option<f32>,
    endpoint: Option<String>,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let url = api_url(&state.endpoint_for(endpoint.as_deref()), "/v1/audio/speech")?;
    let what = "Kokoro synthesis";
    let voice = voice.trim();

    let (req, timeout) = tts_post(url, &text);
    let response = req
        .json(&SpeechRequest {
            model: "kokoro",
            input: &text,
            voice: if voice.is_empty() { DEFAULT_VOICE } else { voice },
            speed: sanitize_speed(speed),
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
            kokoro_speak,
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

    #[test]
    fn kokoro_speed_is_sanitized() {
        assert_eq!(sanitize_speed(None), None);
        assert_eq!(sanitize_speed(Some(f32::NAN)), None);
        assert_eq!(sanitize_speed(Some(0.0)), None);
        assert_eq!(sanitize_speed(Some(1.2)), Some(1.2));
        assert_eq!(sanitize_speed(Some(50.0)), Some(4.0));
    }

    #[test]
    fn kokoro_request_shape() {
        let v = serde_json::to_value(SpeechRequest {
            model: "kokoro",
            input: "hi",
            voice: DEFAULT_VOICE,
            speed: None,
            response_format: "mp3",
        })
        .unwrap();
        assert_eq!(v, serde_json::json!({"model": "kokoro", "input": "hi", "voice": "af_bella", "response_format": "mp3"}));
    }
}
