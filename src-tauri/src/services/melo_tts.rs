//! `melo-tts` plugin: client for a MeloTTS HTTP server (default `http://localhost:8888`,
//! `GET /speakers`, `POST /tts/generate` returning WAV).
//!
//! Every command takes an optional per-call `endpoint` (blank = the one stored by
//! `set_melo_endpoint`). Timeouts: speakers 20 s, health 10 s, synthesis 60 s + 1 s per character
//! (max 10 min) on this PC or the LAN, else 30 s + 100 ms per character (max 120 s); see `tts_post`.

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

const DEFAULT_ENDPOINT: &str = "http://localhost:8888";

#[derive(Clone)]
pub struct MeloTTSState {
    endpoint: Arc<Mutex<String>>,
}

impl MeloTTSState {
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
pub struct MeloSpeaker {
    pub id: String,
    pub name: String,
    pub language: String,
}

#[derive(Serialize, Debug)]
struct GenerateRequest<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_id: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
}

/// Speakers from `GET /speakers`: a bare array of `{id, name, language}` (ids may be numbers), or
/// the same under `speakers` / `voices`.
fn parse_speakers(v: &serde_json::Value) -> Vec<MeloSpeaker> {
    parse_named_list(
        v,
        &["speakers", "voices", "data"],
        &["id", "speaker_id", "name"],
        &["name", "id"],
    )
    .into_iter()
    .map(|e| MeloSpeaker {
        id: e.id,
        name: e.name,
        language: e.language.unwrap_or_default(),
    })
    .collect()
}

/// Set the MeloTTS API endpoint URL (the default for calls without `endpoint`).
#[command]
async fn set_melo_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, MeloTTSState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().unwrap_or_else(PoisonError::into_inner) = endpoint.trim().to_string();
    Ok(())
}

/// Get available speakers from MeloTTS.
#[command]
async fn get_melo_speakers<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, MeloTTSState>,
    endpoint: Option<String>,
) -> Result<Vec<MeloSpeaker>, String> {
    let url = api_url(&state.endpoint_for(endpoint.as_deref()), "/speakers")?;
    let what = "MeloTTS speaker list";

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
    Ok(parse_speakers(&v))
}

/// Generate speech using MeloTTS and return audio bytes (WAV).
#[command]
async fn melo_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, MeloTTSState>,
    text: String,
    speaker_id: Option<String>,
    speed: Option<f32>,
    endpoint: Option<String>,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let url = api_url(&state.endpoint_for(endpoint.as_deref()), "/tts/generate")?;
    let what = "MeloTTS synthesis";

    let (req, timeout) = tts_post(url, &text);
    let response = req
        .json(&GenerateRequest {
            text: &text,
            speaker_id: speaker_id.as_deref().map(str::trim).filter(|s| !s.is_empty()),
            speed: speed.filter(|s| s.is_finite() && *s > 0.0),
            sample_rate: Some(22050),
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

/// Check if the MeloTTS service answers (`GET /speakers` within 10 s).
#[command]
async fn check_melo_availability<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, MeloTTSState>,
    endpoint: Option<String>,
) -> Result<bool, String> {
    let Ok(url) = api_url(&state.endpoint_for(endpoint.as_deref()), "/speakers") else {
        return Ok(false);
    };
    Ok(probe_ok("MeloTTS availability check", http_client().get(url)).await)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("melo-tts")
        .setup(|app, _api| {
            app.manage(MeloTTSState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_melo_endpoint,
            get_melo_speakers,
            melo_speak,
            check_melo_availability
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn melo_speakers_accept_numeric_ids_and_missing_language() {
        let v = serde_json::json!([
            {"id": "EN-US", "name": "American", "language": "EN"},
            {"id": 3, "name": "Japanese"}
        ]);
        let s = parse_speakers(&v);
        assert_eq!(s[0], MeloSpeaker { id: "EN-US".into(), name: "American".into(), language: "EN".into() });
        assert_eq!(s[1].id, "3");
        assert_eq!(s[1].language, "");
    }

    #[test]
    fn melo_request_omits_blank_speaker() {
        let v = serde_json::to_value(GenerateRequest {
            text: "hi",
            speaker_id: None,
            speed: None,
            sample_rate: Some(22050),
        })
        .unwrap();
        assert_eq!(v, serde_json::json!({"text": "hi", "sample_rate": 22050}));
    }
}
