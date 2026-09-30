//! `melo-tts` plugin: client for a MeloTTS HTTP server (default `http://localhost:8888`,
//! `GET /speakers`, `POST /tts/generate` returning WAV).
//!
//! Synthesis streams into the TTS sink (`melo_speak_to_sink`, P2-I): the WAV body is decoded as it
//! arrives by the shared HTTP speech core (`openai_speech::speak_http`), so the output device, the
//! voice-changer route, stop and skip apply and no audio crosses IPC. The server's API is not
//! OpenAI-compatible, so the request keeps its own shape (`{text, speaker_id, speed, sample_rate}`).
//!
//! Every command takes an optional per-call `endpoint` (blank = the one stored by
//! `set_melo_endpoint`). Timeouts: speakers 20 s, health 10 s; synthesis on this PC or the LAN
//! 60 s + 1 s per character to the first audio (max 10 min), else 60 s (see `openai_speech`).

use crate::services::http_client::{
    api_url, describe_api_error, describe_request_error, http_client, parse_named_list,
    pick_endpoint, probe_ok, LIST_TIMEOUT,
};
use crate::services::openai_speech::{speak_http, HttpSpeech, SpeakReport, SpeechFormat};
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

/// Sample rate asked from the server (MeloTTS models run at 44.1 kHz; 22.05 kHz halves the body).
const REQUEST_SAMPLE_RATE: u32 = 22_050;

/// The streaming request for one sentence: `POST {base}/tts/generate`, WAV back.
fn melo_spec(base: &str, text: &str, speaker_id: Option<&str>, speed: Option<f32>) -> Result<HttpSpeech, String> {
    let url = api_url(base, "/tts/generate")?;
    let body = serde_json::to_value(GenerateRequest {
        text,
        speaker_id: speaker_id.map(str::trim).filter(|s| !s.is_empty()),
        speed: speed.filter(|s| s.is_finite() && *s > 0.0),
        sample_rate: Some(REQUEST_SAMPLE_RATE),
    })
    .map_err(|e| format!("MeloTTS request: {e}"))?;
    Ok(HttpSpeech::new("MeloTTS synthesis", url, body, text, SpeechFormat::Wav))
}

/// Speak `text` into the begun TTS sink utterance `utteranceId` (streamed WAV). Resolves once the
/// last audio was queued; `Err("cancelled")` after stop / skip / `openai_speech_cancel`.
#[command]
async fn melo_speak_to_sink<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, MeloTTSState>,
    text: String,
    speaker_id: Option<String>,
    speed: Option<f32>,
    endpoint: Option<String>,
    utterance_id: String,
) -> Result<SpeakReport, String> {
    let base = state.endpoint_for(endpoint.as_deref());
    let spec = melo_spec(&base, &text, speaker_id.as_deref(), speed)?;
    speak_http(spec, &utterance_id).await
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
            melo_speak_to_sink,
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

    #[test]
    fn melo_spec_keeps_the_legacy_request() {
        let spec = melo_spec("http://localhost:8888/", "Hello.", Some(" EN-US "), Some(1.2)).unwrap();
        assert_eq!(spec.url().as_str(), "http://localhost:8888/tts/generate");
        assert_eq!(
            spec.body(),
            &serde_json::json!({"text": "Hello.", "speaker_id": "EN-US", "speed": 1.2f32, "sample_rate": 22050})
        );
        let spec = melo_spec("http://localhost:8888", "Hello.", Some("  "), Some(f32::NAN)).unwrap();
        assert_eq!(spec.body(), &serde_json::json!({"text": "Hello.", "sample_rate": 22050}));
        assert!(melo_spec("", "Hello.", None, None).is_err());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn melo_streams_the_wav_answer_into_the_sink() {
        use crate::services::openai_speech::testing::*;
        let values: Vec<i16> = (0..2205).map(|i| (i % 64) as i16 * 100).collect();
        let (base, server) = serve_once(full_response("200 OK", "audio/wav", &wav16(&values, 22_050, 1))).await;
        let (sink, pushes) = Recorder::new();
        let report = speak(melo_spec(&base, "Hello.", None, None).unwrap(), "melo-test", Box::new(sink)).await.unwrap();
        let seen = server.await.unwrap();
        assert!(seen.head.starts_with("POST /tts/generate "), "{}", seen.head);
        assert_eq!(report.sample_rate, 22_050);
        assert_eq!(report.frames, values.len() as u64);
        assert_eq!(pushes.samples(), f(&values));
    }
}
