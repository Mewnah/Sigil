//! HTTP sidecar TTS for heavy / custom neural models (`POST {endpoint}/speech` → WAV,
//! `GET {endpoint}/health`; reference server: `scripts/neural-sidecar/tts_sidecar.py`).
//!
//! Every command takes an optional per-call `endpoint` (blank = the one stored by
//! `set_neural_sidecar_tts_endpoint`). Timeouts: health 10 s; synthesis 60 s + 1 s per character
//! (max 10 min) on this PC or the LAN, else 30 s + 100 ms per character (max 120 s) (`tts_post`);
//! the body is read with a 30 s stall limit.

use crate::services::http_client::{
    api_url, content_type, describe_api_error, describe_request_error, ensure_audio_body,
    http_client, pick_endpoint, probe_ok, read_body_capped, tts_post, MAX_AUDIO_BYTES,
    TTS_STALL_TIMEOUT,
};
use crate::services::voice_changer::{push_aux_wav_bytes, VoiceChangerState};
use serde::Serialize;
use std::sync::{Arc, Mutex, PoisonError};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://127.0.0.1:8766";

#[derive(Clone)]
pub struct NeuralSidecarTtsState {
    endpoint: Arc<Mutex<String>>,
}

impl NeuralSidecarTtsState {
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

#[derive(Serialize, Debug)]
struct SpeechRequest<'a> {
    text: &'a str,
    voice: &'a str,
}

#[command]
async fn set_neural_sidecar_tts_endpoint<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    endpoint: String,
) -> Result<(), String> {
    *state.endpoint.lock().unwrap_or_else(PoisonError::into_inner) =
        endpoint.trim().trim_end_matches('/').to_string();
    Ok(())
}

async fn sidecar_fetch_wav_bytes(base: &str, text: &str, voice: &str) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let url = api_url(base, "/speech")?;
    let what = "Neural sidecar TTS";
    let voice = voice.trim();
    let (req, timeout) = tts_post(url, text);
    let response = req
        .json(&SpeechRequest {
            text,
            voice: if voice.is_empty() { "default" } else { voice },
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

#[command]
async fn neural_sidecar_tts_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    text: String,
    voice: String,
    endpoint: Option<String>,
) -> Result<Vec<u8>, String> {
    let base = state.endpoint_for(endpoint.as_deref());
    sidecar_fetch_wav_bytes(&base, &text, &voice).await
}

/// One IPC hop: fetch WAV from the sidecar and mix it into the voice changer. Decoding, resampling
/// and queueing run on a blocking thread, never on an async worker.
#[command]
async fn neural_sidecar_tts_speak_into_voice_changer<R: Runtime>(
    app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    text: String,
    voice: String,
    endpoint: Option<String>,
) -> Result<(), String> {
    let base = state.endpoint_for(endpoint.as_deref());
    let wav = sidecar_fetch_wav_bytes(&base, &text, &voice).await?;
    if wav.is_empty() {
        return Ok(());
    }
    tauri::async_runtime::spawn_blocking(move || {
        let vc = app
            .try_state::<VoiceChangerState>()
            .ok_or("The voice changer is not available.")?;
        push_aux_wav_bytes(&vc, wav)
    })
    .await
    .map_err(|e| format!("Voice changer hand-off failed: {e}"))?
}

#[command]
async fn check_neural_sidecar_tts<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    endpoint: Option<String>,
) -> Result<bool, String> {
    let Ok(url) = api_url(&state.endpoint_for(endpoint.as_deref()), "/health") else {
        return Ok(false);
    };
    Ok(probe_ok("Neural sidecar TTS availability check", http_client().get(url)).await)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("neural-sidecar-tts")
        .setup(|app, _api| {
            app.manage(NeuralSidecarTtsState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_neural_sidecar_tts_endpoint,
            neural_sidecar_tts_speak,
            neural_sidecar_tts_speak_into_voice_changer,
            check_neural_sidecar_tts,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sidecar_endpoint_override_does_not_touch_stored_value() {
        let s = NeuralSidecarTtsState::new();
        assert_eq!(s.endpoint_for(Some("http://10.0.0.2:9000")), "http://10.0.0.2:9000");
        assert_eq!(s.endpoint_for(None), DEFAULT_ENDPOINT);
    }

    #[tokio::test]
    async fn empty_text_makes_no_request() {
        // An unroutable endpoint proves no request is attempted.
        assert_eq!(sidecar_fetch_wav_bytes("http://invalid.invalid", "  ", "v").await.unwrap(), Vec::<u8>::new());
    }
}
