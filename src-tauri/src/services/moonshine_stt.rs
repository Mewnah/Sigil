//! `moonshine-stt` plugin: client for a Moonshine HTTP server (`POST {endpoint}/transcribe` with a
//! base64 WAV, `GET {endpoint}/health`). Legacy: the speech pipeline's HTTP engine
//! (`speech::engines::http`, preset `moonshine`) sends one request per VAD utterance from Rust and
//! shares the request code below.
//!
//! Timeouts: health 10 s; transcription 30 s plus the clip's duration (estimated from the WAV size
//! at 16 kHz mono 16-bit), capped at 120 s. Without them a hung server held an utterance for up to
//! the shared client's 2 h download cap.

use crate::services::http_client::{api_url, http_client, probe_ok, stt_timeout_for_wav_bytes};
use crate::services::speech::engines::http::{moonshine_request, Failure};
use std::sync::{Arc, Mutex, PoisonError};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://localhost:8090";

#[derive(Clone)]
pub struct MoonshineSttState {
    endpoint: Arc<Mutex<String>>,
}

impl MoonshineSttState {
    fn new() -> Self {
        Self {
            endpoint: Arc::new(Mutex::new(DEFAULT_ENDPOINT.to_string())),
        }
    }

    fn endpoint(&self) -> String {
        self.endpoint
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }
}

/// Decoded size of a base64 payload (what the WAV occupies), for the timeout estimate.
fn base64_decoded_len(b64: &str) -> usize {
    let trimmed = b64.trim_end_matches('=');
    trimmed.len() / 4 * 3 + (trimmed.len() % 4).saturating_sub(1)
}

/// Set the Moonshine API endpoint URL
#[command]
async fn set_moonshine_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, MoonshineSttState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().unwrap_or_else(PoisonError::into_inner) = endpoint.trim().to_string();
    Ok(())
}

/// Transcribe audio using the Moonshine API.
/// Audio should be base64-encoded WAV or raw PCM data.
#[command]
async fn moonshine_transcribe<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, MoonshineSttState>,
    audio_base64: String,
    language: Option<String>,
) -> Result<String, String> {
    if audio_base64.is_empty() {
        return Ok(String::new());
    }
    let url = api_url(&state.endpoint(), "/transcribe")?;
    let timeout = stt_timeout_for_wav_bytes(base64_decoded_len(&audio_base64));
    moonshine_request(url, &audio_base64, language.as_deref(), timeout)
        .await
        .map_err(|f| match f {
            Failure::Cancelled => "Moonshine transcription cancelled".to_string(),
            Failure::Retryable(m) | Failure::Fatal(m) => m,
            Failure::RateLimited { message, .. } => message,
        })
}

/// Check if the Moonshine service answers (`GET /health` within 10 s).
#[command]
async fn check_moonshine_availability<R: Runtime>(_app: AppHandle<R>, state: State<'_, MoonshineSttState>) -> Result<bool, String> {
    let Ok(url) = api_url(&state.endpoint(), "/health") else {
        return Ok(false);
    };
    Ok(probe_ok("Moonshine availability check", http_client().get(url)).await)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("moonshine-stt")
        .setup(|app, _api| {
            app.manage(MoonshineSttState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_moonshine_endpoint,
            moonshine_transcribe,
            check_moonshine_availability
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::http_client::{STT_TIMEOUT, STT_TIMEOUT_MAX};
    use base64::{engine::general_purpose::STANDARD, Engine as _};
    use std::time::Duration;

    #[test]
    fn base64_length_estimate_is_exact() {
        for n in 0..50usize {
            let b64 = STANDARD.encode(vec![7u8; n]);
            assert_eq!(base64_decoded_len(&b64), n, "n = {n}");
        }
    }

    #[test]
    fn moonshine_timeout_follows_clip_length() {
        let ten_seconds = STANDARD.encode(vec![0u8; 32_000 * 10]);
        assert_eq!(
            stt_timeout_for_wav_bytes(base64_decoded_len(&ten_seconds)),
            Duration::from_secs(40)
        );
        assert_eq!(stt_timeout_for_wav_bytes(0), STT_TIMEOUT);
        assert_eq!(stt_timeout_for_wav_bytes(usize::MAX), STT_TIMEOUT_MAX);
    }
}
