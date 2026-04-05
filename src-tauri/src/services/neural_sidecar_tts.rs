//! HTTP sidecar TTS for heavy / custom neural models (same pattern as Kokoro).

use crate::services::http_client::http_client;
use crate::services::voice_changer::{push_aux_wav_bytes, VoiceChangerState};
use serde::Serialize;
use std::sync::{Arc, Mutex};
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
}

#[derive(Serialize, Debug)]
struct SpeechRequest {
    text: String,
    voice: String,
}

#[command]
async fn set_neural_sidecar_tts_endpoint<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    endpoint: String,
) -> Result<(), String> {
    *state.endpoint.lock().map_err(|_| "Lock failed")? = endpoint.trim_end_matches('/').to_string();
    Ok(())
}

async fn sidecar_fetch_wav_bytes(
    state: &NeuralSidecarTtsState,
    text: String,
    voice: String,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/speech", endpoint);
    let body = SpeechRequest {
        text,
        voice: if voice.trim().is_empty() {
            "default".to_string()
        } else {
            voice
        },
    };
    let response = http_client()
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| format!("Neural sidecar TTS connection failed: {}", e))?;
    if !response.status().is_success() {
        return Err(format!("Neural sidecar TTS error: {}", response.status()));
    }
    response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read audio: {}", e))
        .map(|b| b.to_vec())
}

#[command]
async fn neural_sidecar_tts_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    text: String,
    voice: String,
) -> Result<Vec<u8>, String> {
    sidecar_fetch_wav_bytes(&state, text, voice).await
}

/// One IPC hop: fetch WAV from sidecar and mix into the voice changer.
#[command]
async fn neural_sidecar_tts_speak_into_voice_changer<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
    vc: State<'_, VoiceChangerState>,
    text: String,
    voice: String,
) -> Result<(), String> {
    let wav = sidecar_fetch_wav_bytes(&state, text, voice).await?;
    push_aux_wav_bytes(&vc, wav)
}

#[command]
async fn check_neural_sidecar_tts<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, NeuralSidecarTtsState>,
) -> Result<bool, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/health", endpoint);
    match http_client().get(&url).send().await {
        Ok(r) => Ok(r.status().is_success()),
        Err(_) => Ok(false),
    }
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
