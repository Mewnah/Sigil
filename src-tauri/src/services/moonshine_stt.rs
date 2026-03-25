use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
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
}

#[derive(Serialize, Debug)]
struct TranscribeRequest {
    audio: String, // base64 encoded audio
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
}

#[derive(Deserialize, Debug)]
struct TranscribeResponse {
    text: String,
    #[serde(default)]
    #[allow(dead_code)]
    language: Option<String>,
}

/// Set the Moonshine API endpoint URL
#[command]
async fn set_moonshine_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, MoonshineSttState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().map_err(|_| "Lock failed")? = endpoint;
    Ok(())
}

/// Transcribe audio using Moonshine API
/// Audio should be base64-encoded WAV or raw PCM data
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

    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/transcribe", endpoint);

    let client = reqwest::Client::new();
    let request_body = TranscribeRequest {
        audio: audio_base64,
        language,
    };

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Transcription request failed: {}. Is Moonshine server running?", e))?;

    if !response.status().is_success() {
        return Err(format!("Transcription failed: {}", response.status()));
    }

    let result: TranscribeResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse transcription response: {}", e))?;

    Ok(result.text)
}

/// Check if Moonshine service is available
#[command]
async fn check_moonshine_availability<R: Runtime>(_app: AppHandle<R>, state: State<'_, MoonshineSttState>) -> Result<bool, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/health", endpoint);

    match reqwest::get(&url).await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("moonshine_stt")
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
