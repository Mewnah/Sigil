use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
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
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MeloSpeaker {
    pub id: String,
    pub name: String,
    pub language: String,
}

#[derive(Serialize, Debug)]
struct GenerateRequest {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    speaker_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    sample_rate: Option<u32>,
}

/// Set the MeloTTS API endpoint URL
#[command]
async fn set_melo_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, MeloTTSState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().map_err(|_| "Lock failed")? = endpoint;
    Ok(())
}

/// Get available speakers from MeloTTS
#[command]
async fn get_melo_speakers<R: Runtime>(_app: AppHandle<R>, state: State<'_, MeloTTSState>) -> Result<Vec<MeloSpeaker>, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/speakers", endpoint);

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Failed to connect to MeloTTS: {}. Is the service running?", e))?;

    if !response.status().is_success() {
        return Err(format!("Failed to get speakers: {}", response.status()));
    }

    let speakers: Vec<MeloSpeaker> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse speakers: {}", e))?;

    Ok(speakers)
}

/// Generate speech using MeloTTS and return audio bytes (WAV)
#[command]
async fn melo_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, MeloTTSState>,
    text: String,
    speaker_id: Option<String>,
    speed: Option<f32>,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/tts/generate", endpoint);

    let client = reqwest::Client::new();
    let request_body = GenerateRequest {
        text,
        speaker_id,
        speed,
        sample_rate: Some(22050),
    };

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Speech request failed: {}. Is MeloTTS running?", e))?;

    if !response.status().is_success() {
        return Err(format!("Speech synthesis failed: {}", response.status()));
    }

    let audio_bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read audio: {}", e))?;

    Ok(audio_bytes.to_vec())
}

/// Check if MeloTTS service is available
#[command]
async fn check_melo_availability<R: Runtime>(_app: AppHandle<R>, state: State<'_, MeloTTSState>) -> Result<bool, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/speakers", endpoint);

    match reqwest::get(&url).await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("melo_tts")
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
