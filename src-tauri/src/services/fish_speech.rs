use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://localhost:8080";

#[derive(Clone)]
pub struct FishSpeechState {
    endpoint: Arc<Mutex<String>>,
}

impl FishSpeechState {
    fn new() -> Self {
        Self {
            endpoint: Arc::new(Mutex::new(DEFAULT_ENDPOINT.to_string())),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FishSpeechVoice {
    pub id: String,
    pub name: String,
    pub language: String,
}

#[derive(Serialize, Debug)]
struct SynthesizeRequest {
    text: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_audio: Option<String>, // base64
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    normalize: Option<bool>,
    format: String,
}

/// Set the Fish Speech API endpoint URL
#[command]
async fn set_fish_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, FishSpeechState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().map_err(|_| "Lock failed")? = endpoint;
    Ok(())
}

/// Get available models/voices from Fish Speech
#[command]
async fn get_fish_voices<R: Runtime>(_app: AppHandle<R>, state: State<'_, FishSpeechState>) -> Result<Vec<FishSpeechVoice>, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/models", endpoint);

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Failed to connect to Fish Speech: {}", e))?;

    if !response.status().is_success() {
        // Return default voice if models endpoint not available
        return Ok(vec![FishSpeechVoice {
            id: "default".to_string(),
            name: "Default".to_string(),
            language: "en".to_string(),
        }]);
    }

    let voices: Vec<FishSpeechVoice> = response.json().await.unwrap_or_else(|_| vec![]);

    Ok(voices)
}

/// Synthesize speech using Fish Speech
#[command]
async fn fish_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, FishSpeechState>,
    text: String,
    reference_id: Option<String>,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/tts", endpoint);

    let client = reqwest::Client::new();
    let request_body = SynthesizeRequest {
        text,
        reference_id,
        reference_audio: None,
        reference_text: None,
        normalize: Some(true),
        format: "wav".to_string(),
    };

    let response = client
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Speech request failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Speech synthesis failed: {}", response.status()));
    }

    let audio_bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read audio: {}", e))?;

    Ok(audio_bytes.to_vec())
}

/// Check if Fish Speech service is available
#[command]
async fn check_fish_availability<R: Runtime>(_app: AppHandle<R>, state: State<'_, FishSpeechState>) -> Result<bool, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/models", endpoint);

    match reqwest::get(&url).await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("fish_speech")
        .setup(|app, _api| {
            app.manage(FishSpeechState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_fish_endpoint,
            get_fish_voices,
            fish_speak,
            check_fish_availability
        ])
        .build()
}
