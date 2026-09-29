use crate::services::http_client::http_client;
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

/// Reference voices saved on the fish-speech server (`GET /v1/references/list`).
#[derive(Deserialize, Debug, Default)]
struct ListReferencesResponse {
    #[serde(default)]
    reference_ids: Vec<String>,
}

/// Get available voices from Fish Speech: the built-in default plus saved reference voices.
#[command]
async fn get_fish_voices<R: Runtime>(_app: AppHandle<R>, state: State<'_, FishSpeechState>) -> Result<Vec<FishSpeechVoice>, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/references/list", endpoint.trim_end_matches('/'));

    let response = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Fish Speech: {}", e))?;

    let mut voices = vec![FishSpeechVoice {
        id: "default".to_string(),
        name: "Default".to_string(),
        language: "auto".to_string(),
    }];
    if response.status().is_success() {
        let list: ListReferencesResponse = response.json().await.unwrap_or_default();
        voices.extend(list.reference_ids.into_iter().map(|id| FishSpeechVoice {
            name: id.clone(),
            id,
            language: "auto".to_string(),
        }));
    }
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
    let url = format!("{}/v1/tts", endpoint.trim_end_matches('/'));

    let request_body = SynthesizeRequest {
        text,
        // "default" is the synthetic entry from get_fish_voices, not a saved reference.
        reference_id: reference_id.filter(|id| !id.is_empty() && id != "default"),
        reference_audio: None,
        reference_text: None,
        normalize: Some(true),
        format: "wav".to_string(),
    };

    let response = http_client()
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
    let url = format!("{}/v1/health", endpoint.trim_end_matches('/'));

    match http_client().get(&url).send().await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("fish-speech")
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
