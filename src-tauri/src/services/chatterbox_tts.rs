use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://localhost:5555";

#[derive(Clone)]
pub struct ChatterboxTTSState {
    endpoint: Arc<Mutex<String>>,
}

impl ChatterboxTTSState {
    fn new() -> Self {
        Self {
            endpoint: Arc::new(Mutex::new(DEFAULT_ENDPOINT.to_string())),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ChatterboxVoice {
    pub id: String,
    pub name: String,
}

#[derive(Serialize, Debug)]
struct SpeechRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    exaggeration: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    cfg_weight: Option<f32>,
    response_format: String,
}

/// Set the Chatterbox API endpoint URL
#[command]
async fn set_chatterbox_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, ChatterboxTTSState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().map_err(|_| "Lock failed")? = endpoint;
    Ok(())
}

/// Get available voices from Chatterbox
#[command]
async fn get_chatterbox_voices<R: Runtime>(_app: AppHandle<R>, state: State<'_, ChatterboxTTSState>) -> Result<Vec<ChatterboxVoice>, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/audio/voices", endpoint);

    let response = reqwest::get(&url)
        .await
        .map_err(|e| format!("Failed to connect to Chatterbox: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Failed to get voices: {}", response.status()));
    }

    #[derive(Deserialize)]
    struct VoicesResponse {
        voices: Vec<ChatterboxVoice>,
    }

    let result: VoicesResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse voices: {}", e))?;

    Ok(result.voices)
}

/// Generate speech using Chatterbox TTS (with voice cloning)
#[command]
async fn chatterbox_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, ChatterboxTTSState>,
    text: String,
    voice: String,
    speed: Option<f32>,
    exaggeration: Option<f32>,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/audio/speech", endpoint);

    let client = reqwest::Client::new();
    let request_body = SpeechRequest {
        model: "chatterbox".to_string(),
        input: text,
        voice: if voice.is_empty() { "default".to_string() } else { voice },
        speed,
        exaggeration,
        cfg_weight: None,
        response_format: "mp3".to_string(),
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

/// Check if Chatterbox service is available
#[command]
async fn check_chatterbox_availability<R: Runtime>(_app: AppHandle<R>, state: State<'_, ChatterboxTTSState>) -> Result<bool, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/audio/voices", endpoint);

    match reqwest::get(&url).await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("chatterbox_tts")
        .setup(|app, _api| {
            app.manage(ChatterboxTTSState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_chatterbox_endpoint,
            get_chatterbox_voices,
            chatterbox_speak,
            check_chatterbox_availability
        ])
        .build()
}
