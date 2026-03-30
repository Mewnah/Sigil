use crate::services::http_client::http_client;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://localhost:8880";

#[derive(Clone)]
pub struct KokoroTTSState {
    endpoint: Arc<Mutex<String>>,
}

impl KokoroTTSState {
    fn new() -> Self {
        Self {
            endpoint: Arc::new(Mutex::new(DEFAULT_ENDPOINT.to_string())),
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KokoroVoice {
    pub name: String,
}

#[derive(Serialize, Debug)]
struct SpeechRequest {
    model: String,
    input: String,
    voice: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    speed: Option<f32>,
    response_format: String,
}

/// Set the Kokoro API endpoint URL
#[command]
async fn set_kokoro_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, KokoroTTSState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().map_err(|_| "Lock failed")? = endpoint;
    Ok(())
}

/// Get available voices from Kokoro-FastAPI
#[command]
async fn get_kokoro_voices<R: Runtime>(_app: AppHandle<R>, state: State<'_, KokoroTTSState>) -> Result<Vec<KokoroVoice>, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/audio/voices", endpoint);

    let response = http_client()
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("Failed to connect to Kokoro: {}. Is the service running?", e))?;

    if !response.status().is_success() {
        return Err(format!("Failed to get voices: {}", response.status()));
    }

    #[derive(Deserialize)]
    struct VoicesResponse {
        voices: Vec<KokoroVoice>,
    }

    let result: VoicesResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse voices: {}", e))?;

    Ok(result.voices)
}

/// Generate speech using Kokoro TTS and return audio bytes
#[command]
async fn kokoro_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, KokoroTTSState>,
    text: String,
    voice: String,
    speed: Option<f32>,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }

    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/audio/speech", endpoint);

    let request_body = SpeechRequest {
        model: "kokoro".to_string(),
        input: text,
        voice: if voice.is_empty() { "af_bella".to_string() } else { voice },
        speed,
        response_format: "mp3".to_string(),
    };

    let response = http_client()
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Speech request failed: {}. Is Kokoro running?", e))?;

    if !response.status().is_success() {
        return Err(format!("Speech synthesis failed: {}", response.status()));
    }

    let audio_bytes = response
        .bytes()
        .await
        .map_err(|e| format!("Failed to read audio: {}", e))?;

    Ok(audio_bytes.to_vec())
}

/// Check if Kokoro service is available
#[command]
async fn check_kokoro_availability<R: Runtime>(_app: AppHandle<R>, state: State<'_, KokoroTTSState>) -> Result<bool, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/v1/audio/voices", endpoint);

    match http_client().get(&url).send().await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("kokoro-tts")
        .setup(|app, _api| {
            app.manage(KokoroTTSState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_kokoro_endpoint,
            get_kokoro_voices,
            kokoro_speak,
            check_kokoro_availability
        ])
        .build()
}
