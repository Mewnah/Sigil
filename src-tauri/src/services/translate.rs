use crate::services::http_client::http_client;
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://localhost:5000";

#[derive(Clone)]
pub struct TranslateState {
    endpoint: Arc<Mutex<String>>,
    /// Optional LibreTranslate API key (sent in JSON body as `api_key` and as `api_key` query on GET).
    api_key: Arc<Mutex<String>>,
}

impl TranslateState {
    fn new() -> Self {
        Self {
            endpoint: Arc::new(Mutex::new(DEFAULT_ENDPOINT.to_string())),
            api_key: Arc::new(Mutex::new(String::new())),
        }
    }

    /// Value for optional LibreTranslate `api_key` JSON / query field.
    fn api_key_for_request(&self) -> Result<Option<String>, String> {
        let k = self.api_key.lock().map_err(|_| "Lock failed")?.clone();
        let t = k.trim();
        if t.is_empty() {
            Ok(None)
        } else {
            Ok(Some(t.to_string()))
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Language {
    pub code: String,
    pub name: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct TranslateRequest {
    q: String,
    source: String,
    target: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct TranslateResponse {
    #[serde(rename = "translatedText")]
    translated_text: String,
}

#[derive(Serialize, Deserialize, Debug)]
struct DetectRequest {
    q: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct DetectResponse {
    confidence: f64,
    language: String,
}

/// Set the LibreTranslate API endpoint URL
#[command]
async fn set_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, TranslateState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().map_err(|_| "Lock failed")? = endpoint;
    Ok(())
}

/// Set optional LibreTranslate API key (empty string clears).
#[command]
async fn set_api_key<R: Runtime>(_app: AppHandle<R>, state: State<'_, TranslateState>, api_key: String) -> Result<(), String> {
    *state.api_key.lock().map_err(|_| "Lock failed")? = api_key;
    Ok(())
}

/// Get available languages from LibreTranslate
#[command]
async fn get_languages<R: Runtime>(_app: AppHandle<R>, state: State<'_, TranslateState>) -> Result<Vec<Language>, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/languages", endpoint.trim_end_matches('/'));
    let api_key = state.api_key_for_request()?;

    let mut req = http_client().get(&url);
    if let Some(ref k) = api_key {
        req = req.query(&[("api_key", k.as_str())]);
    }

    let response = req
        .send()
        .await
        .map_err(|e| format!("Failed to connect to LibreTranslate: {}. Is the service running?", e))?;

    let languages: Vec<Language> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse language list: {}", e))?;

    Ok(languages)
}

/// Translate text from source language to target language
#[command]
async fn translate<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, TranslateState>,
    text: String,
    source: String,
    target: String,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Ok(String::new());
    }

    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/translate", endpoint.trim_end_matches('/'));
    let api_key = state.api_key_for_request()?;

    let request_body = TranslateRequest {
        q: text,
        source,
        target,
        api_key,
    };

    let response = http_client()
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Translation request failed: {}. Is LibreTranslate running?", e))?;

    if !response.status().is_success() {
        return Err(format!("Translation failed with status: {}", response.status()));
    }

    let result: TranslateResponse = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse translation response: {}", e))?;

    Ok(result.translated_text)
}

/// Detect the language of the input text
#[command]
async fn detect_language<R: Runtime>(_app: AppHandle<R>, state: State<'_, TranslateState>, text: String) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err("Cannot detect language of empty text".to_string());
    }

    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/detect", endpoint.trim_end_matches('/'));
    let api_key = state.api_key_for_request()?;

    let request_body = DetectRequest {
        q: text,
        api_key,
    };

    let response = http_client()
        .post(&url)
        .json(&request_body)
        .send()
        .await
        .map_err(|e| format!("Language detection failed: {}", e))?;

    if !response.status().is_success() {
        return Err(format!("Detection failed with status: {}", response.status()));
    }

    let results: Vec<DetectResponse> = response
        .json()
        .await
        .map_err(|e| format!("Failed to parse detection response: {}", e))?;

    results
        .first()
        .map(|r| r.language.clone())
        .ok_or_else(|| "No language detected".to_string())
}

/// Check if LibreTranslate service is available
#[command]
async fn check_availability<R: Runtime>(_app: AppHandle<R>, state: State<'_, TranslateState>) -> Result<bool, String> {
    let endpoint = state.endpoint.lock().map_err(|_| "Lock failed")?.clone();
    let url = format!("{}/languages", endpoint.trim_end_matches('/'));
    let api_key = state.api_key_for_request()?;

    let mut req = http_client().get(&url);
    if let Some(ref k) = api_key {
        req = req.query(&[("api_key", k.as_str())]);
    }

    match req.send().await {
        Ok(response) => Ok(response.status().is_success()),
        Err(_) => Ok(false),
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("translate")
        .setup(|app, _api| {
            app.manage(TranslateState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_endpoint,
            set_api_key,
            get_languages,
            translate,
            detect_language,
            check_availability
        ])
        .build()
}
