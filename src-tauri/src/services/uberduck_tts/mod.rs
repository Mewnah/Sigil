use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::audio::{play_async, RpcAudioPlayAsync};
use crate::services::http_client::http_client;

const API_BASE: &str = "https://api.uberduck.ai/v1";

/// `secret_key` is only used by the retired pre-v1 API; v1 authenticates with a Bearer API key.
#[derive(Serialize, Deserialize, Debug)]
struct UberDuckAuth {
    api_key: String,
    #[serde(default)]
    secret_key: String,
}

/// Shape the frontend expects (kept from the pre-v1 API).
#[derive(Serialize, Deserialize, Debug)]
struct Voice {
    model_id: String,
    voicemodel_uuid: String,
    display_name: String,
}

#[derive(Deserialize, Debug)]
struct V1Voice {
    voicemodel_uuid: Option<String>,
    name: Option<String>,
    display_name: Option<String>,
}

#[derive(Deserialize, Debug)]
struct V1VoicesResponse {
    #[serde(default)]
    voices: Vec<V1Voice>,
}

#[derive(Serialize, Deserialize, Debug)]
struct UberduckRequest {
    auth: UberDuckAuth,
    text: String,
    device_name: String,
    voicemodel_uuid: String,
    volume: f32,
}

#[derive(Serialize, Debug)]
struct V1SpeechRequest<'a> {
    text: &'a str,
    voice: &'a str,
    output_format: &'a str,
}

#[derive(Deserialize, Debug)]
struct V1SpeechResponse {
    audio_url: String,
}

#[command]
async fn get_voices(auth: UberDuckAuth) -> Result<Vec<Voice>, String> {
    let resp = http_client()
        .get(format!("{API_BASE}/voices"))
        .query(&[("limit", "1000")])
        .bearer_auth(auth.api_key.trim())
        .send()
        .await
        .map_err(|e| format!("Unable to load voices: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Unable to load voices: HTTP {}", resp.status()));
    }
    let body: V1VoicesResponse = resp
        .json()
        .await
        .map_err(|e| format!("Unable to parse voices: {e}"))?;
    Ok(body
        .voices
        .into_iter()
        .filter_map(|v| {
            let id = v.voicemodel_uuid.or(v.name.clone())?;
            Some(Voice {
                model_id: String::new(),
                display_name: v.display_name.or(v.name).unwrap_or_else(|| id.clone()),
                voicemodel_uuid: id,
            })
        })
        .collect())
}

#[command]
async fn speak(data: UberduckRequest) -> Result<(), String> {
    let text = data.text.trim();
    if text.is_empty() {
        return Ok(());
    }
    let resp = http_client()
        .post(format!("{API_BASE}/text-to-speech"))
        .bearer_auth(data.auth.api_key.trim())
        .json(&V1SpeechRequest {
            text,
            voice: &data.voicemodel_uuid,
            output_format: "wav",
        })
        .send()
        .await
        .map_err(|e| format!("Request failed: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("Uberduck speech failed: HTTP {}", resp.status()));
    }
    let speech: V1SpeechResponse = resp
        .json()
        .await
        .map_err(|e| format!("Unable to parse speech response: {e}"))?;
    let audio = http_client()
        .get(&speech.audio_url)
        .send()
        .await
        .and_then(|r| r.error_for_status())
        .map_err(|e| format!("Unable to download audio: {e}"))?
        .bytes()
        .await
        .map_err(|e| format!("Unable to download audio: {e}"))?;
    play_async(RpcAudioPlayAsync {
        device_name: data.device_name,
        data: audio.to_vec(),
        volume: data.volume,
        rate: 1.0,
    })
    .await
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("uberduck-tts")
        .invoke_handler(tauri::generate_handler![speak, get_voices])
        .build()
}
