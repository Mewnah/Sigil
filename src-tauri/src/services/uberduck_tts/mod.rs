//! `uberduck-tts` plugin: Uberduck v1 API (`https://api.uberduck.ai/v1`, Bearer API key).
//!
//! - Voices: `GET /v1/voices` → `{ voices: [{ voicemodel_uuid, display_name, ... }] }`.
//! - Speech: `POST /v1/text-to-speech { text, voice, output_format: "wav" }` → `{ audio_url, ... }`;
//!   the clip is downloaded and played on the requested output device via the `audio` plugin, on
//!   the blocking pool (playback blocks until the clip ends).
//!
//! Timeouts: voice list 20 s; synthesis and audio download each 30 s + 100 ms per character
//! (max 120 s). The retired pre-v1 endpoints and the `secret_key` credential are no longer used.

use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::audio::{play_async, RpcAudioPlayAsync};
use crate::services::http_client::{
    content_type, describe_api_error, describe_request_error, ensure_audio_body, http_client,
    parse_named_list, read_body_capped, tts_timeout, LIST_TIMEOUT, MAX_AUDIO_BYTES,
    TTS_STALL_TIMEOUT,
};

const API_BASE: &str = "https://api.uberduck.ai/v1";
/// Uberduck's documented text limit.
const MAX_TEXT_CHARS: usize = 10_000;

/// `secret_key` is only used by the retired pre-v1 API; v1 authenticates with a Bearer API key.
#[derive(Serialize, Deserialize, Debug)]
struct UberDuckAuth {
    api_key: String,
    #[serde(default)]
    secret_key: String,
}

impl UberDuckAuth {
    fn bearer(&self) -> Result<&str, String> {
        let key = self.api_key.trim();
        if key.is_empty() {
            Err("Uberduck API key is missing.".into())
        } else {
            Ok(key)
        }
    }
}

/// Shape the frontend expects (kept from the pre-v1 API).
#[derive(Serialize, Deserialize, Debug, PartialEq, Eq)]
struct Voice {
    model_id: String,
    voicemodel_uuid: String,
    display_name: String,
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

/// Voices from `GET /v1/voices`. `voicemodel_uuid` is the id `text-to-speech` accepts as `voice`;
/// other id fields are fallbacks in case the listing changes shape.
fn parse_voices(v: &serde_json::Value) -> Vec<Voice> {
    parse_named_list(
        v,
        &["voices", "data", "items"],
        &["voicemodel_uuid", "voice_id", "id", "name"],
        &["display_name", "name", "voicemodel_uuid"],
    )
    .into_iter()
    .map(|e| Voice {
        model_id: String::new(),
        voicemodel_uuid: e.id,
        display_name: e.name,
    })
    .collect()
}

/// The `audio_url` from the API must be a plain http(s) URL (it is fetched without credentials).
fn audio_url(raw: &str) -> Result<reqwest::Url, String> {
    let url = reqwest::Url::parse(raw.trim())
        .map_err(|e| format!("Uberduck returned an invalid audio URL: {e}"))?;
    if !matches!(url.scheme(), "https" | "http") {
        return Err("Uberduck returned an unsupported audio URL.".into());
    }
    Ok(url)
}

#[command]
async fn get_voices(auth: UberDuckAuth) -> Result<Vec<Voice>, String> {
    let what = "Uberduck voice list";
    let resp = http_client()
        .get(format!("{API_BASE}/voices"))
        .query(&[("limit", "1000")])
        .bearer_auth(auth.bearer()?)
        .timeout(LIST_TIMEOUT)
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, LIST_TIMEOUT))?;
    if !resp.status().is_success() {
        return Err(describe_api_error(what, resp).await);
    }
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("{what}: unexpected response ({e})"))?;
    Ok(parse_voices(&body))
}

#[command]
async fn speak(data: UberduckRequest) -> Result<(), String> {
    let text = data.text.trim();
    if text.is_empty() {
        return Ok(());
    }
    if text.chars().count() > MAX_TEXT_CHARS {
        return Err(format!("Uberduck accepts at most {MAX_TEXT_CHARS} characters per request."));
    }
    let voice = data.voicemodel_uuid.trim();
    if voice.is_empty() {
        return Err("Choose an Uberduck voice first.".into());
    }
    let key = data.auth.bearer()?;
    let timeout = tts_timeout(text);

    let what = "Uberduck speech";
    let resp = http_client()
        .post(format!("{API_BASE}/text-to-speech"))
        .bearer_auth(key)
        .timeout(timeout)
        .json(&V1SpeechRequest {
            text,
            voice,
            output_format: "wav",
        })
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, timeout))?;
    if !resp.status().is_success() {
        return Err(describe_api_error(what, resp).await);
    }
    let speech: V1SpeechResponse = resp
        .json()
        .await
        .map_err(|e| format!("{what}: unexpected response ({e})"))?;

    let what = "Uberduck audio download";
    let resp = http_client()
        .get(audio_url(&speech.audio_url)?)
        .timeout(timeout)
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, timeout))?;
    if !resp.status().is_success() {
        return Err(describe_api_error(what, resp).await);
    }
    let ct = content_type(&resp);
    let audio = read_body_capped(what, resp, MAX_AUDIO_BYTES, TTS_STALL_TIMEOUT).await?;
    ensure_audio_body(what, ct.as_deref(), &audio)?;

    let clip = RpcAudioPlayAsync {
        device_name: data.device_name,
        data: audio,
        volume: if data.volume.is_finite() { data.volume.max(0.0) } else { 1.0 },
        rate: 1.0,
    };
    // `play_async` waits for the clip to end with rodio's blocking `sleep_until_end`, so it runs on
    // the blocking pool instead of holding a runtime worker for the whole clip.
    off_runtime(move || futures::executor::block_on(play_async(clip))).await
}

/// Run blocking work (audio playback) on the blocking thread pool and await its result.
async fn off_runtime<T: Send + 'static>(
    job: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(job)
        .await
        .map_err(|e| format!("Uberduck playback failed: {e}"))?
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("uberduck-tts")
        .invoke_handler(tauri::generate_handler![speak, get_voices])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn voices_use_voicemodel_uuid_as_id() {
        let v = serde_json::json!({"voices": [
            {"voicemodel_uuid": "polly_joanna", "display_name": "Joanna", "gender": "female"},
            {"voicemodel_uuid": "google_en_us", "name": "en-US"},
            {"display_name": "no id"}
        ]});
        assert_eq!(
            parse_voices(&v),
            vec![
                Voice { model_id: String::new(), voicemodel_uuid: "polly_joanna".into(), display_name: "Joanna".into() },
                Voice { model_id: String::new(), voicemodel_uuid: "google_en_us".into(), display_name: "en-US".into() },
            ]
        );
    }

    #[test]
    fn speech_request_matches_v1_contract() {
        let v = serde_json::to_value(V1SpeechRequest {
            text: "hi",
            voice: "polly_joanna",
            output_format: "wav",
        })
        .unwrap();
        assert_eq!(v, serde_json::json!({"text": "hi", "voice": "polly_joanna", "output_format": "wav"}));
        let r: V1SpeechResponse = serde_json::from_value(serde_json::json!({
            "audio_url": "https://static.uberduck.ai/output/x.wav", "duration_seconds": 1.5, "model": "polly_neural"
        }))
        .unwrap();
        assert!(audio_url(&r.audio_url).is_ok());
    }

    #[test]
    fn auth_requires_a_key_and_ignores_legacy_secret() {
        let a: UberDuckAuth = serde_json::from_value(serde_json::json!({"api_key": "  "})).unwrap();
        assert!(a.bearer().is_err());
        let a: UberDuckAuth = serde_json::from_value(serde_json::json!({"api_key": " k ", "secret_key": "old"})).unwrap();
        assert_eq!(a.bearer().unwrap(), "k");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn playback_does_not_hold_the_async_worker() {
        // Review finding: rodio's sleep_until_end blocked a runtime worker for the whole clip. On a
        // single-threaded runtime the timer below only fires on time if the job runs elsewhere.
        use std::time::{Duration, Instant};
        let started = Instant::now();
        let job = tokio::spawn(off_runtime(|| {
            std::thread::sleep(Duration::from_millis(400));
            Ok(7)
        }));
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(started.elapsed() < Duration::from_millis(300), "playback blocked the runtime worker");
        assert_eq!(job.await.unwrap().unwrap(), 7);

        let err = off_runtime(|| -> Result<(), String> { Err("no such device".into()) }).await.unwrap_err();
        assert_eq!(err, "no such device");
    }

    #[test]
    fn audio_url_rejects_non_http_schemes() {
        assert!(audio_url("file:///C:/Windows/win.ini").is_err());
        assert!(audio_url("not a url").is_err());
    }
}
