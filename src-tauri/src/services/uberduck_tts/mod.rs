//! `uberduck-tts` plugin: Uberduck v1 API (`https://api.uberduck.ai/v1`, Bearer API key).
//!
//! - Voices: `GET /v1/voices` → `{ voices: [{ voicemodel_uuid, display_name, ... }] }`.
//! - Speech: `POST /v1/text-to-speech { text, voice, output_format: "wav" }` → `{ audio_url, ... }`;
//!   the clip is downloaded, decoded (WAV, or MP3 / FLAC / Ogg if the service sends those) and
//!   pushed into the unified TTS sink (`tts_sink`) on the blocking pool; no audio crosses IPC.
//!   - `speak_to_sink { data, utteranceId }` → [`SpeakReport`]: into the sink utterance the TS
//!     sentence pipeline began (route, device and volume are the utterance's); stops early with
//!     `"cancelled"` once stop / skip cancelled it, abandoning a synthesis request or download
//!     still in flight within ~25 ms.
//!   - `speak { data }` (legacy whole-line path): begins its own utterance on `device_name` with
//!     `volume` and resolves once the clip is queued (the sink plays utterances in order).
//!
//! Timeouts: voice list 20 s; synthesis and audio download each 30 s + 100 ms per character
//! (max 120 s). The retired pre-v1 endpoints and the `secret_key` credential are no longer used.

use std::future::Future;
use std::io::Cursor;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rodio::Source;
use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::tts_sink::{self, SinkError, SinkRoute, SinkTarget, SpeakReport, UtteranceSpec, UtteranceWriter};
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

/// `speak_to_sink` argument: no device or volume (they belong to the sink utterance).
#[derive(Deserialize, Debug)]
struct UberduckSinkRequest {
    auth: UberDuckAuth,
    text: String,
    voicemodel_uuid: String,
}

/// Pushed per sink call (seconds), so a cancel is noticed quickly.
const PUSH_SECONDS: usize = 1;
/// How often a request in flight checks whether its sink utterance was cancelled.
const CANCEL_POLL: Duration = Duration::from_millis(25);
/// Ids of the legacy `speak` utterances.
static NEXT_LEGACY_ID: AtomicU64 = AtomicU64::new(0);

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

/// Checked text and voice of a speech request (`None`: nothing to say).
fn checked_request<'a>(text: &'a str, voice: &'a str) -> Result<Option<(&'a str, &'a str)>, String> {
    let text = text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    if text.chars().count() > MAX_TEXT_CHARS {
        return Err(format!("Uberduck accepts at most {MAX_TEXT_CHARS} characters per request."));
    }
    let voice = voice.trim();
    if voice.is_empty() {
        return Err("Choose an Uberduck voice first.".into());
    }
    Ok(Some((text, voice)))
}

/// Synthesize `text` and download the clip (encoded audio).
async fn fetch_audio(auth: &UberDuckAuth, text: &str, voice: &str) -> Result<Vec<u8>, String> {
    let key = auth.bearer()?;
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
    Ok(audio)
}

/// Decode a downloaded clip (WAV / MP3 / FLAC / Ogg) into interleaved f32, its rate and channels.
fn decode_audio(bytes: Vec<u8>) -> Result<(Vec<f32>, u32, u16), String> {
    let decoder =
        rodio::Decoder::new(Cursor::new(bytes)).map_err(|e| format!("Uberduck audio cannot be decoded: {e}"))?;
    let rate = decoder.sample_rate();
    let channels = decoder.channels().max(1);
    let samples: Vec<f32> = decoder.map(|s| f32::from(s) / 32_768.0).collect();
    Ok((samples, rate, channels))
}

fn elapsed_ms(since: Instant) -> u32 {
    since.elapsed().as_millis().min(u128::from(u32::MAX)) as u32
}

/// Push a whole clip into the sink in ~1 s pieces (so a cancel stops it early).
fn push_clip(
    writer: &mut UtteranceWriter,
    samples: &[f32],
    rate: u32,
    channels: u16,
    started: Instant,
) -> Result<SpeakReport, SinkError> {
    let ch = usize::from(channels.max(1));
    let frames = samples.len() / ch;
    let step = (rate as usize * PUSH_SECONDS).max(1) * ch;
    let mut first_audio_ms = 0;
    for (i, part) in samples[..frames * ch].chunks(step).enumerate() {
        writer.push(part, rate, channels)?;
        if i == 0 {
            first_audio_ms = elapsed_ms(started);
        }
    }
    Ok(SpeakReport {
        sample_rate: rate,
        frames: frames as u64,
        first_audio_ms,
        total_ms: elapsed_ms(started),
    })
}

/// Await `fut`, giving up (`None`, dropping it: an HTTP request in flight is aborted) as soon as
/// `cancelled()` returns true; checked every [`CANCEL_POLL`].
async fn unless_cancelled<F: Future>(fut: F, cancelled: impl Fn() -> bool) -> Option<F::Output> {
    tokio::pin!(fut);
    loop {
        if cancelled() {
            return None;
        }
        tokio::select! {
            biased;
            out = &mut fut => return Some(out),
            _ = tokio::time::sleep(CANCEL_POLL) => {}
        }
    }
}

/// Speak one sentence into the sink utterance `utterance_id` (begun by the TS sentence pipeline).
/// Stop / skip cancel the utterance: the synthesis request or download in flight is abandoned at
/// once (`"cancelled"`), so the pipeline's next sentence does not wait for it.
#[command]
async fn speak_to_sink(data: UberduckSinkRequest, utterance_id: String) -> Result<SpeakReport, String> {
    let started = Instant::now();
    let mut writer = tts_sink::writer(&utterance_id)?;
    let Some((text, voice)) = checked_request(&data.text, &data.voicemodel_uuid)? else {
        return Ok(SpeakReport::default());
    };
    let Some(audio) = unless_cancelled(fetch_audio(&data.auth, text, voice), || writer.is_cancelled()).await else {
        return Err(SinkError::Cancelled.into());
    };
    let audio = audio?;
    if writer.is_cancelled() {
        return Err(SinkError::Cancelled.into());
    }
    off_runtime(move || {
        let (samples, rate, channels) = decode_audio(audio)?;
        push_clip(&mut writer, &samples, rate, channels, started).map_err(String::from)
    })
    .await
}

/// Legacy whole-line path: its own sink utterance on `device_name`.
#[command]
async fn speak(data: UberduckRequest) -> Result<(), String> {
    let Some((text, voice)) = checked_request(&data.text, &data.voicemodel_uuid)? else {
        return Ok(());
    };
    let audio = fetch_audio(&data.auth, text, voice).await?;
    let id = format!("uberduck-{}", NEXT_LEGACY_ID.fetch_add(1, Ordering::Relaxed));
    let volume = if data.volume.is_finite() { data.volume.clamp(0.0, 2.0) } else { 1.0 };
    tts_sink::begin(UtteranceSpec {
        id: id.clone(),
        target: SinkTarget {
            route: SinkRoute::Device,
            device: data.device_name,
            volume,
        },
    })?;
    let result = off_runtime({
        let id = id.clone();
        move || {
            let (samples, rate, channels) = decode_audio(audio)?;
            let mut writer = tts_sink::writer(&id)?;
            push_clip(&mut writer, &samples, rate, channels, Instant::now())?;
            tts_sink::end(&id).map_err(String::from)
        }
    })
    .await;
    if result.is_err() {
        tts_sink::cancel(Some(&id));
    }
    result
}

/// Run blocking work (decoding, pushes that wait for room) on the blocking thread pool.
async fn off_runtime<T: Send + 'static>(
    job: impl FnOnce() -> Result<T, String> + Send + 'static,
) -> Result<T, String> {
    tauri::async_runtime::spawn_blocking(job)
        .await
        .map_err(|e| format!("Uberduck playback task failed: {e}"))?
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("uberduck-tts")
        .invoke_handler(tauri::generate_handler![speak, speak_to_sink, get_voices])
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

    /// Review finding: a cancelled sentence waited for both HTTP requests (up to 2 x 54 s) before
    /// it settled, holding the pipeline's synthesis slot. The request is now abandoned at once.
    #[tokio::test(flavor = "current_thread")]
    async fn a_cancel_abandons_the_request_in_flight() {
        use std::sync::atomic::AtomicBool;
        use std::sync::Arc;
        use std::time::{Duration, Instant};
        let cancelled = Arc::new(AtomicBool::new(false));
        let flag = cancelled.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(60)).await;
            flag.store(true, Ordering::Release);
        });
        let started = Instant::now();
        let out = unless_cancelled(std::future::pending::<Result<Vec<u8>, String>>(), || cancelled.load(Ordering::Acquire)).await;
        assert!(out.is_none());
        assert!(started.elapsed() < Duration::from_secs(5), "{:?}", started.elapsed());
        // A request that finishes first is returned; one cancelled before it starts never runs.
        let done = unless_cancelled(async { 7 }, || false).await;
        assert_eq!(done, Some(7));
        let ran = AtomicBool::new(false);
        let never = unless_cancelled(async { ran.store(true, Ordering::Release) }, || true).await;
        assert_eq!(never, None);
        assert!(!ran.load(Ordering::Acquire));
    }

    #[test]
    fn audio_url_rejects_non_http_schemes() {
        assert!(audio_url("file:///C:/Windows/win.ini").is_err());
        assert!(audio_url("not a url").is_err());
    }

    #[test]
    fn requests_are_checked_before_any_network_call() {
        assert_eq!(checked_request("  ", "v"), Ok(None));
        assert_eq!(checked_request(" hi ", " v "), Ok(Some(("hi", "v"))));
        assert!(checked_request("hi", " ").unwrap_err().contains("voice"));
        let long = "a".repeat(MAX_TEXT_CHARS + 1);
        assert!(checked_request(&long, "v").unwrap_err().contains("characters"));
    }

    /// Downloaded WAV clips decode to f32 with their own rate and channel count (the sink converts
    /// them to the device / voice changer); garbage is a readable error.
    #[test]
    fn downloaded_clips_decode_to_pcm() {
        let mut cur = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec {
                channels: 2,
                sample_rate: 22_050,
                bits_per_sample: 16,
                sample_format: hound::SampleFormat::Int,
            };
            let mut w = hound::WavWriter::new(&mut cur, spec).unwrap();
            for s in [16_384i16, -16_384, 8_192, 0] {
                w.write_sample(s).unwrap();
            }
            w.finalize().unwrap();
        }
        let (samples, rate, channels) = decode_audio(cur.into_inner()).unwrap();
        assert_eq!((rate, channels), (22_050, 2));
        assert_eq!(samples, vec![0.5, -0.5, 0.25, 0.0]);
        assert!(decode_audio(b"<html>not audio</html>".to_vec()).unwrap_err().contains("cannot be decoded"));
    }
}
