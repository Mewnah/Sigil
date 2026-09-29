//! `fish-speech` plugin: client for a self-hosted fish-speech server (`tools/api_server.py`, Fish
//! S1 / S2 Pro) and for Fish Audio's hosted API (`https://api.fish.audio`, including the free
//! `s2.1-pro-free` model).
//!
//! # Endpoints
//! - Health: `GET /v1/health`. The hosted API has none, so when it answers 404/405 and an API key is
//!   set, the key is validated with `GET /model?self=true&page_size=1` instead.
//! - Voices: `GET /v1/references/list` (`{ reference_ids: [...] }`) on self-hosted servers; on the
//!   hosted API the user's own voice models (`GET /model?self=true`, `items[]._id` / `title`). A
//!   synthetic `default` voice is always first and is never sent as `reference_id`. When neither
//!   route exists (an older self-hosted server run with `--api-key`), only `default` is listed.
//! - Synthesis: `POST /v1/tts` `{ text, reference_id?, normalize, format: "wav" }`, with the local
//!   budget ([`crate::services::http_client::tts_post`]) for servers on this PC or the LAN.
//!
//! # Auth and model
//! Every command takes optional per-call `endpoint`, `apiKey` and (for synthesis) `model`:
//! the key goes in `Authorization: Bearer …` (both the hosted API and `api_server.py --api-key`
//! use it) and the model in the `model` header (`s1`, `s2-pro`, `s2.1-pro`, `s2.1-pro-free`, …).
//! Blank values are not sent. A blank `endpoint` uses the one stored by `set_fish_endpoint`.
//!
//! # Streamed WAV
//! The audio body is read chunk by chunk with a stall timeout ([`TTS_STALL_TIMEOUT`]) and a size cap,
//! so the hosted API's chunked stream is consumed as it arrives and a server that stops mid-stream
//! fails fast. A WAV written for streaming declares placeholder sizes (0 or `0xFFFFFFFF`); those are
//! repaired ([`repair_streamed_wav`]) so the webview and `hound` decode the whole clip. Self-hosted
//! servers are asked for a complete WAV (`streaming` is not requested) because their streaming mode
//! is not guaranteed to send a WAV header at all.

use crate::services::http_client::{
    api_url, content_type, describe_api_error, describe_request_error, ensure_audio_body,
    http_client, is_auth_failure, non_empty_key, parse_named_list, pick_endpoint, probe,
    read_body_capped, tts_post, NamedEntry, LIST_TIMEOUT, MAX_AUDIO_BYTES, TTS_STALL_TIMEOUT,
};
use reqwest::header::HeaderValue;
use reqwest::{Client, RequestBuilder, StatusCode, Url};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, PoisonError};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://localhost:8080";
/// Synthetic list entry for "no reference voice"; never sent to the server.
const DEFAULT_VOICE_ID: &str = "default";
/// Hosted API: list of voice models (not under `/v1`).
const HOSTED_MODELS_PATH: &str = "/model";

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

    fn endpoint_for(&self, call: Option<&str>) -> String {
        let stored = self.endpoint.lock().unwrap_or_else(PoisonError::into_inner);
        pick_endpoint(call, &stored)
    }
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct FishSpeechVoice {
    pub id: String,
    pub name: String,
    pub language: String,
}

impl FishSpeechVoice {
    fn default_voice() -> Self {
        Self {
            id: DEFAULT_VOICE_ID.to_string(),
            name: "Default".to_string(),
            language: "auto".to_string(),
        }
    }
}

#[derive(Serialize, Debug)]
struct TtsRequest<'a> {
    text: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    reference_id: Option<&'a str>,
    normalize: bool,
    format: &'static str,
}

/// The `model` header value, or `None` when blank. Model ids are short ASCII tokens
/// (`s2.1-pro-free`); anything else is rejected instead of producing an invalid header.
fn model_header(model: Option<&str>) -> Result<Option<HeaderValue>, String> {
    let Some(m) = model.map(str::trim).filter(|m| !m.is_empty()) else {
        return Ok(None);
    };
    let valid = m.len() <= 128
        && m
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '-' | '_' | ':' | '/'));
    if !valid {
        return Err(format!(
            "Invalid Fish Speech model '{m}': use a model id such as s2.1-pro-free."
        ));
    }
    HeaderValue::from_str(m)
        .map(Some)
        .map_err(|e| format!("Invalid Fish Speech model '{m}': {e}"))
}

fn with_auth(req: RequestBuilder, api_key: Option<&str>) -> RequestBuilder {
    match api_key {
        Some(key) => req.bearer_auth(key),
        None => req,
    }
}

fn hosted_models_url(base: &str, page_size: u32) -> Result<Url, String> {
    let mut url = api_url(base, HOSTED_MODELS_PATH)?;
    url.query_pairs_mut()
        .append_pair("self", "true")
        .append_pair("page_size", &page_size.to_string());
    Ok(url)
}

fn to_voices(entries: Vec<NamedEntry>) -> impl Iterator<Item = FishSpeechVoice> {
    entries
        .into_iter()
        .filter(|e| e.id != DEFAULT_VOICE_ID)
        .map(|e| FishSpeechVoice {
            id: e.id,
            name: e.name,
            language: e.language.unwrap_or_else(|| "auto".to_string()),
        })
}

/// Saved references from `GET /v1/references/list` (`{"success":true,"reference_ids":[...]}`),
/// tolerating a bare array or object items.
fn parse_reference_list(v: &serde_json::Value) -> Vec<FishSpeechVoice> {
    to_voices(parse_named_list(
        v,
        &["reference_ids", "references", "voices", "items"],
        &["id", "reference_id", "name"],
        &["name", "title", "id"],
    ))
    .collect()
}

/// Voice models from the hosted `GET /model` (`{"total":n,"items":[{"_id","title","languages"}]}`).
fn parse_hosted_models(v: &serde_json::Value) -> Vec<FishSpeechVoice> {
    to_voices(parse_named_list(v, &["items"], &["_id", "id"], &["title", "name"])).collect()
}

/// Fix the size fields of a WAV written for streaming, where the header was sent before the length
/// was known (data size 0 or `0xFFFFFFFF`, or larger than what arrived). A partial last sample frame
/// is dropped. Returns whether anything changed; complete WAVs and non-WAV bodies are untouched.
pub(crate) fn repair_streamed_wav(buf: &mut Vec<u8>) -> bool {
    fn read_u32(b: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }
    if buf.len() < 12 || buf[0..4] != *b"RIFF" || buf[8..12] != *b"WAVE" {
        return false;
    }
    let mut pos = 12usize;
    let mut block_align = 1usize;
    let mut changed = false;
    while pos + 8 <= buf.len() {
        let id = [buf[pos], buf[pos + 1], buf[pos + 2], buf[pos + 3]];
        let size = read_u32(buf, pos + 4);
        let body = pos + 8;
        if id == *b"fmt " && body + 14 <= buf.len() {
            block_align = usize::from(u16::from_le_bytes([buf[body + 12], buf[body + 13]])).max(1);
        }
        if id == *b"data" {
            let available = buf.len() - body;
            if size == 0 || size == u32::MAX || size as usize > available {
                // Whole sample frames only; bodies are capped far below 4 GiB (MAX_AUDIO_BYTES).
                let whole = (available - available % block_align).min(u32::MAX as usize);
                if whole != size as usize || buf.len() != body + whole {
                    buf.truncate(body + whole);
                    buf[pos + 4..pos + 8].copy_from_slice(&(whole as u32).to_le_bytes());
                    changed = true;
                }
            }
            break;
        }
        let next = body as u64 + u64::from(size) + u64::from(size & 1);
        if next > buf.len() as u64 {
            break; // malformed chunk before `data`: leave the body for the decoder to judge
        }
        pos = next as usize;
    }
    let riff = read_u32(buf, 4);
    let actual = (buf.len() - 8).min(u32::MAX as usize) as u32;
    if (changed || riff == 0 || riff == u32::MAX || riff > actual) && riff != actual {
        buf[4..8].copy_from_slice(&actual.to_le_bytes());
        changed = true;
    }
    changed
}

/// Set the Fish Speech API endpoint URL (the default for calls without `endpoint`).
#[command]
async fn set_fish_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, FishSpeechState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().unwrap_or_else(PoisonError::into_inner) = endpoint.trim().to_string();
    Ok(())
}

/// Whether `status` means "this server has no such route" (the endpoint belongs to another kind of
/// server, e.g. a self-hosted fish-speech without the hosted API's `/model`).
fn is_missing_route(status: StatusCode) -> bool {
    matches!(status, StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED)
}

/// The hosted API's own voice models (`GET /model?self=true`). Only a rejected key or a real hosted
/// failure (5xx, 429, unreachable) is an error. A missing route or a non-JSON answer means the
/// endpoint is not the hosted API (an older self-hosted server run with `--api-key`), so no models
/// are added and the caller still lists the default voice.
async fn fetch_hosted_models(client: &Client, base: &str, api_key: &str) -> Result<Vec<FishSpeechVoice>, String> {
    let what = "Fish Audio voice list";
    let response = client
        .get(hosted_models_url(base, 100)?)
        .bearer_auth(api_key)
        .timeout(LIST_TIMEOUT)
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, LIST_TIMEOUT))?;
    let status = response.status();
    if status.is_success() {
        return match response.json::<serde_json::Value>().await {
            Ok(v) => Ok(parse_hosted_models(&v)),
            Err(e) => {
                tracing::info!(target: "sigil::http", "{what}: not a voice-model list ({e}); listing only the default voice");
                Ok(Vec::new())
            }
        };
    }
    if is_missing_route(status) {
        tracing::info!(target: "sigil::http", "{what}: HTTP {status}; not the hosted API, listing only the default voice");
        return Ok(Vec::new());
    }
    Err(describe_api_error(what, response).await)
}

/// The voice list for `base`: the built-in default plus saved references (self-hosted) or, when the
/// server has no references API and a key is set, the account's hosted voice models.
async fn list_voices(client: &Client, base: &str, key: Option<&str>) -> Result<Vec<FishSpeechVoice>, String> {
    let url = api_url(base, "/v1/references/list")?;
    let what = "Fish Speech voice list";

    let response = with_auth(client.get(url), key)
        .timeout(LIST_TIMEOUT)
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, LIST_TIMEOUT))?;

    let mut voices = vec![FishSpeechVoice::default_voice()];
    let status = response.status();
    if status.is_success() {
        let v: serde_json::Value = response
            .json()
            .await
            .map_err(|e| format!("{what}: unexpected response ({e})"))?;
        voices.extend(parse_reference_list(&v));
    } else if is_auth_failure(status) {
        return Err(describe_api_error(what, response).await);
    } else if let Some(k) = key.filter(|_| is_missing_route(status)) {
        // Hosted API: no references endpoint, list the account's voice models instead.
        voices.extend(fetch_hosted_models(client, base, k).await?);
    } else {
        // Older servers without the references API: only the default voice.
        tracing::info!(target: "sigil::http", "{what}: HTTP {status}; listing only the default voice");
    }
    Ok(voices)
}

/// Available voices: the built-in default plus saved references (self-hosted) or the account's own
/// voice models (hosted API with a key).
#[command]
async fn get_fish_voices<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, FishSpeechState>,
    endpoint: Option<String>,
    api_key: Option<String>,
) -> Result<Vec<FishSpeechVoice>, String> {
    let base = state.endpoint_for(endpoint.as_deref());
    let key = non_empty_key(api_key.as_deref());
    list_voices(http_client(), &base, key.as_deref()).await
}

/// Synthesize speech; returns WAV bytes.
#[command]
async fn fish_speak<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, FishSpeechState>,
    text: String,
    reference_id: Option<String>,
    endpoint: Option<String>,
    api_key: Option<String>,
    model: Option<String>,
) -> Result<Vec<u8>, String> {
    if text.trim().is_empty() {
        return Ok(Vec::new());
    }
    let base = state.endpoint_for(endpoint.as_deref());
    let key = non_empty_key(api_key.as_deref());
    let model = model_header(model.as_deref())?;
    let url = api_url(&base, "/v1/tts")?;
    let what = "Fish Speech synthesis";

    let body = TtsRequest {
        text: &text,
        // "default" is the synthetic entry from get_fish_voices, not a saved reference.
        reference_id: reference_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty() && *id != DEFAULT_VOICE_ID),
        normalize: true,
        format: "wav",
    };
    // Self-hosted servers on this PC / LAN (often CPU, non-streaming) get the longer local budget.
    let (req, timeout) = tts_post(url, &text);
    let mut req = with_auth(req, key.as_deref()).json(&body);
    if let Some(m) = model {
        req = req.header("model", m);
    }
    let response = req
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, timeout))?;
    if !response.status().is_success() {
        return Err(describe_api_error(what, response).await);
    }
    let ct = content_type(&response);
    let mut audio = read_body_capped(what, response, MAX_AUDIO_BYTES, TTS_STALL_TIMEOUT).await?;
    ensure_audio_body(what, ct.as_deref(), &audio)?;
    repair_streamed_wav(&mut audio);
    Ok(audio)
}

/// Whether the server answers. `Err` (instead of `false`) when it answers but rejects the API key,
/// so the UI can say why.
#[command]
async fn check_fish_availability<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, FishSpeechState>,
    endpoint: Option<String>,
    api_key: Option<String>,
) -> Result<bool, String> {
    let base = state.endpoint_for(endpoint.as_deref());
    let key = non_empty_key(api_key.as_deref());
    let Ok(url) = api_url(&base, "/v1/health") else {
        return Ok(false);
    };
    let what = "Fish Speech availability check";
    let key_rejected = |status: StatusCode| {
        format!("Fish Speech rejected the API key (HTTP {status}). Check the key in the Fish Speech settings.")
    };

    match probe(what, with_auth(http_client().get(url), key.as_deref())).await {
        Err(_) => Ok(false),
        Ok(status) if status.is_success() => Ok(true),
        Ok(status) if is_auth_failure(status) => Err(key_rejected(status)),
        Ok(status) if key.is_some() && is_missing_route(status) => {
            // Hosted API: no health route; validate the key against the voice-model list.
            let Ok(url) = hosted_models_url(&base, 1) else {
                return Ok(false);
            };
            match probe(what, with_auth(http_client().get(url), key.as_deref())).await {
                Ok(s) if s.is_success() => Ok(true),
                Ok(s) if is_auth_failure(s) => Err(key_rejected(s)),
                _ => Ok(false),
            }
        }
        Ok(status) => {
            tracing::info!(target: "sigil::http", "{what}: HTTP {status}");
            Ok(false)
        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
    use std::io::Cursor;

    fn wav_i16(samples: &[i16], sample_rate: u32) -> Vec<u8> {
        let spec = WavSpec {
            channels: 1,
            sample_rate,
            bits_per_sample: 16,
            sample_format: SampleFormat::Int,
        };
        let mut cur = Cursor::new(Vec::new());
        {
            let mut w = WavWriter::new(&mut cur, spec).unwrap();
            for s in samples {
                w.write_sample(*s).unwrap();
            }
            w.finalize().unwrap();
        }
        cur.into_inner()
    }

    fn data_chunk_pos(wav: &[u8]) -> usize {
        wav.windows(4).position(|w| w == b"data").unwrap()
    }

    fn decode(wav: &[u8]) -> Vec<i16> {
        WavReader::new(Cursor::new(wav.to_vec()))
            .unwrap()
            .samples::<i16>()
            .collect::<Result<_, _>>()
            .unwrap()
    }

    fn with_placeholder_sizes(mut wav: Vec<u8>, placeholder: u32) -> Vec<u8> {
        let d = data_chunk_pos(&wav);
        wav[d + 4..d + 8].copy_from_slice(&placeholder.to_le_bytes());
        wav[4..8].copy_from_slice(&placeholder.max(36).to_le_bytes());
        wav
    }

    #[test]
    fn repairs_zero_size_streamed_header() {
        let samples: Vec<i16> = (0..4410).map(|i| (i % 300) as i16 - 150).collect();
        let mut wav = with_placeholder_sizes(wav_i16(&samples, 44_100), 0);
        assert!(repair_streamed_wav(&mut wav));
        assert_eq!(decode(&wav), samples);
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()) as usize, wav.len() - 8);
    }

    #[test]
    fn repairs_max_size_streamed_header() {
        let samples = vec![1i16, -2, 3, -4];
        let mut wav = with_placeholder_sizes(wav_i16(&samples, 24_000), u32::MAX);
        assert!(repair_streamed_wav(&mut wav));
        assert_eq!(decode(&wav), samples);
    }

    #[test]
    fn drops_partial_last_frame_of_truncated_stream() {
        let samples = vec![10i16, 20, 30];
        let mut wav = with_placeholder_sizes(wav_i16(&samples, 16_000), 0);
        wav.push(0x7f); // half of a 16-bit sample
        assert!(repair_streamed_wav(&mut wav));
        assert_eq!(decode(&wav), samples);
    }

    #[test]
    fn complete_wav_and_non_wav_are_untouched() {
        let original = wav_i16(&[5, 6, 7, 8], 22_050);
        let mut wav = original.clone();
        assert!(!repair_streamed_wav(&mut wav));
        assert_eq!(wav, original);

        let empty = wav_i16(&[], 44_100);
        let mut still_empty = empty.clone();
        assert!(!repair_streamed_wav(&mut still_empty), "a genuinely empty WAV is valid");
        assert_eq!(still_empty, empty);

        let mut mp3 = b"ID3\x04\x00\x00\x00\x00\x00\x00".to_vec();
        assert!(!repair_streamed_wav(&mut mp3));
        let mut tiny = b"RIFF".to_vec();
        assert!(!repair_streamed_wav(&mut tiny));
    }

    /// `wav` with a `LIST` chunk of `payload` (plus its RIFF pad byte when odd) before `data`.
    fn with_list_chunk(wav: &[u8], payload: &[u8]) -> Vec<u8> {
        let d = data_chunk_pos(wav);
        let mut out = wav[..d].to_vec();
        out.extend_from_slice(b"LIST");
        out.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        out.extend_from_slice(payload);
        if payload.len() % 2 == 1 {
            out.push(0);
        }
        out.extend_from_slice(&wav[d..]);
        out
    }

    #[test]
    fn skips_chunks_before_data() {
        let samples = vec![100i16, 200, 300, 400];
        let wav = wav_i16(&samples, 44_100);

        let mut even = with_placeholder_sizes(with_list_chunk(&wav, b"abcd"), 0);
        assert!(repair_streamed_wav(&mut even));
        assert_eq!(decode(&even), samples);

        // Odd-sized chunk: the pad byte must be skipped to find `data` (checked on the raw bytes,
        // since hound itself ignores RIFF padding).
        let mut odd = with_placeholder_sizes(with_list_chunk(&wav, b"abc"), 0);
        assert!(repair_streamed_wav(&mut odd));
        let d = data_chunk_pos(&odd);
        assert_eq!(u32::from_le_bytes(odd[d + 4..d + 8].try_into().unwrap()), 8);
    }

    #[test]
    fn model_header_validation() {
        assert!(model_header(None).unwrap().is_none());
        assert!(model_header(Some("  ")).unwrap().is_none());
        assert_eq!(model_header(Some(" s2.1-pro-free ")).unwrap().unwrap(), "s2.1-pro-free");
        assert!(model_header(Some("s2 pro")).is_err());
        assert!(model_header(Some("s2\r\nX-Evil: 1")).is_err());
        assert!(model_header(Some("モデル")).is_err());
    }

    #[test]
    fn parses_official_reference_list() {
        let v = serde_json::json!({"success": true, "reference_ids": ["alice", "bob", "default"], "message": "ok"});
        let voices = parse_reference_list(&v);
        assert_eq!(voices.iter().map(|v| v.id.as_str()).collect::<Vec<_>>(), ["alice", "bob"]);
        assert_eq!(voices[0].language, "auto");
        assert!(parse_reference_list(&serde_json::json!({"success": false})).is_empty());
    }

    #[test]
    fn parses_hosted_model_list() {
        let v = serde_json::json!({
            "total": 2,
            "items": [
                {"_id": "8ef4a238714b45718ce04243307c57a7", "title": "My voice", "languages": ["ja", "en"], "state": "trained"},
                {"_id": "abc", "title": ""}
            ]
        });
        let voices = parse_hosted_models(&v);
        assert_eq!(voices.len(), 2);
        assert_eq!(voices[0].name, "My voice");
        assert_eq!(voices[0].language, "ja");
        assert_eq!(voices[1].name, "abc", "blank titles fall back to the id");
    }

    #[test]
    fn hosted_models_url_has_no_secret() {
        let url = hosted_models_url("https://api.fish.audio/", 1).unwrap();
        assert_eq!(url.as_str(), "https://api.fish.audio/model?self=true&page_size=1");
    }

    // -----------------------------------------------------------------------------------------
    // Voice-list routing against a loopback mock server
    // -----------------------------------------------------------------------------------------

    use std::time::Duration;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A request the mock server saw: path (no query) and whether it carried the Bearer key.
    #[derive(Debug, Clone, PartialEq)]
    struct Seen {
        path: String,
        bearer: bool,
    }

    /// Serve `routes` (path → (status line, JSON body)) on 127.0.0.1 until the test ends; unknown
    /// paths get 404. Returns the base URL and the log of requests.
    async fn mock_server(routes: Vec<(&'static str, &'static str, &'static str)>) -> (String, Arc<Mutex<Vec<Seen>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = seen.clone();
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    if sock.read(&mut byte).await.unwrap_or(0) == 0 {
                        break;
                    }
                    head.push(byte[0]);
                }
                let text = String::from_utf8_lossy(&head).to_string();
                let target = text.split_whitespace().nth(1).unwrap_or("/").to_string();
                let path = target.split('?').next().unwrap_or("/").to_string();
                let bearer = text.to_ascii_lowercase().contains("authorization: bearer test-key");
                log.lock().unwrap().push(Seen { path: path.clone(), bearer });
                let (status, body) = routes
                    .iter()
                    .find(|(p, _, _)| *p == path)
                    .map(|(_, s, b)| (*s, *b))
                    .unwrap_or(("404 Not Found", r#"{"detail":"Not Found"}"#));
                let resp = format!(
                    "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = sock.write_all(resp.as_bytes()).await;
                let _ = sock.shutdown().await;
            }
        });
        (base, seen)
    }

    fn client() -> Client {
        Client::builder().no_proxy().timeout(Duration::from_secs(5)).build().unwrap()
    }

    fn ids(voices: &[FishSpeechVoice]) -> Vec<&str> {
        voices.iter().map(|v| v.id.as_str()).collect()
    }

    fn paths(seen: &Arc<Mutex<Vec<Seen>>>) -> Vec<String> {
        seen.lock().unwrap().iter().map(|s| s.path.clone()).collect()
    }

    #[tokio::test]
    async fn old_server_with_key_and_no_list_routes_still_lists_default() {
        // Review finding: an older api_server run with --api-key has neither /v1/references/list nor
        // the hosted /model. That used to fail the whole list with "HTTP 404".
        let (base, seen) = mock_server(vec![]).await;
        let voices = list_voices(&client(), &base, Some("test-key")).await.unwrap();
        assert_eq!(ids(&voices), ["default"]);
        assert_eq!(paths(&seen), ["/v1/references/list", "/model"]);
        assert!(seen.lock().unwrap().iter().all(|s| s.bearer), "the key goes in the header");

        // 405 on both routes behaves the same.
        let (base, _) = mock_server(vec![
            ("/v1/references/list", "405 Method Not Allowed", "{}"),
            ("/model", "405 Method Not Allowed", "{}"),
        ])
        .await;
        let voices = list_voices(&client(), &base, Some("test-key")).await.unwrap();
        assert_eq!(ids(&voices), ["default"]);
    }

    #[tokio::test]
    async fn hosted_fallback_lists_models_and_reports_real_failures() {
        let (base, _) = mock_server(vec![(
            "/model",
            "200 OK",
            r#"{"total":1,"items":[{"_id":"abc123","title":"My voice","languages":["en"]}]}"#,
        )])
        .await;
        let voices = list_voices(&client(), &base, Some("test-key")).await.unwrap();
        assert_eq!(ids(&voices), ["default", "abc123"]);

        // A rejected key is still an error, so the UI can say why.
        let (base, _) = mock_server(vec![("/model", "401 Unauthorized", r#"{"message":"invalid key"}"#)]).await;
        let err = list_voices(&client(), &base, Some("test-key")).await.unwrap_err();
        assert!(err.contains("401") && err.contains("invalid key"), "{err}");

        // A hosted outage is reported rather than silently hiding the account's voices.
        let (base, _) = mock_server(vec![("/model", "503 Service Unavailable", r#"{"message":"down"}"#)]).await;
        let err = list_voices(&client(), &base, Some("test-key")).await.unwrap_err();
        assert!(err.contains("503"), "{err}");

        // A catch-all route answering 200 with a non-JSON page is not the hosted API.
        let (base, _) = mock_server(vec![("/model", "200 OK", "<html>not json</html>")]).await;
        let voices = list_voices(&client(), &base, Some("test-key")).await.unwrap();
        assert_eq!(ids(&voices), ["default"]);
    }

    #[tokio::test]
    async fn references_list_and_no_key_paths() {
        let (base, seen) = mock_server(vec![(
            "/v1/references/list",
            "200 OK",
            r#"{"success":true,"reference_ids":["alice"]}"#,
        )])
        .await;
        let voices = list_voices(&client(), &base, None).await.unwrap();
        assert_eq!(ids(&voices), ["default", "alice"]);
        assert!(!seen.lock().unwrap()[0].bearer, "no key, no Authorization header");

        // Without a key, a missing references route never probes the hosted API.
        let (base, seen) = mock_server(vec![]).await;
        let voices = list_voices(&client(), &base, None).await.unwrap();
        assert_eq!(ids(&voices), ["default"]);
        assert_eq!(paths(&seen), ["/v1/references/list"]);

        // An auth failure on the references route is an error.
        let (base, _) = mock_server(vec![("/v1/references/list", "403 Forbidden", r#"{"detail":"bad key"}"#)]).await;
        let err = list_voices(&client(), &base, Some("test-key")).await.unwrap_err();
        assert!(err.contains("403") && err.contains("bad key"), "{err}");
    }

    #[test]
    fn tts_request_omits_default_reference() {
        let body = serde_json::to_value(TtsRequest {
            text: "hi",
            reference_id: None,
            normalize: true,
            format: "wav",
        })
        .unwrap();
        assert!(body.get("reference_id").is_none());
        assert_eq!(body["format"], "wav");
        assert!(body.get("streaming").is_none());
    }
}
