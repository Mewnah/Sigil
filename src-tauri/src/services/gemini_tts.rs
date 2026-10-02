//! `gemini-tts` plugin: Google Gemini TTS on the user's own key (P6-H; scaffolded by P6-S0).
//!
//! Gemini 3.8 Flash TTS / Flash-Lite TTS (model ids are editable: preview names change) through the
//! Generative Language API, with the key in the `x-goog-api-key` header only (never in a URL or a
//! log). The answer streams as server-sent events carrying base64 16-bit PCM (24 kHz mono), decoded
//! as it arrives into the TTS sink utterance it is given; no PCM crosses JSON IPC.
//!
//! # Frozen contract (P6-S0; change only through INTEGRATE; structs may gain fields)
//! - IPC (`plugin-commands/gemini-tts.txt`; wire camelCase; TS mirror `RustGeminiTts*` in
//!   `src/types.ts`):
//!   - `gemini_tts_speak { request: GeminiTtsRequest, utteranceId } -> SpeakReport`: speak one
//!     sentence into the begun `tts_sink` utterance `utteranceId`.
//!   - `gemini_tts_cancel { utteranceId? } -> ()`.
//!   - `gemini_tts_check { apiKey, model, baseUrl? } -> bool`: the key and model work (a tiny
//!     request; no audio played).
//!
//! # Behaviour (P6-H)
//! - Request: `POST {base}/v1beta/interactions` (the API the 3.x TTS models are documented on)
//!   with `{model, input: [user_input: text (+ speech_metadata style)], response_format: audio,
//!   generation_config.speech_config: [{voice}], stream: true}`. A base URL without the API path
//!   gets `/v1beta`. Where `/interactions` does not exist (an older proxy or API version: HTTP 404
//!   that is not about the model), the request falls back to
//!   `models/{model}:streamGenerateContent?alt=sse` with `responseModalities: ["AUDIO"]` and the
//!   prebuilt voice, and remembers that for the base URL. That API has no style field: the style
//!   goes into the prompt as words, except for models that read the prompt verbatim (3.8 and
//!   later), which then get the text alone.
//! - Answer: audio parts are found wherever the answer puts them (Interactions `step.delta` events
//!   with `delta: {type: "audio", data}` as documented in 2026-10, older `content.delta` events,
//!   `generateContent` `inlineData`); a summary event that repeats the audio
//!   (`interaction.complete`, `step.complete`) is used only when nothing streamed. Streamed audio
//!   is headerless 16-bit PCM (`audio/l16`, 24 kHz mono); WAV (the unary default) and (should the
//!   default change) MP3 / Ogg parts also decode.
//! - Errors carry the cloud failure tag (`edge_tts::cloud`): 429 / `RESOURCE_EXHAUSTED` is
//!   `rate_limited` with Google's `RetryInfo` delay, a refused key `auth`, 5xx `unavailable`, an
//!   unknown model, voice or request shape (or an account refused by location / billing)
//!   `rejected`, an answer without audio (text, a safety stop) `no_audio`. A 400 about this
//!   sentence's text (e.g. "Model tried to generate text, but it should only be used for TTS")
//!   is `no_audio` too: the local voice says that sentence and Gemini is asked for the next one
//!   (a `rejected` would keep every sentence on the local voice for 10 minutes). A failure of the
//!   sink rejects untagged with its own text.
//! - `timeoutMs` bounds the wait for the first audio and every later gap (0.5–120 s); 0 =
//!   automatic (30 s for the first audio, 20 s between chunks). The TS backend passes its short
//!   "switch to the local voice" timeout when the local fallback is ready. With an explicit
//!   timeout a new connection must be made within a third of it (at least 1 s): an unreachable
//!   Google is a `network` failure after about 1 s (3 s timeout), an error answer (429, 5xx)
//!   is reported at once, a service that connects but sends no audio is `stalled` at the
//!   timeout.
//! - A missing key fails at once (`auth`); the TS backend refuses to start without one.

mod stream;

#[cfg(test)]
mod tests;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderValue, ACCEPT};
use reqwest::{Client, Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::edge_tts::cloud::{
    cancel_command, ms, parse_retry_after, sink_for, unix_now, watched, CancelWatch, Failure, FailureKind, HandOff, Interrupted,
    Registry, SpeakError,
};
use crate::services::http_client::{
    content_type, describe_request_error, error_body_summary, http_client, normalize_base_url, HEALTH_TIMEOUT, STALL_TIMEOUT,
};
use crate::services::openai_speech::PcmSink;
use crate::services::tts_sink::SpeakReport;
use stream::{event_role, extract, google_error, json_event_name, AudioDecoder, AudioPart, EventRole, Extracted, SseParser};

/// Plugin name (`plugin:gemini-tts|...`, capability `gemini-tts:default`).
pub const PLUGIN_NAME: &str = "gemini-tts";
/// Default API base (`baseUrl` "" means this).
pub const DEFAULT_BASE_URL: &str = "https://generativelanguage.googleapis.com";
/// The model of a request that names none.
pub const DEFAULT_MODEL: &str = "gemini-3.8-flash-tts";
/// The voice of a request that names none.
pub const DEFAULT_VOICE: &str = "Kore";
/// Header that carries the key.
const KEY_HEADER: &str = "x-goog-api-key";
/// Name used in messages.
const WHAT: &str = "Gemini TTS";

/// Automatic limits (no fallback is waiting).
const AUTO_FIRST_AUDIO: Duration = Duration::from_secs(30);
const AUTO_STALL: Duration = Duration::from_secs(20);
/// Range of an explicit `timeoutMs`.
const MIN_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_TIMEOUT: Duration = Duration::from_secs(120);
/// Network time after the first-audio budget for the rest of the answer.
const BODY_BUDGET: Duration = Duration::from_secs(600);
/// Largest answer accepted (base64 audio of a long sentence is a few MB).
const MAX_BODY_BYTES: usize = 96 * 1024 * 1024;
/// Bytes of an error answer read for its message, and how long that may take.
const ERROR_BODY_LIMIT: usize = 16 * 1024;
const ERROR_BODY_TIMEOUT: Duration = Duration::from_secs(5);

/// One sentence to speak (TS `RustGeminiTtsRequest`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct GeminiTtsRequest {
    /// Sent only in the `x-goog-api-key` header. Never logged.
    pub api_key: String,
    /// API base URL; "" = [`DEFAULT_BASE_URL`].
    pub base_url: String,
    /// Model id (editable in the UI).
    pub model: String,
    pub text: String,
    /// Prebuilt voice name (`Kore`, `Puck`, ...).
    pub voice: String,
    /// Natural-language style instruction ("" = none), sent separately from the text.
    pub style: String,
    /// Lower-case BCP-47 language of the text when known (Gemini detects the language itself;
    /// kept for the request log and future API options).
    pub language: Option<String>,
    /// Time to the first audio (ms); 0 = automatic.
    pub timeout_ms: u32,
}

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

/// Which API shape a request uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Api {
    Interactions,
    GenerateContent,
}

/// API roots where `/interactions` answered 404 (use `generateContent` there).
static NO_INTERACTIONS: LazyLock<Mutex<HashSet<String>>> = LazyLock::new(Default::default);

/// The API root (`…/v1beta`) of a base URL. A URL without a scheme is https (a cloud API).
fn api_root(base_url: &str) -> Result<String, String> {
    let raw = base_url.trim();
    let raw = if raw.is_empty() { DEFAULT_BASE_URL.to_string() } else if raw.contains("://") { raw.to_string() } else { format!("https://{raw}") };
    let base = normalize_base_url(&raw)?;
    let url = Url::parse(&base).map_err(|e| format!("Invalid Gemini API URL: {e}"))?;
    if url.query().is_some() {
        return Err("The Gemini API URL must not have a query (the key goes in a header).".into());
    }
    let path = url.path().trim_end_matches('/');
    if ["/v1beta", "/v1alpha", "/v1"].iter().any(|v| path.ends_with(v)) {
        Ok(base)
    } else {
        Ok(format!("{base}/v1beta"))
    }
}

/// The model id without a `models/` prefix; only the characters model ids use (it goes into a
/// URL path).
fn model_id(model: &str) -> Result<String, String> {
    let m = model.trim();
    let m = m.strip_prefix("models/").unwrap_or(m);
    let m = if m.is_empty() { DEFAULT_MODEL } else { m };
    if m.len() > 128 || !m.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '.' | '_')) {
        return Err(format!("\"{}\" is not a Gemini model id.", m.chars().take(64).collect::<String>()));
    }
    Ok(m.to_string())
}

/// The key header (marked sensitive so it never shows in debug output).
fn key_header(api_key: &str) -> Result<HeaderValue, String> {
    let key = api_key.trim();
    if key.is_empty() {
        return Err(format!("{WHAT}: no API key. Add your Gemini API key in the Gemini TTS settings."));
    }
    let mut value = HeaderValue::from_str(key).map_err(|_| format!("{WHAT}: the API key contains characters that cannot be sent."))?;
    value.set_sensitive(true);
    Ok(value)
}

/// `timeoutMs` as (first audio, stall).
fn timing(timeout_ms: u32) -> (Duration, Duration) {
    if timeout_ms == 0 {
        return (AUTO_FIRST_AUDIO, AUTO_STALL);
    }
    let t = Duration::from_millis(u64::from(timeout_ms)).clamp(MIN_TIMEOUT, MAX_TIMEOUT);
    (t, t)
}

fn voice_of(r: &GeminiTtsRequest) -> &str {
    let v = r.voice.trim();
    if v.is_empty() {
        DEFAULT_VOICE
    } else {
        v
    }
}

/// Whether `model` reads its whole text prompt aloud (Gemini 3.8 TTS and later: "the text field is a
/// verbatim transcript"; delivery goes only in `speech_metadata.style`). Earlier TTS models (2.5,
/// 3.1) take the delivery as words in the prompt ("Say cheerfully: …"). Ids that do not parse as
/// `gemini-<major>.<minor>…` count as earlier models.
fn reads_prompt_verbatim(model: &str) -> bool {
    let Some(version) = model.strip_prefix("gemini-") else {
        return false;
    };
    let mut parts = version.split(['.', '-']);
    let major: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    let minor: u32 = parts.next().and_then(|p| p.parse().ok()).unwrap_or(0);
    major > 3 || (major == 3 && minor >= 8)
}

/// The request body for `api`.
fn request_body(api: Api, r: &GeminiTtsRequest, model: &str, text: &str) -> Value {
    let style = r.style.trim();
    match api {
        Api::Interactions => {
            let mut part = json!({ "type": "text", "text": text });
            if !style.is_empty() {
                part["annotations"] = json!([{ "type": "speech_metadata", "style": style }]);
            }
            json!({
                "model": model,
                "input": [{ "type": "user_input", "content": [part] }],
                "response_format": { "type": "audio" },
                "generation_config": { "speech_config": [{ "voice": voice_of(r) }] },
                "stream": true,
            })
        }
        Api::GenerateContent => {
            // `generateContent` has no style field: a model that reads its prompt verbatim would
            // speak the instruction, so it gets the text alone there.
            let prompt = if style.is_empty() || reads_prompt_verbatim(model) {
                text.to_string()
            } else {
                format!("Say in a {style} tone: {text}")
            };
            json!({
                "contents": [{ "role": "user", "parts": [{ "text": prompt }] }],
                "generationConfig": {
                    "responseModalities": ["AUDIO"],
                    "speechConfig": { "voiceConfig": { "prebuiltVoiceConfig": { "voiceName": voice_of(r) } } },
                },
            })
        }
    }
}

fn request_url(api: Api, root: &str, model: &str) -> Result<Url, String> {
    let url = match api {
        Api::Interactions => format!("{root}/interactions"),
        Api::GenerateContent => format!("{root}/models/{model}:streamGenerateContent?alt=sse"),
    };
    Url::parse(&url).map_err(|e| format!("Invalid Gemini API URL: {e}"))
}

// ---------------------------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------------------------

/// Up to `limit` bytes of `resp`'s body, within a few seconds.
async fn read_capped(mut resp: Response, limit: usize) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(ERROR_BODY_TIMEOUT, async {
        while buf.len() < limit {
            match resp.chunk().await {
                Ok(Some(c)) => buf.extend_from_slice(&c[..c.len().min(limit - buf.len())]),
                _ => break,
            }
        }
    })
    .await;
    buf
}

/// Whether a 404 answer is about the model (rather than a missing API route).
fn mentions_model(body: &[u8]) -> bool {
    let text = String::from_utf8_lossy(body).to_ascii_lowercase();
    text.contains("models/") || text.contains("model ")
}

/// Whether the answer is for a spoken sentence (a refusal of its text affects that sentence only)
/// or for the settings check (every refusal is about the settings).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Asked {
    Speech,
    Check,
}

/// The failure of a non-success answer (`status`, its body, the `Retry-After` header).
fn api_failure(status: u16, body: &[u8], retry_after: Option<Duration>, asked: Asked) -> Failure {
    let parsed: Option<Value> = serde_json::from_slice(body).ok();
    let ge = parsed.as_ref().map(google_error).unwrap_or_default();
    failure_from(status, &ge, error_body_summary(body), retry_after, asked)
}

/// Whether a refusal (lower-case message) is about the setup rather than one sentence's text: the
/// model (unknown, not a TTS model, no audio output), the voice, the request's shape (a changed
/// API), or the account (location, billing). Such a refusal repeats for every sentence. A message
/// about the text itself (its language, its length, a model that took it for a chat prompt) is
/// about that sentence, whatever else it names.
fn about_setup(lower: &str) -> bool {
    const SENTENCE: &[&str] = &["should only be used for tts", "language", "too long", "too many tokens", "input text", "the text"];
    if SENTENCE.iter().any(|w| lower.contains(w)) {
        return false;
    }
    const SETUP: &[&str] = &[
        // The model.
        "models/",
        "not found",
        "not supported",
        "does not support",
        "unsupported",
        "modality",
        "modalities",
        // The voice.
        "voice",
        "speaker",
        // The request's shape.
        "invalid json",
        "unknown name",
        "cannot find field",
        "invalid value at",
        "unknown field",
        "speech_config",
        "speechconfig",
        "generation_config",
        "generationconfig",
        "response_format",
        // The account.
        "location",
        "billing",
        "country",
        "region",
        "permission",
    ];
    SETUP.iter().any(|w| lower.contains(w))
}

fn failure_from(status: u16, ge: &stream::GoogleError, summary: Option<String>, retry_after: Option<Duration>, asked: Asked) -> Failure {
    let mut kind = FailureKind::for_status(status);
    let reason = ge.reason.as_deref().unwrap_or("");
    let message = ge.message.clone().or(summary).unwrap_or_default();
    let lower = message.to_ascii_lowercase();
    if reason.starts_with("API_KEY") || (lower.contains("api key") && (lower.contains("invalid") || lower.contains("not valid") || lower.contains("expired"))) {
        kind = FailureKind::Auth;
    } else if ge.status.as_deref() == Some("RESOURCE_EXHAUSTED") {
        kind = FailureKind::RateLimited;
    } else if matches!(ge.status.as_deref(), Some("UNAVAILABLE" | "INTERNAL" | "DEADLINE_EXCEEDED")) {
        kind = FailureKind::Unavailable;
    } else if asked == Asked::Speech
        && matches!(status, 400 | 413)
        && matches!(ge.status.as_deref(), None | Some("INVALID_ARGUMENT"))
        && !about_setup(&lower)
    {
        // A refusal of this sentence's text (e.g. "Model tried to generate text, but it should
        // only be used for TTS", a text too long): that sentence only, never the breaker.
        kind = FailureKind::NoAudio;
    }
    let what = match kind {
        FailureKind::Auth => "the API key was refused. Check the key in the Gemini TTS settings".to_string(),
        FailureKind::RateLimited => "the request limit or the free quota is used up for now".to_string(),
        FailureKind::Unavailable => "the service is busy or unavailable".to_string(),
        FailureKind::NoAudio => "the model could not say this sentence".to_string(),
        FailureKind::Rejected if status == 404 => "the model was not found. Check the model id".to_string(),
        FailureKind::Rejected => "the request was refused. Check the model and the voice".to_string(),
        _ => "the request failed".to_string(),
    };
    let detail = if message.is_empty() { String::new() } else { format!(": {}", crate::services::http_client::truncate_chars(&message, 200)) };
    let retry = ge.retry_ms.map(Duration::from_millis).or(retry_after);
    Failure::new(kind, format!("{WHAT}: {what} (HTTP {status}{detail}).")).with_retry_after(retry)
}

/// The failure of a request that got no answer: no connection (DNS, TCP, TLS, or not within the
/// connect bound) is `network`; a timeout after connecting is `stalled`.
fn send_failure(e: &reqwest::Error, first: Duration) -> SpeakError {
    if e.is_connect() {
        let host = e.url().and_then(|u| u.host_str()).unwrap_or("the Gemini API").to_string();
        let why = if e.is_timeout() { "no connection in time".to_string() } else { connect_cause(e) };
        return SpeakError::fail(FailureKind::Network, format!("{WHAT}: cannot reach {host} ({why}). Check the internet connection."));
    }
    let kind = if e.is_timeout() { FailureKind::Stalled } else { FailureKind::Network };
    SpeakError::fail(kind, describe_request_error(WHAT, e, first))
}

/// The innermost cause of a connect error (reqwest's own text only says "error sending request").
fn connect_cause(e: &reqwest::Error) -> String {
    let mut cause: &(dyn std::error::Error + 'static) = e;
    while let Some(next) = cause.source() {
        cause = next;
    }
    crate::services::http_client::truncate_chars(&cause.to_string(), 160)
}

/// An error object inside a streamed answer.
fn stream_failure(err: &Value) -> Failure {
    let ge = google_error(err);
    let status = ge.code.unwrap_or(500);
    failure_from(status, &ge, None, None, Asked::Speech)
}

/// The failure of an answer that ended without audio.
fn no_audio(extracted: &Extracted) -> SpeakError {
    let why = match (&extracted.finish, extracted.text) {
        (Some(reason), _) => format!("{WHAT}: no audio for this sentence (the model stopped: {reason})."),
        (None, true) => format!("{WHAT}: the model answered with text instead of audio for this sentence."),
        (None, false) => format!("{WHAT}: the service returned no audio for this sentence."),
    };
    SpeakError::fail(FailureKind::NoAudio, why)
}

// ---------------------------------------------------------------------------------------------
// Connecting
// ---------------------------------------------------------------------------------------------

/// Shortest connect bound of a request with an explicit first-audio limit.
const MIN_CONNECT: Duration = Duration::from_secs(1);
/// Connect bounds are rounded up to this step (one HTTP client per step in use).
const CONNECT_STEP_MS: u128 = 250;
/// Clients kept (more distinct limits than this in one session only rebuild them).
const MAX_CLIENTS: usize = 8;

/// The connect bound (DNS, TCP and TLS of a new connection) for an explicit first-audio limit: a
/// third of it, at least 1 s (never more than the limit itself), rounded up to 250 ms, as for the
/// Edge voices. While the local voice waits behind Gemini (a short limit), an unreachable Google is
/// then reported as a `network` failure after about 1 s (3 s limit) instead of a `stalled` one at
/// the full limit; a connection kept alive from an earlier sentence needs no connect at all, and
/// reqwest races IPv6 and IPv4 itself. A slow link that needs longer gets it with a longer
/// "switch to the local voice" time (the bound scales with it).
fn connect_bound(first: Duration) -> Duration {
    let ms = (first.as_millis() / 3).max(MIN_CONNECT.as_millis()).min(first.as_millis().max(1));
    Duration::from_millis(u64::try_from(ms.div_ceil(CONNECT_STEP_MS) * CONNECT_STEP_MS).unwrap_or(u64::MAX))
}

/// Clients by connect bound (ms).
static CLIENTS: LazyLock<Mutex<HashMap<u64, Client>>> = LazyLock::new(Default::default);

/// The HTTP client of a request with `timeout_ms` (0 = automatic: the shared client and its 30 s
/// connect limit, as nothing better waits).
fn client_for(timeout_ms: u32) -> Client {
    if timeout_ms == 0 {
        return http_client().clone();
    }
    let bound = connect_bound(timing(timeout_ms).0);
    let key = u64::try_from(bound.as_millis()).unwrap_or(u64::MAX);
    let mut clients = CLIENTS.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(c) = clients.get(&key) {
        return c.clone();
    }
    let built = Client::builder()
        .connect_timeout(bound)
        .read_timeout(STALL_TIMEOUT)
        .user_agent(format!("Sigil/{} (desktop; https://github.com/Mewnah/Sigil)", env!("CARGO_PKG_VERSION")))
        .build();
    match built {
        Ok(c) => {
            if clients.len() >= MAX_CLIENTS {
                clients.clear();
            }
            clients.insert(key, c.clone());
            c
        }
        Err(e) => {
            tracing::warn!(target: "sigil::http", "{WHAT}: cannot build an HTTP client ({e}); using the shared one");
            http_client().clone()
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Speaking
// ---------------------------------------------------------------------------------------------

/// Running requests by utterance id (`gemini_tts_cancel`).
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

/// The state of one streamed answer.
struct Answer {
    hand: HandOff,
    decoder: AudioDecoder,
    /// Audio of a summary event, used only when nothing streamed.
    summary: Vec<AudioPart>,
    seen: Extracted,
    last_audio: Option<Instant>,
    /// The hand-off thread stopped (a push failed): stop reading; its outcome says why.
    closed: bool,
}

impl Answer {
    /// Take one parsed JSON event (or a whole non-streamed answer).
    async fn take(&mut self, v: &Value, name: Option<&str>, cancel: &CancelWatch) -> Result<(), SpeakError> {
        let mut x = Extracted::default();
        extract(v, &mut x);
        if let Some(err) = &x.error {
            return Err(stream_failure(err).into());
        }
        self.seen.text |= x.text;
        if self.seen.finish.is_none() {
            self.seen.finish = x.finish.clone();
        }
        match event_role(name) {
            EventRole::Incremental => {
                for part in &x.audio {
                    self.send(part, cancel).await?;
                }
            }
            EventRole::Summary => {
                if !x.audio.is_empty() {
                    self.summary = x.audio;
                }
            }
        }
        Ok(())
    }

    async fn send(&mut self, part: &AudioPart, cancel: &CancelWatch) -> Result<(), SpeakError> {
        let block = self
            .decoder
            .decode(part)
            .map_err(|e| SpeakError::fail(FailureKind::Other, format!("{WHAT}: {e}")))?;
        let Some(block) = block else {
            return Ok(());
        };
        match self.hand.send(block, cancel).await {
            Err(_) => Err(SpeakError::Cancelled),
            // The hand-off thread stopped: its outcome says why (read at the end).
            Ok(false) => {
                self.closed = true;
                Ok(())
            }
            Ok(true) => {
                self.last_audio = Some(Instant::now());
                Ok(())
            }
        }
    }
}

/// Speak `request` into `sink` (registered under `utterance_id` for cancellation).
pub(crate) async fn speak_with(request: &GeminiTtsRequest, utterance_id: &str, sink: Box<dyn PcmSink>, client: &Client) -> Result<SpeakReport, SpeakError> {
    let text = request.text.trim();
    if text.is_empty() {
        return Ok(SpeakReport { sample_rate: stream::DEFAULT_RATE, ..SpeakReport::default() });
    }
    let key = key_header(&request.api_key).map_err(|e| SpeakError::fail(FailureKind::Auth, e))?;
    let model = model_id(&request.model).map_err(|e| SpeakError::fail(FailureKind::Rejected, format!("{WHAT}: {e}")))?;
    let root = api_root(&request.base_url).map_err(|e| SpeakError::fail(FailureKind::Rejected, format!("{WHAT}: {e}")))?;
    // A second request for the same utterance is the caller's mistake, not the service's.
    let registration = REGISTRY.register(utterance_id).map_err(SpeakError::Local)?;
    let cancel = CancelWatch::new(Arc::clone(&registration.flag), sink.cancel_probe());
    if cancel.is_set() {
        return Err(SpeakError::Cancelled);
    }
    let (first, stall) = timing(request.timeout_ms);
    let started = Instant::now();
    let first_deadline = started + first;
    let budget_deadline = first_deadline + BODY_BUDGET;
    let secs = |d: Duration| d.as_millis().div_ceil(1000).max(1);
    let stalled_first = || SpeakError::fail(FailureKind::Stalled, format!("{WHAT}: no audio within {} s (the service did not answer in time).", secs(first)));

    // Send (falling back to generateContent where /interactions does not exist).
    let mut api = if NO_INTERACTIONS.lock().unwrap_or_else(PoisonError::into_inner).contains(&root) { Api::GenerateContent } else { Api::Interactions };
    let resp = loop {
        let url = request_url(api, &root, &model).map_err(|e| SpeakError::fail(FailureKind::Rejected, format!("{WHAT}: {e}")))?;
        let req = client
            .post(url)
            .header(KEY_HEADER, key.clone())
            .header(ACCEPT, "text/event-stream, application/json")
            .json(&request_body(api, request, &model, text))
            // An outer bound only: the waits are timed here.
            .timeout(first + BODY_BUDGET + Duration::from_secs(600));
        let resp = match watched(req.send(), Some(first_deadline), &cancel).await {
            Err(Interrupted::Cancelled) => return Err(SpeakError::Cancelled),
            Err(Interrupted::TimedOut) => return Err(stalled_first()),
            Ok(Err(e)) => return Err(send_failure(&e, first)),
            Ok(Ok(resp)) => resp,
        };
        let status = resp.status();
        if status.is_success() {
            break resp;
        }
        let retry_after = resp
            .headers()
            .get("retry-after")
            .and_then(|v| v.to_str().ok())
            .and_then(|v| parse_retry_after(v, unix_now()));
        let body = read_capped(resp, ERROR_BODY_LIMIT).await;
        if status == StatusCode::NOT_FOUND && api == Api::Interactions && !mentions_model(&body) {
            tracing::info!(target: "sigil::tts", "{WHAT}: {root} has no /interactions; using generateContent");
            NO_INTERACTIONS.lock().unwrap_or_else(PoisonError::into_inner).insert(root.clone());
            api = Api::GenerateContent;
            continue;
        }
        return Err(api_failure(status.as_u16(), &body, retry_after, Asked::Speech).into());
    };

    // Stream.
    let is_sse = content_type(&resp).unwrap_or_default().to_ascii_lowercase().contains("event-stream");
    let mut resp = resp;
    let mut parser = SseParser::default();
    let mut whole = Vec::new();
    let mut received = 0usize;
    let mut answer = Answer {
        hand: HandOff::start(sink, started),
        decoder: AudioDecoder::default(),
        summary: Vec::new(),
        seen: Extracted::default(),
        last_audio: None,
        closed: false,
    };
    let result: Result<(), SpeakError> = async {
        loop {
            if answer.closed {
                return Ok(());
            }
            let limit = match answer.last_audio {
                Some(at) => at + stall,
                None => first_deadline,
            };
            let chunk = match watched(resp.chunk(), Some(limit.min(budget_deadline)), &cancel).await {
                Err(Interrupted::Cancelled) => return Err(SpeakError::Cancelled),
                Err(Interrupted::TimedOut) => {
                    return Err(if answer.last_audio.is_some() {
                        SpeakError::fail(FailureKind::Stalled, format!("{WHAT}: the service stopped sending audio for {} s (stalled).", secs(stall)))
                    } else {
                        stalled_first()
                    })
                }
                Ok(Err(e)) => return Err(SpeakError::fail(FailureKind::Network, describe_request_error(WHAT, &e, stall))),
                Ok(Ok(None)) => break,
                Ok(Ok(Some(bytes))) => bytes,
            };
            received = received.saturating_add(chunk.len());
            if received > MAX_BODY_BYTES {
                return Err(SpeakError::fail(FailureKind::Other, format!("{WHAT}: the answer is too large.")));
            }
            if is_sse {
                for ev in parser.feed(&chunk) {
                    take_event(&mut answer, ev, &cancel).await?;
                }
            } else {
                whole.extend_from_slice(&chunk);
            }
        }
        if is_sse {
            for ev in parser.finish() {
                take_event(&mut answer, ev, &cancel).await?;
            }
        } else if !whole.is_empty() {
            let v: Value = serde_json::from_slice(&whole).map_err(|_| {
                let summary = error_body_summary(&whole).unwrap_or_default();
                SpeakError::fail(FailureKind::Other, format!("{WHAT}: unexpected answer ({summary})."))
            })?;
            // A whole answer (an array of chunks, or one interaction): every part is new.
            answer.take(&v, None, &cancel).await?;
        }
        if answer.hand.handed == 0 && !answer.summary.is_empty() {
            for part in std::mem::take(&mut answer.summary) {
                answer.send(&part, &cancel).await?;
            }
        }
        Ok(())
    }
    .await;
    let handed = answer.hand.handed;
    if let Err(e) = result {
        return Err(e.with_partial(handed > 0));
    }
    let seen = answer.seen.clone();
    let outcome = match answer.hand.finish(&cancel).await {
        Ok(o) => o,
        Err(_) => return Err(SpeakError::Cancelled),
    };
    if let Some(e) = outcome.error {
        // The sink refused or timed out a push: this PC's failure, sent untagged (never the service's).
        return Err(SpeakError::from_sink(e));
    }
    if cancel.is_set() {
        return Err(SpeakError::Cancelled);
    }
    if outcome.frames == 0 {
        return Err(no_audio(&seen));
    }
    let first_ms = outcome.first.unwrap_or_default();
    Ok(SpeakReport {
        sample_rate: outcome.rate,
        frames: outcome.frames,
        first_audio_ms: ms(first_ms),
        total_ms: ms(outcome.last.unwrap_or(first_ms)),
    })
}

/// One SSE event of the answer.
async fn take_event(answer: &mut Answer, ev: stream::SseEvent, cancel: &CancelWatch) -> Result<(), SpeakError> {
    let data = ev.data.trim();
    if data.is_empty() || data == "[DONE]" {
        return Ok(());
    }
    let Ok(v) = serde_json::from_str::<Value>(data) else {
        tracing::debug!(target: "sigil::tts", "{WHAT}: skipped an event that is not JSON");
        return Ok(());
    };
    let name = ev.event.clone().filter(|n| !n.is_empty() && n != "message").or_else(|| json_event_name(&v));
    answer.take(&v, name.as_deref(), cancel).await
}

async fn speak(request: GeminiTtsRequest, utterance_id: String) -> Result<SpeakReport, String> {
    let id = utterance_id.trim();
    if id.is_empty() {
        return Err("utteranceId must not be empty".into());
    }
    let client = client_for(request.timeout_ms);
    let result = async {
        let sink = sink_for(id)?;
        speak_with(&request, id, sink, &client).await
    }
    .await;
    match result {
        Ok(report) => {
            tracing::debug!(
                target: "sigil::tts",
                "{WHAT}: {} frames at {} Hz, first audio after {} ms, done after {} ms",
                report.frames,
                report.sample_rate,
                report.first_audio_ms,
                report.total_ms
            );
            Ok(report)
        }
        Err(SpeakError::Failed(f)) => {
            tracing::info!(target: "sigil::tts", "{} ({})", f.message, f.kind.wire());
            Err(f.to_wire())
        }
        Err(SpeakError::Local(message)) => {
            tracing::info!(target: "sigil::tts", "{WHAT}: {message}");
            Err(message)
        }
        Err(e) => Err(e.into_wire()),
    }
}

/// Whether `api_key` and `model` work: `GET {root}/models/{model}` (no audio, no quota for speech).
/// `Ok(false)` when the service cannot be reached; `Err` naming the problem when the key or the
/// model is refused.
async fn check_with(client: &Client, api_key: &str, model: &str, base_url: &str) -> Result<bool, String> {
    let key = key_header(api_key)?;
    let model = model_id(model).map_err(|e| format!("{WHAT}: {e}"))?;
    let root = api_root(base_url).map_err(|e| format!("{WHAT}: {e}"))?;
    let url = Url::parse(&format!("{root}/models/{model}")).map_err(|e| format!("Invalid Gemini API URL: {e}"))?;
    let resp = match client.get(url).header(KEY_HEADER, key).header(ACCEPT, "application/json").timeout(HEALTH_TIMEOUT).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::info!(target: "sigil::http", "{}", describe_request_error(WHAT, &e, HEALTH_TIMEOUT));
            return Ok(false);
        }
    };
    let status = resp.status();
    if status.is_success() || status == StatusCode::TOO_MANY_REQUESTS {
        return Ok(true);
    }
    let body = read_capped(resp, ERROR_BODY_LIMIT).await;
    let failure = api_failure(status.as_u16(), &body, None, Asked::Check);
    match failure.kind {
        FailureKind::Unavailable => {
            tracing::info!(target: "sigil::http", "{}", failure.message);
            Ok(false)
        }
        FailureKind::Rejected if status == StatusCode::NOT_FOUND => Err(format!("{WHAT}: Gemini does not know the model \"{model}\". Check the model id.")),
        _ => Err(failure.message),
    }
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

#[command]
async fn gemini_tts_speak(request: GeminiTtsRequest, utterance_id: String) -> Result<SpeakReport, String> {
    speak(request, utterance_id).await
}

#[command]
async fn gemini_tts_cancel(utterance_id: Option<String>) -> Result<(), String> {
    cancel_command(&REGISTRY, utterance_id.as_deref());
    Ok(())
}

#[command]
async fn gemini_tts_check(api_key: String, model: String, base_url: Option<String>) -> Result<bool, String> {
    check_with(http_client(), &api_key, &model, base_url.as_deref().unwrap_or("")).await
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![gemini_tts_speak, gemini_tts_cancel, gemini_tts_check])
        .build()
}
