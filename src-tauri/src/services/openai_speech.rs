//! Generic OpenAI-compatible streaming TTS client (`openai-speech` plugin, P2-I), and the streaming
//! core every HTTP speech backend uses to play through the TTS sink.
//!
//! # The client
//! `POST {baseUrl}/audio/speech` with `{model, input, voice, response_format: pcm|wav, speed,
//! stream, ...extra}` and an optional `Authorization: Bearer` header (keys only in headers, never in
//! URLs or logs). The response body is decoded **as it arrives** and pushed into the `tts_sink`
//! utterance the TS sentence pipeline began, so the first audio plays before the server finished
//! the sentence. Made for Kokoro-FastAPI, Chatterbox servers, vLLM-Omni (Voxtral-4B-TTS,
//! IndexTTS-2.5, VoxCPM2) and qwentts.cpp's `tts-server`; the presets live in the TS service
//! (`src/core/services/tts/services/openaiSpeech.ts`). OpenAI's own API follows the same request
//! shape (not verified live: it needs a paid key). The legacy `kokoro` and `chatterbox` backends map
//! onto this client with their saved settings.
//!
//! - A base URL without a path gets `/v1` (`http://127.0.0.1:8880` → `http://127.0.0.1:8880/v1`).
//! - Body: `model` / `voice` only when set, `speed` only when it is not 1 (clamped to 0.25–4),
//!   `stream: true` only when asked and never to `api.openai.com` / Azure OpenAI (their schema has
//!   no such member, and they stream anyway); `extra` members are merged last and may override
//!   anything but `input`.
//! - Audio: raw `pcm` is little-endian s16 mono at `pcmSampleRate` (a `rate=` parameter of the
//!   response's Content-Type wins); `wav` may be a streamed WAV (placeholder sizes), a complete WAV,
//!   or several WAVs back to back; any integer or float sample format and any channel count. A
//!   server that answers with a WAV although `pcm` was asked is decoded as WAV. MP3 / Ogg / FLAC
//!   answers (servers that ignore `response_format`) are buffered and decoded once complete; they
//!   are recognized by their Content-Type or signature (`ID3`, `OggS`, `fLaC`, or two consecutive
//!   MPEG / ADTS frame headers), so an untagged MP3 is never played as raw PCM. A JSON or text
//!   answer fails with the server's message.
//!
//! # Streaming core (shared with the legacy HTTP plugins)
//! [`HttpSpeech`] + [`speak_http`] stream any HTTP speech response into a sink utterance. The
//! `melo-tts`, `neural-sidecar-tts` and `fish-speech` plugins build their own request (their APIs are
//! not OpenAI-compatible) and use it for their `*_to_sink` commands, so every HTTP backend honours
//! the output device, the voice-changer route, stop and skip, and no audio crosses IPC.
//! - Network work never waits on playback: a blocking hand-off thread pushes decoded blocks into the
//!   sink (whose back-pressure blocks), through a bounded channel.
//! - Timeouts ([`Timing`]): the server has `answer` to deliver the first audio (response headers
//!   included), then at most `stall` between two pieces of audio, and `answer` + 10 min of network
//!   time in total. Automatic values: a server on this PC or the LAN gets 60 s + 1 s per character
//!   (max 10 min; slow CPU servers answer only when the clip is done) and a 60 s stall limit; any
//!   other server 60 s (a little more for long text, max 120 s) and a 30 s stall limit. An explicit
//!   `timeoutMs` sets `answer` (1 s–10 min) and `stall` (1–110 s). Both stay inside the TTS sink's
//!   own bounds (11 min for a sentence's first audio, 120 s between pushes once it plays), so the
//!   sink never drops a sentence whose request is still within its budget.
//! - Cancel: `openai_speech_cancel` (any running HTTP speech stream, by utterance id or all), and a
//!   cancelled sink utterance (stop / skip) both end the request within ~25 ms with
//!   `Err("cancelled")`.
//!
//! # Frozen contract (P2-S0; change only through an INTEGRATE step)
//! Types may gain fields; names and wire formats stay (camelCase). TS mirrors:
//! `RustOpenAiSpeechRequest` / `RustTtsSpeakReport` in `src/types.ts`.
//! - `plugin:openai-speech|openai_speech_speak { request: OpenAiSpeechRequest, utteranceId }` →
//!   [`SpeakReport`]: streams the response into the `tts_sink` utterance `utteranceId` (begun and
//!   later ended by the TS pipeline); resolves when the last audio was pushed; a cancelled
//!   utterance ends the request with `Err("cancelled")`.
//! - `plugin:openai-speech|openai_speech_cancel { utteranceId?: string | null }` → `void`
//!   (`null` = every running request). Also stops the streams of the `melo-tts`,
//!   `neural-sidecar-tts` and `fish-speech` `*_to_sink` commands, which share this core.
//! - `plugin:openai-speech|openai_speech_check { baseUrl, apiKey? }` → `boolean` (server reachable;
//!   never throws for a down server; rejects when the server refuses the key, so the UI can say
//!   why).
//! - `plugin:openai-speech|openai_speech_list_voices { baseUrl, apiKey? }` → `string[]` (servers
//!   that list voices, e.g. Kokoro-FastAPI `/audio/voices`; `[]` when unsupported).

use std::collections::HashMap;
use std::future::Future;
use std::io::Cursor;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderName, HeaderValue, AUTHORIZATION};
use reqwest::{Client, Response, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};
use tokio::sync::mpsc;

use crate::services::http_client::{
    api_url, content_type, describe_api_error, describe_request_error, error_body_summary,
    http_client, is_auth_failure, is_local_endpoint, local_http_client, local_tts_timeout,
    non_empty_key, normalize_base_url, parse_named_list, probe, redact_url, secs, tts_post,
    tts_timeout, LIST_TIMEOUT, LOCAL_TTS_TIMEOUT, LOCAL_TTS_TIMEOUT_MAX, MAX_AUDIO_BYTES, STALL_TIMEOUT,
    TTS_STALL_TIMEOUT,
};
use crate::services::tts_sink::{self, SinkError, UtteranceWriter};

pub use crate::services::tts_sink::SpeakReport;

/// Plugin name (`plugin:openai-speech|...`, capability `openai-speech:default`).
pub const PLUGIN_NAME: &str = "openai-speech";

/// Error text of a cancelled request (the sink's `SinkError::Cancelled` text; TS `SINK_CANCELLED`).
pub const CANCELLED: &str = "cancelled";

/// Sample rate of raw `pcm` responses when the request does not say (OpenAI, Kokoro-FastAPI).
pub const DEFAULT_PCM_RATE: u32 = 24_000;
/// Accepted `pcmSampleRate` range.
const PCM_RATE_RANGE: std::ops::RangeInclusive<u32> = 8_000..=192_000;
/// Speed range sent to servers (OpenAI's).
const SPEED_RANGE: (f32, f32) = (0.25, 4.0);
/// Automatic time a non-local server gets for its first audio.
const DEFAULT_ANSWER: Duration = Duration::from_secs(60);
/// Shortest explicit timeout.
const MIN_TIMEOUT: Duration = Duration::from_secs(1);
/// The TTS sink fails the playing sentence (the queue head) when no audio was pushed for this long
/// once its audio started (`tts_sink::STALL_TIMEOUT`, the default `SinkConfig::stall_timeout`).
/// The sink's cancel then ends the request as `"cancelled"`.
const SINK_STALL_TIMEOUT: Duration = crate::services::tts_sink::STALL_TIMEOUT;
/// The sink's bound for a head sentence without any audio yet (`tts_sink::FIRST_AUDIO_TIMEOUT`:
/// [`LOCAL_TTS_TIMEOUT_MAX`] + 60 s). Every `answer` budget stays below it.
#[cfg(test)]
const SINK_FIRST_AUDIO_TIMEOUT: Duration = crate::services::tts_sink::FIRST_AUDIO_TIMEOUT;
/// Longest gap between two pieces of audio: below [`SINK_STALL_TIMEOUT`], so a stalled server ends
/// with this module's own "stalled" message before the sink drops the sentence.
const MAX_STALL: Duration = Duration::from_secs(SINK_STALL_TIMEOUT.as_secs() - 10);
/// Network time allowed after `answer` (on top of it) for the rest of the body.
const BODY_BUDGET: Duration = LOCAL_TTS_TIMEOUT_MAX;
/// Extra room of the request's outer (reqwest) timeout for time spent waiting on playback.
const PLAYBACK_ALLOWANCE: Duration = Duration::from_secs(600);
/// How often a waiting request checks for cancellation.
const CANCEL_POLL: Duration = Duration::from_millis(25);
/// Decoded blocks buffered between the network and the sink hand-off.
const CHANNEL_BLOCKS: usize = 64;
/// Longest single push into the sink (a cancel between pushes stops the rest).
const PUSH_MS: usize = 500;
/// Bytes of a text answer read for its message.
const TEXT_BODY_LIMIT: usize = 4096;
/// How long reading such a text answer may take.
const TEXT_BODY_TIMEOUT: Duration = Duration::from_secs(5);

/// `response_format` of the request. Wire: lower-case.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeechFormat {
    /// Raw little-endian s16 mono at [`OpenAiSpeechRequest::pcm_sample_rate`].
    #[default]
    Pcm,
    /// A WAV stream (header gives the format).
    Wav,
}

impl SpeechFormat {
    pub fn as_str(self) -> &'static str {
        match self {
            SpeechFormat::Pcm => "pcm",
            SpeechFormat::Wav => "wav",
        }
    }
}

/// One synthesis request.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct OpenAiSpeechRequest {
    /// Server base URL including the API prefix, e.g. `http://127.0.0.1:8880/v1` (no path → `/v1`).
    pub base_url: String,
    /// Sent only as `Authorization: Bearer`; empty = no header. Never logged.
    pub api_key: String,
    /// `""` = not sent (the server's default model).
    pub model: String,
    /// Text of one sentence.
    pub input: String,
    /// `""` = not sent (the server's default voice).
    pub voice: String,
    pub response_format: SpeechFormat,
    /// 1 = not sent; otherwise clamped to 0.25–4.
    pub speed: f32,
    /// Ask the server to stream (`"stream": true`; not sent when false, nor to OpenAI's own API).
    pub stream: bool,
    /// Sample rate of raw `pcm` responses (servers differ: Kokoro-FastAPI, OpenAI, Voxtral and
    /// qwentts use 24000, VoxCPM2 48000). 0 = 24000.
    pub pcm_sample_rate: u32,
    /// Extra JSON members merged into the request body (server-specific options); `null` or an
    /// object. May override every member except `input`.
    pub extra: Value,
    /// Per-request timeout (ms): time to the first audio (at most 10 min) and longest gap between
    /// two pieces of audio (at most 110 s). 0 = automatic (see the module docs; 60 s for a remote
    /// server).
    pub timeout_ms: u32,
}

impl Default for OpenAiSpeechRequest {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            api_key: String::new(),
            model: String::new(),
            input: String::new(),
            voice: String::new(),
            response_format: SpeechFormat::Pcm,
            speed: 1.0,
            stream: true,
            pcm_sample_rate: DEFAULT_PCM_RATE,
            extra: Value::Null,
            timeout_ms: 0,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

/// The API base for `raw`: validated, without trailing slashes, and with `/v1` when it has no path.
pub(crate) fn speech_base(raw: &str) -> Result<String, String> {
    let base = normalize_base_url(raw)?;
    let parsed = Url::parse(&base).map_err(|e| format!("Invalid server URL: {e}"))?;
    if parsed.path().trim_matches('/').is_empty() && parsed.query().is_none() {
        Ok(format!("{}/v1", base.trim_end_matches('/')))
    } else {
        Ok(base)
    }
}

/// A finite speed other than 1, clamped to [`SPEED_RANGE`]; `None` = do not send.
fn speed_member(speed: f32) -> Option<f32> {
    if !speed.is_finite() || speed <= 0.0 {
        return None;
    }
    let s = speed.clamp(SPEED_RANGE.0, SPEED_RANGE.1);
    ((s - 1.0).abs() > 1e-3).then_some(s)
}

/// Whether `url` is OpenAI's own API (or Azure OpenAI): `/audio/speech` there streams by default
/// and its schema has no `stream` member (it has `stream_format`), so the member is not sent (a
/// strict schema would refuse the whole request). An explicit `extra` member is still sent.
fn omits_stream_member(url: &Url) -> bool {
    url.host_str()
        .map(|h| h.trim_end_matches('.'))
        .is_some_and(|h| h == "api.openai.com" || h.ends_with(".openai.azure.com"))
}

/// The JSON body of `r` (see the module docs); `stream_member` = false never sends `stream`.
fn request_body(r: &OpenAiSpeechRequest, stream_member: bool) -> Result<Value, String> {
    let mut body = Map::new();
    let model = r.model.trim();
    if !model.is_empty() {
        body.insert("model".into(), Value::from(model));
    }
    body.insert("input".into(), Value::from(r.input.as_str()));
    let voice = r.voice.trim();
    if !voice.is_empty() {
        body.insert("voice".into(), Value::from(voice));
    }
    body.insert("response_format".into(), Value::from(r.response_format.as_str()));
    if let Some(speed) = speed_member(r.speed) {
        body.insert("speed".into(), Value::from(speed));
    }
    if r.stream && stream_member {
        body.insert("stream".into(), Value::Bool(true));
    }
    match &r.extra {
        Value::Null => {}
        Value::Object(extra) => {
            for (k, v) in extra {
                if k != "input" {
                    body.insert(k.clone(), v.clone());
                }
            }
        }
        _ => return Err("Extra request fields must be a JSON object.".into()),
    }
    Ok(Value::Object(body))
}

/// `Authorization: Bearer <key>` (marked sensitive), or `None` for a blank key.
pub(crate) fn bearer_header(key: Option<&str>) -> Result<Option<HeaderValue>, String> {
    let Some(key) = non_empty_key(key) else {
        return Ok(None);
    };
    let mut value = HeaderValue::from_str(&format!("Bearer {key}"))
        .map_err(|_| "The API key contains characters that cannot be sent in an HTTP header.".to_string())?;
    value.set_sensitive(true);
    Ok(Some(value))
}

/// One HTTP speech request to stream into a sink utterance (see the module docs).
pub(crate) struct HttpSpeech {
    /// Names the operation in messages ("MeloTTS synthesis").
    what: String,
    url: Url,
    body: Value,
    /// The text being spoken (empty = nothing to do; its length scales the automatic timeout).
    text: String,
    bearer: Option<HeaderValue>,
    headers: Vec<(HeaderName, HeaderValue)>,
    expect: SpeechFormat,
    pcm_rate: u32,
    timeout: Option<Duration>,
    /// Client override (tests); `None` = the shared client for the endpoint.
    client: Option<Client>,
}

impl HttpSpeech {
    /// A JSON `POST` of `body` to `url` speaking `text`, expecting `expect` audio.
    pub(crate) fn new(what: impl Into<String>, url: Url, body: Value, text: &str, expect: SpeechFormat) -> Self {
        Self {
            what: what.into(),
            url,
            body,
            text: text.to_string(),
            bearer: None,
            headers: Vec::new(),
            expect,
            pcm_rate: DEFAULT_PCM_RATE,
            timeout: None,
            client: None,
        }
    }

    /// Send `key` as a Bearer token (blank = no header).
    pub(crate) fn bearer(mut self, key: Option<&str>) -> Result<Self, String> {
        self.bearer = bearer_header(key)?;
        Ok(self)
    }

    /// An extra request header.
    pub(crate) fn header(mut self, name: HeaderName, value: HeaderValue) -> Self {
        self.headers.push((name, value));
        self
    }

    /// Sample rate of a raw `pcm` answer.
    pub(crate) fn pcm_rate(mut self, rate: u32) -> Self {
        self.pcm_rate = rate;
        self
    }

    /// Explicit timeout in ms (0 = automatic).
    pub(crate) fn timeout_ms(mut self, ms: u32) -> Self {
        self.timeout = (ms > 0).then(|| Duration::from_millis(u64::from(ms)));
        self
    }

    #[cfg(test)]
    fn with_client(mut self, client: Client) -> Self {
        self.client = Some(client);
        self
    }

    #[cfg(test)]
    pub(crate) fn body(&self) -> &Value {
        &self.body
    }

    #[cfg(test)]
    pub(crate) fn url(&self) -> &Url {
        &self.url
    }
}

/// The streaming spec of an OpenAI-compatible request.
fn openai_spec(r: &OpenAiSpeechRequest) -> Result<HttpSpeech, String> {
    let base = speech_base(&r.base_url)?;
    let url = api_url(&base, "/audio/speech")?;
    let body = request_body(r, !omits_stream_member(&url))?;
    let pcm_rate = match r.pcm_sample_rate {
        0 => DEFAULT_PCM_RATE,
        n if PCM_RATE_RANGE.contains(&n) => n,
        n => {
            return Err(format!(
                "PCM sample rate {n} Hz is out of range ({}-{} Hz).",
                PCM_RATE_RANGE.start(),
                PCM_RATE_RANGE.end()
            ))
        }
    };
    HttpSpeech::new("Speech request", url, body, &r.input, r.response_format)
        .pcm_rate(pcm_rate)
        .timeout_ms(r.timeout_ms)
        .bearer(Some(&r.api_key))
}

/// Timeouts of one request (see the module docs). Both fit inside the sink's own bounds: `answer`
/// (at most [`LOCAL_TTS_TIMEOUT_MAX`]) below its first-audio bound, `stall` (at most
/// [`MAX_STALL`]) below its stall timeout, so the sink never drops a sentence this request still
/// waits for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Timing {
    /// Until the first audio (response headers included).
    answer: Duration,
    /// Longest gap between two pieces of audio after the first one.
    stall: Duration,
    /// Endpoint on this PC or the LAN.
    local: bool,
}

impl Timing {
    fn for_request(url: &Url, text: &str, explicit: Option<Duration>) -> Self {
        let local = is_local_endpoint(url);
        let (answer, stall) = match explicit {
            Some(t) => (t.clamp(MIN_TIMEOUT, LOCAL_TTS_TIMEOUT_MAX), t.max(MIN_TIMEOUT)),
            None if local => (local_tts_timeout(text), LOCAL_TTS_TIMEOUT),
            None => (tts_timeout(text).max(DEFAULT_ANSWER), TTS_STALL_TIMEOUT),
        };
        Self { answer, stall: stall.min(MAX_STALL), local }
    }

    /// Total network time.
    fn budget(&self) -> Duration {
        self.answer + BODY_BUDGET
    }

    /// The reqwest timeout: an outer bound only (network waits are timed here, and time spent
    /// waiting for playback must not count).
    fn outer(&self) -> Duration {
        self.budget() + PLAYBACK_ALLOWANCE
    }

    /// Whether the request needs the long-read client (a silent non-streaming server may think
    /// longer than the shared client's read timeout).
    fn long(&self) -> bool {
        self.local || self.answer > STALL_TIMEOUT || self.stall > STALL_TIMEOUT
    }
}

// ---------------------------------------------------------------------------------------------
// Cancellation registry
// ---------------------------------------------------------------------------------------------

/// Running HTTP speech streams by utterance id.
static RUNNING: LazyLock<Mutex<HashMap<String, Arc<AtomicBool>>>> = LazyLock::new(Default::default);

/// A registered stream; unregisters on drop.
struct Registration {
    id: String,
    flag: Arc<AtomicBool>,
}

impl Drop for Registration {
    fn drop(&mut self) {
        let mut map = RUNNING.lock().unwrap_or_else(PoisonError::into_inner);
        if map.get(&self.id).is_some_and(|f| Arc::ptr_eq(f, &self.flag)) {
            map.remove(&self.id);
        }
    }
}

fn register(id: &str) -> Result<Registration, String> {
    let mut map = RUNNING.lock().unwrap_or_else(PoisonError::into_inner);
    if map.contains_key(id) {
        return Err(format!("TTS utterance {id:?} is already being spoken."));
    }
    let flag = Arc::new(AtomicBool::new(false));
    map.insert(id.to_string(), Arc::clone(&flag));
    Ok(Registration { id: id.to_string(), flag })
}

/// Cancel the HTTP speech stream of utterance `id`, or (`None`) every running one.
pub(crate) fn cancel(id: Option<&str>) {
    let map = RUNNING.lock().unwrap_or_else(PoisonError::into_inner);
    match id {
        Some(id) => {
            if let Some(flag) = map.get(id) {
                flag.store(true, Ordering::Release);
            }
        }
        None => map.values().for_each(|f| f.store(true, Ordering::Release)),
    }
}

// ---------------------------------------------------------------------------------------------
// Sink hand-off
// ---------------------------------------------------------------------------------------------

/// Checks whether the sink utterance was cancelled, from any thread.
pub(crate) type CancelProbe = Arc<dyn Fn() -> bool + Send + Sync>;

/// Where decoded audio goes: a sink utterance ([`UtteranceWriter`]) or a test double.
pub(crate) trait PcmSink: Send {
    /// Queue interleaved samples (may block for back-pressure).
    fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError>;
    /// A probe the network side polls while this sink is busy on the hand-off thread.
    fn cancel_probe(&self) -> CancelProbe;
}

struct SinkWriter(UtteranceWriter);

impl PcmSink for SinkWriter {
    fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError> {
        self.0.push(samples, sample_rate, channels)
    }

    fn cancel_probe(&self) -> CancelProbe {
        // A second writer of the same utterance shares its cancel flag.
        match tts_sink::writer(self.0.id()) {
            Ok(w) => Arc::new(move || w.is_cancelled()),
            Err(_) => Arc::new(|| true),
        }
    }
}

/// A sink error as the command's error text (`"cancelled"` for a cancelled utterance).
pub(crate) fn sink_error(e: SinkError) -> String {
    match e {
        SinkError::Cancelled => CANCELLED.into(),
        other => other.to_string(),
    }
}

/// What the hand-off thread did.
#[derive(Debug, Default)]
struct PushOutcome {
    frames: u64,
    rate: u32,
    first: Option<Duration>,
    last: Option<Duration>,
    error: Option<SinkError>,
}

/// Push every received block into `sink`, in ≤ [`PUSH_MS`] pieces, until the channel closes, a push
/// fails or `abort` is set.
fn push_loop(mut sink: Box<dyn PcmSink>, mut rx: mpsc::Receiver<Block>, started: Instant, abort: Arc<AtomicBool>) -> PushOutcome {
    let mut out = PushOutcome::default();
    while let Some(block) = rx.blocking_recv() {
        let ch = usize::from(block.channels.max(1));
        let whole = block.samples.len() / ch * ch;
        let step = (block.rate as usize * PUSH_MS / 1000).max(1) * ch;
        for piece in block.samples[..whole].chunks(step) {
            if abort.load(Ordering::Acquire) {
                return out;
            }
            if let Err(e) = sink.push(piece, block.rate, block.channels) {
                out.error = Some(e);
                return out;
            }
            out.frames += (piece.len() / ch) as u64;
            let now = started.elapsed();
            out.first.get_or_insert(now);
            out.last = Some(now);
            if out.rate == 0 {
                out.rate = block.rate;
            }
        }
    }
    out
}

/// Sets the flag when dropped (stops the hand-off thread when a request ends early).
struct AbortOnDrop(Arc<AtomicBool>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------------------------

/// Why a watched wait ended early.
#[derive(Debug, PartialEq, Eq)]
enum Stop {
    Cancelled,
    TimedOut,
}

/// The cancel sources of one stream: `openai_speech_cancel` and the sink utterance.
struct CancelWatch {
    flag: Arc<AtomicBool>,
    probe: CancelProbe,
}

impl CancelWatch {
    fn is_set(&self) -> bool {
        self.flag.load(Ordering::Acquire) || (self.probe)()
    }
}

/// Await `fut`, giving up at `deadline` or once `cancel` is set (checked every [`CANCEL_POLL`]).
async fn watched<F: Future>(fut: F, deadline: Option<Instant>, cancel: &CancelWatch) -> Result<F::Output, Stop> {
    tokio::pin!(fut);
    loop {
        if cancel.is_set() {
            return Err(Stop::Cancelled);
        }
        let now = Instant::now();
        let tick = match deadline {
            Some(d) if d <= now => return Err(Stop::TimedOut),
            Some(d) => (d - now).min(CANCEL_POLL),
            None => CANCEL_POLL,
        };
        tokio::select! {
            biased;
            out = &mut fut => return Ok(out),
            _ = tokio::time::sleep(tick) => {}
        }
    }
}

/// How the network side ended.
enum NetEnd {
    /// Every block was handed off.
    Done,
    /// The hand-off thread stopped (its outcome says why).
    SinkClosed,
}

/// Speak `spec` into the begun sink utterance `utterance_id`. Resolves once the last audio was
/// pushed; `Err("cancelled")` when cancelled (see the module docs).
pub(crate) async fn speak_http(spec: HttpSpeech, utterance_id: &str) -> Result<SpeakReport, String> {
    let id = utterance_id.trim();
    if id.is_empty() {
        return Err("utteranceId must not be empty".into());
    }
    if spec.text.trim().is_empty() {
        return Ok(SpeakReport { sample_rate: spec.pcm_rate, ..SpeakReport::default() });
    }
    let writer = tts_sink::writer(id).map_err(sink_error)?;
    speak_into(spec, id, Box::new(SinkWriter(writer))).await
}

/// [`speak_http`] into any [`PcmSink`] (registered for cancellation under `id`).
pub(crate) async fn speak_into(spec: HttpSpeech, id: &str, sink: Box<dyn PcmSink>) -> Result<SpeakReport, String> {
    let registration = register(id)?;
    let started = Instant::now();
    let result = run_speech(spec, sink, Arc::clone(&registration.flag), started).await;
    drop(registration);
    result
}

async fn run_speech(spec: HttpSpeech, sink: Box<dyn PcmSink>, flag: Arc<AtomicBool>, started: Instant) -> Result<SpeakReport, String> {
    let cancel = CancelWatch { flag, probe: sink.cancel_probe() };
    if cancel.is_set() {
        return Err(CANCELLED.into());
    }
    let timing = Timing::for_request(&spec.url, &spec.text, spec.timeout);
    let what = spec.what.clone();
    let url = spec.url.clone();
    let req = match (&spec.client, spec.timeout) {
        (Some(c), _) => c.post(url),
        // The shared synthesis request: the long-read client for a server on this PC or the LAN.
        (None, None) => tts_post(url, &spec.text).0,
        (None, Some(_)) if timing.long() => local_http_client().post(url),
        (None, Some(_)) => http_client().post(url),
    };
    // Network waits are timed here; the reqwest timeout is only an outer bound (see `Timing`).
    let mut req = req.json(&spec.body).timeout(timing.outer());
    if let Some(b) = &spec.bearer {
        req = req.header(AUTHORIZATION, b.clone());
    }
    for (name, value) in &spec.headers {
        req = req.header(name.clone(), value.clone());
    }

    let abort = Arc::new(AtomicBool::new(false));
    let _abort_on_return = AbortOnDrop(Arc::clone(&abort));
    let (tx, rx) = mpsc::channel::<Block>(CHANNEL_BLOCKS);
    let pusher = {
        let abort = Arc::clone(&abort);
        tokio::task::spawn_blocking(move || push_loop(sink, rx, started, abort))
    };

    let net = stream_body(req, &spec, timing, started, &cancel, tx).await?;
    // Every block is queued for the hand-off (or it stopped): wait for the last push.
    let outcome = match watched(pusher, None, &cancel).await {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(e)) => return Err(format!("{what}: the audio hand-off failed ({e}).")),
        Err(_) => return Err(CANCELLED.into()),
    };
    if let Some(e) = outcome.error {
        return Err(sink_error(e));
    }
    if matches!(net, NetEnd::SinkClosed) && cancel.is_set() {
        return Err(CANCELLED.into());
    }
    if outcome.frames == 0 {
        return Err(format!("{what}: the server returned no audio."));
    }
    let first = outcome.first.unwrap_or_default();
    let last = outcome.last.unwrap_or(first);
    tracing::debug!(
        target: "sigil::tts",
        "{what} ({}): {} frames at {} Hz, first audio after {} ms, done after {} ms",
        redact_url(&spec.url),
        outcome.frames,
        outcome.rate,
        first.as_millis(),
        last.as_millis()
    );
    Ok(SpeakReport {
        sample_rate: outcome.rate,
        frames: outcome.frames,
        first_audio_ms: ms(first),
        total_ms: ms(last),
    })
}

fn ms(d: Duration) -> u32 {
    d.as_millis().min(u128::from(u32::MAX)) as u32
}

/// Send the request, decode the body as it arrives and hand the blocks off through `tx`.
async fn stream_body(
    req: reqwest::RequestBuilder,
    spec: &HttpSpeech,
    timing: Timing,
    started: Instant,
    cancel: &CancelWatch,
    tx: mpsc::Sender<Block>,
) -> Result<NetEnd, String> {
    let what = spec.what.as_str();
    let target = redact_url(&spec.url);
    let answer_deadline = started + timing.answer;
    let budget_deadline = started + timing.budget();

    let resp = match watched(req.send(), Some(answer_deadline), cancel).await {
        Err(Stop::Cancelled) => return Err(CANCELLED.into()),
        Err(Stop::TimedOut) => {
            return Err(format!("{what} ({target}): no response within {} s (timed out).", secs(timing.answer)))
        }
        Ok(Err(e)) => return Err(describe_request_error(what, &e, timing.answer)),
        Ok(Ok(resp)) => resp,
    };
    if !resp.status().is_success() {
        return Err(describe_api_error(what, resp).await);
    }
    let ct = content_type(&resp);
    let kind = ContentKind::parse(ct.as_deref());
    if kind == ContentKind::Textual {
        let prefix = read_text_prefix(resp).await;
        let summary = error_body_summary(&prefix).unwrap_or_default();
        return Err(format!(
            "{what}: expected audio but got {}{}{summary}",
            ct.as_deref().unwrap_or("a text response"),
            if summary.is_empty() { "" } else { ": " },
        ));
    }

    let mut decoder = AudioDecoder::new(spec.expect, spec.pcm_rate, kind, MAX_AUDIO_BYTES);
    let mut resp = resp;
    // When the last audio was handed off (after any wait for the hand-off): the stall limit counts
    // from there, like the sink's own stall timeout (body chunks without audio do not reset it).
    let mut last_audio: Option<Instant> = None;
    loop {
        let limit = match last_audio {
            Some(at) => at + timing.stall,
            None => answer_deadline,
        };
        let chunk = match watched(resp.chunk(), Some(limit.min(budget_deadline)), cancel).await {
            Err(Stop::Cancelled) => return Err(CANCELLED.into()),
            Err(Stop::TimedOut) => {
                return Err(if Instant::now() >= budget_deadline {
                    format!("{what}: the audio took longer than {} s to arrive.", secs(timing.budget()))
                } else if last_audio.is_some() {
                    format!("{what}: the server stopped sending audio for {} s (stalled).", secs(timing.stall))
                } else {
                    format!("{what} ({target}): no audio within {} s (timed out).", secs(timing.answer))
                })
            }
            Ok(Err(e)) if e.is_timeout() => return Err(format!("{what}: timed out while receiving the audio.")),
            Ok(Err(e)) => return Err(describe_request_error(what, &e, timing.stall)),
            Ok(Ok(None)) => break,
            Ok(Ok(Some(bytes))) => bytes,
        };
        let blocks = decoder.feed(&chunk).map_err(|e| format!("{what}: {e}"))?;
        for block in blocks {
            if block.samples.is_empty() {
                continue;
            }
            if !hand_off(&tx, block, cancel).await? {
                return Ok(NetEnd::SinkClosed);
            }
            last_audio = Some(Instant::now());
        }
    }
    let (blocks, compressed) = decoder.finish().map_err(|e| format!("{what}: {e}"))?;
    for block in blocks {
        if !hand_off(&tx, block, cancel).await? {
            return Ok(NetEnd::SinkClosed);
        }
    }
    if let Some(bytes) = compressed {
        let decoded = match watched(tokio::task::spawn_blocking(move || decode_compressed(bytes)), None, cancel).await {
            Err(_) => return Err(CANCELLED.into()),
            Ok(Err(e)) => return Err(format!("{what}: decoding failed ({e}).")),
            Ok(Ok(result)) => result.map_err(|e| format!("{what}: {e}"))?,
        };
        if !hand_off(&tx, decoded, cancel).await? {
            return Ok(NetEnd::SinkClosed);
        }
    }
    Ok(NetEnd::Done)
}

/// Queue `block` for the hand-off thread. `Ok(false)` when that thread stopped.
async fn hand_off(tx: &mpsc::Sender<Block>, block: Block, cancel: &CancelWatch) -> Result<bool, String> {
    if block.samples.is_empty() {
        return Ok(true);
    }
    match watched(tx.send(block), None, cancel).await {
        Err(_) => Err(CANCELLED.into()),
        Ok(Err(_)) => Ok(false),
        Ok(Ok(())) => Ok(true),
    }
}

/// Up to [`TEXT_BODY_LIMIT`] bytes of a text answer, within [`TEXT_BODY_TIMEOUT`].
async fn read_text_prefix(mut resp: Response) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(TEXT_BODY_TIMEOUT, async {
        while buf.len() < TEXT_BODY_LIMIT {
            match resp.chunk().await {
                Ok(Some(c)) => buf.extend_from_slice(&c[..c.len().min(TEXT_BODY_LIMIT - buf.len())]),
                _ => break,
            }
        }
    })
    .await;
    buf
}

// ---------------------------------------------------------------------------------------------
// Decoding
// ---------------------------------------------------------------------------------------------

/// Decoded interleaved samples.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Block {
    pub samples: Vec<f32>,
    pub rate: u32,
    pub channels: u16,
}

/// What the response's Content-Type says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ContentKind {
    /// Missing, `application/octet-stream` or unknown: the requested format decides.
    Unknown,
    /// JSON / text / HTML: an error message, not audio.
    Textual,
    Wav,
    /// Raw PCM, with the `rate=` parameter when present.
    Pcm(Option<u32>),
    /// MP3, AAC, Ogg, Opus, FLAC, WebM, MP4.
    Compressed,
}

impl ContentKind {
    fn parse(ct: Option<&str>) -> Self {
        let Some(ct) = ct else {
            return Self::Unknown;
        };
        let lower = ct.to_ascii_lowercase();
        let mut parts = lower.split(';').map(str::trim);
        let mime = parts.next().unwrap_or("");
        if mime.starts_with("text/") || mime == "application/json" || mime.ends_with("+json") || mime.ends_with("/xml") {
            return Self::Textual;
        }
        match mime {
            "audio/wav" | "audio/x-wav" | "audio/wave" | "audio/vnd.wave" | "audio/x-pn-wav" => Self::Wav,
            "audio/pcm" | "audio/l16" | "audio/raw" | "audio/x-raw" | "audio/s16le" => {
                let rate = parts.find_map(|p| {
                    let (k, v) = p.split_once('=')?;
                    matches!(k.trim(), "rate" | "samplerate" | "sample_rate")
                        .then(|| v.trim().trim_matches('"').parse::<u32>().ok())
                        .flatten()
                        .filter(|r| PCM_RATE_RANGE.contains(r))
                });
                Self::Pcm(rate)
            }
            "audio/mpeg" | "audio/mp3" | "audio/aac" | "audio/aacp" | "audio/ogg" | "audio/opus" | "audio/flac"
            | "audio/x-flac" | "audio/webm" | "audio/mp4" | "audio/x-m4a" => Self::Compressed,
            _ => Self::Unknown,
        }
    }
}

/// Sample encodings of a WAV data chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SampleType {
    U8,
    I16,
    I24,
    I32,
    F32,
    F64,
}

impl SampleType {
    fn from_fmt(format: u16, bits: u16) -> Result<Self, String> {
        match (format, bits) {
            (WAVE_FORMAT_PCM, 8) => Ok(Self::U8),
            (WAVE_FORMAT_PCM, 16) => Ok(Self::I16),
            (WAVE_FORMAT_PCM, 24) => Ok(Self::I24),
            (WAVE_FORMAT_PCM, 32) => Ok(Self::I32),
            (WAVE_FORMAT_FLOAT, 32) => Ok(Self::F32),
            (WAVE_FORMAT_FLOAT, 64) => Ok(Self::F64),
            _ => Err(format!("WAV format {format} with {bits}-bit samples is not supported (PCM or float only).")),
        }
    }

    fn bytes(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::I16 => 2,
            Self::I24 => 3,
            Self::I32 | Self::F32 => 4,
            Self::F64 => 8,
        }
    }

    fn read(self, b: &[u8]) -> f32 {
        let v = match self {
            Self::U8 => (f32::from(b[0]) - 128.0) / 128.0,
            Self::I16 => f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0,
            Self::I24 => (i32::from_le_bytes([0, b[0], b[1], b[2]]) >> 8) as f32 / 8_388_608.0,
            Self::I32 => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32 / 2_147_483_648.0,
            Self::F32 => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
            Self::F64 => f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32,
        };
        if v.is_finite() {
            v
        } else {
            0.0
        }
    }
}

const WAVE_FORMAT_PCM: u16 = 1;
const WAVE_FORMAT_FLOAT: u16 = 3;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
/// Bytes needed to tell the body's format.
const SNIFF_BYTES: usize = 12;
/// Largest header (everything before `data`) accepted.
const MAX_WAV_HEADER: usize = 64 * 1024;
/// Channels the sink accepts.
const MAX_CHANNELS: u16 = 16;

/// Format of a WAV data chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct WavFmt {
    rate: u32,
    channels: u16,
    sample: SampleType,
}

impl WavFmt {
    fn frame_bytes(&self) -> usize {
        self.sample.bytes() * usize::from(self.channels)
    }
}

fn read_u16(b: &[u8], at: usize) -> u16 {
    u16::from_le_bytes([b[at], b[at + 1]])
}

fn read_u32(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn parse_fmt(b: &[u8]) -> Result<WavFmt, String> {
    if b.len() < 16 {
        return Err("the WAV fmt chunk is truncated.".into());
    }
    let mut format = read_u16(b, 0);
    let channels = read_u16(b, 2);
    let rate = read_u32(b, 4);
    let bits = read_u16(b, 14);
    if format == WAVE_FORMAT_EXTENSIBLE && b.len() >= 26 {
        // The sub-format GUID starts with the plain format code.
        format = read_u16(b, 24);
    }
    if channels == 0 || channels > MAX_CHANNELS {
        return Err(format!("WAV audio with {channels} channels is not supported."));
    }
    if !(4_000..=384_000).contains(&rate) {
        return Err(format!("WAV audio at {rate} Hz is not supported."));
    }
    Ok(WavFmt { rate, channels, sample: SampleType::from_fmt(format, bits)? })
}

/// Parse a WAV header at the start of `buf`: `Ok(None)` = need more bytes, else the format, the
/// offset of the audio and the declared data size.
fn parse_wav_header(buf: &[u8]) -> Result<Option<(WavFmt, usize, u32)>, String> {
    if buf.len() < SNIFF_BYTES {
        return Ok(None);
    }
    if &buf[0..4] != b"RIFF" || &buf[8..12] != b"WAVE" {
        return Err("expected a WAV (RIFF/WAVE) stream.".into());
    }
    let mut pos = SNIFF_BYTES;
    let mut fmt = None;
    loop {
        if buf.len() < pos + 8 {
            return Ok(None);
        }
        let id = &buf[pos..pos + 4];
        let size = read_u32(buf, pos + 4);
        let body = pos + 8;
        if id == b"data" {
            let fmt = fmt.ok_or_else(|| "the WAV data chunk comes before its fmt chunk.".to_string())?;
            return Ok(Some((fmt, body, size)));
        }
        let size = size as usize;
        if body + size > MAX_WAV_HEADER {
            return Err("the WAV header is too large (no audio data found).".into());
        }
        if buf.len() < body + size {
            return Ok(None);
        }
        if id == b"fmt " {
            fmt = Some(parse_fmt(&buf[body..body + size])?);
        }
        // Chunks are word-aligned; a missing pad byte at the very end is tolerated by the check above.
        pos = body + size + (size & 1);
    }
}

/// Whether `p` looks like a text answer (JSON / HTML) rather than audio.
fn looks_textual(p: &[u8]) -> bool {
    let head = &p[..p.len().min(64)];
    let Some(first) = head.iter().find(|b| !b.is_ascii_whitespace()) else {
        return false;
    };
    if !matches!(first, b'{' | b'[' | b'<') {
        return false;
    }
    let text = match std::str::from_utf8(head) {
        Ok(t) => t,
        Err(e) if e.error_len().is_none() => std::str::from_utf8(&head[..e.valid_up_to()]).unwrap_or(""),
        Err(_) => return false,
    };
    text.chars().all(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
}

/// MPEG audio frame sync (MP3 / AAC ADTS without an ID3 tag).
fn is_mpeg_sync(p: &[u8]) -> bool {
    p.len() >= 2 && p[0] == 0xFF && (p[1] & 0xE0) == 0xE0
}

/// Bytes an ADTS header needs to tell its frame length.
const ADTS_HEADER: usize = 7;

/// Length of the MPEG audio (Layer I–III, MPEG 1 / 2 / 2.5) or AAC ADTS frame whose header starts
/// `h`, or `None` when `h` does not start with a valid header (reserved or free-format values).
fn mpeg_frame_len(h: &[u8]) -> Option<usize> {
    if h.len() < 4 || !is_mpeg_sync(h) {
        return None;
    }
    let layer = (h[1] >> 1) & 3;
    if layer == 0 {
        // ADTS: 12-bit sync, layer 00, sampling index <= 12, 13-bit frame length (header included).
        if h.len() < ADTS_HEADER || h[1] & 0xF0 != 0xF0 || (h[2] >> 2) & 0x0F > 12 {
            return None;
        }
        let len = (usize::from(h[3] & 3) << 11) | (usize::from(h[4]) << 3) | usize::from(h[5] >> 5);
        return (len > ADTS_HEADER).then_some(len);
    }
    // Version: 0 = MPEG 2.5, 1 = reserved, 2 = MPEG 2, 3 = MPEG 1. Layer: 1 = III, 2 = II, 3 = I.
    let version = (h[1] >> 3) & 3;
    let bitrate_index = usize::from(h[2] >> 4);
    let rate_index = usize::from((h[2] >> 2) & 3);
    if version == 1 || bitrate_index == 0 || bitrate_index == 15 || rate_index == 3 || h[3] & 3 == 2 {
        return None;
    }
    const V1_L1: [u32; 14] = [32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448];
    const V1_L2: [u32; 14] = [32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384];
    const V1_L3: [u32; 14] = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320];
    const V2_L1: [u32; 14] = [32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256];
    const V2_L23: [u32; 14] = [8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160];
    let v1 = version == 3;
    let table = match (v1, layer) {
        (true, 3) => &V1_L1,
        (true, 2) => &V1_L2,
        (true, _) => &V1_L3,
        (false, 3) => &V2_L1,
        (false, _) => &V2_L23,
    };
    let bitrate = table[bitrate_index - 1] * 1000;
    let rate = [44_100u32, 48_000, 32_000][rate_index] >> (3 - u32::from(version)).min(2);
    let padding = u32::from((h[2] >> 1) & 1);
    let len = match layer {
        3 => (12 * bitrate / rate + padding) * 4,
        2 => 144 * bitrate / rate + padding,
        _ if v1 => 144 * bitrate / rate + padding,
        _ => 72 * bitrate / rate + padding,
    };
    Some(len as usize)
}

/// What the start of a body says about headerless MPEG audio.
#[derive(Debug, PartialEq, Eq)]
enum MpegSniff {
    /// Two consecutive frame headers of the same stream: MP3 / MP2 / ADTS AAC.
    Yes,
    No,
    /// A valid first header: the next one decides.
    NeedMore,
}

/// Whether `p` (the start of a body) is MPEG audio without an ID3 tag. One valid header is not
/// enough (raw s16 PCM starting with a sample like -1 has the frame sync), so the next frame's
/// header must follow at the first frame's length, with the same version, layer and sample rate.
/// At the end of the body a single frame (~26 ms) is not MPEG audio.
fn sniff_mpeg(p: &[u8], at_end: bool) -> MpegSniff {
    let Some(len) = mpeg_frame_len(p) else {
        return MpegSniff::No;
    };
    if p.len() < len + ADTS_HEADER {
        return if at_end { MpegSniff::No } else { MpegSniff::NeedMore };
    }
    let next = &p[len..];
    let adts = (p[1] >> 1) & 3 == 0;
    // Same version and layer; same sampling index (ADTS: 4 bits, MPEG: 2 bits).
    let rate_mask = if adts { 0x3C } else { 0x0C };
    let same = mpeg_frame_len(next).is_some() && next[1] & 0xFE == p[1] & 0xFE && next[2] & rate_mask == p[2] & rate_mask;
    if same {
        MpegSniff::Yes
    } else {
        MpegSniff::No
    }
}

#[derive(Debug)]
enum Stage {
    /// Waiting for [`SNIFF_BYTES`] to tell the format.
    Sniff,
    /// Collecting a WAV header.
    WavHeader,
    /// Inside a WAV data chunk (`remaining` = `None` for a streamed WAV of unknown length).
    WavData { fmt: WavFmt, remaining: Option<u64>, odd: bool },
    /// After a sized data chunk: another WAV may follow; anything else is ignored.
    WavTail { skip_pad: bool, junk: bool },
    /// Raw s16le mono.
    Pcm16 { rate: u32 },
    /// Buffered until the end (MP3, Ogg, FLAC, ...).
    Compressed,
}

/// Incremental decoder of a speech response body (see the module docs).
pub(crate) struct AudioDecoder {
    stage: Stage,
    /// Bytes not consumed yet.
    buf: Vec<u8>,
    expect: SpeechFormat,
    pcm_rate: u32,
    kind: ContentKind,
    received: usize,
    max_bytes: usize,
}

impl AudioDecoder {
    fn new(expect: SpeechFormat, pcm_rate: u32, kind: ContentKind, max_bytes: usize) -> Self {
        Self { stage: Stage::Sniff, buf: Vec::new(), expect, pcm_rate, kind, received: 0, max_bytes }
    }

    /// Decode the next body chunk; returns the complete frames it finished.
    fn feed(&mut self, bytes: &[u8]) -> Result<Vec<Block>, String> {
        self.received = self.received.saturating_add(bytes.len());
        if self.received > self.max_bytes {
            return Err(format!("the response is larger than {} MB.", self.max_bytes.div_ceil(1024 * 1024)));
        }
        self.buf.extend_from_slice(bytes);
        let mut out = Vec::new();
        self.drain(&mut out, false)?;
        Ok(out)
    }

    /// The body ended: the last frames, plus the whole body when it must be decoded as a file.
    /// A partial last frame is dropped.
    fn finish(mut self) -> Result<(Vec<Block>, Option<Vec<u8>>), String> {
        let mut out = Vec::new();
        if self.received == 0 {
            return Ok((out, None));
        }
        self.drain(&mut out, true)?;
        match self.stage {
            Stage::WavHeader => Err("the WAV stream ended inside its header.".into()),
            Stage::Compressed => Ok((out, Some(std::mem::take(&mut self.buf)))),
            _ => Ok((out, None)),
        }
    }

    /// The stage for the body's first bytes; `Ok(None)` = more bytes are needed (never at the end).
    fn classify(&self, at_end: bool) -> Result<Option<Stage>, String> {
        let p = self.buf.as_slice();
        if p.starts_with(b"RIFF") {
            return if p.len() >= SNIFF_BYTES && &p[8..12] == b"WAVE" {
                Ok(Some(Stage::WavHeader))
            } else if p.len() >= SNIFF_BYTES {
                Err("the server sent a RIFF file that is not WAV audio.".into())
            } else {
                Err("the WAV stream ended inside its header.".into())
            };
        }
        if p.starts_with(b"ID3") || p.starts_with(b"OggS") || p.starts_with(b"fLaC") || self.kind == ContentKind::Compressed {
            return Ok(Some(Stage::Compressed));
        }
        if looks_textual(p) {
            let summary = error_body_summary(p).unwrap_or_default();
            return Err(format!("expected audio but got a text response: {summary}"));
        }
        let wants_wav = self.kind == ContentKind::Wav || (self.expect == SpeechFormat::Wav && !matches!(self.kind, ContentKind::Pcm(_)));
        if wants_wav {
            if is_mpeg_sync(p) {
                return Ok(Some(Stage::Compressed));
            }
            return Err("expected a WAV stream but the server sent no WAV header. Choose PCM output (with the server's sample rate) for this server.".into());
        }
        // Raw PCM asked, and the Content-Type does not say PCM: a server that ignores
        // `response_format` may answer MP3 / AAC frames without an ID3 tag (played as s16 they would
        // be loud noise). A PCM Content-Type is trusted.
        if self.kind == ContentKind::Unknown {
            match sniff_mpeg(p, at_end) {
                MpegSniff::Yes => return Ok(Some(Stage::Compressed)),
                MpegSniff::NeedMore => return Ok(None),
                MpegSniff::No => {}
            }
        }
        let rate = match self.kind {
            ContentKind::Pcm(Some(rate)) => rate,
            _ => self.pcm_rate,
        };
        Ok(Some(Stage::Pcm16 { rate }))
    }

    fn drain(&mut self, out: &mut Vec<Block>, at_end: bool) -> Result<(), String> {
        loop {
            match &mut self.stage {
                Stage::Sniff => {
                    if self.buf.len() < SNIFF_BYTES && !at_end {
                        return Ok(());
                    }
                    match self.classify(at_end)? {
                        Some(stage) => self.stage = stage,
                        None => return Ok(()),
                    }
                }
                Stage::WavHeader => match parse_wav_header(&self.buf)? {
                    None => {
                        if self.buf.len() > MAX_WAV_HEADER {
                            return Err("the WAV header is too large (no audio data found).".into());
                        }
                        return Ok(());
                    }
                    Some((fmt, offset, size)) => {
                        self.buf.drain(..offset);
                        // Streamed WAVs declare 0 or 0xFFFFFFFF: read to the end of the body.
                        let remaining = (size != 0 && size != u32::MAX).then_some(u64::from(size));
                        self.stage = Stage::WavData { fmt, remaining, odd: size & 1 == 1 };
                    }
                },
                Stage::WavData { fmt, remaining, odd } => {
                    let fmt = *fmt;
                    let frame = fmt.frame_bytes();
                    let avail = match *remaining {
                        Some(r) => (self.buf.len() as u64).min(r) as usize,
                        None => self.buf.len(),
                    };
                    let whole = avail - avail % frame;
                    if whole > 0 {
                        let samples = self.buf[..whole].chunks_exact(fmt.sample.bytes()).map(|b| fmt.sample.read(b)).collect();
                        out.push(Block { samples, rate: fmt.rate, channels: fmt.channels });
                        self.buf.drain(..whole);
                        if let Some(r) = remaining.as_mut() {
                            *r -= whole as u64;
                        }
                    }
                    // A declared size that is not whole frames: drop the rest of it.
                    if let Some(r) = *remaining {
                        if r > 0 && r < frame as u64 && self.buf.len() as u64 >= r {
                            self.buf.drain(..r as usize);
                            *remaining = Some(0);
                        }
                    }
                    if *remaining == Some(0) {
                        self.stage = Stage::WavTail { skip_pad: *odd, junk: false };
                        continue;
                    }
                    return Ok(());
                }
                Stage::WavTail { skip_pad, junk } => {
                    if *junk {
                        self.buf.clear();
                        return Ok(());
                    }
                    if *skip_pad && !self.buf.is_empty() {
                        if self.buf[0] == 0 {
                            self.buf.drain(..1);
                        }
                        *skip_pad = false;
                    }
                    if self.buf.len() < SNIFF_BYTES && !at_end {
                        return Ok(());
                    }
                    if self.buf.len() >= SNIFF_BYTES && self.buf.starts_with(b"RIFF") && &self.buf[8..12] == b"WAVE" {
                        // Another WAV follows (servers that send one WAV per chunk).
                        self.stage = Stage::WavHeader;
                        continue;
                    }
                    // Trailing metadata (LIST, id3 ...): ignored.
                    *junk = true;
                    self.buf.clear();
                    return Ok(());
                }
                Stage::Pcm16 { rate } => {
                    let whole = self.buf.len() & !1;
                    if whole > 0 {
                        let samples = self.buf[..whole]
                            .as_chunks::<2>().0.iter()
                            .map(|b| f32::from(i16::from_le_bytes([b[0], b[1]])) / 32_768.0)
                            .collect();
                        out.push(Block { samples, rate: *rate, channels: 1 });
                        self.buf.drain(..whole);
                    }
                    return Ok(());
                }
                Stage::Compressed => return Ok(()),
            }
        }
    }
}

/// Decode a complete compressed clip (MP3 / Ogg Vorbis / FLAC / WAV) with rodio.
fn decode_compressed(bytes: Vec<u8>) -> Result<Block, String> {
    use rodio::Source;
    let decoder = rodio::Decoder::new(Cursor::new(bytes)).map_err(|e| format!("the audio cannot be decoded ({e})."))?;
    let rate = decoder.sample_rate();
    let channels = decoder.channels().max(1);
    let samples: Vec<f32> = decoder.map(|s| f32::from(s) / 32_768.0).collect();
    Ok(Block { samples, rate, channels })
}

// ---------------------------------------------------------------------------------------------
// Server checks
// ---------------------------------------------------------------------------------------------

fn with_bearer(req: reqwest::RequestBuilder, key: Option<&str>) -> Result<reqwest::RequestBuilder, String> {
    Ok(match bearer_header(key)? {
        Some(v) => req.header(AUTHORIZATION, v),
        None => req,
    })
}

/// Whether the server at `base_url` answers (`GET {base}/models`): 2xx or a client error such as
/// 404 (it answered, it just has no model list) = reachable; 5xx or no answer = not; 401/403 =
/// `Err` naming the rejected key.
async fn check_with(client: &Client, base_url: &str, api_key: Option<&str>) -> Result<bool, String> {
    let base = speech_base(base_url)?;
    let url = api_url(&base, "/models")?;
    let what = "Speech server check";
    match probe(what, with_bearer(client.get(url), api_key)?).await {
        Err(_) => Ok(false),
        Ok(status) if is_auth_failure(status) => Err(format!(
            "The speech server at {} rejected the API key (HTTP {status}). Check the key.",
            redact_url(&Url::parse(&base).map_err(|e| e.to_string())?)
        )),
        Ok(status) if status.is_server_error() => {
            tracing::info!(target: "sigil::http", "{what}: HTTP {status}");
            Ok(false)
        }
        Ok(_) => Ok(true),
    }
}

/// Voice ids of the server at `base_url`: `GET {base}/audio/voices`, then `GET {base}/voices`; `[]`
/// when neither route exists or answers with something other than a list.
async fn list_voices_with(client: &Client, base_url: &str, api_key: Option<&str>) -> Result<Vec<String>, String> {
    let base = speech_base(base_url)?;
    let what = "Speech server voice list";
    for path in ["/audio/voices", "/voices"] {
        let url = api_url(&base, path)?;
        let resp = with_bearer(client.get(url), api_key)?
            .timeout(LIST_TIMEOUT)
            .send()
            .await
            .map_err(|e| describe_request_error(what, &e, LIST_TIMEOUT))?;
        let status = resp.status();
        if matches!(status, StatusCode::NOT_FOUND | StatusCode::METHOD_NOT_ALLOWED) {
            continue;
        }
        if !status.is_success() {
            return Err(describe_api_error(what, resp).await);
        }
        let Ok(v) = resp.json::<Value>().await else {
            tracing::info!(target: "sigil::http", "{what}: {path} is not a voice list");
            return Ok(Vec::new());
        };
        return Ok(parse_named_list(
            &v,
            &["voices", "data", "items", "results"],
            &["id", "voice_id", "voice", "name"],
            &["name"],
        )
        .into_iter()
        .map(|e| e.id)
        .collect());
    }
    Ok(Vec::new())
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

#[command]
async fn openai_speech_speak(request: OpenAiSpeechRequest, utterance_id: String) -> Result<SpeakReport, String> {
    let spec = openai_spec(&request)?;
    speak_http(spec, &utterance_id).await
}

#[command]
async fn openai_speech_cancel(utterance_id: Option<String>) -> Result<(), String> {
    match utterance_id.as_deref().map(str::trim) {
        None => cancel(None),
        Some("") => {}
        Some(id) => cancel(Some(id)),
    }
    Ok(())
}

#[command]
async fn openai_speech_check(base_url: String, api_key: Option<String>) -> Result<bool, String> {
    check_with(http_client(), &base_url, api_key.as_deref()).await
}

#[command]
async fn openai_speech_list_voices(base_url: String, api_key: Option<String>) -> Result<Vec<String>, String> {
    list_voices_with(http_client(), &base_url, api_key.as_deref()).await
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            openai_speech_speak,
            openai_speech_cancel,
            openai_speech_check,
            openai_speech_list_voices,
        ])
        .build()
}

// ---------------------------------------------------------------------------------------------
// Test support (also used by the legacy HTTP plugins' tests)
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
pub(crate) mod testing {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    /// A client that ignores system proxies (loopback test servers).
    pub(crate) fn client() -> Client {
        Client::builder().no_proxy().build().unwrap()
    }

    /// Stream `spec` into `sink` with the test client.
    pub(crate) async fn speak(spec: HttpSpeech, id: &str, sink: Box<dyn PcmSink>) -> Result<SpeakReport, String> {
        speak_into(spec.with_client(client()), id, sink).await
    }

    /// One push: samples, rate, channels.
    pub(crate) type Push = (Vec<f32>, u32, u16);

    /// Every push a [`Recorder`] received.
    #[derive(Clone, Default)]
    pub(crate) struct Pushes(pub Arc<Mutex<Vec<Push>>>);

    impl Pushes {
        pub(crate) fn samples(&self) -> Vec<f32> {
            self.0.lock().unwrap().iter().flat_map(|(s, _, _)| s.clone()).collect()
        }
        pub(crate) fn count(&self) -> usize {
            self.0.lock().unwrap().len()
        }
        pub(crate) fn formats(&self) -> Vec<(u32, u16)> {
            self.0.lock().unwrap().iter().map(|(_, r, c)| (*r, *c)).collect()
        }
    }

    /// A sink double: records pushes, can be cancelled, can fail, signals the first push.
    pub(crate) struct Recorder {
        pub pushes: Pushes,
        pub cancelled: Arc<AtomicBool>,
        pub fail_with: Option<SinkError>,
        pub on_first: Option<tokio::sync::oneshot::Sender<()>>,
        /// Cancel the utterance right after the first push.
        pub cancel_after_first: bool,
    }

    impl Recorder {
        pub(crate) fn new() -> (Self, Pushes) {
            let pushes = Pushes::default();
            (
                Self {
                    pushes: pushes.clone(),
                    cancelled: Arc::new(AtomicBool::new(false)),
                    fail_with: None,
                    on_first: None,
                    cancel_after_first: false,
                },
                pushes,
            )
        }
    }

    impl PcmSink for Recorder {
        fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError> {
            if self.cancelled.load(Ordering::Acquire) {
                return Err(SinkError::Cancelled);
            }
            if let Some(e) = self.fail_with.clone() {
                return Err(e);
            }
            self.pushes.0.lock().unwrap().push((samples.to_vec(), sample_rate, channels));
            if let Some(tx) = self.on_first.take() {
                let _ = tx.send(());
            }
            if self.cancel_after_first {
                self.cancelled.store(true, Ordering::Release);
            }
            Ok(())
        }

        fn cancel_probe(&self) -> CancelProbe {
            let flag = Arc::clone(&self.cancelled);
            Arc::new(move || flag.load(Ordering::Acquire))
        }
    }

    /// A request the test server received.
    #[derive(Clone, Debug, Default)]
    pub(crate) struct Seen {
        pub head: String,
        pub body: Vec<u8>,
    }

    impl Seen {
        pub(crate) fn path(&self) -> String {
            self.head.split_whitespace().nth(1).unwrap_or("").to_string()
        }
        pub(crate) fn header(&self, name: &str) -> Option<String> {
            let prefix = format!("{}:", name.to_ascii_lowercase());
            self.head
                .lines()
                .find(|l| l.to_ascii_lowercase().starts_with(&prefix))
                .map(|l| l[prefix.len()..].trim().to_string())
        }
        pub(crate) fn json(&self) -> Value {
            serde_json::from_slice(&self.body).unwrap()
        }
    }

    /// Read one request (head and a Content-Length body).
    pub(crate) async fn read_request(sock: &mut TcpStream) -> Seen {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..end]).to_string();
                let len = head
                    .lines()
                    .find_map(|l| {
                        let (k, v) = l.split_once(':')?;
                        k.trim().eq_ignore_ascii_case("content-length").then(|| v.trim().parse::<usize>().ok())?
                    })
                    .unwrap_or(0);
                while buf.len() < end + 4 + len {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                let body = buf[end + 4..(end + 4 + len).min(buf.len())].to_vec();
                return Seen { head, body };
            }
            let n = sock.read(&mut tmp).await.unwrap_or(0);
            if n == 0 {
                return Seen { head: String::from_utf8_lossy(&buf).to_string(), body: Vec::new() };
            }
            buf.extend_from_slice(&tmp[..n]);
        }
    }

    /// A bound loopback listener and its base URL.
    pub(crate) async fn listener() -> (TcpListener, String) {
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", l.local_addr().unwrap());
        (l, base)
    }

    /// Response head for a chunked body.
    pub(crate) fn chunked_head(content_type: &str) -> Vec<u8> {
        format!("HTTP/1.1 200 OK\r\nContent-Type: {content_type}\r\nTransfer-Encoding: chunked\r\n\r\n").into_bytes()
    }

    /// One chunk of a chunked body.
    pub(crate) fn chunk(data: &[u8]) -> Vec<u8> {
        let mut out = format!("{:x}\r\n", data.len()).into_bytes();
        out.extend_from_slice(data);
        out.extend_from_slice(b"\r\n");
        out
    }

    /// The last chunk of a chunked body.
    pub(crate) const LAST_CHUNK: &[u8] = b"0\r\n\r\n";

    /// A complete response with `status`, `content_type` and `body`.
    pub(crate) fn full_response(status: &str, content_type: &str, body: &[u8]) -> Vec<u8> {
        let mut out = format!(
            "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        )
        .into_bytes();
        out.extend_from_slice(body);
        out
    }

    /// Serve one connection: answer every request with `response`, then close.
    pub(crate) async fn serve_once(response: Vec<u8>) -> (String, tokio::task::JoinHandle<Seen>) {
        let (l, base) = listener().await;
        let task = tokio::spawn(async move {
            let (mut sock, _) = l.accept().await.unwrap();
            let seen = read_request(&mut sock).await;
            let _ = sock.write_all(&response).await;
            let _ = sock.shutdown().await;
            seen
        });
        (base, task)
    }

    /// Little-endian s16 bytes.
    pub(crate) fn pcm16(samples: &[i16]) -> Vec<u8> {
        samples.iter().flat_map(|s| s.to_le_bytes()).collect()
    }

    /// A WAV header for `format` / `bits` with `data_size` declared (use 0 or u32::MAX for streams).
    pub(crate) fn wav_header(rate: u32, channels: u16, format: u16, bits: u16, data_size: u32) -> Vec<u8> {
        let block = channels * bits / 8;
        let mut h = Vec::new();
        h.extend_from_slice(b"RIFF");
        h.extend_from_slice(&data_size.saturating_add(36).to_le_bytes());
        h.extend_from_slice(b"WAVEfmt ");
        h.extend_from_slice(&16u32.to_le_bytes());
        h.extend_from_slice(&format.to_le_bytes());
        h.extend_from_slice(&channels.to_le_bytes());
        h.extend_from_slice(&rate.to_le_bytes());
        h.extend_from_slice(&(rate * u32::from(block)).to_le_bytes());
        h.extend_from_slice(&block.to_le_bytes());
        h.extend_from_slice(&bits.to_le_bytes());
        h.extend_from_slice(b"data");
        h.extend_from_slice(&data_size.to_le_bytes());
        h
    }

    /// A complete 16-bit PCM WAV.
    pub(crate) fn wav16(samples: &[i16], rate: u32, channels: u16) -> Vec<u8> {
        let data = pcm16(samples);
        let mut w = wav_header(rate, channels, 1, 16, data.len() as u32);
        w.extend_from_slice(&data);
        w
    }

    /// `v / 32768` for comparisons with decoded s16.
    pub(crate) fn f(v: &[i16]) -> Vec<f32> {
        v.iter().map(|&s| f32::from(s) / 32_768.0).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::testing::*;
    use super::*;
    use tokio::io::AsyncWriteExt;

    fn req(base: &str) -> OpenAiSpeechRequest {
        OpenAiSpeechRequest { base_url: base.into(), input: "Hello there.".into(), ..Default::default() }
    }

    #[test]
    fn request_defaults_and_wire_names_are_frozen() {
        let r: OpenAiSpeechRequest =
            serde_json::from_str(r#"{"baseUrl":"http://127.0.0.1:8880/v1","input":"hi","voice":"af_bella","responseFormat":"wav"}"#).unwrap();
        assert_eq!(r.response_format, SpeechFormat::Wav);
        assert_eq!(r.speed, 1.0);
        assert!(r.stream);
        assert_eq!(r.pcm_sample_rate, 24_000);
        let v = serde_json::to_value(OpenAiSpeechRequest::default()).unwrap();
        for key in ["baseUrl", "apiKey", "model", "input", "voice", "responseFormat", "speed", "stream", "pcmSampleRate", "extra", "timeoutMs"] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
    }

    #[test]
    fn openai_speech_body_sends_only_what_is_set() {
        let body = request_body(&req("x"), true).unwrap();
        assert_eq!(body, serde_json::json!({"input": "Hello there.", "response_format": "pcm", "stream": true}));

        let r = OpenAiSpeechRequest {
            model: " kokoro ".into(),
            voice: " af_heart ".into(),
            speed: 1.25,
            stream: false,
            response_format: SpeechFormat::Wav,
            ..req("x")
        };
        assert_eq!(
            request_body(&r, true).unwrap(),
            serde_json::json!({"model": "kokoro", "input": "Hello there.", "voice": "af_heart", "response_format": "wav", "speed": 1.25})
        );
    }

    #[test]
    fn openai_speech_speed_is_sanitized() {
        assert_eq!(speed_member(1.0), None);
        assert_eq!(speed_member(f32::NAN), None);
        assert_eq!(speed_member(0.0), None);
        assert_eq!(speed_member(-2.0), None);
        assert_eq!(speed_member(10.0), Some(4.0));
        assert_eq!(speed_member(0.1), Some(0.25));
        assert_eq!(speed_member(0.8), Some(0.8));
    }

    #[test]
    fn openai_speech_extra_merges_but_never_replaces_the_text() {
        let r = OpenAiSpeechRequest {
            model: "kokoro".into(),
            extra: serde_json::json!({"lang_code": "j", "model": "other", "input": "injected", "exaggeration": 0.7}),
            ..req("x")
        };
        let body = request_body(&r, true).unwrap();
        assert_eq!(body["lang_code"], "j");
        assert_eq!(body["model"], "other", "extra may override the standard members");
        assert_eq!(body["input"], "Hello there.");
        assert_eq!(body["exaggeration"], 0.7);

        let bad = OpenAiSpeechRequest { extra: serde_json::json!([1, 2]), ..req("x") };
        assert!(request_body(&bad, true).unwrap_err().contains("JSON object"));
        let null = OpenAiSpeechRequest { extra: Value::Null, ..req("x") };
        assert!(request_body(&null, true).is_ok());
    }

    #[test]
    fn openai_speech_base_gets_v1_only_without_a_path() {
        assert_eq!(speech_base("http://127.0.0.1:8880").unwrap(), "http://127.0.0.1:8880/v1");
        assert_eq!(speech_base("localhost:8000/").unwrap(), "http://localhost:8000/v1");
        assert_eq!(speech_base("http://127.0.0.1:8880/v1/").unwrap(), "http://127.0.0.1:8880/v1");
        assert_eq!(speech_base("https://proxy.example/tts/v1").unwrap(), "https://proxy.example/tts/v1");
        assert!(speech_base("").is_err());
        assert!(speech_base("ftp://host").is_err());
        let spec = openai_spec(&req("http://127.0.0.1:5555")).unwrap();
        assert_eq!(spec.url().as_str(), "http://127.0.0.1:5555/v1/audio/speech");
    }

    #[test]
    fn openai_speech_spec_validates_rate_and_key() {
        assert_eq!(openai_spec(&OpenAiSpeechRequest { pcm_sample_rate: 0, ..req("x") }).unwrap().pcm_rate, 24_000);
        assert_eq!(openai_spec(&OpenAiSpeechRequest { pcm_sample_rate: 48_000, ..req("x") }).unwrap().pcm_rate, 48_000);
        assert!(openai_spec(&OpenAiSpeechRequest { pcm_sample_rate: 1_000, ..req("x") }).is_err());
        let spec = openai_spec(&OpenAiSpeechRequest { api_key: " sk-1 ".into(), ..req("x") }).unwrap();
        let bearer = spec.bearer.as_ref().unwrap();
        assert_eq!(bearer.to_str().unwrap(), "Bearer sk-1");
        assert!(bearer.is_sensitive(), "keys never show up in debug output");
        assert!(openai_spec(&OpenAiSpeechRequest { api_key: "  ".into(), ..req("x") }).unwrap().bearer.is_none());
        assert!(openai_spec(&OpenAiSpeechRequest { api_key: "a\nb".into(), ..req("x") }).is_err());
    }

    #[test]
    fn openai_speech_timing_scales_for_local_servers() {
        let local = Url::parse("http://127.0.0.1:8880/v1/audio/speech").unwrap();
        let cloud = Url::parse("https://api.openai.com/v1/audio/speech").unwrap();
        let t = Timing::for_request(&local, "hello", None);
        assert!(t.local && t.long());
        assert_eq!(t.answer, local_tts_timeout("hello"));
        assert_eq!(t.stall, LOCAL_TTS_TIMEOUT);
        let t = Timing::for_request(&cloud, "hello", None);
        assert!(!t.long());
        assert_eq!(t.answer, DEFAULT_ANSWER);
        assert_eq!(t.stall, TTS_STALL_TIMEOUT);
        let t = Timing::for_request(&cloud, "hello", Some(Duration::from_millis(10)));
        assert_eq!((t.answer, t.stall), (MIN_TIMEOUT, MIN_TIMEOUT));
        let t = Timing::for_request(&cloud, "hello", Some(Duration::from_secs(3600)));
        assert_eq!(t.answer, LOCAL_TTS_TIMEOUT_MAX);
        assert_eq!(t.stall, MAX_STALL, "the gap between two pieces of audio stays below the sink's stall timeout");
        assert!(t.long(), "a long explicit timeout needs the long-read client");
    }

    /// Review finding: the sink fails a head sentence after its own bounds (11 min before the first
    /// audio, 120 s between pushes once it plays) and its cancel ends the request as "cancelled", so
    /// every budget of this client must end first, or a slow CPU server's sentence would be dropped.
    #[test]
    fn openai_speech_timing_fits_inside_the_sink_bounds() {
        let local = Url::parse("http://localhost:5555/v1/audio/speech").unwrap();
        let lan = Url::parse("http://192.168.1.20:8000/v1/audio/speech").unwrap();
        let cloud = Url::parse("https://api.openai.com/v1/audio/speech").unwrap();
        let long_text = "a".repeat(240);
        let huge_text = "a".repeat(100_000);
        let explicit = [None, Some(Duration::from_secs(1)), Some(Duration::from_secs(300)), Some(Duration::from_secs(3600))];
        for url in [&local, &lan, &cloud] {
            for text in ["hi", long_text.as_str(), huge_text.as_str()] {
                for t in explicit {
                    let timing = Timing::for_request(url, text, t);
                    assert!(timing.answer < SINK_FIRST_AUDIO_TIMEOUT, "{url} {t:?}: {timing:?}");
                    assert!(timing.stall < SINK_STALL_TIMEOUT, "{url} {t:?}: {timing:?}");
                }
            }
        }
        // The finding's case: a 240-character sentence on a CPU Chatterbox keeps its full 300 s for
        // the first audio (the whole clip), and so does an explicit 300 s timeout.
        assert_eq!(Timing::for_request(&local, &long_text, None).answer, Duration::from_secs(300));
        let t = Timing::for_request(&cloud, "hi", Some(Duration::from_secs(300)));
        assert_eq!((t.answer, t.stall), (Duration::from_secs(300), MAX_STALL));
    }

    #[test]
    fn openai_speech_never_sends_stream_to_openai_itself() {
        let body = |base: &str, extra: Value| {
            openai_spec(&OpenAiSpeechRequest { base_url: base.into(), stream: true, extra, ..req("") })
                .unwrap()
                .body()
                .clone()
        };
        for base in ["https://api.openai.com/v1", "https://API.OpenAI.com", "https://my-res.openai.azure.com/openai/deployments/tts"] {
            assert!(body(base, Value::Null).get("stream").is_none(), "{base}");
        }
        for base in ["http://127.0.0.1:8880/v1", "https://proxy.example/v1", "https://openai.com.example/v1"] {
            assert_eq!(body(base, Value::Null)["stream"], true, "{base}");
        }
        // An explicit extra member is the user's choice.
        assert_eq!(body("https://api.openai.com/v1", serde_json::json!({"stream": true}))["stream"], true);
        assert!(omits_stream_member(&Url::parse("https://api.openai.com./v1").unwrap()));
    }

    #[test]
    fn openai_speech_content_types() {
        assert_eq!(ContentKind::parse(None), ContentKind::Unknown);
        assert_eq!(ContentKind::parse(Some("application/octet-stream")), ContentKind::Unknown);
        assert_eq!(ContentKind::parse(Some("application/json; charset=utf-8")), ContentKind::Textual);
        assert_eq!(ContentKind::parse(Some("text/html")), ContentKind::Textual);
        assert_eq!(ContentKind::parse(Some("audio/x-wav")), ContentKind::Wav);
        assert_eq!(ContentKind::parse(Some("audio/pcm")), ContentKind::Pcm(None));
        assert_eq!(ContentKind::parse(Some("audio/L16; rate=16000; channels=1")), ContentKind::Pcm(Some(16_000)));
        assert_eq!(ContentKind::parse(Some("audio/pcm;rate=5")), ContentKind::Pcm(None));
        assert_eq!(ContentKind::parse(Some("audio/mpeg")), ContentKind::Compressed);
    }

    // -----------------------------------------------------------------------------------------
    // Decoder
    // -----------------------------------------------------------------------------------------

    fn decode_all(dec: &mut AudioDecoder, parts: &[&[u8]]) -> Vec<Block> {
        let mut out = Vec::new();
        for p in parts {
            out.extend(dec.feed(p).unwrap());
        }
        out
    }

    fn samples(blocks: &[Block]) -> Vec<f32> {
        blocks.iter().flat_map(|b| b.samples.clone()).collect()
    }

    fn pcm_decoder() -> AudioDecoder {
        AudioDecoder::new(SpeechFormat::Pcm, 24_000, ContentKind::Unknown, MAX_AUDIO_BYTES)
    }

    fn wav_decoder() -> AudioDecoder {
        AudioDecoder::new(SpeechFormat::Wav, 24_000, ContentKind::Unknown, MAX_AUDIO_BYTES)
    }

    #[test]
    fn openai_speech_decodes_pcm_split_inside_samples() {
        let values: Vec<i16> = (0..400).map(|i| (i * 37 % 2000) as i16 - 1000).collect();
        let bytes = pcm16(&values);
        // Odd-sized chunks split samples between chunks.
        let parts: Vec<&[u8]> = bytes.chunks(13).collect();
        let mut dec = pcm_decoder();
        let mut blocks = decode_all(&mut dec, &parts);
        let (tail, compressed) = dec.finish().unwrap();
        blocks.extend(tail);
        assert!(compressed.is_none());
        assert_eq!(samples(&blocks), f(&values));
        assert!(blocks.iter().all(|b| b.rate == 24_000 && b.channels == 1));
    }

    #[test]
    fn openai_speech_decodes_a_streamed_wav_byte_by_byte() {
        let values: Vec<i16> = (0..300).map(|i| (i * 91 % 4000) as i16 - 2000).collect();
        for placeholder in [0u32, u32::MAX] {
            let mut body = wav_header(22_050, 2, 1, 16, placeholder);
            body.extend_from_slice(&pcm16(&values));
            let parts: Vec<&[u8]> = body.chunks(1).collect();
            let mut dec = pcm_decoder(); // a WAV answer to a pcm request is still decoded as WAV
            let mut blocks = decode_all(&mut dec, &parts);
            blocks.extend(dec.finish().unwrap().0);
            assert_eq!(samples(&blocks), f(&values), "placeholder {placeholder:#x}");
            assert!(blocks.iter().all(|b| b.rate == 22_050 && b.channels == 2));
        }
    }

    #[test]
    fn openai_speech_wav_drops_a_partial_last_frame() {
        let mut body = wav_header(16_000, 1, 1, 16, 0);
        body.extend_from_slice(&pcm16(&[10, 20, 30]));
        body.push(0x7f); // half of a sample
        let mut dec = wav_decoder();
        let mut blocks = decode_all(&mut dec, &[&body]);
        blocks.extend(dec.finish().unwrap().0);
        assert_eq!(samples(&blocks), f(&[10, 20, 30]));
    }

    #[test]
    fn openai_speech_wav_skips_chunks_and_ignores_trailing_metadata() {
        // LIST (odd size, padded) before data; a sized data chunk; LIST after it.
        let data = pcm16(&[100, 200, 300, 400]);
        let mut body = wav_header(44_100, 1, 1, 16, data.len() as u32);
        let data_pos = body.len() - 8;
        let mut list = b"LIST".to_vec();
        list.extend_from_slice(&3u32.to_le_bytes());
        list.extend_from_slice(b"abc\0");
        body.splice(data_pos..data_pos, list);
        body.extend_from_slice(&data);
        body.extend_from_slice(b"LIST\x04\x00\x00\x00info");
        let mut dec = wav_decoder();
        let mut blocks = decode_all(&mut dec, &[&body[..20], &body[20..]]);
        blocks.extend(dec.finish().unwrap().0);
        assert_eq!(samples(&blocks), f(&[100, 200, 300, 400]));
    }

    #[test]
    fn openai_speech_decodes_back_to_back_wavs() {
        // Servers that send one complete WAV per chunk (odd data size → pad byte).
        let a = wav16(&[1, 2, 3], 24_000, 1);
        let mut b_body = wav_header(48_000, 1, 1, 8, 3);
        b_body.extend_from_slice(&[128, 255, 0, 0]); // three u8 samples + pad
        let c = wav16(&[7, 8], 24_000, 1);
        let body = [a, b_body, c].concat();
        let mut dec = wav_decoder();
        let mut blocks = decode_all(&mut dec, &body.chunks(5).collect::<Vec<_>>());
        blocks.extend(dec.finish().unwrap().0);
        let rates: Vec<u32> = blocks.iter().map(|b| b.rate).collect();
        assert!(rates.contains(&48_000));
        let all = samples(&blocks);
        assert_eq!(all.len(), 3 + 3 + 2);
        assert_eq!(&all[..3], f(&[1, 2, 3]).as_slice());
        assert_eq!(&all[3..6], &[0.0, 127.0 / 128.0, -1.0]);
        assert_eq!(&all[6..], f(&[7, 8]).as_slice());
    }

    #[test]
    fn openai_speech_decodes_every_wav_sample_format() {
        // 24-bit
        let mut w = wav_header(24_000, 1, 1, 24, 6);
        w.extend_from_slice(&[0x00, 0x00, 0x40, 0x00, 0x00, 0xC0]);
        let mut dec = wav_decoder();
        assert_eq!(samples(&decode_all(&mut dec, &[&w])), [0.5, -0.5]);
        // 32-bit int
        let mut w = wav_header(24_000, 1, 1, 32, 4);
        w.extend_from_slice(&(i32::MIN / 2).to_le_bytes());
        let mut dec = wav_decoder();
        assert_eq!(samples(&decode_all(&mut dec, &[&w])), [-0.5]);
        // 32-bit float, non-finite → 0
        let mut w = wav_header(24_000, 1, 3, 32, 8);
        w.extend_from_slice(&0.25f32.to_le_bytes());
        w.extend_from_slice(&f32::NAN.to_le_bytes());
        let mut dec = wav_decoder();
        assert_eq!(samples(&decode_all(&mut dec, &[&w])), [0.25, 0.0]);
        // 64-bit float
        let mut w = wav_header(24_000, 1, 3, 64, 8);
        w.extend_from_slice(&(-0.75f64).to_le_bytes());
        let mut dec = wav_decoder();
        assert_eq!(samples(&decode_all(&mut dec, &[&w])), [-0.75]);
        // WAVE_FORMAT_EXTENSIBLE with a float sub-format
        let mut w = b"RIFF\0\0\0\0WAVEfmt ".to_vec();
        w.extend_from_slice(&40u32.to_le_bytes());
        w.extend_from_slice(&0xFFFEu16.to_le_bytes());
        w.extend_from_slice(&1u16.to_le_bytes());
        w.extend_from_slice(&24_000u32.to_le_bytes());
        w.extend_from_slice(&96_000u32.to_le_bytes());
        w.extend_from_slice(&4u16.to_le_bytes());
        w.extend_from_slice(&32u16.to_le_bytes());
        w.extend_from_slice(&22u16.to_le_bytes());
        w.extend_from_slice(&32u16.to_le_bytes());
        w.extend_from_slice(&4u32.to_le_bytes());
        w.extend_from_slice(&3u16.to_le_bytes());
        w.extend_from_slice(&[0u8; 14]);
        w.extend_from_slice(b"data");
        w.extend_from_slice(&u32::MAX.to_le_bytes());
        w.extend_from_slice(&0.5f32.to_le_bytes());
        let mut dec = wav_decoder();
        assert_eq!(samples(&decode_all(&mut dec, &[&w])), [0.5]);
        // Unsupported: A-law
        let w = wav_header(8_000, 1, 6, 8, 0);
        assert!(wav_decoder().feed(&w).unwrap_err().contains("not supported"));
    }

    #[test]
    fn openai_speech_decoder_rejects_what_is_not_audio() {
        let err = pcm_decoder().feed(br#"{"detail":"Voice 'zz' not found"}"#).unwrap_err();
        assert!(err.contains("text response") && err.contains("Voice 'zz' not found"), "{err}");
        // Raw audio that happens to start with a brace is not text.
        let mut raw = vec![b'{', 0x00, 0x01, 0xFF, 0x80, 0x02];
        raw.extend_from_slice(&[0u8; 20]);
        assert!(!pcm_decoder().feed(&raw).unwrap().is_empty());
        // WAV asked, no header.
        let err = wav_decoder().feed(&pcm16(&[1, 2, 3, 4, 5, 6, 7, 8])).unwrap_err();
        assert!(err.contains("no WAV header"), "{err}");
        // A RIFF that is not WAVE.
        assert!(wav_decoder().feed(b"RIFF\x04\0\0\0AVI LIST").unwrap_err().contains("not WAV"));
        // Size cap.
        let mut small = AudioDecoder::new(SpeechFormat::Pcm, 24_000, ContentKind::Unknown, 16);
        assert!(small.feed(&[0u8; 17]).unwrap_err().contains("larger than"));
        // A header cut short.
        let mut dec = wav_decoder();
        dec.feed(&wav_header(24_000, 1, 1, 16, 0)[..20]).unwrap();
        assert!(dec.finish().unwrap_err().contains("inside its header"));
    }

    #[test]
    fn openai_speech_compressed_answers_are_buffered_then_decoded() {
        for (kind, head) in [
            (ContentKind::Unknown, b"ID3\x04\0\0\0\0\0\0\0\0\0".to_vec()),
            (ContentKind::Unknown, b"OggS\0\x02\0\0\0\0\0\0\0\0".to_vec()),
            (ContentKind::Compressed, vec![0u8; 16]),
            (ContentKind::Wav, vec![0xFF, 0xFB, 0x90, 0x64, 0, 0, 0, 0, 0, 0, 0, 0]),
        ] {
            let mut dec = AudioDecoder::new(SpeechFormat::Wav, 24_000, kind, MAX_AUDIO_BYTES);
            assert!(dec.feed(&head).unwrap().is_empty());
            let (blocks, bytes) = dec.finish().unwrap();
            assert!(blocks.is_empty());
            assert_eq!(bytes.unwrap(), head);
        }
        // The file decoder (rodio) handles whole clips; a WAV stands in for MP3 here.
        let block = decode_compressed(wav16(&[1000, -1000, 2000, -2000], 16_000, 2)).unwrap();
        assert_eq!((block.rate, block.channels), (16_000, 2));
        assert_eq!(block.samples, f(&[1000, -1000, 2000, -2000]));
        assert!(decode_compressed(b"ID3garbage".to_vec()).unwrap_err().contains("cannot be decoded"));
    }

    /// `frames` MPEG frames (header `h` + zero payload, each `len` bytes long).
    fn mpeg_frames(h: &[u8], len: usize, frames: usize) -> Vec<u8> {
        let mut frame = h.to_vec();
        frame.resize(len, 0);
        frame.repeat(frames)
    }

    /// An ADTS header (AAC LC, 44.1 kHz, stereo) of a `len`-byte frame.
    fn adts_header(len: usize) -> [u8; 7] {
        [0xFF, 0xF1, 0x50, 0x80 | ((len >> 11) & 3) as u8, ((len >> 3) & 0xFF) as u8, (((len & 7) << 5) | 0x1F) as u8, 0xFC]
    }

    #[test]
    fn openai_speech_mpeg_frame_lengths() {
        // MPEG 1 Layer III, 128 kbps, 44.1 kHz, without / with padding.
        assert_eq!(mpeg_frame_len(&[0xFF, 0xFB, 0x90, 0x64]), Some(417));
        assert_eq!(mpeg_frame_len(&[0xFF, 0xFB, 0x92, 0x64]), Some(418));
        // MPEG 2 Layer III, 64 kbps, 24 kHz (OpenAI-style mono MP3).
        assert_eq!(mpeg_frame_len(&[0xFF, 0xF3, 0x84, 0xC4]), Some(192));
        // MPEG 1 Layer I, 32 kbps, 48 kHz.
        assert_eq!(mpeg_frame_len(&[0xFF, 0xFF, 0x14, 0x00]), Some(32));
        assert_eq!(mpeg_frame_len(&adts_header(371)), Some(371));
        // Reserved / free-format values and non-sync bytes.
        assert_eq!(mpeg_frame_len(&[0xFF, 0xFB, 0x00, 0x64]), None, "free format");
        assert_eq!(mpeg_frame_len(&[0xFF, 0xFB, 0xF0, 0x64]), None, "bad bitrate");
        assert_eq!(mpeg_frame_len(&[0xFF, 0xFB, 0x9C, 0x64]), None, "reserved sample rate");
        assert_eq!(mpeg_frame_len(&[0xFF, 0xEB, 0x90, 0x64]), None, "reserved version");
        assert_eq!(mpeg_frame_len(&[0xFF, 0xFB, 0x90, 0x66]), None, "reserved emphasis");
        assert_eq!(mpeg_frame_len(&[0xFE, 0xFB, 0x90, 0x64]), None);
        assert_eq!(mpeg_frame_len(&[0xFF, 0xFB]), None);
    }

    /// Review finding: an untagged MP3 / ADTS answer to a `pcm` request with no or an
    /// `application/octet-stream` Content-Type was played as s16 samples (loud noise).
    #[test]
    fn openai_speech_untagged_mp3_to_a_pcm_request_is_decoded_as_a_file() {
        let mp3 = mpeg_frames(&[0xFF, 0xFB, 0x90, 0x64], 417, 3);
        let adts = [mpeg_frames(&adts_header(371), 371, 2), mpeg_frames(&adts_header(300), 300, 1)].concat();
        for (name, body) in [("mp3", &mp3), ("adts", &adts)] {
            for piece in [body.len(), 100, 7] {
                // `ContentKind::Unknown` = no Content-Type or application/octet-stream.
                let mut dec = pcm_decoder();
                let parts: Vec<&[u8]> = body.chunks(piece).collect();
                assert!(decode_all(&mut dec, &parts).is_empty(), "{name}/{piece}: nothing is played as PCM");
                let (blocks, bytes) = dec.finish().unwrap();
                assert!(blocks.is_empty(), "{name}/{piece}");
                assert_eq!(bytes.as_deref(), Some(body.as_slice()), "{name}/{piece}: the whole body goes to the file decoder");
            }
        }
        // An explicit PCM Content-Type is trusted.
        let mut dec = AudioDecoder::new(SpeechFormat::Pcm, 24_000, ContentKind::Pcm(None), MAX_AUDIO_BYTES);
        assert_eq!(samples(&dec.feed(&mp3).unwrap()).len(), mp3.len() / 2);
    }

    /// Raw PCM that merely starts like an MPEG header (a first sample of -1 followed by a small
    /// positive one) is still PCM: the next frame header is missing.
    #[test]
    fn openai_speech_pcm_that_looks_like_an_mpeg_header_stays_pcm() {
        let mut values: Vec<i16> = vec![-1, 0x0010];
        values.extend((0..600).map(|i| ((i * 13) % 200) as i16 - 100));
        assert!(mpeg_frame_len(&pcm16(&values)).is_some(), "the test body starts with a valid header");
        let bytes = pcm16(&values);
        for piece in [bytes.len(), 16, 5] {
            let mut dec = pcm_decoder();
            let parts: Vec<&[u8]> = bytes.chunks(piece).collect();
            let mut blocks = decode_all(&mut dec, &parts);
            let (tail, compressed) = dec.finish().unwrap();
            blocks.extend(tail);
            assert!(compressed.is_none(), "{piece}");
            assert_eq!(samples(&blocks), f(&values), "{piece}");
        }
        // A body shorter than one frame + the next header, at its end: PCM too.
        let short = pcm16(&[-1, 0x0010, 5, 6, 7, 8]);
        let mut dec = pcm_decoder();
        assert!(dec.feed(&short).unwrap().is_empty(), "waits for the next header");
        let (blocks, compressed) = dec.finish().unwrap();
        assert!(compressed.is_none());
        assert_eq!(samples(&blocks), f(&[-1, 0x0010, 5, 6, 7, 8]));
    }

    #[test]
    fn openai_speech_empty_body_decodes_to_nothing() {
        let (blocks, bytes) = pcm_decoder().finish().unwrap();
        assert!(blocks.is_empty() && bytes.is_none());
        // A tiny pcm body still decodes.
        let mut dec = pcm_decoder();
        assert!(dec.feed(&[1, 0, 2]).unwrap().is_empty());
        let (blocks, _) = dec.finish().unwrap();
        assert_eq!(samples(&blocks), f(&[1]));
    }

    // -----------------------------------------------------------------------------------------
    // Streaming against loopback servers
    // -----------------------------------------------------------------------------------------

    fn spec_for(base: &str, r: OpenAiSpeechRequest) -> HttpSpeech {
        openai_spec(&OpenAiSpeechRequest { base_url: base.into(), ..r }).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_first_audio_arrives_before_the_full_response() {
        let first: Vec<i16> = (0..2400).map(|i| (i % 100) as i16).collect();
        let second: Vec<i16> = (0..4800).map(|i| -((i % 50) as i16)).collect();
        let (first_tx, first_rx) = tokio::sync::oneshot::channel::<()>();
        let (l, base) = listener().await;
        let (p1, p2) = (pcm16(&first), pcm16(&second));
        let server = tokio::spawn(async move {
            let (mut sock, _) = l.accept().await.unwrap();
            let seen = read_request(&mut sock).await;
            sock.write_all(&chunked_head("audio/pcm")).await.unwrap();
            sock.write_all(&chunk(&p1)).await.unwrap();
            sock.flush().await.unwrap();
            // The rest of the answer is sent only once the client played the first part: a client
            // that waits for the whole response never gets it.
            let played_first = tokio::time::timeout(Duration::from_secs(10), first_rx).await.is_ok();
            sock.write_all(&chunk(&p2)).await.unwrap();
            sock.write_all(LAST_CHUNK).await.unwrap();
            let _ = sock.shutdown().await;
            (seen, played_first)
        });
        let (mut sink, pushes) = Recorder::new();
        sink.on_first = Some(first_tx);
        let report = speak(spec_for(&base, req("")), "t-first-audio", Box::new(sink)).await.unwrap();
        let (seen, played_first) = server.await.unwrap();
        assert!(played_first, "audio reached the sink before the server finished the response");
        assert_eq!(seen.path(), "/v1/audio/speech");
        assert_eq!(report.frames, (first.len() + second.len()) as u64);
        assert_eq!(report.sample_rate, 24_000);
        assert!(report.first_audio_ms <= report.total_ms);
        assert_eq!(pushes.samples(), [f(&first), f(&second)].concat());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_sends_the_request_on_the_wire() {
        let (base, server) = serve_once(full_response("200 OK", "audio/pcm", &pcm16(&[1, 2, 3, 4]))).await;
        let r = OpenAiSpeechRequest {
            api_key: "sk-test".into(),
            model: "kokoro".into(),
            voice: "af_heart".into(),
            speed: 1.5,
            extra: serde_json::json!({"lang_code": "a"}),
            ..req("")
        };
        let (sink, pushes) = Recorder::new();
        speak(spec_for(&format!("{base}/v1"), r), "t-wire", Box::new(sink)).await.unwrap();
        let seen = server.await.unwrap();
        assert!(seen.head.starts_with("POST /v1/audio/speech "), "{}", seen.head);
        assert_eq!(seen.header("authorization").as_deref(), Some("Bearer sk-test"));
        assert_eq!(seen.header("content-type").as_deref(), Some("application/json"));
        assert_eq!(
            seen.json(),
            serde_json::json!({"model": "kokoro", "input": "Hello there.", "voice": "af_heart", "response_format": "pcm", "speed": 1.5, "stream": true, "lang_code": "a"})
        );
        assert_eq!(pushes.samples(), f(&[1, 2, 3, 4]));

        // No key, no header.
        let (base, server) = serve_once(full_response("200 OK", "audio/pcm", &pcm16(&[1, 2]))).await;
        let (sink, _) = Recorder::new();
        speak(spec_for(&base, req("")), "t-wire-2", Box::new(sink)).await.unwrap();
        assert!(server.await.unwrap().header("authorization").is_none());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_streams_a_wav_with_placeholder_sizes() {
        let values: Vec<i16> = (0..9000).map(|i| ((i * 7) % 3000) as i16 - 1500).collect();
        let mut body = chunked_head("audio/wav");
        let mut wav = wav_header(44_100, 1, 1, 16, u32::MAX);
        wav.extend_from_slice(&pcm16(&values));
        for part in wav.chunks(1001) {
            body.extend_from_slice(&chunk(part));
        }
        body.extend_from_slice(LAST_CHUNK);
        let (base, server) = serve_once(body).await;
        let (sink, pushes) = Recorder::new();
        let r = OpenAiSpeechRequest { response_format: SpeechFormat::Wav, ..req("") };
        let report = speak(spec_for(&base, r), "t-wav", Box::new(sink)).await.unwrap();
        assert_eq!(server.await.unwrap().json()["response_format"], "wav");
        assert_eq!(report.sample_rate, 44_100);
        assert_eq!(report.frames, values.len() as u64);
        assert_eq!(pushes.samples(), f(&values));
        assert!(pushes.formats().iter().all(|&(r, c)| r == 44_100 && c == 1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_pcm_rate_comes_from_the_content_type() {
        let (base, _server) = serve_once(full_response("200 OK", "audio/L16; rate=16000", &pcm16(&[5; 32]))).await;
        let (sink, pushes) = Recorder::new();
        let report = speak(spec_for(&base, req("")), "t-rate", Box::new(sink)).await.unwrap();
        assert_eq!(report.sample_rate, 16_000);
        assert!(pushes.formats().iter().all(|&(r, _)| r == 16_000));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_long_pushes_are_split() {
        // 2 s of audio in one body: pushed in ≤ 0.5 s pieces so a cancel stops it early.
        let (base, _server) = serve_once(full_response("200 OK", "audio/pcm", &pcm16(&vec![1; 48_000]))).await;
        let (sink, pushes) = Recorder::new();
        let report = speak(spec_for(&base, req("")), "t-split", Box::new(sink)).await.unwrap();
        assert_eq!(report.frames, 48_000);
        assert!(pushes.count() >= 4);
        assert!(pushes.0.lock().unwrap().iter().all(|(s, _, _)| s.len() <= 12_000));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_reports_server_errors() {
        let (base, _s) = serve_once(full_response(
            "422 Unprocessable Entity",
            "application/json",
            br#"{"detail":[{"loc":["body","voice"],"msg":"Voice 'nope' not found","type":"value_error"}]}"#,
        ))
        .await;
        let (sink, _) = Recorder::new();
        let err = speak(spec_for(&base, req("")), "t-422", Box::new(sink)).await.unwrap_err();
        assert!(err.contains("422") && err.contains("Voice 'nope' not found"), "{err}");

        // 200 with a JSON error body.
        let (base, _s) = serve_once(full_response("200 OK", "application/json", br#"{"error":{"message":"model not loaded"}}"#)).await;
        let (sink, _) = Recorder::new();
        let err = speak(spec_for(&base, req("")), "t-200-json", Box::new(sink)).await.unwrap_err();
        assert!(err.contains("expected audio") && err.contains("model not loaded"), "{err}");

        // An empty answer.
        let (base, _s) = serve_once(full_response("200 OK", "audio/pcm", b"")).await;
        let (sink, _) = Recorder::new();
        let err = speak(spec_for(&base, req("")), "t-empty", Box::new(sink)).await.unwrap_err();
        assert!(err.contains("returned no audio"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_reports_a_server_that_is_not_running() {
        let (l, base) = listener().await;
        drop(l);
        let (sink, _) = Recorder::new();
        let err = speak(spec_for(&base, req("")), "t-refused", Box::new(sink)).await.unwrap_err();
        assert!(err.contains("cannot connect"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_sink_errors_end_the_request() {
        let (base, _s) = serve_once(full_response("200 OK", "audio/pcm", &pcm16(&[1; 100]))).await;
        let (mut sink, _) = Recorder::new();
        sink.fail_with = Some(SinkError::Device("the device was unplugged".into()));
        let err = speak(spec_for(&base, req("")), "t-device", Box::new(sink)).await.unwrap_err();
        assert!(err.contains("the device was unplugged"), "{err}");
    }

    /// A server that sends one chunk of audio and then keeps the response open.
    async fn endless_server() -> (String, tokio::task::JoinHandle<()>) {
        let (l, base) = listener().await;
        let task = tokio::spawn(async move {
            let (mut sock, _) = l.accept().await.unwrap();
            let _ = read_request(&mut sock).await;
            let _ = sock.write_all(&chunked_head("audio/pcm")).await;
            let _ = sock.write_all(&chunk(&pcm16(&[3; 2400]))).await;
            let _ = sock.flush().await;
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        (base, task)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_cancel_command_stops_a_running_stream() {
        let (base, server) = endless_server().await;
        let (first_tx, first_rx) = tokio::sync::oneshot::channel();
        let (mut sink, _) = Recorder::new();
        sink.on_first = Some(first_tx);
        let run = tokio::spawn(speak(spec_for(&base, req("")), "t-cancel-cmd", Box::new(sink)));
        first_rx.await.unwrap();
        let at = Instant::now();
        cancel(Some("t-cancel-cmd"));
        let err = tokio::time::timeout(Duration::from_secs(5), run).await.unwrap().unwrap().unwrap_err();
        assert_eq!(err, CANCELLED);
        assert!(at.elapsed() < Duration::from_secs(2));
        assert!(!RUNNING.lock().unwrap().contains_key("t-cancel-cmd"), "unregistered after the end");
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_cancelled_sink_utterance_stops_the_stream() {
        let (base, server) = endless_server().await;
        let (mut sink, _) = Recorder::new();
        sink.cancel_after_first = true;
        let err = tokio::time::timeout(Duration::from_secs(5), speak(spec_for(&base, req("")), "t-cancel-sink", Box::new(sink)))
            .await
            .unwrap()
            .unwrap_err();
        assert_eq!(err, CANCELLED);
        server.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_times_out_a_stalled_or_silent_server() {
        // Stalls after the first audio.
        let (base, server) = endless_server().await;
        let (sink, _) = Recorder::new();
        let r = OpenAiSpeechRequest { timeout_ms: 1_000, ..req("") };
        let err = tokio::time::timeout(Duration::from_secs(10), speak(spec_for(&base, r.clone()), "t-stall", Box::new(sink)))
            .await
            .unwrap()
            .unwrap_err();
        assert!(err.contains("stopped sending audio"), "{err}");
        server.abort();

        // Never answers.
        let (l, base) = listener().await;
        let silent = tokio::spawn(async move {
            let (mut sock, _) = l.accept().await.unwrap();
            let _ = read_request(&mut sock).await;
            tokio::time::sleep(Duration::from_secs(30)).await;
        });
        let (sink, _) = Recorder::new();
        let err = tokio::time::timeout(Duration::from_secs(10), speak(spec_for(&base, r), "t-silent", Box::new(sink)))
            .await
            .unwrap()
            .unwrap_err();
        assert!(err.contains("no response within 1 s"), "{err}");
        silent.abort();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn openai_speech_rejects_a_second_stream_for_the_same_utterance() {
        let (base, server) = endless_server().await;
        let (first_tx, first_rx) = tokio::sync::oneshot::channel();
        let (mut sink, _) = Recorder::new();
        sink.on_first = Some(first_tx);
        let run = tokio::spawn(speak(spec_for(&base, req("")), "t-dup", Box::new(sink)));
        first_rx.await.unwrap();
        let (sink2, _) = Recorder::new();
        let err = speak(spec_for(&base, req("")), "t-dup", Box::new(sink2)).await.unwrap_err();
        assert!(err.contains("already being spoken"), "{err}");
        cancel(Some("t-dup"));
        assert_eq!(run.await.unwrap().unwrap_err(), CANCELLED);
        server.abort();
    }

    /// Manual check against a real server, e.g. the reference sidecar:
    /// `python scripts/neural-sidecar/tts_sidecar.py --chunk-delay 0.2`, then
    /// `SIGIL_TEST_OPENAI_SPEECH_URL=http://127.0.0.1:8766/v1 cargo test openai_speech_live -- --ignored --nocapture`
    /// (optional `SIGIL_TEST_OPENAI_SPEECH_MODEL` / `_VOICE`, e.g. `kokoro` / `af_heart` for Kokoro-FastAPI).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "needs a running OpenAI-compatible speech server in SIGIL_TEST_OPENAI_SPEECH_URL"]
    async fn openai_speech_live_server_streams() {
        let Ok(base) = std::env::var("SIGIL_TEST_OPENAI_SPEECH_URL") else {
            eprintln!("SIGIL_TEST_OPENAI_SPEECH_URL is not set; skipped");
            return;
        };
        let model = std::env::var("SIGIL_TEST_OPENAI_SPEECH_MODEL").unwrap_or_default();
        let voice = std::env::var("SIGIL_TEST_OPENAI_SPEECH_VOICE").unwrap_or_default();
        assert!(check_with(&client(), &base, None).await.unwrap(), "no server at {base}");
        println!("voices: {:?}", list_voices_with(&client(), &base, None).await.unwrap());
        for format in [SpeechFormat::Pcm, SpeechFormat::Wav] {
            let r = OpenAiSpeechRequest {
                base_url: base.clone(),
                model: model.clone(),
                voice: voice.clone(),
                input: "This sentence is long enough to arrive in several pieces while it is generated.".into(),
                response_format: format,
                ..Default::default()
            };
            let (sink, pushes) = Recorder::new();
            let id = format!("live-{}", format.as_str());
            let report = speak(openai_spec(&r).unwrap(), &id, Box::new(sink)).await.unwrap();
            println!("{}: {report:?}, {} pushes", format.as_str(), pushes.count());
            assert!(report.frames > 0);
            assert!(report.first_audio_ms < report.total_ms, "the first audio came before the end of the response");
        }
    }

    #[tokio::test]
    async fn openai_speech_empty_text_makes_no_request() {
        let spec = HttpSpeech::new("x", Url::parse("http://invalid.invalid/v1/audio/speech").unwrap(), Value::Null, "  ", SpeechFormat::Pcm);
        let report = speak_http(spec, "t-empty-text").await.unwrap();
        assert_eq!(report.frames, 0);
        let spec = HttpSpeech::new("x", Url::parse("http://invalid.invalid/").unwrap(), Value::Null, "hi", SpeechFormat::Pcm);
        assert!(speak_http(spec, " ").await.unwrap_err().contains("utteranceId"));
    }

    // -----------------------------------------------------------------------------------------
    // Check and voice list
    // -----------------------------------------------------------------------------------------

    #[tokio::test]
    async fn openai_speech_check_reports_reachability_and_rejected_keys() {
        let c = client();
        for (status, expect) in [("200 OK", Some(true)), ("404 Not Found", Some(true)), ("503 Service Unavailable", Some(false))] {
            let (base, server) = serve_once(full_response(status, "application/json", b"{}")).await;
            assert_eq!(check_with(&c, &base, Some("k")).await.ok(), expect, "{status}");
            let seen = server.await.unwrap();
            assert!(seen.head.starts_with("GET /v1/models "), "{}", seen.head);
            assert_eq!(seen.header("authorization").as_deref(), Some("Bearer k"));
        }
        let (base, _s) = serve_once(full_response("401 Unauthorized", "application/json", b"{}")).await;
        let err = check_with(&c, &base, Some("k")).await.unwrap_err();
        assert!(err.contains("rejected the API key") && err.contains("401"), "{err}");
        let (l, base) = listener().await;
        drop(l);
        assert!(!check_with(&c, &base, None).await.unwrap());
        assert!(check_with(&c, "", None).await.is_err());
    }

    #[tokio::test]
    async fn openai_speech_lists_voices_from_known_routes() {
        let c = client();
        let (base, server) = serve_once(full_response("200 OK", "application/json", br#"{"voices":["af_heart","jf_alpha"]}"#)).await;
        assert_eq!(list_voices_with(&c, &format!("{base}/v1"), None).await.unwrap(), ["af_heart", "jf_alpha"]);
        assert!(server.await.unwrap().head.starts_with("GET /v1/audio/voices "));

        // 404 on /audio/voices, then /voices answers.
        let (l, base) = listener().await;
        let server = tokio::spawn(async move {
            let mut paths = Vec::new();
            for answer in [
                full_response("404 Not Found", "application/json", br#"{"detail":"Not Found"}"#),
                full_response("200 OK", "application/json", br#"{"data":[{"id":"casual_male","name":"Casual"}]}"#),
            ] {
                let (mut sock, _) = l.accept().await.unwrap();
                paths.push(read_request(&mut sock).await.path());
                sock.write_all(&answer).await.unwrap();
                let _ = sock.shutdown().await;
            }
            paths
        });
        assert_eq!(list_voices_with(&c, &base, None).await.unwrap(), ["casual_male"]);
        assert_eq!(server.await.unwrap(), ["/v1/audio/voices", "/v1/voices"]);

        // A server without voice routes: an empty list, not an error.
        let (l, base) = listener().await;
        tokio::spawn(async move {
            for _ in 0..2 {
                let (mut sock, _) = l.accept().await.unwrap();
                let _ = read_request(&mut sock).await;
                let _ = sock.write_all(&full_response("404 Not Found", "text/plain", b"nope")).await;
                let _ = sock.shutdown().await;
            }
        });
        assert!(list_voices_with(&c, &base, None).await.unwrap().is_empty());

        // A rejected key is an error.
        let (base, _s) = serve_once(full_response("403 Forbidden", "application/json", br#"{"error":"bad key"}"#)).await;
        let err = list_voices_with(&c, &base, Some("k")).await.unwrap_err();
        assert!(err.contains("403") && err.contains("bad key"), "{err}");
    }
}
