//! HTTP utterance engine: OpenAI-compatible `/v1/audio/transcriptions` (a user-run server such as
//! speaches, LocalAI, faster-whisper-server or the whisper.cpp server; Groq) and Moonshine HTTP.
//!
//! Owned by P2-A. One request per closed utterance (or per merged batch when the pipeline fell
//! behind); silence never produces a request (the pipeline's Silero gate).
//!
//! `EngineSpec::options` (camelCase; may hold the key: never logged, emitted or echoed):
//! - `preset`: `"openai"` (default), `"groq"` or `"moonshine"`;
//! - `baseUrl`: server base (`http://localhost:8000/v1`); Groq always uses
//!   `https://api.groq.com/openai/v1`, Moonshine defaults to `http://localhost:8090`;
//! - `model`: required for `openai`; Groq defaults to `whisper-large-v3-turbo`;
//! - `apiKey`: sent only as `Authorization: Bearer …` (required for Groq);
//! - `language`, `prompt`: defaults under the per-utterance `DecodeOptions`.
//!
//! Requests: OpenAI-compatible servers get `multipart/form-data` (`file` = 16 kHz mono 16-bit WAV,
//! `model`, `response_format=json`, `temperature=0`, optional ISO-639-1 `language` and `prompt` with
//! the hotwords appended); Moonshine gets `POST {base}/transcribe` with JSON
//! `{ audio: <base64 WAV>, language }` (after a `GET {base}/health` at creation).
//!
//! Failures: 429 waits for `Retry-After` (else 5 s) and retries; timeouts, connection errors, 408,
//! 425 and 5xx are retried up to 4 attempts with 1-2-4 s backoff (`network` afterwards); a 4xx
//! whose message names the language is retried once without it; any other 4xx or an unreadable
//! response is `invalid_input` (a bad key, model or URL: retrying cannot help, the pipeline stops).
//! Waits and retries report `speech:status` through the engine context and are cancelled by the
//! pipeline's stop. A rate-limit wait (the Groq limiter or a 429) that would pass the utterance's
//! latency deadline (`worker::scope`) gives the utterance up instead (`rate_limited`), so late
//! captions never replace recent speech; a retry backoff past it ends the attempts.
//!
//! Groq (the preset, or any `*.groq.com` base URL) is paced by one process-wide [`GroqLimiter`]
//! so the free plan's limits are never hit (20 requests / min, 2,000 / day, 7,200 audio seconds /
//! hour, each request billed as at least 10 s): the same policy as the TS `GroqRateLimiter`
//! (P0-D). Its daily pace also asks the pipeline to hold short utterances until they add up to the
//! billing minimum ([`batch_policy`]).
//!
//! # Frozen signature (P2-S0)
//! `create(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError>` (called by
//! `engines::create`; blocking).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use base64::{engine::general_purpose::STANDARD, Engine as _};
use reqwest::header::{HeaderMap, AUTHORIZATION, CONTENT_TYPE, RETRY_AFTER};
use reqwest::{StatusCode, Url};
use serde::{Deserialize, Serialize};
use tokio::sync::Notify;

use super::super::engine_api::{
    AsrEngine, AsrError, AsrErrorKind, AsrResult, Canceller, DecodeOptions, EngineCaps, EngineContext, EngineInfo,
    EngineKind, EngineSpec, EngineStatus, StatusLevel, StatusSink,
};
use super::super::events::codes;
use super::super::worker::{scope, BatchPolicy};
use crate::services::http_client::{
    api_url, describe_api_error, describe_request_error, http_client, is_local_endpoint, normalize_base_url,
    probe_ok, redact_url, stt_timeout, HEALTH_TIMEOUT, STT_TIMEOUT_MAX,
};
use crate::services::speech_core::SAMPLE_RATE;

/// Log target.
const LOG: &str = "sigil::stt::http";
/// Groq's OpenAI-compatible API.
pub(crate) const GROQ_BASE_URL: &str = "https://api.groq.com/openai/v1";
/// Groq's default model.
pub(crate) const GROQ_DEFAULT_MODEL: &str = "whisper-large-v3-turbo";
/// Moonshine server default.
pub(crate) const MOONSHINE_DEFAULT_URL: &str = "http://localhost:8090";
/// Attempts per request for retryable failures.
const MAX_ATTEMPTS: u32 = 4;
const RETRY_BASE: Duration = Duration::from_secs(1);
const RETRY_MAX: Duration = Duration::from_secs(15);
/// Wait after a 429 without a readable `Retry-After`.
const RATE_LIMIT_FALLBACK: Duration = Duration::from_secs(5);
/// A limiter or server wait longer than this gives the utterance up (`rate_limited`).
const MAX_WAIT: Duration = Duration::from_secs(600);
/// Waits shorter than this are not reported.
const STATUS_DELAY: Duration = Duration::from_millis(1_500);
/// Longest prompt sent (bytes; OpenAI keeps the last 224 tokens).
const MAX_PROMPT_BYTES: usize = 1024;

// ---------------------------------------------------------------------------------------------
// Options
// ---------------------------------------------------------------------------------------------

/// Which API the engine talks to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Preset {
    OpenAi,
    Groq,
    Moonshine,
}

/// `EngineSpec::options`. No `Debug`: it may hold the API key.
#[derive(Default, Deserialize)]
#[serde(rename_all = "camelCase", default)]
struct HttpOptions {
    preset: Option<String>,
    base_url: Option<String>,
    model: Option<String>,
    api_key: Option<String>,
    language: Option<String>,
    prompt: Option<String>,
}

/// Resolved configuration. No `Debug`: it holds the API key.
#[derive(Clone)]
pub(crate) struct HttpConfig {
    pub preset: Preset,
    /// Request URL (`…/audio/transcriptions` or `…/transcribe`).
    pub url: Url,
    /// Base URL (for the Moonshine health probe).
    pub base: String,
    pub model: String,
    api_key: Option<String>,
    pub language: Option<String>,
    pub prompt: Option<String>,
    /// Paced by the Groq limiter.
    pub groq_limits: bool,
}

fn trimmed(v: Option<String>) -> Option<String> {
    v.map(|s| s.trim().to_string()).filter(|s| !s.is_empty())
}

/// Whether `url` is Groq's API (the preset, or a custom base URL typed by hand).
pub(crate) fn is_groq_url(url: &Url) -> bool {
    url.host_str()
        .map(|h| h.to_ascii_lowercase())
        .is_some_and(|h| h == "api.groq.com" || h.ends_with(".groq.com"))
}

impl HttpConfig {
    /// Parse and validate the options of `spec` (no network).
    pub(crate) fn from_spec(spec: &EngineSpec) -> Result<Self, AsrError> {
        let o: HttpOptions = if spec.options.is_null() {
            HttpOptions::default()
        } else {
            serde_json::from_value(spec.options.clone())
                .map_err(|e| AsrError::invalid_input(format!("invalid HTTP engine options: {e}")))?
        };
        let preset = match o.preset.as_deref().map(|p| p.trim().to_ascii_lowercase()).as_deref() {
            None | Some("") | Some("openai") | Some("custom") | Some("openai_audio") => Preset::OpenAi,
            Some("groq") => Preset::Groq,
            Some("moonshine") => Preset::Moonshine,
            Some(other) => return Err(AsrError::invalid_input(format!("unknown HTTP engine preset '{other}'"))),
        };
        let api_key = trimmed(o.api_key);
        let (base, model) = match preset {
            Preset::Groq => {
                if api_key.is_none() {
                    return Err(AsrError::invalid_input("Groq needs an API key (console.groq.com/keys)"));
                }
                (
                    GROQ_BASE_URL.to_string(),
                    trimmed(o.model).unwrap_or_else(|| GROQ_DEFAULT_MODEL.to_string()),
                )
            }
            Preset::Moonshine => (
                trimmed(o.base_url).unwrap_or_else(|| MOONSHINE_DEFAULT_URL.to_string()),
                trimmed(o.model).unwrap_or_else(|| "moonshine".to_string()),
            ),
            Preset::OpenAi => {
                let base = trimmed(o.base_url)
                    .ok_or_else(|| AsrError::invalid_input("the transcription server URL is empty"))?;
                let model =
                    trimmed(o.model).ok_or_else(|| AsrError::invalid_input("the transcription model name is empty"))?;
                (base, model)
            }
        };
        let base = normalize_base_url(&base).map_err(AsrError::invalid_input)?;
        let path = if preset == Preset::Moonshine { "/transcribe" } else { "/audio/transcriptions" };
        let url = api_url(&base, path).map_err(AsrError::invalid_input)?;
        let groq_limits = preset == Preset::Groq || is_groq_url(&url);
        Ok(Self {
            preset,
            url,
            base,
            model,
            api_key,
            language: trimmed(o.language).and_then(|l| super::super::lang::normalize(&l)),
            prompt: trimmed(o.prompt),
            groq_limits,
        })
    }

    /// Runtime label for status and logs.
    fn runtime(&self) -> &'static str {
        match self.preset {
            Preset::Groq => "HTTP (Groq)",
            Preset::Moonshine => "HTTP (Moonshine)",
            Preset::OpenAi if self.groq_limits => "HTTP (Groq)",
            Preset::OpenAi => "HTTP (OpenAI-compatible)",
        }
    }

    /// Request timeout for `seconds` of audio: a self-hosted server may be slow on the CPU.
    fn timeout(&self, seconds: f64) -> Duration {
        if is_local_endpoint(&self.url) {
            STT_TIMEOUT_MAX
        } else {
            stt_timeout(seconds)
        }
    }
}

/// The code an OpenAI-style (Whisper) server accepts: Sigil's code in Whisper's spelling (`nb` →
/// `no`, `fil` → `tl`, `jv` → `jw`, `yue` → `zh`), two letters only. Languages Whisper does not
/// know are left to detection rather than sent (a server rejects them with a 400).
fn api_language(code: Option<&str>) -> Option<String> {
    let c = super::whisper::whisper_code(code?, false)?;
    (c.len() == 2).then(|| c.to_string())
}

/// A rejected request whose error names the language (a server that spells a code differently):
/// the request is retried once without it.
fn rejects_language(message: &str) -> bool {
    message.to_ascii_lowercase().contains("language")
}

/// Prompt with the hotwords appended (`prompt hot1, hot2.`), capped on a char boundary.
fn compose_prompt(prompt: Option<&str>, hotwords: &[String]) -> Option<String> {
    let mut out = prompt.unwrap_or("").trim().to_string();
    let words: Vec<&str> = hotwords.iter().map(|w| w.trim()).filter(|w| !w.is_empty()).collect();
    if !words.is_empty() {
        if !out.is_empty() {
            out.push(' ');
        }
        out.push_str(&words.join(", "));
        out.push('.');
    }
    out.retain(|c| c != '\0');
    if out.len() > MAX_PROMPT_BYTES {
        let mut cut = MAX_PROMPT_BYTES;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
    }
    (!out.is_empty()).then_some(out)
}

// ---------------------------------------------------------------------------------------------
// Encoding
// ---------------------------------------------------------------------------------------------

/// 16-bit PCM mono WAV of `samples` (clamped to [-1, 1]).
pub(crate) fn wav_pcm16(samples: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + samples.len() * 2);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_len).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for &s in samples {
        let s = if s.is_finite() { s.clamp(-1.0, 1.0) } else { 0.0 };
        out.extend_from_slice(&((s * 32767.0).round() as i16).to_le_bytes());
    }
    out
}

/// A `multipart/form-data` body.
pub(crate) struct Multipart {
    boundary: String,
    body: Vec<u8>,
}

impl Multipart {
    pub(crate) fn new() -> Self {
        static SEQ: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let seq = SEQ.fetch_add(1, Ordering::Relaxed);
        Self {
            boundary: format!("----sigil-{nanos:032x}-{seq:08x}"),
            body: Vec::new(),
        }
    }

    pub(crate) fn text(&mut self, name: &str, value: &str) {
        self.body.extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body
            .extend_from_slice(format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n").as_bytes());
        self.body.extend_from_slice(value.as_bytes());
        self.body.extend_from_slice(b"\r\n");
    }

    pub(crate) fn file(&mut self, name: &str, filename: &str, content_type: &str, bytes: &[u8]) {
        self.body.extend_from_slice(format!("--{}\r\n", self.boundary).as_bytes());
        self.body.extend_from_slice(
            format!("Content-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n")
                .as_bytes(),
        );
        self.body.extend_from_slice(bytes);
        self.body.extend_from_slice(b"\r\n");
    }

    /// `(content type, body)`.
    pub(crate) fn finish(mut self) -> (String, Vec<u8>) {
        self.body.extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        (format!("multipart/form-data; boundary={}", self.boundary), self.body)
    }
}

// ---------------------------------------------------------------------------------------------
// Groq pacing (port of the TS GroqRateLimiter; times are ms on a monotonic clock)
// ---------------------------------------------------------------------------------------------

/// Exact sliding window: the weight recorded in any `window_ms` never exceeds `limit`.
#[derive(Clone, Debug)]
pub(crate) struct SlidingWindow {
    window_ms: f64,
    limit: f64,
    entries: VecDeque<(f64, f64)>,
}

impl SlidingWindow {
    pub(crate) fn new(window_ms: f64, limit: f64) -> Self {
        Self {
            window_ms,
            limit,
            entries: VecDeque::new(),
        }
    }

    fn prune(&mut self, now: f64) {
        while self.entries.front().is_some_and(|&(at, _)| at + self.window_ms <= now) {
            self.entries.pop_front();
        }
    }

    pub(crate) fn used(&mut self, now: f64) -> f64 {
        self.prune(now);
        self.entries.iter().map(|&(_, w)| w).sum()
    }

    /// Milliseconds until `weight` fits (a weight above the limit fits only in an empty window).
    pub(crate) fn wait_ms(&mut self, now: f64, weight: f64) -> f64 {
        let need = weight.clamp(0.0, self.limit);
        let mut used = self.used(now);
        if used + need <= self.limit {
            return 0.0;
        }
        let mut last = now;
        for &(at, w) in &self.entries {
            if used + need <= self.limit {
                break;
            }
            used -= w;
            last = at;
        }
        (last + self.window_ms - now).max(0.0)
    }

    pub(crate) fn record(&mut self, now: f64, weight: f64) {
        self.prune(now);
        // Keep entries ordered even if the clock steps back a little.
        let at = self.entries.back().map_or(now, |&(last, _)| last.max(now));
        self.entries.push_back((at, weight));
    }
}

/// Token bucket (`capacity`, refilled at `refill_per_s`); `take` may go into debt down to `min`.
#[derive(Clone, Debug)]
pub(crate) struct TokenBucket {
    capacity: f64,
    refill_per_s: f64,
    tokens: f64,
    last: f64,
    min: f64,
}

impl TokenBucket {
    pub(crate) fn new(capacity: f64, refill_per_s: f64, now: f64, initial: f64, min: f64) -> Self {
        Self {
            capacity,
            refill_per_s,
            tokens: initial.min(capacity),
            last: now,
            min,
        }
    }

    fn refill(&mut self, now: f64) {
        if now <= self.last {
            return;
        }
        self.tokens = (self.tokens + (now - self.last) / 1000.0 * self.refill_per_s).min(self.capacity);
        self.last = now;
    }

    pub(crate) fn available(&mut self, now: f64) -> f64 {
        self.refill(now);
        self.tokens
    }

    /// Milliseconds until `cost` tokens are available (a cost above the capacity waits for a full
    /// bucket).
    pub(crate) fn wait_ms(&mut self, now: f64, cost: f64) -> f64 {
        let need = cost.clamp(0.0, self.capacity);
        let have = self.available(now);
        if have >= need - 1e-6 {
            return 0.0;
        }
        ((need - have) / self.refill_per_s * 1000.0).ceil()
    }

    pub(crate) fn take(&mut self, now: f64, cost: f64) {
        self.refill(now);
        self.tokens = (self.tokens - cost.max(0.0)).max(self.min);
    }
}

/// Groq's free plan for whisper-large-v3(-turbo) (console.groq.com/docs/rate-limits, 2026-09).
#[derive(Clone, Copy, Debug)]
pub(crate) struct GroqPlan {
    pub requests_per_minute: f64,
    pub requests_per_day: f64,
    pub audio_seconds_per_hour: f64,
    /// Every request is billed as at least this much audio.
    pub min_billed_seconds: f64,
}

pub(crate) const GROQ_FREE_PLAN: GroqPlan = GroqPlan {
    requests_per_minute: 20.0,
    requests_per_day: 2000.0,
    audio_seconds_per_hour: 7200.0,
    min_billed_seconds: 10.0,
};

/// Paces requests so the plan's per-minute, per-day and per-hour audio limits are never exceeded
/// (same parameters as the TS `GroqRateLimiter`: 2 s window margin, burst of 5 requests then an even
/// pace, 97 % of the hourly audio with a 900 s burst, and a daily pace for 8 h of talking after 100
/// requests, beyond which short utterances are merged up to the billing minimum).
#[derive(Clone, Debug)]
pub(crate) struct GroqLimiter {
    plan: GroqPlan,
    requests: Option<TokenBucket>,
    per_minute: SlidingWindow,
    per_day: SlidingWindow,
    audio: TokenBucket,
    pace: TokenBucket,
}

impl GroqLimiter {
    pub(crate) fn new(now: f64, plan: GroqPlan) -> Self {
        let margin = 2_000.0;
        let minute_window_s = (60_000.0 + margin) / 1000.0;
        let burst = 5.0f64.min(plan.requests_per_minute - 1.0).max(1.0);
        let requests = (plan.requests_per_minute > burst).then(|| {
            TokenBucket::new(burst, (plan.requests_per_minute - burst) / minute_window_s, now, burst, f64::NEG_INFINITY)
        });
        let hourly = plan.audio_seconds_per_hour * 0.97;
        let audio_burst = 900.0f64.min(hourly / 2.0);
        let pace_hours = 8.0;
        let pace_burst = 100.0f64.min(plan.requests_per_day / 2.0).max(1.0);
        Self {
            plan,
            requests,
            per_minute: SlidingWindow::new(60_000.0 + margin, plan.requests_per_minute),
            per_day: SlidingWindow::new(86_400_000.0 + margin, plan.requests_per_day),
            audio: TokenBucket::new(audio_burst, (hourly - audio_burst) / 3600.0, now, audio_burst, f64::NEG_INFINITY),
            pace: TokenBucket::new(
                pace_burst,
                (plan.requests_per_day - pace_burst) / (pace_hours * 3600.0),
                now,
                pace_burst,
                -pace_burst,
            ),
        }
    }

    fn billed(&self, audio_seconds: f64) -> f64 {
        let a = if audio_seconds.is_finite() { audio_seconds } else { 0.0 };
        a.max(self.plan.min_billed_seconds)
    }

    /// The billing minimum while requests run ahead of the daily pace, else 0.
    pub(crate) fn min_batch_seconds(&mut self, now: f64) -> f64 {
        if self.pace.available(now) >= 1.0 {
            0.0
        } else {
            self.plan.min_billed_seconds.max(0.0)
        }
    }

    /// Milliseconds until a request with `audio_seconds` may start.
    pub(crate) fn wait_ms(&mut self, now: f64, audio_seconds: f64) -> f64 {
        let billed = self.billed(audio_seconds);
        let requests = self.requests.as_mut().map_or(0.0, |b| b.wait_ms(now, 1.0));
        requests
            .max(self.per_minute.wait_ms(now, 1.0))
            .max(self.per_day.wait_ms(now, 1.0))
            .max(self.audio.wait_ms(now, billed))
    }

    pub(crate) fn record(&mut self, now: f64, audio_seconds: f64) {
        let billed = self.billed(audio_seconds);
        self.pace.take(now, 1.0);
        if let Some(b) = self.requests.as_mut() {
            b.take(now, 1.0);
        }
        self.per_minute.record(now, 1.0);
        self.per_day.record(now, 1.0);
        self.audio.take(now, billed);
    }
}

/// Milliseconds on the process's monotonic clock.
fn now_ms() -> f64 {
    static BASE: OnceLock<Instant> = OnceLock::new();
    BASE.get_or_init(Instant::now).elapsed().as_secs_f64() * 1000.0
}

/// The process-wide Groq limiter: restarting STT never resets the request history.
fn groq_limiter() -> &'static Mutex<GroqLimiter> {
    static LIMITER: OnceLock<Mutex<GroqLimiter>> = OnceLock::new();
    LIMITER.get_or_init(|| Mutex::new(GroqLimiter::new(now_ms(), GROQ_FREE_PLAN)))
}

/// The Groq limiter's daily pace as a pipeline batch policy.
struct GroqBatchPolicy;

impl BatchPolicy for GroqBatchPolicy {
    fn min_batch_seconds(&self) -> f32 {
        groq_limiter()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .min_batch_seconds(now_ms()) as f32
    }
}

/// Batch policy of an HTTP spec (Groq's billing minimum); `None` for other servers.
pub(crate) fn batch_policy(spec: &EngineSpec) -> Option<Arc<dyn BatchPolicy>> {
    let config = HttpConfig::from_spec(spec).ok()?;
    config
        .groq_limits
        .then(|| Arc::new(GroqBatchPolicy) as Arc<dyn BatchPolicy>)
}

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

/// How one request failed.
#[derive(Debug, PartialEq)]
pub(crate) enum Failure {
    Cancelled,
    /// Worth another attempt (network, timeout, 408 / 425 / 5xx, a reply that is not the API's:
    /// an HTML page or an unreadable body, which a captive portal, a proxy or a cut connection
    /// produces and the next request may not).
    Retryable(String),
    /// HTTP 429.
    RateLimited { retry_after: Option<Duration>, message: String },
    /// Retrying cannot help: the API itself refused the request (bad key / model / URL, an error
    /// it reported in its answer).
    Fatal(String),
}

/// `Retry-After` as a delay: delta-seconds (an HTTP date is not parsed: the fallback applies).
pub(crate) fn parse_retry_after(headers: &HeaderMap) -> Option<Duration> {
    let v = headers.get(RETRY_AFTER)?.to_str().ok()?.trim();
    let secs: f64 = v.parse().ok()?;
    (secs.is_finite() && secs >= 0.0).then(|| Duration::from_secs_f64(secs.min(86_400.0)))
}

/// Whether an error reply passes by itself. 408, 425 and 5xx do. So does an HTML error page that is
/// not about the key or the URL: no transcription API answers HTML, so a captive portal, a proxy
/// or a firewall in the way did (401, 403, 404, 407 and 410 stay the user's to fix, whatever the
/// body). Any other 4xx is the API refusing the request and is fatal.
fn status_passes(status: StatusCode, html: bool) -> bool {
    status == StatusCode::REQUEST_TIMEOUT
        || status.as_u16() == 425
        || status.is_server_error()
        || (html && status.is_client_error() && !matches!(status.as_u16(), 401 | 403 | 404 | 407 | 410))
}

/// Whether a `Content-Type` is HTML.
fn is_html(content_type: &str) -> bool {
    content_type.trim_start().to_ascii_lowercase().starts_with("text/html")
}

/// Classify a non-success response (reads a bounded part of the body for the message).
async fn failure_of(what: &str, resp: reqwest::Response) -> Failure {
    let status = resp.status();
    let retry_after = parse_retry_after(resp.headers());
    let html = is_html(resp.headers().get(CONTENT_TYPE).and_then(|v| v.to_str().ok()).unwrap_or(""));
    let message = describe_api_error(what, resp).await;
    if status == StatusCode::TOO_MANY_REQUESTS {
        Failure::RateLimited { retry_after, message }
    } else if status_passes(status, html) {
        Failure::Retryable(message)
    } else {
        Failure::Fatal(message)
    }
}

#[derive(Deserialize)]
struct TranscriptionJson {
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    error: Option<serde_json::Value>,
}

/// The text of a successful transcription body (`{"text": …}`, or plain text).
pub(crate) fn parse_transcription(what: &str, content_type: &str, body: &[u8]) -> Result<String, Failure> {
    match serde_json::from_slice::<TranscriptionJson>(body) {
        Ok(j) => {
            if let Some(err) = j.error.filter(|e| !e.is_null()) {
                let msg = err
                    .get("message")
                    .and_then(|m| m.as_str())
                    .map(str::to_string)
                    .or_else(|| err.as_str().map(str::to_string))
                    .unwrap_or_else(|| "the server reported an error".to_string());
                return Err(Failure::Fatal(format!(
                    "{what}: {}",
                    crate::services::http_client::truncate_chars(&msg, 180)
                )));
            }
            Ok(j.text.unwrap_or_default().trim().to_string())
        }
        Err(_) if content_type.starts_with("text/plain") => {
            Ok(String::from_utf8_lossy(body).trim().to_string())
        }
        // A login page or a proxy's answer, or a body cut short: not the API's verdict on the request.
        Err(_) => Err(Failure::Retryable(format!(
            "{what}: the server returned a response that is not JSON (a login page or proxy in the way?)"
        ))),
    }
}

/// Moonshine request body.
#[derive(Serialize)]
struct MoonshineRequest<'a> {
    audio: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<&'a str>,
}

/// One Moonshine `POST /transcribe` with a base64 WAV (shared with the legacy `moonshine-stt`
/// plugin).
pub(crate) async fn moonshine_request(
    url: Url,
    audio_b64: &str,
    language: Option<&str>,
    timeout: Duration,
) -> Result<String, Failure> {
    let what = "Moonshine transcription";
    let response = http_client()
        .post(url)
        .timeout(timeout)
        .json(&MoonshineRequest {
            audio: audio_b64,
            language: language.map(str::trim).filter(|l| !l.is_empty()),
        })
        .send()
        .await
        .map_err(|e| Failure::Retryable(describe_request_error(what, &e, timeout)))?;
    read_transcription(what, response, timeout).await
}

/// Read and parse a transcription response.
async fn read_transcription(what: &str, response: reqwest::Response, timeout: Duration) -> Result<String, Failure> {
    if !response.status().is_success() {
        return Err(failure_of(what, response).await);
    }
    let content_type = response
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let body = response
        .bytes()
        .await
        .map_err(|e| Failure::Retryable(describe_request_error(what, &e, timeout)))?;
    parse_transcription(what, &content_type, &body)
}

/// Interrupts the engine's waits and requests (a cancel generation plus a wake-up).
#[derive(Default)]
struct CancelState {
    generation: AtomicU64,
    notify: Notify,
}

impl CancelState {
    fn cancel(&self) {
        self.generation.fetch_add(1, Ordering::SeqCst);
        self.notify.notify_waiters();
    }

    fn current(&self) -> u64 {
        self.generation.load(Ordering::SeqCst)
    }

    /// Resolves once the generation moved past `mine`.
    async fn cancelled(&self, mine: u64) {
        loop {
            let notified = self.notify.notified();
            tokio::pin!(notified);
            notified.as_mut().enable();
            if self.current() != mine {
                return;
            }
            notified.await;
        }
    }

    /// Sleep `d` unless cancelled first (`Err(Cancelled)`).
    async fn sleep(&self, mine: u64, d: Duration) -> Result<(), Failure> {
        tokio::select! {
            _ = tokio::time::sleep(d) => Ok(()),
            _ = self.cancelled(mine) => Err(Failure::Cancelled),
        }
    }
}

/// The HTTP engine.
pub(crate) struct HttpEngine {
    config: HttpConfig,
    info: EngineInfo,
    status: StatusSink,
    cancel: Arc<CancelState>,
    /// Nothing is sent before this (a 429 from a server without the Groq limiter).
    blocked_until: Option<Instant>,
    /// The last `transcribe` gave its utterance up: a rate-limit wait would pass the deadline.
    past_deadline: bool,
}

impl HttpEngine {
    pub(crate) fn new(config: HttpConfig, status: StatusSink) -> Self {
        let prompts = config.preset != Preset::Moonshine;
        let info = EngineInfo {
            kind: EngineKind::Http,
            model: config.model.clone(),
            runtime: config.runtime().to_string(),
            device: "remote".to_string(),
            caps: EngineCaps {
                streaming: false,
                utterance: true,
                // Whisper-style servers detect the language when none is sent.
                detects_language: true,
                languages: Vec::new(),
                needs_language: false,
                hotwords: prompts,
                prompt: prompts,
                tags: false,
            },
            vram_mb: 0,
            fallback_reason: None,
        };
        Self {
            config,
            info,
            status,
            cancel: Arc::new(CancelState::default()),
            blocked_until: None,
            past_deadline: false,
        }
    }

    fn report(&self, code: &str, detail: String, level: StatusLevel) {
        self.status.report(EngineStatus {
            code: code.to_string(),
            detail,
            level,
            ..EngineStatus::default()
        });
    }

    /// One request (no retries).
    async fn request(&self, wav: &[u8], seconds: f64, language: Option<&str>, prompt: Option<&str>, mine: u64) -> Result<String, Failure> {
        let timeout = self.config.timeout(seconds);
        let fut = async {
            match self.config.preset {
                Preset::Moonshine => {
                    let b64 = STANDARD.encode(wav);
                    moonshine_request(self.config.url.clone(), &b64, language, timeout).await
                }
                Preset::OpenAi | Preset::Groq => {
                    let what = if self.config.groq_limits { "Groq transcription" } else { "HTTP transcription" };
                    let mut form = Multipart::new();
                    form.file("file", "audio.wav", "audio/wav", wav);
                    form.text("model", &self.config.model);
                    form.text("response_format", "json");
                    form.text("temperature", "0");
                    if let Some(l) = api_language(language) {
                        form.text("language", &l);
                    }
                    if let Some(p) = prompt {
                        form.text("prompt", p);
                    }
                    let (content_type, body) = form.finish();
                    let mut req = http_client()
                        .post(self.config.url.clone())
                        .timeout(timeout)
                        .header(CONTENT_TYPE, content_type)
                        .body(body);
                    if let Some(key) = &self.config.api_key {
                        req = req.header(AUTHORIZATION, format!("Bearer {key}"));
                    }
                    let response = req
                        .send()
                        .await
                        .map_err(|e| Failure::Retryable(describe_request_error(what, &e, timeout)))?;
                    read_transcription(what, response, timeout).await
                }
            }
        };
        tokio::select! {
            r = fut => r,
            _ = self.cancel.cancelled(mine) => Err(Failure::Cancelled),
        }
    }

    /// Wait for the Groq limiter (and a server's 429 block), then claim the request. A wait that
    /// would pass `deadline` gives the utterance up (`past_deadline`) instead.
    async fn pace(&mut self, seconds: f64, mine: u64, deadline: Option<Instant>) -> Result<(), AsrError> {
        let mut reported = false;
        loop {
            let block = self
                .blocked_until
                .map_or(Duration::ZERO, |t| t.saturating_duration_since(Instant::now()));
            let limiter_wait = if self.config.groq_limits {
                let mut l = groq_limiter().lock().unwrap_or_else(PoisonError::into_inner);
                let now = now_ms();
                let w = l.wait_ms(now, seconds);
                if w <= 0.0 && block.is_zero() {
                    l.record(now, seconds);
                    return Ok(());
                }
                Duration::from_secs_f64(w.max(0.0) / 1000.0)
            } else if block.is_zero() {
                return Ok(());
            } else {
                Duration::ZERO
            };
            let wait = block.max(limiter_wait);
            if wait > MAX_WAIT {
                return Err(AsrError::new(
                    AsrErrorKind::RateLimited,
                    format!(
                        "the request limit is used up for now (the next request would wait {} min)",
                        wait.as_secs().div_ceil(60)
                    ),
                ));
            }
            if deadline.is_some_and(|d| Instant::now() + wait > d) {
                self.past_deadline = true;
                return Err(AsrError::new(
                    AsrErrorKind::RateLimited,
                    format!("the request limit allows the next request in {:.1} s", wait.as_secs_f32()),
                ));
            }
            if wait >= STATUS_DELAY && !reported {
                reported = true;
                self.report(
                    codes::RATE_LIMITED,
                    format!("waiting {} s for the request limit", wait.as_secs().max(1)),
                    StatusLevel::Warning,
                );
            }
            // Re-plan at least every 5 s (another session may have used the slot meanwhile).
            self.cancel
                .sleep(mine, wait.min(Duration::from_secs(5)).max(Duration::from_millis(20)))
                .await
                .map_err(|_| AsrError::cancelled())?;
        }
    }

    async fn run(
        &mut self,
        wav: Vec<u8>,
        seconds: f64,
        mut language: Option<String>,
        prompt: Option<String>,
        mine: u64,
        deadline: Option<Instant>,
    ) -> Result<String, AsrError> {
        let mut attempts = 0u32;
        let mut waited = Duration::ZERO;
        loop {
            self.pace(seconds, mine, deadline).await?;
            match self.request(&wav, seconds, language.as_deref(), prompt.as_deref(), mine).await {
                Ok(text) => return Ok(text),
                Err(Failure::Cancelled) => return Err(AsrError::cancelled()),
                Err(Failure::Fatal(message)) if language.is_some() && rejects_language(&message) => {
                    // A server that spells this language differently: let it detect instead.
                    tracing::info!(
                        target: LOG,
                        "the server rejected the language '{}' ({message}); retrying without it",
                        language.as_deref().unwrap_or("")
                    );
                    language = None;
                }
                Err(Failure::Fatal(message)) => return Err(AsrError::invalid_input(message)),
                Err(Failure::RateLimited { retry_after, message }) => {
                    let wait = retry_after.unwrap_or(RATE_LIMIT_FALLBACK).max(Duration::from_millis(250));
                    waited += wait;
                    if waited > MAX_WAIT {
                        return Err(AsrError::new(AsrErrorKind::RateLimited, message));
                    }
                    tracing::info!(target: LOG, "rate limited; retrying in {} ms", wait.as_millis());
                    if wait >= STATUS_DELAY {
                        self.report(
                            codes::RATE_LIMITED,
                            format!("rate limited by the server; retrying in {} s", wait.as_secs().max(1)),
                            StatusLevel::Warning,
                        );
                    }
                    self.blocked_until = Some(Instant::now() + wait);
                }
                Err(Failure::Retryable(message)) => {
                    attempts += 1;
                    if attempts >= MAX_ATTEMPTS {
                        return Err(AsrError::new(
                            AsrErrorKind::Network,
                            format!("{message} (failed {attempts} times)"),
                        ));
                    }
                    let wait = RETRY_BASE.saturating_mul(1 << (attempts - 1)).min(RETRY_MAX);
                    if deadline.is_some_and(|d| Instant::now() + wait > d) {
                        return Err(AsrError::new(
                            AsrErrorKind::Network,
                            format!("{message} (failed {attempts} times; the utterance is past its latency budget)"),
                        ));
                    }
                    tracing::info!(target: LOG, "attempt {attempts} failed ({message}); retrying in {} ms", wait.as_millis());
                    self.report(
                        codes::ENGINE_ERROR,
                        format!("{message}; retrying ({attempts}/{MAX_ATTEMPTS})"),
                        StatusLevel::Warning,
                    );
                    self.cancel.sleep(mine, wait).await.map_err(|_| AsrError::cancelled())?;
                }
            }
        }
    }
}

impl AsrEngine for HttpEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
        let started = Instant::now();
        if audio.is_empty() {
            return Ok(AsrResult::default());
        }
        let mine = self.cancel.current();
        // After the snapshot: a stop that came before it is seen here, any later one interrupts
        // the waits and the request through the generation.
        if scope::abandoned() {
            return Err(AsrError::cancelled());
        }
        let deadline = scope::deadline();
        let seconds = audio.len() as f64 / f64::from(SAMPLE_RATE);
        let wav = wav_pcm16(audio, SAMPLE_RATE);
        let language = options.language.clone().or_else(|| self.config.language.clone());
        let prompt = match self.config.preset {
            Preset::Moonshine => None,
            _ => compose_prompt(options.prompt.as_deref().or(self.config.prompt.as_deref()), &options.hotwords),
        };
        let lang_out = language.clone();
        self.past_deadline = false;
        let result = tauri::async_runtime::block_on(self.run(wav, seconds, language, prompt, mine, deadline));
        if self.past_deadline {
            scope::give_up();
        }
        let text = result?;
        Ok(AsrResult {
            text,
            lang: lang_out,
            decode_ms: started.elapsed().as_millis().min(u128::from(u32::MAX)) as u32,
            ..AsrResult::default()
        })
    }

    fn canceller(&self) -> Option<Canceller> {
        let cancel = Arc::clone(&self.cancel);
        Some(Canceller::new(move || cancel.cancel()))
    }
}

/// Create the engine (blocking: Moonshine is health-checked first).
pub fn create(spec: &EngineSpec, ctx: &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError> {
    let config = HttpConfig::from_spec(spec)?;
    if config.preset == Preset::Moonshine {
        let health = api_url(&config.base, "/health").map_err(AsrError::invalid_input)?;
        let shown = redact_url(&health);
        let ok = tauri::async_runtime::block_on(probe_ok(
            "Moonshine availability check",
            http_client().get(health).timeout(HEALTH_TIMEOUT),
        ));
        if !ok {
            return Err(AsrError::new(
                AsrErrorKind::Network,
                format!("the Moonshine server does not answer at {shown}; is it running?"),
            ));
        }
    }
    tracing::info!(
        target: LOG,
        "{} engine for {} (model {})",
        config.runtime(),
        redact_url(&config.url),
        config.model
    );
    Ok(Box::new(HttpEngine::new(config, ctx.status.clone())))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::io::{Read, Write};
    use std::net::TcpListener;
    use std::sync::mpsc;

    fn spec(options: serde_json::Value) -> EngineSpec {
        EngineSpec {
            options,
            ..EngineSpec::new(EngineKind::Http, "")
        }
    }

    #[test]
    fn options_are_validated_per_preset() {
        let groq = HttpConfig::from_spec(&spec(json!({"preset": "groq", "apiKey": " gsk_x "}))).unwrap();
        assert_eq!(groq.url.as_str(), "https://api.groq.com/openai/v1/audio/transcriptions");
        assert_eq!(groq.model, GROQ_DEFAULT_MODEL);
        assert!(groq.groq_limits);
        assert_eq!(groq.api_key.as_deref(), Some("gsk_x"));
        let e = HttpConfig::from_spec(&spec(json!({"preset": "groq"}))).err().unwrap();
        assert_eq!(e.kind, AsrErrorKind::InvalidInput);

        let local = HttpConfig::from_spec(&spec(json!({"baseUrl": "localhost:8000/v1/", "model": "Systran/faster-whisper-small", "language": "ja-JP"}))).unwrap();
        assert_eq!(local.url.as_str(), "http://localhost:8000/v1/audio/transcriptions");
        assert_eq!(local.language.as_deref(), Some("ja"));
        assert!(!local.groq_limits);
        assert_eq!(local.timeout(5.0), STT_TIMEOUT_MAX, "self-hosted servers get the long timeout");
        assert!(HttpConfig::from_spec(&spec(json!({"model": "m"}))).is_err(), "base URL required");
        assert!(HttpConfig::from_spec(&spec(json!({"baseUrl": "http://x"}))).is_err(), "model required");
        assert!(HttpConfig::from_spec(&spec(json!({"baseUrl": "ftp://x", "model": "m"}))).is_err());
        assert!(HttpConfig::from_spec(&spec(json!({"preset": "carrier-pigeon"}))).is_err());

        // A Groq URL typed into the custom preset is still paced.
        let typed = HttpConfig::from_spec(&spec(json!({"baseUrl": "https://api.groq.com/openai/v1", "model": "whisper-large-v3"}))).unwrap();
        assert!(typed.groq_limits);
        assert!(batch_policy(&spec(json!({"baseUrl": "https://api.groq.com/openai/v1", "model": "w"}))).is_some());
        assert!(batch_policy(&spec(json!({"baseUrl": "http://localhost:1", "model": "w"}))).is_none());

        let moon = HttpConfig::from_spec(&spec(json!({"preset": "moonshine"}))).unwrap();
        assert_eq!(moon.url.as_str(), "http://localhost:8090/transcribe");
        assert!(!HttpEngine::new(moon, StatusSink::discard()).info.caps.prompt);
    }

    #[test]
    fn languages_and_prompts_follow_the_openai_api() {
        assert_eq!(api_language(Some("en-US")).as_deref(), Some("en"));
        assert_eq!(api_language(Some("yue")).as_deref(), Some("zh"));
        // Sigil's codes in Whisper's spelling (the API rejects `nb` / `fil` / `jv` with a 400).
        assert_eq!(api_language(Some("nb-NO")).as_deref(), Some("no"));
        assert_eq!(api_language(Some("fil-PH")).as_deref(), Some("tl"));
        assert_eq!(api_language(Some("jv-ID")).as_deref(), Some("jw"));
        assert_eq!(api_language(Some("haw")), None, "three-letter codes are left to detection");
        assert_eq!(api_language(Some("xx")), None, "languages Whisper does not know are not sent");
        assert_eq!(api_language(Some("auto")), None);
        assert_eq!(api_language(None), None);
        assert!(rejects_language("HTTP transcription failed (400): Invalid language 'nb'"));
        assert!(!rejects_language("HTTP transcription failed (401): Invalid API Key"));
        assert_eq!(compose_prompt(Some(" Hi. "), &["Sigil".into(), "".into()]).as_deref(), Some("Hi. Sigil."));
        assert_eq!(compose_prompt(None, &[]), None);
        let long = compose_prompt(Some(&"語".repeat(1000)), &[]).unwrap();
        assert!(long.len() <= MAX_PROMPT_BYTES);
    }

    #[test]
    fn wav_and_multipart_encoding() {
        let wav = wav_pcm16(&[0.0, 1.0, -1.0, 2.0, f32::NAN], 16_000);
        assert_eq!(wav.len(), 44 + 10);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(u32::from_le_bytes(wav[4..8].try_into().unwrap()), 36 + 10);
        assert_eq!(&wav[8..16], b"WAVEfmt ");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16_000);
        assert_eq!(u32::from_le_bytes(wav[40..44].try_into().unwrap()), 10);
        let s: Vec<i16> = wav[44..].chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(s, vec![0, 32767, -32767, 32767, 0]);
        let reader = hound::WavReader::new(std::io::Cursor::new(wav)).unwrap();
        assert_eq!(reader.spec().sample_rate, 16_000);

        let mut m = Multipart::new();
        m.text("model", "whisper-1");
        m.file("file", "audio.wav", "audio/wav", b"RIFFxx");
        let (ct, body) = m.finish();
        let boundary = ct.strip_prefix("multipart/form-data; boundary=").unwrap().to_string();
        let text = String::from_utf8_lossy(&body);
        assert!(text.starts_with(&format!("--{boundary}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n")));
        assert!(text.contains("name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\nRIFFxx\r\n"));
        assert!(text.ends_with(&format!("--{boundary}--\r\n")));
        assert_ne!(Multipart::new().boundary, Multipart::new().boundary);
    }

    #[test]
    fn responses_are_parsed_and_classified() {
        assert_eq!(parse_transcription("t", "application/json", br#"{"text":" hello "}"#), Ok("hello".into()));
        assert_eq!(parse_transcription("t", "application/json", br#"{"segments":[]}"#), Ok(String::new()));
        assert!(matches!(
            parse_transcription("t", "application/json", br#"{"error":{"message":"bad model"}}"#),
            Err(Failure::Fatal(m)) if m.contains("bad model")
        ));
        assert_eq!(parse_transcription("t", "text/plain; charset=utf-8", b" plain "), Ok("plain".into()));
        // An unreadable reply is not the API's verdict: a captive portal's page, a cut body, an empty one.
        for (content_type, body) in [
            ("text/html", &b"<html>"[..]),
            ("application/json", &br#"{"text":"cut off"#[..]),
            ("application/json", &b""[..]),
        ] {
            assert!(
                matches!(parse_transcription("t", content_type, body), Err(Failure::Retryable(_))),
                "{content_type} {body:?}"
            );
        }
        let mut h = HeaderMap::new();
        assert_eq!(parse_retry_after(&h), None);
        h.insert(RETRY_AFTER, "2".parse().unwrap());
        assert_eq!(parse_retry_after(&h), Some(Duration::from_secs(2)));
        h.insert(RETRY_AFTER, "1.5".parse().unwrap());
        assert_eq!(parse_retry_after(&h), Some(Duration::from_millis(1500)));
        h.insert(RETRY_AFTER, "Wed, 21 Oct 2015 07:28:00 GMT".parse().unwrap());
        assert_eq!(parse_retry_after(&h), None);
    }

    #[test]
    fn error_pages_of_a_proxy_pass_and_api_refusals_do_not() {
        let st = |c: u16| StatusCode::from_u16(c).unwrap();
        // Transient statuses pass whatever the body.
        for code in [408, 425, 500, 502, 503, 504] {
            assert!(status_passes(st(code), false), "{code}");
        }
        // The API refusing the request (JSON or plain body) is fatal.
        for code in [400, 401, 403, 404, 413, 415, 422] {
            assert!(!status_passes(st(code), false), "{code}");
        }
        // An HTML error page is a proxy's, a portal's or a firewall's, except where it is about the
        // key or the URL.
        for code in [400, 405, 409, 413, 418, 451] {
            assert!(status_passes(st(code), true), "html {code}");
        }
        for code in [401, 403, 404, 407, 410] {
            assert!(!status_passes(st(code), true), "html {code}");
        }
        assert!(is_html("text/html; charset=utf-8") && is_html(" Text/HTML"));
        assert!(!is_html("application/json") && !is_html(""));
    }

    #[test]
    fn groq_limiter_never_exceeds_twenty_requests_per_minute() {
        let mut l = GroqLimiter::new(0.0, GROQ_FREE_PLAN);
        let mut now = 0.0;
        let mut sent = Vec::new();
        // A fast talker: a 2 s utterance every second for 5 minutes.
        while now < 300_000.0 {
            let w = l.wait_ms(now, 2.0);
            if w <= 0.0 {
                l.record(now, 2.0);
                sent.push(now);
                now += 1_000.0;
            } else {
                now += w;
            }
        }
        for (i, &t) in sent.iter().enumerate() {
            let in_window = sent[i..].iter().take_while(|&&u| u < t + 60_000.0).count();
            assert!(in_window <= 20, "{in_window} requests within a minute from {t}");
        }
        // Burst then an even pace: the first 5 go out back to back.
        assert!(sent[4] - sent[0] <= 4_000.0 + 1e-6);
        // 5 at once, then an even ~14.5 per minute (the bucket refills (20 - 5) per 62 s).
        assert!((70..=100).contains(&sent.len()), "{} requests in 5 minutes", sent.len());
    }

    #[test]
    fn groq_limiter_paces_the_daily_cap_and_asks_for_batches() {
        let mut l = GroqLimiter::new(0.0, GROQ_FREE_PLAN);
        assert_eq!(l.min_batch_seconds(0.0), 0.0);
        for i in 0..100 {
            l.record(f64::from(i) * 10.0, 1.0);
        }
        // 100 requests within a second exhaust the pace bucket: short utterances get merged.
        assert_eq!(l.min_batch_seconds(1_000.0), GROQ_FREE_PLAN.min_billed_seconds);
        // Hours later the pace refilled.
        assert_eq!(l.min_batch_seconds(1_000.0 + 8.0 * 3_600_000.0), 0.0);
        // The daily window is a hard cap.
        let mut d = SlidingWindow::new(1_000.0, 2.0);
        d.record(0.0, 1.0);
        d.record(100.0, 1.0);
        assert_eq!(d.wait_ms(200.0, 1.0), 800.0);
        assert_eq!(d.wait_ms(1_000.0, 1.0), 0.0);
        let mut b = TokenBucket::new(2.0, 1.0, 0.0, 2.0, f64::NEG_INFINITY);
        b.take(0.0, 5.0);
        assert_eq!(b.wait_ms(0.0, 1.0), 4_000.0, "debt is repaid first");
    }

    // --- loopback server ------------------------------------------------------------------------

    /// One recorded request.
    struct Seen {
        head: String,
        body: Vec<u8>,
    }

    /// Serve `responses` (raw status line + headers + body) in order on 127.0.0.1, recording each
    /// request. `delay` holds a response back.
    fn serve(responses: Vec<(String, Duration)>) -> (String, mpsc::Receiver<Seen>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = mpsc::channel();
        std::thread::spawn(move || {
            for (response, delay) in responses {
                let Ok((mut stream, _)) = listener.accept() else { return };
                stream.set_read_timeout(Some(Duration::from_secs(10))).ok();
                let mut buf = Vec::new();
                let mut chunk = [0u8; 8192];
                let head_end = loop {
                    let n = stream.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        break None;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                    if let Some(i) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                        break Some(i + 4);
                    }
                };
                let Some(head_end) = head_end else { continue };
                let head = String::from_utf8_lossy(&buf[..head_end]).to_string();
                let len = head
                    .lines()
                    .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                    .unwrap_or(0);
                while buf.len() < head_end + len {
                    let n = stream.read(&mut chunk).unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                let _ = tx.send(Seen {
                    head,
                    body: buf[head_end..].to_vec(),
                });
                std::thread::sleep(delay);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        (format!("http://127.0.0.1:{}/v1", addr.port()), rx)
    }

    fn reply(status: &str, extra: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n{extra}\r\n{body}",
            body.len()
        )
    }

    fn engine(base: &str) -> HttpEngine {
        let config = HttpConfig::from_spec(&spec(json!({
            "baseUrl": base, "model": "tiny", "apiKey": "sk-secret-123", "prompt": "VRChat"
        })))
        .unwrap();
        HttpEngine::new(config, StatusSink::discard())
    }

    #[test]
    fn transcribes_one_multipart_request_with_the_key_in_the_header() {
        let (base, seen) = serve(vec![(reply("200 OK", "", r#"{"text":"hello there"}"#), Duration::ZERO)]);
        let mut e = engine(&base);
        let options = DecodeOptions {
            language: Some("en".into()),
            hotwords: vec!["Sigil".into()],
            prompt: None,
        };
        let r = e.transcribe(&vec![0.1f32; 16_000], &options).unwrap();
        assert_eq!(r.text, "hello there");
        assert_eq!(r.lang.as_deref(), Some("en"));
        let req = seen.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(req.head.starts_with("POST /v1/audio/transcriptions HTTP/1.1"), "{}", req.head);
        assert!(req.head.to_ascii_lowercase().contains("authorization: bearer sk-secret-123"));
        assert!(!req.head.contains("sk-secret-123?") && !req.head.lines().next().unwrap().contains("sk-"), "never in the URL");
        let body = String::from_utf8_lossy(&req.body);
        assert!(body.contains("name=\"model\"\r\n\r\ntiny\r\n"));
        assert!(body.contains("name=\"language\"\r\n\r\nen\r\n"));
        assert!(body.contains("name=\"prompt\"\r\n\r\nVRChat Sigil.\r\n"));
        assert!(body.contains("name=\"response_format\"\r\n\r\njson\r\n"));
        assert!(body.contains("filename=\"audio.wav\""));
        assert!(req.body.windows(4).any(|w| w == b"RIFF"));
    }

    #[test]
    fn rate_limits_and_server_errors_are_retried_and_bad_requests_are_fatal() {
        let (base, seen) = serve(vec![
            (reply("429 Too Many Requests", "Retry-After: 0\r\n", r#"{"error":{"message":"slow down"}}"#), Duration::ZERO),
            (reply("503 Service Unavailable", "", "{}"), Duration::ZERO),
            (reply("200 OK", "", r#"{"text":"ok"}"#), Duration::ZERO),
        ]);
        let mut e = engine(&base);
        let r = e.transcribe(&[0.1; 1600], &DecodeOptions::default()).unwrap();
        assert_eq!(r.text, "ok");
        assert_eq!(seen.try_iter().count(), 3);

        let (base, _seen) = serve(vec![(reply("401 Unauthorized", "", r#"{"error":{"message":"Invalid API Key"}}"#), Duration::ZERO)]);
        let mut e = engine(&base);
        let err = e.transcribe(&[0.1; 1600], &DecodeOptions::default()).unwrap_err();
        assert_eq!(err.kind, AsrErrorKind::InvalidInput);
        assert!(err.message.contains("401") && err.message.contains("Invalid API Key"), "{err}");
        assert!(!err.message.contains("sk-secret-123"), "the key never appears in errors");
    }

    /// A captive portal answers with its own page (HTTP 200, or an error status): the next
    /// request, once the network is fine again, is a normal one.
    #[test]
    fn a_login_page_in_the_way_is_retried() {
        let page = |status: &str| {
            let body = "<html><body>Sign in to the network</body></html>";
            format!(
                "HTTP/1.1 {status}\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                body.len()
            )
        };
        // A 200 page, then the real answer.
        let (base, seen) = serve(vec![
            (page("200 OK"), Duration::ZERO),
            (reply("200 OK", "", r#"{"text":"back online"}"#), Duration::ZERO),
        ]);
        let mut e = engine(&base);
        assert_eq!(e.transcribe(&[0.1; 1600], &DecodeOptions::default()).unwrap().text, "back online");
        assert_eq!(seen.try_iter().count(), 2);
        // A 403 page is about the key: fatal at once, as before.
        let (base, seen) = serve(vec![(page("403 Forbidden"), Duration::ZERO)]);
        let err = engine(&base).transcribe(&[0.1; 1600], &DecodeOptions::default()).unwrap_err();
        assert_eq!(err.kind, AsrErrorKind::InvalidInput, "{err}");
        assert_eq!(seen.try_iter().count(), 1);
    }

    #[test]
    fn a_rejected_language_is_retried_once_without_it() {
        let (base, seen) = serve(vec![
            (reply("400 Bad Request", "", r#"{"error":{"message":"language 'no' is not supported"}}"#), Duration::ZERO),
            (reply("200 OK", "", r#"{"text":"hei"}"#), Duration::ZERO),
        ]);
        let mut e = engine(&base);
        let options = DecodeOptions {
            language: Some("nb".into()),
            ..DecodeOptions::default()
        };
        assert_eq!(e.transcribe(&[0.1; 1600], &options).unwrap().text, "hei");
        let first = seen.recv_timeout(Duration::from_secs(5)).unwrap();
        let second = seen.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(String::from_utf8_lossy(&first.body).contains("name=\"language\"\r\n\r\nno\r\n"));
        assert!(!String::from_utf8_lossy(&second.body).contains("name=\"language\""));
        // Without a language to drop, the same answer is fatal.
        let (base, _seen) = serve(vec![(
            reply("400 Bad Request", "", r#"{"error":{"message":"language 'no' is not supported"}}"#),
            Duration::ZERO,
        )]);
        let err = engine(&base).transcribe(&[0.1; 1600], &DecodeOptions::default()).unwrap_err();
        assert_eq!(err.kind, AsrErrorKind::InvalidInput);
    }

    #[test]
    fn a_rate_limit_wait_past_the_deadline_gives_the_utterance_up() {
        let (base, seen) = serve(vec![(
            reply("429 Too Many Requests", "Retry-After: 30\r\n", r#"{"error":{"message":"slow down"}}"#),
            Duration::ZERO,
        )]);
        let mut e = engine(&base);
        let started = Instant::now();
        let deadline = Some(Instant::now() + Duration::from_secs(2));
        let (r, gave_up) = scope::run(deadline, None, || e.transcribe(&[0.1; 1600], &DecodeOptions::default()));
        let err = r.unwrap_err();
        assert!(gave_up, "{err}");
        assert_eq!(err.kind, AsrErrorKind::RateLimited);
        assert!(started.elapsed() < Duration::from_secs(2), "it never waits for the 30 s");
        assert_eq!(seen.try_iter().count(), 1);
        // The block holds for the next utterance too: it is given up without a request.
        let (r, gave_up) = scope::run(deadline, None, || e.transcribe(&[0.1; 1600], &DecodeOptions::default()));
        assert!(gave_up && r.is_err());
        // Outside a deadline scope (no pipeline budget) nothing is given up early.
        assert!(scope::deadline().is_none());
    }

    #[test]
    fn a_decode_for_an_ended_session_sends_nothing() {
        let (base, seen) = serve(vec![(reply("200 OK", "", r#"{"text":"late"}"#), Duration::ZERO)]);
        let mut e = engine(&base);
        let ended: scope::Live = Arc::new(|| false);
        let (r, gave_up) = scope::run(None, Some(ended), || e.transcribe(&[0.1; 1600], &DecodeOptions::default()));
        assert!(r.unwrap_err().is_cancelled() && !gave_up);
        assert!(seen.recv_timeout(Duration::from_millis(200)).is_err(), "no request was sent");
    }

    #[test]
    fn a_cancel_interrupts_a_slow_request() {
        let (base, _seen) = serve(vec![(reply("200 OK", "", r#"{"text":"late"}"#), Duration::from_secs(5))]);
        let mut e = engine(&base);
        let canceller = e.canceller().unwrap();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(150));
            canceller.cancel();
        });
        let started = Instant::now();
        let err = e.transcribe(&[0.1; 1600], &DecodeOptions::default()).unwrap_err();
        t.join().unwrap();
        assert!(err.is_cancelled(), "{err}");
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn moonshine_posts_base64_json() {
        let (base, seen) = serve(vec![(reply("200 OK", "", r#"{"text":"moon"}"#), Duration::ZERO)]);
        let base = base.trim_end_matches("/v1").to_string();
        let config = HttpConfig::from_spec(&spec(json!({"preset": "moonshine", "baseUrl": base}))).unwrap();
        let mut e = HttpEngine::new(config, StatusSink::discard());
        let r = e
            .transcribe(&[0.1; 1600], &DecodeOptions { language: Some("en".into()), ..DecodeOptions::default() })
            .unwrap();
        assert_eq!(r.text, "moon");
        let req = seen.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(req.head.starts_with("POST /transcribe HTTP/1.1"));
        let body: serde_json::Value = serde_json::from_slice(&req.body).unwrap();
        assert_eq!(body["language"], "en");
        let wav = STANDARD.decode(body["audio"].as_str().unwrap()).unwrap();
        assert_eq!(&wav[..4], b"RIFF");
    }
}
