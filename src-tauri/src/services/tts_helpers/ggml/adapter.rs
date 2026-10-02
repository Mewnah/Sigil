//! In-process request adapter for helper servers that do not take Sigil's request shape (Breeze's
//! native form request, s2.cpp's multipart `/generate`), so TypeScript speaks to every ggml helper
//! through the same OpenAI-compatible client (`openai_speech_speak`).
//!
//! - Listens on `127.0.0.1` (random port) and requires its own random key (`Authorization:
//!   Bearer`, compared in constant time) on every route but `GET /health`. Requests that carry an
//!   `Origin` header (web pages) or a `Host` other than 127.0.0.1 / localhost / [::1] (DNS
//!   rebinding) are refused.
//! - `POST /v1/audio/speech` with an OpenAI-style JSON body (`input`, `voice`, `instructions`,
//!   `stream`, sampling members; `sigil_reference_voice` names a voice-library voice to clone,
//!   read here so no audio crosses IPC) becomes the engine's own request, sent to the helper's
//!   current port (it can change after a restart; connection errors are retried while the process
//!   host restarts it and the model loads again: at least 30 s, up to the engine's load timeout,
//!   and only while the client still waits).
//! - The answer streams back as `audio/pcm; rate=<n>` (the server's sample-rate header), or
//!   `audio/wav` when the server sent a WAV. Upstream errors keep their status (busy becomes 503)
//!   with an OpenAI-style `{"error": {"message"}}` body; a helper that cannot be reached is a 502
//!   whose message says "cannot connect" (the TS backend then finds the helper again and retries).
//! - One request at a time (both servers refuse a second one with 409 / 503). The exchange with
//!   the engine runs in its own task that holds the one-request lock, so a client that goes away
//!   (stop, skip) neither cancels a request the engine already works on nor frees the lock early:
//!   whether it leaves before the answer begins (a whole-clip server answers only when the sentence
//!   is done; hyper drops the handler as soon as the client's connection closes) or mid-stream, the
//!   rest of the answer is read and discarded before the next request is sent. That also keeps
//!   Breeze-TTS-2.cpp's issue #20 away (its lock stays taken when a streaming client disconnects),
//!   and s2.cpp keeps generating an abandoned sentence anyway.
//! - Never forwards a voice path or a voice folder (s2.cpp would read any `.s2voice` path it is
//!   given); voices are files Sigil reads from its own library.
//! - The adapter only protects its own port: the upstream server's loopback port stays keyless
//!   (see the residual risk in [`super`]).

use std::convert::Infallible;
use std::path::{Path, PathBuf};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::{Stream, StreamExt};
use serde_json::{json, Map, Value};
use tokio::sync::{oneshot, OwnedMutexGuard};
use warp::http::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE, HOST, ORIGIN};
use warp::http::{Response, StatusCode};
use warp::hyper::body::{Body, Bytes};
use warp::Filter;

use super::launch::AdapterKind;
use super::runner::{loopback_client, TARGET};
use crate::services::{diagnostics, process_host, voice_library};

/// Largest request body (JSON).
const MAX_BODY: u64 = 256 * 1024;
/// Longest text forwarded (characters).
pub const MAX_INPUT_CHARS: usize = 4000;
/// Longest instruction forwarded (characters).
const MAX_INSTRUCTION_CHARS: usize = 1000;
/// Largest reference clip read from the library.
const MAX_REFERENCE_BYTES: u64 = 16 << 20;
/// How long a request waits for the one before it (and its drain).
const BUSY_WAIT: Duration = Duration::from_secs(180);
/// Shortest time connection errors are retried while the process host restarts the helper (the
/// runner passes the engine's load timeout when that is longer: a restarted s2.cpp listens only
/// once its model is loaded again).
pub const RECONNECT_FOR: Duration = Duration::from_secs(30);
/// Longest one engine request may take, answer and body (a whole-clip server answers only when the
/// sentence is done; the TTS sink drops a sentence long before this).
const ANSWER_TIMEOUT: Duration = Duration::from_secs(600);
/// Longest drain of an abandoned answer.
const DRAIN_LIMIT: Duration = Duration::from_secs(180);
/// Bytes of an upstream error body kept for the message.
const ERROR_BODY_LIMIT: usize = 4096;

/// Where the adapter sends requests.
#[derive(Clone, Debug)]
pub enum Upstream {
    /// The process host helper of that name (its current port).
    Helper(String),
    /// A fixed base URL (tests).
    #[cfg(test)]
    Fixed(String),
}

impl Upstream {
    fn base(&self) -> Option<String> {
        match self {
            Upstream::Helper(name) => process_host::handle(name)?.base_url,
            #[cfg(test)]
            Upstream::Fixed(url) => Some(url.clone()),
        }
    }
}

/// A running adapter; dropping it shuts the listener down (a request in flight finishes).
pub struct Adapter {
    port: u16,
    key: String,
    shutdown: Option<oneshot::Sender<()>>,
}

impl std::fmt::Debug for Adapter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Adapter").field("port", &self.port).finish_non_exhaustive()
    }
}

impl Adapter {
    /// `http://127.0.0.1:{port}/v1`.
    pub fn base_url(&self) -> String {
        format!("http://127.0.0.1:{}/v1", self.port)
    }

    pub fn key(&self) -> &str {
        &self.key
    }

    pub fn port(&self) -> u16 {
        self.port
    }
}

impl Drop for Adapter {
    fn drop(&mut self) {
        if let Some(tx) = self.shutdown.take() {
            let _ = tx.send(());
        }
    }
}

struct Ctx {
    kind: AdapterKind,
    upstream: Upstream,
    key: String,
    /// Sigil's app data (the voice library).
    app_data: Option<PathBuf>,
    /// How long connection errors are retried (at least [`RECONNECT_FOR`]).
    reconnect_for: Duration,
    busy: Arc<tokio::sync::Mutex<()>>,
}

/// Start an adapter for `kind` in front of `upstream`; connection errors are retried for
/// `reconnect_for` (at least [`RECONNECT_FOR`]).
pub async fn start(kind: AdapterKind, upstream: Upstream, reconnect_for: Duration) -> Result<Adapter, String> {
    start_with(kind, upstream, diagnostics::app_data_dir(), reconnect_for).await
}

/// [`start`] with an explicit app-data folder (tests).
pub async fn start_with(kind: AdapterKind, upstream: Upstream, app_data: Option<PathBuf>, reconnect_for: Duration) -> Result<Adapter, String> {
    let key = process_host::spawn::random_key()?;
    let ctx = Arc::new(Ctx {
        kind,
        upstream,
        key: key.clone(),
        app_data,
        reconnect_for: reconnect_for.max(RECONNECT_FOR),
        busy: Arc::new(tokio::sync::Mutex::new(())),
    });
    let health = warp::path!("health")
        .and(warp::get())
        .map(|| json_response(StatusCode::OK, json!({ "status": "ok" })));
    let with_ctx = warp::any().map(move || ctx.clone());
    let speech = warp::path!("v1" / "audio" / "speech")
        .and(warp::post())
        .and(warp::header::headers_cloned())
        .and(warp::body::content_length_limit(MAX_BODY))
        .and(warp::body::bytes())
        .and(with_ctx)
        .and_then(|headers: HeaderMap, body: Bytes, ctx: Arc<Ctx>| async move { Ok::<_, Infallible>(handle_speech(&headers, &body, &ctx).await) });
    let routes = health.or(speech).unify().recover(|r: warp::Rejection| async move {
        let (status, message) = if r.is_not_found() {
            (StatusCode::NOT_FOUND, "not found")
        } else if r.find::<warp::reject::PayloadTooLarge>().is_some() {
            (StatusCode::PAYLOAD_TOO_LARGE, "request body too large")
        } else if r.find::<warp::reject::MethodNotAllowed>().is_some() {
            (StatusCode::METHOD_NOT_ALLOWED, "method not allowed")
        } else {
            (StatusCode::BAD_REQUEST, "bad request")
        };
        Ok::<_, Infallible>(error_response(status, message))
    });
    let (tx, rx) = oneshot::channel::<()>();
    let (addr, server) = warp::serve(routes)
        .try_bind_with_graceful_shutdown(([127, 0, 0, 1], 0), async {
            let _ = rx.await;
        })
        .map_err(|e| format!("cannot open the speech adapter on 127.0.0.1: {e}"))?;
    tokio::spawn(server);
    tracing::debug!(target: TARGET, "{kind:?} speech adapter on 127.0.0.1:{}", addr.port());
    Ok(Adapter {
        port: addr.port(),
        key,
        shutdown: Some(tx),
    })
}

// ---------------------------------------------------------------------------------------------
// Request checks
// ---------------------------------------------------------------------------------------------

fn json_response(status: StatusCode, value: Value) -> Response<Body> {
    let mut r = Response::new(Body::from(value.to_string()));
    *r.status_mut() = status;
    r.headers_mut().insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
    r
}

/// An OpenAI-style error answer.
fn error_response(status: StatusCode, message: &str) -> Response<Body> {
    json_response(status, json!({ "error": { "message": message, "type": "sigil_adapter" } }))
}

/// Constant-time comparison (the key's length is not secret: always 64 hex characters).
pub fn same_secret(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b).fold(0u8, |acc, (x, y)| acc | (x ^ y)) == 0
}

/// The host name of a `Host` header (`127.0.0.1:5000` -> `127.0.0.1`, `[::1]:80` -> `[::1]`).
fn host_name(value: &str) -> &str {
    let v = value.trim();
    if v.starts_with('[') {
        return v.split_inclusive(']').next().unwrap_or(v);
    }
    v.rsplit_once(':').map_or(v, |(h, _)| h)
}

/// Why a request is refused before its body is read (`None` = accepted).
pub fn check_request(headers: &HeaderMap, key: &str) -> Option<(StatusCode, &'static str)> {
    if headers.contains_key(ORIGIN) {
        return Some((StatusCode::FORBIDDEN, "requests from web pages are not accepted"));
    }
    let host_ok = headers
        .get(HOST)
        .and_then(|h| h.to_str().ok())
        .map(host_name)
        .is_some_and(|h| h.eq_ignore_ascii_case("127.0.0.1") || h.eq_ignore_ascii_case("localhost") || h == "[::1]");
    if !host_ok {
        return Some((StatusCode::FORBIDDEN, "only requests to 127.0.0.1 are accepted"));
    }
    let presented = headers
        .get(AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim)
        .unwrap_or("");
    if !same_secret(presented.as_bytes(), key.as_bytes()) {
        return Some((StatusCode::UNAUTHORIZED, "missing or wrong API key"));
    }
    None
}

// ---------------------------------------------------------------------------------------------
// The engines' requests
// ---------------------------------------------------------------------------------------------

/// A voice-library clip to clone.
#[derive(Clone, Debug)]
pub struct Reference {
    pub wav: Vec<u8>,
    pub transcript: String,
}

/// Read library voice `id` (blocking).
pub fn load_reference(app_data: &Path, id: &str) -> Result<Reference, String> {
    let clip = voice_library::reference_clip(app_data, id)?;
    let transcript = clip.transcript.as_deref().map(str::trim).unwrap_or("").to_string();
    if transcript.is_empty() {
        return Err(format!("the library voice {:?} has no transcript yet: wait for it, or type it in the voice library", clip.name));
    }
    let len = std::fs::metadata(&clip.path).map_err(|e| format!("cannot read {}: {e}", clip.path.display()))?.len();
    if len > MAX_REFERENCE_BYTES {
        return Err(format!("the clip of {:?} is too large ({len} bytes)", clip.name));
    }
    let wav = std::fs::read(&clip.path).map_err(|e| format!("cannot read {}: {e}", clip.path.display()))?;
    Ok(Reference { wav, transcript })
}

/// A `multipart/form-data` body.
pub struct Multipart {
    boundary: String,
    body: Vec<u8>,
}

impl Multipart {
    pub fn new() -> Multipart {
        let mut raw = [0u8; 12];
        let suffix = match process_host::spawn::os_random(&mut raw) {
            Ok(()) => hex::encode(raw),
            Err(_) => {
                let nanos = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos())
                    .unwrap_or_default();
                format!("{:x}", nanos ^ u128::from(std::process::id()))
            }
        };
        Multipart {
            boundary: format!("----SigilAdapter{suffix}"),
            body: Vec::new(),
        }
    }

    fn head(&mut self, disposition: &str, content_type: Option<&str>) {
        self.body.extend_from_slice(b"--");
        self.body.extend_from_slice(self.boundary.as_bytes());
        self.body.extend_from_slice(b"\r\nContent-Disposition: form-data; ");
        self.body.extend_from_slice(disposition.as_bytes());
        self.body.extend_from_slice(b"\r\n");
        if let Some(ct) = content_type {
            self.body.extend_from_slice(b"Content-Type: ");
            self.body.extend_from_slice(ct.as_bytes());
            self.body.extend_from_slice(b"\r\n");
        }
        self.body.extend_from_slice(b"\r\n");
    }

    /// A text field (`name` is one of the adapter's own constants).
    pub fn text(&mut self, name: &str, value: &str) {
        self.head(&format!("name=\"{name}\""), None);
        self.body.extend_from_slice(value.as_bytes());
        self.body.extend_from_slice(b"\r\n");
    }

    pub fn file(&mut self, name: &str, filename: &str, content_type: &str, bytes: &[u8]) {
        self.head(&format!("name=\"{name}\"; filename=\"{filename}\""), Some(content_type));
        self.body.extend_from_slice(bytes);
        self.body.extend_from_slice(b"\r\n");
    }

    /// `(content type, body)`.
    pub fn finish(mut self) -> (String, Vec<u8>) {
        self.body.extend_from_slice(b"--");
        self.body.extend_from_slice(self.boundary.as_bytes());
        self.body.extend_from_slice(b"--\r\n");
        (format!("multipart/form-data; boundary={}", self.boundary), self.body)
    }
}

impl Default for Multipart {
    fn default() -> Self {
        Self::new()
    }
}

/// A finite number member of `body`, formatted for a form field.
fn number_field(body: &Map<String, Value>, key: &str, integer: bool) -> Option<String> {
    let v = body.get(key)?;
    let n = v.as_f64().or_else(|| v.as_str().and_then(|s| s.trim().parse::<f64>().ok()))?;
    if !n.is_finite() {
        return None;
    }
    Some(if integer { format!("{}", n.round() as i64) } else { format!("{n}") })
}

fn text_member<'a>(body: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    body.get(key).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty())
}

/// A saved-voice id Breeze can look up (no paths).
fn plain_voice_id(v: &str) -> bool {
    !v.is_empty() && v.len() <= 64 && v.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-' | '.')) && !v.contains("..")
}

/// Truncate to `max` characters.
fn clip_chars(s: &str, max: usize) -> String {
    s.chars().take(max).collect()
}

/// The engine request for an OpenAI-style `body`: `(path, content type, body)`.
pub fn build_upstream(kind: AdapterKind, body: &Map<String, Value>, input: &str, reference: Option<&Reference>) -> (&'static str, String, Vec<u8>) {
    let mut form = Multipart::new();
    match kind {
        AdapterKind::Breeze => {
            form.text("text", input);
            if let Some(i) = text_member(body, "instructions").or_else(|| text_member(body, "instruction")) {
                form.text("instruction", &clip_chars(i, MAX_INSTRUCTION_CHARS));
            }
            if reference.is_none() {
                if let Some(v) = text_member(body, "voice").filter(|v| plain_voice_id(v) && !v.eq_ignore_ascii_case("default")) {
                    form.text("voice_id", v);
                }
            }
            for (key, integer) in [
                ("cfg_scale", false),
                ("seed", true),
                ("temperature", false),
                ("top_k", true),
                ("top_p", false),
                ("repetition_penalty", false),
                ("max_new_tokens", true),
                ("split_chars", true),
            ] {
                if let Some(v) = number_field(body, key, integer) {
                    form.text(key, &v);
                }
            }
            if let Some(r) = reference {
                form.file("ref_audio", "reference.wav", "audio/wav", &r.wav);
                form.text("ref_text", &r.transcript);
            }
            let (ct, bytes) = form.finish();
            ("/v1/audio/speech", ct, bytes)
        }
        AdapterKind::S2 => {
            form.text("text", input);
            let mut params = Map::new();
            params.insert("output_format".into(), Value::from("pcm_s16le"));
            // Live chunks re-decode the confirmed prefix (slower overall); whole clips by default.
            params.insert("chunked".into(), Value::Bool(body.get("stream").and_then(Value::as_bool).unwrap_or(false)));
            for (key, integer) in [
                ("temperature", false),
                ("top_p", false),
                ("top_k", true),
                ("max_new_tokens", true),
                ("min_tokens_before_end", true),
                ("sentence_pause_ms", true),
            ] {
                if let Some(v) = number_field(body, key, integer) {
                    let n: Value = if integer {
                        v.parse::<i64>().map(Value::from).unwrap_or(Value::Null)
                    } else {
                        v.parse::<f64>().map(Value::from).unwrap_or(Value::Null)
                    };
                    if !n.is_null() {
                        params.insert(key.into(), n);
                    }
                }
            }
            for key in ["segment_sentences", "low_latency"] {
                if let Some(b) = body.get(key).and_then(Value::as_bool) {
                    params.insert(key.into(), Value::Bool(b));
                }
            }
            form.text("params", &Value::Object(params).to_string());
            if let Some(r) = reference {
                form.file("reference", "reference.wav", "audio/wav", &r.wav);
                form.text("reference_text", &r.transcript);
            }
            let (ct, bytes) = form.finish();
            ("/generate", ct, bytes)
        }
    }
}

/// Sample rate the engine streams when its answer names none.
pub fn default_rate(kind: AdapterKind) -> u32 {
    match kind {
        AdapterKind::Breeze => 24_000,
        AdapterKind::S2 => 44_100,
    }
}

/// The answer's sample rate (`X-Sample-Rate` from Breeze, `X-Audio-Sample-Rate` from s2.cpp, or a
/// `rate=` content-type parameter).
pub fn answer_rate(headers: &reqwest::header::HeaderMap, kind: AdapterKind) -> u32 {
    let from = |name: &str| {
        headers
            .get(name)
            .and_then(|v| v.to_str().ok())
            .and_then(|v| v.trim().parse::<u32>().ok())
    };
    from("x-sample-rate")
        .or_else(|| from("x-audio-sample-rate"))
        .or_else(|| {
            headers
                .get(reqwest::header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .and_then(super::runner::rate_of_content_type)
        })
        .filter(|r| (8_000..=192_000).contains(r))
        .unwrap_or_else(|| default_rate(kind))
}

/// The status Sigil's client sees for an upstream error.
pub fn map_upstream_status(status: u16) -> StatusCode {
    match status {
        // The other request is still running: the client may retry.
        409 | 503 => StatusCode::SERVICE_UNAVAILABLE,
        400 | 404 | 413 | 422 => StatusCode::from_u16(status).unwrap_or(StatusCode::BAD_REQUEST),
        _ => StatusCode::BAD_GATEWAY,
    }
}

/// A readable message from an upstream error body (JSON `error.message` / `detail` / `message`, or
/// the text).
pub fn upstream_message(body: &[u8]) -> String {
    let text = String::from_utf8_lossy(&body[..body.len().min(ERROR_BODY_LIMIT)]).trim().to_string();
    if let Ok(v) = serde_json::from_str::<Value>(&text) {
        for ptr in ["/error/message", "/error", "/detail", "/message"] {
            if let Some(s) = v.pointer(ptr).and_then(Value::as_str) {
                return s.to_string();
            }
        }
    }
    text.chars().take(300).collect()
}

async fn handle_speech(headers: &HeaderMap, raw: &Bytes, ctx: &Arc<Ctx>) -> Response<Body> {
    if let Some((status, message)) = check_request(headers, &ctx.key) {
        return error_response(status, message);
    }
    let Ok(Value::Object(body)) = serde_json::from_slice::<Value>(raw) else {
        return error_response(StatusCode::BAD_REQUEST, "the request body must be a JSON object");
    };
    let input = text_member(&body, "input").map(|s| clip_chars(s, MAX_INPUT_CHARS)).unwrap_or_default();
    if input.is_empty() {
        return error_response(StatusCode::BAD_REQUEST, "input is required");
    }
    let reference = match text_member(&body, "sigil_reference_voice") {
        Some(id) => {
            let Some(app_data) = ctx.app_data.clone() else {
                return error_response(StatusCode::INTERNAL_SERVER_ERROR, "Sigil's app-data folder is unknown");
            };
            let id = id.to_string();
            match tauri::async_runtime::spawn_blocking(move || load_reference(&app_data, &id)).await {
                Ok(Ok(r)) => Some(r),
                Ok(Err(e)) => return error_response(StatusCode::BAD_REQUEST, &e),
                Err(e) => return error_response(StatusCode::INTERNAL_SERVER_ERROR, &format!("reading the voice failed: {e}")),
            }
        }
        None => None,
    };
    let guard = match tokio::time::timeout(BUSY_WAIT, ctx.busy.clone().lock_owned()).await {
        Ok(g) => g,
        Err(_) => return error_response(StatusCode::SERVICE_UNAVAILABLE, "the voice engine is busy with another sentence"),
    };
    let (path, content_type, bytes) = build_upstream(ctx.kind, &body, &input, reference.as_ref());
    // The exchange runs in its own task, which owns the one-request lock: when the client goes
    // away, hyper drops this handler (and `rx`), but a request the engine already works on is
    // neither cancelled nor followed by the next one before its answer ended. The task then gets
    // its answer back from `send` and drops it, which drains a streamed answer under the lock
    // ([`Relay`]'s drop).
    let (tx, rx) = oneshot::channel::<Response<Body>>();
    let (kind, upstream, reconnect_for) = (ctx.kind, ctx.upstream.clone(), ctx.reconnect_for);
    tokio::spawn(async move {
        let answer = exchange(kind, &upstream, reconnect_for, guard, path, content_type, bytes, || tx.is_closed()).await;
        if let Err(abandoned) = tx.send(answer) {
            drop(abandoned);
        }
    });
    rx.await
        .unwrap_or_else(|_| error_response(StatusCode::INTERNAL_SERVER_ERROR, "the speech adapter's request task failed"))
}

/// One engine request under the one-request lock (`guard`): the answer to relay, or an error
/// answer. `abandoned` says whether the client went away (connection retries then stop: the engine
/// never received the request).
#[allow(clippy::too_many_arguments)]
async fn exchange<F: Fn() -> bool>(
    kind: AdapterKind,
    upstream: &Upstream,
    reconnect_for: Duration,
    guard: OwnedMutexGuard<()>,
    path: &str,
    content_type: String,
    bytes: Vec<u8>,
    abandoned: F,
) -> Response<Body> {
    let resp = match send(upstream, path, &content_type, bytes, reconnect_for, &abandoned).await {
        Ok(r) => r,
        Err(e) => return error_response(StatusCode::BAD_GATEWAY, &e),
    };
    let status = resp.status();
    if !status.is_success() {
        let mut collected = Vec::new();
        let mut stream = resp.bytes_stream();
        // A server that never ends its error body still gets its first bytes reported.
        let _read_in_time = tokio::time::timeout(Duration::from_secs(10), async {
            while let Some(Ok(chunk)) = stream.next().await {
                collected.extend_from_slice(&chunk);
                if collected.len() >= ERROR_BODY_LIMIT {
                    break;
                }
            }
        })
        .await
        .is_ok();
        let message = upstream_message(&collected);
        let message = if message.is_empty() { format!("the voice engine answered {status}") } else { message };
        return error_response(map_upstream_status(status.as_u16()), &message);
    }
    let rate = answer_rate(resp.headers(), kind);
    let wav = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| {
            let v = v.to_ascii_lowercase();
            v.starts_with("audio/wav") || v.starts_with("audio/x-wav") || v.starts_with("audio/wave")
        });
    let ctype = if wav { "audio/wav".to_string() } else { format!("audio/pcm; rate={rate}") };
    let relay = Relay {
        inner: Some(Box::pin(resp.bytes_stream())),
        guard: Some(guard),
        finished: false,
    };
    let mut r = Response::new(Body::wrap_stream(relay));
    *r.status_mut() = StatusCode::OK;
    if let Ok(v) = HeaderValue::from_str(&ctype) {
        r.headers_mut().insert(CONTENT_TYPE, v);
    }
    r
}

/// The client went away while the engine could not be reached yet (nothing was sent to it).
const ABANDONED: &str = "the request was abandoned before the voice engine could be reached";

/// Send the engine request to the helper's current port, retrying connection errors for
/// `reconnect_for` while the process host restarts it (and its model loads), as long as the
/// client still waits (`abandoned`).
async fn send<F: Fn() -> bool>(
    upstream: &Upstream,
    path: &str,
    content_type: &str,
    body: Vec<u8>,
    reconnect_for: Duration,
    abandoned: &F,
) -> Result<reqwest::Response, String> {
    let deadline = Instant::now() + reconnect_for;
    loop {
        let Some(base) = upstream.base() else {
            if Instant::now() >= deadline {
                return Err("the voice engine is not running (cannot connect): it stopped, or is still restarting".into());
            }
            if abandoned() {
                return Err(ABANDONED.into());
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
            continue;
        };
        let result = loopback_client()
            .post(format!("{}{path}", base.trim_end_matches('/')))
            .header(reqwest::header::CONTENT_TYPE, content_type)
            .body(body.clone())
            .timeout(ANSWER_TIMEOUT)
            .send()
            .await;
        match result {
            Ok(r) => return Ok(r),
            // A connection error means the engine never got the request: safe to try again.
            Err(e) if e.is_connect() && Instant::now() < deadline => {
                if abandoned() {
                    return Err(ABANDONED.into());
                }
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            Err(e) => return Err(describe_send_error(&e, reconnect_for)),
        }
    }
}

/// The 502 message of a failed engine request; a connection failure says "cannot connect" (the TS
/// backend's cue to find the helper again and retry the sentence once).
pub fn describe_send_error(e: &reqwest::Error, reconnect_for: Duration) -> String {
    let mut cause = e.to_string();
    let mut source = std::error::Error::source(e);
    while let Some(s) = source {
        cause = s.to_string();
        source = s.source();
    }
    if e.is_connect() {
        format!(
            "the voice engine is not running or still loading (cannot connect after {} s): {cause}",
            reconnect_for.as_secs()
        )
    } else if e.is_timeout() {
        format!("the voice engine did not answer within {} s", ANSWER_TIMEOUT.as_secs())
    } else {
        format!("the voice engine did not answer: {cause}")
    }
}

type ByteStream = Pin<Box<dyn Stream<Item = reqwest::Result<Bytes>> + Send>>;

/// The upstream answer, streamed to the client; holds the one-request lock until it ends, and
/// drains the rest when the client goes away.
struct Relay {
    inner: Option<ByteStream>,
    guard: Option<OwnedMutexGuard<()>>,
    finished: bool,
}

impl Stream for Relay {
    type Item = reqwest::Result<Bytes>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let Some(inner) = this.inner.as_mut() else {
            return Poll::Ready(None);
        };
        match inner.as_mut().poll_next(cx) {
            Poll::Ready(None) => {
                this.finished = true;
                this.inner = None;
                this.guard = None;
                Poll::Ready(None)
            }
            Poll::Ready(Some(Err(e))) => {
                this.finished = true;
                this.inner = None;
                this.guard = None;
                Poll::Ready(Some(Err(e)))
            }
            other => other,
        }
    }
}

impl Drop for Relay {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        let (Some(mut inner), guard) = (self.inner.take(), self.guard.take()) else {
            return;
        };
        if let Ok(handle) = tokio::runtime::Handle::try_current() {
            handle.spawn(async move {
                let drained = tokio::time::timeout(DRAIN_LIMIT, async {
                    let mut bytes = 0usize;
                    while let Some(Ok(chunk)) = inner.next().await {
                        bytes += chunk.len();
                    }
                    bytes
                })
                .await;
                match drained {
                    Ok(bytes) => tracing::debug!(target: TARGET, "speech adapter: drained {bytes} bytes of an abandoned answer"),
                    Err(_) => tracing::warn!(target: TARGET, "speech adapter: an abandoned answer did not end within {} s", DRAIN_LIMIT.as_secs()),
                }
                drop(guard);
            });
        }
    }
}
