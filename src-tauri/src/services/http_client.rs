//! Shared outbound HTTP for plugins: local/cloud TTS and STT servers, translation, model downloads.
//!
//! # Timeouts (no call may hang for hours)
//! The shared [`http_client`] keeps a 2 h whole-request cap because multi-GB model downloads go through
//! it ([`http_download_get`]), plus a [`STALL_TIMEOUT`] read timeout: a connection that delivers no
//! bytes for that long fails, downloads included. **Every API call must also set a per-request
//! timeout** with `.timeout(...)`, using the classes below:
//! - [`HEALTH_TIMEOUT`] (10 s) for availability probes,
//! - [`LIST_TIMEOUT`] (20 s) for voice / language / model lists,
//! - [`translate_timeout`] (20 s, a little more for long text) for machine translation,
//! - [`tts_timeout`] (30 s, scaled with text length up to [`TTS_TIMEOUT_MAX`]) for synthesis,
//! - [`stt_timeout_for_wav_bytes`] (30 s, scaled with audio length up to [`STT_TIMEOUT_MAX`]) for
//!   transcription.
//!
//! Self-hosted speech servers on this PC or the LAN (Chatterbox, fish-speech, the neural sidecar
//! on CPU, ...) are often non-streaming and slow: they send nothing until the whole clip is done.
//! Synthesis requests should go through [`tts_post`], which gives such endpoints
//! ([`is_local_endpoint`]) the longer [`local_tts_timeout`] budget (60 s + 1 s per character, at
//! most [`LOCAL_TTS_TIMEOUT_MAX`] = 10 min) on [`local_http_client`], whose read timeout is that
//! long too. Cloud endpoints keep the tight [`tts_timeout`] on the shared client.
//!
//! # Errors
//! [`describe_request_error`] and [`describe_api_error`] build user-facing messages. They never
//! include query strings or URL credentials (API keys must travel in headers or bodies anyway), and
//! every body preview is truncated on a character boundary, so a Japanese or Chinese HTML error
//! page cannot panic the caller.

use reqwest::header::{HeaderMap, HeaderValue, ACCEPT, CONTENT_TYPE};
use reqwest::{Client, Response, StatusCode, Url};
use std::error::Error as StdError;
use std::sync::OnceLock;
use std::time::Duration;

/// Whole-request cap for large model downloads (and the shared client's default).
pub const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(7200);
/// Connect-phase cap for every request.
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(30);
/// Read (idle) timeout on the shared client: no bytes for this long means the peer is gone.
/// Also bounds how long a non-streaming server may think before sending response headers.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(120);

/// Availability / health probes.
pub const HEALTH_TIMEOUT: Duration = Duration::from_secs(10);
/// Voice, speaker, language and model lists.
pub const LIST_TIMEOUT: Duration = Duration::from_secs(20);
/// Machine translation / language detection (floor; see [`translate_timeout`]).
pub const TRANSLATE_TIMEOUT: Duration = Duration::from_secs(20);
/// Upper bound of [`translate_timeout`].
pub const TRANSLATE_TIMEOUT_MAX: Duration = Duration::from_secs(60);
/// Speech synthesis (floor; see [`tts_timeout`]).
pub const TTS_TIMEOUT: Duration = Duration::from_secs(30);
/// Upper bound of [`tts_timeout`]; equal to [`STALL_TIMEOUT`] so a silent non-streaming server
/// is never allowed longer than the client-wide read timeout anyway.
pub const TTS_TIMEOUT_MAX: Duration = Duration::from_secs(120);
/// Longest gap between two body chunks of a streamed audio response.
pub const TTS_STALL_TIMEOUT: Duration = Duration::from_secs(30);
/// Synthesis on a self-hosted server on this PC or the LAN (floor; see [`local_tts_timeout`]).
pub const LOCAL_TTS_TIMEOUT: Duration = Duration::from_secs(60);
/// Upper bound of [`local_tts_timeout`], and the read (idle) timeout of [`local_http_client`], so a
/// non-streaming local server may think this long before sending its response headers.
pub const LOCAL_TTS_TIMEOUT_MAX: Duration = Duration::from_secs(600);
/// Speech recognition (floor; see [`stt_timeout_for_wav_bytes`]).
pub const STT_TIMEOUT: Duration = Duration::from_secs(30);
/// Upper bound of [`stt_timeout_for_wav_bytes`].
pub const STT_TIMEOUT_MAX: Duration = Duration::from_secs(120);

/// Largest audio body a TTS call buffers (it still crosses IPC as a JSON array today).
pub const MAX_AUDIO_BYTES: usize = 64 * 1024 * 1024;

/// Extra synthesis time allowed per character of input text.
const TTS_MS_PER_CHAR: u64 = 100;
/// Extra synthesis time per character for a self-hosted server (CPU synthesis of fish-speech or
/// Chatterbox can run several times slower than real time).
const LOCAL_TTS_MS_PER_CHAR: u64 = 1000;
/// Extra translation time allowed per character of input text.
const TRANSLATE_MS_PER_CHAR: u64 = 10;
/// Bytes of an error body read for a message / preview.
const ERROR_BODY_LIMIT: usize = 4096;
/// Characters of an error body shown to the user.
const ERROR_PREVIEW_CHARS: usize = 180;
/// How long reading an error body may take once the status line is in.
const ERROR_BODY_TIMEOUT: Duration = Duration::from_secs(5);

fn default_headers() -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(
        ACCEPT,
        HeaderValue::from_static("application/octet-stream, */*"),
    );
    h
}

fn build_client(read_timeout: Duration) -> Client {
    let ua = format!(
        "Sigil/{} (desktop; https://github.com/Mewnah/Sigil)",
        env!("CARGO_PKG_VERSION")
    );
    Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .read_timeout(read_timeout)
        // Whole-request cap for large downloads; API calls override it per request.
        .timeout(DOWNLOAD_TIMEOUT)
        .user_agent(ua)
        .default_headers(default_headers())
        .build()
        .expect("reqwest Client builder")
}

/// Shared client for outbound HTTP from plugins (local TTS/STT, translate, model downloads).
///
/// Its whole-request default is [`DOWNLOAD_TIMEOUT`] so large model streams are not cut off, and its
/// read timeout ([`STALL_TIMEOUT`]) fails dead connections. API calls must set a per-request
/// `.timeout(...)` from the classes in this module, no longer than [`STALL_TIMEOUT`].
pub fn http_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| build_client(STALL_TIMEOUT))
}

/// Client for synthesis on self-hosted servers on this PC or the LAN (see [`tts_post`]). Same as
/// [`http_client`] except that its read timeout is [`LOCAL_TTS_TIMEOUT_MAX`], so a non-streaming
/// server may take up to its per-request budget before sending anything.
pub fn local_http_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| build_client(LOCAL_TTS_TIMEOUT_MAX))
}

/// GET with an explicit timeout suitable for multi‑GB model files (the stall timeout still applies).
pub fn http_download_get(url: &str) -> reqwest::RequestBuilder {
    http_client().get(url).timeout(DOWNLOAD_TIMEOUT)
}

// ---------------------------------------------------------------------------------------------
// Per-request timeout classes
// ---------------------------------------------------------------------------------------------

fn scaled(floor: Duration, extra_ms: u64, max: Duration) -> Duration {
    floor.saturating_add(Duration::from_millis(extra_ms)).min(max)
}

/// Timeout for synthesizing `text`: [`TTS_TIMEOUT`] plus 100 ms per character, capped at
/// [`TTS_TIMEOUT_MAX`]. A hung server fails within ~30–40 s for a sentence, while a slow CPU backend
/// still gets time for a paragraph.
pub fn tts_timeout(text: &str) -> Duration {
    let chars = text.chars().count() as u64;
    scaled(TTS_TIMEOUT, chars.saturating_mul(TTS_MS_PER_CHAR), TTS_TIMEOUT_MAX)
}

/// Timeout for synthesizing `text` on a self-hosted server on this PC or the LAN:
/// [`LOCAL_TTS_TIMEOUT`] plus 1 s per character, capped at [`LOCAL_TTS_TIMEOUT_MAX`]. A 150-character
/// caption gets 210 s, so slow CPU synthesis plays late instead of being dropped.
pub fn local_tts_timeout(text: &str) -> Duration {
    let chars = text.chars().count() as u64;
    scaled(
        LOCAL_TTS_TIMEOUT,
        chars.saturating_mul(LOCAL_TTS_MS_PER_CHAR),
        LOCAL_TTS_TIMEOUT_MAX,
    )
}

/// A POST for synthesizing `text` at `url`, with the matching client and timeout: a local or LAN
/// endpoint ([`is_local_endpoint`]) gets [`local_tts_timeout`] on [`local_http_client`], anything
/// else [`tts_timeout`] on [`http_client`]. Returns the timeout as well, for
/// [`describe_request_error`].
pub fn tts_post(url: Url, text: &str) -> (reqwest::RequestBuilder, Duration) {
    let (client, timeout) = if is_local_endpoint(&url) {
        (local_http_client(), local_tts_timeout(text))
    } else {
        (http_client(), tts_timeout(text))
    };
    (client.post(url).timeout(timeout), timeout)
}

/// Timeout for translating `text`: [`TRANSLATE_TIMEOUT`] plus 10 ms per character, capped at
/// [`TRANSLATE_TIMEOUT_MAX`]. Caption-length text gets the 20 s floor.
pub fn translate_timeout(text: &str) -> Duration {
    let chars = text.chars().count() as u64;
    scaled(
        TRANSLATE_TIMEOUT,
        chars.saturating_mul(TRANSLATE_MS_PER_CHAR),
        TRANSLATE_TIMEOUT_MAX,
    )
}

/// Timeout for transcribing `seconds` of audio: [`STT_TIMEOUT`] plus one real-time factor,
/// capped at [`STT_TIMEOUT_MAX`].
pub fn stt_timeout(seconds: f64) -> Duration {
    let extra_ms = if seconds.is_finite() && seconds > 0.0 {
        (seconds * 1000.0).min(u64::MAX as f64) as u64
    } else {
        0
    };
    scaled(STT_TIMEOUT, extra_ms, STT_TIMEOUT_MAX)
}

/// [`stt_timeout`] for a WAV payload of `wav_bytes` bytes, assuming 16 kHz mono 16-bit PCM
/// (32 000 bytes per second, what Sigil's capture path sends). Larger formats only get more time.
pub fn stt_timeout_for_wav_bytes(wav_bytes: usize) -> Duration {
    stt_timeout(wav_bytes as f64 / 32_000.0)
}

/// Whole seconds of `d`, rounded up, for messages.
pub fn secs(d: Duration) -> u64 {
    d.as_millis().div_ceil(1000) as u64
}

// ---------------------------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------------------------

/// Validate a user-entered base URL and return it without trailing slashes.
///
/// A missing scheme defaults to `http://` (`localhost:8880` → `http://localhost:8880`). Only
/// `http` and `https` are accepted, and a host is required. The path is kept, so reverse-proxy
/// prefixes like `https://host/tts` keep working.
pub fn normalize_base_url(raw: &str) -> Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("The endpoint URL is empty.".into());
    }
    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let parsed = Url::parse(&with_scheme)
        .map_err(|e| format!("Invalid endpoint URL '{}': {e}", redact_str(trimmed)))?;
    if !matches!(parsed.scheme(), "http" | "https") {
        return Err(format!(
            "Invalid endpoint URL '{}': only http:// and https:// are supported.",
            redact_str(trimmed)
        ));
    }
    if parsed.host_str().is_none_or(str::is_empty) {
        return Err(format!(
            "Invalid endpoint URL '{}': missing host.",
            redact_str(trimmed)
        ));
    }
    Ok(with_scheme.trim_end_matches('/').to_string())
}

/// `base` (see [`normalize_base_url`]) joined with an absolute API `path` such as `/v1/tts`.
pub fn api_url(base: &str, path: &str) -> Result<Url, String> {
    let base = normalize_base_url(base)?;
    let path = if path.starts_with('/') {
        path.to_string()
    } else {
        format!("/{path}")
    };
    Url::parse(&format!("{base}{path}"))
        .map_err(|e| format!("Invalid endpoint URL '{}': {e}", redact_str(&base)))
}

/// A per-call endpoint override when it is non-blank, else the stored endpoint.
pub fn pick_endpoint(call: Option<&str>, stored: &str) -> String {
    match call.map(str::trim) {
        Some(e) if !e.is_empty() => e.to_string(),
        _ => stored.to_string(),
    }
}

/// Whether `url` points at this PC or the local network, i.e. a self-hosted server rather than a
/// cloud API: loopback, private, link-local and CGNAT/Tailscale (100.64.0.0/10) IPv4, loopback,
/// unique-local and link-local IPv6, `localhost`, single-label host names (`gpu-box`) and the
/// `.local` / `.lan` / `.home.arpa` / `.internal` / `.localhost` suffixes. Only used to pick timeouts.
pub fn is_local_endpoint(url: &Url) -> bool {
    use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};

    fn v4(ip: Ipv4Addr) -> bool {
        let [a, b, ..] = ip.octets();
        ip.is_loopback()
            || ip.is_private()
            || ip.is_link_local()
            || ip.is_unspecified()
            || (a == 100 && (64..128).contains(&b))
    }
    fn v6(ip: Ipv6Addr) -> bool {
        let first = ip.segments()[0];
        ip.is_loopback()
            || ip.is_unspecified()
            || (first & 0xfe00) == 0xfc00 // unique local fc00::/7
            || (first & 0xffc0) == 0xfe80 // link local fe80::/10
            || ip.to_ipv4_mapped().is_some_and(v4)
    }
    let Some(host) = url.host_str() else {
        return false;
    };
    // IPv6 hosts come bracketed (`[::1]`).
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    match bare.parse::<IpAddr>() {
        Ok(IpAddr::V4(ip)) => v4(ip),
        Ok(IpAddr::V6(ip)) => v6(ip),
        Err(_) => {
            let d = bare.trim_end_matches('.').to_ascii_lowercase();
            !d.is_empty()
                && (!d.contains('.')
                    || [".localhost", ".local", ".lan", ".home.arpa", ".internal"]
                        .iter()
                        .any(|s| d.ends_with(s)))
        }
    }
}

/// `url` without query, fragment or credentials, for messages and logs.
pub fn redact_url(url: &Url) -> String {
    let mut u = url.clone();
    u.set_query(None);
    u.set_fragment(None);
    let _ = u.set_username("");
    let _ = u.set_password(None);
    u.to_string()
}

/// [`redact_url`] for a string that may not parse; falls back to cutting at `?`/`#` and `@`.
fn redact_str(raw: &str) -> String {
    if let Ok(u) = Url::parse(raw) {
        return redact_url(&u);
    }
    let cut = raw.find(['?', '#']).map_or(raw, |i| &raw[..i]);
    match (cut.find("://"), cut.rfind('@')) {
        (Some(s), Some(at)) if at > s => format!("{}{}", &cut[..s + 3], &cut[at + 1..]),
        _ => cut.to_string(),
    }
}

// ---------------------------------------------------------------------------------------------
// Error descriptions
// ---------------------------------------------------------------------------------------------

/// The innermost cause of `err` (e.g. "No connection could be made ... (os error 10061)"), which is
/// more useful than reqwest's top-level "error sending request for url (...)".
fn root_cause(err: &(dyn StdError + 'static)) -> String {
    let mut cur: &(dyn StdError + 'static) = err;
    while let Some(next) = cur.source() {
        cur = next;
    }
    cur.to_string()
}

/// User-facing text for a request that failed before a response arrived.
///
/// `what` names the operation ("Kokoro speech request"); `timeout` is the per-request timeout the
/// call used, shown when the failure is a timeout.
///
/// A client's read (idle) timeout reports the same timeout error as the per-request one. Every API
/// class is at most its client's read timeout ([`STALL_TIMEOUT`] on [`http_client`],
/// [`LOCAL_TTS_TIMEOUT_MAX`] on [`local_http_client`]), so there the per-request timeout is the one
/// that fired. Longer timeouts are downloads ([`DOWNLOAD_TIMEOUT`] on [`http_client`]), where a
/// stall usually trips [`STALL_TIMEOUT`] first, so the message names both limits.
pub fn describe_request_error(what: &str, err: &reqwest::Error, timeout: Duration) -> String {
    let target = err
        .url()
        .map(|u| format!(" ({})", redact_url(u)))
        .unwrap_or_default();
    if err.is_timeout() {
        return if timeout > LOCAL_TTS_TIMEOUT_MAX {
            format!(
                "{what}{target}: timed out (no data for {} s, or not finished within {} s).",
                secs(STALL_TIMEOUT),
                secs(timeout)
            )
        } else {
            format!(
                "{what}{target}: no response within {} s (timed out).",
                secs(timeout)
            )
        };
    }
    let cause = root_cause(err);
    if err.is_connect() {
        format!("{what}{target}: cannot connect. Is the server running? ({cause})")
    } else {
        format!("{what}{target} failed: {cause}")
    }
}

/// Truncate `s` to at most `max_chars` characters (never inside a UTF-8 sequence), appending `…`
/// when something was cut.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    match s.char_indices().nth(max_chars) {
        Some((byte_idx, _)) => {
            let mut out = s[..byte_idx].trim_end().to_string();
            out.push('…');
            out
        }
        None => s.to_string(),
    }
}

/// A one-line, UTF-8-safe preview of an HTTP body: lossy-decoded, control characters and runs of
/// whitespace collapsed to single spaces, at most `max_chars` characters.
pub fn body_preview(bytes: &[u8], max_chars: usize) -> String {
    let text = String::from_utf8_lossy(bytes);
    let mut collapsed = String::with_capacity(text.len().min(max_chars * 4 + 4));
    let mut pending_space = false;
    for c in text.chars() {
        if c.is_whitespace() || c.is_control() {
            pending_space = !collapsed.is_empty();
            continue;
        }
        if pending_space {
            collapsed.push(' ');
            pending_space = false;
        }
        collapsed.push(c);
    }
    truncate_chars(&collapsed, max_chars)
}

fn json_message(v: &serde_json::Value) -> Option<String> {
    use serde_json::Value;
    let non_empty = |s: &str| {
        let t = s.trim();
        (!t.is_empty()).then(|| t.to_string())
    };
    match v {
        Value::String(s) => non_empty(s),
        Value::Object(map) => {
            for key in ["error", "detail", "message", "msg", "error_description", "reason"] {
                if let Some(inner) = map.get(key) {
                    if let Some(m) = json_message(inner) {
                        return Some(m);
                    }
                }
            }
            None
        }
        // FastAPI validation errors: {"detail": [{"loc": [...], "msg": "...", ...}]}
        Value::Array(items) => {
            let msgs: Vec<String> = items.iter().filter_map(json_message).take(3).collect();
            (!msgs.is_empty()).then(|| msgs.join("; "))
        }
        _ => None,
    }
}

/// `<title>` of an HTML document, if present.
fn html_title(text: &str) -> Option<String> {
    let lower = text.to_ascii_lowercase();
    let open = lower.find("<title")?;
    let start = open + lower[open..].find('>')? + 1;
    let end = start + lower[start..].find("</title")?;
    // `to_ascii_lowercase` keeps byte offsets, so these indices are valid in `text` too.
    let title = text.get(start..end)?.trim();
    (!title.is_empty()).then(|| title.to_string())
}

/// The most useful human-readable message in an error body: a JSON `error` / `detail` / `message`
/// field, an HTML `<title>`, or a whitespace-collapsed preview. Always UTF-8 safe and bounded.
pub fn error_body_summary(bytes: &[u8]) -> Option<String> {
    let bytes = &bytes[..bytes.len().min(ERROR_BODY_LIMIT)];
    if bytes.iter().all(u8::is_ascii_whitespace) {
        return None;
    }
    if let Ok(v) = serde_json::from_slice::<serde_json::Value>(bytes) {
        if let Some(m) = json_message(&v) {
            return Some(body_preview(m.as_bytes(), ERROR_PREVIEW_CHARS));
        }
    }
    let text = String::from_utf8_lossy(bytes);
    if let Some(title) = html_title(&text) {
        return Some(body_preview(title.as_bytes(), ERROR_PREVIEW_CHARS));
    }
    let preview = body_preview(bytes, ERROR_PREVIEW_CHARS);
    (!preview.is_empty()).then_some(preview)
}

/// Read at most `limit` bytes of `resp`'s body, giving up after [`ERROR_BODY_TIMEOUT`]. Used for
/// error bodies, which are never needed in full.
async fn read_prefix(mut resp: Response, limit: usize) -> Result<Vec<u8>, String> {
    let mut buf = Vec::new();
    let read = async {
        while buf.len() < limit {
            match resp.chunk().await {
                Ok(Some(chunk)) => {
                    let take = chunk.len().min(limit - buf.len());
                    buf.extend_from_slice(&chunk[..take]);
                }
                Ok(None) => break,
                Err(e) => return Err(root_cause(&e)),
            }
        }
        Ok(())
    };
    let outcome = tokio::time::timeout(ERROR_BODY_TIMEOUT, read).await;
    match outcome {
        // A slow or failing error body still yields whatever arrived.
        Ok(Ok(())) | Err(_) => Ok(buf),
        Ok(Err(e)) if buf.is_empty() => Err(e),
        Ok(Err(_)) => Ok(buf),
    }
}

/// User-facing text for a non-success API response, e.g.
/// `Kokoro speech request: HTTP 403 Forbidden: Invalid API key`.
pub async fn describe_api_error(what: &str, resp: Response) -> String {
    let status = resp.status();
    match read_prefix(resp, ERROR_BODY_LIMIT).await {
        Ok(bytes) => match error_body_summary(&bytes) {
            Some(summary) => format!("{what}: HTTP {status}: {summary}"),
            None => format!("{what}: HTTP {status}"),
        },
        Err(e) => format!("{what}: HTTP {status} (failed to read body: {e})"),
    }
}

/// For failed downloads: status, Content-Type, and a short body preview (helps diagnose HTML / WAF
/// pages). UTF-8 safe: multibyte error pages (Japanese, Chinese, ...) are cut on a character
/// boundary, and at most a few KB of the body are read.
pub async fn describe_failed_response(resp: Response) -> String {
    let status = resp.status();
    let ct = resp
        .headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = match read_prefix(resp, ERROR_BODY_LIMIT).await {
        Ok(b) => b,
        Err(e) => {
            return format!("HTTP {status} (failed to read body: {e})");
        }
    };
    let preview = body_preview(&bytes, ERROR_PREVIEW_CHARS);
    format!("HTTP {status} content-type={ct} body_preview={preview:?}")
}

// ---------------------------------------------------------------------------------------------
// Bodies
// ---------------------------------------------------------------------------------------------

/// Read a response body chunk by chunk ("streamed"), failing when no chunk arrives for `stall`
/// or when the body exceeds `max_bytes`. The request's own `.timeout(...)` still bounds the total.
pub async fn read_body_capped(
    what: &str,
    mut resp: Response,
    max_bytes: usize,
    stall: Duration,
) -> Result<Vec<u8>, String> {
    let too_large = || {
        format!(
            "{what}: response is larger than {} MB.",
            max_bytes.div_ceil(1024 * 1024)
        )
    };
    let declared = resp.content_length();
    if declared.is_some_and(|n| n > max_bytes as u64) {
        return Err(too_large());
    }
    let mut buf = Vec::with_capacity(declared.map_or(0, |n| n as usize));
    loop {
        match tokio::time::timeout(stall, resp.chunk()).await {
            Err(_) => {
                return Err(format!(
                    "{what}: the server stopped sending data for {} s (stalled).",
                    secs(stall)
                ))
            }
            Ok(Err(e)) => {
                return Err(if e.is_timeout() {
                    format!("{what}: timed out while receiving the response.")
                } else {
                    format!("{what}: failed while receiving the response: {}", root_cause(&e))
                })
            }
            Ok(Ok(None)) => return Ok(buf),
            Ok(Ok(Some(chunk))) => {
                if buf.len().saturating_add(chunk.len()) > max_bytes {
                    return Err(too_large());
                }
                buf.extend_from_slice(&chunk);
            }
        }
    }
}

/// Reject a 2xx body that is clearly not audio (a JSON or text/HTML error page served with 200),
/// so the caller reports the server's message instead of a silent decode failure.
pub fn ensure_audio_body(what: &str, content_type: Option<&str>, body: &[u8]) -> Result<(), String> {
    if body.is_empty() {
        return Err(format!("{what}: the server returned no audio."));
    }
    let ct = content_type.unwrap_or("").trim().to_ascii_lowercase();
    let looks_textual = ct.starts_with("application/json")
        || ct.starts_with("text/")
        || (ct.is_empty() && matches!(body.iter().find(|b| !b.is_ascii_whitespace()), Some(b'{' | b'[' | b'<')));
    if looks_textual {
        let summary = error_body_summary(body).unwrap_or_default();
        return Err(format!(
            "{what}: expected audio but got {}{}{}",
            if ct.is_empty() { "a text response" } else { ct.as_str() },
            if summary.is_empty() { "" } else { ": " },
            summary
        ));
    }
    Ok(())
}

/// The response's `Content-Type`, if it is valid text.
pub fn content_type(resp: &Response) -> Option<String> {
    resp.headers()
        .get(CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .map(str::to_string)
}

/// Whether `status` means the credentials were rejected.
pub fn is_auth_failure(status: StatusCode) -> bool {
    matches!(status, StatusCode::UNAUTHORIZED | StatusCode::FORBIDDEN)
}

/// A trimmed, non-empty API key, or `None`.
pub fn non_empty_key(key: Option<&str>) -> Option<String> {
    key.map(str::trim).filter(|k| !k.is_empty()).map(str::to_string)
}

/// Probe `req` (a GET to a health or list endpoint) with [`HEALTH_TIMEOUT`]: `Ok(status)` when the
/// server answered, `Err(reason)` when it could not be reached. The reason is also logged, because
/// availability commands only return a boolean to the UI.
pub async fn probe(what: &str, req: reqwest::RequestBuilder) -> Result<StatusCode, String> {
    match req.timeout(HEALTH_TIMEOUT).send().await {
        Ok(resp) => Ok(resp.status()),
        Err(e) => {
            let reason = describe_request_error(what, &e, HEALTH_TIMEOUT);
            tracing::info!(target: "sigil::http", "{reason}");
            Err(reason)
        }
    }
}

/// [`probe`] reduced to "answered with 2xx".
pub async fn probe_ok(what: &str, req: reqwest::RequestBuilder) -> bool {
    match probe(what, req).await {
        Ok(status) if status.is_success() => true,
        Ok(status) => {
            tracing::info!(target: "sigil::http", "{what}: HTTP {status}");
            false
        }
        Err(_) => false,
    }
}

// ---------------------------------------------------------------------------------------------
// Shared response-shape helpers
// ---------------------------------------------------------------------------------------------

/// One entry of a voice / speaker list.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NamedEntry {
    /// The value to send back to the server (voice id / name).
    pub id: String,
    /// Display name (falls back to `id`).
    pub name: String,
    /// Language tag when the server reports one.
    pub language: Option<String>,
}

fn scalar_string(v: &serde_json::Value) -> Option<String> {
    match v {
        serde_json::Value::String(s) if !s.trim().is_empty() => Some(s.trim().to_string()),
        serde_json::Value::Number(n) => Some(n.to_string()),
        _ => None,
    }
}

fn first_field(obj: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<String> {
    keys.iter().find_map(|k| obj.get(*k).and_then(scalar_string))
}

/// Parse a voice / speaker list from the common response shapes: a bare array, or an object whose
/// `list_keys` field (tried in order) holds the array. Items may be strings (`"af_bella"`) or
/// objects; for objects the id comes from the first present of `id_keys` and the display name from
/// `name_keys` (falling back to the id). Duplicate ids are dropped, order is kept.
pub fn parse_named_list(
    v: &serde_json::Value,
    list_keys: &[&str],
    id_keys: &[&str],
    name_keys: &[&str],
) -> Vec<NamedEntry> {
    let items: &[serde_json::Value] = match v {
        serde_json::Value::Array(items) => items,
        serde_json::Value::Object(map) => match list_keys
            .iter()
            .find_map(|k| map.get(*k).and_then(|x| x.as_array()))
        {
            Some(items) => items,
            None => return Vec::new(),
        },
        _ => return Vec::new(),
    };
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::with_capacity(items.len());
    for item in items {
        let entry = match item {
            serde_json::Value::Object(obj) => {
                let Some(id) = first_field(obj, id_keys) else {
                    continue;
                };
                let name = first_field(obj, name_keys).unwrap_or_else(|| id.clone());
                let language = first_field(obj, &["language", "lang", "locale"]).or_else(|| {
                    obj.get("languages")
                        .and_then(|l| l.as_array())
                        .and_then(|l| l.iter().find_map(scalar_string))
                });
                NamedEntry { id, name, language }
            }
            other => match scalar_string(other) {
                Some(id) => NamedEntry {
                    name: id.clone(),
                    id,
                    language: None,
                },
                None => continue,
            },
        };
        if seen.insert(entry.id.clone()) {
            out.push(entry);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// Loopback-only client (no system proxy) for the mini-server tests.
    fn test_client() -> Client {
        Client::builder().no_proxy().build().unwrap()
    }

    /// Serve one connection on 127.0.0.1: read the request head, then write `parts` with `gap`
    /// between them. Returns the base URL.
    async fn serve_once(parts: Vec<Vec<u8>>, gap: Duration) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                if sock.read(&mut byte).await.unwrap_or(0) == 0 {
                    return;
                }
                head.push(byte[0]);
            }
            for (i, part) in parts.iter().enumerate() {
                if i > 0 {
                    tokio::time::sleep(gap).await;
                }
                if sock.write_all(part).await.is_err() {
                    return;
                }
            }
            // Keep the socket open (a "hung" server) until the client gives up.
            tokio::time::sleep(Duration::from_secs(5)).await;
        });
        format!("http://{addr}")
    }

    #[test]
    fn truncate_chars_never_splits_multibyte() {
        let ja = "サーバーエラーが発生しました。しばらくしてから再度お試しください。";
        for max in 0..ja.chars().count() + 2 {
            let t = truncate_chars(ja, max);
            assert!(t.chars().count() <= max + 1, "max {max}: {t}");
        }
        assert_eq!(truncate_chars("abc", 3), "abc");
        assert_eq!(truncate_chars("abcd", 3), "abc…");
    }

    #[test]
    fn body_preview_is_utf8_safe_at_every_cut() {
        // Japanese HTML error page: 3-byte characters, so byte 180 / 384 cuts would split them.
        let page = format!(
            "<!DOCTYPE html>\n<html lang=\"ja\"><head><title>503 サービス利用不可</title></head>\n<body>{}</body></html>",
            "現在メンテナンス中です。".repeat(60)
        );
        let bytes = page.as_bytes();
        for cut in 0..bytes.len().min(1200) {
            let p = body_preview(&bytes[..cut], ERROR_PREVIEW_CHARS);
            assert!(p.chars().count() <= ERROR_PREVIEW_CHARS + 1);
            assert!(!p.contains('\n'));
        }
    }

    #[test]
    fn body_preview_collapses_whitespace_and_controls() {
        assert_eq!(body_preview(b"  a\r\n\tb \x00 c  ", 50), "a b c");
        assert_eq!(body_preview(b"", 10), "");
    }

    #[test]
    fn error_summary_prefers_json_message_then_html_title() {
        assert_eq!(
            error_body_summary(br#"{"error":"Invalid API key"}"#).as_deref(),
            Some("Invalid API key")
        );
        assert_eq!(
            error_body_summary(br#"{"error":{"message":"quota exceeded","code":429}}"#).as_deref(),
            Some("quota exceeded")
        );
        assert_eq!(
            error_body_summary(br#"{"detail":[{"loc":["body","text"],"msg":"field required"}]}"#)
                .as_deref(),
            Some("field required")
        );
        assert_eq!(
            error_body_summary("<html><head><title>502 不正なゲートウェイ</title></head></html>".as_bytes())
                .as_deref(),
            Some("502 不正なゲートウェイ")
        );
        assert_eq!(error_body_summary(b"   ").as_deref(), None);
        assert_eq!(error_body_summary(b"plain text").as_deref(), Some("plain text"));
    }

    #[test]
    fn tts_timeout_scales_and_caps() {
        assert_eq!(tts_timeout(""), TTS_TIMEOUT);
        assert_eq!(tts_timeout(&"a".repeat(100)), Duration::from_secs(40));
        // Characters, not bytes: 100 kana = 300 bytes but 10 s extra.
        assert_eq!(tts_timeout(&"あ".repeat(100)), Duration::from_secs(40));
        assert_eq!(tts_timeout(&"a".repeat(100_000)), TTS_TIMEOUT_MAX);
        assert!(TTS_TIMEOUT_MAX <= STALL_TIMEOUT);
    }

    #[test]
    fn translate_and_stt_timeouts_are_bounded() {
        assert_eq!(translate_timeout("hello"), Duration::from_millis(20_050));
        assert_eq!(translate_timeout(&"x".repeat(1_000_000)), TRANSLATE_TIMEOUT_MAX);
        assert_eq!(stt_timeout(0.0), STT_TIMEOUT);
        assert_eq!(stt_timeout(f64::NAN), STT_TIMEOUT);
        assert_eq!(stt_timeout(-5.0), STT_TIMEOUT);
        assert_eq!(stt_timeout(10.0), Duration::from_secs(40));
        assert_eq!(stt_timeout(1e12), STT_TIMEOUT_MAX);
        assert_eq!(stt_timeout_for_wav_bytes(32_000 * 20), Duration::from_secs(50));
        for d in [HEALTH_TIMEOUT, LIST_TIMEOUT, TRANSLATE_TIMEOUT_MAX, TTS_TIMEOUT_MAX, STT_TIMEOUT_MAX] {
            assert!(d <= Duration::from_secs(120), "no API timeout class may exceed 2 min");
            assert!(d <= STALL_TIMEOUT, "shared-client API classes must fit its read timeout");
        }
    }

    #[test]
    fn local_tts_timeout_is_generous_but_bounded() {
        assert_eq!(local_tts_timeout(""), LOCAL_TTS_TIMEOUT);
        // The review scenario: a 150-character caption that takes ~60 s on CPU must not be dropped.
        assert_eq!(local_tts_timeout(&"a".repeat(150)), Duration::from_secs(210));
        assert!(local_tts_timeout(&"a".repeat(150)) > tts_timeout(&"a".repeat(150)));
        assert_eq!(local_tts_timeout(&"あ".repeat(100)), Duration::from_secs(160));
        assert_eq!(local_tts_timeout(&"a".repeat(1_000_000)), LOCAL_TTS_TIMEOUT_MAX);
        // Still nowhere near the old 2 h hang, and the per-request budget fires before (or with)
        // the local client's read timeout.
        assert!(LOCAL_TTS_TIMEOUT_MAX <= Duration::from_secs(600));
        assert!(LOCAL_TTS_TIMEOUT_MAX < DOWNLOAD_TIMEOUT);
    }

    fn url(s: &str) -> Url {
        Url::parse(s).unwrap()
    }

    #[test]
    fn local_endpoints_are_detected() {
        for local in [
            "http://localhost:8080",
            "http://LOCALHOST.:5555/v1",
            "http://127.0.0.1:8766",
            "http://127.1:8080",
            "http://0.0.0.0:8080",
            "http://192.168.1.20:8080",
            "http://10.0.0.5:8880",
            "http://172.20.3.4:8888",
            "http://169.254.10.1",
            "http://100.101.102.103:8080", // Tailscale
            "http://[::1]:8080",
            "http://[fd12:3456::1]:8080",
            "http://[fe80::1]:8080",
            "http://[::ffff:192.168.0.2]:8080",
            "http://gpu-box:8080",
            "http://tts.local:8080",
            "http://nas.lan",
            "http://server.home.arpa",
            "http://tts.internal",
            "http://kokoro.localhost",
        ] {
            assert!(is_local_endpoint(&url(local)), "{local} should be local");
        }
        for cloud in [
            "https://api.fish.audio",
            "https://api.uberduck.ai/v1",
            "https://8.8.8.8",
            "http://100.128.0.1",
            "http://172.32.0.1",
            "http://[2001:4860:4860::8888]",
            "https://my.tts.example.com",
            "https://local.example.com",
        ] {
            assert!(!is_local_endpoint(&url(cloud)), "{cloud} should be remote");
        }
    }

    #[test]
    fn tts_post_picks_budget_by_endpoint() {
        let text = "a".repeat(150);
        assert_eq!(tts_post(url("http://localhost:8080/v1/tts"), &text).1, Duration::from_secs(210));

        let (req, timeout) = tts_post(url("http://127.0.0.1:5555/v1/audio/speech"), &text);
        assert_eq!(timeout, Duration::from_secs(210));
        let req = req.build().unwrap();
        assert_eq!(req.timeout(), Some(&timeout));
        assert_eq!(req.method(), reqwest::Method::POST);

        let (req, timeout) = tts_post(url("https://api.fish.audio/v1/tts"), &text);
        assert_eq!(timeout, Duration::from_secs(45));
        assert_eq!(req.build().unwrap().timeout(), Some(&timeout));
    }

    #[test]
    fn normalize_base_url_accepts_and_rejects() {
        assert_eq!(normalize_base_url(" http://localhost:8880/ ").unwrap(), "http://localhost:8880");
        assert_eq!(normalize_base_url("localhost:5000").unwrap(), "http://localhost:5000");
        assert_eq!(normalize_base_url("https://api.fish.audio").unwrap(), "https://api.fish.audio");
        assert_eq!(normalize_base_url("https://host/tts//").unwrap(), "https://host/tts");
        assert!(normalize_base_url("").is_err());
        assert!(normalize_base_url("   ").is_err());
        assert!(normalize_base_url("ftp://host").is_err());
        assert!(normalize_base_url("file:///C:/x").is_err());
    }

    #[test]
    fn api_url_keeps_proxy_prefix() {
        assert_eq!(
            api_url("https://host/libre/", "/translate").unwrap().as_str(),
            "https://host/libre/translate"
        );
        assert_eq!(api_url("127.0.0.1:8090", "health").unwrap().as_str(), "http://127.0.0.1:8090/health");
    }

    #[test]
    fn pick_endpoint_prefers_non_blank_call_value() {
        assert_eq!(pick_endpoint(Some(" http://a "), "http://b"), "http://a");
        assert_eq!(pick_endpoint(Some("  "), "http://b"), "http://b");
        assert_eq!(pick_endpoint(None, "http://b"), "http://b");
    }

    #[test]
    fn redaction_strips_keys_and_credentials() {
        let u = Url::parse("https://user:pw@host/languages?api_key=SECRET#x").unwrap();
        let r = redact_url(&u);
        assert!(!r.contains("SECRET") && !r.contains("pw") && !r.contains("user"), "{r}");
        assert_eq!(redact_str("not a url?api_key=SECRET"), "not a url");
        assert_eq!(redact_str("http://u:p@host:1/x?k=1"), "http://host:1/x");
    }

    #[test]
    fn ensure_audio_body_rejects_text_responses() {
        assert!(ensure_audio_body("t", Some("audio/wav"), b"RIFF....").is_ok());
        assert!(ensure_audio_body("t", None, b"ID3\x03").is_ok());
        let e = ensure_audio_body("t", Some("application/json"), br#"{"detail":"bad voice"}"#).unwrap_err();
        assert!(e.contains("bad voice"), "{e}");
        assert!(ensure_audio_body("t", None, b"<html><title>oops</title></html>").is_err());
        assert!(ensure_audio_body("t", Some("audio/mpeg"), b"").is_err());
    }

    #[test]
    fn parse_named_list_accepts_all_shapes() {
        let strings = serde_json::json!({"voices": ["af_bella", "af_sky", "af_bella"]});
        let objs = serde_json::json!({"voices": [{"name": "af_bella"}, {"id": 7, "name": "Seven"}]});
        let bare = serde_json::json!([{"voice_id": "v1", "display_name": "Voice 1", "languages": ["ja"]}]);
        let ids = ["voice_id", "id", "name"];
        let names = ["display_name", "name"];
        let a = parse_named_list(&strings, &["voices"], &ids, &names);
        assert_eq!(a.iter().map(|e| e.id.as_str()).collect::<Vec<_>>(), ["af_bella", "af_sky"]);
        let b = parse_named_list(&objs, &["voices"], &ids, &names);
        assert_eq!(b[0], NamedEntry { id: "af_bella".into(), name: "af_bella".into(), language: None });
        assert_eq!(b[1].id, "7");
        assert_eq!(b[1].name, "Seven");
        let c = parse_named_list(&bare, &["voices"], &ids, &names);
        assert_eq!(c[0].language.as_deref(), Some("ja"));
        assert!(parse_named_list(&serde_json::json!({"other": []}), &["voices"], &ids, &names).is_empty());
        assert!(parse_named_list(&serde_json::json!(42), &["voices"], &ids, &names).is_empty());
    }

    #[tokio::test]
    async fn describe_failed_response_handles_japanese_html() {
        let body = format!(
            "<html><head><title>エラー</title></head><body>{}</body></html>",
            "内部サーバーエラーが発生しました。".repeat(40)
        );
        let resp = format!(
            "HTTP/1.1 503 Service Unavailable\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let base = serve_once(vec![resp.into_bytes()], Duration::ZERO).await;
        let r = test_client().get(&base).timeout(Duration::from_secs(5)).send().await.unwrap();
        let msg = describe_failed_response(r).await;
        assert!(msg.starts_with("HTTP 503"), "{msg}");
        assert!(msg.contains("text/html"), "{msg}");
    }

    #[tokio::test]
    async fn describe_api_error_extracts_json_message() {
        let body = r#"{"error":"Visit https://portal.libretranslate.com to get an API key"}"#;
        let resp = format!(
            "HTTP/1.1 403 Forbidden\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        );
        let base = serve_once(vec![resp.into_bytes()], Duration::ZERO).await;
        let r = test_client().get(&base).timeout(Duration::from_secs(5)).send().await.unwrap();
        let msg = describe_api_error("Translation", r).await;
        assert_eq!(msg, "Translation: HTTP 403 Forbidden: Visit https://portal.libretranslate.com to get an API key");
    }

    #[tokio::test]
    async fn hung_server_times_out_per_request() {
        // Accepts the connection and never answers: the per-request timeout must fire.
        let base = serve_once(vec![], Duration::ZERO).await;
        let timeout = Duration::from_millis(300);
        let started = std::time::Instant::now();
        let err = test_client().get(&base).timeout(timeout).send().await.unwrap_err();
        assert!(started.elapsed() < Duration::from_secs(3));
        let msg = describe_request_error("Kokoro speech request", &err, timeout);
        assert!(msg.contains("timed out"), "{msg}");
        assert!(msg.contains("no response within 1 s"), "{msg}");

        // A local synthesis budget (<= the local client's read timeout) is reported as such.
        let msg = describe_request_error("Fish Speech synthesis", &err, Duration::from_secs(210));
        assert!(msg.contains("no response within 210 s"), "{msg}");
    }

    #[tokio::test]
    async fn stalled_download_names_the_stall_timeout() {
        // A download's timeout error may come from the 120 s read timeout, long before the 2 h
        // whole-request cap: the message must not claim "no response within 7200 s".
        let base = serve_once(vec![], Duration::ZERO).await;
        let err = test_client()
            .get(&base)
            .timeout(Duration::from_millis(200))
            .send()
            .await
            .unwrap_err();
        assert!(err.is_timeout());
        let msg = describe_request_error("NVIDIA pack download", &err, DOWNLOAD_TIMEOUT);
        assert!(msg.contains("no data for 120 s"), "{msg}");
        assert!(msg.contains("not finished within 7200 s"), "{msg}");
        assert!(!msg.contains("no response within 7200 s"), "{msg}");
    }

    #[tokio::test]
    async fn read_body_capped_detects_stalls_and_oversize() {
        // Chunked body: one chunk, then silence.
        let head = b"HTTP/1.1 200 OK\r\nContent-Type: audio/wav\r\nTransfer-Encoding: chunked\r\n\r\n4\r\nRIFF\r\n".to_vec();
        let base = serve_once(vec![head], Duration::ZERO).await;
        let r = test_client().get(&base).timeout(Duration::from_secs(5)).send().await.unwrap();
        let err = read_body_capped("Fish speech", r, 1024, Duration::from_millis(300)).await.unwrap_err();
        assert!(err.contains("stalled"), "{err}");

        // Two chunks with a gap shorter than the stall timeout, then the terminator.
        let head = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n".to_vec();
        let tail = b"3\r\ndef\r\n0\r\n\r\n".to_vec();
        let base = serve_once(vec![head, tail], Duration::from_millis(100)).await;
        let r = test_client().get(&base).timeout(Duration::from_secs(5)).send().await.unwrap();
        let body = read_body_capped("x", r, 1024, Duration::from_secs(2)).await.unwrap();
        assert_eq!(body, b"abcdef");

        // Declared length over the cap is refused before reading.
        let resp = b"HTTP/1.1 200 OK\r\nContent-Length: 4096\r\n\r\n".to_vec();
        let base = serve_once(vec![resp], Duration::ZERO).await;
        let r = test_client().get(&base).timeout(Duration::from_secs(5)).send().await.unwrap();
        let err = read_body_capped("x", r, 1024, Duration::from_secs(2)).await.unwrap_err();
        assert!(err.contains("larger than"), "{err}");
    }
}
