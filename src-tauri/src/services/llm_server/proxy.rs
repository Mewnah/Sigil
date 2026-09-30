//! `llm_proxy_chat` / `llm_proxy_cancel`: one HTTP request performed for the webview, its response
//! streamed back through a Tauri `Channel`. The webview's CSP blocks plain-http LAN hosts (an
//! Ollama or llama-server on another PC) and some servers reject the webview's origin (CORS), so the
//! LLM client (`src/core/services/llm`, transport `rust`) sends those requests here.
//!
//! Rules (frozen by P4-S0): `http`/`https` URLs only (no credentials in the URL), `GET` or `POST`,
//! no redirects (a 3xx is returned as is), headers passed through and never logged, a 30 s stall
//! timeout plus `timeoutMs` overall when set, response bodies capped at 8 MB. Events: `response`
//! (status, lower-case headers), then `chunk`s of UTF-8 text (a multi-byte character split between
//! network reads is never split between chunks), then `done`; or one `error` (`cancelled` after
//! `llm_proxy_cancel`). The command resolves after the last event; it rejects (`invalid_input`) only
//! when the request is refused before anything is sent.
//!
//! Timeouts (P4-B): the stall timeout applies once the response headers have arrived (no body
//! bytes for 30 s fails the request). Waiting for the headers is bounded by `timeoutMs`, else by
//! [`HEADERS_TIMEOUT`]: a non-streaming request sends nothing until generation ends, and a cold
//! server (an Ollama that unloaded its model) sends nothing while it loads. A cancel that arrives
//! before its request is registered (the two commands race) is remembered for
//! [`CANCEL_TOMBSTONE_TTL`], and that request then ends with `cancelled` without being sent.

use super::{LlmProxyEvent, LlmProxyRequest, LlmServerError, LlmServerErrorKind};
use crate::services::http_client::{describe_request_error, is_local_endpoint};
use reqwest::header::{HeaderMap, HeaderName, HeaderValue, CONTENT_TYPE};
use reqwest::{Client, Method, Url};
use std::collections::{BTreeMap, HashMap};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};
use tokio::sync::watch;

/// No body bytes for this long (once the headers arrived) fails the request.
pub const STALL_TIMEOUT: Duration = Duration::from_secs(30);
/// Longest wait for the response headers when the caller set no `timeoutMs` (a model loading on a
/// cold server, a long non-streaming generation on a CPU).
pub const HEADERS_TIMEOUT: Duration = Duration::from_secs(5 * 60);
/// How long a cancel for a request that is not running yet is remembered.
pub const CANCEL_TOMBSTONE_TTL: Duration = Duration::from_secs(60);
/// Most remembered early cancels (ids are unique per request; old ones are dropped first).
const MAX_TOMBSTONES: usize = 256;
pub const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
/// Response bodies larger than this fail (a chat reply is a few KB).
pub const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;
/// Largest request body accepted from the webview.
pub const MAX_REQUEST_BYTES: usize = 4 * 1024 * 1024;
/// Longest overall deadline.
pub const MAX_TIMEOUT_MS: u64 = 10 * 60 * 1000;
const MAX_URL_LEN: usize = 8 * 1024;
const MAX_REQUEST_ID_LEN: usize = 128;
const MAX_HEADERS: usize = 64;
const MAX_HEADER_VALUE_LEN: usize = 16 * 1024;
/// Response headers forwarded to the webview.
const MAX_RESPONSE_HEADERS: usize = 64;
const MAX_RESPONSE_HEADER_LEN: usize = 4096;
/// Message of the `error` event after `llm_proxy_cancel`.
pub const CANCELLED: &str = "cancelled";

/// Request headers the proxy sets itself (or that must not be forwarded).
const FORBIDDEN_HEADERS: &[&str] = &[
    "host",
    "content-length",
    "transfer-encoding",
    "connection",
    "keep-alive",
    "upgrade",
    "te",
    "trailer",
    "proxy-authorization",
    "proxy-connection",
];

fn invalid(message: impl Into<String>) -> LlmServerError {
    LlmServerError::new(LlmServerErrorKind::InvalidInput, message)
}

/// A request that passed validation.
#[derive(Debug)]
pub struct Checked {
    pub id: String,
    pub url: Url,
    pub method: Method,
    pub headers: HeaderMap,
    pub body: Option<String>,
    pub stream: bool,
    pub timeout: Option<Duration>,
}

/// Validate a webview request. Error messages never contain header values (they may be keys).
pub fn check(req: &LlmProxyRequest) -> Result<Checked, LlmServerError> {
    let id = req.request_id.trim();
    if id.is_empty() || id.len() > MAX_REQUEST_ID_LEN || id.chars().any(char::is_control) {
        return Err(invalid("the request id must be 1 to 128 printable characters"));
    }
    if req.url.len() > MAX_URL_LEN {
        return Err(invalid("the URL is too long"));
    }
    let url = Url::parse(req.url.trim()).map_err(|_| invalid("the URL is not a valid absolute http(s) URL"))?;
    if !matches!(url.scheme(), "http" | "https") {
        return Err(invalid("only http:// and https:// URLs can be proxied"));
    }
    if url.host_str().is_none_or(str::is_empty) {
        return Err(invalid("the URL has no host"));
    }
    if !url.username().is_empty() || url.password().is_some() {
        return Err(invalid("credentials in the URL are not allowed; send the key in a header"));
    }
    let method = match req
        .method
        .as_deref()
        .map(str::trim)
        .unwrap_or("POST")
        .to_ascii_uppercase()
        .as_str()
    {
        "POST" => Method::POST,
        "GET" => Method::GET,
        other => return Err(invalid(format!("method {other} is not supported (GET or POST)"))),
    };
    if req.headers.len() > MAX_HEADERS {
        return Err(invalid("too many headers"));
    }
    let mut headers = HeaderMap::new();
    for (name, value) in &req.headers {
        let hn = HeaderName::from_bytes(name.trim().as_bytes()).map_err(|_| invalid("a header name is not valid"))?;
        if FORBIDDEN_HEADERS.contains(&hn.as_str()) {
            return Err(invalid(format!("header {} is set by the proxy", hn.as_str())));
        }
        if value.len() > MAX_HEADER_VALUE_LEN {
            return Err(invalid(format!("header {} is too long", hn.as_str())));
        }
        let hv = HeaderValue::from_str(value).map_err(|_| invalid(format!("header {} has an invalid value", hn.as_str())))?;
        headers.insert(hn, hv);
    }
    let body = match (method == Method::POST, &req.body) {
        (true, Some(b)) => {
            if b.len() > MAX_REQUEST_BYTES {
                return Err(invalid("the request body is larger than 4 MB"));
            }
            if !headers.contains_key(CONTENT_TYPE) {
                headers.insert(CONTENT_TYPE, HeaderValue::from_static("application/json"));
            }
            Some(b.clone())
        }
        _ => None,
    };
    let timeout = match req.timeout_ms {
        None | Some(0) => None,
        Some(ms) => Some(Duration::from_millis(ms.min(MAX_TIMEOUT_MS))),
    };
    Ok(Checked {
        id: id.to_string(),
        url,
        method,
        headers,
        body,
        stream: req.stream,
        timeout,
    })
}

/// Incremental UTF-8 decoding across network reads.
#[derive(Default)]
pub struct Utf8Stream {
    carry: Vec<u8>,
}

impl Utf8Stream {
    /// Text decodable so far; an incomplete trailing sequence waits for the next read, invalid
    /// bytes become U+FFFD.
    pub fn push(&mut self, bytes: &[u8]) -> String {
        self.carry.extend_from_slice(bytes);
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.carry) {
                Ok(s) => {
                    out.push_str(s);
                    self.carry.clear();
                    return out;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(&String::from_utf8_lossy(&self.carry[..valid]));
                    match e.error_len() {
                        None => {
                            self.carry.drain(..valid);
                            return out;
                        }
                        Some(n) => {
                            out.push('\u{FFFD}');
                            self.carry.drain(..valid + n);
                        }
                    }
                }
            }
        }
    }

    /// Whatever is left at the end of the body.
    pub fn finish(&mut self) -> String {
        let rest = String::from_utf8_lossy(&self.carry).into_owned();
        self.carry.clear();
        rest
    }
}

/// No read timeout on the client: reqwest's also covers the wait for the response headers, which
/// the proxy bounds separately (see the module docs).
fn build_client(system_proxy: bool) -> Client {
    let mut b = Client::builder()
        .connect_timeout(CONNECT_TIMEOUT)
        .redirect(reqwest::redirect::Policy::none())
        .user_agent(format!("Sigil/{} (desktop; https://github.com/Mewnah/Sigil)", env!("CARGO_PKG_VERSION")));
    if !system_proxy {
        b = b.no_proxy();
    }
    b.build().expect("reqwest Client builder")
}

/// The client for `url`: hosts on this PC or the LAN never go through a system proxy.
pub fn client_for(url: &Url) -> &'static Client {
    static DIRECT: OnceLock<Client> = OnceLock::new();
    static PROXIED: OnceLock<Client> = OnceLock::new();
    if is_local_endpoint(url) {
        DIRECT.get_or_init(|| build_client(false))
    } else {
        PROXIED.get_or_init(|| build_client(true))
    }
}

/// A client without a proxy (tests).
#[cfg(test)]
pub fn test_client() -> Client {
    build_client(false)
}

#[derive(Default)]
struct Registry {
    /// Requests in flight.
    running: HashMap<String, watch::Sender<bool>>,
    /// Cancels that arrived before their request (id → when).
    cancelled: HashMap<String, Instant>,
}

impl Registry {
    fn prune(&mut self, now: Instant) {
        self.cancelled
            .retain(|_, at| now.saturating_duration_since(*at) < CANCEL_TOMBSTONE_TTL);
        while self.cancelled.len() >= MAX_TOMBSTONES {
            let Some(oldest) = self
                .cancelled
                .iter()
                .min_by_key(|(_, at)| **at)
                .map(|(id, _)| id.clone())
            else {
                break;
            };
            self.cancelled.remove(&oldest);
        }
    }

    /// Consume an early cancel of `id`.
    fn take_tombstone(&mut self, id: &str, now: Instant) -> bool {
        match self.cancelled.remove(id) {
            Some(at) => now.saturating_duration_since(at) < CANCEL_TOMBSTONE_TTL,
            None => false,
        }
    }
}

fn registry() -> &'static Mutex<Registry> {
    static R: OnceLock<Mutex<Registry>> = OnceLock::new();
    R.get_or_init(Default::default)
}

struct Registration(String);

impl Drop for Registration {
    fn drop(&mut self) {
        registry()
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .running
            .remove(&self.0);
    }
}

/// Abort the request `id`. A request that is not running yet is cancelled when it arrives (within
/// [`CANCEL_TOMBSTONE_TTL`]); ids are unique per request, so a cancel of one that already finished
/// only leaves a tombstone nobody consumes.
pub fn cancel(id: &str) {
    let id = id.trim();
    if id.is_empty() || id.len() > MAX_REQUEST_ID_LEN {
        return;
    }
    let mut r = registry().lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(tx) = r.running.get(id) {
        tx.send_replace(true);
        return;
    }
    let now = Instant::now();
    r.prune(now);
    r.cancelled.insert(id.to_string(), now);
}

/// Requests in flight (tests).
#[cfg(test)]
pub fn in_flight(id: &str) -> bool {
    registry()
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .running
        .contains_key(id)
}

/// Perform `req`, reporting through `sink` (which returns false once the webview is gone). `client`
/// picks the HTTP client for the URL.
pub async fn run(
    req: LlmProxyRequest,
    sink: impl FnMut(LlmProxyEvent) -> bool,
    client: impl FnOnce(&Url) -> Client,
) -> Result<(), LlmServerError> {
    run_with(req, sink, client, STALL_TIMEOUT).await
}

/// [`run`] with an explicit body stall timeout (tests).
pub async fn run_with(
    req: LlmProxyRequest,
    mut sink: impl FnMut(LlmProxyEvent) -> bool,
    client: impl FnOnce(&Url) -> Client,
    stall: Duration,
) -> Result<(), LlmServerError> {
    let checked = check(&req)?;
    let (tx, mut rx) = watch::channel(false);
    {
        let mut r = registry().lock().unwrap_or_else(PoisonError::into_inner);
        if r.take_tombstone(&checked.id, Instant::now()) {
            drop(r);
            tracing::debug!(target: "sigil::llm_server", "proxy request {}: cancelled before it started", checked.id);
            sink(LlmProxyEvent::Error { message: CANCELLED.into() });
            return Ok(());
        }
        if r.running.contains_key(&checked.id) {
            return Err(invalid("a proxy request with this id is already running"));
        }
        r.running.insert(checked.id.clone(), tx);
    }
    let _registration = Registration(checked.id.clone());
    let http = client(&checked.url);
    let timeout = checked.timeout;
    let outcome = {
        let work = perform(&http, checked, &mut sink, stall);
        let bounded = async {
            match timeout {
                Some(t) => tokio::time::timeout(t, work)
                    .await
                    .unwrap_or_else(|_| Err(format!("LLM request: no complete response within {:.1} s", t.as_secs_f64()))),
                None => work.await,
            }
        };
        tokio::select! {
            biased;
            _ = rx.wait_for(|c| *c) => Err(CANCELLED.to_string()),
            r = bounded => r,
        }
    };
    if let Err(message) = outcome {
        tracing::debug!(target: "sigil::llm_server", "proxy request {}: {message}", req.request_id);
        sink(LlmProxyEvent::Error { message });
    }
    Ok(())
}

const WINDOW_GONE: &str = "the window that asked for this request is gone";

async fn perform(http: &Client, req: Checked, sink: &mut impl FnMut(LlmProxyEvent) -> bool, stall: Duration) -> Result<(), String> {
    let what = "LLM request";
    // The overall deadline (when set) already bounds the whole request in `run`.
    let headers_deadline = req.timeout.unwrap_or(HEADERS_TIMEOUT);
    let mut rb = http.request(req.method, req.url).headers(req.headers);
    if let Some(body) = req.body {
        rb = rb.body(body);
    }
    let mut resp = tokio::time::timeout(headers_deadline, rb.send())
        .await
        .map_err(|_| format!("{what}: no response within {:.1} s", headers_deadline.as_secs_f64()))?
        .map_err(|e| describe_request_error(what, &e, headers_deadline))?;
    let mut headers = BTreeMap::new();
    for (name, value) in resp.headers() {
        if headers.len() >= MAX_RESPONSE_HEADERS {
            break;
        }
        if let Ok(v) = value.to_str() {
            if v.len() <= MAX_RESPONSE_HEADER_LEN {
                headers.insert(name.as_str().to_string(), v.to_string());
            }
        }
    }
    if !sink(LlmProxyEvent::Response {
        status: resp.status().as_u16(),
        headers,
    }) {
        return Err(WINDOW_GONE.into());
    }
    let mut total = 0usize;
    let mut text = Utf8Stream::default();
    let mut whole = String::new();
    loop {
        let chunk = tokio::time::timeout(stall, resp.chunk())
            .await
            .map_err(|_| format!("{what}: the server sent nothing for {:.1} s (stalled)", stall.as_secs_f64()))?
            .map_err(|e| describe_request_error(what, &e, stall))?;
        let Some(bytes) = chunk else {
            break;
        };
        total = total.saturating_add(bytes.len());
        if total > MAX_RESPONSE_BYTES {
            return Err("LLM request: the response is larger than 8 MB".into());
        }
        let decoded = text.push(&bytes);
        if req.stream {
            if !decoded.is_empty() && !sink(LlmProxyEvent::Chunk { data: decoded }) {
                return Err(WINDOW_GONE.into());
            }
        } else {
            whole.push_str(&decoded);
        }
    }
    let rest = text.finish();
    if req.stream {
        if !rest.is_empty() && !sink(LlmProxyEvent::Chunk { data: rest }) {
            return Err(WINDOW_GONE.into());
        }
    } else {
        whole.push_str(&rest);
        if !whole.is_empty() && !sink(LlmProxyEvent::Chunk { data: whole }) {
            return Err(WINDOW_GONE.into());
        }
    }
    if !sink(LlmProxyEvent::Done) {
        return Err(WINDOW_GONE.into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Instant;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::{TcpListener, TcpStream};

    fn req(id: &str, url: &str) -> LlmProxyRequest {
        LlmProxyRequest {
            request_id: id.into(),
            url: url.into(),
            ..Default::default()
        }
    }

    #[test]
    fn validation() {
        assert!(check(&req("a", "http://192.168.1.20:11434/v1/chat/completions")).is_ok());
        let err = |r: LlmProxyRequest| check(&r).unwrap_err();
        assert_eq!(err(req("", "http://x/")).kind, LlmServerErrorKind::InvalidInput);
        assert!(err(req("a", "file:///C:/x")).message.contains("http"));
        assert!(err(req("a", "not a url")).message.contains("valid"));
        assert!(err(req("a", "http://user:secret@host/"))
            .message
            .contains("credentials"));
        let mut r = req("a", "http://host/");
        r.method = Some("DELETE".into());
        assert!(err(r).message.contains("DELETE"));
        let mut r = req("a", "http://host/");
        r.headers.insert("Host".into(), "evil".into());
        assert!(err(r).message.contains("host"));
        // A bad header value is refused without echoing it (it may be a key).
        let mut r = req("a", "http://host/");
        r.headers
            .insert("authorization".into(), "Bearer sk-secret\nx".into());
        let e = err(r);
        assert!(e.message.contains("authorization") && !e.message.contains("sk-secret"), "{}", e.message);
        // POST bodies get a JSON content type; GET bodies are dropped; timeouts are capped.
        let mut r = req("a", "https://api.example/v1/models");
        r.body = Some("{}".into());
        r.timeout_ms = Some(u64::MAX);
        let c = check(&r).unwrap();
        assert_eq!(c.headers.get(CONTENT_TYPE).unwrap(), "application/json");
        assert_eq!(c.timeout, Some(Duration::from_millis(MAX_TIMEOUT_MS)));
        r.method = Some("get".into());
        let c = check(&r).unwrap();
        assert!(c.body.is_none() && c.method == Method::GET);
        let mut r = req("a", "http://host/");
        r.body = Some("x".repeat(MAX_REQUEST_BYTES + 1));
        assert!(err(r).message.contains("4 MB"));
    }

    #[test]
    fn utf8_is_never_split_between_chunks() {
        let text = "こんにちは, world 🌍!";
        let bytes = text.as_bytes();
        for cut in 0..bytes.len() {
            let mut d = Utf8Stream::default();
            let mut out = d.push(&bytes[..cut]);
            out.push_str(&d.push(&bytes[cut..]));
            out.push_str(&d.finish());
            assert_eq!(out, text, "cut {cut}");
        }
        // Byte by byte.
        let mut d = Utf8Stream::default();
        let out: String = bytes.iter().map(|b| d.push(&[*b])).collect();
        assert_eq!(out, text);
        // Invalid bytes are replaced, the rest kept; a dangling lead byte at the end is flushed lossy.
        let mut d = Utf8Stream::default();
        assert_eq!(d.push(b"a\xffb"), "a\u{FFFD}b");
        assert_eq!(d.push(b"c\xe3\x81"), "c");
        assert_eq!(d.finish(), "\u{FFFD}");
    }

    // --- a tiny HTTP server for the transport tests ---

    async fn read_request(sock: &mut TcpStream) -> (String, String) {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 4096];
        loop {
            let n = sock.read(&mut tmp).await.unwrap_or(0);
            if n == 0 {
                break;
            }
            buf.extend_from_slice(&tmp[..n]);
            if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..end]).to_string();
                let len = head
                    .lines()
                    .find_map(|l| {
                        l.to_ascii_lowercase()
                            .strip_prefix("content-length:")
                            .map(|v| v.trim().parse::<usize>().unwrap_or(0))
                    })
                    .unwrap_or(0);
                while buf.len() < end + 4 + len {
                    let n = sock.read(&mut tmp).await.unwrap_or(0);
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&tmp[..n]);
                }
                let body = String::from_utf8_lossy(&buf[end + 4..]).to_string();
                return (head, body);
            }
        }
        (String::new(), String::new())
    }

    async fn serve(sock: &mut TcpStream) {
        let (head, body) = read_request(sock).await;
        let path = head.split_whitespace().nth(1).unwrap_or("/").to_string();
        let ok_head = |ct: &str| format!("HTTP/1.1 200 OK\r\nContent-Type: {ct}\r\nX-Test: yes\r\nConnection: close\r\n\r\n");
        match path.as_str() {
            "/echo" => {
                let payload = format!("{head}\n\n{body}");
                let _ = sock
                    .write_all(
                        format!(
                            "HTTP/1.1 201 Created\r\nContent-Type: text/plain\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{payload}",
                            payload.len()
                        )
                        .as_bytes(),
                    )
                    .await;
            }
            "/sse" => {
                let _ = sock
                    .write_all(ok_head("text/event-stream").as_bytes())
                    .await;
                // "データ" split inside a character across two writes.
                let msg = "data: {\"t\":\"データ\"}\n\n".as_bytes();
                let (a, b) = msg.split_at(14);
                for part in [a, b, b"data: [DONE]\n\n".as_slice()] {
                    let _ = sock.write_all(part).await;
                    let _ = sock.flush().await;
                    tokio::time::sleep(Duration::from_millis(30)).await;
                }
            }
            "/slow-headers" => {
                // A non-streaming reply: nothing at all until "generation" ends.
                tokio::time::sleep(Duration::from_millis(700)).await;
                let body = "{\"choices\":[{\"message\":{\"content\":\"ok\"}}]}";
                let _ = sock
                    .write_all(
                        format!(
                            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
                            body.len()
                        )
                        .as_bytes(),
                    )
                    .await;
            }
            "/no-headers" => {
                tokio::time::sleep(Duration::from_secs(5)).await;
            }
            "/redirect" => {
                let _ = sock
                    .write_all(b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:1/elsewhere\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
            }
            "/huge" => {
                let _ = sock.write_all(ok_head("text/plain").as_bytes()).await;
                let block = vec![b'x'; 64 * 1024];
                for _ in 0..(MAX_RESPONSE_BYTES / block.len() + 2) {
                    if sock.write_all(&block).await.is_err() {
                        break;
                    }
                }
            }
            "/stall" | "/forever" => {
                let _ = sock
                    .write_all(ok_head("text/event-stream").as_bytes())
                    .await;
                let _ = sock.write_all(b"data: first\n\n").await;
                let _ = sock.flush().await;
                if path == "/forever" {
                    for _ in 0..200 {
                        tokio::time::sleep(Duration::from_millis(50)).await;
                        if sock.write_all(b"data: tick\n\n").await.is_err() {
                            break;
                        }
                    }
                } else {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                }
            }
            _ => {
                let _ = sock
                    .write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    .await;
            }
        }
        let _ = sock.shutdown().await;
    }

    async fn server() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move {
            while let Ok((mut sock, _)) = listener.accept().await {
                tokio::spawn(async move { serve(&mut sock).await });
            }
        });
        format!("http://{addr}")
    }

    async fn collect(r: LlmProxyRequest, stall: Duration) -> (Result<(), LlmServerError>, Vec<LlmProxyEvent>) {
        let mut events = Vec::new();
        let res = run_with(
            r,
            |e| {
                events.push(e);
                true
            },
            |_| test_client(),
            stall,
        )
        .await;
        (res, events)
    }

    fn chunks(events: &[LlmProxyEvent]) -> String {
        events
            .iter()
            .filter_map(|e| match e {
                LlmProxyEvent::Chunk { data } => Some(data.as_str()),
                _ => None,
            })
            .collect()
    }

    #[tokio::test]
    async fn streams_sse_with_headers_and_status() {
        let base = server().await;
        let mut r = req("t-sse", &format!("{base}/sse"));
        r.stream = true;
        r.body = Some("{\"stream\":true}".into());
        let (res, events) = collect(r, STALL_TIMEOUT).await;
        res.unwrap();
        match &events[0] {
            LlmProxyEvent::Response { status, headers } => {
                assert_eq!(*status, 200);
                assert_eq!(headers.get("content-type").map(String::as_str), Some("text/event-stream"));
                assert_eq!(headers.get("x-test").map(String::as_str), Some("yes"), "names are lower-case");
            }
            e => panic!("first event {e:?}"),
        }
        assert!(events.len() >= 4, "several chunks: {events:?}");
        assert_eq!(chunks(&events), "data: {\"t\":\"データ\"}\n\ndata: [DONE]\n\n");
        assert_eq!(events.last(), Some(&LlmProxyEvent::Done));
        assert!(!in_flight("t-sse"));
    }

    #[tokio::test]
    async fn forwards_headers_and_body_and_returns_the_whole_body_once() {
        let base = server().await;
        let mut r = req("t-echo", &format!("{base}/echo"));
        r.headers.insert("Authorization".into(), "Bearer k1".into());
        r.headers.insert("x-goog-api-key".into(), "g2".into());
        r.body = Some("{\"messages\":[]}".into());
        let (res, events) = collect(r, STALL_TIMEOUT).await;
        res.unwrap();
        assert!(matches!(events[0], LlmProxyEvent::Response { status: 201, .. }));
        assert_eq!(events.len(), 3, "response, one chunk, done: {events:?}");
        let echoed = chunks(&events).to_ascii_lowercase();
        assert!(echoed.starts_with("post /echo http/1.1"), "{echoed}");
        assert!(
            echoed.contains("authorization: bearer k1") && echoed.contains("x-goog-api-key: g2"),
            "{echoed}"
        );
        assert!(echoed.contains("content-type: application/json"), "{echoed}");
        assert!(echoed.ends_with("{\"messages\":[]}"), "{echoed}");
    }

    #[tokio::test]
    async fn never_follows_redirects() {
        let base = server().await;
        let (res, events) = collect(req("t-redir", &format!("{base}/redirect")), STALL_TIMEOUT).await;
        res.unwrap();
        match &events[0] {
            LlmProxyEvent::Response { status, headers } => {
                assert_eq!(*status, 302);
                assert_eq!(headers.get("location").map(String::as_str), Some("http://127.0.0.1:1/elsewhere"));
            }
            e => panic!("{e:?}"),
        }
        assert_eq!(events.last(), Some(&LlmProxyEvent::Done));
    }

    #[tokio::test]
    async fn caps_the_body() {
        let base = server().await;
        let mut r = req("t-huge", &format!("{base}/huge"));
        r.stream = true;
        let (res, events) = collect(r, STALL_TIMEOUT).await;
        res.unwrap();
        match events.last() {
            Some(LlmProxyEvent::Error { message }) => assert!(message.contains("8 MB"), "{message}"),
            e => panic!("{e:?}"),
        }
    }

    #[tokio::test]
    async fn stall_and_deadline_fail_the_request() {
        let base = server().await;
        let mut r = req("t-stall", &format!("{base}/stall"));
        r.stream = true;
        let t0 = Instant::now();
        let (res, events) = collect(r, Duration::from_millis(400)).await;
        res.unwrap();
        assert!(t0.elapsed() < Duration::from_secs(3));
        assert_eq!(chunks(&events), "data: first\n\n", "what arrived before the stall is delivered");
        assert!(matches!(events.last(), Some(LlmProxyEvent::Error { .. })), "{events:?}");

        let mut r = req("t-deadline", &format!("{base}/forever"));
        r.stream = true;
        r.timeout_ms = Some(300);
        let t0 = Instant::now();
        let (res, events) = collect(r, STALL_TIMEOUT).await;
        res.unwrap();
        assert!(t0.elapsed() < Duration::from_secs(3));
        match events.last() {
            Some(LlmProxyEvent::Error { message }) => assert!(message.contains("0.3 s"), "{message}"),
            e => panic!("{e:?}"),
        }
    }

    #[tokio::test]
    async fn the_stall_timeout_does_not_cut_the_wait_for_headers() {
        let base = server().await;
        // Headers after 0.7 s with a 0.3 s stall timeout: the request still succeeds (a
        // non-streaming generation or a cold model load sends nothing until it is done).
        let (res, events) = collect(req("t-slow-headers", &format!("{base}/slow-headers")), Duration::from_millis(300)).await;
        res.unwrap();
        assert!(matches!(events[0], LlmProxyEvent::Response { status: 200, .. }), "{events:?}");
        assert!(chunks(&events).contains("\"ok\""), "{events:?}");
        assert_eq!(events.last(), Some(&LlmProxyEvent::Done));
        // The caller's deadline bounds the wait for headers.
        let mut r = req("t-no-headers", &format!("{base}/no-headers"));
        r.timeout_ms = Some(300);
        let t0 = Instant::now();
        let (res, events) = collect(r, Duration::from_millis(100)).await;
        res.unwrap();
        assert!(t0.elapsed() < Duration::from_secs(3));
        assert_eq!(events.len(), 1, "{events:?}");
        match &events[0] {
            LlmProxyEvent::Error { message } => assert!(message.contains("0.3 s"), "{message}"),
            e => panic!("{e:?}"),
        }
    }

    #[tokio::test]
    async fn a_cancel_that_arrives_before_its_request_still_cancels_it() {
        cancel("t-early");
        // Nothing listens on port 1: had the request been sent, the error would be a connection error.
        let (res, events) = collect(req("t-early", "http://127.0.0.1:1/v1/chat/completions"), STALL_TIMEOUT).await;
        res.unwrap();
        assert_eq!(events, vec![LlmProxyEvent::Error { message: CANCELLED.into() }]);
        assert!(!in_flight("t-early"));
        // The early cancel is consumed once.
        let base = server().await;
        let (res, events) = collect(req("t-early", &format!("{base}/echo")), STALL_TIMEOUT).await;
        res.unwrap();
        assert_eq!(events.last(), Some(&LlmProxyEvent::Done));
    }

    #[test]
    fn early_cancels_expire_and_are_bounded() {
        let t0 = Instant::now();
        let mut r = Registry::default();
        r.cancelled.insert("old".into(), t0);
        assert!(!r.take_tombstone("old", t0 + CANCEL_TOMBSTONE_TTL + Duration::from_secs(1)), "expired");
        assert!(r.cancelled.is_empty(), "consumed either way");
        r.cancelled.insert("fresh".into(), t0);
        assert!(r.take_tombstone("fresh", t0 + Duration::from_secs(1)));
        assert!(!r.take_tombstone("fresh", t0 + Duration::from_secs(1)), "only once");
        for i in 0..(MAX_TOMBSTONES as u64 * 2) {
            let now = t0 + Duration::from_millis(i);
            r.prune(now);
            r.cancelled.insert(format!("id{i}"), now);
        }
        assert!(r.cancelled.len() <= MAX_TOMBSTONES, "{}", r.cancelled.len());
        assert!(r.cancelled.contains_key(&format!("id{}", MAX_TOMBSTONES * 2 - 1)), "the newest are kept");
        assert!(!r.cancelled.contains_key("id0"), "the oldest go first");
        r.prune(t0 + CANCEL_TOMBSTONE_TTL * 2);
        assert!(r.cancelled.is_empty());
        // Ids the proxy would refuse are not remembered.
        cancel("");
        cancel(&"x".repeat(MAX_REQUEST_ID_LEN + 1));
        let reg = registry().lock().unwrap_or_else(PoisonError::into_inner);
        assert!(!reg.cancelled.contains_key("") && !reg.cancelled.keys().any(|k| k.len() > MAX_REQUEST_ID_LEN));
    }

    #[tokio::test]
    async fn cancel_aborts_a_running_request() {
        let base = server().await;
        let mut r = req("t-cancel", &format!("{base}/forever"));
        r.stream = true;
        let events = Arc::new(std::sync::Mutex::new(Vec::new()));
        let ev = events.clone();
        let task = tokio::spawn(async move {
            run(
                r,
                move |e| {
                    ev.lock().unwrap().push(e);
                    true
                },
                |_| test_client(),
            )
            .await
        });
        // Wait for the first chunk, then cancel.
        let t0 = Instant::now();
        while !events
            .lock()
            .unwrap()
            .iter()
            .any(|e| matches!(e, LlmProxyEvent::Chunk { .. }))
        {
            assert!(t0.elapsed() < Duration::from_secs(5));
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        // A second request with the same id is refused while the first runs.
        let (dup, _) = collect(req("t-cancel", &format!("{base}/sse")), STALL_TIMEOUT).await;
        assert_eq!(dup.unwrap_err().kind, LlmServerErrorKind::InvalidInput);
        cancel("t-cancel");
        task.await.unwrap().unwrap();
        let events = events.lock().unwrap();
        assert_eq!(events.last(), Some(&LlmProxyEvent::Error { message: CANCELLED.into() }));
        assert!(!in_flight("t-cancel"));
        cancel("t-cancel"); // finished: only a tombstone nobody consumes (ids are unique)
    }

    #[tokio::test]
    async fn a_closed_window_stops_the_request_and_connection_errors_are_events() {
        let base = server().await;
        let mut r = req("t-gone", &format!("{base}/forever"));
        r.stream = true;
        let mut n = 0;
        let t0 = Instant::now();
        let res = run(
            r,
            |_| {
                n += 1;
                n < 2
            },
            |_| test_client(),
        )
        .await;
        res.unwrap();
        assert!(t0.elapsed() < Duration::from_secs(3), "stops as soon as the sink refuses");

        // Nothing listens on port 1: an error event (no response event), and the call still resolves.
        let (res, events) = collect(req("t-refused", "http://127.0.0.1:1/v1/chat/completions"), STALL_TIMEOUT).await;
        res.unwrap();
        assert_eq!(events.len(), 1);
        match &events[0] {
            LlmProxyEvent::Error { message } => assert!(message.contains("LLM request"), "{message}"),
            e => panic!("{e:?}"),
        }
    }

    #[test]
    fn local_hosts_skip_the_system_proxy() {
        // Same client instance per class.
        let a = client_for(&Url::parse("http://127.0.0.1:5000/v1").unwrap()) as *const Client;
        let b = client_for(&Url::parse("http://192.168.1.2:11434/v1").unwrap()) as *const Client;
        let c = client_for(&Url::parse("https://openrouter.ai/api/v1").unwrap()) as *const Client;
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
