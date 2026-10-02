//! One Read-aloud request on a blocking thread (P6-H): connect (TCP, TLS, WebSocket upgrade with
//! the browser's headers), send `speech.config` + `ssml`, and decode the MP3 the service streams
//! back **as it arrives** into the sink utterance.
//!
//! The WebSocket is read through [`AudioReader`], an `io::Read` that the MP3 decoder (rodio's
//! symphonia decoder) pulls from: it returns audio bytes as they come, end of stream at `turn.end`,
//! and an error (recorded in [`ReadState`]) on cancel, timeout or a broken connection. Decoded
//! samples are pushed into the sink in ~100 ms blocks on the same thread; while a push waits for
//! the sink's back-pressure the socket is simply not read (TCP buffers the rest), and that time
//! never counts as a stall: the stall clock starts when the decoder asks for more bytes.
//!
//! Timing: the first audio must arrive by `first_audio` after the request started (connect and
//! TLS included); later, the service may pause at most `stall` between two pieces of audio. The
//! service must also be reachable (DNS and TCP connect) within `connect`, a third of the first-audio
//! limit (at least 1 s): a connect that takes longer means a lost SYN or a dead route, which no
//! retransmit within the limit would fix, so an unreachable service is reported after about 1 s
//! instead of 3 (and the local voice takes over). The service's addresses are raced ("Happy
//! Eyeballs", RFC 8305, [`race_connect`]): `speech.platform.bing.com` is dual-stack, and on a PC
//! whose IPv6 path is dead (a VPN that drops IPv6, a router with broken IPv6) the IPv4 attempt
//! starts 250 ms after the IPv6 one instead of after the whole connect bound. A `403` at the
//! upgrade means the `Sec-MS-GEC` token was refused, usually because the PC clock is off: the
//! clock skew is taken from the server's `Date` header and the request is retried once.
//!
//! A close by the service after the SSML was sent is classified by its code and reason: a refused
//! voice (or an unsupported request) is `rejected` (a settings problem), a payload or size refusal
//! is about this sentence's text and is `no_audio` (that sentence only, no breaker), a throttling
//! policy is `rate_limited`, anything else `unavailable`.

use std::collections::VecDeque;
use std::io::{self, Read, Seek, SeekFrom};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use rodio::Source;
use tungstenite::client::IntoClientRequest;
use tungstenite::handshake::HandshakeError;
use tungstenite::http::header::{HeaderName, HeaderValue, CACHE_CONTROL, COOKIE, ORIGIN, PRAGMA, USER_AGENT};
use tungstenite::protocol::frame::coding::CloseCode;
use tungstenite::stream::MaybeTlsStream;
use tungstenite::{Message, WebSocket};

use super::cloud::{parse_http_date, parse_retry_after, push_pieces, unix_now, Block, CancelWatch, Failure, FailureKind, PushOutcome, SpeakError};
use super::protocol::{self, header, parse_binary_message, split_text_message, SAMPLE_RATE};
use crate::services::openai_speech::PcmSink;
use crate::services::tts_sink::SpeakReport;

/// Longest single socket wait, so cancellation is noticed promptly.
const POLL: Duration = Duration::from_millis(50);
/// Decoded audio collected before a push into the sink (~100 ms at 24 kHz).
const PUSH_SAMPLES: usize = 2400;
/// Name used in messages.
pub(crate) const WHAT: &str = "Edge voices";

type Socket = WebSocket<MaybeTlsStream<TcpStream>>;

/// Everything one request needs (built and validated by the plugin).
#[derive(Clone, Debug)]
pub(crate) struct Plan {
    /// `wss://…/edge/v1` (a `ws://` test server in tests).
    pub wss_base: String,
    /// The service's voice name (long form).
    pub voice: String,
    pub rate: String,
    pub pitch: String,
    /// Escaped text, one request per piece (a sentence is one piece).
    pub pieces: Vec<String>,
    pub first_audio: Duration,
    pub stall: Duration,
    /// Bound of DNS + TCP connect ([`connect_bound`]).
    pub connect: Duration,
}

/// Shortest connect bound.
const MIN_CONNECT: Duration = Duration::from_secs(1);

/// The DNS + TCP connect bound for a first-audio limit: a third of it, at least 1 s (never more
/// than the limit itself).
pub(crate) fn connect_bound(first_audio: Duration) -> Duration {
    (first_audio / 3).max(MIN_CONNECT).min(first_audio)
}

/// How far one request got (read by the async watchdog while the blocking side may be stuck in
/// DNS, connect or TLS).
#[derive(Debug, Default)]
pub(crate) struct Progress {
    /// A TCP connection to the service was made.
    pub connected: AtomicBool,
    /// Audio bytes arrived.
    pub audio: AtomicBool,
}

/// Seconds for messages, rounded up, at least 1.
pub(crate) fn secs(d: Duration) -> u64 {
    d.as_millis().div_ceil(1000).max(1) as u64
}

fn unreachable_message(connect: Duration) -> String {
    format!("{WHAT}: cannot reach the service (no connection within {} s). Check the internet connection.", secs(connect))
}

/// The failure of a service that could not be reached within `connect`.
pub(crate) fn unreachable(connect: Duration) -> SpeakError {
    SpeakError::fail(FailureKind::Network, unreachable_message(connect))
}

fn stalled_before_audio(first_audio: Duration) -> SpeakError {
    SpeakError::fail(
        FailureKind::Stalled,
        format!("{WHAT}: no audio within {} s (the service did not answer in time).", secs(first_audio)),
    )
}

/// A random 32-character hex id (connection / request ids, the `muid` cookie).
pub(crate) fn random_id() -> String {
    match crate::services::process_host::spawn::random_key() {
        Ok(k) => k[..32].to_string(),
        Err(_) => {
            // The system RNG is always there on Windows; an id only has to be unique.
            let n = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_nanos());
            format!("{:032x}", n ^ (u128::from(std::process::id()) << 64))
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Connect
// ---------------------------------------------------------------------------------------------

/// Delay before the next address is tried while the earlier attempts are still connecting
/// (RFC 8305 "Connection Attempt Delay", its recommended value).
pub(crate) const ATTEMPT_DELAY: Duration = Duration::from_millis(250);

/// `addrs` in the order RFC 8305 §4 tries them: the address families alternate, starting with the
/// first address's family (the system's preference), each family keeping its own order.
pub(crate) fn interleave_families(addrs: Vec<SocketAddr>) -> Vec<SocketAddr> {
    let Some(first_v6) = addrs.first().map(SocketAddr::is_ipv6) else {
        return addrs;
    };
    let (mut first, mut other): (VecDeque<_>, VecDeque<_>) = addrs.into_iter().partition(|a| a.is_ipv6() == first_v6);
    let mut out = Vec::with_capacity(first.len() + other.len());
    while !first.is_empty() || !other.is_empty() {
        out.extend(first.pop_front());
        out.extend(other.pop_front());
    }
    out
}

/// Why [`race_connect`] made no connection.
#[derive(Debug)]
pub(crate) enum RaceFail {
    /// No address answered by the deadline.
    TimedOut,
    /// Every address failed; the last error.
    Failed(io::Error),
    /// There was no address to try.
    NoAddress,
}

/// Connect to the first of `addrs` that answers by `deadline`, racing them ("Happy Eyeballs",
/// RFC 8305): the attempts start [`ATTEMPT_DELAY`] apart (the next one at once when an attempt
/// fails) and run in parallel; the first connection wins. An attempt still running when this
/// returns ends on its own by `deadline` (a late connection is dropped unused).
pub(crate) fn race_connect(addrs: &[SocketAddr], deadline: Instant) -> Result<TcpStream, RaceFail> {
    race_connect_with(addrs, deadline, |addr, left| TcpStream::connect_timeout(&addr, left))
}

/// [`race_connect`] with the connect function given (`connect(addr, time_left)`; tests simulate an
/// address that never answers).
pub(crate) fn race_connect_with<C>(addrs: &[SocketAddr], deadline: Instant, connect: C) -> Result<TcpStream, RaceFail>
where
    C: Fn(SocketAddr, Duration) -> io::Result<TcpStream> + Clone + Send + 'static,
{
    let (tx, rx) = mpsc::channel::<io::Result<TcpStream>>();
    let mut next = 0;
    let mut running = 0usize;
    let mut next_start = Instant::now();
    let mut last_error: Option<io::Error> = None;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(RaceFail::TimedOut);
        }
        if next < addrs.len() && (running == 0 || now >= next_start) {
            let addr = addrs[next];
            let left = deadline - now;
            let tx = tx.clone();
            let connect = connect.clone();
            let spawned = std::thread::Builder::new()
                .name("edge-tts-connect".into())
                .spawn(move || {
                    let _ = tx.send(connect(addr, left));
                });
            match spawned {
                Ok(_) => running += 1,
                Err(e) => last_error = Some(e),
            }
            next += 1;
            next_start = now + ATTEMPT_DELAY;
            continue;
        }
        if running == 0 {
            // Every address was tried and failed.
            return Err(last_error.map_or(RaceFail::NoAddress, RaceFail::Failed));
        }
        let wake = if next < addrs.len() { next_start.min(deadline) } else { deadline };
        match rx.recv_timeout(wake.saturating_duration_since(now)) {
            Ok(Ok(stream)) => return Ok(stream),
            Ok(Err(e)) => {
                running -= 1;
                last_error = Some(e);
                // A refused or failed attempt starts the next address at once.
                next_start = Instant::now();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            // Cannot happen (`tx` is held here); end the race rather than spin.
            Err(mpsc::RecvTimeoutError::Disconnected) => return Err(last_error.map_or(RaceFail::TimedOut, RaceFail::Failed)),
        }
    }
}

/// Why a connection attempt failed.
enum ConnectFail {
    /// The token was refused (HTTP 403): the server's clock (unix seconds), when it said.
    Forbidden(Option<i64>),
    Failed(SpeakError),
}

fn is_timeout(e: &tungstenite::Error) -> bool {
    // Unix reports a socket read timeout as WouldBlock, Windows as TimedOut.
    matches!(e, tungstenite::Error::Io(io) if matches!(io.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut))
}

fn set_timeouts(stream: &TcpStream, t: Duration) {
    let t = Some(t.max(Duration::from_millis(1)));
    let _ = stream.set_read_timeout(t);
    let _ = stream.set_write_timeout(t);
}

fn set_socket_timeout(socket: &Socket, t: Duration) {
    match socket.get_ref() {
        MaybeTlsStream::Plain(s) => set_timeouts(s, t),
        MaybeTlsStream::NativeTls(s) => set_timeouts(s.get_ref(), t),
        #[allow(unreachable_patterns)]
        _ => {}
    }
}

fn header_value(v: &str) -> HeaderValue {
    HeaderValue::from_str(v).unwrap_or_else(|_| HeaderValue::from_static(""))
}

/// Open the WebSocket for one request: DNS and TCP connect bounded by `connect_deadline` (then
/// `progress.connected` is set), TLS and the upgrade by `deadline`.
fn connect(plan: &Plan, deadline: Instant, connect_deadline: Instant, progress: &Progress, now_unix: i64) -> Result<Socket, ConnectFail> {
    let fail = |kind, msg: String| ConnectFail::Failed(SpeakError::fail(kind, msg));
    let gec = protocol::sec_ms_gec(now_unix);
    let url = protocol::wss_url(&plan.wss_base, &random_id(), &gec);
    let mut request = url
        .as_str()
        .into_client_request()
        .map_err(|e| fail(FailureKind::Other, format!("{WHAT}: invalid service URL ({e}).")))?;
    {
        let h = request.headers_mut();
        h.insert(PRAGMA, HeaderValue::from_static("no-cache"));
        h.insert(CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        h.insert(ORIGIN, HeaderValue::from_static(protocol::ORIGIN));
        h.insert(USER_AGENT, header_value(&protocol::user_agent()));
        h.insert(HeaderName::from_static("accept-encoding"), HeaderValue::from_static("gzip, deflate, br, zstd"));
        h.insert(HeaderName::from_static("accept-language"), HeaderValue::from_static("en-US,en;q=0.9"));
        h.insert(COOKIE, header_value(&format!("muid={};", random_id().to_ascii_uppercase())));
    }
    let uri = request.uri().clone();
    let tls = match uri.scheme_str() {
        Some("wss") => true,
        Some("ws") => false,
        _ => return Err(fail(FailureKind::Other, format!("{WHAT}: the service URL must be ws:// or wss://."))),
    };
    let host = uri.host().unwrap_or("").trim_start_matches('[').trim_end_matches(']').to_string();
    let port = uri.port_u16().unwrap_or(if tls { 443 } else { 80 });
    let timed_out = || fail(FailureKind::Stalled, format!("{WHAT}: the service did not answer in time."));
    let connect_deadline = connect_deadline.min(deadline);
    let unreachable_now = || ConnectFail::Failed(unreachable(plan.connect));

    // DNS (blocking; the caller's watchdog bounds it from the async side), then the addresses
    // raced within the connect bound.
    let addrs: Vec<_> = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|e| fail(FailureKind::Network, format!("{WHAT}: cannot reach {host} ({e}). Check the internet connection.")))?
        .collect();
    let stream = match race_connect(&interleave_families(addrs), connect_deadline) {
        Ok(s) => s,
        Err(RaceFail::TimedOut) => return Err(unreachable_now()),
        Err(RaceFail::Failed(e)) if e.kind() == io::ErrorKind::TimedOut => return Err(unreachable_now()),
        Err(RaceFail::Failed(e)) => {
            return Err(fail(FailureKind::Network, format!("{WHAT}: cannot connect to {host} ({e}). Check the internet connection.")))
        }
        Err(RaceFail::NoAddress) => return Err(fail(FailureKind::Network, format!("{WHAT}: no address for {host}."))),
    };
    progress.connected.store(true, Ordering::Release);
    let _ = stream.set_nodelay(true);
    let left = deadline.saturating_duration_since(Instant::now());
    if left.is_zero() {
        return Err(timed_out());
    }
    set_timeouts(&stream, left);

    match tungstenite::client_tls_with_config(request, stream, None, None) {
        Ok((socket, _)) => Ok(socket),
        Err(HandshakeError::Interrupted(_)) => Err(timed_out()),
        Err(HandshakeError::Failure(tungstenite::Error::Http(resp))) => {
            let status = resp.status().as_u16();
            let date = resp.headers().get("date").and_then(|v| v.to_str().ok()).and_then(parse_http_date);
            if status == 403 {
                return Err(ConnectFail::Forbidden(date));
            }
            let retry = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| parse_retry_after(v, date.unwrap_or_else(unix_now)));
            let body = resp.body().as_deref().and_then(crate::services::http_client::error_body_summary);
            let reason = match status {
                429 => "the service is limiting requests".to_string(),
                500..=599 => "the service is unavailable".to_string(),
                _ => "the service refused the request".to_string(),
            };
            let message = match body {
                Some(b) => format!("{WHAT}: {reason} (HTTP {status}: {b})."),
                None => format!("{WHAT}: {reason} (HTTP {status})."),
            };
            Err(ConnectFail::Failed(Failure::new(FailureKind::for_status(status), message).with_retry_after(retry).into()))
        }
        Err(HandshakeError::Failure(e)) if is_timeout(&e) => Err(timed_out()),
        Err(HandshakeError::Failure(tungstenite::Error::Tls(e))) => {
            Err(fail(FailureKind::Network, format!("{WHAT}: the secure connection to {host} failed ({e}).")))
        }
        Err(HandshakeError::Failure(e)) => Err(fail(FailureKind::Network, format!("{WHAT}: the connection to {host} failed ({e})."))),
    }
}

// ---------------------------------------------------------------------------------------------
// Reading
// ---------------------------------------------------------------------------------------------

/// What the reader saw (shared with the request after the decoder took the reader).
#[derive(Debug, Default)]
pub(crate) struct ReadState {
    /// `turn.end` arrived: the audio is complete.
    pub done: bool,
    /// MP3 bytes received.
    pub audio_bytes: usize,
    /// Why reading stopped early.
    pub stop: Option<SpeakError>,
}

/// The WebSocket as the MP3 byte stream of one request (see the module docs).
pub(crate) struct AudioReader {
    // The decoder needs `Sync`; a TLS stream is only `Send`.
    socket: Mutex<Socket>,
    state: Arc<Mutex<ReadState>>,
    pending: Vec<u8>,
    pos: usize,
    consumed: u64,
    first_deadline: Instant,
    first_audio: Duration,
    stall: Duration,
    cancel: CancelWatch,
    /// `audio` is set when the first audio bytes arrive (read by the async watchdog).
    progress: Arc<Progress>,
}

impl AudioReader {
    fn lock_state(&self) -> std::sync::MutexGuard<'_, ReadState> {
        self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn halt(&self, why: SpeakError) -> io::Error {
        let mut st = self.lock_state();
        if st.stop.is_none() {
            st.stop = Some(why);
        }
        io::Error::other("the Edge audio stream stopped")
    }

    fn close_failure(frame: Option<tungstenite::protocol::CloseFrame<'_>>) -> SpeakError {
        let (code, reason) = match &frame {
            Some(f) => (Some(f.code), f.reason.trim().to_string()),
            None => (None, String::new()),
        };
        let kind = close_kind(code, &reason);
        let detail = if reason.is_empty() { String::new() } else { format!(": {reason}") };
        let hint = match kind {
            FailureKind::Rejected => " Check the voice name.",
            FailureKind::NoAudio => " The service could not say this sentence.",
            _ => "",
        };
        SpeakError::fail(kind, format!("{WHAT}: the service ended the request without finishing the audio{detail}.{hint}"))
    }
}

/// The failure kind of a close by the service after the request was sent (see the module docs).
pub(crate) fn close_kind(code: Option<CloseCode>, reason: &str) -> FailureKind {
    let r = reason.to_ascii_lowercase();
    let refusal = matches!(code, Some(CloseCode::Invalid | CloseCode::Unsupported | CloseCode::Policy | CloseCode::Size));
    if refusal && r.contains("voice") {
        // The voice of the setup was refused: every sentence would fail the same way.
        return FailureKind::Rejected;
    }
    if refusal && ["quota", "throttl", "too many", "rate limit"].iter().any(|w| r.contains(w)) {
        return FailureKind::RateLimited;
    }
    match code {
        // The payload (this sentence's SSML text) or its size was refused: that sentence only.
        Some(CloseCode::Invalid | CloseCode::Size) => FailureKind::NoAudio,
        // The request itself is not supported: the service changed or the setup is wrong.
        Some(CloseCode::Unsupported) => FailureKind::Rejected,
        // A policy close without a reason, an error or a restart: the service's trouble (backoff).
        _ => FailureKind::Unavailable,
    }
}

impl Read for AudioReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let wait_start = Instant::now();
        loop {
            if self.pos < self.pending.len() {
                let n = (self.pending.len() - self.pos).min(buf.len());
                buf[..n].copy_from_slice(&self.pending[self.pos..self.pos + n]);
                self.pos += n;
                self.consumed += n as u64;
                if self.pos == self.pending.len() {
                    self.pending.clear();
                    self.pos = 0;
                }
                return Ok(n);
            }
            {
                let st = self.lock_state();
                if st.done {
                    return Ok(0);
                }
                if st.stop.is_some() {
                    return Err(io::Error::other("the Edge audio stream stopped"));
                }
            }
            if self.cancel.is_set() {
                return Err(self.halt(SpeakError::Cancelled));
            }
            let got_audio = self.progress.audio.load(Ordering::Acquire);
            let deadline = if got_audio { wait_start + self.stall } else { self.first_deadline };
            let now = Instant::now();
            if now >= deadline {
                let why = if got_audio {
                    SpeakError::fail(
                        FailureKind::Stalled,
                        format!("{WHAT}: the service stopped sending audio for {} s (stalled).", secs(self.stall)),
                    )
                } else {
                    stalled_before_audio(self.first_audio)
                };
                return Err(self.halt(why));
            }
            let result = {
                let socket = self.socket.get_mut().unwrap_or_else(PoisonError::into_inner);
                set_socket_timeout(socket, (deadline - now).min(POLL));
                socket.read()
            };
            match result {
                Ok(Message::Binary(data)) => {
                    let msg = match parse_binary_message(&data) {
                        Ok(m) => m,
                        Err(e) => return Err(self.halt(SpeakError::fail(FailureKind::Other, format!("{WHAT}: {e}")))),
                    };
                    if !msg.path.as_deref().is_some_and(|p| p.eq_ignore_ascii_case("audio")) || msg.data.is_empty() {
                        // Other binary messages, and the empty audio message that ends the stream.
                        continue;
                    }
                    if let Some(ct) = msg.content_type.as_deref() {
                        if !ct.to_ascii_lowercase().starts_with("audio/mpeg") {
                            return Err(self.halt(SpeakError::fail(
                                FailureKind::Other,
                                format!("{WHAT}: the service sent audio in an unexpected format ({ct})."),
                            )));
                        }
                    }
                    self.pending.extend_from_slice(msg.data);
                    self.lock_state().audio_bytes += msg.data.len();
                    self.progress.audio.store(true, Ordering::Release);
                }
                Ok(Message::Text(text)) => {
                    let (headers, _) = split_text_message(&text);
                    if header(&headers, "Path").is_some_and(|p| p.eq_ignore_ascii_case("turn.end")) {
                        self.lock_state().done = true;
                    }
                    // turn.start, response, audio.metadata: nothing to do.
                }
                Ok(Message::Close(frame)) => {
                    if self.lock_state().done {
                        return Ok(0);
                    }
                    return Err(self.halt(Self::close_failure(frame)));
                }
                Ok(_) => {}
                Err(e) if is_timeout(&e) => {}
                Err(tungstenite::Error::ConnectionClosed | tungstenite::Error::AlreadyClosed) => {
                    if self.lock_state().done {
                        return Ok(0);
                    }
                    return Err(self.halt(Self::close_failure(None)));
                }
                Err(e) => {
                    return Err(self.halt(SpeakError::fail(FailureKind::Network, format!("{WHAT}: the connection was lost ({e})."))));
                }
            }
        }
    }
}

impl Seek for AudioReader {
    /// A live stream cannot seek; the decoder only asks for its position.
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        match pos {
            SeekFrom::Current(0) => Ok(self.consumed),
            _ => Err(io::Error::new(io::ErrorKind::Unsupported, "the Edge audio stream cannot seek")),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// One request
// ---------------------------------------------------------------------------------------------

/// The clock the token is computed from, and how to correct it (the plugin's clock skew).
pub(crate) trait Clock: Send + Sync {
    /// Unix seconds, corrected for the measured skew.
    fn now(&self) -> i64;
    /// The server said its time is `server_unix`.
    fn adjust(&self, server_unix: i64);
}

/// Speak every piece of `plan` into `sink`. `progress` records the first connection and the first
/// audio (the async watchdog's connect and first-audio bounds).
pub(crate) fn run(
    plan: &Plan,
    sink: &mut dyn PcmSink,
    cancel: &CancelWatch,
    progress: &Arc<Progress>,
    clock: &dyn Clock,
    started: Instant,
) -> Result<SpeakReport, SpeakError> {
    let mut out = PushOutcome::default();
    for (i, piece) in plan.pieces.iter().enumerate() {
        let deadline = if i == 0 { started + plan.first_audio } else { Instant::now() + plan.first_audio };
        let partial = out.frames > 0;
        let result = speak_piece(plan, piece, deadline, sink, cancel, progress, clock, started, &mut out);
        result.map_err(|e| e.with_partial(partial || out.frames > 0))?;
    }
    if out.frames == 0 {
        return Err(SpeakError::fail(FailureKind::NoAudio, format!("{WHAT}: the service returned no audio for this text.")));
    }
    let first = out.first.unwrap_or_default();
    Ok(SpeakReport {
        sample_rate: if out.rate == 0 { SAMPLE_RATE } else { out.rate },
        frames: out.frames,
        first_audio_ms: super::cloud::ms(first),
        total_ms: super::cloud::ms(out.last.unwrap_or(first)),
    })
}

#[allow(clippy::too_many_arguments)]
fn speak_piece(
    plan: &Plan,
    piece: &str,
    deadline: Instant,
    sink: &mut dyn PcmSink,
    cancel: &CancelWatch,
    progress: &Arc<Progress>,
    clock: &dyn Clock,
    started: Instant,
    out: &mut PushOutcome,
) -> Result<(), SpeakError> {
    // Connect; a refused token is retried once with the server's clock.
    let mut socket = None;
    for attempt in 0..2 {
        if cancel.is_set() {
            return Err(SpeakError::Cancelled);
        }
        let connect_deadline = Instant::now() + plan.connect;
        match connect(plan, deadline, connect_deadline, progress, clock.now()) {
            Ok(s) => {
                socket = Some(s);
                break;
            }
            Err(ConnectFail::Forbidden(server)) => {
                if attempt == 0 {
                    if let Some(t) = server {
                        clock.adjust(t);
                    }
                    continue;
                }
                return Err(SpeakError::fail(
                    FailureKind::Auth,
                    format!("{WHAT}: the service refused the connection (HTTP 403). Check that the PC clock is right; the service may also have changed."),
                ));
            }
            Err(ConnectFail::Failed(e)) => return Err(e),
        }
    }
    let Some(mut socket) = socket else {
        return Err(stalled_before_audio(plan.first_audio));
    };
    if cancel.is_set() {
        return Err(SpeakError::Cancelled);
    }

    // The request.
    let left = deadline.saturating_duration_since(Instant::now()).max(Duration::from_millis(500));
    set_socket_timeout(&socket, left);
    let timestamp = protocol::js_date(clock.now());
    let messages = [
        protocol::speech_config_message(&timestamp),
        protocol::ssml_message(&random_id(), &timestamp, &plan.voice, &plan.rate, &plan.pitch, piece),
    ];
    for m in messages {
        if let Err(e) = socket.send(Message::Text(m)) {
            let kind = if is_timeout(&e) { FailureKind::Stalled } else { FailureKind::Network };
            return Err(SpeakError::fail(kind, format!("{WHAT}: sending the request failed ({e}).")));
        }
    }

    // The audio.
    let state = Arc::new(Mutex::new(ReadState::default()));
    let reader = AudioReader {
        socket: Mutex::new(socket),
        state: Arc::clone(&state),
        pending: Vec::new(),
        pos: 0,
        consumed: 0,
        first_deadline: deadline,
        first_audio: plan.first_audio,
        stall: plan.stall,
        cancel: cancel.clone(),
        progress: Arc::clone(progress),
    };
    let frames_before = out.frames;
    let read_state = |st: &Arc<Mutex<ReadState>>| -> (bool, usize, Option<SpeakError>) {
        let s = st.lock().unwrap_or_else(PoisonError::into_inner);
        (s.done, s.audio_bytes, s.stop.clone())
    };
    let decoder = match rodio::Decoder::new_mp3(reader) {
        Ok(d) => d,
        Err(e) => {
            let (done, bytes, stop) = read_state(&state);
            if cancel.is_set() {
                return Err(SpeakError::Cancelled);
            }
            return Err(match stop {
                Some(why) => why,
                None if done && bytes == 0 => {
                    SpeakError::fail(FailureKind::NoAudio, format!("{WHAT}: the service returned no audio for this text."))
                }
                None => SpeakError::fail(FailureKind::Other, format!("{WHAT}: the audio could not be decoded ({e}).")),
            });
        }
    };
    let rate = decoder.sample_rate().max(1);
    let channels = decoder.channels().max(1);
    let stop = || cancel.is_set();
    let mut block = Block { samples: Vec::with_capacity(PUSH_SAMPLES * 2), rate, channels };
    let flush = |block: &mut Block, sink: &mut dyn PcmSink, out: &mut PushOutcome| -> Result<(), SpeakError> {
        if block.samples.is_empty() {
            return Ok(());
        }
        let ok = push_pieces(sink, block, started, out, &stop);
        block.samples.clear();
        if ok {
            Ok(())
        } else {
            // A sink failure is this PC's, not the service's (sent untagged).
            Err(out.error.take().map_or(SpeakError::Cancelled, SpeakError::from_sink))
        }
    };
    let threshold = PUSH_SAMPLES * usize::from(channels);
    for sample in decoder {
        block.samples.push(f32::from(sample) / 32_768.0);
        if block.samples.len() >= threshold {
            flush(&mut block, sink, out)?;
        }
    }
    flush(&mut block, sink, out)?;

    let (done, bytes, stop_reason) = read_state(&state);
    if cancel.is_set() {
        return Err(SpeakError::Cancelled);
    }
    if let Some(why) = stop_reason {
        return Err(why);
    }
    if !done {
        // The decoder gave up before the service finished: corrupt audio.
        return Err(SpeakError::fail(FailureKind::Other, format!("{WHAT}: the audio could not be decoded.")));
    }
    if bytes == 0 || out.frames == frames_before {
        return Err(SpeakError::fail(FailureKind::NoAudio, format!("{WHAT}: the service returned no audio for this text.")));
    }
    Ok(())
}
