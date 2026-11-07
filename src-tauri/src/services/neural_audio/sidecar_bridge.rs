//! WebSocket clients for neural voice-conversion sidecars.
//!
//! Two protocols share this file:
//! - **v1** ([`SidecarVoiceBridge`], the voice changer's "another program over WebSocket" mode): a
//!   synchronous same-length PCM exchange per chunk, bounded by the latency budget.
//! - **v2** ([`v2`], P5-E): sequenced, pipelined frames with a per-frame deadline, rate and frame
//!   negotiation, reference-voice upload and per-frame pitch. The `sidecar_v2` voice family
//!   (`vc::families::sidecar_v2`) wraps it as a `VoiceConversionModel` inside the stream harness,
//!   so a slow or stalled sidecar never blocks the voice changer's loop. The v2 client also talks
//!   to v1 servers (compatibility mode: in-order replies, no pitch, no reference). The wire format
//!   is specified in `scripts/neural-sidecar/PROTOCOL_V2.md`.
//!
//! # Protocol v1
//! **Note:** [`SidecarVoiceBridge::exchange`] blocks the calling thread until the server responds
//! or the exchange timeout passes. It runs on the voice-changer worker, so the voice changer sets
//! the timeout to its latency budget: a stalled sidecar costs at most one budget per block (the
//! caller mutes that block, never plays the unconverted voice), never an unbounded wait.
//!
//! Handshake: after the WebSocket upgrade the bridge sends one Text JSON
//! `{"type":"sigil_vc_handshake","protocol_version":1,"sample_rate_in","channels_in","chunk_frames"}`
//! and waits (bounded) for the sidecar's first message. Any Text or Binary message counts as the
//! acknowledgement (`{"ok":true}` is the reference form); only a JSON object with `"ok": false`
//! refuses the session (its `error` / `message` / `reason` string is shown).
//!
//! Protocol v1 has no sequence numbers: every binary request gets exactly one same-length binary
//! reply, in order. After a timeout the late reply still arrives, so the bridge counts outstanding
//! replies and discards stale ones before accepting the reply to the current chunk; replies never
//! drift one block behind.

use serde::Serialize;
use std::io;
use std::net::{TcpStream, ToSocketAddrs};
use std::time::{Duration, Instant};
use tungstenite::client::IntoClientRequest;
use tungstenite::handshake::HandshakeError;
use tungstenite::{stream::MaybeTlsStream, Message, WebSocket};

/// TCP connect timeout per resolved address.
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);
/// WebSocket upgrade + Sigil handshake ack (a sidecar may load its model before acknowledging).
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);
/// Default exchange timeout when the caller sets none (the voice changer passes its budget).
const DEFAULT_EXCHANGE_TIMEOUT: Duration = Duration::from_millis(300);
/// Replies still owed after timeouts before the bridge stops sending new chunks and only drains.
const MAX_OUTSTANDING: usize = 8;
/// Wait per drain attempt while too many replies are outstanding.
const DRAIN_WAIT: Duration = Duration::from_millis(2);

/// What one message read while waiting for the handshake acknowledgement means.
#[derive(Debug, PartialEq, Eq)]
enum AckVerdict {
    /// The sidecar is ready: any first text or binary message (protocol v1 never defined the ack's
    /// format, and sidecars answer `{"ok":true}`, `{"status":"ready"}`, plain `ok`, ...).
    Accepted,
    /// The sidecar explicitly refused (`{"ok": false}`, with its `error` / `message` if any).
    Rejected(String),
    /// Control traffic (ping / pong / raw frame): keep waiting.
    Continue,
}

/// Interpret a message received in place of the handshake acknowledgement. Only a JSON object with
/// `"ok": false` refuses; anything else the sidecar says first counts as ready, as before v1 had a
/// timeout.
fn ack_verdict(msg: &Message) -> AckVerdict {
    match msg {
        Message::Text(s) => match serde_json::from_str::<serde_json::Value>(s) {
            Ok(v) if v.get("ok").and_then(serde_json::Value::as_bool) == Some(false) => {
                let why = ["error", "message", "reason"]
                    .iter()
                    .find_map(|k| v.get(*k).and_then(serde_json::Value::as_str))
                    .map(|m| format!(": {}", m.chars().take(200).collect::<String>()))
                    .unwrap_or_default();
                AckVerdict::Rejected(format!("sidecar handshake rejected (ok: false){why}"))
            }
            _ => AckVerdict::Accepted,
        },
        Message::Binary(_) => AckVerdict::Accepted,
        Message::Ping(_) | Message::Pong(_) | Message::Frame(_) => AckVerdict::Continue,
        // Handled by the caller (a close is an error).
        Message::Close(_) => AckVerdict::Continue,
    }
}

#[derive(Serialize)]
struct Handshake {
    r#type: &'static str,
    protocol_version: u32,
    sample_rate_in: u32,
    channels_in: usize,
    chunk_frames: usize,
}

fn encode_pcm_chunk(samples: &[f32]) -> Vec<u8> {
    let mut v = Vec::with_capacity(4 + samples.len() * 4);
    v.extend_from_slice(&(samples.len() as u32).to_le_bytes());
    for s in samples {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

fn decode_pcm_chunk(data: &[u8]) -> Result<Vec<f32>, String> {
    if data.len() < 4 {
        return Err("sidecar: packet too short".into());
    }
    let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    let need = 4usize.saturating_add(n.saturating_mul(4));
    if data.len() < need {
        return Err(format!(
            "sidecar: need {} bytes for {} floats, have {}",
            need,
            n,
            data.len()
        ));
    }
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let o = 4 + i * 4;
        out.push(f32::from_le_bytes([
            data[o],
            data[o + 1],
            data[o + 2],
            data[o + 3],
        ]));
    }
    Ok(out)
}

fn is_timeout(e: &tungstenite::Error) -> bool {
    // Unix reports a socket read timeout as WouldBlock, Windows as TimedOut.
    matches!(e, tungstenite::Error::Io(io) if matches!(io.kind(), io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut))
}

/// Set the read (and write) timeout of the TCP socket under the WebSocket.
fn set_socket_timeout(socket: &WebSocket<MaybeTlsStream<TcpStream>>, t: Duration) -> io::Result<()> {
    let t = Some(t.max(Duration::from_millis(1)));
    let tcp: &TcpStream = match socket.get_ref() {
        MaybeTlsStream::Plain(s) => s,
        MaybeTlsStream::NativeTls(s) => s.get_ref(),
        #[allow(unreachable_patterns)]
        _ => return Ok(()),
    };
    tcp.set_read_timeout(t)?;
    tcp.set_write_timeout(t)
}

fn connect_tcp(host: &str, port: u16) -> Result<TcpStream, String> {
    let addrs = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("sidecar: cannot resolve {host}:{port}: {e}"))?;
    let mut last = format!("sidecar: no address for {host}:{port}");
    for addr in addrs {
        match TcpStream::connect_timeout(&addr, CONNECT_TIMEOUT) {
            Ok(s) => return Ok(s),
            Err(e) => last = format!("sidecar WebSocket connect {addr}: {e}"),
        }
    }
    Err(last)
}

pub struct SidecarVoiceBridge {
    socket: WebSocket<MaybeTlsStream<TcpStream>>,
    exchange_timeout: Duration,
    /// Replies owed for chunks whose exchange timed out (plus the one in flight).
    outstanding: usize,
}

impl SidecarVoiceBridge {
    /// Connect (bounded by connect and handshake timeouts) and perform the Sigil v1 handshake.
    /// `exchange_timeout` bounds every later [`Self::exchange`].
    pub fn connect(
        ws_url: &str,
        sample_rate_in: u32,
        channels_in: usize,
        chunk_frames: usize,
        exchange_timeout: Duration,
    ) -> Result<Self, String> {
        let request = ws_url
            .trim()
            .into_client_request()
            .map_err(|e| format!("sidecar WebSocket URL: {e}"))?;
        let uri = request.uri().clone();
        let tls = match uri.scheme_str() {
            Some("ws") => false,
            Some("wss") => true,
            _ => return Err("sidecar WebSocket URL must start with ws:// or wss://".into()),
        };
        let host = uri.host().ok_or("sidecar WebSocket URL has no host")?;
        let host = host.trim_start_matches('[').trim_end_matches(']');
        let port = uri.port_u16().unwrap_or(if tls { 443 } else { 80 });
        let stream = connect_tcp(host, port)?;
        let _ = stream.set_nodelay(true);
        stream
            .set_read_timeout(Some(HANDSHAKE_TIMEOUT))
            .and_then(|_| stream.set_write_timeout(Some(HANDSHAKE_TIMEOUT)))
            .map_err(|e| format!("sidecar socket: {e}"))?;
        let (socket, _) = tungstenite::client_tls_with_config(request, stream, None, None).map_err(|e| match e {
            HandshakeError::Failure(f) => format!("sidecar WebSocket connect: {f}"),
            HandshakeError::Interrupted(_) => "sidecar WebSocket connect: the server did not answer in time".to_string(),
        })?;
        let mut socket = socket;
        let handshake = Handshake {
            r#type: "sigil_vc_handshake",
            protocol_version: 1,
            sample_rate_in,
            channels_in,
            chunk_frames,
        };
        let text = serde_json::to_string(&handshake).map_err(|e| e.to_string())?;
        socket
            .send(Message::Text(text))
            .map_err(|e| format!("sidecar handshake: {}", e))?;
        loop {
            match socket.read() {
                Ok(Message::Close(_)) => return Err("sidecar closed during handshake".into()),
                Ok(msg) => match ack_verdict(&msg) {
                    AckVerdict::Accepted => break,
                    AckVerdict::Rejected(why) => return Err(why),
                    AckVerdict::Continue => continue,
                },
                Err(e) if is_timeout(&e) => return Err("sidecar handshake: no acknowledgement in time".into()),
                Err(e) => return Err(format!("sidecar handshake read: {}", e)),
            }
        }
        let exchange_timeout = if exchange_timeout.is_zero() { DEFAULT_EXCHANGE_TIMEOUT } else { exchange_timeout };
        set_socket_timeout(&socket, exchange_timeout).map_err(|e| format!("sidecar socket: {e}"))?;
        Ok(Self {
            socket,
            exchange_timeout,
            outstanding: 0,
        })
    }

    /// Change the exchange timeout (the voice changer's latency budget).
    pub fn set_exchange_timeout(&mut self, timeout: Duration) {
        if !timeout.is_zero() {
            self.exchange_timeout = timeout;
        }
    }

    /// Replies still owed from timed-out exchanges.
    #[cfg(test)]
    pub fn outstanding_replies(&self) -> usize {
        self.outstanding
    }

    /// Send one PCM chunk and wait (at most the exchange timeout) for the converted chunk (same
    /// layout). Stale replies to earlier timed-out chunks are read and discarded first. While
    /// [`MAX_OUTSTANDING`] replies are owed nothing new is sent: the call only drains (briefly)
    /// and fails, so a hung sidecar is not flooded and the caller keeps real time.
    pub fn exchange(&mut self, samples: &[f32]) -> Result<Vec<f32>, String> {
        if self.outstanding >= MAX_OUTSTANDING {
            self.drain(DRAIN_WAIT)?;
            if self.outstanding >= MAX_OUTSTANDING {
                return Err(format!("sidecar is not answering ({} replies overdue)", self.outstanding));
            }
        }
        let deadline = Instant::now() + self.exchange_timeout;
        set_socket_timeout(&self.socket, self.exchange_timeout).map_err(|e| format!("sidecar socket: {e}"))?;
        let payload = encode_pcm_chunk(samples);
        match self.socket.send(Message::Binary(payload)) {
            Ok(()) => {}
            // The frame stays queued in tungstenite and goes out with the next write.
            Err(e) if is_timeout(&e) => {}
            Err(e) => return Err(format!("sidecar send: {}", e)),
        }
        self.outstanding += 1;
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(self.timeout_error());
            }
            set_socket_timeout(&self.socket, deadline - now).map_err(|e| format!("sidecar socket: {e}"))?;
            match self.socket.read() {
                Ok(Message::Binary(b)) => {
                    self.outstanding = self.outstanding.saturating_sub(1);
                    if self.outstanding == 0 {
                        return decode_pcm_chunk(&b);
                    }
                    // A late reply to an earlier chunk: that block was already muted, drop it.
                }
                Ok(Message::Close(_)) => return Err("sidecar closed".into()),
                Ok(Message::Ping(_)) | Ok(Message::Pong(_)) | Ok(Message::Frame(_)) => {}
                Ok(Message::Text(_)) => return Err("sidecar: unexpected message type".into()),
                Err(e) if is_timeout(&e) => return Err(self.timeout_error()),
                Err(e) => return Err(format!("sidecar read: {}", e)),
            }
        }
    }

    /// Read and discard overdue replies for up to `wait` per read.
    fn drain(&mut self, wait: Duration) -> Result<(), String> {
        set_socket_timeout(&self.socket, wait).map_err(|e| format!("sidecar socket: {e}"))?;
        while self.outstanding > 0 {
            match self.socket.read() {
                Ok(Message::Binary(_)) => self.outstanding -= 1,
                Ok(Message::Close(_)) => return Err("sidecar closed".into()),
                Ok(_) => {}
                Err(e) if is_timeout(&e) => break,
                Err(e) => return Err(format!("sidecar read: {}", e)),
            }
        }
        Ok(())
    }

    fn timeout_error(&self) -> String {
        format!(
            "sidecar did not answer within {} ms (the latency budget); this block is muted",
            self.exchange_timeout.as_millis()
        )
    }
}

/// Sidecar protocol v2 (P5-E): the client side. See `scripts/neural-sidecar/PROTOCOL_V2.md` for the
/// wire format; the summary:
///
/// 1. WebSocket upgrade to a loopback `ws://` URL (`Authorization: Bearer <key>` when the helper
///    was started with a key).
/// 2. Client hello (Text): `{"type":"sigil_vc_handshake","protocol_version":2, ...}`; it also
///    carries the v1 fields (`sample_rate_in`, `channels_in`, `chunk_frames`), so a v1 server
///    answers it with its plain acknowledgement and the client falls back to v1 framing.
/// 3. Server welcome (Text): `{"type":"sigil_vc_welcome","protocol_version":2,"sample_rate_in",
///    "sample_rate_out","frame_samples","latency_ms","delay_samples","reference_needed", ...}`:
///    the server picks the input rate from the client's list, its output rate and its frame.
/// 4. Optional reference upload: a Text `reference` message, then Binary frames of kind
///    [`KIND_REFERENCE`] (the last one flagged [`FLAG_LAST`]); the server answers
///    `{"type":"reference_ready"}` once it has prepared the voice.
/// 5. Streaming: Binary [`KIND_AUDIO_IN`] frames (20-byte header: `SGV2`, kind, flags, sequence
///    number, sample count, pitch in semitones; then f32 LE PCM), answered by [`KIND_AUDIO_OUT`]
///    frames with the same sequence number, possibly several frames later (pipelined). Text
///    `params` and `cancel` messages go the other way; the server reports `dropped` and `error`.
///
/// The client runs the socket on its own thread with an event-driven loop (a current-thread tokio
/// runtime over the same tungstenite state machine the handshake used), so [`SidecarClient`]'s
/// methods never block: frames go into a bounded queue and replies come back through a channel
/// the caller polls. A lost connection is reported ([`Event::Lost`]) and re-established in the
/// background with backoff (handshake and reference upload again, [`Event::Restored`]). The first
/// connection can be called off ([`ClientConfig::cancel`]): a voice nobody wants any more never
/// waits out a slow handshake.
pub mod v2 {
    use std::collections::VecDeque;
    use std::io::{self, Read, Write};
    use std::net::{IpAddr, Shutdown, TcpStream};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{mpsc as std_mpsc, Arc, Mutex, PoisonError};
    use std::time::{Duration, Instant};

    use serde::Serialize;
    use serde_json::{json, Map, Value};
    use tokio::sync::mpsc;
    use tungstenite::client::IntoClientRequest;
    use tungstenite::handshake::HandshakeError;
    use tungstenite::http::header::AUTHORIZATION;
    use tungstenite::http::HeaderValue;
    use tungstenite::protocol::WebSocketConfig;
    use tungstenite::{Message, WebSocket};

    use super::{ack_verdict, connect_tcp, decode_pcm_chunk, encode_pcm_chunk, is_timeout, AckVerdict};

    /// Protocol version of this client.
    pub const PROTOCOL_VERSION: u32 = 2;
    /// `type` of the client hello (the v1 handshake's type, so v1 servers accept it).
    pub const HELLO_TYPE: &str = "sigil_vc_handshake";
    /// `type` of a v2 server's welcome.
    pub const WELCOME_TYPE: &str = "sigil_vc_welcome";
    /// First four bytes of every v2 binary frame.
    pub const FRAME_MAGIC: [u8; 4] = *b"SGV2";
    /// Bytes before the PCM of a v2 binary frame.
    pub const HEADER_LEN: usize = 20;
    /// Frame kind: microphone audio, client to server.
    pub const KIND_AUDIO_IN: u8 = 1;
    /// Frame kind: converted audio, server to client.
    pub const KIND_AUDIO_OUT: u8 = 2;
    /// Frame kind: a chunk of the reference voice, client to server (handshake only).
    pub const KIND_REFERENCE: u8 = 3;
    /// Audio-in flag: this frame does not continue the previous one (a capture gap, a dropped
    /// frame, a reconnect); the server may reset its streaming state.
    pub const FLAG_DISCONTINUITY: u8 = 0x01;
    /// Reference flag: the last chunk of the reference voice.
    pub const FLAG_LAST: u8 = 0x02;
    /// Largest sample count one frame may declare (protects the decoder's allocation).
    pub const MAX_FRAME_SAMPLES: usize = 1 << 20;
    /// Samples per reference chunk (256 KiB: under the 1 MiB default message cap of Python's
    /// `websockets`).
    pub const REFERENCE_CHUNK_SAMPLES: usize = 1 << 16;
    /// Rates a server may pick or answer at.
    const MIN_RATE: u32 = 8_000;
    const MAX_RATE: u32 = 192_000;
    /// Frame durations a server may pick.
    const MIN_FRAME_MS: f64 = 10.0;
    const MAX_FRAME_MS: f64 = 2_000.0;
    /// Longest algorithmic delay a server may declare.
    const MAX_DELAY_SECONDS: u64 = 10;
    /// Reference length sent when the server names no limit.
    const DEFAULT_MAX_REFERENCE_SECONDS: f64 = 30.0;
    /// tungstenite's write buffer cap: a server that stops reading costs at most this much memory
    /// before frames are dropped (the caller's in-flight cap normally stops long before).
    const MAX_WRITE_BUFFER: usize = 8 << 20;
    /// Reconnect backoff bounds.
    const RECONNECT_MIN: Duration = Duration::from_millis(250);
    const RECONNECT_MAX: Duration = Duration::from_secs(5);
    /// How often a waiting I/O thread checks for shutdown and drains queued frames.
    const IDLE_POLL: Duration = Duration::from_millis(50);
    /// Longest server message quoted in an error.
    const MAX_QUOTE: usize = 200;
    /// How often [`SidecarClient::connect`] looks at [`ClientConfig::cancel`].
    const CANCEL_POLL: Duration = Duration::from_millis(50);
    /// [`SidecarClient::connect`]'s error when [`ClientConfig::cancel`] was set.
    pub const CONNECT_CANCELLED: &str = "the sidecar connection was cancelled";
    /// `dropped` reason of a single-stream server: another voice's stream holds its model.
    pub const DROP_SUPERSEDED: &str = "superseded";

    // -----------------------------------------------------------------------------------------
    // Wire format
    // -----------------------------------------------------------------------------------------

    /// Header of a v2 binary frame.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct FrameHeader {
        pub kind: u8,
        pub flags: u8,
        pub seq: u32,
        /// Audio-in: the voice's pitch shift in semitones; 0 elsewhere.
        pub pitch: f32,
    }

    /// Encode one frame: `SGV2`, kind u8, flags u8, reserved u16 (0), seq u32, sample count u32,
    /// pitch f32, then the samples (all little-endian).
    pub fn encode_frame(header: &FrameHeader, pcm: &[f32]) -> Vec<u8> {
        let mut v = Vec::with_capacity(HEADER_LEN + pcm.len() * 4);
        v.extend_from_slice(&FRAME_MAGIC);
        v.push(header.kind);
        v.push(header.flags);
        v.extend_from_slice(&0u16.to_le_bytes());
        v.extend_from_slice(&header.seq.to_le_bytes());
        v.extend_from_slice(&(pcm.len() as u32).to_le_bytes());
        v.extend_from_slice(&header.pitch.to_le_bytes());
        for s in pcm {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    /// Decode one frame (the length must match the declared sample count exactly).
    pub fn decode_frame(data: &[u8]) -> Result<(FrameHeader, Vec<f32>), String> {
        if data.len() < HEADER_LEN {
            return Err(format!("sidecar frame too short ({} bytes)", data.len()));
        }
        if data[..4] != FRAME_MAGIC {
            return Err("sidecar frame without the SGV2 header (is it a protocol v1 server?)".into());
        }
        let u32_at = |o: usize| u32::from_le_bytes([data[o], data[o + 1], data[o + 2], data[o + 3]]);
        let n = u32_at(12) as usize;
        if n > MAX_FRAME_SAMPLES {
            return Err(format!("sidecar frame declares {n} samples (at most {MAX_FRAME_SAMPLES})"));
        }
        if data.len() != HEADER_LEN + n * 4 {
            return Err(format!(
                "sidecar frame declares {n} samples but carries {} bytes of audio",
                data.len() - HEADER_LEN
            ));
        }
        let header = FrameHeader {
            kind: data[4],
            flags: data[5],
            seq: u32_at(8),
            pitch: f32::from_le_bytes([data[16], data[17], data[18], data[19]]),
        };
        let pcm = data[HEADER_LEN..]
            .as_chunks::<4>().0.iter()
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok((header, pcm))
    }

    // -----------------------------------------------------------------------------------------
    // Configuration and session
    // -----------------------------------------------------------------------------------------

    /// Where to connect (the key is never logged).
    #[derive(Clone)]
    pub struct Endpoint {
        pub url: String,
        pub api_key: Option<String>,
    }

    impl std::fmt::Debug for Endpoint {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Endpoint")
                .field("url", &self.url)
                .field("api_key", &self.api_key.as_ref().map(|_| "<redacted>"))
                .finish()
        }
    }

    /// Resolves the endpoint before every connection attempt (a restarted helper may listen on a
    /// new port).
    pub type EndpointFn = Arc<dyn Fn() -> Result<Endpoint, String> + Send + Sync>;

    /// What the server learns about the reference voice (the clip itself follows as frames).
    #[derive(Clone, Debug, Default, PartialEq, Serialize)]
    pub struct ReferenceMeta {
        pub voice_id: String,
        pub name: String,
        /// sha256 of the clip file: servers may cache prepared voices by it.
        pub sha256: String,
        pub duration_ms: u64,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub transcript: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub language: Option<String>,
    }

    /// Decodes the reference clip as mono f32 at the rate the server asks for.
    pub type ReferenceLoader = Arc<dyn Fn(u32) -> Result<Vec<f32>, String> + Send + Sync>;

    /// The reference voice to upload.
    #[derive(Clone)]
    pub struct ReferenceSource {
        pub meta: ReferenceMeta,
        pub load: ReferenceLoader,
    }

    /// How to connect.
    #[derive(Clone)]
    pub struct ClientConfig {
        /// For logs and the I/O thread's messages (e.g. `xvc`).
        pub label: String,
        pub endpoint: EndpointFn,
        /// Input rates the client can feed, preferred first (the first is the v1 rate).
        pub input_rates: Vec<u32>,
        /// Proposed frame duration; the server may choose another.
        pub frame_ms: f64,
        /// The caller's per-frame deadline (sent to the server so it can drop stale frames).
        pub frame_timeout: Duration,
        /// Frames the caller keeps in flight at most (sent to the server).
        pub max_in_flight: usize,
        /// Initial voice parameters (`formant_semitones`, `speaker_id`, ...).
        pub params: Map<String, Value>,
        pub reference: Option<ReferenceSource>,
        /// Accept a protocol v1 server (no pitch, no reference).
        pub allow_v1: bool,
        /// Reconnect in the background after the connection is lost.
        pub reconnect: bool,
        /// Upgrade and welcome (the server prepares a session).
        pub handshake_timeout: Duration,
        /// Reference upload and preparation.
        pub reference_timeout: Duration,
        /// Set while [`SidecarClient::connect`] runs when nobody wants the voice any more: it
        /// returns [`CONNECT_CANCELLED`] within [`CANCEL_POLL`] and the handshake's socket is shut
        /// (the first connection only; reconnects of a loaded voice do not look at it).
        pub cancel: Option<Arc<AtomicBool>>,
    }

    impl ClientConfig {
        pub fn new(label: impl Into<String>, endpoint: EndpointFn) -> ClientConfig {
            ClientConfig {
                label: label.into(),
                endpoint,
                input_rates: vec![16_000, 22_050, 24_000, 44_100, 48_000],
                frame_ms: 100.0,
                frame_timeout: Duration::from_millis(300),
                max_in_flight: 8,
                params: Map::new(),
                reference: None,
                allow_v1: true,
                reconnect: true,
                handshake_timeout: Duration::from_secs(30),
                reference_timeout: Duration::from_secs(120),
                cancel: None,
            }
        }

        fn cancelled(&self) -> bool {
            self.cancel.as_ref().is_some_and(|c| c.load(Ordering::Relaxed))
        }

        /// A fixed endpoint.
        pub fn fixed(url: impl Into<String>, api_key: Option<String>) -> EndpointFn {
            let endpoint = Endpoint { url: url.into(), api_key };
            Arc::new(move || Ok(endpoint.clone()))
        }

        fn validate(&self) -> Result<(), String> {
            if self.input_rates.is_empty() || self.input_rates.iter().any(|r| !(MIN_RATE..=MAX_RATE).contains(r)) {
                return Err(format!("sidecar input rates must be {MIN_RATE}-{MAX_RATE} Hz"));
            }
            if !(self.frame_ms.is_finite() && (MIN_FRAME_MS..=MAX_FRAME_MS).contains(&self.frame_ms)) {
                return Err(format!("sidecar frame must be {MIN_FRAME_MS}-{MAX_FRAME_MS} ms"));
            }
            if self.max_in_flight == 0 {
                return Err("sidecar max_in_flight must be at least 1".into());
            }
            Ok(())
        }
    }

    /// What client and server agreed on.
    #[derive(Clone, Debug, PartialEq)]
    pub struct Session {
        /// 2, or 1 for a v1 server (in-order same-length replies, no pitch, no reference).
        pub protocol: u32,
        pub input_rate: u32,
        pub output_rate: u32,
        /// Samples per frame at the input rate.
        pub frame_in: usize,
        /// Samples per frame at the output rate (`frame_in * output_rate / input_rate`, exact).
        pub frame_out: usize,
        /// The server's estimate of its compute time per frame.
        pub latency_ms: Option<f64>,
        /// The server's algorithmic delay (output samples): a reply to frame n holds the audio that
        /// ends this much before frame n's end.
        pub delay_out: usize,
        pub model: String,
        /// Device the server runs on (`cuda`, `dml`, `cpu`, ...), when it says.
        pub device: Option<String>,
        /// GPU memory the server holds, when it says.
        pub vram_mb: Option<f64>,
        /// Parameters the server understands (`formant_semitones`, `speaker_id`, ...).
        pub params: Vec<String>,
        /// The server converts toward the client's reference voice (uploaded, or prepared from an
        /// earlier upload of the same clip).
        pub reference_used: bool,
        /// The client uploads the reference during the handshake (the server has not cached it).
        pub reference_upload: bool,
        /// The server applies the per-frame pitch shift (welcome `"pitch": true`).
        pub pitch_supported: bool,
        /// The server cannot work without a reference voice.
        pub reference_required: bool,
        /// Rate of the reference upload.
        pub reference_rate: u32,
        /// Longest reference the server takes.
        pub max_reference_seconds: f64,
        /// Frames the server queues at most, when its welcome says (`None`: no limit of its own;
        /// the caller sizes its window from the negotiated frame, not from what it offered).
        pub max_in_flight: Option<usize>,
    }

    impl Session {
        /// The server applies the per-frame pitch shift (protocol v2 servers that say so).
        pub fn pitch(&self) -> bool {
            self.protocol >= 2 && self.pitch_supported
        }

        /// Frame duration in milliseconds.
        pub fn frame_ms(&self) -> f64 {
            self.frame_in as f64 * 1000.0 / f64::from(self.input_rate.max(1))
        }

        /// Why a reconnected session cannot continue this one's stream, if it cannot.
        fn incompatible_with(&self, next: &Session) -> Option<String> {
            (self.protocol != next.protocol
                || self.input_rate != next.input_rate
                || self.output_rate != next.output_rate
                || self.frame_in != next.frame_in)
                .then(|| {
                    format!(
                        "the sidecar restarted with other stream settings (v{} {}->{} Hz, {} samples per frame; was v{} {}->{} Hz, {}); load the voice again",
                        next.protocol, next.input_rate, next.output_rate, next.frame_in, self.protocol, self.input_rate, self.output_rate, self.frame_in
                    )
                })
        }
    }

    // -----------------------------------------------------------------------------------------
    // Negotiation (pure, tested without a socket)
    // -----------------------------------------------------------------------------------------

    /// What the client offered in its hello.
    #[derive(Clone, Debug)]
    pub(crate) struct Offer {
        pub rates: Vec<u32>,
        pub frame_ms: f64,
        pub has_reference: bool,
        pub allow_v1: bool,
    }

    impl Offer {
        fn of(config: &ClientConfig) -> Offer {
            Offer {
                rates: config.input_rates.clone(),
                frame_ms: config.frame_ms,
                has_reference: config.reference.is_some(),
                allow_v1: config.allow_v1,
            }
        }

        /// The v1 chunk (samples at the preferred rate).
        fn v1_chunk(&self) -> usize {
            ((self.frame_ms * f64::from(self.rates[0]) / 1000.0).round() as usize).max(1)
        }
    }

    /// What one message read while waiting for the welcome means.
    #[derive(Clone, Debug, PartialEq)]
    pub(crate) enum Reply {
        Ready(Session),
        Refused(String),
        Wait,
    }

    fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a.max(1)
    }

    /// The frame (input samples) nearest `frame_ms` whose output length is a whole number of
    /// samples (a multiple of `rate_in / gcd(rate_in, rate_out)`).
    pub(crate) fn whole_frame(frame_ms: f64, rate_in: u32, rate_out: u32) -> usize {
        let unit = u64::from(rate_in) / gcd(u64::from(rate_in), u64::from(rate_out));
        let raw = frame_ms * f64::from(rate_in) / 1000.0;
        let k = (raw / unit as f64).round().max(1.0) as u64;
        (k * unit) as usize
    }

    /// The first `error` / `message` / `reason` string of a server message, shortened.
    fn quoted(v: &Value) -> String {
        ["error", "message", "reason"]
            .iter()
            .find_map(|k| v.get(*k).and_then(Value::as_str))
            .map(|m| m.chars().take(MAX_QUOTE).collect::<String>())
            .unwrap_or_default()
    }

    /// Interpret the server's answer to the hello.
    pub(crate) fn negotiate(offer: &Offer, msg: &Message) -> Reply {
        let v1 = |offer: &Offer| {
            let rate = offer.rates[0];
            let frame = offer.v1_chunk();
            Session {
                protocol: 1,
                input_rate: rate,
                output_rate: rate,
                frame_in: frame,
                frame_out: frame,
                latency_ms: None,
                delay_out: 0,
                model: String::new(),
                device: None,
                vram_mb: None,
                params: Vec::new(),
                reference_used: false,
                reference_upload: false,
                pitch_supported: false,
                reference_required: false,
                reference_rate: rate,
                max_reference_seconds: 0.0,
                max_in_flight: None,
            }
        };
        let text = match msg {
            Message::Text(t) => t,
            other => {
                return match ack_verdict(other) {
                    AckVerdict::Accepted if offer.allow_v1 => Reply::Ready(v1(offer)),
                    AckVerdict::Accepted => Reply::Refused(v1_refusal()),
                    AckVerdict::Rejected(why) => Reply::Refused(why),
                    AckVerdict::Continue => Reply::Wait,
                }
            }
        };
        let value: Value = serde_json::from_str(text).unwrap_or(Value::Null);
        if value.get("ok").and_then(Value::as_bool) == Some(false) {
            let why = quoted(&value);
            return Reply::Refused(if why.is_empty() {
                "the sidecar refused the session".into()
            } else {
                format!("the sidecar refused the session: {why}")
            });
        }
        let is_v2 = value.get("type").and_then(Value::as_str) == Some(WELCOME_TYPE)
            && value.get("protocol_version").and_then(Value::as_u64) == Some(u64::from(PROTOCOL_VERSION));
        if !is_v2 {
            // Any other first message is a v1 acknowledgement (`{"ok":true}`, `ok`, ...).
            return if offer.allow_v1 { Reply::Ready(v1(offer)) } else { Reply::Refused(v1_refusal()) };
        }
        match welcome_session(offer, &value) {
            Ok(s) => Reply::Ready(s),
            Err(e) => Reply::Refused(e),
        }
    }

    fn v1_refusal() -> String {
        "the sidecar speaks protocol v1 only; this voice needs protocol v2 (pitch and the reference voice)".into()
    }

    fn rate_field(v: &Value, key: &str) -> Result<Option<u32>, String> {
        match v.get(key) {
            None | Some(Value::Null) => Ok(None),
            Some(x) => x
                .as_u64()
                .and_then(|r| u32::try_from(r).ok())
                .filter(|r| (MIN_RATE..=MAX_RATE).contains(r))
                .map(Some)
                .ok_or_else(|| format!("the sidecar's {key} must be {MIN_RATE}-{MAX_RATE} Hz (got {x})")),
        }
    }

    /// The session of a v2 welcome, checked against the offer.
    fn welcome_session(offer: &Offer, v: &Value) -> Result<Session, String> {
        let input_rate = rate_field(v, "sample_rate_in")?.ok_or("the sidecar's welcome has no sample_rate_in")?;
        if !offer.rates.contains(&input_rate) {
            return Err(format!(
                "the sidecar wants {input_rate} Hz input; Sigil offered {}",
                offer.rates.iter().map(u32::to_string).collect::<Vec<_>>().join(", ")
            ));
        }
        let output_rate = rate_field(v, "sample_rate_out")?.unwrap_or(input_rate);
        let frame_in = match v.get("frame_samples") {
            None | Some(Value::Null) => whole_frame(offer.frame_ms, input_rate, output_rate),
            Some(x) => x
                .as_u64()
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| format!("the sidecar's frame_samples must be a whole number (got {x})"))?,
        };
        let frame_ms = frame_in as f64 * 1000.0 / f64::from(input_rate);
        if !(MIN_FRAME_MS..=MAX_FRAME_MS).contains(&frame_ms) {
            return Err(format!("the sidecar's frame of {frame_in} samples ({frame_ms:.1} ms) is outside {MIN_FRAME_MS}-{MAX_FRAME_MS} ms"));
        }
        let scaled = frame_in as u64 * u64::from(output_rate);
        if !scaled.is_multiple_of(u64::from(input_rate)) {
            return Err(format!(
                "the sidecar's frame of {frame_in} samples at {input_rate} Hz is not a whole number of samples at {output_rate} Hz"
            ));
        }
        let frame_out = (scaled / u64::from(input_rate)) as usize;
        let latency_ms = v.get("latency_ms").and_then(Value::as_f64).filter(|l| l.is_finite() && (0.0..=60_000.0).contains(l));
        let delay_out = v.get("delay_samples").and_then(Value::as_u64).unwrap_or(0);
        if delay_out > MAX_DELAY_SECONDS * u64::from(output_rate) {
            return Err(format!("the sidecar declares a delay of {delay_out} samples (over {MAX_DELAY_SECONDS} s)"));
        }
        let reference_required = v.get("reference_required").and_then(Value::as_bool).unwrap_or(false);
        if reference_required && !offer.has_reference {
            return Err("this sidecar converts toward a reference voice: choose one from the voice library".into());
        }
        let reference_upload = offer.has_reference && v.get("reference_needed").and_then(Value::as_bool).unwrap_or(reference_required);
        let reference_cached = offer.has_reference && v.get("reference_cached").and_then(Value::as_bool).unwrap_or(false);
        let reference_rate = rate_field(v, "reference_sample_rate")?.unwrap_or(input_rate);
        let max_reference_seconds = v
            .get("max_reference_seconds")
            .and_then(Value::as_f64)
            .filter(|s| s.is_finite() && *s > 0.0)
            .map_or(DEFAULT_MAX_REFERENCE_SECONDS, |s| s.min(120.0));
        let strings = |key: &str| -> Vec<String> {
            v.get(key)
                .and_then(Value::as_array)
                .map(|a| a.iter().filter_map(Value::as_str).take(64).map(|s| s.chars().take(64).collect()).collect())
                .unwrap_or_default()
        };
        let text = |key: &str| v.get(key).and_then(Value::as_str).map(|s| s.chars().take(MAX_QUOTE).collect::<String>());
        let max_in_flight = v.get("max_in_flight").and_then(Value::as_u64).map(|m| (m as usize).clamp(1, 1024));
        Ok(Session {
            protocol: PROTOCOL_VERSION,
            input_rate,
            output_rate,
            frame_in,
            frame_out,
            latency_ms,
            delay_out: delay_out as usize,
            model: text("model").unwrap_or_default(),
            device: text("device").map(|d| d.to_ascii_lowercase()),
            vram_mb: v.get("vram_mb").and_then(Value::as_f64).filter(|m| m.is_finite() && *m >= 0.0),
            params: strings("params"),
            reference_used: reference_upload || reference_cached,
            reference_upload,
            pitch_supported: v.get("pitch").and_then(Value::as_bool).unwrap_or(false),
            reference_required,
            reference_rate,
            max_reference_seconds,
            max_in_flight,
        })
    }

    /// The client hello (v2 fields plus the v1 ones).
    fn hello(config: &ClientConfig, offer: &Offer) -> String {
        json!({
            "type": HELLO_TYPE,
            "protocol_version": PROTOCOL_VERSION,
            "sample_rate_in": offer.rates[0],
            "channels_in": 1,
            "chunk_frames": offer.v1_chunk(),
            "client": concat!("sigil/", env!("CARGO_PKG_VERSION")),
            "sample_rates": offer.rates,
            "frame_ms": config.frame_ms,
            "frame_timeout_ms": config.frame_timeout.as_millis() as u64,
            "max_in_flight": config.max_in_flight,
            "params": config.params,
            "reference": config.reference.as_ref().map(|r| &r.meta),
            "features": ["pitch", "params", "cancel", "reference", "discontinuity"],
        })
        .to_string()
    }

    /// Hosts a v2 sidecar may live on (this PC only).
    pub fn is_loopback_host(host: &str) -> bool {
        let h = host.trim().trim_start_matches('[').trim_end_matches(']');
        h.eq_ignore_ascii_case("localhost") || h.parse::<IpAddr>().is_ok_and(|ip| ip.is_loopback())
    }

    // -----------------------------------------------------------------------------------------
    // Socket
    // -----------------------------------------------------------------------------------------

    /// The TCP stream under the WebSocket: blocking with timeouts for the handshake, then
    /// non-blocking (tokio readiness) for streaming. tungstenite's buffers survive the switch.
    pub(crate) struct IoStream {
        inner: IoInner,
    }

    enum IoInner {
        Blocking(TcpStream),
        Async(Arc<tokio::net::TcpStream>),
        Detached,
    }

    impl IoStream {
        fn blocking(stream: TcpStream) -> IoStream {
            IoStream { inner: IoInner::Blocking(stream) }
        }

        /// Read and write timeout of the blocking phase.
        fn set_timeout(&mut self, timeout: Duration) -> Result<(), String> {
            if let IoInner::Blocking(s) = &self.inner {
                let t = Some(timeout.max(Duration::from_millis(1)));
                s.set_read_timeout(t)
                    .and_then(|_| s.set_write_timeout(t))
                    .map_err(|e| format!("sidecar socket: {e}"))?;
            }
            Ok(())
        }

        /// Switch to non-blocking I/O on the current tokio runtime (call inside `Runtime::enter`).
        fn make_async(&mut self) -> io::Result<Arc<tokio::net::TcpStream>> {
            match std::mem::replace(&mut self.inner, IoInner::Detached) {
                IoInner::Blocking(s) => {
                    s.set_read_timeout(None)?;
                    s.set_write_timeout(None)?;
                    s.set_nonblocking(true)?;
                    let t = Arc::new(tokio::net::TcpStream::from_std(s)?);
                    self.inner = IoInner::Async(t.clone());
                    Ok(t)
                }
                IoInner::Async(t) => {
                    self.inner = IoInner::Async(t.clone());
                    Ok(t)
                }
                IoInner::Detached => Err(io::ErrorKind::NotConnected.into()),
            }
        }
    }

    impl Read for IoStream {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            match &mut self.inner {
                IoInner::Blocking(s) => s.read(buf),
                IoInner::Async(s) => s.try_read(buf),
                IoInner::Detached => Err(io::ErrorKind::NotConnected.into()),
            }
        }
    }

    impl Write for IoStream {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            match &mut self.inner {
                IoInner::Blocking(s) => s.write(buf),
                IoInner::Async(s) => s.try_write(buf),
                IoInner::Detached => Err(io::ErrorKind::NotConnected.into()),
            }
        }

        fn flush(&mut self) -> io::Result<()> {
            match &mut self.inner {
                IoInner::Blocking(s) => s.flush(),
                _ => Ok(()),
            }
        }
    }

    fn ws_config() -> WebSocketConfig {
        WebSocketConfig {
            // Write every message at once (the loop flushes after each batch).
            write_buffer_size: 0,
            max_write_buffer_size: MAX_WRITE_BUFFER,
            max_message_size: Some(64 << 20),
            max_frame_size: Some(16 << 20),
            ..WebSocketConfig::default()
        }
    }

    fn would_block(e: &tungstenite::Error) -> bool {
        matches!(e, tungstenite::Error::Io(io) if io.kind() == io::ErrorKind::WouldBlock)
    }

    /// Read one application message before `deadline` (blocking phase).
    fn read_blocking(ws: &mut WebSocket<IoStream>, deadline: Instant, late: &str) -> Result<Message, String> {
        loop {
            let now = Instant::now();
            if now >= deadline {
                return Err(late.to_string());
            }
            ws.get_mut().set_timeout(deadline - now)?;
            match ws.read() {
                Ok(Message::Close(_)) => return Err("the sidecar closed the connection during the handshake".into()),
                Ok(Message::Ping(_)) | Ok(Message::Pong(_)) | Ok(Message::Frame(_)) => continue,
                Ok(m) => return Ok(m),
                Err(e) if is_timeout(&e) => return Err(late.to_string()),
                Err(e) => return Err(format!("sidecar handshake read: {e}")),
            }
        }
    }

    /// Send one message before `deadline` (blocking phase).
    fn send_blocking(ws: &mut WebSocket<IoStream>, msg: Message, deadline: Instant) -> Result<(), String> {
        let now = Instant::now();
        if now >= deadline {
            return Err("the sidecar handshake took too long".into());
        }
        ws.get_mut().set_timeout(deadline - now)?;
        ws.send(msg).map_err(|e| {
            if is_timeout(&e) {
                "the sidecar stopped reading during the handshake".to_string()
            } else {
                format!("sidecar handshake write: {e}")
            }
        })
    }

    /// The reference voice at the rate of the last session (reloaded when a server asks for
    /// another rate).
    #[derive(Default)]
    struct ReferenceCache {
        pcm: Option<(u32, Arc<Vec<f32>>)>,
    }

    impl ReferenceCache {
        fn get(&mut self, source: &ReferenceSource, rate: u32) -> Result<Arc<Vec<f32>>, String> {
            if let Some((r, pcm)) = &self.pcm {
                if *r == rate {
                    return Ok(pcm.clone());
                }
            }
            let pcm = Arc::new((source.load)(rate)?);
            self.pcm = Some((rate, pcm.clone()));
            Ok(pcm)
        }
    }

    /// Upload the reference voice and wait until the server has prepared it.
    fn upload_reference(
        ws: &mut WebSocket<IoStream>,
        config: &ClientConfig,
        session: &Session,
        cache: &mut ReferenceCache,
    ) -> Result<(), String> {
        let Some(source) = &config.reference else {
            return Ok(());
        };
        let deadline = Instant::now() + config.reference_timeout;
        let rate = session.reference_rate;
        let all = cache.get(source, rate)?;
        let max = (session.max_reference_seconds * f64::from(rate)) as usize;
        let pcm = &all[..all.len().min(max)];
        let meta = json!({
            "type": "reference",
            "voice_id": source.meta.voice_id,
            "name": source.meta.name,
            "sha256": source.meta.sha256,
            "transcript": source.meta.transcript,
            "language": source.meta.language,
            "sample_rate": rate,
            "samples": pcm.len(),
        });
        send_blocking(ws, Message::Text(meta.to_string()), deadline)?;
        let chunks: Vec<&[f32]> = if pcm.is_empty() {
            vec![&[]]
        } else {
            pcm.chunks(REFERENCE_CHUNK_SAMPLES).collect()
        };
        let last = chunks.len() - 1;
        for (i, chunk) in chunks.into_iter().enumerate() {
            let header = FrameHeader {
                kind: KIND_REFERENCE,
                flags: if i == last { FLAG_LAST } else { 0 },
                seq: i as u32,
                pitch: 0.0,
            };
            send_blocking(ws, Message::Binary(encode_frame(&header, chunk)), deadline)?;
        }
        loop {
            let msg = read_blocking(ws, deadline, "the sidecar did not finish preparing the reference voice in time")?;
            let Message::Text(t) = msg else { continue };
            let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
            match v.get("type").and_then(Value::as_str) {
                Some("reference_ready") => return Ok(()),
                Some("error") => return Err(format!("the sidecar refused the reference voice: {}", quoted(&v))),
                _ if v.get("ok").and_then(Value::as_bool) == Some(false) => {
                    return Err(format!("the sidecar refused the reference voice: {}", quoted(&v)))
                }
                _ => {}
            }
        }
    }

    /// The socket of the first handshake, shared with [`SidecarClient::connect`]: a cancel shuts it,
    /// so a read blocked on a slow server returns at once and the handshake thread ends.
    type HandshakeSocket = Arc<Mutex<Option<TcpStream>>>;

    /// Connect, say hello, read the welcome and upload the reference (blocking, bounded). `abort`:
    /// the first connection's socket slot (see [`HandshakeSocket`]); `None` for reconnects.
    fn establish(
        config: &ClientConfig,
        cache: &mut ReferenceCache,
        abort: Option<&HandshakeSocket>,
    ) -> Result<(WebSocket<IoStream>, Session), String> {
        if abort.is_some() && config.cancelled() {
            return Err(CONNECT_CANCELLED.into());
        }
        let endpoint = (config.endpoint)()?;
        let mut request = endpoint
            .url
            .trim()
            .into_client_request()
            .map_err(|e| format!("sidecar URL: {e}"))?;
        let uri = request.uri().clone();
        if uri.scheme_str() != Some("ws") {
            return Err("a sidecar URL must start with ws:// (sidecars run on this PC)".into());
        }
        let host = uri.host().ok_or("the sidecar URL has no host")?;
        let host = host.trim_start_matches('[').trim_end_matches(']').to_string();
        if !is_loopback_host(&host) {
            return Err(format!("sidecars run on this PC only (127.0.0.1, localhost or [::1]), not {host}"));
        }
        let port = uri.port_u16().unwrap_or(80);
        if let Some(key) = endpoint.api_key.as_deref().filter(|k| !k.is_empty()) {
            let value = HeaderValue::from_str(&format!("Bearer {key}")).map_err(|_| "the sidecar key is not a valid header value")?;
            request.headers_mut().insert(AUTHORIZATION, value);
        }
        let deadline = Instant::now() + config.handshake_timeout;
        let tcp = connect_tcp(&host, port)?;
        if let Some(slot) = abort {
            // Checked under the slot's lock: a cancel either finds the socket to shut or is seen here.
            let mut s = slot.lock().unwrap_or_else(PoisonError::into_inner);
            if config.cancelled() {
                return Err(CONNECT_CANCELLED.into());
            }
            *s = tcp.try_clone().ok();
        }
        let _ = tcp.set_nodelay(true);
        let mut stream = IoStream::blocking(tcp);
        stream.set_timeout(config.handshake_timeout)?;
        let (mut ws, _) = tungstenite::client::client_with_config(request, stream, Some(ws_config())).map_err(|e| match e {
            HandshakeError::Failure(tungstenite::Error::Http(resp)) if matches!(resp.status().as_u16(), 401 | 403) => {
                format!("the sidecar refused Sigil's key (HTTP {})", resp.status().as_u16())
            }
            HandshakeError::Failure(f) => format!("sidecar WebSocket connect: {f}"),
            HandshakeError::Interrupted(_) => "sidecar WebSocket connect: the server did not answer in time".to_string(),
        })?;
        let offer = Offer::of(config);
        send_blocking(&mut ws, Message::Text(hello(config, &offer)), deadline)?;
        let session = loop {
            let msg = read_blocking(&mut ws, deadline, "the sidecar did not answer the handshake in time")?;
            match negotiate(&offer, &msg) {
                Reply::Ready(s) => break s,
                Reply::Refused(why) => return Err(why),
                Reply::Wait => continue,
            }
        };
        if session.reference_upload {
            upload_reference(&mut ws, config, &session, cache)?;
        }
        Ok((ws, session))
    }

    // -----------------------------------------------------------------------------------------
    // Client
    // -----------------------------------------------------------------------------------------

    enum Cmd {
        Audio { seq: u32, flags: u8, pitch: f32, pcm: Vec<f32> },
        Params(Map<String, Value>),
        /// A cancel is pending in the shared watermark ([`CancelMark`]); this only wakes the loop.
        Wake,
    }

    /// `a` comes before `b` (32-bit sequence numbers, wrapping): the frames a cancel drops.
    pub(crate) fn seq_before(a: u32, b: u32) -> bool {
        a != b && b.wrapping_sub(a) < 0x8000_0000
    }

    /// The newest cancel the caller asked for, shared with the I/O thread. A cancel is never lost:
    /// it is stored here first and the queued [`Cmd::Wake`] only wakes the loop (when the queue is
    /// full, the queued frames wake it anyway).
    #[derive(Default)]
    struct CancelMark(std::sync::atomic::AtomicU64);

    impl CancelMark {
        const SET: u64 = 1 << 32;

        fn set(&self, before: u32) {
            self.0.store(Self::SET | u64::from(before), std::sync::atomic::Ordering::Release);
        }

        /// The pending cancel, if any (taken: the next call returns `None` until a new one).
        fn take(&self) -> Option<u32> {
            let v = self.0.swap(0, std::sync::atomic::Ordering::AcqRel);
            (v & Self::SET != 0).then_some(v as u32)
        }
    }

    /// What the I/O thread reports.
    #[derive(Debug, PartialEq)]
    pub enum Event {
        /// Converted audio for frame `seq`.
        Audio { seq: u32, flags: u8, pcm: Vec<f32> },
        /// Frame `seq` will not be answered (the server dropped it, refused it, or it could not be
        /// sent).
        Dropped { seq: u32, reason: String },
        /// Frame `seq` was not converted because another voice's stream holds the model of a
        /// single-stream server (`dropped` with reason [`DROP_SUPERSEDED`]): this voice is being
        /// replaced (the moment before a voice switch). Silence, not an error.
        Superseded { seq: u32 },
        /// The connection was lost (frames in flight will not be answered); reconnecting.
        Lost(String),
        /// Reconnected with an equivalent session (handshake and reference again).
        Restored(Session),
        /// The link is gone for good (no reconnect, or the server came back with other settings).
        Failed(String),
        /// Something worth logging (an unexpected message, a non-fatal server error).
        Notice(String),
    }

    /// A connected v2 (or v1-compatible) sidecar. Every method returns at once.
    pub struct SidecarClient {
        session: Session,
        tx: mpsc::Sender<Cmd>,
        cancel: Arc<CancelMark>,
        events: std_mpsc::Receiver<Event>,
    }

    impl SidecarClient {
        /// Connect and negotiate (blocking, bounded by the config's timeouts): call it on a loader
        /// thread, never on the audio or worker thread. Returns [`CONNECT_CANCELLED`] soon after
        /// [`ClientConfig::cancel`] is set. The I/O thread keeps running until the client is
        /// dropped.
        pub fn connect(config: ClientConfig) -> Result<SidecarClient, String> {
            config.validate()?;
            // The queue holds frames not yet written (written frames are the server's); room for a
            // short frame's whole window over the budget.
            let capacity = (config.max_in_flight * 2 + 16).max(64);
            let (tx, rx) = mpsc::channel(capacity);
            let (ev_tx, events) = std_mpsc::channel();
            let (ready_tx, ready_rx) = std_mpsc::channel();
            let cancel = Arc::new(CancelMark::default());
            let io_cancel = cancel.clone();
            let abort: HandshakeSocket = Arc::default();
            let io_abort = abort.clone();
            let called_off = config.cancel.clone();
            // Connect timeouts per address are in `connect_tcp`; this bounds the whole handshake.
            let deadline = Instant::now() + config.handshake_timeout + config.reference_timeout + Duration::from_secs(10);
            let label = config.label.clone();
            std::thread::Builder::new()
                .name("sigil-vc-sidecar".into())
                .spawn(move || io_thread(config, rx, io_cancel, ev_tx, ready_tx, io_abort))
                .map_err(|e| format!("cannot start the sidecar thread: {e}"))?;
            // Shut the handshake's socket: a read blocked on the server returns, the thread ends.
            let give_up = |why: &str| -> Result<SidecarClient, String> {
                if let Some(s) = abort.lock().unwrap_or_else(PoisonError::into_inner).take() {
                    let _ = s.shutdown(Shutdown::Both);
                }
                Err(why.to_string())
            };
            loop {
                match ready_rx.recv_timeout(CANCEL_POLL) {
                    Ok(Ok(session)) => {
                        tracing::info!(
                            "voice changer: sidecar {label} connected (protocol v{}, {} -> {} Hz, {:.0} ms frames, model {:?}, device {:?})",
                            session.protocol,
                            session.input_rate,
                            session.output_rate,
                            session.frame_ms(),
                            session.model,
                            session.device
                        );
                        return Ok(SidecarClient { session, tx, cancel, events });
                    }
                    Ok(Err(e)) => return Err(e),
                    Err(std_mpsc::RecvTimeoutError::Disconnected) => return Err("the sidecar connection thread stopped".into()),
                    Err(std_mpsc::RecvTimeoutError::Timeout) => {
                        if called_off.as_ref().is_some_and(|c| c.load(Ordering::Relaxed)) {
                            tracing::info!("voice changer: sidecar {label}: connection cancelled (the voice is no longer wanted)");
                            return give_up(CONNECT_CANCELLED);
                        }
                        if Instant::now() >= deadline {
                            return give_up("the sidecar did not finish its handshake in time");
                        }
                    }
                }
            }
        }

        /// The negotiated session (of the first connection; a reconnect keeps it, see
        /// [`Event::Restored`]).
        pub fn session(&self) -> &Session {
            &self.session
        }

        /// Queue one frame of input audio (never blocks).
        pub fn send_audio(&self, seq: u32, flags: u8, pitch: f32, pcm: Vec<f32>) -> Result<(), String> {
            self.tx.try_send(Cmd::Audio { seq, flags, pitch, pcm }).map_err(|e| match e {
                mpsc::error::TrySendError::Full(_) => "the sidecar link is busy".to_string(),
                mpsc::error::TrySendError::Closed(_) => "the sidecar link is closed".to_string(),
            })
        }

        /// Queue a parameter change (applies from the next frame); false when it could not be
        /// queued (retry later).
        pub fn set_params(&self, params: Map<String, Value>) -> bool {
            self.tx.try_send(Cmd::Params(params)).is_ok()
        }

        /// Tell the server that frames before `seq` are no longer wanted (frames before it that are
        /// still queued here are not sent at all). Never lost: the newest cancel waits for the I/O
        /// thread even when the queue is full. False only when the link is closed for good.
        pub fn cancel_before(&self, seq: u32) -> bool {
            self.cancel.set(seq);
            match self.tx.try_send(Cmd::Wake) {
                Ok(()) | Err(mpsc::error::TrySendError::Full(_)) => true,
                Err(mpsc::error::TrySendError::Closed(_)) => false,
            }
        }

        /// The next event, if any.
        pub fn try_event(&self) -> Option<Event> {
            self.events.try_recv().ok()
        }
    }

    // -----------------------------------------------------------------------------------------
    // I/O thread
    // -----------------------------------------------------------------------------------------

    enum PumpExit {
        Shutdown,
        Lost(String),
    }

    fn io_thread(
        config: ClientConfig,
        mut rx: mpsc::Receiver<Cmd>,
        cancel: Arc<CancelMark>,
        events: std_mpsc::Sender<Event>,
        ready: std_mpsc::Sender<Result<Session, String>>,
        abort: HandshakeSocket,
    ) {
        let rt = match tokio::runtime::Builder::new_current_thread().enable_io().enable_time().build() {
            Ok(rt) => rt,
            Err(e) => {
                let _ = ready.send(Err(format!("cannot start the sidecar I/O runtime: {e}")));
                return;
            }
        };
        let mut cache = ReferenceCache::default();
        let first = establish(&config, &mut cache, Some(&abort));
        // The handshake is over: a later cancel has nothing to shut (the caller drops the client).
        drop(abort.lock().unwrap_or_else(PoisonError::into_inner).take());
        let (mut ws, mut session) = match first {
            Ok(v) => v,
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        if ready.send(Ok(session.clone())).is_err() {
            return;
        }
        loop {
            // Register the socket with this thread's runtime, then stream until it ends.
            let tcp = {
                let _guard = rt.enter();
                ws.get_mut().make_async()
            };
            let exit = match tcp {
                Ok(tcp) => rt.block_on(pump(&mut ws, tcp, &mut rx, &cancel, &events, session.protocol)),
                Err(e) => PumpExit::Lost(format!("sidecar socket: {e}")),
            };
            let reason = match exit {
                PumpExit::Shutdown => return,
                PumpExit::Lost(reason) => reason,
            };
            drop(ws);
            tracing::warn!("voice changer: sidecar {} connection lost: {reason}", config.label);
            let _ = events.send(Event::Lost(reason.clone()));
            if !config.reconnect {
                let _ = events.send(Event::Failed(reason));
                return;
            }
            let mut backoff = RECONNECT_MIN;
            loop {
                if wait_draining(&mut rx, &events, backoff) {
                    return;
                }
                match establish(&config, &mut cache, None) {
                    Ok((w, s)) => {
                        if let Some(why) = session.incompatible_with(&s) {
                            tracing::warn!("voice changer: sidecar {}: {why}", config.label);
                            let _ = events.send(Event::Failed(why));
                            return;
                        }
                        tracing::info!("voice changer: sidecar {} reconnected", config.label);
                        ws = w;
                        session = s;
                        let _ = events.send(Event::Restored(session.clone()));
                        break;
                    }
                    Err(e) => {
                        tracing::debug!("voice changer: sidecar {} reconnect failed: {e}", config.label);
                        backoff = (backoff * 2).min(RECONNECT_MAX);
                    }
                }
            }
        }
    }

    /// Wait `dur` while disconnected, answering queued frames as dropped; true on shutdown.
    fn wait_draining(rx: &mut mpsc::Receiver<Cmd>, events: &std_mpsc::Sender<Event>, dur: Duration) -> bool {
        let end = Instant::now() + dur;
        loop {
            loop {
                match rx.try_recv() {
                    Ok(Cmd::Audio { seq, .. }) => {
                        let _ = events.send(Event::Dropped {
                            seq,
                            reason: "the sidecar is reconnecting".into(),
                        });
                    }
                    Ok(_) => {}
                    Err(mpsc::error::TryRecvError::Empty) => break,
                    Err(mpsc::error::TryRecvError::Disconnected) => return true,
                }
            }
            let now = Instant::now();
            if now >= end {
                return false;
            }
            std::thread::sleep((end - now).min(IDLE_POLL));
        }
    }

    /// Flush tungstenite's buffers; `Ok(true)` while bytes remain (wait for writability).
    fn flush(ws: &mut WebSocket<IoStream>) -> Result<bool, String> {
        match ws.flush() {
            Ok(()) => Ok(false),
            Err(e) if would_block(&e) => Ok(true),
            Err(e) => Err(format!("sidecar write: {e}")),
        }
    }

    /// What the streaming loop keeps between commands.
    struct Outbox {
        protocol: u32,
        /// Sequence numbers of v1 frames in flight (v1 replies come in order and carry none).
        v1_fifo: VecDeque<u32>,
        /// Frames before this one were cancelled: still-queued ones are not sent.
        cancelled_before: Option<u32>,
    }

    impl Outbox {
        /// Send the pending cancel (if any) before the next queued command.
        fn take_cancel(&mut self, ws: &mut WebSocket<IoStream>, cancel: &CancelMark, events: &std_mpsc::Sender<Event>) -> Result<(), String> {
            let Some(before) = cancel.take() else {
                return Ok(());
            };
            self.cancelled_before = Some(before);
            if self.protocol < 2 {
                // v1 servers know no cancellation; the stale frames still queued are skipped.
                return Ok(());
            }
            let msg = Message::Text(json!({"type": "cancel", "before_seq": before}).to_string());
            write_message(ws, msg, None, self, events)
        }

        fn stale(&self, seq: u32) -> bool {
            self.cancelled_before.is_some_and(|before| seq_before(seq, before))
        }
    }

    /// The streaming loop: queued commands out, replies in, until shutdown or a lost connection.
    async fn pump(
        ws: &mut WebSocket<IoStream>,
        tcp: Arc<tokio::net::TcpStream>,
        rx: &mut mpsc::Receiver<Cmd>,
        cancel: &CancelMark,
        events: &std_mpsc::Sender<Event>,
        protocol: u32,
    ) -> PumpExit {
        let mut out = Outbox {
            protocol,
            v1_fifo: VecDeque::new(),
            cancelled_before: None,
        };
        // Messages the handshake reads may have left in tungstenite's buffer.
        if let Err(e) = read_available(ws, events, protocol, &mut out.v1_fifo) {
            return PumpExit::Lost(e);
        }
        let mut want_flush = false;
        loop {
            tokio::select! {
                cmd = rx.recv() => {
                    let Some(cmd) = cmd else {
                        let _ = ws.close(None);
                        return PumpExit::Shutdown;
                    };
                    let mut next = Some(cmd);
                    while let Some(cmd) = next {
                        if let Err(e) = out.take_cancel(ws, cancel, events).and_then(|()| write_cmd(ws, cmd, &mut out, events)) {
                            return PumpExit::Lost(e);
                        }
                        next = rx.try_recv().ok();
                    }
                    if let Err(e) = out.take_cancel(ws, cancel, events) {
                        return PumpExit::Lost(e);
                    }
                    match flush(ws) {
                        Ok(pending) => want_flush = pending,
                        Err(e) => return PumpExit::Lost(e),
                    }
                }
                ready = tcp.readable() => {
                    if let Err(e) = ready {
                        return PumpExit::Lost(format!("sidecar socket: {e}"));
                    }
                    if let Err(e) = read_available(ws, events, protocol, &mut out.v1_fifo) {
                        return PumpExit::Lost(e);
                    }
                    // Pong and close replies tungstenite queued while reading.
                    match flush(ws) {
                        Ok(pending) => want_flush = pending,
                        Err(e) => return PumpExit::Lost(e),
                    }
                }
                ready = tcp.writable(), if want_flush => {
                    if let Err(e) = ready {
                        return PumpExit::Lost(format!("sidecar socket: {e}"));
                    }
                    match flush(ws) {
                        Ok(pending) => want_flush = pending,
                        Err(e) => return PumpExit::Lost(e),
                    }
                }
            }
        }
    }

    fn write_cmd(ws: &mut WebSocket<IoStream>, cmd: Cmd, out: &mut Outbox, events: &std_mpsc::Sender<Event>) -> Result<(), String> {
        let (msg, audio_seq) = match cmd {
            // The caller gave this frame up already (it played silence for it): never send it.
            Cmd::Audio { seq, .. } if out.stale(seq) => return Ok(()),
            Cmd::Audio { seq, flags, pitch, pcm } => {
                if out.protocol >= 2 {
                    let header = FrameHeader { kind: KIND_AUDIO_IN, flags, seq, pitch };
                    (Message::Binary(encode_frame(&header, &pcm)), Some(seq))
                } else {
                    out.v1_fifo.push_back(seq);
                    (Message::Binary(encode_pcm_chunk(&pcm)), Some(seq))
                }
            }
            Cmd::Wake => return Ok(()),
            // v1 servers know no parameters.
            Cmd::Params(_) if out.protocol < 2 => return Ok(()),
            Cmd::Params(params) => (Message::Text(json!({"type": "params", "params": params}).to_string()), None),
        };
        write_message(ws, msg, audio_seq, out, events)
    }

    /// Queue one message on the socket; a frame (`audio_seq`) that does not fit in the write
    /// buffer is reported dropped, a control message that does not fit is let go (the server drops
    /// stale frames by itself).
    fn write_message(
        ws: &mut WebSocket<IoStream>,
        msg: Message,
        audio_seq: Option<u32>,
        out: &mut Outbox,
        events: &std_mpsc::Sender<Event>,
    ) -> Result<(), String> {
        match ws.write(msg) {
            Ok(()) => Ok(()),
            // Buffered by tungstenite; the loop flushes when the socket is writable.
            Err(e) if would_block(&e) => Ok(()),
            Err(tungstenite::Error::WriteBufferFull(_)) => {
                if let Some(seq) = audio_seq {
                    if out.protocol < 2 {
                        out.v1_fifo.pop_back();
                    }
                    let _ = events.send(Event::Dropped {
                        seq,
                        reason: "the sidecar is not reading its input".into(),
                    });
                }
                Ok(())
            }
            Err(e) => Err(format!("sidecar write: {e}")),
        }
    }

    /// Read every message available now.
    fn read_available(
        ws: &mut WebSocket<IoStream>,
        events: &std_mpsc::Sender<Event>,
        protocol: u32,
        v1_fifo: &mut VecDeque<u32>,
    ) -> Result<(), String> {
        loop {
            match ws.read() {
                Ok(msg) => handle_message(msg, events, protocol, v1_fifo)?,
                Err(e) if would_block(&e) => return Ok(()),
                Err(tungstenite::Error::ConnectionClosed) | Err(tungstenite::Error::AlreadyClosed) => {
                    return Err("the sidecar closed the connection".into())
                }
                Err(e) => return Err(format!("sidecar read: {e}")),
            }
        }
    }

    fn handle_message(msg: Message, events: &std_mpsc::Sender<Event>, protocol: u32, v1_fifo: &mut VecDeque<u32>) -> Result<(), String> {
        match msg {
            Message::Binary(b) if protocol >= 2 => {
                let (header, pcm) = decode_frame(&b)?;
                if header.kind == KIND_AUDIO_OUT {
                    let _ = events.send(Event::Audio { seq: header.seq, flags: header.flags, pcm });
                } else {
                    let _ = events.send(Event::Notice(format!("sidecar sent a frame of kind {}", header.kind)));
                }
            }
            Message::Binary(b) => {
                let Some(seq) = v1_fifo.pop_front() else {
                    let _ = events.send(Event::Notice("sidecar sent audio nobody asked for".into()));
                    return Ok(());
                };
                let _ = events.send(match decode_pcm_chunk(&b) {
                    Ok(pcm) => Event::Audio { seq, flags: 0, pcm },
                    Err(reason) => Event::Dropped { seq, reason },
                });
            }
            Message::Text(t) if protocol >= 2 => handle_control(&t, events)?,
            Message::Close(frame) => {
                let why = frame.map(|f| f.reason.to_string()).filter(|r| !r.is_empty());
                return Err(match why {
                    Some(r) => format!("the sidecar closed the connection: {}", r.chars().take(MAX_QUOTE).collect::<String>()),
                    None => "the sidecar closed the connection".into(),
                });
            }
            _ => {}
        }
        Ok(())
    }

    fn handle_control(text: &str, events: &std_mpsc::Sender<Event>) -> Result<(), String> {
        let Ok(v) = serde_json::from_str::<Value>(text) else {
            return Ok(());
        };
        let seq = v.get("seq").and_then(Value::as_u64).and_then(|s| u32::try_from(s).ok());
        match v.get("type").and_then(Value::as_str).unwrap_or_default() {
            "dropped" => {
                if let Some(seq) = seq {
                    let why = quoted(&v);
                    if why == DROP_SUPERSEDED {
                        let _ = events.send(Event::Superseded { seq });
                        return Ok(());
                    }
                    let _ = events.send(Event::Dropped {
                        seq,
                        reason: if why.is_empty() {
                            "the sidecar dropped the frame (it was stale)".into()
                        } else {
                            format!("the sidecar dropped the frame ({why})")
                        },
                    });
                }
            }
            "error" => {
                let message = quoted(&v);
                if v.get("fatal").and_then(Value::as_bool).unwrap_or(false) {
                    return Err(format!("sidecar error: {message}"));
                }
                let _ = events.send(match seq {
                    Some(seq) => Event::Dropped { seq, reason: format!("sidecar error: {message}") },
                    None => Event::Notice(format!("sidecar: {message}")),
                });
            }
            // `stats` and anything newer: nothing to do.
            _ => {}
        }
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::super::test_server::{Behavior, TestServer};
        use super::*;

        fn offer() -> Offer {
            Offer {
                rates: vec![16_000, 22_050, 48_000],
                frame_ms: 100.0,
                has_reference: true,
                allow_v1: true,
            }
        }

        fn welcome(extra: Value) -> Message {
            let mut v = json!({"type": WELCOME_TYPE, "protocol_version": 2, "sample_rate_in": 16000, "sample_rate_out": 22050});
            if let (Some(o), Some(e)) = (v.as_object_mut(), extra.as_object()) {
                o.extend(e.clone());
            }
            Message::Text(v.to_string())
        }

        #[test]
        fn frames_round_trip_and_bad_frames_are_refused() {
            let h = FrameHeader { kind: KIND_AUDIO_IN, flags: FLAG_DISCONTINUITY, seq: 0xDEAD_BEEF, pitch: -3.5 };
            let pcm = vec![0.25f32, -1.0, 0.5];
            let enc = encode_frame(&h, &pcm);
            assert_eq!(enc.len(), HEADER_LEN + 12);
            assert_eq!(&enc[..4], b"SGV2");
            assert_eq!(decode_frame(&enc).unwrap(), (h, pcm));
            assert!(decode_frame(&enc[..HEADER_LEN - 1]).is_err());
            assert!(decode_frame(&enc[..enc.len() - 1]).unwrap_err().contains("declares 3 samples"));
            let mut bad = enc.clone();
            bad[0] = b'X';
            assert!(decode_frame(&bad).unwrap_err().contains("SGV2"));
            let mut huge = enc.clone();
            huge[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
            assert!(decode_frame(&huge).is_err());
            // An empty frame is valid (the last reference chunk of an empty clip).
            assert!(decode_frame(&encode_frame(&h, &[])).unwrap().1.is_empty());
        }

        #[test]
        fn whole_frames_map_to_whole_output_samples() {
            assert_eq!(whole_frame(100.0, 16_000, 22_050), 1600);
            assert_eq!(whole_frame(90.0, 16_000, 22_050), 1600, "rounded to 20 ms units (320 samples)");
            assert_eq!(whole_frame(120.0, 16_000, 16_000), 1920);
            assert_eq!(whole_frame(1.0, 16_000, 22_050), 320, "at least one unit");
            for (ms, a, b) in [(80.0, 16_000, 44_100), (92.0, 16_000, 24_000), (180.0, 22_050, 48_000)] {
                let f = whole_frame(ms, a, b);
                assert_eq!(f as u64 * u64::from(b) % u64::from(a), 0, "{ms} {a} {b}");
            }
        }

        #[test]
        fn v2_welcome_negotiates_rates_frame_and_reference() {
            let Reply::Ready(s) = negotiate(
                &offer(),
                &welcome(json!({"frame_samples": 1920, "latency_ms": 58.0, "delay_samples": 2646, "reference_needed": true, "pitch": true,
                    "params": ["formant_semitones", 3], "model": "x-vc", "device": "CUDA", "vram_mb": 4100.5, "max_in_flight": 4})),
            ) else {
                panic!("not ready")
            };
            assert_eq!((s.protocol, s.input_rate, s.output_rate, s.frame_in, s.frame_out), (2, 16_000, 22_050, 1920, 2646));
            assert_eq!((s.latency_ms, s.delay_out, s.max_in_flight), (Some(58.0), 2646, Some(4)));
            assert!(s.reference_used && s.reference_upload && !s.reference_required && s.pitch());
            assert_eq!(s.params, ["formant_semitones"]);
            assert_eq!((s.model.as_str(), s.device.as_deref(), s.vram_mb), ("x-vc", Some("cuda"), Some(4100.5)));
            assert_eq!(s.reference_rate, 16_000);
            assert!((s.frame_ms() - 120.0).abs() < 1e-9);
            // No frame given: the client's 100 ms, rounded to whole output samples.
            let Reply::Ready(s) = negotiate(&offer(), &welcome(json!({}))) else { panic!() };
            assert_eq!((s.frame_in, s.frame_out), (1600, 2205));
            assert_eq!(s.max_in_flight, None, "no limit of the server's own: the client's offer is not one");
            assert!(!s.reference_used, "the server did not ask for it");
            assert!(!s.pitch(), "pitch only when the server says it applies it");
            // A clip the server prepared before (same sha256) is used without uploading it again.
            let Reply::Ready(s) = negotiate(&offer(), &welcome(json!({"reference_needed": false, "reference_cached": true}))) else { panic!() };
            assert!(s.reference_used && !s.reference_upload);
        }

        #[test]
        fn bad_welcomes_are_refused_with_a_reason() {
            let refused = |extra: Value| match negotiate(&offer(), &welcome(extra)) {
                Reply::Refused(why) => why,
                other => panic!("{other:?}"),
            };
            assert!(refused(json!({"sample_rate_in": 32000})).contains("32000 Hz input"));
            assert!(refused(json!({"sample_rate_out": 1})).contains("sample_rate_out"));
            assert!(refused(json!({"frame_samples": 1601})).contains("whole number of samples"));
            assert!(refused(json!({"frame_samples": 16})).contains("outside"));
            assert!(refused(json!({"frame_samples": "x"})).contains("whole number"));
            assert!(refused(json!({"delay_samples": 10_000_000})).contains("delay"));
            let mut no_ref = offer();
            no_ref.has_reference = false;
            match negotiate(&no_ref, &welcome(json!({"reference_required": true}))) {
                Reply::Refused(why) => assert!(why.contains("reference voice"), "{why}"),
                other => panic!("{other:?}"),
            }
            match negotiate(&offer(), &Message::Text(r#"{"ok":false,"error":"no GPU"}"#.into())) {
                Reply::Refused(why) => assert_eq!(why, "the sidecar refused the session: no GPU"),
                other => panic!("{other:?}"),
            }
        }

        #[test]
        fn v1_acknowledgements_fall_back_to_v1_framing() {
            for ack in [Message::Text(r#"{"ok":true,"mode":"echo"}"#.into()), Message::Text("ok".into()), Message::Binary(vec![1])] {
                let Reply::Ready(s) = negotiate(&offer(), &ack) else { panic!("{ack:?}") };
                assert_eq!((s.protocol, s.input_rate, s.output_rate, s.frame_in, s.frame_out), (1, 16_000, 16_000, 1600, 1600));
                assert!(!s.pitch() && !s.reference_used);
            }
            assert_eq!(negotiate(&offer(), &Message::Ping(vec![])), Reply::Wait);
            let mut strict = offer();
            strict.allow_v1 = false;
            assert!(matches!(negotiate(&strict, &Message::Text("ok".into())), Reply::Refused(_)));
        }

        #[test]
        fn only_loopback_hosts_are_sidecar_hosts() {
            for ok in ["127.0.0.1", "localhost", "LOCALHOST", "[::1]", "::1", "127.0.0.2"] {
                assert!(is_loopback_host(ok), "{ok}");
            }
            for bad in ["192.168.1.2", "example.com", "0.0.0.0", "", "127.0.0.1.evil"] {
                assert!(!is_loopback_host(bad), "{bad}");
            }
        }

        fn config(url: &str) -> ClientConfig {
            let mut c = ClientConfig::new("test", ClientConfig::fixed(url, None));
            c.handshake_timeout = Duration::from_secs(5);
            c.reference_timeout = Duration::from_secs(5);
            c
        }

        /// Wait (bounded) for events until `done` says stop; returns everything received.
        fn collect(client: &SidecarClient, max: Duration, mut done: impl FnMut(&[Event]) -> bool) -> Vec<Event> {
            let end = Instant::now() + max;
            let mut got = Vec::new();
            while Instant::now() < end && !done(&got) {
                match client.try_event() {
                    Some(e) => got.push(e),
                    None => std::thread::sleep(Duration::from_millis(2)),
                }
            }
            got
        }

        fn audio_seqs(events: &[Event]) -> Vec<u32> {
            events
                .iter()
                .filter_map(|e| match e {
                    Event::Audio { seq, .. } => Some(*seq),
                    _ => None,
                })
                .collect()
        }

        /// Frames are pipelined: ten frames go out at once without waiting, and each is answered
        /// (700 ms later) with its own sequence number, pitch and parameters reaching the server.
        #[test]
        fn frames_are_pipelined_and_answered_by_sequence() {
            let server = TestServer::spawn(Behavior::delayed(Duration::from_millis(700)));
            let client = SidecarClient::connect(config(&server.url)).unwrap();
            let s = client.session().clone();
            assert_eq!((s.protocol, s.frame_in, s.frame_out), (2, 1600, 1600));
            let t0 = Instant::now();
            assert!(client.set_params(Map::from_iter([("formant_semitones".to_string(), json!(2.0))])));
            for seq in 0..10u32 {
                client.send_audio(seq, if seq == 0 { FLAG_DISCONTINUITY } else { 0 }, 1.5, vec![seq as f32; 1600]).unwrap();
            }
            assert!(t0.elapsed() < Duration::from_millis(50), "sending never waits: {:?}", t0.elapsed());
            let got = collect(&client, Duration::from_secs(5), |e| audio_seqs(e).len() == 10);
            assert_eq!(audio_seqs(&got), (0..10).collect::<Vec<_>>());
            let elapsed = t0.elapsed();
            assert!(elapsed < Duration::from_millis(1600), "pipelined, not 10 x 700 ms: {elapsed:?}");
            for e in &got {
                if let Event::Audio { seq, pcm, .. } = e {
                    assert!(pcm.len() == 1600 && pcm.iter().all(|&x| x == *seq as f32));
                }
            }
            let log = server.log();
            assert_eq!(log.frames.len(), 10);
            assert_eq!((log.frames[0].1, log.frames[1].1), (FLAG_DISCONTINUITY, 0));
            assert!(log.frames.iter().all(|f| f.2 == 1.5), "pitch rides on every frame");
            assert_eq!(log.params, [json!({"formant_semitones": 2.0})]);
            let hello = log.hello.clone().unwrap();
            assert_eq!(hello["protocol_version"], 2);
            assert_eq!(hello["sample_rates"][0], 16000);
        }

        /// A server that never answers costs the caller nothing: every call returns at once, and
        /// a cancel tells the server which frames are stale.
        #[test]
        fn a_stalled_server_never_blocks_the_caller() {
            let server = TestServer::spawn(Behavior::stalled());
            let client = SidecarClient::connect(config(&server.url)).unwrap();
            let t0 = Instant::now();
            for seq in 0..50u32 {
                // The queue fills up (the server reads but never answers): later frames are
                // refused at once instead of waiting.
                let _ = client.send_audio(seq, 0, 0.0, vec![0.0; 1600]);
                assert!(client.try_event().is_none());
            }
            // The cancel is never lost, even with the queue full.
            assert!(client.cancel_before(40));
            assert!(t0.elapsed() < Duration::from_millis(100), "{:?}", t0.elapsed());
            let end = Instant::now() + Duration::from_secs(3);
            while Instant::now() < end && !server.log().cancels.contains(&40) {
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(server.log().cancels.contains(&40), "{:?}", server.log().cancels);
            // Frames sent after the cancel go out as usual.
            client.send_audio(50, 0, 0.0, vec![0.0; 1600]).unwrap();
            let end = Instant::now() + Duration::from_secs(3);
            while Instant::now() < end && !server.log().frames.iter().any(|f| f.0 == 50) {
                std::thread::sleep(Duration::from_millis(5));
            }
            assert!(server.log().frames.iter().any(|f| f.0 == 50));
        }

        #[test]
        fn cancels_compare_wrapping_sequence_numbers_and_keep_only_the_newest() {
            assert!(seq_before(1, 2) && !seq_before(2, 1) && !seq_before(5, 5));
            assert!(seq_before(u32::MAX, 3), "wraps");
            assert!(!seq_before(3, u32::MAX));
            let mark = CancelMark::default();
            assert_eq!(mark.take(), None);
            mark.set(7);
            mark.set(9);
            assert_eq!(mark.take(), Some(9), "the newest cancel wins");
            assert_eq!(mark.take(), None, "taken once");
            mark.set(0);
            assert_eq!(mark.take(), Some(0), "seq 0 is a valid cancel");
        }

        /// The reference voice is uploaded in chunks at the rate the server asks for, and the
        /// session is ready only once the server has prepared it.
        #[test]
        fn the_reference_voice_is_uploaded_before_streaming() {
            let mut b = Behavior::delayed(Duration::ZERO);
            b.welcome = json!({"type": WELCOME_TYPE, "protocol_version": 2, "sample_rate_in": 16000, "sample_rate_out": 24000,
                "frame_samples": 1600, "reference_required": true, "reference_sample_rate": 22050, "max_reference_seconds": 5});
            b.reference_delay = Duration::from_millis(300);
            b.ratio = (3, 2);
            let server = TestServer::spawn(b);
            let mut c = config(&server.url);
            let asked = Arc::new(std::sync::Mutex::new(Vec::new()));
            let asked2 = asked.clone();
            c.reference = Some(ReferenceSource {
                meta: ReferenceMeta { voice_id: "v1".into(), name: "Voice".into(), sha256: "ab".into(), duration_ms: 8000, ..Default::default() },
                load: Arc::new(move |rate| {
                    asked2.lock().unwrap().push(rate);
                    Ok((0..rate as usize * 8).map(|i| (i % 7) as f32).collect())
                }),
            });
            let t0 = Instant::now();
            let client = SidecarClient::connect(c).unwrap();
            assert!(t0.elapsed() >= Duration::from_millis(300), "waited for reference_ready");
            assert!(client.session().reference_used);
            assert_eq!(client.session().frame_out, 2400);
            assert_eq!(*asked.lock().unwrap(), [22_050]);
            let log = server.log();
            assert_eq!(log.reference.len(), 22_050 * 5, "cut to the server's 5 s");
            assert!(log.reference_chunks > 1, "sent in chunks");
            assert_eq!(log.reference_meta.as_ref().unwrap()["voice_id"], "v1");
            // Streams at the negotiated rates.
            client.send_audio(0, 0, 0.0, vec![0.5; 1600]).unwrap();
            let got = collect(&client, Duration::from_secs(3), |e| !audio_seqs(e).is_empty());
            assert!(matches!(&got[..], [Event::Audio { seq: 0, pcm, .. }] if pcm.len() == 2400));
        }

        #[test]
        fn a_required_reference_without_one_fails_the_connect() {
            let mut b = Behavior::delayed(Duration::ZERO);
            b.welcome["reference_required"] = json!(true);
            let server = TestServer::spawn(b);
            let err = SidecarClient::connect(config(&server.url)).err().unwrap();
            assert!(err.contains("reference voice"), "{err}");
        }

        /// A voice nobody wants any more stops connecting at once, even while the server takes its
        /// time preparing the reference (the handshake's socket is shut); a flag set before the
        /// call refuses before contacting the server.
        #[test]
        fn a_cancelled_connect_returns_promptly() {
            let mut b = Behavior::delayed(Duration::ZERO);
            b.welcome["reference_required"] = json!(true);
            b.reference_delay = Duration::from_secs(5);
            let server = TestServer::spawn(b);
            let reference = || ReferenceSource {
                meta: ReferenceMeta { voice_id: "v".into(), sha256: "s".into(), ..Default::default() },
                load: Arc::new(|rate| Ok(vec![0.1; rate as usize])),
            };
            let mut c = config(&server.url);
            c.reference_timeout = Duration::from_secs(60);
            c.reference = Some(reference());
            let flag = Arc::new(AtomicBool::new(false));
            c.cancel = Some(flag.clone());
            let setter = flag.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(300));
                setter.store(true, Ordering::Relaxed);
            });
            let t0 = Instant::now();
            let err = SidecarClient::connect(c).err().unwrap();
            assert_eq!(err, CONNECT_CANCELLED);
            assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
            assert!(server.log().reference_meta.is_some(), "it was waiting for reference_ready");
            let mut c = config(&server.url);
            c.reference = Some(reference());
            c.cancel = Some(Arc::new(AtomicBool::new(true)));
            assert_eq!(SidecarClient::connect(c).err().unwrap(), CONNECT_CANCELLED);
            assert_eq!(server.log().connections, 1, "the second never reached the server");
        }

        /// The key goes in the Authorization header; a wrong key is refused by the server.
        #[test]
        fn the_key_is_sent_as_a_bearer_header() {
            let mut b = Behavior::delayed(Duration::ZERO);
            b.expect_key = Some("k3y".into());
            let server = TestServer::spawn(b);
            let mut c = config(&server.url);
            c.endpoint = ClientConfig::fixed(server.url.clone(), Some("k3y".into()));
            assert!(SidecarClient::connect(c).is_ok());
            let mut c = config(&server.url);
            c.endpoint = ClientConfig::fixed(server.url.clone(), Some("wrong".into()));
            let err = SidecarClient::connect(c).err().unwrap();
            assert!(err.contains("refused Sigil's key"), "{err}");
            assert!(SidecarClient::connect(config(&server.url)).is_err(), "no key, refused");
        }

        /// A v1 server works through the v2 client: replies are matched in order.
        #[test]
        fn a_v1_server_is_driven_in_order() {
            let server = TestServer::spawn(Behavior::v1());
            let client = SidecarClient::connect(config(&server.url)).unwrap();
            assert_eq!(client.session().protocol, 1);
            for seq in 0..5u32 {
                client.send_audio(seq, 0, 7.0, vec![seq as f32; 1600]).unwrap();
            }
            client.set_params(Map::new());
            let got = collect(&client, Duration::from_secs(3), |e| audio_seqs(e).len() == 5);
            assert_eq!(audio_seqs(&got), [0, 1, 2, 3, 4]);
            for e in &got {
                if let Event::Audio { seq, pcm, .. } = e {
                    assert!(pcm.iter().all(|&x| x == *seq as f32));
                }
            }
            let log = server.log();
            assert!(log.params.is_empty(), "no params for v1");
            assert_eq!(log.hello.as_ref().unwrap()["chunk_frames"], 1600);
        }

        /// A connection that drops is reported, re-established in the background (handshake and
        /// reference again) and streaming continues; frames sent while it is down are dropped.
        #[test]
        fn a_lost_connection_reconnects_in_the_background() {
            let mut b = Behavior::delayed(Duration::ZERO);
            b.close_after_frames = Some(3);
            let server = TestServer::spawn(b);
            let client = SidecarClient::connect(config(&server.url)).unwrap();
            for seq in 0..3u32 {
                client.send_audio(seq, 0, 0.0, vec![0.0; 1600]).unwrap();
            }
            let got = collect(&client, Duration::from_secs(5), |e| e.iter().any(|e| matches!(e, Event::Lost(_))));
            assert!(got.iter().any(|e| matches!(e, Event::Lost(_))), "{got:?}");
            let got = collect(&client, Duration::from_secs(10), |e| e.iter().any(|e| matches!(e, Event::Restored(_))));
            assert!(got.iter().any(|e| matches!(e, Event::Restored(s) if s.frame_in == 1600)), "{got:?}");
            client.send_audio(10, 0, 0.0, vec![1.0; 1600]).unwrap();
            let got = collect(&client, Duration::from_secs(3), |e| audio_seqs(e).contains(&10));
            assert!(audio_seqs(&got).contains(&10), "{got:?}");
            assert_eq!(server.log().connections, 2);
        }

        /// Server-side drops and frame errors are reported per frame; a fatal error drops the
        /// connection (and reconnects).
        #[test]
        fn server_drops_and_errors_reach_the_caller() {
            let (tx, _rx) = std_mpsc::channel();
            handle_control(r#"{"type":"dropped","seq":7,"reason":"stale"}"#, &tx).unwrap();
            handle_control(r#"{"type":"dropped","seq":9,"reason":"superseded"}"#, &tx).unwrap();
            handle_control(r#"{"type":"error","seq":8,"message":"CUDA OOM"}"#, &tx).unwrap();
            handle_control(r#"{"type":"error","message":"slow"}"#, &tx).unwrap();
            handle_control(r#"{"type":"stats","compute_ms":3}"#, &tx).unwrap();
            handle_control("not json", &tx).unwrap();
            assert!(handle_control(r#"{"type":"error","message":"model died","fatal":true}"#, &tx).unwrap_err().contains("model died"));
            drop(tx);
            let got: Vec<Event> = _rx.try_iter().collect();
            assert_eq!(
                got,
                [
                    Event::Dropped { seq: 7, reason: "the sidecar dropped the frame (stale)".into() },
                    Event::Superseded { seq: 9 },
                    Event::Dropped { seq: 8, reason: "sidecar error: CUDA OOM".into() },
                    Event::Notice("sidecar: slow".into()),
                ]
            );
        }

        #[test]
        fn non_loopback_urls_are_refused_before_connecting() {
            for url in ["ws://192.168.1.5:9/vc", "wss://127.0.0.1:9/vc", "http://127.0.0.1:9"] {
                let err = SidecarClient::connect(config(url)).err().unwrap();
                assert!(err.contains("this PC") || err.contains("ws://"), "{url}: {err}");
            }
        }
    }
}

/// A scripted sidecar for tests (protocol v2 or v1) on 127.0.0.1: answers each frame after a
/// per-frame delay (or never), records what it received, and can drop the connection.
#[cfg(test)]
pub(crate) mod test_server {
    use std::collections::VecDeque;
    use std::net::TcpListener;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex};
    use std::thread;
    use std::time::{Duration, Instant};

    use serde_json::{json, Value};
    use tungstenite::handshake::server::{ErrorResponse, Request, Response};
    use tungstenite::Message;

    use super::v2::{decode_frame, encode_frame, FrameHeader, FLAG_LAST, KIND_AUDIO_OUT, KIND_REFERENCE};
    use super::{decode_pcm_chunk, encode_pcm_chunk};

    /// Per-frame reply delay (`None` = never answer).
    pub(crate) type DelayFn = Arc<dyn Fn(u32) -> Option<Duration> + Send + Sync>;

    #[derive(Clone)]
    pub(crate) struct Behavior {
        /// The welcome (v2) or acknowledgement (v1) sent after the hello.
        pub welcome: Value,
        pub v1: bool,
        pub delay: DelayFn,
        pub reference_delay: Duration,
        pub expect_key: Option<String>,
        /// Close the connection after this many frames (once; later connections stay up).
        pub close_after_frames: Option<usize>,
        /// Output samples per input sample (`out / in` of the welcome).
        pub ratio: (usize, usize),
        /// v2 frames answered `dropped` / `superseded` at once (a single-stream server whose model
        /// another voice holds).
        pub superseded: Arc<dyn Fn(u32) -> bool + Send + Sync>,
    }

    impl Behavior {
        /// v2 echo at 16 kHz in and out, every frame answered after `d`.
        pub(crate) fn delayed(d: Duration) -> Behavior {
            Behavior {
                welcome: json!({"type": "sigil_vc_welcome", "protocol_version": 2, "sample_rate_in": 16000,
                    "sample_rate_out": 16000, "frame_samples": 1600, "latency_ms": 10.0, "pitch": true}),
                v1: false,
                delay: Arc::new(move |_| Some(d)),
                reference_delay: Duration::ZERO,
                expect_key: None,
                close_after_frames: None,
                ratio: (1, 1),
                superseded: Arc::new(|_| false),
            }
        }

        /// v2, never answers a frame.
        pub(crate) fn stalled() -> Behavior {
            Behavior {
                delay: Arc::new(|_| None),
                ..Behavior::delayed(Duration::ZERO)
            }
        }

        /// v1 echo (`{"ok":true}`), in order, no delay.
        pub(crate) fn v1() -> Behavior {
            Behavior {
                welcome: json!({"ok": true, "mode": "echo"}),
                v1: true,
                ..Behavior::delayed(Duration::ZERO)
            }
        }
    }

    #[derive(Clone, Debug, Default)]
    pub(crate) struct Log {
        pub hello: Option<Value>,
        /// `(seq, flags, pitch, samples)` per audio frame.
        pub frames: Vec<(u32, u8, f32, usize)>,
        pub params: Vec<Value>,
        pub cancels: Vec<u32>,
        pub reference_meta: Option<Value>,
        pub reference: Vec<f32>,
        pub reference_chunks: usize,
        pub connections: usize,
    }

    pub(crate) struct TestServer {
        pub url: String,
        log: Arc<Mutex<Log>>,
        stop: Arc<AtomicBool>,
    }

    impl Drop for TestServer {
        fn drop(&mut self) {
            self.stop.store(true, Ordering::Release);
        }
    }

    impl TestServer {
        pub(crate) fn spawn(behavior: Behavior) -> TestServer {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            listener.set_nonblocking(true).unwrap();
            let url = format!("ws://{}/vc", listener.local_addr().unwrap());
            let log = Arc::new(Mutex::new(Log::default()));
            let stop = Arc::new(AtomicBool::new(false));
            let (log2, stop2) = (log.clone(), stop.clone());
            thread::spawn(move || {
                let closed_once = Arc::new(AtomicBool::new(false));
                while !stop2.load(Ordering::Acquire) {
                    match listener.accept() {
                        Ok((tcp, _)) => {
                            let (b, l, s, c) = (behavior.clone(), log2.clone(), stop2.clone(), closed_once.clone());
                            thread::spawn(move || serve(tcp, b, l, s, c));
                        }
                        Err(_) => thread::sleep(Duration::from_millis(5)),
                    }
                }
            });
            TestServer { url, log, stop }
        }

        pub(crate) fn log(&self) -> Log {
            self.log.lock().unwrap().clone()
        }
    }

    // tungstenite's header callback returns its `ErrorResponse` by value.
    #[allow(clippy::result_large_err)]
    fn serve(tcp: std::net::TcpStream, b: Behavior, log: Arc<Mutex<Log>>, stop: Arc<AtomicBool>, closed_once: Arc<AtomicBool>) {
        tcp.set_nonblocking(false).unwrap();
        tcp.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
        let key = b.expect_key.clone();
        let check = move |req: &Request, resp: Response| -> Result<Response, ErrorResponse> {
            let got = req.headers().get("authorization").and_then(|v| v.to_str().ok()).map(str::to_string);
            match &key {
                Some(k) if got.as_deref() != Some(&format!("Bearer {k}")) => {
                    let mut e = ErrorResponse::new(Some("bad key".into()));
                    *e.status_mut() = tungstenite::http::StatusCode::UNAUTHORIZED;
                    Err(e)
                }
                _ => Ok(resp),
            }
        };
        let Ok(mut ws) = tungstenite::accept_hdr(tcp, check) else { return };
        log.lock().unwrap().connections += 1;
        match ws.read() {
            Ok(Message::Text(t)) => log.lock().unwrap().hello = serde_json::from_str(&t).ok(),
            _ => return,
        }
        if ws.send(Message::Text(b.welcome.to_string())).is_err() {
            return;
        }
        let wants_reference = b.welcome.get("reference_required").and_then(Value::as_bool).unwrap_or(false)
            || b.welcome.get("reference_needed").and_then(Value::as_bool).unwrap_or(false);
        let hello_has_reference = log.lock().unwrap().hello.as_ref().is_some_and(|h| !h["reference"].is_null());
        if wants_reference && hello_has_reference {
            loop {
                match ws.read() {
                    Ok(Message::Text(t)) => log.lock().unwrap().reference_meta = serde_json::from_str(&t).ok(),
                    Ok(Message::Binary(f)) => {
                        let (h, pcm) = decode_frame(&f).unwrap();
                        assert_eq!(h.kind, KIND_REFERENCE);
                        let mut l = log.lock().unwrap();
                        l.reference.extend(pcm);
                        l.reference_chunks += 1;
                        if h.flags & FLAG_LAST != 0 {
                            break;
                        }
                    }
                    _ => return,
                }
            }
            thread::sleep(b.reference_delay);
            if ws.send(Message::Text(json!({"type": "reference_ready"}).to_string())).is_err() {
                return;
            }
        }
        // (due, seq, flags, reply pcm), in due order for v2 and arrival order for v1.
        let mut pending: VecDeque<(Instant, u32, Vec<f32>)> = VecDeque::new();
        let mut frames = 0usize;
        while !stop.load(Ordering::Acquire) {
            let wait = pending
                .front()
                .map(|p| p.0.saturating_duration_since(Instant::now()))
                .unwrap_or(Duration::from_millis(20))
                .clamp(Duration::from_millis(1), Duration::from_millis(20));
            let _ = ws.get_ref().set_read_timeout(Some(wait));
            match ws.read() {
                Ok(Message::Binary(f)) => {
                    frames += 1;
                    let (seq, pcm) = if b.v1 {
                        let pcm = decode_pcm_chunk(&f).unwrap();
                        let mut l = log.lock().unwrap();
                        let seq = l.frames.len() as u32;
                        l.frames.push((seq, 0, 0.0, pcm.len()));
                        (seq, pcm)
                    } else {
                        let (h, pcm) = decode_frame(&f).unwrap();
                        log.lock().unwrap().frames.push((h.seq, h.flags, h.pitch, pcm.len()));
                        (h.seq, pcm)
                    };
                    if !b.v1 && (b.superseded)(seq) {
                        let msg = json!({"type": "dropped", "seq": seq, "reason": "superseded"});
                        if ws.send(Message::Text(msg.to_string())).is_err() {
                            return;
                        }
                    } else if let Some(d) = (b.delay)(seq) {
                        let n_out = pcm.len() * b.ratio.0 / b.ratio.1;
                        let reply: Vec<f32> = (0..n_out).map(|i| pcm[i * b.ratio.1 / b.ratio.0]).collect();
                        let due = Instant::now() + d;
                        let at = if b.v1 { pending.len() } else { pending.iter().position(|p| p.0 > due).unwrap_or(pending.len()) };
                        pending.insert(at, (due, seq, reply));
                    }
                    if b.close_after_frames == Some(frames) && !closed_once.swap(true, Ordering::AcqRel) {
                        let _ = ws.close(None);
                        let _ = ws.flush();
                        return;
                    }
                }
                Ok(Message::Text(t)) => {
                    let v: Value = serde_json::from_str(&t).unwrap_or(Value::Null);
                    let mut l = log.lock().unwrap();
                    match v["type"].as_str() {
                        Some("params") => l.params.push(v["params"].clone()),
                        Some("cancel") => {
                            let before = v["before_seq"].as_u64().unwrap_or(0) as u32;
                            l.cancels.push(before);
                            pending.retain(|p| p.1 >= before);
                        }
                        _ => {}
                    }
                }
                Ok(Message::Close(_)) => return,
                Ok(_) => {}
                Err(tungstenite::Error::Io(e)) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
                Err(_) => return,
            }
            let now = Instant::now();
            while pending.front().is_some_and(|p| p.0 <= now) {
                let (_, seq, pcm) = pending.pop_front().unwrap();
                let msg = if b.v1 {
                    encode_pcm_chunk(&pcm)
                } else {
                    encode_frame(&FrameHeader { kind: KIND_AUDIO_OUT, flags: 0, seq, pitch: 0.0 }, &pcm)
                };
                if ws.send(Message::Binary(msg)).is_err() {
                    return;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::sync::mpsc;
    use std::thread;

    #[test]
    fn pcm_chunk_roundtrip() {
        let v = vec![0.25f32, -0.5, 1.0];
        let enc = encode_pcm_chunk(&v);
        assert_eq!(decode_pcm_chunk(&enc).unwrap(), v);
    }

    /// A local v1 sidecar that answers the handshake, then echoes chunks after the given delays
    /// (one per chunk; missing entries = no delay). Returns the ws:// URL.
    fn spawn_sidecar(delays_ms: Vec<u64>) -> (String, thread::JoinHandle<()>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("ws://{}", listener.local_addr().unwrap());
        let h = thread::spawn(move || {
            let (tcp, _) = listener.accept().unwrap();
            let mut ws = tungstenite::accept(tcp).unwrap();
            match ws.read().unwrap() {
                Message::Text(_) => ws.send(Message::Text(r#"{"ok":true}"#.into())).unwrap(),
                other => panic!("expected handshake, got {other:?}"),
            }
            let mut i = 0;
            loop {
                match ws.read() {
                    Ok(Message::Binary(b)) => {
                        let d = delays_ms.get(i).copied().unwrap_or(0);
                        i += 1;
                        thread::sleep(Duration::from_millis(d));
                        if ws.send(Message::Binary(b)).is_err() {
                            return;
                        }
                    }
                    Ok(_) => {}
                    Err(_) => return,
                }
            }
        });
        (url, h)
    }

    /// The exchange is bounded by the timeout, and after a timeout the late reply is discarded
    /// instead of being taken as the answer to the next chunk.
    #[test]
    fn exchange_times_out_and_resynchronizes() {
        // Chunk 1 is answered after 400 ms (> 150 ms timeout); later chunks at once.
        let (url, server) = spawn_sidecar(vec![400]);
        let mut bridge = SidecarVoiceBridge::connect(&url, 48_000, 1, 4, Duration::from_millis(150)).unwrap();
        let t0 = Instant::now();
        let err = bridge.exchange(&[1.0, 1.0, 1.0, 1.0]).unwrap_err();
        assert!(err.contains("did not answer"), "{err}");
        assert!(t0.elapsed() < Duration::from_millis(350), "bounded by the timeout: {:?}", t0.elapsed());
        assert_eq!(bridge.outstanding_replies(), 1);
        // Wait for the late reply to be in flight, then exchange chunk 2: the stale chunk-1 reply
        // (all ones) is skipped and chunk 2's own reply is returned.
        thread::sleep(Duration::from_millis(400));
        let out = bridge.exchange(&[2.0, 2.0, 2.0, 2.0]).unwrap();
        assert_eq!(out, vec![2.0, 2.0, 2.0, 2.0]);
        assert_eq!(bridge.outstanding_replies(), 0);
        let out = bridge.exchange(&[3.0, 3.0, 3.0, 3.0]).unwrap();
        assert_eq!(out, vec![3.0, 3.0, 3.0, 3.0]);
        drop(bridge);
        server.join().unwrap();
    }

    /// Any first text or binary message acknowledges the handshake (the v1 docs never fixed its
    /// format); only an explicit `{"ok": false}` refuses, with the sidecar's reason.
    #[test]
    fn handshake_accepts_any_ack_but_an_explicit_refusal() {
        for accepted in [
            Message::Text(r#"{"ok":true,"mode":"echo"}"#.into()),
            Message::Text(r#"{"status":"ready"}"#.into()),
            Message::Text("ok".into()),
            Message::Text(r#"{"ok":"false"}"#.into()),
            Message::Binary(vec![1, 2, 3]),
            Message::Binary(Vec::new()),
        ] {
            assert_eq!(ack_verdict(&accepted), AckVerdict::Accepted, "{accepted:?}");
        }
        assert_eq!(ack_verdict(&Message::Ping(vec![])), AckVerdict::Continue);
        assert_eq!(
            ack_verdict(&Message::Text(r#"{"ok":false}"#.into())),
            AckVerdict::Rejected("sidecar handshake rejected (ok: false)".into())
        );
        assert_eq!(
            ack_verdict(&Message::Text(r#"{"ok":false,"error":"model not loaded"}"#.into())),
            AckVerdict::Rejected("sidecar handshake rejected (ok: false): model not loaded".into())
        );
    }

    /// A sidecar that acknowledges in its own format (here a status object, then a binary frame)
    /// connects and exchanges as before; an explicit refusal fails the connect with its reason.
    #[test]
    fn connect_accepts_foreign_acks_and_reports_refusals() {
        let serve = |ack: Message| {
            let listener = TcpListener::bind("127.0.0.1:0").unwrap();
            let url = format!("ws://{}", listener.local_addr().unwrap());
            let h = thread::spawn(move || {
                let (tcp, _) = listener.accept().unwrap();
                let mut ws = tungstenite::accept(tcp).unwrap();
                assert!(matches!(ws.read().unwrap(), Message::Text(_)));
                ws.send(ack).unwrap();
                while let Ok(m) = ws.read() {
                    if let Message::Binary(b) = m {
                        if ws.send(Message::Binary(b)).is_err() {
                            return;
                        }
                    }
                }
            });
            (url, h)
        };
        for ack in [Message::Text(r#"{"status":"ready"}"#.into()), Message::Binary(vec![0; 4])] {
            let (url, server) = serve(ack);
            let mut bridge = SidecarVoiceBridge::connect(&url, 48_000, 1, 2, Duration::from_millis(500)).unwrap();
            assert_eq!(bridge.exchange(&[0.5, -0.5]).unwrap(), vec![0.5, -0.5]);
            drop(bridge);
            server.join().unwrap();
        }
        let (url, server) = serve(Message::Text(r#"{"ok":false,"message":"busy"}"#.into()));
        let err = SidecarVoiceBridge::connect(&url, 48_000, 1, 2, Duration::from_millis(500)).err().unwrap();
        assert_eq!(err, "sidecar handshake rejected (ok: false): busy");
        server.join().unwrap();
    }

    /// Connecting to a port nobody listens on fails promptly with a message (no hang).
    #[test]
    fn connect_failure_is_prompt() {
        let port = {
            let l = TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let (tx, rx) = mpsc::channel();
        thread::spawn(move || {
            let r = SidecarVoiceBridge::connect(&format!("ws://127.0.0.1:{port}"), 48_000, 1, 4, Duration::from_millis(100));
            let _ = tx.send(r.err());
        });
        let err = rx.recv_timeout(Duration::from_secs(10)).expect("connect returned").expect("an error");
        assert!(err.contains("sidecar"), "{err}");
        assert!(SidecarVoiceBridge::connect("http://127.0.0.1:1", 48_000, 1, 4, Duration::from_millis(100)).is_err());
    }
}
