//! WebSocket client for neural voice conversion sidecar (synchronous PCM exchange per chunk).
//!
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
