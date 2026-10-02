//! Edge client tests against a local mock of the Read-aloud service (plain `ws://`, no network).

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use tungstenite::handshake::server::{Request, Response};
use tungstenite::protocol::frame::coding::CloseCode;
use tungstenite::protocol::CloseFrame;
use tungstenite::Message;

use super::cloud::{FailureKind, SpeakError};
use super::protocol::{self, sec_ms_gec};
use super::session::Clock;
use super::*;
use crate::services::openai_speech::testing::{self as http_testing, Recorder};
use crate::services::tts_sink::SinkError;

// ---------------------------------------------------------------------------------------------
// Mock service
// ---------------------------------------------------------------------------------------------

/// A silent MPEG-2 Layer III frame: 24 kHz mono, 48 kbit/s (144 bytes, 576 samples), the stream
/// format the service uses. All-zero side info decodes to silence.
fn silent_frame() -> Vec<u8> {
    let mut f = vec![0u8; 144];
    f[..4].copy_from_slice(&[0xFF, 0xF3, 0x64, 0xC0]);
    f
}

fn mp3(frames: usize) -> Vec<u8> {
    (0..frames).flat_map(|_| silent_frame()).collect()
}

/// A binary audio message as the service frames it.
fn audio_message(data: &[u8], content_type: Option<&str>) -> Vec<u8> {
    let mut head = String::from("X-RequestId:abc\r\n");
    if let Some(ct) = content_type {
        head.push_str(&format!("Content-Type:{ct}\r\n"));
    }
    head.push_str("Path:audio\r\n");
    let mut msg = (head.len() as u16).to_be_bytes().to_vec();
    msg.extend_from_slice(head.as_bytes());
    msg.extend_from_slice(data);
    msg
}

fn text_message(path: &str, body: &str) -> String {
    format!("X-RequestId:abc\r\nContent-Type:application/json; charset=utf-8\r\nPath:{path}\r\n\r\n{body}")
}

/// What one connection of the mock does after the upgrade.
#[derive(Clone)]
enum Script {
    /// Answer with `frames` MP3 frames in `chunks` messages, then `turn.end`.
    Speak { frames: usize, chunks: usize, pause: Duration },
    /// Refuse the upgrade with an HTTP status (and extra header lines).
    Http { status: u16, headers: String },
    /// Read the request and never answer.
    Silent(Duration),
    /// Read the request and close with a code.
    Close(CloseCode, &'static str),
    /// Send some audio, then go silent.
    StallAfter { frames: usize, hold: Duration },
}

/// What the mock saw on one connection.
#[derive(Clone, Debug, Default)]
struct Seen {
    uri: String,
    headers: Vec<(String, String)>,
    texts: Vec<String>,
}

impl Seen {
    fn header(&self, name: &str) -> Option<String> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.clone())
    }
    fn query(&self, key: &str) -> Option<String> {
        let q = self.uri.split_once('?')?.1;
        q.split('&').find_map(|kv| kv.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v.to_string()))
    }
}

/// Serve one connection per script entry, in order; returns the base URL and what was seen.
fn mock(scripts: Vec<Script>) -> (String, Arc<Mutex<Vec<Seen>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("ws://{}/consumer/speech/synthesize/readaloud/edge/v1", listener.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&seen);
    thread::spawn(move || {
        for script in scripts {
            let Ok((stream, _)) = listener.accept() else { return };
            let log = Arc::clone(&log);
            thread::spawn(move || serve(stream, script, log));
        }
    });
    (base, seen)
}

// The handshake callback's error type is tungstenite's (an HTTP response).
#[allow(clippy::result_large_err)]
fn serve(mut stream: TcpStream, script: Script, log: Arc<Mutex<Vec<Seen>>>) {
    if let Script::Http { status, headers } = &script {
        let mut buf = Vec::new();
        let mut tmp = [0u8; 1024];
        while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
            match stream.read(&mut tmp) {
                Ok(0) | Err(_) => break,
                Ok(n) => buf.extend_from_slice(&tmp[..n]),
            }
        }
        let head = String::from_utf8_lossy(&buf).to_string();
        let uri = head.split_whitespace().nth(1).unwrap_or("").to_string();
        log.lock().unwrap().push(Seen { uri, ..Seen::default() });
        let _ = stream.write_all(format!("HTTP/1.1 {status} Nope\r\n{headers}Content-Length: 0\r\nConnection: close\r\n\r\n").as_bytes());
        return;
    }
    let mut seen = Seen::default();
    let cb = |req: &Request, resp: Response| {
        seen.uri = req.uri().to_string();
        seen.headers = req
            .headers()
            .iter()
            .map(|(k, v)| (k.as_str().to_string(), v.to_str().unwrap_or("").to_string()))
            .collect();
        Ok(resp)
    };
    let Ok(mut ws) = tungstenite::accept_hdr(stream, cb) else { return };
    // The two request messages.
    while seen.texts.len() < 2 {
        match ws.read() {
            Ok(Message::Text(t)) => seen.texts.push(t),
            Ok(_) => {}
            Err(_) => break,
        }
    }
    log.lock().unwrap().push(seen);
    match script {
        Script::Speak { frames, chunks, pause } => {
            let _ = ws.send(Message::Text(text_message("turn.start", "{}")));
            let data = mp3(frames);
            let per = data.len().div_ceil(chunks.max(1));
            for (i, part) in data.chunks(per).enumerate() {
                if i == 1 {
                    let _ = ws.send(Message::Text(text_message("audio.metadata", "{\"Metadata\":[]}")));
                }
                let _ = ws.send(Message::Binary(audio_message(part, Some("audio/mpeg"))));
                if !pause.is_zero() {
                    thread::sleep(pause);
                }
            }
            let _ = ws.send(Message::Binary(audio_message(&[], None)));
            let _ = ws.send(Message::Text(text_message("turn.end", "{}")));
            // Wait for the client to hang up.
            let _ = ws.read();
        }
        Script::Silent(hold) => thread::sleep(hold),
        Script::Close(code, reason) => {
            let _ = ws.close(Some(CloseFrame { code, reason: reason.into() }));
            let _ = ws.flush();
            thread::sleep(Duration::from_millis(200));
        }
        Script::StallAfter { frames, hold } => {
            let _ = ws.send(Message::Binary(audio_message(&mp3(frames), Some("audio/mpeg"))));
            thread::sleep(hold);
        }
        Script::Http { .. } => unreachable!(),
    }
}

/// A clock frozen at 2026-10-01 12:00:00 UTC that records adjustments.
struct FakeClock {
    now: AtomicI64,
    adjusted: Mutex<Vec<i64>>,
}

impl FakeClock {
    fn new() -> Arc<Self> {
        Arc::new(Self { now: AtomicI64::new(cloud::days_from_civil(2026, 10, 1) * 86_400 + 12 * 3600), adjusted: Mutex::default() })
    }
}

impl Clock for FakeClock {
    fn now(&self) -> i64 {
        self.now.load(Ordering::SeqCst)
    }
    fn adjust(&self, server_unix: i64) {
        self.adjusted.lock().unwrap().push(server_unix);
        self.now.store(server_unix, Ordering::SeqCst);
    }
}

fn endpoint(wss: &str) -> Endpoint {
    Endpoint { wss: wss.to_string(), voices: String::new() }
}

fn request(text: &str, timeout_ms: u32) -> EdgeTtsRequest {
    EdgeTtsRequest { text: text.into(), voice: "ja-JP-NanamiNeural".into(), rate: 1.2, pitch_hz: -10, timeout_ms }
}

fn failure(r: Result<SpeakReport, SpeakError>) -> cloud::Failure {
    match r {
        Err(SpeakError::Failed(f)) => f,
        other => panic!("expected a failure, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_streams_mp3_into_the_sink_with_the_browser_handshake() {
    let (base, seen) = mock(vec![Script::Speak { frames: 50, chunks: 5, pause: Duration::ZERO }]);
    let (rec, pushes) = Recorder::new();
    let clock = FakeClock::new();
    let report = speak_with(&request("こんにちは <世界> & you", 3000), "edge-ok", &endpoint(&base), Box::new(rec), clock.clone())
        .await
        .unwrap();
    assert_eq!(report.sample_rate, 24_000);
    // 50 frames of 576 samples; the decoder may hold back at most one frame.
    assert!((49 * 576..=50 * 576).contains(&(report.frames as usize)), "{}", report.frames);
    assert_eq!(pushes.samples().len() as u64, report.frames);
    assert!(pushes.formats().iter().all(|f| *f == (24_000, 1)));
    assert!(pushes.samples().iter().all(|s| s.abs() < 1e-3), "silence decodes to silence");

    let seen = seen.lock().unwrap()[0].clone();
    assert_eq!(seen.query("TrustedClientToken").as_deref(), Some(protocol::TRUSTED_CLIENT_TOKEN));
    assert_eq!(seen.query("Sec-MS-GEC").unwrap(), sec_ms_gec(clock.now()));
    assert_eq!(seen.query("Sec-MS-GEC-Version").unwrap(), protocol::sec_ms_gec_version());
    assert_eq!(seen.query("ConnectionId").unwrap().len(), 32);
    assert_eq!(seen.header("origin").as_deref(), Some(protocol::ORIGIN));
    assert!(seen.header("user-agent").unwrap().contains("Edg/"));
    assert_eq!(seen.header("pragma").as_deref(), Some("no-cache"));
    let cookie = seen.header("cookie").unwrap();
    assert!(cookie.starts_with("muid=") && cookie.len() == "muid=;".len() + 32, "{cookie}");

    assert!(seen.texts[0].contains("Path:speech.config"));
    assert!(seen.texts[0].contains(protocol::OUTPUT_FORMAT));
    let ssml = &seen.texts[1];
    assert!(ssml.contains("Path:ssml"));
    assert!(ssml.contains("<voice name='Microsoft Server Speech Text to Speech Voice (ja-JP, NanamiNeural)'>"));
    assert!(ssml.contains("rate='+20%'") && ssml.contains("pitch='-10Hz'"));
    assert!(ssml.contains("こんにちは &lt;世界&gt; &amp; you"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_first_audio_arrives_before_the_sentence_finished() {
    // Five chunks 150 ms apart: the first push comes long before the last chunk.
    let (base, _) = mock(vec![Script::Speak { frames: 60, chunks: 5, pause: Duration::from_millis(150) }]);
    let (rec, _) = Recorder::new();
    let report = speak_with(&request("Hello there.", 3000), "edge-stream", &endpoint(&base), Box::new(rec), FakeClock::new())
        .await
        .unwrap();
    assert!(report.first_audio_ms + 300 < report.total_ms, "{report:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_outages_fail_fast_with_their_kind() {
    for (status, kind, id) in [(503, FailureKind::Unavailable, "edge-503"), (429, FailureKind::RateLimited, "edge-429"), (400, FailureKind::Rejected, "edge-400")] {
        let (base, _) = mock(vec![Script::Http { status, headers: "Retry-After: 7\r\n".into() }]);
        let (rec, _) = Recorder::new();
        let t = Instant::now();
        let f = failure(speak_with(&request("Hi.", 3000), id, &endpoint(&base), Box::new(rec), FakeClock::new()).await);
        assert_eq!(f.kind, kind, "{status}: {f:?}");
        assert!(f.message.contains(&status.to_string()), "{}", f.message);
        assert_eq!(f.retry_after, Some(Duration::from_secs(7)));
        assert!(!f.partial);
        assert!(f.to_wire().starts_with(&format!("[cloud-tts:{}", kind.wire())));
        assert!(t.elapsed() < Duration::from_secs(1), "an outage is reported at once");
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_no_listener_is_a_network_failure() {
    let port = TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port();
    let (rec, _) = Recorder::new();
    let t = Instant::now();
    let f = failure(
        speak_with(&request("Hi.", 3000), "edge-down", &endpoint(&format!("ws://127.0.0.1:{port}/edge/v1")), Box::new(rec), FakeClock::new())
            .await,
    );
    assert_eq!(f.kind, FailureKind::Network, "{f:?}");
    // Windows retries a refused connect for about 2 s; the 1 s connect bound (3 s request) ends it
    // first, so the local voice can take over within about a second.
    assert!(t.elapsed() < Duration::from_millis(1500), "{:?}", t.elapsed());
}

#[test]
fn edge_tts_connect_bounds_scale_with_the_timeout() {
    use super::session::connect_bound;
    assert_eq!(connect_bound(Duration::from_secs(3)), Duration::from_secs(1));
    assert_eq!(connect_bound(Duration::from_millis(500)), Duration::from_millis(500));
    assert_eq!(connect_bound(Duration::from_secs(10)).as_millis(), 3333);
    assert_eq!(connect_bound(AUTO_TIMEOUT), Duration::from_secs(5));
    let p = plan(&request("Hi.", 3000), &endpoint("ws://127.0.0.1:9/x")).unwrap().unwrap();
    assert_eq!((p.first_audio, p.connect), (Duration::from_secs(3), Duration::from_secs(1)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_a_silent_service_stalls_within_the_timeout() {
    let (base, _) = mock(vec![Script::Silent(Duration::from_secs(5))]);
    let (rec, _) = Recorder::new();
    let t = Instant::now();
    let f = failure(speak_with(&request("Hi.", 600), "edge-silent", &endpoint(&base), Box::new(rec), FakeClock::new()).await);
    let took = t.elapsed();
    assert_eq!(f.kind, FailureKind::Stalled, "{f:?}");
    assert!(!f.partial);
    assert!(took >= Duration::from_millis(550) && took < Duration::from_millis(1500), "{took:?}");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_a_stall_after_audio_is_partial() {
    let (base, _) = mock(vec![Script::StallAfter { frames: 20, hold: Duration::from_secs(5) }]);
    let (rec, pushes) = Recorder::new();
    let f = failure(speak_with(&request("Hi.", 600), "edge-stall", &endpoint(&base), Box::new(rec), FakeClock::new()).await);
    assert_eq!(f.kind, FailureKind::Stalled, "{f:?}");
    assert!(f.partial, "audio was already pushed");
    assert!(pushes.count() > 0);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_a_refused_voice_closes_as_rejected() {
    let (base, _) = mock(vec![Script::Close(CloseCode::Invalid, "Unsupported voice")]);
    let (rec, _) = Recorder::new();
    let f = failure(speak_with(&request("Hi.", 3000), "edge-close", &endpoint(&base), Box::new(rec), FakeClock::new()).await);
    assert_eq!(f.kind, FailureKind::Rejected, "{f:?}");
    assert!(f.message.contains("Unsupported voice"));

    let (base, _) = mock(vec![Script::Close(CloseCode::Error, "")]);
    let (rec, _) = Recorder::new();
    let f = failure(speak_with(&request("Hi.", 3000), "edge-close2", &endpoint(&base), Box::new(rec), FakeClock::new()).await);
    assert_eq!(f.kind, FailureKind::Unavailable, "{f:?}");

    // A refusal of this sentence's text (its payload or size) is about that sentence only: `no_audio`,
    // which never opens the breaker (a `rejected` would keep every sentence on the local voice).
    let (base, _) = mock(vec![Script::Close(CloseCode::Size, "Message too big")]);
    let (rec, _) = Recorder::new();
    let f = failure(speak_with(&request("Hi.", 3000), "edge-close3", &endpoint(&base), Box::new(rec), FakeClock::new()).await);
    assert_eq!(f.kind, FailureKind::NoAudio, "{f:?}");
    assert!(f.message.contains("could not say this sentence"), "{}", f.message);
}

#[test]
fn edge_tts_closes_are_classified_by_code_and_reason() {
    use super::session::close_kind;
    // The voice of the setup was refused: every sentence would fail the same way.
    assert_eq!(close_kind(Some(CloseCode::Invalid), "Unsupported voice"), FailureKind::Rejected);
    assert_eq!(close_kind(Some(CloseCode::Policy), "Voice not available"), FailureKind::Rejected);
    // The request shape is not supported: the service changed.
    assert_eq!(close_kind(Some(CloseCode::Unsupported), ""), FailureKind::Rejected);
    // This sentence's text or its size: that sentence only.
    assert_eq!(close_kind(Some(CloseCode::Invalid), "Invalid SSML"), FailureKind::NoAudio);
    assert_eq!(close_kind(Some(CloseCode::Invalid), ""), FailureKind::NoAudio);
    assert_eq!(close_kind(Some(CloseCode::Size), "Message too big"), FailureKind::NoAudio);
    // Throttling.
    assert_eq!(close_kind(Some(CloseCode::Policy), "Quota exceeded"), FailureKind::RateLimited);
    assert_eq!(close_kind(Some(CloseCode::Policy), "Too many requests"), FailureKind::RateLimited);
    // Anything else is the service's trouble (backoff).
    assert_eq!(close_kind(Some(CloseCode::Policy), ""), FailureKind::Unavailable);
    assert_eq!(close_kind(Some(CloseCode::Error), "Internal error"), FailureKind::Unavailable);
    assert_eq!(close_kind(Some(CloseCode::Away), ""), FailureKind::Unavailable);
    assert_eq!(close_kind(None, ""), FailureKind::Unavailable);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_sink_failures_are_this_pc_s_not_the_service_s() {
    // The sink refuses a push (its queue timed out while the voice-changer route stopped reading):
    // untagged, with the sink's own text, so the TS chain neither blames Edge nor retries locally.
    let (base, _) = mock(vec![Script::Speak { frames: 50, chunks: 5, pause: Duration::ZERO }]);
    let (mut rec, _) = Recorder::new();
    rec.fail_with = Some(SinkError::Unavailable("timed out waiting for queued speech to play".into()));
    let r = speak_with(&request("Hi.", 3000), "edge-sinkfail", &endpoint(&base), Box::new(rec), FakeClock::new()).await;
    assert_eq!(r, Err(SpeakError::Local("TTS sink unavailable: timed out waiting for queued speech to play".into())));
    let wire = r.unwrap_err().into_wire();
    assert!(!wire.starts_with(cloud::TAG_PREFIX), "{wire}");

    // A second request for an utterance already being spoken: the caller's mistake, untagged too.
    let held = REGISTRY.register("edge-dup").unwrap();
    let (rec, _) = Recorder::new();
    let r = speak_with(&request("Hi.", 3000), "edge-dup", &endpoint("ws://127.0.0.1:9/x"), Box::new(rec), FakeClock::new()).await;
    assert!(matches!(&r, Err(SpeakError::Local(m)) if m.contains("already being spoken")), "{r:?}");
    drop(held);
}

// ---------------------------------------------------------------------------------------------
// Dual-stack hosts (Happy Eyeballs)
// ---------------------------------------------------------------------------------------------

#[test]
fn edge_tts_addresses_alternate_families_starting_with_the_preferred_one() {
    use super::session::interleave_families;
    let a6: SocketAddr = "[2620:1ec:33::10]:443".parse().unwrap();
    let b6: SocketAddr = "[2620:1ec:33:1::10]:443".parse().unwrap();
    let a4: SocketAddr = "150.171.27.10:443".parse().unwrap();
    let b4: SocketAddr = "150.171.28.10:443".parse().unwrap();
    assert_eq!(interleave_families(vec![a6, b6, a4, b4]), vec![a6, a4, b6, b4]);
    assert_eq!(interleave_families(vec![a4, a6, b6]), vec![a4, a6, b6]);
    assert_eq!(interleave_families(vec![a4, b4]), vec![a4, b4]);
    assert!(interleave_families(Vec::new()).is_empty());
}

/// A connect function for [`session::race_connect_with`]: the `dead` addresses never answer (a
/// blackholed path: the attempt hangs until its time is up), `refused` ones fail at once, the others
/// connect for real.
fn connector(dead: Vec<SocketAddr>, refused: Vec<SocketAddr>) -> impl Fn(SocketAddr, Duration) -> std::io::Result<TcpStream> + Clone + Send + 'static {
    move |addr, left| {
        if dead.contains(&addr) {
            thread::sleep(left);
            Err(std::io::Error::new(std::io::ErrorKind::TimedOut, "no answer"))
        } else if refused.contains(&addr) {
            Err(std::io::Error::new(std::io::ErrorKind::ConnectionRefused, "refused"))
        } else {
            TcpStream::connect_timeout(&addr, left)
        }
    }
}

#[test]
fn edge_tts_a_dead_first_address_does_not_use_up_the_connect_bound() {
    use super::session::{race_connect_with, RaceFail, ATTEMPT_DELAY};
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let alive = listener.local_addr().unwrap();
    // The IPv6 address the system prefers, on a PC whose IPv6 path drops everything (a VPN with
    // IPv6 leak protection, a router with broken IPv6).
    let dead: SocketAddr = "[2001:db8::1]:443".parse().unwrap();
    let t = Instant::now();
    let stream = race_connect_with(&[dead, alive], t + Duration::from_secs(1), connector(vec![dead], vec![])).unwrap();
    let took = t.elapsed();
    assert_eq!(stream.peer_addr().unwrap(), alive);
    // The next address started after the attempt delay, not after the whole bound.
    assert!(took >= ATTEMPT_DELAY - Duration::from_millis(20) && took < Duration::from_millis(700), "{took:?}");

    // A refused address starts the next one at once.
    let refused: SocketAddr = "[2001:db8::3]:443".parse().unwrap();
    let t = Instant::now();
    let stream = race_connect_with(&[refused, alive], t + Duration::from_secs(1), connector(vec![], vec![refused])).unwrap();
    assert_eq!(stream.peer_addr().unwrap(), alive);
    assert!(t.elapsed() < ATTEMPT_DELAY, "{:?}", t.elapsed());

    // Every address dead: given up at the bound, not later.
    let dead2: SocketAddr = "[2001:db8::2]:443".parse().unwrap();
    let t = Instant::now();
    let r = race_connect_with(&[dead, dead2], t + Duration::from_millis(600), connector(vec![dead, dead2], vec![]));
    let timed_out = match &r {
        Err(RaceFail::TimedOut) => true,
        Err(RaceFail::Failed(e)) => e.kind() == std::io::ErrorKind::TimedOut,
        _ => false,
    };
    assert!(timed_out, "{r:?}");
    assert!(t.elapsed() < Duration::from_millis(800), "{:?}", t.elapsed());

    // Every address refused, or none at all.
    let r = race_connect_with(&[refused], Instant::now() + Duration::from_secs(1), connector(vec![], vec![refused]));
    assert!(matches!(&r, Err(RaceFail::Failed(e)) if e.kind() == std::io::ErrorKind::ConnectionRefused), "{r:?}");
    assert!(matches!(race_connect_with(&[], Instant::now() + Duration::from_secs(1), connector(vec![], vec![])), Err(RaceFail::NoAddress)));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_a_dual_stack_host_with_one_dead_family_still_connects() {
    // `localhost` resolves to ::1 and 127.0.0.1 on Windows (IPv6 first); the mock listens on IPv4
    // only, so the IPv6 attempt is refused, which Windows retries for about 2 s (longer than the
    // 1 s connect bound). Racing the addresses starts IPv4 250 ms later and the request succeeds.
    let (base, seen) = mock(vec![Script::Speak { frames: 10, chunks: 1, pause: Duration::ZERO }]);
    let base = base.replace("127.0.0.1", "localhost");
    let (rec, _) = Recorder::new();
    let t = Instant::now();
    let report = speak_with(&request("Hi.", 3000), "edge-dual", &endpoint(&base), Box::new(rec), FakeClock::new()).await.unwrap();
    assert!(report.frames > 0, "{report:?}");
    assert!(t.elapsed() < Duration::from_millis(1500), "{:?}", t.elapsed());
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_retries_once_with_the_server_clock_on_403() {
    let server_date = "Thu, 01 Oct 2026 12:07:30 GMT";
    let (base, seen) = mock(vec![
        Script::Http { status: 403, headers: format!("Date: {server_date}\r\n") },
        Script::Speak { frames: 10, chunks: 1, pause: Duration::ZERO },
    ]);
    let (rec, _) = Recorder::new();
    let clock = FakeClock::new();
    speak_with(&request("Hi.", 3000), "edge-403", &endpoint(&base), Box::new(rec), clock.clone()).await.unwrap();
    let server = cloud::parse_http_date(server_date).unwrap();
    assert_eq!(*clock.adjusted.lock().unwrap(), vec![server]);
    {
        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 2);
        assert_eq!(seen[1].query("Sec-MS-GEC").unwrap(), sec_ms_gec(server));
        assert_ne!(seen[0].query("Sec-MS-GEC"), seen[1].query("Sec-MS-GEC"));
    }

    // Refused twice: an auth failure (not retried forever).
    let (base, seen) = mock(vec![
        Script::Http { status: 403, headers: String::new() },
        Script::Http { status: 403, headers: String::new() },
    ]);
    let (rec, _) = Recorder::new();
    let f = failure(speak_with(&request("Hi.", 3000), "edge-403b", &endpoint(&base), Box::new(rec), FakeClock::new()).await);
    assert_eq!(f.kind, FailureKind::Auth);
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_cancel_ends_a_waiting_request_at_once() {
    let (base, _) = mock(vec![Script::Silent(Duration::from_secs(5))]);
    let (rec, _) = Recorder::new();
    let task = tokio::spawn(async move {
        speak_with(&request("Hi.", 10_000), "edge-cancel", &endpoint(&base), Box::new(rec), FakeClock::new()).await
    });
    tokio::time::sleep(Duration::from_millis(300)).await;
    let t = Instant::now();
    cancel_command(&REGISTRY, Some("edge-cancel"));
    let result = task.await.unwrap();
    assert_eq!(result, Err(SpeakError::Cancelled));
    assert!(t.elapsed() < Duration::from_millis(300));
    assert_eq!(SpeakError::Cancelled.into_wire(), "cancelled");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_a_cancelled_utterance_stops_the_stream() {
    let (base, _) = mock(vec![Script::Speak { frames: 200, chunks: 20, pause: Duration::from_millis(20) }]);
    let (mut rec, pushes) = Recorder::new();
    rec.cancel_after_first = true;
    let result = speak_with(&request("Hi.", 3000), "edge-sink-cancel", &endpoint(&base), Box::new(rec), FakeClock::new()).await;
    assert_eq!(result, Err(SpeakError::Cancelled));
    assert_eq!(pushes.count(), 1);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_long_text_is_spoken_in_several_requests() {
    let text = "word ".repeat(1500); // 7.5 KB: two requests
    let (base, seen) = mock(vec![
        Script::Speak { frames: 5, chunks: 1, pause: Duration::ZERO },
        Script::Speak { frames: 5, chunks: 1, pause: Duration::ZERO },
    ]);
    let (rec, _) = Recorder::new();
    let report = speak_with(&request(&text, 3000), "edge-long", &endpoint(&base), Box::new(rec), FakeClock::new()).await.unwrap();
    assert!(report.frames >= 8 * 576, "{report:?}");
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_empty_text_and_bad_voices() {
    let (rec, pushes) = Recorder::new();
    let report = speak_with(&request("   ", 3000), "edge-empty", &endpoint("ws://127.0.0.1:9/x"), Box::new(rec), FakeClock::new()).await.unwrap();
    assert_eq!(report.frames, 0);
    assert_eq!(pushes.count(), 0);

    let (rec, _) = Recorder::new();
    let mut r = request("Hi.", 3000);
    r.voice = "en-US-Ava'/><x".into();
    let f = failure(speak_with(&r, "edge-badvoice", &endpoint("ws://127.0.0.1:9/x"), Box::new(rec), FakeClock::new()).await);
    assert_eq!(f.kind, FailureKind::Rejected);
}

#[test]
fn edge_tts_timeouts_are_clamped() {
    assert_eq!(timing(0), (AUTO_TIMEOUT, AUTO_TIMEOUT));
    assert_eq!(timing(3000), (Duration::from_secs(3), Duration::from_secs(3)));
    assert_eq!(timing(1), (MIN_TIMEOUT, MIN_TIMEOUT));
    assert_eq!(timing(10_000_000), (MAX_TIMEOUT, MAX_TIMEOUT));
}

#[test]
fn edge_tts_request_wire_is_camel_case() {
    let r: EdgeTtsRequest = serde_json::from_value(serde_json::json!({ "text": "a", "voice": "v", "pitchHz": 5, "timeoutMs": 3000 })).unwrap();
    assert_eq!(r.pitch_hz, 5);
    assert_eq!(r.timeout_ms, 3000);
    assert_eq!(r.rate, 1.0);
    let v = serde_json::to_value(EdgeTtsVoice { short_name: "a".into(), ..Default::default() }).unwrap();
    assert!(v.get("shortName").is_some() && v.get("friendlyName").is_some());
}

#[test]
fn edge_tts_voice_list_parses_the_service_answer() {
    let v = serde_json::json!([
        { "Name": "Microsoft Server Speech Text to Speech Voice (ja-JP, NanamiNeural)", "ShortName": "ja-JP-NanamiNeural", "Gender": "Female", "Locale": "ja-JP", "FriendlyName": "Microsoft Nanami Online (Natural) - Japanese (Japan)" },
        { "ShortName": "en-US-AvaMultilingualNeural", "Gender": "Female", "Locale": "en-US", "FriendlyName": "Microsoft AvaMultilingual Online (Natural) - English (United States)" },
        { "ShortName": "en-US-AvaMultilingualNeural", "Gender": "Female", "Locale": "en-US" },
        { "ShortName": "", "Locale": "xx-XX" },
        { "ShortName": "de-DE-KatjaNeural", "Gender": "Female" }
    ]);
    let voices = voices::parse_voices(&v);
    let names: Vec<_> = voices.iter().map(|v| v.short_name.as_str()).collect();
    assert_eq!(names, ["de-DE-KatjaNeural", "en-US-AvaMultilingualNeural", "ja-JP-NanamiNeural"]);
    assert_eq!(voices[0].locale, "de-DE");
    assert_eq!(voices[2].gender, "Female");
    assert!(voices::parse_voices(&serde_json::json!({ "error": 1 })).is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn edge_tts_voice_list_request_looks_like_the_browser() {
    let body = br#"[{"ShortName":"en-US-AvaMultilingualNeural","Gender":"Female","Locale":"en-US","FriendlyName":"Ava"}]"#;
    let (base, task) = http_testing::serve_once(http_testing::full_response("200 OK", "application/json", body)).await;
    let voices = voices::fetch(&http_testing::client(), &format!("{base}/voices/list"), FakeClock::new().as_ref()).await.unwrap();
    assert_eq!(voices.len(), 1);
    let seen = task.await.unwrap();
    assert!(seen.path().starts_with("/voices/list?trustedclienttoken="));
    assert!(seen.path().contains("&Sec-MS-GEC="));
    assert!(seen.header("user-agent").unwrap().contains("Edg/"));
    assert!(seen.header("sec-ch-ua").unwrap().contains("Microsoft Edge"));
    assert!(seen.header("cookie").unwrap().starts_with("muid="));
    assert!(seen.header("accept-encoding").is_none_or(|v| !v.contains("br")));
}
