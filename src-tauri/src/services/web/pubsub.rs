//! `/pubsub?id=<peer>` WebSocket: JSON text events between the host webview and external pages
//! (mic.html, obs-captions.html, inputfield.html), Link apps and local tools.
//!
//! - Host → peers: `plugin:web|pubsub_broadcast` strings are sent to every connected peer.
//! - Peer → host: text frames are forwarded to the host webview as the `pubsub` event, where
//!   `consumePubSubMessage` validates the topic against its allow-list and re-broadcasts what it
//!   accepts. Peers are not relayed to each other directly, so nothing reaches another peer without
//!   passing the host's validation.
//! - Handshakes pass the [`OriginGuard`](super::origin::OriginGuard) first (403 otherwise). A peer
//!   admitted only because the check was off is closed (1008) when the check is turned back on.
//! - Binary frames are refused (close 1003): the old unauthenticated "f32 PCM into Whisper" path is
//!   gone.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use futures::StreamExt;
use serde::Deserialize;
use tauri::async_runtime::RwLock;
use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;
use warp::{
    filters::BoxedFilter,
    http::StatusCode,
    reply::Response,
    ws::{Message, WebSocket, Ws},
    Filter, Reply,
};

use super::origin::{self, Admission, ConnInfo, OriginGuard};

/// Largest accepted text message. Real events are a few KB (captions, text with an emote map).
pub const MAX_MESSAGE_BYTES: usize = 256 * 1024;

/// WebSocket close code for a binary frame (RFC 6455 1003, "unsupported data").
pub const CLOSE_UNSUPPORTED_DATA: u16 = 1003;

/// WebSocket close code when the Origin check is turned back on for a peer it refuses (RFC 6455
/// 1008, "policy violation").
pub const CLOSE_POLICY_VIOLATION: u16 = 1008;

/// Close reason sent with [`CLOSE_POLICY_VIOLATION`].
pub const REVOKED_REASON: &str = "the Origin check was turned on";

/// Longest accepted `?id=` (clients use random ids of ~20–60 characters).
pub const MAX_PEER_ID_LEN: usize = 256;

#[derive(Deserialize)]
pub struct PeerQueryData {
    id: String,
}

type PeerTx = mpsc::UnboundedSender<Result<Message, warp::Error>>;

/// A connected peer. `conn` tells a reconnect with the same id apart from the old socket.
struct PeerEntry {
    conn: u64,
    tx: PeerTx,
}

type Peers = Arc<RwLock<HashMap<String, PeerEntry>>>;

static NEXT_CONNECTION: AtomicU64 = AtomicU64::new(1);

pub(super) fn text_reply(status: StatusCode, body: &'static str) -> Response {
    warp::reply::with_status(body, status).into_response()
}

/// The `/pubsub` route. `input` carries host broadcasts, `output` peer messages to the host.
pub fn path(mut input: mpsc::Receiver<String>, output: mpsc::Sender<String>, guard: Arc<OriginGuard>) -> BoxedFilter<(Response,)> {
    let peers = Peers::default();

    let input_peers = peers.clone();
    tauri::async_runtime::spawn(async move {
        // Ends when the sender (plugin state) is dropped, instead of spinning on a closed channel.
        while let Some(text) = input.recv().await {
            let message = Message::text(text);
            for peer in input_peers.read().await.values() {
                let _ = peer.tx.send(Ok(message.clone()));
            }
        }
    });

    let peers = warp::any().map(move || peers.clone());
    let output = warp::any().map(move || output.clone());
    let guard = warp::any().map(move || guard.clone());
    warp::path("pubsub")
        .and(warp::ws())
        .and(origin::conn_info())
        .and(guard)
        .and(warp::query::<PeerQueryData>())
        .and(peers)
        .and(output)
        .map(
            |ws: Ws, info: ConnInfo, guard: Arc<OriginGuard>, query: PeerQueryData, peers: Peers, output: mpsc::Sender<String>| {
                let Ok(admission) = guard.admit(&info, "pubsub") else {
                    return text_reply(StatusCode::FORBIDDEN, "Forbidden: this origin may not connect to Sigil");
                };
                if query.id.is_empty() || query.id.len() > MAX_PEER_ID_LEN {
                    return text_reply(StatusCode::BAD_REQUEST, "Bad Request: invalid id");
                }
                ws.max_message_size(MAX_MESSAGE_BYTES)
                    .max_frame_size(MAX_MESSAGE_BYTES)
                    .on_upgrade(move |socket| peer_handler(socket, peers, output, query, admission, guard))
                    .into_response()
            },
        )
        .boxed()
}

async fn peer_handler(
    ws: WebSocket,
    peers: Peers,
    output: mpsc::Sender<String>,
    query: PeerQueryData,
    mut admission: Admission,
    guard: Arc<OriginGuard>,
) {
    let id = query.id;
    tracing::debug!(
        target: "sigil::web",
        "[pubsub] peer connected: {id:?} ({:?}{})",
        admission.allowed,
        if admission.is_revocable() { ", closed if the Origin check is turned on" } else { "" }
    );
    let (peer_tx, mut peer_rx) = ws.split();

    let (tx, rx) = mpsc::unbounded_channel();
    tauri::async_runtime::spawn(UnboundedReceiverStream::new(rx).forward(peer_tx));

    // A reconnect with the same id replaces the old entry; `conn` keeps the old socket's cleanup
    // from removing the new one.
    let conn = NEXT_CONNECTION.fetch_add(1, Ordering::Relaxed);
    peers
        .write()
        .await
        .insert(id.clone(), PeerEntry { conn, tx: tx.clone() });

    loop {
        let result = tokio::select! {
            next = peer_rx.next() => next,
            rejection = admission.revoked() => {
                guard.report_revoked(rejection);
                let _ = tx.send(Ok(Message::close_with(CLOSE_POLICY_VIOLATION, REVOKED_REASON)));
                break;
            }
        };
        let Some(Ok(msg)) = result else {
            break;
        };
        if msg.is_close() {
            break;
        }
        if msg.is_ping() || msg.is_pong() {
            continue;
        }
        if msg.is_binary() {
            tracing::warn!(target: "sigil::web", "[pubsub] peer {id:?} sent a binary frame; closing (binary frames are not accepted)");
            let _ = tx.send(Ok(Message::close_with(CLOSE_UNSUPPORTED_DATA, "binary frames are not accepted")));
            break;
        }
        let Ok(text) = msg.to_str() else {
            break;
        };
        if output.send(text.to_string()).await.is_err() {
            break;
        }
    }

    let mut peers = peers.write().await;
    if peers.get(&id).is_some_and(|entry| entry.conn == conn) {
        peers.remove(&id);
    }
    tracing::debug!(target: "sigil::web", "[pubsub] peer disconnected: {id:?}");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::web::origin::{LocalMachine, OriginRules};
    use std::net::{IpAddr, SocketAddr};
    use std::time::Duration;

    struct NoInterfaces;
    impl LocalMachine for NoInterfaces {
        fn is_own_ip(&self, _: IpAddr) -> bool {
            false
        }
        fn is_own_hostname(&self, _: &str) -> bool {
            false
        }
    }

    fn guard_with(port: u16, check_enabled: bool) -> Arc<OriginGuard> {
        let rules = OriginRules { port, dev_port: None };
        Arc::new(OriginGuard::new(rules, check_enabled, None, Box::new(NoInterfaces)))
    }

    struct Harness {
        route: BoxedFilter<(Response,)>,
        to_peers: mpsc::Sender<String>,
        from_peers: mpsc::Receiver<String>,
        guard: Arc<OriginGuard>,
    }

    fn harness_with(port: u16, check_enabled: bool) -> Harness {
        let (to_peers, input) = mpsc::channel(4);
        let (output, from_peers) = mpsc::channel(4);
        let guard = guard_with(port, check_enabled);
        Harness {
            route: path(input, output, guard.clone()),
            to_peers,
            from_peers,
            guard,
        }
    }

    fn harness(port: u16) -> Harness {
        harness_with(port, true)
    }

    fn upgrade_request(path: &str) -> warp::test::RequestBuilder {
        warp::test::request()
            .path(path)
            .header("connection", "upgrade")
            .header("upgrade", "websocket")
            .header("sec-websocket-version", "13")
            .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
    }

    #[tokio::test]
    async fn handshake_is_gated_by_origin() {
        let h = harness(3030);
        let lan_peer = SocketAddr::from(([192, 168, 1, 77], 50000));
        let loopback = SocketAddr::from(([127, 0, 0, 1], 50000));
        let cases: [(Option<&str>, SocketAddr, StatusCode); 8] = [
            (Some("https://tauri.localhost"), lan_peer, StatusCode::SWITCHING_PROTOCOLS),
            (Some("http://localhost:3030"), loopback, StatusCode::SWITCHING_PROTOCOLS),
            (Some("http://127.0.0.1:3030"), lan_peer, StatusCode::SWITCHING_PROTOCOLS),
            (None, loopback, StatusCode::SWITCHING_PROTOCOLS),
            (Some("https://evil.example"), loopback, StatusCode::FORBIDDEN),
            (Some("http://localhost:1420"), loopback, StatusCode::FORBIDDEN),
            (Some("null"), loopback, StatusCode::FORBIDDEN),
            (None, lan_peer, StatusCode::FORBIDDEN),
        ];
        for (origin, remote, expected) in cases {
            let mut req = upgrade_request("/pubsub?id=test").remote_addr(remote);
            if let Some(origin) = origin {
                req = req.header("origin", origin);
            }
            let status = req.reply(&h.route).await.status();
            assert_eq!(status, expected, "origin {origin:?} from {remote}");
        }
    }

    #[tokio::test]
    async fn rejects_bad_ids() {
        let h = harness(3030);
        let long = format!("/pubsub?id={}", "a".repeat(MAX_PEER_ID_LEN + 1));
        for path in ["/pubsub?id=", long.as_str()] {
            let status = upgrade_request(path)
                .remote_addr(SocketAddr::from(([127, 0, 0, 1], 1)))
                .header("origin", "http://localhost:3030")
                .reply(&h.route)
                .await
                .status();
            assert_eq!(status, StatusCode::BAD_REQUEST, "{path}");
        }
    }

    #[tokio::test]
    async fn forwards_text_and_refuses_binary() {
        let mut h = harness(3030);
        let mut client = warp::test::ws()
            .path("/pubsub?id=page")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .expect("allowed handshake");

        client
            .send_text(r#"{"topic":"text.textfield","data":{"type":0,"value":"hi"}}"#)
            .await;
        let got = tokio::time::timeout(Duration::from_secs(5), h.from_peers.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got, r#"{"topic":"text.textfield","data":{"type":0,"value":"hi"}}"#);

        // Host broadcasts reach the peer.
        h.to_peers
            .send(r#"{"topic":"obs.caption"}"#.to_string())
            .await
            .unwrap();
        let msg = tokio::time::timeout(Duration::from_secs(5), client.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(msg.to_str().unwrap(), r#"{"topic":"obs.caption"}"#);

        // Binary PCM (the removed Whisper injection path) closes the socket with 1003.
        client.send(Message::binary(vec![0u8; 16])).await;
        let closed = tokio::time::timeout(Duration::from_secs(5), client.recv())
            .await
            .unwrap();
        assert!(closed.is_err(), "expected the server to close, got {closed:?}");
        // Nothing from the binary frame reached the host.
        assert!(h.from_peers.try_recv().is_err());
    }

    #[tokio::test]
    async fn re_enabling_the_check_closes_foreign_peers_only() {
        let mut h = harness_with(3030, false);
        // The kill switch is off: a foreign page gets in.
        let mut foreign = warp::test::ws()
            .path("/pubsub?id=evil")
            .header("origin", "https://evil.example")
            .handshake(h.route.clone())
            .await
            .expect("admitted while the check is off");
        let mut page = warp::test::ws()
            .path("/pubsub?id=page")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .unwrap();
        foreign
            .send_text(r#"{"topic":"text.textfield","data":{"type":0,"value":"x"}}"#)
            .await;
        tokio::time::timeout(Duration::from_secs(5), h.from_peers.recv())
            .await
            .unwrap()
            .unwrap();

        assert!(h.guard.set_setting(true));
        let closed = tokio::time::timeout(Duration::from_secs(5), foreign.recv())
            .await
            .expect("the foreign peer is closed");
        // Either a close frame or an already-closed socket.
        if let Ok(msg) = closed {
            assert!(msg.is_close(), "expected a close frame, got {msg:?}");
        }
        // Nothing more reaches the host from it, while the Sigil page keeps receiving.
        foreign
            .send_text(r#"{"topic":"text.textfield","data":{"type":0,"value":"late"}}"#)
            .await;
        h.to_peers.send("after".to_string()).await.unwrap();
        let msg = tokio::time::timeout(Duration::from_secs(5), page.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(msg.to_str().unwrap(), "after");
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert!(h.from_peers.try_recv().is_err(), "a revoked peer must not reach the host");
        // New foreign handshakes are refused again.
        let again = warp::test::ws()
            .path("/pubsub?id=evil2")
            .header("origin", "https://evil.example")
            .handshake(h.route.clone())
            .await;
        assert!(again.is_err());
    }

    #[tokio::test]
    async fn foreign_handshake_fails_end_to_end() {
        let h = harness(3030);
        let result = warp::test::ws()
            .path("/pubsub?id=evil")
            .header("origin", "https://evil.example")
            .handshake(h.route.clone())
            .await;
        assert!(result.is_err(), "foreign origin must not upgrade");
    }

    #[tokio::test]
    async fn peers_are_not_relayed_to_each_other() {
        let mut h = harness(3030);
        let mut a = warp::test::ws()
            .path("/pubsub?id=a")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .unwrap();
        let mut b = warp::test::ws()
            .path("/pubsub?id=b")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .unwrap();
        a.send_text(r#"{"topic":"stt.external.command","data":{"cmd":"start"}}"#)
            .await;
        // The host sees it (and decides); peer b does not get it straight from a.
        let got = tokio::time::timeout(Duration::from_secs(5), h.from_peers.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(got.contains("stt.external.command"));
        h.to_peers.send("from-host".to_string()).await.unwrap();
        let first_for_b = tokio::time::timeout(Duration::from_secs(5), b.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first_for_b.to_str().unwrap(), "from-host");
    }

    #[tokio::test]
    async fn reconnect_with_same_id_keeps_the_new_socket() {
        let h = harness(3030);
        let old = warp::test::ws()
            .path("/pubsub?id=same")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .unwrap();
        let mut new = warp::test::ws()
            .path("/pubsub?id=same")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .unwrap();
        drop(old);
        // Give the old handler time to notice the close and run its cleanup.
        tokio::time::sleep(Duration::from_millis(200)).await;
        h.to_peers.send("still-here".to_string()).await.unwrap();
        let msg = tokio::time::timeout(Duration::from_secs(5), new.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(msg.to_str().unwrap(), "still-here");
    }
}
