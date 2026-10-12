//! `/pubsub?id=<peer>[&token=<LAN token>]` WebSocket: JSON text events between the host webview and
//! external pages (mic.html, obs-captions.html, inputfield.html), Link apps and local tools.
//!
//! - Host → peers: `plugin:web|pubsub_broadcast` strings are sent to every connected peer.
//! - Peer → host: text frames are forwarded to the host webview as the `pubsub` event, where
//!   `consumePubSubMessage` validates the topic against its allow-list and re-broadcasts what it
//!   accepts. Peers are not relayed to each other directly, so nothing reaches another peer without
//!   passing the host's validation.
//! - Handshakes pass the [`AccessGuard`] first: a refused Origin gets `403`; a client on another
//!   machine without the right LAN token (or while the server is set to this PC only) is upgraded
//!   and closed at once with 4401 / 4403 ([`refuse_after_upgrade`]), so the page can say why. An
//!   admitted peer is closed when a later change refuses it (Origin check back on: 1008; new
//!   token, legacy off, this PC only: 4401 / 4403).
//! - Binary frames are refused (close 1003): the old unauthenticated "f32 PCM into Whisper" path is
//!   gone.
//! - An id belongs to one socket: a second connection with the same id takes it over and the older
//!   socket is closed with 4409 ([`CLOSE_REPLACED`]). It used to stay open but unlisted, so it
//!   looked connected and never received another broadcast. Ids are per socket: a client that
//!   reconnects on close must send a fresh random id each time (Sigil's own pages do), and 4409
//!   means "pick another id". Two clients that share a fixed id and both reconnect on close would
//!   evict each other in a loop, so an id that was just taken over (within [`TAKEOVER_COOLDOWN`])
//!   is not taken over again: the newcomer is closed with 4409 and the current holder stays.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::{SinkExt, StreamExt};
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

use super::auth::{Access, AccessGuard, AccessRejection, LanRejected, Refusal};
use super::origin::{self, ConnInfo};

/// Largest accepted text message. Real events are a few KB (captions, text with an emote map).
pub const MAX_MESSAGE_BYTES: usize = 256 * 1024;

/// WebSocket close code for a binary frame (RFC 6455 1003, "unsupported data").
pub const CLOSE_UNSUPPORTED_DATA: u16 = 1003;

/// WebSocket close code when the Origin check is turned back on for a peer it refuses (RFC 6455
/// 1008, "policy violation").
pub const CLOSE_POLICY_VIOLATION: u16 = 1008;

/// Close reason sent with [`CLOSE_POLICY_VIOLATION`].
pub const REVOKED_REASON: &str = "the Origin check was turned on";

/// WebSocket close code (4xxx: application use) for a socket whose id another connection took
/// over. A page that reconnects on close then comes back with a fresh id.
pub const CLOSE_REPLACED: u16 = 4409;

/// Close reason sent with [`CLOSE_REPLACED`].
pub const REPLACED_REASON: &str = "another connection took this id";

/// Close reason sent with [`CLOSE_REPLACED`] to a newcomer that is refused because its id was
/// taken over a moment ago (see [`TAKEOVER_COOLDOWN`]).
pub const ID_IN_USE_REASON: &str = "this id is in use by another connection; use a unique id per socket";

/// An id that one connection just took over from another is not taken over again this soon: clients
/// that fight over an id and reconnect on every close would otherwise swap it back and forth as
/// fast as they reconnect, and both would lose events. A legitimate reconnect (a page that
/// reloaded, a half-open socket) is a single takeover and is never affected.
#[cfg(not(test))]
const TAKEOVER_COOLDOWN: Duration = Duration::from_secs(10);
#[cfg(test)]
const TAKEOVER_COOLDOWN: Duration = Duration::from_millis(1000);

/// Longest accepted `?id=` (clients use random ids of ~20–60 characters).
pub const MAX_PEER_ID_LEN: usize = 256;

/// How long a refused client gets to answer our close frame before the socket is dropped (waiting
/// lets the browser read the close code instead of a connection reset).
const REFUSAL_LINGER: Duration = Duration::from_secs(2);

/// Body of the `403` for a refused Origin.
pub(super) const FORBIDDEN_ORIGIN: &str = "Forbidden: this origin may not connect to Sigil";

#[derive(Deserialize)]
pub struct PeerQueryData {
    id: String,
    /// LAN token (stage 2); only needed from other machines.
    #[serde(default)]
    token: Option<String>,
}

type PeerTx = mpsc::UnboundedSender<Result<Message, warp::Error>>;

/// A connected peer. `conn` tells a reconnect with the same id apart from the old socket.
struct PeerEntry {
    conn: u64,
    tx: PeerTx,
    /// When this connection took the id over from another one (not set for a first connection).
    took_over_at: Option<Instant>,
}

type Peers = Arc<RwLock<HashMap<String, PeerEntry>>>;

static NEXT_CONNECTION: AtomicU64 = AtomicU64::new(1);

pub(super) fn text_reply(status: StatusCode, body: &'static str) -> Response {
    warp::reply::with_status(body, status).into_response()
}

/// Close frame for a connection that was admitted and is refused now ([`Access::revoked`]).
pub(super) fn revocation_frame(rejection: &AccessRejection) -> Message {
    match rejection.reason {
        Refusal::Origin(_) => Message::close_with(CLOSE_POLICY_VIOLATION, REVOKED_REASON),
        Refusal::Lan(lan) => {
            let (code, reason) = lan.close_frame();
            Message::close_with(code, reason)
        }
    }
}

/// Upgrade, optionally send `first` (PeerJS's `INVALID-KEY`), then close with the LAN refusal's
/// code and wait briefly for the client's close reply. Nothing the client sends is used.
pub(super) fn refuse_after_upgrade(ws: Ws, refusal: LanRejected, first: Option<Message>) -> Response {
    let (code, reason) = refusal.close_frame();
    ws.max_message_size(4096)
        .max_frame_size(4096)
        .on_upgrade(move |mut socket: WebSocket| async move {
            if let Some(message) = first {
                let _ = socket.send(message).await;
            }
            if socket.send(Message::close_with(code, reason)).await.is_ok() {
                let _ = tokio::time::timeout(REFUSAL_LINGER, async {
                    while let Some(Ok(message)) = socket.next().await {
                        if message.is_close() {
                            break;
                        }
                    }
                })
                .await;
            }
        })
        .into_response()
}

/// The `/pubsub` route. `input` carries host broadcasts, `output` peer messages to the host.
pub fn path(mut input: mpsc::Receiver<String>, output: mpsc::Sender<String>, guard: Arc<AccessGuard>) -> BoxedFilter<(Response,)> {
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
            |ws: Ws, info: ConnInfo, guard: Arc<AccessGuard>, query: PeerQueryData, peers: Peers, output: mpsc::Sender<String>| {
                let access = match guard.admit(info, query.token, "pubsub") {
                    Ok(access) => access,
                    Err(Refusal::Origin(_)) => return text_reply(StatusCode::FORBIDDEN, FORBIDDEN_ORIGIN),
                    Err(Refusal::Lan(refusal)) => return refuse_after_upgrade(ws, refusal, None),
                };
                if query.id.is_empty() || query.id.len() > MAX_PEER_ID_LEN {
                    return text_reply(StatusCode::BAD_REQUEST, "Bad Request: invalid id");
                }
                let id = query.id;
                ws.max_message_size(MAX_MESSAGE_BYTES)
                    .max_frame_size(MAX_MESSAGE_BYTES)
                    .on_upgrade(move |socket| peer_handler(socket, peers, output, id, access, guard))
                    .into_response()
            },
        )
        .boxed()
}

async fn peer_handler(ws: WebSocket, peers: Peers, output: mpsc::Sender<String>, id: String, mut access: Access, guard: Arc<AccessGuard>) {
    tracing::debug!(target: "sigil::web", "[pubsub] peer connected: {id:?} ({:?})", access.grant);
    let (peer_tx, mut peer_rx) = ws.split();

    let (tx, rx) = mpsc::unbounded_channel();
    tauri::async_runtime::spawn(UnboundedReceiverStream::new(rx).forward(peer_tx));

    // A connection with the same id replaces the old entry (`conn` keeps the old socket's cleanup
    // from removing the new one). Host broadcasts go to the entries only, so the old socket would
    // stay open and never hear anything again: close it, and its page reconnects.
    let conn = NEXT_CONNECTION.fetch_add(1, Ordering::Relaxed);
    {
        let mut map = peers.write().await;
        let current = map.get(&id);
        if current.is_some_and(|holder| holder.took_over_at.is_some_and(|at| at.elapsed() < TAKEOVER_COOLDOWN)) {
            // Two clients are fighting over this id: the holder keeps it, the newcomer is told why.
            tracing::debug!(target: "sigil::web", "[pubsub] id {id:?} was just taken over; refusing another connection with it");
            let _ = tx.send(Ok(Message::close_with(CLOSE_REPLACED, ID_IN_USE_REASON)));
            return;
        }
        let took_over_at = current.is_some().then(Instant::now);
        let replaced = map.insert(id.clone(), PeerEntry { conn, tx: tx.clone(), took_over_at });
        if let Some(old) = replaced {
            tracing::debug!(target: "sigil::web", "[pubsub] id {id:?} taken over by a new connection; closing the old one");
            let _ = old.tx.send(Ok(Message::close_with(CLOSE_REPLACED, REPLACED_REASON)));
        }
    }

    loop {
        let result = tokio::select! {
            next = peer_rx.next() => next,
            rejection = access.revoked(&guard) => {
                let frame = revocation_frame(&rejection);
                guard.report_revoked(rejection);
                let _ = tx.send(Ok(frame));
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
    use crate::services::web::auth::BindMode;
    use crate::services::web::origin::LocalMachine;
    use crate::services::web::test_support::{self, lan_policy, raw_connect, LanOnly, NoInterfaces, RawOutcome, OTHER_TOKEN, TOKEN};
    use std::net::SocketAddr;

    struct Harness {
        route: BoxedFilter<(Response,)>,
        to_peers: mpsc::Sender<String>,
        from_peers: mpsc::Receiver<String>,
        guard: Arc<AccessGuard>,
    }

    fn harness_for(check_enabled: bool, machine: Box<dyn LocalMachine>, policy: crate::services::web::auth::LanPolicy) -> Harness {
        let (to_peers, input) = mpsc::channel(4);
        let (output, from_peers) = mpsc::channel(4);
        let guard = test_support::guard(3030, check_enabled, machine, policy);
        Harness {
            route: path(input, output, guard.clone()),
            to_peers,
            from_peers,
            guard,
        }
    }

    fn harness_with(check_enabled: bool) -> Harness {
        harness_for(check_enabled, Box::new(NoInterfaces), lan_policy())
    }

    fn harness() -> Harness {
        harness_with(true)
    }

    /// Every test socket counts as a client on another machine.
    fn lan_harness() -> Harness {
        harness_for(true, Box::new(LanOnly), lan_policy())
    }

    fn upgrade_request(path: &str) -> warp::test::RequestBuilder {
        warp::test::request()
            .path(path)
            .header("connection", "upgrade")
            .header("upgrade", "websocket")
            .header("sec-websocket-version", "13")
            .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
    }

    const WAIT: Duration = Duration::from_secs(5);
    const QUIET: Duration = Duration::from_millis(150);

    #[tokio::test]
    async fn handshake_is_gated_by_origin() {
        let h = harness();
        let lan_peer = SocketAddr::from(([192, 168, 1, 77], 50000));
        let loopback = SocketAddr::from(([127, 0, 0, 1], 50000));
        let with_token = format!("/pubsub?id=test&token={TOKEN}");
        let cases: [(Option<&str>, SocketAddr, &str, StatusCode); 10] = [
            (Some("https://tauri.localhost"), lan_peer, &with_token, StatusCode::SWITCHING_PROTOCOLS),
            (Some("http://localhost:3030"), loopback, "/pubsub?id=test", StatusCode::SWITCHING_PROTOCOLS),
            (Some("http://127.0.0.1:3030"), lan_peer, &with_token, StatusCode::SWITCHING_PROTOCOLS),
            (None, loopback, "/pubsub?id=test", StatusCode::SWITCHING_PROTOCOLS),
            // An origin-less tool on another machine: only with the token.
            (None, lan_peer, &with_token, StatusCode::SWITCHING_PROTOCOLS),
            (None, lan_peer, "/pubsub?id=test", StatusCode::FORBIDDEN),
            // Foreign pages are refused before the token is even looked at.
            (Some("https://evil.example"), loopback, &with_token, StatusCode::FORBIDDEN),
            (Some("https://evil.example"), lan_peer, &with_token, StatusCode::FORBIDDEN),
            (Some("http://localhost:1420"), loopback, "/pubsub?id=test", StatusCode::FORBIDDEN),
            (Some("null"), loopback, "/pubsub?id=test", StatusCode::FORBIDDEN),
        ];
        for (origin, remote, path, expected) in cases {
            let mut req = upgrade_request(path).remote_addr(remote);
            if let Some(origin) = origin {
                req = req.header("origin", origin);
            }
            let status = req.reply(&h.route).await.status();
            assert_eq!(status, expected, "origin {origin:?} from {remote} on {path}");
        }
    }

    #[tokio::test]
    async fn rejects_bad_ids() {
        let h = harness();
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
        let mut h = harness();
        let mut client = warp::test::ws()
            .path("/pubsub?id=page")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .expect("allowed handshake");

        client
            .send_text(r#"{"topic":"text.textfield","data":{"type":0,"value":"hi"}}"#)
            .await;
        let got = tokio::time::timeout(WAIT, h.from_peers.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(got, r#"{"topic":"text.textfield","data":{"type":0,"value":"hi"}}"#);

        // Host broadcasts reach the peer.
        h.to_peers
            .send(r#"{"topic":"obs.caption"}"#.to_string())
            .await
            .unwrap();
        let msg = tokio::time::timeout(WAIT, client.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(msg.to_str().unwrap(), r#"{"topic":"obs.caption"}"#);

        // Binary PCM (the removed Whisper injection path) closes the socket with 1003.
        client.send(Message::binary(vec![0u8; 16])).await;
        let closed = tokio::time::timeout(WAIT, client.recv()).await.unwrap();
        assert!(closed.is_err(), "expected the server to close, got {closed:?}");
        // Nothing from the binary frame reached the host.
        assert!(h.from_peers.try_recv().is_err());
    }

    #[tokio::test]
    async fn re_enabling_the_check_closes_foreign_peers_only() {
        let mut h = harness_with(false);
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
        tokio::time::timeout(WAIT, h.from_peers.recv())
            .await
            .unwrap()
            .unwrap();

        assert!(h.guard.origin().set_setting(true));
        let closed = tokio::time::timeout(WAIT, foreign.recv())
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
        let msg = tokio::time::timeout(WAIT, page.recv())
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
        let h = harness();
        let result = warp::test::ws()
            .path("/pubsub?id=evil")
            .header("origin", "https://evil.example")
            .handshake(h.route.clone())
            .await;
        assert!(result.is_err(), "foreign origin must not upgrade");
    }

    #[tokio::test]
    async fn peers_are_not_relayed_to_each_other() {
        let mut h = harness();
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
        let got = tokio::time::timeout(WAIT, h.from_peers.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(got.contains("stt.external.command"));
        h.to_peers.send("from-host".to_string()).await.unwrap();
        let first_for_b = tokio::time::timeout(WAIT, b.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(first_for_b.to_str().unwrap(), "from-host");
    }

    #[tokio::test]
    async fn reconnect_with_same_id_keeps_the_new_socket() {
        let h = harness();
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
        let msg = tokio::time::timeout(WAIT, new.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(msg.to_str().unwrap(), "still-here");
    }

    #[tokio::test]
    async fn a_second_connection_with_the_same_id_closes_the_first() {
        let h = harness();
        // The first page stays connected (warp's test client cannot read close frames: a raw one can).
        let (first, mut second) = tokio::join!(
            raw_connect(h.route.clone(), "/pubsub?id=same", Some("http://localhost:3030"), WAIT),
            async {
                tokio::time::sleep(Duration::from_millis(300)).await;
                warp::test::ws()
                    .path("/pubsub?id=same")
                    .header("origin", "http://localhost:3030")
                    .handshake(h.route.clone())
                    .await
                    .unwrap()
            }
        );
        // It does not stay open and deaf: it is told, with a code a page can reconnect on.
        assert_eq!(
            first,
            RawOutcome::Closed {
                texts: vec![],
                code: CLOSE_REPLACED,
                reason: REPLACED_REASON.into()
            }
        );
        // The old socket's cleanup leaves the new entry alone.
        tokio::time::sleep(Duration::from_millis(200)).await;
        h.to_peers.send("for-the-new-one".to_string()).await.unwrap();
        let msg = tokio::time::timeout(WAIT, second.recv()).await.unwrap().unwrap();
        assert_eq!(msg.to_str().unwrap(), "for-the-new-one");
    }

    #[tokio::test]
    async fn clients_that_fight_over_an_id_settle_instead_of_evicting_each_other() {
        let h = harness();
        let origin = Some("http://localhost:3030");
        // The first client holds the id until the second takes it over.
        let first = tokio::spawn(raw_connect(h.route.clone(), "/pubsub?id=shared", origin, WAIT));
        tokio::time::sleep(Duration::from_millis(300)).await;
        let mut second = warp::test::ws()
            .path("/pubsub?id=shared")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .unwrap();
        assert_eq!(
            first.await.unwrap(),
            RawOutcome::Closed {
                texts: vec![],
                code: CLOSE_REPLACED,
                reason: REPLACED_REASON.into()
            }
        );

        // The evicted one reconnects on close (as a client that shares a fixed id would): this time
        // it is the newcomer that is told no, and the holder keeps hearing the host.
        let retry = raw_connect(h.route.clone(), "/pubsub?id=shared", origin, WAIT).await;
        assert_eq!(
            retry,
            RawOutcome::Closed {
                texts: vec![],
                code: CLOSE_REPLACED,
                reason: ID_IN_USE_REASON.into()
            }
        );
        h.to_peers.send("for-the-holder".to_string()).await.unwrap();
        let msg = tokio::time::timeout(WAIT, second.recv()).await.unwrap().unwrap();
        assert_eq!(msg.to_str().unwrap(), "for-the-holder");

        // Once the cooldown is over a takeover works again (a page that reloaded with its id).
        tokio::time::sleep(TAKEOVER_COOLDOWN + Duration::from_millis(100)).await;
        let mut third = warp::test::ws()
            .path("/pubsub?id=shared")
            .header("origin", "http://localhost:3030")
            .handshake(h.route.clone())
            .await
            .unwrap();
        let closed = tokio::time::timeout(WAIT, second.recv()).await.unwrap();
        assert!(closed.is_err() || closed.unwrap().is_close(), "the old holder is closed");
        h.to_peers.send("for-the-third".to_string()).await.unwrap();
        let msg = tokio::time::timeout(WAIT, third.recv()).await.unwrap().unwrap();
        assert_eq!(msg.to_str().unwrap(), "for-the-third");
    }

    #[tokio::test]
    async fn a_first_connection_is_never_refused_for_its_id() {
        let h = harness();
        // Different ids never interact, and a lone reconnect after the old socket went away is a
        // plain registration.
        for id in ["a", "b", "a"] {
            let mut socket = warp::test::ws()
                .path(&format!("/pubsub?id={id}"))
                .header("origin", "http://localhost:3030")
                .handshake(h.route.clone())
                .await
                .unwrap();
            h.to_peers.send(format!("for-{id}")).await.unwrap();
            let msg = tokio::time::timeout(WAIT, socket.recv()).await.unwrap().unwrap();
            assert_eq!(msg.to_str().unwrap(), format!("for-{id}"));
            drop(socket);
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }

    #[tokio::test]
    async fn lan_clients_need_the_token() {
        let mut h = lan_harness();
        // With the token: text flows both ways (Link apps, captions page on another PC).
        let mut linked = warp::test::ws()
            .path(&format!("/pubsub?id=link&token={}", TOKEN.to_uppercase()))
            .header("origin", "https://tauri.localhost")
            .handshake(h.route.clone())
            .await
            .unwrap();
        linked
            .send_text(r#"{"topic":"text.textfield","data":{"type":0,"value":"from-lan"}}"#)
            .await;
        let got = tokio::time::timeout(WAIT, h.from_peers.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(got.contains("from-lan"));

        // Without it, or with a wrong one: closed at once, nothing reaches the host or the client.
        for path in ["/pubsub?id=nope".to_string(), format!("/pubsub?id=nope&token={OTHER_TOKEN}"), "/pubsub?id=nope&token=".to_string()] {
            let mut refused = warp::test::ws()
                .path(&path)
                .header("origin", "http://192.168.1.50:3030")
                .handshake(h.route.clone())
                .await
                .expect("upgraded so the close code reaches the page");
            refused
                .send_text(r#"{"topic":"text.textfield","data":{"type":0,"value":"sneaky"}}"#)
                .await;
            h.to_peers.send("secret-caption".to_string()).await.unwrap();
            let closed = tokio::time::timeout(WAIT, refused.recv()).await.unwrap();
            assert!(closed.is_err(), "{path}: expected close, got {closed:?}");
            // The linked peer got the broadcast; the refused one did not.
            let msg = tokio::time::timeout(WAIT, linked.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(msg.to_str().unwrap(), "secret-caption");
        }
        tokio::time::sleep(QUIET).await;
        assert!(h.from_peers.try_recv().is_err(), "a refused client must not reach the host");
    }

    #[tokio::test]
    async fn a_new_token_or_this_pc_only_disconnects_lan_clients() {
        let h = lan_harness();
        let mut old = warp::test::ws()
            .path(&format!("/pubsub?id=old&token={TOKEN}"))
            .header("origin", "https://tauri.localhost")
            .handshake(h.route.clone())
            .await
            .unwrap();
        h.guard.set_token(OTHER_TOKEN.to_string());
        let closed = tokio::time::timeout(WAIT, old.recv()).await.unwrap();
        assert!(closed.is_err() || closed.unwrap().is_close());

        let mut current = warp::test::ws()
            .path(&format!("/pubsub?id=new&token={OTHER_TOKEN}"))
            .header("origin", "https://tauri.localhost")
            .handshake(h.route.clone())
            .await
            .unwrap();
        h.to_peers.send("hello".to_string()).await.unwrap();
        assert_eq!(
            tokio::time::timeout(WAIT, current.recv())
                .await
                .unwrap()
                .unwrap()
                .to_str()
                .unwrap(),
            "hello"
        );
        h.guard
            .set_lan_policy(test_support::policy(BindMode::Localhost, OTHER_TOKEN, false));
        let closed = tokio::time::timeout(WAIT, current.recv()).await.unwrap();
        assert!(closed.is_err() || closed.unwrap().is_close());
    }

    #[tokio::test]
    async fn refusals_carry_close_codes() {
        let page = Some("http://192.168.1.50:3030");
        let missing = raw_connect(lan_harness().route, "/pubsub?id=a", page, WAIT).await;
        assert_eq!(
            missing,
            RawOutcome::Closed {
                texts: vec![],
                code: 4401,
                reason: "missing_token".into()
            }
        );
        let wrong = raw_connect(lan_harness().route, &format!("/pubsub?id=a&token={OTHER_TOKEN}"), page, WAIT).await;
        assert_eq!(
            wrong,
            RawOutcome::Closed {
                texts: vec![],
                code: 4401,
                reason: "invalid_token".into()
            }
        );
        let local_only = harness_for(true, Box::new(LanOnly), test_support::policy(BindMode::Localhost, TOKEN, false));
        let disabled = raw_connect(local_only.route, &format!("/pubsub?id=a&token={TOKEN}"), page, WAIT).await;
        assert_eq!(
            disabled,
            RawOutcome::Closed {
                texts: vec![],
                code: 4403,
                reason: "lan_disabled".into()
            }
        );
        let ok = raw_connect(lan_harness().route, &format!("/pubsub?id=a&token={TOKEN}"), page, QUIET).await;
        assert_eq!(ok, RawOutcome::Open { texts: vec![] });
        let foreign = raw_connect(lan_harness().route, &format!("/pubsub?id=a&token={TOKEN}"), Some("https://evil.example"), WAIT).await;
        assert_eq!(foreign, RawOutcome::Refused(403));
    }

    #[tokio::test]
    async fn revoked_lan_clients_get_the_close_code() {
        let h = lan_harness();
        let route = h.route.clone();
        let guard = h.guard.clone();
        let flip = tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(400)).await;
            guard.set_token(OTHER_TOKEN.to_string());
        });
        let outcome = raw_connect(route, &format!("/pubsub?id=a&token={TOKEN}"), Some("https://tauri.localhost"), WAIT).await;
        flip.await.unwrap();
        assert_eq!(
            outcome,
            RawOutcome::Closed {
                texts: vec![],
                code: 4401,
                reason: "invalid_token".into()
            }
        );
    }
}
