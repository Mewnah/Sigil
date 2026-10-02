//! Minimal PeerJS signaling server at `/peer?id=<peer>[&key=<LAN token>]` (the host registers
//! `server`; `/client` mirrors register random ids and exchange OFFER/ANSWER/CANDIDATE with it).
//!
//! Handshakes pass the [`AccessGuard`] first: a refused Origin gets `403`. A client on another
//! machine must carry the LAN token as the PeerJS `key` (PeerJS's own `token` query parameter is a
//! random per-socket value and is ignored); without it the socket is upgraded, sent PeerJS's
//! `INVALID-KEY` (the page shows `invalid-key`) and closed with 4401 / 4403. An admitted peer is
//! closed when a later change refuses it. Binary frames are refused (PeerJS signaling is JSON
//! text only).
//!
//! **Departures reach the data channel.** The canvas document and the pubsub broadcasts travel
//! over a WebRTC DataConnection between the host webview and the mirror, not over this socket, so
//! closing a revoked mirror's signaling socket alone would leave that channel up for a client that
//! ignores the loss of signaling. Whenever a registered socket goes away (revoked by a rules
//! change, closed by the client, or dropped), the server therefore sends PeerJS's `LEAVE` (src =
//! the departed id) to every peer it signaled. PeerJS closes all connections with a peer that
//! leaves, and the host's provider (src/shared/services/peer/provider.ts) drops them as well, so
//! the host stops streaming to it. A mirror can only keep a data channel while its signaling
//! socket stays registered, and a registered socket is closed as soon as the rules refuse it.
//! Honest `/client` pages tear down and reconnect on any signaling loss anyway.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::Value as SerdeValue;
use tauri::async_runtime::RwLock;
use warp::{
    filters::BoxedFilter,
    http::StatusCode,
    reply::Response,
    ws::{Message, WebSocket, Ws},
    Filter, Reply,
};

use tokio::sync::mpsc;
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::auth::{Access, AccessGuard, LanRejected, Refusal};
use super::origin::{self, ConnInfo};
use super::pubsub::{refuse_after_upgrade, revocation_frame, text_reply, CLOSE_UNSUPPORTED_DATA, FORBIDDEN_ORIGIN, MAX_PEER_ID_LEN};

/// Largest accepted signaling message (SDP offers are a few KB).
pub const MAX_MESSAGE_BYTES: usize = 128 * 1024;

#[derive(Deserialize)]
pub struct PeerQueryData {
    id: String,
    /// The PeerJS API key, which carries the LAN token (stage 2).
    #[serde(default)]
    key: Option<String>,
}

type PeerTx = mpsc::UnboundedSender<Result<Message, warp::Error>>;

/// Distinct peers remembered per peer as signaled; past this, its departure is announced to every
/// registered peer instead (an admitted client could address many ids).
const MAX_CONTACTED: usize = 64;

/// A registered signaling socket.
struct Registered {
    tx: PeerTx,
    /// Peers this one relayed signaling to (OFFER, ANSWER, CANDIDATE): any WebRTC connection it
    /// has is with one of them, so they are told when it leaves.
    contacted: HashSet<String>,
    /// It signaled more than [`MAX_CONTACTED`] peers: its departure goes to everyone.
    contacted_overflow: bool,
}

impl Registered {
    fn new(tx: PeerTx) -> Self {
        Self {
            tx,
            contacted: HashSet::new(),
            contacted_overflow: false,
        }
    }

    fn remember(&mut self, dst: &str) {
        if self.contacted_overflow || self.contacted.contains(dst) {
            return;
        }
        if self.contacted.len() >= MAX_CONTACTED {
            self.contacted_overflow = true;
            self.contacted.clear();
            return;
        }
        self.contacted.insert(dst.to_string());
    }
}

type Peers = Arc<RwLock<HashMap<String, Registered>>>;

pub fn path(guard: Arc<AccessGuard>) -> BoxedFilter<(Response,)> {
    let peers = Peers::default();
    let peers = warp::any().map(move || peers.clone());
    let guard = warp::any().map(move || guard.clone());
    warp::path("peer")
        .and(warp::ws())
        .and(origin::conn_info())
        .and(guard)
        .and(peers)
        .and(warp::query::<PeerQueryData>())
        .map(|ws: Ws, info: ConnInfo, guard: Arc<AccessGuard>, peers: Peers, query: PeerQueryData| {
            let access = match guard.admit(info, query.key, "peer") {
                Ok(access) => access,
                Err(Refusal::Origin(_)) => return text_reply(StatusCode::FORBIDDEN, FORBIDDEN_ORIGIN),
                Err(Refusal::Lan(refusal)) => return refuse_after_upgrade(ws, refusal, Some(invalid_key_message(refusal))),
            };
            if query.id.is_empty() || query.id.len() > MAX_PEER_ID_LEN {
                return text_reply(StatusCode::BAD_REQUEST, "Bad Request: invalid id");
            }
            let id = query.id;
            ws.max_message_size(MAX_MESSAGE_BYTES)
                .max_frame_size(MAX_MESSAGE_BYTES)
                .on_upgrade(move |socket| peer_handler(socket, peers, id, access, guard))
                .into_response()
        })
        .boxed()
}

/// PeerJS's `INVALID-KEY` server message: the client aborts with error type `invalid-key`.
fn invalid_key_message(refusal: LanRejected) -> Message {
    let msg = match refusal {
        LanRejected::LanDisabled => "This Sigil only accepts connections from its own PC.",
        LanRejected::MissingToken | LanRejected::InvalidToken => {
            "The LAN token is missing or out of date. Copy the LAN link again in Sigil (Settings, LAN access)."
        }
    };
    let message = PeerMessage {
        t: PeerMessageType::InvalidKey,
        src: String::new(),
        dst: String::new(),
        payload: serde_json::json!({ "msg": msg }),
    };
    Message::text(serde_json::to_string(&message).unwrap_or_default())
}

async fn peer_handler(ws: WebSocket, peers: Peers, id: String, mut access: Access, guard: Arc<AccessGuard>) {
    let (peer_tx, mut peer_rx) = ws.split();

    let (tx, rx) = mpsc::unbounded_channel();
    let rx = UnboundedReceiverStream::new(rx);
    tauri::async_runtime::spawn(rx.forward(peer_tx));

    // Check and register under one lock, so two sockets can never both hold an id (the loser's
    // departure would otherwise unregister the winner and announce it as gone).
    {
        let mut registered = peers.write().await;
        if registered.contains_key(&id) {
            tracing::debug!(target: "sigil::web", "[peer] already registered: {id:?}");
            let _ = tx.send(Ok(PeerMessageType::IdTaken.into()));
            return;
        }
        tx.send(Ok(PeerMessageType::Open.into())).ok();
        registered.insert(id.clone(), Registered::new(tx.clone()));
    }
    tracing::debug!(target: "sigil::web", "[peer] {id:?} registered ({:?})", access.grant);

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
        if msg.is_binary() {
            tracing::warn!(target: "sigil::web", "[peer] {id:?} sent a binary frame; closing");
            let _ = tx.send(Ok(Message::close_with(CLOSE_UNSUPPORTED_DATA, "binary frames are not accepted")));
            break;
        }
        handle_message(&id, msg, &peers).await;
    }
    depart(&peers, &id).await;
}

/// Unregister `id` and send PeerJS's `LEAVE` from it to the peers it signaled, which closes their
/// WebRTC connections with it (see the module docs). Done under the registry lock, so each `LEAVE`
/// arrives after everything `id` relayed to that peer and before anything from a later
/// registration of the same id.
async fn depart(peers: &Peers, id: &str) {
    let mut registered = peers.write().await;
    let Some(gone) = registered.remove(id) else {
        return;
    };
    let notify: Vec<(&String, &Registered)> = if gone.contacted_overflow {
        registered.iter().collect()
    } else {
        gone.contacted
            .iter()
            .filter_map(|dst| registered.get_key_value(dst.as_str()))
            .collect()
    };
    for (dst, peer) in notify {
        let _ = peer.tx.send(Ok(leave_message(id, dst)));
    }
}

/// PeerJS's `LEAVE` server message: the peer `src` is gone; the receiver closes its connections
/// with it.
fn leave_message(src: &str, dst: &str) -> Message {
    let message = PeerMessage {
        t: PeerMessageType::Leave,
        src: src.to_string(),
        dst: dst.to_string(),
        payload: SerdeValue::Null,
    };
    Message::text(serde_json::to_string(&message).unwrap_or_default())
}

/// PeerJS `ServerMessageType` values (wire names are upper case).
#[derive(Serialize, Deserialize, PartialEq, Default)]
#[serde(rename_all = "UPPERCASE")]
enum PeerMessageType {
    Open, // socket ready
    Leave,
    Candidate,
    Offer,
    Answer,
    Expire, // host not found
    Heartbeat,
    /// PeerJS expects `ID-TAKEN` (it ignored the old `ID_TAKEN` spelling).
    #[serde(rename = "ID-TAKEN", alias = "IDTAKEN", alias = "ID_TAKEN")]
    IdTaken,
    /// The API key (here: the LAN token) is wrong; PeerJS aborts with `invalid-key`.
    #[serde(rename = "INVALID-KEY")]
    InvalidKey,
    #[default]
    Error,
}

impl From<PeerMessageType> for Message {
    fn from(t: PeerMessageType) -> Message {
        Message::text(serde_json::to_string(&PeerMessageShort { t }).unwrap_or_default())
    }
}

#[derive(Serialize, Deserialize, PartialEq)]
struct PeerMessage {
    #[serde(rename(serialize = "type", deserialize = "type"))]
    #[serde(default)]
    pub t: PeerMessageType,
    #[serde(default)]
    pub src: String,
    #[serde(default)]
    pub dst: String,
    #[serde(default)]
    pub payload: SerdeValue,
}

#[derive(Serialize, Deserialize, PartialEq)]
struct PeerMessageShort {
    #[serde(rename(serialize = "type", deserialize = "type"))]
    #[serde(default)]
    pub t: PeerMessageType,
}

async fn handle_message(peer_id: &str, msg: Message, users: &Peers) {
    let Ok(msg_str) = msg.to_str() else { return };
    let Ok(mut msg) = serde_json::from_str::<PeerMessage>(msg_str) else {
        return;
    };

    // Write lock: a relay also records whom the sender signaled (for its `LEAVE`).
    let mut users = users.write().await;

    if msg.t == PeerMessageType::Offer && !users.contains_key(msg.dst.as_str()) {
        let Some(peer) = users.get(peer_id) else { return };
        let Ok(msg_str) = serde_json::to_string(&PeerMessage {
            t: PeerMessageType::Expire,
            src: msg.dst,
            dst: peer_id.to_string(),
            payload: msg.payload,
        }) else {
            return;
        };
        peer.tx.send(Ok(Message::text(msg_str))).ok();
    } else if !msg.dst.is_empty() && users.contains_key(&msg.dst) {
        // The server stamps the sender, so a peer can only ever speak (and `LEAVE`) as itself.
        msg.src = peer_id.to_string();
        let Ok(msg_str) = serde_json::to_string(&msg) else { return };
        if let Some(sender) = users.get_mut(peer_id) {
            sender.remember(&msg.dst);
        }
        let Some(peer) = users.get(&msg.dst) else { return };
        peer.tx.send(Ok(Message::text(msg_str))).ok();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::web::auth::BindMode;
    use crate::services::web::test_support::{self, lan_policy, raw_connect, LanOnly, NoInterfaces, RawOutcome, OTHER_TOKEN, TOKEN};
    use std::net::SocketAddr;
    use std::time::Duration;

    const WAIT: Duration = Duration::from_secs(5);

    fn route_for(check_enabled: bool, machine: Box<dyn crate::services::web::origin::LocalMachine>) -> (BoxedFilter<(Response,)>, Arc<AccessGuard>) {
        let guard = test_support::guard(3030, check_enabled, machine, lan_policy());
        (path(guard.clone()), guard)
    }

    fn route() -> BoxedFilter<(Response,)> {
        route_for(true, Box::new(NoInterfaces)).0
    }

    async fn expect_open(socket: &mut warp::test::WsClient) {
        let open = tokio::time::timeout(WAIT, socket.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(open.to_str().unwrap(), r#"{"type":"OPEN"}"#);
    }

    #[tokio::test]
    async fn re_enabling_the_check_closes_foreign_signaling_peers() {
        let (route, guard) = route_for(false, Box::new(NoInterfaces));
        let mut foreign = warp::test::ws()
            .path("/peer/peerjs?id=evil")
            .header("origin", "https://evil.example")
            .handshake(route.clone())
            .await
            .expect("admitted while the check is off");
        let mut host = warp::test::ws()
            .path("/peer/peerjs?id=server")
            .header("origin", "https://tauri.localhost")
            .handshake(route.clone())
            .await
            .unwrap();
        for socket in [&mut foreign, &mut host] {
            expect_open(socket).await;
        }

        guard.origin().set_setting(true);
        let closed = tokio::time::timeout(WAIT, foreign.recv())
            .await
            .expect("the foreign peer is closed");
        if let Ok(msg) = closed {
            assert!(msg.is_close(), "expected a close frame, got {msg:?}");
        }
        // The host's own registration is untouched: a Sigil page can still signal it.
        let mut client = warp::test::ws()
            .path("/peer/peerjs?id=client1")
            .header("origin", "http://localhost:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut client).await;
        client
            .send_text(r#"{"type":"OFFER","dst":"server","payload":{"sdp":"x"}}"#)
            .await;
        let offer = tokio::time::timeout(WAIT, host.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(offer.to_str().unwrap().contains("OFFER"));
    }

    #[tokio::test]
    async fn handshake_is_gated_by_origin() {
        let route = route();
        for (origin, expected) in [
            ("https://tauri.localhost", StatusCode::SWITCHING_PROTOCOLS),
            ("http://localhost:3030", StatusCode::SWITCHING_PROTOCOLS),
            ("https://evil.example", StatusCode::FORBIDDEN),
            ("http://evil.example:3030", StatusCode::FORBIDDEN),
        ] {
            let status = warp::test::request()
                .path(&format!("/peer/peerjs?key={TOKEN}&id=client1&token=abc"))
                .header("connection", "upgrade")
                .header("upgrade", "websocket")
                .header("sec-websocket-version", "13")
                .header("sec-websocket-key", "dGhlIHNhbXBsZSBub25jZQ==")
                .header("origin", origin)
                .remote_addr(SocketAddr::from(([192, 168, 1, 77], 1)))
                .reply(&route)
                .await
                .status();
            assert_eq!(status, expected, "{origin}");
        }
    }

    #[tokio::test]
    async fn signaling_relays_and_refuses_binary() {
        let route = route();
        let mut server = warp::test::ws()
            .path("/peer/peerjs?key=&id=server")
            .header("origin", "https://tauri.localhost")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut server).await;

        let mut client = warp::test::ws()
            .path("/peer/peerjs?id=client1")
            .header("origin", "http://localhost:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut client).await;

        client
            .send_text(r#"{"type":"OFFER","dst":"server","payload":{"sdp":"x"}}"#)
            .await;
        let offer = tokio::time::timeout(WAIT, server.recv())
            .await
            .unwrap()
            .unwrap();
        let offer: serde_json::Value = serde_json::from_str(offer.to_str().unwrap()).unwrap();
        assert_eq!(offer["type"], "OFFER");
        assert_eq!(offer["src"], "client1");

        // An OFFER to an unknown peer comes back as EXPIRE.
        client
            .send_text(r#"{"type":"OFFER","dst":"nobody","payload":{}}"#)
            .await;
        let expire = tokio::time::timeout(WAIT, client.recv())
            .await
            .unwrap()
            .unwrap();
        let expire: serde_json::Value = serde_json::from_str(expire.to_str().unwrap()).unwrap();
        assert_eq!(expire["type"], "EXPIRE");
        assert_eq!(expire["src"], "nobody");

        // A second registration of the same id is refused with ID-TAKEN.
        let mut dup = warp::test::ws()
            .path("/peer/peerjs?id=server")
            .header("origin", "http://localhost:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        let taken = tokio::time::timeout(WAIT, dup.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(taken.to_str().unwrap(), r#"{"type":"ID-TAKEN"}"#);

        client.send(Message::binary(vec![1u8, 2, 3])).await;
        let closed = tokio::time::timeout(WAIT, client.recv()).await.unwrap();
        assert!(closed.is_err(), "expected close, got {closed:?}");
    }

    #[tokio::test]
    async fn lan_mirrors_carry_the_token_as_the_peerjs_key() {
        let (route, _guard) = route_for(true, Box::new(LanOnly));
        // The host (as if it were remote here) and a LAN /client with the token can signal.
        let mut server = warp::test::ws()
            .path(&format!("/peer/peerjs?key={TOKEN}&id=server&token=r4nd0m&version=1.5.5"))
            .header("origin", "https://tauri.localhost")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut server).await;
        let mut client = warp::test::ws()
            .path(&format!("/peer/peerjs?key={TOKEN}&id=lan-client&token=r4nd0m&version=1.5.5"))
            .header("origin", "http://192.168.1.50:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut client).await;
        client
            .send_text(r#"{"type":"OFFER","dst":"server","payload":{"sdp":"x"}}"#)
            .await;
        let offer = tokio::time::timeout(WAIT, server.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(offer.to_str().unwrap().contains(r#""src":"lan-client""#));

        // PeerJS's own `token` parameter is not the LAN token, and an empty key is no key.
        for path in [
            format!("/peer/peerjs?key=&id=intruder&token={TOKEN}"),
            "/peer/peerjs?id=intruder".to_string(),
            format!("/peer/peerjs?key={OTHER_TOKEN}&id=intruder"),
        ] {
            let mut intruder = warp::test::ws()
                .path(&path)
                .header("origin", "http://192.168.1.50:3030")
                .handshake(route.clone())
                .await
                .unwrap();
            let first = tokio::time::timeout(WAIT, intruder.recv())
                .await
                .unwrap()
                .unwrap();
            let first: serde_json::Value = serde_json::from_str(first.to_str().unwrap()).unwrap();
            assert_eq!(first["type"], "INVALID-KEY", "{path}");
            assert!(!first.to_string().contains(TOKEN), "never echo a token");
            let closed = tokio::time::timeout(WAIT, intruder.recv()).await.unwrap();
            assert!(closed.is_err(), "{path}: expected close, got {closed:?}");
            // It never registered: an OFFER to it expires.
            client
                .send_text(r#"{"type":"OFFER","dst":"intruder","payload":{}}"#)
                .await;
            let expire = tokio::time::timeout(WAIT, client.recv())
                .await
                .unwrap()
                .unwrap();
            assert!(expire.to_str().unwrap().contains("EXPIRE"), "{path}");
        }
    }

    #[tokio::test]
    async fn refused_mirrors_get_invalid_key_and_a_close_code() {
        let (route, _) = route_for(true, Box::new(LanOnly));
        let outcome = raw_connect(route, "/peer/peerjs?key=&id=x&token=abc", Some("http://192.168.1.50:3030"), WAIT).await;
        let RawOutcome::Closed { texts, code, reason } = outcome else {
            panic!("expected a close, got {outcome:?}");
        };
        assert_eq!(code, 4401);
        assert_eq!(reason, "missing_token");
        assert_eq!(texts.len(), 1);
        assert!(texts[0].contains(r#""type":"INVALID-KEY""#), "{texts:?}");

        let guard = test_support::guard(3030, true, Box::new(LanOnly), test_support::policy(BindMode::Localhost, TOKEN, false));
        let outcome = raw_connect(path(guard), &format!("/peer/peerjs?key={TOKEN}&id=x"), Some("http://192.168.1.50:3030"), WAIT).await;
        let RawOutcome::Closed { code, reason, .. } = outcome else {
            panic!("expected a close, got {outcome:?}");
        };
        assert_eq!((code, reason.as_str()), (4403, "lan_disabled"));
    }

    #[tokio::test]
    async fn a_new_token_closes_lan_mirrors() {
        let (route, guard) = route_for(true, Box::new(LanOnly));
        let mut client = warp::test::ws()
            .path(&format!("/peer/peerjs?key={TOKEN}&id=lan-client"))
            .header("origin", "http://192.168.1.50:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut client).await;
        guard.set_token(OTHER_TOKEN.to_string());
        let closed = tokio::time::timeout(WAIT, client.recv()).await.unwrap();
        assert!(closed.is_err() || closed.unwrap().is_close());
        // Its id is free again (once the handler has cleaned up) for a client with the new token.
        tokio::time::sleep(Duration::from_millis(200)).await;
        let mut again = warp::test::ws()
            .path(&format!("/peer/peerjs?key={OTHER_TOKEN}&id=lan-client"))
            .header("origin", "http://192.168.1.50:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        let open = tokio::time::timeout(WAIT, again.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(open.to_str().unwrap(), r#"{"type":"OPEN"}"#);
    }

    // -- Departures reach the host's data channel ------------------------------------------------

    type Client = warp::test::WsClient;

    async fn recv_json(socket: &mut Client) -> serde_json::Value {
        let msg = tokio::time::timeout(WAIT, socket.recv())
            .await
            .expect("a message")
            .expect("still open");
        serde_json::from_str(msg.to_str().unwrap()).unwrap()
    }

    async fn expect_quiet(socket: &mut Client, what: &str) {
        let got = tokio::time::timeout(Duration::from_millis(250), socket.recv()).await;
        assert!(got.is_err(), "{what}: expected nothing, got {got:?}");
    }

    async fn expect_closed(socket: &mut Client, what: &str) {
        let closed = tokio::time::timeout(WAIT, socket.recv())
            .await
            .unwrap_or_else(|_| panic!("{what}: still open"));
        assert!(closed.is_err() || closed.unwrap().is_close(), "{what}");
    }

    async fn expect_leave(socket: &mut Client, src: &str, dst: &str) {
        let leave = recv_json(socket).await;
        assert_eq!(leave["type"], "LEAVE", "{leave}");
        assert_eq!(leave["src"], src, "{leave}");
        assert_eq!(leave["dst"], dst, "{leave}");
    }

    /// The host webview: a plain loopback socket (this PC, never needs the token).
    async fn host(route: &BoxedFilter<(Response,)>) -> Client {
        let mut host = warp::test::ws()
            .path("/peer/peerjs?key=&id=server")
            .header("origin", "https://tauri.localhost")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut host).await;
        host
    }

    /// A `/client` mirror on another machine. Warp's test sockets all come from 127.0.0.1, so this
    /// one goes through a forwarding proxy, which never counts as this PC.
    async fn lan_mirror(route: &BoxedFilter<(Response,)>, id: &str, key: Option<&str>) -> Client {
        let key = key.map(|k| format!("key={k}&")).unwrap_or_default();
        let mut mirror = warp::test::ws()
            .path(&format!("/peer/peerjs?{key}id={id}"))
            .header("origin", "http://localhost:3030")
            .header("x-forwarded-for", "192.168.1.77")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut mirror).await;
        mirror
    }

    /// A `/client` mirror on this PC (OBS here).
    async fn local_mirror(route: &BoxedFilter<(Response,)>, id: &str) -> Client {
        let mut mirror = warp::test::ws()
            .path(&format!("/peer/peerjs?key=&id={id}"))
            .header("origin", "http://localhost:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut mirror).await;
        mirror
    }

    /// The WebRTC offer/answer as signaling sees it: OFFER mirror → host, ANSWER host → mirror.
    async fn connect_to_host(host: &mut Client, mirror: &mut Client, id: &str) {
        mirror
            .send_text(r#"{"type":"OFFER","dst":"server","payload":{"sdp":"x","connectionId":"dc_1"}}"#)
            .await;
        let offer = recv_json(host).await;
        assert_eq!((offer["type"].as_str(), offer["src"].as_str()), (Some("OFFER"), Some(id)));
        host.send_text(format!(r#"{{"type":"ANSWER","dst":"{id}","payload":{{"sdp":"y","connectionId":"dc_1"}}}}"#))
            .await;
        let answer = recv_json(mirror).await;
        assert_eq!((answer["type"].as_str(), answer["src"].as_str()), (Some("ANSWER"), Some("server")));
    }

    #[tokio::test]
    async fn revoking_a_mirror_tells_the_host_to_close_its_data_channel() {
        for (case, initial, key, tightened) in [
            ("new token", lan_policy(), Some(TOKEN), test_support::policy(BindMode::Lan, OTHER_TOKEN, false)),
            ("this PC only", lan_policy(), Some(TOKEN), test_support::policy(BindMode::Localhost, TOKEN, false)),
            ("legacy off", test_support::policy(BindMode::Lan, TOKEN, true), None, lan_policy()),
        ] {
            let guard = test_support::guard(3030, true, Box::new(NoInterfaces), initial);
            let route = path(guard.clone());
            let mut host = host(&route).await;
            let mut lan = lan_mirror(&route, "lan-client", key).await;
            connect_to_host(&mut host, &mut lan, "lan-client").await;
            let mut obs_here = local_mirror(&route, "obs-here").await;
            connect_to_host(&mut host, &mut obs_here, "obs-here").await;

            assert!(guard.set_lan_policy(tightened), "{case}");
            expect_closed(&mut lan, case).await;
            // PeerJS on the host closes every connection with a peer that leaves; without this
            // the data channel (document + pubsub) would outlive the revoked signaling socket.
            expect_leave(&mut host, "lan-client", "server").await;
            expect_quiet(&mut host, case).await;
            // The mirror on this PC keeps signaling.
            obs_here
                .send_text(r#"{"type":"CANDIDATE","dst":"server","payload":{"candidate":"c"}}"#)
                .await;
            let candidate = recv_json(&mut host).await;
            assert_eq!(candidate["src"], "obs-here", "{case}");
        }
    }

    #[tokio::test]
    async fn every_departure_reaches_the_peers_it_signaled() {
        let (route, _guard) = route_for(true, Box::new(NoInterfaces));
        let mut host = host(&route).await;
        let mut lan = lan_mirror(&route, "lan-client", Some(TOKEN)).await;
        connect_to_host(&mut host, &mut lan, "lan-client").await;

        // A peer that never signaled the host leaves without a word to it.
        let mut bystander = local_mirror(&route, "bystander").await;
        bystander.send(Message::close()).await;
        expect_quiet(&mut host, "bystander").await;

        // A second socket refused the mirror's id announces nothing, and the mirror stays.
        let mut dup = lan_mirror_refused_id(&route, "lan-client").await;
        expect_closed(&mut dup, "duplicate").await;
        expect_quiet(&mut host, "duplicate").await;

        // A mirror that drops its own signaling socket (a client ignoring the rules would keep its
        // data channel and dodge a later revocation) leaves the host at once.
        lan.send(Message::close()).await;
        expect_leave(&mut host, "lan-client", "server").await;

        // The host leaving tells the mirrors it answered, so they reconnect at once.
        let mut obs_here = local_mirror(&route, "obs-here").await;
        connect_to_host(&mut host, &mut obs_here, "obs-here").await;
        host.send(Message::close()).await;
        expect_leave(&mut obs_here, "server", "obs-here").await;
    }

    async fn lan_mirror_refused_id(route: &BoxedFilter<(Response,)>, id: &str) -> Client {
        let mut dup = warp::test::ws()
            .path(&format!("/peer/peerjs?key={TOKEN}&id={id}"))
            .header("origin", "http://localhost:3030")
            .header("x-forwarded-for", "192.168.1.78")
            .handshake(route.clone())
            .await
            .unwrap();
        let taken = recv_json(&mut dup).await;
        assert_eq!(taken["type"], "ID-TAKEN");
        dup
    }

    #[test]
    fn contacted_peers_are_bounded() {
        let (tx, _rx) = mpsc::unbounded_channel();
        let mut peer = Registered::new(tx);
        for i in 0..MAX_CONTACTED {
            peer.remember(&format!("p{i}"));
        }
        peer.remember("p0");
        assert!(!peer.contacted_overflow);
        assert_eq!(peer.contacted.len(), MAX_CONTACTED);
        // One more: forget the list and announce the departure to everyone instead.
        peer.remember("one-more");
        assert!(peer.contacted_overflow);
        assert!(peer.contacted.is_empty());
        peer.remember("p1");
        assert!(peer.contacted.is_empty());
        assert_eq!(
            leave_message("a", "server").to_str().unwrap(),
            r#"{"type":"LEAVE","src":"a","dst":"server","payload":null}"#
        );
    }
}
