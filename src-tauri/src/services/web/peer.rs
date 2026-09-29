//! Minimal PeerJS signaling server at `/peer?id=<peer>` (the host registers `server`; `/client`
//! mirrors register random ids and exchange OFFER/ANSWER/CANDIDATE with it).
//!
//! Handshakes pass the [`OriginGuard`](super::origin::OriginGuard) first (403 otherwise); a peer
//! admitted only because the check was off is closed (1008) when the check is turned back on.
//! Binary frames are refused (PeerJS signaling is JSON text only).

use std::{collections::HashMap, sync::Arc};

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

use super::origin::{self, Admission, ConnInfo, OriginGuard};
use super::pubsub::{text_reply, CLOSE_POLICY_VIOLATION, CLOSE_UNSUPPORTED_DATA, MAX_PEER_ID_LEN, REVOKED_REASON};

/// Largest accepted signaling message (SDP offers are a few KB).
pub const MAX_MESSAGE_BYTES: usize = 128 * 1024;

#[derive(Deserialize)]
pub struct PeerQueryData {
    id: String,
}

pub type Peers = Arc<RwLock<HashMap<String, mpsc::UnboundedSender<Result<Message, warp::Error>>>>>;

pub fn path(guard: Arc<OriginGuard>) -> BoxedFilter<(Response,)> {
    let peers = Peers::default();
    let peers = warp::any().map(move || peers.clone());
    let guard = warp::any().map(move || guard.clone());
    warp::path("peer")
        .and(warp::ws())
        .and(origin::conn_info())
        .and(guard)
        .and(peers)
        .and(warp::query::<PeerQueryData>())
        .map(|ws: Ws, info: ConnInfo, guard: Arc<OriginGuard>, peers: Peers, query: PeerQueryData| {
            let Ok(admission) = guard.admit(&info, "peer") else {
                return text_reply(StatusCode::FORBIDDEN, "Forbidden: this origin may not connect to Sigil");
            };
            if query.id.is_empty() || query.id.len() > MAX_PEER_ID_LEN {
                return text_reply(StatusCode::BAD_REQUEST, "Bad Request: invalid id");
            }
            ws.max_message_size(MAX_MESSAGE_BYTES)
                .max_frame_size(MAX_MESSAGE_BYTES)
                .on_upgrade(move |socket| peer_handler(socket, peers, query, admission, guard))
                .into_response()
        })
        .boxed()
}

pub async fn peer_handler(ws: WebSocket, peers: Peers, query: PeerQueryData, mut admission: Admission, guard: Arc<OriginGuard>) {
    let (peer_tx, mut peer_rx) = ws.split();

    let (tx, rx) = mpsc::unbounded_channel();
    let rx = UnboundedReceiverStream::new(rx);
    tauri::async_runtime::spawn(rx.forward(peer_tx));

    if peers.read().await.contains_key(&query.id) {
        tracing::debug!(target: "sigil::web", "[peer] already registered: {:?}", query.id);
        let _ = tx.send(Ok(PeerMessageType::IdTaken.into()));
        return;
    }

    tx.send(Ok(PeerMessageType::Open.into())).ok();

    peers.write().await.insert(query.id.clone(), tx.clone());

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
        if msg.is_binary() {
            tracing::warn!(target: "sigil::web", "[peer] {:?} sent a binary frame; closing", query.id);
            let _ = tx.send(Ok(Message::close_with(CLOSE_UNSUPPORTED_DATA, "binary frames are not accepted")));
            break;
        }
        handle_message(&query.id, msg, &peers).await;
    }
    peers.write().await.remove(&query.id);
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

    let users = users.read().await;

    if msg.t == PeerMessageType::Offer && !users.contains_key(msg.dst.as_str()) {
        let Some(peer_tx) = users.get(peer_id) else { return };
        let Ok(msg_str) = serde_json::to_string(&PeerMessage {
            t: PeerMessageType::Expire,
            src: msg.dst,
            dst: peer_id.to_string(),
            payload: msg.payload,
        }) else {
            return;
        };
        peer_tx.send(Ok(Message::text(msg_str))).ok();
    } else if !msg.dst.is_empty() && users.contains_key(&msg.dst) {
        msg.src = peer_id.to_string();
        let Some(peer_tx) = users.get(&msg.dst) else { return };
        let Ok(msg_str) = serde_json::to_string(&msg) else { return };
        peer_tx.send(Ok(Message::text(msg_str))).ok();
    }
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

    fn guard(check_enabled: bool) -> Arc<OriginGuard> {
        let rules = OriginRules { port: 3030, dev_port: None };
        Arc::new(OriginGuard::new(rules, check_enabled, None, Box::new(NoInterfaces)))
    }

    fn route() -> BoxedFilter<(Response,)> {
        path(guard(true))
    }

    #[tokio::test]
    async fn re_enabling_the_check_closes_foreign_signaling_peers() {
        let guard = guard(false);
        let route = path(guard.clone());
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
            let open = tokio::time::timeout(Duration::from_secs(5), socket.recv())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(open.to_str().unwrap(), r#"{"type":"OPEN"}"#);
        }

        guard.set_setting(true);
        let closed = tokio::time::timeout(Duration::from_secs(5), foreign.recv())
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
        let _open = tokio::time::timeout(Duration::from_secs(5), client.recv())
            .await
            .unwrap()
            .unwrap();
        client
            .send_text(r#"{"type":"OFFER","dst":"server","payload":{"sdp":"x"}}"#)
            .await;
        let offer = tokio::time::timeout(Duration::from_secs(5), host.recv())
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
                .path("/peer/peerjs?key=&id=client1&token=abc")
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
            .path("/peer/peerjs?id=server")
            .header("origin", "https://tauri.localhost")
            .handshake(route.clone())
            .await
            .unwrap();
        let open = tokio::time::timeout(Duration::from_secs(5), server.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(open.to_str().unwrap(), r#"{"type":"OPEN"}"#);

        let mut client = warp::test::ws()
            .path("/peer/peerjs?id=client1")
            .header("origin", "http://localhost:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        let _open = tokio::time::timeout(Duration::from_secs(5), client.recv())
            .await
            .unwrap()
            .unwrap();

        client
            .send_text(r#"{"type":"OFFER","dst":"server","payload":{"sdp":"x"}}"#)
            .await;
        let offer = tokio::time::timeout(Duration::from_secs(5), server.recv())
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
        let expire = tokio::time::timeout(Duration::from_secs(5), client.recv())
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
        let taken = tokio::time::timeout(Duration::from_secs(5), dup.recv())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(taken.to_str().unwrap(), r#"{"type":"ID-TAKEN"}"#);

        client.send(Message::binary(vec![1u8, 2, 3])).await;
        let closed = tokio::time::timeout(Duration::from_secs(5), client.recv())
            .await
            .unwrap();
        assert!(closed.is_err(), "expected close, got {closed:?}");
    }
}
