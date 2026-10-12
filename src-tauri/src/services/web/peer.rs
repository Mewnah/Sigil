//! Minimal PeerJS signaling server at `/peer?id=<peer>[&key=<LAN token>]` (the host registers
//! `server`; `/client` mirrors register random ids and exchange OFFER/ANSWER/CANDIDATE with it).
//!
//! Handshakes pass the [`AccessGuard`] first: a refused Origin gets `403`. A client on another
//! machine must carry the LAN token as the PeerJS `key` (PeerJS's own `token` query parameter is a
//! random per-Peer value, not a credential; it only tells the host's re-registration from a new
//! page, below); without it the socket is upgraded, sent PeerJS's
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
//! the host stops streaming to it. The `/client` mirror keeps an open data channel through the
//! loss of its own signaling socket (a Wi-Fi roam must not blank a source), so this `LEAVE` is
//! what ends the channel of a revoked mirror, together with the host's own side closing it.
//!
//! **The host is the exception, for a moment.** Its signaling socket can drop while its page is
//! alive (a security tool resetting loopback, the web server rebinding): PeerJS then re-registers
//! the same Peer, whose data channels never closed. An immediate `LEAVE` would tear down every OBS
//! source's healthy channel for nothing, so the departure of the host's socket is held back for
//! [`HOST_LEAVE_GRACE`] when it ended by itself (not refused). The host's PeerJS `token` (random
//! per Peer object, kept across its own reconnects) tells what comes back: the same token takes the
//! mirrors over unannounced; a new token (the page reloaded, its channels are dead) or the end of
//! the grace period announces the departure as before. A host without a token is announced at once.
//! The same token also takes over a registration whose old socket the server has not noticed
//! closing (there is no ping here), instead of answering `ID-TAKEN` for as long as that socket
//! lingers, but only from this PC.
//!
//! **A departure nobody could receive is replayed.** A mirror that leaves (revoked, closed,
//! dropped) while no host is registered has nobody to send its `LEAVE` to, yet the host's Peer
//! may still hold that mirror's data channel and would keep streaming to it once it is back. Such
//! departures (of mirrors that had signaled the host) are remembered, bounded by [`MAX_MISSED`],
//! and sent to the next host registration, whichever Peer it is and however late.

use std::{
    collections::{HashMap, HashSet},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
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

use tokio::sync::{mpsc, Notify};
use tokio_stream::wrappers::UnboundedReceiverStream;

use super::auth::{Access, AccessGuard, LanGrant, LanRejected, Refusal};
use super::origin::{self, ConnInfo};
use super::pubsub::{
    refuse_after_upgrade, revocation_frame, text_reply, CLOSE_REPLACED, CLOSE_UNSUPPORTED_DATA, FORBIDDEN_ORIGIN, MAX_PEER_ID_LEN, REPLACED_REASON,
};

/// Largest accepted signaling message (SDP offers are a few KB).
pub const MAX_MESSAGE_BYTES: usize = 128 * 1024;

#[derive(Deserialize)]
pub struct PeerQueryData {
    id: String,
    /// The PeerJS API key, which carries the LAN token (stage 2).
    #[serde(default)]
    key: Option<String>,
    /// PeerJS's random per-Peer-object token (not the LAN token): it stays the same when that Peer
    /// re-registers over a new socket, and differs for a new Peer.
    #[serde(default)]
    token: Option<String>,
}

/// The id the Sigil host webview registers (`startServer` in src/shared/services/peer/index.ts);
/// the `/client` mirrors connect to it.
pub const HOST_PEER_ID: &str = "server";

/// How long the host's departure is held back before the mirrors it signaled are told (see the
/// module docs). The host retries its registration after about 1, 3, 7, 15 and 25 s and then every
/// 10 s (`peerReconnectDelayMs` with `SERVER_RETRY_CEILING_MS` in src/shared/services/peer/
/// provider.ts), and a minimized webview's timers run late: the grace has to outlast the first
/// several attempts, not just the first three, or a slow recovery still tears every source down.
const HOST_LEAVE_GRACE_SECS: u64 = 30;
#[cfg(not(test))]
const HOST_LEAVE_GRACE: Duration = Duration::from_secs(HOST_LEAVE_GRACE_SECS);
#[cfg(test)]
const HOST_LEAVE_GRACE: Duration = Duration::from_millis(1500);

type PeerTx = mpsc::UnboundedSender<Result<Message, warp::Error>>;

/// Source of [`Registered::conn`].
static NEXT_CONNECTION: AtomicU64 = AtomicU64::new(1);

/// Departed mirrors remembered while no host is registered (see the module docs); past this the
/// news is simply not kept (an admitted client could churn through many ids).
const MAX_MISSED: usize = 256;

/// Distinct peers remembered per peer as signaled; past this, its departure is announced to every
/// registered peer instead (an admitted client could address many ids).
const MAX_CONTACTED: usize = 64;

/// A registered signaling socket.
struct Registered {
    /// Tells this socket from a later one with the same id: the cleanup of a socket whose
    /// registration was taken over must not unregister (or announce) its successor.
    conn: u64,
    tx: PeerTx,
    /// Wakes the socket's handler when its registration was taken over, so it ends at once.
    evicted: Arc<Notify>,
    /// The PeerJS token the socket presented, if any (see [`PeerQueryData::token`]).
    token: Option<String>,
    /// Peers this one relayed signaling to (OFFER, ANSWER, CANDIDATE): any WebRTC connection it
    /// has is with one of them, so they are told when it leaves.
    contacted: HashSet<String>,
    /// It signaled more than [`MAX_CONTACTED`] peers: its departure goes to everyone.
    contacted_overflow: bool,
}

impl Registered {
    fn new(tx: PeerTx, token: Option<String>) -> Self {
        Self {
            conn: NEXT_CONNECTION.fetch_add(1, Ordering::Relaxed),
            tx,
            evicted: Arc::new(Notify::new()),
            token,
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

/// The host's departure while it is held back ([`HOST_LEAVE_GRACE`]).
struct Away {
    /// The token of the Peer that left: only it may take the mirrors back unannounced.
    token: String,
    contacted: HashSet<String>,
    contacted_overflow: bool,
    /// Tells this departure from a later one when its timer fires.
    serial: u64,
}

#[derive(Default)]
struct Registry {
    live: HashMap<String, Registered>,
    away: Option<Away>,
    away_serial: u64,
    /// Mirrors that had signaled the host and left while it was not registered: nobody could be
    /// told then, so the next host registration is (see the module docs).
    missed: HashSet<String>,
}

impl Registry {
    /// Send PeerJS's `LEAVE` (src = `id`) to the peers `id` signaled.
    fn announce_departure(&self, id: &str, contacted: &HashSet<String>, contacted_overflow: bool) {
        let notify: Vec<(&String, &Registered)> = if contacted_overflow {
            self.live.iter().collect()
        } else {
            contacted
                .iter()
                .filter_map(|dst| self.live.get_key_value(dst.as_str()))
                .collect()
        };
        for (dst, peer) in notify {
            let _ = peer.tx.send(Ok(leave_message(id, dst)));
        }
    }

    /// A registration of `id` with the same PeerJS token as the live one, from this PC: the host's
    /// Peer re-registering over a new socket while the server still holds the old, dead one.
    fn is_same_host_peer(&self, id: &str, token: Option<&str>, access: &Access) -> bool {
        id == HOST_PEER_ID
            && access.grant.lan == LanGrant::ThisMachine
            && token.is_some()
            && self.live.get(id).is_some_and(|old| old.token.as_deref() == token)
    }

    /// `registered` takes the place of the live socket of the same Peer: it inherits whom that one
    /// signaled, and the old handler is told to end (its cleanup then finds it replaced).
    fn take_over(&mut self, id: &str, registered: &mut Registered) {
        let Some(old) = self.live.remove(id) else {
            return;
        };
        registered.contacted = old.contacted;
        registered.contacted_overflow = old.contacted_overflow;
        let _ = old.tx.send(Ok(Message::close_with(CLOSE_REPLACED, REPLACED_REASON)));
        old.evicted.notify_one();
    }

    /// A mirror left with no host registered to tell: remember it if the host may hold a channel
    /// with it (it signaled the host), for [`Registry::replay_missed`].
    fn note_missed(&mut self, id: &str, gone: &Registered) {
        if id == HOST_PEER_ID || self.live.contains_key(HOST_PEER_ID) {
            return;
        }
        if !(gone.contacted_overflow || gone.contacted.contains(HOST_PEER_ID)) {
            return;
        }
        if self.missed.len() >= MAX_MISSED {
            tracing::debug!(target: "sigil::web", "[peer] too many departures while the host is away; {id:?} is not remembered");
            return;
        }
        self.missed.insert(id.to_string());
    }

    /// The host just registered: send it the `LEAVE` of every mirror that left while it was away,
    /// unless that id has registered again since (its new connections are not the old ones).
    fn replay_missed(&mut self, host_tx: &PeerTx) {
        for gone in std::mem::take(&mut self.missed) {
            if !self.live.contains_key(&gone) {
                let _ = host_tx.send(Ok(leave_message(&gone, HOST_PEER_ID)));
            }
        }
    }

    /// `registered` is about to take `id`: if that is the host coming back while its departure is
    /// held back, either keep its mirrors (same Peer) or announce the departure now (a new one).
    fn settle_away(&mut self, id: &str, registered: &mut Registered) {
        if id != HOST_PEER_ID {
            return;
        }
        let Some(away) = self.away.take() else {
            return;
        };
        if registered.token.as_deref() == Some(away.token.as_str()) {
            registered.contacted = away.contacted;
            registered.contacted_overflow = away.contacted_overflow;
        } else {
            self.announce_departure(id, &away.contacted, away.contacted_overflow);
        }
    }
}

type Peers = Arc<RwLock<Registry>>;

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
            let token = query.token.filter(|t| !t.is_empty() && t.len() <= MAX_PEER_ID_LEN);
            ws.max_message_size(MAX_MESSAGE_BYTES)
                .max_frame_size(MAX_MESSAGE_BYTES)
                .on_upgrade(move |socket| peer_handler(socket, peers, id, token, access, guard))
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

async fn peer_handler(ws: WebSocket, peers: Peers, id: String, token: Option<String>, mut access: Access, guard: Arc<AccessGuard>) {
    let (peer_tx, mut peer_rx) = ws.split();

    let (tx, rx) = mpsc::unbounded_channel();
    let rx = UnboundedReceiverStream::new(rx);
    tauri::async_runtime::spawn(rx.forward(peer_tx));

    // Check and register under one lock, so two sockets can never both hold an id (the loser's
    // departure would otherwise unregister the winner and announce it as gone).
    let (conn, evicted) = {
        let mut registry = peers.write().await;
        let mut registered = Registered::new(tx.clone(), token);
        if registry.live.contains_key(&id) {
            if !registry.is_same_host_peer(&id, registered.token.as_deref(), &access) {
                tracing::debug!(target: "sigil::web", "[peer] already registered: {id:?}");
                let _ = tx.send(Ok(PeerMessageType::IdTaken.into()));
                return;
            }
            // The host's own Peer, back over a new socket before the server noticed the old one
            // closing: answering ID-TAKEN would keep it out until that socket times out.
            tracing::debug!(target: "sigil::web", "[peer] {id:?} re-registered by the same Peer; replacing its old socket");
            registry.take_over(&id, &mut registered);
        } else {
            registry.settle_away(&id, &mut registered);
        }
        let (conn, evicted) = (registered.conn, registered.evicted.clone());
        tx.send(Ok(PeerMessageType::Open.into())).ok();
        registry.live.insert(id.clone(), registered);
        if id == HOST_PEER_ID {
            registry.replay_missed(&tx);
        }
        (conn, evicted)
    };
    tracing::debug!(target: "sigil::web", "[peer] {id:?} registered ({:?})", access.grant);

    // A socket that ended by itself may be the host's Peer about to re-register; one that was
    // refused or misbehaved is announced at once.
    let mut ended_by_itself = true;
    loop {
        let result = tokio::select! {
            next = peer_rx.next() => next,
            rejection = access.revoked(&guard) => {
                let frame = revocation_frame(&rejection);
                guard.report_revoked(rejection);
                let _ = tx.send(Ok(frame));
                ended_by_itself = false;
                break;
            }
            _ = evicted.notified() => break,
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
            ended_by_itself = false;
            break;
        }
        handle_message(&id, msg, &peers).await;
    }
    depart(&peers, &id, conn, ended_by_itself).await;
}

/// Unregister `id` and send PeerJS's `LEAVE` from it to the peers it signaled, which closes their
/// WebRTC connections with it (see the module docs). Done under the registry lock, so each `LEAVE`
/// arrives after everything `id` relayed to that peer and before anything from a later
/// registration of the same id. The host's socket, when it ended by itself and carried a token, is
/// only announced after [`HOST_LEAVE_GRACE`] unless the same Peer re-registers first. A socket whose
/// registration (`conn`) was taken over has nothing left to do.
async fn depart(peers: &Peers, id: &str, conn: u64, ended_by_itself: bool) {
    let mut registry = peers.write().await;
    if registry.live.get(id).map(|registered| registered.conn) != Some(conn) {
        return;
    }
    let Some(gone) = registry.live.remove(id) else {
        return;
    };
    if ended_by_itself && id == HOST_PEER_ID && (!gone.contacted.is_empty() || gone.contacted_overflow) {
        if let Some(token) = gone.token {
            registry.away_serial += 1;
            let serial = registry.away_serial;
            registry.away = Some(Away {
                token,
                contacted: gone.contacted,
                contacted_overflow: gone.contacted_overflow,
                serial,
            });
            let peers = peers.clone();
            tauri::async_runtime::spawn(async move {
                tokio::time::sleep(HOST_LEAVE_GRACE).await;
                announce_held_departure(&peers, serial).await;
            });
            return;
        }
    }
    registry.announce_departure(id, &gone.contacted, gone.contacted_overflow);
    registry.note_missed(id, &gone);
}

/// The grace period of the host's departure `serial` is over and nobody took its mirrors back.
async fn announce_held_departure(peers: &Peers, serial: u64) {
    let mut registry = peers.write().await;
    if registry.away.as_ref().map(|away| away.serial) != Some(serial) {
        return;
    }
    if let Some(away) = registry.away.take() {
        registry.announce_departure(HOST_PEER_ID, &away.contacted, away.contacted_overflow);
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
    let mut registry = users.write().await;
    let users = &mut registry.live;

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

    /// The host webview with the token its PeerJS Peer object carries (PeerJS puts it on the socket
    /// URL and keeps it when the same Peer re-registers).
    async fn host_peer(route: &BoxedFilter<(Response,)>, token: &str) -> Client {
        let mut host = warp::test::ws()
            .path(&format!("/peer/peerjs?key=&id=server&token={token}&version=1.5.5"))
            .header("origin", "https://tauri.localhost")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut host).await;
        host
    }

    #[tokio::test]
    async fn the_hosts_peer_re_registering_keeps_its_mirrors_without_a_leave() {
        let (route, _guard) = route_for(true, Box::new(NoInterfaces));
        let mut host = host_peer(&route, "peer-one").await;
        let mut mirror = local_mirror(&route, "obs-here").await;
        connect_to_host(&mut host, &mut mirror, "obs-here").await;

        // The host's signaling socket drops while its page lives on: PeerJS keeps the Peer and its
        // data channels and registers again. The mirror hears nothing, not even after the grace
        // period the departure was held back for.
        host.send(Message::close()).await;
        expect_quiet(&mut mirror, "host socket dropped").await;
        let mut host = host_peer(&route, "peer-one").await;
        tokio::time::sleep(HOST_LEAVE_GRACE + Duration::from_millis(400)).await;
        expect_quiet(&mut mirror, "after the grace period").await;
        mirror
            .send_text(r#"{"type":"CANDIDATE","dst":"server","payload":{"candidate":"c"}}"#)
            .await;
        let candidate = recv_json(&mut host).await;
        assert_eq!(candidate["src"], "obs-here");

        // It still knows whom to tell: gone for good this time, the mirror hears it once the grace
        // period is over.
        host.send(Message::close()).await;
        expect_quiet(&mut mirror, "held back").await;
        expect_leave(&mut mirror, "server", "obs-here").await;
    }

    #[tokio::test]
    async fn a_new_peer_taking_the_host_id_announces_the_old_ones_departure_at_once() {
        let (route, _guard) = route_for(true, Box::new(NoInterfaces));
        let mut host = host_peer(&route, "peer-one").await;
        let mut mirror = local_mirror(&route, "obs-here").await;
        connect_to_host(&mut host, &mut mirror, "obs-here").await;

        host.send(Message::close()).await;
        expect_quiet(&mut mirror, "host socket dropped").await;
        // The page reloaded: a new Peer, whose data channels are not the mirror's. Waiting out the
        // grace period would leave the mirror on a dead channel.
        let _reloaded = host_peer(&route, "peer-two").await;
        tokio::time::timeout(Duration::from_millis(700), expect_leave(&mut mirror, "server", "obs-here"))
            .await
            .expect("announced when the new Peer registered, not after the grace period");
        // The new registration holds the id; nothing is announced again later.
        tokio::time::sleep(HOST_LEAVE_GRACE + Duration::from_millis(300)).await;
        expect_quiet(&mut mirror, "after the grace period").await;
    }

    #[tokio::test]
    async fn a_mirror_revoked_while_the_host_is_away_is_announced_when_the_host_returns() {
        let guard = test_support::guard(3030, true, Box::new(NoInterfaces), lan_policy());
        let route = path(guard.clone());
        let mut host = host_peer(&route, "peer-one").await;
        let mut lan = lan_mirror(&route, "lan-client", Some(TOKEN)).await;
        connect_to_host(&mut host, &mut lan, "lan-client").await;
        let mut obs_here = local_mirror(&route, "obs-here").await;
        connect_to_host(&mut host, &mut obs_here, "obs-here").await;
        // Never signaled the host, so the host holds no channel with it.
        let mut bystander = local_mirror(&route, "bystander").await;

        // The host's signaling socket drops; its Peer (and the channels) live on and will register again.
        host.send(Message::close()).await;
        expect_quiet(&mut obs_here, "host socket dropped").await;

        // Meanwhile the LAN mirror is revoked (new token) and the bystander closes: the host is not
        // registered, so there is nobody to tell yet.
        assert!(guard.set_lan_policy(test_support::policy(BindMode::Lan, OTHER_TOKEN, false)));
        expect_closed(&mut lan, "revoked mirror").await;
        bystander.send(Message::close()).await;
        tokio::time::sleep(Duration::from_millis(200)).await;

        // The same Peer is back: its mirror on this PC is kept silently, and it learns that the
        // revoked one is gone, so its data channel (document + pubsub) does not outlive the revocation.
        let mut host = host_peer(&route, "peer-one").await;
        expect_leave(&mut host, "lan-client", "server").await;
        expect_quiet(&mut host, "only the revoked mirror, once").await;
        expect_quiet(&mut obs_here, "the mirror on this PC").await;

        // Told once: the next registration of the host hears nothing about it again.
        host.send(Message::close()).await;
        let mut host = host_peer(&route, "peer-one").await;
        expect_quiet(&mut host, "nothing left to replay").await;
    }

    #[tokio::test]
    async fn a_departure_after_the_grace_period_is_replayed_to_the_returning_host_too() {
        let guard = test_support::guard(3030, true, Box::new(NoInterfaces), lan_policy());
        let route = path(guard.clone());
        let mut host = host_peer(&route, "peer-one").await;
        let mut lan = lan_mirror(&route, "lan-client", Some(TOKEN)).await;
        connect_to_host(&mut host, &mut lan, "lan-client").await;

        host.send(Message::close()).await;
        // The grace period ends without the host: the mirror is told, as before.
        tokio::time::sleep(HOST_LEAVE_GRACE + Duration::from_millis(300)).await;
        expect_leave(&mut lan, "server", "lan-client").await;

        // The mirror is revoked later, still with no host registered.
        assert!(guard.set_lan_policy(test_support::policy(BindMode::Lan, OTHER_TOKEN, false)));
        expect_closed(&mut lan, "revoked mirror").await;
        tokio::time::sleep(Duration::from_millis(200)).await;

        // The host's Peer registers again, long after the grace period: it still hears of it.
        let mut host = host_peer(&route, "peer-one").await;
        expect_leave(&mut host, "lan-client", "server").await;
        expect_quiet(&mut host, "once").await;
    }

    #[test]
    fn missed_departures_are_bounded_and_skip_ids_that_registered_again() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let mut registry = Registry::default();
        let signaled_host = |id: &str| {
            let (tx, _rx) = mpsc::unbounded_channel();
            let mut gone = Registered::new(tx, None);
            gone.remember(HOST_PEER_ID);
            (id.to_string(), gone)
        };
        for i in 0..MAX_MISSED + 10 {
            let (id, gone) = signaled_host(&format!("m{i}"));
            registry.note_missed(&id, &gone);
        }
        assert_eq!(registry.missed.len(), MAX_MISSED);

        // A live host has the departure delivered at once instead (see `depart`); the host's own id
        // and mirrors that never signaled it are not remembered.
        let mut registry = Registry::default();
        let never = Registered::new(mpsc::unbounded_channel().0, None);
        registry.note_missed("quiet", &never);
        registry.note_missed(HOST_PEER_ID, &signaled_host("x").1);
        assert!(registry.missed.is_empty());

        let (id, gone) = signaled_host("back-again");
        registry.note_missed(&id, &gone);
        let (id, gone) = signaled_host("gone");
        registry.note_missed(&id, &gone);
        // `back-again` registered again meanwhile: its new connections are not the old ones.
        registry
            .live
            .insert("back-again".to_string(), Registered::new(mpsc::unbounded_channel().0, None));
        registry.replay_missed(&tx);
        let sent: Vec<String> = std::iter::from_fn(|| rx.try_recv().ok())
            .map(|m| m.unwrap().to_str().unwrap().to_string())
            .collect();
        assert_eq!(sent, vec![r#"{"type":"LEAVE","src":"gone","dst":"server","payload":null}"#]);
        assert!(registry.missed.is_empty());
    }

    #[test]
    fn the_grace_period_outlasts_the_hosts_retry_ladder() {
        // The host's attempts (src/shared/services/peer/provider.ts: 1, 2, 4, 8 s, then every 10 s)
        // land at about 1, 3, 7, 15, 25 s. At 10 s the old grace expired after the third.
        const { assert!(HOST_LEAVE_GRACE_SECS >= 25 + 5) };
    }

    #[tokio::test]
    async fn the_same_peer_replaces_a_host_socket_the_server_has_not_noticed_closing() {
        let (route, _guard) = route_for(true, Box::new(NoInterfaces));
        let mut stale = host_peer(&route, "peer-one").await;
        let mut mirror = local_mirror(&route, "obs-here").await;
        connect_to_host(&mut stale, &mut mirror, "obs-here").await;

        // The old socket is dead but still open as far as the server knows. The same Peer registers
        // again: it is let in (not `ID-TAKEN`), keeps whom the old socket signaled, and the mirror
        // hears nothing.
        let mut host = host_peer(&route, "peer-one").await;
        expect_closed(&mut stale, "the replaced socket").await;
        expect_quiet(&mut mirror, "mirror").await;
        mirror
            .send_text(r#"{"type":"CANDIDATE","dst":"server","payload":{"candidate":"c"}}"#)
            .await;
        let candidate = recv_json(&mut host).await;
        assert_eq!(candidate["src"], "obs-here");

        // The replaced socket's cleanup did not unregister its successor or announce anything, not
        // even after the grace period.
        tokio::time::sleep(HOST_LEAVE_GRACE + Duration::from_millis(300)).await;
        expect_quiet(&mut mirror, "after the grace period").await;
        mirror
            .send_text(r#"{"type":"CANDIDATE","dst":"server","payload":{"candidate":"d"}}"#)
            .await;
        assert_eq!(recv_json(&mut host).await["src"], "obs-here");

        // It still knows whom to tell when it is gone for good.
        host.send(Message::close()).await;
        expect_leave(&mut mirror, "server", "obs-here").await;
    }

    #[tokio::test]
    async fn only_the_same_peer_from_this_pc_may_replace_the_host_registration() {
        let (route, _guard) = route_for(true, Box::new(NoInterfaces));
        let mut host = host_peer(&route, "peer-one").await;

        let taken = |mut socket: Client| async move { recv_json(&mut socket).await["type"].clone() };
        // Another Peer, no token at all, or an empty one: the id is taken.
        for path in [
            "/peer/peerjs?key=&id=server&token=peer-two",
            "/peer/peerjs?key=&id=server",
            "/peer/peerjs?key=&id=server&token=",
        ] {
            let socket = warp::test::ws()
                .path(path)
                .header("origin", "https://tauri.localhost")
                .handshake(route.clone())
                .await
                .unwrap();
            assert_eq!(taken(socket).await, "ID-TAKEN", "{path}");
        }
        // The right token from another machine (here: through a forwarding proxy) is not the host.
        let remote = warp::test::ws()
            .path(&format!("/peer/peerjs?key={TOKEN}&id=server&token=peer-one"))
            .header("origin", "http://localhost:3030")
            .header("x-forwarded-for", "192.168.1.77")
            .handshake(route.clone())
            .await
            .unwrap();
        assert_eq!(taken(remote).await, "ID-TAKEN");
        // The registered host is untouched by all of it.
        let mut mirror = local_mirror(&route, "obs-here").await;
        connect_to_host(&mut host, &mut mirror, "obs-here").await;
    }

    #[tokio::test]
    async fn only_the_host_gets_a_grace_period() {
        let (route, _guard) = route_for(true, Box::new(NoInterfaces));
        let mut host = host(&route).await;
        let mut mirror = warp::test::ws()
            .path("/peer/peerjs?key=&id=obs-here&token=mirror-peer")
            .header("origin", "http://localhost:3030")
            .handshake(route.clone())
            .await
            .unwrap();
        expect_open(&mut mirror).await;
        connect_to_host(&mut host, &mut mirror, "obs-here").await;

        // A mirror's departure reaches the host at once, token or not (it may have been revoked).
        mirror.send(Message::close()).await;
        tokio::time::timeout(Duration::from_millis(700), expect_leave(&mut host, "obs-here", "server"))
            .await
            .expect("a mirror's departure is announced at once");
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
        let mut peer = Registered::new(tx, None);
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
