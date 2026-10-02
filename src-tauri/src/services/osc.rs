//! `osc` plugin: VRChat OSC.
//!
//! **Send** (`send`): one OSC message to `host:port` (VRChat listens on 9000). Sockets are picked by
//! destination: `127.0.0.1:0` for loopback (bound eagerly; its result is `bind_status`),
//! `0.0.0.0:0` for a VRChat on another PC (a socket bound to loopback cannot reach the LAN), and
//! the IPv6 equivalents, each bound on first use. `localhost` always means 127.0.0.1 (Windows
//! resolves it to `::1` first, which an IPv4 VRChat never hears); other host names are resolved
//! once per [`DNS_CACHE_TTL`], preferring IPv4.
//!
//! **Receive** (P6-C; `listen_start` / `listen_stop` / `listen_status`): a UDP receiver for what
//! VRChat sends, today `/avatar/parameters/MuteSelf` (VRChat's microphone mute):
//! - With OSCQuery (VRChat on this PC): a free loopback port advertised over mDNS
//!   ([`osc_query`]), so other OSC apps keep port 9001. VRChat's own OSCQuery service is browsed
//!   to read its current MuteSelf at once, and to notice when VRChat quits.
//! - Otherwise, or when OSCQuery cannot start: the configured port (default 9001) on 127.0.0.1, or
//!   on 0.0.0.0 when VRChat runs on another PC (packets are then accepted only from that PC and
//!   loopback). It never shares a port another OSC app holds: a fixed port on 127.0.0.1 is probed
//!   on 0.0.0.0 first, because Windows would let it share one held there and take that app's
//!   loopback traffic. A bind failure (port taken) is reported in the status and retried every
//!   [`BIND_RETRY`]; sending is never affected.
//!
//! MuteSelf changes are emitted as [`MUTE_SELF_EVENT`] (`{muted, origin}`), deduplicated (VRChat
//! re-sends parameters on avatar changes). With `muteSync`, a change to muted also closes the
//! capture hub's STT gate right here, before the event reaches the webview, so the speech engines
//! stop getting microphone audio within milliseconds even when the UI is busy; the webview then
//! mutes speech-to-text (`stt.setMuted`), which drives the same gate. Unmuting is left to the
//! webview, which knows whether the user had muted speech-to-text themselves.
//!
//! Receiver state is emitted as [`RECEIVER_EVENT`] ([`OscReceiverStatus`]), VRChat appearing or
//! quitting (OSCQuery browse) as [`VRCHAT_EVENT`]. Epoch discipline: every start / stop bumps the
//! epoch under the stop lock (`Receiver::run`) and the epoch lock; every emit and gate change holds
//! the epoch lock with its run's epoch current, so a stopped or replaced run never emits.

use std::collections::HashMap;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, ToSocketAddrs, UdpSocket};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use rosc::{encoder, OscMessage, OscPacket, OscType};
use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Manager, RunEvent, Runtime, State,
};

use super::osc_query::{self, Advertiser, HostInfo, QueryServer, VrchatEvent, FALLBACK_LISTEN_PORT, MUTE_SELF};

/// Tauri event: VRChat's MuteSelf changed ([`MuteSelfEvent`]).
pub const MUTE_SELF_EVENT: &str = "osc:mute_self";
/// Tauri event: the receiver's state changed ([`OscReceiverStatus`]).
pub const RECEIVER_EVENT: &str = "osc:receiver";
/// Tauri event: VRChat on this PC appeared or quit ([`VrchatPresenceEvent`]; OSCQuery browse only,
/// never sent because the receiver restarted).
pub const VRCHAT_EVENT: &str = "osc:vrchat";

/// How long a resolved host name is reused.
const DNS_CACHE_TTL: Duration = Duration::from_secs(30);
/// Retry interval of a receiver whose port is taken.
const BIND_RETRY: Duration = Duration::from_secs(5);
/// Receive timeout (bounds how long a stop can take if the wake datagram is lost).
const RECV_TIMEOUT: Duration = Duration::from_millis(500);
/// Timeout of the HTTP query for VRChat's current MuteSelf.
const VRCHAT_QUERY_TIMEOUT: Duration = Duration::from_millis(1500);
/// Largest UDP datagram.
const MAX_DATAGRAM: usize = 65_536;
/// Bundles nested deeper than this are ignored.
const MAX_BUNDLE_DEPTH: usize = 8;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

// ---------------------------------------------------------------------------------------------
// Send
// ---------------------------------------------------------------------------------------------

/// Local socket used for a destination.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SendSocket {
    LoopbackV4,
    AnyV4,
    LoopbackV6,
    AnyV6,
}

impl SendSocket {
    fn for_destination(ip: IpAddr) -> Self {
        match ip.to_canonical() {
            IpAddr::V4(v4) if v4.is_loopback() => Self::LoopbackV4,
            IpAddr::V4(_) => Self::AnyV4,
            IpAddr::V6(v6) if v6.is_loopback() => Self::LoopbackV6,
            IpAddr::V6(_) => Self::AnyV6,
        }
    }

    fn index(self) -> usize {
        self as usize
    }

    fn bind_addr(self) -> SocketAddr {
        match self {
            Self::LoopbackV4 => SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
            Self::AnyV4 => SocketAddr::from((Ipv4Addr::UNSPECIFIED, 0)),
            Self::LoopbackV6 => SocketAddr::from((Ipv6Addr::LOCALHOST, 0)),
            Self::AnyV6 => SocketAddr::from((Ipv6Addr::UNSPECIFIED, 0)),
        }
    }

    fn bind(self) -> Result<UdpSocket, String> {
        let addr = self.bind_addr();
        UdpSocket::bind(addr).map_err(|e| format!("UDP bind {addr} failed: {e}"))
    }
}

struct OscSender {
    /// One slot per [`SendSocket`]; a failed bind is retried on the next send.
    sockets: Mutex<[Option<Result<UdpSocket, String>>; 4]>,
    dns: Mutex<HashMap<String, (IpAddr, Instant)>>,
}

impl OscSender {
    fn new() -> Self {
        let sender = Self {
            sockets: Mutex::new([None, None, None, None]),
            dns: Mutex::new(HashMap::new()),
        };
        lock(&sender.sockets)[SendSocket::LoopbackV4.index()] = Some(SendSocket::LoopbackV4.bind());
        sender
    }

    /// Result of the loopback socket's bind (the common VRChat-on-this-PC case).
    fn loopback_bind_error(&self) -> Option<String> {
        match &lock(&self.sockets)[SendSocket::LoopbackV4.index()] {
            Some(Err(e)) => Some(e.clone()),
            _ => None,
        }
    }

    fn resolve(&self, host: &str, port: u16) -> Result<SocketAddr, String> {
        if port == 0 {
            return Err("OSC destination port must not be 0".into());
        }
        if let Some(ip) = literal_ip(host)? {
            return Ok(SocketAddr::new(ip, port));
        }
        let key = host.trim().to_ascii_lowercase();
        if let Some((ip, at)) = lock(&self.dns).get(&key) {
            if at.elapsed() < DNS_CACHE_TTL {
                return Ok(SocketAddr::new(*ip, port));
            }
        }
        let addr = resolve_host(&key, port)?;
        lock(&self.dns).insert(key, (addr.ip(), Instant::now()));
        Ok(addr)
    }

    fn send(&self, rpc: &RpcOscMessage) -> Result<(), String> {
        let addr = self.resolve(&rpc.host, rpc.port)?;
        let buf = encode_message(rpc)?;
        let kind = SendSocket::for_destination(addr.ip());
        let mut sockets = lock(&self.sockets);
        let slot = &mut sockets[kind.index()];
        if !matches!(slot, Some(Ok(_))) {
            *slot = Some(kind.bind());
        }
        let socket = match slot {
            Some(Ok(s)) => s,
            Some(Err(e)) => return Err(e.clone()),
            None => return Err("OSC socket unavailable".into()),
        };
        socket
            .send_to(&buf, addr)
            .map_err(|e| format!("OSC send_to {addr} failed: {e}"))?;
        Ok(())
    }
}

/// `Some(ip)` for an IP literal (`[::1]` too) or `localhost`; `None` for a name to resolve.
fn literal_ip(host: &str) -> Result<Option<IpAddr>, String> {
    let host = host.trim();
    if host.is_empty() {
        return Err("OSC host is empty".into());
    }
    let bare = host.strip_prefix('[').and_then(|h| h.strip_suffix(']')).unwrap_or(host);
    if let Ok(ip) = bare.parse::<IpAddr>() {
        return Ok(Some(ip.to_canonical()));
    }
    if bare.eq_ignore_ascii_case("localhost") {
        return Ok(Some(IpAddr::V4(Ipv4Addr::LOCALHOST)));
    }
    Ok(None)
}

/// DNS lookup preferring IPv4 (VRChat listens on IPv4).
fn resolve_host(host: &str, port: u16) -> Result<SocketAddr, String> {
    let addrs: Vec<SocketAddr> = (host, port)
        .to_socket_addrs()
        .map_err(|e| format!("Invalid OSC host/port {host}:{port}: {e}"))?
        .collect();
    addrs
        .iter()
        .find(|a| a.is_ipv4())
        .or_else(|| addrs.first())
        .map(|a| SocketAddr::new(a.ip().to_canonical(), port))
        .ok_or_else(|| format!("Could not resolve OSC address {host}:{port}"))
}

fn encode_message(rpc: &RpcOscMessage) -> Result<Vec<u8>, String> {
    let args: Vec<OscType> = rpc
        .args
        .iter()
        .map(|arg| match arg {
            OscValue::Bool(v) => OscType::Bool(*v),
            OscValue::Float(v) => OscType::Float(*v as f32),
            OscValue::Int(v) => OscType::Int((*v).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as i32),
            OscValue::String(v) => OscType::String(v.clone()),
        })
        .collect();
    encoder::encode(&OscPacket::Message(OscMessage {
        addr: rpc.path.clone(),
        args,
    }))
    .map_err(|e| format!("OSC encode failed: {e}"))
}

#[derive(Serialize, Deserialize, Debug, Clone)]
enum OscValue {
    Bool(bool),
    Float(f64),
    Int(i64),
    String(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcOscMessage {
    path: String,
    args: Vec<OscValue>,
    #[serde(default = "default_osc_host")]
    host: String,
    #[serde(default = "default_osc_port")]
    port: u16,
}

fn default_osc_host() -> String {
    "127.0.0.1".into()
}

fn default_osc_port() -> u16 {
    9000
}

#[derive(Serialize)]
pub struct OscBindStatus {
    pub ok: bool,
    pub error: Option<String>,
}

// ---------------------------------------------------------------------------------------------
// Receive
// ---------------------------------------------------------------------------------------------

fn default_listen_port() -> u16 {
    FALLBACK_LISTEN_PORT
}

fn default_true() -> bool {
    true
}

/// Receiver settings (`plugin:osc|listen_start`).
#[derive(Deserialize, Serialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct OscListenOptions {
    /// Where VRChat runs (the send host): loopback → OSCQuery or 127.0.0.1; else 0.0.0.0 + filter.
    #[serde(default = "default_osc_host")]
    pub host: String,
    /// UDP port without OSCQuery (0 = [`FALLBACK_LISTEN_PORT`]).
    #[serde(default = "default_listen_port")]
    pub port: u16,
    /// Advertise a free port with OSCQuery (VRChat on this PC only).
    #[serde(default = "default_true")]
    pub osc_query: bool,
    /// Close the capture hub's STT gate as soon as VRChat reports a mute.
    #[serde(default)]
    pub mute_sync: bool,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum ReceiverMode {
    /// Not listening.
    #[default]
    Off,
    /// A free port advertised with OSCQuery.
    OscQuery,
    /// A fixed port (VRChat's default 9001 unless changed).
    Port,
}

/// Where a MuteSelf value came from.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum MuteOrigin {
    /// An OSC message from VRChat.
    Osc,
    /// Read from VRChat's OSCQuery server (the value when Sigil found VRChat).
    OscQuery,
}

/// Payload of [`MUTE_SELF_EVENT`].
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct MuteSelfEvent {
    pub muted: bool,
    pub origin: MuteOrigin,
}

/// Payload of [`VRCHAT_EVENT`].
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct VrchatPresenceEvent {
    pub present: bool,
}

/// Receiver state (`listen_status`, [`RECEIVER_EVENT`]).
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct OscReceiverStatus {
    /// A socket is bound and receiving.
    pub running: bool,
    pub mode: ReceiverMode,
    /// Bound UDP address (`127.0.0.1:53124`).
    pub address: Option<String>,
    /// Bound (or wanted, while retrying) UDP port.
    pub port: Option<u16>,
    /// Port of Sigil's OSCQuery HTTP server.
    pub osc_query_port: Option<u16>,
    /// Sigil is announced over mDNS.
    pub advertised: bool,
    /// VRChat's OSCQuery service on this PC was seen.
    pub vrchat_found: bool,
    /// Last MuteSelf value from VRChat (`None` = not known yet).
    pub mute_self: Option<bool>,
    /// OSC datagrams accepted since the start.
    pub packets: u64,
    /// Milliseconds since the last accepted datagram.
    pub last_packet_ago_ms: Option<u64>,
    /// Why the receiver is not running (e.g. port taken).
    pub error: Option<String>,
    /// Running, but not the way asked (e.g. OSCQuery unavailable, fixed port used instead).
    pub warning: Option<String>,
}

/// Where receiver events go (the app in production, a recorder in tests).
trait ReceiverSink: Send + Sync {
    fn mute_self(&self, event: &MuteSelfEvent);
    fn status(&self, status: &OscReceiverStatus);
    fn vrchat(&self, event: &VrchatPresenceEvent);
}

struct AppSink<R: Runtime>(AppHandle<R>);

impl<R: Runtime> ReceiverSink for AppSink<R> {
    fn mute_self(&self, event: &MuteSelfEvent) {
        if let Err(e) = self.0.emit(MUTE_SELF_EVENT, event) {
            tracing::warn!(target: "sigil::osc", "emit {MUTE_SELF_EVENT} failed: {e}");
        }
    }

    fn status(&self, status: &OscReceiverStatus) {
        let _ = self.0.emit(RECEIVER_EVENT, status);
    }

    fn vrchat(&self, event: &VrchatPresenceEvent) {
        let _ = self.0.emit(VRCHAT_EVENT, event);
    }
}

/// Closes (true) or opens (false) the capture hub's STT mute gate.
type MuteGate = Arc<dyn Fn(bool) + Send + Sync>;

fn hub_stt_gate() -> MuteGate {
    Arc::new(|muted| match crate::services::audio::hub::global() {
        Ok(hub) => hub.set_muted(crate::services::audio::hub::MuteGroup::Stt, muted),
        Err(e) => tracing::warn!(target: "sigil::osc", "capture hub unavailable for the VRChat mute: {e}"),
    })
}

/// Stop request of one run, waking a receive loop or a bind-retry wait at once.
struct StopSignal {
    stopped: Mutex<bool>,
    cv: Condvar,
    flag: AtomicBool,
}

impl StopSignal {
    fn new() -> Self {
        Self {
            stopped: Mutex::new(false),
            cv: Condvar::new(),
            flag: AtomicBool::new(false),
        }
    }

    fn stop(&self) {
        self.flag.store(true, Ordering::Release);
        *lock(&self.stopped) = true;
        self.cv.notify_all();
    }

    fn is_stopped(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    /// Waits up to `d`; true when stopped.
    fn wait(&self, d: Duration) -> bool {
        let guard = lock(&self.stopped);
        let (guard, _) = self
            .cv
            .wait_timeout_while(guard, d, |stopped| !*stopped)
            .unwrap_or_else(PoisonError::into_inner);
        *guard
    }
}

/// State shared by the controller and every run.
struct Shared {
    /// Current run epoch (0 = never started). Emits and gate changes hold this lock.
    epoch: Mutex<u64>,
    status: Mutex<OscReceiverStatus>,
    packets: AtomicU64,
    last_packet: Mutex<Option<Instant>>,
}

impl Shared {
    fn snapshot(&self) -> OscReceiverStatus {
        let mut s = lock(&self.status).clone();
        s.packets = self.packets.load(Ordering::Relaxed);
        s.last_packet_ago_ms = lock(&self.last_packet).map(|t| t.elapsed().as_millis().min(u128::from(u64::MAX)) as u64);
        s
    }
}

/// One run of the receiver (one start).
struct RunCore {
    epoch: u64,
    shared: Arc<Shared>,
    sink: Arc<dyn ReceiverSink>,
    gate: MuteGate,
    mute_sync: bool,
    /// Non-loopback sources accepted (the remote VRChat PC).
    allowed: Vec<IpAddr>,
    /// Last MuteSelf seen (dedup).
    mute: Mutex<Option<bool>>,
}

impl RunCore {
    /// Applies `f` to the status and emits it, if this run is still current.
    fn update(&self, f: impl FnOnce(&mut OscReceiverStatus)) {
        let epoch = lock(&self.shared.epoch);
        if *epoch != self.epoch {
            return;
        }
        f(&mut lock(&self.shared.status));
        let snap = self.shared.snapshot();
        self.sink.status(&snap);
    }

    fn source_allowed(&self, ip: IpAddr) -> bool {
        let ip = ip.to_canonical();
        ip.is_loopback() || self.allowed.contains(&ip)
    }

    /// A MuteSelf value from VRChat: deduplicated, gate first (mute only), then the event.
    fn observe_mute(&self, muted: bool, origin: MuteOrigin) {
        let epoch = lock(&self.shared.epoch);
        if *epoch != self.epoch {
            return;
        }
        {
            let mut last = lock(&self.mute);
            if *last == Some(muted) {
                return;
            }
            *last = Some(muted);
        }
        if muted && self.mute_sync {
            (self.gate)(true);
        }
        tracing::info!(target: "sigil::osc", "VRChat MuteSelf = {muted} ({origin:?})");
        self.sink.mute_self(&MuteSelfEvent { muted, origin });
        lock(&self.shared.status).mute_self = Some(muted);
        let snap = self.shared.snapshot();
        self.sink.status(&snap);
    }

    /// One datagram from `from`.
    fn handle_datagram(&self, from: SocketAddr, data: &[u8]) {
        if data.is_empty() || !self.source_allowed(from.ip()) {
            return;
        }
        // A run being replaced must not count into the next run's statistics.
        if *lock(&self.shared.epoch) != self.epoch {
            return;
        }
        let first = self.shared.packets.fetch_add(1, Ordering::Relaxed) == 0;
        *lock(&self.shared.last_packet) = Some(Instant::now());
        if first {
            // The status shows "receiving" from now on.
            self.update(|_| {});
        }
        // VRChat streams every avatar parameter here; only MuteSelf (or a bundle) is worth decoding.
        if !data.starts_with(b"#bundle\0") && !is_mute_self_message(data) {
            return;
        }
        let Ok((_, packet)) = rosc::decoder::decode_udp(data) else {
            return;
        };
        let mut mute = None;
        visit_messages(&packet, 0, &mut |msg| {
            if msg.addr == MUTE_SELF {
                if let Some(v) = msg.args.first().and_then(osc_bool) {
                    mute = Some(v);
                }
            }
        });
        if let Some(muted) = mute {
            self.observe_mute(muted, MuteOrigin::Osc);
        }
    }

    /// Emits [`VRCHAT_EVENT`] if this run is still current.
    fn presence(&self, present: bool) {
        let epoch = lock(&self.shared.epoch);
        if *epoch == self.epoch {
            self.sink.vrchat(&VrchatPresenceEvent { present });
        }
    }

    fn vrchat_event(&self, ev: VrchatEvent) {
        match ev {
            VrchatEvent::Found(addr) => {
                self.update(|s| s.vrchat_found = true);
                self.presence(true);
                match osc_query::query_value(addr, MUTE_SELF, VRCHAT_QUERY_TIMEOUT) {
                    Ok(Some(v)) => {
                        if let Some(muted) = osc_query::value_as_bool(&v) {
                            self.observe_mute(muted, MuteOrigin::OscQuery);
                        }
                    }
                    Ok(None) => {}
                    Err(e) => tracing::debug!(target: "sigil::osc", "VRChat MuteSelf query failed: {e}"),
                }
            }
            VrchatEvent::Lost => {
                // VRChat quit (osc_query confirms an mDNS removal before reporting it): its next
                // start reports the mute state afresh.
                *lock(&self.mute) = None;
                self.update(|s| {
                    s.vrchat_found = false;
                    s.mute_self = None;
                });
                self.presence(false);
            }
            VrchatEvent::Error(e) => self.update(|s| {
                s.warning = Some(format!("Sigil may not be visible to VRChat (mDNS: {e}). If mute sync does not react, turn off OSCQuery to use port {FALLBACK_LISTEN_PORT}."));
            }),
        }
    }
}

/// Whether a raw OSC message is addressed to MuteSelf (the address is NUL-terminated).
fn is_mute_self_message(data: &[u8]) -> bool {
    data.len() > MUTE_SELF.len() && data.starts_with(MUTE_SELF.as_bytes()) && data[MUTE_SELF.len()] == 0
}

fn visit_messages(packet: &OscPacket, depth: usize, f: &mut impl FnMut(&OscMessage)) {
    match packet {
        OscPacket::Message(m) => f(m),
        OscPacket::Bundle(b) if depth < MAX_BUNDLE_DEPTH => {
            for p in &b.content {
                visit_messages(p, depth + 1, f);
            }
        }
        OscPacket::Bundle(_) => {}
    }
}

/// An OSC argument read as a bool (VRChat sends MuteSelf as a bool; ints and floats tolerated).
fn osc_bool(arg: &OscType) -> Option<bool> {
    match arg {
        OscType::Bool(b) => Some(*b),
        OscType::Int(i) => Some(*i != 0),
        OscType::Long(i) => Some(*i != 0),
        OscType::Float(f) => Some(*f > 0.5),
        OscType::Double(f) => Some(*f > 0.5),
        _ => None,
    }
}

/// Whether `host` (the VRChat send host) is this PC.
fn is_loopback_host(host: &str) -> bool {
    match literal_ip(host) {
        Ok(Some(ip)) => ip.is_loopback(),
        _ => false,
    }
}

/// How a start binds.
#[derive(Debug, Clone, PartialEq)]
struct BindPlan {
    mode: ReceiverMode,
    bind: SocketAddr,
    allowed: Vec<IpAddr>,
    warning: Option<String>,
}

fn plan(options: &OscListenOptions) -> BindPlan {
    let port = if options.port == 0 { FALLBACK_LISTEN_PORT } else { options.port };
    if is_loopback_host(&options.host) {
        if options.osc_query {
            return BindPlan {
                mode: ReceiverMode::OscQuery,
                bind: SocketAddr::from((Ipv4Addr::LOCALHOST, 0)),
                allowed: Vec::new(),
                warning: None,
            };
        }
        return BindPlan {
            mode: ReceiverMode::Port,
            bind: SocketAddr::from((Ipv4Addr::LOCALHOST, port)),
            allowed: Vec::new(),
            warning: None,
        };
    }
    // VRChat on another PC: listen on every interface, accept that PC only. OSCQuery only finds a
    // VRChat on this PC, so `osc_query` does not apply here (and is not worth a warning: the
    // settings hide it for another PC and explain the fixed port instead).
    let (allowed, resolve_warning) = match literal_ip(&options.host) {
        Ok(Some(ip)) => (vec![ip], None),
        Ok(None) => match resolve_host(options.host.trim(), port) {
            Ok(addr) => (vec![addr.ip()], None),
            Err(e) => (Vec::new(), Some(e)),
        },
        Err(e) => (Vec::new(), Some(e)),
    };
    BindPlan {
        mode: ReceiverMode::Port,
        bind: SocketAddr::from((Ipv4Addr::UNSPECIFIED, port)),
        allowed,
        warning: resolve_warning.map(|e| format!("Only this PC can send: {e}")),
    }
}

/// Whether a bind error means another socket holds the port (`WSAEADDRINUSE`, or `WSAEACCES` for
/// one bound with exclusive access or protected from a takeover).
fn port_taken(e: &std::io::Error) -> bool {
    e.kind() == std::io::ErrorKind::AddrInUse
        || e.kind() == std::io::ErrorKind::PermissionDenied
        || matches!(e.raw_os_error(), Some(10048) | Some(10013))
}

fn port_taken_message(port: u16, e: &std::io::Error) -> String {
    format!("UDP port {port} is in use by another app (often another OSC tool). Turn on OSCQuery, close that app, or pick another port. ({e})")
}

/// Binds the receive socket, with a friendly message for a port another app holds.
///
/// A fixed port on a specific address (`127.0.0.1:9001`) is first probed on the wildcard address:
/// Windows lets the specific bind succeed while another OSC app (VRCOSC, OSCLeash, a python-osc
/// script, VRChat itself on 9000) listens on `0.0.0.0:<port>`, and Sigil would then silently take
/// every loopback datagram meant for that app (checked on Windows 11; `SO_EXCLUSIVEADDRUSE` on the
/// specific bind does not prevent it). The probe is released at once. A later app binding the
/// wildcard cannot be stopped this way (it then gets only non-loopback traffic), while one binding
/// Sigil's own address is refused by Windows (a plain socket is protected from `SO_REUSEADDR`).
fn bind_receiver(addr: SocketAddr) -> Result<UdpSocket, String> {
    if addr.port() != 0 && !addr.ip().is_unspecified() {
        let wildcard = match addr {
            SocketAddr::V4(_) => SocketAddr::from((Ipv4Addr::UNSPECIFIED, addr.port())),
            SocketAddr::V6(_) => SocketAddr::from((Ipv6Addr::UNSPECIFIED, addr.port())),
        };
        match UdpSocket::bind(wildcard) {
            Ok(probe) => drop(probe),
            Err(e) if port_taken(&e) => return Err(port_taken_message(addr.port(), &e)),
            // Anything else (no IPv6 stack, ...) is left to the real bind below.
            Err(e) => tracing::debug!(target: "sigil::osc", "UDP probe of {wildcard} failed: {e}"),
        }
    }
    let socket = UdpSocket::bind(addr).map_err(|e| {
        if port_taken(&e) {
            port_taken_message(addr.port(), &e)
        } else {
            format!("UDP bind {addr} failed: {e}")
        }
    })?;
    socket
        .set_read_timeout(Some(RECV_TIMEOUT))
        .map_err(|e| format!("UDP socket setup failed: {e}"))?;
    Ok(socket)
}

/// A started run: its threads and services.
struct Run {
    options: OscListenOptions,
    stop: Arc<StopSignal>,
    wake: Arc<Mutex<Option<SocketAddr>>>,
    udp: Option<JoinHandle<()>>,
    query: Option<QueryServer>,
    advertiser: Option<Advertiser>,
}

impl Run {
    fn shutdown(mut self) {
        self.stop.stop();
        if let Some(addr) = *lock(&self.wake) {
            // Wake the blocking receive (an empty datagram is ignored).
            let target = if addr.ip().is_unspecified() {
                SocketAddr::from((Ipv4Addr::LOCALHOST, addr.port()))
            } else {
                addr
            };
            if let Ok(s) = UdpSocket::bind(SocketAddr::from((Ipv4Addr::LOCALHOST, 0))) {
                let _ = s.send_to(&[], target);
            }
        }
        if let Some(t) = self.udp.take() {
            let _ = t.join();
        }
        if let Some(adv) = self.advertiser.take() {
            adv.stop();
        }
        if let Some(mut q) = self.query.take() {
            q.stop();
        }
    }
}

fn receive_loop(core: &RunCore, socket: &UdpSocket, stop: &StopSignal) {
    let mut buf = vec![0u8; MAX_DATAGRAM];
    while !stop.is_stopped() {
        match socket.recv_from(&mut buf) {
            Ok((n, from)) => {
                if stop.is_stopped() {
                    break;
                }
                core.handle_datagram(from, &buf[..n]);
            }
            Err(e) if matches!(e.kind(), std::io::ErrorKind::WouldBlock | std::io::ErrorKind::TimedOut) => {}
            // Windows reports an ICMP "port unreachable" for an earlier datagram as a reset on UDP.
            Err(e) if e.kind() == std::io::ErrorKind::ConnectionReset => {}
            Err(e) => {
                tracing::debug!(target: "sigil::osc", "OSC receive error: {e}");
                if stop.wait(Duration::from_millis(100)) {
                    break;
                }
            }
        }
    }
}

/// The receiver controller (plugin state).
struct Receiver {
    shared: Arc<Shared>,
    /// The stop lock: the running run. Start and stop hold it for their whole duration.
    run: Mutex<Option<Run>>,
    sink: Arc<dyn ReceiverSink>,
    gate: MuteGate,
}

impl Receiver {
    fn new(sink: Arc<dyn ReceiverSink>, gate: MuteGate) -> Self {
        Self {
            shared: Arc::new(Shared {
                epoch: Mutex::new(0),
                status: Mutex::new(OscReceiverStatus::default()),
                packets: AtomicU64::new(0),
                last_packet: Mutex::new(None),
            }),
            run: Mutex::new(None),
            sink,
            gate,
        }
    }

    fn status(&self) -> OscReceiverStatus {
        self.shared.snapshot()
    }

    /// Bumps the epoch (stopping every emit of older runs) and resets the status to `status`.
    fn new_epoch(&self, status: OscReceiverStatus) -> u64 {
        let mut epoch = lock(&self.shared.epoch);
        *epoch += 1;
        *lock(&self.shared.status) = status;
        self.shared.packets.store(0, Ordering::Relaxed);
        *lock(&self.shared.last_packet) = None;
        *epoch
    }

    /// Emits the current status if `epoch` is still current.
    fn emit_status(&self, epoch: u64) -> OscReceiverStatus {
        let guard = lock(&self.shared.epoch);
        let snap = self.shared.snapshot();
        if *guard == epoch {
            self.sink.status(&snap);
        }
        snap
    }

    fn stop(&self) {
        let mut run = lock(&self.run);
        let Some(old) = run.take() else {
            return;
        };
        let epoch = self.new_epoch(OscReceiverStatus::default());
        old.shutdown();
        tracing::info!(target: "sigil::osc", "OSC receiver stopped");
        self.emit_status(epoch);
    }

    fn start(&self, options: OscListenOptions) -> OscReceiverStatus {
        let mut run = lock(&self.run);
        if let Some(current) = run.as_ref() {
            if current.options == options {
                return self.status();
            }
        }
        let old = run.take();
        let plan = plan(&options);
        let epoch = self.new_epoch(OscReceiverStatus {
            mode: plan.mode,
            port: Some(plan.bind.port()).filter(|p| *p != 0),
            warning: plan.warning.clone(),
            ..Default::default()
        });
        if let Some(old) = old {
            old.shutdown();
        }

        let core = Arc::new(RunCore {
            epoch,
            shared: self.shared.clone(),
            sink: self.sink.clone(),
            gate: self.gate.clone(),
            mute_sync: options.mute_sync,
            allowed: plan.allowed.iter().map(|ip| ip.to_canonical()).collect(),
            mute: Mutex::new(None),
        });
        let stop = Arc::new(StopSignal::new());
        let wake = Arc::new(Mutex::new(None));
        let mut new_run = Run {
            options: options.clone(),
            stop: stop.clone(),
            wake: wake.clone(),
            udp: None,
            query: None,
            advertiser: None,
        };

        let mut socket: Option<UdpSocket> = None;
        let mut bind_at = plan.bind;
        if plan.mode == ReceiverMode::OscQuery {
            match self.start_osc_query(&core) {
                Ok((s, query, advertiser)) => {
                    let local = s.local_addr().ok();
                    let http = query.port();
                    socket = Some(s);
                    new_run.query = Some(query);
                    new_run.advertiser = Some(advertiser);
                    if let Some(local) = local {
                        bind_at = local;
                    }
                    let mut st = lock(&self.shared.status);
                    st.osc_query_port = Some(http);
                    st.advertised = true;
                    st.port = Some(bind_at.port());
                    tracing::info!(target: "sigil::osc", "OSC receiver on {bind_at} (OSCQuery HTTP port {http})");
                }
                Err(e) => {
                    let port = if options.port == 0 { FALLBACK_LISTEN_PORT } else { options.port };
                    tracing::warn!(target: "sigil::osc", "OSCQuery unavailable ({e}); falling back to port {port}");
                    bind_at = SocketAddr::from((Ipv4Addr::LOCALHOST, port));
                    let mut st = lock(&self.shared.status);
                    st.mode = ReceiverMode::Port;
                    st.port = Some(port);
                    st.warning = Some(format!("OSCQuery could not start ({e}); listening on port {port} instead."));
                }
            }
        }
        if socket.is_none() {
            match bind_receiver(bind_at) {
                Ok(s) => {
                    socket = Some(s);
                    tracing::info!(target: "sigil::osc", "OSC receiver on {bind_at}");
                }
                Err(e) => {
                    tracing::warn!(target: "sigil::osc", "{e}");
                    lock(&self.shared.status).error = Some(e);
                }
            }
        }
        if let Some(s) = &socket {
            let local = s.local_addr().unwrap_or(bind_at);
            *lock(&wake) = Some(local);
            let mut st = lock(&self.shared.status);
            st.running = true;
            st.address = Some(local.to_string());
            st.port = Some(local.port());
        }

        let thread_core = core.clone();
        let thread_stop = stop.clone();
        let thread_wake = wake.clone();
        let spawned = thread::Builder::new().name("osc-receiver".into()).spawn(move || {
            let mut socket = socket;
            loop {
                if thread_stop.is_stopped() {
                    return;
                }
                if let Some(s) = socket.as_ref() {
                    receive_loop(&thread_core, s, &thread_stop);
                    return;
                }
                if thread_stop.wait(BIND_RETRY) {
                    return;
                }
                match bind_receiver(bind_at) {
                    Ok(s) => {
                        let local = s.local_addr().unwrap_or(bind_at);
                        *lock(&thread_wake) = Some(local);
                        tracing::info!(target: "sigil::osc", "OSC receiver on {local} (port free again)");
                        thread_core.update(|st| {
                            st.running = true;
                            st.error = None;
                            st.address = Some(local.to_string());
                            st.port = Some(local.port());
                        });
                        socket = Some(s);
                    }
                    Err(e) => thread_core.update(|st| {
                        if st.error.as_deref() != Some(e.as_str()) {
                            st.error = Some(e);
                        }
                    }),
                }
            }
        });
        match spawned {
            Ok(handle) => new_run.udp = Some(handle),
            Err(e) => {
                let mut st = lock(&self.shared.status);
                st.running = false;
                st.error = Some(format!("OSC receiver thread failed: {e}"));
            }
        }
        *run = Some(new_run);
        self.emit_status(epoch)
    }

    /// Binds a free loopback port, serves OSCQuery for it and announces both over mDNS.
    fn start_osc_query(&self, core: &Arc<RunCore>) -> Result<(UdpSocket, QueryServer, Advertiser), String> {
        let socket = bind_receiver(SocketAddr::from((Ipv4Addr::LOCALHOST, 0)))?;
        let local = socket.local_addr().map_err(|e| format!("UDP socket address: {e}"))?;
        let instance = osc_query::instance_name();
        let query = QueryServer::start(
            osc_query::SIGIL_ENDPOINTS,
            HostInfo {
                name: instance.clone(),
                osc_ip: local.ip(),
                osc_port: local.port(),
            },
        )
        .map_err(|e| format!("OSCQuery HTTP server: {e}"))?;
        let events = core.clone();
        let advertiser = Advertiser::start(&instance, query.port(), local.port(), move |ev| events.vrchat_event(ev))?;
        Ok((socket, query, advertiser))
    }
}

/// Managed state of the `osc` plugin.
pub struct OscPlugin {
    sender: OscSender,
    receiver: Arc<Receiver>,
}

impl OscPlugin {
    fn new(sink: Arc<dyn ReceiverSink>, gate: MuteGate) -> Self {
        Self {
            sender: OscSender::new(),
            receiver: Arc::new(Receiver::new(sink, gate)),
        }
    }
}

#[command]
fn send(rpc: RpcOscMessage, state: State<OscPlugin>) -> Result<(), String> {
    state.sender.send(&rpc)
}

#[command]
fn bind_status(state: State<OscPlugin>) -> OscBindStatus {
    match state.sender.loopback_bind_error() {
        Some(err) => OscBindStatus {
            ok: false,
            error: Some(err),
        },
        None => OscBindStatus { ok: true, error: None },
    }
}

/// Starts the receiver, or restarts it with new options (identical options: no change).
#[command]
async fn listen_start(options: OscListenOptions, state: State<'_, OscPlugin>) -> Result<OscReceiverStatus, String> {
    let receiver = state.receiver.clone();
    tauri::async_runtime::spawn_blocking(move || receiver.start(options))
        .await
        .map_err(|e| format!("OSC receiver task failed: {e}"))
}

#[command]
async fn listen_stop(state: State<'_, OscPlugin>) -> Result<(), String> {
    let receiver = state.receiver.clone();
    tauri::async_runtime::spawn_blocking(move || receiver.stop())
        .await
        .map_err(|e| format!("OSC receiver task failed: {e}"))
}

#[command]
fn listen_status(state: State<OscPlugin>) -> OscReceiverStatus {
    state.receiver.status()
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("osc")
        .invoke_handler(tauri::generate_handler![send, bind_status, listen_start, listen_stop, listen_status])
        .setup(|app, _api| {
            app.manage(OscPlugin::new(Arc::new(AppSink(app.clone())), hub_stt_gate()));
            Ok(())
        })
        .on_event(|app, event| {
            if let RunEvent::Exit = event {
                // mDNS goodbye, threads joined (bounded waits).
                if let Some(state) = app.try_state::<OscPlugin>() {
                    state.receiver.stop();
                }
            }
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rosc::OscBundle;

    #[derive(Default)]
    struct Recorder {
        mutes: Mutex<Vec<MuteSelfEvent>>,
        statuses: Mutex<Vec<OscReceiverStatus>>,
        presence: Mutex<Vec<bool>>,
    }

    impl ReceiverSink for Recorder {
        fn mute_self(&self, event: &MuteSelfEvent) {
            lock(&self.mutes).push(event.clone());
        }
        fn status(&self, status: &OscReceiverStatus) {
            lock(&self.statuses).push(status.clone());
        }
        fn vrchat(&self, event: &VrchatPresenceEvent) {
            lock(&self.presence).push(event.present);
        }
    }

    struct Harness {
        rec: Arc<Recorder>,
        gate: Arc<Mutex<Vec<bool>>>,
        receiver: Receiver,
    }

    fn harness() -> Harness {
        let rec = Arc::new(Recorder::default());
        let gate = Arc::new(Mutex::new(Vec::new()));
        let g = gate.clone();
        let receiver = Receiver::new(rec.clone(), Arc::new(move |m| lock(&g).push(m)));
        Harness { rec, gate, receiver }
    }

    fn port_options(mute_sync: bool) -> OscListenOptions {
        OscListenOptions {
            host: "127.0.0.1".into(),
            // A free port picked by the OS, released just before the receiver binds it.
            port: UdpSocket::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port(),
            osc_query: false,
            mute_sync,
        }
    }

    fn mute_packet(muted: bool) -> Vec<u8> {
        encoder::encode(&OscPacket::Message(OscMessage {
            addr: MUTE_SELF.into(),
            args: vec![OscType::Bool(muted)],
        }))
        .unwrap()
    }

    fn wait_for(mut cond: impl FnMut() -> bool) -> bool {
        let deadline = Instant::now() + Duration::from_secs(3);
        while Instant::now() < deadline {
            if cond() {
                return true;
            }
            thread::sleep(Duration::from_millis(5));
        }
        cond()
    }

    #[test]
    fn localhost_and_ip_literals_skip_dns_and_prefer_ipv4() {
        assert_eq!(literal_ip("localhost").unwrap(), Some(IpAddr::V4(Ipv4Addr::LOCALHOST)));
        assert_eq!(literal_ip(" 192.168.1.5 ").unwrap(), Some("192.168.1.5".parse().unwrap()));
        assert_eq!(literal_ip("[::1]").unwrap(), Some(IpAddr::V6(Ipv6Addr::LOCALHOST)));
        assert_eq!(literal_ip("::ffff:10.0.0.1").unwrap(), Some("10.0.0.1".parse().unwrap()));
        assert_eq!(literal_ip("vrchat-pc").unwrap(), None);
        assert!(literal_ip("  ").is_err());
    }

    #[test]
    fn send_sockets_follow_the_destination() {
        assert_eq!(SendSocket::for_destination("127.0.0.1".parse().unwrap()), SendSocket::LoopbackV4);
        assert_eq!(SendSocket::for_destination("192.168.1.20".parse().unwrap()), SendSocket::AnyV4);
        assert_eq!(SendSocket::for_destination("::1".parse().unwrap()), SendSocket::LoopbackV6);
        assert_eq!(SendSocket::for_destination("fe80::1".parse().unwrap()), SendSocket::AnyV6);
        assert_eq!(SendSocket::for_destination("::ffff:127.0.0.1".parse().unwrap()), SendSocket::LoopbackV4);
        assert!(SendSocket::AnyV4.bind_addr().ip().is_unspecified());
    }

    #[test]
    fn send_reaches_a_loopback_listener_and_rejects_bad_destinations() {
        let sender = OscSender::new();
        assert_eq!(sender.loopback_bind_error(), None);
        let listener = UdpSocket::bind("127.0.0.1:0").unwrap();
        listener.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let port = listener.local_addr().unwrap().port();
        let rpc = RpcOscMessage {
            path: "/chatbox/input".into(),
            args: vec![OscValue::String("hi".into()), OscValue::Bool(true), OscValue::Int(i64::MAX)],
            host: "localhost".into(),
            port,
        };
        sender.send(&rpc).unwrap();
        let mut buf = [0u8; 1024];
        let (n, _) = listener.recv_from(&mut buf).unwrap();
        let (_, packet) = rosc::decoder::decode_udp(&buf[..n]).unwrap();
        let OscPacket::Message(m) = packet else { panic!("message expected") };
        assert_eq!(m.addr, "/chatbox/input");
        assert_eq!(m.args, vec![OscType::String("hi".into()), OscType::Bool(true), OscType::Int(i32::MAX)]);
        assert!(sender.send(&RpcOscMessage { port: 0, ..rpc }).is_err());
    }

    #[test]
    fn plans_follow_where_vrchat_runs() {
        let local = plan(&OscListenOptions {
            host: "127.0.0.1".into(),
            port: 9001,
            osc_query: true,
            mute_sync: true,
        });
        assert_eq!(local.mode, ReceiverMode::OscQuery);
        assert_eq!(local.bind, SocketAddr::from((Ipv4Addr::LOCALHOST, 0)));
        let fixed = plan(&OscListenOptions {
            host: "localhost".into(),
            port: 0,
            osc_query: false,
            mute_sync: true,
        });
        assert_eq!(fixed.mode, ReceiverMode::Port);
        assert_eq!(fixed.bind, SocketAddr::from((Ipv4Addr::LOCALHOST, FALLBACK_LISTEN_PORT)));
        let remote = plan(&OscListenOptions {
            host: "192.168.1.20".into(),
            port: 9002,
            osc_query: true,
            mute_sync: true,
        });
        assert_eq!(remote.mode, ReceiverMode::Port);
        assert_eq!(remote.bind, SocketAddr::from((Ipv4Addr::UNSPECIFIED, 9002)));
        assert_eq!(remote.allowed, vec!["192.168.1.20".parse::<IpAddr>().unwrap()]);
        // OSCQuery does not apply to another PC; that is no reason for a warning the user cannot clear.
        assert_eq!(remote.warning, None);
    }

    #[test]
    fn osc_bools_and_bundles() {
        assert_eq!(osc_bool(&OscType::Bool(true)), Some(true));
        assert_eq!(osc_bool(&OscType::Int(0)), Some(false));
        assert_eq!(osc_bool(&OscType::Float(1.0)), Some(true));
        assert_eq!(osc_bool(&OscType::String("1".into())), None);
        let bundle = OscPacket::Bundle(OscBundle {
            timetag: (0, 1).into(),
            content: vec![
                OscPacket::Message(OscMessage { addr: "/avatar/parameters/VelocityX".into(), args: vec![OscType::Float(0.2)] }),
                OscPacket::Message(OscMessage { addr: MUTE_SELF.into(), args: vec![OscType::Bool(true)] }),
            ],
        });
        let mut seen = Vec::new();
        visit_messages(&bundle, 0, &mut |m| seen.push(m.addr.clone()));
        assert_eq!(seen, vec!["/avatar/parameters/VelocityX".to_string(), MUTE_SELF.to_string()]);

        // The pre-filter skips other parameters without decoding them, and keeps MuteSelf and bundles.
        assert!(is_mute_self_message(&mute_packet(true)));
        let other = encoder::encode(&OscPacket::Message(OscMessage {
            addr: format!("{MUTE_SELF}X"),
            args: vec![OscType::Bool(true)],
        }))
        .unwrap();
        assert!(!is_mute_self_message(&other));
        assert!(!is_mute_self_message(MUTE_SELF.as_bytes()));

        let core = test_core(Arc::new(Recorder::default()), 1, 1);
        core.handle_datagram("127.0.0.1:9000".parse().unwrap(), &encoder::encode(&bundle).unwrap());
        assert_eq!(*lock(&core.mute), Some(true));
        core.handle_datagram("127.0.0.1:9000".parse().unwrap(), &other);
        assert_eq!(*lock(&core.mute), Some(true));
        assert_eq!(core.shared.packets.load(Ordering::Relaxed), 2);
    }

    /// A run core with `epoch` while the shared epoch is `current`.
    fn test_core(sink: Arc<Recorder>, epoch: u64, current: u64) -> RunCore {
        RunCore {
            epoch,
            shared: Arc::new(Shared {
                epoch: Mutex::new(current),
                status: Mutex::new(OscReceiverStatus::default()),
                packets: AtomicU64::new(0),
                last_packet: Mutex::new(None),
            }),
            sink,
            gate: Arc::new(|_| {}),
            mute_sync: true,
            allowed: Vec::new(),
            mute: Mutex::new(None),
        }
    }

    #[test]
    fn mute_self_closes_the_gate_and_is_deduplicated() {
        let h = harness();
        let status = h.receiver.start(port_options(true));
        assert!(status.running, "{status:?}");
        assert_eq!(status.mode, ReceiverMode::Port);
        let addr: SocketAddr = status.address.clone().unwrap().parse().unwrap();
        let tx = UdpSocket::bind("127.0.0.1:0").unwrap();

        let sent_at = Instant::now();
        tx.send_to(&mute_packet(true), addr).unwrap();
        assert!(wait_for(|| !lock(&h.gate).is_empty()));
        // Acceptance: a VRChat mute stops the engine feed within 200 ms (here: well under it).
        assert!(sent_at.elapsed() < Duration::from_millis(200), "{:?}", sent_at.elapsed());
        assert!(wait_for(|| lock(&h.rec.mutes).len() == 1));
        // The gate closed before the event was emitted, and only for the mute.
        assert_eq!(*lock(&h.gate), vec![true]);
        assert_eq!(lock(&h.rec.mutes)[0], MuteSelfEvent { muted: true, origin: MuteOrigin::Osc });

        // VRChat re-sends parameters (avatar change): no new event, no gate call.
        tx.send_to(&mute_packet(true), addr).unwrap();
        tx.send_to(&mute_packet(false), addr).unwrap();
        assert!(wait_for(|| lock(&h.rec.mutes).len() == 2));
        assert!(!lock(&h.rec.mutes)[1].muted);
        assert_eq!(*lock(&h.gate), vec![true], "unmuting is left to the webview");

        let st = h.receiver.status();
        assert_eq!(st.mute_self, Some(false));
        assert!(st.packets >= 3);
        assert!(st.last_packet_ago_ms.is_some());

        h.receiver.stop();
        let st = h.receiver.status();
        assert!(!st.running && st.mode == ReceiverMode::Off);
        // A stopped receiver emits nothing more.
        let before = lock(&h.rec.mutes).len();
        let _ = tx.send_to(&mute_packet(true), addr);
        thread::sleep(Duration::from_millis(50));
        assert_eq!(lock(&h.rec.mutes).len(), before);
    }

    #[test]
    fn without_mute_sync_the_gate_is_left_alone() {
        let h = harness();
        let status = h.receiver.start(port_options(false));
        let addr: SocketAddr = status.address.unwrap().parse().unwrap();
        let tx = UdpSocket::bind("127.0.0.1:0").unwrap();
        tx.send_to(&mute_packet(true), addr).unwrap();
        assert!(wait_for(|| lock(&h.rec.mutes).len() == 1));
        assert!(lock(&h.gate).is_empty());
        h.receiver.stop();
    }

    #[test]
    fn a_taken_port_is_reported_and_retried_without_failing() {
        let blocker = UdpSocket::bind("127.0.0.1:0").unwrap();
        let port = blocker.local_addr().unwrap().port();
        let h = harness();
        let status = h.receiver.start(OscListenOptions {
            host: "127.0.0.1".into(),
            port,
            osc_query: false,
            mute_sync: true,
        });
        assert!(!status.running);
        assert!(status.error.as_deref().unwrap_or("").contains(&port.to_string()), "{status:?}");
        assert_eq!(status.port, Some(port));
        // Stopping a receiver that waits to retry is immediate.
        let t = Instant::now();
        h.receiver.stop();
        assert!(t.elapsed() < Duration::from_secs(2));
        drop(blocker);
    }

    /// Another OSC tool listening on every interface (`0.0.0.0:<port>`, the common case): on Windows
    /// a plain bind of `127.0.0.1:<port>` would succeed and silently take its loopback traffic.
    #[test]
    fn a_port_held_on_every_interface_is_reported_not_shared() {
        let blocker = UdpSocket::bind("0.0.0.0:0").unwrap();
        let port = blocker.local_addr().unwrap().port();
        let err = bind_receiver(SocketAddr::from((Ipv4Addr::LOCALHOST, port))).unwrap_err();
        assert!(err.contains("in use by another app"), "{err}");

        let h = harness();
        let status = h.receiver.start(OscListenOptions {
            host: "127.0.0.1".into(),
            port,
            osc_query: false,
            mute_sync: true,
        });
        assert!(!status.running, "{status:?}");
        assert!(status.error.as_deref().unwrap_or("").contains("in use by another app"), "{status:?}");
        // The other app still gets what is sent to it over loopback.
        blocker.set_read_timeout(Some(Duration::from_secs(2))).unwrap();
        let tx = UdpSocket::bind("127.0.0.1:0").unwrap();
        tx.send_to(&mute_packet(true), SocketAddr::from((Ipv4Addr::LOCALHOST, port))).unwrap();
        let mut buf = [0u8; 256];
        assert!(blocker.recv_from(&mut buf).is_ok());
        assert!(lock(&h.rec.mutes).is_empty());
        h.receiver.stop();
    }

    #[test]
    fn identical_options_keep_the_run_and_new_options_restart_it() {
        let h = harness();
        let opts = port_options(true);
        let first = h.receiver.start(opts.clone());
        let again = h.receiver.start(opts.clone());
        assert_eq!(first.address, again.address);
        let moved = h.receiver.start(port_options(true));
        assert!(moved.running);
        assert_ne!(first.address, moved.address);
        // The old port is free again.
        let old: SocketAddr = first.address.unwrap().parse().unwrap();
        assert!(UdpSocket::bind(old).is_ok());
        h.receiver.stop();
    }

    #[test]
    fn foreign_sources_are_ignored_when_listening_for_another_pc() {
        let core = RunCore {
            epoch: 1,
            shared: Arc::new(Shared {
                epoch: Mutex::new(1),
                status: Mutex::new(OscReceiverStatus::default()),
                packets: AtomicU64::new(0),
                last_packet: Mutex::new(None),
            }),
            sink: Arc::new(Recorder::default()),
            gate: Arc::new(|_| {}),
            mute_sync: true,
            allowed: vec!["192.168.1.20".parse().unwrap()],
            mute: Mutex::new(None),
        };
        assert!(core.source_allowed("127.0.0.1".parse().unwrap()));
        assert!(core.source_allowed("192.168.1.20".parse().unwrap()));
        assert!(core.source_allowed("::ffff:192.168.1.20".parse().unwrap()));
        assert!(!core.source_allowed("192.168.1.99".parse().unwrap()));
        core.handle_datagram("192.168.1.99:9000".parse().unwrap(), &mute_packet(true));
        assert_eq!(*lock(&core.mute), None);
        assert_eq!(core.shared.packets.load(Ordering::Relaxed), 0);
        core.handle_datagram("192.168.1.20:9000".parse().unwrap(), &mute_packet(true));
        assert_eq!(*lock(&core.mute), Some(true));
        // Garbage is counted as traffic but changes nothing.
        core.handle_datagram("127.0.0.1:9000".parse().unwrap(), b"not osc");
        assert_eq!(*lock(&core.mute), Some(true));
    }

    #[test]
    fn stale_runs_cannot_emit_or_touch_the_gate() {
        let rec = Arc::new(Recorder::default());
        let gate = Arc::new(Mutex::new(Vec::new()));
        let g = gate.clone();
        let shared = Arc::new(Shared {
            epoch: Mutex::new(2),
            status: Mutex::new(OscReceiverStatus::default()),
            packets: AtomicU64::new(0),
            last_packet: Mutex::new(None),
        });
        let core = RunCore {
            epoch: 1,
            shared,
            sink: rec.clone(),
            gate: Arc::new(move |m| lock(&g).push(m)),
            mute_sync: true,
            allowed: Vec::new(),
            mute: Mutex::new(None),
        };
        core.observe_mute(true, MuteOrigin::Osc);
        core.vrchat_event(VrchatEvent::Lost);
        assert!(lock(&rec.mutes).is_empty());
        assert!(lock(&rec.statuses).is_empty());
        assert!(lock(&rec.presence).is_empty());
        assert!(lock(&gate).is_empty());
    }

    #[test]
    fn vrchat_leaving_forgets_its_mute_state() {
        let rec = Arc::new(Recorder::default());
        let core = test_core(rec.clone(), 1, 1);
        core.observe_mute(true, MuteOrigin::OscQuery);
        core.vrchat_event(VrchatEvent::Lost);
        assert_eq!(core.shared.snapshot().mute_self, None);
        assert_eq!(*lock(&rec.presence), vec![false]);
        // The same value after VRChat comes back is a change again.
        core.observe_mute(true, MuteOrigin::OscQuery);
        assert_eq!(lock(&rec.mutes).len(), 2);
    }

    #[test]
    fn wire_format_is_camel_case() {
        let opts: OscListenOptions = serde_json::from_str(r#"{"host":"127.0.0.1","port":9001,"oscQuery":true,"muteSync":true}"#).unwrap();
        assert!(opts.osc_query && opts.mute_sync);
        let defaults: OscListenOptions = serde_json::from_str("{}").unwrap();
        assert_eq!(defaults.port, FALLBACK_LISTEN_PORT);
        assert!(defaults.osc_query && !defaults.mute_sync);
        let ev = serde_json::to_value(MuteSelfEvent { muted: true, origin: MuteOrigin::OscQuery }).unwrap();
        assert_eq!(ev, serde_json::json!({ "muted": true, "origin": "oscQuery" }));
        assert_eq!(
            serde_json::to_value(VrchatPresenceEvent { present: false }).unwrap(),
            serde_json::json!({ "present": false })
        );
        assert_eq!(MUTE_SELF_EVENT, "osc:mute_self");
        assert_eq!(RECEIVER_EVENT, "osc:receiver");
        assert_eq!(VRCHAT_EVENT, "osc:vrchat");
        let st = serde_json::to_value(OscReceiverStatus { mode: ReceiverMode::OscQuery, ..Default::default() }).unwrap();
        for key in ["running", "mode", "address", "port", "oscQueryPort", "advertised", "vrchatFound", "muteSelf", "packets", "lastPacketAgoMs", "error", "warning"] {
            assert!(st.get(key).is_some(), "{key}");
        }
        assert_eq!(st["mode"], "oscQuery");
    }
}
