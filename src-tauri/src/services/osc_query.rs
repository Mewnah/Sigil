//! OSCQuery for VRChat (P6-C).
//!
//! VRChat (2023.3.1 and later) finds OSC apps through OSCQuery instead of a fixed port: an app runs
//! a small HTTP server describing the OSC addresses it wants (its "address tree") and advertises
//! it over mDNS as `_oscjson._tcp` (plus its OSC port as `_osc._udp`). VRChat then sends avatar
//! data, including `/avatar/parameters/MuteSelf`, to the advertised UDP port. That port is any
//! free one, so Sigil never fights other OSC tools (face tracking, OSC routers, ...) over UDP
//! 9001, VRChat's default send port. When OSCQuery cannot be used, `osc.rs` listens on
//! [`FALLBACK_LISTEN_PORT`] (or the port the user picked) instead; a failed bind is reported,
//! never fatal, and sending keeps working either way.
//!
//! Pieces (all on this PC only):
//! - [`QueryServer`]: HTTP on `127.0.0.1:<free port>` answering `GET /?HOST_INFO` and the address
//!   tree ([`SIGIL_ENDPOINTS`]: `/avatar/change` and `/avatar/parameters/MuteSelf`). One request
//!   per connection, at most [`MAX_CONNECTIONS`] at once, short timeouts: the only expected
//!   client is VRChat on this PC, queried once when it discovers Sigil.
//! - [`Advertiser`]: the mDNS records (via `mdns-sd`, addresses `127.0.0.1`, so they are only
//!   answered on the loopback interface) and a browse for VRChat's own OSCQuery service
//!   (`VRChat-Client-*`), reported as [`VrchatEvent`]s (a removal only once VRChat's server stops
//!   answering, [`VrchatTracker`]). Sigil reads VRChat's current MuteSelf from
//!   it with [`query_value`] so mute sync starts in the right state, not only on the next change.
//!
//! Everything here is off unless the VRChat settings turn on mute sync with OSCQuery.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{self, Read, Write};
use std::net::{IpAddr, Ipv4Addr, Shutdown, SocketAddr, TcpListener, TcpStream};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use mdns_sd::{DaemonEvent, ServiceDaemon, ServiceEvent, ServiceInfo};
use serde_json::{json, Map, Value};

/// UDP port VRChat sends OSC to when OSCQuery is not used.
pub const FALLBACK_LISTEN_PORT: u16 = 9001;
/// mDNS service type of an OSCQuery HTTP server.
pub const OSCJSON_SERVICE: &str = "_oscjson._tcp.local.";
/// mDNS service type of an OSC (UDP) server.
pub const OSC_SERVICE: &str = "_osc._udp.local.";
/// Instance-name prefix of VRChat's own OSCQuery service (`VRChat-Client-XXXXXX`).
pub const VRCHAT_INSTANCE_PREFIX: &str = "VRChat-Client-";
/// VRChat's microphone mute (bool avatar parameter, sent on every change).
pub const MUTE_SELF: &str = "/avatar/parameters/MuteSelf";
/// Sent by VRChat on every avatar change (the avatar id).
pub const AVATAR_CHANGE: &str = "/avatar/change";

/// Parallel HTTP connections served; more are closed at once.
pub const MAX_CONNECTIONS: usize = 8;
/// Longest request head read (bytes).
const MAX_REQUEST_HEAD: usize = 8 * 1024;
/// Read / write timeout of one HTTP connection.
const HTTP_IO_TIMEOUT: Duration = Duration::from_secs(2);
/// Longest response body read from VRChat's server (bytes).
const MAX_RESPONSE: usize = 1024 * 1024;
/// How long [`Advertiser::stop`] waits for each mDNS goodbye / shutdown acknowledgement.
const MDNS_STOP_WAIT: Duration = Duration::from_millis(500);
/// Poll interval of the browse thread (also bounds how long stopping it takes).
const BROWSE_POLL: Duration = Duration::from_millis(200);
/// How often a VRChat client whose mDNS removal was not confirmed is checked again.
const SUSPECT_RECHECK: Duration = Duration::from_secs(5);
/// Timeout of the check whether a VRChat client's OSCQuery server still answers (it is on this PC:
/// a closed port is refused at once).
const PROBE_TIMEOUT: Duration = Duration::from_millis(500);

/// OSCQuery `ACCESS`: no value (a container).
const ACCESS_NONE: u8 = 0;
/// OSCQuery `ACCESS`: write only (VRChat sends it to Sigil; Sigil has no value to report).
const ACCESS_WRITE_ONLY: u8 = 2;

/// One OSC address Sigil asks VRChat to send.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Endpoint {
    pub path: &'static str,
    /// OSC type tag (`T` = bool, `s` = string, `i` = int, `f` = float).
    pub osc_type: &'static str,
    pub description: &'static str,
}

/// Sigil's address tree. `/avatar/change` makes VRChat treat Sigil as an avatar-data consumer (it
/// sends avatar parameters to services exposing `/avatar` endpoints); MuteSelf is what Sigil
/// reads.
pub const SIGIL_ENDPOINTS: &[Endpoint] = &[
    Endpoint {
        path: AVATAR_CHANGE,
        osc_type: "s",
        description: "Avatar changed",
    },
    Endpoint {
        path: MUTE_SELF,
        osc_type: "T",
        description: "VRChat microphone muted",
    },
];

/// `HOST_INFO` of Sigil's OSCQuery server.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HostInfo {
    pub name: String,
    pub osc_ip: IpAddr,
    pub osc_port: u16,
}

impl HostInfo {
    pub fn to_json(&self) -> Value {
        json!({
            "NAME": self.name,
            "EXTENSIONS": {
                "ACCESS": true,
                "CLIPMODE": false,
                "RANGE": false,
                "TYPE": true,
                "VALUE": false,
                "DESCRIPTION": true,
            },
            "OSC_IP": self.osc_ip.to_string(),
            "OSC_PORT": self.osc_port,
            "OSC_TRANSPORT": "UDP",
        })
    }
}

/// `/a/b/` → `/a/b`; `None` for anything that is not an absolute OSC path.
fn normalize_path(path: &str) -> Option<String> {
    if !path.starts_with('/') {
        return None;
    }
    let trimmed = if path.len() > 1 { path.trim_end_matches('/') } else { path };
    let trimmed = if trimmed.is_empty() { "/" } else { trimmed };
    if trimmed.contains("//") || trimmed.split('/').any(|s| s == "." || s == "..") {
        return None;
    }
    Some(trimmed.to_string())
}

fn endpoint_json(e: &Endpoint) -> Value {
    json!({
        "FULL_PATH": e.path,
        "ACCESS": ACCESS_WRITE_ONLY,
        "TYPE": e.osc_type,
        "DESCRIPTION": e.description,
    })
}

/// The OSCQuery node at `path`: an endpoint, or a container with every endpoint below it.
pub fn node(endpoints: &[Endpoint], path: &str) -> Option<Value> {
    let path = normalize_path(path)?;
    if let Some(e) = endpoints.iter().find(|e| e.path == path) {
        return Some(endpoint_json(e));
    }
    let prefix = if path == "/" { "/".to_string() } else { format!("{path}/") };
    let mut children: BTreeMap<String, Value> = BTreeMap::new();
    for e in endpoints {
        let Some(rest) = e.path.strip_prefix(prefix.as_str()) else {
            continue;
        };
        let Some(name) = rest.split('/').next().filter(|n| !n.is_empty()) else {
            continue;
        };
        if children.contains_key(name) {
            continue;
        }
        if let Some(child) = node(endpoints, &format!("{prefix}{name}")) {
            children.insert(name.to_string(), child);
        }
    }
    if children.is_empty() {
        return None;
    }
    let mut contents = Map::new();
    for (k, v) in children {
        contents.insert(k, v);
    }
    Some(json!({
        "FULL_PATH": path,
        "ACCESS": ACCESS_NONE,
        "DESCRIPTION": if path == "/" { "Sigil" } else { "" },
        "CONTENTS": Value::Object(contents),
    }))
}

/// Node attributes a client may ask for alone (`/path?TYPE`).
const NODE_ATTRIBUTES: &[&str] = &[
    "FULL_PATH",
    "CONTENTS",
    "TYPE",
    "ACCESS",
    "VALUE",
    "DESCRIPTION",
    "RANGE",
    "TAGS",
    "EXTENDED_TYPE",
    "UNIT",
    "CRITICAL",
    "CLIPMODE",
    "OVERLOADS",
];

/// One HTTP reply of the OSCQuery server.
#[derive(Debug, PartialEq)]
pub struct Reply {
    pub status: u16,
    pub body: Option<String>,
}

impl Reply {
    fn json(v: Value) -> Self {
        Self {
            status: 200,
            body: Some(v.to_string()),
        }
    }
    fn status(status: u16) -> Self {
        Self { status, body: None }
    }
}

/// Answers one request (`method`, request target) of the OSCQuery server.
pub fn handle_request(method: &str, target: &str, endpoints: &[Endpoint], host: &HostInfo) -> Reply {
    if method != "GET" && method != "HEAD" {
        return Reply::status(405);
    }
    // Absolute-form targets (`http://127.0.0.1:1234/path`) carry the path after the authority.
    let target = match target.find("://") {
        Some(i) => {
            let after = &target[i + 3..];
            after.find('/').map(|j| &after[j..]).unwrap_or("/")
        }
        None => target,
    };
    let (path, query) = target.split_once('?').unwrap_or((target, ""));
    let attribute = query.split(['&', '=']).next().unwrap_or("").trim();
    if attribute.eq_ignore_ascii_case("HOST_INFO") {
        return Reply::json(host.to_json());
    }
    let Some(node) = node(endpoints, path) else {
        return Reply::status(404);
    };
    if attribute.is_empty() {
        return Reply::json(node);
    }
    let upper = attribute.to_ascii_uppercase();
    if !NODE_ATTRIBUTES.contains(&upper.as_str()) {
        return Reply::status(400);
    }
    match node.get(upper.as_str()) {
        Some(v) => {
            let mut one = Map::new();
            one.insert(upper, v.clone());
            Reply::json(Value::Object(one))
        }
        None => Reply::status(204),
    }
}

fn reason(status: u16) -> &'static str {
    match status {
        200 => "OK",
        204 => "No Content",
        400 => "Bad Request",
        404 => "Not Found",
        405 => "Method Not Allowed",
        _ => "Error",
    }
}

/// Reads a request head (up to the blank line) from `stream`.
fn read_head(stream: &mut impl Read) -> io::Result<String> {
    let mut head = Vec::with_capacity(512);
    let mut chunk = [0u8; 1024];
    loop {
        let n = stream.read(&mut chunk)?;
        if n == 0 {
            break;
        }
        head.extend_from_slice(&chunk[..n]);
        if head.windows(4).any(|w| w == b"\r\n\r\n") || head.windows(2).any(|w| w == b"\n\n") {
            break;
        }
        if head.len() > MAX_REQUEST_HEAD {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "request head too long"));
        }
    }
    Ok(String::from_utf8_lossy(&head).into_owned())
}

fn serve_connection(mut stream: TcpStream, endpoints: &[Endpoint], host: &HostInfo) {
    let _ = stream.set_read_timeout(Some(HTTP_IO_TIMEOUT));
    let _ = stream.set_write_timeout(Some(HTTP_IO_TIMEOUT));
    let reply = match read_head(&mut stream) {
        Ok(head) => {
            let line = head.lines().next().unwrap_or("");
            let mut parts = line.split_whitespace();
            match (parts.next(), parts.next()) {
                (Some(method), Some(target)) => {
                    let reply = handle_request(method, target, endpoints, host);
                    if method == "HEAD" {
                        // Same headers, no body.
                        let len = reply.body.as_ref().map_or(0, String::len);
                        let head = format!(
                            "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
                            reply.status,
                            reason(reply.status)
                        );
                        let _ = stream.write_all(head.as_bytes());
                        let _ = stream.shutdown(Shutdown::Both);
                        return;
                    }
                    reply
                }
                _ => Reply::status(400),
            }
        }
        Err(_) => Reply::status(400),
    };
    let body = reply.body.unwrap_or_default();
    // One write: the whole response leaves in as few packets as possible.
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        reply.status,
        reason(reply.status),
        body.len()
    );
    let _ = stream.write_all(response.as_bytes());
    let _ = stream.flush();
    let _ = stream.shutdown(Shutdown::Both);
}

/// Sigil's OSCQuery HTTP server on `127.0.0.1:<free port>`. Stops on [`QueryServer::stop`] / drop.
pub struct QueryServer {
    port: u16,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
}

impl QueryServer {
    /// Binds a free loopback port and serves `endpoints` with `host` as `HOST_INFO`.
    pub fn start(endpoints: &'static [Endpoint], host: HostInfo) -> io::Result<Self> {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0))?;
        let port = listener.local_addr()?.port();
        let stop = Arc::new(AtomicBool::new(false));
        let active = Arc::new(AtomicUsize::new(0));
        let host = Arc::new(host);
        let stop_flag = stop.clone();
        let thread = thread::Builder::new().name("osc-query-http".into()).spawn(move || {
            for conn in listener.incoming() {
                if stop_flag.load(Ordering::Acquire) {
                    break;
                }
                let stream = match conn {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::debug!(target: "sigil::osc", "OSCQuery accept failed: {e}");
                        // Back off instead of spinning on a persistent error (e.g. no free handles).
                        thread::sleep(Duration::from_millis(50));
                        continue;
                    }
                };
                if active.fetch_add(1, Ordering::AcqRel) >= MAX_CONNECTIONS {
                    active.fetch_sub(1, Ordering::AcqRel);
                    let _ = stream.shutdown(Shutdown::Both);
                    continue;
                }
                let active_conn = active.clone();
                let host = host.clone();
                let spawned = thread::Builder::new().name("osc-query-conn".into()).spawn(move || {
                    serve_connection(stream, endpoints, &host);
                    active_conn.fetch_sub(1, Ordering::AcqRel);
                });
                if spawned.is_err() {
                    active.fetch_sub(1, Ordering::AcqRel);
                }
            }
        })?;
        Ok(Self {
            port,
            stop,
            thread: Some(thread),
        })
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// Stops accepting and joins the accept thread (connections in flight finish on their own).
    pub fn stop(&mut self) {
        let Some(thread) = self.thread.take() else {
            return;
        };
        self.stop.store(true, Ordering::Release);
        // Wake the blocking accept.
        let _ = TcpStream::connect_timeout(&SocketAddr::from((Ipv4Addr::LOCALHOST, self.port)), Duration::from_millis(500));
        let _ = thread.join();
    }
}

impl Drop for QueryServer {
    fn drop(&mut self) {
        self.stop();
    }
}

/// What the mDNS browse learned about VRChat's own OSCQuery service.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VrchatEvent {
    /// A VRChat client on this PC: its OSCQuery HTTP address.
    Found(SocketAddr),
    /// The last VRChat client on this PC went away: mDNS removed it and its OSCQuery server no
    /// longer answers ([`VrchatTracker`]).
    Lost,
    /// mDNS reported a problem (Sigil may not be visible to VRChat).
    Error(String),
}

/// Whether an mDNS instance full name (`VRChat-Client-AB12CD._oscjson._tcp.local.`) is VRChat's.
pub fn is_vrchat_instance(fullname: &str) -> bool {
    fullname.starts_with(VRCHAT_INSTANCE_PREFIX)
}

/// The address to query a VRChat client at: a loopback address first, else one of this PC's own
/// addresses (`local`). A client elsewhere on the LAN is ignored: mute sync follows the VRChat
/// on this PC only.
pub fn pick_local_address(addrs: &[IpAddr], port: u16, local: &[IpAddr]) -> Option<SocketAddr> {
    let canonical: Vec<IpAddr> = addrs.iter().map(|a| a.to_canonical()).collect();
    canonical
        .iter()
        .find(|a| a.is_ipv4() && a.is_loopback())
        .or_else(|| canonical.iter().find(|a| a.is_loopback()))
        .or_else(|| canonical.iter().find(|a| local.contains(a)))
        .map(|ip| SocketAddr::new(*ip, port))
}

fn local_addresses() -> Vec<IpAddr> {
    local_ip_address::list_afinet_netifas()
        .map(|list| list.into_iter().map(|(_, ip)| ip.to_canonical()).collect())
        .unwrap_or_default()
}

/// A random-ish instance name (`Sigil-1A2B3C`): unique enough on one PC, stable for one run.
pub fn instance_name() -> String {
    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0) as u64;
    let mix = (nanos ^ (u64::from(std::process::id()) << 20)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    format!("Sigil-{:06X}", (mix >> 40) & 0xFF_FFFF)
}

/// mDNS records for Sigil's OSCQuery and OSC servers, plus the VRChat browse.
pub struct Advertiser {
    daemon: ServiceDaemon,
    registered: Vec<String>,
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    /// The daemon was shut down (by `stop` or drop).
    shut_down: bool,
}

impl Advertiser {
    /// Registers `instance` as `_oscjson._tcp` on `http_port` and `_osc._udp` on `osc_port`
    /// (address 127.0.0.1) and starts browsing for VRChat, reporting to `on_event` from a
    /// background thread.
    pub fn start(
        instance: &str,
        http_port: u16,
        osc_port: u16,
        on_event: impl FnMut(VrchatEvent) + Send + 'static,
    ) -> Result<Self, String> {
        let daemon = ServiceDaemon::new().map_err(|e| format!("mDNS unavailable: {e}"))?;
        let mut advertiser = Self {
            daemon,
            registered: Vec::new(),
            stop: Arc::new(AtomicBool::new(false)),
            thread: None,
            shut_down: false,
        };
        let host = format!("{}.local.", instance.to_ascii_lowercase());
        let mut txt = HashMap::new();
        txt.insert("txtvers".to_string(), "1".to_string());
        for (ty, port) in [(OSCJSON_SERVICE, http_port), (OSC_SERVICE, osc_port)] {
            let info = ServiceInfo::new(ty, instance, &host, IpAddr::V4(Ipv4Addr::LOCALHOST), port, txt.clone())
                .map_err(|e| format!("mDNS record for {ty} rejected: {e}"))?;
            let fullname = info.get_fullname().to_string();
            advertiser
                .daemon
                .register(info)
                .map_err(|e| format!("mDNS register {ty} failed: {e}"))?;
            advertiser.registered.push(fullname);
        }
        let browse = advertiser
            .daemon
            .browse(OSCJSON_SERVICE)
            .map_err(|e| format!("mDNS browse failed: {e}"))?;
        let monitor = advertiser.daemon.monitor().ok();
        let stop = advertiser.stop.clone();
        let thread = thread::Builder::new()
            .name("osc-query-mdns".into())
            .spawn(move || browse_loop(browse, monitor, stop, on_event))
            .map_err(|e| format!("mDNS browse thread failed: {e}"))?;
        advertiser.thread = Some(thread);
        Ok(advertiser)
    }

    /// Sends goodbyes for Sigil's records, stops the browse and shuts the daemon down (bounded waits).
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        if self.shut_down {
            return;
        }
        self.shut_down = true;
        self.stop.store(true, Ordering::Release);
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
        for fullname in self.registered.drain(..) {
            if let Ok(rx) = self.daemon.unregister(&fullname) {
                let _ = rx.recv_timeout(MDNS_STOP_WAIT);
            }
        }
        if let Ok(rx) = self.daemon.shutdown() {
            let _ = rx.recv_timeout(MDNS_STOP_WAIT);
        }
    }
}

impl Drop for Advertiser {
    fn drop(&mut self) {
        // Also after a failed start: the daemon thread must not outlive its advertiser.
        self.shutdown();
    }
}

/// The VRChat clients on this PC as the mDNS browse reports them, with removals confirmed.
///
/// An mDNS removal is not trusted at once: the daemon listens on every interface, so a goodbye from
/// any LAN host (spoofed or not), or cache churn when a network interface changes, removes a
/// record although VRChat still runs, and "VRChat quit" lifts a mute VRChat caused. A removal
/// counts only once the client's OSCQuery server (on this PC) stops answering; a client that still
/// answers is checked again every [`SUSPECT_RECHECK`] until it stops answering (VRChat quit after
/// all) or mDNS reports it again.
#[derive(Debug, Default)]
pub struct VrchatTracker {
    /// Full name → OSCQuery address.
    clients: HashMap<String, SocketAddr>,
    /// Clients whose removal was reported while their server still answered → next check.
    suspects: HashMap<String, Instant>,
}

impl VrchatTracker {
    /// A VRChat client on this PC was resolved: [`VrchatEvent::Found`] when it is new or moved.
    pub fn resolved(&mut self, fullname: &str, addr: SocketAddr) -> Option<VrchatEvent> {
        self.suspects.remove(fullname);
        if self.clients.insert(fullname.to_string(), addr) == Some(addr) {
            return None;
        }
        tracing::info!(target: "sigil::osc", "VRChat OSCQuery found at {addr}");
        Some(VrchatEvent::Found(addr))
    }

    /// mDNS removed a client: [`VrchatEvent::Lost`] when it was the last one and its server no
    /// longer answers (`alive`).
    pub fn removed(&mut self, fullname: &str, now: Instant, alive: impl Fn(SocketAddr) -> bool) -> Option<VrchatEvent> {
        let addr = *self.clients.get(fullname)?;
        if alive(addr) {
            tracing::info!(target: "sigil::osc", "mDNS removed VRChat's record but its OSCQuery server at {addr} still answers; keeping it");
            self.suspects.insert(fullname.to_string(), now + SUSPECT_RECHECK);
            return None;
        }
        self.forget(fullname)
    }

    /// Re-checks clients whose removal was not confirmed: [`VrchatEvent::Lost`] when the last one
    /// stopped answering.
    pub fn tick(&mut self, now: Instant, alive: impl Fn(SocketAddr) -> bool) -> Option<VrchatEvent> {
        let due: Vec<String> = self
            .suspects
            .iter()
            .filter(|(_, at)| **at <= now)
            .map(|(name, _)| name.clone())
            .collect();
        let mut lost = None;
        for name in due {
            match self.clients.get(&name).copied() {
                Some(addr) if alive(addr) => {
                    self.suspects.insert(name, now + SUSPECT_RECHECK);
                }
                Some(_) => lost = lost.or(self.forget(&name)),
                None => {
                    self.suspects.remove(&name);
                }
            }
        }
        lost
    }

    fn forget(&mut self, fullname: &str) -> Option<VrchatEvent> {
        self.suspects.remove(fullname);
        if self.clients.remove(fullname).is_some() && self.clients.is_empty() {
            tracing::info!(target: "sigil::osc", "VRChat OSCQuery service gone");
            return Some(VrchatEvent::Lost);
        }
        None
    }
}

/// Whether an OSCQuery server still answers at `addr` (`HOST_INFO` with a JSON reply).
pub fn oscquery_alive(addr: SocketAddr) -> bool {
    matches!(get_json(addr, "/?HOST_INFO", PROBE_TIMEOUT), Ok(Some(_)))
}

fn browse_loop(
    browse: mdns_sd::Receiver<ServiceEvent>,
    monitor: Option<mdns_sd::Receiver<DaemonEvent>>,
    stop: Arc<AtomicBool>,
    mut on_event: impl FnMut(VrchatEvent),
) {
    let mut tracker = VrchatTracker::default();
    let mut reported_errors: HashSet<String> = HashSet::new();
    let local = local_addresses();
    while !stop.load(Ordering::Acquire) {
        if let Some(monitor) = &monitor {
            while let Ok(ev) = monitor.try_recv() {
                if let DaemonEvent::Error(e) = ev {
                    let msg = e.to_string();
                    if reported_errors.insert(msg.clone()) {
                        tracing::warn!(target: "sigil::osc", "mDNS: {msg}");
                        on_event(VrchatEvent::Error(msg));
                    }
                }
            }
        }
        if let Some(ev) = tracker.tick(Instant::now(), oscquery_alive) {
            on_event(ev);
        }
        let ev = match browse.recv_timeout(BROWSE_POLL) {
            Ok(ev) => ev,
            Err(mdns_sd::RecvTimeoutError::Timeout) => continue,
            Err(mdns_sd::RecvTimeoutError::Disconnected) => break,
        };
        match ev {
            ServiceEvent::ServiceResolved(info) => {
                if !is_vrchat_instance(info.get_fullname()) {
                    continue;
                }
                let addrs: Vec<IpAddr> = info.get_addresses().iter().map(|a| a.to_ip_addr()).collect();
                let Some(addr) = pick_local_address(&addrs, info.get_port(), &local) else {
                    tracing::debug!(target: "sigil::osc", "ignoring VRChat OSCQuery service not on this PC: {}", info.get_fullname());
                    continue;
                };
                if let Some(ev) = tracker.resolved(info.get_fullname(), addr) {
                    on_event(ev);
                }
            }
            ServiceEvent::ServiceRemoved(_, fullname) => {
                if let Some(ev) = tracker.removed(&fullname, Instant::now(), oscquery_alive) {
                    on_event(ev);
                }
            }
            _ => {}
        }
    }
}

/// Decodes an HTTP/1.1 chunked body.
fn decode_chunked(mut body: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::new();
    loop {
        let line_end = body.windows(2).position(|w| w == b"\r\n")?;
        let size_line = std::str::from_utf8(&body[..line_end]).ok()?;
        let size = usize::from_str_radix(size_line.split(';').next()?.trim(), 16).ok()?;
        body = &body[line_end + 2..];
        if size == 0 {
            return Some(out);
        }
        if body.len() < size || out.len() + size > MAX_RESPONSE {
            return None;
        }
        out.extend_from_slice(&body[..size]);
        body = body.get(size + 2..)?;
    }
}

/// Splits a raw HTTP response into (status, body).
fn parse_response(raw: &[u8]) -> Result<(u16, Vec<u8>), String> {
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .ok_or_else(|| "malformed HTTP response".to_string())?;
    let head = String::from_utf8_lossy(&raw[..split]);
    let mut lines = head.lines();
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| format!("malformed HTTP status line {status_line:?}"))?;
    let chunked = lines.any(|l| {
        let l = l.to_ascii_lowercase();
        l.starts_with("transfer-encoding:") && l.contains("chunked")
    });
    let body = &raw[split + 4..];
    let body = if chunked {
        decode_chunked(body).ok_or_else(|| "malformed chunked HTTP body".to_string())?
    } else {
        body.to_vec()
    };
    Ok((status, body))
}

/// Whether `raw` already holds a whole response with a JSON body (or an empty one).
fn complete_json_response(raw: &[u8]) -> bool {
    match parse_response(raw) {
        Ok((_, body)) => body.is_empty() || serde_json::from_slice::<Value>(&body).is_ok(),
        Err(_) => false,
    }
}

/// `GET <path>` from an OSCQuery server and returns the node's first `VALUE` (`None` when the
/// node has no value or does not exist).
pub fn query_value(addr: SocketAddr, path: &str, timeout: Duration) -> Result<Option<Value>, String> {
    let node = get_json(addr, path, timeout)?;
    Ok(node
        .as_ref()
        .and_then(|n| n.get("VALUE"))
        .and_then(Value::as_array)
        .and_then(|a| a.first())
        .cloned())
}

/// `GET <path>` (path and query) from an OSCQuery server: the JSON body, `None` for 204 / 404.
fn get_json(addr: SocketAddr, path: &str, timeout: Duration) -> Result<Option<Value>, String> {
    let mut stream = TcpStream::connect_timeout(&addr, timeout).map_err(|e| format!("connect {addr}: {e}"))?;
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    // HTTP/1.0: the server closes the connection after the body and never uses chunked encoding.
    // One write, so a server that answers after its first read sees the whole request.
    let request = format!("GET {path} HTTP/1.0\r\nHost: {addr}\r\nAccept: application/json\r\nConnection: close\r\n\r\n");
    stream
        .write_all(request.as_bytes())
        .map_err(|e| format!("request to {addr}: {e}"))?;
    let mut raw = Vec::new();
    let mut chunk = [0u8; 4096];
    loop {
        match stream.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                raw.extend_from_slice(&chunk[..n]);
                if raw.len() > MAX_RESPONSE {
                    return Err(format!("response from {addr} is too large"));
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            // A reset right after a complete response (the server closed with bytes unread) still counts.
            Err(e) => {
                if complete_json_response(&raw) {
                    break;
                }
                return Err(format!("response from {addr}: {e}"));
            }
        }
    }
    let (status, body) = parse_response(&raw)?;
    match status {
        200 => {}
        204 | 404 => return Ok(None),
        other => return Err(format!("HTTP {other} from {addr}")),
    }
    let node: Value = serde_json::from_slice(&body).map_err(|e| format!("invalid OSCQuery JSON from {addr}: {e}"))?;
    Ok(Some(node))
}

/// An OSCQuery / OSC value read as a bool (numbers: non-zero is true).
pub fn value_as_bool(v: &Value) -> Option<bool> {
    match v {
        Value::Bool(b) => Some(*b),
        Value::Number(n) => n.as_f64().map(|f| f.abs() > f64::EPSILON),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn host() -> HostInfo {
        HostInfo {
            name: "Sigil-ABCDEF".into(),
            osc_ip: IpAddr::V4(Ipv4Addr::LOCALHOST),
            osc_port: 54_321,
        }
    }

    fn body(r: &Reply) -> Value {
        serde_json::from_str(r.body.as_deref().expect("body")).expect("json")
    }

    #[test]
    fn tree_exposes_mute_self_and_avatar_change_under_containers() {
        let root = node(SIGIL_ENDPOINTS, "/").unwrap();
        assert_eq!(root["FULL_PATH"], "/");
        assert_eq!(root["ACCESS"], 0);
        let avatar = &root["CONTENTS"]["avatar"];
        assert_eq!(avatar["FULL_PATH"], "/avatar");
        assert_eq!(avatar["CONTENTS"]["change"]["TYPE"], "s");
        let mute = &avatar["CONTENTS"]["parameters"]["CONTENTS"]["MuteSelf"];
        assert_eq!(mute["FULL_PATH"], MUTE_SELF);
        assert_eq!(mute["TYPE"], "T");
        assert_eq!(mute["ACCESS"], 2);
        assert!(mute.get("VALUE").is_none());
    }

    #[test]
    fn nodes_are_found_by_path_with_or_without_trailing_slash() {
        assert_eq!(node(SIGIL_ENDPOINTS, "/avatar/parameters/").unwrap()["FULL_PATH"], "/avatar/parameters");
        assert_eq!(node(SIGIL_ENDPOINTS, MUTE_SELF).unwrap()["TYPE"], "T");
        for missing in ["/nope", "avatar", "", "/avatar//change", "/avatar/../avatar", "/avatar/parameters/Other"] {
            assert!(node(SIGIL_ENDPOINTS, missing).is_none(), "{missing}");
        }
    }

    #[test]
    fn host_info_names_the_udp_port() {
        let r = handle_request("GET", "/?HOST_INFO", SIGIL_ENDPOINTS, &host());
        assert_eq!(r.status, 200);
        let v = body(&r);
        assert_eq!(v["OSC_PORT"], 54_321);
        assert_eq!(v["OSC_IP"], "127.0.0.1");
        assert_eq!(v["OSC_TRANSPORT"], "UDP");
        assert_eq!(v["NAME"], "Sigil-ABCDEF");
        // Any path, case-insensitive, absolute-form targets.
        assert_eq!(handle_request("GET", "/avatar?host_info", SIGIL_ENDPOINTS, &host()).status, 200);
        assert_eq!(
            body(&handle_request("GET", "http://127.0.0.1:1/?HOST_INFO", SIGIL_ENDPOINTS, &host()))["OSC_PORT"],
            54_321
        );
    }

    #[test]
    fn attribute_queries_and_errors() {
        let r = handle_request("GET", "/avatar/parameters/MuteSelf?TYPE", SIGIL_ENDPOINTS, &host());
        assert_eq!(body(&r), json!({ "TYPE": "T" }));
        assert_eq!(handle_request("GET", "/avatar/parameters/MuteSelf?VALUE", SIGIL_ENDPOINTS, &host()).status, 204);
        assert_eq!(handle_request("GET", "/avatar?bogus", SIGIL_ENDPOINTS, &host()).status, 400);
        assert_eq!(handle_request("GET", "/missing", SIGIL_ENDPOINTS, &host()).status, 404);
        assert_eq!(handle_request("POST", "/", SIGIL_ENDPOINTS, &host()).status, 405);
        assert_eq!(handle_request("HEAD", "/", SIGIL_ENDPOINTS, &host()).status, 200);
    }

    #[test]
    fn server_answers_over_loopback_and_query_value_reads_it() {
        let mut server = QueryServer::start(SIGIL_ENDPOINTS, host()).expect("bind loopback");
        let addr = SocketAddr::from((Ipv4Addr::LOCALHOST, server.port()));
        // A write-only node has no VALUE: answered as "no value".
        assert_eq!(query_value(addr, MUTE_SELF, Duration::from_secs(2)).unwrap(), None);
        assert_eq!(query_value(addr, "/nope", Duration::from_secs(2)).unwrap(), None);
        // HOST_INFO over real HTTP.
        let mut s = TcpStream::connect(addr).unwrap();
        write!(s, "GET /?HOST_INFO HTTP/1.1\r\nHost: x\r\n\r\n").unwrap();
        let mut raw = Vec::new();
        s.read_to_end(&mut raw).unwrap();
        let (status, b) = parse_response(&raw).unwrap();
        assert_eq!(status, 200);
        let v: Value = serde_json::from_slice(&b).unwrap();
        assert_eq!(v["OSC_PORT"], 54_321);
        // The liveness check used to confirm an mDNS removal.
        assert!(oscquery_alive(addr));
        server.stop();
        // Stopped: nothing listens any more (connect fails or the connection is closed unanswered).
        let after = query_value(addr, "/", Duration::from_millis(300));
        assert!(after.is_err(), "{after:?}");
        assert!(!oscquery_alive(addr));
    }

    const VRC_A: &str = "VRChat-Client-A1B2C3._oscjson._tcp.local.";
    const VRC_B: &str = "VRChat-Client-D4E5F6._oscjson._tcp.local.";

    fn at(port: u16) -> SocketAddr {
        SocketAddr::from((Ipv4Addr::LOCALHOST, port))
    }

    #[test]
    fn a_removal_counts_only_once_vrchat_stops_answering() {
        let t0 = Instant::now();
        let mut tr = VrchatTracker::default();
        assert_eq!(tr.resolved(VRC_A, at(4000)), Some(VrchatEvent::Found(at(4000))));
        // mDNS re-resolves the same client: nothing new.
        assert_eq!(tr.resolved(VRC_A, at(4000)), None);
        // A goodbye from the LAN (or cache churn) while VRChat still answers: ignored.
        assert_eq!(tr.removed(VRC_A, t0, |_| true), None);
        // Re-checked later: still there, still ignored; nothing before the recheck is due.
        assert_eq!(tr.tick(t0 + SUSPECT_RECHECK / 2, |_| panic!("not due yet")), None);
        assert_eq!(tr.tick(t0 + SUSPECT_RECHECK, |_| true), None);
        // VRChat quits for real: the next check reports it.
        assert_eq!(tr.tick(t0 + SUSPECT_RECHECK * 2, |a| a != at(4000)), Some(VrchatEvent::Lost));
        // Nothing left to check, and an unknown removal changes nothing.
        assert_eq!(tr.tick(t0 + SUSPECT_RECHECK * 10, |_| panic!("nothing to check")), None);
        assert_eq!(tr.removed(VRC_A, t0, |_| panic!("unknown client")), None);
    }

    #[test]
    fn a_confirmed_removal_is_reported_at_once() {
        let mut tr = VrchatTracker::default();
        tr.resolved(VRC_A, at(4000));
        assert_eq!(tr.removed(VRC_A, Instant::now(), |_| false), Some(VrchatEvent::Lost));
        // VRChat comes back on another port: found again.
        assert_eq!(tr.resolved(VRC_A, at(4001)), Some(VrchatEvent::Found(at(4001))));
    }

    #[test]
    fn a_resolve_clears_a_suspected_removal_and_lost_waits_for_the_last_client() {
        let t0 = Instant::now();
        let mut tr = VrchatTracker::default();
        tr.resolved(VRC_A, at(4000));
        assert_eq!(tr.removed(VRC_A, t0, |_| true), None);
        // mDNS learns the record again: no longer suspected (no more probes).
        assert_eq!(tr.resolved(VRC_A, at(4000)), None);
        assert_eq!(tr.tick(t0 + SUSPECT_RECHECK * 2, |_| panic!("not suspected")), None);
        // Two clients (VRChat restarting): the old one going away is not "VRChat quit".
        tr.resolved(VRC_B, at(5000));
        assert_eq!(tr.removed(VRC_A, t0, |_| false), None);
        assert_eq!(tr.removed(VRC_B, t0, |_| false), Some(VrchatEvent::Lost));
    }

    #[test]
    fn query_value_reads_the_first_value_of_a_node() {
        // A tiny fake VRChat OSCQuery server answering one request.
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).unwrap();
        let addr = listener.local_addr().unwrap();
        let server = thread::spawn(move || {
            let (mut s, _) = listener.accept().unwrap();
            let head = read_head(&mut s).unwrap();
            assert!(head.starts_with("GET /avatar/parameters/MuteSelf HTTP/1.0"), "{head}");
            let body = r#"{"FULL_PATH":"/avatar/parameters/MuteSelf","ACCESS":3,"TYPE":"T","VALUE":[true]}"#;
            write!(s, "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}", body.len()).unwrap();
        });
        let v = query_value(addr, MUTE_SELF, Duration::from_secs(2)).unwrap();
        server.join().unwrap();
        assert_eq!(v.as_ref().and_then(value_as_bool), Some(true));
    }

    #[test]
    fn chunked_bodies_and_bool_values() {
        let raw = b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\n{\"VAL\r\n8\r\nUE\":[0]}\r\n0\r\n\r\n";
        let (status, b) = parse_response(raw).unwrap();
        assert_eq!(status, 200);
        assert_eq!(b, br#"{"VALUE":[0]}"#);
        assert!(parse_response(b"garbage").is_err());
        assert_eq!(value_as_bool(&json!(false)), Some(false));
        assert_eq!(value_as_bool(&json!(1)), Some(true));
        assert_eq!(value_as_bool(&json!(0.0)), Some(false));
        assert_eq!(value_as_bool(&json!("true")), None);
    }

    #[test]
    fn vrchat_services_on_this_pc_only() {
        assert!(is_vrchat_instance("VRChat-Client-A1B2C3._oscjson._tcp.local."));
        assert!(!is_vrchat_instance("Sigil-A1B2C3._oscjson._tcp.local."));
        let lan: IpAddr = "192.168.1.20".parse().unwrap();
        let me: IpAddr = "192.168.1.10".parse().unwrap();
        let lo: IpAddr = "127.0.0.1".parse().unwrap();
        assert_eq!(pick_local_address(&[lan, lo], 9000, &[me]), Some(SocketAddr::new(lo, 9000)));
        assert_eq!(pick_local_address(&[me], 9000, &[me]), Some(SocketAddr::new(me, 9000)));
        assert_eq!(pick_local_address(&[lan], 9000, &[me]), None);
        let mapped: IpAddr = "::ffff:127.0.0.1".parse().unwrap();
        assert_eq!(pick_local_address(&[mapped], 1, &[]), Some(SocketAddr::new(lo, 1)));
    }

    #[test]
    fn instance_names_are_short_and_mdns_safe() {
        let n = instance_name();
        assert!(n.starts_with("Sigil-") && n.len() == 12, "{n}");
        assert!(n[6..].chars().all(|c| c.is_ascii_hexdigit()), "{n}");
    }

    #[test]
    fn request_heads_are_bounded() {
        let mut huge = io::Cursor::new(vec![b'a'; MAX_REQUEST_HEAD * 2]);
        assert!(read_head(&mut huge).is_err());
        let mut ok = io::Cursor::new(b"GET / HTTP/1.1\r\n\r\nrest".to_vec());
        assert!(read_head(&mut ok).unwrap().starts_with("GET / HTTP/1.1"));
    }
}
