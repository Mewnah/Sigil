//! Origin allow-list for the local web server's WebSocket routes (`/pubsub`, `/peer`).
//!
//! LAN security stage 1 (P0-I). The server binds `0.0.0.0:<port>` with no authentication, so any web
//! page open in the user's browser could otherwise open `ws://localhost:<port>/pubsub` and inject or
//! read captions (cross-site WebSocket hijacking). Browsers always send an `Origin` header on
//! WebSocket handshakes and pages cannot forge it, so an allow-list of Sigil's own origins closes that
//! hole without breaking any Sigil client:
//!
//! | Client | Origin | Allowed from |
//! |---|---|---|
//! | Host webview, Link apps from another Sigil | `https://tauri.localhost`, `http://tauri.localhost`, `tauri://localhost` | anywhere |
//! | Pages this server serves (`/client`, OBS, `mic.html`, `obs-captions.html`, `inputfield.html`) | `http://<this machine>:<port>` | anywhere |
//! | Vite dev server (debug builds) | `http://<this machine>:1420` | anywhere |
//! | Non-browser local tools (no `Origin`) | none | this machine only |
//!
//! `<this machine>` is `localhost`, `*.localhost`, a loopback or unspecified IP, one of this
//! machine's interface addresses, or its host name (`NAME` / `NAME.local`). Everything else is
//! refused with `403 Forbidden` before the upgrade: other origins, `Origin: null` (sandboxed frames,
//! `file://` pages), malformed values, and origin-less clients on other machines (unless they
//! present the LAN token, stage 2). Non-browser LAN clients can forge any header, so the LAN token
//! of stage 2 ([`super::auth`]) is what authenticates them.
//!
//! Kill switch: the persisted root setting `lanOriginCheck` (read from the settings file at startup,
//! then pushed live by the host through `plugin:web|set_origin_check`), overridden by the
//! [`ORIGIN_CHECK_ENV`] environment variable (`0`/`false`/`off` disables, `1`/`true`/`on` enables).
//! The switch is a watch channel ([`OriginGuard::subscribe`]): turning the check back on closes the
//! connections that got in only because it was off ([`super::auth::Access::revoked`]). Reporting of
//! refusals and that revocation live in [`super::auth::AccessGuard`], which combines this check
//! with the LAN token.

use std::fmt;
use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};
use std::path::Path;
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant};

use serde::Serialize;
use tokio::sync::watch;
use warp::http::header::{HeaderMap, HeaderName, ORIGIN};
use warp::Filter;

/// Environment variable that overrides the persisted `lanOriginCheck` setting.
pub const ORIGIN_CHECK_ENV: &str = "SIGIL_LAN_ORIGIN_CHECK";

/// Port of the Vite dev server (`build.devUrl` in tauri.conf.json); allowed in debug builds only.
pub const DEV_SERVER_PORT: u16 = 1420;

/// Origins of Sigil's own webviews: Windows (`useHttpsScheme` on/off, Android) and macOS/Linux.
const TAURI_ORIGINS: [(&str, &str); 3] = [("https", "tauri.localhost"), ("http", "tauri.localhost"), ("tauri", "localhost")];

/// Longest `Origin` value we try to parse (real ones are well under 100 bytes).
const MAX_ORIGIN_LEN: usize = 512;

/// How long an interface-address snapshot is reused before re-enumerating (DHCP, VPN changes).
const LOCAL_IP_CACHE_TTL: Duration = Duration::from_secs(5);

/// Longest origin text copied into logs and events.
const MAX_REPORTED_ORIGIN_CHARS: usize = 200;

/// Request headers a reverse proxy or tunnel adds. A connection that carries one of them came
/// through a proxy, so its loopback peer address says nothing about where the client is: such
/// connections never get the same-PC exemption from the LAN token ([`ConnInfo::forwarded`]).
const FORWARDING_HEADERS: [&str; 8] = [
    "forwarded",
    "x-forwarded-for",
    "x-forwarded-host",
    "x-forwarded-proto",
    "x-real-ip",
    "x-client-ip",
    "cf-connecting-ip",
    "true-client-ip",
];

/// Host part of a parsed origin.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OriginHost {
    Ip(IpAddr),
    /// Lower-case DNS name without a trailing dot.
    Name(String),
}

/// A serialized origin (`scheme://host[:port]`, RFC 6454), lower-cased.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParsedOrigin {
    pub scheme: String,
    pub host: OriginHost,
    /// Explicit port, if the origin had one.
    pub port: Option<u16>,
}

impl ParsedOrigin {
    /// Explicit port, else the scheme's default (`http`/`ws` 80, `https`/`wss` 443).
    pub fn effective_port(&self) -> Option<u16> {
        self.port.or(match self.scheme.as_str() {
            "http" | "ws" => Some(80),
            "https" | "wss" => Some(443),
            _ => None,
        })
    }
}

/// Parse an `Origin` header value. Returns `None` for `null`, opaque or malformed values and for
/// anything carrying a path, query, fragment or user info (which a real origin never has).
pub fn parse_origin(raw: &str) -> Option<ParsedOrigin> {
    let raw = raw.trim();
    if raw.is_empty() || raw.len() > MAX_ORIGIN_LEN || !raw.is_ascii() {
        return None;
    }
    let lower = raw.to_ascii_lowercase();
    let (scheme, rest) = lower.split_once("://")?;
    let scheme_ok = scheme.starts_with(|c: char| c.is_ascii_alphabetic())
        && scheme
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'));
    if !scheme_ok || rest.is_empty() {
        return None;
    }
    if rest.contains(|c: char| matches!(c, '/' | '?' | '#' | '@' | '\\') || c.is_ascii_whitespace() || c.is_ascii_control()) {
        return None;
    }

    let (host, port) = if let Some(bracketed) = rest.strip_prefix('[') {
        let (ip, tail) = bracketed.split_once(']')?;
        let ip: Ipv6Addr = ip.parse().ok()?;
        let port = match tail {
            "" => None,
            tail => Some(parse_port(tail.strip_prefix(':')?)?),
        };
        (OriginHost::Ip(IpAddr::V6(ip)), port)
    } else {
        let (host, port) = match rest.split_once(':') {
            Some((host, port)) => (host, Some(parse_port(port)?)),
            None => (rest, None),
        };
        let host = host.strip_suffix('.').unwrap_or(host);
        if host.is_empty() {
            return None;
        }
        if let Ok(v4) = host.parse::<Ipv4Addr>() {
            (OriginHost::Ip(IpAddr::V4(v4)), port)
        } else {
            let name_ok = host.split('.').all(|label| {
                !label.is_empty()
                    && label
                        .chars()
                        .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            });
            if !name_ok {
                return None;
            }
            (OriginHost::Name(host.to_string()), port)
        }
    };

    Some(ParsedOrigin {
        scheme: scheme.to_string(),
        host,
        port,
    })
}

fn parse_port(s: &str) -> Option<u16> {
    if s.is_empty() || s.len() > 5 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Maps IPv4-mapped IPv6 addresses (`::ffff:a.b.c.d`) to IPv4 so loopback checks see through them.
pub fn canonical_ip(ip: IpAddr) -> IpAddr {
    match ip {
        IpAddr::V6(v6) => v6
            .to_ipv4_mapped()
            .map(IpAddr::V4)
            .unwrap_or(IpAddr::V6(v6)),
        v4 => v4,
    }
}

/// What the check needs to know about this machine. Injected so tests don't depend on the host.
pub trait LocalMachine: Send + Sync {
    /// Whether `ip` is assigned to one of this machine's network interfaces.
    fn is_own_ip(&self, ip: IpAddr) -> bool;
    /// Whether `name` (lower-case, no trailing dot) is this machine's host name.
    fn is_own_hostname(&self, name: &str) -> bool;
    /// Whether a TCP peer at `ip` is a process on this machine: loopback, or one of this machine's
    /// own interface addresses (a page opened at `http://<LAN IP>:<port>` on this PC). Tests
    /// override it to make loopback sockets count as LAN peers.
    fn is_this_machine(&self, ip: IpAddr) -> bool {
        let ip = canonical_ip(ip);
        ip.is_loopback() || self.is_own_ip(ip)
    }
}

/// [`LocalMachine`] backed by the OS: interface addresses (cached for a few seconds) and the host
/// name from the environment.
#[derive(Default)]
pub struct SystemMachine {
    ips: Mutex<Option<(Instant, Vec<IpAddr>)>>,
    hostnames: OnceLock<Vec<String>>,
}

impl SystemMachine {
    fn interface_ips(&self) -> Vec<IpAddr> {
        let mut cache = self.ips.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some((at, ips)) = cache.as_ref() {
            if at.elapsed() < LOCAL_IP_CACHE_TTL {
                return ips.clone();
            }
        }
        let ips: Vec<IpAddr> = local_ip_address::list_afinet_netifas()
            .map(|list| list.into_iter().map(|(_, ip)| canonical_ip(ip)).collect())
            .unwrap_or_default();
        *cache = Some((Instant::now(), ips.clone()));
        ips
    }

    fn hostnames(&self) -> &[String] {
        self.hostnames.get_or_init(|| {
            let name = ["COMPUTERNAME", "HOSTNAME"]
                .iter()
                .find_map(|var| std::env::var(var).ok())
                .or_else(|| std::fs::read_to_string("/etc/hostname").ok())
                .map(|n| n.trim().trim_end_matches('.').to_ascii_lowercase())
                .filter(|n| !n.is_empty() && n != "localhost");
            match name {
                Some(n) => vec![format!("{n}.local"), n],
                None => Vec::new(),
            }
        })
    }
}

impl LocalMachine for SystemMachine {
    fn is_own_ip(&self, ip: IpAddr) -> bool {
        let ip = canonical_ip(ip);
        self.interface_ips().contains(&ip)
    }

    fn is_own_hostname(&self, name: &str) -> bool {
        self.hostnames().iter().any(|n| n == name)
    }
}

/// Ports the allow-list matches against.
#[derive(Clone, Copy, Debug)]
pub struct OriginRules {
    /// Port of this web server (`--port`, default 3030).
    pub port: u16,
    /// Vite dev server port, allowed only when `Some` (debug builds).
    pub dev_port: Option<u16>,
}

impl OriginRules {
    /// Rules for this build: the dev server origin only in debug builds.
    pub fn for_port(port: u16) -> Self {
        Self {
            port,
            dev_port: cfg!(debug_assertions).then_some(DEV_SERVER_PORT),
        }
    }
}

/// Why a handshake was accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Allowed {
    /// The kill switch is off.
    CheckDisabled,
    /// Sigil's own webview (host app, Link apps from another Sigil).
    TauriApp,
    /// A page served by this web server.
    ServedPage,
    /// The Vite dev server (debug builds).
    DevServer,
    /// A non-browser client on this machine.
    LocalWithoutOrigin,
}

/// Why a handshake was refused.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Rejected {
    /// A well-formed origin that is not one of Sigil's.
    ForeignOrigin,
    /// `Origin: null` (sandboxed iframe, `file://` page, some redirects).
    NullOrigin,
    /// Unparseable, non-ASCII or repeated `Origin` header.
    MalformedOrigin,
    /// No `Origin` header from another machine (or an unknown peer address).
    MissingOriginRemote,
}

impl fmt::Display for Rejected {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Rejected::ForeignOrigin => "origin is not a Sigil page",
            Rejected::NullOrigin => "opaque (null) origin",
            Rejected::MalformedOrigin => "malformed Origin header",
            Rejected::MissingOriginRemote => "no Origin header from another machine",
        })
    }
}

/// The `Origin` header of a request as the check sees it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OriginHeader {
    Absent,
    Present(String),
    /// Non-ASCII bytes or more than one `Origin` header.
    Invalid,
}

impl OriginHeader {
    pub fn from_headers(headers: &HeaderMap) -> Self {
        let mut values = headers.get_all(ORIGIN).iter();
        let Some(first) = values.next() else {
            return OriginHeader::Absent;
        };
        if values.next().is_some() {
            return OriginHeader::Invalid;
        }
        match first.to_str() {
            Ok(s) => OriginHeader::Present(s.to_string()),
            Err(_) => OriginHeader::Invalid,
        }
    }

    /// The value as logged and reported (truncated; `None` when absent).
    pub fn for_report(&self) -> Option<String> {
        match self {
            OriginHeader::Absent => None,
            OriginHeader::Invalid => Some("<invalid>".to_string()),
            OriginHeader::Present(s) => Some(s.chars().take(MAX_REPORTED_ORIGIN_CHARS).collect()),
        }
    }
}

/// Connection facts the WebSocket routes pass to the access check.
#[derive(Clone, Debug)]
pub struct ConnInfo {
    pub origin: OriginHeader,
    pub remote: Option<SocketAddr>,
    /// The request carries a reverse-proxy header ([`FORWARDING_HEADERS`]): the peer address is the
    /// proxy's, not the client's.
    pub forwarded: bool,
}

impl ConnInfo {
    /// Facts for a direct connection (no proxy headers).
    #[cfg(test)]
    pub fn direct(origin: OriginHeader, remote: Option<SocketAddr>) -> Self {
        Self {
            origin,
            remote,
            forwarded: false,
        }
    }

    pub fn from_parts(headers: &HeaderMap, remote: Option<SocketAddr>) -> Self {
        Self {
            origin: OriginHeader::from_headers(headers),
            remote,
            forwarded: has_forwarding_headers(headers),
        }
    }
}

/// Whether `headers` contain a header a reverse proxy or tunnel adds.
pub fn has_forwarding_headers(headers: &HeaderMap) -> bool {
    FORWARDING_HEADERS
        .iter()
        .any(|name| headers.contains_key(HeaderName::from_static(name)))
}

/// Warp filter extracting [`ConnInfo`] (never rejects).
pub fn conn_info() -> impl Filter<Extract = (ConnInfo,), Error = std::convert::Infallible> + Clone {
    warp::header::headers_cloned()
        .and(warp::addr::remote())
        .map(|headers: HeaderMap, remote: Option<SocketAddr>| ConnInfo::from_parts(&headers, remote))
}

fn host_is_this_machine(host: &OriginHost, machine: &dyn LocalMachine) -> bool {
    match host {
        OriginHost::Ip(ip) => {
            let ip = canonical_ip(*ip);
            ip.is_loopback() || ip.is_unspecified() || machine.is_own_ip(ip)
        }
        OriginHost::Name(name) => name == "localhost" || name.ends_with(".localhost") || machine.is_own_hostname(name),
    }
}

fn peer_is_this_machine(remote: Option<IpAddr>, machine: &dyn LocalMachine) -> bool {
    remote.is_some_and(|ip| machine.is_this_machine(ip))
}

/// The allow-list decision for one handshake (kill switch not applied). Pure apart from the
/// [`LocalMachine`] lookups, which only happen when loopback rules don't already decide.
pub fn evaluate(origin: &OriginHeader, remote: Option<IpAddr>, rules: &OriginRules, machine: &dyn LocalMachine) -> Result<Allowed, Rejected> {
    let raw = match origin {
        OriginHeader::Absent => {
            return if peer_is_this_machine(remote, machine) {
                Ok(Allowed::LocalWithoutOrigin)
            } else {
                Err(Rejected::MissingOriginRemote)
            };
        }
        OriginHeader::Invalid => return Err(Rejected::MalformedOrigin),
        OriginHeader::Present(raw) => raw.trim(),
    };
    if raw.eq_ignore_ascii_case("null") {
        return Err(Rejected::NullOrigin);
    }
    let Some(origin) = parse_origin(raw) else {
        return Err(Rejected::MalformedOrigin);
    };

    if origin.port.is_none() {
        if let OriginHost::Name(name) = &origin.host {
            if TAURI_ORIGINS
                .iter()
                .any(|(scheme, host)| origin.scheme == *scheme && name == host)
            {
                return Ok(Allowed::TauriApp);
            }
        }
    }

    if origin.scheme != "http" {
        return Err(Rejected::ForeignOrigin);
    }
    let port = origin.effective_port();
    let served = port == Some(rules.port);
    let dev = rules.dev_port.is_some() && port == rules.dev_port;
    if (served || dev) && host_is_this_machine(&origin.host, machine) {
        return Ok(if served { Allowed::ServedPage } else { Allowed::DevServer });
    }
    Err(Rejected::ForeignOrigin)
}

/// Parse a boolean-ish override value (`1/0`, `true/false`, `on/off`, `yes/no`).
pub fn parse_override(value: &str) -> Option<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" | "enabled" => Some(true),
        "0" | "false" | "off" | "no" | "disabled" => Some(false),
        _ => None,
    }
}

/// `lanOriginCheck` from the host's persisted settings JSON (`<app_data>/user/settings`).
/// `None` when the bytes are not a JSON object; otherwise the value, where anything but an explicit
/// `false` means enabled (the schema default, like `zSafe` in src/core/schema.ts).
pub fn origin_check_from_settings(bytes: &[u8]) -> Option<bool> {
    let value: serde_json::Value = serde_json::from_slice(bytes).ok()?;
    let object = value.as_object()?;
    Some(!matches!(object.get("lanOriginCheck"), Some(serde_json::Value::Bool(false))))
}

/// Parse the host's persisted settings with `parse`, reading `<app_data>/user/settings` and falling
/// back to `settings.bak` (atomic writer backup) when the main file is missing or `parse` refuses
/// it (corrupt or half-written). `None` if neither is usable.
pub fn read_persisted_settings<T>(app_data_dir: &Path, parse: impl Fn(&[u8]) -> Option<T>) -> Option<T> {
    let dir = app_data_dir.join("user");
    ["settings", "settings.bak"]
        .iter()
        .find_map(|name| std::fs::read(dir.join(name)).ok().and_then(|bytes| parse(&bytes)))
}

/// Read `lanOriginCheck` from the persisted settings ([`read_persisted_settings`]).
pub fn persisted_origin_check(app_data_dir: &Path) -> Option<bool> {
    read_persisted_settings(app_data_dir, origin_check_from_settings)
}

/// Allow-list state for the WebSocket routes: rules, machine facts and the kill switch. Refusals
/// are reported, and connections revoked, by [`super::auth::AccessGuard`].
pub struct OriginGuard {
    rules: OriginRules,
    /// The `lanOriginCheck` setting; a watch channel so connections admitted while it was off can
    /// close themselves when it is turned back on.
    setting: watch::Sender<bool>,
    env_override: Option<bool>,
    machine: Box<dyn LocalMachine>,
}

impl OriginGuard {
    pub fn new(rules: OriginRules, enabled: bool, env_override: Option<bool>, machine: Box<dyn LocalMachine>) -> Self {
        Self {
            rules,
            setting: watch::Sender::new(enabled),
            env_override,
            machine,
        }
    }

    /// Guard for the running app: OS-backed machine info, env override from [`ORIGIN_CHECK_ENV`].
    pub fn for_app(port: u16, persisted: Option<bool>) -> Self {
        let env_override = std::env::var(ORIGIN_CHECK_ENV)
            .ok()
            .and_then(|v| parse_override(&v));
        Self::new(
            OriginRules::for_port(port),
            persisted.unwrap_or(true),
            env_override,
            Box::new(SystemMachine::default()),
        )
    }

    /// The persisted / UI setting (before the environment override).
    pub fn setting(&self) -> bool {
        *self.setting.borrow()
    }

    /// Update the persisted / UI setting. Returns whether the check is now in effect. Subscribers
    /// ([`subscribe`](Self::subscribe)) are notified, so turning it on closes the connections that
    /// were admitted only because it was off.
    pub fn set_setting(&self, enabled: bool) -> bool {
        self.setting.send_replace(enabled);
        self.is_enforced()
    }

    /// Notified on every [`set_setting`](Self::set_setting).
    pub fn subscribe(&self) -> watch::Receiver<bool> {
        self.setting.subscribe()
    }

    /// Environment override, if [`ORIGIN_CHECK_ENV`] is set to a recognised value.
    pub fn env_override(&self) -> Option<bool> {
        self.env_override
    }

    /// Whether the allow-list is currently enforced.
    pub fn is_enforced(&self) -> bool {
        self.env_override.unwrap_or_else(|| self.setting())
    }

    /// Facts about this machine (interface addresses, host name, what counts as a local peer).
    pub fn machine(&self) -> &dyn LocalMachine {
        self.machine.as_ref()
    }

    /// The allow-list decision for one handshake with the kill switch applied: `CheckDisabled`
    /// while the check is off. Nothing is logged or reported here.
    pub fn verdict(&self, info: &ConnInfo) -> Result<Allowed, Rejected> {
        if !self.is_enforced() {
            return Ok(Allowed::CheckDisabled);
        }
        evaluate(&info.origin, info.remote.map(|a| a.ip()), &self.rules, self.machine.as_ref())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PORT: u16 = 3030;
    const LAN_IP: [u8; 4] = [192, 168, 1, 50];

    /// This machine is 192.168.1.50 / fe80::50, named `gaming-pc`.
    struct FakeMachine;

    impl LocalMachine for FakeMachine {
        fn is_own_ip(&self, ip: IpAddr) -> bool {
            ip == IpAddr::from(LAN_IP) || ip == "fe80::50".parse::<IpAddr>().unwrap()
        }
        fn is_own_hostname(&self, name: &str) -> bool {
            name == "gaming-pc" || name == "gaming-pc.local"
        }
    }

    fn rules(dev: bool) -> OriginRules {
        OriginRules {
            port: PORT,
            dev_port: dev.then_some(DEV_SERVER_PORT),
        }
    }

    fn eval(origin: Option<&str>, remote: [u8; 4]) -> Result<Allowed, Rejected> {
        let header = origin.map_or(OriginHeader::Absent, |o| OriginHeader::Present(o.to_string()));
        evaluate(&header, Some(IpAddr::from(remote)), &rules(true), &FakeMachine)
    }

    const LOOPBACK: [u8; 4] = [127, 0, 0, 1];
    const OTHER_PC: [u8; 4] = [192, 168, 1, 77];

    #[test]
    fn parses_origins() {
        let o = parse_origin("HTTP://LocalHost:3030").unwrap();
        assert_eq!(o.scheme, "http");
        assert_eq!(o.host, OriginHost::Name("localhost".into()));
        assert_eq!(o.port, Some(3030));

        let o = parse_origin("http://[::1]:3030").unwrap();
        assert_eq!(o.host, OriginHost::Ip("::1".parse().unwrap()));
        assert_eq!(o.effective_port(), Some(3030));

        let o = parse_origin("https://tauri.localhost").unwrap();
        assert_eq!(o.port, None);
        assert_eq!(o.effective_port(), Some(443));
        assert_eq!(parse_origin("tauri://localhost").unwrap().effective_port(), None);
        assert_eq!(parse_origin("http://localhost.:3030").unwrap().host, OriginHost::Name("localhost".into()));

        for bad in [
            "",
            "null",
            "localhost:3030",
            "http://",
            "http://localhost:",
            "http://localhost:+3030",
            "http://localhost:99999",
            "http://localhost:3030/",
            "http://localhost:3030/path",
            "http://user@localhost:3030",
            "http://evil.com?x=http://localhost:3030",
            "http://[::1",
            "http://[nope]:3030",
            "http://a..b:3030",
            "http://ex ample.com",
            "http://exämple.com",
            "1http://localhost",
        ] {
            assert_eq!(parse_origin(bad), None, "{bad:?} should not parse");
        }
    }

    #[test]
    fn allows_tauri_webviews_from_anywhere() {
        for origin in [
            "https://tauri.localhost",
            "http://tauri.localhost",
            "tauri://localhost",
            "HTTPS://TAURI.LOCALHOST",
        ] {
            assert_eq!(eval(Some(origin), LOOPBACK), Ok(Allowed::TauriApp), "{origin}");
            // Link apps: another Sigil on the LAN connects from its own webview.
            assert_eq!(eval(Some(origin), OTHER_PC), Ok(Allowed::TauriApp), "{origin}");
        }
        assert_eq!(eval(Some("https://tauri.localhost:8443"), LOOPBACK), Err(Rejected::ForeignOrigin));
        assert_eq!(eval(Some("tauri://evil"), LOOPBACK), Err(Rejected::ForeignOrigin));
        assert_eq!(eval(Some("https://tauri.localhost.evil.com"), LOOPBACK), Err(Rejected::ForeignOrigin));
    }

    #[test]
    fn allows_pages_served_by_this_server() {
        for origin in [
            "http://localhost:3030",
            "http://127.0.0.1:3030",
            "http://127.0.0.2:3030",
            "http://[::1]:3030",
            "http://0.0.0.0:3030",
            "http://sigil.localhost:3030",
            "http://192.168.1.50:3030",
            "http://[fe80::50]:3030",
            "http://gaming-pc:3030",
            "http://GAMING-PC.local:3030",
        ] {
            // OBS / mic.html on this PC, and /client in a browser on another PC.
            assert_eq!(eval(Some(origin), LOOPBACK), Ok(Allowed::ServedPage), "{origin}");
            assert_eq!(eval(Some(origin), OTHER_PC), Ok(Allowed::ServedPage), "{origin}");
        }
    }

    #[test]
    fn rejects_foreign_origins() {
        for origin in [
            "https://evil.example",
            "http://evil.example:3030",
            // DNS rebinding keeps the attacker's host name in the Origin.
            "http://rebind.attacker.test:3030",
            "http://gaming-pc.attacker.test:3030",
            "http://localhost:3031",
            "http://localhost",
            "https://localhost:3030",
            "ws://localhost:3030",
            "http://192.168.1.77:3030",
            "http://203.0.113.9:3030",
            "chrome-extension://abcdefghijklmnop",
            "file://",
        ] {
            let verdict = eval(Some(origin), LOOPBACK);
            assert!(
                matches!(verdict, Err(Rejected::ForeignOrigin | Rejected::MalformedOrigin)),
                "{origin} -> {verdict:?}"
            );
        }
        assert_eq!(eval(Some("null"), LOOPBACK), Err(Rejected::NullOrigin));
        assert_eq!(eval(Some("NULL"), LOOPBACK), Err(Rejected::NullOrigin));
        assert_eq!(eval(Some("not an origin"), LOOPBACK), Err(Rejected::MalformedOrigin));
        assert_eq!(
            evaluate(&OriginHeader::Invalid, Some(IpAddr::from(LOOPBACK)), &rules(true), &FakeMachine),
            Err(Rejected::MalformedOrigin)
        );
    }

    #[test]
    fn dev_server_origin_only_when_enabled() {
        assert_eq!(eval(Some("http://localhost:1420"), LOOPBACK), Ok(Allowed::DevServer));
        assert_eq!(eval(Some("http://192.168.1.50:1420"), OTHER_PC), Ok(Allowed::DevServer));
        assert_eq!(eval(Some("http://evil.example:1420"), LOOPBACK), Err(Rejected::ForeignOrigin));
        let release = evaluate(
            &OriginHeader::Present("http://localhost:1420".into()),
            Some(IpAddr::from(LOOPBACK)),
            &rules(false),
            &FakeMachine,
        );
        assert_eq!(release, Err(Rejected::ForeignOrigin));
    }

    #[test]
    fn missing_origin_only_from_this_machine() {
        assert_eq!(eval(None, LOOPBACK), Ok(Allowed::LocalWithoutOrigin));
        assert_eq!(eval(None, [127, 5, 5, 5]), Ok(Allowed::LocalWithoutOrigin));
        assert_eq!(eval(None, LAN_IP), Ok(Allowed::LocalWithoutOrigin));
        let mapped: IpAddr = "::ffff:127.0.0.1".parse().unwrap();
        assert_eq!(
            evaluate(&OriginHeader::Absent, Some(mapped), &rules(true), &FakeMachine),
            Ok(Allowed::LocalWithoutOrigin)
        );
        assert_eq!(eval(None, OTHER_PC), Err(Rejected::MissingOriginRemote));
        assert_eq!(
            evaluate(&OriginHeader::Absent, None, &rules(true), &FakeMachine),
            Err(Rejected::MissingOriginRemote)
        );
    }

    #[test]
    fn origin_header_extraction() {
        let mut headers = HeaderMap::new();
        assert_eq!(OriginHeader::from_headers(&headers), OriginHeader::Absent);
        headers.insert(ORIGIN, "http://localhost:3030".parse().unwrap());
        assert_eq!(
            OriginHeader::from_headers(&headers),
            OriginHeader::Present("http://localhost:3030".into())
        );
        headers.append(ORIGIN, "https://evil.example".parse().unwrap());
        assert_eq!(OriginHeader::from_headers(&headers), OriginHeader::Invalid);
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, warp::http::HeaderValue::from_bytes(b"http://\xffx").unwrap());
        assert_eq!(OriginHeader::from_headers(&headers), OriginHeader::Invalid);
    }

    fn guard(enabled: bool, env: Option<bool>) -> OriginGuard {
        OriginGuard::new(rules(false), enabled, env, Box::new(FakeMachine))
    }

    fn conn(origin: &str, remote: [u8; 4]) -> ConnInfo {
        ConnInfo::direct(OriginHeader::Present(origin.to_string()), Some(SocketAddr::from((remote, 50000))))
    }

    #[test]
    fn kill_switch_and_env_override() {
        let g = guard(true, None);
        assert!(g.is_enforced());
        assert_eq!(g.verdict(&conn("https://evil.example", LOOPBACK)), Err(Rejected::ForeignOrigin));
        assert!(!g.set_setting(false));
        assert_eq!(g.verdict(&conn("https://evil.example", LOOPBACK)), Ok(Allowed::CheckDisabled));
        assert!(g.set_setting(true));
        assert_eq!(g.verdict(&conn("https://evil.example", LOOPBACK)), Err(Rejected::ForeignOrigin));
        assert_eq!(g.verdict(&conn("http://localhost:3030", LOOPBACK)), Ok(Allowed::ServedPage));

        let forced_off = guard(true, Some(false));
        assert!(!forced_off.is_enforced());
        assert!(forced_off.setting());
        assert_eq!(forced_off.verdict(&conn("https://evil.example", LOOPBACK)), Ok(Allowed::CheckDisabled));

        let forced_on = guard(false, Some(true));
        assert!(forced_on.is_enforced());
        assert!(forced_on.set_setting(false));
        assert_eq!(forced_on.verdict(&conn("https://evil.example", LOOPBACK)), Err(Rejected::ForeignOrigin));
    }

    #[tokio::test]
    async fn setting_changes_notify_subscribers() {
        let g = guard(false, None);
        let mut rx = g.subscribe();
        assert!(!*rx.borrow_and_update());
        g.set_setting(true);
        tokio::time::timeout(Duration::from_secs(5), rx.changed())
            .await
            .expect("notified")
            .unwrap();
        assert!(*rx.borrow_and_update());
    }

    #[test]
    fn this_machine_means_loopback_or_an_own_address() {
        for ip in ["127.0.0.1", "127.9.9.9", "::1", "::ffff:127.0.0.1", "192.168.1.50", "fe80::50", "::ffff:192.168.1.50"] {
            assert!(FakeMachine.is_this_machine(ip.parse().unwrap()), "{ip}");
        }
        for ip in ["192.168.1.77", "10.0.0.1", "0.0.0.0", "203.0.113.9", "fe80::77"] {
            assert!(!FakeMachine.is_this_machine(ip.parse().unwrap()), "{ip}");
        }
    }

    #[test]
    fn proxy_headers_are_detected() {
        let remote = Some(SocketAddr::from((LOOPBACK, 1)));
        let mut headers = HeaderMap::new();
        headers.insert(ORIGIN, "http://localhost:3030".parse().unwrap());
        assert!(!ConnInfo::from_parts(&headers, remote).forwarded);
        for name in FORWARDING_HEADERS {
            let mut proxied = headers.clone();
            proxied.insert(HeaderName::from_static(name), "203.0.113.9".parse().unwrap());
            let info = ConnInfo::from_parts(&proxied, remote);
            assert!(info.forwarded, "{name}");
            assert_eq!(info.origin, OriginHeader::Present("http://localhost:3030".into()));
        }
        // Header names are case-insensitive on the wire.
        let mut upper = HeaderMap::new();
        upper.insert(HeaderName::from_bytes(b"X-Forwarded-For").unwrap(), "1.2.3.4".parse().unwrap());
        assert!(has_forwarding_headers(&upper));
    }

    #[test]
    fn reads_the_persisted_setting() {
        assert_eq!(origin_check_from_settings(br#"{"lanOriginCheck":false,"id":"x"}"#), Some(false));
        assert_eq!(origin_check_from_settings(br#"{"lanOriginCheck":true}"#), Some(true));
        assert_eq!(origin_check_from_settings(br#"{"id":"x"}"#), Some(true));
        assert_eq!(origin_check_from_settings(br#"{"lanOriginCheck":"false"}"#), Some(true));
        assert_eq!(origin_check_from_settings(b"[false]"), None);
        assert_eq!(origin_check_from_settings(b"{corrupt"), None);

        let dir = std::env::temp_dir().join(format!("sigil-web-origin-test-{}", std::process::id()));
        let user = dir.join("user");
        std::fs::create_dir_all(&user).unwrap();
        assert_eq!(persisted_origin_check(&dir), None);
        std::fs::write(user.join("settings.bak"), br#"{"lanOriginCheck":false}"#).unwrap();
        assert_eq!(persisted_origin_check(&dir), Some(false), "falls back to the backup");
        std::fs::write(user.join("settings"), b"{half-written").unwrap();
        assert_eq!(persisted_origin_check(&dir), Some(false), "corrupt main file uses the backup");
        std::fs::write(user.join("settings"), br#"{"uiLanguage":"en"}"#).unwrap();
        assert_eq!(persisted_origin_check(&dir), Some(true));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn override_values() {
        for v in ["0", "false", "OFF", " no ", "disabled"] {
            assert_eq!(parse_override(v), Some(false), "{v}");
        }
        for v in ["1", "true", "On", "yes", "enabled"] {
            assert_eq!(parse_override(v), Some(true), "{v}");
        }
        assert_eq!(parse_override(""), None);
        assert_eq!(parse_override("maybe"), None);
    }
}
