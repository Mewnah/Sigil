//! `web` plugin: the local HTTP/WebSocket server (PeerJS signaling at `/peer`, pubsub at `/pubsub`,
//! `/ping`, and the app's static assets for `/client`, OBS and the helper pages), plus the
//! host-side commands around it.
//!
//! LAN security stage 1 (P0-I): WebSocket handshakes pass an Origin allow-list ([`origin`]) with a
//! kill switch (`lanOriginCheck` setting / `SIGIL_LAN_ORIGIN_CHECK`; re-enabling it closes foreign
//! connections admitted while it was off), interactive helper pages cannot be framed by other
//! sites ([`assets`]), binary pubsub frames are refused, and a bind failure is reported
//! (`web:server_status`, `plugin:web|server_status`) instead of panicking the server task and
//! leaving the event pump spinning.
//!
//! LAN security stage 2 (P6-D, [`auth`]): the root setting `lanAccess {mode, token,
//! legacyNoToken}` (read from the settings file at startup, then pushed live by the host through
//! `plugin:web|set_lan_access`). Clients on other machines need the persistent LAN token; this PC
//! never does. The mode chooses the listen address (`lan` = `0.0.0.0`, `localhost` = `127.0.0.1`)
//! and a change rebinds live ([`serve_supervised`]); if the new address cannot be bound, the
//! server keeps listening on the previous one and reports why. The supervisor never gives up while
//! the app runs: a bind that keeps failing (on the requested address, and on the previous mode's
//! when a switch failed), a server that ends or panics by itself, or a supervisor that panics, is
//! retried with a capped backoff (`DOWN_RETRY_DELAYS_MS`) until it listens again. The first
//! `Listening` also ends the start's turn of the pre-check (`port_precheck::release_start_turn`).
//!
//! Commands: `config`, `pubsub_broadcast`, `open_browser`, `server_status`, `set_origin_check`,
//! `set_lan_access`, `regenerate_lan_token`.
//! Events: `pubsub` (peer message text → host), `web:server_status` ([`WebServerStatus`]),
//! `web:origin_rejected` ([`auth::AccessRejection`], throttled per origin / peer address; despite
//! the name it also reports LAN token refusals).

use futures::future::BoxFuture;
use local_ip_address::local_ip;
use serde::{Deserialize, Serialize};
use std::{
    future::Future,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::Path,
    process::Command,
    sync::{Arc, Mutex, PoisonError},
    time::{Duration, Instant},
};
use tauri::{
    async_runtime::Mutex as AsyncMutex,
    command,
    plugin::{Builder, TauriPlugin},
    Emitter, Manager, Runtime, State,
};
use tokio::sync::{mpsc, oneshot, watch};
use warp::Filter;

use super::AppConfiguration;

mod assets;
mod auth;
mod origin;
mod peer;
mod pubsub;
#[cfg(test)]
mod test_support;

use auth::{AccessGuard, BindMode, LanAccessSettings, LanPolicy};
use origin::OriginGuard;

/// Event carrying a peer message (text) to the host webview.
pub const PUBSUB_EVENT: &str = "pubsub";
/// Event emitted when the server starts listening, fails to bind or stops ([`WebServerStatus`]).
pub const SERVER_STATUS_EVENT: &str = "web:server_status";
/// Event emitted for refused or revoked WebSocket connections ([`auth::AccessRejection`], throttled).
pub const ORIGIN_REJECTED_EVENT: &str = "web:origin_rejected";

/// Delays between bind attempts: a previous Sigil instance may still be releasing the port.
const BIND_RETRY_DELAYS_MS: [u64; 5] = [250, 500, 1_000, 2_000, 4_000];

/// Delays between the quiet bind attempts while nothing listens (every bind attempt of a round
/// failed, or the server ended by itself). The last delay repeats for as long as Sigil runs, so a
/// port that frees up later, or a listener that died, comes back without anyone touching Settings.
const DOWN_RETRY_DELAYS_MS: [u64; 5] = [1_000, 2_000, 5_000, 10_000, 15_000];

/// A server that ran this long before it ended starts over at the first (shortest) delay; one that
/// keeps ending sooner backs off instead of restarting in a tight loop.
const STABLE_UPTIME: Duration = Duration::from_secs(60);

/// How long in-flight HTTP requests may finish after a bind-mode switch closed the old listener.
/// Upgraded WebSockets are separate tasks: the access rules close the ones the new mode refuses.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(3);

struct PubSubInput {
    tx: AsyncMutex<mpsc::Sender<String>>,
}

#[command]
async fn pubsub_broadcast(value: String, input: State<'_, PubSubInput>) -> Result<(), String> {
    let tx = input.tx.lock().await;
    tx.send(value).await.map_err(|e| e.to_string())
}

/// An IPv4 address of this PC that other devices may reach it on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
struct LanAddress {
    ip: String,
    /// Network adapter name (for picking the right one when there are several).
    interface: String,
}

#[derive(Serialize)]
struct WebConfig {
    pub local_ip: String,
    pub port: String,
    pub peer_path: String,
    pub pubsub_path: String,
    /// Candidate LAN addresses for links to other devices, `local_ip` first.
    pub lan_addresses: Vec<LanAddress>,
}

/// Usable LAN addresses from the interface list: IPv4, no loopback / link-local / unspecified /
/// multicast / broadcast, de-duplicated. Order: `primary`, then private ranges, then the rest
/// (VPN / CGNAT ranges), each in interface order.
fn order_lan_addresses(interfaces: Vec<(String, IpAddr)>, primary: Option<IpAddr>) -> Vec<LanAddress> {
    let usable = |ip: &Ipv4Addr| !(ip.is_loopback() || ip.is_link_local() || ip.is_unspecified() || ip.is_multicast() || ip.is_broadcast());
    let mut seen = Vec::<Ipv4Addr>::new();
    let mut ranked: Vec<(u8, usize, LanAddress)> = Vec::new();
    for (index, (interface, ip)) in interfaces.into_iter().enumerate() {
        let IpAddr::V4(v4) = origin::canonical_ip(ip) else {
            continue;
        };
        if !usable(&v4) || seen.contains(&v4) {
            continue;
        }
        seen.push(v4);
        let rank = if primary.map(origin::canonical_ip) == Some(IpAddr::V4(v4)) {
            0
        } else if v4.is_private() {
            1
        } else {
            2
        };
        ranked.push((rank, index, LanAddress { ip: v4.to_string(), interface }));
    }
    ranked.sort_by_key(|(rank, index, _)| (*rank, *index));
    ranked.into_iter().map(|(_, _, address)| address).collect()
}

/// The address the app advertises for this PC. `default_route_ip` is `local_ip()`, which on Windows needs a
/// default route and is `None` on a PC that is offline (no network at all, airplane mode, a cable that is not
/// plugged in, Windows Sandbox with networking off) or on a LAN without a gateway. That must not stop the app
/// from starting (it used to: the front end showed "Application failed to start: Error retrieving local IP"), so:
/// the default-route address, else the first usable LAN address, else loopback.
fn primary_ip(default_route_ip: Option<IpAddr>, interfaces: &[(String, IpAddr)]) -> IpAddr {
    default_route_ip
        .or_else(|| {
            order_lan_addresses(interfaces.to_vec(), None)
                .first()
                .and_then(|a| a.ip.parse::<Ipv4Addr>().ok())
                .map(IpAddr::V4)
        })
        .unwrap_or(IpAddr::V4(Ipv4Addr::LOCALHOST))
}

#[command]
async fn config(config: State<'_, AppConfiguration>) -> Result<WebConfig, String> {
    let interfaces = local_ip_address::list_afinet_netifas().unwrap_or_default();
    let default_route_ip = local_ip().ok();
    if default_route_ip.is_none() {
        tracing::info!(target: "sigil::web", "no default route (offline?): advertising the first LAN address, else loopback");
    }
    let ip = primary_ip(default_route_ip, &interfaces);
    Ok(WebConfig {
        local_ip: ip.to_string(),
        port: config.port.to_string(),
        peer_path: "peer".to_string(),
        pubsub_path: "pubsub".to_string(),
        lan_addresses: order_lan_addresses(interfaces, Some(ip)),
    })
}

/// The address the server will listen on at startup (`lanAccess.mode` from the persisted settings,
/// see [`auth::startup_lan_access`]: `127.0.0.1` for `localhost` and for a new install with no
/// settings file yet, `0.0.0.0` for `lan`, which an existing profile without the key keeps). For
/// main.rs's port pre-check before Tauri starts, so it checks the address the server really binds
/// (P6-D hotspot request).
pub fn startup_bind_ip() -> IpAddr {
    bind_ip_for(super::diagnostics::app_data_dir().as_deref())
}

/// [`startup_bind_ip`] for a given app data folder. The plugin's init gets its settings from the same
/// [`auth::startup_lan_access`] (the persisted-settings read behind it is private to `auth`), so the port
/// pre-check and the bind cannot probe one address and take another.
fn bind_ip_for(app_data: Option<&Path>) -> IpAddr {
    auth::startup_lan_access(app_data).mode.ip()
}

/// Lifecycle of the local web server.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerState {
    /// Binding (including retries and bind-mode switches).
    Starting,
    Listening,
    /// Could not bind after every start-up retry; `error` says why. The supervisor keeps trying in
    /// the background and reports `Listening` when it works.
    Failed,
    /// The server future ended or panicked, or the supervisor itself did (should not happen while
    /// the app runs); it is bound again after a short wait, so `Starting` and `Listening` follow.
    Stopped,
}

/// LAN access part of [`WebServerStatus`]. Never contains the token.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LanStatus {
    /// The requested mode (`lanAccess.mode`).
    pub mode: BindMode,
    /// The mode actually being served; differs from `mode` while a switch failed (then `error`
    /// says why) and is `None` while not listening.
    pub bound_mode: Option<BindMode>,
    /// The listening socket address, e.g. `0.0.0.0:3030`.
    pub bound_address: Option<String>,
    /// Whether a valid LAN token is in force (the host generates one when not).
    pub token_set: bool,
    pub legacy_no_token: bool,
}

/// Snapshot returned by `plugin:web|server_status` and emitted as `web:server_status`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WebServerStatus {
    pub state: ServerState,
    pub port: u16,
    /// Bind / server error text when `state` is `failed` or `stopped`, or why a bind-mode switch
    /// did not happen while still `listening` on the previous mode.
    pub error: Option<String>,
    /// Whether the Origin allow-list is currently enforced.
    pub origin_check: bool,
    /// The `lanOriginCheck` setting as last pushed (or read from the settings file).
    pub origin_check_setting: bool,
    /// `SIGIL_LAN_ORIGIN_CHECK` override, if set; it wins over the setting.
    pub origin_check_env: Option<bool>,
    pub lan: LanStatus,
}

/// What the server supervisor reports.
#[derive(Clone, Debug, PartialEq, Eq)]
enum ServerEvent {
    Starting { mode: BindMode },
    /// `error` is set when this is the previous mode, kept because the requested one failed.
    Listening { mode: BindMode, addr: SocketAddr, error: Option<String> },
    Failed { error: String },
    Stopped { error: String },
}

#[derive(Clone, Debug)]
struct Lifecycle {
    state: ServerState,
    error: Option<String>,
    bound: Option<(BindMode, SocketAddr)>,
}

/// Plugin state shared by the server task and the commands.
struct WebServer {
    port: u16,
    access: Arc<AccessGuard>,
    lifecycle: Mutex<Lifecycle>,
    /// The requested bind mode; the supervisor rebinds when it changes.
    modes: watch::Sender<BindMode>,
}

/// Whether a request for `wanted` must wake the supervisor: the mode changed, or it is the same
/// request while that mode is not being served (a failed bind or a fallback), which retries.
fn should_rebind(requested: BindMode, wanted: BindMode, bound: Option<BindMode>) -> bool {
    requested != wanted || bound != Some(wanted)
}

impl WebServer {
    fn lifecycle(&self) -> Lifecycle {
        self.lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone()
    }

    fn status(&self) -> WebServerStatus {
        let Lifecycle { state, error, bound } = self.lifecycle();
        let origin = self.access.origin();
        let policy = self.access.lan_policy();
        WebServerStatus {
            state,
            port: self.port,
            error,
            origin_check: origin.is_enforced(),
            origin_check_setting: origin.setting(),
            origin_check_env: origin.env_override(),
            lan: LanStatus {
                mode: *self.modes.borrow(),
                bound_mode: bound.map(|(mode, _)| mode),
                bound_address: bound.map(|(_, addr)| addr.to_string()),
                token_set: policy.token_set(),
                legacy_no_token: policy.legacy_no_token,
            },
        }
    }

    fn apply(&self, event: &ServerEvent) {
        let next = match event {
            ServerEvent::Starting { .. } => Lifecycle {
                state: ServerState::Starting,
                error: None,
                bound: None,
            },
            ServerEvent::Listening { mode, addr, error } => Lifecycle {
                state: ServerState::Listening,
                error: error.clone(),
                bound: Some((*mode, *addr)),
            },
            ServerEvent::Failed { error } => Lifecycle {
                state: ServerState::Failed,
                error: Some(error.clone()),
                bound: None,
            },
            ServerEvent::Stopped { error } => Lifecycle {
                state: ServerState::Stopped,
                error: Some(error.clone()),
                bound: None,
            },
        };
        *self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = next;
    }

    /// Apply the host's `lanAccess` setting: the access rules first (connections they now refuse
    /// close themselves), then the listen address.
    fn apply_lan_access(&self, settings: &LanAccessSettings) {
        let policy = LanPolicy::from_settings(settings);
        if !settings.token.is_empty() && !policy.token_set() {
            tracing::warn!(
                target: "sigil::web",
                "ignoring a malformed LAN token: other machines are refused until a new one is generated"
            );
        }
        if self.access.set_lan_policy(policy.clone()) {
            tracing::info!(
                target: "sigil::web",
                "LAN access: {} ({}), devices without the token {}",
                policy.mode.as_str(),
                if policy.token_set() { "token set" } else { "no token yet" },
                if policy.legacy_no_token { "allowed (legacy)" } else { "refused" },
            );
        }
        let bound = self.lifecycle().bound.map(|(mode, _)| mode);
        self.modes.send_if_modified(|requested| {
            let rebind = should_rebind(*requested, policy.mode, bound);
            *requested = policy.mode;
            rebind
        });
    }
}

#[command]
fn server_status(server: State<'_, Arc<WebServer>>) -> WebServerStatus {
    server.status()
}

/// Kill switch for the Origin allow-list (the host pushes the `lanOriginCheck` setting here).
/// Turning it on also closes `/pubsub` and `/peer` connections that got in only while it was off
/// and that the allow-list refuses (close code 1008).
#[command]
fn set_origin_check(enabled: bool, server: State<'_, Arc<WebServer>>) -> WebServerStatus {
    let origin = server.access.origin();
    if origin.setting() != enabled {
        tracing::info!(target: "sigil::web", "Origin check setting changed to {enabled}");
    }
    let enforced = origin.set_setting(enabled);
    if enforced != enabled {
        tracing::warn!(
            target: "sigil::web",
            "{} overrides the lanOriginCheck setting: the Origin check stays {}",
            origin::ORIGIN_CHECK_ENV,
            if enforced { "on" } else { "off" }
        );
    }
    server.status()
}

/// The host's `lanAccess` setting (LAN security stage 2). Returns the status after applying it; a
/// bind-mode switch finishes later and is reported through `web:server_status`.
#[command]
fn set_lan_access(settings: LanAccessSettings, server: State<'_, Arc<WebServer>>) -> WebServerStatus {
    server.apply_lan_access(&settings);
    server.status()
}

/// Reply of `plugin:web|regenerate_lan_token`: the new token (for the host to persist and put in
/// LAN links) and the status.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct LanTokenReply {
    token: String,
    status: WebServerStatus,
}

/// Generate a new LAN token and put it in force at once: clients on other machines still using the
/// old one are disconnected (4401). The host persists the returned token.
#[command]
fn regenerate_lan_token(server: State<'_, Arc<WebServer>>) -> Result<LanTokenReply, String> {
    let token = auth::generate_token()?;
    server.access.set_token(token.clone());
    tracing::info!(target: "sigil::web", "generated a new LAN token; devices on other machines need the new LAN links");
    Ok(LanTokenReply {
        token,
        status: server.status(),
    })
}

#[derive(Serialize, Deserialize)]
struct OpenBrowserCommand {
    browser: String,
    url: String,
}

fn spawn_browser_windows_exe(exe: &str, url: &str) -> Result<(), std::io::Error> {
    Command::new(exe).arg(url).spawn().map(|_| ())
}

/// Opens `url` in a named browser when possible, otherwise the OS default handler.
#[command]
fn open_browser(data: OpenBrowserCommand) -> Result<(), String> {
    if !data.url.starts_with("http://") && !data.url.starts_with("https://") {
        return Err("Only http and https URLs are allowed".to_string());
    }

    let browser = data.browser.as_str();

    if cfg!(target_os = "windows") {
        if browser == "chrome" {
            for exe in [
                r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
            ] {
                if spawn_browser_windows_exe(exe, &data.url).is_ok() {
                    return Ok(());
                }
            }
        } else if browser == "edge" {
            for exe in [
                r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
            ] {
                if spawn_browser_windows_exe(exe, &data.url).is_ok() {
                    return Ok(());
                }
            }
        }

        return Command::new("cmd")
            .args(["/C", "start", "", &data.url])
            .spawn()
            .map_err(|e| format!("Failed to open browser: {}", e))
            .map(|_| ());
    }

    if cfg!(target_os = "macos") {
        let app = match browser {
            "chrome" => Some("Google Chrome"),
            "edge" => Some("Microsoft Edge"),
            _ => None,
        };
        if let Some(app) = app {
            if Command::new("open")
                .args(["-a", app, &data.url])
                .spawn()
                .is_ok()
            {
                return Ok(());
            }
        }

        return Command::new("open")
            .arg(&data.url)
            .spawn()
            .map_err(|e| format!("Failed to open browser: {}", e))
            .map(|_| ());
    }

    // Linux, *BSD, etc.
    if browser == "chrome" {
        for bin in ["google-chrome", "google-chrome-stable", "chromium", "chromium-browser"] {
            if Command::new(bin).arg(&data.url).spawn().is_ok() {
                return Ok(());
            }
        }
    } else if browser == "edge" {
        for bin in ["microsoft-edge", "microsoft-edge-stable"] {
            if Command::new(bin).arg(&data.url).spawn().is_ok() {
                return Ok(());
            }
        }
    }

    Command::new("xdg-open")
        .arg(&data.url)
        .spawn()
        .map_err(|e| format!("Failed to open browser (xdg-open): {}", e))?;
    Ok(())
}

/// A bound server: its address, the serving future and the graceful-shutdown trigger.
struct Running {
    addr: SocketAddr,
    server: BoxFuture<'static, ()>,
    stop: oneshot::Sender<()>,
}

enum BindError {
    /// The requested mode changed while retrying.
    ModeChanged,
    Failed(String),
}

fn try_bind<F>(routes: &F, addr: SocketAddr) -> Result<Running, String>
where
    F: Filter + Clone + Send + Sync + 'static,
    F::Extract: warp::Reply,
{
    let (stop, stop_rx) = oneshot::channel::<()>();
    let signal = async move {
        let _ = stop_rx.await;
    };
    match warp::serve(routes.clone()).try_bind_with_graceful_shutdown(addr, signal) {
        Ok((bound, server)) => Ok(Running {
            addr: bound,
            server: Box::pin(server),
            stop,
        }),
        Err(e) => Err(bind_error_text(addr, &e)),
    }
}

/// `could not listen on <addr>: <error>`, plus a reason when Windows refused the bind outright
/// (access denied, WSAEACCES): a port inside a range Windows reserved (Hyper-V, WSL, Docker) never
/// frees up by itself, unlike a port another program holds for a while.
fn bind_error_text(addr: SocketAddr, error: &(dyn std::error::Error + 'static)) -> String {
    let mut text = format!("could not listen on {addr}: {error}");
    if is_access_denied(error) {
        text.push_str(". Windows may have reserved this port");
    }
    text
}

/// Whether `error`, or anything it wraps, is an I/O error with kind `PermissionDenied`.
fn is_access_denied(error: &(dyn std::error::Error + 'static)) -> bool {
    let mut current = Some(error);
    while let Some(e) = current {
        if e.downcast_ref::<std::io::Error>().is_some_and(|io| io.kind() == std::io::ErrorKind::PermissionDenied) {
            return true;
        }
        current = e.source();
    }
    false
}

/// The serving future on a task of its own. A panic in it (the accept loop, a request filter) ends
/// that task only and comes back as a `JoinError`, instead of unwinding through the supervisor; and
/// the task is aborted when this is dropped, so a listener never outlives the supervisor's round.
struct ServerTask(tokio::task::JoinHandle<()>);

impl ServerTask {
    fn spawn(server: BoxFuture<'static, ()>) -> Self {
        Self(tokio::spawn(server))
    }
}

impl Drop for ServerTask {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// Why a serving task ended, for the `Stopped` report.
fn server_end_text(ended: Result<(), tokio::task::JoinError>) -> String {
    match ended {
        Ok(()) => "the local web server stopped unexpectedly".to_string(),
        Err(e) if e.is_panic() => "the local web server crashed".to_string(),
        Err(e) => format!("the local web server task was cancelled ({e})"),
    }
}

/// Completes when the requested mode is (re)notified; never when the sender is gone.
async fn next_mode_change(modes: &mut watch::Receiver<BindMode>) {
    if modes.changed().await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// Timing of [`serve_supervised`] (a parameter so tests run in milliseconds).
#[derive(Clone, Copy)]
struct Retry {
    /// Delays between the bind attempts of one round (`Starting`): a previous Sigil may still hold the port.
    bind_ms: &'static [u64],
    /// Delays between the quiet bind attempts while nothing listens; the last one repeats.
    down_ms: &'static [u64],
    /// How long a server must have run for the next restart to start over at the first delay.
    stable_uptime: Duration,
}

impl Retry {
    const APP: Retry = Retry {
        bind_ms: &BIND_RETRY_DELAYS_MS,
        down_ms: &DOWN_RETRY_DELAYS_MS,
        stable_uptime: STABLE_UPTIME,
    };
}

/// Capped backoff over a delay list: each call returns the next delay and the last one repeats.
struct Backoff {
    delays_ms: &'static [u64],
    step: usize,
}

impl Backoff {
    fn new(delays_ms: &'static [u64]) -> Self {
        Self { delays_ms, step: 0 }
    }

    /// Never zero, so a bad list cannot make the supervisor spin.
    fn next_delay(&mut self) -> Duration {
        let last = self.delays_ms.len().saturating_sub(1);
        let ms = self.delays_ms.get(self.step.min(last)).copied().unwrap_or(1_000).max(1);
        self.step = self.step.saturating_add(1);
        Duration::from_millis(ms)
    }

    fn reset(&mut self) {
        self.step = 0;
    }
}

/// Bind `addr` with `bind`, retrying after each of `retry_delays_ms` while the port is busy.
async fn bind_retrying<B>(bind: &B, addr: SocketAddr, retry_delays_ms: &[u64], modes: &mut watch::Receiver<BindMode>) -> Result<Running, BindError>
where
    B: Fn(SocketAddr) -> Result<Running, String>,
{
    let mut attempt = 0usize;
    loop {
        match bind(addr) {
            Ok(running) => return Ok(running),
            Err(error) => {
                let Some(delay) = retry_delays_ms.get(attempt) else {
                    return Err(BindError::Failed(error));
                };
                tracing::warn!(target: "sigil::web", "{error}; retrying in {delay} ms");
                attempt += 1;
                tokio::select! {
                    () = tokio::time::sleep(Duration::from_millis(*delay)) => {}
                    () = next_mode_change(modes) => return Err(BindError::ModeChanged),
                }
            }
        }
    }
}

/// A server bound while the requested mode was down.
struct Rebound {
    /// The mode it serves: the requested one, or the previous one when only that could be bound.
    mode: BindMode,
    running: Running,
    /// Why the requested mode is not served (`Some` only for the previous mode).
    error: Option<String>,
}

/// Bind again, quietly, while nothing listens (every bind attempt of a round failed): one attempt
/// after each `backoff` delay, the last delay repeating, until a bind works (`Some`) or the requested
/// mode is notified (`None`: the caller starts a full round at once).
///
/// Each attempt tries the requested mode (`wanted`) first. When it still fails and `previous` is
/// set (the mode that served before, which a failed switch could not fall back to), that one is
/// tried too, so the server comes back on whichever address works instead of staying down while the
/// requested one is blocked.
///
/// `Starting` is never reported from here, and `Failed` only when the requested mode's error text
/// differs from `reported`, so the front end (which toasts once per status and error) does not nag
/// at every attempt; the caller reports `Listening` for the `Some` it gets.
async fn rebind_while_down<B>(
    bind: &B,
    wanted: (BindMode, SocketAddr),
    previous: Option<(BindMode, SocketAddr)>,
    backoff: &mut Backoff,
    modes: &mut watch::Receiver<BindMode>,
    report: &impl Fn(ServerEvent),
    mut reported: String,
) -> Option<Rebound>
where
    B: Fn(SocketAddr) -> Result<Running, String>,
{
    let mut attempts = 0u32;
    loop {
        let delay = backoff.next_delay();
        tokio::select! {
            () = tokio::time::sleep(delay) => {}
            () = next_mode_change(modes) => return None,
        }
        attempts += 1;
        let error = match bind(wanted.1) {
            Ok(running) => {
                tracing::info!(target: "sigil::web", "bound {} after {attempts} quiet retries", wanted.1);
                return Some(Rebound { mode: wanted.0, running, error: None });
            }
            Err(error) => error,
        };
        if error == reported {
            tracing::debug!(target: "sigil::web", "{error}; still retrying (attempt {attempts})");
        } else {
            tracing::warn!(target: "sigil::web", "{error}; still retrying (attempt {attempts})");
            report(ServerEvent::Failed { error: error.clone() });
            reported = error.clone();
        }
        if let Some((mode, addr)) = previous {
            match bind(addr) {
                Ok(running) => {
                    tracing::warn!(
                        target: "sigil::web",
                        "{error}; listening on {} ({} mode) again after {attempts} quiet retries",
                        running.addr,
                        mode.as_str()
                    );
                    return Some(Rebound { mode, running, error: Some(error) });
                }
                Err(e) => tracing::debug!(target: "sigil::web", "{e} (the {} mode, attempt {attempts})", mode.as_str()),
            }
        }
    }
}

/// Serve on the address of the requested [`BindMode`] (`addr_for`), rebinding whenever `modes` is
/// notified. `bind` opens a listener on an address ([`try_bind`] on the routes in the app). Every
/// state change goes to `report`. Never panics or busy-loops, and never gives up while the app runs:
///
/// - A bind round is retried after each of `retry.bind_ms` (a previous Sigil may still hold the port).
/// - On a switch the old listener closes at once (graceful shutdown), in-flight requests get
///   [`DRAIN_TIMEOUT`], and the new address is bound.
/// - If the new address fails, the previous mode is bound again and reported as `Listening` with
///   the error, so OBS and the pages on this PC keep working; a later notification retries.
/// - With nothing to fall back to (or a fallback that fails too), `Failed` is reported and the bind
///   is retried quietly after each of `retry.down_ms` (the last repeats), the previous mode as well
///   as the requested one, until one works; a notification starts a full round at once.
/// - If the server ends by itself (or its task panics), `Stopped` is reported and, after the next
///   `retry.down_ms` delay (back to the first once a server ran for `retry.stable_uptime`, or at
///   once on a notification), a full round binds it again.
async fn serve_supervised<B>(bind: B, mut modes: watch::Receiver<BindMode>, addr_for: impl Fn(BindMode) -> SocketAddr, retry: Retry, report: impl Fn(ServerEvent))
where
    B: Fn(SocketAddr) -> Result<Running, String>,
{
    let mut last_good: Option<BindMode> = None;
    let mut backoff = Backoff::new(retry.down_ms);
    loop {
        let wanted = *modes.borrow_and_update();
        report(ServerEvent::Starting { mode: wanted });
        let running = match bind_retrying(&bind, addr_for(wanted), retry.bind_ms, &mut modes).await {
            Ok(running) => {
                tracing::info!(target: "sigil::web", "local web server listening on {} ({} mode)", running.addr, wanted.as_str());
                last_good = Some(wanted);
                report(ServerEvent::Listening {
                    mode: wanted,
                    addr: running.addr,
                    error: None,
                });
                running
            }
            Err(BindError::ModeChanged) => continue,
            Err(BindError::Failed(error)) => {
                let previous = last_good.filter(|previous| *previous != wanted);
                let fallback = previous.and_then(|previous| match bind(addr_for(previous)) {
                    Ok(running) => Some((previous, running)),
                    Err(e) => {
                        tracing::warn!(target: "sigil::web", "{e} (falling back to the {} mode)", previous.as_str());
                        None
                    }
                });
                match fallback {
                    Some((previous, running)) => {
                        tracing::warn!(
                            target: "sigil::web",
                            "{error}; still listening on {} ({} mode)",
                            running.addr,
                            previous.as_str()
                        );
                        report(ServerEvent::Listening {
                            mode: previous,
                            addr: running.addr,
                            error: Some(error),
                        });
                        running
                    }
                    None => {
                        tracing::error!(
                            target: "sigil::web",
                            "{error}. /client, OBS, mic.html and Link apps are unavailable until the port is free; Sigil keeps retrying (changing LAN access retries at once)."
                        );
                        report(ServerEvent::Failed { error: error.clone() });
                        let wanted_at = (wanted, addr_for(wanted));
                        let previous_at = previous.map(|previous| (previous, addr_for(previous)));
                        match rebind_while_down(&bind, wanted_at, previous_at, &mut backoff, &mut modes, &report, error).await {
                            Some(Rebound { mode, running, error }) => {
                                tracing::info!(target: "sigil::web", "local web server listening on {} ({} mode)", running.addr, mode.as_str());
                                last_good = Some(mode);
                                report(ServerEvent::Listening {
                                    mode,
                                    addr: running.addr,
                                    error,
                                });
                                running
                            }
                            None => {
                                backoff.reset();
                                continue;
                            }
                        }
                    }
                }
            }
        };

        let up_since = Instant::now();
        let Running { addr, server, stop } = running;
        let mut server = ServerTask::spawn(server);
        tokio::select! {
            ended = &mut server.0 => {
                let error = server_end_text(ended);
                tracing::error!(target: "sigil::web", "{error} ({addr})");
                report(ServerEvent::Stopped { error });
                if up_since.elapsed() >= retry.stable_uptime {
                    backoff.reset();
                }
                let delay = backoff.next_delay();
                tracing::warn!(target: "sigil::web", "binding {addr} again in {delay:?}");
                tokio::select! {
                    () = tokio::time::sleep(delay) => {}
                    () = next_mode_change(&mut modes) => backoff.reset(),
                }
            }
            () = next_mode_change(&mut modes) => {
                tracing::info!(target: "sigil::web", "closing {addr} to apply the LAN access mode");
                let _ = stop.send(());
                if tokio::time::timeout(DRAIN_TIMEOUT, &mut server.0).await.is_err() {
                    tracing::warn!(target: "sigil::web", "requests on {addr} did not finish within {DRAIN_TIMEOUT:?}; dropping them");
                }
                backoff.reset();
            }
        }
    }
}

/// Runs a whole [`serve_supervised`] round (`round` makes it) as a task of its own and, whenever
/// that task ends, reports `Stopped` and starts a new round after the next `retry.down_ms` delay
/// (back to the first once a round ran for `retry.stable_uptime`). A round never returns by itself,
/// so an end is a panic in the supervisor: without this the listener would be gone for the rest of
/// the session while the status still said `listening`.
async fn keep_supervised<Fut>(round: impl Fn() -> Fut, retry: Retry, report: impl Fn(ServerEvent))
where
    Fut: Future<Output = ()> + Send + 'static,
{
    let mut backoff = Backoff::new(retry.down_ms);
    loop {
        let started = Instant::now();
        let error = match tokio::spawn(round()).await {
            Ok(()) => "the local web server supervisor ended".to_string(),
            Err(e) if e.is_panic() => "the local web server supervisor crashed".to_string(),
            Err(e) => format!("the local web server supervisor was cancelled ({e})"),
        };
        if started.elapsed() >= retry.stable_uptime {
            backoff.reset();
        }
        let delay = backoff.next_delay();
        tracing::error!(target: "sigil::web", "{error}; starting it again in {delay:?}");
        report(ServerEvent::Stopped { error });
        tokio::time::sleep(delay).await;
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    let (pubsub_input_tx, pubsub_input_rx) = mpsc::channel::<String>(1); // to pubsub
    let (pubsub_output_tx, mut pubsub_output_rx) = mpsc::channel::<String>(1); // to js
    Builder::new("web")
        .invoke_handler(tauri::generate_handler![
            open_browser,
            pubsub_broadcast,
            config,
            server_status,
            set_origin_check,
            set_lan_access,
            regenerate_lan_token
        ])
        .setup(|app, _api| {
            app.manage(PubSubInput {
                tx: AsyncMutex::new(pubsub_input_tx),
            });

            let app_port = app.state::<AppConfiguration>().port;

            // The persisted settings are read here so the Origin switch, the LAN token and the bind
            // mode apply before the webview has loaded (OBS and /client reconnect as soon as the
            // server is up); the host then pushes live changes through `set_origin_check` and
            // `set_lan_access`.
            let app_data = match app.path().app_data_dir() {
                Ok(dir) => Some(dir),
                Err(e) => {
                    tracing::warn!(target: "sigil::web", "no app data folder ({e}); the web server uses its default access settings");
                    None
                }
            };
            let persisted_origin_check = app_data.as_deref().and_then(origin::persisted_origin_check);
            // The same read as `startup_bind_ip`, so the pre-check and the bind agree: a new install
            // (no settings file) starts "This PC only" and binds 127.0.0.1.
            let lan_settings = auth::startup_lan_access(app_data.as_deref());

            let origin_guard = OriginGuard::for_app(app_port, persisted_origin_check);
            if let Some(forced) = origin_guard.env_override() {
                tracing::warn!(
                    target: "sigil::web",
                    "{} is set: the Origin check is forced {}",
                    origin::ORIGIN_CHECK_ENV,
                    if forced { "on" } else { "off" }
                );
            } else if !origin_guard.is_enforced() {
                tracing::warn!(target: "sigil::web", "the Origin check is disabled by the lanOriginCheck setting");
            }
            let policy = LanPolicy::from_settings(&lan_settings);
            tracing::info!(
                target: "sigil::web",
                "LAN access: {} ({}), devices without the token {}",
                policy.mode.as_str(),
                if policy.token_set() { "token set" } else { "no token yet" },
                if policy.legacy_no_token { "allowed (legacy)" } else { "refused" },
            );
            let access = Arc::new(AccessGuard::new(origin_guard, policy));
            let rejection_handle = app.clone();
            access.set_notifier(move |rejection| {
                let _ = rejection_handle.emit(ORIGIN_REJECTED_EVENT, rejection.clone());
            });

            let (modes_tx, modes_rx) = watch::channel(lan_settings.mode);
            let server = Arc::new(WebServer {
                port: app_port,
                access: access.clone(),
                lifecycle: Mutex::new(Lifecycle {
                    state: ServerState::Starting,
                    error: None,
                    bound: None,
                }),
                modes: modes_tx,
            });
            app.manage(server.clone());

            let status_handle = app.clone();
            let a = Arc::new(app.asset_resolver());
            tauri::async_runtime::spawn(async move {
                let routes = warp::path!("ping")
                    .map(|| "pong".to_string())
                    .or(peer::path(access.clone()))
                    .or(pubsub::path(pubsub_input_rx, pubsub_output_tx, access))
                    .or(assets::path(a));

                let report: Arc<dyn Fn(ServerEvent) + Send + Sync> = Arc::new(move |event| {
                    if matches!(event, ServerEvent::Listening { .. }) {
                        // The port is held by the server now: the next start of Sigil (a double click
                        // that waited for its turn, see `port_precheck`) finds this copy and its window.
                        crate::port_precheck::release_start_turn();
                    }
                    server.apply(&event);
                    let _ = status_handle.emit(SERVER_STATUS_EVENT, server.status());
                });
                let supervisor_report = report.clone();

                // `lan`: all IPv4 interfaces, as upstream Curses (so `http://localhost:PORT/client`
                // and LAN clients work; a 127.0.0.1-only bind rejected some localhost paths there).
                // `localhost`: 127.0.0.1 only. The supervisor runs as a task that is started again
                // if it ever panics.
                keep_supervised(
                    move || {
                        let routes = routes.clone();
                        let report = report.clone();
                        serve_supervised(
                            move |addr| try_bind(&routes, addr),
                            modes_rx.clone(),
                            move |mode| mode.socket_addr(app_port),
                            Retry::APP,
                            move |event| report(event),
                        )
                    },
                    Retry::APP,
                    move |event| supervisor_report(event),
                )
                .await;
            });

            let handle = app.clone();
            tauri::async_runtime::spawn(async move {
                // Ends once every sender is gone (the server task ended) instead of spinning.
                while let Some(output) = pubsub_output_rx.recv().await {
                    let _ = handle.emit(PUBSUB_EVENT, output);
                }
            });

            Ok(())
        })
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// From an app data folder to the address the server binds: the step the port pre-check (main.rs) and the
    /// plugin's init share (`auth::startup_lan_access`). A new install binds 127.0.0.1 on its very first launch
    /// (a 0.0.0.0 listener raises the Windows Firewall prompt); an existing profile keeps what it had.
    #[test]
    fn a_new_install_binds_loopback_and_an_existing_profile_keeps_its_address() {
        let loopback = IpAddr::V4(Ipv4Addr::LOCALHOST);
        let any = IpAddr::V4(Ipv4Addr::UNSPECIFIED);
        let mut new_installs = 0;
        auth::startup_fixture::each("bind", |case, dir| {
            let want = if case.mode == BindMode::Localhost { loopback } else { any };
            assert_eq!(bind_ip_for(Some(dir)), want, "{}", case.name);
            // What init hands the supervisor for the first bind is the same address, on the app's port.
            let first_bind = auth::startup_lan_access(Some(dir)).mode.socket_addr(3030);
            assert_eq!(first_bind, SocketAddr::new(want, 3030), "{}", case.name);
            if case.new_install {
                assert_eq!(bind_ip_for(Some(dir)), loopback, "{}", case.name);
                new_installs += 1;
            }
        });
        assert!(new_installs >= 2, "the table has new installs");
        // A folder that was never created is a new install too; no folder at all keeps the default.
        assert_eq!(bind_ip_for(Some(&std::env::temp_dir().join("sigil-web-bind-never-created"))), loopback);
        assert_eq!(bind_ip_for(None), any);
    }

    type Seen = Arc<StdMutex<Vec<ServerEvent>>>;

    fn recorder() -> (Seen, impl Fn(ServerEvent) + Send + 'static) {
        let seen: Seen = Arc::default();
        let sink = seen.clone();
        (seen, move |event| sink.lock().unwrap().push(event))
    }

    /// Wait until an event matching `pred` was recorded; returns it.
    async fn wait_for(seen: &Seen, what: &str, pred: impl Fn(&ServerEvent) -> bool) -> ServerEvent {
        for _ in 0..500 {
            if let Some(event) = seen.lock().unwrap().iter().rev().find(|e| pred(e)) {
                return event.clone();
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("no {what} event: {:?}", seen.lock().unwrap());
    }

    /// Wait until the recorded events satisfy `cond`.
    async fn wait_events(seen: &Seen, what: &str, cond: impl Fn(&[ServerEvent]) -> bool) {
        for _ in 0..500 {
            if cond(&seen.lock().unwrap()) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!("no {what}: {:?}", seen.lock().unwrap());
    }

    fn routes() -> impl Filter<Extract = (String,), Error = warp::Rejection> + Clone + Send + Sync + 'static {
        warp::path!("ping").map(|| "pong".to_string())
    }

    /// The bind the app uses (real sockets on the test's loopback addresses).
    fn real_bind() -> impl Fn(SocketAddr) -> Result<Running, String> + Send + Sync + 'static {
        let routes = routes();
        move |addr| try_bind(&routes, addr)
    }

    /// Millisecond timing: `bind_ms` between the bind attempts of a round, 5 ms between the quiet
    /// attempts while down.
    fn retry(bind_ms: &'static [u64]) -> Retry {
        Retry {
            bind_ms,
            down_ms: &[5],
            stable_uptime: Duration::from_secs(60),
        }
    }

    /// How a fake server ends.
    enum Ending {
        /// The future returns, like a listener that died.
        End,
        /// The future panics, like a bug in the accept loop.
        Crash,
    }

    /// Servers without sockets: a bind works unless `fail` (every address) or `fail_at` (one
    /// address) set an error, and a server ends when `end_latest` or `crash_latest` says so (or when
    /// the supervisor stops it), like a listener that died. `panic_next_bind` makes the next bind
    /// panic, like a bug in the supervisor itself.
    #[derive(Default)]
    struct FakeServers {
        error: StdMutex<Option<String>>,
        errors_at: StdMutex<Vec<(SocketAddr, String)>>,
        ends: StdMutex<Vec<oneshot::Sender<Ending>>>,
        binds: std::sync::atomic::AtomicUsize,
        bind_times: StdMutex<Vec<Instant>>,
        panic_next_bind: std::sync::atomic::AtomicBool,
    }

    impl FakeServers {
        fn bind(&self, addr: SocketAddr) -> Result<Running, String> {
            if self.panic_next_bind.swap(false, std::sync::atomic::Ordering::SeqCst) {
                panic!("fake supervisor bug");
            }
            self.bind_times.lock().unwrap().push(Instant::now());
            let error = self
                .errors_at
                .lock()
                .unwrap()
                .iter()
                .find(|(at, _)| *at == addr)
                .map(|(_, error)| error.clone())
                .or_else(|| self.error.lock().unwrap().clone());
            let result = match error {
                Some(error) => Err(error),
                None => {
                    let (end, end_rx) = oneshot::channel::<Ending>();
                    let (stop, stop_rx) = oneshot::channel::<()>();
                    self.ends.lock().unwrap().push(end);
                    let server = async move {
                        tokio::select! {
                            ending = end_rx => {
                                if matches!(ending, Ok(Ending::Crash)) {
                                    panic!("fake server crash");
                                }
                            }
                            _ = stop_rx => {}
                        }
                    };
                    Ok(Running {
                        addr,
                        server: Box::pin(server),
                        stop,
                    })
                }
            };
            // Counted last, so a test that waited for the n-th bind can already end its server.
            self.binds.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            result
        }

        fn fail(&self, error: Option<&str>) {
            *self.error.lock().unwrap() = error.map(str::to_string);
        }

        /// Binds to `addr` fail with `error` (or work again with `None`), whatever `fail` says.
        fn fail_at(&self, addr: SocketAddr, error: Option<&str>) {
            let mut errors = self.errors_at.lock().unwrap();
            errors.retain(|(at, _)| *at != addr);
            if let Some(error) = error {
                errors.push((addr, error.to_string()));
            }
        }

        fn end_latest(&self) {
            let end = self.ends.lock().unwrap().pop().expect("a running server");
            let _ = end.send(Ending::End);
        }

        fn crash_latest(&self) {
            let end = self.ends.lock().unwrap().pop().expect("a running server");
            let _ = end.send(Ending::Crash);
        }

        fn binds(&self) -> usize {
            self.binds.load(std::sync::atomic::Ordering::SeqCst)
        }

        /// When the n-th (1-based) bind was tried.
        fn bind_time(&self, n: usize) -> Instant {
            self.bind_times.lock().unwrap()[n - 1]
        }

        async fn wait_binds(&self, n: usize) {
            for _ in 0..500 {
                if self.binds() >= n {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
            panic!("only {} binds, wanted {n}", self.binds());
        }
    }

    fn loopback(port: u16) -> SocketAddr {
        SocketAddr::from(([127, 0, 0, 1], port))
    }

    async fn is_listening(addr: SocketAddr) -> bool {
        tokio::net::TcpStream::connect(addr).await.is_ok()
    }

    async fn ping(addr: SocketAddr) -> String {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};
        let mut stream = tokio::net::TcpStream::connect(addr).await.unwrap();
        stream
            .write_all(b"GET /ping HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
            .await
            .unwrap();
        let mut body = String::new();
        stream.read_to_string(&mut body).await.unwrap();
        body
    }

    #[tokio::test]
    async fn bind_failure_reports_instead_of_panicking() {
        // Hold a port so every bind attempt fails.
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = holder.local_addr().unwrap();
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let task = tokio::spawn(async move { serve_supervised(real_bind(), modes_rx, move |_| addr, retry(&[1, 1]), report).await });

        let failed = wait_for(&seen, "failed", |e| matches!(e, ServerEvent::Failed { .. })).await;
        let ServerEvent::Failed { error } = failed else { unreachable!() };
        assert!(error.contains(&addr.to_string()), "{error}");
        assert_eq!(seen.lock().unwrap()[0], ServerEvent::Starting { mode: BindMode::Lan });
        assert!(!task.is_finished(), "it keeps retrying instead of ending");
        task.abort();
        drop(holder);
    }

    #[tokio::test]
    async fn a_failed_bind_retries_quietly_until_the_port_frees() {
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = holder.local_addr().unwrap();
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let task = tokio::spawn(async move { serve_supervised(real_bind(), modes_rx, move |_| addr, retry(&[1]), report).await });
        wait_for(&seen, "failed", |e| matches!(e, ServerEvent::Failed { .. })).await;

        // Many quiet attempts fail the same way: no new Starting (the front end would toast again) and
        // no repeated Failed.
        tokio::time::sleep(Duration::from_millis(120)).await;
        assert!(!task.is_finished());
        let failed_only = seen.lock().unwrap().clone();
        assert_eq!(failed_only.len(), 2, "{failed_only:?}");
        assert!(matches!(failed_only[0], ServerEvent::Starting { .. }));
        assert!(matches!(failed_only[1], ServerEvent::Failed { .. }));

        // The port frees up and nobody does anything: the supervisor binds it by itself.
        drop(holder);
        let listening = wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        let ServerEvent::Listening { mode, addr: bound, error } = listening else { unreachable!() };
        assert_eq!((mode, bound, error), (BindMode::Lan, addr, None), "a successful bind clears the error");
        assert!(ping(addr).await.ends_with("pong"));
        let events = seen.lock().unwrap().clone();
        assert!(!events[2..].iter().any(|e| matches!(e, ServerEvent::Starting { .. } | ServerEvent::Failed { .. })), "{events:?}");
        task.abort();
    }

    #[tokio::test]
    async fn a_new_error_while_down_is_reported_once_and_recovery_clears_it() {
        let fake = Arc::new(FakeServers::default());
        fake.fail(Some("could not listen on A: busy"));
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        let task = tokio::spawn(async move { serve_supervised(move |a| servers.bind(a), modes_rx, |_| loopback(3030), retry(&[]), report).await });
        wait_events(&seen, "first failure", |e| e.len() >= 2).await;
        fake.wait_binds(4).await; // several quiet attempts with the same error
        fake.fail(Some("could not listen on A: access denied"));
        wait_events(&seen, "second error", |e| e.len() >= 3).await;
        fake.wait_binds(fake.binds() + 3).await;
        fake.fail(None);
        wait_events(&seen, "listening", |e| e.iter().any(|x| matches!(x, ServerEvent::Listening { .. }))).await;

        let events = seen.lock().unwrap().clone();
        assert_eq!(
            events,
            [
                ServerEvent::Starting { mode: BindMode::Lan },
                ServerEvent::Failed { error: "could not listen on A: busy".into() },
                ServerEvent::Failed { error: "could not listen on A: access denied".into() },
                ServerEvent::Listening { mode: BindMode::Lan, addr: loopback(3030), error: None },
            ]
        );
        task.abort();
    }

    #[tokio::test]
    async fn changing_the_mode_while_down_binds_at_once() {
        // The quiet retry would only come in an hour: the notification has to do it.
        let fake = Arc::new(FakeServers::default());
        fake.fail(Some("busy"));
        let (seen, report) = recorder();
        let (modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        let plan = Retry {
            bind_ms: &[],
            down_ms: &[3_600_000],
            stable_uptime: Duration::from_secs(60),
        };
        let task = tokio::spawn(async move {
            serve_supervised(move |a| servers.bind(a), modes_rx, |mode| loopback(if mode == BindMode::Lan { 3030 } else { 3031 }), plan, report).await
        });
        wait_for(&seen, "failed", |e| matches!(e, ServerEvent::Failed { .. })).await;
        fake.fail(None);
        modes_tx.send_replace(BindMode::Localhost);
        let event = wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        assert_eq!(event, ServerEvent::Listening { mode: BindMode::Localhost, addr: loopback(3031), error: None });
        assert!(seen.lock().unwrap().contains(&ServerEvent::Starting { mode: BindMode::Localhost }));
        task.abort();
    }

    #[tokio::test]
    async fn a_server_that_ends_is_bound_again() {
        let fake = Arc::new(FakeServers::default());
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        let task = tokio::spawn(async move { serve_supervised(move |a| servers.bind(a), modes_rx, |_| loopback(3030), retry(&[]), report).await });
        wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;

        fake.end_latest();
        wait_events(&seen, "listening again", |e| e.iter().filter(|x| matches!(x, ServerEvent::Listening { .. })).count() == 2).await;
        assert!(!task.is_finished(), "the supervisor does not return when the server ends");
        assert_eq!(fake.binds(), 2);
        let events = seen.lock().unwrap().clone();
        let listening = ServerEvent::Listening { mode: BindMode::Lan, addr: loopback(3030), error: None };
        assert_eq!(events.len(), 5, "{events:?}");
        assert_eq!(events[0], ServerEvent::Starting { mode: BindMode::Lan });
        assert_eq!(events[1], listening);
        assert!(matches!(&events[2], ServerEvent::Stopped { error } if error.contains("stopped unexpectedly")));
        assert_eq!(events[3], ServerEvent::Starting { mode: BindMode::Lan });
        assert_eq!(events[4], listening);

        // It serves again: ending it once more is picked up the same way.
        fake.end_latest();
        fake.wait_binds(3).await;
        task.abort();
    }

    #[tokio::test]
    async fn a_server_that_keeps_ending_backs_off() {
        let fake = Arc::new(FakeServers::default());
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        let plan = Retry {
            bind_ms: &[],
            down_ms: &[20, 80],
            stable_uptime: Duration::from_secs(60),
        };
        let task = tokio::spawn(async move { serve_supervised(move |a| servers.bind(a), modes_rx, |_| loopback(3030), plan, report).await });
        wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;

        // Each server ends right after it started (well before `stable_uptime`): 20 ms, then 80 ms, then 80 ms again.
        for (n, least) in [(2, 20), (3, 80), (4, 80)] {
            let ended = Instant::now();
            fake.end_latest();
            fake.wait_binds(n).await;
            assert!(ended.elapsed() >= Duration::from_millis(least), "restart {n} came after {:?}", ended.elapsed());
            tokio::time::sleep(Duration::from_millis(5)).await; // let the new server register its end handle
        }
        task.abort();
    }

    #[tokio::test]
    async fn a_server_that_ran_stably_restarts_after_the_first_delay() {
        let fake = Arc::new(FakeServers::default());
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        // A stable server (any uptime counts) starts over at the 10 ms delay; the 1 hour one is never reached.
        let plan = Retry {
            bind_ms: &[],
            down_ms: &[10, 3_600_000],
            stable_uptime: Duration::ZERO,
        };
        let task = tokio::spawn(async move { serve_supervised(move |a| servers.bind(a), modes_rx, |_| loopback(3030), plan, report).await });
        wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        for n in 2..=4 {
            fake.end_latest();
            fake.wait_binds(n).await;
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        task.abort();
    }

    #[tokio::test]
    async fn a_mode_change_cuts_short_the_wait_after_a_server_ended() {
        let fake = Arc::new(FakeServers::default());
        let (seen, report) = recorder();
        let (modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        // The restart would only come in an hour: the notification has to do it.
        let plan = Retry {
            bind_ms: &[],
            down_ms: &[3_600_000],
            stable_uptime: Duration::from_secs(60),
        };
        let task = tokio::spawn(async move {
            serve_supervised(move |a| servers.bind(a), modes_rx, |mode| loopback(if mode == BindMode::Lan { 3030 } else { 3031 }), plan, report).await
        });
        wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        fake.end_latest();
        wait_for(&seen, "stopped", |e| matches!(e, ServerEvent::Stopped { .. })).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(fake.binds(), 1, "it is waiting out the delay");

        modes_tx.send_replace(BindMode::Localhost);
        let event = wait_for(&seen, "localhost listening", |e| matches!(e, ServerEvent::Listening { mode: BindMode::Localhost, .. })).await;
        assert_eq!(event, ServerEvent::Listening { mode: BindMode::Localhost, addr: loopback(3031), error: None });
        assert_eq!(fake.binds(), 2);
        task.abort();
    }

    #[tokio::test]
    async fn a_server_that_ends_and_cannot_be_bound_again_retries_quietly_with_the_backoff_that_is_left() {
        let fake = Arc::new(FakeServers::default());
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        // Two delays: the wait after Stopped uses the first (20 ms); the quiet retries that follow are the shared
        // backoff's next steps, so they start at the second (60 ms), not over at 20 ms.
        let plan = Retry {
            bind_ms: &[],
            down_ms: &[20, 60],
            stable_uptime: Duration::from_secs(60),
        };
        let task = tokio::spawn(async move { serve_supervised(move |a| servers.bind(a), modes_rx, |_| loopback(3030), plan, report).await });
        wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;

        fake.fail(Some("busy"));
        fake.end_latest();
        fake.wait_binds(3).await; // 2: the round after the wait fails; 3: the first quiet retry
        assert!(fake.bind_time(3) - fake.bind_time(2) >= Duration::from_millis(60), "the quiet retry came {:?} after the round", fake.bind_time(3) - fake.bind_time(2));

        fake.fail(None);
        wait_events(&seen, "listening again", |e| e.iter().filter(|x| matches!(x, ServerEvent::Listening { .. })).count() == 2).await;
        let events = seen.lock().unwrap().clone();
        assert_eq!(events.len(), 6, "{events:?}");
        assert_eq!(events[0], ServerEvent::Starting { mode: BindMode::Lan });
        assert!(matches!(&events[2], ServerEvent::Stopped { .. }), "{events:?}");
        assert_eq!(events[3], ServerEvent::Starting { mode: BindMode::Lan });
        assert_eq!(events[4], ServerEvent::Failed { error: "busy".into() }, "reported once, however many quiet retries failed");
        assert_eq!(events[5], ServerEvent::Listening { mode: BindMode::Lan, addr: loopback(3030), error: None });
        task.abort();
    }

    #[tokio::test]
    async fn a_fallback_listener_that_ends_is_replaced_and_its_error_clears_once_the_requested_mode_binds() {
        let fake = Arc::new(FakeServers::default());
        let (seen, report) = recorder();
        let (modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        let task = tokio::spawn(async move {
            serve_supervised(move |a| servers.bind(a), modes_rx, |mode| loopback(if mode == BindMode::Lan { 3030 } else { 3031 }), retry(&[]), report).await
        });
        wait_for(&seen, "lan listening", |e| matches!(e, ServerEvent::Listening { .. })).await;

        // The switch fails: the previous mode keeps serving, with the error.
        fake.fail_at(loopback(3031), Some("busy"));
        modes_tx.send_replace(BindMode::Localhost);
        wait_events(&seen, "fallback", |e| e.len() == 4).await;

        // That fallback listener ends: it is bound again (the requested mode is tried first and still fails).
        fake.end_latest();
        wait_events(&seen, "fallback again", |e| e.len() == 7).await;

        // The requested mode works by the time the fallback ends the next time: it takes over and the error is gone.
        fake.fail_at(loopback(3031), None);
        fake.end_latest();
        wait_events(&seen, "requested mode", |e| e.len() == 10).await;

        let lan = |error: Option<&str>| ServerEvent::Listening { mode: BindMode::Lan, addr: loopback(3030), error: error.map(str::to_string) };
        let events = seen.lock().unwrap().clone();
        assert_eq!(events[..4], [ServerEvent::Starting { mode: BindMode::Lan }, lan(None), ServerEvent::Starting { mode: BindMode::Localhost }, lan(Some("busy"))]);
        assert!(matches!(&events[4], ServerEvent::Stopped { .. }), "{events:?}");
        assert_eq!(events[5], ServerEvent::Starting { mode: BindMode::Localhost });
        assert_eq!(events[6], lan(Some("busy")), "still the previous mode, still with the reason");
        assert!(matches!(&events[7], ServerEvent::Stopped { .. }), "{events:?}");
        assert_eq!(events[8], ServerEvent::Starting { mode: BindMode::Localhost });
        assert_eq!(events[9], ServerEvent::Listening { mode: BindMode::Localhost, addr: loopback(3031), error: None });
        task.abort();
    }

    #[tokio::test]
    async fn while_down_the_previous_mode_is_retried_as_well() {
        let fake = Arc::new(FakeServers::default());
        let (seen, report) = recorder();
        let (modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        let task = tokio::spawn(async move {
            serve_supervised(move |a| servers.bind(a), modes_rx, |mode| loopback(if mode == BindMode::Lan { 3030 } else { 3031 }), retry(&[]), report).await
        });
        wait_for(&seen, "lan listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        seen.lock().unwrap().clear();

        // The switch fails and so does binding the previous mode again (an antivirus hook, a network change).
        fake.fail_at(loopback(3030), Some("hook"));
        fake.fail_at(loopback(3031), Some("busy"));
        modes_tx.send_replace(BindMode::Localhost);
        wait_for(&seen, "failed", |e| matches!(e, ServerEvent::Failed { .. })).await;

        // Only the previous mode's address works again: the quiet retry brings the server back on it.
        fake.fail_at(loopback(3030), None);
        let event = wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        assert_eq!(event, ServerEvent::Listening { mode: BindMode::Lan, addr: loopback(3030), error: Some("busy".into()) });
        let events = seen.lock().unwrap().clone();
        assert_eq!(events.len(), 3, "{events:?}");
        assert_eq!(events[0], ServerEvent::Starting { mode: BindMode::Localhost });
        assert_eq!(events[1], ServerEvent::Failed { error: "busy".into() });

        // The next notification (the host re-sends the setting) tries the requested mode again.
        fake.fail_at(loopback(3031), None);
        modes_tx.send_modify(|_| {});
        let event = wait_for(&seen, "localhost listening", |e| matches!(e, ServerEvent::Listening { mode: BindMode::Localhost, .. })).await;
        assert_eq!(event, ServerEvent::Listening { mode: BindMode::Localhost, addr: loopback(3031), error: None });
        task.abort();
    }

    #[tokio::test]
    async fn a_server_that_panics_is_reported_and_bound_again() {
        let fake = Arc::new(FakeServers::default());
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let servers = fake.clone();
        let task = tokio::spawn(async move { serve_supervised(move |a| servers.bind(a), modes_rx, |_| loopback(3030), retry(&[]), report).await });
        wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;

        fake.crash_latest();
        wait_events(&seen, "listening again", |e| e.iter().filter(|x| matches!(x, ServerEvent::Listening { .. })).count() == 2).await;
        assert!(!task.is_finished(), "a panic in the server does not take the supervisor with it");
        let events = seen.lock().unwrap().clone();
        assert!(matches!(&events[2], ServerEvent::Stopped { error } if error.contains("crashed")), "{events:?}");
        assert_eq!(events[3], ServerEvent::Starting { mode: BindMode::Lan });
        task.abort();
    }

    #[tokio::test]
    async fn a_supervisor_that_panics_is_started_again() {
        let fake = Arc::new(FakeServers::default());
        fake.panic_next_bind.store(true, std::sync::atomic::Ordering::SeqCst);
        let (seen, report) = recorder();
        let report = Arc::new(report);
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let (round_report, restart_report) = (report.clone(), report.clone());
        let servers = fake.clone();
        let task = tokio::spawn(async move {
            keep_supervised(
                move || {
                    let servers = servers.clone();
                    let report = round_report.clone();
                    serve_supervised(move |a| servers.bind(a), modes_rx.clone(), |_| loopback(3030), retry(&[]), move |e| report(e))
                },
                retry(&[]),
                move |e| restart_report(e),
            )
            .await
        });
        wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        let events = seen.lock().unwrap().clone();
        assert_eq!(events.len(), 4, "{events:?}");
        assert_eq!(events[0], ServerEvent::Starting { mode: BindMode::Lan });
        assert!(matches!(&events[1], ServerEvent::Stopped { error } if error.contains("supervisor crashed")), "{events:?}");
        assert_eq!(events[2], ServerEvent::Starting { mode: BindMode::Lan });
        assert_eq!(events[3], ServerEvent::Listening { mode: BindMode::Lan, addr: loopback(3030), error: None });
        assert!(!task.is_finished());
        task.abort();
    }

    #[test]
    fn an_access_denied_bind_says_why_and_other_failures_do_not() {
        let addr = SocketAddr::from(([0, 0, 0, 0], 3030));
        // WSAEACCES (10013): a port inside a range Windows reserved.
        let denied = std::io::Error::from_raw_os_error(10013);
        assert_eq!(denied.kind(), std::io::ErrorKind::PermissionDenied);
        let text = bind_error_text(addr, &denied);
        assert!(text.starts_with("could not listen on 0.0.0.0:3030: "), "{text}");
        assert!(text.contains("os error 10013") && text.ends_with("Windows may have reserved this port"), "{text}");

        // The io error can sit behind other errors, as it does in hyper's.
        #[derive(Debug)]
        struct Wrapper(std::io::Error);
        impl std::fmt::Display for Wrapper {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "error creating server listener")
            }
        }
        impl std::error::Error for Wrapper {
            fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
                Some(&self.0)
            }
        }
        assert!(bind_error_text(addr, &Wrapper(std::io::Error::from_raw_os_error(10013))).ends_with("Windows may have reserved this port"));

        // Address in use (10048): another program holds it for now; no claim about reservations.
        let busy = bind_error_text(addr, &Wrapper(std::io::Error::from_raw_os_error(10048)));
        assert!(busy.starts_with("could not listen on 0.0.0.0:3030: error creating server listener"), "{busy}");
        assert!(!busy.contains("reserved"), "{busy}");
    }

    #[tokio::test]
    async fn a_busy_port_reports_the_listen_error_without_a_reservation_hint() {
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = holder.local_addr().unwrap();
        let Err(error) = try_bind(&routes(), addr) else { panic!("the port is held") };
        assert!(error.starts_with(&format!("could not listen on {addr}: ")), "{error}");
        assert!(!error.contains("reserved"), "{error}");
    }

    #[test]
    fn backoff_walks_the_list_then_repeats_the_last_delay_until_reset() {
        let mut backoff = Backoff::new(&DOWN_RETRY_DELAYS_MS);
        let secs: Vec<u64> = (0..8).map(|_| backoff.next_delay().as_secs()).collect();
        assert_eq!(secs, [1, 2, 5, 10, 15, 15, 15, 15]);
        backoff.reset();
        assert_eq!(backoff.next_delay(), Duration::from_secs(1));

        // A bad list never yields a zero delay (no spinning).
        let mut empty = Backoff::new(&[]);
        assert!(empty.next_delay() > Duration::ZERO);
        let mut zero = Backoff::new(&[0]);
        assert!(zero.next_delay() > Duration::ZERO);
    }

    #[tokio::test]
    async fn binds_and_reports_listening() {
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Localhost);
        let task = tokio::spawn(async move { serve_supervised(real_bind(), modes_rx, |_| loopback(0), retry(&[]), report).await });
        let event = wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        let ServerEvent::Listening { mode, addr, error } = event else { unreachable!() };
        assert_eq!(mode, BindMode::Localhost);
        assert_eq!(error, None);
        assert!(ping(addr).await.ends_with("pong"));
        task.abort();
    }

    #[tokio::test]
    async fn switching_the_mode_rebinds_and_closes_the_old_listener() {
        // `localhost` gets a reserved port, `lan` an ephemeral one (loopback only: no firewall prompt).
        let reserved = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let localhost_port = reserved.local_addr().unwrap().port();
        let (seen, report) = recorder();
        let (modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let task = tokio::spawn(async move {
            serve_supervised(
                real_bind(),
                modes_rx,
                move |mode| match mode {
                    BindMode::Lan => loopback(0),
                    BindMode::Localhost => loopback(localhost_port),
                },
                retry(&[1, 1, 1]),
                report,
            )
            .await
        });
        let first = wait_for(&seen, "lan listening", |e| matches!(e, ServerEvent::Listening { mode: BindMode::Lan, .. })).await;
        let ServerEvent::Listening { addr: lan_addr, .. } = first else { unreachable!() };
        assert!(ping(lan_addr).await.ends_with("pong"));

        drop(reserved);
        modes_tx.send_replace(BindMode::Localhost);
        let second = wait_for(&seen, "localhost listening", |e| matches!(e, ServerEvent::Listening { mode: BindMode::Localhost, .. })).await;
        let ServerEvent::Listening { addr, error, .. } = second else { unreachable!() };
        assert_eq!(addr, loopback(localhost_port));
        assert_eq!(error, None);
        assert!(ping(addr).await.ends_with("pong"));
        assert!(!is_listening(lan_addr).await, "the old listener is closed");
        task.abort();
    }

    #[tokio::test]
    async fn a_failed_switch_keeps_serving_the_previous_mode_until_a_retry_works() {
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let busy = holder.local_addr().unwrap().port();
        let (seen, report) = recorder();
        let (modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        let task = tokio::spawn(async move {
            serve_supervised(
                real_bind(),
                modes_rx,
                move |mode| match mode {
                    BindMode::Lan => loopback(0),
                    BindMode::Localhost => loopback(busy),
                },
                retry(&[1, 1]),
                report,
            )
            .await
        });
        wait_for(&seen, "lan listening", |e| matches!(e, ServerEvent::Listening { mode: BindMode::Lan, .. })).await;
        seen.lock().unwrap().clear();

        modes_tx.send_replace(BindMode::Localhost);
        let fallback = wait_for(&seen, "fallback", |e| matches!(e, ServerEvent::Listening { .. })).await;
        let ServerEvent::Listening { mode, addr, error } = fallback else { unreachable!() };
        assert_eq!(mode, BindMode::Lan, "still the previous mode");
        assert!(error.as_deref().unwrap_or("").contains(&busy.to_string()), "{error:?}");
        assert!(ping(addr).await.ends_with("pong"));
        assert!(!seen.lock().unwrap().iter().any(|e| matches!(e, ServerEvent::Failed { .. })));

        // The port frees up; the host re-sends the same request, which retries.
        drop(holder);
        seen.lock().unwrap().clear();
        assert!(should_rebind(BindMode::Localhost, BindMode::Localhost, Some(BindMode::Lan)));
        modes_tx.send_modify(|_| {});
        let switched = wait_for(&seen, "localhost listening", |e| matches!(e, ServerEvent::Listening { mode: BindMode::Localhost, .. })).await;
        let ServerEvent::Listening { addr, error, .. } = switched else { unreachable!() };
        assert_eq!(addr, loopback(busy));
        assert_eq!(error, None);
        task.abort();
    }

    #[tokio::test]
    async fn a_failed_start_retries_when_notified() {
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = holder.local_addr().unwrap();
        let (seen, report) = recorder();
        let (modes_tx, modes_rx) = watch::channel(BindMode::Lan);
        // The quiet retry would only come in an hour: the notification has to do it.
        let plan = Retry {
            bind_ms: &[1],
            down_ms: &[3_600_000],
            stable_uptime: Duration::from_secs(60),
        };
        let task = tokio::spawn(async move { serve_supervised(real_bind(), modes_rx, move |_| addr, plan, report).await });
        wait_for(&seen, "failed", |e| matches!(e, ServerEvent::Failed { .. })).await;
        drop(holder);
        assert!(should_rebind(BindMode::Lan, BindMode::Lan, None));
        modes_tx.send_modify(|_| {});
        let event = wait_for(&seen, "listening", |e| matches!(e, ServerEvent::Listening { .. })).await;
        assert!(matches!(event, ServerEvent::Listening { addr: a, .. } if a == addr));
        task.abort();
    }

    #[test]
    fn rebind_rule() {
        use BindMode::*;
        // A real change always rebinds.
        assert!(should_rebind(Lan, Localhost, Some(Lan)));
        assert!(should_rebind(Localhost, Lan, Some(Localhost)));
        // The same request while it is served: nothing to do (the host re-sends on every start).
        assert!(!should_rebind(Lan, Lan, Some(Lan)));
        assert!(!should_rebind(Localhost, Localhost, Some(Localhost)));
        // The same request while it is not served (failed, fallback, still starting): retry.
        assert!(should_rebind(Lan, Lan, None));
        assert!(should_rebind(Localhost, Localhost, Some(Lan)));
    }

    #[test]
    fn lan_addresses_put_the_primary_and_private_ranges_first() {
        let list = vec![
            ("Loopback".to_string(), "127.0.0.1".parse().unwrap()),
            ("Tailscale".to_string(), "100.101.102.103".parse().unwrap()),
            ("vEthernet (WSL)".to_string(), "172.28.16.1".parse().unwrap()),
            ("Ethernet".to_string(), "192.168.1.50".parse().unwrap()),
            ("Ethernet".to_string(), "fe80::1".parse().unwrap()),
            ("APIPA".to_string(), "169.254.10.10".parse().unwrap()),
            ("Ethernet dup".to_string(), "192.168.1.50".parse().unwrap()),
            ("Mapped".to_string(), "::ffff:10.0.0.7".parse().unwrap()),
        ];
        let ordered = order_lan_addresses(list, Some("192.168.1.50".parse().unwrap()));
        let ips: Vec<&str> = ordered.iter().map(|a| a.ip.as_str()).collect();
        assert_eq!(ips, ["192.168.1.50", "172.28.16.1", "10.0.0.7", "100.101.102.103"]);
        assert_eq!(ordered[0].interface, "Ethernet");
        assert!(order_lan_addresses(vec![], None).is_empty());
    }

    #[test]
    fn an_offline_pc_still_gets_an_address() {
        let ip = |s: &str| -> IpAddr { s.parse().unwrap() };
        let lan = vec![
            ("Loopback".to_string(), ip("127.0.0.1")),
            ("APIPA".to_string(), ip("169.254.10.10")),
            ("Ethernet".to_string(), ip("192.168.1.50")),
        ];
        // Normal case: the default-route address wins.
        assert_eq!(primary_ip(Some(ip("10.0.0.7")), &lan), ip("10.0.0.7"));
        // A LAN without a gateway: the first usable address.
        assert_eq!(primary_ip(None, &lan), ip("192.168.1.50"));
        // Offline (Windows Sandbox with networking off, airplane mode): loopback, never an error.
        assert_eq!(primary_ip(None, &[("Loopback".to_string(), ip("127.0.0.1")), ("APIPA".to_string(), ip("169.254.10.10"))]), ip("127.0.0.1"));
        assert_eq!(primary_ip(None, &[]), ip("127.0.0.1"));
    }

    /// A `WebServer` in its initial state, with the LAN rules of the tests.
    fn test_server() -> WebServer {
        let (modes, _rx) = watch::channel(BindMode::Lan);
        let rules = origin::OriginRules { port: 3030, dev_port: None };
        let access = Arc::new(AccessGuard::new(
            OriginGuard::new(rules, true, None, Box::new(test_support::NoInterfaces)),
            test_support::lan_policy(),
        ));
        WebServer {
            port: 3030,
            access,
            lifecycle: Mutex::new(Lifecycle {
                state: ServerState::Starting,
                error: None,
                bound: None,
            }),
            modes,
        }
    }

    #[test]
    fn a_restarted_server_clears_the_error_it_showed() {
        let server = test_server();
        let addr = SocketAddr::from(([0, 0, 0, 0], 3030));
        let listening = |error: Option<&str>| ServerEvent::Listening { mode: BindMode::Lan, addr, error: error.map(str::to_string) };

        // A fallback listener shows why the requested mode is not served ...
        server.apply(&listening(Some("busy")));
        assert_eq!(server.status().error.as_deref(), Some("busy"));
        // ... then it ends: stopped, with its own reason and nothing bound ...
        server.apply(&ServerEvent::Stopped { error: "the local web server stopped unexpectedly".into() });
        let status = server.status();
        assert_eq!(status.state, ServerState::Stopped);
        assert_eq!(status.error.as_deref(), Some("the local web server stopped unexpectedly"));
        assert!(status.lan.bound_mode.is_none() && status.lan.bound_address.is_none());
        // ... and the restart that binds the requested mode reports no error, with or without a Starting in between.
        server.apply(&listening(None));
        assert_eq!((server.status().state, server.status().error), (ServerState::Listening, None));
        server.apply(&ServerEvent::Stopped { error: "x".into() });
        server.apply(&ServerEvent::Starting { mode: BindMode::Lan });
        assert_eq!((server.status().state, server.status().error), (ServerState::Starting, None));
        server.apply(&listening(None));
        assert_eq!((server.status().state, server.status().error), (ServerState::Listening, None));
    }

    #[test]
    fn status_never_carries_the_token() {
        let server = test_server();
        server.apply(&ServerEvent::Listening {
            mode: BindMode::Lan,
            addr: SocketAddr::from(([0, 0, 0, 0], 3030)),
            error: None,
        });
        let json = serde_json::to_string(&server.status()).unwrap();
        assert!(!json.contains(test_support::TOKEN), "{json}");
        assert!(json.contains(r#""lan":{"mode":"lan","boundMode":"lan","boundAddress":"0.0.0.0:3030","tokenSet":true,"legacyNoToken":false}"#), "{json}");

        // Applying the setting updates the rules; the same mode while served does not rebind.
        let mut rx = server.modes.subscribe();
        rx.borrow_and_update();
        server.apply_lan_access(&LanAccessSettings {
            mode: BindMode::Lan,
            token: String::new(),
            legacy_no_token: true,
        });
        assert!(!rx.has_changed().unwrap());
        let status = server.status();
        assert!(!status.lan.token_set);
        assert!(status.lan.legacy_no_token);
        server.apply_lan_access(&LanAccessSettings {
            mode: BindMode::Localhost,
            token: test_support::TOKEN.to_string(),
            legacy_no_token: false,
        });
        assert!(rx.has_changed().unwrap());
        assert_eq!(server.status().lan.mode, BindMode::Localhost);
        assert_eq!(server.status().lan.bound_mode, Some(BindMode::Lan), "until the supervisor rebinds");
    }
}
