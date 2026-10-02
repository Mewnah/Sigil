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
//! server keeps listening on the previous one and reports why.
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
    net::{IpAddr, Ipv4Addr, SocketAddr},
    process::Command,
    sync::{Arc, Mutex, PoisonError},
    time::Duration,
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

#[command]
async fn config(config: State<'_, AppConfiguration>) -> Result<WebConfig, String> {
    let Ok(ip) = local_ip() else {
        return Err("Error retrieving local IP".to_string());
    };
    let interfaces = local_ip_address::list_afinet_netifas().unwrap_or_default();
    Ok(WebConfig {
        local_ip: ip.to_string(),
        port: config.port.to_string(),
        peer_path: "peer".to_string(),
        pubsub_path: "pubsub".to_string(),
        lan_addresses: order_lan_addresses(interfaces, Some(ip)),
    })
}

/// The address the server will listen on at startup (`lanAccess.mode` from the persisted settings:
/// `0.0.0.0` for `lan`, the default, or `127.0.0.1` for `localhost`). For main.rs's port pre-check
/// before Tauri starts, so it checks the address the server really binds (P6-D hotspot request).
pub fn startup_bind_ip() -> IpAddr {
    super::diagnostics::app_data_dir()
        .and_then(|dir| auth::persisted_lan_access(&dir))
        .unwrap_or_default()
        .mode
        .ip()
}

/// Lifecycle of the local web server.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerState {
    /// Binding (including retries and bind-mode switches).
    Starting,
    Listening,
    /// Could not bind after every retry; `error` says why.
    Failed,
    /// The server future ended (should not happen while the app runs).
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
        Err(e) => Err(format!("could not listen on {addr}: {e}")),
    }
}

/// Completes when the requested mode is (re)notified; never when the sender is gone.
async fn next_mode_change(modes: &mut watch::Receiver<BindMode>) {
    if modes.changed().await.is_err() {
        std::future::pending::<()>().await;
    }
}

/// Bind `addr`, retrying after each of `retry_delays_ms` while the port is busy.
async fn bind_retrying<F>(routes: &F, addr: SocketAddr, retry_delays_ms: &[u64], modes: &mut watch::Receiver<BindMode>) -> Result<Running, BindError>
where
    F: Filter + Clone + Send + Sync + 'static,
    F::Extract: warp::Reply,
{
    let mut attempt = 0usize;
    loop {
        match try_bind(routes, addr) {
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

/// Serve `routes` on the address of the requested [`BindMode`] (`addr_for`), rebinding whenever
/// `modes` is notified. Every state change goes to `report`; never panics or busy-loops.
///
/// - A bind is retried after each of `retry_delays_ms` (a previous Sigil may still hold the port).
/// - On a switch the old listener closes at once (graceful shutdown), in-flight requests get
///   [`DRAIN_TIMEOUT`], and the new address is bound.
/// - If the new address fails, the previous mode is bound again and reported as `Listening` with
///   the error, so OBS and the pages on this PC keep working; a later notification retries.
/// - With nothing to fall back to, `Failed` is reported and the next notification retries.
async fn serve_supervised<F>(routes: F, mut modes: watch::Receiver<BindMode>, addr_for: impl Fn(BindMode) -> SocketAddr, retry_delays_ms: &[u64], report: impl Fn(ServerEvent))
where
    F: Filter + Clone + Send + Sync + 'static,
    F::Extract: warp::Reply,
{
    let mut last_good: Option<BindMode> = None;
    loop {
        let wanted = *modes.borrow_and_update();
        report(ServerEvent::Starting { mode: wanted });
        let running = match bind_retrying(&routes, addr_for(wanted), retry_delays_ms, &mut modes).await {
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
                let fallback = last_good
                    .filter(|previous| *previous != wanted)
                    .and_then(|previous| match try_bind(&routes, addr_for(previous)) {
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
                            "{error}. /client, OBS, mic.html and Link apps are unavailable until the port is free (switch LAN access or restart Sigil)."
                        );
                        report(ServerEvent::Failed { error });
                        next_mode_change(&mut modes).await;
                        continue;
                    }
                }
            }
        };

        let Running { addr, mut server, stop } = running;
        tokio::select! {
            () = &mut server => {
                let error = "the local web server stopped unexpectedly".to_string();
                tracing::error!(target: "sigil::web", "{error} ({addr})");
                report(ServerEvent::Stopped { error });
                return;
            }
            () = next_mode_change(&mut modes) => {
                tracing::info!(target: "sigil::web", "closing {addr} to apply the LAN access mode");
                let _ = stop.send(());
                if tokio::time::timeout(DRAIN_TIMEOUT, &mut server).await.is_err() {
                    tracing::warn!(target: "sigil::web", "requests on {addr} did not finish within {DRAIN_TIMEOUT:?}; dropping them");
                }
            }
        }
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
            let lan_settings = app_data
                .as_deref()
                .and_then(auth::persisted_lan_access)
                .unwrap_or_default();

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

                // `lan`: all IPv4 interfaces, as upstream Curses (so `http://localhost:PORT/client`
                // and LAN clients work; a 127.0.0.1-only bind rejected some localhost paths there).
                // `localhost`: 127.0.0.1 only.
                serve_supervised(
                    routes,
                    modes_rx,
                    move |mode| mode.socket_addr(app_port),
                    &BIND_RETRY_DELAYS_MS,
                    move |event| {
                        server.apply(&event);
                        let _ = status_handle.emit(SERVER_STATUS_EVENT, server.status());
                    },
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

    fn routes() -> impl Filter<Extract = (String,), Error = warp::Rejection> + Clone + Send + Sync + 'static {
        warp::path!("ping").map(|| "pong".to_string())
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
        let task = tokio::spawn(async move { serve_supervised(routes(), modes_rx, move |_| addr, &[1, 1], report).await });

        let failed = wait_for(&seen, "failed", |e| matches!(e, ServerEvent::Failed { .. })).await;
        let ServerEvent::Failed { error } = failed else { unreachable!() };
        assert!(error.contains(&addr.to_string()), "{error}");
        assert_eq!(seen.lock().unwrap()[0], ServerEvent::Starting { mode: BindMode::Lan });
        assert!(!task.is_finished(), "it waits for a retry instead of ending");
        task.abort();
        drop(holder);
    }

    #[tokio::test]
    async fn binds_and_reports_listening() {
        let (seen, report) = recorder();
        let (_modes_tx, modes_rx) = watch::channel(BindMode::Localhost);
        let task = tokio::spawn(async move { serve_supervised(routes(), modes_rx, |_| loopback(0), &[], report).await });
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
                routes(),
                modes_rx,
                move |mode| match mode {
                    BindMode::Lan => loopback(0),
                    BindMode::Localhost => loopback(localhost_port),
                },
                &[1, 1, 1],
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
                routes(),
                modes_rx,
                move |mode| match mode {
                    BindMode::Lan => loopback(0),
                    BindMode::Localhost => loopback(busy),
                },
                &[1, 1],
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
        let task = tokio::spawn(async move { serve_supervised(routes(), modes_rx, move |_| addr, &[1], report).await });
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
    fn status_never_carries_the_token() {
        let (modes, _rx) = watch::channel(BindMode::Lan);
        let rules = origin::OriginRules { port: 3030, dev_port: None };
        let access = Arc::new(AccessGuard::new(
            OriginGuard::new(rules, true, None, Box::new(test_support::NoInterfaces)),
            test_support::lan_policy(),
        ));
        let server = WebServer {
            port: 3030,
            access,
            lifecycle: Mutex::new(Lifecycle {
                state: ServerState::Starting,
                error: None,
                bound: None,
            }),
            modes,
        };
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
