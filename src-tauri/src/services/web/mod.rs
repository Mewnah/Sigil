//! `web` plugin: the local HTTP/WebSocket server on `0.0.0.0:<port>` (PeerJS signaling at `/peer`,
//! pubsub at `/pubsub`, `/ping`, and the app's static assets for `/client`, OBS and the helper
//! pages), plus the host-side commands around it.
//!
//! LAN security stage 1 (P0-I): WebSocket handshakes pass an Origin allow-list ([`origin`]) with a
//! kill switch (`lanOriginCheck` setting / `SIGIL_LAN_ORIGIN_CHECK`; re-enabling it closes foreign
//! connections admitted while it was off), interactive helper pages cannot be framed by other
//! sites ([`assets`]), binary pubsub frames are refused, and a bind failure is reported
//! (`web:server_status`, `plugin:web|server_status`) instead of panicking the server task and
//! leaving the event pump spinning.
//!
//! Commands: `config`, `pubsub_broadcast`, `open_browser`, `server_status`, `set_origin_check`.
//! Events: `pubsub` (peer message text → host), `web:server_status` ([`WebServerStatus`]),
//! `web:origin_rejected` ([`origin::OriginRejection`], throttled per origin).

use local_ip_address::local_ip;
use serde::{Deserialize, Serialize};
use std::{
    net::SocketAddr,
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
use tokio::sync::mpsc;
use warp::Filter;

use super::AppConfiguration;

mod assets;
mod origin;
mod peer;
mod pubsub;

use origin::OriginGuard;

/// Event carrying a peer message (text) to the host webview.
pub const PUBSUB_EVENT: &str = "pubsub";
/// Event emitted when the server starts listening, fails to bind or stops ([`WebServerStatus`]).
pub const SERVER_STATUS_EVENT: &str = "web:server_status";
/// Event emitted for refused WebSocket handshakes ([`origin::OriginRejection`], throttled).
pub const ORIGIN_REJECTED_EVENT: &str = "web:origin_rejected";

/// Delays between bind attempts: a previous Sigil instance may still be releasing the port.
const BIND_RETRY_DELAYS_MS: [u64; 5] = [250, 500, 1_000, 2_000, 4_000];

struct PubSubInput {
    tx: AsyncMutex<mpsc::Sender<String>>,
}

#[command]
async fn pubsub_broadcast(value: String, input: State<'_, PubSubInput>) -> Result<(), String> {
    let tx = input.tx.lock().await;
    tx.send(value).await.map_err(|e| e.to_string())
}

#[derive(Serialize)]
struct WebConfig {
    pub local_ip: String,
    pub port: String,
    pub peer_path: String,
    pub pubsub_path: String,
}

#[command]
async fn config(config: State<'_, AppConfiguration>) -> Result<WebConfig, String> {
    let Ok(ip) = local_ip() else {
        return Err("Error retrieving local IP".to_string());
    };
    Ok(WebConfig {
        local_ip: ip.to_string(),
        port: config.port.to_string(),
        peer_path: "peer".to_string(),
        pubsub_path: "pubsub".to_string(),
    })
}

/// Lifecycle of the local web server.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ServerState {
    /// Binding (including retries).
    Starting,
    Listening,
    /// Could not bind after every retry; `error` says why.
    Failed,
    /// The server future ended (should not happen while the app runs).
    Stopped,
}

/// Snapshot returned by `plugin:web|server_status` and emitted as `web:server_status`.
#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct WebServerStatus {
    pub state: ServerState,
    pub port: u16,
    /// Bind / server error text when `state` is `failed` or `stopped`.
    pub error: Option<String>,
    /// Whether the Origin allow-list is currently enforced.
    pub origin_check: bool,
    /// The `lanOriginCheck` setting as last pushed (or read from the settings file).
    pub origin_check_setting: bool,
    /// `SIGIL_LAN_ORIGIN_CHECK` override, if set; it wins over the setting.
    pub origin_check_env: Option<bool>,
}

/// Plugin state shared by the server task and the commands.
struct WebServer {
    port: u16,
    guard: Arc<OriginGuard>,
    lifecycle: Mutex<(ServerState, Option<String>)>,
}

impl WebServer {
    fn status(&self) -> WebServerStatus {
        let (state, error) = self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        WebServerStatus {
            state,
            port: self.port,
            error,
            origin_check: self.guard.is_enforced(),
            origin_check_setting: self.guard.setting(),
            origin_check_env: self.guard.env_override(),
        }
    }

    fn set_lifecycle(&self, state: ServerState, error: Option<String>) {
        *self
            .lifecycle
            .lock()
            .unwrap_or_else(PoisonError::into_inner) = (state, error);
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
    if server.guard.setting() != enabled {
        tracing::info!(target: "sigil::web", "Origin check setting changed to {enabled}");
    }
    let enforced = server.guard.set_setting(enabled);
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

/// Bind `routes` on `addr`, retrying after each of `retry_delays_ms` while the port is busy, then
/// serve until the server ends. Every state change goes to `report`; never panics or busy-loops.
async fn serve_with_retry<F>(routes: F, addr: SocketAddr, retry_delays_ms: &[u64], report: impl Fn(ServerState, Option<String>))
where
    F: Filter + Clone + Send + Sync + 'static,
    F::Extract: warp::Reply,
{
    let mut attempt = 0usize;
    loop {
        match warp::serve(routes.clone()).try_bind_ephemeral(addr) {
            Ok((bound, server)) => {
                tracing::info!(target: "sigil::web", "local web server listening on {bound}");
                report(ServerState::Listening, None);
                server.await;
                let error = "the local web server stopped unexpectedly".to_string();
                tracing::error!(target: "sigil::web", "{error} ({bound})");
                report(ServerState::Stopped, Some(error));
                return;
            }
            Err(e) => {
                let error = format!("could not listen on {addr}: {e}");
                let Some(delay) = retry_delays_ms.get(attempt) else {
                    tracing::error!(
                        target: "sigil::web",
                        "{error}. /client, OBS, mic.html and Link apps are unavailable until Sigil restarts with a free port."
                    );
                    report(ServerState::Failed, Some(error));
                    return;
                };
                tracing::warn!(target: "sigil::web", "{error}; retrying in {delay} ms");
                attempt += 1;
                tokio::time::sleep(Duration::from_millis(*delay)).await;
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
            set_origin_check
        ])
        .setup(|app, _api| {
            app.manage(PubSubInput {
                tx: AsyncMutex::new(pubsub_input_tx),
            });

            let app_port = app.state::<AppConfiguration>().port;

            // Kill switch: the persisted setting is read here so it applies before the webview has
            // loaded (OBS and /client reconnect as soon as the server is up); the host then pushes
            // live changes through `set_origin_check`.
            let persisted = match app.path().app_data_dir() {
                Ok(dir) => origin::persisted_origin_check(&dir),
                Err(e) => {
                    tracing::warn!(target: "sigil::web", "no app data folder ({e}); the Origin check uses its default");
                    None
                }
            };
            let guard = Arc::new(OriginGuard::for_app(app_port, persisted));
            if let Some(forced) = guard.env_override() {
                tracing::warn!(
                    target: "sigil::web",
                    "{} is set: the Origin check is forced {}",
                    origin::ORIGIN_CHECK_ENV,
                    if forced { "on" } else { "off" }
                );
            } else if !guard.is_enforced() {
                tracing::warn!(target: "sigil::web", "the Origin check is disabled by the lanOriginCheck setting");
            }
            let rejection_handle = app.clone();
            guard.set_notifier(move |rejection| {
                let _ = rejection_handle.emit(ORIGIN_REJECTED_EVENT, rejection.clone());
            });

            let server = Arc::new(WebServer {
                port: app_port,
                guard: guard.clone(),
                lifecycle: Mutex::new((ServerState::Starting, None)),
            });
            app.manage(server.clone());

            let status_handle = app.clone();
            let a = Arc::new(app.asset_resolver());
            tauri::async_runtime::spawn(async move {
                let routes = warp::path!("ping")
                    .map(|| "pong".to_string())
                    .or(peer::path(guard.clone()))
                    .or(pubsub::path(pubsub_input_rx, pubsub_output_tx, guard))
                    .or(assets::path(a));

                // Match upstream Curses: bind all IPv4 interfaces so `http://localhost:PORT/client`
                // works the same as in the original app (127.0.0.1-only rejects some localhost paths).
                let addr = SocketAddr::from(([0, 0, 0, 0], app_port));
                serve_with_retry(routes, addr, &BIND_RETRY_DELAYS_MS, |state, error| {
                    server.set_lifecycle(state, error);
                    let _ = status_handle.emit(SERVER_STATUS_EVENT, server.status());
                })
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

    type Seen = Arc<StdMutex<Vec<(ServerState, Option<String>)>>>;

    #[tokio::test]
    async fn bind_failure_reports_instead_of_panicking() {
        // Hold a port so every bind attempt fails.
        let holder = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = holder.local_addr().unwrap();
        let seen: Seen = Arc::default();
        let sink = seen.clone();
        let routes = warp::path!("ping").map(|| "pong");

        serve_with_retry(routes, addr, &[1, 1], move |state, error| sink.lock().unwrap().push((state, error))).await;

        let seen = seen.lock().unwrap();
        assert_eq!(seen.len(), 1, "{seen:?}");
        assert_eq!(seen[0].0, ServerState::Failed);
        assert!(
            seen[0]
                .1
                .as_deref()
                .unwrap_or("")
                .contains(&addr.to_string()),
            "{seen:?}"
        );
        drop(holder);
    }

    #[tokio::test]
    async fn binds_and_reports_listening() {
        let seen: Seen = Arc::default();
        let sink = seen.clone();
        let routes = warp::path!("ping").map(|| "pong");
        let addr = SocketAddr::from(([127, 0, 0, 1], 0));
        let task = tokio::spawn(serve_with_retry(routes, addr, &[], move |state, error| {
            sink.lock().unwrap().push((state, error))
        }));
        for _ in 0..200 {
            if !seen.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        assert_eq!(seen.lock().unwrap().first().map(|s| s.0), Some(ServerState::Listening));
        task.abort();
    }
}
