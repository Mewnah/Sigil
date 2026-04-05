use local_ip_address::local_ip;
use serde::{Deserialize, Serialize};
use std::{process::Command, sync::Arc};
use tauri::{
    async_runtime::Mutex,
    command,
    plugin::{Builder, TauriPlugin},
    Emitter, Manager, Runtime, State,
};
use tokio::sync::mpsc;
use warp::Filter;

use super::AppConfiguration;

mod assets;
mod peer;
mod pubsub;

struct PubSubInput {
    tx: Mutex<mpsc::Sender<String>>,
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
    return Ok(WebConfig {
        local_ip: ip.to_string(),
        port: config.port.to_string(),
        peer_path: "peer".to_string(),
        pubsub_path: "pubsub".to_string(),
    });
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
        if browser == "chrome" {
            if Command::new("open")
                .args(["-a", "Google Chrome", &data.url])
                .spawn()
                .is_ok()
            {
                return Ok(());
            }
        } else if browser == "edge" {
            if Command::new("open")
                .args(["-a", "Microsoft Edge", &data.url])
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
        for bin in [
            "google-chrome",
            "google-chrome-stable",
            "chromium",
            "chromium-browser",
        ] {
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

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    let (pubsub_input_tx, pubsub_input_rx) = mpsc::channel::<String>(1); // to pubsub
    let (pubsub_output_tx, mut pubsub_output_rx) = mpsc::channel::<String>(1); // to js
    Builder::new("web")
        .invoke_handler(tauri::generate_handler![open_browser, pubsub_broadcast, config])
        .setup(|app, _api| {
            app.manage(PubSubInput {
                tx: Mutex::new(pubsub_input_tx),
            });

            let app_port = app.state::<AppConfiguration>().port;

            let app_handle = app.clone();
            let a = Arc::new(app.asset_resolver());
            tauri::async_runtime::spawn(async move {
                let routes = warp::path!("ping")
                    .map(|| format!("pong"))
                    .or(peer::path())
                    .or(pubsub::path(pubsub_input_rx, pubsub_output_tx, app_handle))
                    .or(assets::path(a));

                // Match upstream Curses: bind all IPv4 interfaces so `http://localhost:PORT/client`
                // works the same as in the original app (127.0.0.1-only rejects some localhost paths).
                loop {
                    warp::serve(routes.clone())
                        .run(([0, 0, 0, 0], app_port))
                        .await
                }
            });
            let handle = app.clone();
            tauri::async_runtime::spawn(async move {
                loop {
                    if let Some(output) = pubsub_output_rx.recv().await {
                        let _ = handle.emit("pubsub", output);
                    }
                }
            });

            Ok(())
        })
        .build()
}
