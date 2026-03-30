//! Local HTTP server for OAuth redirects from the system browser. Emits `sigil-oauth` to the main webview.
//! Twitch Public apps: implicit return at `/oauth/twitch/implicit` (fragment → JS → POST `/done`). Code flow: `/oauth/{provider}/callback`.

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::Ipv4Addr;
use std::sync::OnceLock;
use std::time::Duration;

use serde::Serialize;
use tauri::{AppHandle, Emitter};
use tokio::sync::{oneshot, Mutex};
use warp::http::StatusCode;
use warp::Filter;

/// Fixed port — register `http://localhost:{PORT}/oauth/twitch|kick/callback` in developer consoles.
///
/// Use **`localhost`**, not `127.0.0.1`, in the redirect URL: Twitch often allows plain HTTP only for
/// `localhost` loopback while rejecting `http://127.0.0.1`. The server binds IPv4 loopback; browsers
/// usually resolve `localhost` to `127.0.0.1` on Windows.
pub const OAUTH_LOOPBACK_PORT: u16 = 17890;

pub const OAUTH_LOOPBACK_REDIRECT_HOST: &str = "localhost";

static ACTIVE_SHUTDOWN: OnceLock<Mutex<Option<oneshot::Sender<()>>>> = OnceLock::new();

fn active_shutdown_slot() -> &'static Mutex<Option<oneshot::Sender<()>>> {
    ACTIVE_SHUTDOWN.get_or_init(|| Mutex::new(None))
}

#[derive(Serialize)]
struct OauthEmitPayload {
    provider: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    code: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    access_token: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    state: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error_description: Option<String>,
}

fn emit_oauth_and_shutdown(app: AppHandle, payload: OauthEmitPayload) {
    let payload_json = serde_json::to_string(&payload).unwrap_or_else(|e| {
        eprintln!("[oauth_loopback] serialize oauth payload: {e}");
        serde_json::json!({
            "provider": &payload.provider,
            "error": "internal",
            "error_description": "Failed to serialize OAuth callback payload"
        })
        .to_string()
    });
    let app_emit = app.clone();
    if let Err(e) = app.run_on_main_thread(move || {
        if let Err(err) = app_emit.emit_str("sigil-oauth", payload_json) {
            eprintln!("[oauth_loopback] emit_str(sigil-oauth) failed: {err}");
        }
    }) {
        eprintln!("[oauth_loopback] run_on_main_thread(emit) failed: {e}");
    }
}

async fn shutdown_loopback_server() {
    let mut g = active_shutdown_slot().lock().await;
    if let Some(tx) = g.take() {
        let _ = tx.send(());
    }
}

/// HTML for Twitch implicit return: fragment is not sent to the server, so JS posts it to `/done`.
const TWITCH_IMPLICIT_LANDING: &str = r#"<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Sigil</title></head>
<body>
<p id="m">Completing sign-in…</p>
<script>
(function(){
  function send(body) {
    fetch('/oauth/twitch/implicit/done', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: body })
      .catch(function(){});
    document.getElementById('m').textContent = 'You can close this tab and return to Sigil.';
  }
  var qs = new URLSearchParams(window.location.search);
  if (qs.get('error')) {
    send(new URLSearchParams({
      error: qs.get('error') || '',
      error_description: qs.get('error_description') || '',
      state: qs.get('state') || ''
    }).toString());
    return;
  }
  var h = (location.hash || '').replace(/^#/, '');
  var p = new URLSearchParams(h);
  send(new URLSearchParams({
    access_token: p.get('access_token') || '',
    state: p.get('state') || '',
    error: p.get('error') || '',
    error_description: p.get('error_description') || ''
  }).toString());
})();
</script>
</body></html>"#;

/// Starts a short-lived server and returns the exact `redirect_uri` to use for the authorize request.
/// `mode`: omit or `"code"` → `/oauth/{provider}/callback`. `"implicit"` → `/oauth/twitch/implicit` (Twitch only).
#[tauri::command]
pub async fn oauth_loopback_start(
    app: AppHandle,
    provider: String,
    mode: Option<String>,
) -> Result<String, String> {
    if provider != "twitch" && provider != "kick" {
        return Err("invalid OAuth provider".to_string());
    }

    let implicit = mode.as_deref() == Some("implicit");
    if implicit && provider != "twitch" {
        return Err("implicit OAuth mode is only supported for twitch".to_string());
    }

    let redirect_uri = if implicit {
        format!(
            "http://{}:{}/oauth/twitch/implicit",
            OAUTH_LOOPBACK_REDIRECT_HOST, OAUTH_LOOPBACK_PORT
        )
    } else {
        format!(
            "http://{}:{}/oauth/{}/callback",
            OAUTH_LOOPBACK_REDIRECT_HOST, OAUTH_LOOPBACK_PORT, provider
        )
    };

    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    {
        let mut g = active_shutdown_slot().lock().await;
        if let Some(old) = g.take() {
            let _ = old.send(());
        }
        *g = Some(shutdown_tx);
    }

    tokio::time::sleep(Duration::from_millis(200)).await;

    let app_handle_code = app.clone();
    let expected_provider = provider.clone();

    let code_route = warp::path!("oauth" / String / "callback")
        .and(warp::query::<HashMap<String, String>>())
        .and_then(move |path_provider: String, q: HashMap<String, String>| {
            let app = app_handle_code.clone();
            let expected = expected_provider.clone();
            async move {
                if path_provider != expected {
                    return Ok::<_, Infallible>(warp::reply::with_status(
                        warp::reply::html("Not found"),
                        StatusCode::NOT_FOUND,
                    ));
                }

                let code = q.get("code").cloned();
                let state = q.get("state").cloned();
                let error = q.get("error").cloned();
                let ed = q.get("error_description").cloned().unwrap_or_default();
                let error_description = if ed.is_empty() {
                    None
                } else {
                    Some(ed)
                };

                let payload = OauthEmitPayload {
                    provider: path_provider,
                    code,
                    access_token: None,
                    state,
                    error,
                    error_description,
                };

                emit_oauth_and_shutdown(app.clone(), payload);
                shutdown_loopback_server().await;

                Ok::<_, Infallible>(warp::reply::with_status(
                    warp::reply::html(
                        "<!DOCTYPE html><html><head><meta charset=\"utf-8\"><title>Sigil</title></head>\
                         <body><p>You can close this tab and return to Sigil.</p></body></html>",
                    ),
                    StatusCode::OK,
                ))
            }
        });

    let twitch_implicit_get =
        warp::path!("oauth" / "twitch" / "implicit")
            .and(warp::get())
            .map(|| warp::reply::html(TWITCH_IMPLICIT_LANDING));

    let app_handle_implicit = app.clone();
    let twitch_implicit_post = warp::path!("oauth" / "twitch" / "implicit" / "done")
        .and(warp::post())
        .and(warp::body::content_length_limit(32_768))
        .and(warp::body::form::<HashMap<String, String>>())
        .and_then(move |form: HashMap<String, String>| {
            let app = app_handle_implicit.clone();
            async move {
                let access_token = form
                    .get("access_token")
                    .filter(|s| !s.is_empty())
                    .cloned();
                let state = form.get("state").cloned();
                let error = form.get("error").filter(|s| !s.is_empty()).cloned();
                let ed = form
                    .get("error_description")
                    .cloned()
                    .unwrap_or_default();
                let error_description = if ed.is_empty() {
                    None
                } else {
                    Some(ed)
                };

                let payload = OauthEmitPayload {
                    provider: "twitch".to_string(),
                    code: None,
                    access_token,
                    state,
                    error,
                    error_description,
                };

                emit_oauth_and_shutdown(app.clone(), payload);
                shutdown_loopback_server().await;

                Ok::<_, Infallible>(warp::reply::with_status(
                    warp::reply::html("<!DOCTYPE html><html><body>OK</body></html>"),
                    StatusCode::OK,
                ))
            }
        });

    let routes = code_route
        .or(twitch_implicit_get)
        .or(twitch_implicit_post);

    let shutdown = async move {
        let _ = shutdown_rx.await;
    };

    let serve = warp::serve(routes);
    match serve.try_bind_with_graceful_shutdown(
        (Ipv4Addr::LOCALHOST, OAUTH_LOOPBACK_PORT),
        shutdown,
    ) {
        Ok((_addr, server)) => {
            tauri::async_runtime::spawn(server);
            Ok(redirect_uri)
        }
        Err(e) => {
            let mut g = active_shutdown_slot().lock().await;
            g.take();
            Err(format!(
                "Could not bind OAuth callback on 127.0.0.1:{} (serves http://{}:{}/oauth/…) — {e}. Close other Sigil instances or apps using this port.",
                OAUTH_LOOPBACK_PORT,
                OAUTH_LOOPBACK_REDIRECT_HOST,
                OAUTH_LOOPBACK_PORT
            ))
        }
    }
}
