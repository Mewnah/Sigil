//! `plugin:google-fonts|google_fonts_list`: the Google Fonts Developer API family list, fetched in Rust
//! so the API key stays out of the JS bundle (and out of the web server that hands the bundle to the
//! LAN).
//!
//! The Developer API (`GET https://www.googleapis.com/webfonts/v1/webfonts?key=<key>`) cannot be called
//! without a key, so the request has to be made with one. The key is compiled in by `build.rs`
//! ([`crate::services::embedded_secrets`], lightly obfuscated: not protection; restrict the key to the
//! Fonts API in the Google Cloud console and rotate it if it leaks). Only the list comes back,
//! `{"items": [...]}` exactly as Google sends it (family, variants, files, ...): never the key.
//!
//! - IPC (`plugin-commands/google-fonts.txt`; wire camelCase; caller `Service_Files.loadFonts` in
//!   `src/client/services/files/index.ts`, host only).
//! - Caching: the frontend keeps the list for the life of the page, as before; this side also keeps the
//!   last good answer for [`CACHE_TTL`], so a reloaded page does not download the megabyte again. A
//!   failure is not cached.
//! - A per-request [`LIST_TIMEOUT`] on the shared [`http_client`]. A build without a key compiles; the
//!   command then says which variable to set.
//! - The key is a query parameter by Google's design, so errors are built with
//!   [`describe_request_error`] (which drops the query) and then [`scrub`]bed. Nothing logs the URL.

use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use reqwest::header::ACCEPT;
use reqwest::Client;
use serde_json::{json, Value};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::embedded_secrets::{self, scrub};
use crate::services::http_client::{
    describe_api_error, describe_request_error, http_client, read_body_capped, HEALTH_TIMEOUT, LIST_TIMEOUT,
};

const PLUGIN_NAME: &str = "google-fonts";
const FONTS_URL: &str = "https://www.googleapis.com/webfonts/v1/webfonts";
/// How long a good answer is reused (the list changes a few times a month).
const CACHE_TTL: Duration = Duration::from_secs(6 * 60 * 60);
/// Largest list read (the real one is about 1.5 MB).
const MAX_LIST_BYTES: usize = 16 * 1024 * 1024;

const MISSING_KEY_MESSAGE: &str = "The Google Fonts list is not available in this build: no API key was compiled in. \
    Set SIGIL_GOOGLE_FONTS_API_KEY (or the older VITE_GOOGLE_FONTS_API_KEY) in .env or the build environment and rebuild Sigil.";

/// The last good list and when it arrived.
struct Cached {
    at: Instant,
    list: Arc<Value>,
}

static CACHE: Mutex<Option<Cached>> = Mutex::new(None);

/// The cached list if it is younger than `ttl` at `now`.
fn fresh(cache: &Option<Cached>, now: Instant, ttl: Duration) -> Option<Arc<Value>> {
    cache
        .as_ref()
        .filter(|c| now.saturating_duration_since(c.at) < ttl)
        .map(|c| Arc::clone(&c.list))
}

/// `{"items": [...]}` out of Google's answer (other top-level fields such as `kind` are dropped).
fn parse_list(body: &[u8]) -> Result<Value, String> {
    let json: Value = serde_json::from_slice(body)
        .map_err(|_| "Google Fonts list: the answer was not JSON.".to_string())?;
    match json.get("items") {
        Some(items) if items.is_array() => Ok(json!({ "items": items })),
        _ => Err("Google Fonts list: the answer has no font list.".to_string()),
    }
}

/// Download the list from `url` with `key` as the `key` query parameter.
async fn fetch_list(client: &Client, url: &str, key: &str) -> Result<Value, String> {
    let what = "Google Fonts list";
    let sent = client
        .get(url)
        .query(&[("key", key)])
        .header(ACCEPT, "application/json")
        .timeout(LIST_TIMEOUT)
        .send()
        .await;
    let resp = match sent {
        Ok(resp) => resp,
        Err(e) => return Err(scrub(&describe_request_error(what, &e, LIST_TIMEOUT), &[key])),
    };
    if !resp.status().is_success() {
        return Err(scrub(&describe_api_error(what, resp).await, &[key]));
    }
    let body = read_body_capped(what, resp, MAX_LIST_BYTES, HEALTH_TIMEOUT).await?;
    parse_list(&body)
}

/// The list: from the cache when fresh, else fetched with `key` and cached.
async fn list_with(client: &Client, url: &str, key: Option<&str>, cache: &Mutex<Option<Cached>>) -> Result<Value, String> {
    if let Some(hit) = fresh(&cache.lock().unwrap_or_else(PoisonError::into_inner), Instant::now(), CACHE_TTL) {
        return Ok((*hit).clone());
    }
    let Some(key) = key else {
        return Err(MISSING_KEY_MESSAGE.to_string());
    };
    let list = fetch_list(client, url, key).await?;
    *cache.lock().unwrap_or_else(PoisonError::into_inner) = Some(Cached { at: Instant::now(), list: Arc::new(list.clone()) });
    Ok(list)
}

/// The Google Fonts family list, `{"items": [...]}`.
#[command]
pub async fn google_fonts_list() -> Result<Value, String> {
    let key = embedded_secrets::google_fonts_api_key();
    let outcome = list_with(http_client(), FONTS_URL, key.as_deref(), &CACHE).await;
    if let Err(e) = &outcome {
        tracing::warn!(target: "sigil::fonts", "{e}");
    }
    outcome
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![google_fonts_list])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::http_client::STALL_TIMEOUT;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;
    use tokio::sync::oneshot;

    // A fake key; the real one never appears in a test.
    const FAKE_KEY: &str = "fake-fonts-key-0001";

    fn test_client() -> Client {
        Client::builder().no_proxy().build().unwrap()
    }

    /// One-shot server on 127.0.0.1 that answers `response` and returns the request head it saw.
    async fn serve_capture(response: String) -> (String, oneshot::Receiver<String>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let (tx, rx) = oneshot::channel();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut head = Vec::new();
            let mut byte = [0u8; 1];
            while !head.ends_with(b"\r\n\r\n") {
                if sock.read(&mut byte).await.unwrap_or(0) == 0 {
                    break;
                }
                head.push(byte[0]);
            }
            let _ = sock.write_all(response.as_bytes()).await;
            let _ = sock.shutdown().await;
            let _ = tx.send(String::from_utf8_lossy(&head).to_string());
        });
        (format!("http://{addr}/webfonts/v1/webfonts"), rx)
    }

    fn json_response(status: &str, body: &str) -> String {
        format!(
            "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
    }

    const LIST: &str = r#"{"kind":"webfonts#webfontList","items":[{"family":"Roboto","variants":["regular","700"],"files":{"regular":"http://fonts.gstatic.com/r.ttf"},"category":"sans-serif"}]}"#;

    #[test]
    fn parse_list_keeps_items_and_drops_the_rest() {
        let v = parse_list(LIST.as_bytes()).unwrap();
        assert_eq!(v.as_object().unwrap().keys().collect::<Vec<_>>(), ["items"]);
        assert_eq!(v["items"][0]["family"], "Roboto");
        assert_eq!(v["items"][0]["files"]["regular"], "http://fonts.gstatic.com/r.ttf");
        for bad in ["", "<html>", "{}", r#"{"items":"nope"}"#, "[]"] {
            assert!(parse_list(bad.as_bytes()).is_err(), "{bad:?}");
        }
    }

    #[tokio::test]
    async fn fetch_sends_the_key_as_the_query_parameter_and_returns_the_list() {
        let (url, head) = serve_capture(json_response("200 OK", LIST)).await;
        let list = fetch_list(&test_client(), &url, FAKE_KEY).await.unwrap();
        assert_eq!(list["items"][0]["family"], "Roboto");
        let head = head.await.unwrap();
        let line = head.lines().next().unwrap();
        assert!(line.starts_with("GET /webfonts/v1/webfonts?key=fake-fonts-key-0001 "), "{line}");
        assert!(head.to_ascii_lowercase().contains("accept: application/json"), "{head}");
        // The key is never part of what comes back.
        assert!(!list.to_string().contains(FAKE_KEY));
    }

    #[tokio::test]
    async fn errors_never_contain_the_key() {
        // Google-style rejection that happens to echo the key.
        let body = format!(r#"{{"error":{{"code":400,"message":"API key not valid: {FAKE_KEY}"}}}}"#);
        let (url, _head) = serve_capture(json_response("400 Bad Request", &body)).await;
        let err = fetch_list(&test_client(), &url, FAKE_KEY).await.unwrap_err();
        assert!(err.starts_with("Google Fonts list: HTTP 400"), "{err}");
        assert!(err.contains("API key not valid") && err.contains("<redacted>"), "{err}");
        assert!(!err.contains(FAKE_KEY), "{err}");

        // A dead endpoint: reqwest's own message would carry the URL with ?key=..., ours must not.
        let port = TcpListener::bind("127.0.0.1:0").await.unwrap().local_addr().unwrap().port();
        let err = fetch_list(&test_client(), &format!("http://127.0.0.1:{port}/webfonts"), FAKE_KEY)
            .await
            .unwrap_err();
        assert!(err.starts_with("Google Fonts list"), "{err}");
        assert!(!err.contains(FAKE_KEY) && !err.contains("key="), "{err}");

        // A 200 that is not a list is reported without quoting it.
        let (url, _head) = serve_capture(json_response("200 OK", r#"{"oops":"fake-leak"}"#)).await;
        let err = fetch_list(&test_client(), &url, FAKE_KEY).await.unwrap_err();
        assert!(!err.contains("fake-leak"), "{err}");
    }

    #[tokio::test]
    async fn a_build_without_a_key_says_which_variable_to_set() {
        let cache = Mutex::new(None);
        let err = list_with(&test_client(), "http://127.0.0.1:9/never", None, &cache).await.unwrap_err();
        assert!(err.contains("SIGIL_GOOGLE_FONTS_API_KEY") && err.contains("VITE_GOOGLE_FONTS_API_KEY"), "{err}");
        assert!(err.contains("rebuild"), "{err}");
    }

    #[tokio::test]
    async fn a_good_list_is_cached_and_a_failure_is_not() {
        let cache = Mutex::new(None);
        // Failure first: nothing cached, and the next call still goes out.
        let (bad_url, _h) = serve_capture(json_response("500 Internal Server Error", "{}")).await;
        assert!(list_with(&test_client(), &bad_url, Some(FAKE_KEY), &cache).await.is_err());
        assert!(cache.lock().unwrap().is_none());

        let (url, head) = serve_capture(json_response("200 OK", LIST)).await;
        let first = list_with(&test_client(), &url, Some(FAKE_KEY), &cache).await.unwrap();
        head.await.unwrap();
        assert!(cache.lock().unwrap().is_some());

        // The server is gone (it served one connection): a second call is answered from the cache,
        // even without a key.
        let second = list_with(&test_client(), &url, None, &cache).await.unwrap();
        assert_eq!(first, second);
    }

    #[test]
    fn the_cache_expires_after_its_ttl() {
        let t0 = Instant::now();
        let cache = Some(Cached { at: t0, list: Arc::new(json!({"items": []})) });
        assert!(fresh(&cache, t0, CACHE_TTL).is_some());
        assert!(fresh(&cache, t0 + CACHE_TTL - Duration::from_secs(1), CACHE_TTL).is_some());
        assert!(fresh(&cache, t0 + CACHE_TTL, CACHE_TTL).is_none());
        assert!(fresh(&None, t0, CACHE_TTL).is_none());
    }

    #[test]
    fn the_request_timeout_is_the_list_class() {
        assert_eq!(LIST_TIMEOUT, Duration::from_secs(20));
        assert!(LIST_TIMEOUT <= STALL_TIMEOUT);
        assert_eq!(FONTS_URL, "https://www.googleapis.com/webfonts/v1/webfonts");
    }
}
