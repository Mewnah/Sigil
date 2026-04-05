use reqwest::header::{HeaderMap, HeaderValue, ACCEPT};
use reqwest::{Client, Response};
use std::sync::OnceLock;
use std::time::Duration;

const DOWNLOAD_TIMEOUT: Duration = Duration::from_secs(7200);

fn default_headers() -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(
        ACCEPT,
        HeaderValue::from_static("application/octet-stream, */*"),
    );
    h
}

/// Shared client for outbound HTTP from plugins (local TTS/STT, translate, model downloads).
/// Uses a long idle timeout so large model streams are not cut off; connect timeout stays bounded.
pub fn http_client() -> &'static Client {
    static CLIENT: OnceLock<Client> = OnceLock::new();
    CLIENT.get_or_init(|| {
        let ua = format!(
            "Sigil/{} (desktop; https://github.com/Mewnah/Sigil)",
            env!("CARGO_PKG_VERSION")
        );
        Client::builder()
            .connect_timeout(Duration::from_secs(30))
            // Whole-request cap for large downloads; short API calls still fail quickly on connect errors.
            .timeout(DOWNLOAD_TIMEOUT)
            .user_agent(ua)
            .default_headers(default_headers())
            .build()
            .expect("reqwest Client builder")
    })
}

/// GET with an explicit timeout suitable for multi‑GB model files.
pub fn http_download_get(url: &str) -> reqwest::RequestBuilder {
    http_client().get(url).timeout(DOWNLOAD_TIMEOUT)
}

/// For failed responses: status, Content-Type, and a short body preview (helps diagnose HTML / WAF pages).
pub async fn describe_failed_response(resp: Response) -> String {
    let status = resp.status();
    let ct = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_string();
    let bytes = match resp.bytes().await {
        Ok(b) => b,
        Err(e) => {
            return format!("HTTP {status} (failed to read body: {e})");
        }
    };
    let mut preview: String = String::from_utf8_lossy(&bytes[..bytes.len().min(384)]).into_owned();
    preview.retain(|c| c != '\n' && c != '\r');
    if preview.len() > 180 {
        preview.truncate(180);
        preview.push('…');
    }
    format!("HTTP {status} content-type={ct} body_preview={preview:?}")
}
