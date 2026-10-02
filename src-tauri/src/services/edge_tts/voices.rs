//! The Read-aloud voice list (P6-H): `GET …/voices/list` with the browser's headers and the
//! `Sec-MS-GEC` token, cached for a few hours (the list changes rarely; the panel asks often).

use std::sync::{LazyLock, Mutex, PoisonError};
use std::time::{Duration, Instant};

use reqwest::header::{HeaderName, HeaderValue, ACCEPT, ACCEPT_LANGUAGE, COOKIE, USER_AGENT};
use reqwest::Client;
use serde_json::Value;

use super::cloud::parse_http_date;
use super::protocol;
use super::session::{random_id, Clock, WHAT};
use super::EdgeTtsVoice;
use crate::services::http_client::{describe_api_error, describe_request_error, LIST_TIMEOUT};

/// How long a fetched list is reused.
const CACHE_TTL: Duration = Duration::from_secs(6 * 3600);

/// The last fetched list and when it was fetched.
type Cached = Option<(Instant, Vec<EdgeTtsVoice>)>;

static CACHE: LazyLock<Mutex<Cached>> = LazyLock::new(Default::default);

/// Voices from a list answer (an array of `{ShortName, FriendlyName, Locale, Gender, …}`), sorted
/// by locale and name, without duplicates.
pub(crate) fn parse_voices(v: &Value) -> Vec<EdgeTtsVoice> {
    let Some(items) = v.as_array() else {
        return Vec::new();
    };
    let text = |o: &Value, k: &str| o.get(k).and_then(Value::as_str).map(str::trim).unwrap_or("").to_string();
    let mut voices: Vec<EdgeTtsVoice> = items
        .iter()
        .filter_map(|o| {
            let short_name = text(o, "ShortName");
            if short_name.is_empty() {
                return None;
            }
            let mut friendly_name = text(o, "FriendlyName");
            if friendly_name.is_empty() {
                friendly_name = text(o, "DisplayName");
            }
            let mut locale = text(o, "Locale");
            if locale.is_empty() {
                locale = short_name.splitn(3, '-').take(2).collect::<Vec<_>>().join("-");
            }
            Some(EdgeTtsVoice { short_name, friendly_name, locale, gender: text(o, "Gender") })
        })
        .collect();
    voices.sort_by(|a, b| a.locale.cmp(&b.locale).then_with(|| a.short_name.cmp(&b.short_name)));
    voices.dedup_by(|a, b| a.short_name == b.short_name);
    voices
}

fn request(client: &Client, base: &str, now_unix: i64) -> reqwest::RequestBuilder {
    let gec = protocol::sec_ms_gec(now_unix);
    let static_header = |name: &'static str, value: &str| {
        (HeaderName::from_static(name), HeaderValue::from_str(value).unwrap_or_else(|_| HeaderValue::from_static("")))
    };
    let mut req = client
        .get(protocol::voices_url(base, &gec))
        .timeout(LIST_TIMEOUT)
        .header(USER_AGENT, protocol::user_agent())
        .header(ACCEPT_LANGUAGE, "en-US,en;q=0.9")
        .header(ACCEPT, "*/*")
        .header(COOKIE, format!("muid={};", random_id().to_ascii_uppercase()));
    // No Accept-Encoding: the HTTP client has no decompression (the browser's `br, zstd` would be
    // answered compressed).
    for (name, value) in [
        static_header("authority", "speech.platform.bing.com"),
        static_header("sec-ch-ua", &protocol::sec_ch_ua()),
        static_header("sec-ch-ua-mobile", "?0"),
        static_header("sec-fetch-site", "none"),
        static_header("sec-fetch-mode", "cors"),
        static_header("sec-fetch-dest", "empty"),
    ] {
        req = req.header(name, value);
    }
    req
}

/// Fetch the voice list (no cache).
pub(crate) async fn fetch(client: &Client, base: &str, clock: &dyn Clock) -> Result<Vec<EdgeTtsVoice>, String> {
    let what = format!("{WHAT} voice list");
    for attempt in 0..2 {
        let resp = request(client, base, clock.now())
            .send()
            .await
            .map_err(|e| describe_request_error(&what, &e, LIST_TIMEOUT))?;
        let status = resp.status();
        if status.as_u16() == 403 && attempt == 0 {
            // The token was refused: correct the clock from the server's and try once more.
            if let Some(t) = resp.headers().get("date").and_then(|v| v.to_str().ok()).and_then(parse_http_date) {
                clock.adjust(t);
            }
            continue;
        }
        if !status.is_success() {
            return Err(describe_api_error(&what, resp).await);
        }
        let v: Value = resp.json().await.map_err(|e| format!("{what}: unexpected answer ({e})."))?;
        let voices = parse_voices(&v);
        if voices.is_empty() {
            return Err(format!("{what}: the service listed no voices."));
        }
        return Ok(voices);
    }
    Err(format!("{what}: the service refused the request (HTTP 403). Check that the PC clock is right."))
}

/// The voice list, from the cache when it is fresh.
pub(crate) async fn list(client: &Client, base: &str, clock: &dyn Clock) -> Result<Vec<EdgeTtsVoice>, String> {
    if let Some((at, voices)) = CACHE.lock().unwrap_or_else(PoisonError::into_inner).as_ref() {
        if at.elapsed() < CACHE_TTL {
            return Ok(voices.clone());
        }
    }
    let voices = fetch(client, base, clock).await?;
    *CACHE.lock().unwrap_or_else(PoisonError::into_inner) = Some((Instant::now(), voices.clone()));
    Ok(voices)
}
