//! `translate` plugin: LibreTranslate HTTP client.
//!
//! # Per-call contract (frozen by P0-S0 for P0-G)
//! `get_languages`, `translate`, `detect_language` and `check_availability` take optional
//! `endpoint?: string` and `apiKey?: string`. They apply to that call only:
//! - `endpoint` absent or blank → the endpoint stored by `set_endpoint` is used;
//! - `apiKey` absent (`undefined` / `null`) → the key stored by `set_api_key` is used;
//!   `apiKey: ""` explicitly means "no key" for that call.
//!
//! `set_endpoint` / `set_api_key` stay for callers that configure once. The API key is sent only in
//! the JSON body of `/translate` and `/detect` (LibreTranslate has no header auth); it is never put
//! in a URL, and `/languages` is public so it gets no key at all.
//!
//! Every call has a per-request timeout (health 10 s, lists 20 s, translate 20 s + 10 ms/char up to
//! 60 s), so a hung server can no longer block a caption for the shared client's 2 h download cap.
//!
//! DeepL API Free / Pro (P4-D): `deepl_translate { request }`, `deepl_usage { apiKey, plan? }` and
//! `deepl_languages { apiKey, plan?, kind }` live in [`deepl`] (`translate/deepl.rs`): the key only in the
//! `Authorization: DeepL-Auth-Key` header, the Free host for `:fx` keys, a per-request timeout.

pub mod deepl;

use crate::services::http_client::{
    api_url, describe_api_error, describe_request_error, http_client, non_empty_key,
    pick_endpoint, probe_ok, translate_timeout, LIST_TIMEOUT, TRANSLATE_TIMEOUT,
};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex, PoisonError};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    AppHandle, Manager, Runtime, State,
};

const DEFAULT_ENDPOINT: &str = "http://localhost:5000";

#[derive(Clone)]
pub struct TranslateState {
    endpoint: Arc<Mutex<String>>,
    /// Optional LibreTranslate API key, sent in the JSON body as `api_key`.
    api_key: Arc<Mutex<String>>,
}

impl TranslateState {
    fn new() -> Self {
        Self {
            endpoint: Arc::new(Mutex::new(DEFAULT_ENDPOINT.to_string())),
            api_key: Arc::new(Mutex::new(String::new())),
        }
    }

    /// The endpoint and key for one call (see the module docs for the precedence rules).
    fn target(&self, endpoint: Option<String>, api_key: Option<String>) -> Target {
        let stored_endpoint = self
            .endpoint
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        let stored_key = self
            .api_key
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .clone();
        resolve_target(&stored_endpoint, &stored_key, endpoint.as_deref(), api_key.as_deref())
    }
}

/// Where one call goes and with which key.
#[derive(Debug, PartialEq, Eq)]
struct Target {
    endpoint: String,
    api_key: Option<String>,
}

/// Per-call values win over stored ones; a blank endpoint counts as absent, while an explicit empty
/// key means "no key".
fn resolve_target(
    stored_endpoint: &str,
    stored_key: &str,
    call_endpoint: Option<&str>,
    call_key: Option<&str>,
) -> Target {
    Target {
        endpoint: pick_endpoint(call_endpoint, stored_endpoint),
        api_key: match call_key {
            Some(k) => non_empty_key(Some(k)),
            None => non_empty_key(Some(stored_key)),
        },
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Language {
    pub code: String,
    pub name: String,
    /// Target codes this source can translate into (LibreTranslate ≥ 1.3); omitted when unknown.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub targets: Option<Vec<String>>,
}

#[derive(Serialize, Debug)]
struct TranslateRequest<'a> {
    q: &'a str,
    source: &'a str,
    target: &'a str,
    format: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<&'a str>,
}

#[derive(Deserialize, Debug)]
struct TranslateResponse {
    #[serde(rename = "translatedText")]
    translated_text: String,
}

#[derive(Serialize, Debug)]
struct DetectRequest<'a> {
    q: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    api_key: Option<&'a str>,
}

#[derive(Deserialize, Debug)]
struct DetectResponse {
    #[serde(default)]
    confidence: f64,
    language: String,
}

/// The most confident detection (servers usually sort, but do not promise to).
fn best_detection(results: Vec<DetectResponse>) -> Option<String> {
    results
        .into_iter()
        .filter(|r| !r.language.trim().is_empty())
        .max_by(|a, b| a.confidence.total_cmp(&b.confidence))
        .map(|r| r.language)
}

/// Set the LibreTranslate API endpoint URL (the default for calls without `endpoint`).
#[command]
async fn set_endpoint<R: Runtime>(_app: AppHandle<R>, state: State<'_, TranslateState>, endpoint: String) -> Result<(), String> {
    *state.endpoint.lock().unwrap_or_else(PoisonError::into_inner) = endpoint.trim().to_string();
    Ok(())
}

/// Set the optional LibreTranslate API key (empty string clears; the default for calls without `apiKey`).
#[command]
async fn set_api_key<R: Runtime>(_app: AppHandle<R>, state: State<'_, TranslateState>, api_key: String) -> Result<(), String> {
    *state.api_key.lock().unwrap_or_else(PoisonError::into_inner) = api_key;
    Ok(())
}

/// Get available languages from LibreTranslate.
#[command]
async fn get_languages<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, TranslateState>,
    endpoint: Option<String>,
    api_key: Option<String>,
) -> Result<Vec<Language>, String> {
    // `/languages` is public: the key is resolved (for symmetry) but never sent.
    let target = state.target(endpoint, api_key);
    let url = api_url(&target.endpoint, "/languages")?;
    let what = "LibreTranslate language list";

    let response = http_client()
        .get(url)
        .timeout(LIST_TIMEOUT)
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, LIST_TIMEOUT))?;
    if !response.status().is_success() {
        return Err(describe_api_error(what, response).await);
    }
    response
        .json::<Vec<Language>>()
        .await
        .map_err(|e| format!("{what}: unexpected response ({e})"))
}

/// Translate text from source language to target language.
#[command]
async fn translate<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, TranslateState>,
    text: String,
    source: String,
    target: String,
    endpoint: Option<String>,
    api_key: Option<String>,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Ok(String::new());
    }
    let call = state.target(endpoint, api_key);
    let url = api_url(&call.endpoint, "/translate")?;
    let what = "LibreTranslate translation";
    let timeout = translate_timeout(&text);

    let response = http_client()
        .post(url)
        .timeout(timeout)
        .json(&TranslateRequest {
            q: &text,
            source: &source,
            target: &target,
            format: "text",
            api_key: call.api_key.as_deref(),
        })
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, timeout))?;
    if !response.status().is_success() {
        return Err(describe_api_error(what, response).await);
    }
    let result: TranslateResponse = response
        .json()
        .await
        .map_err(|e| format!("{what}: unexpected response ({e})"))?;
    Ok(result.translated_text)
}

/// Detect the language of the input text.
#[command]
async fn detect_language<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, TranslateState>,
    text: String,
    endpoint: Option<String>,
    api_key: Option<String>,
) -> Result<String, String> {
    if text.trim().is_empty() {
        return Err("Cannot detect language of empty text".to_string());
    }
    let call = state.target(endpoint, api_key);
    let url = api_url(&call.endpoint, "/detect")?;
    let what = "LibreTranslate language detection";

    let response = http_client()
        .post(url)
        .timeout(TRANSLATE_TIMEOUT)
        .json(&DetectRequest {
            q: &text,
            api_key: call.api_key.as_deref(),
        })
        .send()
        .await
        .map_err(|e| describe_request_error(what, &e, TRANSLATE_TIMEOUT))?;
    if !response.status().is_success() {
        return Err(describe_api_error(what, response).await);
    }
    let results: Vec<DetectResponse> = response
        .json()
        .await
        .map_err(|e| format!("{what}: unexpected response ({e})"))?;
    best_detection(results).ok_or_else(|| "No language detected".to_string())
}

/// Check if LibreTranslate is reachable (`GET /languages` answers 2xx within 10 s).
#[command]
async fn check_availability<R: Runtime>(
    _app: AppHandle<R>,
    state: State<'_, TranslateState>,
    endpoint: Option<String>,
    api_key: Option<String>,
) -> Result<bool, String> {
    let target = state.target(endpoint, api_key);
    let Ok(url) = api_url(&target.endpoint, "/languages") else {
        return Ok(false);
    };
    Ok(probe_ok("LibreTranslate availability check", http_client().get(url)).await)
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("translate")
        .setup(|app, _api| {
            app.manage(TranslateState::new());
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            set_endpoint,
            set_api_key,
            get_languages,
            translate,
            detect_language,
            check_availability,
            deepl::deepl_translate,
            deepl::deepl_usage,
            deepl::deepl_languages
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_call_values_override_stored_ones() {
        let t = resolve_target("http://stored:5000", "stored-key", Some("http://call:5000"), Some("call-key"));
        assert_eq!(
            t,
            Target {
                endpoint: "http://call:5000".into(),
                api_key: Some("call-key".into())
            }
        );
    }

    #[test]
    fn absent_values_fall_back_to_stored() {
        let t = resolve_target("http://stored:5000", " stored-key ", None, None);
        assert_eq!(t.endpoint, "http://stored:5000");
        assert_eq!(t.api_key.as_deref(), Some("stored-key"));
    }

    #[test]
    fn blank_endpoint_is_absent_but_empty_key_means_no_key() {
        let t = resolve_target("http://stored:5000", "stored-key", Some("   "), Some(""));
        assert_eq!(t.endpoint, "http://stored:5000");
        assert_eq!(t.api_key, None, "an explicit empty key must not leak the stored key");
        let t = resolve_target("http://stored:5000", "   ", None, None);
        assert_eq!(t.api_key, None);
    }

    #[test]
    fn concurrent_callers_do_not_share_endpoints() {
        // The inspector's language probe and the running service each pass their own endpoint;
        // neither mutates the stored default.
        let state = TranslateState::new();
        let a = state.target(Some("http://inspector:5000".into()), Some("".into()));
        let b = state.target(None, None);
        assert_eq!(a.endpoint, "http://inspector:5000");
        assert_eq!(b.endpoint, DEFAULT_ENDPOINT);
    }

    #[test]
    fn key_is_never_part_of_a_url() {
        let t = resolve_target("http://host:5000", "SECRET", None, None);
        for path in ["/languages", "/translate", "/detect"] {
            let url = api_url(&t.endpoint, path).unwrap();
            assert!(!url.as_str().contains("SECRET"), "{url}");
        }
    }

    #[test]
    fn request_bodies_carry_the_key_only_when_set() {
        let with = serde_json::to_value(TranslateRequest {
            q: "hi",
            source: "en",
            target: "ja",
            format: "text",
            api_key: Some("k"),
        })
        .unwrap();
        assert_eq!(with["api_key"], "k");
        assert_eq!(with["format"], "text");
        let without = serde_json::to_value(DetectRequest { q: "hi", api_key: None }).unwrap();
        assert!(without.get("api_key").is_none());
    }

    #[test]
    fn detection_picks_highest_confidence() {
        let r = vec![
            DetectResponse { confidence: 12.0, language: "en".into() },
            DetectResponse { confidence: 88.5, language: "ja".into() },
            DetectResponse { confidence: 99.0, language: " ".into() },
        ];
        assert_eq!(best_detection(r).as_deref(), Some("ja"));
        assert_eq!(best_detection(vec![]), None);
    }

    #[test]
    fn languages_parse_with_and_without_targets() {
        let v: Vec<Language> =
            serde_json::from_str(r#"[{"code":"en","name":"English","targets":["ja","de"]},{"code":"ja","name":"Japanese"}]"#)
                .unwrap();
        assert_eq!(v[0].targets.as_deref(), Some(&["ja".to_string(), "de".to_string()][..]));
        let out = serde_json::to_value(&v[1]).unwrap();
        assert!(out.get("targets").is_none());
    }
}
