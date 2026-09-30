//! DeepL API client (`plugin:translate|deepl_*`): API Free (`api-free.deepl.com`, keys ending in `:fx`) or Pro
//! (`api.deepl.com`), key only in the `Authorization: DeepL-Auth-Key <key>` header (marked sensitive; never in a URL
//! or log), per-request timeout. Runs in Rust because of CORS / CSP in the webview.
//!
//! - `deepl_translate { request }` → `POST /v2/translate` (JSON) with one text, `target_lang`, optional
//!   `source_lang` (absent = DeepL detects it), `formality`, `context` (earlier lines: not translated, not billed)
//!   and `model_type`; `show_billed_characters` is always on.
//! - `deepl_usage { apiKey, plan? }` → `GET /v2/usage` (characters used and allowed this billing period).
//! - `deepl_languages { apiKey, plan?, kind }` → `GET /v2/languages?type=source|target`.
//!
//! Errors are plain strings in the `translate` plugin's style with the HTTP status first, so the TS side can
//! classify them: `"DeepL translation: HTTP 456: quota exceeded ..."`. DeepL's own statuses get a readable reason
//! (403 wrong key or plan, 456 quota used up, 429 / 529 too many requests) plus DeepL's `message` when it sends one.
//!
//! Owned by P4-D (with `translate.rs`).

use std::time::Duration;

use reqwest::header::{HeaderMap, HeaderValue, AUTHORIZATION, CONTENT_TYPE};
use reqwest::{Method, StatusCode, Url};
use serde::{Deserialize, Serialize};
use tauri::command;

use crate::services::http_client::{describe_request_error, error_body_summary, http_client, LIST_TIMEOUT};

/// API Free host.
pub const FREE_BASE: &str = "https://api-free.deepl.com";
/// API Pro host.
pub const PRO_BASE: &str = "https://api.deepl.com";
/// Default per-request timeout of `deepl_translate`.
pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(10);
/// Longest accepted per-request timeout.
const MAX_TIMEOUT: Duration = Duration::from_secs(60);
/// Shortest accepted per-request timeout.
const MIN_TIMEOUT: Duration = Duration::from_millis(1_000);
/// DeepL accepts up to 128 KiB per request; captions are far shorter.
const MAX_TEXT_BYTES: usize = 100 * 1024;
/// Error bodies are read for at most this long.
const ERROR_BODY_TIMEOUT: Duration = Duration::from_secs(3);

/// Which API host to use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DeeplPlan {
    /// Free when the key ends in `:fx`, else Pro.
    #[default]
    Auto,
    Free,
    Pro,
}

/// Argument `request` of `deepl_translate`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DeeplTranslateRequest {
    pub text: String,
    /// DeepL source code (`EN`, `JA`); `None` = DeepL detects it.
    pub source: Option<String>,
    /// DeepL target code (`JA`, `EN-US`, `PT-BR`, `ZH-HANS`).
    pub target: String,
    pub api_key: String,
    pub plan: DeeplPlan,
    /// `default`, `more`, `less`, `prefer_more`, `prefer_less`; `None` = not sent.
    pub formality: Option<String>,
    /// Earlier lines as context (not translated, not billed).
    pub context: Option<String>,
    /// `latency_optimized`, `quality_optimized`, `prefer_quality_optimized`; `None` = not sent.
    pub model_type: Option<String>,
    /// Per-request timeout (default 10 s).
    pub timeout_ms: Option<u64>,
}

/// Reply of `deepl_translate`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DeeplTranslateResponse {
    pub text: String,
    pub detected_source: Option<String>,
    pub billed_characters: Option<u64>,
}

/// Reply of `deepl_usage` (characters this billing period).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DeeplUsage {
    pub character_count: u64,
    pub character_limit: u64,
}

/// Row of `deepl_languages`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DeeplLanguage {
    pub code: String,
    pub name: String,
    pub supports_formality: bool,
}

// ---------------------------------------------------------------------------------------------
// Wire format (DeepL API v2)
// ---------------------------------------------------------------------------------------------

#[derive(Serialize, Debug, PartialEq)]
struct TranslateBody<'a> {
    text: [&'a str; 1],
    target_lang: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    source_lang: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    formality: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    context: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    model_type: Option<&'a str>,
    show_billed_characters: bool,
}

#[derive(Deserialize, Debug)]
struct TranslateReply {
    #[serde(default)]
    translations: Vec<TranslationRow>,
}

#[derive(Deserialize, Debug)]
struct TranslationRow {
    #[serde(default)]
    text: String,
    #[serde(default)]
    detected_source_language: Option<String>,
    #[serde(default)]
    billed_characters: Option<u64>,
}

#[derive(Deserialize, Debug)]
struct UsageReply {
    #[serde(default)]
    character_count: u64,
    #[serde(default)]
    character_limit: u64,
}

#[derive(Deserialize, Debug)]
struct LanguageRow {
    language: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    supports_formality: bool,
}

// ---------------------------------------------------------------------------------------------
// Validation and helpers (pure)
// ---------------------------------------------------------------------------------------------

/// The API base URL for a key and plan: `auto` picks Free for keys ending in `:fx`.
pub fn base_url(api_key: &str, plan: DeeplPlan) -> &'static str {
    match plan {
        DeeplPlan::Free => FREE_BASE,
        DeeplPlan::Pro => PRO_BASE,
        DeeplPlan::Auto if api_key.trim().ends_with(":fx") => FREE_BASE,
        DeeplPlan::Auto => PRO_BASE,
    }
}

/// The `Authorization` header value, marked sensitive (redacted from `Debug` output). A key that cannot be a header
/// (control characters, line breaks) is refused without echoing it.
fn auth_header(api_key: &str) -> Result<HeaderValue, String> {
    let key = api_key.trim();
    if key.is_empty() {
        return Err("DeepL: enter your DeepL API key first".into());
    }
    if key.chars().any(|c| c.is_control() || c.is_whitespace()) {
        return Err("DeepL: the API key contains spaces or control characters; paste it again".into());
    }
    let mut value =
        HeaderValue::from_str(&format!("DeepL-Auth-Key {key}")).map_err(|_| "DeepL: the API key is not valid text".to_string())?;
    value.set_sensitive(true);
    Ok(value)
}

fn headers(api_key: &str) -> Result<HeaderMap, String> {
    let mut h = HeaderMap::new();
    h.insert(AUTHORIZATION, auth_header(api_key)?);
    Ok(h)
}

/// A DeepL language code: letters with an optional `-` region / script (`EN-US`, `ZH-HANS`), upper-cased.
fn language_code(code: &str, what: &str) -> Result<String, String> {
    let c = code.trim().to_ascii_uppercase();
    let ok = !c.is_empty()
        && c.len() <= 12
        && c.split('-').all(|p| !p.is_empty() && p.chars().all(|ch| ch.is_ascii_alphanumeric()));
    if ok {
        Ok(c)
    } else {
        Err(format!("DeepL: invalid {what} language code {:?}", code.trim()))
    }
}

/// An optional enum-like option: trimmed, `None` when empty or `default`, else it must be one of `allowed`.
fn option_value(value: Option<&str>, allowed: &[&str], what: &str) -> Result<Option<String>, String> {
    match value.map(str::trim) {
        None | Some("") | Some("default") => Ok(None),
        Some(v) if allowed.contains(&v) => Ok(Some(v.to_string())),
        Some(v) => Err(format!("DeepL: unknown {what} {v:?}")),
    }
}

fn request_timeout(timeout_ms: Option<u64>) -> Duration {
    timeout_ms
        .map(Duration::from_millis)
        .unwrap_or(DEFAULT_TIMEOUT)
        .clamp(MIN_TIMEOUT, MAX_TIMEOUT)
}

/// A readable reason for DeepL's statuses (their meaning is DeepL-specific: 456 is the quota).
fn status_reason(status: StatusCode) -> &'static str {
    match status.as_u16() {
        400 => "bad request (unsupported language or option)",
        401 | 403 => "the API key was rejected (check the key and whether it is a Free or a Pro key)",
        404 => "not found (check the Free / Pro setting)",
        413 => "the text is too long",
        414 => "the request is too long",
        429 => "too many requests; wait a moment",
        456 => "the monthly character quota is used up",
        500..=502 | 504 => "DeepL had an internal error",
        503 => "DeepL is temporarily unavailable",
        529 => "too many requests; DeepL is busy",
        _ => "request failed",
    }
}

/// `"<what>: HTTP <code>: <reason>[ (<DeepL's message>)]"`.
fn describe_status(what: &str, status: StatusCode, body: &[u8]) -> String {
    let reason = status_reason(status);
    match error_body_summary(body) {
        Some(detail) if !detail.eq_ignore_ascii_case(reason) => format!("{what}: HTTP {}: {reason} ({detail})", status.as_u16()),
        _ => format!("{what}: HTTP {}: {reason}", status.as_u16()),
    }
}

// ---------------------------------------------------------------------------------------------
// HTTP
// ---------------------------------------------------------------------------------------------

async fn send(
    what: &str,
    method: Method,
    url: Url,
    api_key: &str,
    body: Option<serde_json::Value>,
    timeout: Duration,
) -> Result<Vec<u8>, String> {
    let mut req = http_client()
        .request(method, url)
        .headers(headers(api_key)?)
        .timeout(timeout);
    if let Some(body) = body {
        req = req
            .header(CONTENT_TYPE, "application/json")
            .body(serde_json::to_vec(&body).map_err(|e| format!("{what}: {e}"))?);
    }
    let resp = req.send().await.map_err(|e| describe_request_error(what, &e, timeout))?;
    let status = resp.status();
    if !status.is_success() {
        let body = tokio::time::timeout(ERROR_BODY_TIMEOUT, resp.bytes())
            .await
            .ok()
            .and_then(Result::ok)
            .map(|b| b.to_vec())
            .unwrap_or_default();
        return Err(describe_status(what, status, &body));
    }
    resp.bytes()
        .await
        .map(|b| b.to_vec())
        .map_err(|e| describe_request_error(what, &e, timeout))
}

fn endpoint(base: &str, path: &str) -> Result<Url, String> {
    Url::parse(base)
        .and_then(|u| u.join(path))
        .map_err(|e| format!("DeepL: bad URL ({e})"))
}

/// `deepl_translate` against `base` (tests point it at a local server).
pub(crate) async fn translate_at(base: &str, request: &DeeplTranslateRequest) -> Result<DeeplTranslateResponse, String> {
    const WHAT: &str = "DeepL translation";
    if request.text.trim().is_empty() {
        return Ok(DeeplTranslateResponse::default());
    }
    if request.text.len() > MAX_TEXT_BYTES {
        return Err(format!("{WHAT}: the text is too long ({} bytes)", request.text.len()));
    }
    let target = language_code(&request.target, "target")?;
    let source = match request.source.as_deref().map(str::trim) {
        None | Some("") => None,
        Some(s) => Some(language_code(s, "source")?),
    };
    let formality = option_value(
        request.formality.as_deref(),
        &["more", "less", "prefer_more", "prefer_less"],
        "formality",
    )?;
    let model_type = option_value(
        request.model_type.as_deref(),
        &["latency_optimized", "quality_optimized", "prefer_quality_optimized"],
        "model type",
    )?;
    let context = request.context.as_deref().map(str::trim).filter(|c| !c.is_empty());
    let body = TranslateBody {
        text: [request.text.as_str()],
        target_lang: &target,
        source_lang: source.as_deref(),
        formality: formality.as_deref(),
        context,
        model_type: model_type.as_deref(),
        show_billed_characters: true,
    };
    let body = serde_json::to_value(&body).map_err(|e| format!("{WHAT}: {e}"))?;
    let timeout = request_timeout(request.timeout_ms);
    let bytes = send(WHAT, Method::POST, endpoint(base, "/v2/translate")?, &request.api_key, Some(body), timeout).await?;
    parse_translate_reply(&bytes)
}

fn parse_translate_reply(bytes: &[u8]) -> Result<DeeplTranslateResponse, String> {
    let reply: TranslateReply =
        serde_json::from_slice(bytes).map_err(|e| format!("DeepL translation: unexpected response ({e})"))?;
    let row = reply
        .translations
        .into_iter()
        .next()
        .ok_or("DeepL translation: the response has no translation")?;
    Ok(DeeplTranslateResponse {
        text: row.text,
        detected_source: row.detected_source_language.filter(|s| !s.trim().is_empty()),
        billed_characters: row.billed_characters,
    })
}

pub(crate) async fn usage_at(base: &str, api_key: &str) -> Result<DeeplUsage, String> {
    const WHAT: &str = "DeepL usage";
    let bytes = send(WHAT, Method::GET, endpoint(base, "/v2/usage")?, api_key, None, LIST_TIMEOUT).await?;
    let reply: UsageReply = serde_json::from_slice(&bytes).map_err(|e| format!("{WHAT}: unexpected response ({e})"))?;
    Ok(DeeplUsage {
        character_count: reply.character_count,
        character_limit: reply.character_limit,
    })
}

pub(crate) async fn languages_at(base: &str, api_key: &str, kind: &str) -> Result<Vec<DeeplLanguage>, String> {
    const WHAT: &str = "DeepL language list";
    let kind = match kind.trim() {
        "source" => "source",
        "target" => "target",
        other => return Err(format!("{WHAT}: kind must be source or target, not {other:?}")),
    };
    let mut url = endpoint(base, "/v2/languages")?;
    url.query_pairs_mut().append_pair("type", kind);
    let bytes = send(WHAT, Method::GET, url, api_key, None, LIST_TIMEOUT).await?;
    let rows: Vec<LanguageRow> = serde_json::from_slice(&bytes).map_err(|e| format!("{WHAT}: unexpected response ({e})"))?;
    Ok(rows
        .into_iter()
        .filter(|r| !r.language.trim().is_empty())
        .map(|r| DeeplLanguage {
            code: r.language.trim().to_ascii_uppercase(),
            name: r.name,
            supports_formality: r.supports_formality,
        })
        .collect())
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

#[command]
pub async fn deepl_translate(request: DeeplTranslateRequest) -> Result<DeeplTranslateResponse, String> {
    translate_at(base_url(&request.api_key, request.plan), &request).await
}

#[command]
pub async fn deepl_usage(api_key: String, plan: Option<DeeplPlan>) -> Result<DeeplUsage, String> {
    usage_at(base_url(&api_key, plan.unwrap_or_default()), &api_key).await
}

/// `kind`: `source` or `target`.
#[command]
pub async fn deepl_languages(api_key: String, plan: Option<DeeplPlan>, kind: String) -> Result<Vec<DeeplLanguage>, String> {
    languages_at(base_url(&api_key, plan.unwrap_or_default()), &api_key, &kind).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    #[test]
    fn request_wire_format() {
        let req: DeeplTranslateRequest = serde_json::from_value(serde_json::json!({
            "text": "hello",
            "target": "JA",
            "apiKey": "k:fx",
            "plan": "free",
            "modelType": "latency_optimized",
        }))
        .unwrap();
        assert_eq!(req.plan, DeeplPlan::Free);
        assert!(req.source.is_none());
        assert_eq!(req.model_type.as_deref(), Some("latency_optimized"));
        let usage = serde_json::to_value(DeeplUsage {
            character_count: 1,
            character_limit: 500_000,
        })
        .unwrap();
        assert_eq!(usage["characterLimit"], 500_000);
        let resp = serde_json::to_value(DeeplTranslateResponse {
            text: "x".into(),
            detected_source: Some("EN".into()),
            billed_characters: Some(5),
        })
        .unwrap();
        assert_eq!(resp["detectedSource"], "EN");
        assert_eq!(resp["billedCharacters"], 5);
    }

    #[test]
    fn plans_pick_the_host() {
        assert_eq!(base_url("abc:fx", DeeplPlan::Auto), FREE_BASE);
        assert_eq!(base_url(" abc:fx ", DeeplPlan::Auto), FREE_BASE);
        assert_eq!(base_url("abc", DeeplPlan::Auto), PRO_BASE);
        assert_eq!(base_url("abc", DeeplPlan::Free), FREE_BASE);
        assert_eq!(base_url("abc:fx", DeeplPlan::Pro), PRO_BASE);
    }

    #[test]
    fn the_key_goes_only_into_a_sensitive_header() {
        let h = auth_header(" secret-key:fx ").unwrap();
        assert!(h.is_sensitive());
        assert_eq!(h.to_str().unwrap(), "DeepL-Auth-Key secret-key:fx");
        assert!(!format!("{h:?}").contains("secret"), "Debug output is redacted");
        for bad in ["", "   ", "a\nb", "a b", "key\r\nX-Evil: 1"] {
            let e = auth_header(bad).unwrap_err();
            assert!(!e.contains("Evil") && !e.contains("key\r"), "{e}");
        }
        // The URLs never carry it.
        for path in ["/v2/translate", "/v2/usage", "/v2/languages"] {
            assert!(!endpoint(FREE_BASE, path).unwrap().as_str().contains("secret"));
        }
    }

    #[test]
    fn body_has_only_the_set_options() {
        let body = TranslateBody {
            text: ["Hi"],
            target_lang: "JA",
            source_lang: None,
            formality: None,
            context: None,
            model_type: None,
            show_billed_characters: true,
        };
        let v = serde_json::to_value(&body).unwrap();
        assert_eq!(v, serde_json::json!({"text": ["Hi"], "target_lang": "JA", "show_billed_characters": true}));
        assert_eq!(language_code(" en-us ", "target").unwrap(), "EN-US");
        assert_eq!(language_code("zh-hans", "target").unwrap(), "ZH-HANS");
        for bad in ["", "EN US", "en/us", "e-", "-x", "toolongcode-xx"] {
            assert!(language_code(bad, "target").is_err(), "{bad}");
        }
        assert_eq!(option_value(Some("default"), &["more"], "f").unwrap(), None);
        assert_eq!(option_value(Some(" more "), &["more"], "f").unwrap().as_deref(), Some("more"));
        assert_eq!(option_value(None, &["more"], "f").unwrap(), None);
        assert!(option_value(Some("most"), &["more"], "f").is_err());
        assert_eq!(request_timeout(None), DEFAULT_TIMEOUT);
        assert_eq!(request_timeout(Some(5)), MIN_TIMEOUT);
        assert_eq!(request_timeout(Some(3_600_000)), MAX_TIMEOUT);
        assert_eq!(request_timeout(Some(2_500)), Duration::from_millis(2_500));
    }

    #[test]
    fn replies_and_errors_parse() {
        let r = parse_translate_reply(
            br#"{"translations":[{"detected_source_language":"EN","text":"Hallo","billed_characters":5,"model_type_used":"quality_optimized"}]}"#,
        )
        .unwrap();
        assert_eq!(
            r,
            DeeplTranslateResponse {
                text: "Hallo".into(),
                detected_source: Some("EN".into()),
                billed_characters: Some(5)
            }
        );
        assert!(parse_translate_reply(br#"{"translations":[]}"#).is_err());
        assert!(parse_translate_reply(b"<html>").unwrap_err().contains("unexpected response"));
        let e = describe_status("DeepL translation", StatusCode::from_u16(456).unwrap(), br#"{"message":"Quota exceeded"}"#);
        assert_eq!(e, "DeepL translation: HTTP 456: the monthly character quota is used up (Quota exceeded)");
        let e = describe_status("DeepL usage", StatusCode::FORBIDDEN, b"");
        assert!(e.starts_with("DeepL usage: HTTP 403: the API key was rejected"), "{e}");
        let e = describe_status("DeepL translation", StatusCode::from_u16(529).unwrap(), b"<html><title>Busy</title></html>");
        assert_eq!(e, "DeepL translation: HTTP 529: too many requests; DeepL is busy (Busy)");
    }

    /// A one-request HTTP server on 127.0.0.1 that records the request and answers `status` + `body`.
    async fn serve_once(status: u16, body: &'static str, delay: Duration) -> (String, Arc<Mutex<String>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(String::new()));
        let record = seen.clone();
        tokio::spawn(async move {
            let (mut sock, _) = listener.accept().await.unwrap();
            let mut buf = Vec::new();
            let mut chunk = [0u8; 4096];
            // Read the head, then Content-Length bytes of body.
            loop {
                let n = sock.read(&mut chunk).await.unwrap();
                if n == 0 {
                    break;
                }
                buf.extend_from_slice(&chunk[..n]);
                let text = String::from_utf8_lossy(&buf).to_string();
                if let Some(head_end) = text.find("\r\n\r\n") {
                    let len = text[..head_end]
                        .lines()
                        .find_map(|l| l.to_ascii_lowercase().strip_prefix("content-length:").map(|v| v.trim().parse::<usize>().unwrap()))
                        .unwrap_or(0);
                    if buf.len() >= head_end + 4 + len {
                        break;
                    }
                }
            }
            *record.lock().unwrap() = String::from_utf8_lossy(&buf).to_string();
            tokio::time::sleep(delay).await;
            let reply = format!(
                "HTTP/1.1 {status} X\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{body}",
                body.len()
            );
            let _ = sock.write_all(reply.as_bytes()).await;
        });
        (base, seen)
    }

    fn request() -> DeeplTranslateRequest {
        DeeplTranslateRequest {
            text: "Hello, world".into(),
            source: Some("en".into()),
            target: "de".into(),
            api_key: "test-key:fx".into(),
            plan: DeeplPlan::Auto,
            formality: Some("prefer_more".into()),
            context: Some("Earlier line.".into()),
            model_type: Some("default".into()),
            timeout_ms: None,
        }
    }

    #[tokio::test]
    async fn translate_sends_the_key_as_a_header_and_parses_the_reply() {
        let (base, seen) = serve_once(
            200,
            r#"{"translations":[{"detected_source_language":"EN","text":"Hallo, Welt","billed_characters":12}]}"#,
            Duration::ZERO,
        )
        .await;
        let r = translate_at(&base, &request()).await.unwrap();
        assert_eq!(r.text, "Hallo, Welt");
        assert_eq!(r.detected_source.as_deref(), Some("EN"));
        assert_eq!(r.billed_characters, Some(12));
        let raw = seen.lock().unwrap().clone();
        let (head, body) = raw.split_once("\r\n\r\n").unwrap();
        assert!(head.starts_with("POST /v2/translate HTTP/1.1"), "{head}");
        assert!(head.to_ascii_lowercase().contains("authorization: deepl-auth-key test-key:fx"), "{head}");
        assert!(!head.lines().next().unwrap().contains("test-key"), "no key in the request line");
        let body: serde_json::Value = serde_json::from_str(body).unwrap();
        assert_eq!(
            body,
            serde_json::json!({
                "text": ["Hello, world"],
                "target_lang": "DE",
                "source_lang": "EN",
                "formality": "prefer_more",
                "context": "Earlier line.",
                "show_billed_characters": true
            })
        );
    }

    #[tokio::test]
    async fn statuses_become_classifiable_errors() {
        let (base, _) = serve_once(456, r#"{"message":"Quota Exceeded"}"#, Duration::ZERO).await;
        let e = translate_at(&base, &request()).await.unwrap_err();
        assert!(e.starts_with("DeepL translation: HTTP 456: "), "{e}");
        let (base, _) = serve_once(403, r#"{"message":"Wrong endpoint"}"#, Duration::ZERO).await;
        let e = usage_at(&base, "k").await.unwrap_err();
        assert!(e.starts_with("DeepL usage: HTTP 403: "), "{e}");
        assert!(!e.contains("DeepL-Auth-Key"));
    }

    #[tokio::test]
    async fn slow_servers_time_out_per_request() {
        let (base, _) = serve_once(200, r#"{"translations":[{"text":"late"}]}"#, Duration::from_secs(5)).await;
        let mut r = request();
        r.timeout_ms = Some(1_000);
        let started = std::time::Instant::now();
        let e = translate_at(&base, &r).await.unwrap_err();
        assert!(e.contains("timed out"), "{e}");
        assert!(started.elapsed() < Duration::from_secs(4));
    }

    #[tokio::test]
    async fn usage_and_languages_parse() {
        let (base, seen) = serve_once(200, r#"{"character_count":180118,"character_limit":500000}"#, Duration::ZERO).await;
        let u = usage_at(&base, "k:fx").await.unwrap();
        assert_eq!((u.character_count, u.character_limit), (180_118, 500_000));
        assert!(seen.lock().unwrap().starts_with("GET /v2/usage HTTP/1.1"));

        let (base, seen) = serve_once(
            200,
            r#"[{"language":"DE","name":"German","supports_formality":true},{"language":"EN-US","name":"English (American)","supports_formality":false},{"language":" ","name":"?"}]"#,
            Duration::ZERO,
        )
        .await;
        let langs = languages_at(&base, "k:fx", "target").await.unwrap();
        assert_eq!(langs.len(), 2);
        assert_eq!(langs[0], DeeplLanguage { code: "DE".into(), name: "German".into(), supports_formality: true });
        assert!(seen.lock().unwrap().starts_with("GET /v2/languages?type=target HTTP/1.1"));
        assert!(languages_at(&base, "k", "both").await.is_err());
    }

    #[tokio::test]
    async fn bad_input_fails_before_any_request() {
        let mut r = request();
        r.api_key = String::new();
        assert!(translate_at("http://127.0.0.1:9", &r).await.unwrap_err().contains("API key"));
        let mut r = request();
        r.target = "de de".into();
        assert!(translate_at("http://127.0.0.1:9", &r).await.unwrap_err().contains("target"));
        let mut r = request();
        r.formality = Some("very".into());
        assert!(translate_at("http://127.0.0.1:9", &r).await.is_err());
        let mut r = request();
        r.text = "   ".into();
        assert_eq!(translate_at("http://127.0.0.1:9", &r).await.unwrap(), DeeplTranslateResponse::default());
    }

    /// Live check with a real key: `SIGIL_DEEPL_KEY=<key> cargo test deepl_live -- --ignored`.
    #[tokio::test]
    #[ignore = "needs SIGIL_DEEPL_KEY (a DeepL API key) and the network"]
    async fn deepl_live() {
        let key = std::env::var("SIGIL_DEEPL_KEY").expect("SIGIL_DEEPL_KEY");
        let usage = deepl_usage(key.clone(), None).await.unwrap();
        eprintln!("usage {}/{}", usage.character_count, usage.character_limit);
        let targets = deepl_languages(key.clone(), None, "target".into()).await.unwrap();
        assert!(targets.iter().any(|l| l.code == "JA"));
        let r = deepl_translate(DeeplTranslateRequest {
            text: "Good morning, everyone!".into(),
            target: "JA".into(),
            api_key: key,
            ..Default::default()
        })
        .await
        .unwrap();
        eprintln!("{r:?}");
        assert!(!r.text.is_empty());
        assert_eq!(r.detected_source.as_deref(), Some("EN"));
    }
}
