//! `gpt-sovits` plugin: a client for GPT-SoVITS's `api_v2.py` server (v2, v2Pro / v2ProPlus, v3,
//! v4), streaming each sentence into the TTS sink (P6-G).
//!
//! The user runs GPT-SoVITS with `python api_v2.py -a 127.0.0.1 -p 9880` (the integrated package's
//! `runtime\python.exe api_v2.py`). Sigil:
//! - **speaks** with `POST {baseUrl}/tts` (`text`, `text_lang`, `ref_audio_path`, `prompt_text`,
//!   `prompt_lang`, `media_type: "wav"`, `streaming_mode`, `speed_factor`, then the user's `extra`
//!   members), decoding the answer as it arrives (`openai_speech::speak_http`): with streaming the
//!   server sends a WAV header without a size, then raw 16-bit chunks, so the first audio plays
//!   while the rest is generated;
//! - **references** a voice-library voice (a copy of its `ref.wav` named after the clip's content,
//!   [`versioned_clip`], and its transcript, for a server on this PC: the server reads the clip from
//!   disk) or a clip path on the server's machine;
//! - **switches weights** with `GET /set_gpt_weights?weights_path=` and `/set_sovits_weights`
//!   (paths on the server's machine; the server keeps them until changed);
//! - **checks** the server through FastAPI's `/openapi.json` (it must list `/tts`, which tells
//!   `api_v2.py` from the older `api.py`).
//!
//! `streaming_mode`: off sends `false`; on sends `true` (every api_v2 version: whole fragments,
//! best quality); the v2ProPlus `api_v2.py` also takes 2 (medium quality, sooner) and 3 (fixed-length
//! chunks, soonest), sent only when chosen ([`GptSovitsRequest::streaming_mode`]).
//!
//! # Frozen contract (P6-S0; change only through INTEGRATE; structs may gain fields)
//! - IPC (`plugin-commands/gpt-sovits.txt`; wire camelCase; TS mirror `RustGptSovits*` in
//!   `src/types.ts`):
//!   - `gpt_sovits_speak { request: GptSovitsRequest, utteranceId } -> SpeakReport`.
//!   - `gpt_sovits_cancel { utteranceId? } -> ()`.
//!   - `gpt_sovits_check { baseUrl } -> bool`: the server answers (`Err` when it answers but is not
//!     `api_v2.py`, so the UI can say why).
//!   - `gpt_sovits_set_weights { baseUrl, gptWeights, sovitsWeights } -> ()`: switch the model
//!     (server-side paths; "" leaves that half unchanged).

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use reqwest::{Client, StatusCode, Url};
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use crate::services::http_client::{
    api_url, content_type, describe_request_error, error_body_summary, http_client, local_http_client, normalize_base_url, redact_url,
};
use crate::services::openai_speech::{speak_http, HttpSpeech, SpeechFormat};
use crate::services::tts_sink::SpeakReport;
use crate::services::voice_library;

/// Plugin name (`plugin:gpt-sovits|...`, capability `gpt-sovits:default`).
pub const PLUGIN_NAME: &str = "gpt-sovits";
/// `api_v2.py`'s default address.
pub const DEFAULT_BASE_URL: &str = "http://127.0.0.1:9880";
/// Speaking-rate range sent as `speed_factor`.
const SPEED_RANGE: (f32, f32) = (0.5, 2.0);
/// How long switching weights may take (loading a v4 SoVITS model takes tens of seconds).
const WEIGHTS_TIMEOUT: Duration = Duration::from_secs(300);
/// How long the server check may take.
const CHECK_TIMEOUT: Duration = Duration::from_secs(10);
/// Bytes of an error body read.
const ERROR_BODY_LIMIT: usize = 8 * 1024;
/// Copies of voice-library clips sent to a server on this PC (under app data; see [`versioned_clip`]).
const REFS_DIR: &str = "tts/gpt-sovits/refs";
/// Hex digits of the clip hash in a copy's name.
const REF_HASH_LEN: usize = 12;
/// Makes every temporary file of [`versioned_clip`] unique.
static REF_TMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// One sentence to speak (TS `RustGptSovitsRequest`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct GptSovitsRequest {
    /// Server base URL ("" = [`DEFAULT_BASE_URL`]).
    pub base_url: String,
    pub text: String,
    /// `text_lang` (`ja`, `zh`, `en`, `ko`, `yue`, `auto`, ...).
    pub text_lang: String,
    /// Voice-library voice whose clip and transcript are the reference ("" = use `ref_audio_path`).
    pub reference_voice_id: String,
    /// Reference clip path on the server's machine (when no library voice is used).
    pub ref_audio_path: String,
    /// Transcript of the reference clip ("" = the library voice's transcript).
    pub prompt_text: String,
    /// Language of the reference transcript.
    pub prompt_lang: String,
    /// Ask the server to stream (`streaming_mode`).
    pub streaming: bool,
    /// Speaking rate (`speed_factor`), 1 = normal.
    pub speed: f32,
    /// Extra JSON members merged into the request (`top_k`, `temperature`, ...); `null` or an object.
    pub extra: Value,
    /// Time to the first audio (ms); 0 = automatic.
    pub timeout_ms: u32,
    /// P6-G: the v2ProPlus `api_v2.py` streaming mode when `streaming` (1 = fragments, best quality,
    /// sent as `true` for older servers; 2 = medium; 3 = fixed-length chunks, soonest); 0 = 1.
    pub streaming_mode: u8,
}

impl Default for GptSovitsRequest {
    fn default() -> Self {
        Self {
            base_url: String::new(),
            text: String::new(),
            text_lang: "auto".into(),
            reference_voice_id: String::new(),
            ref_audio_path: String::new(),
            prompt_text: String::new(),
            prompt_lang: "auto".into(),
            streaming: true,
            speed: 1.0,
            extra: Value::Null,
            timeout_ms: 0,
            streaming_mode: 0,
        }
    }
}

/// The server base for `raw` ("" = [`DEFAULT_BASE_URL`]), validated, without trailing slashes.
fn base_of(raw: &str) -> Result<String, String> {
    let raw = raw.trim();
    normalize_base_url(if raw.is_empty() { DEFAULT_BASE_URL } else { raw })
}

/// Whether `base` is a server on this PC (it can read the voice library's files).
fn is_this_pc(base: &str) -> bool {
    let Ok(url) = Url::parse(base) else { return false };
    let Some(host) = url.host_str() else { return false };
    let bare = host.trim_start_matches('[').trim_end_matches(']');
    match bare.parse::<std::net::IpAddr>() {
        Ok(ip) => ip.is_loopback(),
        Err(_) => {
            let d = bare.trim_end_matches('.').to_ascii_lowercase();
            d == "localhost" || d.ends_with(".localhost")
        }
    }
}

/// GPT-SoVITS's language code for a language setting or a BCP-47 tag (`ja-JP` → `ja`;
/// `yue`, `zh-HK`, `zh-MO`, `zh-yue` → `yue`; any other `zh-*`, Traditional `zh-Hant` / `zh-TW`
/// included, → `zh`: Traditional script is not Cantonese); its own codes (`all_ja`, `auto_yue`, ...)
/// pass through; blank → `auto`. Same rule as TS `gsvLanguageOf`.
pub fn gsv_language(lang: &str) -> String {
    let l = lang.trim().to_ascii_lowercase().replace('_', "-");
    if l.is_empty() {
        return "auto".into();
    }
    // GPT-SoVITS's own codes (all_zh, auto_yue, ...).
    let raw = lang.trim().to_ascii_lowercase();
    if raw.starts_with("all_") || raw.starts_with("auto") {
        return raw;
    }
    let base = l.split('-').next().unwrap_or("");
    match base {
        "zh" if l.split('-').skip(1).any(|t| matches!(t, "hk" | "mo" | "yue")) => "yue".into(),
        "yue" => "yue".into(),
        "zh" | "cmn" => "zh".into(),
        "ja" | "jp" => "ja".into(),
        "en" => "en".into(),
        "ko" | "kr" => "ko".into(),
        other => other.into(),
    }
}

/// The voice id of a reference copy named `<voiceId>-<hash>.wav` ([`versioned_clip`]).
fn copy_owner(name: &str) -> Option<&str> {
    let stem = name.strip_suffix(".wav")?;
    let (id, hash) = stem.rsplit_once('-')?;
    let hex = hash.len() == REF_HASH_LEN && hash.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b));
    (hex && voice_library::valid_voice_id(id)).then_some(id)
}

/// The clip of library voice `voice_id` as a file whose path changes whenever the clip does:
/// `app_data/tts/gpt-sovits/refs/<voiceId>-<first 12 hex of the clip's sha256>.wav`.
///
/// GPT-SoVITS caches the reference by path (`TTS.run` re-reads the clip only when `ref_audio_path`
/// differs from the cached one) but re-reads `prompt_text` whenever that changes. The voice library
/// replaces a re-recorded clip in place (same id, same `ref.wav`), so sending `ref.wav` itself would
/// pair the new transcript with the old clip's audio prompt until the server restarts. The copy is
/// written once per clip version (atomically); older copies of the voice, and copies of voices no
/// longer in the library, are removed (best effort: the server may still hold one open).
fn versioned_clip(app_data: &Path, voice_id: &str, clip: &Path) -> Result<PathBuf, String> {
    let bytes = std::fs::read(clip).map_err(|e| format!("cannot read the reference clip {}: {e}", clip.display()))?;
    let hash = hex::encode(Sha256::digest(&bytes));
    let dir = app_data.join(REFS_DIR);
    let name = format!("{voice_id}-{}.wav", &hash[..REF_HASH_LEN]);
    let target = dir.join(&name);
    let len = bytes.len() as u64;
    let present = |p: &Path| std::fs::metadata(p).is_ok_and(|m| m.is_file() && m.len() == len);
    if !present(&target) {
        std::fs::create_dir_all(&dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        let tmp = dir.join(format!(".{name}.{}-{}.tmp", std::process::id(), REF_TMP_SEQ.fetch_add(1, Ordering::Relaxed)));
        let written = std::fs::write(&tmp, &bytes).and_then(|()| std::fs::rename(&tmp, &target));
        if let Err(e) = written {
            let _ = std::fs::remove_file(&tmp);
            // Another sentence may have written the same copy meanwhile.
            if !present(&target) {
                return Err(format!("cannot copy the reference clip to {}: {e}", target.display()));
            }
        }
    }
    for e in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
        let file = e.file_name().to_string_lossy().into_owned();
        let Some(owner) = copy_owner(&file) else { continue };
        let stale = if owner == voice_id { file != name } else { voice_library::voice_dir(app_data, owner).is_ok_and(|d| !d.is_dir()) };
        if stale {
            let _ = std::fs::remove_file(e.path());
        }
    }
    Ok(target)
}

/// The reference clip and its transcript for `r`.
#[derive(Clone, Debug, PartialEq)]
struct Reference {
    path: String,
    prompt_text: String,
    prompt_lang: String,
}

/// Resolve the reference of `r` against the voice library in `app_data` (see the module docs).
fn reference(r: &GptSovitsRequest, base: &str, app_data: Option<&Path>) -> Result<Reference, String> {
    let voice_id = r.reference_voice_id.trim();
    let explicit_lang = !r.prompt_lang.trim().is_empty() && !r.prompt_lang.trim().eq_ignore_ascii_case("auto");
    if !voice_id.is_empty() {
        if !is_this_pc(base) {
            return Err(
                "A voice-library reference works only with a GPT-SoVITS server on this PC. For a server on another machine, enter the path of the reference clip on that machine instead."
                    .into(),
            );
        }
        let app_data = app_data.ok_or("the voice library is not available (no app data folder)")?;
        let meta = voice_library::store::get(app_data, voice_id).map_err(|e| format!("reference voice: {e}"))?;
        let clip: PathBuf = voice_library::voice_dir(app_data, voice_id)?.join(voice_library::CLIP_FILE);
        if !clip.is_file() {
            return Err(format!("the reference voice {:?} has no clip", meta.name));
        }
        let prompt_text = Some(r.prompt_text.trim()).filter(|t| !t.is_empty()).map(str::to_string).or(meta.transcript.clone()).unwrap_or_default();
        let prompt_lang = if explicit_lang {
            gsv_language(&r.prompt_lang)
        } else {
            meta.language.as_deref().map(gsv_language).unwrap_or_else(|| "auto".into())
        };
        let path = versioned_clip(app_data, voice_id, &clip)?;
        return Ok(Reference { path: path.to_string_lossy().into_owned(), prompt_text, prompt_lang });
    }
    let path = r.ref_audio_path.trim();
    if path.is_empty() {
        return Err("Choose a reference voice from the voice library, or enter the path of a reference clip on the GPT-SoVITS server.".into());
    }
    if is_this_pc(base) && Path::new(path).is_absolute() && !Path::new(path).is_file() {
        return Err(format!("the reference clip {path} does not exist on this PC"));
    }
    Ok(Reference {
        path: path.to_string(),
        prompt_text: r.prompt_text.trim().to_string(),
        prompt_lang: if explicit_lang { gsv_language(&r.prompt_lang) } else { "auto".into() },
    })
}

/// The `/tts` body of `r` with reference `rf` (see the module docs).
fn tts_body(r: &GptSovitsRequest, rf: &Reference) -> Result<Value, String> {
    let mut body = Map::new();
    body.insert("text".into(), Value::from(r.text.as_str()));
    body.insert("text_lang".into(), Value::from(gsv_language(&r.text_lang)));
    body.insert("ref_audio_path".into(), Value::from(rf.path.as_str()));
    body.insert("prompt_text".into(), Value::from(rf.prompt_text.as_str()));
    body.insert("prompt_lang".into(), Value::from(rf.prompt_lang.as_str()));
    body.insert("media_type".into(), Value::from("wav"));
    let streaming = match (r.streaming, r.streaming_mode) {
        (false, _) => Value::Bool(false),
        (true, 2 | 3) => Value::from(r.streaming_mode),
        (true, _) => Value::Bool(true),
    };
    body.insert("streaming_mode".into(), streaming);
    if r.speed.is_finite() && r.speed > 0.0 {
        let s = r.speed.clamp(SPEED_RANGE.0, SPEED_RANGE.1);
        if (s - 1.0).abs() > 1e-3 {
            body.insert("speed_factor".into(), Value::from(s));
        }
    }
    match &r.extra {
        Value::Null => {}
        Value::Object(extra) => {
            for (k, v) in extra {
                if k != "text" {
                    body.insert(k.clone(), v.clone());
                }
            }
        }
        _ => return Err("Extra request fields must be a JSON object.".into()),
    }
    Ok(Value::Object(body))
}

/// The streaming request of one sentence.
fn speak_spec(r: &GptSovitsRequest, app_data: Option<&Path>) -> Result<HttpSpeech, String> {
    let base = base_of(&r.base_url)?;
    let rf = reference(r, &base, app_data)?;
    let url = api_url(&base, "/tts")?;
    let body = tts_body(r, &rf)?;
    Ok(HttpSpeech::new("GPT-SoVITS synthesis", url, body, &r.text, SpeechFormat::Wav).timeout_ms(r.timeout_ms))
}

/// The message of an api_v2 error body: `{"message": "...", "Exception": "..."}` (both shown), or
/// any other error text.
fn api_v2_error(body: &[u8]) -> Option<String> {
    if let Ok(Value::Object(m)) = serde_json::from_slice::<Value>(body) {
        let message = m.get("message").and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty());
        let exception = m.get("Exception").or_else(|| m.get("exception")).and_then(Value::as_str).map(str::trim).filter(|s| !s.is_empty());
        match (message, exception) {
            (Some(msg), Some(ex)) => return Some(format!("{msg}: {}", crate::services::http_client::truncate_chars(ex, 300))),
            (Some(msg), None) => return Some(msg.to_string()),
            (None, Some(ex)) => return Some(crate::services::http_client::truncate_chars(ex, 300)),
            _ => {}
        }
    }
    error_body_summary(body)
}

async fn read_prefix(mut resp: reqwest::Response) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(Duration::from_secs(5), async {
        while buf.len() < ERROR_BODY_LIMIT {
            match resp.chunk().await {
                Ok(Some(c)) => buf.extend_from_slice(&c[..c.len().min(ERROR_BODY_LIMIT - buf.len())]),
                _ => break,
            }
        }
    })
    .await;
    buf
}

/// Whether the server at `base_url` answers as `api_v2.py` (see the module docs).
async fn check_with(client: &Client, base_url: &str) -> Result<bool, String> {
    let base = base_of(base_url)?;
    let url = api_url(&base, "/openapi.json")?;
    let resp = match client.get(url).timeout(CHECK_TIMEOUT).send().await {
        Ok(r) => r,
        Err(e) => {
            tracing::info!(target: "sigil::http", "{}", describe_request_error("GPT-SoVITS server check", &e, CHECK_TIMEOUT));
            return Ok(false);
        }
    };
    let status = resp.status();
    if status.is_server_error() {
        tracing::info!(target: "sigil::http", "GPT-SoVITS server check: HTTP {status}");
        return Ok(false);
    }
    if !status.is_success() {
        // FastAPI's schema can be turned off: the server still answered.
        return Ok(true);
    }
    let is_json = content_type(&resp).is_some_and(|c| c.to_ascii_lowercase().contains("json"));
    let body = read_prefix_unbounded(resp).await;
    let paths = if is_json { serde_json::from_slice::<Value>(&body).ok().and_then(|v| v.get("paths").cloned()) } else { None };
    match paths {
        Some(Value::Object(p)) if p.contains_key("/tts") => Ok(true),
        Some(Value::Object(_)) => Err(format!(
            "The server at {} answers, but it is not GPT-SoVITS api_v2 (it has no /tts). Start api_v2.py, for example: python api_v2.py -a 127.0.0.1 -p 9880",
            redact_url(&Url::parse(&base).map_err(|e| e.to_string())?)
        )),
        _ => Ok(true),
    }
}

/// The whole (small) body of a schema answer, at most 4 MB.
async fn read_prefix_unbounded(mut resp: reqwest::Response) -> Vec<u8> {
    const LIMIT: usize = 4 * 1024 * 1024;
    let mut buf = Vec::new();
    let _ = tokio::time::timeout(CHECK_TIMEOUT, async {
        while buf.len() < LIMIT {
            match resp.chunk().await {
                Ok(Some(c)) => buf.extend_from_slice(&c[..c.len().min(LIMIT - buf.len())]),
                _ => break,
            }
        }
    })
    .await;
    buf
}

/// `GET {base}/{route}?weights_path=<path>`; `Err` with the server's message.
async fn set_one(client: &Client, base: &str, route: &str, path: &str, what: &str) -> Result<(), String> {
    let mut url = api_url(base, route)?;
    url.query_pairs_mut().append_pair("weights_path", path);
    let resp = client
        .get(url)
        .timeout(WEIGHTS_TIMEOUT)
        .send()
        .await
        .map_err(|e| describe_request_error(&format!("Switching the {what} weights"), &e, WEIGHTS_TIMEOUT))?;
    let status = resp.status();
    if status.is_success() {
        return Ok(());
    }
    let body = read_prefix(resp).await;
    let why = api_v2_error(&body).unwrap_or_default();
    if status == StatusCode::NOT_FOUND && why.to_ascii_lowercase().contains("not found") {
        return Err(format!("The server has no {route} (is it GPT-SoVITS api_v2?)."));
    }
    Err(format!("Switching the {what} weights failed (HTTP {status}){}{why}", if why.is_empty() { "" } else { ": " }))
}

/// Switch the GPT and / or SoVITS weights ("" leaves that half unchanged).
async fn set_weights_with(client: &Client, base_url: &str, gpt: &str, sovits: &str) -> Result<(), String> {
    let base = base_of(base_url)?;
    let (gpt, sovits) = (gpt.trim(), sovits.trim());
    if !gpt.is_empty() {
        set_one(client, &base, "/set_gpt_weights", gpt, "GPT").await?;
    }
    if !sovits.is_empty() {
        set_one(client, &base, "/set_sovits_weights", sovits, "SoVITS").await?;
    }
    Ok(())
}

// -------------------------------------------------------------------------------------------------
// Commands
// -------------------------------------------------------------------------------------------------

#[command]
async fn gpt_sovits_speak(request: GptSovitsRequest, utterance_id: String) -> Result<SpeakReport, String> {
    let app_data = crate::services::diagnostics::app_data_dir();
    // Resolving a library reference reads (and may copy) the clip: off the async runtime.
    let spec = tokio::task::spawn_blocking(move || speak_spec(&request, app_data.as_deref()))
        .await
        .map_err(|e| format!("GPT-SoVITS request failed: {e}"))??;
    speak_http(spec, &utterance_id).await
}

#[command]
async fn gpt_sovits_cancel(utterance_id: Option<String>) -> Result<(), String> {
    match utterance_id.as_deref().map(str::trim) {
        None => crate::services::openai_speech::cancel(None),
        Some("") => {}
        Some(id) => crate::services::openai_speech::cancel(Some(id)),
    }
    Ok(())
}

#[command]
async fn gpt_sovits_check(base_url: String) -> Result<bool, String> {
    check_with(http_client(), &base_url).await
}

#[command]
async fn gpt_sovits_set_weights(base_url: String, gpt_weights: String, sovits_weights: String) -> Result<(), String> {
    set_weights_with(local_http_client(), &base_url, &gpt_weights, &sovits_weights).await
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![
            gpt_sovits_speak,
            gpt_sovits_cancel,
            gpt_sovits_check,
            gpt_sovits_set_weights,
        ])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::openai_speech::testing::*;
    use std::sync::{Arc, Mutex};
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    /// A reference clip that exists on this PC (servers on this PC get local paths checked).
    fn ref_clip() -> String {
        let path = std::env::temp_dir().join(format!("sigil-gsv-ref-{}.wav", std::process::id()));
        if !path.is_file() {
            std::fs::write(&path, wav16(&[0; 160], 16_000, 1)).unwrap();
        }
        path.to_string_lossy().into_owned()
    }

    fn req(base: &str) -> GptSovitsRequest {
        GptSovitsRequest { base_url: base.into(), text: "こんにちは。".into(), ref_audio_path: ref_clip(), ..GptSovitsRequest::default() }
    }

    #[test]
    fn request_wire_names_and_defaults() {
        let r: GptSovitsRequest = serde_json::from_str(
            r#"{"baseUrl":"http://127.0.0.1:9880","text":"hi","textLang":"en","referenceVoiceId":"alice-1a2b3c","promptText":"x","promptLang":"ja","streaming":false,"speed":1.2,"extra":{"top_k":5},"timeoutMs":0,"streamingMode":3}"#,
        )
        .unwrap();
        assert_eq!(r.reference_voice_id, "alice-1a2b3c");
        assert_eq!(r.streaming_mode, 3);
        let d: GptSovitsRequest = serde_json::from_str(r#"{"text":"hi"}"#).unwrap();
        assert_eq!((d.text_lang.as_str(), d.prompt_lang.as_str(), d.streaming, d.speed), ("auto", "auto", true, 1.0));
    }

    #[test]
    fn languages_map_to_gpt_sovits_codes() {
        assert_eq!(gsv_language(""), "auto");
        assert_eq!(gsv_language("ja-JP"), "ja");
        assert_eq!(gsv_language("zh-CN"), "zh");
        assert_eq!(gsv_language("zh-HK"), "yue");
        assert_eq!(gsv_language("zh-Hant-HK"), "yue");
        assert_eq!(gsv_language("zh_MO"), "yue");
        assert_eq!(gsv_language("zh-yue"), "yue");
        assert_eq!(gsv_language("yue"), "yue");
        // Traditional Chinese (what the translation layer calls zh-TW, zh-HK and zh-MO) is Mandarin
        // unless a Cantonese marker says otherwise.
        assert_eq!(gsv_language("zh-Hant"), "zh");
        assert_eq!(gsv_language("zh-TW"), "zh");
        assert_eq!(gsv_language("zh-Hant-TW"), "zh");
        assert_eq!(gsv_language("zh-Hans"), "zh");
        assert_eq!(gsv_language("EN"), "en");
        assert_eq!(gsv_language("ko"), "ko");
        assert_eq!(gsv_language("all_ja"), "all_ja");
        assert_eq!(gsv_language("auto_yue"), "auto_yue");
    }

    #[test]
    fn bodies_carry_the_reference_streaming_and_extras() {
        let mut r = req("");
        r.text_lang = "ja".into();
        r.prompt_text = "参考です".into();
        r.prompt_lang = "ja".into();
        r.extra = serde_json::json!({ "top_k": 5, "text": "ignored", "text_split_method": "cut0" });
        let spec = speak_spec(&r, None).unwrap();
        assert_eq!(spec.url().as_str(), "http://127.0.0.1:9880/tts");
        let b = spec.body();
        assert_eq!(b["text"], "こんにちは。");
        assert_eq!(b["text_lang"], "ja");
        assert_eq!(b["ref_audio_path"], ref_clip().as_str());
        assert_eq!(b["prompt_text"], "参考です");
        assert_eq!(b["prompt_lang"], "ja");
        assert_eq!(b["media_type"], "wav");
        assert_eq!(b["streaming_mode"], true);
        assert!(b.get("speed_factor").is_none(), "speed 1 is not sent");
        assert_eq!(b["top_k"], 5);
        assert_eq!(b["text_split_method"], "cut0");

        r.streaming_mode = 3;
        r.speed = 1.25;
        let b = speak_spec(&r, None).unwrap().body().clone();
        assert_eq!(b["streaming_mode"], 3);
        assert_eq!(b["speed_factor"], 1.25);
        r.streaming = false;
        assert_eq!(speak_spec(&r, None).unwrap().body()["streaming_mode"], false);
        r.extra = serde_json::json!([1]);
        assert!(speak_spec(&r, None).is_err());
    }

    #[test]
    fn references_come_from_the_library_on_this_pc_only() {
        let app = std::env::temp_dir().join(format!("sigil-gsv-lib-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&app);
        let dir = voice_library::voice_dir(&app, "alice-1a2b3c").unwrap();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join(voice_library::CLIP_FILE), voice_library::store::tests::wav(4.0)).unwrap();
        std::fs::write(dir.join(voice_library::META_FILE), r#"{"name":"Alice","transcript":"今日はいい天気","language":"ja"}"#).unwrap();

        let r = GptSovitsRequest { reference_voice_id: "alice-1a2b3c".into(), ref_audio_path: String::new(), ..req("http://localhost:9880") };
        let rf = reference(&r, &base_of(&r.base_url).unwrap(), Some(&app)).unwrap();
        // A copy of the clip named after its content (see `versioned_clip`).
        let sent = PathBuf::from(&rf.path);
        assert_eq!(sent.parent().unwrap(), app.join(REFS_DIR));
        assert_eq!(copy_owner(&sent.file_name().unwrap().to_string_lossy()), Some("alice-1a2b3c"));
        assert_eq!(std::fs::read(&sent).unwrap(), std::fs::read(dir.join(voice_library::CLIP_FILE)).unwrap());
        assert_eq!(rf.prompt_text, "今日はいい天気");
        assert_eq!(rf.prompt_lang, "ja", "the voice's language when the setting is auto");
        // An explicit transcript and language win.
        let r2 = GptSovitsRequest { prompt_text: "override".into(), prompt_lang: "en".into(), ..r.clone() };
        let rf = reference(&r2, "http://127.0.0.1:9880", Some(&app)).unwrap();
        assert_eq!((rf.prompt_text.as_str(), rf.prompt_lang.as_str()), ("override", "en"));
        // A server elsewhere cannot read this PC's library; a missing voice is named.
        assert!(reference(&r, "http://192.168.1.20:9880", Some(&app)).unwrap_err().contains("this PC"));
        let gone = GptSovitsRequest { reference_voice_id: "gone-000001".into(), ..r.clone() };
        assert!(reference(&gone, "http://127.0.0.1:9880", Some(&app)).is_err());
        // Without any reference: a clear refusal; a missing local path is caught before the request.
        let none = GptSovitsRequest { ref_audio_path: " ".into(), ..req("") };
        assert!(reference(&none, DEFAULT_BASE_URL, Some(&app)).unwrap_err().contains("reference"));
        let missing = GptSovitsRequest { ref_audio_path: app.join("nope.wav").to_string_lossy().into_owned(), ..req("") };
        assert!(reference(&missing, DEFAULT_BASE_URL, Some(&app)).unwrap_err().contains("does not exist"));
        // A path on another machine is passed through untouched.
        let remote = GptSovitsRequest { ref_audio_path: "C:/nowhere/ref.wav".into(), ..req("http://192.168.1.20:9880") };
        assert_eq!(reference(&remote, "http://192.168.1.20:9880", None).unwrap().path, "C:/nowhere/ref.wav");
        let _ = std::fs::remove_dir_all(&app);
    }

    #[test]
    fn a_replaced_clip_is_sent_under_a_new_path() {
        // GPT-SoVITS caches the reference audio by path: a clip re-recorded in place must not reuse it.
        let app = std::env::temp_dir().join(format!("sigil-gsv-replace-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&app);
        let voice = |id: &str, seconds: f64| {
            let dir = voice_library::voice_dir(&app, id).unwrap();
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join(voice_library::CLIP_FILE), voice_library::store::tests::wav(seconds)).unwrap();
            std::fs::write(dir.join(voice_library::META_FILE), r#"{"name":"V","transcript":"はい","language":"ja"}"#).unwrap();
        };
        voice("alice-1a2b3c", 4.0);
        voice("alice-1a2b3c-2", 4.5);
        let r = GptSovitsRequest { reference_voice_id: "alice-1a2b3c".into(), ref_audio_path: String::new(), ..req(DEFAULT_BASE_URL) };
        let other = GptSovitsRequest { reference_voice_id: "alice-1a2b3c-2".into(), ..r.clone() };
        let sent = |r: &GptSovitsRequest| PathBuf::from(reference(r, DEFAULT_BASE_URL, Some(&app)).unwrap().path);

        let first = sent(&r);
        let other_first = sent(&other);
        assert_eq!(sent(&r), first, "the same clip keeps its path (the server's cache stays valid)");
        // Re-recorded in place: a new path holding the new clip; the old copy is gone, and a voice
        // whose id starts with this one's keeps its copy.
        voice("alice-1a2b3c", 5.0);
        let second = sent(&r);
        assert_ne!(second, first);
        assert!(!first.exists());
        assert!(other_first.exists());
        let clip = voice_library::voice_dir(&app, "alice-1a2b3c").unwrap().join(voice_library::CLIP_FILE);
        assert_eq!(std::fs::read(&second).unwrap(), std::fs::read(clip).unwrap());
        // Copies of voices no longer in the library are cleared on the way.
        std::fs::remove_dir_all(voice_library::voice_dir(&app, "alice-1a2b3c-2").unwrap()).unwrap();
        let _ = sent(&r);
        assert!(!other_first.exists());
        assert!(second.exists());
        // Only `<voiceId>-<12 hex>.wav` names are copies.
        assert_eq!(copy_owner("bob-01-0123456789ab.wav"), Some("bob-01"));
        assert_eq!(copy_owner("bob-0123456789ab.txt"), None);
        assert_eq!(copy_owner("bob-0123456789AB.wav"), None);
        assert_eq!(copy_owner("bob-01234.wav"), None);
        assert_eq!(copy_owner("-0123456789ab.wav"), None);
        let _ = std::fs::remove_dir_all(&app);
    }

    #[test]
    fn error_bodies_keep_the_exception() {
        assert_eq!(
            api_v2_error(br#"{"message":"tts failed","Exception":"File not found: ref.wav"}"#).as_deref(),
            Some("tts failed: File not found: ref.wav")
        );
        assert_eq!(api_v2_error(br#"{"message":"text is required"}"#).as_deref(), Some("text is required"));
        assert_eq!(api_v2_error(br#"{"detail":"Not Found"}"#).as_deref(), Some("Not Found"));
    }

    #[test]
    fn this_pc_means_loopback() {
        assert!(is_this_pc("http://127.0.0.1:9880"));
        assert!(is_this_pc("http://localhost:9880"));
        assert!(is_this_pc("http://[::1]:9880"));
        assert!(!is_this_pc("http://192.168.0.5:9880"));
        assert!(!is_this_pc("http://gpu-box:9880"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_answers_play_as_they_arrive() {
        // api_v2 streaming: a WAV header with no data size (Python's wave module writes 0), then raw
        // 16-bit chunks, in a chunked body.
        let a: Vec<i16> = (0..3200).map(|i| (i % 200) as i16 - 100).collect();
        let b: Vec<i16> = (0..1600).map(|i| (i % 50) as i16).collect();
        let mut body = chunked_head("audio/wav");
        body.extend(chunk(&wav_header(32_000, 1, 1, 16, 0)));
        body.extend(chunk(&pcm16(&a)));
        body.extend(chunk(&pcm16(&b)));
        body.extend_from_slice(LAST_CHUNK);
        let (base, server) = serve_once(body).await;
        let spec = speak_spec(&req(&base), None).unwrap();
        let (sink, pushes) = Recorder::new();
        let report = speak(spec, "gsv-stream", Box::new(sink)).await.unwrap();
        let seen = server.await.unwrap();
        assert!(seen.head.starts_with("POST /tts "), "{}", seen.head);
        assert_eq!(seen.json()["streaming_mode"], true);
        assert_eq!(report.sample_rate, 32_000);
        assert_eq!(report.frames, (a.len() + b.len()) as u64);
        let mut all = a.clone();
        all.extend(&b);
        assert_eq!(pushes.samples(), f(&all));
        assert!(pushes.formats().iter().all(|(r, c)| *r == 32_000 && *c == 1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn whole_answers_and_server_errors() {
        // Streaming off: one complete WAV (v4 voices answer at 48 kHz).
        let v: Vec<i16> = (0..4800).map(|i| (i % 64) as i16).collect();
        let (base, _server) = serve_once(full_response("200 OK", "audio/wav", &wav16(&v, 48_000, 1))).await;
        let mut r = req(&base);
        r.streaming = false;
        let (sink, pushes) = Recorder::new();
        let report = speak(speak_spec(&r, None).unwrap(), "gsv-whole", Box::new(sink)).await.unwrap();
        assert_eq!((report.sample_rate, report.frames), (48_000, 4800));
        assert_eq!(pushes.samples(), f(&v));
        // A 400 with api_v2's JSON is an error naming the problem.
        let (base, _server) =
            serve_once(full_response("400 Bad Request", "application/json", br#"{"message":"ref_audio_path is required"}"#)).await;
        let (sink, _) = Recorder::new();
        let e = speak(speak_spec(&req(&base), None).unwrap(), "gsv-err", Box::new(sink)).await.unwrap_err();
        assert!(e.contains("ref_audio_path is required"), "{e}");
    }

    /// Serve `routes` (path → (status line, content type, body)) until the test ends, recording each
    /// request target (path and query).
    async fn routes_server(routes: Vec<(&'static str, &'static str, &'static str, String)>) -> (String, Arc<Mutex<Vec<String>>>) {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let seen = Arc::new(Mutex::new(Vec::new()));
        let log = Arc::clone(&seen);
        tokio::spawn(async move {
            loop {
                let Ok((mut sock, _)) = listener.accept().await else { return };
                let mut head = Vec::new();
                let mut byte = [0u8; 1];
                while !head.ends_with(b"\r\n\r\n") {
                    if sock.read(&mut byte).await.unwrap_or(0) == 0 {
                        break;
                    }
                    head.push(byte[0]);
                }
                let text = String::from_utf8_lossy(&head).to_string();
                let target = text.split_whitespace().nth(1).unwrap_or("/").to_string();
                let path = target.split('?').next().unwrap_or("/").to_string();
                log.lock().unwrap().push(target);
                let (status, ct, body) = routes
                    .iter()
                    .find(|(p, _, _, _)| *p == path)
                    .map(|(_, s, c, b)| (*s, *c, b.clone()))
                    .unwrap_or(("404 Not Found", "application/json", r#"{"detail":"Not Found"}"#.to_string()));
                let _ = sock.write_all(&full_response(status, ct, body.as_bytes())).await;
                let _ = sock.shutdown().await;
            }
        });
        (base, seen)
    }

    #[tokio::test]
    async fn the_check_tells_api_v2_from_other_servers() {
        let schema = serde_json::json!({ "paths": { "/tts": {}, "/set_gpt_weights": {} } }).to_string();
        let (base, _) = routes_server(vec![("/openapi.json", "200 OK", "application/json", schema)]).await;
        assert!(check_with(&client(), &base).await.unwrap());
        // The older api.py has no /tts.
        let old = serde_json::json!({ "paths": { "/": {}, "/change_refer": {} } }).to_string();
        let (base, _) = routes_server(vec![("/openapi.json", "200 OK", "application/json", old)]).await;
        assert!(check_with(&client(), &base).await.unwrap_err().contains("api_v2"));
        // No schema (turned off): it answered.
        let (base, _) = routes_server(vec![]).await;
        assert!(check_with(&client(), &base).await.unwrap());
        // Nothing listening.
        let free = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let dead = format!("http://{}", free.local_addr().unwrap());
        drop(free);
        assert!(!check_with(&client(), &dead).await.unwrap());
    }

    #[tokio::test]
    async fn weights_switch_in_order_with_encoded_paths() {
        let ok = r#"{"message":"success"}"#.to_string();
        let (base, seen) = routes_server(vec![
            ("/set_gpt_weights", "200 OK", "application/json", ok.clone()),
            ("/set_sovits_weights", "200 OK", "application/json", ok),
        ])
        .await;
        set_weights_with(&client(), &base, "GPT_weights_v2/Mao-e15.ckpt", " SoVITS weights/Mao e8.pth ").await.unwrap();
        let s = seen.lock().unwrap().clone();
        assert_eq!(s.len(), 2);
        assert!(s[0].starts_with("/set_gpt_weights?weights_path=GPT_weights_v2%2FMao-e15.ckpt"), "{}", s[0]);
        assert!(s[1].starts_with("/set_sovits_weights?weights_path=SoVITS+weights%2FMao+e8.pth"), "{}", s[1]);
        // Blank halves are left alone.
        set_weights_with(&client(), &base, "", "").await.unwrap();
        assert_eq!(seen.lock().unwrap().len(), 2);

        let fail = r#"{"message":"change gpt weight failed","Exception":"No such file"}"#.to_string();
        let (base, _) = routes_server(vec![("/set_gpt_weights", "400 Bad Request", "application/json", fail)]).await;
        let e = set_weights_with(&client(), &base, "x.ckpt", "").await.unwrap_err();
        assert!(e.contains("change gpt weight failed: No such file"), "{e}");
        let (base, _) = routes_server(vec![]).await;
        assert!(set_weights_with(&client(), &base, "x.ckpt", "").await.unwrap_err().contains("api_v2"));
    }
}
