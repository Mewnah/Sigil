//! Clone registry: voice-library voices become voices of the running Qwen3-TTS server.
//!
//! `qwen_tts_prepare_clone { voiceId }` registers the library voice with the running server (a Base
//! talker) under [`server_voice_name`] and returns that name; TypeScript sends it as the speech
//! request's `voice`. Each call is cheap once a voice is registered with the running instance (an
//! in-memory check of the instance's pid, the clip's sha256 and the transcript), so the backend
//! calls it before every sentence: a re-recorded clip, an edited transcript or a restarted helper
//! (whose registry is empty) register again by themselves.
//!
//! Speaker data cache (`voices/<id>/cache/qwen3-tts.bin`, [`super::CACHE_BACKEND`]): the server
//! answers a WAV registration with the latents it extracted (the speaker embedding, which depends
//! on the talker's speaker encoder, and the reference's 12 Hz codec codes, which depend on the
//! tokenizer). They are stored per (talker file, tokenizer file) for the clip's sha256, so the next
//! registration sends them instead of the clip and the server skips the extraction. A stale or
//! unreadable cache is ignored; a refused cached registration falls back to the clip.

use super::server::{self, Launch, TARGET};
use super::{QwenCloneVoice, CACHE_BACKEND};
use crate::services::tts_helpers::TtsHelperEndpoint;
use crate::services::voice_library;
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

/// Prefix of the server voice names Sigil registers.
pub const VOICE_PREFIX: &str = "sigil-";
/// `format` of the cache file.
pub const CACHE_FORMAT: &str = "sigil-qwen3-tts-voice";
pub const CACHE_VERSION: u32 = 1;
/// Talker / tokenizer pairs kept per voice (1.7B and 0.6B, a quantization change).
pub const MAX_CACHE_ENTRIES: usize = 4;
/// Largest clip sent to the server (the server accepts 32 MB bodies; library clips are ≤ 10 s).
const MAX_CLIP_BYTES: u64 = 16 << 20;
/// Bounds of decoded latents (the 1.7B speaker embedding is 2048 floats; 10 s of codes is ~3 KB).
const MAX_SPK_BYTES: usize = 64 << 10;
const MAX_RVQ_BYTES: usize = 1 << 20;
/// A registration extracts on the GPU in well under a second; on the CPU pack it can take seconds.
const REGISTER_TIMEOUT: Duration = Duration::from_secs(120);

/// The server voice name of library voice `voice_id` (the server lower-cases names; ids already are).
pub fn server_voice_name(voice_id: &str) -> String {
    format!("{VOICE_PREFIX}{voice_id}")
}

// ---------------------------------------------------------------------------------------------
// Cache file
// ---------------------------------------------------------------------------------------------

/// `voices/<id>/cache/qwen3-tts.bin` (JSON).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheFile {
    pub format: String,
    pub version: u32,
    /// sha256 of the clip the latents were extracted from.
    pub clip_sha256: String,
    /// Newest first.
    pub entries: Vec<CacheEntry>,
}

/// Latents of one talker / tokenizer pair (base64 of the server's `.spk` and `.rvq` payloads).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CacheEntry {
    pub talker: String,
    pub codec: String,
    pub spk_b64: String,
    pub rvq_b64: String,
    pub ref_t: u32,
    pub spk_dim: u32,
}

/// Valid base64 latents within bounds.
fn latents_ok(spk_b64: &str, rvq_b64: &str) -> bool {
    let b64 = base64::engine::general_purpose::STANDARD;
    let spk = b64.decode(spk_b64.as_bytes());
    let rvq = b64.decode(rvq_b64.as_bytes());
    matches!((spk, rvq), (Ok(s), Ok(r)) if !s.is_empty() && s.len() % 4 == 0 && s.len() <= MAX_SPK_BYTES && !r.is_empty() && r.len() <= MAX_RVQ_BYTES)
}

/// The cache of the clip `clip_sha256`, or `None` when `bytes` is not one (another format or
/// version, another clip, damaged).
pub fn decode_cache(bytes: &[u8], clip_sha256: &str) -> Option<CacheFile> {
    let mut f: CacheFile = serde_json::from_slice(bytes).ok()?;
    if f.format != CACHE_FORMAT || f.version != CACHE_VERSION || !f.clip_sha256.eq_ignore_ascii_case(clip_sha256) {
        return None;
    }
    f.entries.retain(|e| latents_ok(&e.spk_b64, &e.rvq_b64));
    f.entries.truncate(MAX_CACHE_ENTRIES);
    Some(f)
}

/// The entry for a talker / tokenizer pair.
pub fn find_entry<'a>(f: &'a CacheFile, talker: &str, codec: &str) -> Option<&'a CacheEntry> {
    f.entries
        .iter()
        .find(|e| e.talker.eq_ignore_ascii_case(talker) && e.codec.eq_ignore_ascii_case(codec))
}

/// `existing` (if it belongs to this clip) with `entry` first, replacing its pair, at most
/// [`MAX_CACHE_ENTRIES`].
pub fn merge_cache(existing: Option<CacheFile>, clip_sha256: &str, entry: CacheEntry) -> CacheFile {
    let mut entries = vec![entry.clone()];
    if let Some(f) = existing.filter(|f| f.clip_sha256.eq_ignore_ascii_case(clip_sha256)) {
        entries.extend(f.entries.into_iter().filter(|e| !(e.talker.eq_ignore_ascii_case(&entry.talker) && e.codec.eq_ignore_ascii_case(&entry.codec))));
    }
    entries.truncate(MAX_CACHE_ENTRIES);
    CacheFile {
        format: CACHE_FORMAT.into(),
        version: CACHE_VERSION,
        clip_sha256: clip_sha256.to_ascii_lowercase(),
        entries,
    }
}

// ---------------------------------------------------------------------------------------------
// Registry of the running instance
// ---------------------------------------------------------------------------------------------

/// What the running server holds for a library voice.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Registered {
    pid: u32,
    clip_sha256: String,
    ref_text: String,
}

fn registry() -> std::sync::MutexGuard<'static, HashMap<String, Registered>> {
    static R: OnceLock<Mutex<HashMap<String, Registered>>> = OnceLock::new();
    R.get_or_init(Default::default)
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

/// Forget every registration (the helper stopped, restarted or was replaced).
pub fn forget_all() {
    registry().clear();
}

fn gate() -> &'static tokio::sync::Mutex<()> {
    static G: OnceLock<tokio::sync::Mutex<()>> = OnceLock::new();
    G.get_or_init(Default::default)
}

// ---------------------------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------------------------

/// The server's answer to `POST /v1/audio/voices` (Sigil's build adds the extracted latents).
#[derive(Clone, Debug, Default, Deserialize)]
#[serde(default)]
pub struct RegisterReply {
    pub name: String,
    pub status: String,
    pub extracted: bool,
    pub spk_b64: Option<String>,
    pub rvq_b64: Option<String>,
    #[serde(rename = "ref_T")]
    pub ref_t: u32,
    pub spk_dim: u32,
}

/// The JSON body registering `name` from a WAV.
pub fn wav_body(name: &str, ref_text: &str, wav: &[u8]) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "ref_text": ref_text,
        "wav_b64": base64::engine::general_purpose::STANDARD.encode(wav),
    })
}

/// The JSON body registering `name` from cached latents.
pub fn latents_body(name: &str, ref_text: &str, entry: &CacheEntry) -> serde_json::Value {
    serde_json::json!({
        "name": name,
        "ref_text": ref_text,
        "spk_b64": entry.spk_b64,
        "rvq_b64": entry.rvq_b64,
    })
}

async fn register(endpoint: &TtsHelperEndpoint, body: &serde_json::Value) -> Result<RegisterReply, String> {
    let resp = server::client()
        .post(format!("{}/audio/voices", endpoint.base_url))
        .bearer_auth(&endpoint.api_key)
        .json(body)
        .timeout(REGISTER_TIMEOUT)
        .send()
        .await
        .map_err(|e| format!("cannot reach the Qwen3-TTS helper: {e}"))?;
    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        let message = serde_json::from_str::<serde_json::Value>(&text)
            .ok()
            .and_then(|v| v["error"]["message"].as_str().map(str::to_string))
            .unwrap_or_else(|| text.chars().take(200).collect());
        return Err(format!("the Qwen3-TTS helper refused the voice ({status}): {message}"));
    }
    let reply = resp
        .json::<RegisterReply>()
        .await
        .map_err(|e| format!("unreadable voice registration answer: {e}"))?;
    if !reply.status.is_empty() && reply.status != "registered" {
        return Err(format!("the Qwen3-TTS helper did not register the voice ({})", reply.status));
    }
    Ok(reply)
}

/// What the clone registry needs of a library voice's clip.
struct LibraryClip {
    path: PathBuf,
    /// sha256 of the clip file (the cache key).
    sha256: String,
    /// What is said in the clip, trimmed ("" when unknown).
    transcript: String,
}

/// The clip, its transcript and the cache of library voice `voice_id` (blocking).
fn read_voice(app_data: &Path, voice_id: &str) -> Result<(LibraryClip, Option<CacheFile>), String> {
    let clip = voice_library::reference_clip(app_data, voice_id)?;
    let cache = voice_library::read_cache(app_data, voice_id, CACHE_BACKEND).and_then(|b| decode_cache(&b, &clip.sha256));
    let clip = LibraryClip {
        transcript: clip.transcript.as_deref().map(str::trim).unwrap_or("").to_string(),
        path: clip.path,
        sha256: clip.sha256,
    };
    Ok((clip, cache))
}

fn read_clip_bytes(path: &Path) -> Result<Vec<u8>, String> {
    let len = std::fs::metadata(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?
        .len();
    if len > MAX_CLIP_BYTES {
        return Err(format!("{} is too large for a reference clip ({len} bytes)", path.display()));
    }
    std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))
}

/// Register library voice `voice_id` with the running server (see the module docs).
pub async fn prepare(voice_id: &str) -> Result<QwenCloneVoice, String> {
    let voice_id = voice_id.trim().to_string();
    if !voice_library::valid_voice_id(&voice_id) {
        return Err(format!("invalid voice id {voice_id:?}"));
    }
    let app_data = crate::services::diagnostics::app_data_dir().ok_or("the app data folder is not known")?;
    prepare_in(&app_data, &voice_id).await
}

/// [`prepare`] against the voice library under `app_data` (tests).
pub async fn prepare_in(app_data: &Path, voice_id: &str) -> Result<QwenCloneVoice, String> {
    let _gate = gate().lock().await;
    let (endpoint, pid, launch) = server::current_target()?;
    if launch.mode != super::plan::Mode::Clone {
        return Err(format!(
            "Qwen3-TTS is running {} ({} mode), which cannot clone voices: switch the mode to voice cloning",
            launch.model,
            launch.mode.as_str()
        ));
    }
    let name = server_voice_name(voice_id);
    let dir = app_data.to_path_buf();
    let id = voice_id.to_string();
    let (clip, cache) = tauri::async_runtime::spawn_blocking(move || read_voice(&dir, &id))
        .await
        .map_err(|e| format!("voice library task failed: {e}"))??;
    let ref_text = clip.transcript.clone();

    let wanted = Registered {
        pid,
        clip_sha256: clip.sha256.clone(),
        ref_text: ref_text.clone(),
    };
    if registry().get(voice_id) == Some(&wanted) {
        return Ok(QwenCloneVoice {
            voice_id: voice_id.to_string(),
            voice: name,
            cached: true,
        });
    }
    if ref_text.is_empty() {
        tracing::info!(target: TARGET, "voice {voice_id} has no transcript: cloning from its timbre only (lower similarity)");
    }

    // Cached latents first.
    let talker = launch.talker_file();
    let codec = launch.codec_file();
    if let Some(entry) = cache.as_ref().and_then(|c| find_entry(c, &talker, &codec)) {
        match register(&endpoint, &latents_body(&name, &ref_text, entry)).await {
            Ok(_) => {
                record(voice_id, wanted, &launch);
                tracing::info!(target: TARGET, "voice {voice_id} registered from its cached speaker data");
                return Ok(QwenCloneVoice {
                    voice_id: voice_id.to_string(),
                    voice: name,
                    cached: true,
                });
            }
            Err(e) => tracing::warn!(target: TARGET, "cached speaker data of {voice_id} refused, extracting again: {e}"),
        }
    }

    // Extract from the clip.
    let path = clip.path.clone();
    let wav = tauri::async_runtime::spawn_blocking(move || read_clip_bytes(&path))
        .await
        .map_err(|e| format!("voice library task failed: {e}"))??;
    let started = std::time::Instant::now();
    let reply = register(&endpoint, &wav_body(&name, &ref_text, &wav)).await?;
    tracing::info!(
        target: TARGET,
        "voice {voice_id} registered as {} from its clip in {} ms ({} codec frames, {} speaker values, extracted: {})",
        reply.name,
        started.elapsed().as_millis(),
        reply.ref_t,
        reply.spk_dim,
        reply.extracted
    );
    record(voice_id, wanted, &launch);
    if let (Some(spk), Some(rvq)) = (reply.spk_b64.as_deref(), reply.rvq_b64.as_deref()) {
        if latents_ok(spk, rvq) {
            let entry = CacheEntry {
                talker,
                codec,
                spk_b64: spk.to_string(),
                rvq_b64: rvq.to_string(),
                ref_t: reply.ref_t,
                spk_dim: reply.spk_dim,
            };
            let dir = app_data.to_path_buf();
            let id = voice_id.to_string();
            let sha = clip.sha256.clone();
            let written = tauri::async_runtime::spawn_blocking(move || {
                let bytes = serde_json::to_vec(&merge_cache(cache, &sha, entry)).map_err(|e| e.to_string())?;
                voice_library::write_cache(&dir, &id, CACHE_BACKEND, &bytes)
            })
            .await
            .map_err(|e| e.to_string())
            .and_then(|r| r);
            if let Err(e) = written {
                tracing::warn!(target: TARGET, "cannot cache the speaker data of {voice_id}: {e}");
            }
        } else {
            tracing::warn!(target: TARGET, "the helper answered unusable speaker data for {voice_id}; not cached");
        }
    }
    Ok(QwenCloneVoice {
        voice_id: voice_id.to_string(),
        voice: name,
        cached: false,
    })
}

/// Remember the registration, unless the helper was replaced or restarted meanwhile.
fn record(voice_id: &str, wanted: Registered, launch: &Launch) {
    let still = server::current_target().is_ok_and(|(_, pid, l)| pid == wanted.pid && l.same_talker(launch));
    if still {
        registry().insert(voice_id.to_string(), wanted);
    }
}
