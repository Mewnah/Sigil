//! `windows-tts` plugin: Windows SAPI 5 voices.
//!
//! Speaking runs on one dedicated COM single-threaded apartment thread ([`sta`], `sigil-sapi`)
//! that owns those SAPI objects and pumps their messages; listing runs on a blocking thread with a
//! voice of its own. The commands are async and never touch SAPI on the webview's main thread.
//!
//! # IPC (`plugin-commands/windows-tts.txt`)
//! - `get_voices` → [`RpcWindowsTTSConfig`]: SAPI audio outputs and voices (token id + description).
//!   The TTS sink accepts an output's token id as its device (it resolves the endpoint), so a saved
//!   SAPI output keeps working.
//! - `speak_to_sink { request: SinkSpeakRequest, utteranceId }` → `tts_sink::SpeakReport` (P6-J):
//!   renders one sentence with the voice's own output format and streams it into the begun
//!   `tts_sink` utterance ([`render`]), so SAPI plays through the sink like every PCM backend: the
//!   top-level route (device / voice changer / both), the device and the volume are the sink's.
//!   Resolves after the last push; a stop / skip answers `"cancelled"`.
//! - `cancel_speak_to_sink { utteranceId?: string | null }`: cancel one queued or rendering
//!   sentence (`null`: all of them) and its sink utterance.
//! - `speak { data }` (legacy, kept for one release): speaks straight to a SAPI audio output with
//!   SAPI's own playback (no sink, no voice changer). The TS backend no longer uses it.
//!
//! # Wire formats
//! `SinkSpeakRequest { voice: string ("" = the Windows default voice), text: string,
//! rate?: number }` (rate: Sigil's multiplier 0.1-1-5, see [`sapi_rate`]; `null` / a string /
//! missing = 1).

mod pcm;
mod render;
mod sta;
mod stream;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Instant;

use serde::{Deserialize, Deserializer, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};
use windows::{
    core::BSTR,
    Win32::Media::Speech::{ISpeechObjectToken, ISpeechObjectTokens, ISpeechVoice, SVSFDefault, SVSFlagsAsync, SpVoice, SpeechVoiceSpeakFlags},
    Win32::System::Com::{CoCreateInstance, CoInitializeEx, CoUninitialize, CLSCTX_ALL, COINIT_MULTITHREADED},
};

use self::pcm::PcmOut;
use self::render::{RenderError, RenderRequest};
use crate::services::tts_sink::{self, SinkError, SpeakReport, UtteranceWriter};

/// SAPI `SetRate` range.
const SAPI_RATE_MIN: i32 = -10;
const SAPI_RATE_MAX: i32 = 10;
/// Lowest rate multiplier the UI offers; maps to SAPI -10.
const RATE_MULTIPLIER_MIN: f32 = 0.1;
/// Highest rate multiplier the UI offers; maps to SAPI +10.
const RATE_MULTIPLIER_MAX: f32 = 5.0;
/// Error text when the SAPI thread could not answer.
const THREAD_GONE: &str = "Windows speech stopped unexpectedly";

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
struct SpeechObject {
    pub id: String,
    pub label: String,
}

#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
pub struct RpcWindowsTTSConfig {
    devices: Vec<SpeechObject>,
    voices: Vec<SpeechObject>,
}

/// Id and description of every token (tokens that fail to answer are skipped).
fn describe_tokens(tokens: &ISpeechObjectTokens) -> Vec<SpeechObject> {
    // SAFETY: automation calls on the SAPI thread that owns the collection.
    let count = unsafe { tokens.Count() }.unwrap_or(0);
    (0..count)
        .filter_map(|i| {
            // SAFETY: as above.
            let token: ISpeechObjectToken = unsafe { tokens.Item(i) }.ok()?;
            let id = unsafe { token.Id() }.ok()?;
            let label = unsafe { token.GetDescription(0) }.ok()?;
            Some(SpeechObject {
                id: id.to_string(),
                label: label.to_string(),
            })
        })
        .collect()
}

/// The token with id `id` in `tokens`.
fn find_token(tokens: &ISpeechObjectTokens, id: &str) -> Option<ISpeechObjectToken> {
    // SAFETY: automation calls on the SAPI thread that owns the collection.
    let count = unsafe { tokens.Count() }.ok()?;
    (0..count).find_map(|i| {
        let token: ISpeechObjectToken = unsafe { tokens.Item(i) }.ok()?;
        let tid = unsafe { token.Id() }.ok()?;
        (tid == id).then_some(token)
    })
}

/// COM on a blocking-pool thread (multithreaded apartment), balanced on drop. A thread that is
/// already in another apartment keeps it (COM works there too).
struct ComScope(bool);

impl ComScope {
    fn enter() -> Self {
        // SAFETY: COM initialization of the calling thread, balanced in `drop` when it succeeded.
        Self(unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }.is_ok())
    }
}

impl Drop for ComScope {
    fn drop(&mut self) {
        if self.0 {
            // SAFETY: balances the successful CoInitializeEx of `enter` on this thread.
            unsafe { CoUninitialize() };
        }
    }
}

/// SAPI audio outputs and voices. Runs on its own (blocking) thread with its own voice object, so
/// opening the panel never waits behind a long sentence on the SAPI thread.
fn list_voices() -> Result<RpcWindowsTTSConfig, String> {
    let _com = ComScope::enter();
    // SAFETY: COM is initialized on this thread; every object below is created and released here.
    unsafe {
        let voice: ISpeechVoice =
            CoCreateInstance(&SpVoice, None, CLSCTX_ALL).map_err(|e| format!("Windows speech is not available: {}", e.message()))?;
        let outputs = voice.GetAudioOutputs(&BSTR::new(), &BSTR::new()).map_err(|_| "Failed to get device list".to_string())?;
        let voices = voice.GetVoices(&BSTR::new(), &BSTR::new()).map_err(|_| "Failed to get voice list".to_string())?;
        Ok(RpcWindowsTTSConfig {
            devices: describe_tokens(&outputs),
            voices: describe_tokens(&voices),
        })
    }
}

#[command]
async fn get_voices() -> Result<RpcWindowsTTSConfig, String> {
    tauri::async_runtime::spawn_blocking(list_voices)
        .await
        .map_err(|e| format!("Windows speech: {e}"))?
}

/// A number that may arrive as a JSON number, a numeric string, or `null` (the webview serializes
/// `NaN` from a cleared input as `null`, which used to fail the whole command).
fn lenient_f32<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f32>, D::Error> {
    let v = serde_json::Value::deserialize(d)?;
    Ok(match v {
        serde_json::Value::Number(n) => n.as_f64().map(|x| x as f32),
        serde_json::Value::String(s) => s.trim().parse::<f32>().ok(),
        _ => None,
    }
    .filter(|x| x.is_finite()))
}

#[derive(Serialize, Deserialize, Debug)]
pub struct RpcWindowsTTSSpeak {
    device: String,
    voice: String,
    value: String,
    /// 0 – 1 (missing / invalid = 1).
    #[serde(default, deserialize_with = "lenient_f32")]
    volume: Option<f32>,
    /// Multiplier: 0.1 – 1 – 5 (missing / invalid = 1).
    #[serde(default, deserialize_with = "lenient_f32")]
    rate: Option<f32>,
}

/// `speak_to_sink` request: one sentence.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct SinkSpeakRequest {
    /// SAPI voice token id (`""` = the Windows default voice).
    #[serde(default)]
    voice: String,
    text: String,
    /// Multiplier: 0.1 – 1 – 5 (missing / invalid = 1).
    #[serde(default, deserialize_with = "lenient_f32")]
    rate: Option<f32>,
}

impl SinkSpeakRequest {
    fn render_request(&self) -> RenderRequest {
        RenderRequest {
            voice: self.voice.trim().to_string(),
            text: self.text.clone(),
            rate: sapi_rate(self.rate.unwrap_or(1.0)),
        }
    }
}

/// Sigil's rate multiplier → SAPI `SetRate` (-10..=10).
///
/// `[1, 5]` maps linearly onto `[0, 10]` (unchanged from before, truncating toward zero), and
/// `[0.1, 1)` now maps linearly onto `[-10, 0)`. The old `rate < 1` branch produced -10..-100,
/// outside SAPI's range, so `SetRate` failed and every rate below 1.0 made speech fail.
pub(crate) fn sapi_rate(multiplier: f32) -> i32 {
    if !multiplier.is_finite() {
        return 0;
    }
    let steps = if multiplier >= 1.0 {
        (multiplier - 1.0) / (RATE_MULTIPLIER_MAX - 1.0) * SAPI_RATE_MAX as f32
    } else {
        (multiplier - 1.0) / (1.0 - RATE_MULTIPLIER_MIN) * -(SAPI_RATE_MIN as f32)
    };
    // `as` saturates on overflow; the clamp keeps out-of-range multipliers valid.
    (steps as i32).clamp(SAPI_RATE_MIN, SAPI_RATE_MAX)
}

/// Volume 0–1 → SAPI `SetVolume` (0..=100).
pub(crate) fn sapi_volume(volume: f32) -> i32 {
    if !volume.is_finite() {
        return 100;
    }
    ((volume * 100.0).round() as i32).clamp(0, 100)
}

/// Legacy: speak straight to a SAPI audio output (SAPI plays it; asynchronous).
fn speak_to_device(voice: &ISpeechVoice, data: RpcWindowsTTSSpeak) -> Result<(), String> {
    // SAFETY: automation calls on the SAPI thread that owns the voice.
    unsafe {
        if voice.SetVolume(sapi_volume(data.volume.unwrap_or(1.0))).is_err() {
            return Err("Unable to update volume".into());
        }
        if voice.SetRate(sapi_rate(data.rate.unwrap_or(1.0))).is_err() {
            return Err("Unable to update rate".into());
        }
        // An empty device keeps SAPI's current (default) output.
        if !data.device.is_empty() {
            let applied = voice
                .GetAudioOutputs(&BSTR::new(), &BSTR::new())
                .ok()
                .and_then(|outputs| find_token(&outputs, &data.device))
                .and_then(|token| voice.putref_AudioOutput(&token).ok());
            if applied.is_none() {
                return Err("Failed to apply device".into());
            }
        }
        let applied = voice
            .GetVoices(&BSTR::new(), &BSTR::new())
            .ok()
            .and_then(|voices| find_token(&voices, &data.voice))
            .and_then(|token| voice.putref_Voice(&token).ok());
        if applied.is_none() {
            return Err("Failed to apply voice".into());
        }
        voice
            .Speak(&data.value.into(), SpeechVoiceSpeakFlags(SVSFDefault.0 | SVSFlagsAsync.0))
            .map(|_| ())
            .map_err(|_| "Unable to process text".into())
    }
}

#[command]
async fn speak(data: RpcWindowsTTSSpeak) -> Result<(), String> {
    if data.value.trim().is_empty() {
        return Ok(());
    }
    sta::run(move |st| speak_to_device(st.device_voice()?, data))
        .await
        .unwrap_or_else(|_| Err(THREAD_GONE.into()))
}

// ---------------------------------------------------------------------------------------------
// Rendering into the TTS sink
// ---------------------------------------------------------------------------------------------

/// Sentences queued for or rendering on the SAPI thread, with their cancel flags.
static RENDERS: Mutex<Vec<(String, Arc<AtomicBool>)>> = Mutex::new(Vec::new());

fn renders() -> MutexGuard<'static, Vec<(String, Arc<AtomicBool>)>> {
    RENDERS.lock().unwrap_or_else(PoisonError::into_inner)
}

/// One sentence's entry in [`RENDERS`], removed on drop (also when its job never ran).
struct Registered {
    id: String,
    flag: Arc<AtomicBool>,
}

impl Registered {
    fn new(id: &str) -> Self {
        let flag = Arc::new(AtomicBool::new(false));
        renders().push((id.to_string(), flag.clone()));
        Self { id: id.to_string(), flag }
    }

    fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }
}

impl Drop for Registered {
    fn drop(&mut self) {
        renders().retain(|(id, f)| !(id == &self.id && Arc::ptr_eq(f, &self.flag)));
    }
}

/// Cancel the registered sentence `id` (`None`: every one). Returns the utterance ids cancelled.
fn cancel_registered(id: Option<&str>) -> Vec<String> {
    let list = renders();
    list.iter()
        .filter(|(rid, _)| id.is_none_or(|want| want == rid))
        .map(|(rid, flag)| {
            flag.store(true, Ordering::Release);
            rid.clone()
        })
        .collect()
}

/// The sink utterance as the renderer's output.
struct SinkOut(UtteranceWriter);

impl PcmOut for SinkOut {
    fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError> {
        self.0.push(samples, sample_rate, channels)
    }
}

/// IPC text of a sink error (`"cancelled"` for a stop / skip).
fn sink_error(e: SinkError) -> String {
    e.to_string()
}

/// Render one sentence on the SAPI thread into the sink utterance `id`.
fn speak_job(st: &mut sta::StaState, request: &RenderRequest, id: &str, job: &Registered, received: Instant) -> Result<SpeakReport, String> {
    if job.is_cancelled() {
        return Err(sink_error(SinkError::Cancelled));
    }
    // One writer feeds the audio (on SAPI's thread); the other only watches for a cancel.
    let writer = tts_sink::writer(id).map_err(sink_error)?;
    let watch = tts_sink::writer(id).map_err(sink_error)?;
    let renderer = st.renderer()?;
    let cancelled = || job.is_cancelled() || watch.is_cancelled();
    match renderer.render(request, Box::new(SinkOut(writer)), &cancelled, received) {
        Ok(report) => Ok(report),
        Err(RenderError::Wedged(message)) => {
            tracing::warn!(target: "sigil::tts", "Windows speech: {message}");
            st.abandon_renderer();
            Err(message)
        }
        Err(e) => Err(e.message()),
    }
}

#[command]
async fn speak_to_sink(request: SinkSpeakRequest, utterance_id: String) -> Result<SpeakReport, String> {
    let id = utterance_id.trim().to_string();
    if id.is_empty() {
        return Err("utteranceId must not be empty".into());
    }
    let received = Instant::now();
    let job = Registered::new(&id);
    let render = request.render_request();
    sta::run(move |st| speak_job(st, &render, &id, &job, received))
        .await
        .unwrap_or_else(|_| Err(THREAD_GONE.into()))
}

/// Cancel one sentence (`None`: all of them): it stops rendering at once and its sink utterance is
/// cancelled too, so a render waiting for room in the sink ends as well.
#[command]
async fn cancel_speak_to_sink(utterance_id: Option<String>) -> Result<(), String> {
    for id in cancel_registered(utterance_id.as_deref().map(str::trim)) {
        tts_sink::cancel(Some(&id));
    }
    Ok(())
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new("windows-tts")
        .invoke_handler(tauri::generate_handler![speak, get_voices, speak_to_sink, cancel_speak_to_sink])
        .build()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sapi_rate_stays_in_range_for_every_ui_value() {
        let mut r = 0.0f32;
        while r <= 6.0 {
            let s = sapi_rate(r);
            assert!((SAPI_RATE_MIN..=SAPI_RATE_MAX).contains(&s), "rate {r} -> {s}");
            r += 0.05;
        }
        for x in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -3.0, 1e9] {
            let s = sapi_rate(x);
            assert!((SAPI_RATE_MIN..=SAPI_RATE_MAX).contains(&s), "rate {x} -> {s}");
        }
    }

    #[test]
    fn sapi_rate_mapping_points() {
        assert_eq!(sapi_rate(1.0), 0);
        assert_eq!(sapi_rate(5.0), 10);
        assert_eq!(sapi_rate(3.0), 5);
        // Unchanged from the previous mapping for >= 1 (truncation toward zero).
        assert_eq!(sapi_rate(1.2), 0);
        assert_eq!(sapi_rate(2.0), 2);
        // Previously -10..-100 (rejected by SAPI); now a proportional slow-down.
        assert_eq!(sapi_rate(0.1), -10);
        assert_eq!(sapi_rate(0.55), -5);
        assert_eq!(sapi_rate(0.95), 0);
        assert_eq!(sapi_rate(0.0), -10);
        assert_eq!(sapi_rate(f32::NAN), 0);
    }

    #[test]
    fn sapi_rate_is_monotonic() {
        let mut prev = i32::MIN;
        let mut r = 0.1f32;
        while r <= 5.0 {
            let s = sapi_rate(r);
            assert!(s >= prev, "rate {r}: {s} < {prev}");
            prev = s;
            r += 0.05;
        }
    }

    #[test]
    fn sapi_volume_is_clamped() {
        assert_eq!(sapi_volume(1.0), 100);
        assert_eq!(sapi_volume(0.29), 29);
        assert_eq!(sapi_volume(0.0), 0);
        assert_eq!(sapi_volume(-1.0), 0);
        assert_eq!(sapi_volume(3.0), 100);
        assert_eq!(sapi_volume(f32::NAN), 100);
    }

    #[test]
    fn speak_payload_tolerates_null_and_strings() {
        let v: RpcWindowsTTSSpeak = serde_json::from_value(serde_json::json!({
            "device": "d", "voice": "v", "value": "hello", "volume": null, "rate": "0.5"
        }))
        .unwrap();
        assert_eq!(v.volume, None);
        assert_eq!(v.rate, Some(0.5));
        let v: RpcWindowsTTSSpeak = serde_json::from_value(serde_json::json!({
            "device": "d", "voice": "v", "value": "hello", "volume": 0.8, "rate": 1.5
        }))
        .unwrap();
        assert_eq!(v.volume, Some(0.8));
        assert_eq!(v.rate, Some(1.5));
        let v: RpcWindowsTTSSpeak =
            serde_json::from_value(serde_json::json!({"device": "", "voice": "v", "value": "x", "rate": "fast"})).unwrap();
        assert_eq!((v.volume, v.rate), (None, None));
    }

    #[test]
    fn sink_request_wire_format() {
        let r: SinkSpeakRequest = serde_json::from_value(serde_json::json!({
            "voice": "  HKEY\\Voices\\Tokens\\X ", "text": "Hi.", "rate": "5"
        }))
        .unwrap();
        assert_eq!(
            r.render_request(),
            RenderRequest {
                voice: "HKEY\\Voices\\Tokens\\X".into(),
                text: "Hi.".into(),
                rate: 10
            }
        );
        // Voice and rate are optional; a bad rate is 1 (SAPI 0).
        let r: SinkSpeakRequest = serde_json::from_value(serde_json::json!({"text": "x", "rate": null})).unwrap();
        assert_eq!(r.render_request().voice, "");
        assert_eq!(r.render_request().rate, 0);
        assert!(serde_json::from_value::<SinkSpeakRequest>(serde_json::json!({"voice": "v"})).is_err(), "text is required");
    }

    #[test]
    fn cancel_flags_follow_registration() {
        let a = Registered::new("p6j-test-a");
        let b = Registered::new("p6j-test-b");
        let a2 = Registered::new("p6j-test-a"); // a repeated id has its own flag
        assert_eq!(cancel_registered(Some("p6j-test-b")), vec!["p6j-test-b".to_string()]);
        assert!(b.is_cancelled() && !a.is_cancelled() && !a2.is_cancelled());
        drop(a2);
        let ids = cancel_registered(None);
        assert!(ids.contains(&"p6j-test-a".to_string()) && ids.contains(&"p6j-test-b".to_string()));
        assert!(a.is_cancelled());
        drop(a);
        drop(b);
        assert!(cancel_registered(Some("p6j-test-a")).is_empty(), "dropped entries are gone");
        assert!(cancel_registered(Some("p6j-test-b")).is_empty());
    }

    #[test]
    fn a_sentence_cancelled_before_it_ran_never_touches_the_sink() {
        let job = Registered::new("p6j-test-early");
        job.flag.store(true, Ordering::Release);
        let mut st = sta::StaState::default();
        let req = RenderRequest {
            text: "x".into(),
            ..RenderRequest::default()
        };
        assert_eq!(speak_job(&mut st, &req, "p6j-test-early", &job, Instant::now()), Err("cancelled".into()));
    }

    #[test]
    fn voices_and_outputs_are_listed() {
        let got = tauri::async_runtime::block_on(get_voices());
        match got {
            // Every voice and output has an id; outputs are endpoint-named.
            Ok(cfg) => {
                assert!(cfg.voices.iter().all(|v| !v.id.is_empty()));
                assert!(cfg.devices.iter().all(|d| !d.id.is_empty()));
            }
            Err(e) => eprintln!("SAPI unavailable here ({e}): skipped"),
        }
    }
}
