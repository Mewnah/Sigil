//! `edge-tts` plugin: Microsoft Edge "Read aloud" voices, keyless (P6-H; scaffolded by P6-S0).
//!
//! A native Rust client of the Read-aloud WebSocket over `tungstenite`: the browser's headers, the
//! `Sec-MS-GEC` token and the clock-skew retry ([`protocol`], [`session`]). The streamed MP3 is
//! decoded as it arrives and pushed into the TTS sink utterance it is given, like `openai_speech`
//! (no PCM crosses JSON IPC; the sink applies device, volume and route). The service is unofficial
//! and may change or fail (503s, stalls): the TS chain falls back to the local voice (Kokoro) on any
//! failure and on a stall (`src/core/services/tts/pipeline.ts`, `services/edge.ts`).
//!
//! # Frozen contract (P6-S0; change only through INTEGRATE; structs may gain fields)
//! - IPC (`plugin-commands/edge-tts.txt`; wire camelCase; TS mirror `RustEdgeTts*` in
//!   `src/types.ts`):
//!   - `edge_tts_speak { request: EdgeTtsRequest, utteranceId } -> SpeakReport`: speak one
//!     sentence into the begun `tts_sink` utterance `utteranceId` (the sink applies device, volume
//!     and route).
//!   - `edge_tts_cancel { utteranceId? } -> ()`: abort one request (or all) promptly.
//!   - `edge_tts_list_voices -> EdgeTtsVoice[]`: the service's voice list (cached).
//!
//! # Behaviour (P6-H)
//! - Errors carry the cloud failure tag ([`cloud`]): `"[cloud-tts:<kind>…] <message>"`; a
//!   cancelled request (stop / skip, `edge_tts_cancel`) rejects with `"cancelled"`; a failure of
//!   the sink (or a second request for the same utterance) rejects untagged with its own text.
//! - `timeoutMs` bounds the wait for the first audio (connect included) and every later gap
//!   (0.5–60 s); 0 = automatic (15 s: there is nothing better to wait for). The TS backend passes
//!   its short "switch to the local voice" timeout (3 s by default) when the fallback is ready.
//!   Reaching the service (DNS + TCP connect, its IPv6 and IPv4 addresses raced 250 ms apart) has
//!   a third of that (at least 1 s): an unreachable service is a `network` failure after about
//!   1 s, an error answer (429, 5xx) at once, a service that connects but sends no audio a
//!   `stalled` one at the timeout.
//! - Text over the service's 4096-byte limit is spoken as several requests into the same
//!   utterance. Each request uses its own connection (as the browser's Read aloud does per
//!   paragraph), so a broken connection never affects the next sentence.

pub(crate) mod cloud;
mod protocol;
mod session;
mod voices;

#[cfg(test)]
mod tests;

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{
    command,
    plugin::{Builder, TauriPlugin},
    Runtime,
};

use cloud::{cancel_command, sink_for, unix_now, CancelWatch, FailureKind, Registry, SpeakError};
use session::{Clock, Plan, Progress, WHAT};

use crate::services::openai_speech::PcmSink;
use crate::services::tts_sink::SpeakReport;

/// Plugin name (`plugin:edge-tts|...`, capability `edge-tts:default`).
pub const PLUGIN_NAME: &str = "edge-tts";
/// The voice of a request that names none.
pub const DEFAULT_VOICE: &str = "en-US-AvaMultilingualNeural";

/// Automatic first-audio and stall limits (no fallback is waiting).
const AUTO_TIMEOUT: Duration = Duration::from_secs(15);
/// Range of an explicit `timeoutMs`.
const MIN_TIMEOUT: Duration = Duration::from_millis(500);
const MAX_TIMEOUT: Duration = Duration::from_secs(60);
/// Time the async watchdog leaves the blocking request to report its own first-audio timeout.
const WATCHDOG_GRACE: Duration = Duration::from_millis(250);
/// The same for the connect bound. The blocking side's TCP connect ends at that bound too (with the
/// same `network` failure), so this only has to cover a DNS lookup that hangs: kept short, so an
/// unreachable service hands over to the local voice about 1 s after the request started.
const UNREACHABLE_GRACE: Duration = Duration::from_millis(50);

/// One sentence to speak (TS `RustEdgeTtsRequest`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct EdgeTtsRequest {
    pub text: String,
    /// Voice short name, e.g. `en-US-AvaMultilingualNeural`, `ja-JP-NanamiNeural`.
    pub voice: String,
    /// Speaking rate, 1 = normal (sent as a percentage; clamped to 0.5-2).
    pub rate: f32,
    /// Pitch offset in Hz (0 = voice default).
    pub pitch_hz: i32,
    /// Time to the first audio (ms); 0 = automatic.
    pub timeout_ms: u32,
}

impl Default for EdgeTtsRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            voice: String::new(),
            rate: 1.0,
            pitch_hz: 0,
            timeout_ms: 0,
        }
    }
}

/// A voice of the service (TS `RustEdgeTtsVoice`).
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct EdgeTtsVoice {
    /// `en-US-AvaMultilingualNeural` (the request's `voice`).
    pub short_name: String,
    /// Display name.
    pub friendly_name: String,
    /// `en-US`.
    pub locale: String,
    /// `Female` / `Male`.
    pub gender: String,
}

// ---------------------------------------------------------------------------------------------
// Clock skew (the token is time-based)
// ---------------------------------------------------------------------------------------------

/// Seconds the service's clock is ahead of the PC's (measured from a refused request's `Date`).
static CLOCK_SKEW: AtomicI64 = AtomicI64::new(0);

/// The PC clock corrected by [`CLOCK_SKEW`].
struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> i64 {
        unix_now() + CLOCK_SKEW.load(Ordering::Relaxed)
    }

    fn adjust(&self, server_unix: i64) {
        let skew = server_unix - unix_now();
        CLOCK_SKEW.store(skew, Ordering::Relaxed);
        tracing::info!(target: "sigil::tts", "{WHAT}: the service's clock differs from this PC's by {skew} s; adjusted");
    }
}

// ---------------------------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------------------------

/// Where the service is (tests point it at a local server).
#[derive(Clone, Debug)]
pub(crate) struct Endpoint {
    pub wss: String,
    pub voices: String,
}

impl Endpoint {
    fn service() -> Self {
        Self { wss: protocol::WSS_URL.into(), voices: protocol::VOICES_URL.into() }
    }
}

/// `timeoutMs` as (first audio, stall).
fn timing(timeout_ms: u32) -> (Duration, Duration) {
    if timeout_ms == 0 {
        return (AUTO_TIMEOUT, AUTO_TIMEOUT);
    }
    let t = Duration::from_millis(u64::from(timeout_ms)).clamp(MIN_TIMEOUT, MAX_TIMEOUT);
    (t, t)
}

/// The validated plan of `request`, or `None` when there is nothing to say.
fn plan(request: &EdgeTtsRequest, endpoint: &Endpoint) -> Result<Option<Plan>, SpeakError> {
    let text = request.text.trim();
    if text.is_empty() {
        return Ok(None);
    }
    let voice = if request.voice.trim().is_empty() { DEFAULT_VOICE } else { request.voice.trim() };
    let voice = protocol::voice_name(voice).map_err(|e| SpeakError::fail(FailureKind::Rejected, format!("{WHAT}: {e}")))?;
    let pieces = protocol::split_escaped(&protocol::escape_text(text), protocol::MAX_TEXT_BYTES);
    if pieces.is_empty() {
        return Ok(None);
    }
    let (first_audio, stall) = timing(request.timeout_ms);
    Ok(Some(Plan {
        wss_base: endpoint.wss.clone(),
        voice,
        rate: protocol::rate_attr(request.rate),
        pitch: protocol::pitch_attr(request.pitch_hz),
        pieces,
        first_audio,
        stall,
        connect: session::connect_bound(first_audio),
    }))
}

/// Running requests by utterance id (`edge_tts_cancel`).
static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

/// Speak `request` into `sink` (registered under `utterance_id` for cancellation).
pub(crate) async fn speak_with(
    request: &EdgeTtsRequest,
    utterance_id: &str,
    endpoint: &Endpoint,
    sink: Box<dyn PcmSink>,
    clock: Arc<dyn Clock>,
) -> Result<SpeakReport, SpeakError> {
    let Some(plan) = plan(request, endpoint)? else {
        return Ok(SpeakReport { sample_rate: protocol::SAMPLE_RATE, ..SpeakReport::default() });
    };
    // A second request for the same utterance is the caller's mistake, not the service's.
    let registration = REGISTRY.register(utterance_id).map_err(SpeakError::Local)?;
    let cancel = CancelWatch::new(Arc::clone(&registration.flag), sink.cancel_probe());
    if cancel.is_set() {
        return Err(SpeakError::Cancelled);
    }
    let started = Instant::now();
    let first_deadline = started + plan.first_audio;
    let connect_deadline = started + plan.connect;
    let first_audio = plan.first_audio;
    let connect = plan.connect;
    let progress = Arc::new(Progress::default());
    let mut task = {
        let cancel = cancel.clone();
        let progress = Arc::clone(&progress);
        tokio::task::spawn_blocking(move || {
            let mut sink = sink;
            catch_unwind(AssertUnwindSafe(|| session::run(&plan, sink.as_mut(), &cancel, &progress, clock.as_ref(), started)))
                .unwrap_or_else(|_| Err(SpeakError::fail(FailureKind::Other, format!("{WHAT}: the request failed unexpectedly."))))
        })
    };
    // Cancellation answers at once, and the connect and first-audio bounds hold even while the
    // blocking side waits in DNS, connect or TLS (it notices on its own and ends later).
    loop {
        if cancel.is_set() {
            return Err(SpeakError::Cancelled);
        }
        let now = Instant::now();
        let unreachable = !progress.connected.load(Ordering::Acquire) && now >= connect_deadline + UNREACHABLE_GRACE;
        let silent = !progress.audio.load(Ordering::Acquire) && now >= first_deadline + WATCHDOG_GRACE;
        if unreachable || silent {
            // The blocking side must never push audio after this: the fallback voice may already
            // be speaking into the same utterance.
            registration.flag.store(true, Ordering::Release);
            return Err(if unreachable {
                session::unreachable(connect)
            } else {
                SpeakError::fail(
                    FailureKind::Stalled,
                    format!("{WHAT}: no audio within {} s (the service did not answer in time).", session::secs(first_audio)),
                )
            });
        }
        tokio::select! {
            biased;
            joined = &mut task => {
                return match joined {
                    Ok(result) => result,
                    Err(e) => Err(SpeakError::fail(FailureKind::Other, format!("{WHAT}: the request failed ({e})."))),
                };
            }
            _ = tokio::time::sleep(cloud::CANCEL_POLL) => {}
        }
    }
}

async fn speak(request: EdgeTtsRequest, utterance_id: String) -> Result<SpeakReport, String> {
    let id = utterance_id.trim();
    if id.is_empty() {
        return Err("utteranceId must not be empty".into());
    }
    let result = async {
        let sink = sink_for(id)?;
        speak_with(&request, id, &Endpoint::service(), sink, Arc::new(SystemClock)).await
    }
    .await;
    match result {
        Ok(report) => {
            tracing::debug!(
                target: "sigil::tts",
                "{WHAT}: {} frames, first audio after {} ms, done after {} ms",
                report.frames,
                report.first_audio_ms,
                report.total_ms
            );
            Ok(report)
        }
        Err(SpeakError::Failed(f)) => {
            tracing::info!(target: "sigil::tts", "{} ({})", f.message, f.kind.wire());
            Err(f.to_wire())
        }
        Err(SpeakError::Local(message)) => {
            tracing::info!(target: "sigil::tts", "{WHAT}: {message}");
            Err(message)
        }
        Err(e) => Err(e.into_wire()),
    }
}

// ---------------------------------------------------------------------------------------------
// Commands
// ---------------------------------------------------------------------------------------------

#[command]
async fn edge_tts_speak(request: EdgeTtsRequest, utterance_id: String) -> Result<SpeakReport, String> {
    speak(request, utterance_id).await
}

#[command]
async fn edge_tts_cancel(utterance_id: Option<String>) -> Result<(), String> {
    cancel_command(&REGISTRY, utterance_id.as_deref());
    Ok(())
}

#[command]
async fn edge_tts_list_voices() -> Result<Vec<EdgeTtsVoice>, String> {
    voices::list(crate::services::http_client::http_client(), &Endpoint::service().voices, &SystemClock).await
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .invoke_handler(tauri::generate_handler![edge_tts_speak, edge_tts_cancel, edge_tts_list_voices])
        .build()
}
