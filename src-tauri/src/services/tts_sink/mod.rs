//! The unified TTS audio sink (`tts-sink` plugin; owned by P2-G, API frozen by P2-S0).
//!
//! Every PCM-producing TTS backend plays through this sink: a persistent output stream per device,
//! an ordered utterance queue with cancel / skip / flush, and a top-level route: the output device,
//! the voice changer's aux input (P0 back-pressure, the VC's own aux mic gate), or both.
//!
//! # Utterance model
//! The TS sentence pipeline (P2-G, `src/core/services/tts/pipeline.ts`) owns utterance order and
//! lifetime: it **begins** one utterance per sentence, in sentence order, before synthesis starts
//! (N ahead), and **ends** it when synthesis settled (or **cancels** it on stop / skip). Producers
//! only **push** PCM into a begun utterance: TS backends through the binary `push_pcm` command,
//! in-process Rust engines (local-tts, openai-speech) through [`writer`] / [`UtteranceWriter`].
//! Utterances play strictly in begin order; an utterance starts when the previous one finished and
//! its own prebuffer is filled, and it finishes when it was ended and fully played. Pushes may carry
//! any sample rate / channel count (the sink resamples to the device with rubato); a push blocks
//! for back-pressure (bounded, checking cancellation) and fails with [`SinkError::Cancelled`] once
//! the utterance was cancelled, which tells an engine to stop synthesizing.
//!
//! # Frozen contract (P2-S0; change only through an INTEGRATE step)
//! Types may gain fields / variants; names and wire formats stay. TS mirrors: `RustTtsSink*` and
//! `TTS_SINK_EVENTS` in `src/types.ts`.
//! - Rust API: [`begin`], [`writer`] → [`UtteranceWriter::push`] / [`UtteranceWriter::is_cancelled`],
//!   [`end`], [`cancel`], [`flush`], [`status`]; errors [`SinkError`]; [`SpeakReport`] is what
//!   engines that speak into the sink return over IPC. P6-J added [`apply_mic_policy`] /
//!   [`mic_policy`] ([`MicPolicy`]).
//! - IPC (`plugin-commands/tts-sink.txt`):
//!   - `plugin:tts-sink|begin_utterance { spec: UtteranceSpec }` → `void`
//!   - `plugin:tts-sink|push_pcm` with a raw body of little-endian f32 samples (interleaved) and
//!     the headers [`HEADER_UTTERANCE`], [`HEADER_SAMPLE_RATE`], [`HEADER_CHANNELS`]:
//!     `invoke("plugin:tts-sink|push_pcm", new Uint8Array(f32.buffer), { headers })` → `void`
//!     (no JSON number arrays).
//!   - `plugin:tts-sink|end_utterance { id }` → `void`
//!   - `plugin:tts-sink|cancel_utterance { id?: string | null }` → `void` (`null` = every queued and
//!     playing utterance; skip = cancel the playing one)
//!   - `plugin:tts-sink|flush_sink` → `void` (cancel everything and drop queued audio; devices stay
//!     open)
//!   - `plugin:tts-sink|get_sink_status` → [`SinkStatus`]
//!   - `plugin:tts-sink|set_mic_policy { policy: "none" | "duck" | "mute" }` → `void` (P6-J; an
//!     unknown value is `none`): the microphone while TTS plays into the voice changer
//!     ([`mic_gate`]). TS forwards the voice changer's aux mic gate setting.
//! - Events: [`STATE_EVENT`] (`tts_sink:state`, [`SinkStatus`], at most ~4 Hz) and
//!   [`UTTERANCE_EVENT`] (`tts_sink:utterance`, [`UtteranceEvent`] on every state change).
//! - Real-time rules: the output callback never allocates, locks, logs or does IPC (rtrb rings).
//!
//! # Producers
//! Every TTS backend that makes PCM plays here (P6-J moved Windows SAPI over: `windows_tts`
//! renders each sentence into its utterance). Web Speech (`speechSynthesis`) is the one backend
//! that cannot: the webview plays it on the default output and gives no audio back, so it is
//! device-only and never reaches the voice changer.
//!
//! # Implementation
//! - [`engine`]: the queue, back-pressure (head 30 s / total 120 s of unplayed audio; a push is
//!   admitted whole), the 200 ms prebuffer, the stall timeout (120 s without a push once audio
//!   started; 11 min before the first audio, so slow synthesis plays late) and the scheduler thread
//!   (`sigil-tts-sink`, started on the first utterance), which writes the playing utterance ~200 ms
//!   ahead into the output ring and feeds the voice changer. Utterances join gaplessly.
//! - [`output`]: one persistent cpal stream per device (opened on first use, closed after 5 min
//!   unused, re-opened when the Windows default changes while idle), a ring of slot-tagged samples
//!   so stop / skip drop audio already in the ring at once, click-free joins; device names match
//!   exactly, then ignoring case / Windows renumbering / old 31-character SAPI names; a SAPI
//!   audio-output token id (the Windows voice's saved device) resolves through its endpoint id.
//! - [`convert`]: stateful band-limited resampling (rubato), channel adaptation and gain.
//! - [`vc_route`]: the voice changer's tagged aux queue (`voice_changer::aux_feed`).
//! - [`mic_gate`]: with the `mute` policy, the capture hub's `voice_changer` gate is closed while
//!   TTS audio flows into the voice changer (plus a short hold, or up to 2 s while the reply's next
//!   sentence is still synthesizing), so the microphone never enters it meanwhile; the voice
//!   changer runs on silence plus the TTS. At a reply's onset the gate closes ~120 ms before the
//!   first TTS block goes in, so the stream restart the gate's edge causes in the voice changer
//!   lands on silence. `duck` is the voice changer's own.
//! - Route `voiceChanger` plays on the device while the voice changer is not running; `both` then
//!   plays on the device only.

mod convert;
mod engine;
pub(crate) mod mic_gate;
mod output;
mod vc_route;

#[cfg(test)]
mod tests;

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};

use serde::{Deserialize, Serialize};
use tauri::{
    command,
    ipc::{InvokeBody, Request},
    plugin::{Builder, TauriPlugin},
    AppHandle, Emitter, Runtime,
};

use engine::{SinkConfig, SinkCore, SinkEvents};
pub use mic_gate::MicPolicy;
/// The sink's default bounds (clients that time their requests against them reference these).
pub(crate) use engine::STALL_TIMEOUT;
#[cfg(test)]
pub(crate) use engine::FIRST_AUDIO_TIMEOUT;

/// Plugin name (`plugin:tts-sink|...`, capability `tts-sink:default`).
pub const PLUGIN_NAME: &str = "tts-sink";
/// Event with a [`SinkStatus`] payload.
pub const STATE_EVENT: &str = "tts_sink:state";
/// Event with an [`UtteranceEvent`] payload.
pub const UTTERANCE_EVENT: &str = "tts_sink:utterance";
/// `push_pcm` header: the utterance id.
pub const HEADER_UTTERANCE: &str = "x-sigil-utterance";
/// `push_pcm` header: sample rate of the body (Hz).
pub const HEADER_SAMPLE_RATE: &str = "x-sigil-sample-rate";
/// `push_pcm` header: interleaved channel count of the body.
pub const HEADER_CHANNELS: &str = "x-sigil-channels";

/// Where an utterance plays. Wire: `device` | `voiceChanger` | `both`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SinkRoute {
    /// The output device only.
    #[default]
    Device,
    /// Into the running voice changer's aux input only (TTS speaks with the converted voice).
    VoiceChanger,
    /// Both.
    Both,
}

/// Output settings of one utterance.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct SinkTarget {
    pub route: SinkRoute,
    /// Output device (cpal name as listed by `plugin:audio|list_output_devices`); `""` = the
    /// system default output.
    pub device: String,
    /// Linear gain 0..=2 applied to this utterance.
    pub volume: f32,
}

impl Default for SinkTarget {
    fn default() -> Self {
        Self {
            route: SinkRoute::Device,
            device: String::new(),
            volume: 1.0,
        }
    }
}

/// `begin_utterance` argument.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UtteranceSpec {
    /// Unique id chosen by the caller (the TS pipeline uses `tts-<session>-<seq>`).
    pub id: String,
    #[serde(default)]
    pub target: SinkTarget,
}

/// Lifecycle of an utterance. Wire: lower-case.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum UtteranceState {
    #[default]
    Queued,
    Playing,
    /// Ended and fully played.
    Done,
    Cancelled,
    /// Device / voice-changer failure (see `error`).
    Error,
}

/// Payload of [`UTTERANCE_EVENT`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct UtteranceEvent {
    pub id: String,
    pub state: UtteranceState,
    pub error: Option<String>,
    /// Audio played so far (ms).
    pub played_ms: u32,
}

/// Payload of [`STATE_EVENT`] and result of `get_sink_status`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SinkStatus {
    /// Utterance playing now.
    pub playing: Option<String>,
    /// Begun utterances waiting to play, in order.
    pub queued: Vec<String>,
    /// Buffered audio not yet played (ms).
    pub queued_ms: u32,
    /// Output devices with an open stream.
    pub devices: Vec<String>,
}

/// What an in-process engine (local-tts, openai-speech) returns after speaking into the sink.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakReport {
    /// Sample rate the engine produced.
    pub sample_rate: u32,
    /// Frames pushed.
    pub frames: u64,
    /// Milliseconds from the request to the first pushed audio.
    pub first_audio_ms: u32,
    /// Milliseconds from the request to the last pushed audio.
    pub total_ms: u32,
}

/// Why a sink call failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SinkError {
    /// No begun utterance with this id (never begun, or already ended / finished).
    NotFound(String),
    /// The utterance was cancelled (stop / skip / flush): stop producing audio.
    Cancelled,
    /// Bad sample rate / channel count / body.
    InvalidFormat(String),
    /// Output device or voice-changer route failed.
    Device(String),
    /// The sink is not available (not initialized, too many utterances queued, or it timed out
    /// waiting for queued speech to play).
    Unavailable(String),
}

impl fmt::Display for SinkError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SinkError::NotFound(id) => write!(f, "no TTS utterance {id:?}"),
            SinkError::Cancelled => f.write_str("cancelled"),
            SinkError::InvalidFormat(m) => write!(f, "invalid audio: {m}"),
            SinkError::Device(m) => write!(f, "audio output: {m}"),
            SinkError::Unavailable(m) => write!(f, "TTS sink unavailable: {m}"),
        }
    }
}

impl std::error::Error for SinkError {}

impl From<SinkError> for String {
    fn from(e: SinkError) -> Self {
        e.to_string()
    }
}

/// The app's sink, created by the plugin's setup.
static SINK: OnceLock<Arc<SinkCore>> = OnceLock::new();

fn global() -> Result<&'static Arc<SinkCore>, SinkError> {
    SINK.get()
        .ok_or_else(|| SinkError::Unavailable("the TTS sink is not initialized".into()))
}

/// Begin an utterance (queued after every begun one). Fails for a duplicate or empty id.
pub fn begin(spec: UtteranceSpec) -> Result<(), SinkError> {
    let sink = global()?;
    sink.ensure_thread()?;
    sink.begin(spec)
}

/// A writer for a begun utterance (in-process engines).
pub fn writer(id: &str) -> Result<UtteranceWriter, SinkError> {
    let sink = global()?;
    let cancelled = sink.cancel_flag(id)?;
    Ok(UtteranceWriter {
        id: id.to_string(),
        sink: Arc::clone(sink),
        cancelled,
    })
}

/// Mark an utterance complete: it finishes once its queued audio played.
pub fn end(id: &str) -> Result<(), SinkError> {
    global()?.end(id)
}

/// Cancel one utterance, or (`None`) every queued and playing one.
pub fn cancel(id: Option<&str>) {
    if let Ok(sink) = global() {
        sink.cancel(id);
    }
}

/// Cancel everything and drop buffered audio (devices stay open).
pub fn flush() {
    if let Ok(sink) = global() {
        sink.flush();
    }
}

/// Current state.
pub fn status() -> SinkStatus {
    global().map(|s| s.status()).unwrap_or_default()
}

/// Set the microphone policy while TTS plays into the voice changer ([`mic_gate`]).
pub fn apply_mic_policy(policy: MicPolicy) -> Result<(), SinkError> {
    global()?.set_mic_policy(policy);
    Ok(())
}

/// The microphone policy in force ([`MicPolicy::None`] before the sink exists).
#[allow(dead_code)] // public API for diagnostics and other engines
pub fn mic_policy() -> MicPolicy {
    global().map(|s| s.mic_policy()).unwrap_or_default()
}

/// Pushes PCM into one begun utterance.
pub struct UtteranceWriter {
    id: String,
    sink: Arc<SinkCore>,
    cancelled: Arc<AtomicBool>,
}

impl fmt::Debug for UtteranceWriter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UtteranceWriter")
            .field("id", &self.id)
            .field("cancelled", &self.is_cancelled())
            .finish()
    }
}

impl UtteranceWriter {
    // Frozen API for in-process engines (local-tts, openai-speech).
    #[allow(dead_code)]
    pub fn id(&self) -> &str {
        &self.id
    }

    /// Queue interleaved f32 samples. Blocks (bounded) for back-pressure; `Err(Cancelled)` once the
    /// utterance was cancelled. Never call it while holding a lock that the producer of an
    /// *earlier* utterance needs (the earlier one must be able to finish for room to appear).
    pub fn push(&mut self, samples: &[f32], sample_rate: u32, channels: u16) -> Result<(), SinkError> {
        if self.is_cancelled() {
            return Err(SinkError::Cancelled);
        }
        self.sink.push(&self.id, samples, sample_rate, channels)
    }

    /// True once the utterance was cancelled (engines poll it between chunks).
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------------------------
// IPC
// ---------------------------------------------------------------------------------------------

/// A parsed `push_pcm` request.
#[derive(Debug, PartialEq)]
struct PcmPush {
    id: String,
    sample_rate: u32,
    channels: u16,
    samples: Vec<f32>,
}

fn header<'a>(headers: &'a tauri::http::HeaderMap, name: &str) -> Result<&'a str, SinkError> {
    headers
        .get(name)
        .ok_or_else(|| SinkError::InvalidFormat(format!("missing header {name}")))?
        .to_str()
        .map(str::trim)
        .map_err(|_| SinkError::InvalidFormat(format!("header {name} is not text")))
}

/// Little-endian f32 samples from the raw body (whole samples only).
fn f32_le(bytes: &[u8]) -> Result<Vec<f32>, SinkError> {
    if !bytes.len().is_multiple_of(4) {
        return Err(SinkError::InvalidFormat(format!("{} bytes are not whole f32 samples", bytes.len())));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

fn parse_push(headers: &tauri::http::HeaderMap, body: &InvokeBody) -> Result<PcmPush, SinkError> {
    let id = header(headers, HEADER_UTTERANCE)?.to_string();
    let sample_rate: u32 = header(headers, HEADER_SAMPLE_RATE)?
        .parse()
        .map_err(|_| SinkError::InvalidFormat(format!("bad {HEADER_SAMPLE_RATE}")))?;
    let channels: u16 = header(headers, HEADER_CHANNELS)?
        .parse()
        .map_err(|_| SinkError::InvalidFormat(format!("bad {HEADER_CHANNELS}")))?;
    let samples = match body {
        InvokeBody::Raw(bytes) => f32_le(bytes)?,
        // The postMessage IPC fallback delivers a byte array as a JSON array of numbers.
        InvokeBody::Json(serde_json::Value::Array(values)) => {
            let bytes: Option<Vec<u8>> = values
                .iter()
                .map(|v| v.as_u64().and_then(|n| u8::try_from(n).ok()))
                .collect();
            f32_le(&bytes.ok_or_else(|| SinkError::InvalidFormat("the body is not a byte array".into()))?)?
        }
        InvokeBody::Json(_) => {
            return Err(SinkError::InvalidFormat(
                "expected a binary body of little-endian f32 samples".into(),
            ));
        }
    };
    Ok(PcmPush {
        id,
        sample_rate,
        channels,
        samples,
    })
}

#[command]
async fn begin_utterance(spec: UtteranceSpec) -> Result<(), String> {
    begin(spec).map_err(String::from)
}

/// Raw little-endian f32 body plus the `x-sigil-*` headers (no JSON number arrays). Resolves once
/// the audio was queued (after waiting for room when the queue is full).
#[command]
async fn push_pcm(request: Request<'_>) -> Result<(), String> {
    let push = parse_push(request.headers(), request.body())?;
    let sink = Arc::clone(global()?);
    tauri::async_runtime::spawn_blocking(move || sink.push_owned(&push.id, push.samples, push.sample_rate, push.channels))
        .await
        .map_err(|e| format!("TTS sink push task: {e}"))?
        .map_err(String::from)
}

#[command]
async fn end_utterance(id: String) -> Result<(), String> {
    end(&id).map_err(String::from)
}

#[command]
async fn cancel_utterance(id: Option<String>) -> Result<(), String> {
    cancel(id.as_deref());
    Ok(())
}

#[command]
async fn flush_sink() -> Result<(), String> {
    flush();
    Ok(())
}

#[command]
async fn get_sink_status() -> Result<SinkStatus, String> {
    Ok(status())
}

/// `policy`: `none` | `duck` | `mute` (anything else is `none`).
#[command]
async fn set_mic_policy(policy: String) -> Result<(), String> {
    apply_mic_policy(MicPolicy::from_str_lossy(&policy)).map_err(String::from)
}

/// Sink events as Tauri events.
struct TauriEvents<R: Runtime>(AppHandle<R>);

impl<R: Runtime> SinkEvents for TauriEvents<R> {
    fn utterance(&self, event: &UtteranceEvent) {
        let _ = self.0.emit(UTTERANCE_EVENT, event.clone());
    }

    fn status(&self, status: &SinkStatus) {
        let _ = self.0.emit(STATE_EVENT, status.clone());
    }
}

/// Plugin setup: create the sink (its scheduler thread and output streams start on first use).
fn setup<R: Runtime>(app: &AppHandle<R>) {
    let core = SinkCore::new(
        SinkConfig::default(),
        Box::new(output::CpalBackend),
        Box::new(vc_route::AppVc(app.clone())),
        Box::new(TauriEvents(app.clone())),
        Box::new(mic_gate::HubVoiceChangerGate),
    );
    if SINK.set(core).is_err() {
        tracing::warn!(target: "sigil::tts_sink", "the TTS sink was already initialized");
    }
}

pub fn init<R: Runtime>() -> TauriPlugin<R> {
    Builder::new(PLUGIN_NAME)
        .setup(|app, _api| {
            setup(app);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            begin_utterance,
            push_pcm,
            end_utterance,
            cancel_utterance,
            flush_sink,
            get_sink_status,
            set_mic_policy,
        ])
        .build()
}
