//! Keyed speech pipelines and the `speech` IPC commands (P2-A).
//!
//! ```text
//! capture hub consumer (raw 16 kHz mono, before the voice changer; mute-gated per group)
//!   -> capture thread: Silero segmenter (speech_core), segment framing, Smart Turn at pauses
//!        -> finals-only route: ordered utterance worker (latency budget, backlog cap, merge on
//!           backlog, language ID, CPU fallback)                         -> speech:final
//!        -> streaming route:   stream worker (partials, first-pass final) -> speech:partial / final
//!             -> two-pass:     `two_pass::TwoPass` re-decode (P2-C)     -> speech:final (second)
//!        -> capture-only:      retained PCM for `take_pipeline_utterance`
//! ```
//!
//! # Sessions, epochs and segment ids
//! One pipeline runs per key (`pipelineId`: `mic`, `loopback`, ...). Every start and stop bumps the
//! key's epoch while holding the key's emit lock, and every event is emitted while holding that
//! lock with the session's epoch still current, so once `stop_pipeline` (or a restart) returns,
//! no event of the ended session is ever delivered (plan: epoch discipline). A stop also cancels
//! the starts of its key that were invoked before it and still wait behind a model load (a start
//! ticket), so nothing of the key runs after `stop_pipeline` returns. `stop_pipeline` with an
//! `epoch` stops only that session, while it is still the key's current one, and cancels no start
//! (a driver cleaning up after a start it gave up on never cancels a newer one). The last event of an
//! epoch is `speech:status` `stopped` (stop / restart) or `error` (the pipeline stopped itself).
//! Engines see the session's liveness and each utterance's latency deadline through
//! `worker::scope`.
//! `start_pipeline` resolves only once the engines are loaded and the capture runs (or with the
//! error); its progress arrives as `speech:status` of the new epoch, which may be delivered before
//! the command's result, so the TS driver accepts the first epoch above the one it last stopped.
//!
//! Segment ids come from a per-key counter that keeps counting across sessions, so a
//! `(pipelineId, segmentId)` pair is never reused. Per segment: `speech_start` (VAD onset,
//! confirmed after `min_speech_ms`) → `utterance` (closed) → optional `turn_end` → finals; or
//! `dropped` instead of `utterance` when the STT mute starts while it is open. With an engine,
//! every closed utterance ends with at least one final or exactly one `dropped` (reasons in
//! `utterance::reasons`: `backlog`, `latency_budget`, `no_speech`, `merged`, `engine_error`, ...),
//! so the driver can always clear its "listening" indicator. Revisions rise with every partial and
//! final of a segment (see `speech::events`).
//!
//! # Latency budget and backlog
//! Closed utterances wait for the utterance engine in a drop-oldest queue of `maxBacklog`; one that
//! waited longer than `latencyBudgetMs` (default 20 s) is dropped instead of decoded late. When the
//! engine falls behind, the waiting utterances are decoded together as one batch (≤ 28 s, 200 ms of
//! silence between them; the first segment's final carries the text and the last segment's end
//! and turn verdict, the others end with `dropped` / `merged`), which is also how the HTTP engine
//! stays inside Groq's request limits (it may hold a short batch until the billing minimum is
//! reached, see `engines::http`; a rate-limit wait that would pass the budget drops the batch as
//! `rate_limited` instead of decoding it late). A
//! streaming engine that falls more than 3 s behind skips partials for the current utterance,
//! which then gets only the two-pass final.
//!
//! # Language
//! An explicit `language` goes to every engine. With `"auto"`, utterance engines detect it
//! themselves unless the route has a language-ID engine (run on each closed utterance first); a
//! streaming engine that needs a language uses the latest detected one. Every engine gets only a
//! language it lists (or any when it lists none). Before a usable language is known (none detected
//! yet, or one the streaming engine does not list) such a stream is skipped when a two-pass final
//! covers the utterance (and the spec names no fallback `options.language`); otherwise it runs in
//! the engine's own auto mode.
//!
//! # Capture-only pipelines
//! `captureOnly: true` (or a config that resolves no engine and sets `retainPcm`) only segments:
//! `speech_start` / `utterance` / `dropped` (muted) events, and with `retainPcm` the PCM through
//! `take_pipeline_utterance`. The explicit flag never asks the router, so such a pipeline stays
//! engine-free whatever the router would plan for an empty request.
//!
//! # Frozen signatures (P2-S0)
//! The command names, argument names and result types below are the IPC contract (see
//! `speech/mod.rs`). Crate-internal: [`start_with`] / [`stop_pipeline_key`] let the legacy
//! `whisper` plugin run its `start_recording` through a pipeline with its own event names.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex, MutexGuard, OnceLock, PoisonError, Weak};
use std::thread::{self, Thread};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use tauri::{command, AppHandle, Runtime};
use tokio::sync::oneshot;

use super::engine_api::{
    AsrEngine, AsrError, AsrErrorKind, AsrResult, Canceller, DecodeOptions, EngineContext, EngineInfo,
    EngineKind, EngineSpec, EngineStatus, PipelineConfig, RoutePlan, RouteRequest, StatusLevel, StatusSink, Tier,
};
use super::events::{
    self, codes, FinalPass, SpeechFinal, SpeechPartial, SpeechState, SpeechStatus, SpeechTurnEnd, SpeechUtterance,
};
use super::mute::{self, gap_action, GapAction};
use super::two_pass::{TwoPass, TwoPassConfig, TwoPassOutcome};
use super::utterance::{
    ms_to_samples, reasons, samples_to_ms, tail, AudioHistory, SegmentBook, SessionClock, TURN_WINDOW_MS,
};
use super::worker::{
    self, BatchPolicy, ClosedSegment, Fallback, SecondPass, SecondPassSetup, StreamMsg, StreamWorkerSetup,
    UtteranceJob, UtteranceWorkerSetup,
};
use super::{engines, lang, router, smart_turn};
use crate::services::audio::hub::{self, CaptureConsumer, ConsumerOptions, MuteGroup, SourceSpec};
use crate::services::speech_core::hallucination::clean_transcript;
use crate::services::speech_core::utterance::{f32_to_le_bytes, BoundedQueue, UtteranceStore};
use crate::services::speech_core::{SegmenterEvent, Segmenter, SpeechSegmenter, VadConfig};

/// Log target of the speech pipeline.
pub(crate) const LOG: &str = "sigil::speech";

/// Default `latencyBudgetMs`.
pub(crate) const DEFAULT_LATENCY_BUDGET_MS: u32 = 20_000;
const MIN_LATENCY_BUDGET_MS: u32 = 1_000;
const MAX_LATENCY_BUDGET_MS: u32 = 120_000;
/// Pipeline keys a process may use (a typo in a driver must not grow the registry forever).
const MAX_PIPELINES: usize = 16;
/// Keys reserved for crate-internal pipelines (the legacy Whisper recording).
pub(crate) const INTERNAL_PREFIX: &str = "legacy-";
/// Capture thread wake-up period when no audio arrives (the hub wakes it every ~20 ms of audio).
const IDLE_WAIT: Duration = Duration::from_millis(100);
/// Loopback: silence after which idle time is fed to the VAD as zeros (WASAPI loopback delivers
/// nothing while nothing plays, and an utterance cut off by that silence must still close).
const LOOPBACK_IDLE_AFTER: Duration = Duration::from_millis(300);
/// Loopback: at least this much synthetic silence per idle period (more when `min_silence_ms`
/// needs it, see [`loopback_idle_fill`]).
const LOOPBACK_IDLE_FILL: Duration = Duration::from_millis(1_500);
/// Loopback: synthetic silence beyond `min_silence_ms` (the segmenter's end pad plus a frame or
/// two of slack).
const LOOPBACK_IDLE_MARGIN_MS: u64 = 300;
/// Bound on waiting for the capture thread to release the device at stop.
const CAPTURE_STOP_TIMEOUT: Duration = Duration::from_secs(3);
/// Bound on waiting for the workers (and their models) to go away at stop.
const WORKER_STOP_TIMEOUT: Duration = Duration::from_secs(2);
/// A streaming engine this far behind skips partials for the current utterance.
pub(crate) const STREAM_MAX_LAG_MS: u32 = 3_000;
/// Minimum spacing of repeated warnings with the same code.
const WARN_INTERVAL: Duration = Duration::from_secs(5);
/// Longest prompt passed to engines (bytes; engines cap further).
const MAX_PROMPT_BYTES: usize = 1024;
/// Hotwords passed to engines, and their length.
const MAX_HOTWORDS: usize = 100;
const MAX_HOTWORD_CHARS: usize = 64;

// ---------------------------------------------------------------------------------------------
// Frozen IPC types
// ---------------------------------------------------------------------------------------------

/// Result of `start_pipeline`.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineStarted {
    pub pipeline_id: String,
    /// Epoch of the new session: every event of this session carries it.
    pub epoch: u64,
    /// The engines the pipeline resolved (router plan, or the explicit engines of the config).
    pub plan: RoutePlan,
}

/// One entry of `get_pipeline_status`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PipelineStatus {
    pub pipeline_id: String,
    pub epoch: u64,
    pub state: SpeechState,
    /// Capture source in use.
    pub source: String,
    pub streaming: Option<EngineInfo>,
    #[serde(rename = "final")]
    pub final_engine: Option<EngineInfo>,
    /// Closed utterances waiting for a decode.
    #[serde(default)]
    pub backlog: u32,
    /// Utterances dropped by the budget / backlog cap in this session.
    #[serde(default)]
    pub dropped: u32,
}

// ---------------------------------------------------------------------------------------------
// Injection points (production implementations below; tests swap them)
// ---------------------------------------------------------------------------------------------

/// Where a session's events go. [`TauriSink`] emits the `speech:*` events; the legacy Whisper
/// recording maps them to its `whisper:*` events. Called with the key's emit lock held: keep it
/// quick and never call back into the pipeline.
pub(crate) trait EventSink: Send + Sync + 'static {
    fn partial(&self, event: &SpeechPartial);
    fn final_transcript(&self, event: &SpeechFinal);
    fn status(&self, event: &SpeechStatus);
    /// `audio_ms`: length of the utterance audio (0 when unknown).
    fn utterance(&self, event: &SpeechUtterance, audio_ms: u64);
    fn turn_end(&self, event: &SpeechTurnEnd);
}

/// The `speech:*` Tauri events.
pub(crate) struct TauriSink<R: Runtime> {
    app: AppHandle<R>,
}

impl<R: Runtime> TauriSink<R> {
    pub(crate) fn new(app: AppHandle<R>) -> Self {
        Self { app }
    }
}

impl<R: Runtime> EventSink for TauriSink<R> {
    fn partial(&self, event: &SpeechPartial) {
        events::emit_partial(&self.app, event);
    }
    fn final_transcript(&self, event: &SpeechFinal) {
        events::emit_final(&self.app, event);
    }
    fn status(&self, event: &SpeechStatus) {
        events::emit_status(&self.app, event);
    }
    fn utterance(&self, event: &SpeechUtterance, _audio_ms: u64) {
        events::emit_utterance(&self.app, event);
    }
    fn turn_end(&self, event: &SpeechTurnEnd) {
        events::emit_turn_end(&self.app, event);
    }
}

/// One read of an [`AudioSource`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SourceRead {
    /// The audio is discontinuous right after the samples of this read (restart, device move,
    /// mute edge, overrun): close or drop the open utterance now.
    pub gap_after: bool,
    /// The discontinuity is a stream restart or a move to another device.
    pub restarted: bool,
    /// QPC capture time (ns) just past the newest sample read.
    pub end_qpc_ns: Option<i64>,
}

/// 16 kHz mono capture for one session (production: a capture-hub consumer).
pub(crate) trait AudioSource: Send {
    /// Append everything available (up to the next discontinuity) to `out`; never blocks.
    fn read(&mut self, out: &mut Vec<f32>) -> SourceRead;
    /// Park the calling thread until audio arrives, at most `timeout` (an `unpark` ends it early).
    fn wait(&self, timeout: Duration);
    /// The source's mute group is muted now.
    fn muted(&self) -> bool;
    /// The capture is over for good (the error text); only the unsupervised legacy hub ends.
    fn ended(&self) -> Option<String>;
    fn is_loopback(&self) -> bool;
    /// Source key for status (`default`, a device name, `loopback:<output>`).
    fn key(&self) -> String;
    /// Frames dropped because the capture thread fell behind.
    fn dropped_frames(&self) -> u64;
    /// Release the device (blocking, bounded).
    fn close(self: Box<Self>);
}

/// The production source: a capture-hub consumer gated by the source's mute group.
struct HubSource {
    consumer: CaptureConsumer,
    group: MuteGroup,
}

impl AudioSource for HubSource {
    fn read(&mut self, out: &mut Vec<f32>) -> SourceRead {
        let info = self.consumer.read(out);
        SourceRead {
            gap_after: info.gap_after,
            restarted: info.restarted,
            end_qpc_ns: info.end_qpc_ns,
        }
    }
    fn wait(&self, timeout: Duration) {
        self.consumer.wait(timeout);
    }
    fn muted(&self) -> bool {
        mute::hub_muted(self.group)
    }
    fn ended(&self) -> Option<String> {
        self.consumer.ended()
    }
    fn is_loopback(&self) -> bool {
        self.consumer.is_loopback()
    }
    fn key(&self) -> String {
        self.consumer.source().to_string()
    }
    fn dropped_frames(&self) -> u64 {
        self.consumer.dropped_frames() + self.consumer.producer_dropped_frames()
    }
    fn close(self: Box<Self>) {
        self.consumer.close_and_wait();
    }
}

/// Smart Turn at a pause (production: `smart_turn::SmartTurn`, P2-C).
pub(crate) trait TurnDetector {
    /// Turn-end probability 0..1 for 16 kHz audio that ends now.
    fn predict(&mut self, audio_16k: &[f32]) -> Result<f32, String>;
}

impl TurnDetector for smart_turn::SmartTurn {
    fn predict(&mut self, audio_16k: &[f32]) -> Result<f32, String> {
        smart_turn::SmartTurn::predict(self, audio_16k)
    }
}

impl SecondPass for TwoPass {
    fn submit(&self, job: super::two_pass::FinalJob) -> bool {
        TwoPass::submit(self, job)
    }
    fn shutdown(self: Box<Self>) {
        TwoPass::shutdown(*self);
    }
}

/// Creates an engine (blocking: loads the model). Production: [`engines::create`].
pub(crate) type EngineFactory =
    Arc<dyn Fn(&EngineSpec, &EngineContext) -> Result<Box<dyn AsrEngine>, AsrError> + Send + Sync>;
/// Attaches the capture source (blocking: may open the device).
pub(crate) type SourceFactory = Box<dyn FnOnce(&str) -> Result<Box<dyn AudioSource>, String> + Send>;
/// Builds the segmenter on the capture thread (loads Silero).
pub(crate) type SegmenterFactory = Box<dyn FnOnce(&VadConfig) -> Result<Box<dyn SpeechSegmenter>, String> + Send>;
/// Loads Smart Turn on the capture thread.
pub(crate) type TurnFactory = Box<dyn FnOnce() -> Result<Box<dyn TurnDetector>, String> + Send>;
/// Callback of the two-pass queue.
pub(crate) type OutcomeFn = Box<dyn FnMut(TwoPassOutcome) + Send>;
/// Starts the two-pass queue on the stream worker thread (production: `TwoPass::spawn`).
pub(crate) type SecondPassFactory = Box<
    dyn FnOnce(
            Box<dyn AsrEngine>,
            Option<Box<dyn AsrEngine>>,
            TwoPassConfig,
            OutcomeFn,
        ) -> Result<Box<dyn SecondPass>, AsrError>
        + Send,
>;
/// The router (production: `router::plan`, P2-C).
pub(crate) type RouteFn = Box<dyn FnOnce(&RouteRequest) -> RoutePlan + Send>;
/// Catalog ids an explicit engine needs, and whether each is installed.
pub(crate) type RequirementsFn = Arc<dyn Fn(&EngineSpec) -> Vec<(String, bool)> + Send + Sync>;
/// Request pacing an utterance engine asks the worker for (HTTP / Groq).
pub(crate) type BatchPolicyFn = Arc<dyn Fn(&EngineSpec) -> Option<Arc<dyn BatchPolicy>> + Send + Sync>;

/// Everything a session is built from. [`SessionDeps::production`] is the app's.
pub(crate) struct SessionDeps {
    pub sink: Arc<dyn EventSink>,
    pub engines: EngineFactory,
    pub source: SourceFactory,
    pub segmenter: SegmenterFactory,
    pub turn: TurnFactory,
    pub second_pass: SecondPassFactory,
    pub route: RouteFn,
    pub requirements: RequirementsFn,
    pub batch_policy: BatchPolicyFn,
    /// Segment times follow the capture clock (QPC) across gaps, zeroed when the capture starts;
    /// `false` = audio time only (tests).
    pub capture_clock: bool,
}

impl SessionDeps {
    /// The app's dependencies: capture hub, Silero, `engines::create`, Smart Turn, `TwoPass`, the
    /// router and the model store.
    pub(crate) fn production(sink: Arc<dyn EventSink>) -> Self {
        Self {
            sink,
            engines: Arc::new(engines::create),
            source: Box::new(attach_hub_source),
            segmenter: Box::new(|vad: &VadConfig| {
                Ok(Box::new(Segmenter::new(vad.clone())?) as Box<dyn SpeechSegmenter>)
            }),
            turn: Box::new(|| Ok(Box::new(smart_turn::SmartTurn::load()?) as Box<dyn TurnDetector>)),
            second_pass: Box::new(|primary, fallback, config, on_outcome| {
                TwoPass::spawn(primary, fallback, config, on_outcome).map(|p| Box::new(p) as Box<dyn SecondPass>)
            }),
            route: Box::new(router::plan),
            requirements: Arc::new(engines::requirements),
            batch_policy: Arc::new(engines::batch_policy),
            capture_clock: true,
        }
    }

    /// Replace the engine factory (the legacy Whisper recording shares its loaded model).
    pub(crate) fn with_engines(mut self, engines: EngineFactory) -> Self {
        self.engines = engines;
        self
    }
}

fn attach_hub_source(source: &str) -> Result<Box<dyn AudioSource>, String> {
    let spec = SourceSpec::parse(Some(source));
    let group = mute::mute_group(source);
    let opts = ConsumerOptions::speech("speech").with_mute_group(Some(group));
    let consumer = hub::global()?.attach(spec, opts)?;
    Ok(Box::new(HubSource { consumer, group }))
}

// ---------------------------------------------------------------------------------------------
// Registry, slots and the emit gate
// ---------------------------------------------------------------------------------------------

/// A key's epoch and the lock every emit of that key holds.
pub(crate) struct Gate {
    lock: Mutex<()>,
    epoch: AtomicU64,
}

/// Held emit lock of a [`Gate`].
pub(crate) struct GateGuard<'a> {
    gate: &'a Gate,
    _guard: MutexGuard<'a, ()>,
}

impl Gate {
    fn new() -> Self {
        Self {
            lock: Mutex::new(()),
            epoch: AtomicU64::new(0),
        }
    }

    fn lock(&self) -> GateGuard<'_> {
        GateGuard {
            gate: self,
            _guard: self.lock.lock().unwrap_or_else(PoisonError::into_inner),
        }
    }

    fn current(&self) -> u64 {
        self.epoch.load(Ordering::SeqCst)
    }
}

impl GateGuard<'_> {
    fn current(&self) -> u64 {
        self.gate.current()
    }

    /// Start a new epoch (the previous one's events are refused from now on).
    fn bump(&self) -> u64 {
        self.gate.epoch.fetch_add(1, Ordering::SeqCst) + 1
    }
}

/// State of a key shared with its sessions (no reference back to them).
pub(crate) struct SlotShared {
    id: String,
    gate: Gate,
    /// Next segment id (never reused within the process).
    next_segment: AtomicU64,
    /// Retained PCM of closed utterances (`retainPcm`), keyed by `(epoch, segment id)`: a stop
    /// or restart clears only the epochs it ended, and only a current session inserts.
    pcm: Mutex<UtteranceStore>,
}

impl SlotShared {
    fn next_segment_id(&self) -> u64 {
        self.next_segment.fetch_add(1, Ordering::SeqCst)
    }

    fn pcm(&self) -> MutexGuard<'_, UtteranceStore> {
        self.pcm.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// One pipeline key.
struct PipelineSlot {
    shared: Arc<SlotShared>,
    /// The running session (installed once fully started).
    session: Mutex<Option<Session>>,
    /// The context of the current epoch (starting or running): ends it with `stopped`.
    live: Mutex<Option<Arc<SessionCtx>>>,
    /// Serializes session builds of this key (a restart waits for a model load in progress
    /// instead of loading a second copy next to it). Stops never take it.
    ops: tokio::sync::Mutex<()>,
    /// Tickets of `start_session` calls, taken before waiting for `ops`.
    start_tickets: AtomicU64,
    /// The newest start ticket issued when the last stop ran (under the emit lock): a start with
    /// a ticket at or below it was overtaken by that stop and gives up once it gets `ops`, so
    /// nothing of the key runs after `stop_pipeline` returns, even a start that was still queued.
    stopped_through: AtomicU64,
}

impl PipelineSlot {
    fn take_session(&self) -> Option<Session> {
        self.session.lock().unwrap_or_else(PoisonError::into_inner).take()
    }

    fn live(&self) -> MutexGuard<'_, Option<Arc<SessionCtx>>> {
        self.live.lock().unwrap_or_else(PoisonError::into_inner)
    }
}

/// All pipeline keys of the process (tests build their own).
pub(crate) struct Registry {
    slots: Mutex<HashMap<String, Arc<PipelineSlot>>>,
}

impl Registry {
    pub(crate) fn new() -> Self {
        Self {
            slots: Mutex::new(HashMap::new()),
        }
    }

    fn slots(&self) -> MutexGuard<'_, HashMap<String, Arc<PipelineSlot>>> {
        self.slots.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn slot(&self, id: &str) -> Result<Arc<PipelineSlot>, String> {
        let mut slots = self.slots();
        if let Some(s) = slots.get(id) {
            return Ok(Arc::clone(s));
        }
        if slots.len() >= MAX_PIPELINES {
            return Err(format!("too many speech pipelines (at most {MAX_PIPELINES} keys)"));
        }
        let slot = Arc::new(PipelineSlot {
            shared: Arc::new(SlotShared {
                id: id.to_string(),
                gate: Gate::new(),
                next_segment: AtomicU64::new(1),
                pcm: Mutex::new(UtteranceStore::default()),
            }),
            session: Mutex::new(None),
            live: Mutex::new(None),
            ops: tokio::sync::Mutex::new(()),
            start_tickets: AtomicU64::new(0),
            stopped_through: AtomicU64::new(0),
        });
        slots.insert(id.to_string(), Arc::clone(&slot));
        Ok(slot)
    }

    fn existing(&self, id: &str) -> Option<Arc<PipelineSlot>> {
        self.slots().get(id).cloned()
    }

    /// Status of every running pipeline.
    pub(crate) fn statuses(&self) -> Vec<PipelineStatus> {
        let slots: Vec<Arc<PipelineSlot>> = self.slots().values().cloned().collect();
        let mut out: Vec<PipelineStatus> = slots
            .iter()
            .filter_map(|slot| {
                let session = slot.session.lock().unwrap_or_else(PoisonError::into_inner);
                session.as_ref().filter(|s| s.ctx.is_current()).map(Session::status)
            })
            .collect();
        out.sort_by(|a, b| a.pipeline_id.cmp(&b.pipeline_id));
        out
    }

    /// Take the retained PCM of a closed utterance (once; segment ids are unique per key).
    pub(crate) fn take_pcm(&self, pipeline_id: &str, segment_id: u64) -> Option<Vec<f32>> {
        self.existing(pipeline_id)?.shared.pcm().take_seq(segment_id)
    }
}

/// The app's pipelines.
pub(crate) fn registry() -> &'static Registry {
    static REGISTRY: OnceLock<Registry> = OnceLock::new();
    REGISTRY.get_or_init(Registry::new)
}

// ---------------------------------------------------------------------------------------------
// Session context: every emit of a session goes through it
// ---------------------------------------------------------------------------------------------

/// Resolved per-session settings.
#[derive(Clone, Debug)]
pub(crate) struct SessionSettings {
    /// Explicit language (normalized); `None` = auto.
    pub language: Option<String>,
    pub hotwords: Vec<String>,
    pub prompt: Option<String>,
    pub hallucination_filter: bool,
    pub latency_budget: Duration,
    pub max_backlog: usize,
    pub retain_pcm: bool,
    pub smart_turn: bool,
    pub vad: VadConfig,
    pub source: String,
}

/// Optional parts of a status event.
#[derive(Clone, Debug, Default)]
pub(crate) struct StatusExtra {
    pub progress: Option<f32>,
    pub model_id: Option<String>,
    pub engine: Option<String>,
    pub device: Option<String>,
}

/// What a final carries besides its text.
#[derive(Clone, Debug)]
pub(crate) struct FinalMeta {
    pub lang: Option<String>,
    pub tags: Vec<String>,
    pub pass: FinalPass,
    /// `EngineKind` wire name.
    pub engine: String,
    pub model: String,
    /// A merged batch: the last segment of it (the final spans to its end).
    pub merged_until: Option<u64>,
}

/// One session of one key. Shared by the capture thread, the workers and the engines' status
/// sinks; holds nothing that refers back to them.
pub(crate) struct SessionCtx {
    pub(crate) pipeline_id: String,
    pub(crate) epoch: u64,
    shared: Arc<SlotShared>,
    slot: Weak<PipelineSlot>,
    sink: Arc<dyn EventSink>,
    book: Mutex<SegmentBook>,
    state: Mutex<SpeechState>,
    pub(crate) settings: SessionSettings,
    /// Latest detected language (auto mode), for streaming engines that need one.
    sticky_language: Mutex<Option<String>>,
    dropped: AtomicU32,
    backlog: AtomicU32,
    last_warn: Mutex<HashMap<String, Instant>>,
}

impl SessionCtx {
    fn new(slot: &Arc<PipelineSlot>, epoch: u64, sink: Arc<dyn EventSink>, settings: SessionSettings) -> Self {
        Self {
            pipeline_id: slot.shared.id.clone(),
            epoch,
            shared: Arc::clone(&slot.shared),
            slot: Arc::downgrade(slot),
            sink,
            book: Mutex::new(SegmentBook::default()),
            state: Mutex::new(SpeechState::Starting),
            settings,
            sticky_language: Mutex::new(None),
            dropped: AtomicU32::new(0),
            backlog: AtomicU32::new(0),
            last_warn: Mutex::new(HashMap::new()),
        }
    }

    fn book(&self) -> MutexGuard<'_, SegmentBook> {
        self.book.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// This session's epoch is still the key's current one.
    pub(crate) fn is_current(&self) -> bool {
        self.shared.gate.current() == self.epoch
    }

    fn state(&self) -> SpeechState {
        *self.state.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Run `f` with the emit lock held while this session is current.
    fn with_gate<T>(&self, f: impl FnOnce(&GateGuard<'_>) -> T) -> Option<T> {
        let g = self.shared.gate.lock();
        (g.current() == self.epoch).then(|| f(&g))
    }

    fn status_event(&self, state: SpeechState, code: &str, detail: String, level: StatusLevel, extra: StatusExtra) -> SpeechStatus {
        SpeechStatus {
            pipeline_id: self.pipeline_id.clone(),
            epoch: self.epoch,
            state,
            code: code.to_string(),
            detail,
            progress: extra.progress,
            level,
            model_id: extra.model_id,
            engine: extra.engine,
            device: extra.device,
        }
    }

    /// Emit a status (and move to `state` when given).
    pub(crate) fn status(
        &self,
        state: Option<SpeechState>,
        code: &str,
        detail: impl Into<String>,
        level: StatusLevel,
        extra: StatusExtra,
    ) {
        let detail = detail.into();
        self.with_gate(|_| {
            let state = {
                let mut s = self.state.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(new) = state {
                    *s = new;
                }
                *s
            };
            self.sink.status(&self.status_event(state, code, detail, level, extra));
        });
    }

    /// A warning, at most once per [`WARN_INTERVAL`] per code (always logged at debug).
    pub(crate) fn warn(&self, code: &str, detail: impl Into<String>) {
        let detail = detail.into();
        tracing::debug!(target: LOG, "[{}] {code}: {detail}", self.pipeline_id);
        {
            let mut last = self.last_warn.lock().unwrap_or_else(PoisonError::into_inner);
            if last.get(code).is_some_and(|t| t.elapsed() < WARN_INTERVAL) {
                return;
            }
            last.insert(code.to_string(), Instant::now());
        }
        tracing::warn!(target: LOG, "[{}] {code}: {detail}", self.pipeline_id);
        self.status(None, code, detail, StatusLevel::Warning, StatusExtra::default());
    }

    /// Forward an engine's load / download / fallback report.
    fn engine_status(&self, kind: EngineKind, st: EngineStatus) {
        let extra = StatusExtra {
            progress: st.progress,
            model_id: st.model_id,
            engine: Some(kind.as_str().to_string()),
            device: None,
        };
        self.status(None, &st.code, st.detail, st.level, extra);
    }

    /// A status sink for an engine of this session.
    pub(crate) fn engine_status_sink(self: &Arc<Self>, kind: EngineKind) -> StatusSink {
        let ctx = Arc::clone(self);
        StatusSink::new(move |st| ctx.engine_status(kind, st))
    }

    /// The pipeline cannot go on: emit `error` as the last event of this epoch, end the epoch and
    /// tear the session down (asynchronously; callable from any thread of the session).
    pub(crate) fn fatal(&self, code: &str, detail: impl Into<String>, model_id: Option<String>) {
        let detail = detail.into();
        let ended = self
            .with_gate(|g| {
                *self.state.lock().unwrap_or_else(PoisonError::into_inner) = SpeechState::Error;
                let extra = StatusExtra {
                    model_id,
                    ..StatusExtra::default()
                };
                self.sink
                    .status(&self.status_event(SpeechState::Error, code, detail.clone(), StatusLevel::Error, extra));
                g.bump();
                if let Some(slot) = self.slot.upgrade() {
                    let mut live = slot.live();
                    if live.as_ref().is_some_and(|c| c.epoch == self.epoch) {
                        *live = None;
                    }
                }
            })
            .is_some();
        if !ended {
            return;
        }
        tracing::warn!(target: LOG, "[{}] pipeline stopped: {code}: {detail}", self.pipeline_id);
        let Some(slot) = self.slot.upgrade() else { return };
        let epoch = self.epoch;
        let session = {
            let mut s = slot.session.lock().unwrap_or_else(PoisonError::into_inner);
            if s.as_ref().is_some_and(|s| s.ctx.epoch == epoch) {
                s.take()
            } else {
                None
            }
        };
        if let Some(mut session) = session {
            // Stop at once (threads, engines); wait for the device release off this thread.
            session.signal_stop();
            tauri::async_runtime::spawn(async move { session.shutdown().await });
        }
    }

    pub(crate) fn set_backlog(&self, n: usize) {
        self.backlog.store(n.min(u32::MAX as usize) as u32, Ordering::Relaxed);
    }

    // --- language ------------------------------------------------------------------------------

    /// The explicit language, else the latest detected one.
    pub(crate) fn stream_language(&self) -> Option<String> {
        self.settings
            .language
            .clone()
            .or_else(|| self.sticky_language.lock().unwrap_or_else(PoisonError::into_inner).clone())
    }

    /// Remember a detected language (auto mode only).
    pub(crate) fn note_language(&self, detected: Option<&str>) {
        if self.settings.language.is_some() {
            return;
        }
        if let Some(code) = detected.and_then(lang::normalize) {
            *self.sticky_language.lock().unwrap_or_else(PoisonError::into_inner) = Some(code);
        }
    }

    /// Decode options for one utterance / stream.
    pub(crate) fn decode_options(&self, language: Option<String>) -> DecodeOptions {
        DecodeOptions {
            language,
            hotwords: self.settings.hotwords.clone(),
            prompt: self.settings.prompt.clone(),
        }
    }

    /// The text of a result after the session's post-filter, `None` when nothing is left.
    pub(crate) fn clean_text(&self, result: &AsrResult) -> Option<String> {
        let raw = result.text.trim();
        if raw.is_empty() {
            return None;
        }
        if self.settings.hallucination_filter {
            let lang = result.lang.as_deref().and_then(lang::normalize);
            clean_transcript(raw, lang.as_deref(), result.no_speech_prob.unwrap_or(0.0))
        } else {
            Some(raw.split_whitespace().collect::<Vec<_>>().join(" ")).filter(|t| !t.is_empty())
        }
    }

    // --- segment events -------------------------------------------------------------------------

    fn speech_start(&self, segment_id: u64, start_ms: u64) {
        self.with_gate(|_| {
            self.book().open(segment_id, start_ms);
            self.sink.utterance(
                &SpeechUtterance::SpeechStart {
                    pipeline_id: self.pipeline_id.clone(),
                    epoch: self.epoch,
                    segment_id,
                    start_ms,
                },
                0,
            );
        });
    }

    fn utterance_closed(&self, segment_id: u64, end_ms: u64, samples: usize, forced: bool, closed_at: Instant) {
        self.with_gate(|_| {
            let start_ms = {
                let mut book = self.book();
                book.close(segment_id, end_ms, forced, closed_at);
                book.get(segment_id).map_or(end_ms, |e| e.start_ms)
            };
            self.sink.utterance(
                &SpeechUtterance::Utterance {
                    pipeline_id: self.pipeline_id.clone(),
                    epoch: self.epoch,
                    segment_id,
                    start_ms,
                    end_ms: end_ms.max(start_ms),
                    samples: samples.min(u32::MAX as usize) as u32,
                    forced,
                },
                samples_to_ms(samples as u64),
            );
        });
    }

    /// The segment ends without a final (see `utterance::reasons`). No-op when it already has a
    /// final or was dropped.
    pub(crate) fn dropped(&self, segment_id: u64, reason: &str, audio_ms: u64) -> bool {
        self.with_gate(|_| {
            {
                let mut book = self.book();
                match book.get_mut(segment_id) {
                    Some(e) if e.is_pending() => e.dropped = true,
                    Some(_) => return false,
                    None => {}
                }
            }
            if reasons::is_loss(reason) {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
            self.sink.utterance(
                &SpeechUtterance::Dropped {
                    pipeline_id: self.pipeline_id.clone(),
                    epoch: self.epoch,
                    segment_id,
                    reason: reason.to_string(),
                },
                audio_ms,
            );
            true
        })
        .unwrap_or(false)
    }

    /// The segment has neither a final nor a `dropped` yet.
    pub(crate) fn is_pending(&self, segment_id: u64) -> bool {
        self.book().get(segment_id).is_some_and(|e| e.is_pending())
    }

    /// The revision the next partial / final of the segment gets.
    pub(crate) fn peek_revision(&self, segment_id: u64) -> Option<u32> {
        self.book().peek_revision(segment_id)
    }

    /// A streaming hypothesis (ignored once the segment has a final or was dropped).
    pub(crate) fn partial(&self, segment_id: u64, text: &str, lang: Option<String>) -> bool {
        self.with_gate(|_| {
            let revision = {
                let mut book = self.book();
                if !book.get(segment_id).is_some_and(|e| e.is_pending()) {
                    return false;
                }
                match book.take_revision(segment_id) {
                    Some(r) => r,
                    None => return false,
                }
            };
            self.sink.partial(&SpeechPartial {
                pipeline_id: self.pipeline_id.clone(),
                epoch: self.epoch,
                segment_id,
                revision,
                text: text.to_string(),
                lang,
            });
            true
        })
        .unwrap_or(false)
    }

    /// A final of the segment (first pass, or the two-pass replacement). The final of a merged
    /// batch spans to the batch's last segment and carries that segment's end and turn verdict;
    /// when the last segment ended the turn, `speech:turn_end` is repeated for the carrying
    /// segment first (the last one's own ends as `merged`).
    pub(crate) fn final_transcript(&self, segment_id: u64, text: String, meta: FinalMeta) -> bool {
        self.with_gate(|_| {
            let (revision, entry, turn_moved) = {
                let mut book = self.book();
                if book.get(segment_id).is_none_or(|e| e.dropped) {
                    return false;
                }
                let last = meta.merged_until.and_then(|last| book.get(last)).cloned();
                let Some(revision) = book.take_revision(segment_id) else { return false };
                let entry = book.get_mut(segment_id).expect("revision taken above");
                entry.finals += 1;
                let mut turn_moved = None;
                if let Some(last) = &last {
                    let first_final = entry.finals == 1;
                    if first_final && last.turn_end == Some(true) && entry.turn_end != Some(true) {
                        turn_moved = last.turn_probability;
                    }
                    entry.end_ms = last.end_ms.max(entry.start_ms);
                    entry.forced = last.forced;
                    entry.turn_end = last.turn_end;
                    entry.turn_probability = last.turn_probability;
                }
                (revision, entry.clone(), turn_moved)
            };
            if let Some(probability) = turn_moved {
                self.sink.turn_end(&SpeechTurnEnd {
                    pipeline_id: self.pipeline_id.clone(),
                    epoch: self.epoch,
                    segment_id,
                    probability: probability.clamp(0.0, 1.0),
                    lang: meta.lang.clone(),
                });
            }
            let (end_ms, forced) = (entry.end_ms, entry.forced);
            let latency_ms = entry
                .closed_at
                .map_or(0, |t| t.elapsed().as_millis().min(u128::from(u32::MAX)) as u32);
            self.sink.final_transcript(&SpeechFinal {
                pipeline_id: self.pipeline_id.clone(),
                epoch: self.epoch,
                segment_id,
                revision,
                text,
                lang: meta.lang,
                tags: meta.tags,
                pass: meta.pass,
                engine: meta.engine,
                model: meta.model,
                start_ms: entry.start_ms,
                end_ms: end_ms.max(entry.start_ms),
                forced,
                latency_ms,
                turn_end: entry.turn_end,
            });
            true
        })
        .unwrap_or(false)
    }

    /// Record the Smart Turn verdict of a closed segment; emits `speech:turn_end` at or above the
    /// threshold.
    fn turn_verdict(&self, segment_id: u64, probability: f32) {
        let turn_end = probability >= smart_turn::THRESHOLD;
        self.with_gate(|_| {
            if let Some(e) = self.book().get_mut(segment_id) {
                e.turn_end = Some(turn_end);
                e.turn_probability = Some(probability);
            }
            if turn_end {
                self.sink.turn_end(&SpeechTurnEnd {
                    pipeline_id: self.pipeline_id.clone(),
                    epoch: self.epoch,
                    segment_id,
                    probability: probability.clamp(0.0, 1.0),
                    lang: self.stream_language(),
                });
            }
        });
    }
}

// ---------------------------------------------------------------------------------------------
// Sessions
// ---------------------------------------------------------------------------------------------

/// Sender side of a session's stream worker.
pub(crate) struct StreamSender {
    tx: mpsc::Sender<StreamMsg>,
    /// Samples sent and not processed yet.
    queued: Arc<AtomicUsize>,
}

impl StreamSender {
    fn send(&self, msg: StreamMsg) {
        let samples = msg.samples();
        self.queued.fetch_add(samples, Ordering::SeqCst);
        if self.tx.send(msg).is_err() {
            self.queued.fetch_sub(samples, Ordering::SeqCst);
        }
    }

    fn queued(&self) -> usize {
        self.queued.load(Ordering::SeqCst)
    }
}

/// Signals a thread's end when dropped.
pub(crate) struct DoneGuard(#[allow(dead_code)] oneshot::Sender<()>);

/// A running (or starting) session. Dropping it stops everything without waiting.
pub(crate) struct Session {
    ctx: Arc<SessionCtx>,
    stop: Arc<AtomicBool>,
    capture: Option<Thread>,
    capture_done: Option<oneshot::Receiver<()>>,
    utterances: Option<Arc<BoundedQueue<UtteranceJob>>>,
    stream_tx: Option<Arc<Mutex<Option<StreamSender>>>>,
    workers_done: Vec<oneshot::Receiver<()>>,
    cancellers: Vec<Canceller>,
    source_key: String,
    streaming_info: Option<EngineInfo>,
    final_info: Option<EngineInfo>,
}

impl Session {
    fn new(ctx: Arc<SessionCtx>) -> Self {
        Self {
            ctx,
            stop: Arc::new(AtomicBool::new(false)),
            capture: None,
            capture_done: None,
            utterances: None,
            stream_tx: None,
            workers_done: Vec::new(),
            cancellers: Vec::new(),
            source_key: String::new(),
            streaming_info: None,
            final_info: None,
        }
    }

    /// Stop the capture thread and the workers and cancel running decodes; returns at once.
    fn signal_stop(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some(t) = &self.capture {
            t.unpark();
        }
        if let Some(q) = &self.utterances {
            q.close();
        }
        if let Some(tx) = &self.stream_tx {
            // Disconnects the stream worker (it ends after the message in progress).
            tx.lock().unwrap_or_else(PoisonError::into_inner).take();
        }
        for c in &self.cancellers {
            c.cancel();
        }
    }

    /// Stop and wait (bounded) until the device is released and the workers are gone.
    async fn shutdown(mut self) {
        self.signal_stop();
        if let Some(done) = self.capture_done.take() {
            if tokio::time::timeout(CAPTURE_STOP_TIMEOUT, done).await.is_err() {
                tracing::warn!(target: LOG, "[{}] the capture thread did not stop in time", self.ctx.pipeline_id);
            }
        }
        let workers = std::mem::take(&mut self.workers_done);
        let all = futures::future::join_all(workers);
        if tokio::time::timeout(WORKER_STOP_TIMEOUT, all).await.is_err() {
            tracing::debug!(target: LOG, "[{}] an engine is still finishing a decode", self.ctx.pipeline_id);
        }
    }

    fn status(&self) -> PipelineStatus {
        PipelineStatus {
            pipeline_id: self.ctx.pipeline_id.clone(),
            epoch: self.ctx.epoch,
            state: self.ctx.state(),
            source: self.source_key.clone(),
            streaming: self.streaming_info.clone(),
            final_engine: self.final_info.clone(),
            backlog: self.ctx.backlog.load(Ordering::Relaxed),
            dropped: self.ctx.dropped.load(Ordering::Relaxed),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.signal_stop();
    }
}

// ---------------------------------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------------------------------

/// Validate a pipeline key: `[A-Za-z0-9._-]{1,32}` (it becomes part of the TS segment id).
fn validate_pipeline_id(raw: &str, internal: bool) -> Result<String, String> {
    let id = raw.trim();
    let id = if id.is_empty() { super::engine_api::PIPELINE_MIC } else { id };
    let ok = id.len() <= 32 && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'));
    if !ok {
        return Err(format!("invalid pipeline id '{}' (use 1-32 of A-Z a-z 0-9 . _ -)", id.escape_debug()));
    }
    if !internal && id.starts_with(INTERNAL_PREFIX) {
        return Err(format!("pipeline ids starting with '{INTERNAL_PREFIX}' are reserved"));
    }
    Ok(id.to_string())
}

/// Prompt: trimmed, no NUL, at most [`MAX_PROMPT_BYTES`] (cut on a char boundary).
fn sanitize_prompt(raw: &str) -> Option<String> {
    let mut p: String = raw.trim().chars().filter(|&c| c != '\0').collect();
    if p.len() > MAX_PROMPT_BYTES {
        let mut cut = MAX_PROMPT_BYTES;
        while !p.is_char_boundary(cut) {
            cut -= 1;
        }
        p.truncate(cut);
    }
    (!p.is_empty()).then_some(p)
}

/// Hotwords: trimmed, non-empty, unique (case-insensitively), bounded.
fn sanitize_hotwords(raw: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for w in raw {
        let w: String = w.trim().chars().filter(|&c| c != '\0').take(MAX_HOTWORD_CHARS).collect();
        let w = w.trim().to_string();
        if w.is_empty() || out.iter().any(|o| o.eq_ignore_ascii_case(&w)) {
            continue;
        }
        out.push(w);
        if out.len() >= MAX_HOTWORDS {
            break;
        }
    }
    out
}

fn resolve_settings(config: &PipelineConfig) -> SessionSettings {
    let budget = match config.latency_budget_ms {
        0 => DEFAULT_LATENCY_BUDGET_MS,
        ms => ms.clamp(MIN_LATENCY_BUDGET_MS, MAX_LATENCY_BUDGET_MS),
    };
    SessionSettings {
        language: lang::normalize(&config.language),
        hotwords: sanitize_hotwords(&config.hotwords),
        prompt: sanitize_prompt(&config.prompt),
        hallucination_filter: config.hallucination_filter,
        latency_budget: Duration::from_millis(u64::from(budget)),
        max_backlog: config.max_backlog.clamp(1, 10) as usize,
        retain_pcm: config.retain_pcm,
        smart_turn: config.smart_turn,
        vad: config.vad.sanitized(),
        source: config.source.trim().to_string(),
    }
}

fn is_gpu_variant(v: &str) -> bool {
    let v = v.trim().to_ascii_lowercase();
    v.starts_with("cuda") || v == "vulkan" || v == "directml" || v == "dml"
}

/// The plan of an explicit config (no router).
/// The plan of a capture-only pipeline (`PipelineConfig::capture_only`): no engines, nothing to
/// install.
fn capture_only_plan(settings: &SessionSettings) -> RoutePlan {
    RoutePlan {
        language: settings.language.clone().unwrap_or_else(|| lang::AUTO.to_string()),
        tier: Tier::Cpu,
        streaming: None,
        final_engine: None,
        fallback: None,
        language_id: None,
        models: Vec::new(),
        missing: Vec::new(),
        vram_mb: 0,
        notes: vec!["capture_only".to_string()],
    }
}

fn explicit_plan(config: &PipelineConfig, settings: &SessionSettings, requirements: &RequirementsFn) -> RoutePlan {
    let specs: Vec<&EngineSpec> = [&config.streaming, &config.final_engine].into_iter().flatten().collect();
    let tier = if specs.iter().any(|s| s.variant.as_deref().is_some_and(is_gpu_variant)) {
        Tier::Gpu
    } else {
        Tier::Cpu
    };
    let mut models = Vec::new();
    let mut missing = Vec::new();
    for spec in &specs {
        for (id, installed) in requirements(spec) {
            if !models.contains(&id) {
                if !installed {
                    missing.push(id.clone());
                }
                models.push(id);
            }
        }
    }
    RoutePlan {
        language: settings.language.clone().unwrap_or_else(|| lang::AUTO.to_string()),
        tier,
        streaming: config.streaming.clone(),
        final_engine: config.final_engine.clone(),
        fallback: None,
        language_id: None,
        models,
        missing,
        vram_mb: 0,
        notes: vec!["explicit_engines".to_string()],
    }
}

/// A label for logs and status (never a secret: HTTP engines keep their model in `options`).
fn spec_label(spec: &EngineSpec) -> String {
    let model = spec.model.trim();
    let model = if std::path::Path::new(model).is_absolute() {
        std::path::Path::new(model)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default()
    } else {
        model.to_string()
    };
    if model.is_empty() {
        spec.engine.as_str().to_string()
    } else {
        format!("{} {model}", spec.engine)
    }
}

/// A catalog-looking model id of a spec (for `modelId` in status events).
fn spec_model_id(spec: &EngineSpec) -> Option<String> {
    let m = spec.model.trim();
    (!m.is_empty() && !std::path::Path::new(m).is_absolute() && m.contains('/')).then(|| m.to_string())
}

/// The catalog id a `not_installed` message names (engines lead with it: `<id>` or `<id>: why`).
pub(crate) fn catalog_id_in(message: &str) -> Option<String> {
    let head = message.split(": ").next().unwrap_or("").trim();
    let id_like = !head.is_empty()
        && head.len() <= 128
        && head.contains('/')
        && !head.contains(char::is_whitespace)
        && !std::path::Path::new(head).is_absolute();
    id_like.then(|| head.to_string())
}

/// Start the session and fail with a status event of the new epoch.
fn start_failed(ctx: &SessionCtx, code: &str, detail: String, model_id: Option<String>) -> String {
    ctx.fatal(code, detail.clone(), model_id);
    detail
}

const CANCELLED: &str = "cancelled";

/// Create one engine on a blocking thread, reporting `loading` first.
async fn create_engine(
    ctx: &Arc<SessionCtx>,
    factory: &EngineFactory,
    spec: &EngineSpec,
    role: &str,
) -> Result<Box<dyn AsrEngine>, AsrError> {
    ctx.status(
        Some(SpeechState::Loading),
        codes::LOADING_MODEL,
        format!("loading the {role} engine ({})", spec_label(spec)),
        StatusLevel::Info,
        StatusExtra {
            model_id: spec_model_id(spec),
            engine: Some(spec.engine.as_str().to_string()),
            ..StatusExtra::default()
        },
    );
    let engine_ctx = EngineContext {
        status: ctx.engine_status_sink(spec.engine),
    };
    let factory = Arc::clone(factory);
    let spec2 = spec.clone();
    let engine = tauri::async_runtime::spawn_blocking(move || factory(&spec2, &engine_ctx))
        .await
        .map_err(|e| AsrError::runtime(format!("the engine task failed: {e}")))??;
    let info = engine.info();
    tracing::info!(
        target: LOG,
        "[{}] {role} engine ready: {} {} on {} ({})",
        ctx.pipeline_id,
        info.kind,
        info.model,
        info.device,
        info.runtime
    );
    if let Some(reason) = info.fallback_reason.clone() {
        ctx.warn(codes::ENGINE_FALLBACK, reason);
    }
    Ok(engine)
}

/// Map an engine creation error to a status code.
fn engine_error_code(e: &AsrError) -> &'static str {
    match e.kind {
        AsrErrorKind::NotInstalled => codes::NOT_INSTALLED,
        AsrErrorKind::RateLimited => codes::RATE_LIMITED,
        _ => codes::ENGINE_ERROR,
    }
}

/// Engines of a session before they move to their threads.
#[derive(Default)]
struct Engines {
    streaming: Option<Box<dyn AsrEngine>>,
    final_engine: Option<Box<dyn AsrEngine>>,
    fallback: Option<Box<dyn AsrEngine>>,
    language_id: Option<Box<dyn AsrEngine>>,
}

impl Engines {
    fn cancellers(&self) -> Vec<Canceller> {
        [&self.streaming, &self.final_engine, &self.fallback, &self.language_id]
            .into_iter()
            .flatten()
            .filter_map(|e| e.canceller())
            .collect()
    }
}

/// Build and start a session of `config.pipeline_id` (replacing a running one).
pub(crate) async fn start_session(
    registry: &Registry,
    config: PipelineConfig,
    deps: SessionDeps,
    internal: bool,
) -> Result<PipelineStarted, String> {
    let pipeline_id = validate_pipeline_id(&config.pipeline_id, internal)?;
    let slot = registry.slot(&pipeline_id)?;
    // Taken before waiting for `ops`: a stop issued from now on cancels this start.
    let ticket = slot.start_tickets.fetch_add(1, Ordering::SeqCst) + 1;
    let _build = slot.ops.lock().await;
    let settings = resolve_settings(&config);
    let SessionDeps {
        sink,
        engines: engine_factory,
        source: source_factory,
        segmenter: segmenter_factory,
        turn: turn_factory,
        second_pass: second_pass_factory,
        route,
        requirements,
        batch_policy,
        capture_clock,
    } = deps;

    // New epoch: the previous session (running or starting) ends with `stopped`. A stop that came
    // after this call (while it waited for `ops`) wins: nothing starts.
    let ctx = {
        let g = slot.shared.gate.lock();
        if slot.stopped_through.load(Ordering::SeqCst) >= ticket {
            tracing::debug!(target: LOG, "[{pipeline_id}] start cancelled by a stop issued while it was queued");
            return Err(CANCELLED.into());
        }
        if let Some(old) = slot.live().take() {
            end_with_stopped(&old, &g);
        }
        let epoch = g.bump();
        let ctx = Arc::new(SessionCtx::new(&slot, epoch, sink, settings.clone()));
        *slot.live() = Some(Arc::clone(&ctx));
        ctx
    };
    if let Some(previous) = slot.take_session() {
        previous.shutdown().await;
    }
    slot.shared.pcm().clear_before(ctx.epoch);
    ctx.status(Some(SpeechState::Starting), codes::STARTING, "", StatusLevel::Info, StatusExtra::default());

    // 1. The plan: none (capture only), explicit engines, or the router.
    let plan = if config.capture_only {
        capture_only_plan(&settings)
    } else if config.streaming.is_some() || config.final_engine.is_some() {
        explicit_plan(&config, &settings, &requirements)
    } else {
        let mut request = config.route.clone();
        if lang::is_auto(&request.language) {
            if let Some(l) = &settings.language {
                request.language = l.clone();
            }
        }
        match tauri::async_runtime::spawn_blocking(move || route(&request)).await {
            Ok(plan) => plan,
            Err(e) => return Err(start_failed(&ctx, codes::ENGINE_ERROR, format!("the router failed: {e}"), None)),
        }
    };
    if !ctx.is_current() {
        return Err(CANCELLED.into());
    }
    if plan.streaming.is_none() && plan.final_engine.is_none() && !settings.retain_pcm && !config.capture_only {
        let detail = format!(
            "no speech engine for this request (language {}, tier {:?}; {})",
            plan.language,
            plan.tier,
            if plan.notes.is_empty() { "no notes".to_string() } else { plan.notes.join(", ") }
        );
        return Err(start_failed(&ctx, codes::ENGINE_ERROR, detail, None));
    }

    // 2. Engines (blocking model loads, one at a time; a stop in between cancels the start).
    let mut engines = Engines::default();
    let two_pass = plan.streaming.is_some() && plan.final_engine.is_some();
    let auto = settings.language.is_none();
    let roles: [(&Option<EngineSpec>, &str, bool); 4] = [
        (&plan.streaming, "streaming", true),
        (&plan.final_engine, "final", true),
        (&plan.fallback, "fallback", two_pass),
        (&plan.language_id, "language-ID", auto),
    ];
    for (i, (spec, role, used)) in roles.into_iter().enumerate() {
        let (Some(spec), true) = (spec, used) else { continue };
        match create_engine(&ctx, &engine_factory, spec, role).await {
            Ok(engine) => {
                let target = match i {
                    0 => &mut engines.streaming,
                    1 => &mut engines.final_engine,
                    2 => &mut engines.fallback,
                    _ => &mut engines.language_id,
                };
                *target = Some(engine);
            }
            Err(e) if !ctx.is_current() => {
                tracing::debug!(target: LOG, "[{pipeline_id}] start cancelled while loading ({e})");
                return Err(CANCELLED.into());
            }
            Err(e) => {
                let code = engine_error_code(&e);
                let model_id = match e.kind {
                    AsrErrorKind::NotInstalled => catalog_id_in(&e.message).or_else(|| spec_model_id(spec)),
                    _ => spec_model_id(spec),
                };
                tracing::warn!(target: LOG, "[{pipeline_id}] the {role} engine failed to load: {e}");
                return Err(start_failed(&ctx, code, e.to_string(), model_id));
            }
        }
        if !ctx.is_current() {
            return Err(CANCELLED.into());
        }
    }
    let streaming_info = engines.streaming.as_ref().map(|e| e.info().clone());
    let final_info = engines.final_engine.as_ref().map(|e| e.info().clone());

    // 3. Workers.
    let mut session = Session::new(Arc::clone(&ctx));
    session.cancellers = engines.cancellers();
    session.streaming_info = streaming_info.clone();
    session.final_info = final_info.clone();
    let speaking = Arc::new(AtomicBool::new(false));
    let dispatch = match build_workers(
        &ctx,
        &mut session,
        engines,
        &plan,
        &engine_factory,
        second_pass_factory,
        &batch_policy,
        &speaking,
    ) {
        Ok(d) => d,
        Err(e) => {
            session.shutdown().await;
            return Err(start_failed(&ctx, codes::ENGINE_ERROR, e, None));
        }
    };

    // 4. The capture thread: Silero (and Smart Turn) load before the device opens.
    let (ready_tx, ready_rx) = oneshot::channel::<Result<(), String>>();
    let (source_tx, source_rx) = mpsc::channel::<Box<dyn AudioSource>>();
    let (done_tx, done_rx) = oneshot::channel::<()>();
    let capture = {
        let ctx = Arc::clone(&ctx);
        let stop = Arc::clone(&session.stop);
        let vad = settings.vad.clone();
        let smart_turn = settings.smart_turn;
        let speaking = Arc::clone(&speaking);
        thread::Builder::new()
            .name(format!("sigil-speech-{pipeline_id}"))
            .spawn(move || {
                let _done = DoneGuard(done_tx);
                let segmenter = match segmenter_factory(&vad) {
                    Ok(s) => s,
                    Err(e) => {
                        let _ = ready_tx.send(Err(format!("speech detection (Silero VAD) failed to load: {e}")));
                        return;
                    }
                };
                let turn = if smart_turn {
                    match turn_factory() {
                        Ok(t) => Some(t),
                        Err(e) => {
                            ctx.status(
                                None,
                                codes::NOT_INSTALLED,
                                format!("Smart Turn is unavailable ({e}); turn ends are not detected"),
                                StatusLevel::Warning,
                                StatusExtra {
                                    model_id: Some(smart_turn::MODEL_ID.to_string()),
                                    ..StatusExtra::default()
                                },
                            );
                            None
                        }
                    }
                } else {
                    None
                };
                if ready_tx.send(Ok(())).is_err() {
                    return;
                }
                // The source arrives once the device is attached (the sender drops on failure).
                let Ok(source) = source_rx.recv() else { return };
                let start_qpc_ns = capture_clock.then(hub::now_qpc_ns);
                let front = FrontEnd::new(Arc::clone(&ctx), segmenter, turn, dispatch, start_qpc_ns, speaking);
                let source = run_capture(front, source, &stop);
                source.close();
            })
    };
    let capture = match capture {
        Ok(handle) => handle.thread().clone(),
        Err(e) => {
            session.shutdown().await;
            return Err(start_failed(&ctx, codes::ENGINE_ERROR, format!("cannot start the capture thread: {e}"), None));
        }
    };
    session.capture = Some(capture);
    session.capture_done = Some(done_rx);
    match ready_rx.await {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            drop(source_tx);
            session.shutdown().await;
            return Err(start_failed(&ctx, codes::ENGINE_ERROR, e, None));
        }
        Err(_) => {
            drop(source_tx);
            session.shutdown().await;
            return Err(start_failed(&ctx, codes::ENGINE_ERROR, "the capture thread exited during start".into(), None));
        }
    }
    if !ctx.is_current() {
        // The capture thread waits for the source: dropping the sender ends it.
        drop(source_tx);
        session.shutdown().await;
        return Err(CANCELLED.into());
    }

    // 5. The device (a first consumer opens it).
    let source_id = settings.source.clone();
    let source = tauri::async_runtime::spawn_blocking(move || source_factory(&source_id))
        .await
        .map_err(|e| format!("the audio capture task failed: {e}"))
        .and_then(|r| r);
    let source = match source {
        Ok(s) => s,
        Err(e) => {
            drop(source_tx);
            session.shutdown().await;
            return Err(start_failed(&ctx, codes::CAPTURE_ERROR, e, None));
        }
    };
    session.source_key = source.key();
    let source_key = session.source_key.clone();
    if source_tx.send(source).is_err() {
        drop(source_tx);
        session.shutdown().await;
        return Err(start_failed(&ctx, codes::CAPTURE_ERROR, "the capture thread exited during start".into(), None));
    }

    // 6. Install (unless a stop or a newer start came first).
    let cancelled = {
        let _g = slot.shared.gate.lock();
        if slot.shared.gate.current() == ctx.epoch {
            *slot.session.lock().unwrap_or_else(PoisonError::into_inner) = Some(session);
            None
        } else {
            Some(session)
        }
    };
    if let Some(session) = cancelled {
        session.shutdown().await;
        return Err(CANCELLED.into());
    }
    tracing::info!(
        target: LOG,
        "[{pipeline_id}] session {} listening on {source_key} (streaming: {}, final: {}, language: {}, budget {} ms, backlog {})",
        ctx.epoch,
        streaming_info.as_ref().map_or("none".to_string(), |i| format!("{} {}", i.kind, i.model)),
        final_info.as_ref().map_or("none".to_string(), |i| format!("{} {}", i.kind, i.model)),
        settings.language.as_deref().unwrap_or("auto"),
        settings.latency_budget.as_millis(),
        settings.max_backlog,
    );
    ctx.status(
        Some(SpeechState::Listening),
        codes::LISTENING,
        source_key,
        StatusLevel::Info,
        StatusExtra {
            engine: final_info.as_ref().or(streaming_info.as_ref()).map(|i| i.kind.as_str().to_string()),
            device: final_info.as_ref().or(streaming_info.as_ref()).map(|i| i.device.clone()),
            ..StatusExtra::default()
        },
    );
    Ok(PipelineStarted {
        pipeline_id,
        epoch: ctx.epoch,
        plan: redacted(plan),
    })
}

/// Whether an engine option key may hold a secret (`apiKey`, `token`, ...).
fn is_secret_key(key: &str) -> bool {
    let k = key.to_ascii_lowercase();
    ["key", "token", "secret", "password", "authorization"]
        .iter()
        .any(|s| k.contains(s))
}

/// `plan` without secret-looking engine options (it goes back over IPC; keys are never echoed).
fn redacted(mut plan: RoutePlan) -> RoutePlan {
    for spec in [&mut plan.streaming, &mut plan.final_engine, &mut plan.fallback, &mut plan.language_id]
        .into_iter()
        .flatten()
    {
        if let Some(options) = spec.options.as_object_mut() {
            options.retain(|k, _| !is_secret_key(k));
        }
    }
    plan
}

/// Emit `stopped` as the last event of `ctx`'s epoch (with the emit lock held).
fn end_with_stopped(ctx: &SessionCtx, g: &GateGuard<'_>) {
    if g.current() != ctx.epoch {
        return;
    }
    *ctx.state.lock().unwrap_or_else(PoisonError::into_inner) = SpeechState::Stopped;
    ctx.sink.status(&ctx.status_event(
        SpeechState::Stopped,
        codes::STOPPED,
        String::new(),
        StatusLevel::Info,
        StatusExtra::default(),
    ));
}

/// Spawn the workers of `plan` and return where the capture thread sends closed segments.
#[allow(clippy::too_many_arguments)] // One call site; every argument is a distinct part of the session.
fn build_workers(
    ctx: &Arc<SessionCtx>,
    session: &mut Session,
    engines: Engines,
    plan: &RoutePlan,
    engine_factory: &EngineFactory,
    second_pass_factory: SecondPassFactory,
    batch_policy: &BatchPolicyFn,
    speaking: &Arc<AtomicBool>,
) -> Result<Dispatch, String> {
    let Engines {
        streaming,
        final_engine,
        fallback,
        language_id,
    } = engines;
    if let Some(stream_engine) = streaming {
        // A stream that needs a language waits for one (the two-pass final detects it) only when
        // the route has that final and the spec names no fallback language; otherwise it runs
        // (the engine falls back to its own auto mode and says so).
        let spec_language = plan
            .streaming
            .as_ref()
            .and_then(|s| s.options.get("language"))
            .and_then(|l| l.as_str())
            .is_some_and(|l| !lang::is_auto(l));
        let wait_for_language = Arc::new(AtomicBool::new(
            stream_engine.info().caps.needs_language && final_engine.is_some() && !spec_language,
        ));
        // An auto plan limits the stream to the detected languages its route streams (never
        // zh / yue); every other utterance goes to the final alone.
        let stream_languages = plan.streaming.as_ref().map_or_else(
            || stream_engine.info().caps.languages.clone(),
            |spec| router::stream_languages(spec, &stream_engine.info().caps.languages),
        );
        let (tx, rx) = mpsc::channel();
        let queued = Arc::new(AtomicUsize::new(0));
        let second = final_engine.map(|final_engine| SecondPassSetup {
            factory: second_pass_factory,
            final_engine,
            fallback,
            // P2-C's budget: a replacement that arrives late only makes captions flicker.
            config: TwoPassConfig::default(),
        });
        let (done_tx, done_rx) = oneshot::channel();
        worker::spawn_stream_worker(
            StreamWorkerSetup {
                ctx: Arc::clone(ctx),
                rx,
                queued: Arc::clone(&queued),
                engine: stream_engine,
                second,
                language_id,
                wait_for_language: Arc::clone(&wait_for_language),
            },
            DoneGuard(done_tx),
        )?;
        session.workers_done.push(done_rx);
        let sender = Arc::new(Mutex::new(Some(StreamSender { tx, queued })));
        session.stream_tx = Some(Arc::clone(&sender));
        return Ok(Dispatch::Stream(StreamFeed::new(
            sender,
            wait_for_language,
            stream_languages,
            &ctx.settings.vad,
        )));
    }
    if let Some(final_engine) = final_engine {
        let queue = Arc::new(BoundedQueue::new(ctx.settings.max_backlog));
        let fallback = plan.fallback.clone().map(|spec| Fallback {
            spec,
            factory: Arc::clone(engine_factory),
        });
        let policy = plan.final_engine.as_ref().and_then(|spec| batch_policy(spec));
        let (done_tx, done_rx) = oneshot::channel();
        worker::spawn_utterance_worker(
            UtteranceWorkerSetup {
                ctx: Arc::clone(ctx),
                queue: Arc::clone(&queue),
                engine: final_engine,
                fallback,
                language_id,
                policy,
                speaking: Arc::clone(speaking),
            },
            DoneGuard(done_tx),
        )?;
        session.workers_done.push(done_rx);
        session.utterances = Some(Arc::clone(&queue));
        return Ok(Dispatch::Utterance(queue));
    }
    Ok(Dispatch::CaptureOnly)
}

/// Stop the session of `pipeline_id`. After it returns no event of that session is emitted and
/// the device was released (bounded wait). Returns whether a session was running.
pub(crate) async fn stop_session(registry: &Registry, pipeline_id: &str) -> bool {
    stop_session_at(registry, pipeline_id, None).await
}

/// [`stop_session`], or with `epoch` only the session of that epoch: while it is the key's
/// current epoch (checked under the emit lock) it stops, and starts queued behind it keep their
/// tickets (a newer start replaces the session itself); otherwise nothing happens and it returns
/// `false`. A driver cleaning up after a start it gave up on can never cancel a newer start.
pub(crate) async fn stop_session_at(registry: &Registry, pipeline_id: &str, epoch: Option<u64>) -> bool {
    let Some(slot) = registry.existing(pipeline_id.trim()) else {
        return false;
    };
    let (was_live, stop_epoch) = {
        let g = slot.shared.gate.lock();
        if let Some(epoch) = epoch {
            if g.current() != epoch {
                return false;
            }
        } else {
            // Starts invoked before this stop (still waiting for `ops`) give up when they get it.
            let issued = slot.start_tickets.load(Ordering::SeqCst);
            slot.stopped_through.fetch_max(issued, Ordering::SeqCst);
        }
        let live = slot.live().take();
        if let Some(old) = &live {
            end_with_stopped(old, &g);
        }
        (live.is_some(), g.bump())
    };
    let session = slot.take_session();
    let running = session.is_some();
    if let Some(session) = session {
        session.shutdown().await;
        tracing::info!(target: LOG, "[{pipeline_id}] stopped");
    }
    // Only the ended sessions' PCM: a start that ran meanwhile has a newer epoch.
    slot.shared.pcm().clear_before(stop_epoch);
    running || was_live
}

// ---------------------------------------------------------------------------------------------
// Capture thread
// ---------------------------------------------------------------------------------------------

/// Where closed segments go.
enum Dispatch {
    /// No engine: PCM for `take_pipeline_utterance` only.
    CaptureOnly,
    /// Finals-only route: the ordered utterance worker.
    Utterance(Arc<BoundedQueue<UtteranceJob>>),
    /// Streaming route (with the optional two-pass re-decode inside the stream worker).
    Stream(StreamFeed),
}

/// Feeds the stream worker from the capture thread.
struct StreamFeed {
    sender: Arc<Mutex<Option<StreamSender>>>,
    history: AudioHistory,
    /// Absolute position up to which audio went to a stream (a stream never gets audio twice).
    fed_to: u64,
    /// Segment with an open stream.
    open: Option<u64>,
    /// The engine fell behind: no stream starts until it caught up.
    degraded: bool,
    /// Streams start only once a language the engine knows is known (explicit or detected); the
    /// stream worker clears it when the two-pass final that would cover the others cannot start.
    wait_for_language: Arc<AtomicBool>,
    /// Languages of the streaming engine (empty = any): it never gets another one.
    languages: Vec<String>,
    max_lag: usize,
}

impl StreamFeed {
    fn new(
        sender: Arc<Mutex<Option<StreamSender>>>,
        wait_for_language: Arc<AtomicBool>,
        languages: Vec<String>,
        vad: &VadConfig,
    ) -> Self {
        // Enough history for the pre-roll of a segment confirmed after `min_speech_ms`.
        let history_ms = (vad.pre_roll_ms + vad.min_speech_ms + 1_000).max(2_000);
        Self {
            sender,
            history: AudioHistory::new(ms_to_samples(history_ms)),
            fed_to: 0,
            open: None,
            degraded: false,
            wait_for_language,
            languages,
            max_lag: ms_to_samples(STREAM_MAX_LAG_MS),
        }
    }

    fn with_sender<T>(&self, f: impl FnOnce(&StreamSender) -> T) -> Option<T> {
        self.sender.lock().unwrap_or_else(PoisonError::into_inner).as_ref().map(f)
    }

    fn send(&self, msg: StreamMsg) {
        self.with_sender(|s| s.send(msg));
    }

    fn queued(&self) -> usize {
        self.with_sender(StreamSender::queued).unwrap_or(0)
    }

    /// Open a stream for a confirmed segment; `false` when it is not streamed.
    fn begin(&mut self, ctx: &SessionCtx, segment_id: u64, start_sample: u64, pos: u64) -> bool {
        if let Some(open) = self.open.take() {
            self.send(StreamMsg::Abort { segment_id: open });
        }
        if self.degraded {
            if self.queued() > self.max_lag / 2 {
                return false;
            }
            self.degraded = false;
        }
        // Only a language the engine lists (a detected one may be any): a needy engine without a
        // usable one waits for the two-pass final instead of failing to open the stream.
        let language = worker::supported_language(&self.languages, ctx.stream_language());
        if language.is_none() && self.wait_for_language.load(Ordering::SeqCst) {
            return false;
        }
        let from = start_sample.max(self.fed_to);
        let audio = self.history.range(from, pos);
        self.fed_to = pos.max(self.fed_to);
        self.send(StreamMsg::Begin {
            segment_id,
            options: ctx.decode_options(language),
            audio,
        });
        self.open = Some(segment_id);
        true
    }

    /// Push new audio to the open stream (or give the stream up when the engine is too far behind).
    fn feed(&mut self, ctx: &SessionCtx, pos: u64) {
        let Some(segment_id) = self.open else { return };
        if self.queued() > self.max_lag {
            self.send(StreamMsg::Abort { segment_id });
            self.open = None;
            self.degraded = true;
            ctx.warn(
                codes::BACKLOG_DROPPED,
                "the streaming engine is slower than real time; live text is skipped for this utterance",
            );
            return;
        }
        if pos <= self.fed_to {
            return;
        }
        let audio = self.history.range(self.fed_to, pos);
        self.fed_to = pos;
        if !audio.is_empty() {
            self.send(StreamMsg::Push { segment_id, audio });
        }
    }

    /// A segment closed at absolute position `end`.
    fn close(&mut self, segment_id: u64, end: u64, closed: ClosedSegment) {
        if self.open == Some(segment_id) {
            if end > self.fed_to {
                let audio = self.history.range(self.fed_to, end);
                self.fed_to = end;
                if !audio.is_empty() {
                    self.send(StreamMsg::Push { segment_id, audio });
                }
            }
            self.open = None;
            self.send(StreamMsg::End { segment_id, closed });
        } else {
            self.send(StreamMsg::FinalOnly { segment_id, closed });
        }
    }

    /// Drop the open stream of `segment_id` (muted).
    fn abort(&mut self, segment_id: u64) {
        if self.open == Some(segment_id) {
            self.open = None;
            self.send(StreamMsg::Abort { segment_id });
        }
    }
}

/// An announced, not yet closed segment.
struct OpenSegment {
    segment_id: u64,
    start_sample: u64,
}

/// State of the capture thread.
struct FrontEnd {
    ctx: Arc<SessionCtx>,
    segmenter: Box<dyn SpeechSegmenter>,
    turn: Option<Box<dyn TurnDetector>>,
    dispatch: Dispatch,
    clock: SessionClock,
    /// Samples pushed to the segmenter so far.
    pos: u64,
    events: Vec<SegmenterEvent>,
    /// Segmenter `seq` → segment id of announced segments.
    seq_ids: Vec<(u64, u64)>,
    open: Option<OpenSegment>,
    /// A segment is open (the utterance worker may hold a short batch longer).
    speaking: Arc<AtomicBool>,
    turn_warned: bool,
}

impl FrontEnd {
    fn new(
        ctx: Arc<SessionCtx>,
        segmenter: Box<dyn SpeechSegmenter>,
        turn: Option<Box<dyn TurnDetector>>,
        dispatch: Dispatch,
        start_qpc_ns: Option<i64>,
        speaking: Arc<AtomicBool>,
    ) -> Self {
        Self {
            ctx,
            segmenter,
            turn,
            dispatch,
            clock: SessionClock::new(start_qpc_ns),
            pos: 0,
            events: Vec::new(),
            seq_ids: Vec::new(),
            open: None,
            speaking,
            turn_warned: false,
        }
    }

    /// Segment 16 kHz audio and route the events.
    fn process(&mut self, chunk: &[f32]) -> Result<(), String> {
        if let Dispatch::Stream(feed) = &mut self.dispatch {
            feed.history.push(self.pos, chunk);
        }
        let mut events = std::mem::take(&mut self.events);
        events.clear();
        let pushed = self.segmenter.push(chunk, &mut events);
        self.pos += chunk.len() as u64;
        for ev in events.drain(..) {
            self.handle(ev);
        }
        self.events = events;
        self.sync_speaking();
        pushed?;
        if let Dispatch::Stream(feed) = &mut self.dispatch {
            feed.feed(&self.ctx, self.pos);
        }
        Ok(())
    }

    /// A capture discontinuity.
    fn gap(&mut self, action: GapAction, restarted: bool) -> Result<(), String> {
        match action {
            GapAction::DropOpen => {
                self.segmenter.reset();
                self.seq_ids.clear();
                self.speaking.store(false, Ordering::SeqCst);
                if let Some(open) = self.open.take() {
                    if let Dispatch::Stream(feed) = &mut self.dispatch {
                        feed.abort(open.segment_id);
                    }
                    let audio_ms = samples_to_ms(self.pos.saturating_sub(open.start_sample));
                    self.ctx.dropped(open.segment_id, reasons::MUTED, audio_ms);
                }
                tracing::debug!(target: LOG, "[{}] muted: open utterance dropped", self.ctx.pipeline_id);
            }
            GapAction::CloseOpen => {
                let mut events = std::mem::take(&mut self.events);
                events.clear();
                let flushed = self.segmenter.flush(&mut events);
                for ev in events.drain(..) {
                    self.handle(ev);
                }
                self.events = events;
                self.sync_speaking();
                flushed?;
                if restarted {
                    self.ctx.warn(
                        codes::CAPTURE_GAP,
                        "the audio device restarted or changed; the open utterance was closed",
                    );
                }
            }
        }
        Ok(())
    }

    /// `speaking` follows the announced open segment once a batch of segmenter events is handled:
    /// a forced close whose remainder never confirms (silence after a max-length split) or a
    /// flush at a device gap leaves nobody speaking.
    fn sync_speaking(&self) {
        self.speaking.store(self.open.is_some(), Ordering::SeqCst);
    }

    fn handle(&mut self, event: SegmenterEvent) {
        match event {
            SegmenterEvent::SpeechStart { seq, start_sample } => self.speech_start(seq, start_sample),
            SegmenterEvent::Segment(seg) => {
                let segment_id = match self.seq_ids.iter().position(|(s, _)| *s == seg.seq) {
                    Some(i) => self.seq_ids.remove(i).1,
                    // Never happens (every segment is announced first); announce it now.
                    None => self.announce(seg.seq, seg.start_sample),
                };
                let start_sample = seg.start_sample;
                let end = start_sample + seg.samples.len() as u64;
                self.closed(segment_id, end, seg.samples, seg.forced);
            }
        }
    }

    fn announce(&mut self, _seq: u64, start_sample: u64) -> u64 {
        let segment_id = self.ctx.shared.next_segment_id();
        self.ctx.speech_start(segment_id, self.clock.ms(start_sample));
        segment_id
    }

    fn speech_start(&mut self, seq: u64, start_sample: u64) {
        let segment_id = self.announce(seq, start_sample);
        self.seq_ids.push((seq, segment_id));
        if self.seq_ids.len() > 8 {
            self.seq_ids.remove(0);
        }
        self.open = Some(OpenSegment {
            segment_id,
            start_sample,
        });
        self.speaking.store(true, Ordering::SeqCst);
        if let Dispatch::Stream(feed) = &mut self.dispatch {
            feed.begin(&self.ctx, segment_id, start_sample, self.pos);
        }
    }

    fn closed(&mut self, segment_id: u64, end: u64, samples: Vec<f32>, forced: bool) {
        if self.open.as_ref().is_some_and(|o| o.segment_id == segment_id) {
            self.open = None;
        }
        let closed_at = Instant::now();
        let len = samples.len();
        let audio: Arc<[f32]> = Arc::from(samples);
        if self.ctx.settings.retain_pcm {
            // Only while this session is current (a stale capture thread still finishing a buffer
            // must not fill, or evict from, the next session's store).
            let evicted = self
                .ctx
                .with_gate(|_| self.ctx.shared.pcm().insert(self.ctx.epoch, segment_id, audio.to_vec()))
                .unwrap_or_default();
            for (_, old) in evicted {
                tracing::debug!(target: LOG, "[{}] retained utterance {old} was never taken; dropped", self.ctx.pipeline_id);
            }
        }
        self.ctx
            .utterance_closed(segment_id, self.clock.ms(end), len, forced, closed_at);
        if !forced {
            self.speaking.store(false, Ordering::SeqCst);
            self.smart_turn(segment_id, &audio);
        }
        match &mut self.dispatch {
            Dispatch::CaptureOnly => {}
            Dispatch::Utterance(queue) => {
                let evicted = queue.push(UtteranceJob {
                    segment_id,
                    audio,
                    closed_at,
                });
                for old in evicted {
                    self.ctx
                        .dropped(old.segment_id, reasons::BACKLOG, samples_to_ms(old.audio.len() as u64));
                    self.ctx.warn(
                        codes::BACKLOG_DROPPED,
                        format!(
                            "speech recognition is slower than real time; an utterance was dropped (backlog cap {})",
                            queue.capacity()
                        ),
                    );
                }
                self.ctx.set_backlog(queue.len());
            }
            Dispatch::Stream(feed) => feed.close(segment_id, end, ClosedSegment { audio, closed_at }),
        }
    }

    fn smart_turn(&mut self, segment_id: u64, audio: &[f32]) {
        let Some(turn) = self.turn.as_mut() else { return };
        match turn.predict(tail(audio, TURN_WINDOW_MS)) {
            Ok(p) if p.is_finite() => self.ctx.turn_verdict(segment_id, p),
            Ok(_) => {}
            Err(e) => {
                if !self.turn_warned {
                    self.turn_warned = true;
                    self.ctx.warn(codes::ENGINE_ERROR, format!("Smart Turn failed: {e}"));
                }
            }
        }
    }
}

/// Synthetic silence fed per loopback idle period: enough for the VAD's `min_silence_ms` (up to
/// 5 s) to close an utterance that playback stopping cut off.
fn loopback_idle_fill(vad: &VadConfig) -> Duration {
    LOOPBACK_IDLE_FILL.max(Duration::from_millis(u64::from(vad.min_silence_ms) + LOOPBACK_IDLE_MARGIN_MS))
}

/// The capture loop; returns the source for release.
fn run_capture(mut front: FrontEnd, mut source: Box<dyn AudioSource>, stop: &AtomicBool) -> Box<dyn AudioSource> {
    let loopback = source.is_loopback();
    let idle_fill_max = loopback_idle_fill(&front.ctx.settings.vad);
    let mut buf: Vec<f32> = Vec::with_capacity(16_384);
    let mut last_audio = Instant::now();
    let mut idle_filled = Duration::ZERO;
    let mut reported_drops = 0u64;
    let mut last_drop_log: Option<Instant> = None;
    let id = front.ctx.pipeline_id.clone();
    loop {
        if stop.load(Ordering::Acquire) || !front.ctx.is_current() {
            return source;
        }
        buf.clear();
        let info = source.read(&mut buf);
        if !buf.is_empty() {
            last_audio = Instant::now();
            idle_filled = Duration::ZERO;
            front.clock.observe(front.pos + buf.len() as u64, info.end_qpc_ns);
            if let Err(e) = front.process(&buf) {
                front
                    .ctx
                    .fatal(codes::CAPTURE_ERROR, format!("speech detection failed: {e}"), None);
                return source;
            }
        }
        if info.gap_after {
            let action = gap_action(source.muted());
            if let Err(e) = front.gap(action, info.restarted) {
                front
                    .ctx
                    .fatal(codes::CAPTURE_ERROR, format!("speech detection failed: {e}"), None);
                return source;
            }
        }
        let dropped = source.dropped_frames();
        if dropped > reported_drops && last_drop_log.is_none_or(|t| t.elapsed() >= Duration::from_secs(10)) {
            tracing::warn!(
                target: LOG,
                "[{id}] the speech capture thread fell behind: {} device frames dropped",
                dropped - reported_drops
            );
            reported_drops = dropped;
            last_drop_log = Some(Instant::now());
        }
        if !buf.is_empty() || info.gap_after {
            continue;
        }
        if let Some(message) = source.ended() {
            front
                .ctx
                .fatal(codes::CAPTURE_ERROR, format!("audio device error: {message}"), None);
            return source;
        }
        if loopback {
            let idle = last_audio.elapsed();
            if idle >= LOOPBACK_IDLE_AFTER && idle_filled < idle_fill_max {
                let fill = idle.min(idle_fill_max) - idle_filled;
                let samples = ms_to_samples(fill.as_millis().min(u128::from(u32::MAX)) as u32);
                if samples > 0 {
                    buf.clear();
                    buf.resize(samples, 0.0);
                    if let Err(e) = front.process(&buf) {
                        front
                            .ctx
                            .fatal(codes::CAPTURE_ERROR, format!("speech detection failed: {e}"), None);
                        return source;
                    }
                }
                idle_filled += fill;
            }
        }
        source.wait(IDLE_WAIT);
    }
}

// ---------------------------------------------------------------------------------------------
// Crate API (legacy Whisper recording) and IPC
// ---------------------------------------------------------------------------------------------

/// Start a pipeline with custom dependencies (crate-internal keys allowed).
pub(crate) async fn start_with(config: PipelineConfig, deps: SessionDeps) -> Result<PipelineStarted, String> {
    start_session(registry(), config, deps, true).await
}

/// Stop a pipeline key; `true` when a session was running or starting.
pub(crate) async fn stop_pipeline_key(pipeline_id: &str) -> bool {
    stop_session(registry(), pipeline_id).await
}

/// Plugin setup hook (called once from `speech::init`'s setup; must return quickly).
pub fn setup<R: Runtime>(app: &AppHandle<R>) {
    let _ = app;
    tracing::debug!(target: LOG, "speech pipelines ready");
}

#[command]
pub async fn start_pipeline<R: Runtime>(app: AppHandle<R>, config: PipelineConfig) -> Result<PipelineStarted, String> {
    let sink: Arc<dyn EventSink> = Arc::new(TauriSink::new(app));
    start_session(registry(), config, SessionDeps::production(sink), false).await
}

#[command]
pub async fn stop_pipeline<R: Runtime>(app: AppHandle<R>, pipeline_id: String, epoch: Option<u64>) -> Result<(), String> {
    let _ = app;
    let id = validate_pipeline_id(&pipeline_id, false)?;
    stop_session_at(registry(), &id, epoch).await;
    Ok(())
}

#[command]
pub async fn get_pipeline_status() -> Result<Vec<PipelineStatus>, String> {
    Ok(registry().statuses())
}

#[command]
pub async fn take_pipeline_utterance(pipeline_id: String, segment_id: u64) -> Result<tauri::ipc::Response, String> {
    match registry().take_pcm(pipeline_id.trim(), segment_id) {
        Some(samples) => Ok(tauri::ipc::Response::new(f32_to_le_bytes(&samples))),
        None => Err(format!(
            "utterance {segment_id} of pipeline '{}' is not available (already taken, evicted, or retainPcm is off)",
            pipeline_id.trim().escape_debug()
        )),
    }
}

#[cfg(test)]
mod tests {
    //! Pipeline tests on fake engines, a scripted source and a synthetic VAD classifier (a sample's
    //! value is the frame's speech probability): epoch isolation, no stale segments, utterance
    //! framing, mute, gaps, backlog / budget / merge, streaming revisions, two-pass, Smart Turn,
    //! capture-only PCM and fatal errors. No device, model or network.

    use super::*;
    use crate::services::speech::engine_api::{AsrErrorKind, EngineCaps, Hypothesis};
    use crate::services::speech::two_pass::FinalJob;
    use crate::services::speech_core::segmenter::{tests::LevelClassifier, SegmenterEngine};
    use std::collections::VecDeque;
    use std::sync::Condvar;

    /// Samples per millisecond.
    const MS: usize = 16;
    const WAIT: Duration = Duration::from_secs(10);

    // --- recording sink ---------------------------------------------------------------------------

    #[derive(Clone, Debug)]
    enum Ev {
        Partial(SpeechPartial),
        Final(SpeechFinal),
        Status(SpeechStatus),
        Utterance(SpeechUtterance, #[allow(dead_code)] u64),
        TurnEnd(SpeechTurnEnd),
        Marker(&'static str),
    }

    impl Ev {
        fn epoch(&self) -> Option<u64> {
            match self {
                Ev::Partial(e) => Some(e.epoch),
                Ev::Final(e) => Some(e.epoch),
                Ev::Status(e) => Some(e.epoch),
                Ev::TurnEnd(e) => Some(e.epoch),
                Ev::Utterance(u, _) => Some(match u {
                    SpeechUtterance::SpeechStart { epoch, .. }
                    | SpeechUtterance::Utterance { epoch, .. }
                    | SpeechUtterance::Dropped { epoch, .. } => *epoch,
                }),
                Ev::Marker(_) => None,
            }
        }
    }

    #[derive(Default)]
    struct Recorder {
        events: Mutex<Vec<Ev>>,
        changed: Condvar,
    }

    impl Recorder {
        fn push(&self, e: Ev) {
            self.events.lock().unwrap().push(e);
            self.changed.notify_all();
        }

        fn all(&self) -> Vec<Ev> {
            self.events.lock().unwrap().clone()
        }

        /// Block until `done(events)` holds (panics with the events after [`WAIT`]).
        fn wait_for(&self, what: &str, done: impl Fn(&[Ev]) -> bool) -> Vec<Ev> {
            let deadline = Instant::now() + WAIT;
            let mut events = self.events.lock().unwrap();
            loop {
                if done(&events) {
                    return events.clone();
                }
                let now = Instant::now();
                if now >= deadline {
                    panic!("timed out waiting for {what}; events: {:#?}", *events);
                }
                events = self.changed.wait_timeout(events, deadline - now).unwrap().0;
            }
        }
    }

    struct RecSink(Arc<Recorder>);

    impl EventSink for RecSink {
        fn partial(&self, e: &SpeechPartial) {
            self.0.push(Ev::Partial(e.clone()));
        }
        fn final_transcript(&self, e: &SpeechFinal) {
            self.0.push(Ev::Final(e.clone()));
        }
        fn status(&self, e: &SpeechStatus) {
            self.0.push(Ev::Status(e.clone()));
        }
        fn utterance(&self, e: &SpeechUtterance, audio_ms: u64) {
            self.0.push(Ev::Utterance(e.clone(), audio_ms));
        }
        fn turn_end(&self, e: &SpeechTurnEnd) {
            self.0.push(Ev::TurnEnd(e.clone()));
        }
    }

    fn finals(events: &[Ev]) -> Vec<SpeechFinal> {
        events
            .iter()
            .filter_map(|e| match e {
                Ev::Final(f) => Some(f.clone()),
                _ => None,
            })
            .collect()
    }

    fn partials(events: &[Ev]) -> Vec<SpeechPartial> {
        events
            .iter()
            .filter_map(|e| match e {
                Ev::Partial(p) => Some(p.clone()),
                _ => None,
            })
            .collect()
    }

    fn statuses(events: &[Ev]) -> Vec<SpeechStatus> {
        events
            .iter()
            .filter_map(|e| match e {
                Ev::Status(s) => Some(s.clone()),
                _ => None,
            })
            .collect()
    }

    /// `(kind, segment id)` of every `speech:utterance`: kind is `start`, `closed` or the dropped
    /// reason.
    fn utterances(events: &[Ev]) -> Vec<(String, u64)> {
        events
            .iter()
            .filter_map(|e| match e {
                Ev::Utterance(SpeechUtterance::SpeechStart { segment_id, .. }, _) => Some(("start".into(), *segment_id)),
                Ev::Utterance(SpeechUtterance::Utterance { segment_id, .. }, _) => Some(("closed".into(), *segment_id)),
                Ev::Utterance(SpeechUtterance::Dropped { segment_id, reason, .. }, _) => {
                    Some((reason.clone(), *segment_id))
                }
                _ => None,
            })
            .collect()
    }

    /// `(segment id, start ms, end ms, samples, forced)` of every closed utterance.
    fn closed(events: &[Ev]) -> Vec<(u64, u64, u64, u32, bool)> {
        events
            .iter()
            .filter_map(|e| match e {
                Ev::Utterance(
                    SpeechUtterance::Utterance {
                        segment_id,
                        start_ms,
                        end_ms,
                        samples,
                        forced,
                        ..
                    },
                    _,
                ) => Some((*segment_id, *start_ms, *end_ms, *samples, *forced)),
                _ => None,
            })
            .collect()
    }

    fn count_closed(events: &[Ev]) -> usize {
        closed(events).len()
    }

    // --- scripted source --------------------------------------------------------------------------

    enum Item {
        Audio(Vec<f32>),
        Gap { restarted: bool, muted: bool },
    }

    #[derive(Default)]
    struct Script {
        items: Mutex<VecDeque<Item>>,
        muted: AtomicBool,
        attached: AtomicBool,
        closed: AtomicBool,
        /// The source is a loopback capture (idle time is filled with silence).
        loopback: AtomicBool,
        /// `close` blocks on this latch (a slow device release).
        close_gate: Mutex<Option<Latch>>,
        close_called: AtomicBool,
    }

    impl Script {
        /// Queue `(probability, milliseconds)` runs in 20 ms chunks.
        fn feed(&self, runs: &[(f32, usize)]) {
            let signal: Vec<f32> = runs.iter().flat_map(|&(p, ms)| std::iter::repeat_n(p, ms * MS)).collect();
            let mut items = self.items.lock().unwrap();
            for chunk in signal.chunks(20 * MS) {
                items.push_back(Item::Audio(chunk.to_vec()));
            }
        }

        fn gap(&self, restarted: bool, muted: bool) {
            self.items.lock().unwrap().push_back(Item::Gap { restarted, muted });
        }
    }

    struct ScriptSource(Arc<Script>);

    impl AudioSource for ScriptSource {
        fn read(&mut self, out: &mut Vec<f32>) -> SourceRead {
            let item = self.0.items.lock().unwrap().pop_front();
            match item {
                Some(Item::Audio(a)) => {
                    out.extend_from_slice(&a);
                    SourceRead::default()
                }
                Some(Item::Gap { restarted, muted }) => {
                    self.0.muted.store(muted, Ordering::SeqCst);
                    SourceRead {
                        gap_after: true,
                        restarted,
                        end_qpc_ns: None,
                    }
                }
                None => SourceRead::default(),
            }
        }
        fn wait(&self, timeout: Duration) {
            thread::park_timeout(timeout.min(Duration::from_millis(5)));
        }
        fn muted(&self) -> bool {
            self.0.muted.load(Ordering::SeqCst)
        }
        fn ended(&self) -> Option<String> {
            None
        }
        fn is_loopback(&self) -> bool {
            self.0.loopback.load(Ordering::SeqCst)
        }
        fn key(&self) -> String {
            "test-source".into()
        }
        fn dropped_frames(&self) -> u64 {
            0
        }
        fn close(self: Box<Self>) {
            self.0.close_called.store(true, Ordering::SeqCst);
            let gate = self.0.close_gate.lock().unwrap().clone();
            if let Some(gate) = gate {
                gate.wait(|| false);
            }
            self.0.closed.store(true, Ordering::SeqCst);
        }
    }

    // --- fake engines -----------------------------------------------------------------------------

    /// A gate a fake engine (or factory) waits on.
    #[derive(Clone, Default)]
    struct Latch(Arc<(Mutex<bool>, Condvar)>);

    impl Latch {
        fn open(&self) {
            *self.0 .0.lock().unwrap() = true;
            self.0 .1.notify_all();
        }

        /// Wait until open; `false` when `cancelled()` became true first.
        fn wait(&self, cancelled: impl Fn() -> bool) -> bool {
            let mut open = self.0 .0.lock().unwrap();
            while !*open {
                if cancelled() {
                    return false;
                }
                open = self.0 .1.wait_timeout(open, Duration::from_millis(5)).unwrap().0;
            }
            true
        }
    }

    /// Audio length and language of every `transcribe` call.
    type Calls = Arc<Mutex<Vec<(usize, Option<String>)>>>;

    #[derive(Clone, Default)]
    struct Behavior {
        /// Block every `transcribe` until opened.
        gate: Option<Latch>,
        /// Block only the first `transcribe`.
        gate_first: Option<Latch>,
        /// Keep going after a cancel (an engine without a native cancel).
        ignore_cancel: bool,
        fail: Option<AsrErrorKind>,
        text: Option<String>,
        no_speech: Option<f32>,
        lang: Option<String>,
        /// Streaming engine (partials).
        streaming: bool,
        needs_language: bool,
        /// `caps.languages`: a stream refuses (`unsupported`) any other language.
        languages: Vec<String>,
        /// `begin_stream` fails with this.
        refuse_begin: Option<AsrErrorKind>,
        /// `transcribe` gives the utterance up for its deadline (a rate limit).
        give_up: bool,
        /// Audio length and language of every `transcribe`.
        calls: Calls,
        /// Time left to the decode deadline at every `transcribe`.
        deadlines: Arc<Mutex<Vec<Option<Duration>>>>,
        /// Language of every `begin_stream`.
        begins: Arc<Mutex<Vec<Option<String>>>>,
        /// Text of every first-pass (stream) final; `None` = "stream <ms>ms".
        stream_text: Option<String>,
        /// Audio length of every stream that ended.
        streamed: Arc<Mutex<Vec<usize>>>,
    }

    struct FakeEngine {
        info: EngineInfo,
        b: Behavior,
        cancel: Arc<AtomicU64>,
        stream: Option<usize>,
    }

    impl FakeEngine {
        fn new(model: &str, b: Behavior) -> Self {
            Self {
                info: EngineInfo {
                    kind: if b.streaming { EngineKind::SherpaOnline } else { EngineKind::Whisper },
                    model: model.to_string(),
                    runtime: "fake".into(),
                    device: "cpu".into(),
                    caps: EngineCaps {
                        streaming: b.streaming,
                        utterance: !b.streaming,
                        needs_language: b.needs_language,
                        languages: b.languages.clone(),
                        ..EngineCaps::default()
                    },
                    vram_mb: 0,
                    fallback_reason: None,
                },
                b,
                cancel: Arc::new(AtomicU64::new(0)),
                stream: None,
            }
        }
    }

    impl AsrEngine for FakeEngine {
        fn info(&self) -> &EngineInfo {
            &self.info
        }

        fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError> {
            let first = self.b.calls.lock().unwrap().is_empty();
            self.b.calls.lock().unwrap().push((audio.len(), options.language.clone()));
            self.b
                .deadlines
                .lock()
                .unwrap()
                .push(worker::scope::deadline().map(|d| d.saturating_duration_since(Instant::now())));
            if self.b.give_up {
                worker::scope::give_up();
                return Err(AsrError::new(AsrErrorKind::RateLimited, "the next request is allowed in 45 s"));
            }
            let mine = self.cancel.load(Ordering::SeqCst);
            let cancel = Arc::clone(&self.cancel);
            let ignore = self.b.ignore_cancel;
            let cancelled = move || !ignore && cancel.load(Ordering::SeqCst) != mine;
            let gate = self.b.gate.clone().or_else(|| self.b.gate_first.clone().filter(|_| first));
            if let Some(g) = gate {
                if !g.wait(&cancelled) {
                    return Err(AsrError::cancelled());
                }
            }
            if cancelled() {
                return Err(AsrError::cancelled());
            }
            if let Some(kind) = self.b.fail {
                return Err(AsrError::new(kind, "fake failure"));
            }
            Ok(AsrResult {
                text: self.b.text.clone().unwrap_or_else(|| format!("{} {}ms", self.info.model, audio.len() / MS)),
                lang: self.b.lang.clone().or_else(|| options.language.clone()),
                no_speech_prob: self.b.no_speech,
                ..AsrResult::default()
            })
        }

        fn begin_stream(&mut self, options: &DecodeOptions) -> Result<(), AsrError> {
            self.b.begins.lock().unwrap().push(options.language.clone());
            if !self.b.streaming {
                return Err(AsrError::unsupported("not streaming"));
            }
            if let Some(kind) = self.b.refuse_begin {
                return Err(AsrError::new(kind, "the stream was refused"));
            }
            if let Some(l) = &options.language {
                if !self.b.languages.is_empty() && !self.b.languages.contains(l) {
                    return Err(AsrError::unsupported(format!("the model does not know '{l}'")));
                }
            }
            self.stream = Some(0);
            Ok(())
        }

        fn push_stream(&mut self, audio: &[f32]) -> Result<Option<Hypothesis>, AsrError> {
            let Some(total) = self.stream.as_mut() else {
                return Err(AsrError::invalid_input("no stream"));
            };
            *total += audio.len();
            Ok(Some(Hypothesis {
                text: format!("partial {}ms", *total / MS),
                lang: Some("en".into()),
            }))
        }

        fn end_stream(&mut self) -> Result<AsrResult, AsrError> {
            let total = self.stream.take().ok_or_else(|| AsrError::invalid_input("no stream"))?;
            self.b.streamed.lock().unwrap().push(total);
            Ok(AsrResult {
                text: self
                    .b
                    .stream_text
                    .clone()
                    .unwrap_or_else(|| format!("stream {}ms", total / MS)),
                lang: Some("en".into()),
                ..AsrResult::default()
            })
        }

        fn abort_stream(&mut self) {
            self.stream = None;
        }

        fn canceller(&self) -> Option<Canceller> {
            let c = Arc::clone(&self.cancel);
            Some(Canceller::new(move || {
                c.fetch_add(1, Ordering::SeqCst);
            }))
        }
    }

    /// Engine factory: `spec.model` picks the behavior (unknown models are not installed).
    fn engines(map: Vec<(&str, Behavior)>) -> EngineFactory {
        let map: HashMap<String, Behavior> = map.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
        Arc::new(move |spec: &EngineSpec, _ctx: &EngineContext| match map.get(&spec.model) {
            Some(b) => Ok(Box::new(FakeEngine::new(&spec.model, b.clone())) as Box<dyn AsrEngine>),
            None => Err(AsrError::not_installed(spec.model.clone())),
        })
    }

    struct FakeTwoPass {
        tx: Option<mpsc::Sender<FinalJob>>,
        worker: Option<thread::JoinHandle<()>>,
    }

    impl SecondPass for FakeTwoPass {
        fn submit(&self, job: FinalJob) -> bool {
            self.tx.as_ref().is_some_and(|tx| tx.send(job).is_ok())
        }
        fn shutdown(mut self: Box<Self>) {
            self.tx.take();
            if let Some(w) = self.worker.take() {
                let _ = w.join();
            }
        }
    }

    /// A two-pass queue that re-decodes each job with the final engine (no budget).
    fn fake_two_pass() -> SecondPassFactory {
        Box::new(|mut primary, _fallback, _config, mut on_outcome| {
            let (tx, rx) = mpsc::channel::<FinalJob>();
            let worker = thread::spawn(move || {
                while let Ok(job) = rx.recv() {
                    let info = primary.info().clone();
                    match primary.transcribe(&job.audio, &job.options) {
                        Ok(result) if !result.text.is_empty() => on_outcome(TwoPassOutcome::Final {
                            segment_id: job.segment_id,
                            revision: job.revision,
                            result,
                            engine: info.kind.as_str().to_string(),
                            model: info.model,
                            promoted_fallback: false,
                        }),
                        _ => on_outcome(TwoPassOutcome::Skipped {
                            segment_id: job.segment_id,
                            reason: "empty".into(),
                        }),
                    }
                }
            });
            Ok(Box::new(FakeTwoPass {
                tx: Some(tx),
                worker: Some(worker),
            }) as Box<dyn SecondPass>)
        })
    }

    /// [`fake_two_pass`] that also records, per submitted job, whether it carried a first pass.
    fn recording_two_pass(seen: Arc<Mutex<Vec<bool>>>) -> SecondPassFactory {
        struct Recording {
            inner: Box<dyn SecondPass>,
            seen: Arc<Mutex<Vec<bool>>>,
        }
        impl SecondPass for Recording {
            fn submit(&self, job: FinalJob) -> bool {
                self.seen.lock().unwrap().push(job.first_pass.is_some());
                self.inner.submit(job)
            }
            fn shutdown(self: Box<Self>) {
                self.inner.shutdown();
            }
        }
        let inner = fake_two_pass();
        Box::new(move |primary, fallback, config, on_outcome| {
            let inner = inner(primary, fallback, config, on_outcome)?;
            Ok(Box::new(Recording { inner, seen }) as Box<dyn SecondPass>)
        })
    }

    struct FakeTurn(f32);

    impl TurnDetector for FakeTurn {
        fn predict(&mut self, audio_16k: &[f32]) -> Result<f32, String> {
            assert!(audio_16k.len() <= ms_to_samples(TURN_WINDOW_MS));
            Ok(self.0)
        }
    }

    struct Harness {
        registry: Registry,
        rec: Arc<Recorder>,
        script: Arc<Script>,
    }

    impl Harness {
        fn new() -> Self {
            Self {
                registry: Registry::new(),
                rec: Arc::new(Recorder::default()),
                script: Arc::new(Script::default()),
            }
        }

        fn deps(&self, engine_factory: EngineFactory) -> SessionDeps {
            let script = Arc::clone(&self.script);
            SessionDeps {
                sink: Arc::new(RecSink(Arc::clone(&self.rec))),
                engines: engine_factory,
                source: Box::new(move |_source: &str| {
                    script.attached.store(true, Ordering::SeqCst);
                    Ok(Box::new(ScriptSource(script)) as Box<dyn AudioSource>)
                }),
                segmenter: Box::new(|vad: &VadConfig| {
                    Ok(Box::new(SegmenterEngine::new(LevelClassifier, vad)) as Box<dyn SpeechSegmenter>)
                }),
                turn: Box::new(|| Err("Smart Turn is not installed".into())),
                second_pass: Box::new(|_, _, _, _| Err(AsrError::unsupported("two-pass is off in this test"))),
                route: Box::new(|_| RoutePlan::default()),
                requirements: Arc::new(|_| Vec::new()),
                batch_policy: Arc::new(|_| None),
                capture_clock: false,
            }
        }

        async fn start(&self, config: PipelineConfig, deps: SessionDeps) -> PipelineStarted {
            start_session(&self.registry, config, deps, false).await.expect("start")
        }

        async fn stop(&self, id: &str) -> bool {
            let r = stop_session(&self.registry, id).await;
            self.rec.push(Ev::Marker("stopped"));
            r
        }
    }

    fn finals_config(model: &str) -> PipelineConfig {
        PipelineConfig {
            pipeline_id: "mic".into(),
            language: "en".into(),
            final_engine: Some(EngineSpec::new(EngineKind::Whisper, model)),
            ..PipelineConfig::default()
        }
    }

    /// One utterance: 512 ms of silence, `speech_ms` of speech, 1024 ms of silence.
    fn utterance(speech_ms: usize) -> Vec<(f32, usize)> {
        vec![(0.0, 512), (0.9, speech_ms), (0.0, 1024)]
    }

    fn wait_until(what: &str, done: impl Fn() -> bool) {
        let deadline = Instant::now() + WAIT;
        while !done() {
            assert!(Instant::now() < deadline, "timed out waiting for {what}");
            thread::sleep(Duration::from_millis(5));
        }
    }

    // --- tests ------------------------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn utterances_are_framed_numbered_and_decoded_in_order() {
        let h = Harness::new();
        let b = Behavior::default();
        let started = h
            .start(finals_config("fake-final"), h.deps(engines(vec![("fake-final", b.clone())])))
            .await;
        assert_eq!(started.pipeline_id, "mic");
        assert!(started.plan.final_engine.is_some() && started.plan.streaming.is_none());
        h.script.feed(&utterance(1024));
        // One at a time: utterances that wait for a busy engine are merged (tested below).
        h.rec.wait_for("the first final", |e| finals(e).len() == 1);
        h.script.feed(&[(0.9, 640), (0.0, 1024)]);
        let events = h.rec.wait_for("two finals", |e| finals(e).len() == 2);

        let fs = finals(&events);
        let (s1, s2) = (fs[0].segment_id, fs[1].segment_id);
        assert_eq!(s2, s1 + 1, "segment ids count up");
        for f in &fs {
            assert_eq!((f.epoch, f.revision, f.pass), (started.epoch, 0, FinalPass::First));
            assert_eq!(f.pipeline_id, "mic");
            assert_eq!(f.lang.as_deref(), Some("en"));
            assert!(!f.forced && f.turn_end.is_none());
        }
        // 512 ms of silence, speech from 512 ms: the 400 ms pre-roll starts at 112 ms; the segment
        // ends 200 ms after the speech (1736 ms).
        let c = closed(&events);
        assert_eq!(c[0], (s1, 112, 1736, ((1736 - 112) * MS) as u32, false));
        assert_eq!((fs[0].start_ms, fs[0].end_ms), (112, 1736));
        assert_eq!(fs[0].text, format!("fake-final {}ms", 1736 - 112), "the engine got the whole segment");
        // The second utterance's pre-roll never reaches back into the first.
        assert!(c[1].1 >= c[0].2);
        // Per segment: start, closed, final.
        let order: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Ev::Utterance(SpeechUtterance::SpeechStart { segment_id, .. }, _) if *segment_id == s1 => Some("start"),
                Ev::Utterance(SpeechUtterance::Utterance { segment_id, .. }, _) if *segment_id == s1 => Some("closed"),
                Ev::Final(f) if f.segment_id == s1 => Some("final"),
                _ => None,
            })
            .collect();
        assert_eq!(order, ["start", "closed", "final"]);
        // Lifecycle: starting, loading, listening.
        let st = statuses(&events);
        assert_eq!(st[0].state, SpeechState::Starting);
        assert!(st.iter().any(|s| s.state == SpeechState::Loading && s.code == codes::LOADING_MODEL));
        assert!(st.iter().any(|s| s.state == SpeechState::Listening && s.code == codes::LISTENING));
        assert!(st.iter().all(|s| s.epoch == started.epoch && s.pipeline_id == "mic"));

        let status = h.registry.statuses();
        assert_eq!(status.len(), 1);
        assert_eq!(
            (status[0].state, status[0].source.as_str(), status[0].dropped),
            (SpeechState::Listening, "test-source", 0)
        );
        assert_eq!(status[0].final_engine.as_ref().map(|i| i.model.as_str()), Some("fake-final"));
        assert!(h.stop("mic").await);
        assert!(h.script.closed.load(Ordering::SeqCst), "the device is released when stop returns");
        assert!(h.registry.statuses().is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn no_event_of_a_stopped_session_is_ever_delivered() {
        for ignore_cancel in [false, true] {
            let h = Harness::new();
            let gate = Latch::default();
            let b = Behavior {
                gate: Some(gate.clone()),
                ignore_cancel,
                ..Behavior::default()
            };
            let factory = engines(vec![("fake-final", b.clone())]);
            let first = h.start(finals_config("fake-final"), h.deps(Arc::clone(&factory))).await;
            h.script.feed(&utterance(640));
            h.script.feed(&utterance(640));
            // The engine holds the first utterance; the second waits in the queue.
            h.rec.wait_for("two closed utterances", |e| count_closed(e) == 2);
            wait_until("the decode", || !b.calls.lock().unwrap().is_empty());
            assert!(h.stop("mic").await);
            // The decode in flight finishes (or is cancelled) after the stop.
            gate.open();
            h.script.feed(&utterance(640));
            thread::sleep(Duration::from_millis(150));
            let events = h.rec.all();
            let marker = events.iter().position(|e| matches!(e, Ev::Marker("stopped"))).unwrap();
            assert!(
                events[marker..].iter().all(|e| e.epoch() != Some(first.epoch)),
                "ignore_cancel={ignore_cancel}: stale events {:#?}",
                &events[marker..]
            );
            // The ended session's last event is `stopped`.
            let last = events[..marker].iter().rev().find(|e| e.epoch() == Some(first.epoch)).unwrap();
            assert!(matches!(last, Ev::Status(s) if s.state == SpeechState::Stopped), "{last:?}");
            assert!(finals(&events).is_empty(), "nothing was decoded before the stop");

            // A new session gets a new epoch, and segment ids are never reused.
            let seen: Vec<u64> = utterances(&events).into_iter().map(|(_, id)| id).collect();
            h.script.items.lock().unwrap().clear();
            let second = h.start(finals_config("fake-final"), h.deps(factory)).await;
            assert!(second.epoch > first.epoch);
            h.script.feed(&utterance(640));
            let events = h
                .rec
                .wait_for("a final of the new session", |e| finals(e).iter().any(|f| f.epoch == second.epoch));
            let f = finals(&events).into_iter().find(|f| f.epoch == second.epoch).unwrap();
            assert!(seen.iter().all(|&old| f.segment_id > old), "{} reused ({seen:?})", f.segment_id);
            h.stop("mic").await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_restart_ends_the_previous_session_with_stopped() {
        let h = Harness::new();
        let factory = engines(vec![("fake-final", Behavior::default())]);
        let first = h.start(finals_config("fake-final"), h.deps(Arc::clone(&factory))).await;
        let second = h.start(finals_config("fake-final"), h.deps(factory)).await;
        assert!(second.epoch > first.epoch);
        let events = h.rec.all();
        let first_stop = events
            .iter()
            .position(|e| matches!(e, Ev::Status(s) if s.epoch == first.epoch && s.state == SpeechState::Stopped))
            .expect("stopped");
        assert!(events[first_stop + 1..].iter().all(|e| e.epoch() != Some(first.epoch)));
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("a final", |e| !finals(e).is_empty());
        assert_eq!(finals(&events)[0].epoch, second.epoch);
        assert_eq!(h.registry.statuses().len(), 1);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_mute_gate_drops_the_open_utterance_and_speech_after_it_works() {
        let h = Harness::new();
        let b = Behavior::default();
        h.start(finals_config("fake-final"), h.deps(engines(vec![("fake-final", b.clone())])))
            .await;
        // Speech confirmed (> 250 ms), then the mute starts mid-utterance.
        h.script.feed(&[(0.0, 512), (0.9, 640)]);
        h.script.gap(false, true);
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("a final after the mute", |e| !finals(e).is_empty());
        let u = utterances(&events);
        let s1 = u[0].1;
        assert_eq!(u[0], ("start".to_string(), s1));
        assert_eq!(u[1], ("muted".to_string(), s1), "{u:?}");
        assert!(!u.iter().any(|(k, id)| k == "closed" && *id == s1), "the muted utterance never closes");
        let fs = finals(&events);
        assert_eq!(fs.len(), 1);
        assert!(fs[0].segment_id > s1);
        assert_eq!(b.calls.lock().unwrap().len(), 1, "muted speech is never decoded");
        assert_eq!(h.registry.statuses()[0].dropped, 1);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_device_restart_closes_the_open_utterance_and_decodes_it() {
        let h = Harness::new();
        h.start(finals_config("fake-final"), h.deps(engines(vec![("fake-final", Behavior::default())])))
            .await;
        h.script.feed(&[(0.0, 512), (0.9, 800)]);
        h.script.gap(true, false);
        let events = h.rec.wait_for("the forced final and the warning", |e| {
            !finals(e).is_empty() && statuses(e).iter().any(|s| s.code == codes::CAPTURE_GAP)
        });
        assert!(finals(&events)[0].forced);
        assert!(closed(&events)[0].4);
        assert!(statuses(&events)
            .iter()
            .any(|s| s.code == codes::CAPTURE_GAP && s.level == StatusLevel::Warning));
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_backlog_cap_drops_the_oldest_waiting_utterance() {
        let h = Harness::new();
        let gate = Latch::default();
        let b = Behavior {
            gate_first: Some(gate.clone()),
            ..Behavior::default()
        };
        let config = PipelineConfig {
            max_backlog: 1,
            ..finals_config("fake-final")
        };
        h.start(config, h.deps(engines(vec![("fake-final", b.clone())]))).await;
        h.script.feed(&utterance(640));
        h.rec.wait_for("the first closed utterance", |e| count_closed(e) == 1);
        wait_until("the first decode", || !b.calls.lock().unwrap().is_empty());
        h.script.feed(&utterance(640));
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("three closed utterances", |e| count_closed(e) == 3);
        let ids: Vec<u64> = closed(&events).iter().map(|c| c.0).collect();
        gate.open();
        let events = h.rec.wait_for("two finals and the drop", |e| {
            finals(e).len() == 2 && utterances(e).iter().any(|(k, _)| k == "backlog")
        });
        let u = utterances(&events);
        assert!(u.contains(&("backlog".to_string(), ids[1])), "{u:?}");
        let got: Vec<u64> = finals(&events).iter().map(|f| f.segment_id).collect();
        assert_eq!(got, vec![ids[0], ids[2]]);
        assert!(statuses(&events).iter().any(|s| s.code == codes::BACKLOG_DROPPED));
        assert_eq!(h.registry.statuses()[0].dropped, 1);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn waiting_utterances_are_merged_into_one_decode() {
        let h = Harness::new();
        let gate = Latch::default();
        let b = Behavior {
            gate_first: Some(gate.clone()),
            ..Behavior::default()
        };
        h.start(finals_config("fake-final"), h.deps(engines(vec![("fake-final", b.clone())])))
            .await;
        h.script.feed(&utterance(640));
        h.rec.wait_for("the first closed utterance", |e| count_closed(e) == 1);
        wait_until("the first decode", || !b.calls.lock().unwrap().is_empty());
        h.script.feed(&utterance(640));
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("three closed utterances", |e| count_closed(e) == 3);
        let c = closed(&events);
        gate.open();
        let merged = ("merged".to_string(), c[2].0);
        let events = h
            .rec
            .wait_for("the merged final", |e| finals(e).len() == 2 && utterances(e).contains(&merged));
        let fs = finals(&events);
        assert_eq!(fs[0].segment_id, c[0].0);
        assert_eq!(fs[1].segment_id, c[1].0, "the batch's final is its first segment's");
        assert_eq!((fs[1].start_ms, fs[1].end_ms), (c[1].1, c[2].2), "and spans the whole batch");
        let lens: Vec<usize> = b.calls.lock().unwrap().iter().map(|c| c.0).collect();
        let gap = ms_to_samples(super::super::utterance::MERGE_GAP_MS);
        assert_eq!(lens, vec![c[0].3 as usize, c[1].3 as usize + gap + c[2].3 as usize]);
        assert_eq!(h.registry.statuses()[0].dropped, 0, "merging loses nothing");
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn utterances_older_than_the_latency_budget_are_dropped() {
        let h = Harness::new();
        let gate = Latch::default();
        let b = Behavior {
            gate_first: Some(gate.clone()),
            ..Behavior::default()
        };
        let config = PipelineConfig {
            latency_budget_ms: 1_000,
            ..finals_config("fake-final")
        };
        h.start(config, h.deps(engines(vec![("fake-final", b.clone())]))).await;
        h.script.feed(&utterance(640));
        h.rec.wait_for("the first closed utterance", |e| count_closed(e) == 1);
        wait_until("the first decode", || !b.calls.lock().unwrap().is_empty());
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("two closed utterances", |e| count_closed(e) == 2);
        let c = closed(&events);
        thread::sleep(Duration::from_millis(1_200));
        gate.open();
        let events = h.rec.wait_for("the late one dropped", |e| {
            utterances(e).iter().any(|(k, _)| k == "latency_budget") && finals(e).len() == 1
        });
        assert!(utterances(&events).contains(&("latency_budget".to_string(), c[1].0)));
        assert_eq!(finals(&events)[0].segment_id, c[0].0);
        assert_eq!(b.calls.lock().unwrap().len(), 1, "a stale utterance is never decoded");
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn empty_and_hallucinated_results_end_with_no_speech() {
        for (text, no_speech) in [("", None), ("Thanks for watching!", Some(0.9)), ("[MUSIC]", None)] {
            let h = Harness::new();
            let b = Behavior {
                text: Some(text.into()),
                no_speech,
                ..Behavior::default()
            };
            h.start(finals_config("fake-final"), h.deps(engines(vec![("fake-final", b)]))).await;
            h.script.feed(&utterance(640));
            let events = h.rec.wait_for("no_speech", |e| utterances(e).iter().any(|(k, _)| k == "no_speech"));
            assert!(finals(&events).is_empty(), "{text}");
            h.stop("mic").await;
        }
        // With the filter off, a stock phrase is kept (whitespace normalized).
        let h = Harness::new();
        let b = Behavior {
            text: Some("  Thanks   for watching! ".into()),
            no_speech: Some(0.9),
            ..Behavior::default()
        };
        let config = PipelineConfig {
            hallucination_filter: false,
            ..finals_config("fake-final")
        };
        h.start(config, h.deps(engines(vec![("fake-final", b)]))).await;
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("a final", |e| !finals(e).is_empty());
        assert_eq!(finals(&events)[0].text, "Thanks for watching!");
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_fatal_engine_error_ends_the_session_with_error() {
        let h = Harness::new();
        let b = Behavior {
            fail: Some(AsrErrorKind::NotInstalled),
            ..Behavior::default()
        };
        let started = h.start(finals_config("stt/fake"), h.deps(engines(vec![("stt/fake", b)]))).await;
        h.script.feed(&utterance(640));
        let events = h
            .rec
            .wait_for("the error", |e| statuses(e).iter().any(|s| s.state == SpeechState::Error));
        let err = statuses(&events).into_iter().find(|s| s.state == SpeechState::Error).unwrap();
        assert_eq!(
            (err.code.as_str(), err.level, err.model_id.as_deref()),
            (codes::NOT_INSTALLED, StatusLevel::Error, Some("stt/fake"))
        );
        assert!(utterances(&events).iter().any(|(k, _)| k == "engine_error"));
        // The error is the last event of the epoch; the session is gone and the device released.
        wait_until("the device release", || h.script.closed.load(Ordering::SeqCst));
        assert!(h.registry.statuses().is_empty());
        h.script.feed(&utterance(640));
        thread::sleep(Duration::from_millis(100));
        let events = h.rec.all();
        let err_at = events
            .iter()
            .position(|e| matches!(e, Ev::Status(s) if s.state == SpeechState::Error))
            .unwrap();
        assert!(events[err_at + 1..].iter().all(|e| e.epoch() != Some(started.epoch)));
        assert!(!h.stop("mic").await, "nothing left to stop");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn transient_errors_warn_until_five_in_a_row() {
        let h = Harness::new();
        let b = Behavior {
            fail: Some(AsrErrorKind::Network),
            ..Behavior::default()
        };
        h.start(finals_config("fake-final"), h.deps(engines(vec![("fake-final", b.clone())])))
            .await;
        h.script.feed(&utterance(640));
        let events = h
            .rec
            .wait_for("a warning", |e| statuses(e).iter().any(|s| s.code == codes::ENGINE_ERROR));
        assert!(statuses(&events).iter().all(|s| s.state != SpeechState::Error), "one failure is not fatal");
        for n in 2..=5 {
            h.script.feed(&utterance(640));
            h.rec.wait_for("the next failure", |e| {
                utterances(e).iter().filter(|(k, _)| k == "engine_error").count() == n
            });
        }
        let events = h
            .rec
            .wait_for("the error", |e| statuses(e).iter().any(|s| s.state == SpeechState::Error));
        assert_eq!(utterances(&events).iter().filter(|(k, _)| k == "engine_error").count(), 5);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_engine_that_cannot_load_fails_the_start() {
        let h = Harness::new();
        let err = start_session(&h.registry, finals_config("stt/missing"), h.deps(engines(vec![])), false)
            .await
            .unwrap_err();
        assert!(err.contains("not_installed") && err.contains("stt/missing"), "{err}");
        let st = statuses(&h.rec.all());
        let last = st.last().unwrap();
        assert_eq!(
            (last.state, last.code.as_str(), last.model_id.as_deref()),
            (SpeechState::Error, codes::NOT_INSTALLED, Some("stt/missing"))
        );
        assert!(!h.script.attached.load(Ordering::SeqCst), "the device is never opened");
        assert!(h.registry.statuses().is_empty());
        // A route without engines fails too, unless PCM is retained.
        let err = start_session(
            &h.registry,
            PipelineConfig {
                pipeline_id: "mic".into(),
                ..PipelineConfig::default()
            },
            h.deps(engines(vec![])),
            false,
        )
        .await
        .unwrap_err();
        assert!(err.contains("no speech engine"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stop_during_a_model_load_cancels_the_start() {
        let h = Arc::new(Harness::new());
        let gate = Latch::default();
        let loading: EngineFactory = {
            let gate = gate.clone();
            let inner = engines(vec![("fake-final", Behavior::default())]);
            Arc::new(move |spec: &EngineSpec, ctx: &EngineContext| {
                gate.wait(|| false);
                inner(spec, ctx)
            })
        };
        let deps = h.deps(loading);
        let task = {
            let h = Arc::clone(&h);
            tokio::spawn(async move { start_session(&h.registry, finals_config("fake-final"), deps, false).await })
        };
        h.rec
            .wait_for("loading", |e| statuses(e).iter().any(|s| s.state == SpeechState::Loading));
        assert!(h.stop("mic").await, "a starting session counts as running");
        gate.open();
        let err = task.await.unwrap().unwrap_err();
        assert_eq!(err, "cancelled");
        let events = h.rec.all();
        let marker = events.iter().position(|e| matches!(e, Ev::Marker("stopped"))).unwrap();
        assert!(events[marker..].iter().all(|e| e.epoch().is_none()), "{:#?}", &events[marker..]);
        assert!(!h.script.attached.load(Ordering::SeqCst));
        assert!(h.registry.statuses().is_empty());
    }

    fn streaming_config(stream: &str, final_model: Option<&str>) -> PipelineConfig {
        PipelineConfig {
            pipeline_id: "mic".into(),
            language: "en".into(),
            streaming: Some(EngineSpec::new(EngineKind::SherpaOnline, stream)),
            final_engine: final_model.map(|m| EngineSpec::new(EngineKind::SherpaOffline, m)),
            ..PipelineConfig::default()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn streaming_partials_precede_the_final_with_rising_revisions() {
        let h = Harness::new();
        let s = Behavior {
            streaming: true,
            ..Behavior::default()
        };
        h.start(streaming_config("stream", None), h.deps(engines(vec![("stream", s.clone())])))
            .await;
        // Feed slowly enough for several partials (they are throttled to one per 80 ms).
        h.script.feed(&[(0.0, 512)]);
        for _ in 0..8 {
            h.script.feed(&[(0.9, 128)]);
            thread::sleep(Duration::from_millis(100));
        }
        h.script.feed(&[(0.0, 1024)]);
        let events = h.rec.wait_for("the streaming final", |e| !finals(e).is_empty());
        let f = &finals(&events)[0];
        let ps: Vec<SpeechPartial> = partials(&events).into_iter().filter(|p| p.segment_id == f.segment_id).collect();
        assert!(ps.len() >= 2, "{ps:?}");
        for (i, p) in ps.iter().enumerate() {
            assert_eq!(p.revision, i as u32, "partials count revisions from 0");
            assert_eq!(p.lang.as_deref(), Some("en"));
        }
        assert_eq!(f.revision, ps.len() as u32, "the final follows the last partial");
        assert_eq!((f.pass, f.engine.as_str()), (FinalPass::First, "sherpa-online"));
        // The stream got the segment from its pre-roll to its end, plus a little trailing silence.
        let streamed = s.streamed.lock().unwrap()[0];
        let c = closed(&events)[0];
        assert!(streamed >= c.3 as usize, "streamed {streamed} of {} samples", c.3);
        assert!(streamed <= c.3 as usize + 1024 * MS, "trailing silence only ({streamed})");
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_forced_split_never_streams_audio_twice() {
        let h = Harness::new();
        let s = Behavior {
            streaming: true,
            ..Behavior::default()
        };
        let config = PipelineConfig {
            vad: VadConfig {
                max_utterance_ms: 4_000,
                ..VadConfig::default()
            },
            ..streaming_config("stream", None)
        };
        h.start(config, h.deps(engines(vec![("stream", s.clone())]))).await;
        h.script.feed(&[(0.0, 512), (0.9, 6_000), (0.0, 1024)]);
        let events = h.rec.wait_for("two finals", |e| finals(e).len() == 2);
        let total_closed: u64 = closed(&events).iter().map(|c| u64::from(c.3)).sum();
        let streamed: usize = s.streamed.lock().unwrap().iter().sum();
        assert!(finals(&events)[0].forced);
        assert!(
            streamed as u64 <= total_closed + (1024 * MS) as u64,
            "streamed {streamed}, segments {total_closed}"
        );
        assert!(
            streamed as u64 + (400 * MS) as u64 >= total_closed,
            "streamed {streamed}, segments {total_closed}"
        );
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_pass_finals_replace_the_first_under_the_same_segment() {
        let h = Harness::new();
        let s = Behavior {
            streaming: true,
            ..Behavior::default()
        };
        let f = Behavior {
            text: Some("second pass".into()),
            ..Behavior::default()
        };
        let mut deps = h.deps(engines(vec![("stream", s), ("final", f.clone())]));
        deps.second_pass = fake_two_pass();
        h.start(streaming_config("stream", Some("final")), deps).await;
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("two finals", |e| finals(e).len() == 2);
        let fs = finals(&events);
        assert_eq!(fs[0].segment_id, fs[1].segment_id);
        assert_eq!((fs[0].pass, fs[1].pass), (FinalPass::First, FinalPass::Second));
        assert_eq!(fs[1].revision, fs[0].revision + 1);
        assert_eq!((fs[1].text.as_str(), fs[1].model.as_str()), ("second pass", "final"));
        assert_eq!((fs[1].start_ms, fs[1].end_ms), (fs[0].start_ms, fs[0].end_ms));
        // The re-decode got exactly the closed segment.
        assert_eq!(f.calls.lock().unwrap()[0].0, closed(&events)[0].3 as usize);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn without_a_known_language_a_needy_stream_waits_for_the_second_pass() {
        let h = Harness::new();
        let s = Behavior {
            streaming: true,
            needs_language: true,
            ..Behavior::default()
        };
        let f = Behavior {
            lang: Some("ja".into()),
            ..Behavior::default()
        };
        let mut deps = h.deps(engines(vec![("stream", s.clone()), ("final", f)]));
        deps.second_pass = fake_two_pass();
        let config = PipelineConfig {
            language: "auto".into(),
            ..streaming_config("stream", Some("final"))
        };
        h.start(config, deps).await;
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("the second-pass final", |e| !finals(e).is_empty());
        let fs = finals(&events);
        assert_eq!((fs.len(), fs[0].pass, fs[0].revision), (1, FinalPass::Second, 0));
        assert_eq!(fs[0].lang.as_deref(), Some("ja"));
        assert!(partials(&events).is_empty() && s.streamed.lock().unwrap().is_empty());
        // The detected language lets the next utterance stream.
        h.script.feed(&[(0.9, 640), (0.0, 1024)]);
        h.rec
            .wait_for("a streamed final", |e| finals(e).iter().any(|f| f.pass == FinalPass::First));
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_needy_stream_without_a_second_pass_still_streams() {
        let h = Harness::new();
        let s = Behavior {
            streaming: true,
            needs_language: true,
            ..Behavior::default()
        };
        let config = PipelineConfig {
            language: "auto".into(),
            ..streaming_config("stream", None)
        };
        h.start(config, h.deps(engines(vec![("stream", s)]))).await;
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("the streamed final", |e| !finals(e).is_empty());
        assert_eq!(finals(&events)[0].pass, FinalPass::First);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn smart_turn_verdicts_precede_finals_and_skip_forced_splits() {
        for (p, emitted) in [(0.9f32, true), (0.2, false)] {
            let h = Harness::new();
            let mut deps = h.deps(engines(vec![("fake-final", Behavior::default())]));
            deps.turn = Box::new(move || Ok(Box::new(FakeTurn(p)) as Box<dyn TurnDetector>));
            let config = PipelineConfig {
                smart_turn: true,
                vad: VadConfig {
                    max_utterance_ms: 4_000,
                    ..VadConfig::default()
                },
                ..finals_config("fake-final")
            };
            h.start(config, deps).await;
            h.script.feed(&utterance(640));
            let events = h.rec.wait_for("a final", |e| !finals(e).is_empty());
            let f = &finals(&events)[0];
            assert_eq!(f.turn_end, Some(emitted));
            let turn_at = events
                .iter()
                .position(|e| matches!(e, Ev::TurnEnd(t) if t.segment_id == f.segment_id));
            let closed_at = events
                .iter()
                .position(|e| matches!(e, Ev::Utterance(SpeechUtterance::Utterance { segment_id, .. }, _) if *segment_id == f.segment_id))
                .unwrap();
            let final_at = events
                .iter()
                .position(|e| matches!(e, Ev::Final(x) if x.segment_id == f.segment_id))
                .unwrap();
            match turn_at {
                Some(t) => {
                    assert!(emitted && closed_at < t && t < final_at);
                    let Ev::TurnEnd(t) = &events[t] else { unreachable!() };
                    assert_eq!(t.probability, p);
                }
                None => assert!(!emitted),
            }
            // A forced split is not a pause: no verdict. Its final may also carry the remainder
            // (when both waited for the engine together); it then ends where the remainder ended,
            // with the remainder's verdict.
            h.script.feed(&[(0.9, 5_000), (0.0, 1024)]);
            let events = h.rec.wait_for("a forced utterance", |e| closed(e).iter().any(|c| c.4));
            let forced = closed(&events).into_iter().find(|c| c.4).unwrap().0;
            let rest = forced + 1;
            let rest_merged = ("merged".to_string(), rest);
            let events = h.rec.wait_for("both parts' outcomes", |e| {
                finals(e).iter().any(|f| f.segment_id == forced)
                    && (finals(e).iter().any(|f| f.segment_id == rest) || utterances(e).contains(&rest_merged))
            });
            let f = finals(&events).into_iter().find(|f| f.segment_id == forced).unwrap();
            let turn_on_forced = events
                .iter()
                .any(|e| matches!(e, Ev::TurnEnd(t) if t.segment_id == forced));
            if utterances(&events).contains(&rest_merged) {
                assert_eq!(f.turn_end, Some(emitted));
                assert_eq!(turn_on_forced, emitted, "a turn end is repeated for the carrying segment");
            } else {
                assert_eq!(f.turn_end, None);
                assert!(!turn_on_forced);
            }
            h.stop("mic").await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_missing_smart_turn_model_only_warns() {
        let h = Harness::new();
        let config = PipelineConfig {
            smart_turn: true,
            ..finals_config("fake-final")
        };
        h.start(config, h.deps(engines(vec![("fake-final", Behavior::default())]))).await;
        let st = statuses(&h.rec.all());
        assert!(st
            .iter()
            .any(|s| s.code == codes::NOT_INSTALLED && s.model_id.as_deref() == Some(smart_turn::MODEL_ID)));
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("a final", |e| !finals(e).is_empty());
        assert_eq!(finals(&events)[0].turn_end, None);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn capture_only_pipelines_retain_pcm_once() {
        let h = Harness::new();
        let config = PipelineConfig {
            pipeline_id: "loopback".into(),
            retain_pcm: true,
            capture_only: true,
            ..PipelineConfig::default()
        };
        // A router that would plan an engine is never asked: capture-only stays engine-free.
        let mut deps = h.deps(engines(vec![("m", Behavior::default())]));
        deps.route = Box::new(|_| RoutePlan {
            final_engine: Some(EngineSpec::new(EngineKind::Whisper, "m")),
            ..RoutePlan::default()
        });
        let started = h.start(config, deps).await;
        assert!(started.plan.final_engine.is_none());
        assert_eq!(started.plan.notes, vec!["capture_only".to_string()]);
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("a closed utterance", |e| count_closed(e) == 1);
        let (id, _, _, samples, _) = closed(&events)[0];
        let pcm = h.registry.take_pcm("loopback", id).unwrap();
        assert_eq!(pcm.len(), samples as usize);
        assert!(h.registry.take_pcm("loopback", id).is_none(), "taken once");
        assert!(h.registry.take_pcm("mic", id).is_none());
        thread::sleep(Duration::from_millis(50));
        assert!(finals(&h.rec.all()).is_empty());
        h.stop("loopback").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn language_id_runs_first_in_auto_mode() {
        let h = Harness::new();
        let f = Behavior::default();
        let lid = Behavior {
            text: Some(String::new()),
            lang: Some("ja-JP".into()),
            ..Behavior::default()
        };
        let mut deps = h.deps(engines(vec![("final", f.clone()), ("lid", lid)]));
        deps.route = Box::new(|_| RoutePlan {
            language: "auto".into(),
            tier: Tier::Cpu,
            final_engine: Some(EngineSpec::new(EngineKind::SherpaOffline, "final")),
            language_id: Some(EngineSpec::new(EngineKind::SherpaOffline, "lid")),
            ..RoutePlan::default()
        });
        h.start(PipelineConfig::default(), deps).await;
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("a final", |e| !finals(e).is_empty());
        assert_eq!(f.calls.lock().unwrap()[0].1.as_deref(), Some("ja"));
        assert_eq!(finals(&events)[0].lang.as_deref(), Some("ja"));
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_failing_final_engine_is_replaced_by_its_fallback() {
        let h = Harness::new();
        let gpu = Behavior {
            fail: Some(AsrErrorKind::Runtime),
            ..Behavior::default()
        };
        let mut deps = h.deps(engines(vec![("gpu", gpu), ("cpu", Behavior::default())]));
        deps.route = Box::new(|_| RoutePlan {
            language: "en".into(),
            tier: Tier::Gpu,
            final_engine: Some(EngineSpec::new(EngineKind::Transcribe, "gpu")),
            fallback: Some(EngineSpec::new(EngineKind::SherpaOffline, "cpu")),
            ..RoutePlan::default()
        });
        h.start(PipelineConfig::default(), deps).await;
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("a final", |e| !finals(e).is_empty());
        assert_eq!(finals(&events)[0].model, "cpu");
        assert!(statuses(&events).iter().any(|s| s.code == codes::ENGINE_FALLBACK));
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_batch_policy_holds_short_utterances_for_the_billing_minimum() {
        struct Min(f32);
        impl BatchPolicy for Min {
            fn min_batch_seconds(&self) -> f32 {
                self.0
            }
        }
        let h = Harness::new();
        let b = Behavior::default();
        let mut deps = h.deps(engines(vec![("groq", b.clone())]));
        deps.batch_policy = Arc::new(|_| Some(Arc::new(Min(10.0)) as Arc<dyn BatchPolicy>));
        h.start(finals_config("groq"), deps).await;
        h.script.feed(&utterance(640));
        h.rec.wait_for("one closed", |e| count_closed(e) == 1);
        thread::sleep(Duration::from_millis(300));
        assert!(b.calls.lock().unwrap().is_empty(), "held for more audio");
        h.script.feed(&[(0.9, 640), (0.0, 1024)]);
        let events = h.rec.wait_for("the merged final", |e| {
            !finals(e).is_empty() && utterances(e).iter().any(|(k, _)| k == "merged")
        });
        assert_eq!(b.calls.lock().unwrap().len(), 1, "one request for both");
        let c = closed(&events);
        assert!(utterances(&events).contains(&("merged".to_string(), c[1].0)));
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_unsupported_detected_language_skips_the_stream_instead_of_failing() {
        let h = Harness::new();
        let s = Behavior {
            streaming: true,
            needs_language: true,
            languages: vec!["en".into()],
            ..Behavior::default()
        };
        let f = Behavior {
            lang: Some("ja".into()),
            ..Behavior::default()
        };
        let mut deps = h.deps(engines(vec![("stream", s.clone()), ("final", f)]));
        deps.second_pass = fake_two_pass();
        let config = PipelineConfig {
            language: "auto".into(),
            ..streaming_config("stream", Some("final"))
        };
        h.start(config, deps).await;
        h.script.feed(&utterance(640));
        h.rec.wait_for("the first final", |e| finals(e).len() == 1);
        // `ja` is now the detected language, which the streaming model does not list: the next
        // utterance is left to the two-pass final instead of opening a stream that must fail.
        h.script.feed(&[(0.9, 640), (0.0, 1024)]);
        let events = h.rec.wait_for("the second final", |e| finals(e).len() == 2);
        assert!(
            statuses(&events).iter().all(|st| st.state != SpeechState::Error),
            "{:#?}",
            statuses(&events)
        );
        assert!(finals(&events)
            .iter()
            .all(|f| f.pass == FinalPass::Second && f.lang.as_deref() == Some("ja")));
        assert!(s.begins.lock().unwrap().is_empty(), "no stream was opened");
        assert!(partials(&events).is_empty());
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_auto_plan_streams_only_the_languages_its_route_streams() {
        // The router's auto plan limits the stream (`streamLanguages`): a detected zh goes to the
        // final alone, a detected ja streams (the engine itself lists no languages).
        for (detected, streams) in [("zh", false), ("ja", true)] {
            let h = Harness::new();
            let s = Behavior {
                streaming: true,
                needs_language: true,
                ..Behavior::default()
            };
            let f = Behavior {
                lang: Some(detected.into()),
                ..Behavior::default()
            };
            let mut deps = h.deps(engines(vec![("stream", s.clone()), ("final", f)]));
            deps.second_pass = fake_two_pass();
            let mut options = serde_json::Map::new();
            options.insert(router::STREAM_LANGUAGES_OPTION.into(), serde_json::json!(["en", "ja"]));
            let stream = EngineSpec {
                options: serde_json::Value::Object(options),
                ..EngineSpec::new(EngineKind::SherpaOnline, "stream")
            };
            let config = PipelineConfig {
                language: "auto".into(),
                streaming: Some(stream),
                ..streaming_config("stream", Some("final"))
            };
            h.start(config, deps).await;
            // The first utterance teaches the language (no stream yet).
            h.script.feed(&utterance(640));
            h.rec.wait_for("the first final", |e| finals(e).len() == 1);
            h.script.feed(&[(0.9, 640), (0.0, 1024)]);
            let events = if streams {
                h.rec
                    .wait_for("a streamed final", |e| finals(e).iter().any(|f| f.pass == FinalPass::First))
            } else {
                h.rec.wait_for("the second final", |e| finals(e).len() == 2)
            };
            let begins = s.begins.lock().unwrap().clone();
            if streams {
                assert_eq!(begins, [Some("ja".to_string())], "{detected}");
            } else {
                assert!(begins.is_empty(), "{detected}: {begins:?}");
                assert!(finals(&events)
                    .iter()
                    .all(|f| f.pass == FinalPass::Second && f.lang.as_deref() == Some("zh")));
                assert!(partials(&events).is_empty());
            }
            h.stop("mic").await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_first_pass_the_post_filter_removed_is_no_first_final_for_the_second_pass() {
        // A filtered first pass was never emitted: its job must get the final-only budget (and
        // never be evicted), so it may not claim a first pass; an emitted one does.
        for (stream_text, emitted) in [("[MUSIC]", false), ("hello there", true)] {
            let h = Harness::new();
            let s = Behavior {
                streaming: true,
                stream_text: Some(stream_text.into()),
                ..Behavior::default()
            };
            let f = Behavior {
                text: Some("second pass".into()),
                ..Behavior::default()
            };
            let seen = Arc::new(Mutex::new(Vec::new()));
            let mut deps = h.deps(engines(vec![("stream", s), ("final", f)]));
            deps.second_pass = recording_two_pass(Arc::clone(&seen));
            h.start(streaming_config("stream", Some("final")), deps).await;
            h.script.feed(&utterance(640));
            let events = h
                .rec
                .wait_for("the second-pass final", |e| finals(e).iter().any(|f| f.pass == FinalPass::Second));
            assert_eq!(*seen.lock().unwrap(), [emitted], "{stream_text}");
            let passes: Vec<FinalPass> = finals(&events).iter().map(|f| f.pass).collect();
            if emitted {
                assert_eq!(passes, [FinalPass::First, FinalPass::Second]);
            } else {
                assert_eq!(passes, [FinalPass::Second], "the filtered first pass was never emitted");
            }
            h.stop("mic").await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stream_the_engine_refuses_costs_only_its_segment() {
        let h = Harness::new();
        let s = Behavior {
            streaming: true,
            refuse_begin: Some(AsrErrorKind::Unsupported),
            ..Behavior::default()
        };
        // The final engine knows only English: its re-decodes get no `ja` hint (it detects).
        let f = Behavior {
            text: Some("second pass".into()),
            languages: vec!["en".into()],
            ..Behavior::default()
        };
        let mut deps = h.deps(engines(vec![("stream", s.clone()), ("final", f.clone())]));
        deps.second_pass = fake_two_pass();
        let config = PipelineConfig {
            language: "ja".into(),
            ..streaming_config("stream", Some("final"))
        };
        h.start(config, deps).await;
        // More than the five failures in a row that would end the pipeline.
        for n in 1..=6 {
            h.script.feed(&utterance(640));
            h.rec.wait_for("the next final", |e| finals(e).len() == n);
        }
        let events = h.rec.all();
        assert!(statuses(&events).iter().all(|st| st.state != SpeechState::Error));
        assert!(statuses(&events)
            .iter()
            .any(|st| st.code == codes::ENGINE_ERROR && st.level == StatusLevel::Warning));
        assert!(finals(&events).iter().all(|f| f.pass == FinalPass::Second));
        assert_eq!(s.begins.lock().unwrap().len(), 6);
        let calls = f.calls.lock().unwrap().clone();
        assert!(calls.iter().all(|(_, l)| l.is_none()), "{calls:?}");
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn needy_streams_run_when_the_second_pass_cannot_start() {
        let h = Harness::new();
        let s = Behavior {
            streaming: true,
            needs_language: true,
            ..Behavior::default()
        };
        // The harness's two-pass factory fails, and no language-ID engine is routed.
        let deps = h.deps(engines(vec![("stream", s), ("final", Behavior::default())]));
        let config = PipelineConfig {
            language: "auto".into(),
            ..streaming_config("stream", Some("final"))
        };
        h.start(config, deps).await;
        h.rec.wait_for("the two-pass warning", |e| {
            statuses(e).iter().any(|st| st.code == codes::ENGINE_FALLBACK)
        });
        h.script.feed(&utterance(640));
        let events = h
            .rec
            .wait_for("a streamed final", |e| finals(e).iter().any(|f| f.pass == FinalPass::First));
        assert!(!utterances(&events).iter().any(|(k, _)| k == reasons::STREAM_SKIPPED));
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stop_cancels_a_start_still_waiting_behind_a_model_load() {
        let h = Arc::new(Harness::new());
        let gate = Latch::default();
        let loading: EngineFactory = {
            let gate = gate.clone();
            let inner = engines(vec![("fake-final", Behavior::default())]);
            Arc::new(move |spec: &EngineSpec, ctx: &EngineContext| {
                gate.wait(|| false);
                inner(spec, ctx)
            })
        };
        let spawn_start = |deps: SessionDeps| {
            let h = Arc::clone(&h);
            tokio::spawn(async move { start_session(&h.registry, finals_config("fake-final"), deps, false).await })
        };
        let first = spawn_start(h.deps(Arc::clone(&loading)));
        h.rec
            .wait_for("loading", |e| statuses(e).iter().any(|s| s.state == SpeechState::Loading));
        // A second start (a settings change) queues behind the load, then STT is turned off.
        let second = spawn_start(h.deps(loading));
        let slot = h.registry.existing("mic").unwrap();
        wait_until("the queued start", || slot.start_tickets.load(Ordering::SeqCst) == 2);
        assert!(h.stop("mic").await);
        gate.open();
        assert_eq!(first.await.unwrap().unwrap_err(), "cancelled");
        assert_eq!(second.await.unwrap().unwrap_err(), "cancelled", "the queued start never runs");
        thread::sleep(Duration::from_millis(50));
        let events = h.rec.all();
        let marker = events.iter().position(|e| matches!(e, Ev::Marker("stopped"))).unwrap();
        assert!(events[marker..].iter().all(|e| e.epoch().is_none()), "{:#?}", &events[marker..]);
        assert!(!h.script.attached.load(Ordering::SeqCst), "the device is never opened");
        assert!(h.registry.statuses().is_empty());
        // A start issued after the stop runs.
        let started = h
            .start(finals_config("fake-final"), h.deps(engines(vec![("fake-final", Behavior::default())])))
            .await;
        assert_eq!(h.registry.statuses()[0].epoch, started.epoch);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_stop_never_clears_the_pcm_of_a_session_started_meanwhile() {
        let h = Arc::new(Harness::new());
        let config = || PipelineConfig {
            pipeline_id: "loopback".into(),
            retain_pcm: true,
            capture_only: true,
            ..PipelineConfig::default()
        };
        h.start(config(), h.deps(engines(vec![]))).await;
        // The stop waits for a slow device release.
        let release = Latch::default();
        *h.script.close_gate.lock().unwrap() = Some(release.clone());
        let stop = {
            let h = Arc::clone(&h);
            tokio::spawn(async move { stop_session(&h.registry, "loopback").await })
        };
        wait_until("the release", || h.script.close_called.load(Ordering::SeqCst));
        *h.script.close_gate.lock().unwrap() = None;
        // Meanwhile a new session starts (a start does not wait for a stop) and keeps an utterance.
        let started = h.start(config(), h.deps(engines(vec![]))).await;
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("the new session's utterance", |e| count_closed(e) == 1);
        let (id, _, _, samples, _) = closed(&events)[0];
        release.open();
        assert!(stop.await.unwrap());
        assert_eq!(h.registry.statuses()[0].epoch, started.epoch);
        let pcm = h
            .registry
            .take_pcm("loopback", id)
            .expect("the old stop kept the new session's PCM");
        assert_eq!(pcm.len(), samples as usize);
        h.stop("loopback").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_epoch_stop_stops_only_its_session_and_never_a_queued_start() {
        let h = Arc::new(Harness::new());
        let factory = engines(vec![("fake-final", Behavior::default())]);
        // A stale epoch is refused: the newer session keeps running.
        let first = h.start(finals_config("fake-final"), h.deps(Arc::clone(&factory))).await;
        let second = h.start(finals_config("fake-final"), h.deps(Arc::clone(&factory))).await;
        assert!(!stop_session_at(&h.registry, "mic", Some(first.epoch)).await);
        assert_eq!(h.registry.statuses()[0].epoch, second.epoch);
        assert!(!stop_session_at(&h.registry, "none", Some(second.epoch)).await, "unknown key");
        // The current epoch stops.
        assert!(stop_session_at(&h.registry, "mic", Some(second.epoch)).await);
        assert!(h.registry.statuses().is_empty());

        // A start loading a model, and a start queued behind it: an epoch stop of the loading
        // session cancels only that one; the queued start keeps its ticket and runs.
        let gate = Latch::default();
        let loading: EngineFactory = {
            let gate = gate.clone();
            let inner = Arc::clone(&factory);
            Arc::new(move |spec: &EngineSpec, ctx: &EngineContext| {
                gate.wait(|| false);
                inner(spec, ctx)
            })
        };
        let spawn_start = |deps: SessionDeps| {
            let h = Arc::clone(&h);
            tokio::spawn(async move { start_session(&h.registry, finals_config("fake-final"), deps, false).await })
        };
        let loading_start = spawn_start(h.deps(loading));
        let is_loading = |s: &SpeechStatus| s.state == SpeechState::Loading && s.epoch > second.epoch;
        let events = h.rec.wait_for("loading", |e| statuses(e).iter().any(is_loading));
        let loading_epoch = statuses(&events).into_iter().find(is_loading).unwrap().epoch;
        let slot = h.registry.existing("mic").unwrap();
        let tickets = slot.start_tickets.load(Ordering::SeqCst);
        let queued = spawn_start(h.deps(Arc::clone(&factory)));
        wait_until("the queued start", || slot.start_tickets.load(Ordering::SeqCst) == tickets + 1);
        assert!(stop_session_at(&h.registry, "mic", Some(loading_epoch)).await);
        gate.open();
        assert_eq!(loading_start.await.unwrap().unwrap_err(), "cancelled");
        let started = queued.await.unwrap().expect("the queued start runs");
        assert!(started.epoch > loading_epoch);
        assert_eq!(h.registry.statuses()[0].epoch, started.epoch);
        h.stop("mic").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn decodes_get_their_deadline_and_give_ups_are_not_failures() {
        let h = Harness::new();
        let b = Behavior {
            give_up: true,
            ..Behavior::default()
        };
        let config = PipelineConfig {
            latency_budget_ms: 5_000,
            ..finals_config("fake-final")
        };
        h.start(config, h.deps(engines(vec![("fake-final", b.clone())]))).await;
        // More than the five failures in a row that would end the pipeline.
        for n in 1..=6 {
            h.script.feed(&utterance(640));
            h.rec.wait_for("the next give-up", |e| {
                utterances(e).iter().filter(|(k, _)| k == reasons::RATE_LIMITED).count() == n
            });
        }
        let events = h.rec.all();
        assert!(statuses(&events).iter().all(|st| st.state != SpeechState::Error));
        assert!(statuses(&events)
            .iter()
            .any(|st| st.code == codes::RATE_LIMITED && st.level == StatusLevel::Warning));
        assert_eq!(h.registry.statuses()[0].dropped, 6);
        // Each decode saw its utterance's deadline: its close plus the budget.
        let deadlines = b.deadlines.lock().unwrap().clone();
        assert_eq!(deadlines.len(), 6);
        for d in deadlines {
            let left = d.expect("a deadline");
            assert!(left <= Duration::from_secs(5) && left > Duration::from_secs(3), "{left:?}");
        }
        h.stop("mic").await;
    }

    /// Smart Turn verdicts in order, then 0.
    struct ScriptedTurn(Arc<Mutex<VecDeque<f32>>>);

    impl TurnDetector for ScriptedTurn {
        fn predict(&mut self, _audio_16k: &[f32]) -> Result<f32, String> {
            Ok(self.0.lock().unwrap().pop_front().unwrap_or(0.0))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_merged_final_carries_the_turn_verdict_of_its_last_segment() {
        let h = Harness::new();
        let gate = Latch::default();
        let b = Behavior {
            gate_first: Some(gate.clone()),
            ..Behavior::default()
        };
        let mut deps = h.deps(engines(vec![("fake-final", b.clone())]));
        // A pause mid-sentence, another one, then the end of the turn.
        let verdicts = Arc::new(Mutex::new(VecDeque::from(vec![0.2f32, 0.2, 0.9])));
        deps.turn = Box::new(move || Ok(Box::new(ScriptedTurn(verdicts)) as Box<dyn TurnDetector>));
        let config = PipelineConfig {
            smart_turn: true,
            ..finals_config("fake-final")
        };
        h.start(config, deps).await;
        h.script.feed(&utterance(640));
        h.rec.wait_for("the first closed utterance", |e| count_closed(e) == 1);
        wait_until("the first decode", || !b.calls.lock().unwrap().is_empty());
        h.script.feed(&utterance(640));
        h.script.feed(&utterance(640));
        let events = h.rec.wait_for("three closed utterances", |e| count_closed(e) == 3);
        let c = closed(&events);
        gate.open();
        let merged = ("merged".to_string(), c[2].0);
        let events = h
            .rec
            .wait_for("the merged final", |e| finals(e).len() == 2 && utterances(e).contains(&merged));
        let f = finals(&events).into_iter().find(|f| f.segment_id == c[1].0).unwrap();
        assert_eq!(f.turn_end, Some(true), "the batch ends where its last segment ended the turn");
        assert_eq!((f.start_ms, f.end_ms), (c[1].1, c[2].2));
        // The turn end is repeated for the carrying segment, before its final.
        let turn_at = events
            .iter()
            .position(|e| matches!(e, Ev::TurnEnd(t) if t.segment_id == c[1].0 && t.probability == 0.9))
            .expect("a turn end for the carrying segment");
        let final_at = events
            .iter()
            .position(|e| matches!(e, Ev::Final(x) if x.segment_id == c[1].0))
            .unwrap();
        assert!(turn_at < final_at);
        h.stop("mic").await;
    }

    #[test]
    fn speaking_ends_when_a_forced_split_leaves_only_silence() {
        let registry = Registry::new();
        let slot = registry.slot("mic").unwrap();
        let rec = Arc::new(Recorder::default());
        let vad = VadConfig {
            max_utterance_ms: 4_000,
            ..VadConfig::default()
        }
        .sanitized();
        let settings = resolve_settings(&PipelineConfig {
            vad: vad.clone(),
            ..PipelineConfig::default()
        });
        let ctx = Arc::new(SessionCtx::new(&slot, 0, Arc::new(RecSink(Arc::clone(&rec))), settings));
        let speaking = Arc::new(AtomicBool::new(false));
        let segmenter = Box::new(SegmenterEngine::new(LevelClassifier, &vad)) as Box<dyn SpeechSegmenter>;
        let mut front = FrontEnd::new(ctx, segmenter, None, Dispatch::CaptureOnly, None, Arc::clone(&speaking));
        let mut feed = |p: f32, ms: usize| {
            for chunk in vec![p; ms * MS].chunks(20 * MS) {
                front.process(chunk).unwrap();
            }
        };
        // Speech from 512 ms to 3904 ms: the segment (from 112 ms with its pre-roll) reaches the
        // 4 s maximum at 4112 ms, inside the silence, so the split's remainder is only silence.
        feed(0.0, 512);
        feed(0.9, 3_392);
        assert!(speaking.load(Ordering::SeqCst), "speaking once the segment is confirmed");
        feed(0.0, 2_048);
        let c = closed(&rec.all());
        assert_eq!(c.len(), 1, "{c:?}");
        assert!(c[0].4, "a forced split");
        assert!(!speaking.load(Ordering::SeqCst), "nobody speaks after it");
    }

    #[test]
    fn the_loopback_idle_fill_covers_the_minimum_silence() {
        assert_eq!(loopback_idle_fill(&VadConfig::default()), LOOPBACK_IDLE_FILL);
        let long = VadConfig {
            min_silence_ms: 5_000,
            ..VadConfig::default()
        };
        assert_eq!(loopback_idle_fill(&long.sanitized()), Duration::from_millis(5_300));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn loopback_silence_closes_an_utterance_with_a_long_minimum_silence() {
        let h = Harness::new();
        h.script.loopback.store(true, Ordering::SeqCst);
        let config = PipelineConfig {
            pipeline_id: "loopback".into(),
            capture_only: true,
            vad: VadConfig {
                min_silence_ms: 2_000,
                ..VadConfig::default()
            },
            ..PipelineConfig::default()
        };
        h.start(config, h.deps(engines(vec![]))).await;
        // Playback stops mid-utterance: WASAPI loopback then delivers nothing at all.
        h.script.feed(&[(0.0, 512), (0.9, 640)]);
        h.rec.wait_for("the utterance closed by the idle fill", |e| count_closed(e) == 1);
        h.stop("loopback").await;
    }

    /// Real Silero + real Whisper through the pipeline (everything but the device). Manual:
    /// `SIGIL_TEST_MODELS_DIR` must hold `ggml-tiny.en.bin`, `ggml-base.en.bin`, `ggml-base.bin` or
    /// `ggml-tiny.bin`; run with `cargo test real_model -- --ignored`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with a ggml Whisper model"]
    async fn real_model_whisper_transcribes_through_the_pipeline() {
        let Some(model) = std::env::var("SIGIL_TEST_MODELS_DIR").ok().and_then(|d| {
            ["ggml-tiny.en.bin", "ggml-base.en.bin", "ggml-base.bin", "ggml-tiny.bin"]
                .iter()
                .map(|f| std::path::PathBuf::from(&d).join(f))
                .find(|p| p.exists())
        }) else {
            return;
        };
        let h = Harness::new();
        let mut deps = h.deps(Arc::new(engines::create));
        deps.segmenter = Box::new(|vad: &VadConfig| {
            Ok(Box::new(Segmenter::new(vad.clone())?) as Box<dyn SpeechSegmenter>)
        });
        let config = PipelineConfig {
            pipeline_id: "mic".into(),
            language: "en".into(),
            final_engine: Some(EngineSpec::new(EngineKind::Whisper, model.to_string_lossy())),
            hotwords: vec!["weather".into()],
            ..PipelineConfig::default()
        };
        h.start(config, deps).await;
        let speech = crate::services::speech_core::silero::tests::speech_fixture();
        let mut audio = vec![0.0f32; 16_000];
        audio.extend_from_slice(&speech);
        audio.extend(std::iter::repeat_n(0.0f32, 32_000));
        {
            let mut items = h.script.items.lock().unwrap();
            for chunk in audio.chunks(320) {
                items.push_back(Item::Audio(chunk.to_vec()));
            }
        }
        let events = h.rec.wait_for("the transcript", |e| !finals(e).is_empty());
        let f = &finals(&events)[0];
        assert!(f.text.to_lowercase().contains("weather"), "{f:?}");
        assert_eq!((f.engine.as_str(), f.lang.as_deref(), f.pass), ("whisper", Some("en"), FinalPass::First));
        assert_eq!(closed(&events).len(), 1, "one utterance");
        // Only the utterance reached Whisper: 10 s of silence produce nothing.
        h.script.feed(&[(0.0, 10_000)]);
        thread::sleep(Duration::from_millis(500));
        assert_eq!(finals(&h.rec.all()).len(), 1);
        h.stop("mic").await;
    }

    #[test]
    fn pipeline_ids_and_settings_are_validated() {
        assert_eq!(validate_pipeline_id("", false).unwrap(), "mic");
        assert_eq!(validate_pipeline_id(" loopback ", false).unwrap(), "loopback");
        assert!(validate_pipeline_id("a b", false).is_err());
        assert!(validate_pipeline_id(&"x".repeat(33), false).is_err());
        assert!(validate_pipeline_id("legacy-whisper", false).is_err());
        assert_eq!(validate_pipeline_id("legacy-whisper", true).unwrap(), "legacy-whisper");
        let s = resolve_settings(&PipelineConfig {
            language: "ja-JP".into(),
            hotwords: vec![" Sigil ".into(), "sigil".into(), "".into(), "x".repeat(200)],
            prompt: format!("  a\0b{}", "語".repeat(600)),
            latency_budget_ms: 5,
            max_backlog: 99,
            ..PipelineConfig::default()
        });
        assert_eq!(s.language.as_deref(), Some("ja"));
        assert_eq!(s.hotwords.len(), 2);
        assert_eq!(s.hotwords[0], "Sigil");
        assert_eq!(s.hotwords[1].chars().count(), MAX_HOTWORD_CHARS);
        let p = s.prompt.unwrap();
        assert!(p.starts_with("ab") && p.len() <= MAX_PROMPT_BYTES);
        assert_eq!((s.latency_budget, s.max_backlog), (Duration::from_millis(1_000), 10));
        let d = resolve_settings(&PipelineConfig::default());
        assert_eq!(
            (d.language, d.latency_budget, d.prompt),
            (None, Duration::from_millis(u64::from(DEFAULT_LATENCY_BUDGET_MS)), None)
        );
        // A registry holds a bounded number of keys.
        let r = Registry::new();
        for i in 0..MAX_PIPELINES {
            r.slot(&format!("p{i}")).unwrap();
        }
        assert!(r.slot("one-too-many").is_err());
        assert!(r.slot("p0").is_ok());
    }

    #[test]
    fn explicit_plans_list_their_models() {
        let config = PipelineConfig {
            final_engine: Some(EngineSpec {
                variant: Some("cuda13".into()),
                ..EngineSpec::new(EngineKind::Transcribe, "stt/qwen3-asr-1.7b-q8_0")
            }),
            streaming: Some(EngineSpec::new(EngineKind::SherpaOnline, "stt/nemotron")),
            ..PipelineConfig::default()
        };
        let requirements: RequirementsFn = Arc::new(|s| vec![(s.model.clone(), s.model.contains("nemotron"))]);
        let plan = explicit_plan(&config, &resolve_settings(&config), &requirements);
        assert_eq!(plan.tier, Tier::Gpu);
        assert_eq!(plan.language, "auto");
        assert_eq!(plan.models, vec!["stt/nemotron".to_string(), "stt/qwen3-asr-1.7b-q8_0".to_string()]);
        assert_eq!(plan.missing, vec!["stt/qwen3-asr-1.7b-q8_0".to_string()]);
        assert_eq!(plan.notes, vec!["explicit_engines".to_string()]);
        assert_eq!(spec_label(&EngineSpec::new(EngineKind::Http, "")), "http");
        let abs = std::env::temp_dir().join("my-model.bin");
        assert_eq!(
            spec_label(&EngineSpec::new(EngineKind::Whisper, abs.to_string_lossy())),
            "whisper my-model.bin"
        );
        assert_eq!(
            spec_model_id(&EngineSpec::new(EngineKind::Whisper, "whisper/base")).as_deref(),
            Some("whisper/base")
        );
        assert_eq!(spec_model_id(&EngineSpec::new(EngineKind::Whisper, abs.to_string_lossy())), None);
    }

    #[test]
    fn not_installed_messages_name_their_catalog_id() {
        assert_eq!(catalog_id_in("stt/missing").as_deref(), Some("stt/missing"));
        assert_eq!(
            catalog_id_in("runtime/sherpa-onnx: the sherpa-onnx speech runtime is not installed").as_deref(),
            Some("runtime/sherpa-onnx")
        );
        assert_eq!(catalog_id_in("the Whisper model is not loaded"), None);
        assert_eq!(catalog_id_in("Anime-Whisper is bring-your-own: convert ..."), None);
        assert_eq!(catalog_id_in("C:/models/x.bin: missing"), None);
    }

    #[test]
    fn returned_plans_never_echo_secrets() {
        let spec = EngineSpec {
            options: serde_json::json!({"preset": "groq", "apiKey": "gsk_secret", "accessToken": "t", "model": "m"}),
            ..EngineSpec::new(EngineKind::Http, "")
        };
        let plan = redacted(RoutePlan {
            final_engine: Some(spec),
            ..RoutePlan::default()
        });
        let json = serde_json::to_string(&plan).unwrap();
        assert!(!json.contains("gsk_secret") && !json.contains("accessToken"), "{json}");
        assert!(json.contains("\"preset\":\"groq\"") && json.contains("\"model\":\"m\""));
    }
}
