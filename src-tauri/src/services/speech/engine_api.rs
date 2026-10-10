//! The frozen ASR engine contract of the `speech` plugin (P2-S0).
//!
//! Every recognizer the pipeline drives (Whisper, the HTTP utterance engine, sherpa-onnx online /
//! offline, transcribe.cpp) implements [`AsrEngine`] and is created by
//! `speech::engines::create(&EngineSpec, &EngineContext)`. The router
//! (`speech::router`) turns a [`RouteRequest`] into a [`RoutePlan`] of [`EngineSpec`]s; the
//! pipeline (`speech::pipeline`) builds a [`PipelineConfig`] into running engines.
//!
//! # Frozen contract (P2-S0; change only through an INTEGRATE step)
//! Structs and enums here may gain fields or variants (new fields are `#[serde(default)]`), but
//! nothing is renamed or removed, and wire names never change.
//!
//! - Audio is always 16 kHz mono f32 in `[-1, 1]` ([`SAMPLE_RATE`]); the pipeline resamples.
//! - Everything in [`AsrEngine`] blocks: call it on a worker thread / `spawn_blocking`, never on an
//!   async runtime thread or an audio callback. Engines are `Send` (moved into their worker), not
//!   necessarily `Sync`; `&mut self` serializes calls on one engine.
//! - One stream at a time per engine: `begin_stream` → `push_stream`* → `end_stream` (or
//!   `abort_stream`). A second `begin_stream` without an end aborts the first. `transcribe` while a
//!   stream is open may fail with [`AsrErrorKind::Busy`] (transcribe.cpp) — the pipeline never does
//!   that; it uses a separate engine instance for finals.
//! - Engines never download. A model or runtime pack that is not installed is
//!   [`AsrErrorKind::NotInstalled`] (message names the catalog id); the TS driver installs the
//!   route's `missing` ids with the user's consent first (`<ModelRequirement>`). Models are located
//!   only with `model_store::installed_path(id)`.
//! - [`Canceller::cancel`] may be called from any thread and makes the running `transcribe` /
//!   `push_stream` / `end_stream` return [`AsrErrorKind::Cancelled`] soon (engines without a native
//!   cancel return their result, which the pipeline then drops by epoch).
//! - [`EngineSpec::options`] may carry secrets (the HTTP engine's `apiKey`): never log, emit or
//!   echo `options`, and keep keys in HTTP headers only.
//! - Engines report load / download progress only through [`EngineContext::status`]; they never
//!   emit Tauri events themselves (the pipeline adds `pipelineId` / `epoch` and emits
//!   `speech:status`).
//! - Language codes are lower-case BCP-47 base subtags as normalized by `speech::lang::normalize`
//!   (`en`, `ja`, `zh`, `yue`, `ko`, `ru`, ...); `None` / `"auto"` means "detect".
//! - Tags ([`AsrResult::tags`]) are lower-case `<kind>:<value>` strings: `emotion:<happy|sad|angry|
//!   neutral|fearful|disgusted|surprised>`, `event:<bgm|applause|laughter|cry|sneeze|breath|cough>`
//!   (SenseVoice), `lang:<code>` when an engine detected the language itself. Consumers ignore
//!   unknown tags.

// Implementers (P2-A..D) consume this API during P2; until then parts are unused.
#![allow(dead_code)]

use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::services::speech_core::VadConfig;

/// Sample rate of all engine input.
pub const SAMPLE_RATE: u32 = crate::services::speech_core::SAMPLE_RATE;

/// Pipeline key of the microphone pipeline (Service_STT).
pub const PIPELINE_MIC: &str = "mic";
/// Pipeline key of the other-players (loopback) pipeline (P6-A).
pub const PIPELINE_LOOPBACK: &str = "loopback";

// ---------------------------------------------------------------------------------------------
// Engine selection
// ---------------------------------------------------------------------------------------------

/// Which engine implementation runs a model. Wire: kebab-case.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum EngineKind {
    /// whisper-rs (Whisper, Anime-Whisper BYO): utterance finals. `speech::engines::whisper` (P2-A).
    Whisper,
    /// OpenAI-compatible `/v1/audio/transcriptions` (custom server, Groq) and Moonshine HTTP:
    /// utterance finals. `speech::engines::http` (P2-A).
    Http,
    /// sherpa-onnx `OnlineRecognizer` (Nemotron 3.5 streaming, streaming transducers): partials
    /// and stream finals. `speech::engines::sherpa_online` (P2-B).
    SherpaOnline,
    /// sherpa-onnx `OfflineRecognizer` (Parakeet Ultra, Qwen3-ASR-0.6B, SenseVoice, Moonshine v2):
    /// utterance finals. `speech::engines::sherpa_offline` (P2-B).
    SherpaOffline,
    /// transcribe.cpp `sigil_transcribe.dll` (Qwen3-ASR-1.7B, Cohere, Granite, Voxtral realtime,
    /// R2T2): utterance finals, streaming for realtime models. `speech::engines::transcribe` (P2-D).
    Transcribe,
}

impl EngineKind {
    /// Every kind, in dispatch order.
    pub const ALL: [EngineKind; 5] = [
        EngineKind::Whisper,
        EngineKind::Http,
        EngineKind::SherpaOnline,
        EngineKind::SherpaOffline,
        EngineKind::Transcribe,
    ];

    /// The wire / log name (`whisper`, `http`, `sherpa-online`, `sherpa-offline`, `transcribe`).
    pub fn as_str(self) -> &'static str {
        match self {
            EngineKind::Whisper => "whisper",
            EngineKind::Http => "http",
            EngineKind::SherpaOnline => "sherpa-online",
            EngineKind::SherpaOffline => "sherpa-offline",
            EngineKind::Transcribe => "transcribe",
        }
    }
}

impl fmt::Display for EngineKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One engine to create: which implementation, which model, and engine-specific options.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineSpec {
    pub engine: EngineKind,
    /// Model-store catalog id (`stt/...`, `whisper/<id>`), or an absolute path for a
    /// bring-your-own model (validated by the engine; sherpa folders go through
    /// `speech_rt::sherpa::probe::validate_in_child`). Empty for engines without a model (HTTP).
    #[serde(default)]
    pub model: String,
    /// Inference threads; `None` = the engine's default (sherpa streaming 1, others 2, max 4).
    #[serde(default)]
    pub threads: Option<u32>,
    /// Runtime variant to force (`cpu`, `cpu-avx2`, `cpu-avx512`, `vulkan`, `cuda13`); `None` =
    /// best installed variant allowed by the resource governor.
    #[serde(default)]
    pub variant: Option<String>,
    /// Engine-specific options (HTTP: `baseUrl`, `model`, `apiKey`, `preset`, ...; Whisper: the
    /// hardening settings). May contain secrets: never log or emit.
    #[serde(default)]
    pub options: serde_json::Value,
}

impl EngineSpec {
    /// A spec with default threads / variant and no options.
    pub fn new(engine: EngineKind, model: impl Into<String>) -> Self {
        Self {
            engine,
            model: model.into(),
            threads: None,
            variant: None,
            options: serde_json::Value::Null,
        }
    }
}

/// Hardware tier the router may use. Wire: lower-case.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tier {
    /// Pick from the system probe and the resource governor ("Best for my PC").
    #[default]
    Auto,
    /// CPU engines only.
    Cpu,
    /// GPU engines where available (falls back to CPU with a note).
    Gpu,
}

/// What the user asked the router for (`preview_route`, and `PipelineConfig::route`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RouteRequest {
    /// `"auto"` (spoken language ID) or a language code; empty = the pipeline's `language`.
    pub language: String,
    pub tier: Tier,
    /// Re-decode each closed utterance with the final model (revision + 1 replaces the partial).
    pub two_pass: bool,
    /// Live partial captions wanted (default true). `false`: where the plan would run a streaming
    /// engine next to a final engine, the streaming engine is not planned at all (never loaded or
    /// run), so the final engine transcribes each closed utterance alone. A streaming engine that
    /// is the plan's only engine stays (its finals are the captions).
    pub interim: bool,
}

impl Default for RouteRequest {
    fn default() -> Self {
        Self {
            language: "auto".into(),
            tier: Tier::Auto,
            two_pass: true,
            interim: true,
        }
    }
}

/// The router's answer: which engines to run for a request on this PC.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct RoutePlan {
    /// Resolved language (`"auto"` when spoken language ID decides per utterance).
    pub language: String,
    /// Resolved tier (never `Auto`).
    pub tier: Tier,
    /// Streaming engine for partials (and first-pass finals); `None` = finals only.
    pub streaming: Option<EngineSpec>,
    /// Utterance engine. Without `streaming` it produces the finals; with `streaming` (two-pass)
    /// its result replaces the streaming final under the same `segmentId` with `revision + 1`.
    #[serde(rename = "final")]
    pub final_engine: Option<EngineSpec>,
    /// CPU engine promoted when a GPU `final_engine` fails (two-pass fallback).
    pub fallback: Option<EngineSpec>,
    /// Spoken language ID engine for `"auto"` (sherpa LID or an engine that detects itself).
    pub language_id: Option<EngineSpec>,
    /// Every catalog id the plan needs (models and runtime packs), for `<ModelRequirement>`.
    pub models: Vec<String>,
    /// The subset of `models` that is not installed yet.
    pub missing: Vec<String>,
    /// Estimated VRAM of the GPU engines (MB; 0 on CPU).
    pub vram_mb: u64,
    /// Machine-readable reasons for the choice (`gpu_denied_vc_running`, `cohere_not_for_ja`,
    /// `nemotron_needs_language`, ...); the UI maps known codes to text.
    pub notes: Vec<String>,
}

/// Everything `start_pipeline` needs. Wire: camelCase, every field optional.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct PipelineConfig {
    /// Pipeline key: [`PIPELINE_MIC`] or [`PIPELINE_LOOPBACK`]. One running pipeline per key;
    /// starting a running key restarts it (new epoch).
    pub pipeline_id: String,
    /// Capture source for the hub (`audio::hub::SourceSpec::parse`): `""` / `"default"` = default
    /// microphone, a device name, or `loopback:default` / `loopback:<output name>`.
    pub source: String,
    /// Silero segmentation (shared with Whisper / HTTP utterance capture).
    pub vad: VadConfig,
    /// `"auto"` or a language code.
    pub language: String,
    /// Router input, used when neither `streaming` nor `final_engine` is given.
    pub route: RouteRequest,
    /// Explicit streaming engine (bypasses the router).
    pub streaming: Option<EngineSpec>,
    /// Explicit utterance engine (bypasses the router), e.g. the legacy Whisper / HTTP backends.
    #[serde(rename = "final")]
    pub final_engine: Option<EngineSpec>,
    /// Run Smart Turn at each pause and emit `speech:turn_end` (commit hints only).
    pub smart_turn: bool,
    /// Names and terms to bias recognition towards (engines without biasing ignore them).
    pub hotwords: Vec<String>,
    /// Decoder prompt / context (engines without prompts ignore it).
    pub prompt: String,
    /// Whole-utterance hallucination filter (`speech_core::hallucination`) on finals.
    pub hallucination_filter: bool,
    /// A closed utterance still waiting for its first decode after this long is dropped
    /// (`speech:utterance` kind `dropped`); 0 = the pipeline default.
    pub latency_budget_ms: u32,
    /// Closed utterances that may wait for the engine before the oldest is dropped (1-10).
    pub max_backlog: u32,
    /// Keep each closed utterance's PCM for `take_pipeline_utterance` (TS-side engines).
    pub retain_pcm: bool,
    /// Voice detection only: never start an engine (no router, no explicit engines); closed
    /// utterances are reported and, with `retain_pcm`, kept for `take_pipeline_utterance`.
    /// Explicit, so a capture-only pipeline stays engine-free whatever the router would plan.
    pub capture_only: bool,
    /// P6-L A/B (default off): drop closed segments the FireRedVAD music / singing gate
    /// (`speech::music_gate`) hears as music or singing, before any engine sees them
    /// (`speech:utterance` kind `dropped`, reason `music`). Off when the model is not installed.
    pub music_gate: bool,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            pipeline_id: PIPELINE_MIC.into(),
            source: String::new(),
            vad: VadConfig::default(),
            language: "auto".into(),
            route: RouteRequest::default(),
            streaming: None,
            final_engine: None,
            smart_turn: false,
            hotwords: Vec::new(),
            prompt: String::new(),
            hallucination_filter: true,
            latency_budget_ms: 0,
            max_backlog: 3,
            retain_pcm: false,
            capture_only: false,
            music_gate: false,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Engine interface
// ---------------------------------------------------------------------------------------------

/// Per-utterance / per-stream decoding options.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct DecodeOptions {
    /// Normalized language code; `None` = the engine detects (or its model's only language).
    pub language: Option<String>,
    /// Hotwords / context terms (transducer biasing, Qwen3-ASR context, Whisper prompt).
    pub hotwords: Vec<String>,
    /// Free-form decoder prompt.
    pub prompt: Option<String>,
}

/// A closed-utterance (or end-of-stream) result.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct AsrResult {
    /// Recognized text, trimmed; empty = nothing recognized (the pipeline emits no final).
    pub text: String,
    /// Language the engine used or detected.
    pub lang: Option<String>,
    /// `<kind>:<value>` metadata tags (see the module docs).
    pub tags: Vec<String>,
    /// Engine's no-speech probability, when it has one (Whisper).
    pub no_speech_prob: Option<f32>,
    /// Engine confidence 0..1, when it has one.
    pub confidence: Option<f32>,
    /// Wall-clock decode time.
    pub decode_ms: u32,
}

/// A streaming partial hypothesis.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct Hypothesis {
    pub text: String,
    pub lang: Option<String>,
}

/// What an engine can do.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct EngineCaps {
    /// `begin_stream` / `push_stream` / `end_stream` produce partials.
    pub streaming: bool,
    /// `transcribe` decodes closed utterances.
    pub utterance: bool,
    /// Languages the loaded model handles (normalized codes); empty = any / unknown.
    pub languages: Vec<String>,
    /// Must be given an explicit language (Nemotron 3.5 streaming).
    pub needs_language: bool,
    /// Detects the spoken language itself (Whisper auto, SenseVoice, Qwen3-ASR).
    pub detects_language: bool,
    /// Uses [`DecodeOptions::hotwords`].
    pub hotwords: bool,
    /// Uses [`DecodeOptions::prompt`].
    pub prompt: bool,
    /// Emits [`AsrResult::tags`].
    pub tags: bool,
}

/// What an engine instance is and where it runs (for status, logs and the UI).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EngineInfo {
    pub kind: EngineKind,
    /// Catalog id or file name of the loaded model (never a secret).
    pub model: String,
    /// Runtime and version, e.g. `sherpa-onnx 1.13.8`, `transcribe.cpp 0.2.4 cuda13`, `whisper-rs`.
    pub runtime: String,
    /// Device actually used: `cpu`, `cuda`, `vulkan`, `directml` or `remote` (HTTP).
    pub device: String,
    pub caps: EngineCaps,
    /// VRAM the engine holds (MB; 0 on CPU / remote).
    #[serde(default)]
    pub vram_mb: u64,
    /// Why the engine runs on a lesser device / variant than requested.
    #[serde(default)]
    pub fallback_reason: Option<String>,
}

/// Error category (wire: snake_case; the TS driver maps it to a localized message).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AsrErrorKind {
    /// A model or runtime pack is missing (message names the catalog id).
    NotInstalled,
    /// Bad audio, options or model files.
    InvalidInput,
    /// The engine / model cannot do this (e.g. streaming on an offline model).
    Unsupported,
    /// Cancelled through a [`Canceller`] or by stopping the pipeline.
    Cancelled,
    /// The engine is busy with another stream / decode.
    Busy,
    /// Not enough (V)RAM, or the resource governor refused the claim.
    OutOfMemory,
    /// HTTP engine: connection / server failure.
    Network,
    /// HTTP engine: rate limited (Groq); the message may carry the retry delay.
    RateLimited,
    /// Anything else from the runtime.
    Runtime,
}

impl AsrErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            AsrErrorKind::NotInstalled => "not_installed",
            AsrErrorKind::InvalidInput => "invalid_input",
            AsrErrorKind::Unsupported => "unsupported",
            AsrErrorKind::Cancelled => "cancelled",
            AsrErrorKind::Busy => "busy",
            AsrErrorKind::OutOfMemory => "out_of_memory",
            AsrErrorKind::Network => "network",
            AsrErrorKind::RateLimited => "rate_limited",
            AsrErrorKind::Runtime => "runtime",
        }
    }
}

/// An engine error: a category plus an English diagnostic message (logged; the UI shows a
/// localized text for the category and the message as detail).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AsrError {
    pub kind: AsrErrorKind,
    pub message: String,
}

impl AsrError {
    pub fn new(kind: AsrErrorKind, message: impl Into<String>) -> Self {
        Self {
            kind,
            message: message.into(),
        }
    }
    pub fn not_installed(message: impl Into<String>) -> Self {
        Self::new(AsrErrorKind::NotInstalled, message)
    }
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::new(AsrErrorKind::InvalidInput, message)
    }
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::new(AsrErrorKind::Unsupported, message)
    }
    pub fn cancelled() -> Self {
        Self::new(AsrErrorKind::Cancelled, "cancelled")
    }
    pub fn runtime(message: impl Into<String>) -> Self {
        Self::new(AsrErrorKind::Runtime, message)
    }
    /// A stub's answer until its workstream lands.
    pub fn not_implemented(what: &str, workstream: &str) -> Self {
        Self::new(AsrErrorKind::Unsupported, format!("{what} is not implemented yet ({workstream})"))
    }
    pub fn is_cancelled(&self) -> bool {
        self.kind == AsrErrorKind::Cancelled
    }
}

impl fmt::Display for AsrError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.kind.as_str(), self.message)
    }
}

impl std::error::Error for AsrError {}

impl From<AsrError> for String {
    fn from(e: AsrError) -> Self {
        e.to_string()
    }
}

/// Interrupts a running decode from another thread. Cheap to clone.
#[derive(Clone)]
pub struct Canceller(Arc<dyn Fn() + Send + Sync>);

impl Canceller {
    pub fn new(f: impl Fn() + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
    /// A canceller that sets `flag` (for engines that poll an [`AtomicBool`]).
    pub fn from_flag(flag: Arc<AtomicBool>) -> Self {
        Self::new(move || flag.store(true, Ordering::SeqCst))
    }
    pub fn cancel(&self) {
        (self.0)()
    }
}

impl fmt::Debug for Canceller {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("Canceller")
    }
}

/// Severity of a status line (styling only).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StatusLevel {
    #[default]
    Info,
    Warning,
    Error,
}

/// A load / download / fallback report from an engine (forwarded as `speech:status`).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", default)]
pub struct EngineStatus {
    /// Machine code (see `events::codes`); the TS driver localizes known codes.
    pub code: String,
    /// English detail for logs and as a fallback text (no secrets).
    pub detail: String,
    /// 0-100 for a determinate progress bar.
    pub progress: Option<f32>,
    pub level: StatusLevel,
    /// Catalog id the status is about, if any.
    pub model_id: Option<String>,
}

/// Where an engine reports [`EngineStatus`]. Cheap to clone; callable from any thread.
#[derive(Clone)]
pub struct StatusSink(Arc<dyn Fn(EngineStatus) + Send + Sync>);

impl StatusSink {
    pub fn new(f: impl Fn(EngineStatus) + Send + Sync + 'static) -> Self {
        Self(Arc::new(f))
    }
    /// A sink that drops every report (tests, headless probes).
    pub fn discard() -> Self {
        Self::new(|_| {})
    }
    pub fn report(&self, status: EngineStatus) {
        (self.0)(status)
    }
}

impl fmt::Debug for StatusSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("StatusSink")
    }
}

/// Context handed to `engines::create`.
#[derive(Clone, Debug)]
pub struct EngineContext {
    pub status: StatusSink,
}

impl EngineContext {
    /// A context whose status reports are dropped (tests, headless probes).
    pub fn headless() -> Self {
        Self {
            status: StatusSink::discard(),
        }
    }
}

/// A speech recognizer. See the module docs for the threading and streaming rules.
pub trait AsrEngine: Send {
    /// What this instance is and where it runs.
    fn info(&self) -> &EngineInfo;

    /// Decode one closed utterance (16 kHz mono).
    fn transcribe(&mut self, audio: &[f32], options: &DecodeOptions) -> Result<AsrResult, AsrError>;

    /// Open the (single) stream of this engine. Engines without `caps.streaming` keep the default.
    fn begin_stream(&mut self, options: &DecodeOptions) -> Result<(), AsrError> {
        let _ = options;
        Err(AsrError::unsupported(format!("{} does not stream", self.info().kind)))
    }

    /// Feed audio (any length) to the open stream; `Some` when the partial hypothesis changed.
    fn push_stream(&mut self, audio: &[f32]) -> Result<Option<Hypothesis>, AsrError> {
        let _ = audio;
        Err(AsrError::unsupported(format!("{} does not stream", self.info().kind)))
    }

    /// Close the stream (input finished) and return its final result.
    fn end_stream(&mut self) -> Result<AsrResult, AsrError> {
        Err(AsrError::unsupported(format!("{} does not stream", self.info().kind)))
    }

    /// Drop the open stream without a result (no-op when none is open).
    fn abort_stream(&mut self) {}

    /// A handle that interrupts a running call from another thread, if the engine supports it.
    fn canceller(&self) -> Option<Canceller> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_kind_wire_names_are_frozen() {
        for kind in EngineKind::ALL {
            let json = serde_json::to_string(&kind).unwrap();
            assert_eq!(json, format!("\"{}\"", kind.as_str()));
            let back: EngineKind = serde_json::from_str(&json).unwrap();
            assert_eq!(back, kind);
        }
    }

    #[test]
    fn pipeline_config_accepts_partial_json_and_final_key() {
        let c: PipelineConfig = serde_json::from_str(
            r#"{"pipelineId":"mic","language":"ja","final":{"engine":"whisper","model":"whisper/base-q5_1"},"vad":{"threshold":0.6}}"#,
        )
        .unwrap();
        assert_eq!(c.language, "ja");
        assert_eq!(c.vad.threshold, 0.6);
        assert_eq!(c.max_backlog, 3);
        let f = c.final_engine.unwrap();
        assert_eq!(f.engine, EngineKind::Whisper);
        assert_eq!(f.model, "whisper/base-q5_1");
        assert!(f.options.is_null());
        assert!(c.streaming.is_none());
        assert_eq!(c.route, RouteRequest::default());
    }

    #[test]
    fn route_request_interim_defaults_to_on_and_reads_camel_case() {
        // A request from before the field existed (or from a caller that does not know it) keeps live partials.
        let old: RouteRequest = serde_json::from_str(r#"{"language":"en","tier":"cpu","twoPass":false}"#).unwrap();
        assert!(old.interim);
        assert!(!old.two_pass);
        assert!(serde_json::from_str::<RouteRequest>("{}").unwrap().interim);
        assert!(RouteRequest::default().interim);
        let off: RouteRequest = serde_json::from_str(r#"{"interim":false}"#).unwrap();
        assert!(!off.interim);
        assert!(off.two_pass, "the other fields keep their defaults");
        // Inside a pipeline config too.
        let c: PipelineConfig = serde_json::from_str(r#"{"route":{"language":"ja","interim":false}}"#).unwrap();
        assert!(!c.route.interim);
        assert_eq!(serde_json::to_value(&off).unwrap()["interim"], false);
    }

    #[test]
    fn route_plan_wire_format_is_frozen() {
        let plan = RoutePlan {
            language: "en".into(),
            tier: Tier::Cpu,
            streaming: Some(EngineSpec::new(EngineKind::SherpaOnline, "stt/nemotron")),
            final_engine: None,
            ..RoutePlan::default()
        };
        let v = serde_json::to_value(&plan).unwrap();
        assert_eq!(v["tier"], "cpu");
        assert_eq!(v["streaming"]["engine"], "sherpa-online");
        assert!(v.get("final").is_some());
        assert!(v.get("languageId").is_some());
        assert!(v.get("vramMb").is_some());
    }

    #[test]
    fn errors_render_kind_and_message() {
        let e = AsrError::not_installed("stt/x");
        assert_eq!(e.to_string(), "not_installed: stt/x");
        assert_eq!(serde_json::to_value(&e).unwrap()["kind"], "not_installed");
        assert!(AsrError::cancelled().is_cancelled());
    }

    #[test]
    fn canceller_from_flag_sets_it() {
        let flag = Arc::new(AtomicBool::new(false));
        let c = Canceller::from_flag(flag.clone());
        c.clone().cancel();
        assert!(flag.load(Ordering::SeqCst));
    }
}
