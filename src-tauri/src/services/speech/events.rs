//! Tauri events of the `speech` plugin and their payloads (frozen by P2-S0).
//!
//! # Frozen contract (P2-S0; change only through an INTEGRATE step)
//! Payloads may gain fields (`#[serde(default)]` on the TS side: optional), never lose or rename
//! them. Wire names are camelCase. TS mirrors: `RustSpeech*` and `SPEECH_EVENTS` in `src/types.ts`.
//!
//! Every payload carries `pipelineId` and `epoch`. Epoch discipline (plan global rules): the
//! pipeline bumps its epoch under its stop lock on every start and stop, re-checks it before each
//! emit and emits while it is still current, so after `stop_pipeline` resolves no event of the
//! stopped session is delivered. The TS driver also drops payloads whose `(pipelineId, epoch)` is
//! not the session it started.
//!
//! Segments: `segmentId` is a per-pipeline counter (it keeps counting across restarts of one
//! pipeline key in a process, so a `(pipelineId, segmentId)` pair is never reused). Within a segment
//! `revision` starts at 0 and increases by one with every partial and final, so the latest revision
//! always wins: partials `0..n`, the first final `n+1`, a two-pass final `n+2`. A final never
//! follows a later final of the same segment. The TS driver maps `(pipelineId, epoch, segmentId)`
//! to `TextEvent.segmentId` (a string) and copies `revision` / `lang`.
//!
//! | Event | Payload | When |
//! |---|---|---|
//! | `speech:partial` | [`SpeechPartial`] | streaming hypothesis changed |
//! | `speech:final` | [`SpeechFinal`] | first-pass or two-pass final (`pass`) |
//! | `speech:status` | [`SpeechStatus`] | lifecycle, model load / download progress, warnings, errors |
//! | `speech:utterance` | [`SpeechUtterance`] | VAD: speech start, utterance closed, utterance dropped |
//! | `speech:turn_end` | [`SpeechTurnEnd`] | Smart Turn: the speaker probably finished (commit hint only; never shortens finals) |
//!
//! `speech:turn_end` is emitted at most once per segment, after its `speech:utterance` kind
//! `utterance` and independently of its finals (it may arrive before or after them); finals emitted
//! after the verdict also carry it as [`SpeechFinal::turn_end`].

// Emitted by the pipeline (P2-A) and two-pass / Smart Turn (P2-C) once they land.
#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Runtime};

use super::engine_api::StatusLevel;

pub const PARTIAL_EVENT: &str = "speech:partial";
pub const FINAL_EVENT: &str = "speech:final";
pub const STATUS_EVENT: &str = "speech:status";
pub const UTTERANCE_EVENT: &str = "speech:utterance";
pub const TURN_END_EVENT: &str = "speech:turn_end";

/// A streaming partial hypothesis of one segment.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechPartial {
    pub pipeline_id: String,
    pub epoch: u64,
    pub segment_id: u64,
    pub revision: u32,
    pub text: String,
    pub lang: Option<String>,
}

/// Which decode produced a final.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FinalPass {
    /// The streaming engine's end-of-segment result, or the only final of a finals-only route.
    #[default]
    First,
    /// The two-pass re-decode that replaces the first final.
    Second,
}

/// A final transcript of one segment (may be replaced by a later revision: two-pass).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechFinal {
    pub pipeline_id: String,
    pub epoch: u64,
    pub segment_id: u64,
    pub revision: u32,
    pub text: String,
    pub lang: Option<String>,
    /// `<kind>:<value>` tags (emotion / event / lang); see `engine_api`.
    pub tags: Vec<String>,
    pub pass: FinalPass,
    /// `EngineKind` wire name of the engine that produced it (`sherpa-offline`, ...).
    pub engine: String,
    /// Model id / file name that produced it.
    pub model: String,
    /// Segment bounds in ms since the pipeline session started.
    pub start_ms: u64,
    pub end_ms: u64,
    /// Closed by the max-length split or a flush rather than by end silence.
    pub forced: bool,
    /// Milliseconds from the end of speech (VAD close) to this emit.
    pub latency_ms: u32,
    /// Smart Turn verdict for this segment when it was known before this final was emitted (the
    /// pipeline runs Smart Turn before the first final when it is on); `None` = not evaluated. The
    /// TS driver copies it to `TextEvent.turnEnd`.
    pub turn_end: Option<bool>,
}

/// Pipeline lifecycle state (wire: snake_case).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpeechState {
    /// `start_pipeline` accepted; capture / engines are being prepared.
    #[default]
    Starting,
    /// Loading or downloading a model / runtime (see `progress`).
    Loading,
    /// Engines loaded, capture running.
    Listening,
    /// Stopped by the user (last event of an epoch).
    Stopped,
    /// Fatal error; the pipeline stopped itself (last event of an epoch).
    Error,
}

/// Status of a pipeline: lifecycle, progress, warnings. UI only: never published as text.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechStatus {
    pub pipeline_id: String,
    pub epoch: u64,
    pub state: SpeechState,
    /// Machine code ([`codes`]); the TS driver localizes known codes and shows `detail` otherwise.
    pub code: String,
    /// English detail (no secrets).
    pub detail: String,
    /// 0-100 for a determinate progress bar.
    pub progress: Option<f32>,
    pub level: StatusLevel,
    /// Catalog id the status is about, if any.
    pub model_id: Option<String>,
    /// Engine kind / device the status is about (`sherpa-offline`, `cuda`, ...), if any.
    pub engine: Option<String>,
    pub device: Option<String>,
}

/// Machine codes of [`SpeechStatus::code`] (and `EngineStatus::code`). New codes may be added.
pub mod codes {
    pub const STARTING: &str = "starting";
    pub const DOWNLOADING: &str = "downloading";
    pub const LOADING_MODEL: &str = "loading_model";
    pub const LOADING_RUNTIME: &str = "loading_runtime";
    pub const LISTENING: &str = "listening";
    pub const STOPPED: &str = "stopped";
    /// A model / runtime pack is not installed (`modelId` names it).
    pub const NOT_INSTALLED: &str = "not_installed";
    /// An engine fell back (GPU → CPU, variant) (`detail` says why).
    pub const ENGINE_FALLBACK: &str = "engine_fallback";
    /// Utterances were dropped because the engine could not keep up (latency budget / backlog).
    pub const BACKLOG_DROPPED: &str = "backlog_dropped";
    /// The capture stream had a gap or restarted (device lost / recovered).
    pub const CAPTURE_GAP: &str = "capture_gap";
    /// A device error ended the capture.
    pub const CAPTURE_ERROR: &str = "capture_error";
    /// HTTP engine rate limited; waiting.
    pub const RATE_LIMITED: &str = "rate_limited";
    /// An engine failed (fatal when `state` is `error`).
    pub const ENGINE_ERROR: &str = "engine_error";
    /// A sherpa streaming model that needs the spoken language got none (or one it does not know)
    /// and runs on its automatic language detection, which is less accurate (warning, once).
    pub const SHERPA_LANGUAGE_AUTO: &str = "sherpa_language_auto";
    /// Hotwords could not be enabled for a sherpa model; recognition continues without them.
    pub const SHERPA_HOTWORDS_UNAVAILABLE: &str = "sherpa_hotwords_unavailable";
}

/// VAD lifecycle of a segment (wire: `kind`-tagged, snake_case kinds, camelCase fields).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SpeechUtterance {
    /// Speech onset: a segment opened.
    #[serde(rename_all = "camelCase")]
    SpeechStart {
        pipeline_id: String,
        epoch: u64,
        segment_id: u64,
        start_ms: u64,
    },
    /// The segment closed (end silence, max length or flush); finals follow.
    #[serde(rename_all = "camelCase")]
    Utterance {
        pipeline_id: String,
        epoch: u64,
        segment_id: u64,
        start_ms: u64,
        end_ms: u64,
        /// 16 kHz mono samples (`take_pipeline_utterance` returns them when `retainPcm`).
        samples: u32,
        forced: bool,
    },
    /// The segment was dropped before decoding (latency budget, backlog cap, too short, mute).
    #[serde(rename_all = "camelCase")]
    Dropped {
        pipeline_id: String,
        epoch: u64,
        segment_id: u64,
        reason: String,
    },
}

/// Smart Turn end-of-turn hint for a closed segment (commit translation / chatbox / TTS sooner).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeechTurnEnd {
    pub pipeline_id: String,
    pub epoch: u64,
    /// The segment whose end was classified.
    pub segment_id: u64,
    /// Turn-end probability 0..1 (emitted only at or above the threshold).
    pub probability: f32,
    pub lang: Option<String>,
}

/// Emit helpers (log and ignore emit failures: the window may be closing).
pub fn emit_partial<R: Runtime>(app: &AppHandle<R>, payload: &SpeechPartial) {
    emit(app, PARTIAL_EVENT, payload);
}
pub fn emit_final<R: Runtime>(app: &AppHandle<R>, payload: &SpeechFinal) {
    emit(app, FINAL_EVENT, payload);
}
pub fn emit_status<R: Runtime>(app: &AppHandle<R>, payload: &SpeechStatus) {
    emit(app, STATUS_EVENT, payload);
}
pub fn emit_utterance<R: Runtime>(app: &AppHandle<R>, payload: &SpeechUtterance) {
    emit(app, UTTERANCE_EVENT, payload);
}
pub fn emit_turn_end<R: Runtime>(app: &AppHandle<R>, payload: &SpeechTurnEnd) {
    emit(app, TURN_END_EVENT, payload);
}

fn emit<R: Runtime, T: Serialize + Clone>(app: &AppHandle<R>, event: &str, payload: &T) {
    if let Err(e) = app.emit(event, payload.clone()) {
        tracing::debug!(target: "sigil::speech", "emit {event} failed: {e}");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_wire_format_is_frozen() {
        let f = SpeechFinal {
            pipeline_id: "mic".into(),
            epoch: 2,
            segment_id: 7,
            revision: 3,
            text: "hi".into(),
            lang: Some("en".into()),
            tags: vec!["emotion:happy".into()],
            pass: FinalPass::Second,
            engine: "transcribe".into(),
            model: "stt/qwen3-asr-1.7b-q8_0".into(),
            start_ms: 100,
            end_ms: 900,
            forced: false,
            latency_ms: 350,
            turn_end: Some(true),
        };
        assert_eq!(
            serde_json::to_string(&f).unwrap(),
            r#"{"pipelineId":"mic","epoch":2,"segmentId":7,"revision":3,"text":"hi","lang":"en","tags":["emotion:happy"],"pass":"second","engine":"transcribe","model":"stt/qwen3-asr-1.7b-q8_0","startMs":100,"endMs":900,"forced":false,"latencyMs":350,"turnEnd":true}"#
        );
    }

    #[test]
    fn utterance_and_status_wire_formats_are_frozen() {
        let u = SpeechUtterance::SpeechStart {
            pipeline_id: "mic".into(),
            epoch: 1,
            segment_id: 4,
            start_ms: 20,
        };
        assert_eq!(
            serde_json::to_string(&u).unwrap(),
            r#"{"kind":"speech_start","pipelineId":"mic","epoch":1,"segmentId":4,"startMs":20}"#
        );
        let d = SpeechUtterance::Dropped {
            pipeline_id: "mic".into(),
            epoch: 1,
            segment_id: 5,
            reason: "latency_budget".into(),
        };
        assert_eq!(
            serde_json::to_string(&d).unwrap(),
            r#"{"kind":"dropped","pipelineId":"mic","epoch":1,"segmentId":5,"reason":"latency_budget"}"#
        );
        let s = SpeechStatus {
            pipeline_id: "mic".into(),
            epoch: 1,
            state: SpeechState::Loading,
            code: codes::DOWNLOADING.into(),
            progress: Some(12.5),
            ..SpeechStatus::default()
        };
        let v = serde_json::to_value(&s).unwrap();
        assert_eq!(v["state"], "loading");
        assert_eq!(v["level"], "info");
        assert_eq!(v["code"], "downloading");
        assert!(v.get("modelId").is_some());
    }

    #[test]
    fn turn_end_and_partial_wire_formats_are_frozen() {
        let t = SpeechTurnEnd {
            pipeline_id: "mic".into(),
            epoch: 1,
            segment_id: 9,
            probability: 0.75,
            lang: None,
        };
        assert_eq!(
            serde_json::to_string(&t).unwrap(),
            r#"{"pipelineId":"mic","epoch":1,"segmentId":9,"probability":0.75,"lang":null}"#
        );
        let p = SpeechPartial {
            pipeline_id: "loopback".into(),
            epoch: 3,
            segment_id: 1,
            revision: 0,
            text: "he".into(),
            lang: Some("en".into()),
        };
        assert_eq!(
            serde_json::to_string(&p).unwrap(),
            r#"{"pipelineId":"loopback","epoch":3,"segmentId":1,"revision":0,"text":"he","lang":"en"}"#
        );
    }
}
