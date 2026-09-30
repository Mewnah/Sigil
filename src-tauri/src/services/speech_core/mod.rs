//! Speech core: the shared 16 kHz mono speech front end (Silero VAD v6.2 segmenter, streaming
//! rubato resampling, hallucination filter) used by Whisper and by VAD-closed utterance capture.
//!
//! Owned by P0-C (P2-A since P2: the speech pipeline drives it); scaffolded by P0-S0. Submodules:
//! - [`silero`]: Silero VAD v6.2 (embedded ONNX, MIT) on Sigil's ORT, CPU, 1 thread, stateful.
//! - [`segmenter`]: the segmentation state machine behind [`Segmenter`] (hysteresis, pre-roll,
//!   min speech/silence, forced split at the lowest-probability frame).
//! - [`resample`]: streaming band-limited rubato conversion of device audio to 16 kHz mono.
//! - [`hallucination`]: Whisper post-filter (markers, repeat loops, whole-utterance phrases).
//! - [`utterance`]: [`utterance::SpeechFrontEnd`] (resample + segment), the drop-oldest
//!   [`utterance::BoundedQueue`] and the take-once [`utterance::UtteranceStore`].
//! - [`SpeechSegmenter`]: the segmenter interface the P2 speech pipeline (`speech::pipeline`)
//!   drives, so its tests can run on a synthetic classifier.
//!
//! # Frozen contract (P0-S0; change only through an INTEGRATE step)
//!
//! Rust API:
//! - [`VadConfig`] (serde, snake_case, every field optional with the defaults below) and
//!   [`VadConfig::sanitized`].
//! - [`Segmenter::new`], [`Segmenter::push`], [`Segmenter::flush`], [`Segmenter::reset`],
//!   [`Segmenter::config`]; events are [`SegmenterEvent`] / [`Segment`]. Input is 16 kHz mono f32;
//!   nothing is emitted for silence; a segment is forced closed at `max_utterance_ms`.
//! - [`UtteranceEvent`]: payload of the Tauri event [`UTTERANCE_EVENT`] (`"audio:utterance"`),
//!   serialized with `kind` as the tag and snake_case fields. It carries metadata only (plan rule:
//!   no PCM over JSON IPC).
//!
//! IPC (audio plugin, implemented by P0-C, listed in `plugin-commands/audio.txt`):
//! - `plugin:audio|start_utterance_capture { deviceName?: string | null, vad?: VadConfig }` → `number`
//!   (the capture id). Replaces any running utterance capture; independent of `start_audio_capture`.
//! - `plugin:audio|stop_utterance_capture` → `void`. After it resolves, no event of that capture is emitted.
//! - `plugin:audio|take_utterance { captureId: number, seq: number }` → binary `ArrayBuffer`
//!   (`tauri::ipc::Response`): raw little-endian f32 mono PCM at [`SAMPLE_RATE`]. Each utterance can
//!   be taken once; untaken utterances are dropped oldest-first past a small backlog (reported as
//!   `dropped`). TS: `new Float32Array(await invoke<ArrayBuffer>(...))`.
//!
//! TS mirror types: `RustVadConfig` / `RustUtteranceEvent` in `src/types.ts`.

pub mod hallucination;
pub mod resample;
pub mod segmenter;
pub mod silero;
pub mod utterance;

use serde::{Deserialize, Serialize};

/// Sample rate of everything that leaves the speech front end.
pub const SAMPLE_RATE: u32 = 16_000;
/// Silero VAD v6 frame at 16 kHz (32 ms). The model also needs a 64-sample context per frame.
pub const VAD_FRAME_SAMPLES: usize = 512;
/// Tauri event carrying [`UtteranceEvent`] payloads.
pub const UTTERANCE_EVENT: &str = "audio:utterance";

/// Voice-activity segmentation settings. Defaults follow the P0 plan (hysteresis 0.5 / 0.35,
/// 400 ms pre-roll, 500 ms end silence, forced flush at 18 s).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VadConfig {
    /// Speech starts when the frame probability reaches this (0..1).
    pub threshold: f32,
    /// Speech ends after `min_silence_ms` below this (0..1, below `threshold`).
    pub neg_threshold: f32,
    /// Audio kept before the detected start.
    pub pre_roll_ms: u32,
    /// Silence that closes a segment.
    pub min_silence_ms: u32,
    /// Segments with less speech than this are discarded (never emitted); the remainder of a forced
    /// split is exempt when it holds any frame at or above `threshold` (it continues confirmed speech).
    pub min_speech_ms: u32,
    /// A segment is force-closed at this length (split at the lowest-probability point).
    pub max_utterance_ms: u32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: 0.35,
            pre_roll_ms: 400,
            min_silence_ms: 500,
            min_speech_ms: 250,
            max_utterance_ms: 18_000,
        }
    }
}

impl VadConfig {
    /// Clamp every field to a usable range (`neg_threshold` stays below `threshold`).
    pub fn sanitized(&self) -> Self {
        let threshold = if self.threshold.is_finite() { self.threshold.clamp(0.05, 0.95) } else { 0.5 };
        let neg = if self.neg_threshold.is_finite() { self.neg_threshold } else { 0.35 };
        Self {
            threshold,
            neg_threshold: neg.clamp(0.01, threshold - 0.01),
            pre_roll_ms: self.pre_roll_ms.min(2_000),
            min_silence_ms: self.min_silence_ms.clamp(100, 5_000),
            min_speech_ms: self.min_speech_ms.min(5_000),
            max_utterance_ms: self.max_utterance_ms.clamp(2_000, 60_000),
        }
    }
}

/// One closed segment of 16 kHz mono speech (pre-roll included).
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// Monotonic per [`Segmenter`] (starts at 0, bumped per `SpeechStart`).
    pub seq: u64,
    /// Index of the first sample in the segmenter's input stream.
    pub start_sample: u64,
    pub samples: Vec<f32>,
    /// Closed by `max_utterance_ms` or [`Segmenter::flush`] rather than by end silence.
    pub forced: bool,
}

/// Output of [`Segmenter::push`] / [`Segmenter::flush`].
#[derive(Clone, Debug, PartialEq)]
pub enum SegmenterEvent {
    /// Speech onset (before the segment is complete); `seq` matches the later [`Segment`].
    SpeechStart { seq: u64, start_sample: u64 },
    /// A closed segment.
    Segment(Segment),
}

/// Streaming Silero VAD segmenter (CPU, 1 thread, spinning off, stateful frames). Semantics are
/// documented in [`segmenter`]; every `SpeechStart` is followed by exactly one `Segment` with the
/// same `seq` (unless the segmenter is reset or dropped first).
pub struct Segmenter {
    engine: segmenter::SegmenterEngine<silero::SileroVad>,
}

impl Segmenter {
    /// Build a segmenter (loads the bundled Silero model on Sigil's ORT; tens of milliseconds, so
    /// call it off the async runtime and off audio threads).
    pub fn new(config: VadConfig) -> Result<Self, String> {
        let vad = silero::SileroVad::new()?;
        Ok(Self {
            engine: segmenter::SegmenterEngine::new(vad, &config),
        })
    }

    /// Settings in use (sanitized).
    #[allow(dead_code)] // Frozen P0-S0 API; used by the P2 speech engines.
    pub fn config(&self) -> &VadConfig {
        self.engine.config()
    }

    /// Feed 16 kHz mono samples (any length); appends zero or more events to `out`.
    pub fn push(&mut self, samples: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        self.engine.push(samples, out)
    }

    /// Close any open segment (as `forced`) and emit it; used at the end of a stream.
    pub fn flush(&mut self, out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        self.engine.flush(out);
        Ok(())
    }

    /// Drop all state (model state, pre-roll, open segment); `seq` keeps counting.
    pub fn reset(&mut self) {
        self.engine.reset();
    }
}

/// The segmentation interface the speech pipeline (`speech::pipeline`) drives: [`Segmenter`] in
/// production, a [`segmenter::SegmenterEngine`] over a synthetic classifier in tests. Same
/// semantics as the inherent methods of [`Segmenter`].
pub trait SpeechSegmenter {
    /// Feed 16 kHz mono samples (any length); appends zero or more events to `out`.
    fn push(&mut self, samples: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String>;
    /// Close any open segment (as `forced`) if it holds enough speech, else discard it.
    fn flush(&mut self, out: &mut Vec<SegmenterEvent>) -> Result<(), String>;
    /// Drop the open segment, pre-roll and model state (`seq` and the stream position keep counting).
    fn reset(&mut self);
}

impl SpeechSegmenter for Segmenter {
    fn push(&mut self, samples: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        Segmenter::push(self, samples, out)
    }
    fn flush(&mut self, out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        Segmenter::flush(self, out)
    }
    fn reset(&mut self) {
        Segmenter::reset(self);
    }
}

impl<C: segmenter::FrameClassifier> SpeechSegmenter for segmenter::SegmenterEngine<C> {
    fn push(&mut self, samples: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        segmenter::SegmenterEngine::push(self, samples, out)
    }
    fn flush(&mut self, out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        segmenter::SegmenterEngine::flush(self, out);
        Ok(())
    }
    fn reset(&mut self) {
        segmenter::SegmenterEngine::reset(self);
    }
}

/// Payload of [`UTTERANCE_EVENT`]. Times are milliseconds since the capture started.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UtteranceEvent {
    /// Speech onset: show an interim indicator.
    SpeechStart { capture_id: u64, seq: u64, start_ms: u64 },
    /// A closed utterance is ready: fetch its PCM once with `take_utterance`.
    Utterance {
        capture_id: u64,
        seq: u64,
        start_ms: u64,
        end_ms: u64,
        sample_rate: u32,
        /// Number of samples `take_utterance` returns.
        samples: u32,
        forced: bool,
    },
    /// An utterance was dropped before it was taken (backlog cap, or too short).
    Dropped { capture_id: u64, seq: u64, reason: String },
    /// Capture failed (device lost, model error); the capture is stopped.
    Error { capture_id: u64, message: String },
}

#[cfg(test)]
mod tests {
    use super::{UtteranceEvent, VadConfig};

    #[test]
    fn vad_config_defaults_and_partial_json() {
        let c: VadConfig = serde_json::from_str(r#"{"threshold":0.6}"#).unwrap();
        assert_eq!(c.threshold, 0.6);
        assert_eq!(c.neg_threshold, 0.35);
        assert_eq!(c.max_utterance_ms, 18_000);
        let s = VadConfig { threshold: 0.3, neg_threshold: 0.9, ..VadConfig::default() }.sanitized();
        assert!(s.neg_threshold < s.threshold);
    }

    #[test]
    fn utterance_event_wire_format_is_frozen() {
        let e = UtteranceEvent::Utterance {
            capture_id: 3,
            seq: 7,
            start_ms: 100,
            end_ms: 2100,
            sample_rate: 16_000,
            samples: 32_000,
            forced: false,
        };
        assert_eq!(
            serde_json::to_string(&e).unwrap(),
            r#"{"kind":"utterance","capture_id":3,"seq":7,"start_ms":100,"end_ms":2100,"sample_rate":16000,"samples":32000,"forced":false}"#
        );
        let s = UtteranceEvent::SpeechStart { capture_id: 3, seq: 8, start_ms: 2500 };
        assert_eq!(serde_json::to_string(&s).unwrap(), r#"{"kind":"speech_start","capture_id":3,"seq":8,"start_ms":2500}"#);
    }
}
