//! Streaming VAD segmentation state machine (Silero semantics, hysteresis, pre-roll, forced split).
//!
//! Frame probabilities come from a [`FrameClassifier`] (Silero in production, a synthetic source in
//! tests). Rules, all in 16 kHz samples:
//! - **Start**: a frame with `p >= threshold` opens a segment. It starts `pre_roll` samples earlier
//!   (never before the end of the previous segment, never before the stream start).
//! - **Hysteresis**: once open, frames with `neg_threshold <= p < threshold` keep the segment open;
//!   a frame with `p < neg_threshold` starts a silence run, and any `p >= threshold` frame ends it.
//! - **End**: a silence run of `min_silence` closes the segment at the silence start plus a short
//!   tail pad ([`END_PAD_MS`]); the rest of the silence becomes pre-roll history again.
//! - **Confirmation**: [`SegmenterEvent::SpeechStart`] is emitted only once the segment holds
//!   `min_speech` of speech (counted from the trigger frame to the current silence start). A segment
//!   that never gets there is discarded silently (except a forced-split remainder, below); one
//!   emitted before confirmation gets its `SpeechStart` right before its `Segment`. So every
//!   `SpeechStart` is matched by exactly one later `Segment` with the same `seq` (unless the caller
//!   resets or drops the segmenter).
//! - **Forced split**: at `max_utterance` samples (pre-roll included) the segment is split at the
//!   centre of the lowest-probability frame in its second half and emitted with `forced: true`; the
//!   remainder continues as the next segment. The remainder is the rest of confirmed speech, so
//!   `min_speech` does not apply to it: it is emitted whenever it holds a frame at or above
//!   `threshold` (a last word just after the split is not lost), and discarded only when it is
//!   pure silence / sub-threshold noise.
//! - Pure silence emits nothing at all.

use std::collections::VecDeque;

use super::{Segment, SegmenterEvent, VadConfig, SAMPLE_RATE, VAD_FRAME_SAMPLES};

/// Speech kept after the detected silence start, so trailing consonants are not clipped.
pub const END_PAD_MS: u32 = 200;

/// Source of per-frame speech probabilities.
pub trait FrameClassifier {
    /// Speech probability (0..1) of one [`VAD_FRAME_SAMPLES`]-sample frame, in stream order.
    fn frame_probability(&mut self, frame: &[f32]) -> Result<f32, String>;
    /// Forget all stream state.
    #[allow(dead_code)] // Behind the frozen `Segmenter::reset`.
    fn reset(&mut self);
}

impl FrameClassifier for super::silero::SileroVad {
    fn frame_probability(&mut self, frame: &[f32]) -> Result<f32, String> {
        self.predict(frame)
    }

    fn reset(&mut self) {
        super::silero::SileroVad::reset(self);
    }
}

fn ms_to_samples(ms: u32) -> usize {
    (u64::from(ms) * u64::from(SAMPLE_RATE) / 1000) as usize
}

/// A segment being collected.
struct OpenSegment {
    /// Stream index of `samples[0]` (pre-roll included).
    start_sample: u64,
    samples: Vec<f32>,
    /// Stream index where speech was detected (trigger frame start, or the split point).
    speech_start: u64,
    /// `(frame start, probability)` of every classified frame since `speech_start`.
    probs: Vec<(u64, f32)>,
    /// Stream index where the current silence run began (`p < neg_threshold`).
    silence_start: Option<u64>,
    /// `seq` once `SpeechStart` has been emitted.
    seq: Option<u64>,
    /// The remainder of a forced split: continues confirmed speech, so it is emitted when it holds
    /// any frame at or above `threshold`, however short.
    continuation: bool,
}

impl OpenSegment {
    fn end_sample(&self) -> u64 {
        self.start_sample + self.samples.len() as u64
    }

    /// Speech held so far: trigger to silence start (or to the end when speaking).
    fn speech_len(&self) -> u64 {
        self.silence_start.unwrap_or_else(|| self.end_sample()).saturating_sub(self.speech_start)
    }
}

/// The segmentation engine over any [`FrameClassifier`].
pub struct SegmenterEngine<C: FrameClassifier> {
    classifier: C,
    #[allow(dead_code)] // Behind the frozen `Segmenter::config`.
    config: VadConfig,
    threshold: f32,
    neg_threshold: f32,
    pre_roll: usize,
    min_silence: u64,
    min_speech: u64,
    max_len: usize,
    end_pad: u64,
    /// Samples of the next frame not classified yet (< one frame).
    pending: Vec<f32>,
    /// Stream index of the next sample `push` receives.
    next_sample: u64,
    /// Most recent unsegmented samples (pre-roll source), ending at `next_sample - pending.len()`.
    history: VecDeque<f32>,
    open: Option<OpenSegment>,
    next_seq: u64,
    /// End of the last emitted segment; pre-roll never reaches back before it.
    last_end: u64,
}

impl<C: FrameClassifier> SegmenterEngine<C> {
    pub fn new(classifier: C, config: &VadConfig) -> Self {
        let config = config.sanitized();
        let max_len = ms_to_samples(config.max_utterance_ms);
        Self {
            classifier,
            threshold: config.threshold,
            neg_threshold: config.neg_threshold,
            // At most a quarter of the maximum length, so a forced split always has speech to split.
            pre_roll: ms_to_samples(config.pre_roll_ms).min(max_len / 4),
            min_silence: ms_to_samples(config.min_silence_ms) as u64,
            // At most a quarter of the maximum length, so the first part of a forced split is always
            // confirmed speech (never silently discarded).
            min_speech: ms_to_samples(config.min_speech_ms).min(max_len / 4) as u64,
            max_len,
            end_pad: ms_to_samples(END_PAD_MS.min(config.min_silence_ms)) as u64,
            config,
            pending: Vec::with_capacity(VAD_FRAME_SAMPLES),
            next_sample: 0,
            history: VecDeque::new(),
            open: None,
            next_seq: 0,
            last_end: 0,
        }
    }

    #[allow(dead_code)] // Behind the frozen `Segmenter::config`.
    pub fn config(&self) -> &VadConfig {
        &self.config
    }

    /// Whether a segment is being collected (speech detected, not closed yet).
    #[cfg(test)]
    pub fn in_speech(&self) -> bool {
        self.open.is_some()
    }

    /// Stream index of the next sample (= samples consumed so far).
    #[cfg(test)]
    pub fn position(&self) -> u64 {
        self.next_sample
    }

    pub fn push(&mut self, mut samples: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        while !samples.is_empty() {
            let take = (VAD_FRAME_SAMPLES - self.pending.len()).min(samples.len());
            self.pending.extend_from_slice(&samples[..take]);
            self.next_sample += take as u64;
            samples = &samples[take..];
            if self.pending.len() == VAD_FRAME_SAMPLES {
                let frame = std::mem::take(&mut self.pending);
                let result = self.process_frame(&frame, out);
                self.pending = frame;
                self.pending.clear();
                result?;
            }
        }
        Ok(())
    }

    /// Close any open segment as `forced` (including the unclassified tail) if it holds enough
    /// speech; otherwise discard it. The stream position keeps counting.
    pub fn flush(&mut self, out: &mut Vec<SegmenterEvent>) {
        let tail = std::mem::take(&mut self.pending);
        if let Some(mut open) = self.open.take() {
            open.samples.extend_from_slice(&tail);
            let end = open.end_sample();
            self.finish(open, end, true, out);
        }
        self.history.clear();
        self.pending = tail;
        self.pending.clear();
    }

    /// Drop the open segment, history and model state; `seq` and the stream position keep counting.
    #[allow(dead_code)] // Behind the frozen `Segmenter::reset`.
    pub fn reset(&mut self) {
        self.classifier.reset();
        self.open = None;
        self.history.clear();
        self.pending.clear();
        self.last_end = self.next_sample;
    }

    fn process_frame(&mut self, frame: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        let frame_start = self.next_sample - VAD_FRAME_SAMPLES as u64;
        let p = self.classifier.frame_probability(frame)?;
        let p = if p.is_finite() { p } else { 0.0 };

        let Some(open) = self.open.as_mut() else {
            if p >= self.threshold {
                self.open_segment(frame_start, frame, p);
                self.maybe_confirm(out);
            } else {
                self.history.extend(frame.iter().copied());
                let excess = self.history.len().saturating_sub(self.pre_roll);
                self.history.drain(..excess);
            }
            return Ok(());
        };

        open.samples.extend_from_slice(frame);
        open.probs.push((frame_start, p));
        if p >= self.threshold {
            open.silence_start = None;
        } else if p < self.neg_threshold && open.silence_start.is_none() {
            open.silence_start = Some(frame_start);
        }

        let frame_end = frame_start + VAD_FRAME_SAMPLES as u64;
        if let Some(silence) = open.silence_start {
            if frame_end - silence >= self.min_silence {
                let open = self.open.take().expect("open segment");
                let end = (silence + self.end_pad).min(frame_end);
                self.finish(open, end, false, out);
                return Ok(());
            }
        }
        self.maybe_confirm(out);
        if self.open.as_ref().is_some_and(|o| o.samples.len() >= self.max_len) {
            self.force_split(out);
        }
        Ok(())
    }

    /// Open a segment at `frame_start` with up to `pre_roll` samples of history before it.
    fn open_segment(&mut self, frame_start: u64, frame: &[f32], p: f32) {
        let history_start = frame_start - self.history.len() as u64;
        let start = frame_start
            .saturating_sub(self.pre_roll as u64)
            .max(self.last_end)
            .max(history_start);
        let skip = (start - history_start) as usize;
        let mut samples = Vec::with_capacity(self.max_len + VAD_FRAME_SAMPLES);
        samples.extend(self.history.iter().skip(skip).copied());
        samples.extend_from_slice(frame);
        self.history.clear();
        self.open = Some(OpenSegment {
            start_sample: start,
            samples,
            speech_start: frame_start,
            probs: vec![(frame_start, p)],
            silence_start: None,
            seq: None,
            continuation: false,
        });
    }

    /// Emit `SpeechStart` once the open segment holds `min_speech` of speech.
    fn maybe_confirm(&mut self, out: &mut Vec<SegmenterEvent>) {
        let Some(open) = self.open.as_mut() else { return };
        if open.seq.is_none() && open.silence_start.is_none() && open.speech_len() >= self.min_speech.max(1) {
            let seq = self.next_seq;
            self.next_seq += 1;
            open.seq = Some(seq);
            out.push(SegmenterEvent::SpeechStart {
                seq,
                start_sample: open.start_sample,
            });
        }
    }

    /// Close `open` at stream index `end` (clamped to its samples). Emits it when it holds enough
    /// speech (confirming it first if needed), or, for the remainder of a forced split, any frame
    /// at or above `threshold`; otherwise discards it. The samples after `end` (and a discarded
    /// segment's tail) become pre-roll history.
    fn finish(&mut self, mut open: OpenSegment, end: u64, forced: bool, out: &mut Vec<SegmenterEvent>) {
        let end = end.clamp(open.start_sample, open.end_sample());
        let keep = (end - open.start_sample) as usize;
        let speech = open.silence_start.unwrap_or(end).min(end).saturating_sub(open.speech_start);
        let voiced_tail = open.continuation
            && open.probs.iter().any(|&(start, p)| start < end && p >= self.threshold);
        let emit = open.seq.is_some() || speech >= self.min_speech.max(1) || voiced_tail;
        if emit {
            let tail = open.samples.split_off(keep);
            self.refill_history(&tail);
            let seq = match open.seq {
                Some(seq) => seq,
                None => {
                    let seq = self.next_seq;
                    self.next_seq += 1;
                    out.push(SegmenterEvent::SpeechStart {
                        seq,
                        start_sample: open.start_sample,
                    });
                    seq
                }
            };
            self.last_end = end;
            out.push(SegmenterEvent::Segment(Segment {
                seq,
                start_sample: open.start_sample,
                samples: open.samples,
                forced,
            }));
        } else {
            // Too short: nothing was announced, nothing is emitted; its audio stays available as
            // pre-roll for speech that follows right away.
            self.refill_history(&open.samples);
        }
    }

    fn refill_history(&mut self, samples: &[f32]) {
        self.history.clear();
        let from = samples.len().saturating_sub(self.pre_roll);
        self.history.extend(samples[from..].iter().copied());
    }

    /// Split the open (over-long) segment at the lowest-probability frame of its second half.
    fn force_split(&mut self, out: &mut Vec<SegmenterEvent>) {
        let Some(mut open) = self.open.take() else { return };
        let half = open.start_sample + (self.max_len / 2) as u64;
        let end = open.end_sample();
        let half_frame = (VAD_FRAME_SAMPLES / 2) as u64;
        // Lowest probability, latest on ties (keeps the first part as long as possible).
        let split = open
            .probs
            .iter()
            .filter(|(start, _)| start + half_frame >= half && start + half_frame < end)
            .fold(None::<(u64, f32)>, |best, &(start, p)| match best {
                Some((_, bp)) if bp < p => best,
                _ => Some((start, p)),
            })
            .map(|(start, _)| start + half_frame)
            .unwrap_or(end);

        let remainder_samples = open.samples.split_off((split - open.start_sample) as usize);
        let remainder_probs: Vec<(u64, f32)> = open.probs.iter().copied().filter(|(s, _)| *s >= split).collect();
        let remainder_silence = open.silence_start.map(|s| s.max(split));

        // First part: always emitted (it holds at least max_len/2 - pre_roll of speech).
        let first_end = split;
        let mut first = open;
        first.silence_start = first.silence_start.filter(|&s| s < first_end);
        self.finish(first, first_end, true, out);
        // `finish` refilled the history from an empty tail; the remainder continues the speech.
        self.history.clear();

        if remainder_samples.is_empty() {
            return;
        }
        self.open = Some(OpenSegment {
            start_sample: split,
            samples: remainder_samples,
            speech_start: split,
            probs: remainder_probs,
            silence_start: remainder_silence,
            seq: None,
            continuation: true,
        });
        self.maybe_confirm(out);
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// Synthetic classifier: a frame's probability is its first sample (tests build the signal as
    /// constant runs of the wanted probability).
    pub(crate) struct LevelClassifier;

    impl FrameClassifier for LevelClassifier {
        fn frame_probability(&mut self, frame: &[f32]) -> Result<f32, String> {
            Ok(frame[0])
        }
        fn reset(&mut self) {}
    }

    const MS: usize = 16; // samples per ms at 16 kHz

    /// `(probability, milliseconds)` runs → samples (frame-aligned when every run is a multiple of 32 ms).
    fn signal(runs: &[(f32, usize)]) -> Vec<f32> {
        runs.iter().flat_map(|&(p, ms)| std::iter::repeat_n(p, ms * MS)).collect()
    }

    fn run(config: &VadConfig, audio: &[f32], chunk: usize) -> Vec<SegmenterEvent> {
        let mut seg = SegmenterEngine::new(LevelClassifier, config);
        let mut out = Vec::new();
        for c in audio.chunks(chunk) {
            seg.push(c, &mut out).unwrap();
        }
        out
    }

    fn segments(events: &[SegmenterEvent]) -> Vec<&Segment> {
        events
            .iter()
            .filter_map(|e| match e {
                SegmenterEvent::Segment(s) => Some(s),
                _ => None,
            })
            .collect()
    }

    #[test]
    fn ten_seconds_of_silence_emit_nothing() {
        let cfg = VadConfig::default();
        assert!(run(&cfg, &signal(&[(0.0, 10_000)]), 480).is_empty());
        // Sub-threshold noise (hysteresis band never reaches `threshold`) emits nothing either.
        assert!(run(&cfg, &signal(&[(0.45, 10_000)]), 441).is_empty());
    }

    #[test]
    fn pre_roll_is_kept_and_segment_closes_after_min_silence() {
        let cfg = VadConfig::default(); // 400 ms pre-roll, 500 ms silence, 250 ms min speech
        let audio = signal(&[(0.0, 2048), (0.9, 1024), (0.0, 1536)]);
        let events = run(&cfg, &audio, 333);
        assert_eq!(events.len(), 2, "{events:?}");
        let SegmenterEvent::SpeechStart { seq: 0, start_sample } = events[0] else { panic!("{events:?}") };
        let speech_at = (2048 * MS) as u64;
        assert_eq!(start_sample, speech_at - 400 * MS as u64);
        let seg = segments(&events)[0];
        assert_eq!(seg.seq, 0);
        assert!(!seg.forced);
        assert_eq!(seg.start_sample, start_sample);
        // Pre-roll: the 400 ms before the trigger are the (silent) history samples.
        assert!(seg.samples[..400 * MS].iter().all(|&s| s == 0.0));
        assert!(seg.samples[400 * MS..400 * MS + 1024 * MS].iter().all(|&s| s == 0.9));
        // Ends END_PAD_MS after the silence start.
        assert_eq!(seg.samples.len(), (400 + 1024 + END_PAD_MS as usize) * MS);
    }

    #[test]
    fn hysteresis_band_keeps_the_segment_open() {
        let cfg = VadConfig::default();
        // 0.4 is between neg_threshold (0.35) and threshold (0.5): speech continues through it.
        let audio = signal(&[(0.9, 512), (0.4, 1024), (0.9, 512), (0.0, 1024)]);
        let segs = run(&cfg, &audio, 512);
        let segs = segments(&segs);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].samples.len(), (512 + 1024 + 512 + END_PAD_MS as usize) * MS);
        // A short dip below neg_threshold (< min_silence) does not split either.
        let audio = signal(&[(0.9, 512), (0.1, 320), (0.9, 512), (0.0, 1024)]);
        assert_eq!(segments(&run(&cfg, &audio, 512)).len(), 1);
    }

    #[test]
    fn too_short_blips_are_discarded_without_speech_start() {
        let cfg = VadConfig::default();
        // 128 ms of "speech" < 250 ms min_speech.
        let audio = signal(&[(0.0, 1024), (0.9, 128), (0.0, 2048)]);
        assert!(run(&cfg, &audio, 256).is_empty());
        // A later real utterance still gets seq 0 and a full, contiguous pre-roll.
        let audio = signal(&[(0.0, 1024), (0.9, 128), (0.0, 1024), (0.9, 640), (0.0, 1024)]);
        let events = run(&cfg, &audio, 256);
        let segs = segments(&events);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].seq, 0);
        assert_eq!(segs[0].start_sample, ((1024 + 128 + 1024 - 400) * MS) as u64);
        assert!(segs[0].samples[..400 * MS].iter().all(|&s| s == 0.0));
        // A blip followed by speech within min_silence is the same utterance (Silero semantics).
        let audio = signal(&[(0.0, 1024), (0.9, 128), (0.0, 160), (0.9, 640), (0.0, 1024)]);
        let events = run(&cfg, &audio, 256);
        let segs = segments(&events);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].start_sample, ((1024 - 400) * MS) as u64);
    }

    #[test]
    fn forced_split_at_eighteen_seconds_cuts_at_the_lowest_probability_frame() {
        let cfg = VadConfig::default(); // max 18 s
        // 20 s of speech with a single weak frame (0.2) at 12.0 s and a weaker one (0.05) at 3 s
        // (outside the second half, so it must not be chosen).
        let mut audio = signal(&[(0.9, 22_000)]);
        let frame_at = |ms: usize| ms * MS / VAD_FRAME_SAMPLES * VAD_FRAME_SAMPLES;
        let weak = frame_at(12_000);
        audio[weak..weak + VAD_FRAME_SAMPLES].fill(0.2);
        let early = frame_at(3_000);
        audio[early..early + VAD_FRAME_SAMPLES].fill(0.05);
        audio.extend(signal(&[(0.0, 1024)]));
        let events = run(&cfg, &audio, 1000);
        let segs = segments(&events);
        assert_eq!(segs.len(), 2, "{events:?}");
        assert!(segs[0].forced);
        assert_eq!(segs[0].start_sample, 0);
        let split = (weak + VAD_FRAME_SAMPLES / 2) as u64;
        assert_eq!(segs[0].samples.len() as u64, split);
        // The remainder continues seamlessly from the split point and closes on silence.
        assert_eq!(segs[1].start_sample, split);
        assert!(!segs[1].forced);
        assert_eq!(segs[1].seq, 1);
        // Order: SpeechStart(0), Segment(0), SpeechStart(1), Segment(1).
        let kinds: Vec<(bool, u64)> = events
            .iter()
            .map(|e| match e {
                SegmenterEvent::SpeechStart { seq, .. } => (true, *seq),
                SegmenterEvent::Segment(s) => (false, s.seq),
            })
            .collect();
        assert_eq!(kinds, vec![(true, 0), (false, 0), (true, 1), (false, 1)]);
        // Nothing is lost or duplicated at the seam.
        let total: usize = segs.iter().map(|s| s.samples.len()).sum();
        assert_eq!(total as u64, segs[1].start_sample + segs[1].samples.len() as u64);
    }

    fn event_kinds(events: &[SegmenterEvent]) -> Vec<(bool, u64)> {
        events
            .iter()
            .map(|e| match e {
                SegmenterEvent::SpeechStart { seq, .. } => (true, *seq),
                SegmenterEvent::Segment(s) => (false, s.seq),
            })
            .collect()
    }

    #[test]
    fn short_voiced_tail_after_a_forced_split_is_not_lost() {
        let cfg = VadConfig::default(); // max 18 s, 250 ms min speech
        // Speech 1.024 s .. 18.656 s with one weak (hysteresis-band) frame at 18.4 s, the lowest in
        // the second half: the split lands there and only 240 ms (< min_speech) of speech follow.
        let frame_at = |ms: usize| ms * MS / VAD_FRAME_SAMPLES * VAD_FRAME_SAMPLES;
        let mut audio = signal(&[(0.0, 1024), (0.9, 18_656 - 1024), (0.0, 1024)]);
        let weak = frame_at(18_400);
        audio[weak..weak + VAD_FRAME_SAMPLES].fill(0.4);
        for chunk in [1000, 512, 333] {
            let events = run(&cfg, &audio, chunk);
            let segs = segments(&events);
            assert_eq!(segs.len(), 2, "chunk {chunk}: {events:?}");
            assert!(segs[0].forced);
            let split = (weak + VAD_FRAME_SAMPLES / 2) as u64;
            assert_eq!(segs[0].start_sample + segs[0].samples.len() as u64, split);
            // The tail is emitted, seamlessly, and ends END_PAD_MS after the speech.
            assert_eq!(segs[1].start_sample, split);
            assert!(!segs[1].forced);
            let speech_end = (18_656 * MS) as u64;
            assert_eq!(segs[1].start_sample + segs[1].samples.len() as u64, speech_end + u64::from(END_PAD_MS) * MS as u64);
            // Every voiced sample reaches a segment.
            let voiced_in = audio.iter().filter(|&&s| s >= 0.35).count();
            let voiced_out: usize = segs.iter().map(|s| s.samples.iter().filter(|&&v| v >= 0.35).count()).sum();
            assert_eq!(voiced_out, voiced_in);
            assert_eq!(event_kinds(&events), vec![(true, 0), (false, 0), (true, 1), (false, 1)]);
        }
        // The same tail is emitted when the stream ends right after it (flush).
        let mut seg = SegmenterEngine::new(LevelClassifier, &cfg);
        let mut out = Vec::new();
        seg.push(&audio[..18_656 * MS], &mut out).unwrap();
        seg.flush(&mut out);
        let segs = segments(&out);
        assert_eq!(segs.len(), 2, "{out:?}");
        assert_eq!(segs[1].start_sample + segs[1].samples.len() as u64, (18_656 * MS) as u64);
        assert_eq!(event_kinds(&out), vec![(true, 0), (false, 0), (true, 1), (false, 1)]);
    }

    #[test]
    fn silent_remainder_of_a_forced_split_is_discarded() {
        let cfg = VadConfig::default();
        // Speech stops at 18.24 s; the segment (opened with pre-roll at 0.624 s) hits 18 s at
        // 18.624 s, before the 500 ms of silence close it, so the split lands in the silence and
        // the remainder holds no speech frame: nothing beyond the first part is emitted.
        let audio = signal(&[(0.0, 1024), (0.9, 18_240 - 1024), (0.0, 2048)]);
        let events = run(&cfg, &audio, 512);
        let segs = segments(&events);
        assert_eq!(segs.len(), 1, "{events:?}");
        assert!(segs[0].forced);
        assert_eq!(event_kinds(&events), vec![(true, 0), (false, 0)]);
        // A remainder holding only sub-threshold (hysteresis-band) frames after the split dip is
        // discarded too, as the same frames after an end-silence start would be.
        let audio = signal(&[(0.0, 1024), (0.9, 18_400 - 1024), (0.1, 32), (0.4, 480), (0.0, 2048)]);
        let events = run(&cfg, &audio, 512);
        assert_eq!(event_kinds(&events), vec![(true, 0), (false, 0)], "{events:?}");
    }

    #[test]
    fn continuous_speech_never_exceeds_max_length() {
        let cfg = VadConfig {
            max_utterance_ms: 4_000,
            ..VadConfig::default()
        };
        let events = run(&cfg, &signal(&[(0.9, 30_000)]), 777);
        let segs = segments(&events);
        assert!(segs.len() >= 7);
        assert!(segs.iter().all(|s| s.samples.len() <= 4_000 * MS && s.forced));
        // Contiguous coverage.
        for w in segs.windows(2) {
            assert_eq!(w[0].start_sample + w[0].samples.len() as u64, w[1].start_sample);
        }
    }

    #[test]
    fn flush_emits_confirmed_speech_and_drops_unconfirmed() {
        let cfg = VadConfig::default();
        let mut seg = SegmenterEngine::new(LevelClassifier, &cfg);
        let mut out = Vec::new();
        seg.push(&signal(&[(0.0, 640), (0.9, 640)]), &mut out).unwrap();
        seg.push(&[0.9; 100], &mut out).unwrap(); // unclassified tail
        assert!(seg.in_speech());
        seg.flush(&mut out);
        let segs = segments(&out);
        assert_eq!(segs.len(), 1);
        assert!(segs[0].forced);
        assert_eq!(segs[0].samples.len(), (400 + 640) * MS + 100);
        assert_eq!(seg.position(), ((640 + 640) * MS + 100) as u64);

        let mut out = Vec::new();
        seg.push(&signal(&[(0.9, 64)]), &mut out).unwrap();
        seg.flush(&mut out);
        assert!(out.is_empty(), "64 ms is below min_speech");
    }

    #[test]
    fn pre_roll_never_overlaps_the_previous_segment() {
        let cfg = VadConfig::default();
        // Two utterances 544 ms apart: the second pre-roll (400 ms) would reach into the first
        // segment's 200 ms tail pad; it must start at the first segment's end instead.
        let audio = signal(&[(0.9, 640), (0.0, 544), (0.9, 640), (0.0, 1024)]);
        let events = run(&cfg, &audio, 512);
        let segs = segments(&events);
        assert_eq!(segs.len(), 2);
        let first_end = segs[0].start_sample + segs[0].samples.len() as u64;
        assert!(segs[1].start_sample >= first_end);
    }
}
