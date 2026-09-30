//! Utterance framing for the speech pipeline (P2-A): segment bookkeeping, the session clock, the
//! streaming pre-roll history and batch merging. Internal to the pipeline: no frozen API.
//!
//! - [`SegmentBook`]: per-segment timing, the revision counter, the Smart Turn verdict and whether
//!   the segment already ended (a final or a `dropped`). Every `speech:*` emit of a segment goes
//!   through it under the pipeline's emit lock, so revisions are strictly increasing in emit order
//!   and a segment that ended with `dropped` never gets a final (and vice versa).
//! - [`SessionClock`]: segment times in milliseconds since the session started, on the capture
//!   clock (QPC) when the hub provides it, so mute gaps and stream restarts do not make later
//!   segments look earlier than they were.
//! - [`AudioHistory`]: the last few seconds of 16 kHz audio, from which a streaming engine gets
//!   the pre-roll of a segment that the VAD confirms only after `min_speech_ms`.
//! - [`merge_audio`]: closed utterances waiting for a slow engine are decoded as one batch.
//!
//! Segment ids: a per-pipeline-key counter that is never reused (see `speech::events`); the
//! retained PCM of `take_pipeline_utterance` lives in `speech_core::utterance::UtteranceStore`,
//! keyed by that id.

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use crate::services::speech_core::SAMPLE_RATE;

/// `speech:utterance` kind `dropped` reasons (the TS driver maps them; unknown reasons are shown
/// as-is). Every closed utterance of a pipeline with an engine ends with at least one final or
/// exactly one `dropped`; a segment dropped while open (muted) gets `dropped` instead of
/// `utterance`.
pub(crate) mod reasons {
    /// More utterances waited for the engine than `maxBacklog`; the oldest was dropped.
    pub const BACKLOG: &str = "backlog";
    /// The utterance waited longer than `latencyBudgetMs` for its first decode.
    pub const LATENCY_BUDGET: &str = "latency_budget";
    /// The STT mute started while the utterance was open.
    pub const MUTED: &str = "muted";
    /// The engine heard no speech (no-speech probability, empty result or the hallucination
    /// filter).
    pub const NO_SPEECH: &str = "no_speech";
    /// Decoded together with the previous utterance, whose final carries the text of both.
    pub const MERGED: &str = "merged";
    /// The engine failed on this utterance (a status event says why).
    pub const ENGINE_ERROR: &str = "engine_error";
    /// The HTTP engine was rate limited and gave up on this utterance.
    pub const RATE_LIMITED: &str = "rate_limited";
    /// A streaming-only route skipped the utterance (the streaming engine fell behind, or needs a
    /// language that is not known yet).
    pub const STREAM_SKIPPED: &str = "stream_skipped";
    /// The two-pass queue refused the re-decode and there was no first-pass final.
    pub const QUEUE_FULL: &str = "queue_full";

    /// Whether the reason means speech was lost (counted in `PipelineStatus::dropped`).
    pub fn is_loss(reason: &str) -> bool {
        !matches!(reason, NO_SPEECH | MERGED)
    }
}

/// Silence inserted between merged utterances.
pub(crate) const MERGE_GAP_MS: u32 = 200;
/// Longest merged batch (inside Whisper's 30 s window).
pub(crate) const MERGE_MAX_MS: u32 = 28_000;
/// Segments the book remembers (older ones cannot get a late final any more).
pub(crate) const BOOK_CAPACITY: usize = 128;
/// Audio Smart Turn looks at (the end of the segment).
pub(crate) const TURN_WINDOW_MS: u32 = 8_000;

/// 16 kHz samples in `ms` milliseconds.
pub(crate) fn ms_to_samples(ms: u32) -> usize {
    (u64::from(ms) * u64::from(SAMPLE_RATE) / 1000) as usize
}

/// Milliseconds of `samples` 16 kHz samples.
pub(crate) fn samples_to_ms(samples: u64) -> u64 {
    samples * 1000 / u64::from(SAMPLE_RATE)
}

/// What the pipeline knows about one segment.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SegmentEntry {
    pub start_ms: u64,
    pub end_ms: u64,
    pub forced: bool,
    /// When the VAD closed it (`None` while open).
    pub closed_at: Option<Instant>,
    /// Revision the next partial / final gets.
    pub next_revision: u32,
    /// Finals emitted so far.
    pub finals: u32,
    /// Ended with `dropped`.
    pub dropped: bool,
    /// Smart Turn verdict (`None` = not evaluated).
    pub turn_end: Option<bool>,
    /// Smart Turn probability of the verdict.
    pub turn_probability: Option<f32>,
}

impl SegmentEntry {
    fn new(start_ms: u64) -> Self {
        Self {
            start_ms,
            end_ms: start_ms,
            forced: false,
            closed_at: None,
            next_revision: 0,
            finals: 0,
            dropped: false,
            turn_end: None,
            turn_probability: None,
        }
    }

    /// No final and no `dropped` yet.
    pub fn is_pending(&self) -> bool {
        self.finals == 0 && !self.dropped
    }
}

/// Bounded map of the session's recent segments (oldest forgotten first).
#[derive(Debug)]
pub(crate) struct SegmentBook {
    entries: VecDeque<(u64, SegmentEntry)>,
    capacity: usize,
}

impl Default for SegmentBook {
    fn default() -> Self {
        Self::new(BOOK_CAPACITY)
    }
}

impl SegmentBook {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            capacity: capacity.max(1),
        }
    }

    /// Record a new (open) segment.
    pub fn open(&mut self, id: u64, start_ms: u64) {
        if let Some(e) = self.get_mut(id) {
            *e = SegmentEntry::new(start_ms);
            return;
        }
        while self.entries.len() >= self.capacity {
            self.entries.pop_front();
        }
        self.entries.push_back((id, SegmentEntry::new(start_ms)));
    }

    /// The segment closed at `end_ms` (VAD end silence, max length or flush).
    pub fn close(&mut self, id: u64, end_ms: u64, forced: bool, closed_at: Instant) {
        if self.get(id).is_none() {
            self.open(id, end_ms);
        }
        if let Some(e) = self.get_mut(id) {
            e.end_ms = end_ms.max(e.start_ms);
            e.forced = forced;
            e.closed_at = Some(closed_at);
        }
    }

    pub fn get(&self, id: u64) -> Option<&SegmentEntry> {
        self.entries.iter().find(|(i, _)| *i == id).map(|(_, e)| e)
    }

    pub fn get_mut(&mut self, id: u64) -> Option<&mut SegmentEntry> {
        self.entries.iter_mut().find(|(i, _)| *i == id).map(|(_, e)| e)
    }

    /// The revision a partial / final of `id` gets now (and advance the counter); `None` when the
    /// segment is unknown (forgotten) or already ended with `dropped`.
    pub fn take_revision(&mut self, id: u64) -> Option<u32> {
        let e = self.get_mut(id)?;
        if e.dropped {
            return None;
        }
        let r = e.next_revision;
        e.next_revision = r.saturating_add(1);
        Some(r)
    }

    /// The revision the next partial / final of `id` would get.
    pub fn peek_revision(&self, id: u64) -> Option<u32> {
        self.get(id).map(|e| e.next_revision)
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }
}

/// Maps 16 kHz sample positions of the session's audio to milliseconds since the session started.
///
/// Between discontinuities the audio clock is exact; the offset to wall time is re-measured on
/// every read that carries a QPC capture time and only ever grows (time spent muted or while a
/// device restarts is not audio, but it did pass).
#[derive(Clone, Debug, Default)]
pub(crate) struct SessionClock {
    start_qpc_ns: Option<i64>,
    offset_ms: u64,
}

impl SessionClock {
    /// A clock whose zero is `start_qpc_ns` (the capture clock at session start), if known.
    pub fn new(start_qpc_ns: Option<i64>) -> Self {
        Self {
            start_qpc_ns,
            offset_ms: 0,
        }
    }

    /// After a read: `pos` samples were consumed in total, the newest of them captured at
    /// `end_qpc_ns`.
    pub fn observe(&mut self, pos: u64, end_qpc_ns: Option<i64>) {
        let (Some(start), Some(end)) = (self.start_qpc_ns, end_qpc_ns) else { return };
        let wall_ms = u64::try_from((end - start).max(0) / 1_000_000).unwrap_or(0);
        let audio_ms = samples_to_ms(pos);
        self.offset_ms = self.offset_ms.max(wall_ms.saturating_sub(audio_ms));
    }

    /// Milliseconds since the session started at sample position `sample`.
    pub fn ms(&self, sample: u64) -> u64 {
        self.offset_ms + samples_to_ms(sample)
    }
}

/// The most recent 16 kHz audio of the session (a ring over absolute sample positions).
#[derive(Debug)]
pub(crate) struct AudioHistory {
    buf: VecDeque<f32>,
    /// Absolute position of `buf[0]`.
    start: u64,
    capacity: usize,
}

impl AudioHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            buf: VecDeque::with_capacity(capacity.max(1)),
            start: 0,
            capacity: capacity.max(1),
        }
    }

    /// Append `chunk`, which starts at absolute position `pos` (a jump forward leaves no gap
    /// data: the history restarts there).
    pub fn push(&mut self, pos: u64, chunk: &[f32]) {
        if pos != self.end() {
            self.buf.clear();
            self.start = pos;
        }
        self.buf.extend(chunk.iter().copied());
        let excess = self.buf.len().saturating_sub(self.capacity);
        if excess > 0 {
            self.buf.drain(..excess);
            self.start += excess as u64;
        }
    }

    /// Absolute position just past the newest sample.
    pub fn end(&self) -> u64 {
        self.start + self.buf.len() as u64
    }

    /// Samples `[from, to)` that are still held (clamped to what is available).
    pub fn range(&self, from: u64, to: u64) -> Vec<f32> {
        let from = from.max(self.start);
        let to = to.min(self.end());
        if from >= to {
            return Vec::new();
        }
        let a = (from - self.start) as usize;
        let b = (to - self.start) as usize;
        self.buf.range(a..b).copied().collect()
    }
}

/// Concatenate utterances with `gap` samples of silence between them.
pub(crate) fn merge_audio(parts: &[Arc<[f32]>], gap: usize) -> Arc<[f32]> {
    match parts {
        [] => Arc::from(Vec::<f32>::new()),
        [one] => Arc::clone(one),
        _ => {
            let total = parts.iter().map(|p| p.len()).sum::<usize>() + gap * (parts.len() - 1);
            let mut out = Vec::with_capacity(total);
            for (i, p) in parts.iter().enumerate() {
                if i > 0 {
                    out.resize(out.len() + gap, 0.0);
                }
                out.extend_from_slice(p);
            }
            Arc::from(out)
        }
    }
}

/// The last `window_ms` of `audio`.
pub(crate) fn tail(audio: &[f32], window_ms: u32) -> &[f32] {
    let n = ms_to_samples(window_ms);
    &audio[audio.len().saturating_sub(n)..]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn book_numbers_revisions_and_forgets_the_oldest() {
        let mut b = SegmentBook::new(2);
        b.open(1, 100);
        assert_eq!(b.take_revision(1), Some(0));
        assert_eq!(b.take_revision(1), Some(1));
        assert_eq!(b.peek_revision(1), Some(2));
        b.close(1, 900, true, Instant::now());
        let e = b.get(1).unwrap();
        assert_eq!((e.start_ms, e.end_ms, e.forced), (100, 900, true));
        assert!(e.closed_at.is_some() && e.is_pending());
        b.open(2, 1000);
        b.open(3, 2000);
        assert_eq!(b.len(), 2);
        assert!(b.get(1).is_none(), "oldest forgotten");
        assert_eq!(b.take_revision(1), None);
        b.get_mut(2).unwrap().dropped = true;
        assert_eq!(b.take_revision(2), None, "a dropped segment gets nothing more");
        // Closing an unknown segment records it (a flush after the book forgot it).
        b.close(9, 50, false, Instant::now());
        assert_eq!(b.get(9).unwrap().end_ms, 50);
    }

    #[test]
    fn clock_follows_wall_time_across_gaps_and_never_goes_back() {
        let mut c = SessionClock::new(Some(1_000_000_000));
        assert_eq!(c.ms(16_000), 1000);
        // 1 s of audio captured by t = 1 s: no offset.
        c.observe(16_000, Some(2_000_000_000));
        assert_eq!(c.ms(16_000), 1000);
        // A 5 s mute: the next second of audio ends at t = 7 s.
        c.observe(32_000, Some(8_000_000_000));
        assert_eq!(c.ms(16_000), 6000);
        assert_eq!(c.ms(32_000), 7000);
        // Jitter (a read reported slightly early) never moves times back.
        c.observe(48_000, Some(8_900_000_000));
        assert_eq!(c.ms(32_000), 7000);
        // Without a capture clock the audio clock is used as is.
        let mut a = SessionClock::new(None);
        a.observe(32_000, Some(99_000_000_000));
        assert_eq!(a.ms(32_000), 2000);
    }

    #[test]
    fn history_keeps_the_newest_samples_by_absolute_position() {
        let mut h = AudioHistory::new(5);
        h.push(0, &[0.0, 1.0, 2.0]);
        h.push(3, &[3.0, 4.0, 5.0, 6.0]);
        assert_eq!(h.end(), 7);
        assert_eq!(h.range(0, 7), vec![2.0, 3.0, 4.0, 5.0, 6.0], "oldest two evicted");
        assert_eq!(h.range(4, 6), vec![4.0, 5.0]);
        assert!(h.range(6, 4).is_empty());
        // A jump restarts the history.
        h.push(100, &[9.0]);
        assert_eq!(h.range(0, 200), vec![9.0]);
        assert_eq!(h.end(), 101);
    }

    #[test]
    fn merging_inserts_gaps_and_keeps_single_parts() {
        let a: Arc<[f32]> = Arc::from(vec![1.0f32, 1.0]);
        let b: Arc<[f32]> = Arc::from(vec![2.0f32]);
        assert!(Arc::ptr_eq(&merge_audio(std::slice::from_ref(&a), 3), &a));
        assert_eq!(&*merge_audio(&[a, b], 2), &[1.0, 1.0, 0.0, 0.0, 2.0]);
        assert!(merge_audio(&[], 2).is_empty());
        assert_eq!(tail(&[1.0; 20_000], 1000).len(), 16_000);
        assert_eq!(tail(&[1.0; 10], 1000).len(), 10);
        assert!(reasons::is_loss(reasons::MUTED) && !reasons::is_loss(reasons::MERGED));
    }
}
