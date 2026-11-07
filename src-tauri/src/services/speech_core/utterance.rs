//! Speech front end and the bounded, ordered hand-off structures behind VAD-closed utterances.
//!
//! - [`SpeechFrontEnd`]: interleaved device audio → 16 kHz mono ([`StreamResampler`]) → Silero
//!   [`Segmenter`] events. Runs on a capture's worker thread (never in the audio callback).
//! - [`BoundedQueue`]: FIFO with a hard cap that drops the OLDEST entry when full; one consumer
//!   blocks on it. Whisper's single ordered worker uses it, so segments are transcribed strictly in
//!   order and a model slower than real time can never build an unbounded backlog.
//! - [`UtteranceStore`]: take-once PCM of closed utterances for `plugin:audio|take_utterance`,
//!   capped by count and total length (oldest dropped first).

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex, PoisonError};
use std::time::{Duration, Instant};

use super::resample::StreamResampler;
use super::{Segmenter, SegmenterEvent, VadConfig, SAMPLE_RATE};

/// Device audio → VAD events.
pub struct SpeechFrontEnd {
    resampler: StreamResampler,
    segmenter: Segmenter,
    mono: Vec<f32>,
}

impl SpeechFrontEnd {
    /// `input_rate`/`channels`: the capture format. Loads the Silero model (tens of ms): call it off
    /// the async runtime and off audio threads.
    pub fn new(input_rate: u32, channels: u16, vad: &VadConfig) -> Result<Self, String> {
        Ok(Self {
            resampler: StreamResampler::new(input_rate, channels)?,
            segmenter: Segmenter::new(vad.clone())?,
            mono: Vec::with_capacity(8192),
        })
    }

    /// Feed interleaved samples in the capture format.
    pub fn push_interleaved(&mut self, interleaved: &[f32], out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        self.mono.clear();
        self.resampler.push_interleaved(interleaved, &mut self.mono)?;
        self.segmenter.push(&self.mono, out)
    }

    /// End of stream: convert the resampler tail and close any open utterance (as `forced`).
    pub fn flush(&mut self, out: &mut Vec<SegmenterEvent>) -> Result<(), String> {
        self.mono.clear();
        self.resampler.flush(&mut self.mono)?;
        self.segmenter.push(&self.mono, out)?;
        self.segmenter.flush(out)
    }
}

/// Milliseconds of 16 kHz audio at sample index `samples`.
pub fn samples_to_ms(samples: u64) -> u64 {
    samples * 1000 / u64::from(SAMPLE_RATE)
}

/// Result of [`BoundedQueue::pop_timeout`].
#[derive(Debug, PartialEq, Eq)]
pub enum Pop<T> {
    Item(T),
    /// The queue was closed (pending entries were discarded).
    Closed,
    TimedOut,
}

/// Bounded FIFO that drops the oldest entry when full. Many producers, one blocking consumer.
pub struct BoundedQueue<T> {
    inner: Mutex<QueueInner<T>>,
    ready: Condvar,
    capacity: usize,
}

struct QueueInner<T> {
    items: VecDeque<T>,
    closed: bool,
}

impl<T> BoundedQueue<T> {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            inner: Mutex::new(QueueInner {
                items: VecDeque::with_capacity(capacity),
                closed: false,
            }),
            ready: Condvar::new(),
            capacity,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner).items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Append `item`; returns the entries evicted to make room (oldest first). After [`close`] the
    /// item is rejected and returned as evicted.
    ///
    /// [`close`]: BoundedQueue::close
    pub fn push(&self, item: T) -> Vec<T> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if inner.closed {
            return vec![item];
        }
        let mut dropped = Vec::new();
        while inner.items.len() >= self.capacity {
            if let Some(old) = inner.items.pop_front() {
                dropped.push(old);
            }
        }
        inner.items.push_back(item);
        drop(inner);
        self.ready.notify_one();
        dropped
    }

    /// Next entry in FIFO order, blocking until one arrives. `None` once closed (pending entries are
    /// discarded by [`close`](BoundedQueue::close)).
    pub fn pop(&self) -> Option<T> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if inner.closed {
                return None;
            }
            if let Some(item) = inner.items.pop_front() {
                return Some(item);
            }
            inner = self.ready.wait(inner).unwrap_or_else(PoisonError::into_inner);
        }
    }

    /// Like [`pop`](BoundedQueue::pop), but gives up after `timeout`.
    pub fn pop_timeout(&self, timeout: Duration) -> Pop<T> {
        let deadline = Instant::now() + timeout;
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        loop {
            if inner.closed {
                return Pop::Closed;
            }
            if let Some(item) = inner.items.pop_front() {
                return Pop::Item(item);
            }
            let now = Instant::now();
            if now >= deadline {
                return Pop::TimedOut;
            }
            inner = self
                .ready
                .wait_timeout(inner, deadline - now)
                .unwrap_or_else(PoisonError::into_inner)
                .0;
        }
    }

    /// Remove and return the oldest entry when `take(&entry)` says so; never blocks. `None` when
    /// the queue is empty or closed, or `take` declined.
    pub fn pop_front_if(&self, take: impl FnOnce(&T) -> bool) -> Option<T> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        if inner.closed || !inner.items.front().is_some_and(take) {
            return None;
        }
        inner.items.pop_front()
    }

    /// Discard pending entries and wake the consumer; later pushes are rejected.
    pub fn close(&self) -> Vec<T> {
        let mut inner = self.inner.lock().unwrap_or_else(PoisonError::into_inner);
        inner.closed = true;
        let pending = inner.items.drain(..).collect();
        drop(inner);
        self.ready.notify_all();
        pending
    }
}

/// Take-once store of closed utterances, keyed by `(capture_id, seq)`.
pub struct UtteranceStore {
    entries: VecDeque<StoredUtterance>,
    max_entries: usize,
    max_samples: usize,
    total_samples: usize,
}

struct StoredUtterance {
    capture_id: u64,
    seq: u64,
    samples: Vec<f32>,
}

/// Default cap on untaken utterances.
pub const MAX_UNTAKEN_UTTERANCES: usize = 8;
/// Default cap on untaken audio (2 minutes of 16 kHz mono, ~7.7 MB).
pub const MAX_UNTAKEN_SAMPLES: usize = 120 * SAMPLE_RATE as usize;

impl Default for UtteranceStore {
    fn default() -> Self {
        Self::new(MAX_UNTAKEN_UTTERANCES, MAX_UNTAKEN_SAMPLES)
    }
}

impl UtteranceStore {
    pub fn new(max_entries: usize, max_samples: usize) -> Self {
        Self {
            entries: VecDeque::new(),
            max_entries: max_entries.max(1),
            max_samples: max_samples.max(1),
            total_samples: 0,
        }
    }

    /// Store an utterance; returns the `seq`s evicted (oldest first) to respect the caps. The newest
    /// utterance is always kept, even when it alone exceeds `max_samples`.
    pub fn insert(&mut self, capture_id: u64, seq: u64, samples: Vec<f32>) -> Vec<(u64, u64)> {
        self.total_samples += samples.len();
        self.entries.push_back(StoredUtterance { capture_id, seq, samples });
        let mut dropped = Vec::new();
        while self.entries.len() > 1
            && (self.entries.len() > self.max_entries || self.total_samples > self.max_samples)
        {
            if let Some(old) = self.entries.pop_front() {
                self.total_samples -= old.samples.len();
                dropped.push((old.capture_id, old.seq));
            }
        }
        dropped
    }

    /// Remove and return an utterance (each can be taken once).
    pub fn take(&mut self, capture_id: u64, seq: u64) -> Option<Vec<f32>> {
        let idx = self
            .entries
            .iter()
            .position(|e| e.capture_id == capture_id && e.seq == seq)?;
        let entry = self.entries.remove(idx)?;
        self.total_samples -= entry.samples.len();
        Some(entry.samples)
    }

    /// Remove and return the utterance `seq` of whichever capture holds it (for stores whose
    /// `seq`s are unique across captures, such as the speech pipeline's segment ids).
    pub fn take_seq(&mut self, seq: u64) -> Option<Vec<f32>> {
        let idx = self.entries.iter().position(|e| e.seq == seq)?;
        let entry = self.entries.remove(idx)?;
        self.total_samples -= entry.samples.len();
        Some(entry.samples)
    }

    /// Drop the utterances of every capture id below `capture_id` (the captures that ended),
    /// keeping those of newer captures.
    pub fn clear_before(&mut self, capture_id: u64) {
        let mut removed = 0;
        self.entries.retain(|e| {
            let keep = e.capture_id >= capture_id;
            if !keep {
                removed += e.samples.len();
            }
            keep
        });
        self.total_samples -= removed;
    }

    /// Drop everything (capture stopped or replaced).
    pub fn clear(&mut self) {
        self.entries.clear();
        self.total_samples = 0;
    }

    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[cfg(test)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// Little-endian f32 bytes (the `take_utterance` wire format).
pub fn f32_to_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::time::Duration;

    #[test]
    fn ordered_queue_drops_the_oldest_past_its_cap() {
        let q = BoundedQueue::new(3);
        assert!(q.push(1).is_empty());
        assert!(q.push(2).is_empty());
        assert!(q.push(3).is_empty());
        assert_eq!(q.push(4), vec![1]);
        assert_eq!(q.push(5), vec![2]);
        assert_eq!(q.len(), 3);
        assert_eq!((q.pop(), q.pop(), q.pop()), (Some(3), Some(4), Some(5)));
        assert!(q.is_empty());
    }

    #[test]
    fn ordered_worker_processes_in_order_and_close_wakes_it() {
        let q = Arc::new(BoundedQueue::new(64));
        let worker = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || {
                let mut seen = Vec::new();
                while let Some(v) = q.pop() {
                    seen.push(v);
                    if seen.len() == 50 {
                        break;
                    }
                }
                seen
            })
        };
        for i in 0..50 {
            assert!(q.push(i).is_empty());
        }
        assert_eq!(worker.join().unwrap(), (0..50).collect::<Vec<_>>());

        // A consumer blocked in pop() returns None on close; pending items are discarded.
        let q = Arc::new(BoundedQueue::<u32>::new(2));
        let waiter = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || q.pop())
        };
        std::thread::sleep(Duration::from_millis(20));
        q.close();
        assert_eq!(waiter.join().unwrap(), None);
        assert_eq!(q.push(7), vec![7], "closed queue rejects pushes");
        assert_eq!(q.pop(), None);
    }

    #[test]
    fn pop_timeout_and_conditional_pop() {
        let q = Arc::new(BoundedQueue::new(4));
        assert_eq!(q.pop_timeout(Duration::from_millis(10)), Pop::TimedOut);
        q.push(1);
        q.push(2);
        assert_eq!(q.pop_front_if(|&v| v == 2), None, "front is 1");
        assert_eq!(q.pop_front_if(|&v| v == 1), Some(1));
        assert_eq!(q.pop_timeout(Duration::from_millis(10)), Pop::Item(2));
        assert_eq!(q.pop_front_if(|_| true), None, "empty");
        // A waiting pop_timeout wakes for a push and for close.
        let producer = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                q.push(3);
            })
        };
        assert_eq!(q.pop_timeout(Duration::from_secs(5)), Pop::Item(3));
        producer.join().unwrap();
        let closer = {
            let q = Arc::clone(&q);
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                q.close();
            })
        };
        assert_eq!(q.pop_timeout(Duration::from_secs(5)), Pop::Closed);
        closer.join().unwrap();
        assert_eq!(q.pop_front_if(|_| true), None, "closed");
    }

    #[test]
    fn utterance_store_is_take_once_and_bounded() {
        let mut s = UtteranceStore::new(3, 100);
        assert!(s.insert(1, 0, vec![0.0; 10]).is_empty());
        assert!(s.insert(1, 1, vec![0.0; 10]).is_empty());
        assert!(s.insert(1, 2, vec![0.0; 10]).is_empty());
        assert_eq!(s.insert(1, 3, vec![0.0; 10]), vec![(1, 0)]);
        assert_eq!(s.take(1, 2).map(|v| v.len()), Some(10));
        assert_eq!(s.take(1, 2), None, "taken once");
        assert_eq!(s.take(2, 3), None, "other capture");
        // Sample cap: a big utterance evicts older ones but is itself kept.
        assert_eq!(s.insert(1, 4, vec![0.0; 150]), vec![(1, 1), (1, 3)]);
        assert_eq!(s.len(), 1);
        assert_eq!(s.take(1, 4).map(|v| v.len()), Some(150));
        assert!(s.is_empty());
    }

    #[test]
    fn utterance_store_clears_only_ended_captures() {
        let mut s = UtteranceStore::new(8, 100);
        s.insert(1, 10, vec![0.0; 30]);
        s.insert(2, 11, vec![0.0; 30]);
        s.insert(3, 12, vec![0.0; 30]);
        // A stop that ended epochs below 3 keeps the newer session's utterance.
        s.clear_before(3);
        assert_eq!(s.len(), 1);
        assert_eq!(s.take_seq(10), None);
        // The sample total follows the removals: 70 more samples fit without an eviction.
        assert!(s.insert(3, 13, vec![0.0; 70]).is_empty());
        assert_eq!(s.take_seq(12).map(|v| v.len()), Some(30), "found by seq whatever the capture");
        assert_eq!(s.take_seq(12), None, "taken once");
        assert_eq!(s.take_seq(13).map(|v| v.len()), Some(70));
        assert!(s.is_empty());
    }

    #[test]
    fn le_bytes_round_trip() {
        let v = [0.5f32, -1.0, 0.25];
        let b = f32_to_le_bytes(&v);
        assert_eq!(b.len(), 12);
        let back: Vec<f32> = b.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c)).collect();
        assert_eq!(back, v);
        assert_eq!(samples_to_ms(16_000), 1000);
    }

    /// End to end on the real model: 48 kHz stereo capture of real speech padded with silence
    /// yields exactly one utterance that covers the speech plus pre-roll.
    #[test]
    fn front_end_segments_real_speech_from_a_48k_stereo_capture() {
        use rubato::{FftFixedIn, Resampler};
        let speech16 = crate::services::speech_core::silero::tests::speech_fixture();
        // Upsample the 16 kHz fixture to 48 kHz for a realistic device format.
        let mut up = FftFixedIn::<f32>::new(16_000, 48_000, 1024, 1, 1).unwrap();
        let mut speech48 = Vec::new();
        let mut padded = speech16.clone();
        padded.resize(speech16.len().div_ceil(1024) * 1024 + 2048, 0.0);
        for chunk in padded.chunks(1024) {
            speech48.extend(up.process(&[chunk], None).unwrap().remove(0));
        }
        let lead = vec![0.0f32; 48_000 * 2];
        let tail = vec![0.0f32; 48_000 * 2];
        let mono48: Vec<f32> = lead.iter().chain(&speech48).chain(&tail).copied().collect();
        let stereo: Vec<f32> = mono48.iter().flat_map(|&s| [s, s]).collect();

        let mut fe = SpeechFrontEnd::new(48_000, 2, &VadConfig::default()).unwrap();
        let mut events = Vec::new();
        for chunk in stereo.chunks(960) {
            fe.push_interleaved(chunk, &mut events).unwrap();
        }
        fe.flush(&mut events).unwrap();
        let segs: Vec<_> = events
            .iter()
            .filter_map(|e| match e {
                SegmenterEvent::Segment(s) => Some(s),
                _ => None,
            })
            .collect();
        let starts = events
            .iter()
            .filter(|e| matches!(e, SegmenterEvent::SpeechStart { .. }))
            .count();
        assert_eq!((segs.len(), starts), (1, 1), "{:?}", segs.iter().map(|s| (s.start_sample, s.samples.len())).collect::<Vec<_>>());
        let seg = segs[0];
        assert!(!seg.forced);
        // Starts in the 2 s lead-in (pre-roll before the first word), ends in the tail silence.
        let start_ms = samples_to_ms(seg.start_sample);
        let end_ms = samples_to_ms(seg.start_sample + seg.samples.len() as u64);
        assert!((1_400..2_100).contains(&start_ms), "start {start_ms} ms");
        // The fixture has trailing synthesizer silence: measure where the voice actually stops.
        let last_voiced = speech16.iter().rposition(|s| s.abs() > 0.02).unwrap() as u64;
        let speech_end_ms = 2_000 + samples_to_ms(last_voiced);
        assert!(
            end_ms >= speech_end_ms && end_ms <= speech_end_ms + 700,
            "end {end_ms} ms vs last voiced sample at {speech_end_ms} ms"
        );
        let first_voiced = speech16.iter().position(|s| s.abs() > 0.02).unwrap() as u64;
        assert!(start_ms + 100 <= 2_000 + samples_to_ms(first_voiced), "pre-roll precedes the first word");
    }

    #[test]
    fn front_end_emits_nothing_for_ten_seconds_of_silence_on_the_real_model() {
        let mut fe = SpeechFrontEnd::new(44_100, 1, &VadConfig::default()).unwrap();
        let mut events = Vec::new();
        let silence = vec![0.0f32; 441];
        for _ in 0..1000 {
            fe.push_interleaved(&silence, &mut events).unwrap();
        }
        fe.flush(&mut events).unwrap();
        assert!(events.is_empty(), "{events:?}");
    }
}
