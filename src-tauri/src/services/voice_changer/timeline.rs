//! Per-tick timeline of the running voice changer (P3-E produces it, P3-F's live-record benchmark
//! and the metrics consume it; plan "Telemetry: a per-tick timeline (capture, start, end, enqueue,
//! estimated playout)").
//!
//! The worker publishes one [`VcTick`] per converted block; subscribers get a bounded,
//! drop-oldest copy (a slow reader never blocks or allocates on the worker). Timestamps are
//! nanoseconds on the capture hub's QPC clock (`audio::hub::now_qpc_ns`, the clock of
//! `ReadInfo::end_qpc_ns`), so capture-to-playout latency is a difference of two fields.
//!
//! # How it works
//! One process-wide broadcast ring of [`RING_CAPACITY`] ticks, written by the worker of the current
//! run (the worker publishes only while its run is current). Each slot is a seqlock over atomics:
//! the writer claims the tick's index (`fetch_add`, so two writers never share one), takes the slot
//! by moving its stamp from an older tick's even stamp to its own odd one (a compare-exchange: a
//! writer lapped by a whole ring never writes into a slot another writer holds or already filled
//! with a newer tick; its tick is dropped instead), stores the packed tick, then marks the slot with
//! the tick's even stamp; a reader copies the words and accepts them only when the stamp before
//! and after is the one it expects. Every access is atomic, so a reader racing a writer sees a torn
//! slot as "overwritten" and counts it dropped, never garbage. Subscribers keep their own cursor;
//! one that falls more than its `capacity` behind skips to the newest ticks and counts the skipped
//! ones in [`TimelineSubscriber::dropped`].
//!
//! [`publish`] is lock- and allocation-free and costs one atomic load while nobody subscribes.
//!
//! # Frozen by P3-S0 (signatures; P3-E implements the channel)
//! [`VcTick`] (+ [`tick_flags`]), [`subscribe`]`(capacity) -> TimelineSubscriber`,
//! [`TimelineSubscriber::drain`], [`TimelineSubscriber::dropped`], [`publish`] (allocation- and
//! lock-free on the worker; a no-op without subscribers).

use serde::Serialize;
use std::sync::atomic::{fence, AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

/// Bits of [`VcTick::flags`].
pub(crate) mod tick_flags {
    /// The model failed; the block was concealed (muted with click-free edges).
    pub const ERROR: u8 = 1;
    /// The output ran dry since the previous tick.
    pub const UNDERRUN: u8 = 2;
    /// The latency budget dropped input before this tick.
    pub const BUDGET_DROP: u8 = 4;
    /// Capture gap or device restart before this tick.
    pub const GAP: u8 = 8;
    /// The governor changed the quality level at this tick.
    pub const GOVERNOR_STEP: u8 = 16;
    /// Silence gate: no model inference for this block.
    pub const SILENT: u8 = 32;
}

/// One converted block.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize)]
pub(crate) struct VcTick {
    pub seq: u64,
    /// End of the block's newest captured frame (QPC ns).
    pub capture_end_ns: i64,
    /// Worker started / finished the block (QPC ns).
    pub start_ns: i64,
    pub end_ns: i64,
    /// Converted audio handed to the output queue (QPC ns).
    pub enqueue_ns: i64,
    /// Estimated time its first sample plays (enqueue + queued output + device latency, QPC ns).
    pub est_playout_ns: i64,
    /// Device frames in the block.
    pub block_frames: u32,
    /// Capture waiting after the budget (ms).
    pub backlog_ms: f32,
    /// Input dropped at this wake (ms).
    pub dropped_ms: f32,
    /// Output queued after the push (ms).
    pub queued_out_ms: f32,
    /// Model `convert` time within the block (ms).
    pub model_ms: f32,
    /// Governor quality level (0 = full quality).
    pub governor_level: u8,
    /// [`tick_flags`] bits.
    pub flags: u8,
}

/// Ticks the broadcast ring holds (about 40 s of 10 ms blocks); a subscriber's `capacity` is capped
/// at this.
pub(crate) const RING_CAPACITY: usize = 4096;
/// 64-bit words of a packed [`VcTick`].
const WORDS: usize = 10;

impl VcTick {
    fn pack(&self) -> [u64; WORDS] {
        let f = |x: f32| u64::from(x.to_bits());
        [
            self.seq,
            self.capture_end_ns as u64,
            self.start_ns as u64,
            self.end_ns as u64,
            self.enqueue_ns as u64,
            self.est_playout_ns as u64,
            u64::from(self.block_frames) | (u64::from(self.governor_level) << 32) | (u64::from(self.flags) << 40),
            f(self.backlog_ms) | (f(self.dropped_ms) << 32),
            f(self.queued_out_ms) | (f(self.model_ms) << 32),
            0,
        ]
    }

    fn unpack(w: &[u64; WORDS]) -> VcTick {
        let f = |x: u64| f32::from_bits((x & 0xffff_ffff) as u32);
        VcTick {
            seq: w[0],
            capture_end_ns: w[1] as i64,
            start_ns: w[2] as i64,
            end_ns: w[3] as i64,
            enqueue_ns: w[4] as i64,
            est_playout_ns: w[5] as i64,
            block_frames: (w[6] & 0xffff_ffff) as u32,
            governor_level: ((w[6] >> 32) & 0xff) as u8,
            flags: ((w[6] >> 40) & 0xff) as u8,
            backlog_ms: f(w[7]),
            dropped_ms: f(w[7] >> 32),
            queued_out_ms: f(w[8]),
            model_ms: f(w[8] >> 32),
        }
    }
}

/// One seqlocked slot: `stamp` is `2 * index + 2` once tick `index` is complete in it, odd while
/// it is being written, 0 before its first write.
struct Slot {
    stamp: AtomicU64,
    words: [AtomicU64; WORDS],
}

/// The broadcast ring.
struct Ring {
    slots: Box<[Slot]>,
    /// Index the next writer claims.
    claim: AtomicU64,
    /// Ticks `0..head` were published (or dropped by a lapped writer).
    head: AtomicU64,
    subscribers: AtomicUsize,
}

impl Ring {
    fn new(capacity: usize) -> Ring {
        let slots = (0..capacity.max(1))
            .map(|_| Slot {
                stamp: AtomicU64::new(0),
                words: std::array::from_fn(|_| AtomicU64::new(0)),
            })
            .collect();
        Ring {
            slots,
            claim: AtomicU64::new(0),
            head: AtomicU64::new(0),
            subscribers: AtomicUsize::new(0),
        }
    }

    fn capacity(&self) -> u64 {
        self.slots.len() as u64
    }

    fn write(&self, tick: &VcTick) {
        let index = self.claim.fetch_add(1, Ordering::AcqRel);
        let slot = &self.slots[(index % self.capacity()) as usize];
        let odd = 2 * index + 1;
        // Take the slot from an older, complete tick only (see the module docs).
        let mut cur = slot.stamp.load(Ordering::Relaxed);
        loop {
            if cur % 2 == 1 || cur > odd {
                self.head.fetch_max(index + 1, Ordering::Release);
                return;
            }
            match slot.stamp.compare_exchange_weak(cur, odd, Ordering::Acquire, Ordering::Relaxed) {
                Ok(_) => break,
                Err(now) => cur = now,
            }
        }
        // The odd stamp must be visible before any word changes.
        fence(Ordering::Release);
        for (w, v) in slot.words.iter().zip(tick.pack()) {
            w.store(v, Ordering::Relaxed);
        }
        slot.stamp.store(2 * index + 2, Ordering::Release);
        self.head.fetch_max(index + 1, Ordering::Release);
    }

    /// Tick `index`, or `None` when the writer overwrote (or is overwriting) its slot.
    fn read(&self, index: u64) -> Option<VcTick> {
        let slot = &self.slots[(index % self.capacity()) as usize];
        let want = 2 * index + 2;
        if slot.stamp.load(Ordering::Acquire) != want {
            return None;
        }
        let mut words = [0u64; WORDS];
        for (dst, w) in words.iter_mut().zip(slot.words.iter()) {
            *dst = w.load(Ordering::Relaxed);
        }
        // The words must be read before the stamp is checked again.
        fence(Ordering::Acquire);
        (slot.stamp.load(Ordering::Relaxed) == want).then(|| VcTick::unpack(&words))
    }
}

fn ring() -> &'static Ring {
    static RING: OnceLock<Ring> = OnceLock::new();
    RING.get_or_init(|| Ring::new(RING_CAPACITY))
}

/// A subscriber's end of the timeline. Dropping it unsubscribes.
pub(crate) struct TimelineSubscriber {
    ring: &'static Ring,
    /// Next tick index to read.
    cursor: u64,
    capacity: u64,
    dropped: u64,
}

impl TimelineSubscriber {
    /// Move every pending tick into `out` (oldest first; appended); returns how many.
    pub(crate) fn drain(&mut self, out: &mut Vec<VcTick>) -> usize {
        let head = self.ring.head.load(Ordering::Acquire);
        let oldest = head.saturating_sub(self.capacity);
        if self.cursor < oldest {
            self.dropped += oldest - self.cursor;
            self.cursor = oldest;
        }
        let mut n = 0;
        while self.cursor < head {
            match self.ring.read(self.cursor) {
                Some(t) => {
                    out.push(t);
                    n += 1;
                }
                // Overwritten while we read: the writer lapped this subscriber.
                None => self.dropped += 1,
            }
            self.cursor += 1;
        }
        n
    }

    /// Ticks lost because this subscriber fell behind.
    pub(crate) fn dropped(&self) -> u64 {
        self.dropped
    }
}

impl Drop for TimelineSubscriber {
    fn drop(&mut self) {
        self.ring.subscribers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Subscribe with room for `capacity` ticks (the oldest are dropped when full; at most
/// [`RING_CAPACITY`]). Only ticks published after this call are delivered.
pub(crate) fn subscribe(capacity: usize) -> TimelineSubscriber {
    let ring = ring();
    ring.subscribers.fetch_add(1, Ordering::AcqRel);
    TimelineSubscriber {
        ring,
        cursor: ring.head.load(Ordering::Acquire),
        capacity: (capacity.max(1) as u64).min(ring.capacity()),
        dropped: 0,
    }
}

/// Whether anyone subscribes (the worker skips assembling ticks otherwise).
pub(crate) fn has_subscribers() -> bool {
    ring().subscribers.load(Ordering::Acquire) > 0
}

/// Publish one tick to every subscriber (worker thread; lock- and allocation-free; a no-op without
/// subscribers).
pub(crate) fn publish(tick: &VcTick) {
    let ring = ring();
    if ring.subscribers.load(Ordering::Acquire) == 0 {
        return;
    }
    ring.write(tick);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    /// The ring is process-wide: tests that publish run one at a time.
    static SERIAL: Mutex<()> = Mutex::new(());

    fn tick(seq: u64) -> VcTick {
        VcTick {
            seq,
            capture_end_ns: -7 - seq as i64,
            start_ns: 1_000 + seq as i64,
            end_ns: 2_000 + seq as i64,
            enqueue_ns: 3_000 + seq as i64,
            est_playout_ns: i64::MAX - seq as i64,
            block_frames: 512,
            backlog_ms: 1.5,
            dropped_ms: -0.0,
            queued_out_ms: 35.25,
            model_ms: f32::MIN_POSITIVE,
            governor_level: 3,
            flags: tick_flags::ERROR | tick_flags::SILENT,
        }
    }

    #[test]
    fn ticks_pack_losslessly() {
        for t in [tick(0), tick(1 << 40), VcTick { seq: u64::MAX, ..VcTick::default() }] {
            assert_eq!(VcTick::unpack(&t.pack()), t);
        }
    }

    /// Nothing is written without subscribers; a subscriber sees only what is published after it
    /// subscribed, in order, and each tick once.
    #[test]
    fn subscribers_get_ticks_published_after_subscribing() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let head0 = ring().head.load(Ordering::SeqCst);
        if !has_subscribers() {
            publish(&tick(1));
            assert_eq!(ring().head.load(Ordering::SeqCst), head0, "no subscriber: no write");
        }
        let mut sub = subscribe(64);
        assert!(has_subscribers());
        let mut out = Vec::new();
        assert_eq!(sub.drain(&mut out), 0);
        for s in 10..20 {
            publish(&tick(s));
        }
        assert_eq!(sub.drain(&mut out), 10);
        assert_eq!(out.iter().map(|t| t.seq).collect::<Vec<_>>(), (10..20).collect::<Vec<_>>());
        assert_eq!(out[3], tick(13));
        assert_eq!(sub.drain(&mut out), 0, "each tick once");
        assert_eq!(sub.dropped(), 0);
    }

    /// A subscriber that falls behind its capacity loses the oldest ticks (counted), never the newest.
    #[test]
    fn a_slow_subscriber_drops_the_oldest() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut sub = subscribe(8);
        for s in 0..30 {
            publish(&tick(s));
        }
        let mut out = Vec::new();
        assert_eq!(sub.drain(&mut out), 8);
        assert_eq!(out.first().unwrap().seq, 22);
        assert_eq!(out.last().unwrap().seq, 29);
        assert_eq!(sub.dropped(), 22);
        // A capacity above the ring is capped at the ring.
        let big = subscribe(usize::MAX);
        assert_eq!(big.capacity, RING_CAPACITY as u64);
    }

    /// A reader racing the writer never sees a torn tick: every tick it gets is one that was
    /// published intact, in increasing order, and the rest are counted as dropped.
    #[test]
    fn concurrent_readers_never_see_torn_ticks() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut sub = subscribe(16);
        let total = 200_000u64;
        let writer = std::thread::spawn(move || {
            for s in 0..total {
                publish(&tick(s));
            }
        });
        let mut out = Vec::new();
        let mut last: Option<u64> = None;
        let mut got = 0u64;
        loop {
            let finished = writer.is_finished();
            out.clear();
            sub.drain(&mut out);
            for t in &out {
                assert_eq!(*t, tick(t.seq), "torn tick");
                if let Some(l) = last {
                    assert!(t.seq > l);
                }
                last = Some(t.seq);
                got += 1;
            }
            if finished {
                break;
            }
        }
        writer.join().unwrap();
        out.clear();
        sub.drain(&mut out);
        got += out.len() as u64;
        assert_eq!(got + sub.dropped(), total, "every tick delivered or counted");
    }

    /// Two writers at once (a superseded worker still publishing): they never share a slot, so a
    /// reader never accepts a tick that mixes the fields of two ticks.
    #[test]
    fn concurrent_writers_never_tear_a_tick() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut sub = subscribe(64);
        let per = 100_000u64;
        let writers: Vec<_> = (0..2u64)
            .map(|w| {
                std::thread::spawn(move || {
                    for s in 0..per {
                        publish(&tick(w * (1 << 40) + s));
                    }
                })
            })
            .collect();
        let mut out = Vec::new();
        let mut got = 0u64;
        loop {
            let finished = writers.iter().all(|w| w.is_finished());
            out.clear();
            sub.drain(&mut out);
            for t in &out {
                assert_eq!(*t, tick(t.seq), "torn tick");
                got += 1;
            }
            if finished {
                break;
            }
        }
        for w in writers {
            w.join().unwrap();
        }
        out.clear();
        sub.drain(&mut out);
        for t in &out {
            assert_eq!(*t, tick(t.seq), "torn tick");
        }
        got += out.len() as u64;
        assert_eq!(got + sub.dropped(), 2 * per, "every tick delivered or counted");
    }
}
