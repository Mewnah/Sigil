//! Input-side latency budget: the worker-owned capture backlog and the capture-callback push.
//!
//! The capture callback copies whole frames into a lock-free SPSC ring ([`push_capture`]) and
//! wakes the worker. On every wake the worker drains the ring completely into an
//! [`InputBacklog`] (its own pre-accumulator), enforces the latency budget there and then takes
//! fixed-size blocks for conversion. Doing the budget on the worker side is what makes "drop the
//! oldest audio" possible with an SPSC ring: the producer can only ever refuse the newest frames,
//! the consumer can discard the oldest.
//!
//! Budget rules (plan: "latency budgets drop stale audio instead of queueing it", research §4.3#2):
//! - The backlog (ring + pre-accumulator) may hold at most `max(budget, one block)` frames. Above
//!   that, the oldest whole frames are dropped down to a low-water mark of `max(block, budget / 2)`,
//!   so a PC that is persistently a little too slow skips audio in a few larger steps instead of
//!   clicking on every block, and a stalled GPU is back within budget on the very next wake.
//! - The drop is a splice, not a cut: the first [`SPLICE_FADE_MS`] of the dropped audio (which
//!   continues the audio already converted) crossfades into the first kept frames, so every
//!   converter downstream (classic, ONNX, sidecar, RVC harness) sees a continuous signal.
//! - Frames the ring had to refuse (the worker stalled for longer than the ring holds) are newer
//!   than everything in the backlog and are gone; they count as dropped, and the audio around the
//!   hole fades out / in.
//! - Accounting is exact, in whole frames: `captured == consumed + dropped + backlog (+ a pending
//!   overflow not yet enforced)` at every point.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};

/// Default latency budget (ms) of captured audio waiting to be converted.
pub(crate) const DEFAULT_LATENCY_BUDGET_MS: f64 = 300.0;
pub(crate) const MIN_LATENCY_BUDGET_MS: f64 = 100.0;
pub(crate) const MAX_LATENCY_BUDGET_MS: f64 = 2000.0;
/// Crossfade length at a budget splice, and fade length around an overflow hole.
pub(crate) const SPLICE_FADE_MS: f64 = 5.0;
/// Capture ring headroom beyond the largest budget: a worker stall shorter than this (plus the
/// largest budget) never makes the ring refuse audio, so the budget can still keep the newest.
const RING_STALL_HEADROOM_MS: f64 = 2000.0;
/// Ring size bounds (interleaved samples) for exotic device formats (e.g. 192 kHz x 8 ch).
const MIN_RING_SAMPLES: usize = 16_384;
const MAX_RING_SAMPLES: usize = 4_194_304;

/// Clamp a latency budget from IPC / disk to the supported range (non-finite = default).
pub(crate) fn clamp_latency_budget_ms(ms: f64) -> f64 {
    if ms.is_finite() {
        ms.clamp(MIN_LATENCY_BUDGET_MS, MAX_LATENCY_BUDGET_MS)
    } else {
        DEFAULT_LATENCY_BUDGET_MS
    }
}

/// Device chunks of margin the latency budget keeps beyond an engine block (see [`effective_latency_budget_ms`]): the chunk
/// that was waiting when the model call began and the jitter in when the capture delivers the next one.
const BUDGET_BLOCK_MARGIN_CHUNKS: f64 = 2.0;

/// The latency budget in force (ms): the stored one, raised to what the playing voice's model call needs.
///
/// A voice that converts a block of `engine_block_ms` in one call keeps nothing out of the backlog while the call runs, so
/// a call that takes as long as its block (real time) leaves a block of captured audio waiting, plus the chunk that
/// arrived meanwhile. A budget under that drops input after every call, however fast the PC is: a 100 ms budget with
/// 120 ms blocks threw away most of the speech. The stored budget is kept for what it is for (a PC that falls
/// further behind than a block still drops), and only a value under `engine_block_ms` plus
/// [`BUDGET_BLOCK_MARGIN_CHUNKS`] device chunks (`chunk_ms`) is raised to it. `None` (no fixed block) leaves the budget
/// alone. The default 300 ms covers every block up to 200 ms.
pub(crate) fn effective_latency_budget_ms(budget_ms: f64, engine_block_ms: Option<f64>, chunk_ms: f64) -> f64 {
    let stored = clamp_latency_budget_ms(budget_ms);
    match engine_block_ms {
        Some(block) if block.is_finite() && block > 0.0 => {
            let chunk = if chunk_ms.is_finite() { chunk_ms.max(0.0) } else { 0.0 };
            stored.max(block + BUDGET_BLOCK_MARGIN_CHUNKS * chunk)
        }
        _ => stored,
    }
}

/// Frames in `ms` at `rate` Hz, rounded to the nearest frame.
pub(crate) fn ms_to_frames(ms: f64, rate: u32) -> usize {
    (ms.max(0.0) * f64::from(rate) / 1000.0).round() as usize
}

/// Capacity (interleaved samples, whole frames) of the lock-free capture ring: the largest budget
/// plus [`RING_STALL_HEADROOM_MS`], and never below four blocks.
pub(crate) fn capture_ring_capacity(rate: u32, ch: usize, block_frames: usize) -> usize {
    let ch = ch.max(1);
    let frames = ms_to_frames(MAX_LATENCY_BUDGET_MS + RING_STALL_HEADROOM_MS, rate).max(block_frames.saturating_mul(4));
    let samples = frames.saturating_mul(ch).clamp(MIN_RING_SAMPLES, MAX_RING_SAMPLES);
    (samples / ch).max(1) * ch
}

/// Copy the whole frames of one capture callback (f32, from `pcm_convert`) into the ring.
/// Real-time safe: no allocation, no lock; one ring commit per call, so the worker never observes
/// a partial frame. Returns the number of frames the ring had no room for (the newest ones; the
/// caller counts them).
pub(crate) fn push_capture(producer: &mut rtrb::Producer<f32>, data: &[f32], ch: usize) -> usize {
    let ch = ch.max(1);
    let frames = data.len() / ch;
    let n = frames.min(producer.slots() / ch);
    if n > 0 {
        match producer.write_chunk_uninit(n * ch) {
            Ok(chunk) => {
                chunk.fill_from_iter(data[..n * ch].iter().copied());
            }
            Err(_) => return frames,
        }
    }
    frames - n
}

/// What one [`InputBacklog::enforce`] call discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DropEvent {
    /// Oldest backlog frames dropped to get back within the budget.
    pub(crate) frames: usize,
    /// Frames the capture ring refused (lost before reaching the backlog).
    pub(crate) overflow_frames: u64,
}

/// Worker-owned capture backlog (interleaved f32, whole frames) with the latency budget.
pub(crate) struct InputBacklog {
    buf: VecDeque<f32>,
    ch: usize,
    rate: u32,
    budget_frames: usize,
    fade_frames: usize,
    /// Frames the ring refused since the last `enforce` (newer than everything in `buf`).
    gap_frames: u64,
    /// Frames of fade-in still to apply to the next audio pushed (after a hole or a total drop).
    fade_in_left: usize,
    captured: u64,
    consumed: u64,
    dropped: u64,
    drop_events: u64,
}

impl InputBacklog {
    /// `capacity_samples` pre-allocates the backlog so a drained stall does not reallocate.
    pub(crate) fn new(ch: usize, rate: u32, budget_ms: f64, capacity_samples: usize) -> Self {
        let ch = ch.max(1);
        let rate = rate.max(1);
        let mut s = Self {
            buf: VecDeque::with_capacity(capacity_samples.max(ch)),
            ch,
            rate,
            budget_frames: 0,
            fade_frames: ms_to_frames(SPLICE_FADE_MS, rate).max(1),
            gap_frames: 0,
            fade_in_left: 0,
            captured: 0,
            consumed: 0,
            dropped: 0,
            drop_events: 0,
        };
        s.set_budget_ms(budget_ms);
        s
    }

    /// Change the budget (clamped); takes effect at the next [`Self::enforce`].
    pub(crate) fn set_budget_ms(&mut self, budget_ms: f64) {
        self.budget_frames = ms_to_frames(clamp_latency_budget_ms(budget_ms), self.rate).max(1);
    }

    #[cfg(test)]
    pub(crate) fn budget_frames(&self) -> usize {
        self.budget_frames
    }

    /// Frames waiting in the backlog.
    pub(crate) fn frames(&self) -> usize {
        self.buf.len() / self.ch
    }

    #[cfg(test)]
    pub(crate) fn captured_frames(&self) -> u64 {
        self.captured
    }

    #[cfg(test)]
    pub(crate) fn consumed_frames(&self) -> u64 {
        self.consumed
    }

    /// Frames dropped by the budget plus frames the ring refused, since this backlog was created.
    pub(crate) fn dropped_frames(&self) -> u64 {
        self.dropped
    }

    pub(crate) fn drop_events(&self) -> u64 {
        self.drop_events
    }

    /// Ring overflow not yet accounted by [`Self::enforce`].
    #[cfg(test)]
    pub(crate) fn pending_overflow_frames(&self) -> u64 {
        self.gap_frames
    }

    /// Append captured audio (whole frames; a trailing partial frame is ignored).
    #[cfg(test)]
    pub(crate) fn push(&mut self, interleaved: &[f32]) {
        let n = interleaved.len() / self.ch * self.ch;
        self.push_slice(&interleaved[..n]);
    }

    fn push_slice(&mut self, samples: &[f32]) {
        let ch = self.ch;
        let frames = samples.len() / ch;
        self.captured += frames as u64;
        if self.fade_in_left == 0 {
            self.buf.extend(samples.iter().copied());
            return;
        }
        let total = self.fade_frames;
        for frame in samples.chunks_exact(ch) {
            if self.fade_in_left > 0 {
                let done = total - self.fade_in_left;
                let g = fade_gain(done, total);
                self.buf.extend(frame.iter().map(|&s| s * g));
                self.fade_in_left -= 1;
            } else {
                self.buf.extend(frame.iter().copied());
            }
        }
    }

    /// Move everything the capture callback has committed (whole frames) into the backlog.
    pub(crate) fn drain_from(&mut self, consumer: &mut rtrb::Consumer<f32>) {
        let n = consumer.slots() / self.ch * self.ch;
        if n == 0 {
            return;
        }
        if let Ok(chunk) = consumer.read_chunk(n) {
            let (a, b) = chunk.as_slices();
            // `a` and `b` together are whole frames; each on its own may split one, so push them
            // as one run (the fade-in walks frames across the seam correctly only if joined).
            if a.len() % self.ch == 0 {
                self.push_slice(a);
                self.push_slice(b);
            } else {
                let split = a.len() - a.len() % self.ch;
                self.push_slice(&a[..split]);
                let mut seam = [0.0f32; 16];
                let tail = &a[split..];
                let head_len = self.ch - tail.len();
                if self.ch <= seam.len() {
                    seam[..tail.len()].copy_from_slice(tail);
                    seam[tail.len()..self.ch].copy_from_slice(&b[..head_len]);
                    let frame = seam;
                    self.push_slice(&frame[..self.ch]);
                } else {
                    let frame: Vec<f32> = tail.iter().chain(b[..head_len].iter()).copied().collect();
                    self.push_slice(&frame);
                }
                self.push_slice(&b[head_len..]);
            }
            chunk.commit_all();
        }
    }

    /// One wake's intake: everything the ring holds, then the frames the capture callback had to
    /// refuse (its `overflow` counter, reset).
    ///
    /// The order is what places a hole correctly. The callback can only refuse audio while the
    /// ring is full, and the ring only gains room when this drain commits, so every refusal
    /// counted when the counter is read here happened after the frames just drained and before
    /// anything still in the ring: the hole really is right after the backlog, as
    /// [`Self::enforce`] assumes. Reading the counter first would let a refusal that lands between
    /// the read and the drain be charged to the next wake, after audio captured behind it.
    pub(crate) fn ingest(&mut self, consumer: &mut rtrb::Consumer<f32>, overflow: &AtomicU64) {
        self.ingest_with(consumer, overflow, || {});
    }

    /// [`Self::ingest`] with a hook between the drain and the counter read (tests race a capture
    /// callback there).
    fn ingest_with(&mut self, consumer: &mut rtrb::Consumer<f32>, overflow: &AtomicU64, between: impl FnOnce()) {
        self.drain_from(consumer);
        between();
        self.note_overflow(overflow.swap(0, Ordering::Relaxed));
    }

    /// Record frames the capture ring refused (the newest audio of a stall longer than the ring).
    pub(crate) fn note_overflow(&mut self, frames: u64) {
        if frames > 0 {
            self.gap_frames += frames;
            self.captured += frames;
        }
    }

    /// Enforce the budget for blocks of `block_frames`. Returns what was dropped, if anything.
    pub(crate) fn enforce(&mut self, block_frames: usize) -> Option<DropEvent> {
        let ch = self.ch;
        let block_frames = block_frames.max(1);
        let limit = self.budget_frames.max(block_frames);
        let gap = std::mem::take(&mut self.gap_frames);
        let have = self.frames();
        if have as u64 + gap <= limit as u64 {
            if gap == 0 {
                return None;
            }
            // The ring refused audio right after what the backlog holds: fade out into the hole
            // and fade the audio after it back in.
            self.fade_out_tail();
            self.fade_in_left = self.fade_frames;
            self.dropped += gap;
            self.drop_events += 1;
            return Some(DropEvent { frames: 0, overflow_frames: gap });
        }
        let target = (self.budget_frames / 2).max(block_frames).min(limit);
        let keep = ((target as u64).saturating_sub(gap)).min(have as u64) as usize;
        let drop = have - keep;
        if keep > 0 && drop > 0 {
            self.splice(drop, keep);
        }
        self.buf.drain(..drop * ch);
        if gap > 0 {
            // The kept audio (if any) ends at the hole the ring left.
            self.fade_out_tail();
        }
        if gap > 0 || keep == 0 {
            // Whatever comes next does not continue what the backlog last held.
            self.fade_in_left = self.fade_frames;
        }
        self.dropped += drop as u64 + gap;
        self.drop_events += 1;
        Some(DropEvent { frames: drop, overflow_frames: gap })
    }

    /// Crossfade the first frames of the region about to be dropped (`[0, drop)`, which continues
    /// the audio already converted) into the first kept frames (`[drop, drop + keep)`), equal power.
    fn splice(&mut self, drop: usize, keep: usize) {
        let ch = self.ch;
        let n = self.fade_frames.min(drop).min(keep);
        for i in 0..n {
            let t = (i as f32 + 0.5) / n as f32;
            let w_in = (t * std::f32::consts::FRAC_PI_2).sin();
            let w_out = (t * std::f32::consts::FRAC_PI_2).cos();
            for c in 0..ch {
                let old = self.buf[i * ch + c];
                let new = &mut self.buf[(drop + i) * ch + c];
                *new = *new * w_in + old * w_out;
            }
        }
    }

    /// Fade the newest frames of the backlog out to silence.
    fn fade_out_tail(&mut self) {
        let ch = self.ch;
        let frames = self.frames();
        let n = self.fade_frames.min(frames);
        for i in 0..n {
            // i = 0 is the newest frame (quietest).
            let g = fade_gain(i, n);
            let f = frames - 1 - i;
            for c in 0..ch {
                self.buf[f * ch + c] *= g;
            }
        }
    }

    /// Move the oldest `frames` frames into `out` (cleared first). `false` (and `out` untouched)
    /// while fewer frames are waiting.
    pub(crate) fn take_block(&mut self, out: &mut Vec<f32>, frames: usize) -> bool {
        let n = frames * self.ch;
        if frames == 0 || self.buf.len() < n {
            return false;
        }
        out.clear();
        out.extend(self.buf.drain(..n));
        self.consumed += frames as u64;
        true
    }
}

/// Raised-cosine fade-in gain for frame `done` of `total` (never exactly 0 or 1 inside the fade).
fn fade_gain(done: usize, total: usize) -> f32 {
    let t = (done as f32 + 0.5) / total.max(1) as f32;
    0.5 - 0.5 * (std::f32::consts::PI * t).cos()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtrb::RingBuffer;

    /// Frames numbered from `from` (every channel of frame n holds n), so order, loss and splits
    /// are visible.
    fn numbered(from: u32, frames: usize, ch: usize) -> Vec<f32> {
        (0..frames).flat_map(|i| std::iter::repeat_n((from as usize + i) as f32, ch)).collect()
    }

    fn assert_accounting(b: &InputBacklog) {
        assert_eq!(
            b.captured_frames(),
            b.consumed_frames() + b.dropped_frames() + b.frames() as u64 + b.pending_overflow_frames(),
            "captured == consumed + dropped + backlog"
        );
    }

    #[test]
    fn budget_is_clamped_and_converted_to_frames() {
        assert_eq!(clamp_latency_budget_ms(50.0), MIN_LATENCY_BUDGET_MS);
        assert_eq!(clamp_latency_budget_ms(5_000.0), MAX_LATENCY_BUDGET_MS);
        assert_eq!(clamp_latency_budget_ms(f64::NAN), DEFAULT_LATENCY_BUDGET_MS);
        assert_eq!(clamp_latency_budget_ms(250.0), 250.0);
        let b = InputBacklog::new(2, 48_000, 300.0, 0);
        assert_eq!(b.budget_frames(), 14_400);
        // The ring holds the largest budget plus the stall headroom, in whole frames.
        let cap = capture_ring_capacity(48_000, 2, 512);
        assert_eq!(cap, 4 * 48_000 * 2);
        assert_eq!(capture_ring_capacity(192_000, 6, 512) % 6, 0);
        assert!(capture_ring_capacity(192_000, 8, 512) <= MAX_RING_SAMPLES);
    }

    /// Under the budget nothing is touched; above it the oldest whole frames go, down to the
    /// low-water mark, and the counts add up exactly.
    #[test]
    fn budget_drops_the_oldest_whole_frames_and_keeps_accounting_exact() {
        let ch = 2;
        let rate = 16_000; // 100 ms budget = 1600 frames
        let block = 160;
        let mut b = InputBacklog::new(ch, rate, 100.0, 0);
        assert_eq!(b.budget_frames(), 1_600);
        b.push(&numbered(0, 1_600, ch));
        assert_eq!(b.enforce(block), None);
        assert_eq!(b.frames(), 1_600);
        // One frame over: drop to max(block, budget / 2) = 800 frames, the newest ones.
        b.push(&numbered(1_600, 1, ch));
        let ev = b.enforce(block).unwrap();
        assert_eq!(ev, DropEvent { frames: 801, overflow_frames: 0 });
        assert_eq!(b.frames(), 800);
        assert_eq!(b.dropped_frames(), 801);
        assert_accounting(&b);
        // Whole frames only: every frame still has equal channels, and past the splice the kept
        // frames are exactly 801..=1600 in order.
        let mut out = Vec::new();
        assert!(b.take_block(&mut out, 800));
        assert_eq!(out.len(), 800 * ch);
        for (i, f) in out.chunks_exact(ch).enumerate() {
            assert_eq!(f[0], f[1], "frame {i} split");
            if i >= b.fade_frames {
                assert_eq!(f[0], (801 + i) as f32);
            }
        }
        assert_eq!(b.consumed_frames(), 800);
        assert_accounting(&b);
        // Blocks come out whole, oldest first; a short backlog yields nothing.
        b.push(&numbered(5_000, 100, ch));
        let before = out.clone();
        assert!(!b.take_block(&mut out, 160));
        assert_eq!(out, before);
        assert_accounting(&b);
    }

    /// The drop is a splice: the audio right after what was already converted crossfades into
    /// the kept audio, so a continuous tone stays continuous (no step at the join).
    #[test]
    fn budget_splice_is_continuous() {
        let rate = 48_000;
        let mut b = InputBacklog::new(1, rate, 100.0, 0);
        // A slow 50 Hz sine: across a jump of `drop` frames the value changes, but the splice
        // moves from one to the other smoothly.
        let sine: Vec<f32> = (0..20_000).map(|i| (2.0 * std::f32::consts::PI * 50.0 * i as f32 / rate as f32).sin()).collect();
        b.push(&sine);
        let ev = b.enforce(480).unwrap();
        let mut out = Vec::new();
        assert!(b.take_block(&mut out, 480));
        // The first kept sample starts (almost) where the converted audio ended: frame 0 of the
        // dropped region.
        assert!((out[0] - sine[0]).abs() < 0.01, "{} vs {}", out[0], sine[0]);
        // No step anywhere inside the splice larger than the sine's own slope allows (plus a bit).
        let max_step = 2.0 * std::f32::consts::PI * 50.0 / rate as f32;
        for w in out.windows(2) {
            assert!((w[1] - w[0]).abs() < 12.0 * max_step, "step {} at the splice", (w[1] - w[0]).abs());
        }
        // After the fade the kept audio is the newest audio, untouched.
        let first_kept = ev.frames;
        let fade = b.fade_frames;
        assert_eq!(out[fade], sine[first_kept + fade]);
    }

    /// Simulated capture at real time with a worker that converts one block per callback period
    /// and stalls for 3 s: the backlog never exceeds budget + one block after an enforce, is back
    /// within budget on the first wake after the stall, and every frame is accounted.
    #[test]
    fn backlog_stays_within_budget_plus_one_block_under_an_injected_stall() {
        let rate = 48_000;
        let ch = 2;
        let block = 512;
        let period = 480; // 10 ms WASAPI capture period
        let cap = capture_ring_capacity(rate, ch, block);
        let (mut prod, mut cons) = RingBuffer::<f32>::new(cap);
        let mut b = InputBacklog::new(ch, rate, 300.0, cap);
        let limit = b.budget_frames().max(block);
        let mut out = Vec::new();
        let mut frame_no = 0u32;
        let mut overflow = 0u64;
        let stall = 300..600; // periods 3.0 s .. 6.0 s: no wake at all
        let mut first_after_stall = None;
        for p in 0..1_000usize {
            let data = numbered(frame_no, period, ch);
            frame_no += period as u32;
            overflow += push_capture(&mut prod, &data, ch) as u64;
            if stall.contains(&p) {
                continue;
            }
            b.note_overflow(std::mem::take(&mut overflow));
            b.drain_from(&mut cons);
            b.enforce(block);
            let backlog = b.frames() + cons.slots() / ch;
            assert!(backlog <= limit + block, "period {p}: backlog {backlog} > {limit} + {block}");
            if p >= stall.end && first_after_stall.is_none() {
                first_after_stall = Some(backlog);
            }
            // The worker keeps up: it converts every whole block it has.
            while b.take_block(&mut out, block) {}
            assert_accounting(&b);
        }
        // Back within budget right away after the stall (not after draining 3 s of backlog).
        assert!(first_after_stall.unwrap() <= limit);
        // The stall's stale audio was dropped; nothing was lost to the ring (it holds 4 s).
        assert!(b.dropped_frames() > 3 * 48_000 - limit as u64);
        assert_eq!(b.captured_frames(), u64::from(frame_no));
    }

    /// A stall longer than the ring: the ring refuses the newest audio; that is counted as
    /// dropped, the (now older) backlog is discarded, and the next audio fades in.
    #[test]
    fn ring_overflow_counts_as_dropped_and_fades_back_in() {
        let ch = 1;
        let (mut prod, mut cons) = RingBuffer::<f32>::new(1_000);
        let mut b = InputBacklog::new(ch, 16_000, 100.0, 0);
        let lost = push_capture(&mut prod, &vec![0.5f32; 1_300], ch);
        assert_eq!(lost, 300);
        b.note_overflow(lost as u64);
        b.drain_from(&mut cons);
        assert_accounting(&b);
        // 1000 kept + 300 lost > 1600? No: 1300 <= 1600, so only the hole is faded.
        let ev = b.enforce(160).unwrap();
        assert_eq!(ev, DropEvent { frames: 0, overflow_frames: 300 });
        assert_eq!(b.frames(), 1_000);
        assert_accounting(&b);
        let mut out = Vec::new();
        assert!(b.take_block(&mut out, 1_000));
        // The audio before the hole fades out to (near) silence.
        assert!(out[999].abs() < 0.01 && out[0] == 0.5);
        // The audio after the hole fades in.
        push_capture(&mut prod, &vec![0.5f32; 400], ch);
        b.drain_from(&mut cons);
        assert!(b.take_block(&mut out, 400));
        assert!(out[0] < 0.01 && out[399] == 0.5);
        assert_accounting(&b);

        // Now far over budget including the hole: everything held is older than the hole.
        let mut b = InputBacklog::new(ch, 16_000, 100.0, 0);
        push_capture(&mut prod, &vec![0.25f32; 1_000], ch);
        b.drain_from(&mut cons);
        b.note_overflow(5_000);
        let ev = b.enforce(160).unwrap();
        assert_eq!(ev, DropEvent { frames: 1_000, overflow_frames: 5_000 });
        assert_eq!(b.frames(), 0);
        assert_eq!(b.dropped_frames(), 6_000);
        assert_accounting(&b);
    }

    /// After a stall longer than the ring, a capture callback can land at any point of a wake's
    /// intake. The hole it leaves is always placed right after the audio drained before it, and
    /// audio captured behind the hole is kept and faded in, never mistaken for the audio before
    /// it (the old order faded out the freshest audio and left the real hole unfaded).
    #[test]
    fn ingest_places_a_racing_overflow_hole_correctly() {
        let (mut prod, mut cons) = RingBuffer::<f32>::new(1_000);
        let overflow = AtomicU64::new(0);
        let mut b = InputBacklog::new(1, 16_000, 2_000.0, 0);
        let value = |n: u32| 1.0 + n as f32 / 10_000.0;
        let frames = |from: u32, n: u32| -> Vec<f32> { (from..from + n).map(value).collect() };
        // The ring is full (frames 0..1000) and the next callback (1000..1300) is refused.
        assert_eq!(push_capture(&mut prod, &frames(0, 1_000), 1), 0);
        overflow.fetch_add(push_capture(&mut prod, &frames(1_000, 300), 1) as u64, Ordering::Relaxed);
        // A callback racing the intake (1300..1700) finds the ring drained: nothing refused.
        b.ingest_with(&mut cons, &overflow, || {
            assert_eq!(push_capture(&mut prod, &frames(1_300, 400), 1), 0);
        });
        assert_eq!(overflow.load(Ordering::Relaxed), 0);
        assert_eq!(b.enforce(160), Some(DropEvent { frames: 0, overflow_frames: 300 }));
        assert_accounting(&b);
        let mut out = Vec::new();
        assert!(b.take_block(&mut out, 1_000));
        // Before the hole: untouched, then faded out into it.
        assert_eq!(out[0], value(0));
        assert_eq!(out[900], value(900));
        assert!(out[999].abs() < 0.01, "{}", out[999]);
        // Behind the hole: the racing callback's audio, faded in, then untouched.
        b.ingest(&mut cons, &overflow);
        assert_eq!(b.enforce(160), None);
        assert!(b.take_block(&mut out, 400));
        assert!(out[0].abs() < 0.01, "{}", out[0]);
        assert_eq!(out[399], value(1_699));
        assert_accounting(&b);
        assert_eq!(b.captured_frames(), 1_700);
        assert_eq!(b.dropped_frames(), 300);
    }

    /// A voice that converts a long block per model call needs a budget that holds a call's worth of capture: a stored
    /// budget under it is raised (and nothing else is touched), so a call that finishes inside its block drops nothing.
    #[test]
    fn the_budget_is_raised_to_what_a_model_call_needs() {
        let chunk = 512.0 * 1000.0 / 48_000.0;
        // No fixed block, or nonsense: the stored (clamped) budget.
        assert_eq!(effective_latency_budget_ms(300.0, None, chunk), 300.0);
        assert_eq!(effective_latency_budget_ms(50.0, None, chunk), MIN_LATENCY_BUDGET_MS);
        assert_eq!(effective_latency_budget_ms(300.0, Some(0.0), chunk), 300.0);
        assert_eq!(effective_latency_budget_ms(300.0, Some(f64::NAN), chunk), 300.0);
        assert_eq!(effective_latency_budget_ms(f64::NAN, Some(120.0), chunk), DEFAULT_LATENCY_BUDGET_MS);
        // The default covers every block the engine plans (200 ms at most).
        for block in [20.0, 40.0, 120.0, 160.0, 200.0] {
            assert_eq!(effective_latency_budget_ms(DEFAULT_LATENCY_BUDGET_MS, Some(block), chunk), DEFAULT_LATENCY_BUDGET_MS, "{block} ms");
        }
        // A budget under the block plus two device chunks is raised to it; a larger one is kept.
        assert!((effective_latency_budget_ms(100.0, Some(120.0), chunk) - (120.0 + 2.0 * chunk)).abs() < 1e-9);
        assert!((effective_latency_budget_ms(150.0, Some(200.0), chunk) - (200.0 + 2.0 * chunk)).abs() < 1e-9);
        assert_eq!(effective_latency_budget_ms(250.0, Some(120.0), chunk), 250.0);
        assert_eq!(effective_latency_budget_ms(150.0, Some(120.0), chunk), 150.0);

        // The effect: a 100 ms budget against 120 ms blocks. The call takes the whole block (real time), so a block's
        // worth of capture piles up and the chunk that arrived with it. The stored budget drops most of it; the
        // budget in force drops none.
        let rate = 48_000;
        let waiting = ms_to_frames(120.0 + chunk, rate);
        let block_frames = ms_to_frames(chunk, rate);
        let mut stored = InputBacklog::new(1, rate, 100.0, 0);
        stored.push(&vec![0.1; waiting]);
        let dropped = stored.enforce(block_frames).expect("a 100 ms budget drops after a 120 ms call").frames;
        assert!(dropped > waiting / 3, "{dropped} of {waiting} frames");
        let mut raised = InputBacklog::new(1, rate, effective_latency_budget_ms(100.0, Some(120.0), chunk), 0);
        raised.push(&vec![0.1; waiting]);
        assert_eq!(raised.enforce(block_frames), None, "nothing is dropped inside the raised budget");
        // A call twice as long as its block (RTF 2) still drops, as it should.
        let mut behind = InputBacklog::new(1, rate, effective_latency_budget_ms(100.0, Some(120.0), chunk), 0);
        behind.push(&vec![0.1; ms_to_frames(240.0 + chunk, rate)]);
        assert!(behind.enforce(block_frames).is_some());
    }

    /// A block longer than the budget must still be convertible (the budget never starves it).
    #[test]
    fn block_longer_than_budget_is_not_starved() {
        let mut b = InputBacklog::new(1, 48_000, 100.0, 0); // 4800 frames
        let block = 8_192;
        b.push(&vec![0.1; 8_000]);
        assert_eq!(b.enforce(block), None);
        b.push(&vec![0.1; 192]);
        assert_eq!(b.enforce(block), None);
        let mut out = Vec::new();
        assert!(b.take_block(&mut out, block));
        assert_accounting(&b);
    }

    /// The ring only ever takes whole frames; what does not fit is reported as lost.
    #[test]
    fn capture_push_takes_whole_frames_only() {
        let (mut prod, mut cons) = RingBuffer::<f32>::new(64);
        assert_eq!(push_capture(&mut prod, &[0.5f32, -0.5, 0.25, 1.0], 2), 0);
        let got: Vec<f32> = std::iter::from_fn(|| cons.pop().ok()).collect();
        assert_eq!(got, vec![0.5, -0.5, 0.25, 1.0]);
        // A nearly full ring takes only whole frames and reports the rest as lost.
        let (mut prod, cons) = RingBuffer::<f32>::new(5);
        assert_eq!(push_capture(&mut prod, &[0.1f32; 8], 2), 2);
        assert_eq!(cons.slots(), 4);
    }

    /// Draining a wrapped ring whose split falls inside a frame still yields whole frames, and
    /// the fade-in after a hole walks across the seam.
    #[test]
    fn drain_handles_frames_split_by_the_ring_wrap() {
        let ch = 3;
        let (mut prod, mut cons) = RingBuffer::<f32>::new(10);
        let mut b = InputBacklog::new(ch, 16_000, 100.0, 0);
        // Advance the ring's read position by 2 samples so later frames straddle the wrap.
        push_capture(&mut prod, &[9.0f32; 3], ch);
        b.drain_from(&mut cons);
        let mut out = Vec::new();
        assert!(b.take_block(&mut out, 1));
        for round in 0..20u32 {
            let data = numbered(round * 3, 3, ch);
            assert_eq!(push_capture(&mut prod, &data, ch), 0);
            b.drain_from(&mut cons);
            assert!(b.take_block(&mut out, 3));
            assert_eq!(out, data, "round {round}");
        }
    }
}
