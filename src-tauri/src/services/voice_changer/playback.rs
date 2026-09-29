//! Output side of the voice changer: the worker-owned queue ([`OutputPlayback`]) that hands whole
//! frames to a lock-free ring, the real-time playout ([`RingPlayout`]), and the interleaved
//! channel adapter used by the monitor and aux paths (sample-rate conversion lives in
//! `resample.rs` and `aux_feed.rs`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

/// Extra ring top-up (frames per channel) beyond one device callback + one chunk, absorbing
/// scheduling jitter between the worker tick and the device callback.
const RING_TOPUP_MARGIN_FRAMES: usize = 64;

/// Floor of the output and monitor rings (interleaved samples). The ring must hold its top-up
/// target: prefill (or one device callback) + one chunk + one capture callback + margin. With a
/// 16384 floor, 48 kHz stereo, a 35 ms prefill and 512-frame chunks left room for capture
/// callbacks of only ~124 ms; 32768 covers ~290 ms.
pub(super) const OUT_RING_MIN_SAMPLES: usize = 32_768;
pub(super) const OUT_RING_MAX_SAMPLES: usize = 262_144;

/// After an underrun, blend this many **frames** (per channel) from the held sample back to live
/// audio so playback resumes without a click.
const RECOVER_FRAMES: usize = 16;

pub(super) struct OutputPlayback {
    deque: VecDeque<f32>,
    prefill_target: usize,
    max_queue_samples: usize,
    prefilled: bool,
    truncate_silent_streak: u32,
    /// Length of the most recently pushed chunk (ring top-up allowance beyond prefill).
    last_chunk_len: usize,
    /// Interleaved channel count: trims and ring hand-off move whole frames only, so the
    /// device callback never sees a split frame (which would swap channels).
    ch: usize,
    /// Samples requested by the device's latest output callback (written by [`RingPlayout`];
    /// 0 before the first callback).
    callback_len: Arc<AtomicUsize>,
    /// Capture-side burst: interleaved samples of the latest input callback (0 before the first,
    /// or when not wired). The worker produces nothing between two capture callbacks, so the
    /// ring must also bridge one capture period.
    in_burst_len: Arc<AtomicUsize>,
    /// Converts `in_burst_len` into this output's samples: input channels, and the output/input
    /// sample rates (`r_out / r_in`).
    in_burst_ch: usize,
    in_burst_r_in: u64,
    in_burst_r_out: u64,
}

fn rms_interleaved(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|x| x * x).sum();
    (sum / samples.len() as f32).sqrt()
}

impl OutputPlayback {
    pub(super) fn new(
        prefill_target: usize,
        max_queue_samples: usize,
        ch_out: usize,
        callback_len: Arc<AtomicUsize>,
    ) -> Self {
        Self {
            deque: VecDeque::with_capacity(max_queue_samples.max(prefill_target) + 8192),
            prefill_target,
            max_queue_samples: max_queue_samples.max(prefill_target + 1),
            prefilled: false,
            truncate_silent_streak: 0,
            last_chunk_len: 0,
            ch: ch_out.max(1),
            callback_len,
            in_burst_len: Arc::new(AtomicUsize::new(0)),
            in_burst_ch: 1,
            in_burst_r_in: 1,
            in_burst_r_out: 1,
        }
    }

    /// Wire the capture callback length (`in_burst_len`, interleaved samples at `ch_in` / `r_in`)
    /// into the ring target of this output (`r_out`).
    pub(super) fn with_capture_burst(
        mut self,
        in_burst_len: Arc<AtomicUsize>,
        ch_in: usize,
        r_in: u32,
        r_out: u32,
    ) -> Self {
        self.in_burst_len = in_burst_len;
        self.in_burst_ch = ch_in.max(1);
        self.in_burst_r_in = u64::from(r_in.max(1));
        self.in_burst_r_out = u64::from(r_out.max(1));
        self
    }

    pub(super) fn queue_len(&self) -> usize {
        self.deque.len()
    }

    /// Latest capture callback converted to this output's interleaved samples (whole frames,
    /// rounded up).
    fn in_burst_samples(&self) -> usize {
        let frames_in = (self.in_burst_len.load(Ordering::Relaxed) / self.in_burst_ch) as u64;
        let frames_out = (frames_in * self.in_burst_r_out).div_ceil(self.in_burst_r_in);
        (frames_out as usize).saturating_mul(self.ch)
    }

    /// Level at which the prefill gate opens: the prefill, or one full device callback when that
    /// is longer (a gate below one callback would underrun the very first period).
    fn gate_target(&self) -> usize {
        self.prefill_target.max(self.callback_len.load(Ordering::Relaxed))
    }

    /// Samples the ring should hold after a top-up: one full device callback (which can be
    /// longer than the prefill, e.g. a 10 ms WASAPI period vs. a 5 ms prefill) or the prefill,
    /// whichever is larger, plus a small jitter margin, plus the longest stretch with no top-up.
    /// Top-ups happen only when a chunk is produced, and chunks are produced only when capture
    /// delivers: the gap between two top-ups is below one chunk plus one capture callback (e.g.
    /// a 20 ms capture period feeding 3.3 ms chunks, or a capture period just short of a chunk).
    fn ring_target(&self) -> usize {
        self.gate_target()
            .max(1)
            .saturating_add(self.last_chunk_len)
            .saturating_add(self.in_burst_samples())
            .saturating_add(RING_TOPUP_MARGIN_FRAMES * self.ch)
    }

    /// Queue cap over deque + ring. Never below the ring target: a cap smaller than one device
    /// callback plus a chunk plus one capture callback would otherwise drop audio on every tick.
    fn queue_cap(&self) -> usize {
        self.max_queue_samples.max(self.ring_target())
    }

    /// Drop at least `n` samples from the front of the deque, rounded up to whole frames and
    /// capped at the deque length (the ring part cannot be trimmed from here).
    fn drop_front(&mut self, n: usize) {
        let ch = self.ch;
        let n = (n.div_ceil(ch) * ch).min(self.deque.len());
        if n > 0 {
            self.deque.drain(0..n);
        }
    }

    /// Apply new prefill/max targets from hot params without restarting cpal (may reset prefill gating).
    /// `ring_used` = samples already handed to the lock-free output ring (still queued for playback).
    pub(super) fn update_limits(&mut self, prefill_target: usize, max_queue_samples: usize, ring_used: usize) {
        let max_queue_samples = max_queue_samples.max(prefill_target.saturating_add(1));
        if self.prefill_target == prefill_target && self.max_queue_samples == max_queue_samples {
            return;
        }
        self.prefill_target = prefill_target;
        self.max_queue_samples = max_queue_samples;
        self.drop_front((self.deque.len() + ring_used).saturating_sub(self.queue_cap()));
        if self.deque.len() + ring_used < self.prefill_target {
            self.prefilled = false;
        }
    }

    pub(super) fn push_interleaved(
        &mut self,
        samples: &[f32],
        truncate_enabled: bool,
        truncate_rms: f32,
        truncate_chunks: u32,
        ring_used: usize,
    ) {
        self.last_chunk_len = samples.len();
        self.deque.extend(samples.iter().copied());
        // The queue cap covers deque + ring; only the deque part can be trimmed here.
        self.drop_front((self.deque.len() + ring_used).saturating_sub(self.queue_cap()));

        if truncate_enabled && truncate_chunks > 0 && !samples.is_empty() {
            let rms = rms_interleaved(samples);
            if rms < truncate_rms {
                self.truncate_silent_streak = self.truncate_silent_streak.saturating_add(1);
            } else {
                self.truncate_silent_streak = 0;
            }
            if self.truncate_silent_streak >= truncate_chunks {
                self.drop_front((self.deque.len() + ring_used).saturating_sub(self.prefill_target));
                self.truncate_silent_streak = 0;
            }
        } else {
            self.truncate_silent_streak = 0;
        }
    }

    /// Hand whole frames to the lock-free output ring once the prefill gate opens. The ring is
    /// only topped up to [`Self::ring_target`], so any backlog stays in the deque where the
    /// queue cap and silence truncation can still trim it.
    pub(super) fn flush_to_ring(&mut self, producer: &mut rtrb::Producer<f32>, ring_used: usize) {
        if !self.prefilled {
            if self.deque.len() + ring_used >= self.gate_target() {
                self.prefilled = true;
            } else {
                return;
            }
        }
        let ch = self.ch;
        let budget = self.ring_target().saturating_sub(ring_used);
        let n = (budget / ch)
            .min(self.deque.len() / ch)
            .min(producer.slots() / ch)
            * ch;
        if n == 0 {
            return;
        }
        // One commit for the whole run of frames: the consumer never observes a partial frame.
        if let Ok(chunk) = producer.write_chunk_uninit(n) {
            chunk.fill_from_iter(self.deque.drain(..n));
        }
    }
}

/// Real-time side of an output device: pulls whole frames from the lock-free ring, holds the
/// last frame on underrun and ramps back to live audio after a starve. All state is
/// closure-local (no locks, no allocation per callback).
pub(super) struct RingPlayout {
    consumer: rtrb::Consumer<f32>,
    /// Last emitted frame (one value per channel).
    hold: Vec<f32>,
    starving: bool,
    /// Frames left in the post-underrun recovery ramp.
    recover_left: usize,
    /// Length (samples) of the latest callback, read by the worker's ring top-up.
    callback_len: Arc<AtomicUsize>,
    /// Callbacks that ran (partly) dry after playback first began (the prefill wait before the
    /// first audio is not an underrun).
    underruns: Option<Arc<AtomicU64>>,
    played_any: bool,
}

impl RingPlayout {
    pub(super) fn new(consumer: rtrb::Consumer<f32>, ch: usize, callback_len: Arc<AtomicUsize>) -> Self {
        Self {
            consumer,
            hold: vec![0.0; ch.max(1)],
            // Start "starved" so the first audio after prefill fades in from silence.
            starving: true,
            recover_left: 0,
            callback_len,
            underruns: None,
            played_any: false,
        }
    }

    /// Count underruns into `counter` (the main output's `underruns` metric).
    pub(super) fn with_underrun_counter(mut self, counter: Arc<AtomicU64>) -> Self {
        self.underruns = Some(counter);
        self
    }

    pub(super) fn fill(&mut self, data: &mut [f32]) {
        self.callback_len.store(data.len(), Ordering::Relaxed);
        let ch = self.hold.len();
        let avail = (self.consumer.slots() / ch).min(data.len() / ch);
        let mut filled = 0usize;
        if avail > 0 {
            if let Ok(chunk) = self.consumer.read_chunk(avail * ch) {
                let (a, b) = chunk.as_slices();
                let mut t = 1.0f32;
                for (i, &v) in a.iter().chain(b.iter()).enumerate() {
                    let c = i % ch;
                    if c == 0 {
                        if self.starving {
                            self.starving = false;
                            self.recover_left = RECOVER_FRAMES;
                        }
                        t = if self.recover_left > 0 {
                            let done = RECOVER_FRAMES - self.recover_left;
                            self.recover_left -= 1;
                            (done + 1) as f32 / RECOVER_FRAMES as f32
                        } else {
                            1.0
                        };
                    }
                    let out = if t < 1.0 {
                        self.hold[c] + (v - self.hold[c]) * t
                    } else {
                        v
                    };
                    data[i] = out;
                    self.hold[c] = out;
                }
                chunk.commit_all();
                filled = avail * ch;
                self.played_any = true;
            }
        }
        if filled < data.len() {
            if self.played_any {
                if let Some(c) = &self.underruns {
                    c.fetch_add(1, Ordering::Relaxed);
                }
            }
            self.starving = true;
            for (i, s) in data.iter_mut().enumerate().skip(filled) {
                *s = self.hold[i % ch];
            }
        }
    }
}

pub(super) fn adapt_channels_interleaved_into(data: &[f32], from_ch: usize, to_ch: usize, out: &mut Vec<f32>) {
    out.clear();
    if from_ch == 0 || to_ch == 0 || data.is_empty() || !data.len().is_multiple_of(from_ch) {
        return;
    }
    let frames = data.len() / from_ch;
    out.reserve(frames * to_ch);
    for f in 0..frames {
        let base = f * from_ch;
        let frame = &data[base..base + from_ch];
        match (from_ch, to_ch) {
            (a, b) if a == b => out.extend_from_slice(frame),
            (_, 1) => {
                let s: f32 = frame.iter().sum::<f32>() / from_ch as f32;
                out.push(s);
            }
            (1, _) => {
                for _ in 0..to_ch {
                    out.push(frame[0]);
                }
            }
            _ => {
                for c in 0..to_ch {
                    out.push(frame[c % from_ch]);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{adapt_channels_interleaved_into, OutputPlayback, RingPlayout, RECOVER_FRAMES};
    use rtrb::RingBuffer;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    const RING_CAP: usize = 16_384;

    fn ring_used(p: &rtrb::Producer<f32>) -> usize {
        RING_CAP - p.slots()
    }

    /// 5 ms prefill + 160-frame chunks vs. a 10 ms (960-sample stereo) WASAPI callback: the
    /// ring must cover a whole callback, so after start-up no callback runs dry.
    #[test]
    fn ring_top_up_covers_long_device_callback() {
        let ch = 2usize;
        let prefill = 240 * ch; // 5 ms @ 48 kHz
        let max_queue = 480 * ch; // 10 ms
        let chunk = vec![0.25f32; 160 * ch];
        let cb_len = Arc::new(AtomicUsize::new(0));
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(prefill, max_queue, ch, cb_len.clone());
        let mut playout = RingPlayout::new(cons, ch, cb_len.clone());
        let mut data = vec![0.0f32; 480 * ch];
        let mut underrun_periods = 0usize;
        for period in 0..300 {
            // One 10 ms input burst = three 160-frame chunks, then the output callback.
            for _ in 0..3 {
                let used = ring_used(&prod);
                pb.push_interleaved(&chunk, false, 0.0, 0, used);
                let used = ring_used(&prod);
                pb.flush_to_ring(&mut prod, used);
            }
            playout.fill(&mut data);
            if period > 0 && playout.starving {
                underrun_periods += 1;
            }
        }
        assert_eq!(cb_len.load(Ordering::Relaxed), 480 * ch);
        assert_eq!(underrun_periods, 0);
        // Nothing was dropped: produced - consumed stays bounded (= what is still queued).
        assert!(pb.queue_len() + ring_used(&prod) < 2 * 480 * ch);
    }

    /// A 20 ms capture callback (six 160-frame chunks back to back) against 10 ms output
    /// callbacks: nothing tops the ring up between two capture bursts, so the ring must bridge a
    /// whole capture period. Without the capture-burst term the second callback of every period
    /// ran dry however much backlog sat in the deque.
    #[test]
    fn ring_top_up_bridges_long_capture_bursts() {
        let ch = 2usize;
        let prefill = 240 * ch; // 5 ms @ 48 kHz
        let max_queue = 480 * ch; // 10 ms
        let chunk = vec![0.25f32; 160 * ch];
        let cb_len = Arc::new(AtomicUsize::new(0));
        let burst = Arc::new(AtomicUsize::new(0));
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(prefill, max_queue, ch, cb_len.clone())
            .with_capture_burst(burst.clone(), 1, 48_000, 48_000);
        let mut playout = RingPlayout::new(cons, ch, cb_len.clone());
        let mut data = vec![0.0f32; 480 * ch];
        let mut underruns = 0usize;
        for period in 0..300 {
            // Mono capture: 960 interleaved samples = 960 frames = 960 stereo output frames.
            burst.store(960, Ordering::Relaxed);
            for _ in 0..6 {
                let used = ring_used(&prod);
                pb.push_interleaved(&chunk, false, 0.0, 0, used);
                let used = ring_used(&prod);
                pb.flush_to_ring(&mut prod, used);
            }
            for _ in 0..2 {
                playout.fill(&mut data);
                if period > 0 && playout.starving {
                    underruns += 1;
                }
            }
        }
        assert_eq!(pb.in_burst_samples(), 960 * ch);
        assert_eq!(underruns, 0);
        assert!(pb.queue_len() + ring_used(&prod) < 3 * 960 * ch);
    }

    /// The capture burst is converted to the output's rate and channel count.
    #[test]
    fn capture_burst_converts_to_output_samples() {
        let burst = Arc::new(AtomicUsize::new(441 * 2)); // 10 ms stereo @ 44.1 kHz
        let pb = OutputPlayback::new(0, 0, 1, Arc::new(AtomicUsize::new(0)))
            .with_capture_burst(burst, 2, 44_100, 48_000);
        assert_eq!(pb.in_burst_samples(), 480); // 10 ms mono @ 48 kHz
        let unwired = OutputPlayback::new(0, 0, 2, Arc::new(AtomicUsize::new(0)));
        assert_eq!(unwired.in_burst_samples(), 0);
    }

    /// The prefill gate opens at one full device callback when that exceeds the prefill.
    #[test]
    fn prefill_gate_waits_for_one_device_callback() {
        let ch = 2usize;
        let cb_len = Arc::new(AtomicUsize::new(480 * ch)); // 10 ms callback already seen
        let (mut prod, _cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(240 * ch, 960 * ch, ch, cb_len);
        let chunk = vec![0.1f32; 160 * ch];
        for _ in 0..2 {
            pb.push_interleaved(&chunk, false, 0.0, 0, 0);
            pb.flush_to_ring(&mut prod, 0);
            // 160 / 320 frames: past the 240-frame prefill at the second push, still gated.
            assert_eq!(ring_used(&prod), 0);
        }
        pb.push_interleaved(&chunk, false, 0.0, 0, 0);
        pb.flush_to_ring(&mut prod, 0);
        assert_eq!(ring_used(&prod), 480 * ch);
    }

    #[test]
    fn ring_hand_off_and_trims_keep_whole_frames() {
        let ch = 2usize;
        let cb_len = Arc::new(AtomicUsize::new(0));
        let (mut prod, _cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(4 * ch, 8 * ch, ch, cb_len);
        let chunk: Vec<f32> = (0..10 * ch).map(|i| i as f32).collect();
        pb.push_interleaved(&chunk, false, 0.0, 0, 0);
        pb.drop_front(3);
        assert_eq!(pb.queue_len() % ch, 0);
        pb.flush_to_ring(&mut prod, 0);
        assert_eq!(ring_used(&prod) % ch, 0);
        assert_eq!(pb.queue_len() % ch, 0);
    }

    #[test]
    fn playout_ramps_back_after_underrun_per_channel() {
        let ch = 2usize;
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut playout = RingPlayout::new(cons, ch, Arc::new(AtomicUsize::new(0)));
        // Starved callback: holds silence.
        let mut data = vec![1.0f32; 8 * ch];
        playout.fill(&mut data);
        assert!(data.iter().all(|&v| v == 0.0));
        assert!(playout.starving);
        // Live audio L = +1, R = -1 resumes through the ramp, channel by channel.
        for _ in 0..(2 * RECOVER_FRAMES) {
            prod.push(1.0).unwrap();
            prod.push(-1.0).unwrap();
        }
        let mut data = vec![0.0f32; 2 * RECOVER_FRAMES * ch];
        playout.fill(&mut data);
        assert!(!playout.starving);
        let first_l = data[0];
        assert!((first_l - 1.0 / RECOVER_FRAMES as f32).abs() < 1e-6);
        assert!((data[1] + first_l).abs() < 1e-6);
        for f in 1..RECOVER_FRAMES {
            assert!(data[f * ch] >= data[(f - 1) * ch]);
            assert!(data[f * ch] > 0.0 && data[f * ch + 1] < 0.0);
        }
        // Fully recovered at the end of the ramp and afterwards.
        for f in (RECOVER_FRAMES - 1)..(2 * RECOVER_FRAMES) {
            assert_eq!(data[f * ch], 1.0);
            assert_eq!(data[f * ch + 1], -1.0);
        }
    }

    #[test]
    fn device_resample_ratio_standard() {
        let r_in = 44100u32;
        let r_out = 48000u32;
        let ratio = r_out as f64 / r_in as f64;
        assert!((ratio - 48000.0 / 44100.0).abs() < 1e-6);
    }

    /// Dry callbacks are counted once playback has begun, not while the prefill is building.
    #[test]
    fn underruns_count_only_after_playback_began() {
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
        let mut playout = RingPlayout::new(cons, 1, Arc::new(AtomicUsize::new(0))).with_underrun_counter(counter.clone());
        let mut data = vec![0.0f32; 64];
        playout.fill(&mut data);
        playout.fill(&mut data);
        assert_eq!(counter.load(Ordering::Relaxed), 0, "prefill wait");
        for _ in 0..64 {
            prod.push(0.5).unwrap();
        }
        playout.fill(&mut data);
        assert_eq!(counter.load(Ordering::Relaxed), 0, "a full callback");
        for _ in 0..10 {
            prod.push(0.5).unwrap();
        }
        playout.fill(&mut data);
        playout.fill(&mut data);
        assert_eq!(counter.load(Ordering::Relaxed), 2, "a short and an empty callback");
    }

    #[test]
    fn adapt_channels_mono_to_stereo() {
        let mut o = Vec::new();
        adapt_channels_interleaved_into(&[0.5f32], 1, 2, &mut o);
        assert_eq!(o, vec![0.5, 0.5]);
    }

    #[test]
    fn adapt_channels_stereo_to_mono() {
        let mut o = Vec::new();
        adapt_channels_interleaved_into(&[0.2f32, 0.4f32], 2, 1, &mut o);
        assert!((o[0] - 0.3).abs() < 1e-6);
    }
}
