//! Adaptive output cushion (params `output_cushion = adaptive`, P3B-H): sizes the standing level of the
//! output queue from the lateness the stream actually shows, instead of a fixed 35 ms prefill.
//!
//! Pure and allocation-light: it holds no queue and no clock. [`OutputPlayback`](super::playback::OutputPlayback)
//! feeds it one [`Observation`] per worker push (the queue level before the push, how much audio the
//! queue discarded so far, the device's consumption in samples as the clock) and acts on what it
//! returns, so the same code runs in the app and in the benchmark's `OutputSide`.
//!
//! # What it measures
//! The queue (worker deque plus lock-free ring) only runs dry just before a push: the level falls with
//! the device and rises by a chunk at every push. So the safest summary of the stream's lateness is
//! the **valley**, the smallest level before a push over a long window. Trims and cuts shift every
//! later level, so levels are kept in *undiscarded* coordinates (`level + audio discarded so far`,
//! the supply minus the device's consumption) and shifted back by what has been discarded since:
//! the valley the history would have had at today's queue. Over a window of minutes it holds the
//! worst lateness of block-completing ticks, the render/capture phase sweeps of two device clocks
//! included.
//!
//! # What it does with it
//! - **Start.** The target (the valley it holds) is the user's prefill: nothing is known yet.
//! - **Shrink.** After [`MIN_HISTORY_MS`] of device time, and while the valley of the window exceeds
//!   the floor ([`Floor`]: the longer of one device callback and one capture burst, which is the
//!   phase two device clocks can sweep through, plus [`MARGIN_MS`], plus the window's own spread:
//!   how much deeper its worst dip is than its typical one) by more than [`SHED_HYSTERESIS_MS`], it
//!   sheds the excess, never more than the valley allows, so the horizon's worst dip would just
//!   touch the floor and never below. Audio is shed (a) in silence:
//!   the leading frames of a quiet chunk are dropped, which nobody hears, or (b) after
//!   [`SPEECH_NO_SILENCE_MS`] without any silence, as a [`SPEECH_SHED_MS`] splice (the playback's
//!   existing 5 ms crossfaded trim) at most once per [`SPEECH_SHED_EVERY_MS`]. It never re-gates:
//!   a re-gate while playing is a dropout.
//! - **Grow.** Only after an underrun, when the queue is empty anyway: the target rises toward the
//!   user's prefill (halfway there, at least [`GROW_STEP_MS`]), the playback re-gates at it (the
//!   queue refills before playing: free, it is already empty), the window is forgotten and no
//!   shrinking happens for [`COOLDOWN_MS`] (doubling with each underrun, up to [`COOLDOWN_MAX_MS`]).
//! - **Never above the user's value.** The prefill is the target's cap, so the adaptive latency is
//!   never more than the fixed one; a voice that needs a larger output buffer (`min_output_buffer_ms`)
//!   is its floor.
//!
//! # The target is a valley, not the standing level
//! The target is the level the stream's worst dip lands on (the valley after the shed). Between dips the
//! queue stands the depth of that dip above it. Whatever else steers the queue must therefore not use the
//! target as a *ceiling*: the playback's clock-drift trim keeps the user's prefill ([`AdaptiveCushion::cap`])
//! as its ceiling, as with the fixed cushion, and leaves everything below it to this controller. (A
//! drift ceiling keyed to the target trimmed the queue back to about one period over it on the first
//! dip-free window, so the next 25 ms late tick ran the device dry.)

use std::collections::VecDeque;
use std::fs::File;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// Device time observed before the first shed (ms): the window must have seen the stream.
pub(crate) const MIN_HISTORY_MS: f64 = 30_000.0;
/// Device time after the first observation that the window ignores (ms): a stream's first seconds
/// (cold caches, GPU clocks ramping) are not its steady lateness, and the cushion is still whole then.
pub(crate) const SETTLE_MS: f64 = 3_000.0;
/// The window the valley is taken over (ms): "never shrink below the maximum of the last N minutes".
pub(crate) const HORIZON_MS: f64 = 120_000.0;
/// Resolution of the window (ms).
const BUCKET_MS: f64 = 4_000.0;
/// Margin above the render / capture phase (ms): unseen lateness beyond the window.
pub(crate) const MARGIN_MS: f64 = 2.0;
/// The share of the window's spread (how much deeper its worst bucket is than its median one) the
/// floor adds: a stream with rare deep dips keeps more than one whose dips are all alike.
pub(crate) const SPREAD_FACTOR: f64 = 0.5;
/// `SIGIL_VC_CUSHION_SPREAD=<factor>` / `SIGIL_VC_CUSHION_MARGIN_MS=<ms>`: process overrides of
/// [`SPREAD_FACTOR`] and [`MARGIN_MS`] for sweeps (benchmarks); read when a controller is built.
pub(crate) const SPREAD_ENV: &str = "SIGIL_VC_CUSHION_SPREAD";
pub(crate) const MARGIN_ENV: &str = "SIGIL_VC_CUSHION_MARGIN_MS";

/// `SIGIL_VC_CUSHION_TRACE=<file>`: append one line per shed / underrun and a status line every
/// [`TRACE_STATUS_EVERY_MS`] of device time (benchmarks and field diagnosis; the headless benchmark has
/// no log file). Written from the worker thread, never from an audio callback.
pub(crate) const TRACE_ENV: &str = "SIGIL_VC_CUSHION_TRACE";
pub(crate) const TRACE_STATUS_EVERY_MS: f64 = 5_000.0;

static TRACE: OnceLock<Option<(Mutex<File>, Instant)>> = OnceLock::new();

/// Whether the trace file is on.
pub(crate) fn trace_on() -> bool {
    TRACE
        .get_or_init(|| {
            let path = std::env::var_os(TRACE_ENV)?;
            let file = std::fs::OpenOptions::new().create(true).append(true).open(path).ok()?;
            Some((Mutex::new(file), Instant::now()))
        })
        .is_some()
}

/// Append a line to the trace file (a no-op when it is off).
pub(crate) fn trace(line: std::fmt::Arguments<'_>) {
    if !trace_on() {
        return;
    }
    if let Some(Some((file, t0))) = TRACE.get() {
        if let Ok(mut f) = file.lock() {
            let _ = writeln!(f, "{:9.3} {line}", t0.elapsed().as_secs_f64());
        }
    }
}

fn env_f64(name: &str, default: f64, lo: f64, hi: f64) -> f64 {
    std::env::var(name)
        .ok()
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|x| x.is_finite())
        .map_or(default, |x| x.clamp(lo, hi))
}
/// `SIGIL_VC_CUSHION_HISTORY_MS=<ms>`: [`MIN_HISTORY_MS`] for sweeps (benchmarks).
pub(crate) const HISTORY_ENV: &str = "SIGIL_VC_CUSHION_HISTORY_MS";
/// After an underrun the floor never again drops below the cushion that failed plus this (ms): the
/// stream showed a lateness the window had not seen.
pub(crate) const LEARN_STEP_MS: f64 = 6.0;
/// The learned floor relaxes by [`LEARN_DECAY_MS`] per this much underrun-free device time (ms).
pub(crate) const LEARN_DECAY_EVERY_MS: f64 = 120_000.0;
pub(crate) const LEARN_DECAY_MS: f64 = 1.0;
/// Valley excess (ms) worth shedding.
pub(crate) const SHED_HYSTERESIS_MS: f64 = 2.5;
/// Quiet audio (ms) before a silent chunk's head may be cut.
pub(crate) const SILENCE_HOLD_MS: f64 = 120.0;
/// A splice shed's size (ms); the playback crossfades it over its 5 ms fade.
pub(crate) const SPEECH_SHED_MS: f64 = 2.0;
/// Fewest ms between two splice sheds.
pub(crate) const SPEECH_SHED_EVERY_MS: f64 = 1_500.0;
/// Audio without a silence (ms) before a splice shed is allowed (silence is always preferred).
pub(crate) const SPEECH_NO_SILENCE_MS: f64 = 6_000.0;
/// Least growth after an underrun (ms).
pub(crate) const GROW_STEP_MS: f64 = 6.0;
/// No shrinking after an underrun (ms), doubling per underrun up to the maximum.
pub(crate) const COOLDOWN_MS: f64 = 30_000.0;
pub(crate) const COOLDOWN_MAX_MS: f64 = 240_000.0;
/// RMS (linear) below which a chunk counts as silence whatever the user's truncation threshold.
pub(crate) const SILENCE_RMS_FLOOR: f32 = 0.001;

/// One worker push, as the playback sees it before queueing the chunk. Samples are interleaved at
/// the output's format.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Observation {
    /// Queue level (deque plus ring) before the push.
    pub level_before: usize,
    /// Samples the queue discarded so far (trims, cuts, drops), this controller's own sheds included.
    pub removed_total: u64,
    /// Samples the device consumed so far: the clock.
    pub consumed_total: u64,
    /// Samples of the pushed chunk.
    pub chunk_len: usize,
    /// The chunk is below the silence threshold.
    pub silent: bool,
    /// The device's latest callback length (0 until the first).
    pub callback_len: usize,
    /// The latest capture burst converted to this output's samples (0 when not wired).
    pub capture_burst: usize,
}

/// What the playback should shed from this push.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shed {
    None,
    /// Drop this many samples (whole frames are the caller's rounding) from the head of the quiet chunk.
    Silent(usize),
    /// Trim this many samples from the queue as a crossfaded splice.
    Splice(usize),
}

/// The floor of the target, with its parts (samples), for the metrics and the tests.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Floor {
    /// The longer of the device callback and the capture burst: the render / capture phase the two
    /// clocks can sweep through over minutes (the window may have seen only a favourable part).
    pub phase: usize,
    pub margin: usize,
    /// A share of how much deeper the window's worst dip is than its typical one (a jittery stream
    /// keeps more).
    pub spread: usize,
    /// A voice's own minimum (`min_output_buffer_ms`).
    pub voice: usize,
    /// What the stream's underruns taught: the cushion that failed plus [`LEARN_STEP_MS`], relaxing
    /// slowly while no underrun follows.
    pub learned: usize,
}

impl Floor {
    pub(crate) fn total(&self) -> usize {
        (self.phase + self.margin + self.spread).max(self.voice).max(self.learned)
    }
}

#[derive(Clone, Copy, Debug)]
struct Bucket {
    /// Clock at the bucket's start.
    start: u64,
    /// Smallest undiscarded level before a push in it.
    min_s: u64,
}

/// The controller. See the module docs.
#[derive(Debug)]
pub(crate) struct AdaptiveCushion {
    rate: u32,
    ch: usize,
    /// The user's prefill: the cap.
    cap: usize,
    /// A voice's own minimum (samples).
    voice_floor: usize,
    /// The valley it holds (samples).
    target: usize,
    floor: Floor,
    buckets: VecDeque<Bucket>,
    first_clock: Option<u64>,
    /// Quiet audio since the last loud chunk, and loud audio since the last quiet one (samples).
    silent_run: u64,
    since_silence: u64,
    last_splice: u64,
    cooldown_until: u64,
    cooldown_ms: f64,
    spread_factor: f64,
    margin_ms: f64,
    min_history_ms: f64,
    /// The learned floor as taught (before its decay) and the clock of the underrun that taught it.
    learned: usize,
    learned_at: u64,
    sheds: u64,
    shed_samples: u64,
    grows: u64,
}

impl AdaptiveCushion {
    pub(crate) fn new(rate: u32, ch: usize, cap: usize, voice_floor: usize) -> AdaptiveCushion {
        let mut c = AdaptiveCushion {
            rate: rate.max(1),
            ch: ch.max(1),
            cap,
            voice_floor,
            target: cap,
            floor: Floor::default(),
            buckets: VecDeque::new(),
            first_clock: None,
            silent_run: 0,
            since_silence: 0,
            last_splice: 0,
            cooldown_until: 0,
            cooldown_ms: COOLDOWN_MS,
            spread_factor: env_f64(SPREAD_ENV, SPREAD_FACTOR, 0.0, 4.0),
            margin_ms: env_f64(MARGIN_ENV, MARGIN_MS, 0.0, 30.0),
            min_history_ms: env_f64(HISTORY_ENV, MIN_HISTORY_MS, 1_000.0, 120_000.0),
            learned: 0,
            learned_at: 0,
            sheds: 0,
            shed_samples: 0,
            grows: 0,
        };
        c.floor.voice = voice_floor;
        c
    }

    fn samples(&self, ms: f64) -> u64 {
        (ms * f64::from(self.rate) / 1000.0).round() as u64 * self.ch as u64
    }

    /// The output's sample rate changed (the device moved): the constants follow.
    pub(crate) fn set_rate(&mut self, rate: u32) {
        self.rate = rate.max(1);
    }

    /// The user's prefill changed: the cap and the start over (the window knew another cushion).
    pub(crate) fn set_cap(&mut self, cap: usize) {
        if cap != self.cap {
            self.cap = cap;
            self.restart();
        }
    }

    /// A voice's own minimum output buffer (samples; 0 = none).
    pub(crate) fn set_voice_floor(&mut self, floor: usize) {
        if floor != self.voice_floor {
            self.voice_floor = floor;
            self.floor.voice = floor;
            self.target = self.target.max(floor.min(self.cap.max(floor)));
        }
    }

    /// Forget the window and return to the cap (a new stream, a new limit).
    pub(crate) fn restart(&mut self) {
        self.learned = 0;
        self.floor.learned = 0;
        self.target = self.cap.max(self.voice_floor);
        self.buckets.clear();
        self.first_clock = None;
        self.silent_run = 0;
        self.since_silence = 0;
        self.cooldown_ms = COOLDOWN_MS;
        self.cooldown_until = 0;
    }

    /// Tests: the history before the first shed (the unit tests run on a short one).
    #[cfg(test)]
    pub(crate) fn set_history_ms(&mut self, ms: f64) {
        self.min_history_ms = ms;
    }

    /// The learned floor now: what the last underrun taught, relaxed by the underrun-free time since.
    fn learned_now(&self, clock: u64) -> usize {
        if self.learned == 0 {
            return 0;
        }
        let every = self.samples(LEARN_DECAY_EVERY_MS).max(1);
        let relax = (clock.saturating_sub(self.learned_at) / every) as usize * self.samples(LEARN_DECAY_MS) as usize;
        self.learned.saturating_sub(relax)
    }

    /// Samples as ms at this stream's rate (logs, metrics).
    pub(crate) fn ms(&self, samples: usize) -> f64 {
        samples as f64 / self.ch as f64 / f64::from(self.rate) * 1000.0
    }

    /// The valley it holds now (samples): the gate the playback uses.
    pub(crate) fn target(&self) -> usize {
        self.target
    }

    /// The most the target may be (samples): the user's prefill, or the voice's own minimum when that
    /// is larger. The fixed cushion's gate level, so it is also the level the playback's clock-drift
    /// trim holds the queue to (see `OutputPlayback::drift_gate`).
    pub(crate) fn cap(&self) -> usize {
        self.cap.max(self.voice_floor)
    }

    #[cfg(test)]
    pub(crate) fn floor(&self) -> Floor {
        self.floor
    }

    /// One status line (ms): target, valley of the window, the floor and its parts, sheds, growths.
    pub(crate) fn status(&self, removed_total: u64) -> String {
        format!(
            "status target {:.1} valley {:.1} floor {:.1} (phase {:.1} margin {:.1} spread {:.1} voice {:.1} learned {:.1}) sheds {} ({:.1} ms) grows {} buckets {}",
            self.ms(self.target),
            self.valley(removed_total).map_or(-1.0, |v| self.ms(v)),
            self.ms(self.floor.total()),
            self.ms(self.floor.phase),
            self.ms(self.floor.margin),
            self.ms(self.floor.spread),
            self.ms(self.floor.voice),
            self.ms(self.floor.learned),
            self.sheds,
            self.ms(self.shed_samples as usize),
            self.grows,
            self.buckets.len()
        )
    }

    /// The floor as of the latest push (samples).
    pub(crate) fn floor_total(&self) -> usize {
        self.floor.total()
    }

    /// The part of the floor the stream's underruns taught (samples).
    pub(crate) fn learned_floor(&self) -> usize {
        self.floor.learned
    }

    pub(crate) fn sheds(&self) -> (u64, u64) {
        (self.sheds, self.shed_samples)
    }

    pub(crate) fn grows(&self) -> u64 {
        self.grows
    }

    /// The valley over the window in today's coordinates (`None` before the first observation).
    pub(crate) fn valley(&self, removed_total: u64) -> Option<usize> {
        self.buckets
            .iter()
            .map(|b| b.min_s)
            .min()
            .map(|m| m.saturating_sub(removed_total) as usize)
    }

    /// How much deeper the window's worst bucket valley is than its median one (trim-invariant: both
    /// shift together).
    fn spread(&self) -> usize {
        if self.buckets.len() < 2 {
            return 0;
        }
        let mut v: Vec<u64> = self.buckets.iter().map(|b| b.min_s).collect();
        v.sort_unstable();
        (v[v.len() / 2] - v[0]) as usize
    }

    fn note(&mut self, o: &Observation) {
        self.floor.phase = o.callback_len.max(o.capture_burst);
        self.floor.margin = self.samples(self.margin_ms) as usize;
        self.floor.voice = self.voice_floor;
        let clock = o.consumed_total;
        let first = *self.first_clock.get_or_insert(clock);
        let s = o.level_before as u64 + o.removed_total;
        let bucket = self.samples(BUCKET_MS).max(1);
        // The first seconds only start the clock.
        if clock.saturating_sub(first) >= self.samples(SETTLE_MS) {
            match self.buckets.back_mut() {
                Some(b) if clock < b.start + bucket => b.min_s = b.min_s.min(s),
                _ => self.buckets.push_back(Bucket { start: clock, min_s: s }),
            }
        }
        let horizon = self.samples(HORIZON_MS);
        while self.buckets.front().is_some_and(|b| b.start + bucket + horizon <= clock) {
            self.buckets.pop_front();
        }
        if o.silent {
            self.silent_run += o.chunk_len as u64;
            self.since_silence = 0;
        } else {
            self.silent_run = 0;
            self.since_silence += o.chunk_len as u64;
        }
    }

    /// One push: what to shed from it (module docs). Never sheds before the window is old enough,
    /// during a cooldown, or below the floor.
    pub(crate) fn observe(&mut self, o: &Observation) -> Shed {
        if o.chunk_len == 0 {
            return Shed::None;
        }
        self.note(o);
        self.floor.spread = (self.spread() as f64 * self.spread_factor) as usize;
        self.floor.learned = self.learned_now(o.consumed_total);
        let floor = self.floor.total();
        let first = self.first_clock.unwrap_or(o.consumed_total);
        if o.callback_len == 0
            || o.consumed_total < self.cooldown_until
            || o.consumed_total.saturating_sub(first) < self.samples(self.min_history_ms)
        {
            return Shed::None;
        }
        let Some(valley) = self.valley(o.removed_total) else { return Shed::None };
        let hysteresis = self.samples(SHED_HYSTERESIS_MS) as usize;
        if valley <= floor + hysteresis {
            // Nothing to shed. The target stays: it only moves when audio is actually shed or after an
            // underrun (an underrun re-gates at it, so lowering it here would refill a failed
            // cushion with less).
            return Shed::None;
        }
        let slack = valley - floor;
        let shed = if self.silent_run >= self.samples(SILENCE_HOLD_MS) {
            Shed::Silent(slack.min(o.chunk_len))
        } else if self.since_silence >= self.samples(SPEECH_NO_SILENCE_MS)
            && o.consumed_total.saturating_sub(self.last_splice) >= self.samples(SPEECH_SHED_EVERY_MS)
            && slack >= 2 * self.samples(SPEECH_SHED_MS) as usize
        {
            self.last_splice = o.consumed_total;
            Shed::Splice(self.samples(SPEECH_SHED_MS) as usize)
        } else {
            Shed::None
        };
        let dropped = match shed {
            Shed::Silent(n) | Shed::Splice(n) => n,
            Shed::None => 0,
        };
        if dropped > 0 {
            self.sheds += 1;
            self.shed_samples += dropped as u64;
            self.target = (valley - dropped).max(floor).min(self.target);
        }
        shed
    }

    /// The device ran dry while playing: grow the target toward the cap, forget the window and hold
    /// off shrinking. Returns the new target (the playback re-gates at it: the queue is empty).
    pub(crate) fn on_underrun(&mut self, consumed_total: u64) -> usize {
        let cap = self.cap.max(self.voice_floor);
        // The cushion that just failed, and a little more, is the least it may shrink to again.
        let learned = (self.target + self.samples(LEARN_STEP_MS) as usize).min(cap);
        self.learned = self.learned_now(consumed_total).max(learned);
        self.learned_at = consumed_total;
        self.floor.learned = self.learned;
        let step = (cap.saturating_sub(self.target) / 2).max(self.samples(GROW_STEP_MS) as usize);
        self.target = (self.target + step).min(cap).max(self.learned);
        self.grows += 1;
        self.buckets.clear();
        self.first_clock = None;
        self.silent_run = 0;
        self.since_silence = 0;
        self.cooldown_until = consumed_total + self.samples(self.cooldown_ms);
        self.cooldown_ms = (self.cooldown_ms * 2.0).min(COOLDOWN_MAX_MS);
        self.target
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const RATE: u32 = 48_000;
    const CH: usize = 2;
    const CB: usize = 480 * CH; // 10 ms device callbacks
    const BURST: usize = 480 * CH;

    fn ms(x: f64) -> usize {
        (x * f64::from(RATE) / 1000.0).round() as usize * CH
    }

    /// Pushes of 10.67 ms chunks every 10.67 ms of device time with a queue level that dips by
    /// `jitter(k)` samples below `base` before push `k`.
    struct Feed {
        c: AdaptiveCushion,
        clock: u64,
        removed: u64,
        k: u64,
        chunk: usize,
    }

    impl Feed {
        /// A feed with the 8 s history the unit tests use (the shipped one is [`MIN_HISTORY_MS`]).
        fn new(cap_ms: f64) -> Feed {
            let mut f = Feed::shipped(cap_ms);
            f.c.set_history_ms(8_000.0);
            f
        }

        fn shipped(cap_ms: f64) -> Feed {
            Feed { c: AdaptiveCushion::new(RATE, CH, ms(cap_ms), 0), clock: 0, removed: 0, k: 0, chunk: 512 * CH }
        }

        fn push(&mut self, level: usize, silent: bool) -> Shed {
            self.clock += self.chunk as u64;
            self.k += 1;
            let o = Observation {
                level_before: level,
                removed_total: self.removed,
                consumed_total: self.clock,
                chunk_len: self.chunk,
                silent,
                callback_len: CB,
                capture_burst: BURST,
            };
            let shed = self.c.observe(&o);
            if let Shed::Silent(n) | Shed::Splice(n) = shed {
                self.removed += n as u64;
            }
            shed
        }

        /// The queue level the stream would show now: `base` minus what was shed, minus a dip.
        fn level(&self, base: usize, dip: usize) -> usize {
            base.saturating_sub(self.removed as usize).saturating_sub(dip)
        }

        /// Run `ms` of audio at a steady base level with `dip_every` pushes dipping by `dip`.
        fn run(&mut self, run_ms: f64, base: usize, dip: usize, dip_every: u64, silent: impl Fn(u64) -> bool) -> (usize, usize) {
            let pushes = (run_ms * f64::from(RATE) / 1000.0 / 512.0) as u64;
            let (mut min_level, mut shed_total) = (usize::MAX, 0);
            for _ in 0..pushes {
                let d = if dip_every > 0 && self.k.is_multiple_of(dip_every) { dip } else { 0 };
                let level = self.level(base, d);
                min_level = min_level.min(level);
                let silent_now = silent(self.k);
                if let Shed::Silent(n) | Shed::Splice(n) = self.push(level, silent_now) {
                    shed_total += n;
                }
            }
            (min_level, shed_total)
        }
    }

    #[test]
    fn the_cushion_starts_at_the_users_prefill_and_waits_for_history() {
        let mut f = Feed::shipped(35.0);
        assert_eq!(f.c.target(), ms(35.0));
        // Twenty-nine seconds of quiet audio with a level far above the floor: nothing is shed yet.
        let (_, shed) = f.run(29_000.0, ms(46.0), 0, 0, |_| true);
        assert_eq!(shed, 0);
        assert_eq!(f.c.target(), ms(35.0));
        // Past the history the quiet chunks give the slack back, down to the floor and no further:
        // a 10 ms callback / capture burst + 2 ms, a steady stream having no spread.
        let (min_level, shed) = f.run(3_000.0, ms(46.0), 0, 0, |_| true);
        assert!(shed > 0);
        assert_eq!(f.c.floor().total(), ms(12.0));
        assert!(min_level >= ms(12.0), "level {min_level} vs floor {}", ms(12.0));
        assert!(f.c.target() <= ms(15.0) && f.c.target() >= ms(12.0), "{}", f.c.target());
    }

    #[test]
    fn the_first_seconds_do_not_pin_the_window() {
        // A bad start (a 25 ms dip in the first second: cold caches, GPU clocks ramping) followed by a
        // steady stream: the window never saw it, so the slack is the steady stream's.
        let mut f = Feed::new(35.0);
        f.run(500.0, ms(46.0), ms(25.0), 1, |_| false);
        let (min_level, shed) = f.run(15_000.0, ms(46.0), 0, 0, |_| true);
        assert!(shed > 0);
        assert!(min_level >= ms(12.0), "{min_level}");
        assert!(f.c.target() < ms(20.0), "{}", f.c.target());
    }

    #[test]
    fn it_never_sheds_below_what_the_worst_dip_of_the_window_needs() {
        let mut f = Feed::new(35.0);
        // A level of 46 ms with a 12 ms dip every 40th push (every window sees it): the valley is
        // 34 ms, the floor 12 ms, so 22 ms can go; after that the dips touch the floor and no more
        // is shed.
        let (min_level, _) = f.run(60_000.0, ms(46.0), ms(12.0), 40, |k| k % 25 < 20);
        assert!(min_level >= ms(12.0), "the worst dip reached {min_level} vs the floor {}", ms(12.0));
        let valley = f.c.valley(f.removed).unwrap();
        assert!(valley >= ms(12.0) && valley <= ms(12.0) + ms(2.5) + 512 * CH, "valley {valley}");
        // Further quiet audio sheds nothing more.
        let (_, shed) = f.run(10_000.0, ms(46.0), ms(12.0), 40, |_| true);
        assert_eq!(shed, 0);
    }

    #[test]
    fn a_deeper_dip_is_remembered_for_the_whole_window() {
        let mut f = Feed::new(35.0);
        // One run of 28 ms dips early: the valley is 18 ms, and the window's spread (28 ms between its
        // typical and its worst bucket) lifts the floor to 40 ms: nothing is shed while the window
        // remembers it; the slack opens when it leaves.
        f.run(5_000.0, ms(46.0), 0, 0, |_| false);
        f.run(60.0, ms(46.0), ms(28.0), 1, |_| false);
        let (_, shed) = f.run(100_000.0, ms(46.0), 0, 0, |_| true);
        assert_eq!(shed, 0, "the dip is inside the window");
        // Another 40 s: the dip leaves the 2 minute window.
        let (min_level, shed) = f.run(40_000.0, ms(46.0), 0, 0, |_| true);
        assert!(shed > 0, "the dip is forgotten after the horizon");
        assert!(min_level >= ms(12.0));
    }

    #[test]
    fn speech_without_a_pause_is_spliced_slowly_and_in_small_steps() {
        let mut f = Feed::new(35.0);
        let mut splices = 0;
        let mut biggest = 0;
        let mut at = Vec::new();
        let pushes = (30_000.0 * f64::from(RATE) / 1000.0 / 512.0) as u64;
        for _ in 0..pushes {
            let level = f.level(ms(46.0), 0);
            if let Shed::Splice(n) = f.push(level, false) {
                splices += 1;
                biggest = biggest.max(n);
                at.push(f.clock);
            }
        }
        assert!(splices >= 3, "{splices}");
        assert!(biggest <= ms(SPEECH_SHED_MS), "{biggest}");
        assert!(at[0] >= f.c.samples(8_000.0_f64.max(SPEECH_NO_SILENCE_MS)));
        assert!(at.windows(2).all(|w| w[1] - w[0] >= f.c.samples(SPEECH_SHED_EVERY_MS)), "rate limited");
        // No silent cuts happened.
        assert_eq!(f.c.sheds().0, splices as u64);
    }

    #[test]
    fn an_underrun_teaches_a_floor_that_relaxes_with_quiet_time() {
        let mut f = Feed::new(35.0);
        f.run(12_000.0, ms(46.0), 0, 0, |_| true);
        let shrunk = f.c.target();
        f.c.on_underrun(f.clock);
        let learned = f.c.learned_floor();
        assert_eq!(learned, (shrunk + ms(LEARN_STEP_MS)).min(ms(35.0)));
        // Past the cooldown the window may shed again, but never below what failed plus the step.
        f.removed = 0;
        f.run(40_000.0, ms(46.0), 0, 0, |_| true);
        f.run(30_000.0, ms(46.0), 0, 0, |_| true);
        assert!(f.c.target() >= learned.min(ms(35.0)), "{} vs learned {learned}", f.c.target());
        assert_eq!(f.c.learned_floor(), learned, "no quiet time has passed yet");
        // Four underrun-free minutes relax it by 1 ms per two minutes (LEARN_DECAY_MS per
        // LEARN_DECAY_EVERY_MS): 2 ms.
        f.run(240_000.0, ms(46.0), 0, 0, |_| true);
        assert_eq!(f.c.learned_floor(), learned.saturating_sub(ms(2.0)));
        // A restart (new stream, new limit) forgets it.
        f.c.restart();
        assert_eq!(f.c.learned_floor(), 0);
    }

    #[test]
    fn an_underrun_grows_the_target_and_holds_off_shrinking() {
        let mut f = Feed::new(35.0);
        f.run(12_000.0, ms(46.0), 0, 0, |_| true);
        let shrunk = f.c.target();
        assert!(shrunk < ms(20.0));
        let grown = f.c.on_underrun(f.clock);
        assert!(grown > shrunk && grown <= ms(35.0), "{shrunk} -> {grown}");
        assert_eq!(f.c.grows(), 1);
        // The playback re-gates at the grown target: the queue stands that much higher again.
        f.removed -= (grown - shrunk) as u64;
        // The window is forgotten and a cooldown holds the next shed off.
        assert_eq!(f.c.valley(f.removed), None);
        let (_, shed) = f.run(20_000.0, ms(46.0), 0, 0, |_| true);
        assert_eq!(shed, 0, "inside the cooldown");
        let (_, shed) = f.run(20_000.0, ms(46.0), 0, 0, |_| true);
        assert!(shed > 0, "after it the cushion shrinks again");
        // The cooldown doubles with every underrun and growth stops at the cap.
        let mut last = 0;
        for _ in 0..6 {
            last = f.c.on_underrun(f.clock);
        }
        assert_eq!(last, ms(35.0));
    }

    #[test]
    fn discarded_audio_shifts_the_window_with_it() {
        // 100 ms of 46 ms levels, then the playback trims 20 ms by itself (drift): the window's
        // valley follows the trim instead of reading the old, higher levels.
        let mut f = Feed::new(35.0);
        f.run(9_000.0, ms(46.0), 0, 0, |_| false);
        let before = f.c.valley(f.removed).unwrap();
        f.removed += ms(20.0) as u64;
        let after = f.c.valley(f.removed).unwrap();
        assert_eq!(before - after, ms(20.0));
    }

    #[test]
    fn limits_and_voice_floors_restart_or_raise_the_floor() {
        let mut f = Feed::new(35.0);
        f.run(12_000.0, ms(46.0), 0, 0, |_| true);
        assert!(f.c.target() < ms(35.0));
        f.c.set_cap(ms(60.0));
        assert_eq!(f.c.target(), ms(60.0), "a new prefill starts over");
        assert_eq!(f.c.valley(0), None);
        // A voice that needs 50 ms never goes below it.
        let mut g = Feed::new(70.0);
        g.c.set_voice_floor(ms(50.0));
        assert_eq!(g.c.target(), ms(70.0));
        let (min_level, _) = g.run(30_000.0, ms(80.0), 0, 0, |_| true);
        assert!(min_level >= ms(50.0), "{min_level}");
        assert!(g.c.target() >= ms(50.0));
        // Unknown device callback: nothing is shed.
        let mut h = Feed::new(35.0);
        h.chunk = 512 * CH;
        for _ in 0..4_000 {
            h.clock += h.chunk as u64;
            let o = Observation { level_before: ms(46.0), consumed_total: h.clock, chunk_len: h.chunk, silent: true, ..Observation::default() };
            assert_eq!(h.c.observe(&o), Shed::None);
        }
    }
}
