//! Output side of the voice changer: the worker-owned queue ([`OutputPlayback`]) that hands whole
//! frames to a lock-free ring, the real-time playout ([`RingPlayout`]), and the interleaved
//! channel adapter used by the monitor and aux paths (sample-rate conversion lives in
//! `resample.rs` and `aux_feed.rs`).

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use super::cushion::{AdaptiveCushion, Observation, Shed, SILENCE_RMS_FLOOR};
use super::params::OutputCushion;

/// Extra ring top-up (frames per channel) beyond one device callback + one chunk, absorbing
/// scheduling jitter between the worker tick and the device callback.
const RING_TOPUP_MARGIN_FRAMES: usize = 64;

/// Floor of the output and monitor rings (interleaved samples). The ring must hold its top-up
/// target: prefill (or one device callback) + one chunk + one capture callback + margin. With a
/// 16384 floor, 48 kHz stereo, a 35 ms prefill and 512-frame chunks left room for capture
/// callbacks of only ~124 ms; 32768 covers ~290 ms.
pub(crate) const OUT_RING_MIN_SAMPLES: usize = 32_768;
pub(crate) const OUT_RING_MAX_SAMPLES: usize = 262_144;

/// After an underrun, blend this many **frames** (per channel) from the held sample back to live
/// audio so playback resumes without a click.
const RECOVER_FRAMES: usize = 16;

/// While starved, the held sample decays to silence over this many frames (a starve that lasts
/// never parks the speaker on a DC offset, which would thump when playback resumes).
const HOLD_DECAY_FRAMES: usize = 96;

/// Crossfade of a latency trim splice (ms).
pub(crate) const TRIM_FADE_MS: f64 = 5.0;
/// Pushed audio over which the smallest queue level is tracked for drift trimming (ms).
const DRIFT_WINDOW_MS: f64 = 1_000.0;
/// Shortest settle window of a recovery (ms of pushed audio).
const SETTLE_MIN_MS: f64 = 20.0;
/// Output rate assumed until the caller says (only sizes the trim fade and the windows).
const DEFAULT_RATE: u32 = 48_000;

/// The shared `callback_len` word: the latest device callback's length (low 32 bits, interleaved
/// samples) and, on 64-bit targets, the count of callbacks that ran (partly) dry after playback
/// began (high 32 bits). [`RingPlayout`] writes it every callback; [`OutputPlayback`] reads both,
/// so the queue learns about a starved device without any extra wiring.
const LEN_BITS: u32 = 32;
const LEN_MASK: usize = (1usize << (usize::BITS / 2)).wrapping_sub(1);

#[inline]
fn pack_callback(len: usize, starves: usize) -> usize {
    if usize::BITS >= 64 {
        ((starves & LEN_MASK) << (usize::BITS - LEN_BITS)) | (len & LEN_MASK)
    } else {
        len
    }
}

#[inline]
fn unpack_callback(word: usize) -> (usize, usize) {
    if usize::BITS >= 64 {
        (word & LEN_MASK, word >> (usize::BITS - LEN_BITS))
    } else {
        (word, 0)
    }
}

/// Longest a recovery lasts (device time, ms): it normally ends within a few windows.
const RECOVERY_MAX_MS: f64 = 3_000.0;
/// Pushes remembered for a recovery's valley.
const VALLEY_PUSHES: usize = 256;

/// Worker-owned output queue in front of a device's lock-free ring.
///
/// Latency control (P3-E; the benchmark's post-stall finding: after a stall the queue refilled to
/// `output_max_queue_ms` and stayed there, 74 -> 94 ms, for the rest of the session):
/// - **Recovery** starts when the device starved while playing (a stall, a slow block; the playout
///   counts dry callbacks into the shared `callback_len` word) or on
///   [`OutputPlayback::note_discontinuity`] (the worker's latency budget dropped input, or the
///   capture had a gap). Playback resumes at once (the catch-up itself refills the cushion). During
///   a recovery every push trims the queue to the ring's top-up target (the most a steady queue
///   holds right after a push), which bounds the catch-up burst; and once the device has consumed a
///   settle window (two of the longest period: chunk, capture or device callback) the valley of the
///   queue over the last such window of *device time* is trimmed back to the steady valley (the
///   lowest queue level before a push that steady playback showed before, at least one device
///   callback; the gate level, the prefill or one device callback, until one was measured). Pushes of an instant catch-up burst
///   all happen at one device time, so they drop out of the window as soon as the device plays on:
///   the valley is measured on steady playback. The recovery ends when a window's valley is already
///   at the steady one. A single stall therefore never leaves standing latency (the queue returns to
///   its pre-stall level), and a short glitch costs nothing beyond itself. Stalls that **repeat** are
///   another matter: a voice that converts in bursts (one long model call per block while nothing is
///   pushed) dries the device again before each recovery can end, so the recovery never ends and every
///   catch-up burst is trimmed to the ring target, which is lost audio, not latency control (38-50 % of
///   the converted audio on the CPU rows at the 35 / 70 ms default). The output queue must cover such
///   a voice's call; the cap is no help here, since the trim to the ring target (which follows the
///   prefill) comes first. `block_buffer` sizes the prefill for it.
/// - **Drift:** otherwise, when the smallest queue level over a second of pushed audio stays above
///   the gate level by more than one chunk / capture period / device callback (clock drift between
///   capture and playback devices), the excess is trimmed. (The adaptive cushion keeps the user's
///   prefill as this gate level, see [`OutputPlayback::drift_gate`]: its own target is the valley the
///   worst dip lands on, not the level the queue stands at between dips.)
/// - Every trim (and the `max_queue` cap) is a splice, not a cut: [`TRIM_FADE_MS`] of the audio
///   that continues what the device already has crossfades (equal power) into the first kept
///   audio; when the kept audio has not been converted yet, the fade finishes on the next push.
///
/// A trim never goes below the level steady playback already reached (or the gate level), so it
/// never causes an underrun by itself. Whole frames only; nothing here runs on the audio callback.
///
/// **Adaptive cushion** (`output_cushion = adaptive`, P3B-H, [`super::cushion`]): the gate level is not
/// the user's prefill but the valley the stream needs, found from the lateness of its pushes over a
/// window of minutes and shed with the audio that is silent anyway (or, after a long stretch without
/// silence, in small crossfaded splices). It never re-gates while playing (a dropout); only after an
/// underrun, with the queue empty anyway, it re-gates one step toward the user's value. The user's
/// prefill is its upper bound.
pub(crate) struct OutputPlayback {
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
    /// The device's latest callback length and dry-callback count (packed; written by
    /// [`RingPlayout`], 0 before the first callback).
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
    // ---- latency control (P3-E) ----
    rate: u32,
    /// Frames of a trim crossfade at this output's rate.
    fade_frames: usize,
    /// Interleaved samples of pushed audio per drift window.
    drift_window: usize,
    recovering: bool,
    /// Device consumption so far (interleaved samples; the ring level after each flush minus the
    /// level at the next push), the clock of a recovery's windows.
    consumed_total: u64,
    ring_after_flush: usize,
    /// Recovery: `(consumed_total, level before the push)` of the recent pushes, the device time
    /// the current window and the recovery started.
    valley: VecDeque<(u64, usize)>,
    valley_since: u64,
    recovery_since: u64,
    /// The lowest queue level before a push that steady playback showed (the minimum over the
    /// completed drift windows outside recoveries; reset by new limits).
    steady_valley: Option<usize>,
    /// Dry callbacks seen at the previous push.
    starves_seen: usize,
    /// Fade-out audio of a splice whose fade-in audio is not (all) queued yet (whole frames;
    /// capacity reserved for one fade).
    pending_out: Vec<f32>,
    /// Frames of `pending_out` already blended into the queue.
    pending_pos: usize,
    /// Queue frame where the next frame of `pending_out` blends (frames before it are blended).
    pending_at: usize,
    /// Smallest queue level (before a push) over the current window, and the audio pushed in it
    /// (interleaved samples).
    window_min: usize,
    window_pushed: usize,
    trimmed_frames: u64,
    recoveries: u64,
    /// How the prefill is sized (`fixed`: the user's prefill; `adaptive`: [`AdaptiveCushion`]).
    cushion: OutputCushion,
    /// The adaptive controller (`Some` exactly when `cushion` is [`OutputCushion::Adaptive`]).
    adaptive: Option<AdaptiveCushion>,
    /// Interleaved samples the queue discarded so far (every trim, cut and drop): the coordinate shift
    /// of the adaptive window.
    removed_total: u64,
    /// The last status slot written to the cushion trace (`SIGIL_VC_CUSHION_TRACE`).
    trace_slot: u64,
    /// Frames trimmed by each rule (recovery, drift, cap), for tests.
    #[cfg(test)]
    trimmed_by: [u64; 3],
    /// Tests: the drift trim keyed to the adaptive target (what it was before P3B-H's fix), to show the
    /// test that catches it.
    #[cfg(test)]
    drift_keyed_to_target: bool,
}

fn rms_interleaved(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|x| x * x).sum();
    (sum / samples.len() as f32).sqrt()
}

/// Equal-power crossfade weights `(fade_in, fade_out)` at frame `i` of an `n`-frame fade.
#[inline]
fn xfade_weights(i: usize, n: usize) -> (f32, f32) {
    let t = (i as f32 + 0.5) / n.max(1) as f32;
    let a = t * std::f32::consts::FRAC_PI_2;
    (a.sin(), a.cos())
}

impl OutputPlayback {
    pub(crate) fn new(
        prefill_target: usize,
        max_queue_samples: usize,
        ch_out: usize,
        callback_len: Arc<AtomicUsize>,
    ) -> Self {
        let ch = ch_out.max(1);
        let starves_seen = unpack_callback(callback_len.load(Ordering::Relaxed)).1;
        let mut s = Self {
            deque: VecDeque::with_capacity(max_queue_samples.max(prefill_target) + 8192),
            prefill_target,
            max_queue_samples: max_queue_samples.max(prefill_target + 1),
            prefilled: false,
            truncate_silent_streak: 0,
            last_chunk_len: 0,
            ch,
            callback_len,
            in_burst_len: Arc::new(AtomicUsize::new(0)),
            in_burst_ch: 1,
            in_burst_r_in: 1,
            in_burst_r_out: 1,
            rate: DEFAULT_RATE,
            fade_frames: 1,
            drift_window: 1,
            recovering: false,
            consumed_total: 0,
            ring_after_flush: 0,
            valley: VecDeque::with_capacity(VALLEY_PUSHES),
            valley_since: 0,
            recovery_since: 0,
            steady_valley: None,
            starves_seen,
            pending_out: Vec::new(),
            pending_pos: 0,
            pending_at: 0,
            window_min: usize::MAX,
            window_pushed: 0,
            trimmed_frames: 0,
            recoveries: 0,
            cushion: OutputCushion::Fixed,
            adaptive: None,
            removed_total: 0,
            trace_slot: 0,
            #[cfg(test)]
            trimmed_by: [0; 3],
            #[cfg(test)]
            drift_keyed_to_target: false,
        };
        s.set_output_rate(DEFAULT_RATE);
        s
    }

    /// Wire the capture callback length (`in_burst_len`, interleaved samples at `ch_in` / `r_in`)
    /// into the ring target of this output (`r_out`, also this output's rate).
    pub(crate) fn with_capture_burst(
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
        self.set_output_rate(r_out);
        self
    }

    /// This output's sample rate (sizes the trim crossfade and the windows).
    #[allow(dead_code)] // `with_capture_burst` sets it for every caller today.
    pub(crate) fn with_output_rate(mut self, rate: u32) -> Self {
        self.set_output_rate(rate);
        self
    }

    /// The output cushion mode (params `output_cushion`): `adaptive` sizes the gate from the stream
    /// (see the type docs), with the prefill as its cap.
    pub(crate) fn with_cushion(mut self, cushion: OutputCushion) -> Self {
        self.cushion = cushion;
        self.adaptive = (cushion == OutputCushion::Adaptive).then(|| AdaptiveCushion::new(self.rate, self.ch, self.prefill_target, 0));
        self
    }

    /// Tests: the adaptive history before the first shed.
    #[cfg(test)]
    pub(crate) fn set_cushion_history_ms(&mut self, ms: f64) {
        if let Some(a) = self.adaptive.as_mut() {
            a.set_history_ms(ms);
        }
    }

    /// The smallest cushion a voice allows (`min_output_buffer_ms` of a burst family, interleaved
    /// samples; 0 = none): the adaptive floor. A no-op in fixed mode, where the prefill already is it.
    pub(crate) fn set_cushion_floor(&mut self, samples: usize) {
        if let Some(a) = self.adaptive.as_mut() {
            a.set_voice_floor(samples);
        }
    }

    /// The cushion the queue holds now (interleaved samples): the adaptive target, else the gate level
    /// (the user's prefill). For the metrics and the UI.
    pub(crate) fn cushion_samples(&self) -> usize {
        self.gate_base()
    }

    /// Sheds of the adaptive cushion `(events, samples)` and its underrun growths (tests, metrics).
    #[allow(dead_code)] // Diagnostics accessor.
    pub(crate) fn adaptive_stats(&self) -> Option<((u64, u64), u64)> {
        self.adaptive.as_ref().map(|a| (a.sheds(), a.grows()))
    }

    /// The output cushion mode in effect.
    #[allow(dead_code)] // P3B-S0 frozen API (P3B-H, the benchmark).
    pub(crate) fn cushion(&self) -> OutputCushion {
        self.cushion
    }

    fn set_output_rate(&mut self, rate: u32) {
        self.rate = rate.max(1);
        if let Some(a) = self.adaptive.as_mut() {
            a.set_rate(self.rate);
        }
        let rate = f64::from(self.rate);
        self.fade_frames = ((TRIM_FADE_MS * rate / 1000.0).round() as usize).max(1);
        self.drift_window = ((DRIFT_WINDOW_MS * rate / 1000.0).round() as usize).max(1) * self.ch;
        self.pending_out = Vec::with_capacity(self.fade_frames * self.ch);
    }

    pub(crate) fn queue_len(&self) -> usize {
        self.deque.len()
    }

    /// Frames removed by latency trims so far (recovery, drift and the queue cap).
    pub(crate) fn trimmed_frames(&self) -> u64 {
        self.trimmed_frames
    }

    /// Recoveries started by a starved device or a discontinuity.
    pub(crate) fn recoveries(&self) -> u64 {
        self.recoveries
    }

    /// Whether the queue is being trimmed back after an underrun or a discontinuity.
    #[cfg(test)]
    pub(crate) fn is_recovering(&self) -> bool {
        self.recovering
    }

    /// The input side skipped audio (the latency budget dropped stale input, or the capture had a
    /// gap): whatever the catch-up produces beyond the steady level is stale; trim it back.
    pub(crate) fn note_discontinuity(&mut self) {
        self.start_recovery();
    }

    fn start_recovery(&mut self) {
        if !self.recovering {
            self.recoveries += 1;
            self.recovery_since = self.consumed_total;
        }
        self.recovering = true;
        self.valley.clear();
        self.valley_since = self.consumed_total;
        self.reset_window();
    }

    /// One recovery step at a push (`level_before`: the queue before it). See the type docs.
    fn recover(&mut self, level_before: usize, ring_used: usize) {
        let target = self.ring_target();
        let level = self.deque.len() + ring_used;
        if level > target {
            self.trim_counted(level - target, 0);
        }
        let now = self.consumed_total;
        let window = self.settle_window() as u64;
        if self.valley.len() == VALLEY_PUSHES {
            self.valley.pop_front();
        }
        self.valley.push_back((now, level_before));
        // Only pushes inside the last window of device time (the catch-up burst's all happened at
        // the window's start and drop out here).
        while self.valley.front().is_some_and(|&(t, _)| t + window <= now) {
            self.valley.pop_front();
        }
        if now.saturating_sub(self.valley_since) < window {
            return;
        }
        let valley = self.valley.iter().map(|&(_, l)| l).min().unwrap_or(0);
        let peak = self.valley.iter().map(|&(_, l)| l).max().unwrap_or(0);
        let floor = RING_TOPUP_MARGIN_FRAMES * self.ch;
        // The steady valley, but never less than one device callback (a valley measured at push
        // instants that happened to fall right after a callback can be lower than what keeps every
        // callback fed), nor above the gate level.
        let gate = self.gate_target();
        let goal = self
            .steady_valley
            .map_or(gate, |v| v.max(gate.min(self.callback_samples())))
            .max(floor);
        let expired = now.saturating_sub(self.recovery_since) >= ((RECOVERY_MAX_MS * f64::from(self.rate) / 1000.0) as u64) * self.ch as u64;
        if valley > goal + floor && !expired {
            self.trim_counted(valley - goal, 0);
            // The recorded levels predate the trim: measure a fresh window.
            self.valley.clear();
            self.valley_since = now;
        } else if expired || peak <= valley + self.longest_period() + floor {
            // Steady playback at (or below) the goal: recovered. A window whose levels still climb
            // (a catch-up burst in progress) keeps sliding instead.
            self.recovering = false;
            self.valley.clear();
            self.reset_window();
        }
    }

    fn reset_window(&mut self) {
        self.window_min = usize::MAX;
        self.window_pushed = 0;
    }

    /// The latest device callback's length (interleaved samples).
    fn callback_samples(&self) -> usize {
        unpack_callback(self.callback_len.load(Ordering::Relaxed)).0
    }

    /// Latest capture callback converted to this output's interleaved samples (whole frames,
    /// rounded up).
    fn in_burst_samples(&self) -> usize {
        let frames_in = (self.in_burst_len.load(Ordering::Relaxed) / self.in_burst_ch) as u64;
        let frames_out = (frames_in * self.in_burst_r_out).div_ceil(self.in_burst_r_in);
        (frames_out as usize).saturating_mul(self.ch)
    }

    /// The prefill the gate is built on: the user's, or the adaptive controller's target.
    fn gate_base(&self) -> usize {
        match &self.adaptive {
            Some(a) => a.target(),
            None => self.prefill_target,
        }
    }

    /// Level at which the prefill gate opens: the prefill, or one full device callback when that
    /// is longer (a gate below one callback would underrun the very first period).
    fn gate_target(&self) -> usize {
        self.gate_base().max(self.callback_samples())
    }

    /// The level the clock-drift trim brings the drift window's smallest queue level down to: the
    /// gate level of a fixed cushion; with the adaptive one the gate level of the USER's prefill (the
    /// controller's cap), not of its target. The target is the valley, the level the stream's worst dip
    /// lands on; between dips the queue stands the dip's depth above it. A drift ceiling keyed to the
    /// target trimmed that standing level away on the first dip-free window (a window's smallest level
    /// over the target plus one period is "drift"), so the next late tick deeper than the slack ran the
    /// device dry. The controller sheds what the stream's lateness allows below the cap; this trim
    /// only bounds clock drift, as it does with the fixed cushion, so the adaptive queue never
    /// stands higher than the fixed one would.
    fn drift_gate(&self) -> usize {
        #[cfg(test)]
        if self.drift_keyed_to_target {
            return self.gate_target();
        }
        match &self.adaptive {
            Some(a) => self.gate_target().max(a.cap()),
            None => self.gate_target(),
        }
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

    /// The longest of the periods that shape the queue: a chunk, a capture callback, a device
    /// callback.
    fn longest_period(&self) -> usize {
        self.last_chunk_len.max(self.in_burst_samples()).max(self.callback_samples())
    }

    /// Queue level above the gate level that steady playback can legitimately hold at its lowest
    /// point (the longest period plus the top-up margin).
    fn drift_slack(&self) -> usize {
        self.longest_period().saturating_add(RING_TOPUP_MARGIN_FRAMES * self.ch)
    }

    /// Pushed audio of a recovery's settle window: two of the longest periods, at least
    /// [`SETTLE_MIN_MS`].
    fn settle_window(&self) -> usize {
        let min = ((SETTLE_MIN_MS * f64::from(self.rate) / 1000.0).round() as usize).max(1) * self.ch;
        self.longest_period().saturating_mul(2).max(min)
    }

    /// Drop at least `n` samples from the front of the deque, rounded up to whole frames and
    /// capped at the deque length (the ring part cannot be trimmed from here). A plain cut: only
    /// for silence (buffer truncation) and tests.
    fn drop_front(&mut self, n: usize) {
        let ch = self.ch;
        let n = (n.div_ceil(ch) * ch).min(self.deque.len());
        if n > 0 {
            self.deque.drain(0..n);
            self.pending_at = self.pending_at.saturating_sub(n / ch);
            self.removed_total += n as u64;
        }
    }

    /// Blend the unfinished splice fade into the queued frames that follow it.
    fn blend_pending(&mut self) {
        if self.pending_out.is_empty() {
            return;
        }
        let ch = self.ch;
        let total = self.pending_out.len() / ch;
        let queued = self.deque.len() / ch;
        while self.pending_pos < total && self.pending_at < queued {
            let (w_in, w_out) = xfade_weights(self.pending_pos, total);
            for c in 0..ch {
                let old = self.pending_out[self.pending_pos * ch + c];
                let s = &mut self.deque[self.pending_at * ch + c];
                *s = *s * w_in + old * w_out;
            }
            self.pending_pos += 1;
            self.pending_at += 1;
        }
        if self.pending_pos >= total {
            self.pending_out.clear();
            self.pending_pos = 0;
            self.pending_at = 0;
        }
    }

    /// Drop up to `frames` frames from the front of the deque as a crossfaded splice (the ring
    /// part cannot be trimmed from here). Skipped while an earlier splice is still fading.
    fn trim_frames(&mut self, frames: usize) {
        let ch = self.ch;
        let queued = self.deque.len() / ch;
        let drop = frames.min(queued);
        if drop == 0 || !self.pending_out.is_empty() {
            return;
        }
        // The fade-out: the audio that continues what the device already has.
        let fade = self.fade_frames.min(queued);
        self.pending_out.clear();
        self.pending_out.extend(self.deque.range(..fade * ch).copied());
        self.pending_pos = 0;
        self.pending_at = 0;
        self.deque.drain(..drop * ch);
        self.removed_total += (drop * ch) as u64;
        self.trimmed_frames += drop as u64;
        // The fade-in: whatever audio follows the dropped part (the rest arrives with the next push).
        self.blend_pending();
    }

    /// Trim `samples` (rounded up to whole frames) of queue level.
    fn trim_samples(&mut self, samples: usize) {
        self.trim_frames(samples.div_ceil(self.ch));
    }

    /// [`Self::trim_samples`], counted per rule in tests (0 recovery, 1 drift, 2 cap).
    fn trim_counted(&mut self, samples: usize, _rule: usize) {
        #[cfg(test)]
        let before = self.trimmed_frames;
        self.trim_samples(samples);
        #[cfg(test)]
        {
            self.trimmed_by[_rule] += self.trimmed_frames - before;
        }
    }

    /// Apply new prefill/max targets from hot params without restarting cpal (may reset prefill gating).
    /// `ring_used` = samples already handed to the lock-free output ring (still queued for playback).
    pub(crate) fn update_limits(&mut self, prefill_target: usize, max_queue_samples: usize, ring_used: usize) {
        let max_queue_samples = max_queue_samples.max(prefill_target.saturating_add(1));
        if self.prefill_target == prefill_target && self.max_queue_samples == max_queue_samples {
            return;
        }
        self.prefill_target = prefill_target;
        self.max_queue_samples = max_queue_samples;
        if let Some(a) = self.adaptive.as_mut() {
            // The user's prefill is the adaptive cap: a new one starts the window over.
            a.set_cap(prefill_target);
        }
        let over = (self.deque.len() + ring_used).saturating_sub(self.queue_cap());
        self.trim_samples(over);
        if self.deque.len() + ring_used < self.gate_base() {
            self.prefilled = false;
        }
        self.steady_valley = None;
        self.reset_window();
    }

    pub(crate) fn push_interleaved(
        &mut self,
        samples: &[f32],
        truncate_enabled: bool,
        truncate_rms: f32,
        truncate_chunks: u32,
        ring_used: usize,
    ) {
        let level_before = self.deque.len() + ring_used;
        // What the device played since the last flush (nothing else lowers the ring).
        self.consumed_total += self.ring_after_flush.saturating_sub(ring_used) as u64;
        self.ring_after_flush = ring_used;
        let starves = unpack_callback(self.callback_len.load(Ordering::Relaxed)).1;
        let starved = std::mem::replace(&mut self.starves_seen, starves) != starves;
        if starved && self.prefilled && !samples.is_empty() {
            // The device ran dry while playing: whatever catch-up follows is stale.
            self.start_recovery();
            // Adaptive: the cushion was too tight. The queue is empty anyway, so this is the one
            // moment to re-gate: it refills to the grown target before it plays.
            if let Some(a) = self.adaptive.as_mut() {
                let before = a.target();
                a.on_underrun(self.consumed_total);
                tracing::debug!(
                    target: "sigil::vc::cushion",
                    "output cushion: underrun, target {:.1} -> {:.1} ms (floor {:.1} ms, learned {:.1} ms)",
                    a.ms(before),
                    a.ms(a.target()),
                    a.ms(a.floor_total()),
                    a.ms(a.learned_floor())
                );
                crate::services::voice_changer::cushion::trace(format_args!(
                    "underrun at {:.2} s: target {:.1} -> {:.1} ms, learned {:.1} ms",
                    a.ms(self.consumed_total as usize),
                    a.ms(before),
                    a.ms(a.target()),
                    a.ms(a.learned_floor())
                ));
                self.prefilled = false;
                self.steady_valley = None;
            }
        }
        // Adaptive: what the controller sheds from this push (silence from the head of a quiet
        // chunk, or a crossfaded splice of the queue after the chunk is queued).
        let mut samples = samples;
        let mut splice = 0usize;
        if self.prefilled && !starved && !self.recovering && !samples.is_empty() && self.adaptive.is_some() {
            let silent = rms_interleaved(samples) < truncate_rms.max(SILENCE_RMS_FLOOR);
            let obs = Observation {
                level_before,
                removed_total: self.removed_total,
                consumed_total: self.consumed_total,
                chunk_len: samples.len(),
                silent,
                callback_len: self.callback_samples(),
                capture_burst: self.in_burst_samples(),
            };
            let shed = self.adaptive.as_mut().map(|a| a.observe(&obs)).unwrap_or(Shed::None);
            if let (Some(a), Shed::Silent(n) | Shed::Splice(n)) = (self.adaptive.as_ref(), shed) {
                tracing::debug!(
                    target: "sigil::vc::cushion",
                    "output cushion: shed {:.1} ms ({}), valley {:.1} ms, floor {:.1} ms, target {:.1} ms",
                    a.ms(n),
                    if matches!(shed, Shed::Silent(_)) { "silence" } else { "splice" },
                    a.valley(self.removed_total).map_or(0.0, |v| a.ms(v)),
                    a.ms(a.floor_total()),
                    a.ms(a.target())
                );
            }
            if let (Some(a), Shed::Silent(n) | Shed::Splice(n)) = (self.adaptive.as_ref(), shed) {
                crate::services::voice_changer::cushion::trace(format_args!(
                    "shed {:.1} ms ({}) at {:.2} s: valley {:.1} floor {:.1} target {:.1}",
                    a.ms(n),
                    if matches!(shed, Shed::Silent(_)) { "silence" } else { "splice" },
                    a.ms(self.consumed_total as usize),
                    a.valley(self.removed_total).map_or(0.0, |v| a.ms(v)),
                    a.ms(a.floor_total()),
                    a.ms(a.target())
                ));
            }
            if let Some(a) = self.adaptive.as_ref() {
                if crate::services::voice_changer::cushion::trace_on() {
                    let every = (crate::services::voice_changer::cushion::TRACE_STATUS_EVERY_MS * f64::from(self.rate) / 1000.0) as u64 * self.ch as u64;
                    let slot = self.consumed_total / every.max(1);
                    if slot != self.trace_slot {
                        self.trace_slot = slot;
                        crate::services::voice_changer::cushion::trace(format_args!(
                            "{:.1} s level {:.1} {}",
                            a.ms(self.consumed_total as usize),
                            a.ms(level_before),
                            a.status(self.removed_total)
                        ));
                    }
                }
            }
            match shed {
                Shed::Silent(n) => {
                    // Whole frames from the head of the quiet chunk: nobody hears them.
                    let drop = (n / self.ch * self.ch).min(samples.len());
                    samples = &samples[drop..];
                    self.removed_total += drop as u64;
                }
                Shed::Splice(n) => splice = n,
                Shed::None => {}
            }
        }
        self.last_chunk_len = samples.len();
        self.deque.extend(samples.iter().copied());
        self.blend_pending();
        if splice > 0 {
            self.trim_samples(splice);
        }

        if !samples.is_empty() && self.prefilled {
            if self.recovering {
                self.recover(level_before, ring_used);
            } else {
                self.window_min = self.window_min.min(level_before);
                self.window_pushed += samples.len();
                if self.window_pushed >= self.drift_window {
                    let gate = self.drift_gate();
                    if self.window_min != usize::MAX && self.window_min > gate + self.drift_slack() {
                        self.trim_counted(self.window_min - gate, 1);
                    } else if self.window_min != usize::MAX {
                        self.steady_valley = Some(self.steady_valley.map_or(self.window_min, |v| v.min(self.window_min)));
                    }
                    self.reset_window();
                }
            }
        }
        // The queue cap covers deque + ring; only the deque part can be trimmed here.
        let over = (self.deque.len() + ring_used).saturating_sub(self.queue_cap());
        self.trim_counted(over, 2);

        if truncate_enabled && truncate_chunks > 0 && !samples.is_empty() {
            let rms = rms_interleaved(samples);
            if rms < truncate_rms {
                self.truncate_silent_streak = self.truncate_silent_streak.saturating_add(1);
            } else {
                self.truncate_silent_streak = 0;
            }
            if self.truncate_silent_streak >= truncate_chunks {
                self.drop_front((self.deque.len() + ring_used).saturating_sub(self.gate_base()));
                self.truncate_silent_streak = 0;
            }
        } else {
            self.truncate_silent_streak = 0;
        }
    }

    /// Hand whole frames to the lock-free output ring once the prefill gate opens. The ring is
    /// only topped up to [`Self::ring_target`], so any backlog stays in the deque where the
    /// queue cap and silence truncation can still trim it. Frames of an unfinished splice fade are
    /// held back until the fade completes.
    pub(crate) fn flush_to_ring(&mut self, producer: &mut rtrb::Producer<f32>, ring_used: usize) {
        if !self.prefilled {
            if self.deque.len() + ring_used >= self.gate_target() {
                self.prefilled = true;
                // Dry callbacks while the gate was closed are not starvation.
                self.starves_seen = unpack_callback(self.callback_len.load(Ordering::Relaxed)).1;
            } else {
                return;
            }
        }
        let ch = self.ch;
        let budget = self.ring_target().saturating_sub(ring_used);
        let ready = if self.pending_out.is_empty() {
            self.deque.len() / ch
        } else {
            self.pending_at
        };
        let n = (budget / ch).min(ready).min(producer.slots() / ch) * ch;
        if n == 0 {
            return;
        }
        // One commit for the whole run of frames: the consumer never observes a partial frame.
        if let Ok(chunk) = producer.write_chunk_uninit(n) {
            chunk.fill_from_iter(self.deque.drain(..n));
            self.pending_at = self.pending_at.saturating_sub(n / ch);
            self.ring_after_flush = ring_used + n;
        }
    }
}

/// Real-time side of an output device: pulls whole frames from the lock-free ring, holds the
/// last frame on underrun (decaying to silence) and ramps back to live audio after a starve. All
/// state is closure-local (no locks, no allocation per callback).
pub(crate) struct RingPlayout {
    consumer: rtrb::Consumer<f32>,
    /// Last emitted frame (one value per channel).
    hold: Vec<f32>,
    starving: bool,
    /// Frames left in the post-underrun recovery ramp.
    recover_left: usize,
    /// Frames of the hold decay still to go while starving.
    decay_left: usize,
    /// Latest callback length and dry-callback count, packed (read by [`OutputPlayback`]).
    callback_len: Arc<AtomicUsize>,
    /// Callbacks that ran (partly) dry after playback began (packed into `callback_len`).
    starves: usize,
    /// Callbacks that ran (partly) dry after playback first began (the prefill wait before the
    /// first audio is not an underrun).
    underruns: Option<Arc<AtomicU64>>,
    /// Callbacks so far (the worker's output stall detector).
    callbacks: Option<Arc<AtomicU64>>,
    played_any: bool,
}

impl RingPlayout {
    pub(crate) fn new(consumer: rtrb::Consumer<f32>, ch: usize, callback_len: Arc<AtomicUsize>) -> Self {
        Self {
            consumer,
            hold: vec![0.0; ch.max(1)],
            // Start "starved" so the first audio after prefill fades in from silence.
            starving: true,
            recover_left: 0,
            decay_left: 0,
            callback_len,
            starves: 0,
            underruns: None,
            callbacks: None,
            played_any: false,
        }
    }

    /// Count underruns into `counter` (the main output's `underruns` metric).
    pub(crate) fn with_underrun_counter(mut self, counter: Arc<AtomicU64>) -> Self {
        self.underruns = Some(counter);
        self
    }

    /// Count callbacks into `counter` (a device that stops calling back is lost).
    pub(crate) fn with_callback_counter(mut self, counter: Arc<AtomicU64>) -> Self {
        self.callbacks = Some(counter);
        self
    }

    pub(crate) fn fill(&mut self, data: &mut [f32]) {
        if let Some(c) = &self.callbacks {
            c.fetch_add(1, Ordering::Relaxed);
        }
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
                self.starves = self.starves.wrapping_add(1);
                if let Some(c) = &self.underruns {
                    c.fetch_add(1, Ordering::Relaxed);
                }
            }
            if !self.starving {
                self.starving = true;
                self.decay_left = HOLD_DECAY_FRAMES;
            }
            for frame in data[filled..].chunks_mut(ch) {
                if self.decay_left > 0 {
                    // Linear decay of the held frame to silence.
                    let g = (self.decay_left - 1) as f32 / self.decay_left as f32;
                    self.decay_left -= 1;
                    for h in self.hold.iter_mut() {
                        *h *= g;
                    }
                }
                for (s, &h) in frame.iter_mut().zip(self.hold.iter()) {
                    *s = h;
                }
            }
        }
        self.callback_len.store(pack_callback(data.len(), self.starves), Ordering::Relaxed);
    }
}

pub(crate) fn adapt_channels_interleaved_into(data: &[f32], from_ch: usize, to_ch: usize, out: &mut Vec<f32>) {
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
    use super::{adapt_channels_interleaved_into, OutputCushion, OutputPlayback, RingPlayout, HOLD_DECAY_FRAMES, RECOVER_FRAMES};
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
        assert_eq!(super::unpack_callback(cb_len.load(Ordering::Relaxed)).0, 480 * ch);
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

    // ---- latency control (P3-E) ----

    /// A simulated run on one 48 kHz clock: capture delivers `cap` mono frames per period, the
    /// worker converts whole `chunk`-frame blocks the moment they are complete (a stall holds them
    /// back; afterwards the latency budget keeps `budget_keep` frames and the worker catches up in a
    /// burst), the device pulls `cb` stereo frames per callback, half a period out of phase. The
    /// audio is a continuous 220 Hz sine on the input clock, so splices and drops are audible in the
    /// played stream.
    struct Sim {
        pb: OutputPlayback,
        prod: rtrb::Producer<f32>,
        playout: RingPlayout,
        underruns: Arc<std::sync::atomic::AtomicU64>,
        burst: Arc<AtomicUsize>,
        chunk: usize,
        cap: usize,
        /// Device frames between two device callbacks (== `cb` unless the clocks drift).
        cb_period: usize,
        input_waiting: usize,
        input_pos: u64,
        stalled: bool,
        /// The worker tells the queue about budget drops (`note_discontinuity`).
        notify_drops: bool,
        /// Blocks the worker converts per wake (its compute speed; `usize::MAX` = instant).
        blocks_per_wake: usize,
        played: Vec<f32>,
        /// (time in frames, queue level in frames) at every push, before the push.
        push_levels: Vec<(u64, usize)>,
        /// `(period, quiet)` in blocks: the last `quiet` blocks of every `period` are silent (a
        /// pause in the speech); `None` = a continuous sine.
        quiet: Option<(u64, u64)>,
        block_index: u64,
        /// Frames of voiced (non-silent) audio pushed.
        voiced_pushed: u64,
        /// Extra frames the worker's push is late by at the blocks `late_every` apart (jitter).
        late: Option<(u64, u64)>,
        next_late_at: u64,
        t: u64,
        next_cap: u64,
        next_cb: u64,
        buf: Vec<f32>,
    }

    const SIM_RATE: u32 = 48_000;
    const SIM_CH: usize = 2;
    const SIM_RING: usize = 65_536;

    impl Sim {
        fn new(prefill_ms: f64, max_queue_ms: f64, chunk: usize, cap: usize, cb: usize) -> Sim {
            Self::with_cushion(prefill_ms, max_queue_ms, chunk, cap, cb, OutputCushion::Fixed)
        }

        fn with_cushion(prefill_ms: f64, max_queue_ms: f64, chunk: usize, cap: usize, cb: usize, cushion: OutputCushion) -> Sim {
            let frames = |ms: f64| (f64::from(SIM_RATE) * ms / 1000.0).ceil() as usize;
            let cb_len = Arc::new(AtomicUsize::new(0));
            let burst = Arc::new(AtomicUsize::new(0));
            let (prod, cons) = RingBuffer::<f32>::new(SIM_RING);
            let underruns = Arc::new(std::sync::atomic::AtomicU64::new(0));
            let mut pb = OutputPlayback::new(frames(prefill_ms) * SIM_CH, frames(max_queue_ms) * SIM_CH, SIM_CH, cb_len.clone())
                .with_capture_burst(burst.clone(), 1, SIM_RATE, SIM_RATE)
                .with_cushion(cushion);
            // The simulations are minutes long: a short history before the first shed.
            pb.set_cushion_history_ms(8_000.0);
            let playout = RingPlayout::new(cons, SIM_CH, cb_len).with_underrun_counter(underruns.clone());
            Sim {
                pb,
                prod,
                playout,
                underruns,
                burst,
                chunk,
                cap,
                cb_period: cb,
                input_waiting: 0,
                input_pos: 0,
                stalled: false,
                notify_drops: true,
                blocks_per_wake: usize::MAX,
                played: Vec::new(),
                push_levels: Vec::new(),
                quiet: None,
                block_index: 0,
                voiced_pushed: 0,
                late: None,
                next_late_at: 0,
                t: 0,
                next_cap: cap as u64,
                next_cb: (cb / 2) as u64,
                buf: vec![0.0; cb * SIM_CH],
            }
        }

        fn ring_used(&self) -> usize {
            SIM_RING - self.prod.slots()
        }

        fn level_frames(&self) -> usize {
            (self.pb.queue_len() + self.ring_used()) / SIM_CH
        }

        /// Convert every whole block waiting.
        fn convert_waiting(&mut self) {
            let mut n = 0;
            while self.input_waiting >= self.chunk && n < self.blocks_per_wake {
                n += 1;
                self.input_waiting -= self.chunk;
                let first = self.input_pos;
                self.input_pos += self.chunk as u64;
                let quiet = self.quiet.is_some_and(|(period, q)| self.block_index % period >= period - q);
                self.block_index += 1;
                if !quiet {
                    self.voiced_pushed += self.chunk as u64;
                }
                let block: Vec<f32> = (0..self.chunk as u64)
                    .flat_map(|i| {
                        let v = if quiet {
                            0.0
                        } else {
                            (2.0 * std::f64::consts::PI * 220.0 * (first + i) as f64 / f64::from(SIM_RATE)).sin() as f32 * 0.5
                        };
                        [v, v]
                    })
                    .collect();
                self.push_levels.push((self.t, self.level_frames()));
                let used = self.ring_used();
                self.pb.push_interleaved(&block, false, 0.0, 0, used);
                let used = self.ring_used();
                self.pb.flush_to_ring(&mut self.prod, used);
            }
        }

        /// The worker wakes after a stall: the latency budget keeps the newest `keep` frames.
        fn end_stall(&mut self, keep: usize) {
            self.stalled = false;
            if self.input_waiting > keep {
                let dropped = self.input_waiting - keep;
                self.input_waiting = keep;
                self.input_pos += dropped as u64;
                if self.notify_drops {
                    self.pb.note_discontinuity();
                }
            }
            self.convert_waiting();
        }

        fn run_for(&mut self, frames: u64) {
            let end = self.t + frames;
            while self.t < end {
                let next = self.next_cap.min(self.next_cb);
                self.t = next;
                if self.next_cap == next {
                    self.next_cap += self.cap as u64;
                    self.burst.store(self.cap, Ordering::Relaxed);
                    self.input_waiting += self.cap;
                    // Jitter: every `every`-th capture period the worker is `extra` frames late.
                    let held = self.late.is_some_and(|(every, extra)| {
                        let k = (self.next_cap / self.cap as u64) % every.max(1);
                        k == 0 && extra > 0 && {
                            self.next_late_at = self.t + extra;
                            true
                        }
                    });
                    // A worker that is late stays late: later capture periods do not wake it (the hold
                    // ends at the first event at or after `next_late_at`, below).
                    if !self.stalled && !held && self.next_late_at == 0 {
                        self.convert_waiting();
                    }
                }
                if self.next_late_at != 0 && self.t >= self.next_late_at {
                    self.next_late_at = 0;
                    if !self.stalled {
                        self.convert_waiting();
                    }
                }
                if self.next_cb == next {
                    self.next_cb += self.cb_period as u64;
                    self.playout.fill(&mut self.buf);
                    self.played.extend(self.buf.iter().step_by(SIM_CH));
                }
            }
        }

        fn levels_between(&self, from_s: f64, to_s: f64) -> Vec<usize> {
            let (a, b) = ((from_s * f64::from(SIM_RATE)) as u64, (to_s * f64::from(SIM_RATE)) as u64);
            self.push_levels.iter().filter(|(t, _)| *t >= a && *t < b).map(|x| x.1).collect()
        }

        fn underruns(&self) -> u64 {
            self.underruns.load(Ordering::Relaxed)
        }
    }

    fn median(mut v: Vec<usize>) -> usize {
        v.sort_unstable();
        v[v.len() / 2]
    }

    /// The largest sample-to-sample step of a mono stream.
    fn max_step(s: &[f32]) -> f32 {
        s.windows(2).map(|w| (w[1] - w[0]).abs()).fold(0.0, f32::max)
    }

    /// The benchmark's post-stall finding (74 -> 94 ms, persistent): after a 3 s stall the catch-up
    /// burst refilled the queue to `max_queue` and it stayed there. Now the underrun re-gates, the
    /// recovery trims the burst away, and the queue is back at its pre-stall level within a fraction
    /// of a second, with no further underruns and no click.
    #[test]
    fn post_stall_catch_up_does_not_become_standing_latency() {
        post_stall_recovers(Sim::new(35.0, 70.0, 512, 480, 480));
        // Without the budget's notice (the benchmark's driver): the starved device alone.
        let mut quiet = Sim::new(35.0, 70.0, 512, 480, 480);
        quiet.notify_drops = false;
        post_stall_recovers(quiet);
        // A chunk longer than the prefill.
        post_stall_recovers(Sim::new(35.0, 70.0, 2048, 480, 1024));
        // A worker that catches up at about 3x real time (the burst takes a while).
        let mut slow = Sim::new(35.0, 70.0, 512, 480, 480);
        slow.blocks_per_wake = 3;
        post_stall_recovers(slow);
    }

    fn post_stall_recovers(mut sim: Sim) {
        let (chunk, cb) = (sim.chunk, sim.cb_period);
        sim.run_for(3 * 48_000);
        let before = median(sim.levels_between(1.0, 3.0));
        sim.stalled = true;
        sim.run_for(3 * 48_000);
        // The budget (300 ms) keeps its low-water mark, 150 ms, then the worker catches up.
        sim.end_stall(7_200);
        let underruns_after_stall = sim.underruns();
        sim.run_for(5 * 48_000);
        assert_eq!(sim.pb.recoveries(), 1, "the starve started one recovery");
        assert!(!sim.pb.is_recovering());
        let after = sim.levels_between(7.0, 11.0);
        let (m_after, worst) = (median(after.clone()), *after.iter().max().unwrap());
        assert!(
            m_after.abs_diff(before) <= chunk,
            "queue level {m_after} frames after the stall vs {before} before (one chunk tolerance)"
        );
        assert!(worst <= before + chunk + cb, "chunk {chunk}: peak {worst} vs steady {before}");
        // The recovery happens within half a second of the stall's end.
        let settled: Vec<usize> = sim.levels_between(6.5, 7.0);
        assert!(settled.iter().all(|&l| l <= before + chunk + cb), "{settled:?} vs {before}");
        assert_eq!(sim.underruns(), underruns_after_stall, "no underrun after the recovery");
        assert!(sim.pb.trimmed_frames() > 0);
        // No click anywhere: splices and the starve edges are all smooth.
        let slope = 2.0 * std::f32::consts::PI * 220.0 / SIM_RATE as f32 * 0.5;
        let steps = max_step(&sim.played);
        assert!(steps < 4.0 * slope, "largest step {steps} vs sine slope {slope}");
    }

    /// Steady playback, in every chunk / capture / callback relation the voice changer produces,
    /// never trims and never re-gates: the latency control only acts on real excess.
    #[test]
    fn steady_playback_never_trims() {
        for (chunk, cap, cb) in [(512, 480, 480), (160, 480, 480), (160, 960, 480), (1024, 480, 480), (240, 480, 441), (2048, 480, 1024)] {
            let mut sim = Sim::new(35.0, 70.0, chunk, cap, cb);
            sim.run_for(20 * 48_000);
            assert_eq!(sim.pb.trimmed_frames(), 0, "chunk {chunk} cap {cap} cb {cb}: {:?}", sim.pb.trimmed_by);
            assert_eq!(sim.pb.recoveries(), 0, "chunk {chunk} cap {cap} cb {cb}");
            assert_eq!(sim.underruns(), 0, "chunk {chunk} cap {cap} cb {cb}");
        }
    }

    /// Capture clock 0.2 % faster than the playback clock: the queue would creep up to the cap and
    /// be cut there. The drift trimmer keeps it near the steady level with crossfaded splices.
    #[test]
    fn clock_drift_is_trimmed_long_before_the_cap() {
        let mut sim = Sim::new(35.0, 150.0, 512, 480, 480);
        sim.cb_period = 481; // the device consumes 480 frames per 481 frames of capture time
        sim.run_for(60 * 48_000);
        let steady = sim.levels_between(1.0, 3.0);
        let (steady_min, steady_max) = (*steady.iter().min().unwrap(), *steady.iter().max().unwrap());
        let late = sim.levels_between(30.0, 60.0);
        let worst = *late.iter().max().unwrap();
        // 0.2 % of 60 s is 120 ms (5760 frames) of excess without trimming. With it, the smallest
        // level of a window is kept at most the drift slack (a chunk plus margin) above the gate
        // level (1680 frames), plus one window's drift: the steady range shifted by that much.
        let allowed = (1_680usize + 512 + 64).saturating_sub(steady_min) + 96 + 64;
        assert!(worst <= steady_max + allowed, "{worst} frames vs steady {steady_min}..{steady_max} (+{allowed})");
        assert!(worst < 7_200 / 2, "far below the 150 ms cap");
        assert!(sim.pb.trimmed_by[1] > 4_000, "{:?}", sim.pb.trimmed_by);
        assert_eq!(sim.pb.trimmed_by[2], 0, "never cut at the cap");
        assert_eq!(sim.pb.recoveries(), 0);
        let slope = 2.0 * std::f32::consts::PI * 220.0 / SIM_RATE as f32 * 0.5;
        assert!(max_step(&sim.played[48_000..]) < 4.0 * slope);
    }

    /// A trim is a whole-frame splice: the audio continuing the device's crossfades into the kept
    /// audio (equal power), channels never swap, and the dropped frames are counted.
    #[test]
    fn trims_are_crossfaded_whole_frame_splices() {
        let cb = Arc::new(AtomicUsize::new(0));
        // 1 kHz output: a 5 ms fade is 5 frames.
        let mut pb = OutputPlayback::new(0, 10_000, 2, cb).with_output_rate(1_000);
        let frames: Vec<f32> = (0..50).flat_map(|i| [i as f32, -(i as f32)]).collect();
        pb.push_interleaved(&frames, false, 0.0, 0, 0);
        pb.trim_frames(10);
        assert_eq!(pb.trimmed_frames(), 10);
        assert_eq!(pb.queue_len(), 40 * 2);
        let q: Vec<f32> = pb.deque.iter().copied().collect();
        for k in 0..5 {
            let (w_in, w_out) = super::xfade_weights(k, 5);
            let want = (10 + k) as f32 * w_in + k as f32 * w_out;
            assert!((q[2 * k] - want).abs() < 1e-5, "frame {k}: {} vs {want}", q[2 * k]);
            assert_eq!(q[2 * k + 1], -q[2 * k], "frame {k} keeps its channels");
        }
        assert_eq!(&q[10..], &frames[30..], "past the fade the kept audio is untouched");
    }

    /// A splice whose kept audio is not converted yet finishes its fade on the next push, and the
    /// ring never receives a frame before its fade is complete.
    #[test]
    fn a_splice_finishes_on_the_next_push() {
        let cb = Arc::new(AtomicUsize::new(0));
        let mut pb = OutputPlayback::new(0, 10_000, 1, cb).with_output_rate(1_000);
        let (mut prod, mut cons) = RingBuffer::<f32>::new(RING_CAP);
        // Playing: 10 frames in the ring.
        pb.push_interleaved(&[0.5; 10], false, 0.0, 0, 0);
        pb.flush_to_ring(&mut prod, 0);
        assert_eq!(ring_used(&prod), 10);
        pb.push_interleaved(&[1.0, 1.0, 1.0], false, 0.0, 0, 10);
        pb.trim_frames(3);
        assert_eq!(pb.queue_len(), 0);
        pb.flush_to_ring(&mut prod, 10);
        assert_eq!(ring_used(&prod), 10, "nothing leaves before its fade is complete");
        pb.push_interleaved(&[-1.0; 8], false, 0.0, 0, 10);
        let used = ring_used(&prod);
        pb.flush_to_ring(&mut prod, used);
        let got: Vec<f32> = std::iter::from_fn(|| cons.pop().ok()).skip(10).collect();
        assert_eq!(got.len(), 8);
        for (k, g) in got.iter().take(3).enumerate() {
            let (w_in, w_out) = super::xfade_weights(k, 3);
            assert!((g - (-w_in + w_out)).abs() < 1e-6, "{k}: {g}");
        }
        assert!(got[3..].iter().all(|&v| v == -1.0));
    }

    /// A device that ran dry while playing starts a recovery through the shared callback word (no
    /// extra wiring); dry callbacks during the start-up prefill wait do not.
    #[test]
    fn a_starved_device_starts_a_recovery() {
        let cb = Arc::new(AtomicUsize::new(0));
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut pb = OutputPlayback::new(64, 1_000, 1, cb.clone());
        let mut playout = RingPlayout::new(cons, 1, cb.clone());
        let mut data = vec![0.0f32; 32];
        playout.fill(&mut data); // before any audio: not starvation
        pb.push_interleaved(&[0.1; 64], false, 0.0, 0, ring_used(&prod));
        let used = ring_used(&prod);
        pb.flush_to_ring(&mut prod, used);
        assert!(!pb.is_recovering());
        playout.fill(&mut data);
        playout.fill(&mut data);
        pb.push_interleaved(&[0.1; 16], false, 0.0, 0, ring_used(&prod));
        assert!(!pb.is_recovering(), "exactly drained is not starved");
        let used = ring_used(&prod);
        pb.flush_to_ring(&mut prod, used);
        playout.fill(&mut data); // 16 of 32: dry
        pb.push_interleaved(&[0.1; 16], false, 0.0, 0, ring_used(&prod));
        assert!(pb.is_recovering());
        assert_eq!(pb.recoveries(), 1);
        assert_eq!(super::unpack_callback(cb.load(Ordering::Relaxed)), (32, 1));
    }

    /// While starved the held frame decays to silence instead of parking the speaker on a DC
    /// offset; playback then ramps back in from silence.
    #[test]
    fn the_hold_decays_to_silence_while_starved() {
        let (mut prod, cons) = RingBuffer::<f32>::new(RING_CAP);
        let mut playout = RingPlayout::new(cons, 1, Arc::new(AtomicUsize::new(0)));
        for _ in 0..64 {
            prod.push(0.5).unwrap();
        }
        let mut data = vec![0.0f32; 64];
        playout.fill(&mut data);
        assert_eq!(data[63], 0.5);
        let mut starved = vec![1.0f32; 256];
        playout.fill(&mut starved);
        assert!(starved[0] < 0.5 && starved[0] > 0.49, "{}", starved[0]);
        for w in starved.windows(2) {
            assert!(w[1] <= w[0]);
        }
        assert!(starved[HOLD_DECAY_FRAMES - 1].abs() < 1e-6);
        assert!(starved[HOLD_DECAY_FRAMES..].iter().all(|&v| v == 0.0));
    }

    // ---- adaptive cushion (P3B-H) ----

    /// A speech-like stream (pauses of 0.4 s every 1.2 s): the adaptive cushion starts at the user's
    /// 35 ms, sheds the silence once it has watched the stream, and holds a smaller queue than the
    /// fixed one, with no underrun, no trim of voiced audio and none of the user's speech lost.
    #[test]
    fn the_adaptive_cushion_sheds_silence_and_never_underruns() {
        let run = |cushion| {
            let mut sim = Sim::with_cushion(35.0, 70.0, 512, 480, 480, cushion);
            sim.quiet = Some((112, 38));
            sim.run_for(60 * 48_000);
            sim
        };
        let (fixed, adaptive) = (run(OutputCushion::Fixed), run(OutputCushion::Adaptive));
        assert_eq!(fixed.pb.adaptive_stats(), None);
        let (steady_fixed, steady_adaptive) = (median(fixed.levels_between(20.0, 60.0)), median(adaptive.levels_between(20.0, 60.0)));
        // The fixed queue holds its 35 ms gate level; the adaptive one the stream's floor: a 10 ms
        // callback / capture burst + 2 ms (the spread of this steady stream is a few ms).
        eprintln!("steady pre-push queue: fixed {steady_fixed} frames, adaptive {steady_adaptive} frames");
        assert!(steady_fixed >= 1_500, "the fixed queue holds its 35 ms: {steady_fixed} frames");
        assert!(steady_adaptive + 600 <= steady_fixed, "adaptive {steady_adaptive} frames vs fixed {steady_fixed}");
        assert!(steady_adaptive >= 12 * 48, "never below the floor: {steady_adaptive}");
        assert_eq!(adaptive.underruns(), 0);
        assert_eq!(adaptive.pb.recoveries(), 0, "no recovery was needed");
        let ((sheds, shed_samples), grows) = adaptive.pb.adaptive_stats().unwrap();
        assert!(sheds > 0 && shed_samples > 0 && grows == 0, "{sheds} {shed_samples} {grows}");
        // The cushion shown is the controller's target, below the user's prefill.
        assert!(adaptive.pb.cushion_samples() < 1_680 * SIM_CH);
        assert_eq!(fixed.pb.cushion_samples(), 1_680 * SIM_CH);
        // Only silence was shed: every frame of voiced audio pushed was played or is still queued.
        let played_voiced = adaptive.played.iter().filter(|v| v.abs() > 0.0).count() as u64;
        let queued = (adaptive.pb.queue_len() + adaptive.ring_used()) as u64 / SIM_CH as u64;
        assert!(played_voiced + queued + 480 >= adaptive.voiced_pushed, "{played_voiced} played + {queued} queued of {}", adaptive.voiced_pushed);
        assert_eq!(adaptive.pb.trimmed_frames(), 0, "silence is dropped from the chunk, nothing is spliced");
    }

    /// Jitter the stream really has (a worker 8 ms late every 40th capture period, i.e. a block-
    /// completing tick) is inside the window the controller watches: it sheds only down to what that
    /// dip leaves, so the jittered stream still never runs dry.
    #[test]
    fn the_adaptive_cushion_keeps_what_the_observed_lateness_needs() {
        let mut sim = Sim::with_cushion(35.0, 70.0, 512, 480, 480, OutputCushion::Adaptive);
        sim.quiet = Some((112, 38));
        sim.late = Some((40, 8 * 48));
        sim.run_for(120 * 48_000);
        assert_eq!(sim.underruns(), 0, "levels {:?}", sim.levels_between(100.0, 101.0));
        let low = *sim.levels_between(30.0, 120.0).iter().min().unwrap();
        assert!(low >= 480, "the queue never fell below one device callback: {low} frames");
        let ((sheds, _), _) = sim.pb.adaptive_stats().unwrap();
        assert!(sheds > 0);
    }

    /// Rare late ticks DEEPER than the drift trim's slack above its gate (14-25 ms late every 3 s, where
    /// the slack is one period plus a margin, about 12 ms), with the larger prefill a jittery system
    /// uses (60 ms). The controller sheds down to what the dips leave: its target (the valley, the
    /// level the worst dip lands on) becomes the floor, and the queue stands the dip's depth above it
    /// between dips. A drift trim keyed to that target trimmed the standing level away on the first
    /// dip-free window and the next late tick ran the device dry, until the target had grown back.
    /// The drift trim keeps the user's prefill as its ceiling in adaptive mode (the fixed cushion's
    /// own), so the controller's cushion stands.
    #[test]
    fn the_adaptive_cushion_survives_late_ticks_deeper_than_the_drift_slack() {
        let run = |prefill_ms: f64, late_ms: u64, keyed_to_target: bool| {
            let mut sim = Sim::with_cushion(prefill_ms, 2.0 * prefill_ms, 512, 480, 480, OutputCushion::Adaptive);
            sim.pb.drift_keyed_to_target = keyed_to_target;
            sim.quiet = Some((112, 38));
            sim.late = Some((300, late_ms * 48));
            sim.run_for(150 * 48_000);
            sim
        };
        // The drift trim keyed to the target (the old behaviour) runs the device dry: this test catches it.
        let old = run(60.0, 14, true);
        let (_, old_grows) = old.pb.adaptive_stats().unwrap();
        assert!(old.underruns() > 0 && old_grows > 0, "the trim keyed to the target should underrun here ({} underruns, trims {:?})", old.underruns(), old.pb.trimmed_by);
        assert!(old.pb.trimmed_by[1] > 0, "by trimming the standing level: {:?}", old.pb.trimmed_by);
        for (prefill_ms, late_ms) in [(60.0, 14), (60.0, 20), (60.0, 25), (80.0, 25)] {
            let sim = run(prefill_ms, late_ms, false);
            let ((sheds, _), grows) = sim.pb.adaptive_stats().unwrap();
            let label = format!("prefill {prefill_ms} ms, {late_ms} ms late every 3 s: {sheds} sheds, {grows} growths, trims {:?}", sim.pb.trimmed_by);
            assert_eq!(sim.underruns(), 0, "{label}");
            assert!(sheds > 0 && grows == 0, "{label}");
            assert_eq!(sim.pb.recoveries(), 0, "{label}");
            assert_eq!(sim.pb.trimmed_by[1], 0, "no drift trim: the cushion is the controller's: {label}");
            // Never above the fixed cushion: the queue stands the dip's depth over the floor between
            // dips, which is below the user's prefill plus the slack.
            let standing = median(sim.levels_between(60.0, 150.0));
            assert!(standing <= (prefill_ms * 48.0) as usize + 480, "{label}: standing {standing} frames");
            // The worst dip never took the queue under one device callback.
            let low = *sim.levels_between(30.0, 150.0).iter().min().unwrap();
            assert!(low >= 480, "{label}: the queue fell to {low} frames");
        }
    }

    /// A worker stall the window never saw runs the queue dry: the cushion then grows one step toward
    /// the user's prefill, re-gates once (the queue is empty anyway) and refills before playing, and
    /// the stream settles again without a second underrun. Nothing re-gates while playing normally.
    #[test]
    fn a_stall_grows_the_adaptive_cushion_once() {
        let mut sim = Sim::with_cushion(35.0, 70.0, 512, 480, 480, OutputCushion::Adaptive);
        sim.quiet = Some((112, 38));
        sim.run_for(30 * 48_000);
        assert_eq!(sim.underruns(), 0);
        let shrunk = sim.pb.cushion_samples();
        assert!(shrunk < 1_680 * SIM_CH);
        // A 0.5 s stall: the device starves.
        sim.stalled = true;
        sim.run_for(24_000);
        sim.end_stall(7_200);
        sim.run_for(10 * 48_000);
        let ((_, _), grows) = sim.pb.adaptive_stats().unwrap();
        assert_eq!(grows, 1);
        assert!(sim.pb.cushion_samples() > shrunk && sim.pb.cushion_samples() <= 1_680 * SIM_CH, "{} -> {}", shrunk, sim.pb.cushion_samples());
        let after_stall = sim.underruns();
        sim.run_for(60 * 48_000);
        assert_eq!(sim.underruns(), after_stall, "no further underrun once it settled");
        // The cooldown held shrinking off for the 30 s after the underrun: the cushion is still
        // above the shrunk one 10 s later, and the stream plays on.
        assert!(sim.levels_between(40.0, 45.0).iter().min().copied().unwrap_or(0) >= 480);
    }

    /// A burst worker: a template voice whose model runs once per 120 ms block on the processor. The
    /// eleven cheap ticks push a chunk each; the twelfth runs the call (about 90 ms here) and pushes
    /// nothing, then the worker catches up with a burst of chunks at one instant. The device runs dry
    /// during every call when the queue stands shorter than it, which starts a recovery that trims each
    /// catch-up back to the ring target; the next call dries it again, so the recovery never ends
    /// (P7-B: 38-50 % of the converted audio lost at 35 / 70 ms on the CPU rows).
    fn burst_worker(prefill_ms: f64, max_queue_ms: f64) -> Sim {
        let mut sim = Sim::new(prefill_ms, max_queue_ms, 512, 480, 480);
        sim.late = Some((12, 4_300));
        sim.run_for(60 * 48_000);
        sim
    }

    /// [`burst_worker`] with a model call of `call_ms` and a device callback of `cb` frames.
    fn burst_worker_with(prefill_ms: f64, max_queue_ms: f64, call_ms: f64, cb: usize) -> Sim {
        let mut sim = Sim::new(prefill_ms, max_queue_ms, 512, 480, cb);
        sim.late = Some((12, (call_ms * 48.0).round() as u64));
        sim.run_for(60 * 48_000);
        sim
    }

    /// Playback of a burst worker (`burst_worker_with`): the audio the trims discarded (share of the
    /// pushed) and the dry callbacks in 60 s.
    fn burst_outcome(prefill_ms: f64, max_queue_ms: f64, call_ms: f64, cb: usize) -> (f64, u64) {
        let sim = burst_worker_with(prefill_ms, max_queue_ms, call_ms, cb);
        (trimmed_share(&sim), sim.underruns())
    }

    /// The criterion that decides whether a typed output buffer is raised (`block_buffer::prefill_cannot_hold_call`:
    /// the stored prefill is shorter than the median model call) is the knee of the burst worker, on a
    /// 10 ms device callback and 512-frame chunks with a call every 120 ms (calls and times on the
    /// simulation's 5 ms grid): wherever it says the buffer cannot hold the call, the output loses at
    /// least 1 % of the audio and runs dry in at least 100 callbacks of the minute; wherever it does not,
    /// the output plays at least 98 % of it. So a raise never touches a buffer that plays, and a buffer
    /// it leaves alone loses at most the borderline 2 %.
    #[test]
    fn the_proof_criterion_is_the_knee_of_the_burst_worker() {
        use super::super::block_buffer::prefill_cannot_hold_call;
        let buffers = [(35.0, 70.0), (60.0, 120.0), (80.0, 150.0), (100.0, 125.0), (110.0, 136.0)];
        let (mut raised, mut kept) = (0, 0);
        for call in [70.0, 78.0, 90.0, 98.0, 110.0] {
            for (prefill, queue) in buffers {
                let (lost, dry) = burst_outcome(prefill, queue, call, 480);
                if prefill_cannot_hold_call(prefill, call) {
                    raised += 1;
                    assert!(lost >= 0.01 && dry >= 100, "{prefill}/{queue} with {call} ms calls: raised, but plays ({:.2} % lost, {dry} dry)", 100.0 * lost);
                } else {
                    kept += 1;
                    assert!(lost <= 0.02, "{prefill}/{queue} with {call} ms calls: kept, but loses {:.2} % ({dry} dry)", 100.0 * lost);
                }
            }
        }
        assert!(raised >= 10 && kept >= 10, "the matrix must reach both sides: {raised} raised, {kept} kept");
    }

    /// The floor has no term for the device's callback period: `OutputPlayback` stands the queue a
    /// callback above the gate by itself (`ring_target`, `drift_slack`), so 110 / 136 ms plays 98 and
    /// 110 ms calls on 10, 20, 30 and 40 ms callbacks with no dry callback and nothing lost, as it does
    /// on 10 ms ones. (A reviewer proposed adding the excess of a long period to the floor; this is the
    /// evidence that it is not needed.)
    #[test]
    fn a_longer_device_callback_needs_no_larger_floor() {
        for cb in [480usize, 960, 1440, 1920] {
            for call in [90.0, 98.0, 110.0] {
                let (lost, dry) = burst_outcome(110.0, 136.0, call, cb);
                assert!(lost <= 0.005 && dry == 0, "callback {} ms, {call} ms calls: {:.2} % lost, {dry} dry", cb / 48, 100.0 * lost);
            }
        }
    }

    /// A call that outruns the standing queue costs a dry hold and a recovery, not the whole output:
    /// 100 / 125 ms against calls 10 ms longer than the prefill loses about 1.4 % (371 dry callbacks a
    /// minute) where 110 / 136 ms loses nothing (the trap guard of `block_buffer` keeps the prefill at
    /// the first-call limit for calls a little longer than the cover, and this is what that costs).
    #[test]
    fn a_call_ten_ms_past_the_prefill_costs_a_little_audio_not_all_of_it() {
        let (lost, dry) = burst_outcome(100.0, 125.0, 110.0, 480);
        assert!((0.005..=0.03).contains(&lost), "{:.2} % lost", 100.0 * lost);
        assert!(dry > 100, "{dry} dry callbacks");
        let (lost, dry) = burst_outcome(110.0, 136.0, 110.0, 480);
        assert!(lost <= 0.005 && dry == 0, "a call as long as the prefill: {:.2} % lost, {dry} dry", 100.0 * lost);
    }

    /// Share of the pushed audio the trims discarded.
    fn trimmed_share(sim: &Sim) -> f64 {
        sim.pb.trimmed_frames() as f64 / sim.input_pos as f64
    }

    /// The queue cap is not the lever of a burst worker, the prefill is: at the shipped 35 / 70 ms the
    /// recovery (rule 0) discards a large share of the converted audio, and a cap of 600 ms changes
    /// nothing because the recovery trims to `ring_target()` (which follows the prefill) before the
    /// cap is ever reached. With the floor `block_buffer` gives such a voice (110 / 136 ms for 120 ms
    /// blocks) the same worker plays it all.
    #[test]
    fn a_burst_workers_loss_is_a_recovery_trim_that_only_the_prefill_removes() {
        let stock = burst_worker(35.0, 70.0);
        let wide_cap = burst_worker(35.0, 600.0);
        for (name, sim) in [("35/70", &stock), ("35/600", &wide_cap)] {
            let share = trimmed_share(sim);
            let [recovery, drift, cap] = sim.pb.trimmed_by;
            eprintln!("burst worker {name}: {:.1} % trimmed ({recovery} recovery, {drift} drift, {cap} cap), {} underruns, {} recoveries", 100.0 * share, sim.underruns(), sim.pb.recoveries());
            assert!(share >= 0.30, "{name}: only {:.1} % trimmed", 100.0 * share);
            assert!(recovery as f64 >= 0.95 * (recovery + drift + cap) as f64, "{name}: trims by rule {:?}", sim.pb.trimmed_by);
            assert!(sim.underruns() > 100, "{name}: the device runs dry during the calls ({} underruns)", sim.underruns());
        }
        let (a, b) = (trimmed_share(&stock), trimmed_share(&wide_cap));
        assert!((a - b).abs() < 0.03, "the cap changes nothing: {a:.3} vs {b:.3}");

        let floored = burst_worker(110.0, 136.0);
        let share = trimmed_share(&floored);
        eprintln!("burst worker 110/136: {:.2} % trimmed, {} underruns", 100.0 * share, floored.underruns());
        assert!(share <= 0.02, "110/136 plays it all: {:.2} % trimmed ({:?})", 100.0 * share, floored.pb.trimmed_by);
        assert!(1.0 - share >= 0.98);
        assert!(floored.underruns() <= 1, "no dry callbacks once the queue covers the call: {}", floored.underruns());
        assert_eq!(floored.pb.recoveries(), 0, "no recovery at all");
    }

    /// The user's prefill is the cap and the floor of a voice that needs a larger buffer is
    /// respected: the adaptive queue is never above the fixed one and never below the floor.
    #[test]
    fn the_adaptive_cushion_respects_its_cap_and_a_voice_floor() {
        let mut sim = Sim::with_cushion(35.0, 70.0, 512, 480, 480, OutputCushion::Adaptive);
        sim.pb.set_cushion_floor(30 * 48 * SIM_CH);
        sim.quiet = Some((112, 38));
        sim.run_for(60 * 48_000);
        let low = *sim.levels_between(20.0, 60.0).iter().min().unwrap();
        assert!(low >= 30 * 48, "a voice's 30 ms minimum holds: {low} frames");
        assert!(sim.pb.cushion_samples() <= 1_680 * SIM_CH);
        assert_eq!(sim.underruns(), 0);
    }
}
