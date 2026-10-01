//! A sidecar v2 voice as a `VoiceConversionModel` (P5-E).
//!
//! The sidecar converts whole *frames* (its own block, e.g. 120 ms for X-VC, 180 ms for Seed-VC),
//! but the stream harness runs this model in fixed-block mode with a *slot* of about 20 ms (a
//! divisor of the frame at both rates) as the block. Every `convert` call takes one slot of input
//! and plays one slot of output, and never waits for the sidecar:
//!
//! 1. it takes the replies that arrived since the last call (non-blocking),
//! 2. on the call that completes a frame, queues that frame with its sequence number and the
//!    voice's pitch (non-blocking),
//! 3. plays one slot from the [`Playout`]: a slot of a converted frame, a gap, or a failure (a
//!    frame not answered within the latency budget, refused by the sidecar or lost with the
//!    connection),
//! 4. returns the newest samples of its own output timeline (slot plus SOLA tail; the timeline is
//!    contiguous, so the harness's crossfade joins identical audio).
//!
//! Playing slots instead of whole frames means a converted frame starts playing on the first slot
//! after it is back: the delay is the frame plus the sidecar's compute time (rounded up to a slot),
//! not two frames.
//!
//! The playout is a jitter buffer measured in slots. It holds a *lag* (slots between the newest
//! input and the slot played), set from the first frame's round trip plus a little slack. A frame
//! that is not back when due is waited for (a gap; the lag grows by one slot) until the lag reaches
//! the budget, then it is given up: the call returns an error naming the timeout (the worker mutes
//! the block), the sidecar is told to drop everything older, and a late reply is discarded as
//! stale. When the sidecar has answered faster than the lag (with slack) for a whole window, the
//! target lag drops to what that window needed plus the slack, and the lag follows by skipping
//! quiet slots (never in the middle of speech). So the voice runs at the smallest delay the sidecar
//! sustains, returns to it within a window or two after a stall, the budget bounds it, and stale
//! audio never plays. Gaps and skips fade (5 ms) on both sides. After a long pause the stream
//! starts over from the starting lag (what a stall grew it to is forgotten).
//!
//! A single-stream sidecar (StreamVoiceAnon) converts for one voice at a time: the frames of the
//! voice being replaced, for the moment before a switch, come back `superseded` and play as
//! silence without an error.

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serde_json::{json, Map, Value};

use super::recipes::HelperLease;
use crate::services::neural_audio::sidecar_bridge::v2::{Event, Session, SidecarClient, FLAG_DISCONTINUITY};
use crate::services::neural_audio::vc::families::sidecar_v2::in_flight_window;
use crate::services::neural_audio::vc::traits::{ConversionCtx, VoiceControls, VoiceConversionModel};
use crate::services::resources::VramClaim;

/// Preferred slot length.
const SLOT_MS: f64 = 20.0;
/// Shortest slot (a frame that divides no better plays in whole frames).
const MIN_SLOT_MS: f64 = 8.0;
/// Fade at the edges of a gap or a skip.
const FADE_MS: f64 = 5.0;
/// Margin kept above the lag the sidecar needs (scheduling jitter).
const SLACK_MS: f64 = 30.0;
/// A slot whose RMS is below this (-45 dBFS) may be skipped to lower the lag.
const QUIET_RMS: f32 = 0.005_623;
/// Replies observed before the lag may be lowered (and ticks since the last change).
const WINDOW_MS: f64 = 4_000.0;
/// Margin over the sidecar's latency estimate for the starting lag.
const START_MARGIN_MS: f64 = 15.0;
/// A pause between calls longer than this (plus the budget) starts a new stream: everything
/// still outstanding is stale (a hot voice that was not playing, a stopped device).
const RESTART_AFTER: Duration = Duration::from_secs(2);
/// Replies whose length is off by at most this fraction (or 16 samples) are padded or cut.
const LENGTH_TOLERANCE: f64 = 0.02;
/// Gap between two warnings about the same kind of problem.
const LOG_EVERY: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------------------------
// Playout (pure: tested without a sidecar)
// ---------------------------------------------------------------------------------------------

/// What one tick plays.
#[derive(Debug, PartialEq)]
pub(crate) enum Play {
    /// One slot of a converted frame (`after_skip`: one quiet slot was dropped before it).
    Audio { pcm: Vec<f32>, after_skip: bool },
    /// Silence: the pipeline is filling, the due frame is late (the lag grows), or the rest of a
    /// frame whose failure was already reported.
    Gap,
    /// The due frame will not come (`timeout`: not answered within the budget; the sidecar may
    /// drop every frame before `cancel_before`). Reported once per frame.
    Failed { reason: String, timeout: bool, cancel_before: Option<u64> },
}

/// Counters (debug logs, tests).
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct PlayoutStats {
    /// Slots played.
    pub played: u64,
    /// Silent slots after playback started (late frames).
    pub gaps: u64,
    /// Frames given up at the budget.
    pub timeouts: u64,
    /// Frames the sidecar refused or that could not be sent.
    pub failed: u64,
    /// Replies that came after their frame was given up (discarded).
    pub stale: u64,
    /// Quiet slots dropped to lower the lag.
    pub skipped: u64,
    pub grown: u64,
    pub lowered: u64,
    /// Frames a single-stream sidecar left unconverted while another voice held it (silence).
    pub superseded: u64,
}

/// Why a frame will not come.
#[derive(Clone, Debug, PartialEq)]
struct Failure {
    reason: String,
    /// Given up at the budget.
    timeout: bool,
    /// Plays as silence without an error (another voice holds a single-stream sidecar).
    quiet: bool,
}

impl Failure {
    fn new(reason: impl Into<String>, timeout: bool) -> Failure {
        Failure {
            reason: reason.into(),
            timeout,
            quiet: false,
        }
    }
}

/// The jitter buffer (see the module docs). Ticks are `convert` calls; frame `f` is slots
/// `f * k .. (f + 1) * k` and is sent on tick `(f + 1) * k - 1`.
pub(crate) struct Playout {
    k: u64,
    /// The tick in progress.
    tick: u64,
    next_play: u64,
    /// Frames sent (or refused) so far: the next frame's index.
    frames: u64,
    /// The lag (slots) the stream holds.
    target: u64,
    /// The lag a stream starts from (the estimate; [`Self::restart`] goes back to it).
    start_target: u64,
    /// Lag at which a frame's first slot is given up (the budget).
    max_lag: u64,
    slack: u64,
    started: bool,
    /// Converted slots waiting for their turn.
    ready: BTreeMap<u64, Vec<f32>>,
    /// The lag each answered frame needed (for the start).
    need_of: BTreeMap<u64, u64>,
    /// Frames that will not come, with why.
    failed: BTreeMap<u64, Failure>,
    /// The last frame whose failure was reported.
    reported: Option<u64>,
    in_flight: BTreeSet<u64>,
    /// Lag each recent reply needed.
    needs: VecDeque<u64>,
    window: usize,
    since_change: usize,
    pub stats: PlayoutStats,
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / x.len() as f64).sqrt() as f32
}

impl Playout {
    /// `k` slots per frame; lags in slots.
    pub(crate) fn new(k: u64, target: u64, max_lag: u64, slack: u64, window: usize) -> Playout {
        let k = k.max(1);
        let max_lag = max_lag.max(k);
        let target = target.clamp(1, max_lag);
        Playout {
            k,
            tick: 0,
            next_play: 0,
            frames: 0,
            target,
            start_target: target,
            max_lag,
            slack: slack.max(1),
            started: false,
            ready: BTreeMap::new(),
            need_of: BTreeMap::new(),
            failed: BTreeMap::new(),
            reported: None,
            in_flight: BTreeSet::new(),
            needs: VecDeque::new(),
            window: window.max(4),
            since_change: 0,
            stats: PlayoutStats::default(),
        }
    }

    pub(crate) fn target(&self) -> u64 {
        self.target
    }

    pub(crate) fn max_lag(&self) -> u64 {
        self.max_lag
    }

    /// Frames in flight.
    pub(crate) fn in_flight(&self) -> usize {
        self.in_flight.len()
    }

    /// The next frame's index (sequence numbers widen against it).
    pub(crate) fn next_frame(&self) -> u64 {
        self.frames
    }

    /// The frame this tick completes, if it completes one.
    pub(crate) fn completes_frame(&self) -> Option<u64> {
        (self.tick % self.k == self.k - 1).then_some(self.tick / self.k)
    }

    /// Frame `frame` went out (`Ok`) or could not (`Err`: it fails when it is due).
    pub(crate) fn sent(&mut self, frame: u64, result: Result<(), String>) {
        match result {
            Ok(()) => {
                self.in_flight.insert(frame);
            }
            Err(reason) => {
                self.failed.insert(frame, Failure::new(reason, false));
            }
        }
        self.frames = self.frames.max(frame + 1);
    }

    fn note_need(&mut self, need: u64) {
        self.needs.push_back(need);
        while self.needs.len() > self.window {
            self.needs.pop_front();
        }
    }

    /// A reply for `frame` (`k` slots of audio), observed during the tick in progress; false when
    /// it is stale (given up already, or never asked for).
    pub(crate) fn reply(&mut self, frame: u64, pcm: Vec<f32>) -> bool {
        if !self.in_flight.remove(&frame) {
            self.stats.stale += 1;
            return false;
        }
        let first = frame * self.k;
        let need = self.tick.saturating_sub(first);
        self.note_need(need);
        self.need_of.insert(frame, need);
        let slot = (pcm.len() / self.k as usize).max(1);
        for (j, chunk) in pcm.chunks(slot).take(self.k as usize).enumerate() {
            self.ready.insert(first + j as u64, chunk.to_vec());
        }
        true
    }

    /// `frame` will not be answered.
    pub(crate) fn fail(&mut self, frame: u64, reason: String) {
        if self.in_flight.remove(&frame) {
            self.failed.insert(frame, Failure::new(reason, false));
        }
    }

    /// `frame` was left unconverted because another voice holds a single-stream sidecar: it plays
    /// as silence, without an error.
    pub(crate) fn superseded(&mut self, frame: u64) {
        if self.in_flight.remove(&frame) {
            self.stats.superseded += 1;
            self.failed.insert(
                frame,
                Failure {
                    quiet: true,
                    ..Failure::new("superseded", false)
                },
            );
        }
    }

    /// Every frame in flight is lost (a dropped connection).
    pub(crate) fn lose_in_flight(&mut self, reason: &str) {
        for f in std::mem::take(&mut self.in_flight) {
            self.failed.insert(f, Failure::new(reason, false));
        }
    }

    /// Start over from the frame being captured (after a long pause): what is outstanding is stale,
    /// and the lag goes back to the starting estimate (a lag a stall grew to in the old stream is
    /// not the new stream's; the first reply sets it again).
    pub(crate) fn restart(&mut self) {
        self.in_flight.clear();
        self.ready.clear();
        self.need_of.clear();
        self.failed.clear();
        self.reported = None;
        self.next_play = self.tick / self.k * self.k;
        self.started = false;
        self.target = self.start_target;
        self.needs.clear();
        self.since_change = 0;
    }

    /// When every reply of a full window needed less than the lag (with slack), lower the target to
    /// what the slowest of them needed plus the slack, in one step (the lag follows by skipping
    /// quiet slots), so the voice returns to its steady delay within a window or two after a stall
    /// instead of one slot per window.
    fn maybe_lower(&mut self) {
        if self.target <= 1 || self.needs.len() < self.window || self.since_change < self.window {
            return;
        }
        let want = (self.needs.iter().copied().max().unwrap_or(0) + self.slack).max(1);
        if want < self.target {
            self.target = want;
            self.since_change = 0;
            self.stats.lowered += 1;
        }
    }

    /// The slot of this tick (after this tick's frame, if any, was [`sent`](Self::sent)); ends
    /// the tick.
    pub(crate) fn play(&mut self) -> Play {
        let t = self.tick;
        self.tick += 1;
        self.since_change += 1;
        self.maybe_lower();
        let k = self.k;
        let mut p = self.next_play;
        let lag = t.saturating_sub(p);
        let f = p / k;
        if !self.started {
            if let Some(&need) = self.need_of.get(&f).filter(|_| self.ready.contains_key(&p)) {
                // Start `slack` above what the first frame needed (lowered later, on a quiet slot,
                // if the sidecar proves faster).
                let start = self.target.max(need + self.slack).min(self.max_lag);
                if lag < start {
                    self.target = start;
                    return Play::Gap;
                }
                self.started = true;
                self.target = start;
                self.since_change = 0;
            } else if !self.failed.contains_key(&f) && lag < self.max_lag {
                return Play::Gap;
            }
        }
        let mut after_skip = false;
        if self.started && lag > self.target && self.ready.get(&p).is_some_and(|x| rms(x) < QUIET_RMS) && self.ready.contains_key(&(p + 1)) {
            self.ready.remove(&p);
            if p % k == k - 1 {
                // The skipped slot ended its frame.
                self.need_of.remove(&(p / k));
            }
            p += 1;
            self.next_play = p;
            after_skip = true;
            self.stats.skipped += 1;
        }
        let f = p / k;
        if let Some(pcm) = self.ready.remove(&p) {
            self.next_play = p + 1;
            if p % k == k - 1 {
                self.need_of.remove(&f);
            }
            self.stats.played += 1;
            return Play::Audio { pcm, after_skip };
        }
        if let Some(failure) = self.failed.get(&f).cloned() {
            return self.play_failed(p, f, failure, None);
        }
        let pending = self.in_flight.contains(&f) || f >= self.frames;
        if pending && lag < self.max_lag {
            // Late: wait for it (a gap now, one slot more delay from here on).
            if self.started {
                self.target = (lag + 1).min(self.max_lag);
                self.since_change = 0;
                self.stats.grown += 1;
                self.stats.gaps += 1;
            }
            return Play::Gap;
        }
        // Not back within the budget (or never sent): give the frame up.
        let was_in_flight = self.in_flight.remove(&f);
        self.stats.timeouts += 1;
        self.note_need(self.max_lag + 1);
        let failure = Failure::new("timeout", was_in_flight);
        self.failed.insert(f, failure.clone());
        self.play_failed(p, f, failure, Some(f + 1))
    }

    /// A slot of a frame that will not come: reported on its first slot, silent after that (and
    /// silent throughout for a quiet failure).
    fn play_failed(&mut self, p: u64, f: u64, failure: Failure, cancel_before: Option<u64>) -> Play {
        self.next_play = p + 1;
        if p % self.k == self.k - 1 {
            self.failed.remove(&f);
            self.need_of.remove(&f);
        }
        if self.reported == Some(f) || failure.quiet {
            return Play::Gap;
        }
        self.reported = Some(f);
        if failure.reason != "timeout" {
            self.stats.failed += 1;
        }
        Play::Failed {
            reason: failure.reason,
            timeout: failure.timeout,
            cancel_before,
        }
    }
}

/// Slots per frame: the divisor of both frame lengths whose slot is nearest [`SLOT_MS`] and at
/// least [`MIN_SLOT_MS`] (1 when none: whole frames).
pub(crate) fn slots_per_frame(frame_in: usize, frame_out: usize, rate_in: u32) -> u64 {
    let g = {
        let (mut a, mut b) = (frame_in as u64, frame_out as u64);
        while b != 0 {
            (a, b) = (b, a % b);
        }
        a.max(1)
    };
    let ms = |k: u64| frame_in as f64 / k as f64 * 1000.0 / f64::from(rate_in.max(1));
    (1..=g)
        .filter(|k| g % k == 0 && ms(*k) >= MIN_SLOT_MS)
        .min_by(|a, b| (ms(*a) - SLOT_MS).abs().total_cmp(&(ms(*b) - SLOT_MS).abs()))
        .unwrap_or(1)
}

// ---------------------------------------------------------------------------------------------
// The model
// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Link {
    Up,
    /// Lost; the client reconnects in the background.
    Down(String),
    /// Gone for good.
    Failed(String),
}

/// See the module docs.
pub(crate) struct SidecarModel {
    client: SidecarClient,
    label: String,
    session: Session,
    playout: Playout,
    /// Input and output samples per slot.
    slot_in: usize,
    slot_out: usize,
    max_in_flight: usize,
    budget_ms: f64,
    /// Contiguous converted audio at the output rate (the newest samples are returned).
    timeline: Vec<f32>,
    keep: usize,
    fade: usize,
    last_gap: bool,
    discontinuity: bool,
    link: Link,
    /// Parameters as the sidecar should have them, and whether they still need sending.
    params: Map<String, Value>,
    params_dirty: bool,
    last_call: Option<Instant>,
    logged: Option<Instant>,
    last_stats: PlayoutStats,
    /// A pitch shift the sidecar ignores was logged.
    pitch_noted: bool,
    _lease: Option<Arc<HelperLease>>,
    _vram: Option<VramClaim>,
}

/// Sequence numbers are u32 on the wire: the u64 frame index nearest at or below `next`.
fn widen(seq: u32, next: u64) -> u64 {
    let cand = (next & !0xFFFF_FFFF) | u64::from(seq);
    if cand > next && cand >= 1 << 32 {
        cand - (1 << 32)
    } else {
        cand
    }
}

/// Raised-cosine gain of sample `i` of an `n`-sample fade-in.
fn fade_in_gain(i: usize, n: usize) -> f32 {
    let x = (i as f32 + 0.5) / n.max(1) as f32;
    0.5 - 0.5 * (std::f32::consts::PI * x).cos()
}

impl SidecarModel {
    /// Wrap a connected client. `budget_ms` bounds how long a frame may take; `device_block_ms`
    /// (the capture block) adds the burst slack of a capture block longer than a slot.
    pub(crate) fn new(
        client: SidecarClient,
        label: impl Into<String>,
        budget_ms: f64,
        device_block_ms: f64,
        lease: Option<Arc<HelperLease>>,
        vram: Option<VramClaim>,
    ) -> SidecarModel {
        let session = client.session().clone();
        let k = slots_per_frame(session.frame_in, session.frame_out, session.input_rate);
        let slot_in = session.frame_in / k as usize;
        let slot_out = session.frame_out / k as usize;
        let slot_ms = (slot_in as f64 * 1000.0 / f64::from(session.input_rate.max(1))).max(1.0);
        let budget_slots = ((budget_ms / slot_ms).floor() as u64).max(1);
        let max_lag = k - 1 + budget_slots;
        // Ticks come in bursts when the capture block is longer than a slot.
        let burst = ((device_block_ms / slot_ms).ceil() as u64).max(1) - 1;
        let slack = (SLACK_MS / slot_ms).ceil() as u64;
        let compute = ((session.latency_ms.unwrap_or(0.0) + START_MARGIN_MS) / slot_ms).ceil() as u64;
        let playout = Playout::new(k, k - 1 + compute + burst + slack, max_lag, slack, ((WINDOW_MS / slot_ms).ceil() as usize).max(8));
        // From the frame the server chose (not the one the hello proposed), capped only by a limit
        // the server states itself.
        let own = in_flight_window(budget_ms, session.frame_ms());
        let max_in_flight = session.max_in_flight.map_or(own, |m| m.min(own)).max(1);
        let fade = ((FADE_MS * f64::from(session.output_rate) / 1000.0).round() as usize).clamp(1, (slot_out / 2).max(1));
        let keep = session.frame_out * 2 + slot_out * 4;
        let mut params = Map::new();
        for (key, v) in [("formant_semitones", json!(0.0)), ("speaker_id", json!(0))] {
            if session.params.iter().any(|p| p == key) {
                params.insert(key.into(), v);
            }
        }
        SidecarModel {
            client,
            label: label.into(),
            playout,
            slot_in,
            slot_out,
            max_in_flight,
            budget_ms,
            timeline: vec![0.0; keep],
            keep,
            fade,
            last_gap: true,
            // The first frame starts a stream.
            discontinuity: true,
            link: Link::Up,
            params,
            params_dirty: false,
            last_call: None,
            logged: None,
            last_stats: PlayoutStats::default(),
            pitch_noted: false,
            session,
            _lease: lease,
            _vram: vram,
        }
    }

    /// The delay the voice starts with (16 kHz samples): the starting lag plus the sidecar's own
    /// algorithmic delay.
    pub(crate) fn start_delay_16k(&self) -> usize {
        let s = &self.session;
        let lag_in = self.playout.target() as f64 * self.slot_in as f64;
        let algo_in = s.delay_out as f64 * f64::from(s.input_rate) / f64::from(s.output_rate.max(1));
        ((lag_in + algo_in) * 16_000.0 / f64::from(s.input_rate.max(1))).round() as usize
    }

    /// Set a parameter the sidecar understands (sent before the next frame when it changed).
    fn set_param(&mut self, key: &str, value: Value) {
        if !self.session.params.iter().any(|p| p == key) || self.params.get(key) == Some(&value) {
            return;
        }
        self.params.insert(key.into(), value);
        self.params_dirty = true;
    }

    fn flush_params(&mut self) {
        if self.params_dirty && self.link == Link::Up && self.session.protocol >= 2 && self.client.set_params(self.params.clone()) {
            self.params_dirty = false;
        }
    }

    /// A reply cut or padded to the frame, or why it cannot be used.
    fn fit(&self, mut pcm: Vec<f32>) -> Result<Vec<f32>, String> {
        let want = self.session.frame_out;
        if pcm.iter().any(|x| !x.is_finite()) {
            return Err("the sidecar returned non-finite audio (NaN/inf)".into());
        }
        let tolerance = ((want as f64 * LENGTH_TOLERANCE) as usize).max(16);
        if pcm.len().abs_diff(want) > tolerance {
            return Err(format!("the sidecar answered {} samples for a frame of {want}", pcm.len()));
        }
        pcm.resize(want, 0.0);
        Ok(pcm)
    }

    /// Take everything the I/O thread reported.
    fn poll(&mut self) {
        while let Some(event) = self.client.try_event() {
            match event {
                Event::Audio { seq, pcm, .. } => {
                    let f = widen(seq, self.playout.next_frame());
                    match self.fit(pcm) {
                        Ok(pcm) => {
                            self.playout.reply(f, pcm);
                        }
                        Err(e) => self.playout.fail(f, e),
                    }
                }
                Event::Dropped { seq, reason } => {
                    let f = widen(seq, self.playout.next_frame());
                    self.playout.fail(f, reason);
                }
                Event::Superseded { seq } => {
                    let f = widen(seq, self.playout.next_frame());
                    self.playout.superseded(f);
                }
                Event::Lost(reason) => {
                    let msg = format!("the {} sidecar connection was lost ({reason}); reconnecting", self.label);
                    self.playout.lose_in_flight(&msg);
                    self.link = Link::Down(msg);
                }
                Event::Restored(session) => {
                    tracing::info!("voice changer: the {} sidecar is back (model {:?})", self.label, session.model);
                    self.link = Link::Up;
                    self.discontinuity = true;
                    self.params_dirty = !self.params.is_empty();
                }
                Event::Failed(reason) => {
                    let msg = format!("the {} sidecar stopped: {reason}", self.label);
                    self.playout.lose_in_flight(&msg);
                    self.link = Link::Failed(msg);
                }
                Event::Notice(n) => tracing::debug!("voice changer: {} sidecar: {n}", self.label),
            }
        }
    }

    /// Queue the frame that ends with this call's input.
    fn send_frame(&mut self, frame: u64, ctx: &ConversionCtx<'_>) {
        let frame_in = self.session.frame_in;
        let src = &ctx.mono_16k[ctx.mono_16k.len().saturating_sub(frame_in)..];
        let mut pcm = Vec::with_capacity(frame_in);
        pcm.resize(frame_in - src.len(), 0.0);
        pcm.extend_from_slice(src);
        let sent = match &self.link {
            Link::Up if self.playout.in_flight() >= self.max_in_flight => Err(format!(
                "the {} sidecar is {} frames behind; this block is muted",
                self.label,
                self.playout.in_flight()
            )),
            Link::Up => {
                let flags = if self.discontinuity { FLAG_DISCONTINUITY } else { 0 };
                let pitch = if self.session.pitch() {
                    ctx.pitch_shift_semitones
                } else {
                    if ctx.pitch_shift_semitones != 0.0 && !self.pitch_noted {
                        self.pitch_noted = true;
                        tracing::info!(
                            "voice changer: the {} sidecar does not apply a pitch shift; {:+} semitones are ignored",
                            self.label,
                            ctx.pitch_shift_semitones
                        );
                    }
                    0.0
                };
                self.client.send_audio(frame as u32, flags, pitch, pcm).map(|()| self.discontinuity = false)
            }
            Link::Down(r) | Link::Failed(r) => Err(r.clone()),
        };
        if sent.is_err() {
            // The sidecar misses this frame: the next one does not continue the last it heard.
            self.discontinuity = true;
        }
        self.playout.sent(frame, sent);
    }

    fn trim(&mut self) {
        if self.timeline.len() > 2 * self.keep {
            let drop = self.timeline.len() - self.keep;
            self.timeline.drain(..drop);
        }
    }

    /// Fade out the newest samples (the end of the audio before a gap or a skip).
    fn fade_tail(&mut self) {
        if !self.last_gap {
            let n = self.timeline.len();
            let f = self.fade.min(n);
            for (i, s) in self.timeline[n - f..].iter_mut().enumerate() {
                *s *= 1.0 - fade_in_gain(i, f);
            }
            self.last_gap = true;
        }
    }

    fn push_audio(&mut self, mut pcm: Vec<f32>) {
        pcm.resize(self.slot_out, 0.0);
        if self.last_gap {
            let f = self.fade.min(pcm.len());
            for (i, s) in pcm[..f].iter_mut().enumerate() {
                *s *= fade_in_gain(i, f);
            }
        }
        self.timeline.extend_from_slice(&pcm);
        self.last_gap = false;
        self.trim();
    }

    fn push_gap(&mut self) {
        self.fade_tail();
        let n = self.timeline.len() + self.slot_out;
        self.timeline.resize(n, 0.0);
        self.trim();
    }

    /// The newest `n` samples of the timeline.
    fn newest(&self, n: usize) -> Vec<f32> {
        let len = self.timeline.len();
        if len >= n {
            self.timeline[len - n..].to_vec()
        } else {
            let mut v = vec![0.0; n - len];
            v.extend_from_slice(&self.timeline);
            v
        }
    }

    fn failure_message(&self, reason: String, timeout: bool) -> String {
        if timeout {
            format!(
                "the {} sidecar did not answer within {:.0} ms (the latency budget); this block is muted",
                self.label, self.budget_ms
            )
        } else if reason == "timeout" {
            format!("the {} sidecar frame was lost; this block is muted", self.label)
        } else {
            reason
        }
    }

    /// Summaries for the log when something went wrong since the last one.
    fn maybe_log(&mut self) {
        let now = Instant::now();
        if self.logged.is_some_and(|t| now.duration_since(t) < LOG_EVERY) {
            return;
        }
        let s = &self.playout.stats;
        let l = &self.last_stats;
        if s.timeouts == l.timeouts && s.failed == l.failed && s.gaps == l.gaps && s.stale == l.stale {
            return;
        }
        tracing::warn!(
            "voice changer: {} sidecar since the last report: {} late slots, {} timeouts, {} refused/lost, {} stale replies; lag {} of {} slots ({:.0} ms frames)",
            self.label,
            s.gaps - l.gaps,
            s.timeouts - l.timeouts,
            s.failed - l.failed,
            s.stale - l.stale,
            self.playout.target(),
            self.playout.max_lag(),
            self.session.frame_ms()
        );
        self.last_stats = s.clone();
        self.logged = Some(now);
    }

    /// Output samples owed for `new_frames` input samples (the trait's rounding).
    fn out_len(&self, new_frames: usize) -> usize {
        let (i, o) = (u64::from(self.session.input_rate.max(1)), u64::from(self.session.output_rate));
        ((new_frames as u64 * o + i / 2) / i) as usize
    }

    #[cfg(test)]
    pub(crate) fn stats(&self) -> &PlayoutStats {
        &self.playout.stats
    }

    #[cfg(test)]
    pub(crate) fn max_in_flight(&self) -> usize {
        self.max_in_flight
    }

    /// One line about how the voice ran (logged when it unloads).
    fn summary(&self) -> String {
        let s = &self.playout.stats;
        let slot_ms = self.slot_in as f64 * 1000.0 / f64::from(self.session.input_rate.max(1));
        format!(
            "{} slots played, {} late slots, {} frames past the {:.0} ms budget, {} refused or lost, {} while another voice held the sidecar, {} stale replies, {} quiet slots skipped; lag {} slots ({:.0} ms) of at most {}",
            s.played,
            s.gaps,
            s.timeouts,
            self.budget_ms,
            s.failed,
            s.superseded,
            s.stale,
            s.skipped,
            self.playout.target(),
            self.playout.target() as f64 * slot_ms,
            self.playout.max_lag()
        )
    }
}

impl Drop for SidecarModel {
    fn drop(&mut self) {
        if self.playout.stats.played > 0 || self.playout.stats.timeouts > 0 {
            tracing::info!("voice changer: {} sidecar voice unloaded: {}", self.label, self.summary());
        }
    }
}

impl VoiceConversionModel for SidecarModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let now = Instant::now();
        let pause = RESTART_AFTER + Duration::from_secs_f64(self.budget_ms.max(0.0) / 1000.0);
        if self.last_call.is_some_and(|t| now.duration_since(t) > pause) {
            // Nothing outstanding can still be on time.
            self.playout.restart();
            self.discontinuity = true;
            if self.link == Link::Up {
                self.client.cancel_before(self.playout.next_frame() as u32);
            }
        }
        self.last_call = Some(now);
        self.poll();
        self.set_param("speaker_id", json!(ctx.speaker_id));
        self.flush_params();
        if let Some(frame) = self.playout.completes_frame() {
            self.send_frame(frame, ctx);
        }
        let result = match self.playout.play() {
            Play::Audio { pcm, after_skip } => {
                if after_skip {
                    self.fade_tail();
                }
                self.push_audio(pcm);
                Ok(())
            }
            Play::Gap => {
                self.push_gap();
                Ok(())
            }
            Play::Failed { reason, timeout, cancel_before } => {
                if let (Some(c), Link::Up) = (cancel_before, &self.link) {
                    self.client.cancel_before(c as u32);
                }
                self.push_gap();
                Err(self.failure_message(reason, timeout))
            }
        };
        self.maybe_log();
        let out = self.newest(self.out_len(ctx.new_frames));
        result.map(|()| out)
    }

    fn input_sample_rate(&self) -> u32 {
        self.session.input_rate
    }

    fn output_sample_rate(&self) -> u32 {
        self.session.output_rate
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        self.set_param("formant_semitones", json!(controls.formant_semitones));
        self.set_param("noise_scale", json!(controls.noise_scale));
    }

    fn reset_stream(&mut self) {
        self.discontinuity = true;
    }

    fn fixed_block(&self) -> Option<usize> {
        Some(self.slot_in)
    }

    fn configure_stream(&mut self, block: usize, want: usize, _extra: usize) -> Result<usize, String> {
        if block != self.slot_in {
            return Err(format!("the sidecar voice plays {}-sample slots, not {block}", self.slot_in));
        }
        // The sidecar keeps its own context: the harness only holds the frame being captured (sent
        // from the context when it completes) and the SOLA tail it asks for.
        self.keep = self.keep.max(self.out_len(want) + self.session.frame_out);
        if self.timeline.len() < self.keep {
            let pad = self.keep - self.timeline.len();
            self.timeline.splice(0..0, std::iter::repeat_n(0.0, pad));
        }
        Ok(want.max(self.session.frame_in))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::sidecar_bridge::test_server::{Behavior, TestServer};
    use crate::services::neural_audio::sidecar_bridge::v2::ClientConfig;
    use crate::services::neural_audio::vc::stream::VcStreamHarness;

    // --- Playout ------------------------------------------------------------------------------

    /// Drive a playout for `ticks` ticks (from where it is) where frame `f` comes back `need(f)`
    /// ticks after it was sent (`None` = never); returns what each tick played (`'A'` audio, `'.'`
    /// gap, `'X'` failure).
    fn simulate(p: &mut Playout, ticks: u64, need: impl Fn(u64) -> Option<u64>, loud: bool) -> String {
        let mut out = String::new();
        let mut due: Vec<(u64, u64)> = Vec::new();
        let k = p.k;
        for _ in 0..ticks {
            let t = p.tick;
            // Replies observed at the start of tick t.
            due.retain(|&(at, f)| {
                if at <= t {
                    p.reply(f, vec![if loud { 0.5 } else { 0.0 }; 4 * k as usize]);
                    false
                } else {
                    true
                }
            });
            if let Some(f) = p.completes_frame() {
                p.sent(f, Ok(()));
                if let Some(n) = need(f) {
                    due.push((t + n, f));
                }
            }
            out.push(match p.play() {
                Play::Audio { .. } => 'A',
                Play::Gap => '.',
                Play::Failed { .. } => 'X',
            });
        }
        out
    }

    #[test]
    fn a_steady_sidecar_plays_every_frame_at_its_lag() {
        let mut p = Playout::new(1, 3, 10, 1, 8);
        let got = simulate(&mut p, 20, |_| Some(2), true);
        assert_eq!(got, "...AAAAAAAAAAAAAAAAA", "fills to the starting lag, then never misses");
        assert_eq!((p.stats.played, p.stats.gaps, p.stats.timeouts), (17, 0, 0));
    }

    /// The first frame needs more than the estimate: playback starts `slack` above the lag it
    /// needed and stays there without gaps.
    #[test]
    fn the_start_keeps_slack() {
        let mut p = Playout::new(1, 1, 10, 1, 8);
        let got = simulate(&mut p, 12, |_| Some(4), true);
        assert_eq!(got, ".....AAAAAAA");
        assert_eq!(p.target(), 5);
        assert_eq!(p.stats.gaps, 0);
        // One frame of jitter after the start costs nothing.
        let mut p = Playout::new(1, 1, 10, 1, 8);
        let got = simulate(&mut p, 20, |f| Some(if f % 3 == 0 { 5 } else { 4 }), true);
        assert!(got[6..].chars().all(|c| c == 'A'), "{got}");
    }

    /// A late frame is waited for (one gap, the lag grows by one) and nothing is lost.
    #[test]
    fn a_late_frame_grows_the_lag_once() {
        let mut p = Playout::new(1, 2, 10, 1, 8);
        let got = simulate(&mut p, 14, |f| Some(if f == 5 { 4 } else { 2 }), true);
        assert_eq!(got, "...AAAAA.AAAAA");
        assert_eq!((p.target(), p.stats.grown, p.stats.timeouts, p.stats.played), (4, 1, 0, 10));
    }

    /// Whole-frame acceptance case: a sidecar that answers each 100 ms frame after 700 ms (7
    /// frames) stays within a 1000 ms budget (10 frames): every frame plays, none times out.
    #[test]
    fn a_700_ms_sidecar_stays_within_a_1000_ms_budget() {
        let mut p = Playout::new(1, 1, 10, 1, 40);
        let got = simulate(&mut p, 60, |_| Some(7), true);
        assert_eq!(&got[..8], "........");
        assert!(got[8..].chars().all(|c| c == 'A'), "{got}");
        assert_eq!(p.stats.timeouts, 0);
    }

    /// Frames slower than the budget are given up at the budget (never later), and replies that
    /// come after that are stale.
    #[test]
    fn frames_slower_than_the_budget_time_out_at_the_budget() {
        let mut p = Playout::new(1, 2, 3, 1, 8);
        let got = simulate(&mut p, 12, |_| Some(7), true);
        assert_eq!(&got[..3], "...");
        assert!(got[3..].chars().all(|c| c == 'X'), "{got}");
        assert!(p.stats.stale > 0, "late replies were discarded");
        // A stalled sidecar: the same, without replies.
        let mut p = Playout::new(1, 2, 3, 1, 8);
        let got = simulate(&mut p, 12, |_| None, true);
        assert_eq!(got, "...XXXXXXXXX");
    }

    /// Once the sidecar is faster than the lag for a whole window, quiet slots are skipped and the
    /// lag drops; loud slots are never skipped.
    #[test]
    fn the_lag_drops_only_on_quiet_frames() {
        // The first frame is slow (lag 5), the rest need 2.
        let need = |f: u64| Some(if f == 0 { 5 } else { 2 });
        let mut loud = Playout::new(1, 1, 10, 1, 8);
        simulate(&mut loud, 60, need, true);
        assert_eq!(loud.stats.skipped, 0);
        assert!(loud.target() < 5, "the target is lowered");
        let mut quiet = Playout::new(1, 1, 10, 1, 8);
        let got = simulate(&mut quiet, 60, need, false);
        assert!(quiet.stats.skipped >= 1 && quiet.target() <= 3, "{got} {:?}", quiet.stats);
        assert_eq!(quiet.stats.timeouts, 0);
        assert!(!got[6..].contains('.'), "skips are not gaps: {got}");
        // Bookkeeping stays bounded: only the frames still to play are tracked.
        let lag = quiet.target() as usize;
        assert!(quiet.need_of.len() <= lag && quiet.failed.is_empty(), "{:?} (lag {lag})", quiet.need_of);
        assert!(quiet.need_of.keys().all(|&f| f >= quiet.next_play), "{:?}", quiet.need_of);
    }

    /// After a stall pushed the lag to the budget, the voice comes back to its steady delay within a
    /// couple of windows (skipping quiet slots), not one slot per window.
    #[test]
    fn the_lag_recovers_quickly_after_a_stall() {
        // Frames normally come back 2 ticks after they are sent; frames 10-14 take 9 (a stall).
        let need = |f: u64| Some(if (10..15).contains(&f) { 9 } else { 2 });
        let mut p = Playout::new(1, 1, 10, 1, 8);
        let got = simulate(&mut p, 80, need, false);
        assert_eq!(p.stats.timeouts, 0, "{got}");
        assert!(p.stats.grown >= 5, "the stall grew the lag: {:?}", p.stats);
        assert!(p.target() <= 4, "back near the steady lag of 3: target {} ({:?})", p.target(), p.stats);
        assert!(p.stats.skipped >= 4, "{:?}", p.stats);
    }

    /// Slots: a frame of 5 slots answered 3 slots after it was sent plays from the first slot
    /// after it is back (8 slots of delay, not two frames), and continuously.
    #[test]
    fn frames_play_slot_by_slot_as_soon_as_they_are_back() {
        let mut p = Playout::new(5, 5, 4 + 50, 2, 40);
        let got = simulate(&mut p, 60, |_| Some(3), true);
        // Frame 0 is sent on tick 4 and back on tick 7: need 7 slots, plus 2 of slack.
        assert_eq!(&got[..9], ".........", "{got}");
        assert!(got[9..].chars().all(|c| c == 'A'), "{got}");
        assert_eq!(p.target(), 9);
    }

    /// A late frame costs only the slots it is late by, and a timed-out frame is reported once
    /// (its other slots are silent).
    #[test]
    fn lateness_and_failures_are_counted_in_slots() {
        // Frames are back 3 slots after they are sent (a lag of 8 with 1 slot of slack); frame 4
        // takes 6: it is 2 slots late.
        let mut p = Playout::new(5, 5, 4 + 50, 1, 40);
        let got = simulate(&mut p, 60, |f| Some(if f == 4 { 6 } else { 3 }), true);
        assert_eq!(got.matches('.').count(), 8 + 2, "2 slots late: {got}");
        assert_eq!(p.stats.grown, 2);
        let mut p = Playout::new(5, 5, 4 + 4, 1, 40);
        let got = simulate(&mut p, 40, |f| if f == 3 { None } else { Some(3) }, true);
        assert_eq!(got.matches('X').count(), 1, "one report per frame: {got}");
        assert_eq!(p.stats.timeouts, 1);
        assert!(got.ends_with("AAAAA"), "{got}");
    }

    #[test]
    fn failures_are_reported_when_due_and_restart_forgets_the_past() {
        let mut p = Playout::new(1, 1, 4, 1, 8);
        p.sent(0, Err("busy".into()));
        assert_eq!(p.play(), Play::Failed { reason: "busy".into(), timeout: false, cancel_before: None });
        p.sent(1, Ok(()));
        p.fail(1, "refused".into());
        assert!(matches!(p.play(), Play::Failed { reason, .. } if reason == "refused"));
        p.sent(2, Ok(()));
        p.lose_in_flight("lost");
        assert!(matches!(p.play(), Play::Failed { reason, .. } if reason == "lost"));
        p.restart();
        assert!(!p.reply(2, vec![0.0]), "outstanding before the restart is stale");
        assert_eq!(p.stats.stale, 1);
        p.sent(3, Ok(()));
        assert_eq!(p.play(), Play::Gap, "a new stream fills again");
    }

    /// A stall grows the lag toward the budget; after a long pause the new stream starts from the
    /// starting estimate again, not from the stalled lag.
    #[test]
    fn a_restart_goes_back_to_the_starting_lag() {
        let mut p = Playout::new(1, 2, 12, 1, 40);
        // Steady at 2 ticks, then a stall: frames 6-8 need 9.
        simulate(&mut p, 30, |f| Some(if (6..9).contains(&f) { 9 } else { 2 }), true);
        assert!(p.target() >= 8, "the stall grew the lag: {}", p.target());
        p.restart();
        assert_eq!(p.target(), 2, "the new stream starts from the estimate");
        let got = simulate(&mut p, 12, |_| Some(2), true);
        assert_eq!(&got[..3], "...", "the first reply's lag plus slack, not the stall's: {got}");
        assert!(got[3..].chars().all(|c| c == 'A'), "{got}");
        assert_eq!(p.target(), 3);
    }

    /// Frames a single-stream sidecar leaves to another voice play as silence, never as an error,
    /// and are counted apart from failures.
    #[test]
    fn superseded_frames_are_quiet_gaps() {
        let mut p = Playout::new(5, 5, 4 + 8, 1, 40);
        let mut plays = Vec::new();
        for _ in 0..10 {
            if let Some(f) = p.completes_frame() {
                p.sent(f, Ok(()));
                p.superseded(f);
            }
            plays.push(p.play());
        }
        assert!(plays.iter().all(|x| *x == Play::Gap), "{plays:?}");
        assert_eq!((p.stats.superseded, p.stats.failed, p.stats.timeouts), (2, 0, 0));
        assert!(!p.reply(0, vec![0.5; 5]), "a late reply to a superseded frame is stale");
    }

    #[test]
    fn slots_divide_both_rates_near_20_ms() {
        assert_eq!(slots_per_frame(2880, 3969, 16_000), 9, "Seed-VC: 180 ms -> 20 ms slots");
        assert_eq!(slots_per_frame(1920, 1920, 16_000), 6, "X-VC: 120 ms -> 20 ms");
        assert_eq!(slots_per_frame(1600, 1600, 16_000), 5);
        assert_eq!(slots_per_frame(2048, 2048, 22_050), 4, "StreamVoiceAnon: 92.9 ms -> 23.2 ms");
        assert_eq!(slots_per_frame(160, 160, 16_000), 1, "a 10 ms frame stays whole");
        assert_eq!(slots_per_frame(1601, 2207, 16_000), 1, "nothing divides: whole frames");
    }

    #[test]
    fn wire_sequence_numbers_widen_to_frame_indices() {
        assert_eq!(widen(5, 7), 5);
        assert_eq!(widen(u32::MAX, (1 << 32) + 3), u64::from(u32::MAX));
        assert_eq!(widen(2, (1 << 32) + 3), (1 << 32) + 2);
        assert_eq!(widen(9, 3), 9, "a future number stays (and is refused as never sent)");
    }

    // --- Through the harness, against a scripted sidecar --------------------------------------

    fn connect(server: &TestServer, budget_ms: f64) -> SidecarModel {
        let mut c = ClientConfig::new("test", ClientConfig::fixed(server.url.clone(), None));
        c.handshake_timeout = Duration::from_secs(5);
        c.frame_timeout = Duration::from_secs_f64(budget_ms / 1000.0);
        let client = SidecarClient::connect(c).unwrap();
        SidecarModel::new(client, "test", budget_ms, 20.0, None, None)
    }

    /// 20 ms of a 220 Hz tone at 48 kHz, continuing from `k` blocks.
    fn tone_block(k: usize) -> Vec<f32> {
        (0..960).map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * (k * 960 + i) as f32 / 48_000.0).sin()).collect()
    }

    /// Feed `blocks` 20 ms device blocks in real time; returns (output RMS per block, errors,
    /// longest call).
    fn run(h: &mut VcStreamHarness, blocks: usize) -> (Vec<f32>, usize, Duration) {
        let mut rms_out = Vec::new();
        let mut errors = 0;
        let mut longest = Duration::ZERO;
        let start = Instant::now();
        for k in 0..blocks {
            let due = start + Duration::from_millis(20 * k as u64);
            if let Some(d) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(d);
            }
            let t0 = Instant::now();
            let r = h.push_and_convert(&tone_block(k));
            longest = longest.max(t0.elapsed());
            match r {
                Ok(out) => {
                    assert_eq!(out.len(), h.last_owed_frames());
                    rms_out.push(rms(&out));
                }
                Err(_) => {
                    errors += 1;
                    rms_out.push(0.0);
                }
            }
        }
        (rms_out, errors, longest)
    }

    /// Verify target: an echo sidecar that answers every 100 ms frame 700 ms late stays within a
    /// 1000 ms budget: audio comes back (about 0.8 s in: the frame, the 700 ms, a little slack),
    /// no block fails, and no call waits for the sidecar.
    #[test]
    fn sidecar_700_ms_delay_stays_within_budget() {
        let server = TestServer::spawn(Behavior::delayed(Duration::from_millis(700)));
        let model = connect(&server, 1000.0);
        assert_eq!(model.fixed_block(), Some(320), "20 ms slots of the 100 ms frame");
        let mut h = VcStreamHarness::new(Box::new(model), 48_000, 48_000, 1, 1, 0, 320, 960).unwrap();
        let (rms_out, errors, longest) = run(&mut h, 150);
        assert_eq!(errors, 0, "nothing timed out");
        assert!(longest < Duration::from_millis(50), "no call waited for the sidecar: {longest:?}");
        let first = rms_out.iter().position(|&r| r > 0.05).expect("audio came back");
        let delay_ms = first * 20;
        assert!((780..=1100).contains(&delay_ms), "about 0.8-1.0 s of delay: {delay_ms} ms; {rms_out:?}");
        let quiet = rms_out[first + 2..].iter().filter(|&&r| r < 0.1).count();
        assert!(quiet <= 2, "continuous after the start ({quiet} quiet blocks): {rms_out:?}");
    }

    /// Acceptance: a sidecar that stops answering never blocks the loop. Its blocks fail at the
    /// budget (the worker mutes them) and every call returns at once.
    #[test]
    fn sidecar_stall_never_blocks_the_loop() {
        let server = TestServer::spawn(Behavior::stalled());
        let model = connect(&server, 300.0);
        let mut h = VcStreamHarness::new(Box::new(model), 48_000, 48_000, 1, 1, 0, 320, 960).unwrap();
        let (_, errors, longest) = run(&mut h, 75);
        assert!(longest < Duration::from_millis(50), "{longest:?}");
        assert!(errors >= 8, "frames past the budget fail (once per frame): {errors}");
        // The sidecar is told what to drop, and the frames it holds stay bounded.
        let log = server.log();
        assert!(!log.cancels.is_empty());
        assert!(log.frames.len() <= 15, "{}", log.frames.len());
    }

    /// The voice being replaced on a single-stream sidecar (its frames answered `superseded`) plays
    /// silence without an error, and plays again once the sidecar converts for it.
    #[test]
    fn superseded_frames_play_as_silence_without_an_error() {
        let mut b = Behavior::delayed(Duration::from_millis(20));
        b.superseded = Arc::new(|seq| seq < 8);
        let server = TestServer::spawn(b);
        let model = connect(&server, 400.0);
        let mut h = VcStreamHarness::new(Box::new(model), 48_000, 48_000, 1, 1, 0, 320, 960).unwrap();
        let (rms_out, errors, longest) = run(&mut h, 100);
        assert_eq!(errors, 0, "silence, not failures: {rms_out:?}");
        assert!(longest < Duration::from_millis(50), "{longest:?}");
        let first = rms_out.iter().position(|&r| r > 0.05).expect("audio once the sidecar converts for it");
        assert!(first >= 8 * 5, "nothing played while another voice held the sidecar: block {first}");
    }

    /// The in-flight window follows the frame the server chose, not the one the hello proposed;
    /// only a limit the server states caps it.
    #[test]
    fn the_in_flight_window_follows_the_negotiated_frame() {
        let model = |b: Behavior| {
            let server = TestServer::spawn(b);
            // A recipe's 180 ms proposal at a 600 ms budget offers 6 frames; the server picks 100 ms.
            let mut c = ClientConfig::new("test", ClientConfig::fixed(server.url.clone(), None));
            c.frame_ms = 180.0;
            c.max_in_flight = in_flight_window(600.0, 180.0);
            c.frame_timeout = Duration::from_millis(600);
            c.handshake_timeout = Duration::from_secs(5);
            let client = SidecarClient::connect(c).unwrap();
            assert_eq!(client.session().frame_ms(), 100.0);
            SidecarModel::new(client, "test", 600.0, 20.0, None, None)
        };
        assert_eq!(in_flight_window(600.0, 180.0), 6);
        assert_eq!(model(Behavior::delayed(Duration::ZERO)).max_in_flight(), 8, "ceil(600 / 100) + 2");
        let mut b = Behavior::delayed(Duration::ZERO);
        b.welcome["max_in_flight"] = serde_json::json!(3);
        assert_eq!(model(b).max_in_flight(), 3, "the server's own limit");
    }

    /// The sidecar's answer after a stall: the stalled frames fail at the budget, the sidecar is
    /// told to drop them, fresh audio plays again.
    #[test]
    fn sidecar_recovers_after_a_stall_and_drops_stale_audio() {
        let mut b = Behavior::delayed(Duration::from_millis(50));
        // Frames 5-14 are answered 2 s late (a GPU stall); the budget is 400 ms.
        b.delay = Arc::new(|seq| Some(Duration::from_millis(if (5..15).contains(&seq) { 2000 } else { 50 })));
        let server = TestServer::spawn(b);
        let mut model = connect(&server, 400.0);
        let slot = model.fixed_block().unwrap();
        let mut audio = vec![0.3f32; 1600];
        let mut results = Vec::new();
        let start = Instant::now();
        for k in 0..200u64 {
            let due = start + Duration::from_millis(20 * k);
            if let Some(d) = due.checked_duration_since(Instant::now()) {
                std::thread::sleep(d);
            }
            if k % 5 == 0 {
                audio.iter_mut().for_each(|x| *x = -*x);
            }
            let ctx = ConversionCtx {
                mono_16k: &audio,
                new_frames: slot,
                pitch_shift_semitones: 2.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: k * slot as u64,
                reference: None,
                singing: Default::default(),
            };
            results.push(model.convert(&ctx).map(|o| rms(&o)));
        }
        let fails = results.iter().filter(|r| r.is_err()).count();
        assert!((8..=12).contains(&fails), "the stalled frames fail once each, {fails}: {results:?}");
        assert!(results[150..].iter().all(|r| matches!(r, Ok(x) if *x > 0.2)), "audio again: {results:?}");
        assert_eq!(model.stats().timeouts as usize, fails, "{:?}", model.stats());
        // The sidecar was told to drop each stalled frame once it was given up, so their late
        // audio never came back (a server that ignores the cancel has it discarded as stale).
        let log = server.log();
        assert!(log.cancels.contains(&15), "{:?}", log.cancels);
        assert!(log.frames.iter().all(|f| f.2 == 2.0), "pitch on every frame");
    }
}
