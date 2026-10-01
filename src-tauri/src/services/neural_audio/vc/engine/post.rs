//! Post stage at model rate (P3-D): rms_mix envelope matching against the input block, the
//! cepstral-envelope formant shift (±12 semitones, `vc::formant`; it replaces the one-pole tilt on
//! the RVC path), a dB silence gate with hangover, and an output limiter.
//!
//! With the default [`VoiceControls`] (rms_mix 1, formant 0) and the gate and limiter off, the stage
//! is a bit-exact pass-through: the legacy behaviour.
//!
//! # Ticks overlap
//! The engine calls [`PostStage::process`] once per tick with the tick's audio: the new block plus
//! the SOLA tail, so consecutive calls cover overlapping spans and the harness crossfades them.
//! Every step is therefore a function of the audio's absolute time, never of "how many calls
//! happened":
//! - rms_mix and the formant shift analyse each tick on its own (RVC realtime's rms_mix does the
//!   same; the formant frames are anchored to the tick's end, `vc::formant`);
//! - the gate and the limiter carry state, so the stage keeps a per-sample trace of it and starts
//!   each tick from the state recorded at the tick's first sample. Where that sample is in the last
//!   tick comes from a [`SliceTracker`] on the `source_16k` blocks (they are slices of one rolling
//!   ring, so the overlap matches sample for sample; ticks overlap by the SOLA tail only, 32-73
//!   samples at a 0-3 ms crossfade, which the tracker matches down to [`MIN_TRACK_MATCH`] samples),
//!   or from the caller ([`PostStage::process_at`]). A tick that does not continue the last one (a
//!   stream reset) starts from a fresh state.
//!
//! # Levels
//! - rms_mix: RVC's `rms_mix_rate`: the output envelope becomes `out * (rms_in / rms_out)^(1 -
//!   rms_mix)` with 40 ms RMS frames every 10 ms, linearly interpolated (as RVC realtime); the gain is
//!   capped at [`RMS_MIX_MAX_GAIN`] so a model that goes quiet on a loud input cannot blow up noise.
//! - Silence gate: a 10 ms input frame is loud when its RMS is at least `VoiceControls::silence_db`.
//!   The gate opens within [`GATE_ATTACK_MS`] of a loud frame and closes over [`GATE_RELEASE_MS`]
//!   once the input has been quiet for the hangover (`PostConfig::silence_hangover_ms`).
//! - Limiter: ceiling [`LIMITER_CEILING_DB`] dBFS, [`LIMITER_LOOKAHEAD_MS`] look-ahead within the
//!   tick, instant attack, [`LIMITER_RELEASE_MS`] release, and a final clamp so nothing leaves above
//!   the ceiling.
//!
//! # Frozen by P3-S0 (signatures)
//! [`PostConfig`], [`PostStage::new`], [`PostStage::process`] (in place, allocation-free after
//! construction), [`PostStage::reset`].
//!
//! # Added by P3-D
//! [`SliceTracker`] (also used by the streaming PESTO extractor), [`PostStage::process_at`],
//! [`PostStage::gate_gain`], the constants.

use super::super::formant::FormantShifter;
use super::super::traits::VoiceControls;

#[derive(Clone, Copy, Debug)]
pub struct PostConfig {
    /// Model output rate of `audio`.
    pub sample_rate: u32,
    /// Output limiter on (ceiling just under full scale).
    pub limiter: bool,
    /// Silence gate on (`VoiceControls::silence_db`), with this hangover.
    pub silence_gate: bool,
    pub silence_hangover_ms: f32,
}

/// Largest rms_mix gain (+20 dB).
pub const RMS_MIX_MAX_GAIN: f32 = 10.0;
/// rms_mix frame and hop (RVC realtime: `frame_length = 4 * zc`, `hop_length = zc`, zc = 10 ms).
const RMS_FRAME_MS: f32 = 40.0;
const RMS_HOP_16K: usize = 160;
/// rms_mix is off at or above this (the default 1.0 is exactly off).
const RMS_MIX_OFF: f32 = 0.999;
/// Gate timing.
pub const GATE_ATTACK_MS: f32 = 2.0;
pub const GATE_RELEASE_MS: f32 = 20.0;
/// Limiter.
pub const LIMITER_CEILING_DB: f32 = -0.3;
pub const LIMITER_LOOKAHEAD_MS: f32 = 1.5;
pub const LIMITER_RELEASE_MS: f32 = 60.0;
/// Input frame of the gate's level detector (10 ms at 16 kHz).
const GATE_FRAME_16K: usize = 160;
/// Samples of the previous slice compared when looking for it in the next one.
const TRACK_TAIL: usize = 96;
/// Fewest samples a match compares when consecutive slices overlap by less than [`TRACK_TAIL`] (the
/// engine's ticks overlap by the SOLA tail only: 40 samples at 40 kHz with a 0 ms crossfade).
pub const MIN_TRACK_MATCH: usize = 16;

/// Finds how far a stream advanced between two overlapping slices of it (the harness hands out
/// slices of one rolling ring, so the shared part matches sample for sample).
///
/// [`SliceTracker::advance`] returns the number of new samples at the end of `slice` since the last
/// call, or `None` when `slice` does not continue the last one (the first call, a stream reset, a
/// gap): the caller starts fresh. The last [`TRACK_TAIL`] samples of the previous slice are looked
/// for in the new one; where the two overlap by less, the overlapping part is compared (at least
/// [`MIN_TRACK_MATCH`] samples), smallest advance (largest overlap) first. A compared part that is
/// all silent (or constant) is ambiguous; it is accepted only at the advance the last call measured.
#[derive(Default)]
pub struct SliceTracker {
    tail: Vec<f32>,
    last_new: Option<usize>,
}

impl SliceTracker {
    pub fn new() -> SliceTracker {
        SliceTracker {
            tail: Vec::with_capacity(TRACK_TAIL),
            last_new: None,
        }
    }

    /// New samples at the end of `slice` since the last call (see the type docs).
    pub fn advance(&mut self, slice: &[f32]) -> Option<usize> {
        let found = self.find(slice);
        self.tail.clear();
        let k = TRACK_TAIL.min(slice.len());
        self.tail.extend_from_slice(&slice[slice.len() - k..]);
        self.last_new = found;
        found
    }

    /// Whether the previous slice ends at `end` in `slice`: the last `min(tail, end)` tracked
    /// samples (at least [`MIN_TRACK_MATCH`], or the whole tail when it is shorter) equal
    /// `slice[..end]`'s last ones. Returns the compared length on a match.
    fn match_len(&self, slice: &[f32], end: usize) -> Option<usize> {
        let k = self.tail.len();
        let m = k.min(end);
        if m == 0 || m < MIN_TRACK_MATCH.min(k) || end > slice.len() {
            return None;
        }
        let seg = &self.tail[k - m..];
        (slice[end - 1] == seg[m - 1] && slice[end - m..end] == *seg).then_some(m)
    }

    fn find(&self, slice: &[f32]) -> Option<usize> {
        let k = self.tail.len();
        let min = MIN_TRACK_MATCH.min(k);
        let n = slice.len();
        if k == 0 || n < min {
            return None;
        }
        if let Some(new) = self.last_new.filter(|&new| new <= n - min) {
            if self.match_len(slice, n - new).is_some() {
                return Some(new);
            }
        }
        // The tail is constant from this index on: a match comparing only that part is ambiguous.
        let last = self.tail[k - 1];
        let constant_from = self
            .tail
            .iter()
            .rposition(|&v| v != last)
            .map_or(0, |i| i + 1);
        // Smallest advance first (the previous end sits as late as possible in the new slice).
        (0..=n - min).find(|&new| matches!(self.match_len(slice, n - new), Some(m) if k - m < constant_from))
    }

    /// Forget the last slice (the next call returns `None`).
    pub fn reset(&mut self) {
        self.tail.clear();
        self.last_new = None;
    }
}

/// Gate and limiter state after one output sample.
#[derive(Clone, Copy, Debug, PartialEq)]
struct State {
    /// Gate gain, 0..1.
    gate: f32,
    /// Output samples since the last loud input frame (saturating).
    quiet: u32,
    /// Limiter gain, 0..1.
    limiter: f32,
}

impl State {
    const FRESH: State = State {
        gate: 1.0,
        quiet: 0,
        limiter: 1.0,
    };
}

fn db_to_amp(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

pub struct PostStage {
    config: PostConfig,
    formant: FormantShifter,
    tracker: SliceTracker,
    /// State after each output sample of the last tick.
    trace: Vec<State>,
    // ---- scratch ----
    env_in: Vec<f32>,
    env_out: Vec<f32>,
    loud: Vec<bool>,
    required: Vec<f32>,
}

impl PostStage {
    pub fn new(config: PostConfig) -> PostStage {
        let sr = config.sample_rate.max(8_000);
        // Scratch for ticks up to 250 ms (the engine's longest block plus its SOLA tail), so the
        // first ticks do not allocate either; a longer tick grows it once.
        let max_out = sr as usize / 4;
        let max_frames = 16_000 / 4 / GATE_FRAME_16K + 2;
        PostStage {
            config: PostConfig { sample_rate: sr, ..config },
            formant: FormantShifter::with_capacity(sr, max_out),
            tracker: SliceTracker::new(),
            trace: Vec::with_capacity(max_out),
            env_in: Vec::with_capacity(max_frames),
            env_out: Vec::with_capacity(max_frames),
            loud: Vec::with_capacity(max_frames),
            required: Vec::with_capacity(max_out),
        }
    }

    /// Process one tick in place: `source_16k` is the input block the audio was converted from
    /// (rms_mix reference and gate level), `audio` the model-rate output of the same span.
    pub fn process(&mut self, source_16k: &[f32], audio: &mut [f32], controls: &VoiceControls) -> Result<(), String> {
        let advanced = self.tracker.advance(source_16k);
        self.run(advanced, source_16k, audio, controls)
    }

    /// [`process`](Self::process) for a caller that knows how many new 16 kHz samples this tick
    /// adds (`None` = a new stream); skips the slice matching.
    pub fn process_at(&mut self, new_16k: Option<usize>, source_16k: &[f32], audio: &mut [f32], controls: &VoiceControls) -> Result<(), String> {
        self.tracker.reset();
        let _ = self.tracker.advance(source_16k);
        self.run(new_16k, source_16k, audio, controls)
    }

    /// The gate's gain at the end of the last tick (1 = open).
    pub fn gate_gain(&self) -> f32 {
        self.trace.last().map_or(1.0, |s| s.gate)
    }

    fn run(&mut self, advanced: Option<usize>, source_16k: &[f32], audio: &mut [f32], controls: &VoiceControls) -> Result<(), String> {
        if audio.is_empty() {
            return Ok(());
        }
        for v in audio.iter_mut() {
            if !v.is_finite() {
                *v = 0.0;
            }
        }
        let sr = self.config.sample_rate as f64;
        // State at the tick's first sample: what the last tick recorded just before it.
        let start = match advanced {
            Some(new) if !self.trace.is_empty() => {
                let new_out = ((new as f64) * sr / 16_000.0).round() as usize;
                let len = self.trace.len();
                if new_out == 0 {
                    // Nothing new: the state before the previous tick's first sample is unknown;
                    // continue from its end.
                    self.trace[len - 1]
                } else if new_out <= len {
                    self.trace[new_out - 1]
                } else {
                    self.trace[len - 1]
                }
            }
            _ => State::FRESH,
        };
        if controls.rms_mix.is_finite() && controls.rms_mix < RMS_MIX_OFF && !source_16k.is_empty() {
            self.rms_mix(source_16k, audio, controls.rms_mix.max(0.0));
        }
        self.formant.process(audio, controls.formant_semitones)?;
        let gate_on = self.config.silence_gate && !source_16k.is_empty();
        if gate_on {
            self.detect_loud(source_16k, controls.silence_db);
        }
        if self.config.limiter {
            self.limiter_requirements(audio);
        }
        self.trace.clear();
        let mut state = start;
        let per_16k = 16_000.0 / sr;
        let hangover = (self.config.silence_hangover_ms.max(0.0) as f64 * sr / 1000.0) as u32;
        let attack = 1.0 / ((GATE_ATTACK_MS as f64 * sr / 1000.0).max(1.0) as f32);
        let release = 1.0 / ((GATE_RELEASE_MS as f64 * sr / 1000.0).max(1.0) as f32);
        let lim_release = (-1.0 / (LIMITER_RELEASE_MS as f64 * sr / 1000.0)).exp() as f32;
        let ceiling = db_to_amp(LIMITER_CEILING_DB);
        let frames = self.loud.len();
        for (i, x) in audio.iter_mut().enumerate() {
            if gate_on && frames > 0 {
                let f = (((i as f64) * per_16k) as usize / GATE_FRAME_16K).min(frames - 1);
                if self.loud[f] {
                    state.quiet = 0;
                } else {
                    state.quiet = state.quiet.saturating_add(1);
                }
                let target = if state.quiet <= hangover { 1.0 } else { 0.0 };
                if state.gate < target {
                    state.gate = (state.gate + attack).min(1.0);
                } else if state.gate > target {
                    state.gate = (state.gate - release).max(0.0);
                }
                *x *= state.gate;
            }
            if self.config.limiter {
                // Instant attack to the look-ahead requirement, exponential release toward it.
                let need = self.required[i];
                state.limiter = if need < state.limiter {
                    need
                } else {
                    need + (state.limiter - need) * lim_release
                };
                *x = (*x * state.limiter).clamp(-ceiling, ceiling);
            }
            self.trace.push(state);
        }
        Ok(())
    }

    /// RVC rms_mix on the tick.
    fn rms_mix(&mut self, source_16k: &[f32], audio: &mut [f32], rms_mix: f32) {
        let frames = source_16k.len().div_ceil(RMS_HOP_16K) + 1;
        let half_in = (RMS_FRAME_MS * 16.0 / 2.0) as usize;
        let sr = self.config.sample_rate as f64;
        let scale = sr / 16_000.0;
        let half_out = ((half_in as f64) * scale) as usize;
        rms_envelope(source_16k, frames, RMS_HOP_16K as f64, half_in, &mut self.env_in);
        rms_envelope(audio, frames, RMS_HOP_16K as f64 * scale, half_out, &mut self.env_out);
        let exponent = 1.0 - rms_mix.clamp(0.0, 1.0);
        let per_frame = 1.0 / (RMS_HOP_16K as f64 * scale);
        for (i, x) in audio.iter_mut().enumerate() {
            // Linear interpolation of both envelopes at this sample (RVC interpolates the frames).
            let u = i as f64 * per_frame;
            let f0 = (u.floor() as usize).min(frames - 1);
            let f1 = (f0 + 1).min(frames - 1);
            let t = (u - f0 as f64) as f32;
            let rin = self.env_in[f0] + (self.env_in[f1] - self.env_in[f0]) * t;
            let rout = (self.env_out[f0] + (self.env_out[f1] - self.env_out[f0]) * t).max(1e-3);
            let g = (rin / rout).powf(exponent).min(RMS_MIX_MAX_GAIN);
            *x *= if g.is_finite() { g } else { 1.0 };
        }
    }

    /// Loudness of each 10 ms input frame.
    fn detect_loud(&mut self, source_16k: &[f32], silence_db: f32) {
        let threshold = db_to_amp(if silence_db.is_finite() {
            silence_db
        } else {
            crate::services::neural_audio::vc::traits::DEFAULT_SILENCE_DB
        });
        let t2 = threshold * threshold;
        self.loud.clear();
        for frame in source_16k.chunks(GATE_FRAME_16K) {
            let ms: f32 = frame
                .iter()
                .map(|&v| if v.is_finite() { v * v } else { 0.0 })
                .sum::<f32>()
                / frame.len() as f32;
            self.loud.push(ms >= t2);
        }
    }

    /// Per-sample limiter gain requirement: the smallest `ceiling / |x|` within the look-ahead.
    fn limiter_requirements(&mut self, audio: &[f32]) {
        let ceiling = db_to_amp(LIMITER_CEILING_DB);
        let look = ((LIMITER_LOOKAHEAD_MS as f64 * self.config.sample_rate as f64 / 1000.0) as usize).max(1);
        let n = audio.len();
        self.required.clear();
        self.required.extend(audio.iter().map(|&x| {
            let a = x.abs();
            if a > ceiling {
                ceiling / a
            } else {
                1.0
            }
        }));
        // Look-ahead: ramp linearly from unity down to each peak's requirement over the `look`
        // samples before it (the binding peak is the one whose ramp is lowest at the sample).
        let mut bind: Option<(usize, f32)> = None;
        for i in (0..n).rev() {
            let own = self.required[i];
            let ramp = bind.and_then(|(p, need)| (p - i <= look).then(|| need + (1.0 - need) * (p - i) as f32 / look as f32));
            match ramp {
                Some(r) if r <= own => self.required[i] = r,
                _ if own < 1.0 => bind = Some((i, own)),
                _ => {}
            }
        }
    }

    /// Clear envelopes and gate state (stream gap / restart).
    pub fn reset(&mut self) {
        self.tracker.reset();
        self.trace.clear();
        self.formant.reset();
    }
}

/// RMS of `frames` windows of ±`half` samples centred every `hop` samples of `x` (windows truncated
/// at the ends, so the edges are not biased toward silence).
fn rms_envelope(x: &[f32], frames: usize, hop: f64, half: usize, out: &mut Vec<f32>) {
    out.clear();
    let n = x.len();
    for f in 0..frames {
        let c = ((f as f64) * hop).round() as usize;
        let lo = c.saturating_sub(half).min(n);
        let hi = (c + half).min(n);
        if hi <= lo {
            out.push(out.last().copied().unwrap_or(0.0));
            continue;
        }
        let ms: f64 = x[lo..hi]
            .iter()
            .map(|&v| f64::from(v) * f64::from(v))
            .sum::<f64>()
            / (hi - lo) as f64;
        out.push(ms.sqrt() as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 40_000;

    fn stage(limiter: bool, gate: bool) -> PostStage {
        PostStage::new(PostConfig {
            sample_rate: SR,
            limiter,
            silence_gate: gate,
            silence_hangover_ms: 150.0,
        })
    }

    fn tone(len: usize, sr: u32, hz: f32, amp: f32) -> Vec<f32> {
        (0..len)
            .map(|i| amp * (std::f32::consts::TAU * hz * i as f32 / sr as f32).sin())
            .collect()
    }

    fn rms(x: &[f32]) -> f32 {
        (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt()
    }

    #[test]
    fn neutral_controls_are_a_bit_exact_pass_through() {
        let src = tone(1_600, 16_000, 200.0, 0.3);
        let out = tone(4_000, SR, 200.0, 0.4);
        let mut y = out.clone();
        stage(false, false)
            .process(&src, &mut y, &VoiceControls::default())
            .unwrap();
        assert_eq!(y, out);
    }

    #[test]
    fn slice_tracker_finds_the_advance_and_resets_on_gaps() {
        let ring: Vec<f32> = (0..20_000)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5 + (i % 7) as f32 * 1e-3)
            .collect();
        let mut t = SliceTracker::new();
        assert_eq!(t.advance(&ring[0..2_000]), None);
        assert_eq!(t.advance(&ring[640..2_640]), Some(640));
        // Slices of another length continue too.
        assert_eq!(t.advance(&ring[1_000..3_280]), Some(640));
        assert_eq!(t.advance(&ring[2_000..3_280]), Some(0));
        // A jump the slice does not overlap is a reset.
        assert_eq!(t.advance(&ring[10_000..12_000]), None);
        assert_eq!(t.advance(&ring[10_500..12_500]), Some(500));
        // Silence: accepted only at the last measured advance.
        let mut silent = vec![0.0f32; 6_000];
        silent[..100].copy_from_slice(&ring[..100]);
        let mut t = SliceTracker::new();
        t.advance(&silent[0..2_000]);
        assert_eq!(t.advance(&silent[..2_000]), None);
        let mut t = SliceTracker::new();
        let mut voiced_then_silent = ring[..3_000].to_vec();
        voiced_then_silent.extend(vec![0.0; 3_000]);
        assert_eq!(t.advance(&voiced_then_silent[0..3_000]), None);
        assert_eq!(t.advance(&voiced_then_silent[640..3_640]), Some(640));
        assert_eq!(t.advance(&voiced_then_silent[1_280..4_280]), Some(640));
    }

    /// The engine's ticks overlap by the SOLA tail only (40 samples at 40 kHz with a 0 ms
    /// crossfade): the tracker still finds the advance, down to `MIN_TRACK_MATCH` samples.
    #[test]
    fn slice_tracker_matches_overlaps_shorter_than_its_tail() {
        let ring: Vec<f32> = (0..20_000)
            .map(|i| ((i as f32) * 0.013).sin() * 0.5 + (i % 7) as f32 * 1e-3)
            .collect();
        for overlap in [80usize, 40, 32, MIN_TRACK_MATCH] {
            let mut t = SliceTracker::new();
            let len = 640 + overlap;
            assert_eq!(t.advance(&ring[0..len]), None);
            for k in 1..6 {
                assert_eq!(t.advance(&ring[k * 640..k * 640 + len]), Some(640), "overlap {overlap}, tick {k}");
            }
            // A jump is still a reset.
            assert_eq!(t.advance(&ring[12_000..12_000 + len]), None, "overlap {overlap}");
        }
        // Below the minimum match nothing is trusted (a fresh state every tick).
        let mut t = SliceTracker::new();
        let len = 640 + MIN_TRACK_MATCH - 1;
        t.advance(&ring[0..len]);
        assert_eq!(t.advance(&ring[640..640 + len]), None);
        // A short overlap that is silent is accepted only at the measured advance.
        let mut speech_then_silence = ring[..3_000].to_vec();
        speech_then_silence.extend(vec![0.0; 3_000]);
        let mut t = SliceTracker::new();
        let len = 680;
        t.advance(&speech_then_silence[1_640..1_640 + len]);
        assert_eq!(t.advance(&speech_then_silence[2_280..2_280 + len]), Some(640));
        assert_eq!(t.advance(&speech_then_silence[2_920..2_920 + len]), Some(640));
        // The tracked tail is silence now: the measured advance still continues the stream.
        assert_eq!(t.advance(&speech_then_silence[3_560..3_560 + len]), Some(640));
        let mut fresh = SliceTracker::new();
        fresh.advance(&speech_then_silence[3_560..3_560 + len]);
        assert_eq!(fresh.advance(&speech_then_silence[4_200..4_200 + len]), None);
    }

    #[test]
    fn rms_mix_follows_the_input_envelope() {
        // Input: loud then quiet; output: constant. rms_mix 0 hands the input's envelope over.
        let mut src = tone(3_200, 16_000, 180.0, 0.4);
        for v in &mut src[1_600..] {
            *v *= 0.1;
        }
        let out = tone(8_000, SR, 180.0, 0.2);
        let mut y = out.clone();
        let c = VoiceControls {
            rms_mix: 0.0,
            ..VoiceControls::default()
        };
        stage(false, false).process(&src, &mut y, &c).unwrap();
        let head = rms(&y[800..3_000]);
        let tail = rms(&y[5_000..7_200]);
        assert!((head / rms(&src[320..1_200]) - 1.0).abs() < 0.1, "head {head}");
        assert!((tail / rms(&src[2_000..2_880]) - 1.0).abs() < 0.1, "tail {tail}");
        // rms_mix 0.5 lands in between (geometric).
        let mut y = out.clone();
        let c = VoiceControls {
            rms_mix: 0.5,
            ..VoiceControls::default()
        };
        stage(false, false).process(&src, &mut y, &c).unwrap();
        let want = (rms(&out[5_000..7_200]) * rms(&src[2_000..2_880])).sqrt();
        assert!((rms(&y[5_000..7_200]) / want - 1.0).abs() < 0.1);
        // The gain cap holds when the model goes silent on a loud input.
        let mut y = vec![1e-5f32; 4_000];
        let c = VoiceControls {
            rms_mix: 0.0,
            ..VoiceControls::default()
        };
        stage(false, false)
            .process(&tone(1_600, 16_000, 100.0, 0.5), &mut y, &c)
            .unwrap();
        assert!(y
            .iter()
            .all(|&v| v.abs() <= 1e-5 * RMS_MIX_MAX_GAIN * 1.001));
    }

    #[test]
    fn gate_closes_after_the_hangover_and_reopens_fast() {
        // 16 kHz input: 100 ms loud, then silence; ticks of 40 ms overlapping by the SOLA tail, as
        // the engine calls them: 20 ms, and the 40 / 32 samples of a 0 ms crossfade at 40 / 48 kHz
        // (the gate must close there too, not restart open every tick).
        for tail in [320usize, 40, 32] {
            gate_closes_with_tail(tail);
        }
    }

    fn gate_closes_with_tail(tail: usize) {
        let total_16k = 16_000;
        let mut input = tone(total_16k, 16_000, 200.0, 0.3);
        for v in &mut input[1_600..9_600] {
            *v = 0.0;
        }
        let model_out = tone(total_16k * 5 / 2, SR, 200.0, 0.3);
        let mut post = stage(false, true);
        let block = 640usize;
        let mut emitted = vec![0.0f32; total_16k * 5 / 2];
        let mut end = block + tail;
        while end <= total_16k {
            let src = &input[end - block - tail..end];
            let o0 = (end - block - tail) * 5 / 2;
            let mut y = model_out[o0..end * 5 / 2].to_vec();
            post.process(src, &mut y, &VoiceControls::default())
                .unwrap();
            // Keep the block part (what SOLA would emit).
            let b0 = (end - block) * 5 / 2;
            emitted[b0..end * 5 / 2].copy_from_slice(&y[y.len() - block * 5 / 2..]);
            end += block;
        }
        let at = |ms: usize| rms(&emitted[ms * 40..ms * 40 + 400]);
        // Loud part passes, the first 150 ms of silence still pass (hangover), then it is gated.
        assert!(at(60) > 0.2, "tail {tail}: {}", at(60));
        assert!(at(200) > 0.2, "tail {tail}: hangover: {}", at(200));
        assert!(at(300) < 1e-3, "tail {tail}: gated: {}", at(300));
        assert!(at(500) < 1e-3, "tail {tail}: stays gated: {}", at(500));
        // Reopens within a few ms of the input returning at 600 ms.
        assert!(at(610) > 0.2, "tail {tail}: reopen: {}", at(610));
        assert!(post.gate_gain() > 0.99);
    }

    #[test]
    fn limiter_holds_the_ceiling_and_recovers() {
        let ceiling = db_to_amp(LIMITER_CEILING_DB);
        let mut y = tone(8_000, SR, 300.0, 0.5);
        for v in &mut y[2_000..2_400] {
            *v *= 3.0;
        }
        let x = y.clone();
        stage(true, false)
            .process(&[], &mut y, &VoiceControls::default())
            .unwrap();
        assert!(y.iter().all(|v| v.abs() <= ceiling + 1e-6));
        // Before the look-ahead reaches the peak, nothing changes.
        assert_eq!(&y[..1_800], &x[..1_800]);
        // 250 ms after the burst the gain is back near unity.
        assert!(rms(&y[7_000..8_000]) > 0.9 * 0.5 / 2f32.sqrt());
        // The gain ramps down before the burst instead of jumping (no click at the look-ahead edge):
        // over the look-ahead, the per-sample gain never falls by more than a ramp step.
        let look = (LIMITER_LOOKAHEAD_MS * SR as f32 / 1000.0) as usize;
        let gains: Vec<(usize, f32)> = (1_900..2_000)
            .filter(|&i| x[i].abs() > 0.05)
            .map(|i| (i, y[i] / x[i]))
            .collect();
        let step = (1.0 - ceiling / 1.5) / look as f32;
        for w in gains.windows(2) {
            let ((i0, g0), (i1, g1)) = (w[0], w[1]);
            assert!(g0 - g1 <= step * (i1 - i0) as f32 + 1e-4, "gain {g0} at {i0} -> {g1} at {i1}");
        }
        assert!(gains.last().unwrap().1 < 0.97, "the ramp starts before the burst: {gains:?}");
    }

    /// The gate / limiter state follows absolute time: two runs whose ticks overlap differently
    /// give the same audio in the part both keep.
    #[test]
    fn state_follows_absolute_time_across_overlapping_ticks() {
        let input: Vec<f32> = (0..16_000)
            .map(|i| if (i / 2_000) % 2 == 0 { 0.3 * ((i as f32) * 0.07).sin() } else { 0.0 })
            .collect();
        let model_out: Vec<f32> = (0..40_000)
            .map(|i| 1.4 * ((i as f32) * 0.03).sin())
            .collect();
        let run = |tail: usize| -> Vec<f32> {
            let mut post = stage(true, true);
            let block = 640;
            let mut kept = vec![0.0f32; 40_000];
            let mut end = 2_560usize;
            while end <= 16_000 {
                let src = &input[end - block - tail..end];
                let mut y = model_out[(end - block - tail) * 5 / 2..end * 5 / 2].to_vec();
                post.process(src, &mut y, &VoiceControls::default())
                    .unwrap();
                kept[(end - block) * 5 / 2..end * 5 / 2].copy_from_slice(&y[y.len() - block * 5 / 2..]);
                end += block;
            }
            kept
        };
        let a = run(320);
        // Skip the first ticks (a fresh stream) and compare the rest; 40 samples is the SOLA tail
        // of a 0 ms crossfade at 40 kHz (the limiter's gain must not restart at unity every tick).
        for tail in [800usize, 40] {
            let b = run(tail);
            let mut worst = 0.0f32;
            for i in 20_000..40_000 {
                worst = worst.max((a[i] - b[i]).abs());
            }
            assert!(worst < 1e-4, "tail {tail}: {worst}");
        }
    }

    /// Plan T2: the post stage allocates nothing per tick (every step on: rms_mix, formant, gate,
    /// limiter).
    #[test]
    fn process_is_allocation_free_after_construction() {
        use crate::services::neural_audio::gpu::alloc_probe::allocations_during;
        let mut post = stage(true, true);
        let c = VoiceControls {
            rms_mix: 0.3,
            formant_semitones: 4.0,
            ..VoiceControls::default()
        };
        let input = tone(16_000, 16_000, 190.0, 0.3);
        let out = tone(40_000, SR, 190.0, 0.9);
        let (block, tail) = (640usize, 481usize);
        let mut y = vec![0.0f32; (block + tail) * 5 / 2];
        let mut end = block + tail;
        let mut total = 0;
        while end <= 16_000 {
            let o0 = (end - block - tail) * 5 / 2;
            let n = y.len();
            y.copy_from_slice(&out[o0..o0 + n]);
            let src = &input[end - block - tail..end];
            let (r, n) = allocations_during(|| post.process(src, &mut y, &c));
            r.unwrap();
            total += n;
            end += block;
        }
        assert_eq!(total, 0, "allocations in the post stage");
    }

    /// Dev rig: per-tick cost of the post stage (40 ms block + the SVE's 481-sample SOLA tail) with
    /// every step on, at 40 and 48 kHz. `cargo test post_stage_cost -- --ignored --nocapture`
    #[test]
    #[ignore = "dev-rig timing"]
    fn post_stage_cost() {
        for sr in [40_000u32, 48_000] {
            let mut post = PostStage::new(PostConfig {
                sample_rate: sr,
                limiter: true,
                silence_gate: true,
                silence_hangover_ms: 150.0,
            });
            let input = tone(16_000 * 4, 16_000, 190.0, 0.3);
            let out = tone(sr as usize * 4, sr, 190.0, 0.9);
            let (block, tail) = (640usize, 481usize);
            let per = |n: usize| n * sr as usize / 16_000;
            for (label, c) in [
                ("gate+limiter", VoiceControls::default()),
                (
                    "all on (rms_mix 0.3, formant +4)",
                    VoiceControls {
                        rms_mix: 0.3,
                        formant_semitones: 4.0,
                        ..VoiceControls::default()
                    },
                ),
            ] {
                let mut times = Vec::new();
                let mut end = block + tail;
                while end <= input.len() {
                    let o0 = per(end - block - tail);
                    let mut y = out[o0..o0 + per(block + tail)].to_vec();
                    let t0 = std::time::Instant::now();
                    post.process(&input[end - block - tail..end], &mut y, &c)
                        .unwrap();
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    end += block;
                }
                times.sort_by(f64::total_cmp);
                println!(
                    "post stage {sr} Hz {label}: p50 {:.3} ms, p95 {:.3} ms ({} ticks)",
                    times[times.len() / 2],
                    times[times.len() * 95 / 100],
                    times.len()
                );
            }
        }
    }
}
