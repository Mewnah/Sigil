//! Aux feed: decoded TTS / external PCM mixed into the live capture stream before conversion (so
//! TTS speaks with the converted voice).
//!
//! (Named `aux_feed` rather than `aux`: AUX is a reserved device name on Windows.)
//!
//! - Clips are resampled once, band-limited (rubato FFT resampler), to the capture rate and
//!   channel count, then queued whole as one buffer each ([`AuxQueue`], a `VecDeque` of clips), so
//!   a push holds the lock only to move a `Vec` in, never to copy audio.
//! - Back-pressure instead of truncation: a clip is accepted whole while less than
//!   [`AUX_QUEUE_CAP_SECS`] of audio is queued (a clip is never cut and queued audio is never
//!   dropped from the front); otherwise the push is refused with an error. The IPC commands wait
//!   (up to [`AUX_PUSH_WAIT`]) for room instead of failing at once, in a strict arrival-order line
//!   that pushes which do not wait never overtake, so clips always play in push order.
//! - A source clip is checked before any conversion: its sample rate must be sane
//!   ([`AUX_MIN_SAMPLE_RATE`]-[`AUX_MAX_SAMPLE_RATE`]) and its length (in source frames) at most
//!   [`AUX_MAX_CLIP_SECS`], so a bogus WAV header or an over-long clip is refused before it is
//!   resampled.
//! - The lock tolerates poisoning: a panic while it was held (e.g. in a TTS caller) cannot wedge
//!   the voice changer; the queue's bookkeeping is repaired on recovery.
//! - [`AuxMixer`] mixes the queue into each block and applies the aux mic gate (none / duck /
//!   mute) with smooth attack, hold and release so the mic never clicks.
//! - Clips carry a tag (0 = untagged IPC pushes). The unified TTS sink (`tts_sink`, P2-G) tags
//!   each utterance's audio, so it can count what of an utterance is still queued
//!   ([`VoiceChangerState::tts_aux_pending`]) and cancel it (stop / skip,
//!   [`VoiceChangerState::tts_aux_cancel`]): a clip that already started playing ends with a short
//!   fade instead of a click, clips that did not start are dropped.

use super::params::AuxMicGate;
use super::playback::adapt_channels_interleaved_into;
use super::state::VoiceChangerState;
use base64::{engine::general_purpose::STANDARD as B64_ENGINE, Engine as _};
use hound::WavReader;
use rubato::{FftFixedIn, Resampler};
use serde::Deserialize;
use std::collections::VecDeque;
use std::io::Cursor;
use std::sync::{Condvar, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};
use tauri::{command, State};

/// Queued aux audio above which new clips are refused (back-pressure).
pub(super) const AUX_QUEUE_CAP_SECS: u64 = 60;
/// Longest single clip accepted (a clip is never truncated; longer ones are refused whole).
pub(super) const AUX_MAX_CLIP_SECS: u64 = 600;
/// How long an IPC push waits for the queue to drain below the cap.
pub(super) const AUX_PUSH_WAIT: Duration = Duration::from_secs(30);
/// Source sample rates accepted from TTS / WAV headers (anything else is refused before
/// conversion).
pub(super) const AUX_MIN_SAMPLE_RATE: u32 = 4_000;
pub(super) const AUX_MAX_SAMPLE_RATE: u32 = 384_000;

/// Fade applied to a tagged clip that is cancelled while it plays (ms), so stop / skip never clicks.
const CANCEL_FADE_MS: u32 = 5;

/// Mic gain while ducked (about -12 dB).
const DUCK_GAIN: f32 = 0.25;
/// Gate time constants: fast attack so the first TTS syllable is clean, a hold that bridges the
/// pauses between sentences, then a gentle release.
const GATE_ATTACK_MS: f32 = 5.0;
const GATE_HOLD_MS: f32 = 250.0;
const GATE_RELEASE_MS: f32 = 80.0;

/// Why a clip was not queued.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum AuxPushError {
    NotRunning,
    /// `queued` seconds already waiting (>= the cap).
    Full { queued_secs: u64 },
    /// The clip alone is longer than [`AUX_MAX_CLIP_SECS`].
    TooLong { clip_secs: u64 },
}

impl AuxPushError {
    pub(super) fn message(&self) -> String {
        match self {
            AuxPushError::NotRunning => "Voice changer is not running".into(),
            AuxPushError::Full { queued_secs } => format!(
                "Voice-changer TTS queue is full ({queued_secs} s waiting to play); the clip was not queued. Try again when the current speech has played."
            ),
            AuxPushError::TooLong { clip_secs } => format!(
                "TTS clip is too long for the voice changer ({clip_secs} s; at most {AUX_MAX_CLIP_SECS} s); it was not played."
            ),
        }
    }
}

/// One queued clip (interleaved f32 at the capture rate / channels) and its owner tag.
struct AuxClip {
    samples: Vec<f32>,
    /// 0 = untagged (IPC pushes); otherwise the TTS sink utterance that owns it.
    tag: u64,
}

/// Queued aux clips with a read offset into the front clip.
#[derive(Default)]
pub(super) struct AuxQueue {
    clips: VecDeque<AuxClip>,
    /// Samples of the front clip already mixed.
    head: usize,
    /// Samples not yet mixed, over all clips.
    queued: usize,
    /// Capture format the queued audio is in (0 = not running; pushes are refused).
    rate: u32,
    ch: usize,
    /// Tickets of the pushers blocked waiting for room, in arrival order: only the front one may
    /// take room that appears, and pushes that do not wait queue behind them (clips play in the
    /// order they were pushed). The mixer only signals while someone waits.
    waiting: VecDeque<u64>,
    next_ticket: u64,
    /// Tag of the clip the last mix ended in while aux audio flowed without a gap (0: none, or
    /// the queue ran dry, so whatever comes next starts from silence). A tag's next clip at read
    /// offset 0 continues audio that is playing right now when this is its tag.
    last_tag: u64,
}

impl AuxQueue {
    pub(super) fn queued_samples(&self) -> usize {
        self.queued
    }

    fn queued_secs(&self) -> u64 {
        let per_sec = (u64::from(self.rate) * self.ch.max(1) as u64).max(1);
        self.queued as u64 / per_sec
    }

    /// Whether the queue is running and below the cap.
    fn has_room(&self) -> bool {
        self.rate > 0 && (self.queued as u64) < AUX_QUEUE_CAP_SECS * u64::from(self.rate) * self.ch as u64
    }

    /// Whether a push that does not wait is accepted now: room, and no earlier push still waiting
    /// for it (a newcomer never overtakes a waiting clip).
    fn admits_new(&self) -> bool {
        self.has_room() && self.waiting.is_empty()
    }

    /// Queue a whole clip already in the capture format, or refuse it whole (never overtakes a
    /// waiting push).
    pub(super) fn try_push(&mut self, clip: Vec<f32>) -> Result<(), AuxPushError> {
        self.try_push_tagged(clip, 0)
    }

    /// [`Self::try_push`] for a clip owned by `tag`.
    fn try_push_tagged(&mut self, clip: Vec<f32>, tag: u64) -> Result<(), AuxPushError> {
        let admitted = self.admits_new();
        self.push_checked(clip, admitted, tag)
    }

    /// Queue `clip` if `admitted` (the caller's turn and room), refusing it whole otherwise.
    fn push_checked(&mut self, clip: Vec<f32>, admitted: bool, tag: u64) -> Result<(), AuxPushError> {
        if self.rate == 0 {
            return Err(AuxPushError::NotRunning);
        }
        let per_sec = u64::from(self.rate) * self.ch.max(1) as u64;
        if clip.len() as u64 > AUX_MAX_CLIP_SECS * per_sec {
            return Err(AuxPushError::TooLong { clip_secs: clip.len() as u64 / per_sec });
        }
        if !admitted {
            return Err(AuxPushError::Full { queued_secs: self.queued_secs() });
        }
        let n = clip.len() / self.ch.max(1) * self.ch.max(1);
        if n == 0 {
            return Ok(());
        }
        let mut clip = clip;
        clip.truncate(n);
        self.queued += n;
        self.clips.push_back(AuxClip { samples: clip, tag });
        Ok(())
    }

    /// Samples of `tag`'s clips not yet mixed.
    fn pending_tagged(&self, tag: u64) -> usize {
        self.clips
            .iter()
            .enumerate()
            .filter(|(_, c)| c.tag == tag)
            .map(|(i, c)| if i == 0 { c.samples.len().saturating_sub(self.head) } else { c.samples.len() })
            .sum()
    }

    /// Drop `tag`'s clips (tag 0 is never cancelled this way). When the tag's audio is playing
    /// (its front clip is partly mixed, or the last mix ended exactly on the boundary to it), only a
    /// `fade_frames` fade-out from the read position is kept, so the mix never clicks.
    fn cancel_tagged(&mut self, tag: u64, fade_frames: usize) {
        if tag == 0 {
            return;
        }
        let ch = self.ch.max(1);
        let mut keep_front: Option<AuxClip> = None;
        if self.head > 0 || self.last_tag == tag {
            if let Some(front) = self.clips.front() {
                if front.tag == tag {
                    let start = self.head.min(front.samples.len()) / ch * ch;
                    let n = (fade_frames * ch).min(front.samples.len() - start) / ch * ch;
                    let frames = n / ch;
                    let mut tail = front.samples[start..start + n].to_vec();
                    for (f, frame) in tail.chunks_exact_mut(ch).enumerate() {
                        let g = 1.0 - (f + 1) as f32 / (frames + 1) as f32;
                        frame.iter_mut().for_each(|s| *s *= g);
                    }
                    keep_front = Some(AuxClip { samples: tail, tag });
                }
            }
        }
        if let Some(fade) = keep_front {
            self.clips.pop_front();
            self.head = 0;
            self.clips.retain(|c| c.tag != tag);
            if !fade.samples.is_empty() {
                self.clips.push_front(fade);
            }
        } else {
            // The front clip is another owner's (its read offset stays valid), or it is this tag's
            // but has not started playing (nothing of it mixed, and the mix is not running into
            // it from an earlier clip of the tag): drop every clip of the tag.
            if self.clips.front().is_some_and(|c| c.tag == tag) {
                self.head = 0;
            }
            self.clips.retain(|c| c.tag != tag);
        }
        self.repair();
    }

    /// Copy up to `max` queued samples into `out` (cleared first), oldest first; returns how many.
    fn take_into(&mut self, out: &mut Vec<f32>, max: usize) -> usize {
        out.clear();
        while out.len() < max {
            let Some(front) = self.clips.front() else { break };
            self.last_tag = front.tag;
            let front = &front.samples;
            let avail = front.len().saturating_sub(self.head);
            let n = avail.min(max - out.len());
            out.extend_from_slice(&front[self.head..self.head + n]);
            self.head += n;
            if self.head >= front.len() {
                self.clips.pop_front();
                self.head = 0;
            }
        }
        if out.len() < max {
            // Ran dry inside this block: the next clip starts from silence.
            self.last_tag = 0;
        }
        self.queued = self.queued.saturating_sub(out.len());
        out.len()
    }

    /// Drop everything and set the capture format new pushes are converted to (0 = stopped).
    pub(super) fn reset(&mut self, rate: u32, ch: usize) {
        self.clips.clear();
        self.head = 0;
        self.queued = 0;
        self.last_tag = 0;
        self.rate = rate;
        self.ch = ch.max(1);
    }

    /// Restore the bookkeeping invariants after a panic interrupted an update (poisoned lock).
    fn repair(&mut self) {
        while self.clips.front().is_some_and(|c| self.head >= c.samples.len()) {
            self.clips.pop_front();
            self.head = 0;
        }
        if self.clips.is_empty() {
            self.head = 0;
        }
        let total: usize = self.clips.iter().map(|c| c.samples.len()).sum();
        self.queued = total.saturating_sub(self.head);
    }
}

/// The aux queue shared by pushers (IPC / TTS threads) and the worker.
#[derive(Default)]
pub(crate) struct AuxFeed {
    queue: Mutex<AuxQueue>,
    /// Signalled when the mixer consumed audio (room may have appeared) or the run ended.
    space: Condvar,
}

impl AuxFeed {
    /// Lock the queue, recovering (and repairing) it if a panic poisoned the lock.
    pub(super) fn lock(&self) -> MutexGuard<'_, AuxQueue> {
        match self.queue.lock() {
            Ok(g) => g,
            Err(poisoned) => {
                let mut g = poisoned.into_inner();
                g.repair();
                self.queue.clear_poison();
                g
            }
        }
    }

    /// Start (capture format) or stop (`rate == 0`) accepting audio; drops anything queued and
    /// wakes blocked pushers so they re-check.
    pub(super) fn reset(&self, rate: u32, ch: usize) {
        self.lock().reset(rate, ch);
        self.space.notify_all();
    }

    /// Capture format pushes are converted to, `None` while not running.
    fn format(&self) -> Option<(u32, usize)> {
        let q = self.lock();
        (q.rate > 0).then_some((q.rate, q.ch))
    }

    /// Queue a clip in the capture format without waiting.
    #[cfg(test)]
    pub(super) fn try_push(&self, clip: Vec<f32>) -> Result<(), AuxPushError> {
        self.lock().try_push(clip)
    }

    /// Queue a clip converted to `format` without waiting; refused as not running when the run
    /// restarted in another capture format meanwhile.
    pub(super) fn try_push_in(&self, clip: Vec<f32>, format: (u32, usize)) -> Result<(), AuxPushError> {
        let mut q = self.lock();
        if (q.rate, q.ch) != format {
            return Err(AuxPushError::NotRunning);
        }
        q.try_push(clip)
    }

    /// Queue a clip converted to `format`, waiting up to `wait` for room. Waiting pushes are
    /// admitted strictly in arrival order (a ticket line), so sentences pushed back to back play
    /// in order even when both had to wait.
    pub(super) fn push_waiting(&self, clip: Vec<f32>, format: (u32, usize), wait: Duration) -> Result<(), AuxPushError> {
        let deadline = Instant::now() + wait;
        let mut q = self.lock();
        if (q.rate, q.ch) != format {
            return Err(AuxPushError::NotRunning);
        }
        if q.admits_new() {
            return q.push_checked(clip, true, 0);
        }
        let ticket = q.next_ticket;
        q.next_ticket = q.next_ticket.wrapping_add(1);
        q.waiting.push_back(ticket);
        let result = loop {
            if q.rate == 0 || (q.rate, q.ch) != format {
                break Err(AuxPushError::NotRunning);
            }
            if q.waiting.front() == Some(&ticket) && q.has_room() {
                q.waiting.pop_front();
                break q.push_checked(clip, true, 0);
            }
            let now = Instant::now();
            if now >= deadline {
                break Err(AuxPushError::Full { queued_secs: q.queued_secs() });
            }
            let (g, _) = self
                .space
                .wait_timeout(q, deadline - now)
                .unwrap_or_else(PoisonError::into_inner);
            q = g;
        };
        // Leave the line (already done when admitted), then let the next in line re-check: room
        // may be left over, or this ticket was the one holding it up.
        if let Some(pos) = q.waiting.iter().position(|&t| t == ticket) {
            q.waiting.remove(pos);
        }
        drop(q);
        self.space.notify_all();
        result
    }

    /// Mix queued audio into one capture block (see [`AuxMixer::mix`]).
    pub(super) fn mix_into(&self, block: &mut [f32], mixer: &mut AuxMixer, gate: AuxMicGate) {
        let mut q = self.lock();
        let before = q.queued;
        mixer.mix(block, &mut q, gate);
        let consumed = before != q.queued;
        let waiting = !q.waiting.is_empty();
        drop(q);
        if consumed && waiting {
            self.space.notify_all();
        }
    }
}

/// Worker-side aux mixer: mic gate envelope plus a scratch buffer (no allocation per block after
/// the first).
pub(super) struct AuxMixer {
    ch: usize,
    gain: f32,
    hold_left: usize,
    hold_frames: usize,
    attack: f32,
    release: f32,
    scratch: Vec<f32>,
}

impl AuxMixer {
    pub(super) fn new(rate: u32, ch: usize) -> Self {
        let rate = rate.max(1) as f32;
        let coef = |ms: f32| 1.0 - (-1.0 / (ms / 1000.0 * rate)).exp();
        Self {
            ch: ch.max(1),
            gain: 1.0,
            hold_left: 0,
            hold_frames: (GATE_HOLD_MS / 1000.0 * rate) as usize,
            attack: coef(GATE_ATTACK_MS),
            release: coef(GATE_RELEASE_MS),
            scratch: Vec::new(),
        }
    }

    /// Current mic gain of the gate (1 = open).
    #[cfg(test)]
    fn mic_gain(&self) -> f32 {
        self.gain
    }

    /// Mix queued aux audio into `block` (interleaved, capture format) and gate the mic while aux
    /// plays: `duck` lowers it, `mute` silences it, `none` leaves it. Output is clamped to +-1.
    pub(super) fn mix(&mut self, block: &mut [f32], q: &mut AuxQueue, gate: AuxMicGate) {
        let ch = self.ch;
        let frames = block.len() / ch;
        let taken = q.take_into(&mut self.scratch, frames * ch);
        let aux_frames = taken / ch;
        let closed = match gate {
            AuxMicGate::None => 1.0,
            AuxMicGate::Duck => DUCK_GAIN,
            AuxMicGate::Mute => 0.0,
        };
        if aux_frames == 0 && self.hold_left == 0 && self.gain >= 1.0 {
            return; // Nothing playing, gate open: the mic passes untouched.
        }
        for f in 0..frames {
            let active = f < aux_frames;
            if active {
                self.hold_left = self.hold_frames;
            } else if self.hold_left > 0 {
                self.hold_left -= 1;
            }
            let target = if active || self.hold_left > 0 { closed } else { 1.0 };
            let coef = if target < self.gain { self.attack } else { self.release };
            self.gain += (target - self.gain) * coef;
            if (self.gain - target).abs() < 1e-4 {
                self.gain = target;
            }
            for c in 0..ch {
                let i = f * ch + c;
                let aux = if active { self.scratch[i] } else { 0.0 };
                block[i] = (block[i] * self.gain + aux).clamp(-1.0, 1.0);
            }
        }
    }
}

/// Decode a WAV (PCM int or float) into interleaved f32, its rate and channel count.
fn decode_wav(raw: Vec<u8>) -> Result<(Vec<f32>, u32, usize), String> {
    let mut reader = WavReader::new(Cursor::new(raw)).map_err(|e| format!("WAV: {}", e))?;
    let spec = reader.spec();
    let ch = usize::from(spec.channels.max(1));
    let samples: Result<Vec<f32>, String> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.map_err(|e| e.to_string())).collect(),
        hound::SampleFormat::Int => {
            let bits = u32::from(spec.bits_per_sample.clamp(1, 32));
            let scale = 1.0 / f64::from(1u32 << (bits - 1)).max(1.0);
            reader
                .samples::<i32>()
                .map(|s| s.map_err(|e| e.to_string()).map(|v| (f64::from(v) * scale) as f32))
                .collect()
        }
    };
    Ok((samples?, spec.sample_rate, ch))
}

fn gcd_u32(mut a: u32, mut b: u32) -> u32 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a.max(1)
}

/// Band-limited resampling of a whole clip (interleaved) from `from` to `to` Hz: exactly
/// `round(frames * to / from)` frames, time-aligned (the resampler delay is removed).
pub(super) fn resample_clip(interleaved: &[f32], ch: usize, from: u32, to: u32) -> Result<Vec<f32>, String> {
    let ch = ch.max(1);
    let frames = interleaved.len() / ch;
    if from == to || frames == 0 || from == 0 || to == 0 {
        return Ok(interleaved[..frames * ch].to_vec());
    }
    let expected = ((frames as u64 * u64::from(to) + u64::from(from) / 2) / u64::from(from)) as usize;
    // rubato's FFT resampler works in blocks of `from / gcd` input frames and reports half its
    // output FFT length as the delay; an odd output FFT would leave the clip half a sample late,
    // so use an even number of blocks (about 1024 input frames per call).
    let g = gcd_u32(from, to);
    let block_in = (from / g) as usize;
    let blocks = 1024usize.div_ceil(block_in).max(2).next_multiple_of(2);
    let mut rs = FftFixedIn::<f32>::new(from as usize, to as usize, blocks * block_in, 1, ch)
        .map_err(|e| format!("aux resampler {from}->{to}: {e}"))?;
    let delay = rs.output_delay();
    let planar: Vec<Vec<f32>> = (0..ch).map(|c| interleaved.iter().skip(c).step_by(ch).copied().collect()).collect();
    let mut out: Vec<Vec<f32>> = vec![Vec::with_capacity(expected + delay + 2 * rs.output_frames_max()); ch];
    let mut buf = rs.output_buffer_allocate(true);
    let mut pos = 0usize;
    let mut process = |rs: &mut FftFixedIn<f32>, input: Option<&[&[f32]]>, out: &mut Vec<Vec<f32>>| -> Result<usize, String> {
        let (used, produced) = match input {
            Some(i) if i[0].len() >= rs.input_frames_next() => rs.process_into_buffer(i, &mut buf, None),
            Some(i) => rs.process_partial_into_buffer(Some(i), &mut buf, None),
            None => rs.process_partial_into_buffer(None::<&[&[f32]]>, &mut buf, None),
        }
        .map_err(|e| format!("aux resampler: {e}"))?;
        for (o, b) in out.iter_mut().zip(buf.iter()) {
            o.extend_from_slice(&b[..produced]);
        }
        Ok(used)
    };
    while pos < frames {
        let n = rs.input_frames_next().min(frames - pos);
        let slices: Vec<&[f32]> = planar.iter().map(|p| &p[pos..pos + n]).collect();
        process(&mut rs, Some(&slices), &mut out)?;
        pos += n;
    }
    // Push the filter's delayed tail out.
    let mut guard = 0;
    while out[0].len() < delay + expected && guard < 64 {
        process(&mut rs, None, &mut out)?;
        guard += 1;
    }
    let mut interleaved_out = Vec::with_capacity(expected * ch);
    for f in delay..(delay + expected).min(out[0].len()) {
        for o in out.iter() {
            interleaved_out.push(o[f]);
        }
    }
    interleaved_out.resize(expected * ch, 0.0);
    Ok(interleaved_out)
}

/// Convert a clip (any rate / channel count) to the capture format.
fn to_capture_format(samples: Vec<f32>, sample_rate: u32, channels: usize, rate: u32, ch: usize) -> Result<Vec<f32>, String> {
    let channels = channels.max(1);
    let adapted = if channels != ch {
        let mut v = Vec::new();
        adapt_channels_interleaved_into(&samples[..samples.len() / channels * channels], channels, ch, &mut v);
        v
    } else {
        samples
    };
    resample_clip(&adapted, ch, sample_rate, rate)
}

/// Check a source clip before any conversion: a sane sample rate (a header claiming 1 Hz would
/// otherwise expand into billions of samples) and at most [`AUX_MAX_CLIP_SECS`] long, measured
/// in source frames, so an over-long clip is refused before it is resampled.
fn check_source_clip(samples: usize, sample_rate: u32, channels: usize) -> Result<(), String> {
    if !(AUX_MIN_SAMPLE_RATE..=AUX_MAX_SAMPLE_RATE).contains(&sample_rate) {
        return Err(format!(
            "Aux audio sample rate {sample_rate} Hz is not supported ({AUX_MIN_SAMPLE_RATE}-{AUX_MAX_SAMPLE_RATE} Hz)"
        ));
    }
    let frames = (samples / channels.max(1)) as u64;
    if frames > AUX_MAX_CLIP_SECS * u64::from(sample_rate) {
        return Err(AuxPushError::TooLong {
            clip_secs: frames / u64::from(sample_rate),
        }
        .message());
    }
    Ok(())
}

fn push_pcm(state: &VoiceChangerState, samples: Vec<f32>, sample_rate: u32, channels: usize, wait: Option<Duration>) -> Result<(), String> {
    let Some((rate, ch)) = state.aux_feed.format() else {
        return Err(AuxPushError::NotRunning.message());
    };
    check_source_clip(samples.len(), sample_rate, channels)?;
    // A fast refusal before resampling a clip that cannot be queued anyway.
    if wait.is_none() {
        let q = state.aux_feed.lock();
        if !q.admits_new() {
            return Err(AuxPushError::Full { queued_secs: q.queued_secs() }.message());
        }
    }
    let clip = to_capture_format(samples, sample_rate, channels, rate, ch)?;
    let res = match wait {
        Some(w) => state.aux_feed.push_waiting(clip, (rate, ch), w),
        None => state.aux_feed.try_push_in(clip, (rate, ch)),
    };
    res.map_err(|e| e.message())
}

/// IPC body for [`voice_changer_push_aux_wav`].
/// Prefer `wavBytesB64` from the webview (compact over IPC vs. JSON number arrays).
#[derive(Deserialize)]
pub(super) struct PushAuxWavRequest {
    #[serde(default, rename = "wavBytes", alias = "wav_bytes")]
    wav_bytes: Option<Vec<u8>>,
    #[serde(default, rename = "wavBytesB64", alias = "wav_bytes_b64")]
    wav_bytes_b64: Option<String>,
}

fn decode_push_aux_wav_payload(req: PushAuxWavRequest) -> Result<Vec<u8>, String> {
    if let Some(s) = req.wav_bytes_b64.as_ref().map(|x| x.trim()).filter(|x| !x.is_empty()) {
        return B64_ENGINE
            .decode(s.as_bytes())
            .map_err(|e| format!("Invalid base64 WAV: {}", e));
    }
    req.wav_bytes
        .filter(|b| !b.is_empty())
        .ok_or_else(|| "Missing WAV payload (wavBytes or wavBytesB64).".into())
}

/// Decode WAV PCM and queue it into the running voice changer (used by ONNX/sidecar TTS
/// in-process). Never blocks: when [`AUX_QUEUE_CAP_SECS`] or more is already queued the clip is
/// refused whole with an error (back-pressure), never truncated.
pub fn push_aux_wav_bytes(state: &VoiceChangerState, raw: Vec<u8>) -> Result<(), String> {
    if raw.is_empty() {
        return Ok(());
    }
    let (samples, sr, ch) = decode_wav(raw)?;
    push_pcm(state, samples, sr, ch, None)
}

/// The unified TTS sink's route into the voice changer (`tts_sink`, P2-G). The sink converts each
/// utterance to the capture format itself (a stateful resampler, so streamed audio has no seams)
/// and feeds it in small tagged blocks, one utterance after another, so these never wait: a full
/// queue hands the block back and the sink retries it on its next tick (back-pressure).
impl VoiceChangerState {
    /// Capture format `(sample rate, channels)` TTS audio must be converted to; `None` while the
    /// voice changer is not running.
    pub(crate) fn tts_aux_format(&self) -> Option<(u32, usize)> {
        self.aux_feed.format()
    }

    /// Queue one block of `tag`'s audio (interleaved, already in `format`) without waiting.
    /// `Err(Some(block))`: the queue is full (or older pushes are waiting); retry the same block
    /// later. `Err(None)`: not running, or restarted in another capture format (re-read
    /// [`Self::tts_aux_format`]).
    pub(crate) fn tts_aux_try_push(&self, tag: u64, block: Vec<f32>, format: (u32, usize)) -> Result<(), Option<Vec<f32>>> {
        let mut q = self.aux_feed.lock();
        if q.rate == 0 || (q.rate, q.ch) != format {
            return Err(None);
        }
        if !q.admits_new() {
            return Err(Some(block));
        }
        q.push_checked(block, true, tag).map_err(|_| None)
    }

    /// Samples (interleaved, capture format) of `tag`'s audio not yet mixed into the voice.
    pub(crate) fn tts_aux_pending(&self, tag: u64) -> usize {
        self.aux_feed.lock().pending_tagged(tag)
    }

    /// Drop `tag`'s queued audio (stop / skip); a block already playing fades out over a few ms.
    pub(crate) fn tts_aux_cancel(&self, tag: u64) {
        {
            let mut q = self.aux_feed.lock();
            let fade = (u64::from(q.rate) * u64::from(CANCEL_FADE_MS) / 1000) as usize;
            q.cancel_tagged(tag, fade);
        }
        self.aux_feed.space.notify_all();
    }
}

/// Push decoded f32 PCM (interleaved) to be mixed into the live capture stream at `sample_rate` Hz.
/// Waits (up to 30 s) for room when the queue is full.
#[command]
pub(super) async fn voice_changer_push_aux_pcm(
    state: State<'_, VoiceChangerState>,
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
) -> Result<(), String> {
    let st = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || push_pcm(&st, samples, sample_rate, usize::from(channels.max(1)), Some(AUX_PUSH_WAIT)))
        .await
        .map_err(|e| format!("aux push task: {e}"))?
}

/// WAV (PCM) from TTS etc., mixed into the voice changer input while running. Waits (up to 30 s)
/// for room when the queue is full.
#[command]
pub(super) async fn voice_changer_push_aux_wav(
    state: State<'_, VoiceChangerState>,
    req: PushAuxWavRequest,
) -> Result<(), String> {
    let raw = decode_push_aux_wav_payload(req)?;
    let st = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        if raw.is_empty() {
            return Ok(());
        }
        let (samples, sr, ch) = decode_wav(raw)?;
        push_pcm(&st, samples, sr, ch, Some(AUX_PUSH_WAIT))
    })
    .await
    .map_err(|e| format!("aux push task: {e}"))?
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn running_feed(rate: u32, ch: usize) -> AuxFeed {
        let f = AuxFeed::default();
        f.reset(rate, ch);
        f
    }

    fn wav_bytes(samples: &[i16], rate: u32, ch: u16) -> Vec<u8> {
        let mut cur = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec { channels: ch, sample_rate: rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
            let mut w = hound::WavWriter::new(&mut cur, spec).unwrap();
            for &s in samples {
                w.write_sample(s).unwrap();
            }
            w.finalize().unwrap();
        }
        cur.into_inner()
    }

    /// A 60 s TTS clip (the old queue kept only the last 2-4 s of it) plays intact: every sample
    /// reaches the mix, in order, and nothing else does.
    #[test]
    fn sixty_second_aux_clip_stays_intact() {
        let (rate, ch) = (48_000u32, 2usize);
        let feed = running_feed(rate, ch);
        let n = 60 * rate as usize * ch;
        // Distinct, exactly representable values so any loss or reordering shows.
        let clip: Vec<f32> = (0..n).map(|i| ((i % 4096) as f32 - 2048.0) / 8192.0).collect();
        feed.try_push(clip.clone()).unwrap();
        assert_eq!(feed.lock().queued_samples(), n);
        let mut mixer = AuxMixer::new(rate, ch);
        let mut played = Vec::with_capacity(n);
        let mut block = vec![0.0f32; 512 * ch];
        while played.len() < n {
            block.iter_mut().for_each(|s| *s = 0.0);
            feed.mix_into(&mut block, &mut mixer, AuxMicGate::None);
            played.extend_from_slice(&block);
        }
        assert_eq!(&played[..n], &clip[..]);
        assert!(played[n..].iter().all(|&s| s == 0.0));
        assert_eq!(feed.lock().queued_samples(), 0);
    }

    /// Back-pressure: clips are refused whole (never truncated, nothing dropped from the front)
    /// once the cap is reached, and accepted again when it drains.
    #[test]
    fn full_queue_refuses_whole_clips_and_recovers() {
        let (rate, ch) = (16_000u32, 1usize);
        let feed = running_feed(rate, ch);
        let sec = rate as usize;
        feed.try_push(vec![0.1; 59 * sec]).unwrap();
        // Below the cap: accepted whole even though it ends above the cap.
        feed.try_push(vec![0.2; 5 * sec]).unwrap();
        assert_eq!(feed.lock().queued_samples(), 64 * sec);
        assert_eq!(feed.try_push(vec![0.3; sec]), Err(AuxPushError::Full { queued_secs: 64 }));
        assert_eq!(feed.lock().queued_samples(), 64 * sec, "a refused clip changes nothing");
        // Play 5 s: below the cap again.
        let mut mixer = AuxMixer::new(rate, ch);
        let mut block = vec![0.0f32; 5 * sec];
        feed.mix_into(&mut block, &mut mixer, AuxMicGate::None);
        assert!(block.iter().all(|&s| s == 0.1), "oldest audio plays first");
        feed.try_push(vec![0.3; sec]).unwrap();
        // Too long on its own, and not running.
        assert_eq!(
            feed.try_push(vec![0.0; (AUX_MAX_CLIP_SECS as usize + 1) * sec]),
            Err(AuxPushError::TooLong { clip_secs: AUX_MAX_CLIP_SECS + 1 })
        );
        feed.reset(0, 1);
        assert_eq!(feed.try_push(vec![0.1; 10]), Err(AuxPushError::NotRunning));
    }

    /// A waiting push is admitted as soon as the mixer frees room.
    #[test]
    fn waiting_push_is_admitted_when_the_mixer_drains() {
        let (rate, ch) = (8_000u32, 1usize);
        let feed = Arc::new(running_feed(rate, ch));
        feed.try_push(vec![0.1; 61 * rate as usize]).unwrap();
        let pusher = {
            let feed = feed.clone();
            std::thread::spawn(move || feed.push_waiting(vec![0.5; 100], (rate, ch), Duration::from_secs(20)))
        };
        // Wait until the pusher is blocked, then play 2 s.
        let deadline = Instant::now() + Duration::from_secs(10);
        while feed.lock().waiting.is_empty() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        let mut mixer = AuxMixer::new(rate, ch);
        let mut block = vec![0.0f32; 2 * rate as usize];
        feed.mix_into(&mut block, &mut mixer, AuxMicGate::None);
        assert_eq!(pusher.join().unwrap(), Ok(()));
        assert_eq!(feed.lock().queued_samples(), 59 * rate as usize + 100);
    }

    /// Pushes that had to wait are admitted in the order they arrived, and a push that does not
    /// wait never overtakes them: sentences 12 and 13 pushed back to back play as 12, 13.
    #[test]
    fn waiting_pushes_are_admitted_in_arrival_order() {
        let (rate, ch) = (8_000u32, 1usize);
        let feed = Arc::new(running_feed(rate, ch));
        let cap = AUX_QUEUE_CAP_SECS as usize * rate as usize;
        feed.try_push(vec![0.0; cap + rate as usize]).unwrap();
        let mut pushers = Vec::new();
        for (i, value) in [12.0f32, 13.0, 14.0].into_iter().enumerate() {
            let f = feed.clone();
            pushers.push(std::thread::spawn(move || f.push_waiting(vec![value; 10], (rate, ch), Duration::from_secs(20))));
            // Each one is in line before the next arrives.
            let deadline = Instant::now() + Duration::from_secs(10);
            while feed.lock().waiting.len() <= i && Instant::now() < deadline {
                std::thread::yield_now();
            }
            assert_eq!(feed.lock().waiting.len(), i + 1);
        }
        // A newcomer that does not wait is refused while others wait, even once room appears.
        let mut mixer = AuxMixer::new(rate, ch);
        let mut block = vec![0.0f32; 2 * rate as usize];
        {
            let mut q = feed.lock();
            mixer.mix(&mut block, &mut q, AuxMicGate::None); // room now, nobody woken yet
            assert!(q.has_room());
            assert!(matches!(q.try_push(vec![99.0; 10]), Err(AuxPushError::Full { .. })));
        }
        feed.space.notify_all();
        for p in pushers {
            assert_eq!(p.join().unwrap(), Ok(()));
        }
        // Play everything: the three clips come out in arrival order, after the backlog.
        let q = feed.lock();
        let tail: Vec<f32> = q.clips.iter().skip(1).map(|c| c.samples[0]).collect();
        assert_eq!(tail, vec![12.0, 13.0, 14.0]);
        assert!(q.waiting.is_empty());
    }

    /// A waiting push gives up at its deadline (refused whole, out of the line) and one waiting
    /// across a restart in another capture format is refused instead of queued in the old format.
    #[test]
    fn waiting_push_times_out_or_sees_a_restart() {
        let (rate, ch) = (8_000u32, 1usize);
        let feed = Arc::new(running_feed(rate, ch));
        feed.try_push(vec![0.0; 61 * rate as usize]).unwrap();
        let err = feed.push_waiting(vec![0.5; 10], (rate, ch), Duration::from_millis(30)).unwrap_err();
        assert!(matches!(err, AuxPushError::Full { .. }), "{err:?}");
        assert!(feed.lock().waiting.is_empty());
        let f = feed.clone();
        let pusher = std::thread::spawn(move || f.push_waiting(vec![0.5; 10], (rate, ch), Duration::from_secs(20)));
        let deadline = Instant::now() + Duration::from_secs(10);
        while feed.lock().waiting.is_empty() && Instant::now() < deadline {
            std::thread::yield_now();
        }
        feed.reset(48_000, 2);
        assert_eq!(pusher.join().unwrap(), Err(AuxPushError::NotRunning));
        assert_eq!(feed.try_push_in(vec![0.5; 10], (rate, ch)), Err(AuxPushError::NotRunning));
        assert_eq!(feed.lock().queued_samples(), 0);
    }

    /// Source clips are checked before any conversion: an absurd header rate (which would expand
    /// into billions of samples) and an over-long clip are refused without resampling.
    #[test]
    fn source_clips_are_checked_before_conversion() {
        assert!(check_source_clip(48_000, 48_000, 1).is_ok());
        let e = check_source_clip(1_000_000, 1, 1).unwrap_err();
        assert!(e.contains("sample rate 1 Hz"), "{e}");
        assert!(check_source_clip(10, 0, 1).is_err());
        assert!(check_source_clip(10, 768_000, 1).is_err());
        assert!(check_source_clip(10, AUX_MIN_SAMPLE_RATE, 1).is_ok());
        assert!(check_source_clip(10, AUX_MAX_SAMPLE_RATE, 2).is_ok());
        // 11 minutes of stereo at 24 kHz: refused by length in source frames.
        let long = 11 * 60 * 24_000 * 2;
        assert_eq!(
            check_source_clip(long, 24_000, 2),
            Err(AuxPushError::TooLong { clip_secs: 660 }.message())
        );
        assert!(check_source_clip(AUX_MAX_CLIP_SECS as usize * 24_000 * 2, 24_000, 2).is_ok());
        // Through the push path: refused while running, without touching the queue.
        let st = VoiceChangerState::new();
        st.aux_feed.reset(48_000, 1);
        let e = push_aux_wav_bytes(&st, wav_bytes(&[0; 64], 1, 1)).unwrap_err();
        assert!(e.contains("sample rate"), "{e}");
        assert_eq!(st.aux_feed.lock().queued_samples(), 0);
        push_aux_wav_bytes(&st, wav_bytes(&[1_000; 64], 16_000, 1)).unwrap();
        assert_eq!(st.aux_feed.lock().queued_samples(), 192);
    }

    /// A panic while the lock was held (poisoned) does not wedge the voice changer: pushes and
    /// mixing keep working and the bookkeeping is repaired.
    #[test]
    fn poisoned_aux_lock_recovers() {
        let feed = Arc::new(running_feed(16_000, 1));
        feed.try_push(vec![0.25; 1_000]).unwrap();
        let f2 = feed.clone();
        let _ = std::thread::spawn(move || {
            let mut q = f2.queue.lock().unwrap();
            // Leave the bookkeeping inconsistent, then panic with the lock held.
            q.head = 5_000;
            q.queued = 12_345;
            panic!("TTS caller crashed");
        })
        .join();
        assert!(feed.queue.is_poisoned());
        {
            let q = feed.lock();
            assert_eq!(q.queued_samples(), 0, "repaired: the half-updated clip is gone");
        }
        assert!(!feed.queue.is_poisoned());
        feed.try_push(vec![0.5; 100]).unwrap();
        let mut mixer = AuxMixer::new(16_000, 1);
        let mut block = vec![0.0f32; 160];
        feed.mix_into(&mut block, &mut mixer, AuxMicGate::None);
        assert!(block[..100].iter().all(|&s| s == 0.5) && block[100..].iter().all(|&s| s == 0.0));
    }

    /// Duck lowers and mute silences the mic while aux plays (smoothly), the hold bridges short
    /// pauses, and the gate opens again afterwards. `none` never touches the mic.
    #[test]
    fn mic_gate_ducks_mutes_and_releases() {
        let (rate, ch) = (16_000u32, 1usize);
        for (gate, closed) in [(AuxMicGate::Duck, DUCK_GAIN), (AuxMicGate::Mute, 0.0), (AuxMicGate::None, 1.0)] {
            let feed = running_feed(rate, ch);
            let mut mixer = AuxMixer::new(rate, ch);
            feed.try_push(vec![0.0; rate as usize]).unwrap(); // 1 s of (silent) TTS
            let mut block = vec![0.5f32; 1_600]; // 100 ms blocks of mic at 0.5
            feed.mix_into(&mut block, &mut mixer, gate);
            // No click: the first sample is still (almost) the open mic.
            assert!(block[0] > 0.45, "{gate:?}: {}", block[0]);
            // Settled to the closed gain within the block.
            assert!((block[1_599] - 0.5 * closed).abs() < 1e-3, "{gate:?}: {}", block[1_599]);
            for _ in 0..9 {
                block.fill(0.5);
                feed.mix_into(&mut block, &mut mixer, gate);
            }
            // Aux has run out; the hold keeps the gate closed for a moment...
            block.fill(0.5);
            feed.mix_into(&mut block, &mut mixer, gate);
            assert!((mixer.mic_gain() - closed).abs() < 1e-3, "{gate:?}: held");
            // ...then it releases back to the open mic.
            for _ in 0..12 {
                block.fill(0.5);
                feed.mix_into(&mut block, &mut mixer, gate);
            }
            assert_eq!(mixer.mic_gain(), 1.0, "{gate:?}");
            assert!(block.iter().all(|&s| s == 0.5), "{gate:?}");
        }
    }

    /// Clips are resampled band-limited to the capture format with the exact length.
    #[test]
    fn clips_resample_to_the_capture_format_exactly() {
        for &(from, to) in &[(22_050u32, 48_000u32), (24_000, 44_100), (48_000, 16_000), (44_100, 48_000)] {
            let frames = from as usize * 3 / 2 + 7; // 1.5 s + a few frames
            let tone: Vec<f32> = (0..frames).map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / from as f32).sin()).collect();
            let out = resample_clip(&tone, 1, from, to).unwrap();
            let expected = ((frames as u64 * u64::from(to) + u64::from(from) / 2) / u64::from(from)) as usize;
            assert_eq!(out.len(), expected, "{from}->{to}");
            // Time-aligned (the delay is removed) and band-limited: away from the clip edges the
            // output is the same 440 Hz tone.
            let mut max_err = 0.0f32;
            for (k, &v) in out.iter().enumerate().take(expected - 2_000).skip(2_000) {
                let t = k as f32 / to as f32;
                max_err = max_err.max((v - 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()).abs());
            }
            assert!(max_err < 5e-3, "{from}->{to}: max error {max_err}");
        }
    }

    /// WAV decoding covers 16- and 32-bit integer PCM (the old 32-bit scale was off by 2x), and a
    /// stereo WAV mixes into a mono capture as the channel average.
    #[test]
    fn wav_push_decodes_and_adapts_channels() {
        let (samples, sr, ch) = decode_wav(wav_bytes(&[16_384, -16_384, 8_192, 0], 16_000, 2)).unwrap();
        assert_eq!((sr, ch), (16_000, 2));
        assert_eq!(samples, vec![0.5, -0.5, 0.25, 0.0]);
        let mono = to_capture_format(samples, 16_000, 2, 16_000, 1).unwrap();
        assert_eq!(mono, vec![0.0, 0.125]);

        let mut cur = Cursor::new(Vec::new());
        {
            let spec = hound::WavSpec { channels: 1, sample_rate: 8_000, bits_per_sample: 32, sample_format: hound::SampleFormat::Int };
            let mut w = hound::WavWriter::new(&mut cur, spec).unwrap();
            w.write_sample(i32::MIN).unwrap();
            w.write_sample(1i32 << 30).unwrap();
            w.finalize().unwrap();
        }
        let (s32, _, _) = decode_wav(cur.into_inner()).unwrap();
        assert_eq!(s32, vec![-1.0, 0.5]);
    }

    /// TTS sink route: tagged blocks are counted per tag, a cancelled tag that did not start is
    /// dropped whole, one that is playing ends with a short monotonic fade (no click), and other
    /// owners' audio (and the untagged tag 0) is never touched.
    #[test]
    fn tagged_clips_count_and_cancel_with_a_fade() {
        let (rate, ch) = (16_000u32, 2usize);
        let feed = running_feed(rate, ch);
        {
            let mut q = feed.lock();
            q.try_push_tagged(vec![0.5; 1_000], 7).unwrap();
            q.try_push_tagged(vec![0.25; 400], 8).unwrap();
            q.try_push_tagged(vec![0.5; 600], 7).unwrap();
            q.try_push(vec![0.1; 200]).unwrap();
            assert_eq!(q.pending_tagged(7), 1_600);
            assert_eq!(q.pending_tagged(8), 400);
            assert_eq!(q.pending_tagged(0), 200);
        }
        // Play 100 frames of tag 7's first clip, then cancel tag 7.
        let mut mixer = AuxMixer::new(rate, ch);
        let mut block = vec![0.0f32; 200];
        feed.mix_into(&mut block, &mut mixer, AuxMicGate::None);
        assert!(block.iter().all(|&s| s == 0.5));
        let fade_frames = 80usize; // 5 ms at 16 kHz
        feed.lock().cancel_tagged(7, fade_frames);
        {
            let q = feed.lock();
            assert_eq!(q.pending_tagged(7), fade_frames * ch, "only the fade tail is left");
            assert_eq!(q.pending_tagged(8), 400);
            assert_eq!(q.pending_tagged(0), 200);
            assert_eq!(q.queued_samples(), fade_frames * ch + 600);
        }
        let mut tail = vec![0.0f32; fade_frames * ch];
        feed.mix_into(&mut tail, &mut mixer, AuxMicGate::None);
        let left: Vec<f32> = tail.iter().step_by(ch).copied().collect();
        assert!(left[0] < 0.5 && left[0] > 0.45, "fade starts near the playing level: {}", left[0]);
        assert!(left.windows(2).all(|w| w[1] <= w[0]), "fade is monotonic");
        assert!(*left.last().unwrap() < 0.01);
        // Then tag 8 and the untagged clip follow, intact.
        let mut rest = vec![0.0f32; 600];
        feed.mix_into(&mut rest, &mut mixer, AuxMicGate::None);
        assert!(rest[..400].iter().all(|&s| s == 0.25) && rest[400..].iter().all(|&s| s == 0.1));
        // Cancelling a tag whose clips did not start drops them whole; tag 0 cannot be cancelled.
        let mut q = feed.lock();
        q.try_push_tagged(vec![0.3; 10], 9).unwrap();
        q.try_push(vec![0.3; 10]).unwrap();
        q.cancel_tagged(9, fade_frames);
        q.cancel_tagged(0, fade_frames);
        assert_eq!((q.pending_tagged(9), q.pending_tagged(0), q.queued_samples()), (0, 10, 10));
    }

    /// Review finding: the sink feeds an utterance as many small clips, so a mix can end exactly on
    /// the boundary between two of them (read offset 0 on the next one). A cancel at that moment
    /// still fades out instead of cutting the voice from full level to silence; after a gap (the
    /// queue ran dry) the tag's next clip has not started and is dropped whole.
    #[test]
    fn cancel_on_a_clip_boundary_still_fades() {
        let (rate, ch) = (48_000u32, 1usize);
        let fade_frames = 240usize; // 5 ms
        let feed = running_feed(rate, ch);
        {
            let mut q = feed.lock();
            for _ in 0..3 {
                q.try_push_tagged(vec![0.5; 4_800], 7).unwrap();
            }
            q.try_push_tagged(vec![0.25; 480], 8).unwrap();
        }
        // 30 capture chunks of 160 frames end exactly on the first clip's last sample.
        let mut mixer = AuxMixer::new(rate, ch);
        let mut block = vec![0.0f32; 160];
        for _ in 0..30 {
            block.fill(0.0);
            feed.mix_into(&mut block, &mut mixer, AuxMicGate::None);
        }
        assert_eq!(feed.lock().head, 0, "the mix ended on the boundary");
        feed.lock().cancel_tagged(7, fade_frames);
        assert_eq!(feed.lock().pending_tagged(7), fade_frames, "a fade tail is kept");
        let mut tail = vec![0.0f32; fade_frames];
        feed.mix_into(&mut tail, &mut mixer, AuxMicGate::None);
        assert!(tail[0] > 0.45, "the fade continues from the playing level: {}", tail[0]);
        assert!(tail.windows(2).all(|w| w[1] <= w[0]), "fade is monotonic");
        assert!(*tail.last().unwrap() < 0.01);
        // Tag 8 follows intact.
        let mut rest = vec![0.0f32; 480];
        feed.mix_into(&mut rest, &mut mixer, AuxMicGate::None);
        assert!(rest.iter().all(|&s| s == 0.25));

        // A gap: the queue ran dry, so the tag's next clip starts from silence and a cancel
        // before it plays drops it whole.
        let mut silence = vec![0.0f32; 160];
        feed.mix_into(&mut silence, &mut mixer, AuxMicGate::None);
        let mut q = feed.lock();
        q.try_push_tagged(vec![0.5; 4_800], 9).unwrap();
        q.cancel_tagged(9, fade_frames);
        assert_eq!(q.queued_samples(), 0);
        // Not the tag that was playing: dropped whole as well.
        q.try_push_tagged(vec![0.5; 480], 10).unwrap();
        q.try_push_tagged(vec![0.5; 480], 11).unwrap();
        let mut scratch = Vec::new();
        assert_eq!(q.take_into(&mut scratch, 480), 480);
        q.cancel_tagged(11, fade_frames);
        assert_eq!(q.queued_samples(), 0, "tag 10 ended on the boundary; tag 11 never started");
    }

    /// The sink's non-blocking push: refused as not running (or after a format change), handed back
    /// whole while the queue is full, accepted and counted otherwise; cancel wakes waiters.
    #[test]
    fn tts_route_push_hands_back_blocks_when_full() {
        let st = VoiceChangerState::new();
        assert_eq!(st.tts_aux_format(), None);
        assert_eq!(st.tts_aux_try_push(3, vec![0.1; 10], (48_000, 1)), Err(None));
        st.aux_feed.reset(8_000, 1);
        assert_eq!(st.tts_aux_format(), Some((8_000, 1)));
        assert_eq!(st.tts_aux_try_push(3, vec![0.1; 10], (48_000, 1)), Err(None), "stale format");
        let cap = AUX_QUEUE_CAP_SECS as usize * 8_000;
        st.tts_aux_try_push(3, vec![0.1; cap], (8_000, 1)).unwrap();
        assert_eq!(st.tts_aux_try_push(4, vec![0.2; 5], (8_000, 1)), Err(Some(vec![0.2; 5])));
        assert_eq!(st.tts_aux_pending(3), cap);
        st.tts_aux_cancel(3);
        assert_eq!(st.tts_aux_pending(3), 0);
        st.tts_aux_try_push(4, vec![0.2; 5], (8_000, 1)).unwrap();
        assert_eq!(st.tts_aux_pending(4), 5);
    }
}
