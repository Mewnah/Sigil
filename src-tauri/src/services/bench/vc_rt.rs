//! The voice changer's real-time input and output path, for the paced driver.
//!
//! The paced benchmark must run the *production* latency budget, output queue, ring playout and
//! failed-block concealment, not a re-implementation that could drift from them, so it uses the
//! voice changer's own modules (`pub(crate)` since P1-I0):
//!
//! - `voice_changer/budget.rs`   → [`InputSide`]  (`push_capture`, `InputBacklog`, the budget)
//! - `voice_changer/playback.rs` → [`OutputSide`] (`OutputPlayback`, `RingPlayout`)
//! - `voice_changer/fallback.rs` → [`Conceal`]    (`MuteConceal`)
//!
//! The wrappers expose only what the driver needs and keep the worker's exact call order.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

use crate::services::neural_audio::chunk_processor::{apply_formant_interleaved, FormantLpState};
use crate::services::neural_audio::VcStreamHarness;
use crate::services::voice_changer::{budget, fallback, playback};

use super::spec::StreamSpec;

/// Capture ring + worker backlog with the latency budget (the worker's input side).
pub(super) struct InputSide {
    producer: rtrb::Producer<f32>,
    consumer: rtrb::Consumer<f32>,
    overflow: AtomicU64,
    backlog: budget::InputBacklog,
    ch: usize,
    budget_frames: usize,
}

/// What one wake's budget enforcement discarded.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct Dropped {
    /// Oldest backlog frames dropped.
    pub frames: usize,
    /// Frames the capture ring refused (lost before reaching the backlog).
    pub overflow_frames: u64,
}

impl InputSide {
    /// Sized exactly as `worker.rs` sizes them for this device format and block.
    pub(super) fn new(rate: u32, ch: usize, block_frames: usize, budget_ms: f64) -> Self {
        let ch = ch.max(1);
        let cap = budget::capture_ring_capacity(rate, ch, block_frames);
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(cap);
        let need = block_frames * ch;
        let backlog = budget::InputBacklog::new(ch, rate, budget_ms, cap + cap / 2 + need);
        Self {
            producer,
            consumer,
            overflow: AtomicU64::new(0),
            backlog,
            ch,
            budget_frames: budget::ms_to_frames(budget::clamp_latency_budget_ms(budget_ms), rate).max(1),
        }
    }

    /// One capture callback (the callback's `push_capture` plus its overflow count). Returns the
    /// frames written to the ring (the rest were refused).
    pub(super) fn capture(&mut self, data: &[f32]) -> usize {
        let frames = data.len() / self.ch;
        let lost = budget::push_capture(&mut self.producer, data, self.ch);
        if lost > 0 {
            self.overflow.fetch_add(lost as u64, Ordering::Relaxed);
        }
        frames - lost
    }

    /// The worker's wake: ingest everything captured, then enforce the budget.
    pub(super) fn wake(&mut self, block_frames: usize) -> Dropped {
        self.backlog.ingest(&mut self.consumer, &self.overflow);
        self.backlog
            .enforce(block_frames)
            .map(|d| Dropped {
                frames: d.frames,
                overflow_frames: d.overflow_frames,
            })
            .unwrap_or_default()
    }

    /// Frames waiting (backlog plus anything still in the ring), as the worker's metric counts.
    pub(super) fn waiting_frames(&self) -> usize {
        self.backlog.frames() + self.consumer.slots() / self.ch
    }

    pub(super) fn budget_frames(&self) -> usize {
        self.budget_frames
    }

    pub(super) fn take_block(&mut self, out: &mut Vec<f32>, frames: usize) -> bool {
        self.backlog.take_block(out, frames)
    }
}

/// Worker output queue → lock-free ring → the playback device's callback.
pub(super) struct OutputSide {
    playback: playback::OutputPlayback,
    producer: rtrb::Producer<f32>,
    playout: playback::RingPlayout,
    ring_cap: usize,
    ch: usize,
    underruns: Arc<AtomicU64>,
    buf: Vec<f32>,
}

/// One device callback: which ring frames it played.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Played {
    /// Ring frames read (the rest of the callback was held / silent).
    pub frames: usize,
    /// Frames the callback asked for.
    pub requested: usize,
}

/// One worker push: what happened to the pushed frames.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) struct Pushed {
    /// Oldest queued frames the queue cap trimmed.
    pub trimmed_frames: usize,
    /// Frames handed to the ring.
    pub flushed_frames: usize,
}

impl OutputSide {
    /// Queue, ring and playout sized as `worker.rs` sizes them (`prefill_ms` / `max_queue_ms`
    /// already within the voice changer's clamp).
    pub(super) fn new(rate: u32, ch: usize, prefill_ms: f64, max_queue_ms: f64, in_rate: u32, in_ch: usize, capture_burst_samples: usize) -> Self {
        let ch = ch.max(1);
        let ms_to_samples = |ms: f64| (f64::from(rate) * ms / 1000.0).ceil() as usize * ch;
        let prefill = ms_to_samples(prefill_ms);
        let max_queue = ms_to_samples(max_queue_ms);
        let ring_cap = max_queue
            .saturating_mul(2)
            .clamp(playback::OUT_RING_MIN_SAMPLES, playback::OUT_RING_MAX_SAMPLES)
            .max(prefill.saturating_add(1));
        let (producer, consumer) = rtrb::RingBuffer::<f32>::new(ring_cap);
        let callback_len = Arc::new(AtomicUsize::new(0));
        let in_burst = Arc::new(AtomicUsize::new(capture_burst_samples));
        let underruns = Arc::new(AtomicU64::new(0));
        let playback = playback::OutputPlayback::new(prefill, max_queue, ch, callback_len.clone())
            .with_capture_burst(in_burst, in_ch, in_rate, rate);
        let playout = playback::RingPlayout::new(consumer, ch, callback_len).with_underrun_counter(underruns.clone());
        Self {
            playback,
            producer,
            playout,
            ring_cap,
            ch,
            underruns,
            buf: Vec::new(),
        }
    }

    fn ring_used(&self) -> usize {
        self.ring_cap.saturating_sub(self.producer.slots())
    }

    /// The worker's push of one processed block (no silence truncation, as the default).
    pub(super) fn push(&mut self, interleaved: &[f32]) -> Pushed {
        let ch = self.ch;
        let used = self.ring_used();
        let q0 = self.playback.queue_len();
        self.playback.push_interleaved(interleaved, false, 0.0, 0, used);
        let q1 = self.playback.queue_len();
        self.playback.flush_to_ring(&mut self.producer, used);
        let q2 = self.playback.queue_len();
        Pushed {
            trimmed_frames: (q0 + interleaved.len()).saturating_sub(q1) / ch,
            flushed_frames: q1.saturating_sub(q2) / ch,
        }
    }

    /// One device callback of `frames` frames (the audio it played is appended to `record`).
    pub(super) fn callback(&mut self, frames: usize, record: Option<&mut Vec<f32>>) -> Played {
        let available = self.ring_used() / self.ch;
        self.buf.clear();
        self.buf.resize(frames * self.ch, 0.0);
        self.playout.fill(&mut self.buf);
        if let Some(r) = record {
            r.extend_from_slice(&self.buf);
        }
        Played {
            frames: available.min(frames),
            requested: frames,
        }
    }

    /// Callbacks that ran (partly) dry after playback began.
    pub(super) fn underruns(&self) -> u64 {
        self.underruns.load(Ordering::Relaxed)
    }

    /// Frames queued for playback (worker queue plus ring).
    pub(super) fn queued_frames(&self) -> usize {
        (self.playback.queue_len() + self.ring_used()) / self.ch
    }
}

/// The worker's RVC-mode block processing, in its order: input gain, controls, harness
/// conversion (a failed tick is muted by [`Conceal`] for exactly the owed frames), formant tilt,
/// output gain. Aux (TTS) mixing is not part of a benchmark block.
pub(super) struct VcTick {
    pub(super) harness: VcStreamHarness,
    conceal: Conceal,
    formant_states: Vec<FormantLpState>,
    ch_out: usize,
    r_out: u32,
    input_gain: f32,
    output_gain: f32,
    formant: f32,
    controls: (f32, f32, f32, i64),
}

/// One processed block.
pub(super) struct TickOut {
    /// Interleaved output, exactly the device frames the harness owed.
    pub(super) audio: Vec<f32>,
    pub(super) owed_frames: usize,
    pub(super) error: Option<String>,
}

impl VcTick {
    pub(super) fn new(harness: VcStreamHarness, stream: &StreamSpec) -> Self {
        let ch_out = usize::from(stream.output_channels).max(1);
        Self {
            harness,
            conceal: Conceal::new(ch_out, stream.output_rate),
            formant_states: vec![FormantLpState::default(); ch_out],
            ch_out,
            r_out: stream.output_rate,
            input_gain: stream.input_gain,
            output_gain: stream.output_gain,
            formant: stream.formant_shift,
            controls: (stream.pitch_semitones, stream.index_rate, stream.protect, stream.speaker_id),
        }
    }

    /// Process one capture block (interleaved at the input format).
    pub(super) fn process(&mut self, block: &mut [f32]) -> TickOut {
        if (self.input_gain - 1.0).abs() > 1e-6 {
            for s in block.iter_mut() {
                *s = (*s * self.input_gain).clamp(-1.0, 1.0);
            }
        }
        let (pitch, index_rate, protect, speaker) = self.controls;
        self.harness.set_controls(pitch, index_rate, protect, speaker);
        let (mut audio, error) = match self.harness.push_and_convert(block) {
            Ok(mut v) => {
                self.conceal.pass(&mut v);
                (v, None)
            }
            Err(e) => (self.conceal.mute(self.harness.last_owed_frames()), Some(e)),
        };
        apply_formant_interleaved(&mut audio, self.ch_out, self.formant, &mut self.formant_states, f64::from(self.r_out));
        if (self.output_gain - 1.0).abs() > 1e-6 {
            for s in audio.iter_mut() {
                *s = (*s * self.output_gain).clamp(-1.0, 1.0);
            }
        }
        TickOut {
            owed_frames: self.harness.last_owed_frames(),
            audio,
            error,
        }
    }
}

/// Failed-block concealment at the output (muted with click-free edges, never the dry voice).
pub(super) struct Conceal(fallback::MuteConceal);

impl Conceal {
    pub(super) fn new(ch: usize, rate: u32) -> Self {
        Self(fallback::MuteConceal::new(ch, rate))
    }

    pub(super) fn pass(&mut self, block: &mut [f32]) {
        self.0.pass(block);
    }

    pub(super) fn mute(&mut self, frames: usize) -> Vec<f32> {
        let mut out = Vec::new();
        self.0.mute_into(&mut out, frames);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_side_drops_the_oldest_beyond_the_budget() {
        let mut input = InputSide::new(48_000, 1, 512, 300.0);
        assert_eq!(input.budget_frames(), 14_400);
        // 1 s captured with no wake (a stalled worker): the next wake keeps at most the budget.
        for _ in 0..100 {
            assert_eq!(input.capture(&[0.1; 480]), 480);
        }
        let d = input.wake(512);
        assert!(d.frames > 0 && d.overflow_frames == 0, "{d:?}");
        assert!(input.waiting_frames() <= input.budget_frames(), "{}", input.waiting_frames());
        let mut block = Vec::new();
        assert!(input.take_block(&mut block, 512));
        assert_eq!(block.len(), 512);
    }

    #[test]
    fn output_side_counts_played_frames_and_underruns() {
        let mut out = OutputSide::new(48_000, 2, 35.0, 70.0, 48_000, 1, 480);
        // Before anything is queued: silence, not an underrun.
        assert_eq!(out.callback(480, None).frames, 0);
        assert_eq!(out.underruns(), 0);
        let block = vec![0.25f32; 512 * 2];
        let mut flushed = 0;
        for _ in 0..6 {
            flushed += out.push(&block).flushed_frames;
        }
        assert!(flushed >= 1_680, "prefill of 35 ms must reach the ring ({flushed})");
        let mut rec = Vec::new();
        let p = out.callback(480, Some(&mut rec));
        assert_eq!(p, Played { frames: 480, requested: 480 });
        assert_eq!(rec.len(), 960);
        // Drain everything: the dry callbacks after playback began count as underruns.
        for _ in 0..20 {
            out.callback(480, None);
        }
        assert!(out.underruns() > 0);
        let mut c = Conceal::new(2, 48_000);
        let mut b = vec![0.5f32; 64];
        c.pass(&mut b);
        // 5 ms at 48 kHz = 240 frames of fade, then silence.
        let muted = c.mute(400);
        assert_eq!(muted.len(), 800);
        assert!(muted[0] > 0.4 && muted[799] == 0.0, "fades from the last frame to silence");
    }
}
