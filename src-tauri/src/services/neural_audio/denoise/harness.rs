//! The 16 kHz denoiser inside the stream harness: [`DenoisedModel`] wraps the conversion model and
//! denoises its input stream (plan: "GTCRN at 16 kHz inside the harness").
//!
//! `VcStreamHarness` calls the model with its rolling 16 kHz context; `ConversionCtx::stream_pos_16k`
//! says where that context sits on the stream, so the wrapper knows exactly which samples are new
//! since its last call. It runs only those through the (streaming, constant-delay) denoiser,
//! appends them to its own denoised copy of the context, and hands the inner model the same window
//! of the denoised stream. The inner model therefore sees a clean, continuous stream delayed by the
//! denoiser's latency; the harness's length accounting is untouched (the call's shapes do not
//! change). A jump in the stream position (a harness that restarted its timeline) re-synchronizes
//! with fresh denoiser state; `reset_stream` (a capture gap or budget drop) clears the denoiser's
//! recurrent state and keeps going. A stream's first call (the whole context, whose length need
//! not be a multiple of the denoiser's chunk promise: a template voice's context is 160 x t samples
//! against GTCRN's 128-sample alignment) is padded at its front with silence up to that alignment
//! (`Denoiser::granularity`), so the denoiser's buffers never run dry and its delay stays the one it
//! reports.

use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::traits::{ConversionCtx, StreamReport, VoiceControls, VoiceConversionModel};

use super::Denoiser;

/// A conversion model whose 16 kHz input runs through a denoiser first.
pub struct DenoisedModel {
    inner: Box<dyn VoiceConversionModel>,
    den: Box<dyn Denoiser>,
    /// Denoised stream; its last sample is absolute 16 kHz index `end - 1`.
    ring: Vec<f32>,
    end: Option<u64>,
    scratch: Vec<f32>,
    /// When a denoiser failure was last logged (at most every 30 s).
    warned_at: Option<std::time::Instant>,
}

impl DenoisedModel {
    /// Wrap `inner` (16 kHz input) with `den` (16 kHz). Errors when the rates differ.
    pub fn new(inner: Box<dyn VoiceConversionModel>, den: Box<dyn Denoiser>) -> Result<DenoisedModel, String> {
        if inner.input_sample_rate() != den.sample_rate() {
            return Err(format!(
                "the {} Hz denoiser cannot run on a {} Hz model input",
                den.sample_rate(),
                inner.input_sample_rate()
            ));
        }
        Ok(DenoisedModel {
            inner,
            den,
            ring: Vec::new(),
            end: None,
            scratch: Vec::new(),
            warned_at: None,
        })
    }

    /// Silence to put before a stream's first call of `len` samples so it is a multiple of
    /// `granularity`.
    fn alignment_pad(len: usize, granularity: usize) -> usize {
        let g = granularity.max(1);
        (g - len % g) % g
    }

    /// Bring the denoised copy up to `ctx`'s end; returns the window matching `ctx.mono_16k`.
    fn follow(&mut self, ctx: &ConversionCtx<'_>) -> std::ops::Range<usize> {
        let len = ctx.mono_16k.len();
        let ctx_end = ctx.stream_pos_16k + len as u64;
        let (new, start_over) = match self.end {
            Some(end) if ctx_end >= end && ctx_end - end <= len as u64 => ((ctx_end - end) as usize, false),
            _ => {
                // First call, or the stream position jumped: start over from this context.
                self.den.reset();
                self.ring.clear();
                (len, true)
            }
        };
        if new > 0 {
            // A stream starts as if silence had been flowing: pad its first call up to the
            // denoiser's alignment (the padding's output is dropped; the timing is unchanged).
            let pad = if start_over { Self::alignment_pad(new, self.den.granularity()) } else { 0 };
            self.scratch.clear();
            self.scratch.resize(pad, 0.0);
            self.scratch.extend_from_slice(&ctx.mono_16k[len - new..]);
            if let Err(e) = self.den.process(&mut self.scratch) {
                // The audio passed un-denoised; the voice keeps converting.
                if self.warned_at.is_none_or(|t| t.elapsed() >= std::time::Duration::from_secs(30)) {
                    self.warned_at = Some(std::time::Instant::now());
                    tracing::warn!("voice changer: the in-harness denoiser failed ({e}); that audio passed un-denoised");
                }
            }
            self.ring.extend_from_slice(&self.scratch[pad..]);
        }
        self.end = Some(ctx_end);
        // Keep what this context length needs (plus room, so trims are rare and cheap).
        if self.ring.len() > len.saturating_mul(4).max(len + 16_000) {
            let cut = self.ring.len() - len;
            self.ring.drain(..cut);
        }
        if self.ring.len() < len {
            // Only after a start-over shorter than the context (cannot happen: `new == len`).
            let missing = len - self.ring.len();
            self.ring.splice(0..0, std::iter::repeat_n(0.0, missing));
        }
        self.ring.len() - len..self.ring.len()
    }
}

impl VoiceConversionModel for DenoisedModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let window = self.follow(ctx);
        let denoised = ConversionCtx {
            mono_16k: &self.ring[window],
            new_frames: ctx.new_frames,
            pitch_shift_semitones: ctx.pitch_shift_semitones,
            index_rate: ctx.index_rate,
            protect: ctx.protect,
            speaker_id: ctx.speaker_id,
            stream_pos_16k: ctx.stream_pos_16k,
            reference: ctx.reference,
            singing: ctx.singing,
        };
        self.inner.convert(&denoised)
    }

    fn input_sample_rate(&self) -> u32 {
        self.inner.input_sample_rate()
    }

    fn output_sample_rate(&self) -> u32 {
        self.inner.output_sample_rate()
    }

    fn min_context_samples(&self) -> usize {
        self.inner.min_context_samples()
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        // Warm the denoiser's session on a copy (aligned like a stream's first call), then start clean.
        self.scratch.clear();
        self.scratch.resize(Self::alignment_pad(ctx.mono_16k.len(), self.den.granularity()), 0.0);
        self.scratch.extend_from_slice(ctx.mono_16k);
        let _ = self.den.process(&mut self.scratch);
        self.den.reset();
        self.end = None;
        self.ring.clear();
        self.inner.warmup_at(ctx)
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        self.inner.set_voice_controls(controls);
    }

    fn apply_quality(&mut self, level: &QualityLevel) -> bool {
        self.inner.apply_quality(level)
    }

    fn reset_stream(&mut self) {
        self.den.reset();
        self.inner.reset_stream();
    }

    fn fixed_block(&self) -> Option<usize> {
        self.inner.fixed_block()
    }

    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        self.inner.configure_stream(block, want, extra)
    }

    fn stream_report(&self) -> Option<StreamReport> {
        self.inner.stream_report()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::denoise::stft::tests::{gtcrn_cfg, Identity};
    use crate::services::neural_audio::denoise::stft::StftStream;
    use crate::services::neural_audio::vc::stream::VcStreamHarness;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, Mutex};

    /// A "denoiser" that is a pure delay of `d` samples and records what it was fed.
    struct Delay {
        d: usize,
        line: std::collections::VecDeque<f32>,
        fed: Arc<Mutex<Vec<f32>>>,
        resets: Arc<Mutex<u32>>,
    }

    type Probes = (Arc<Mutex<Vec<f32>>>, Arc<Mutex<u32>>);

    impl Delay {
        fn new(d: usize) -> (Delay, Probes) {
            let fed = Arc::new(Mutex::new(Vec::new()));
            let resets = Arc::new(Mutex::new(0));
            (
                Delay {
                    d,
                    line: std::iter::repeat_n(0.0, d).collect(),
                    fed: fed.clone(),
                    resets: resets.clone(),
                },
                (fed, resets),
            )
        }
    }

    impl Denoiser for Delay {
        fn sample_rate(&self) -> u32 {
            16_000
        }
        fn latency_samples(&self) -> usize {
            self.d
        }
        fn process(&mut self, audio: &mut [f32]) -> Result<(), String> {
            self.fed.lock().unwrap().extend_from_slice(audio);
            for s in audio.iter_mut() {
                self.line.push_back(*s);
                *s = self.line.pop_front().unwrap();
            }
            Ok(())
        }
        fn reset(&mut self) {
            *self.resets.lock().unwrap() += 1;
            self.line = std::iter::repeat_n(0.0, self.d).collect();
        }
    }

    /// An inner model that returns the newest `new_frames` of its context (16 kHz in and out) and
    /// records the contexts it saw.
    struct Tail {
        seen: Arc<Mutex<Vec<Vec<f32>>>>,
    }

    impl VoiceConversionModel for Tail {
        fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
            self.seen.lock().unwrap().push(ctx.mono_16k.to_vec());
            Ok(ctx.mono_16k[ctx.mono_16k.len() - ctx.new_frames..].to_vec())
        }
        fn input_sample_rate(&self) -> u32 {
            16_000
        }
        fn output_sample_rate(&self) -> u32 {
            16_000
        }
    }

    fn ctx(ring: &[f32], pos: u64) -> ConversionCtx<'_> {
        ConversionCtx {
            mono_16k: ring,
            new_frames: 10,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: pos,
            reference: None,
            singing: Default::default(),
        }
    }

    /// Only the new samples of each call go through the denoiser, in stream order, and the inner
    /// model sees the denoised stream at the same window (here: the input delayed by `d`).
    #[test]
    fn only_new_samples_are_denoised_and_the_window_is_kept() {
        let (den, (fed, _)) = Delay::new(5);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut m = DenoisedModel::new(Box::new(Tail { seen: seen.clone() }), Box::new(den)).unwrap();
        let stream: Vec<f32> = (0..200).map(|i| i as f32 + 1.0).collect();
        // A 40-sample context advancing by 16 samples per call (the harness's rolling ring).
        let mut pos = 0u64;
        while pos as usize + 40 <= stream.len() {
            let window = &stream[pos as usize..pos as usize + 40];
            m.convert(&ctx(window, pos)).unwrap();
            pos += 16;
        }
        // The denoiser saw every sample of the stream exactly once, in order.
        let fed = fed.lock().unwrap().clone();
        assert_eq!(fed, stream[..fed.len()].to_vec());
        // Each context the model saw is the matching window of the input delayed by 5.
        for (k, seen) in seen.lock().unwrap().iter().enumerate() {
            let start = k * 16;
            assert_eq!(seen.len(), 40);
            for (j, &v) in seen.iter().enumerate() {
                let raw = start + j;
                let want = if raw >= 5 { stream[raw - 5] } else { 0.0 };
                assert_eq!(v, want, "call {k} sample {j}");
            }
        }
    }

    /// A jump of the stream position starts over with fresh denoiser state; a stream reset clears
    /// the denoiser's state and keeps the timeline.
    #[test]
    fn jumps_resync_and_resets_clear_state() {
        let (den, (fed, resets)) = Delay::new(3);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut m = DenoisedModel::new(Box::new(Tail { seen }), Box::new(den)).unwrap();
        let a = vec![1.0f32; 20];
        m.convert(&ctx(&a, 0)).unwrap();
        m.convert(&ctx(&a, 10)).unwrap();
        assert_eq!(fed.lock().unwrap().len(), 30);
        m.convert(&ctx(&a, 1_000)).unwrap();
        assert_eq!(fed.lock().unwrap().len(), 50, "a jump re-reads the whole context");
        let before = *resets.lock().unwrap();
        m.reset_stream();
        assert_eq!(*resets.lock().unwrap(), before + 1);
        m.convert(&ctx(&a, 1_004)).unwrap();
        assert_eq!(fed.lock().unwrap().len(), 54, "after a reset only the new samples");
    }

    /// Inside the real harness the lengths are exact and the output is the input delayed by the
    /// denoiser's latency (plus the harness's own).
    #[test]
    fn in_the_harness_lengths_stay_exact() {
        let (den, _) = Delay::new(7);
        let seen = Arc::new(Mutex::new(Vec::new()));
        let model = DenoisedModel::new(Box::new(Tail { seen }), Box::new(den)).unwrap();
        let mut h = VcStreamHarness::new(Box::new(model), 48_000, 48_000, 1, 1, 1_600, 320, 480).unwrap();
        let mut total = 0usize;
        for _ in 0..200 {
            let out = h.push_and_convert(&[0.1f32; 480]).unwrap();
            total += out.len();
            assert_eq!(out.len(), h.last_owed_frames());
        }
        assert_eq!(total, 200 * 480);
    }

    /// A real STFT denoiser (the identity model on GTCRN's framing) built with GTCRN's in-harness
    /// chunk promise (640-sample model blocks on a 256 hop: 128-sample alignment), counting its
    /// buffer underflows.
    struct Stft {
        stream: StftStream<Identity>,
        underflows: Arc<AtomicU64>,
    }

    impl Denoiser for Stft {
        fn sample_rate(&self) -> u32 {
            16_000
        }
        fn latency_samples(&self) -> usize {
            self.stream.latency_samples()
        }
        fn process(&mut self, audio: &mut [f32]) -> Result<(), String> {
            self.stream.process(audio);
            self.underflows.store(self.stream.underflows(), Ordering::SeqCst);
            Ok(())
        }
        fn reset(&mut self) {
            self.stream.reset();
        }
        fn granularity(&self) -> usize {
            self.stream.granularity()
        }
    }

    /// A template voice's context (160 x t samples) is not a multiple of the 128-sample alignment
    /// unless t % 4 == 0. The stream's first call (the whole context) is padded, so the STFT never
    /// runs dry (it used to zero 32 samples of the first context here and run 2 ms later than it
    /// reported from then on), and the model sees the input delayed by exactly the reported latency.
    #[test]
    fn a_start_over_keeps_the_chunk_promise() {
        let (ctx_len, block) = (160 * 145, 640usize);
        assert_ne!(ctx_len % 128, 0);
        let underflows = Arc::new(AtomicU64::new(0));
        let den = Stft {
            stream: StftStream::new(Identity, gtcrn_cfg(), Some(block)).unwrap(),
            underflows: underflows.clone(),
        };
        let d = den.latency_samples();
        assert_eq!((d, den.granularity()), (256 + 128, 128));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let mut m = DenoisedModel::new(Box::new(Tail { seen: seen.clone() }), Box::new(den)).unwrap();
        let calls = 40;
        let stream: Vec<f32> = (0..ctx_len + block * calls).map(|i| (i as f32 * 0.0123).sin() * 0.5).collect();
        let warm = ConversionCtx {
            mono_16k: &stream[..ctx_len],
            new_frames: block,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
            reference: None,
            singing: Default::default(),
        };
        m.warmup_at(&warm).unwrap();
        for k in 0..calls {
            let end = ctx_len + k * block;
            let ctx = ConversionCtx {
                mono_16k: &stream[end - ctx_len..end],
                stream_pos_16k: (end - ctx_len) as u64,
                ..warm
            };
            m.convert(&ctx).unwrap();
        }
        assert_eq!(underflows.load(Ordering::SeqCst), 0, "the chunk promise held");
        let seen = seen.lock().unwrap();
        for (k, window) in seen.iter().enumerate() {
            let start = k * block;
            for (j, &v) in window.iter().enumerate() {
                let raw = start + j;
                let want = if raw >= d { stream[raw - d] } else { 0.0 };
                assert!((v - want).abs() < 1e-4, "call {k} sample {j}: {v} vs {want}");
            }
        }
    }

    #[test]
    fn rates_must_match() {
        struct At48k;
        impl Denoiser for At48k {
            fn sample_rate(&self) -> u32 {
                48_000
            }
            fn latency_samples(&self) -> usize {
                0
            }
            fn process(&mut self, _: &mut [f32]) -> Result<(), String> {
                Ok(())
            }
            fn reset(&mut self) {}
        }
        let seen = Arc::new(Mutex::new(Vec::new()));
        assert!(DenoisedModel::new(Box::new(Tail { seen }), Box::new(At48k)).is_err());
    }
}
