//! Voice-conversion traits for RVC (and future engines).
//! Operates on a rolling 16 kHz context; F0 is on the 10 ms (hop-160) grid.

/// F0 per 10 ms frame (16 kHz / hop-160 grid), Hz, 0.0 = unvoiced.
///
/// Frame `j` describes the audio around sample `160 * j` of `mono_16k`. Extractors may return a
/// frame more or fewer than `mono_16k.len() / 160`; the caller aligns the contour to its grid.
pub trait PitchExtractor: Send {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String>;
}

/// Context for one conversion tick (extra context + new block @ 16 kHz mono).
pub struct ConversionCtx<'a> {
    /// Rolling context; the newest sample is last.
    pub mono_16k: &'a [f32],
    /// Trailing samples of `mono_16k` whose conversion is returned (new block + SOLA tail).
    pub new_frames: usize,
    pub pitch_shift_semitones: f32,
    pub index_rate: f32,
    pub protect: f32,
    pub speaker_id: i64,
}

/// A full voice-conversion backend. Owns sessions/state.
pub trait VoiceConversionModel: Send {
    /// Returns exactly `round(new_frames * output_sample_rate / input_sample_rate)` samples
    /// (fewer only if the model itself produced less audio), time-aligned so that the last
    /// returned sample corresponds to the last sample of `mono_16k`.
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String>;
    fn input_sample_rate(&self) -> u32;
    fn output_sample_rate(&self) -> u32;
    /// Fewest context samples (at `input_sample_rate`) `convert` needs to produce real output.
    /// The stream harness emits silence instead of calling the model until the ring holds this
    /// much audio.
    fn min_context_samples(&self) -> usize {
        0
    }
    /// Run one conversion at the input shape live ticks will use (bypassing any silence skip) so
    /// GPU execution providers pick their kernels before the first spoken tick. The result is
    /// discarded. The stream harness calls this from `VcStreamHarness::new`. Default: nothing to
    /// warm.
    fn warmup_at(&mut self, _ctx: &ConversionCtx<'_>) -> Result<(), String> {
        Ok(())
    }
}
