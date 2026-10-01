//! Voice-conversion traits for RVC (and future engines).
//! Operates on a rolling 16 kHz context; F0 is on the 10 ms (hop-160) grid.
//!
//! P3-S0 added (frozen; change only through INTEGRATE): [`ConversionCtx::stream_pos_16k`],
//! [`VoiceControls`] with [`VoiceConversionModel::set_voice_controls`], the governor hook
//! [`VoiceConversionModel::apply_quality`] and [`VoiceConversionModel::reset_stream`]. The new
//! methods have no-op defaults, so the legacy models are unchanged.
//!
//! P3-C1 added (additive, defaults keep every existing model unchanged):
//! - fixed-block streaming: [`VoiceConversionModel::fixed_block`] and
//!   [`VoiceConversionModel::configure_stream`]. A model that returns `Some(block)` gets exactly
//!   `block` new input samples per `convert`, at one fixed context length, from
//!   `VcStreamHarness` (its pre-accumulator absorbs the input resampler's floor / ceil jitter), so
//!   every session runs one shape (the Sigil Voice Engine; CUDA graphs and IOBinding need it);
//! - [`PitchExtractor::receptive_frames`]: how far an extractor looks around a frame, which sizes
//!   the incremental F0 cache (`engine::f0_cache`);
//! - [`VoiceConversionModel::stream_report`] / [`StreamReport`]: the outcomes and levers after the
//!   harness's `configure_stream` re-planned the sessions.

use crate::services::diagnostics::EpOutcome;

use super::engine::governor::QualityLevel;
use super::engine::options::EffectiveLevers;

/// RVC's flow prior noise scale (`noise_scale` default).
pub const DEFAULT_NOISE_SCALE: f32 = 0.66666;
/// Silence-gate threshold (RMS dBFS) default; the legacy RMS skip is 0.008 (-41.9 dBFS).
pub const DEFAULT_SILENCE_DB: f32 = -42.0;

/// F0 per 10 ms frame (16 kHz / hop-160 grid), Hz, 0.0 = unvoiced.
///
/// Frame `j` describes the audio around sample `160 * j` of `mono_16k`. Extractors may return a
/// frame more or fewer than `mono_16k.len() / 160`; the caller aligns the contour to its grid.
pub trait PitchExtractor: Send {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String>;

    /// `(left, right)`: how many 10 ms frames before and after a frame can change its F0 (analysis
    /// window plus smoothing). The incremental F0 cache feeds at least `left` frames of lead-in
    /// before the frames it keeps and re-extracts the newest `right` frames of the previous tick,
    /// so a cached contour equals a whole-context extraction wherever the extractor's view is
    /// local. `None` (default): unknown, the cache uses its conservative defaults (neural
    /// extractors with recurrent layers see the whole input; their error decays with distance).
    fn receptive_frames(&self) -> Option<(usize, usize)> {
        None
    }
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
    /// Absolute 16 kHz index of `mono_16k[0]` in the current stream (the harness's priming
    /// silence counts), so engines can anchor noise, F0 caches and NSF phase to time. `0` for
    /// one-off conversions and warm-ups.
    pub stream_pos_16k: u64,
}

/// Per-voice controls applied while running (the hot SVE params). The defaults are neutral: a
/// model without a post stage ignores them and sounds exactly as before.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoiceControls {
    /// Loudness envelope mix (RVC `rms_mix_rate`): 1 = the model's own envelope (off), 0 = the
    /// input's envelope.
    pub rms_mix: f32,
    /// True formant shift in semitones (cepstral envelope, RVC path), 0 = off.
    pub formant_semitones: f32,
    /// Flow prior noise scale of template voices.
    pub noise_scale: f32,
    /// RMS level (dBFS) below which a block counts as silent, when the silence gate is on.
    pub silence_db: f32,
}

impl Default for VoiceControls {
    fn default() -> Self {
        Self {
            rms_mix: 1.0,
            formant_semitones: 0.0,
            noise_scale: DEFAULT_NOISE_SCALE,
            silence_db: DEFAULT_SILENCE_DB,
        }
    }
}

/// A full voice-conversion backend. Owns sessions/state.
pub trait VoiceConversionModel: Send {
    /// Returns exactly `round(new_frames * output_sample_rate / input_sample_rate)` samples
    /// (fewer only if the model itself produced less audio), time-aligned so that the last
    /// returned sample corresponds to the last sample of `mono_16k` (a model with lookahead, such
    /// as the Sigil Voice Engine with `set_lookahead_ms`, ends that much earlier on every call: a
    /// constant delay, so the harness's length accounting is unchanged).
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
    /// Hot per-voice controls changed (called between ticks, never from an audio callback).
    /// Default: ignored.
    fn set_voice_controls(&mut self, _controls: &VoiceControls) {}
    /// The adaptive governor's quality level (P3-E). Returns whether the model applied it; the
    /// governor treats `false` as "this step does nothing here". Default: not supported.
    fn apply_quality(&mut self, _level: &QualityLevel) -> bool {
        false
    }
    /// The stream had a gap or restarted (device restart, capture gap, latency-budget drop): drop
    /// state keyed to the old timeline (F0 cache, NSF phase). Default: nothing to reset.
    fn reset_stream(&mut self) {}
    /// Fixed-block streaming (P3-C1): `Some(block)` asks the stream harness to call `convert` with
    /// exactly `block` new input samples each time (whatever the device block is; a
    /// pre-accumulator absorbs the resampler's per-call jitter) and a fixed context length (see
    /// [`configure_stream`](Self::configure_stream)), so the model runs one tensor shape for the
    /// whole stream and `ConversionCtx::stream_pos_16k` advances by exactly `block` per call.
    /// Default: `None`, the legacy feed (one call per device block, floor / ceil lengths).
    fn fixed_block(&self) -> Option<usize> {
        None
    }
    /// Fixed-block streaming: called once by the harness (before its warm-up) with the block, the
    /// samples every call returns (`want` = block + SOLA tail, the `ConversionCtx::new_frames` of
    /// every live call) and the extra context the user asked for. Returns the context length
    /// (`ConversionCtx::mono_16k.len()` of every live call; at least `want`). The model may rebuild
    /// its fixed-shape sessions here. Only called when [`fixed_block`](Self::fixed_block) is
    /// `Some`. Default: `extra + block` (at least `want` and the minimum context).
    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        Ok(extra.saturating_add(block).max(want).max(self.min_context_samples()))
    }
    /// What the model runs now (P3-C1 fixer): the EP outcome per session component, the levers in
    /// effect and the load warnings. [`configure_stream`](Self::configure_stream) may rebuild
    /// sessions for the stream's shape (a fallback EP, a CUDA graph that no longer captures), so a
    /// caller that copied them at load refreshes them from here after building the harness
    /// (`VcStreamHarness::model_report`): components replace the ones of the same name, lever notes
    /// and warnings add up. Wrappers forward it. Default: `None`, nothing changes after load.
    fn stream_report(&self) -> Option<StreamReport> {
        None
    }
}

/// See [`VoiceConversionModel::stream_report`].
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamReport {
    pub outcomes: Vec<(String, EpOutcome)>,
    pub levers: EffectiveLevers,
    pub warnings: Vec<String>,
}
