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
//!
//! P5-S0 added (frozen; change only through INTEGRATE; additive, neutral defaults):
//! - [`ConversionCtx::reference`]: the zero-shot reference embedding of the voice
//!   (`families::reference::ReferenceEmbedding`), installed in the harness by the engine factory
//!   (`VcStreamHarness::with_context` / `set_reference`), so warm-ups and every tick see it;
//!   `None` for every family without a reference.
//! - [`ConversionCtx::singing`] / [`SingingFlags`]: the voice's singing flags (`f0_range =
//!   singing`), from `VcStreamHarness::with_context` / `set_singing`; speech by default.
//! - A wrapper that rebuilds a context (the denoiser harness, the engine's own warm-up) forwards
//!   both fields unchanged; one-off contexts (warm-ups, tests) use `reference: None` and
//!   `singing: Default::default()`.
//!
//! P3B-S0 added (frozen; change only through INTEGRATE; additive, defaults keep every model and
//! extractor unchanged):
//! - [`PitchExtractor::preferred_input_frames`], [`PitchExtractor::max_call_frames`],
//!   [`PitchExtractor::last_salience`] and [`PitchExtractor::f0_into`] (P3B-E implements them for
//!   RMVPE, and where cheap for FCPE / CREPE / SwiftF0; P3B-F's F0 cache and cross-tick decoder
//!   read them). `f0_into` defaults to [`PitchExtractor::f0`] copied into the caller's buffer.
//! - [`VoiceConversionModel::time_anchored`] (the template engine: anchored noise and the NSF
//!   phase both in effect; P3B-H narrows the SOLA search only for such models) and
//!   [`VoiceConversionModel::telemetry`] → [`ModelTelemetry`] (the fed F0 per absolute frame and the
//!   last per-call [`StageRecord`]; P3B-B's scorer reads it, P3B-A reports it). Wrappers forward
//!   both, like `stream_report`.

use crate::services::diagnostics::EpOutcome;

use super::engine::governor::QualityLevel;
use super::engine::options::{EffectiveLevers, F0Range};
pub use super::families::reference::ReferenceEmbedding;

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

    /// P3B-S0: the audio frames (10 ms) one call should carry to fill the extractor's fixed shape
    /// exactly (RMVPE's 32-frame plan: 31 audio frames), so the F0 cache can lead each incremental
    /// call with as much real audio as the shape costs anyway. `None` (default): no preference.
    #[allow(dead_code)] // P3B-S0 frozen API; its consumer lands with the owning workstream.
    fn preferred_input_frames(&self) -> Option<usize> {
        None
    }

    /// P3B-S0: the most audio frames one call may carry without the extractor growing or
    /// rebuilding its fixed shape (longer inputs are split by the extractor or the cache). `None`
    /// (default): unbounded.
    #[allow(dead_code)] // P3B-S0 frozen API; its consumer lands with the owning workstream.
    fn max_call_frames(&self) -> Option<usize> {
        None
    }

    /// P3B-S0: the per-frame salience of the last call, `(values, bins)` with `values.len() =
    /// frames × bins` row-major (RMVPE / CREPE: 360 bins on the 20-cent grid), for cross-tick
    /// decoding (P3B-F). `None` (default): the extractor has no salience to share.
    #[allow(dead_code)] // P3B-S0 frozen API; its consumer lands with the owning workstream.
    fn last_salience(&self) -> Option<(&[f32], usize)> {
        None
    }

    /// P3B-S0: [`f0`](Self::f0) into a caller-owned buffer (cleared, then filled), so an extractor
    /// that implements it runs allocation-free on the tick path. Default: `f0`, copied.
    #[allow(dead_code)] // P3B-S0 frozen API; its consumer lands with the owning workstream.
    fn f0_into(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        let f0 = self.f0(mono_16k)?;
        out.clear();
        out.extend_from_slice(&f0);
        Ok(())
    }
}

/// Compute of one model call by stage (ms), for the benchmark's attribution (P3B-S0, frozen; may
/// gain fields). A stage the model does not have stays 0.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StageRecord {
    pub content_ms: f64,
    pub retrieval_ms: f64,
    pub f0_ms: f64,
    pub enc_ms: f64,
    pub flowdec_ms: f64,
    pub post_ms: f64,
    /// The whole call.
    pub total_ms: f64,
    /// F0 frames the extractor analysed.
    pub f0_frames: usize,
    /// The call was skipped as silence (no inference).
    pub skipped: bool,
    /// Absolute 16 kHz position of the call's context (`ConversionCtx::stream_pos_16k`).
    pub stream_pos_16k: u64,
}

/// The F0 a model fed its generator, on the absolute 10 ms grid (P3B-S0, frozen).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FedF0<'a> {
    /// Absolute frame of `hz[0]` (`stream_pos_16k / 160` units).
    pub start_frame: u64,
    /// Hz per frame (after the pitch shift; 0 = unvoiced or not converted).
    pub hz: &'a [f32],
}

/// What a model can tell a benchmark about its last calls (P3B-S0, frozen; P3B-B fills the
/// template engine's through `engine::probe`). Every hook is off unless a benchmark enabled it, so
/// a shipping run records nothing.
#[allow(dead_code)] // P3B-S0 frozen API; P3B-B reads it.
pub trait ModelTelemetry {
    /// The F0 fed to the generator for every returned frame since the tap was enabled (`None`
    /// when the tap is off or nothing was recorded). A frame returned by two calls (the SOLA tail)
    /// holds the later call's value.
    fn fed_f0(&self) -> Option<FedF0<'_>>;
    /// The stage record of the last call (`None` before the first call or when not recorded).
    fn last_stages(&self) -> Option<StageRecord>;
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
    /// The zero-shot reference this tick converts toward (P5-S0): the embedding the voice's family
    /// computed for its reference voice (`FamilyEngine::reference`), installed by the engine
    /// factory in the voice's harness, so the harness's own warm-up and every tick carry it.
    /// `None` for every family without a reference and for one-off conversions. A family keys its
    /// prepared state on [`ReferenceEmbedding::same_source`].
    pub reference: Option<&'a ReferenceEmbedding>,
    /// The voice's singing flags (P5-S0). Neutral (speech) unless the harness was given others.
    pub singing: SingingFlags,
}

/// Singing flags of a voice (P5-S0, frozen; may gain fields, the default stays neutral).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SingingFlags {
    /// Singing mode: the voice runs with `f0_range = singing` (its pitch extractor covers E2–D#7
    /// with no coarse clamp). A family with speech-only F0 smoothing (octave gate, median) keeps
    /// vibrato and slides when it is set.
    pub singing: bool,
}

impl SingingFlags {
    /// Speech (the neutral flags).
    pub const SPEECH: SingingFlags = SingingFlags { singing: false };

    /// The flags of a voice with this F0 range.
    pub fn from_range(range: F0Range) -> SingingFlags {
        SingingFlags { singing: range == F0Range::Singing }
    }
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
    /// P3B-S0: whether consecutive calls decode identical audio on the frames they share
    /// (time-anchored noise and a continuous NSF phase both in effect), so SOLA joins sit at their
    /// nominal point. Wrappers forward it. Default: `false`.
    #[allow(dead_code)] // P3B-S0 frozen API; its consumer lands with the owning workstream.
    fn time_anchored(&self) -> bool {
        false
    }
    /// P3B-S0: the model's telemetry hooks ([`ModelTelemetry`]; benchmarks only). Wrappers forward
    /// it. Default: `None`.
    #[allow(dead_code)] // P3B-S0 frozen API; its consumer lands with the owning workstream.
    fn telemetry(&self) -> Option<&dyn ModelTelemetry> {
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
