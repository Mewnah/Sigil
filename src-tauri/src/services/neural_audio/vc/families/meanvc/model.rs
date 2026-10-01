//! [`MeanVcModel`]: the MeanVC2 loop behind the engine's `VoiceConversionModel` contract.
//!
//! MeanVC2 is a stateful streaming model (ASR and DiT caches, a vocoder overlap), not a function of
//! a context window like RVC. The model therefore:
//! - runs **fixed blocks** (`fixed_block`, a multiple of 40 ms: half the tick for the 80 ms tick,
//!   the whole tick for 160 ms; `meanvc.rs` decides): the harness hands it exactly one block of new
//!   audio per call, and the loop consumes it once, running at most one DiT call per 40 ms of block
//!   (`MeanVcStream::set_dit_limit`), so a 40 ms block carries half an 80 ms tick's work and every
//!   call returns exactly one block;
//! - keeps an **output history**: every call appends exactly one block of the loop's output, so
//!   the history's end stays aligned with the input's end, and each call returns the history's
//!   newest `new_frames` samples. Consecutive returns overlap sample for sample, so the harness's
//!   SOLA join finds them identical (no seams); the content lags by the model's own delay
//!   (`meanvc::MODEL_DELAY_16K`), reported to the factory;
//! - **starts from steady silence**: at load the loop runs [`SILENCE_PRIME_16K`] of silence from
//!   cold, and the state it reaches is kept ([`MeanVcModel::silence_state`]). The stream starts
//!   there (as if silence had been flowing, like the harness's zero-filled ring), and every
//!   discontinuity (a stream reset, a skipped tick of the harness, a model error) returns there
//!   with a 5 ms fade out of the history and a 5 ms fade in of the next output;
//! - **skips silence** (`silence_skip`): once the input has been quiet for [`SKIP_AFTER_QUIET_16K`]
//!   (every 10 ms frame below `VoiceControls::silence_db`), nothing runs and silence is appended
//!   (the output had settled already: the quiet stretch is longer than the model's delay). The
//!   first loud block restores the steady silence, replays the previous (quiet) block to warm the
//!   caches, then converts, so a soft onset that started under the threshold is not lost;
//! - follows a **hot reference change**: a `ConversionCtx::reference` from this family that is not
//!   the one in use re-runs the GTM (well under a millisecond) and converts toward it from the next
//!   DiT call on;
//! - answers the **governor**: from the level that halves the context (level 2) on, a voice set to
//!   2 mean-flow steps runs 1 (about 30 % less compute; ADR 0006);
//! - runs the engine's **post stage** (`engine::post`, as template voices do; the factory adds none
//!   to family voices): `rms_mix` (`VoiceControls::rms_mix`, the input's loudness envelope; MeanVC2
//!   otherwise speaks at its own level, a quiet microphone included), the cepstral formant shift
//!   (`formant_semitones`) and the output limiter. Its input reference is the input delayed by the
//!   model's content delay (`ModelConfig::align_16k`), so the envelopes line up with the output.
//!   At the default controls it only limits (a pass-through below the ceiling). Its silence gate is
//!   off: the silence skip already ends the output after a quiet stretch, and short pauses keep the
//!   model's own rendering.

use std::sync::Arc;

use super::speaker;
use super::stream::{MeanVcStream, StreamState, DIT_HOP_16K};
use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::engine::post::{PostConfig, PostStage};
use crate::services::neural_audio::vc::traits::{ConversionCtx, ReferenceEmbedding, VoiceControls, VoiceConversionModel, DEFAULT_SILENCE_DB};

/// MeanVC2's input and output rate.
pub const SAMPLE_RATE: u32 = 16_000;
/// The loop's natural step: one ASR call (8 fbank frames).
pub const TICK_16K: usize = 1280;
/// Silence the loop runs from cold at load to reach its steady silence state (a multiple of every
/// block: 40, 80 and 160 ms).
pub const SILENCE_PRIME_16K: usize = 8 * TICK_16K;
/// Quiet input before the silence skip stops running the model (longer than the model's delay
/// plus the DiT's 120 ms attention span, so the output has settled to silence).
pub const SKIP_AFTER_QUIET_16K: usize = 6400;
/// Fade (samples) at every discontinuity of the output stream.
pub const FADE_16K: usize = 80;
/// RMS frame of the silence detector (10 ms).
const QUIET_FRAME: usize = 160;

/// How a voice runs.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ModelConfig {
    /// New samples per call (a multiple of [`TICK_16K`]).
    pub block_16k: usize,
    /// Mean-flow steps (1 or 2).
    pub steps: u32,
    pub silence_skip: bool,
    /// How far the output's content lags the input (16 kHz samples): the post stage reads the
    /// input this much earlier as the reference of the output it processes.
    pub align_16k: usize,
}

/// See the module docs.
pub struct MeanVcModel {
    stream: MeanVcStream,
    block: usize,
    silence: StreamState,
    /// The newest output samples (`max(want, block)`), oldest first.
    hist: Vec<f32>,
    /// The newest input samples (`hist.len() + align`), oldest first: `input[..hist.len()]` lines up
    /// with `hist` (the post stage's reference).
    input: Vec<f32>,
    align: usize,
    post: PostStage,
    controls: VoiceControls,
    configured_steps: u32,
    skip_enabled: bool,
    threshold_db: f32,
    quiet_16k: usize,
    skipping: bool,
    /// The newest input block (replayed when the silence skip resumes).
    last_block: Vec<f32>,
    fade_in: bool,
    produced: Vec<f32>,
    discard: Vec<f32>,
    /// `stream_pos_16k` the next call should carry (a jump is a discontinuity).
    next_pos: Option<u64>,
    /// The reference the speaker came from.
    reference: Option<Arc<ReferenceEmbedding>>,
    /// The last reference that could not be used (`voice_id`, `backend`, `clip_sha256`).
    rejected: Option<(String, String, String)>,
    length_warned: bool,
    pub(super) blocks_converted: u64,
    pub(super) blocks_skipped: u64,
}

/// Whether any 10 ms frame of `x` reaches `threshold_db` (RMS dBFS).
pub fn block_is_loud(x: &[f32], threshold_db: f32) -> bool {
    if threshold_db.is_nan() || threshold_db <= -120.0 {
        return true;
    }
    let thr = 10f32.powf(threshold_db / 20.0);
    let thr2 = thr * thr;
    x.chunks(QUIET_FRAME)
        .any(|f| f.iter().map(|v| v * v).sum::<f32>() / f.len() as f32 >= thr2)
}

fn fade(x: &mut [f32], rising: bool) {
    let n = x.len();
    for (i, v) in x.iter_mut().enumerate() {
        let g = (i + 1) as f32 / (n + 1) as f32;
        *v *= if rising { g } else { 1.0 - g };
    }
}

/// Append `block` to `buf` and drop as many samples from its front (`buf` keeps its length).
fn shift_in(buf: &mut [f32], block: &[f32]) {
    let (h, b) = (buf.len(), block.len());
    if b >= h {
        buf.copy_from_slice(&block[b - h..]);
    } else {
        buf.copy_within(b.., 0);
        buf[h - b..].copy_from_slice(block);
    }
}

/// The post stage of a MeanVC2 voice (module docs): limiter, no gate.
fn post_stage() -> PostStage {
    PostStage::new(PostConfig {
        sample_rate: SAMPLE_RATE,
        limiter: true,
        silence_gate: false,
        silence_hangover_ms: 0.0,
    })
}

impl MeanVcModel {
    /// Wrap a stream (built for this voice's speaker); runs the silence prime (see the module docs).
    pub fn new(mut stream: MeanVcStream, config: &ModelConfig) -> Result<MeanVcModel, String> {
        let block = config.block_16k;
        if block == 0 || !block.is_multiple_of(DIT_HOP_16K) || !SILENCE_PRIME_16K.is_multiple_of(block) {
            return Err(format!(
                "MeanVC2 blocks are multiples of {DIT_HOP_16K} samples that divide {SILENCE_PRIME_16K}, not {block}"
            ));
        }
        // One DiT call per 40 ms of block: the work of a tick spreads over its blocks.
        stream.set_dit_limit(Some(block / DIT_HOP_16K));
        stream.reset_cold();
        let zeros = vec![0.0f32; block];
        let mut sink = Vec::with_capacity(2 * block);
        let mut last = 0;
        for _ in 0..SILENCE_PRIME_16K / block {
            sink.clear();
            last = stream.process(&zeros, &mut sink)?;
        }
        if last != block {
            return Err(format!(
                "MeanVC2 did not reach its steady state on silence ({last} samples per {block}); the graphs do not match the export"
            ));
        }
        let silence = stream.state().clone();
        let configured_steps = config.steps.clamp(1, 2);
        stream.set_steps(configured_steps);
        Ok(MeanVcModel {
            stream,
            block,
            silence,
            hist: vec![0.0; block],
            input: vec![0.0; block + config.align_16k],
            align: config.align_16k,
            post: post_stage(),
            controls: VoiceControls::default(),
            configured_steps,
            skip_enabled: config.silence_skip,
            threshold_db: DEFAULT_SILENCE_DB,
            quiet_16k: 0,
            skipping: false,
            last_block: Vec::with_capacity(block),
            fade_in: false,
            produced: Vec::with_capacity(2 * block),
            discard: Vec::with_capacity(2 * block),
            next_pos: None,
            reference: None,
            rejected: None,
            length_warned: false,
            blocks_converted: 0,
            blocks_skipped: 0,
        })
    }

    /// The reference the voice converts toward (what `ConversionCtx::reference` is compared with).
    pub fn set_reference(&mut self, reference: Option<Arc<ReferenceEmbedding>>) {
        self.reference = reference;
    }

    /// The steady silence state the stream starts from.
    #[cfg(test)]
    pub fn silence_state(&self) -> &StreamState {
        &self.silence
    }

    #[cfg(test)]
    pub fn stream(&self) -> &MeanVcStream {
        &self.stream
    }

    #[cfg(test)]
    pub fn steps(&self) -> u32 {
        self.stream.steps()
    }

    /// Back to the steady silence with fades on both sides of the cut.
    fn discontinuity(&mut self) {
        self.stream.restore(&self.silence);
        self.skipping = false;
        self.quiet_16k = 0;
        let h = self.hist.len();
        let n = FADE_16K.min(h);
        fade(&mut self.hist[h - n..], false);
        self.fade_in = true;
    }

    fn follow_reference(&mut self, reference: Option<&ReferenceEmbedding>) {
        let Some(r) = reference else { return };
        if self
            .reference
            .as_deref()
            .is_some_and(|cur| cur.same_source(r))
        {
            return;
        }
        if self
            .rejected
            .as_ref()
            .is_some_and(|(v, b, s)| *v == r.voice_id && *b == r.backend && *s == r.clip_sha256)
        {
            return;
        }
        match speaker::spks_of(r).and_then(|spks| self.stream.set_speaker(&spks)) {
            Ok(()) => {
                self.reference = Some(Arc::new(r.clone()));
                self.rejected = None;
            }
            Err(e) => {
                tracing::warn!("voice changer: MeanVC2 keeps its speaker; the new reference cannot be used: {e}");
                self.rejected = Some((r.voice_id.clone(), r.backend.clone(), r.clip_sha256.clone()));
            }
        }
    }

    /// Run the loop over `new` (after replaying the previous block when resuming from a skip).
    fn run_block(&mut self, new: &[f32], resume: bool) -> Result<(), String> {
        if resume {
            self.stream.restore(&self.silence);
            if self.last_block.len() == self.block {
                self.discard.clear();
                let previous = std::mem::take(&mut self.last_block);
                let replay = self.stream.process(&previous, &mut self.discard);
                self.last_block = previous;
                replay?;
            }
        }
        self.stream.process(new, &mut self.produced).map(|_| ())
    }

    /// One call: the block through the loop (or skipped), exactly one block onto the history.
    fn step(&mut self, new: &[f32]) -> Result<(), String> {
        let block = self.block;
        let loud = !self.skip_enabled || block_is_loud(new, self.threshold_db);
        self.quiet_16k = if loud { 0 } else { self.quiet_16k.saturating_add(block) };
        self.produced.clear();
        let mut result = Ok(());
        let mut converted = false;
        if self.skipping && !loud {
            self.blocks_skipped += 1;
        } else {
            let resume = std::mem::replace(&mut self.skipping, false);
            match self.run_block(new, resume) {
                Ok(()) => {
                    converted = true;
                    self.blocks_converted += 1;
                    self.fade_in |= resume;
                }
                Err(e) => {
                    self.stream.restore(&self.silence);
                    self.produced.clear();
                    let h = self.hist.len();
                    fade(&mut self.hist[h - FADE_16K.min(h)..], false);
                    self.fade_in = true;
                    result = Err(e);
                }
            }
        }
        // Exactly one block joins the history per call (the steady state gives exactly that).
        let n = self.produced.len();
        if n != block {
            if converted && !self.length_warned {
                self.length_warned = true;
                tracing::warn!("voice changer: MeanVC2 produced {n} samples for a {block}-sample block; the output is realigned");
            }
            if n > block {
                self.produced.drain(..n - block);
            } else {
                self.produced
                    .splice(0..0, std::iter::repeat_n(0.0, block - n));
            }
        }
        if converted {
            if std::mem::take(&mut self.fade_in) {
                let n = FADE_16K.min(block);
                fade(&mut self.produced[..n], true);
            }
            if self.skip_enabled && self.quiet_16k >= SKIP_AFTER_QUIET_16K {
                let n = FADE_16K.min(block);
                fade(&mut self.produced[block - n..], false);
                self.skipping = true;
            }
        }
        shift_in(&mut self.hist, &self.produced);
        shift_in(&mut self.input, new);
        result
    }

    /// The newest `n` samples of the history (zeros in front when it is shorter).
    fn tail(&self, n: usize) -> Vec<f32> {
        let h = self.hist.len();
        if n <= h {
            self.hist[h - n..].to_vec()
        } else {
            let mut out = vec![0.0; n - h];
            out.extend_from_slice(&self.hist);
            out
        }
    }
}

impl VoiceConversionModel for MeanVcModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let block = self.block;
        let len = ctx.mono_16k.len();
        if len < block {
            return Err(format!("MeanVC2 converts {block}-sample blocks; this call carried {len} samples"));
        }
        if let Some(expected) = self.next_pos {
            if ctx.stream_pos_16k != expected {
                self.discontinuity();
            }
        }
        self.next_pos = Some(ctx.stream_pos_16k.wrapping_add(block as u64));
        self.follow_reference(ctx.reference);
        let new = &ctx.mono_16k[len - block..];
        let result = self.step(new);
        self.last_block.clear();
        self.last_block.extend_from_slice(new);
        result?;
        let mut out = self.tail(ctx.new_frames);
        // The post stage over the part the history holds, against the input it lines up with.
        let h = self.hist.len();
        let n = out.len().min(h);
        let at = out.len() - n;
        self.post
            .process(&self.input[h - n..h], &mut out[at..], &self.controls)?;
        Ok(out)
    }

    fn input_sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn output_sample_rate(&self) -> u32 {
        SAMPLE_RATE
    }

    fn min_context_samples(&self) -> usize {
        self.block
    }

    /// Runs one block of the warm-up audio from the steady silence (the loop's live shapes), then
    /// returns there.
    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        let len = ctx.mono_16k.len();
        if len < self.block {
            return Ok(());
        }
        self.stream.restore(&self.silence);
        self.discard.clear();
        let result = self
            .stream
            .process(&ctx.mono_16k[len - self.block..], &mut self.discard);
        self.stream.restore(&self.silence);
        self.next_pos = None;
        result.map(|_| ())
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        if controls.silence_db.is_finite() {
            self.threshold_db = controls.silence_db.clamp(-120.0, 0.0);
        }
        self.controls = *controls;
    }

    fn apply_quality(&mut self, level: &QualityLevel) -> bool {
        if self.configured_steps <= 1 {
            return false;
        }
        let steps = if level.context_scale < 1.0 { 1 } else { self.configured_steps };
        if steps == self.stream.steps() {
            return false;
        }
        self.stream.set_steps(steps);
        true
    }

    fn reset_stream(&mut self) {
        self.discontinuity();
        // A new stream: the harness's input ring starts from silence, and so does the reference.
        self.input.fill(0.0);
        self.post.reset();
    }

    fn fixed_block(&self) -> Option<usize> {
        Some(self.block)
    }

    /// The loop needs no context beyond the block (it carries its own state): the context is the
    /// harness's `want` (block plus the SOLA tail), which sizes the output history.
    fn configure_stream(&mut self, block: usize, want: usize, _extra: usize) -> Result<usize, String> {
        if block != self.block {
            return Err(format!("MeanVC2 runs {}-sample blocks, the harness asked for {block}", self.block));
        }
        let len = want.max(block);
        self.hist = vec![0.0; len];
        self.input = vec![0.0; len + self.align];
        self.post.reset();
        Ok(len)
    }
}
