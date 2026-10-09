//! Sigil Voice Engine (SVE): the `SigilRvc` family over the split templates (ADR 0004; plan "VC
//! best engine"). A voice imported from `.pth` / `.zip` (`manifest.format == "sigil-rvc-1"`) runs
//! here; bring-your-own ONNX voices keep the legacy full-decode `RvcModel`.
//!
//! One tick (exactly `block_16k` new 16 kHz samples, a fixed context, fixed shapes):
//! 1. content features over the context from the shared encoder pool ([`content`]);
//! 2. F0 over the new block plus a margin, with a rolling cache ([`f0_cache`] over the pitch bank);
//! 3. exact retrieval blend ([`retrieval`], P3-D);
//! 4. `enc` over the context, then `flowdec` over the return window only (partial decode,
//!    [`generator`]) with counter-based time-anchored noise and a continuous NSF phase
//!    ([`noise`]), all at the shapes of the [`plan`] of the preset ([`presets`]);
//! 5. the post stage at model rate ([`post`], P3-D: rms_mix, formant, silence gate, limiter).
//!
//! Sessions run through `neural_audio::gpu::SessionPlan` (P3-C2: IOBinding, CUDA graphs,
//! priority stream / queue, TensorRT-RTX), with the legacy host-tensor path whenever a GPU lever
//! is off or unavailable. The adaptive [`governor`] (P3-E) steps quality down under load through
//! `VoiceConversionModel::apply_quality`.
//!
//! # Ownership (P3)
//! | module | owner |
//! |---|---|
//! | `mod.rs`, `content`, `generator`, `noise`, `f0_cache`, `plan`, `presets` | P3-C1 |
//! | `retrieval`, `post` | P3-D |
//! | `governor` | P3-E |
//! | `options` | P3-S0 (frozen vocabulary; INTEGRATE only) |
//!
//! # Frozen by P3-S0 (change only through INTEGRATE; implementers may add, not rename)
//! - [`available`]`() -> bool`: the family can run in this build (the factory and
//!   `voice_changer::capabilities` gate `sigil-rvc-1` slots and the engine levers on it).
//! - [`SigilRvcRequest`] (owned fields, built by `voice_changer::factory` and the benchmark) and
//!   [`SigilRvcEngine::load`]`(SigilRvcRequest) -> Result<SigilRvcEngine, String>`, plus
//!   [`SigilRvcEngine::effective_levers`], [`SigilRvcEngine::ep_outcomes`] and
//!   [`SigilRvcEngine::warnings`]. The engine implements `VoiceConversionModel` (16 kHz in, the
//!   template's rate out) with the harness contract of `traits.rs`.
//!
//! # The engine and the stream harness (P3-C1)
//! The engine declares `fixed_block()` = its plan's block, so `VcStreamHarness` feeds it exactly
//! one block per call at one context length (`configure_stream` sizes the plan to the harness's
//! SOLA tail and extra context, rebuilding the sessions when that changes the shapes). For the
//! lowest latency the device block should be the plan's block at the device rate (`block_16k ×
//! rate / 16000` frames): then every block completes exactly when its audio is due and the harness
//! holds nothing back (`VcStreamHarness::fixed_hold_back_frames`).
//!
//! Every lever falls back to its legacy path independently ([`EffectiveLevers`] says which run):
//! `partial_decode` off decodes the whole context; `anchored_noise` off draws fresh noise every
//! tick; `nsf_phase0` off starts the NSF sine at phase 0 every tick; `f0_cache` off extracts F0 over
//! the whole context every tick.
//!
//! # Block joins and lookahead
//! The SOLA overlap of each join is the newest audio of the previous tick, decoded without future
//! context (the flow, the decoder, the text encoder and the content features all see the window's
//! right edge), so consecutive ticks never agree exactly there. Anchored noise and the carried NSF
//! phase remove the random part of the difference; [`SigilRvcEngine::set_lookahead_ms`] (default 0,
//! off) removes most of the rest by decoding a few frames of future context, at exactly that much
//! latency. Measured on the MIT default voice (DirectML, 40 ms blocks, 20 ms crossfade, two speech
//! clips; normalized cross-correlation of the overlapping audio at the chosen join): legacy paths
//! 0.69, all levers 0.76, all levers + 10 / 20 / 40 ms lookahead 0.83 / 0.88 / 0.92 (the legacy
//! paths with 20 ms lookahead reach 0.78). The benchmark decides the per-preset lookahead.
//!
//! # Added by P3-C1
//! [`SigilRvcEngine::set_silence_skip`], [`SigilRvcEngine::set_speaker_blend`],
//! [`SigilRvcEngine::set_lookahead_ms`], [`SigilRvcEngine::plan`], [`SigilRvcEngine::embedder`],
//! [`SigilRvcEngine::flowdec_precision`], [`SigilRvcEngine::tick_stats`], and the
//! `VoiceConversionModel::stream_report` of the engine (outcomes, levers and warnings after the
//! harness's `configure_stream`).
//!
//! # Silence
//! `set_silence_skip` is one switch for both silence paths: the engine-level skip (no inference
//! after [`SILENCE_HANGOVER_MS`] of quiet) and the post stage's silence gate (output muted after
//! its 150 ms hangover), both at `VoiceControls::silence_db`. With the skip off a quiet input is
//! converted and heard, as on a bring-your-own ONNX voice.
//!
//! # P3B layout (P3B-S0 split; owners)
//! | file | what | owner |
//! |---|---|---|
//! | `mod.rs` | types, accessors, the `VoiceConversionModel` impl | P3B-S0 / P3B-I0 (contract) |
//! | `load.rs` | load, generator build, re-plan, warm-up, `configure_stream` body | P3B-H |
//! | `tick.rs` | one tick (`convert_inner` / `convert_tick`) | P3B-D |
//! | `probe.rs` | benchmark hooks ([`probe::EngineProbe`], the engine's [`ModelTelemetry`]) | P3B-B |
//! | `overlap.rs` | the overlapped-F0 helper ([`overlap::F0Helper`]) | P3B-D |
//! | `content_stats.rs` | running content normalisation ([`content_stats::ContentStats`]) | P3B-D |
//! | `load_tests.rs` / `tick_tests.rs` | tests | P3B-H / P3B-D |
//! | `test_support.rs` | shared test fixtures (frozen) | P3B-S0 |
//!
//! # Frozen by P3B-S0 (change only through INTEGRATE; additive, defaults reproduce P5)
//! - [`SigilRvcRequest::lookahead_ms`] (0), [`SigilRvcRequest::stream`] ([`StreamGeometry`];
//!   `None` = the plan's default SOLA tail, `plan::DEFAULT_SOLA_TAIL_16K`) and
//!   [`SigilRvcRequest::content_context_16k`] (`None` = the `enc` window). Accepted and kept by
//!   the load (P3B-H makes the load use them; P3B-D the content context); until then the factory
//!   still calls [`SigilRvcEngine::set_lookahead_ms`] after the load.
//! - Engine fields `probe` ([`probe::EngineProbe`]), `content_stats`
//!   ([`content_stats::ContentStats`]) and `f0_helper` ([`overlap::F0Helper`]), all inert by
//!   default; a workstream keeps its own state inside its struct (rule R10).
//! - [`SigilRvcEngine::probe`] / [`SigilRvcEngine::probe_mut`] (the benchmark enables hooks before
//!   streaming), `VoiceConversionModel::time_anchored` (`anchored_noise` and `nsf_phase0` both in
//!   effect) and `VoiceConversionModel::telemetry` (the probe).

// The factory (`voice_changer::factory`) and the benchmark drive the engine; accessors such as
// `set_speaker_blend`, `tick_stats` and the submodules' diagnostics are for tests, the benchmark
// and later UI (speaker blending), so the module keeps its dead-code allowance.
#![allow(dead_code)]

pub mod content;
pub mod content_stats;
pub mod f0_cache;
pub mod generator;
pub mod governor;
mod load;
pub mod noise;
pub mod options;
pub mod overlap;
pub mod plan;
pub mod post;
pub mod presets;
pub mod probe;
pub mod retrieval;
mod tick;
#[cfg(test)]
mod load_tests;
#[cfg(test)]
mod test_support;
#[cfg(test)]
mod tick_tests;

use std::path::PathBuf;

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::import::layout::Template;

use super::manifest::RvcManifest;
use super::traits::{ConversionCtx, ModelTelemetry, PitchExtractor, StreamReport, VoiceControls, VoiceConversionModel};

use content::{ContentEncoderPool, EmbedderKind, SharedEncoder};
use f0_cache::F0Cache;
use generator::{SpeakerTable, SplitGenerator};
use governor::QualityLevel;
use noise::{NoiseFrames, PhaseTracker};
use plan::ShapePlan;
use post::{PostConfig, PostStage};
use retrieval::Retrieval;

pub use options::{Accelerator, DenoiserKind, EffectiveLevers, F0Range, Lever, LeverMode, LeverRequest, LeverSet, PrecisionMode};

/// Whether the SigilRvc family (template voices) runs in this build. P3-C1 returns `true` once
/// the engine converts audio; until then template slots are refused with a clear message.
pub fn available() -> bool {
    true
}

/// Engine-level silence skip: quiet ticks after this much quiet are not converted (no GPU work),
/// so word endings and breaths finish before the voice goes silent.
const SILENCE_HANGOVER_MS: f64 = 80.0;
/// Output frames of speech the silence gate lets ring out after the input falls quiet (post stage).
const POST_GATE_HANGOVER_MS: f32 = 150.0;

/// The stream harness's geometry, known when the voice loads (P3B-S0, frozen; P3B-H fills it from
/// the factory so the load plans the final shapes once).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct StreamGeometry {
    /// Samples every call returns at 16 kHz (block + SOLA tail; `ConversionCtx::new_frames`).
    pub want_16k: usize,
    /// The extra context the harness asks for (16 kHz samples).
    pub harness_extra_16k: usize,
    /// The harness's crossfade (ms).
    pub crossfade_ms: f64,
}

/// Everything needed to load one template voice. Built by `voice_changer::factory` from the
/// params (and by the benchmark from its spec), so both run the same engine.
pub struct SigilRvcRequest {
    /// The slot folder (`manifest.json`, `weights.f32.bin`, `speakers.f32.bin`, optional index).
    pub slot_dir: PathBuf,
    /// Its manifest (format `sigil-rvc-1`, read with `manifest::load_slot_manifest`).
    pub manifest: RvcManifest,
    /// Folder holding `<template id>/layout.json` (Tauri's resource dir, or
    /// `import::default_templates_dir`).
    pub templates_dir: PathBuf,
    /// EP ladder id the sessions are built on (`cpu` / `cuda` / `dml`; accelerators are in
    /// `levers.accelerator`).
    pub ep: String,
    pub session: VcSessionOptions,
    /// Resolved levers, accelerator and precision request (`capabilities::resolve_levers`).
    pub levers: EffectiveLevers,
    /// The pitch extractor of the run (already resolved by the pitch rules of `params.rs`).
    pub pitch: Box<dyn PitchExtractor>,
    /// Explicit content-encoder model (the UI's HuBERT path); `None` = the shared pool resolves the
    /// slot's embedder through the model store.
    pub content_encoder: Option<PathBuf>,
    /// FAISS index adopted into the slot, if any.
    pub index: Option<PathBuf>,
    /// Shapes and defaults of the preset (block, context, precision).
    pub preset: presets::EnginePreset,
    pub f0_range: F0Range,
    /// Seed of the time-anchored noise (fixed per voice, so a restart reproduces the same noise).
    pub noise_seed: u64,
    // ---- P3B-S0 (frozen; defaults reproduce P5) ----
    /// Lookahead (ms) the load plans with (0 = off). Kept by the load until P3B-H plans with it;
    /// until then the factory calls [`SigilRvcEngine::set_lookahead_ms`] after the load.
    pub lookahead_ms: f64,
    /// The stream harness's geometry; `None` = the plan's default SOLA tail
    /// (`plan::DEFAULT_SOLA_TAIL_16K`) until `configure_stream` reports the real one.
    pub stream: Option<StreamGeometry>,
    /// Left context of the content encoder (16 kHz samples); `None` = the `enc` window (P3B-D).
    pub content_context_16k: Option<usize>,
}

/// Load-time switches that are not part of the IPC contract.
#[derive(Clone, Copy, Debug, Default)]
struct LoadOptions {
    /// Accept templates whose frames are not 10 ms (the test fixture's toy template).
    allow_any_frame_rate: bool,
}

/// Per-tick compute breakdown (ms) of the last converted tick, for the benchmark and telemetry.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TickStats {
    pub content_ms: f64,
    pub f0_ms: f64,
    pub enc_ms: f64,
    pub flowdec_ms: f64,
    pub total_ms: f64,
    /// F0 frames the extractor analysed.
    pub f0_frames: usize,
    /// Whether the tick was skipped as silence (no inference).
    pub skipped: bool,
}

/// A loaded template voice.
pub struct SigilRvcEngine {
    levers: EffectiveLevers,
    outcomes: Vec<(String, EpOutcome)>,
    warnings: Vec<String>,
    output_sample_rate: u32,

    // ---- configuration ----
    slot_dir: PathBuf,
    weights_file: PathBuf,
    template: Template,
    ep: String,
    session: VcSessionOptions,
    preset_extra_16k: usize,
    /// Extra context of the stream (the harness's, at least the preset's).
    stream_extra_16k: usize,
    want_f16: bool,
    embedder: EmbedderKind,
    emb: usize,
    latent: usize,
    upp: usize,
    has_f0: bool,
    f0_range: F0Range,
    load_opts: LoadOptions,
    /// P3B-S0: the request's lookahead, stream geometry and content context (see
    /// [`SigilRvcRequest`]), kept for the workstreams that plan with them.
    requested_lookahead_ms: f64,
    stream_geometry: Option<StreamGeometry>,
    content_context_16k: Option<usize>,

    // ---- models ----
    plan: ShapePlan,
    /// `None` only while [`replan`](Self::replan) rebuilds it (and after a rebuild that could not
    /// restore the old shapes either: every tick then fails with a reason).
    generator: Option<SplitGenerator>,
    /// Always `Some` while the engine lives (taken in `Drop` so the pool can release it).
    encoder: Option<SharedEncoder>,
    pitch: Box<dyn PitchExtractor>,
    f0_cache: F0Cache,
    retrieval: Option<Retrieval>,
    retrieval_on: bool,
    speakers: SpeakerTable,
    blend: Option<Vec<(i64, f32)>>,
    post: PostStage,

    // ---- stream state ----
    noise_seed: u64,
    z_frames: NoiseFrames,
    src_frames: NoiseFrames,
    phase: PhaseTracker,
    /// Absolute frame the tracker's phase belongs to (the last window start).
    phase_frame: Option<u64>,
    /// F0 (Hz, shifted) of the last decoded window and its absolute start frame.
    prev_f0: Vec<f32>,
    prev_f0_start: u64,
    controls: VoiceControls,
    silence_skip: bool,
    quiet_ticks: u32,
    tick: u64,
    stats: TickStats,
    /// Generator builds since load (the load's included; tests and telemetry).
    generator_builds: u32,

    // ---- P3B-S0 stubs (inert by default; each workstream keeps its state inside its struct) ----
    /// Benchmark hooks and the per-call stage record (P3B-B).
    probe: probe::EngineProbe,
    /// Running content normalisation (P3B-D, the `content_norm` lever).
    content_stats: content_stats::ContentStats,
    /// F0 alongside the content encoder (P3B-D).
    f0_helper: overlap::F0Helper,

    // ---- scratch (reused every tick) ----
    ctx_buf: Vec<f32>,
    feats: Vec<f32>,
    feats0: Vec<f32>,
    phone: Vec<f32>,
    f0_ctx: Vec<f32>,
    pitchf: Vec<f32>,
    coarse: Vec<i64>,
    g: Vec<f32>,
    z: Vec<f32>,
    src: Vec<f32>,
    audio: Vec<f32>,
}

impl SigilRvcEngine {
    /// Levers the loaded engine actually uses (a requested lever the slot or EP cannot run is off,
    /// with a note).
    pub fn effective_levers(&self) -> &EffectiveLevers {
        &self.levers
    }

    /// EP outcome per session component (`vc.enc`, `vc.flowdec`, `vc.knn`, `vc.content`,
    /// `vc.pitch`).
    pub fn ep_outcomes(&self) -> &[(String, EpOutcome)] {
        &self.outcomes
    }

    /// Non-fatal load issues (shown as `rvc_warning`).
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Skip inference while the input is quiet (the legacy `silence_skip`, default on) and gate the
    /// converted output below the same threshold (`VoiceControls::silence_db`, the post stage's
    /// silence gate). Off converts and plays every tick however quiet the input is (a whisper, a
    /// low-gain microphone), like a bring-your-own ONNX voice with the skip off.
    pub fn set_silence_skip(&mut self, on: bool) {
        if on != self.silence_skip {
            self.post = post_stage(self.output_sample_rate, on);
        }
        self.silence_skip = on;
        self.quiet_ticks = 0;
    }

    /// Blend several speakers of a multi-speaker voice (weights normalised); `None` returns to the
    /// speaker id of each tick.
    pub fn set_speaker_blend(&mut self, weights: Option<&[(i64, f32)]>) -> Result<(), String> {
        match weights {
            None => self.blend = None,
            Some(w) => {
                let mut probe = vec![0.0; self.speakers.dim()];
                self.speakers.blend(w, &mut probe)?;
                self.blend = Some(w.to_vec());
            }
        }
        Ok(())
    }

    /// The shapes the engine runs.
    pub fn plan(&self) -> &ShapePlan {
        &self.plan
    }

    pub fn embedder(&self) -> EmbedderKind {
        self.embedder
    }

    /// `fp32` or `fp16` (the flowdec graph; `enc` is always fp32).
    pub fn flowdec_precision(&self) -> &'static str {
        self.generator
            .as_ref()
            .map_or("fp32", SplitGenerator::flowdec_precision)
    }

    /// Compute breakdown of the last tick.
    pub fn tick_stats(&self) -> TickStats {
        self.stats
    }

    /// The benchmark hooks (P3B-S0; read side).
    pub fn probe(&self) -> &probe::EngineProbe {
        &self.probe
    }

    /// The benchmark hooks (P3B-S0): a benchmark enables them here before streaming (P3B-B).
    pub fn probe_mut(&mut self) -> &mut probe::EngineProbe {
        &mut self.probe
    }

    /// Whether consecutive ticks decode identical audio on the frames they share: time-anchored
    /// noise and the continuous NSF phase both in effect (P3B-S0).
    pub fn is_time_anchored(&self) -> bool {
        self.levers.levers.get(Lever::AnchoredNoise) && self.levers.levers.get(Lever::NsfPhase0)
    }
}

/// The post stage of a template voice: limiter always, the silence gate with the silence skip.
fn post_stage(sample_rate: u32, silence_gate: bool) -> PostStage {
    PostStage::new(PostConfig {
        sample_rate,
        limiter: true,
        silence_gate,
        silence_hangover_ms: POST_GATE_HANGOVER_MS,
    })
}

impl Drop for SigilRvcEngine {
    fn drop(&mut self) {
        // Hand the encoder back so the pool frees it when no other hot slot uses it.
        drop(self.encoder.take());
        ContentEncoderPool::global().release_unused();
    }
}

impl VoiceConversionModel for SigilRvcEngine {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        self.convert_inner(ctx, true)
    }

    fn input_sample_rate(&self) -> u32 {
        16_000
    }

    fn output_sample_rate(&self) -> u32 {
        self.output_sample_rate
    }

    fn min_context_samples(&self) -> usize {
        super::rvc::MIN_CONTEXT_16K
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        let _span = crate::services::neural_audio::telemetry::load_span(crate::services::neural_audio::telemetry::LoadStage::HarnessWarmup);
        // The live shape, silence skip bypassed, twice (whole-window and incremental F0), then the
        // stream starts clean.
        let first = self.convert_inner(ctx, false).map(|_| ());
        let shifted = ConversionCtx {
            mono_16k: ctx.mono_16k,
            new_frames: ctx.new_frames,
            pitch_shift_semitones: ctx.pitch_shift_semitones,
            index_rate: ctx.index_rate,
            protect: ctx.protect,
            speaker_id: ctx.speaker_id,
            stream_pos_16k: ctx.stream_pos_16k + self.plan.block_16k as u64,
            reference: ctx.reference,
            singing: ctx.singing,
        };
        let second = self.convert_inner(&shifted, false).map(|_| ());
        self.reset_stream();
        first.and(second)
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        self.controls = *controls;
    }

    fn apply_quality(&mut self, level: &QualityLevel) -> bool {
        // Retrieval is the step this engine applies itself (context and F0 steps need the other
        // pre-planned shapes and extractors, P3-E).
        if self.retrieval.is_some() && self.retrieval_on != level.retrieval {
            self.retrieval_on = level.retrieval;
            return true;
        }
        false
    }

    fn reset_stream(&mut self) {
        self.f0_cache.reset();
        self.phase.reset();
        self.phase_frame = None;
        self.prev_f0.clear();
        self.quiet_ticks = 0;
        self.post.reset();
        self.content_stats.reset();
        self.probe.reset_stream();
    }

    fn fixed_block(&self) -> Option<usize> {
        Some(self.plan.block_16k)
    }

    fn stream_report(&self) -> Option<StreamReport> {
        Some(StreamReport {
            outcomes: self.outcomes.clone(),
            levers: self.levers.clone(),
            warnings: self.warnings.clone(),
        })
    }

    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        self.configure_stream_shapes(block, want, extra)
    }

    fn time_anchored(&self) -> bool {
        self.is_time_anchored()
    }

    fn telemetry(&self) -> Option<&dyn ModelTelemetry> {
        Some(&self.probe)
    }
}
