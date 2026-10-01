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

// The factory (`voice_changer::factory`) and the benchmark drive the engine; accessors such as
// `set_speaker_blend`, `tick_stats` and the submodules' diagnostics are for tests, the benchmark
// and later UI (speaker blending), so the module keeps its dead-code allowance.
#![allow(dead_code)]

pub mod content;
pub mod f0_cache;
pub mod generator;
pub mod governor;
pub mod noise;
pub mod options;
pub mod plan;
pub mod post;
pub mod presets;
pub mod retrieval;

use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{normalize_ep, VcSessionOptions};
use crate::services::neural_audio::vc::import::layout::Template;
use crate::services::neural_audio::vc::import::session::WeightImages;
use crate::services::neural_audio::vc::import::{writer, IMPORT_RECORD, SPEAKERS_FILE};

use super::manifest::{RvcManifest, SlotFormat};
use super::rvc::coarse_pitch;
use super::traits::{ConversionCtx, PitchExtractor, StreamReport, VoiceControls, VoiceConversionModel};

use content::{ContentEncoderPool, EmbedderKind, SharedEncoder};
use f0_cache::{F0Cache, F0CacheConfig};
use generator::{SpeakerTable, SplitGenerator};
use governor::QualityLevel;
use noise::{AnchoredNoise, NoiseFrames, NoiseStream, PhaseTracker};
use plan::{ShapePlan, HOP_16K};
use post::{PostConfig, PostStage};
use retrieval::{Retrieval, RetrievalBackend};

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

fn load_err(slot: &Path, e: impl std::fmt::Display) -> String {
    format!("voice {}: {e}", slot.display())
}

/// `weights_id` / `template` of the slot's import record, when the record exists and parses.
fn import_record_ids(slot_dir: &Path, manifest: &RvcManifest) -> Option<(String, String)> {
    let name = manifest.import_record.as_deref().unwrap_or(IMPORT_RECORD);
    let bytes = std::fs::read(slot_dir.join(name)).ok()?;
    let v: serde_json::Value = serde_json::from_slice(&bytes).ok()?;
    Some((v.get("weights_id")?.as_str()?.to_string(), v.get("template")?.as_str()?.to_string()))
}

/// Safe file name inside a slot (no separators, no parent references).
fn slot_file(slot_dir: &Path, name: &str) -> Result<PathBuf, String> {
    let n = name.trim();
    if n.is_empty() || n.contains(['/', '\\', ':']) || n.starts_with('.') {
        return Err(format!(
            "manifest.json names the file {name:?}, which is not a plain file name in the slot"
        ));
    }
    Ok(slot_dir.join(n))
}

fn db_to_rms(db: f32) -> f32 {
    10f32.powf(db / 20.0)
}

impl SigilRvcEngine {
    /// Load the voice: template, weights, shared encoder, pitch cache, generator sessions, warm-up
    /// and graph capture at the plan's shapes.
    pub fn load(request: SigilRvcRequest) -> Result<SigilRvcEngine, String> {
        Self::load_with(request, LoadOptions::default())
    }

    fn load_with(request: SigilRvcRequest, load_opts: LoadOptions) -> Result<SigilRvcEngine, String> {
        let SigilRvcRequest {
            slot_dir,
            manifest,
            templates_dir,
            ep,
            session,
            levers: requested,
            pitch,
            content_encoder,
            index,
            preset,
            f0_range,
            noise_seed,
        } = request;
        let fail = |e: String| load_err(&slot_dir, e);
        if manifest.slot_format() != SlotFormat::SigilRvc {
            return Err(fail(format!("format '{}' is not a Sigil template voice", manifest.format)));
        }
        let template_id = manifest.template.clone().unwrap_or_default();
        let template = Template::load(&templates_dir, template_id.trim()).map_err(|e| fail(e.to_string()))?;
        let layout = &template.layout;
        let arch = layout.arch.clone();
        if arch.frame_rate != 100 && !load_opts.allow_any_frame_rate {
            return Err(fail(format!(
                "template {} has {} frames per second (RVC uses 100)",
                layout.template, arch.frame_rate
            )));
        }
        let mut warnings: Vec<String> = Vec::new();
        match import_record_ids(&slot_dir, &manifest) {
            Some((weights_id, tpl)) if weights_id != layout.weights.weights_id || tpl != layout.template => {
                return Err(fail(format!(
                    "was imported for another export of template {tpl}; import the voice again to use it with this version of Sigil"
                )));
            }
            Some(_) => {}
            None => warnings.push("the slot has no readable import record (sigil-import.json); its weights are checked by size only".into()),
        }
        let weights_file = slot_file(
            &slot_dir,
            manifest
                .weights
                .as_deref()
                .unwrap_or(super::import::WEIGHTS_FILE),
        )
        .map_err(&fail)?;
        let speakers_file = slot_file(&slot_dir, manifest.speakers.as_deref().unwrap_or(SPEAKERS_FILE)).map_err(&fail)?;
        let speakers = SpeakerTable::load(&speakers_file, layout.speakers.dim).map_err(&fail)?;
        if let Some(n) = manifest
            .speaker_count
            .filter(|&n| n as usize != speakers.count())
        {
            warnings.push(format!("manifest.json lists {n} speakers, the speaker table holds {}", speakers.count()));
        }
        let embedder = EmbedderKind::from_manifest(manifest.embedder.as_deref(), arch.emb_channels).map_err(&fail)?;
        let ep = normalize_ep(&ep).to_string();
        let gpu = ep != "cpu";
        let out_sr = arch.upp * 100;

        // Levers the voice can use (the request is already resolved against the build and EP).
        let mut levers = requested;
        if !arch.f0 {
            for lever in [Lever::NsfPhase0, Lever::F0Cache] {
                if levers.levers.get(lever) {
                    levers.levers.set(lever, false);
                    levers
                        .notes
                        .push(format!("{}: the voice has no pitch guidance", lever.as_str()));
                }
            }
        }
        let plan = ShapePlan::new(&layout.streaming, &preset, out_sr, levers.levers.get(Lever::PartialDecode)).map_err(&fail)?;

        // fp16 weights only when a GPU (or an explicit fp16 request) will use them.
        let wants_fp16 = match levers.precision {
            PrecisionMode::Fp16 => true,
            PrecisionMode::Fp32 => false,
            PrecisionMode::Auto => gpu,
        };
        let want_f16 = wants_fp16 && manifest.fp16_ok != Some(false);
        if wants_fp16 && manifest.fp16_ok == Some(false) {
            warnings.push("precision: the importer marked this voice fp32-only (its fp16 check failed)".into());
        }
        let generator = Self::build_generator(&template, &weights_file, &plan, &ep, &session, &levers, want_f16, &mut warnings).map_err(&fail)?;

        // Content encoder from the shared pool.
        let explicit = content_encoder
            .as_deref()
            .filter(|p| !p.as_os_str().is_empty());
        if explicit.is_some() && !matches!(embedder, EmbedderKind::ContentVec768 | EmbedderKind::ContentVec256) {
            warnings.push(format!(
                "the voice uses the {} content encoder; the HuBERT / ContentVec path from the settings is not used for it",
                embedder.as_str()
            ));
        }
        let encoder = content::acquire_for_slot(
            embedder,
            explicit,
            Some(&slot_dir),
            &ep,
            &session,
            levers.precision,
            plan.context_16k,
            &mut warnings,
        )
        .map_err(&fail)?;
        let encoder_outcome = encoder
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .outcome()
            .clone();

        // Retrieval (P3-D backends behind one API).
        let emb = arch.emb_channels as usize;
        let index = index
            .or_else(|| manifest.index_path(&slot_dir))
            .filter(|p| p.is_file());
        let mut retrieval = None;
        if let Some(path) = index {
            let backend = if levers.levers.get(Lever::GpuKnn) && retrieval::gpu_knn_available() {
                RetrievalBackend::GpuKnn
            } else {
                RetrievalBackend::CpuIvf
            };
            let loaded = Retrieval::load(&path, backend, &ep, &session, levers.precision).or_else(|e| {
                if backend == RetrievalBackend::GpuKnn {
                    levers
                        .notes
                        .push(format!("gpu_knn: {e}; using the CPU index"));
                    Retrieval::load(&path, RetrievalBackend::CpuIvf, &ep, &session, levers.precision)
                } else {
                    Err(e)
                }
            });
            match loaded {
                Ok(r) if r.dim() == emb || r.dim() == 0 => retrieval = Some(r),
                Ok(r) => warnings.push(format!(
                    "retrieval index {} has {} dimensions but the voice uses {emb}; index rate has no effect",
                    path.display(),
                    r.dim()
                )),
                Err(e) => warnings.push(format!(
                    "retrieval index {} could not be read ({e}); index rate has no effect",
                    path.display()
                )),
            }
        }
        if levers.levers.get(Lever::GpuKnn)
            && retrieval
                .as_ref()
                .is_none_or(|r| r.backend() != RetrievalBackend::GpuKnn)
        {
            levers.levers.set(Lever::GpuKnn, false);
            if retrieval.is_none() {
                levers
                    .notes
                    .push("gpu_knn: the voice has no retrieval index".into());
            }
        }
        // GPU levers: what the session plans really enabled (P3-C2).
        correct_gpu_levers(&mut levers, &generator);
        warnings.extend(generator.notes().iter().cloned());

        let mut outcomes: Vec<(String, EpOutcome)> = generator.ep_outcomes().to_vec();
        outcomes.push(("vc.content".to_string(), encoder_outcome));

        let f0_cache = Self::new_f0_cache(pitch.as_ref(), &plan);
        let noise = AnchoredNoise::new(noise_seed);
        let upp = arch.upp as usize;
        let latent = arch.inter_channels as usize;
        let ring = plan.flow_frames + plan.block_16k / HOP_16K + 8;
        // The post stage's silence gate follows the silence skip (on by default, `set_silence_skip`).
        let post = post_stage(out_sr, true);
        let mut engine = SigilRvcEngine {
            levers,
            outcomes,
            warnings,
            output_sample_rate: out_sr,
            slot_dir,
            weights_file,
            preset_extra_16k: (preset.extra_context_ms.max(0.0) * 16.0).round() as usize,
            stream_extra_16k: (preset.extra_context_ms.max(0.0) * 16.0).round() as usize,
            want_f16,
            embedder,
            emb,
            latent,
            upp,
            has_f0: arch.f0,
            f0_range,
            load_opts,
            plan,
            generator: Some(generator),
            encoder: Some(encoder),
            pitch,
            f0_cache,
            retrieval,
            retrieval_on: true,
            speakers,
            blend: None,
            post,
            noise_seed,
            z_frames: NoiseFrames::new(noise, NoiseStream::Flow, latent, ring),
            src_frames: NoiseFrames::new(noise, NoiseStream::Source, upp, ring),
            phase: PhaseTracker::new(upp, arch.sample_rate),
            phase_frame: None,
            prev_f0: Vec::new(),
            prev_f0_start: 0,
            controls: VoiceControls::default(),
            silence_skip: true,
            quiet_ticks: 0,
            tick: 0,
            stats: TickStats::default(),
            generator_builds: 1,
            ctx_buf: Vec::new(),
            feats: Vec::new(),
            feats0: Vec::new(),
            phone: Vec::new(),
            f0_ctx: Vec::new(),
            pitchf: Vec::new(),
            coarse: Vec::new(),
            g: Vec::new(),
            z: Vec::new(),
            src: Vec::new(),
            audio: Vec::new(),
            template,
            ep,
            session,
        };
        engine.size_buffers();
        // One conversion at the live shape: a template / encoder that loads but cannot run (a graph
        // whose inputs differ, an encoder with the wrong width) fails the load, not the first tick.
        engine
            .warm_once()
            .map_err(|e| load_err(&engine.slot_dir, format!("test conversion failed: {e}")))?;
        engine.reset_stream();
        Ok(engine)
    }

    #[allow(clippy::too_many_arguments)]
    fn build_generator(
        template: &Template,
        weights_file: &Path,
        plan: &ShapePlan,
        ep: &str,
        session: &VcSessionOptions,
        levers: &EffectiveLevers,
        want_f16: bool,
        warnings: &mut Vec<String>,
    ) -> Result<SplitGenerator, String> {
        let layout = &template.layout;
        let f32_image = std::fs::read(weights_file).map_err(|e| format!("{}: {e}", weights_file.display()))?;
        if f32_image.len() as u64 != layout.weights.bytes_f32 {
            return Err(format!(
                "{} holds {} bytes, template {} expects {}; import the voice again",
                weights_file.display(),
                f32_image.len(),
                layout.template,
                layout.weights.bytes_f32
            ));
        }
        let f16_image = if want_f16 {
            match writer::f16_image(&f32_image) {
                Ok(img) => Some(img),
                Err(e) => {
                    warnings.push(format!("precision: fp32 only ({e})"));
                    None
                }
            }
        } else {
            None
        };
        let images = WeightImages {
            f32: &f32_image,
            f16: f16_image.as_deref(),
        };
        SplitGenerator::load(template, &images, plan, ep, session, levers)
        // The images drop here: ONNX Runtime copied the weights into its sessions.
    }

    fn new_f0_cache(pitch: &dyn PitchExtractor, plan: &ShapePlan) -> F0Cache {
        let (lead, margin) = match pitch.receptive_frames() {
            Some((left, right)) => (left, right.max(1)),
            None => (plan.f0_lead_frames, plan.f0_margin_frames),
        };
        F0Cache::new(F0CacheConfig {
            margin_frames: margin,
            capacity_frames: plan.enc_frames * 2,
        })
        .with_lead_frames(lead)
    }

    /// Size the per-tick buffers for the plan (no allocation per tick afterwards).
    fn size_buffers(&mut self) {
        let p = self.plan;
        let t = p.enc_frames;
        self.ctx_buf.reserve(p.context_16k);
        self.feats.reserve(t * self.emb);
        self.feats0.reserve(t * self.emb);
        self.phone.resize(t * self.emb, 0.0);
        self.f0_ctx.resize(t, 0.0);
        self.pitchf.resize(t, 0.0);
        self.coarse.resize(t, 1);
        self.z.resize(self.latent * p.flow_frames, 0.0);
        self.src.resize(p.flow_frames * self.upp, 0.0);
        self.audio.resize(p.decode_frames * self.upp, 0.0);
        self.prev_f0.reserve(p.flow_frames);
        // The noise caches hold one window plus a block of look-back.
        let ring = p.flow_frames + p.block_16k / HOP_16K + 8;
        let noise = AnchoredNoise::new(self.noise_seed);
        self.z_frames = NoiseFrames::new(noise, NoiseStream::Flow, self.latent, ring);
        self.src_frames = NoiseFrames::new(noise, NoiseStream::Source, self.upp, ring);
    }

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

    /// Lookahead (ms, rounded up to 10 ms frames, 0..=80): the returned audio ends that much before
    /// the newest input, so every returned sample (the SOLA overlap included) was decoded with
    /// future context. Adds exactly that much latency. Call before the stream harness is built
    /// (it sizes its context from the plan); rebuilds the sessions when the shapes change.
    pub fn set_lookahead_ms(&mut self, ms: f64) -> Result<(), String> {
        let frames = if ms.is_finite() && ms > 0.0 {
            (ms / 10.0 - 1e-9).ceil() as usize
        } else {
            0
        };
        let p = self.plan;
        let streaming = self.template.layout.streaming.clone();
        let plan = ShapePlan::for_stream_with_lookahead(
            &streaming,
            self.output_sample_rate,
            p.partial_decode,
            p.block_16k,
            p.want_16k,
            self.stream_extra_16k,
            frames,
        )?;
        if plan != self.plan {
            self.replan(plan)?;
        }
        Ok(())
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

    /// Switch to a new plan. The generator's sessions are rebuilt (weights re-read from the slot)
    /// only when its tensor shapes change ([`generator_shape`]); a plan that differs only in the
    /// returned span (another crossfade within the same frames) keeps them. A rebuild frees the old
    /// sessions first, so VRAM peaks at one generator (on a failure the old shapes are rebuilt and
    /// the voice keeps running as it was), and refreshes the EP outcomes, the GPU-lever corrections
    /// and the notes from the new session plans (plan T7: the EP in use is what the voice reports;
    /// [`VoiceConversionModel::stream_report`] hands them to the factory after the harness's
    /// `configure_stream`).
    fn replan(&mut self, plan: ShapePlan) -> Result<(), String> {
        if generator_shape(&plan) != generator_shape(&self.plan) || self.generator.is_none() {
            let old_plan = self.plan;
            self.generator = None;
            let mut notes = Vec::new();
            let built = Self::build_generator(
                &self.template,
                &self.weights_file,
                &plan,
                &self.ep,
                &self.session,
                &self.levers,
                self.want_f16,
                &mut notes,
            );
            let generator = match built {
                Ok(g) => g,
                Err(e) => {
                    let mut ignored = Vec::new();
                    match Self::build_generator(
                        &self.template,
                        &self.weights_file,
                        &old_plan,
                        &self.ep,
                        &self.session,
                        &self.levers,
                        self.want_f16,
                        &mut ignored,
                    ) {
                        Ok(g) => self.generator = Some(g),
                        Err(e2) => tracing::warn!("voice changer: restoring the generator after a failed re-plan failed too: {e2}"),
                    }
                    return Err(e);
                }
            };
            correct_gpu_levers(&mut self.levers, &generator);
            notes.extend(generator.notes().iter().cloned());
            for n in notes {
                if !self.warnings.contains(&n) {
                    self.warnings.push(n);
                }
            }
            // The generator's components from the new sessions; the rest (the content encoder)
            // are unchanged.
            let mut outcomes = generator.ep_outcomes().to_vec();
            outcomes.extend(
                self.outcomes
                    .drain(..)
                    .filter(|(c, _)| !generator.ep_outcomes().iter().any(|(g, _)| g == c)),
            );
            self.outcomes = outcomes;
            self.generator = Some(generator);
            self.generator_builds += 1;
        }
        self.plan = plan;
        self.f0_cache = Self::new_f0_cache(self.pitch.as_ref(), &plan);
        self.size_buffers();
        self.reset_stream();
        Ok(())
    }

    /// Run one conversion on a quiet tone at the plan's shape (twice, so the F0 cache's incremental
    /// shape runs too); the stream state is reset afterwards by the caller.
    fn warm_once(&mut self) -> Result<(), String> {
        let n = self.plan.context_16k;
        let tone: Vec<f32> = (0..n + self.plan.block_16k)
            .map(|i| 0.05 * (std::f32::consts::TAU * 150.0 * i as f32 / 16_000.0).sin())
            .collect();
        let want = self.plan.want_16k;
        for (k, start) in [0usize, self.plan.block_16k].into_iter().enumerate() {
            let ctx = ConversionCtx {
                mono_16k: &tone[start..start + n],
                new_frames: want,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: (k * self.plan.block_16k) as u64,
            };
            let out = self.convert_inner(&ctx, false)?;
            if out.iter().any(|x| !x.is_finite()) {
                return Err("the voice produced non-finite audio".into());
            }
        }
        Ok(())
    }

    /// Speaker vector of this tick into `self.g`.
    fn speaker_vector(&mut self, speaker_id: i64) -> Result<(), String> {
        self.g.resize(self.speakers.dim(), 0.0);
        match &self.blend {
            Some(w) => self.speakers.blend(w, &mut self.g),
            None => {
                self.g.copy_from_slice(self.speakers.vector(speaker_id));
                Ok(())
            }
        }
    }

    fn convert_inner(&mut self, ctx: &ConversionCtx<'_>, allow_skip: bool) -> Result<Vec<f32>, String> {
        let want = ctx.new_frames.min(ctx.mono_16k.len());
        if want == 0 {
            return Ok(Vec::new());
        }
        let started = Instant::now();
        let out_len = ((want as u64 * u64::from(self.output_sample_rate) + 8_000) / 16_000) as usize;
        let c = self.plan.context_16k;
        // The fixed shape; any other call (one-off conversions) is cut / zero-padded to it.
        let direct = ctx.mono_16k.len() == c && ctx.stream_pos_16k.is_multiple_of(HOP_16K as u64);
        let mut buf = std::mem::take(&mut self.ctx_buf);
        let (mono, pos): (&[f32], u64) = if direct {
            (ctx.mono_16k, ctx.stream_pos_16k)
        } else {
            let len = ctx.mono_16k.len();
            buf.clear();
            if len >= c {
                buf.extend_from_slice(&ctx.mono_16k[len - c..]);
            } else {
                buf.resize(c - len, 0.0);
                buf.extend_from_slice(ctx.mono_16k);
            }
            let end = ctx.stream_pos_16k + len as u64;
            let pos = end.saturating_sub(c as u64);
            (&buf, pos - pos % HOP_16K as u64)
        };
        let result = self.convert_tick(ctx, mono, pos, want, out_len, allow_skip);
        self.ctx_buf = buf;
        let mut out = result?;
        self.stats.total_ms = started.elapsed().as_secs_f64() * 1000.0;
        if !self.stats.skipped {
            // The input span the returned audio was converted from (earlier by the lookahead).
            let len = ctx.mono_16k.len();
            let end = len.saturating_sub(self.plan.lookahead_frames * HOP_16K);
            let source = &ctx.mono_16k[end.saturating_sub(want)..end];
            self.post.process(source, &mut out, &self.controls)?;
        }
        Ok(out)
    }

    fn convert_tick(
        &mut self,
        ctx: &ConversionCtx<'_>,
        mono: &[f32],
        pos: u64,
        want: usize,
        out_len: usize,
        allow_skip: bool,
    ) -> Result<Vec<f32>, String> {
        self.stats = TickStats::default();
        // Silence skip: no inference while the returned span is quiet (after a short hangover).
        if allow_skip && self.silence_skip {
            let tail = &mono[mono.len() - want.min(mono.len())..];
            let rms = (tail
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / tail.len().max(1) as f64)
                .sqrt() as f32;
            if rms < db_to_rms(self.controls.silence_db) {
                self.quiet_ticks = self.quiet_ticks.saturating_add(1);
                let block_ms = self.plan.block_16k as f64 / 16.0;
                let hangover = (SILENCE_HANGOVER_MS / block_ms).ceil() as u32;
                if self.quiet_ticks > hangover {
                    self.stats.skipped = true;
                    return Ok(vec![0.0; out_len]);
                }
            } else {
                self.quiet_ticks = 0;
            }
        }
        let p = self.plan;
        let (t, emb) = (p.enc_frames, self.emb);

        // 1. Content features over the context.
        let t0 = Instant::now();
        let frames50 = {
            let encoder = self
                .encoder
                .as_ref()
                .ok_or("the content encoder was released")?;
            let mut enc = encoder
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            enc.encode(mono, &mut self.feats)?
        };
        if frames50 == 0 {
            return Err("the content encoder returned no frames".into());
        }
        self.stats.content_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let protect_on = self.has_f0 && ctx.protect < 0.5;
        if protect_on {
            self.feats0.clear();
            self.feats0.extend_from_slice(&self.feats);
        }
        // 2. Retrieval on the frames that reach the flowdec window.
        if let Some(r) = self.retrieval.as_mut() {
            if self.retrieval_on && ctx.index_rate > 0.0 {
                let skip = (p.window_start / 2).min(frames50 - 1);
                r.blend(&mut self.feats[skip * emb..frames50 * emb], frames50 - skip, ctx.index_rate)?;
            }
        }
        // 3. 2x upsample onto the 10 ms grid (RVC realtime: duplicate the last 50 Hz frame).
        upsample_into(&self.feats[..frames50 * emb], emb, t, &mut self.phone);

        // 4. F0 (cached) → shifted Hz and coarse bins; protect.
        if self.has_f0 {
            let t1 = Instant::now();
            let f0 = if self.levers.levers.get(Lever::F0Cache) {
                self.f0_cache.update(self.pitch.as_mut(), mono, pos)?
            } else {
                self.f0_cache.extract_window(self.pitch.as_mut(), mono)?
            };
            if f0.len() < t {
                return Err(format!("the F0 contour has {} frames, the context {t}", f0.len()));
            }
            self.f0_ctx.copy_from_slice(&f0[..t]);
            self.stats.f0_frames = self.f0_cache.last_extracted_frames();
            self.stats.f0_ms = t1.elapsed().as_secs_f64() * 1000.0;
            let shift = 2f32.powf(ctx.pitch_shift_semitones / 12.0);
            for ((pf, cb), &hz) in self
                .pitchf
                .iter_mut()
                .zip(self.coarse.iter_mut())
                .zip(self.f0_ctx.iter())
            {
                let v = hz * shift;
                *pf = if v.is_finite() && v > 0.0 { v } else { 0.0 };
                *cb = coarse_pitch(*pf);
            }
            if protect_on {
                let protect = ctx.protect.max(0.0);
                let last = frames50 - 1;
                for (j, &pf) in self.pitchf.iter().enumerate() {
                    if pf < 1.0 {
                        let src = (j / 2).min(last) * emb;
                        for (x, &x0) in self.phone[j * emb..(j + 1) * emb]
                            .iter_mut()
                            .zip(&self.feats0[src..src + emb])
                        {
                            *x = *x * protect + x0 * (1.0 - protect);
                        }
                    }
                }
            }
        }

        // 5. enc over the context.
        let t2 = Instant::now();
        self.generator
            .as_mut()
            .ok_or(GENERATOR_MISSING)?
            .encode(&self.phone, &self.coarse)?;
        self.stats.enc_ms = t2.elapsed().as_secs_f64() * 1000.0;

        // 6. The flowdec window: anchored noise, NSF phase, speaker vector.
        let (ws, f) = (p.window_start, p.flow_frames);
        let abs_ws = pos / HOP_16K as u64 + ws as u64;
        let noise_scale = if self.controls.noise_scale.is_finite() {
            self.controls.noise_scale.max(0.0)
        } else {
            0.0
        };
        if self.levers.levers.get(Lever::AnchoredNoise) {
            self.z_frames
                .copy_window(abs_ws, f, noise_scale, true, &mut self.z);
            self.src_frames
                .copy_window(abs_ws, f, 1.0, false, &mut self.src);
        } else {
            // Legacy: fresh noise every tick.
            let fresh = AnchoredNoise::new(
                self.noise_seed
                    ^ self
                        .tick
                        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
                        .wrapping_add(1),
            );
            fresh.fill(NoiseStream::Flow, 0, noise_scale, &mut self.z);
            fresh.fill(NoiseStream::Source, 0, 1.0, &mut self.src);
        }
        let phase0 = if self.has_f0 && self.levers.levers.get(Lever::NsfPhase0) {
            match self.phase_frame {
                Some(pf) if pf == abs_ws => {}
                Some(pf) if pf < abs_ws && pf >= self.prev_f0_start && abs_ws <= self.prev_f0_start + self.prev_f0.len() as u64 => {
                    let from = (pf - self.prev_f0_start) as usize;
                    let to = (abs_ws - self.prev_f0_start) as usize;
                    self.phase.advance(&self.prev_f0[from..to]);
                }
                _ => self.phase.reset(),
            }
            self.phase_frame = Some(abs_ws);
            self.phase.phase_at()
        } else {
            0.0
        };
        self.speaker_vector(ctx.speaker_id)?;

        // 7. flowdec over the window (partial decode) or the whole context (legacy).
        let t3 = Instant::now();
        self.generator
            .as_mut()
            .ok_or(GENERATOR_MISSING)?
            .decode_window(ws, &self.z, &self.g, &self.pitchf[ws..ws + f], phase0, &self.src, &mut self.audio)?;
        self.stats.flowdec_ms = t3.elapsed().as_secs_f64() * 1000.0;
        if self.has_f0 {
            self.prev_f0.clear();
            self.prev_f0.extend_from_slice(&self.pitchf[ws..ws + f]);
            self.prev_f0_start = abs_ws;
        }
        self.tick = self.tick.wrapping_add(1);
        // Lookahead: the newest decoded frames are context only.
        let end = self
            .audio
            .len()
            .saturating_sub(p.lookahead_frames * self.upp);
        let n = out_len.min(end);
        Ok(self.audio[end - n..end].to_vec())
    }
}

/// Why a tick cannot run after a failed re-plan.
const GENERATOR_MISSING: &str = "the voice's generator could not be rebuilt for the stream's shape; restart the voice changer";

/// The post stage of a template voice: limiter always, the silence gate with the silence skip.
fn post_stage(sample_rate: u32, silence_gate: bool) -> PostStage {
    PostStage::new(PostConfig {
        sample_rate,
        limiter: true,
        silence_gate,
        silence_hangover_ms: POST_GATE_HANGOVER_MS,
    })
}

/// The generator's tensor shapes in a plan: two plans with the same shapes run the same sessions.
fn generator_shape(plan: &ShapePlan) -> (usize, usize, usize, usize) {
    (plan.enc_frames, plan.flow_frames, plan.decode_frames, plan.upp)
}

/// Turn off, with a note, the GPU levers and the accelerator the generator's session plans did not
/// enable (P3-C2: a plan falls back to host tensors, no graph, the default stream or plain CUDA).
fn correct_gpu_levers(levers: &mut EffectiveLevers, generator: &SplitGenerator) {
    let (enc_l, fd_l) = generator.plan_levers();
    for (lever, used) in [
        (Lever::IoBinding, enc_l.io_binding && fd_l.io_binding),
        (Lever::CudaGraphs, enc_l.cuda_graphs && fd_l.cuda_graphs),
        (Lever::GpuPriority, enc_l.priority && fd_l.priority),
    ] {
        if levers.levers.get(lever) && !used {
            levers.levers.set(lever, false);
            levers
                .notes
                .push(format!("{}: the session plans did not enable it", lever.as_str()));
        }
    }
    if levers.accelerator != Accelerator::None && fd_l.accelerator != levers.accelerator {
        levers
            .notes
            .push(format!("{}: the session plans did not enable it", levers.accelerator.as_str()));
        levers.accelerator = Accelerator::None;
    }
}

/// Nearest 2x upsampling of 50 Hz features onto `t` 10 ms frames into `out` (`t × emb`): frame `j`
/// takes 50 Hz frame `min(j / 2, last)` (RVC realtime's duplicated last frame + interpolate ×2).
fn upsample_into(feats: &[f32], emb: usize, t: usize, out: &mut Vec<f32>) {
    out.resize(t * emb, 0.0);
    let frames = feats.len() / emb.max(1);
    if frames == 0 {
        out.fill(0.0);
        return;
    }
    for j in 0..t {
        let src = (j / 2).min(frames - 1) * emb;
        out[j * emb..(j + 1) * emb].copy_from_slice(&feats[src..src + emb]);
    }
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
        if block != self.plan.block_16k {
            return Err(format!(
                "the stream harness asked for {block}-sample blocks, the plan runs {}",
                self.plan.block_16k
            ));
        }
        self.stream_extra_16k = extra.max(self.preset_extra_16k);
        let streaming = self.template.layout.streaming.clone();
        let plan = ShapePlan::for_stream_with_lookahead(
            &streaming,
            self.output_sample_rate,
            self.plan.partial_decode,
            block,
            want,
            self.stream_extra_16k,
            self.plan.lookahead_frames,
        )?;
        if plan != self.plan {
            self.replan(plan)?;
        }
        Ok(self.plan.context_16k)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::neural_audio::vc::manifest::load_slot_manifest;
    use crate::services::neural_audio::vc::stream::VcStreamHarness;
    use crate::services::neural_audio::vc::test_models;
    use crate::services::neural_audio::vc::world::DioPitchExtractor;
    use generator::tests::{tiny_dir, tiny_voice};

    /// A slot folder holding the tiny fixture voice (imported, with a manifest) and the stand-in
    /// content encoder; removed on drop.
    struct TinySlot {
        root: PathBuf,
        slot: PathBuf,
        encoder: PathBuf,
    }

    impl Drop for TinySlot {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.root);
        }
    }

    fn tiny_slot(tag: &str) -> TinySlot {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!("sigil-sve-{tag}-{}-{nanos}", std::process::id()));
        let slot = root.join("voice-changer").join("models").join("tiny");
        std::fs::create_dir_all(&slot).unwrap();
        let voice = tiny_voice();
        // The slot files the importer writes (P3-B's slot store owns the real writer): the weights
        // image, the speaker table and the import record fields the engine checks.
        std::fs::write(slot.join("weights.f32.bin"), &voice.weights.f32_image).unwrap();
        let speakers: Vec<u8> = voice
            .weights
            .speakers
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(slot.join("speakers.f32.bin"), speakers).unwrap();
        let record = serde_json::json!({
            "format": "sigil-rvc-import",
            "template": voice.template.layout.template,
            "weights_id": voice.template.layout.weights.weights_id,
        });
        std::fs::write(slot.join("sigil-import.json"), serde_json::to_vec_pretty(&record).unwrap()).unwrap();
        let manifest = serde_json::json!({
            "format": "sigil-rvc-1",
            "template": voice.template.layout.template,
            "weights": "weights.f32.bin",
            "speakers": "speakers.f32.bin",
            "speaker_count": voice.weights.n_speakers,
            "embedder": "contentvec",
            "vocoder": "hifigan",
            "fp16_ok": true,
            "import_record": "sigil-import.json"
        });
        std::fs::write(slot.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        let encoder = root.join("content.onnx");
        std::fs::write(&encoder, test_models::content_encoder_bytes()).unwrap();
        TinySlot { root, slot, encoder }
    }

    fn levers_all(on: bool) -> EffectiveLevers {
        let mut e = EffectiveLevers {
            precision: PrecisionMode::Fp32,
            ..EffectiveLevers::default()
        };
        for lever in [Lever::PartialDecode, Lever::AnchoredNoise, Lever::NsfPhase0, Lever::F0Cache] {
            e.levers.set(lever, on);
        }
        e
    }

    fn request(s: &TinySlot, levers: EffectiveLevers, block_ms: f64, extra_ms: f64) -> SigilRvcRequest {
        SigilRvcRequest {
            slot_dir: s.slot.clone(),
            manifest: load_slot_manifest(&s.slot).unwrap(),
            templates_dir: tiny_dir(),
            ep: "cpu".into(),
            session: VcSessionOptions::default(),
            levers,
            pitch: Box::new(DioPitchExtractor::new()),
            content_encoder: Some(s.encoder.clone()),
            index: None,
            preset: presets::engine_preset(RvcPresetProfile::LowLatency, block_ms, 0, 48_000, extra_ms, PrecisionMode::Fp32),
            f0_range: F0Range::Speech,
            noise_seed: 1234,
        }
    }

    fn tiny_engine(s: &TinySlot, levers: EffectiveLevers) -> SigilRvcEngine {
        SigilRvcEngine::load_with(request(s, levers, 40.0, 100.0), LoadOptions { allow_any_frame_rate: true }).unwrap()
    }

    use crate::services::neural_audio::RvcPresetProfile;

    /// A voiced test signal with pauses (16 kHz).
    fn voice_16k(len: usize) -> Vec<f32> {
        let mut phase = 0.0f64;
        (0..len)
            .map(|i| {
                let t = i as f64 / 16_000.0;
                phase += (140.0 + 40.0 * (t * 1.3).sin()) / 16_000.0;
                let gate = if (t % 1.0) > 0.8 { 0.0 } else { 1.0 };
                (gate * 0.3 * (std::f64::consts::TAU * phase).sin()) as f32
            })
            .collect()
    }

    /// The production loader refuses the toy fixture template (its frames are not 10 ms); the test
    /// loader runs it. Everything the engine reports is filled in.
    #[test]
    fn tiny_voice_loads_with_its_levers_and_outcomes() {
        let s = tiny_slot("load");
        let err = SigilRvcEngine::load(request(&s, levers_all(true), 40.0, 100.0))
            .err()
            .unwrap();
        assert!(err.contains("frames per second"), "{err}");
        let e = tiny_engine(&s, levers_all(true));
        assert_eq!(e.output_sample_rate(), 800);
        assert_eq!(e.fixed_block(), Some(640));
        assert_eq!(e.embedder(), EmbedderKind::ContentVec768);
        assert_eq!(e.flowdec_precision(), "fp32");
        let comps: Vec<&str> = e.ep_outcomes().iter().map(|(c, _)| c.as_str()).collect();
        assert_eq!(comps, ["vc.enc", "vc.flowdec", "vc.content"]);
        assert!(e.ep_outcomes().iter().all(|(_, o)| o.active == "cpu"));
        for lever in [Lever::PartialDecode, Lever::AnchoredNoise, Lever::NsfPhase0, Lever::F0Cache] {
            assert!(e.effective_levers().levers.get(lever), "{}", lever.as_str());
        }
        assert!(e.warnings().is_empty(), "{:?}", e.warnings());
        let p = *e.plan();
        assert_eq!((p.block_16k, p.partial_decode), (640, true));
        assert!(p.enc_frames >= p.flow_frames);
        // A lever the build resolved on but the plans cannot run is reported off with a note.
        let mut l = levers_all(true);
        l.levers.set(Lever::CudaGraphs, true);
        let e = tiny_engine(&s, l);
        assert!(!e.effective_levers().levers.get(Lever::CudaGraphs));
        assert!(e
            .effective_levers()
            .notes
            .iter()
            .any(|n| n.starts_with("cuda_graphs:")));
    }

    /// The whole engine through the fixed-block stream harness: exact lengths, one shape per call,
    /// finite non-silent output, with every lever on and with every lever off (legacy paths).
    #[test]
    fn engine_streams_through_the_fixed_block_harness() {
        let s = tiny_slot("stream");
        for on in [true, false] {
            let engine = tiny_engine(&s, levers_all(on));
            let (block, context) = (engine.plan().block_16k, engine.plan().context_16k);
            let mut h = VcStreamHarness::new(Box::new(engine), 48_000, 48_000, 1, 1, 1_600, 320, 1_920).unwrap();
            let (hb, hc) = h.fixed_shape().expect("fixed-block mode");
            assert_eq!(hb, block);
            assert!(hc >= context);
            assert_eq!(h.fixed_hold_back_frames(), 0, "device block = model block");
            let input: Vec<f32> = voice_16k(16_000 * 3)
                .iter()
                .flat_map(|&x| [x, x, x])
                .collect();
            let mut out = Vec::new();
            for chunk in input.chunks_exact(1_920) {
                let o = h
                    .push_and_convert(chunk)
                    .unwrap_or_else(|e| panic!("levers {on}: {e}"));
                assert_eq!(o.len(), h.last_owed_frames());
                out.extend(o);
            }
            assert_eq!(out.len(), input.len());
            assert!(out.iter().all(|x| x.is_finite()));
            let loud = out.iter().filter(|x| x.abs() > 1e-4).count();
            assert!(loud > out.len() / 4, "levers {on}: {loud} of {} samples carry audio", out.len());
        }
    }

    /// The levers' legacy paths are what they say: F0 over the whole context and flowdec over the
    /// whole context when off; the new block plus the margins, and the return window, when on.
    #[test]
    fn levers_select_incremental_or_whole_context_work() {
        let s = tiny_slot("levers");
        let audio = voice_16k(16_000 * 2);
        for on in [true, false] {
            let mut e = tiny_engine(&s, levers_all(on));
            // The test signal pauses; every tick must convert here.
            e.set_silence_skip(false);
            let p = *e.plan();
            assert_eq!(p.flow_frames == p.enc_frames, !on);
            let mut pos = 0usize;
            for tick in 0..20 {
                let ctx = ConversionCtx {
                    mono_16k: &audio[pos..pos + p.context_16k],
                    new_frames: p.want_16k,
                    pitch_shift_semitones: 0.0,
                    index_rate: 0.0,
                    protect: 0.33,
                    speaker_id: 0,
                    stream_pos_16k: pos as u64,
                };
                let out = e.convert(&ctx).unwrap();
                assert_eq!(out.len(), p.out_samples());
                let st = e.tick_stats();
                assert!(!st.skipped);
                if on && tick > 0 {
                    let cache = e.f0_cache.config();
                    assert_eq!(
                        st.f0_frames,
                        p.block_16k / 160 + cache.margin_frames + e.f0_cache.lead_frames(),
                        "tick {tick}"
                    );
                } else {
                    assert_eq!(st.f0_frames, p.enc_frames, "tick {tick}");
                }
                pos += p.block_16k;
            }
        }
    }

    /// Silence costs no inference after a short hangover, and speech resumes cleanly; with the
    /// skip off every tick converts.
    #[test]
    fn silence_skips_inference_after_the_hangover() {
        let s = tiny_slot("silence");
        let mut e = tiny_engine(&s, levers_all(true));
        let p = *e.plan();
        let quiet = vec![0.0f32; p.context_16k];
        let loud = voice_16k(p.context_16k);
        let mut skipped = Vec::new();
        for tick in 0..8u64 {
            let buf = if tick < 6 { &quiet } else { &loud };
            let ctx = ConversionCtx {
                mono_16k: buf,
                new_frames: p.want_16k,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: tick * p.block_16k as u64,
            };
            let out = e.convert(&ctx).unwrap();
            assert_eq!(out.len(), p.out_samples());
            skipped.push(e.tick_stats().skipped);
            if e.tick_stats().skipped {
                assert!(out.iter().all(|&x| x == 0.0));
            }
        }
        // 40 ms blocks, 80 ms hangover: two quiet ticks convert, the next four skip, speech converts.
        assert_eq!(skipped, [false, false, true, true, true, true, false, false]);
        e.set_silence_skip(false);
        let ctx = ConversionCtx {
            mono_16k: &quiet,
            new_frames: p.want_16k,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 100 * p.block_16k as u64,
        };
        for _ in 0..5 {
            e.convert(&ctx).unwrap();
            assert!(!e.tick_stats().skipped);
        }
    }

    /// With the skip off, quiet speech (below `silence_db`) is converted *and* heard: the post
    /// stage's silence gate follows the skip (review of P3-C1: it muted it after 150 ms). With the
    /// skip on the same input goes silent after the hangover.
    #[test]
    fn silence_skip_off_keeps_quiet_speech_audible() {
        let s = tiny_slot("quiet");
        // A steady voiced tone at -48 dBFS RMS, below the default -42 dB threshold.
        let rms = 10f32.powf(-48.0 / 20.0);
        let tone: Vec<f32> = (0..16_000 * 2)
            .map(|i| rms * std::f32::consts::SQRT_2 * (std::f32::consts::TAU * 160.0 * i as f32 / 16_000.0).sin())
            .collect();
        for skip in [true, false] {
            let mut e = tiny_engine(&s, levers_all(true));
            e.set_silence_skip(skip);
            let p = *e.plan();
            let mut late_energy = 0.0f64;
            for tick in 0..16usize {
                let pos = tick * p.block_16k;
                let ctx = ConversionCtx {
                    mono_16k: &tone[pos..pos + p.context_16k],
                    new_frames: p.want_16k,
                    pitch_shift_semitones: 0.0,
                    index_rate: 0.0,
                    protect: 0.33,
                    speaker_id: 0,
                    stream_pos_16k: pos as u64,
                };
                let out = e.convert(&ctx).unwrap();
                assert_eq!(out.len(), p.out_samples());
                // 40 ms blocks, 80 ms hangover: two quiet ticks convert, then the skip.
                assert_eq!(e.tick_stats().skipped, skip && tick >= 2, "skip {skip} tick {tick}");
                // 40 ms blocks: from tick 8 on the post gate's 150 ms hangover is long over.
                if tick >= 8 {
                    late_energy += out.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>();
                }
            }
            if skip {
                assert_eq!(late_energy, 0.0);
            } else {
                assert!(late_energy > 1e-9, "quiet speech with the skip off was muted ({late_energy})");
            }
        }
        // Switching the skip back on gates again.
        let mut e = tiny_engine(&s, levers_all(true));
        e.set_silence_skip(false);
        e.set_silence_skip(true);
        assert!(e.silence_skip);
    }

    /// A re-plan rebuilds the generator only for new tensor shapes, and afterwards the report the
    /// factory reads (`stream_report`) carries the outcomes and levers of the sessions that run; a
    /// shape the template cannot build leaves the old sessions running.
    #[test]
    fn replans_rebuild_only_new_shapes_and_refresh_the_report() {
        let s = tiny_slot("replan");
        let mut l = levers_all(true);
        l.levers.set(Lever::CudaGraphs, true);
        let mut e = tiny_engine(&s, l);
        let before = *e.plan();
        assert_eq!(e.generator_builds, 1);
        // Another SOLA tail within the same frames: same sessions.
        let want = before.return_frames * HOP_16K - 1;
        assert!(want > before.want_16k);
        e.configure_stream(640, want, 1_600).unwrap();
        assert_eq!((e.generator_builds, e.plan().want_16k), (1, want));
        assert_eq!(generator_shape(e.plan()), generator_shape(&before));
        // A longer tail: new shapes, one rebuild, a report of what runs now.
        e.configure_stream(640, 640 + 2_000, 1_600).unwrap();
        assert_eq!(e.generator_builds, 2);
        let r = e.stream_report().expect("the engine reports");
        let comps: Vec<&str> = r.outcomes.iter().map(|(c, _)| c.as_str()).collect();
        assert_eq!(comps, ["vc.enc", "vc.flowdec", "vc.content"]);
        assert_eq!(&r.levers, e.effective_levers());
        assert!(!r.levers.levers.get(Lever::CudaGraphs));
        let notes = r.levers.notes.iter().filter(|n| n.starts_with("cuda_graphs:")).count();
        assert_eq!(notes, 1, "{:?}", r.levers.notes);
        let p = *e.plan();
        let audio = voice_16k(p.context_16k);
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: p.want_16k,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
        };
        assert_eq!(e.convert(&ctx).unwrap().len(), p.out_samples());
        // A plan at another output rate cannot be built for this template: the error is returned
        // and the old sessions keep running at the old plan.
        let streaming = e.template.layout.streaming.clone();
        let bad = ShapePlan::for_stream(&streaming, e.output_sample_rate + 100, true, 640, p.want_16k, 1_600).unwrap();
        assert!(e.replan(bad).is_err());
        assert_eq!(*e.plan(), p);
        assert!(e.generator.is_some());
        assert_eq!(e.convert(&ctx).unwrap().len(), p.out_samples());
    }

    /// The harness's real SOLA tail and extra context resize the plan (and rebuild the sessions)
    /// when they differ from the preset's defaults; a block other than the plan's is refused.
    #[test]
    fn configure_stream_replans_for_the_harness_shape() {
        let s = tiny_slot("configure");
        let mut e = tiny_engine(&s, levers_all(true));
        let before = *e.plan();
        let same = e.configure_stream(640, before.want_16k, 1_600).unwrap();
        assert_eq!(same, before.context_16k);
        assert_eq!(*e.plan(), before, "an unchanged shape does not rebuild");
        let longer = e.configure_stream(640, 640 + 2_000, 4_800).unwrap();
        let after = *e.plan();
        assert_eq!(longer, after.context_16k);
        assert!(after.return_frames > before.return_frames);
        assert!(after.context_16k >= 4_800 + 640);
        let audio = voice_16k(after.context_16k);
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: after.want_16k,
            pitch_shift_semitones: 2.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 2,
            stream_pos_16k: 0,
        };
        assert_eq!(e.convert(&ctx).unwrap().len(), after.out_samples());
        assert!(e.configure_stream(320, 1_000, 0).is_err());
        // A one-off conversion at another shape is cut / padded to the plan (fewer samples at most).
        let short = voice_16k(3_000);
        let ctx = ConversionCtx {
            mono_16k: &short,
            new_frames: 1_000,
            ..ctx
        };
        let out = e.convert(&ctx).unwrap();
        assert_eq!(out.len(), (1_000 * 800 + 8_000) / 16_000);
    }

    /// Lookahead grows the decoded span by its frames and returns the same number of samples,
    /// through the harness too (the harness sizes its context from the replanned engine).
    #[test]
    fn lookahead_decodes_future_context_and_keeps_lengths() {
        let s = tiny_slot("lookahead");
        let mut e = tiny_engine(&s, levers_all(true));
        let base = *e.plan();
        assert_eq!(base.lookahead_frames, 0);
        assert!(e.set_lookahead_ms(90.0).is_err(), "at most 80 ms");
        e.set_lookahead_ms(20.0).unwrap();
        let p = *e.plan();
        assert_eq!(p.lookahead_frames, 2);
        assert_eq!(p.return_frames, base.return_frames + 2);
        assert_eq!(p.out_samples(), base.out_samples());
        let audio = voice_16k(p.context_16k);
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: p.want_16k,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
        };
        assert_eq!(e.convert(&ctx).unwrap().len(), p.out_samples());
        let mut h = VcStreamHarness::new(Box::new(e), 48_000, 48_000, 1, 1, 1_600, 320, 1_920).unwrap();
        let input: Vec<f32> = voice_16k(16_000).iter().flat_map(|&x| [x, x, x]).collect();
        let mut total = 0;
        for chunk in input.chunks_exact(1_920) {
            total += h.push_and_convert(chunk).unwrap().len();
        }
        assert_eq!(total, input.len());
    }

    /// Speaker blends, the governor's retrieval step (no index here: nothing to apply) and stream
    /// resets.
    #[test]
    fn speaker_blend_quality_steps_and_resets() {
        let s = tiny_slot("blend");
        let mut e = tiny_engine(&s, levers_all(true));
        assert!(e.set_speaker_blend(Some(&[(0, 0.0)])).is_err());
        e.set_speaker_blend(Some(&[(0, 1.0), (2, 1.0)])).unwrap();
        let p = *e.plan();
        let audio = voice_16k(p.context_16k);
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: p.want_16k,
            pitch_shift_semitones: 0.0,
            index_rate: 0.5,
            protect: 0.2,
            speaker_id: 1,
            stream_pos_16k: 0,
        };
        let blended = e.convert(&ctx).unwrap();
        e.set_speaker_blend(None).unwrap();
        e.reset_stream();
        let single = e.convert(&ctx).unwrap();
        assert_eq!(blended.len(), single.len());
        assert_ne!(blended, single);
        let off = QualityLevel {
            level: 1,
            retrieval: false,
            ..QualityLevel::FULL
        };
        assert!(!e.apply_quality(&off), "no index: the step does nothing here");
        e.set_voice_controls(&VoiceControls {
            noise_scale: 0.0,
            ..VoiceControls::default()
        });
        e.reset_stream();
        let quiet_noise = e.convert(&ctx).unwrap();
        assert!(quiet_noise.iter().all(|x| x.is_finite()));
    }

    /// Slots that do not match their template are refused with a reason, before any session runs.
    #[test]
    fn mismatched_slots_are_refused() {
        let s = tiny_slot("refuse");
        let load = |s: &TinySlot| SigilRvcEngine::load_with(request(s, levers_all(true), 40.0, 100.0), LoadOptions { allow_any_frame_rate: true });
        // An import record from another export of the template.
        let record = s.slot.join("sigil-import.json");
        let original = std::fs::read(&record).unwrap();
        let mut v: serde_json::Value = serde_json::from_slice(&original).unwrap();
        v["weights_id"] = serde_json::json!("0".repeat(64));
        std::fs::write(&record, serde_json::to_vec(&v).unwrap()).unwrap();
        let err = load(&s).err().unwrap();
        assert!(err.contains("import the voice again"), "{err}");
        std::fs::write(&record, &original).unwrap();
        // A truncated weights file.
        let weights = s.slot.join("weights.f32.bin");
        let full = std::fs::read(&weights).unwrap();
        std::fs::write(&weights, &full[..full.len() - 64]).unwrap();
        let err = load(&s).err().unwrap();
        assert!(err.contains("bytes"), "{err}");
        std::fs::write(&weights, &full).unwrap();
        // A file name that leaves the slot.
        let mut m = load_slot_manifest(&s.slot).unwrap();
        m.weights = Some("../weights.f32.bin".into());
        let mut req = request(&s, levers_all(true), 40.0, 100.0);
        req.manifest = m;
        let err = SigilRvcEngine::load_with(req, LoadOptions { allow_any_frame_rate: true })
            .err()
            .unwrap();
        assert!(err.contains("plain file name"), "{err}");
        // An embedder Sigil does not have.
        let mut req = request(&s, levers_all(true), 40.0, 100.0);
        req.manifest.embedder = Some("wavlm".into());
        let err = SigilRvcEngine::load_with(req, LoadOptions { allow_any_frame_rate: true })
            .err()
            .unwrap();
        assert!(err.contains("wavlm"), "{err}");
        // And the intact slot still loads.
        assert!(load(&s).is_ok());
    }

    // ---- dev-rig measurements (ignored; need the MIT default voice) ---------------------------

    /// Spectral flux (L1 of the positive magnitude change between frames) of `x` with a `win`-sample
    /// Hann window every `hop` samples: block-boundary clicks show up as spikes.
    fn spectral_flux(x: &[f32], win: usize, hop: usize) -> Vec<f64> {
        use realfft::RealFftPlanner;
        let fft = RealFftPlanner::<f32>::new().plan_fft_forward(win);
        let (mut buf, mut spec) = (fft.make_input_vec(), fft.make_output_vec());
        let hann: Vec<f32> = (0..win)
            .map(|i| 0.5 - 0.5 * (std::f32::consts::TAU * i as f32 / win as f32).cos())
            .collect();
        let mut prev: Vec<f32> = vec![0.0; spec.len()];
        let mut flux = Vec::new();
        let mut i = 0;
        while i + win <= x.len() {
            for (k, b) in buf.iter_mut().enumerate() {
                *b = x[i + k] * hann[k];
            }
            fft.process(&mut buf, &mut spec).unwrap();
            let mut f = 0.0f64;
            for (p, c) in prev.iter_mut().zip(&spec) {
                let m = c.norm();
                f += f64::from((m - *p).max(0.0));
                *p = m;
            }
            flux.push(f);
            i += hop;
        }
        flux
    }

    /// Renders a speech clip through the default voice with the fixed-block harness, all levers on
    /// and all four legacy paths, writes both WAVs for listening, and prints the spectral-flux
    /// spike statistics (a click detector: the ratio of the 99.9th flux percentile to the median).
    /// `SIGIL_TEST_MODELS_DIR`, `SIGIL_TEST_CONTENT_ENCODER` (a ContentVec 768 ONNX; required),
    /// optional `SIGIL_SVE_CLIP` (16 kHz mono WAV; default the dev cache's
    /// `sherpa-onnx/clips/en.wav`), `SIGIL_SVE_RENDER_DIR` (where the WAVs go), `SIGIL_BENCH_EPS`
    /// (first EP used, default `dml`).
    #[test]
    #[ignore = "dev-rig listening render: needs the default voice and a ContentVec export"]
    fn sve_render_default_voice() {
        let (Some(models), Some(encoder)) = (
            std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from),
            std::env::var_os("SIGIL_TEST_CONTENT_ENCODER").map(PathBuf::from),
        ) else {
            eprintln!("SIGIL_TEST_MODELS_DIR / SIGIL_TEST_CONTENT_ENCODER not set; skipped");
            return;
        };
        let clip = std::env::var_os("SIGIL_SVE_CLIP")
            .map(PathBuf::from)
            .unwrap_or_else(|| models.join("sherpa-onnx").join("clips").join("en.wav"));
        let mut reader = hound::WavReader::open(&clip).expect("clip");
        let spec = reader.spec();
        assert_eq!((spec.sample_rate, spec.channels), (16_000, 1), "a 16 kHz mono clip");
        let input: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => reader
                .samples::<i32>()
                .map(|s| s.unwrap() as f32 / (1u64 << (spec.bits_per_sample - 1)) as f32)
                .collect(),
            hound::SampleFormat::Float => reader.samples::<f32>().map(Result::unwrap).collect(),
        };
        let templates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("vc-templates");
        let voice =
            crate::services::neural_audio::vc::import::import_file(&models.join("rvc-default-voice").join("default.pth"), &templates).unwrap();
        let root = std::env::var_os("SIGIL_SVE_RENDER_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::temp_dir().join(format!("sigil-sve-render-{}", std::process::id())));
        let slot = root.join("slot");
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::write(slot.join("weights.f32.bin"), &voice.weights.f32_image).unwrap();
        let speakers: Vec<u8> = voice
            .weights
            .speakers
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(slot.join("speakers.f32.bin"), speakers).unwrap();
        let record = serde_json::json!({"template": voice.template.layout.template, "weights_id": voice.template.layout.weights.weights_id});
        std::fs::write(slot.join("sigil-import.json"), serde_json::to_vec(&record).unwrap()).unwrap();
        let manifest = serde_json::json!({"format": "sigil-rvc-1", "template": voice.template.layout.template, "weights": "weights.f32.bin", "speakers": "speakers.f32.bin", "embedder": "contentvec"});
        std::fs::write(slot.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let ep = std::env::var("SIGIL_BENCH_EPS")
            .ok()
            .and_then(|v| v.split(',').next().map(str::to_string))
            .unwrap_or_else(|| "dml".into());
        let all = [Lever::PartialDecode, Lever::AnchoredNoise, Lever::NsfPhase0, Lever::F0Cache];
        let variants: [(&str, &[Lever], f64); 9] = [
            ("levers-on", &all, 0.0),
            ("levers-legacy", &[], 0.0),
            ("no-anchored-noise", &[Lever::PartialDecode, Lever::NsfPhase0, Lever::F0Cache], 0.0),
            ("no-phase0", &[Lever::PartialDecode, Lever::AnchoredNoise, Lever::F0Cache], 0.0),
            ("no-f0-cache", &[Lever::PartialDecode, Lever::AnchoredNoise, Lever::NsfPhase0], 0.0),
            ("levers-on-lookahead10", &all, 10.0),
            ("levers-on-lookahead20", &all, 20.0),
            ("levers-on-lookahead40", &all, 40.0),
            ("levers-legacy-lookahead20", &[], 20.0),
        ];
        for (label, on, lookahead_ms) in variants {
            let mut levers = EffectiveLevers::default();
            for &lever in on {
                levers.levers.set(lever, true);
            }
            let req = SigilRvcRequest {
                slot_dir: slot.clone(),
                manifest: load_slot_manifest(&slot).unwrap(),
                templates_dir: templates.clone(),
                ep: ep.clone(),
                session: VcSessionOptions::default(),
                levers,
                pitch: Box::new(DioPitchExtractor::new()),
                content_encoder: Some(encoder.clone()),
                index: None,
                preset: presets::engine_preset(RvcPresetProfile::LowLatency, 40.0, 0, 16_000, 96.0, PrecisionMode::Auto),
                f0_range: F0Range::Speech,
                noise_seed: 3,
            };
            let mut engine = SigilRvcEngine::load(req).unwrap();
            engine.set_silence_skip(false);
            engine.set_lookahead_ms(lookahead_ms).unwrap();
            let block = engine.plan().block_16k;
            let mut h = VcStreamHarness::new(Box::new(engine), 16_000, 48_000, 1, 1, 1_536, 960, block).unwrap();
            h.set_controls(4.0, 0.0, 0.33, 0);
            let mut out = Vec::new();
            let mut joins: Vec<(isize, f32)> = Vec::new();
            for chunk in input.chunks_exact(block) {
                out.extend(h.push_and_convert(chunk).unwrap());
                joins.extend(h.last_join());
            }
            let mut nccs: Vec<f64> = joins.iter().map(|&(_, c)| f64::from(c)).collect();
            let centred = joins.iter().filter(|&&(o, _)| o == 0).count() as f64 / joins.len().max(1) as f64;
            let mean_shift = joins
                .iter()
                .map(|&(o, _)| o.unsigned_abs() as f64)
                .sum::<f64>()
                / joins.len().max(1) as f64;
            let mean_ncc = nccs.iter().sum::<f64>() / nccs.len().max(1) as f64;
            let p10_ncc = percentile(&mut nccs, 0.1);
            let path = root.join(format!("{label}.wav"));
            let mut w = hound::WavWriter::create(
                &path,
                hound::WavSpec {
                    channels: 1,
                    sample_rate: 48_000,
                    bits_per_sample: 32,
                    sample_format: hound::SampleFormat::Float,
                },
            )
            .unwrap();
            for &s in &out {
                w.write_sample(s).unwrap();
            }
            w.finalize().unwrap();
            // Seams: the harness joins model blocks every `block` input samples (1920 output
            // frames here). Flux over 2.5 ms hops folded onto the 16 phases of that period: speech
            // spreads evenly over the phases, clicks at the joins pile up on one.
            let period = block * 3;
            let hop = period / 16;
            let flux = spectral_flux(&out, 4 * hop, hop);
            let mut phases = vec![(0.0f64, 0usize); 16];
            for (i, &f) in flux.iter().enumerate() {
                let p = &mut phases[i % 16];
                p.0 += f;
                p.1 += 1;
            }
            let mut means: Vec<f64> = phases.iter().map(|&(s, n)| s / n.max(1) as f64).collect();
            let top = means.iter().cloned().fold(0.0, f64::max);
            let median = percentile(&mut means, 0.5);
            let rms = (out
                .iter()
                .map(|&x| f64::from(x) * f64::from(x))
                .sum::<f64>()
                / out.len() as f64)
                .sqrt();
            eprintln!(
                "{ep} {label}: {:.2} s, rms {rms:.4}, seam flux (max phase / median phase) {:.3}, joins {}: NCC mean {mean_ncc:.3} p10 {p10_ncc:.3}, centred {:.0} %, mean shift {mean_shift:.1} samples → {}",
                out.len() as f64 / 48_000.0,
                top / median.max(1e-12),
                joins.len(),
                centred * 100.0,
                path.display()
            );
        }
    }

    fn percentile(v: &mut [f64], p: f64) -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        if v.is_empty() {
            return f64::NAN;
        }
        let k = ((v.len() - 1) as f64 * p).round() as usize;
        v[k]
    }

    /// Per-block compute of the engine on the MIT default voice (v2 40k, bundled template) for
    /// growing extra context, with each algorithmic lever on and off (plan T2: "growing the extra
    /// context from 64 ms to 512 ms raises p95 by ≤ 30 %"). Prints a table; asserts nothing about
    /// speed (the dev rig is shared).
    ///
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache` (holds `rvc-default-voice/default.pth`),
    /// optional `SIGIL_TEST_CONTENT_ENCODER=<ContentVec 768 ONNX>` (else a 400/320 conv stand-in,
    /// flagged in the output), `SIGIL_BENCH_EPS=cpu,dml,cuda` (CUDA also needs
    /// `SIGIL_TEST_CUDA_BIN` / `SIGIL_TEST_CUDNN_BIN`), `SIGIL_BENCH_TICKS` (default 150):
    /// `cargo test --manifest-path src-tauri/Cargo.toml sve_context_scaling_bench -- --ignored --nocapture`
    #[test]
    #[ignore = "dev-rig benchmark: needs SIGIL_TEST_MODELS_DIR with rvc-default-voice/default.pth"]
    fn sve_context_scaling_bench() {
        let Some(models) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR is not set; skipped");
            return;
        };
        let pth = models.join("rvc-default-voice").join("default.pth");
        let templates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("vc-templates");
        let voice = crate::services::neural_audio::vc::import::import_file(&pth, &templates).expect("default voice imports");
        let root = std::env::temp_dir().join(format!("sigil-sve-bench-{}", std::process::id()));
        let slot = root.join("voice-changer").join("models").join("default");
        std::fs::create_dir_all(&slot).unwrap();
        std::fs::write(slot.join("weights.f32.bin"), &voice.weights.f32_image).unwrap();
        let speakers: Vec<u8> = voice
            .weights
            .speakers
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        std::fs::write(slot.join("speakers.f32.bin"), speakers).unwrap();
        let record = serde_json::json!({"template": voice.template.layout.template, "weights_id": voice.template.layout.weights.weights_id});
        std::fs::write(slot.join("sigil-import.json"), serde_json::to_vec(&record).unwrap()).unwrap();
        let manifest = serde_json::json!({
            "format": "sigil-rvc-1", "template": voice.template.layout.template, "weights": "weights.f32.bin",
            "speakers": "speakers.f32.bin", "embedder": "contentvec", "fp16_ok": true
        });
        std::fs::write(slot.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        // A ContentVec export folder (`model.<variant>.onnx`, P3-A's layout): each EP gets the
        // variant the pool would pick; else one explicit file; else the conv stand-in.
        let encoder_dir = std::env::var_os("SIGIL_TEST_CONTENT_ENCODER_DIR").map(PathBuf::from);
        let (fixed_encoder, encoder_label) = match (&encoder_dir, std::env::var_os("SIGIL_TEST_CONTENT_ENCODER").map(PathBuf::from)) {
            (Some(_), _) => (None, "ContentVec (per-EP variant)"),
            (None, Some(p)) => (Some(p), "ContentVec"),
            (None, None) => {
                let p = root.join("standin-content.onnx");
                std::fs::write(&p, test_models::content_encoder_bytes()).unwrap();
                (Some(p), "stand-in conv (not ContentVec)")
            }
        };
        let encoder_for = |ep: &str| -> (PathBuf, &'static str) {
            if let Some(p) = &fixed_encoder {
                return (p.clone(), "file");
            }
            let dir = encoder_dir.as_ref().unwrap();
            for v in content::variant_order(
                ep != "cpu",
                PrecisionMode::Auto,
                crate::services::neural_audio::vc::int8_gate::int8_content_encoder_enabled(),
            ) {
                let p = dir.join(crate::services::neural_audio::vc::assets::file_name(v));
                if p.is_file() {
                    return (p, v.unwrap_or("model"));
                }
            }
            panic!("no content encoder variant in {}", dir.display());
        };
        let cuda_ok = crate::services::neural_audio::ort_session::preload_test_cuda_runtime();
        let eps: Vec<String> = std::env::var("SIGIL_BENCH_EPS")
            .unwrap_or_else(|_| "cpu".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty() && (s != "cuda" || cuda_ok))
            .collect();
        let ticks: usize = std::env::var("SIGIL_BENCH_TICKS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(150);
        let audio = voice_16k(16_000 * 30);
        eprintln!("SVE per-block compute, default voice v2 40k, 40 ms block, dio F0, content encoder: {encoder_label}, {ticks} ticks");
        eprintln!("| EP | levers | extra ms | ctx frames | flowdec F | precision | p50 ms | p95 ms | content | f0 | enc | flowdec | f0 frames |");
        eprintln!("|---|---|---|---|---|---|---|---|---|---|---|---|---|");
        let extras: Vec<f64> = std::env::var("SIGIL_BENCH_EXTRA_MS")
            .ok()
            .map(|v| v.split(',').filter_map(|x| x.trim().parse().ok()).collect())
            .unwrap_or_else(|| vec![64.0, 192.0, 512.0, 1_000.0, 2_000.0]);
        for ep in &eps {
            let (encoder, variant) = encoder_for(ep);
            eprintln!("  {ep}: content encoder {} ({variant})", encoder.display());
            for (label, partial, f0c, precision) in [
                ("all on", true, true, PrecisionMode::Auto),
                ("all on fp32", true, true, PrecisionMode::Fp32),
                ("partial off", false, true, PrecisionMode::Auto),
                ("f0 cache off", true, false, PrecisionMode::Auto),
            ] {
                if precision == PrecisionMode::Fp32 && ep == "cpu" {
                    continue;
                }
                for &extra in &extras {
                    let mut levers = EffectiveLevers {
                        precision,
                        ..EffectiveLevers::default()
                    };
                    for lever in [Lever::AnchoredNoise, Lever::NsfPhase0] {
                        levers.levers.set(lever, true);
                    }
                    levers.levers.set(Lever::PartialDecode, partial);
                    levers.levers.set(Lever::F0Cache, f0c);
                    let req = SigilRvcRequest {
                        slot_dir: slot.clone(),
                        manifest: load_slot_manifest(&slot).unwrap(),
                        templates_dir: templates.clone(),
                        ep: ep.clone(),
                        session: VcSessionOptions {
                            intra_threads: Some(2),
                            component_prefix: Some("bench"),
                            ..VcSessionOptions::default()
                        },
                        levers,
                        pitch: Box::new(DioPitchExtractor::new()),
                        content_encoder: Some(encoder.clone()),
                        index: None,
                        preset: presets::engine_preset(RvcPresetProfile::LowLatency, 40.0, 0, 48_000, extra, PrecisionMode::Auto),
                        f0_range: F0Range::Speech,
                        noise_seed: 7,
                    };
                    let mut e = SigilRvcEngine::load(req).expect("engine loads");
                    e.set_silence_skip(false);
                    let p = *e.plan();
                    let (mut total, mut content, mut f0, mut enc, mut fd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
                    let mut f0_frames = 0;
                    for tick in 0..ticks + 20 {
                        let pos = (tick * p.block_16k) % (audio.len() - p.context_16k);
                        let pos = pos - pos % p.block_16k;
                        let ctx = ConversionCtx {
                            mono_16k: &audio[pos..pos + p.context_16k],
                            new_frames: p.want_16k,
                            pitch_shift_semitones: 0.0,
                            index_rate: 0.0,
                            protect: 0.33,
                            speaker_id: 0,
                            stream_pos_16k: (tick * p.block_16k) as u64,
                        };
                        let started = Instant::now();
                        e.convert(&ctx).unwrap();
                        let ms = started.elapsed().as_secs_f64() * 1000.0;
                        if tick >= 20 {
                            let st = e.tick_stats();
                            total.push(ms);
                            content.push(st.content_ms);
                            f0.push(st.f0_ms);
                            enc.push(st.enc_ms);
                            fd.push(st.flowdec_ms);
                            f0_frames = st.f0_frames;
                        }
                    }
                    let outcome = e
                        .ep_outcomes()
                        .iter()
                        .find(|(c, _)| c == "vc.flowdec")
                        .map(|(_, o)| o.clone())
                        .expect("flowdec outcome");
                    if let Some(why) = &outcome.reason {
                        eprintln!("  ({ep} fell back: {why})");
                    }
                    let active = outcome.active;
                    eprintln!(
                        "| {ep}→{active} | {label} | {extra} | {} | {} | {} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {f0_frames} |",
                        p.enc_frames,
                        p.flow_frames,
                        e.flowdec_precision(),
                        percentile(&mut total.clone(), 0.5),
                        percentile(&mut total, 0.95),
                        percentile(&mut content, 0.5),
                        percentile(&mut f0, 0.5),
                        percentile(&mut enc, 0.5),
                        percentile(&mut fd, 0.5),
                    );
                }
            }
        }
        let _ = std::fs::remove_dir_all(&root);
    }
}
