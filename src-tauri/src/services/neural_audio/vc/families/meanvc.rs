//! MeanVC2 CPU zero-shot voices (experimental; P5-C).
//!
//! MeanVC2 (ASLP-lab, 40 ms chunk + 40 ms future model, 16 kHz, Apache-2.0) converts speech toward
//! any speaker from one embedding of a reference clip: a streaming ASR encoder (Fast-U2++) turns
//! the input into content features, a 4-layer DiT with mean flows (1 or 2 steps) generates the
//! target speaker's mel frames, and Vocos turns them into audio. Sigil runs its own export of the
//! graphs (`scripts/vc-families/meanvc/export.py`, ADR 0006) on ONNX Runtime's CPU EP at one thread,
//! behind the shared stream harness (budget, resampling, TTS mixing, metrics).
//!
//! # Pieces (child modules under `meanvc/`)
//! - `graphs`: the five graphs (`fbank`, `asr`, `gtm`, `dit_step`, `vocos`), their I/O contract and
//!   the ONNX Runtime sessions behind the `MeanVcGraphs` trait.
//! - `stream`: the streaming loop, upstream `run_rt.py` `VCRunner` state for state (with the 4k + 1
//!   BN generalisation, so it ticks every 80 ms, a 45 ms phase lead and anchored noise).
//! - `model`: the `VoiceConversionModel` (fixed blocks, an output history that keeps joins
//!   seamless, steady-silence start, silence skip, hot reference switch, governor steps, and the
//!   engine's post stage for `rms_mix`, the formant shift and the limiter).
//! - `speaker`, `npy`: the speaker embedding (a voice folder's `speaker.npy`, the voice library's
//!   cache, or the speaker encoder graph over the reference clip).
//!
//! # A voice
//! A slot is a manifest (`manifest.format` [`super::MEANVC_FORMAT`], options [`MeanVcOptions`] in
//! `manifest.family`) made from a voice-library reference (`reference_voice_id` in `slot.json`).
//! Its graphs come from the model store ([`CATALOG_ID`], the family's only hard requirement); a
//! voice folder holding the five graph files runs those instead (a developer's own export). Its
//! speaker comes from `speaker.rs`: the speaker encoder ([`SPEAKER_ID`]) is optional (a voice
//! folder's `speaker.npy` or a cached embedding does without it), so only a load that needs it
//! names it.
//!
//! # Execution provider
//! The voice runs on the CPU whatever rung the voice changer uses (the family lists every rung, so a
//! GPU setup keeps its rung for its other voices; a lever note says so): one thread runs a 40 ms
//! output chunk in 4.2 ms with 1 step and 5.9 ms with 2 steps on the dev rig (RTF 0.11 / 0.15, ADR
//! 0006), a second thread does not help, and it uses no VRAM. CUDA is faster only with CUDA graphs
//! on the DiT and Vocos (RTF 0.09), which is not worth a GPU context next to a game.
//!
//! # Latency
//! The model delays its content by about [`MODEL_DELAY_16K`] samples (115 ms) on top of the tick
//! (80 or 160 ms) and the harness's SOLA: the 40 ms future block of the DiT, the vocoder's 10 ms
//! overlap, the ASR's chunking and the alignment the DiT learned (measured, see the constant). The
//! 80 ms tick runs as two 40 ms harness blocks with its work spread over both
//! ([`MeanVcOptions::block_16k`]), which keeps the latency and halves the compute per call. The
//! benchmark's algorithmic latency is about 225 ms at the 80 ms tick and 300 ms at 160 ms (P5-C).
//! The 160 ms tick computes in one burst per tick (33-43 ms idle, 62 ms p99 under a game load), more
//! than the voice changer's default 35 ms output cushion, so it declares the Stable output buffer
//! ([`min_output_buffer_ms`], 80 / 150 ms).
//!
//! # Frozen by P5-S0 (signatures; P5-C implements; see `families` for the contract)
//! [`available`], [`info`], [`create`], [`check_slot`], [`normalize_options`], [`create_slot`],
//! [`run_import_job`], [`MeanVcOptions`] (the `manifest.family` object; TS `VcMeanVcOptions`).
//! [`min_output_buffer_ms`] (P5-C fix) is the family's per-voice output-buffer minimum for
//! `families::min_output_buffer_ms`.

mod graphs;
mod model;
mod npy;
mod speaker;
mod stream;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use super::reference::{cached_embedding, ReferenceEmbedding};
use super::{CreatedSlot, FamilyEngine, FamilyInfo, FamilyKind, FamilyRequest, FamilySlotContext, FamilySlotRequest, CREATION_REFERENCE};
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::engine::options::{Accelerator, EffectiveLevers, F0Range, Lever, PrecisionMode};
use crate::services::neural_audio::vc::import::ImportJob;
use crate::services::neural_audio::vc::manifest::RvcManifest;
use graphs::{GraphSource, OrtGraphs};
use model::{MeanVcModel, ModelConfig};
use stream::MeanVcStream;

/// Model-store entry of the five graphs (`model-catalog/vc-meanvc.json`; a local export).
pub const CATALOG_ID: &str = "vc/meanvc2";
/// Model-store entry of the speaker encoder (`model.onnx`; a local export of the manually
/// downloaded WavLM-Large + ECAPA-TDNN weights).
pub const SPEAKER_ID: &str = "vc/meanvc2-speaker";
/// How far the converted content lags the input (16 kHz samples), on top of the harness's block
/// and SOLA, when a whole tick runs per call. Measured on the dev rig as the spectral-flux
/// cross-correlation peak between input and output over four speech clips (en, ja, ko, ru;
/// 1720-1900 samples, mean 1834) with whole 80 ms ticks; the loop's structure alone accounts for
/// 1480 (the DiT's 640-sample future, the vocoder's 160-sample overlap, the ASR chunk and the BN
/// upsampling), the rest is the alignment the DiT learned. The same at 80 and 160 ms ticks (the
/// tick adds to the harness's block instead). A split tick adds its deferred half
/// ([`MeanVcOptions::model_delay_16k`]).
pub const MODEL_DELAY_16K: usize = 1840;
/// The output buffer (prefill, max queue; ms) a voice whose blocks compute a whole 160 ms tick plays
/// with at least: the P5-C bench's p99 call under the H3 game load (62.4 ms) plus 15 ms, rounded up
/// to 5 ms (the rule of `sigil_zs::tune::output_buffer_ms`), which is the Stable template's 80 / 150.
/// At the default 35 / 70 ms it underran 265 times per 20 s idle and 162 under H3.
pub const WHOLE_TICK_OUTPUT_BUFFER_MS: (f64, f64) = (80.0, 150.0);

/// The family's options (`manifest.family`, snake_case).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct MeanVcOptions {
    /// Mean-flow steps: 1 or 2 (default 2).
    pub steps: u32,
    /// Tick length in ms: 80 or 160 (default 80).
    pub tick_ms: u32,
}

impl Default for MeanVcOptions {
    fn default() -> Self {
        MeanVcOptions { steps: 2, tick_ms: 80 }
    }
}

impl MeanVcOptions {
    /// The tick in 16 kHz samples.
    pub fn tick_16k(&self) -> usize {
        self.tick_ms as usize * 16
    }

    /// New 16 kHz samples per call (the harness's fixed block). The 80 ms tick runs as two 40 ms
    /// blocks with its work spread over both (`model.rs`, `stream.rs`): the same latency (40 ms of
    /// block, 40 ms more model delay) for about half the compute per call, which keeps each call
    /// well inside the output queue's cushion on a busy CPU (P5-C bench: the whole-tick call
    /// underran the default 35 ms cushion under the H3 game load). The 160 ms tick runs whole:
    /// fewer, larger calls, so it plays with a larger output buffer ([`Self::min_output_buffer_ms`]).
    pub fn block_16k(&self) -> usize {
        if self.tick_ms <= 80 {
            self.tick_16k() / 2
        } else {
            self.tick_16k()
        }
    }

    /// The model's own delay for this tick: [`MODEL_DELAY_16K`] plus the part of the tick the
    /// block does not cover (the deferred half of a split tick).
    pub fn model_delay_16k(&self) -> usize {
        MODEL_DELAY_16K + self.tick_16k().saturating_sub(self.block_16k())
    }

    /// The smallest output buffer this voice plays with (`None`: the user's): a block that carries
    /// more than one 40 ms DiT call computes them in one burst (the 160 ms tick, about 4× the split
    /// tick's per-call work), so it needs [`WHOLE_TICK_OUTPUT_BUFFER_MS`]. The split 80 ms tick ran
    /// clean at the default buffer (0 underruns paced, 3 per 20 s under H3).
    pub fn min_output_buffer_ms(&self) -> Option<(f64, f64)> {
        (self.block_16k() > stream::DIT_HOP_16K).then_some(WHOLE_TICK_OUTPUT_BUFFER_MS)
    }
}

/// The smallest output buffer (prefill, max queue; ms) a voice with this manifest plays with, for
/// `families::min_output_buffer_ms` (the voice changer and the benchmark's paced runs use the larger
/// of it and the user's): the Stable 80 / 150 ms for the 160 ms tick, `None` for the default 80 ms
/// tick. Options that do not parse are refused at load, so they ask for nothing here.
pub fn min_output_buffer_ms(manifest: &RvcManifest) -> Option<(f64, f64)> {
    options_of(manifest).ok()?.min_output_buffer_ms()
}

/// Whether this build runs MeanVC2 voices.
pub fn available() -> bool {
    true
}

pub fn info() -> FamilyInfo {
    FamilyInfo {
        family: "meanvc",
        format: super::MEANVC_FORMAT,
        experimental: true,
        zero_shot: true,
        singing: false,
        // It runs on the CPU on every rung (module docs).
        execution_providers: &["cpu", "cuda", "dml"],
        needs_pitch: false,
        needs_content_encoder: false,
        creation: CREATION_REFERENCE,
        import_extensions: &[],
        // The post stage's loudness and formant controls, and the silence skip's threshold.
        controls: &["formant", "rms_mix", "silence_db"],
        // Only the graphs: the speaker encoder (SPEAKER_ID) is optional (a voice folder's
        // speaker.npy or a cached embedding does without it; its weights are a manual download), so
        // listing it would keep the requirement banner up on voices that work. A load that needs it
        // names it (`missing_encoder`).
        requires: &[CATALOG_ID],
    }
}

/// The voice's options from its manifest (normalized).
fn options_of(manifest: &RvcManifest) -> Result<MeanVcOptions, String> {
    let normalized = normalize_options(&super::options_of(manifest))?;
    serde_json::from_value(normalized).map_err(|e| format!("invalid MeanVC2 options: {e}"))
}

fn missing_graphs() -> String {
    format!(
        "MeanVC2 voices need their graphs ({CATALOG_ID}, a local export of scripts/vc-families/meanvc/export.py): import them under Models & Runtimes"
    )
}

/// Where a voice's graphs are: its folder when it holds the whole set, else the model store's
/// [`CATALOG_ID`].
fn graphs_dir(slot_dir: &Path) -> Result<PathBuf, String> {
    if graphs::missing_files(slot_dir).is_empty() {
        return Ok(slot_dir.to_path_buf());
    }
    let dir = crate::services::model_store::installed_path(CATALOG_ID).ok_or_else(missing_graphs)?;
    let missing = graphs::missing_files(&dir);
    if !missing.is_empty() {
        return Err(format!(
            "the MeanVC2 graphs in {} are incomplete (no {}): import {CATALOG_ID} again",
            dir.display(),
            missing.join(", ")
        ));
    }
    Ok(dir)
}

/// Sessions of the real-time loop: the CPU at one thread (ADR 0006: a second thread is slower on
/// these small graphs), with the worker's thread policy and the caller's diagnostics prefix.
fn realtime_session_options(base: &VcSessionOptions) -> VcSessionOptions {
    VcSessionOptions {
        cuda_device_index: 0,
        streamer_profile: true,
        intra_threads: Some(1),
        thread_priority: base.thread_priority,
        vram_cap_mb: None,
        component_prefix: base.component_prefix,
        cpu_arena: false,
        dynamic_block_base: None,
    }
}

/// The speaker encoder's one-off session (on the loader thread; background sessions use at most
/// two threads).
fn background_session_options(base: &VcSessionOptions) -> VcSessionOptions {
    VcSessionOptions {
        cuda_device_index: 0,
        streamer_profile: false,
        intra_threads: Some(2),
        thread_priority: None,
        vram_cap_mb: None,
        component_prefix: base.component_prefix,
        cpu_arena: true,
        dynamic_block_base: None,
    }
}

fn missing_encoder() -> String {
    format!(
        "MeanVC2 needs its speaker encoder ({SPEAKER_ID}) to analyse a reference voice: its fine-tuned weights are a manual download, \
         so import a local export under Models & Runtimes, or put a pre-extracted speaker.npy in the voice's folder"
    )
}

/// The speaker embedding of a voice (see `speaker.rs` for the order).
fn resolve_speaker(request: &FamilyRequest, warnings: &mut Vec<String>) -> Result<ReferenceEmbedding, String> {
    let voice_id = request
        .reference
        .as_ref()
        .map(|c| c.voice_id.as_str())
        .unwrap_or_default();
    if let Some(npy) = speaker::slot_npy(&request.slot_dir) {
        let embedding = speaker::from_npy(&npy, voice_id)?;
        if request.reference.is_some() {
            warnings.push(format!(
                "MeanVC2: this voice converts toward its folder's {} (a pre-extracted speaker embedding); its reference voice is not analysed",
                speaker::SPEAKER_NPY
            ));
        }
        return Ok(embedding);
    }
    let clip = request
        .reference
        .as_ref()
        .ok_or("choose a reference voice from the voice library for this MeanVC2 voice")?;
    let Some(encoder) = speaker::encoder_path(&request.slot_dir) else {
        // No encoder now: an embedding an earlier encoder cached for this very clip still works.
        let cached = request
            .app_data
            .as_deref()
            .and_then(|app_data| speaker::latest_cached(app_data, clip))
            .ok_or_else(missing_encoder)?;
        warnings.push(format!(
            "MeanVC2: the speaker encoder ({SPEAKER_ID}) is not installed, so this voice uses the embedding cached earlier for {}",
            if clip.name.is_empty() { clip.voice_id.as_str() } else { clip.name.as_str() }
        ));
        return Ok(cached);
    };
    // Cached per encoder (its fingerprint), so a re-imported or per-voice encoder never reuses
    // another encoder's embedding.
    let backend = speaker::encoder_backend(&encoder)?;
    let opts = background_session_options(&request.session);
    cached_embedding(request.app_data.as_deref(), clip, &backend, |clip| {
        speaker::from_clip(clip, &encoder, &backend, &opts)
    })
}

/// What the voice really runs, for the metrics: anchored noise always (upstream reuses each mel
/// frame's noise), the governor when the run has it; notes for the RVC levers it does not use.
fn levers_for(request: &FamilyRequest) -> EffectiveLevers {
    let asked = &request.levers;
    let mut levers = EffectiveLevers {
        precision: PrecisionMode::Fp32,
        ..EffectiveLevers::default()
    };
    levers.levers.set(Lever::AnchoredNoise, true);
    levers
        .levers
        .set(Lever::Governor, asked.levers.get(Lever::Governor));
    if !asked.levers.get(Lever::AnchoredNoise) {
        levers
            .notes
            .push("anchored_noise: always on for MeanVC2 voices (each mel frame's noise is drawn once, as upstream does)".into());
    }
    let unused: Vec<&str> = Lever::ALL
        .into_iter()
        .filter(|&l| !matches!(l, Lever::AnchoredNoise | Lever::Governor) && asked.levers.get(l))
        .map(Lever::as_str)
        .collect();
    if !unused.is_empty() {
        levers.notes.push(format!(
            "{}: not used by MeanVC2 voices (a streaming model on the CPU)",
            unused.join(", ")
        ));
    }
    if request.ep != "cpu" {
        levers
            .notes
            .push("MeanVC2 voices run on the CPU on every execution provider (small graphs, no VRAM)".into());
    }
    if asked.accelerator != Accelerator::None {
        levers
            .notes
            .push(format!("{}: not used by MeanVC2 voices", asked.accelerator.as_str()));
    }
    if request.precision == PrecisionMode::Fp16 {
        levers
            .notes
            .push("precision fp16: MeanVC2 runs its fp32 graphs".into());
    }
    levers
}

/// Load a MeanVC2 voice (`request.reference` holds the clip).
pub fn create(request: FamilyRequest) -> Result<FamilyEngine, String> {
    let options = options_of(&request.manifest)?;
    let graphs_dir = graphs_dir(&request.slot_dir)?;
    let mut warnings = Vec::new();
    let embedding = resolve_speaker(&request, &mut warnings)?;
    let spks = speaker::spks_of(&embedding)?;
    let (graphs, outcomes) = OrtGraphs::load(&GraphSource::Dir(&graphs_dir), &realtime_session_options(&request.session))?;
    let stream = MeanVcStream::new(Box::new(graphs), &spks, options.steps, request.noise_seed)?;
    let mut model = MeanVcModel::new(
        stream,
        &ModelConfig {
            block_16k: options.block_16k(),
            steps: options.steps,
            silence_skip: request.silence_skip,
            align_16k: options.model_delay_16k(),
        },
    )?;
    let reference = Arc::new(embedding);
    model.set_reference(Some(reference.clone()));
    if request.f0_range == F0Range::Singing {
        warnings.push("MeanVC2 is a speech model: singing keeps the reference speaker's own pitch contour".into());
    }
    Ok(FamilyEngine {
        model: Box::new(model),
        levers: levers_for(&request),
        outcomes,
        warnings,
        model_delay_16k: options.model_delay_16k(),
        reference: Some(reference),
        // The kind's `families::min_output_buffer_ms` reads the same options from the manifest.
        min_output_buffer_ms: None,
    })
}

/// Cheap checks of one slot: its options, its graphs and a `speaker.npy` it holds.
pub fn check_slot(slot_dir: &Path, manifest: &RvcManifest) -> Result<(), String> {
    options_of(manifest)?;
    graphs_dir(slot_dir)?;
    if let Some(npy) = speaker::slot_npy(slot_dir) {
        speaker::from_npy(&npy, "")?;
    }
    Ok(())
}

/// Validate and normalize [`MeanVcOptions`].
pub fn normalize_options(options: &serde_json::Value) -> Result<serde_json::Value, String> {
    let mut o: MeanVcOptions = serde_json::from_value(options.clone()).map_err(|e| format!("invalid MeanVC2 options: {e}"))?;
    o.steps = o.steps.clamp(1, 2);
    o.tick_ms = if o.tick_ms >= 120 { 160 } else { 80 };
    serde_json::to_value(o).map_err(|e| e.to_string())
}

/// A MeanVC2 slot holds only its manifest (graphs in the model store; the reference is the voice's
/// `reference_voice_id`).
pub fn create_slot(_request: &FamilySlotRequest, ctx: &FamilySlotContext<'_>) -> Result<CreatedSlot, String> {
    super::default_create_slot(FamilyKind::MeanVc, ctx)
}

/// MeanVC2 voices are never imported from files.
pub fn run_import_job(_job: &ImportJob) -> Result<String, String> {
    Err("job: MeanVC2 voices are made from a library voice, not imported".into())
}
