//! Sigil-ZS: Seed-VC real-time zero-shot conversion in the engine (P5-B; ADR 0006 "Sigil-ZS").
//!
//! Sigil's exports of `seed-uvit-tat-xlsr-tiny` (`scripts/vc-families/seed_vc/export.py`; the model
//! store's [`graphs::ZS_GRAPHS`] / [`graphs::ZS_GRAPHS_FP16`]) driven by a Rust port of upstream's
//! real-time loop (`sampler.py`, upstream `real-time-gui.py` `custom_infer`):
//!
//! | module | what |
//! |---|---|
//! | [`preset`] | the ADR presets, the options' overrides ([`preset::Settings`]) and the tick [`preset::Geometry`] |
//! | [`reference`] | the reference (CAM++ style, prompt mel, prompt condition) on the CPU, cached per library voice |
//! | [`sampler`] | the CFM / Euler loop with CFG over the `dit_step` graph ([`sampler::Stage`]) |
//! | [`plans`] | the fixed-shape sessions of one stream shape (`gpu::SessionPlan`: IOBinding, CUDA graphs, priority), a process-wide pool, the tick |
//! | [`noise`] | `z`, the NSF source noise and the harmonic phases, anchored per absolute mel frame |
//! | [`dsp`] | torch's linspace, torchaudio's resample, the Kaldi fbank input and the NSF phase helpers |
//! | [`engine`] | [`engine::SigilZsModel`], the `VoiceConversionModel` |
//! | [`tune`] | the stream configuration from measured ticks (preset, EP, output buffer; the family's minimum buffer) |
//!
//! Runs on CUDA (CUDA graphs per preset shape) and DirectML (HiFT unpinned), fp32 by default (the
//! HiFT decoder's fp16 graph and the XLS-R / DiT fp16 graphs are switches; ADR 0006). The CPU is no
//! live mode (a tick takes 3-9× its block there), so the factory refuses it, and a GPU voice whose
//! XLS-R, DiT or vocoder session fell to the CPU through the EP ladder is refused when its stream is
//! configured; one-off conversions (the benchmark's offline ceiling) run on the CPU.
//!
//! # Frozen by P5-S0 (signatures; see `families` for the contract)
//! [`available`], [`info`], [`create`], [`check_slot`], [`normalize_options`], [`create_slot`],
//! [`run_import_job`], [`SigilZsOptions`] (the `manifest.family` object; TS `VcSigilZsOptions`).

pub mod dsp;
pub mod engine;
pub mod graphs;
pub mod noise;
pub mod plans;
pub mod preset;
pub mod reference;
pub mod sampler;
#[cfg(test)]
mod tests;
pub mod tune;

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::{CreatedSlot, FamilyEngine, FamilyInfo, FamilyKind, FamilyRequest, FamilySlotContext, FamilySlotRequest, CREATION_REFERENCE};
use crate::services::neural_audio::ort_session::{normalize_ep, EP_CPU};
use crate::services::neural_audio::vc::import::ImportJob;
use crate::services::neural_audio::vc::manifest::RvcManifest;

/// Presets ([`SigilZsOptions::preset`]; ADR 0006, prompts chosen by P5-B's benchmark): `sigil_low`
/// (100 ms, 0.3 s DiT context, 6 steps, CFG 0.7, 4 s prompt; the default), `sigil_lean` (100 ms,
/// 4 steps, no CFG, 3 s prompt), `upstream_readme` (180 ms, 0.5 s, 10 steps, CFG 0.7, 3 s prompt).
pub const PRESETS: [&str; 3] = ["sigil_low", "sigil_lean", "upstream_readme"];

/// The family's options (`manifest.family`, snake_case). Unset fields take the preset's values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SigilZsOptions {
    /// One of [`PRESETS`].
    pub preset: String,
    /// Diffusion (Euler) steps (1..=30; `None` = the preset's).
    pub steps: Option<u32>,
    /// Classifier-free guidance rate (0..=1; 0 runs the batch-1 step; `None` = the preset's).
    pub cfg_rate: Option<f32>,
    /// Longest reference prompt in seconds (0.5..=10; `None` = the preset's).
    pub prompt_s: Option<f32>,
    /// Run XLS-R in fp16 (off by default; passes its stage gate, not the waveform one).
    pub fp16_xlsr: bool,
    /// Run the DiT in fp16 (off by default; fails waveform parity: stays behind this switch).
    pub fp16_dit: bool,
    /// Run the HiFT decoder in fp16 (`None` = the benchmark-decided default,
    /// [`preset::DEFAULT_FP16_VOCODER`]).
    pub fp16_vocoder: Option<bool>,
}

impl Default for SigilZsOptions {
    fn default() -> Self {
        SigilZsOptions {
            preset: PRESETS[0].to_string(),
            steps: None,
            cfg_rate: None,
            prompt_s: None,
            fp16_xlsr: false,
            fp16_dit: false,
            fp16_vocoder: None,
        }
    }
}

/// Whether this build runs Sigil-ZS voices.
pub fn available() -> bool {
    true
}

pub fn info() -> FamilyInfo {
    FamilyInfo {
        family: "sigil_zs",
        format: super::SIGIL_ZS_FORMAT,
        experimental: false,
        zero_shot: true,
        singing: false,
        execution_providers: &["cuda", "dml"],
        needs_pitch: false,
        needs_content_encoder: false,
        creation: CREATION_REFERENCE,
        import_extensions: &[],
        controls: &["silence_db"],
        requires: &[graphs::ZS_GRAPHS],
    }
}

/// The options of a slot (normalized; a slot written by hand with bad values gets the defaults).
fn options_of(manifest: &RvcManifest) -> Result<SigilZsOptions, String> {
    let v = normalize_options(&super::options_of(manifest))?;
    serde_json::from_value(v).map_err(|e| format!("invalid Sigil-ZS options: {e}"))
}

/// Load a Sigil-ZS voice (`request.reference` holds the clip). The reference is computed (or read
/// from the voice library's cache) here; the GPU sessions are built for the stream's shape when the
/// harness configures it.
pub fn create(request: FamilyRequest) -> Result<FamilyEngine, String> {
    let ep = normalize_ep(&request.ep);
    if ep == EP_CPU {
        return Err("Sigil-ZS voices run on CUDA or DirectML (the CPU cannot convert them in real time)".into());
    }
    let options = options_of(&request.manifest)?;
    let settings = preset::Settings::new(&options, request.precision);
    let graphs = graphs::GraphSet::installed()?;
    let clip = request
        .reference
        .as_ref()
        .ok_or("choose a reference voice from the voice library for this zero-shot voice")?;
    let reference = reference::embedding(&graphs, clip, settings.preset.prompt_s, request.app_data.as_deref(), &request.session)?;
    let prepared = reference::prepare(&reference)?;
    let mut warnings = Vec::new();
    if (settings.fp16_xlsr || settings.fp16_dit) && graphs.fp16.is_none() {
        warnings.push(format!(
            "the fp16 XLS-R / DiT graphs ({}) are not installed; this voice runs them in fp32",
            graphs::ZS_GRAPHS_FP16
        ));
    }
    let model = engine::SigilZsModel::new(engine::ModelConfig {
        graphs,
        ep: ep.to_string(),
        session: request.session.clone(),
        levers: request.levers.clone(),
        settings,
        reference: reference.clone(),
        prepared,
        noise_seed: request.noise_seed,
        silence_skip: request.silence_skip,
        warnings: warnings.clone(),
    });
    let levers = model.effective_levers().clone();
    Ok(FamilyEngine {
        model: Box::new(model),
        levers,
        outcomes: Vec::new(),
        warnings,
        model_delay_16k: settings.preset.right_units * preset::UNIT_16K,
        reference: Some(reference),
        // The sessions are built when the harness configures the stream; the kind's
        // `families::min_output_buffer_ms` (this module's, per preset and the request's EP) applies.
        min_output_buffer_ms: None,
    })
}

/// The smallest output buffer `(prefill, max queue)` (ms) a voice of this slot plays with on `ep`
/// (`tune::min_output_buffer_ms` of its preset): `families::min_output_buffer_ms` for Sigil-ZS
/// voices (the registry passes the request's EP and the slot's manifest).
pub fn min_output_buffer_ms(ep: &str, manifest: &RvcManifest) -> (f64, f64) {
    let preset = options_of(manifest).map(|o| o.preset).unwrap_or_else(|_| PRESETS[0].to_string());
    tune::min_output_buffer_ms(&preset, normalize_ep(ep))
}

/// Cheap checks of one slot: valid options and the graphs installed.
pub fn check_slot(_slot_dir: &Path, manifest: &RvcManifest) -> Result<(), String> {
    options_of(manifest)?;
    graphs::GraphSet::installed().map(|_| ())
}

/// Validate and normalize [`SigilZsOptions`].
pub fn normalize_options(options: &serde_json::Value) -> Result<serde_json::Value, String> {
    let mut o: SigilZsOptions = serde_json::from_value(options.clone()).map_err(|e| format!("invalid Sigil-ZS options: {e}"))?;
    if !PRESETS.contains(&o.preset.as_str()) {
        o.preset = PRESETS[0].to_string();
    }
    o.steps = o.steps.map(|s| s.clamp(1, 30));
    o.cfg_rate = o.cfg_rate.filter(|v| v.is_finite()).map(|v| v.clamp(0.0, 1.0));
    o.prompt_s = o.prompt_s.filter(|v| v.is_finite()).map(|v| v.clamp(0.5, 10.0));
    serde_json::to_value(o).map_err(|e| e.to_string())
}

/// A Sigil-ZS slot holds only its manifest (the graphs are in the model store; the reference is the
/// voice's `reference_voice_id`).
pub fn create_slot(_request: &FamilySlotRequest, ctx: &FamilySlotContext<'_>) -> Result<CreatedSlot, String> {
    super::default_create_slot(FamilyKind::SigilZs, ctx)
}

/// Sigil-ZS voices are never imported from files.
pub fn run_import_job(_job: &ImportJob) -> Result<String, String> {
    Err("job: Sigil-ZS voices are made from a library voice, not imported".into())
}
