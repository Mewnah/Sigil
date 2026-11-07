//! DDSP-SVC 6.3 singing voices in the engine (P5-A; ADR 0006 "DDSP-SVC 6.3 singing").
//!
//! A user's DDSP-SVC 6.3 voice (the training checkpoint `model_*.pt` and its `config.yaml`) runs in
//! Sigil's own engine behind the shared stream harness:
//! - **Graphs** are Sigil's (`scripts/vc-families/ddsp_svc/export_templates.py`), not upstream's
//!   exporter: `ddsp` (upstream's inference path with the vocoder's own mel, a NaN-free sinc, the
//!   formant shift, and an overlap-add inverse STFT that DirectML runs in 3 ms where the
//!   ConvTranspose form took 2 s) and `reflow_step` (one Euler step of the rectified flow), both
//!   **templates without weights** in the model-store entry [`TEMPLATE_ID`], plus the openvpi
//!   PC-NSF-HiFiGAN 44.1 kHz vocoder ([`VOCODER_ID`]) with a `phase0` input.
//! - **Import** ([`create_slot`] → the isolated worker → [`run_import_job`]): the checkpoint is
//!   parsed by the allow-list unpickler, checked against the template's layout (embedded in this
//!   build: `ddsp_svc/layout.json`) and written as the slot's `weights.f32.bin`; nothing else of
//!   the checkpoint is ever executed or kept.
//! - **The tick** ([`model`]): ContentVec twice (tta2x) from the shared encoder pool, F0 from the
//!   voice's pitch extractor with upstream's post-processing and no coarse clamp (E2–D#7 and up to
//!   the vocoder source's Nyquist frequency), the `ddsp` graph, a Rust Euler loop over
//!   `reflow_step` with a fixed step count (a captured CUDA graph per step), the vocoder with its
//!   NSF phase carried across ticks, and upstream's volume gate (the voice's `silence_db`; always
//!   applied, the silence-skip lever only skips the graphs where the gate closes anyway). The flow
//!   starts at the voice's `t_start`, never below the one it was trained from. Frames, noise and
//!   phase are anchored to absolute time, so consecutive ticks agree.
//! - **EPs**: CUDA (graphs for the flow step and the vocoder; `ddsp` bound, its runtime slice
//!   keeps it off a graph) and DirectML (dynamic shapes for `ddsp`, whose pinned shapes DirectML's
//!   graph fusion refuses). The CPU cannot keep up live (ADR 0006: 3.4–5.6× the block).
//!
//! Presets ([`DdspSvcOptions::preset`]): `sigil_singing` (100 ms blocks, 1 s of context, 10 flow
//! steps), `sigil_singing_fast` (5 steps), `upstream_gui` (upstream's GUI: 500 ms, 2 s, 20 steps).
//! An explicit block from the settings (`block_ms` > 0) replaces the preset's (at least 50 ms).
//!
//! Pitch: a voice follows the voice changer's F0 method in the singing range. CREPE, SwiftF0 and
//! fusion track a sung E2 → D#6 sweep whole; RMVPE reads the notes above about D6 an octave low and
//! FCPE reads them flat ([`high_note_caveat`] warns; `tests::real_model_singing_range_and_cost`
//! measures).
//!
//! # Frozen by P5-S0 (signatures; P5-A implements; see `families` for the contract)
//! [`available`], [`info`], [`create`], [`check_slot`], [`normalize_options`], [`create_slot`],
//! [`run_import_job`], [`DdspSvcOptions`] (the `manifest.family` object; TS `VcDdspSvcOptions`).
//!
//! # Added by P5-A
//! [`TEMPLATE_ID`], [`VOCODER_ID`], [`preset_shape`] / [`PresetShape`], [`min_output_buffer_ms`] /
//! [`voice_output_buffer_ms`] (the output buffer a voice's bursts need, per preset and EP), and the
//! child modules
//! (`config`: the config.yaml subset; `layout`: the template table; `checkpoint`: the torch reader;
//! `import`; `dsp`: the tick's host-side signal processing; `model`: the voice model).

mod checkpoint;
mod config;
mod dsp;
mod import;
mod layout;
mod model;
#[cfg(test)]
mod tests;

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use super::{CreatedSlot, FamilyEngine, FamilyInfo, FamilyKind, FamilyRequest, FamilySlotContext, FamilySlotRequest, CREATION_IMPORT};
use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::normalize_ep;
use crate::services::neural_audio::vc::assets;
use crate::services::neural_audio::vc::engine::content::{self, EmbedderKind};
use crate::services::neural_audio::vc::engine::generator::component_key;
use crate::services::neural_audio::vc::engine::options::{F0Range, Lever, PrecisionMode};
use crate::services::neural_audio::vc::f0_post::F0Limits;
use crate::services::neural_audio::vc::import::worker::{self, WorkerCommand, WorkerJob, WorkerLimits};
use crate::services::neural_audio::vc::import::{ErrorCode, ImportError, ImportJob, IMPORT_RECORD_FORMAT};
use crate::services::neural_audio::vc::manifest::RvcManifest;

use import::{DdspRecord, MAX_SPEAKER_TABLE_BYTES, SPEAKERS_FILE, WEIGHTS_FILE};
use layout::{DdspLayout, GRAPH_DDSP, GRAPH_REFLOW};
use model::{DdspFiles, DdspSvcModel, ModelParts, PooledUnits, Tuning, VocoderRuntime, VoiceSetup};

/// Model-store id of the template graphs (`ddsp.onnx`, `reflow_step.onnx`, `layout.json`; a Sigil
/// export, `unpublished`).
pub const TEMPLATE_ID: &str = "vc/ddsp-svc-6.3";
/// Model-store id of the vocoder (`model.onnx` + `config.json`: openvpi PC-NSF-HiFiGAN 44.1 kHz,
/// hop 512, 128 mels, 2025.02, with Sigil's `phase0` input).
pub const VOCODER_ID: &str = "vc/vocoder-pc-nsf-hifigan-44k";

/// Presets ([`DdspSvcOptions::preset`]; ADR 0006 cost table): `sigil_singing` (100 ms, 10 steps,
/// the default), `sigil_singing_fast` (100 ms, 5 steps), `upstream_gui` (500 ms window, 20 steps).
pub const PRESETS: [&str; 3] = ["sigil_singing", "sigil_singing_fast", "upstream_gui"];

/// Shortest block an explicit `block_ms` may ask for (the tick's cost is mostly its window's, so
/// shorter blocks only multiply it).
const MIN_BLOCK_MS: f64 = 50.0;
/// Longest block.
const MAX_BLOCK_MS: f64 = 500.0;
/// Upstream's default `t_start` (`gui_reflow.py`).
const DEFAULT_T_START: f32 = 0.7;
/// The voice changer's default crossfade (ms): the shapes the voice is planned for at load (a
/// harness with another crossfade re-plans once).
const DEFAULT_CROSSFADE_MS: f64 = 20.0;

/// The family's options (`manifest.family`, snake_case). Unset fields take the preset's values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DdspSvcOptions {
    /// One of [`PRESETS`].
    pub preset: String,
    /// Reflow Euler steps (1..=50; `None` = the preset's).
    pub steps: Option<u32>,
    /// Where the flow starts between the DDSP mel and noise (0..=1; `None` = 0.7, upstream's). Never
    /// below the voice's trained `model.t_start` (see `effective_t_start`).
    pub t_start: Option<f32>,
}

impl Default for DdspSvcOptions {
    fn default() -> Self {
        DdspSvcOptions { preset: PRESETS[0].to_string(), steps: None, t_start: None }
    }
}

/// The stream shape of a preset.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PresetShape {
    pub block_ms: f64,
    pub extra_ms: f64,
    pub steps: u32,
}

/// [`PresetShape`] of a preset name (unknown names: the default preset).
pub fn preset_shape(preset: &str) -> PresetShape {
    match preset {
        "sigil_singing_fast" => PresetShape { block_ms: 100.0, extra_ms: 1000.0, steps: 5 },
        "upstream_gui" => PresetShape { block_ms: 500.0, extra_ms: 2000.0, steps: 20 },
        _ => PresetShape { block_ms: 100.0, extra_ms: 1000.0, steps: 10 },
    }
}

/// Whether a voice ticks in upstream_gui-sized bursts: that preset (a 2 s context), or any block
/// of 300 ms and more (a long decoded region for the flow and the vocoder).
fn long_burst(options: &DdspSvcOptions, block_ms: f64) -> bool {
    let block = if block_ms > 0.0 && block_ms.is_finite() { block_ms } else { preset_shape(&options.preset).block_ms };
    options.preset == "upstream_gui" || block >= 300.0
}

/// The smallest output buffer (prefill, max queue; ms) a DDSP-SVC voice plays with, for its preset,
/// steps and block (`block_ms` as the settings give it, 0 = the preset's) on the EP its graphs run
/// on (`active_ep`: CUDA, or anything slower, which gets the DirectML figures).
///
/// Each block computes in one burst while the output queue plays out, so the prefill must cover
/// the burst's p99 plus one device block and some jitter (25 ms; a 1024-frame block at 48 kHz is
/// 21 ms). The bursts at the presets' steps, from P5-A's paced runs (512-frame device blocks, the
/// sung sweep, `docs/benchmarks/VC_BENCHMARKS.md`): 100 ms blocks call p99 28-56 ms on CUDA and
/// 37-74 ms on DirectML (0 underruns per 9 s at 70 ms on CUDA, at 80 ms on DirectML with CREPE and
/// RMVPE; 39-94 at 35 ms); `upstream_gui` 48-59 ms on CUDA (0 at 100 ms) and 90-132 ms on DirectML
/// (30-58 underruns per 9 s at 70 ms, 0 at 150 ms). Steps beyond the preset's add their flow cost.
/// An adaptive buffer sized from the measured tick is a P7 item.
pub fn min_output_buffer_ms(options: &DdspSvcOptions, block_ms: f64, active_ep: &str) -> (f64, f64) {
    /// Burst p99 (ms) at the preset's steps and the cost of one more flow step:
    /// `[100 ms class, upstream_gui class]` × (CUDA, DirectML).
    const BURST_MS: [(f64, f64); 2] = [(45.0, 55.0), (75.0, 125.0)];
    const STEP_MS: [(f64, f64); 2] = [(0.7, 1.0), (1.2, 4.0)];
    const MARGIN_MS: f64 = 25.0;
    let class = usize::from(long_burst(options, block_ms));
    let cuda = matches!(active_ep.trim().to_ascii_lowercase().as_str(), "cuda" | "gpu" | "tensorrt_rtx" | "tensorrt");
    let pick = |(c, d): (f64, f64)| if cuda { c } else { d };
    let preset = preset_shape(&options.preset);
    let extra_steps = f64::from(options.steps.unwrap_or(preset.steps).clamp(1, 50).saturating_sub(preset.steps));
    let burst = pick(BURST_MS[class]) + extra_steps * pick(STEP_MS[class]);
    let prefill = (((burst + MARGIN_MS) / 10.0).ceil() * 10.0).clamp(70.0, 300.0);
    (prefill, (2.0 * prefill).min(600.0))
}

/// [`min_output_buffer_ms`] of a loaded voice: the CUDA figures when its three graphs all run on a
/// CUDA-class EP, else the slower ones (DirectML, or a fallback further down the ladder).
pub fn voice_output_buffer_ms(options: &DdspSvcOptions, block_ms: f64, outcomes: &[(String, EpOutcome)]) -> (f64, f64) {
    let graphs: Vec<&str> = outcomes
        .iter()
        .filter(|(k, _)| ["vc.ddsp", "vc.reflow", "vc.vocoder"].iter().any(|c| k == c || k.ends_with(&format!(".{c}"))))
        .map(|(_, o)| o.active.as_str())
        .collect();
    let cuda = !graphs.is_empty() && graphs.iter().all(|a| matches!(a.trim().to_ascii_lowercase().as_str(), "cuda" | "gpu" | "tensorrt_rtx" | "tensorrt"));
    min_output_buffer_ms(options, block_ms, if cuda { "cuda" } else { "dml" })
}

/// Whether this build runs DDSP-SVC voices.
pub fn available() -> bool {
    true
}

pub fn info() -> FamilyInfo {
    FamilyInfo {
        family: "ddsp_svc",
        format: super::DDSP_SVC_FORMAT,
        experimental: false,
        zero_shot: false,
        singing: true,
        execution_providers: &["cuda", "dml"],
        needs_pitch: true,
        needs_content_encoder: true,
        creation: CREATION_IMPORT,
        import_extensions: &[".pt", ".zip"],
        controls: &["pitch", "formant", "speaker", "silence_db"],
        requires: &[assets::EMBEDDER_CONTENTVEC, TEMPLATE_ID, VOCODER_ID],
    }
}

fn options_from(manifest: &RvcManifest) -> DdspSvcOptions {
    let raw = super::options_of(manifest);
    normalize_options(&raw)
        .ok()
        .and_then(|v| serde_json::from_value(v).ok())
        .unwrap_or_default()
}

/// The installed template (graphs, checked against the layout this build carries) and vocoder.
struct Assets {
    files: DdspFiles,
    vocoder: VocoderRuntime,
}

fn installed(id: &str) -> Result<PathBuf, String> {
    crate::services::model_store::installed_path(id)
        .filter(|p| p.is_dir())
        .ok_or_else(|| format!("DDSP-SVC voices need '{id}': add it in Models & Runtimes"))
}

fn resolve_assets(layout: &DdspLayout, template_dir: &Path, vocoder_dir: &Path) -> Result<Assets, String> {
    let installed_layout = DdspLayout::read(template_dir).map_err(|e| format!("{TEMPLATE_ID}: {e}"))?;
    if let Some(why) = layout.export_mismatch(&installed_layout) {
        return Err(format!("the installed {TEMPLATE_ID} does not match this build: {why}; reinstall it in Models & Runtimes"));
    }
    let mut graphs = layout
        .graph_files(template_dir)
        .map_err(|e| format!("{TEMPLATE_ID}: {e}; reinstall it in Models & Runtimes"))?;
    let mut take = |g: &str| graphs.remove(g).ok_or_else(|| format!("layout: no graph {g}"));
    let (ddsp, reflow) = (take(GRAPH_DDSP)?, take(GRAPH_REFLOW)?);
    let vocoder = vocoder_dir.join(assets::file_name(None));
    if !vocoder.is_file() {
        return Err(format!("{VOCODER_ID}: model.onnx is missing; reinstall it in Models & Runtimes"));
    }
    let rt = VocoderRuntime::from_config(&vocoder_dir.join("config.json"), layout)?;
    Ok(Assets { files: DdspFiles { ddsp, reflow, vocoder }, vocoder: rt })
}

/// The voice's record, checked against the template this build runs.
fn voice_record(slot_dir: &Path, layout: &DdspLayout) -> Result<DdspRecord, String> {
    let record = DdspRecord::read(slot_dir).ok_or_else(|| format!("the voice's {} is missing or unreadable; import it again", import::RECORD_FILE))?;
    if record.weights_id != layout.weights.weights_id {
        return Err(format!(
            "this voice was imported for another DDSP-SVC template ({}); import it again",
            record.template
        ));
    }
    Ok(record)
}

/// The slot's speaker table (`count × dim`), when the voice has one.
fn read_speakers(slot_dir: &Path, record: &DdspRecord, dim: usize) -> Result<Option<(Vec<f32>, usize)>, String> {
    if record.speakers.is_none() {
        return Ok(None);
    }
    let path = slot_dir.join(SPEAKERS_FILE);
    let bytes = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    if bytes.is_empty() || bytes.len() % (4 * dim) != 0 || bytes.len() > MAX_SPEAKER_TABLE_BYTES {
        return Err(format!("{}: {} bytes is not a table of {dim}-wide speaker rows", path.display(), bytes.len()));
    }
    let rows: Vec<f32> = bytes.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    if rows.iter().any(|v| !v.is_finite()) {
        return Err(format!("{}: non-finite values", path.display()));
    }
    Ok(Some((rows, dim)))
}

/// What a pitch method does with the top of the singing range, measured on a sung E2 → D#6 sweep
/// (`tests::real_model_singing_range_and_cost`): RMVPE reads the notes above about D6 (1.15 kHz) an
/// octave low and FCPE reads them up to two semitones flat, while CREPE, SwiftF0 and fusion follow
/// the whole sweep (0 gross errors, 5-7 cents).
fn high_note_caveat(pitch_method: &str) -> Option<String> {
    let what = match pitch_method.trim() {
        "rmvpe" => "RMVPE reads sung notes above about D6 (1.15 kHz) an octave low",
        "fcpe" => "FCPE reads sung notes above about C6 (1 kHz) up to two semitones flat",
        _ => return None,
    };
    Some(format!("pitch: {what}; CREPE, SwiftF0 or fusion follow a singer's whole range"))
}

fn ms_to_16k(ms: f64) -> usize {
    ((ms.max(0.0) / 10.0).round() as usize) * 160
}

/// Where a voice's flow starts: the asked `t_start` (the option, else upstream's 0.7), never below
/// the `t_start` the voice was trained from (`model.t_start` of its config.yaml: the trainer draws
/// `t` only in `[t_start, 1]`, so the velocity field is untrained below it; upstream
/// `main_reflow.py` takes the larger of the two as well). An explicit option that was raised is
/// reported.
fn effective_t_start(options: &DdspSvcOptions, trained_t_start: f32, warnings: &mut Vec<String>) -> f32 {
    let floor = if trained_t_start.is_finite() { trained_t_start.clamp(0.0, 1.0) } else { 0.0 };
    let asked = options.t_start.filter(|t| t.is_finite()).unwrap_or(DEFAULT_T_START).clamp(0.0, 1.0);
    if asked >= floor {
        return asked;
    }
    if let Some(t) = options.t_start {
        warnings.push(format!(
            "DDSP-SVC: this voice was trained with t_start {floor}, so its flow starts there, not at the {t} the voice's options ask for"
        ));
    }
    floor
}

/// The stream tuning of a voice: the preset's block (or the settings' explicit `block_ms`), context,
/// steps and `t_start` (at least the voice's trained one, [`effective_t_start`]), the settings'
/// lookahead.
fn tuning_for(options: &DdspSvcOptions, trained_t_start: f32, block_ms: f64, lookahead_ms: f64, warnings: &mut Vec<String>) -> Tuning {
    let preset = preset_shape(&options.preset);
    let block = if block_ms > 0.0 && block_ms.is_finite() {
        let b = (block_ms / 10.0).ceil() * 10.0;
        let clamped = b.clamp(MIN_BLOCK_MS, MAX_BLOCK_MS);
        if (clamped - b).abs() > f64::EPSILON {
            warnings.push(format!("DDSP-SVC: a {block_ms} ms block runs as {clamped} ms (voices of this family run {MIN_BLOCK_MS}-{MAX_BLOCK_MS} ms blocks)"));
        }
        clamped
    } else {
        preset.block_ms
    };
    Tuning {
        block_16k: ms_to_16k(block),
        extra_16k: ms_to_16k(preset.extra_ms),
        lookahead_16k: ms_to_16k(lookahead_ms.clamp(0.0, 80.0)),
        steps: options.steps.unwrap_or(preset.steps).clamp(1, 50),
        t_start: effective_t_start(options, trained_t_start, warnings),
    }
}

/// The 16 kHz samples the harness asks for per call with the default crossfade (block + its SOLA
/// tail; `VcStreamHarness`'s formula), to plan the sessions once at load.
fn default_want(block_16k: usize, rate: u32) -> usize {
    let overlap = ((DEFAULT_CROSSFADE_MS / 1000.0) * f64::from(rate)).round() as u64;
    let tail = (overlap + overlap / 2 + 2) * 16_000;
    block_16k + tail.div_ceil(u64::from(rate)) as usize
}

/// Load a DDSP-SVC voice.
pub fn create(request: FamilyRequest) -> Result<FamilyEngine, String> {
    let layout = DdspLayout::embedded()?;
    let record = voice_record(&request.slot_dir, &layout)?;
    let config = record.config().ok_or("the voice's import record has no readable config; import it again")?;
    let assets = resolve_assets(&layout, &installed(TEMPLATE_ID)?, &installed(VOCODER_ID)?)?;
    let options = options_from(&request.manifest);
    let mut warnings: Vec<String> = Vec::new();
    let mut outcomes = Vec::new();
    let tuning = tuning_for(&options, config.t_start, request.block_ms, request.lookahead_ms, &mut warnings);
    if request.f0_range == F0Range::Speech {
        // The pitch extractor was built for speech: notes above its 1.1 kHz limit come out held.
        warnings.push(format!(
            "pitch range: this singing voice tracks pitch in the speech range (up to {} Hz); choose the singing range to follow notes up to {} Hz",
            F0Limits::SPEECH.max_hz,
            F0Limits::SINGING.max_hz
        ));
    }
    if let Some(note) = high_note_caveat(&request.pitch_method) {
        warnings.push(note);
    }
    let gpu = normalize_ep(&request.ep) != "cpu";
    if request.precision == PrecisionMode::Fp16 {
        warnings.push("precision: DDSP-SVC graphs run in fp32 (TF32 on CUDA); fp16 applies to the content encoder only".into());
    }
    let encoder = content::acquire_for_slot(
        EmbedderKind::ContentVec768,
        request.content_encoder.as_deref(),
        Some(&request.slot_dir),
        &request.ep,
        &request.session,
        if gpu { request.precision } else { PrecisionMode::Fp32 },
        tuning.extra_16k + tuning.block_16k,
        &mut warnings,
    )?;
    outcomes.push((
        component_key(&request.session, "vc.content"),
        encoder.lock().unwrap_or_else(std::sync::PoisonError::into_inner).outcome().clone(),
    ));
    let pitch = request.pitch.ok_or("DDSP-SVC voices need a pitch extractor")?;
    let setup = VoiceSetup {
        tta2x: config.tta2x(),
        f0_min: config.f0_min,
        volume_window: config.volume_window() as usize,
        speakers: read_speakers(&request.slot_dir, &record, layout.speakers.dim)?,
    };
    if !config.use_pitch_aug {
        warnings.push("formant: this voice was trained without pitch augmentation, so the formant control does nothing".into());
    }
    let mut levers = request.levers.clone();
    if levers.levers.get(Lever::PartialDecode) {
        // DDSP-SVC decodes only the frames it returns plus a margin (upstream's silence_front).
        levers.notes.push("partial_decode: DDSP-SVC always decodes only its returned frames".into());
    }
    let rate = layout.arch.sampling_rate;
    let mut model = DdspSvcModel::new(ModelParts {
        layout,
        files: assets.files,
        vocoder: assets.vocoder,
        weights: request.slot_dir.join(WEIGHTS_FILE),
        setup,
        tuning,
        units: Box::new(PooledUnits(encoder)),
        pitch,
        ep: request.ep.clone(),
        session: request.session.clone(),
        levers,
        noise_seed: request.noise_seed,
        silence_skip: request.silence_skip,
        outcomes,
        warnings,
    })?;
    // Plan the sessions now at the shape the voice changer's defaults give (a harness with another
    // crossfade or context re-plans in `configure_stream`).
    let want = default_want(tuning.block_16k, rate);
    model.configure(tuning.block_16k, want, ms_to_16k(request.extra_context_ms))?;
    let lookahead = tuning.lookahead_16k;
    // What this preset needs on the EP its graphs landed on (the voice changer plays the voice
    // with it; it overrides the kind's `families::min_output_buffer_ms`).
    let (prefill, queue) = voice_output_buffer_ms(&options, request.block_ms, model.outcomes());
    tracing::info!(target: "sigil::vc", "DDSP-SVC voice ({}): its blocks compute in one burst; it needs at least a {prefill:.0} / {queue:.0} ms output buffer", options.preset);
    Ok(FamilyEngine {
        levers: model.levers().clone(),
        outcomes: model.outcomes().to_vec(),
        warnings: model.warnings().to_vec(),
        model: Box::new(model),
        model_delay_16k: lookahead,
        reference: None,
        min_output_buffer_ms: Some((prefill, queue)),
    })
}

/// Cheap checks of one slot: its import fits this build's template, its files are whole, and the
/// template and vocoder it runs on are installed.
pub fn check_slot(slot_dir: &Path, _manifest: &RvcManifest) -> Result<(), String> {
    let layout = DdspLayout::embedded()?;
    let record = voice_record(slot_dir, &layout)?;
    let weights = std::fs::metadata(slot_dir.join(WEIGHTS_FILE)).map(|m| m.len()).unwrap_or(0);
    if weights != layout.weights.bytes {
        return Err(format!("the voice's {WEIGHTS_FILE} is missing or incomplete; import it again"));
    }
    if record.speakers.is_some() && !slot_dir.join(SPEAKERS_FILE).is_file() {
        return Err(format!("the voice's {SPEAKERS_FILE} is missing; import it again"));
    }
    let config = record.config().ok_or("the voice's import record has no readable config; import it again")?;
    if let Some(why) = config.mismatch(&layout.arch, &layout.vocoder) {
        return Err(format!("this DDSP-SVC voice cannot run on Sigil's template: {why}"));
    }
    resolve_assets(&layout, &installed(TEMPLATE_ID)?, &installed(VOCODER_ID)?).map(|_| ())
}

/// Validate and normalize [`DdspSvcOptions`].
pub fn normalize_options(options: &serde_json::Value) -> Result<serde_json::Value, String> {
    let mut o: DdspSvcOptions = serde_json::from_value(options.clone()).map_err(|e| format!("invalid DDSP-SVC options: {e}"))?;
    if !PRESETS.contains(&o.preset.as_str()) {
        o.preset = PRESETS[0].to_string();
    }
    o.steps = o.steps.map(|s| s.clamp(1, 50));
    o.t_start = o.t_start.filter(|t| t.is_finite()).map(|t| t.clamp(0.0, 1.0));
    serde_json::to_value(o).map_err(|e| e.to_string())
}

/// Import into `out_dir` in this process (the worker's body, and the fallback where the isolated
/// worker is not available).
fn import_in_process(source: &Path, out_dir: &Path, name: &str, options: serde_json::Value) -> Result<DdspRecord, ImportError> {
    let started = Instant::now();
    let layout = DdspLayout::embedded().map_err(|e| ImportError::new(ErrorCode::LayoutInvalid, e))?;
    let voice = import::import_voice(source, &layout)?;
    import::write_slot(out_dir, &voice, &layout, name, options, started)
}

/// The worker's stdout: the importer's record envelope (its `format` is what the worker protocol
/// accepts) with the DDSP-SVC record inside.
fn envelope(record: &DdspRecord) -> serde_json::Value {
    serde_json::json!({ "format": IMPORT_RECORD_FORMAT, "family": FamilyKind::DdspSvc.as_str(), "ddsp": record })
}

/// Import a user's checkpoint into the new slot (through the isolated worker).
pub fn create_slot(request: &FamilySlotRequest, ctx: &FamilySlotContext<'_>) -> Result<CreatedSlot, String> {
    let source = request.source.as_deref().map(str::trim).filter(|s| !s.is_empty()).ok_or("job: choose a DDSP-SVC checkpoint to import")?;
    let source = PathBuf::from(source);
    if !source.exists() {
        return Err(format!("io: {} does not exist", source.display()));
    }
    (ctx.progress)("reading", 0.1, "Reading the DDSP-SVC voice");
    let warnings = match (worker::available(), ctx.exe) {
        (true, Some(exe)) => {
            let job = WorkerJob {
                smoke: false,
                name: Some(ctx.name.clone()),
                family: Some(FamilyKind::DdspSvc.as_str().to_string()),
                family_options: Some(ctx.options.clone()),
                ..WorkerJob::new(source.clone(), ctx.slot_dir.clone(), None)
            };
            let limits = WorkerLimits { memory_limit_mb: 3072, time_limit_s: 300 };
            let out = worker::run_isolated_with(&WorkerCommand::app(exe), &job, limits, ctx.cancel, None).map_err(|e| e.to_string())?;
            let record: DdspRecord = serde_json::from_value(out.get("ddsp").cloned().unwrap_or_default())
                .map_err(|e| format!("io: the import worker returned no DDSP-SVC record ({e})"))?;
            record.warnings
        }
        _ => {
            if ctx.cancel.load(Ordering::Acquire) {
                return Err("job: the import was cancelled".into());
            }
            import_in_process(&source, &ctx.slot_dir, &ctx.name, ctx.options.clone()).map_err(|e| e.to_string())?.warnings
        }
    };
    (ctx.progress)("writing", 0.9, "Writing the voice");
    Ok(CreatedSlot { warnings })
}

/// The isolated worker's body for a DDSP-SVC import (`job.family = "ddsp_svc"`).
pub fn run_import_job(job: &ImportJob) -> Result<String, String> {
    if job.version != 1 {
        return Err(format!("job: job version {} is not supported", job.version));
    }
    let options = normalize_options(job.family_options.as_ref().unwrap_or(&serde_json::json!({}))).map_err(|e| format!("job: {e}"))?;
    let name = job
        .name
        .as_deref()
        .and_then(crate::services::neural_audio::vc::import::clean_name)
        .or_else(|| job.source.file_stem().and_then(|s| crate::services::neural_audio::vc::import::clean_name(&s.to_string_lossy())))
        .unwrap_or_else(|| "DDSP-SVC voice".into());
    let record = import_in_process(&job.source, &job.out_dir, &name, options).map_err(|e| e.to_string())?;
    serde_json::to_string_pretty(&envelope(&record)).map_err(|e| format!("io: {e}"))
}
