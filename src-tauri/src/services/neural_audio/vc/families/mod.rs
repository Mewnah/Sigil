//! Voice-conversion families beyond RVC (P5): the registry the engine factory, the slot store, the
//! start preflight, the import worker and the capabilities read (P5-S0, scaffold-owned).
//!
//! | family id | `manifest.format` | module | owner | what it is |
//! |---|---|---|---|---|
//! | `ddsp_svc` | [`DDSP_SVC_FORMAT`] | [`ddsp_svc`] | P5-A | DDSP-SVC 6.3 singing voices (users' checkpoints, Sigil's graphs, ADR 0006) |
//! | `sigil_zs` | [`SIGIL_ZS_FORMAT`] | [`sigil_zs`] | P5-B | Sigil-ZS: Seed-VC real-time zero-shot in the engine |
//! | `meanvc` | [`MEANVC_FORMAT`] | [`meanvc`] | P5-C | MeanVC2 CPU zero-shot (experimental) |
//! | `sovits` | [`SOVITS_FORMAT`] | [`sovits`] | P5-D | so-vits-svc 4.1 bring-your-own ONNX exports |
//! | `sidecar_v2` | [`SIDECAR_V2_FORMAT`] | [`sidecar_v2`] | P5-E | research engines (X-VC, StreamVoiceAnon, seed-vc-realtime) over sidecar protocol v2 |
//!
//! RVC itself stays outside this module: `rvc` (bring-your-own ONNX, `vc::rvc`) and `sigil-rvc-1`
//! (template voices, `vc::engine`) are the factory's own arms and never change because a family was
//! added.
//!
//! # Frozen contract (P5-S0; change only through INTEGRATE; structs may gain fields)
//! Every family module implements the same free functions (the stubs return "not available yet"
//! until their workstream lands; nothing outside the module needs editing when it does):
//! - `available() -> bool`: this build runs the family (the capabilities, the slot store and the
//!   factory all read it).
//! - `info() -> FamilyInfo`: static facts for the UI and the generic code ([`FamilyInfo`]).
//! - `create(FamilyRequest) -> Result<FamilyEngine, String>`: load a voice (on the loader thread;
//!   never on the audio or worker thread). The model runs behind the shared `VcStreamHarness`
//!   (fixed blocks via `fixed_block` / `configure_stream`, the latency budget, SOLA, resampling,
//!   the in-harness denoiser and the governor wrapper are the harness's and the factory's).
//! - `check_slot(slot_dir, &RvcManifest) -> Result<(), String>`: cheap file checks of one slot (the
//!   slot listing's `problem` and the start preflight); no model loads.
//! - `normalize_options(&serde_json::Value) -> Result<serde_json::Value, String>`: validate and
//!   normalize the family's options (`manifest.family`), for `voice_changer_create_family_slot` and
//!   `voice_changer_set_family_options`.
//! - `create_slot(&FamilySlotRequest, &FamilySlotContext) -> Result<CreatedSlot, String>`: fill the
//!   new slot folder (`ctx.slot_dir`, already created and empty): the manifest
//!   ([`write_family_manifest`]) and whatever the family imports. Families made from a library voice
//!   or a sidecar recipe use [`default_create_slot`].
//! - `run_import_job(&ImportJob) -> Result<String, String>`: the isolated import worker's body for
//!   a family import (`Sigil.exe --headless vc-import-worker <job.json>` with `job.family` set;
//!   untrusted checkpoints are parsed only there, with `import::pickle`'s allow-list VM). Returns the
//!   JSON the worker prints. Only families that import untrusted files use it (DDSP-SVC).
//!
//! The registry functions below dispatch on [`FamilyKind`] and add what is common: unavailable
//! families are refused by name, options are objects, slots are named and checked.
//!
//! Conventions for the family workstreams:
//! - Session builders go through `neural_audio::ort_session` (`build_session_with`, spinning off);
//!   GPU families use `gpu::plan::SessionPlan` for fixed shapes (IOBinding, CUDA graphs) and report
//!   their sessions' EP outcomes in `FamilyEngine::outcomes`; DirectML sessions are kept for the
//!   process where possible (ADR 0006 finding 5).
//! - Voice controls (`VoiceConversionModel::set_voice_controls`, the per-tick pitch / index /
//!   protect / speaker of `ConversionCtx`) reach a family model like any voice; the factory adds no
//!   post stage to family voices (RVC voices get `engine::post::PostStage` from it), so a family that
//!   answers `rms_mix` or `formant` runs a `PostStage` itself, and `FamilyInfo::controls` lists only
//!   what the family really does. The in-harness denoiser (GTCRN) and the governor's light-F0
//!   wrapper are added by the factory as for RVC voices.
//! - VRAM: the factory measures and claims the load's VRAM (`LoadedVoice::load_cancellable` →
//!   `resources::claim_vram(Subsystem::Vc, ..)`), as for RVC voices; a family that starts a helper
//!   with its own GPU memory reports it with `resources::claim_external_vram` (sidecars).
//! - Model files come from the model store only (`model_store::installed_path(id)`); a family's
//!   catalog fragment is its own (`model-catalog/vc-ddsp.json` P5-A, `vc-zeroshot.json` P5-B,
//!   `vc-meanvc.json` P5-C), entries `unpublished` until the user publishes Sigil exports.
//! - A module named `families/<x>.rs` may grow child modules in `families/<x>/` (Rust's 2018
//!   layout); `sigil_zs` is a folder already. The registry (this file) and `reference.rs` stay the
//!   scaffold's.
//! - Zero-shot families get the reference clip in `FamilyRequest::reference` and return their
//!   embedding in `FamilyEngine::reference` (computed through `reference::cached_embedding`, so the
//!   voice library caches it per backend); the harness hands it back on every tick
//!   (`ConversionCtx::reference`).
//! - Output buffer (P5-I0): a family whose blocks compute in one burst declares the smallest
//!   output prefill / queue it plays with in [`min_output_buffer_ms`] (per kind, EP and slot
//!   options), and a voice whose need is only known once it loaded (the EP its graphs landed on)
//!   returns it in `FamilyEngine::min_output_buffer_ms`, which wins; the factory carries it on the
//!   voice and the worker (and the benchmark's paced runs) use the larger of it and the user's.
//!   DDSP-SVC per voice: 70 / 140 ms (100 ms presets on CUDA), 80 / 160 (DirectML), 100 / 200 and
//!   150 / 300 (`upstream_gui` on CUDA / DirectML); the kind's 70 / 140 is the fallback. Sigil-ZS
//!   per preset and EP (`sigil_zs::min_output_buffer_ms`), MeanVC2 for its 160 ms tick only.

pub mod ddsp_svc;
pub mod meanvc;
pub mod reference;
pub mod sidecar_v2;
pub mod sigil_zs;
pub mod sovits;

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::RvcPresetProfile;

use super::engine::options::{EffectiveLevers, F0Range, PrecisionMode};
use super::import::ImportJob;
use super::manifest::RvcManifest;
use super::traits::{PitchExtractor, VoiceConversionModel};
use reference::{ReferenceClip, ReferenceEmbedding};

/// `manifest.format` of a DDSP-SVC 6.3 voice (P5-A).
pub const DDSP_SVC_FORMAT: &str = "ddsp-svc-6.3";
/// `manifest.format` of a Sigil-ZS zero-shot slot (P5-B).
pub const SIGIL_ZS_FORMAT: &str = "sigil-zs-1";
/// `manifest.format` of a MeanVC2 zero-shot slot (P5-C).
pub const MEANVC_FORMAT: &str = "meanvc-2";
/// `manifest.format` of a so-vits-svc 4.1 ONNX voice (P5-D).
pub const SOVITS_FORMAT: &str = "sovits-4.1";
/// `manifest.format` of a sidecar protocol v2 voice (P5-E).
pub const SIDECAR_V2_FORMAT: &str = "sidecar-v2";

/// [`FamilyInfo::creation`]: slots are made from a voice-library reference (zero-shot).
pub const CREATION_REFERENCE: &str = "reference";
/// [`FamilyInfo::creation`]: slots are made by importing a file (`FamilySlotRequest::source`).
pub const CREATION_IMPORT: &str = "import";
/// [`FamilyInfo::creation`]: slots are made from a sidecar recipe or URL (`options`).
pub const CREATION_SIDECAR: &str = "sidecar";

/// Hot voice-control ids for [`FamilyInfo::controls`] (what the inspector shows for a voice).
#[allow(dead_code)] // P5-S0 frozen contract (TS `VC_FAMILY_CONTROLS`); checked by the tests.
pub const CONTROL_IDS: [&str; 8] = ["pitch", "formant", "rms_mix", "index", "protect", "speaker", "noise_scale", "silence_db"];

/// A voice-conversion family of this registry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FamilyKind {
    DdspSvc,
    SigilZs,
    #[serde(rename = "meanvc")]
    MeanVc,
    Sovits,
    SidecarV2,
}

impl FamilyKind {
    pub const ALL: [FamilyKind; 5] = [
        FamilyKind::DdspSvc,
        FamilyKind::SigilZs,
        FamilyKind::MeanVc,
        FamilyKind::Sovits,
        FamilyKind::SidecarV2,
    ];

    /// Family id (snake_case; the UI's i18n keys use it).
    pub fn as_str(self) -> &'static str {
        match self {
            FamilyKind::DdspSvc => "ddsp_svc",
            FamilyKind::SigilZs => "sigil_zs",
            FamilyKind::MeanVc => "meanvc",
            FamilyKind::Sovits => "sovits",
            FamilyKind::SidecarV2 => "sidecar_v2",
        }
    }

    /// `manifest.format` of its slots.
    pub fn format(self) -> &'static str {
        match self {
            FamilyKind::DdspSvc => DDSP_SVC_FORMAT,
            FamilyKind::SigilZs => SIGIL_ZS_FORMAT,
            FamilyKind::MeanVc => MEANVC_FORMAT,
            FamilyKind::Sovits => SOVITS_FORMAT,
            FamilyKind::SidecarV2 => SIDECAR_V2_FORMAT,
        }
    }

    /// The family of a `manifest.format` (exact, trimmed).
    pub fn from_format(format: &str) -> Option<FamilyKind> {
        let f = format.trim();
        FamilyKind::ALL.into_iter().find(|k| k.format() == f)
    }

    /// A family id or format (trimmed, case-insensitive): what IPC requests name.
    pub fn from_name(name: &str) -> Option<FamilyKind> {
        let n = name.trim().to_ascii_lowercase();
        FamilyKind::ALL.into_iter().find(|k| k.as_str() == n || k.format() == n)
    }

    /// Whether this build runs the family.
    pub fn available(self) -> bool {
        match self {
            FamilyKind::DdspSvc => ddsp_svc::available(),
            FamilyKind::SigilZs => sigil_zs::available(),
            FamilyKind::MeanVc => meanvc::available(),
            FamilyKind::Sovits => sovits::available(),
            FamilyKind::SidecarV2 => sidecar_v2::available(),
        }
    }

    pub fn info(self) -> FamilyInfo {
        match self {
            FamilyKind::DdspSvc => ddsp_svc::info(),
            FamilyKind::SigilZs => sigil_zs::info(),
            FamilyKind::MeanVc => meanvc::info(),
            FamilyKind::Sovits => sovits::info(),
            FamilyKind::SidecarV2 => sidecar_v2::info(),
        }
    }
}

/// Static facts about a family (UI labels come from i18n by `family`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct FamilyInfo {
    /// Family id ([`FamilyKind::as_str`]).
    pub family: &'static str,
    /// `manifest.format` of its slots.
    pub format: &'static str,
    /// Labelled experimental in the UI.
    pub experimental: bool,
    /// Converts toward a reference voice from the shared voice library (params
    /// `reference_voice_id`, per voice).
    pub zero_shot: bool,
    /// A singing family (the UI offers it under singing).
    pub singing: bool,
    /// EP ladder rungs it runs on (`cuda`, `dml`, `cpu`), best first; the UI labels a voice that
    /// cannot run on the selected rung.
    pub execution_providers: &'static [&'static str],
    /// The factory builds the voice's pitch extractor for it (`FamilyRequest::pitch`), and the
    /// start preflight resolves the pitch model files.
    pub needs_pitch: bool,
    /// It runs the shared content encoder (`manifest.embedder`, default `contentvec`): the start
    /// preflight resolves it (`FamilyRequest::content_encoder`).
    pub needs_content_encoder: bool,
    /// How its slots are made: [`CREATION_REFERENCE`], [`CREATION_IMPORT`] or [`CREATION_SIDECAR`].
    pub creation: &'static str,
    /// File extensions `create_slot` takes as `source` (lower case, with the dot).
    pub import_extensions: &'static [&'static str],
    /// Hot voice controls it answers ([`CONTROL_IDS`]); the inspector hides the others.
    pub controls: &'static [&'static str],
    /// Model-store ids it needs installed (its catalog fragment defines them; the UI shows a
    /// requirement card).
    pub requires: &'static [&'static str],
}

/// What a family's `create` gets (owned; built by the engine factory from its `EngineRequest`).
#[allow(dead_code)] // P5-S0 frozen contract: the family workstreams read these fields.
pub struct FamilyRequest {
    pub kind: FamilyKind,
    pub slot_dir: PathBuf,
    /// The slot's manifest (its options are `manifest.family`).
    pub manifest: RvcManifest,
    pub app_data: Option<PathBuf>,
    /// EP ladder rung (`cpu` / `cuda` / `dml`).
    pub ep: String,
    pub session: VcSessionOptions,
    /// The resolved levers (the family keeps what it runs and notes the rest in
    /// `FamilyEngine::levers`).
    pub levers: EffectiveLevers,
    pub precision: PrecisionMode,
    pub preset: RvcPresetProfile,
    /// Engine block (ms; 0 = from `chunk_frames`), device block and rate, extra context, lookahead:
    /// the stream shape the user asked for (a family with its own presets may round them).
    pub block_ms: f64,
    pub chunk_frames: u32,
    pub device_rate: u32,
    pub extra_context_ms: f64,
    pub lookahead_ms: f64,
    pub f0_range: F0Range,
    pub silence_skip: bool,
    /// Time-anchored noise seed of the voice (stable per voice, `factory::noise_seed_for`).
    pub noise_seed: u64,
    /// The voice's F0 extractor (only when `info().needs_pitch`), built by the pitch bank with the
    /// voice's GPU levers (and the governor's light-F0 switch when that lever is on).
    pub pitch: Option<Box<dyn PitchExtractor>>,
    /// The F0 method that runs (after the build's stand-ins).
    pub pitch_method: String,
    /// Explicit content-encoder file (params `rvc_hubert_path`); `None` = the shared encoder pool /
    /// model store (`engine::content`).
    pub content_encoder: Option<PathBuf>,
    /// The reference clip of a zero-shot voice (`info().zero_shot`; the factory refuses to load a
    /// family made from a reference without it).
    pub reference: Option<ReferenceClip>,
    /// The latency budget (ms) when the voice loads: the default stall timeout of a family that
    /// waits on something outside the process (a sidecar's per-frame timeout).
    pub latency_budget_ms: f64,
    /// Set when the voice changer no longer wants the voice (another was selected, or it stopped):
    /// families that wait on something outside the process (a sidecar helper's start or handshake)
    /// stop waiting. `None` = the load is never called off.
    pub cancel: Option<Arc<AtomicBool>>,
}

/// A loaded family voice.
pub struct FamilyEngine {
    pub model: Box<dyn VoiceConversionModel>,
    /// Levers the voice really uses (notes for the rest).
    pub levers: EffectiveLevers,
    /// EP outcome per session component (`vc.<component>`).
    pub outcomes: Vec<(String, EpOutcome)>,
    /// Non-fatal load issues (shown as `rvc_warning`).
    pub warnings: Vec<String>,
    /// Delay the model itself adds (16 kHz samples), on top of the harness's algorithmic latency.
    pub model_delay_16k: usize,
    /// The reference embedding a zero-shot model converts toward (installed in the harness).
    pub reference: Option<Arc<ReferenceEmbedding>>,
    /// The smallest output buffer (prefill, max queue; ms) this voice plays with, when it depends
    /// on its preset and the EP its sessions landed on (overrides [`min_output_buffer_ms`] of its
    /// kind); `None` = the kind's.
    pub min_output_buffer_ms: Option<(f64, f64)>,
}

/// `voice_changer_create_family_slot { request }` (snake_case payload).
#[derive(Clone, Debug, Default, Deserialize)]
pub struct FamilySlotRequest {
    /// Family id (`sigil_zs`) or its `manifest.format`.
    pub family: String,
    /// Display name (default: from the reference voice or the imported file, else the family).
    #[serde(default)]
    pub name: Option<String>,
    /// The library voice a zero-shot voice starts with (stored as the voice's
    /// `reference_voice_id` in its `slot.json`).
    #[serde(default)]
    pub reference_voice_id: Option<String>,
    /// The file (or folder) to import, for [`CREATION_IMPORT`] families.
    #[serde(default)]
    pub source: Option<String>,
    /// The family's options (`manifest.family`; normalized by the family).
    #[serde(default)]
    pub options: Option<serde_json::Value>,
    /// Make it the active voice when done.
    #[serde(default)]
    pub activate: bool,
}

/// Where and how `create_slot` works.
#[allow(dead_code)] // P5-S0 frozen contract: the importing families (P5-A, P5-D) read these fields.
pub struct FamilySlotContext<'a> {
    /// `app_data/voice-changer/models`.
    pub models_dir: &'a Path,
    /// The new slot's folder (created, empty). On an error the caller removes it.
    pub slot_dir: PathBuf,
    /// Display name (resolved by the caller).
    pub name: String,
    /// The options after [`normalize_options`].
    pub options: serde_json::Value,
    pub app_data: Option<&'a Path>,
    /// The bundled templates folder (Tauri's resource dir), when known.
    pub templates_dir: Option<&'a Path>,
    /// The Sigil executable, for the isolated import worker (`import::worker::run_isolated`).
    pub exe: Option<&'a Path>,
    /// Set when the user cancelled (`voice_changer_cancel_import { jobId }`).
    pub cancel: &'a AtomicBool,
    /// Progress (stage, fraction 0..1, message) → `voice_changer:import_progress`; stages are the
    /// importer's (`reading`, `converting`, `smoke`, `writing`).
    pub progress: &'a (dyn Fn(&str, f64, &str) + Send + Sync),
}

/// What `create_slot` reports.
#[derive(Clone, Debug, Default)]
pub struct CreatedSlot {
    pub warnings: Vec<String>,
}

fn not_available(kind: FamilyKind) -> String {
    format!("the {} voice family is not available in this build yet", kind.as_str())
}

/// Load a voice of `request.kind` (refused by name when this build cannot run the family).
pub fn create(request: FamilyRequest) -> Result<FamilyEngine, String> {
    let kind = request.kind;
    if !kind.available() {
        return Err(not_available(kind));
    }
    match kind {
        FamilyKind::DdspSvc => ddsp_svc::create(request),
        FamilyKind::SigilZs => sigil_zs::create(request),
        FamilyKind::MeanVc => meanvc::create(request),
        FamilyKind::Sovits => sovits::create(request),
        FamilyKind::SidecarV2 => sidecar_v2::create(request),
    }
}

/// Why a slot of `kind` cannot run in this build, if it cannot (slot listing, start preflight).
pub fn slot_problem(kind: FamilyKind, slot_dir: &Path, manifest: &RvcManifest) -> Option<String> {
    if !kind.available() {
        return Some(not_available(kind));
    }
    let checked = match kind {
        FamilyKind::DdspSvc => ddsp_svc::check_slot(slot_dir, manifest),
        FamilyKind::SigilZs => sigil_zs::check_slot(slot_dir, manifest),
        FamilyKind::MeanVc => meanvc::check_slot(slot_dir, manifest),
        FamilyKind::Sovits => sovits::check_slot(slot_dir, manifest),
        FamilyKind::SidecarV2 => sidecar_v2::check_slot(slot_dir, manifest),
    };
    checked.err()
}

/// Validate and normalize `options` (`null` = the family's defaults; anything but an object is
/// refused before the family sees it).
pub fn normalize_options(kind: FamilyKind, options: Option<&serde_json::Value>) -> Result<serde_json::Value, String> {
    let options = match options {
        None | Some(serde_json::Value::Null) => serde_json::Value::Object(Default::default()),
        Some(v @ serde_json::Value::Object(_)) => v.clone(),
        Some(_) => return Err(format!("the {} options must be an object", kind.as_str())),
    };
    match kind {
        FamilyKind::DdspSvc => ddsp_svc::normalize_options(&options),
        FamilyKind::SigilZs => sigil_zs::normalize_options(&options),
        FamilyKind::MeanVc => meanvc::normalize_options(&options),
        FamilyKind::Sovits => sovits::normalize_options(&options),
        FamilyKind::SidecarV2 => sidecar_v2::normalize_options(&options),
    }
}

/// Fill a new slot folder (refused by name when this build cannot run the family).
pub fn create_slot(kind: FamilyKind, request: &FamilySlotRequest, ctx: &FamilySlotContext<'_>) -> Result<CreatedSlot, String> {
    if !kind.available() {
        return Err(not_available(kind));
    }
    match kind {
        FamilyKind::DdspSvc => ddsp_svc::create_slot(request, ctx),
        FamilyKind::SigilZs => sigil_zs::create_slot(request, ctx),
        FamilyKind::MeanVc => meanvc::create_slot(request, ctx),
        FamilyKind::Sovits => sovits::create_slot(request, ctx),
        FamilyKind::SidecarV2 => sidecar_v2::create_slot(request, ctx),
    }
}

/// The import worker's body for a family job (`job.family`; see the module docs).
pub fn run_import_job(job: &ImportJob) -> Result<String, String> {
    let name = job.family.as_deref().unwrap_or_default();
    let kind = FamilyKind::from_name(name).ok_or_else(|| format!("job: unknown voice family {name:?}"))?;
    if !kind.available() {
        return Err(format!("job: {}", not_available(kind)));
    }
    match kind {
        FamilyKind::DdspSvc => ddsp_svc::run_import_job(job),
        FamilyKind::SigilZs => sigil_zs::run_import_job(job),
        FamilyKind::MeanVc => meanvc::run_import_job(job),
        FamilyKind::Sovits => sovits::run_import_job(job),
        FamilyKind::SidecarV2 => sidecar_v2::run_import_job(job),
    }
}

/// The manifest of a new family slot: its format, display name and options (every other field at
/// its default; a family sets the ones it uses before writing).
pub fn family_manifest(kind: FamilyKind, name: &str, options: serde_json::Value) -> RvcManifest {
    RvcManifest {
        format: kind.format().to_string(),
        name: Some(name.to_string()),
        family: Some(options),
        created_unix_ms: Some(super::import::unix_ms()),
        ..RvcManifest::default()
    }
}

/// Write `manifest` as `<slot_dir>/manifest.json` (atomically).
pub fn write_family_manifest(slot_dir: &Path, manifest: &RvcManifest) -> Result<(), String> {
    let bytes = serde_json::to_vec_pretty(manifest).map_err(|e| e.to_string())?;
    super::import::write_atomic(&slot_dir.join("manifest.json"), &bytes).map_err(|e| e.to_string())
}

/// `create_slot` of a family whose slots hold only a manifest (made from a library voice or a
/// sidecar recipe): [`family_manifest`] with the context's name and options.
pub fn default_create_slot(kind: FamilyKind, ctx: &FamilySlotContext<'_>) -> Result<CreatedSlot, String> {
    write_family_manifest(&ctx.slot_dir, &family_manifest(kind, &ctx.name, ctx.options.clone()))?;
    Ok(CreatedSlot::default())
}

/// The smallest output buffer (prefill, max queue; ms) a voice of `kind` with `manifest` plays with
/// on the EP rung `ep` (P5-I0): the voice changer and the benchmark's paced runs use the larger of
/// this and the user's values. A loaded voice's `FamilyEngine::min_output_buffer_ms` wins over it.
///
/// Fixed-block families that compute a whole block at once leave the output queue to cover the
/// tick's compute; the voice changer's default 35 / 70 ms runs dry while they compute (measured,
/// `docs/benchmarks/VC_BENCHMARKS.md`):
/// - DDSP-SVC (P5-A): 27-50 ms per 100 ms block; 0-94 underruns per 9 s at 35 ms, 0 at 70 ms. The
///   loaded voice sets its own per preset and the EP its graphs landed on (70 / 140 ms for the
///   100 ms presets on CUDA, 80 / 160 on DirectML, 100 / 200 and 150 / 300 for `upstream_gui` on
///   CUDA / DirectML); 70 / 140 is the fallback.
/// - Sigil-ZS (P5-B): 15-35 ms per tick at full clocks, 80-105 ms spikes on an idle NVIDIA GPU;
///   145-527 underruns per 20 s at 35 / 70 ms. Per preset and EP (`sigil_zs::min_output_buffer_ms`):
///   `sigil_low` 95 / 165 ms on CUDA (80 / 150 underran 75 times per 20 s on idle CUDA, p99 tick
///   82 ms) and 85 / 155 on DirectML, `sigil_lean` 80 / 150, `upstream_readme` 120 / 190 (CUDA) and
///   125 / 195 (DirectML); never below 80 / 150. The prefill stays below one 100 ms burst: a prefill
///   of a whole block waits for a second burst (+50 ms measured at 100 ms against +10.7 ms at 95).
/// - MeanVC2: the default 80 ms tick (split into 40 ms blocks) keeps the user's buffer; the 160 ms
///   tick computes each tick in one 33-62 ms burst (265 / 162 underruns per 20 s at 35 / 70 ms in
///   P5-C's bench), so it plays with 80 / 150 ms (P5-C fix A/B: 0-3 underruns paced, 0 under H3 on
///   an uncontended rig).
///
/// so-vits-svc (the user's block) and sidecar voices (their own jitter buffer) keep the user's
/// buffer.
pub fn min_output_buffer_ms(kind: FamilyKind, ep: &str, manifest: &RvcManifest) -> Option<(f64, f64)> {
    match kind {
        FamilyKind::DdspSvc => Some((70.0, 140.0)),
        FamilyKind::SigilZs => Some(sigil_zs::min_output_buffer_ms(ep, manifest)),
        FamilyKind::MeanVc => meanvc::min_output_buffer_ms(manifest),
        FamilyKind::Sovits | FamilyKind::SidecarV2 => None,
    }
}

/// The family options of a manifest (`manifest.family`, `{}` when unset).
pub fn options_of(manifest: &RvcManifest) -> serde_json::Value {
    match &manifest.family {
        Some(v @ serde_json::Value::Object(_)) => v.clone(),
        _ => serde_json::Value::Object(Default::default()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_and_formats_are_unique_and_round_trip() {
        for k in FamilyKind::ALL {
            assert_eq!(FamilyKind::from_format(k.format()), Some(k));
            assert_eq!(FamilyKind::from_name(k.as_str()), Some(k));
            assert_eq!(FamilyKind::from_name(&format!(" {} ", k.format().to_uppercase())), Some(k));
            assert_eq!(serde_json::to_value(k).unwrap(), k.as_str());
            let info = k.info();
            assert_eq!((info.family, info.format), (k.as_str(), k.format()), "{k:?}");
            assert!([CREATION_REFERENCE, CREATION_IMPORT, CREATION_SIDECAR].contains(&info.creation), "{k:?}");
            assert!(info.controls.iter().all(|c| CONTROL_IDS.contains(c)), "{k:?}");
            assert!(info.execution_providers.iter().all(|e| ["cuda", "dml", "cpu"].contains(e)), "{k:?}");
            assert!(!info.execution_providers.is_empty(), "{k:?}");
            assert!(info.creation != CREATION_REFERENCE || info.zero_shot, "{k:?}: made from a reference, so zero-shot");
            for ext in info.import_extensions {
                assert!(ext.starts_with('.') && ext.to_lowercase() == *ext, "{k:?} {ext}");
            }
        }
        let mut formats: Vec<&str> = FamilyKind::ALL.iter().map(|k| k.format()).collect();
        formats.extend(["rvc", "sigil-rvc-1"]);
        let n = formats.len();
        formats.sort_unstable();
        formats.dedup();
        assert_eq!(formats.len(), n, "formats are unique and never an RVC one");
        assert!(FamilyKind::from_format("rvc").is_none() && FamilyKind::from_name("sigil-rvc-1").is_none());
    }

    /// Stubs refuse by name; a family turns on only through its own `available()`.
    #[test]
    fn unavailable_families_are_refused_by_name() {
        for k in FamilyKind::ALL {
            let manifest = family_manifest(k, "x", serde_json::json!({}));
            let problem = slot_problem(k, Path::new("C:/nowhere"), &manifest);
            if !k.available() {
                assert!(problem.unwrap().contains(k.as_str()));
                let job = ImportJob::for_family(k.as_str());
                assert!(run_import_job(&job).unwrap_err().starts_with("job: "));
            }
        }
        assert!(run_import_job(&ImportJob::for_family("nope")).unwrap_err().contains("unknown voice family"));
    }

    /// The minimums stay inside the voice changer's output-buffer range (`clamp_output_buffer_ms`:
    /// prefill 5-300 ms, queue at least prefill + 5 ms) and above its default 35 / 70 ms.
    #[test]
    fn output_buffer_minimums_are_valid_buffers() {
        let defaults = |k| family_manifest(k, "x", serde_json::json!({}));
        for k in FamilyKind::ALL {
            for ep in ["cuda", "dml"] {
                if let Some((prefill, queue)) = min_output_buffer_ms(k, ep, &defaults(k)) {
                    assert!((35.0..=300.0).contains(&prefill), "{k:?} {ep} prefill {prefill}");
                    assert!(queue >= prefill + 5.0 && queue <= 600.0, "{k:?} {ep} queue {queue}");
                }
            }
        }
        let zs = defaults(FamilyKind::SigilZs);
        assert_eq!(min_output_buffer_ms(FamilyKind::SigilZs, "cuda", &zs), Some((95.0, 165.0)), "sigil_low on CUDA");
        assert_eq!(min_output_buffer_ms(FamilyKind::SigilZs, "dml", &zs), Some((85.0, 155.0)));
        assert_eq!(min_output_buffer_ms(FamilyKind::DdspSvc, "dml", &defaults(FamilyKind::DdspSvc)), Some((70.0, 140.0)), "the fallback");
        assert_eq!(min_output_buffer_ms(FamilyKind::MeanVc, "cpu", &defaults(FamilyKind::MeanVc)), None, "the split 80 ms tick");
        let long_tick = family_manifest(FamilyKind::MeanVc, "x", serde_json::json!({"tick_ms": 160}));
        assert_eq!(min_output_buffer_ms(FamilyKind::MeanVc, "cpu", &long_tick), Some((80.0, 150.0)));
        assert_eq!(min_output_buffer_ms(FamilyKind::Sovits, "cuda", &defaults(FamilyKind::Sovits)), None);
    }

    #[test]
    fn options_must_be_objects() {
        for k in FamilyKind::ALL {
            assert!(normalize_options(k, Some(&serde_json::json!([1]))).is_err());
            assert!(normalize_options(k, Some(&serde_json::json!("x"))).is_err());
            // Defaults are valid except where the family needs a choice (a custom sidecar's URL).
            if k != FamilyKind::SidecarV2 {
                assert!(normalize_options(k, None).unwrap().is_object());
                assert!(normalize_options(k, Some(&serde_json::Value::Null)).unwrap().is_object());
            }
        }
    }

    #[test]
    fn family_manifests_round_trip_through_the_slot_loader() {
        let dir = std::env::temp_dir().join(format!("sigil-vc-family-{}-{}", std::process::id(), super::super::import::unix_ms()));
        for k in FamilyKind::ALL {
            let slot = dir.join(k.as_str());
            std::fs::create_dir_all(&slot).unwrap();
            let m = family_manifest(k, "Voice", serde_json::json!({"preset": "x"}));
            write_family_manifest(&slot, &m).unwrap();
            let back = super::super::manifest::load_slot_manifest(&slot).unwrap();
            assert_eq!(back.slot_format(), super::super::manifest::SlotFormat::Family(k));
            assert_eq!(options_of(&back), serde_json::json!({"preset": "x"}));
            assert_eq!(back.name.as_deref(), Some("Voice"));
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
