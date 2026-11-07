//! Loading and re-planning a template voice (split out of `engine/mod.rs` by P3B-S0): the load
//! ([`SigilRvcEngine::load`] / [`SigilRvcEngine::load_hooked`]), the generator build,
//! [`SigilRvcEngine::set_lookahead_ms`], the re-plan, the load-time test conversion and the body of
//! `VoiceConversionModel::configure_stream`.
//!
//! # Owner (P3B)
//! P3B-H (final shapes at load, parallel cold start). Keep the telemetry load spans
//! (`neural_audio::telemetry::load_span`) on every path.
//!
//! # Final shapes at load (P3B-H)
//! The request carries what the stream harness will ask for (`SigilRvcRequest::lookahead_ms`,
//! `::stream`, `::content_context_16k`), so the load plans the shapes the voice runs once: the plan
//! with the lookahead and the harness's SOLA tail and extra context, the content encoder at its
//! window, and the generator built a single time. `configure_stream` then finds the plan final and
//! rebuilds nothing (before, a lookahead or a crossfade other than 20 ms built the sessions twice).
//! A DirectML content encoder that cannot run at a fixed shape drops the lookahead before the
//! generator exists (it would cost 35-49 ms per block there), which is free at that point.
//! `SIGIL_VC_FINAL_SHAPES=0` ignores `lookahead_ms` and `stream` (P5's flow: the caller re-plans).
//!
//! # Parallel cold start (P3B-H)
//! The branches of a load do not depend on one another: the pitch bank (the caller's builder,
//! including RMVPE's plan), the content encoder (acquire and, on a GPU rung, its fixed-shape plan,
//! built here instead of at the first tick), the generator (weights, fp16 image, `enc` and
//! `flowdec`; P3B-G builds those two beside each other) and the retrieval index. They run on scoped
//! threads that take the loading thread's priority and EcoQoS state, and join before the voice's
//! test conversion; every join checks the cancel flag, so a load nobody wants any more stops at the
//! next stage. On DirectML the content encoder is acquired first (its shape decides the lookahead).
//! A background hot-switch load is the same code on a lower-priority thread. Kill switch
//! `SIGIL_VC_PARALLEL_LOAD=0` (everything in P5's order on this thread); `=1` turns on P3B-G's
//! `enc` / `flowdec` concurrency as well.
//!
//! # Single validation conversion
//! The factory's load is always followed by the stream harness's own two warm-up ticks, so it asks
//! for one validation conversion here (`harness_warmup_follows`); every other caller keeps two.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::ort_session::{normalize_ep, VcSessionOptions};
use crate::services::neural_audio::telemetry::{self, LoadStage, ReplanReason};
use crate::services::neural_audio::vc::import::layout::{Streaming, Template};
use crate::services::neural_audio::vc::import::session::WeightImages;
use crate::services::neural_audio::vc::import::{writer, IMPORT_RECORD, SPEAKERS_FILE};

use super::super::manifest::{RvcManifest, SlotFormat};
use super::super::traits::{ConversionCtx, PitchExtractor, VoiceControls, VoiceConversionModel};
use super::content::{self, ContentLease, EmbedderKind, PlanReport, SharedEncoder};
use super::content_stats::ContentStats;
use super::f0_cache::{F0Cache, F0CacheConfig};
use super::generator::{SpeakerTable, SplitGenerator};
use super::noise::{AnchoredNoise, NoiseFrames, NoiseStream, PhaseTracker};
use super::plan::{block_frames, ShapePlan, DEFAULT_SOLA_TAIL_16K, HOP_16K};
use super::presets::EnginePreset;
use super::retrieval::{Retrieval, RetrievalBackend};
use super::{post_stage, retrieval, Accelerator, EffectiveLevers, F0Range, Lever, LoadOptions, PrecisionMode, SigilRvcEngine, SigilRvcRequest, StreamGeometry, TickStats};

/// The error of a load called off (the factory's `LOAD_CANCELLED` is the same text).
pub(super) const CANCELLED: &str = "the voice load was cancelled";

/// `SIGIL_VC_PARALLEL_LOAD=0|1`: the cold-start branches run beside each other (see the module docs).
pub const PARALLEL_LOAD_ENV: &str = "SIGIL_VC_PARALLEL_LOAD";
/// Whether the branches run beside each other when [`PARALLEL_LOAD_ENV`] is unset.
pub const PARALLEL_LOAD_DEFAULT: bool = true;
/// `SIGIL_VC_FINAL_SHAPES=0`: the load ignores the request's lookahead and stream geometry.
pub const FINAL_SHAPES_ENV: &str = "SIGIL_VC_FINAL_SHAPES";

fn env_switch(name: &str, default: bool) -> bool {
    match std::env::var(name).as_deref().map(str::trim) {
        Ok("1" | "on" | "true" | "yes") => true,
        Ok("0" | "off" | "false" | "no") => false,
        _ => default,
    }
}

/// How the pitch bank is built beside the other branches: it gets the audio frames of one incremental
/// F0 call and the longest whole-window call (16 kHz samples) of the final plan, which size RMVPE's plan.
pub type PitchBuilder<'a> = Box<dyn FnOnce(usize, usize) -> Result<Box<dyn PitchExtractor>, String> + Send + 'a>;

/// What a load takes besides its request.
#[derive(Default)]
pub(super) struct LoadParams<'a> {
    /// Builds the pitch extractor beside the other branches; the request's `pitch` is then a
    /// placeholder. `None`: the request's extractor is used as it is.
    pub pitch: Option<PitchBuilder<'a>>,
    /// Set when nobody wants the voice any more.
    pub cancel: Option<&'a AtomicBool>,
    /// The caller runs the stream harness's warm-up ticks next: one validation conversion here.
    pub harness_warmup_follows: bool,
    /// Plan the request's lookahead and stream geometry (`None`: `SIGIL_VC_FINAL_SHAPES`; tests set it).
    pub final_shapes: Option<bool>,
    /// Run the branches beside each other (`None`: `SIGIL_VC_PARALLEL_LOAD`; tests set it).
    pub parallel: Option<bool>,
}

/// The extractor a request carries while the real one is built by its `pitch` builder: every call
/// is refused.
struct PendingPitch;

impl PitchExtractor for PendingPitch {
    fn f0(&mut self, _: &[f32]) -> Result<Vec<f32>, String> {
        Err("the pitch extractor is still loading".into())
    }
}

pub(super) fn load_err(slot: &Path, e: impl std::fmt::Display) -> String {
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

/// Lookahead frames of `ms` (rounded up to 10 ms frames; 0 for off, non-finite or negative).
fn lookahead_frames_of(ms: f64) -> usize {
    if ms.is_finite() && ms > 0.0 {
        (ms / 10.0 - 1e-9).ceil() as usize
    } else {
        0
    }
}

fn ms_to_16k(ms: f64) -> usize {
    if ms.is_finite() && ms > 0.0 {
        (ms * 16.0).round() as usize
    } else {
        0
    }
}

/// The plan a voice loads with: the preset's block, the harness's SOLA tail and extra context when the
/// request carries its geometry (else P3's defaults), and `lookahead_frames` of lookahead.
fn load_plan(
    streaming: &Streaming,
    preset: &EnginePreset,
    out_sr: u32,
    partial_decode: bool,
    lookahead_frames: usize,
    stream: Option<&StreamGeometry>,
) -> Result<ShapePlan, String> {
    let block_16k = block_frames(preset.block_ms)? * HOP_16K;
    let preset_extra = ms_to_16k(preset.extra_context_ms);
    let (want, extra) = match stream {
        Some(g) => (
            if g.want_16k >= block_16k { g.want_16k } else { block_16k + DEFAULT_SOLA_TAIL_16K },
            g.harness_extra_16k.max(preset_extra),
        ),
        None => (block_16k + DEFAULT_SOLA_TAIL_16K, preset_extra),
    };
    ShapePlan::for_stream_with_lookahead(streaming, out_sr, partial_decode, block_16k, want, extra, lookahead_frames)
}

/// The loading thread's priority and EcoQoS state, copied onto a branch's thread (a background
/// hot-voice load stays a background load on every thread).
#[derive(Clone, Copy, Debug, Default)]
struct ThreadPolicy {
    #[cfg(windows)]
    priority: Option<i32>,
    #[cfg(windows)]
    throttling: Option<(u32, u32)>,
}

#[cfg(windows)]
impl ThreadPolicy {
    fn of_current() -> ThreadPolicy {
        use windows::Win32::System::Threading::{
            GetCurrentThread, GetThreadInformation, GetThreadPriority, ThreadPowerThrottling, THREAD_POWER_THROTTLING_CURRENT_VERSION,
            THREAD_POWER_THROTTLING_STATE,
        };
        let mut state = THREAD_POWER_THROTTLING_STATE { Version: THREAD_POWER_THROTTLING_CURRENT_VERSION, ControlMask: 0, StateMask: 0 };
        // SAFETY: the pseudo-handle of the calling thread; pointer and size describe `state`.
        unsafe {
            let priority = GetThreadPriority(GetCurrentThread());
            let throttling = GetThreadInformation(
                GetCurrentThread(),
                ThreadPowerThrottling,
                &mut state as *mut _ as *mut core::ffi::c_void,
                std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
            )
            .ok()
            .map(|_| (state.ControlMask, state.StateMask));
            // THREAD_PRIORITY_ERROR_RETURN (MAXLONG) when the call failed.
            ThreadPolicy { priority: (priority != i32::MAX).then_some(priority), throttling }
        }
    }

    fn apply_to_current(&self) {
        use windows::Win32::System::Threading::{
            GetCurrentThread, SetThreadInformation, SetThreadPriority, ThreadPowerThrottling, THREAD_POWER_THROTTLING_CURRENT_VERSION,
            THREAD_POWER_THROTTLING_STATE, THREAD_PRIORITY,
        };
        // SAFETY: the pseudo-handle of the calling thread; pointer and size describe `state`.
        unsafe {
            if let Some(p) = self.priority {
                let _ = SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY(p));
            }
            if let Some((control, state)) = self.throttling {
                let state = THREAD_POWER_THROTTLING_STATE { Version: THREAD_POWER_THROTTLING_CURRENT_VERSION, ControlMask: control, StateMask: state };
                let _ = SetThreadInformation(
                    GetCurrentThread(),
                    ThreadPowerThrottling,
                    &state as *const _ as *const core::ffi::c_void,
                    std::mem::size_of::<THREAD_POWER_THROTTLING_STATE>() as u32,
                );
            }
        }
    }
}

#[cfg(not(windows))]
impl ThreadPolicy {
    fn of_current() -> ThreadPolicy {
        ThreadPolicy::default()
    }

    fn apply_to_current(&self) {}
}

/// A branch running on a scoped thread (or the reason it could not start).
type Spawned<'scope, T> = Result<std::thread::ScopedJoinHandle<'scope, T>, String>;

fn spawn_branch<'scope, 'env, T: Send + 'scope>(
    scope: &'scope std::thread::Scope<'scope, 'env>,
    name: &'static str,
    policy: ThreadPolicy,
    f: impl FnOnce() -> T + Send + 'scope,
) -> Spawned<'scope, T> {
    std::thread::Builder::new()
        .name(name.into())
        .spawn_scoped(scope, move || {
            policy.apply_to_current();
            f()
        })
        .map_err(|e| format!("could not start the {name} thread: {e}"))
}

fn join_branch<T>(h: Spawned<'_, T>, what: &str) -> Result<T, String> {
    h?.join().map_err(|_| format!("the {what} load panicked"))
}

/// The content branch's result.
struct ContentOut {
    encoder: SharedEncoder,
    warnings: Vec<String>,
    /// The fixed-shape plan, built here instead of at the voice's first tick.
    prebuilt: Option<(ContentLease, PlanReport)>,
}

/// The retrieval branch's result.
struct RetrievalOut {
    retrieval: Option<Retrieval>,
    warnings: Vec<String>,
    /// Lever notes (`gpu_knn: ...`).
    notes: Vec<String>,
}

/// Quiet 150 Hz tone (voiced for the pitch extractor, above any silence gate): the load's test input.
fn warm_tone(len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| 0.05 * (std::f32::consts::TAU * 150.0 * i as f32 / 16_000.0).sin())
        .collect()
}

/// The content encoder's EP outcome and plan notes after its plan was built or refused.
fn note_content_plan(outcomes: &mut [(String, EpOutcome)], warnings: &mut Vec<String>, report: PlanReport) {
    if let Some(o) = outcomes.iter_mut().find(|(c, _)| c == "vc.content") {
        o.1 = report.outcome;
    }
    for n in report.notes {
        if !warnings.contains(&n) {
            warnings.push(n);
        }
    }
}

/// The content encoder of the voice from the pool (acquired as a deferred, session-less encoder on a
/// GPU rung with the plan path on, so no dynamic session is built and released).
#[allow(clippy::too_many_arguments)]
fn acquire_content(
    embedder: EmbedderKind,
    explicit: Option<&Path>,
    slot_dir: &Path,
    ep: &str,
    session: &VcSessionOptions,
    levers: &EffectiveLevers,
    plan: &ShapePlan,
    content_context_16k: Option<usize>,
) -> Result<ContentOut, String> {
    let mut warnings = Vec::new();
    let window = plan.content_window_16k(content_context_16k);
    let encoder = content::acquire_for_engine(embedder, explicit, Some(slot_dir), ep, session, levers.precision, window, &mut warnings)?;
    Ok(ContentOut { encoder, warnings, prebuilt: None })
}

/// Build the fixed-shape plan of the voice's content window now, on a GPU rung (the voice's first
/// tick would otherwise, serially after the generator): the lease the voice will hold and the plan's
/// report. `None` when there is nothing to build (the CPU, the kill switch, `content_norm`, an encoder
/// that cannot run plans) or the build failed (the first tick then reports it).
fn prebuild_content_plan(
    encoder: &SharedEncoder,
    levers: &EffectiveLevers,
    plan: &ShapePlan,
    content_context_16k: Option<usize>,
) -> Option<(ContentLease, PlanReport)> {
    if levers.levers.get(Lever::ContentNorm) || !content::content_plan_switch() {
        return None;
    }
    let mut enc = encoder.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    if enc.plan_blocker().is_some() || !content::plan_rung(&enc.outcome().active) {
        return None;
    }
    let mut plan_levers = content::plan_levers(levers);
    if !content::content_graph_switch() {
        plan_levers.cuda_graphs = false;
    }
    let window = plan.content_window_16k(content_context_16k);
    let (mut lease, mut feats) = (None, Vec::new());
    match enc.encode_fixed(&warm_tone(window), plan_levers, &mut lease, &mut feats) {
        Ok((_, Some(report))) => lease.map(|l| (l, report)),
        _ => None,
    }
}

/// The retrieval index of the voice, if it has one (P3-D backends behind one API).
fn retrieval_branch(index: Option<PathBuf>, levers: &EffectiveLevers, ep: &str, session: &VcSessionOptions, emb: usize) -> RetrievalOut {
    let mut out = RetrievalOut {
        retrieval: None,
        warnings: Vec::new(),
        notes: Vec::new(),
    };
    let Some(path) = index else { return out };
    let backend = if levers.levers.get(Lever::GpuKnn) && retrieval::gpu_knn_available() {
        RetrievalBackend::GpuKnn
    } else {
        RetrievalBackend::CpuIvf
    };
    let loaded = Retrieval::load(&path, backend, ep, session, levers.precision).or_else(|e| {
        if backend == RetrievalBackend::GpuKnn {
            out.notes.push(format!("gpu_knn: {e}; using the CPU index"));
            Retrieval::load(&path, RetrievalBackend::CpuIvf, ep, session, levers.precision)
        } else {
            Err(e)
        }
    });
    match loaded {
        Ok(r) if r.dim() == emb || r.dim() == 0 => out.retrieval = Some(r),
        Ok(r) => out.warnings.push(format!(
            "retrieval index {} has {} dimensions but the voice uses {emb}; index rate has no effect",
            path.display(),
            r.dim()
        )),
        Err(e) => out.warnings.push(format!(
            "retrieval index {} could not be read ({e}); index rate has no effect",
            path.display()
        )),
    }
    out
}

impl SigilRvcEngine {
    /// Whether the load plans the request's lookahead and stream geometry (`SIGIL_VC_FINAL_SHAPES`,
    /// on unless `0`): a caller whose load did not (the switch is off) sets the lookahead afterwards.
    pub fn final_shapes_enabled() -> bool {
        env_switch(FINAL_SHAPES_ENV, true)
    }

    /// Whether a load runs its branches beside each other (`SIGIL_VC_PARALLEL_LOAD`).
    pub fn parallel_load_enabled() -> bool {
        env_switch(PARALLEL_LOAD_ENV, PARALLEL_LOAD_DEFAULT)
    }

    /// Load the voice: template, weights, shared encoder, pitch cache, generator sessions, warm-up
    /// and graph capture at the plan's shapes.
    pub fn load(request: SigilRvcRequest) -> Result<SigilRvcEngine, String> {
        Self::load_with(request, LoadOptions::default())
    }

    /// [`load`](Self::load) as the voice changer's factory runs it: the pitch extractor is built by
    /// `pitch` beside the other branches (the request's own extractor is a placeholder; `pitch`
    /// gets the audio frames of one incremental F0 call and the whole window's length in 16 kHz
    /// samples, which size RMVPE's plan), a set `cancel` stops the load at its next stage, and
    /// `harness_warmup_follows` says the stream harness warms the voice next (one validation
    /// conversion here instead of two), and `background` marks a hot-switch load beside a playing
    /// voice (sequential branches).
    pub fn load_hooked<'a>(
        request: SigilRvcRequest,
        pitch: Option<PitchBuilder<'a>>,
        cancel: Option<&'a AtomicBool>,
        harness_warmup_follows: bool,
        background: bool,
    ) -> Result<SigilRvcEngine, String> {
        // A background hot-switch load (another voice plays) builds its branches one after the other
        // unless the environment asks for the parallel load explicitly.
        let forced_parallel = matches!(std::env::var(PARALLEL_LOAD_ENV).as_deref().map(str::trim), Ok("1" | "on" | "true" | "yes"));
        Self::load_full(
            request,
            LoadOptions::default(),
            LoadParams {
                pitch,
                cancel,
                harness_warmup_follows,
                parallel: (background && !forced_parallel).then_some(false),
                ..LoadParams::default()
            },
        )
    }

    /// A `PitchExtractor` to put in a request whose real extractor `load_hooked`'s builder makes.
    pub fn pending_pitch() -> Box<dyn PitchExtractor> {
        Box::new(PendingPitch)
    }

    pub(super) fn load_with(request: SigilRvcRequest, load_opts: LoadOptions) -> Result<SigilRvcEngine, String> {
        Self::load_full(request, load_opts, LoadParams::default())
    }

    pub(super) fn load_full(request: SigilRvcRequest, load_opts: LoadOptions, mut params: LoadParams<'_>) -> Result<SigilRvcEngine, String> {
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
            lookahead_ms,
            stream,
            content_context_16k,
        } = request;
        let fail = |e: String| load_err(&slot_dir, e);
        let cancel = params.cancel;
        let cancelled = || cancel.is_some_and(|c| c.load(Ordering::Relaxed));
        if cancelled() {
            return Err(CANCELLED.into());
        }
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
                .unwrap_or(super::super::import::WEIGHTS_FILE),
        )
        .map_err(fail)?;
        let speakers_file = slot_file(&slot_dir, manifest.speakers.as_deref().unwrap_or(SPEAKERS_FILE)).map_err(fail)?;
        let speakers = SpeakerTable::load(&speakers_file, layout.speakers.dim).map_err(fail)?;
        if let Some(n) = manifest
            .speaker_count
            .filter(|&n| n as usize != speakers.count())
        {
            warnings.push(format!("manifest.json lists {n} speakers, the speaker table holds {}", speakers.count()));
        }
        let embedder = EmbedderKind::from_manifest(manifest.embedder.as_deref(), arch.emb_channels).map_err(fail)?;
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

        // ---- The shapes the voice runs: the request's lookahead and the harness's geometry. ----
        let partial_decode = levers.levers.get(Lever::PartialDecode);
        let final_shapes = params.final_shapes.unwrap_or_else(Self::final_shapes_enabled);
        let (plan_lookahead, plan_stream) = if final_shapes { (lookahead_frames_of(lookahead_ms), stream.as_ref()) } else { (0, None) };
        let mut plan = load_plan(&layout.streaming, &preset, out_sr, partial_decode, plan_lookahead, plan_stream).map_err(fail)?;

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

        // ---- The content encoder's request (the engine's own embedder wins over the settings'). ----
        let explicit = content_encoder
            .as_deref()
            .filter(|p| !p.as_os_str().is_empty());
        if explicit.is_some() && !matches!(embedder, EmbedderKind::ContentVec768 | EmbedderKind::ContentVec256) {
            warnings.push(format!(
                "the voice uses the {} content encoder; the HuBERT / ContentVec path from the settings is not used for it",
                embedder.as_str()
            ));
        }
        let index = index
            .or_else(|| manifest.index_path(&slot_dir))
            .filter(|p| p.is_file());
        let emb = arch.emb_channels as usize;
        let parallel = params.parallel.unwrap_or_else(Self::parallel_load_enabled);

        // DirectML: acquire the content encoder first. A dynamic-shape encoder costs 35-49 ms per
        // call with lookahead there, so the lookahead is dropped before the generator is built (free
        // now: no generator exists yet).
        let mut content_out: Option<ContentOut> = None;
        if ep == "dml" {
            let out = acquire_content(embedder, explicit, &slot_dir, &ep, &session, &levers, &plan, content_context_16k)?;
            if plan.lookahead_frames > 0 && !Self::content_shape_is_fixed(&out.encoder, &ep) {
                plan = load_plan(&layout.streaming, &preset, out_sr, partial_decode, 0, plan_stream).map_err(fail)?;
                levers
                    .notes
                    .push("lookahead: not used while the content encoder runs on DirectML (it would cost 35-49 ms per block there)".to_string());
            }
            content_out = Some(out);
        }
        if cancelled() {
            return Err(CANCELLED.into());
        }

        // ---- The branches (module docs): pitch bank, content encoder, retrieval index beside the
        // generator on this thread. ----
        let policy = ThreadPolicy::of_current();
        let pitch_builder = params.pitch.take();
        let f0_frames = plan.block_16k / HOP_16K + plan.f0_margin_frames + plan.f0_lead_frames;
        let window_16k = plan.context_16k;
        let build_gen = |plan: &ShapePlan, levers: &EffectiveLevers| -> Result<(SplitGenerator, Vec<String>), String> {
            let mut notes = Vec::new();
            let g = Self::build_generator(&template, &weights_file, plan, &ep, &session, levers, want_f16, parallel, &mut notes)?;
            Ok((g, notes))
        };
        let (pitch_res, generator_res, content_res, retrieval_out) = if parallel {
            let (plan_ref, levers_ref, ep_ref, session_ref, slot_ref) = (&plan, &levers, &ep, &session, &slot_dir);
            std::thread::scope(|scope| {
                let pitch_h = pitch_builder.map(|b| spawn_branch(scope, "sigil-vc-load-pitch", policy, move || b(f0_frames, window_16k)));
                // The content branch: acquire and prebuild the plan, or (DirectML, acquired above)
                // just prebuild.
                let acquired = content_out.take();
                let content_h = spawn_branch(scope, "sigil-vc-load-content", policy, move || match acquired {
                    Some(mut out) => {
                        out.prebuilt = prebuild_content_plan(&out.encoder, levers_ref, plan_ref, content_context_16k);
                        Ok(out)
                    }
                    None => acquire_content(embedder, explicit, slot_ref, ep_ref, session_ref, levers_ref, plan_ref, content_context_16k).map(|mut out| {
                        out.prebuilt = prebuild_content_plan(&out.encoder, levers_ref, plan_ref, content_context_16k);
                        out
                    }),
                });
                let retrieval_h = index.is_some().then(|| {
                    let index = index.clone();
                    spawn_branch(scope, "sigil-vc-load-index", policy, move || retrieval_branch(index, levers_ref, ep_ref, session_ref, emb))
                });
                // The generator builds on this thread.
                let generator = build_gen(plan_ref, levers_ref);
                let pitch_r = pitch_h.map(|h| join_branch(h, "pitch").and_then(|r| r));
                let content_r = join_branch(content_h, "content encoder").and_then(|r| r);
                let retrieval_r = retrieval_h.map(|h| join_branch(h, "index"));
                (pitch_r, generator, content_r, retrieval_r)
            })
        } else {
            // P5's order on this thread: pitch bank, generator, content encoder, retrieval.
            let pitch_r = pitch_builder.map(|b| b(f0_frames, window_16k));
            if let Some(Err(e)) = &pitch_r {
                return Err(fail(e.clone()));
            }
            // Each stage checks the flag, as each join does in the parallel order.
            if cancelled() {
                return Err(CANCELLED.into());
            }
            let generator = build_gen(&plan, &levers);
            if let Err(e) = &generator {
                return Err(fail(e.clone()));
            }
            if cancelled() {
                return Err(CANCELLED.into());
            }
            let content_r = match content_out.take() {
                Some(c) => Ok(c),
                None => acquire_content(embedder, explicit, &slot_dir, &ep, &session, &levers, &plan, content_context_16k),
            };
            let retrieval_r = index.is_some().then(|| Ok(retrieval_branch(index.clone(), &levers, &ep, &session, emb)));
            (pitch_r, generator, content_r, retrieval_r)
        };
        if cancelled() {
            return Err(CANCELLED.into());
        }
        // The first failure in P5's order: pitch bank, generator, content encoder.
        let pitch = match pitch_res {
            Some(r) => r.map_err(fail)?,
            None => pitch,
        };
        let (generator, generator_notes) = generator_res.map_err(fail)?;
        let ContentOut {
            encoder,
            warnings: content_warnings,
            prebuilt,
        } = content_res.map_err(fail)?;
        warnings.extend(content_warnings);
        let encoder_outcome = encoder
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .outcome()
            .clone();

        // Retrieval (P3-D backends behind one API).
        let mut retrieval = None;
        if let Some(r) = retrieval_out {
            let r = r.map_err(fail)?;
            levers.notes.extend(r.notes);
            warnings.extend(r.warnings);
            retrieval = r.retrieval;
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
        warnings.extend(generator_notes);
        warnings.extend(generator.notes().iter().cloned());

        let mut outcomes: Vec<(String, EpOutcome)> = generator.ep_outcomes().to_vec();
        outcomes.push(("vc.content".to_string(), encoder_outcome));
        let mut content_stats = ContentStats::default();
        if let Some((lease, report)) = prebuilt {
            content_stats.lease = Some(lease);
            note_content_plan(&mut outcomes, &mut warnings, report);
        }

        let f0_cache = Self::new_f0_cache(pitch.as_ref(), &plan, &levers, f0_range);
        let noise = AnchoredNoise::new(noise_seed);
        let upp = arch.upp as usize;
        let latent = arch.inter_channels as usize;
        let ring = plan.flow_frames + plan.block_16k / HOP_16K + 8;
        // The post stage's silence gate follows the silence skip (on by default, `set_silence_skip`).
        let post = post_stage(out_sr, true);
        let preset_extra_16k = ms_to_16k(preset.extra_context_ms);
        let stream_extra_16k = plan_stream.map_or(preset_extra_16k, |g| g.harness_extra_16k.max(preset_extra_16k));
        let mut engine = SigilRvcEngine {
            levers,
            outcomes,
            warnings,
            output_sample_rate: out_sr,
            slot_dir,
            weights_file,
            preset_extra_16k,
            stream_extra_16k,
            want_f16,
            embedder,
            emb,
            latent,
            upp,
            has_f0: arch.f0,
            f0_range,
            load_opts,
            requested_lookahead_ms: lookahead_ms,
            stream_geometry: stream,
            content_context_16k,
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
            probe: Default::default(),
            content_stats,
            f0_helper: Default::default(),
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
        // Conversions at the live shape: a template / encoder that loads but cannot run (a graph
        // whose inputs differ, an encoder with the wrong width) fails the load, not the first tick.
        let passes = if params.harness_warmup_follows { 1 } else { 2 };
        engine
            .warm_once(passes)
            .map_err(|e| load_err(&engine.slot_dir, format!("test conversion failed: {e}")))?;
        // The content encoder's plan is known now: DirectML pays 35-49 ms per call for a lookahead
        // window on a dynamic shape. (The shape check above predicts this; it only fires when the plan
        // was refused after all.)
        if engine.ep == "dml" && engine.plan.lookahead_frames > 0 && !engine.content_runs_fixed_shape() {
            let replanned = engine.set_lookahead_ms(0.0);
            replanned.map_err(|e| load_err(&engine.slot_dir, e))?;
            engine.levers.notes.push(
                "lookahead: not used while the content encoder runs on DirectML (it would cost 35-49 ms per block there)".to_string(),
            );
            engine
                .warm_once(passes)
                .map_err(|e| load_err(&engine.slot_dir, format!("test conversion failed: {e}")))?;
        }
        if let Some(note) = engine.f0_cache.decode_note() {
            let note = note.to_string();
            if !engine.warnings.contains(&note) {
                engine.warnings.push(note);
            }
        }
        engine.reset_stream();
        Ok(engine)
    }

    /// Whether `encoder` will run fixed-shape plans on `ep` (the plan path is on, the rung is a GPU
    /// and the encoder's I/O allows a plan): what a lookahead window needs on DirectML.
    fn content_shape_is_fixed(encoder: &SharedEncoder, ep: &str) -> bool {
        content::runs_fixed_shape(ep)
            && encoder
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .plan_blocker()
                .is_none()
    }

    #[allow(clippy::too_many_arguments)]
    pub(super) fn build_generator(
        template: &Template,
        weights_file: &Path,
        plan: &ShapePlan,
        ep: &str,
        session: &VcSessionOptions,
        levers: &EffectiveLevers,
        want_f16: bool,
        parallel: bool,
        warnings: &mut Vec<String>,
    ) -> Result<SplitGenerator, String> {
        let layout = &template.layout;
        let f32_image = {
            let _span = telemetry::load_span(LoadStage::WeightsRead);
            std::fs::read(weights_file).map_err(|e| format!("{}: {e}", weights_file.display()))?
        };
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
            let _span = telemetry::load_span(LoadStage::F16Image);
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
        // `enc` and `flowdec` build beside each other only when the engine load runs its branches
        // beside each other (`parallel`) and the generator's own switch allows it.
        SplitGenerator::load_branch(template, &images, plan, ep, session, levers, parallel)
        // The images drop here: ONNX Runtime copied the weights into its sessions.
    }

    /// The F0 cache of a plan: its lead and margin from the extractor's receptive field (else the
    /// plan's), and the voice's `f0_decode` lever and range (P3B-F: the cross-tick decoder).
    pub(super) fn new_f0_cache(pitch: &dyn PitchExtractor, plan: &ShapePlan, levers: &EffectiveLevers, f0_range: F0Range) -> F0Cache {
        let (lead, margin) = match pitch.receptive_frames() {
            Some((left, right)) => (left, right.max(1)),
            None => (plan.f0_lead_frames, plan.f0_margin_frames),
        };
        let mut cache = F0Cache::new(F0CacheConfig {
            margin_frames: margin,
            capacity_frames: plan.enc_frames * 2,
        })
        .with_lead_frames(lead);
        cache.set_decode(levers.levers.get(Lever::F0Decode), f0_range);
        cache
    }

    /// Size the per-tick buffers for the plan (no allocation per tick afterwards).
    pub(super) fn size_buffers(&mut self) {
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

    /// Lookahead (ms, rounded up to 10 ms frames, 0..=80): the returned audio ends that much before
    /// the newest input, so every returned sample (the SOLA overlap included) was decoded with
    /// future context. Adds exactly that much latency. Call before the stream harness is built
    /// (it sizes its context from the plan); rebuilds the sessions when the shapes change. The
    /// voice changer's factory plans the lookahead in the load instead (`SigilRvcRequest::lookahead_ms`).
    pub fn set_lookahead_ms(&mut self, ms: f64) -> Result<(), String> {
        let frames = lookahead_frames_of(ms);
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
            self.replan_for(plan, ReplanReason::Lookahead)?;
        }
        Ok(())
    }

    /// [`replan_for`](Self::replan_for) with no particular reason (tests, the benchmark).
    pub(super) fn replan(&mut self, plan: ShapePlan) -> Result<(), String> {
        self.replan_for(plan, ReplanReason::Other)
    }

    /// Switch to a new plan. The generator's sessions are rebuilt (weights re-read from the slot)
    /// only when its tensor shapes change ([`generator_shape`]); a plan that differs only in the
    /// returned span (another crossfade within the same frames) keeps them. A rebuild frees the old
    /// sessions first, so VRAM peaks at one generator (on a failure the old shapes are rebuilt and
    /// the voice keeps running as it was), and refreshes the EP outcomes, the GPU-lever corrections
    /// and the notes from the new session plans (plan T7: the EP in use is what the voice reports;
    /// `VoiceConversionModel::stream_report` hands them to the factory after the harness's
    /// `configure_stream`).
    pub(super) fn replan_for(&mut self, plan: ShapePlan, reason: ReplanReason) -> Result<(), String> {
        let _span = telemetry::load_span(LoadStage::Replan(reason));
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
                Self::parallel_load_enabled(),
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
                        Self::parallel_load_enabled(),
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
        self.f0_cache = Self::new_f0_cache(self.pitch.as_ref(), &plan, &self.levers, self.f0_range);
        self.size_buffers();
        self.reset_stream();
        Ok(())
    }

    /// Run `passes` conversions on a quiet tone at the plan's shape (the first with the F0 cache
    /// empty, the second its incremental shape); the stream state is reset afterwards by the caller.
    pub(super) fn warm_once(&mut self, passes: usize) -> Result<(), String> {
        let _span = telemetry::load_span(LoadStage::WarmOnce);
        let n = self.plan.context_16k;
        let tone = warm_tone(n + self.plan.block_16k);
        let want = self.plan.want_16k;
        for (k, start) in [0usize, self.plan.block_16k].into_iter().take(passes.max(1)).enumerate() {
            let ctx = ConversionCtx {
                mono_16k: &tone[start..start + n],
                new_frames: want,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: (k * self.plan.block_16k) as u64,
                reference: None,
                singing: Default::default(),
            };
            let out = self.convert_inner(&ctx, false)?;
            if out.iter().any(|x| !x.is_finite()) {
                return Err("the voice produced non-finite audio".into());
            }
        }
        Ok(())
    }

    /// The body of `VoiceConversionModel::configure_stream` (the trait impl in `mod.rs` calls it):
    /// re-plan for the harness's block, returned span (`want`) and extra context; returns the
    /// context length. A load that planned the harness's geometry (`SigilRvcRequest::stream`)
    /// finds its plan equal and rebuilds nothing.
    pub(super) fn configure_stream_shapes(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
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
            self.replan_for(plan, ReplanReason::ConfigureStream)?;
        }
        Ok(self.plan.context_16k)
    }
}

/// The generator's tensor shapes in a plan: two plans with the same shapes run the same sessions.
pub(super) fn generator_shape(plan: &ShapePlan) -> (usize, usize, usize, usize) {
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
