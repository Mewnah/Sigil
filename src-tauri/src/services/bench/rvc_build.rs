//! Real voices for benchmark runs (P3-F): an `rvc` engine spec resolved with the voice changer's
//! own rules and built the way the voice changer builds it.
//!
//! Resolution ([`plan`]) uses `voice_changer::params` and `voice_changer::capabilities` directly
//! (no mirrored copies): the preset's default EP, the EP ladder rung of an accelerator id
//! (`tensorrt_rtx` runs on the CUDA rung, `dml_hp` on DirectML), the F0 method's stand-in and
//! GPU-only fallback, the lever resolution (request + availability + the preset's
//! benchmark-decided `auto` table), the ONNX Runtime thread policy of the worker priority.
//!
//! Building ([`build`]):
//! - `build: auto` / `factory`: `voice_changer::factory::create_engine`, the engine factory the
//!   worker uses, so the benchmark measures exactly what the user runs.
//! - `build: direct` (and `auto` with `lookahead_ms`): the benchmark builds the model itself with
//!   the same parts: the legacy `RvcModel` for bring-your-own ONNX slots, `SigilRvcEngine` for
//!   template voices (`sigil-rvc-1`), wrapped to report its per-component compute breakdown.
//!
//! Every session is recorded under the `bench.` component prefix, so a benchmark never overwrites
//! the voice changer's EP diagnostics rows.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use crate::services::diagnostics::{self, EpOutcome};
use crate::services::neural_audio::denoise::{self, DenoiserKind};
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::bases::{resolve_fcpe, resolve_hubert, resolve_rmvpe};
use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, F0Range, Lever, LeverMode, LeverRequest, PrecisionMode};
use crate::services::neural_audio::vc::engine::plan::MAX_CONTEXT_FRAMES;
use crate::services::neural_audio::vc::engine::presets::{engine_preset, EnginePreset};
use crate::services::neural_audio::vc::engine::{SigilRvcEngine, SigilRvcRequest, TickStats};
use crate::services::neural_audio::gpu::GpuLevers;
use crate::services::neural_audio::vc::f0_fusion::{create_pitch_extractor_with, PitchBankRequest};
use crate::services::neural_audio::vc::manifest::{load_slot_manifest, SlotFormat};
use crate::services::neural_audio::vc::traits::{ConversionCtx, PitchExtractor, VoiceControls};
use crate::services::neural_audio::{RvcModel, RvcPresetProfile, VoiceConversionModel};
use crate::services::rt_thread::WorkerPriority;
use crate::services::voice_changer::capabilities::resolve_levers;
use crate::services::voice_changer::factory::{self, EngineRequest};
use crate::services::voice_changer::params::{base_ep, effective_pitch_extractor, normalize_execution_provider, normalize_pitch_extractor, pitch_fallback_reason, requested_rvc_ep};

use super::engine::{lock, EngineInfo};
use super::spec::{CeilingSpec, RvcSpec, StreamSpec};

/// Session component prefix of every benchmark session.
const BENCH_COMPONENT_PREFIX: &str = "bench";
/// The legacy loader's components (as recorded with the prefix).
const RVC_COMPONENTS: [&str; 3] = ["bench.vc", "bench.vc.content", "bench.vc.pitch"];

/// Everything the voice changer's rules resolve from an `rvc` engine spec (and what the ceiling
/// and the autotuner rebuild variants from).
#[derive(Clone, Debug)]
pub(crate) struct RvcPlan {
    pub spec: RvcSpec,
    pub slot: PathBuf,
    pub format: SlotFormat,
    pub preset: RvcPresetProfile,
    /// The EP id asked for (after the preset default; may be an accelerator).
    pub requested_ep: String,
    /// The ladder rung sessions are built on.
    pub ep: &'static str,
    pub pitch_chosen: &'static str,
    /// The F0 method that runs.
    pub pitch: String,
    pub pitch_note: Option<String>,
    pub levers: EffectiveLevers,
    /// The spec's `anchored_noise` mode (`levers` holds its resolution; a bring-your-own ONNX voice
    /// anchors its noise only on an explicit `on`, as in the voice changer).
    pub anchored_noise_mode: LeverMode,
    pub precision: PrecisionMode,
    pub f0_range: F0Range,
    /// Engine block (ms; the stream's device block when the spec gives none).
    pub block_ms: f64,
    pub extra_context_ms: f64,
    pub chunk_frames: u32,
    pub device_rate: u32,
    pub session: VcSessionOptions,
    pub app_data: Option<PathBuf>,
    pub noise_seed: u64,
}

/// Resolve `spec` for a stream (see the module docs).
pub(crate) fn plan(spec: &RvcSpec, stream: &StreamSpec, priority: WorkerPriority) -> Result<RvcPlan, String> {
    let slot = spec.slot_dir.clone();
    if !slot.is_dir() {
        return Err(format!("RVC slot folder not found: {}", slot.display()));
    }
    let format = factory::slot_format(&slot)?;
    if let SlotFormat::Unknown(f) = &format {
        return Err(format!("{}: slot format '{f}' is not a voice this build runs", slot.display()));
    }
    let preset = RvcPresetProfile::from_str(&spec.preset);
    let requested_ep = requested_rvc_ep(&normalize_execution_provider(&spec.execution_provider), preset).to_string();
    let ep = base_ep(&requested_ep);
    let pitch_chosen = normalize_pitch_extractor(&spec.pitch_extractor);
    let (pitch, _) = effective_pitch_extractor(&spec.pitch_extractor, ep);
    let pitch_note = pitch_fallback_reason(&spec.pitch_extractor, ep);
    let mut request = LeverRequest::default();
    for (name, mode) in &spec.levers {
        if let Some(lever) = Lever::from_name(name.trim()) {
            request.set(lever, LeverMode::from_str_lossy(mode));
        }
    }
    let precision = PrecisionMode::from_str_lossy(&spec.precision);
    let levers = resolve_levers(&request, preset, &requested_ep, precision);
    let session = VcSessionOptions {
        cuda_device_index: spec.gpu_device_index,
        streamer_profile: preset == RvcPresetProfile::Streamer,
        intra_threads: spec.intra_threads.or((priority == WorkerPriority::ProtectGame).then_some(2)),
        thread_priority: Some(priority),
        vram_cap_mb: if ep == "cuda" { spec.vram_cap_mb } else { None },
        component_prefix: Some(BENCH_COMPONENT_PREFIX),
        // Same CPU allocator as the voice changer it measures.
        cpu_arena: false,
    };
    // The engine factory's seed rule (the voice's source hash, else its folder name), so a direct
    // build and the offline ceiling draw the same time-anchored noise as the factory's engine.
    let noise_seed = spec.noise_seed.unwrap_or_else(|| {
        let sha = load_slot_manifest(&slot).ok().and_then(|m| m.source_sha256);
        factory::noise_seed_for(&slot, sha.as_deref())
    });
    Ok(RvcPlan {
        spec: spec.clone(),
        slot,
        format,
        preset,
        requested_ep,
        ep,
        pitch_chosen,
        pitch,
        pitch_note,
        levers,
        anchored_noise_mode: request.get(Lever::AnchoredNoise),
        precision,
        f0_range: F0Range::from_str_lossy(&spec.f0_range),
        block_ms: if spec.block_ms > 0.0 { spec.block_ms } else { stream.block_ms() },
        extra_context_ms: stream.extra_context_ms,
        chunk_frames: stream.block_frames,
        device_rate: stream.input_rate,
        session,
        app_data: diagnostics::app_data_dir(),
        noise_seed,
    })
}

/// A built voice.
pub(crate) struct BuiltVoice {
    pub model: Box<dyn VoiceConversionModel>,
    pub info: EngineInfo,
    pub plan: RvcPlan,
    /// Latest tick breakdown (direct template voices).
    pub stats: Option<Arc<Mutex<Option<TickStats>>>>,
}

fn explicit(p: &str) -> Option<PathBuf> {
    let t = p.trim();
    (!t.is_empty()).then(|| PathBuf::from(t))
}

/// The EP a set of sessions landed on: a component that fell back wins, else `prefer` (the
/// generator), else any.
fn active_outcome(outcomes: &BTreeMap<String, EpOutcome>, prefer: &[&str]) -> Option<EpOutcome> {
    outcomes
        .values()
        .find(|o| o.fell_back())
        .or_else(|| prefer.iter().find_map(|p| outcomes.iter().find(|(k, _)| k.ends_with(p)).map(|(_, v)| v)))
        .or_else(|| outcomes.values().next())
        .cloned()
}

fn base_info(p: &RvcPlan, format: &str, output_rate: u32, min_context: usize) -> EngineInfo {
    let mut info = EngineInfo::new("rvc", 16_000, output_rate, min_context);
    info.format = Some(format.to_string());
    info.execution_provider = Some(p.ep.to_string());
    info.execution_provider_requested = Some(p.requested_ep.clone());
    info.pitch_extractor = Some(p.pitch.clone());
    info.pitch_extractor_requested = Some(p.pitch_chosen.to_string());
    info.f0_range = Some(p.f0_range.as_str().to_string());
    if let Some(n) = &p.pitch_note {
        info.warnings.push(n.clone());
    }
    info
}

fn finish_info(info: &mut EngineInfo, p: &RvcPlan, outcomes: BTreeMap<String, EpOutcome>, prefer: &[&str]) {
    let (active, reason) = match active_outcome(&outcomes, prefer) {
        Some(o) => (o.active, o.reason),
        None => (p.ep.to_string(), None),
    };
    info.execution_provider_active = Some(active);
    info.execution_provider_reason = reason;
    info.ep_outcomes = outcomes;
    let voice = p
        .slot
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    info.description = format!(
        "{} {voice} ({} preset, {}, {}{})",
        if info.format.as_deref() == Some("sigil-rvc-1") { "SVE" } else { "RVC" },
        p.spec.preset,
        p.pitch,
        p.requested_ep,
        info.build.as_deref().map(|b| format!(", {b}")).unwrap_or_default()
    );
}

/// Build `spec` for `stream` (see the module docs).
pub(crate) fn build(spec: &RvcSpec, stream: &StreamSpec, priority: WorkerPriority) -> Result<BuiltVoice, String> {
    let p = plan(spec, stream, priority)?;
    let mode = spec.build.trim().to_ascii_lowercase();
    let direct = mode == "direct" || (mode == "auto" && spec.lookahead_ms > 0.0);
    let (denoiser, in_engine, mut notes) = denoiser_plan(&spec.denoiser, direct);
    let mut built = if direct { build_direct(&p)? } else { build_factory(&p, in_engine)? };
    notes.append(&mut built.info.warnings);
    built.info.warnings = notes;
    if denoiser != DenoiserKind::Off {
        built.info.description = format!("{} [denoiser {}]", built.info.description, denoiser.as_str());
    }
    Ok(built)
}

/// The denoiser a spec asks for (after this build's availability rule, as the voice changer
/// resolves it), the one that runs inside the measured engine (`gtcrn` on factory builds, as in the
/// voice changer's RVC mode), and the notes for what the run leaves out.
fn denoiser_plan(requested: &str, direct: bool) -> (DenoiserKind, DenoiserKind, Vec<String>) {
    let (kind, note) = denoise::effective_kind(DenoiserKind::from_str_lossy(requested));
    let mut notes: Vec<String> = note.into_iter().map(|n| format!("denoiser: {n}")).collect();
    let in_engine = match kind {
        DenoiserKind::Off => DenoiserKind::Off,
        DenoiserKind::Gtcrn if !direct => DenoiserKind::Gtcrn,
        DenoiserKind::Gtcrn => {
            notes.push("denoiser gtcrn: runs inside the engine on factory builds only; this direct build excludes it".into());
            DenoiserKind::Off
        }
        other => {
            notes.push(format!(
                "denoiser {}: the voice changer runs it on the capture's device-rate copy, outside the engine; this run excludes its cost and latency",
                other.as_str()
            ));
            DenoiserKind::Off
        }
    };
    (kind, in_engine, notes)
}

fn build_factory(p: &RvcPlan, denoiser: DenoiserKind) -> Result<BuiltVoice, String> {
    let request = EngineRequest {
        slot_dir: p.slot.clone(),
        app_data: p.app_data.clone(),
        templates_dir: None,
        hubert_path: p.spec.hubert.clone(),
        fcpe_path: p.spec.fcpe.clone(),
        rmvpe_path: p.spec.rmvpe.clone(),
        pitch_extractor: p.pitch.clone(),
        ep: p.ep.to_string(),
        session: p.session.clone(),
        levers: p.levers.clone(),
        anchored_noise_mode: p.anchored_noise_mode,
        silence_skip: p.spec.silence_skip,
        preset: p.preset,
        precision: p.precision,
        block_ms: p.spec.block_ms,
        chunk_frames: p.chunk_frames,
        device_rate: p.device_rate,
        extra_context_ms: p.extra_context_ms,
        // 0 today: spec validation sends lookahead runs to the direct build (component breakdown).
        lookahead_ms: p.spec.lookahead_ms,
        f0_range: p.f0_range,
        speaker_id: 0,
    };
    let since = unix_ms();
    let created = if denoiser == DenoiserKind::Off {
        factory::create_engine(request)?
    } else {
        factory::create_engine_with(request, denoiser)?
    };
    let format = match &created.format {
        SlotFormat::SigilRvc => "sigil-rvc-1",
        _ => "rvc",
    };
    let mut info = base_info(p, format, created.output_sample_rate, created.model.min_context_samples());
    info.build = Some("factory".into());
    // The in-engine denoiser delays the audio too (the worker counts both).
    info.model_delay_ms = (created.model_delay_16k + created.denoiser_latency_16k) as f64 / 16.0;
    info.levers = Some(created.levers.clone());
    info.warnings.extend(created.warnings.iter().cloned());
    let mut outcomes: BTreeMap<String, EpOutcome> = if created.format == SlotFormat::SigilRvc {
        created.outcomes.iter().cloned().collect()
    } else {
        // The legacy generator and content encoder return no outcome (only the pitch bank does):
        // every session's row comes from the diagnostics log, where the pitch rows are too.
        legacy_outcomes_since(since)
    };
    if created.format == SlotFormat::SigilRvc {
        add_recorded_outcomes(&mut outcomes, since);
    }
    finish_info(&mut info, p, outcomes, &["vc.flowdec", "vc"]);
    Ok(BuiltVoice {
        model: created.model,
        info,
        plan: p.clone(),
        stats: None,
    })
}

fn build_direct(p: &RvcPlan) -> Result<BuiltVoice, String> {
    match p.format {
        SlotFormat::SigilRvc => {
            let since = unix_ms();
            let (engine, notes) = build_sve(p, None)?;
            let mut info = base_info(p, "sigil-rvc-1", engine.output_sample_rate(), engine.min_context_samples());
            info.build = Some("direct".into());
            info.warnings.extend(notes);
            info.warnings.extend(engine.warnings().iter().cloned());
            info.levers = Some(engine.effective_levers().clone());
            info.embedder = Some(engine.embedder().as_str().to_string());
            info.flowdec_precision = Some(engine.flowdec_precision().to_string());
            let plan = engine.plan();
            info.plan = Some(serde_json::json!({
                "block_16k": plan.block_16k,
                "context_16k": plan.context_16k,
                "enc_frames": plan.enc_frames,
                "flow_frames": plan.flow_frames,
                "decode_frames": plan.decode_frames,
                "lookahead_frames": plan.lookahead_frames,
                "partial_decode": plan.partial_decode,
            }));
            info.model_delay_ms = plan.lookahead_frames as f64 * 10.0;
            let mut outcomes: BTreeMap<String, EpOutcome> = engine.ep_outcomes().iter().cloned().collect();
            add_recorded_outcomes(&mut outcomes, since);
            finish_info(&mut info, p, outcomes, &["vc.flowdec", "vc.enc"]);
            let stats = Arc::new(Mutex::new(None));
            Ok(BuiltVoice {
                model: Box::new(SveProbe {
                    inner: engine,
                    last: stats.clone(),
                }),
                info,
                plan: p.clone(),
                stats: Some(stats),
            })
        }
        _ => {
            let since = unix_ms();
            let model = build_legacy(p)?;
            let mut info = base_info(p, "rvc", model.output_sample_rate(), model.min_context_samples());
            info.build = Some("direct".into());
            info.warnings.extend(model.warnings().iter().cloned());
            info.levers = Some(EffectiveLevers {
                notes: vec!["bring-your-own ONNX slot: the engine levers apply to template voices".into()],
                ..EffectiveLevers::default()
            });
            finish_info(&mut info, p, legacy_outcomes_since(since), &["bench.vc"]);
            Ok(BuiltVoice {
                model: Box::new(model),
                info,
                plan: p.clone(),
                stats: None,
            })
        }
    }
}

fn unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// The legacy loader's EP outcomes recorded since `since` (unix ms).
/// Add the template voice's sessions its builder did not report from the diagnostics rows recorded under the benchmark's prefix since
/// `since`, keyed like the engine's own (`vc.pitch`, ...), so the report says where every
/// component ran.
fn add_recorded_outcomes(outcomes: &mut BTreeMap<String, EpOutcome>, since: u64) {
    merge_recorded(outcomes, diagnostics::ep_outcomes().into_iter().filter(|e| e.at_unix_ms >= since).map(|e| (e.component, e.outcome)));
}

fn merge_recorded(outcomes: &mut BTreeMap<String, EpOutcome>, recorded: impl Iterator<Item = (String, EpOutcome)>) {
    let prefix = format!("{BENCH_COMPONENT_PREFIX}.");
    for (component, outcome) in recorded {
        if let Some(bare) = component.strip_prefix(&prefix).filter(|c| c.starts_with("vc.")) {
            outcomes.entry(bare.to_string()).or_insert(outcome);
        }
    }
}

fn legacy_outcomes_since(since: u64) -> BTreeMap<String, EpOutcome> {
    diagnostics::ep_outcomes()
        .into_iter()
        .filter(|e| e.at_unix_ms >= since && RVC_COMPONENTS.contains(&e.component.as_str()))
        .map(|e| (e.component, e.outcome))
        .collect()
}

/// The pitch extractor of a plan (the pitch bank; explicit FCPE / RMVPE files, else the slot and
/// `base/` candidates, else the model store), with the voice's GPU levers: on CUDA, IOBinding plus
/// CUDA graphs run RMVPE as a graph plan (as the engine factory does since P3-I0).
fn pitch_for(p: &RvcPlan) -> Result<(Box<dyn PitchExtractor>, Vec<String>), String> {
    let ad = p.app_data.as_deref();
    let fcpe = explicit(&p.spec.fcpe).or_else(|| ad.and_then(|a| resolve_fcpe("", a, &p.slot)));
    let rmvpe = explicit(&p.spec.rmvpe).or_else(|| ad.and_then(|a| resolve_rmvpe("", a, &p.slot)));
    let bank = create_pitch_extractor_with(
        &PitchBankRequest {
            method: &p.pitch,
            ep: p.ep,
            opts: &p.session,
            range: p.f0_range,
            fcpe_path: fcpe.as_deref(),
            rmvpe_path: rmvpe.as_deref(),
            app_data: ad,
            small_models_on_cpu: true,
        },
        GpuLevers::from_effective(&p.levers),
    )?;
    Ok((bank.extractor, bank.notes))
}

fn build_legacy(p: &RvcPlan) -> Result<RvcModel, String> {
    let hubert = match explicit(&p.spec.hubert) {
        Some(h) if h.is_file() => h,
        Some(h) => return Err(format!("HuBERT / ContentVec ONNX not found: {}", h.display())),
        None => p
            .app_data
            .as_deref()
            .and_then(|a| resolve_hubert("", a, &p.slot))
            .ok_or("HuBERT / ContentVec ONNX not found: set its path in the spec, or place it under app_data/voice-changer/base")?,
    };
    let (pitch, _) = pitch_for(p)?;
    RvcModel::load(&p.slot, &hubert.to_string_lossy(), pitch, p.ep, &p.session, p.spec.silence_skip)
}

/// A template voice through `SigilRvcEngine::load` with the plan's (or `preset_override`'s)
/// shapes; returns the engine and load notes.
fn build_sve(p: &RvcPlan, preset_override: Option<EnginePreset>) -> Result<(SigilRvcEngine, Vec<String>), String> {
    let manifest = load_slot_manifest(&p.slot)?;
    let templates_dir = crate::services::neural_audio::vc::import::templates_dir()
        .ok_or("the bundled vc-templates folder was not found")?;
    let (pitch, notes) = pitch_for(p)?;
    let preset = preset_override.unwrap_or_else(|| {
        engine_preset(p.preset, p.block_ms, p.chunk_frames, p.device_rate, p.extra_context_ms, p.precision)
    });
    let mut engine = SigilRvcEngine::load(SigilRvcRequest {
        slot_dir: p.slot.clone(),
        manifest,
        templates_dir,
        ep: p.ep.to_string(),
        session: p.session.clone(),
        levers: p.levers.clone(),
        pitch,
        content_encoder: explicit(&p.spec.hubert),
        index: explicit(&p.spec.index),
        preset,
        f0_range: p.f0_range,
        noise_seed: p.noise_seed,
    })?;
    engine.set_silence_skip(p.spec.silence_skip);
    if preset_override.is_none() && p.spec.lookahead_ms > 0.0 {
        engine.set_lookahead_ms(p.spec.lookahead_ms)?;
    }
    Ok((engine, notes))
}

/// The offline ceiling's instance of a template voice: `cfg`'s block and lookahead with
/// `cfg.extra_context_ms` of extra context (never less than the run's own; 0 = the longest the
/// engine's plan allows), silence skip off, the same precision, EP and noise seed as the run, and
/// its levers without the streaming approximations when `cfg.exact` ([`ceiling_levers`]).
pub(crate) fn ceiling_engine(p: &RvcPlan, cfg: &CeilingSpec) -> Result<SigilRvcEngine, String> {
    let mut p = p.clone();
    p.levers = ceiling_levers(&p.levers, cfg.exact);
    let p = &p;
    let block_frames = ((cfg.block_ms / 10.0 - 1e-9).ceil() as usize).div_ceil(2) * 2;
    let max_extra_ms = (MAX_CONTEXT_FRAMES.saturating_sub(block_frames + 2) * 10) as f64;
    let extra = if cfg.extra_context_ms > 0.0 {
        cfg.extra_context_ms.max(p.extra_context_ms).min(max_extra_ms)
    } else {
        max_extra_ms
    };
    let preset = EnginePreset {
        profile: p.preset,
        block_ms: cfg.block_ms,
        extra_context_ms: extra,
        precision: p.precision,
    };
    let (mut engine, _) = build_sve(p, Some(preset))?;
    engine.set_silence_skip(false);
    engine.set_lookahead_ms(cfg.lookahead_ms)?;
    // One shape: exactly one block returned per call (no SOLA tail), the whole context.
    let block = engine.plan().block_16k;
    let extra_16k = (extra * 16.0).round() as usize;
    engine.configure_stream(block, block, extra_16k)?;
    Ok(engine)
}

/// The ceiling's levers: the run's, minus partial decode when `exact` (it trades decoder context
/// for compute; an offline conversion decodes everything). The F0 cache stays (each frame's F0 is
/// computed once, so adjacent blocks agree at their joins; see `CeilingSpec::exact`), and so do
/// anchored noise and phase0: they define the rendition, not its cost.
pub(crate) fn ceiling_levers(run: &EffectiveLevers, exact: bool) -> EffectiveLevers {
    let mut l = run.clone();
    if exact && l.levers.get(Lever::PartialDecode) {
        l.levers.set(Lever::PartialDecode, false);
        l.notes.push("ceiling: partial_decode off (exact)".to_string());
    }
    l
}

/// A template voice that reports each tick's compute breakdown.
struct SveProbe {
    inner: SigilRvcEngine,
    last: Arc<Mutex<Option<TickStats>>>,
}

impl VoiceConversionModel for SveProbe {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let r = self.inner.convert(ctx);
        *lock(&self.last) = Some(self.inner.tick_stats());
        r
    }

    fn input_sample_rate(&self) -> u32 {
        self.inner.input_sample_rate()
    }

    fn output_sample_rate(&self) -> u32 {
        self.inner.output_sample_rate()
    }

    fn min_context_samples(&self) -> usize {
        self.inner.min_context_samples()
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        self.inner.warmup_at(ctx)
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        self.inner.set_voice_controls(controls);
    }

    fn apply_quality(&mut self, level: &QualityLevel) -> bool {
        self.inner.apply_quality(level)
    }

    fn reset_stream(&mut self) {
        self.inner.reset_stream();
    }

    fn fixed_block(&self) -> Option<usize> {
        self.inner.fixed_block()
    }

    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        self.inner.configure_stream(block, want, extra)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rvc(v: serde_json::Value) -> RvcSpec {
        serde_json::from_value(v).unwrap()
    }

    /// The voice changer's rules, not a copy: accelerator ids run on their rung, a stand-in F0
    /// method runs with a note, levers resolve through `resolve_levers`.
    #[test]
    fn plans_follow_the_voice_changer_rules() {
        let (root, slot, hubert) = crate::services::neural_audio::vc::test_models::write_slot("bench-rvcplan");
        let stream = StreamSpec::default();
        let p = plan(
            &rvc(serde_json::json!({"slot_dir": slot, "hubert": hubert.to_string_lossy(), "execution_provider": "dml_hp",
                                    "pitch_extractor": "rmvpe", "levers": {"partial_decode": "off"}})),
            &stream,
            WorkerPriority::Auto,
        )
        .unwrap();
        assert_eq!((p.requested_ep.as_str(), p.ep), ("dml_hp", "dml"));
        assert_eq!(p.format, SlotFormat::LegacyOnnxRvc);
        assert!(!p.levers.levers.get(Lever::PartialDecode), "off is always off");
        assert!((p.block_ms - stream.block_ms()).abs() < 1e-9, "block from the device block");
        // rmvpe on the CPU rung runs as fcpe, with the voice changer's wording.
        let cpu = plan(
            &rvc(serde_json::json!({"slot_dir": slot, "execution_provider": "cpu", "pitch_extractor": "rmvpe"})),
            &stream,
            WorkerPriority::Auto,
        )
        .unwrap();
        assert_eq!(cpu.pitch, "fcpe");
        assert!(cpu.pitch_note.as_deref().unwrap().contains("GPU-only"), "{:?}", cpu.pitch_note);
        // studio_trt asks for TensorRT-RTX on the CUDA rung.
        let trt = plan(&rvc(serde_json::json!({"slot_dir": slot, "preset": "studio_trt"})), &stream, WorkerPriority::Auto).unwrap();
        assert_eq!((trt.requested_ep.as_str(), trt.ep), ("tensorrt_rtx", "cuda"));
        let _ = std::fs::remove_dir_all(root);
    }

    /// GTCRN runs inside factory-built engines; every other denoiser (and GTCRN on a direct build)
    /// is noted as excluded.
    #[test]
    fn denoisers_run_where_the_voice_changer_runs_them() {
        let (kind, inside, notes) = denoiser_plan("off", false);
        assert_eq!((kind, inside), (DenoiserKind::Off, DenoiserKind::Off));
        assert!(notes.is_empty());
        let (kind, inside, notes) = denoiser_plan("gtcrn", false);
        if kind == DenoiserKind::Gtcrn {
            assert_eq!(inside, DenoiserKind::Gtcrn);
            assert!(notes.is_empty(), "{notes:?}");
            let (_, inside, notes) = denoiser_plan("gtcrn", true);
            assert_eq!(inside, DenoiserKind::Off);
            assert!(notes.iter().any(|n| n.contains("direct")), "{notes:?}");
        } else {
            assert_eq!(inside, DenoiserKind::Off, "unavailable: off, with the availability note");
            assert!(!notes.is_empty());
        }
        let (kind, inside, notes) = denoiser_plan("dpdfnet", false);
        assert_eq!(inside, DenoiserKind::Off, "the capture-copy denoisers never run inside the engine");
        if kind != DenoiserKind::Off {
            assert!(notes.iter().any(|n| n.contains("outside the engine")), "{notes:?}");
        }
    }

    #[test]
    fn active_outcome_prefers_a_fallback() {
        let mut m = BTreeMap::new();
        m.insert("bench.vc.enc".to_string(), EpOutcome::new("cuda", "cuda", None));
        m.insert("bench.vc.flowdec".to_string(), EpOutcome::new("cuda", "cuda", None));
        assert_eq!(active_outcome(&m, &["vc.flowdec"]).unwrap().active, "cuda");
        m.insert("bench.vc.content".to_string(), EpOutcome::new("cuda", "cpu", Some("no driver".into())));
        let o = active_outcome(&m, &["vc.flowdec"]).unwrap();
        assert_eq!((o.active.as_str(), o.reason.as_deref()), ("cpu", Some("no driver")));
        assert!(active_outcome(&BTreeMap::new(), &[]).is_none());
    }

    /// Engine components the builder did not report are filled from the benchmark's diagnostics
    /// rows (bare keys, the builder's own rows win, other prefixes and non-engine rows are ignored).
    #[test]
    fn recorded_rows_fill_unreported_components() {
        let o = |active: &str| EpOutcome {
            requested: "cuda".into(),
            active: active.into(),
            reason: None,
        };
        let mut outcomes: BTreeMap<String, EpOutcome> = [("vc.enc".to_string(), o("cuda"))].into_iter().collect();
        merge_recorded(
            &mut outcomes,
            [
                ("bench.vc.pitch".to_string(), o("cuda")),
                ("bench.vc.enc".to_string(), o("cpu")),
                ("vc.flowdec".to_string(), o("cpu")),
                ("bench.analysis.pitch".to_string(), o("cpu")),
            ]
            .into_iter(),
        );
        assert_eq!(outcomes.keys().collect::<Vec<_>>(), ["vc.enc", "vc.pitch"]);
        assert_eq!(outcomes["vc.enc"].active, "cuda", "the builder's own row wins");
    }

    /// The exact ceiling drops only partial decode; the F0 cache, the rendition-defining levers
    /// (anchored noise, phase0) and the GPU levers stay as the run has them.
    #[test]
    fn the_exact_ceiling_drops_only_partial_decode() {
        let mut run = EffectiveLevers::default();
        for lever in Lever::ALL {
            run.levers.set(lever, true);
        }
        let exact = ceiling_levers(&run, true);
        for lever in Lever::ALL {
            assert_eq!(exact.levers.get(lever), lever != Lever::PartialDecode, "{}", lever.as_str());
        }
        assert_eq!(exact.notes.len(), 1, "{:?}", exact.notes);
        let same = ceiling_levers(&run, false);
        assert!(Lever::ALL.iter().all(|&l| same.levers.get(l)) && same.notes.is_empty());
        // Nothing to drop: no note.
        assert!(ceiling_levers(&EffectiveLevers::default(), true).notes.is_empty());
    }
}
