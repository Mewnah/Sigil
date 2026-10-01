//! Engine factory (P3-E): turns the resolved params into the model behind the stream harness,
//! dispatching on the slot's `manifest.format` (plan: "Model families behind VoiceConversionModel
//! with a manifest.format factory"):
//! - `rvc` (bring-your-own ONNX, or no manifest): the legacy full-decode `RvcModel` (unchanged;
//!   the `anchored_noise` lever feeds its `rnd` input time-anchored noise, and an export with
//!   `skip_head` / `return_length` decodes only the returned frames);
//! - `sigil-rvc-1` (imported template voice): the Sigil Voice Engine (`vc::engine`, P3-C1);
//! - anything else: refused (later families register here, P5).
//!
//! The pitch extractor comes from the pitch bank factory (`vc::f0_fusion::create_pitch_extractor_with`,
//! P3-D) with the voice's GPU levers, so RMVPE (alone or inside the fusion) runs as the pitch bank's
//! fixed-shape plan: a CUDA graph with `io_binding` + `cuda_graphs` on (P3-I0). A `gtcrn` denoiser runs inside the harness: the model is wrapped in
//! `denoise::harness::DenoisedModel` (16 kHz, exactly the new samples of every call). Legacy voices
//! get the post stage's per-voice controls too (`engine::post`: rms_mix and the true formant shift;
//! a bit-exact pass-through with the neutral defaults), so every voice answers every control.
//!
//! The worker and the benchmark (P3-F) both build engines through [`create_engine`], so a
//! benchmark measures exactly what the user runs.
//!
//! **Governor steps** (`engine::governor`, the `governor` lever): the model applies the retrieval
//! step itself (template voices). With the lever on, the factory also builds the light F0 extractor
//! (SwiftF0 on the CPU, one thread) next to a heavier one (RMVPE, FCPE, CREPE, PESTO, fusion) and
//! wraps the model so the governor's `light_f0` step switches between them at the next tick
//! ([`GovernedModel`]; the F0 cache sizes its lead-in for both). The context step needs other
//! pre-planned shapes and is a no-op the governor skips.
//!
//! **Hot slots** ([`VoiceSet`]): the running voice plus up to `hot_slots - 1` recently used voices
//! stay loaded (within `vram_budget_mb` when set). Switching to a loaded voice happens on the next
//! block; any other voice loads on a background thread while the current one keeps playing, then
//! switches. Loads never overlap (each measures its own VRAM, and two never compete with the live
//! voice for the GPU): a voice asked for while another loads waits, the newest request wins, and a
//! load nobody wants any more stops at its next stage (before its sessions, its warm-up, its VRAM
//! claim). The worker crossfades the two voices over the switch block ([`crossfade_into`]) after
//! priming the incoming voice's context with the most recent input.
//!
//! # Frozen by P3-S0 (signatures; P3-E implements)
//! [`slot_format`] (implemented), [`EngineRequest`], [`CreatedEngine`], [`create_engine`].
//!
//! # Added by P3-E
//! [`create_engine_with`] (the in-harness denoiser), `CreatedEngine::{vram_mb,
//! denoiser_latency_16k}`, [`HarnessSpec`],
//! [`build_harness`], [`EngineKey`], [`LoadedVoice`], [`VoiceSet`], [`crossfade_into`],
//! [`noise_seed_for`].

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::denoise::{self, harness::DenoisedModel, DenoiserKind};
use crate::services::neural_audio::gpu::arena::VramProbe;
use crate::services::neural_audio::gpu::GpuLevers;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::engine::{self, presets, EffectiveLevers, F0Range, Lever, LeverMode, PrecisionMode, SigilRvcEngine, SigilRvcRequest};
use crate::services::neural_audio::vc::f0_fusion::{create_pitch_extractor, create_pitch_extractor_with, PitchBankRequest};
use crate::services::neural_audio::vc::manifest::{load_slot_manifest, SlotFormat};
use crate::services::neural_audio::vc::rvc::LegacyNoise;
use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::engine::post::{PostConfig, PostStage};
use crate::services::neural_audio::vc::traits::{ConversionCtx, PitchExtractor, StreamReport, VoiceControls, DEFAULT_NOISE_SCALE};
use crate::services::neural_audio::vc::{assets, import, VcStreamHarness, VoiceConversionModel};
use crate::services::neural_audio::{RvcModel, RvcPresetProfile};
use crate::services::resources::{self, Subsystem, VramClaim};

/// How the slot in `slot_dir` runs: its manifest's format, or a legacy ONNX slot when it has no
/// `manifest.json` (the manifest is then inferred from the generator, `bases.rs`).
pub(crate) fn slot_format(slot_dir: &Path) -> Result<SlotFormat, String> {
    if slot_dir.join("manifest.json").is_file() {
        load_slot_manifest(slot_dir).map(|m| m.slot_format())
    } else {
        Ok(SlotFormat::LegacyOnnxRvc)
    }
}

/// Everything the factory needs (owned; the benchmark builds one from its spec).
pub(crate) struct EngineRequest {
    pub slot_dir: PathBuf,
    pub app_data: Option<PathBuf>,
    /// Folder of the bundled templates (Tauri's resource dir); `None` = `import::default_templates_dir`.
    pub templates_dir: Option<PathBuf>,
    /// Explicit base-model paths from the params (`""` = automatic).
    pub hubert_path: String,
    pub fcpe_path: String,
    pub rmvpe_path: String,
    /// The F0 method that runs (after `params::effective_pitch_extractor`).
    pub pitch_extractor: String,
    /// EP ladder rung (`cpu` / `cuda` / `dml`).
    pub ep: String,
    pub session: VcSessionOptions,
    pub levers: EffectiveLevers,
    /// The user's `anchored_noise` mode (`levers` holds only its resolution): a bring-your-own ONNX
    /// voice anchors its noise only on an explicit `on` (`presets::byo_lever`; T8: `auto` keeps
    /// its export's proven zero noise).
    pub anchored_noise_mode: LeverMode,
    pub silence_skip: bool,
    pub preset: RvcPresetProfile,
    pub precision: PrecisionMode,
    pub block_ms: f64,
    pub chunk_frames: u32,
    pub device_rate: u32,
    pub extra_context_ms: f64,
    /// Lookahead of template voices (ms, 0 = off; `SigilRvcEngine::set_lookahead_ms`, P3-I0).
    /// Bring-your-own ONNX voices run without it (a lever note says so).
    pub lookahead_ms: f64,
    pub f0_range: F0Range,
    /// The speaker at load (each tick carries its own; kept for the benchmark's reports).
    #[allow(dead_code)]
    pub speaker_id: i64,
}

/// A loaded engine.
pub(crate) struct CreatedEngine {
    pub model: Box<dyn VoiceConversionModel>,
    pub format: SlotFormat,
    pub output_sample_rate: u32,
    /// Levers the engine really uses (legacy slots: the ones their export supports).
    pub levers: EffectiveLevers,
    /// EP outcome per session component.
    pub outcomes: Vec<(String, EpOutcome)>,
    /// Non-fatal load issues (shown as `rvc_warning`).
    pub warnings: Vec<String>,
    /// Process VRAM the load added (MiB; 0 on the CPU or when DXGI cannot tell).
    pub vram_mb: f64,
    /// Latency the in-harness denoiser adds (16 kHz samples; 0 without one).
    pub denoiser_latency_16k: usize,
    /// Delay the model itself adds (a template voice's lookahead; 16 kHz samples; 0 without), on
    /// top of the harness's algorithmic latency (P3-I0).
    pub model_delay_16k: usize,
}

/// Noise seed of a voice: fixed per voice (its source hash, else its folder name), so a restart
/// reproduces the same time-anchored noise.
pub(crate) fn noise_seed_for(slot_dir: &Path, source_sha256: Option<&str>) -> u64 {
    let key = source_sha256
        .map(str::to_string)
        .unwrap_or_else(|| slot_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
    // FNV-1a: stable across builds and platforms.
    key.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3))
}

fn opt_path(p: &str) -> Option<PathBuf> {
    let t = p.trim();
    (!t.is_empty()).then(|| PathBuf::from(t))
}

/// A built pitch extractor, its notes and its sessions' EP outcomes.
struct BuiltPitch {
    extractor: Box<dyn PitchExtractor>,
    notes: Vec<String>,
    outcomes: Vec<(String, EpOutcome)>,
    /// The governor's light-F0 switch, when a light extractor was built next to the voice's.
    light_f0: Option<Arc<AtomicBool>>,
    /// Why the governor's light-F0 step does nothing for this voice (a lever note).
    light_note: Option<String>,
}

/// F0 methods that are already the light one (the governor's `light_f0` step keeps them).
fn is_light_f0(method: &str) -> bool {
    matches!(method.trim(), "swiftf0" | "dio")
}

/// The pitch extractor of a request (pitch bank factory), its notes and EP outcomes; with the
/// `governor` lever on, also the light extractor its `light_f0` step switches to.
fn build_pitch(request: &EngineRequest) -> Result<BuiltPitch, String> {
    let fcpe = opt_path(&request.fcpe_path);
    let rmvpe = opt_path(&request.rmvpe_path);
    let bank = create_pitch_extractor_with(
        &PitchBankRequest {
            method: &request.pitch_extractor,
            ep: &request.ep,
            opts: &request.session,
            range: request.f0_range,
            fcpe_path: fcpe.as_deref(),
            rmvpe_path: rmvpe.as_deref(),
            app_data: request.app_data.as_deref(),
            small_models_on_cpu: true,
        },
        GpuLevers::from_effective(&request.levers),
    )?;
    let mut built = BuiltPitch {
        extractor: bank.extractor,
        notes: bank.notes,
        outcomes: bank.outcomes,
        light_f0: None,
        light_note: None,
    };
    if request.levers.levers.get(Lever::Governor) && !is_light_f0(bank.method) {
        // The light extractor records its session under its own diagnostics key, so the voice's
        // F0 row keeps showing the extractor that normally runs.
        let opts = VcSessionOptions {
            component_prefix: Some(request.session.component_prefix.unwrap_or("governor")),
            ..request.session.clone()
        };
        let light = create_pitch_extractor(&PitchBankRequest {
            method: "swiftf0",
            ep: &request.ep,
            opts: &opts,
            range: request.f0_range,
            fcpe_path: None,
            rmvpe_path: None,
            app_data: request.app_data.as_deref(),
            small_models_on_cpu: true,
        });
        match light {
            Ok(light) => {
                let mut light = light.extractor;
                // Its first run allocates; do it now, not on the tick the governor steps down.
                let tone: Vec<f32> = (0..3_200).map(|i| 0.05 * (i as f32 * 0.059).sin()).collect();
                let _ = light.f0(&tone);
                let (pitch, flag) = SwitchablePitch::new(built.extractor, light);
                built.extractor = Box::new(pitch);
                built.light_f0 = Some(flag);
            }
            Err(e) => {
                built.light_note = Some(format!(
                    "governor: the light F0 step needs SwiftF0 ({}: {e}); that step is skipped",
                    assets::PITCH_SWIFTF0
                ));
            }
        }
    }
    Ok(built)
}

/// The voice's F0 extractor and the governor's light one; the governor's `light_f0` step picks which
/// runs (through the flag [`GovernedModel`] sets). The receptive field is the wider of the two, so
/// the incremental F0 cache feeds enough lead-in for either (unknown when either is unknown: the
/// cache's conservative default).
struct SwitchablePitch {
    main: Box<dyn PitchExtractor>,
    light: Box<dyn PitchExtractor>,
    use_light: Arc<AtomicBool>,
    receptive: Option<(usize, usize)>,
}

impl SwitchablePitch {
    fn new(main: Box<dyn PitchExtractor>, light: Box<dyn PitchExtractor>) -> (SwitchablePitch, Arc<AtomicBool>) {
        let receptive = match (main.receptive_frames(), light.receptive_frames()) {
            (Some(a), Some(b)) => Some((a.0.max(b.0), a.1.max(b.1))),
            _ => None,
        };
        let flag = Arc::new(AtomicBool::new(false));
        (
            SwitchablePitch {
                main,
                light,
                use_light: flag.clone(),
                receptive,
            },
            flag,
        )
    }
}

impl PitchExtractor for SwitchablePitch {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        if self.use_light.load(Ordering::Relaxed) {
            self.light.f0(mono_16k)
        } else {
            self.main.f0(mono_16k)
        }
    }

    fn receptive_frames(&self) -> Option<(usize, usize)> {
        self.receptive
    }
}

/// A model whose voice has the governor's light F0 extractor ([`SwitchablePitch`]): the
/// `light_f0` step switches it (from the next tick; the F0 cache keeps its frames, so the contour
/// continues), the model applies the other steps itself.
struct GovernedModel {
    inner: Box<dyn VoiceConversionModel>,
    light_f0: Arc<AtomicBool>,
}

impl VoiceConversionModel for GovernedModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        self.inner.convert(ctx)
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
        let model = self.inner.apply_quality(level);
        let f0 = self.light_f0.swap(level.light_f0, Ordering::Relaxed) != level.light_f0;
        model || f0
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

    fn stream_report(&self) -> Option<StreamReport> {
        self.inner.stream_report()
    }
}


/// Wrap `model` for the governor's light-F0 step when its voice has one.
fn governed(model: Box<dyn VoiceConversionModel>, light_f0: Option<Arc<AtomicBool>>) -> Box<dyn VoiceConversionModel> {
    match light_f0 {
        Some(flag) => Box::new(GovernedModel { inner: model, light_f0: flag }),
        None => model,
    }
}

/// Load the engine for `request.slot_dir`.
pub(crate) fn create_engine(request: EngineRequest) -> Result<CreatedEngine, String> {
    create_engine_with(request, DenoiserKind::Off)
}

/// [`create_engine`] with the 16 kHz denoiser that runs inside the harness (`gtcrn`; any other kind
/// is not the factory's and is ignored here: the worker runs it on the capture's device-rate copy).
pub(crate) fn create_engine_with(request: EngineRequest, denoiser: DenoiserKind) -> Result<CreatedEngine, String> {
    let format = slot_format(&request.slot_dir)?;
    let probe = VramProbe::for_rung(&request.ep, request.session.cuda_device_index);
    let mut created = match format {
        SlotFormat::LegacyOnnxRvc => create_legacy(&request)?,
        SlotFormat::SigilRvc => create_sigil(&request)?,
        SlotFormat::Unknown(f) => return Err(format!("the voice-changer format '{f}' is not supported by this build")),
    };
    if denoiser == DenoiserKind::Gtcrn {
        wrap_gtcrn(&mut created, &request);
    }
    created.vram_mb = probe.and_then(|p| p.delta_mb()).unwrap_or(0.0);
    Ok(created)
}

fn create_legacy(request: &EngineRequest) -> Result<CreatedEngine, String> {
    let since_ms = crate::services::diagnostics::logging::unix_ms_now();
    let BuiltPitch {
        extractor: pitch,
        notes: mut warnings,
        mut outcomes,
        light_f0,
        light_note,
    } = build_pitch(request)?;
    let mut model = RvcModel::load(
        &request.slot_dir,
        &request.hubert_path,
        pitch,
        &request.ep,
        &request.session,
        request.silence_skip,
    )?;
    warnings.extend(model.warnings().iter().cloned());
    // The legacy model runs the levers its export supports; the rest keep their legacy path. The
    // governor is the runtime's (it steps every voice).
    let mut levers = request.levers.clone();
    levers.notes.extend(light_note);
    for lever in Lever::ALL {
        let resolved = levers.levers.get(lever);
        let used = match lever {
            // `auto` keeps the export's zero noise; only an explicit `on` anchors it.
            Lever::AnchoredNoise => presets::byo_lever(lever, request.anchored_noise_mode, resolved),
            Lever::Governor => resolved,
            Lever::PartialDecode => model.partial_decode(),
            _ => false,
        };
        if resolved && !used {
            // Resolved on but unused anchored noise is an `auto` (an `on` is used).
            levers.notes.push(if lever == Lever::AnchoredNoise {
                format!(
                    "{}: auto keeps bring-your-own ONNX voices on their export's zero noise (set it to on to anchor it)",
                    lever.as_str()
                )
            } else {
                format!("{}: not used by bring-your-own ONNX voices (import the .pth for the Sigil engine)", lever.as_str())
            });
        }
        levers.levers.set(lever, used);
    }
    if request.lookahead_ms > 0.0 {
        levers
            .notes
            .push("lookahead: not used by bring-your-own ONNX voices (import the .pth for the Sigil engine)".to_string());
    }
    if levers.levers.get(Lever::AnchoredNoise) {
        model.set_noise(LegacyNoise::Anchored {
            seed: noise_seed_for(&request.slot_dir, None),
            scale: DEFAULT_NOISE_SCALE,
        });
    }
    levers.accelerator = engine::Accelerator::None;
    outcomes.retain(|(c, _)| !c.is_empty());
    // The legacy model's sessions (generator `vc`, content encoder `vc.content`) report only to the
    // diagnostics registry; list them first, so the voice's outcomes name the generator's EP (and
    // its fallback) rather than only the pitch extractor's.
    let prefix = request.session.component_prefix.filter(|p| !p.is_empty()).map(|p| format!("{p}.")).unwrap_or_default();
    let mut recorded: Vec<(String, EpOutcome)> = Vec::new();
    for e in crate::services::diagnostics::ep_outcomes().into_iter().filter(|e| e.at_unix_ms >= since_ms) {
        if let Some(bare) = e.component.strip_prefix(prefix.as_str()).filter(|c| matches!(*c, "vc" | "vc.content")) {
            if !recorded.iter().any(|(c, _)| c == bare) {
                recorded.push((bare.to_string(), e.outcome));
            }
        }
    }
    outcomes.splice(0..0, recorded);
    let output_sample_rate = model.output_sample_rate();
    Ok(CreatedEngine {
        model: governed(Box::new(PostModel::new(Box::new(model))), light_f0),
        format: SlotFormat::LegacyOnnxRvc,
        output_sample_rate,
        levers,
        outcomes,
        warnings,
        vram_mb: 0.0,
        denoiser_latency_16k: 0,
        model_delay_16k: 0,
    })
}

fn create_sigil(request: &EngineRequest) -> Result<CreatedEngine, String> {
    if !engine::available() {
        return Err("Sigil template voices (sigil-rvc-1) are not supported by this build".into());
    }
    let manifest = load_slot_manifest(&request.slot_dir)?;
    let templates_dir = request
        .templates_dir
        .clone()
        .filter(|d| d.is_dir())
        .or_else(import::templates_dir)
        .ok_or("the bundled vc-templates folder was not found; reinstall Sigil")?;
    let BuiltPitch {
        extractor: pitch,
        notes: mut warnings,
        mut outcomes,
        light_f0,
        light_note,
    } = build_pitch(request)?;
    let preset = presets::engine_preset(
        request.preset,
        request.block_ms,
        request.chunk_frames,
        request.device_rate,
        request.extra_context_ms,
        request.precision,
    );
    let noise_seed = noise_seed_for(&request.slot_dir, manifest.source_sha256.as_deref());
    let mut engine = SigilRvcEngine::load(SigilRvcRequest {
        slot_dir: request.slot_dir.clone(),
        manifest,
        templates_dir,
        ep: request.ep.clone(),
        session: request.session.clone(),
        levers: request.levers.clone(),
        pitch,
        content_encoder: opt_path(&request.hubert_path),
        index: None,
        preset,
        f0_range: request.f0_range,
        noise_seed,
    })?;
    engine.set_silence_skip(request.silence_skip);
    // Lookahead (before the harness is built: it sizes its context from the plan). Not on a
    // DirectML content encoder yet: with lookahead its window shape costs 35-49 ms per call there
    // instead of about 4 ms (the same DirectML re-plan as a crossfade other than 20 ms; P3-I0
    // measurement), which no 40 ms block survives.
    let content_on_dml = engine
        .ep_outcomes()
        .iter()
        .any(|(c, o)| c.ends_with("vc.content") && o.active == "dml");
    let mut lookahead_note = None;
    if request.lookahead_ms > 0.0 {
        if content_on_dml {
            lookahead_note = Some(
                "lookahead: not used while the content encoder runs on DirectML (it would cost 35-49 ms per block there)"
                    .to_string(),
            );
        } else {
            engine.set_lookahead_ms(request.lookahead_ms)?;
        }
    }
    let model_delay_16k = engine.plan().lookahead_frames * 160;
    warnings.extend(engine.warnings().iter().cloned());
    outcomes.extend(engine.ep_outcomes().iter().cloned());
    let mut levers = engine.effective_levers().clone();
    levers.notes.extend(light_note);
    levers.notes.extend(lookahead_note);
    let output_sample_rate = engine.output_sample_rate();
    Ok(CreatedEngine {
        model: governed(Box::new(engine), light_f0),
        format: SlotFormat::SigilRvc,
        output_sample_rate,
        levers,
        outcomes,
        warnings,
        vram_mb: 0.0,
        denoiser_latency_16k: 0,
        model_delay_16k,
    })
}

/// Wrap the model with GTCRN at 16 kHz (CPU, one thread: it is tiny). A denoiser that cannot load
/// is a warning; the voice runs without it.
fn wrap_gtcrn(created: &mut CreatedEngine, request: &EngineRequest) {
    let opts = VcSessionOptions {
        intra_threads: Some(1),
        vram_cap_mb: None,
        streamer_profile: false,
        ..request.session.clone()
    };
    let chunk = created.model.fixed_block();
    let loaded = denoise::model_path(DenoiserKind::Gtcrn).and_then(|p| denoise::gtcrn::load_from(&p, "cpu", &opts, chunk));
    match loaded {
        Ok(den) => {
            let latency = denoise::Denoiser::latency_samples(&den);
            let model = std::mem::replace(&mut created.model, Box::new(NullModel));
            match DenoisedModel::new(model, Box::new(den)) {
                Ok(wrapped) => {
                    created.model = Box::new(wrapped);
                    created.denoiser_latency_16k = latency;
                }
                Err(e) => {
                    // Cannot happen for 16 kHz models; keep the voice working regardless.
                    created.warnings.push(format!("denoiser: {e}"));
                }
            }
        }
        Err(e) => created.warnings.push(format!("denoiser: {e}; the voice runs without it")),
    }
}

/// Legacy voices get the post stage's per-voice controls (rms_mix, the true formant shift) at the
/// model rate, as template voices have them inside the engine. With the neutral controls (and the
/// gate and limiter off) the stage is a bit-exact pass-through: the legacy sound.
struct PostModel {
    inner: Box<dyn VoiceConversionModel>,
    post: PostStage,
    controls: VoiceControls,
}

impl PostModel {
    fn new(inner: Box<dyn VoiceConversionModel>) -> PostModel {
        let post = PostStage::new(PostConfig {
            sample_rate: inner.output_sample_rate(),
            limiter: false,
            silence_gate: false,
            silence_hangover_ms: 150.0,
        });
        PostModel {
            inner,
            post,
            controls: VoiceControls::default(),
        }
    }
}

impl VoiceConversionModel for PostModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let mut out = self.inner.convert(ctx)?;
        let source = &ctx.mono_16k[ctx.mono_16k.len().saturating_sub(ctx.new_frames)..];
        // A post-stage failure keeps the converted audio unchanged (the stage is optional polish).
        let _ = self.post.process(source, &mut out, &self.controls);
        Ok(out)
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
        self.controls = *controls;
        self.inner.set_voice_controls(controls);
    }

    fn apply_quality(&mut self, level: &QualityLevel) -> bool {
        self.inner.apply_quality(level)
    }

    fn reset_stream(&mut self) {
        self.post.reset();
        self.inner.reset_stream();
    }

    fn fixed_block(&self) -> Option<usize> {
        self.inner.fixed_block()
    }

    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        self.inner.configure_stream(block, want, extra)
    }

    fn stream_report(&self) -> Option<StreamReport> {
        self.inner.stream_report()
    }
}

/// Placeholder while the model is being wrapped (never converts).
struct NullModel;

impl VoiceConversionModel for NullModel {
    fn convert(&mut self, _: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        Err("no model".into())
    }
    fn input_sample_rate(&self) -> u32 {
        16_000
    }
    fn output_sample_rate(&self) -> u32 {
        16_000
    }
}

// ---------------------------------------------------------------------------------------------
// Harness and hot slots
// ---------------------------------------------------------------------------------------------

/// Fewest device blocks of recent input a voice is primed with before it takes over (refreshes its
/// SOLA tail and the newest context; the outputs are discarded). See [`LoadedVoice::prime_blocks`].
pub(crate) const SWITCH_PRIME_BLOCKS: usize = 2;

/// The device side of a run's harness (fixed for the run).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct HarnessSpec {
    pub r_in: u32,
    pub r_out: u32,
    pub ch_in: usize,
    pub ch_out: usize,
    pub extra_context_ms: f64,
    pub crossfade_ms: f64,
    pub chunk_frames: usize,
}

/// Build the stream harness around a created engine (warms it at the live shape).
pub(crate) fn build_harness(model: Box<dyn VoiceConversionModel>, spec: &HarnessSpec) -> Result<VcStreamHarness, String> {
    let model_out_sr = model.output_sample_rate();
    let extra_samples = ((spec.extra_context_ms / 1000.0) * 16_000.0).round().max(0.0) as usize;
    let crossfade_samples = ((spec.crossfade_ms / 1000.0) * f64::from(model_out_sr)).round().max(0.0) as usize;
    VcStreamHarness::new(
        model,
        spec.r_in,
        spec.r_out,
        spec.ch_in,
        spec.ch_out,
        extra_samples,
        crossfade_samples,
        spec.chunk_frames,
    )
}

/// What identifies a loaded voice: a switch reuses a loaded one only when all of it matches.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EngineKey {
    pub slot_dir: PathBuf,
    pub pitch_extractor: String,
    pub f0_range: F0Range,
    pub denoiser: DenoiserKind,
}

/// A voice loaded behind its own harness.
pub(crate) struct LoadedVoice {
    pub key: EngineKey,
    pub harness: VcStreamHarness,
    pub format: SlotFormat,
    pub output_sample_rate: u32,
    pub levers: EffectiveLevers,
    pub outcomes: Vec<(String, EpOutcome)>,
    pub warnings: Vec<String>,
    pub vram_mb: f64,
    pub denoiser_latency_16k: usize,
    /// The model's own delay (a template voice's lookahead; 16 kHz samples).
    pub model_delay_16k: usize,
    /// The model's input rate (16 kHz for the RVC family), for the switch priming.
    pub model_input_rate: u32,
    /// Resource-governor claim (resized to the measured VRAM; 0 MB on the CPU).
    _claim: VramClaim,
    pub load_ms: f64,
    /// Its load warnings were shown (they are toasted on its first activation only, not on every
    /// switch back to it).
    pub warnings_shown: bool,
}

/// Error of a background load stopped because nobody wanted the voice any more.
pub(crate) const LOAD_CANCELLED: &str = "the voice load was cancelled";

impl LoadedVoice {
    /// Build a voice: engine, harness (warm-up included) and its VRAM claim.
    pub(crate) fn load(key: EngineKey, request: EngineRequest, spec: &HarnessSpec) -> Result<LoadedVoice, String> {
        Self::load_cancellable(key, request, spec, &AtomicBool::new(false))
    }

    /// [`Self::load`] that stops at its next stage (before the sessions, the warm-up at the live
    /// shape, the VRAM claim) once `cancel` is set, with [`LOAD_CANCELLED`].
    pub(crate) fn load_cancellable(key: EngineKey, request: EngineRequest, spec: &HarnessSpec, cancel: &AtomicBool) -> Result<LoadedVoice, String> {
        let check = || if cancel.load(Ordering::Relaxed) { Err(LOAD_CANCELLED.to_string()) } else { Ok(()) };
        check()?;
        let t0 = Instant::now();
        let label = format!("vc: {}", key.slot_dir.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
        let gpu = request.ep != "cpu";
        let probe = VramProbe::for_rung(&request.ep, request.session.cuda_device_index);
        let created = create_engine_with(request, key.denoiser)?;
        check()?;
        let model_input_rate = created.model.input_sample_rate();
        let harness = build_harness(created.model, spec)?;
        check()?;
        // The harness configured the model's stream, which may have rebuilt sessions (a CPU
        // fallback, a CUDA graph that no longer captures): the voice reports what runs now (T7).
        let (mut outcomes, mut levers, mut warnings) = (created.outcomes, created.levers, created.warnings);
        if let Some(report) = harness.model_report() {
            merge_stream_report(&mut outcomes, &mut levers, &mut warnings, report);
        }
        // The warm-up above allocated the sessions' working memory: measure after it.
        let vram_mb = probe.and_then(|p| p.delta_mb()).unwrap_or(created.vram_mb).max(created.vram_mb);
        let claim = resources::claim_vram(Subsystem::Vc, &label, if gpu { vram_mb.round() as u64 } else { 0 });
        Ok(LoadedVoice {
            key,
            harness,
            format: created.format,
            output_sample_rate: created.output_sample_rate,
            levers,
            outcomes,
            warnings,
            vram_mb,
            denoiser_latency_16k: created.denoiser_latency_16k,
            model_delay_16k: created.model_delay_16k,
            model_input_rate,
            _claim: claim,
            load_ms: t0.elapsed().as_secs_f64() * 1000.0,
            warnings_shown: false,
        })
    }

    /// A stand-in voice around `model` (tests of the switching logic elsewhere in the crate).
    #[cfg(test)]
    pub(crate) fn for_test(key: EngineKey, model: Box<dyn VoiceConversionModel>, spec: &HarnessSpec, vram_mb: f64) -> LoadedVoice {
        let model_input_rate = model.input_sample_rate();
        let output_sample_rate = model.output_sample_rate();
        LoadedVoice {
            key,
            harness: build_harness(model, spec).expect("test harness"),
            format: SlotFormat::LegacyOnnxRvc,
            output_sample_rate,
            levers: EffectiveLevers::default(),
            outcomes: Vec::new(),
            warnings: Vec::new(),
            vram_mb,
            denoiser_latency_16k: 0,
            model_delay_16k: 0,
            model_input_rate,
            _claim: resources::claim_vram(Subsystem::Vc, "test: voice", 0),
            load_ms: 0.0,
            warnings_shown: false,
        }
    }

    /// Device blocks of recent input (at least [`SWITCH_PRIME_BLOCKS`]) to push through this voice
    /// before it takes over, so its first block after the switch already plays converted audio.
    ///
    /// The legacy feed converts every device block, so two refresh the SOLA tail and the newest
    /// context. A fixed-block voice (the Sigil Voice Engine) converts nothing until a whole model
    /// block has arrived, and a fresh fixed-block harness then holds back its output hold-back
    /// (`VcStreamHarness::fixed_hold_back_frames`) on top of the jitter cushion before it plays; so it
    /// is primed with one model block plus that hold-back (in input frames) plus one device block for
    /// the pre-accumulator's phase and the cushion. Without it the first 60-90 ms after a switch to a
    /// template voice would be silence.
    pub(crate) fn prime_blocks(&self, spec: &HarnessSpec) -> usize {
        let Some((block, _)) = self.harness.fixed_shape() else {
            return SWITCH_PRIME_BLOCKS;
        };
        let (r_in, r_out) = (u64::from(spec.r_in.max(1)), u64::from(spec.r_out.max(1)));
        let block_frames = (block as u64 * r_in).div_ceil(u64::from(self.model_input_rate.max(1)));
        let hold_back = (self.harness.fixed_hold_back_frames() as u64 * r_in).div_ceil(r_out);
        let frames = (block_frames + hold_back) as usize;
        (frames.div_ceil(spec.chunk_frames.max(1)) + 1).max(SWITCH_PRIME_BLOCKS)
    }
}

/// Fold a model's [`StreamReport`] (after the harness configured its stream) into what a voice
/// reports: its components replace the ones of the same name (others, such as the pitch extractor
/// or the denoiser, stay), the levers in effect and the accelerator are the report's, and its lever
/// notes and warnings are added to the existing ones (no duplicates).
fn merge_stream_report(outcomes: &mut Vec<(String, EpOutcome)>, levers: &mut EffectiveLevers, warnings: &mut Vec<String>, report: StreamReport) {
    for (component, outcome) in report.outcomes {
        match outcomes.iter_mut().find(|(c, _)| *c == component) {
            Some(slot) => slot.1 = outcome,
            None => outcomes.push((component, outcome)),
        }
    }
    levers.levers = report.levers.levers;
    levers.accelerator = report.levers.accelerator;
    for note in report.levers.notes {
        if !levers.notes.contains(&note) {
            levers.notes.push(note);
        }
    }
    for warning in report.warnings {
        if !warnings.contains(&warning) {
            warnings.push(warning);
        }
    }
}

/// A background load in flight.
struct PendingLoad {
    key: EngineKey,
    rx: Receiver<Result<LoadedVoice, String>>,
    started: Instant,
    /// Set when nobody wants the voice any more: the loader stops at its next stage.
    cancel: Arc<AtomicBool>,
}

/// What [`VoiceSet::request`] did.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum SwitchPlan {
    /// The voice is the active one.
    Active,
    /// It is loaded: [`VoiceSet::take_ready`] returns it now.
    Ready,
    /// It is loading in the background (or waits for the load in flight to finish).
    Loading,
}

/// Builds a voice's request on the loader thread (it reads the slot's manifest and resolves its base
/// models: file I/O the worker never does).
pub(crate) type RequestBuilder = Box<dyn FnOnce() -> Result<EngineRequest, String> + Send>;

/// Loads one voice on the loader thread (the factory's [`LoadedVoice::load_cancellable`]; tests
/// replace it).
pub(crate) type VoiceLoader =
    Arc<dyn Fn(EngineKey, EngineRequest, &HarnessSpec, &AtomicBool) -> Result<LoadedVoice, String> + Send + Sync>;

/// The run's voices: the active one, the idle loaded ones (most recent first), a voice ready to
/// switch to, and at most one background load (see the module docs).
pub(crate) struct VoiceSet {
    active: LoadedVoice,
    idle: Vec<LoadedVoice>,
    ready: Option<LoadedVoice>,
    /// The one load running.
    pending: Option<PendingLoad>,
    /// The newest voice asked for while another load was running: it starts when that one ends.
    queued: Option<(EngineKey, RequestBuilder, HarnessSpec)>,
    /// The voice the selection waits for (`None`: the selection is the active, the ready or no
    /// voice). `queued`, when set, is this voice.
    wanted: Option<EngineKey>,
    loader: VoiceLoader,
    /// Voices kept loaded, the active one included (`hot_slots`).
    max_loaded: usize,
    /// VRAM the loaded voices may hold together (MiB, 0 = no limit).
    vram_budget_mb: u32,
}

impl VoiceSet {
    pub(crate) fn new(active: LoadedVoice, hot_slots: u32, vram_budget_mb: u32) -> VoiceSet {
        Self::with_loader(
            active,
            hot_slots,
            vram_budget_mb,
            Arc::new(LoadedVoice::load_cancellable),
        )
    }

    pub(crate) fn with_loader(active: LoadedVoice, hot_slots: u32, vram_budget_mb: u32, loader: VoiceLoader) -> VoiceSet {
        VoiceSet {
            active,
            idle: Vec::new(),
            ready: None,
            pending: None,
            queued: None,
            wanted: None,
            loader,
            max_loaded: hot_slots.max(1) as usize,
            vram_budget_mb,
        }
    }

    pub(crate) fn active(&self) -> &LoadedVoice {
        &self.active
    }

    pub(crate) fn active_mut(&mut self) -> &mut LoadedVoice {
        &mut self.active
    }

    /// The voice ready to switch to (the worker primes it before it takes over).
    pub(crate) fn ready_mut(&mut self) -> Option<&mut LoadedVoice> {
        self.ready.as_mut()
    }

    /// Voices loaded (active, idle, a ready switch).
    pub(crate) fn loaded(&self) -> usize {
        1 + self.idle.len() + usize::from(self.ready.is_some())
    }

    /// Slot folders of the voices it holds: the active one, a voice ready to switch to, the idle hot
    /// voices and the load in flight (in that order; a voice may appear twice while it reloads).
    /// Never allocates.
    pub(crate) fn slot_dirs_iter(&self) -> impl Iterator<Item = &Path> + '_ {
        std::iter::once(self.active.key.slot_dir.as_path())
            .chain(self.ready.iter().map(|v| v.key.slot_dir.as_path()))
            .chain(self.idle.iter().map(|v| v.key.slot_dir.as_path()))
            .chain(self.pending.iter().map(|p| p.key.slot_dir.as_path()))
    }

    /// [`Self::slot_dirs_iter`], collected (the import commands refuse to replace or delete these;
    /// `VoiceChangerState::held_slots`).
    pub(crate) fn slot_dirs(&self) -> Vec<PathBuf> {
        self.slot_dirs_iter().map(Path::to_path_buf).collect()
    }

    /// VRAM held by the loaded voices (MiB).
    pub(crate) fn vram_mb(&self) -> f64 {
        self.active.vram_mb + self.idle.iter().map(|v| v.vram_mb).sum::<f64>() + self.ready.as_ref().map_or(0.0, |v| v.vram_mb)
    }

    /// Nothing to load: a queued request is dropped and the load in flight is told to stop (it is
    /// still waited for, so loads never overlap).
    fn want_nothing(&mut self) {
        self.wanted = None;
        self.queued = None;
        if let Some(p) = &self.pending {
            p.cancel.store(true, Ordering::Relaxed);
        }
    }

    /// Ask for the voice `key`: the active one, a loaded one (ready to switch now), or load it in the
    /// background with the request `make_request` builds on the loader thread (after the load in
    /// flight, if any: the newest request wins, and a load of another voice is told to stop). A
    /// request that cannot be built fails like a load.
    pub(crate) fn request(
        &mut self,
        key: EngineKey,
        spec: HarnessSpec,
        make_request: impl FnOnce() -> Result<EngineRequest, String> + Send + 'static,
    ) -> SwitchPlan {
        if self.active.key == key {
            // Back to the voice that plays: a prepared switch is kept loaded, a load is abandoned.
            self.abandon();
            return SwitchPlan::Active;
        }
        if self.ready.as_ref().is_some_and(|v| v.key == key) {
            self.want_nothing();
            return SwitchPlan::Ready;
        }
        if let Some(r) = self.ready.take() {
            self.keep_idle(r);
        }
        if let Some(i) = self.idle.iter().position(|v| v.key == key) {
            self.ready = Some(self.idle.remove(i));
            self.want_nothing();
            return SwitchPlan::Ready;
        }
        self.wanted = Some(key.clone());
        match &self.pending {
            Some(p) if p.key == key && !p.cancel.load(Ordering::Relaxed) => {
                self.queued = None;
            }
            Some(p) => {
                // Another voice (or this one, already told to stop) is loading: stop it and load this
                // one after it.
                p.cancel.store(true, Ordering::Relaxed);
                if self.queued.as_ref().is_none_or(|q| q.0 != key) {
                    self.queued = Some((key, Box::new(make_request), spec));
                }
            }
            None => {
                self.queued = None;
                self.spawn(key, Box::new(make_request), spec);
            }
        }
        SwitchPlan::Loading
    }

    fn spawn(&mut self, key: EngineKey, request: RequestBuilder, spec: HarnessSpec) {
        let (tx, rx) = mpsc::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (load_key, flag, loader) = (key.clone(), cancel.clone(), self.loader.clone());
        let spawned = std::thread::Builder::new().name("sigil-vc-load".into()).spawn(move || {
            let _ = tx.send(request().and_then(|request| loader(load_key, request, &spec, &flag)));
        });
        let rx = match spawned {
            Ok(_) => rx,
            Err(e) => {
                let (tx, rx) = mpsc::channel();
                let _ = tx.send(Err(format!("could not start loading the voice: {e}")));
                rx
            }
        };
        self.pending = Some(PendingLoad {
            key,
            rx,
            started: Instant::now(),
            cancel,
        });
    }

    /// The voice the selection waits for (loading, or queued behind another load).
    #[cfg(test)]
    pub(crate) fn loading(&self) -> Option<&EngineKey> {
        self.wanted.as_ref()
    }

    /// Stay on the active voice: a load in flight is told to stop (its result is dropped when it
    /// arrives) and a prepared switch is kept loaded.
    pub(crate) fn abandon(&mut self) {
        self.want_nothing();
        if let Some(r) = self.ready.take() {
            self.keep_idle(r);
        }
    }

    /// A finished background load: `Some(Ok(key))` when the voice the selection waits for is ready
    /// to switch to, `Some(Err)` when it failed, `None` otherwise (nothing finished, or a load nobody
    /// wants any more, which is dropped). The newest queued request then starts.
    pub(crate) fn poll(&mut self) -> Option<Result<(EngineKey, Duration), (EngineKey, String)>> {
        let result = match self.pending.as_ref()?.rx.try_recv() {
            Ok(r) => r,
            Err(TryRecvError::Empty) => return None,
            Err(TryRecvError::Disconnected) => Err("the voice loader stopped unexpectedly".into()),
        };
        let p = self.pending.take().expect("pending checked above");
        let took = p.started.elapsed();
        let wanted = self.wanted.as_ref() == Some(&p.key);
        let cancelled = p.cancel.load(Ordering::Relaxed);
        let report = match result {
            Ok(voice) if wanted => {
                if let Some(r) = self.ready.take() {
                    self.keep_idle(r);
                }
                self.ready = Some(voice);
                self.wanted = None;
                self.queued = None;
                Some(Ok((p.key, took)))
            }
            // Superseded: freed off the worker thread.
            Ok(voice) => {
                release(Some(voice));
                None
            }
            // A load that was told to stop never reports (if its voice is wanted again, it is queued).
            Err(e) if wanted && !cancelled => {
                self.wanted = None;
                self.queued = None;
                Some(Err((p.key, e)))
            }
            Err(_) => None,
        };
        if let Some((key, request, spec)) = self.queued.take() {
            if self.wanted.as_ref() == Some(&key) {
                self.spawn(key, request, spec);
            }
        }
        report
    }

    /// Make the ready voice active; returns the previous active voice (for the crossfade tick; hand
    /// it back with [`Self::retire`]).
    pub(crate) fn take_ready(&mut self) -> Option<LoadedVoice> {
        let next = self.ready.take()?;
        Some(std::mem::replace(&mut self.active, next))
    }

    /// Keep a voice that stopped being active loaded, if the limits allow (else it is dropped,
    /// releasing its sessions and VRAM claim).
    pub(crate) fn retire(&mut self, voice: LoadedVoice) {
        self.keep_idle(voice);
    }

    fn keep_idle(&mut self, voice: LoadedVoice) {
        self.idle.insert(0, voice);
        self.evict();
    }

    /// Drop the least recently used idle voices beyond `max_loaded` or the VRAM budget.
    fn evict(&mut self) {
        while !self.idle.is_empty() && self.loaded() > self.max_loaded {
            release(self.idle.pop());
        }
        if self.vram_budget_mb > 0 {
            while !self.idle.is_empty() && self.vram_mb() > f64::from(self.vram_budget_mb) {
                release(self.idle.pop());
            }
        }
    }
}

/// Free an evicted voice off the worker thread (releasing ONNX Runtime sessions and GPU memory can
/// take tens of milliseconds; the worker must not stall on it).
fn release(voice: Option<LoadedVoice>) {
    let Some(voice) = voice else { return };
    if let Err(e) = std::thread::Builder::new().name("sigil-vc-release".into()).spawn(move || drop(voice)) {
        tracing::debug!("voice changer: releasing a voice on the worker thread ({e})");
    }
}

/// Crossfade `from` (the outgoing voice's block) into `into` (the incoming one's) over the whole
/// block, equal power per frame, in place in `into`. The blocks may differ by a frame (each harness
/// keeps its own rounding carry): the common part fades, the rest is the incoming voice's.
pub(crate) fn crossfade_into(from: &[f32], into: &mut [f32], ch: usize) {
    let ch = ch.max(1);
    let frames = (from.len().min(into.len())) / ch;
    for f in 0..frames {
        let t = (f as f32 + 0.5) / frames.max(1) as f32;
        let a = t * std::f32::consts::FRAC_PI_2;
        let (w_in, w_out) = (a.sin(), a.cos());
        for c in 0..ch {
            let i = f * ch + c;
            into[i] = into[i] * w_in + from[i] * w_out;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn slot_format_reads_the_manifest_or_assumes_legacy_onnx() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("sigil-vc-factory-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(slot_format(&dir).unwrap(), SlotFormat::LegacyOnnxRvc);
        std::fs::write(
            dir.join("manifest.json"),
            br#"{"format":"sigil-rvc-1","template":"v2-40k-f0-hifigan","weights":"weights.f32.bin"}"#,
        )
        .unwrap();
        assert_eq!(slot_format(&dir).unwrap(), SlotFormat::SigilRvc);
        std::fs::write(dir.join("manifest.json"), br#"{"format":"ddsp-svc-6.3"}"#).unwrap();
        assert!(slot_format(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn noise_seeds_are_stable_per_voice() {
        let a = noise_seed_for(Path::new("C:/voices/alice"), None);
        assert_eq!(a, noise_seed_for(Path::new("D:/other/alice"), None), "the folder name, not the path");
        assert_ne!(a, noise_seed_for(Path::new("C:/voices/bob"), None));
        assert_eq!(noise_seed_for(Path::new("x"), Some("abc")), noise_seed_for(Path::new("y"), Some("abc")));
    }

    /// The switch crossfade starts on the outgoing voice and ends on the incoming one, smoothly, and
    /// tolerates a one-frame length difference.
    #[test]
    fn switch_crossfade_goes_from_old_to_new() {
        let from = vec![1.0f32; 2 * 100];
        let mut into = vec![-1.0f32; 2 * 101];
        crossfade_into(&from, &mut into, 2);
        assert!(into[0] > 0.95 && into[1] > 0.95, "starts on the old voice");
        assert!(into[2 * 99] < -0.95, "ends on the new voice");
        assert_eq!(into[2 * 100], -1.0, "the extra frame is the new voice's");
        for w in into.chunks(2).collect::<Vec<_>>().windows(2) {
            assert!(w[1][0] <= w[0][0] + 1e-6);
        }
    }

    /// Unsupported formats are refused with a clear message before anything loads.
    #[test]
    fn unknown_formats_are_refused() {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("sigil-vc-factory-unknown-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("manifest.json"), br#"{"format":"so-vits-4.1"}"#).unwrap();
        let err = create_engine(test_request(&dir)).err().unwrap();
        assert!(err.contains("so-vits-4.1"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    pub(crate) fn test_request(slot: &Path) -> EngineRequest {
        EngineRequest {
            slot_dir: slot.to_path_buf(),
            app_data: None,
            templates_dir: None,
            hubert_path: String::new(),
            fcpe_path: String::new(),
            rmvpe_path: String::new(),
            pitch_extractor: "dio".into(),
            ep: "cpu".into(),
            session: VcSessionOptions::default(),
            levers: EffectiveLevers::default(),
            anchored_noise_mode: LeverMode::Auto,
            silence_skip: true,
            preset: RvcPresetProfile::LowLatency,
            precision: PrecisionMode::Auto,
            block_ms: 0.0,
            chunk_frames: 480,
            device_rate: 48_000,
            extra_context_ms: 64.0,
            lookahead_ms: 0.0,
            f0_range: F0Range::Speech,
            speaker_id: 0,
        }
    }

    // ---- hot voice set ----

    /// A stand-in voice: returns its constant level (so a crossfade is visible), 16 kHz in and out.
    struct Level(f32);

    impl VoiceConversionModel for Level {
        fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
            Ok(vec![self.0; ctx.new_frames])
        }
        fn input_sample_rate(&self) -> u32 {
            16_000
        }
        fn output_sample_rate(&self) -> u32 {
            16_000
        }
    }

    fn spec() -> HarnessSpec {
        HarnessSpec {
            r_in: 48_000,
            r_out: 48_000,
            ch_in: 1,
            ch_out: 1,
            extra_context_ms: 32.0,
            crossfade_ms: 10.0,
            chunk_frames: 480,
        }
    }

    fn key(name: &str) -> EngineKey {
        EngineKey {
            slot_dir: PathBuf::from(format!("C:/voices/{name}")),
            pitch_extractor: "dio".into(),
            f0_range: F0Range::Speech,
            denoiser: DenoiserKind::Off,
        }
    }

    fn voice(name: &str, level: f32, vram_mb: f64) -> LoadedVoice {
        LoadedVoice {
            key: key(name),
            harness: build_harness(Box::new(Level(level)), &spec()).unwrap(),
            format: SlotFormat::LegacyOnnxRvc,
            output_sample_rate: 16_000,
            levers: EffectiveLevers::default(),
            outcomes: Vec::new(),
            warnings: Vec::new(),
            vram_mb,
            denoiser_latency_16k: 0,
            model_delay_16k: 0,
            model_input_rate: 16_000,
            _claim: resources::claim_vram(Subsystem::Vc, "test: voice", 0),
            load_ms: 0.0,
            warnings_shown: false,
        }
    }

    fn no_request() -> Result<EngineRequest, String> {
        panic!("a loaded voice needs no request")
    }

    /// Switching among loaded voices: the active one needs nothing; an idle one is ready at once
    /// and swaps in; the previous one stays loaded within `hot_slots`; the least recently used
    /// beyond the limit (or the VRAM budget) is released.
    #[test]
    fn hot_voices_switch_without_loading_and_respect_the_limits() {
        let mut set = VoiceSet::new(voice("a", 0.1, 100.0), 3, 0);
        assert_eq!(set.request(key("a"), spec(), no_request), SwitchPlan::Active);
        set.retire(voice("b", 0.2, 100.0));
        set.retire(voice("c", 0.3, 100.0));
        assert_eq!(set.loaded(), 3);
        assert_eq!(set.request(key("b"), spec(), no_request), SwitchPlan::Ready);
        let old = set.take_ready().unwrap();
        assert_eq!(old.key, key("a"));
        assert_eq!(set.active().key, key("b"));
        set.retire(old);
        assert_eq!(set.loaded(), 3);
        // A fourth voice evicts the least recently used idle one (c).
        set.retire(voice("d", 0.4, 100.0));
        assert_eq!(set.loaded(), 3);
        assert_eq!(set.request(key("c"), spec(), || Ok(test_request(Path::new("Z:/no/such/voice")))), SwitchPlan::Loading);
        set.abandon();
        assert!(set.loading().is_none());
        // The VRAM budget evicts too.
        let mut tight = VoiceSet::new(voice("a", 0.1, 300.0), 8, 500);
        tight.retire(voice("b", 0.2, 300.0));
        assert_eq!(tight.loaded(), 1, "a + b would exceed 500 MB");
        assert_eq!(tight.vram_mb(), 300.0);
    }

    /// The slot folders a run holds (what the import commands refuse to replace or delete): the
    /// active voice, the one ready to switch to, the idle hot voices and the load in flight.
    #[test]
    fn the_held_slot_folders_cover_every_loaded_voice() {
        let dirs = |set: &VoiceSet| set.slot_dirs().into_iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>();
        let mut set = VoiceSet::new(voice("a", 0.1, 0.0), 3, 0);
        assert_eq!(dirs(&set), ["C:/voices/a"]);
        set.retire(voice("b", 0.2, 0.0));
        set.retire(voice("c", 0.3, 0.0));
        assert_eq!(dirs(&set), ["C:/voices/a", "C:/voices/c", "C:/voices/b"]);
        assert_eq!(set.request(key("b"), spec(), no_request), SwitchPlan::Ready);
        assert_eq!(dirs(&set), ["C:/voices/a", "C:/voices/b", "C:/voices/c"]);
        assert!(set.slot_dirs_iter().eq(set.slot_dirs().iter().map(PathBuf::as_path)));
        // A load in flight is held too (its files are being read).
        let gate = Arc::new(AtomicBool::new(false));
        let release_gate = gate.clone();
        let loader: VoiceLoader = Arc::new(move |_: EngineKey, _: EngineRequest, _: &HarnessSpec, _: &AtomicBool| {
            while !release_gate.load(Ordering::Relaxed) {
                std::thread::sleep(Duration::from_millis(2));
            }
            Err("stopped by the test".into())
        });
        let mut loading = VoiceSet::with_loader(voice("a", 0.1, 0.0), 2, 0, loader);
        assert_eq!(
            loading.request(key("d"), spec(), || Ok(test_request(Path::new("C:/voices/d")))),
            SwitchPlan::Loading
        );
        assert_eq!(dirs(&loading), ["C:/voices/a", "C:/voices/d"]);
        gate.store(true, Ordering::Relaxed);
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        while loading.poll().is_none() {
            assert!(std::time::Instant::now() < deadline, "the loader never answered");
            std::thread::sleep(Duration::from_millis(2));
        }
        assert_eq!(dirs(&loading), ["C:/voices/a"], "a finished load is no longer held");
    }

    /// A model's report after the harness configured its stream replaces the components it names,
    /// sets the levers in effect and adds its notes and warnings once.
    #[test]
    fn a_stream_report_refreshes_what_a_voice_reports() {
        let mut outcomes = vec![
            ("vc.pitch".to_string(), EpOutcome::granted("cuda")),
            ("vc.flowdec".to_string(), EpOutcome::granted("cuda")),
        ];
        let mut levers = EffectiveLevers::default();
        levers.levers.set(Lever::CudaGraphs, true);
        levers.notes.push("light f0: swiftf0".into());
        let mut warnings = vec!["w1".to_string()];
        let mut report = StreamReport {
            outcomes: vec![
                ("vc.flowdec".to_string(), EpOutcome::granted("cpu")),
                ("vc.enc".to_string(), EpOutcome::granted("cpu")),
            ],
            levers: EffectiveLevers::default(),
            warnings: vec!["w1".into(), "w2".into()],
        };
        report.levers.levers.set(Lever::PartialDecode, true);
        report.levers.notes.push("cuda_graphs: the graph no longer captures".into());
        merge_stream_report(&mut outcomes, &mut levers, &mut warnings, report.clone());
        let active: Vec<(&str, &str)> = outcomes.iter().map(|(c, o)| (c.as_str(), o.active.as_str())).collect();
        assert_eq!(active, [("vc.pitch", "cuda"), ("vc.flowdec", "cpu"), ("vc.enc", "cpu")]);
        assert!(levers.levers.get(Lever::PartialDecode));
        assert!(!levers.levers.get(Lever::CudaGraphs), "the report's levers are the ones in effect");
        assert_eq!(levers.notes, ["light f0: swiftf0", "cuda_graphs: the graph no longer captures"]);
        assert_eq!(warnings, ["w1", "w2"]);
        // Merging the same report again changes nothing.
        let before = (outcomes.clone(), levers.clone(), warnings.clone());
        merge_stream_report(&mut outcomes, &mut levers, &mut warnings, report);
        assert_eq!((outcomes, levers, warnings), before);
    }

    /// A voice that is not loaded loads on a background thread while the current one keeps
    /// playing; a load that fails is reported and the current voice stays.
    #[test]
    fn a_cold_voice_loads_in_the_background_and_failures_are_reported() {
        let mut set = VoiceSet::new(voice("a", 0.1, 0.0), 2, 0);
        let missing = std::env::temp_dir().join(format!("sigil-vc-missing-voice-{}", std::process::id()));
        std::fs::create_dir_all(&missing).unwrap();
        let k = EngineKey {
            slot_dir: missing.clone(),
            ..key("x")
        };
        let req = test_request(&missing);
        assert_eq!(set.request(k.clone(), spec(), move || Ok(req)), SwitchPlan::Loading);
        assert_eq!(set.request(k.clone(), spec(), no_request), SwitchPlan::Loading, "the same load is not started twice");
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let result = loop {
            if let Some(r) = set.poll() {
                break r;
            }
            assert!(std::time::Instant::now() < deadline, "the loader never answered");
            std::thread::sleep(Duration::from_millis(5));
        };
        let (failed, why) = result.expect_err("an empty folder has no generator");
        assert_eq!(failed, k);
        assert!(!why.is_empty());
        assert_eq!(set.active().key, key("a"), "the current voice keeps playing");
        assert!(set.take_ready().is_none());
        let _ = std::fs::remove_dir_all(&missing);
    }

    /// Loads never overlap and the newest request wins: cycling through cold voices while one loads
    /// runs one load at a time (each measures its own VRAM; none competes with another for the
    /// GPU); a load nobody wants any more is told to stop and is never reported; a request
    /// superseded before its load started never starts; the voice asked for last becomes ready.
    #[test]
    fn cycling_voices_runs_one_load_at_a_time_and_the_newest_wins() {
        use std::sync::atomic::AtomicUsize;
        use std::sync::Mutex;
        let running = Arc::new(AtomicUsize::new(0));
        let most = Arc::new(AtomicUsize::new(0));
        let started: Arc<Mutex<Vec<String>>> = Arc::default();
        let stopped: Arc<Mutex<Vec<String>>> = Arc::default();
        let loader: VoiceLoader = {
            let (running, most, started, stopped) = (running.clone(), most.clone(), started.clone(), stopped.clone());
            Arc::new(move |key: EngineKey, _request: EngineRequest, spec: &HarnessSpec, cancel: &AtomicBool| {
                let name = key.slot_dir.file_name().unwrap().to_string_lossy().into_owned();
                let n = running.fetch_add(1, Ordering::SeqCst) + 1;
                most.fetch_max(n, Ordering::SeqCst);
                started.lock().unwrap().push(name.clone());
                // A slow load (sessions, warm-up) that looks for a cancel between its stages.
                let mut cancelled = false;
                for _ in 0..100 {
                    if cancel.load(Ordering::Relaxed) {
                        cancelled = true;
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(2));
                }
                running.fetch_sub(1, Ordering::SeqCst);
                if cancelled {
                    stopped.lock().unwrap().push(name);
                    return Err(LOAD_CANCELLED.to_string());
                }
                Ok(LoadedVoice::for_test(key, Box::new(Level(0.5)), spec, 100.0))
            })
        };
        let mut set = VoiceSet::with_loader(voice("a", 0.1, 0.0), 3, 0, loader);
        for name in ["b", "c", "d"] {
            assert_eq!(set.request(key(name), spec(), || Ok(test_request(Path::new("unused")))), SwitchPlan::Loading);
        }
        assert_eq!(set.loading(), Some(&key("d")));
        let deadline = std::time::Instant::now() + Duration::from_secs(20);
        let result = loop {
            if let Some(r) = set.poll() {
                break r;
            }
            assert!(std::time::Instant::now() < deadline, "the loader never answered");
            std::thread::sleep(Duration::from_millis(2));
        };
        let (ready, _) = result.expect("the newest voice loads");
        assert_eq!(ready, key("d"));
        assert_eq!(most.load(Ordering::SeqCst), 1, "loads never overlap");
        assert_eq!(*started.lock().unwrap(), ["b", "d"], "c was superseded before it started");
        assert_eq!(*stopped.lock().unwrap(), ["b"], "the unwanted load stopped at its next stage");
        assert!(set.loading().is_none());
        let old = set.take_ready().unwrap();
        assert_eq!(set.active().key, key("d"));
        set.retire(old);
        // The VRAM the voice reports is its own (100 MB), not inflated by a concurrent load.
        assert_eq!(set.active().vram_mb, 100.0);
    }

    /// A cancelled load stops before its next stage with [`LOAD_CANCELLED`].
    #[test]
    fn a_cancelled_load_stops_before_it_starts() {
        let cancel = AtomicBool::new(true);
        let missing = Path::new("Z:/no/such/voice");
        let err = LoadedVoice::load_cancellable(key("x"), test_request(missing), &spec(), &cancel).err().unwrap();
        assert_eq!(err, LOAD_CANCELLED);
    }

    /// The switch block: the outgoing voice fades out while the incoming one fades in, and both
    /// harnesses keep exact lengths.
    #[test]
    fn a_switch_crossfades_two_harness_outputs() {
        let mut set = VoiceSet::new(voice("a", 0.5, 0.0), 2, 0);
        let block = vec![0.1f32; 480];
        for _ in 0..40 {
            set.active_mut().harness.push_and_convert(&block).unwrap();
        }
        set.retire(voice("b", -0.5, 0.0));
        assert_eq!(set.request(key("b"), spec(), no_request), SwitchPlan::Ready);
        let mut old = set.take_ready().unwrap();
        for _ in 0..40 {
            set.active_mut().harness.push_and_convert(&block).unwrap();
        }
        let mut new_block = set.active_mut().harness.push_and_convert(&block).unwrap();
        let old_block = old.harness.push_and_convert(&block).unwrap();
        assert_eq!(new_block.len(), 480);
        assert_eq!(old_block.len(), 480);
        crossfade_into(&old_block, &mut new_block, 1);
        assert!(new_block[0] > 0.45, "starts on the outgoing voice: {}", new_block[0]);
        assert!(new_block[479] < -0.45, "ends on the incoming voice: {}", new_block[479]);
        set.retire(old);
        assert_eq!(set.loaded(), 2);
    }

    /// A voice that streams in fixed model blocks converts nothing until a whole block arrived and
    /// then holds back its output cushion, so the two legacy priming blocks would leave the first
    /// blocks after a switch silent. `prime_blocks` primes enough for the switch block to play,
    /// across device blocks and rates, and never more than the hold-back (plus rounding) over the
    /// minimum found by trying.
    #[test]
    fn fixed_block_voices_are_primed_until_they_play() {
        struct Fixed(usize);
        impl VoiceConversionModel for Fixed {
            fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
                Ok(vec![0.5; ctx.new_frames * 2])
            }
            fn input_sample_rate(&self) -> u32 {
                16_000
            }
            fn output_sample_rate(&self) -> u32 {
                32_000
            }
            fn fixed_block(&self) -> Option<usize> {
                Some(self.0)
            }
        }
        assert_eq!(voice("legacy", 0.5, 0.0).prime_blocks(&spec()), SWITCH_PRIME_BLOCKS);
        for (r_in, r_out, chunk, block) in [
            (48_000u32, 48_000u32, 512usize, 640usize),
            (48_000, 48_000, 480, 640),
            (48_000, 44_100, 256, 640),
            (44_100, 48_000, 1024, 640),
            (48_000, 48_000, 512, 1280),
            (48_000, 48_000, 128, 640),
            (48_000, 48_000, 4096, 640),
        ] {
            let sp = HarnessSpec {
                r_in,
                r_out,
                ch_in: 1,
                ch_out: 1,
                extra_context_ms: 64.0,
                crossfade_ms: 20.0,
                chunk_frames: chunk,
            };
            let input = vec![0.1f32; chunk];
            // Prime `k` blocks, then: do the switch block and the one after it play the voice?
            let plays_after = |k: usize| {
                let mut h = build_harness(Box::new(Fixed(block)), &sp).unwrap();
                for _ in 0..k {
                    h.push_and_convert(&input).unwrap();
                }
                (0..2).all(|_| h.push_and_convert(&input).unwrap().iter().all(|&s| (s - 0.5).abs() < 1e-2))
            };
            let mut v = voice("fixed", 0.5, 0.0);
            v.harness = build_harness(Box::new(Fixed(block)), &sp).unwrap();
            let k = v.prime_blocks(&sp);
            let case = format!("{r_in} -> {r_out} Hz, {chunk}-frame device blocks, {block}-sample model blocks");
            assert!(plays_after(k), "{case}: {k} priming blocks leave silence");
            // It never overshoots by more than the hold-back it assumes (which a harness whose first
            // model block already covers the cushion does not need) and its rounding.
            let min = (0..k).find(|&n| plays_after(n)).unwrap_or(k);
            let hold_back_in = v.harness.fixed_hold_back_frames() * r_in as usize / r_out as usize;
            assert!((k - min) * chunk <= hold_back_in + 2 * chunk, "{case}: {k} blocks, {min} would do");
            if block == 640 && chunk <= 512 {
                assert!(!plays_after(SWITCH_PRIME_BLOCKS), "{case}: two blocks are not enough");
            }
        }
    }

    /// A stand-in F0 extractor: a constant contour, a given receptive field.
    struct ConstF0(f32, Option<(usize, usize)>);

    impl PitchExtractor for ConstF0 {
        fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
            Ok(vec![self.0; mono_16k.len() / 160 + 1])
        }
        fn receptive_frames(&self) -> Option<(usize, usize)> {
            self.1
        }
    }

    /// The governor's light-F0 step: the switchable extractor runs the voice's own until the step,
    /// then the light one, and back; its receptive field covers both (unknown if either is); the
    /// model wrapper reports the step as applied only when it changes something, and passes the
    /// other steps to the model.
    #[test]
    fn the_light_f0_step_switches_the_extractor() {
        let (mut pitch, flag) = SwitchablePitch::new(Box::new(ConstF0(200.0, Some((6, 6)))), Box::new(ConstF0(100.0, Some((10, 8)))));
        assert_eq!(pitch.receptive_frames(), Some((10, 8)));
        let (unknown, _) = SwitchablePitch::new(Box::new(ConstF0(200.0, None)), Box::new(ConstF0(100.0, Some((10, 8)))));
        assert_eq!(unknown.receptive_frames(), None);
        let audio = vec![0.0f32; 1_600];
        assert_eq!(pitch.f0(&audio).unwrap()[0], 200.0);

        /// Applies the retrieval step (like a template voice with an index).
        struct Retrieval(bool);
        impl VoiceConversionModel for Retrieval {
            fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
                Ok(vec![0.0; ctx.new_frames])
            }
            fn input_sample_rate(&self) -> u32 {
                16_000
            }
            fn output_sample_rate(&self) -> u32 {
                16_000
            }
            fn apply_quality(&mut self, level: &QualityLevel) -> bool {
                std::mem::replace(&mut self.0, level.retrieval) != level.retrieval
            }
        }
        let mut model = governed(Box::new(Retrieval(true)), Some(flag));
        // Level 1 (retrieval off): the model's step; F0 unchanged.
        assert!(model.apply_quality(&QualityLevel::at(1)));
        assert_eq!(pitch.f0(&audio).unwrap()[0], 200.0);
        // Level 2 (less context) changes nothing here: the governor skips it.
        assert!(!model.apply_quality(&QualityLevel::at(2)));
        // Level 3: the light extractor runs from the next tick.
        assert!(model.apply_quality(&QualityLevel::at(3)));
        assert_eq!(pitch.f0(&audio).unwrap()[0], 100.0);
        assert!(!model.apply_quality(&QualityLevel::at(3)), "already there");
        // Back to full quality: the voice's own extractor and retrieval again.
        assert!(model.apply_quality(&QualityLevel::FULL));
        assert_eq!(pitch.f0(&audio).unwrap()[0], 200.0);
        // Without a light extractor the wrapper is not added: the step is the model's alone.
        let mut plain = governed(Box::new(Retrieval(false)), None);
        assert!(!plain.apply_quality(&QualityLevel::at(3)));
        assert!(is_light_f0("swiftf0") && is_light_f0(" dio ") && !is_light_f0("rmvpe") && !is_light_f0("fusion"));
    }

    /// With the governor on, a voice whose F0 is already the light one gets no second extractor,
    /// and a legacy voice keeps the governor lever (it is the runtime's) while its unsupported
    /// levers fall back.
    #[test]
    fn the_governor_lever_is_kept_for_every_voice() {
        use crate::services::neural_audio::vc::test_models;
        let (root, slot, hubert) = test_models::write_slot("factory-governor");
        let mut req = test_request(&slot);
        req.hubert_path = hubert.to_string_lossy().into_owned();
        req.levers.levers.set(Lever::Governor, true);
        req.levers.levers.set(Lever::GpuKnn, true);
        let built = build_pitch(&req).unwrap();
        assert!(built.light_f0.is_none() && built.light_note.is_none(), "dio is already light");
        let created = create_engine(req).unwrap();
        assert_eq!(created.format, SlotFormat::LegacyOnnxRvc);
        assert!(created.levers.levers.get(Lever::Governor), "the runtime's lever stays on");
        assert!(!created.levers.levers.get(Lever::GpuKnn), "not run by bring-your-own ONNX voices");
        assert!(created.levers.notes.iter().any(|n| n.starts_with("gpu_knn:")), "{:?}", created.levers.notes);
        let _ = std::fs::remove_dir_all(&root);
    }

    /// T8: a bring-your-own ONNX voice keeps its export's zero noise when anchored noise is `auto`
    /// (the lever table is measured on template voices only); an explicit `on` anchors it.
    #[test]
    fn byo_voices_anchor_their_noise_only_when_asked() {
        use crate::services::neural_audio::vc::test_models;
        let (root, slot, hubert) = test_models::write_slot("factory-byo-noise");
        for (mode, used) in [(LeverMode::Auto, false), (LeverMode::On, true)] {
            let mut req = test_request(&slot);
            req.hubert_path = hubert.to_string_lossy().into_owned();
            req.levers.levers.set(Lever::AnchoredNoise, true);
            req.anchored_noise_mode = mode;
            let created = create_engine(req).unwrap();
            assert_eq!(created.format, SlotFormat::LegacyOnnxRvc);
            assert_eq!(created.levers.levers.get(Lever::AnchoredNoise), used, "{mode:?}");
            let noted = created.levers.notes.iter().any(|n| n.starts_with("anchored_noise:"));
            assert_eq!(noted, !used, "{mode:?}: {:?}", created.levers.notes);
        }
        let _ = std::fs::remove_dir_all(&root);
    }

    /// A `sigil-rvc-1` slot reaches the Sigil Voice Engine with the request the factory built (the
    /// test fixture's toy template runs at another frame rate, which the engine refuses by name:
    /// that refusal proves the template, weights, speakers and encoder were all resolved).
    #[test]
    fn template_voices_dispatch_to_the_sigil_engine() {
        use crate::services::neural_audio::vc::engine::generator::tests::{tiny_dir, tiny_voice};
        use crate::services::neural_audio::vc::test_models;
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!("sigil-vc-factory-sve-{}-{nanos}", std::process::id()));
        let slot = root.join("tiny");
        std::fs::create_dir_all(&slot).unwrap();
        let voice = tiny_voice();
        std::fs::write(slot.join("weights.f32.bin"), &voice.weights.f32_image).unwrap();
        let speakers: Vec<u8> = voice.weights.speakers.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(slot.join("speakers.f32.bin"), speakers).unwrap();
        let manifest = serde_json::json!({
            "format": "sigil-rvc-1",
            "template": voice.template.layout.template,
            "weights": "weights.f32.bin",
            "speakers": "speakers.f32.bin",
            "embedder": "contentvec",
        });
        std::fs::write(slot.join("manifest.json"), serde_json::to_vec(&manifest).unwrap()).unwrap();
        let encoder = root.join("content.onnx");
        std::fs::write(&encoder, test_models::content_encoder_bytes()).unwrap();
        let mut req = test_request(&slot);
        req.templates_dir = Some(tiny_dir());
        req.hubert_path = encoder.to_string_lossy().into_owned();
        assert_eq!(slot_format(&slot).unwrap(), SlotFormat::SigilRvc);
        let err = create_engine(req).err().expect("the toy template is not 100 frames per second");
        assert!(err.contains("frames per second"), "{err}");
        let _ = std::fs::remove_dir_all(&root);
    }
}
