//! Engine factory (P3-E): turns the resolved params into the model behind the stream harness,
//! dispatching on the slot's `manifest.format` (plan: "Model families behind VoiceConversionModel
//! with a manifest.format factory"):
//! - `rvc` (bring-your-own ONNX, or no manifest): the legacy full-decode `RvcModel` (unchanged;
//!   the `anchored_noise` lever feeds its `rnd` input time-anchored noise, and an export with
//!   `skip_head` / `return_length` decodes only the returned frames);
//! - `sigil-rvc-1` (imported template voice): the Sigil Voice Engine (`vc::engine`, P3-C1);
//! - a P5 family format (`ddsp-svc-6.3`, `sigil-zs-1`, `meanvc-2`, `sovits-4.1`, `sidecar-v2`):
//!   `vc::families::create` ([`create_family`]; P5-S0), refused by name while the family is not
//!   available in the build, on an EP rung it does not run on, or (zero-shot) without a reference;
//! - anything else: refused.
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
//!
//! # Added by P5-S0 (frozen; change only through INTEGRATE)
//! - `EngineRequest::{reference_voice, reference_clip}`: the zero-shot reference (a voice-library
//!   id, params `reference_voice_id`; or a WAV, benchmarks only), resolved by [`resolve_reference`].
//! - `CreatedEngine::reference`: the family's reference embedding; [`LoadedVoice::load_cancellable`] installs
//!   it, with the voice's singing flags (`f0_range`), in the harness
//!   ([`build_harness_with`] / `VcStreamHarness::with_context`).
//! - `EngineKey::reference_voice`: a voice with another reference is another voice (a switch loads
//!   it in the background; families may share their graphs between such loads).
//! - [`with_context`]: a wrapper that hands a model its context on every call when no harness of the
//!   factory's builds it (the benchmark's own harnesses).
//!
//! # Added by P5-I0
//! - `CreatedEngine::min_output_buffer_ms` / `LoadedVoice::min_output_buffer_ms`: the voice's
//!   smallest output buffer (the loaded voice's `FamilyEngine::min_output_buffer_ms`, else
//!   `families::min_output_buffer_ms` of its kind, slot options and EP); the worker plays the active
//!   voice with the larger of it and the user's prefill / queue.
//! - The in-harness denoiser is skipped (with a warning) for a model whose input is not 16 kHz.
//! - `EngineRequest::cancel`: a background load ([`VoiceSet`]) and the worker's first load hand the
//!   family their cancel flag, so a sidecar voice stops waiting for its helper once nobody wants it.

//! # Added by P3B-H
//! - [`create_engine_for_stream`] / [`StreamHint`]: a template voice behind a stream harness of known
//!   geometry plans its final shapes at load (the lookahead, the harness's SOLA tail and extra
//!   context), builds its generator once and its pitch bank, content encoder (with its plan) and
//!   generator beside each other (`SigilRvcEngine::load_hooked`); `SIGIL_VC_FINAL_SHAPES=0` and
//!   `SIGIL_VC_PARALLEL_LOAD=0` are the kill switches. A background hot-switch load
//!   ([`LoadedVoice::load_background`]) builds sequentially at a lower priority.
//! - [`HarnessSpec::sola_search_ms`]: the SOLA search window of the harness (and of the planned tail).
//! - [`LoadedVoice::own_mb`] / [`LoadedVoice::shared`]: a shared RMVPE plan is counted once by
//!   [`VoiceSet::vram_mb`] and leaves with its last holder.

use std::cell::Cell;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::denoise::{self, harness::DenoisedModel, DenoiserKind};
use crate::services::neural_audio::gpu::arena::VramProbe;
use crate::services::neural_audio::gpu::GpuLevers;
use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::telemetry::{self, LoadStage};
use crate::services::neural_audio::vc::engine::{
    self, presets, EffectiveLevers, F0Range, Lever, LeverMode, PrecisionMode, SigilRvcEngine, SigilRvcRequest, StreamGeometry,
};
use crate::services::neural_audio::vc::f0_fusion::{create_pitch_extractor, create_pitch_extractor_with, PitchBankRequest};
use crate::services::neural_audio::vc::manifest::{load_slot_manifest, SlotFormat};
use crate::services::neural_audio::vc::rmvpe::shared_components as shared_pitch_components;
use crate::services::neural_audio::vc::rvc::LegacyNoise;
use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::engine::post::{PostConfig, PostStage};
use crate::services::neural_audio::vc::families::{self, reference::ReferenceClip, FamilyKind, FamilyRequest, CREATION_REFERENCE};
use crate::services::neural_audio::vc::stream::{sola_want_16k, HarnessContext, HarnessOptions};
use crate::services::neural_audio::vc::traits::{
    ConversionCtx, ModelTelemetry, PitchExtractor, ReferenceEmbedding, SingingFlags, StreamReport, VoiceControls, DEFAULT_NOISE_SCALE,
};
use crate::services::neural_audio::vc::{assets, import, VcStreamHarness, VoiceConversionModel};
use crate::services::neural_audio::{RvcModel, RvcPresetProfile};
use crate::services::resources::{self, Subsystem, VramClaim};
use crate::services::rt_thread::{self, WorkerPriority};

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
    /// P5-S0: the voice-library voice a zero-shot family converts toward (params
    /// `reference_voice_id`; `""` = none). Ignored by families without a reference.
    pub reference_voice: String,
    /// P5-S0: a reference WAV used directly (wins over `reference_voice`; benchmarks and tests).
    pub reference_clip: Option<PathBuf>,
    /// P5-S0: the latency budget (ms) when the voice loads: a family's default stall timeout (a
    /// sidecar's per-frame timeout); the runtime's budget itself stays the worker's.
    pub latency_budget_ms: f64,
    /// Set when the voice changer no longer wants this voice (another was selected, or it stopped;
    /// not persisted): a family that waits on something outside the process (a sidecar helper's
    /// start or handshake) stops waiting (`FamilyRequest::cancel`). `None` = never called off.
    pub cancel: Option<Arc<AtomicBool>>,
    /// P3B-S0: the content encoder's left context of template voices (ms, params
    /// `content_context_ms`; `None` = the preset's), handed to `SigilRvcRequest::content_context_16k`
    /// (a no-op until P3B-D).
    pub content_context_ms: Option<f64>,
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
    /// The reference embedding of a zero-shot family voice (P5-S0; installed in its harness).
    pub reference: Option<Arc<ReferenceEmbedding>>,
    /// The smallest output buffer (prefill, max queue; ms) the voice plays with
    /// (`families::min_output_buffer_ms`, P5-I0); `None` = the user's.
    pub min_output_buffer_ms: Option<(f64, f64)>,
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
/// `governor` lever on, also the light extractor its `light_f0` step switches to. `shapes` are the
/// audio frames of one incremental F0 call and the longest whole-window call (16 kHz samples) of the
/// voice's final plan, which size RMVPE's plan (P3B-E / P3B-H); `None` when the voice's plan is not
/// known yet (the plan then assumes the Low-latency window).
fn build_pitch(request: &EngineRequest, shapes: Option<(usize, usize)>) -> Result<BuiltPitch, String> {
    let _span = telemetry::load_span(LoadStage::PitchPlain);
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
            precision: request.precision,
            expected_call_frames: shapes.map(|s| s.0),
            max_context_16k: shapes.map(|s| s.1),
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
            precision: request.precision,
            expected_call_frames: None,
            max_context_16k: None,
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

    // P3B-S0: the voice's extractor sizes the cache geometry; the light one never needs more.
    fn preferred_input_frames(&self) -> Option<usize> {
        self.main.preferred_input_frames()
    }

    fn max_call_frames(&self) -> Option<usize> {
        match (self.main.max_call_frames(), self.light.max_call_frames()) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        }
    }

    fn last_salience(&self) -> Option<(&[f32], usize)> {
        if self.use_light.load(Ordering::Relaxed) {
            self.light.last_salience()
        } else {
            self.main.last_salience()
        }
    }

    fn f0_into(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<(), String> {
        if self.use_light.load(Ordering::Relaxed) {
            self.light.f0_into(mono_16k, out)
        } else {
            self.main.f0_into(mono_16k, out)
        }
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

    fn time_anchored(&self) -> bool {
        self.inner.time_anchored()
    }

    fn telemetry(&self) -> Option<&dyn ModelTelemetry> {
        self.inner.telemetry()
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
    create_engine_for_stream(request, denoiser, None)
}

thread_local! {
    /// Set on the loader thread of a background hot-switch load (see [`BackgroundLoad`]).
    static BACKGROUND_LOAD: Cell<bool> = const { Cell::new(false) };
}

/// Marks the calling thread as a background voice load for its lifetime (P3B-H): the load runs its
/// branches one after the other (the live voice keeps the cores) and the thread runs below normal
/// priority with the system-managed EcoQoS state. The first load of a run is in the foreground and
/// builds in parallel. It also holds an RMVPE [`WarmupScope`](crate::services::neural_audio::vc::rmvpe::WarmupScope)
/// for its lifetime, so a voice that joined the live voice's shared RMVPE plan runs its warm-up
/// calls right behind the live voice's call instead of taking the plan's lock at an arbitrary moment.
struct BackgroundLoad {
    _warmup: crate::services::neural_audio::vc::rmvpe::WarmupScope,
}

impl BackgroundLoad {
    fn enter() -> BackgroundLoad {
        BACKGROUND_LOAD.with(|b| b.set(true));
        // The loader thread is dedicated to this load: nothing to restore afterwards.
        let _ = rt_thread::set_worker_priority(WorkerPriority::ProtectGame, false);
        BackgroundLoad {
            _warmup: crate::services::neural_audio::vc::rmvpe::warmup_scope(),
        }
    }

    fn active() -> bool {
        BACKGROUND_LOAD.with(Cell::get)
    }
}

impl Drop for BackgroundLoad {
    fn drop(&mut self) {
        BACKGROUND_LOAD.with(|b| b.set(false));
    }
}

/// What the stream harness around a template voice will ask for (P3B-H): the load plans the voice's
/// final shapes from it, so the harness's `configure_stream` finds them equal and rebuilds nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StreamHint {
    /// The harness's crossfade (ms).
    pub crossfade_ms: f64,
    /// The harness's extra context (ms).
    pub extra_context_ms: f64,
    /// The harness's SOLA search window (ms); `None` = automatic.
    pub sola_search_ms: Option<f64>,
}

impl StreamHint {
    /// The hint of a run's harness spec.
    pub(crate) fn of(spec: &HarnessSpec) -> StreamHint {
        StreamHint {
            crossfade_ms: spec.crossfade_ms,
            extra_context_ms: spec.extra_context_ms,
            sola_search_ms: spec.sola_search_ms,
        }
    }

    /// The geometry of a template voice with engine blocks of `block_16k` samples and output rate
    /// `out_sr`: the tick's `want` (block + SOLA tail, as `VcStreamHarness` computes it), the extra
    /// context and the crossfade.
    fn geometry(&self, block_16k: usize, out_sr: u32) -> StreamGeometry {
        StreamGeometry {
            want_16k: sola_want_16k(block_16k, self.crossfade_ms, out_sr, harness_options(self.sola_search_ms)),
            harness_extra_16k: ((self.extra_context_ms / 1000.0) * 16_000.0).round().max(0.0) as usize,
            crossfade_ms: self.crossfade_ms,
        }
    }
}

/// The harness options of a run: its SOLA search window, the environment's overrides for the rest.
fn harness_options(sola_search_ms: Option<f64>) -> HarnessOptions {
    HarnessOptions {
        sola_search_ms,
        ..HarnessOptions::default()
    }
    .with_env()
}

/// [`create_engine_with`] for an engine that will run behind a stream harness of the given geometry
/// (`LoadedVoice::load_cancellable`, the benchmark): a template voice plans its final shapes at load.
pub(crate) fn create_engine_for_stream(request: EngineRequest, denoiser: DenoiserKind, hint: Option<&StreamHint>) -> Result<CreatedEngine, String> {
    let format = slot_format(&request.slot_dir)?;
    let probe = VramProbe::for_rung(&request.ep, request.session.cuda_device_index);
    let mut created = match format {
        SlotFormat::LegacyOnnxRvc => create_legacy(&request)?,
        SlotFormat::SigilRvc => create_sigil(&request, hint)?,
        SlotFormat::Family(kind) => create_family(&request, kind)?,
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
    } = build_pitch(request, None)?;
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
        reference: None,
        min_output_buffer_ms: None,
    })
}

/// What a template voice's pitch branch reports besides the extractor (filled on its thread).
#[derive(Default)]
struct PitchMeta {
    notes: Vec<String>,
    outcomes: Vec<(String, EpOutcome)>,
    light_f0: Option<Arc<AtomicBool>>,
    light_note: Option<String>,
}

fn create_sigil(request: &EngineRequest, hint: Option<&StreamHint>) -> Result<CreatedEngine, String> {
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
    let preset = presets::engine_preset(
        request.preset,
        request.block_ms,
        request.chunk_frames,
        request.device_rate,
        request.extra_context_ms,
        request.precision,
    );
    let noise_seed = noise_seed_for(&request.slot_dir, manifest.source_sha256.as_deref());
    // P3B-H: the load plans the voice's final shapes (lookahead, the harness's SOLA tail and extra
    // context) and builds the pitch bank beside its other branches. With the kill switch
    // (`SIGIL_VC_FINAL_SHAPES=0`) the load plans P3's defaults and the lookahead is applied below.
    let final_shapes = SigilRvcEngine::final_shapes_enabled();
    let stream = hint
        .filter(|_| final_shapes)
        .and_then(|h| plan_block_16k(&preset).map(|block| h.geometry(block, manifest.sample_rate)));
    let pitch_meta: Mutex<Option<PitchMeta>> = Mutex::new(None);
    let pitch_builder = |f0_frames: usize, window_16k: usize| -> Result<Box<dyn PitchExtractor>, String> {
        let b = build_pitch(request, Some((f0_frames, window_16k)))?;
        *pitch_meta.lock().unwrap_or_else(std::sync::PoisonError::into_inner) = Some(PitchMeta {
            notes: b.notes,
            outcomes: b.outcomes,
            light_f0: b.light_f0,
            light_note: b.light_note,
        });
        Ok(b.extractor)
    };
    let mut engine = SigilRvcEngine::load_hooked(
        SigilRvcRequest {
            slot_dir: request.slot_dir.clone(),
            manifest,
            templates_dir,
            ep: request.ep.clone(),
            session: request.session.clone(),
            levers: request.levers.clone(),
            pitch: SigilRvcEngine::pending_pitch(),
            content_encoder: opt_path(&request.hubert_path),
            index: None,
            preset,
            f0_range: request.f0_range,
            noise_seed,
            lookahead_ms: if final_shapes { request.lookahead_ms } else { 0.0 },
            stream,
            content_context_16k: request
                .content_context_ms
                .filter(|ms| ms.is_finite() && *ms >= 0.0)
                .map(|ms| (ms * 16.0).round() as usize),
        },
        Some(Box::new(pitch_builder)),
        request.cancel.as_deref(),
        // The load validates the voice with one conversion where a stream harness warms it next (a
        // caller that passes the stream's geometry builds that harness: `LoadedVoice::load_cancellable`,
        // which refuses the voice if the warm-up fails, see `warmup_verdict`); the other callers (the
        // benchmark's builds) keep the two validation conversions.
        hint.is_some(),
        // A background hot-switch load builds sequentially.
        BackgroundLoad::active(),
    )?;
    let PitchMeta {
        notes: mut warnings,
        mut outcomes,
        light_f0,
        light_note,
    } = pitch_meta
        .into_inner()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .unwrap_or_default();
    engine.set_silence_skip(request.silence_skip);
    let mut lookahead_note = None;
    if !final_shapes && request.lookahead_ms > 0.0 {
        // The kill switch's flow (P5): lookahead after the load, before the harness is built (it sizes
        // its context from the plan). Not on a DirectML content encoder without a fixed shape: with
        // lookahead its window shape costs 35-49 ms per call there instead of about 4 ms.
        let content_on_dml = engine
            .ep_outcomes()
            .iter()
            .any(|(c, o)| c.ends_with("vc.content") && o.active == "dml");
        // The engine has run its warm-up tick by now, so this reports what the loaded voice really
        // runs (an encoder whose plan was refused keeps the skip).
        if content_on_dml && !engine.content_runs_fixed_shape() {
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
        reference: None,
        min_output_buffer_ms: None,
    })
}

/// The engine block of a preset in 16 kHz samples (`None` for a block the plan refuses: the load
/// reports that error).
fn plan_block_16k(preset: &presets::EnginePreset) -> Option<usize> {
    engine::plan::block_frames(preset.block_ms).ok().map(|f| f * 160)
}

/// The reference clip of `request`: its explicit WAV, else its voice-library voice (from
/// `request.app_data`), else none.
pub(crate) fn resolve_reference(request: &EngineRequest) -> Result<Option<ReferenceClip>, String> {
    if let Some(path) = &request.reference_clip {
        return ReferenceClip::from_wav(path).map(Some);
    }
    let id = request.reference_voice.trim();
    if id.is_empty() {
        return Ok(None);
    }
    let app_data = request
        .app_data
        .as_deref()
        .ok_or("the voice library is not available (no app data folder)")?;
    crate::services::voice_library::reference_clip(app_data, id).map(Some)
}

/// A P5 family voice (`vc::families`): its pitch extractor (when the family uses one, with the
/// governor's light-F0 switch), its reference clip (zero-shot families), then the family's own
/// loader. Refused by name when the family is not available in this build, does not run on the
/// request's EP rung, or (made from a reference) has no reference voice.
fn create_family(request: &EngineRequest, kind: FamilyKind) -> Result<CreatedEngine, String> {
    if !kind.available() {
        return Err(format!("the {} voice family is not available in this build yet", kind.as_str()));
    }
    let info = kind.info();
    if !info.execution_providers.contains(&request.ep.as_str()) {
        return Err(format!(
            "{} voices run on {} only (this voice changer runs on {}); choose another execution provider",
            kind.as_str(),
            info.execution_providers.join(" / "),
            request.ep
        ));
    }
    let manifest = load_slot_manifest(&request.slot_dir)?;
    if let Some(problem) = families::slot_problem(kind, &request.slot_dir, &manifest) {
        return Err(problem);
    }
    let reference = if info.zero_shot { resolve_reference(request)? } else { None };
    if info.creation == CREATION_REFERENCE && reference.is_none() {
        return Err(format!("choose a reference voice from the voice library for this {} voice", kind.as_str()));
    }
    let (pitch, mut warnings, mut outcomes, light_f0, light_note) = if info.needs_pitch {
        let b = build_pitch(request, None)?;
        (Some(b.extractor), b.notes, b.outcomes, b.light_f0, b.light_note)
    } else {
        (None, Vec::new(), Vec::new(), None, None)
    };
    let noise_seed = noise_seed_for(&request.slot_dir, manifest.source_sha256.as_deref());
    // The kind's minimum for this slot's options on the request's EP (before the manifest moves);
    // a loaded voice that knows better (the EP its graphs landed on) overrides it below.
    let kind_min_output_buffer_ms = families::min_output_buffer_ms(kind, &request.ep, &manifest);
    let engine = families::create(FamilyRequest {
        kind,
        slot_dir: request.slot_dir.clone(),
        manifest,
        app_data: request.app_data.clone(),
        ep: request.ep.clone(),
        session: request.session.clone(),
        levers: request.levers.clone(),
        precision: request.precision,
        preset: request.preset,
        block_ms: request.block_ms,
        chunk_frames: request.chunk_frames,
        device_rate: request.device_rate,
        extra_context_ms: request.extra_context_ms,
        lookahead_ms: request.lookahead_ms,
        f0_range: request.f0_range,
        silence_skip: request.silence_skip,
        noise_seed,
        pitch,
        pitch_method: request.pitch_extractor.clone(),
        content_encoder: opt_path(&request.hubert_path),
        reference,
        latency_budget_ms: request.latency_budget_ms,
        cancel: request.cancel.clone(),
    })?;
    warnings.extend(engine.warnings);
    outcomes.extend(engine.outcomes);
    let mut levers = engine.levers;
    levers.notes.extend(light_note);
    let min_output_buffer_ms = engine.min_output_buffer_ms.or(kind_min_output_buffer_ms);
    if let Some((prefill, queue)) = min_output_buffer_ms {
        levers.notes.push(format!(
            "output buffer: at least {prefill:.0} / {queue:.0} ms for {} voices (each block computes in one burst)",
            kind.as_str()
        ));
    }
    let output_sample_rate = engine.model.output_sample_rate();
    Ok(CreatedEngine {
        model: governed(engine.model, light_f0),
        format: SlotFormat::Family(kind),
        output_sample_rate,
        levers,
        outcomes,
        warnings,
        vram_mb: 0.0,
        denoiser_latency_16k: 0,
        model_delay_16k: engine.model_delay_16k,
        reference: engine.reference,
        min_output_buffer_ms,
    })
}

/// Hands `model` the harness context on every call whose context lacks it (a harness the factory
/// did not build, such as the benchmark's): the reference when the call has none, the singing flags
/// when the call has the neutral ones. A harness built with the context passes it already.
pub(crate) fn with_context(model: Box<dyn VoiceConversionModel>, context: HarnessContext) -> Box<dyn VoiceConversionModel> {
    if context.reference.is_none() && context.singing == SingingFlags::SPEECH {
        return model;
    }
    Box::new(ContextModel { inner: model, context })
}

/// See [`with_context`].
struct ContextModel {
    inner: Box<dyn VoiceConversionModel>,
    context: HarnessContext,
}

impl VoiceConversionModel for ContextModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let reference = ctx.reference.or(self.context.reference.as_deref());
        let singing = if ctx.singing == SingingFlags::SPEECH { self.context.singing } else { ctx.singing };
        let filled = ConversionCtx { reference, singing, ..*ctx };
        self.inner.convert(&filled)
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
        let reference = ctx.reference.or(self.context.reference.as_deref());
        let singing = if ctx.singing == SingingFlags::SPEECH { self.context.singing } else { ctx.singing };
        let filled = ConversionCtx { reference, singing, ..*ctx };
        self.inner.warmup_at(&filled)
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

    fn stream_report(&self) -> Option<StreamReport> {
        self.inner.stream_report()
    }

    fn time_anchored(&self) -> bool {
        self.inner.time_anchored()
    }

    fn telemetry(&self) -> Option<&dyn ModelTelemetry> {
        self.inner.telemetry()
    }
}

/// Wrap the model with GTCRN at 16 kHz (CPU, one thread: it is tiny). A denoiser that cannot load
/// is a warning; the voice runs without it. A model whose input is not 16 kHz (a 22.05 kHz
/// sidecar, P5-I0) runs without it too: the denoiser never takes the model away.
fn wrap_gtcrn(created: &mut CreatedEngine, request: &EngineRequest) {
    let rate = created.model.input_sample_rate();
    if rate != DENOISER_RATE {
        created.warnings.push(format!(
            "denoiser: the 16 kHz denoiser cannot run on this voice's {rate} Hz input; the voice runs without it"
        ));
        return;
    }
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

    fn time_anchored(&self) -> bool {
        self.inner.time_anchored()
    }

    fn telemetry(&self) -> Option<&dyn ModelTelemetry> {
        self.inner.telemetry()
    }
}

/// The in-harness denoiser's rate (GTCRN).
const DENOISER_RATE: u32 = 16_000;

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
    /// SOLA search window (ms; params `sola_search_ms`, P3B-H); `None` = automatic (half the overlap).
    pub sola_search_ms: Option<f64>,
}

/// Build the stream harness around a created engine (warms it at the live shape).
#[allow(dead_code)] // P3-E frozen API (voices load through `build_harness_with` since P5-S0).
pub(crate) fn build_harness(model: Box<dyn VoiceConversionModel>, spec: &HarnessSpec) -> Result<VcStreamHarness, String> {
    build_harness_with(model, spec, HarnessContext::default())
}

/// [`build_harness`] with the voice's context (its reference and singing flags), passed to the
/// warm-up and every call (P5-S0).
pub(crate) fn build_harness_with(model: Box<dyn VoiceConversionModel>, spec: &HarnessSpec, context: HarnessContext) -> Result<VcStreamHarness, String> {
    let model_out_sr = model.output_sample_rate();
    let extra_samples = ((spec.extra_context_ms / 1000.0) * 16_000.0).round().max(0.0) as usize;
    let crossfade_samples = ((spec.crossfade_ms / 1000.0) * f64::from(model_out_sr)).round().max(0.0) as usize;
    VcStreamHarness::with_options(
        model,
        spec.r_in,
        spec.r_out,
        spec.ch_in,
        spec.ch_out,
        extra_samples,
        crossfade_samples,
        spec.chunk_frames,
        context,
        harness_options(spec.sola_search_ms),
    )
}

/// What identifies a loaded voice: a switch reuses a loaded one only when all of it matches.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EngineKey {
    pub slot_dir: PathBuf,
    pub pitch_extractor: String,
    pub f0_range: F0Range,
    pub denoiser: DenoiserKind,
    /// P5-S0: the zero-shot reference voice (params `reference_voice_id`, trimmed; `""` = none).
    pub reference_voice: String,
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
    /// The smallest output buffer (prefill, max queue; ms) it plays with (P5-I0; the worker uses
    /// the larger of it and the user's while this voice is active).
    pub min_output_buffer_ms: Option<(f64, f64)>,
    /// VRAM of this voice alone (MiB): `vram_mb` less the shared components it built (P3B-H; equal to
    /// `vram_mb` when it built none).
    pub own_mb: f64,
    /// The shared components (a pooled RMVPE plan) it holds, built or joined. [`VoiceSet::vram_mb`]
    /// counts each once across the set, and a component goes with its last holder.
    pub shared: Vec<SharedHold>,
    /// Resource-governor claim (resized to the measured VRAM; 0 MB on the CPU).
    _claim: VramClaim,
    pub load_ms: f64,
    /// Its load warnings were shown (they are toasted on its first activation only, not on every
    /// switch back to it).
    pub warnings_shown: bool,
}

/// A VRAM component several voices of a run can hold at once (P3B-E's RMVPE plan; the content
/// encoder's plans are part of the building voice's measurement until the pool lists them).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SharedHold {
    pub id: String,
    pub mb: f64,
}

/// `(id, holders, MiB)` of every shared component alive now.
fn shared_snapshot() -> Vec<(String, usize, f64)> {
    shared_pitch_components()
        .into_iter()
        .map(|c| (c.id, c.users, c.vram_mb.unwrap_or(0.0)))
        .collect()
}

/// Voices released so far (started, finished): [`release`] frees an evicted voice on its own thread,
/// which drops that voice's hold on a shared plan at some point between the two. A load compares the
/// marks around its shared-plan snapshots to know whether holders may have left meanwhile.
static RELEASES_STARTED: AtomicU64 = AtomicU64::new(0);
static RELEASES_DONE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ReleaseMark {
    started: u64,
    done: u64,
}

impl ReleaseMark {
    fn now() -> ReleaseMark {
        // `done` first: a release that starts and finishes in between makes `started` larger, never smaller.
        let done = RELEASES_DONE.load(Ordering::SeqCst);
        ReleaseMark { started: RELEASES_STARTED.load(Ordering::SeqCst), done }
    }

    fn in_flight(self) -> bool {
        self.started != self.done
    }

    /// Whether any voice release began, ended or was under way between the marks `before` (taken
    /// before the first snapshot) and `after` (taken after the second): its drop may have lowered a
    /// plan's holder count between the snapshots.
    fn churned(before: ReleaseMark, after: ReleaseMark) -> bool {
        before != after || before.in_flight() || after.in_flight()
    }
}

/// F0 methods whose plan can be a shared RMVPE plan on a GPU rung (`rmvpe`, the fusion's pitch part,
/// and `crepe`, which runs RMVPE while its own model is not installed).
fn may_hold_rmvpe_plan(ep: &str, pitch_method: &str) -> bool {
    !ep.trim().eq_ignore_ascii_case("cpu") && matches!(pitch_method.trim(), "rmvpe" | "fusion" | "crepe")
}

/// The components the voice that loaded between `before` and `after` holds (more holders than
/// before: built or joined), and the VRAM of those it built (nobody held them before).
///
/// A join is seen as one more holder, which a holder leaving between the snapshots hides (voices
/// are released on their own thread). With `uncertain` (voices were released around the load, and
/// this voice can hold such a plan) every plan that was held before and still is counts as held by
/// this voice too: a plan credited to a voice that does not hold it only makes the budget evict early,
/// one that is missed leaves VRAM over-committed. (Exact bookkeeping needs the extractor to report the
/// plan it holds, `RmvpePitchExtractor::shared_component_id`; it does not reach the factory yet.)
fn holds_since(before: &[(String, usize, f64)], after: &[(String, usize, f64)], uncertain: bool) -> (Vec<SharedHold>, f64) {
    let (mut holds, mut built) = (Vec::new(), 0.0);
    for (id, users, mb) in after {
        let was = before.iter().find(|b| &b.0 == id).map_or(0, |b| b.1);
        if *users > was || (uncertain && was > 0) {
            holds.push(SharedHold { id: id.clone(), mb: *mb });
            if was == 0 {
                built += mb;
            }
        }
    }
    (holds, built)
}

/// A template voice whose stream warm-up failed does not load: its two F0 paths (whole window and
/// incremental) run in that warm-up, so a failure there fails every live tick. (The load's own check
/// is one conversion where a harness warm-up follows; this keeps the second one fatal, as before.)
fn warmup_verdict(format: &SlotFormat, slot_dir: &Path, failure: Option<&str>) -> Result<(), String> {
    match failure {
        Some(e) if *format == SlotFormat::SigilRvc => Err(format!("voice {}: test conversion failed: {e}", slot_dir.display())),
        _ => Ok(()),
    }
}

/// Error of a background load stopped because nobody wanted the voice any more.
pub(crate) const LOAD_CANCELLED: &str = "the voice load was cancelled";

impl LoadedVoice {
    /// Build a voice: engine, harness (warm-up included) and its VRAM claim. Stops at its next
    /// stage (before the sessions, the warm-up at the live shape, the VRAM claim) once `cancel` is
    /// set, with [`LOAD_CANCELLED`] (the worker's first load and every background load pass their
    /// own flag, also as `request.cancel`).
    pub(crate) fn load_cancellable(key: EngineKey, request: EngineRequest, spec: &HarnessSpec, cancel: &AtomicBool) -> Result<LoadedVoice, String> {
        let check = || if cancel.load(Ordering::Relaxed) { Err(LOAD_CANCELLED.to_string()) } else { Ok(()) };
        check()?;
        let t0 = Instant::now();
        let label = format!("vc: {}", key.slot_dir.file_name().map(|n| n.to_string_lossy()).unwrap_or_default());
        let gpu = request.ep != "cpu";
        let probe = VramProbe::for_rung(&request.ep, request.session.cuda_device_index);
        let may_hold_plan = may_hold_rmvpe_plan(&request.ep, &request.pitch_extractor);
        let releases_before = ReleaseMark::now();
        let shared_before = shared_snapshot();
        let singing = SingingFlags::from_range(request.f0_range);
        let created = create_engine_for_stream(request, key.denoiser, Some(&StreamHint::of(spec)))?;
        check()?;
        let model_input_rate = created.model.input_sample_rate();
        let context = HarnessContext {
            reference: created.reference.clone(),
            singing,
        };
        let harness = build_harness_with(created.model, spec, context)?;
        warmup_verdict(&created.format, &key.slot_dir, harness.warmup_failure())?;
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
        let shared_after = shared_snapshot();
        let uncertain = may_hold_plan && ReleaseMark::churned(releases_before, ReleaseMark::now());
        let (shared, built_mb) = holds_since(&shared_before, &shared_after, uncertain);
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
            min_output_buffer_ms: created.min_output_buffer_ms,
            own_mb: (vram_mb - built_mb).max(0.0),
            shared,
            _claim: claim,
            load_ms: t0.elapsed().as_secs_f64() * 1000.0,
            warnings_shown: false,
        })
    }

    /// [`load_cancellable`](Self::load_cancellable) for a hot-switch load on its own thread while
    /// another voice plays (`VoiceSet`): sequential branches at a lower priority (P3B-H).
    pub(crate) fn load_background(key: EngineKey, request: EngineRequest, spec: &HarnessSpec, cancel: &AtomicBool) -> Result<LoadedVoice, String> {
        let _background = BackgroundLoad::enter();
        Self::load_cancellable(key, request, spec, cancel)
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
            min_output_buffer_ms: None,
            own_mb: vram_mb,
            shared: Vec::new(),
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
            Arc::new(LoadedVoice::load_background),
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

    /// VRAM held by the loaded voices (MiB): each voice's own, plus every shared component (a pooled
    /// RMVPE plan) once however many voices hold it, for as long as one of them does (P3B-H: the voice
    /// that built a shared plan leaving does not free it).
    pub(crate) fn vram_mb(&self) -> f64 {
        let voices = || std::iter::once(&self.active).chain(self.ready.iter()).chain(self.idle.iter());
        let own: f64 = voices().map(|v| v.own_mb).sum();
        let mut seen: Vec<&str> = Vec::new();
        let mut shared = 0.0;
        for hold in voices().flat_map(|v| v.shared.iter()) {
            if !seen.contains(&hold.id.as_str()) {
                seen.push(&hold.id);
                shared += hold.mb;
            }
        }
        own + shared
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
            let _ = tx.send(request().and_then(|mut request| {
                request.cancel = Some(flag.clone());
                loader(load_key, request, &spec, &flag)
            }));
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
    // Counted around the drop (the start before anything is dropped): `ReleaseMark` tells a load whether
    // holders of a shared plan may have left while it measured.
    RELEASES_STARTED.fetch_add(1, Ordering::SeqCst);
    let released = std::thread::Builder::new().name("sigil-vc-release".into()).spawn(move || {
        drop(voice);
        RELEASES_DONE.fetch_add(1, Ordering::SeqCst);
    });
    if let Err(e) = released {
        // The closure (and the voice in it) was dropped by the failed spawn, here.
        RELEASES_DONE.fetch_add(1, Ordering::SeqCst);
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
        assert_eq!(slot_format(&dir).unwrap(), SlotFormat::Family(FamilyKind::DdspSvc));
        std::fs::write(dir.join("manifest.json"), br#"{"format":"future-family-9"}"#).unwrap();
        assert!(slot_format(&dir).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Hot voices that share a pitch plan count it once against the VRAM budget, and it stays counted
    /// until its last holder goes, whichever voice built it (P3B-H).
    #[test]
    fn shared_components_are_counted_once_and_leave_with_their_last_holder() {
        let shared_voice = |name: &str, own: f64, shared: &[(&str, f64)]| {
            let mut v = voice(name, 0.1, own + shared.iter().map(|s| s.1).sum::<f64>());
            v.own_mb = own;
            v.shared = shared.iter().map(|(id, mb)| SharedHold { id: (*id).into(), mb: *mb }).collect();
            v
        };
        let no_loads: VoiceLoader = Arc::new(|_, _, _, _| Err("no loads".into()));
        // A built the RMVPE plan (400 MB), B joined it; each has 100 MB of its own; C holds none.
        let mut set = VoiceSet::with_loader(shared_voice("a", 100.0, &[("rmvpe-1", 400.0)]), 3, 0, no_loads.clone());
        set.idle.push(shared_voice("b", 100.0, &[("rmvpe-1", 400.0)]));
        set.idle.push(shared_voice("c", 100.0, &[]));
        assert_eq!(set.vram_mb(), 100.0 + 100.0 + 100.0 + 400.0, "the plan is counted once");
        // The builder leaves: the plan stays (B holds it), so only A's own 100 MB go.
        drop(std::mem::replace(&mut set.active, shared_voice("d", 100.0, &[])));
        assert_eq!(set.vram_mb(), 100.0 + 100.0 + 100.0 + 400.0);
        // The last holder leaves and the plan goes with it.
        set.idle.retain(|v| v.key.slot_dir != key("b").slot_dir);
        assert_eq!(set.vram_mb(), 100.0 + 100.0);
        // A budget evicts by the shared-aware total: 400 + 3 x 100 does not fit 600 MB, so the
        // oldest idle voice goes.
        let mut set = VoiceSet::with_loader(shared_voice("a", 100.0, &[("rmvpe-1", 400.0)]), 4, 600, no_loads);
        set.retire(shared_voice("b", 100.0, &[("rmvpe-1", 400.0)]));
        set.retire(shared_voice("c", 100.0, &[("rmvpe-1", 400.0)]));
        assert_eq!(set.vram_mb(), 600.0, "one voice was evicted to fit 600 MB");
        assert_eq!(set.slot_dirs().len(), 2);
    }

    fn row(id: &str, users: usize, mb: f64) -> (String, usize, f64) {
        (id.to_string(), users, mb)
    }

    /// A voice joins a shared plan (2 -> 3 holders) while an evicted voice, released on its own thread,
    /// drops its hold (3 -> 2): the counts between the load's snapshots show no join. Knowing that
    /// voices were released around the load, the join is credited anyway.
    #[test]
    fn a_join_hidden_by_a_leaving_holder_is_still_credited() {
        let before = [row("rmvpe-1", 2, 400.0)];
        // Plain join: one more holder, nothing built.
        let (holds, built) = holds_since(&before, &[row("rmvpe-1", 3, 400.0)], false);
        assert_eq!(holds, vec![SharedHold { id: "rmvpe-1".into(), mb: 400.0 }]);
        assert_eq!(built, 0.0);
        // Masked by the leaving holder: counting holders alone misses it ...
        let masked = [row("rmvpe-1", 2, 400.0)];
        assert!(holds_since(&before, &masked, false).0.is_empty());
        // ... the churn of releases makes the credit conservative.
        let (holds, built) = holds_since(&before, &masked, true);
        assert_eq!(holds, vec![SharedHold { id: "rmvpe-1".into(), mb: 400.0 }]);
        assert_eq!(built, 0.0, "a plan someone held before was not built by this voice");
        // A plan this voice built (nobody held it before) is credited and counted as built either way.
        for uncertain in [false, true] {
            let (holds, built) = holds_since(&[], &[row("rmvpe-2", 1, 300.0)], uncertain);
            assert_eq!((holds.len(), built), (1, 300.0));
        }
        // A plan that is gone (its last holder left) is not held by anyone.
        assert!(holds_since(&before, &[], true).0.is_empty());
    }

    #[test]
    fn release_churn_is_read_from_the_marks() {
        let m = |started, done| ReleaseMark { started, done };
        assert!(!ReleaseMark::churned(m(3, 3), m(3, 3)), "nothing was released around the load");
        assert!(ReleaseMark::churned(m(3, 3), m(4, 4)), "a release began and ended");
        assert!(ReleaseMark::churned(m(3, 3), m(4, 3)), "a release is under way");
        assert!(ReleaseMark::churned(m(4, 3), m(4, 4)), "one ended during the load");
        assert!(ReleaseMark::churned(m(4, 3), m(4, 3)), "one was under way throughout");
        // The real counters: a release is counted when it starts, before anything is dropped.
        let before = ReleaseMark::now();
        release(Some(voice("released", 0.1, 0.0)));
        assert!(ReleaseMark::churned(before, ReleaseMark::now()));
    }

    #[test]
    fn only_voices_that_can_hold_an_rmvpe_plan_get_the_conservative_credit() {
        for method in ["rmvpe", "fusion", "crepe", " rmvpe "] {
            assert!(may_hold_rmvpe_plan("cuda", method), "{method}");
            assert!(may_hold_rmvpe_plan("dml", method), "{method}");
            assert!(!may_hold_rmvpe_plan("cpu", method), "{method}: RMVPE is GPU-only");
        }
        assert!(!may_hold_rmvpe_plan(" CPU ", "rmvpe"));
        for method in ["swiftf0", "fcpe", "dio", "pesto"] {
            assert!(!may_hold_rmvpe_plan("cuda", method), "{method}");
        }
    }

    /// The load validates a template voice with one conversion where the stream harness warms it next;
    /// the harness's warm-up (both F0 paths) failing refuses the voice, as the load's second pass did.
    #[test]
    fn a_template_voice_with_a_failed_warmup_does_not_load() {
        let dir = Path::new("C:/voices/alice");
        let err = warmup_verdict(&SlotFormat::SigilRvc, dir, Some("rvc_convert: boom")).unwrap_err();
        assert!(err.contains("alice") && err.contains("test conversion failed") && err.contains("boom"), "{err}");
        assert!(warmup_verdict(&SlotFormat::SigilRvc, dir, None).is_ok());
        // Bring-your-own and family voices keep a warn-only warm-up.
        assert!(warmup_verdict(&SlotFormat::LegacyOnnxRvc, dir, Some("boom")).is_ok());
        assert!(warmup_verdict(&SlotFormat::Family(FamilyKind::DdspSvc), dir, Some("boom")).is_ok());
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

    /// A voice whose model does not take 16 kHz (a 22.05 kHz sidecar, P5-E's report) keeps its
    /// model when the 16 kHz denoiser is on, with a warning (P5-I0: it used to be left with the
    /// placeholder and play silence).
    #[test]
    fn the_denoiser_never_takes_a_non_16k_model_away() {
        struct At22k;
        impl VoiceConversionModel for At22k {
            fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
                Ok(vec![0.5; ctx.new_frames])
            }
            fn input_sample_rate(&self) -> u32 {
                22_050
            }
            fn output_sample_rate(&self) -> u32 {
                22_050
            }
        }
        let mut created = CreatedEngine {
            model: Box::new(At22k),
            format: SlotFormat::Family(FamilyKind::SidecarV2),
            output_sample_rate: 22_050,
            levers: EffectiveLevers::default(),
            outcomes: Vec::new(),
            warnings: Vec::new(),
            vram_mb: 0.0,
            denoiser_latency_16k: 0,
            model_delay_16k: 0,
            reference: None,
            min_output_buffer_ms: None,
        };
        wrap_gtcrn(&mut created, &test_request(Path::new("C:/nowhere")));
        assert_eq!(created.model.input_sample_rate(), 22_050, "the model stays");
        assert_eq!(created.denoiser_latency_16k, 0);
        assert!(created.warnings.iter().any(|w| w.contains("22050 Hz")), "{:?}", created.warnings);
        let input = vec![0.0f32; 441];
        let ctx = ConversionCtx {
            mono_16k: &input,
            new_frames: 441,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
            reference: None,
            singing: SingingFlags::SPEECH,
        };
        assert_eq!(created.model.convert(&ctx).unwrap().len(), 441);
    }

    /// Family voices that compute a block in one burst carry their output-buffer minimum (P5-I0).
    #[test]
    fn burst_families_carry_an_output_buffer_minimum() {
        let defaults = |k| families::family_manifest(k, "z", serde_json::json!({}));
        let zs = defaults(FamilyKind::SigilZs);
        assert_eq!(families::min_output_buffer_ms(FamilyKind::SigilZs, "cuda", &zs), Some((95.0, 165.0)));
        assert_eq!(families::min_output_buffer_ms(FamilyKind::SigilZs, "dml", &zs), Some((85.0, 155.0)));
        for ep in ["cuda", "dml", "cpu"] {
            assert_eq!(families::min_output_buffer_ms(FamilyKind::DdspSvc, ep, &defaults(FamilyKind::DdspSvc)), Some((70.0, 140.0)));
            assert_eq!(families::min_output_buffer_ms(FamilyKind::MeanVc, ep, &defaults(FamilyKind::MeanVc)), None);
        }
        let long_tick = families::family_manifest(FamilyKind::MeanVc, "z", serde_json::json!({"tick_ms": 160}));
        assert_eq!(families::min_output_buffer_ms(FamilyKind::MeanVc, "cpu", &long_tick), Some((80.0, 150.0)));
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
            reference_voice: String::new(),
            reference_clip: None,
            latency_budget_ms: 300.0,
            cancel: None,
            content_context_ms: None,
        }
    }

    /// The dev rig's request for the default voice on `ep` (40 ms blocks, RMVPE, the Low-latency
    /// preset's levers) with the app-data folder `app` (`SIGIL_TEST_APP_DATA`).
    fn dev_rig_request(app: &Path, ep: &str) -> EngineRequest {
        use crate::services::neural_audio::vc::engine::LeverRequest;
        let slot = app.join("voice-changer").join("models").join("default");
        let mut rvc = super::super::params::RvcParamsState::default();
        super::super::worker::fill_store_pitch_paths(&mut rvc, app);
        let mut request = test_request(&slot);
        request.app_data = Some(app.to_path_buf());
        request.ep = ep.to_string();
        request.pitch_extractor = "rmvpe".into();
        request.rmvpe_path = rvc.rmvpe_path.clone();
        request.block_ms = 40.0;
        request.chunk_frames = 512;
        request.levers = super::super::capabilities::resolve_levers(&LeverRequest::default(), RvcPresetProfile::LowLatency, ep, PrecisionMode::Auto);
        request
    }

    /// The reference stream (48 kHz in, 48 kHz stereo out, 512-frame blocks, 20 ms crossfade).
    fn dev_rig_spec() -> HarnessSpec {
        HarnessSpec {
            r_in: 48_000,
            r_out: 48_000,
            ch_in: 1,
            ch_out: 2,
            extra_context_ms: 64.0,
            crossfade_ms: 20.0,
            chunk_frames: 512,
            sola_search_ms: None,
        }
    }

    /// P3B-H dev-rig stress of the load path: `SIGIL_STRESS_LOADS` (default 200) loads of the real
    /// default voice per rung of `SIGIL_BENCH_EPS` (default `dml,cuda`) through the factory in one
    /// process, alternating the parallel and the sequential load, each followed by a few converted
    /// blocks and a drop (the ADR 0006 release-then-run crash class, a CUDA graph capture beside
    /// another thread's session creation, a hang). Prints the median load time of each order.
    /// `SIGIL_TEST_APP_DATA` is an app-data folder with `voice-changer/models/default` and the model
    /// store; run it alone (`--ignored load_stress_dev_rig --nocapture`) and hold the GPU timing lock.
    #[test]
    #[ignore = "dev-rig: needs SIGIL_TEST_APP_DATA (a model store with the default voice) and a GPU"]
    fn load_stress_dev_rig() {
        let Some(app) = std::env::var_os("SIGIL_TEST_APP_DATA").map(PathBuf::from) else {
            eprintln!("SIGIL_TEST_APP_DATA is not set; skipped");
            return;
        };
        std::env::set_var("SIGIL_APP_DATA_DIR", &app);
        let loads: usize = std::env::var("SIGIL_STRESS_LOADS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
        let eps = std::env::var("SIGIL_BENCH_EPS").unwrap_or_else(|_| "dml,cuda".into());
        for ep in eps.split(',').map(str::trim).filter(|e| !e.is_empty()) {
            let mut times: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
            let started = Instant::now();
            for i in 0..loads {
                let parallel = i % 2 == 0;
                std::env::set_var("SIGIL_VC_PARALLEL_LOAD", if parallel { "1" } else { "0" });
                let request = dev_rig_request(&app, ep);
                let spec = dev_rig_spec();
                let t = Instant::now();
                let created = create_engine_for_stream(request, DenoiserKind::Off, Some(&StreamHint::of(&spec))).unwrap_or_else(|e| panic!("{ep} load {i} (parallel {parallel}): {e}"));
                let load_ms = t.elapsed().as_secs_f64() * 1000.0;
                let mut harness = build_harness(created.model, &spec).unwrap_or_else(|e| panic!("{ep} harness {i}: {e}"));
                let block: Vec<f32> = (0..512).map(|k| 0.1 * (k as f32 * 0.05).sin()).collect();
                for _ in 0..16 {
                    harness.push_and_convert(&block).unwrap_or_else(|e| panic!("{ep} convert {i}: {e}"));
                }
                times[usize::from(!parallel)].push(load_ms);
                if i % 20 == 0 {
                    eprintln!("{ep}: load {i}/{loads} ({} ms, parallel {parallel}), {:.0} s elapsed", load_ms.round(), started.elapsed().as_secs_f64());
                }
                drop(harness);
            }
            let median = |v: &mut Vec<f64>| {
                v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
            };
            let (par, seq) = (median(&mut times[0]), median(&mut times[1]));
            eprintln!("{ep}: {loads} loads without a crash or a hang; median load parallel {par:.0} ms, sequential {seq:.0} ms");
        }
    }

    /// Dev-rig smoke of the real load path (`LoadedVoice::load_cancellable`, as the worker and `VoiceSet`
    /// run it): the default voice loads on the first rung of `SIGIL_BENCH_EPS` (default `cpu`) with the
    /// stream geometry as its hint, its stream warm-up passes the template verdict, the shared plan it
    /// built (a GPU rung with RMVPE) is on its holds, and it converts a few blocks. Needs
    /// `SIGIL_TEST_APP_DATA` like the stress test; run it alone (`--ignored --exact --nocapture`).
    #[test]
    #[ignore = "dev-rig: needs SIGIL_TEST_APP_DATA (a model store with the default voice)"]
    fn load_cancellable_smoke_dev_rig() {
        let Some(app) = std::env::var_os("SIGIL_TEST_APP_DATA").map(PathBuf::from) else {
            eprintln!("SIGIL_TEST_APP_DATA is not set; skipped");
            return;
        };
        std::env::set_var("SIGIL_APP_DATA_DIR", &app);
        let eps = std::env::var("SIGIL_BENCH_EPS").unwrap_or_else(|_| "cpu".into());
        let ep = eps.split(',').map(str::trim).find(|e| !e.is_empty()).unwrap_or("cpu").to_string();
        let request = dev_rig_request(&app, &ep);
        let key = EngineKey {
            slot_dir: request.slot_dir.clone(),
            pitch_extractor: request.pitch_extractor.clone(),
            f0_range: request.f0_range,
            denoiser: DenoiserKind::Off,
            reference_voice: String::new(),
        };
        let spec = dev_rig_spec();
        let cancel = AtomicBool::new(false);
        let mut voice = LoadedVoice::load_cancellable(key, request, &spec, &cancel).unwrap_or_else(|e| panic!("{ep}: {e}"));
        assert_eq!(voice.harness.warmup_failure(), None, "{ep}: a template voice with a failed warm-up must not load");
        assert_eq!(voice.format, SlotFormat::SigilRvc);
        let block: Vec<f32> = (0..512).map(|k| 0.1 * (k as f32 * 0.05).sin()).collect();
        for tick in 0..24 {
            let out = voice.harness.push_and_convert(&block).unwrap_or_else(|e| panic!("{ep} tick {tick}: {e}"));
            assert!(out.iter().all(|x| x.is_finite()), "{ep} tick {tick}");
        }
        eprintln!(
            "SMOKE {ep}: loaded in {:.0} ms, vram {:.0} MB (own {:.0}), shared holds {:?}, outcomes {:?}",
            voice.load_ms, voice.vram_mb, voice.own_mb, voice.shared, voice.outcomes
        );
    }

    /// P3B-H dev-rig measurement of the GPU prewarm: one process, one cold load of the default voice
    /// on `SIGIL_BENCH_EPS` (the first rung), after the prewarm (`SIGIL_TEST_PREWARM=1`: the cached
    /// preflight and canary of `ort_session::check_gpu_ep`, as the app runs them at start) or without
    /// it (`0`). Prints one `PREWARM_RESULT` line: the prewarm's wall time, the load's, and the
    /// process VRAM (MiB, DXGI) before, after the prewarm, after the load and after the voice is gone
    /// (the idle cost of keeping the warm-up). Run it in fresh processes, ABBA, under the GPU lock.
    #[test]
    #[ignore = "dev-rig: needs SIGIL_TEST_APP_DATA (a model store with the default voice) and a GPU"]
    fn prewarm_first_load_dev_rig() {
        use crate::services::bench::metrics::process_vram_mb;
        let Some(app) = std::env::var_os("SIGIL_TEST_APP_DATA").map(PathBuf::from) else {
            eprintln!("SIGIL_TEST_APP_DATA is not set; skipped");
            return;
        };
        std::env::set_var("SIGIL_APP_DATA_DIR", &app);
        let ep = std::env::var("SIGIL_BENCH_EPS").unwrap_or_else(|_| "cuda".into());
        let ep = ep.split(',').next().unwrap_or("cuda").trim().to_string();
        let warm = std::env::var("SIGIL_TEST_PREWARM").is_ok_and(|v| v.trim() == "1");
        let vram = || process_vram_mb().map_or(-1.0, |v| v as f64);
        let _recording = telemetry::start_recording();
        let mark0 = telemetry::load_mark();
        let before = vram();
        let t = Instant::now();
        if warm {
            crate::services::neural_audio::ort_session::check_gpu_ep(&ep, 0).unwrap_or_else(|e| panic!("prewarm {ep}: {e}"));
        }
        let prewarm_ms = t.elapsed().as_secs_f64() * 1000.0;
        let after_prewarm = vram();
        let mark1 = telemetry::load_mark();
        let request = dev_rig_request(&app, &ep);
        let spec = dev_rig_spec();
        let t = Instant::now();
        let created = create_engine_for_stream(request, DenoiserKind::Off, Some(&StreamHint::of(&spec))).unwrap_or_else(|e| panic!("{ep} load: {e}"));
        let load_ms = t.elapsed().as_secs_f64() * 1000.0;
        let after_load = vram();
        let harness = build_harness(created.model, &spec).unwrap_or_else(|e| panic!("{ep} harness: {e}"));
        drop(harness);
        let after_drop = vram();
        // The longest preload / canary span of the prewarm and of the load, and the CUDA context span.
        let longest = |events: &[telemetry::LoadEvent], key: &str| events.iter().filter(|e| e.stage.key() == key).map(|e| e.duration_ms()).fold(0.0, f64::max);
        let all = telemetry::load_events_since(mark0);
        let (pre, load_events) = all.split_at(mark1.saturating_sub(mark0).min(all.len()));
        if std::env::var_os("SIGIL_TEST_PREWARM_SPANS").is_some() {
            let t0 = load_events.iter().map(|e| e.start_ns).min().unwrap_or(0);
            let mut sorted: Vec<_> = load_events.to_vec();
            sorted.sort_by_key(|e| e.start_ns);
            for e in sorted {
                eprintln!("SPAN t{} {:8.1} +{:7.1} {}", e.thread, (e.start_ns - t0) as f64 / 1e6, e.duration_ms(), e.stage.key());
            }
        }
        eprintln!(
            "PREWARM_RESULT ep={ep} warm={warm} prewarm_ms={prewarm_ms:.0} load_ms={load_ms:.0} prewarm_canary_ms={:.0} load_canary_max_ms={:.0} load_cuda_context_ms={:.0} vram_before={before} vram_after_prewarm={after_prewarm} vram_after_load={after_load} vram_after_drop={after_drop}",
            longest(pre, "preload_canary"),
            longest(load_events, "preload_canary"),
            longest(load_events, "cuda_context"),
        );
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
            sola_search_ms: None,
        }
    }

    fn key(name: &str) -> EngineKey {
        EngineKey {
            slot_dir: PathBuf::from(format!("C:/voices/{name}")),
            pitch_extractor: "dio".into(),
            f0_range: F0Range::Speech,
            denoiser: DenoiserKind::Off,
            reference_voice: String::new(),
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
            min_output_buffer_ms: None,
            own_mb: vram_mb,
            shared: Vec::new(),
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
            Arc::new(move |key: EngineKey, request: EngineRequest, spec: &HarnessSpec, cancel: &AtomicBool| {
                // The family gets the load's own flag (a sidecar stops waiting for its helper).
                let family_cancel = request.cancel.expect("the request carries the load's cancel flag");
                assert!(std::ptr::eq(Arc::as_ptr(&family_cancel), cancel));
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
                sola_search_ms: None,
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
        let built = build_pitch(&req, None).unwrap();
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

    /// A background hot-switch load marks only its own thread, for exactly as long as it runs.
    #[test]
    fn a_background_load_marks_only_its_own_thread() {
        assert!(!BackgroundLoad::active());
        std::thread::spawn(|| {
            let guard = BackgroundLoad::enter();
            assert!(BackgroundLoad::active());
            assert!(crate::services::neural_audio::vc::rmvpe::warming_now(), "a background load holds the RMVPE warm-up scope");
            drop(guard);
            assert!(!BackgroundLoad::active());
        })
        .join()
        .unwrap();
        assert!(!BackgroundLoad::active(), "the calling thread was never marked");
    }

    /// The hint of a run's harness plans the geometry the harness asks for: the tick's `want`
    /// (block + SOLA tail) and the extra context in 16 kHz samples, with the SOLA search window.
    #[test]
    fn the_stream_hint_plans_the_harness_geometry() {
        let mut sp = spec();
        sp.crossfade_ms = 10.0;
        sp.extra_context_ms = 64.0;
        let hint = StreamHint::of(&sp);
        assert_eq!((hint.crossfade_ms, hint.extra_context_ms, hint.sola_search_ms), (10.0, 64.0, None));
        let g = hint.geometry(640, 40_000);
        assert_eq!(g.harness_extra_16k, 1_024);
        assert_eq!(g.crossfade_ms, 10.0);
        assert_eq!(g.want_16k, sola_want_16k(640, 10.0, 40_000, harness_options(None)));
        // A narrower search window asks the model for less.
        sp.sola_search_ms = Some(2.0);
        let narrow = StreamHint::of(&sp).geometry(640, 40_000);
        assert!(narrow.want_16k < g.want_16k, "{} vs {}", narrow.want_16k, g.want_16k);
    }

    // ---- P5 families (P5-S0) ----

    fn family_slot(tag: &str, kind: FamilyKind) -> (PathBuf, PathBuf) {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!("sigil-vc-factory-{tag}-{}-{nanos}", std::process::id()));
        let slot = root.join(kind.as_str());
        std::fs::create_dir_all(&slot).unwrap();
        families::write_family_manifest(&slot, &families::family_manifest(kind, "Fam", serde_json::json!({}))).unwrap();
        (root, slot)
    }

    /// A family slot reaches its family through the factory; a stub (not available yet) is refused
    /// by name before anything loads.
    #[test]
    fn family_slots_dispatch_to_their_family() {
        for kind in FamilyKind::ALL {
            let (root, slot) = family_slot("dispatch", kind);
            assert_eq!(slot_format(&slot).unwrap(), SlotFormat::Family(kind));
            let result = create_engine(test_request(&slot));
            if !kind.available() {
                let err = result.err().expect("a stub family does not load");
                assert!(err.contains(kind.as_str()), "{err}");
            }
            let _ = std::fs::remove_dir_all(&root);
        }
    }

    /// The reference resolves from an explicit WAV first, then the library; none is `None`.
    #[test]
    fn references_resolve_from_a_clip_or_the_library() {
        let mut req = test_request(Path::new("C:/nowhere"));
        assert!(resolve_reference(&req).unwrap().is_none());
        req.reference_voice = "alice".into();
        assert!(resolve_reference(&req).is_err(), "no app data: the library cannot be read");
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("sigil-vc-factory-ref-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let wav = dir.join("ref.wav");
        crate::services::bench::signals::write_wav(&wav, &vec![0.1f32; 16_000], 16_000, 1).unwrap();
        req.reference_clip = Some(wav.clone());
        let clip = resolve_reference(&req).unwrap().unwrap();
        assert_eq!((clip.path.as_path(), clip.duration_ms), (wav.as_path(), 1000));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The context wrapper hands a model the reference and singing flags its harness lacks, and
    /// is not added when there is nothing to hand.
    #[test]
    fn the_context_wrapper_fills_missing_context() {
        struct Probe(Arc<std::sync::Mutex<Vec<(bool, bool)>>>);
        impl VoiceConversionModel for Probe {
            fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
                self.0.lock().unwrap().push((ctx.reference.is_some(), ctx.singing.singing));
                Ok(vec![0.0; ctx.new_frames])
            }
            fn input_sample_rate(&self) -> u32 {
                16_000
            }
            fn output_sample_rate(&self) -> u32 {
                16_000
            }
        }
        let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
        let context = HarnessContext {
            reference: Some(Arc::new(ReferenceEmbedding { voice_id: "v".into(), ..Default::default() })),
            singing: SingingFlags { singing: true },
        };
        let mut m = with_context(Box::new(Probe(seen.clone())), context);
        let audio = vec![0.0f32; 320];
        let ctx = ConversionCtx {
            mono_16k: &audio,
            new_frames: 160,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
            reference: None,
            singing: SingingFlags::SPEECH,
        };
        m.convert(&ctx).unwrap();
        assert_eq!(seen.lock().unwrap().as_slice(), &[(true, true)]);
        // Nothing to hand: no wrapper (the model itself comes back).
        let plain = with_context(Box::new(Level(0.5)), HarnessContext::default());
        assert_eq!(plain.output_sample_rate(), 16_000);
    }
}
