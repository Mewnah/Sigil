//! The Sigil-ZS voice: `VoiceConversionModel` over the tick plans (ADR 0006; plan "VC best engine"
//! → zero-shot in-engine).
//!
//! **Streaming** (the voice changer and the benchmark's paced runs): fixed blocks of the preset
//! (`fixed_block`), the harness's SOLA tail decides the returned span, `configure_stream` sizes the
//! [`Geometry`] and takes the plans of that shape from the pool. Each tick:
//! 1. the silence skip (no GPU work after [`SILENCE_HANGOVER_MS`] of quiet below `silence_db`);
//! 2. noise: `z` and the NSF source noise of absolute grid frame `k · F + j` (`anchored_noise`), or
//!    fresh draws per tick (upstream);
//! 3. the NSF phase: anchored where this tick's returned audio meets the previous tick's (the
//!    middle of the expected SOLA crossfade; `nsf_phase0`, partial decode only), else the voice's
//!    start phases carried to the decoded slice;
//! 4. the plans' tick (content, length regulator, CFM Euler steps with CFG, partial decode or the
//!    whole-tick vocoder) into the returned span.
//!
//! **One-off conversions** (any other window, e.g. the benchmark's offline ceiling over a whole
//! utterance): the same model offline on CPU plans at that window's shape (built on first use and
//! kept for the next call of the same shape), decoded whole with upstream's `hift`.
//!
//! The voice returns its audio `right` (20 ms) before the newest input: a constant delay reported as
//! `FamilyEngine::model_delay_16k`, so the harness's length accounting is unchanged.
//!
//! **Governor** (the `governor` lever): its reduced-context levels run [`reduced_steps`] Euler
//! steps instead of the voice's (6 → 3 for `sigil_low`); the shapes, the graphs and the reference
//! stay as they are, so stepping down or up takes effect on the next tick.
//!
//! **Shared plans** (`plans::shared`): a voice that reuses plans another voice already ticked on
//! skips the warm-up ticks (nothing to warm; they would only hold the plans' lock while that voice
//! plays). The voice that built a set measures what it added on the GPU (around the build and the
//! warm-up, as its load's VRAM claim does) and hands that claim to the plans when it goes while
//! other voices still use them. A GPU voice whose heavy stages fell to the CPU through the EP
//! ladder is refused (`PlanReport::cpu_refusal`), as an explicit CPU request is.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, TryLockError};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::gpu::arena::VramProbe;
use crate::services::neural_audio::gpu::GpuLevers;
use crate::services::neural_audio::ort_session::{normalize_ep, VcSessionOptions};
use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::engine::options::{Accelerator, EffectiveLevers, Lever};
use crate::services::neural_audio::vc::families::reference::ReferenceEmbedding;
use crate::services::neural_audio::vc::traits::{ConversionCtx, StreamReport, VoiceControls, VoiceConversionModel};

use super::dsp::nsf_cycles_at;
use super::graphs::GraphSet;
use super::noise::ZsNoise;
use super::plans::{shared, PhaseMode, PlanConfig, Prepared, TickInputs, TickPlans};
use super::preset::{out_len, Geometry, Settings, HOP, MEL_BINS, N_HARM, SR_OUT, UNIT_16K, XLSR_FIELD};
use super::sampler::time_grid;

/// Quiet ticks are converted for this long before the silence skip stops inference (word endings
/// and breaths finish first).
pub const SILENCE_HANGOVER_MS: f64 = 160.0;

/// Euler steps the adaptive governor's reduced levels run (`QualityLevel::context_scale` < 1, as
/// DDSP-SVC and MeanVC2 map it): half the voice's steps, at least 3 (fewer only when the voice
/// itself runs fewer). The step count is a host loop over one graph, so the switch costs nothing.
pub fn reduced_steps(steps: u32) -> u32 {
    steps.div_ceil(2).max(steps.min(3))
}

/// The harness's SOLA geometry (`stream.rs`: overlap = crossfade clamped to [64, block], search =
/// overlap / 2, 2 samples of slack), recovered from the returned span: `(overlap, centre)` in output
/// samples.
fn sola_geometry(out_samples: usize, block_out: usize) -> (usize, usize) {
    let tail = out_samples.saturating_sub(block_out);
    let overlap = (tail.saturating_sub(2) * 2 / 3).max(1);
    let centre = tail.saturating_sub(overlap) / 2;
    (overlap, centre)
}

static NEXT_ID: AtomicU64 = AtomicU64::new(1);

/// Which sample of the previous tick the phase anchor pairs with this tick's crossfade (see
/// [`SigilZsModel::join_anchor`]). Voices use `Grid`; the other pairings are the alternatives the
/// GPU test measures (`SIGIL_ZS_JOIN`; in P5-B's join matrix `Grid` joined best).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JoinShift {
    /// `F · 256` later: where the anchored noise grid lines up (ADR 0006).
    Grid,
    /// `frames_per_block · 256` later: where the content lines up.
    #[cfg_attr(not(test), allow(dead_code))]
    Content,
    /// One block of output later: the same output time.
    #[cfg_attr(not(test), allow(dead_code))]
    Output,
}

/// The previous tick, for the phase-continuous join.
struct PhaseState {
    tick: u64,
    f0: Vec<f32>,
    harm_phase: [f32; N_HARM],
}

/// Everything a voice is built from.
pub struct ModelConfig {
    pub graphs: GraphSet,
    pub ep: String,
    pub session: VcSessionOptions,
    /// The run's levers (as resolved for the voice changer); the voice keeps what it runs.
    pub levers: EffectiveLevers,
    pub settings: Settings,
    pub reference: Arc<ReferenceEmbedding>,
    pub prepared: Prepared,
    pub noise_seed: u64,
    pub silence_skip: bool,
    pub warnings: Vec<String>,
}

/// A loaded Sigil-ZS voice (see the module docs).
pub struct SigilZsModel {
    id: u64,
    graphs: GraphSet,
    ep: String,
    session: VcSessionOptions,
    gpu: GpuLevers,
    settings: Settings,
    split: bool,
    anchored: bool,
    phase_lever: bool,
    reference: Arc<ReferenceEmbedding>,
    prepared: Prepared,
    ref_gen: u64,
    /// The flow's time grid at the voice's steps, and at [`reduced_steps`] (the governor's reduced
    /// levels; both computed at load, so a level change allocates nothing).
    t_span: Vec<f32>,
    t_span_reduced: Vec<f32>,
    /// The governor asked for the reduced step count.
    reduced: bool,
    geometry: Option<Geometry>,
    plans: Option<Arc<Mutex<TickPlans>>>,
    /// This voice built `plans` (its load measured, and claims, their VRAM).
    built: bool,
    /// What the process held on the plans' GPU before this voice built them (until the warm-up
    /// measured them).
    vram_probe: Option<VramProbe>,
    offline: Option<TickPlans>,
    noise: ZsNoise,
    z: Vec<f32>,
    sine: Vec<f32>,
    phase: Option<PhaseState>,
    silence_skip: bool,
    quiet_ticks: u32,
    controls: VoiceControls,
    requested: EffectiveLevers,
    levers: EffectiveLevers,
    outcomes: Vec<(String, EpOutcome)>,
    warnings: Vec<String>,
    reference_warned: bool,
    join_shift: JoinShift,
}

impl SigilZsModel {
    pub fn new(cfg: ModelConfig) -> SigilZsModel {
        let mut gpu = GpuLevers::from_effective(&cfg.levers);
        // TF32 only on an explicit fp16 request: on the flow and the content encoder it moves the
        // generated mel the way the fp16 graphs do (HiFT then predicts another F0), which the
        // default must not trade for speed (ADR 0006 "Precision").
        gpu.tf32 = cfg.levers.precision == crate::services::neural_audio::vc::engine::options::PrecisionMode::Fp16;
        let split = cfg.levers.levers.get(Lever::PartialDecode);
        let anchored = cfg.levers.levers.get(Lever::AnchoredNoise);
        let phase_lever = cfg.levers.levers.get(Lever::NsfPhase0) && split;
        let mut levers = cfg.levers.clone();
        if levers.accelerator == Accelerator::TensorRtRtx {
            // TensorRT engines of these graphs (the 657 MB XLS-R, the DiT step) were never built
            // or measured: the voice runs its CUDA plans (graphs, IOBinding) instead.
            gpu.accelerator = Accelerator::None;
            levers.accelerator = Accelerator::None;
            levers.notes.push("tensorrt_rtx: not measured for Sigil-ZS voices; running on CUDA".into());
        }
        levers.levers.set(Lever::NsfPhase0, phase_lever);
        // No F0 cache (HiFT predicts its F0 from the generated mel) and no retrieval in this family.
        levers.levers.set(Lever::F0Cache, false);
        levers.levers.set(Lever::GpuKnn, false);
        if cfg.levers.levers.get(Lever::NsfPhase0) && !split {
            levers.notes.push("nsf_phase0: the phase-continuous join needs partial decode; off".into());
        }
        SigilZsModel {
            id: NEXT_ID.fetch_add(1, Ordering::Relaxed),
            graphs: cfg.graphs,
            ep: cfg.ep,
            session: cfg.session,
            gpu,
            t_span: time_grid(cfg.settings.preset.steps),
            t_span_reduced: time_grid(reduced_steps(cfg.settings.preset.steps)),
            reduced: false,
            settings: cfg.settings,
            split,
            anchored,
            phase_lever,
            reference: cfg.reference,
            prepared: cfg.prepared,
            ref_gen: 0,
            geometry: None,
            plans: None,
            built: false,
            vram_probe: None,
            offline: None,
            noise: ZsNoise::new(cfg.noise_seed, 64),
            z: Vec::new(),
            sine: Vec::new(),
            phase: None,
            silence_skip: cfg.silence_skip,
            quiet_ticks: 0,
            controls: VoiceControls::default(),
            requested: cfg.levers,
            levers,
            outcomes: Vec::new(),
            warnings: cfg.warnings,
            reference_warned: false,
            join_shift: JoinShift::Grid,
        }
    }

    /// Choose the phase anchor's pairing (the GPU test's join matrix).
    #[cfg(test)]
    pub fn set_join_shift(&mut self, shift: JoinShift) {
        self.join_shift = shift;
    }

    /// The stream shape the plans run (after `configure_stream`; tests).
    #[cfg(test)]
    pub fn geometry(&self) -> Option<Geometry> {
        self.geometry
    }

    /// The levers the voice runs.
    pub fn effective_levers(&self) -> &EffectiveLevers {
        &self.levers
    }

    /// The previous tick's phase state `(tick, decoded F0, harm_phase)` (tests).
    #[cfg(test)]
    pub(super) fn last_phase(&self) -> Option<(u64, Vec<f32>, [f32; N_HARM])> {
        self.phase.as_ref().map(|p| (p.tick, p.f0.clone(), p.harm_phase))
    }

    /// [`Self::join_anchor`] of the configured stream (tests).
    #[cfg(test)]
    pub(super) fn join_anchor_for_tests(&self) -> (usize, usize) {
        self.join_anchor(&self.geometry.expect("configured"))
    }

    fn owner(&self) -> u64 {
        (self.id << 20) | (self.ref_gen & 0xF_FFFF)
    }

    fn plan_config<'a>(
        &'a self,
        geometry: Geometry,
        ep: &'a str,
        opts: &'a VcSessionOptions,
        levers: GpuLevers,
        split: bool,
        component: &'a str,
    ) -> PlanConfig<'a> {
        PlanConfig {
            graphs: &self.graphs,
            ep,
            opts,
            levers,
            geometry,
            cfg: self.settings.preset.cfg_rate > 0.0,
            split,
            fp16_xlsr: self.settings.fp16_xlsr,
            fp16_dit: self.settings.fp16_dit,
            fp16_vocoder: self.settings.fp16_vocoder,
            component,
        }
    }

    /// Use the reference a tick carries when it is another clip with the same prompt length (a hot
    /// reference swap); otherwise keep the loaded one (a reference change reloads the voice).
    fn sync_reference(&mut self, r: Option<&ReferenceEmbedding>) {
        let Some(r) = r else { return };
        if std::ptr::eq(r, Arc::as_ptr(&self.reference)) || r.same_source(&self.reference) {
            return;
        }
        match super::reference::prepare(r) {
            Ok(p) if p.frames == self.prepared.frames => {
                self.prepared = p;
                self.reference = Arc::new(r.clone());
                self.ref_gen += 1;
                self.reference_warned = false;
            }
            Ok(_) | Err(_) if !self.reference_warned => {
                tracing::warn!(target: "sigil::vc", "Sigil-ZS: the new reference does not fit the loaded voice; it applies when the voice reloads");
                self.reference_warned = true;
            }
            _ => {}
        }
    }

    fn quiet(&mut self, ctx: &ConversionCtx<'_>, g: &Geometry) -> bool {
        if !self.silence_skip {
            self.quiet_ticks = 0;
            return false;
        }
        let end = ctx.mono_16k.len().saturating_sub(g.right_16k);
        let span = &ctx.mono_16k[end.saturating_sub(g.want_16k)..end];
        let rms = (span.iter().map(|&x| f64::from(x) * f64::from(x)).sum::<f64>() / span.len().max(1) as f64).sqrt();
        let threshold = 10f64.powf(f64::from(self.controls.silence_db) / 20.0);
        if rms < threshold {
            self.quiet_ticks = self.quiet_ticks.saturating_add(1);
            let block_ms = g.block_16k.max(1) as f64 / 16.0;
            self.quiet_ticks > (SILENCE_HANGOVER_MS / block_ms).ceil() as u32
        } else {
            self.quiet_ticks = 0;
            false
        }
    }

    fn stream_tick(&mut self, ctx: &ConversionCtx<'_>, allow_skip: bool) -> Result<Vec<f32>, String> {
        let g = self.geometry.ok_or("Sigil-ZS: the stream is not configured")?;
        let mut out = vec![0.0f32; g.out_samples];
        if allow_skip && self.quiet(ctx, &g) {
            // Silence: no inference; the next spoken tick starts a fresh join.
            self.phase = None;
            return Ok(out);
        }
        let tick = ctx.stream_pos_16k / g.block_16k.max(1) as u64;
        let origin = self.noise.window_origin(self.anchored, tick, g.grid_frames);
        self.noise.fill_z(origin, g.n_mel, &mut self.z);
        self.noise.fill_sine(origin, g.dec_start, g.dec_frames(), &mut self.sine);
        let phase = match (&self.phase, self.phase_lever) {
            (Some(prev), true) if prev.tick + 1 == tick => {
                let (n_this, n_prev) = self.join_anchor(&g);
                if n_prev < prev.f0.len() * HOP {
                    PhaseMode::Anchor { n: n_this, target: nsf_cycles_at(&prev.f0, &prev.harm_phase, n_prev) }
                } else {
                    PhaseMode::Start(self.noise.base_phase(self.anchored))
                }
            }
            _ => PhaseMode::Start(self.noise.base_phase(self.anchored)),
        };
        let plans = self.plans.clone().ok_or("Sigil-ZS: the stream is not configured")?;
        let mut p = plans.lock().map_err(|_| "Sigil-ZS: the plans are poisoned".to_string())?;
        let inputs = TickInputs {
            owner: self.owner(),
            reference: &self.prepared,
            t_span: if self.reduced { &self.t_span_reduced } else { &self.t_span },
            cfg_rate: self.settings.preset.cfg_rate,
            z: &self.z[..MEL_BINS * g.n_mel],
            sine: &self.sine[..N_HARM * g.dec_samples()],
            phase,
        };
        let used = match p.tick(ctx.mono_16k, &inputs, &mut out) {
            Ok(used) => used,
            Err(e) => {
                self.phase = None;
                return Err(e);
            }
        };
        if self.phase_lever {
            let f0 = p.decoded_f0();
            let state = self.phase.get_or_insert_with(|| PhaseState { tick, f0: Vec::with_capacity(f0.len()), harm_phase: [0.0; N_HARM] });
            state.tick = tick;
            state.f0.clear();
            state.f0.extend_from_slice(f0);
            state.harm_phase = used.harm_phase;
        }
        Ok(out)
    }

    /// `(n_this, n_prev)`: the middle of the SOLA crossfade at the start of this tick's returned
    /// audio, as a sample of this tick's decoded slice, and the sample of the previous tick's slice
    /// holding the same audio (`F · 256` later with anchored noise, whose draws line up there; one
    /// block of output later otherwise). The expected SOLA offset (the content moves `F · 256`
    /// samples per block, the output one block) is applied when it falls inside the search window
    /// (ADR 0006 `Preset.join_anchor`).
    fn join_anchor(&self, g: &Geometry) -> (usize, usize) {
        let block_out = out_len(g.block_16k);
        let (overlap, centre) = sola_geometry(g.out_samples, block_out);
        let shift = match self.join_shift {
            JoinShift::Grid if self.anchored => g.grid_frames * HOP,
            JoinShift::Content => (g.frames_per_block() * HOP as f64).round() as usize,
            _ => block_out,
        };
        let window = g.out_samples.saturating_sub(block_out + overlap);
        let expected = (centre as i64 + block_out as i64 - shift as i64).clamp(0, window as i64) as usize;
        let n_this = g.returned().0 + expected + overlap / 2;
        (n_this, n_this + shift)
    }

    /// A one-off conversion (see the module docs).
    fn offline_convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        let len = ctx.mono_16k.len();
        let want = ctx.new_frames.min(len);
        if len < XLSR_FIELD + UNIT_16K {
            return Ok(vec![0.0; out_len(want)]);
        }
        let g = Geometry::offline(len, want, self.prepared.frames)?;
        if self.offline.as_ref().map(|p| p.geometry) != Some(g) {
            self.offline = None;
            let opts = VcSessionOptions {
                intra_threads: Some(2),
                streamer_profile: false,
                vram_cap_mb: None,
                ..self.session.clone()
            };
            // Its own diagnostics rows: the stream's `vc.zs.*` rows keep showing the voice's EP.
            let cfg = self.plan_config(g, "cpu", &opts, GpuLevers::default(), false, "vc.zs.offline");
            self.offline = Some(TickPlans::build(&cfg)?);
        }
        self.noise.reserve(g.n_mel);
        let mut z = vec![0.0f32; MEL_BINS * g.n_mel];
        let mut sine = vec![0.0f32; N_HARM * g.dec_samples()];
        let origin = (ctx.stream_pos_16k / UNIT_16K as u64).wrapping_mul(SR_OUT as u64) / (50 * HOP as u64);
        self.noise.fill_z(origin, g.n_mel, &mut z);
        self.noise.fill_sine(origin, 0, g.n_mel, &mut sine);
        let mut out = vec![0.0f32; g.out_samples];
        let inputs = TickInputs {
            owner: self.owner(),
            reference: &self.prepared,
            t_span: &self.t_span,
            cfg_rate: self.settings.preset.cfg_rate,
            z: &z,
            sine: &sine,
            phase: PhaseMode::Start(self.noise.base_phase(true)),
        };
        let plans = self.offline.as_mut().ok_or("Sigil-ZS: no offline plans")?;
        plans.tick(ctx.mono_16k, &inputs, &mut out)?;
        Ok(out)
    }

    fn convert_inner(&mut self, ctx: &ConversionCtx<'_>, allow_skip: bool) -> Result<Vec<f32>, String> {
        self.sync_reference(ctx.reference);
        match self.geometry {
            Some(g) if ctx.mono_16k.len() == g.n16 && ctx.new_frames == g.want_16k => self.stream_tick(ctx, allow_skip),
            _ => self.offline_convert(ctx),
        }
    }

    /// Let go of the stream's plans. When this voice built them (its load claimed their VRAM) and
    /// other voices still use them, the plans claim their footprint themselves until the last of
    /// those voices goes (`plans` module docs: VRAM accounting).
    fn release_plans(&mut self) {
        self.vram_probe = None;
        let Some(plans) = self.plans.take() else {
            self.built = false;
            return;
        };
        if std::mem::take(&mut self.built) {
            // Claimed even when this looks like the last holder: another load may take the plans
            // from the pool in between, and plans that drop release the claim with them.
            let mut p = plans.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            p.claim_footprint();
        }
    }

    /// Whether the stream's plans are already warm: another voice ticked on them (or is ticking on
    /// them right now, so they are warm or being warmed by the voice that built them).
    fn plans_warm(&self) -> bool {
        let Some(plans) = &self.plans else { return false };
        match plans.try_lock() {
            Ok(p) => p.warmed(),
            Err(TryLockError::WouldBlock) => true,
            Err(TryLockError::Poisoned(_)) => false,
        }
    }

    /// Record what building (and warming) the plans added on the GPU, when this voice built them.
    fn measure_footprint(&self) {
        let (Some(probe), Some(plans), true) = (self.vram_probe, &self.plans, self.built) else {
            return;
        };
        if let Some(mb) = probe.delta_mb() {
            if let Ok(mut p) = plans.lock() {
                p.note_footprint(mb.round() as u64);
            }
        }
    }

    /// Whether this voice built its stream's plans (tests).
    #[cfg(test)]
    pub(super) fn built_plans(&self) -> bool {
        self.built
    }

    /// The stream plans' footprint, as if this voice had measured it (tests: the CPU has no VRAM).
    #[cfg(test)]
    pub(super) fn note_plans_footprint(&self, mb: u64) {
        self.plans.as_ref().expect("configured").lock().unwrap().note_footprint(mb);
    }

    /// The stream plans' own VRAM claim `(id, MiB)` (tests).
    #[cfg(test)]
    pub(super) fn plans_claim(&self) -> Option<(u64, u64)> {
        self.plans.as_ref().expect("configured").lock().unwrap().own_claim()
    }

    /// Ticks that ran on the stream's plans, by any voice (tests).
    #[cfg(test)]
    pub(super) fn plan_ticks(&self) -> u64 {
        self.plans.as_ref().expect("configured").lock().unwrap().ticks()
    }

    /// The stream's plans (tests: hold their lock as a ticking voice does).
    #[cfg(test)]
    pub(super) fn plans_handle(&self) -> Arc<Mutex<TickPlans>> {
        self.plans.clone().expect("configured")
    }

    /// Fold a plan set's report into the voice's levers, outcomes and notes.
    fn adopt_report(&mut self, plans: &TickPlans) {
        let r = &plans.report;
        let mut levers = self.levers.clone();
        levers.levers.set(Lever::IoBinding, self.requested.levers.get(Lever::IoBinding) && r.io_binding);
        levers.levers.set(Lever::CudaGraphs, self.requested.levers.get(Lever::CudaGraphs) && r.cuda_graphs);
        levers.levers.set(Lever::GpuPriority, self.requested.levers.get(Lever::GpuPriority) && r.priority);
        levers.levers.set(Lever::PartialDecode, plans.split);
        for n in &r.notes {
            if !levers.notes.contains(n) {
                levers.notes.push(n.clone());
            }
        }
        self.levers = levers;
        for (component, outcome) in &r.outcomes {
            match self.outcomes.iter_mut().find(|(c, _)| c == component) {
                Some(slot) => slot.1 = outcome.clone(),
                None => self.outcomes.push((component.clone(), outcome.clone())),
            }
        }
        for n in r.notes.iter().filter(|n| n.contains("not installed")) {
            if !self.warnings.contains(n) {
                self.warnings.push(n.clone());
            }
        }
    }
}

impl VoiceConversionModel for SigilZsModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        self.convert_inner(ctx, true)
    }

    fn input_sample_rate(&self) -> u32 {
        16_000
    }

    fn output_sample_rate(&self) -> u32 {
        SR_OUT
    }

    fn min_context_samples(&self) -> usize {
        XLSR_FIELD
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        let live = self.geometry.filter(|g| ctx.mono_16k.len() == g.n16 && ctx.new_frames == g.want_16k);
        if live.is_some() && self.plans_warm() {
            // Shared plans another voice already ticked on (kernels chosen, CUDA graphs captured):
            // two warm-up ticks would only hold their lock while that voice plays. (When this voice
            // built them, the other voice's ticks warmed them: measure now.)
            self.reset_stream();
            self.measure_footprint();
            self.vram_probe = None;
            return Ok(());
        }
        // The live shape twice (a fresh start, then an anchored join), silence skip bypassed; the
        // stream then starts clean.
        let first = self.convert_inner(ctx, false).map(|_| ());
        if let Some(g) = self.geometry.filter(|g| ctx.mono_16k.len() == g.n16) {
            let next = ConversionCtx { stream_pos_16k: ctx.stream_pos_16k + g.block_16k as u64, ..*ctx };
            let second = self.convert_inner(&next, false).map(|_| ());
            self.reset_stream();
            // The warm-up allocated the sessions' working memory and captured their CUDA graphs.
            self.measure_footprint();
            self.vram_probe = None;
            return first.and(second);
        }
        self.reset_stream();
        first
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        self.controls = *controls;
    }

    /// The governor's reduced-context levels run [`reduced_steps`] Euler steps (the only knob of
    /// this family: retrieval, F0 and context do not apply; the tick shapes stay fixed).
    fn apply_quality(&mut self, level: &QualityLevel) -> bool {
        let reduced = level.context_scale < 1.0 && self.t_span_reduced.len() < self.t_span.len();
        let changed = reduced != self.reduced;
        self.reduced = reduced;
        changed
    }

    fn reset_stream(&mut self) {
        self.phase = None;
        self.quiet_ticks = 0;
    }

    fn fixed_block(&self) -> Option<usize> {
        Some(self.settings.block_16k())
    }

    fn configure_stream(&mut self, block: usize, want: usize, _extra: usize) -> Result<usize, String> {
        let g = Geometry::stream(&self.settings.preset, block, want, self.prepared.frames, self.split)?;
        if self.geometry == Some(g) && self.plans.is_some() {
            return Ok(g.n16);
        }
        // Release this voice's hold on plans of another shape before building (a shape change frees
        // its sessions first when no other voice uses them).
        self.release_plans();
        self.geometry = None;
        let ep = self.ep.clone();
        let opts = self.session.clone();
        let probe = VramProbe::for_rung(normalize_ep(&ep), opts.cuda_device_index);
        let (plans, reused) = shared(&self.plan_config(g, &ep, &opts, self.gpu, self.split, "vc.zs"))?;
        {
            let mut p = plans.lock().map_err(|_| "Sigil-ZS: the plans are poisoned".to_string())?;
            if let Some(refusal) = p.report.cpu_refusal() {
                return Err(refusal);
            }
            if !reused {
                if let Some(mb) = probe.and_then(|v| v.delta_mb()) {
                    p.note_footprint(mb.round() as u64);
                }
            }
            self.adopt_report(&p);
        }
        if reused {
            tracing::info!(target: "sigil::vc", "Sigil-ZS: sharing the loaded sessions of this shape");
        }
        self.built = !reused;
        self.vram_probe = if reused { None } else { probe };
        self.geometry = Some(g);
        self.plans = Some(plans);
        self.noise.reserve(g.n_mel);
        self.z.resize(MEL_BINS * g.n_mel, 0.0);
        self.sine.resize(N_HARM * g.dec_samples(), 0.0);
        self.phase = None;
        Ok(g.n16)
    }

    fn stream_report(&self) -> Option<StreamReport> {
        Some(StreamReport {
            outcomes: self.outcomes.clone(),
            levers: self.levers.clone(),
            warnings: self.warnings.clone(),
        })
    }
}

impl Drop for SigilZsModel {
    /// Hands the plans' VRAM claim to them when this voice built them and others still use them.
    fn drop(&mut self) {
        self.release_plans();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sola_geometry_recovers_the_harness_window() {
        // 20 ms crossfade at 22.05 kHz: overlap 441, search 220, +2 slack -> 663 samples of tail.
        let block_out = out_len(1_600);
        let (overlap, centre) = sola_geometry(block_out + 663, block_out);
        assert_eq!(overlap, 440);
        assert_eq!(centre, (663 - 440) / 2);
        assert_eq!(sola_geometry(block_out, block_out).0, 1);
    }

    #[test]
    fn the_governor_halves_the_steps_down_to_three() {
        let table: Vec<(u32, u32)> = [1u32, 2, 3, 4, 6, 10, 30].iter().map(|&s| (s, reduced_steps(s))).collect();
        assert_eq!(table, vec![(1, 1), (2, 2), (3, 3), (4, 3), (6, 3), (10, 5), (30, 15)]);
    }
}
