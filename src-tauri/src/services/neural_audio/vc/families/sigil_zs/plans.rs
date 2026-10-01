//! The fixed-shape sessions of one Sigil-ZS stream shape and the tick that drives them (ADR 0006
//! "Loop"; `sampler.py` `SigilZS.tick`).
//!
//! | stage | graph | shapes (fixed per [`Geometry`]) | GPU levers |
//! |---|---|---|---|
//! | content | `xlsr` | wav16k `[1, n16]` → feats `[1, T, 1024]` | IOBinding, CUDA graph |
//! | length regulator | `lr` | feats `[1, lr_in, 1024]`, n_frames → cond `[1, n_mel, 384]` | IOBinding (its `n_frames` is read on the CPU: no graph) |
//! | flow | `dit_step` / `dit_step1` | x, prompt_x `[1, 80, L]`, cond `[1, L, 384]`, style, t, dt (cfg_rate), prompt_len → x_next | IOBinding, CUDA graph |
//! | vocoder, partial decode | `hift_f0` + `hift_nsf` | mel `[1, 80, n_mel]` → f0; mel / f0 / sine noise over `dec_start..n_mel` → audio | IOBinding, CUDA graph |
//! | vocoder, whole tick | `hift` | mel, harm_phase, sine noise over the tick → audio, f0 | IOBinding, CUDA graph |
//!
//! Every session is a `gpu::SessionPlan` (EP ladder, outcome records, IOBinding / CUDA graphs /
//! priority per lever; zero host allocations per tick on the bound and device paths). DirectML
//! refuses the pinned dimensions of the HiFT graphs (ADR 0006 finding 5), so they run with dynamic
//! shapes there; any plan built with pinned dimensions that fell to the CPU from a GPU request
//! (DirectML's refusal, also after a CUDA request fell to DirectML) is rebuilt unpinned once
//! ([`build_plan`]).
//!
//! **Sharing.** The plans of one shape are a process-wide pool ([`shared`]): voices with the same
//! shape (another reference voice with a prompt as long, a hot voice, the incoming voice of a
//! reference switch) use the same sessions instead of loading another 0.9 GB. One mutex covers the
//! whole set for a tick, and the voice-specific inputs (the reference's prompt, style and length,
//! the CFG rate) are rewritten whenever another voice (or another reference) ran the plans last.
//!
//! **VRAM accounting.** The voice changer claims each voice's VRAM from what its load added to the
//! process (`LoadedVoice::load_cancellable`): the voice that built a set of plans claims their whole footprint,
//! a voice that reused them about 0 MB. So the footprint the building voice measured is kept with
//! the plans ([`TickPlans::note_footprint`]), and when that voice goes while others still use them
//! the plans claim it themselves ([`TickPlans::claim_footprint`]) until the last of them goes: the
//! resource governor never counts a live set of sessions as free.

use std::sync::{Arc, Mutex, Weak};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::gpu::{GpuLevers, PlanSpec, SessionPlan, TensorSpec};
use crate::services::neural_audio::ort_session::{normalize_ep, ModelSource, VcSessionOptions, EP_CPU, EP_DML};
use crate::services::resources::{self, Subsystem, VramClaim};

use super::dsp::{carry_phase, harm_phase_for_anchor};
use super::graphs::{GraphName, GraphSet};
use super::preset::{Geometry, COND_DIM, CONTENT_DIM, HOP, MEL_BINS, N_HARM, STYLE_DIM};
use super::sampler::{solve_euler, Stage};

/// What a voice converts toward, ready for the DiT (from `ReferenceEmbedding`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prepared {
    /// CAM++ style `[192]`.
    pub style: Vec<f32>,
    /// Prompt mel `[80][P]` (channel-major).
    pub prompt_mel: Vec<f32>,
    /// Prompt condition `[P][384]`.
    pub prompt_cond: Vec<f32>,
    /// `P`.
    pub frames: usize,
}

/// How the tick sets the NSF phase.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum PhaseMode {
    /// Each harmonic's phase (radians) at the window's first sample (upstream; with partial decode
    /// it is carried to the decoded slice's first frame, so the slice continues the window).
    Start([f32; N_HARM]),
    /// The phase-continuous join (partial decode only): sample `n` of the decoded slice gets phase
    /// `target` (cycles).
    Anchor { n: usize, target: [f64; N_HARM] },
}

/// One tick's inputs besides the window.
pub struct TickInputs<'a> {
    /// The voice (and reference generation) writing the inputs: the voice-specific inputs are
    /// rewritten when it is not the one that ran the plans last.
    pub owner: u64,
    pub reference: &'a Prepared,
    /// The flow's time grid (`linspace_f32(steps)`).
    pub t_span: &'a [f32],
    pub cfg_rate: f32,
    /// The starting noise of the tick's mel frames, `[80][n_mel]`.
    pub z: &'a [f32],
    /// The NSF source noise of the decoded frames, `[9][dec_frames × 256]`.
    pub sine: &'a [f32],
    pub phase: PhaseMode,
}

/// What a tick used, for the next tick's phase anchor.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TickPhase {
    /// The decoded slice's `harm_phase` (radians at its first sample).
    pub harm_phase: [f32; N_HARM],
}

/// The sessions' outcomes and levers.
#[derive(Clone, Debug, Default)]
pub struct PlanReport {
    /// `(component, outcome)` per session.
    pub outcomes: Vec<(String, EpOutcome)>,
    /// IOBinding on every session that asked for it.
    pub io_binding: bool,
    /// CUDA graphs captured on every session that asked for one (all but `lr`).
    pub cuda_graphs: bool,
    /// Priority scheduling on every session.
    pub priority: bool,
    /// TF32 on the CUDA sessions.
    pub tf32: bool,
    /// Why a requested lever is off, per session (`<component>: <lever>: <reason>`).
    pub notes: Vec<String>,
    /// Graphs whose session landed on the CPU although a GPU was requested (the EP ladder's
    /// fallback), with the outcome's reason.
    pub on_cpu: Vec<(GraphName, Option<String>)>,
}

/// Stages of a tick the CPU cannot run in real time (ADR 0006 cost table, `sigil_low` on two CPU
/// threads per 100 ms block: `xlsr` 122 ms, each of the 6 DiT steps 99 ms, `hift` 127 ms,
/// `hift_nsf` 93 ms). `lr` (1.3 ms) and `hift_f0` (1.5 ms) can.
const CPU_TOO_SLOW: [GraphName; 5] = [GraphName::Xlsr, GraphName::DitStep, GraphName::DitStep1, GraphName::Hift, GraphName::HiftNsf];

impl PlanReport {
    /// Why a voice that asked for a GPU cannot run on these sessions: a stage the CPU cannot run in
    /// real time fell to the CPU (VRAM full, a GPU runtime missing). `None` when every such stage
    /// runs on a GPU. The family refuses an explicit CPU request for the same reason.
    pub fn cpu_refusal(&self) -> Option<String> {
        let slow: Vec<String> = self
            .on_cpu
            .iter()
            .filter(|(g, _)| CPU_TOO_SLOW.contains(g))
            .map(|(g, why)| match why.as_deref().map(str::trim).filter(|w| !w.is_empty()) {
                Some(why) => format!("{} ({why})", g.stem()),
                None => g.stem().to_string(),
            })
            .collect();
        if slow.is_empty() {
            return None;
        }
        Some(format!(
            "Sigil-ZS: {} fell back to the CPU, which cannot convert this voice in real time; free GPU memory (close other GPU apps) or check the GPU runtime in Settings > GPU & diagnostics",
            slow.join(", ")
        ))
    }
}

/// The sessions of one shape (see the module docs).
pub struct TickPlans {
    pub geometry: Geometry,
    /// `dit_step` (CFG) or `dit_step1`.
    pub cfg: bool,
    /// Partial decode (`hift_f0` + `hift_nsf`) or the whole-tick `hift`.
    pub split: bool,
    xlsr: Box<dyn Stage>,
    lr: Box<dyn Stage>,
    dit: Box<dyn Stage>,
    hift_f0: Option<Box<dyn Stage>>,
    hift_nsf: Option<Box<dyn Stage>>,
    hift: Option<Box<dyn Stage>>,
    pub report: PlanReport,
    /// Who wrote the voice-specific inputs last (`u64::MAX`: nobody yet).
    owner: u64,
    /// Ticks that ran on these sessions (one warms them: kernels chosen, CUDA graphs captured).
    ticks: u64,
    /// VRAM these sessions added to the process (MiB), as the voice that built them measured it.
    footprint_mb: u64,
    /// The plans' own claim of [`Self::footprint_mb`], once the voice whose load claimed them is
    /// gone (see the module docs); released with the plans.
    claim: Option<VramClaim>,
    x: Vec<f32>,
    mel: Vec<f32>,
    f0: Vec<f32>,
}

/// What a set of plans is built from.
pub struct PlanConfig<'a> {
    pub graphs: &'a GraphSet,
    /// EP ladder rung (`cuda` / `dml` / `cpu`).
    pub ep: &'a str,
    pub opts: &'a VcSessionOptions,
    pub levers: GpuLevers,
    pub geometry: Geometry,
    pub cfg: bool,
    pub split: bool,
    pub fp16_xlsr: bool,
    pub fp16_dit: bool,
    pub fp16_vocoder: bool,
    /// Diagnostics component prefix (`vc.zs`).
    pub component: &'a str,
}

/// The tensor specs of one graph at a geometry: `(inputs, outputs, pinned dims)`.
pub fn graph_io(graph: GraphName, g: &Geometry) -> (Vec<TensorSpec>, Vec<TensorSpec>, Vec<(&'static str, i64)>) {
    let w = g.dec_frames();
    match graph {
        GraphName::Xlsr => (
            vec![TensorSpec::f32("wav16k", &[1, g.n16])],
            vec![TensorSpec::f32("feats", &[1, g.xlsr_frames, CONTENT_DIM])],
            vec![("N", g.n16 as i64)],
        ),
        GraphName::Lr => (
            vec![TensorSpec::f32("feats", &[1, g.lr_in, CONTENT_DIM]), TensorSpec::i64("n_frames", &[1])],
            vec![TensorSpec::f32("cond", &[1, g.n_mel, COND_DIM])],
            vec![("T", g.lr_in as i64)],
        ),
        GraphName::DitStep | GraphName::DitStep1 => {
            let mut inputs = vec![
                TensorSpec::f32("x", &[1, MEL_BINS, g.big_l]),
                TensorSpec::f32("prompt_x", &[1, MEL_BINS, g.big_l]),
                TensorSpec::f32("cond", &[1, g.big_l, COND_DIM]),
                TensorSpec::f32("style", &[1, STYLE_DIM]),
                TensorSpec::f32("t", &[1]),
                TensorSpec::f32("dt", &[1]),
            ];
            if graph == GraphName::DitStep {
                inputs.push(TensorSpec::f32("cfg_rate", &[1]));
            }
            inputs.push(TensorSpec::i64("prompt_len", &[1]));
            (inputs, vec![TensorSpec::f32("x_next", &[1, MEL_BINS, g.big_l])], vec![("L", g.big_l as i64)])
        }
        GraphName::HiftF0 => (
            vec![TensorSpec::f32("mel", &[1, MEL_BINS, g.n_mel])],
            vec![TensorSpec::f32("f0", &[1, g.n_mel])],
            vec![("M", g.n_mel as i64)],
        ),
        GraphName::HiftNsf => (
            vec![
                TensorSpec::f32("mel", &[1, MEL_BINS, w]),
                TensorSpec::f32("f0", &[1, w]),
                TensorSpec::f32("harm_phase", &[1, N_HARM, 1]),
                TensorSpec::f32("sine_noise", &[1, N_HARM, w * HOP]),
            ],
            vec![TensorSpec::f32("audio", &[1, w * HOP])],
            vec![("W", w as i64), ("S", (w * HOP) as i64)],
        ),
        GraphName::Hift => (
            vec![
                TensorSpec::f32("mel", &[1, MEL_BINS, g.n_mel]),
                TensorSpec::f32("harm_phase", &[1, N_HARM, 1]),
                TensorSpec::f32("sine_noise", &[1, N_HARM, g.n_mel * HOP]),
            ],
            vec![TensorSpec::f32("audio", &[1, g.n_mel * HOP]), TensorSpec::f32("f0", &[1, g.n_mel])],
            vec![("M", g.n_mel as i64), ("S", (g.n_mel * HOP) as i64)],
        ),
        GraphName::Campplus | GraphName::Mel => (Vec::new(), Vec::new(), Vec::new()),
    }
}

/// A graph's I/O at one shape: the tensor specs and the symbolic dimensions to pin.
#[derive(Clone, Copy)]
pub struct GraphIo<'a> {
    pub inputs: &'a [TensorSpec],
    pub outputs: &'a [TensorSpec],
    pub dims: &'a [(&'a str, i64)],
}

/// Build one plan (ADR 0006 finding 5). DirectML's graph fusion refuses the pinned dimensions of
/// the HiFT graphs (`pin_on_dml = false`), so a DirectML request builds them with dynamic shapes.
/// A plan built with pinned dimensions that fell to the CPU from a GPU request is rebuilt once
/// without them. The second case covers a CUDA request whose ladder (CUDA → DirectML → CPU) fell to
/// DirectML, for example without the NVIDIA runtime pack: DirectML then refuses the pinned HiFT
/// shapes and the vocoder would land on the CPU (93-127 ms per call instead of 6-7 ms). DDSP-SVC
/// handles the same case the same way (`ddsp_svc::model::build_plan`).
pub fn build_plan(
    component: &str,
    ep: &str,
    opts: &VcSessionOptions,
    path: &std::path::Path,
    io: GraphIo<'_>,
    levers: GpuLevers,
    pin_on_dml: bool,
) -> Result<SessionPlan, String> {
    let build = |dims: &[(&str, i64)]| {
        SessionPlan::build(&PlanSpec {
            component,
            ep,
            opts,
            model: ModelSource::File(path),
            weights: None,
            dims,
            inputs: io.inputs,
            outputs: io.outputs,
            levers,
        })
    };
    build_with_fallback(component, ep, io.dims, pin_on_dml, build, |plan| plan.outcome().active == EP_CPU)
}

/// [`build_plan`]'s decisions over any builder (`on_cpu`: the built plan landed on the CPU), so
/// they can be tested without a GPU.
pub(super) fn build_with_fallback<P>(
    component: &str,
    ep: &str,
    dims: &[(&str, i64)],
    pin_on_dml: bool,
    mut build: impl FnMut(&[(&str, i64)]) -> Result<P, String>,
    on_cpu: impl Fn(&P) -> bool,
) -> Result<P, String> {
    let requested = normalize_ep(ep);
    let pinned: &[(&str, i64)] = if requested == EP_DML && !pin_on_dml { &[] } else { dims };
    let plan = build(pinned)?;
    if requested == EP_CPU || pinned.is_empty() || !on_cpu(&plan) {
        return Ok(plan);
    }
    tracing::info!(target: "sigil::vc", "{component}: no GPU took the pinned shapes; rebuilding with dynamic shapes");
    drop(plan);
    build(&[])
}

impl TickPlans {
    /// Build every session of `cfg` (the largest, XLS-R, first).
    pub fn build(cfg: &PlanConfig<'_>) -> Result<TickPlans, String> {
        let g = cfg.geometry;
        let mut notes: Vec<String> = Vec::new();
        let mut report = PlanReport { io_binding: true, cuda_graphs: true, priority: true, tf32: true, ..Default::default() };
        let no_graph = GpuLevers { cuda_graphs: false, ..cfg.levers };
        let gpu_requested = normalize_ep(cfg.ep) != EP_CPU;
        let mut make = |graph: GraphName, fp16: bool, levers: GpuLevers, pin_on_dml: bool, wants_graph: bool| -> Result<Box<dyn Stage>, String> {
            let (path, why) = cfg.graphs.path(graph, fp16);
            notes.extend(why);
            let (inputs, outputs, dims) = graph_io(graph, &g);
            let component = format!("{}.{}", cfg.component, graph.stem());
            let io = GraphIo { inputs: &inputs, outputs: &outputs, dims: &dims };
            let plan = build_plan(&component, cfg.ep, cfg.opts, &path, io, levers, pin_on_dml)?;
            let got = plan.levers();
            report.outcomes.push((component.clone(), plan.outcome().clone()));
            if gpu_requested && plan.outcome().active == EP_CPU {
                report.on_cpu.push((graph, plan.outcome().reason.clone()));
            }
            if levers.io_binding {
                report.io_binding &= got.io_binding;
            }
            if wants_graph && levers.cuda_graphs {
                report.cuda_graphs &= got.cuda_graphs;
            }
            if levers.priority {
                report.priority &= got.priority;
            }
            if plan.outcome().active == "cuda" {
                report.tf32 &= got.tf32;
            }
            report.notes.extend(plan.notes().iter().map(|n| format!("{}: {n}", graph.stem())));
            Ok(Box::new(plan))
        };
        let xlsr = make(GraphName::Xlsr, cfg.fp16_xlsr, cfg.levers, true, true)?;
        let lr = make(GraphName::Lr, false, no_graph, true, false)?;
        let dit_graph = if cfg.cfg { GraphName::DitStep } else { GraphName::DitStep1 };
        let dit = make(dit_graph, cfg.fp16_dit, cfg.levers, true, true)?;
        let (hift_f0, hift_nsf, hift) = if cfg.split {
            (
                Some(make(GraphName::HiftF0, false, cfg.levers, false, true)?),
                Some(make(GraphName::HiftNsf, cfg.fp16_vocoder, cfg.levers, false, true)?),
                None,
            )
        } else {
            (None, None, Some(make(GraphName::Hift, cfg.fp16_vocoder, cfg.levers, false, true)?))
        };
        report.io_binding &= cfg.levers.io_binding;
        report.cuda_graphs &= cfg.levers.cuda_graphs;
        report.priority &= cfg.levers.priority;
        report.tf32 &= cfg.levers.tf32;
        report.notes.extend(notes);
        Ok(TickPlans::from_stages(g, cfg.cfg, cfg.split, xlsr, lr, dit, hift_f0, hift_nsf, hift, report))
    }

    /// Plans over the given stages (the tests' analytic graphs use it too).
    #[allow(clippy::too_many_arguments)]
    pub fn from_stages(
        geometry: Geometry,
        cfg: bool,
        split: bool,
        xlsr: Box<dyn Stage>,
        lr: Box<dyn Stage>,
        dit: Box<dyn Stage>,
        hift_f0: Option<Box<dyn Stage>>,
        hift_nsf: Option<Box<dyn Stage>>,
        hift: Option<Box<dyn Stage>>,
        report: PlanReport,
    ) -> TickPlans {
        let g = geometry;
        TickPlans {
            geometry,
            cfg,
            split,
            xlsr,
            lr,
            dit,
            hift_f0,
            hift_nsf,
            hift,
            report,
            owner: u64::MAX,
            ticks: 0,
            footprint_mb: 0,
            claim: None,
            x: vec![0.0; MEL_BINS * g.big_l],
            mel: vec![0.0; MEL_BINS * g.n_mel],
            f0: vec![0.0; g.n_mel],
        }
    }

    /// The decoded slice's F0 of the last tick (`dec_frames` values; the whole tick's without
    /// partial decode).
    pub fn decoded_f0(&self) -> &[f32] {
        &self.f0[self.geometry.dec_start..]
    }

    /// The last tick's generated mel `[80][n_mel]` (tests and parity).
    #[cfg(test)]
    pub fn mel(&self) -> &[f32] {
        &self.mel
    }

    /// Whether a tick ran on these sessions: their kernels are chosen and their CUDA graphs
    /// captured, so a voice that reuses them has nothing to warm.
    pub fn warmed(&self) -> bool {
        self.ticks > 0
    }

    /// Ticks that ran on these sessions (tests).
    #[cfg(test)]
    pub fn ticks(&self) -> u64 {
        self.ticks
    }

    /// Record the VRAM these sessions added (MiB; the largest measurement wins), and resize the
    /// plans' own claim when they hold one.
    pub fn note_footprint(&mut self, mb: u64) {
        self.footprint_mb = self.footprint_mb.max(mb);
        if let Some(claim) = &self.claim {
            claim.resize(self.footprint_mb);
        }
    }

    /// The voice whose load claimed these sessions' VRAM is going: the plans claim their footprint
    /// themselves, for as long as another voice still uses them (released when they drop; see the
    /// module docs). Nothing to claim when the footprint is unknown (the voice's own claim was 0 MB
    /// too: no DXGI measurement).
    pub fn claim_footprint(&mut self) {
        if self.claim.is_none() && self.footprint_mb > 0 {
            self.claim = Some(resources::claim_vram(Subsystem::Vc, "vc: Sigil-ZS sessions (shared)", self.footprint_mb));
        }
    }

    /// The plans' own claim: `(claim id, MiB)` (tests).
    #[cfg(test)]
    pub fn own_claim(&self) -> Option<(u64, u64)> {
        self.claim.as_ref().map(|c| (c.id(), self.footprint_mb))
    }

    /// Write the voice-specific inputs of the flow (the reference and the CFG rate).
    fn write_voice(&mut self, inputs: &TickInputs<'_>) -> Result<(), String> {
        let g = self.geometry;
        let r = inputs.reference;
        if r.frames != g.prompt || r.prompt_mel.len() != MEL_BINS * g.prompt || r.prompt_cond.len() != g.prompt * COND_DIM || r.style.len() != STYLE_DIM {
            return Err(format!("Sigil-ZS: the reference ({} prompt frames) does not fit plans for {}", r.frames, g.prompt));
        }
        {
            let px = self.dit.input_f32("prompt_x")?;
            px.fill(0.0);
            for c in 0..MEL_BINS {
                px[c * g.big_l..c * g.big_l + g.prompt].copy_from_slice(&r.prompt_mel[c * g.prompt..(c + 1) * g.prompt]);
            }
        }
        self.dit.input_f32("cond")?[..g.prompt * COND_DIM].copy_from_slice(&r.prompt_cond);
        self.dit.input_f32("style")?.copy_from_slice(&r.style);
        self.dit.input_i64("prompt_len")?[0] = g.prompt as i64;
        if self.cfg {
            self.dit.input_f32("cfg_rate")?[0] = inputs.cfg_rate;
        }
        self.owner = inputs.owner;
        Ok(())
    }

    /// Convert one window (`n16` samples) into `out` (`out_samples` long; see [`Geometry`]).
    pub fn tick(&mut self, window: &[f32], inputs: &TickInputs<'_>, out: &mut [f32]) -> Result<TickPhase, String> {
        let g = self.geometry;
        if window.len() != g.n16 || out.len() != g.out_samples {
            return Err(format!(
                "Sigil-ZS: a {}-sample window for {} returned samples does not fit plans for {} / {}",
                window.len(),
                out.len(),
                g.n16,
                g.out_samples
            ));
        }
        if inputs.z.len() != MEL_BINS * g.n_mel || inputs.sine.len() != N_HARM * g.dec_samples() {
            return Err("Sigil-ZS: the tick's noise does not fit its plans".into());
        }
        // 1. Content features over the window.
        self.xlsr.input_f32("wav16k")?.copy_from_slice(window);
        self.xlsr.run()?;
        // 2. The length regulator over the DiT's part of them.
        {
            let feats = self.xlsr.output("feats")?;
            let from = g.ce * CONTENT_DIM;
            let to = from + g.lr_in * CONTENT_DIM;
            if feats.len() < to {
                return Err(format!("Sigil-ZS: xlsr returned {} values, {to} expected", feats.len()));
            }
            self.lr.input_f32("feats")?.copy_from_slice(&feats[from..to]);
        }
        self.lr.input_i64("n_frames")?[0] = g.n_mel as i64;
        self.lr.run()?;
        // 3. The flow over [prompt | tick].
        if self.owner != inputs.owner {
            self.write_voice(inputs)?;
        }
        {
            let cond = self.lr.output("cond")?;
            self.dit.input_f32("cond")?[g.prompt * COND_DIM..].copy_from_slice(cond);
        }
        for c in 0..MEL_BINS {
            let row = &mut self.x[c * g.big_l..(c + 1) * g.big_l];
            row[..g.prompt].fill(0.0);
            row[g.prompt..].copy_from_slice(&inputs.z[c * g.n_mel..(c + 1) * g.n_mel]);
        }
        solve_euler(self.dit.as_mut(), inputs.t_span, &mut self.x)?;
        for c in 0..MEL_BINS {
            self.mel[c * g.n_mel..(c + 1) * g.n_mel].copy_from_slice(&self.x[c * g.big_l + g.prompt..(c + 1) * g.big_l]);
        }
        // 4. The vocoder.
        let d = g.dec_start;
        let w = g.dec_frames();
        let phase;
        let audio: &[f32] = if self.split {
            let (Some(f0p), Some(nsf)) = (self.hift_f0.as_mut(), self.hift_nsf.as_mut()) else {
                return Err("Sigil-ZS: the split vocoder is not built".into());
            };
            f0p.input_f32("mel")?.copy_from_slice(&self.mel);
            f0p.run()?;
            self.f0.copy_from_slice(f0p.output("f0")?);
            phase = match inputs.phase {
                PhaseMode::Start(p) => carry_phase(&self.f0, &p, d),
                PhaseMode::Anchor { n, target } => harm_phase_for_anchor(&self.f0[d..], n, &target),
            };
            {
                let m = nsf.input_f32("mel")?;
                for c in 0..MEL_BINS {
                    m[c * w..(c + 1) * w].copy_from_slice(&self.mel[c * g.n_mel + d..(c + 1) * g.n_mel]);
                }
            }
            nsf.input_f32("f0")?.copy_from_slice(&self.f0[d..]);
            nsf.input_f32("harm_phase")?.copy_from_slice(&phase);
            nsf.input_f32("sine_noise")?.copy_from_slice(inputs.sine);
            nsf.run()?;
            nsf.output("audio")?
        } else {
            let Some(h) = self.hift.as_mut() else {
                return Err("Sigil-ZS: the vocoder is not built".into());
            };
            phase = match inputs.phase {
                PhaseMode::Start(p) => p,
                PhaseMode::Anchor { .. } => return Err("Sigil-ZS: the NSF phase anchor needs partial decode".into()),
            };
            h.input_f32("mel")?.copy_from_slice(&self.mel);
            h.input_f32("harm_phase")?.copy_from_slice(&phase);
            h.input_f32("sine_noise")?.copy_from_slice(inputs.sine);
            h.run()?;
            self.f0.copy_from_slice(h.output("f0")?);
            h.output("audio")?
        };
        // 5. The returned span.
        let (first, n) = g.returned();
        if audio.len() < first + n {
            return Err(format!("Sigil-ZS: the vocoder returned {} samples, {} expected", audio.len(), first + n));
        }
        out[..g.pad_front].fill(0.0);
        out[g.pad_front..].copy_from_slice(&audio[first..first + n]);
        self.ticks = self.ticks.saturating_add(1);
        Ok(TickPhase { harm_phase: phase })
    }
}

/// What identifies a set of plans in the pool.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct PoolKey {
    ep: String,
    device: i32,
    prefix: Option<&'static str>,
    files: Vec<std::path::PathBuf>,
    geometry: Geometry,
    cfg: bool,
    split: bool,
    levers: (bool, bool, bool, bool, &'static str),
}

type Pool = Mutex<Vec<(PoolKey, Weak<Mutex<TickPlans>>)>>;

fn pool() -> &'static Pool {
    static POOL: std::sync::OnceLock<Pool> = std::sync::OnceLock::new();
    POOL.get_or_init(|| Mutex::new(Vec::new()))
}

/// The plans of `cfg` from the process-wide pool, built if no live voice holds them (see the module
/// docs). The pool's lock is held while building, so two loads of one shape never build it twice.
pub fn shared(cfg: &PlanConfig<'_>) -> Result<(Arc<Mutex<TickPlans>>, bool), String> {
    let dit = if cfg.cfg { GraphName::DitStep } else { GraphName::DitStep1 };
    let mut files = vec![
        cfg.graphs.path(GraphName::Xlsr, cfg.fp16_xlsr).0,
        cfg.graphs.path(dit, cfg.fp16_dit).0,
    ];
    if cfg.split {
        files.push(cfg.graphs.path(GraphName::HiftNsf, cfg.fp16_vocoder).0);
    } else {
        files.push(cfg.graphs.path(GraphName::Hift, cfg.fp16_vocoder).0);
    }
    files.push(cfg.graphs.main.clone());
    let key = PoolKey {
        ep: normalize_ep(cfg.ep).to_string(),
        device: cfg.opts.cuda_device_index,
        prefix: cfg.opts.component_prefix,
        files,
        geometry: cfg.geometry,
        cfg: cfg.cfg,
        split: cfg.split,
        levers: (cfg.levers.io_binding, cfg.levers.cuda_graphs, cfg.levers.priority, cfg.levers.tf32, cfg.levers.accelerator.as_str()),
    };
    let mut entries = pool().lock().map_err(|_| "Sigil-ZS: the plan pool is poisoned".to_string())?;
    entries.retain(|(_, w)| w.strong_count() > 0);
    if let Some(plans) = entries.iter().find(|(k, _)| *k == key).and_then(|(_, w)| w.upgrade()) {
        return Ok((plans, true));
    }
    let plans = Arc::new(Mutex::new(TickPlans::build(cfg)?));
    entries.push((key, Arc::downgrade(&plans)));
    Ok((plans, false))
}
