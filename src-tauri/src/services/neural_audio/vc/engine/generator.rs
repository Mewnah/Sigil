//! Split generator (P3-C1): `enc` over the context, then `flowdec` over the return window only
//! (partial decode, ADR 0004), plus the speaker table (`g` computed in Rust, which makes speaker
//! blending free). Legacy lever path (`partial_decode` off): `flowdec` over the whole context
//! window, returning the same tail.
//!
//! `flowdec` inputs cover one window of `F = return + flow_margin + dec_margin` frames: `m_p`,
//! `logs_p`, `z_noise` `[1,C,F]`, `g` `[1,gin,1]`, `f0` `[1,F]` (Hz), `phase0` `[1,1]` (cycles),
//! `src_noise` `[1,F·upp]`; output `audio` `[1,return·upp]`. `enc` always runs fp32 (ADR 0004).
//!
//! # Frozen by P3-S0 (signatures; P3-C1 implements)
//! - [`SpeakerTable::load`], [`SpeakerTable::vector`], [`SpeakerTable::blend`].
//! - [`SplitGenerator::load`], [`SplitGenerator::encode`], [`SplitGenerator::decode_window`],
//!   [`SplitGenerator::ep_outcomes`]. Sessions are built through `neural_audio::gpu::SessionPlan`
//!   (P3-C2), never directly.
//!
//! # Precision (ADR 0004 policy)
//! `enc` is always fp32. `flowdec` follows the precision request: `auto` = the template's
//! `layout.json` recommendation for the device class of the requested EP (fp16 on GPUs, fp32 on the
//! CPU); `fp16` / `fp32` force it. fp16 needs the voice's fp16 image (its weights fit half
//! precision) and a variant that passes parity on that device class; otherwise fp32 runs, with a
//! note. An `auto` fp16 session that the EP ladder puts on the CPU is rebuilt fp32.
//!
//! # Added by P3-C1
//! [`SpeakerTable::from_rows`], [`SpeakerTable::dim`], [`SplitGenerator::flowdec_precision`],
//! [`SplitGenerator::notes`], [`SplitGenerator::plan_levers`], [`SplitGenerator::template_has_f0`].
//!
//! # Device-resident generator (P3B-G)
//! When both plans keep their I/O on the GPU (CUDA with IOBinding and graphs), `enc`'s `m_p` and
//! `logs_p` never leave the device: [`encode`](SplitGenerator::encode) only enqueues `enc`
//! (`SessionPlan::run_async`), and [`decode_window`](SplitGenerator::decode_window) copies the
//! window's columns into `flowdec`'s inputs with a strided device-to-device copy that `flowdec`'s
//! stream orders after `enc` (a CUDA event), then runs `flowdec` and waits once. One host wait per
//! generator call instead of two, no 2 x latent x T download and re-upload, and the CPU prepares the
//! window's noise and phase while `enc` runs. Bit-identical to the host path (the same values are
//! copied). `SIGIL_VC_DEVICE_CHAIN=0` keeps the host path ([`DEVICE_CHAIN_ENV`]).
//! [`SplitGenerator::device_chain`] reports which path runs. The stage spans then time `enc` as
//! its enqueue only and `flowdec` as both GPU runs; the benchmark's per-plan GPU times
//! (`SIGIL_BENCH_GPU_TIMING=1`) split them, and its `vc.enc_flowdec_gap` time is the device gap
//! between `enc`'s end and `flowdec`'s start (`SessionPlan::gpu_gap_ms_to`), on either path.
//!
//! # Concurrent builds (P3B-G, T9)
//! `enc` and `flowdec` build on two threads ([`PARALLEL_LOAD_DEFAULT`] on, `SIGIL_VC_PARALLEL_LOAD=0`
//! turns it off; the helper thread takes the loading thread's priority and EcoQoS state, so a
//! background load stays a background load, and the engine load passes `false` for one). Their VRAM
//! is measured over both builds together and split between the two plans in proportion to what each
//! saw. The same switch drives the engine load's parallel cold start (`load.rs`).

use std::path::Path;

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::gpu::arena::{self, VramProbe};
use crate::services::neural_audio::gpu::{ExternalWeights, GpuLevers, PlanSpec, Residency, RowCopy, SessionPlan, TensorSpec};
use crate::services::neural_audio::ort_session::{device_for, normalize_ep, ModelSource, VcSessionOptions};
use crate::services::neural_audio::telemetry::{self, LoadStage};
use crate::services::neural_audio::vc::import::layout::{Parity, Template};
use crate::services::neural_audio::vc::import::session::WeightImages;

use super::options::{EffectiveLevers, PrecisionMode};
use super::plan::ShapePlan;

/// Speaker vectors of a voice (`speakers.f32.bin`: `count × dim` f32 little endian).
pub struct SpeakerTable {
    dim: usize,
    rows: Vec<f32>,
}

impl SpeakerTable {
    pub fn load(path: &Path, dim: usize) -> Result<SpeakerTable, String> {
        if dim == 0 {
            return Err("speaker vectors of dimension 0".into());
        }
        let bytes = std::fs::read(path).map_err(|e| format!("speaker table {}: {e}", path.display()))?;
        if bytes.is_empty() || !bytes.len().is_multiple_of(4 * dim) {
            return Err(format!(
                "speaker table {} holds {} bytes, not a whole number of {dim}-value rows",
                path.display(),
                bytes.len()
            ));
        }
        let rows: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Self::from_rows(rows, dim).map_err(|e| format!("speaker table {}: {e}", path.display()))
    }

    /// A table from `count × dim` values (row-major).
    pub fn from_rows(rows: Vec<f32>, dim: usize) -> Result<SpeakerTable, String> {
        if dim == 0 || rows.is_empty() || !rows.len().is_multiple_of(dim) {
            return Err(format!("{} values are not whole {dim}-value speaker rows", rows.len()));
        }
        if rows.iter().any(|v| !v.is_finite()) {
            return Err("the speaker table holds non-finite values".into());
        }
        Ok(SpeakerTable { dim, rows })
    }

    pub fn count(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.rows.len() / self.dim
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The vector of speaker `id` (clamped to the table).
    pub fn vector(&self, id: i64) -> &[f32] {
        let n = self.count();
        if n == 0 {
            return &[];
        }
        let i = id.clamp(0, n as i64 - 1) as usize;
        &self.rows[i * self.dim..(i + 1) * self.dim]
    }

    /// Weighted blend of speakers into `out` (`dim` values; weights are normalised).
    pub fn blend(&self, weights: &[(i64, f32)], out: &mut [f32]) -> Result<(), String> {
        if out.len() != self.dim {
            return Err(format!("speaker blend into {} values, the table has {}", out.len(), self.dim));
        }
        let total: f32 = weights
            .iter()
            .filter(|(_, w)| w.is_finite() && *w > 0.0)
            .map(|(_, w)| *w)
            .sum();
        if !(total.is_finite() && total > 0.0) {
            return Err("a speaker blend needs at least one positive weight".into());
        }
        out.fill(0.0);
        for &(id, w) in weights.iter().filter(|(_, w)| w.is_finite() && *w > 0.0) {
            let k = w / total;
            for (o, &v) in out.iter_mut().zip(self.vector(id)) {
                *o += k * v;
            }
        }
        Ok(())
    }
}

/// `SIGIL_VC_DEVICE_CHAIN=0` keeps `enc`'s outputs on the host path (P3B-G's kill switch; see the
/// module docs).
pub const DEVICE_CHAIN_ENV: &str = "SIGIL_VC_DEVICE_CHAIN";

/// `SIGIL_VC_PARALLEL_LOAD=0|1` builds `enc` and `flowdec` concurrently or one after the other (P3B-G;
/// see the module docs).
pub const PARALLEL_LOAD_ENV: &str = "SIGIL_VC_PARALLEL_LOAD";

/// Whether `enc` and `flowdec` build concurrently when [`PARALLEL_LOAD_ENV`] is unset: on (P3B-I0,
/// from P3B-G's interleaved load gates: DirectML Hodges-Lehmann -209 ms (95 % CI -451..-107, 7
/// pairs), CUDA -114 ms over 16 pairs (CI -621..-64), VRAM equal; the engine load's own parallel
/// cold start, `load::PARALLEL_LOAD_DEFAULT`, measured both together). `SIGIL_VC_PARALLEL_LOAD=0`
/// turns both off. The engine load passes `false` for a background hot-switch load, which builds
/// one session at a time beside the playing voice ([`SplitGenerator::load_branch`]).
pub const PARALLEL_LOAD_DEFAULT: bool = true;

fn device_chain_enabled() -> bool {
    !matches!(std::env::var(DEVICE_CHAIN_ENV).as_deref().map(str::trim), Ok("0" | "off" | "false"))
}

fn parallel_load_enabled() -> bool {
    match std::env::var(PARALLEL_LOAD_ENV).as_deref().map(str::trim) {
        Ok("1" | "on" | "true") => true,
        Ok("0" | "off" | "false") => false,
        _ => PARALLEL_LOAD_DEFAULT,
    }
}

/// Device class of the first rung of `ep`'s ladder (`cpu` / `gpu`).
fn device_class(ep: &str) -> &'static str {
    if normalize_ep(ep) == "cpu" {
        "cpu"
    } else {
        "gpu"
    }
}

/// The flowdec variant for a precision request (see the module docs), and why fp16 was refused.
fn flowdec_variant(template: &Template, precision: PrecisionMode, class: &str, f16_image: bool) -> (&'static str, Option<String>) {
    let layout = &template.layout;
    let recommended = layout
        .graphs
        .get("flowdec")
        .and_then(|g| g.recommended.get(class))
        .map(String::as_str)
        .unwrap_or("fp32");
    let wanted = match precision {
        PrecisionMode::Auto => recommended,
        PrecisionMode::Fp32 => "fp32",
        PrecisionMode::Fp16 => "fp16",
    };
    if wanted != "fp16" {
        return ("fp32", None);
    }
    let class_ep = if class == "cpu" { "cpu" } else { "cuda" };
    if layout.graph_variant("flowdec", "fp16").is_err() {
        return (
            "fp32",
            Some(format!("precision: template {} has no fp16 flowdec; running fp32", layout.template)),
        );
    }
    if layout.parity("flowdec", "fp16", class_ep) == Parity::Fail {
        return (
            "fp32",
            Some(format!(
                "precision: fp16 flowdec fails the parity gate on {class} execution providers; running fp32"
            )),
        );
    }
    if !f16_image {
        return ("fp32", Some("precision: this voice's weights do not fit fp16; running fp32".into()));
    }
    ("fp16", None)
}

/// The `enc` and `flowdec` sessions of one voice at the plan's fixed shapes.
pub struct SplitGenerator {
    enc: SessionPlan,
    flowdec: SessionPlan,
    outcomes: Vec<(String, EpOutcome)>,
    notes: Vec<String>,
    f0: bool,
    latent: usize,
    gin: usize,
    enc_frames: usize,
    flow_frames: usize,
    samples: usize,
    flowdec_precision: &'static str,
    /// `enc`'s outputs reach `flowdec` on the device (see the module docs).
    chain: bool,
}

impl SplitGenerator {
    /// Build both sessions (weights from memory, free dimensions pinned to `plan`) on `ep` with the
    /// GPU levers of `levers`.
    pub fn load(
        template: &Template,
        weights: &WeightImages<'_>,
        plan: &ShapePlan,
        ep: &str,
        opts: &VcSessionOptions,
        levers: &EffectiveLevers,
    ) -> Result<SplitGenerator, String> {
        Self::load_branch(template, weights, plan, ep, opts, levers, true)
    }

    /// [`load`](Self::load) as the engine's load runs it: `allow_parallel` false builds `enc` and
    /// `flowdec` one after the other whatever [`PARALLEL_LOAD_ENV`] / [`PARALLEL_LOAD_DEFAULT`] say
    /// (a background hot-switch load beside a playing voice, or the engine load's own switch off).
    pub fn load_branch(
        template: &Template,
        weights: &WeightImages<'_>,
        plan: &ShapePlan,
        ep: &str,
        opts: &VcSessionOptions,
        levers: &EffectiveLevers,
        allow_parallel: bool,
    ) -> Result<SplitGenerator, String> {
        Self::load_ext(template, weights, plan, ep, opts, levers, device_chain_enabled(), allow_parallel && parallel_load_enabled())
    }

    /// [`load`](Self::load) with the device chain allowed or not and the builds concurrent or not
    /// (the environment switches' values; tests compare both paths in one process).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn load_ext(
        template: &Template,
        weights: &WeightImages<'_>,
        plan: &ShapePlan,
        ep: &str,
        opts: &VcSessionOptions,
        levers: &EffectiveLevers,
        allow_chain: bool,
        parallel: bool,
    ) -> Result<SplitGenerator, String> {
        let layout = &template.layout;
        let arch = &layout.arch;
        let (emb, latent, gin, upp) = (
            arch.emb_channels as usize,
            arch.inter_channels as usize,
            arch.gin_channels as usize,
            arch.upp as usize,
        );
        if upp != plan.upp {
            return Err(format!("template {} has {upp} samples per frame, the plan {}", layout.template, plan.upp));
        }
        let gpu_levers = GpuLevers::from_effective(levers);
        let mut notes = Vec::new();
        if levers.precision == PrecisionMode::Fp16 {
            notes.push("precision: enc always runs fp32 (fp16 misses the parity gate on GPUs, ADR 0004)".to_string());
        }
        let (t, f) = (plan.enc_frames, plan.flow_frames);
        let mut enc_inputs = vec![TensorSpec::f32("phone", &[1, t, emb])];
        if arch.f0 {
            enc_inputs.push(TensorSpec::i64("pitch", &[1, t]));
        }
        let enc_outputs = [TensorSpec::f32("m_p", &[1, latent, t]), TensorSpec::f32("logs_p", &[1, latent, t])];

        let class = device_class(ep);
        let (variant, note) = flowdec_variant(template, levers.precision, class, weights.f16.is_some());
        notes.extend(note);
        let samples = f * upp;
        let mut fd_inputs = vec![
            TensorSpec::f32("m_p", &[1, latent, f]),
            TensorSpec::f32("logs_p", &[1, latent, f]),
            TensorSpec::f32("z_noise", &[1, latent, f]),
            TensorSpec::f32("g", &[1, gin, 1]),
        ];
        if arch.f0 {
            fd_inputs.push(TensorSpec::f32("f0", &[1, f]));
            fd_inputs.push(TensorSpec::f32("phase0", &[1, 1]));
            fd_inputs.push(TensorSpec::f32("src_noise", &[1, samples]));
        }
        let fd_outputs = [TensorSpec::f32("audio", &[1, plan.decode_frames * upp])];
        let dims = [("F", f as i64), ("S", samples as i64)];
        let build_enc = || {
            let _span = telemetry::load_span(LoadStage::EncSessionInit);
            build_plan(template, "enc", "fp32", weights, ep, opts, &[("T", t as i64)], &enc_inputs, &enc_outputs, gpu_levers, "vc.enc")
        };
        let build_flowdec = || {
            let _span = telemetry::load_span(LoadStage::FlowdecSessionInit);
            build_plan(template, "flowdec", variant, weights, ep, opts, &dims, &fd_inputs, &fd_outputs, gpu_levers, "vc.flowdec")
        };
        let (enc, mut flowdec) = if parallel {
            build_concurrently(ep, opts, build_enc, build_flowdec)?
        } else {
            let enc = build_enc()?;
            (enc, build_flowdec()?)
        };
        let mut flowdec_precision = variant;
        let mut flowdec_outcome = flowdec.outcome().clone();
        if variant == "fp16" && levers.precision == PrecisionMode::Auto && flowdec_outcome.active == "cpu" {
            // The ladder fell back to the CPU, where the recommendation is fp32 (the CPU fp16
            // path is emulated and slow). The outcome keeps the original request and reason.
            let _span = telemetry::load_span(LoadStage::FlowdecSessionInit);
            flowdec = build_plan(
                template,
                "flowdec",
                "fp32",
                weights,
                "cpu",
                opts,
                &dims,
                &fd_inputs,
                &fd_outputs,
                gpu_levers,
                "vc.flowdec",
            )?;
            flowdec_precision = "fp32";
            notes.push(format!(
                "precision: flowdec fell back to the CPU ({}); running the fp32 graph there",
                flowdec_outcome
                    .reason
                    .as_deref()
                    .unwrap_or("GPU unavailable")
            ));
            crate::services::diagnostics::record_ep_outcome(&component_key(opts, "vc.flowdec"), flowdec_outcome.clone());
        } else {
            flowdec_outcome = flowdec.outcome().clone();
        }
        let outcomes = vec![("vc.enc".to_string(), enc.outcome().clone()), ("vc.flowdec".to_string(), flowdec_outcome)];
        let mut enc = enc;
        let chain = allow_chain && enc.residency() == Residency::Device && flowdec.residency() == Residency::Device;
        if chain {
            // m_p / logs_p are read only by the device copy into flowdec.
            enc.set_output_download("m_p", false)?;
            enc.set_output_download("logs_p", false)?;
        }
        Ok(SplitGenerator {
            enc,
            flowdec,
            outcomes,
            notes,
            f0: arch.f0,
            latent,
            gin,
            enc_frames: t,
            flow_frames: f,
            samples,
            flowdec_precision,
            chain,
        })
    }

    /// Run `enc` over the context: `phone` `[T, emb]` (2x-upsampled content features on the
    /// 10 ms grid), `pitch` `[T]` (coarse f0 bins; empty for nof0 templates).
    pub fn encode(&mut self, phone: &[f32], pitch: &[i64]) -> Result<(), String> {
        let dst = self.enc.input_f32("phone")?;
        if dst.len() != phone.len() {
            return Err(format!("enc phone: {} values, the plan takes {}", phone.len(), dst.len()));
        }
        dst.copy_from_slice(phone);
        if self.f0 {
            let dst = self.enc.input_i64("pitch")?;
            if dst.len() != pitch.len() {
                return Err(format!("enc pitch: {} frames, the plan takes {}", pitch.len(), dst.len()));
            }
            dst.copy_from_slice(pitch);
        }
        if self.chain {
            // Enqueued only: flowdec's stream waits for it on the device (`decode_window`).
            self.enc.run_async().map_err(|e| format!("enc: {e}"))
        } else {
            self.enc.run().map_err(|e| format!("enc: {e}"))
        }
    }

    /// Run `flowdec` over the window starting at frame `window_start` of the last `encode`,
    /// writing `return_frames · upp` samples into `out`.
    #[allow(clippy::too_many_arguments)]
    pub fn decode_window(
        &mut self,
        window_start: usize,
        z_noise: &[f32],
        g: &[f32],
        f0_hz: &[f32],
        phase0: f64,
        src_noise: &[f32],
        out: &mut [f32],
    ) -> Result<(), String> {
        let (t, f, c) = (self.enc_frames, self.flow_frames, self.latent);
        if window_start + f > t {
            return Err(format!("flowdec window {window_start}+{f} runs past the {t}-frame context"));
        }
        if self.chain {
            // The window's columns of every channel, device to device on flowdec's stream after
            // enc's run.
            let copy = RowCopy { rows: c, row_len: f, src_stride: t, src_offset: window_start, dst_stride: f };
            for name in ["m_p", "logs_p"] {
                self.enc
                    .copy_rows_to(name, &mut self.flowdec, name, copy)
                    .map_err(|e| format!("enc → flowdec: {e}"))?;
            }
        } else {
            for name in ["m_p", "logs_p"] {
                let src = self.enc.output(name)?;
                let dst = self.flowdec.input_f32(name)?;
                for ch in 0..c {
                    dst[ch * f..(ch + 1) * f].copy_from_slice(&src[ch * t + window_start..ch * t + window_start + f]);
                }
            }
        }
        copy_exact(self.flowdec.input_f32("z_noise")?, z_noise, "z_noise")?;
        copy_exact(self.flowdec.input_f32("g")?, g, "g")?;
        if self.f0 {
            copy_exact(self.flowdec.input_f32("f0")?, f0_hz, "f0")?;
            self.flowdec.input_f32("phase0")?[0] = phase0.rem_euclid(1.0) as f32;
            copy_exact(self.flowdec.input_f32("src_noise")?, src_noise, "src_noise")?;
        }
        self.flowdec.run().map_err(|e| format!("flowdec: {e}"))?;
        if self.chain {
            // flowdec ran after enc on the device: enc has completed; settle it without a wait.
            self.enc.poll().map_err(|e| format!("enc: {e}"))?;
        }
        if telemetry::gpu_timing_enabled() {
            // The benchmark's T2 probe: the device gap between enc's end and flowdec's start.
            if let Some(ms) = self.enc.gpu_gap_ms_to(&self.flowdec) {
                telemetry::record_gpu_time("vc.enc_flowdec_gap", ms);
            }
        }
        let audio = self.flowdec.output("audio")?;
        if audio.len() != out.len() {
            return Err(format!("flowdec returned {} samples, expected {}", audio.len(), out.len()));
        }
        out.copy_from_slice(audio);
        Ok(())
    }

    /// EP outcome per session (`vc.enc`, `vc.flowdec`).
    pub fn ep_outcomes(&self) -> &[(String, EpOutcome)] {
        &self.outcomes
    }

    /// Why a requested precision is not used (shown with the load warnings).
    pub fn notes(&self) -> &[String] {
        &self.notes
    }

    /// `fp32` or `fp16`.
    pub fn flowdec_precision(&self) -> &'static str {
        self.flowdec_precision
    }

    /// The GPU levers each plan really uses (`enc`, `flowdec`).
    pub fn plan_levers(&self) -> (GpuLevers, GpuLevers) {
        (self.enc.levers(), self.flowdec.levers())
    }

    /// Whether `enc`'s outputs reach `flowdec` on the device (P3B-G; see the module docs).
    pub fn device_chain(&self) -> bool {
        self.chain
    }

    pub fn template_has_f0(&self) -> bool {
        self.f0
    }

    /// Speaker vector width (`gin_channels`).
    pub fn speaker_dim(&self) -> usize {
        self.gin
    }

    /// NSF source samples of one window (`F · upp`).
    pub fn window_samples(&self) -> usize {
        self.samples
    }
}

/// Build `enc` and `flowdec` on two threads (see the module docs): `enc` on a scoped helper thread
/// that takes this thread's scheduling policy, `flowdec` here. The two plans' VRAM is measured over
/// both builds and split in proportion to what each build saw on its own (each saw the other's
/// allocations too).
fn build_concurrently(
    ep: &str,
    opts: &VcSessionOptions,
    build_enc: impl FnOnce() -> Result<SessionPlan, String> + Send,
    build_flowdec: impl FnOnce() -> Result<SessionPlan, String>,
) -> Result<(SessionPlan, SessionPlan), String> {
    let base = normalize_ep(ep);
    let probe = arena::dxgi_adapter_for(base, device_for(base, base, opts)).and_then(VramProbe::start);
    let policy = ThreadPolicy::of_current();
    let (enc, flowdec) = std::thread::scope(|scope| {
        let helper = std::thread::Builder::new()
            .name("sigil-vc-load".into())
            .spawn_scoped(scope, move || {
                policy.apply_to_current();
                build_enc()
            });
        let flowdec = build_flowdec();
        let enc = match helper {
            Ok(h) => h.join().unwrap_or_else(|_| Err("the enc build panicked".to_string())),
            Err(e) => Err(format!("the enc build thread could not start: {e}")),
        };
        (enc, flowdec)
    });
    let (enc, flowdec) = (enc?, flowdec?);
    if let (Some(joint), Some(e), Some(d)) = (probe.and_then(|p| p.delta_mb()), enc.vram_mb(), flowdec.vram_mb()) {
        if e + d > 0.0 {
            enc.set_vram_mb(joint * e / (e + d));
            flowdec.set_vram_mb(joint * d / (e + d));
        }
    }
    Ok((enc, flowdec))
}

/// The loading thread's priority and EcoQoS state, copied onto the concurrent build's helper thread
/// (a background hot-voice load stays background on both threads).
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

/// The diagnostics row of `component` for sessions built with `opts` (`build_session_with`
/// prefixes it with `component_prefix`, e.g. `bench.vc.flowdec`).
pub(crate) fn component_key(opts: &VcSessionOptions, component: &str) -> String {
    match opts.component_prefix {
        Some(p) if !p.is_empty() => format!("{p}.{component}"),
        _ => component.to_string(),
    }
}

fn copy_exact(dst: &mut [f32], src: &[f32], what: &str) -> Result<(), String> {
    if dst.len() != src.len() {
        return Err(format!("flowdec {what}: {} values, the plan takes {}", src.len(), dst.len()));
    }
    dst.copy_from_slice(src);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_plan(
    template: &Template,
    graph: &str,
    variant: &'static str,
    weights: &WeightImages<'_>,
    ep: &str,
    opts: &VcSessionOptions,
    dims: &[(&str, i64)],
    inputs: &[TensorSpec],
    outputs: &[TensorSpec],
    levers: GpuLevers,
    component: &str,
) -> Result<SessionPlan, String> {
    let layout = &template.layout;
    let v = layout
        .graph_variant(graph, variant)
        .map_err(|e| e.to_string())?;
    let model = template
        .graph_bytes(graph, variant)
        .map_err(|e| e.to_string())?;
    let external = match &v.weights_file {
        Some(file) => {
            let extent = layout.graph_extent_f32(graph) as usize;
            let (image, need) = match variant {
                "fp16" => (
                    weights
                        .f16
                        .ok_or_else(|| format!("{graph}.fp16 needs the voice's fp16 weights"))?,
                    extent / 2,
                ),
                _ => (weights.f32, extent),
            };
            if image.len() < need {
                return Err(format!(
                    "{graph}.{variant}: weights image of {} bytes, the graph reads {need}",
                    image.len()
                ));
            }
            Some(ExternalWeights {
                file: file.as_str(),
                image: &image[..need],
            })
        }
        None => None,
    };
    SessionPlan::build(&PlanSpec {
        component,
        ep,
        opts,
        model: ModelSource::Memory(&model),
        weights: external,
        dims,
        inputs,
        outputs,
        levers,
    })
    .map_err(|e| format!("{graph}.{variant} of {}: {e}", layout.template))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::services::neural_audio::vc::engine::noise::{AnchoredNoise, NoiseFrames, NoiseStream, PhaseTracker};
    use crate::services::neural_audio::vc::import::{import_file, ImportedVoice};
    use std::path::PathBuf;

    pub(crate) fn tiny_dir() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("test-fixtures")
            .join("vc")
            .join("tiny")
    }

    pub(crate) fn tiny_voice() -> ImportedVoice {
        let dir = tiny_dir();
        import_file(&dir.join("tiny.pth"), &dir).expect("the tiny fixture imports")
    }

    fn streaming_plan(voice: &ImportedVoice, block_16k: usize, want_16k: usize, extra_16k: usize, partial: bool) -> ShapePlan {
        let st = &voice.template.layout.streaming;
        let upp = voice.template.layout.arch.upp;
        ShapePlan::for_stream(st, upp * 100, partial, block_16k, want_16k, extra_16k).unwrap()
    }

    #[test]
    fn speaker_vectors_clamp_and_blend() {
        let t = SpeakerTable::from_rows(vec![1.0, 0.0, 0.0, 1.0, 2.0, 2.0], 2).unwrap();
        assert_eq!(t.count(), 3);
        assert_eq!(t.vector(1), &[0.0, 1.0]);
        assert_eq!(t.vector(-4), &[1.0, 0.0]);
        assert_eq!(t.vector(99), &[2.0, 2.0]);
        let mut out = [0.0; 2];
        t.blend(&[(0, 1.0), (1, 3.0), (2, f32::NAN)], &mut out)
            .unwrap();
        assert_eq!(out, [0.25, 0.75]);
        assert!(t.blend(&[(0, 0.0)], &mut out).is_err());
        assert!(t.blend(&[(0, 1.0)], &mut [0.0; 3]).is_err());
        assert!(SpeakerTable::from_rows(vec![1.0, f32::INFINITY], 2).is_err());
        assert!(SpeakerTable::from_rows(vec![1.0, 2.0, 3.0], 2).is_err());
        let dir = std::env::temp_dir().join(format!("sigil-speakers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("speakers.f32.bin");
        std::fs::write(
            &p,
            [1.0f32, 2.0, 3.0, 4.0]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<u8>>(),
        )
        .unwrap();
        let t = SpeakerTable::load(&p, 2).unwrap();
        assert_eq!(t.vector(1), &[3.0, 4.0]);
        assert!(SpeakerTable::load(&p, 3).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Deterministic content / pitch inputs for `t` frames starting at absolute frame `abs0`
    /// (smooth in time, so consecutive windows describe the same "voice" frame for frame).
    fn inputs_at(abs0: u64, t: usize, emb: usize) -> (Vec<f32>, Vec<i64>, Vec<f32>) {
        let mut phone = Vec::with_capacity(t * emb);
        let mut pitch = Vec::with_capacity(t);
        let mut f0 = Vec::with_capacity(t);
        for j in 0..t {
            let a = (abs0 + j as u64) as f32;
            for k in 0..emb {
                phone.push(0.5 * (0.13 * a + 0.07 * k as f32).sin());
            }
            let hz = if (a as u64 % 50) < 42 { 120.0 + 60.0 * (0.05 * a).sin() } else { 0.0 };
            f0.push(hz);
            pitch.push(if hz > 0.0 { 40 + (hz as i64 % 200) } else { 1 });
        }
        (phone, pitch, f0)
    }

    /// Partial vs full decode of the same inputs on `ep` at `precision` (see the test below):
    /// `(max |error| of the partial decode against the full decode's tail, peak, flowdec
    /// precision, EP in use)`.
    fn partial_vs_full(voice: &ImportedVoice, ep: &str, precision: PrecisionMode, extra_16k: usize) -> (f32, f32, &'static str, String) {
        let arch = voice.template.layout.arch.clone();
        let (emb, c, upp) = (arch.emb_channels as usize, arch.inter_channels as usize, arch.upp as usize);
        let f16 = (precision != PrecisionMode::Fp32)
            .then(|| crate::services::neural_audio::vc::import::writer::f16_image(&voice.weights.f32_image).unwrap());
        let images = WeightImages {
            f32: &voice.weights.f32_image,
            f16: f16.as_deref(),
        };
        let opts = VcSessionOptions::default();
        let levers = EffectiveLevers {
            precision,
            ..EffectiveLevers::default()
        };
        let partial = streaming_plan(voice, 640, 640 + 481, extra_16k, true);
        let full = streaming_plan(voice, 640, 640 + 481, extra_16k, false);
        assert_eq!(partial.enc_frames, full.enc_frames);
        let mut gp = SplitGenerator::load(&voice.template, &images, &partial, ep, &opts, &levers).unwrap();
        let mut gf = SplitGenerator::load(&voice.template, &images, &full, ep, &opts, &levers).unwrap();
        assert_eq!(gp.flowdec_precision(), gf.flowdec_precision());
        assert_eq!(gp.ep_outcomes().len(), 2);
        let t = partial.enc_frames;
        let (phone, pitch, f0) = inputs_at(1_000, t, emb);
        gp.encode(&phone, &pitch).unwrap();
        gf.encode(&phone, &pitch).unwrap();
        let noise = AnchoredNoise::new(11);
        let speakers = SpeakerTable::from_rows(voice.weights.speakers.clone(), voice.weights.speaker_dim).unwrap();
        let g = speakers.vector(1).to_vec();
        let run = |gen: &mut SplitGenerator, plan: &ShapePlan, phase0: f64| -> Vec<f32> {
            let f = plan.flow_frames;
            let first = 1_000 + plan.window_start as u64;
            let mut z = vec![0.0; c * f];
            NoiseFrames::new(noise, NoiseStream::Flow, c, f).copy_window(first, f, 0.66666, true, &mut z);
            let mut s = vec![0.0; f * upp];
            NoiseFrames::new(noise, NoiseStream::Source, upp, f).copy_window(first, f, 1.0, false, &mut s);
            let mut out = vec![0.0; plan.decode_frames * upp];
            gen.decode_window(
                plan.window_start,
                &z,
                &g,
                &f0[plan.window_start..plan.window_start + f],
                phase0,
                &s,
                &mut out,
            )
            .unwrap();
            out
        };
        let whole = run(&mut gf, &full, 0.3);
        let mut tracker = PhaseTracker::new(upp, arch.sample_rate);
        let skip = partial.window_start - full.window_start;
        tracker.advance(&f0[..skip]);
        let phase = (0.3 + tracker.phase_at()).rem_euclid(1.0);
        let tail = run(&mut gp, &partial, phase);
        assert_eq!(tail.len(), partial.return_frames * upp);
        let expect = &whole[whole.len() - tail.len()..];
        let peak = expect.iter().fold(0f32, |a, &x| a.max(x.abs()));
        let err = tail
            .iter()
            .zip(expect)
            .fold(0f32, |a, (x, y)| a.max((x - y).abs()));
        let active = gp.ep_outcomes()[1].1.active.clone();
        (err, peak, gp.flowdec_precision(), active)
    }

    /// ADR 0004 on the engine's own sessions: the partial decode's audio equals the tail of a full
    /// decode of the same inputs (same noise, phase0 advanced over the skipped frames), fp32 CPU.
    #[test]
    fn partial_decode_equals_the_full_decode_tail() {
        let voice = tiny_voice();
        let (err, peak, precision, active) = partial_vs_full(&voice, "cpu", PrecisionMode::Fp32, 3_200);
        assert_eq!((precision, active.as_str()), ("fp32", "cpu"));
        assert!(peak > 1e-4, "the tiny voice produced near-silence ({peak})");
        assert!(err <= 1e-4, "partial decode differs from the full decode's tail by {err} (peak {peak})");
    }

    /// The same on the MIT default voice (v2 40k, bundled template) on every EP of
    /// `SIGIL_BENCH_EPS` (default `cpu,dml`), fp32 and the GPU fp16 recommendation; gates of ADR
    /// 0004 (fp32 1e-4, fp16 5e-3).
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache cargo test default_voice_partial_decode -- --ignored --nocapture`
    #[test]
    #[ignore = "needs SIGIL_TEST_MODELS_DIR with rvc-default-voice/default.pth"]
    fn default_voice_partial_decode_matches_full_decode() {
        let Some(models) = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from) else {
            eprintln!("SIGIL_TEST_MODELS_DIR is not set; skipped");
            return;
        };
        let templates = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("resources")
            .join("vc-templates");
        let voice = import_file(&models.join("rvc-default-voice").join("default.pth"), &templates).unwrap();
        let cuda_ok = crate::services::neural_audio::ort_session::preload_test_cuda_runtime();
        let eps = std::env::var("SIGIL_BENCH_EPS").unwrap_or_else(|_| "cpu,dml".into());
        for ep in eps
            .split(',')
            .map(str::trim)
            .filter(|e| !e.is_empty() && (*e != "cuda" || cuda_ok))
        {
            for precision in [PrecisionMode::Fp32, PrecisionMode::Auto] {
                if ep == "cpu" && precision == PrecisionMode::Auto {
                    continue;
                }
                for extra in [1_024usize, 8_000] {
                    let (err, peak, used, active) = partial_vs_full(&voice, ep, precision, extra);
                    eprintln!("{ep}→{active} {used} extra {extra}: partial vs full tail max |err| {err:.2e} (peak {peak:.3})");
                    let gate = if used == "fp16" { 5e-3 } else { 1e-4 };
                    // ONNX Runtime's CUDA EP computes fp32 convolutions in TF32 by default (ADR 0004).
                    let gate = if active == "cuda" && used == "fp32" { 2e-3 } else { gate };
                    assert!(err <= gate, "{ep} {used}: {err} over {gate}");
                }
            }
        }
    }

    /// Two consecutive ticks (windows one block apart over the same content) decode identical audio
    /// on the frames they share, wherever neither window's edge reaches (the flow and decoder see
    /// `flow_margin + dec_margin` frames each way), when the noise is anchored and the NSF phase is
    /// carried; the legacy paths (fresh noise per tick, phase 0) differ there by the signal level.
    /// (Within reach of the older window's right edge both paths also carry that edge's effect;
    /// the random tiny voice makes that part a coin toss, so it is measured on real voices by the
    /// benchmark, not asserted here.)
    #[test]
    fn anchored_noise_and_phase0_make_overlaps_identical() {
        let voice = tiny_voice();
        let arch = voice.template.layout.arch.clone();
        let (emb, c, upp) = (arch.emb_channels as usize, arch.inter_channels as usize, arch.upp as usize);
        let margin = (voice.template.layout.streaming.flow_margin + voice.template.layout.streaming.dec_margin) as usize;
        let images = WeightImages {
            f32: &voice.weights.f32_image,
            f16: None,
        };
        let opts = VcSessionOptions::default();
        let levers = EffectiveLevers {
            precision: PrecisionMode::Fp32,
            ..EffectiveLevers::default()
        };
        let block = 4usize;
        // A long return window so the shared frames include some far from both window edges; extra
        // context so both windows fit in one encode (identical m_p / logs_p for both).
        let ret = 2 * margin + block + 16;
        let plan = streaming_plan(&voice, block * 160, ret * 160, (ret + margin + block + 2) * 160, true);
        assert!(plan.window_start >= block);
        let mut gen = SplitGenerator::load(&voice.template, &images, &plan, "cpu", &opts, &levers).unwrap();
        let speakers = SpeakerTable::from_rows(voice.weights.speakers.clone(), voice.weights.speaker_dim).unwrap();
        let g = speakers.vector(0).to_vec();
        let noise = AnchoredNoise::new(5);
        let (t, f) = (plan.enc_frames, plan.flow_frames);
        let (phone, pitch, f0) = inputs_at(0, t, emb);
        gen.encode(&phone, &pitch).unwrap();
        let decode = |gen: &mut SplitGenerator, ws: usize, fresh_seed: Option<u64>, phase0: f64| -> Vec<f32> {
            let mut z = vec![0.0; c * f];
            let mut s = vec![0.0; f * upp];
            match fresh_seed {
                None => {
                    NoiseFrames::new(noise, NoiseStream::Flow, c, f).copy_window(ws as u64, f, 0.66666, true, &mut z);
                    NoiseFrames::new(noise, NoiseStream::Source, upp, f).copy_window(ws as u64, f, 1.0, false, &mut s);
                }
                Some(seed) => {
                    let fresh = AnchoredNoise::new(seed);
                    fresh.fill(NoiseStream::Flow, 0, 0.66666, &mut z);
                    fresh.fill(NoiseStream::Source, 0, 1.0, &mut s);
                }
            }
            let mut out = vec![0.0; plan.decode_frames * upp];
            gen.decode_window(ws, &z, &g, &f0[ws..ws + f], phase0, &s, &mut out)
                .unwrap();
            out
        };
        let (ws0, ws1) = (plan.window_start - block, plan.window_start);
        let mut tracker = PhaseTracker::new(upp, arch.sample_rate);
        tracker.advance(&f0[..ws0]);
        let a = decode(&mut gen, ws0, None, tracker.phase_at());
        tracker.advance(&f0[ws0..ws1]);
        let b = decode(&mut gen, ws1, None, tracker.phase_at());
        let la = decode(&mut gen, ws0, Some(100), 0.0);
        let lb = decode(&mut gen, ws1, Some(101), 0.0);
        // Absolute frame j is output frame j - (ws + margin) of a window starting at ws.
        let frame = |out: &[f32], ws: usize, j: usize| -> Vec<f32> {
            let k = j - (ws + margin);
            out[k * upp..(k + 1) * upp].to_vec()
        };
        let max_diff = |x: &[f32], ws_x: usize, y: &[f32], ws_y: usize, frames: std::ops::Range<usize>| -> f32 {
            frames
                .flat_map(|j| {
                    frame(x, ws_x, j)
                        .into_iter()
                        .zip(frame(y, ws_y, j))
                        .map(|(p, q)| (p - q).abs())
                        .collect::<Vec<_>>()
                })
                .fold(0.0f32, f32::max)
        };
        let shared = (ws1 + margin)..(ws0 + f);
        let interior = (ws1 + margin)..(ws0 + f - margin - 2);
        let interior_len = interior.len();
        assert!(interior_len >= 8, "{interior:?}");
        let peak = a.iter().fold(0f32, |m, &x| m.max(x.abs()));
        assert!(peak > 1e-4, "near-silent tiny voice ({peak})");
        let anchored_interior = max_diff(&a, ws0, &b, ws1, interior.clone());
        let legacy_interior = max_diff(&la, ws0, &lb, ws1, interior);
        assert!(
            anchored_interior <= 1e-4 * peak.max(1.0),
            "anchored interior differs by {anchored_interior} (peak {peak})"
        );
        assert!(legacy_interior > 0.05 * peak, "legacy interior {legacy_interior} (peak {peak})");
        assert!(shared.len() > interior_len, "the edge region is part of the overlap");
    }

    /// Run `ticks` generator ticks (consecutive windows of one stream) and return every tick's
    /// audio: the same deterministic inputs for every generator it is given.
    fn render_ticks(gen: &mut SplitGenerator, voice: &ImportedVoice, plan: &ShapePlan, ticks: usize) -> Vec<Vec<f32>> {
        let arch = &voice.template.layout.arch;
        let (emb, c, upp) = (arch.emb_channels as usize, arch.inter_channels as usize, arch.upp as usize);
        let noise = AnchoredNoise::new(23);
        let speakers = SpeakerTable::from_rows(voice.weights.speakers.clone(), voice.weights.speaker_dim).unwrap();
        let g = speakers.vector(0).to_vec();
        let (t, f) = (plan.enc_frames, plan.flow_frames);
        (0..ticks)
            .map(|i| {
                let abs0 = (i * 4) as u64;
                let (phone, pitch, f0) = inputs_at(abs0, t, emb);
                gen.encode(&phone, &pitch).unwrap();
                let ws = plan.window_start;
                let mut z = vec![0.0; c * f];
                NoiseFrames::new(noise, NoiseStream::Flow, c, f).copy_window(abs0 + ws as u64, f, 0.66666, true, &mut z);
                let mut src = vec![0.0; f * upp];
                NoiseFrames::new(noise, NoiseStream::Source, upp, f).copy_window(abs0 + ws as u64, f, 1.0, false, &mut src);
                let mut out = vec![0.0; plan.decode_frames * upp];
                gen.decode_window(ws, &z, &g, &f0[ws..ws + f], 0.25, &src, &mut out).unwrap();
                out
            })
            .collect()
    }

    /// P3B-G: building `enc` and `flowdec` concurrently changes nothing but the load's wall time
    /// (CPU here: bit-identical audio over consecutive ticks), and the device chain never engages
    /// off the CUDA device path.
    #[test]
    fn concurrent_builds_render_the_same_audio() {
        let voice = tiny_voice();
        let images = WeightImages { f32: &voice.weights.f32_image, f16: None };
        let opts = VcSessionOptions::default();
        let levers = EffectiveLevers { precision: PrecisionMode::Fp32, ..EffectiveLevers::default() };
        let plan = streaming_plan(&voice, 640, 640 + 481, 1_024, true);
        let mut serial = SplitGenerator::load_ext(&voice.template, &images, &plan, "cpu", &opts, &levers, true, false).unwrap();
        let mut parallel = SplitGenerator::load_ext(&voice.template, &images, &plan, "cpu", &opts, &levers, true, true).unwrap();
        assert!(!serial.device_chain() && !parallel.device_chain(), "no device chain on the CPU");
        assert_eq!(serial.ep_outcomes().len(), parallel.ep_outcomes().len());
        let a = render_ticks(&mut serial, &voice, &plan, 6);
        let b = render_ticks(&mut parallel, &voice, &plan, 6);
        assert_eq!(a, b, "concurrently built plans render other audio");
        assert!(a.iter().flatten().any(|v| v.abs() > 1e-5), "near-silent render");
    }

    /// P3B-G on CUDA (dev rig): with IOBinding and graphs both plans are device-resident, so the
    /// device chain engages; it renders bit-identical audio to the host path tick after tick (the
    /// same values are copied, on the device instead of through the host), also when the plans were
    /// built concurrently.
    #[test]
    #[ignore = "needs an NVIDIA GPU, the CUDA provider next to the test executable and the NVIDIA runtime (SIGIL_APP_DATA_DIR or SIGIL_TEST_CUDA_BIN)"]
    fn device_chain_matches_the_host_path_on_cuda() {
        use super::super::options::Lever;
        crate::services::neural_audio::ort_session::preload_test_cuda_runtime_quiet();
        let voice = tiny_voice();
        let images = WeightImages { f32: &voice.weights.f32_image, f16: None };
        let opts = VcSessionOptions::default();
        let mut levers = EffectiveLevers { precision: PrecisionMode::Fp32, ..EffectiveLevers::default() };
        levers.levers.set(Lever::IoBinding, true);
        levers.levers.set(Lever::CudaGraphs, true);
        let plan = streaming_plan(&voice, 640, 640 + 481, 1_024, true);
        let mut host = SplitGenerator::load_ext(&voice.template, &images, &plan, "cuda", &opts, &levers, false, false).unwrap();
        if host.ep_outcomes()[0].1.active != "cuda" {
            eprintln!("CUDA is not usable here ({:?}); skipped", host.ep_outcomes());
            return;
        }
        let mut chained = SplitGenerator::load_ext(&voice.template, &images, &plan, "cuda", &opts, &levers, true, false).unwrap();
        let mut both = SplitGenerator::load_ext(&voice.template, &images, &plan, "cuda", &opts, &levers, true, true).unwrap();
        assert!(!host.device_chain());
        assert!(chained.device_chain() && both.device_chain(), "levers {:?}", chained.plan_levers());
        let a = render_ticks(&mut host, &voice, &plan, 12);
        let b = render_ticks(&mut chained, &voice, &plan, 12);
        let c = render_ticks(&mut both, &voice, &plan, 12);
        assert_eq!(a, b, "the device chain changed the audio");
        assert_eq!(a, c, "the concurrently built chain changed the audio");
        // The device gap between enc's end and flowdec's start (GPU timing on): the host round trip
        // (download, host copy, re-upload) is gone with the chain.
        let _recording = telemetry::start_recording();
        telemetry::set_gpu_timing(true);
        let gap = |gen: &mut SplitGenerator| -> f64 {
            let mut v: Vec<f64> = (0..16)
                .filter_map(|_| {
                    render_ticks(gen, &voice, &plan, 1);
                    gen.enc.gpu_gap_ms_to(&gen.flowdec)
                })
                .collect();
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied().unwrap_or(f64::NAN)
        };
        let (host_gap, chain_gap) = (gap(&mut host), gap(&mut chained));
        telemetry::set_gpu_timing(false);
        eprintln!("enc -> flowdec device gap (median): host path {host_gap:.3} ms, device chain {chain_gap:.3} ms");
        assert!(chain_gap.is_finite() && host_gap.is_finite(), "no gap measured");
        assert!(chain_gap < host_gap, "the chain did not shorten the gap: {chain_gap} vs {host_gap}");
    }
}
