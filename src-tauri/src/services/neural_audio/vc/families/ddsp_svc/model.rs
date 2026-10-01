//! The DDSP-SVC 6.3 voice model (P5-A): upstream `SvcDDSP.infer` / `Unit2Wav.forward` as a
//! fixed-shape tick over three session plans, behind the shared `VcStreamHarness`.
//!
//! One tick (exactly one block of new 16 kHz audio, one fixed context; ADR 0006 "DDSP-SVC"):
//! 1. units: the shared ContentVec encoder twice (the context, and the context from sample 160),
//!    interleaved (upstream's `contentvec768l12tta2x`), aligned to the model's frames (`mel2ph`);
//! 2. F0 from the voice's pitch extractor (the incremental F0 cache when that lever is on) with
//!    upstream's post-processing (`dsp::f0_frames_into`), shifted by the voice's pitch, no upper
//!    clamp but the vocoder source's Nyquist frequency;
//! 3. the volume envelope (`dsp::VolumeScratch`) and the DDSP noise excitation;
//! 4. `ddsp` → the normalised DDSP mel `x` and its log-mel `cond` over the decoded frames;
//! 5. the rectified flow from `t_start`: `x0 = t_start · x + (1 − t_start) · z`, then `steps` ×
//!    `reflow_step` (one Euler step per call; a captured CUDA graph per step shape);
//! 6. the vocoder (PC-NSF-HiFiGAN 44.1 kHz) over the decoded frames with the NSF phase carried
//!    from the previous tick (`dsp::NsfPhase`);
//! 7. upstream's volume gate, and the returned samples cut from the decoded audio.
//!
//! **Frames are anchored to absolute time** (frame `k` = model-rate samples `[k·hop, (k+1)·hop)`),
//! unlike upstream's GUI, whose frames start at each window's first sample: consecutive ticks then
//! see the same frames with the same features, the DDSP noise and the flow's start noise are drawn
//! per absolute sample / frame (`anchored_noise`), and the NSF phase continues across ticks
//! (`nsf_phase0`), so the SOLA crossfade joins two renditions that agree except near the window
//! edges. The window floats by up to one frame inside the fixed 16 kHz context to stay on the grid.
//!
//! The DDSP's F0 covers the whole window (upstream's GUI pads the context frames with the first
//! decoded frame's pitch to save an extraction; the model was trained on whole contours).

use std::path::{Path, PathBuf};
use std::sync::PoisonError;
use std::time::Instant;

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::gpu::plan::{ExternalWeights, PlanSpec, SessionPlan, TensorSpec};
use crate::services::neural_audio::gpu::GpuLevers;
use crate::services::neural_audio::ort_session::{normalize_ep, ModelSource, VcSessionOptions, EP_CPU, EP_DML};
use crate::services::neural_audio::vc::engine::content::SharedEncoder;
use crate::services::neural_audio::vc::engine::f0_cache::{F0Cache, F0CacheConfig, DEFAULT_LEAD_FRAMES, DEFAULT_MARGIN_FRAMES};
use crate::services::neural_audio::vc::engine::generator::component_key;
use crate::services::neural_audio::vc::engine::governor::QualityLevel;
use crate::services::neural_audio::vc::engine::noise::{AnchoredNoise, NoiseFrames, NoiseStream};
use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, Lever};
use crate::services::neural_audio::vc::traits::{ConversionCtx, PitchExtractor, StreamReport, VoiceControls, VoiceConversionModel, DEFAULT_SILENCE_DB};

use super::dsp::{self, F0Scratch, NsfPhase, VolumeScratch};
use super::layout::{DdspLayout, GRAPH_DDSP, GRAPH_REFLOW};

/// The model's input rate (the harness's).
pub const INPUT_RATE: u32 = 16_000;
/// 10 ms at 16 kHz: blocks and contexts are whole multiples of it.
const HOP_16K: usize = 160;
/// Decoded frames beyond the returned samples (left context of the flow and the vocoder inside the
/// decoded region; upstream's GUI keeps 60 ms beyond the crossfade).
pub const DECODE_MARGIN_FRAMES: usize = 6;
/// Fewest frames of DDSP context before the decoded region.
const MIN_LEFT_FRAMES: usize = 16;
/// Highest F0 the vocoder's sine source can carry (a little under its Nyquist frequency).
const NYQUIST_MARGIN: f64 = 0.98;
/// Formant shift range (semitones) handed to the model's pitch-augmentation embedding.
const FORMANT_RANGE: f32 = 12.0;

/// The vocoder's runtime facts (its `config.json`).
#[derive(Clone, Debug, PartialEq)]
pub struct VocoderRuntime {
    pub sampling_rate: u32,
    pub hop: usize,
    pub num_mels: usize,
    /// NSF source samples per frame and the source's rate (`mini_nsf`: the first two upsampling
    /// stages; else the whole hop at the output rate).
    pub upp: usize,
    pub source_sr: f64,
}

impl VocoderRuntime {
    /// Read and check a PC-NSF-HiFiGAN `config.json` against the template's vocoder settings.
    pub fn from_config(path: &Path, layout: &DdspLayout) -> Result<VocoderRuntime, String> {
        let meta = std::fs::metadata(path).map_err(|e| format!("vocoder config {}: {e}", path.display()))?;
        if meta.len() > 1 << 20 {
            return Err(format!("vocoder config {} is too large", path.display()));
        }
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?)
            .map_err(|e| format!("vocoder config {}: {e}", path.display()))?;
        let num = |k: &str| v.get(k).and_then(serde_json::Value::as_f64).ok_or_else(|| format!("vocoder config: {k} missing"));
        let rates: Vec<usize> = v
            .get("upsample_rates")
            .and_then(serde_json::Value::as_array)
            .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
            .unwrap_or_default();
        let sampling_rate = num("sampling_rate")? as u32;
        let hop = num("hop_size")? as usize;
        let num_mels = num("num_mels")? as usize;
        if rates.is_empty() || rates.iter().product::<usize>() != hop || rates.contains(&0) {
            return Err(format!("vocoder config: upsample_rates {rates:?} do not make the hop {hop}"));
        }
        let want = &layout.vocoder;
        let same = |a: f64, b: f64| (a - b).abs() < 1e-6;
        if sampling_rate != want.sampling_rate
            || hop != want.hop_size as usize
            || num_mels != want.num_mels as usize
            || !same(num("n_fft")?, f64::from(want.n_fft))
            || !same(num("win_size")?, f64::from(want.win_size))
            || !same(num("fmin")?, want.fmin)
            || !same(num("fmax")?, want.fmax)
        {
            return Err(format!(
                "the installed vocoder ({sampling_rate} Hz, hop {hop}, {num_mels} mels) is not the one the DDSP-SVC template was built for ({} Hz, hop {}, {} mels, n_fft {}, {}-{} Hz)",
                want.sampling_rate, want.hop_size, want.num_mels, want.n_fft, want.fmin, want.fmax
            ));
        }
        let mini = v.get("mini_nsf").and_then(serde_json::Value::as_bool).unwrap_or(false);
        let (upp, source_sr) = if mini && rates.len() > 2 {
            let upp = rates[..2].iter().product::<usize>();
            (upp, f64::from(sampling_rate) / rates[2..].iter().product::<usize>() as f64)
        } else {
            (hop, f64::from(sampling_rate))
        };
        Ok(VocoderRuntime { sampling_rate, hop, num_mels, upp, source_sr })
    }
}

/// Where the content units come from (the shared encoder pool in the app; a stand-in in tests).
pub trait UnitsSource: Send {
    /// Feature channels per frame.
    fn channels(&self) -> usize;
    /// Encode `mono_16k` into `out` (`frames × channels`); returns the frame count.
    fn encode(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<usize, String>;
}

/// The pooled ContentVec encoder.
pub struct PooledUnits(pub SharedEncoder);

impl UnitsSource for PooledUnits {
    fn channels(&self) -> usize {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).channels()
    }

    fn encode(&mut self, mono_16k: &[f32], out: &mut Vec<f32>) -> Result<usize, String> {
        self.0.lock().unwrap_or_else(PoisonError::into_inner).encode(mono_16k, out)
    }
}

/// The graph files of a voice.
#[derive(Clone, Debug)]
pub struct DdspFiles {
    pub ddsp: PathBuf,
    pub reflow: PathBuf,
    pub vocoder: PathBuf,
}

/// What a voice was imported with (its record) that the tick needs.
#[derive(Clone, Debug)]
pub struct VoiceSetup {
    /// `contentvec768l12tta2x` units (one per 10 ms) or plain `contentvec768l12` (one per 20 ms).
    pub tta2x: bool,
    pub f0_min: f32,
    /// `Volume_Extractor` window (model-rate samples).
    pub volume_window: usize,
    /// Speaker table `[count × dim]` of a multi-speaker voice.
    pub speakers: Option<(Vec<f32>, usize)>,
}

/// Stream shapes of the tick (and how the user asked for them).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tuning {
    pub block_16k: usize,
    /// Extra context before the block (16 kHz samples; at least the preset's).
    pub extra_16k: usize,
    pub lookahead_16k: usize,
    pub steps: u32,
    pub t_start: f32,
}

/// Fixed shapes of one stream configuration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub block_16k: usize,
    pub want_16k: usize,
    pub context_16k: usize,
    /// Window frames (`T`), the left-context frames before the decoded ones, and the decoded
    /// frames (`R = T − start`).
    pub frames: usize,
    pub start: usize,
    pub reflow: usize,
    /// Unit frames of the two encoder passes and in total (`U`).
    pub units_a: usize,
    pub units_b: usize,
    pub units: usize,
    /// Model-rate samples returned per call and the lookahead they end before the context's end.
    pub out: usize,
    pub lookahead: usize,
}

impl Shape {
    /// The shapes for a block, the harness's returned length and extra context (16 kHz samples).
    pub fn plan(layout: &DdspLayout, setup: &VoiceSetup, tuning: &Tuning, block_16k: usize, want_16k: usize, extra_16k: usize) -> Result<Shape, String> {
        let sr = u64::from(layout.arch.sampling_rate);
        let hop = layout.hop();
        if !sr.is_multiple_of(100) {
            return Err(format!("DDSP-SVC needs a sampling rate that is a whole number of samples per 10 ms ({sr} Hz)"));
        }
        if block_16k == 0 || !block_16k.is_multiple_of(HOP_16K) {
            return Err(format!("DDSP-SVC blocks are whole 10 ms ({block_16k} samples at 16 kHz)"));
        }
        let to_out = |n16: usize| -> usize { ((n16 as u64 * sr + 8_000) / 16_000) as usize };
        let out = to_out(want_16k.max(block_16k));
        let lookahead = to_out(tuning.lookahead_16k.next_multiple_of(HOP_16K));
        let reflow = (out + lookahead + DECODE_MARGIN_FRAMES * hop).div_ceil(hop) + 1;
        let extra = extra_16k.max(tuning.extra_16k);
        let frames = ((to_out(extra + block_16k) + lookahead).div_ceil(hop) + 1).max(reflow + MIN_LEFT_FRAMES);
        let start = frames - reflow;
        // The context covers the window's frames, their volume windows and the frame alignment.
        let need_out = frames * hop + setup.volume_window / 2 + hop;
        let context_16k = ((need_out as u64 * 16_000).div_ceil(sr) as usize)
            .max(want_16k + tuning.lookahead_16k)
            .max(extra + block_16k)
            .next_multiple_of(HOP_16K);
        let (units_a, units_b) = if setup.tta2x {
            dsp::tta2x_counts(context_16k)
        } else {
            (dsp::tta2x_counts(context_16k).0, 0)
        };
        if units_a == 0 {
            return Err("DDSP-SVC: the context is too short for the content encoder".into());
        }
        Ok(Shape {
            block_16k,
            want_16k,
            context_16k,
            frames,
            start,
            reflow,
            units_a,
            units_b,
            units: units_a + units_b,
            out,
            lookahead,
        })
    }
}

/// The three session plans of a shape.
struct Graphs {
    shape: Shape,
    ddsp: SessionPlan,
    reflow: SessionPlan,
    vocoder: SessionPlan,
}

/// Per-stage compute of the last tick (ms), for tests and the benchmark.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TickStats {
    pub units_ms: f64,
    pub f0_ms: f64,
    pub ddsp_ms: f64,
    pub flow_ms: f64,
    pub vocoder_ms: f64,
    pub total_ms: f64,
    pub skipped: bool,
}

/// Everything a model is built from (owned).
pub struct ModelParts {
    pub layout: DdspLayout,
    pub files: DdspFiles,
    pub vocoder: VocoderRuntime,
    pub weights: PathBuf,
    pub setup: VoiceSetup,
    pub tuning: Tuning,
    pub units: Box<dyn UnitsSource>,
    pub pitch: Box<dyn PitchExtractor>,
    pub ep: String,
    pub session: VcSessionOptions,
    pub levers: EffectiveLevers,
    pub noise_seed: u64,
    /// Skip the graphs on a tick whose returned samples the volume gate closes entirely (the
    /// voice changer's silence-skip lever: it saves compute and changes nothing heard, since the
    /// gated tick would return zeros there too). The gate itself is the model's (upstream's
    /// `SvcDDSP.infer` always applies it) and runs either way.
    pub silence_skip: bool,
    pub outcomes: Vec<(String, EpOutcome)>,
    pub warnings: Vec<String>,
}

/// Explicit inputs of one tick of the graphs (the parity tests feed upstream's draws).
pub struct TickInputs<'a> {
    pub units: &'a [f32],
    pub mel2ph: &'a [i64],
    pub f0: &'a [f32],
    pub volume: &'a [f32],
    pub randn: &'a [f32],
    pub spk: &'a [f32],
    pub formant: f32,
    pub z: &'a [f32],
    pub phase0: f32,
}

/// A loaded DDSP-SVC voice.
pub struct DdspSvcModel {
    layout: DdspLayout,
    files: DdspFiles,
    vocoder_rt: VocoderRuntime,
    weights: PathBuf,
    setup: VoiceSetup,
    tuning: Tuning,
    units: Box<dyn UnitsSource>,
    pitch: Box<dyn PitchExtractor>,
    ep: String,
    session: VcSessionOptions,
    requested: EffectiveLevers,
    levers: EffectiveLevers,
    outcomes: Vec<(String, EpOutcome)>,
    warnings: Vec<String>,
    graphs: Option<Graphs>,
    controls: VoiceControls,
    silence_skip: bool,
    /// Flow steps in use (the governor may lower them).
    steps: u32,
    // ---- per-stream state ----
    f0_cache: F0Cache,
    nsf: NsfPhase,
    noise: AnchoredNoise,
    sample_noise: NoiseFrames,
    flow_noise: NoiseFrames,
    fresh_ticks: u64,
    stats: TickStats,
    // ---- per-tick buffers (sized at configure; no allocation per tick afterwards) ----
    enc_a: Vec<f32>,
    enc_b: Vec<f32>,
    units_buf: Vec<f32>,
    mel2ph: Vec<i64>,
    pos_10ms: Vec<f64>,
    f0: Vec<f32>,
    f0_scratch: F0Scratch,
    volume: Vec<f32>,
    vol_scratch: VolumeScratch,
    randn: Vec<f32>,
    z: Vec<f32>,
    spk: Vec<f32>,
    xs: Vec<f32>,
    mask: Vec<f32>,
    audio: Vec<f32>,
}

#[allow(clippy::too_many_arguments)]
fn plan_spec<'a>(
    component: &'a str,
    ep: &'a str,
    opts: &'a VcSessionOptions,
    path: &'a Path,
    weights: Option<ExternalWeights<'a>>,
    dims: &'a [(&'a str, i64)],
    inputs: &'a [TensorSpec],
    outputs: &'a [TensorSpec],
    levers: GpuLevers,
) -> PlanSpec<'a> {
    PlanSpec { component, ep, opts, model: ModelSource::File(path), weights, dims, inputs, outputs, levers }
}

/// Build a plan; on DirectML a graph whose pinned shapes DirectML's fusion refuses (`ddsp`) runs
/// with dynamic shapes, and a plan that fell to the CPU from a GPU request with pinned shapes is
/// rebuilt once without them (ADR 0006 finding 5). The second case covers a CUDA request whose
/// ladder fell to DirectML (no NVIDIA runtime, an old driver): DirectML then refuses `ddsp`'s
/// pinned shapes and the graph would land on the CPU, 30 ms instead of 4 ms per call.
fn build_plan(spec: &PlanSpec<'_>, pin_on_dml: bool) -> Result<SessionPlan, String> {
    let requested = normalize_ep(spec.ep);
    if requested == EP_DML && !pin_on_dml {
        return SessionPlan::build(&PlanSpec { dims: &[], ..*spec });
    }
    let plan = SessionPlan::build(spec)?;
    if requested == EP_CPU || spec.dims.is_empty() || plan.outcome().active != EP_CPU {
        return Ok(plan);
    }
    tracing::info!(target: "sigil::vc", "{}: no GPU took the pinned shapes; rebuilding with dynamic shapes", spec.component);
    drop(plan);
    SessionPlan::build(&PlanSpec { dims: &[], ..*spec })
}

impl DdspSvcModel {
    pub fn new(parts: ModelParts) -> Result<DdspSvcModel, String> {
        let layout = parts.layout;
        if parts.units.channels() != layout.arch.encoder_out_channels as usize {
            return Err(format!(
                "DDSP-SVC: the content encoder gives {} channels, the voice needs {}",
                parts.units.channels(),
                layout.arch.encoder_out_channels
            ));
        }
        if parts.vocoder.hop != layout.hop() || parts.vocoder.num_mels != layout.mels() {
            return Err("DDSP-SVC: the vocoder does not fit the template".into());
        }
        if let Some((rows, dim)) = &parts.setup.speakers {
            if *dim != layout.speakers.dim || rows.is_empty() || rows.len() % dim != 0 {
                return Err("DDSP-SVC: the voice's speaker table does not fit the template".into());
            }
        }
        let noise = AnchoredNoise::new(parts.noise_seed);
        let hop = layout.hop();
        let mels = layout.mels();
        let (left, right) = parts.pitch.receptive_frames().unwrap_or((DEFAULT_LEAD_FRAMES, DEFAULT_MARGIN_FRAMES));
        let f0_cache = F0Cache::new(F0CacheConfig { margin_frames: right.max(1), capacity_frames: 4096 }).with_lead_frames(left);
        let steps = parts.tuning.steps.max(1);
        Ok(DdspSvcModel {
            nsf: NsfPhase::new(parts.vocoder.upp, parts.vocoder.source_sr),
            sample_noise: NoiseFrames::new(noise, NoiseStream::Source, hop, 8),
            flow_noise: NoiseFrames::new(noise, NoiseStream::Flow, mels, 8),
            noise,
            layout,
            files: parts.files,
            vocoder_rt: parts.vocoder,
            weights: parts.weights,
            setup: parts.setup,
            tuning: parts.tuning,
            units: parts.units,
            pitch: parts.pitch,
            ep: parts.ep,
            session: parts.session,
            requested: parts.levers.clone(),
            levers: parts.levers,
            outcomes: parts.outcomes,
            warnings: parts.warnings,
            graphs: None,
            controls: VoiceControls::default(),
            silence_skip: parts.silence_skip,
            steps,
            f0_cache,
            fresh_ticks: 0,
            stats: TickStats::default(),
            enc_a: Vec::new(),
            enc_b: Vec::new(),
            units_buf: Vec::new(),
            mel2ph: Vec::new(),
            pos_10ms: Vec::new(),
            f0: Vec::new(),
            f0_scratch: F0Scratch::default(),
            volume: Vec::new(),
            vol_scratch: VolumeScratch::default(),
            randn: Vec::new(),
            z: Vec::new(),
            spk: Vec::new(),
            xs: Vec::new(),
            mask: Vec::new(),
            audio: Vec::new(),
        })
    }

    /// The configured stream shape (tests and the cost figures).
    #[cfg(test)]
    pub fn shape(&self) -> Option<Shape> {
        self.graphs.as_ref().map(|g| g.shape)
    }

    /// Per-stage compute of the last tick (tests and the cost figures).
    #[cfg(test)]
    pub fn tick_stats(&self) -> TickStats {
        self.stats
    }

    /// The absolute first decoded frame and the F0 (Hz) the vocoder received for the decoded frames
    /// of the last tick that ran the graphs (tests: the pitch path of the T10 figures).
    #[cfg(test)]
    pub fn last_decoded_f0(&self) -> Option<(i64, &[f32])> {
        self.nsf.last_decoded()
    }

    pub fn levers(&self) -> &EffectiveLevers {
        &self.levers
    }

    pub fn outcomes(&self) -> &[(String, EpOutcome)] {
        &self.outcomes
    }

    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }

    /// Steps of the flow in use (the governor may lower them).
    #[cfg(test)]
    pub fn steps(&self) -> u32 {
        self.steps
    }

    /// Replace the requested levers (tests of the legacy lever paths; the next plan uses them).
    #[cfg(test)]
    pub(super) fn set_levers_for_test(&mut self, levers: EffectiveLevers) {
        self.requested = levers.clone();
        self.levers = levers;
    }

    /// The last `ddsp` outputs (`x`, `cond`) and the flow's result (normalised mel), for tests.
    #[cfg(test)]
    pub(super) fn last_graph_outputs(&self) -> Option<(&[f32], &[f32], &[f32])> {
        let g = self.graphs.as_ref()?;
        Some((g.ddsp.output("x").ok()?, g.ddsp.output("cond").ok()?, &self.xs))
    }

    fn gate_threshold(&self) -> f32 {
        let db = self.controls.silence_db;
        if db.is_finite() {
            db.clamp(-90.0, 0.0)
        } else {
            DEFAULT_SILENCE_DB
        }
    }

    /// Build (or rebuild) the plans for `shape` and size the tick buffers.
    pub(super) fn build_graphs(&mut self, shape: Shape) -> Result<(), String> {
        // Rebuilds free the old sessions' memory first.
        self.graphs = None;
        let l = &self.layout;
        let (hop, mels) = (l.hop(), l.mels());
        let (c, d) = (l.arch.encoder_out_channels as usize, l.speakers.dim);
        let (t, r, u) = (shape.frames, shape.reflow, shape.units);
        let image = std::fs::read(&self.weights).map_err(|e| format!("{}: {e}", self.weights.display()))?;
        if image.len() as u64 != l.weights.bytes {
            return Err(format!(
                "{}: {} bytes, the DDSP-SVC template needs {} (import the voice again)",
                self.weights.display(),
                image.len(),
                l.weights.bytes
            ));
        }
        let section = |g: &str| -> Result<(&str, &[u8]), String> {
            let s = l.section(g).ok_or_else(|| format!("layout: no section {g}"))?;
            Ok((s.location.as_str(), &image[s.offset as usize..(s.offset + s.bytes) as usize]))
        };
        let gl = GpuLevers::from_effective(&self.requested);
        let opts = &self.session;
        let ep = self.ep.as_str();

        let (loc, img) = section(GRAPH_DDSP)?;
        let d_in = [
            TensorSpec::f32("hubert", &[1, u, c]),
            TensorSpec::i64("mel2ph", &[1, t]),
            TensorSpec::f32("f0", &[1, t]),
            TensorSpec::f32("volume", &[1, t]),
            TensorSpec::f32("randn", &[1, hop * t]),
            TensorSpec::f32("spk", &[1, d]),
            TensorSpec::f32("formant_shift", &[1, 1]),
            TensorSpec::i64("start_frame", &[1]),
        ];
        let d_out = [TensorSpec::f32("x", &[1, 1, mels, r]), TensorSpec::f32("cond", &[1, mels, r])];
        let d_dims = [("U", u as i64), ("T", t as i64), ("S", (hop * t) as i64)];
        let ddsp = build_plan(
            &plan_spec("vc.ddsp", ep, opts, &self.files.ddsp, Some(ExternalWeights { file: loc, image: img }), &d_dims, &d_in, &d_out, gl),
            false,
        )?;

        let (loc, img) = section(GRAPH_REFLOW)?;
        let f_in = [
            TensorSpec::f32("x", &[1, 1, mels, r]),
            TensorSpec::f32("t", &[1]),
            TensorSpec::f32("dt", &[1]),
            TensorSpec::f32("cond", &[1, mels, r]),
        ];
        let f_out = [TensorSpec::f32("x_next", &[1, 1, mels, r])];
        let f_dims = [("R", r as i64)];
        let reflow = build_plan(
            &plan_spec("vc.reflow", ep, opts, &self.files.reflow, Some(ExternalWeights { file: loc, image: img }), &f_dims, &f_in, &f_out, gl),
            true,
        )?;
        drop(image);

        let v_in = [TensorSpec::f32("mel", &[1, mels, r]), TensorSpec::f32("f0", &[1, r]), TensorSpec::f32("phase0", &[1])];
        let v_out = [TensorSpec::f32("audio", &[1, hop * r])];
        let v_dims = [("T", r as i64)];
        let vocoder = build_plan(&plan_spec("vc.vocoder", ep, opts, &self.files.vocoder, None, &v_dims, &v_in, &v_out, gl), true)?;

        // What runs: EP outcomes and the GPU levers every plan really uses.
        let plans = [("vc.ddsp", &ddsp), ("vc.reflow", &reflow), ("vc.vocoder", &vocoder)];
        self.outcomes.retain(|(k, _)| !plans.iter().any(|(c, _)| *k == component_key(opts, c)));
        let mut levers = self.requested.clone();
        let mut notes: Vec<String> = Vec::new();
        for (component, plan) in &plans {
            self.outcomes.push((component_key(opts, component), plan.outcome().clone()));
            for n in plan.notes() {
                notes.push(format!("{component}: {n}"));
            }
        }
        let io = plans.iter().all(|(_, p)| p.levers().io_binding);
        let graphs_on = reflow.levers().cuda_graphs && vocoder.levers().cuda_graphs;
        let prio = plans.iter().all(|(_, p)| p.levers().priority);
        if levers.levers.get(Lever::IoBinding) && !io {
            levers.levers.set(Lever::IoBinding, false);
        }
        if levers.levers.get(Lever::CudaGraphs) && !graphs_on {
            levers.levers.set(Lever::CudaGraphs, false);
        }
        if levers.levers.get(Lever::GpuPriority) && !prio {
            levers.levers.set(Lever::GpuPriority, false);
        }
        if levers.levers.get(Lever::GpuKnn) {
            levers.levers.set(Lever::GpuKnn, false);
            notes.push("gpu_knn: DDSP-SVC voices have no retrieval index".into());
        }
        levers.notes.extend(notes);
        self.levers = levers;

        // Buffers at their final sizes.
        self.units_buf.reserve(u * c);
        self.mel2ph.resize(t, 0);
        self.pos_10ms.resize(t, 0.0);
        self.f0.resize(t, 0.0);
        self.volume.resize(t, 0.0);
        self.randn.resize(hop * t, 0.0);
        self.z.resize(mels * r, 0.0);
        self.spk.resize(d, 0.0);
        self.xs.resize(mels * r, 0.0);
        self.mask.resize(hop * r, 0.0);
        self.audio.resize(hop * r, 0.0);
        self.sample_noise = NoiseFrames::new(self.noise, NoiseStream::Source, hop, t + 64);
        self.flow_noise = NoiseFrames::new(self.noise, NoiseStream::Flow, mels, t + 64);
        self.f0_cache = F0Cache::new(F0CacheConfig {
            margin_frames: self.f0_cache.config().margin_frames,
            capacity_frames: 2 * shape.context_16k / HOP_16K + 16,
        })
        .with_lead_frames(self.f0_cache.lead_frames());
        self.nsf.reset();
        self.graphs = Some(Graphs { shape, ddsp, reflow, vocoder });
        tracing::info!(
            target: "sigil::vc",
            "DDSP-SVC voice: {} ms blocks, {} ms of context, {} frames ({} decoded), {} flow steps from t = {}",
            shape.block_16k / 16,
            shape.context_16k / 16,
            shape.frames,
            shape.reflow,
            self.steps,
            self.tuning.t_start
        );
        Ok(())
    }

    /// Plan (or re-plan) the stream: a block, the samples each call returns and the extra context
    /// (16 kHz samples); returns the context length.
    pub fn configure(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        let shape = Shape::plan(&self.layout, &self.setup, &self.tuning, block, want, extra)?;
        if self.graphs.as_ref().map(|g| g.shape) != Some(shape) {
            self.build_graphs(shape)?;
        }
        Ok(shape.context_16k)
    }

    /// Run the graphs on explicit inputs (the window's frames; `out` receives the decoded audio,
    /// `hop × R` samples, before the gate). The tick and the parity tests share it.
    pub fn run_graphs(&mut self, inputs: &TickInputs<'_>, steps: u32, t_start: f32, out: &mut Vec<f32>) -> Result<(), String> {
        let g = self.graphs.as_mut().ok_or("DDSP-SVC: the stream is not configured")?;
        let sh = g.shape;
        let (spec_min, spec_max) = (self.layout.spec_min, self.layout.spec_max);
        let r = sh.reflow;
        let t0 = Instant::now();
        {
            let p = &mut g.ddsp;
            copy_into(p.input_f32("hubert")?, inputs.units, "hubert")?;
            copy_into(p.input_i64("mel2ph")?, inputs.mel2ph, "mel2ph")?;
            copy_into(p.input_f32("f0")?, inputs.f0, "f0")?;
            copy_into(p.input_f32("volume")?, inputs.volume, "volume")?;
            copy_into(p.input_f32("randn")?, inputs.randn, "randn")?;
            copy_into(p.input_f32("spk")?, inputs.spk, "spk")?;
            p.input_f32("formant_shift")?[0] = inputs.formant;
            p.input_i64("start_frame")?[0] = sh.start as i64;
            p.run().map_err(|e| format!("ddsp: {e}"))?;
        }
        let t1 = Instant::now();
        // Flow: x0 = t_start · x + (1 − t_start) · z, then Euler steps (float32 like upstream).
        let ts = t_start;
        let one_minus = (1.0f64 - f64::from(t_start)) as f32;
        {
            let x = g.ddsp.output("x")?;
            if x.len() != self.xs.len() || inputs.z.len() != x.len() {
                return Err(format!("flow: {} / {} values for {} (128 x R)", x.len(), inputs.z.len(), self.xs.len()));
            }
            for ((d, &a), &b) in self.xs.iter_mut().zip(x).zip(inputs.z) {
                *d = ts * a + one_minus * b;
            }
            let cond = g.ddsp.output("cond")?;
            copy_into(g.reflow.input_f32("cond")?, cond, "cond")?;
        }
        // `t_start` 1 is the DDSP mel itself (upstream skips the flow).
        let steps = if t_start >= 1.0 { 0 } else { steps.max(1) };
        let dt = ((1.0f64 - f64::from(t_start)) / f64::from(steps.max(1))) as f32;
        let mut t = t_start;
        for _ in 0..steps {
            g.reflow.input_f32("x")?.copy_from_slice(&self.xs);
            g.reflow.input_f32("t")?[0] = t;
            g.reflow.input_f32("dt")?[0] = dt;
            g.reflow.run().map_err(|e| format!("reflow_step: {e}"))?;
            self.xs.copy_from_slice(g.reflow.output("x_next")?);
            t += dt;
        }
        let t2 = Instant::now();
        {
            let mel = g.vocoder.input_f32("mel")?;
            for (m, &x) in mel.iter_mut().zip(&self.xs) {
                *m = (x + 1.0) / 2.0 * (spec_max - spec_min) + spec_min;
            }
            copy_into(g.vocoder.input_f32("f0")?, inputs.f0.get(sh.start..).unwrap_or_default(), "f0")?;
            g.vocoder.input_f32("phase0")?[0] = inputs.phase0;
            g.vocoder.run().map_err(|e| format!("vocoder: {e}"))?;
            out.clear();
            out.extend_from_slice(g.vocoder.output("audio")?);
        }
        let t3 = Instant::now();
        debug_assert_eq!(out.len(), self.layout.hop() * r);
        self.stats.ddsp_ms = (t1 - t0).as_secs_f64() * 1e3;
        self.stats.flow_ms = (t2 - t1).as_secs_f64() * 1e3;
        self.stats.vocoder_ms = (t3 - t2).as_secs_f64() * 1e3;
        Ok(())
    }

    /// One tick (module docs). `force` runs the models on silence too (warm-up).
    fn tick(&mut self, ctx: &ConversionCtx<'_>, force: bool) -> Result<Vec<f32>, String> {
        let started = Instant::now();
        let len = ctx.mono_16k.len();
        // A call at another shape than the configured one (a caller without the fixed-block
        // harness): configure for it.
        let configured = self.graphs.as_ref().map(|g| g.shape);
        let shape = match configured {
            Some(s) if s.context_16k == len && s.want_16k == ctx.new_frames => s,
            _ => {
                let block = configured.map(|s| s.block_16k).unwrap_or(self.tuning.block_16k);
                let extra = len.saturating_sub(block);
                self.configure(block, ctx.new_frames, extra)?;
                let s = self.graphs.as_ref().map(|g| g.shape).ok_or("DDSP-SVC: no shape")?;
                if s.context_16k != len {
                    return Err(format!("DDSP-SVC: a {len}-sample context does not fit the voice's shape ({} samples)", s.context_16k));
                }
                s
            }
        };
        let l = &self.layout;
        let sr = i64::from(l.arch.sampling_rate);
        let hop = l.hop();
        let hop_i = hop as i64;
        let to_out = |n16: i64| -> i64 { (n16 * sr + 8_000).div_euclid(16_000) };
        let p = ctx.stream_pos_16k as i64;
        let p_out = to_out(p);
        let e_out = to_out(p + len as i64);
        let k_last = (e_out + hop_i - 1).div_euclid(hop_i) - 1;
        let k0 = k_last + 1 - shape.frames as i64;
        let first = k0 + shape.start as i64;
        let region = e_out - shape.lookahead as i64 - shape.out as i64;
        let off = region - first * hop_i;
        if off < 0 || off as usize + shape.out > shape.reflow * hop {
            return Err(format!("DDSP-SVC: the returned samples ({off}..) leave the decoded frames"));
        }
        // 1. Volume (and the silence skip: a tick whose returned samples the gate closes returns
        //    the zeros the gated tick would, without running the models).
        self.vol_scratch
            .envelope_into(ctx.mono_16k, p_out, k0, hop, self.setup.volume_window, l.arch.sampling_rate, shape.frames, &mut self.volume);
        let thr = self.gate_threshold();
        let first_out = ((region - k0 * hop_i) / hop_i) as usize;
        let last_out = ((region + shape.out as i64 - 1 - k0 * hop_i) / hop_i) as usize;
        if self.silence_skip && !force && !dsp::gate_open(&self.volume, thr, first_out, last_out) {
            self.nsf.reset();
            self.stats = TickStats { total_ms: started.elapsed().as_secs_f64() * 1e3, skipped: true, ..TickStats::default() };
            return Ok(vec![0.0; shape.out]);
        }
        // 2. Units.
        let t_units = Instant::now();
        let c = self.units.channels();
        let na = self.units.encode(ctx.mono_16k, &mut self.enc_a)?;
        if na != shape.units_a {
            return Err(format!("content encoder: {na} frames, the shape expects {}", shape.units_a));
        }
        let units = if self.setup.tta2x {
            let nb = self.units.encode(&ctx.mono_16k[HOP_16K..], &mut self.enc_b)?;
            if nb < shape.units_b {
                return Err(format!("content encoder (shifted): {nb} frames, the shape expects {}", shape.units_b));
            }
            dsp::tta2x_interleave(&self.enc_a, na, &self.enc_b, shape.units_b, c, &mut self.units_buf)?
        } else {
            self.units_buf.clear();
            self.units_buf.extend_from_slice(&self.enc_a[..na * c]);
            na
        };
        let unit_period = if self.setup.tta2x { 0.01 } else { 0.02 };
        dsp::mel2ph_into(k0, shape.frames, hop, p_out, l.arch.sampling_rate, unit_period, units, &mut self.mel2ph);
        // 3. F0.
        let t_f0 = Instant::now();
        let f0_10 = if self.levers.levers.get(Lever::F0Cache) {
            self.f0_cache.update(self.pitch.as_mut(), ctx.mono_16k, ctx.stream_pos_16k)?
        } else {
            self.f0_cache.extract_window(self.pitch.as_mut(), ctx.mono_16k)?
        };
        let per_10ms = (sr / 100) as f64;
        for (i, pos) in self.pos_10ms.iter_mut().enumerate() {
            *pos = ((k0 + i as i64) * hop_i - p_out) as f64 / per_10ms;
        }
        let max_hz = (self.vocoder_rt.source_sr / 2.0 * NYQUIST_MARGIN) as f32;
        dsp::f0_frames_into(f0_10, &self.pos_10ms, self.setup.f0_min, ctx.pitch_shift_semitones, max_hz, &mut self.f0_scratch, &mut self.f0);
        let f0_ms = t_f0.elapsed().as_secs_f64() * 1e3;
        let units_ms = (t_f0 - t_units).as_secs_f64() * 1e3;
        // 4. Noise, speaker, formant.
        let anchored = self.levers.levers.get(Lever::AnchoredNoise) && k0 >= 0;
        if anchored {
            self.sample_noise.copy_window(k0 as u64, shape.frames, 1.0, false, &mut self.randn);
            self.flow_noise.copy_window(first as u64, shape.reflow, 1.0, true, &mut self.z);
        } else {
            // The legacy draw: fresh noise every tick (a stream of its own per tick).
            self.fresh_ticks += 1;
            let base = self.fresh_ticks << 40;
            self.noise.fill(NoiseStream::Source, base, 1.0, &mut self.randn);
            self.noise.fill(NoiseStream::Flow, base, 1.0, &mut self.z);
        }
        match &self.setup.speakers {
            Some((rows, dim)) => {
                let count = rows.len() / dim;
                let i = (ctx.speaker_id.max(0) as usize).min(count - 1);
                self.spk.copy_from_slice(&rows[i * dim..(i + 1) * dim]);
            }
            None => self.spk.iter_mut().for_each(|v| *v = 0.0),
        }
        let formant = if self.controls.formant_semitones.is_finite() {
            self.controls.formant_semitones.clamp(-FORMANT_RANGE, FORMANT_RANGE)
        } else {
            0.0
        };
        let phase0 = if self.levers.levers.get(Lever::NsfPhase0) { self.nsf.phase_at(first) } else { 0.0 };
        // 5-6. Graphs.
        let mut audio = std::mem::take(&mut self.audio);
        let (randn, z, spk, f0, volume, mel2ph, units_buf) = (
            std::mem::take(&mut self.randn),
            std::mem::take(&mut self.z),
            std::mem::take(&mut self.spk),
            std::mem::take(&mut self.f0),
            std::mem::take(&mut self.volume),
            std::mem::take(&mut self.mel2ph),
            std::mem::take(&mut self.units_buf),
        );
        let ran = self.run_graphs(
            &TickInputs {
                units: &units_buf,
                mel2ph: &mel2ph,
                f0: &f0,
                volume: &volume,
                randn: &randn,
                spk: &spk,
                formant,
                z: &z,
                phase0: phase0 as f32,
            },
            self.steps,
            self.tuning.t_start,
            &mut audio,
        );
        if ran.is_ok() {
            self.nsf.commit(first, phase0, &f0[shape.start..]);
        }
        // 7. Upstream's volume gate (part of the model's output, whatever the silence skip) and the
        //    returned samples.
        let result = ran.map(|()| {
            dsp::gate_mask_into(&volume, thr, shape.reflow, hop, &mut self.mask);
            for (a, m) in audio.iter_mut().zip(&self.mask) {
                *a *= m;
            }
            audio[off as usize..off as usize + shape.out].to_vec()
        });
        (self.randn, self.z, self.spk, self.f0, self.volume, self.mel2ph, self.units_buf, self.audio) =
            (randn, z, spk, f0, volume, mel2ph, units_buf, audio);
        let (ddsp_ms, flow_ms, vocoder_ms) = (self.stats.ddsp_ms, self.stats.flow_ms, self.stats.vocoder_ms);
        self.stats = TickStats {
            units_ms,
            f0_ms,
            ddsp_ms,
            flow_ms,
            vocoder_ms,
            total_ms: started.elapsed().as_secs_f64() * 1e3,
            skipped: false,
        };
        result
    }
}

fn copy_into<T: Copy>(dst: &mut [T], src: &[T], what: &str) -> Result<(), String> {
    if dst.len() != src.len() {
        return Err(format!("{what}: {} values, the plan takes {}", src.len(), dst.len()));
    }
    dst.copy_from_slice(src);
    Ok(())
}

impl VoiceConversionModel for DdspSvcModel {
    fn convert(&mut self, ctx: &ConversionCtx<'_>) -> Result<Vec<f32>, String> {
        self.tick(ctx, false)
    }

    fn input_sample_rate(&self) -> u32 {
        INPUT_RATE
    }

    fn output_sample_rate(&self) -> u32 {
        self.layout.arch.sampling_rate
    }

    fn min_context_samples(&self) -> usize {
        self.graphs.as_ref().map(|g| g.shape.context_16k).unwrap_or(0)
    }

    fn warmup_at(&mut self, ctx: &ConversionCtx<'_>) -> Result<(), String> {
        let r = self.tick(ctx, true).map(|_| ());
        // The warm-up's phase and F0 belong to no stream.
        self.reset_stream();
        r
    }

    fn set_voice_controls(&mut self, controls: &VoiceControls) {
        self.controls = *controls;
    }

    fn apply_quality(&mut self, level: &QualityLevel) -> bool {
        // The governor's context step maps onto the flow's step count (the costliest knob a
        // DDSP-SVC voice has): half the steps at reduced context, at least 2.
        let full = self.tuning.steps.max(1);
        let want = if level.context_scale < 1.0 { (full / 2).max(2).min(full) } else { full };
        if want != self.steps {
            self.steps = want;
            return true;
        }
        false
    }

    fn reset_stream(&mut self) {
        self.nsf.reset();
        self.f0_cache.reset();
    }

    fn fixed_block(&self) -> Option<usize> {
        Some(self.graphs.as_ref().map(|g| g.shape.block_16k).unwrap_or(self.tuning.block_16k))
    }

    fn configure_stream(&mut self, block: usize, want: usize, extra: usize) -> Result<usize, String> {
        self.configure(block, want, extra)
    }

    fn stream_report(&self) -> Option<StreamReport> {
        Some(StreamReport { outcomes: self.outcomes.clone(), levers: self.levers.clone(), warnings: self.warnings.clone() })
    }
}
