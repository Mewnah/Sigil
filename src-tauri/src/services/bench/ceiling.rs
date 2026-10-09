//! The offline ceiling (plan "Benchmark harness": the same model converting the whole utterance
//! in-process), against which streaming fidelity is scored (T5).
//!
//! The streaming run records the exact 16 kHz stream its model received ([`StreamRecorder`]: every
//! `ConversionCtx` at its absolute `stream_pos_16k`, so the priming silence and the resampler's
//! delay are included), and the ceiling converts that same stream at the same absolute positions:
//! - **Dynamic-shape models** (bring-your-own ONNX `RvcModel`, the stub): the whole stream in one
//!   `convert` call (segments of 20 s with 2 s of context beyond 30 s).
//! - **Fixed-shape models** (the Sigil Voice Engine): a second instance of the voice with long
//!   shapes (200 ms blocks, 80 ms lookahead, 500 ms of extra context by default: `CeilingSpec`
//!   explains why not the longest context) converting the stream block after block with no
//!   crossfade. Its time-anchored noise is keyed to the same absolute frames as the stream's, so
//!   the two renditions differ only by what streaming changes (block, context, joins). The first
//!   block starts where its whole context lies inside the stream; the quality run leads the clip
//!   with silence (`lead_s`) so every sample of the clip is covered.
//! - **Voice families** (P5-I0, `family_ceiling`): Sigil-ZS and so-vits-svc voices convert the
//!   whole stream as a one-off call of the stream's model (each family plans any window); DDSP-SVC
//!   through a second instance configured for the whole recording; MeanVC2 (its offline path is
//!   the streaming loop) and sidecar voices have none.
//!
//! Stream and ceiling are placed against each other from this timeline (`fidelity.rs`): the
//! ceiling's first absolute position, where the input entered the recorded stream, and the
//! probed harness latency.
//!
//! # Scorer v2 ceilings (P3B-B, ADR 0009)
//! - **Long shape (ceiling A)** is the fixed-shape ceiling above; [`long_shape`] also renders it
//!   with its block grid shifted (the same-seed floor: ceiling A against itself with block
//!   boundaries 100 ms apart), with another noise seed (the different-seed floor, an upper bound),
//!   with the engine probe on (its fed F0, oracle overrides for the ablations) and with a pinned
//!   content-encoder variant (`rvc.content_variant`).
//! - **Ceiling B** ([`render_blocks`]): 400 ms blocks, 160 ms of lookahead, at most 500 ms of left
//!   context (never 1 s: a second of context mutes an utterance's onset after silence on this
//!   voice), so it only adds right context to A; `mcd_offline_floor = MCD(A, B)`. Its shapes are
//!   beyond the engine's plan limits (200 ms blocks, 80 ms lookahead), so it runs on the offline
//!   renderer below.
//! - **Whole utterance** ([`render_whole`]): the input region (speech plus a margin) with about 1 s
//!   of reflect padding on each side (RVC's `x_pad`, not the digital-silence lead) converted in one
//!   pass: content features and whole-clip F0 over the padded utterance, `enc` and a full `flowdec`
//!   over all of it. It runs at the run's flowdec precision (fp32 as an extra row), on a separate
//!   dynamic-shape session path on the CPU EP (not the engine's `ShapePlan`, no
//!   `MAX_CONTEXT_FRAMES`), and its result is placed on the absolute timeline. Its content
//!   features start on the stream's 320-sample content grid, so they are also the oracle content
//!   of the ablations ([`OfflineAnalysis`]).
//!
//! [`CallSpan`] records each call's context start and, when the stream loop reports it
//! ([`StreamRecorder::note_join`]), the SOLA offset its join landed at (the offset-corrected MCD).

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use serde::Serialize;

use crate::services::neural_audio::ort_session::VcSessionOptions;
use crate::services::neural_audio::vc::engine::content::{resolve_encoder_file, ContentEncoder, EmbedderKind};
use crate::services::neural_audio::vc::engine::generator::{SpeakerTable, SplitGenerator};
use crate::services::neural_audio::vc::engine::noise::{AnchoredNoise, NoiseFrames, NoiseStream, PhaseTracker};
use crate::services::neural_audio::vc::engine::options::{EffectiveLevers, PrecisionMode};
use crate::services::neural_audio::vc::engine::plan::{ShapePlan, HOP_16K};
use crate::services::neural_audio::vc::engine::post::{PostConfig, PostStage};
use crate::services::neural_audio::vc::engine::probe::{OracleContent, OracleF0, ProbeCounts, ProbeSetup};
use crate::services::neural_audio::vc::families::FamilyKind;
use crate::services::neural_audio::vc::import::layout::Template;
use crate::services::neural_audio::vc::import::session::WeightImages;
use crate::services::neural_audio::vc::manifest::load_slot_manifest;
use crate::services::neural_audio::vc::traits::ConversionCtx;
use crate::services::neural_audio::VoiceConversionModel;

use super::engine::Engine;
use super::quality::F0Analyzer;
use super::rvc_build::{self, RvcPlan};
use super::spec::{CeilingSpec, StreamSpec};

/// One model call as the stream recorder saw it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct CallSpan {
    /// Absolute 16 kHz position one past the call's newest sample.
    pub end_16k: u64,
    /// Samples the call returned (block + SOLA tail).
    pub returned_16k: usize,
    // ---- P3B-B ----
    /// Absolute 16 kHz position of the call's first context sample (`stream_pos_16k`).
    pub start_16k: u64,
    /// Where SOLA joined this call's audio, relative to the window centre (16 kHz samples; the
    /// harness's `last_join`), when the stream loop reported it.
    pub sola_offset_16k: Option<f64>,
    /// Normalized cross-correlation of that join.
    pub join_ncc: Option<f32>,
    /// The stream loop saw this call's own join result (`Some` or no join). The harness keeps only
    /// its last join, so when one device block makes several model calls the earlier calls' joins
    /// are unobserved: `sola_offset_16k` is `None` for them, which is not "joined at the centre".
    pub join_seen: bool,
}

/// The 16 kHz stream a model received, by absolute position (warm-ups are not recorded: the
/// harness warms through `warmup_at`).
#[derive(Debug, Default)]
pub struct StreamRecorder {
    start: Option<u64>,
    data: Vec<f32>,
    calls: Vec<CallSpan>,
    /// Calls whose join [`note_join`](Self::note_join) has already attributed.
    noted: usize,
}

impl StreamRecorder {
    pub fn record(&mut self, ctx: &ConversionCtx<'_>) {
        let pos = ctx.stream_pos_16k;
        let len = ctx.mono_16k.len();
        let start = match self.start {
            Some(s) if pos < s => {
                // Earlier than anything recorded (not expected from the harness): grow the front.
                let grow = (s - pos) as usize;
                self.data.splice(0..0, std::iter::repeat_n(0.0, grow));
                self.start = Some(pos);
                pos
            }
            Some(s) => s,
            None => {
                self.start = Some(pos);
                pos
            }
        };
        let off = (pos - start) as usize;
        if self.data.len() < off + len {
            self.data.resize(off + len, 0.0);
        }
        self.data[off..off + len].copy_from_slice(ctx.mono_16k);
        self.calls.push(CallSpan {
            end_16k: pos + len as u64,
            returned_16k: ctx.new_frames,
            start_16k: pos,
            sola_offset_16k: None,
            join_ncc: None,
            join_seen: false,
        });
    }

    /// The harness's last join (`VcStreamHarness::last_join`: offset from the window centre in
    /// output samples at `out_rate`, and its NCC) belongs to the newest recorded call (P3B-B; the
    /// stream loop calls this after every device block). A block without a model call has nothing
    /// new to attribute. The harness keeps only its last join, so when a block made several model
    /// calls only the newest one is attributed; the earlier ones stay `join_seen == false` and the
    /// scorers search their displacement instead of assuming the window centre.
    pub fn note_join(&mut self, join: Option<(isize, f32)>, out_rate: u32) {
        let n = self.calls.len();
        if n == self.noted {
            return;
        }
        self.noted = n;
        let c = &mut self.calls[n - 1];
        c.join_seen = true;
        if let Some((off, ncc)) = join {
            c.sola_offset_16k = Some(off as f64 * 16_000.0 / f64::from(out_rate.max(1)));
            c.join_ncc = Some(ncc);
        }
    }

    /// `(absolute position of data[0], data)`.
    pub fn stream(&self) -> (u64, &[f32]) {
        (self.start.unwrap_or(0), &self.data)
    }

    pub fn calls(&self) -> &[CallSpan] {
        &self.calls
    }
}

/// The ceiling's audio.
#[derive(Debug)]
pub struct CeilingOutput {
    /// At the model's output rate; sample `i` is the conversion of absolute 16 kHz position
    /// `start_16k + i · 16000 / rate`.
    pub audio: Vec<f32>,
    pub rate: u32,
    pub start_16k: u64,
    /// How it was made.
    pub method: String,
    pub compute_ms: f64,
    pub notes: Vec<String>,
}

/// Samples of `data` (starting at absolute `start`) over absolute `[from, from + len)`, zero
/// outside.
fn gather(data: &[f32], start: u64, from: u64, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    for (i, o) in out.iter_mut().enumerate() {
        let p = from + i as u64;
        if p >= start {
            if let Some(&v) = data.get((p - start) as usize) {
                *o = v;
            }
        }
    }
    out
}

fn ctx<'a>(mono: &'a [f32], new_frames: usize, pos: u64, stream: &StreamSpec) -> ConversionCtx<'a> {
    ConversionCtx {
        mono_16k: mono,
        new_frames,
        pitch_shift_semitones: stream.pitch_semitones,
        index_rate: stream.index_rate,
        protect: stream.protect,
        speaker_id: stream.speaker_id,
        stream_pos_16k: pos,
        reference: None,
        singing: Default::default(),
    }
}

/// Whole-utterance segments: one call up to 30 s, else 20 s pieces with 2 s of context.
const WHOLE_MAX_16K: usize = 30 * 16_000;
const SEGMENT_16K: usize = 20 * 16_000;
const SEGMENT_CONTEXT_16K: usize = 2 * 16_000;

/// Convert the recorded stream offline (see the module docs).
pub fn ceiling(engine: &Engine, stream: &StreamSpec, rec: &StreamRecorder, cfg: &CeilingSpec) -> Result<CeilingOutput, String> {
    let (start, data) = rec.stream();
    if data.is_empty() {
        return Err("the ceiling has no input: the stream recorder saw no model call".into());
    }
    let started = Instant::now();
    if let Some((plan, kind)) = engine.rvc.as_ref().and_then(|p| p.format.family().map(|k| (p, k))) {
        return family_ceiling(engine, plan, kind, stream, start, data, started);
    }
    let fixed = engine.fixed_block().is_some();
    match (&engine.rvc, fixed) {
        (Some(plan), true) => long_shape(plan, stream, rec, cfg, &LongShapeOpts::default()).map(|o| o.out),
        _ => whole_utterance(engine, stream, start, data, started),
    }
}

/// How a long-shape ceiling is rendered (P3B-B; the default is the P3 ceiling).
#[derive(Clone, Debug, Default)]
pub struct LongShapeOpts {
    /// Start the block grid this many 16 kHz samples later (a multiple of 160).
    pub shift_16k: u64,
    /// Another noise seed than the run's (the different-seed floor).
    pub noise_seed: Option<u64>,
    /// The engine probe for this render (fed F0, oracle overrides).
    pub probe: Option<ProbeSetup>,
}

/// A long-shape render and what its probe recorded.
#[derive(Debug)]
pub struct LongShapeOutput {
    pub out: CeilingOutput,
    /// The F0 fed to the generator `(absolute start frame, Hz after the shift)`, when the probe
    /// recorded it.
    pub fed_f0: Option<(u64, Vec<f32>)>,
    pub applied: ProbeCounts,
}

/// The fixed-shape ceiling of a template voice (ceiling A; see the module docs).
pub(crate) fn long_shape(plan: &RvcPlan, stream: &StreamSpec, rec: &StreamRecorder, cfg: &CeilingSpec, opts: &LongShapeOpts) -> Result<LongShapeOutput, String> {
    use crate::services::neural_audio::vc::traits::ModelTelemetry;
    let (start, data) = rec.stream();
    if data.is_empty() {
        return Err("the ceiling has no input: the stream recorder saw no model call".into());
    }
    let started = Instant::now();
    let mut p = plan.clone();
    if let Some(seed) = opts.noise_seed {
        p.noise_seed = seed;
    }
    let mut ce = rvc_build::ceiling_engine(&p, cfg)?;
    ce.set_voice_controls(&stream.voice_controls());
    if let Some(setup) = &opts.probe {
        ce.probe_mut().configure(setup.clone());
    }
    let sp = *ce.plan();
    let block = sp.block_16k;
    let la = sp.lookahead_frames * 160;
    let context = sp.context_16k;
    let rate = ce.output_sample_rate();
    let per_block = (block as u64 * u64::from(rate) / 16_000) as usize;
    let end = start + data.len() as u64;
    // First block whose whole context is inside the stream's absolute timeline.
    let min_k0 = (context as u64).saturating_sub((block + la) as u64);
    let first = start.max(min_k0);
    let first = first.div_ceil(160) * 160 + opts.shift_16k.div_ceil(160) * 160;
    let mut out = Vec::with_capacity(((end.saturating_sub(first)) as usize / block + 1) * per_block);
    let mut k0 = first;
    while k0 < end {
        let ctx_start = k0 + (block + la) as u64 - context as u64;
        let window = gather(data, start, ctx_start, context);
        let got = ce.convert(&ctx(&window, block, ctx_start, stream))?;
        if got.len() != per_block {
            return Err(format!("ceiling: the engine returned {} samples for a {per_block}-sample block", got.len()));
        }
        out.extend_from_slice(&got);
        k0 += block as u64;
    }
    let mut notes = vec![format!(
        "ceiling: {} ms blocks with {} ms of context and {} ms of lookahead, back to back (no crossfade){}",
        block / 16,
        context / 16,
        la / 16,
        if cfg.exact { ", full decode" } else { ", the run's levers" }
    )];
    if opts.shift_16k > 0 {
        notes.push(format!("ceiling: block grid shifted by {} ms", opts.shift_16k / 16));
    }
    if let Some(seed) = opts.noise_seed {
        notes.push(format!("ceiling: noise seed {seed:#x} (not the run's)"));
    }
    if first > start {
        notes.push(format!(
            "ceiling: the first {:.2} s of the stream precede the ceiling's first block",
            (first - start) as f64 / 16_000.0
        ));
    }
    let fed_f0 = ce.probe().fed_f0().map(|f| (f.start_frame, f.hz.to_vec()));
    Ok(LongShapeOutput {
        out: CeilingOutput {
            audio: out,
            rate,
            start_16k: first,
            method: "fixed-shape".into(),
            compute_ms: started.elapsed().as_secs_f64() * 1000.0,
            notes,
        },
        fed_f0,
        applied: ce.probe().applied(),
    })
}

/// Whole-utterance pieces of `len` samples: one up to 30 s, else 20 s pieces (`(start, end)`).
fn pieces(len: usize) -> Vec<(usize, usize)> {
    if len <= WHOLE_MAX_16K {
        return vec![(0, len)];
    }
    let mut out = Vec::new();
    let mut s = 0usize;
    while s < len {
        let e = (s + SEGMENT_16K).min(len);
        out.push((s, e));
        s = e;
    }
    out
}

/// Dynamic-shape models: the stream's own model converts the whole recording (one call per piece,
/// 2 s of context before each later piece).
fn whole_utterance(engine: &Engine, stream: &StreamSpec, start: u64, data: &[f32], started: Instant) -> Result<CeilingOutput, String> {
    let rate = engine.info.output_rate;
    let out = engine.with_model(|m| -> Result<Vec<f32>, String> {
        let mut out = Vec::new();
        for (s, e) in pieces(data.len()) {
            let c0 = s.saturating_sub(SEGMENT_CONTEXT_16K);
            out.extend(m.convert(&ctx(&data[c0..e], e - s, start + c0 as u64, stream))?);
        }
        Ok(out)
    })?;
    let method = if data.len() <= WHOLE_MAX_16K { "whole utterance" } else { "20 s segments" };
    Ok(CeilingOutput {
        audio: out,
        rate,
        start_16k: start,
        method: method.into(),
        compute_ms: started.elapsed().as_secs_f64() * 1000.0,
        notes: vec![format!("ceiling: {method}, one call per piece")],
    })
}

/// Voice families (P5-I0): each family's own offline conversion of the recorded stream.
/// - Sigil-ZS and so-vits-svc convert a one-off window of any length (Sigil-ZS on CPU plans at
///   that shape, so-vits-svc's generator over the whole window): the stream's model, whole.
/// - DDSP-SVC plans any shape: a second instance configured for the whole recording
///   ([`second_instance_whole`]; P5-A's real-model ceiling).
/// - MeanVC2's offline conversion is the same streaming loop, and a sidecar converts in its own
///   process: no ceiling (the run reports why).
fn family_ceiling(
    engine: &Engine,
    plan: &RvcPlan,
    kind: FamilyKind,
    stream: &StreamSpec,
    start: u64,
    data: &[f32],
    started: Instant,
) -> Result<CeilingOutput, String> {
    match kind {
        FamilyKind::SigilZs | FamilyKind::Sovits => {
            let mut out = whole_utterance(engine, stream, start, data, started)?;
            out.method = format!("{} ({} one-off)", out.method, kind.as_str());
            out.notes.push(format!("ceiling: the {} voice's one-off conversion of the recorded stream", kind.as_str()));
            Ok(out)
        }
        FamilyKind::DdspSvc => second_instance_whole(plan, kind, stream, start, data, started),
        FamilyKind::MeanVc => Err(
            "the offline ceiling is not defined for MeanVC2 voices: their offline conversion is the same streaming loop, so the stream is its own ceiling"
                .into(),
        ),
        FamilyKind::SidecarV2 => Err("the offline ceiling is not defined for sidecar voices: the sidecar converts in its own process".into()),
    }
}

/// A second instance of the voice (the factory's, silence skip off) configured for each whole
/// piece (`configure_stream(n, n, context)`), one call per piece. The window ends the model's own
/// delay after the piece, so the returned samples are the piece's; positions before the recording
/// are silence (the window's stream position is shifted to stay non-negative).
fn second_instance_whole(plan: &RvcPlan, kind: FamilyKind, stream: &StreamSpec, start: u64, data: &[f32], started: Instant) -> Result<CeilingOutput, String> {
    // Families plan whole 10 ms frames.
    const UNIT_16K: usize = 160;
    let len = data.len() / UNIT_16K * UNIT_16K;
    if len == 0 {
        return Err("the ceiling has no input: the recorded stream is shorter than 10 ms".into());
    }
    let (mut m, delay) = rvc_build::family_ceiling_model(plan)?;
    m.set_voice_controls(&stream.voice_controls());
    let rate = m.output_sample_rate();
    let mut out = Vec::with_capacity(len * rate as usize / 16_000 + 1);
    let mut shift: Option<i64> = None;
    for (s, e) in pieces(len) {
        let (s, e) = (s / UNIT_16K * UNIT_16K, e / UNIT_16K * UNIT_16K);
        let n = e - s;
        let extra = if s == 0 { 0 } else { SEGMENT_CONTEXT_16K.min(s) };
        let window_len = m.configure_stream(n, n, extra)?;
        let window_end = start as i64 + (e + delay) as i64;
        let window_start = window_end - window_len as i64;
        let shift = *shift.get_or_insert((-window_start).max(0));
        let window: Vec<f32> = (0..window_len as i64)
            .map(|i| {
                let abs = window_start + i - start as i64;
                usize::try_from(abs).ok().and_then(|k| data.get(k)).copied().unwrap_or(0.0)
            })
            .collect();
        let got = m.convert(&ctx(&window, n, (window_start + shift).max(0) as u64, stream))?;
        let want = (n as u64 * u64::from(rate) / 16_000) as usize;
        if got.len() != want {
            return Err(format!("ceiling: the {} voice returned {} samples for a {want}-sample piece", kind.as_str(), got.len()));
        }
        out.extend_from_slice(&got);
    }
    let method = if len <= WHOLE_MAX_16K { "whole utterance" } else { "20 s segments" };
    Ok(CeilingOutput {
        audio: out,
        rate,
        start_16k: start,
        method: format!("{method} ({} second instance)", kind.as_str()),
        compute_ms: started.elapsed().as_secs_f64() * 1000.0,
        notes: vec![format!(
            "ceiling: a second {} instance converting the recorded stream as {method} (one call per piece, no streaming)",
            kind.as_str()
        )],
    })
}

// ---------------------------------------------------------------------------------------------
// The offline renderer of template voices (ceiling B and the whole utterance; P3B-B)
// ---------------------------------------------------------------------------------------------

/// 16 kHz samples per content frame (the content encoder's 20 ms hop).
pub const CONTENT_HOP_16K: u64 = 320;
/// Reflect padding on each side of the whole-utterance ceiling's region (RVC's `x_pad`).
pub const WHOLE_PAD_16K: u64 = 16_000;
/// Longest region the whole-utterance ceiling converts in one pass.
pub const WHOLE_MAX_REGION_16K: u64 = 60 * 16_000;
/// Least the whole-utterance ceiling's region may start into the recording (s): the decoder returns
/// frames from its margin on (the shipped templates' 24 + 13 frames = 370 ms, plus a content frame
/// and the grid phase), so the first speech needs that much stream (digital-silence lead) before it.
pub const WHOLE_MIN_START_S: f64 = 0.4;
/// Ceiling B's shapes (ADR 0009): 400 ms blocks, 160 ms of lookahead, at most 500 ms of context.
pub const CEILING_B_BLOCK_16K: usize = 6_400;
pub const CEILING_B_LOOKAHEAD_16K: usize = 2_560;
pub const CEILING_B_CONTEXT_16K: usize = 8_000;

/// Precision and placement of an offline render.
#[derive(Clone, Debug)]
pub struct RenderOpts {
    /// `flowdec` precision (the run's: `fp16` when its flowdec runs fp16; `enc` is always fp32).
    pub precision: PrecisionMode,
    /// Content-encoder variant (`fp32` / `fp16` / `int8`), `None` = the stream's resolution.
    pub content_variant: Option<String>,
    /// F0 to decode with (Hz before the shift, absolute 10 ms grid) where it has frames; elsewhere
    /// the run's F0 method over each window.
    pub f0: Option<Arc<OracleF0>>,
    /// Content features where they have frames (the stream's 320-sample grid).
    pub content: Option<Arc<OracleContent>>,
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self {
            precision: PrecisionMode::Fp32,
            content_variant: None,
            f0: None,
            content: None,
        }
    }
}

/// A template voice's parts for dynamic-shape offline conversion on the CPU EP: the template,
/// its weight images, speaker table, a content encoder and an F0 analyzer of the run's method.
pub struct OfflineVoice {
    template: Template,
    weights_f32: Vec<u8>,
    weights_f16: Option<Vec<u8>>,
    speakers: SpeakerTable,
    encoder: ContentEncoder,
    pitch: F0Analyzer,
    opts: VcSessionOptions,
    levers: EffectiveLevers,
    noise_seed: u64,
    emb: usize,
    latent: usize,
    upp: usize,
    has_f0: bool,
    margin: usize,
    out_sr: u32,
    /// The template's audio rate (the NSF phase increments).
    sample_rate: u32,
    /// The generator of the last window shape (rebuilt when the shape changes).
    generator: Option<(usize, SplitGenerator)>,
    pub notes: Vec<String>,
}

/// The EP of the offline renderer (dynamic shapes, no GPU contention, deterministic).
const RENDER_EP: &str = "cpu";

impl OfflineVoice {
    /// Load `plan`'s voice for offline rendering at `opts`' precisions.
    pub fn load(plan: &RvcPlan, opts: &RenderOpts) -> Result<OfflineVoice, String> {
        let manifest = load_slot_manifest(&plan.slot)?;
        let templates_dir = plan
            .templates_dir
            .clone()
            .or_else(crate::services::neural_audio::vc::import::templates_dir)
            .ok_or("the bundled vc-templates folder was not found")?;
        let template_id = manifest.template.clone().unwrap_or_default();
        let template = Template::load(&templates_dir, template_id.trim()).map_err(|e| e.to_string())?;
        let arch = template.layout.arch.clone();
        let weights_file = plan.slot.join(manifest.weights.as_deref().unwrap_or(crate::services::neural_audio::vc::import::WEIGHTS_FILE));
        let weights_f32 = std::fs::read(&weights_file).map_err(|e| format!("{}: {e}", weights_file.display()))?;
        if weights_f32.len() as u64 != template.layout.weights.bytes_f32 {
            return Err(format!("{} does not fit template {}", weights_file.display(), template.layout.template));
        }
        let mut notes = Vec::new();
        let weights_f16 = if opts.precision == PrecisionMode::Fp16 {
            match crate::services::neural_audio::vc::import::writer::f16_image(&weights_f32) {
                Ok(img) => Some(img),
                Err(e) => {
                    notes.push(format!("offline render: fp32 only ({e})"));
                    None
                }
            }
        } else {
            None
        };
        let speakers_file = plan.slot.join(manifest.speakers.as_deref().unwrap_or(crate::services::neural_audio::vc::import::SPEAKERS_FILE));
        let speakers = SpeakerTable::load(&speakers_file, template.layout.speakers.dim)?;
        let kind = EmbedderKind::from_manifest(manifest.embedder.as_deref(), arch.emb_channels)?;
        let session = VcSessionOptions {
            intra_threads: Some(4),
            component_prefix: Some("bench.ceiling"),
            ..VcSessionOptions::default()
        };
        let explicit = Some(PathBuf::from(plan.spec.hubert.trim())).filter(|p| !p.as_os_str().is_empty());
        let encoder_path = match opts.content_variant.as_deref() {
            Some(v) => crate::services::neural_audio::vc::assets::model_file(kind.catalog_id(), Some(v))
                .ok_or_else(|| format!("content_variant {v}: {} has no installed model.{v}.onnx", kind.catalog_id()))?,
            None => resolve_encoder_file(kind, explicit.as_deref(), Some(&plan.slot), plan.ep != "cpu", plan.precision)?.path,
        };
        let variant: &'static str = match opts.content_variant.as_deref() {
            Some("fp16") => "fp16",
            Some("int8") => "int8",
            Some(_) => "fp32",
            None => "resolved",
        };
        // The stream's variant on the CPU EP; an unpinned GPU variant the CPU cannot run falls back
        // to fp32 with a note.
        let encoder = match ContentEncoder::load(kind, &encoder_path, variant, RENDER_EP, &session) {
            Ok(e) => e,
            Err(e) if opts.content_variant.is_none() => {
                let fp32 = crate::services::neural_audio::vc::assets::model_file(kind.catalog_id(), Some("fp32")).ok_or(e.clone())?;
                notes.push(format!("offline render: the stream's content encoder does not run on the CPU EP ({e}); using fp32"));
                ContentEncoder::load(kind, &fp32, "fp32", RENDER_EP, &session)?
            }
            Err(e) => return Err(e),
        };
        let pitch = F0Analyzer::new(&plan.pitch, explicit_pitch(plan).as_deref(), RENDER_EP, plan.f0_range);
        notes.extend(pitch.notes.iter().cloned());
        notes.push(format!(
            "offline render: CPU EP, content encoder {} ({variant}), F0 {} over each window",
            encoder_path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default(),
            pitch.method
        ));
        let levers = EffectiveLevers {
            precision: opts.precision,
            ..EffectiveLevers::default()
        };
        let margin = (template.layout.streaming.flow_margin + template.layout.streaming.dec_margin) as usize;
        Ok(OfflineVoice {
            emb: arch.emb_channels as usize,
            latent: arch.inter_channels as usize,
            upp: arch.upp as usize,
            has_f0: arch.f0,
            out_sr: arch.upp * 100,
            sample_rate: arch.sample_rate,
            margin,
            template,
            weights_f32,
            weights_f16,
            speakers,
            encoder,
            pitch,
            opts: session,
            levers,
            noise_seed: plan.noise_seed,
            generator: None,
            notes,
        })
    }

    /// The generator for windows of `t` frames (full decode: frames `[margin, t)` come back).
    fn generator(&mut self, t: usize) -> Result<&mut SplitGenerator, String> {
        if self.generator.as_ref().is_none_or(|(n, _)| *n != t) {
            self.generator = None;
            let plan = ShapePlan {
                block_16k: t * HOP_16K,
                context_16k: t * HOP_16K,
                enc_frames: t,
                return_frames: t - self.margin,
                flow_frames: t,
                upp: self.upp,
                f0_margin_frames: 0,
                want_16k: (t - self.margin) * HOP_16K,
                window_start: 0,
                decode_frames: t - self.margin,
                margin_frames: self.margin,
                f0_lead_frames: 0,
                partial_decode: false,
                lookahead_frames: 0,
            };
            let images = WeightImages {
                f32: &self.weights_f32,
                f16: self.weights_f16.as_deref(),
            };
            let generator = SplitGenerator::load(&self.template, &images, &plan, RENDER_EP, &self.opts, &self.levers)?;
            self.generator = Some((t, generator));
        }
        Ok(&mut self.generator.as_mut().expect("built above").1)
    }

    /// `flowdec` precision the renders run (after the template's parity gate).
    pub fn flowdec_precision(&mut self, t: usize) -> Result<&'static str, String> {
        Ok(self.generator(t)?.flowdec_precision())
    }

    /// Whole-window content features and F0 of `x` (16 kHz, first sample at absolute
    /// `origin_16k`): features `frames50 × emb` on the 320-sample grid from the origin, F0 (Hz before
    /// the shift) on the 10 ms grid from `origin_16k / 160`.
    pub fn analyse(&mut self, x: &[f32], origin_16k: u64) -> Result<OfflineAnalysis, String> {
        let mut feats = Vec::new();
        let frames50 = self.encoder.encode(x, &mut feats)?;
        if frames50 == 0 {
            return Err("offline render: the content encoder returned no frames".into());
        }
        let f0 = if self.has_f0 { self.pitch.contour(x)? } else { Vec::new() };
        Ok(OfflineAnalysis {
            content: OracleContent {
                origin_16k,
                emb: self.emb,
                feats,
            },
            f0: OracleF0 {
                start_frame: origin_16k / HOP_16K as u64,
                hz: f0,
            },
        })
    }

    /// Convert one window: `x` (16 kHz, `t · 160` samples from absolute `origin_16k`, a multiple of
    /// 160) with `analysis` (or the overrides of `opts` where they have frames); returns the audio
    /// of frames `[margin, t)` and the F0 (Hz after the shift) of all `t` frames.
    fn convert_window(
        &mut self,
        x: &[f32],
        origin_16k: u64,
        analysis: &OfflineAnalysis,
        opts: &RenderOpts,
        stream: &StreamSpec,
        phase0: f64,
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        let t = x.len() / HOP_16K;
        if t <= self.margin + 1 {
            return Err(format!("offline render: a {t}-frame window is shorter than the decoder's margin"));
        }
        let (emb, upp, latent) = (self.emb, self.upp, self.latent);
        // Content on the 10 ms grid (the engine's nearest 2x upsampling), with the oracle's frames.
        let mut feats = analysis.content.feats.clone();
        let frames50 = feats.len() / emb;
        if let Some(o) = opts.content.as_deref().filter(|o| o.emb == emb) {
            for k in 0..frames50 {
                let pos = origin_16k + CONTENT_HOP_16K * k as u64;
                if let Some(i) = pos.checked_sub(o.origin_16k).filter(|d| d % CONTENT_HOP_16K == 0).map(|d| (d / CONTENT_HOP_16K) as usize) {
                    if i < o.frames() {
                        feats[k * emb..(k + 1) * emb].copy_from_slice(&o.feats[i * emb..(i + 1) * emb]);
                    }
                }
            }
        }
        let mut phone = vec![0.0f32; t * emb];
        for j in 0..t {
            let src = (j / 2).min(frames50 - 1) * emb;
            phone[j * emb..(j + 1) * emb].copy_from_slice(&feats[src..src + emb]);
        }
        let shift = 2f32.powf(stream.pitch_semitones / 12.0);
        let first_frame = origin_16k / HOP_16K as u64;
        let mut pitchf = vec![0.0f32; t];
        let mut coarse = vec![1i64; t];
        if self.has_f0 {
            for (i, (pf, cb)) in pitchf.iter_mut().zip(coarse.iter_mut()).enumerate() {
                let abs = first_frame + i as u64;
                let hz = opts
                    .f0
                    .as_deref()
                    .and_then(|o| o.at(abs))
                    .or_else(|| analysis.f0.at(abs))
                    .unwrap_or(0.0);
                let v = hz * shift;
                *pf = if v.is_finite() && v > 0.0 { v } else { 0.0 };
                *cb = crate::services::neural_audio::vc::rvc::coarse_pitch(*pf);
            }
        }
        let controls = stream.voice_controls();
        let noise_scale = if controls.noise_scale.is_finite() { controls.noise_scale.max(0.0) } else { 0.0 };
        let noise = AnchoredNoise::new(self.noise_seed);
        let mut z = vec![0.0f32; latent * t];
        NoiseFrames::new(noise, NoiseStream::Flow, latent, t).copy_window(first_frame, t, noise_scale, true, &mut z);
        let mut src = vec![0.0f32; t * upp];
        NoiseFrames::new(noise, NoiseStream::Source, upp, t).copy_window(first_frame, t, 1.0, false, &mut src);
        let g = self.speakers.vector(stream.speaker_id).to_vec();
        let margin = self.margin;
        let gen = self.generator(t)?;
        gen.encode(&phone, if gen.template_has_f0() { &coarse } else { &[] })?;
        let mut audio = vec![0.0f32; (t - margin) * upp];
        gen.decode_window(0, &z, &g, &pitchf, phase0, &src, &mut audio)?;
        Ok((audio, pitchf))
    }
}

/// The run's explicit F0 model file (RMVPE / FCPE), if the spec names one.
fn explicit_pitch(plan: &RvcPlan) -> Option<PathBuf> {
    let pick = match plan.pitch.as_str() {
        "fcpe" => plan.spec.fcpe.trim(),
        _ => plan.spec.rmvpe.trim(),
    };
    Some(PathBuf::from(pick)).filter(|p| !p.as_os_str().is_empty())
}

/// Content features and F0 of a window (or of the whole padded utterance).
#[derive(Clone, Debug)]
pub struct OfflineAnalysis {
    pub content: OracleContent,
    pub f0: OracleF0,
}

/// A render of the offline renderer, with the F0 its frames were decoded with.
#[derive(Debug)]
pub struct RenderOutput {
    pub out: CeilingOutput,
    /// `(absolute start frame, Hz after the shift)` of the returned frames.
    pub fed_f0: (u64, Vec<f32>),
    /// The whole-utterance analysis (content, F0) the oracles of the ablations come from.
    pub analysis: Option<OfflineAnalysis>,
}

/// The whole-utterance ceiling (see the module docs): absolute region `[a, b)` of the recorded
/// stream, reflect-padded by about [`WHOLE_PAD_16K`] on each side from an origin on the stream's
/// 320-sample content grid (`grid_phase` = the stream's context starts modulo 320), converted in
/// one pass. The output covers the whole 10 ms frames of `[a, b)`.
pub(crate) fn render_whole(
    voice: &mut OfflineVoice,
    stream: &StreamSpec,
    rec: &StreamRecorder,
    region: (u64, u64),
    grid_phase: u64,
    opts: &RenderOpts,
) -> Result<RenderOutput, String> {
    let started = Instant::now();
    let (start, data) = rec.stream();
    let (a, b) = region;
    let end = start + data.len() as u64;
    if a < start || b > end || b <= a + 2 * HOP_16K as u64 {
        return Err(format!("whole ceiling: region [{a}, {b}) is not inside the recorded stream [{start}, {end})"));
    }
    if b - a > WHOLE_MAX_REGION_16K {
        return Err(format!("whole ceiling: the {:.1} s region is longer than {} s", (b - a) as f64 / 16_000.0, WHOLE_MAX_REGION_16K / 16_000));
    }
    let phase = grid_phase % CONTENT_HOP_16K;
    // Origin: the latest grid position at least the pad before `a` (not before position 0).
    let want = a.saturating_sub(WHOLE_PAD_16K);
    let origin = if want >= phase { want - (want - phase) % CONTENT_HOP_16K } else { phase.min(a) };
    let origin = if origin > a { a } else { origin };
    let left = (a - origin) as usize;
    let core = &data[(a - start) as usize..(b - start) as usize];
    let mut total = left + core.len() + WHOLE_PAD_16K as usize;
    total = total.div_ceil(CONTENT_HOP_16K as usize) * CONTENT_HOP_16K as usize;
    let right = total - left - core.len();
    let x = super::dsp::reflect_pad(core, left, right);
    let analysis = voice.analyse(&x, origin)?;
    let t = x.len() / HOP_16K;
    let precision = voice.flowdec_precision(t)?;
    let (audio, pitchf) = voice.convert_window(&x, origin, &analysis, opts, stream, 0.0)?;
    let fed = &pitchf[voice.margin..];
    // Frames [margin, t) came back: keep the whole frames of [a, b).
    let upp = voice.upp;
    let first_frame = origin / HOP_16K as u64 + voice.margin as u64;
    let keep_from = a.div_ceil(HOP_16K as u64);
    let keep_to = b / HOP_16K as u64;
    if keep_from < first_frame || keep_to <= keep_from {
        return Err(format!(
            "whole ceiling: the region starts {:.2} s into the recording, but the decoder's margin of {} frames needs {:.2} s before it (give the run a lead_s of at least {WHOLE_MIN_START_S} s)",
            a as f64 / 16_000.0,
            voice.margin,
            (origin / HOP_16K as u64 + voice.margin as u64) as f64 * HOP_16K as f64 / 16_000.0
        ));
    }
    let i0 = (keep_from - first_frame) as usize;
    let i1 = ((keep_to - first_frame) as usize).min(fed.len());
    let mut out = audio[i0 * upp..i1 * upp].to_vec();
    let mut post = PostStage::new(PostConfig {
        sample_rate: voice.out_sr,
        limiter: true,
        silence_gate: false,
        silence_hangover_ms: 150.0,
    });
    let src0 = (keep_from * HOP_16K as u64 - start) as usize;
    let src1 = (keep_to * HOP_16K as u64 - start) as usize;
    post.process_at(None, &data[src0..src1.min(data.len())], &mut out, &stream.voice_controls())?;
    let mut notes = voice.notes.clone();
    notes.push(format!(
        "whole ceiling: {:.2} s of input ({:.2} s of speech region) with {:.2} / {:.2} s of reflect padding, one pass, flowdec {precision}, phase 0",
        x.len() as f64 / 16_000.0,
        core.len() as f64 / 16_000.0,
        left as f64 / 16_000.0,
        right as f64 / 16_000.0
    ));
    Ok(RenderOutput {
        out: CeilingOutput {
            audio: out,
            rate: voice.out_sr,
            start_16k: keep_from * HOP_16K as u64,
            method: "whole utterance (offline renderer)".into(),
            compute_ms: started.elapsed().as_secs_f64() * 1000.0,
            notes,
        },
        fed_f0: (keep_from, fed[i0..i1].to_vec()),
        analysis: Some(analysis),
    })
}

/// Ceiling B (see the module docs): [`CEILING_B_BLOCK_16K`] blocks with
/// [`CEILING_B_LOOKAHEAD_16K`] of lookahead and [`CEILING_B_CONTEXT_16K`] of left context over the
/// recorded stream, the NSF phase carried from block to block, back to back. Its block grid starts
/// on `grid_phase` (the stream's content grid).
pub(crate) fn render_blocks(voice: &mut OfflineVoice, stream: &StreamSpec, rec: &StreamRecorder, grid_phase: u64, opts: &RenderOpts) -> Result<RenderOutput, String> {
    let started = Instant::now();
    let (start, data) = rec.stream();
    if data.is_empty() {
        return Err("ceiling B has no input: the stream recorder saw no model call".into());
    }
    let (block, la, context) = (CEILING_B_BLOCK_16K, CEILING_B_LOOKAHEAD_16K, CEILING_B_CONTEXT_16K);
    let window = context + block + la;
    let t = window / HOP_16K;
    let end = start + data.len() as u64;
    let phase = grid_phase % CONTENT_HOP_16K;
    // First block whose context starts inside the stream, on the content grid.
    let mut k0 = start + context as u64;
    k0 += (CONTENT_HOP_16K + phase - (k0 - context as u64) % CONTENT_HOP_16K) % CONTENT_HOP_16K;
    let first = k0;
    let precision = voice.flowdec_precision(t)?;
    let upp = voice.upp;
    let margin = voice.margin;
    let mut out = Vec::new();
    let mut fed = Vec::new();
    let mut tracker = PhaseTracker::new(upp, voice.sample_rate);
    let mut prev: Option<(u64, Vec<f32>)> = None;
    while k0 < end {
        let origin = k0 - context as u64;
        let x = gather(data, start, origin, window);
        let analysis = voice.analyse(&x, origin)?;
        // Phase-continuous excitation: advance over the frames the window slid by.
        let phase0 = match &prev {
            Some((p_origin, f0)) => {
                let slide = ((origin - p_origin) / HOP_16K as u64) as usize;
                tracker.advance(&f0[..slide.min(f0.len())]);
                tracker.phase_at()
            }
            None => 0.0,
        };
        let (audio, pitchf) = voice.convert_window(&x, origin, &analysis, opts, stream, phase0)?;
        // The block's frames sit `context` into the window; the decode starts at `margin`.
        let b0 = context / HOP_16K - margin;
        let b1 = b0 + block / HOP_16K;
        out.extend_from_slice(&audio[b0 * upp..b1 * upp]);
        fed.extend_from_slice(&pitchf[margin + b0..margin + b1]);
        // The tracker's phase is at the window's first frame: keep the whole window's F0.
        prev = Some((origin, pitchf));
        k0 += block as u64;
    }
    let mut post = PostStage::new(PostConfig {
        sample_rate: voice.out_sr,
        limiter: true,
        silence_gate: false,
        silence_hangover_ms: 150.0,
    });
    let src0 = (first - start) as usize;
    let src1 = (src0 + out.len() * 16_000 / voice.out_sr as usize).min(data.len());
    post.process_at(None, &data[src0..src1], &mut out, &stream.voice_controls())?;
    let mut notes = voice.notes.clone();
    notes.push(format!(
        "ceiling B: {} ms blocks with {} ms of context and {} ms of lookahead, back to back, full decode, flowdec {precision}",
        block / 16,
        context / 16,
        la / 16
    ));
    Ok(RenderOutput {
        out: CeilingOutput {
            audio: out,
            rate: voice.out_sr,
            start_16k: first,
            method: "ceiling B (offline renderer)".into(),
            compute_ms: started.elapsed().as_secs_f64() * 1000.0,
            notes,
        },
        fed_f0: (first / HOP_16K as u64, fed),
        analysis: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorder_rebuilds_the_stream_by_absolute_position() {
        let mut r = StreamRecorder::default();
        let a: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let c = |data: &[f32], pos: u64| -> Vec<f32> {
            // An overlapping window of the same signal: positions pos..pos+len carry value = pos.
            let _ = data;
            (pos..pos + 60).map(|p| p as f32).collect()
        };
        let w1 = c(&a, 1_000);
        let w2 = c(&a, 1_040);
        for (w, pos) in [(&w1, 1_000u64), (&w2, 1_040)] {
            r.record(&ConversionCtx {
                mono_16k: w,
                new_frames: 40,
                pitch_shift_semitones: 0.0,
                index_rate: 0.0,
                protect: 0.33,
                speaker_id: 0,
                stream_pos_16k: pos,
                reference: None,
                singing: Default::default(),
            });
        }
        // The harness reports its join after the second call: it belongs to that call. The first
        // call (made in the same device block) has no observed join of its own.
        r.note_join(Some((-5, 0.8)), 40_000);
        let (start, data) = r.stream();
        assert_eq!(start, 1_000);
        assert_eq!(data.len(), 100);
        assert!(data.iter().enumerate().all(|(i, &v)| v == (1_000 + i) as f32));
        assert_eq!(
            r.calls(),
            &[
                CallSpan {
                    end_16k: 1_060,
                    returned_16k: 40,
                    start_16k: 1_000,
                    sola_offset_16k: None,
                    join_ncc: None,
                    join_seen: false
                },
                CallSpan {
                    end_16k: 1_100,
                    returned_16k: 40,
                    start_16k: 1_040,
                    sola_offset_16k: Some(-2.0),
                    join_ncc: Some(0.8),
                    join_seen: true
                }
            ]
        );
        // gather: zero outside the recorded span.
        let g = gather(data, start, 990, 20);
        assert_eq!(&g[..10], &[0.0; 10]);
        assert_eq!(g[10], 1_000.0);
        // A block without a model call reports the same join again: nothing new to attribute.
        r.note_join(None, 40_000);
        assert_eq!(r.calls()[1].sola_offset_16k, Some(-2.0));
        // A new call whose block had no join (a silent tail): seen, with no offset.
        r.record(&ConversionCtx {
            mono_16k: &w2,
            new_frames: 40,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 1_080,
            reference: None,
            singing: Default::default(),
        });
        r.note_join(None, 40_000);
        assert_eq!((r.calls()[2].join_seen, r.calls()[2].sola_offset_16k), (true, None));
    }

    /// The fixture voice through the offline renderer: the whole-utterance ceiling covers the
    /// region's whole frames on the absolute timeline, its analysis starts on the content grid, and
    /// ceiling B covers the stream from its first full context; both are deterministic.
    #[test]
    fn the_offline_renderer_places_whole_and_block_ceilings_on_the_timeline() {
        let (plan, _root) = fixture_plan("render").expect("the tiny fixture plans");
        let stream: StreamSpec = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut rec = StreamRecorder::default();
        // A recorded stream from absolute 640: 0.5 s silence, 1.5 s of voice, 0.5 s silence.
        let mut data = vec![0.0f32; 8_000];
        data.extend(crate::services::bench::signals::synth(crate::services::bench::spec::SignalKind::Voice, 16_000, 1.5, 0.5, 3));
        data.extend(vec![0.0f32; 8_000]);
        rec.record(&ConversionCtx {
            mono_16k: &data,
            new_frames: 640,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 640,
            reference: None,
            singing: Default::default(),
        });
        let mut voice = OfflineVoice::load(&plan, &RenderOpts::default()).expect("the fixture loads for offline rendering");
        let region = (640 + 7_000, 640 + 33_100);
        let whole = render_whole(&mut voice, &stream, &rec, region, 640, &RenderOpts::default()).unwrap();
        assert_eq!(whole.out.start_16k, region.0.div_ceil(160) * 160);
        let frames = (region.1 / 160 - region.0.div_ceil(160)) as usize;
        assert_eq!(whole.out.audio.len(), frames * voice.upp);
        assert_eq!(whole.fed_f0.1.len(), frames);
        let a = whole.analysis.as_ref().unwrap();
        assert_eq!(a.content.origin_16k % CONTENT_HOP_16K, 0, "the analysis starts on the stream's content grid");
        // A second of padding where the timeline has room (here it does not: the region starts
        // 0.48 s into it), the stream's grid either way.
        assert!(a.content.origin_16k <= region.0);
        assert!(region.0 - a.content.origin_16k >= WHOLE_PAD_16K || a.content.origin_16k < CONTENT_HOP_16K);
        let later = (640 + 20_000, 640 + 33_100);
        let w2 = render_whole(&mut voice, &stream, &rec, later, 640, &RenderOpts::default()).unwrap();
        let o2 = w2.analysis.unwrap().content.origin_16k;
        assert!(later.0 - o2 >= WHOLE_PAD_16K && later.0 - o2 < WHOLE_PAD_16K + CONTENT_HOP_16K && o2.is_multiple_of(CONTENT_HOP_16K), "{o2}");
        let again = render_whole(&mut voice, &stream, &rec, region, 640, &RenderOpts::default()).unwrap();
        assert!(whole.out.audio.iter().zip(&again.out.audio).all(|(x, y)| x.to_bits() == y.to_bits()), "deterministic");
        let b = render_blocks(&mut voice, &stream, &rec, 640, &RenderOpts::default()).unwrap();
        assert!(b.out.start_16k >= 640 + CEILING_B_CONTEXT_16K as u64);
        assert_eq!((b.out.start_16k - 640) % CONTENT_HOP_16K, 0);
        assert_eq!(b.out.audio.len() % (CEILING_B_BLOCK_16K / 160 * voice.upp), 0);
        assert_eq!(b.fed_f0.1.len() * voice.upp, b.out.audio.len());
        assert!(b.out.audio.iter().all(|v| v.is_finite()));
    }

    /// The decoder returns frames from its margin on, so a region that starts less than the margin
    /// into the recording cannot be rendered whole: an error that names the fix (a longer
    /// `lead_s`), not a failure of the padding; one frame further in renders.
    #[test]
    fn the_whole_ceiling_needs_the_decoders_margin_before_its_region() {
        let (plan, _root) = fixture_plan("whole-margin").expect("the tiny fixture plans");
        let stream: StreamSpec = serde_json::from_value(serde_json::json!({})).unwrap();
        let mut rec = StreamRecorder::default();
        let data = crate::services::bench::signals::synth(crate::services::bench::spec::SignalKind::Voice, 16_000, 2.0, 0.5, 3);
        rec.record(&ConversionCtx {
            mono_16k: &data,
            new_frames: 640,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
            reference: None,
            singing: Default::default(),
        });
        let mut voice = OfflineVoice::load(&plan, &RenderOpts::default()).expect("the fixture loads for offline rendering");
        let m = voice.margin as u64;
        assert!(m >= 2, "the fixture template has a decoder margin ({m})");
        let early = (HOP_16K as u64 * (m - 1), HOP_16K as u64 * (m - 1) + 12_000);
        let err = render_whole(&mut voice, &stream, &rec, early, 0, &RenderOpts::default()).unwrap_err();
        assert!(err.contains("decoder's margin") && err.contains("lead_s"), "{err}");
        let fine = (HOP_16K as u64 * (m + 1), HOP_16K as u64 * (m + 1) + 12_000);
        let w = render_whole(&mut voice, &stream, &rec, fine, 0, &RenderOpts::default()).unwrap();
        assert_eq!(w.out.start_16k, fine.0);
        // And a region beyond the one-pass limit is refused with the limit named.
        let long: Vec<f32> = vec![0.0; (WHOLE_MAX_REGION_16K + 16_000) as usize];
        let mut long_rec = StreamRecorder::default();
        long_rec.record(&ConversionCtx {
            mono_16k: &long,
            new_frames: 640,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 0,
            reference: None,
            singing: Default::default(),
        });
        let err = render_whole(&mut voice, &stream, &long_rec, (8_000, 8_000 + WHOLE_MAX_REGION_16K + 160), 0, &RenderOpts::default()).unwrap_err();
        assert!(err.contains("longer than 60 s"), "{err}");
    }

    /// The fixture voice as a benchmark plan (template slot, CPU), and its temporary folder.
    fn fixture_plan(tag: &str) -> Option<(RvcPlan, FixtureDir)> {
        let (root, slot) = fixture_slot(tag);
        let spec: super::super::spec::RvcSpec = serde_json::from_value(serde_json::json!({
            "slot_dir": slot, "execution_provider": "cpu", "pitch_extractor": "dio", "precision": "fp32",
            "hubert": root.join("content.onnx").to_string_lossy()
        }))
        .ok()?;
        let mut plan = super::super::rvc_build::plan(&spec, &StreamSpec::default(), crate::services::rt_thread::WorkerPriority::Auto).ok()?;
        plan.templates_dir = Some(crate::services::neural_audio::vc::engine::generator::tests::tiny_dir());
        Some((plan, FixtureDir(root)))
    }

    /// A temporary folder removed on drop.
    pub(crate) struct FixtureDir(pub std::path::PathBuf);

    impl Drop for FixtureDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    /// A slot folder holding the tiny fixture voice (as the importer writes it) and the stand-in
    /// content encoder (`content.onnx` next to `voice-changer/`): `(root, slot)`.
    pub(crate) fn fixture_slot(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let root = std::env::temp_dir().join(format!("sigil-bench-{tag}-{}-{nanos}", std::process::id()));
        let slot = root.join("voice-changer").join("models").join("tiny");
        std::fs::create_dir_all(&slot).unwrap();
        let voice = crate::services::neural_audio::vc::engine::generator::tests::tiny_voice();
        std::fs::write(slot.join("weights.f32.bin"), &voice.weights.f32_image).unwrap();
        let speakers: Vec<u8> = voice.weights.speakers.iter().flat_map(|v| v.to_le_bytes()).collect();
        std::fs::write(slot.join("speakers.f32.bin"), speakers).unwrap();
        let record = serde_json::json!({
            "format": "sigil-rvc-import",
            "template": voice.template.layout.template,
            "weights_id": voice.template.layout.weights.weights_id,
        });
        std::fs::write(slot.join("sigil-import.json"), serde_json::to_vec_pretty(&record).unwrap()).unwrap();
        let manifest = serde_json::json!({
            "format": "sigil-rvc-1",
            "template": voice.template.layout.template,
            "weights": "weights.f32.bin",
            "speakers": "speakers.f32.bin",
            "speaker_count": voice.weights.n_speakers,
            "embedder": "contentvec",
            "vocoder": "hifigan",
            "fp16_ok": true,
            "import_record": "sigil-import.json"
        });
        std::fs::write(slot.join("manifest.json"), serde_json::to_vec_pretty(&manifest).unwrap()).unwrap();
        std::fs::write(root.join("content.onnx"), crate::services::neural_audio::vc::test_models::content_encoder_bytes()).unwrap();
        (root, slot)
    }
}
