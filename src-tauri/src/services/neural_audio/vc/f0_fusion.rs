//! Confidence-weighted F0 fusion (P3-D): voicing from one extractor (SwiftF0), pitch from another
//! (RMVPE), with the octave gate and median of `f0_post`, and the singing range (`F0Range`) up to
//! 2.5 kHz. Plan acceptance: fusion lowers gross pitch error against the best single extractor.
//!
//! Also the **pitch bank factory** ([`create_pitch_extractor`]): the one place that turns an F0
//! method name (after `voice_changer::params::effective_pitch_extractor`) into a running extractor,
//! so the worker, the engine factory and the benchmark all build the same thing; and the pitch
//! **A/B evaluator** ([`ab`]): a synthetic corpus with exact ground truth, scored offline and
//! through the production F0 cache (streaming).
//!
//! # Fusion, per 10 ms frame
//! - **Anchors**: frames where both extractors are voiced within [`AGREE_CENTS`]; their pitch is the
//!   confidence-weighted mean in log frequency. A frame the voicing extractor left unvoiced counts
//!   when its candidate pitch agrees with the pitch extractor and its confidence is at least
//!   [`CONFIRM_CONFIDENCE`], or [`CONFIRM_FLOOR`] when the pitch extractor is itself sure (at least
//!   [`STRONG_PITCH_CONFIDENCE`]) ("confirmed voicing": two extractors agreeing are more believable
//!   than either alone).
//! - **Voicing** otherwise comes from the voicing extractor (SwiftF0's own decision, with its edge
//!   rule for the newest frames), plus the frames the pitch extractor is sure of on its own (at
//!   least [`PITCH_VOICING_CONFIDENCE`]; a plain extractor such as FCPE counts 1 for its voiced
//!   frames). The newest frames of a streaming tick are exactly where SwiftF0's confidence drops
//!   and RMVPE / FCPE keep theirs.
//! - **Disagreements** (and frames voiced by the pitch extractor alone, between SwiftF0's candidate
//!   and the pitch extractor's value): the candidate nearer the closest anchor (looked for up to
//!   [`ANCHOR_RANGE`] frames either side) wins; without an anchor, the voicing extractor's pitch
//!   (SwiftF0's gross errors are a tenth of RMVPE's, whose missing-fundamental errors come in whole
//!   segments that have no anchors). A frame voiced by the pitch extractor with no candidate keeps
//!   its value, folded by whole octaves toward the nearest anchor.
//! - Then the pitch bank's post-processing (`f0_post::F0Post`: octave island gate, median, range).
//!
//! On the **CPU rung** the pitch extractor is FCPE, whose gross errors are 25x SwiftF0's: it only
//! assists the voicing ([`FusionConfig::cpu`]: a frame FCPE voices alone takes SwiftF0's candidate
//! when the two agree, every other frame is SwiftF0's).
//!
//! # A/B ([`ab`], `fusion_sweep`)
//! 27 synthetic clips x 6 s, 40 ms blocks through the production F0 cache, streaming (the frames each
//! tick decodes). The constants were chosen on the tuning corpus (seeds 1-27) and are reported here
//! on the held-out corpus ([`ab::HELD_OUT_SEED`], never used for tuning). GPE = gross pitch error
//! (> 20 %, over frames both call voiced), VDE = voicing error, FFE = F0 frame error (VDE plus the
//! gross errors, over every frame):
//!
//! | streaming, held-out | GPE | VDE | FFE |
//! |---|---|---|---|
//! | SwiftF0 alone (CPU) | 0.55 % | 22.1 % | 22.4 % |
//! | RMVPE alone (CUDA) | 6.22 % | 12.0 % | 16.5 % |
//! | FCPE alone (CPU) | 13.95 % | 9.2 % | 20.3 % |
//! | fusion SwiftF0 + RMVPE (CUDA), these defaults | 0.98 % | 12.2 % | 12.9 % |
//! | fusion SwiftF0 + FCPE (CPU), voicing assist | 1.53 % | 8.5 % | 9.7 % |
//! | fusion SwiftF0 + FCPE (CPU), the P3-D rule (anchors 12) | 3.45 % | 7.8 % | 10.6 % |
//!
//! (Tuning corpus: CUDA fusion 0.35 % / 10.9 % / 11.2 %, CPU fusion 0.73 % / 8.7 % / 9.3 %; with 20 ms
//! of engine lookahead the held-out CUDA fusion reaches 1.03 % / 7.0 % / 7.7 % against RMVPE's
//! 5.24 % / 8.4 % / 12.4 %.) Fusion has the lowest frame error of every extractor on both rungs and
//! far fewer gross errors than the default extractors (RMVPE, FCPE), but on held-out data its GPE
//! stays above SwiftF0 alone, which keeps its GPE low by leaving a quarter of the voiced frames
//! unvoiced; no variant of the rule beat SwiftF0's held-out streaming GPE.
//!
//! Confidences come from [`ConfidentPitch`] extractors (SwiftF0's confidence, RMVPE's peak
//! salience, CREPE's periodicity, PESTO's confidence); an extractor without one
//! ([`FusedPitchExtractor::new`] with plain extractors) counts 1 for voiced frames.
//!
//! # Frozen by P3-S0 (signatures)
//! - [`available`]`() -> bool`, [`FusionConfig`], [`FusedPitchExtractor::new`] and its
//!   `PitchExtractor` impl.
//!
//! # Added by P3-D
//! [`ConfidentPitch`], [`FuseParams`] (in [`FusionConfig::params`]), [`FusionConfig::cpu`],
//! [`FusedPitchExtractor::with_confidence`], [`FusedPitchExtractor::with_voicing_confidence`],
//! [`PitchBankRequest`], [`PitchBank`], [`create_pitch_extractor`],
//! [`create_pitch_extractor_with`] (the voice's GPU levers), [`pitch_model_id`],
//! [`resolve_pitch_model`], [`ab`] (with [`ab::report`], the benchmark's pitch A/B entry point).

use std::path::{Path, PathBuf};

use crate::services::diagnostics::EpOutcome;
use crate::services::neural_audio::gpu::GpuLevers;
use crate::services::neural_audio::ort_session::VcSessionOptions;

use super::assets;
use super::bases::base_models_dir;
use super::crepe::CrepePitchExtractor;
use super::engine::options::F0Range;
use super::f0_post::{cents, F0Post, F0PostConfig};
use super::fcpe::FcpePitchExtractor;
use super::pesto::PestoPitchExtractor;
use super::rmvpe::RmvpePitchExtractor;
use super::swiftf0::SwiftF0PitchExtractor;
use super::traits::PitchExtractor;
use super::world::DioPitchExtractor;

/// Whether F0 fusion runs in this build (it needs SwiftF0 plus RMVPE, or FCPE on the CPU). Like
/// SwiftF0's and CREPE's, this is about the code, not the models: the inspector offers the method so
/// its model-requirement card can offer the download, and a voice whose models are missing runs the
/// stand-in with a warning ([`create_pitch_extractor_with`]).
pub fn available() -> bool {
    true
}

/// Both voiced and this close (cents): the same pitch.
pub const AGREE_CENTS: f32 = 100.0;
/// Frames either side searched for an agreeing anchor when the extractors disagree (250 ms: an
/// octave error is still twice as far as any intonation step within it).
pub const ANCHOR_RANGE: usize = 25;
/// The voicing extractor's confidence at which a frame it left unvoiced still counts when the pitch
/// extractor is voiced on the same pitch (within [`AGREE_CENTS`]).
pub const CONFIRM_CONFIDENCE: f32 = 0.15;
/// The lower confidence at which such a frame counts when the pitch extractor is itself confident
/// (at least [`STRONG_PITCH_CONFIDENCE`]); SwiftF0's confidence on noise stays under 0.005.
pub const CONFIRM_FLOOR: f32 = 0.02;
/// Pitch-extractor confidence (RMVPE's peak salience) that lets [`CONFIRM_FLOOR`] apply.
pub const STRONG_PITCH_CONFIDENCE: f32 = 0.5;
/// Pitch-extractor confidence at which it voices a frame on its own, whatever the voicing
/// extractor says (infinite = never). A plain pitch extractor (FCPE on the CPU rung) counts 1 for
/// its voiced frames, so its voicing always counts.
pub const PITCH_VOICING_CONFIDENCE: f32 = 0.3;

/// Tunables of the fusion rule (module docs); the defaults are the A/B's choice.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FuseParams {
    /// Both voiced and this close (cents): the same pitch.
    pub agree_cents: f32,
    /// Frames either side searched for an agreeing anchor.
    pub anchor_range: usize,
    /// Voicing-extractor confidence at which an agreeing pitch extractor confirms a frame it left
    /// unvoiced.
    pub confirm_confidence: f32,
    /// ... and the lower one that applies when the pitch extractor's own confidence is at least
    /// `strong_pitch`.
    pub confirm_floor: f32,
    pub strong_pitch: f32,
    /// The pitch extractor voices a frame the voicing extractor left unvoiced on its own at this
    /// confidence; infinite = never. The frame's pitch is the voicing extractor's candidate when it
    /// has one and `voiced_by_pitch_uses_candidate` is set (SwiftF0's pitch stays accurate where
    /// its confidence drops), else the pitch extractor's, folded to the octave of the nearest anchor.
    pub pitch_voicing: f32,
    pub voiced_by_pitch_uses_candidate: bool,
    /// A disagreement with no anchor in range keeps the voicing extractor's pitch (else the more
    /// confident extractor's).
    pub prefer_voicing_without_anchor: bool,
    /// The pitch extractor only assists the voicing (the CPU rung's FCPE, whose gross errors are 30x
    /// SwiftF0's): every voiced frame keeps the voicing extractor's pitch, agreeing frames too, and a
    /// frame the pitch extractor voices alone (at `pitch_voicing`) takes the voicing extractor's
    /// candidate, or stays unvoiced without one. The other rules above do not apply.
    pub voicing_assist_only: bool,
    /// With `voicing_assist_only`: a frame voiced by the pitch extractor alone also needs the
    /// candidate within `agree_cents` of it (and modulo whole octaves with `assist_octave_fold`:
    /// FCPE's octave errors then still vouch for the voicing).
    pub assist_needs_agreement: bool,
    pub assist_octave_fold: bool,
    /// With `voicing_assist_only`: the voicing extractor's own confidence such a frame needs (its
    /// confidence on noise stays under 0.005).
    pub assist_min_confidence: f32,
}

impl Default for FuseParams {
    fn default() -> Self {
        FuseParams {
            agree_cents: AGREE_CENTS,
            anchor_range: ANCHOR_RANGE,
            confirm_confidence: CONFIRM_CONFIDENCE,
            confirm_floor: CONFIRM_FLOOR,
            strong_pitch: STRONG_PITCH_CONFIDENCE,
            pitch_voicing: PITCH_VOICING_CONFIDENCE,
            voiced_by_pitch_uses_candidate: true,
            prefer_voicing_without_anchor: true,
            voicing_assist_only: false,
            assist_needs_agreement: false,
            assist_octave_fold: false,
            assist_min_confidence: 0.0,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FusionConfig {
    pub range: F0Range,
    /// Median window (frames) after fusion.
    pub median_frames: usize,
    /// The fusion rule's tunables (P3-D addition).
    pub params: FuseParams,
}

impl Default for FusionConfig {
    fn default() -> Self {
        FusionConfig {
            range: F0Range::Speech,
            median_frames: 3,
            params: FuseParams::default(),
        }
    }
}

impl FusionConfig {
    /// The fusion of the CPU rung (SwiftF0 with FCPE): FCPE only assists the voicing
    /// ([`FuseParams::voicing_assist_only`]) and a frame it voices alone needs SwiftF0's candidate
    /// on the same pitch ([`FuseParams::assist_needs_agreement`], [`AGREE_CENTS`]). FCPE's gross
    /// errors are 25x SwiftF0's, so its pitch never reaches the output (module docs: A/B).
    pub fn cpu(range: F0Range) -> FusionConfig {
        FusionConfig {
            range,
            params: FuseParams {
                voicing_assist_only: true,
                assist_needs_agreement: true,
                ..FuseParams::default()
            },
            ..FusionConfig::default()
        }
    }
}

/// A pitch extractor that also reports a per-frame confidence (0..1) for its last contour.
pub trait ConfidentPitch: PitchExtractor {
    /// Confidence per frame of the last `f0` call, aligned with its contour.
    fn last_confidence(&self) -> &[f32];
    /// The pitch it saw on every frame before its voicing decision (0 = none), aligned with the
    /// contour; `None` when the extractor does not expose it. Fusion confirms under-confident
    /// voicing with it.
    fn last_candidates(&self) -> Option<&[f32]> {
        None
    }
}

impl ConfidentPitch for SwiftF0PitchExtractor {
    fn last_confidence(&self) -> &[f32] {
        SwiftF0PitchExtractor::last_confidence(self)
    }
    fn last_candidates(&self) -> Option<&[f32]> {
        Some(SwiftF0PitchExtractor::last_candidates(self))
    }
}

impl ConfidentPitch for RmvpePitchExtractor {
    fn last_confidence(&self) -> &[f32] {
        RmvpePitchExtractor::last_confidence(self)
    }
}

impl ConfidentPitch for CrepePitchExtractor {
    fn last_confidence(&self) -> &[f32] {
        CrepePitchExtractor::last_confidence(self)
    }
}

impl ConfidentPitch for PestoPitchExtractor {
    fn last_confidence(&self) -> &[f32] {
        PestoPitchExtractor::last_confidence(self)
    }
}

enum Source {
    Plain(Box<dyn PitchExtractor>),
    Confident(Box<dyn ConfidentPitch>),
}

impl Source {
    /// Run the extractor: F0 into `f0`, confidence (1 / 0 for plain extractors) into `conf`, the
    /// pre-voicing candidates (a copy of `f0` when the extractor has none) into `cand`.
    fn run(&mut self, mono_16k: &[f32], f0: &mut Vec<f32>, conf: &mut Vec<f32>, cand: &mut Vec<f32>) -> Result<(), String> {
        conf.clear();
        cand.clear();
        match self {
            Source::Plain(e) => {
                *f0 = e.f0(mono_16k)?;
                conf.extend(f0.iter().map(|&v| if v > 0.0 { 1.0 } else { 0.0 }));
                cand.extend_from_slice(f0);
            }
            Source::Confident(e) => {
                *f0 = e.f0(mono_16k)?;
                let c = e.last_confidence();
                conf.extend((0..f0.len()).map(|i| {
                    c.get(i)
                        .copied()
                        .unwrap_or(if f0[i] > 0.0 { 1.0 } else { 0.0 })
                }));
                match e.last_candidates() {
                    Some(k) => cand.extend((0..f0.len()).map(|i| k.get(i).copied().unwrap_or(f0[i]))),
                    None => cand.extend_from_slice(f0),
                }
            }
        }
        Ok(())
    }

    fn receptive_frames(&self) -> Option<(usize, usize)> {
        match self {
            Source::Plain(e) => e.receptive_frames(),
            Source::Confident(e) => e.receptive_frames(),
        }
    }
}

/// Voicing from `voicing`, pitch from `pitch`.
pub struct FusedPitchExtractor {
    voicing: Source,
    pitch: Source,
    config: FusionConfig,
    post: F0Post,
    post_cfg: F0PostConfig,
    v: Vec<f32>,
    vc: Vec<f32>,
    vk: Vec<f32>,
    p: Vec<f32>,
    pc: Vec<f32>,
    pk: Vec<f32>,
    scratch: FuseScratch,
    confidence: Vec<f32>,
}

impl FusedPitchExtractor {
    /// Fusion of two plain extractors (each voiced frame counts with confidence 1).
    #[allow(dead_code)] // Frozen P3-S0 constructor; the pitch bank builds the confidence-weighted forms.
    pub fn new(voicing: Box<dyn PitchExtractor>, pitch: Box<dyn PitchExtractor>, config: FusionConfig) -> FusedPitchExtractor {
        Self::build(Source::Plain(voicing), Source::Plain(pitch), config)
    }

    /// Fusion weighted by the extractors' own confidences.
    pub fn with_confidence(voicing: Box<dyn ConfidentPitch>, pitch: Box<dyn ConfidentPitch>, config: FusionConfig) -> FusedPitchExtractor {
        Self::build(Source::Confident(voicing), Source::Confident(pitch), config)
    }

    /// Confident voicing (SwiftF0) with a plain pitch extractor (FCPE on the CPU rung).
    pub fn with_voicing_confidence(voicing: Box<dyn ConfidentPitch>, pitch: Box<dyn PitchExtractor>, config: FusionConfig) -> FusedPitchExtractor {
        Self::build(Source::Confident(voicing), Source::Plain(pitch), config)
    }

    fn build(voicing: Source, pitch: Source, config: FusionConfig) -> FusedPitchExtractor {
        let post_cfg = fusion_post_config(&config);
        FusedPitchExtractor {
            voicing,
            pitch,
            config,
            post: F0Post::new(),
            post_cfg,
            v: Vec::new(),
            vc: Vec::new(),
            vk: Vec::new(),
            p: Vec::new(),
            pc: Vec::new(),
            pk: Vec::new(),
            scratch: FuseScratch::default(),
            confidence: Vec::new(),
        }
    }

    /// Fused confidence per frame of the last contour (the voicing extractor's).
    #[allow(dead_code)] // Diagnostics accessor (tests; the governor may weigh it later).
    pub fn last_confidence(&self) -> &[f32] {
        &self.confidence
    }
}

impl PitchExtractor for FusedPitchExtractor {
    fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
        self.voicing
            .run(mono_16k, &mut self.v, &mut self.vc, &mut self.vk)?;
        self.pitch
            .run(mono_16k, &mut self.p, &mut self.pc, &mut self.pk)?;
        // Both extractors put frame j at sample 160 j; fuse the frames both returned.
        let n = self.v.len().min(self.p.len());
        let mut out = Vec::with_capacity(n);
        fuse(
            &self.v[..n],
            &self.vc[..n],
            &self.vk[..n],
            &self.p[..n],
            &self.pc[..n],
            &mut out,
            &mut self.scratch,
            &self.config.params,
        );
        self.confidence.clear();
        self.confidence.extend_from_slice(&self.vc[..n]);
        self.post.process(&mut out, &self.post_cfg);
        Ok(out)
    }

    fn receptive_frames(&self) -> Option<(usize, usize)> {
        let a = self.voicing.receptive_frames()?;
        let b = self.pitch.receptive_frames()?;
        Some((a.0.max(b.0) + 1, a.1.max(b.1) + 1))
    }
}

/// Post-processing of the fused contour: the pitch bank's gate and range with the fusion's median.
fn fusion_post_config(config: &FusionConfig) -> F0PostConfig {
    F0PostConfig {
        median_frames: config.median_frames.max(1),
        ..F0PostConfig::for_range(config.range)
    }
}

#[derive(Default)]
struct FuseScratch {
    anchor: Vec<bool>,
    prev_anchor: Vec<Option<usize>>,
    next_anchor: Vec<Option<usize>>,
}

/// The fusion rule (module docs) over aligned contours: `voicing` / `pitch` in Hz (0 = unvoiced),
/// with their confidences; `v_cand` is the voicing extractor's pitch before its voicing decision.
#[allow(clippy::too_many_arguments)]
fn fuse(voicing: &[f32], v_conf: &[f32], v_cand: &[f32], pitch: &[f32], p_conf: &[f32], out: &mut Vec<f32>, s: &mut FuseScratch, prm: &FuseParams) {
    let n = voicing.len();
    out.clear();
    let agree = |a: f32, b: f32| a > 0.0 && b > 0.0 && cents(a, b).abs() <= prm.agree_cents;
    if prm.voicing_assist_only {
        // The pitch always comes from the voicing extractor; the pitch extractor only adds voicing.
        out.extend((0..n).map(|j| {
            if voicing[j] > 0.0 {
                return voicing[j];
            }
            let (c, p) = (v_cand[j], pitch[j]);
            let vouched = p > 0.0 && p_conf[j] >= prm.pitch_voicing && c > 0.0 && v_conf[j] >= prm.assist_min_confidence;
            let p = if prm.assist_octave_fold { fold_octaves(p, c) } else { p };
            if vouched && (!prm.assist_needs_agreement || agree(p, c)) {
                c
            } else {
                0.0
            }
        }));
        return;
    }
    // Frames where both are voiced and agree: the anchors, fused by confidence. A frame the voicing
    // extractor left under its own threshold counts when the pitch extractor is voiced on the same
    // pitch (confirmed voicing): two extractors agreeing are more believable than either alone.
    s.anchor.clear();
    for j in 0..n {
        let p = pitch[j];
        let confirmed = v_conf[j] >= prm.confirm_confidence || (v_conf[j] >= prm.confirm_floor && p_conf[j] >= prm.strong_pitch);
        let v = if voicing[j] > 0.0 {
            voicing[j]
        } else if confirmed && agree(v_cand[j], p) {
            v_cand[j]
        } else {
            0.0
        };
        let both = agree(v, p);
        s.anchor.push(both);
        out.push(if both {
            let (wv, wp) = (v_conf[j].max(1e-3), p_conf[j].max(1e-3));
            ((v.log2() * wv + p.log2() * wp) / (wv + wp)).exp2()
        } else {
            0.0
        });
    }
    s.prev_anchor.clear();
    let mut last = None;
    for j in 0..n {
        s.prev_anchor.push(last);
        if s.anchor[j] {
            last = Some(j);
        }
    }
    s.next_anchor.clear();
    s.next_anchor.resize(n, None);
    let mut next = None;
    for j in (0..n).rev() {
        s.next_anchor[j] = next;
        if s.anchor[j] {
            next = Some(j);
        }
    }
    for j in 0..n {
        if s.anchor[j] {
            continue;
        }
        let (v, p) = (voicing[j], pitch[j]);
        let near = [s.prev_anchor[j], s.next_anchor[j]]
            .into_iter()
            .flatten()
            .filter(|&a| a.abs_diff(j) <= prm.anchor_range)
            .min_by_key(|&a| (a.abs_diff(j), a > j));
        // Two candidates disagree: the one nearer the closest anchor; without an anchor, the voicing
        // extractor's (with `prefer_voicing_without_anchor`) or the more confident one.
        let anchor = near.map(|a| out[a]);
        let resolve = |v: f32, p: f32| match anchor {
            Some(r) => {
                if cents(v, r).abs() <= cents(p, r).abs() {
                    v
                } else {
                    p
                }
            }
            None => {
                if prm.prefer_voicing_without_anchor || v_conf[j] >= p_conf[j] {
                    v
                } else {
                    p
                }
            }
        };
        // Voicing: the voicing extractor's own decision (its contour is already thresholded), or a
        // frame the pitch extractor is sure of on its own. Such a frame takes the voicing
        // extractor's candidate or the pitch extractor's value, whichever the anchors support (the
        // candidate stays accurate where SwiftF0's confidence drops); without a candidate, the pitch
        // extractor's value folded to the octave of the nearest anchor.
        if v <= 0.0 {
            out[j] = if p > 0.0 && p_conf[j] >= prm.pitch_voicing {
                let cand = if prm.voiced_by_pitch_uses_candidate { v_cand[j] } else { 0.0 };
                match anchor {
                    _ if cand > 0.0 => resolve(cand, p),
                    Some(r) => fold_octaves(p, r),
                    None => p,
                }
            } else {
                0.0
            };
            continue;
        }
        if p <= 0.0 {
            out[j] = v;
            continue;
        }
        out[j] = resolve(v, p);
    }
}

/// `hz` moved by whole octaves (at most two) toward `reference` when it sits within 300 cents of a
/// whole number of octaves from it (an octave error); anything else is left as it is.
fn fold_octaves(hz: f32, reference: f32) -> f32 {
    let d = (reference / hz).log2();
    let k = d.round().clamp(-2.0, 2.0);
    if k != 0.0 && (d - k).abs() <= 0.25 {
        hz * k.exp2()
    } else {
        hz
    }
}

// ---------------------------------------------------------------------------------------------
// Pitch bank factory
// ---------------------------------------------------------------------------------------------

/// Model-store id of an F0 method's model (`dio` has none; `fusion` uses SwiftF0 plus RMVPE / FCPE).
pub fn pitch_model_id(method: &str) -> Option<&'static str> {
    Some(match method {
        "fcpe" => assets::PITCH_FCPE,
        "rmvpe" => assets::PITCH_RMVPE,
        "swiftf0" => assets::PITCH_SWIFTF0,
        "crepe" => assets::PITCH_CREPE_TINY,
        "pesto" => assets::PITCH_PESTO,
        _ => return None,
    })
}

/// File names looked for in the legacy `app_data/voice-changer/base/` folder.
fn base_candidates(method: &str) -> &'static [&'static str] {
    match method {
        "fcpe" => &["fcpe.onnx", "FCPE.onnx"],
        "rmvpe" => &["rmvpe.onnx", "RMVPE.onnx", "rmvpe.pt.onnx"],
        "swiftf0" => &["swiftf0.onnx", "swift_f0.onnx", "swift.onnx"],
        "crepe" => &["crepe_tiny.onnx", "crepe-tiny.onnx"],
        "pesto" => &["pesto.onnx"],
        _ => &[],
    }
}

/// The model file of an F0 method: the model-store entry, else the legacy `base/` folder of
/// `app_data`.
pub fn resolve_pitch_model(method: &str, app_data: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = pitch_model_id(method).and_then(|id| assets::model_file(id, None)) {
        return Some(p);
    }
    let base = base_models_dir(app_data?);
    base_candidates(method)
        .iter()
        .map(|n| base.join(n))
        .find(|p| p.is_file())
}

/// What the factory needs.
pub struct PitchBankRequest<'a> {
    /// The F0 method that runs (after `params::effective_pitch_extractor`): `fcpe`, `rmvpe`, `dio`,
    /// `swiftf0`, `crepe`, `pesto` or `fusion`.
    pub method: &'a str,
    /// EP ladder rung of the run (`cpu` / `cuda` / `dml`).
    pub ep: &'a str,
    pub opts: &'a VcSessionOptions,
    pub range: F0Range,
    /// Explicit FCPE / RMVPE files (the params' paths as the start preflight resolved them); `None`
    /// = model store, then `base/`.
    pub fcpe_path: Option<&'a Path>,
    pub rmvpe_path: Option<&'a Path>,
    /// For the legacy `base/` candidates.
    pub app_data: Option<&'a Path>,
    /// Run the small CPU-friendly extractors (SwiftF0, PESTO) on the CPU with one thread whatever
    /// the rung (frees the GPU; default true).
    pub small_models_on_cpu: bool,
}

/// A built extractor.
pub struct PitchBank {
    pub extractor: Box<dyn PitchExtractor>,
    /// The method that runs.
    pub method: &'static str,
    /// Non-fatal notes (a fusion part that runs a substitute, ...), shown as `rvc_warning`.
    pub notes: Vec<String>,
    /// EP outcome of each ONNX session (`vc.pitch` rows are also recorded by `ort_session`).
    pub outcomes: Vec<(String, EpOutcome)>,
}

fn small_opts(opts: &VcSessionOptions) -> VcSessionOptions {
    VcSessionOptions {
        intra_threads: Some(1),
        vram_cap_mb: None,
        ..opts.clone()
    }
}

fn need(method: &str, req: &PitchBankRequest<'_>, explicit: Option<&Path>) -> Result<String, String> {
    if let Some(p) = explicit.filter(|p| !p.as_os_str().is_empty()) {
        return if p.is_file() {
            Ok(p.to_string_lossy().into_owned())
        } else {
            Err(format!("{method} model not found: {}", p.display()))
        };
    }
    resolve_pitch_model(method, req.app_data)
        .map(|p| p.to_string_lossy().into_owned())
        .ok_or_else(|| match pitch_model_id(method) {
            Some(id) => format!("the {method} pitch model is not installed (model {id})"),
            None => format!("no model for F0 method {method}"),
        })
}

fn build_swiftf0(req: &PitchBankRequest<'_>) -> Result<SwiftF0PitchExtractor, String> {
    let path = need("swiftf0", req, None)?;
    let (ep, opts) = if req.small_models_on_cpu {
        ("cpu", small_opts(req.opts))
    } else {
        (req.ep, req.opts.clone())
    };
    Ok(SwiftF0PitchExtractor::load(&path, ep, &opts)?.with_range(req.range))
}

/// Build the extractor for `req.method` (no GPU levers). A missing model is an error (the A/B, the
/// autotuner and the governor's light step score or switch to exactly the method they ask for).
pub fn create_pitch_extractor(req: &PitchBankRequest<'_>) -> Result<PitchBank, String> {
    build_bank(req, GpuLevers::default())
}

/// The voice's pitch extractor (the engine factory): [`create_pitch_extractor`] with the voice's GPU
/// levers (`GpuLevers::from_effective`: on CUDA, `io_binding` plus `cuda_graphs` run RMVPE, alone or
/// inside the fusion, as a CUDA-graph plan; on DirectML, `io_binding` runs it as a fixed-shape plan;
/// `rmvpe` module docs), and a P3 method whose model is not installed (SwiftF0, CREPE-tiny, a part
/// of the fusion) runs its stand-in (the contract's `stand_in`, FCPE for RMVPE on the CPU rung) with
/// a note instead of refusing to start. [`PitchBank::method`] is the method that runs.
pub fn create_pitch_extractor_with(req: &PitchBankRequest<'_>, levers: GpuLevers) -> Result<PitchBank, String> {
    let method = req.method.trim();
    let Some(missing) = missing_model(method, req) else {
        return build_bank(req, levers);
    };
    let stand_in = stand_in_on(method, req.ep);
    if stand_in == method {
        return Err(missing);
    }
    let mut bank = build_bank(
        &PitchBankRequest {
            method: stand_in,
            ..*req
        },
        levers,
    )
    .map_err(|e| format!("{missing}; its stand-in {stand_in} cannot run either: {e}"))?;
    bank.notes
        .insert(0, format!("{method}: {missing}; running {stand_in} until it is installed"));
    Ok(bank)
}

/// Why a P3 method cannot be built for `req` (its model, or one of the fusion's, is not installed);
/// `None` for the baseline methods (their errors stand) and when everything is there.
fn missing_model(method: &str, req: &PitchBankRequest<'_>) -> Option<String> {
    match method {
        "swiftf0" | "crepe" => need(method, req, None).err(),
        "fusion" => need("swiftf0", req, None).err().or_else(|| {
            if is_cpu(req.ep) {
                need("fcpe", req, req.fcpe_path).err()
            } else {
                need("rmvpe", req, req.rmvpe_path).err()
            }
        }),
        _ => None,
    }
}

fn is_cpu(ep: &str) -> bool {
    ep.trim().eq_ignore_ascii_case("cpu")
}

/// The method `method` runs as while it cannot (`voice_changer::params::pitch_stand_in`, the
/// contract's `enums.pitch_extractor.stand_in`), on rung `ep`: RMVPE is GPU-only, so FCPE on the CPU.
fn stand_in_on(method: &str, ep: &str) -> &'static str {
    let stand_in = match method {
        "swiftf0" | "pesto" | "fcpe" => "fcpe",
        "crepe" | "fusion" | "rmvpe" => "rmvpe",
        "dio" => "dio",
        _ => "fcpe",
    };
    if stand_in == "rmvpe" && is_cpu(ep) {
        "fcpe"
    } else {
        stand_in
    }
}

fn build_bank(req: &PitchBankRequest<'_>, levers: GpuLevers) -> Result<PitchBank, String> {
    let mut notes = Vec::new();
    let mut outcomes = Vec::new();
    let (extractor, method): (Box<dyn PitchExtractor>, &'static str) = match req.method.trim() {
        "dio" => (Box::new(DioPitchExtractor::new().with_range(req.range)), "dio"),
        "rmvpe" => {
            let e = RmvpePitchExtractor::load_with_levers(&need("rmvpe", req, req.rmvpe_path)?, req.ep, req.opts, levers)?;
            outcomes.push(("vc.pitch".to_string(), e.outcome().clone()));
            (Box::new(e), "rmvpe")
        }
        "swiftf0" => {
            let e = build_swiftf0(req)?;
            outcomes.push(("vc.pitch".to_string(), e.outcome().clone()));
            (Box::new(e), "swiftf0")
        }
        "crepe" => {
            let e = CrepePitchExtractor::load(&need("crepe", req, None)?, req.ep, req.opts)?.with_range(req.range);
            outcomes.push(("vc.pitch".to_string(), e.outcome().clone()));
            (Box::new(e), "crepe")
        }
        "pesto" => {
            let (ep, opts) = if req.small_models_on_cpu {
                ("cpu", small_opts(req.opts))
            } else {
                (req.ep, req.opts.clone())
            };
            let e = PestoPitchExtractor::load(&need("pesto", req, None)?, ep, &opts)?.with_range(req.range);
            outcomes.push(("vc.pitch".to_string(), e.outcome().clone()));
            (Box::new(e), "pesto")
        }
        "fusion" => {
            let voicing = build_swiftf0(req)?;
            outcomes.push(("vc.pitch.voicing".to_string(), voicing.outcome().clone()));
            let config = FusionConfig {
                range: req.range,
                ..FusionConfig::default()
            };
            let fused = if is_cpu(req.ep) {
                notes.push("fusion: RMVPE is GPU-only in Sigil; fusing SwiftF0 with FCPE on the CPU execution provider".to_string());
                let pitch = FcpePitchExtractor::load(&need("fcpe", req, req.fcpe_path)?, req.ep, req.opts)?;
                outcomes.push(("vc.pitch".to_string(), pitch.outcome().clone()));
                FusedPitchExtractor::with_voicing_confidence(Box::new(voicing), Box::new(pitch), FusionConfig::cpu(req.range))
            } else {
                let pitch = RmvpePitchExtractor::load_with_levers(&need("rmvpe", req, req.rmvpe_path)?, req.ep, req.opts, levers)?;
                outcomes.push(("vc.pitch".to_string(), pitch.outcome().clone()));
                FusedPitchExtractor::with_confidence(Box::new(voicing), Box::new(pitch), config)
            };
            (Box::new(fused), "fusion")
        }
        _ => {
            let e = FcpePitchExtractor::load(&need("fcpe", req, req.fcpe_path)?, req.ep, req.opts)?;
            outcomes.push(("vc.pitch".to_string(), e.outcome().clone()));
            (Box::new(e), "fcpe")
        }
    };
    Ok(PitchBank {
        extractor,
        method,
        notes,
        outcomes,
    })
}

/// The model of `name` (`pitch-swiftf0`, `pitch-crepe-tiny`, `pitch-fcpe`, `pitch-rmvpe`,
/// `pitch-pesto`) under `SIGIL_TEST_MODELS_DIR` (the dev cache layout: `vc-assets/vc/<name>/model.onnx`,
/// `vc/<name>/model.onnx`, or `upstream-assets/rmvpe/rmvpe.onnx`).
#[cfg(test)]
pub(crate) fn test_model(name: &str) -> Option<PathBuf> {
    let root = std::env::var_os("SIGIL_TEST_MODELS_DIR").map(PathBuf::from)?;
    let mut candidates = vec![
        root.join("vc-assets")
            .join("vc")
            .join(name)
            .join("model.onnx"),
        root.join("vc").join(name).join("model.onnx"),
    ];
    if name == "pitch-rmvpe" {
        candidates.push(
            root.join("upstream-assets")
                .join("rmvpe")
                .join("rmvpe.onnx"),
        );
    }
    candidates.into_iter().find(|p| p.is_file())
}

// ---------------------------------------------------------------------------------------------
// A/B evaluator
// ---------------------------------------------------------------------------------------------

/// Pitch A/B: a synthetic corpus with exact ground truth (speech-like and sung voices under noise,
/// music bleed and octave traps), scored offline (whole clip) and streaming (through the production
/// incremental F0 cache, `engine::f0_cache`, taking each frame from the tick that decoded it).
/// The benchmark (`--headless bench`, P3-F) and the dev-rig report test call [`run`].
// Only the dev-rig tests reach it until the benchmark's `pitch_ab` mode (P3-F) calls `report`.
#[allow(dead_code)]
pub mod ab {
    use super::super::engine::f0_cache::{F0Cache, F0CacheConfig, DEFAULT_LEAD_FRAMES, DEFAULT_MARGIN_FRAMES, HOP_16K};
    use super::super::engine::options::F0Range;
    use super::super::traits::PitchExtractor;
    use super::{create_pitch_extractor, resolve_pitch_model, PitchBankRequest};
    use crate::services::neural_audio::ort_session::VcSessionOptions;
    use std::path::Path;
    use std::time::Instant;

    const SR: f32 = 16_000.0;

    /// A harmonic tone (six harmonics, falling amplitude) of `len` samples at 16 kHz.
    pub fn harmonic_tone(hz: f32, len: usize, amp: f32) -> Vec<f32> {
        let mut phase = 0.0f64;
        (0..len)
            .map(|_| {
                phase += f64::from(hz) / f64::from(SR);
                (1..=6)
                    .map(|k| amp as f64 / k as f64 * (std::f64::consts::TAU * phase * k as f64).sin())
                    .sum::<f64>() as f32
            })
            .collect()
    }

    /// One scored clip: 16 kHz audio and the true F0 per 10 ms frame (0 = unvoiced, NaN = not
    /// scored: frames next to a voicing change).
    pub struct Clip {
        pub name: String,
        pub audio: Vec<f32>,
        pub truth: Vec<f32>,
    }

    /// Deterministic noise.
    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
        }
        fn gauss(&mut self) -> f64 {
            let (u, v) = (self.next().max(1e-12), self.next());
            (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
        }
    }

    /// Voice kinds of the corpus.
    #[derive(Clone, Copy, Debug)]
    pub enum Voice {
        /// Male speech, 85-180 Hz.
        Low,
        /// Female speech, 160-320 Hz.
        Mid,
        /// Child / high speech, 250-500 Hz.
        High,
        /// Singing: notes 110-880 Hz with vibrato and a real octave leap.
        Singing,
    }

    /// Conditions layered on a voice.
    #[derive(Clone, Copy, Debug, Default)]
    pub struct Condition {
        /// Pink-ish noise at this SNR (dB); `None` = clean.
        pub snr_db: Option<f32>,
        /// Background music (a harmonic chord) at this level relative to the voice (dB).
        pub music_db: Option<f32>,
        /// Remove the first two harmonics (the fundamental is missing).
        pub missing_fundamental: bool,
        /// Period doubling (creak): alternate glottal pulses are weaker.
        pub creak: bool,
    }

    /// Target F0 contour and voicing per sample for `voice` (seeded).
    fn contour(voice: Voice, len: usize, rng: &mut Rng) -> (Vec<f32>, Vec<bool>) {
        let mut f0 = vec![0.0f32; len];
        let mut voiced = vec![false; len];
        let mut i = 0usize;
        let (lo, hi) = match voice {
            Voice::Low => (85.0f64, 180.0),
            Voice::Mid => (160.0, 320.0),
            Voice::High => (250.0, 500.0),
            Voice::Singing => (110.0, 880.0),
        };
        let mut level = (lo * hi).sqrt();
        while i < len {
            // An unvoiced gap (silence or a fricative), then a voiced stretch.
            let gap = (0.06 + 0.2 * rng.next()) * f64::from(SR);
            i += gap as usize;
            let stretch = match voice {
                Voice::Singing => (0.5 + 0.8 * rng.next()) * f64::from(SR),
                _ => (0.25 + 0.6 * rng.next()) * f64::from(SR),
            } as usize;
            let end = (i + stretch).min(len);
            match voice {
                Voice::Singing => {
                    // Notes of 150-400 ms, legato (30 ms glides), 5.5 Hz vibrato of ±40 cents; one
                    // note in five leaps an octave.
                    let mut j = i;
                    let mut note = level;
                    while j < end {
                        let dur = ((0.15 + 0.25 * rng.next()) * f64::from(SR)) as usize;
                        let target = if rng.next() < 0.2 {
                            if note * 2.0 <= hi {
                                note * 2.0
                            } else {
                                note / 2.0
                            }
                        } else {
                            (note * (2f64).powf((rng.next() - 0.5) * 7.0 / 12.0)).clamp(lo, hi)
                        };
                        let glide = (0.03 * f64::from(SR)) as usize;
                        for k in j..(j + dur).min(end) {
                            let g = ((k - j) as f64 / glide as f64).min(1.0);
                            let base = note * (target / note).powf(g);
                            let vib = 40.0 / 1200.0 * (std::f64::consts::TAU * 5.5 * k as f64 / f64::from(SR)).sin();
                            f0[k] = (base * vib.exp2()) as f32;
                            voiced[k] = true;
                        }
                        note = target;
                        j += dur;
                    }
                    level = note;
                }
                _ => {
                    // Declination plus a rise-fall accent, micro-prosody and jitter.
                    let start_level = (level * (2f64).powf((rng.next() - 0.5) * 0.5)).clamp(lo, hi);
                    let accent = 0.1 + 0.4 * rng.next();
                    for k in i..end {
                        let t = (k - i) as f64 / (end - i).max(1) as f64;
                        let shape = -0.25 * t + accent * (std::f64::consts::PI * t).sin();
                        let micro = 0.01 * (std::f64::consts::TAU * 7.0 * k as f64 / f64::from(SR)).sin();
                        f0[k] = (start_level * (shape + micro).exp2()).clamp(lo * 0.9, hi * 1.1) as f32;
                        voiced[k] = true;
                    }
                    level = start_level;
                }
            }
            i = end;
        }
        (f0, voiced)
    }

    /// Render a clip: glottal-like harmonic source through moving vowel formants, fricative noise in
    /// some gaps, and the condition's noise, music and traps.
    pub fn render(name: &str, voice: Voice, condition: Condition, seconds: f32, seed: u64) -> Clip {
        let len = (seconds * SR) as usize;
        let mut rng = Rng(seed ^ 0x5157_4a0f_91d3_c8e1);
        let (f0, voiced) = contour(voice, len, &mut rng);
        // Source: band-limited harmonics with a -9 dB/octave tilt, shimmer, optional creak.
        let mut phase = 0.0f64;
        let mut source = vec![0.0f32; len];
        let mut shimmer = 1.0f64;
        let mut env = 0.0f64;
        for k in 0..len {
            let hz = f64::from(f0[k]);
            let on = voiced[k];
            env = if on {
                (env + 1.0 / 160.0).min(1.0)
            } else {
                (env - 1.0 / 160.0).max(0.0)
            };
            if env <= 0.0 {
                continue;
            }
            let hz = if hz > 0.0 { hz } else { 150.0 };
            let before = phase;
            phase += hz / f64::from(SR);
            if phase.floor() != before.floor() {
                shimmer = 1.0 + 0.08 * (rng.next() - 0.5);
            }
            let harmonics = ((7_800.0 / hz) as usize).max(1);
            let mut s = 0.0f64;
            let first = if condition.missing_fundamental { 3 } else { 1 };
            for h in first..=harmonics {
                let fh = hz * h as f64;
                let taper = ((7_800.0 - fh) / 400.0).clamp(0.0, 1.0);
                s += taper * (h as f64).powf(-1.5) * (std::f64::consts::TAU * phase * h as f64).sin();
            }
            if condition.creak {
                // Alternate periods at 60 % amplitude: a subharmonic at F0 / 2.
                let odd = (phase.floor() as i64) % 2 == 1;
                s *= if odd { 0.6 } else { 1.0 };
            }
            source[k] = (s * shimmer * env) as f32;
        }
        // Vowel formants moving every ~200 ms (two-pole resonators in cascade).
        const VOWELS: [[f64; 3]; 5] = [
            [730.0, 1090.0, 2440.0],
            [270.0, 2290.0, 3010.0],
            [530.0, 1840.0, 2480.0],
            [570.0, 840.0, 2410.0],
            [300.0, 870.0, 2240.0],
        ];
        let scale = match voice {
            Voice::Low => 1.0,
            Voice::Mid => 1.15,
            Voice::High | Voice::Singing => 1.3,
        };
        let mut audio = vec![0.0f32; len];
        let mut states = [[0.0f64; 2]; 3];
        let seg = (0.2 * SR) as usize;
        for k in 0..len {
            let v0 = (k / seg) % VOWELS.len();
            let v1 = (v0 + 1) % VOWELS.len();
            let t = ((k % seg) as f64 / seg as f64).powi(2);
            let mut x = f64::from(source[k]);
            for (fi, st) in states.iter_mut().enumerate() {
                let fc = scale * (VOWELS[v0][fi] * (1.0 - t) + VOWELS[v1][fi] * t);
                let bw = 80.0 + 40.0 * fi as f64;
                let r = (-std::f64::consts::PI * bw / f64::from(SR)).exp();
                let a1 = 2.0 * r * (std::f64::consts::TAU * fc / f64::from(SR)).cos();
                let a2 = -r * r;
                let y = (1.0 - r) * x + a1 * st[0] + a2 * st[1];
                st[1] = st[0];
                st[0] = y;
                x = y;
            }
            audio[k] = x as f32;
        }
        // Normalise the voice to -18 dBFS RMS over its voiced part.
        let (mut e, mut n) = (0.0f64, 0usize);
        for (k, &a) in audio.iter().enumerate() {
            if voiced[k] {
                e += f64::from(a) * f64::from(a);
                n += 1;
            }
        }
        let voice_rms = (e / n.max(1) as f64).sqrt().max(1e-9);
        let gain = 10f64.powf(-18.0 / 20.0) / voice_rms;
        for a in &mut audio {
            *a = (f64::from(*a) * gain) as f32;
        }
        let voice_rms = 10f64.powf(-18.0 / 20.0);
        // Fricatives in some gaps: high-passed noise bursts at -24 dBFS.
        let mut hp = 0.0f64;
        let mut prev = 0.0f64;
        let mut burst = 0usize;
        for k in 0..len {
            if !voiced[k] && burst == 0 && rng.next() < 1.0 / 4_000.0 {
                burst = ((0.05 + 0.1 * rng.next()) * f64::from(SR)) as usize;
            }
            if burst > 0 && !voiced[k] {
                let w = rng.gauss();
                hp = 0.9 * (hp + w - prev);
                prev = w;
                audio[k] += (hp * 0.06) as f32;
                burst -= 1;
            } else {
                burst = 0;
            }
        }
        if let Some(db) = condition.music_db {
            // A slow chord (G3, B3, D4 then C4, E4, G4) under everything.
            let level = voice_rms * 10f64.powf(f64::from(db) / 20.0);
            let chords = [[196.0f64, 246.9, 293.7], [261.6, 329.6, 392.0]];
            for (k, a) in audio.iter_mut().enumerate() {
                let c = chords[(k / (SR as usize)) % 2];
                let t = k as f64 / f64::from(SR);
                let s: f64 = c
                    .iter()
                    .map(|&f| {
                        (1..=4)
                            .map(|h| (std::f64::consts::TAU * f * h as f64 * t).sin() / h as f64)
                            .sum::<f64>()
                    })
                    .sum();
                *a += (s * level * 0.35) as f32;
            }
        }
        if let Some(snr) = condition.snr_db {
            // Pink-ish noise (one-pole low-passed white plus white).
            let level = voice_rms * 10f64.powf(-f64::from(snr) / 20.0);
            let mut lp = 0.0f64;
            for a in audio.iter_mut() {
                let w = rng.gauss();
                lp = 0.97 * lp + 0.03 * w * 5.0;
                *a += ((0.6 * w + 0.8 * lp) * level) as f32;
            }
        }
        // Truth per 10 ms frame; frames within 20 ms of a voicing change are not scored.
        let frames = len / HOP_16K;
        let truth = (0..frames)
            .map(|j| {
                let c = j * HOP_16K;
                let lo = c.saturating_sub(320);
                let hi = (c + 320).min(len - 1);
                let all = voiced[lo..=hi].iter().all(|&v| v);
                let none = voiced[lo..=hi].iter().all(|&v| !v);
                if all {
                    f0[c]
                } else if none {
                    0.0
                } else {
                    f32::NAN
                }
            })
            .collect();
        Clip {
            name: name.to_string(),
            audio,
            truth,
        }
    }

    /// First seed of the held-out corpus ([`corpus_seeded`]): the same 27 voice x condition cells
    /// with other contours, noise and music, never used to tune the fusion's constants (those were
    /// chosen on [`corpus`], seeds 1-27).
    pub const HELD_OUT_SEED: u64 = 1_001;

    /// The standard corpus (the tuning set, seeds 1-27): every voice clean and under noise, music
    /// bleed and the octave traps.
    pub fn corpus(seconds: f32) -> Vec<Clip> {
        corpus_seeded(seconds, 1)
    }

    /// [`corpus`] with the cells seeded from `first_seed` on ([`HELD_OUT_SEED`] = the held-out set).
    pub fn corpus_seeded(seconds: f32, first_seed: u64) -> Vec<Clip> {
        let mut out = Vec::new();
        let mut seed = first_seed;
        for (vname, voice) in [("low", Voice::Low), ("mid", Voice::Mid), ("high", Voice::High), ("sing", Voice::Singing)] {
            for (cname, cond) in [
                ("clean", Condition::default()),
                (
                    "snr20",
                    Condition {
                        snr_db: Some(20.0),
                        ..Condition::default()
                    },
                ),
                (
                    "snr10",
                    Condition {
                        snr_db: Some(10.0),
                        ..Condition::default()
                    },
                ),
                (
                    "snr5",
                    Condition {
                        snr_db: Some(5.0),
                        ..Condition::default()
                    },
                ),
                (
                    "music-20",
                    Condition {
                        music_db: Some(-20.0),
                        ..Condition::default()
                    },
                ),
                (
                    "no-f1",
                    Condition {
                        missing_fundamental: true,
                        ..Condition::default()
                    },
                ),
                (
                    "creak",
                    Condition {
                        creak: true,
                        ..Condition::default()
                    },
                ),
            ] {
                if matches!(voice, Voice::Singing) && cname == "creak" {
                    continue;
                }
                out.push(render(&format!("{vname}/{cname}"), voice, cond, seconds, seed));
                seed += 1;
            }
        }
        out
    }

    /// Scores of one extractor over a corpus.
    #[derive(Clone, Debug, Default, serde::Serialize)]
    pub struct Scores {
        pub name: String,
        pub mode: &'static str,
        /// Voiced frames (truth and estimate) whose error exceeds 20 % (standard GPE), % of frames
        /// voiced in both.
        pub gpe_pct: f64,
        /// ... exceeding 50 cents, %.
        pub gross50_pct: f64,
        /// Octave errors (1200 or 2400 cents ± 150), %.
        pub octave_pct: f64,
        /// Mean absolute error of the non-gross frames, cents.
        pub fpe_cents: f64,
        /// RMS error over all frames voiced in both, cents.
        pub rmse_cents: f64,
        /// Voicing decision error, % of scored frames.
        pub vde_pct: f64,
        /// F0 frame error (Chu & Alwan): voicing errors plus gross (> 20 %) pitch errors, % of scored
        /// frames. GPE alone rewards an extractor for leaving hard frames unvoiced; FFE counts both.
        pub ffe_pct: f64,
        /// Voiced frames the estimate calls unvoiced, % of truly voiced frames.
        pub missed_voicing_pct: f64,
        /// Streaming: the same for the newest frame of each tick only (the edge of the extractor's
        /// view).
        pub missed_newest_pct: f64,
        pub frames: usize,
        /// Extractor time per call (ms): p50 / p95.
        pub call_ms_p50: f64,
        pub call_ms_p95: f64,
    }

    #[derive(Default)]
    struct Acc {
        both: usize,
        gross20: usize,
        gross50: usize,
        octave: usize,
        fine_sum: f64,
        fine_n: usize,
        sq: f64,
        vd_err: usize,
        scored: usize,
        truth_voiced: usize,
        missed: usize,
        newest_voiced: usize,
        newest_missed: usize,
    }

    impl Acc {
        fn add(&mut self, truth: &[f32], est: &[f32]) {
            for (&t, &e) in truth.iter().zip(est) {
                if t.is_nan() {
                    continue;
                }
                self.scored += 1;
                let (tv, ev) = (t > 0.0, e > 0.0 && e.is_finite());
                if tv != ev {
                    self.vd_err += 1;
                }
                if tv {
                    self.truth_voiced += 1;
                    if !ev {
                        self.missed += 1;
                    }
                }
                if tv && ev {
                    self.both += 1;
                    let c = f64::from(1200.0 * (e / t).log2());
                    let ratio = f64::from(e / t);
                    if (ratio - 1.0).abs() > 0.2 {
                        self.gross20 += 1;
                    }
                    let a = c.abs();
                    if a > 50.0 {
                        self.gross50 += 1;
                    } else {
                        self.fine_sum += a;
                        self.fine_n += 1;
                    }
                    if (a - 1200.0).abs() <= 150.0 || (a - 2400.0).abs() <= 150.0 {
                        self.octave += 1;
                    }
                    self.sq += c * c;
                }
            }
        }

        fn scores(&self, name: &str, mode: &'static str, times: &mut [f64]) -> Scores {
            let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
            times.sort_by(f64::total_cmp);
            let q = |p: f64| {
                if times.is_empty() {
                    0.0
                } else {
                    times[((times.len() - 1) as f64 * p).round() as usize]
                }
            };
            Scores {
                name: name.to_string(),
                mode,
                gpe_pct: pct(self.gross20, self.both),
                gross50_pct: pct(self.gross50, self.both),
                octave_pct: pct(self.octave, self.both),
                fpe_cents: if self.fine_n == 0 { 0.0 } else { self.fine_sum / self.fine_n as f64 },
                rmse_cents: if self.both == 0 { 0.0 } else { (self.sq / self.both as f64).sqrt() },
                vde_pct: pct(self.vd_err, self.scored),
                ffe_pct: pct(self.vd_err + self.gross20, self.scored),
                missed_voicing_pct: pct(self.missed, self.truth_voiced),
                missed_newest_pct: pct(self.newest_missed, self.newest_voiced),
                frames: self.scored,
                call_ms_p50: q(0.5),
                call_ms_p95: q(0.95),
            }
        }
    }

    /// Streaming geometry: the engine's block and context (10 ms frames).
    #[derive(Clone, Copy, Debug)]
    pub struct Streaming {
        pub block_frames: usize,
        pub context_frames: usize,
        /// Frames the decoded block lags the newest input (the plan's F0 lookahead): each tick's
        /// frames are `[end - block - lookahead, end - lookahead)`.
        pub lookahead_frames: usize,
    }

    impl Default for Streaming {
        /// 40 ms blocks with a 44-frame context and no lookahead (the SVE's low-latency plan).
        fn default() -> Self {
            Streaming {
                block_frames: 4,
                context_frames: 44,
                lookahead_frames: 0,
            }
        }
    }

    /// Score `extractor` on `clips`: offline (each clip at once) and, with `streaming`, through the
    /// F0 cache tick by tick (each frame from the tick whose new block holds it).
    pub fn run(name: &str, extractor: &mut dyn PitchExtractor, clips: &[Clip], streaming: Option<Streaming>) -> Result<Vec<Scores>, String> {
        run_conditions(name, extractor, clips, streaming).map(|(overall, _)| overall)
    }

    /// [`run`] plus the streaming scores per condition (the clip name after its `/`), named
    /// `<extractor> @ <condition>`.
    ///
    /// Streaming is scored first: on DirectML an extractor pins its shape (and RMVPE its fixed-shape
    /// plan) at its first call, which must be the stream's window as in the engine, not a whole clip
    /// (that would run every tick padded to the clip). The offline calls that follow are longer than
    /// the pin and run at their own shapes. Rows come out offline first, as before.
    pub fn run_conditions(
        name: &str,
        extractor: &mut dyn PitchExtractor,
        clips: &[Clip],
        streaming: Option<Streaming>,
    ) -> Result<(Vec<Scores>, Vec<Scores>), String> {
        let mut out = Vec::new();
        let mut per_condition: Vec<(String, Acc)> = Vec::new();
        let mut streamed = None;
        if let Some(s) = streaming {
            let mut acc = Acc::default();
            let mut times = Vec::new();
            let (lead, margin) = match extractor.receptive_frames() {
                Some((l, r)) => (l, r.max(1)),
                None => (DEFAULT_LEAD_FRAMES, DEFAULT_MARGIN_FRAMES),
            };
            for clip in clips {
                let mut cache = F0Cache::new(F0CacheConfig {
                    margin_frames: margin,
                    capacity_frames: s.context_frames * 2,
                })
                .with_lead_frames(lead);
                let total = clip.truth.len();
                let mut est = vec![0.0f32; total];
                let ctx = s.context_frames * HOP_16K;
                let block = s.block_frames * HOP_16K;
                // Ticks end at every block boundary once a whole context exists (earlier frames are
                // the stream's priming, not scored).
                let mut end = ctx;
                while end <= clip.audio.len() {
                    let t0 = Instant::now();
                    let window = cache.update(extractor, &clip.audio[end - ctx..end], (end - ctx) as u64)?;
                    times.push(t0.elapsed().as_secs_f64() * 1000.0);
                    let la = s
                        .lookahead_frames
                        .min(s.context_frames.saturating_sub(s.block_frames));
                    let first = (end - block) / HOP_16K - la;
                    let w0 = (end - ctx) / HOP_16K;
                    let last = (end / HOP_16K - la).min(total);
                    est[first..last].copy_from_slice(&window[first - w0..last - w0]);
                    end += block;
                }
                let skip = ctx / HOP_16K;
                let truth: Vec<f32> = clip
                    .truth
                    .iter()
                    .enumerate()
                    .map(|(i, &t)| if i < skip { f32::NAN } else { t })
                    .collect();
                acc.add(&truth, &est);
                // The newest frame of each tick.
                let la = s
                    .lookahead_frames
                    .min(s.context_frames.saturating_sub(s.block_frames));
                for f in (skip + s.block_frames - 1 - la..total).step_by(s.block_frames) {
                    if truth[f] > 0.0 {
                        acc.newest_voiced += 1;
                        if est[f] <= 0.0 {
                            acc.newest_missed += 1;
                        }
                    }
                }
                let condition = clip
                    .name
                    .rsplit('/')
                    .next()
                    .unwrap_or(&clip.name)
                    .to_string();
                match per_condition.iter_mut().find(|(c, _)| *c == condition) {
                    Some((_, a)) => a.add(&truth, &est),
                    None => {
                        let mut a = Acc::default();
                        a.add(&truth, &est);
                        per_condition.push((condition, a));
                    }
                }
            }
            streamed = Some(acc.scores(name, "streaming", &mut times));
        }
        let mut acc = Acc::default();
        let mut times = Vec::new();
        for clip in clips {
            let t0 = Instant::now();
            let mut f0 = extractor.f0(&clip.audio)?;
            times.push(t0.elapsed().as_secs_f64() * 1000.0);
            f0.resize(clip.truth.len(), 0.0);
            acc.add(&clip.truth, &f0);
        }
        out.push(acc.scores(name, "offline", &mut times));
        out.extend(streamed);
        let conditions = per_condition
            .into_iter()
            .map(|(c, a)| a.scores(&format!("{name} @ {c}"), "streaming", &mut []))
            .collect();
        Ok((out, conditions))
    }

    /// What [`report`] scores: the pitch bank's methods exactly as the engine builds them
    /// (`create_pitch_extractor`, so SwiftF0 / PESTO run on the CPU and fusion pairs with RMVPE, or
    /// FCPE on the CPU rung).
    pub struct ReportRequest<'a> {
        /// F0 methods to score; empty = every method whose model is installed.
        pub methods: &'a [&'a str],
        /// EP ladder rung (`cpu` / `cuda` / `dml`).
        pub ep: &'a str,
        pub opts: &'a VcSessionOptions,
        pub range: F0Range,
        /// For the legacy `base/` model folder.
        pub app_data: Option<&'a Path>,
        /// Length of each corpus clip (27 clips).
        pub seconds: f32,
        /// First seed of the corpus: 1 = the tuning set ([`corpus`]), [`HELD_OUT_SEED`] = the
        /// held-out set the fusion was never tuned on.
        pub first_seed: u64,
        pub streaming: Streaming,
        /// Also score streaming with this many frames of lookahead (0 = no extra rows).
        pub lookahead_frames: usize,
    }

    /// The A/B's result.
    #[derive(Clone, Debug, Default, serde::Serialize)]
    pub struct Report {
        /// Per method: offline, streaming, and streaming with lookahead when asked for.
        pub rows: Vec<Scores>,
        /// Streaming scores per method and condition (`<method> @ <condition>`).
        pub conditions: Vec<Scores>,
        /// Methods that could not run, with the reason (a missing model, ...).
        pub skipped: Vec<String>,
        /// Both tables as Markdown.
        pub markdown: String,
    }

    /// Every F0 method the A/B knows, in the pitch bank's quality order.
    pub const METHODS: [&str; 7] = ["fusion", "rmvpe", "swiftf0", "crepe", "fcpe", "pesto", "dio"];

    /// Score the pitch bank on the synthetic corpus (the benchmark's pitch A/B mode and the dev-rig
    /// report). A method that cannot be built is skipped with its reason, never an error.
    pub fn report(req: &ReportRequest<'_>) -> Result<Report, String> {
        let clips = corpus_seeded(req.seconds.clamp(1.0, 60.0), req.first_seed);
        let wanted: Vec<&str> = if req.methods.is_empty() {
            METHODS
                .iter()
                .copied()
                .filter(|m| *m == "dio" || *m == "fusion" || resolve_pitch_model(m, req.app_data).is_some())
                .collect()
        } else {
            req.methods.to_vec()
        };
        let mut out = Report::default();
        let lookahead = Streaming {
            lookahead_frames: req.lookahead_frames,
            ..req.streaming
        };
        for method in wanted {
            let build = || {
                create_pitch_extractor(&PitchBankRequest {
                    method,
                    ep: req.ep,
                    opts: req.opts,
                    range: req.range,
                    fcpe_path: None,
                    rmvpe_path: None,
                    app_data: req.app_data,
                    small_models_on_cpu: true,
                })
            };
            let mut bank = match build() {
                Ok(b) => b,
                Err(e) => {
                    out.skipped.push(format!("{method}: {e}"));
                    continue;
                }
            };
            let eps: Vec<&str> = bank
                .outcomes
                .iter()
                .map(|(_, o)| o.active.as_str())
                .collect();
            let name = if eps.is_empty() {
                bank.method.to_string()
            } else {
                format!("{} ({})", bank.method, eps.join("+"))
            };
            let (rows, conditions) = run_conditions(&name, bank.extractor.as_mut(), &clips, Some(req.streaming))?;
            out.rows.extend(rows);
            out.conditions.extend(conditions);
            if req.lookahead_frames > 0 {
                // A fresh extractor: the first one's DirectML pins and plans follow its offline
                // calls by now.
                drop(bank);
                let mut fresh = build()?;
                let label = format!("{name} +{} ms", req.lookahead_frames * 10);
                let rows = run(&label, fresh.extractor.as_mut(), &clips, Some(lookahead))?;
                out.rows
                    .extend(rows.into_iter().filter(|r| r.mode == "streaming"));
            }
        }
        out.conditions.sort_by(|a, b| {
            a.name
                .rsplit(" @ ")
                .next()
                .cmp(&b.name.rsplit(" @ ").next())
                .then(a.name.cmp(&b.name))
        });
        out.markdown = format!(
            "Pitch A/B: {} synthetic clips x {} s (seeds from {}), {} ms blocks, {} ms context.\n\n{}\nStreaming, per condition\n\n{}",
            clips.len(),
            req.seconds,
            req.first_seed,
            req.streaming.block_frames * 10,
            req.streaming.context_frames * 10,
            markdown(&out.rows),
            markdown(&out.conditions)
        );
        Ok(out)
    }

    /// Markdown table of scores.
    pub fn markdown(scores: &[Scores]) -> String {
        let mut s = String::from(
            "| extractor | mode | GPE 20% | >50 c | octave | FPE c | RMSE c | VDE | FFE | missed V | missed V newest | frames | call p50 ms | call p95 ms |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n",
        );
        for r in scores {
            s.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} | {:.2} | {:.1} | {:.1} | {:.2} | {:.2} | {:.2} | {:.2} | {} | {:.2} | {:.2} |\n",
                r.name,
                r.mode,
                r.gpe_pct,
                r.gross50_pct,
                r.octave_pct,
                r.fpe_cents,
                r.rmse_cents,
                r.vde_pct,
                r.ffe_pct,
                r.missed_voicing_pct,
                r.missed_newest_pct,
                r.frames,
                r.call_ms_p50,
                r.call_ms_p95
            ));
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fused(v: &[f32], vc: &[f32], p: &[f32], pc: &[f32]) -> Vec<f32> {
        let mut out = Vec::new();
        fuse(v, vc, v, p, pc, &mut out, &mut FuseScratch::default(), &FuseParams::default());
        out
    }

    /// The confirmation rule alone (the pitch extractor never voices a frame by itself here).
    fn confirm_only() -> FuseParams {
        FuseParams {
            pitch_voicing: f32::INFINITY,
            ..FuseParams::default()
        }
    }

    #[test]
    fn under_confident_voicing_is_confirmed_by_the_pitch_extractor() {
        // SwiftF0 left frames 1-4 unvoiced but saw 150 Hz; RMVPE is voiced on 151 Hz, an octave
        // off on frame 4.
        let v = [150.0, 0.0, 0.0, 0.0, 0.0];
        let cand = [150.0; 5];
        let conf = [0.95, 0.2, 0.05, 0.01, 0.6];
        let p = [150.0, 151.0, 151.0, 151.0, 302.0];
        let mut out = Vec::new();
        // RMVPE sure (0.9): frame 1 (0.2 >= 0.15) and frame 2 (0.05 >= the 0.02 floor) are
        // confirmed; frame 3 (0.01, noise level) is not; frame 4 disagrees.
        fuse(&v, &conf, &cand, &p, &[0.9; 5], &mut out, &mut FuseScratch::default(), &confirm_only());
        assert!(out[1] > 149.0 && out[2] > 149.0, "{out:?}");
        assert_eq!((out[3], out[4]), (0.0, 0.0), "{out:?}");
        // RMVPE unsure (0.3): only the 0.15 threshold confirms.
        fuse(&v, &conf, &cand, &p, &[0.3; 5], &mut out, &mut FuseScratch::default(), &confirm_only());
        assert!(out[1] > 149.0, "{out:?}");
        assert_eq!(out[2], 0.0, "{out:?}");
    }

    #[test]
    fn a_confident_pitch_extractor_can_voice_a_frame_alone() {
        // SwiftF0 left frames 2-3 unvoiced with no candidate; RMVPE is sure of them (0.95), an octave
        // up on frame 3.
        let v = [150.0, 151.0, 0.0, 0.0, 153.0];
        let vc = [1.0, 1.0, 0.0, 0.0, 1.0];
        let p = [150.0, 151.0, 152.0, 304.0, 153.0];
        let pc = [1.0, 1.0, 0.95, 0.95, 1.0];
        let mut out = Vec::new();
        fuse(&v, &vc, &[0.0; 5], &p, &pc, &mut out, &mut FuseScratch::default(), &FuseParams::default());
        assert!((out[2] - 152.0).abs() < 1e-3, "{out:?}");
        // Folded to the octave of the nearest anchor (frame 4, 153 Hz).
        assert!((out[3] - 152.0).abs() < 1e-3, "{out:?}");
        // Below its threshold (or with the rule off) those frames stay unvoiced.
        let prm = FuseParams {
            pitch_voicing: 0.99,
            ..FuseParams::default()
        };
        fuse(&v, &vc, &[0.0; 5], &p, &pc, &mut out, &mut FuseScratch::default(), &prm);
        assert_eq!((out[2], out[3]), (0.0, 0.0));
        fuse(&v, &vc, &[0.0; 5], &p, &pc, &mut out, &mut FuseScratch::default(), &confirm_only());
        assert_eq!((out[2], out[3]), (0.0, 0.0));
    }

    #[test]
    fn frames_voiced_by_the_pitch_extractor_take_the_candidate_the_anchors_support() {
        // Frames 2-4: SwiftF0 unvoiced at noise-level confidence. Frame 2: its candidate (151) and
        // RMVPE (152) agree-ish, the candidate is taken; frame 3: RMVPE reads a fifth up (228, a
        // missing fundamental) while the candidate matches the anchors; frame 4: the candidate is an
        // octave off (306) and RMVPE matches the anchors.
        let v = [150.0, 151.0, 0.0, 0.0, 0.0, 153.0, 154.0];
        let vc = [1.0, 1.0, 0.001, 0.001, 0.001, 1.0, 1.0];
        let cand = [150.0, 151.0, 151.0, 152.0, 306.0, 153.0, 154.0];
        let p = [150.0, 151.0, 152.0, 228.0, 153.0, 153.0, 154.0];
        let pc = [1.0, 1.0, 0.8, 0.8, 0.8, 1.0, 1.0];
        let mut out = Vec::new();
        fuse(&v, &vc, &cand, &p, &pc, &mut out, &mut FuseScratch::default(), &FuseParams::default());
        assert!((out[2] - 151.0).abs() < 1e-3, "{out:?}");
        assert!((out[3] - 152.0).abs() < 1e-3, "{out:?}");
        assert!((out[4] - 153.0).abs() < 1e-3, "{out:?}");
        // Without the candidate the pitch extractor's value stands (folded by octaves only).
        let prm = FuseParams {
            voiced_by_pitch_uses_candidate: false,
            ..FuseParams::default()
        };
        fuse(&v, &vc, &cand, &p, &pc, &mut out, &mut FuseScratch::default(), &prm);
        assert!((out[3] - 228.0).abs() < 1e-3, "{out:?}");
    }

    #[test]
    fn voicing_comes_from_the_voicing_extractor_unless_the_pitch_extractor_is_sure() {
        let v = [0.0, 150.0, 150.0, 0.0];
        let p = [150.0, 0.0, 151.0, 150.0];
        let pc = [1.0, 0.0, 1.0, 0.2];
        let mut out = Vec::new();
        fuse(
            &v,
            &[0.0, 1.0, 1.0, 0.0],
            &v,
            &p,
            &pc,
            &mut out,
            &mut FuseScratch::default(),
            &confirm_only(),
        );
        assert_eq!(out[0], 0.0);
        assert_eq!(out[1], 150.0);
        assert!(out[2] > 150.0 && out[2] < 151.0);
        assert_eq!(out[3], 0.0);
        // By default a sure pitch extractor (1.0) voices frame 0; an unsure one (0.2) does not.
        let out = fused(&v, &[0.0, 1.0, 1.0, 0.0], &p, &pc);
        assert!((out[0] - 150.0).abs() < 1e-3, "{out:?}");
        assert_eq!(out[3], 0.0);
    }

    #[test]
    fn agreement_is_a_confidence_weighted_mean() {
        let out = fused(&[200.0], &[1.0], &[210.0], &[3.0]);
        let want = ((200f32.log2() + 3.0 * 210f32.log2()) / 4.0).exp2();
        assert!((out[0] - want).abs() < 1e-3);
    }

    #[test]
    fn octave_disagreements_follow_the_nearest_anchor() {
        // RMVPE doubles frames 3-4, SwiftF0 halves frame 7: both resolved by the anchors.
        let v = [150.0, 151.0, 152.0, 152.0, 153.0, 153.0, 154.0, 77.0, 155.0, 156.0];
        let p = [150.0, 151.0, 152.0, 304.0, 306.0, 153.0, 154.0, 155.0, 155.0, 156.0];
        let c = [1.0f32; 10];
        let out = fused(&v, &c, &p, &c);
        assert!((out[3] - 152.0).abs() < 1e-3 && (out[4] - 153.0).abs() < 1e-3, "{out:?}");
        assert!((out[7] - 155.0).abs() < 1e-3, "{out:?}");
        // Without anchors the voicing extractor's pitch stands (by default), or the more confident
        // one.
        let out = fused(&[100.0, 100.0], &[0.9, 0.6], &[200.0, 200.0], &[0.8, 0.95]);
        assert_eq!(out, vec![100.0, 100.0]);
        let mut out = Vec::new();
        let prm = FuseParams {
            prefer_voicing_without_anchor: false,
            ..FuseParams::default()
        };
        fuse(
            &[100.0, 100.0],
            &[0.9, 0.6],
            &[100.0, 100.0],
            &[200.0, 200.0],
            &[0.8, 0.95],
            &mut out,
            &mut FuseScratch::default(),
            &prm,
        );
        assert_eq!(out, vec![100.0, 200.0]);
    }

    /// Plain stand-in extractors: a fixed contour per call.
    struct Fixed(Vec<f32>);
    impl PitchExtractor for Fixed {
        fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
            let n = mono_16k.len() / 160 + 1;
            Ok((0..n).map(|i| self.0[i % self.0.len()]).collect())
        }
        fn receptive_frames(&self) -> Option<(usize, usize)> {
            Some((3, 2))
        }
    }

    #[test]
    fn fused_extractor_runs_both_and_post_processes() {
        let v = Fixed(vec![200.0, 200.0, 0.0, 200.0, 200.0, 200.0, 200.0, 200.0]);
        let p = Fixed(vec![200.0, 400.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0]);
        let mut f = FusedPitchExtractor::new(Box::new(v), Box::new(p), FusionConfig::default());
        let out = f.f0(&vec![0.0; 1_120]).unwrap();
        assert_eq!(out.len(), 8);
        // Frame 1: octave disagreement resolved to 200; frame 2: SwiftF0 unvoiced -> 0, then the
        // median fills the single dropout.
        assert!(out.iter().all(|&x| (x - 200.0).abs() < 1e-3), "{out:?}");
        assert_eq!(f.receptive_frames(), Some((4, 3)));
        assert_eq!(f.last_confidence().len(), 8);
    }

    #[test]
    fn model_ids_and_candidates() {
        assert_eq!(pitch_model_id("swiftf0"), Some(assets::PITCH_SWIFTF0));
        assert_eq!(pitch_model_id("crepe"), Some(assets::PITCH_CREPE_TINY));
        assert_eq!(pitch_model_id("dio"), None);
        assert!(base_candidates("rmvpe").contains(&"rmvpe.onnx"));
        // A missing explicit file is an error that names it; dio needs no model.
        let opts = VcSessionOptions::default();
        let req = PitchBankRequest {
            method: "dio",
            ep: "cpu",
            opts: &opts,
            range: F0Range::Singing,
            fcpe_path: None,
            rmvpe_path: None,
            app_data: None,
            small_models_on_cpu: true,
        };
        let bank = create_pitch_extractor(&req).unwrap();
        assert_eq!(bank.method, "dio");
        let missing = Path::new("definitely-missing-rmvpe.onnx");
        let req = PitchBankRequest {
            method: "rmvpe",
            rmvpe_path: Some(missing),
            ..req
        };
        let err = create_pitch_extractor(&req).err().unwrap();
        assert!(err.contains("definitely-missing-rmvpe.onnx"), "{err}");
    }

    /// The CPU rung's voicing-assist rule: the pitch always comes from the voicing extractor, the
    /// pitch extractor only adds voicing (with its candidate), optionally only when they agree
    /// modulo octaves.
    #[test]
    fn voicing_assist_keeps_the_voicing_extractors_pitch() {
        // Frame 0: both voiced, FCPE an octave up -> SwiftF0's 150. Frame 1: agreeing -> SwiftF0's
        // value, not a mean. Frame 2: SwiftF0 unvoiced with a candidate, FCPE voiced -> candidate.
        // Frame 3: FCPE voiced, no candidate -> unvoiced. Frame 4: FCPE an octave below the
        // candidate. Frame 5: FCPE a fifth away from the candidate. Frame 6: nobody voiced.
        let v = [150.0, 152.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let vc = [0.95, 0.95, 0.1, 0.0, 0.1, 0.1, 0.001];
        let cand = [150.0, 152.0, 153.0, 0.0, 154.0, 155.0, 300.0];
        let p = [300.0, 151.0, 152.0, 152.0, 77.0, 232.0, 0.0];
        let pc = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
        let assist = FuseParams {
            voicing_assist_only: true,
            ..FuseParams::default()
        };
        let mut out = Vec::new();
        fuse(&v, &vc, &cand, &p, &pc, &mut out, &mut FuseScratch::default(), &assist);
        assert_eq!(out, vec![150.0, 152.0, 153.0, 0.0, 154.0, 155.0, 0.0]);
        // Agreement: frame 4 (an octave apart) counts only modulo octaves, frame 5 never.
        let agreeing = FuseParams {
            assist_needs_agreement: true,
            ..assist
        };
        fuse(&v, &vc, &cand, &p, &pc, &mut out, &mut FuseScratch::default(), &agreeing);
        assert_eq!(out, vec![150.0, 152.0, 153.0, 0.0, 0.0, 0.0, 0.0]);
        let folded = FuseParams {
            assist_octave_fold: true,
            ..agreeing
        };
        fuse(&v, &vc, &cand, &p, &pc, &mut out, &mut FuseScratch::default(), &folded);
        assert_eq!(out, vec![150.0, 152.0, 153.0, 0.0, 154.0, 0.0, 0.0]);
        // A confidence floor drops candidates the voicing extractor hardly saw.
        let floored = FuseParams {
            assist_min_confidence: 0.2,
            ..assist
        };
        fuse(&v, &vc, &cand, &p, &pc, &mut out, &mut FuseScratch::default(), &floored);
        assert_eq!(out, vec![150.0, 152.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
        // The CPU rung's fusion is the exact-agreement assist (the A/B's choice).
        let cpu = FusionConfig::cpu(F0Range::Singing);
        assert_eq!(cpu.params, agreeing);
        assert_eq!(cpu.range, F0Range::Singing);
        // The default (GPU) rule would have taken other values on frames 0 and 1.
        fuse(&v, &vc, &cand, &p, &pc, &mut out, &mut FuseScratch::default(), &FuseParams::default());
        assert!((out[1] - 152.0).abs() > 1e-3, "{out:?}");
    }

    /// A stand-in FCPE (waveform `[1, N]` in, F0 `[1, N / 160]` out) for the factory tests.
    fn fake_fcpe_file() -> PathBuf {
        use crate::services::neural_audio::test_onnx::{Attr, Dim, Graph, FLOAT};
        let mut g = Graph::new();
        g.input("waveform", FLOAT, &[Dim::Fixed(1), Dim::Named("n")])
            .init_i64("shape", &[3], &[1, -1, 160])
            .init_f32("offset", &[], &[120.0])
            .node("Reshape", &["waveform", "shape"], &["frames"], &[])
            .node("ReduceMean", &["frames"], &["mean"], &[Attr::Ints("axes", vec![2]), Attr::Int("keepdims", 0)])
            .node("Abs", &["mean"], &["level"], &[])
            .node("Add", &["level", "offset"], &["f0"], &[])
            .output("f0", FLOAT, &[Dim::Fixed(1), Dim::Named("t")]);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!("sigil-p3d-fcpe-{}-{nanos}.onnx", std::process::id()));
        std::fs::write(&path, g.model_bytes(17)).unwrap();
        path
    }

    /// The stand-ins mirror the contract (`params::pitch_stand_in`), with RMVPE GPU-only.
    #[test]
    fn stand_ins_follow_the_contract() {
        use crate::services::voice_changer::params::{pitch_stand_in, PITCH_EXTRACTORS};
        for m in PITCH_EXTRACTORS {
            for ep in ["cuda", "dml"] {
                assert_eq!(stand_in_on(m, ep), pitch_stand_in(m), "{m} on {ep}");
            }
            let cpu = if pitch_stand_in(m) == "rmvpe" { "fcpe" } else { pitch_stand_in(m) };
            assert_eq!(stand_in_on(m, "cpu"), cpu, "{m} on the CPU");
        }
    }

    /// A voice whose P3 pitch model is missing runs the stand-in with a note (instead of refusing to
    /// start); the strict factory (A/B, autotune, governor) keeps the error. Holds whether or not the
    /// models are installed on this machine.
    #[test]
    fn a_missing_p3_model_runs_the_stand_in_for_the_voice() {
        let fcpe = fake_fcpe_file();
        let opts = VcSessionOptions::default();
        for method in ["swiftf0", "crepe", "fusion"] {
            let req = PitchBankRequest {
                method,
                ep: "cpu",
                opts: &opts,
                range: F0Range::Speech,
                fcpe_path: Some(&fcpe),
                rmvpe_path: None,
                app_data: None,
                small_models_on_cpu: true,
            };
            let installed = match method {
                "fusion" => resolve_pitch_model("swiftf0", None).is_some(),
                m => resolve_pitch_model(m, None).is_some(),
            };
            let bank = create_pitch_extractor_with(&req, GpuLevers::default()).unwrap();
            if installed {
                assert_eq!(bank.method, method);
                continue;
            }
            assert_eq!(bank.method, "fcpe", "{method}");
            assert!(
                bank.notes[0].starts_with(&format!("{method}: the swiftf0 pitch model is not installed"))
                    || bank.notes[0].starts_with(&format!("{method}: the {method} pitch model is not installed")),
                "{:?}",
                bank.notes
            );
            assert!(bank.notes[0].ends_with("running fcpe until it is installed"), "{:?}", bank.notes);
            let err = create_pitch_extractor(&req).err().expect("the strict factory keeps the error");
            assert!(err.contains("not installed"), "{err}");
        }
        // A baseline method keeps its own error (there is nothing to stand in for it).
        let missing = Path::new("definitely-missing-fcpe.onnx");
        let req = PitchBankRequest {
            method: "fcpe",
            ep: "cpu",
            opts: &opts,
            range: F0Range::Speech,
            fcpe_path: Some(missing),
            rmvpe_path: None,
            app_data: None,
            small_models_on_cpu: true,
        };
        let err = create_pitch_extractor_with(&req, GpuLevers::default()).err().unwrap();
        assert!(err.contains("definitely-missing-fcpe.onnx"), "{err}");
        let _ = std::fs::remove_file(&fcpe);
    }

    /// The held-out corpus has the same cells as the tuning corpus, with other signals.
    #[test]
    fn the_held_out_corpus_differs_from_the_tuning_corpus() {
        let tune = ab::corpus(1.0);
        let held = ab::corpus_seeded(1.0, ab::HELD_OUT_SEED);
        assert_eq!(tune.len(), 27);
        assert_eq!(held.len(), tune.len());
        for (a, b) in tune.iter().zip(&held) {
            assert_eq!(a.name, b.name);
            assert_ne!(a.audio, b.audio, "{}", a.name);
        }
    }

    #[test]
    fn report_scores_the_requested_methods_and_skips_missing_models() {
        let opts = VcSessionOptions::default();
        let req = ab::ReportRequest {
            methods: &["dio", "swiftf0"],
            ep: "cpu",
            opts: &opts,
            range: F0Range::Speech,
            app_data: None,
            seconds: 1.0,
            first_seed: 1,
            streaming: ab::Streaming::default(),
            lookahead_frames: 2,
        };
        let r = ab::report(&req).unwrap();
        // dio: offline, streaming and streaming +20 ms; SwiftF0 has no model in a unit test.
        let dio: Vec<&ab::Scores> = r
            .rows
            .iter()
            .filter(|s| s.name.starts_with("dio"))
            .collect();
        assert_eq!(dio.len(), 3, "{:?}", r.rows);
        assert_eq!(dio[2].name, "dio +20 ms");
        assert!(r.conditions.iter().any(|c| c.name == "dio @ clean"));
        // SwiftF0 runs when its model is installed, else it is skipped with the reason.
        assert!(
            r.skipped.iter().any(|s| s.starts_with("swiftf0")) || r.rows.iter().any(|s| s.name.starts_with("swiftf0")),
            "{:?}",
            r.skipped
        );
        assert!(r.markdown.contains("| dio |"));
        assert!(serde_json::to_string(&r).unwrap().contains("gpe_pct"));
    }

    #[test]
    fn corpus_truth_is_consistent() {
        let clip = ab::render("t", ab::Voice::Mid, ab::Condition::default(), 2.0, 7);
        assert_eq!(clip.truth.len(), 2 * 16_000 / 160);
        let voiced = clip.truth.iter().filter(|t| **t > 0.0).count();
        let unvoiced = clip.truth.iter().filter(|t| **t == 0.0).count();
        assert!(voiced > 40 && unvoiced > 10, "{voiced} voiced, {unvoiced} unvoiced");
        assert!(clip
            .truth
            .iter()
            .filter(|t| **t > 0.0)
            .all(|&t| (140.0..=360.0).contains(&t)));
        assert!(clip.audio.iter().all(|a| a.is_finite() && a.abs() < 1.0));
        // Dio on the clean clip: the evaluator runs and finds a sane error level.
        let mut dio = DioPitchExtractor::new();
        let scores = ab::run("dio", &mut dio, &[clip], Some(ab::Streaming::default())).unwrap();
        assert_eq!(scores.len(), 2);
        assert!(scores[0].gpe_pct < 40.0, "{:?}", scores[0]);
        // FFE counts the voicing errors plus the gross errors, so it is never below the VDE.
        assert!(scores.iter().all(|r| r.ffe_pct >= r.vde_pct && r.ffe_pct <= r.vde_pct + r.gpe_pct), "{scores:?}");
        assert!(!ab::markdown(&scores).is_empty());
    }

    /// The pitch A/B on the dev rig: every installed extractor, offline and streaming, over the
    /// synthetic corpus; prints the table (the P3-D report and `docs/benchmarks` take it from here).
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache [SIGIL_AB_EP=cuda|dml|cpu]
    /// cargo test --release pitch_ab_report -- --ignored --nocapture`
    #[test]
    #[ignore = "dev-rig pitch A/B: needs SIGIL_TEST_MODELS_DIR with the pitch models"]
    fn pitch_ab_report() {
        let seconds: f32 = std::env::var("SIGIL_AB_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6.0);
        let ep = std::env::var("SIGIL_AB_EP").unwrap_or_else(|_| "cuda".to_string());
        // SIGIL_AB_SEED=1001 scores the held-out corpus (ab::HELD_OUT_SEED); default the tuning set.
        let first_seed: u64 = std::env::var("SIGIL_AB_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let clips = ab::corpus_seeded(seconds, first_seed);
        let opts = VcSessionOptions {
            intra_threads: Some(2),
            ..VcSessionOptions::default()
        };
        let one = VcSessionOptions {
            intra_threads: Some(1),
            ..VcSessionOptions::default()
        };
        let stream = Some(ab::Streaming::default());
        let mut all = Vec::new();
        let mut conditions = Vec::new();
        let mut run = |name: &str, e: &mut dyn PitchExtractor| match ab::run_conditions(name, e, &clips, stream) {
            Ok((s, c)) => {
                all.extend(s);
                conditions.extend(c);
            }
            Err(err) => eprintln!("{name}: {err}"),
        };
        let mut dio = DioPitchExtractor::new();
        run("dio", &mut dio);
        if let Some(p) = test_model("pitch-fcpe") {
            if let Ok(mut e) = FcpePitchExtractor::load(&p.to_string_lossy(), "cpu", &one) {
                run("fcpe (cpu)", &mut e);
            }
        }
        if let Some(p) = test_model("pitch-rmvpe") {
            match RmvpePitchExtractor::load(&p.to_string_lossy(), &ep, &opts) {
                Ok(mut e) => run(&format!("rmvpe ({ep})"), &mut e),
                Err(e) => eprintln!("rmvpe: {e}"),
            }
        }
        if let Some(p) = test_model("pitch-swiftf0") {
            let mut e = SwiftF0PitchExtractor::load(&p.to_string_lossy(), "cpu", &one).unwrap();
            run("swiftf0 (cpu)", &mut e);
            for t in [0.7f32, 0.5] {
                let mut e = SwiftF0PitchExtractor::load(&p.to_string_lossy(), "cpu", &one)
                    .unwrap()
                    .with_voicing_threshold(t);
                run(&format!("swiftf0 (cpu, voicing {t})"), &mut e);
            }
            if let Ok(mut e) = SwiftF0PitchExtractor::load(&p.to_string_lossy(), &ep, &opts) {
                run(&format!("swiftf0 ({ep})"), &mut e);
            }
        }
        if let Some(p) = test_model("pitch-crepe-tiny") {
            if let Ok(mut e) = CrepePitchExtractor::load(&p.to_string_lossy(), &ep, &opts) {
                run(&format!("crepe-tiny ({ep})"), &mut e);
            }
        }
        if let Some(p) = test_model("pitch-pesto") {
            match PestoPitchExtractor::load(&p.to_string_lossy(), "cpu", &one) {
                Ok(mut e) => run("pesto (cpu)", &mut e),
                Err(e) => eprintln!("pesto: {e}"),
            }
        }
        if let (Some(sw), Some(rm)) = (test_model("pitch-swiftf0"), test_model("pitch-rmvpe")) {
            let v = SwiftF0PitchExtractor::load(&sw.to_string_lossy(), "cpu", &one).unwrap();
            if let Ok(p) = RmvpePitchExtractor::load(&rm.to_string_lossy(), &ep, &opts) {
                let mut f = FusedPitchExtractor::with_confidence(Box::new(v), Box::new(p), FusionConfig::default());
                run(&format!("fusion swiftf0+rmvpe ({ep})"), &mut f);
            }
        }
        if let (Some(sw), Some(fc)) = (test_model("pitch-swiftf0"), test_model("pitch-fcpe")) {
            let v = SwiftF0PitchExtractor::load(&sw.to_string_lossy(), "cpu", &one).unwrap();
            if let Ok(p) = FcpePitchExtractor::load(&fc.to_string_lossy(), "cpu", &one) {
                let mut f = FusedPitchExtractor::with_voicing_confidence(Box::new(v), Box::new(p), FusionConfig::cpu(F0Range::Speech));
                run("fusion swiftf0+fcpe (cpu)", &mut f);
            }
        }
        // What 20 ms of F0 lookahead (the plan's `lookahead_frames`) buys in streaming.
        let lookahead = Some(ab::Streaming {
            lookahead_frames: 2,
            ..ab::Streaming::default()
        });
        let mut la_rows = Vec::new();
        let mut run_la = |name: &str, e: &mut dyn PitchExtractor| match ab::run(name, e, &clips, lookahead) {
            Ok(s) => la_rows.extend(s.into_iter().filter(|r| r.mode == "streaming")),
            Err(err) => eprintln!("{name}: {err}"),
        };
        if let Some(p) = test_model("pitch-swiftf0") {
            let mut e = SwiftF0PitchExtractor::load(&p.to_string_lossy(), "cpu", &one).unwrap();
            run_la("swiftf0 (cpu) +20 ms lookahead", &mut e);
        }
        if let Some(p) = test_model("pitch-rmvpe") {
            if let Ok(mut e) = RmvpePitchExtractor::load(&p.to_string_lossy(), &ep, &opts) {
                run_la(&format!("rmvpe ({ep}) +20 ms lookahead"), &mut e);
            }
        }
        if let (Some(sw), Some(rm)) = (test_model("pitch-swiftf0"), test_model("pitch-rmvpe")) {
            let v = SwiftF0PitchExtractor::load(&sw.to_string_lossy(), "cpu", &one).unwrap();
            if let Ok(p) = RmvpePitchExtractor::load(&rm.to_string_lossy(), &ep, &opts) {
                let mut f = FusedPitchExtractor::with_confidence(Box::new(v), Box::new(p), FusionConfig::default());
                run_la(&format!("fusion swiftf0+rmvpe ({ep}) +20 ms lookahead"), &mut f);
            }
        }
        all.extend(la_rows);
        println!("pitch A/B: {} clips x {seconds} s, seeds from {first_seed}\n{}", clips.len(), ab::markdown(&all));
        // Per condition, streaming (what the engine decodes with).
        conditions.sort_by(|a, b| {
            a.name
                .rsplit(" @ ")
                .next()
                .cmp(&b.name.rsplit(" @ ").next())
                .then(a.name.cmp(&b.name))
        });
        println!("streaming, per condition\n{}", ab::markdown(&conditions));
    }

    /// Raw outputs of both extractors for one fused call (for replaying the fusion rule).
    struct RawCall {
        v: Vec<f32>,
        vc: Vec<f32>,
        vk: Vec<f32>,
        p: Vec<f32>,
        pc: Vec<f32>,
    }

    /// Runs a fused extractor and records what both parts returned on every call.
    struct Recorder {
        inner: FusedPitchExtractor,
        calls: Vec<RawCall>,
    }

    impl PitchExtractor for Recorder {
        fn f0(&mut self, mono_16k: &[f32]) -> Result<Vec<f32>, String> {
            let out = self.inner.f0(mono_16k)?;
            let i = &self.inner;
            let n = i.v.len().min(i.p.len());
            self.calls.push(RawCall {
                v: i.v[..n].to_vec(),
                vc: i.vc[..n].to_vec(),
                vk: i.vk[..n].to_vec(),
                p: i.p[..n].to_vec(),
                pc: i.pc[..n].to_vec(),
            });
            Ok(out)
        }
        fn receptive_frames(&self) -> Option<(usize, usize)> {
            self.inner.receptive_frames()
        }
    }

    /// What a replay returns for a recorded call.
    #[derive(Clone, Copy)]
    enum ReplayOf {
        Voicing,
        Pitch,
        Fused(FusionConfig),
    }

    /// Replays recorded calls in order (the A/B's call sequence is fixed by its geometry, not by the
    /// audio), so one model pass scores any number of fusion variants.
    struct Replay<'a> {
        calls: &'a [RawCall],
        k: usize,
        of: ReplayOf,
        receptive: Option<(usize, usize)>,
        post: F0Post,
        scratch: FuseScratch,
    }

    impl PitchExtractor for Replay<'_> {
        fn f0(&mut self, _mono_16k: &[f32]) -> Result<Vec<f32>, String> {
            let c = self
                .calls
                .get(self.k)
                .ok_or("the replay ran out of recorded calls")?;
            self.k += 1;
            Ok(match self.of {
                ReplayOf::Voicing => c.v.clone(),
                ReplayOf::Pitch => c.p.clone(),
                ReplayOf::Fused(config) => {
                    let mut out = Vec::new();
                    fuse(&c.v, &c.vc, &c.vk, &c.p, &c.pc, &mut out, &mut self.scratch, &config.params);
                    self.post.process(&mut out, &fusion_post_config(&config));
                    out
                }
            })
        }
        fn receptive_frames(&self) -> Option<(usize, usize)> {
            self.receptive
        }
    }

    /// Dev rig: SwiftF0 end alignment (standalone, off and on) and the fusion rule's tunables, the
    /// latter scored from one recorded model pass (offline, streaming, and streaming with 20 ms of
    /// lookahead). On `SIGIL_AB_EP=cpu` the fusion pairs SwiftF0 with FCPE, as the pitch bank does.
    /// `SIGIL_TEST_MODELS_DIR=%LOCALAPPDATA%\sigil-dev-cache [SIGIL_AB_EP=cuda|dml]
    /// cargo test --release fusion_sweep -- --ignored --nocapture`
    #[test]
    #[ignore = "dev-rig fusion sweep: needs SIGIL_TEST_MODELS_DIR with SwiftF0 and RMVPE"]
    fn fusion_sweep() {
        let (Some(sw), Some(rm)) = (test_model("pitch-swiftf0"), test_model("pitch-rmvpe")) else {
            eprintln!("SwiftF0 / RMVPE not found; skipped");
            return;
        };
        let seconds: f32 = std::env::var("SIGIL_AB_SECONDS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(6.0);
        let ep = std::env::var("SIGIL_AB_EP").unwrap_or_else(|_| "cuda".to_string());
        // SIGIL_AB_SEED=1001 scores the held-out corpus (ab::HELD_OUT_SEED); default the tuning set.
        let first_seed: u64 = std::env::var("SIGIL_AB_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(1);
        let clips = ab::corpus_seeded(seconds, first_seed);
        let opts = VcSessionOptions {
            intra_threads: Some(2),
            ..VcSessionOptions::default()
        };
        let one = VcSessionOptions {
            intra_threads: Some(1),
            ..VcSessionOptions::default()
        };
        // `initial`: the rule before the A/B (confirm at 0.3 only, confidence decides unanchored
        // disagreements, no voicing by the pitch extractor alone).
        let initial = FuseParams {
            confirm_confidence: 0.3,
            confirm_floor: 0.3,
            pitch_voicing: f32::INFINITY,
            voiced_by_pitch_uses_candidate: false,
            prefer_voicing_without_anchor: false,
            ..FuseParams::default()
        };
        let variants: Vec<(&str, FuseParams)> = vec![
            ("initial", initial),
            (
                "confirm 0.15 / floor 0.02",
                FuseParams {
                    confirm_confidence: 0.15,
                    confirm_floor: 0.02,
                    ..initial
                },
            ),
            (
                "+ prefer voicing unanchored",
                FuseParams {
                    confirm_confidence: 0.15,
                    confirm_floor: 0.02,
                    prefer_voicing_without_anchor: true,
                    ..initial
                },
            ),
            ("default (+ pitch voices at 0.3)", FuseParams::default()),
            (
                "default, pitch voices at 0.5",
                FuseParams {
                    pitch_voicing: 0.5,
                    ..FuseParams::default()
                },
            ),
            (
                "default, pitch voices at 0.1",
                FuseParams {
                    pitch_voicing: 0.1,
                    ..FuseParams::default()
                },
            ),
            (
                "default, agree 50 c",
                FuseParams {
                    agree_cents: 50.0,
                    ..FuseParams::default()
                },
            ),
            (
                "default, anchors 12",
                FuseParams {
                    anchor_range: 12,
                    ..FuseParams::default()
                },
            ),
            (
                "default, value from the pitch extractor",
                FuseParams {
                    voiced_by_pitch_uses_candidate: false,
                    ..FuseParams::default()
                },
            ),
            (
                "voicing assist",
                FuseParams {
                    voicing_assist_only: true,
                    ..FuseParams::default()
                },
            ),
            (
                "voicing assist, octave agreement",
                FuseParams {
                    voicing_assist_only: true,
                    assist_needs_agreement: true,
                    assist_octave_fold: true,
                    ..FuseParams::default()
                },
            ),
            (
                "voicing assist, agreement",
                FuseParams {
                    voicing_assist_only: true,
                    assist_needs_agreement: true,
                    ..FuseParams::default()
                },
            ),
            (
                "voicing assist, agreement 50 c",
                FuseParams {
                    voicing_assist_only: true,
                    assist_needs_agreement: true,
                    agree_cents: 50.0,
                    ..FuseParams::default()
                },
            ),
            (
                "voicing assist, agreement, confidence 0.02",
                FuseParams {
                    voicing_assist_only: true,
                    assist_needs_agreement: true,
                    assist_min_confidence: 0.02,
                    ..FuseParams::default()
                },
            ),
            (
                "voicing assist, agreement 50 c, confidence 0.15",
                FuseParams {
                    voicing_assist_only: true,
                    assist_needs_agreement: true,
                    agree_cents: 50.0,
                    assist_min_confidence: 0.15,
                    ..FuseParams::default()
                },
            ),
        ];
        let modes = [
            ("", ab::Streaming::default()),
            (
                " +20 ms",
                ab::Streaming {
                    lookahead_frames: 2,
                    ..ab::Streaming::default()
                },
            ),
        ];
        let mut rows = Vec::new();
        let mut conditions = Vec::new();
        // SwiftF0 on its own (its own receptive field, so its own slice lengths), end alignment off
        // and on.
        for align in [false, true] {
            let mut e = SwiftF0PitchExtractor::load(&sw.to_string_lossy(), "cpu", &one)
                .unwrap()
                .with_end_alignment(align);
            for (suffix, mode) in modes {
                let r = ab::run(&format!("swiftf0 cpu standalone, align {align}{suffix}"), &mut e, &clips, Some(mode)).unwrap();
                rows.extend(
                    r.into_iter()
                        .filter(|s| suffix.is_empty() || s.mode == "streaming"),
                );
            }
        }
        {
            // SwiftF0 end-aligned (its default) inside the fusion.
            let v = SwiftF0PitchExtractor::load(&sw.to_string_lossy(), "cpu", &one).unwrap();
            // The CPU rung pairs SwiftF0 with FCPE (as the pitch bank does).
            let fused = if ep == "cpu" {
                let Some(fc) = test_model("pitch-fcpe") else {
                    eprintln!("FCPE not found; skipped");
                    return;
                };
                let p = FcpePitchExtractor::load(&fc.to_string_lossy(), "cpu", &one).unwrap();
                FusedPitchExtractor::with_voicing_confidence(Box::new(v), Box::new(p), FusionConfig::cpu(F0Range::Speech))
            } else {
                let p = match RmvpePitchExtractor::load(&rm.to_string_lossy(), &ep, &opts) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("rmvpe on {ep}: {e}");
                        return;
                    }
                };
                FusedPitchExtractor::with_confidence(Box::new(v), Box::new(p), FusionConfig::default())
            };
            let receptive = fused.receptive_frames();
            let mut rec = Recorder {
                inner: fused,
                calls: Vec::new(),
            };
            ab::run("fusion (recorded)", &mut rec, &clips, Some(ab::Streaming::default())).unwrap();
            let calls = rec.calls;
            let tag = "aligned";
            let pitch_name = if ep == "cpu" { "fcpe cpu".to_string() } else { format!("rmvpe {ep}") };
            let mut jobs: Vec<(String, ReplayOf)> = vec![
                (format!("swiftf0 cpu {tag}"), ReplayOf::Voicing),
                (format!("{pitch_name} ({tag} run)"), ReplayOf::Pitch),
            ];
            for (name, params) in &variants {
                jobs.push((
                    format!("fusion {tag}, {name}"),
                    ReplayOf::Fused(FusionConfig {
                        params: *params,
                        ..FusionConfig::default()
                    }),
                ));
            }
            for (name, of) in jobs {
                for (suffix, mode) in modes {
                    let mut r = Replay {
                        calls: &calls,
                        k: 0,
                        of,
                        receptive,
                        post: F0Post::new(),
                        scratch: FuseScratch::default(),
                    };
                    let (overall, per) = ab::run_conditions(&format!("{name}{suffix}"), &mut r, &clips, Some(mode)).unwrap();
                    assert_eq!(r.k, calls.len(), "replay call count");
                    rows.extend(
                        overall
                            .into_iter()
                            .filter(|s| suffix.is_empty() || s.mode == "streaming"),
                    );
                    if suffix.is_empty() {
                        conditions.extend(per);
                    }
                }
            }
        }
        println!("fusion sweep ({ep}): {} clips x {seconds} s, seeds from {first_seed}\n{}", clips.len(), ab::markdown(&rows));
        conditions.sort_by(|a, b| {
            a.name
                .rsplit(" @ ")
                .next()
                .cmp(&b.name.rsplit(" @ ").next())
                .then(a.name.cmp(&b.name))
        });
        println!("streaming, per condition\n{}", ab::markdown(&conditions));
    }
}
