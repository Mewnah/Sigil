//! The `quality` run mode (P3-F; plan T5): the spec's input, led by `lead_s` of silence, is
//! converted through the production harness exactly like an offline run while the model's 16 kHz
//! input stream is recorded; the same model then converts that recorded stream offline (the
//! ceiling, `ceiling.rs`), and the streamed output is scored against the ceiling:
//! - alignment from the timeline ([`timeline_lag_16k`]: the ceiling's absolute start, where the
//!   input entered the recorded stream, the probed harness latency), cross-checked on energy
//!   envelopes; a search (envelopes, then waveforms) only when the timeline cannot be derived.
//!   Two renditions of a voice with time-anchored noise need not correlate as waveforms, and an
//!   envelope search alone can lock onto the wrong syllable;
//! - MCD, voiced F0 RMSE / GPE / V/UV (one [`F0Analyzer`] for both), CAM++ speaker cosine,
//!   DNSMOS P.835 of both (and their difference), clicks and artifact events (impulses the ceiling
//!   does not have; clicks = the excess of those in the stream's join windows over the artifact
//!   rate away from the joins, plus two standard deviations), the spectral-flux seam ratio;
//! - CER through a Sigil STT engine when the run has `stt`: the input's transcript (or the given
//!   reference text) against the stream's and the ceiling's (`cer_delta_pts`, T5: ≤ 1 pt).
//!
//! The scoring region is the clip's speech (100 ms around the ceiling's first and last speech
//! frame), so the lead silence and the harness's tail do not count.
//!
//! [`Scorers`] and [`score_pair`] are shared with the autotuner's fidelity check.
//!
//! # Scorer v2 (P3B-B, ADR 0009; `scoring.analyzer: v2`, `v2+ablations`)
//! The v1 run above is unchanged (the default). A v2 run:
//! - streams a twin of the spec's voice with the silence skip off (both renditions are gated the
//!   same way: neither) and, for template voices, the engine probe's fed-F0 tap on (a direct build,
//!   `rvc_build::ProbeChannel`), through its own loop over the production harness that records every
//!   join's SOLA offset ([`stream_pass`]);
//! - scores the clip at every lead of `fidelity.offsets_s` (default: `lead_s`, `lead_s + 0.27 s`
//!   and `lead_s + 0.61 s`: three anchored-noise realizations, because one clip's gross pitch error
//!   and fine pitch error swing by a factor of two between two leads; fewer are allowed with a
//!   note), headline values the medians and the key metrics' spread;
//! - renders the ceilings (`long_shape` = A, `b`, `whole`; all three for template voices, A alone
//!   otherwise): `ceiling.mode` is the primary, against which the v1 metrics are computed. The
//!   ceilings `ceiling.mode` names (and the whole ceiling when the ablations run) must render, else
//!   the run fails; the others, which template voices get for the floors, the identity row and the
//!   onset envelopes, are optional: when one cannot be rendered (a speech region over 60 s, a
//!   `lead_s` under 0.4 s, a recording that does not match the input, no fp32 content encoder on the
//!   CPU EP) the run notes why and leaves that rendition's rows out. And, at the first offset, the
//!   floors:
//!   ceiling A against itself with its blocks shifted by 100 ms (the primary MCD floor), with
//!   another noise seed (an upper bound) and against ceiling B (`mcd_offline_floor_db`);
//! - adds the v2 metrics (`quality::QUALITY_V2_SCALARS`): exact CAM++ on speech frames, the
//!   target-speaker set similarity (`fidelity.reference_clips`), the F0 split, the fed-F0 metrics
//!   against the ceiling's fed F0 and against whole-clip analysis of the input, binomial clicks on
//!   join windows placed for the model's delay and the SOLA centre (wide, and narrow at both
//!   crossfade boundaries), the envelope MCD, the offset-corrected MCD and DNSMOS of every rendition;
//! - with `v2+ablations` (or `SIGIL_BENCH_ABLATIONS=1` on a `v2` run), the oracle ablations at the
//!   first offset (`ablation.rs`).
//!
//! The v1 metrics stay in every v2 report (medians over the offsets of the v2 renditions); the
//! analyzer version and the scorers' sha256 are stamped into the details.
//!
//! # Frozen by P3B-S0 (signature; P3B-B implements)
//! [`t5_shortfall_v2`]`(&metrics) -> f64`: how far a configuration's fidelity metrics fall short
//! of T5, in units of each limit (0 = every measured check passes). The autotuner's verify stage
//! delegates to it. On v2 metrics (medians over offsets) it scores the T5 v2 rows (ADR 0009); on a
//! v1 run's metrics the P3 formula.

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::ablation;
use super::ceiling::{
    ceiling, long_shape, render_blocks, render_whole, CallSpan, CeilingOutput, LongShapeOpts, OfflineAnalysis, OfflineVoice, RenderOpts, StreamRecorder,
    WHOLE_MAX_REGION_16K, WHOLE_MIN_START_S,
};
use super::dsp::align;
use super::engine::{build_engine, lock, probe_algorithmic_latency, Engine, ModelTimes};
use super::metrics::Metrics;
use super::multi_clip;
use super::offline::{run_offline, OfflineInput};
use super::quality::{
    binomial_at_least, binomial_clicks, click_events, click_report, envelope_mcd_db, f0_errors_masked, f0_errors_v2, fed_f0_errors, mcd_db,
    seam_flux_ratio, speech_frames, unmatched, vuv_error_tolerant, ClickConfig, ClickReport, F0Analyzer, F0ErrorsV2, FedF0Errors, QualityMetrics, QUALITY_V2_SPREAD,
};
use super::report::RunOutcome;
use super::rvc_build::{self, ProbeChannel, RvcPlan};
use super::scorers::{Dnsmos, SpeakerEmbedder, Transcriber};
use super::signals::{channel, read_wav, resample, to_mono};
use super::spec::{BenchSpec, EngineSpec, OfflineRun, QualityRun, ScoringSpec, StreamSpec};
use super::vc_rt::VcTick;
use super::Prepared;
use crate::services::neural_audio::vc::engine::options::{F0Range, PrecisionMode};
use crate::services::neural_audio::vc::engine::probe::{ProbeCounts, ProbeSetup};
use crate::services::neural_audio::vc::manifest::SlotFormat;
use crate::services::neural_audio::VcStreamHarness;

/// The models a run scores with (loaded once per run).
pub(crate) struct Scorers {
    pub f0: F0Analyzer,
    /// The v1 speaker scorer (`bench/campplus`, or the spec's file).
    pub speaker: Option<SpeakerEmbedder>,
    pub dnsmos: Option<Dnsmos>,
    pub click: ClickConfig,
    pub notes: Vec<String>,
    // ---- P3B-B ----
    /// The v2 speaker scorer (`bench/campplus-exact`; v2 runs only).
    pub speaker_v2: Option<SpeakerEmbedder>,
    pub analyzer: AnalyzerOpts,
}

/// The scorer set a run asks for (`scoring.analyzer`): `v1` (default), `v2`, `v2+ablations`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub(crate) struct AnalyzerOpts {
    pub v2: bool,
    pub ablations: bool,
}

/// Parse `scoring.analyzer`.
pub(crate) fn analyzer_opts(s: &str) -> Result<AnalyzerOpts, String> {
    let mut parts = s.split('+').map(|t| t.trim().to_ascii_lowercase());
    let mut o = match parts.next().as_deref() {
        None | Some("") | Some("v1") => AnalyzerOpts::default(),
        Some("v2") => AnalyzerOpts {
            v2: true,
            ablations: false,
        },
        Some(other) => return Err(format!("scoring.analyzer '{other}' is not v1 or v2")),
    };
    for t in parts {
        match t.as_str() {
            "ablations" | "ablate" if o.v2 => o.ablations = true,
            "ablations" | "ablate" => return Err("scoring.analyzer: the ablations need v2 (v2+ablations)".into()),
            other => return Err(format!("scoring.analyzer: unknown option '{other}' (v2+ablations)")),
        }
    }
    Ok(o)
}

/// The environment switch of the oracle ablations on a v2 quality run: `SIGIL_BENCH_ABLATIONS=1`
/// (the same as `scoring.analyzer: v2+ablations`).
pub(crate) const ABLATIONS_ENV: &str = "SIGIL_BENCH_ABLATIONS";

fn ablations_from_env() -> bool {
    std::env::var(ABLATIONS_ENV).is_ok_and(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "on" | "yes"))
}

/// A ceiling of `ceiling.mode`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum CeilingMode {
    LongShape,
    B,
    Whole,
}

impl CeilingMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CeilingMode::LongShape => "long_shape",
            CeilingMode::B => "b",
            CeilingMode::Whole => "whole",
        }
    }
}

/// Parse `ceiling.mode`: `long_shape`, `b` or `whole` (the spec takes one; joined by `+` or `,` the
/// first is the primary ceiling and the others are rendered too).
pub(crate) fn ceiling_modes(s: &str) -> Result<Vec<CeilingMode>, String> {
    let mut out = Vec::new();
    for t in s.split(['+', ',']).map(|t| t.trim().to_ascii_lowercase()).filter(|t| !t.is_empty()) {
        let m = match t.as_str() {
            "long_shape" | "long-shape" | "a" => CeilingMode::LongShape,
            "b" | "ceiling_b" => CeilingMode::B,
            "whole" => CeilingMode::Whole,
            other => return Err(format!("ceiling.mode '{other}' is not long_shape, b or whole")),
        };
        if !out.contains(&m) {
            out.push(m);
        }
    }
    if out.is_empty() {
        out.push(CeilingMode::LongShape);
    }
    Ok(out)
}

impl Scorers {
    /// Load what `s` asks for; a scorer whose model is missing is skipped with a note (its metrics
    /// are then absent, and a threshold on them fails as "not measured").
    pub fn load(s: &ScoringSpec, range: F0Range) -> Scorers {
        let ep = s.analysis_ep.trim().to_ascii_lowercase();
        let f0_model = Some(Path::new(s.f0_model.trim())).filter(|p| !p.as_os_str().is_empty());
        let f0 = F0Analyzer::new(&s.f0_method, f0_model, &ep, range);
        let mut notes = f0.notes.clone();
        let off = |m: &str| m.trim().eq_ignore_ascii_case("off");
        let analyzer = analyzer_opts(&s.analyzer).unwrap_or_else(|e| {
            notes.push(format!("{e}; scoring with v1"));
            AnalyzerOpts::default()
        });
        // `legacy` names the v1 file in both scorer sets.
        let v1_model = if s.speaker_model.trim().eq_ignore_ascii_case("legacy") { "" } else { s.speaker_model.as_str() };
        let speaker_v2 = if analyzer.v2 && !off(&s.speaker_model) {
            match SpeakerEmbedder::resolve_v2(&s.speaker_model, &ep) {
                Ok(Some(e)) => Some(e),
                Ok(None) => {
                    notes.push("speaker similarity v2: CAM++ exact is not installed (model bench/campplus-exact); not measured".into());
                    None
                }
                Err(e) => {
                    notes.push(format!("speaker similarity v2: {e}; not measured"));
                    None
                }
            }
        } else {
            None
        };
        let speaker = if off(&s.speaker_model) {
            None
        } else {
            match SpeakerEmbedder::resolve(v1_model, &ep) {
                Ok(Some(e)) => Some(e),
                Ok(None) => {
                    notes.push("speaker similarity: CAM++ is not installed (model bench/campplus); not measured".into());
                    None
                }
                Err(e) => {
                    notes.push(format!("speaker similarity: {e}; not measured"));
                    None
                }
            }
        };
        let dnsmos = if off(&s.dnsmos_model) {
            None
        } else {
            match Dnsmos::resolve(&s.dnsmos_model, &ep) {
                Ok(Some(d)) => Some(d),
                Ok(None) => {
                    notes.push("DNSMOS: not installed (model bench/dnsmos-p835); not measured".into());
                    None
                }
                Err(e) => {
                    notes.push(format!("DNSMOS: {e}; not measured"));
                    None
                }
            }
        };
        Scorers {
            f0,
            speaker,
            dnsmos,
            click: ClickConfig {
                threshold: s.click_threshold,
                ..ClickConfig::default()
            },
            notes,
            speaker_v2,
            analyzer,
        }
    }

    /// Which scorers run (for the report).
    pub fn describe(&self) -> serde_json::Value {
        serde_json::json!({
            "f0_method": self.f0.method,
            "speaker_model": self.speaker.as_ref().map(|s| s.path.display().to_string()),
            "speaker_ep": self.speaker.as_ref().map(|s| s.outcome.active.clone()),
            "dnsmos_model": self.dnsmos.as_ref().map(|d| d.path.display().to_string()),
            "dnsmos_ep": self.dnsmos.as_ref().map(|d| d.outcome.active.clone()),
            "click_threshold": self.click.threshold,
        })
    }

    /// [`describe`](Self::describe) plus the v2 stamps: the analyzer version (method, model sha256,
    /// probe fingerprint) and each speaker model's id and sha256.
    pub fn describe_v2(&mut self) -> serde_json::Value {
        let mut v = self.describe();
        if let Some(o) = v.as_object_mut() {
            o.insert("analyzer".into(), serde_json::to_value(self.analyzer).unwrap_or_default());
            o.insert("f0_analyzer_version".into(), self.f0.version().into());
            o.insert(
                "f0_model".into(),
                self.f0.model_path().map(|p| p.display().to_string()).into(),
            );
            o.insert("speaker_v1".into(), self.speaker.as_ref().map(SpeakerEmbedder::describe).unwrap_or_default());
            o.insert("speaker_v2".into(), self.speaker_v2.as_ref().map(SpeakerEmbedder::describe).unwrap_or_default());
            o.insert(
                "dnsmos_sha256".into(),
                self.dnsmos
                    .as_ref()
                    .and_then(|d| super::scorers::file_sha256(&d.path).ok())
                    .into(),
            );
        }
        v
    }
}

/// How the stream was placed against the ceiling.
#[derive(Debug, Clone, Copy, Serialize)]
pub(crate) struct PairAlignment {
    /// Stream sample (16 kHz) minus ceiling sample of the same instant.
    pub lag: f64,
    /// `timeline` (from the recorded absolute positions and the probed harness latency: exact),
    /// `waveform` or `envelope` (searched, when the timeline is unknown).
    pub method: &'static str,
    /// The energy-envelope search's answer and its correlation (a cross-check of the timeline).
    pub envelope_lag: Option<f64>,
    pub envelope_correlation: Option<f64>,
}

/// Stream and ceiling scored against each other.
#[derive(Debug, Clone, Serialize)]
pub(crate) struct PairScore {
    pub quality: QualityMetrics,
    pub ceiling_dnsmos_ovrl: Option<f64>,
    pub ceiling_seam_flux_ratio: Option<f64>,
    pub alignment: PairAlignment,
    /// Scored span of the ceiling (16 kHz indices).
    pub region: (usize, usize),
    pub edge_events: usize,
    /// Where the artifact events sit on the stream's block grid (phase-locked = seam clicks).
    pub artifact_phase: ClickReport,
    /// Artifact events inside the join windows, the number expected there from the rate away
    /// from the joins, and the share of the region the windows cover (`clicks` is the excess).
    pub join_artifacts: usize,
    pub join_expected: f64,
    pub join_coverage: f64,
    /// P3B-B: the artifact events (ceiling 16 kHz index, in the region), for the v2 click tests.
    #[serde(skip)]
    pub artifacts: Vec<f64>,
}

/// Inputs of [`score_pair`].
pub(crate) struct PairInput<'a> {
    /// The streamed output (mono) at `stream_rate`.
    pub stream: &'a [f32],
    pub stream_rate: u32,
    /// The ceiling at `ceiling_rate`; `ceiling_start_16k` is the absolute 16 kHz position of its
    /// first sample.
    pub ceiling: &'a [f32],
    pub ceiling_rate: u32,
    pub ceiling_start_16k: u64,
    /// Model calls of the stream (absolute 16 kHz), for the join positions.
    pub calls: &'a [CallSpan],
    /// Stream block period in stream samples (the seam statistic), and the ceiling's.
    pub stream_period: usize,
    pub ceiling_period: usize,
    /// Crossfade (ms): how far after a call's first returned sample its join can click.
    pub join_window_ms: f64,
    /// Longest offset searched (s).
    pub max_offset_s: f64,
    /// The stream-minus-ceiling offset (16 kHz samples) known from the timeline
    /// ([`timeline_lag_16k`]); `None` searches for it.
    pub expected_lag_16k: Option<f64>,
}

// T5 v2 limits (ADR 0009, pre-registered before the engine workstreams' final re-score). The
// shortfall uses the Low-latency values (the lenient ones); the Quality rows are the scorecard's.
// Seams: 0 clicks under both binomial tests (wide and narrow windows).

/// Identity: SIM(stream, target set) − SIM(whole ceiling, same set) ≥ −0.02 (Low latency).
pub(crate) const T5_SET_SIM_DELTA_MIN: f64 = -0.02;
/// Identity at Quality: ≥ −0.01.
#[allow(dead_code)] // The scorecard's row (P3B-I0); the shortfall uses Low latency's.
pub(crate) const T5_SET_SIM_DELTA_MIN_QUALITY: f64 = -0.01;
/// The advisory cosine against the ceiling when a run has no target set (the P3 number).
pub(crate) const T5_COSINE_ADVISORY_MIN: f64 = 0.97;
/// Fine pitch error of the fed F0 against whole-clip analysis of the input (cents). A fixed,
/// pre-registered number: the plan's rule (15 cents, or the analyzer floor + 5 when the floor
/// exceeds 10 cents) was applied once, to the corpus floor measured before the engine workstreams'
/// re-score (ADR 0009: `whole_fed_input_fpe_cents` with the RMVPE tracker, below 10 cents), so the
/// limit is 15. The scorecard never derives it from a run's own floor: a tracker whose whole-clip
/// track disagrees with the analyzer would otherwise loosen its own limit. (The autotuner's
/// shortfall, [`t5_shortfall_v2`], does read such a tracker net of its floor: see
/// [`T5_TRACKER_FLOOR_CENTS`].)
pub(crate) const T5_FPE_MAX_CENTS: f64 = 15.0;
/// A run whose own whole-clip track (`whole_fed_input_fpe_cents`) disagrees with the analyzer by
/// more than this (cents, the plan's "floor exceeds 10 cents") runs a tracker that is not the
/// analyzer's (FCPE on the CPU preset reads 34.5 cents over the whole clip). Its pitch rows measure
/// that disagreement more than streaming, so [`t5_shortfall_v2`] charges only what exceeds the
/// run's own floor, or every configuration of such a run would carry the same constant shortfall.
pub(crate) const T5_TRACKER_FLOOR_CENTS: f64 = 10.0;
/// Gross pitch error of the fed F0 (share of voiced frames).
pub(crate) const T5_GPE_MAX: f64 = 0.003;
/// V/UV error of the fed F0 (share of speech frames).
pub(crate) const T5_VUV_MAX: f64 = 0.03;
/// Envelope MCD against the long-shape ceiling over its same-seed floor (dB, Low latency).
pub(crate) const T5_ENV_MCD_OVER_FLOOR_MAX: f64 = 0.6;
/// The same at Quality.
#[allow(dead_code)] // The scorecard's row (P3B-I0); the shortfall uses Low latency's.
pub(crate) const T5_ENV_MCD_OVER_FLOOR_MAX_QUALITY: f64 = 0.3;

/// How far fidelity metrics fall short of T5, each in units of its own limit (0 = every measured
/// check passes). On a v2 run's metrics (medians over its offsets, any of the v2 names present) the
/// T5 v2 rows above: the set-similarity delta (or the advisory cosine v2 when the run had no
/// target set), the fed-F0 fine / gross / V/UV errors against the input (or the output analyzer's
/// split when the voice reports no fed F0), the envelope MCD over its floor, and the larger of the
/// two binomial click counts. On a v1 run's metrics the P3 formula: speaker cosine below 0.97 per
/// 0.03, voiced F0 RMSE above 15 cents per 15 cents, V/UV error above 3 % per 3 %, one per seam
/// click. Unmeasured metrics count 0. The fed-F0 rows of a run whose tracker is not the analyzer's
/// ([`T5_TRACKER_FLOOR_CENTS`]) are charged net of that run's own `whole_fed_input_*` floors; the
/// scorecard's rows are not (ADR 0009). Only the autotuner calls this.
pub(crate) fn t5_shortfall_v2(m: &std::collections::BTreeMap<String, f64>) -> f64 {
    let get = |k: &str| m.get(k).copied().filter(|v| v.is_finite());
    let over = |v: f64, max: f64| ((v - max) / max).max(0.0);
    let v2 = ["speaker_cosine_v2", "env_mcd_db", "fed_input_fpe_cents", "f0_fine_rmse_cents", "clicks_binomial"]
        .iter()
        .any(|k| get(k).is_some());
    if !v2 {
        return get("speaker_cosine").map_or(0.0, |c| ((T5_COSINE_ADVISORY_MIN - c) / 0.03).max(0.0))
            + get("f0_rmse_cents").map_or(0.0, |v| over(v, T5_FPE_MAX_CENTS))
            + get("vuv_error").map_or(0.0, |v| over(v, T5_VUV_MAX))
            + get("clicks").map_or(0.0, |v| v.max(0.0));
    }
    let identity = match get("speaker_set_sim_delta") {
        Some(d) => ((T5_SET_SIM_DELTA_MIN - d) / T5_SET_SIM_DELTA_MIN.abs()).max(0.0),
        None => get("speaker_cosine_v2").map_or(0.0, |c| ((T5_COSINE_ADVISORY_MIN - c) / 0.03).max(0.0)),
    };
    let tracker_disagrees = get("whole_fed_input_fpe_cents").is_some_and(|f| f > T5_TRACKER_FLOOR_CENTS);
    let net = |v: f64, floor: &str| if tracker_disagrees { (v - get(floor).unwrap_or(0.0)).max(0.0) } else { v };
    let fpe = get("fed_input_fpe_cents").map(|v| net(v, "whole_fed_input_fpe_cents")).or_else(|| get("f0_fine_rmse_cents"));
    let gpe = get("fed_input_gpe").map(|v| net(v, "whole_fed_input_gpe")).or_else(|| get("gross_pitch_error"));
    let vuv = get("fed_input_vuv_error").map(|v| net(v, "whole_fed_input_vuv_error")).or_else(|| get("vuv_error"));
    let clicks = get("clicks_binomial").unwrap_or(0.0).max(get("clicks_narrow").unwrap_or(0.0));
    identity
        + fpe.map_or(0.0, |v| over(v, T5_FPE_MAX_CENTS))
        + gpe.map_or(0.0, |v| over(v, T5_GPE_MAX))
        + vuv.map_or(0.0, |v| over(v, T5_VUV_MAX))
        + get("env_mcd_over_floor_db").map_or(0.0, |v| over(v, T5_ENV_MCD_OVER_FLOOR_MAX))
        + clicks.max(0.0)
}

/// The stream-minus-ceiling offset (16 kHz samples) from the timeline: ceiling sample `i` is the
/// conversion of absolute position `ceiling_start_16k + i`; input sample 0 entered the model's
/// stream at absolute `input_start_16k` (priming silence and resampler delay); the harness
/// delivers input sample `n` as output sample `n + latency` (the probed algorithmic latency, which
/// includes the model's declared delay). Both renditions come from the same model, so its
/// inherent delay cancels.
pub(crate) fn timeline_lag_16k(ceiling_start_16k: u64, input_start_16k: f64, latency_frames: f64, output_rate: u32) -> f64 {
    ceiling_start_16k as f64 - input_start_16k + latency_frames * 16_000.0 / f64::from(output_rate.max(1))
}

/// Where input sample 0 sits in the recorded 16 kHz model stream (absolute position): the recorded
/// stream is the input, resampled, after the harness's priming silence, so a waveform alignment
/// finds it to the sample.
pub(crate) fn input_start_16k(rec: &StreamRecorder, input_mono: &[f32], input_rate: u32) -> Result<f64, String> {
    let (start, data) = rec.stream();
    let x16 = resample(input_mono, input_rate, 16_000)?;
    let max_lag = data.len().min(x16.len()).max(1);
    let al = align(&x16, data, 16_000, max_lag, true).ok_or("the recorded model stream could not be aligned with the input")?;
    if !al.waveform || al.correlation < 0.9 {
        return Err(format!(
            "the recorded model stream does not match the input (correlation {:.3}); aligning on envelopes",
            al.correlation
        ));
    }
    Ok(start as f64 + al.lag)
}

/// Offsets further apart than this (16 kHz samples, 30 ms) between the timeline and a confident
/// envelope search are reported.
const TIMELINE_DISAGREE_16K: f64 = 480.0;

/// Seconds of silence kept around the speech region.
const REGION_MARGIN_S: f64 = 0.1;
/// An impulse of the stream within this of one of the ceiling is the same transient.
const MATCH_TOLERANCE_MS: f64 = 0.5;

/// Score a stream against its ceiling (see the module docs).
pub(crate) fn score_pair(sc: &mut Scorers, x: &PairInput<'_>) -> Result<PairScore, String> {
    let s16 = resample(x.stream, x.stream_rate, 16_000)?;
    let c16 = resample(x.ceiling, x.ceiling_rate, 16_000)?;
    let max_lag = (x.max_offset_s * 16_000.0) as usize;
    let searched = align(&c16, &s16, 16_000, max_lag, x.expected_lag_16k.is_none());
    let al = match (x.expected_lag_16k, searched) {
        (Some(lag), s) => PairAlignment {
            lag,
            method: "timeline",
            envelope_lag: s.map(|s| s.lag),
            envelope_correlation: s.map(|s| s.correlation),
        },
        (None, Some(s)) => PairAlignment {
            lag: s.lag,
            method: if s.waveform { "waveform" } else { "envelope" },
            envelope_lag: Some(s.lag),
            envelope_correlation: Some(s.correlation),
        },
        (None, None) => return Err("the stream and the ceiling could not be aligned (too short or silent)".into()),
    };
    let lag = al.lag.round() as isize;
    // Speech region of the ceiling, inside the overlap.
    let mask = speech_frames(&c16, 50.0);
    let first = mask.iter().position(|&m| m).ok_or("the ceiling is silent")?;
    let last = mask.iter().rposition(|&m| m).unwrap_or(first);
    let margin = (REGION_MARGIN_S * 16_000.0) as usize;
    let lo_overlap = (-lag).max(0) as usize;
    let hi_overlap = (c16.len() as isize).min(s16.len() as isize - lag).max(0) as usize;
    let a = (first * 160).saturating_sub(margin).max(lo_overlap);
    let b = ((last + 1) * 160 + margin).min(hi_overlap);
    if b <= a + 16_000 / 2 {
        return Err(format!(
            "the stream and the ceiling overlap on {} ms of speech only (offset {:.1} ms)",
            b.saturating_sub(a) / 16,
            al.lag / 16.0
        ));
    }
    let cs = &c16[a..b];
    let ss = &s16[(a as isize + lag) as usize..(b as isize + lag) as usize];
    let mut q = QualityMetrics {
        offset_ms: Some(al.lag / 16.0),
        seconds: Some((b - a) as f64 / 16_000.0),
        ..QualityMetrics::default()
    };
    q.mcd_db = mcd_db(cs, ss, 16_000).ok();
    // F0 over the ceiling's speech frames of the region.
    let fc = sc.f0.contour(cs)?;
    let fs = sc.f0.contour(ss)?;
    let rmask = speech_frames(cs, 40.0);
    let n = fc.len().min(fs.len()).min(rmask.len());
    if let Ok(e) = f0_errors_masked(&fc[..n], &fs[..n], &rmask[..n]) {
        q.f0_rmse_cents = Some(e.rmse_cents);
        q.gross_pitch_error = Some(e.gross_pitch_error);
        q.vuv_error = Some(e.vuv_error);
        q.f0_frames = Some(e.frames as u64);
    }
    if let Some(sp) = sc.speaker.as_mut() {
        q.speaker_cosine = sp.similarity(cs, ss).ok();
    }
    let mut ceiling_dnsmos = None;
    if let Some(d) = sc.dnsmos.as_mut() {
        if let Ok(v) = d.score(ss) {
            q.dnsmos_ovrl = Some(v.ovrl);
            q.dnsmos_sig = Some(v.sig);
            q.dnsmos_bak = Some(v.bak);
        }
        ceiling_dnsmos = d.score(cs).ok().map(|v| v.ovrl);
    }
    // Clicks: impulses of the stream the ceiling does not have, both analysed at the stream's rate
    // (the ceiling at half the threshold: hysteresis), on the ceiling's 16 kHz timeline; at a join =
    // within the join window after a call's first returned sample.
    let ev_s: Vec<f64> = click_events(x.stream, x.stream_rate, &sc.click)
        .into_iter()
        .map(|e| e as f64 * 16_000.0 / f64::from(x.stream_rate) - al.lag)
        .filter(|&t| t >= a as f64 && t < b as f64)
        .collect();
    let ceiling_at_stream_rate = resample(x.ceiling, x.ceiling_rate, x.stream_rate)?;
    let loose = ClickConfig {
        threshold: sc.click.threshold / 2.0,
        ..sc.click
    };
    let ev_c: Vec<f64> = click_events(&ceiling_at_stream_rate, x.stream_rate, &loose)
        .into_iter()
        .map(|e| e as f64 * 16_000.0 / f64::from(x.stream_rate))
        .collect();
    // Same-model renditions align to the sample: 0.5 ms tells the same transient from a new one
    // without hiding a click next to a pitch pulse; the join window uses the wider tolerance.
    let match_tol = MATCH_TOLERANCE_MS * 16.0;
    let tol = sc.click.tolerance_ms * 16.0;
    let to_int = |v: &[f64]| -> Vec<usize> { v.iter().map(|t| (t.max(0.0) * 4.0).round() as usize).collect() };
    let artifacts = unmatched(&to_int(&ev_s), &to_int(&ev_c), (match_tol * 4.0).round() as usize);
    let joins: Vec<f64> = x
        .calls
        .iter()
        .map(|c| (c.end_16k as f64 - c.returned_16k as f64) - x.ceiling_start_16k as f64)
        .collect();
    let window = x.join_window_ms * 16.0;
    let at_join = artifacts
        .iter()
        .filter(|&&t4| {
            let t = t4 as f64 / 4.0;
            joins.iter().any(|&j| t >= j - tol && t <= j + window + tol)
        })
        .count();
    // Seam clicks = the join windows' excess over the artifact rate away from the joins (plus two
    // standard deviations): a voice that crackles everywhere is not clicking at its seams. With
    // 40 ms blocks and a 20 ms crossfade the windows (crossfade + 5 ms, plus the tolerance on each
    // side) cover about 70 % of the time, so the rate away from the joins rests on a handful of
    // events (v1's statistic; v2 conditions on the events found: `quality::binomial_clicks`).
    let join_cover = covered(joins.iter().map(|&j| (j - tol, j + window + tol)), a as f64, b as f64);
    let off_time = (b - a) as f64 - join_cover;
    let expected_at_join = if off_time > 0.0 {
        (artifacts.len() - at_join) as f64 / off_time * join_cover
    } else {
        0.0
    };
    let seam_clicks = (at_join as f64 - expected_at_join - 2.0 * expected_at_join.sqrt()).max(0.0).round();
    q.artifact_events = Some(artifacts.len() as u64);
    q.clicks = Some(seam_clicks as u64);
    // The artifacts on the stream's block grid (16 kHz ticks of a quarter sample -> samples).
    let period_16k = (x.stream_period as f64 * 16_000.0 / f64::from(x.stream_rate)).round() as usize;
    let artifact_samples: Vec<usize> = artifacts.iter().map(|&t4| t4 / 4).collect();
    let artifact_phase = click_report(&artifact_samples, b - a, 16_000, period_16k, None, sc.click.tolerance_ms);
    let edge_events = ev_s.len();
    q.seam_flux_ratio = seam_flux_ratio(x.stream, x.stream_rate, x.stream_period);
    let ceiling_seam = seam_flux_ratio(x.ceiling, x.ceiling_rate, x.ceiling_period);
    Ok(PairScore {
        quality: q,
        ceiling_dnsmos_ovrl: ceiling_dnsmos,
        ceiling_seam_flux_ratio: ceiling_seam,
        alignment: al,
        region: (a, b),
        edge_events,
        artifact_phase,
        join_artifacts: at_join,
        join_expected: expected_at_join,
        join_coverage: if b > a { join_cover / (b - a) as f64 } else { 0.0 },
        artifacts: artifacts.iter().map(|&t4| t4 as f64 / 4.0).collect(),
    })
}

/// Total length of the union of `spans` clipped to `[lo, hi)`.
fn covered(spans: impl Iterator<Item = (f64, f64)>, lo: f64, hi: f64) -> f64 {
    let mut v: Vec<(f64, f64)> = spans.map(|(s, e)| (s.max(lo), e.min(hi))).filter(|(s, e)| e > s).collect();
    v.sort_by(|x, y| x.0.total_cmp(&y.0));
    let (mut total, mut cur): (f64, Option<(f64, f64)>) = (0.0, None);
    for (s, e) in v {
        cur = match cur {
            Some((cs, ce)) if s <= ce => Some((cs, ce.max(e))),
            Some((cs, ce)) => {
                total += ce - cs;
                Some((s, e))
            }
            None => Some((s, e)),
        };
    }
    total + cur.map_or(0.0, |(s, e)| e - s)
}

/// Run a `quality` run on the spec's engine (v1, or the scorer v2 when `scoring.analyzer` asks
/// for it).
pub(super) fn run_quality(
    q: &QualityRun,
    spec: &BenchSpec,
    p: &Prepared,
    write_files: bool,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<RunOutcome, String> {
    let mut opts = analyzer_opts(&q.scoring.analyzer)?;
    let modes = ceiling_modes(&q.ceiling.mode)?;
    if opts.v2 && ablations_from_env() {
        opts.ablations = true;
    }
    if opts.v2 {
        return run_quality_v2(q, spec, p, write_files, cancel, progress, opts, &modes);
    }
    if modes != [CeilingMode::LongShape] || !q.fidelity.offsets_s.is_empty() || !q.fidelity.reference_clips.is_empty() {
        return Err("ceiling.mode b / whole, fidelity.offsets_s and fidelity.reference_clips need scoring.analyzer v2".into());
    }
    run_quality_v1(q, spec, p, write_files, cancel, progress)
}

/// The P3 quality run (scorer v1).
fn run_quality_v1(
    q: &QualityRun,
    spec: &BenchSpec,
    p: &Prepared,
    write_files: bool,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<RunOutcome, String> {
    let s = &spec.stream;
    let ch_in = usize::from(s.input_channels);
    let lead = (q.lead_s * f64::from(s.input_rate)).round() as usize * ch_in;
    let mut input = vec![0.0f32; lead];
    input.extend_from_slice(&p.input);
    let recorder = Arc::new(Mutex::new(StreamRecorder::default()));
    let offline = OfflineRun {
        name: q.name.clone(),
        repeat: 1,
        thresholds: Default::default(),
    };
    progress(0.0, "streaming through the harness");
    let mut outcome = run_offline(&OfflineInput {
        stream: s,
        run: &offline,
        engine: &p.engine,
        input: &input,
        probe: p.probe,
        record_audio: true,
        cancel,
        progress: &|f, m| progress(0.4 * f, m),
        recorder: Some(recorder.clone()),
    })?;
    let (audio, r_out, ch_out) = outcome.audio.take().ok_or("the streaming pass produced no audio")?;
    let stream_mono = channel(&audio, ch_out, 0);
    progress(0.45, "offline ceiling");
    let rec = lock(&recorder);
    let ce = ceiling(&p.engine, s, &rec, &q.ceiling)?;
    progress(0.7, "scoring");
    let range = p.engine.rvc.as_ref().map_or(F0Range::Speech, |r| r.f0_range);
    let mut sc = Scorers::load(&q.scoring, range);
    let block_16k = p.engine.fixed_block();
    let stream_period = match block_16k {
        Some(b) => (b as u64 * u64::from(r_out) / 16_000) as usize,
        None => (u64::from(s.block_frames) * u64::from(r_out) / u64::from(s.input_rate)) as usize,
    };
    let ceiling_period = (q.ceiling.block_ms / 1000.0 * f64::from(ce.rate)).round() as usize;
    let mut notes = sc.notes.clone();
    notes.extend(ce.notes.iter().cloned());
    // The offset from the timeline (exact); an envelope search only when it cannot be derived.
    let expected_lag_16k = match input_start_16k(&rec, &to_mono(&input, ch_in), s.input_rate) {
        Ok(p0) => Some(timeline_lag_16k(ce.start_16k, p0, p.probe.frames, r_out)),
        Err(e) => {
            notes.push(format!("alignment: {e}"));
            None
        }
    };
    let score = score_pair(
        &mut sc,
        &PairInput {
            stream: &stream_mono,
            stream_rate: r_out,
            ceiling: &ce.audio,
            ceiling_rate: ce.rate,
            ceiling_start_16k: ce.start_16k,
            calls: rec.calls(),
            stream_period,
            ceiling_period,
            join_window_ms: s.crossfade_ms + 5.0,
            max_offset_s: q.lead_s + 2.0 + ce.start_16k as f64 / 16_000.0,
            expected_lag_16k,
        },
    )?;
    if let (Some(env), Some(corr)) = (score.alignment.envelope_lag, score.alignment.envelope_correlation) {
        if score.alignment.method == "timeline" && corr >= 0.8 && (env - score.alignment.lag).abs() > TIMELINE_DISAGREE_16K {
            notes.push(format!(
                "alignment: the energy envelopes put the stream {:.1} ms from the timeline's offset (correlation {corr:.2}); scored on the timeline",
                (env - score.alignment.lag) / 16.0
            ));
        }
    }
    let mut qm = score.quality.clone();
    // Speaker cosine against a reference clip of the target voice.
    let mut reference_cos = None;
    if let (Some(path), Some(sp)) = (&q.scoring.speaker_reference, sc.speaker.as_mut()) {
        let region = region_16k(&stream_mono, r_out, &score)?;
        reference_cos = clip_16k(path).and_then(|r| sp.similarity(&r, &region)).map_err(|e| notes.push(format!("speaker reference: {e}"))).ok();
    }
    // CER through Sigil's STT.
    let mut cer = (None, None, None);
    let mut transcripts = serde_json::Value::Null;
    if let Some(stt) = &q.stt {
        progress(0.85, "transcribing");
        match cer_scores(stt, &p.input, s.input_rate, ch_in, &stream_mono, r_out, &ce.audio, ce.rate, &score) {
            Ok((c, t)) => {
                cer = c;
                transcripts = t;
            }
            Err(e) => notes.push(format!("CER: {e}")),
        }
    }
    qm.cer = cer.0;
    let m = &mut outcome.metrics;
    qm.insert_into("", m);
    if let Some(v) = score.ceiling_dnsmos_ovrl {
        m.insert("ceiling_dnsmos_ovrl".into(), v);
        if let Some(o) = qm.dnsmos_ovrl {
            m.insert("dnsmos_ovrl_delta".into(), o - v);
        }
    }
    if let Some(v) = score.ceiling_seam_flux_ratio {
        m.insert("ceiling_seam_flux_ratio".into(), v);
    }
    if let (Some(c), Some(secs)) = (qm.clicks, qm.seconds) {
        m.insert("clicks_per_10min".into(), c as f64 / secs.max(1e-9) * 600.0);
    }
    if let Some(v) = reference_cos {
        m.insert("speaker_cosine_reference".into(), v);
    }
    if let Some(v) = cer.1 {
        m.insert("cer_ceiling".into(), v);
    }
    if let Some(v) = cer.2 {
        m.insert("cer_input".into(), v);
    }
    if let (Some(a), Some(b)) = (cer.0, cer.1) {
        m.insert("cer_delta_pts".into(), (a - b) * 100.0);
    }
    m.insert("ceiling_compute_ms".into(), ce.compute_ms);
    let ce_secs = ce.audio.len() as f64 / f64::from(ce.rate.max(1));
    if ce_secs > 0.0 {
        m.insert("ceiling_rtf".into(), ce.compute_ms / 1000.0 / ce_secs);
    }
    if let Some(obj) = outcome.details.as_object_mut() {
        obj.insert("lead_s".into(), q.lead_s.into());
        obj.insert("ceiling_method".into(), ce.method.clone().into());
        obj.insert("alignment".into(), serde_json::to_value(score.alignment).unwrap_or_default());
        obj.insert("region_16k".into(), serde_json::json!([score.region.0, score.region.1]));
        obj.insert("stream_impulse_events".into(), score.edge_events.into());
        obj.insert("artifact_phase".into(), serde_json::to_value(&score.artifact_phase).unwrap_or_default());
        obj.insert(
            "joins".into(),
            serde_json::json!({
                "artifacts_in_windows": score.join_artifacts,
                "expected_from_off_join_rate": score.join_expected,
                "window_coverage": score.join_coverage,
            }),
        );
        obj.insert("scorers".into(), sc.describe());
        obj.insert("quality".into(), serde_json::to_value(&qm).unwrap_or_default());
        obj.insert("transcripts".into(), transcripts);
        obj.insert("notes".into(), serde_json::to_value(&notes).unwrap_or_default());
    }
    if spec.output.write_audio && write_files {
        outcome.audio = Some((audio, r_out, ch_out));
        outcome.extra_audio.push(("ceiling".into(), ce.audio, ce.rate, 1));
    }
    progress(1.0, "done");
    Ok(outcome)
}

// =============================================================================================
// Scorer v2 (P3B-B, ADR 0009)
// =============================================================================================

/// A stream pass of the v2 scorer: the harness output, the recorded model stream with the SOLA
/// offset of every join, and what the engine probe recorded.
pub(crate) struct StreamPass {
    /// Interleaved at the stream's output rate and channels.
    pub audio: Vec<f32>,
    pub rate: u32,
    pub channels: usize,
    pub rec: StreamRecorder,
    /// `(absolute start frame, Hz after the shift)` of the F0 the generator was fed.
    pub fed_f0: Option<(u64, Vec<f32>)>,
    pub model_ms: Vec<f64>,
    pub tick_errors: u64,
    pub first_error: Option<String>,
    pub applied: ProbeCounts,
}

impl StreamPass {
    /// The first output channel.
    pub fn mono(&self) -> Vec<f32> {
        channel(&self.audio, self.channels, 0)
    }
}

/// Stream `input` (interleaved at the stream's input format) through the production harness and
/// the worker's block processing with `engine`, recording the model's 16 kHz stream and the SOLA
/// offset of every join (`VcStreamHarness::last_join` after each device block); `probe` asks a
/// direct template build's engine probe for `setup` first.
pub(crate) fn stream_pass(
    engine: &Engine,
    s: &StreamSpec,
    input: &[f32],
    probe: Option<(&ProbeChannel, ProbeSetup)>,
    cancel: &AtomicBool,
) -> Result<StreamPass, String> {
    let ch_in = usize::from(s.input_channels).max(1);
    let ch_out = usize::from(s.output_channels).max(1);
    let block = s.block_frames as usize;
    let recorder = Arc::new(Mutex::new(StreamRecorder::default()));
    let times = Arc::new(Mutex::new(ModelTimes::default()));
    if let Some((ch, setup)) = &probe {
        ch.request(setup.clone());
    }
    engine.set_simulated(false);
    let harness = VcStreamHarness::new(
        engine.handle_recording(times.clone(), Some(recorder.clone())),
        s.input_rate,
        s.output_rate,
        ch_in,
        ch_out,
        s.extra_samples_16k(),
        s.crossfade_samples(engine.info.output_rate),
        block,
    )?;
    let mut tick = VcTick::new(harness, s);
    let blocks = input.len() / ch_in / block.max(1);
    if blocks == 0 {
        return Err(format!("the input is shorter than one block ({block} frames)"));
    }
    let mut audio = Vec::with_capacity(input.len() / ch_in * ch_out * 2);
    let mut work = Vec::with_capacity(block * ch_in);
    let (mut tick_errors, mut first_error) = (0u64, None);
    for b in 0..blocks {
        if cancel.load(std::sync::atomic::Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        work.clear();
        work.extend_from_slice(&input[b * block * ch_in..(b + 1) * block * ch_in]);
        let out = tick.process(&mut work);
        if let Some(e) = out.error {
            tick_errors += 1;
            first_error.get_or_insert(e);
        }
        audio.extend_from_slice(&out.audio);
        lock(&recorder).note_join(tick.harness.last_join(), engine.info.output_rate);
    }
    drop(tick);
    let fed_f0 = engine.with_model(|m| m.telemetry().and_then(|t| t.fed_f0()).map(|f| (f.start_frame, f.hz.to_vec())));
    let applied = probe.map(|(c, _)| c.applied()).unwrap_or_default();
    let rec = std::mem::take(&mut *lock(&recorder));
    let model_ms = lock(&times).convert_ms.clone();
    Ok(StreamPass {
        audio,
        rate: s.output_rate,
        channels: ch_out,
        rec,
        fed_f0,
        model_ms,
        tick_errors,
        first_error,
        applied,
    })
}

/// A rendition at 16 kHz on the absolute timeline: sample `i` is the conversion of absolute
/// position `start_16k + i`.
#[derive(Clone, Debug)]
pub(crate) struct Rendition {
    pub name: String,
    pub audio16: Vec<f32>,
    pub start_16k: u64,
    /// The audio as rendered (for the report's WAVs).
    pub raw: Vec<f32>,
    pub rate: u32,
    pub method: String,
    pub compute_ms: f64,
    pub notes: Vec<String>,
}

impl Rendition {
    pub fn from_ceiling(name: &str, c: CeilingOutput) -> Result<Rendition, String> {
        Ok(Rendition {
            name: name.to_string(),
            audio16: resample(&c.audio, c.rate, 16_000)?,
            start_16k: c.start_16k,
            raw: c.audio,
            rate: c.rate,
            method: c.method,
            compute_ms: c.compute_ms,
            notes: c.notes,
        })
    }

    /// Absolute span `[a, b)`, when the rendition covers all of it.
    pub fn span(&self, a: u64, b: u64) -> Option<&[f32]> {
        let i = a.checked_sub(self.start_16k)? as usize;
        let j = (b.checked_sub(self.start_16k)? as usize).min(self.audio16.len());
        (j > i && j - i == (b - a) as usize).then(|| &self.audio16[i..j])
    }

    /// The absolute span it covers.
    pub fn extent(&self) -> (u64, u64) {
        (self.start_16k, self.start_16k + self.audio16.len() as u64)
    }
}

/// The stream on the absolute timeline: its 16 kHz index of absolute position `P` is `P + shift`
/// (the timeline lag: the probed harness latency minus where the input entered the model stream).
#[derive(Clone, Debug)]
pub(crate) struct StreamTimeline {
    pub s16: Vec<f32>,
    pub shift: i64,
}

impl StreamTimeline {
    pub fn span(&self, a: u64, b: u64) -> Option<&[f32]> {
        let i = a as i64 + self.shift;
        let j = b as i64 + self.shift;
        (i >= 0 && j as usize <= self.s16.len() && j > i).then(|| &self.s16[i as usize..j as usize])
    }

    pub fn extent(&self) -> (u64, u64) {
        let a = (-self.shift).max(0) as u64;
        (a, (self.s16.len() as i64 - self.shift).max(0) as u64)
    }
}

/// `[a, b)` intersected with every extent.
fn intersect(mut a: u64, mut b: u64, extents: &[(u64, u64)]) -> (u64, u64) {
    for &(x, y) in extents {
        a = a.max(x);
        b = b.min(y);
    }
    (a, b.max(a))
}

/// v2 scores of a test rendition against a reference over the same samples.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct PairV2 {
    pub seconds: f64,
    pub mcd_db: Option<f64>,
    pub env_mcd_db: Option<f64>,
    pub speaker_cosine_v2: Option<f64>,
    pub f0: Option<F0ErrorsV2>,
    /// The output analyzer's V/UV error with ±1 frame of tolerance (a SOLA-jitter diagnostic).
    pub vuv_error_tolerant: Option<f64>,
    /// DNSMOS P.835 overall of the reference and the test.
    pub dnsmos_reference: Option<f64>,
    pub dnsmos_test: Option<f64>,
    /// The test's similarity to the target-speaker set (and the reference's).
    pub set_sim_test: Option<f64>,
    pub set_sim_reference: Option<f64>,
}

/// What [`pair_v2`] measures besides the always-on MCD / envelope MCD / F0 split.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct PairWants<'a> {
    pub speaker: bool,
    pub dnsmos: bool,
    pub set: Option<&'a [f32]>,
}

/// Score `test` against `reference` (16 kHz, the same samples): STFT MCD, envelope MCD (with the
/// reference's F0), the F0 split of the output analyzer over the reference's speech frames, and as
/// asked the exact CAM++ cosine on the reference's speech frames, DNSMOS of both, and both
/// renditions' similarity to the target-speaker set's centroid.
pub(crate) fn pair_v2(sc: &mut Scorers, reference: &[f32], test: &[f32], wants: PairWants<'_>, notes: &mut Vec<String>) -> PairV2 {
    let n = reference.len().min(test.len());
    let (r, t) = (&reference[..n], &test[..n]);
    let mut out = PairV2 {
        seconds: n as f64 / 16_000.0,
        mcd_db: mcd_db(r, t, 16_000).ok(),
        ..PairV2::default()
    };
    match (sc.f0.contour(r), sc.f0.contour(t)) {
        (Ok(fr), Ok(ft)) => {
            out.env_mcd_db = envelope_mcd_db(r, t, &fr, 16_000).map_err(|e| notes.push(e)).ok();
            let mask = speech_frames(r, 40.0);
            let k = fr.len().min(ft.len()).min(mask.len());
            out.f0 = f0_errors_v2(&fr[..k], &ft[..k], Some(&mask[..k])).ok();
            out.vuv_error_tolerant = vuv_error_tolerant(&fr[..k], &ft[..k], Some(&mask[..k]), 1);
        }
        (Err(e), _) | (_, Err(e)) => notes.push(format!("F0 analysis: {e}")),
    }
    let mask = super::scorers::speech_frame_mask(r, super::scorers::SPEAKER_SPEECH_RANGE_DB);
    if let Some(sp) = sc.speaker_v2.as_mut() {
        if wants.speaker {
            out.speaker_cosine_v2 = sp.similarity_masked(r, t, &mask).map_err(|e| notes.push(format!("speaker v2: {e}"))).ok();
        }
        if let Some(c) = wants.set {
            out.set_sim_test = sp.embed_masked(t, &mask).map(|e| super::scorers::cosine(&e, c)).ok();
            out.set_sim_reference = sp.embed_masked(r, &mask).map(|e| super::scorers::cosine(&e, c)).ok();
        }
    }
    if wants.dnsmos {
        if let Some(d) = sc.dnsmos.as_mut() {
            out.dnsmos_reference = d.score(r).ok().map(|v| v.ovrl);
            out.dnsmos_test = d.score(t).ok().map(|v| v.ovrl);
        }
    }
    out
}

/// Where a call's join sits on the ceiling's timeline (P3B-B): the returned span starts with the
/// SOLA search margin and the returned audio is the input `delay_16k` earlier (the model's
/// lookahead), so the crossfade of call `c` starts at
/// `c.end_16k - c.returned_16k - delay_16k + centre_16k` and lasts `overlap_16k`.
#[derive(Clone, Copy, Debug, Default, Serialize, serde::Deserialize)]
pub(crate) struct JoinGeometry {
    pub delay_16k: f64,
    pub centre_16k: f64,
    pub overlap_16k: f64,
    /// The harness's SOLA tail (16 kHz samples): a call's block is `returned_16k - tail_16k`.
    pub tail_16k: f64,
}

impl JoinGeometry {
    /// The harness of `stream` in front of a model with output rate `out_sr` and delay
    /// `model_delay_ms` (the SOLA formula of `stream.rs`: overlap = the crossfade at the model rate,
    /// at least 64 samples; search = half of it; the window centre half of the search, rounded).
    pub fn of(stream: &StreamSpec, out_sr: u32, model_delay_ms: f64) -> JoinGeometry {
        let out_sr = out_sr.max(1);
        let overlap_out = stream.crossfade_samples(out_sr).max(64) as f64;
        let tail_16k = crate::services::neural_audio::vc::stream::sola_tail_in_16k(stream.crossfade_ms, out_sr) as f64;
        let to16 = 16_000.0 / f64::from(out_sr);
        let tail_out = (tail_16k / to16).round();
        JoinGeometry {
            delay_16k: model_delay_ms.max(0.0) * 16.0,
            centre_16k: ((tail_out - overlap_out) / 2.0).floor().max(0.0) * to16,
            overlap_16k: overlap_out * to16,
            tail_16k,
        }
    }

    /// The same geometry with the SOLA tail a fixed-block model's calls really returned
    /// (`returned_16k − block`), so a harness whose search window differs from the formula's still
    /// places its joins right.
    pub fn with_tail(mut self, tail_16k: f64, out_sr: u32) -> JoinGeometry {
        let to16 = 16_000.0 / f64::from(out_sr.max(1));
        let tail_out = (tail_16k / to16).round();
        let overlap_out = (self.overlap_16k / to16).round();
        self.tail_16k = tail_16k;
        self.centre_16k = ((tail_out - overlap_out) / 2.0).floor().max(0.0) * to16;
        self
    }

    /// The returned span's first sample in content coordinates (the input it was converted from).
    pub fn returned_start(&self, c: &CallSpan) -> f64 {
        (c.end_16k as f64 - c.returned_16k as f64) - self.delay_16k
    }

    /// Start of the call's crossfade (absolute, content coordinates).
    pub fn crossfade_start(&self, c: &CallSpan) -> f64 {
        self.returned_start(c) + self.centre_16k
    }

    /// The call's emitted block (16 kHz samples).
    pub fn block_16k(&self, c: &CallSpan) -> f64 {
        (c.returned_16k as f64 - self.tail_16k).max(0.0)
    }
}

/// The v2 click tests (see `quality::binomial_clicks`) on the artifact events of one region.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct ClicksV2 {
    pub events: usize,
    /// v1's window shape (the returned span's start to `crossfade + 5 ms` after it, plus the
    /// tolerance), placed for the model's delay.
    pub at_wide: usize,
    pub coverage_wide: f64,
    pub clicks_wide: u64,
    pub p_wide: f64,
    /// Narrow windows (± the tolerance) at both crossfade boundaries.
    pub at_narrow: usize,
    pub coverage_narrow: f64,
    pub clicks_narrow: u64,
    pub p_narrow: f64,
}

/// The v2 click tests of `artifacts` (absolute 16 kHz positions inside `[a, b)`) against the joins
/// of `calls`.
pub(crate) fn clicks_v2(artifacts: &[f64], calls: &[CallSpan], geo: &JoinGeometry, a: f64, b: f64, wide_ms: f64, tol_ms: f64) -> ClicksV2 {
    let tol = tol_ms * 16.0;
    let wide: Vec<(f64, f64)> = calls
        .iter()
        .map(|c| {
            let j = geo.returned_start(c);
            (j - tol, j + wide_ms * 16.0 + tol)
        })
        .collect();
    let narrow: Vec<(f64, f64)> = calls
        .iter()
        .flat_map(|c| {
            let x = geo.crossfade_start(c);
            [(x - tol, x + tol), (x + geo.overlap_16k - tol, x + geo.overlap_16k + tol)]
        })
        .collect();
    let inside = |spans: &[(f64, f64)], t: f64| spans.iter().any(|&(s, e)| t >= s && t <= e);
    let len = (b - a).max(1.0);
    let cov_w = covered(wide.iter().copied(), a, b) / len;
    let cov_n = covered(narrow.iter().copied(), a, b) / len;
    let events: Vec<f64> = artifacts.iter().copied().filter(|&t| t >= a && t < b).collect();
    let at_w = events.iter().filter(|&&t| inside(&wide, t)).count();
    let at_n = events.iter().filter(|&&t| inside(&narrow, t)).count();
    let n = events.len() as u64;
    ClicksV2 {
        events: events.len(),
        at_wide: at_w,
        coverage_wide: cov_w,
        clicks_wide: binomial_clicks(at_w as u64, n, cov_w),
        p_wide: binomial_at_least(at_w as u64, n, cov_w),
        at_narrow: at_n,
        coverage_narrow: cov_n,
        clicks_narrow: binomial_clicks(at_n as u64, n, cov_n),
        p_narrow: binomial_at_least(at_n as u64, n, cov_n),
    }
}

/// The offset-corrected MCD of the stream against `reference` (the ceiling over `[a, b)`, 16 kHz).
#[derive(Clone, Copy, Debug, Serialize)]
pub(crate) struct OffsetCorrected {
    pub mcd_db: f64,
    pub env_mcd_db: Option<f64>,
    /// `recorded SOLA offsets`; `recorded SOLA offsets, searched where unobserved` when some calls
    /// share a device block (the harness keeps one join per block); `searched (±6 ms)` when the
    /// call log has none.
    pub method: &'static str,
    /// Mean |shift| over the scored frames (ms).
    pub mean_abs_shift_ms: f64,
    /// Blocks whose displacement was searched, not read from a recorded join.
    pub searched_blocks: usize,
}

/// The MCD of the stream against `reference` with each frame read where its content is: an
/// emitted block carries the content `d` samples after its nominal place (`d` = its join's SOLA
/// offset from the window centre), so a reference frame centred in that block is compared with the
/// stream's window `d` samples earlier (`quality::mcd_db_shifted`; the stream itself is not spliced).
/// Without a recorded offset (a report with no call log, or a call whose join the stream loop did
/// not observe because its device block made several model calls) the block's `d` is searched within
/// ±6 ms by normalized cross-correlation (a fallback, biased toward agreement), never assumed 0.
/// `None` when the stream does not cover the region.
pub(crate) fn offset_corrected_mcd(
    stream: &StreamTimeline,
    reference: &[f32],
    a: u64,
    b: u64,
    calls: &[CallSpan],
    geo: &JoinGeometry,
    f0_ref: Option<&[f32]>,
) -> Option<OffsetCorrected> {
    const SEARCH_16K: i64 = 96;
    stream.span(a, b)?;
    let at = |x: i64| -> Option<f32> {
        let i = x + stream.shift;
        (i >= 0).then(|| stream.s16.get(i as usize).copied()).flatten()
    };
    let recorded = calls.iter().any(|c| c.sola_offset_16k.is_some());
    // Each call's emitted block on the nominal timeline and its displacement.
    let mut blocks: Vec<(i64, i64, i64)> = Vec::with_capacity(calls.len());
    let mut searched_blocks = 0usize;
    for c in calls {
        let j = geo.crossfade_start(c).round() as i64;
        let blk = geo.block_16k(c).round() as i64;
        if blk <= 0 {
            continue;
        }
        let observed = recorded && c.join_seen;
        let d = if observed {
            // A seen call without an offset had no join (a silent tail): nothing displaced it.
            c.sola_offset_16k.map_or(0, |d| d.round() as i64)
        } else {
            searched_blocks += 1;
            let (lo, hi) = (j.max(a as i64), (j + blk).min(b as i64));
            if hi <= lo {
                0
            } else {
                let mut best = (0i64, f64::NEG_INFINITY);
                for d in -SEARCH_16K..=SEARCH_16K {
                    let (mut ab, mut aa, mut bb) = (0.0f64, 0.0f64, 0.0f64);
                    for x in lo..hi {
                        let Some(r) = usize::try_from(x + d - a as i64).ok().and_then(|i| reference.get(i)) else {
                            continue;
                        };
                        let Some(v) = at(x) else {
                            continue;
                        };
                        ab += f64::from(v) * f64::from(*r);
                        aa += f64::from(v) * f64::from(v);
                        bb += f64::from(*r) * f64::from(*r);
                    }
                    let ncc = if aa > 0.0 && bb > 0.0 { ab / (aa * bb).sqrt() } else { 0.0 };
                    if ncc > best.1 {
                        best = (d, ncc);
                    }
                }
                best.0
            }
        };
        blocks.push((j, j + blk, d));
    }
    blocks.sort_by_key(|&(j, _, _)| j);
    // Frame i of the MCD grid is centred at a + 160 i + 200: its block's content sits d earlier.
    let frames = super::dsp::frames(reference.len(), 400, 160).count();
    let shift: Vec<i64> = (0..frames)
        .map(|i| {
            let x = a as i64 + 160 * i as i64 + 200;
            let k = blocks.partition_point(|&(j, _, _)| j <= x);
            match k.checked_sub(1).map(|k| blocks[k]) {
                Some((_, end, d)) if x < end => -d,
                _ => 0,
            }
        })
        .collect();
    let (mcd_db, env_mcd_db) = super::quality::mcd_db_shifted(reference, &stream.s16, a as i64 + stream.shift, &shift, f0_ref).ok()?;
    let mean_abs_shift_ms = if shift.is_empty() { 0.0 } else { shift.iter().map(|v| v.unsigned_abs() as f64).sum::<f64>() / shift.len() as f64 / 16.0 };
    Some(OffsetCorrected {
        mcd_db,
        env_mcd_db,
        method: match (recorded, searched_blocks) {
            (false, _) => "searched (±6 ms)",
            (true, 0) => "recorded SOLA offsets",
            (true, _) => "recorded SOLA offsets, searched where unobserved",
        },
        mean_abs_shift_ms,
        searched_blocks,
    })
}

/// The frame lag in `-5..=5` at which `test` (read `lag` frames later) best matches `reference`
/// over the frames `mask` keeps (the lowest fine pitch error with at least half of lag 0's
/// voiced-in-both frames): `{best_lag, fpe_at_0, fpe_at_best}`.
fn track_lag(reference: &[Option<f32>], test: &[Option<f32>], mask: &[bool]) -> Option<serde_json::Value> {
    let at = |lag: i64| -> Option<(f64, usize)> {
        let n = reference.len().min(mask.len());
        let shifted: Vec<Option<f32>> = (0..n).map(|i| usize::try_from(i as i64 + lag).ok().and_then(|j| test.get(j).copied().flatten())).collect();
        let e = fed_errors(reference, &shifted, mask)?;
        Some((e.errors.fine_rmse_cents?, e.errors.v1.voiced_frames))
    };
    let (fpe0, n0) = at(0)?;
    let (best, fpe) = (-5i64..=5)
        .filter_map(|l| at(l).filter(|&(_, n)| 2 * n >= n0).map(|(f, _)| (l, f)))
        .min_by(|x, y| x.1.total_cmp(&y.1))?;
    Some(serde_json::json!({"best_lag": best, "fpe_at_0": fpe0, "fpe_at_best": fpe}))
}

/// Values of an absolute-frame track over frames `[f0, f1)` (`None` outside it).
fn track_frames(track: &(u64, Vec<f32>), f0: u64, f1: u64) -> Vec<Option<f32>> {
    (f0..f1)
        .map(|f| f.checked_sub(track.0).and_then(|i| track.1.get(i as usize).copied()))
        .collect()
}

/// [`fed_f0_errors`] over the frames both tracks cover and `mask` keeps.
fn fed_errors(reference: &[Option<f32>], test: &[Option<f32>], mask: &[bool]) -> Option<FedF0Errors> {
    let n = reference.len().min(test.len()).min(mask.len());
    let keep: Vec<bool> = (0..n).map(|i| mask[i] && reference[i].is_some() && test[i].is_some()).collect();
    let r: Vec<f32> = reference[..n].iter().map(|v| v.unwrap_or(0.0)).collect();
    let t: Vec<f32> = test[..n].iter().map(|v| v.unwrap_or(0.0)).collect();
    fed_f0_errors(&r, &t, Some(&keep)).ok()
}

/// Whole-clip F0 of the recorded model stream (Hz × the pitch shift, absolute frames): the input
/// reference of the fed-F0 metrics.
fn input_f0_track(sc: &mut Scorers, rec: &StreamRecorder, semitones: f32) -> Result<(u64, Vec<f32>), String> {
    let (start, data) = rec.stream();
    let skip = ((160 - start % 160) % 160) as usize;
    if data.len() <= skip {
        return Err("the recorded stream is shorter than a frame".into());
    }
    let shift = 2f32.powf(semitones / 12.0);
    let f0: Vec<f32> = sc.f0.contour(&data[skip..])?.into_iter().map(|f| f * shift).collect();
    Ok(((start + skip as u64) / 160, f0))
}

/// Speech frames (within 40 dB of the loudest) of the recorded stream over absolute frames
/// `[f0, f1)`.
fn input_speech_frames(rec: &StreamRecorder, f0: u64, f1: u64) -> Vec<bool> {
    let (start, data) = rec.stream();
    let e: Vec<f64> = (f0..f1)
        .map(|f| {
            let lo = (f * 160).saturating_sub(start) as usize;
            let hi = ((f * 160 + 160).saturating_sub(start) as usize).min(data.len());
            let seg = data.get(lo.min(hi)..hi).unwrap_or(&[]);
            let p = seg.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / seg.len().max(1) as f64;
            10.0 * (p + 1e-12).log10()
        })
        .collect();
    let peak = e.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    e.into_iter().map(|v| v >= peak - 40.0).collect()
}

/// Non-speech frames before a speech frame that make it an utterance onset (200 ms).
const ONSET_GAP_FRAMES: usize = 20;
/// Frames of an onset envelope before and after the onset.
const ONSET_PRE_FRAMES: u64 = 5;
const ONSET_POST_FRAMES: u64 = 30;
/// Frames after an onset whose level is compared with the region's speech level (200 ms).
const ONSET_LEVEL_FRAMES: u64 = 20;

/// Utterance onsets in `[a, b)` (absolute 10 ms frames): speech frames of the recorded input
/// (within 40 dB of its loudest frame there) after at least [`ONSET_GAP_FRAMES`] non-speech frames;
/// the region's first speech frame is one.
pub(crate) fn utterance_onsets(rec: &StreamRecorder, a: u64, b: u64) -> Vec<u64> {
    let fa = a.div_ceil(160);
    let mut out = Vec::new();
    let mut quiet = ONSET_GAP_FRAMES;
    for (i, speech) in input_speech_frames(rec, fa, b / 160).into_iter().enumerate() {
        if speech {
            if quiet >= ONSET_GAP_FRAMES {
                out.push(fa + i as u64);
            }
            quiet = 0;
        } else {
            quiet += 1;
        }
    }
    out
}

/// A rendition's onsets: the energy envelope (dB per 10 ms frame, from [`ONSET_PRE_FRAMES`]
/// before each onset to [`ONSET_POST_FRAMES`] after it; `None` outside the rendition) and the level
/// of the [`ONSET_LEVEL_FRAMES`] after the onsets relative to the rendition's level over the
/// region's speech frames (dB, power means, the mean over the onsets). A muted onset reads far
/// below the other renditions.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct OnsetReport {
    pub rel_db: Option<f64>,
    pub envelopes: Vec<Vec<Option<f64>>>,
}

/// [`OnsetReport`] of 16 kHz `audio16` whose sample 0 is absolute position `start_16k`, for
/// `onsets` and the region's `speech` frames (absolute 10 ms frames).
pub(crate) fn onset_report(audio16: &[f32], start_16k: i64, onsets: &[u64], speech: &[u64]) -> OnsetReport {
    let power = |f: u64| -> Option<f64> {
        let s = f as i64 * 160 - start_16k;
        let s = usize::try_from(s).ok()?;
        let seg = audio16.get(s..s + 160)?;
        Some(seg.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / 160.0)
    };
    let db = |p: f64| 10.0 * (p + 1e-12).log10();
    let mean_db = |frames: &mut dyn Iterator<Item = u64>| -> Option<f64> {
        let v: Vec<f64> = frames.filter_map(power).collect();
        (!v.is_empty()).then(|| db(v.iter().sum::<f64>() / v.len() as f64))
    };
    let level = mean_db(&mut speech.iter().copied());
    let rels: Vec<f64> = onsets
        .iter()
        .filter_map(|&o| Some(mean_db(&mut (o..o + ONSET_LEVEL_FRAMES))? - level?))
        .collect();
    OnsetReport {
        rel_db: (!rels.is_empty()).then(|| rels.iter().sum::<f64>() / rels.len() as f64),
        envelopes: onsets
            .iter()
            .map(|&o| {
                (o.saturating_sub(ONSET_PRE_FRAMES)..o + ONSET_POST_FRAMES)
                    .map(|f| power(f).map(|p| (db(p) * 10.0).round() / 10.0))
                    .collect()
            })
            .collect(),
    }
}

/// Insert `name` when `v` is finite.
fn put(m: &mut Metrics, name: &str, v: Option<f64>) {
    if let Some(v) = v.filter(|v| v.is_finite()) {
        m.insert(name.to_string(), v);
    }
}

/// The fed-F0 metric triple of `e` as `<prefix>fpe_cents`, `<prefix>gpe`, `<prefix>vuv_error`
/// (+ onset lag / missed onsets when `onsets`).
fn put_fed(m: &mut Metrics, prefix: &str, e: Option<&FedF0Errors>, onsets: bool) {
    let Some(e) = e else {
        return;
    };
    put(m, &format!("{prefix}fpe_cents"), e.errors.fine_rmse_cents);
    put(m, &format!("{prefix}gpe"), Some(e.errors.v1.gross_pitch_error));
    put(m, &format!("{prefix}vuv_error"), Some(e.errors.v1.vuv_error));
    if onsets {
        put(m, &format!("{prefix}onset_lag_frames"), e.onset_lag_median());
        // The signed lag is a diagnostic (negative = voicing starts early); gate on this one.
        put(m, &format!("{prefix}onset_lag_abs_frames"), e.onset_lag_abs_median());
        if prefix == "fed_input_" {
            put(m, "fed_input_onsets_missed", Some(e.onsets_missed as f64));
        }
    }
}

/// The stream engine of a v2 run: a twin of the spec's voice (silence skip off; a direct build for
/// template voices, so the probe is reachable), or the spec's engine itself (the stub).
enum StreamEngine<'a> {
    Spec(&'a Engine),
    Twin(Box<Engine>),
}

impl StreamEngine<'_> {
    fn get(&self) -> &Engine {
        match self {
            StreamEngine::Spec(e) => e,
            StreamEngine::Twin(e) => e,
        }
    }
}

fn stream_engine<'a>(p: &'a Prepared, spec: &BenchSpec, notes: &mut Vec<String>) -> Result<StreamEngine<'a>, String> {
    match &spec.engine {
        EngineSpec::Rvc(r) => {
            let mut twin = r.clone();
            twin.silence_skip = false;
            let template = p.engine.rvc.as_ref().is_some_and(|pl| pl.format == SlotFormat::SigilRvc);
            if template {
                twin.build = "direct".into();
            }
            let priority = crate::services::rt_thread::WorkerPriority::from_str_lossy(&spec.stream.worker_priority);
            let e = build_engine(&EngineSpec::Rvc(twin), &spec.stream, priority, spec.seed)?;
            notes.push(format!(
                "scorer v2: the stream is a {} build of the voice with the silence skip off (both renditions converted the same way){}",
                if template { "direct" } else { "factory" },
                if template { ", the engine probe's fed-F0 tap on" } else { "" }
            ));
            Ok(StreamEngine::Twin(Box::new(e)))
        }
        EngineSpec::Stub(_) => Ok(StreamEngine::Spec(&p.engine)),
    }
}

/// The clip's speech region on the recorded stream's absolute timeline (speech within 50 dB of
/// the clip's loudest frame, `margin_s` around it, inside the clip and the recording).
fn input_region(rec: &StreamRecorder, input_start: f64, clip16: &[f32], margin_s: f64) -> Option<(u64, u64)> {
    let mask = speech_frames(clip16, 50.0);
    let first = mask.iter().position(|&m| m)?;
    let last = mask.iter().rposition(|&m| m)?;
    let margin = margin_s * 16_000.0;
    let s0 = input_start.max(0.0);
    let a = (s0 + first as f64 * 160.0 - margin).max(s0).round() as u64;
    let b = (s0 + ((last + 1) * 160) as f64 + margin).min(s0 + clip16.len() as f64).round() as u64;
    let (start, data) = rec.stream();
    let (a, b) = (a.max(start), b.min(start + data.len() as u64));
    (b > a + 3_200).then_some((a, b))
}

/// Everything one offset of a v2 run produced (metrics, details, audio, and what the ablations
/// reuse).
struct OffsetResult {
    metrics: Metrics,
    details: serde_json::Value,
    stream_audio: (Vec<f32>, u32, usize),
    renditions: Vec<Rendition>,
    abl: Option<ablation::AblationBase>,
    model_ms: Vec<f64>,
}

/// The v2 quality run (see the module docs).
#[allow(clippy::too_many_arguments)]
fn run_quality_v2(
    q: &QualityRun,
    spec: &BenchSpec,
    p: &Prepared,
    write_files: bool,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
    opts: AnalyzerOpts,
    modes: &[CeilingMode],
) -> Result<RunOutcome, String> {
    let s = &spec.stream;
    let offsets = v2_offsets(q);
    if offsets.iter().any(|o| !o.is_finite() || *o < 0.0 || *o > 60.0) {
        return Err("fidelity.offsets_s: every lead must be 0-60 s".into());
    }
    // The ceilings the spec asks for must render; the others a template voice gets besides are
    // optional (see the module docs). The ablations need the whole ceiling's analysis.
    let mut required = modes.to_vec();
    if opts.ablations && !required.contains(&CeilingMode::Whole) {
        required.push(CeilingMode::Whole);
    }
    let ch_in = usize::from(s.input_channels).max(1);
    let clip16 = resample(&channel(&p.input, ch_in, 0), s.input_rate, 16_000)?;
    // Fail before the engine loads when a required whole ceiling cannot fit this clip and lead.
    if required.contains(&CeilingMode::Whole) && p.engine.rvc.as_ref().is_some_and(|pl| pl.format == SlotFormat::SigilRvc) {
        for &lead in &offsets {
            whole_region_check(lead, &clip16).map_err(|e| format!("ceiling.mode whole (or the ablations): {e}"))?;
        }
    }
    let range = p.engine.rvc.as_ref().map_or(F0Range::Speech, |r| r.f0_range);
    let mut sc = Scorers::load(&q.scoring, range);
    let mut notes = sc.notes.clone();
    if offsets.len() < 3 {
        notes.push(format!(
            "fidelity.offsets_s names {} lead(s): the T5 v2 rows are medians over at least 3 anchored-noise realizations (one clip's fine pitch error swings by a factor of two between two leads), so these are one or two draws",
            offsets.len()
        ));
    }
    progress(0.0, "loading the probed stream engine");
    let se = stream_engine(p, spec, &mut notes)?;
    let engine = se.get();
    let latency = probe_algorithmic_latency(
        s,
        engine.info.output_rate,
        engine.info.min_context_samples,
        engine.fixed_block(),
        engine.info.model_delay_ms,
    )?;
    let plan = engine.rvc.clone();
    let template = plan.as_ref().is_some_and(|pl| pl.format == SlotFormat::SigilRvc) && engine.fixed_block().is_some();
    if !template && (modes.contains(&CeilingMode::B) || modes.contains(&CeilingMode::Whole)) {
        return Err("ceiling.mode b / whole: only template voices (sigil-rvc-1) have these ceilings; use long_shape".into());
    }
    if opts.ablations && !template {
        return Err("scoring.analyzer v2+ablations: the oracle ablations need a template voice (sigil-rvc-1)".into());
    }
    // Template voices get every v2 ceiling (the floors, the identity row and the onset envelopes
    // need A, B and the whole utterance); `ceiling.mode` picks the primary, against which the v1
    // metrics are computed.
    let modes: Vec<CeilingMode> = if template {
        let mut all = modes.to_vec();
        for m in [CeilingMode::LongShape, CeilingMode::B, CeilingMode::Whole] {
            if !all.contains(&m) {
                all.push(m);
            }
        }
        all
    } else {
        modes.to_vec()
    };
    let modes = modes.as_slice();
    // The target-speaker set: the reference clips' speech, embedded with the v2 scorer.
    let set = reference_set(&mut sc, &q.fidelity.reference_clips, &mut notes);
    let scorers = sc.describe_v2();
    // The offline renderer (ceiling B, the whole ceiling) at the run's flowdec precision.
    let precision = match engine.info.flowdec_precision.as_deref() {
        Some("fp16") => PrecisionMode::Fp16,
        _ => PrecisionMode::Fp32,
    };
    let content_variant = plan
        .as_ref()
        .map(|pl| rvc_build::pinned_content_variant(&pl.spec.content_variant))
        .transpose()?
        .flatten()
        .map(str::to_string);
    let render = RenderOpts {
        precision,
        content_variant: content_variant.clone(),
        ..RenderOpts::default()
    };
    let mut offline = None;
    let mut results: Vec<OffsetResult> = Vec::new();
    for (oi, &lead) in offsets.iter().enumerate() {
        let base = oi as f64 / offsets.len() as f64;
        let span = 1.0 / offsets.len() as f64;
        let prog = |f: f64, m: &str| progress(0.05 + 0.85 * (base + f * span), &format!("lead {lead} s: {m}"));
        let r = score_offset(
            &OffsetInput {
                q,
                spec,
                p,
                engine,
                latency,
                plan: plan.as_ref(),
                template,
                modes,
                required: &required,
                clip16: &clip16,
                first: oi == 0,
                opts,
                set: set.as_ref().map(|(c, _)| c.as_slice()),
                render: &render,
                lead,
            },
            &mut sc,
            &mut offline,
            cancel,
            &prog,
        )?;
        results.push(r);
    }
    // Headline: medians over the offsets, spreads of the key metrics.
    let rows: Vec<Metrics> = results.iter().map(|r| r.metrics.clone()).collect();
    let (mut m, spreads) = multi_clip::median_over(&rows, QUALITY_V2_SPREAD);
    m.extend(spreads);
    m.insert("fidelity_offsets".into(), offsets.len() as f64);
    // T5 v2's MCD row: the median envelope MCD against ceiling A over its same-seed floor.
    if let (Some(e), Some(f)) = (m.get("env_mcd_db_vs_long_shape").copied(), m.get("env_mcd_floor_db").copied()) {
        m.insert("env_mcd_over_floor_db".into(), e - f);
    }
    m.insert("algorithmic_latency_ms".into(), latency.ms);
    m.insert("engine_load_ms".into(), engine.info.load_ms);
    if let Some(first) = results.first() {
        super::metrics::insert_summary(&mut m, "model_ms", &first.model_ms);
    }
    let mut details = serde_json::json!({
        "scorer": "v2",
        "analyzer": opts,
        "ceiling_modes": modes,
        "offsets_s": offsets,
        "lead_s": q.lead_s,
        "scorers": scorers,
        "target_set": set.as_ref().map(|(_, d)| d.clone()),
        "algorithmic_latency": latency,
        "render_precision": match precision { PrecisionMode::Fp16 => "fp16", _ => "fp32" },
        "content_variant": content_variant,
        "offsets": results.iter().map(|r| r.details.clone()).collect::<Vec<_>>(),
        "notes": notes,
    });
    // Ablations at the first offset (the oracles come from its whole-utterance analysis).
    if opts.ablations {
        progress(0.9, "oracle ablations");
        let first = results.first_mut().ok_or("no offset was scored")?;
        let base = first.abl.take().ok_or("the ablations need the first offset's long-shape and whole ceilings")?;
        let table = ablation::run_ablations(&base, &ablation::AblationContext {
            spec,
            q,
            engine,
            plan: plan.as_ref().ok_or("the ablations need a template voice")?,
            render: &render,
            set: set.as_ref().map(|(c, _)| c.as_slice()),
        }, &mut sc, &mut offline, cancel)?;
        // 0 when every oracle reached the generator; the rows say which did not.
        m.insert("ablation_rows_unapplied".into(), table.rows_unapplied() as f64);
        if let Some(o) = details.as_object_mut() {
            o.insert("ablations".into(), serde_json::to_value(&table).unwrap_or_default());
            if let Some(serde_json::Value::Array(run_notes)) = o.get_mut("notes") {
                for r in &table.rows {
                    for p in &r.problems {
                        run_notes.push(format!("ablation {}: NOT APPLIED, its recovered values mean nothing: {p}", r.ablation.as_str()).into());
                    }
                }
            }
        }
    }
    let mut outcome = RunOutcome {
        metrics: m,
        details,
        ..RunOutcome::default()
    };
    if spec.output.write_audio && write_files {
        for (oi, r) in results.into_iter().enumerate() {
            let suffix = if oi == 0 { String::new() } else { format!("_o{oi}") };
            let (audio, rate, ch) = r.stream_audio;
            if oi == 0 {
                outcome.audio = Some((audio, rate, ch));
            } else {
                outcome.extra_audio.push((format!("output{suffix}"), audio, rate, ch));
            }
            for (ri, rend) in r.renditions.into_iter().enumerate() {
                let name = if ri == 0 { format!("ceiling{suffix}") } else { format!("ceiling_{}{suffix}", rend.name) };
                outcome.extra_audio.push((name, rend.raw, rend.rate, 1));
            }
        }
    }
    progress(1.0, "done");
    Ok(outcome)
}

/// The target-speaker set: each reference clip's speech (at least 3 s of speech frames), embedded
/// with the v2 scorer; `(centroid, description)`.
fn reference_set(sc: &mut Scorers, clips: &[std::path::PathBuf], notes: &mut Vec<String>) -> Option<(Vec<f32>, serde_json::Value)> {
    if clips.is_empty() {
        return None;
    }
    let Some(sp) = sc.speaker_v2.as_mut() else {
        notes.push("target set: the v2 speaker scorer is not loaded; no set similarity".into());
        return None;
    };
    let mut embs = Vec::new();
    let mut used = Vec::new();
    for path in clips {
        let r = clip_16k(path).and_then(|x| {
            let mask = super::scorers::speech_frame_mask(&x, super::scorers::SPEAKER_SPEECH_RANGE_DB);
            let speech_s = mask.iter().filter(|&&v| v).count() as f64 / 100.0;
            if speech_s < MIN_SET_SPEECH_S {
                return Err(format!("{:.1} s of speech (at least {MIN_SET_SPEECH_S} s)", speech_s));
            }
            sp.embed_masked(&x, &mask).map(|e| (e, speech_s))
        });
        match r {
            Ok((e, secs)) => {
                embs.push(e);
                used.push(serde_json::json!({"clip": path.display().to_string(), "speech_s": secs}));
            }
            Err(e) => notes.push(format!("target set: {} skipped ({e})", path.display())),
        }
    }
    let c = super::scorers::centroid(&embs)?;
    Some((c, serde_json::json!({"clips": used, "members": embs.len()})))
}

/// Speech a target-set clip needs (s).
const MIN_SET_SPEECH_S: f64 = 3.0;
/// Noise-seed change of the different-seed floor (ceiling A drawn with another seed).
pub(crate) const FLOOR_SEED_XOR: u64 = 0x5EED_F100_D5EE_D000;
/// Block-grid shift of the same-seed floor (16 kHz samples: 100 ms).
pub(crate) const FLOOR_SHIFT_16K: u64 = 1_600;
/// Margin around the clip's speech of the whole-utterance ceiling's region (s).
const WHOLE_REGION_MARGIN_S: f64 = 0.3;

/// Inputs of one offset of a v2 run.
struct OffsetInput<'a> {
    q: &'a QualityRun,
    spec: &'a BenchSpec,
    p: &'a Prepared,
    engine: &'a Engine,
    latency: super::engine::AlgorithmicLatency,
    plan: Option<&'a RvcPlan>,
    template: bool,
    modes: &'a [CeilingMode],
    /// The ceilings whose failure fails the run (`ceiling.mode`, and the whole ceiling for the
    /// ablations); the rest of `modes` are optional.
    required: &'a [CeilingMode],
    /// The clip at 16 kHz mono (without the lead).
    clip16: &'a [f32],
    first: bool,
    opts: AnalyzerOpts,
    set: Option<&'a [f32]>,
    render: &'a RenderOpts,
    lead: f64,
}

/// The offline renderer, loaded on first use.
pub(crate) fn offline_voice<'v>(slot: &'v mut Option<OfflineVoice>, plan: &RvcPlan, render: &RenderOpts) -> Result<&'v mut OfflineVoice, String> {
    if slot.is_none() {
        *slot = Some(OfflineVoice::load(plan, render)?);
    }
    Ok(slot.as_mut().expect("loaded above"))
}

/// Leads of a v2 run after `lead_s` that `fidelity.offsets_s` does not name (s): three
/// anchored-noise realizations of the clip, as the T5 v2 rows are medians over at least three.
const DEFAULT_EXTRA_OFFSETS_S: [f64; 2] = [0.27, 0.61];

/// The leads (s) a v2 run scores the clip at: `fidelity.offsets_s`, else `lead_s` and two more.
pub(crate) fn v2_offsets(q: &QualityRun) -> Vec<f64> {
    if q.fidelity.offsets_s.is_empty() {
        std::iter::once(0.0).chain(DEFAULT_EXTRA_OFFSETS_S).map(|d| q.lead_s + d).collect()
    } else {
        q.fidelity.offsets_s.clone()
    }
}

/// Whether the whole-utterance ceiling fits `clip16` at `lead_s`: the clip's speech region (speech
/// within 50 dB of its loudest frame, 0.3 s around it) is at most [`WHOLE_MAX_REGION_16K`] long,
/// and starts at least [`WHOLE_MIN_START_S`] into the recording (the decoder's margin; the lead
/// silence before the clip counts).
pub(crate) fn whole_region_check(lead_s: f64, clip16: &[f32]) -> Result<(), String> {
    let mask = speech_frames(clip16, 50.0);
    let first = mask.iter().position(|&m| m).ok_or("the clip has no speech region")?;
    let last = mask.iter().rposition(|&m| m).unwrap_or(first);
    let margin_s = WHOLE_REGION_MARGIN_S;
    let clip_s = clip16.len() as f64 / 16_000.0;
    let from_s = (first as f64 * 0.01 - margin_s).max(0.0);
    let to_s = ((last + 1) as f64 * 0.01 + margin_s).min(clip_s);
    if lead_s + from_s < WHOLE_MIN_START_S {
        return Err(format!(
            "lead_s {lead_s} puts the speech region {:.2} s into the recording; the whole ceiling's decoder margin needs at least {WHOLE_MIN_START_S} s (use a lead_s of {WHOLE_MIN_START_S} s or more)",
            lead_s + from_s
        ));
    }
    if to_s - from_s > WHOLE_MAX_REGION_16K as f64 / 16_000.0 {
        return Err(format!(
            "the {:.1} s speech region is longer than the {} s the whole ceiling converts in one pass",
            to_s - from_s,
            WHOLE_MAX_REGION_16K / 16_000
        ));
    }
    Ok(())
}

/// The outcome of rendering the ceiling `mode`: a failure fails the run when `required` names it
/// (`ceiling.mode` asked for it), otherwise it is a note and `None` (the rendition's rows are
/// absent from the report).
fn optional_ceiling<T>(required: &[CeilingMode], mode: CeilingMode, result: Result<T, String>, notes: &mut Vec<String>) -> Result<Option<T>, String> {
    match result {
        Ok(v) => Ok(Some(v)),
        Err(e) if required.contains(&mode) => Err(e),
        Err(e) => {
            notes.push(format!("ceiling {} not rendered (its rows are absent): {e}", mode.as_str()));
            Ok(None)
        }
    }
}

/// One offset of a v2 run: stream pass, ceilings, v1 and v2 scores.
fn score_offset(
    x: &OffsetInput<'_>,
    sc: &mut Scorers,
    offline: &mut Option<OfflineVoice>,
    cancel: &AtomicBool,
    progress: &dyn Fn(f64, &str),
) -> Result<OffsetResult, String> {
    let (q, s) = (x.q, &x.spec.stream);
    let ch_in = usize::from(s.input_channels).max(1);
    let lead = (x.lead * f64::from(s.input_rate)).round() as usize * ch_in;
    let mut input = vec![0.0f32; lead];
    input.extend_from_slice(&x.p.input);
    let probe_ch = x.plan.map(|pl| pl.probe.clone()).filter(ProbeChannel::attached);
    let setup = ProbeSetup {
        fed_f0: true,
        stages: true,
        ..ProbeSetup::default()
    };
    progress(0.0, "streaming");
    let pass = stream_pass(x.engine, s, &input, probe_ch.as_ref().map(|c| (c, setup.clone())), cancel)?;
    let mut notes: Vec<String> = Vec::new();
    if let Some(e) = &pass.first_error {
        notes.push(format!("{} stream ticks failed (first: {e})", pass.tick_errors));
    }
    let stream_mono = pass.mono();
    let r_out = pass.rate;
    let input_mono = to_mono(&input, ch_in);
    let input_start = input_start_16k(&pass.rec, &input_mono, s.input_rate).map_err(|e| notes.push(format!("alignment: {e}"))).ok();
    let grid_phase = pass.rec.calls().first().map_or(0, |c| c.start_16k % super::ceiling::CONTENT_HOP_16K);
    // ---- ceilings ----
    progress(0.3, "ceilings");
    let mut rends: Vec<(CeilingMode, Rendition)> = Vec::new();
    let mut ceiling_fed: Option<(u64, Vec<f32>)> = None;
    let mut whole_analysis = None;
    let mut whole_region = None;
    let mut whole_fed: Option<(u64, Vec<f32>)> = None;
    let clip16 = x.clip16;
    for &mode in x.modes.iter().chain(std::iter::once(&CeilingMode::LongShape)) {
        if rends.iter().any(|(m, _)| *m == mode) {
            continue;
        }
        match mode {
            CeilingMode::LongShape => {
                let r = match (x.template, x.plan) {
                    (true, Some(plan)) => {
                        let o = long_shape(plan, s, &pass.rec, &q.ceiling, &LongShapeOpts {
                            probe: Some(setup.clone()),
                            ..LongShapeOpts::default()
                        })?;
                        ceiling_fed = o.fed_f0;
                        Rendition::from_ceiling("long_shape", o.out)?
                    }
                    _ => Rendition::from_ceiling("long_shape", ceiling(x.engine, s, &pass.rec, &q.ceiling)?)?,
                };
                rends.push((mode, r));
            }
            CeilingMode::B => {
                let rendered = (|| -> Result<Rendition, String> {
                    let voice = offline_voice(offline, x.plan.ok_or("ceiling B needs a template voice")?, x.render)?;
                    let o = render_blocks(voice, s, &pass.rec, grid_phase, x.render)?;
                    Rendition::from_ceiling("b", o.out)
                })();
                if let Some(r) = optional_ceiling(x.required, mode, rendered, &mut notes)? {
                    rends.push((mode, r));
                }
            }
            CeilingMode::Whole => {
                type Whole = (Rendition, Option<OfflineAnalysis>, (u64, Vec<f32>), (u64, u64));
                let rendered = (|| -> Result<Whole, String> {
                    whole_region_check(x.lead, clip16)?;
                    let start = input_start.ok_or("the whole ceiling needs where the input entered the stream (alignment failed)")?;
                    // The clip itself starts after the lead silence.
                    let clip_start = start + (lead / ch_in) as f64 * 16_000.0 / f64::from(s.input_rate.max(1));
                    let region = input_region(&pass.rec, clip_start, clip16, WHOLE_REGION_MARGIN_S).ok_or("the whole ceiling: the clip has no speech region")?;
                    let voice = offline_voice(offline, x.plan.ok_or("the whole ceiling needs a template voice")?, x.render)?;
                    let o = render_whole(voice, s, &pass.rec, region, grid_phase, x.render)?;
                    Ok((Rendition::from_ceiling("whole", o.out)?, o.analysis, o.fed_f0, region))
                })();
                if let Some((r, analysis, fed, region)) = optional_ceiling(x.required, mode, rendered, &mut notes)? {
                    whole_analysis = analysis;
                    whole_fed = Some(fed);
                    whole_region = Some(region);
                    rends.push((mode, r));
                }
            }
        }
    }
    // Keep the primary first (the order of `ceiling.mode`).
    rends.sort_by_key(|(m, _)| x.modes.iter().position(|y| y == m).unwrap_or(usize::MAX));
    let primary = &rends[0].1;
    // ---- v1 metrics against the primary ----
    progress(0.6, "scoring");
    let block_16k = x.engine.fixed_block();
    let stream_period = match block_16k {
        Some(b) => (b as u64 * u64::from(r_out) / 16_000) as usize,
        None => (u64::from(s.block_frames) * u64::from(r_out) / u64::from(s.input_rate)) as usize,
    };
    let expected_lag_16k = input_start.map(|p0| timeline_lag_16k(primary.start_16k, p0, x.latency.frames, r_out));
    let score = score_pair(
        sc,
        &PairInput {
            stream: &stream_mono,
            stream_rate: r_out,
            ceiling: &primary.raw,
            ceiling_rate: primary.rate,
            ceiling_start_16k: primary.start_16k,
            calls: pass.rec.calls(),
            stream_period,
            ceiling_period: (q.ceiling.block_ms / 1000.0 * f64::from(primary.rate)).round() as usize,
            join_window_ms: s.crossfade_ms + 5.0,
            max_offset_s: x.lead + 2.0 + primary.start_16k as f64 / 16_000.0,
            expected_lag_16k,
        },
    )?;
    let mut m = Metrics::new();
    score.quality.insert_into("", &mut m);
    put(&mut m, "ceiling_dnsmos_ovrl", score.ceiling_dnsmos_ovrl);
    if let (Some(c), Some(o)) = (score.ceiling_dnsmos_ovrl, score.quality.dnsmos_ovrl) {
        m.insert("dnsmos_ovrl_delta".into(), o - c);
    }
    put(&mut m, "ceiling_seam_flux_ratio", score.ceiling_seam_flux_ratio);
    if let (Some(c), Some(secs)) = (score.quality.clicks, score.quality.seconds) {
        m.insert("clicks_per_10min".into(), c as f64 / secs.max(1e-9) * 600.0);
    }
    // ---- the absolute region and the stream timeline ----
    let lag = score.alignment.lag.round() as i64;
    let stream = StreamTimeline {
        s16: resample(&stream_mono, r_out, 16_000)?,
        shift: lag - primary.start_16k as i64,
    };
    let (a, b) = (primary.start_16k + score.region.0 as u64, primary.start_16k + score.region.1 as u64);
    // ---- v2 against the primary ----
    let cs = primary.span(a, b).ok_or("the scored region is outside the primary ceiling")?.to_vec();
    let ss = stream.span(a, b).ok_or("the scored region is outside the stream")?.to_vec();
    let pv = pair_v2(sc, &cs, &ss, PairWants { speaker: true, dnsmos: false, set: x.set }, &mut notes);
    put(&mut m, "env_mcd_db", pv.env_mcd_db);
    put(&mut m, "speaker_cosine_v2", pv.speaker_cosine_v2);
    put(&mut m, "vuv_error_tolerant", pv.vuv_error_tolerant);
    if let Some(f) = &pv.f0 {
        put(&mut m, "f0_fine_rmse_cents", f.fine_rmse_cents);
        m.insert("f0_gross_frames".into(), f.gross_frames as f64);
    }
    put(&mut m, "speaker_set_sim", pv.set_sim_test);
    put(&mut m, "ceiling_speaker_set_sim", pv.set_sim_reference);
    // The envelope MCD against the long-shape ceiling (the T5 v2 MCD row; the primary's when it is A).
    let mut env_vs_a = if rends[0].0 == CeilingMode::LongShape { pv.env_mcd_db } else { None };
    // ---- clicks v2 ----
    let mut geo = JoinGeometry::of(s, x.engine.info.output_rate, x.engine.info.model_delay_ms);
    if let (Some(blk), Some(c)) = (block_16k, pass.rec.calls().first()) {
        geo = geo.with_tail(c.returned_16k.saturating_sub(blk) as f64, x.engine.info.output_rate);
    }
    let artifacts_abs: Vec<f64> = score.artifacts.iter().map(|t| t + primary.start_16k as f64).collect();
    let cl = clicks_v2(&artifacts_abs, pass.rec.calls(), &geo, a as f64, b as f64, s.crossfade_ms + 5.0, sc.click.tolerance_ms);
    m.insert("clicks_binomial".into(), cl.clicks_wide as f64);
    m.insert("clicks_narrow".into(), cl.clicks_narrow as f64);
    // ---- offset-corrected MCD ----
    let fc = sc.f0.contour(&cs).ok();
    let offset_corrected = offset_corrected_mcd(&stream, &cs, a, b, pass.rec.calls(), &geo, fc.as_deref());
    if let Some(o) = &offset_corrected {
        put(&mut m, "mcd_offset_corrected_db", Some(o.mcd_db));
        put(&mut m, "env_mcd_offset_corrected_db", o.env_mcd_db);
    }
    // ---- fed F0 ----
    let (fa, fb) = (a.div_ceil(160), b / 160);
    let mut fed_details = serde_json::Value::Null;
    if let Some(stream_fed) = &pass.fed_f0 {
        let mask = input_speech_frames(&pass.rec, fa, fb);
        let st = track_frames(stream_fed, fa, fb);
        let input_track = input_f0_track(sc, &pass.rec, s.pitch_semitones).map_err(|e| notes.push(format!("fed F0: {e}"))).ok();
        let vs_ceiling = ceiling_fed.as_ref().and_then(|c| fed_errors(&track_frames(c, fa, fb), &st, &mask));
        let vs_input = input_track.as_ref().and_then(|t| fed_errors(&track_frames(t, fa, fb), &st, &mask));
        let ceiling_vs_input = match (&input_track, &ceiling_fed) {
            (Some(t), Some(c)) => fed_errors(&track_frames(t, fa, fb), &track_frames(c, fa, fb), &mask),
            _ => None,
        };
        // The whole ceiling decodes with whole-clip F0 of the run's method: against the analyzer it
        // shows how far the two whole-clip trackers agree (a diagnostic of the reference).
        let whole_vs_input = match (&input_track, &whole_fed) {
            (Some(t), Some(w)) => fed_errors(&track_frames(t, fa, fb), &track_frames(w, fa, fb), &mask),
            _ => None,
        };
        put_fed(&mut m, "fed_", vs_ceiling.as_ref(), true);
        put_fed(&mut m, "fed_input_", vs_input.as_ref(), true);
        put_fed(&mut m, "ceiling_fed_input_", ceiling_vs_input.as_ref(), false);
        // The analyzer floor of the fine pitch error (ADR 0009): the run's tracker over the whole
        // clip (the whole ceiling's F0) against the analyzer.
        put_fed(&mut m, "whole_fed_input_", whole_vs_input.as_ref(), false);
        let gross = |e: &Option<FedF0Errors>| {
            e.as_ref()
                .map(|e| e.errors.gross.iter().map(|g| serde_json::json!([fa + g.frame as u64, g.cents.round()])).collect::<Vec<_>>())
        };
        let ceiling_track = ceiling_fed.as_ref().map(|c| track_frames(c, fa, fb));
        let input_frames = input_track.as_ref().map(|t| track_frames(t, fa, fb));
        let round = |v: &[Option<f32>]| v.iter().map(|x| x.map(|h| (h * 10.0).round() / 10.0)).collect::<Vec<_>>();
        fed_details = serde_json::json!({
            "frames": [fa, fb],
            // The grid check: the frame lag (±5) at which the stream's fed F0 best matches each
            // reference; anything but 0 means the tracks are not on one absolute grid.
            "grid_lag_vs_input": input_frames.as_ref().and_then(|t| track_lag(t, &st, &mask)),
            "grid_lag_vs_ceiling": ceiling_track.as_ref().and_then(|c| track_lag(c, &st, &mask)),
            "tracks": {"stream": round(&st), "ceiling": ceiling_track.as_deref().map(round), "input": input_frames.as_deref().map(round), "speech": mask.iter().map(|&b| u8::from(b)).collect::<Vec<_>>()},
            "vs_ceiling": vs_ceiling.as_ref().map(|e| serde_json::json!({"voiced_both": e.errors.v1.voiced_frames, "frames": e.errors.v1.frames, "gross": gross(&vs_ceiling), "onsets": e.onsets, "onsets_missed": e.onsets_missed, "onset_lags": e.onset_lags, "onset_lag_mean": e.onset_lag_mean()})),
            "vs_input": vs_input.as_ref().map(|e| serde_json::json!({"voiced_both": e.errors.v1.voiced_frames, "frames": e.errors.v1.frames, "gross": gross(&vs_input), "onsets": e.onsets, "onsets_missed": e.onsets_missed, "onset_lags": e.onset_lags, "onset_lag_mean": e.onset_lag_mean()})),
            "ceiling_vs_input": ceiling_vs_input.as_ref().map(|e| serde_json::json!({"gross": gross(&ceiling_vs_input), "fpe": e.errors.fine_rmse_cents})),
            "whole_vs_input": whole_vs_input.as_ref().map(|e| serde_json::json!({"gross": gross(&whole_vs_input), "fpe": e.errors.fine_rmse_cents, "gpe": e.errors.v1.gross_pitch_error, "vuv": e.errors.v1.vuv_error})),
        });
    } else if x.template {
        notes.push("fed F0: the stream recorded none (the engine probe was not reachable)".into());
    }
    // ---- the other ceilings ----
    let mut other = serde_json::Map::new();
    for (mode, r) in rends.iter().skip(1) {
        let (ra, rb) = intersect(a, b, &[r.extent(), stream.extent()]);
        let (Some(c), Some(t)) = (r.span(ra, rb), stream.span(ra, rb)) else {
            notes.push(format!("{}: the ceiling does not overlap the scored region", mode.as_str()));
            continue;
        };
        let pv = pair_v2(sc, c, t, PairWants { speaker: true, dnsmos: true, set: x.set }, &mut notes);
        let tag = mode.as_str();
        put(&mut m, &format!("mcd_db_vs_{tag}"), pv.mcd_db);
        put(&mut m, &format!("env_mcd_db_vs_{tag}"), pv.env_mcd_db);
        match mode {
            CeilingMode::Whole => {
                put(&mut m, "speaker_cosine_v2_vs_whole", pv.speaker_cosine_v2);
                put(&mut m, "whole_dnsmos_ovrl", pv.dnsmos_reference);
                put(&mut m, "whole_speaker_set_sim", pv.set_sim_reference);
            }
            CeilingMode::B => put(&mut m, "ceiling_b_dnsmos_ovrl", pv.dnsmos_reference),
            CeilingMode::LongShape => {
                env_vs_a = pv.env_mcd_db;
                put(&mut m, "long_shape_speaker_set_sim", pv.set_sim_reference);
            }
        }
        other.insert(mode.as_str().into(), serde_json::json!({"region_16k": [ra, rb], "scores": pv, "method": r.method, "compute_ms": r.compute_ms, "notes": r.notes}));
    }
    // ---- onset envelopes (a ceiling that mutes utterance onsets is not a ceiling) ----
    let onsets = utterance_onsets(&pass.rec, a, b);
    let speech: Vec<u64> = {
        let (fa, fb) = (a.div_ceil(160), b / 160);
        input_speech_frames(&pass.rec, fa, fb)
            .into_iter()
            .enumerate()
            .filter_map(|(i, s)| s.then_some(fa + i as u64))
            .collect()
    };
    let mut onset_details = serde_json::Map::new();
    let stream_onsets = onset_report(&stream.s16, -stream.shift, &onsets, &speech);
    put(&mut m, "onset_rel_db", stream_onsets.rel_db);
    onset_details.insert("stream".into(), serde_json::to_value(&stream_onsets).unwrap_or_default());
    for (mode, r) in &rends {
        let rep = onset_report(&r.audio16, r.start_16k as i64, &onsets, &speech);
        put(&mut m, &format!("onset_rel_db_{}", mode.as_str()), rep.rel_db);
        onset_details.insert(mode.as_str().into(), serde_json::to_value(&rep).unwrap_or_default());
    }
    onset_details.insert("onset_frames".into(), serde_json::json!(onsets));
    // The identity gate compares with the whole ceiling's similarity when there is one.
    let ceiling_sim = m.get("whole_speaker_set_sim").or_else(|| m.get("ceiling_speaker_set_sim")).copied();
    if let (Some(st), Some(c)) = (m.get("speaker_set_sim").copied(), ceiling_sim) {
        m.insert("speaker_set_sim_delta".into(), st - c);
    }
    // The fp32 row of the whole ceiling (when the run's flowdec is fp16).
    if x.render.precision == PrecisionMode::Fp16 && x.modes.contains(&CeilingMode::Whole) && x.first {
        if let (Some(region), Some(plan)) = (whole_region, x.plan) {
            let fp32 = RenderOpts {
                precision: PrecisionMode::Fp32,
                ..x.render.clone()
            };
            match OfflineVoice::load(plan, &fp32).and_then(|mut v| render_whole(&mut v, s, &pass.rec, region, grid_phase, &fp32)) {
                Ok(o) => {
                    let r = Rendition::from_ceiling("whole_fp32", o.out)?;
                    let (ra, rb) = intersect(a, b, &[r.extent(), stream.extent()]);
                    if let (Some(c), Some(t)) = (r.span(ra, rb), stream.span(ra, rb)) {
                        let pv = pair_v2(sc, c, t, PairWants { speaker: false, dnsmos: true, set: None }, &mut notes);
                        put(&mut m, "mcd_db_vs_whole_fp32", pv.mcd_db);
                        put(&mut m, "env_mcd_db_vs_whole_fp32", pv.env_mcd_db);
                        put(&mut m, "whole_fp32_dnsmos_ovrl", pv.dnsmos_reference);
                    }
                }
                Err(e) => notes.push(format!("whole ceiling fp32 row: {e}")),
            }
        }
    }
    // ---- floors (first offset) ----
    let mut floors = serde_json::Value::Null;
    if x.first {
        if let (true, Some(plan)) = (x.template, x.plan) {
            progress(0.8, "floors");
            floors = measure_floors(x, sc, &pass.rec, plan, &rends, a, b, &mut m, &mut notes)?;
        }
    }
    put(&mut m, "env_mcd_db_vs_long_shape", env_vs_a);
    // ---- CER (first offset) ----
    let mut transcripts = serde_json::Value::Null;
    if let (true, Some(stt)) = (x.first, &q.stt) {
        match cer_scores(stt, &x.p.input, s.input_rate, ch_in, &stream_mono, r_out, &primary.raw, primary.rate, &score) {
            Ok((c, t)) => {
                put(&mut m, "cer", c.0);
                put(&mut m, "cer_ceiling", c.1);
                put(&mut m, "cer_input", c.2);
                if let (Some(a), Some(b)) = (c.0, c.1) {
                    m.insert("cer_delta_pts".into(), (a - b) * 100.0);
                }
                transcripts = t;
            }
            Err(e) => notes.push(format!("CER: {e}")),
        }
    }
    let sola = sola_summary(pass.rec.calls());
    let details = serde_json::json!({
        "lead_s": x.lead,
        "alignment": score.alignment,
        "region_abs_16k": [a, b],
        "input_start_16k": input_start,
        "grid_phase_16k": grid_phase,
        "primary": rends[0].0,
        "ceilings": rends.iter().map(|(mode, r)| serde_json::json!({"mode": mode, "method": r.method, "start_16k": r.start_16k, "seconds": r.audio16.len() as f64 / 16_000.0, "compute_ms": r.compute_ms, "notes": r.notes})).collect::<Vec<_>>(),
        "v1": {"quality": score.quality, "artifact_phase": score.artifact_phase, "joins": {"artifacts_in_windows": score.join_artifacts, "expected_from_off_join_rate": score.join_expected, "window_coverage": score.join_coverage}},
        "clicks_v2": cl,
        "join_geometry": geo,
        "f0_gross": pv.f0.as_ref().map(|f| f.gross.iter().map(|g| serde_json::json!([a / 160 + g.frame as u64, g.cents.round()])).collect::<Vec<_>>()),
        "fed_f0": fed_details,
        "offset_corrected": offset_corrected,
        "sola": sola,
        "other_ceilings": other,
        "onset_envelopes": onset_details,
        "floors": floors,
        "probe_applied": format!("{:?}", pass.applied),
        "transcripts": transcripts,
        "calls": pass.rec.calls().iter().map(|c| serde_json::json!([c.start_16k, c.end_16k, c.returned_16k, c.sola_offset_16k, c.join_seen])).collect::<Vec<_>>(),
        "metrics": m,
        "notes": notes,
    });
    let abl = if x.first && x.opts.ablations {
        Some(ablation::AblationBase {
            input,
            rec: pass.rec,
            region: (a, b),
            input_start,
            stream_fed: pass.fed_f0,
            whole_analysis,
            whole_region,
            grid_phase,
            renditions: rends.iter().map(|(m, r)| (*m, r.clone())).collect(),
            stream: stream.clone(),
        })
    } else {
        None
    };
    Ok(OffsetResult {
        metrics: m,
        details,
        stream_audio: (pass.audio, r_out, pass.channels),
        renditions: rends.into_iter().map(|(_, r)| r).collect(),
        abl,
        model_ms: pass.model_ms,
    })
}

/// The floors of the first offset (see the module docs): ceiling A against itself with its blocks
/// shifted by 100 ms (the primary floor), drawn with another noise seed (an upper bound), and
/// against ceiling B (the offline floor); STFT and envelope MCD of each.
#[allow(clippy::too_many_arguments)]
fn measure_floors(
    x: &OffsetInput<'_>,
    sc: &mut Scorers,
    rec: &StreamRecorder,
    plan: &RvcPlan,
    rends: &[(CeilingMode, Rendition)],
    a: u64,
    b: u64,
    m: &mut Metrics,
    notes: &mut Vec<String>,
) -> Result<serde_json::Value, String> {
    let s = &x.spec.stream;
    let Some((_, ca)) = rends.iter().find(|(mode, _)| *mode == CeilingMode::LongShape) else {
        return Ok(serde_json::Value::Null);
    };
    let mut out = serde_json::Map::new();
    let mut floor = |name: &str, other: &Rendition, sc: &mut Scorers, m: &mut Metrics, notes: &mut Vec<String>, keys: (&str, &str)| {
        let (ra, rb) = intersect(a, b, &[ca.extent(), other.extent()]);
        if let (Some(r), Some(t)) = (ca.span(ra, rb), other.span(ra, rb)) {
            let pv = pair_v2(sc, r, t, PairWants::default(), notes);
            put(m, keys.0, pv.mcd_db);
            put(m, keys.1, pv.env_mcd_db);
            out.insert(name.into(), serde_json::json!({"region_16k": [ra, rb], "mcd_db": pv.mcd_db, "env_mcd_db": pv.env_mcd_db, "seconds": pv.seconds, "method": other.method}));
        } else {
            notes.push(format!("floor {name}: the renditions do not overlap the scored region"));
        }
    };
    let shifted = long_shape(plan, s, rec, &x.q.ceiling, &LongShapeOpts {
        shift_16k: FLOOR_SHIFT_16K,
        ..LongShapeOpts::default()
    })?;
    floor("same_seed_shifted_100ms", &Rendition::from_ceiling("a_shift", shifted.out)?, sc, m, notes, ("mcd_floor_db", "env_mcd_floor_db"));
    let seeded = long_shape(plan, s, rec, &x.q.ceiling, &LongShapeOpts {
        noise_seed: Some(plan.noise_seed ^ FLOOR_SEED_XOR),
        ..LongShapeOpts::default()
    })?;
    floor("different_seed", &Rendition::from_ceiling("a_seed", seeded.out)?, sc, m, notes, ("mcd_floor_seed_db", "env_mcd_floor_seed_db"));
    // Ceiling B was rendered with the other ceilings (an optional one: when it could not be, the
    // notes say why and the offline floor is absent).
    match rends.iter().find(|(mode, _)| *mode == CeilingMode::B) {
        Some((_, b_rend)) => floor("ceiling_b", b_rend, sc, m, notes, ("mcd_offline_floor_db", "env_mcd_offline_floor_db")),
        None => notes.push("floor ceiling_b: ceiling B was not rendered, so there is no offline floor".into()),
    }
    Ok(serde_json::Value::Object(out))
}

/// SOLA offsets of the recorded joins (16 kHz samples from the window centre): their spread and
/// the consecutive changes (a time-anchored stream only jumps when the offset changes).
fn sola_summary(calls: &[CallSpan]) -> serde_json::Value {
    let offs: Vec<f64> = calls.iter().filter_map(|c| c.sola_offset_16k).collect();
    if offs.is_empty() {
        return serde_json::Value::Null;
    }
    let deltas: Vec<f64> = calls
        .windows(2)
        .filter_map(|w| Some((w[1].sola_offset_16k? - w[0].sola_offset_16k?).abs()))
        .collect();
    let ncc: Vec<f64> = calls.iter().filter_map(|c| c.join_ncc.map(f64::from)).collect();
    let mean = |v: &[f64]| if v.is_empty() { None } else { Some(v.iter().sum::<f64>() / v.len() as f64) };
    serde_json::json!({
        "joins": offs.len(),
        // Calls whose join the stream loop did not observe (several model calls in one device
        // block; the harness keeps one join per block): not in the figures below.
        "calls_without_join_info": calls.iter().filter(|c| !c.join_seen).count(),
        "mean_abs_offset_ms": mean(&offs.iter().map(|v| v.abs() / 16.0).collect::<Vec<_>>()),
        "off_centre_1ms_share": offs.iter().filter(|v| v.abs() >= 16.0).count() as f64 / offs.len() as f64,
        "delta_1ms_share": if deltas.is_empty() { None } else { Some(deltas.iter().filter(|v| **v >= 16.0).count() as f64 / deltas.len() as f64) },
        "mean_ncc": mean(&ncc),
    })
}

/// A clip at 16 kHz mono.
pub(crate) fn clip_16k(path: &Path) -> Result<Vec<f32>, String> {
    let (x, sr, ch) = read_wav(path)?;
    resample(&to_mono(&x, ch), sr, 16_000)
}

/// The stream's scored region at 16 kHz.
fn region_16k(stream: &[f32], rate: u32, score: &PairScore) -> Result<Vec<f32>, String> {
    let s16 = resample(stream, rate, 16_000)?;
    let lag = score.alignment.lag.round() as isize;
    let (a, b) = score.region;
    let lo = (a as isize + lag).max(0) as usize;
    let hi = ((b as isize + lag).max(0) as usize).min(s16.len());
    Ok(s16[lo.min(hi)..hi].to_vec())
}

/// `(cer stream, cer ceiling, cer input)`.
type CerTriple = (Option<f64>, Option<f64>, Option<f64>);

/// `(cer stream, cer ceiling, cer input)` and the transcripts.
#[allow(clippy::too_many_arguments)]
fn cer_scores(
    stt: &super::scorers::SttSpec,
    input: &[f32],
    input_rate: u32,
    ch_in: usize,
    stream: &[f32],
    stream_rate: u32,
    ceiling_audio: &[f32],
    ceiling_rate: u32,
    score: &PairScore,
) -> Result<(CerTriple, serde_json::Value), String> {
    let mut t = Transcriber::create(stt)?;
    let input16 = resample(&channel(input, ch_in, 0), input_rate, 16_000)?;
    let text_input = t.transcribe(&input16)?;
    let stream16 = region_16k(stream, stream_rate, score)?;
    let text_stream = t.transcribe(&stream16)?;
    let c16 = resample(ceiling_audio, ceiling_rate, 16_000)?;
    let (a, b) = score.region;
    let text_ceiling = t.transcribe(&c16[a.min(c16.len())..b.min(c16.len())])?;
    let reference = if stt.reference_text.trim().is_empty() { text_input.clone() } else { stt.reference_text.clone() };
    let cer_input = (!stt.reference_text.trim().is_empty()).then(|| super::quality::cer(&reference, &text_input)).flatten();
    Ok((
        (
            super::quality::cer(&reference, &text_stream),
            super::quality::cer(&reference, &text_ceiling),
            cer_input,
        ),
        serde_json::json!({
            "engine": t.description,
            "reference": reference,
            "input": text_input,
            "stream": text_stream,
            "ceiling": text_ceiling,
        }),
    ))
}

#[cfg(test)]
mod tests {
    use super::super::report::RunStatus;
    use super::super::{run_spec, BenchContext, Progress};
    use super::*;

    fn quiet() -> impl Fn(&Progress) + Sync {
        |_: &Progress| {}
    }

    /// The stub (identity) streamed through the harness against its own offline ceiling: the two
    /// renditions are the same audio, so every fidelity metric is at its best and the plan's T5
    /// gates pass; the alignment finds the harness's delay.
    #[test]
    fn an_identity_model_is_at_its_ceiling() {
        for fixed in [0.0, 40.0] {
            let block = if fixed > 0.0 { 1920 } else { 512 };
            let spec = BenchSpec::from_value(serde_json::json!({
                "name": "q",
                "engine": {"kind": "stub", "output_rate": 40000, "fixed_block_ms": fixed},
                "input": {"kind": "synthetic", "signal": "voice", "seconds": 4},
                "stream": {"block_frames": block, "output_channels": 1},
                "runs": [{"mode": "quality", "name": "fid", "lead_s": 1.0,
                          "scoring": {"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio"},
                          "thresholds": {"mcd_db": {"max": 0.5}, "f0_rmse_cents": {"max": 15}, "vuv_error": {"max": 0.03},
                                         "clicks": {"max": 0}, "gross_pitch_error": {"max": 0.01}}}]
            }))
            .unwrap();
            let cancel = AtomicBool::new(false);
            let progress = quiet();
            let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
            let run = &r.runs[0];
            assert_eq!(run.status, RunStatus::Passed, "fixed {fixed}: {:?} {:?} {:?}", run.error, run.thresholds, run.details.get("notes"));
            let m = &run.metrics;
            assert!(m["mcd_db"] < 0.5, "{m:?}");
            assert!(m["quality_seconds"] > 3.0, "{m:?}");
            // The stream is the ceiling delayed by the harness: the offset is positive and near the
            // measured algorithmic latency (the ceiling starts at the recorded stream's origin).
            assert!(m["offset_ms"].abs() < 2_000.0, "{m:?}");
            // Placed on the timeline, which the envelope search confirms (to its 10 ms hop).
            let al = &run.details["alignment"];
            assert_eq!(al["method"], "timeline", "{al}");
            let (lag, env) = (al["lag"].as_f64().unwrap(), al["envelope_lag"].as_f64().unwrap());
            assert!((lag - env).abs() <= 160.0, "{al}");
            assert!(m.contains_key("block_ms_p95") && m.contains_key("ceiling_compute_ms"));
            assert!(!m.contains_key("speaker_cosine") && !m.contains_key("cer"));
        }
    }

    /// A model that pops after every block join: the quality scoring counts the clicks at the
    /// joins, the ceiling (one whole-utterance call) has none.
    #[test]
    fn clicks_at_joins_are_counted() {
        let voice = super::super::signals::synth(super::super::spec::SignalKind::Voice, 48_000, 4.0, 0.3, 5);
        let rec = Arc::new(Mutex::new(StreamRecorder::default()));
        let mut h = crate::services::neural_audio::VcStreamHarness::new(
            Box::new(RecordingPopper { rec: rec.clone() }),
            48_000,
            48_000,
            1,
            1,
            1_024,
            400,
            1_920,
        )
        .unwrap();
        let mut out = Vec::new();
        for block in voice.chunks_exact(1_920) {
            out.extend(h.push_and_convert(block).unwrap());
        }
        let rec = lock(&rec);
        let (start, data) = rec.stream();
        let ceiling = super::super::engine::identity_tail(data, data.len(), 40_000);
        // The timeline offset (recorded positions + probed harness latency) is where a waveform
        // search puts the stream, to the sample.
        let stream: super::super::spec::StreamSpec = serde_json::from_value(serde_json::json!({
            "input_rate": 48000, "output_rate": 48000, "input_channels": 1, "output_channels": 1,
            "block_frames": 1920, "extra_context_ms": 64, "crossfade_ms": 10
        }))
        .unwrap();
        let probe = super::super::engine::probe_algorithmic_latency(&stream, 40_000, 720, Some(640), 0.0).unwrap();
        let p0 = input_start_16k(&rec, &voice, 48_000).unwrap();
        let expected = timeline_lag_16k(start, p0, probe.frames, 48_000);
        let found = align(
            &resample(&ceiling, 40_000, 16_000).unwrap(),
            &resample(&out, 48_000, 16_000).unwrap(),
            16_000,
            32_000,
            true,
        )
        .unwrap();
        assert!(found.waveform && (found.lag - expected).abs() < 1.0, "timeline {expected} vs search {found:?}");
        let mut sc = Scorers {
            f0: F0Analyzer::dio(F0Range::Speech, Vec::new()),
            speaker: None,
            dnsmos: None,
            click: ClickConfig::default(),
            notes: Vec::new(),
            speaker_v2: None,
            analyzer: AnalyzerOpts::default(),
        };
        let score = score_pair(
            &mut sc,
            &PairInput {
                stream: &out,
                stream_rate: 48_000,
                ceiling: &ceiling,
                ceiling_rate: 40_000,
                ceiling_start_16k: start,
                calls: rec.calls(),
                stream_period: 1_920,
                ceiling_period: 8_000,
                join_window_ms: 20.0,
                max_offset_s: 2.0,
                expected_lag_16k: Some(expected),
            },
        )
        .unwrap();
        assert_eq!(score.alignment.method, "timeline");
        let joins_scored = score.quality.seconds.unwrap() / 0.04;
        let clicks = score.quality.clicks.unwrap() as f64;
        assert!(clicks >= 0.5 * joins_scored, "{clicks} clicks over {joins_scored} joins: {:?}", score.quality);
        assert!(score.quality.artifact_events.unwrap() as f64 >= clicks);
    }

    /// Identity with a pop at the start of every returned block, recording its input stream.
    struct RecordingPopper {
        rec: Arc<Mutex<StreamRecorder>>,
    }

    impl crate::services::neural_audio::vc::traits::VoiceConversionModel for RecordingPopper {
        fn convert(&mut self, ctx: &crate::services::neural_audio::vc::traits::ConversionCtx<'_>) -> Result<Vec<f32>, String> {
            lock(&self.rec).record(ctx);
            let mut out = super::super::engine::identity_tail(ctx.mono_16k, ctx.new_frames, 40_000);
            // A 0.2 pulse of 40 samples 12.5 ms into the returned span, past the 10 ms crossfade.
            for v in out.iter_mut().skip(500).take(40) {
                *v += 0.2;
            }
            Ok(out)
        }
        fn input_sample_rate(&self) -> u32 {
            16_000
        }
        fn output_sample_rate(&self) -> u32 {
            40_000
        }
        fn min_context_samples(&self) -> usize {
            720
        }
        fn fixed_block(&self) -> Option<usize> {
            Some(640)
        }
    }

    // ---- scorer v2 ----

    #[test]
    fn analyzer_and_ceiling_modes_parse() {
        assert_eq!(analyzer_opts("").unwrap(), AnalyzerOpts::default());
        assert_eq!(analyzer_opts("V1").unwrap(), AnalyzerOpts::default());
        assert_eq!(analyzer_opts("v2").unwrap(), AnalyzerOpts { v2: true, ablations: false });
        assert_eq!(analyzer_opts("v2+ablations").unwrap(), AnalyzerOpts { v2: true, ablations: true });
        assert!(analyzer_opts("v1+ablations").is_err());
        assert!(analyzer_opts("v3").is_err());
        assert!(analyzer_opts("v2+fast").is_err());
        assert_eq!(ceiling_modes("long_shape").unwrap(), vec![CeilingMode::LongShape]);
        assert_eq!(ceiling_modes("").unwrap(), vec![CeilingMode::LongShape]);
        assert_eq!(ceiling_modes("whole+long_shape, b+whole").unwrap(), vec![CeilingMode::Whole, CeilingMode::LongShape, CeilingMode::B]);
        assert!(ceiling_modes("long").is_err());
    }

    /// Onsets are found after pauses on the recorded input; a rendition whose onset is 20 dB down
    /// reads about 20 dB below one that is not, and the envelopes sit on the absolute frame grid.
    #[test]
    fn onset_envelopes_expose_a_muted_onset() {
        // Absolute 1 600..: 0.3 s silence, 0.5 s tone, 0.3 s silence, 0.5 s tone, 0.2 s silence.
        let tone = |n: usize| (0..n).map(|i| (0.3 * (i as f64 * 0.07).sin()) as f32).collect::<Vec<_>>();
        let mut x = vec![0.0f32; 4_800];
        x.extend(tone(8_000));
        x.extend(vec![0.0f32; 4_800]);
        x.extend(tone(8_000));
        x.extend(vec![0.0f32; 3_200]);
        let mut rec = StreamRecorder::default();
        rec.record(&crate::services::neural_audio::vc::traits::ConversionCtx {
            mono_16k: &x,
            new_frames: 640,
            pitch_shift_semitones: 0.0,
            index_rate: 0.0,
            protect: 0.33,
            speaker_id: 0,
            stream_pos_16k: 1_600,
            reference: None,
            singing: Default::default(),
        });
        let (a, b) = (1_600u64, 1_600 + x.len() as u64);
        let onsets = utterance_onsets(&rec, a, b);
        assert_eq!(onsets, vec![10 + 30, 10 + 30 + 50 + 30]);
        let speech: Vec<u64> = (40..90).chain(120..170).collect();
        let clean = onset_report(&x, 1_600, &onsets, &speech);
        assert!(clean.rel_db.unwrap().abs() < 0.5, "{clean:?}");
        // The second utterance's first 200 ms 20 dB down.
        let mut muted = x.clone();
        for v in &mut muted[(120 * 160 - 1_600)..(140 * 160 - 1_600)] {
            *v *= 0.1;
        }
        let m = onset_report(&muted, 1_600, &onsets, &speech);
        assert!(m.rel_db.unwrap() < clean.rel_db.unwrap() - 8.0, "{m:?}");
        assert_eq!(m.envelopes.len(), 2);
        assert_eq!(m.envelopes[0].len(), (ONSET_PRE_FRAMES + ONSET_POST_FRAMES) as usize);
        // Outside the rendition: no values.
        let late = onset_report(&x[16_000..], 1_600 + 16_000, &onsets, &speech);
        assert!(late.envelopes[0].iter().all(Option::is_none) && late.envelopes[1].iter().all(Option::is_some));
    }

    /// Blocks whose content sits a known number of samples off (their recorded SOLA offsets): the
    /// offset-corrected MCD reads each frame where its content is. A constant displacement (no
    /// discontinuity at the joins) is corrected exactly; with all offsets 0 it is the raw MCD;
    /// varying displacements come out below the raw MCD (the joins' own discontinuities remain, as
    /// in the stream); without recorded offsets the bounded search finds the displacements.
    #[test]
    fn offset_corrected_mcd_reads_blocks_where_their_content_is() {
        const BLK: usize = 1_600;
        let r = super::super::signals::synth(super::super::spec::SignalKind::Voice, 16_000, 3.0, 0.4, 9);
        let n = BLK * 28;
        // The region starts 400 samples in, so every shifted window lies inside the stream.
        let a = 400usize;
        let reference = &r[a..n];
        let geo = JoinGeometry::default();
        let run = |disp: &dyn Fn(usize) -> i64, recorded: bool, zero: bool| {
            let s16: Vec<f32> = (0..n).map(|x| r[(x as i64 + disp(x / BLK)).clamp(0, r.len() as i64 - 1) as usize]).collect();
            let calls: Vec<CallSpan> = (0..28)
                .map(|k| CallSpan {
                    end_16k: (BLK * k + BLK) as u64,
                    returned_16k: BLK,
                    start_16k: 0,
                    sola_offset_16k: recorded.then_some(if zero { 0.0 } else { disp(k) as f64 }),
                    join_ncc: None,
                    join_seen: recorded,
                })
                .collect();
            let stream = StreamTimeline { s16, shift: 0 };
            let raw = mcd_db(reference, &stream.s16[a..n], 16_000).unwrap();
            (raw, offset_corrected_mcd(&stream, reference, a as u64, n as u64, &calls, &geo, None).unwrap())
        };
        // Every block 37 samples late in content: corrected exactly.
        let (raw, fixed) = run(&|_| 37, true, false);
        assert!(raw > 0.0 && fixed.mcd_db == 0.0, "corrected {fixed:?} vs raw {raw}");
        assert_eq!(fixed.method, "recorded SOLA offsets");
        assert!((fixed.mean_abs_shift_ms - 37.0 / 16.0).abs() < 0.2, "{fixed:?}");
        let (_, searched) = run(&|_| 37, false, false);
        assert_eq!((searched.method, searched.mcd_db), ("searched (±6 ms)", 0.0), "{searched:?}");
        // Offsets recorded as 0: the raw MCD.
        let (raw0, zero) = run(&|_| 37, true, true);
        assert_eq!(zero.mcd_db.to_bits(), raw0.to_bits());
        // Varying displacements: closer than raw.
        let ds = [0i64, 14, -9, 22, 5, -17, 30, 0, -25, 11];
        let (raw, varied) = run(&|k| ds[k % ds.len()], true, false);
        assert!(varied.mcd_db < raw, "corrected {varied:?} vs raw {raw}");
    }

    /// v1 metrics keep the P3 formula; v2 metrics use the T5 v2 rows.
    #[test]
    fn t5_shortfall_scores_v1_and_v2_metrics() {
        let m = |kv: &[(&str, f64)]| kv.iter().map(|(k, v)| (k.to_string(), *v)).collect::<std::collections::BTreeMap<String, f64>>();
        assert_eq!(t5_shortfall_v2(&m(&[])), 0.0);
        let v1 = t5_shortfall_v2(&m(&[("speaker_cosine", 0.94), ("f0_rmse_cents", 30.0), ("vuv_error", 0.06), ("clicks", 1.0)]));
        assert!((v1 - (1.0 + 1.0 + 1.0 + 1.0)).abs() < 1e-9, "{v1}");
        // v2: a set-similarity delta of -0.04 is one limit short; FPE 15 c passes; GPE 0.6 % is
        // one limit over; the envelope MCD 0.9 dB over its floor half a limit; one narrow click.
        let v2 = t5_shortfall_v2(&m(&[
            ("speaker_cosine_v2", 0.90),
            ("speaker_set_sim_delta", -0.04),
            ("fed_input_fpe_cents", 15.0),
            ("fed_input_gpe", 0.006),
            ("fed_input_vuv_error", 0.02),
            ("f0_rmse_cents", 80.0),
            ("env_mcd_over_floor_db", 0.9),
            ("clicks_binomial", 0.0),
            ("clicks_narrow", 1.0),
            ("clicks", 9.0),
        ]));
        assert!((v2 - (1.0 + 0.0 + 1.0 + 0.0 + 0.5 + 1.0)).abs() < 1e-9, "{v2}");
        // The limit itself is the pre-registered number, never a run's own floor: an analyzer
        // tracker (RMVPE, floor 6 cents) is charged 30 cents against 15.
        let s = t5_shortfall_v2(&m(&[("fed_input_fpe_cents", 30.0), ("whole_fed_input_fpe_cents", 6.0)]));
        assert!((s - 1.0).abs() < 1e-9, "{s}");
        // A tracker that is not the analyzer's (FCPE on the CPU preset: its whole-clip track is
        // 34.5 cents from the analyzer's) is charged what exceeds its own floors, or every
        // configuration of the run would carry the same constant shortfall: 35.2 cents over a
        // 34.5 floor is within the limit; 50 over 34.5 is (15.5 - 15) / 15 over.
        let cpu = |fpe: f64, gpe: f64, vuv: f64| {
            t5_shortfall_v2(&m(&[
                ("fed_input_fpe_cents", fpe),
                ("fed_input_gpe", gpe),
                ("fed_input_vuv_error", vuv),
                ("whole_fed_input_fpe_cents", 34.5),
                ("whole_fed_input_gpe", 0.037),
                ("whole_fed_input_vuv_error", 0.0866),
            ]))
        };
        assert_eq!(cpu(35.2, 0.0382, 0.1029), 0.0);
        assert!((cpu(50.0, 0.037, 0.0866) - (50.0 - 34.5 - 15.0) / 15.0).abs() < 1e-9);
        // A floor under the threshold (the analyzer's own tracker) subtracts nothing.
        let gpu = t5_shortfall_v2(&m(&[("fed_input_fpe_cents", 22.9), ("whole_fed_input_fpe_cents", 5.7)]));
        assert!((gpu - (22.9 - 15.0) / 15.0).abs() < 1e-9, "{gpu}");
        // Without a set the advisory cosine counts; without fed F0 the output analyzer's split.
        let fallback = t5_shortfall_v2(&m(&[("speaker_cosine_v2", 0.94), ("f0_fine_rmse_cents", 30.0), ("gross_pitch_error", 0.0), ("vuv_error", 0.03)]));
        assert!((fallback - 2.0).abs() < 1e-9, "{fallback}");
    }

    /// A required ceiling that cannot be rendered fails the run; an optional one (the ceilings a
    /// template voice gets besides the ones `ceiling.mode` names) is a note and its rows are left out.
    #[test]
    fn only_a_required_ceiling_fails_the_run() {
        let mut notes = Vec::new();
        let long_shape_only = [CeilingMode::LongShape];
        let fail: Result<u8, String> = Err("the 90.0 s region is longer than 60 s".into());
        assert_eq!(optional_ceiling(&long_shape_only, CeilingMode::Whole, fail.clone(), &mut notes), Ok(None));
        assert_eq!(optional_ceiling(&long_shape_only, CeilingMode::B, Err::<u8, _>("no fp32 content encoder".into()), &mut notes), Ok(None));
        assert_eq!(notes.len(), 2);
        assert!(notes[0].contains("whole") && notes[0].contains("90.0 s") && notes[1].contains("ceiling b"), "{notes:?}");
        // Named by `ceiling.mode` (or required by the ablations): the failure stands.
        let with_whole = [CeilingMode::LongShape, CeilingMode::Whole];
        assert_eq!(optional_ceiling(&with_whole, CeilingMode::Whole, fail, &mut notes), Err("the 90.0 s region is longer than 60 s".to_string()));
        assert_eq!(notes.len(), 2, "a required failure adds no note, it is the error");
        assert_eq!(optional_ceiling(&with_whole, CeilingMode::B, Ok(7u8), &mut notes), Ok(Some(7)));
    }

    /// Whether the whole ceiling fits is known from the lead and the clip: the decoder's margin
    /// needs the speech region to start 0.4 s into the recording, and one pass takes 60 s.
    #[test]
    fn the_whole_ceiling_checks_its_lead_and_region_up_front() {
        let tone = |secs: f64| -> Vec<f32> { (0..(secs * 16_000.0) as usize).map(|i| (0.3 * (i as f64 * 0.07).sin()) as f32).collect() };
        // Speech from the first sample: the region starts at the lead.
        let clip = tone(3.0);
        let err = whole_region_check(0.25, &clip).unwrap_err();
        assert!(err.contains("lead_s 0.25") && err.contains("0.4"), "{err}");
        assert!(whole_region_check(0.0, &clip).is_err());
        assert!(whole_region_check(WHOLE_MIN_START_S, &clip).is_ok());
        assert!(whole_region_check(1.0, &clip).is_ok());
        // A clip that is silent for its first second has its speech region start later (0.3 s of
        // margin before the speech), so a short lead is enough.
        let mut quiet_first = vec![0.0f32; 16_000];
        quiet_first.extend(tone(2.0));
        assert!(whole_region_check(0.0, &quiet_first).is_ok());
        // More than 60 s of speech in one region.
        assert!(whole_region_check(1.0, &tone(59.0)).is_ok());
        let err = whole_region_check(1.0, &tone(61.0)).unwrap_err();
        assert!(err.contains("longer than the 60 s"), "{err}");
        // No clip at all.
        assert!(whole_region_check(1.0, &[]).is_err());
    }

    /// A v2 run that names no offsets scores three (the T5 v2 rows are medians over at least three
    /// anchored-noise realizations); a spec that names fewer is allowed, with a note.
    #[test]
    fn a_v2_run_defaults_to_three_offsets() {
        let spec = |offsets: serde_json::Value| {
            BenchSpec::from_value(serde_json::json!({
                "name": "q2",
                "engine": {"kind": "stub", "output_rate": 40000, "fixed_block_ms": 40},
                "input": {"kind": "synthetic", "signal": "voice", "seconds": 3},
                "stream": {"block_frames": 1920, "output_channels": 1},
                "runs": [{"mode": "quality", "name": "fid", "lead_s": 1.0, "fidelity": {"offsets_s": offsets},
                          "scoring": {"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio", "analyzer": "v2"}}]
            }))
            .unwrap()
        };
        let q = |s: &BenchSpec| match &s.runs[0] {
            super::super::spec::RunSpec::Quality(q) => q.clone(),
            other => panic!("{other:?}"),
        };
        let s = spec(serde_json::json!([]));
        let near = |got: &[f64], want: [f64; 3]| got.len() == 3 && got.iter().zip(want).all(|(g, w)| (g - w).abs() < 1e-12);
        assert!(near(&v2_offsets(&q(&s)), [1.0, 1.27, 1.61]), "{:?}", v2_offsets(&q(&s)));
        assert_eq!(v2_offsets(&q(&spec(serde_json::json!([4.0, 4.5])))), vec![4.0, 4.5]);
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&s, None, &BenchContext { cancel: &cancel, progress: &progress });
        let run = &r.runs[0];
        assert_eq!(run.status, RunStatus::Passed, "{:?}", run.error);
        assert_eq!(run.metrics["fidelity_offsets"], 3.0);
        let reported: Vec<f64> = run.details["offsets_s"].as_array().unwrap().iter().filter_map(|v| v.as_f64()).collect();
        assert!(near(&reported, [1.0, 1.27, 1.61]), "{reported:?}");
        let notes = |r: &super::super::report::RunReport| r.details["notes"].as_array().unwrap().iter().filter_map(|n| n.as_str()).filter(|n| n.contains("anchored-noise")).count();
        assert_eq!(notes(run), 0);
        // One named lead: scored once, with the note.
        let one = run_spec(&spec(serde_json::json!([1.0])), None, &BenchContext { cancel: &cancel, progress: &progress });
        assert_eq!(one.runs[0].status, RunStatus::Passed, "{:?}", one.runs[0].error);
        assert_eq!(one.runs[0].metrics["fidelity_offsets"], 1.0);
        assert_eq!(notes(&one.runs[0]), 1, "{:?}", one.runs[0].details["notes"]);
    }

    /// A device block of two engine blocks makes two model calls but the harness keeps one join:
    /// the stream loop attributes it to the newest call and leaves the earlier one unobserved
    /// (never "joined at the window centre"), and the summary says how many calls it is blind to.
    #[test]
    fn a_device_block_with_two_model_calls_leaves_the_first_join_unobserved() {
        // 20 ms engine blocks behind 40 ms device blocks: two calls per tick.
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "two-calls",
            "engine": {"kind": "stub", "output_rate": 40000, "fixed_block_ms": 20},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 3},
            "stream": {"block_frames": 1920, "output_channels": 1},
            "runs": [{"mode": "quality", "name": "fid"}]
        }))
        .unwrap();
        let priority = crate::services::rt_thread::WorkerPriority::from_str_lossy(&spec.stream.worker_priority);
        let engine = build_engine(&spec.engine, &spec.stream, priority, spec.seed).unwrap();
        let input = super::super::signals::synth(super::super::spec::SignalKind::Voice, 48_000, 3.0, 0.3, 5);
        let pass = stream_pass(&engine, &spec.stream, &input, None, &AtomicBool::new(false)).unwrap();
        let calls = pass.rec.calls();
        assert!(calls.len() > 100, "{}", calls.len());
        // Skip the first tick (priming): from then on calls come in pairs per device block, the
        // second of which is the one the harness's last join belongs to.
        let seen = calls.iter().filter(|c| c.join_seen).count();
        let unseen = calls.len() - seen;
        assert!(seen >= calls.len() / 2 - 2 && seen <= calls.len() / 2 + 2, "{seen} seen of {}", calls.len());
        assert!(unseen >= calls.len() / 2 - 2, "{unseen} unseen of {}", calls.len());
        // Observed joins carry their offsets; the unobserved ones never do.
        assert!(calls.iter().filter(|c| !c.join_seen).all(|c| c.sola_offset_16k.is_none() && c.join_ncc.is_none()));
        assert!(calls.iter().filter(|c| c.join_seen).filter_map(|c| c.sola_offset_16k).count() > calls.len() / 4);
        let summary = sola_summary(calls);
        assert_eq!(summary["calls_without_join_info"], unseen);
        assert!(summary["joins"].as_u64().unwrap() as usize <= seen);
        // And a one-call-per-block stream is fully observed.
        let spec1 = BenchSpec::from_value(serde_json::json!({
            "name": "one-call",
            "engine": {"kind": "stub", "output_rate": 40000, "fixed_block_ms": 40},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 3},
            "stream": {"block_frames": 1920, "output_channels": 1},
            "runs": [{"mode": "quality", "name": "fid"}]
        }))
        .unwrap();
        let engine1 = build_engine(&spec1.engine, &spec1.stream, priority, spec1.seed).unwrap();
        let pass1 = stream_pass(&engine1, &spec1.stream, &input, None, &AtomicBool::new(false)).unwrap();
        assert!(pass1.rec.calls().iter().all(|c| c.join_seen), "{:?}", pass1.rec.calls().iter().filter(|c| !c.join_seen).count());
    }

    /// Calls whose join was not observed are searched, not read as "no displacement": with every
    /// block's content 37 samples late, half of them observed (recorded offset) and half not, the
    /// corrected MCD is exact only because the unobserved half is searched.
    #[test]
    fn unobserved_joins_are_searched_not_assumed_centred() {
        const BLK: usize = 1_600;
        let r = super::super::signals::synth(super::super::spec::SignalKind::Voice, 16_000, 3.0, 0.4, 9);
        let n = BLK * 28;
        let a = 400usize;
        let reference = &r[a..n];
        let s16: Vec<f32> = (0..n).map(|x| r[(x + 37).min(r.len() - 1)]).collect();
        let stream = StreamTimeline { s16, shift: 0 };
        let calls = |observed: &dyn Fn(usize) -> bool| -> Vec<CallSpan> {
            (0..28)
                .map(|k| CallSpan {
                    end_16k: (BLK * k + BLK) as u64,
                    returned_16k: BLK,
                    start_16k: 0,
                    sola_offset_16k: observed(k).then_some(37.0),
                    join_ncc: None,
                    join_seen: observed(k),
                })
                .collect()
        };
        let geo = JoinGeometry::default();
        let all = offset_corrected_mcd(&stream, reference, a as u64, n as u64, &calls(&|_| true), &geo, None).unwrap();
        assert_eq!((all.method, all.searched_blocks, all.mcd_db), ("recorded SOLA offsets", 0, 0.0), "{all:?}");
        let half = offset_corrected_mcd(&stream, reference, a as u64, n as u64, &calls(&|k| k % 2 == 0), &geo, None).unwrap();
        assert_eq!(half.method, "recorded SOLA offsets, searched where unobserved");
        assert!(half.searched_blocks >= 13, "{half:?}");
        assert_eq!(half.mcd_db, 0.0, "the searched blocks find their 37-sample displacement: {half:?}");
        // The old reading (no offset = no displacement) would leave half the blocks 37 samples off.
        let assumed_zero: Vec<CallSpan> = calls(&|k| k % 2 == 0).into_iter().map(|c| CallSpan { join_seen: true, ..c }).collect();
        let wrong = offset_corrected_mcd(&stream, reference, a as u64, n as u64, &assumed_zero, &geo, None).unwrap();
        assert!(wrong.mcd_db > 0.0 && wrong.searched_blocks == 0, "{wrong:?}");
    }

    /// The join geometry is the harness's: at a 20 ms crossfade and 40 kHz the SOLA tail is 481
    /// samples, the window centre 201 output samples (80.4 at 16 kHz) after the returned span's
    /// start, the crossfade 320 samples at 16 kHz; a lookahead moves it back.
    #[test]
    fn join_geometry_follows_the_harness() {
        let s: StreamSpec = serde_json::from_value(serde_json::json!({"crossfade_ms": 20})).unwrap();
        let g = JoinGeometry::of(&s, 40_000, 0.0);
        assert_eq!(g.tail_16k, 481.0);
        assert!((g.centre_16k - 80.4).abs() < 1e-9 && (g.overlap_16k - 320.0).abs() < 1e-9, "{g:?}");
        let c = CallSpan {
            end_16k: 10_000,
            returned_16k: 1_121,
            start_16k: 2_960,
            sola_offset_16k: None,
            join_ncc: None,
            join_seen: false,
        };
        assert_eq!(g.returned_start(&c), 8_879.0);
        assert_eq!(g.block_16k(&c), 640.0);
        let la = JoinGeometry::of(&s, 40_000, 20.0);
        assert_eq!(la.returned_start(&c), 8_879.0 - 320.0);
        // The calls' own tail gives the same centre; a narrower search moves it.
        assert!((g.with_tail(481.0, 40_000).centre_16k - 80.4).abs() < 1e-9);
        let narrow = g.with_tail(361.0, 40_000);
        // 361 samples at 16 kHz = 903 at 40 kHz (902.5 rounded up); (903 - 800) / 2 = 51 samples.
        assert!((narrow.centre_16k - 0.4 * 51.0).abs() < 1e-9, "{narrow:?}");
    }

    /// Planted pops at both crossfade boundaries: the narrow windows find them (the placement is
    /// right), and the wide windows too.
    #[test]
    fn planted_pops_at_both_crossfade_boundaries_are_seam_clicks() {
        // At 10 ms crossfade, 40 kHz: chunk 2203 samples, block 1600, overlap 400, centre 101.
        // Crossfade end = chunk index 501 (this join); 1701 = the next join's crossfade start.
        for (pops, label) in [(vec![501usize], "crossfade end"), (vec![1_701usize], "crossfade start"), (vec![501, 1_701], "both")] {
            let (score, calls, out_sr) = popped_run(Popper::At(pops.clone()));
            // A pop inside the held-back tail enters SOLA's correlation and can move a join by up to
            // the search (and its pop off the boundary); one past the crossfade cannot.
            if pops == [501] {
                assert_centred(&calls);
            }
            let geo = JoinGeometry::of(&popper_stream(), out_sr, 0.0);
            let (a, b) = (score.region.0 as f64, score.region.1 as f64);
            let cl = clicks_v2(&score.artifacts, &calls, &geo, a, b, 15.0, 1.5);
            let joins = (b - a) / 640.0;
            eprintln!("{label}: {cl:?} over {joins:.0} joins");
            assert!(cl.clicks_narrow as f64 >= 0.5 * joins, "{label}: {cl:?} over {joins:.0} joins");
            assert!(cl.at_narrow as f64 >= 0.8 * joins, "{label}: {cl:?}");
            assert!(cl.coverage_narrow < 0.25 && cl.coverage_wide > 0.3, "{cl:?}");
        }
    }

    /// The null test: pops at random places of the emitted blocks (not at the joins) are artifact
    /// events but not seam clicks, under both tests, at N = 20-40 events; and uniformly random
    /// events on a SOLA grid read 0 at least 95 % of the time (the 97.5 % quantile of each test).
    #[test]
    fn events_away_from_the_joins_are_not_seam_clicks() {
        let (score, calls, out_sr) = popped_run(Popper::Random(7));
        assert_centred(&calls);
        let geo = JoinGeometry::of(&popper_stream(), out_sr, 0.0);
        let (a, b) = (score.region.0 as f64, score.region.1 as f64);
        let cl = clicks_v2(&score.artifacts, &calls, &geo, a, b, 15.0, 1.5);
        assert!((20..=60).contains(&cl.events), "{cl:?}");
        assert_eq!((cl.clicks_wide, cl.clicks_narrow), (0, 0), "{cl:?}");
        // Synthetic: 640-sample joins, events uniform over 4 s.
        let calls: Vec<CallSpan> = (0..100)
            .map(|k| CallSpan {
                end_16k: 1_121 + 640 * k as u64,
                returned_16k: 1_121,
                start_16k: 0,
                sola_offset_16k: None,
                join_ncc: None,
                join_seen: false,
            })
            .collect();
        let s: StreamSpec = serde_json::from_value(serde_json::json!({"crossfade_ms": 20})).unwrap();
        let geo = JoinGeometry::of(&s, 40_000, 0.0);
        let mut rng = super::super::signals::SplitMix64::new(11);
        for n in [20usize, 30, 40] {
            let (mut wide, mut narrow) = (0, 0);
            for _ in 0..200 {
                let ev: Vec<f64> = (0..n).map(|_| 3_200.0 + rng.unit() * 64_000.0).collect();
                let c = clicks_v2(&ev, &calls, &geo, 3_200.0, 67_200.0, 25.0, 1.5);
                wide += usize::from(c.clicks_wide > 0);
                narrow += usize::from(c.clicks_narrow > 0);
            }
            assert!(wide <= 10 && narrow <= 10, "N = {n}: false positives {wide} / {narrow} of 200");
        }
    }

    /// The stream of the popper runs: 48 kHz in and out, 40 ms device blocks, 10 ms crossfade.
    fn popper_stream() -> StreamSpec {
        serde_json::from_value(serde_json::json!({
            "input_rate": 48000, "output_rate": 48000, "input_channels": 1, "output_channels": 1,
            "block_frames": 1920, "extra_context_ms": 64, "crossfade_ms": 10
        }))
        .unwrap()
    }

    /// A popper through the harness, scored against its clean ceiling (v1's `score_pair`, which
    /// also returns the artifact events): `(score, calls, model output rate)`.
    fn popped_run(popper: Popper) -> (PairScore, Vec<CallSpan>, u32) {
        let voice = super::super::signals::synth(super::super::spec::SignalKind::Voice, 48_000, 4.0, 0.3, 5);
        let rec = Arc::new(Mutex::new(StreamRecorder::default()));
        let mut h = VcStreamHarness::new(Box::new(PlantedPopper { rec: rec.clone(), popper, calls: 0 }), 48_000, 48_000, 1, 1, 1_024, 400, 1_920).unwrap();
        let mut out = Vec::new();
        for block in voice.chunks_exact(1_920) {
            out.extend(h.push_and_convert(block).unwrap());
            lock(&rec).note_join(h.last_join(), 40_000);
        }
        let rec = lock(&rec);
        let (start, data) = rec.stream();
        let ceiling = super::super::engine::identity_tail(data, data.len(), 40_000);
        let probe = super::super::engine::probe_algorithmic_latency(&popper_stream(), 40_000, 720, Some(640), 0.0).unwrap();
        let p0 = input_start_16k(&rec, &voice, 48_000).unwrap();
        let expected = timeline_lag_16k(start, p0, probe.frames, 48_000);
        let mut sc = Scorers {
            f0: F0Analyzer::dio(F0Range::Speech, Vec::new()),
            speaker: None,
            dnsmos: None,
            click: ClickConfig::default(),
            notes: Vec::new(),
            speaker_v2: None,
            analyzer: AnalyzerOpts::default(),
        };
        // Calls on the ceiling's timeline (its sample 0 is absolute `start`).
        let calls: Vec<CallSpan> = rec
            .calls()
            .iter()
            .map(|c| CallSpan {
                end_16k: c.end_16k - start,
                start_16k: c.start_16k.saturating_sub(start),
                ..*c
            })
            .collect();
        let score = score_pair(
            &mut sc,
            &PairInput {
                stream: &out,
                stream_rate: 48_000,
                ceiling: &ceiling,
                ceiling_rate: 40_000,
                ceiling_start_16k: start,
                calls: rec.calls(),
                stream_period: 1_920,
                ceiling_period: 8_000,
                join_window_ms: 15.0,
                max_offset_s: 2.0,
                expected_lag_16k: Some(expected),
            },
        )
        .unwrap();
        (score, calls, 40_000)
    }

    /// The recorded SOLA offsets of an identity's joins sit at the window centre.
    fn assert_centred(calls: &[CallSpan]) {
        let offs: Vec<f64> = calls.iter().filter_map(|c| c.sola_offset_16k).collect();
        assert!(offs.len() > 50 && offs.iter().filter(|o| o.abs() < 1.0).count() * 10 >= offs.len() * 9, "{offs:?}");
    }

    #[derive(Clone)]
    enum Popper {
        /// Single-sample pops at these chunk indices of every call.
        At(Vec<usize>),
        /// One pop at a pseudo-random place of every other call's emitted block (seeded).
        Random(u64),
    }

    /// Identity with planted single-sample pops, recording its input stream.
    struct PlantedPopper {
        rec: Arc<Mutex<StreamRecorder>>,
        popper: Popper,
        calls: u64,
    }

    impl crate::services::neural_audio::vc::traits::VoiceConversionModel for PlantedPopper {
        fn convert(&mut self, ctx: &crate::services::neural_audio::vc::traits::ConversionCtx<'_>) -> Result<Vec<f32>, String> {
            lock(&self.rec).record(ctx);
            let mut out = super::super::engine::identity_tail(ctx.mono_16k, ctx.new_frames, 40_000);
            self.calls += 1;
            match &self.popper {
                Popper::At(at) => {
                    for &i in at {
                        if let Some(v) = out.get_mut(i) {
                            *v += 0.25;
                        }
                    }
                }
                Popper::Random(seed) => {
                    if self.calls.is_multiple_of(2) {
                        let mut rng = super::super::signals::SplitMix64::new(seed ^ self.calls.wrapping_mul(0x9E37_79B9));
                        // Inside the emitted block, away from both crossfades (501..1701).
                        let i = 560 + (rng.unit() * 1_080.0) as usize;
                        out[i] += 0.25;
                    }
                }
            }
            Ok(out)
        }
        fn input_sample_rate(&self) -> u32 {
            16_000
        }
        fn output_sample_rate(&self) -> u32 {
            40_000
        }
        fn min_context_samples(&self) -> usize {
            720
        }
        fn fixed_block(&self) -> Option<usize> {
            Some(640)
        }
    }

    /// A v2 run on the stub: three offsets, the v1 metrics next to the v2 ones, medians and
    /// spreads; the identity is at its ceiling under every v2 metric too.
    #[test]
    fn a_v2_run_scores_every_offset() {
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "q2",
            "engine": {"kind": "stub", "output_rate": 40000, "fixed_block_ms": 40},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 3},
            "stream": {"block_frames": 1920, "output_channels": 1},
            "runs": [{"mode": "quality", "name": "fid", "lead_s": 1.0,
                      "fidelity": {"offsets_s": [1.0, 1.25, 1.6]},
                      "scoring": {"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio", "analyzer": "v2"},
                      "thresholds": {"mcd_db": {"max": 0.5}, "clicks": {"max": 0}}}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        let run = &r.runs[0];
        assert_eq!(run.status, RunStatus::Passed, "{:?} {:?} {:?}", run.error, run.thresholds, run.details.get("notes"));
        let m = &run.metrics;
        assert_eq!(m["fidelity_offsets"], 3.0);
        for k in ["mcd_db", "env_mcd_db", "f0_rmse_cents", "f0_fine_rmse_cents", "clicks", "clicks_binomial", "clicks_narrow", "mcd_db_spread", "env_mcd_db_spread"] {
            assert!(m.contains_key(k), "{k}: {m:?}");
        }
        assert!(m["env_mcd_db"] < 0.5 && m["mcd_db"] < 0.5, "{m:?}");
        assert_eq!((m["clicks_binomial"], m["clicks_narrow"]), (0.0, 0.0));
        assert!(m.contains_key("mcd_offset_corrected_db"), "the stream loop recorded SOLA offsets: {m:?}");
        let offsets = run.details["offsets"].as_array().unwrap();
        assert_eq!(offsets.len(), 3);
        assert!(offsets.iter().all(|o| o["alignment"]["method"] == "timeline"), "{offsets:?}");
        // Different leads, different absolute positions.
        let starts: Vec<u64> = offsets.iter().map(|o| o["region_abs_16k"][0].as_u64().unwrap()).collect();
        assert!(starts[0] < starts[1] && starts[1] < starts[2], "{starts:?}");
        assert_eq!(run.details["scorer"], "v2");
        assert!(run.details["scorers"]["f0_analyzer_version"].as_str().unwrap().starts_with("dio-"));
        // B and whole are template-voice ceilings; v1 runs refuse v2 options.
        for (scoring, mode) in [("v2", "whole"), ("v1", "b")] {
            let bad = BenchSpec::from_value(serde_json::json!({
                "name": "q3",
                "engine": {"kind": "stub", "output_rate": 40000, "fixed_block_ms": 40},
                "input": {"kind": "synthetic", "signal": "voice", "seconds": 2},
                "stream": {"block_frames": 1920, "output_channels": 1},
                "runs": [{"mode": "quality", "name": "fid", "ceiling": {"mode": mode},
                          "scoring": {"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio", "analyzer": scoring}}]
            }))
            .unwrap();
            let r = run_spec(&bad, None, &BenchContext { cancel: &cancel, progress: &progress });
            assert_eq!(r.runs[0].status, RunStatus::Error, "{scoring} {mode}");
        }
    }

    /// The rescore mode reproduces a v1 run's numbers from its saved WAVs and report (the same
    /// lag, region and scorers), and adds the v2 metrics.
    #[test]
    fn a_v2_runs_offsets_rescore_as_one_clip() {
        let dir = std::env::temp_dir().join(format!("sigil-rescore-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let scoring = serde_json::json!({"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio", "analyzer": "v2"});
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "q",
            "engine": {"kind": "stub", "output_rate": 40000, "fixed_block_ms": 40},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 3},
            "stream": {"block_frames": 1920, "output_channels": 1},
            "output": {"write_audio": true},
            "runs": [{"mode": "quality", "name": "fid", "lead_s": 1.0, "fidelity": {"offsets_s": [1.0, 1.25, 1.6]}, "scoring": scoring}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, Some(&dir), &BenchContext { cancel: &cancel, progress: &progress });
        assert_eq!(r.runs[0].status, RunStatus::Passed, "{:?}", r.runs[0].error);
        std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&r).unwrap()).unwrap();
        let pairs: Vec<serde_json::Value> = ["", "_o1", "_o2"]
            .iter()
            .map(|o| serde_json::json!({"name": format!("p{o}"), "stream": dir.join(format!("fid.output{o}.wav")), "ceiling": dir.join(format!("fid.ceiling{o}.wav"))}))
            .collect();
        let rescore = BenchSpec::from_value(serde_json::json!({
            "name": "r",
            "engine": {"kind": "stub"},
            "runs": [{"mode": "rescore", "name": "again", "scoring": scoring, "pairs": pairs}]
        }))
        .unwrap();
        let rr = run_spec(&rescore, None, &BenchContext { cancel: &cancel, progress: &progress });
        let run = &rr.runs[0];
        assert_eq!(run.status, RunStatus::Passed, "{:?} {:?}", run.error, run.details.get("notes"));
        // Three offsets of one run: one clip, its median over the pairs.
        assert_eq!((run.metrics["clips"], run.metrics["pairs"]), (1.0, 3.0), "{:?}", run.metrics);
        assert_eq!(run.details["single_clip"], true);
        assert_eq!(run.details["pairs"].as_array().unwrap().len(), 3);
        // Each pair sits on its offset's timeline: the v1 numbers of every offset reproduce, so the
        // clip's median is the run's headline.
        for p in run.details["pairs"].as_array().unwrap() {
            for (k, c) in p["details"]["v1_check"].as_object().unwrap() {
                assert!(c["abs_diff"].as_f64().unwrap() < 1e-6, "{}: {k} {c}", p["name"]);
            }
        }
        for k in ["mcd_db", "f0_rmse_cents", "vuv_error"] {
            assert!((run.metrics[k] - r.runs[0].metrics[k]).abs() < 1e-6, "{k}: {} vs {}", run.metrics[k], r.runs[0].metrics[k]);
        }
        assert!(run.metrics.contains_key("mcd_db_spread") || run.metrics.contains_key("env_mcd_db_spread"), "{:?}", run.metrics);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rescore_reproduces_a_v1_run() {
        let dir = std::env::temp_dir().join(format!("sigil-rescore-{}-{}", std::process::id(), line!()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let scoring = serde_json::json!({"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio"});
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "q",
            "engine": {"kind": "stub", "output_rate": 40000, "fixed_block_ms": 40},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 3},
            "stream": {"block_frames": 1920, "output_channels": 2},
            "output": {"write_audio": true},
            "runs": [{"mode": "quality", "name": "fid", "lead_s": 1.0, "scoring": scoring}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = quiet();
        let r = run_spec(&spec, Some(&dir), &BenchContext { cancel: &cancel, progress: &progress });
        assert_eq!(r.runs[0].status, RunStatus::Passed, "{:?}", r.runs[0].error);
        std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&r).unwrap()).unwrap();
        let mut scoring_v2 = scoring.clone();
        scoring_v2["analyzer"] = "v2".into();
        let rescore = BenchSpec::from_value(serde_json::json!({
            "name": "r",
            "engine": {"kind": "stub"},
            "runs": [{"mode": "rescore", "name": "again", "scoring": scoring_v2,
                      "pairs": [{"name": "p", "stream": dir.join("fid.output.wav"), "ceiling": dir.join("fid.ceiling.wav")}]}]
        }))
        .unwrap();
        let rr = run_spec(&rescore, None, &BenchContext { cancel: &cancel, progress: &progress });
        let run = &rr.runs[0];
        assert_eq!(run.status, RunStatus::Passed, "{:?} {:?}", run.error, run.details.get("notes"));
        let (old, new) = (&r.runs[0].metrics, &run.metrics);
        for k in ["mcd_db", "f0_rmse_cents", "gross_pitch_error", "vuv_error", "seam_flux_ratio", "artifact_events", "quality_seconds", "offset_ms"] {
            if let Some(v) = old.get(k) {
                assert!((new[k] - v).abs() < 1e-6, "{k}: rescored {} vs {v}", new[k]);
            }
        }
        assert!(!new.contains_key("clicks"), "a v1 report keeps no call log: {new:?}");
        assert!(new.contains_key("env_mcd_db") && new.contains_key("f0_fine_rmse_cents"), "{new:?}");
        assert_eq!(new["clips"], 1.0);
        assert_eq!(run.details["single_clip"], true);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

