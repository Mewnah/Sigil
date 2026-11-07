//! Compare mode (plan T6): Sigil against the other voice changers on the same voice, the same
//! inputs and the same PC.
//!
//! # v1 (P3-F), kept for continuity
//! One input clip (`compare.input`, or the spec's input) and inline `candidates` (WAVs recorded
//! from other apps): each rendition is aligned to the input on energy envelopes (voices differ,
//! timing does not) and scored over the input's speech region: DNSMOS P.835 (SIG / BAK / OVRL), F0
//! against the input's F0 shifted by the run's transpose (one analyzer), CAM++ cosine against
//! `scoring.speaker_reference`, artifact events per minute (impulses of the rendition with no
//! impulse of the input within 12 ms), and the latency given. Its metrics are Sigil's row
//! (`sigil_*`) and the margins `dnsmos_ovrl_margin` (Sigil − the best competitor),
//! `gross_pitch_error_margin` (the best competitor − Sigil) and `latency_ratio` (Sigil / the fastest
//! competitor). They take no operating point, quality condition or sample size into account, so
//! they are reported for continuity only; the T6 decision is v2's. (Earlier docs said the w-okada
//! algorithm is "emulated exactly by `LegacyOnnxRvc`": it is not. `LegacyOnnxRvc` is the
//! bring-your-own ONNX slot format run through Sigil's own harness, SOLA and resamplers.)
//!
//! # v2 (P3B-C): corpus, operating points, fairness, paired statistics, Pareto
//! Every compare run also runs v2 (`details.v2`; metrics without a `sigil_` prefix). Inputs:
//! - **A corpus** (`compare.corpus`: a [`CorpusManifest`], `sigil-compare-corpus/1`, template
//!   `scripts/bench/corpus.example.json`): clips (each with its sha256, optional tracker-independent
//!   F0 truth from `scripts/vc-bench/egg_f0.py` and optional neutral references from
//!   `scripts/vc-bench/rvc_offline_reference.py`), operating points of other systems (`points`: one
//!   WAV per clip through `{clip}` in its path, plus the JSON sidecar the `scripts/vc-bench` drivers
//!   write next to it), live captures (`captures`: [`run_play_record`] through the system under
//!   test), Sigil's own extra rows (`sigil.points`, ablations included) and the pre-registered
//!   margins. Without a corpus the run's single input and inline `candidates` form a one-clip corpus.
//! - **Sigil's configurations** (`sigil.configs`): `shipping_default` (what a new install runs:
//!   Streamer, CPU, FCPE, the app's default stream), `autotuned` (the autotuner's stored result for
//!   this GPU and driver), `low_latency` and `quality` (the presets' settings on the spec's EP and
//!   engine block), each converted in-process through the production harness; empty = the spec's
//!   engine as the `sigil` row.
//!
//! Each operating point carries its metadata ([`Point`]: system, version, tier, config with block /
//! extra / crossfade / F0 method / index rate / protect / rms_mix / precision / EP / GPU, latency and
//! its kind, input and voice sha256, per-block compute). **Fairness**: a point whose transpose or
//! voice differs from the corpus's, or a rendition whose input hash differs from the clip's, is
//! refused (it never enters a verdict); missing metadata is flagged; latencies are compared only
//! between points of the same tier (offline renders vs live captures) and the same latency kind
//! (`e2e_loopback`, `declared`, `algorithmic`; a live offset nobody calibrated is `uncalibrated` and
//! is compared with nothing); an offline-emulated competitor point counts only when its per-block
//! p99 compute is below its block period on this run's GPU (unknown = undecided, never a silent
//! pass). Sigil's own rows meet the same rules. More fairness rules:
//! - **Every clip's sidecar counts** ([`Point::merge_sidecars`]): sidecars that disagree on the
//!   voice, transpose or block, or contradict what the point states, refuse the point; the per-block
//!   compute is the worst p95 / p99 over the clips (Sigil's own rows aggregate the same way).
//! - **A live capture never vouches for its voice**: its sidecar carries the voice the manifest
//!   asserts for the point, never the corpus's (`voice_asserted`); without one the point is flagged
//!   "voice not verified".
//! - **A row that misses clips decides nothing** ([`unscored_cells`]): a conversion error, a
//!   missing, refused, muted or unalignable rendition, or no DNSMOS where the clip needs one makes
//!   the point incomplete (`unscored_clips`, metrics `points_incomplete` / `clips_unscored`): a
//!   competitor row is undecided, and a Sigil row that missed clips is not used for any verdict.
//! - **Operating points** (`index_rate`, `protect`, `rms_mix`, F0 method) of a corpus competitor that
//!   differ from the Sigil rows' are flagged side by side (`operating point: ...`) and listed in
//!   the T6 decision's `caveats`: a pass over a point at another retrieval strength is not clean.
//! - **Stand-in references** (`stand_in_references`: Sigil's own whole-utterance render) are scored
//!   apart (`mcd_stand_in_db`, ablations only); only upstream references feed the T6 MCD row.
//! - **The GPU** of a Sigil row is the adapter its engine really used ([`engine_gpu`]), not the
//!   first one DXGI lists; the first such row names the run's GPU for the competitors' feasibility.
//! - **The blind A/B manifest** hears only the pairs a verdict would count: refused and
//!   real-time-infeasible points are listed under `make_test.skipped` instead.
//!
//! **Scoring** (per clip and point, over the input's speech region, after alignment):
//! - DNSMOS P.835 after loudness normalisation ([`integrated_loudness`], ITU-R BS.1770-4 gated,
//!   to `loudness_lufs`, default −26 LUFS), only on clips of at least 9.01 s (the model's window;
//!   shorter clips get an advisory tiled score, never used in a verdict);
//! - F0 against tracker-independent truth: the clip's EGG truth when it has one, else the consensus
//!   of the input's analyzers (RMVPE ∩ SwiftF0 ∩ CREPE-tiny ∩ DIO within 50 cents; frames where
//!   they disagree are not scored), times the transpose; the output is analysed by **every**
//!   analyzer (a pitch claim must hold under each), with the share of output frames where the
//!   analyzers agree, and each analyzer's self-referenced errors as an advisory;
//! - envelope MCD (and STFT MCD) to the clip's neutral (upstream) references (mean over seeds; the
//!   seed-to-seed distance is the stochastic floor);
//! - CAM++ (exact pooling) similarity to the target-speaker set (the other clips' references, or
//!   `scoring.speaker_reference`); advisory;
//! - artifact events per minute, and phase-locked seam clicks on the point's **own** block grid.
//!
//! **Statistics**: per-clip paired differences (Sigil point − competitor point) with the Student t
//! 95 % interval (the decision), the Wilcoxon / Hodges-Lehmann interval and a percentile bootstrap
//! over clips (cross-checks). **Verdicts** per competitor point and plan row (margins fixed in the
//! manifest before the run; at least `min_clips` clips, default 20):
//! - DNSMOS: some Sigil point at matched latency (≤ the competitor's, same kind) with mean
//!   difference ≥ `dnsmos_superiority` (+0.05) and t lower bound > 0;
//! - GPE and MCD-to-reference: non-inferiority, t upper bound of (Sigil − competitor) ≤ the margin,
//!   at matched latency; GPE under every output analyzer;
//! - latency ≤ 0.7×: Pareto, some Sigil point with latency ≤ 0.7× the competitor's (same kind) that
//!   is non-inferior on every quality row measured.
//! T6 passes only when every counted competitor point passes every row and no point was refused.
//! Ablations (`role: ablation`) are paired with the Sigil row they ablate and never counted as
//! competitors. The details also carry the Pareto data, the corpus sizing (clips the +0.05 DNSMOS row
//! needs at the measured paired SD) and a `tools/listening-test/make_test.py` manifest of the
//! aligned, trimmed, loudness-matched renditions (written as `<run>.pair.<clip>.<point>.wav` when the
//! spec writes audio); `scripts/vc-bench/compare_export.py` writes both as files.
//!
//! # Frozen by P3-S0 (signatures; P3-F implements)
//! [`CompareRow`] (gained fields), [`compare_wavs`].

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use super::ceiling::{render_whole, OfflineVoice, RenderOpts, StreamRecorder};
use super::dsp::align;
use super::e2e_loopback::{run_play_record, PlayRecordOutcome, PlayRecordSpec};
use super::engine::{build_engine, probe_algorithmic_latency, AlgorithmicLatency, Engine, EngineInfo};
use super::fidelity::{clip_16k, Scorers};
use super::metrics::{paired, Metrics};
use super::multi_clip::mean_ci;
use super::offline::{run_offline, OfflineInput};
use super::quality::{click_events, click_report, envelope_mcd_db, f0_errors_masked, f0_errors_v2, mcd_db, speech_frames, unmatched, F0Analyzer, QualityMetrics};
use super::report::{RunOutcome, RunReport};
use super::scorers::{centroid, cosine, file_sha256, speech_frame_mask, SpeakerEmbedder, DNSMOS_LEN, SPEAKER_SPEECH_RANGE_DB};
use super::signals::{channel, read_wav, render_input, resample, to_mono, write_wav};
use super::spec::{BenchSpec, CallMsStats, CompareRun, EngineSpec, InputSpec, OfflineRun, ScoringSpec, StreamSpec};
use super::Prepared;
use crate::services::diagnostics::gpu_mem::{self, nvidia_adapter_indices, GpuAdapter};
use crate::services::neural_audio::vc::engine::options::F0Range;
use crate::services::neural_audio::vc::manifest::SlotFormat;
use crate::services::neural_audio::vc::traits::ConversionCtx;
use crate::services::rt_thread::WorkerPriority;

// =============================================================================================
// v1 (P3-F)
// =============================================================================================

/// One candidate's scores.
#[derive(Clone, Debug, Default, Serialize)]
pub struct CompareRow {
    pub name: String,
    pub wav: PathBuf,
    pub quality: QualityMetrics,
    pub latency_ms: Option<f64>,
    // ---- P3-F ----
    /// Impulses the conversion added, per minute of speech.
    pub artifact_events_per_min: Option<f64>,
    /// Why a metric is missing, or the row failed.
    pub notes: Vec<String>,
    /// Scored (false: the row could not be scored; see `notes`).
    pub ok: bool,
}

/// Matching tolerance of input and candidate impulses (envelope alignment is 10 ms accurate).
const EVENT_MATCH_S: f64 = 0.012;
/// Silence kept around the input's speech region.
const REGION_MARGIN_S: f64 = 0.1;

/// The input, analysed once.
struct InputAnalysis {
    audio: Vec<f32>,
    /// F0 of the input shifted by the transpose (Hz per 10 ms frame).
    f0: Vec<f32>,
    speech: Vec<bool>,
    /// Speech region (16 kHz samples).
    region: (usize, usize),
    /// Impulse events (s).
    events: Vec<f64>,
}

/// The speech region of a 16 kHz clip (100 ms around its first and last speech frame).
fn speech_region(speech: &[bool], len: usize) -> Option<(usize, usize)> {
    let first = speech.iter().position(|&m| m)?;
    let last = speech.iter().rposition(|&m| m).unwrap_or(first);
    let margin = (REGION_MARGIN_S * 16_000.0) as usize;
    Some(((first * 160).saturating_sub(margin), ((last + 1) * 160 + margin).min(len)))
}

impl InputAnalysis {
    fn new(audio16: Vec<f32>, semitones: f32, sc: &mut Scorers) -> Result<InputAnalysis, String> {
        let speech = speech_frames(&audio16, 40.0);
        let region = speech_region(&speech, audio16.len()).ok_or("the compare input is silent")?;
        let shift = 2f32.powf(semitones / 12.0);
        let f0: Vec<f32> = sc.f0.contour(&audio16)?.into_iter().map(|f| f * shift).collect();
        let events = click_events(&audio16, 16_000, &sc.click)
            .into_iter()
            .map(|e| e as f64 / 16_000.0)
            .collect();
        Ok(InputAnalysis {
            audio: audio16,
            f0,
            speech,
            region,
            events,
        })
    }
}

fn to_ticks(v: &[f64]) -> Vec<usize> {
    v.iter().map(|t| (t.max(0.0) * 1e4).round() as usize).collect()
}

/// What a candidate is scored against.
struct Against<'a> {
    input: &'a InputAnalysis,
    /// A clip of the target voice (speaker cosine).
    reference: Option<&'a [f32]>,
    max_offset_s: f64,
}

/// Score one candidate (16 kHz mono) against the input.
fn score_candidate(name: &str, wav: &Path, cand: &[f32], latency_ms: Option<f64>, against: &Against<'_>, sc: &mut Scorers) -> CompareRow {
    let (ia, reference, max_offset_s) = (against.input, against.reference, against.max_offset_s);
    let mut row = CompareRow {
        name: name.to_string(),
        wav: wav.to_path_buf(),
        latency_ms,
        ..CompareRow::default()
    };
    let Some(al) = align(&ia.audio, cand, 16_000, (max_offset_s * 16_000.0) as usize, false) else {
        row.notes.push("could not be aligned to the input (too short or silent)".into());
        return row;
    };
    let lag = al.lag.round() as isize;
    let (ra, rb) = ia.region;
    let a = ra.max((-lag).max(0) as usize);
    let b = rb.min((cand.len() as isize - lag).max(0) as usize);
    if b <= a + 8_000 {
        row.notes.push(format!("overlaps the input's speech on {} ms only", b.saturating_sub(a) / 16));
        return row;
    }
    let seg = &cand[(a as isize + lag) as usize..(b as isize + lag) as usize];
    let mut q = QualityMetrics {
        offset_ms: Some(al.lag / 16.0),
        seconds: Some((b - a) as f64 / 16_000.0),
        ..QualityMetrics::default()
    };
    if let Some(d) = sc.dnsmos.as_mut() {
        match d.score(seg) {
            Ok(v) => {
                q.dnsmos_ovrl = Some(v.ovrl);
                q.dnsmos_sig = Some(v.sig);
                q.dnsmos_bak = Some(v.bak);
            }
            Err(e) => row.notes.push(format!("DNSMOS: {e}")),
        }
    }
    match sc.f0.contour(seg) {
        Ok(fc) => {
            let (fa, fb) = (a / 160, b / 160);
            let fi = &ia.f0[fa.min(ia.f0.len())..fb.min(ia.f0.len())];
            let mask = &ia.speech[fa.min(ia.speech.len())..fb.min(ia.speech.len())];
            let n = fi.len().min(fc.len()).min(mask.len());
            match f0_errors_masked(&fi[..n], &fc[..n], &mask[..n]) {
                Ok(e) => {
                    q.f0_rmse_cents = Some(e.rmse_cents);
                    q.gross_pitch_error = Some(e.gross_pitch_error);
                    q.vuv_error = Some(e.vuv_error);
                    q.f0_frames = Some(e.frames as u64);
                }
                Err(e) => row.notes.push(format!("F0: {e}")),
            }
        }
        Err(e) => row.notes.push(format!("F0: {e}")),
    }
    if let (Some(r), Some(sp)) = (reference, sc.speaker.as_mut()) {
        match sp.similarity(r, seg) {
            Ok(c) => q.speaker_cosine = Some(c),
            Err(e) => row.notes.push(format!("speaker: {e}")),
        }
    }
    // Impulses the conversion added (on the input's timeline).
    let cand_events: Vec<f64> = click_events(cand, 16_000, &sc.click)
        .into_iter()
        .map(|e| (e as f64 - lag as f64) / 16_000.0)
        .filter(|&t| t >= a as f64 / 16_000.0 && t < b as f64 / 16_000.0)
        .collect();
    let added = unmatched(&to_ticks(&cand_events), &to_ticks(&ia.events), (EVENT_MATCH_S * 1e4) as usize);
    q.artifact_events = Some(added.len() as u64);
    row.artifact_events_per_min = Some(added.len() as f64 / ((b - a) as f64 / 16_000.0 / 60.0));
    row.quality = q;
    row.ok = true;
    row
}

/// Score each `(name, wav)` against the `reference_input` clip with the default scorers (offline
/// RMVPE or DIO, CAM++ / DNSMOS when installed), no transpose. The frozen library entry point of
/// the compare mode (spec runs use `run_compare`).
#[allow(dead_code)] // P3-S0 frozen API; the spec's compare mode shares its scoring.
pub fn compare_wavs(reference_input: &Path, candidates: &[(String, PathBuf)]) -> Result<Vec<CompareRow>, String> {
    let mut sc = Scorers::load(&ScoringSpec::default(), F0Range::Speech);
    let ia = InputAnalysis::new(clip_16k(reference_input)?, 0.0, &mut sc)?;
    let against = Against {
        input: &ia,
        reference: None,
        max_offset_s: 2.0,
    };
    Ok(candidates
        .iter()
        .map(|(name, wav)| match clip_16k(wav) {
            Ok(c) => score_candidate(name, wav, &c, None, &against, &mut sc),
            Err(e) => CompareRow {
                name: name.clone(),
                wav: wav.clone(),
                notes: vec![e],
                ..CompareRow::default()
            },
        })
        .collect())
}

/// Sigil's row and the v1 margins (see the module docs).
fn compare_metrics(rows: &[CompareRow]) -> Metrics {
    let mut m = Metrics::new();
    let sigil = rows.iter().find(|r| r.name == "sigil" && r.ok);
    let competitors: Vec<&CompareRow> = rows.iter().filter(|r| r.name != "sigil" && r.ok).collect();
    m.insert("candidates".into(), rows.iter().filter(|r| r.name != "sigil").count() as f64);
    m.insert("candidates_scored".into(), competitors.len() as f64);
    let best = |f: fn(&CompareRow) -> Option<f64>, max: bool| -> Option<f64> {
        let vals = competitors.iter().filter_map(|r| f(r));
        if max {
            vals.fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.max(v))))
        } else {
            vals.fold(None, |a: Option<f64>, v| Some(a.map_or(v, |a| a.min(v))))
        }
    };
    let ovrl = |r: &CompareRow| r.quality.dnsmos_ovrl;
    let gpe = |r: &CompareRow| r.quality.gross_pitch_error;
    let lat = |r: &CompareRow| r.latency_ms;
    if let Some(s) = sigil {
        let mut put = |k: &str, v: Option<f64>| {
            if let Some(v) = v.filter(|v| v.is_finite()) {
                m.insert(k.to_string(), v);
            }
        };
        put("sigil_dnsmos_ovrl", s.quality.dnsmos_ovrl);
        put("sigil_dnsmos_sig", s.quality.dnsmos_sig);
        put("sigil_dnsmos_bak", s.quality.dnsmos_bak);
        put("sigil_gross_pitch_error", s.quality.gross_pitch_error);
        put("sigil_f0_rmse_cents", s.quality.f0_rmse_cents);
        put("sigil_vuv_error", s.quality.vuv_error);
        put("sigil_speaker_cosine", s.quality.speaker_cosine);
        put("sigil_latency_ms", s.latency_ms);
        put("sigil_artifact_events_per_min", s.artifact_events_per_min);
    }
    if let Some(b) = best(ovrl, true) {
        m.insert("best_competitor_dnsmos_ovrl".into(), b);
        if let Some(s) = sigil.and_then(ovrl) {
            m.insert("dnsmos_ovrl_margin".into(), s - b);
        }
    }
    if let Some(b) = best(gpe, false) {
        m.insert("min_competitor_gross_pitch_error".into(), b);
        if let Some(s) = sigil.and_then(gpe) {
            m.insert("gross_pitch_error_margin".into(), b - s);
        }
    }
    if let Some(b) = best(lat, false) {
        m.insert("best_competitor_latency_ms".into(), b);
        if let Some(s) = sigil.and_then(lat) {
            if b > 0.0 {
                m.insert("latency_ratio".into(), s / b);
            }
        }
    }
    m
}

/// Sigil's latency of a v1 run: the fixed value, or a metric of an earlier run of the spec.
fn sigil_latency_v1(c: &CompareRun, previous: &[RunReport], notes: &mut Vec<String>) -> Option<f64> {
    match c.sigil.latency_ms {
        Some(l) => Some(l),
        None if !c.sigil.latency_from.trim().is_empty() => {
            let from = c.sigil.latency_from.trim();
            let v = previous
                .iter()
                .find(|r| r.name == from)
                .and_then(|r| r.metrics.get(c.sigil.latency_metric.trim()).copied());
            if v.is_none() {
                notes.push(format!("sigil latency: run '{from}' has no {}", c.sigil.latency_metric));
            }
            v
        }
        None => None,
    }
}

// =============================================================================================
// v2 (P3B-C): corpus manifest and driver sidecars
// =============================================================================================

/// Schema id of the compare corpus manifest.
pub const CORPUS_SCHEMA: &str = "sigil-compare-corpus/1";
/// Schema id of the JSON sidecar a `scripts/vc-bench` driver writes next to each WAV
/// (`<wav>.json`).
pub const SIDECAR_SCHEMA: &str = "sigil-vc-bench-sidecar/1";

/// The corpus of a compare v2 run (`compare.corpus`; paths relative to the manifest's folder).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusManifest {
    #[serde(default = "corpus_schema")]
    pub schema: String,
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default, rename = "$comment")]
    pub comment: Option<serde_json::Value>,
    /// The voice every system converted with.
    #[serde(default)]
    pub voice: CorpusVoice,
    /// Transpose every system used (semitones); `None` = the run's `semitones`, and when both are
    /// given they must agree.
    #[serde(default)]
    pub transpose: Option<f32>,
    pub clips: Vec<CorpusClip>,
    /// Other systems' operating points (and captured Sigil live renditions).
    #[serde(default)]
    pub points: Vec<PointSpec>,
    /// Live captures through the system under test (the play-and-record mode).
    #[serde(default)]
    pub captures: Vec<CaptureSpec>,
    #[serde(default)]
    pub sigil: SigilSection,
    /// Pre-registered margins of the verdicts.
    #[serde(default)]
    pub margins: Margins,
    /// Fewest clips a verdict is decided on (fewer: undecided, the statistics still reported).
    #[serde(default = "default_min_clips")]
    pub min_clips: usize,
    /// F0 analyzers of the consensus truth and of the output side (`rmvpe`, `swiftf0`, `crepe`,
    /// `dio`, `fcpe`, `pesto`); one that does not load is dropped with a note.
    #[serde(default = "default_analyzers")]
    pub analyzers: Vec<String>,
    /// Loudness every rendition is normalised to before DNSMOS and in the listening pairs (LUFS).
    #[serde(default = "default_loudness_lufs")]
    pub loudness_lufs: f64,
    #[serde(default)]
    pub listening: ListeningSpec,
    /// Write every analyzer's F0 of each input clip into the report (`details.v2.clips[].f0`, Hz
    /// per 10 ms frame before the transpose): `scripts/vc-bench/fcpe_check.py` compares Sigil's
    /// FCPE with upstream torchfcpe on them.
    #[serde(default)]
    pub dump_f0: bool,
    /// Stand-in references for a dry run before upstream renders exist: the spec's template voice
    /// converting each whole clip offline (`ceiling::render_whole`, fp32, CPU) with these noise
    /// seeds. Labelled as stand-ins in the report: the same model code as Sigil's stream, so they
    /// are neutral between Sigil's own configurations and ablations, not between engines.
    #[serde(default)]
    pub stand_in_references: Option<StandInRefs>,
    /// Analyzers that must agree (voiced within 50 cents, or unvoiced) for a consensus truth frame;
    /// `None` = every analyzer (the plan's strict intersection). A majority at least.
    #[serde(default)]
    pub consensus_min_agree: Option<usize>,
}

/// The stand-in references of a dry run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct StandInRefs {
    #[serde(default = "stand_in_seeds")]
    pub seeds: Vec<u64>,
}

fn stand_in_seeds() -> Vec<u64> {
    vec![7_001, 7_002]
}

fn corpus_schema() -> String {
    CORPUS_SCHEMA.to_string()
}
fn default_min_clips() -> usize {
    20
}
fn default_analyzers() -> Vec<String> {
    ["rmvpe", "swiftf0", "crepe", "dio"].iter().map(|s| s.to_string()).collect()
}
fn default_loudness_lufs() -> f64 {
    -26.0
}

/// The voice of a corpus.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusVoice {
    #[serde(default)]
    pub name: String,
    /// sha256 of the voice file every system loaded (the `.pth`; Sigil's slot records it as
    /// `source_sha256`).
    #[serde(default)]
    pub sha256: String,
    #[serde(default)]
    pub path: String,
}

/// One input clip.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CorpusClip {
    /// Id (letters, digits, `-`, `_`, `.`): `{clip}` in point paths.
    pub id: String,
    pub input: PathBuf,
    /// sha256 of the input file (checked when given).
    #[serde(default)]
    pub sha256: String,
    /// Tracker-independent F0 truth of the input (`scripts/vc-bench/egg_f0.py` CSV: `time_s,f0_hz`,
    /// 0 = unvoiced, on the 10 ms grid; before the transpose).
    #[serde(default)]
    pub f0_truth: Option<PathBuf>,
    /// Neutral references: upstream whole-utterance renders of the input with the same voice
    /// (`scripts/vc-bench/rvc_offline_reference.py`, one per noise seed).
    #[serde(default)]
    pub references: Vec<PathBuf>,
    #[serde(default)]
    pub speaker: Option<String>,
    #[serde(default)]
    pub sex: Option<String>,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub seconds: Option<f64>,
    #[serde(default)]
    pub source: Option<String>,
    #[serde(default)]
    pub license: Option<String>,
    #[serde(default)]
    pub notes: String,
}

/// Offline renders (a driver's block loop over the clip) or live captures through the devices.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    #[default]
    Offline,
    Live,
}

/// What a point is in the comparison.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointRole {
    /// Another system (counted in T6).
    #[default]
    Competitor,
    /// Sigil (the side T6 is about).
    Sigil,
    /// A Sigil ablation: paired with the Sigil row it ablates, never counted as a competitor.
    Ablation,
}

/// One operating point of another system (or a captured Sigil rendition).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PointSpec {
    /// Row name (letters, digits, `-`, `_`, `.`).
    pub name: String,
    /// `rvc_webui`, `applio`, `w_okada`, `deiteris`, `vc_rs`, `rvc_realtime`, `sigil`, ...
    pub system: String,
    #[serde(default)]
    pub app_version: String,
    #[serde(default)]
    pub tier: Tier,
    #[serde(default)]
    pub role: PointRole,
    /// Its rendition of each clip: a path with `{clip}` (relative to the manifest). A JSON sidecar
    /// next to it (`<wav>.json`) fills metadata the point leaves out.
    pub wav: String,
    /// `block_ms`, `extra_ms`, `crossfade_ms`, `f0_method`, `index_rate`, `protect`, `rms_mix`,
    /// `precision`, `ep`, `gpu`, `transpose`.
    #[serde(default)]
    pub config: Option<serde_json::Value>,
    #[serde(default)]
    pub latency_ms: Option<f64>,
    /// `e2e_loopback`, `declared` or `algorithmic`.
    #[serde(default)]
    pub latency_kind: String,
    #[serde(default)]
    pub voice_sha256: String,
    #[serde(default)]
    pub call_ms: Option<CallMsStats>,
    #[serde(default)]
    pub gpu: String,
    #[serde(default)]
    pub notes: String,
}

/// A live capture: play every clip through the system under test and record its response into the
/// point's WAVs ([`run_play_record`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaptureSpec {
    /// The point (tier `live`) whose WAVs are recorded.
    pub point: String,
    /// Must be `true`: the meter plays the clips on `meter.output_device`.
    #[serde(default)]
    pub record: bool,
    /// Record clips whose WAV exists again (default: only the missing ones).
    #[serde(default)]
    pub overwrite: bool,
    pub meter: PlayRecordSpec,
}

/// Sigil's extra rows.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SigilSection {
    /// In-process rows besides `compare.sigil.configs`: the spec's engine and stream patched.
    #[serde(default)]
    pub points: Vec<SigilPointSpec>,
    /// The voice's sha256 when the slot does not record it (`source_sha256`).
    #[serde(default)]
    pub voice_sha256: String,
    /// The GPU of this run (default: the first hardware adapter).
    #[serde(default)]
    pub gpu: String,
}

/// An in-process Sigil row: the spec's engine and stream with fields replaced.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SigilPointSpec {
    pub name: String,
    /// `sigil` (default) or `ablation`.
    #[serde(default = "sigil_role")]
    pub role: PointRole,
    /// An ablation: the Sigil row it is compared with (a config or point name).
    #[serde(default)]
    pub versus: String,
    #[serde(default)]
    pub label: String,
    /// Fields of the spec's `engine` replaced (e.g. `{"levers": {"anchored_noise": "off"}}`).
    #[serde(default)]
    pub engine: serde_json::Map<String, serde_json::Value>,
    /// Fields of the spec's `stream` replaced (e.g. `{"crossfade_ms": 50}`).
    #[serde(default)]
    pub stream: serde_json::Map<String, serde_json::Value>,
}

fn sigil_role() -> PointRole {
    PointRole::Sigil
}

/// The verdicts' margins, fixed before the run.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Margins {
    /// DNSMOS row: Sigil's mean paired difference must reach this (T6: +0.05).
    #[serde(default = "m_dnsmos_sup")]
    pub dnsmos_superiority: f64,
    /// Non-inferiority margin of DNSMOS (the latency row's quality condition).
    #[serde(default = "m_dnsmos_ni")]
    pub dnsmos_noninferiority: f64,
    /// Non-inferiority margin of the gross pitch error (fraction of voiced frames).
    #[serde(default = "m_gpe_ni")]
    pub gpe_noninferiority: f64,
    /// Non-inferiority margin of the envelope MCD to the neutral reference (dB).
    #[serde(default = "m_mcd_ni")]
    pub mcd_noninferiority_db: f64,
    /// Non-inferiority margin of the target-speaker similarity (advisory row).
    #[serde(default = "m_spk_ni")]
    pub speaker_noninferiority: f64,
    /// Latency row: Sigil's latency over the competitor's (T6: 0.7).
    #[serde(default = "m_latency")]
    pub latency_ratio: f64,
}

fn m_dnsmos_sup() -> f64 {
    0.05
}
fn m_dnsmos_ni() -> f64 {
    0.05
}
fn m_gpe_ni() -> f64 {
    0.005
}
fn m_mcd_ni() -> f64 {
    0.1
}
fn m_spk_ni() -> f64 {
    0.01
}
fn m_latency() -> f64 {
    0.7
}

impl Default for Margins {
    fn default() -> Self {
        serde_json::from_value(serde_json::json!({})).expect("margin defaults")
    }
}

/// The listening-test manifest the run emits.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ListeningSpec {
    /// The Sigil row paired with every competitor (default: the first Sigil row).
    #[serde(default)]
    pub sigil: String,
}

impl CorpusManifest {
    /// Read and check a manifest.
    pub fn load(path: &Path) -> Result<CorpusManifest, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("corpus {}: {e}", path.display()))?;
        let m: CorpusManifest = serde_json::from_str(&text).map_err(|e| format!("corpus {}: {e}", path.display()))?;
        m.validate()?;
        Ok(m)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != CORPUS_SCHEMA {
            return Err(format!("corpus schema '{}' (expected '{CORPUS_SCHEMA}')", self.schema));
        }
        if self.clips.is_empty() {
            return Err("the corpus has no clips".into());
        }
        let id_ok = |id: &str| !id.is_empty() && id.len() <= 96 && !id.starts_with('.') && id.chars().all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.'));
        let mut ids = BTreeSet::new();
        for c in &self.clips {
            if !id_ok(&c.id) {
                return Err(format!("clip id '{}' must be letters, digits, '-', '_' or '.'", c.id));
            }
            if !ids.insert(c.id.as_str()) {
                return Err(format!("duplicate clip '{}'", c.id));
            }
        }
        let mut names: BTreeSet<&str> = ["sigil"].into_iter().collect();
        for n in ["shipping_default", "autotuned", "low_latency", "quality"] {
            names.insert(n);
        }
        for p in &self.points {
            if !id_ok(&p.name) || !names.insert(p.name.as_str()) {
                return Err(format!("point '{}': a duplicate or invalid name (Sigil's configuration names are taken)", p.name));
            }
            if p.wav.trim().is_empty() {
                return Err(format!("point '{}' has no wav", p.name));
            }
            if p.role == PointRole::Ablation {
                return Err(format!("point '{}': ablations are Sigil's own rows (sigil.points)", p.name));
            }
            if self.clips.len() > 1 && !p.wav.contains("{clip}") {
                return Err(format!("point '{}': its wav must contain {{clip}} (one rendition per clip)", p.name));
            }
            check_latency_kind(&p.name, &p.latency_kind)?;
        }
        for p in &self.sigil.points {
            if !id_ok(&p.name) || !names.insert(p.name.as_str()) {
                return Err(format!("sigil.points '{}': a duplicate or invalid name", p.name));
            }
            match p.role {
                PointRole::Competitor => return Err(format!("sigil.points '{}': role must be sigil or ablation", p.name)),
                PointRole::Ablation if p.versus.trim().is_empty() => {
                    return Err(format!("sigil.points '{}': an ablation names the Sigil row it ablates (versus)", p.name))
                }
                _ => {}
            }
        }
        for cap in &self.captures {
            let Some(p) = self.points.iter().find(|p| p.name == cap.point) else {
                return Err(format!("captures: no point '{}'", cap.point));
            };
            if p.tier != Tier::Live {
                return Err(format!("captures: point '{}' is not tier live", cap.point));
            }
            cap.meter.validate().map_err(|e| format!("captures '{}': {e}", cap.point))?;
        }
        let m = &self.margins;
        for (what, v, lo, hi) in [
            ("margins.dnsmos_superiority", m.dnsmos_superiority, 0.0, 1.0),
            ("margins.dnsmos_noninferiority", m.dnsmos_noninferiority, 0.0, 1.0),
            ("margins.gpe_noninferiority", m.gpe_noninferiority, 0.0, 0.2),
            ("margins.mcd_noninferiority_db", m.mcd_noninferiority_db, 0.0, 3.0),
            ("margins.speaker_noninferiority", m.speaker_noninferiority, 0.0, 0.5),
            ("margins.latency_ratio", m.latency_ratio, 0.05, 2.0),
            ("loudness_lufs", self.loudness_lufs, -60.0, -10.0),
        ] {
            if !(v.is_finite() && v >= lo && v <= hi) {
                return Err(format!("{what} = {v} is outside {lo}..={hi}"));
            }
        }
        if !(2..=100_000).contains(&self.min_clips) {
            return Err(format!("min_clips = {} must be 2 or more", self.min_clips));
        }
        Ok(())
    }
}

/// The latency kind of a live capture whose delay is only the offset this run measured from the
/// recording, with nothing calibrated against a bare loop (a `play_record.py` capture): it includes
/// the two independent streams' start-up skew and buffers. It is never compared with a calibrated
/// `e2e_loopback` latency, nor with another uncalibrated one (the rigs differ).
const UNCALIBRATED: &str = "uncalibrated";

fn check_latency_kind(name: &str, kind: &str) -> Result<(), String> {
    match kind.trim() {
        "" | "e2e_loopback" | "declared" | "algorithmic" | UNCALIBRATED => Ok(()),
        other => Err(format!("point '{name}': latency_kind '{other}' must be e2e_loopback, declared, algorithmic or uncalibrated")),
    }
}

/// What a driver's sidecar says about one rendition (read leniently: drivers may add fields).
#[derive(Debug, Clone, Default, Serialize)]
pub struct Sidecar {
    pub system: String,
    pub app_version: String,
    pub input_sha256: String,
    pub voice_sha256: String,
    pub config: serde_json::Map<String, serde_json::Value>,
    pub call_ms: Option<CallMsStats>,
    pub gpu: String,
    pub latency_ms: Option<f64>,
    pub latency_kind: String,
    /// The driver's own real-time check (per-block p99 below the block period).
    pub feasible: Option<bool>,
    pub glitches: Option<f64>,
    pub dropouts: Option<f64>,
}

impl Sidecar {
    /// The sidecar of `wav` (`<wav>.json`), when there is one; an unreadable sidecar is an error
    /// (it would hide metadata the checks need).
    pub fn of(wav: &Path) -> Result<Option<Sidecar>, String> {
        let mut p = wav.as_os_str().to_owned();
        p.push(".json");
        let p = PathBuf::from(p);
        if !p.is_file() {
            return Ok(None);
        }
        let v: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&p).map_err(|e| format!("{}: {e}", p.display()))?).map_err(|e| format!("{}: {e}", p.display()))?;
        let schema = v.get("schema").and_then(|s| s.as_str()).unwrap_or("");
        if schema != SIDECAR_SCHEMA {
            return Err(format!("{}: schema '{schema}' (expected '{SIDECAR_SCHEMA}')", p.display()));
        }
        let s = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
        let f = |k: &str| v.get(k).and_then(|x| x.as_f64()).filter(|x| x.is_finite());
        let call_ms = v.get("call_ms").filter(|c| c.is_object()).map(|c| CallMsStats {
            p50: c.get("p50").and_then(|x| x.as_f64()),
            p95: c.get("p95").and_then(|x| x.as_f64()),
            p99: c.get("p99").and_then(|x| x.as_f64()),
        });
        Ok(Some(Sidecar {
            system: s("system"),
            app_version: s("app_version"),
            input_sha256: s("input_sha256").to_ascii_lowercase(),
            voice_sha256: s("voice_sha256").to_ascii_lowercase(),
            config: v.get("config").and_then(|c| c.as_object()).cloned().unwrap_or_default(),
            call_ms,
            gpu: s("gpu"),
            latency_ms: f("latency_ms"),
            latency_kind: s("latency_kind"),
            feasible: v.get("feasible").and_then(|x| x.as_bool()),
            glitches: f("glitches"),
            dropouts: f("dropouts"),
        }))
    }
}

// ---------------------------------------------------------------------------------------------
// Loudness (ITU-R BS.1770-4)
// ---------------------------------------------------------------------------------------------

/// One biquad `[b0, b1, b2, a1, a2]` (a0 = 1).
type Biquad = [f64; 5];

/// The two K-weighting stages at `rate` (the pre-filter's high shelf, +4 dB above ~1.7 kHz, and
/// the RLB high-pass), designed from their analog prototypes by the bilinear transform as
/// libebur128 does: at 48 kHz they are BS.1770-4's coefficient table, and any other rate keeps the
/// same response (the shelf frequency is pre-warped).
fn k_weighting(rate: u32) -> [Biquad; 2] {
    let fs = f64::from(rate);
    let (f0, g, q) = (1_681.974_450_955_533_f64, 3.999_843_853_973_347_f64, 0.707_175_236_955_419_6_f64);
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let vh = 10f64.powf(g / 20.0);
    let vb = vh.powf(0.499_666_774_154_541_6);
    let a0 = 1.0 + k / q + k * k;
    let shelf = [
        (vh + vb * k / q + k * k) / a0,
        2.0 * (k * k - vh) / a0,
        (vh - vb * k / q + k * k) / a0,
        2.0 * (k * k - 1.0) / a0,
        (1.0 - k / q + k * k) / a0,
    ];
    let (f0, q) = (38.135_470_876_024_44_f64, 0.500_327_037_323_877_3_f64);
    let k = (std::f64::consts::PI * f0 / fs).tan();
    let d = 1.0 + k / q + k * k;
    let rlb = [1.0, -2.0, 1.0, 2.0 * (k * k - 1.0) / d, (1.0 - k / q + k * k) / d];
    [shelf, rlb]
}

/// Integrated loudness (LUFS) of a mono clip: K-weighting, 400 ms blocks every 100 ms, the −70
/// LUFS absolute gate and the −10 LU relative gate. `None` for clips under 400 ms or below the
/// absolute gate throughout.
pub fn integrated_loudness(x: &[f32], rate: u32) -> Option<f64> {
    let block = (0.4 * f64::from(rate)).round() as usize;
    let step = (0.1 * f64::from(rate)).round() as usize;
    if rate == 0 || x.len() < block || step == 0 {
        return None;
    }
    let mut y: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
    for [b0, b1, b2, a1, a2] in k_weighting(rate) {
        let (mut x1, mut x2, mut y1, mut y2) = (0.0, 0.0, 0.0, 0.0);
        for v in y.iter_mut() {
            let x0 = *v;
            let y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2;
            x2 = x1;
            x1 = x0;
            y2 = y1;
            y1 = y0;
            *v = y0;
        }
    }
    // Mean squares of the gating blocks (prefix sums keep it linear).
    let mut prefix = Vec::with_capacity(y.len() + 1);
    prefix.push(0.0f64);
    let mut acc = 0.0;
    for v in &y {
        acc += v * v;
        prefix.push(acc);
    }
    let z: Vec<f64> = (0..=(y.len() - block) / step)
        .map(|k| (prefix[k * step + block] - prefix[k * step]) / block as f64)
        .collect();
    let lufs = |ms: f64| -0.691 + 10.0 * ms.max(1e-30).log10();
    let abs: Vec<f64> = z.iter().copied().filter(|&ms| lufs(ms) > -70.0).collect();
    if abs.is_empty() {
        return None;
    }
    let rel = lufs(abs.iter().sum::<f64>() / abs.len() as f64) - 10.0;
    let kept: Vec<f64> = abs.into_iter().filter(|&ms| lufs(ms) > rel).collect();
    (!kept.is_empty()).then(|| lufs(kept.iter().sum::<f64>() / kept.len() as f64))
}

/// Gain (dB) bringing `x` to `target_lufs`; `None` when its loudness is not measurable.
fn loudness_gain_db(x: &[f32], rate: u32, target_lufs: f64) -> Option<(f64, f64)> {
    integrated_loudness(x, rate).map(|l| (l, target_lufs - l))
}

fn apply_gain(x: &[f32], gain_db: f64) -> Vec<f32> {
    let g = 10f64.powf(gain_db / 20.0) as f32;
    x.iter().map(|&v| v * g).collect()
}

// ---------------------------------------------------------------------------------------------
// Paired statistics over clips
// ---------------------------------------------------------------------------------------------

/// Per-clip paired differences `a − b` of one metric and their intervals.
#[derive(Debug, Clone, Serialize)]
pub struct PairedDiff {
    pub n: usize,
    pub mean: f64,
    /// Sample standard deviation of the differences (n − 1).
    pub sd: f64,
    /// Student t 95 % interval of the mean (the verdicts' interval).
    pub t_lo: f64,
    pub t_hi: f64,
    /// Hodges-Lehmann estimate and the exact Wilcoxon signed-rank interval (6 clips or more).
    pub hl: f64,
    pub w_lo: Option<f64>,
    pub w_hi: Option<f64>,
    /// Percentile bootstrap of the mean over clips (cross-check).
    pub boot_lo: f64,
    pub boot_hi: f64,
    pub clips: Vec<String>,
    pub diffs: Vec<f64>,
}

impl PairedDiff {
    /// The differences of the clips where both sides were measured (`None` below two clips).
    pub fn of(pairs: &[(String, f64, f64)]) -> Option<PairedDiff> {
        let kept: Vec<&(String, f64, f64)> = pairs.iter().filter(|p| p.1.is_finite() && p.2.is_finite()).collect();
        let n = kept.len();
        if n < 2 {
            return None;
        }
        let diffs: Vec<f64> = kept.iter().map(|p| p.1 - p.2).collect();
        let mean = diffs.iter().sum::<f64>() / n as f64;
        let sd = (diffs.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / (n - 1) as f64).sqrt();
        let half = paired::t975(n - 1) * sd / (n as f64).sqrt();
        let w = paired::wilcoxon_interval(&diffs, 0.05);
        let (_, boot_lo, boot_hi) = mean_ci(&diffs).unwrap_or((mean, mean, mean));
        Some(PairedDiff {
            n,
            mean,
            sd,
            t_lo: mean - half,
            t_hi: mean + half,
            hl: paired::hodges_lehmann(&diffs),
            w_lo: w.map(|w| w.0),
            w_hi: w.map(|w| w.1),
            boot_lo,
            boot_hi,
            clips: kept.iter().map(|p| p.0.clone()).collect(),
            diffs,
        })
    }

    /// Whether the decision interval and the bootstrap disagree about the sign of the lower bound
    /// (the verdict then rests on the normality of the t-interval: reported).
    fn fragile(&self) -> bool {
        (self.t_lo > 0.0) != (self.boot_lo > 0.0) || (self.t_hi < 0.0) != (self.boot_hi < 0.0)
    }
}

/// Clips the +`delta` DNSMOS row needs at a paired SD (two-sided 95 %): `(50 % power, 80 %)`.
pub fn clips_needed(sd: f64, delta: f64) -> (usize, usize) {
    const Z975: f64 = 1.959_964;
    const Z80: f64 = 0.841_621;
    if !(sd.is_finite() && sd > 0.0 && delta > 0.0) {
        return (2, 2);
    }
    let n = |z: f64| ((z * sd / delta).powi(2)).ceil().max(2.0) as usize;
    (n(Z975), n(Z975 + Z80))
}

/// A verdict on one plan row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Status {
    Pass,
    /// Non-inferiority not shown (the interval crosses the margin).
    NotShown,
    /// Shown worse (the whole interval beyond the margin), or no Sigil row meets the latency.
    Fail,
    /// Not decidable: too few clips, no row at matched latency, a metric not measured, unknown
    /// feasibility.
    Undecided,
    /// An unfair pairing (transpose, voice, input or latency kind): never a pass.
    Refused,
    /// An offline competitor point that cannot run in real time on this GPU: not counted.
    Excluded,
}

/// A row's verdict and why.
#[derive(Debug, Clone, Serialize)]
pub struct RowVerdict {
    pub status: Status,
    /// The Sigil point that decided it (a pass: the first that passes).
    pub by: Option<String>,
    pub detail: String,
}

impl RowVerdict {
    fn new(status: Status, by: Option<&str>, detail: impl Into<String>) -> Self {
        Self {
            status,
            by: by.map(str::to_string),
            detail: detail.into(),
        }
    }
}

/// Non-inferiority of `d` = Sigil − competitor of a lower-is-better metric at `margin`.
fn noninferior_lower(d: &PairedDiff, margin: f64) -> Status {
    if d.t_hi <= margin {
        Status::Pass
    } else if d.t_lo > margin {
        Status::Fail
    } else {
        Status::NotShown
    }
}

/// Non-inferiority of `d` = Sigil − competitor of a higher-is-better metric at `margin`.
fn noninferior_higher(d: &PairedDiff, margin: f64) -> Status {
    if d.t_lo >= -margin {
        Status::Pass
    } else if d.t_hi < -margin {
        Status::Fail
    } else {
        Status::NotShown
    }
}

// ---------------------------------------------------------------------------------------------
// Points
// ---------------------------------------------------------------------------------------------

/// One operating point as the run scores it (Sigil's in-process rows, the corpus's points, v1's
/// inline candidates).
#[derive(Debug, Clone, Serialize)]
pub struct Point {
    pub name: String,
    pub system: String,
    pub app_version: String,
    pub tier: Tier,
    pub role: PointRole,
    /// `in_process` (converted by this run), `manifest`, `inline` (a v1 candidate).
    pub source: &'static str,
    pub label: String,
    /// An ablation's Sigil row.
    pub versus: Option<String>,
    pub config: serde_json::Map<String, serde_json::Value>,
    /// Its block period (ms): the engine block for Sigil, `config.block_ms` otherwise.
    pub block_ms: Option<f64>,
    pub latency_ms: Option<f64>,
    pub latency_kind: String,
    /// Where `latency_ms` comes from (`measured`, `probe`, `manifest`, `sidecar`).
    pub latency_source: String,
    /// The latency this run measured from the renditions: the median offset of the rendition
    /// against the input over the clips, plus one block for offline renders (whose block `b` sits
    /// at `b × block` in the file, while live it plays after the block was captured).
    pub latency_measured_ms: Option<f64>,
    pub voice_sha256: String,
    pub gpu: String,
    pub call_ms: Option<CallMsStats>,
    pub transpose: Option<f64>,
    /// Runs in real time: per-block p99 below the block period (on this run's GPU for offline
    /// points); `None` = unknown. Live captures ran in real time by construction.
    pub feasible: Option<bool>,
    pub feasibility: String,
    /// Missing metadata and other warnings (the point still counts).
    pub flags: Vec<String>,
    /// Unfair pairings (the point never enters a verdict).
    pub refusals: Vec<String>,
    /// Clips with no complete score for this point (a conversion error, a missing, refused or
    /// muted rendition, an alignment failure, no DNSMOS where the clip needs one). A row that
    /// misses clips is incomplete: it decides nothing (see [`unscored_cells`]).
    pub unscored_clips: Vec<String>,
}

impl Point {
    fn new(name: &str, system: &str, tier: Tier, role: PointRole, source: &'static str) -> Point {
        Point {
            name: name.to_string(),
            system: system.to_string(),
            app_version: String::new(),
            tier,
            role,
            source,
            label: String::new(),
            versus: None,
            config: serde_json::Map::new(),
            block_ms: None,
            latency_ms: None,
            latency_kind: String::new(),
            latency_source: String::new(),
            latency_measured_ms: None,
            voice_sha256: String::new(),
            gpu: String::new(),
            call_ms: None,
            transpose: None,
            feasible: None,
            feasibility: String::new(),
            flags: Vec::new(),
            refusals: Vec::new(),
            unscored_clips: Vec::new(),
        }
    }

    fn is_sigil(&self) -> bool {
        self.role == PointRole::Sigil
    }

    fn counted(&self) -> bool {
        self.role == PointRole::Competitor
    }

    /// Fill what the point leaves out from a sidecar, and flag disagreements.
    fn merge_sidecar(&mut self, sc: &Sidecar) {
        let fill = |field: &mut String, v: &str, what: &str, flags: &mut Vec<String>| {
            if field.is_empty() {
                *field = v.to_string();
            } else if !v.is_empty() && !field.eq_ignore_ascii_case(v) {
                flags.push(format!("{what}: the point says '{field}', its sidecar '{v}'"));
            }
        };
        let flags = &mut self.flags;
        fill(&mut self.system, &sc.system, "system", flags);
        fill(&mut self.app_version, &sc.app_version, "app_version", flags);
        fill(&mut self.voice_sha256, &sc.voice_sha256, "voice_sha256", flags);
        fill(&mut self.gpu, &sc.gpu, "gpu", flags);
        fill(&mut self.latency_kind, &sc.latency_kind, "latency_kind", flags);
        for (k, v) in &sc.config {
            match self.config.get(k) {
                None => {
                    self.config.insert(k.clone(), v.clone());
                }
                Some(have) if have != v => flags.push(format!("config.{k}: the point says {have}, its sidecar {v}")),
                Some(_) => {}
            }
        }
        if self.call_ms.is_none() {
            self.call_ms = sc.call_ms;
        }
        if self.latency_ms.is_none() && sc.latency_ms.is_some() {
            self.latency_ms = sc.latency_ms;
            self.latency_source = "sidecar".into();
        }
        if self.feasible.is_none() {
            if let Some(f) = sc.feasible {
                self.feasibility = format!("the driver's own check: {}", if f { "real time" } else { "not real time" });
            }
        }
    }

    /// Merge the sidecars of ALL the point's renditions (one per clip, `(clip, sidecar)`).
    ///
    /// One point is one operating point: sidecars that disagree on the voice, the transpose or the
    /// block (a clip re-run with `--pitch 12`, a stale file of an earlier sweep in the same folder)
    /// refuse the whole point instead of letting the first clip's sidecar speak for the rest. A
    /// sidecar that contradicts what the point itself states about the voice or the transpose
    /// refuses it too. The per-block compute is the worst p95 / p99 over the clips (the median p50),
    /// as Sigil's own rows aggregate, and the feasibility check reads that.
    fn merge_sidecars(&mut self, sides: &[(String, Sidecar)]) {
        let Some((_, first)) = sides.first() else {
            return;
        };
        let stated_voice = self.voice_sha256.trim().to_ascii_lowercase();
        let stated_transpose = config_num(&self.config, TRANSPOSE_KEYS);
        let mut refusals: Vec<String> = Vec::new();
        let list = |vals: Vec<(&str, String)>| vals.iter().map(|(c, v)| format!("{c}: {v}")).collect::<Vec<_>>().join(", ");
        // Between the clips' sidecars.
        let voices: Vec<(&str, String)> = sides.iter().filter(|(_, s)| !s.voice_sha256.is_empty()).map(|(c, s)| (c.as_str(), short(&s.voice_sha256))).collect();
        if voices.iter().any(|v| v.1 != voices[0].1) {
            refusals.push(format!("the clips' sidecars name different voices ({})", list(voices)));
        }
        let transposes: Vec<(&str, f64)> = sides.iter().filter_map(|(c, s)| config_num(&s.config, TRANSPOSE_KEYS).map(|t| (c.as_str(), t))).collect();
        if transposes.iter().any(|t| (t.1 - transposes[0].1).abs() > 1e-6) {
            refusals.push(format!(
                "the clips' sidecars name different transposes ({})",
                list(transposes.iter().map(|(c, t)| (*c, format!("{t}"))).collect())
            ));
        }
        let blocks: Vec<(&str, f64)> = sides.iter().filter_map(|(c, s)| config_num(&s.config, BLOCK_KEYS).map(|b| (c.as_str(), b))).collect();
        if blocks.iter().any(|b| (b.1 - blocks[0].1).abs() > 0.01) {
            refusals.push(format!(
                "the clips' sidecars name different blocks ({})",
                list(blocks.iter().map(|(c, b)| (*c, format!("{b} ms"))).collect())
            ));
        }
        // Against what the point states itself.
        for (clip, s) in sides {
            if !stated_voice.is_empty() && !s.voice_sha256.is_empty() && s.voice_sha256 != stated_voice {
                refusals.push(format!("clip {clip}'s sidecar names voice {}, the point says {}", short(&s.voice_sha256), short(&stated_voice)));
            }
            if let (Some(have), Some(t)) = (config_num(&s.config, TRANSPOSE_KEYS), stated_transpose) {
                if (have - t).abs() > 1e-6 {
                    refusals.push(format!("clip {clip}'s sidecar says transpose {have}, the point says {t}"));
                }
            }
        }
        for why in refusals {
            if !self.refusals.contains(&why) {
                self.refusals.push(why);
            }
        }
        let mut differing = |what: &str, vals: Vec<&str>| {
            let mut seen: Vec<&str> = Vec::new();
            for v in vals.into_iter().filter(|v| !v.is_empty()) {
                if !seen.iter().any(|x| x.eq_ignore_ascii_case(v)) {
                    seen.push(v);
                }
            }
            if seen.len() > 1 {
                self.flags.push(format!("the clips' sidecars name different {what} ({})", seen.join(", ")));
            }
        };
        differing("systems", sides.iter().map(|(_, s)| s.system.as_str()).collect());
        differing("app versions", sides.iter().map(|(_, s)| s.app_version.as_str()).collect());
        differing("GPUs", sides.iter().map(|(_, s)| s.gpu.as_str()).collect());
        differing("latency kinds", sides.iter().map(|(_, s)| s.latency_kind.as_str()).collect());
        // One merged sidecar: the first non-empty value of each field, the worst compute.
        let pick = |f: &dyn Fn(&Sidecar) -> &String| sides.iter().map(|(_, s)| f(s)).find(|v| !v.is_empty()).cloned().unwrap_or_default();
        let mut latencies: Vec<f64> = sides.iter().filter_map(|(_, s)| s.latency_ms).collect();
        latencies.sort_by(f64::total_cmp);
        let merged = Sidecar {
            system: pick(&|s| &s.system),
            app_version: pick(&|s| &s.app_version),
            input_sha256: first.input_sha256.clone(),
            voice_sha256: pick(&|s| &s.voice_sha256),
            config: first.config.clone(),
            call_ms: aggregate_call_ms(sides.iter().filter_map(|(_, s)| s.call_ms.as_ref())),
            gpu: pick(&|s| &s.gpu),
            latency_ms: latencies.get(latencies.len() / 2).copied(),
            latency_kind: pick(&|s| &s.latency_kind),
            feasible: sides.iter().map(|(_, s)| s.feasible).try_fold(true, |all, f| f.map(|f| all && f)),
            glitches: None,
            dropouts: None,
        };
        self.merge_sidecar(&merged);
    }

    /// `block_ms` and `transpose` from the config (the drivers' keys and RVC's aliases).
    fn read_config(&mut self) {
        if self.block_ms.is_none() {
            self.block_ms = config_num(&self.config, BLOCK_KEYS).filter(|v| *v > 0.0);
        }
        if self.transpose.is_none() {
            self.transpose = config_num(&self.config, TRANSPOSE_KEYS);
        }
        if self.gpu.is_empty() {
            if let Some(g) = self.config.get("gpu").and_then(|v| v.as_str()) {
                self.gpu = g.to_string();
            }
        }
    }
}

/// The config keys that carry the transpose (the drivers' and RVC's names) and the block.
const TRANSPOSE_KEYS: &[&str] = &["transpose", "pitch", "f0_up_key", "semitones", "key"];
const BLOCK_KEYS: &[&str] = &["block_ms", "chunk_ms"];

/// The first of `keys` a config holds as a number.
fn config_num(config: &serde_json::Map<String, serde_json::Value>, keys: &[&str]) -> Option<f64> {
    keys.iter().find_map(|k| config.get(*k).and_then(|v| v.as_f64())).filter(|v| v.is_finite())
}

/// Per-block compute over several renditions: the median p50, the worst p95 and p99 (the same rule
/// as Sigil's own rows, so a competitor is not judged on one clip's warm-up or luck).
fn aggregate_call_ms<'a>(stats: impl Iterator<Item = &'a CallMsStats>) -> Option<CallMsStats> {
    let all: Vec<&CallMsStats> = stats.collect();
    if all.is_empty() {
        return None;
    }
    let mut p50: Vec<f64> = all.iter().filter_map(|c| c.p50).collect();
    p50.sort_by(f64::total_cmp);
    let worst = |f: &dyn Fn(&CallMsStats) -> Option<f64>| all.iter().filter_map(|c| f(c)).fold(None, |a: Option<f64>, x| Some(a.map_or(x, |a| a.max(x))));
    Some(CallMsStats {
        p50: p50.get(p50.len() / 2).copied(),
        p95: worst(&|c| c.p95),
        p99: worst(&|c| c.p99),
    })
}

/// The run's GPU name before any Sigil row has run: the manifest's, else the first hardware
/// adapter (with a note when there are several, since the first is not necessarily the one the
/// engine uses). A Sigil row that runs on a GPU then replaces it with the adapter it really used
/// ([`engine_gpu`]); the manifest's name always wins.
fn run_gpu(manifest_gpu: &str, adapters: &[GpuAdapter]) -> (String, Option<String>) {
    if !manifest_gpu.trim().is_empty() {
        return (manifest_gpu.trim().to_string(), None);
    }
    let hardware: Vec<&GpuAdapter> = adapters.iter().filter(|a| !a.software).collect();
    let note = (hardware.len() > 1).then(|| {
        format!(
            "{} GPUs ({}): the run's GPU is the one the first GPU Sigil row runs on, else '{}'; set compare.sigil.gpu to pin it",
            hardware.len(),
            hardware.iter().map(|a| a.name.trim()).collect::<Vec<_>>().join(", "),
            hardware[0].name.trim()
        )
    });
    (hardware.first().map(|a| a.name.trim().to_string()).unwrap_or_default(), note)
}

/// The adapter an engine's sessions run on, by the rules the sessions are built with: CUDA (and
/// TensorRT) at `gpu_device_index` among the NVIDIA adapters; DirectML at that DXGI index, or, when
/// it is a fallback from a CUDA request, the same NVIDIA card ([`device_for`]). `None` for a CPU
/// engine or an unknown adapter.
///
/// [`device_for`]: crate::services::neural_audio::ort_session::device_for
fn engine_gpu(info: &EngineInfo, gpu_device_index: i32, adapters: &[GpuAdapter]) -> Option<String> {
    let wanted = gpu_device_index.max(0) as usize;
    let nvidia = nvidia_adapter_indices(adapters);
    let nvidia_card = || nvidia.get(wanted).or(nvidia.first()).copied();
    let index = match info.execution_provider_active.as_deref()? {
        "cuda" | "tensorrt_rtx" | "tensorrt" => nvidia_card()?,
        "dml" | "dml_hp" => {
            let requested = info.execution_provider_requested.as_deref().or(info.execution_provider.as_deref()).unwrap_or("");
            if matches!(requested, "cuda" | "gpu" | "tensorrt_rtx") {
                nvidia_card()?
            } else {
                wanted as u32
            }
        }
        _ => return None,
    };
    adapters.iter().find(|a| a.index == index && !a.software).map(|a| a.name.trim().to_string())
}

/// The fairness checks of every point (see the module docs): transpose and voice against the
/// corpus's, latency kinds, and the real-time feasibility of offline competitor points.
fn check_points(points: &mut [Point], transpose: f64, voice_sha: &str, gpu: &str) {
    for p in points.iter_mut() {
        p.read_config();
    }
    // The settings besides the engine that change a voice's sound: compared before the loop below
    // so every competitor meets the Sigil rows as converted.
    let sigil: Vec<Point> = points.iter().filter(|p| p.is_sigil() && p.refusals.is_empty()).cloned().collect();
    for p in points.iter_mut().filter(|p| p.counted() && p.source == "manifest") {
        let flags = operating_point_flags(p, &sigil);
        p.flags.extend(flags);
    }
    for p in points.iter_mut() {
        match p.transpose {
            Some(t) if (t - transpose).abs() > 1e-6 => {
                p.refusals.push(format!("transpose {t} semitones, the run's is {transpose}"))
            }
            Some(_) => {}
            None => p.flags.push("transpose not stated (assumed the run's)".into()),
        }
        let vs = p.voice_sha256.trim().to_ascii_lowercase();
        if voice_sha.is_empty() {
            p.flags.push("the corpus names no voice sha256: the voice is not verified".into());
        } else if vs.is_empty() {
            p.flags.push("voice not verified (no voice_sha256)".into());
        } else if vs != voice_sha {
            p.refusals.push(format!("voice {} differs from the corpus's {}", short(&vs), short(voice_sha)));
        }
        if p.latency_ms.is_some() && p.latency_kind.trim().is_empty() {
            p.flags.push("latency kind unknown: excluded from latency comparisons".into());
        }
        if let Err(e) = check_latency_kind(&p.name, &p.latency_kind) {
            p.refusals.push(e);
        }
        // Real-time feasibility of offline renders (live captures ran in real time).
        if p.tier == Tier::Live {
            p.feasible = Some(true);
            p.feasibility = "a live capture".into();
            continue;
        }
        let cpu = p.config.get("ep").and_then(|v| v.as_str()).is_some_and(|e| e.eq_ignore_ascii_case("cpu"));
        // Sigil's own rows ran on this PC, whichever adapter they used.
        let same_gpu = cpu || p.source == "in_process" || (!p.gpu.is_empty() && !gpu.is_empty() && p.gpu.trim().eq_ignore_ascii_case(gpu.trim()));
        match (p.call_ms.and_then(|c| c.p99), p.block_ms) {
            (Some(p99), Some(block)) if same_gpu => {
                p.feasible = Some(p99 < block);
                p.feasibility = format!("per-block p99 {p99:.2} ms against a {block:.1} ms block{}", if cpu { " (CPU)" } else { "" });
            }
            (Some(_), Some(_)) => {
                p.feasibility = format!("measured on '{}', not this run's GPU '{gpu}': feasibility unknown", p.gpu);
            }
            _ if p.feasibility.is_empty() => {
                p.feasibility = "no per-block compute or block size: feasibility unknown".into();
            }
            _ => {}
        }
    }
}

/// Settings that change how a converted voice sounds beyond the engine: a competitor run at
/// another retrieval strength (an index file at 0.75 against none), protect or loudness mix is
/// measuring those settings, not the engine. Differences are flagged (the quality rows then carry
/// the caveat in the T6 decision), never silently paired.
const OPERATING_KEYS: &[&str] = &["index_rate", "protect", "rms_mix"];
/// Two operating-point numbers closer than this are the same setting.
const OPERATING_TOLERANCE: f64 = 0.05;

/// `operating point: ...` flags of competitor `p` against the Sigil rows' settings, side by side.
fn operating_point_flags(p: &Point, sigil: &[Point]) -> Vec<String> {
    let mut out = Vec::new();
    let mut unstated = Vec::new();
    for key in OPERATING_KEYS {
        let ours: Vec<f64> = sigil.iter().filter_map(|s| config_num(&s.config, &[key])).collect();
        if ours.is_empty() {
            continue;
        }
        let mut distinct: Vec<f64> = Vec::new();
        for v in ours {
            if !distinct.iter().any(|d| (d - v).abs() <= 1e-9) {
                distinct.push(v);
            }
        }
        let list = distinct.iter().map(|v| format!("{v}")).collect::<Vec<_>>().join(", ");
        match config_num(&p.config, &[key]) {
            Some(v) if !distinct.iter().any(|d| (d - v).abs() <= OPERATING_TOLERANCE) => {
                out.push(format!("operating point: {key} {v} here, the Sigil rows use {list}"));
            }
            Some(_) => {}
            None => unstated.push(*key),
        }
    }
    if !unstated.is_empty() {
        out.push(format!("operating point: {} not stated for this point (the Sigil rows' settings cannot be checked against it)", unstated.join(", ")));
    }
    let methods: Vec<String> = sigil
        .iter()
        .filter_map(|s| s.config.get("f0_method").and_then(|v| v.as_str()))
        .map(|m| m.trim().to_ascii_lowercase())
        .filter(|m| !m.is_empty())
        .collect();
    if let Some(m) = p.config.get("f0_method").and_then(|v| v.as_str()).map(|m| m.trim().to_ascii_lowercase()).filter(|m| !m.is_empty()) {
        if !methods.is_empty() && !methods.contains(&m) {
            let mut have = methods.clone();
            have.sort();
            have.dedup();
            out.push(format!("operating point: F0 method {m} here, the Sigil rows use {}", have.join(", ")));
        }
    }
    out
}

fn short(sha: &str) -> String {
    sha.chars().take(12).collect()
}

/// The sha256 of the voice a Sigil slot was imported from (`manifest.json` `source_sha256`).
fn slot_voice_sha(spec: &EngineSpec) -> Option<String> {
    let EngineSpec::Rvc(r) = spec else {
        return None;
    };
    let text = std::fs::read_to_string(r.slot_dir.join("manifest.json")).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    v.get("source_sha256").and_then(|s| s.as_str()).map(|s| s.trim().to_ascii_lowercase()).filter(|s| !s.is_empty())
}

// ---------------------------------------------------------------------------------------------
// Sigil's in-process rows
// ---------------------------------------------------------------------------------------------

/// A Sigil row to convert.
struct SigilBuild {
    point: Point,
    /// `None` = the spec's prepared engine.
    engine: Option<EngineSpec>,
    stream: StreamSpec,
}

/// Replace fields of a serialisable value (the spec's own serde, so a typo is an error).
fn patched<T: Serialize + serde::de::DeserializeOwned>(base: &T, patch: &serde_json::Map<String, serde_json::Value>, what: &str) -> Result<T, String> {
    let mut v = serde_json::to_value(base).map_err(|e| format!("{what}: {e}"))?;
    let obj = v.as_object_mut().ok_or_else(|| format!("{what} is not an object"))?;
    for (k, x) in patch {
        obj.insert(k.clone(), x.clone());
    }
    serde_json::from_value(v).map_err(|e| format!("{what}: {e}"))
}

/// The stream every named configuration starts from: the spec's device formats with the voice
/// changer's default settings and the run's transpose.
fn app_stream(spec: &StreamSpec, transpose: f32) -> StreamSpec {
    StreamSpec {
        input_rate: spec.input_rate,
        output_rate: spec.output_rate,
        input_channels: spec.input_channels,
        output_channels: spec.output_channels,
        pitch_semitones: transpose,
        speaker_id: spec.speaker_id,
        ..StreamSpec::default()
    }
}

/// A named Sigil configuration (see the module docs).
fn named_config(name: &str, spec: &BenchSpec, transpose: f32, app_data: Option<&Path>) -> Result<SigilBuild, String> {
    let EngineSpec::Rvc(r) = &spec.engine else {
        return Err(format!("sigil.configs '{name}' needs an rvc engine (the voice to convert)"));
    };
    let mut e = r.clone();
    // The app's settings, not the benchmark knobs of the spec.
    e.levers.clear();
    e.lookahead_ms = 0.0;
    e.precision = "auto".into();
    e.build = "auto".into();
    e.index = String::new();
    e.denoiser = "off".into();
    e.noise_seed = None;
    e.cpu_arena = None;
    e.dynamic_block_base = None;
    e.content_variant = "auto".into();
    e.qos = "app".into();
    e.intra_threads = None;
    e.silence_skip = true;
    let mut s = app_stream(&spec.stream, transpose);
    let gpu_ep = if r.execution_provider.trim().is_empty() { String::new() } else { r.execution_provider.clone() };
    let label;
    match name {
        "shipping_default" => {
            e.preset = "streamer".into();
            e.execution_provider = "cpu".into();
            e.pitch_extractor = "fcpe".into();
            e.block_ms = 0.0;
            e.vram_cap_mb = None;
            s.extra_context_ms = 64.0;
            s.index_rate = 0.0;
            label = "what a new install runs: Streamer preset, CPU, FCPE, the app's default stream (512-frame device block, 64 ms extra, 20 ms crossfade)".to_string();
        }
        "low_latency" => {
            e.preset = "low_latency".into();
            e.execution_provider = gpu_ep;
            e.pitch_extractor = "rmvpe".into();
            e.vram_cap_mb = Some(768);
            s.extra_context_ms = 96.0;
            s.index_rate = 0.0;
            label = format!("the Low latency preset (RMVPE, 96 ms extra) on the spec's EP, engine block {} ms", fmt_block(e.block_ms));
        }
        "quality" => {
            e.preset = "quality".into();
            e.execution_provider = gpu_ep;
            e.pitch_extractor = "rmvpe".into();
            e.lookahead_ms = 20.0;
            e.vram_cap_mb = Some(1536);
            s.extra_context_ms = 192.0;
            s.index_rate = 0.5;
            label = format!("the Quality preset (RMVPE, 192 ms extra, 20 ms lookahead, index 0.5) on the spec's EP, engine block {} ms", fmt_block(e.block_ms));
        }
        "autotuned" => {
            let dir = app_data.ok_or("autotuned: no app data folder (SIGIL_APP_DATA_DIR)")?;
            let t = super::autotune::stored_result(dir).ok_or_else(|| {
                format!(
                    "autotuned: no stored autotune result for this GPU and driver in {} (run the autotuner on this PC first)",
                    dir.join("voice-changer").join(super::autotune::RESULTS_FILE).display()
                )
            })?;
            if let Some(c) = &t.conditions {
                e.preset = c.preset.clone();
                e.lookahead_ms = c.lookahead_ms;
                e.f0_range = c.f0_range.clone();
                e.levers = c.levers.clone();
                e.vram_cap_mb = (c.vram_cap_mb > 0).then_some(c.vram_cap_mb);
                s.block_frames = c.chunk_frames;
                s.crossfade_ms = c.crossfade_ms;
                s.output_prefill_ms = c.output_prefill_ms;
                s.output_max_queue_ms = c.output_max_queue_ms;
                s.latency_budget_ms = c.latency_budget_ms;
                s.index_rate = c.index_rate;
                s.protect = c.protect;
                s.rms_mix = c.rms_mix;
                s.worker_priority = c.worker_priority.clone();
            }
            for (k, v) in &t.params {
                let text = v.as_str().map(str::to_string);
                match k.as_str() {
                    "rvc_preset" => e.preset = text.unwrap_or_default(),
                    "worker_priority" => s.worker_priority = text.unwrap_or_default(),
                    "rvc_execution_provider" => e.execution_provider = text.unwrap_or_default(),
                    "pitch_extractor" => e.pitch_extractor = text.unwrap_or_default(),
                    "precision" => e.precision = text.unwrap_or_default(),
                    "extra_context_ms" => s.extra_context_ms = v.as_f64().unwrap_or(s.extra_context_ms),
                    "block_ms" => e.block_ms = v.as_f64().unwrap_or(e.block_ms),
                    "chunk_frames" => s.block_frames = v.as_u64().map_or(s.block_frames, |n| n as u32),
                    lever if crate::services::neural_audio::vc::engine::options::Lever::from_name(lever).is_some() => {
                        if let Some(m) = text {
                            e.levers.insert(lever.to_string(), m);
                        }
                    }
                    _ => {}
                }
            }
            label = format!(
                "the autotuner's stored result ({} {}, {})",
                t.gpu.as_deref().unwrap_or("?"),
                t.driver.as_deref().unwrap_or("?"),
                t.sigil_version
            );
        }
        other => return Err(format!("sigil.configs: unknown configuration '{other}'")),
    }
    let mut point = Point::new(name, "sigil", Tier::Offline, PointRole::Sigil, "in_process");
    point.label = label;
    Ok(SigilBuild {
        point,
        engine: Some(EngineSpec::Rvc(e)),
        stream: s,
    })
}

fn fmt_block(ms: f64) -> String {
    if ms > 0.0 {
        format!("{ms}")
    } else {
        "from the device block".into()
    }
}

/// Every Sigil row of the run: the configurations (or the spec's engine as `sigil`) and the
/// manifest's points (ablations included). A configuration that cannot be built is noted.
fn sigil_builds(c: &CompareRun, spec: &BenchSpec, manifest: Option<&CorpusManifest>, transpose: f32, notes: &mut Vec<String>) -> Vec<SigilBuild> {
    let mut out = Vec::new();
    if c.sigil.include {
        if c.sigil.configs.is_empty() {
            let mut point = Point::new("sigil", "sigil", Tier::Offline, PointRole::Sigil, "in_process");
            point.label = "the spec's engine and stream".into();
            out.push(SigilBuild {
                point,
                engine: None,
                stream: spec.stream.clone(),
            });
        } else {
            let app_data = crate::services::diagnostics::app_data_dir();
            for name in &c.sigil.configs {
                match named_config(name.trim(), spec, transpose, app_data.as_deref()) {
                    Ok(b) => out.push(b),
                    Err(e) => notes.push(e),
                }
            }
        }
    }
    for p in manifest.map(|m| m.sigil.points.as_slice()).unwrap_or_default() {
        let built = patched(&spec.engine, &p.engine, &format!("sigil.points '{}': engine", p.name))
            .and_then(|e| patched(&spec.stream, &p.stream, &format!("sigil.points '{}': stream", p.name)).map(|s| (e, s)))
            .and_then(|(e, s)| s.validate().map(|_| (e, s)).map_err(|err| format!("sigil.points '{}': {err}", p.name)));
        match built {
            Ok((engine, mut stream)) => {
                if !p.stream.contains_key("pitch_semitones") {
                    stream.pitch_semitones = transpose;
                }
                let mut point = Point::new(&p.name, "sigil", Tier::Offline, p.role, "in_process");
                point.label = if p.label.is_empty() { format!("{} (the spec's engine and stream, patched)", p.name) } else { p.label.clone() };
                point.versus = (!p.versus.trim().is_empty()).then(|| p.versus.trim().to_string());
                point.config.insert("engine_patch".into(), serde_json::Value::Object(p.engine.clone()));
                point.config.insert("stream_patch".into(), serde_json::Value::Object(p.stream.clone()));
                out.push(SigilBuild {
                    point,
                    engine: Some(engine),
                    stream,
                });
            }
            Err(e) => notes.push(e),
        }
    }
    out
}

/// A Sigil row's metadata from its engine, stream and latency probe.
fn describe_sigil(point: &mut Point, engine: &Engine, stream: &StreamSpec, probe: &AlgorithmicLatency, voice_sha: &str, gpu: &str) {
    let info = &engine.info;
    let block_ms = engine
        .rvc
        .as_ref()
        .map(|p| p.block_ms)
        .filter(|b| *b > 0.0)
        .or_else(|| engine.fixed_block().map(|b| b as f64 / 16.0))
        .unwrap_or_else(|| stream.block_ms());
    let mut put = |k: &str, v: serde_json::Value| {
        point.config.insert(k.to_string(), v);
    };
    put("block_ms", block_ms.into());
    put("device_block_ms", stream.block_ms().into());
    put("extra_ms", stream.extra_context_ms.into());
    put("crossfade_ms", stream.crossfade_ms.into());
    put("f0_method", info.pitch_extractor.clone().unwrap_or_default().into());
    put("index_rate", f64::from(stream.index_rate).into());
    put("protect", f64::from(stream.protect).into());
    put("rms_mix", f64::from(stream.rms_mix).into());
    put("precision", info.flowdec_precision.clone().unwrap_or_else(|| "n/a".into()).into());
    put("ep", info.execution_provider_active.clone().or_else(|| info.execution_provider.clone()).unwrap_or_else(|| "cpu".into()).into());
    put("lookahead_ms", info.model_delay_ms.into());
    put("transpose", f64::from(stream.pitch_semitones).into());
    put("build", info.build.clone().unwrap_or_default().into());
    if let Some(plan) = engine.rvc.as_ref() {
        put("preset", plan.spec.preset.clone().into());
    }
    point.block_ms = Some(block_ms);
    // The harness's own delay (the identity probe through the same harness) plus one device block of
    // capture: the offline tier's latency, comparable with the drivers' block + SOLA delay.
    point.latency_ms = Some(probe.ms + stream.block_ms());
    point.latency_kind = "algorithmic".into();
    point.latency_source = "probe".into();
    point.voice_sha256 = voice_sha.to_string();
    point.gpu = gpu.to_string();
    point.app_version = env!("CARGO_PKG_VERSION").to_string();
    point.transpose = Some(f64::from(stream.pitch_semitones));
}

// ---------------------------------------------------------------------------------------------
// Clips, analyzers and references
// ---------------------------------------------------------------------------------------------

/// The F0 analyzers of a run (the consensus truth and the output side).
struct Analyzers {
    list: Vec<F0Analyzer>,
    notes: Vec<String>,
}

impl Analyzers {
    /// Each named analyzer that loads as itself (one that falls back to another method is dropped:
    /// a consensus of DIO with DIO is no consensus), without duplicates.
    fn load(names: &[String], ep: &str, range: F0Range) -> Analyzers {
        let mut list: Vec<F0Analyzer> = Vec::new();
        let mut notes = Vec::new();
        for n in names {
            let want = n.trim().to_ascii_lowercase();
            if want.is_empty() || list.iter().any(|a| a.method == want) {
                continue;
            }
            let a = F0Analyzer::new(&want, None, ep, range);
            if a.method == want {
                list.push(a);
            } else {
                notes.push(format!("F0 analyzer {want} is not available ({}); left out", a.notes.join("; ")));
            }
        }
        Analyzers { list, notes }
    }

    fn names(&self) -> Vec<&'static str> {
        self.list.iter().map(|a| a.method).collect()
    }
}

fn median_f32(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// The consensus of one frame's F0 values (Hz, 0 = unvoiced) with at least `need` analyzers
/// agreeing (a majority at least; `values.len()` = every analyzer, the plan's strict rule):
/// `need` or more voiced within `cents` of their median: `Some(that median)`; `need` or more
/// unvoiced: `Some(0)`; otherwise `None` (no consensus).
fn consensus_frame(values: &[f32], cents: f64, need: usize) -> Option<f32> {
    let n = values.len();
    if n == 0 {
        return None;
    }
    let need = need.clamp(n / 2 + 1, n);
    let mut voiced: Vec<f32> = values.iter().copied().filter(|v| v.is_finite() && *v > 0.0).collect();
    if n - voiced.len() >= need {
        return Some(0.0);
    }
    if voiced.len() < need {
        return None;
    }
    let m = median_f32(&mut voiced);
    let mut agree: Vec<f32> = voiced.into_iter().filter(|&v| (1200.0 * (f64::from(v) / f64::from(m)).log2()).abs() <= cents).collect();
    (agree.len() >= need).then(|| median_f32(&mut agree))
}

/// Consensus truth of several contours of one clip (NaN = no consensus), `need` agreeing.
fn consensus(contours: &[&[f32]], cents: f64, need: usize) -> Vec<f32> {
    let n = contours.iter().map(|c| c.len()).min().unwrap_or(0);
    (0..n)
        .map(|i| {
            let v: Vec<f32> = contours.iter().map(|c| c[i]).collect();
            consensus_frame(&v, cents, need).unwrap_or(f32::NAN)
        })
        .collect()
}

/// Agreement of several contours of one rendition: the share of frames voiced by any of them
/// where all are voiced and within 50 cents of their median.
fn output_agreement(contours: &[Vec<f32>], mask: &[bool]) -> Option<f64> {
    let n = contours.iter().map(Vec::len).min()?.min(mask.len());
    let (mut any, mut agree) = (0usize, 0usize);
    for i in (0..n).filter(|&i| mask[i]) {
        let v: Vec<f32> = contours.iter().map(|c| c[i]).collect();
        if v.iter().any(|x| x.is_finite() && *x > 0.0) {
            any += 1;
            if consensus_frame(&v, 50.0, v.len()).is_some_and(|m| m > 0.0) {
                agree += 1;
            }
        }
    }
    (any > 0).then(|| agree as f64 / any as f64)
}

/// EGG-derived truth (`scripts/vc-bench/egg_f0.py`): `time_s,f0_hz` lines (a header, `#`
/// comments and blank lines skipped), 0 = unvoiced; mapped onto the 10 ms grid (frame `k` at
/// `k × 10 ms`, the nearest truth sample within 5 ms; NaN where none).
fn read_f0_truth(path: &Path, frames: usize) -> Result<Vec<f32>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("F0 truth {}: {e}", path.display()))?;
    let mut pts: Vec<(f64, f64)> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split([',', '\t', ' ']).filter(|s| !s.is_empty());
        let (Some(t), Some(f)) = (it.next(), it.next()) else {
            return Err(format!("F0 truth {} line {}: expected time_s,f0_hz", path.display(), i + 1));
        };
        match (t.parse::<f64>(), f.parse::<f64>()) {
            (Ok(t), Ok(f)) if t.is_finite() => pts.push((t, f)),
            _ if pts.is_empty() => continue, // a header
            _ => return Err(format!("F0 truth {} line {}: '{line}' is not two numbers", path.display(), i + 1)),
        }
    }
    if pts.is_empty() {
        return Err(format!("F0 truth {} has no frames", path.display()));
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut out = vec![f32::NAN; frames];
    let mut j = 0usize;
    for (k, o) in out.iter_mut().enumerate() {
        let t = k as f64 * 0.01;
        while j + 1 < pts.len() && (pts[j + 1].0 - t).abs() <= (pts[j].0 - t).abs() {
            j += 1;
        }
        if (pts[j].0 - t).abs() <= 0.005 + 1e-9 {
            let f = pts[j].1;
            *o = if f.is_finite() && f > 0.0 { f as f32 } else if f.is_finite() { 0.0 } else { f32::NAN };
        }
    }
    Ok(out)
}

/// A neutral reference of a clip, cut to the clip's scored region.
struct RefSeg {
    /// The reference over the input's region `[a, b)` (16 kHz).
    audio: Vec<f32>,
    /// F0 of `audio` (the first analyzer; the envelope MCD's smoothing).
    f0: Vec<f32>,
    /// A stand-in (Sigil's own whole-utterance render): neutral between Sigil's rows and their
    /// ablations, NOT between engines. It is scored apart (`RowScores::mcd_stand_in_db`) and
    /// never feeds the T6 MCD row.
    stand_in: bool,
}

/// Where the clip is and what it is scored against.
struct ClipCtx {
    id: String,
    /// The input file; `None` = the spec's synthetic input.
    path: Option<PathBuf>,
    sha256: String,
    input16: Vec<f32>,
    speech: Vec<bool>,
    region: (usize, usize),
    /// Impulses of the input (s).
    events: Vec<f64>,
    /// The input's F0 by analyzer (times the transpose).
    contours: Vec<Vec<f32>>,
    /// `(egg | consensus, Hz per frame times the transpose; 0 unvoiced, NaN unknown)`.
    truth: Option<(&'static str, Vec<f32>)>,
    /// Share of the speech frames the truth decides.
    truth_coverage: Option<f64>,
    /// Every reference that aligned, upstream and stand-in (see [`RefSeg::stand_in`]).
    refs: Vec<RefSeg>,
    /// Upstream reference files the manifest names / stand-in renders requested.
    ref_files: usize,
    stand_in_files: usize,
    /// Envelope / STFT MCD between the first two references of one kind (the stochastic floor):
    /// the upstream ones, else the stand-ins (`ref_floor_stand_in`).
    ref_floor: Option<(f64, f64)>,
    ref_floor_stand_in: bool,
    /// Embeddings of the upstream references and of the stand-ins (the other clips' target set).
    ref_embeddings: Vec<Vec<f32>>,
    stand_in_embeddings: Vec<Vec<f32>>,
    target: Option<Vec<f32>>,
    /// Where `target` comes from: `upstream`, `stand_in` or `speaker_reference`.
    target_source: &'static str,
    notes: Vec<String>,
}

impl ClipCtx {
    fn dnsmos_ok(&self) -> bool {
        self.input16.len() >= DNSMOS_LEN
    }
}

/// How the clips are scored.
struct ScoreCfg {
    loudness_lufs: f64,
    max_offset_16k: usize,
    /// Analyzers that must agree for a consensus frame (`None` = all).
    consensus_min_agree: Option<usize>,
}

/// A reference before alignment: its name, whether it is a stand-in, and its 16 kHz audio.
struct RefIn {
    name: String,
    stand_in: bool,
    audio: Result<Vec<f32>, String>,
}

/// Analyse one clip: speech region, events, F0 per analyzer, truth, references.
#[allow(clippy::too_many_arguments)]
fn clip_ctx(
    id: &str,
    path: Option<PathBuf>,
    input16: Vec<f32>,
    truth_file: Option<&Path>,
    references: Vec<RefIn>,
    shift: f32,
    an: &mut Analyzers,
    sc: &Scorers,
    spk: Option<&mut SpeakerEmbedder>,
    cfg: &ScoreCfg,
) -> Result<ClipCtx, String> {
    let mut notes = Vec::new();
    let speech = speech_frames(&input16, 40.0);
    let region = speech_region(&speech, input16.len()).ok_or_else(|| format!("clip '{id}' is silent"))?;
    let events = click_events(&input16, 16_000, &sc.click).into_iter().map(|e| e as f64 / 16_000.0).collect();
    let mut contours = Vec::new();
    for a in &mut an.list {
        match a.contour(&input16) {
            Ok(f) => contours.push(f.into_iter().map(|v| v * shift).collect::<Vec<f32>>()),
            Err(e) => {
                notes.push(format!("{}: {e}", a.method));
                contours.push(Vec::new());
            }
        }
    }
    let frames = input16.len() / 160;
    let truth = match truth_file {
        Some(p) => Some(("egg", read_f0_truth(p, frames)?.into_iter().map(|v| if v > 0.0 { v * shift } else { v }).collect::<Vec<f32>>())),
        None => {
            let ok: Vec<&[f32]> = contours.iter().filter(|c| !c.is_empty()).map(|c| c.as_slice()).collect();
            if ok.len() >= 3 {
                Some(("consensus", consensus(&ok, 50.0, cfg.consensus_min_agree.unwrap_or(ok.len()))))
            } else {
                notes.push(format!(
                    "no F0 truth: the clip has no EGG truth and {} analyzer(s) loaded (a consensus needs 3)",
                    ok.len()
                ));
                None
            }
        }
    };
    let truth_coverage = truth.as_ref().map(|(_, t)| {
        let sp: Vec<usize> = (0..speech.len().min(t.len())).filter(|&i| speech[i]).collect();
        sp.iter().filter(|&&i| !t[i].is_nan()).count() as f64 / sp.len().max(1) as f64
    });
    // References: aligned to the input, cut to the region.
    let (a, b) = region;
    let mut refs = Vec::new();
    let mut ref_embeddings = Vec::new();
    let mut stand_in_embeddings = Vec::new();
    let mut spk = spk;
    let ref_files = references.iter().filter(|r| !r.stand_in).count();
    let stand_in_files = references.len() - ref_files;
    for RefIn { name, stand_in, audio } in references {
        let x = match audio {
            Ok(x) => x,
            Err(e) => {
                notes.push(format!("reference {name}: {e}"));
                continue;
            }
        };
        let Some(al) = align(&input16, &x, 16_000, cfg.max_offset_16k, false) else {
            notes.push(format!("reference {name} could not be aligned to the input"));
            continue;
        };
        let lag = al.lag.round() as isize;
        // The region on the reference's timeline; a reference a few frames short at either end
        // (a render ends on whole 10 ms frames) is padded with silence there.
        let (lo, hi) = (a as isize + lag, b as isize + lag);
        let covered = (hi.min(x.len() as isize) - lo.max(0)).max(0) as usize;
        if covered * 100 < (b - a) * 95 {
            notes.push(format!("reference {name} covers {} of the clip's {} ms of speech region", covered / 16, (b - a) / 16));
            continue;
        }
        let audio: Vec<f32> = (lo..hi).map(|i| if i >= 0 { x.get(i as usize).copied().unwrap_or(0.0) } else { 0.0 }).collect();
        let f0 = match an.list.first_mut() {
            Some(a0) => a0.contour(&audio).unwrap_or_default(),
            None => Vec::new(),
        };
        if let Some(s) = spk.as_deref_mut() {
            let mask = speech_frame_mask(&audio, SPEAKER_SPEECH_RANGE_DB);
            if let Ok(e) = s.embed_masked(&audio, &mask) {
                if stand_in {
                    stand_in_embeddings.push(e);
                } else {
                    ref_embeddings.push(e);
                }
            }
        }
        refs.push(RefSeg { audio, f0, stand_in });
    }
    // The stochastic floor of the upstream references, else of the stand-ins (flagged as such).
    let floor_of = |stand_in: bool| {
        let mut of_kind = refs.iter().filter(|r| r.stand_in == stand_in);
        match (of_kind.next(), of_kind.next()) {
            (Some(r0), Some(r1)) => mcd_pair(r0, &r1.audio).ok().map(|(e, s, _)| (e, s)),
            _ => None,
        }
    };
    let (ref_floor, ref_floor_stand_in) = match floor_of(false) {
        Some(f) => (Some(f), false),
        None => {
            let f = floor_of(true);
            (f, f.is_some())
        }
    };
    if refs.iter().any(|r| r.stand_in) {
        notes.push("stand-in references: scored apart (mcd_stand_in_db), never in the T6 MCD row".into());
    }
    Ok(ClipCtx {
        id: id.to_string(),
        path,
        sha256: String::new(),
        input16,
        speech,
        region,
        events,
        contours,
        truth,
        truth_coverage,
        ref_files,
        stand_in_files,
        refs,
        ref_floor,
        ref_floor_stand_in,
        ref_embeddings,
        stand_in_embeddings,
        target: None,
        target_source: "none",
        notes,
    })
}

/// Envelope and STFT MCD of `test` (cut over the same input region as `r`) against reference `r`,
/// after a fine alignment (waveforms within 30 ms; two renditions of one voice correlate).
/// `(envelope dB, STFT dB, shift samples)`.
fn mcd_pair(r: &RefSeg, test: &[f32]) -> Result<(f64, f64, isize), String> {
    let delta = align(&r.audio, test, 16_000, 480, true).map_or(0, |al| al.lag.round() as isize);
    let (ref_part, test_part, f0_from) = if delta >= 0 {
        let d = delta as usize;
        let n = r.audio.len().min(test.len().saturating_sub(d));
        (&r.audio[..n], &test[d..d + n], 0usize)
    } else {
        let d = (-delta) as usize;
        let n = r.audio.len().saturating_sub(d).min(test.len());
        (&r.audio[d..d + n], &test[..n], d / 160)
    };
    if ref_part.len() < 16_000 {
        return Err("MCD: under 1 s of overlap with the reference".into());
    }
    let f0 = r.f0.get(f0_from..).unwrap_or_default();
    let env = envelope_mcd_db(ref_part, test_part, f0, 16_000)?;
    let stft = mcd_db(ref_part, test_part, 16_000)?;
    Ok((env, stft, delta))
}

/// A stand-in reference: `voice` converting the whole 16 kHz clip offline (`render_whole`: the clip
/// with about 1 s of reflect padding, one pass), on the clip's timeline at 16 kHz.
fn stand_in_reference(voice: &mut OfflineVoice, stream: &StreamSpec, input16: &[f32]) -> Result<Vec<f32>, String> {
    // The whole render reflect-pads its region from a grid origin at least a second before it, so
    // the recorded stream starts with a lead (its samples are never used).
    const LEAD_16K: usize = 17_280;
    let mut lead = vec![0.0f32; LEAD_16K];
    lead.extend_from_slice(input16);
    let mut rec = StreamRecorder::default();
    rec.record(&ConversionCtx {
        mono_16k: &lead,
        new_frames: lead.len(),
        pitch_shift_semitones: stream.pitch_semitones,
        index_rate: stream.index_rate,
        protect: stream.protect,
        speaker_id: stream.speaker_id,
        stream_pos_16k: 0,
        reference: None,
        singing: Default::default(),
    });
    let len = (input16.len() / 160 * 160) as u64;
    let a = LEAD_16K as u64;
    let out = render_whole(voice, stream, &rec, (a, a + len), 0, &RenderOpts::default())?;
    let mut x = vec![0.0f32; (out.out.start_16k.saturating_sub(a)) as usize];
    x.extend(resample(&out.out.audio, out.out.rate, 16_000)?);
    Ok(x)
}

// ---------------------------------------------------------------------------------------------
// Scoring one rendition
// ---------------------------------------------------------------------------------------------

/// F0 errors of one output analyzer against a reference contour.
#[derive(Debug, Clone, Serialize)]
pub struct F0Score {
    /// Fine pitch error: RMSE over voiced frames within 20 % (cents).
    pub fpe_cents: Option<f64>,
    pub gpe: f64,
    pub vuv: f64,
    pub voiced_frames: usize,
    pub frames: usize,
}

/// The scores of one rendition of one clip.
#[derive(Debug, Clone, Default, Serialize)]
pub struct RowScores {
    pub offset_ms: Option<f64>,
    pub seconds: Option<f64>,
    pub loudness_lufs: Option<f64>,
    pub gain_db: Option<f64>,
    pub dnsmos_ovrl: Option<f64>,
    pub dnsmos_sig: Option<f64>,
    pub dnsmos_bak: Option<f64>,
    pub dnsmos_seconds: Option<f64>,
    /// Clips under 9.01 s: the tiled score (advisory; never in a verdict).
    pub dnsmos_tiled_ovrl: Option<f64>,
    /// Against the clip's truth, by output analyzer.
    pub f0_truth: BTreeMap<String, F0Score>,
    /// Against the same analyzer on the input (advisory).
    pub f0_self: BTreeMap<String, F0Score>,
    /// Share of output frames where every analyzer agrees within 50 cents.
    pub f0_output_agreement: Option<f64>,
    /// Envelope / STFT MCD to the neutral (upstream) references (mean over them). Only these feed
    /// the T6 MCD row.
    pub mcd_ref_db: Option<f64>,
    pub mcd_ref_stft_db: Option<f64>,
    /// The same to the stand-in references (Sigil's own whole-utterance render): neutral between
    /// Sigil's rows and their ablations only, so it is reported apart and never decides T6.
    pub mcd_stand_in_db: Option<f64>,
    pub mcd_stand_in_stft_db: Option<f64>,
    pub speaker_sim: Option<f64>,
    pub artifacts_per_min: Option<f64>,
    /// Phase-locked clicks on the rendition's own block grid (searched phase).
    pub seam_clicks: Option<u64>,
    pub seam_events: Option<u64>,
    pub seam_period_ms: Option<f64>,
    /// The first 200 ms after each speech onset (after at least 300 ms of silence) against the
    /// rendition's median speech level, minus the same for the input (dB; 0 = the input's own onset
    /// shape, strongly negative = onsets muted, e.g. by a long context window).
    pub onset_level_db: Option<f64>,
    pub onsets: Option<u64>,
}

/// Where a rendition sits on the input's timeline (for the listening pairs).
#[derive(Debug, Clone, Copy)]
struct Placement {
    lag: isize,
    a: usize,
    b: usize,
    gain_db: Option<f64>,
}

fn f0_score(reference: &[f32], test: &[f32], mask: &[bool]) -> Option<F0Score> {
    let n = reference.len().min(test.len()).min(mask.len());
    // NaN reference frames (no truth there) are not scored.
    let keep: Vec<bool> = (0..n).map(|i| mask[i] && !reference[i].is_nan()).collect();
    let r: Vec<f32> = reference[..n].iter().map(|v| if v.is_nan() { 0.0 } else { *v }).collect();
    f0_errors_v2(&r, &test[..n], Some(&keep)).ok().map(|e| F0Score {
        fpe_cents: e.fine_rmse_cents,
        gpe: e.v1.gross_pitch_error,
        vuv: e.v1.vuv_error,
        voiced_frames: e.v1.voiced_frames,
        frames: e.v1.frames,
    })
}

/// Frames (10 ms) where speech starts after at least `gap` frames without speech.
fn speech_onsets(speech: &[bool], gap: usize) -> Vec<usize> {
    let mut out = Vec::new();
    // The clip's start is not an onset: it measures a system's start-up, not onset muting.
    let mut quiet = 0usize;
    for (i, &v) in speech.iter().enumerate() {
        if v {
            if quiet >= gap {
                out.push(i);
            }
            quiet = 0;
        } else {
            quiet += 1;
        }
    }
    out
}

/// Frames of an onset window.
const ONSET_FRAMES: usize = 20;
/// The lowest an onset reads (dB under the rendition's speech level): a muted onset.
const ONSET_FLOOR_DB: f64 = -60.0;

/// The rendition's level over the first 200 ms of each speech onset of the input, relative to the
/// input's own onset level (`(mean dB, onsets)`; 10 ms frame levels of both on the same frames).
/// Speech is the input's loud frames (within 25 dB of its 95th percentile); an onset is a loud
/// frame after at least 300 ms without one (never the clip's start, which measures a system's
/// start-up) whose window really holds speech (within 10 dB of the input's median loud level).
/// Each side is relative to its own median loud level; a muted onset reads at most 60 dB down.
fn onset_levels(input_db: &[f64], rend_db: &[f64]) -> Option<(f64, usize)> {
    let n = input_db.len().min(rend_db.len());
    if n < ONSET_FRAMES * 2 {
        return None;
    }
    let mut sorted = input_db[..n].to_vec();
    sorted.sort_by(f64::total_cmp);
    let p95 = sorted[((n - 1) as f64 * 0.95) as usize];
    let loud: Vec<bool> = input_db[..n].iter().map(|&d| d >= p95 - 25.0).collect();
    let median_of = |db: &[f64]| {
        let mut v: Vec<f64> = (0..n).filter(|&i| loud[i]).map(|i| db[i]).collect();
        v.sort_by(f64::total_cmp);
        v.get(v.len() / 2).copied()
    };
    let (mi, mr) = (median_of(input_db)?, median_of(rend_db)?);
    let window = |db: &[f64], o: usize| 10.0 * (db[o..o + ONSET_FRAMES].iter().map(|d| 10f64.powf(d / 10.0)).sum::<f64>() / ONSET_FRAMES as f64).max(1e-20).log10();
    let vals: Vec<f64> = speech_onsets(&loud, 30)
        .into_iter()
        .filter(|&o| o + ONSET_FRAMES <= n)
        .filter(|&o| window(input_db, o) >= mi - 10.0)
        .map(|o| (window(rend_db, o) - mr).max(ONSET_FLOOR_DB) - (window(input_db, o) - mi))
        .collect();
    (!vals.is_empty()).then(|| (vals.iter().sum::<f64>() / vals.len() as f64, vals.len()))
}

fn frame_db(x: &[f32]) -> Vec<f64> {
    x.chunks(160)
        .map(|c| 10.0 * (c.iter().map(|&v| f64::from(v) * f64::from(v)).sum::<f64>() / c.len().max(1) as f64 + 1e-20).log10())
        .collect()
}

/// Phase-locked seam clicks of a rendition on its own block grid of `period` samples (16 kHz;
/// impulses `events` at the rendition's own sample positions, its file starting a block): the
/// busiest ±1.5 ms phase window holds `k` of the `n` events; under no phase locking `k` is
/// Binomial(`n`, window / period) at a given phase, and the busiest of `period / window` phases is
/// judged at the Bonferroni-corrected 97.5 % quantile. Returns `(clicks beyond it, k)`. The phase is
/// searched because another app's join phase is unknown; every rendition is judged the same way.
fn seam_clicks(events: &[usize], len: usize, period: usize) -> (u64, u64) {
    let rep = click_report(events, len, 16_000, period, None, 1.5);
    let width = (2 * ((1.5 / 1000.0 * 16_000.0) as usize) + 1).min(period.max(1));
    if period == 0 || width >= period || events.is_empty() {
        return (0, rep.edge_events as u64);
    }
    let p = width as f64 / period as f64;
    let phases = (period as f64 / width as f64).max(1.0);
    let crit = super::quality::binomial_quantile(1.0 - 0.025 / phases, events.len() as u64, p);
    ((rep.edge_events as u64).saturating_sub(crit), rep.edge_events as u64)
}

/// Score one rendition (16 kHz) of a clip.
#[allow(clippy::too_many_arguments)]
fn score_rendition(
    ctx: &ClipCtx,
    cand: &[f32],
    block_ms: Option<f64>,
    an: &mut Analyzers,
    sc: &mut Scorers,
    spk: Option<&mut SpeakerEmbedder>,
    cfg: &ScoreCfg,
    notes: &mut Vec<String>,
) -> Option<(RowScores, Placement)> {
    let Some(al) = align(&ctx.input16, cand, 16_000, cfg.max_offset_16k, false) else {
        notes.push("could not be aligned to the input (too short or silent)".into());
        return None;
    };
    let (ra, rb) = ctx.region;
    // Refine the 10 ms envelope estimate on 1 ms hops (the latency read from it, and F0 frames).
    let lag = {
        let coarse = al.lag.round() as isize;
        let from = (ra as isize + coarse - 320).max(0) as usize;
        match super::e2e_loopback::envelope_delay(&ctx.input16[ra..rb], cand.get(from..).unwrap_or_default(), 16_000, 0.04) {
            Some((d, corr)) if corr > 0.5 => from as isize + (d * 16_000.0).round() as isize - ra as isize,
            _ => coarse,
        }
    };
    let a = ra.max((-lag).max(0) as usize);
    let b = rb.min((cand.len() as isize - lag).max(0) as usize);
    if b <= a + 8_000 {
        notes.push(format!("overlaps the input's speech on {} ms only", b.saturating_sub(a) / 16));
        return None;
    }
    if b - a + 160 < rb - ra {
        notes.push(format!("covers {} of the clip's {} ms of speech region", (b - a) / 16, (rb - ra) / 16));
    }
    let seg = &cand[(a as isize + lag) as usize..(b as isize + lag) as usize];
    let mut s = RowScores {
        offset_ms: Some(lag as f64 / 16.0),
        seconds: Some((b - a) as f64 / 16_000.0),
        ..RowScores::default()
    };
    // Loudness, then DNSMOS on clips of a whole model window.
    let loud = loudness_gain_db(seg, 16_000, cfg.loudness_lufs);
    s.loudness_lufs = loud.map(|l| l.0);
    s.gain_db = loud.map(|l| l.1);
    if loud.is_none() {
        // Below the BS.1770 absolute gate (silent, muted or under 400 ms): nothing to normalise,
        // so no DNSMOS; the cell counts as unscored (see `unscored_cells`).
        notes.push("loudness unmeasurable (below the BS.1770 absolute gate): no DNSMOS".into());
    }
    if let (Some(d), Some((_, gain))) = (sc.dnsmos.as_mut(), loud) {
        if ctx.dnsmos_ok() {
            // The region, widened symmetrically to one window where the rendition has the audio.
            let (mut lo, mut hi) = ((a as isize + lag) as usize, (b as isize + lag) as usize);
            while hi - lo < DNSMOS_LEN && (lo > 0 || hi < cand.len()) {
                let need = DNSMOS_LEN - (hi - lo);
                let left = lo.min(need.div_ceil(2));
                lo -= left;
                hi = (hi + (need - left)).min(cand.len());
            }
            if hi - lo >= DNSMOS_LEN {
                match d.score(&apply_gain(&cand[lo..hi], gain)) {
                    Ok(v) => {
                        s.dnsmos_ovrl = Some(v.ovrl);
                        s.dnsmos_sig = Some(v.sig);
                        s.dnsmos_bak = Some(v.bak);
                        s.dnsmos_seconds = Some((hi - lo) as f64 / 16_000.0);
                    }
                    Err(e) => notes.push(format!("DNSMOS: {e}")),
                }
            } else {
                notes.push("DNSMOS: the rendition is shorter than one 9.01 s window".into());
            }
        } else if let Ok(v) = d.score(&apply_gain(seg, gain)) {
            s.dnsmos_tiled_ovrl = Some(v.ovrl);
        }
    }
    // F0: every analyzer on the output, against the truth and against itself on the input.
    let (fa, fb) = (a / 160, b / 160);
    let mask: Vec<bool> = ctx.speech.get(fa..fb.min(ctx.speech.len())).unwrap_or_default().to_vec();
    let mut outs = Vec::new();
    for (i, x) in an.list.iter_mut().enumerate() {
        let name = x.method.to_string();
        match x.contour(seg) {
            Ok(fc) => {
                if let Some((_, truth)) = &ctx.truth {
                    let t = truth.get(fa..fb.min(truth.len())).unwrap_or_default();
                    // Truth frames decide on their own (unvoiced truth counts for V/UV).
                    let all = vec![true; t.len()];
                    if let Some(e) = f0_score(t, &fc, &all) {
                        s.f0_truth.insert(name.clone(), e);
                    }
                }
                if let Some(c) = ctx.contours.get(i).filter(|c| !c.is_empty()) {
                    let c = c.get(fa..fb.min(c.len())).unwrap_or_default();
                    if let Some(e) = f0_score(c, &fc, &mask) {
                        s.f0_self.insert(name, e);
                    }
                }
                outs.push(fc);
            }
            Err(e) => notes.push(format!("F0 {name}: {e}")),
        }
    }
    if outs.len() >= 2 {
        s.f0_output_agreement = output_agreement(&outs, &mask);
    }
    // MCD to the neutral references.
    if !ctx.refs.is_empty() {
        let (mut env, mut stft) = (Vec::new(), Vec::new());
        let (mut env_si, mut stft_si) = (Vec::new(), Vec::new());
        for r in &ctx.refs {
            match mcd_pair(r, seg) {
                Ok((e, st, _)) if r.stand_in => {
                    env_si.push(e);
                    stft_si.push(st);
                }
                Ok((e, st, _)) => {
                    env.push(e);
                    stft.push(st);
                }
                Err(e) => notes.push(format!("MCD to reference: {e}")),
            }
        }
        let mean = |v: &[f64]| (!v.is_empty()).then(|| v.iter().sum::<f64>() / v.len() as f64);
        s.mcd_ref_db = mean(&env);
        s.mcd_ref_stft_db = mean(&stft);
        s.mcd_stand_in_db = mean(&env_si);
        s.mcd_stand_in_stft_db = mean(&stft_si);
    }
    // Target-speaker similarity.
    if let (Some(spk), Some(t)) = (spk, ctx.target.as_deref()) {
        let m = speech_frame_mask(seg, SPEAKER_SPEECH_RANGE_DB);
        match spk.embed_masked(seg, &m) {
            Ok(e) => s.speaker_sim = Some(cosine(&e, t)),
            Err(e) => notes.push(format!("speaker: {e}")),
        }
    }
    // Onsets: the rendition's level after each speech onset, relative to the input's own.
    if let Some((level, n)) = onset_levels(&frame_db(&ctx.input16[a..b]), &frame_db(seg)) {
        s.onset_level_db = Some(level);
        s.onsets = Some(n as u64);
    }
    // Artifacts (on the input's timeline) and seams (on the rendition's own block grid).
    let ev = click_events(cand, 16_000, &sc.click);
    let in_region: Vec<usize> = ev
        .iter()
        .copied()
        .filter(|&e| e as isize >= a as isize + lag && (e as isize) < b as isize + lag)
        .collect();
    let on_input: Vec<f64> = in_region.iter().map(|&e| (e as f64 - lag as f64) / 16_000.0).collect();
    let added = unmatched(&to_ticks(&on_input), &to_ticks(&ctx.events), (EVENT_MATCH_S * 1e4) as usize);
    s.artifacts_per_min = Some(added.len() as f64 / ((b - a) as f64 / 16_000.0 / 60.0));
    if let Some(block) = block_ms.filter(|b| *b >= 5.0) {
        let (clicks, at_phase) = seam_clicks(&in_region, cand.len(), (block * 16.0).round() as usize);
        s.seam_clicks = Some(clicks);
        s.seam_events = Some(at_phase);
        s.seam_period_ms = Some(block);
    }
    Some((
        s,
        Placement {
            lag,
            a,
            b,
            gain_db: loud.map(|l| l.1),
        },
    ))
}

// ---------------------------------------------------------------------------------------------
// Pairings and verdicts
// ---------------------------------------------------------------------------------------------

/// The scored renditions: `rows[point][clip]`.
type Rows = Vec<Vec<Option<RowScores>>>;

/// Per-clip values of one metric for two points: `(clip, a, b)` where both were scored.
fn paired_values(rows: &Rows, clips: &[ClipCtx], a: usize, b: usize, f: &dyn Fn(&RowScores) -> Option<f64>) -> Vec<(String, f64, f64)> {
    clips
        .iter()
        .enumerate()
        .filter_map(|(c, clip)| {
            let x = rows[a][c].as_ref().and_then(f)?;
            let y = rows[b][c].as_ref().and_then(f)?;
            Some((clip.id.clone(), x, y))
        })
        .collect()
}

/// One Sigil point against one other point (a competitor, or the row an ablation ablates).
#[derive(Debug, Clone, Serialize)]
pub struct Pairing {
    pub sigil: String,
    pub other: String,
    pub tier: Tier,
    /// Sigil's latency over the other's, when both are known and of the same kind.
    pub latency_ratio: Option<f64>,
    pub latency_note: String,
    /// Differences Sigil − other.
    pub dnsmos: Option<PairedDiff>,
    /// Gross pitch error against the truth, by output analyzer.
    pub gpe: BTreeMap<String, PairedDiff>,
    pub fpe: BTreeMap<String, PairedDiff>,
    pub mcd: Option<PairedDiff>,
    /// MCD to the stand-in references: for ablations (same model code), never in a T6 verdict.
    pub mcd_stand_in: Option<PairedDiff>,
    pub speaker: Option<PairedDiff>,
    pub artifacts: Option<PairedDiff>,
    pub seam_clicks: Option<PairedDiff>,
    pub onset_level: Option<PairedDiff>,
    /// Advisory: the tiled DNSMOS of clips under 9.01 s.
    pub dnsmos_tiled: Option<PairedDiff>,
}

fn pairing(points: &[Point], rows: &Rows, clips: &[ClipCtx], s: usize, k: usize, analyzers: &[&'static str]) -> Pairing {
    let (ps, pk) = (&points[s], &points[k]);
    let (latency_ratio, latency_note) = match (ps.latency_ms, pk.latency_ms) {
        (Some(_), Some(_)) if ps.latency_kind.is_empty() || pk.latency_kind.is_empty() => (None, "a latency kind is unknown: not compared".to_string()),
        (Some(_), Some(_)) if ps.latency_kind == UNCALIBRATED || pk.latency_kind == UNCALIBRATED => {
            (None, "a live latency is an uncalibrated capture offset: not compared with another system's".to_string())
        }
        (Some(_), Some(_)) if ps.latency_kind != pk.latency_kind => (None, format!("latency kinds differ ({} vs {}): refused", ps.latency_kind, pk.latency_kind)),
        (Some(a), Some(b)) if b > 0.0 => (Some(a / b), format!("{a:.1} / {b:.1} ms ({})", ps.latency_kind)),
        _ => (None, "a latency is unknown".to_string()),
    };
    let v = |f: &dyn Fn(&RowScores) -> Option<f64>| PairedDiff::of(&paired_values(rows, clips, s, k, f));
    let mut gpe = BTreeMap::new();
    let mut fpe = BTreeMap::new();
    for &a in analyzers {
        if let Some(d) = v(&|r| r.f0_truth.get(a).map(|e| e.gpe)) {
            gpe.insert(a.to_string(), d);
        }
        if let Some(d) = v(&|r| r.f0_truth.get(a).and_then(|e| e.fpe_cents)) {
            fpe.insert(a.to_string(), d);
        }
    }
    Pairing {
        sigil: ps.name.clone(),
        other: pk.name.clone(),
        tier: ps.tier,
        latency_ratio,
        latency_note,
        dnsmos: v(&|r| r.dnsmos_ovrl),
        gpe,
        fpe,
        mcd: v(&|r| r.mcd_ref_db),
        mcd_stand_in: v(&|r| r.mcd_stand_in_db),
        speaker: v(&|r| r.speaker_sim),
        artifacts: v(&|r| r.artifacts_per_min),
        seam_clicks: v(&|r| r.seam_clicks.map(|c| c as f64)),
        onset_level: v(&|r| r.onset_level_db),
        dnsmos_tiled: v(&|r| r.dnsmos_tiled_ovrl),
    }
}

/// The T6 verdicts on one competitor point.
#[derive(Debug, Clone, Serialize)]
pub struct CompetitorVerdict {
    pub point: String,
    pub system: String,
    pub tier: Tier,
    pub dnsmos: RowVerdict,
    pub gpe: RowVerdict,
    pub mcd: RowVerdict,
    pub latency: RowVerdict,
}

impl CompetitorVerdict {
    fn rows(&self) -> [&RowVerdict; 4] {
        [&self.dnsmos, &self.gpe, &self.mcd, &self.latency]
    }
}

/// GPE non-inferiority of a pairing under every output analyzer: the worst status over them
/// (Pass only when every analyzer passes); `None` without truth.
fn gpe_status(p: &Pairing, margin: f64, min_clips: usize) -> Option<(Status, String)> {
    if p.gpe.is_empty() {
        return None;
    }
    let mut worst = Status::Pass;
    let mut parts = Vec::new();
    for (a, d) in &p.gpe {
        let st = if d.n < min_clips { Status::Undecided } else { noninferior_lower(d, margin) };
        parts.push(format!("{a}: {:+.4} [{:+.4}, {:+.4}] n {}", d.mean, d.t_lo, d.t_hi, d.n));
        worst = match (worst, st) {
            (Status::Fail, _) | (_, Status::Fail) => Status::Fail,
            (Status::NotShown, _) | (_, Status::NotShown) => Status::NotShown,
            (Status::Undecided, _) | (_, Status::Undecided) => Status::Undecided,
            _ => Status::Pass,
        };
    }
    Some((worst, parts.join("; ")))
}

/// Whether Sigil's row `p` is non-inferior to the competitor on every quality row measured (the
/// latency row's condition).
fn quality_noninferior(p: &Pairing, m: &Margins, min_clips: usize) -> (bool, String) {
    let mut why = Vec::new();
    let mut ok = true;
    match &p.dnsmos {
        Some(d) if d.n >= min_clips => {
            if noninferior_higher(d, m.dnsmos_noninferiority) != Status::Pass {
                ok = false;
                why.push(format!("DNSMOS not non-inferior ({:+.3} [{:+.3}, {:+.3}])", d.mean, d.t_lo, d.t_hi));
            }
        }
        Some(d) => {
            ok = false;
            why.push(format!("DNSMOS on {} clips (< {min_clips})", d.n));
        }
        None => {
            ok = false;
            why.push("DNSMOS not measured on clips of 9.01 s or more".into());
        }
    }
    match gpe_status(p, m.gpe_noninferiority, min_clips) {
        Some((Status::Pass, _)) => {}
        Some((st, d)) => {
            ok = false;
            why.push(format!("GPE {st:?} ({d})"));
        }
        None => why.push("GPE not measured (no truth)".into()),
    }
    if let Some(d) = &p.mcd {
        if d.n < min_clips || noninferior_lower(d, m.mcd_noninferiority_db) != Status::Pass {
            ok = false;
            why.push(format!("MCD not non-inferior ({:+.3} [{:+.3}, {:+.3}] dB, n {})", d.mean, d.t_lo, d.t_hi, d.n));
        }
    }
    (ok, why.join("; "))
}

/// Why a point must not be heard in the blind A/B pairs (`None` = fair to hear): refused, or, for
/// a pair a verdict would count (`real_time`), unable to run in real time on this GPU.
fn unfair_to_hear(p: &Point, real_time: bool) -> Option<String> {
    if !p.refusals.is_empty() {
        Some(format!("refused: {}", p.refusals.join("; ")))
    } else if real_time && p.feasible == Some(false) {
        Some(format!("cannot run in real time on this GPU ({})", p.feasibility))
    } else {
        None
    }
}

/// The clips a row has no complete score for: no scored rendition (a conversion error, a missing
/// or refused WAV, a rendition too short or silent to align), or no DNSMOS where the clip needs one
/// and the scorer is loaded (a rendition muted below the BS.1770 gate, shorter than one DNSMOS
/// window). Pairing would drop these clips silently, and `n` would stay above `min_clips`: a Sigil
/// row that fails 4 of 40 clips would still decide on the other 36, and a competitor that mutes the
/// same clips would be excused. Such a row is incomplete instead.
fn unscored_cells(row: &[Option<RowScores>], clips: &[ClipCtx], dnsmos_loaded: bool) -> Vec<String> {
    clips
        .iter()
        .zip(row)
        .filter(|(clip, cell)| match cell {
            None => true,
            Some(s) => dnsmos_loaded && clip.dnsmos_ok() && s.dnsmos_ovrl.is_none(),
        })
        .map(|(clip, _)| clip.id.clone())
        .collect()
}

/// `a, b, c and 4 more`.
fn list_some(items: &[String], max: usize) -> String {
    let shown = items.iter().take(max).cloned().collect::<Vec<_>>().join(", ");
    if items.len() > max {
        format!("{shown} and {} more", items.len() - max)
    } else {
        shown
    }
}

/// The verdicts of competitor point `k` against the Sigil points `sigil` of its tier.
fn verdict_for(points: &[Point], pairings: &[Pairing], k: usize, sigil: &[usize], m: &Margins, min_clips: usize) -> CompetitorVerdict {
    let pk = &points[k];
    let mk = |st: Status, d: &str| RowVerdict::new(st, None, d);
    let mut v = CompetitorVerdict {
        point: pk.name.clone(),
        system: pk.system.clone(),
        tier: pk.tier,
        dnsmos: mk(Status::Undecided, ""),
        gpe: mk(Status::Undecided, ""),
        mcd: mk(Status::Undecided, ""),
        latency: mk(Status::Undecided, ""),
    };
    if !pk.refusals.is_empty() {
        let why = format!("refused: {}", pk.refusals.join("; "));
        v.dnsmos = mk(Status::Refused, &why);
        v.gpe = mk(Status::Refused, &why);
        v.mcd = mk(Status::Refused, &why);
        v.latency = mk(Status::Refused, &why);
        return v;
    }
    match pk.feasible {
        Some(false) => {
            let why = format!("excluded: cannot run in real time on this GPU ({})", pk.feasibility);
            v.dnsmos = mk(Status::Excluded, &why);
            v.gpe = mk(Status::Excluded, &why);
            v.mcd = mk(Status::Excluded, &why);
            v.latency = mk(Status::Excluded, &why);
            return v;
        }
        None => {
            let why = format!("undecided: {}", pk.feasibility);
            v.dnsmos = mk(Status::Undecided, &why);
            v.gpe = mk(Status::Undecided, &why);
            v.mcd = mk(Status::Undecided, &why);
            v.latency = mk(Status::Undecided, &why);
            return v;
        }
        Some(true) => {}
    }
    if !pk.unscored_clips.is_empty() {
        let why = format!(
            "incomplete: {} clip(s) have no complete score for this point ({}); a row that misses clips decides nothing",
            pk.unscored_clips.len(),
            list_some(&pk.unscored_clips, 6)
        );
        v.dnsmos = mk(Status::Undecided, &why);
        v.gpe = mk(Status::Undecided, &why);
        v.mcd = mk(Status::Undecided, &why);
        v.latency = mk(Status::Undecided, &why);
        return v;
    }
    // Sigil's rows count only when whole: converted, real time and scored on every clip.
    let mine: Vec<&Pairing> = sigil
        .iter()
        .filter(|&&s| points[s].refusals.is_empty() && points[s].feasible != Some(false) && points[s].unscored_clips.is_empty())
        .filter_map(|&s| pairings.iter().find(|p| p.sigil == points[s].name && p.other == pk.name))
        .collect();
    if mine.is_empty() {
        let why = "no fair, real-time Sigil row that scored every clip in this tier";
        v.dnsmos = mk(Status::Undecided, why);
        v.gpe = mk(Status::Undecided, why);
        v.mcd = mk(Status::Undecided, why);
        v.latency = mk(Status::Undecided, why);
        return v;
    }
    let matched: Vec<&&Pairing> = mine.iter().filter(|p| p.latency_ratio.is_some_and(|r| r <= 1.0 + 1e-9)).collect();
    // (a) DNSMOS superiority at matched latency.
    v.dnsmos = if matched.is_empty() {
        mk(Status::Undecided, "no Sigil row at matched latency (≤ the competitor's, same kind)")
    } else {
        let mut best: Option<RowVerdict> = None;
        for p in &matched {
            let st = match &p.dnsmos {
                Some(d) if d.n >= min_clips => {
                    if d.mean >= m.dnsmos_superiority && d.t_lo > 0.0 {
                        Status::Pass
                    } else {
                        Status::Fail
                    }
                }
                _ => Status::Undecided,
            };
            let detail = match &p.dnsmos {
                Some(d) => format!(
                    "{}: {:+.3} [{:+.3}, {:+.3}] over {} clips{}",
                    p.sigil,
                    d.mean,
                    d.t_lo,
                    d.t_hi,
                    d.n,
                    if d.fragile() { " (the bootstrap disagrees: fragile)" } else { "" }
                ),
                None => format!("{}: DNSMOS not measured on clips of 9.01 s or more", p.sigil),
            };
            let cand = RowVerdict::new(st, Some(p.sigil.as_str()), detail);
            best = Some(match best {
                Some(b) if rank(b.status) >= rank(cand.status) => b,
                _ => cand,
            });
        }
        best.expect("matched is not empty")
    };
    // (b) GPE non-inferiority at matched latency, under every analyzer.
    v.gpe = best_of(&matched, |p| {
        gpe_status(p, m.gpe_noninferiority, min_clips).map(|(st, d)| RowVerdict::new(st, Some(p.sigil.as_str()), format!("{}: {d}", p.sigil)))
    })
    .unwrap_or_else(|| mk(Status::Undecided, if matched.is_empty() { "no Sigil row at matched latency" } else { "no F0 truth (EGG or a 3-analyzer consensus)" }));
    // (b) MCD-to-reference non-inferiority at matched latency.
    v.mcd = best_of(&matched, |p| {
        p.mcd.as_ref().map(|d| {
            let st = if d.n < min_clips { Status::Undecided } else { noninferior_lower(d, m.mcd_noninferiority_db) };
            RowVerdict::new(st, Some(p.sigil.as_str()), format!("{}: {:+.3} [{:+.3}, {:+.3}] dB over {} clips", p.sigil, d.mean, d.t_lo, d.t_hi, d.n))
        })
    })
    .unwrap_or_else(|| {
        mk(
            Status::Undecided,
            if matched.is_empty() {
                "no Sigil row at matched latency"
            } else if matched.iter().any(|p| p.mcd_stand_in.is_some()) {
                "only stand-in references (Sigil's own render): not neutral between engines; the MCD row needs upstream references (scripts/vc-bench/rvc_offline_reference.py)"
            } else {
                "no neutral references (scripts/vc-bench/rvc_offline_reference.py)"
            },
        )
    });
    // (c) Latency ≤ 0.7×, as Pareto: some Sigil row at that latency, non-inferior on every row.
    let fast: Vec<&&Pairing> = mine.iter().filter(|p| p.latency_ratio.is_some_and(|r| r <= m.latency_ratio + 1e-9)).collect();
    v.latency = if mine.iter().all(|p| p.latency_ratio.is_none()) {
        mk(Status::Undecided, &format!("no comparable latency: {}", mine.iter().map(|p| p.latency_note.clone()).collect::<Vec<_>>().join("; ")))
    } else if fast.is_empty() {
        mk(Status::Fail, &format!("no Sigil row at ≤ {}× the competitor's latency", m.latency_ratio))
    } else {
        let mut out = None;
        let mut why = Vec::new();
        for p in &fast {
            let (ok, w) = quality_noninferior(p, m, min_clips);
            if ok {
                out = Some(RowVerdict::new(Status::Pass, Some(p.sigil.as_str()), format!("{}: latency ratio {:.2}, non-inferior on every quality row", p.sigil, p.latency_ratio.unwrap_or(f64::NAN))));
                break;
            }
            why.push(format!("{} (ratio {:.2}): {w}", p.sigil, p.latency_ratio.unwrap_or(f64::NAN)));
        }
        out.unwrap_or_else(|| mk(Status::NotShown, &why.join(" | ")))
    };
    v
}

/// Ordering of statuses for "the best Sigil row" (a pass wins).
fn rank(s: Status) -> u8 {
    match s {
        Status::Pass => 5,
        Status::NotShown => 3,
        Status::Fail => 2,
        Status::Undecided => 1,
        Status::Refused | Status::Excluded => 0,
    }
}

fn best_of(pairings: &[&&Pairing], f: impl Fn(&Pairing) -> Option<RowVerdict>) -> Option<RowVerdict> {
    pairings.iter().filter_map(|p| f(p)).max_by_key(|v| rank(v.status))
}

/// The T6 decision over every competitor point.
#[derive(Debug, Clone, Serialize)]
pub struct T6Decision {
    /// `pass`, `fail` or `undecided`.
    pub status: &'static str,
    pub reasons: Vec<String>,
    /// What a reader must weigh next to the status: operating-point differences of the counted
    /// competitor points (another retrieval strength or protect, another F0 method) and rows that
    /// miss clips. A pass with caveats is not a clean pass.
    pub caveats: Vec<String>,
}

fn t6_decision(verdicts: &[CompetitorVerdict]) -> T6Decision {
    let mut reasons = Vec::new();
    let counted: Vec<&CompetitorVerdict> = verdicts.iter().filter(|v| v.rows().iter().all(|r| r.status != Status::Excluded)).collect();
    if counted.is_empty() {
        return T6Decision {
            status: "undecided",
            reasons: vec!["no counted competitor point (none given, or every one excluded)".into()],
            caveats: Vec::new(),
        };
    }
    let mut any_fail = false;
    let mut all_pass = true;
    for v in &counted {
        for (row, r) in ["dnsmos", "gpe", "mcd", "latency"].iter().zip(v.rows()) {
            match r.status {
                Status::Pass => {}
                Status::Fail | Status::NotShown => {
                    any_fail = true;
                    all_pass = false;
                    reasons.push(format!("{} {row}: {:?} ({})", v.point, r.status, r.detail));
                }
                _ => {
                    all_pass = false;
                    reasons.push(format!("{} {row}: {:?} ({})", v.point, r.status, r.detail));
                }
            }
        }
    }
    let status = if all_pass {
        "pass"
    } else if any_fail {
        "fail"
    } else {
        "undecided"
    };
    T6Decision {
        status,
        reasons,
        caveats: Vec::new(),
    }
}

/// The Pareto front of one tier on (latency, mean DNSMOS): no other point is at most as slow and at
/// least as good, with one strict.
#[derive(Debug, Clone, Serialize)]
pub struct ParetoPoint {
    pub name: String,
    pub system: String,
    pub role: PointRole,
    pub tier: Tier,
    pub latency_ms: Option<f64>,
    pub latency_kind: String,
    pub dnsmos_mean: Option<f64>,
    pub dnsmos_ci95: Option<[f64; 2]>,
    pub dnsmos_clips: usize,
    pub front: bool,
}

fn pareto(points: &[Point], rows: &Rows) -> Vec<ParetoPoint> {
    let mut out: Vec<ParetoPoint> = points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.refusals.is_empty())
        .map(|(i, p)| {
            let v: Vec<f64> = rows[i].iter().filter_map(|r| r.as_ref().and_then(|r| r.dnsmos_ovrl)).collect();
            let ci = mean_ci(&v);
            ParetoPoint {
                name: p.name.clone(),
                system: p.system.clone(),
                role: p.role,
                tier: p.tier,
                latency_ms: p.latency_ms,
                latency_kind: p.latency_kind.clone(),
                dnsmos_mean: ci.map(|c| c.0),
                dnsmos_ci95: ci.map(|c| [c.1, c.2]),
                dnsmos_clips: v.len(),
                front: false,
            }
        })
        .collect();
    let snapshot = out.clone();
    for p in out.iter_mut() {
        let (Some(l), Some(q)) = (p.latency_ms, p.dnsmos_mean) else {
            continue;
        };
        if p.latency_kind == UNCALIBRATED {
            continue; // an uncalibrated offset has no place on a latency axis
        }
        p.front = !snapshot.iter().any(|o| {
            o.name != p.name
                && o.tier == p.tier
                && o.latency_kind == p.latency_kind
                && o.latency_kind != UNCALIBRATED
                && matches!((o.latency_ms, o.dnsmos_mean), (Some(ol), Some(oq)) if ol <= l && oq >= q && (ol < l || oq > q))
        });
    }
    out
}

/// An ablation against the Sigil row it ablates (`versus − ablation`).
#[derive(Debug, Clone, Serialize)]
pub struct AblationResult {
    pub ablation: String,
    pub versus: String,
    pub label: String,
    pub pairing: Pairing,
    /// Clips the +0.05 DNSMOS row needs at this pairing's paired SD (50 % / 80 % power).
    pub clips_needed_dnsmos: Option<(usize, usize)>,
    pub clips_needed_dnsmos_tiled: Option<(usize, usize)>,
}

// ---------------------------------------------------------------------------------------------
// The run
// ---------------------------------------------------------------------------------------------

/// Where a clip's audio comes from.
#[derive(Clone)]
enum ClipSource {
    File(PathBuf),
    /// The spec's own input (synthetic or its WAV), at the stream's device format.
    SpecInput,
}

/// A clip before analysis.
struct ClipIn {
    id: String,
    source: ClipSource,
    f0_truth: Option<PathBuf>,
    references: Vec<PathBuf>,
    /// The manifest's sha256 (checked).
    sha256: String,
}

/// A rendition to score: `(point, clip) -> 16 kHz audio`, its file, and its native audio for the
/// listening pairs.
#[derive(Default)]
struct Rendition {
    audio16: Option<Vec<f32>>,
    native: Option<(Vec<f32>, u32)>,
    wav: Option<PathBuf>,
    notes: Vec<String>,
    refused: Option<String>,
}

/// The device-format input of a clip for an in-process conversion.
fn device_input(src: &ClipSource, spec: &BenchSpec, stream: &StreamSpec, prepared: Option<&Prepared>) -> Result<Vec<f32>, String> {
    let ch = usize::from(stream.input_channels);
    match src {
        ClipSource::File(p) => render_input(&InputSpec::Wav { path: p.clone() }, stream.input_rate, ch, spec.seed),
        ClipSource::SpecInput => match prepared {
            Some(p) if stream.input_rate == spec.stream.input_rate && stream.input_channels == spec.stream.input_channels => Ok(p.input.clone()),
            _ => render_input(&spec.input, stream.input_rate, ch, spec.seed),
        },
    }
}

/// Convert a clip with an engine through the production harness (offline): `(16 kHz, native,
/// rate, offline metrics)`.
#[allow(clippy::too_many_arguments)]
fn convert(
    engine: &Engine,
    probe: AlgorithmicLatency,
    stream: &StreamSpec,
    input: &[f32],
    name: &str,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<(Vec<f32>, Vec<f32>, u32, Metrics), String> {
    let run = OfflineRun {
        name: name.to_string(),
        repeat: 1,
        thresholds: Default::default(),
    };
    let out = run_offline(&OfflineInput {
        stream,
        run: &run,
        engine,
        input,
        probe,
        record_audio: true,
        cancel,
        progress,
        recorder: None,
    })?;
    let (audio, rate, ch) = out.audio.ok_or("the conversion produced no audio")?;
    let native = channel(&audio, ch, 0);
    let out16 = resample(&native, rate, 16_000)?;
    Ok((out16, native, rate, out.metrics))
}

/// The clips of the run: the manifest's, or the run's single input.
fn clips_of(c: &CompareRun, manifest: Option<&CorpusManifest>, dir: &Path) -> Vec<ClipIn> {
    match manifest {
        Some(m) => m
            .clips
            .iter()
            .map(|k| {
                let input = dir.join(&k.input);
                let sha = k.sha256.trim().to_ascii_lowercase();
                ClipIn {
                    id: k.id.clone(),
                    // The manifest's hash (checked against the file later), else the file's own.
                    sha256: if sha.is_empty() { file_sha256(&input).unwrap_or_default() } else { sha },
                    source: ClipSource::File(input),
                    f0_truth: k.f0_truth.as_ref().map(|p| dir.join(p)),
                    references: k.references.iter().map(|p| dir.join(p)).collect(),
                }
            })
            .collect(),
        None => vec![ClipIn {
            id: "input".into(),
            source: c.input.clone().map_or(ClipSource::SpecInput, ClipSource::File),
            f0_truth: None,
            references: Vec::new(),
            sha256: String::new(),
        }],
    }
}

/// The 16 kHz audio of a clip.
fn clip_audio16(src: &ClipSource, spec: &BenchSpec, prepared: Option<&Prepared>) -> Result<Vec<f32>, String> {
    match src {
        ClipSource::File(p) => clip_16k(p),
        ClipSource::SpecInput => {
            let s = &spec.stream;
            let ch = usize::from(s.input_channels);
            let dev = match prepared {
                Some(p) => p.input.clone(),
                None => render_input(&spec.input, s.input_rate, ch, spec.seed)?,
            };
            resample(&channel(&dev, ch, 0), s.input_rate, 16_000)
        }
    }
}

fn point_wav(pattern: &str, clip: &str, dir: &Path) -> PathBuf {
    let p = PathBuf::from(pattern.replace("{clip}", clip));
    if p.is_absolute() {
        p
    } else {
        dir.join(p)
    }
}

/// The sidecar of one live capture.
///
/// `voice_sha256` is the one the manifest's point ASSERTS for the system under test, and nothing
/// else: the meter only hears the app's output and cannot know which voice the app loaded, so
/// writing the corpus's own voice here would make the "another voice is refused" check pass by
/// construction. Without an assertion the field stays empty and the point is flagged "voice not
/// verified" (`voice_asserted` says which it was).
fn capture_sidecar(point: &PointSpec, input: &Path, input_sha256: &str, calibration_ms: f64, o: &PlayRecordOutcome) -> serde_json::Value {
    let asserted = point.voice_sha256.trim().to_ascii_lowercase();
    serde_json::json!({
        "schema": SIDECAR_SCHEMA,
        "system": point.system,
        "app_version": point.app_version,
        "input": input.display().to_string(),
        "input_sha256": input_sha256,
        "voice_sha256": asserted,
        "voice_asserted": !asserted.is_empty(),
        "config": point.config.clone().unwrap_or_default(),
        "latency_ms": o.latency_ms,
        "latency_kind": "e2e_loopback",
        "latency_note": "the response envelope against the clip's (about 1 ms); the burst meter (e2e_loopback run) is the latency reference",
        "capture": {
            "output_device": o.output_device,
            "input_device": o.input_device,
            "output_rate": o.output_rate,
            "input_rate": o.input_rate,
            "calibration_ms": calibration_ms,
            "timebase_jitter_ms": [o.timebase_jitter_ms.0, o.timebase_jitter_ms.1],
            "latency_correlation": o.latency_correlation,
        },
        "glitches": o.glitches.len(),
        "glitch_list": o.glitches,
        "dropouts": o.dropouts.len(),
        "dropout_list": o.dropouts,
        "notes": ["live capture by Sigil's play-and-record meter (compare corpus captures)"],
    })
}

/// Record the clips of every capture whose WAVs are missing (or all with `overwrite`), with the
/// play-and-record meter, and write each WAV with its sidecar. Plays audio on the named devices.
fn run_captures(m: &CorpusManifest, clips: &[ClipIn], dir: &Path, cancel: &AtomicBool, progress: &(dyn Fn(f64, &str) + Sync)) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    for cap in &m.captures {
        let Some(point) = m.points.iter().find(|p| p.name == cap.point) else {
            continue;
        };
        if !cap.record {
            out.push(serde_json::json!({"point": cap.point, "skipped": "record is false"}));
            continue;
        }
        for (i, clip) in clips.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return out;
            }
            let wav = point_wav(&point.wav, &clip.id, dir);
            if wav.is_file() && !cap.overwrite {
                continue;
            }
            let ClipSource::File(input) = &clip.source else {
                continue;
            };
            progress(i as f64 / clips.len().max(1) as f64 * 0.1, &format!("capturing {} through {}", clip.id, cap.point));
            let res = read_wav(input).and_then(|(x, sr, ch)| {
                let mono = to_mono(&x, ch);
                run_play_record(&cap.meter, &mono, sr, cancel, &|_, _| {}).map(|o| (o, file_sha256(input).unwrap_or_default()))
            });
            match res {
                Ok((o, sha)) => {
                    if let Some(parent) = wav.parent() {
                        let _ = std::fs::create_dir_all(parent);
                    }
                    let written = write_wav(&wav, &o.response, o.input_rate, 1);
                    let sidecar = capture_sidecar(point, input, &sha, cap.meter.calibration_ms, &o);
                    let mut side = wav.as_os_str().to_owned();
                    side.push(".json");
                    let side_written = std::fs::write(PathBuf::from(side), serde_json::to_string_pretty(&sidecar).unwrap_or_default());
                    out.push(serde_json::json!({
                        "point": cap.point, "clip": clip.id, "wav": wav.display().to_string(),
                        "written": written.is_ok() && side_written.is_ok(),
                        "latency_ms": o.latency_ms, "glitches": o.glitches.len(), "dropouts": o.dropouts.len(),
                        "error": written.err().or(side_written.err().map(|e| e.to_string())),
                    }));
                }
                Err(e) => out.push(serde_json::json!({"point": cap.point, "clip": clip.id, "error": e})),
            }
        }
    }
    out
}

/// Run a `compare` run (v1 and v2; see the module docs).
pub(super) fn run_compare(
    c: &CompareRun,
    spec: &BenchSpec,
    prepared: Option<&Prepared>,
    previous: &[RunReport],
    write_files: bool,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<RunOutcome, String> {
    let range = prepared.and_then(|p| p.engine.rvc.as_ref()).map_or(F0Range::Speech, |r| r.f0_range);
    let mut sc = Scorers::load(&c.scoring, range);
    let mut notes = sc.notes.clone();
    let mut outcome = RunOutcome::default();

    // ---- The corpus ----
    let (manifest, dir) = match &c.corpus {
        Some(p) => (Some(CorpusManifest::load(p)?), p.parent().map(Path::to_path_buf).unwrap_or_default()),
        None => (None, PathBuf::new()),
    };
    if manifest.is_some() && !c.candidates.is_empty() {
        notes.push("compare.candidates are ignored with a corpus (its points are the candidates)".into());
    }
    let transpose = match manifest.as_ref().and_then(|m| m.transpose) {
        Some(t) if (t - c.semitones).abs() > 1e-6 && c.semitones != 0.0 => {
            return Err(format!("the corpus says transpose {t}, the run's semitones {}", c.semitones));
        }
        Some(t) => t,
        None => c.semitones,
    };
    let min_clips = manifest.as_ref().map_or(default_min_clips(), |m| m.min_clips);
    let margins = manifest.as_ref().map(|m| m.margins).unwrap_or_default();
    let loudness_lufs = manifest.as_ref().map_or(default_loudness_lufs(), |m| m.loudness_lufs);
    let analyzer_names = manifest.as_ref().map_or_else(default_analyzers, |m| m.analyzers.clone());
    let clips_in = clips_of(c, manifest.as_ref(), &dir);
    let voice_sha = manifest
        .as_ref()
        .map(|m| m.voice.sha256.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .or_else(|| slot_voice_sha(&spec.engine))
        .unwrap_or_default();
    let adapters = gpu_mem::adapters();
    let manifest_gpu = manifest.as_ref().map_or(String::new(), |m| m.sigil.gpu.trim().to_string());
    let (mut gpu, gpu_note) = run_gpu(&manifest_gpu, &adapters);
    notes.extend(gpu_note);
    let mut gpu_from_engine = false;

    // ---- Live captures (plays audio on the named devices) ----
    let captures = match &manifest {
        Some(m) if !m.captures.is_empty() => run_captures(m, &clips_in, &dir, cancel, progress),
        _ => Vec::new(),
    };
    if cancel.load(Ordering::Relaxed) {
        return Err("cancelled".into());
    }

    // ---- Points and renditions ----
    let mut points: Vec<Point> = Vec::new();
    let mut renditions: Vec<Vec<Rendition>> = Vec::new();
    let keep_native = spec.output.write_audio && write_files;
    let builds = sigil_builds(c, spec, manifest.as_ref(), transpose, &mut notes);
    let sigil_voice = manifest
        .as_ref()
        .map(|m| m.sigil.voice_sha256.trim().to_ascii_lowercase())
        .filter(|s| !s.is_empty())
        .or_else(|| slot_voice_sha(&spec.engine))
        .unwrap_or_default();
    let priority = WorkerPriority::from_str_lossy(&spec.stream.worker_priority);
    let n_builds = builds.len().max(1) as f64;
    for (bi, b) in builds.into_iter().enumerate() {
        let mut point = b.point;
        let mut rends: Vec<Rendition> = clips_in.iter().map(|_| Rendition::default()).collect();
        let base = 0.1 + 0.5 * bi as f64 / n_builds;
        progress(base, &format!("Sigil {}: loading", point.name));
        let built: Result<(Option<Engine>, AlgorithmicLatency), String> = match &b.engine {
            None => prepared.map(|p| (None, p.probe)).ok_or_else(|| "sigil.include needs the spec's engine".to_string()),
            Some(es) => build_engine(es, &b.stream, priority, spec.seed).and_then(|e| {
                if e.rvc.as_ref().is_some_and(|p| p.format.family().is_some()) {
                    return Err("voice families convert with the spec's engine only (configurations need an RVC voice)".into());
                }
                let probe = probe_algorithmic_latency(&b.stream, e.info.output_rate, e.info.min_context_samples, e.fixed_block(), e.info.model_delay_ms)?;
                Ok((Some(e), probe))
            }),
        };
        let (own, probe) = match built {
            Ok(x) => x,
            Err(e) => {
                if b.engine.is_none() && c.sigil.include {
                    return Err(e);
                }
                point.refusals.push(format!("not converted: {e}"));
                points.push(point);
                renditions.push(rends);
                continue;
            }
        };
        let engine: &Engine = match (&own, prepared) {
            (Some(e), _) => e,
            (None, Some(p)) => &p.engine,
            (None, None) => unreachable!("checked above"),
        };
        // The adapter this row really ran on (the first such row also names the run's GPU, which
        // the offline competitors' feasibility is judged against), not "the first adapter".
        let device_index = match b.engine.as_ref().unwrap_or(&spec.engine) {
            EngineSpec::Rvc(r) => r.gpu_device_index,
            EngineSpec::Stub(_) => 0,
        };
        let own_gpu = engine_gpu(&engine.info, device_index, &adapters);
        if let (true, false, Some(g)) = (manifest_gpu.is_empty(), gpu_from_engine, own_gpu.as_ref()) {
            gpu = g.clone();
            gpu_from_engine = true;
        }
        describe_sigil(&mut point, engine, &b.stream, &probe, &sigil_voice, own_gpu.as_deref().unwrap_or(""));
        let (mut p50, mut p95, mut p99) = (Vec::new(), Vec::new(), Vec::new());
        for (ci, clip) in clips_in.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            let step = 0.5 / n_builds / clips_in.len().max(1) as f64;
            progress(base + step * ci as f64, &format!("Sigil {}: converting {}", point.name, clip.id));
            let r = device_input(&clip.source, spec, &b.stream, prepared)
                .and_then(|x| convert(engine, probe, &b.stream, &x, &point.name, cancel, &|_, _| {}));
            match r {
                Ok((a16, native, rate, m)) => {
                    for (k, v) in [("call_ms_p50", &mut p50), ("call_ms_p95", &mut p95), ("call_ms_p99", &mut p99)] {
                        if let Some(x) = m.get(k) {
                            v.push(*x);
                        }
                    }
                    rends[ci].audio16 = Some(a16);
                    rends[ci].native = keep_native.then_some((native, rate));
                }
                Err(e) => {
                    // The v1 contract (one input, the spec's own engine as `sigil`): a failed
                    // conversion fails the run, as `run_offline(...)?` did.
                    if manifest.is_none() && b.engine.is_none() {
                        return Err(e);
                    }
                    notes.push(format!("Sigil {}: clip {}: conversion failed: {e}", point.name, clip.id));
                    rends[ci].notes.push(format!("conversion: {e}"));
                }
            }
        }
        // Per-block compute over the clips: the median p50, the worst p95 / p99 (conservative).
        let med = |v: &mut Vec<f64>| {
            v.sort_by(f64::total_cmp);
            v.get(v.len() / 2).copied()
        };
        let max = |v: &[f64]| v.iter().copied().fold(None, |a: Option<f64>, x| Some(a.map_or(x, |a| a.max(x))));
        point.call_ms = Some(CallMsStats {
            p50: med(&mut p50),
            p95: max(&p95),
            p99: max(&p99),
        });
        points.push(point);
        renditions.push(rends);
        drop(own);
    }
    // The other systems' points (the manifest's, or v1's inline candidates).
    match &manifest {
        Some(m) => {
            for ps in &m.points {
                let mut p = Point::new(&ps.name, &ps.system, ps.tier, ps.role, "manifest");
                p.app_version = ps.app_version.clone();
                p.config = ps.config.as_ref().and_then(|v| v.as_object()).cloned().unwrap_or_default();
                p.latency_ms = ps.latency_ms;
                p.latency_kind = ps.latency_kind.trim().to_string();
                if p.latency_ms.is_some() {
                    p.latency_source = "manifest".into();
                }
                p.voice_sha256 = ps.voice_sha256.trim().to_ascii_lowercase();
                p.call_ms = ps.call_ms;
                p.gpu = ps.gpu.clone();
                p.label = ps.notes.clone();
                let mut rends = Vec::new();
                let mut sides: Vec<(String, Sidecar)> = Vec::new();
                for clip in &clips_in {
                    let wav = point_wav(&ps.wav, &clip.id, &dir);
                    let mut r = Rendition {
                        wav: Some(wav.clone()),
                        ..Rendition::default()
                    };
                    match Sidecar::of(&wav) {
                        Ok(Some(side)) => {
                            if !clip.sha256.is_empty() && !side.input_sha256.is_empty() && side.input_sha256 != clip.sha256 {
                                r.refused = Some(format!("its sidecar's input sha256 {} is not the clip's {}", short(&side.input_sha256), short(&clip.sha256)));
                            }
                            if side.glitches.unwrap_or(0.0) > 0.0 || side.dropouts.unwrap_or(0.0) > 0.0 {
                                r.notes.push(format!(
                                    "capture: {} glitch(es), {} dropout(s) inside speech",
                                    side.glitches.unwrap_or(0.0),
                                    side.dropouts.unwrap_or(0.0)
                                ));
                            }
                            sides.push((clip.id.clone(), side));
                        }
                        Ok(None) => r.notes.push("no sidecar: the input it converted is not verified".into()),
                        Err(e) => r.refused = Some(format!("unreadable sidecar: {e}")),
                    }
                    if r.refused.is_none() {
                        match clip_16k(&wav) {
                            Ok(a) => {
                                r.native = if keep_native { read_wav(&wav).ok().map(|(x, sr, ch)| (to_mono(&x, ch), sr)) } else { None };
                                r.audio16 = Some(a);
                            }
                            Err(e) => r.notes.push(e),
                        }
                    }
                    rends.push(r);
                }
                if sides.is_empty() {
                    p.flags.push("no driver sidecar: metadata as the manifest gives it".into());
                } else {
                    p.merge_sidecars(&sides);
                    if sides.len() < clips_in.len() {
                        p.flags.push(format!("sidecars for {} of {} clips: the metadata comes from those", sides.len(), clips_in.len()));
                    }
                }
                let refused = rends.iter().filter(|r: &&Rendition| r.refused.is_some()).count();
                if refused > 0 {
                    p.flags.push(format!("{refused} rendition(s) refused (another input, or an unreadable sidecar)"));
                }
                points.push(p);
                renditions.push(rends);
            }
        }
        None => {
            for cand in &c.candidates {
                let mut p = Point::new(&cand.name, if cand.system.is_empty() { "unknown" } else { cand.system.as_str() }, Tier::Offline, PointRole::Competitor, "inline");
                p.app_version = cand.app_version.clone();
                p.config = cand.config.as_ref().and_then(|v| v.as_object()).cloned().unwrap_or_default();
                p.latency_ms = cand.latency_ms;
                p.latency_kind = cand.latency_kind.trim().to_string();
                if p.latency_ms.is_some() {
                    p.latency_source = "manifest".into();
                }
                if p.latency_kind == "e2e_loopback" {
                    p.tier = Tier::Live;
                }
                p.voice_sha256 = cand.voice_sha256.trim().to_ascii_lowercase();
                p.call_ms = cand.call_ms;
                p.label = cand.notes.clone();
                let mut r = Rendition {
                    wav: Some(cand.wav.clone()),
                    ..Rendition::default()
                };
                if let Ok(Some(side)) = Sidecar::of(&cand.wav) {
                    p.merge_sidecar(&side);
                }
                let input_sha = match &clips_in[0].source {
                    ClipSource::File(f) => file_sha256(f).ok(),
                    ClipSource::SpecInput => None,
                };
                let want = cand.input_sha256.trim().to_ascii_lowercase();
                match (&input_sha, want.is_empty()) {
                    (Some(have), false) if *have != want => r.refused = Some(format!("input sha256 {} is not the input's {}", short(&want), short(have))),
                    (_, true) => r.notes.push("input_sha256 not given: the input it converted is not verified".into()),
                    _ => {}
                }
                if r.refused.is_none() {
                    match clip_16k(&cand.wav) {
                        Ok(a) => {
                            r.native = if keep_native { read_wav(&cand.wav).ok().map(|(x, sr, ch)| (to_mono(&x, ch), sr)) } else { None };
                            r.audio16 = Some(a);
                        }
                        Err(e) => r.notes.push(e),
                    }
                }
                points.push(p);
                renditions.push(vec![r]);
            }
        }
    }
    for p in points.iter_mut() {
        p.read_config();
    }

    // ---- Clips ----
    progress(0.62, "analysing the clips");
    let ep = c.scoring.analysis_ep.trim().to_ascii_lowercase();
    let mut an = Analyzers::load(&analyzer_names, &ep, range);
    notes.extend(an.notes.iter().cloned());
    let mut spk = if c.scoring.speaker_model.trim().eq_ignore_ascii_case("off") {
        None
    } else {
        match SpeakerEmbedder::resolve_v2(&c.scoring.speaker_model, &ep) {
            Ok(Some(e)) => Some(e),
            Ok(None) => {
                notes.push("v2 speaker similarity: CAM++ exact (bench/campplus-exact) is not installed; not measured".into());
                None
            }
            Err(e) => {
                notes.push(format!("v2 speaker similarity: {e}"));
                None
            }
        }
    };
    let cfg = ScoreCfg {
        loudness_lufs,
        max_offset_16k: (c.max_offset_ms / 1000.0 * 16_000.0) as usize,
        consensus_min_agree: manifest.as_ref().and_then(|m| m.consensus_min_agree),
    };
    let shift = 2f32.powf(transpose / 12.0);
    // Stand-in references (the dry run before upstream renders exist): the spec's template voice
    // converting each whole clip offline, with noise seeds the streams do not use.
    let stand_in_stream = app_stream(&spec.stream, transpose);
    let mut stand_in: Option<Vec<(u64, OfflineVoice)>> = None;
    if let Some(si) = manifest.as_ref().and_then(|m| m.stand_in_references.as_ref()) {
        match prepared.and_then(|p| p.engine.rvc.as_ref()).filter(|p| p.format == SlotFormat::SigilRvc) {
            Some(plan) => {
                let mut voices = Vec::new();
                for &seed in &si.seeds {
                    let mut pl = plan.clone();
                    pl.noise_seed = seed;
                    match OfflineVoice::load(&pl, &RenderOpts::default()) {
                        Ok(v) => voices.push((seed, v)),
                        Err(e) => notes.push(format!("stand-in reference (seed {seed}): {e}")),
                    }
                }
                stand_in = Some(voices);
            }
            None => notes.push("stand_in_references need the spec's engine to be a Sigil template voice: none rendered".into()),
        }
    }
    let mut clips: Vec<ClipCtx> = Vec::new();
    for clip in &clips_in {
        let audio = clip_audio16(&clip.source, spec, prepared)?;
        let path = match &clip.source {
            ClipSource::File(p) => Some(p.clone()),
            ClipSource::SpecInput => None,
        };
        let mut refs: Vec<RefIn> = clip
            .references
            .iter()
            .map(|r| RefIn {
                name: r.display().to_string(),
                stand_in: false,
                audio: clip_16k(r),
            })
            .collect();
        if let Some(voices) = stand_in.as_mut() {
            for (seed, voice) in voices.iter_mut() {
                progress(0.62, &format!("rendering the stand-in reference of {} (seed {seed})", clip.id));
                refs.push(RefIn {
                    name: format!("stand-in seed {seed}"),
                    stand_in: true,
                    audio: stand_in_reference(voice, &stand_in_stream, &audio),
                });
            }
        }
        let mut ctx = clip_ctx(&clip.id, path.clone(), audio, clip.f0_truth.as_deref(), refs, shift, &mut an, &sc, spk.as_mut(), &cfg)?;
        if let Some(p) = &path {
            ctx.sha256 = file_sha256(p).unwrap_or_default();
            if !clip.sha256.is_empty() && clip.sha256 != ctx.sha256 {
                return Err(format!("clip '{}': {} has sha256 {}, the manifest says {}", clip.id, p.display(), short(&ctx.sha256), short(&clip.sha256)));
            }
        }
        clips.push(ctx);
    }
    // Target-speaker sets: the other clips' references (leave one clip out), else the run's
    // speaker reference clip.
    let speaker_reference = match (&c.scoring.speaker_reference, spk.as_mut()) {
        (Some(p), Some(s)) => clip_16k(p).ok().and_then(|x| s.embed_masked(&x, &speech_frame_mask(&x, SPEAKER_SPEECH_RANGE_DB)).ok()),
        _ => None,
    };
    // Upstream references first; the stand-ins (Sigil's own render, which favours Sigil's voice
    // character) only when no clip has an upstream one, and the provenance is recorded per clip.
    let embeddings = |stand_in: bool| -> Vec<(usize, Vec<f32>)> {
        clips
            .iter()
            .enumerate()
            .flat_map(|(i, k)| {
                let set = if stand_in { &k.stand_in_embeddings } else { &k.ref_embeddings };
                set.iter().map(move |e| (i, e.clone()))
            })
            .collect()
    };
    let (upstream, stand_ins) = (embeddings(false), embeddings(true));
    for (i, k) in clips.iter_mut().enumerate() {
        let others = |all: &[(usize, Vec<f32>)]| -> Vec<Vec<f32>> { all.iter().filter(|(j, _)| *j != i).map(|(_, e)| e.clone()).collect() };
        (k.target, k.target_source) = if let Some(t) = centroid(&others(&upstream)) {
            (Some(t), "upstream")
        } else if let Some(t) = centroid(&others(&stand_ins)) {
            (Some(t), "stand_in")
        } else if speaker_reference.is_some() {
            (speaker_reference.clone(), "speaker_reference")
        } else {
            (None, "none")
        };
    }

    // ---- Scoring ----
    let mut rows: Rows = points.iter().map(|_| clips.iter().map(|_| None).collect()).collect();
    let mut placements: BTreeMap<(usize, usize), Placement> = BTreeMap::new();
    let mut row_details = Vec::new();
    let total = (points.len() * clips.len()).max(1) as f64;
    for (pi, p) in points.iter().enumerate() {
        for (ci, clip) in clips.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err("cancelled".into());
            }
            progress(0.65 + 0.33 * (pi * clips.len() + ci) as f64 / total, &format!("scoring {} on {}", p.name, clip.id));
            let r = &renditions[pi][ci];
            let mut rn = r.notes.clone();
            let scored = match (&r.refused, &r.audio16) {
                (Some(why), _) => {
                    rn.push(format!("refused: {why}"));
                    None
                }
                (None, Some(a)) => score_rendition(clip, a, p.block_ms, &mut an, &mut sc, spk.as_mut(), &cfg, &mut rn),
                (None, None) => {
                    if r.wav.as_ref().is_some_and(|w| !w.is_file()) {
                        rn.push("missing WAV".into());
                    }
                    None
                }
            };
            if let Some((s, place)) = scored {
                placements.insert((pi, ci), place);
                row_details.push(serde_json::json!({"point": p.name, "clip": clip.id, "ok": true, "wav": r.wav, "scores": s, "notes": rn}));
                rows[pi][ci] = Some(s);
            } else {
                row_details.push(serde_json::json!({"point": p.name, "clip": clip.id, "ok": false, "wav": r.wav, "notes": rn}));
            }
        }
    }

    // ---- Completeness: a row that misses clips decides nothing ----
    let dnsmos_loaded = sc.dnsmos.is_some();
    for (pi, p) in points.iter_mut().enumerate() {
        p.unscored_clips = unscored_cells(&rows[pi], &clips, dnsmos_loaded);
        if !p.unscored_clips.is_empty() {
            p.flags.push(format!(
                "{} of {} clips not fully scored ({}): the row is incomplete and decides nothing",
                p.unscored_clips.len(),
                clips.len(),
                list_some(&p.unscored_clips, 6)
            ));
        }
    }

    // ---- Latencies measured from the renditions, then the fairness checks ----
    for (pi, p) in points.iter_mut().enumerate() {
        let mut offs: Vec<f64> = rows[pi].iter().filter_map(|r| r.as_ref().and_then(|r| r.offset_ms)).collect();
        offs.sort_by(f64::total_cmp);
        let Some(&off) = offs.get(offs.len() / 2) else { continue };
        // The offline loop's I/O block: Sigil's device block (its engine block accumulates inside
        // the harness, which the offset already holds), a driver's block.
        let io_block = p.config.get("device_block_ms").and_then(|v| v.as_f64()).or(p.block_ms);
        let measured = match p.tier {
            Tier::Live => Some(off),
            Tier::Offline => io_block.map(|b| off + b),
        };
        p.latency_measured_ms = measured;
        let Some(m) = measured else {
            p.flags.push("no block size: the offline latency cannot be measured (offset + block)".into());
            continue;
        };
        let stated = p.latency_ms;
        let keep_stated = p.source == "in_process" || (p.tier == Tier::Live && p.latency_kind == "e2e_loopback" && stated.is_some());
        if keep_stated {
            if let Some(st) = stated.filter(|st| (st - m).abs() > 10.0) {
                p.flags.push(format!("latency: {st:.1} ms ({}) but {m:.1} ms measured from the renditions", p.latency_source));
            }
        } else {
            if let Some(st) = stated.filter(|st| (st - m).abs() > 10.0) {
                p.flags.push(format!("latency: {st:.1} ms stated ({}), {m:.1} ms measured from the renditions (used)", p.latency_kind));
            }
            p.latency_ms = Some(m);
            // A live offset nobody calibrated (no e2e_loopback latency stated, as a play_record.py
            // capture is) is not the same quantity as Sigil's calibrated play-and-record delay.
            p.latency_kind = if p.tier == Tier::Live { UNCALIBRATED } else { "algorithmic" }.into();
            if p.tier == Tier::Live {
                p.flags.push("live latency is the offset measured from the recording, uncalibrated: not compared with other systems' latencies (capture it with a corpus `captures` entry, or state its calibrated e2e_loopback latency)".into());
            }
            p.latency_source = "measured".into();
        }
    }
    check_points(&mut points, f64::from(transpose), &voice_sha, &gpu);

    // ---- v1 rows (one input: continuity) ----
    if manifest.is_none() {
        let mut v1_rows = Vec::new();
        let ia = InputAnalysis::new(clips[0].input16.clone(), c.semitones, &mut sc)?;
        let reference = match &c.scoring.speaker_reference {
            Some(p) => Some(clip_16k(p)?),
            None => None,
        };
        let against = Against {
            input: &ia,
            reference: reference.as_deref(),
            max_offset_s: c.max_offset_ms / 1000.0,
        };
        let latency = sigil_latency_v1(c, previous, &mut notes);
        if let Some(si) = points.iter().position(|p| p.name == "sigil" && p.source == "in_process") {
            if let Some(a) = &renditions[si][0].audio16 {
                let mut row = score_candidate("sigil", Path::new(""), a, latency, &against, &mut sc);
                row.wav = PathBuf::from(format!("{}.output.wav", c.name));
                if (spec.stream.pitch_semitones - c.semitones).abs() > 1e-6 {
                    row.notes.push(format!(
                        "the spec's stream transposes {} semitones, the candidates {}: unfair (v2 refuses this row)",
                        spec.stream.pitch_semitones, c.semitones
                    ));
                }
                v1_rows.push(row);
            }
            if spec.output.write_audio && write_files {
                if let Some((x, rate)) = renditions[si][0].native.clone() {
                    outcome.audio = Some((x, rate, 1));
                }
            }
        }
        for (pi, cand) in c.candidates.iter().enumerate() {
            let idx = points.iter().position(|p| p.name == cand.name && p.source == "inline").unwrap_or(pi);
            let row = match renditions.get(idx).and_then(|r| r[0].audio16.as_ref()) {
                Some(x) => score_candidate(&cand.name, &cand.wav, x, cand.latency_ms, &against, &mut sc),
                None => CompareRow {
                    name: cand.name.clone(),
                    wav: cand.wav.clone(),
                    latency_ms: cand.latency_ms,
                    notes: renditions.get(idx).map(|r| r[0].notes.clone()).unwrap_or_default(),
                    ..CompareRow::default()
                },
            };
            v1_rows.push(row);
        }
        outcome.metrics = compare_metrics(&v1_rows);
        outcome.details = serde_json::json!({
            "rows": v1_rows,
            "semitones": c.semitones,
            "scorers": sc.describe(),
            "input_speech_s": (ia.region.1 - ia.region.0) as f64 / 16_000.0,
        });
    } else {
        outcome.details = serde_json::json!({"rows": [], "semitones": transpose, "scorers": sc.describe()});
    }

    // ---- Pairings, verdicts, ablations, sizing ----
    let analyzers = an.names();
    let mut pairings = Vec::new();
    for (s, ps) in points.iter().enumerate().filter(|(_, p)| p.is_sigil()) {
        for (k, pk) in points.iter().enumerate().filter(|(_, p)| p.counted()) {
            if ps.tier == pk.tier {
                pairings.push(pairing(&points, &rows, &clips, s, k, &analyzers));
            }
        }
    }
    let verdicts: Vec<CompetitorVerdict> = points
        .iter()
        .enumerate()
        .filter(|(_, p)| p.counted())
        .map(|(k, pk)| {
            let sig: Vec<usize> = points.iter().enumerate().filter(|(_, p)| p.is_sigil() && p.tier == pk.tier).map(|(i, _)| i).collect();
            verdict_for(&points, &pairings, k, &sig, &margins, min_clips)
        })
        .collect();
    let mut t6 = t6_decision(&verdicts);
    t6.caveats = points
        .iter()
        .filter(|p| p.counted() && p.refusals.is_empty())
        .flat_map(|p| p.flags.iter().filter(|f| f.starts_with("operating point:")).map(move |f| format!("{}: {f}", p.name)))
        .collect();
    let mut ablations = Vec::new();
    for (a, pa) in points.iter().enumerate().filter(|(_, p)| p.role == PointRole::Ablation) {
        let Some(vs) = pa.versus.as_deref() else { continue };
        let Some(s) = points.iter().position(|p| p.name == vs) else {
            notes.push(format!("ablation {}: no Sigil row '{vs}'", pa.name));
            continue;
        };
        let p = pairing(&points, &rows, &clips, s, a, &analyzers);
        ablations.push(AblationResult {
            ablation: pa.name.clone(),
            versus: vs.to_string(),
            label: pa.label.clone(),
            clips_needed_dnsmos: p.dnsmos.as_ref().map(|d| clips_needed(d.sd, margins.dnsmos_superiority)),
            clips_needed_dnsmos_tiled: p.dnsmos_tiled.as_ref().map(|d| clips_needed(d.sd, margins.dnsmos_superiority)),
            pairing: p,
        });
    }

    // ---- Metrics ----
    let m = &mut outcome.metrics;
    let count = |f: &dyn Fn(&Point) -> bool| points.iter().filter(|p| f(p)).count() as f64;
    m.insert("corpus_clips".into(), clips.len() as f64);
    m.insert("corpus_clips_dnsmos".into(), clips.iter().filter(|k| k.dnsmos_ok()).count() as f64);
    m.insert("points_sigil".into(), count(&|p| p.is_sigil()));
    m.insert("points_competitor".into(), count(&|p| p.counted()));
    m.insert("points_ablation".into(), count(&|p| p.role == PointRole::Ablation));
    m.insert("points_refused".into(), count(&|p| !p.refusals.is_empty()));
    m.insert("points_flagged".into(), count(&|p| !p.flags.is_empty()));
    m.insert("points_excluded".into(), count(&|p| p.counted() && p.feasible == Some(false)));
    m.insert("renditions_refused".into(), renditions.iter().flatten().filter(|r| r.refused.is_some()).count() as f64);
    m.insert("points_incomplete".into(), count(&|p| !p.unscored_clips.is_empty()));
    m.insert("clips_unscored".into(), points.iter().map(|p| p.unscored_clips.len()).sum::<usize>() as f64);
    let statuses: Vec<Status> = verdicts.iter().flat_map(|v| v.rows().map(|r| r.status)).collect();
    for (k, st) in [
        ("t6_rows_pass", Status::Pass),
        ("t6_rows_fail", Status::Fail),
        ("t6_rows_not_shown", Status::NotShown),
        ("t6_rows_undecided", Status::Undecided),
        ("t6_rows_refused", Status::Refused),
        ("t6_rows_excluded", Status::Excluded),
    ] {
        m.insert(k.into(), statuses.iter().filter(|&&s| s == st).count() as f64);
    }
    m.insert("t6_pass".into(), if t6.status == "pass" { 1.0 } else { 0.0 });
    m.insert("t6_undecided".into(), if t6.status == "undecided" { 1.0 } else { 0.0 });
    let sds: Vec<f64> = pairings
        .iter()
        .chain(ablations.iter().map(|a| &a.pairing))
        .filter_map(|p| p.dnsmos.as_ref().map(|d| d.sd))
        .collect();
    let sd_max = sds.iter().copied().fold(None, |a: Option<f64>, x| Some(a.map_or(x, |a| a.max(x))));
    if let Some(sd) = sd_max {
        let (n50, n80) = clips_needed(sd, margins.dnsmos_superiority);
        m.insert("dnsmos_paired_sd".into(), sd);
        m.insert("clips_needed_dnsmos_50".into(), n50 as f64);
        m.insert("clips_needed_dnsmos_80".into(), n80 as f64);
    }
    for a in &ablations {
        let key = |s: &str| format!("ablation_{}_{s}", a.ablation);
        let mut put = |name: &str, d: &Option<PairedDiff>| {
            if let Some(d) = d {
                m.insert(key(&format!("{name}_diff_mean")), d.mean);
                m.insert(key(&format!("{name}_diff_sd")), d.sd);
                m.insert(key(&format!("{name}_diff_lo")), d.t_lo);
                m.insert(key(&format!("{name}_diff_hi")), d.t_hi);
                m.insert(key(&format!("{name}_clips")), d.n as f64);
            }
        };
        put("dnsmos", &a.pairing.dnsmos);
        put("dnsmos_tiled", &a.pairing.dnsmos_tiled);
        put("mcd_ref", &a.pairing.mcd);
        put("mcd_stand_in", &a.pairing.mcd_stand_in);
        put("artifacts", &a.pairing.artifacts);
        put("seam_clicks", &a.pairing.seam_clicks);
        put("onset_level", &a.pairing.onset_level);
        let first_gpe = a.pairing.gpe.values().next().cloned();
        put("gpe", &first_gpe);
    }

    // ---- Listening pairs (aligned, trimmed, loudness-matched) ----
    let listen_sigil = manifest
        .as_ref()
        .map(|m| m.listening.sigil.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| points.iter().find(|p| p.is_sigil()).map(|p| p.name.clone()));
    let mut trials = Vec::new();
    let mut skipped = Vec::new();
    let mut written = BTreeSet::new();
    if let Some(sname) = &listen_sigil {
        if let Some(listen) = points.iter().position(|p| &p.name == sname) {
            for (k, pk) in points.iter().enumerate().filter(|(_, p)| p.counted() || p.role == PointRole::Ablation) {
                // An ablation is heard against the row it ablates (the same shape).
                let s = match pk.versus.as_deref().filter(|_| pk.role == PointRole::Ablation) {
                    Some(vs) => match points.iter().position(|p| p.name == vs) {
                        Some(i) => i,
                        None => continue,
                    },
                    None => listen,
                };
                // The blind A/B result stands in for the quality row when the corpus is too small:
                // it hears only the pairs the verdicts would count, never a refused (another
                // transpose, voice or input) or real-time-infeasible point.
                // (An ablation is heard for how it sounds at its shape, whatever its speed.)
                let real_time = pk.role != PointRole::Ablation;
                if let Some(why) = unfair_to_hear(&points[s], real_time).map(|w| format!("Sigil row {}: {w}", points[s].name)).or_else(|| unfair_to_hear(pk, real_time)) {
                    skipped.push(serde_json::json!({"point": pk.name, "sigil": points[s].name, "why": why}));
                    continue;
                }
                for (ci, clip) in clips.iter().enumerate() {
                    let (Some(ps), Some(pkk)) = (placements.get(&(s, ci)), placements.get(&(k, ci))) else {
                        continue;
                    };
                    let file = |i: usize| format!("{}.pair.{}.{}.wav", c.name, clip.id, points[i].name);
                    trials.push(serde_json::json!({"clip": clip.id, "a": file(s), "system_a": points[s].name, "b": file(k), "system_b": pk.name}));
                    for (i, pl) in [(s, ps), (k, pkk)] {
                        if !written.insert((i, ci)) {
                            continue;
                        }
                        let Some((x, rate)) = renditions[i][ci].native.as_ref() else { continue };
                        // The scored region at the rendition's own rate, at the listening level.
                        let to_native = |v: isize| (v as f64 * f64::from(*rate) / 16_000.0).round() as isize;
                        let lo = to_native(pl.a as isize + pl.lag).max(0) as usize;
                        let hi = (to_native(pl.b as isize + pl.lag).max(0) as usize).min(x.len());
                        if hi > lo {
                            let seg = apply_gain(&x[lo..hi], pl.gain_db.unwrap_or(0.0));
                            outcome.extra_audio.push((format!("pair.{}.{}", clip.id, points[i].name), seg, *rate, 1));
                        }
                    }
                }
            }
        }
    }
    let make_test = serde_json::json!({
        "title": format!("Voice changer A/B: {}", manifest.as_ref().map_or(c.name.as_str(), |m| m.name.as_str())),
        "question": "Which one sounds more natural and closer to a real voice?",
        "trials": trials,
        "skipped": skipped,
        "$comment": "Aligned, trimmed, loudness-matched renditions written next to report.json when the spec writes audio (output.write_audio); `python scripts/vc-bench/compare_export.py report.json` writes this as <run>.make_test.json for tools/listening-test/make_test.py.",
    });

    let dump_f0 = manifest.as_ref().is_some_and(|m| m.dump_f0);
    let clip_details: Vec<serde_json::Value> = clips
        .iter()
        .map(|k| {
            let f0 = dump_f0.then(|| {
                analyzers
                    .iter()
                    .zip(&k.contours)
                    .map(|(a, c)| (a.to_string(), serde_json::json!(c.iter().map(|v| f64::from(*v / shift)).collect::<Vec<f64>>())))
                    .collect::<serde_json::Map<String, serde_json::Value>>()
            });
            let sp: Vec<usize> = (0..k.speech.len()).filter(|&i| k.speech[i]).collect();
            let voiced_share: serde_json::Map<String, serde_json::Value> = analyzers
                .iter()
                .zip(&k.contours)
                .filter(|(_, c)| !c.is_empty())
                .map(|(a, c)| {
                    let v = sp.iter().filter(|&&i| c.get(i).is_some_and(|x| *x > 0.0)).count() as f64 / sp.len().max(1) as f64;
                    (a.to_string(), serde_json::json!(v))
                })
                .collect();
            serde_json::json!({
                "f0": f0,
                "analyzer_voiced_share": voiced_share,
                "id": k.id, "path": k.path, "sha256": k.sha256, "seconds": k.input16.len() as f64 / 16_000.0,
                "dnsmos": k.dnsmos_ok(), "speech_region_s": [k.region.0 as f64 / 16_000.0, k.region.1 as f64 / 16_000.0],
                "truth": k.truth.as_ref().map(|t| t.0), "truth_coverage": k.truth_coverage,
                "references": k.ref_files, "references_used": k.refs.iter().filter(|r| !r.stand_in).count(),
                "stand_in_references": k.stand_in_files, "stand_in_used": k.refs.iter().filter(|r| r.stand_in).count(),
                "reference_floor_db": k.ref_floor.map(|f| f.0), "reference_floor_stft_db": k.ref_floor.map(|f| f.1),
                "reference_floor_stand_in": k.ref_floor.is_some() && k.ref_floor_stand_in,
                "target_set": k.target.is_some(), "target_source": k.target_source, "notes": k.notes,
            })
        })
        .collect();
    let sizing = sd_max.map(|sd| {
        let (n50, n80) = clips_needed(sd, margins.dnsmos_superiority);
        let have = clips.iter().filter(|k| k.dnsmos_ok()).count();
        serde_json::json!({
            "dnsmos_paired_sd": sd, "clips_needed_50": n50, "clips_needed_80": n80, "clips_with_dnsmos": have,
            "enough": have >= n50,
            "note": if have >= n50 { "the corpus can carry the +0.05 DNSMOS row at 50 % power" } else { "the corpus is smaller than the +0.05 DNSMOS row needs: the blind A/B row carries the quality claim" },
        })
    });
    let v2 = serde_json::json!({
        "schema": "sigil-compare-v2/1",
        "corpus": manifest.as_ref().map(|m| serde_json::json!({"name": m.name, "manifest": c.corpus, "description": m.description})),
        "transpose": transpose, "voice_sha256": voice_sha, "gpu": gpu,
        "margins": margins, "min_clips": min_clips, "loudness_lufs": loudness_lufs,
        "analyzers": analyzers,
        "clips": clip_details,
        "points": points,
        "rows": row_details,
        "pairings": pairings,
        "verdicts": verdicts,
        "t6": t6,
        "ablations": ablations,
        "sizing": sizing,
        "pareto": pareto(&points, &rows),
        "make_test": make_test,
        "captures": captures,
        "notes": notes,
    });
    if let Some(o) = outcome.details.as_object_mut() {
        o.insert("notes".into(), serde_json::json!(notes));
        o.insert("v2".into(), v2);
    }
    progress(1.0, "done");
    Ok(outcome)
}

#[cfg(test)]
mod tests {
    use super::super::report::RunStatus;
    use super::super::signals::{synth, write_wav, SplitMix64};
    use super::super::spec::SignalKind;
    use super::super::{run_spec, BenchContext, Progress};
    use super::*;

    fn temp(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("sigil-bench-compare-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Two synthetic "competitors": one delayed, pitch-exact copy, and one with octave errors, pops
    /// and noise. The metrics rank them as expected and the margins come out of Sigil's row.
    #[test]
    fn candidates_are_aligned_and_ranked() {
        let dir = temp("rank");
        let input = synth(SignalKind::Voice, 16_000, 6.0, 0.4, 21);
        let input_path = dir.join("in.wav");
        write_wav(&input_path, &input, 16_000, 1).unwrap();
        // "good": the input delayed by 180 ms at 48 kHz.
        let mut good = vec![0.0f32; 8_640];
        good.extend(synth(SignalKind::Voice, 48_000, 6.0, 0.4, 21));
        let good_path = dir.join("good.wav");
        write_wav(&good_path, &good, 48_000, 1).unwrap();
        // "bad": another voice (other pitch) in the second half, noisy, with pops, delayed by
        // 300 ms.
        let mut rng = SplitMix64::new(2);
        let mut bad = vec![0.0f32; 4_800];
        let half = input.len() / 2;
        let other = synth(SignalKind::Voice, 16_000, 6.0, 0.4, 99);
        bad.extend(input[..half].iter().copied());
        bad.extend(other[half..input.len()].iter().copied());
        for (i, v) in bad.iter_mut().enumerate() {
            *v += (0.02 * rng.range(-1.0, 1.0)) as f32;
            if i % 3_203 == 1_000 {
                *v += 0.3;
            }
        }
        let bad_path = dir.join("bad.wav");
        write_wav(&bad_path, &bad, 16_000, 1).unwrap();
        let rows = compare_wavs(&input_path, &[("good".into(), good_path.clone()), ("bad".into(), bad_path.clone())]).unwrap();
        let (g, b) = (&rows[0], &rows[1]);
        assert!(g.ok && b.ok, "{rows:?}");
        assert!((g.quality.offset_ms.unwrap() - 180.0).abs() <= 10.0, "{:?}", g.quality);
        assert!((b.quality.offset_ms.unwrap() - 300.0).abs() <= 10.0, "{:?}", b.quality);
        assert!(g.quality.gross_pitch_error.unwrap() < 0.02, "{:?}", g.quality);
        assert!(b.quality.gross_pitch_error.unwrap() > 0.2, "{:?}", b.quality);
        assert!(g.artifact_events_per_min.unwrap() < 5.0, "{g:?}");
        assert!(b.artifact_events_per_min.unwrap() > 50.0, "{b:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn margins_come_from_sigil_and_the_best_competitor() {
        let row = |name: &str, ovrl: f64, gpe: f64, lat: Option<f64>| CompareRow {
            name: name.into(),
            ok: true,
            latency_ms: lat,
            quality: QualityMetrics {
                dnsmos_ovrl: Some(ovrl),
                gross_pitch_error: Some(gpe),
                ..QualityMetrics::default()
            },
            ..CompareRow::default()
        };
        let m = compare_metrics(&[
            row("sigil", 3.2, 0.01, Some(60.0)),
            row("a", 3.1, 0.02, Some(120.0)),
            row("b", 3.0, 0.015, Some(100.0)),
        ]);
        assert!((m["dnsmos_ovrl_margin"] - 0.1).abs() < 1e-9);
        assert!((m["gross_pitch_error_margin"] - 0.005).abs() < 1e-9);
        assert!((m["latency_ratio"] - 0.6).abs() < 1e-9);
        assert_eq!(m["candidates_scored"], 2.0);
        let alone = compare_metrics(&[row("sigil", 3.2, 0.01, None)]);
        assert!(!alone.contains_key("dnsmos_ovrl_margin") && !alone.contains_key("latency_ratio"));
    }

    /// A compare run in a spec: Sigil's own row (the stub, identity) against one candidate, with
    /// Sigil's latency taken from an earlier run.
    #[test]
    fn compare_runs_in_a_spec() {
        let dir = temp("spec");
        let other: Vec<f32> = synth(SignalKind::Voice, 48_000, 4.0, 0.25, 3);
        let wav = dir.join("other.wav");
        write_wav(&wav, &other, 48_000, 1).unwrap();
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "cmp", "seed": 3,
            "engine": {"kind": "stub"},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 4, "level_dbfs": -12},
            "runs": [
                {"mode": "paced", "name": "lat", "seconds": 2, "clock": "simulated"},
                {"mode": "compare", "name": "h2h",
                 "sigil": {"latency_from": "lat", "latency_metric": "latency_ms_p50"},
                 "candidates": [{"name": "other", "wav": wav, "latency_ms": 200.0}],
                 "scoring": {"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio"},
                 "thresholds": {"gross_pitch_error_margin": {"min": -0.05}, "latency_ratio": {"max": 0.7}}}
            ]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = |_: &Progress| {};
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        let run = &r.runs[1];
        assert_eq!(run.status, RunStatus::Passed, "{:?} {:?} {}", run.error, run.thresholds, run.details);
        assert!(run.metrics["sigil_latency_ms"] > 0.0);
        assert_eq!(run.details["rows"].as_array().unwrap().len(), 2);
        // v2 rides along: one clip, the stub as Sigil's row, the candidate as a competitor point.
        let v2 = &run.details["v2"];
        assert_eq!(v2["points"].as_array().unwrap().len(), 2, "{v2}");
        assert_eq!(run.metrics["corpus_clips"], 1.0);
        assert_eq!(run.metrics["points_competitor"], 1.0);
        // One clip and no feasibility data: never a T6 pass.
        assert_eq!(run.metrics["t6_pass"], 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- v2: loudness ----

    #[test]
    fn k_weighting_is_bs1770s_table_at_48k() {
        let [shelf, rlb] = k_weighting(48_000);
        let want_shelf = [1.535_124_859_586_97, -2.691_696_189_406_38, 1.198_392_810_852_85, -1.690_659_293_182_41, 0.732_480_774_215_85];
        let want_rlb = [1.0, -2.0, 1.0, -1.990_047_454_833_98, 0.990_072_250_366_21];
        for (got, want) in shelf.iter().zip(&want_shelf).chain(rlb.iter().zip(&want_rlb)) {
            assert!((got - want).abs() < 1e-8, "{got} vs {want}");
        }
    }

    #[test]
    fn loudness_follows_bs1770() {
        // A 997 Hz sine at full scale reads -3.01 LUFS (BS.1770-4's calibration), at any rate.
        for rate in [16_000u32, 44_100, 48_000] {
            for amp in [1.0f32, 0.1] {
                let x: Vec<f32> = (0..rate as usize * 3)
                    .map(|i| amp * (2.0 * std::f32::consts::PI * 997.0 * i as f32 / rate as f32).sin())
                    .collect();
                let l = integrated_loudness(&x, rate).unwrap();
                let want = -3.01 + 20.0 * f64::from(amp).log10();
                assert!((l - want).abs() < 0.05, "{rate} Hz, amp {amp}: {l} LUFS, want {want}");
            }
        }
        // The relative gate ignores pauses: a tone with a long silence reads as the tone.
        let rate = 16_000u32;
        let mut x: Vec<f32> = (0..rate as usize * 2).map(|i| 0.1 * (2.0 * std::f32::consts::PI * 997.0 * i as f32 / rate as f32).sin()).collect();
        x.extend(vec![0.0; rate as usize * 6]);
        // Only the 3 blocks straddling the tone's end lower it (by 0.34 LU); an ungated mean
        // would read 6 dB lower (-29 LUFS).
        let l = integrated_loudness(&x, rate).unwrap();
        assert!((l + 23.35).abs() < 0.05, "{l}");
        assert!(integrated_loudness(&[0.0; 16_000], rate).is_none(), "silence has no loudness");
        assert!(integrated_loudness(&[0.1; 1_000], rate).is_none(), "under one gating block");
        let (l, g) = loudness_gain_db(&x, rate, -26.0).unwrap();
        assert!((l + g + 26.0).abs() < 1e-9);
    }

    // ---- v2: statistics and verdicts on synthetic tables ----

    #[test]
    fn paired_differences_and_sizing() {
        let pairs: Vec<(String, f64, f64)> = (0..10).map(|i| (format!("c{i}"), 3.0 + 0.1 * f64::from(i % 3), 3.0)).collect();
        let d = PairedDiff::of(&pairs).unwrap();
        assert_eq!(d.n, 10);
        assert!((d.mean - 0.09).abs() < 1e-9, "{}", d.mean);
        assert!(d.t_lo < d.mean && d.mean < d.t_hi && d.w_lo.is_some());
        assert!(PairedDiff::of(&pairs[..1]).is_none(), "one clip has no interval");
        // The plan's sizing: about 70 clips at SD 0.21 for 50 % power, about 140 for 80 %.
        let (n50, n80) = clips_needed(0.21, 0.05);
        assert!((66..=70).contains(&n50) && (136..=142).contains(&n80), "{n50} {n80}");
    }

    /// A synthetic table: `clips` clips; Sigil and competitor DNSMOS / GPE / MCD per clip.
    struct Table {
        points: Vec<Point>,
        rows: Rows,
        clips: Vec<ClipCtx>,
    }

    fn ctx(id: &str) -> ClipCtx {
        ClipCtx {
            id: id.into(),
            path: None,
            sha256: String::new(),
            input16: vec![0.0; DNSMOS_LEN],
            speech: Vec::new(),
            region: (0, 0),
            events: Vec::new(),
            contours: Vec::new(),
            truth: None,
            truth_coverage: None,
            refs: Vec::new(),
            ref_files: 0,
            stand_in_files: 0,
            ref_floor: None,
            ref_floor_stand_in: false,
            ref_embeddings: Vec::new(),
            stand_in_embeddings: Vec::new(),
            target: None,
            target_source: "none",
            notes: Vec::new(),
        }
    }

    fn point(name: &str, role: PointRole, latency: f64, kind: &str) -> Point {
        let mut p = Point::new(name, if role == PointRole::Sigil { "sigil" } else { "other" }, Tier::Offline, role, "inline");
        p.latency_ms = Some(latency);
        p.latency_kind = kind.into();
        p.feasible = Some(true);
        p
    }

    /// `f(clip, point) -> (dnsmos, gpe under rmvpe, gpe under dio, mcd)`.
    fn table(n: usize, points: Vec<Point>, f: impl Fn(usize, usize) -> (f64, f64, f64, f64)) -> Table {
        let clips: Vec<ClipCtx> = (0..n).map(|i| ctx(&format!("c{i}"))).collect();
        let rows = (0..points.len())
            .map(|p| {
                (0..n)
                    .map(|c| {
                        let (dn, g1, g2, mcd) = f(c, p);
                        let score = |g: f64| F0Score {
                            fpe_cents: Some(10.0),
                            gpe: g,
                            vuv: 0.02,
                            voiced_frames: 500,
                            frames: 700,
                        };
                        let mut r = RowScores {
                            dnsmos_ovrl: Some(dn),
                            mcd_ref_db: Some(mcd),
                            ..RowScores::default()
                        };
                        r.f0_truth.insert("rmvpe".into(), score(g1));
                        r.f0_truth.insert("dio".into(), score(g2));
                        Some(r)
                    })
                    .collect()
            })
            .collect();
        Table { points, rows, clips }
    }

    fn decide(t: &Table, m: &Margins, min_clips: usize) -> (Vec<CompetitorVerdict>, T6Decision) {
        let an: Vec<&'static str> = vec!["dio", "rmvpe"];
        let mut pairings = Vec::new();
        for s in (0..t.points.len()).filter(|&i| t.points[i].is_sigil()) {
            for k in (0..t.points.len()).filter(|&i| t.points[i].counted()) {
                pairings.push(pairing(&t.points, &t.rows, &t.clips, s, k, &an));
            }
        }
        let sig: Vec<usize> = (0..t.points.len()).filter(|&i| t.points[i].is_sigil()).collect();
        let v: Vec<CompetitorVerdict> = (0..t.points.len())
            .filter(|&k| t.points[k].counted())
            .map(|k| verdict_for(&t.points, &pairings, k, &sig, m, min_clips))
            .collect();
        let d = t6_decision(&v);
        (v, d)
    }

    /// Clip noise shared by both sides (clip difficulty) plus a little per-side noise.
    fn noise(c: usize, p: usize) -> f64 {
        let mut r = SplitMix64::new((c as u64) * 31 + p as u64 * 7 + 1);
        r.range(-0.02, 0.02)
    }

    #[test]
    fn verdicts_pass_when_sigil_is_better_faster_and_not_worse() {
        let m = Margins::default();
        let pts = vec![point("sve", PointRole::Sigil, 80.0, "algorithmic"), point("rvc", PointRole::Competitor, 200.0, "algorithmic")];
        let t = table(24, pts, |c, p| {
            let base = 3.0 + 0.3 * ((c % 5) as f64 / 5.0);
            if p == 0 {
                (base + 0.12 + noise(c, p), 0.010, 0.011, 1.00 + noise(c, p))
            } else {
                (base + noise(c, p), 0.012, 0.012, 1.05 + noise(c, p))
            }
        });
        let (v, d) = decide(&t, &m, 20);
        assert_eq!(v[0].dnsmos.status, Status::Pass, "{:?}", v[0].dnsmos);
        assert_eq!(v[0].gpe.status, Status::Pass, "{:?}", v[0].gpe);
        assert_eq!(v[0].mcd.status, Status::Pass, "{:?}", v[0].mcd);
        assert_eq!(v[0].latency.status, Status::Pass, "{:?}", v[0].latency);
        assert_eq!(d.status, "pass", "{:?}", d.reasons);
        // Too few clips for the pre-registered minimum: undecided, never a pass.
        let (v, d) = decide(&t, &m, 30);
        assert_eq!(v[0].dnsmos.status, Status::Undecided);
        assert_ne!(d.status, "pass");
    }

    #[test]
    fn verdicts_fail_or_stay_open_for_the_right_reasons() {
        let m = Margins::default();
        let mk = |sig_lat: f64, comp_lat: f64, kind2: &str| vec![point("sve", PointRole::Sigil, sig_lat, "algorithmic"), point("rvc", PointRole::Competitor, comp_lat, kind2)];
        // Only +0.02 DNSMOS: the +0.05 row fails (the CI is above 0, the mean below the margin).
        let t = table(24, mk(80.0, 200.0, "algorithmic"), |c, p| (3.0 + if p == 0 { 0.02 } else { 0.0 } + noise(c, p) * 0.1, 0.01, 0.01, 1.0));
        let (v, _) = decide(&t, &m, 20);
        assert_eq!(v[0].dnsmos.status, Status::Fail, "{:?}", v[0].dnsmos);
        // Sigil slower than 0.7x: the latency row fails, and the DNSMOS row has no matched row.
        let t = table(24, mk(150.0, 200.0, "algorithmic"), |c, p| (3.0 + if p == 0 { 0.2 } else { 0.0 } + noise(c, p), 0.01, 0.01, 1.0));
        let (v, _) = decide(&t, &m, 20);
        assert_eq!(v[0].latency.status, Status::Fail, "{:?}", v[0].latency);
        assert_eq!(v[0].dnsmos.status, Status::Pass, "150 <= 200 is matched latency");
        let t = table(24, mk(250.0, 200.0, "algorithmic"), |c, p| (3.0 + if p == 0 { 0.2 } else { 0.0 } + noise(c, p), 0.01, 0.01, 1.0));
        let (v, _) = decide(&t, &m, 20);
        assert_eq!(v[0].dnsmos.status, Status::Undecided, "slower than the competitor: no matched row");
        // Latency kinds differ (algorithmic vs e2e): nothing about latency is decided.
        let t = table(24, mk(80.0, 200.0, "e2e_loopback"), |c, p| (3.0 + if p == 0 { 0.2 } else { 0.0 } + noise(c, p), 0.01, 0.01, 1.0));
        let (v, d) = decide(&t, &m, 20);
        assert_eq!(v[0].latency.status, Status::Undecided, "{:?}", v[0].latency);
        assert!(v[0].latency.detail.contains("kinds differ"), "{:?}", v[0].latency);
        assert_eq!(d.status, "undecided");
        // Worse GPE under one output analyzer only: non-inferiority is not shown (it must hold
        // under every analyzer), and the latency row's quality condition fails with it.
        let t = table(24, mk(80.0, 200.0, "algorithmic"), |c, p| {
            let jitter = noise(c, p) * 0.5;
            if p == 0 {
                (3.2 + noise(c, p), 0.010, 0.020 + jitter, 1.0)
            } else {
                (3.0 + noise(c, p), 0.010, 0.010 + 0.5 * jitter, 1.0)
            }
        });
        let (v, d) = decide(&t, &m, 20);
        assert!(matches!(v[0].gpe.status, Status::NotShown | Status::Fail), "{:?}", v[0].gpe);
        assert!(v[0].gpe.detail.contains("dio") && v[0].gpe.detail.contains("rmvpe"));
        assert_eq!(v[0].latency.status, Status::NotShown, "{:?}", v[0].latency);
        assert_eq!(d.status, "fail");
    }

    #[test]
    fn unfair_or_infeasible_points_never_pass() {
        let m = Margins::default();
        let good = |c: usize, p: usize| (3.0 + if p == 0 { 0.2 } else { 0.0 } + noise(c, p), 0.01, 0.01, 1.0);
        // A refused competitor (another transpose): every row refused, T6 not a pass.
        let mut pts = vec![point("sve", PointRole::Sigil, 80.0, "algorithmic"), point("rvc", PointRole::Competitor, 200.0, "algorithmic")];
        pts[1].transpose = Some(12.0);
        check_points(&mut pts, 0.0, "", "GPU");
        assert!(pts[1].refusals.iter().any(|r| r.contains("transpose")), "{:?}", pts[1].refusals);
        let t = table(24, pts, good);
        let (v, d) = decide(&t, &m, 20);
        assert!(v[0].rows().iter().all(|r| r.status == Status::Refused));
        assert_eq!(d.status, "undecided");
        // An offline competitor that cannot keep up on this GPU is excluded (does not count); one
        // measured on another GPU is undecided, never a silent pass.
        let mut pts = vec![point("sve", PointRole::Sigil, 80.0, "algorithmic"), point("rvc", PointRole::Competitor, 200.0, "algorithmic"), point("app", PointRole::Competitor, 200.0, "algorithmic")];
        for p in pts.iter_mut().skip(1) {
            p.feasible = None;
            p.block_ms = Some(40.0);
        }
        pts[1].gpu = "RTX 4090".into();
        pts[1].call_ms = Some(CallMsStats { p50: Some(30.0), p95: Some(38.0), p99: Some(45.0) });
        pts[2].gpu = "RTX 3060".into();
        pts[2].call_ms = Some(CallMsStats { p50: Some(10.0), p95: Some(12.0), p99: Some(14.0) });
        check_points(&mut pts, 0.0, "", "RTX 4090");
        assert_eq!(pts[1].feasible, Some(false), "{}", pts[1].feasibility);
        assert_eq!(pts[2].feasible, None, "{}", pts[2].feasibility);
        let t = table(24, pts, good);
        let (v, d) = decide(&t, &m, 20);
        assert!(v[0].rows().iter().all(|r| r.status == Status::Excluded));
        assert!(v[1].rows().iter().all(|r| r.status == Status::Undecided));
        assert_eq!(d.status, "undecided", "{:?}", d.reasons);
        // Voice and latency-kind checks.
        let mut pts = vec![point("rvc", PointRole::Competitor, 200.0, "guess")];
        pts[0].voice_sha256 = "aaaa".into();
        check_points(&mut pts, 0.0, "bbbb", "");
        assert!(pts[0].refusals.iter().any(|r| r.contains("voice")), "{:?}", pts[0].refusals);
        assert!(pts[0].refusals.iter().any(|r| r.contains("latency_kind")), "{:?}", pts[0].refusals);
    }

    #[test]
    fn the_pareto_front_is_per_tier_and_latency_kind() {
        let pts = vec![
            point("sve-40", PointRole::Sigil, 60.0, "algorithmic"),
            point("sve-200", PointRole::Sigil, 250.0, "algorithmic"),
            point("rvc-250", PointRole::Competitor, 300.0, "algorithmic"),
            point("wok-live", PointRole::Competitor, 190.0, "e2e_loopback"),
        ];
        let t = table(3, pts, |_, p| ([3.0, 3.3, 3.1, 2.9][p], 0.0, 0.0, 1.0));
        let front: BTreeMap<String, bool> = pareto(&t.points, &t.rows).into_iter().map(|p| (p.name, p.front)).collect();
        assert!(front["sve-40"] && front["sve-200"], "{front:?}");
        assert!(!front["rvc-250"], "slower and worse than sve-200");
        assert!(front["wok-live"], "alone with its latency kind");
    }

    // ---- v2: F0 truth ----

    #[test]
    fn consensus_needs_every_analyzer_within_50_cents() {
        assert!((consensus_frame(&[200.0, 201.0, 199.5, 200.4], 50.0, 4).unwrap() - 200.2).abs() < 1e-3);
        assert_eq!(consensus_frame(&[0.0, 0.0, 0.0], 50.0, 3), Some(0.0), "unvoiced by all");
        assert_eq!(consensus_frame(&[200.0, 0.0, 200.0], 50.0, 3), None, "a voicing disagreement");
        assert_eq!(consensus_frame(&[200.0, 400.0, 200.0], 50.0, 3), None, "an octave error");
        // Three of four: one analyzer's octave error or voicing miss is outvoted.
        assert_eq!(consensus_frame(&[200.0, 400.0, 200.0, 200.0], 50.0, 3), Some(200.0));
        assert_eq!(consensus_frame(&[200.0, 0.0, 200.0, 200.0], 50.0, 3), Some(200.0));
        assert_eq!(consensus_frame(&[0.0, 0.0, 0.0, 200.0], 50.0, 3), Some(0.0));
        assert_eq!(consensus_frame(&[0.0, 0.0, 200.0, 200.0], 50.0, 3), None, "a tie is no consensus");
        assert_eq!(consensus_frame(&[0.0, 0.0, 200.0, 200.0], 50.0, 1), None, "at least a majority");
        let c = consensus(&[&[100.0, 0.0, 300.0], &[101.0, 0.0, 150.0], &[99.0, 0.0, 300.0]], 50.0, 3);
        assert!((c[0] - 100.0).abs() < 1e-6 && c[1] == 0.0 && c[2].is_nan(), "{c:?}");
        let agree = output_agreement(&[vec![100.0, 200.0, 0.0, 300.0], vec![101.0, 400.0, 0.0, 300.0]], &[true; 4]);
        assert!((agree.unwrap() - 2.0 / 3.0).abs() < 1e-9, "{agree:?}");
    }

    #[test]
    fn egg_truth_lands_on_the_10ms_grid() {
        let dir = temp("truth");
        let p = dir.join("t.csv");
        std::fs::write(&p, "time_s,f0_hz\n# comment\n0.000,0\n0.010,120.5\n0.0205,121\n0.030,0\n0.100,130\n").unwrap();
        let t = read_f0_truth(&p, 12).unwrap();
        assert_eq!(t[0], 0.0);
        assert!((t[1] - 120.5).abs() < 1e-6 && (t[2] - 121.0).abs() < 1e-6 && t[3] == 0.0);
        assert!(t[4].is_nan() && t[9].is_nan(), "frames without a truth sample within 5 ms are unknown");
        assert!((t[10] - 130.0).abs() < 1e-6);
        std::fs::write(&p, "0.0,100\n0.01,abc\n").unwrap();
        assert!(read_f0_truth(&p, 3).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- v2: the manifest ----

    #[test]
    fn manifests_are_checked() {
        let base = serde_json::json!({
            "schema": CORPUS_SCHEMA, "name": "t",
            "clips": [{"id": "a", "input": "a.wav"}, {"id": "b", "input": "b.wav"}],
            "points": [{"name": "rvc", "system": "rvc_webui", "wav": "out/{clip}.wav", "latency_kind": "algorithmic"}]
        });
        let parse = |patch: serde_json::Value| -> Result<CorpusManifest, String> {
            let mut v = base.clone();
            for (k, x) in patch.as_object().unwrap() {
                v[k] = x.clone();
            }
            let m: CorpusManifest = serde_json::from_value(v).map_err(|e| e.to_string())?;
            m.validate().map(|_| m)
        };
        let m = parse(serde_json::json!({})).unwrap();
        assert_eq!((m.min_clips, m.loudness_lufs), (20, -26.0));
        assert!((m.margins.dnsmos_superiority - 0.05).abs() < 1e-12 && (m.margins.latency_ratio - 0.7).abs() < 1e-12);
        assert!(parse(serde_json::json!({"points": [{"name": "x", "system": "s", "wav": "same.wav"}]})).unwrap_err().contains("{clip}"));
        assert!(parse(serde_json::json!({"points": [{"name": "low_latency", "system": "s", "wav": "{clip}.wav"}]})).is_err(), "Sigil's names are taken");
        assert!(parse(serde_json::json!({"clips": [{"id": "a", "input": "a.wav"}, {"id": "a", "input": "b.wav"}]})).unwrap_err().contains("duplicate"));
        assert!(parse(serde_json::json!({"sigil": {"points": [{"name": "abl", "role": "ablation"}]}})).unwrap_err().contains("versus"));
        assert!(parse(serde_json::json!({"captures": [{"point": "rvc", "record": true, "meter": {}}]})).unwrap_err().contains("live"));
        assert!(parse(serde_json::json!({"margins": {"latency_ratio": 7}})).is_err());
        assert!(parse(serde_json::json!({"typo": 1})).is_err(), "unknown fields are errors");
        let m = parse(serde_json::json!({"stand_in_references": {}, "dump_f0": true})).unwrap();
        assert_eq!(m.stand_in_references.unwrap().seeds, vec![7_001, 7_002]);
        assert!(m.dump_f0);
        assert!(parse(serde_json::json!({"schema": "other/1"})).is_err());
    }

    /// The template in the repository parses and validates (it documents the schema).
    #[test]
    fn the_example_manifest_is_valid() {
        let m: CorpusManifest = serde_json::from_str(include_str!("../../../../scripts/bench/corpus.example.json")).unwrap();
        m.validate().unwrap();
        assert!(m.points.iter().any(|p| p.tier == Tier::Live) && m.sigil.points.iter().any(|p| p.role == PointRole::Ablation));
    }

    // ---- v2: the DNSMOS rule and seams ----

    #[test]
    fn muted_onsets_read_far_below_the_inputs_own() {
        // Two utterances after silences; the rendition mutes the first 200 ms after each onset.
        let mut speech = vec![false; 40];
        speech.extend(vec![true; 60]);
        speech.extend(vec![false; 40]);
        speech.extend(vec![true; 60]);
        assert_eq!(speech_onsets(&speech, 30), vec![40, 140], "the clip's start is no onset");
        let input: Vec<f64> = speech.iter().map(|&v| if v { -20.0 } else { -90.0 }).collect();
        let (same, n) = onset_levels(&input, &input).unwrap();
        assert!(same.abs() < 1e-9 && n == 2, "{same} {n}");
        let mut muted = input.clone();
        for o in [40, 140] {
            muted[o..o + 20].iter_mut().for_each(|d| *d = -60.0);
        }
        let (r, _) = onset_levels(&input, &muted).unwrap();
        assert!((r + 40.0).abs() < 1e-6, "{r}");
        // A gated (digitally silent) onset reads at the floor, not at -180 dB.
        let mut gated = input.clone();
        gated[40..60].iter_mut().for_each(|d| *d = -200.0);
        let (g, _) = onset_levels(&input, &gated).unwrap();
        assert!((g - (ONSET_FLOOR_DB / 2.0)).abs() < 1e-6, "{g}");
        // A blip of noise in a long pause is not an onset (its window holds no speech).
        let mut long: Vec<f64> = vec![-90.0; 40];
        long.extend(vec![-20.0; 60]);
        long.extend(vec![-90.0; 80]);
        long.extend(vec![-20.0; 60]);
        long[135] = -25.0;
        assert_eq!(speech_onsets(&long.iter().map(|&d| d > -45.0).collect::<Vec<_>>(), 30), vec![40, 135, 180]);
        assert_eq!(onset_levels(&long, &long).unwrap().1, 2);
        assert!(onset_levels(&input[..30], &input[..30]).is_none());
    }

    #[test]
    fn dnsmos_needs_a_whole_window() {
        let mut k = ctx("x");
        k.input16 = vec![0.0; DNSMOS_LEN - 1];
        assert!(!k.dnsmos_ok(), "9.01 s minus one sample");
        k.input16.push(0.0);
        assert!(k.dnsmos_ok());
    }

    /// Seams are counted on the rendition's own block grid: planted joins every 25 ms read as
    /// clicks on a 25 ms grid and not on a 40 ms one.
    #[test]
    fn seams_are_found_on_each_renditions_own_block_grid() {
        let x = synth(SignalKind::Voice, 16_000, 6.0, 0.4, 41);
        let mut y = vec![0.0f32; 3_200];
        y.extend(x.iter().copied());
        let mut planted = 0;
        for j in (3_200..y.len()).step_by(400) {
            let at = j + 7;
            if at < y.len() && y[at].abs() > 0.02 {
                y[at] += 0.4 * y[at].signum();
                planted += 1;
            }
        }
        assert!(planted > 40);
        let scoring: ScoringSpec = serde_json::from_value(serde_json::json!({"f0_method": "dio", "speaker_model": "off", "dnsmos_model": "off"})).unwrap();
        let mut sc = Scorers::load(&scoring, F0Range::Speech);
        let mut an = Analyzers { list: Vec::new(), notes: Vec::new() };
        let cfg = ScoreCfg {
            loudness_lufs: -26.0,
            max_offset_16k: 16_000,
            consensus_min_agree: None,
        };
        let k = clip_ctx("x", None, x, None, Vec::new(), 1.0, &mut an, &sc, None, &cfg).unwrap();
        let mut notes = Vec::new();
        let (own, _) = score_rendition(&k, &y, Some(25.0), &mut an, &mut sc, None, &cfg, &mut notes).unwrap();
        let (other, _) = score_rendition(&k, &y, Some(40.0), &mut an, &mut sc, None, &cfg, &mut notes).unwrap();
        assert!(own.seam_clicks.unwrap() >= 20, "{own:?}");
        assert!(other.seam_clicks.unwrap() < own.seam_clicks.unwrap() / 4, "{other:?}");
        assert!((own.offset_ms.unwrap() - 200.0).abs() <= 10.0);
        assert!(own.loudness_lufs.is_some() && own.dnsmos_ovrl.is_none(), "no DNSMOS model, no DNSMOS");
        let (none, _) = score_rendition(&k, &y, None, &mut an, &mut sc, None, &cfg, &mut notes).unwrap();
        assert!(none.seam_clicks.is_none(), "no block size, no seam count");
    }

    // ---- v2: a whole corpus run ----

    /// A two-clip corpus with the stub as Sigil, an ablation of it, and two other points: one
    /// whose second rendition converted another input (refused by its sidecar's hash) and one
    /// with another transpose (refused). Every refusal is in the report and T6 is not a pass.
    #[test]
    fn a_corpus_run_scores_points_and_refuses_unfair_ones() {
        let dir = temp("corpus");
        std::fs::create_dir_all(dir.join("out/good")).unwrap();
        std::fs::create_dir_all(dir.join("out/shifted")).unwrap();
        std::fs::create_dir_all(dir.join("out/live")).unwrap();
        let mut shas = Vec::new();
        for (i, id) in ["a", "b"].iter().enumerate() {
            let x = synth(SignalKind::Voice, 16_000, 4.0, 0.3, 50 + i as u64);
            let p = dir.join(format!("{id}.wav"));
            write_wav(&p, &x, 16_000, 1).unwrap();
            shas.push(file_sha256(&p).unwrap());
            // A competitor: the clip delayed by 120 ms.
            let mut y = vec![0.0f32; 1_920];
            y.extend(x.iter().copied());
            // Live captures (no sidecars): the same clip, delayed by 120 ms.
            write_wav(&dir.join(format!("out/live/{id}.wav")), &y, 16_000, 1).unwrap();
            for sub in ["good", "shifted"] {
                let w = dir.join(format!("out/{sub}/{id}.wav"));
                write_wav(&w, &y, 16_000, 1).unwrap();
                let input_sha = if sub == "good" && *id == "b" { "0".repeat(64) } else { shas[i].clone() };
                let side = serde_json::json!({
                    "schema": SIDECAR_SCHEMA, "system": "rvc_webui", "input_sha256": input_sha,
                    "config": {"block_ms": 40, "transpose": if sub == "shifted" { 12 } else { 0 }},
                    "call_ms": {"p50": 5.0, "p95": 6.0, "p99": 7.0}, "gpu": "", "latency_ms": 150.0, "latency_kind": "algorithmic"
                });
                std::fs::write(format!("{}.json", w.display()), side.to_string()).unwrap();
            }
        }
        let manifest = serde_json::json!({
            "schema": CORPUS_SCHEMA, "name": "tiny", "min_clips": 2,
            "analyzers": ["dio"],
            "clips": [{"id": "a", "input": "a.wav", "sha256": shas[0]}, {"id": "b", "input": "b.wav", "sha256": shas[1]}],
            "points": [
                {"name": "good", "system": "rvc_webui", "wav": "out/good/{clip}.wav", "config": {"ep": "cpu"}},
                {"name": "shifted", "system": "rvc_webui", "wav": "out/shifted/{clip}.wav", "config": {"ep": "cpu"}},
                {"name": "live-app", "system": "w_okada", "tier": "live", "wav": "out/live/{clip}.wav", "config": {"block_ms": 40, "transpose": 0}},
                {"name": "live-cal", "system": "w_okada", "tier": "live", "wav": "out/live/{clip}.wav", "latency_ms": 130.0,
                 "latency_kind": "e2e_loopback", "config": {"block_ms": 40, "transpose": 0}}
            ],
            "sigil": {"points": [{"name": "slow-stub", "role": "ablation", "versus": "sigil", "engine": {"compute_ms": 1.0}}]}
        });
        let mpath = dir.join("corpus.json");
        std::fs::write(&mpath, manifest.to_string()).unwrap();
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "cmp2", "seed": 5,
            // A 200 ms block keeps the Sigil row real-time on a busy or slow CI machine: its per-call
            // time is the debug harness's wall clock (61 ms p99 under a parallel test run against the
            // 10.7 ms device block), and an infeasible row empties the listening manifest asserted below.
            "engine": {"kind": "stub", "fixed_block_ms": 200},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 4, "level_dbfs": -12},
            "runs": [
                {"mode": "compare", "name": "h2h", "corpus": mpath,
                 "scoring": {"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio"}}
            ]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = |_: &Progress| {};
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        let run = &r.runs[0];
        assert_eq!(run.status, RunStatus::Passed, "{:?} {}", run.error, run.details);
        let v2 = &run.details["v2"];
        let points = v2["points"].as_array().unwrap();
        let by = |n: &str| points.iter().find(|p| p["name"] == n).unwrap_or_else(|| panic!("{n}: {v2}"));
        assert_eq!(by("sigil")["role"], "sigil");
        assert_eq!(by("slow-stub")["role"], "ablation");
        assert!(by("shifted")["refusals"].as_array().unwrap().iter().any(|x| x.as_str().unwrap().contains("transpose")), "{}", by("shifted"));
        assert!(by("good")["refusals"].as_array().unwrap().is_empty(), "{}", by("good"));
        // The good point's second rendition converted another input: refused, the first scored.
        let rows = v2["rows"].as_array().unwrap();
        let row = |p: &str, c: &str| rows.iter().find(|r| r["point"] == p && r["clip"] == c).unwrap();
        assert_eq!(row("good", "a")["ok"], true, "{}", row("good", "a"));
        assert_eq!(row("good", "b")["ok"], false);
        assert!(row("good", "b")["notes"].to_string().contains("refused"), "{}", row("good", "b"));
        assert!((row("good", "a")["scores"]["offset_ms"].as_f64().unwrap() - 120.0).abs() <= 10.0);
        assert_eq!(run.metrics["renditions_refused"], 1.0);
        assert_eq!(run.metrics["points_refused"], 1.0);
        assert_eq!(run.metrics["t6_pass"], 0.0);
        assert_eq!(v2["ablations"].as_array().unwrap().len(), 1, "{}", v2["ablations"]);
        assert_eq!(v2["ablations"][0]["versus"], "sigil");
        // A clip a point has no complete score for makes the whole row incomplete.
        assert_eq!(by("good")["unscored_clips"], serde_json::json!(["b"]), "{}", by("good"));
        assert!(by("good")["flags"].to_string().contains("incomplete"), "{}", by("good"));
        assert!(by("shifted")["unscored_clips"].as_array().unwrap().is_empty(), "{}", by("shifted"));
        assert!(run.metrics["points_incomplete"] >= 1.0 && run.metrics["clips_unscored"] >= 1.0);
        let verdicts = v2["verdicts"].as_array().unwrap();
        let good = verdicts.iter().find(|v| v["point"] == "good").unwrap();
        assert_eq!(good["dnsmos"]["status"], "undecided", "{good}");
        assert!(good["dnsmos"]["detail"].as_str().unwrap().contains("incomplete"), "{good}");
        // A live offset nobody calibrated is labelled so; a stated e2e_loopback latency is kept.
        assert_eq!((by("live-app")["latency_kind"].as_str(), by("live-app")["latency_source"].as_str()), (Some("uncalibrated"), Some("measured")), "{}", by("live-app"));
        assert_eq!((by("live-cal")["latency_kind"].as_str(), by("live-cal")["latency_ms"].as_f64()), (Some("e2e_loopback"), Some(130.0)), "{}", by("live-cal"));
        // The listening manifest pairs Sigil's row with each fair counted point and the ablation,
        // and leaves the refused one out (and says so).
        let trials = v2["make_test"]["trials"].as_array().unwrap();
        assert!(trials.iter().any(|t| t["system_b"] == "good"), "trials {trials:?}, skipped {}", v2["make_test"]["skipped"]);
        assert!(!trials.iter().any(|t| t["system_b"] == "shifted"), "a refused point is not heard: {trials:?}");
        let skipped = v2["make_test"]["skipped"].as_array().unwrap();
        assert!(skipped.iter().any(|s| s["point"] == "shifted" && s["why"].as_str().unwrap().contains("transpose")), "{skipped:?}");
        assert!(v2["pareto"].as_array().unwrap().iter().any(|p| p["name"] == "sigil"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- review fixes: provenance, completeness, fairness ----

    fn side(clip: &str, transpose: f64, block: f64, p99: f64, voice: &str) -> (String, Sidecar) {
        let mut config = serde_json::Map::new();
        config.insert("transpose".into(), transpose.into());
        config.insert("block_ms".into(), block.into());
        config.insert("ep".into(), "cpu".into());
        (
            clip.to_string(),
            Sidecar {
                system: "rvc_webui".into(),
                voice_sha256: voice.into(),
                config,
                call_ms: Some(CallMsStats {
                    p50: Some(p99 / 3.0),
                    p95: Some(p99 / 1.5),
                    p99: Some(p99),
                }),
                latency_ms: Some(150.0),
                latency_kind: "algorithmic".into(),
                ..Sidecar::default()
            },
        )
    }

    fn manifest_point(name: &str) -> Point {
        Point::new(name, "rvc_webui", Tier::Offline, PointRole::Competitor, "manifest")
    }

    /// Stand-in references (Sigil's own render) are scored apart and cannot decide the T6 MCD row.
    #[test]
    fn stand_in_references_are_scored_apart_and_never_decide_the_mcd_row() {
        let m = Margins::default();
        let pts = vec![point("sve", PointRole::Sigil, 80.0, "algorithmic"), point("rvc", PointRole::Competitor, 200.0, "algorithmic")];
        let good = |c: usize, p: usize| (3.0 + if p == 0 { 0.2 } else { 0.0 } + noise(c, p), 0.01, 0.01, 1.0);
        let t = table(24, pts.clone(), good);
        let (v, d) = decide(&t, &m, 20);
        assert_eq!((v[0].mcd.status, d.status), (Status::Pass, "pass"), "control: upstream references decide the row");
        // The same numbers, but measured against stand-ins only.
        let mut t = table(24, pts, good);
        for r in t.rows.iter_mut().flatten().flatten() {
            r.mcd_stand_in_db = r.mcd_ref_db.take();
        }
        let (v, d) = decide(&t, &m, 20);
        assert_eq!(v[0].mcd.status, Status::Undecided, "{:?}", v[0].mcd);
        assert!(v[0].mcd.detail.contains("stand-in"), "{}", v[0].mcd.detail);
        assert_ne!(d.status, "pass", "{:?}", d.reasons);
        let p = pairing(&t.points, &t.rows, &t.clips, 0, 1, &["dio"]);
        assert!(p.mcd.is_none() && p.mcd_stand_in.is_some(), "ablations keep the stand-in numbers");

        // The scoring path: a stand-in reference feeds mcd_stand_in_db, an upstream one mcd_ref_db.
        let scoring: ScoringSpec = serde_json::from_value(serde_json::json!({"f0_method": "dio", "speaker_model": "off", "dnsmos_model": "off"})).unwrap();
        let mut sc = Scorers::load(&scoring, F0Range::Speech);
        let mut an = Analyzers::load(&["dio".to_string()], "", F0Range::Speech);
        let cfg = ScoreCfg {
            loudness_lufs: -26.0,
            max_offset_16k: 16_000,
            consensus_min_agree: None,
        };
        let x = synth(SignalKind::Voice, 16_000, 6.0, 0.4, 41);
        let reference = |stand_in: bool| RefIn {
            name: "r".into(),
            stand_in,
            audio: Ok(x.clone()),
        };
        let mut y = vec![0.0f32; 3_200];
        y.extend(x.iter().copied());
        let mut score = |refs: Vec<RefIn>| {
            let k = clip_ctx("x", None, x.clone(), None, refs, 1.0, &mut an, &sc, None, &cfg).unwrap();
            let mut notes = Vec::new();
            let (s, _) = score_rendition(&k, &y, None, &mut an, &mut sc, None, &cfg, &mut notes).unwrap_or_else(|| panic!("{notes:?}"));
            (k, s, notes)
        };
        let (k, s, notes) = score(vec![reference(false), reference(true)]);
        assert_eq!((k.ref_files, k.stand_in_files, k.refs.len()), (1, 1, 2));
        assert!(s.mcd_ref_db.is_some() && s.mcd_stand_in_db.is_some(), "{s:?} {notes:?}");
        let (k, s, _) = score(vec![reference(true)]);
        assert!(s.mcd_ref_db.is_none() && s.mcd_stand_in_db.is_some(), "{s:?}");
        assert!(!k.ref_floor_stand_in && k.ref_floor.is_none(), "one reference has no floor");
        let (k, _, _) = score(vec![reference(true), reference(true)]);
        assert!(k.ref_floor.is_some() && k.ref_floor_stand_in, "a floor from stand-ins is labelled so");
        assert!(k.notes.iter().any(|n| n.contains("stand-in")), "{:?}", k.notes);
    }

    /// Every clip's sidecar speaks for the point, not the first one's.
    #[test]
    fn every_clips_sidecar_speaks_for_the_point() {
        // Consistent sidecars, but the third clip's block compute is over the 40 ms block: the
        // first clip's sidecar alone said "real time".
        let mut p = manifest_point("rvc");
        p.merge_sidecars(&[side("a", 0.0, 40.0, 6.0, "aa"), side("b", 0.0, 40.0, 7.0, "aa"), side("c", 0.0, 40.0, 45.0, "aa")]);
        assert!(p.refusals.is_empty(), "{:?}", p.refusals);
        assert_eq!(p.call_ms.unwrap().p99, Some(45.0), "the worst p99 over the clips");
        check_points(std::slice::from_mut(&mut p), 0.0, "aa", "GPU");
        assert_eq!(p.feasible, Some(false), "{}", p.feasibility);
        // A clip re-run with another transpose, a stale file with another voice, or another block.
        for (bad, what) in [
            (side("b", 12.0, 40.0, 6.0, "aa"), "transposes"),
            (side("b", 0.0, 40.0, 6.0, "bb"), "voices"),
            (side("b", 0.0, 60.0, 6.0, "aa"), "blocks"),
        ] {
            let mut p = manifest_point("rvc");
            p.merge_sidecars(&[side("a", 0.0, 40.0, 6.0, "aa"), bad]);
            assert!(p.refusals.iter().any(|r| r.contains(what) && r.contains("a:") && r.contains("b:")), "{what}: {:?}", p.refusals);
        }
        // What the point states itself must hold in every sidecar, not just the first.
        let mut p = manifest_point("rvc");
        p.voice_sha256 = "aa".into();
        p.config.insert("transpose".into(), 0.0.into());
        p.merge_sidecars(&[side("a", 0.0, 40.0, 6.0, "aa"), side("b", 0.0, 40.0, 6.0, "cc")]);
        assert!(p.refusals.iter().any(|r| r.contains("clip b") && r.contains("voice")), "{:?}", p.refusals);
        let mut p = manifest_point("rvc");
        p.config.insert("transpose".into(), 0.0.into());
        p.merge_sidecars(&[side("a", 0.0, 40.0, 6.0, "aa"), side("b", 7.0, 40.0, 6.0, "aa")]);
        assert!(p.refusals.iter().any(|r| r.contains("transpose")), "{:?}", p.refusals);
        // Aggregation: the median p50, the worst p95 and p99.
        let c = aggregate_call_ms(
            [
                CallMsStats { p50: Some(1.0), p95: Some(2.0), p99: Some(9.0) },
                CallMsStats { p50: Some(3.0), p95: Some(8.0), p99: Some(4.0) },
                CallMsStats { p50: Some(2.0), p95: Some(4.0), p99: None },
            ]
            .iter(),
        )
        .unwrap();
        assert_eq!((c.p50, c.p95, c.p99), (Some(2.0), Some(8.0), Some(9.0)));
        assert!(aggregate_call_ms([].iter()).is_none());
    }

    /// A live capture cannot attest its own voice.
    #[test]
    fn a_live_capture_never_vouches_for_its_own_voice() {
        let spec = |extra: serde_json::Value| -> PointSpec {
            let mut v = serde_json::json!({"name": "wok", "system": "w_okada", "tier": "live", "wav": "out/{clip}.wav"});
            for (k, x) in extra.as_object().unwrap() {
                v[k] = x.clone();
            }
            serde_json::from_value(v).unwrap()
        };
        let outcome = PlayRecordOutcome {
            output_device: "out".into(),
            input_device: "in".into(),
            output_rate: 48_000,
            input_rate: 48_000,
            response: Vec::new(),
            latency_ms: Some(140.0),
            latency_correlation: Some(0.9),
            timebase_jitter_ms: (0.1, 0.2),
            glitches: Vec::new(),
            dropouts: Vec::new(),
            metrics: Metrics::new(),
        };
        let dir = temp("capture");
        let wav = dir.join("a.wav");
        let side_of = |s: &serde_json::Value| {
            std::fs::write(format!("{}.json", wav.display()), s.to_string()).unwrap();
            Sidecar::of(&wav).unwrap().unwrap()
        };
        // Nobody asserted a voice: the sidecar says none, and the point is flagged, not verified.
        let s = capture_sidecar(&spec(serde_json::json!({})), Path::new("a.wav"), "sha", 3.0, &outcome);
        assert_eq!((s["voice_sha256"].as_str(), s["voice_asserted"].as_bool()), (Some(""), Some(false)));
        let mut p = manifest_point("wok");
        p.tier = Tier::Live;
        p.merge_sidecars(&[("a".into(), side_of(&s))]);
        check_points(std::slice::from_mut(&mut p), 0.0, "corpusvoice", "");
        assert!(p.refusals.is_empty(), "{:?}", p.refusals);
        assert!(p.flags.iter().any(|f| f.contains("voice not verified")), "{:?}", p.flags);
        // The user asserts the voice (any case): the sidecar carries it.
        let s = capture_sidecar(&spec(serde_json::json!({"voice_sha256": "ABC"})), Path::new("a.wav"), "sha", 0.0, &outcome);
        assert_eq!((s["voice_sha256"].as_str(), s["voice_asserted"].as_bool()), (Some("abc"), Some(true)));
        // ... and an asserted other voice is refused, as for any system.
        let mut p = manifest_point("wok");
        p.merge_sidecars(&[("a".into(), side_of(&s))]);
        check_points(std::slice::from_mut(&mut p), 0.0, "corpusvoice", "");
        assert!(p.refusals.iter().any(|r| r.contains("voice")), "{:?}", p.refusals);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A row that misses clips decides nothing, on either side.
    #[test]
    fn rows_that_miss_clips_decide_nothing() {
        let m = Margins::default();
        let good = |c: usize, p: usize| (3.0 + if p == 0 { 0.2 } else { 0.0 } + noise(c, p), 0.01, 0.01, 1.0);
        let pts = || vec![point("sve", PointRole::Sigil, 80.0, "algorithmic"), point("rvc", PointRole::Competitor, 200.0, "algorithmic")];
        let missing = |n: usize| (0..n).map(|i| format!("c{i}")).collect::<Vec<_>>();
        let t = table(24, pts(), good);
        assert_eq!(decide(&t, &m, 20).1.status, "pass", "control");
        // The competitor muted 4 clips: it is not excused by the 20 left.
        let mut t = table(24, pts(), good);
        t.points[1].unscored_clips = missing(4);
        let (v, d) = decide(&t, &m, 20);
        assert!(v[0].rows().iter().all(|r| r.status == Status::Undecided && r.detail.contains("incomplete")), "{:?}", v[0].rows());
        assert_ne!(d.status, "pass");
        // Sigil failed 4 of 24: its row cannot carry a pass on the 20 it did score.
        let mut t = table(24, pts(), good);
        t.points[0].unscored_clips = missing(4);
        let (v, d) = decide(&t, &m, 20);
        assert!(v[0].rows().iter().all(|r| r.status == Status::Undecided && r.detail.contains("scored every clip")), "{:?}", v[0].rows());
        assert_ne!(d.status, "pass");
        // What counts as unscored.
        let clips: Vec<ClipCtx> = (0..4).map(|i| ctx(&format!("c{i}"))).collect();
        let cell = |dn: Option<f64>| {
            Some(RowScores {
                dnsmos_ovrl: dn,
                ..RowScores::default()
            })
        };
        let row = vec![cell(Some(3.0)), None, cell(None), cell(Some(2.0))];
        assert_eq!(unscored_cells(&row, &clips, true), vec!["c1", "c2"], "no rendition, and no DNSMOS where the clip needs one");
        assert_eq!(unscored_cells(&row, &clips, false), vec!["c1"], "without a DNSMOS scorer only missing renditions count");
        let mut short = ctx("s");
        short.input16 = vec![0.0; 1_000];
        assert!(unscored_cells(&[cell(None)], &[short], true).is_empty(), "a clip under 9.01 s needs no DNSMOS");
        assert_eq!(list_some(&missing(8), 3), "c0, c1, c2 and 5 more");
    }

    /// An uncalibrated live offset has no place next to calibrated latencies.
    #[test]
    fn an_uncalibrated_live_offset_is_never_compared() {
        let pts = vec![
            point("sve-live", PointRole::Sigil, 60.0, "e2e_loopback"),
            point("app", PointRole::Competitor, 190.0, UNCALIBRATED),
            point("app2", PointRole::Competitor, 190.0, UNCALIBRATED),
        ];
        let t = table(3, pts, |_, p| ([3.0, 3.1, 3.0][p], 0.0, 0.0, 1.0));
        let p = pairing(&t.points, &t.rows, &t.clips, 0, 1, &["dio"]);
        assert!(p.latency_ratio.is_none() && p.latency_note.contains("uncalibrated"), "{}", p.latency_note);
        // Neither are two uncalibrated offsets compared with each other (different rigs).
        let q = pairing(&t.points, &t.rows, &t.clips, 1, 2, &["dio"]);
        assert!(q.latency_ratio.is_none(), "{}", q.latency_note);
        let front: BTreeMap<String, bool> = pareto(&t.points, &t.rows).into_iter().map(|p| (p.name, p.front)).collect();
        assert!(front["sve-live"] && !front["app"] && !front["app2"], "{front:?}");
        assert!(check_latency_kind("p", UNCALIBRATED).is_ok());
    }

    /// The run's GPU is the adapter the engine used, not the first one listed.
    #[test]
    fn the_gpu_is_the_adapter_the_engine_ran_on() {
        let adapter = |index: u32, vendor_id: u32, name: &str, software: bool| GpuAdapter {
            index,
            name: name.into(),
            vendor: gpu_mem::vendor_name(vendor_id).into(),
            vendor_id,
            device_id: 0,
            luid: String::new(),
            dedicated_vram_mb: 0,
            shared_memory_mb: 0,
            software,
            driver_version: None,
            nvidia_driver: None,
            vram: None,
        };
        let list = vec![
            adapter(0, gpu_mem::VENDOR_INTEL, "Intel UHD", false),
            adapter(1, gpu_mem::VENDOR_NVIDIA, "NVIDIA GeForce RTX 4090", false),
            adapter(2, gpu_mem::VENDOR_MICROSOFT, "Microsoft Basic Render Driver", true),
        ];
        let info = |active: &str, requested: &str| {
            let mut i = EngineInfo::new("rvc", 16_000, 48_000, 0);
            i.execution_provider_active = Some(active.into());
            i.execution_provider_requested = Some(requested.into());
            i
        };
        let rtx = Some("NVIDIA GeForce RTX 4090");
        assert_eq!(engine_gpu(&info("cuda", "cuda"), 0, &list).as_deref(), rtx);
        assert_eq!(engine_gpu(&info("dml", "cuda"), 0, &list).as_deref(), rtx, "a CUDA request that fell back to DirectML stays on the NVIDIA card");
        assert_eq!(engine_gpu(&info("dml", "dml"), 0, &list).as_deref(), Some("Intel UHD"), "DirectML's device id is the DXGI index");
        assert_eq!(engine_gpu(&info("dml_hp", "dml_hp"), 1, &list).as_deref(), rtx);
        assert_eq!(engine_gpu(&info("cpu", "cuda"), 0, &list), None);
        assert_eq!(engine_gpu(&info("dml", "dml"), 2, &list), None, "a software adapter is no GPU");
        // Before any Sigil row has run: the manifest's name, else the first hardware adapter, with
        // a note when several exist.
        let (g, note) = run_gpu("", &list);
        assert_eq!(g, "Intel UHD");
        assert!(note.unwrap().contains("2 GPUs"));
        assert_eq!(run_gpu("Pinned GPU", &list), ("Pinned GPU".to_string(), None));
        assert!(run_gpu("", &list[1..2]).1.is_none());
        assert_eq!(run_gpu("", &[]), (String::new(), None));
    }

    /// Settings besides the engine that change the sound are shown side by side.
    #[test]
    fn operating_point_differences_are_flagged_side_by_side() {
        let mut sigil = Point::new("sve", "sigil", Tier::Offline, PointRole::Sigil, "in_process");
        for (k, v) in [("index_rate", 0.0), ("protect", 0.33), ("rms_mix", 0.25)] {
            sigil.config.insert(k.into(), v.into());
        }
        sigil.config.insert("f0_method".into(), "rmvpe".into());
        let mut applio = manifest_point("applio");
        for (k, v) in [("index_rate", 0.75), ("protect", 0.5), ("rms_mix", 0.25)] {
            applio.config.insert(k.into(), v.into());
        }
        applio.config.insert("f0_method".into(), "RMVPE".into());
        let flags = operating_point_flags(&applio, std::slice::from_ref(&sigil));
        assert!(flags.iter().any(|f| f.contains("index_rate 0.75") && f.contains("Sigil rows use 0")), "{flags:?}");
        assert!(flags.iter().any(|f| f.contains("protect 0.5") && f.contains("0.33")), "{flags:?}");
        assert!(!flags.iter().any(|f| f.contains("rms_mix") || f.contains("F0")), "equal settings are not flagged: {flags:?}");
        // Another F0 method, and a setting the point does not state.
        applio.config.insert("f0_method".into(), "crepe".into());
        applio.config.remove("protect");
        let flags = operating_point_flags(&applio, std::slice::from_ref(&sigil));
        assert!(flags.iter().any(|f| f.contains("F0 method crepe") && f.contains("rmvpe")), "{flags:?}");
        assert!(flags.iter().any(|f| f.contains("protect not stated")), "{flags:?}");
        // check_points applies it to the corpus's competitors only, and the T6 decision carries it.
        let mut pts = vec![sigil, applio];
        check_points(&mut pts, 0.0, "", "GPU");
        assert!(pts[1].flags.iter().any(|f| f.starts_with("operating point:")), "{:?}", pts[1].flags);
        assert!(!pts[0].flags.iter().any(|f| f.starts_with("operating point:")));
        let mut inline = manifest_point("inline");
        inline.source = "inline";
        let mut pts = vec![pts.remove(0), inline];
        check_points(&mut pts, 0.0, "", "GPU");
        assert!(!pts[1].flags.iter().any(|f| f.starts_with("operating point:")), "v1 candidates are not checked");
    }

    /// Every named configuration builds a valid engine and stream spec (the T6 headline rows).
    #[test]
    fn the_named_sigil_configurations_build_valid_specs() {
        let dir = temp("named");
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "cfg", "seed": 1,
            "engine": {"kind": "rvc", "slot_dir": dir.join("slot"), "execution_provider": "cuda", "pitch_extractor": "rmvpe",
                       "levers": {"anchored_noise": "off"}, "lookahead_ms": 10.0, "block_ms": 60.0, "precision": "fp16",
                       "denoiser": "gtcrn", "noise_seed": 3, "intra_threads": 2, "content_variant": "fp16", "qos": "none"},
            "stream": {"input_rate": 48000, "output_rate": 48000, "pitch_semitones": 5.0},
            "runs": [{"mode": "offline", "name": "o"}]
        }))
        .unwrap();
        // The autotuned row reads the stored result of this PC's GPU and driver.
        let (gpu, driver) = super::super::autotune::gpu_identity();
        let mut params = serde_json::Map::new();
        for (k, v) in [
            ("rvc_execution_provider", serde_json::json!("cpu")),
            ("pitch_extractor", serde_json::json!("fcpe")),
            ("precision", serde_json::json!("fp32")),
            ("block_ms", serde_json::json!(40.0)),
            ("extra_context_ms", serde_json::json!(96.0)),
            ("partial_decode", serde_json::json!("on")),
        ] {
            params.insert(k.into(), v);
        }
        let stored = super::super::autotune::AutotuneResult {
            gpu,
            driver,
            params,
            conditions: Some(super::super::autotune::TuneConditions::default()),
            sigil_version: "test".into(),
            ..Default::default()
        };
        super::super::autotune::store_result(&dir, &stored).unwrap();
        for name in ["shipping_default", "autotuned", "low_latency", "quality"] {
            let b = named_config(name, &spec, 5.0, Some(dir.as_path())).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!((b.point.name.as_str(), b.point.role, b.point.tier), (name, PointRole::Sigil, Tier::Offline));
            assert!(!b.point.label.is_empty());
            let Some(EngineSpec::Rvc(e)) = &b.engine else { panic!("{name}: no rvc engine") };
            // The app's settings, not the benchmark knobs of the spec: nothing of the spec's levers,
            // lookahead, noise seed, denoiser, content variant or thread override survives.
            assert!(e.noise_seed.is_none() && e.intra_threads.is_none() && e.denoiser == "off" && e.content_variant == "auto" && e.qos == "app", "{name}: {e:?}");
            assert!(!e.levers.contains_key("anchored_noise") || name == "autotuned", "{name}: {:?}", e.levers);
            assert_eq!(b.stream.pitch_semitones, 5.0, "{name}: the run's transpose");
            // The whole thing is a valid spec again (the engine's and the stream's own validation).
            let mut v = serde_json::to_value(&spec).unwrap();
            v["engine"] = serde_json::to_value(b.engine.as_ref().unwrap()).unwrap();
            v["stream"] = serde_json::to_value(&b.stream).unwrap();
            BenchSpec::from_value(v).unwrap_or_else(|err| panic!("{name}: {err}"));
            match name {
                "shipping_default" => {
                    assert_eq!((e.preset.as_str(), e.execution_provider.as_str(), e.pitch_extractor.as_str()), ("streamer", "cpu", "fcpe"));
                    assert_eq!((e.lookahead_ms, b.stream.extra_context_ms, b.stream.index_rate), (0.0, 64.0, 0.0));
                }
                "low_latency" => {
                    assert_eq!((e.preset.as_str(), e.execution_provider.as_str(), e.pitch_extractor.as_str(), e.lookahead_ms), ("low_latency", "cuda", "rmvpe", 0.0));
                }
                "quality" => {
                    assert_eq!((e.preset.as_str(), e.lookahead_ms, b.stream.extra_context_ms, b.stream.index_rate), ("quality", 20.0, 192.0, 0.5));
                    assert_eq!(e.build, "auto", "a lookahead needs the benchmark's own builder");
                }
                _ => {
                    assert_eq!((e.execution_provider.as_str(), e.pitch_extractor.as_str(), e.precision.as_str()), ("cpu", "fcpe", "fp32"));
                    assert_eq!((e.block_ms, b.stream.extra_context_ms), (40.0, 96.0));
                    assert_eq!(e.levers.get("partial_decode").map(String::as_str), Some("on"), "{:?}", e.levers);
                }
            }
        }
        assert!(named_config("nonsense", &spec, 0.0, Some(dir.as_path())).is_err());
        assert!(named_config("autotuned", &spec, 0.0, None).is_err(), "no app data folder");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A failed Sigil conversion fails a v1 run, as it always did.
    #[test]
    fn a_failed_sigil_conversion_still_fails_a_v1_run() {
        let dir = temp("v1fail");
        let spec = BenchSpec::from_value(serde_json::json!({
            "name": "cmpf", "seed": 1,
            "engine": {"kind": "stub"},
            "input": {"kind": "synthetic", "signal": "voice", "seconds": 4, "level_dbfs": -12},
            "runs": [{"mode": "compare", "name": "h2h", "input": dir.join("missing.wav"),
                      "scoring": {"speaker_model": "off", "dnsmos_model": "off", "f0_method": "dio"}}]
        }))
        .unwrap();
        let cancel = AtomicBool::new(false);
        let progress = |_: &Progress| {};
        let r = run_spec(&spec, None, &BenchContext { cancel: &cancel, progress: &progress });
        let run = &r.runs[0];
        assert_eq!(run.status, RunStatus::Error, "{:?} {}", run.error, run.details);
        assert!(run.error.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
