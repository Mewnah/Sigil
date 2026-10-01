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

use std::path::Path;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use serde::Serialize;

use super::ceiling::{ceiling, CallSpan, StreamRecorder};
use super::dsp::align;
use super::engine::lock;
use super::offline::{run_offline, OfflineInput};
use super::quality::{
    click_events, click_report, f0_errors_masked, mcd_db, seam_flux_ratio, speech_frames, unmatched, ClickConfig, ClickReport, F0Analyzer,
    QualityMetrics,
};
use super::report::RunOutcome;
use super::scorers::{Dnsmos, SpeakerEmbedder, Transcriber};
use super::signals::{channel, read_wav, resample, to_mono};
use super::spec::{BenchSpec, OfflineRun, QualityRun, ScoringSpec};
use super::Prepared;
use crate::services::neural_audio::vc::engine::options::F0Range;

/// The models a run scores with (loaded once per run).
pub(crate) struct Scorers {
    pub f0: F0Analyzer,
    pub speaker: Option<SpeakerEmbedder>,
    pub dnsmos: Option<Dnsmos>,
    pub click: ClickConfig,
    pub notes: Vec<String>,
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
        let speaker = if off(&s.speaker_model) {
            None
        } else {
            match SpeakerEmbedder::resolve(&s.speaker_model, &ep) {
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
    // standard deviations): a voice that crackles everywhere is not clicking at its seams, and
    // with 40 ms blocks the windows cover a third of the time.
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

/// Run a `quality` run on the spec's engine.
pub(super) fn run_quality(
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
}

