//! Compare mode (P3-F; plan T6): scores competitor output WAVs recorded from the same input
//! (w-okada, RVC WebUI realtime, Applio realtime, ...) with the same metrics as Sigil's own output,
//! and, for latency, takes each app's end-to-end latency measured by the e2e-loopback meter.
//! Offline drivers for RVC WebUI's realtime path and Applio realtime live in `scripts/vc-bench`
//! (they run in the user's own Python environment). The w-okada algorithm is emulated exactly by
//! the `LegacyOnnxRvc` family (P5).
//!
//! Each candidate is aligned to the input on energy envelopes (voices differ, timing does not),
//! and scored over the input's speech region:
//! - DNSMOS P.835 (SIG / BAK / OVRL) of the candidate;
//! - F0 against the input's F0 shifted by the run's transpose: voiced RMSE in cents, gross pitch
//!   error, V/UV error (one F0 analyzer for everything);
//! - CAM++ cosine against a clip of the target voice (`scoring.speaker_reference`);
//! - artifact events per minute: impulses of the candidate with no impulse of the input within
//!   12 ms (clicks and pops the conversion added);
//! - the latency given for it.
//!
//! The run's metrics are Sigil's row (`sigil_*`) and the T6 margins against the competitors:
//! `dnsmos_ovrl_margin` = Sigil − the best competitor (T6: ≥ 0.05), `gross_pitch_error_margin` =
//! the best competitor − Sigil (≥ 0: no competitor tracks pitch better), `latency_ratio` = Sigil /
//! the fastest competitor (T6: ≤ 0.7).
//!
//! # Frozen by P3-S0 (signatures; P3-F implements)
//! [`CompareRow`] (gained fields), [`compare_wavs`].

use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

use serde::Serialize;

use super::dsp::align;
use super::fidelity::{clip_16k, Scorers};
use super::metrics::Metrics;
use super::offline::{run_offline, OfflineInput};
use super::quality::{click_events, f0_errors_masked, speech_frames, unmatched, QualityMetrics};
use super::report::{RunOutcome, RunReport};
use super::signals::{channel, render_input, resample, to_interleaved};
use super::spec::{BenchSpec, CompareRun, OfflineRun, ScoringSpec};
use super::Prepared;
use crate::services::neural_audio::vc::engine::options::F0Range;

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

impl InputAnalysis {
    fn new(audio16: Vec<f32>, semitones: f32, sc: &mut Scorers) -> Result<InputAnalysis, String> {
        let speech = speech_frames(&audio16, 40.0);
        let first = speech.iter().position(|&m| m).ok_or("the compare input is silent")?;
        let last = speech.iter().rposition(|&m| m).unwrap_or(first);
        let margin = (REGION_MARGIN_S * 16_000.0) as usize;
        let region = ((first * 160).saturating_sub(margin), ((last + 1) * 160 + margin).min(audio16.len()));
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

/// Run a `compare` run.
pub(super) fn run_compare(
    c: &CompareRun,
    spec: &BenchSpec,
    prepared: Option<&Prepared>,
    previous: &[RunReport],
    write_files: bool,
    cancel: &AtomicBool,
    progress: &(dyn Fn(f64, &str) + Sync),
) -> Result<RunOutcome, String> {
    let s = &spec.stream;
    let ch_in = usize::from(s.input_channels);
    let range = prepared.and_then(|p| p.engine.rvc.as_ref()).map_or(F0Range::Speech, |r| r.f0_range);
    let mut sc = Scorers::load(&c.scoring, range);
    // The input at 16 kHz, and at the stream's device format for Sigil's own conversion.
    let (input16, device_input) = match &c.input {
        Some(path) => {
            let x16 = clip_16k(path)?;
            let dev = to_interleaved(&resample(&x16, 16_000, s.input_rate)?, ch_in);
            (x16, dev)
        }
        None => {
            let dev = match prepared {
                Some(p) => p.input.clone(),
                None => render_input(&spec.input, s.input_rate, ch_in, spec.seed)?,
            };
            (resample(&channel(&dev, ch_in, 0), s.input_rate, 16_000)?, dev)
        }
    };
    let ia = InputAnalysis::new(input16, c.semitones, &mut sc)?;
    let reference = match &c.scoring.speaker_reference {
        Some(p) => Some(clip_16k(p)?),
        None => None,
    };
    let against = Against {
        input: &ia,
        reference: reference.as_deref(),
        max_offset_s: c.max_offset_ms / 1000.0,
    };
    let mut rows = Vec::new();
    let mut outcome = RunOutcome::default();
    let mut notes = sc.notes.clone();
    if c.sigil.include {
        let p = prepared.ok_or("sigil.include needs the spec's engine")?;
        progress(0.05, "converting with Sigil");
        let run = OfflineRun {
            name: c.name.clone(),
            repeat: 1,
            thresholds: Default::default(),
        };
        let offline = run_offline(&OfflineInput {
            stream: s,
            run: &run,
            engine: &p.engine,
            input: &device_input,
            probe: p.probe,
            record_audio: true,
            cancel,
            progress: &|f, m| progress(0.05 + 0.4 * f, m),
            recorder: None,
        })?;
        let (audio, rate, ch) = offline.audio.ok_or("Sigil's conversion produced no audio")?;
        let out16 = resample(&channel(&audio, ch, 0), rate, 16_000)?;
        let latency = match c.sigil.latency_ms {
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
        };
        let mut row = score_candidate("sigil", Path::new(""), &out16, latency, &against, &mut sc);
        row.wav = PathBuf::from(format!("{}.output.wav", c.name));
        rows.push(row);
        if spec.output.write_audio && write_files {
            outcome.audio = Some((audio, rate, ch));
        }
    }
    let n = c.candidates.len().max(1) as f64;
    for (i, cand) in c.candidates.iter().enumerate() {
        progress(0.5 + 0.5 * i as f64 / n, &format!("scoring {}", cand.name));
        let row = match clip_16k(&cand.wav) {
            Ok(x) => score_candidate(&cand.name, &cand.wav, &x, cand.latency_ms, &against, &mut sc),
            Err(e) => CompareRow {
                name: cand.name.clone(),
                wav: cand.wav.clone(),
                latency_ms: cand.latency_ms,
                notes: vec![e],
                ..CompareRow::default()
            },
        };
        rows.push(row);
    }
    outcome.metrics = compare_metrics(&rows);
    outcome.details = serde_json::json!({
        "rows": rows,
        "semitones": c.semitones,
        "scorers": sc.describe(),
        "input_speech_s": (ia.region.1 - ia.region.0) as f64 / 16_000.0,
        "notes": notes,
    });
    Ok(outcome)
}

/// Sigil's row and the T6 margins (see the module docs).
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
        let _ = std::fs::remove_dir_all(&dir);
    }
}
