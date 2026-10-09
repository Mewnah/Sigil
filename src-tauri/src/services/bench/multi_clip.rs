//! Multi-clip scoring and re-scoring (P3B-B, ADR 0009): every local clip (the sherpa-onnx clips and
//! the transcribe samples) with the mean and a bootstrap CI (single-clip rows flagged), and the
//! `rescore` run mode, which scores saved stream / ceiling WAV pairs with the current scorer so any
//! build's output (`exe-s0` included) is compared with one scorer.
//!
//! A `rescore` pair is placed on its run's timeline from the `report.json` next to its stream WAV
//! (`<run>.output.wav` / `<run>.output_o<i>.wav`, the run named `<run>`): the v1 alignment lag and
//! region of a P3 report, or the offset's region, alignment and call log of a v2 report. Without a
//! report the stream is aligned by search (waveforms, then envelopes). The v1 metrics then reproduce
//! the report's numbers wherever their definition is unchanged (same lag, region and scorer files;
//! `details.pairs[].v1_check` lists the differences); clicks need the call log, which P3 reports do
//! not keep, so a P3 pair reports artifact events and their phase, not `clicks`. With
//! `scoring.analyzer: v2` the v2 metrics are added, and the target-speaker similarity uses the
//! other clips' ceilings as the set (leave one clip out; at least two clips).
//!
//! A clip is a report's run: the pairs of one v2 run's offsets (`<run>.output_o<i>.wav` with its
//! `<run>.ceiling[_<mode>]_o<i>.wav`) are one clip, and a pair without a report is its own clip.
//! The run's metrics are the corpus mean (and bootstrap CI, `details.aggregate`) over the clips of
//! each clip's median over its pairs (`details.clips`), as a v2 quality run takes the median over
//! its offsets; `details.pairs` keeps every pair's own scores.
//!
//! # Frozen by P3B-S0 (signatures; P3B-B implements)
//! - [`run_rescore`]`(run, spec, cancel, progress) -> Result<RunOutcome, String>`: the `rescore`
//!   run mode.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use super::ceiling::CallSpan;
use super::fidelity::{
    clicks_v2, clip_16k, offset_corrected_mcd, pair_v2, score_pair, JoinGeometry, PairInput, PairWants, Scorers, StreamTimeline,
};
use super::metrics::Metrics;
use super::quality::QUALITY_V2_SPREAD;
use super::report::RunOutcome;
use super::signals::{channel, read_wav, resample};
use super::spec::{BenchSpec, RescoreRun, StreamSpec};
use crate::services::neural_audio::vc::engine::options::F0Range;

/// Medians over `rows` of every metric any row has, and the spread (max − min) of the
/// `spread_names` that at least two rows have, as `<name>_spread`.
pub(crate) fn median_over(rows: &[Metrics], spread_names: &[&str]) -> (Metrics, Metrics) {
    let keys: BTreeSet<&String> = rows.iter().flat_map(|r| r.keys()).collect();
    let mut med = Metrics::new();
    let mut spread = Metrics::new();
    for k in keys {
        let mut v: Vec<f64> = rows.iter().filter_map(|r| r.get(k).copied()).filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            continue;
        }
        v.sort_by(f64::total_cmp);
        let n = v.len();
        let m = if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) };
        med.insert(k.clone(), m);
        if n >= 2 && spread_names.contains(&k.as_str()) {
            spread.insert(format!("{k}_spread"), v[n - 1] - v[0]);
        }
    }
    (med, spread)
}

/// Speech frames (10 ms) a target-set member needs: 3 s.
const MIN_SET_SPEECH_FRAMES: f64 = 300.0;

/// Resamples of [`mean_ci`].
pub const BOOTSTRAP_RESAMPLES: usize = 10_000;
/// Seed of [`mean_ci`] (fixed: the same values always give the same interval).
const BOOTSTRAP_SEED: u64 = 0x2026_1001;

/// The mean of `values` and its percentile-bootstrap 95 % confidence interval
/// ([`BOOTSTRAP_RESAMPLES`] resamples, a fixed seed); `None` when empty. With one value the interval
/// is that value (and the row is flagged as a single clip by the caller).
pub(crate) fn mean_ci(values: &[f64]) -> Option<(f64, f64, f64)> {
    let v: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return None;
    }
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    if v.len() == 1 {
        return Some((mean, mean, mean));
    }
    let mut rng = super::signals::SplitMix64::new(BOOTSTRAP_SEED);
    let n = v.len();
    let mut means: Vec<f64> = (0..BOOTSTRAP_RESAMPLES)
        .map(|_| (0..n).map(|_| v[((rng.unit() * n as f64) as usize).min(n - 1)]).sum::<f64>() / n as f64)
        .collect();
    means.sort_by(f64::total_cmp);
    let q = |p: f64| means[((means.len() - 1) as f64 * p).round() as usize];
    Some((mean, q(0.025), q(0.975)))
}

/// Means of every metric over `rows`, with `details` holding each metric's mean, 95 % CI and count.
pub(crate) fn aggregate(rows: &[Metrics]) -> (Metrics, serde_json::Value) {
    let keys: BTreeSet<&String> = rows.iter().flat_map(|r| r.keys()).collect();
    let mut means = Metrics::new();
    let mut detail = serde_json::Map::new();
    for k in keys {
        let v: Vec<f64> = rows.iter().filter_map(|r| r.get(k).copied()).collect();
        if let Some((m, lo, hi)) = mean_ci(&v) {
            means.insert(k.clone(), m);
            detail.insert(k.clone(), serde_json::json!({"mean": m, "ci95": [lo, hi], "n": v.len(), "single_clip": v.len() == 1}));
        }
    }
    (means, serde_json::Value::Object(detail))
}

/// How a pair sits on its run's timeline (from the report next to it).
#[derive(Debug, Default)]
struct PairTimeline {
    /// Stream minus ceiling (16 kHz samples; v1's `alignment.lag`).
    lag: Option<f64>,
    /// The call log (v2 reports), in ceiling indices.
    calls: Vec<CallSpan>,
    /// The join geometry the run scored its clicks and offset-corrected MCD with (its SOLA search
    /// window as the harness really had it), when the report has it.
    geometry: Option<JoinGeometry>,
    /// The report's v1 metrics (the reproduction check).
    v1: Metrics,
    stream: Option<StreamSpec>,
    model_delay_ms: f64,
    output_rate: Option<u32>,
    block_ms: Option<f64>,
    report: Option<PathBuf>,
    /// The run of the report the stream belongs to.
    run: Option<String>,
}

/// Which of a v2 offset's ceilings a WAV named `<run>.ceiling[_<mode>][_o<i>].wav` is (its mode,
/// `None` = the primary).
fn ceiling_of(run: &str, file: &str, offset_index: usize) -> Option<Option<String>> {
    let stem = file.strip_prefix(&format!("{run}.ceiling"))?.strip_suffix(".wav")?;
    let stem = if offset_index > 0 { stem.strip_suffix(&format!("_o{offset_index}"))? } else { stem };
    match stem {
        "" => Some(None),
        s => s.strip_prefix('_').map(|m| Some(m.to_string())),
    }
}

/// `<dir>/report.json`'s run for a stream WAV named `<run>.output[_o<i>].wav` and its ceiling WAV.
fn pair_timeline(stream_wav: &Path, ceiling_wav: &Path) -> PairTimeline {
    let mut t = PairTimeline::default();
    let Some(dir) = stream_wav.parent() else {
        return t;
    };
    let report = dir.join("report.json");
    let Some(rep) = std::fs::read(&report).ok().and_then(|b| serde_json::from_slice::<serde_json::Value>(&b).ok()) else {
        return t;
    };
    let file = stream_wav.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    let Some(runs) = rep.get("runs").and_then(|r| r.as_array()) else {
        return t;
    };
    let mut offset_index = 0usize;
    let run = runs.iter().find(|r| {
        let Some(name) = r.get("name").and_then(|n| n.as_str()) else {
            return false;
        };
        if file == format!("{name}.output.wav") {
            return true;
        }
        if let Some(rest) = file.strip_prefix(&format!("{name}.output_o")).and_then(|r| r.strip_suffix(".wav")) {
            if let Ok(i) = rest.parse::<usize>() {
                offset_index = i;
                return true;
            }
        }
        false
    });
    let Some(run) = run else {
        return t;
    };
    t.report = Some(report);
    t.run = run.get("name").and_then(|n| n.as_str()).map(str::to_string);
    t.stream = rep.pointer("/spec/stream").and_then(|s| serde_json::from_value(s.clone()).ok());
    t.model_delay_ms = rep.pointer("/engine/model_delay_ms").and_then(|v| v.as_f64()).unwrap_or(0.0);
    t.output_rate = rep.pointer("/engine/output_rate").and_then(|v| v.as_u64()).map(|v| v as u32);
    t.block_ms = rep.pointer("/spec/engine/block_ms").and_then(|v| v.as_f64()).filter(|v| *v > 0.0);
    let details = run.get("details").cloned().unwrap_or_default();
    let run_name = run.get("name").and_then(|n| n.as_str()).unwrap_or_default();
    let ceiling_file = ceiling_wav.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default();
    match details.get("offsets").and_then(|o| o.as_array()).and_then(|o| o.get(offset_index)) {
        Some(off) => {
            // A v2 report: the offset's alignment (stream minus the primary ceiling) and calls,
            // moved onto the named ceiling's indices.
            let ceilings = off.get("ceilings").and_then(|c| c.as_array()).cloned().unwrap_or_default();
            let start_of = |i: usize| ceilings.get(i).and_then(|c| c.get("start_16k")).and_then(|v| v.as_u64());
            let primary = start_of(0).unwrap_or(0);
            let start = match ceiling_of(run_name, &ceiling_file, offset_index) {
                Some(Some(mode)) => ceilings
                    .iter()
                    .position(|c| c.get("mode").and_then(|m| m.as_str()) == Some(mode.as_str()))
                    .and_then(start_of)
                    .unwrap_or(primary),
                _ => primary,
            };
            t.lag = off
                .pointer("/alignment/lag")
                .and_then(|v| v.as_f64())
                .map(|lag| lag - primary as f64 + start as f64);
            t.calls = off
                .get("calls")
                .and_then(|c| c.as_array())
                .map(|c| {
                    c.iter()
                        .filter_map(|x| {
                            let x = x.as_array()?;
                            let (s0, e, r) = (x.first()?.as_u64()?, x.get(1)?.as_u64()?, x.get(2)?.as_u64()?);
                            Some(CallSpan {
                                start_16k: s0.saturating_sub(start),
                                end_16k: e.saturating_sub(start),
                                returned_16k: r as usize,
                                sola_offset_16k: x.get(3).and_then(|v| v.as_f64()),
                                join_ncc: None,
                                // Reports before the flag recorded `None` for every join without an
                                // offset, and the offset-corrected MCD read it as "no displacement".
                                join_seen: x.get(4).and_then(|v| v.as_bool()).unwrap_or(true),
                            })
                        })
                        .collect()
                })
                .unwrap_or_default();
            t.geometry = off.get("join_geometry").and_then(|g| serde_json::from_value::<JoinGeometry>(g.clone()).ok());
            if let Some(mm) = off.get("metrics").and_then(|m| serde_json::from_value::<Metrics>(m.clone()).ok()) {
                t.v1 = mm;
            }
        }
        None => {
            // A v1 (P3) report.
            t.lag = details.pointer("/alignment/lag").and_then(|v| v.as_f64());
            if let Some(mm) = run.get("metrics").and_then(|m| serde_json::from_value::<Metrics>(m.clone()).ok()) {
                t.v1 = mm;
            }
        }
    }
    t
}

/// One scored pair.
struct PairRow {
    name: String,
    /// The clip the pair belongs to: its report's folder and run (the offsets of one v2 run are
    /// one clip), or the pair's own name without a report.
    group: String,
    metrics: Metrics,
    details: serde_json::Value,
    /// The ceiling's and the stream's region (16 kHz), for the leave-one-out target set.
    ceiling_region: Vec<f32>,
    stream_region: Vec<f32>,
}

/// The `rescore` run mode (see the module docs).
pub(super) fn run_rescore(run: &RescoreRun, _spec: &BenchSpec, cancel: &AtomicBool, progress: &(dyn Fn(f64, &str) + Sync)) -> Result<RunOutcome, String> {
    if run.pairs.is_empty() {
        return Err(format!("run '{}': rescore needs at least one pair", run.name));
    }
    let opts = super::fidelity::analyzer_opts(&run.scoring.analyzer)?;
    if opts.ablations {
        return Err("rescore: ablations need the engine (a quality run), not saved WAVs".into());
    }
    let mut sc = Scorers::load(&run.scoring, F0Range::Speech);
    let scorers = if opts.v2 { sc.describe_v2() } else { sc.describe() };
    let mut rows: Vec<PairRow> = Vec::new();
    let mut notes = sc.notes.clone();
    for (i, pair) in run.pairs.iter().enumerate() {
        if cancel.load(Ordering::Relaxed) {
            return Err("cancelled".into());
        }
        progress(i as f64 / run.pairs.len() as f64, &format!("scoring {}", pair.name));
        match score_saved_pair(&mut sc, pair, opts.v2) {
            Ok(r) => rows.push(r),
            Err(e) => notes.push(format!("{}: {e}", pair.name)),
        }
    }
    if rows.is_empty() {
        return Err(format!("rescore: no pair could be scored ({})", notes.join("; ")));
    }
    let groups = clip_groups(&rows);
    // Target-speaker similarity, leave one clip out: the other clips' ceilings are the set.
    if opts.v2 && groups.len() >= 2 {
        // Set members: each ceiling's own speech frames, at least 3 s of them.
        let embeds: Vec<Option<Vec<f32>>> = rows
            .iter()
            .map(|r| {
                let sp = sc.speaker_v2.as_mut()?;
                let mask = super::scorers::speech_frame_mask(&r.ceiling_region, super::scorers::SPEAKER_SPEECH_RANGE_DB);
                if (mask.iter().filter(|&&m| m).count() as f64) < MIN_SET_SPEECH_FRAMES {
                    return None;
                }
                sp.embed_masked(&r.ceiling_region, &mask).ok()
            })
            .collect();
        let sims: Vec<(Option<f64>, Option<f64>)> = (0..rows.len())
            .map(|i| match super::scorers::centroid(&set_members(&rows, &embeds, i)) {
                Some(centroid) => set_sims(&mut sc, &rows[i], &centroid),
                None => (None, None),
            })
            .collect();
        for (r, (stream_sim, ceiling_sim)) in rows.iter_mut().zip(sims) {
            if let (Some(s), Some(c)) = (stream_sim, ceiling_sim) {
                r.metrics.insert("speaker_set_sim".into(), s);
                r.metrics.insert("ceiling_speaker_set_sim".into(), c);
                r.metrics.insert("speaker_set_sim_delta".into(), s - c);
            }
        }
    } else if opts.v2 {
        notes.push("target set: one clip, so no other ceilings to build a set from (leave one clip out)".into());
    }
    // One row per clip: the median over its pairs (the offsets of a v2 run), then the corpus mean
    // and its CI over the clips (offsets of one clip are not independent samples).
    let clip_rows: Vec<(String, Metrics)> = groups
        .iter()
        .map(|(g, idx)| {
            let rs: Vec<Metrics> = idx.iter().map(|&i| rows[i].metrics.clone()).collect();
            let (mut med, spread) = median_over(&rs, QUALITY_V2_SPREAD);
            med.extend(spread);
            if idx.len() > 1 {
                med.insert("pairs".into(), idx.len() as f64);
            }
            (g.clone(), med)
        })
        .collect();
    let metric_rows: Vec<Metrics> = clip_rows.iter().map(|(_, m)| m.clone()).collect();
    let (mut metrics, aggregate) = aggregate(&metric_rows);
    metrics.remove("pairs");
    metrics.insert("clips".into(), clip_rows.len() as f64);
    metrics.insert("pairs".into(), rows.len() as f64);
    let details = serde_json::json!({
        "scorer": if opts.v2 { "v2" } else { "v1" },
        "scorers": scorers,
        "single_clip": clip_rows.len() == 1,
        "aggregate": aggregate,
        "clips": clip_rows.iter().map(|(g, m)| serde_json::json!({"clip": g, "metrics": m})).collect::<Vec<_>>(),
        "pairs": rows.iter().map(|r| serde_json::json!({"name": r.name, "clip": r.group, "metrics": r.metrics, "details": r.details})).collect::<Vec<_>>(),
        "notes": notes,
    });
    progress(1.0, "done");
    Ok(RunOutcome {
        metrics,
        details,
        ..RunOutcome::default()
    })
}

/// The pairs of every clip ([`PairRow::group`]), in first-seen order: `(clip, pair indices)`.
fn clip_groups(rows: &[PairRow]) -> Vec<(String, Vec<usize>)> {
    let mut out: Vec<(String, Vec<usize>)> = Vec::new();
    for (i, r) in rows.iter().enumerate() {
        match out.iter_mut().find(|(g, _)| *g == r.group) {
            Some((_, idx)) => idx.push(i),
            None => out.push((r.group.clone(), vec![i])),
        }
    }
    out
}

/// The target set of pair `i`, leave one clip out: the member embeddings of every pair of another
/// clip (a clip's other offsets render the same speech, so they would favour its own ceiling).
fn set_members(rows: &[PairRow], embeds: &[Option<Vec<f32>>], i: usize) -> Vec<Vec<f32>> {
    rows.iter()
        .zip(embeds)
        .filter(|(r, _)| r.group != rows[i].group)
        .filter_map(|(_, e)| e.clone())
        .collect()
}

/// SIM(stream, set) and SIM(ceiling, set) of a pair, on the ceiling's speech frames.
fn set_sims(sc: &mut Scorers, row: &PairRow, centroid: &[f32]) -> (Option<f64>, Option<f64>) {
    let Some(sp) = sc.speaker_v2.as_mut() else {
        return (None, None);
    };
    let mask = super::scorers::speech_frame_mask(&row.ceiling_region, super::scorers::SPEAKER_SPEECH_RANGE_DB);
    let s = sp.embed_masked(&row.stream_region, &mask).ok().map(|e| super::scorers::cosine(&e, centroid));
    let c = sp.embed_masked(&row.ceiling_region, &mask).ok().map(|e| super::scorers::cosine(&e, centroid));
    (s, c)
}

/// Score one saved pair (see the module docs).
fn score_saved_pair(sc: &mut Scorers, pair: &super::spec::RescorePair, v2: bool) -> Result<PairRow, String> {
    let (s, s_rate, s_ch) = read_wav(&pair.stream)?;
    let stream = channel(&s, s_ch, 0);
    let (c, c_rate, c_ch) = read_wav(&pair.ceiling)?;
    let ceiling = channel(&c, c_ch, 0);
    let t = pair_timeline(&pair.stream, &pair.ceiling);
    let mut notes = Vec::new();
    if t.report.is_none() {
        notes.push("no report.json next to the stream WAV: aligned by search".to_string());
    }
    let sspec = t.stream.clone().unwrap_or_default();
    let out_rate = t.output_rate.unwrap_or(c_rate);
    let block_ms = t.block_ms.unwrap_or(40.0);
    let stream_period = (block_ms / 1000.0 * f64::from(s_rate)).round() as usize;
    let score = score_pair(
        sc,
        &PairInput {
            stream: &stream,
            stream_rate: s_rate,
            ceiling: &ceiling,
            ceiling_rate: c_rate,
            ceiling_start_16k: 0,
            calls: &t.calls,
            stream_period,
            ceiling_period: (0.2 * f64::from(c_rate)).round() as usize,
            join_window_ms: sspec.crossfade_ms + 5.0,
            max_offset_s: 10.0,
            expected_lag_16k: t.lag,
        },
    )?;
    let mut m = Metrics::new();
    score.quality.insert_into("", &mut m);
    if let Some(v) = score.ceiling_dnsmos_ovrl {
        m.insert("ceiling_dnsmos_ovrl".into(), v);
        if let Some(o) = score.quality.dnsmos_ovrl {
            m.insert("dnsmos_ovrl_delta".into(), o - v);
        }
    }
    if let Some(v) = score.ceiling_seam_flux_ratio {
        m.insert("ceiling_seam_flux_ratio".into(), v);
    }
    if t.calls.is_empty() {
        // The join positions are unknown: the joins-based statistic is not measured.
        m.remove("clicks");
        notes.push("clicks: the report keeps no call log (P3 reports); artifact events and their phase only".into());
    } else if let Some(c) = score.quality.clicks {
        if let Some(secs) = score.quality.seconds {
            m.insert("clicks_per_10min".into(), c as f64 / secs.max(1e-9) * 600.0);
        }
    }
    // The v1 reproduction check against the report's own numbers.
    let mut check = serde_json::Map::new();
    for k in ["mcd_db", "f0_rmse_cents", "gross_pitch_error", "vuv_error", "speaker_cosine", "dnsmos_ovrl", "seam_flux_ratio", "artifact_events"] {
        if let (Some(a), Some(b)) = (m.get(k), t.v1.get(k)) {
            check.insert(k.into(), serde_json::json!({"rescored": a, "report": b, "abs_diff": (a - b).abs()}));
        }
    }
    // Region in ceiling indices (that pair's absolute timeline starts at the ceiling's first sample).
    let lag = score.alignment.lag.round() as i64;
    let (a, b) = (score.region.0 as u64, score.region.1 as u64);
    let c16 = resample(&ceiling, c_rate, 16_000)?;
    let timeline = StreamTimeline {
        s16: resample(&stream, s_rate, 16_000)?,
        shift: lag,
    };
    let cs = c16.get(a as usize..b as usize).ok_or("the region is outside the ceiling")?.to_vec();
    let ss = timeline.span(a, b).ok_or("the region is outside the stream")?.to_vec();
    let mut details = serde_json::json!({
        "stream": pair.stream.display().to_string(),
        "ceiling": pair.ceiling.display().to_string(),
        "report": t.report.as_ref().map(|p| p.display().to_string()),
        "alignment": score.alignment,
        "region_16k": [a, b],
        "artifact_phase": score.artifact_phase,
        "v1_check": check,
    });
    if v2 {
        let pv = pair_v2(sc, &cs, &ss, PairWants { speaker: true, dnsmos: false, set: None }, &mut notes);
        let mut put = |k: &str, v: Option<f64>| {
            if let Some(v) = v.filter(|v| v.is_finite()) {
                m.insert(k.to_string(), v);
            }
        };
        put("env_mcd_db", pv.env_mcd_db);
        put("speaker_cosine_v2", pv.speaker_cosine_v2);
        put("vuv_error_tolerant", pv.vuv_error_tolerant);
        if let Some(f) = &pv.f0 {
            put("f0_fine_rmse_cents", f.fine_rmse_cents);
            put("f0_gross_frames", Some(f.gross_frames as f64));
        }
        if !t.calls.is_empty() {
            // The run's own geometry (the harness's real search window, e.g. under
            // SIGIL_VC_SOLA_SEARCH_MS or an aligned slack); the formula's only for reports without it.
            let geo = match t.geometry {
                Some(g) => g,
                None => {
                    notes.push("join geometry: the report has none, so the harness's default SOLA window is assumed".into());
                    JoinGeometry::of(&sspec, out_rate, t.model_delay_ms)
                }
            };
            let cl = clicks_v2(&score.artifacts, &t.calls, &geo, a as f64, b as f64, sspec.crossfade_ms + 5.0, sc.click.tolerance_ms);
            put("clicks_binomial", Some(cl.clicks_wide as f64));
            put("clicks_narrow", Some(cl.clicks_narrow as f64));
            let fc = sc.f0.contour(&cs).ok();
            if let Some(oc) = offset_corrected_mcd(&timeline, &cs, a, b, &t.calls, &geo, fc.as_deref()) {
                put("mcd_offset_corrected_db", Some(oc.mcd_db));
                put("env_mcd_offset_corrected_db", oc.env_mcd_db);
                if let Some(o) = details.as_object_mut() {
                    o.insert("offset_corrected".into(), serde_json::to_value(oc).unwrap_or_default());
                }
            }
            if let Some(o) = details.as_object_mut() {
                o.insert("clicks_v2".into(), serde_json::to_value(&cl).unwrap_or_default());
            }
        }
        if let Some(o) = details.as_object_mut() {
            o.insert(
                "f0_gross".into(),
                serde_json::to_value(pv.f0.as_ref().map(|f| f.gross.clone())).unwrap_or_default(),
            );
        }
    }
    if let Some(input) = &pair.input {
        // The input is only named for the record: v2 rescoring scores the renditions against each
        // other (the fed F0 needs the engine).
        let _ = clip_16k(input).map_err(|e| notes.push(format!("input: {e}")));
    }
    if let Some(o) = details.as_object_mut() {
        o.insert("notes".into(), notes.into());
    }
    let group = match (&t.report, &t.run) {
        (Some(report), Some(run)) => format!("{}#{run}", report.parent().unwrap_or(report).display()),
        _ => pair.name.clone(),
    };
    Ok(PairRow {
        name: pair.name.clone(),
        group,
        metrics: m,
        details,
        ceiling_region: cs,
        stream_region: ss,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn medians_and_spreads_over_offsets() {
        let row = |a: f64, b: Option<f64>| {
            let mut m = Metrics::new();
            m.insert("env_mcd_db".into(), a);
            if let Some(b) = b {
                m.insert("speaker_cosine_v2".into(), b);
            }
            m
        };
        let (med, spread) = median_over(&[row(2.0, Some(0.9)), row(1.0, None), row(4.0, Some(0.94))], &["env_mcd_db", "speaker_cosine_v2"]);
        assert_eq!(med["env_mcd_db"], 2.0);
        assert!((med["speaker_cosine_v2"] - 0.92).abs() < 1e-12, "the median of the rows that have it");
        assert_eq!(spread["env_mcd_db_spread"], 3.0);
        assert!((spread["speaker_cosine_v2_spread"] - 0.04).abs() < 1e-12);
        let (_, none) = median_over(&[row(1.0, None)], &["env_mcd_db"]);
        assert!(none.is_empty(), "one row has no spread");
    }

    /// The offsets of one run are one clip: grouped together, and left out of each other's target
    /// set (another offset's ceiling renders the same speech).
    #[test]
    fn a_clips_offsets_group_together_and_leave_the_set_together() {
        let row = |name: &str, group: &str| PairRow {
            name: name.into(),
            group: group.into(),
            metrics: Metrics::new(),
            details: serde_json::Value::Null,
            ceiling_region: Vec::new(),
            stream_region: Vec::new(),
        };
        let rows = [row("a0", "A#fidelity"), row("b0", "B#fidelity"), row("a1", "A#fidelity"), row("c0", "C#fidelity")];
        let groups = clip_groups(&rows);
        assert_eq!(groups, vec![("A#fidelity".to_string(), vec![0, 2]), ("B#fidelity".to_string(), vec![1]), ("C#fidelity".to_string(), vec![3])]);
        let embeds = vec![Some(vec![1.0f32]), Some(vec![2.0]), Some(vec![3.0]), None];
        // A's pairs see B's member only (C has too little speech to be a member).
        assert_eq!(set_members(&rows, &embeds, 0), vec![vec![2.0f32]]);
        assert_eq!(set_members(&rows, &embeds, 2), vec![vec![2.0f32]]);
        // B sees both of A's offsets.
        assert_eq!(set_members(&rows, &embeds, 1), vec![vec![1.0f32], vec![3.0]]);
        assert_eq!(set_members(&rows, &embeds, 3), vec![vec![1.0f32], vec![2.0], vec![3.0]]);
    }

    /// A rescore reads back what the run measured with: the join geometry it scored its clicks and
    /// offset-corrected MCD with (a non-default SOLA window must not be re-derived from the
    /// default formula), and which calls' joins the stream loop observed. A report from before the
    /// flag keeps the old reading (a join without an offset was "at the centre").
    #[test]
    fn a_rescore_reads_the_runs_join_geometry_and_observed_joins() {
        let dir = std::env::temp_dir().join(format!("sigil-pair-timeline-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let report = |calls: serde_json::Value, geometry: serde_json::Value| {
            serde_json::json!({
                "spec": {"stream": {"crossfade_ms": 10}},
                "engine": {"model_delay_ms": 0.0, "output_rate": 40000},
                "runs": [{
                    "name": "fid",
                    "details": {"offsets": [{
                        "alignment": {"lag": 1234.0},
                        "ceilings": [{"mode": "long_shape", "start_16k": 0}],
                        "calls": calls,
                        "join_geometry": geometry,
                        "metrics": {}
                    }]}
                }]
            })
        };
        let write = |v: &serde_json::Value| std::fs::write(dir.join("report.json"), serde_json::to_vec(v).unwrap()).unwrap();
        let (stream, ceiling) = (dir.join("fid.output.wav"), dir.join("fid.ceiling.wav"));
        // A run whose SOLA window was not the default formula's (centre 12.5 samples, not 80.4).
        write(&report(
            serde_json::json!([[0, 1121, 1121, 3.5, true], [640, 1761, 1121, null, true], [1280, 2401, 1121, null, false]]),
            serde_json::json!({"delay_16k": 320.0, "centre_16k": 12.5, "overlap_16k": 160.0, "tail_16k": 481.0}),
        ));
        let t = pair_timeline(&stream, &ceiling);
        let g = t.geometry.expect("the run's join geometry");
        assert_eq!((g.delay_16k, g.centre_16k, g.overlap_16k, g.tail_16k), (320.0, 12.5, 160.0, 481.0));
        assert_ne!(g.centre_16k, JoinGeometry::of(&StreamSpec::default(), 40_000, 20.0).centre_16k);
        let seen: Vec<(bool, Option<f64>)> = t.calls.iter().map(|c| (c.join_seen, c.sola_offset_16k)).collect();
        assert_eq!(seen, vec![(true, Some(3.5)), (true, None), (false, None)]);
        // A report from before the flag and the geometry: no geometry, every call read as seen.
        write(&report(serde_json::json!([[0, 1121, 1121, null], [640, 1761, 1121, 2.0]]), serde_json::Value::Null));
        let t = pair_timeline(&stream, &ceiling);
        assert!(t.geometry.is_none());
        assert!(t.calls.iter().all(|c| c.join_seen), "{:?}", t.calls);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn ceiling_wav_names_map_to_their_ceilings() {
        assert_eq!(ceiling_of("fidelity", "fidelity.ceiling.wav", 0), Some(None));
        assert_eq!(ceiling_of("fidelity", "fidelity.ceiling_whole.wav", 0), Some(Some("whole".into())));
        assert_eq!(ceiling_of("fidelity", "fidelity.ceiling_long_shape_o2.wav", 2), Some(Some("long_shape".into())));
        assert_eq!(ceiling_of("fidelity", "fidelity.ceiling_o1.wav", 1), Some(None));
        assert_eq!(ceiling_of("fidelity", "other.ceiling.wav", 0), None);
    }

    #[test]
    fn bootstrap_ci_is_deterministic_and_brackets_the_mean() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (m, lo, hi) = mean_ci(&v).unwrap();
        assert_eq!(m, 4.5);
        assert!(lo < m && m < hi, "{lo} {m} {hi}");
        assert!(lo > 2.5 && hi < 6.5, "{lo} {hi}");
        assert_eq!(mean_ci(&v), Some((m, lo, hi)), "a fixed seed");
        assert_eq!(mean_ci(&[3.0]), Some((3.0, 3.0, 3.0)));
        assert_eq!(mean_ci(&[f64::NAN]), None);
        let mut a = Metrics::new();
        a.insert("x".into(), 1.0);
        let (means, detail) = aggregate(&[a]);
        assert_eq!(means["x"], 1.0);
        assert_eq!(detail["x"]["single_clip"], true);
    }
}
