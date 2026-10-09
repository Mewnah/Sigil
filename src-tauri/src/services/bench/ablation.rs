//! Oracle ablations of the fidelity scorer v2 (P3B-B, ADR 0009): through the engine's probe hooks
//! (`vc::engine::probe`, bench-only and spec-gated: `scoring.analyzer: v2+ablations`) the stream is
//! re-run with (i) oracle F0, (ii) oracle content, (iii) both, (iv) the stream's F0 injected into
//! the ceiling and (v) oracle F0 plus oracle content in both the stream and the long-shape ceiling
//! (the residual: the right-context loss of `enc`, flow and decoder plus the joins), so the T5 gap
//! can be split by cause.
//!
//! The oracles are the whole-utterance analysis of the run's first offset (`ceiling::render_whole`:
//! content features of the reflect-padded input region on the stream's 320-sample content grid,
//! whole-clip F0 of the run's method), restricted to the input region so the lead silence keeps the
//! stream's own features. Every rendition is scored over the run's region against the long-shape
//! ceiling (A) and the whole-utterance ceiling: STFT MCD, envelope MCD, exact CAM++ cosine and the
//! output analyzer's V/UV and fine pitch error. "Recovered" is how much of the baseline distance an
//! ablation removes (positive = closer to that ceiling).
//!
//! Against the whole ceiling, (iii) and (v) coincide: its own content and F0 are the oracles.
//!
//! An ablation only means something if its overrides reached the generator, so every row checks the
//! probe's counters ([`unapplied`]): an oracle that replaced no frame (or whose content grid did not
//! line up with the engine's 320-sample grid) makes the row read "recovered +0.00" with nothing
//! behind it. The row then says so (`problems`), the table repeats it in its notes, and the run
//! counts the rows in `ablation_rows_unapplied` (0 = every oracle was applied).
//!
//! # Frozen by P3B-S0 (signatures; P3B-B implements)
//! - [`Ablation`] (the five variants) with [`Ablation::ALL`] and [`Ablation::as_str`].

use std::collections::BTreeMap;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

use serde::Serialize;

use super::ceiling::{long_shape, render_whole, LongShapeOpts, OfflineAnalysis, OfflineVoice, RenderOpts, StreamRecorder, CONTENT_HOP_16K};
use super::engine::Engine;
use super::fidelity::{offline_voice, pair_v2, stream_pass, CeilingMode, PairV2, PairWants, Rendition, Scorers, StreamTimeline};
use super::rvc_build::RvcPlan;
use super::spec::{BenchSpec, QualityRun};
use crate::services::neural_audio::vc::engine::probe::{OracleContent, OracleF0, ProbeCounts, ProbeSetup};

/// One oracle ablation (see the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ablation {
    OracleF0,
    OracleContent,
    OracleBoth,
    StreamF0InCeiling,
    Residual,
}

impl Ablation {
    pub const ALL: [Ablation; 5] = [
        Ablation::OracleF0,
        Ablation::OracleContent,
        Ablation::OracleBoth,
        Ablation::StreamF0InCeiling,
        Ablation::Residual,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            Ablation::OracleF0 => "oracle_f0",
            Ablation::OracleContent => "oracle_content",
            Ablation::OracleBoth => "oracle_both",
            Ablation::StreamF0InCeiling => "stream_f0_in_ceiling",
            Ablation::Residual => "residual",
        }
    }

    /// What the variant changes.
    pub fn description(self) -> &'static str {
        match self {
            Ablation::OracleF0 => "(i) the stream decodes with whole-clip F0 of the input",
            Ablation::OracleContent => "(ii) the stream decodes with whole-utterance content features",
            Ablation::OracleBoth => "(iii) the stream decodes with both",
            Ablation::StreamF0InCeiling => "(iv) the ceilings decode with the stream's fed F0",
            Ablation::Residual => "(v) oracle F0 and content in the stream and in ceiling A: what remains is right context and joins",
        }
    }
}

/// What the first offset of a v2 run hands the ablations.
pub(crate) struct AblationBase {
    /// The input as streamed (lead silence included, interleaved at the stream's input format).
    pub input: Vec<f32>,
    pub rec: StreamRecorder,
    /// The scored region (absolute 16 kHz).
    pub region: (u64, u64),
    pub input_start: Option<f64>,
    /// The stream's fed F0 (Hz after the shift).
    pub stream_fed: Option<(u64, Vec<f32>)>,
    pub whole_analysis: Option<OfflineAnalysis>,
    pub whole_region: Option<(u64, u64)>,
    pub grid_phase: u64,
    pub renditions: Vec<(CeilingMode, Rendition)>,
    pub stream: StreamTimeline,
}

/// The run around the ablations.
pub(crate) struct AblationContext<'a> {
    pub spec: &'a BenchSpec,
    pub q: &'a QualityRun,
    pub engine: &'a Engine,
    pub plan: &'a RvcPlan,
    pub render: &'a RenderOpts,
    pub set: Option<&'a [f32]>,
}

/// The scores of one rendition against one ceiling.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct Scores {
    pub seconds: f64,
    pub mcd_db: Option<f64>,
    pub env_mcd_db: Option<f64>,
    pub speaker_cosine_v2: Option<f64>,
    pub vuv_error: Option<f64>,
    pub f0_fine_rmse_cents: Option<f64>,
    /// Similarity to the target-speaker set (when the run has one).
    pub set_sim: Option<f64>,
}

impl Scores {
    fn of(p: &PairV2) -> Scores {
        Scores {
            seconds: p.seconds,
            mcd_db: p.mcd_db,
            env_mcd_db: p.env_mcd_db,
            speaker_cosine_v2: p.speaker_cosine_v2,
            vuv_error: p.f0.as_ref().map(|f| f.v1.vuv_error),
            f0_fine_rmse_cents: p.f0.as_ref().and_then(|f| f.fine_rmse_cents),
            set_sim: p.set_sim_test,
        }
    }

    /// How much closer `self` is than `baseline` (positive = recovered; cosine counts up).
    fn recovered_from(&self, baseline: &Scores) -> Scores {
        let down = |b: Option<f64>, a: Option<f64>| Some(b? - a?);
        Scores {
            seconds: self.seconds,
            mcd_db: down(baseline.mcd_db, self.mcd_db),
            env_mcd_db: down(baseline.env_mcd_db, self.env_mcd_db),
            speaker_cosine_v2: down(self.speaker_cosine_v2, baseline.speaker_cosine_v2),
            vuv_error: down(baseline.vuv_error, self.vuv_error),
            f0_fine_rmse_cents: down(baseline.f0_fine_rmse_cents, self.f0_fine_rmse_cents),
            set_sim: down(self.set_sim, baseline.set_sim),
        }
    }
}

/// One row of the table: the variant scored against each ceiling, and what it recovered.
#[derive(Clone, Debug, Serialize)]
pub(crate) struct AblationRow {
    pub ablation: Ablation,
    pub description: &'static str,
    pub vs: BTreeMap<String, Scores>,
    pub recovered: BTreeMap<String, Scores>,
    /// How often the probe's overrides applied (a sanity check of the plumbing).
    pub applied: String,
    /// What the counters say did not happen (empty = every oracle of the row was applied).
    pub problems: Vec<String>,
}

impl AblationRow {
    fn new(ablation: Ablation, vs: BTreeMap<String, Scores>, recovered: BTreeMap<String, Scores>, applied: ProbeCounts) -> AblationRow {
        AblationRow {
            ablation,
            description: ablation.description(),
            vs,
            recovered,
            applied: counts(applied),
            problems: unapplied(ablation, &applied),
        }
    }
}

/// What the probe's counters say about `abl` that would make its row meaningless: the oracle F0 of
/// (i), (iii), (iv) and (v) must have replaced frames, and so must the oracle content of (ii),
/// (iii) and (v), with no call whose content grid missed the oracle's origin.
pub(crate) fn unapplied(abl: Ablation, c: &ProbeCounts) -> Vec<String> {
    let (wants_f0, wants_content) = match abl {
        Ablation::OracleF0 | Ablation::StreamF0InCeiling => (true, false),
        Ablation::OracleContent => (false, true),
        Ablation::OracleBoth | Ablation::Residual => (true, true),
    };
    let mut out = Vec::new();
    if wants_f0 && c.f0_frames == 0 {
        out.push("the oracle F0 replaced no frame (it never reached the generator)".to_string());
    }
    if wants_content && c.content_frames == 0 {
        out.push("the oracle content replaced no frame (it never reached the generator)".to_string());
    }
    if wants_content && c.content_misaligned > 0 {
        out.push(format!(
            "{} call(s) had a content grid that did not line up with the oracle's origin (a block that is not a multiple of 320 samples?)",
            c.content_misaligned
        ));
    }
    out
}

/// The ablation table of a run (in the report's details).
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct AblationTable {
    /// The stream as streamed, against each ceiling.
    pub baseline: BTreeMap<String, Scores>,
    pub rows: Vec<AblationRow>,
    pub oracles: serde_json::Value,
    pub notes: Vec<String>,
}

impl AblationTable {
    /// Rows whose oracle did not (fully) reach the generator.
    pub fn rows_unapplied(&self) -> usize {
        self.rows.iter().filter(|r| !r.problems.is_empty()).count()
    }
}

/// Restrict the whole-utterance analysis to `[a, b)`: F0 frames inside it, content frames whose
/// whole 400-sample receptive field is inside it.
pub(crate) fn oracles_in(analysis: &OfflineAnalysis, a: u64, b: u64) -> (OracleF0, OracleContent) {
    let f = &analysis.f0;
    let (fa, fb) = (a.div_ceil(160), b / 160);
    let lo = fa.saturating_sub(f.start_frame) as usize;
    let hi = (fb.saturating_sub(f.start_frame) as usize).min(f.hz.len());
    let f0 = OracleF0 {
        start_frame: f.start_frame + lo as u64,
        hz: if hi > lo { f.hz[lo..hi].to_vec() } else { Vec::new() },
    };
    let c = &analysis.content;
    let emb = c.emb.max(1);
    let frames = c.feats.len() / emb;
    let first = (0..frames).find(|&k| c.origin_16k + CONTENT_HOP_16K * k as u64 >= a);
    let last = (0..frames).rev().find(|&k| c.origin_16k + CONTENT_HOP_16K * k as u64 + 400 <= b);
    let content = match (first, last) {
        (Some(i), Some(j)) if j >= i => OracleContent {
            origin_16k: c.origin_16k + CONTENT_HOP_16K * i as u64,
            emb: c.emb,
            feats: c.feats[i * emb..(j + 1) * emb].to_vec(),
        },
        _ => OracleContent {
            origin_16k: c.origin_16k,
            emb: c.emb,
            feats: Vec::new(),
        },
    };
    (f0, content)
}

/// Run the ablations (see the module docs).
pub(crate) fn run_ablations(
    base: &AblationBase,
    ctx: &AblationContext<'_>,
    sc: &mut Scorers,
    offline: &mut Option<OfflineVoice>,
    cancel: &AtomicBool,
) -> Result<AblationTable, String> {
    let s = &ctx.spec.stream;
    let mut notes = Vec::new();
    let (a, b) = base.region;
    // The whole-utterance ceiling and its analysis (rendered now when the run's modes had none).
    let mut ceilings: Vec<(String, Rendition)> = base
        .renditions
        .iter()
        .filter(|(m, _)| matches!(m, CeilingMode::LongShape | CeilingMode::Whole))
        .map(|(m, r)| (m.as_str().to_string(), r.clone()))
        .collect();
    let (analysis, whole_region) = match (&base.whole_analysis, base.whole_region) {
        (Some(an), Some(wr)) => (an.clone(), wr),
        _ => {
            let start = base.input_start.ok_or("the ablations need where the input entered the stream")?;
            let ch = usize::from(s.input_channels).max(1);
            let clip16 = super::signals::resample(&super::signals::to_mono(&base.input, ch), s.input_rate, 16_000)?;
            // The input region of the streamed clip (lead included in `clip16`, so search it whole).
            let region = whole_input_region(&base.rec, &clip16, start)?;
            let voice = offline_voice(offline, ctx.plan, ctx.render)?;
            let o = render_whole(voice, s, &base.rec, region, base.grid_phase, ctx.render)?;
            let an = o.analysis.clone().ok_or("the whole ceiling returned no analysis")?;
            ceilings.push(("whole".into(), Rendition::from_ceiling("whole", o.out)?));
            notes.push("ablations: the whole ceiling was rendered for the oracles (not in ceiling.mode)".into());
            (an, region)
        }
    };
    let (of, oc) = oracles_in(&analysis, whole_region.0, whole_region.1);
    let (of, oc) = (Arc::new(of), Arc::new(oc));
    let score_all = |sc: &mut Scorers, stream: &StreamTimeline, against: &[(String, Rendition)], notes: &mut Vec<String>| -> BTreeMap<String, Scores> {
        let mut out = BTreeMap::new();
        for (name, r) in against {
            let (ea, eb) = r.extent();
            let (sa, sb) = stream.extent();
            let (ra, rb) = (a.max(ea).max(sa), b.min(eb).min(sb));
            if let (Some(c), Some(t)) = (r.span(ra, rb), stream.span(ra, rb)) {
                let p = pair_v2(sc, c, t, PairWants { speaker: true, dnsmos: false, set: ctx.set }, notes);
                out.insert(name.clone(), Scores::of(&p));
            }
        }
        out
    };
    let mut table = AblationTable {
        baseline: score_all(sc, &base.stream, &ceilings, &mut notes),
        oracles: serde_json::json!({
            "region_16k": [whole_region.0, whole_region.1],
            "f0_frames": of.hz.len(),
            "f0_start_frame": of.start_frame,
            "content_frames": oc.frames(),
            "content_origin_16k": oc.origin_16k,
        }),
        ..AblationTable::default()
    };
    let rows_vs = |sc: &mut Scorers, notes: &mut Vec<String>, stream: &StreamTimeline, against: &[(String, Rendition)], baseline: &BTreeMap<String, Scores>| {
        let vs = score_all(sc, stream, against, notes);
        let recovered = vs
            .iter()
            .filter_map(|(k, v)| baseline.get(k).map(|b0| (k.clone(), v.recovered_from(b0))))
            .collect();
        (vs, recovered)
    };
    // The stream re-run with oracles: (i), (ii), (iii).
    let channel = ctx.plan.probe.clone();
    if !channel.attached() {
        return Err("the ablations need the engine probe of a direct template build".into());
    }
    let mut stream_both: Option<StreamTimeline> = None;
    for (abl, f0, content) in [
        (Ablation::OracleF0, true, false),
        (Ablation::OracleContent, false, true),
        (Ablation::OracleBoth, true, true),
    ] {
        let setup = ProbeSetup {
            fed_f0: true,
            oracle_f0: f0.then(|| of.clone()),
            oracle_content: content.then(|| oc.clone()),
            ..ProbeSetup::default()
        };
        let pass = stream_pass(ctx.engine, s, &base.input, Some((&channel, setup)), cancel)?;
        let timeline = StreamTimeline {
            s16: super::signals::resample(&pass.mono(), pass.rate, 16_000)?,
            shift: base.stream.shift,
        };
        if pass.rec.calls().first().map(|c| c.start_16k) != base.rec.calls().first().map(|c| c.start_16k) {
            notes.push(format!("{}: the stream's timeline differs from the baseline pass", abl.as_str()));
        }
        let (vs, recovered) = rows_vs(sc, &mut notes, &timeline, &ceilings, &table.baseline);
        table.rows.push(AblationRow::new(abl, vs, recovered, pass.applied));
        if abl == Ablation::OracleBoth {
            stream_both = Some(timeline);
        }
    }
    // (iv) the stream's fed F0 (before the shift) in ceiling A and in the whole ceiling.
    let shift = 2f32.powf(s.pitch_semitones / 12.0);
    match &base.stream_fed {
        Some((start, hz)) => {
            let stream_f0 = Arc::new(OracleF0 {
                start_frame: *start,
                hz: hz.iter().map(|v| v / shift).collect(),
            });
            let a_iv = long_shape(ctx.plan, s, &base.rec, &ctx.q.ceiling, &LongShapeOpts {
                probe: Some(ProbeSetup {
                    oracle_f0: Some(stream_f0.clone()),
                    ..ProbeSetup::default()
                }),
                ..LongShapeOpts::default()
            })?;
            let applied = a_iv.applied;
            let mut against = vec![("long_shape".to_string(), Rendition::from_ceiling("long_shape", a_iv.out)?)];
            let opts = RenderOpts {
                f0: Some(stream_f0),
                ..ctx.render.clone()
            };
            let voice = offline_voice(offline, ctx.plan, ctx.render)?;
            let w = render_whole(voice, s, &base.rec, whole_region, base.grid_phase, &opts)?;
            against.push(("whole".into(), Rendition::from_ceiling("whole", w.out)?));
            let (vs, recovered) = rows_vs(sc, &mut notes, &base.stream, &against, &table.baseline);
            table.rows.push(AblationRow::new(Ablation::StreamF0InCeiling, vs, recovered, applied));
        }
        None => notes.push("stream_f0_in_ceiling: the stream recorded no fed F0".into()),
    }
    // (v) both oracles in the stream (iii) and in ceiling A.
    if let Some(both) = &stream_both {
        let a_v = long_shape(ctx.plan, s, &base.rec, &ctx.q.ceiling, &LongShapeOpts {
            probe: Some(ProbeSetup {
                oracle_f0: Some(of.clone()),
                oracle_content: Some(oc.clone()),
                ..ProbeSetup::default()
            }),
            ..LongShapeOpts::default()
        })?;
        let applied = a_v.applied;
        let mut against = vec![("long_shape".to_string(), Rendition::from_ceiling("long_shape", a_v.out)?)];
        if let Some((_, w)) = ceilings.iter().find(|(n, _)| n == "whole") {
            against.push(("whole".into(), w.clone()));
        }
        let (vs, recovered) = rows_vs(sc, &mut notes, both, &against, &table.baseline);
        table.rows.push(AblationRow::new(Ablation::Residual, vs, recovered, applied));
    }
    table.rows.sort_by_key(|r| Ablation::ALL.iter().position(|a| *a == r.ablation));
    for r in &table.rows {
        for p in &r.problems {
            notes.push(format!("{}: NOT APPLIED, its recovered values mean nothing: {p}", r.ablation.as_str()));
        }
    }
    table.notes = notes;
    Ok(table)
}

fn counts(c: ProbeCounts) -> String {
    format!(
        "content {} calls / {} frames ({} misaligned), F0 {} calls / {} frames",
        c.content_calls, c.content_frames, c.content_misaligned, c.f0_calls, c.f0_frames
    )
}

/// The speech region of the streamed input (`input16`: the streamed input at 16 kHz, lead
/// included, whose sample 0 entered the model stream at absolute `input_start`), 0.3 s around its
/// speech, inside the recording.
fn whole_input_region(rec: &StreamRecorder, input16: &[f32], input_start: f64) -> Result<(u64, u64), String> {
    let mask = super::quality::speech_frames(input16, 50.0);
    let first = mask.iter().position(|&m| m).ok_or("the input has no speech")?;
    let last = mask.iter().rposition(|&m| m).unwrap_or(first);
    let s0 = input_start.max(0.0);
    let a = (s0 + first as f64 * 160.0 - 4_800.0).max(s0).round() as u64;
    let b = (s0 + ((last + 1) * 160) as f64 + 4_800.0).min(s0 + input16.len() as f64).round() as u64;
    let (start, data) = rec.stream();
    let (a, b) = (a.max(start), b.min(start + data.len() as u64));
    if b <= a + 3_200 {
        return Err("the input's speech region is too short".into());
    }
    Ok((a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oracles_keep_only_frames_inside_the_region() {
        let an = OfflineAnalysis {
            content: OracleContent {
                origin_16k: 3_200,
                emb: 2,
                feats: (0..20).map(|i| i as f32).collect(), // 10 frames
            },
            f0: OracleF0 {
                start_frame: 20,
                hz: (0..50).map(|i| 100.0 + i as f32).collect(),
            },
        };
        // Region [4_000, 6_000): F0 frames 25..37, content frames from 4_160 (k = 3) while
        // 3_200 + 320k + 400 <= 6_000 (k <= 7).
        let (f0, c) = oracles_in(&an, 4_000, 6_000);
        assert_eq!(f0.start_frame, 25);
        assert_eq!(f0.hz.len(), 37 - 25);
        assert_eq!(f0.hz[0], 105.0);
        assert_eq!(c.origin_16k, 3_200 + 320 * 3);
        assert_eq!(c.frames(), 5);
        assert_eq!(c.feats[0], 6.0);
        // Outside every frame: empty oracles.
        let (f0, c) = oracles_in(&an, 100_000, 110_000);
        assert!(f0.hz.is_empty() && c.feats.is_empty());
    }

    /// A row whose oracle replaced nothing (or missed the content grid) is flagged, whatever it
    /// recovered; healthy counters (the corpus runs') are not.
    #[test]
    fn an_oracle_that_replaced_nothing_is_flagged() {
        let healthy = ProbeCounts {
            content_calls: 95,
            content_frames: 1_895,
            f0_calls: 96,
            f0_frames: 3_993,
            content_misaligned: 0,
        };
        for abl in Ablation::ALL {
            assert!(unapplied(abl, &healthy).is_empty(), "{abl:?}");
        }
        // The F0-only oracles do not need content frames, and vice versa.
        let f0_only = ProbeCounts {
            f0_frames: 10,
            f0_calls: 2,
            ..ProbeCounts::default()
        };
        assert!(unapplied(Ablation::OracleF0, &f0_only).is_empty());
        assert!(unapplied(Ablation::StreamF0InCeiling, &f0_only).is_empty());
        assert_eq!(unapplied(Ablation::OracleContent, &f0_only).len(), 1);
        assert_eq!(unapplied(Ablation::OracleBoth, &f0_only).len(), 1);
        assert_eq!(unapplied(Ablation::Residual, &f0_only).len(), 1);
        // Nothing applied at all: every row complains (both oracles for the combined ones).
        let none = ProbeCounts::default();
        assert_eq!(unapplied(Ablation::OracleF0, &none).len(), 1);
        assert_eq!(unapplied(Ablation::OracleContent, &none).len(), 1);
        assert_eq!(unapplied(Ablation::OracleBoth, &none).len(), 2);
        // A stream block off the 320-sample grid: content frames may have applied elsewhere, the
        // row is still flagged.
        let misaligned = ProbeCounts {
            content_misaligned: 3,
            ..healthy
        };
        let p = unapplied(Ablation::OracleContent, &misaligned);
        assert_eq!(p.len(), 1);
        assert!(p[0].contains("3 call(s)") && p[0].contains("320"), "{p:?}");
        assert!(unapplied(Ablation::OracleF0, &misaligned).is_empty(), "F0 has no grid to miss");
        // The row carries it, and the table counts the rows.
        let row = AblationRow::new(Ablation::OracleContent, BTreeMap::new(), BTreeMap::new(), none);
        assert!(row.applied.contains("0 frames") && !row.problems.is_empty());
        let table = AblationTable {
            rows: vec![row, AblationRow::new(Ablation::OracleF0, BTreeMap::new(), BTreeMap::new(), healthy)],
            ..AblationTable::default()
        };
        assert_eq!(table.rows_unapplied(), 1);
    }

    #[test]
    fn recovered_counts_closer_as_positive() {
        let base = Scores {
            mcd_db: Some(2.0),
            env_mcd_db: Some(2.5),
            speaker_cosine_v2: Some(0.9),
            vuv_error: Some(0.05),
            f0_fine_rmse_cents: Some(40.0),
            seconds: 3.0,
            set_sim: Some(0.6),
        };
        let better = Scores {
            mcd_db: Some(1.5),
            env_mcd_db: Some(2.0),
            speaker_cosine_v2: Some(0.95),
            vuv_error: Some(0.02),
            f0_fine_rmse_cents: None,
            seconds: 3.0,
            set_sim: Some(0.65),
        };
        let r = better.recovered_from(&base);
        assert_eq!((r.mcd_db, r.env_mcd_db), (Some(0.5), Some(0.5)));
        assert!((r.speaker_cosine_v2.unwrap() - 0.05).abs() < 1e-12);
        assert!((r.vuv_error.unwrap() - 0.03).abs() < 1e-12);
        assert_eq!(r.f0_fine_rmse_cents, None);
        assert!((r.set_sim.unwrap() - 0.05).abs() < 1e-12);
        assert_eq!(Ablation::ALL.iter().map(|a| a.as_str()).collect::<Vec<_>>().len(), 5);
    }
}
